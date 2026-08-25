// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_verified_submit.rs
// ============================================
// Version: 1.0.0-VerifiedSubmitReplayDomain
//
// Creation Reason:
//   [VERIFIED-SUBMIT-REPLAY-DOMAIN 2026-08-25 by Codex] Extract the private
//   verified-submit replay domain from the oversized ChatRelay service while
//   preserving every existing wire, SQLite, retention, and error contract.
//
// Main Functionality:
//   - Models lookup and durable admission states as closed enums.
//   - Owns fixed-capacity same-process replay caching and lock striping.
//   - Protects durable replay responses behind a replaceable trait boundary.
//   - Derives private cache keys and envelope fingerprints without exposing
//     sender keys, request IDs, commitments, or response contents.
//
// Dependencies:
//   - `chat_relay.rs` owns SQLite transactions and composes this capability.
//   - `aeronyx-core` owns the authenticated verified-submit wire models.
//
// Main Logical Flow:
//   1. Derive domain-separated HMAC keys after request authentication.
//   2. Serialize equal private keys through one bounded async lock lane.
//   3. Replay exact process-local results or classify a conflict/miss.
//   4. Seal/open durable responses through the composed protector.
//
// Important Note for Next Developer:
//   - Do not add request-derived values to logs, status, or durable metadata.
//   - Preserve exact response replay and fixed-capacity eviction semantics.
//   - SQLite reservation/response transitions must remain one IMMEDIATE
//     transaction in `chat_relay.rs`; this module does not own persistence.
//   - A replacement protector must authenticate both private fingerprints.
//
// Last Modified:
//   v1.0.0-VerifiedSubmitReplayDomain - Initial trait/composition extraction
// ============================================

use std::collections::{HashMap, VecDeque};

use aeronyx_core::protocol::memchain::{
    ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1,
};
use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;

use super::chat_relay::{ChatRelayError, ChatRelayResult};

type HmacSha256 = Hmac<Sha256>;

const SINGLE_FLIGHT_LANES: usize = 64;
pub(crate) const RESPONSE_NONCE_BYTES: usize = 24;
const RESPONSE_TAG_BYTES: usize = 16;
const MAX_RESPONSE_BYTES: usize = 512;
const RESPONSE_HKDF_SALT: &[u8] = b"AeroNyx-VerifiedSubmit-ResponseCache-v1-key";
const RESPONSE_HKDF_INFO: &[u8] = b"XChaCha20-Poly1305";
const RESPONSE_AAD_DOMAIN: &[u8] = b"AeroNyx-VerifiedSubmit-ResponseCache-v1";
const CACHE_KEY_DOMAIN: &[u8] = b"AeroNyx-VerifiedSubmit-ResponseCache-v1";
const ENVELOPE_FINGERPRINT_DOMAIN: &[u8] = b"AeroNyx-VerifiedSubmit-EnvelopeFingerprint-v1";

/// Result of looking up one authenticated verified-submit request.
pub(crate) enum VerifiedSubmitCacheLookup {
    /// No prior completed request exists under this private cache key.
    Miss,
    /// The exact request completed previously; return this response verbatim.
    Exact(ChatRelayVerifiedSubmitResponseV1),
    /// The same sender/request id was reused with a different envelope.
    Conflict,
    /// The exact request owns an unfinished durable reservation.
    Pending,
}

/// Outcome of atomically reserving capacity before verified-submit side effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VerifiedSubmitAdmission {
    /// This process owns the durable reservation and may perform side effects.
    Reserved,
    /// A replacement process owns an exact abandoned request. Recovery may
    /// repeat only idempotent entry custody and must not select an onion path.
    ReservedForEntryRecovery,
    /// Another process or a prior crashed process owns the exact reservation.
    Pending,
    /// The same sender/request id is reserved for another envelope.
    Conflict,
    /// A completed response appeared between lookup and reservation.
    Completed,
    /// Every configured replay slot is occupied by unexpired safety evidence.
    CapacityExhausted,
}

/// AEAD result retained by the SQLite repository as opaque bytes.
pub(crate) struct ProtectedVerifiedSubmitResponse {
    pub(crate) nonce: [u8; RESPONSE_NONCE_BYTES],
    pub(crate) ciphertext: Vec<u8>,
}

/// Replaceable response-protection capability.
///
/// [VERIFIED-SUBMIT-REPLAY-DOMAIN 2026-08-25 by Codex] This trait isolates the
/// cryptographic mechanism from cache/admission policy. Implementations must
/// bind ciphertext to both private fingerprints and return only stable relay
/// error classes; callers must never log cryptographic inputs or outputs.
pub(crate) trait VerifiedSubmitResponseProtector: Send + Sync {
    fn protect(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        response: &ChatRelayVerifiedSubmitResponseV1,
    ) -> ChatRelayResult<ProtectedVerifiedSubmitResponse>;

    fn recover(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        nonce: &[u8],
        ciphertext: &[u8],
    ) -> ChatRelayResult<ChatRelayVerifiedSubmitResponseV1>;
}

/// Production XChaCha20-Poly1305 response protector.
pub(crate) struct XChaChaVerifiedSubmitResponseProtector {
    key: [u8; 32],
}

impl XChaChaVerifiedSubmitResponseProtector {
    fn new(node_secret: &[u8; 32]) -> ChatRelayResult<Self> {
        let hkdf = hkdf::Hkdf::<Sha256>::new(Some(RESPONSE_HKDF_SALT), node_secret);
        let mut key = [0_u8; 32];
        hkdf.expand(RESPONSE_HKDF_INFO, &mut key)
            .map_err(|_| ChatRelayError::VerifiedSubmitProtectionFailed)?;
        Ok(Self { key })
    }

    fn aad(cache_key: &[u8; 32], envelope_fingerprint: &[u8; 32]) -> Vec<u8> {
        let mut aad = Vec::with_capacity(
            RESPONSE_AAD_DOMAIN.len() + cache_key.len() + envelope_fingerprint.len(),
        );
        aad.extend_from_slice(RESPONSE_AAD_DOMAIN);
        aad.extend_from_slice(cache_key);
        aad.extend_from_slice(envelope_fingerprint);
        aad
    }
}

impl VerifiedSubmitResponseProtector for XChaChaVerifiedSubmitResponseProtector {
    fn protect(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        response: &ChatRelayVerifiedSubmitResponseV1,
    ) -> ChatRelayResult<ProtectedVerifiedSubmitResponse> {
        response
            .validate_shape()
            .map_err(|_| ChatRelayError::VerifiedSubmitProtectionFailed)?;
        let plaintext = bincode::serialize(response)?;
        if plaintext.len() > MAX_RESPONSE_BYTES {
            return Err(ChatRelayError::VerifiedSubmitProtectionFailed);
        }

        let mut nonce = [0_u8; RESPONSE_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let aad = Self::aad(cache_key, envelope_fingerprint);
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: &plaintext,
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::VerifiedSubmitProtectionFailed)?;
        if ciphertext.len() > MAX_RESPONSE_BYTES + RESPONSE_TAG_BYTES {
            return Err(ChatRelayError::VerifiedSubmitProtectionFailed);
        }
        Ok(ProtectedVerifiedSubmitResponse { nonce, ciphertext })
    }

    fn recover(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        nonce: &[u8],
        ciphertext: &[u8],
    ) -> ChatRelayResult<ChatRelayVerifiedSubmitResponseV1> {
        if nonce.len() != RESPONSE_NONCE_BYTES
            || ciphertext.len() <= RESPONSE_TAG_BYTES
            || ciphertext.len() > MAX_RESPONSE_BYTES + RESPONSE_TAG_BYTES
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_ciphertext_shape",
            });
        }
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let aad = Self::aad(cache_key, envelope_fingerprint);
        let plaintext = cipher
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_authentication",
            })?;
        if plaintext.len() > MAX_RESPONSE_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_plaintext_size",
            });
        }
        let response: ChatRelayVerifiedSubmitResponseV1 = bincode::deserialize(&plaintext)?;
        response
            .validate_shape()
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "verified_submit_response_shape",
            })?;
        Ok(response)
    }
}

#[derive(Clone)]
struct VerifiedSubmitCacheEntry {
    envelope_fingerprint: [u8; 32],
    response: ChatRelayVerifiedSubmitResponseV1,
}

/// Fixed-capacity process-local cache for completed verified submissions.
///
/// [CHAT-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-23 by Codex] Keys are
/// domain-separated node-secret HMACs over sender plus request id. Raw routing
/// metadata never becomes a cache key, log field, heartbeat field, or durable
/// row. A fixed number of async lock lanes serializes the same private key
/// while allowing unrelated submissions to proceed concurrently.
struct VerifiedSubmitResponseCache {
    entries: HashMap<[u8; 32], VerifiedSubmitCacheEntry>,
    insertion_order: VecDeque<[u8; 32]>,
    capacity: usize,
}

impl VerifiedSubmitResponseCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            insertion_order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn lookup(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
    ) -> VerifiedSubmitCacheLookup {
        let Some(entry) = self.entries.get(cache_key) else {
            return VerifiedSubmitCacheLookup::Miss;
        };
        if &entry.envelope_fingerprint != envelope_fingerprint {
            return VerifiedSubmitCacheLookup::Conflict;
        }
        VerifiedSubmitCacheLookup::Exact(entry.response.clone())
    }

    fn insert(
        &mut self,
        cache_key: [u8; 32],
        envelope_fingerprint: [u8; 32],
        response: ChatRelayVerifiedSubmitResponseV1,
    ) {
        if self.entries.contains_key(&cache_key) {
            return;
        }
        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self.insertion_order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.insertion_order.push_back(cache_key);
        self.entries.insert(
            cache_key,
            VerifiedSubmitCacheEntry {
                envelope_fingerprint,
                response,
            },
        );
    }
}

/// Composed verified-submit replay capability used by `ChatRelayService`.
pub(crate) struct VerifiedSubmitReplay<P = XChaChaVerifiedSubmitResponseProtector> {
    node_secret: [u8; 32],
    protector: P,
    cache: Mutex<VerifiedSubmitResponseCache>,
    lanes: Box<[tokio::sync::Mutex<()>]>,
}

impl VerifiedSubmitReplay<XChaChaVerifiedSubmitResponseProtector> {
    pub(crate) fn new(node_secret: [u8; 32], capacity: usize) -> ChatRelayResult<Self> {
        let protector = XChaChaVerifiedSubmitResponseProtector::new(&node_secret)?;
        Ok(Self::with_protector(node_secret, capacity, protector))
    }
}

impl<P: VerifiedSubmitResponseProtector> VerifiedSubmitReplay<P> {
    fn with_protector(node_secret: [u8; 32], capacity: usize, protector: P) -> Self {
        Self {
            node_secret,
            protector,
            cache: Mutex::new(VerifiedSubmitResponseCache::new(capacity)),
            lanes: (0..SINGLE_FLIGHT_LANES)
                .map(|_| tokio::sync::Mutex::new(()))
                .collect(),
        }
    }

    pub(crate) fn cache_key(&self, request: &ChatRelayVerifiedSubmitRequestV1) -> [u8; 32] {
        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(CACHE_KEY_DOMAIN);
        mac.update(&request.envelope.sender);
        mac.update(&request.request_id);
        mac.finalize().into_bytes().into()
    }

    pub(crate) fn envelope_fingerprint(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> [u8; 32] {
        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(ENVELOPE_FINGERPRINT_DOMAIN);
        mac.update(&request.envelope_commitment());
        mac.finalize().into_bytes().into()
    }

    pub(crate) async fn lock(
        &self,
        request: &ChatRelayVerifiedSubmitRequestV1,
    ) -> tokio::sync::MutexGuard<'_, ()> {
        let cache_key = self.cache_key(request);
        let lane_seed = u64::from_le_bytes(
            cache_key[..8]
                .try_into()
                .expect("verified submit cache key has eight prefix bytes"),
        );
        let lane = usize::try_from(lane_seed).unwrap_or(usize::MAX) % self.lanes.len();
        self.lanes[lane].lock().await
    }

    pub(crate) fn lookup_cached(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
    ) -> VerifiedSubmitCacheLookup {
        self.cache.lock().lookup(cache_key, envelope_fingerprint)
    }

    pub(crate) fn remember_cached(
        &self,
        cache_key: [u8; 32],
        envelope_fingerprint: [u8; 32],
        response: ChatRelayVerifiedSubmitResponseV1,
    ) {
        self.cache
            .lock()
            .insert(cache_key, envelope_fingerprint, response);
    }

    pub(crate) fn protect_response(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        response: &ChatRelayVerifiedSubmitResponseV1,
    ) -> ChatRelayResult<ProtectedVerifiedSubmitResponse> {
        self.protector
            .protect(cache_key, envelope_fingerprint, response)
    }

    pub(crate) fn recover_response(
        &self,
        cache_key: &[u8; 32],
        envelope_fingerprint: &[u8; 32],
        nonce: &[u8],
        ciphertext: &[u8],
    ) -> ChatRelayResult<ChatRelayVerifiedSubmitResponseV1> {
        self.protector
            .recover(cache_key, envelope_fingerprint, nonce, ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_cache_is_bounded_and_conflict_safe() {
        let mut cache = VerifiedSubmitResponseCache::new(1);
        let first_key = [0xA1; 32];
        let first_fingerprint = [0xA2; 32];
        let first_response = ChatRelayVerifiedSubmitResponseV1::rejected([0xA3; 16], [0xA4; 16]);
        cache.insert(first_key, first_fingerprint, first_response.clone());
        assert!(matches!(
            cache.lookup(&first_key, &[0xA5; 32]),
            VerifiedSubmitCacheLookup::Conflict
        ));
        let VerifiedSubmitCacheLookup::Exact(cached) = cache.lookup(&first_key, &first_fingerprint)
        else {
            panic!("exact verified submit response must remain replayable");
        };
        assert_eq!(cached, first_response);

        let second_key = [0xB1; 32];
        let second_fingerprint = [0xB2; 32];
        cache.insert(
            second_key,
            second_fingerprint,
            ChatRelayVerifiedSubmitResponseV1::rejected([0xB3; 16], [0xB4; 16]),
        );
        assert!(matches!(
            cache.lookup(&first_key, &first_fingerprint),
            VerifiedSubmitCacheLookup::Miss
        ));
        assert!(matches!(
            cache.lookup(&second_key, &second_fingerprint),
            VerifiedSubmitCacheLookup::Exact(_)
        ));
    }
}
