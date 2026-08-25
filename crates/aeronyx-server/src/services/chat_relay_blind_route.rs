// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_blind_route.rs
// ============================================
// Version: 1.0.0-BlindRouteReplayDomain
//
// Creation Reason:
//   [BLIND-ROUTE-REPLAY-DOMAIN 2026-08-25 by Codex] Extract blind-route replay
//   identity and response protection from the oversized ChatRelay service
//   without changing its API, SQLite schema, retention, or error contracts.
//
// Main Functionality:
//   - Models durable blind-route admission as a closed enum.
//   - Derives node-private route keys and request fingerprints.
//   - Protects exact opaque relay responses behind a replaceable trait.
//   - Binds every sealed response to both private replay identifiers.
//
// Dependencies:
//   - `chat_relay.rs` owns SQLite transactions and composes this capability.
//   - `ChatRelayError` provides the existing stable storage error boundary.
//
// Main Logical Flow:
//   1. Derive domain-separated HMAC identifiers after request authentication.
//   2. Let `chat_relay.rs` reserve durable capacity transactionally.
//   3. Seal the exact opaque ACK before atomic reservation completion.
//   4. Authenticate and recover the ACK for an exact restart replay.
//
// Important Note for Next Developer:
//   - Never persist or report raw route ids or request commitments.
//   - A replacement protector must bind both private replay identifiers.
//   - SQLite reservation/response transitions stay in `chat_relay.rs`; moving
//     them requires preserving one IMMEDIATE transaction on the same connection.
//   - Response plaintext is opaque here and must never be parsed or logged.
//
// Last Modified:
//   v1.0.0-BlindRouteReplayDomain - Initial trait/composition extraction
// ============================================

use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hmac::{Hmac, Mac};
use rand::{rngs::OsRng, RngCore};
use sha2::Sha256;

use super::chat_relay::{ChatRelayError, ChatRelayResult};

type HmacSha256 = Hmac<Sha256>;

pub(crate) const RESPONSE_NONCE_BYTES: usize = 24;
const RESPONSE_TAG_BYTES: usize = 16;
const MAX_RESPONSE_BYTES: usize = 2048;
const RESPONSE_HKDF_SALT: &[u8] = b"AeroNyx-BlindRelay-RouteResponse-v1-key";
const RESPONSE_HKDF_INFO: &[u8] = b"XChaCha20-Poly1305";
const RESPONSE_AAD_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-RouteResponse-v1";
const ROUTE_CACHE_KEY_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-RouteCache-v1";
const REQUEST_FINGERPRINT_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-RequestFingerprint-v1";

/// Durable admission result for one authenticated blind-relay request.
///
/// [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] The API owns wire types;
/// this storage boundary returns only sealed opaque response bytes and
/// timestamps. It never depends on HTTP models or parses relay plaintext.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BlindRelayRouteAdmission {
    Reserved,
    /// A previous process armed this exact claim but did not persist its ACK.
    /// The new owner must repeat only idempotent work using the exact request.
    ReservedForRecovery,
    Pending,
    Conflict,
    Completed {
        response: Vec<u8>,
        completed_at: u64,
    },
    CapacityExhausted,
}

/// AEAD output persisted by the SQLite repository as opaque bytes.
pub(crate) struct ProtectedBlindRouteResponse {
    pub(crate) nonce: [u8; RESPONSE_NONCE_BYTES],
    pub(crate) ciphertext: Vec<u8>,
}

/// Replaceable exact-response protection capability.
///
/// [BLIND-ROUTE-REPLAY-DOMAIN 2026-08-25 by Codex] The trait separates the
/// cryptographic mechanism from durable claim ownership. Implementations must
/// authenticate both private replay identifiers and expose no secret material.
pub(crate) trait BlindRouteResponseProtector: Send + Sync {
    fn protect(
        &self,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        response: &[u8],
    ) -> ChatRelayResult<ProtectedBlindRouteResponse>;

    fn recover(
        &self,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        nonce: &[u8],
        ciphertext: &[u8],
    ) -> ChatRelayResult<Vec<u8>>;
}

/// Production XChaCha20-Poly1305 protector for opaque blind-route ACKs.
pub(crate) struct XChaChaBlindRouteResponseProtector {
    key: [u8; 32],
}

impl XChaChaBlindRouteResponseProtector {
    fn new(node_secret: &[u8; 32]) -> ChatRelayResult<Self> {
        let hkdf = hkdf::Hkdf::<Sha256>::new(Some(RESPONSE_HKDF_SALT), node_secret);
        let mut key = [0_u8; 32];
        hkdf.expand(RESPONSE_HKDF_INFO, &mut key)
            .map_err(|_| ChatRelayError::BlindRelayReplayProtectionFailed)?;
        Ok(Self { key })
    }

    fn aad(cache_key: &[u8; 32], request_fingerprint: &[u8; 32]) -> Vec<u8> {
        let mut aad = Vec::with_capacity(
            RESPONSE_AAD_DOMAIN.len() + cache_key.len() + request_fingerprint.len(),
        );
        aad.extend_from_slice(RESPONSE_AAD_DOMAIN);
        aad.extend_from_slice(cache_key);
        aad.extend_from_slice(request_fingerprint);
        aad
    }
}

impl BlindRouteResponseProtector for XChaChaBlindRouteResponseProtector {
    fn protect(
        &self,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        response: &[u8],
    ) -> ChatRelayResult<ProtectedBlindRouteResponse> {
        if response.is_empty() || response.len() > MAX_RESPONSE_BYTES {
            return Err(ChatRelayError::BlindRelayReplayProtectionFailed);
        }
        let mut nonce = [0_u8; RESPONSE_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let aad = Self::aad(cache_key, request_fingerprint);
        let ciphertext = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: response,
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::BlindRelayReplayProtectionFailed)?;
        if ciphertext.len() != response.len() + RESPONSE_TAG_BYTES {
            return Err(ChatRelayError::BlindRelayReplayProtectionFailed);
        }
        Ok(ProtectedBlindRouteResponse { nonce, ciphertext })
    }

    fn recover(
        &self,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        nonce: &[u8],
        ciphertext: &[u8],
    ) -> ChatRelayResult<Vec<u8>> {
        if nonce.len() != RESPONSE_NONCE_BYTES
            || ciphertext.len() <= RESPONSE_TAG_BYTES
            || ciphertext.len() > MAX_RESPONSE_BYTES + RESPONSE_TAG_BYTES
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_shape",
            });
        }
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.key));
        let aad = Self::aad(cache_key, request_fingerprint);
        let response = cipher
            .decrypt(
                XNonce::from_slice(nonce),
                Payload {
                    msg: ciphertext,
                    aad: &aad,
                },
            )
            .map_err(|_| ChatRelayError::BlindRelayReplayProtectionFailed)?;
        if response.is_empty() || response.len() > MAX_RESPONSE_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "blind_relay_route_response_plaintext_shape",
            });
        }
        Ok(response)
    }
}

/// Composed private replay identity and response-protection domain.
pub(crate) struct BlindRouteReplay<P = XChaChaBlindRouteResponseProtector> {
    node_secret: [u8; 32],
    protector: P,
}

impl BlindRouteReplay<XChaChaBlindRouteResponseProtector> {
    pub(crate) fn new(node_secret: [u8; 32]) -> ChatRelayResult<Self> {
        let protector = XChaChaBlindRouteResponseProtector::new(&node_secret)?;
        Ok(Self::with_protector(node_secret, protector))
    }
}

impl<P: BlindRouteResponseProtector> BlindRouteReplay<P> {
    fn with_protector(node_secret: [u8; 32], protector: P) -> Self {
        Self {
            node_secret,
            protector,
        }
    }

    pub(crate) fn cache_key(&self, route_id: &[u8; 16]) -> [u8; 32] {
        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(ROUTE_CACHE_KEY_DOMAIN);
        mac.update(route_id);
        mac.finalize().into_bytes().into()
    }

    pub(crate) fn request_fingerprint(&self, request_commitment: &[u8; 32]) -> [u8; 32] {
        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(REQUEST_FINGERPRINT_DOMAIN);
        mac.update(request_commitment);
        mac.finalize().into_bytes().into()
    }

    pub(crate) fn protect_response(
        &self,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        response: &[u8],
    ) -> ChatRelayResult<ProtectedBlindRouteResponse> {
        self.protector
            .protect(cache_key, request_fingerprint, response)
    }

    pub(crate) fn recover_response(
        &self,
        cache_key: &[u8; 32],
        request_fingerprint: &[u8; 32],
        nonce: &[u8],
        ciphertext: &[u8],
    ) -> ChatRelayResult<Vec<u8>> {
        self.protector
            .recover(cache_key, request_fingerprint, nonce, ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_identity_and_response_protection_are_domain_bound() {
        let replay = BlindRouteReplay::new([0x41; 32]).expect("construct blind route replay");
        let cache_key = replay.cache_key(&[0x42; 16]);
        let request_fingerprint = replay.request_fingerprint(&[0x43; 32]);
        assert_ne!(cache_key, replay.cache_key(&[0x44; 16]));
        assert_ne!(request_fingerprint, replay.request_fingerprint(&[0x45; 32]));

        let protected = replay
            .protect_response(&cache_key, &request_fingerprint, b"opaque-relay-ack")
            .expect("protect exact blind route response");
        let recovered = replay
            .recover_response(
                &cache_key,
                &request_fingerprint,
                &protected.nonce,
                &protected.ciphertext,
            )
            .expect("recover exact blind route response");
        assert_eq!(recovered, b"opaque-relay-ack");

        assert!(matches!(
            replay.recover_response(
                &cache_key,
                &[0x46; 32],
                &protected.nonce,
                &protected.ciphertext,
            ),
            Err(ChatRelayError::BlindRelayReplayProtectionFailed)
        ));
    }
}
