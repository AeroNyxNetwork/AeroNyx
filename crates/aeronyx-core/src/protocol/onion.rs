// ============================================
// File: crates/aeronyx-core/src/protocol/onion.rs
// ============================================
//! # Onion Routing v1 — Layered Per-Hop Encryption
//!
//! ## Creation Reason
//! Upgrades the existing **blind relay** (a single opaque envelope forwarder)
//! into real **onion routing**: the source wraps the payload in one encrypted
//! layer per hop, and each relay peels exactly one layer. A relay learns only
//! the *immediate* next hop — never the original source, the final destination,
//! or the payload. This guarantees that no single honest-but-curious relay can
//! link source and destination together.
//!
//! ## Relationship to the transport
//! This module does NOT define a new wire frame. It restructures the opaque
//! `BlindRelayEnvelope::encrypted_blob` (see `chat.rs`). The envelope and all of
//! its hardened guards (Ed25519 per-hop signature, freshness window, replay
//! cache, abuse guard, routeability gate, TTL, loop detection, probes, counters)
//! are reused unchanged. `envelope.next_hop` always addresses the node that
//! receives *this* envelope; the privacy-sensitive forward target is hidden
//! inside the peeled layer.
//!
//! ## Construction (HPKE-style, RFC 9180 DHKEM shape)
//! Each layer is a single-shot seal to the hop's KEM public key:
//! ```text
//!   ephemeral X25519  ->  ECDH  ->  HKDF-SHA256  ->  XChaCha20-Poly1305
//! ```
//! All primitives are the already-audited ones in `crypto/{keys,kdf}.rs`; no new
//! cryptographic dependency is introduced. The KEM is deliberately abstracted
//! behind a versioned descriptor field (`kem_alg`) so a future release can move
//! to the hybrid post-quantum X-Wing KEM (X25519 + ML-KEM-768) without changing
//! this wire format.
//!
//! ## Layer wire format (content of `encrypted_blob`)
//! ```text
//!   magic:   [0xA0, 0x01]   (2B  — ONION_V1 marker)
//!   eph_pub: [u8; 32]       (client ephemeral X25519 public for THIS hop)
//!   nonce:   [u8; 24]       (random XChaCha20 nonce)
//!   ct:      Vec<u8>        (XChaCha20-Poly1305 over the encoded OnionHopPayload)
//! ```
//! Key derivation (both sides identical):
//! `key = HKDF-SHA256(ikm = ECDH(eph, hop_kem), salt = ONION_SALT,
//!                    info = eph_pub || hop_kem_pub, len = 32)`.
//!
//! ## Threat model (v1)
//! Honest-but-curious relays. v1 does NOT defend against a *global passive
//! observer* that correlates packet lengths/timing (the onion shrinks one layer
//! per hop). That property requires a constant-length Sphinx packet with
//! per-hop replay MACs and ephemeral blinding, which is the documented v2
//! upgrade. See `docs/onion-routing-v1-spec.md`.
//!
//! ## ⚠️ Important Notes for Next Developer
//! - The `OnionHopPayload` bincode layout is a wire contract. Do NOT reorder
//!   fields. Add new fields only with a versioned magic (e.g. `[0xA0, 0x02]`).
//! - `open_onion_layer` must never log plaintext or the peeled `inner` bytes.
//! - A relay's X25519 *public* key is NOT derivable from its Ed25519 `node_id`
//!   (the X25519 secret is `SHA512(ed_secret)[..32]`), so it MUST be published
//!   in the node descriptor. See `discovery::NodeDescriptor::kem_public`.
//!
//! ## Last Modified
//! v1.0.0-OnionV1 — Initial layered onion construction over the blind relay frame

use bincode::Options;
use hkdf::Hkdf;
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

use crate::crypto::keys::{E2eSession, EphemeralKeyPair, IdentityKeyPair};
use crate::error::CoreError;
use crate::protocol::chat::BlindRelayEnvelope;

// ============================================
// Constants
// ============================================

/// Onion layer magic prefix (version 1). Marks `encrypted_blob` as an onion
/// layer to be peeled, distinguishing it from a legacy opaque blind-relay blob.
pub const ONION_MAGIC: [u8; 2] = [0xA0, 0x01];

/// HKDF domain-separation salt for onion layer keys.
pub const ONION_SALT: &[u8] = b"AeroNyx-Onion-v1";

/// KEM algorithm id: classical X25519 (the v1 default).
pub const KEM_ALG_X25519: u8 = 1;

/// KEM algorithm id reserved for the hybrid post-quantum X-Wing KEM
/// (X25519 + ML-KEM-768). Not implemented in v1; reserved so the descriptor
/// field and this module can adopt it without a wire break.
pub const KEM_ALG_XWING: u8 = 2;

/// Fixed layer header length: magic(2) + eph_pub(32) + nonce(24).
const LAYER_HEADER_LEN: usize = 2 + 32 + 24;

/// Upper bound for a decoded `OnionHopPayload` (matches the blind relay frame cap).
const MAX_ONION_PAYLOAD_BYTES: u64 = 256 * 1024;

// ============================================
// Types
// ============================================

/// One hop on an onion path: the relay's node id plus its published KEM key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionHop {
    /// Relay Ed25519 node id (matches `NodeDescriptor::node_id`).
    pub node_id: [u8; 32],
    /// Relay KEM public key (X25519 for v1; from `NodeDescriptor::kem_public`).
    pub kem_pub: [u8; 32],
}

/// Result of peeling exactly one onion layer at a relay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionPeel {
    /// `Some(node_id)` → forward `inner` (the next layer) onward to that hop.
    /// `None` → this node is the terminal hop; `inner` is the delivered payload.
    pub next_hop: Option<[u8; 32]>,
    /// Next layer bytes (when forwarding) or the final payload (when terminal).
    pub inner: Vec<u8>,
}

/// Plaintext carried inside one onion layer. Bincode positional layout is a
/// wire contract — do not reorder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct OnionHopPayload {
    next_hop: Option<[u8; 32]>,
    inner: Vec<u8>,
}

// ============================================
// Public helpers
// ============================================

/// Returns true if `blob` begins with the onion v1 magic prefix.
///
/// Used by relays to decide whether to peel an onion layer or fall back to the
/// legacy opaque blind-relay forwarding path.
#[must_use]
pub fn is_onion_blob(blob: &[u8]) -> bool {
    blob.len() >= 2 && blob[0] == ONION_MAGIC[0] && blob[1] == ONION_MAGIC[1]
}

/// Peels one onion layer, trying each candidate secret in order (typically the
/// node's current onion key, then the previous one during a rotation grace
/// window). Returns the first successful peel.
///
/// This supports **forward secrecy via rotating onion keys**: a relay rotates
/// its onion keypair on a schedule and keeps the previous key only for a short
/// grace window, so an onion built against the just-rotated descriptor still
/// peels. See `aeronyx-server::services::onion_keys`.
///
/// # Errors
/// Returns `CoreError` if none of the candidate secrets peel the layer.
pub fn try_open_onion_layer(
    blob: &[u8],
    node_x25519_secrets: &[StaticSecret],
) -> Result<OnionPeel, CoreError> {
    let mut last_err = CoreError::malformed("onion layer: no candidate keys");
    for secret in node_x25519_secrets {
        match open_onion_layer(blob, secret) {
            Ok(peel) => return Ok(peel),
            Err(err) => last_err = err,
        }
    }
    Err(last_err)
}

/// Peels exactly one onion layer using this node's static X25519 secret.
///
/// # Errors
/// Returns `CoreError` if the blob is not a well-formed onion layer, if AEAD
/// authentication fails (wrong key / tampered bytes), or if the inner payload
/// fails to decode.
pub fn open_onion_layer(
    blob: &[u8],
    node_x25519_sk: &StaticSecret,
) -> Result<OnionPeel, CoreError> {
    if !is_onion_blob(blob) {
        return Err(CoreError::malformed("onion layer: missing magic prefix"));
    }
    if blob.len() < LAYER_HEADER_LEN {
        return Err(CoreError::malformed("onion layer: truncated header"));
    }

    let mut eph_pub = [0u8; 32];
    eph_pub.copy_from_slice(&blob[2..34]);
    let mut nonce = [0u8; 24];
    nonce.copy_from_slice(&blob[34..LAYER_HEADER_LEN]);
    let ciphertext = &blob[LAYER_HEADER_LEN..];

    // This hop's own KEM public key, recomputed from its secret, binds the key
    // derivation to this specific relay (same value the sender used).
    let hop_kem_pub = X25519PublicKey::from(node_x25519_sk).to_bytes();

    let shared = node_x25519_sk.diffie_hellman(&X25519PublicKey::from(eph_pub));
    let mut ecdh = *shared.as_bytes();
    let key = derive_layer_key(&ecdh, &eph_pub, &hop_kem_pub)?;
    ecdh.zeroize();

    // E2eSession uses the 32-byte key directly with XChaCha20-Poly1305 and
    // zeroizes it on drop. peer_public_key is for logging only.
    let session = E2eSession::new(key, eph_pub);
    let plaintext = session
        .decrypt_raw(ciphertext, &nonce)
        .map_err(|_| CoreError::malformed("onion layer: AEAD open failed"))?;

    let payload = decode_payload(&plaintext)?;
    Ok(OnionPeel {
        next_hop: payload.next_hop,
        inner: payload.inner,
    })
}

/// Builds a complete onion-wrapped `BlindRelayEnvelope` for `path`.
///
/// Layers are sealed innermost (exit) → outermost (entry). The returned
/// envelope is addressed to `path[0]` and signed by `source` (which becomes the
/// `previous_hop_node_id` on the wire, exactly as a normal blind relay send).
///
/// `now` is the Unix-seconds timestamp to stamp on the outer envelope (callers
/// pass a clock value; this crate stays clock-free for deterministic tests).
///
/// # Errors
/// Returns `CoreError` if `path` is empty or any layer fails to seal.
pub fn build_onion_envelope(
    path: &[OnionHop],
    final_payload: &[u8],
    route_id: [u8; 16],
    ttl: u8,
    now: u64,
    source: &IdentityKeyPair,
) -> Result<BlindRelayEnvelope, CoreError> {
    if path.is_empty() {
        return Err(CoreError::malformed("onion path: empty"));
    }

    // Start with the raw payload; wrap one layer per hop from the exit inward.
    let mut inner = final_payload.to_vec();
    for i in (0..path.len()).rev() {
        let next_hop = if i + 1 < path.len() {
            Some(path[i + 1].node_id)
        } else {
            None
        };
        let payload = OnionHopPayload { next_hop, inner };
        let encoded = encode_payload(&payload)?;
        inner = seal_layer(&path[i].kem_pub, &encoded)?;
    }

    let envelope = BlindRelayEnvelope {
        route_id,
        next_hop: path[0].node_id,
        ttl,
        encrypted_blob: inner,
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(source);

    Ok(envelope)
}

// ============================================
// Internal
// ============================================

/// Seals one onion layer to `hop_kem_pub`.
fn seal_layer(hop_kem_pub: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, CoreError> {
    let ephemeral = EphemeralKeyPair::generate();
    let eph_pub = ephemeral.public_key_bytes();
    let mut ecdh = ephemeral.exchange(hop_kem_pub);
    let key = derive_layer_key(&ecdh, &eph_pub, hop_kem_pub)?;
    ecdh.zeroize();

    let session = E2eSession::new(key, *hop_kem_pub);
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut nonce);
    let ciphertext = session
        .encrypt_raw(plaintext, &nonce)
        .map_err(|_| CoreError::key_generation("onion layer: AEAD seal failed"))?;

    let mut out = Vec::with_capacity(LAYER_HEADER_LEN + ciphertext.len());
    out.extend_from_slice(&ONION_MAGIC);
    out.extend_from_slice(&eph_pub);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// HKDF-SHA256 layer key derivation. `info = eph_pub || hop_kem_pub` binds the
/// key to both the ephemeral and the specific relay.
fn derive_layer_key(
    ecdh: &[u8; 32],
    eph_pub: &[u8; 32],
    hop_kem_pub: &[u8; 32],
) -> Result<[u8; 32], CoreError> {
    let mut info = [0u8; 64];
    info[..32].copy_from_slice(eph_pub);
    info[32..].copy_from_slice(hop_kem_pub);

    let hk = Hkdf::<Sha256>::new(Some(ONION_SALT), ecdh);
    let mut key = [0u8; 32];
    hk.expand(&info, &mut key)
        .map_err(|_| CoreError::key_generation("onion layer: HKDF expand failed"))?;
    Ok(key)
}

fn encode_payload(payload: &OnionHopPayload) -> Result<Vec<u8>, CoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_ONION_PAYLOAD_BYTES)
        .serialize(payload)
        .map_err(|err| CoreError::malformed(format!("onion payload encode: {err}")))
}

fn decode_payload(bytes: &[u8]) -> Result<OnionHopPayload, CoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_ONION_PAYLOAD_BYTES)
        .deserialize(bytes)
        .map_err(|err| CoreError::malformed(format!("onion payload decode: {err}")))
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn hop_keypair() -> (IdentityKeyPair, OnionHop) {
        let identity = IdentityKeyPair::generate();
        let node_id = identity.public_key_bytes();
        let kem_pub = identity.x25519_public_key_bytes();
        (identity, OnionHop { node_id, kem_pub })
    }

    fn x25519_secret(identity: &IdentityKeyPair) -> StaticSecret {
        identity.to_x25519().0
    }

    #[test]
    fn two_hop_round_trip_delivers_payload() {
        let source = IdentityKeyPair::generate();
        let (entry_id, entry_hop) = hop_keypair();
        let (exit_id, exit_hop) = hop_keypair();
        let payload = b"the inner ChatEnvelope bytes".to_vec();

        let envelope = build_onion_envelope(
            &[entry_hop.clone(), exit_hop.clone()],
            &payload,
            [7u8; 16],
            4,
            1_700_000_000,
            &source,
        )
        .unwrap();

        // Outer envelope is addressed to the entry hop and is a valid onion blob.
        assert_eq!(envelope.next_hop, entry_hop.node_id);
        assert!(is_onion_blob(&envelope.encrypted_blob));

        // Entry peels → forward target is the exit hop, inner is the next layer.
        let entry_peel =
            open_onion_layer(&envelope.encrypted_blob, &x25519_secret(&entry_id)).unwrap();
        assert_eq!(entry_peel.next_hop, Some(exit_hop.node_id));
        assert!(is_onion_blob(&entry_peel.inner));

        // Exit peels → terminal, inner is the original payload.
        let exit_peel = open_onion_layer(&entry_peel.inner, &x25519_secret(&exit_id)).unwrap();
        assert_eq!(exit_peel.next_hop, None);
        assert_eq!(exit_peel.inner, payload);
    }

    #[test]
    fn three_hop_round_trip_delivers_payload() {
        let source = IdentityKeyPair::generate();
        let (a_id, a) = hop_keypair();
        let (b_id, b) = hop_keypair();
        let (c_id, c) = hop_keypair();
        let payload = b"three hop secret".to_vec();

        let env = build_onion_envelope(
            &[a.clone(), b.clone(), c.clone()],
            &payload,
            [1u8; 16],
            8,
            1_700_000_000,
            &source,
        )
        .unwrap();

        let p1 = open_onion_layer(&env.encrypted_blob, &x25519_secret(&a_id)).unwrap();
        assert_eq!(p1.next_hop, Some(b.node_id));
        let p2 = open_onion_layer(&p1.inner, &x25519_secret(&b_id)).unwrap();
        assert_eq!(p2.next_hop, Some(c.node_id));
        let p3 = open_onion_layer(&p2.inner, &x25519_secret(&c_id)).unwrap();
        assert_eq!(p3.next_hop, None);
        assert_eq!(p3.inner, payload);
    }

    #[test]
    fn wrong_hop_key_fails_to_peel() {
        let source = IdentityKeyPair::generate();
        let (_entry_id, entry_hop) = hop_keypair();
        let (_exit_id, exit_hop) = hop_keypair();
        let wrong = IdentityKeyPair::generate();

        let env =
            build_onion_envelope(&[entry_hop, exit_hop], b"x", [0u8; 16], 4, 1, &source).unwrap();

        assert!(open_onion_layer(&env.encrypted_blob, &x25519_secret(&wrong)).is_err());
    }

    #[test]
    fn tampered_ephemeral_or_ciphertext_fails() {
        let source = IdentityKeyPair::generate();
        let (entry_id, entry_hop) = hop_keypair();
        let (_exit_id, exit_hop) = hop_keypair();

        let env =
            build_onion_envelope(&[entry_hop, exit_hop], b"payload", [0u8; 16], 4, 1, &source)
                .unwrap();

        // Flip a byte inside the ephemeral public key region.
        let mut tampered_eph = env.encrypted_blob.clone();
        tampered_eph[3] ^= 0xFF;
        assert!(open_onion_layer(&tampered_eph, &x25519_secret(&entry_id)).is_err());

        // Flip a byte inside the ciphertext region.
        let mut tampered_ct = env.encrypted_blob.clone();
        let last = tampered_ct.len() - 1;
        tampered_ct[last] ^= 0xFF;
        assert!(open_onion_layer(&tampered_ct, &x25519_secret(&entry_id)).is_err());
    }

    #[test]
    fn single_hop_is_immediately_terminal() {
        let source = IdentityKeyPair::generate();
        let (exit_id, exit_hop) = hop_keypair();
        let payload = b"direct".to_vec();

        let env =
            build_onion_envelope(&[exit_hop.clone()], &payload, [0u8; 16], 2, 1, &source).unwrap();
        assert_eq!(env.next_hop, exit_hop.node_id);

        let peel = open_onion_layer(&env.encrypted_blob, &x25519_secret(&exit_id)).unwrap();
        assert_eq!(peel.next_hop, None);
        assert_eq!(peel.inner, payload);
    }

    #[test]
    fn non_onion_blob_is_detected() {
        assert!(!is_onion_blob(b""));
        assert!(!is_onion_blob(&[0x00, 0x01, 0x02]));
        assert!(is_onion_blob(&[0xA0, 0x01, 0x99]));
    }

    #[test]
    fn try_open_succeeds_with_previous_key_in_candidate_set() {
        let source = IdentityKeyPair::generate();
        let (exit_id, exit_hop) = hop_keypair();
        let wrong = IdentityKeyPair::generate();
        let payload = b"rotation grace".to_vec();

        let env = build_onion_envelope(&[exit_hop], &payload, [0u8; 16], 2, 1, &source).unwrap();

        // Correct key second in the list (simulates current=wrong, previous=correct).
        let candidates = [x25519_secret(&wrong), x25519_secret(&exit_id)];
        let peel = try_open_onion_layer(&env.encrypted_blob, &candidates).unwrap();
        assert_eq!(peel.next_hop, None);
        assert_eq!(peel.inner, payload);

        // No correct key → fail.
        let only_wrong = [x25519_secret(&wrong)];
        assert!(try_open_onion_layer(&env.encrypted_blob, &only_wrong).is_err());
    }

    #[test]
    fn empty_path_is_rejected() {
        let source = IdentityKeyPair::generate();
        assert!(build_onion_envelope(&[], b"x", [0u8; 16], 1, 1, &source).is_err());
    }
}
