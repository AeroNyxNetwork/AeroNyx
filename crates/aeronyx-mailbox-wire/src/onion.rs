// ============================================
// File: crates/aeronyx-mailbox-wire/src/onion.rs
// ============================================
//! Onion layers for a client source, byte-identical to
//! `aeronyx-core::protocol::onion`.
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude] Layer: `magic(2) ‖ eph_pub(32) ‖
//! nonce(24) ‖ XChaCha20-Poly1305(payload)`, key = HKDF-SHA256(salt
//! "AeroNyx-Onion-v1", ikm X25519(eph, hop), info eph_pub ‖ hop_kem_pub).
//! Payload: `flags(1) ‖ next_hop(32)? ‖ inner_len(u32 LE) ‖ inner`.
//!
//! Descriptor verification is deliberately NOT here: callers pass hops whose
//! node id and KEM key they already took from a verified signed descriptor.

use hkdf::Hkdf;
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::Sha256;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use zeroize::Zeroize;

use crate::chat::BlindRelayEnvelope;
use crate::crypto::{E2eSession, EphemeralKeyPair, IdentityKeyPair};

/// Onion layer magic.
pub const ONION_MAGIC: [u8; 2] = [0xA0, 0x01];
/// HKDF salt.
pub const ONION_SALT: &[u8] = b"AeroNyx-Onion-v1";
/// Maximum hops a source builds.
pub const MAX_SOURCE_ONION_HOPS: usize = 3;
const LAYER_HEADER_LEN: usize = 2 + 32 + 24;
const MAX_ONION_PAYLOAD_BYTES: usize = 256 * 1024;

/// Onion errors. No key, payload or topology material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum OnionError {
    /// Empty path or more than [`MAX_SOURCE_ONION_HOPS`].
    #[error("invalid onion path length")]
    PathLength,
    /// Payload above the per-layer ceiling.
    #[error("onion payload too large")]
    TooLarge,
    /// Malformed or unauthentic layer.
    #[error("onion layer malformed")]
    Malformed,
}

/// One verified hop: node id and its published X25519 KEM key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnionHop {
    /// Ed25519 node id.
    pub node_id: [u8; 32],
    /// X25519 KEM public key from the node's signed descriptor.
    pub kem_pub: [u8; 32],
}

/// One peeled layer (used by tests and terminals).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnionPeel {
    /// Next hop, `None` at the terminal.
    pub next_hop: Option<[u8; 32]>,
    /// Inner bytes.
    pub inner: Vec<u8>,
}

/// Builds the signed envelope a source posts to `path[0]`. `ttl` is the hop
/// count, as the node's verified route sets it.
///
/// # Errors
/// Fails for an empty or over-long path or an over-size payload.
pub fn build_source_envelope(
    path: &[OnionHop],
    final_payload: &[u8],
    route_id: [u8; 16],
    now: u64,
    source: &IdentityKeyPair,
) -> Result<BlindRelayEnvelope, OnionError> {
    if path.is_empty() || path.len() > MAX_SOURCE_ONION_HOPS {
        return Err(OnionError::PathLength);
    }
    let mut inner = final_payload.to_vec();
    for i in (0..path.len()).rev() {
        let next_hop = path.get(i + 1).map(|hop| hop.node_id);
        let encoded = encode_payload(next_hop, &inner)?;
        inner = seal_layer(&path[i].kem_pub, &encoded, EphemeralKeyPair::generate(), random_nonce())?;
    }
    let ttl = u8::try_from(path.len()).map_err(|_| OnionError::PathLength)?;
    Ok(BlindRelayEnvelope {
        route_id,
        next_hop: path[0].node_id,
        ttl,
        encrypted_blob: inner,
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(source))
}

/// Peels one layer with a hop's X25519 secret.
///
/// # Errors
/// Fails for anything that is not exactly one authentic layer.
pub fn open_onion_layer(blob: &[u8], node_x25519_sk: &StaticSecret) -> Result<OnionPeel, OnionError> {
    if blob.len() < LAYER_HEADER_LEN || blob[..2] != ONION_MAGIC {
        return Err(OnionError::Malformed);
    }
    let mut eph_pub = [0u8; 32];
    eph_pub.copy_from_slice(&blob[2..34]);
    let mut nonce = [0u8; 24];
    nonce.copy_from_slice(&blob[34..LAYER_HEADER_LEN]);
    let hop_kem_pub = X25519PublicKey::from(node_x25519_sk).to_bytes();
    let mut ecdh = node_x25519_sk
        .diffie_hellman(&X25519PublicKey::from(eph_pub))
        .to_bytes();
    let key = derive_layer_key(&ecdh, &eph_pub, &hop_kem_pub)?;
    ecdh.zeroize();
    let plaintext = E2eSession::new(key, eph_pub)
        .decrypt_raw(&blob[LAYER_HEADER_LEN..], &nonce)
        .map_err(|_| OnionError::Malformed)?;
    decode_payload(&plaintext)
}

/// Seals one layer with explicit randomness (golden vectors pin both).
pub(crate) fn seal_layer(
    hop_kem_pub: &[u8; 32],
    plaintext: &[u8],
    ephemeral: EphemeralKeyPair,
    nonce: [u8; 24],
) -> Result<Vec<u8>, OnionError> {
    let eph_pub = ephemeral.public_key_bytes();
    let mut ecdh = ephemeral.exchange(hop_kem_pub);
    let key = derive_layer_key(&ecdh, &eph_pub, hop_kem_pub)?;
    ecdh.zeroize();
    let ciphertext = E2eSession::new(key, *hop_kem_pub)
        .encrypt_raw(plaintext, &nonce)
        .map_err(|_| OnionError::Malformed)?;
    let mut out = Vec::with_capacity(LAYER_HEADER_LEN + ciphertext.len());
    out.extend_from_slice(&ONION_MAGIC);
    out.extend_from_slice(&eph_pub);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn random_nonce() -> [u8; 24] {
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut nonce);
    nonce
}

fn derive_layer_key(
    ecdh: &[u8; 32],
    eph_pub: &[u8; 32],
    hop_kem_pub: &[u8; 32],
) -> Result<[u8; 32], OnionError> {
    let mut info = [0u8; 64];
    info[..32].copy_from_slice(eph_pub);
    info[32..].copy_from_slice(hop_kem_pub);
    let mut key = [0u8; 32];
    Hkdf::<Sha256>::new(Some(ONION_SALT), ecdh)
        .expand(&info, &mut key)
        .map_err(|_| OnionError::Malformed)?;
    Ok(key)
}

pub(crate) fn encode_payload(next_hop: Option<[u8; 32]>, inner: &[u8]) -> Result<Vec<u8>, OnionError> {
    if inner.len() > MAX_ONION_PAYLOAD_BYTES {
        return Err(OnionError::TooLarge);
    }
    let mut out = Vec::with_capacity(1 + 32 + 4 + inner.len());
    match next_hop {
        Some(next_hop) => {
            out.push(0x01);
            out.extend_from_slice(&next_hop);
        }
        None => out.push(0x00),
    }
    let len = u32::try_from(inner.len()).map_err(|_| OnionError::TooLarge)?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(inner);
    Ok(out)
}

fn decode_payload(bytes: &[u8]) -> Result<OnionPeel, OnionError> {
    let (&flags, rest) = bytes.split_first().ok_or(OnionError::Malformed)?;
    let (next_hop, rest) = if flags & 0x01 == 0x01 {
        let (hop, rest) = rest.split_at_checked(32).ok_or(OnionError::Malformed)?;
        (Some(<[u8; 32]>::try_from(hop).map_err(|_| OnionError::Malformed)?), rest)
    } else {
        (None, rest)
    };
    let (len, rest) = rest.split_at_checked(4).ok_or(OnionError::Malformed)?;
    let len = u32::from_le_bytes(len.try_into().map_err(|_| OnionError::Malformed)?) as usize;
    if len > MAX_ONION_PAYLOAD_BYTES || rest.len() != len {
        return Err(OnionError::Malformed);
    }
    Ok(OnionPeel {
        next_hop,
        inner: rest.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_hop_envelope_peels_at_each_hop() {
        let entry = IdentityKeyPair::from_bytes(&[1u8; 32]).unwrap();
        let terminal = IdentityKeyPair::from_bytes(&[2u8; 32]).unwrap();
        let source = IdentityKeyPair::from_bytes(&[3u8; 32]).unwrap();
        let path = [
            OnionHop {
                node_id: entry.public_key_bytes(),
                kem_pub: entry.x25519_public_key_bytes(),
            },
            OnionHop {
                node_id: terminal.public_key_bytes(),
                kem_pub: terminal.x25519_public_key_bytes(),
            },
        ];
        let envelope = build_source_envelope(&path, b"payload", [9u8; 16], 1_800_000_000, &source)
            .unwrap();
        assert_eq!(envelope.ttl, 2);
        assert_eq!(envelope.next_hop, entry.public_key_bytes());
        source
            .public_key()
            .verify(&envelope.signing_data(), &envelope.signature)
            .unwrap();
        let first = open_onion_layer(&envelope.encrypted_blob, &entry.to_x25519().0).unwrap();
        assert_eq!(first.next_hop, Some(terminal.public_key_bytes()));
        let second = open_onion_layer(&first.inner, &terminal.to_x25519().0).unwrap();
        assert_eq!(second.next_hop, None);
        assert_eq!(second.inner, b"payload");
        assert!(open_onion_layer(&first.inner, &entry.to_x25519().0).is_err());
    }
}
