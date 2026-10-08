// ============================================
// File: crates/aeronyx-mailbox-wire/src/chat.rs
// ============================================
//! The chat envelope and blind relay envelope, byte-identical to
//! `aeronyx-core::protocol::chat`.
//!
//! [MAILBOX-WIRE 2026-10-09 by Claude] Only what the mailbox needs: the signed
//! E2E `ChatEnvelope` sealed into mailbox items, and the `BlindRelayEnvelope`
//! a client source posts to an entry node. Field order and serde shapes are
//! wire contracts; do not reorder.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};
use crate::crypto::{CryptoError, IdentityKeyPair, IdentityPublicKey};

/// Maximum encoded chat envelope.
pub const MAX_CHAT_ENVELOPE_BYTES: u64 = 128 * 1024;
const BLIND_RELAY_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-v1";

/// Chat errors. No message material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ChatWireError {
    /// Not one canonical envelope.
    #[error("chat envelope is not canonical")]
    NotCanonical,
    /// Sender signature failed under the strict policy.
    #[error("chat envelope signature rejected")]
    Signature,
}

pub(crate) mod serde_bytes64 {
    use super::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(value: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error> {
        let (lo, hi) = value.split_at(32);
        let lo: [u8; 32] = lo.try_into().expect("32-byte half");
        let hi: [u8; 32] = hi.try_into().expect("32-byte half");
        (lo, hi).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 64], D::Error> {
        let (lo, hi): ([u8; 32], [u8; 32]) = Deserialize::deserialize(deserializer)?;
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&lo);
        out[32..].copy_from_slice(&hi);
        Ok(out)
    }
}

/// Content type; values are frozen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ChatContentType {
    /// Text.
    Text = 0,
    /// Media pointer.
    Media = 1,
    /// System / control.
    System = 2,
}

impl ChatContentType {
    /// Signing discriminant.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Signed, E2E-encrypted chat envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatEnvelope {
    /// Message id.
    pub message_id: [u8; 16],
    /// Sender Ed25519 key.
    pub sender: [u8; 32],
    /// Receiver Ed25519 key.
    pub receiver: [u8; 32],
    /// Unix seconds.
    pub timestamp: u64,
    /// E2E ciphertext.
    pub ciphertext: Vec<u8>,
    /// XChaCha20 nonce.
    pub nonce: [u8; 24],
    /// Content type.
    pub content_type: ChatContentType,
    /// Ed25519 signature over [`ChatEnvelope::sign_data`].
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl ChatEnvelope {
    /// `sender ‖ message_id ‖ receiver ‖ timestamp_le ‖ content_type ‖ SHA256(ciphertext)`.
    #[must_use]
    pub fn sign_data(&self) -> Vec<u8> {
        let ct_hash = Sha256::digest(&self.ciphertext);
        let mut data = Vec::with_capacity(121);
        data.extend_from_slice(&self.sender);
        data.extend_from_slice(&self.message_id);
        data.extend_from_slice(&self.receiver);
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data.push(self.content_type.as_u8());
        data.extend_from_slice(&ct_hash);
        data
    }

    /// Strict sender signature verification.
    ///
    /// # Errors
    /// Fails on a bad key or signature.
    pub fn verify_signature(&self) -> Result<(), CryptoError> {
        IdentityPublicKey::from_bytes(&self.sender)?.verify(&self.sign_data(), &self.signature)
    }
}

/// Encodes an envelope under the 128 KiB ceiling.
///
/// # Errors
/// Fails if encoding exceeds the ceiling.
pub fn encode_envelope(envelope: &ChatEnvelope) -> Result<Vec<u8>, bincode::Error> {
    encode_bincode_bounded(envelope, MAX_CHAT_ENVELOPE_BYTES)
}

/// Decodes exactly one canonical envelope and verifies its signature.
///
/// # Errors
/// Fails for trailing, non-canonical or incorrectly signed input.
pub fn decode_envelope_strict_verified(bytes: &[u8]) -> Result<ChatEnvelope, ChatWireError> {
    let envelope: ChatEnvelope =
        decode_bincode_bounded(bytes, MAX_CHAT_ENVELOPE_BYTES, TrailingBytesPolicy::Reject)
            .map_err(|_| ChatWireError::NotCanonical)?;
    let canonical = encode_envelope(&envelope).map_err(|_| ChatWireError::NotCanonical)?;
    if canonical != bytes {
        return Err(ChatWireError::NotCanonical);
    }
    envelope
        .verify_signature()
        .map_err(|_| ChatWireError::Signature)?;
    Ok(envelope)
}

/// Opaque onion relay envelope posted to an entry node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindRelayEnvelope {
    /// Route correlation id.
    pub route_id: [u8; 16],
    /// The node this envelope is addressed to.
    pub next_hop: [u8; 32],
    /// Hop budget.
    pub ttl: u8,
    /// Opaque onion bytes.
    pub encrypted_blob: Vec<u8>,
    /// Unix seconds.
    pub timestamp: u64,
    /// Previous-hop Ed25519 signature over [`BlindRelayEnvelope::signing_data`].
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindRelayEnvelope {
    /// `domain ‖ route_id ‖ next_hop ‖ ttl ‖ timestamp_le ‖ SHA256(blob)`.
    #[must_use]
    pub fn signing_data(&self) -> Vec<u8> {
        let blob_hash = Sha256::digest(&self.encrypted_blob);
        let mut data = Vec::with_capacity(BLIND_RELAY_SIGNING_DOMAIN.len() + 16 + 32 + 1 + 8 + 32);
        data.extend_from_slice(BLIND_RELAY_SIGNING_DOMAIN);
        data.extend_from_slice(&self.route_id);
        data.extend_from_slice(&self.next_hop);
        data.push(self.ttl);
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data.extend_from_slice(&blob_hash);
        data
    }

    /// Signs with the previous-hop (source) key.
    #[must_use]
    pub fn sign_with(mut self, keypair: &IdentityKeyPair) -> Self {
        self.signature = keypair.sign(&self.signing_data());
        self
    }
}

/// The JSON body a source POSTs to `/api/chat/peer/blind-relay`.
///
/// Mirrors the node's `PeerBlindRelayRequest`; the optional onward fields are
/// always absent for a client source.
#[derive(Debug, Clone, Serialize)]
pub struct PeerBlindRelayRequest {
    /// The envelope.
    pub envelope: BlindRelayEnvelope,
    /// Source Ed25519 key that signed the envelope.
    pub previous_hop_node_id: [u8; 32],
    /// Always `None` for a client source; omitted from JSON like the node does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onward_envelope: Option<BlindRelayEnvelope>,
}
