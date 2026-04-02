// ============================================
// File: crates/aeronyx-core/src/protocol/chat.rs
// ============================================
//! # Chat Protocol — End-to-End Encrypted Messaging
//!
//! ## Creation Reason
//! Defines the data structures for AeroNyx Chat Relay, a zero-knowledge
//! P2P messaging layer built on top of the existing MemChain 0xAE channel.
//!
//! The node acts as a blind relay: it can read routing fields (sender,
//! receiver) for delivery, but cannot decrypt message content.
//!
//! ## Main Functionality
//! - `ChatEnvelope`: The wire format for an E2E-encrypted message
//! - `ChatContentType`: Discriminator for text / media / system messages
//! - `MediaPointer`: Encrypted file reference (lives inside ChatEnvelope.ciphertext)
//! - `sign_envelope_data()`: Canonical byte sequence for Ed25519 signing
//! - `verify_envelope()`: Convenience wrapper for signature verification
//! - `encode_envelope()` / `decode_envelope()`: bincode helpers
//!
//! ## E2E Encryption (Flutter client reference)
//! ```text
//! 1. Alice derives shared_secret = X25519(Alice_x25519_sk, Bob_x25519_pk)
//!    where x25519 keys are converted from Ed25519 via SHA-512 (same as keys.rs)
//! 2. plaintext → XChaCha20-Poly1305(shared_secret, random_nonce_24B)
//! 3. Sign sign_envelope_data(envelope) with Alice's Ed25519 private key
//! 4. Wrap in ChatEnvelope and send via MemChainMessage::ChatRelay
//! ```
//!
//! ## Signature Coverage
//! `sign_envelope_data` covers:
//!   sender(32) || message_id(16) || receiver(32) || timestamp(8) ||
//!   content_type(1) || SHA256(ciphertext)(32)
//!
//! This binding ensures:
//! - sender cannot be replaced (public key is in signed data AND used to verify)
//! - ciphertext cannot be swapped (content hash is signed)
//! - content_type cannot be changed (prevents media→text downgrade attacks)
//!
//! ## Dependencies
//! - `aeronyx-core/src/crypto/keys.rs`: `IdentityPublicKey::verify()` for sig check
//! - `aeronyx-core/src/protocol/memchain.rs`: `MemChainMessage::ChatRelay(ChatEnvelope)`
//! - `aeronyx-server/src/services/chat_relay.rs`: consumes these types for storage/routing
//! - Flutter client: must implement compatible X25519 ECDH + XChaCha20-Poly1305
//!
//! ## ⚠️ Important Notes for Next Developer
//! - NEVER add fields between existing ChatEnvelope fields without a bincode migration plan.
//!   bincode uses positional serialisation — field order is the wire format.
//! - message_id is CLIENT-generated (UUID v4 or random 16 bytes).
//!   The node uses it as SQLite PRIMARY KEY for deduplication only.
//! - nonce is 24 bytes (XChaCha20 requirement). Do NOT shorten to 12 bytes (ChaCha20).
//! - MediaPointer lives INSIDE ciphertext — the node never sees its contents.
//! - thumbnail_b64 in MediaPointer is optional inline preview (< 4KB).
//!   It is base64-encoded JPEG after encryption of the main file, NOT before.
//! - blob_id in MediaPointer is computed by the NODE (HMAC-SHA256 derived),
//!   returned to the client after POST /api/chat/blob, then embedded here.
//! - file_key is independent of the chat E2E shared_secret — double-layer protection.
//!
//! ## Last Modified
//! v1.0.0-ChatRelay — Initial implementation

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use bincode::Options;

use crate::crypto::keys::IdentityPublicKey;
use crate::error::CoreError;

// ============================================
// Deserialisation size limit
// ============================================

/// Maximum accepted byte size for a single `ChatEnvelope` payload.
/// Text ciphertext ≤ 64 KB + fixed fields ≤ ~1 KB overhead.
/// Prevents bincode length-prefix OOM attacks.
const MAX_ENVELOPE_BYTES: u64 = 128 * 1024; // 128 KB

// ============================================
// Serde helper for [u8; 64]
// ============================================
// serde only auto-derives array impls up to [T; 32]. Ed25519 signatures
// are 64 bytes, so we need a manual helper. We serialise as a fixed-length
// byte sequence (no length prefix in bincode, just 64 raw bytes).

mod serde_bytes64 {
    use super::*;
    use serde::de::{self, Visitor};
    use std::fmt;

    pub fn serialize<S: Serializer>(v: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        // Serialize as a tuple of two [u8;32] — both halves have stable serde impls.
        // This produces exactly 64 bytes in bincode (no length prefix).
        let (lo, hi) = v.split_at(32);
        let lo: [u8; 32] = lo.try_into().unwrap();
        let hi: [u8; 32] = hi.try_into().unwrap();
        (lo, hi).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let (lo, hi): ([u8; 32], [u8; 32]) = Deserialize::deserialize(d)?;
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&lo);
        out[32..].copy_from_slice(&hi);
        Ok(out)
    }
}

// ============================================
// ChatContentType
// ============================================

/// Discriminator for the content inside `ChatEnvelope.ciphertext`.
///
/// The node CAN see this field (used for size validation and rate limiting),
/// but CANNOT decrypt the actual content.
///
/// ## Wire Representation
/// `#[repr(u8)]` — serialised as a single byte by bincode.
/// Existing values MUST NOT be renumbered.
///
/// | Value | Variant | ciphertext contents |
/// |-------|---------|---------------------|
/// | 0     | Text    | XChaCha20(UTF-8 text, ≤ 64 KB) |
/// | 1     | Media   | XChaCha20(MediaPointer JSON) |
/// | 2     | System  | XChaCha20(system payload) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ChatContentType {
    /// Plain text message (ciphertext ≤ 64 KB, transmitted over UDP).
    Text = 0,
    /// Media message (ciphertext contains encrypted `MediaPointer`; file travels via HTTP).
    Media = 1,
    /// System message (friend requests, ACKs, notifications).
    System = 2,
}

impl ChatContentType {
    /// Returns the `u8` discriminant for use in signing.
    #[inline]
    #[must_use]
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

// ============================================
// ChatEnvelope
// ============================================

/// End-to-end encrypted chat message envelope.
///
/// ## Wire Layout (bincode, positional)
/// ```text
/// message_id   [u8; 16]      16 bytes  client-generated unique ID
/// sender       [u8; 32]      32 bytes  sender Ed25519 public key (wallet)
/// receiver     [u8; 32]      32 bytes  receiver Ed25519 public key (wallet)
/// timestamp    u64            8 bytes  Unix epoch seconds
/// ciphertext   Vec<u8>       variable  XChaCha20-Poly1305 encrypted payload
/// nonce        [u8; 24]      24 bytes  random nonce for XChaCha20
/// content_type ChatContentType 1 byte  Text/Media/System
/// signature    [u8; 64]      64 bytes  Ed25519 signature over sign_envelope_data()
/// ```
///
/// ## Zero-Knowledge Property
/// The node reads `sender`, `receiver`, `timestamp`, and `content_type` for routing
/// and validation. It CANNOT decrypt `ciphertext` — the shared_secret is derived
/// client-side via X25519 ECDH and never transmitted.
///
/// ## Field Ordering Note
/// Fields are ordered so that the routing-critical fields come first in the
/// binary layout. Do not reorder without a bincode migration plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatEnvelope {
    /// Unique message ID (client-generated, UUID v4 or random 16 bytes).
    /// Used as SQLite PRIMARY KEY for deduplication on the node.
    pub message_id: [u8; 16],

    /// Sender's Ed25519 public key (= wallet address).
    /// Also used as the verification key for `signature`.
    pub sender: [u8; 32],

    /// Receiver's Ed25519 public key (= wallet address).
    /// Routing target — node forwards to this wallet's active session.
    pub receiver: [u8; 32],

    /// Send timestamp (Unix epoch seconds, client clock).
    /// Node validates |now - timestamp| < MAX_CHAT_TIMESTAMP_SKEW (300 s).
    pub timestamp: u64,

    /// E2E encrypted message payload.
    /// - Text:   XChaCha20-Poly1305(shared_secret, nonce, UTF-8 text)
    /// - Media:  XChaCha20-Poly1305(shared_secret, nonce, MediaPointer JSON)
    /// - System: XChaCha20-Poly1305(shared_secret, nonce, system payload)
    pub ciphertext: Vec<u8>,

    /// 24-byte random nonce for XChaCha20-Poly1305.
    /// MUST be unique per message. Client generates this randomly.
    pub nonce: [u8; 24],

    /// Content type (visible to the node for size checks and rate limiting).
    pub content_type: ChatContentType,

    /// Ed25519 signature over `sign_envelope_data(self)`.
    /// Signed with the sender's Ed25519 private key.
    /// Verified using `self.sender` as the public key.
    ///
    /// Serialised as two consecutive `[u8; 32]` halves (bincode has no
    /// built-in impl for arrays larger than 32 bytes).
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl ChatEnvelope {
    /// Computes the canonical byte sequence used for signing and verification.
    ///
    /// Layout:
    /// ```text
    /// sender(32) || message_id(16) || receiver(32) || timestamp_le(8) ||
    /// content_type(1) || SHA256(ciphertext)(32)
    /// ```
    ///
    /// Fixed overhead: 121 bytes (independent of ciphertext size).
    ///
    /// ## Security Properties
    /// - `sender` in signed data → cannot replace sender without breaking sig
    /// - `receiver` bound → cannot redirect message
    /// - `SHA256(ciphertext)` → content integrity without signing large blobs
    /// - `content_type` → prevents media→text downgrade attacks
    #[must_use]
    pub fn sign_data(&self) -> Vec<u8> {
        let ct_hash = Sha256::digest(&self.ciphertext);
        // 32 + 16 + 32 + 8 + 1 + 32 = 121 bytes
        let mut data = Vec::with_capacity(121);
        data.extend_from_slice(&self.sender);
        data.extend_from_slice(&self.message_id);
        data.extend_from_slice(&self.receiver);
        data.extend_from_slice(&self.timestamp.to_le_bytes());
        data.push(self.content_type.as_u8());
        data.extend_from_slice(&ct_hash);
        data
    }

    /// Verifies the envelope's Ed25519 signature.
    ///
    /// Uses `self.sender` as the public key. If the sender field has been
    /// tampered with, this will fail because:
    /// 1. The public key will not match the original signing key.
    /// 2. `sender` is also included in the signed data.
    ///
    /// # Errors
    /// Returns `CoreError` if signature verification fails or if `sender`
    /// bytes do not form a valid Ed25519 public key.
    pub fn verify_signature(&self) -> Result<(), CoreError> {
        // from_bytes expects &[u8; 32] — self.sender is [u8; 32] so &self.sender is &[u8; 32]
        let pk = IdentityPublicKey::from_bytes(&self.sender)?;
        // verify expects (&[u8], &[u8; 64]) — sign_data() returns Vec<u8>, signature is [u8; 64]
        pk.verify(&self.sign_data(), &self.signature)
    }

    /// Returns a compact hex prefix of `message_id` for log output.
    #[must_use]
    pub fn short_id(&self) -> String {
        hex::encode(&self.message_id[..4])
    }

    /// Returns the sender's wallet address as a lowercase hex string.
    #[must_use]
    pub fn sender_hex(&self) -> String {
        hex::encode(self.sender)
    }

    /// Returns the receiver's wallet address as a lowercase hex string.
    #[must_use]
    pub fn receiver_hex(&self) -> String {
        hex::encode(self.receiver)
    }
}

// ============================================
// MediaPointer
// ============================================

/// Pointer to an encrypted media file stored in the node's blob cache.
///
/// This struct is serialised to JSON, encrypted with the chat `shared_secret`,
/// and placed in `ChatEnvelope.ciphertext` when `content_type == Media`.
///
/// ## Zero-Knowledge Property
/// The node NEVER sees this struct — it lives inside the E2E ciphertext.
/// The node only stores `blob_id → encrypted_bytes` without knowing the
/// file type, filename, or decryption key.
///
/// ## Dual-Layer Encryption
/// ```text
/// file_key  (independent 32-byte key, client-generated)
///     └── encrypts raw_file_bytes → encrypted_file  [stored in node blob cache]
///
/// shared_secret  (X25519 ECDH between sender and receiver)
///     └── encrypts MediaPointer JSON → ChatEnvelope.ciphertext
/// ```
///
/// Even if the chat shared_secret leaks, the file cannot be decrypted
/// without also obtaining the MediaPointer (which requires the chat key).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaPointer {
    /// Blob ID on the node (HMAC-SHA256 derived, 32 hex chars).
    /// Returned by `POST /api/chat/blob` — Alice must upload first.
    /// Bob uses this to download: `GET /api/chat/blob/{blob_id}`.
    pub blob_id: String,

    /// Independent 32-byte encryption key for the file content.
    /// Client-generated randomly — NOT derived from the chat shared_secret.
    pub file_key: [u8; 32],

    /// 24-byte nonce for the file encryption (XChaCha20-Poly1305).
    pub file_nonce: [u8; 24],

    /// Original filename (e.g. `"photo_2026.jpg"`).
    pub filename: String,

    /// MIME type (e.g. `"image/jpeg"`, `"application/pdf"`).
    pub mime_type: String,

    /// Size of the original (unencrypted) file in bytes.
    pub file_size: u64,

    /// SHA-256 of the original (pre-encryption) file bytes.
    /// Bob verifies this after decryption to detect corruption or tampering.
    pub plaintext_hash: [u8; 32],

    /// Optional inline thumbnail (base64-encoded JPEG, ≤ 4 KB).
    ///
    /// When present, the receiver can display a preview without downloading
    /// the full encrypted file. Generated by the sender's Flutter client.
    ///
    /// Encoding: `base64_standard(small_jpeg_bytes)`.
    pub thumbnail_b64: Option<String>,
}

// ============================================
// Encode / Decode helpers
// ============================================

/// Encodes a `ChatEnvelope` to bytes using bincode.
///
/// # Errors
/// Returns `bincode::Error` if serialisation fails (should not happen in practice).
pub fn encode_envelope(envelope: &ChatEnvelope) -> Result<Vec<u8>, bincode::Error> {
    bincode::serialize(envelope)
}

/// Decodes a `ChatEnvelope` from a bincode byte slice.
///
/// # Size limit
/// Rejects inputs that would require allocating more than `MAX_ENVELOPE_BYTES`
/// (128 KB). Prevents a malicious length-prefix from triggering large allocations.
///
/// # Errors
/// Returns `bincode::Error` if the bytes are malformed, truncated, or too large.
pub fn decode_envelope(bytes: &[u8]) -> Result<ChatEnvelope, bincode::Error> {
    bincode::options()
        .with_limit(MAX_ENVELOPE_BYTES)
        .with_fixint_encoding()
        .allow_trailing_bytes()
        .deserialize(bytes)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::IdentityKeyPair;

    /// Helper: build a minimal ChatEnvelope with a real Ed25519 signature.
    fn make_signed_envelope(kp: &IdentityKeyPair) -> ChatEnvelope {
        let sender = kp.public_key_bytes();
        let receiver = [0xBBu8; 32];
        let message_id = [0x01u8; 16];
        let timestamp: u64 = 1_700_000_000;
        let ciphertext = b"encrypted_hello_world".to_vec();
        let nonce = [0x02u8; 24];
        let content_type = ChatContentType::Text;

        // Build unsigned envelope to compute sign_data
        let mut env = ChatEnvelope {
            message_id,
            sender,
            receiver,
            timestamp,
            ciphertext,
            nonce,
            content_type,
            signature: [0u8; 64],
        };

        // Sign
        let data = env.sign_data();
        env.signature = kp.sign(&data);
        env
    }

    // ── ChatContentType ──

    #[test]
    fn test_content_type_discriminants() {
        assert_eq!(ChatContentType::Text.as_u8(), 0);
        assert_eq!(ChatContentType::Media.as_u8(), 1);
        assert_eq!(ChatContentType::System.as_u8(), 2);
    }

    #[test]
    fn test_content_type_serde_roundtrip() {
        for ct in [ChatContentType::Text, ChatContentType::Media, ChatContentType::System] {
            let bytes = bincode::serialize(&ct).expect("serialize");
            let decoded: ChatContentType = bincode::deserialize(&bytes).expect("deserialize");
            assert_eq!(ct, decoded);
        }
    }

    // ── ChatEnvelope ──

    #[test]
    fn test_envelope_sign_data_length() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        // sign_data must be exactly 121 bytes
        assert_eq!(env.sign_data().len(), 121);
    }

    #[test]
    fn test_envelope_sign_data_deterministic() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        assert_eq!(env.sign_data(), env.sign_data());
    }

    #[test]
    fn test_envelope_verify_signature_ok() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        assert!(env.verify_signature().is_ok(), "Valid signature must verify");
    }

    #[test]
    fn test_envelope_tampered_sender_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        // Replace sender with a different key
        env.sender = [0xAAu8; 32];
        assert!(
            env.verify_signature().is_err(),
            "Tampered sender must fail verification"
        );
    }

    #[test]
    fn test_envelope_tampered_receiver_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        env.receiver[0] ^= 0xFF;
        assert!(env.verify_signature().is_err(), "Tampered receiver must fail");
    }

    #[test]
    fn test_envelope_tampered_ciphertext_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        env.ciphertext[0] ^= 0xFF;
        assert!(env.verify_signature().is_err(), "Tampered ciphertext must fail");
    }

    #[test]
    fn test_envelope_tampered_timestamp_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        env.timestamp += 1;
        assert!(env.verify_signature().is_err(), "Tampered timestamp must fail");
    }

    #[test]
    fn test_envelope_tampered_content_type_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        // Flip Text → Media
        env.content_type = ChatContentType::Media;
        assert!(env.verify_signature().is_err(), "Tampered content_type must fail");
    }

    #[test]
    fn test_envelope_bincode_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);

        let bytes = encode_envelope(&env).expect("encode");
        let decoded = decode_envelope(&bytes).expect("decode");

        assert_eq!(env.message_id, decoded.message_id);
        assert_eq!(env.sender, decoded.sender);
        assert_eq!(env.receiver, decoded.receiver);
        assert_eq!(env.timestamp, decoded.timestamp);
        assert_eq!(env.ciphertext, decoded.ciphertext);
        assert_eq!(env.nonce, decoded.nonce);
        assert_eq!(env.content_type, decoded.content_type);
        assert_eq!(env.signature, decoded.signature);

        // Decoded envelope must still verify
        assert!(decoded.verify_signature().is_ok());
    }

    #[test]
    fn test_envelope_short_id_is_8_hex_chars() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        // 4 bytes → 8 hex chars
        assert_eq!(env.short_id().len(), 8);
    }

    #[test]
    fn test_envelope_sender_hex_is_64_chars() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        assert_eq!(env.sender_hex().len(), 64);
        assert_eq!(env.receiver_hex().len(), 64);
    }

    // ── MediaPointer ──

    #[test]
    fn test_media_pointer_serde_roundtrip() {
        let mp = MediaPointer {
            blob_id: "abc123def456abc123def456abc123de".to_string(),
            file_key: [0x42u8; 32],
            file_nonce: [0x43u8; 24],
            filename: "photo_2026.jpg".to_string(),
            mime_type: "image/jpeg".to_string(),
            file_size: 1_024_000,
            plaintext_hash: [0x44u8; 32],
            thumbnail_b64: Some("base64thumbnaildata".to_string()),
        };

        let json = serde_json::to_string(&mp).expect("json serialize");
        let decoded: MediaPointer = serde_json::from_str(&json).expect("json deserialize");

        assert_eq!(mp.blob_id, decoded.blob_id);
        assert_eq!(mp.file_key, decoded.file_key);
        assert_eq!(mp.file_nonce, decoded.file_nonce);
        assert_eq!(mp.filename, decoded.filename);
        assert_eq!(mp.mime_type, decoded.mime_type);
        assert_eq!(mp.file_size, decoded.file_size);
        assert_eq!(mp.plaintext_hash, decoded.plaintext_hash);
        assert_eq!(mp.thumbnail_b64, decoded.thumbnail_b64);
    }

    #[test]
    fn test_media_pointer_optional_thumbnail_none() {
        let mp = MediaPointer {
            blob_id: "abc123".to_string(),
            file_key: [0u8; 32],
            file_nonce: [0u8; 24],
            filename: "doc.pdf".to_string(),
            mime_type: "application/pdf".to_string(),
            file_size: 512,
            plaintext_hash: [0u8; 32],
            thumbnail_b64: None,
        };

        let json = serde_json::to_string(&mp).expect("serialize");
        let decoded: MediaPointer = serde_json::from_str(&json).expect("deserialize");
        assert!(decoded.thumbnail_b64.is_none());
    }

    // ── Sign data covers sender (regression for the original omission) ──

    #[test]
    fn test_sign_data_contains_sender_bytes() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        let data = env.sign_data();
        // First 32 bytes of sign_data must equal sender
        assert_eq!(&data[..32], &env.sender);
    }

    #[test]
    fn test_sign_data_contains_content_type_byte() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        let data = env.sign_data();
        // Byte at offset 88 (32+16+32+8) is content_type
        assert_eq!(data[88], ChatContentType::Text.as_u8());
    }

    // ── Different senders produce different sign_data ──

    #[test]
    fn test_different_senders_different_sign_data() {
        let kp1 = IdentityKeyPair::generate();
        let kp2 = IdentityKeyPair::generate();
        let env1 = make_signed_envelope(&kp1);
        let env2 = make_signed_envelope(&kp2);
        // sign_data must differ when sender differs
        assert_ne!(env1.sign_data(), env2.sign_data());
    }
}
