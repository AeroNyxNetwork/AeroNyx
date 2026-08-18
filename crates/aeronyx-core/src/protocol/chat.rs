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
//! - `BlindRelayEnvelope`: node-to-node opaque forwarding frame for future
//!   controlled multi-hop/onion routing. Nodes see route_id/next_hop/ttl and
//!   an opaque encrypted_blob only; they do not parse the inner chat/media data.
//! - `BlindRelayDeliveryReceipt`: terminal-signed proof that an exact opaque
//!   payload reached the store-and-forward acceptance boundary. The receipt
//!   contains no sender, receiver, endpoint, online-state, or plaintext data.
//! - [SIGNED-FAILURE-RECEIPT 2026-08-11 by Codex] `BlindRelayFailureReceipt`
//!   authenticates one immediate hop's coarse failure ACK against the exact
//!   opaque request. It never identifies deeper hops or reveals payload data.
//! - [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Receipt v2 commits to both
//!   the opaque terminal payload and its canonical route purpose without
//!   exposing a purpose field to middle relays. Receipt v1 remains verifiable
//!   for mixed-version forwarding, but cannot prove workload separation.
//! - [BOUNDED-WIRE-CODEC 2026-07-23 by Codex] Chat and blind-relay encoders
//!   share the same byte ceilings as their decoders. Keep legacy small trailing
//!   bytes only for `ChatEnvelope`; blind-relay frames remain canonical.
//! - [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] `CustodyAuditAnchorV1` is a
//!   fixed-size, node-signed commitment to one opaque relay-custody checkpoint.
//!   It contains no audit MAC, path, message metadata, identity relationship,
//!   ciphertext, or backup name. External retainers compare its monotonic
//!   generation; the anchor alone is not a witness receipt or global finality.
//! - [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] `CustodyAuditWitnessReceiptV1`
//!   binds one exact producer anchor to an independently signed, durable
//!   monotonic witness decision. Accepted and rejected outcomes remain
//!   fixed-size and disclose no additional custody or user information.
//! - [CUSTODY-ANCHOR-METADATA-ACCURACY 2026-08-18 by Codex] The portable
//!   anchor also carries coarse archived record and byte totals. Keep that
//!   disclosure explicit; it still contains no user- or message-level data.
//!
//! ## Last Modified
//! v1.9.0-CustodyAnchorMetadataAccuracy - Correct the documented aggregate
//! metadata visible to an external custody witness
//! v1.8.0-CustodyWitnessNetwork - Added canonical anchor/receipt frame digests
//! and verification of portable adverse outcomes for node-to-node exchange
//! v1.7.0-CustodyAuditWitness - Added independent signed checkpoint receipts
//! v1.6.0-CustodyAuditAnchor - Added portable opaque custody checkpoint anchors
//! v1.5.0-SignedFailureReceipt - Authenticate hop-local failure ACKs without exposing route topology
//! v1.4.0-PurposeBoundReceipt - Bound terminal receipts to route purpose with a v2 domain
//! v1.3.0-BoundedWireCodec - Symmetric frame limits and padded-input rejection
//! v1.2.0-BlindRelayDeliveryReceipt - Added terminal-signed opaque delivery receipt
//! v1.1.0-BlindRelayEnvelope — Added opaque node-to-node relay envelope skeleton
//! v1.0.0-ChatRelay — Initial implementation

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};
use crate::protocol::onion::OnionRoutePurpose;

// ============================================
// Deserialisation size limit
// ============================================

/// Maximum accepted byte size for a single `ChatEnvelope` payload.
/// Text ciphertext ≤ 64 KB + fixed fields ≤ ~1 KB overhead.
/// Prevents bincode length-prefix OOM attacks.
const MAX_ENVELOPE_BYTES: u64 = 128 * 1024; // 128 KB
const MAX_BLIND_RELAY_ENVELOPE_BYTES: u64 = 256 * 1024; // 256 KB opaque relay frame cap
const MAX_BLIND_RELAY_BLOB_BYTES: usize = 192 * 1024; // media/file bytes must use blob storage
const BLIND_RELAY_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-v1";
const BLIND_RELAY_DELIVERY_RECEIPT_V1_SIGNING_DOMAIN: &[u8] =
    b"AeroNyx-BlindRelay-DeliveryReceipt-v1";
const BLIND_RELAY_DELIVERY_RECEIPT_V2_SIGNING_DOMAIN: &[u8] =
    b"AeroNyx-BlindRelay-DeliveryReceipt-v2";
const BLIND_RELAY_PURPOSE_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-PurposeCommitment-v1";
const BLIND_RELAY_FAILURE_RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-FailureReceipt-v1";
const BLIND_RELAY_FAILURE_REQUEST_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindRelay-FailureRequest-v1";
const BLIND_RELAY_FAILURE_REASON_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindRelay-FailureReason-v1";
const CUSTODY_AUDIT_ANCHOR_SIGNING_DOMAIN: &[u8] = b"AeroNyx-CustodyAuditAnchor-v1";
const CUSTODY_AUDIT_WITNESS_RECEIPT_SIGNING_DOMAIN: &[u8] =
    b"AeroNyx-CustodyAuditWitnessReceipt-v1";

/// Initial signed blind-relay terminal receipt version.
pub const BLIND_RELAY_DELIVERY_RECEIPT_VERSION: u8 = 1;
/// Purpose-bound terminal receipt version used by current nodes.
pub const BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION: u8 = 2;
/// Initial hop-local signed blind-relay failure receipt version.
pub const BLIND_RELAY_FAILURE_RECEIPT_VERSION: u8 = 1;
/// Initial portable relay-custody checkpoint anchor version.
pub const CUSTODY_AUDIT_ANCHOR_VERSION: u8 = 1;
/// Exact upper bound for one canonical custody audit anchor frame.
pub const MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES: usize = 256;
/// Initial independent custody audit witness receipt version.
pub const CUSTODY_AUDIT_WITNESS_RECEIPT_VERSION: u8 = 1;
/// Exact upper bound for one canonical custody audit witness receipt frame.
pub const MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES: usize = 320;
/// Witness durably accepted its first observation or exact next generation.
pub const CUSTODY_AUDIT_WITNESS_ADVANCED_V1: u8 = 0;
/// Witness already retained the exact generation and frame digest.
pub const CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1: u8 = 1;
/// Requested generation is below the witness's durable high-water mark.
pub const CUSTODY_AUDIT_WITNESS_STALE_V1: u8 = 2;
/// Requested generation reused the high-water generation with another frame.
pub const CUSTODY_AUDIT_WITNESS_CONFLICT_V1: u8 = 3;
/// Requested generation skipped one or more established witness generations.
pub const CUSTODY_AUDIT_WITNESS_GAP_V1: u8 = 4;
/// Terminal accepted the opaque payload into online or durable pending delivery.
pub const BLIND_RELAY_DELIVERY_ACCEPTED: u8 = 1;

// ============================================
// Serde helper for [u8; 64]
// ============================================
// serde only auto-derives array impls up to [T; 32]. Ed25519 signatures
// are 64 bytes, so we need a manual helper. We serialise as a fixed-length
// byte sequence (no length prefix in bincode, just 64 raw bytes).

mod serde_bytes64 {
    use super::*;

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
// BlindRelayEnvelope
// ============================================

/// Opaque node-to-node forwarding envelope for future multi-hop/onion routing.
///
/// ## Blind relay invariant
/// A Rust node can read only:
///
/// - `route_id`
/// - `next_hop`
/// - `ttl`
/// - `encrypted_blob` length/hash
/// - `timestamp`
/// - `signature`
///
/// It must not parse the inner encrypted payload. The payload may contain a
/// chat envelope, agent protocol frame, Memory Chain coordination message, or
/// future onion layer, but this type intentionally treats it as opaque bytes.
///
/// ## Signature model
/// The signing public key is supplied by the caller/transport context instead
/// of being embedded here, keeping the envelope aligned with the minimal field
/// set above. HTTP/gossip layers can bind the previous-hop node identity before
/// calling `verify_signature_from()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindRelayEnvelope {
    /// Route correlation id. Random per route; not a user/message id.
    pub route_id: [u8; 16],
    /// Next AeroNyx node id that should receive this opaque blob.
    pub next_hop: [u8; 32],
    /// Hop budget. Receivers must decrement before forwarding.
    pub ttl: u8,
    /// Opaque encrypted bytes. Nodes must not parse or log contents.
    pub encrypted_blob: Vec<u8>,
    /// Sender timestamp, Unix seconds.
    pub timestamp: u64,
    /// Ed25519 signature over `blind_relay_signing_data(self)`.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindRelayEnvelope {
    /// Builds canonical signing bytes.
    ///
    /// Layout:
    /// ```text
    /// domain || route_id(16) || next_hop(32) || ttl(1) ||
    /// timestamp_le(8) || SHA256(encrypted_blob)(32)
    /// ```
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

    /// Signs this envelope with a previous-hop node key.
    ///
    /// The caller must create the opaque `encrypted_blob` before signing.
    pub fn sign_with(mut self, keypair: &IdentityKeyPair) -> Self {
        self.signature = keypair.sign(&self.signing_data());
        self
    }

    /// Verifies signature using the previous-hop node public key supplied by
    /// the transport/auth layer.
    pub fn verify_signature_from(&self, previous_hop: &IdentityPublicKey) -> Result<(), CoreError> {
        previous_hop.verify(&self.signing_data(), &self.signature)
    }

    /// Returns true if this envelope can be forwarded one more hop.
    #[must_use]
    pub const fn can_forward(&self) -> bool {
        self.ttl > 0
    }

    /// Returns the next-hop envelope after decrementing TTL.
    ///
    /// `ttl` is covered by the signature, so the previous-hop signature is
    /// intentionally cleared after decrementing. The forwarding layer must
    /// re-sign the returned envelope with the current node key before sending
    /// it onward. This keeps each hop accountable without allowing a relay to
    /// mutate signed routing fields silently.
    #[must_use]
    pub fn decremented_ttl(&self) -> Option<Self> {
        if self.ttl == 0 {
            return None;
        }
        let mut next = self.clone();
        next.ttl = next.ttl.saturating_sub(1);
        next.signature = [0u8; 64];
        Some(next)
    }
}

// ============================================
// BlindRelayDeliveryReceipt
// ============================================

/// Terminal-signed proof for one accepted opaque blind-relay payload.
///
/// The payload commitment binds the ACK to the exact bytes delivered at the
/// terminal boundary without returning those bytes through the relay path.
/// The receipt deliberately omits sender/receiver identities, endpoints,
/// online state, mailbox state, content type, and payload size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindRelayDeliveryReceipt {
    /// Receipt schema version.
    pub version: u8,
    /// Random route correlation id supplied by the source for this route.
    pub route_id: [u8; 16],
    /// Versioned SHA-256 commitment to the terminal payload. Version 1 hashes
    /// the exact bytes. Version 2 domain-separates the payload hash by route
    /// purpose, preventing one accepted payload proof from being reinterpreted
    /// as a different terminal workload.
    pub payload_commitment: [u8; 32],
    /// Ed25519 identity of the terminal node that accepted the payload.
    pub terminal_node_id: [u8; 32],
    /// Unix timestamp when terminal acceptance completed.
    pub delivered_at: u64,
    /// Stable disposition. Version 1 accepts only `BLIND_RELAY_DELIVERY_ACCEPTED`.
    pub disposition: u8,
    /// Ed25519 signature over `signing_data()`.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindRelayDeliveryReceipt {
    /// Creates and signs a legacy version-1 terminal receipt.
    ///
    /// Kept for mixed-version compatibility and test fixtures. New terminal
    /// delivery code must use [`Self::accepted_for_purpose`].
    #[must_use]
    pub fn accepted(
        route_id: [u8; 16],
        payload_commitment: [u8; 32],
        delivered_at: u64,
        terminal: &IdentityKeyPair,
    ) -> Self {
        let mut receipt = Self {
            version: BLIND_RELAY_DELIVERY_RECEIPT_VERSION,
            route_id,
            payload_commitment,
            terminal_node_id: terminal.public_key_bytes(),
            delivered_at,
            disposition: BLIND_RELAY_DELIVERY_ACCEPTED,
            signature: [0u8; 64],
        };
        receipt.signature = terminal.sign(&receipt.signing_data());
        receipt
    }

    /// Creates a purpose-bound version-2 terminal receipt.
    ///
    /// [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] The purpose is folded into
    /// the commitment together with the opaque payload hash. It is not emitted
    /// as a low-entropy field that middle relays could inspect or classify.
    #[must_use]
    pub fn accepted_for_purpose(
        route_id: [u8; 16],
        payload: &[u8],
        purpose: OnionRoutePurpose,
        delivered_at: u64,
        terminal: &IdentityKeyPair,
    ) -> Self {
        let mut receipt = Self {
            version: BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION,
            route_id,
            payload_commitment: Self::payload_commitment_for_purpose(payload, purpose),
            terminal_node_id: terminal.public_key_bytes(),
            delivered_at,
            disposition: BLIND_RELAY_DELIVERY_ACCEPTED,
            signature: [0u8; 64],
        };
        receipt.signature = terminal.sign(&receipt.signing_data());
        receipt
    }

    /// Returns the SHA-256 commitment used by terminal and source verification.
    #[must_use]
    pub fn payload_commitment(payload: &[u8]) -> [u8; 32] {
        Sha256::digest(payload).into()
    }

    /// Returns the version-2 commitment for one opaque payload and purpose.
    ///
    /// The nested payload hash keeps the signing input fixed-size. Including
    /// the canonical purpose length makes the byte contract unambiguous if the
    /// public purpose namespace grows in future protocol versions.
    #[must_use]
    pub fn payload_commitment_for_purpose(payload: &[u8], purpose: OnionRoutePurpose) -> [u8; 32] {
        let purpose = purpose.as_str().as_bytes();
        let payload_hash = Self::payload_commitment(payload);
        let mut hasher = Sha256::new();
        hasher.update(BLIND_RELAY_PURPOSE_COMMITMENT_DOMAIN);
        hasher.update([u8::try_from(purpose.len()).expect("route purpose must fit in u8")]);
        hasher.update(purpose);
        hasher.update(payload_hash);
        hasher.finalize().into()
    }

    /// Builds canonical domain-separated receipt signing bytes.
    #[must_use]
    pub fn signing_data(&self) -> Vec<u8> {
        let signing_domain = match self.version {
            BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION => {
                BLIND_RELAY_DELIVERY_RECEIPT_V2_SIGNING_DOMAIN
            }
            _ => BLIND_RELAY_DELIVERY_RECEIPT_V1_SIGNING_DOMAIN,
        };
        let mut data = Vec::with_capacity(signing_domain.len() + 1 + 16 + 32 + 32 + 8 + 1);
        data.extend_from_slice(signing_domain);
        data.push(self.version);
        data.extend_from_slice(&self.route_id);
        data.extend_from_slice(&self.payload_commitment);
        data.extend_from_slice(&self.terminal_node_id);
        data.extend_from_slice(&self.delivered_at.to_le_bytes());
        data.push(self.disposition);
        data
    }

    /// Verifies version, disposition, and the terminal Ed25519 signature.
    pub fn verify_signature(&self) -> Result<(), CoreError> {
        if self.version != BLIND_RELAY_DELIVERY_RECEIPT_VERSION
            && self.version != BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION
        {
            return Err(CoreError::malformed(
                "blind relay delivery receipt: unsupported version",
            ));
        }
        if self.disposition != BLIND_RELAY_DELIVERY_ACCEPTED {
            return Err(CoreError::malformed(
                "blind relay delivery receipt: unsupported disposition",
            ));
        }
        let terminal = IdentityPublicKey::from_bytes(&self.terminal_node_id)?;
        terminal.verify(&self.signing_data(), &self.signature)
    }

    /// Verifies the signature and binds the receipt to the source's expected route.
    pub fn verify_expected(
        &self,
        route_id: &[u8; 16],
        payload_commitment: &[u8; 32],
        terminal_node_id: &[u8; 32],
    ) -> Result<(), CoreError> {
        self.verify_signature()?;
        if &self.route_id != route_id
            || &self.payload_commitment != payload_commitment
            || &self.terminal_node_id != terminal_node_id
        {
            return Err(CoreError::malformed(
                "blind relay delivery receipt: route binding mismatch",
            ));
        }
        Ok(())
    }

    /// Verifies a version-2 receipt against the exact payload and route purpose.
    ///
    /// Legacy version-1 receipts deliberately fail this stronger check: their
    /// signatures remain valid for transport compatibility, but they do not
    /// provide workload-domain separation.
    pub fn verify_expected_for_purpose(
        &self,
        route_id: &[u8; 16],
        payload: &[u8],
        purpose: OnionRoutePurpose,
        terminal_node_id: &[u8; 32],
    ) -> Result<(), CoreError> {
        if self.version != BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION {
            return Err(CoreError::malformed(
                "blind relay delivery receipt: route purpose is not bound",
            ));
        }
        self.verify_expected(
            route_id,
            &Self::payload_commitment_for_purpose(payload, purpose),
            terminal_node_id,
        )
    }
}

// ============================================
// BlindRelayFailureReceipt
// ============================================

/// Immediate-hop signature over one coarse blind-relay failure response.
///
/// [SIGNED-FAILURE-RECEIPT 2026-08-11 by Codex] This receipt authenticates
/// only the node that returned the response to its direct previous hop. It is
/// deliberately not a blame certificate for a deeper onion hop: exposing that
/// identity would let upstream relays reconstruct private route topology.
/// Request and reason commitments keep the signed surface fixed-size and bind
/// it to the exact opaque request without returning payload-derived bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindRelayFailureReceipt {
    /// Failure receipt schema version.
    pub version: u8,
    /// Random route correlation id from the exact failed request.
    pub route_id: [u8; 16],
    /// Domain-separated commitment to the request envelope's signed fields.
    pub request_commitment: [u8; 32],
    /// Ed25519 identity of the immediate node returning this failure.
    pub responder_node_id: [u8; 32],
    /// Unix timestamp when this node completed the failed attempt.
    pub failed_at: u64,
    /// Domain-separated commitment to the coarse public reason bucket.
    pub reason_commitment: [u8; 32],
    /// Ed25519 signature over [`Self::signing_data`].
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindRelayFailureReceipt {
    /// Creates and signs a failure receipt bound to one precomputed request.
    ///
    /// Callers can compute `request_commitment` before moving a large opaque
    /// request into async processing, avoiding a payload clone on every relay.
    #[must_use]
    pub fn failed(
        route_id: [u8; 16],
        request_commitment: [u8; 32],
        reason: &str,
        failed_at: u64,
        responder: &IdentityKeyPair,
    ) -> Self {
        let mut receipt = Self {
            version: BLIND_RELAY_FAILURE_RECEIPT_VERSION,
            route_id,
            request_commitment,
            responder_node_id: responder.public_key_bytes(),
            failed_at,
            reason_commitment: Self::reason_commitment(reason),
            signature: [0u8; 64],
        };
        receipt.signature = responder.sign(&receipt.signing_data());
        receipt
    }

    /// Commits to the exact route fields and opaque payload hash already
    /// covered by the previous hop's envelope signature.
    #[must_use]
    pub fn request_commitment(envelope: &BlindRelayEnvelope) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(BLIND_RELAY_FAILURE_REQUEST_COMMITMENT_DOMAIN);
        hasher.update(envelope.signing_data());
        hasher.finalize().into()
    }

    /// Commits to one public coarse reason without making variable-length
    /// strings part of the failure receipt's canonical signing layout.
    #[must_use]
    pub fn reason_commitment(reason: &str) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(BLIND_RELAY_FAILURE_REASON_COMMITMENT_DOMAIN);
        hasher.update(reason.as_bytes());
        hasher.finalize().into()
    }

    /// Builds canonical, domain-separated failure receipt signing bytes.
    #[must_use]
    pub fn signing_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(
            BLIND_RELAY_FAILURE_RECEIPT_SIGNING_DOMAIN.len() + 1 + 16 + 32 + 32 + 8 + 32,
        );
        data.extend_from_slice(BLIND_RELAY_FAILURE_RECEIPT_SIGNING_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.route_id);
        data.extend_from_slice(&self.request_commitment);
        data.extend_from_slice(&self.responder_node_id);
        data.extend_from_slice(&self.failed_at.to_le_bytes());
        data.extend_from_slice(&self.reason_commitment);
        data
    }

    /// Verifies the schema version and responder Ed25519 signature.
    pub fn verify_signature(&self) -> Result<(), CoreError> {
        if self.version != BLIND_RELAY_FAILURE_RECEIPT_VERSION {
            return Err(CoreError::malformed(
                "blind relay failure receipt: unsupported version",
            ));
        }
        let responder = IdentityPublicKey::from_bytes(&self.responder_node_id)?;
        responder.verify(&self.signing_data(), &self.signature)
    }

    /// Verifies signature, request, reason, and immediate responder binding.
    pub fn verify_expected(
        &self,
        envelope: &BlindRelayEnvelope,
        reason: &str,
        responder_node_id: &[u8; 32],
    ) -> Result<(), CoreError> {
        self.verify_signature()?;
        if self.route_id != envelope.route_id
            || self.request_commitment != Self::request_commitment(envelope)
            || &self.responder_node_id != responder_node_id
            || self.reason_commitment != Self::reason_commitment(reason)
        {
            return Err(CoreError::malformed(
                "blind relay failure receipt: response binding mismatch",
            ));
        }
        Ok(())
    }
}

// ============================================
// CustodyAuditAnchorV1
// ============================================

/// Portable node-signed commitment to one relay-custody audit checkpoint.
///
/// [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] The producer computes
/// `anchor_digest` from private checkpoint authentication state, then discards
/// that private state at this protocol boundary. External retainers receive a
/// fixed-size opaque commitment, monotonic generation, and coarse aggregate
/// archived record/byte totals. Comparing already retained evidence can reveal
/// rollback or equivocation, but this producer signature alone is not an
/// independent witness receipt and exposes no message- or user-level data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustodyAuditAnchorV1 {
    /// Anchor schema version.
    pub version: u8,
    /// Ed25519 identity of the node that owns the private audit.
    pub producer_node_id: [u8; 32],
    /// Positive monotonic checkpoint generation.
    pub checkpoint_generation: u64,
    /// Aggregate maintenance records covered by this checkpoint.
    pub archived_record_count: u64,
    /// Aggregate authenticated audit bytes covered by this checkpoint.
    pub archived_bytes: u64,
    /// Domain-separated opaque digest of the private checkpoint state.
    pub anchor_digest: [u8; 32],
    /// Producer signature over [`Self::signing_data`].
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl CustodyAuditAnchorV1 {
    /// Creates and signs one structurally valid portable anchor.
    ///
    /// # Errors
    /// Returns [`CoreError::MalformedMessage`] for zero/sentinel state or an
    /// impossible generation-to-record relation.
    pub fn signed(
        checkpoint_generation: u64,
        archived_record_count: u64,
        archived_bytes: u64,
        anchor_digest: [u8; 32],
        producer: &IdentityKeyPair,
    ) -> Result<Self, CoreError> {
        let mut anchor = Self {
            version: CUSTODY_AUDIT_ANCHOR_VERSION,
            producer_node_id: producer.public_key_bytes(),
            checkpoint_generation,
            archived_record_count,
            archived_bytes,
            anchor_digest,
            signature: [0u8; 64],
        };
        anchor.validate_structure()?;
        anchor.signature = producer.sign(&anchor.signing_data());
        Ok(anchor)
    }

    /// Builds the canonical fixed-width, domain-separated signing frame.
    #[must_use]
    pub fn signing_data(&self) -> Vec<u8> {
        let mut data =
            Vec::with_capacity(CUSTODY_AUDIT_ANCHOR_SIGNING_DOMAIN.len() + 1 + 32 + (8 * 3) + 32);
        data.extend_from_slice(CUSTODY_AUDIT_ANCHOR_SIGNING_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.producer_node_id);
        data.extend_from_slice(&self.checkpoint_generation.to_le_bytes());
        data.extend_from_slice(&self.archived_record_count.to_le_bytes());
        data.extend_from_slice(&self.archived_bytes.to_le_bytes());
        data.extend_from_slice(&self.anchor_digest);
        data
    }

    fn validate_structure(&self) -> Result<(), CoreError> {
        if self.version != CUSTODY_AUDIT_ANCHOR_VERSION
            || self.checkpoint_generation == 0
            || self.archived_record_count == 0
            || self.archived_bytes == 0
            || self.checkpoint_generation > self.archived_record_count
            || self.anchor_digest == [0u8; 32]
        {
            return Err(CoreError::malformed(
                "custody audit anchor: invalid structural state",
            ));
        }
        Ok(())
    }

    /// Verifies structural constraints and the embedded producer signature.
    ///
    /// # Errors
    /// Returns a malformed/signature error when any covered field is invalid.
    pub fn verify_signature(&self) -> Result<(), CoreError> {
        self.validate_structure()?;
        IdentityPublicKey::from_bytes(&self.producer_node_id)?
            .verify(&self.signing_data(), &self.signature)
    }

    /// Verifies signature plus an explicit producer pin and rollback floor.
    ///
    /// # Errors
    /// Returns a malformed error when identity or checkpoint generation does
    /// not satisfy the verifier's local trust policy.
    pub fn verify_expected(
        &self,
        expected_producer: &[u8; 32],
        minimum_checkpoint_generation: u64,
    ) -> Result<(), CoreError> {
        self.verify_signature()?;
        if minimum_checkpoint_generation == 0
            || &self.producer_node_id != expected_producer
            || self.checkpoint_generation < minimum_checkpoint_generation
        {
            return Err(CoreError::malformed(
                "custody audit anchor: trust policy mismatch",
            ));
        }
        Ok(())
    }
}

/// Encodes one custody audit anchor with a canonical bounded bincode frame.
///
/// # Errors
/// Returns [`CoreError::MalformedMessage`] when bounded encoding fails.
pub fn encode_custody_audit_anchor(anchor: &CustodyAuditAnchorV1) -> Result<Vec<u8>, CoreError> {
    encode_bincode_bounded(anchor, MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES as u64)
        .map_err(|error| CoreError::malformed(format!("custody audit anchor encode: {error}")))
}

/// Returns the SHA-256 of the canonical bounded custody anchor frame.
///
/// [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Network request signatures,
/// receipt bindings, and operator pins must all derive the same digest inside
/// the protocol crate instead of trusting caller-specific serialization.
///
/// # Errors
/// Returns [`CoreError::MalformedMessage`] when canonical encoding fails.
pub fn custody_audit_anchor_frame_sha256(
    anchor: &CustodyAuditAnchorV1,
) -> Result<[u8; 32], CoreError> {
    Ok(Sha256::digest(encode_custody_audit_anchor(anchor)?).into())
}

/// Decodes one complete canonical custody audit anchor frame.
///
/// # Errors
/// Returns [`CoreError::MalformedMessage`] for malformed, padded, or oversized
/// input. Signature and trust-policy verification remain explicit caller steps.
pub fn decode_custody_audit_anchor(bytes: &[u8]) -> Result<CustodyAuditAnchorV1, CoreError> {
    decode_bincode_bounded(
        bytes,
        MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES as u64,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| CoreError::malformed(format!("custody audit anchor decode: {error}")))
}

// ============================================
// CustodyAuditWitnessReceiptV1
// ============================================

/// Independent node's signed durable decision for one exact custody anchor.
///
/// [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] The witness retains one
/// high-water generation and exact producer-frame SHA-256 per producer. The
/// receipt signs both the requested and retained states so rollback,
/// same-generation equivocation, and generation gaps produce portable evidence
/// without disclosing the producer's private audit HMAC or custody contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustodyAuditWitnessReceiptV1 {
    /// Receipt schema version.
    pub version: u8,
    /// Producer identity copied from the verified anchor.
    pub producer_node_id: [u8; 32],
    /// Requested producer checkpoint generation.
    pub requested_checkpoint_generation: u64,
    /// SHA-256 of the exact canonical producer anchor frame.
    pub requested_frame_sha256: [u8; 32],
    /// Independent witness Ed25519 identity.
    pub witness_node_id: [u8; 32],
    /// Witness wall-clock time after its durable decision, in Unix seconds.
    pub observed_at: u64,
    /// Witness's durable high-water generation after evaluating the request.
    pub retained_checkpoint_generation: u64,
    /// Exact producer-frame SHA-256 retained at the high-water generation.
    pub retained_frame_sha256: [u8; 32],
    /// One `CUSTODY_AUDIT_WITNESS_*_V1` decision bucket.
    pub outcome: u8,
    /// Witness signature over [`Self::signing_data`].
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl CustodyAuditWitnessReceiptV1 {
    /// Creates one independently signed witness decision.
    ///
    /// # Errors
    /// Returns [`CoreError::MalformedMessage`] when the decision is internally
    /// inconsistent, uses sentinel values, or is self-witnessed.
    #[allow(clippy::too_many_arguments)]
    pub fn signed(
        producer_node_id: [u8; 32],
        requested_checkpoint_generation: u64,
        requested_frame_sha256: [u8; 32],
        observed_at: u64,
        retained_checkpoint_generation: u64,
        retained_frame_sha256: [u8; 32],
        outcome: u8,
        witness: &IdentityKeyPair,
    ) -> Result<Self, CoreError> {
        let mut receipt = Self {
            version: CUSTODY_AUDIT_WITNESS_RECEIPT_VERSION,
            producer_node_id,
            requested_checkpoint_generation,
            requested_frame_sha256,
            witness_node_id: witness.public_key_bytes(),
            observed_at,
            retained_checkpoint_generation,
            retained_frame_sha256,
            outcome,
            signature: [0u8; 64],
        };
        receipt.validate_structure()?;
        receipt.signature = witness.sign(&receipt.signing_data());
        Ok(receipt)
    }

    /// Returns whether this receipt proves the witness retained this request.
    #[must_use]
    pub const fn accepted(&self) -> bool {
        matches!(
            self.outcome,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1 | CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1
        )
    }

    /// Builds the canonical fixed-width domain-separated witness signing frame.
    #[must_use]
    pub fn signing_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(
            CUSTODY_AUDIT_WITNESS_RECEIPT_SIGNING_DOMAIN.len() + 1 + (32 * 4) + (8 * 3) + 1,
        );
        data.extend_from_slice(CUSTODY_AUDIT_WITNESS_RECEIPT_SIGNING_DOMAIN);
        data.push(self.version);
        data.extend_from_slice(&self.producer_node_id);
        data.extend_from_slice(&self.requested_checkpoint_generation.to_le_bytes());
        data.extend_from_slice(&self.requested_frame_sha256);
        data.extend_from_slice(&self.witness_node_id);
        data.extend_from_slice(&self.observed_at.to_le_bytes());
        data.extend_from_slice(&self.retained_checkpoint_generation.to_le_bytes());
        data.extend_from_slice(&self.retained_frame_sha256);
        data.push(self.outcome);
        data
    }

    fn validate_structure(&self) -> Result<(), CoreError> {
        let request_matches_retained = self.requested_checkpoint_generation
            == self.retained_checkpoint_generation
            && self.requested_frame_sha256 == self.retained_frame_sha256;
        let outcome_is_consistent = match self.outcome {
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1 | CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1 => {
                request_matches_retained
            }
            CUSTODY_AUDIT_WITNESS_STALE_V1 => {
                self.requested_checkpoint_generation < self.retained_checkpoint_generation
            }
            CUSTODY_AUDIT_WITNESS_CONFLICT_V1 => {
                self.requested_checkpoint_generation == self.retained_checkpoint_generation
                    && self.requested_frame_sha256 != self.retained_frame_sha256
            }
            CUSTODY_AUDIT_WITNESS_GAP_V1 => self
                .retained_checkpoint_generation
                .checked_add(1)
                .is_some_and(|next| self.requested_checkpoint_generation > next),
            _ => false,
        };
        if self.version != CUSTODY_AUDIT_WITNESS_RECEIPT_VERSION
            || self.producer_node_id == [0u8; 32]
            || self.witness_node_id == [0u8; 32]
            || self.producer_node_id == self.witness_node_id
            || self.requested_checkpoint_generation == 0
            || self.retained_checkpoint_generation == 0
            || self.requested_frame_sha256 == [0u8; 32]
            || self.retained_frame_sha256 == [0u8; 32]
            || self.observed_at == 0
            || !outcome_is_consistent
        {
            return Err(CoreError::malformed(
                "custody audit witness receipt: invalid structural state",
            ));
        }
        IdentityPublicKey::from_bytes(&self.producer_node_id)?;
        Ok(())
    }

    /// Verifies structural constraints and the independent witness signature.
    ///
    /// # Errors
    /// Returns a malformed/signature error for tampered or inconsistent state.
    pub fn verify_signature(&self) -> Result<(), CoreError> {
        self.validate_structure()?;
        IdentityPublicKey::from_bytes(&self.witness_node_id)?
            .verify(&self.signing_data(), &self.signature)
    }

    /// Verifies any signed witness decision against one exact anchor and pins.
    ///
    /// Negative `stale`, `conflict`, and `gap` decisions remain valid portable
    /// evidence. Call [`Self::verify_accepted_for_anchor`] when policy requires
    /// the witness to have retained the requested anchor.
    ///
    /// # Errors
    /// Returns a malformed/signature error for a wrong producer/witness,
    /// rollback below the verifier floor, or an anchor digest that does not
    /// match the request signed by the witness.
    pub fn verify_for_anchor(
        &self,
        anchor: &CustodyAuditAnchorV1,
        anchor_frame_sha256: &[u8; 32],
        expected_producer: &[u8; 32],
        expected_witness: &[u8; 32],
        minimum_checkpoint_generation: u64,
    ) -> Result<(), CoreError> {
        self.verify_signature()?;
        anchor.verify_expected(expected_producer, minimum_checkpoint_generation)?;
        // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Recompute the canonical
        // frame digest inside the verifier. Future network callers must not be
        // able to satisfy exact-frame binding by supplying their own digest.
        let canonical_anchor_sha256 = custody_audit_anchor_frame_sha256(anchor)?;
        if &self.producer_node_id != expected_producer
            || &self.witness_node_id != expected_witness
            || self.requested_checkpoint_generation != anchor.checkpoint_generation
            || &canonical_anchor_sha256 != anchor_frame_sha256
            || &self.requested_frame_sha256 != anchor_frame_sha256
        {
            return Err(CoreError::malformed(
                "custody audit witness receipt: trust policy mismatch",
            ));
        }
        Ok(())
    }

    /// Verifies an accepted receipt against one exact producer anchor and pins.
    ///
    /// # Errors
    /// Returns a malformed/signature error for a negative witness decision,
    /// wrong producer/witness, rollback below the verifier floor, or an anchor
    /// frame digest that is not exactly the request signed by the witness.
    pub fn verify_accepted_for_anchor(
        &self,
        anchor: &CustodyAuditAnchorV1,
        anchor_frame_sha256: &[u8; 32],
        expected_producer: &[u8; 32],
        expected_witness: &[u8; 32],
        minimum_checkpoint_generation: u64,
    ) -> Result<(), CoreError> {
        self.verify_for_anchor(
            anchor,
            anchor_frame_sha256,
            expected_producer,
            expected_witness,
            minimum_checkpoint_generation,
        )?;
        if !self.accepted() {
            return Err(CoreError::malformed(
                "custody audit witness receipt: witness did not accept anchor",
            ));
        }
        Ok(())
    }
}

/// Encodes one custody audit witness receipt as a canonical bounded frame.
///
/// # Errors
/// Returns [`CoreError::MalformedMessage`] when bounded encoding fails.
pub fn encode_custody_audit_witness_receipt(
    receipt: &CustodyAuditWitnessReceiptV1,
) -> Result<Vec<u8>, CoreError> {
    encode_bincode_bounded(
        receipt,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES as u64,
    )
    .map_err(|error| CoreError::malformed(format!("custody audit witness receipt encode: {error}")))
}

/// Returns the SHA-256 of the canonical bounded witness receipt frame.
///
/// # Errors
/// Returns [`CoreError::MalformedMessage`] when canonical encoding fails.
pub fn custody_audit_witness_receipt_frame_sha256(
    receipt: &CustodyAuditWitnessReceiptV1,
) -> Result<[u8; 32], CoreError> {
    Ok(Sha256::digest(encode_custody_audit_witness_receipt(receipt)?).into())
}

/// Decodes one complete canonical custody audit witness receipt frame.
///
/// # Errors
/// Returns [`CoreError::MalformedMessage`] for malformed, padded, or oversized
/// input. Signature and trust-policy verification remain explicit caller steps.
pub fn decode_custody_audit_witness_receipt(
    bytes: &[u8],
) -> Result<CustodyAuditWitnessReceiptV1, CoreError> {
    decode_bincode_bounded(
        bytes,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES as u64,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| CoreError::malformed(format!("custody audit witness receipt decode: {error}")))
}

/// Encodes a blind relay envelope with a bounded bincode cap.
///
/// # Errors
/// Returns `CoreError::MessageTooLarge` when the opaque blob exceeds its
/// protocol class, or `CoreError::MalformedMessage` when bounded serialization
/// fails.
pub fn encode_blind_relay_envelope(envelope: &BlindRelayEnvelope) -> Result<Vec<u8>, CoreError> {
    if envelope.encrypted_blob.len() > MAX_BLIND_RELAY_BLOB_BYTES {
        return Err(CoreError::MessageTooLarge {
            max: MAX_BLIND_RELAY_BLOB_BYTES,
            actual: envelope.encrypted_blob.len(),
        });
    }
    encode_bincode_bounded(envelope, MAX_BLIND_RELAY_ENVELOPE_BYTES)
        .map_err(|err| CoreError::malformed(format!("blind relay envelope encode: {err}")))
}

/// Decodes a blind relay envelope with size bounds and blob cap checks.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for malformed, non-canonical, or
/// oversized frames, and `CoreError::MessageTooLarge` when the decoded opaque
/// blob exceeds its protocol class.
pub fn decode_blind_relay_envelope(bytes: &[u8]) -> Result<BlindRelayEnvelope, CoreError> {
    let envelope: BlindRelayEnvelope = decode_bincode_bounded(
        bytes,
        MAX_BLIND_RELAY_ENVELOPE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|err| CoreError::malformed(format!("blind relay envelope decode: {err}")))?;
    if envelope.encrypted_blob.len() > MAX_BLIND_RELAY_BLOB_BYTES {
        return Err(CoreError::MessageTooLarge {
            max: MAX_BLIND_RELAY_BLOB_BYTES,
            actual: envelope.encrypted_blob.len(),
        });
    }
    Ok(envelope)
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
/// Returns `bincode::Error` if serialisation fails or the encoded envelope
/// exceeds the 128 KiB protocol ceiling.
pub fn encode_envelope(envelope: &ChatEnvelope) -> Result<Vec<u8>, bincode::Error> {
    encode_bincode_bounded(envelope, MAX_ENVELOPE_BYTES)
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
    decode_bincode_bounded(bytes, MAX_ENVELOPE_BYTES, TrailingBytesPolicy::Allow)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::IdentityKeyPair;
    use bincode::Options;

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

    // [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] These tests lock the
    // canonical fixed-size frame and explicit rollback-floor semantics before
    // any network witness endpoint is allowed to consume the primitive.
    #[test]
    fn custody_audit_anchor_roundtrips_canonically_and_verifies() {
        let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).expect("producer identity");
        let anchor =
            CustodyAuditAnchorV1::signed(3, 65_538, 128 * 1024 * 1024, [0x42; 32], &producer)
                .expect("sign custody audit anchor");
        anchor
            .verify_expected(&producer.public_key_bytes(), 3)
            .expect("verify exact producer and generation");

        let encoded = encode_custody_audit_anchor(&anchor).expect("encode anchor");
        assert_eq!(encoded.len(), 153);
        assert!(encoded.len() <= MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES);
        let decoded = decode_custody_audit_anchor(&encoded).expect("decode anchor");
        assert_eq!(decoded, anchor);
        assert_eq!(
            encode_custody_audit_anchor(&decoded).expect("re-encode anchor"),
            encoded
        );
    }

    #[test]
    fn custody_audit_anchor_rejects_tamper_padding_and_rollback_floor() {
        let producer = IdentityKeyPair::from_bytes(&[0x72; 32]).expect("producer identity");
        let other = IdentityKeyPair::from_bytes(&[0x73; 32]).expect("other identity");
        let anchor =
            CustodyAuditAnchorV1::signed(2, 65_537, 64 * 1024 * 1024, [0x43; 32], &producer)
                .expect("sign custody audit anchor");

        assert!(anchor
            .verify_expected(&producer.public_key_bytes(), 3)
            .is_err());
        assert!(anchor
            .verify_expected(&other.public_key_bytes(), 1)
            .is_err());
        let mut tampered = anchor.clone();
        tampered.archived_bytes += 1;
        assert!(tampered.verify_signature().is_err());

        let mut padded = encode_custody_audit_anchor(&anchor).expect("encode anchor");
        padded.push(0);
        assert!(decode_custody_audit_anchor(&padded).is_err());
        assert!(CustodyAuditAnchorV1::signed(0, 1, 1, [0x44; 32], &producer).is_err());
    }

    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Lock exact-frame binding,
    // independent identity, and negative-outcome consistency before the
    // receipt is accepted by any host-local or network witness workflow.
    #[test]
    fn custody_audit_witness_receipt_roundtrips_and_binds_exact_anchor() {
        let producer = IdentityKeyPair::from_bytes(&[0x74; 32]).expect("producer identity");
        let witness = IdentityKeyPair::from_bytes(&[0x75; 32]).expect("witness identity");
        let anchor =
            CustodyAuditAnchorV1::signed(4, 65_540, 256 * 1024 * 1024, [0x45; 32], &producer)
                .expect("sign custody anchor");
        let anchor_frame = encode_custody_audit_anchor(&anchor).expect("encode custody anchor");
        let anchor_sha256: [u8; 32] = Sha256::digest(&anchor_frame).into();
        let receipt = CustodyAuditWitnessReceiptV1::signed(
            producer.public_key_bytes(),
            4,
            anchor_sha256,
            1_787_200_001,
            4,
            anchor_sha256,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .expect("sign custody witness receipt");
        receipt
            .verify_accepted_for_anchor(
                &anchor,
                &anchor_sha256,
                &producer.public_key_bytes(),
                &witness.public_key_bytes(),
                4,
            )
            .expect("verify custody witness receipt");

        let caller_controlled_digest = [0x99; 32];
        let caller_controlled_receipt = CustodyAuditWitnessReceiptV1::signed(
            producer.public_key_bytes(),
            4,
            caller_controlled_digest,
            1_787_200_002,
            4,
            caller_controlled_digest,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .expect("sign caller-controlled digest receipt");
        assert!(caller_controlled_receipt
            .verify_accepted_for_anchor(
                &anchor,
                &caller_controlled_digest,
                &producer.public_key_bytes(),
                &witness.public_key_bytes(),
                4,
            )
            .is_err());

        let encoded = encode_custody_audit_witness_receipt(&receipt).expect("encode receipt");
        assert_eq!(encoded.len(), 218);
        assert!(encoded.len() <= MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES);
        let decoded =
            decode_custody_audit_witness_receipt(&encoded).expect("decode custody witness receipt");
        assert_eq!(decoded, receipt);
        assert_eq!(
            encode_custody_audit_witness_receipt(&decoded).expect("re-encode receipt"),
            encoded
        );
    }

    #[test]
    fn custody_audit_witness_receipt_rejects_self_witness_and_invalid_outcomes() {
        let producer = IdentityKeyPair::from_bytes(&[0x76; 32]).expect("producer identity");
        let witness = IdentityKeyPair::from_bytes(&[0x77; 32]).expect("witness identity");
        let producer_id = producer.public_key_bytes();
        let requested_sha = [0x46; 32];
        let retained_sha = [0x47; 32];

        assert!(CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            3,
            requested_sha,
            1,
            3,
            requested_sha,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &producer,
        )
        .is_err());
        assert!(CustodyAuditWitnessReceiptV1::signed(
            producer_id,
            3,
            requested_sha,
            1,
            3,
            retained_sha,
            CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
            &witness,
        )
        .is_err());

        for (outcome, requested_generation, retained_generation, requested, retained) in [
            (
                CUSTODY_AUDIT_WITNESS_STALE_V1,
                3,
                4,
                requested_sha,
                retained_sha,
            ),
            (
                CUSTODY_AUDIT_WITNESS_CONFLICT_V1,
                4,
                4,
                requested_sha,
                retained_sha,
            ),
            (
                CUSTODY_AUDIT_WITNESS_GAP_V1,
                6,
                4,
                requested_sha,
                retained_sha,
            ),
        ] {
            let receipt = CustodyAuditWitnessReceiptV1::signed(
                producer_id,
                requested_generation,
                requested,
                2,
                retained_generation,
                retained,
                outcome,
                &witness,
            )
            .expect("sign consistent negative outcome");
            receipt.verify_signature().expect("verify negative receipt");
            assert!(!receipt.accepted());
        }

        let mut padded = encode_custody_audit_witness_receipt(
            &CustodyAuditWitnessReceiptV1::signed(
                producer_id,
                4,
                requested_sha,
                3,
                4,
                requested_sha,
                CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1,
                &witness,
            )
            .expect("sign idempotent receipt"),
        )
        .expect("encode receipt");
        padded.push(0);
        assert!(decode_custody_audit_witness_receipt(&padded).is_err());
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
        for ct in [
            ChatContentType::Text,
            ChatContentType::Media,
            ChatContentType::System,
        ] {
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
        assert!(
            env.verify_signature().is_ok(),
            "Valid signature must verify"
        );
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
        assert!(
            env.verify_signature().is_err(),
            "Tampered receiver must fail"
        );
    }

    #[test]
    fn test_envelope_tampered_ciphertext_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        env.ciphertext[0] ^= 0xFF;
        assert!(
            env.verify_signature().is_err(),
            "Tampered ciphertext must fail"
        );
    }

    #[test]
    fn test_envelope_tampered_timestamp_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        env.timestamp += 1;
        assert!(
            env.verify_signature().is_err(),
            "Tampered timestamp must fail"
        );
    }

    #[test]
    fn test_envelope_tampered_content_type_rejected() {
        let kp = IdentityKeyPair::generate();
        let mut env = make_signed_envelope(&kp);
        // Flip Text → Media
        env.content_type = ChatContentType::Media;
        assert!(
            env.verify_signature().is_err(),
            "Tampered content_type must fail"
        );
    }

    #[test]
    fn test_envelope_bincode_roundtrip() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);

        let bytes = encode_envelope(&env).expect("encode");
        assert_eq!(
            bytes,
            bincode::serialize(&env).expect("legacy wire encoding"),
            "bounded encoding must not alter existing ChatEnvelope wire bytes"
        );
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
    fn test_envelope_codec_preserves_small_trailing_compatibility() {
        let kp = IdentityKeyPair::generate();
        let env = make_signed_envelope(&kp);
        let mut bytes = encode_envelope(&env).expect("encode");
        bytes.extend_from_slice(&[0xAA, 0xBB]);

        let decoded = decode_envelope(&bytes).expect("decode with legacy trailing bytes");
        assert_eq!(decoded.message_id, env.message_id);
        assert_eq!(decoded.ciphertext, env.ciphertext);
    }

    #[test]
    fn test_envelope_codec_rejects_oversized_output_and_padded_input() {
        let kp = IdentityKeyPair::generate();
        let mut oversized = make_signed_envelope(&kp);
        oversized.ciphertext = vec![0xCC; MAX_ENVELOPE_BYTES as usize];
        assert!(
            encode_envelope(&oversized).is_err(),
            "sender must not create an envelope that receivers reject"
        );

        let env = make_signed_envelope(&kp);
        let mut padded = bincode::serialize(&env).expect("legacy wire encoding");
        padded.resize(MAX_ENVELOPE_BYTES as usize + 1, 0);
        assert!(
            decode_envelope(&padded).is_err(),
            "ignored trailing padding must not bypass the complete input ceiling"
        );
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

    // ── BlindRelayEnvelope ──

    fn make_blind_envelope(kp: &IdentityKeyPair) -> BlindRelayEnvelope {
        BlindRelayEnvelope {
            route_id: [0xA1u8; 16],
            next_hop: [0xB2u8; 32],
            ttl: 3,
            encrypted_blob: b"opaque encrypted relay payload".to_vec(),
            timestamp: 1_700_000_100,
            signature: [0u8; 64],
        }
        .sign_with(kp)
    }

    #[test]
    fn test_blind_relay_envelope_signature_ok() {
        let kp = IdentityKeyPair::generate();
        let pk = IdentityPublicKey::from_bytes(&kp.public_key_bytes()).unwrap();
        let env = make_blind_envelope(&kp);

        assert!(env.verify_signature_from(&pk).is_ok());
    }

    #[test]
    fn test_blind_relay_envelope_tampered_blob_rejected() {
        let kp = IdentityKeyPair::generate();
        let pk = IdentityPublicKey::from_bytes(&kp.public_key_bytes()).unwrap();
        let mut env = make_blind_envelope(&kp);
        env.encrypted_blob[0] ^= 0xFF;

        assert!(env.verify_signature_from(&pk).is_err());
    }

    #[test]
    fn test_blind_relay_envelope_tampered_ttl_rejected() {
        let kp = IdentityKeyPair::generate();
        let pk = IdentityPublicKey::from_bytes(&kp.public_key_bytes()).unwrap();
        let mut env = make_blind_envelope(&kp);
        env.ttl = env.ttl.saturating_sub(1);

        assert!(env.verify_signature_from(&pk).is_err());
    }

    #[test]
    fn test_blind_relay_envelope_decrements_ttl_without_parsing_blob() {
        let kp = IdentityKeyPair::generate();
        let pk = IdentityPublicKey::from_bytes(&kp.public_key_bytes()).unwrap();
        let env = make_blind_envelope(&kp);
        let next = env.decremented_ttl().unwrap();

        assert_eq!(next.ttl, 2);
        assert_eq!(next.encrypted_blob, env.encrypted_blob);
        assert_eq!(next.signature, [0u8; 64]);
        assert!(next.verify_signature_from(&pk).is_err());

        let resigned = next.sign_with(&kp);
        assert!(resigned.verify_signature_from(&pk).is_ok());
    }

    #[test]
    fn test_blind_relay_envelope_ttl_zero_not_forwardable() {
        let env = BlindRelayEnvelope {
            route_id: [1u8; 16],
            next_hop: [2u8; 32],
            ttl: 0,
            encrypted_blob: vec![3u8; 16],
            timestamp: 1_700_000_100,
            signature: [0u8; 64],
        };

        assert!(!env.can_forward());
        assert!(env.decremented_ttl().is_none());
    }

    #[test]
    fn test_blind_relay_envelope_bounded_codec() {
        let kp = IdentityKeyPair::generate();
        let env = make_blind_envelope(&kp);

        let bytes = encode_blind_relay_envelope(&env).unwrap();
        assert_eq!(
            bytes,
            bincode::options()
                .with_fixint_encoding()
                .serialize(&env)
                .expect("legacy wire encoding"),
            "shared bounded codec must preserve blind-relay wire bytes"
        );
        let decoded = decode_blind_relay_envelope(&bytes).unwrap();

        assert_eq!(decoded, env);
    }

    #[test]
    fn test_blind_relay_rejects_trailing_bytes() {
        let kp = IdentityKeyPair::generate();
        let env = make_blind_envelope(&kp);
        let mut bytes = encode_blind_relay_envelope(&env).unwrap();
        bytes.push(0xAA);

        assert!(
            decode_blind_relay_envelope(&bytes).is_err(),
            "canonical node-peer frames must reject trailing bytes"
        );
    }

    #[test]
    fn test_blind_relay_rejects_oversized_blob() {
        let kp = IdentityKeyPair::generate();
        let env = BlindRelayEnvelope {
            route_id: [1u8; 16],
            next_hop: [2u8; 32],
            ttl: 1,
            encrypted_blob: vec![7u8; MAX_BLIND_RELAY_BLOB_BYTES + 1],
            timestamp: 1_700_000_100,
            signature: [0u8; 64],
        }
        .sign_with(&kp);

        assert!(encode_blind_relay_envelope(&env).is_err());
    }

    #[test]
    fn test_blind_relay_delivery_receipt_binds_terminal_route_and_payload() {
        let terminal = IdentityKeyPair::generate();
        let route_id = [0x31u8; 16];
        let commitment = BlindRelayDeliveryReceipt::payload_commitment(b"opaque payload");
        let receipt =
            BlindRelayDeliveryReceipt::accepted(route_id, commitment, 1_700_000_200, &terminal);

        assert!(receipt.verify_signature().is_ok());
        assert!(receipt
            .verify_expected(&route_id, &commitment, &terminal.public_key_bytes())
            .is_ok());
    }

    #[test]
    fn test_blind_relay_delivery_receipt_rejects_route_or_payload_substitution() {
        let terminal = IdentityKeyPair::generate();
        let route_id = [0x41u8; 16];
        let commitment = BlindRelayDeliveryReceipt::payload_commitment(b"opaque payload");
        let receipt =
            BlindRelayDeliveryReceipt::accepted(route_id, commitment, 1_700_000_200, &terminal);

        assert!(receipt
            .verify_expected(&[0x42u8; 16], &commitment, &terminal.public_key_bytes())
            .is_err());
        assert!(receipt
            .verify_expected(
                &route_id,
                &BlindRelayDeliveryReceipt::payload_commitment(b"other payload"),
                &terminal.public_key_bytes(),
            )
            .is_err());
    }

    #[test]
    fn test_blind_relay_delivery_receipt_rejects_tampered_signature_fields() {
        let terminal = IdentityKeyPair::generate();
        let commitment = BlindRelayDeliveryReceipt::payload_commitment(b"opaque payload");
        let mut receipt =
            BlindRelayDeliveryReceipt::accepted([0x51u8; 16], commitment, 1_700_000_200, &terminal);
        receipt.delivered_at = receipt.delivered_at.saturating_add(1);

        assert!(receipt.verify_signature().is_err());
    }

    #[test]
    fn test_purpose_bound_receipt_verifies_only_for_exact_workload() {
        let terminal = IdentityKeyPair::generate();
        let route_id = [0x61u8; 16];
        let payload = b"opaque terminal workload";
        let receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
            route_id,
            payload,
            OnionRoutePurpose::MessageRelay,
            1_700_000_300,
            &terminal,
        );

        assert_eq!(
            receipt.version,
            BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION
        );
        assert!(receipt.verify_signature().is_ok());
        assert!(receipt
            .verify_expected_for_purpose(
                &route_id,
                payload,
                OnionRoutePurpose::MessageRelay,
                &terminal.public_key_bytes(),
            )
            .is_ok());
        assert!(receipt
            .verify_expected_for_purpose(
                &route_id,
                payload,
                OnionRoutePurpose::BlindVaultPut,
                &terminal.public_key_bytes(),
            )
            .is_err());
    }

    #[test]
    fn test_purpose_commitment_separates_same_payload_across_workloads() {
        let payload = b"same opaque bytes";
        assert_ne!(
            BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
                payload,
                OnionRoutePurpose::MessageRelay,
            ),
            BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
                payload,
                OnionRoutePurpose::BlindVaultPut,
            )
        );
    }

    #[test]
    fn test_legacy_receipt_remains_verifiable_but_cannot_prove_purpose() {
        let terminal = IdentityKeyPair::generate();
        let route_id = [0x71u8; 16];
        let payload = b"legacy opaque payload";
        let receipt = BlindRelayDeliveryReceipt::accepted(
            route_id,
            BlindRelayDeliveryReceipt::payload_commitment(payload),
            1_700_000_400,
            &terminal,
        );

        assert!(receipt.verify_signature().is_ok());
        assert!(receipt
            .verify_expected_for_purpose(
                &route_id,
                payload,
                OnionRoutePurpose::MessageRelay,
                &terminal.public_key_bytes(),
            )
            .is_err());
    }

    #[test]
    fn test_blind_relay_failure_receipt_binds_request_reason_and_responder() {
        let previous_hop = IdentityKeyPair::generate();
        let responder = IdentityKeyPair::generate();
        let request = make_blind_envelope(&previous_hop);
        let receipt = BlindRelayFailureReceipt::failed(
            request.route_id,
            BlindRelayFailureReceipt::request_commitment(&request),
            "forward_failed",
            1_700_000_500,
            &responder,
        );

        assert_eq!(receipt.version, BLIND_RELAY_FAILURE_RECEIPT_VERSION);
        assert!(receipt.verify_signature().is_ok());
        assert!(receipt
            .verify_expected(&request, "forward_failed", &responder.public_key_bytes(),)
            .is_ok());
    }

    #[test]
    fn test_blind_relay_failure_receipt_rejects_replay_and_reason_substitution() {
        let previous_hop = IdentityKeyPair::generate();
        let responder = IdentityKeyPair::generate();
        let request = make_blind_envelope(&previous_hop);
        let receipt = BlindRelayFailureReceipt::failed(
            request.route_id,
            BlindRelayFailureReceipt::request_commitment(&request),
            "forward_failed",
            1_700_000_500,
            &responder,
        );
        let mut replayed_request = request.clone();
        replayed_request.encrypted_blob[0] ^= 0x01;

        assert!(receipt
            .verify_expected(
                &replayed_request,
                "forward_failed",
                &responder.public_key_bytes(),
            )
            .is_err());
        assert!(receipt
            .verify_expected(&request, "no_route", &responder.public_key_bytes())
            .is_err());
    }

    #[test]
    fn test_blind_relay_failure_receipt_rejects_signer_and_field_tampering() {
        let previous_hop = IdentityKeyPair::generate();
        let responder = IdentityKeyPair::generate();
        let other = IdentityKeyPair::generate();
        let request = make_blind_envelope(&previous_hop);
        let mut receipt = BlindRelayFailureReceipt::failed(
            request.route_id,
            BlindRelayFailureReceipt::request_commitment(&request),
            "forward_failed",
            1_700_000_500,
            &responder,
        );

        assert!(receipt
            .verify_expected(&request, "forward_failed", &other.public_key_bytes())
            .is_err());
        receipt.failed_at = receipt.failed_at.saturating_add(1);
        assert!(receipt.verify_signature().is_err());
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
