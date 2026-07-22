// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault.rs
// ============================================
//! # Blind Vault Protocol v1
//!
//! ## Creation Reason
//! Contact relationships and optional conversation history need durable,
//! multi-device storage without exposing an account identity, correspondent,
//! application namespace, content type, or social graph to a storage node.
//!
//! ## Main Functionality
//! - Defines the immutable encrypted object accepted by a blind vault node.
//! - Defines a node-signed storage receipt suitable for independent replicas.
//! - Provides deterministic signing bytes and bounded binary wire framing.
//! - Enforces coarse ciphertext size classes to reduce content-size leakage.
//! - Keeps application domain separation inside client ciphertext and keys.
//!
//! ## Dependencies
//! - `crypto::keys`: Ed25519 signing and verification wrappers.
//! - `protocol::mod`: public protocol exports.
//! - Future server storage/API modules consume these types without parsing
//!   `ciphertext`.
//!
//! ## Main Logical Flow
//! 1. A client encrypts a padded contact-vault or message-archive segment.
//! 2. It creates random, replica-specific lease/object/request identifiers.
//! 3. It signs `BlindVaultPutRequest::signing_bytes()` with the lease write key.
//! 4. A node validates policy and signature, stores the ciphertext verbatim,
//!    and returns a signed `BlindVaultStoredReceipt`.
//! 5. The client accepts a configured receipt quorum and repairs missing
//!    replicas without revealing that replicas belong to the same logical vault.
//!
//! ## Privacy Invariant
//! The outer frame intentionally has no owner, wallet, sender, receiver,
//! conversation, namespace, content type, relation edge, vector, keyword,
//! or plaintext timestamp field. Nodes may observe a replica-local lease,
//! object size class, expiry, and request timing; clients must use independent
//! wrappers/routes and rotate leases to bound that residual linkability.
//!
//! ## Important Note For The Next Developer
//! - Do not add account or application identifiers to this outer protocol.
//! - Do not reuse MemChain `remember_sealed`; that API exposes owner and index
//!   metadata by design and serves a different retrieval model.
//! - Do not put vault object IDs, commitments, or receipts in the public
//!   directory chain. Public commitments would create durable activity links.
//! - Do not change existing field order. Add a new frame version/kind instead.
//! - Media blobs use a separate bounded blob protocol; this object protocol is
//!   for padded metadata/message-event segments only.
//!
//! Last Modified: v1.0.0-BlindVaultWire - Initial node-blind durable object and
//! signed receipt contract.
//! ============================================

use bincode::Options;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};

/// [BLIND-VAULT-WIRE 2026-07-22 by Codex]
/// Domain separation prevents a valid chat/directory signature from being
/// replayed as blind-vault authorisation.
const PUT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Put-v1";
const RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-StoredReceipt-v1";
const FRAME_MAGIC: [u8; 4] = *b"ANBV";
const FRAME_HEADER_BYTES: usize = 7;
const FRAME_KIND_PUT: u8 = 1;
const FRAME_KIND_STORED_RECEIPT: u8 = 2;

/// Initial blind-vault wire version. This version is independent of the VPN
/// transport and legacy chat-envelope versions.
pub const BLIND_VAULT_PROTOCOL_VERSION: u16 = 1;

/// Maximum encoded frame accepted by the decoder. The largest v1 ciphertext
/// class is 256 KiB; the remaining space covers fixed metadata and framing.
pub const MAX_BLIND_VAULT_FRAME_BYTES: u64 = 272 * 1024;

/// Padded ciphertext size classes accepted by protocol v1.
///
/// Clients should batch small events and pad encryption output to one of these
/// classes. Attachments and media must use the encrypted blob channel.
pub const BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES: [usize; 4] =
    [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024];

/// Binary frame carrying either an immutable put request or a node receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlindVaultFrame {
    /// Client request to persist one immutable encrypted object.
    Put(BlindVaultPutRequest),
    /// Node proof that the exact object was accepted for bounded retention.
    StoredReceipt(BlindVaultStoredReceipt),
}

/// Immutable encrypted object submitted to one blind-vault replica.
///
/// `lease_id`, `object_id`, and `request_id` MUST be independently random for
/// each replica. Reusing them across nodes would allow colluding operators to
/// correlate copies even when `ciphertext` is re-randomised.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultPutRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Random identifier for one replica and rotation epoch.
    pub lease_id: [u8; 32],
    /// Random immutable object identifier, unique within the lease.
    pub object_id: [u8; 32],
    /// Random idempotency identifier retained across retries to this replica.
    pub request_id: [u8; 16],
    /// Client ciphertext including AEAD nonce/tag and size-class padding.
    pub ciphertext: Vec<u8>,
    /// SHA-256 over the exact ciphertext bytes. Used only for idempotency and
    /// receipt binding; it must never be published in the directory chain.
    pub ciphertext_commitment: [u8; 32],
    /// Absolute Unix timestamp in milliseconds after which the node may purge.
    pub expires_at_ms: u64,
    /// Signature by the replica/epoch-specific lease write key.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultPutRequest {
    /// Builds an unsigned request and computes its ciphertext commitment.
    #[must_use]
    pub fn new(
        lease_id: [u8; 32],
        object_id: [u8; 32],
        request_id: [u8; 16],
        ciphertext: Vec<u8>,
        expires_at_ms: u64,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            object_id,
            request_id,
            ciphertext_commitment: sha256(&ciphertext),
            ciphertext,
            expires_at_ms,
            signature: [0; 64],
        }
    }

    /// Canonical, allocation-bounded Ed25519 signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PUT_SIGNING_DOMAIN.len() + 130);
        bytes.extend_from_slice(PUT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.object_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&(self.ciphertext.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&self.ciphertext_commitment);
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        bytes
    }

    /// Signs this request with a lease-scoped write key.
    pub fn sign(&mut self, lease_write_key: &IdentityKeyPair) {
        self.signature = lease_write_key.sign(&self.signing_bytes());
    }

    /// Validates all node-enforceable invariants and the lease signature.
    ///
    /// `maximum_ttl_ms` is operator policy and must be non-zero. It is supplied
    /// by the node so protocol compatibility does not force one retention plan.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_ttl_ms: u64,
        lease_write_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate(now_ms, maximum_ttl_ms)?;
        lease_write_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Validates the request without accessing lease authorisation state.
    pub fn validate(&self, now_ms: u64, maximum_ttl_ms: u64) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("object_id", &self.object_id)?;
        require_non_zero("request_id", &self.request_id)?;

        if !BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES.contains(&self.ciphertext.len()) {
            return Err(BlindVaultError::InvalidCiphertextSize {
                actual: self.ciphertext.len(),
            });
        }
        if sha256(&self.ciphertext) != self.ciphertext_commitment {
            return Err(BlindVaultError::CommitmentMismatch);
        }
        if self.expires_at_ms <= now_ms {
            return Err(BlindVaultError::Expired);
        }
        if maximum_ttl_ms == 0 || self.expires_at_ms.saturating_sub(now_ms) > maximum_ttl_ms {
            return Err(BlindVaultError::LifetimeTooLong);
        }
        Ok(())
    }
}

/// Signed proof that one node accepted one exact opaque object until a bounded
/// time. It proves storage acceptance, not recipient delivery or message read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultStoredReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease accepted by the node.
    pub lease_id: [u8; 32],
    /// Immutable object accepted by the node.
    pub object_id: [u8; 32],
    /// Original put request identifier.
    pub request_id: [u8; 16],
    /// Commitment to the exact ciphertext stored by the node.
    pub ciphertext_commitment: [u8; 32],
    /// Node acceptance time in Unix milliseconds.
    pub accepted_at_ms: u64,
    /// Earliest promised retention deadline in Unix milliseconds.
    pub stored_until_ms: u64,
    /// Descriptor identity of the accepting node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultStoredReceipt {
    /// Creates an unsigned receipt bound to an already validated put request.
    #[must_use]
    pub fn from_put(
        put: &BlindVaultPutRequest,
        accepted_at_ms: u64,
        stored_until_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: put.lease_id,
            object_id: put.object_id,
            request_id: put.request_id,
            ciphertext_commitment: put.ciphertext_commitment,
            accepted_at_ms,
            stored_until_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical receipt signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(RECEIPT_SIGNING_DOMAIN.len() + 162);
        bytes.extend_from_slice(RECEIPT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.object_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.ciphertext_commitment);
        bytes.extend_from_slice(&self.accepted_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.stored_until_ms.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        bytes
    }

    /// Signs this receipt with the node descriptor identity key.
    pub fn sign(&mut self, node_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        if self.node_id != node_key.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates receipt semantics, node identity binding, and signature.
    pub fn validate_and_verify(&self, node_key: &IdentityPublicKey) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("object_id", &self.object_id)?;
        require_non_zero("request_id", &self.request_id)?;
        require_non_zero("ciphertext_commitment", &self.ciphertext_commitment)?;
        if self.accepted_at_ms >= self.stored_until_ms {
            return Err(BlindVaultError::InvalidReceiptWindow);
        }
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Confirms that this receipt is for the exact submitted request.
    #[must_use]
    pub fn matches_put(&self, put: &BlindVaultPutRequest) -> bool {
        self.version == put.version
            && self.lease_id == put.lease_id
            && self.object_id == put.object_id
            && self.request_id == put.request_id
            && self.ciphertext_commitment == put.ciphertext_commitment
            && self.stored_until_ms <= put.expires_at_ms
    }
}

/// Stable, bounded binary encoding with an explicit frame kind outside bincode.
/// This avoids depending on serde enum discriminants for future evolution.
pub fn encode_blind_vault_frame(frame: &BlindVaultFrame) -> Result<Vec<u8>, BlindVaultError> {
    let (kind, body) = match frame {
        BlindVaultFrame::Put(value) => (FRAME_KIND_PUT, serialize_body(value)?),
        BlindVaultFrame::StoredReceipt(value) => {
            (FRAME_KIND_STORED_RECEIPT, serialize_body(value)?)
        }
    };

    let total = FRAME_HEADER_BYTES
        .checked_add(body.len())
        .ok_or(BlindVaultError::FrameTooLarge)?;
    if total as u64 > MAX_BLIND_VAULT_FRAME_BYTES {
        return Err(BlindVaultError::FrameTooLarge);
    }

    let mut encoded = Vec::with_capacity(total);
    encoded.extend_from_slice(&FRAME_MAGIC);
    encoded.extend_from_slice(&BLIND_VAULT_PROTOCOL_VERSION.to_be_bytes());
    encoded.push(kind);
    encoded.extend_from_slice(&body);
    Ok(encoded)
}

/// Decodes one complete v1 frame and rejects unknown kinds/trailing bytes.
pub fn decode_blind_vault_frame(bytes: &[u8]) -> Result<BlindVaultFrame, BlindVaultError> {
    if bytes.len() < FRAME_HEADER_BYTES {
        return Err(BlindVaultError::TruncatedFrame);
    }
    if bytes.len() as u64 > MAX_BLIND_VAULT_FRAME_BYTES {
        return Err(BlindVaultError::FrameTooLarge);
    }
    if bytes[..4] != FRAME_MAGIC {
        return Err(BlindVaultError::InvalidMagic);
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    require_version(version)?;

    let body = &bytes[FRAME_HEADER_BYTES..];
    match bytes[6] {
        FRAME_KIND_PUT => Ok(BlindVaultFrame::Put(deserialize_body(body)?)),
        FRAME_KIND_STORED_RECEIPT => Ok(BlindVaultFrame::StoredReceipt(deserialize_body(body)?)),
        kind => Err(BlindVaultError::UnknownFrameKind(kind)),
    }
}

/// Validation and bounded wire-codec failures for Blind Vault v1.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultError {
    /// Frame or object uses a version unsupported by this implementation.
    #[error("unsupported blind-vault protocol version {0}")]
    UnsupportedVersion(u16),
    /// A security-sensitive random identifier was the all-zero sentinel.
    #[error("{0} must not be all zero")]
    ZeroIdentifier(&'static str),
    /// Ciphertext did not use one of the protocol's coarse padding classes.
    #[error("ciphertext length {actual} is not an allowed padded size class")]
    InvalidCiphertextSize {
        /// Received ciphertext length in bytes.
        actual: usize,
    },
    /// Declared commitment did not match the exact ciphertext bytes.
    #[error("ciphertext commitment does not match ciphertext")]
    CommitmentMismatch,
    /// Requested expiry was not later than node time.
    #[error("object expiry is not in the future")]
    Expired,
    /// Requested retention exceeded the accepting node's policy.
    #[error("object lifetime exceeds node policy")]
    LifetimeTooLong,
    /// Ed25519 verification failed.
    #[error("signature verification failed")]
    InvalidSignature,
    /// Receipt signer did not match the declared descriptor identity.
    #[error("receipt node identity does not match its signing key")]
    NodeIdentityMismatch,
    /// Receipt promised no positive retention interval.
    #[error("receipt storage window is invalid")]
    InvalidReceiptWindow,
    /// Frame did not start with the Blind Vault magic bytes.
    #[error("invalid blind-vault frame magic")]
    InvalidMagic,
    /// Frame did not contain a complete header.
    #[error("blind-vault frame is truncated")]
    TruncatedFrame,
    /// Encoded frame exceeded the hard protocol allocation limit.
    #[error("blind-vault frame exceeds the protocol limit")]
    FrameTooLarge,
    /// Frame kind is not implemented by this protocol version.
    #[error("unknown blind-vault frame kind {0}")]
    UnknownFrameKind(u8),
    /// A typed value could not be serialized within the protocol bound.
    #[error("blind-vault frame serialization failed")]
    Serialization,
    /// A frame body was malformed, oversized, or contained trailing bytes.
    #[error("blind-vault frame deserialization failed")]
    Deserialization,
}

fn require_version(version: u16) -> Result<(), BlindVaultError> {
    if version == BLIND_VAULT_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(BlindVaultError::UnsupportedVersion(version))
    }
}

fn require_non_zero(name: &'static str, bytes: &[u8]) -> Result<(), BlindVaultError> {
    if bytes.iter().any(|byte| *byte != 0) {
        Ok(())
    } else {
        Err(BlindVaultError::ZeroIdentifier(name))
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn serialize_body<T: Serialize>(value: &T) -> Result<Vec<u8>, BlindVaultError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_BLIND_VAULT_FRAME_BYTES)
        .serialize(value)
        .map_err(|_| BlindVaultError::Serialization)
}

fn deserialize_body<T>(bytes: &[u8]) -> Result<T, BlindVaultError>
where
    T: for<'de> Deserialize<'de>,
{
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_BLIND_VAULT_FRAME_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|_| BlindVaultError::Deserialization)
}

mod serde_bytes64 {
    use super::*;

    pub fn serialize<S: Serializer>(value: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error> {
        let mut low = [0u8; 32];
        let mut high = [0u8; 32];
        low.copy_from_slice(&value[..32]);
        high.copy_from_slice(&value[32..]);
        (low, high).serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 64], D::Error> {
        let (low, high): ([u8; 32], [u8; 32]) = Deserialize::deserialize(deserializer)?;
        let mut value = [0u8; 64];
        value[..32].copy_from_slice(&low);
        value[32..].copy_from_slice(&high);
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW_MS: u64 = 1_800_000_000_000;
    const MAX_TTL_MS: u64 = 30 * 24 * 60 * 60 * 1_000;

    fn lease_key() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[7; 32]).expect("valid deterministic lease key")
    }

    fn node_key() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[9; 32]).expect("valid deterministic node key")
    }

    fn signed_put() -> BlindVaultPutRequest {
        let mut put = BlindVaultPutRequest::new(
            [1; 32],
            [2; 32],
            [3; 16],
            vec![0xA5; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0]],
            NOW_MS + 24 * 60 * 60 * 1_000,
        );
        put.sign(&lease_key());
        put
    }

    #[test]
    fn signed_put_validates_and_round_trips() {
        let put = signed_put();
        put.validate_and_verify(NOW_MS, MAX_TTL_MS, &lease_key().public_key())
            .expect("valid put");

        let encoded =
            encode_blind_vault_frame(&BlindVaultFrame::Put(put.clone())).expect("encode put");
        let decoded = decode_blind_vault_frame(&encoded).expect("decode put");
        assert_eq!(decoded, BlindVaultFrame::Put(put));
    }

    #[test]
    fn ciphertext_tampering_breaks_commitment_before_signature_check() {
        let mut put = signed_put();
        put.ciphertext[0] ^= 0xFF;
        assert_eq!(
            put.validate_and_verify(NOW_MS, MAX_TTL_MS, &lease_key().public_key()),
            Err(BlindVaultError::CommitmentMismatch)
        );
    }

    #[test]
    fn only_coarse_padded_size_classes_are_accepted() {
        let mut put = signed_put();
        put.ciphertext.pop();
        put.ciphertext_commitment = sha256(&put.ciphertext);
        put.sign(&lease_key());
        assert_eq!(
            put.validate(NOW_MS, MAX_TTL_MS),
            Err(BlindVaultError::InvalidCiphertextSize { actual: 4095 })
        );
    }

    #[test]
    fn ttl_policy_is_enforced_without_exposing_retention_semantics() {
        let mut put = signed_put();
        put.expires_at_ms = NOW_MS + MAX_TTL_MS + 1;
        put.sign(&lease_key());
        assert_eq!(
            put.validate(NOW_MS, MAX_TTL_MS),
            Err(BlindVaultError::LifetimeTooLong)
        );
    }

    #[test]
    fn stored_receipt_is_node_bound_and_matches_exact_put() {
        let put = signed_put();
        let key = node_key();
        let mut receipt = BlindVaultStoredReceipt::from_put(
            &put,
            NOW_MS + 100,
            put.expires_at_ms,
            key.public_key_bytes(),
        );
        receipt.sign(&key).expect("matching node identity");
        receipt
            .validate_and_verify(&key.public_key())
            .expect("valid receipt");
        assert!(receipt.matches_put(&put));

        receipt.ciphertext_commitment[0] ^= 1;
        assert_eq!(
            receipt.validate_and_verify(&key.public_key()),
            Err(BlindVaultError::InvalidSignature)
        );
        assert!(!receipt.matches_put(&put));
    }

    #[test]
    fn receipt_cannot_be_signed_as_another_node() {
        let put = signed_put();
        let mut receipt =
            BlindVaultStoredReceipt::from_put(&put, NOW_MS + 100, put.expires_at_ms, [4; 32]);
        assert_eq!(
            receipt.sign(&node_key()),
            Err(BlindVaultError::NodeIdentityMismatch)
        );
    }

    #[test]
    fn decoder_rejects_unknown_kind_and_trailing_body_bytes() {
        let put = signed_put();
        let mut encoded = encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode put");
        encoded[6] = 0xFE;
        assert_eq!(
            decode_blind_vault_frame(&encoded),
            Err(BlindVaultError::UnknownFrameKind(0xFE))
        );

        let put = signed_put();
        let mut encoded = encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode put");
        encoded.push(0);
        assert_eq!(
            decode_blind_vault_frame(&encoded),
            Err(BlindVaultError::Deserialization)
        );
    }
}
