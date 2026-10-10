// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/put.rs
// ============================================
//! # Immutable ciphertext writes
//!
//! Owns the immutable encrypted-object put request, the node-signed storage
//! receipt, and the onion put session that verifies it.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use crate::protocol::onion_reply::{
    encode_onion_reply_request, OnionReplyError, OnionReplySession,
    ONION_REPLY_RESPONSE_SIZE_CLASSES,
};

use super::error::BlindVaultError;
use super::frame::{decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultFrame};
use super::terminal_failure::{BlindVaultTerminalFailureCode, BlindVaultTerminalOperation};
use super::{
    require_non_zero, require_version, serde_bytes64, sha256, validate_future_deadline,
    BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES, BLIND_VAULT_PROTOCOL_VERSION, PUT_SIGNING_DOMAIN,
    RECEIPT_SIGNING_DOMAIN,
};

/// Immutable encrypted object submitted to one blind-vault replica.
///
/// `lease_id`, `object_id`, and `request_id` MUST be independently random for
/// each replica. Reusing them across nodes would allow colluding operators to
/// correlate copies even when `ciphertext` is re-randomised.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl std::fmt::Debug for BlindVaultPutRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultPutRequest")
            .field("version", &self.version)
            .field("ciphertext_bytes", &self.ciphertext.len())
            .field("request", &"[REDACTED]")
            .finish_non_exhaustive()
    }
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
        validate_future_deadline(now_ms, self.expires_at_ms, maximum_ttl_ms)
    }
}

/// Signed proof that one node accepted one exact opaque object until a bounded
/// time. It proves storage acceptance, not recipient delivery or message read.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
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

impl std::fmt::Debug for BlindVaultStoredReceipt {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultStoredReceipt")
            .field("version", &self.version)
            .field("receipt", &"[REDACTED]")
            .finish_non_exhaustive()
    }
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
    // [BLIND-VAULT 2026-07-23 by Codex] This intentionally compares the node's
    // retention promise with the client-requested expiry. It is not a repeated
    // struct-field comparison; accepting a longer promise would change the
    // opaque request's retention semantics.
    #[allow(clippy::suspicious_operation_groupings)]
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

/// Source-owned state for one anonymous Blind Vault write with a receipt.
///
/// [BLIND-VAULT-ONION-PUT-RECEIPT 2026-08-28 by Codex] The ciphertext and
/// lease write signature are moved into the encrypted final onion payload.
/// The retained session contains only receipt expectations and a single-use
/// reply key, so opening one response cannot disclose or replay the object.
pub struct BlindVaultOnionPutSession {
    expected_version: u16,
    expected_lease_id: [u8; 32],
    expected_object_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_ciphertext_commitment: [u8; 32],
    expected_expires_at_ms: u64,
    reply_session: OnionReplySession,
}

impl BlindVaultOnionPutSession {
    /// Encodes one signed immutable Put for the final onion layer.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultPutRequest,
        now_ms: u64,
        maximum_object_ttl_ms: u64,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionPutError> {
        request.validate(now_ms, maximum_object_ttl_ms)?;
        if request.ciphertext.len() != BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0] {
            return Err(BlindVaultOnionPutError::UnsupportedInlineCiphertextSize);
        }
        let expected_version = request.version;
        let expected_lease_id = request.lease_id;
        let expected_object_id = request.object_id;
        let expected_request_id = request.request_id;
        let expected_ciphertext_commitment = request.ciphertext_commitment;
        let expected_expires_at_ms = request.expires_at_ms;
        let encoded_put = encode_blind_vault_frame(&BlindVaultFrame::Put(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_put,
        )?;
        let encoded_request = encode_onion_reply_request(&reply_request)?;
        Ok((
            encoded_request,
            Self {
                expected_version,
                expected_lease_id,
                expected_object_id,
                expected_request_id,
                expected_ciphertext_commitment,
                expected_expires_at_ms,
                reply_session,
            },
        ))
    }

    /// Opens and verifies the exact terminal-signed storage receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultStoredReceipt, BlindVaultOnionPutError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::Put {
                return Err(BlindVaultOnionPutError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionPutError::TerminalFailure(failure.code()));
        }
        let BlindVaultFrame::StoredReceipt(receipt) = frame else {
            return Err(BlindVaultOnionPutError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionPutError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.lease_id != self.expected_lease_id
            || receipt.object_id != self.expected_object_id
            || receipt.request_id != self.expected_request_id
            || receipt.ciphertext_commitment != self.expected_ciphertext_commitment
            || receipt.stored_until_ms > self.expected_expires_at_ms
        {
            return Err(BlindVaultOnionPutError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for anonymous Blind Vault writes with receipts.
#[derive(Debug, Error)]
pub enum BlindVaultOnionPutError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion put frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion put reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion put terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// Inline v1 deliberately accepts only the 4 KiB ciphertext class.
    #[error("blind vault onion put requires the 4 KiB ciphertext class")]
    UnsupportedInlineCiphertextSize,
    /// The decrypted workload response was not a storage receipt.
    #[error("unexpected blind vault onion put response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact immutable Put request.
    #[error("blind vault onion put request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion put terminal identity")]
    InvalidTerminalIdentity,
}

#[cfg(test)]
mod tests;
