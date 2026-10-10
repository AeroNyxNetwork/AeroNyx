// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/delete.rs
// ============================================
//! # Object deletion
//!
//! Owns the administration-key object deletion request, the node-signed
//! deletion receipt, and the request-bound onion delete session.
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
    require_non_zero, require_version, serde_bytes64, BLIND_VAULT_PROTOCOL_VERSION,
    DELETE_RECEIPT_SIGNING_DOMAIN, DELETE_SIGNING_DOMAIN,
};

/// Administration-key request to delete one opaque object.
///
/// Nodes retain only a bounded tombstone needed for idempotent retries; no
/// application deletion reason or account identifier belongs on this frame.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultDeleteRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease containing the object.
    pub lease_id: [u8; 32],
    /// Immutable object to remove.
    pub object_id: [u8; 32],
    /// Random idempotency identifier retained across deletion retries.
    pub request_id: [u8; 16],
    /// Client request time in Unix milliseconds for bounded replay rejection.
    pub requested_at_ms: u64,
    /// Signature by the lease administration key.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultDeleteRequest {
    /// Builds an unsigned object-deletion request.
    #[must_use]
    pub fn new(
        lease_id: [u8; 32],
        object_id: [u8; 32],
        request_id: [u8; 16],
        requested_at_ms: u64,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            object_id,
            request_id,
            requested_at_ms,
            signature: [0; 64],
        }
    }

    /// Canonical deletion signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(DELETE_SIGNING_DOMAIN.len() + 90);
        bytes.extend_from_slice(DELETE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.object_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.requested_at_ms.to_be_bytes());
        bytes
    }

    /// Signs the request with the lease administration key.
    pub fn sign(&mut self, admin_key: &IdentityKeyPair) {
        self.signature = admin_key.sign(&self.signing_bytes());
    }

    /// Validates identifiers, request freshness, and administration signature.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("object_id", &self.object_id)?;
        require_non_zero("request_id", &self.request_id)?;
        let skew = if now_ms >= self.requested_at_ms {
            now_ms - self.requested_at_ms
        } else {
            self.requested_at_ms - now_ms
        };
        if skew > maximum_clock_skew_ms {
            return Err(BlindVaultError::RequestTimestampOutsideWindow);
        }
        admin_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }
}

/// Signed node proof that an opaque object was deleted or had already been
/// deleted under the same tombstone commitment.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultDeletedReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease from the deletion request.
    pub lease_id: [u8; 32],
    /// Removed immutable object identifier.
    pub object_id: [u8; 32],
    /// Original deletion request identifier.
    pub request_id: [u8; 16],
    /// Commitment of the removed ciphertext, retained only in the tombstone.
    pub previous_ciphertext_commitment: [u8; 32],
    /// Node deletion time in Unix milliseconds.
    pub deleted_at_ms: u64,
    /// Descriptor identity of the deleting node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultDeletedReceipt {
    /// Builds an unsigned deletion receipt.
    #[must_use]
    pub fn new(
        delete: &BlindVaultDeleteRequest,
        previous_ciphertext_commitment: [u8; 32],
        deleted_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: delete.lease_id,
            object_id: delete.object_id,
            request_id: delete.request_id,
            previous_ciphertext_commitment,
            deleted_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical deletion-receipt signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(DELETE_RECEIPT_SIGNING_DOMAIN.len() + 154);
        bytes.extend_from_slice(DELETE_RECEIPT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.object_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.previous_ciphertext_commitment);
        bytes.extend_from_slice(&self.deleted_at_ms.to_be_bytes());
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

    /// Validates node identity binding and signature.
    pub fn validate_and_verify(&self, node_key: &IdentityPublicKey) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("object_id", &self.object_id)?;
        require_non_zero("request_id", &self.request_id)?;
        require_non_zero(
            "previous_ciphertext_commitment",
            &self.previous_ciphertext_commitment,
        )?;
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Confirms that this receipt answers the exact deletion request.
    #[must_use]
    pub fn matches_delete(&self, delete: &BlindVaultDeleteRequest) -> bool {
        self.version == delete.version
            && self.lease_id == delete.lease_id
            && self.object_id == delete.object_id
            && self.request_id == delete.request_id
    }
}

/// Source-owned state for one anonymous Blind Vault deletion request.
///
/// [BLIND-VAULT-ONION-DELETE-SESSION 2026-08-28 by Codex] The administration
/// key remains client-side. Only its signed request enters the final onion
/// layer, while the consumed session retains non-secret receipt expectations
/// and the single-use encrypted reply state.
pub struct BlindVaultOnionDeleteSession {
    expected_version: u16,
    expected_lease_id: [u8; 32],
    expected_object_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_ciphertext_commitment: [u8; 32],
    reply_session: OnionReplySession,
}

impl BlindVaultOnionDeleteSession {
    /// Encodes one signed delete request for the final onion layer.
    ///
    /// `expected_ciphertext_commitment` comes from the client's accepted store
    /// receipt and prevents a terminal from proving deletion of different
    /// bytes under the same opaque object identifier.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        expected_ciphertext_commitment: [u8; 32],
        request: BlindVaultDeleteRequest,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionDeleteError> {
        require_version(request.version)?;
        require_non_zero("lease_id", &request.lease_id)?;
        require_non_zero("object_id", &request.object_id)?;
        require_non_zero("request_id", &request.request_id)?;
        require_non_zero(
            "expected_ciphertext_commitment",
            &expected_ciphertext_commitment,
        )?;
        let expected_version = request.version;
        let expected_lease_id = request.lease_id;
        let expected_object_id = request.object_id;
        let expected_request_id = request.request_id;
        let encoded_delete = encode_blind_vault_frame(&BlindVaultFrame::Delete(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_delete,
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
                reply_session,
            },
        ))
    }

    /// Opens and verifies the exact terminal-signed deletion receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultDeletedReceipt, BlindVaultOnionDeleteError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::Delete {
                return Err(BlindVaultOnionDeleteError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionDeleteError::TerminalFailure(failure.code()));
        }
        let BlindVaultFrame::DeletedReceipt(receipt) = frame else {
            return Err(BlindVaultOnionDeleteError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionDeleteError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.lease_id != self.expected_lease_id
            || receipt.object_id != self.expected_object_id
            || receipt.request_id != self.expected_request_id
            || receipt.previous_ciphertext_commitment != self.expected_ciphertext_commitment
        {
            return Err(BlindVaultOnionDeleteError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for anonymous Blind Vault deletion.
#[derive(Debug, Error)]
pub enum BlindVaultOnionDeleteError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion delete frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion delete reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion delete terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// The decrypted workload response was not a deletion receipt.
    #[error("unexpected blind vault onion delete response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact signed request.
    #[error("blind vault onion delete request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion delete terminal identity")]
    InvalidTerminalIdentity,
}

#[cfg(test)]
mod tests;
