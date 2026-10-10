// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/lease_retire.rs
// ============================================
//! # Complete lease retirement
//!
//! Owns the administration-key lease retirement request, its terminal-signed
//! aggregate receipt, and the onion lease-retirement session.
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
    require_non_zero, require_version, serde_bytes64, sha256, BLIND_VAULT_PROTOCOL_VERSION,
    LEASE_RETIRED_RECEIPT_SIGNING_DOMAIN, LEASE_RETIRE_REQUEST_COMMITMENT_DOMAIN,
    LEASE_RETIRE_SIGNING_DOMAIN,
};

/// Administration-key request to retire one complete replica-local lease.
///
/// [BLIND-VAULT-LEASE-RETIRE 2026-08-28 by Codex] Retirement removes every
/// ciphertext object and object tombstone under the lease in one transaction.
/// No user identity, application reason, or cross-replica identifier belongs
/// in this request.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseRetireRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease to retire.
    pub lease_id: [u8; 32],
    /// Random idempotency identifier retained across exact retries.
    pub request_id: [u8; 16],
    /// Client request time in Unix milliseconds for bounded replay rejection.
    pub requested_at_ms: u64,
    /// Signature by the lease administration key.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseRetireRequest {
    /// Builds an unsigned lease-retirement request.
    #[must_use]
    pub fn new(lease_id: [u8; 32], request_id: [u8; 16], requested_at_ms: u64) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            request_id,
            requested_at_ms,
            signature: [0; 64],
        }
    }

    /// Canonical lease-retirement signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_RETIRE_SIGNING_DOMAIN.len() + 58);
        bytes.extend_from_slice(LEASE_RETIRE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.requested_at_ms.to_be_bytes());
        bytes
    }

    /// Domain-separated commitment to the complete signed-request semantics.
    ///
    /// [BLIND-VAULT-LEASE-RETIRE-COMMITMENT 2026-08-28 by Codex] The
    /// commitment binds the timestamp as well as both identifiers. A retained
    /// retry must match it exactly before the node reuses a retirement result.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        let signing_bytes = self.signing_bytes();
        let mut bytes =
            Vec::with_capacity(LEASE_RETIRE_REQUEST_COMMITMENT_DOMAIN.len() + signing_bytes.len());
        bytes.extend_from_slice(LEASE_RETIRE_REQUEST_COMMITMENT_DOMAIN);
        bytes.extend_from_slice(&signing_bytes);
        sha256(&bytes)
    }

    /// Signs the request with the lease administration key.
    pub fn sign(&mut self, admin_key: &IdentityKeyPair) {
        self.signature = admin_key.sign(&self.signing_bytes());
    }

    /// Validates freshness and verifies the lease administration signature.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_and_verify_signature(admin_key)?;
        let skew = if now_ms >= self.requested_at_ms {
            now_ms - self.requested_at_ms
        } else {
            self.requested_at_ms - now_ms
        };
        if maximum_clock_skew_ms == 0 || skew > maximum_clock_skew_ms {
            return Err(BlindVaultError::RequestTimestampOutsideWindow);
        }
        Ok(())
    }

    /// Verifies an exact retained retry without reapplying its original clock window.
    ///
    /// Storage nodes may call this only after matching `lease_id` and
    /// `request_id` against a still-live retirement tombstone.
    pub fn validate_and_verify_signature(
        &self,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
        admin_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }
}

/// Signed node proof that one complete replica lease was retired.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseRetiredReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease from the retirement request.
    pub lease_id: [u8; 32],
    /// Original retirement request identifier.
    pub request_id: [u8; 16],
    /// Domain-separated commitment to the exact signed retirement semantics.
    pub request_commitment: [u8; 32],
    /// Durable retirement time in Unix milliseconds.
    pub retired_at_ms: u64,
    /// Number of ciphertext objects removed by the first successful request.
    pub deleted_object_count: u64,
    /// Total ciphertext bytes removed by the first successful request.
    pub deleted_ciphertext_bytes: u64,
    /// Descriptor identity of the retiring terminal node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseRetiredReceipt {
    /// Builds an unsigned lease-retirement receipt.
    #[must_use]
    pub fn new(
        request: &BlindVaultLeaseRetireRequest,
        retired_at_ms: u64,
        deleted_object_count: u64,
        deleted_ciphertext_bytes: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: request.lease_id,
            request_id: request.request_id,
            request_commitment: request.commitment(),
            retired_at_ms,
            deleted_object_count,
            deleted_ciphertext_bytes,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical lease-retirement receipt signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_RETIRED_RECEIPT_SIGNING_DOMAIN.len() + 138);
        bytes.extend_from_slice(LEASE_RETIRED_RECEIPT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.request_commitment);
        bytes.extend_from_slice(&self.retired_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.deleted_object_count.to_be_bytes());
        bytes.extend_from_slice(&self.deleted_ciphertext_bytes.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        bytes
    }

    /// Signs the receipt with the terminal descriptor identity.
    pub fn sign(&mut self, node_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates summary invariants, terminal identity, and signature.
    pub fn validate_and_verify(&self, node_key: &IdentityPublicKey) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Confirms that this receipt answers the exact retirement request.
    #[must_use]
    pub fn matches_retire(&self, request: &BlindVaultLeaseRetireRequest) -> bool {
        self.version == request.version
            && self.lease_id == request.lease_id
            && self.request_id == request.request_id
            && self.request_commitment == request.commitment()
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
        require_non_zero("request_commitment", &self.request_commitment)?;
        require_non_zero("node_id", &self.node_id)?;
        if self.retired_at_ms == 0
            || (self.deleted_object_count == 0) != (self.deleted_ciphertext_bytes == 0)
        {
            return Err(BlindVaultError::InvalidRetirementSummary);
        }
        Ok(())
    }
}

/// Source-owned state for one anonymous complete lease retirement.
///
/// [BLIND-VAULT-ONION-LEASE-RETIRE 2026-08-28 by Codex] The administration
/// signature exists only in the final onion payload. The consumed session
/// retains random request identifiers and the single-use reply key needed to
/// verify one terminal-signed retirement receipt.
pub struct BlindVaultOnionLeaseRetireSession {
    expected_version: u16,
    expected_lease_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_request_commitment: [u8; 32],
    reply_session: OnionReplySession,
}

impl BlindVaultOnionLeaseRetireSession {
    /// Encodes one signed complete lease retirement for the final onion layer.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultLeaseRetireRequest,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionLeaseRetireError> {
        require_version(request.version)?;
        require_non_zero("lease_id", &request.lease_id)?;
        require_non_zero("request_id", &request.request_id)?;
        let expected_version = request.version;
        let expected_lease_id = request.lease_id;
        let expected_request_id = request.request_id;
        let expected_request_commitment = request.commitment();
        let encoded_retire = encode_blind_vault_frame(&BlindVaultFrame::LeaseRetire(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_retire,
        )?;
        let encoded_request = encode_onion_reply_request(&reply_request)?;
        Ok((
            encoded_request,
            Self {
                expected_version,
                expected_lease_id,
                expected_request_id,
                expected_request_commitment,
                reply_session,
            },
        ))
    }

    /// Opens and verifies the exact terminal-signed retirement receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultLeaseRetiredReceipt, BlindVaultOnionLeaseRetireError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::LeaseRetire {
                return Err(BlindVaultOnionLeaseRetireError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionLeaseRetireError::TerminalFailure(
                failure.code(),
            ));
        }
        let BlindVaultFrame::LeaseRetiredReceipt(receipt) = frame else {
            return Err(BlindVaultOnionLeaseRetireError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionLeaseRetireError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.lease_id != self.expected_lease_id
            || receipt.request_id != self.expected_request_id
            || receipt.request_commitment != self.expected_request_commitment
        {
            return Err(BlindVaultOnionLeaseRetireError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for anonymous complete lease retirement.
#[derive(Debug, Error)]
pub enum BlindVaultOnionLeaseRetireError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion lease retirement frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion lease retirement reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion lease retirement terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// The decrypted workload response was not a retirement receipt.
    #[error("unexpected blind vault onion lease retirement response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact retirement request.
    #[error("blind vault onion lease retirement request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion lease retirement terminal identity")]
    InvalidTerminalIdentity,
}
