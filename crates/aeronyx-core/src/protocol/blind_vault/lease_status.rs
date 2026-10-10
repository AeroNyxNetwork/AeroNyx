// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/lease_status.rs
// ============================================
//! # Private lease status observations
//!
//! Owns the administration-key lease status request, its terminal-signed
//! status receipt, and the onion lease-status session.
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
    LEASE_STATUS_RECEIPT_SIGNING_DOMAIN, LEASE_STATUS_REQUEST_COMMITMENT_DOMAIN,
    LEASE_STATUS_SIGNING_DOMAIN,
};

/// Administration-key request for one replica-local lease observation.
///
/// [BLIND-VAULT-LEASE-STATUS 2026-08-28 by Codex] The query is carried only
/// inside the encrypted terminal layer. It authorizes disclosure of aggregate
/// lease-local retention state to the administrator without publishing a
/// lease identifier, activity timestamp, or usage count to route relays.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseStatusRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease whose status is requested.
    pub lease_id: [u8; 32],
    /// Random request identifier binding the response to this observation.
    pub request_id: [u8; 16],
    /// Client request time in Unix milliseconds for bounded replay rejection.
    pub requested_at_ms: u64,
    /// Signature by the lease administration key.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseStatusRequest {
    /// Builds an unsigned lease-status request.
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

    /// Canonical lease-status signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_STATUS_SIGNING_DOMAIN.len() + 58);
        bytes.extend_from_slice(LEASE_STATUS_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.requested_at_ms.to_be_bytes());
        bytes
    }

    /// Domain-separated commitment to the exact signed observation request.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        let signing_bytes = self.signing_bytes();
        let mut bytes =
            Vec::with_capacity(LEASE_STATUS_REQUEST_COMMITMENT_DOMAIN.len() + signing_bytes.len());
        bytes.extend_from_slice(LEASE_STATUS_REQUEST_COMMITMENT_DOMAIN);
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
        self.validate_freshness(now_ms, maximum_clock_skew_ms)?;
        admin_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Validates non-secret fields before constructing an onion route.
    pub fn validate_freshness(
        &self,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
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
}

/// Terminal-signed coherent status for one still-live replica lease.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseStatusReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease observed by the terminal.
    pub lease_id: [u8; 32],
    /// Request identifier copied from the signed query.
    pub request_id: [u8; 16],
    /// Commitment to the complete status-request semantics.
    pub request_commitment: [u8; 32],
    /// Current lease expiry observed under the storage lock.
    pub expires_at_ms: u64,
    /// Number of still-live opaque objects at observation time.
    pub live_object_count: u64,
    /// Total bytes of still-live ciphertext at observation time.
    pub live_ciphertext_bytes: u64,
    /// Terminal observation time in Unix milliseconds.
    pub observed_at_ms: u64,
    /// Descriptor identity of the observing terminal node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseStatusReceipt {
    /// Builds an unsigned status receipt from one coherent node observation.
    #[must_use]
    pub fn new(
        request: &BlindVaultLeaseStatusRequest,
        expires_at_ms: u64,
        live_object_count: u64,
        live_ciphertext_bytes: u64,
        observed_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: request.lease_id,
            request_id: request.request_id,
            request_commitment: request.commitment(),
            expires_at_ms,
            live_object_count,
            live_ciphertext_bytes,
            observed_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical terminal-signing input for the status receipt.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_STATUS_RECEIPT_SIGNING_DOMAIN.len() + 146);
        bytes.extend_from_slice(LEASE_STATUS_RECEIPT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.request_commitment);
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.live_object_count.to_be_bytes());
        bytes.extend_from_slice(&self.live_ciphertext_bytes.to_be_bytes());
        bytes.extend_from_slice(&self.observed_at_ms.to_be_bytes());
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

    /// Confirms that this receipt answers the exact signed status request.
    #[must_use]
    pub fn matches_status(&self, request: &BlindVaultLeaseStatusRequest) -> bool {
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
        if self.observed_at_ms == 0
            || self.expires_at_ms <= self.observed_at_ms
            || (self.live_object_count == 0) != (self.live_ciphertext_bytes == 0)
        {
            return Err(BlindVaultError::InvalidLeaseStatusSummary);
        }
        Ok(())
    }
}

/// Source-owned state for one anonymous lease-status observation.
///
/// [BLIND-VAULT-ONION-LEASE-STATUS 2026-08-28 by Codex] Only request
/// commitments and the single-use reply key remain after route construction.
/// The administration signature and lease-local status never enter a middle
/// relay-visible frame.
pub struct BlindVaultOnionLeaseStatusSession {
    expected_version: u16,
    expected_lease_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_request_commitment: [u8; 32],
    reply_session: OnionReplySession,
}

impl BlindVaultOnionLeaseStatusSession {
    /// Encodes one administration-authorized status query for the terminal.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultLeaseStatusRequest,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionLeaseStatusError> {
        request.validate_freshness(now_ms, maximum_clock_skew_ms)?;
        let expected_version = request.version;
        let expected_lease_id = request.lease_id;
        let expected_request_id = request.request_id;
        let expected_request_commitment = request.commitment();
        let encoded_status = encode_blind_vault_frame(&BlindVaultFrame::LeaseStatus(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_status,
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

    /// Opens and verifies the exact terminal-signed status receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultLeaseStatusReceipt, BlindVaultOnionLeaseStatusError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::LeaseStatus {
                return Err(BlindVaultOnionLeaseStatusError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionLeaseStatusError::TerminalFailure(
                failure.code(),
            ));
        }
        let BlindVaultFrame::LeaseStatusReceipt(receipt) = frame else {
            return Err(BlindVaultOnionLeaseStatusError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionLeaseStatusError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.lease_id != self.expected_lease_id
            || receipt.request_id != self.expected_request_id
            || receipt.request_commitment != self.expected_request_commitment
        {
            return Err(BlindVaultOnionLeaseStatusError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for anonymous lease-status observation.
#[derive(Debug, Error)]
pub enum BlindVaultOnionLeaseStatusError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion lease status frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion lease status reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion lease status terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// The decrypted workload response was not a status receipt.
    #[error("unexpected blind vault onion lease status response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact signed status request.
    #[error("blind vault onion lease status request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion lease status terminal identity")]
    InvalidTerminalIdentity,
}
