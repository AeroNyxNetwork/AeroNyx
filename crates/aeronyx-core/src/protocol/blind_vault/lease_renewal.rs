// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/lease_renewal.rs
// ============================================
//! # Blind-authorized lease renewal
//!
//! Owns the administration-key lease renewal request, the blind-issued
//! renewal wrapper, the terminal-signed renewed receipt, and the onion
//! lease-renewal session.
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
use super::lease_admission::BlindVaultBlindAdmissionToken;
use super::terminal_failure::{BlindVaultTerminalFailureCode, BlindVaultTerminalOperation};
use super::{
    require_non_zero, require_version, serde_bytes64, sha256, validate_future_deadline,
    BLIND_VAULT_PROTOCOL_VERSION, LEASE_RENEWED_RECEIPT_SIGNING_DOMAIN,
    LEASE_RENEW_REQUEST_COMMITMENT_DOMAIN, LEASE_RENEW_SIGNING_DOMAIN,
};

/// Administration-key request to extend one live replica-local lease.
///
/// [BLIND-VAULT-LEASE-RENEWAL 2026-08-28 by Codex] The signed request binds
/// both the expected current expiry and requested next expiry. This makes a
/// stale or reordered renewal fail closed instead of silently extending a
/// different lease generation.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseRenewRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease to extend.
    pub lease_id: [u8; 32],
    /// Random idempotency identifier retained across exact retries.
    pub request_id: [u8; 16],
    /// Exact lease expiry the client expects before this renewal.
    pub expected_expires_at_ms: u64,
    /// Exact bounded lease expiry requested after this renewal.
    pub requested_expires_at_ms: u64,
    /// Client request time in Unix milliseconds for bounded replay rejection.
    pub requested_at_ms: u64,
    /// Signature by the lease administration key.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseRenewRequest {
    /// Builds an unsigned lease-renewal request.
    #[must_use]
    pub fn new(
        lease_id: [u8; 32],
        request_id: [u8; 16],
        expected_expires_at_ms: u64,
        requested_expires_at_ms: u64,
        requested_at_ms: u64,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            request_id,
            expected_expires_at_ms,
            requested_expires_at_ms,
            requested_at_ms,
            signature: [0; 64],
        }
    }

    /// Canonical lease-renewal signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_RENEW_SIGNING_DOMAIN.len() + 74);
        bytes.extend_from_slice(LEASE_RENEW_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.expected_expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.requested_expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.requested_at_ms.to_be_bytes());
        bytes
    }

    /// Domain-separated commitment to one exact renewal transition.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        let signing_bytes = self.signing_bytes();
        let mut bytes =
            Vec::with_capacity(LEASE_RENEW_REQUEST_COMMITMENT_DOMAIN.len() + signing_bytes.len());
        bytes.extend_from_slice(LEASE_RENEW_REQUEST_COMMITMENT_DOMAIN);
        bytes.extend_from_slice(&signing_bytes);
        sha256(&bytes)
    }

    /// Signs the request with the lease administration key.
    pub fn sign(&mut self, admin_key: &IdentityKeyPair) {
        self.signature = admin_key.sign(&self.signing_bytes());
    }

    /// Validates the transition, freshness, and administration signature.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
        maximum_clock_skew_ms: u64,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_transition(now_ms, maximum_lease_ttl_ms)?;
        let skew = if now_ms >= self.requested_at_ms {
            now_ms - self.requested_at_ms
        } else {
            self.requested_at_ms - now_ms
        };
        if maximum_clock_skew_ms == 0 || skew > maximum_clock_skew_ms {
            return Err(BlindVaultError::RequestTimestampOutsideWindow);
        }
        self.validate_and_verify_signature(admin_key)
    }

    /// Validates non-secret transition fields before building an onion route.
    pub fn validate_transition(
        &self,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
    ) -> Result<(), BlindVaultError> {
        self.validate_shape()?;
        if self.expected_expires_at_ms <= now_ms {
            return Err(BlindVaultError::InvalidLeaseRenewalWindow);
        }
        validate_future_deadline(now_ms, self.requested_expires_at_ms, maximum_lease_ttl_ms)
    }

    /// Verifies an exact retained retry without reapplying its clock window.
    pub fn validate_and_verify_signature(
        &self,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_shape()?;
        admin_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_shape(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
        if self.expected_expires_at_ms == 0
            || self.requested_expires_at_ms <= self.expected_expires_at_ms
        {
            return Err(BlindVaultError::InvalidLeaseRenewalWindow);
        }
        Ok(())
    }
}

/// Atomic V2 renewal pairing one unlinkable capacity token with one lease.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindLeaseRenewalRequest {
    /// New RFC 9474 one-time credential authorizing additional retention.
    pub admission: BlindVaultBlindAdmissionToken,
    /// Existing lease transition signed by its independent administration key.
    pub renewal: BlindVaultLeaseRenewRequest,
}

/// Terminal-signed proof that one blind-authorized renewal was committed.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindLeaseRenewedReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Domain-separated replay marker of the consumed blind credential.
    pub admission_spend_id: [u8; 32],
    /// Replica-local lease renewed by the terminal.
    pub lease_id: [u8; 32],
    /// Original renewal idempotency identifier.
    pub request_id: [u8; 16],
    /// Domain-separated commitment to the exact renewal transition.
    pub request_commitment: [u8; 32],
    /// Lease expiry replaced by the successful transition.
    pub previous_expires_at_ms: u64,
    /// Exact new lease expiry committed by the terminal.
    pub renewed_expires_at_ms: u64,
    /// Commit time in Unix milliseconds.
    pub renewed_at_ms: u64,
    /// Descriptor identity of the renewing terminal node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultBlindLeaseRenewedReceipt {
    /// Builds an unsigned receipt for one committed renewal.
    #[must_use]
    pub fn new(
        request: &BlindVaultBlindLeaseRenewalRequest,
        renewed_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            admission_spend_id: request.admission.spend_id(),
            lease_id: request.renewal.lease_id,
            request_id: request.renewal.request_id,
            request_commitment: request.renewal.commitment(),
            previous_expires_at_ms: request.renewal.expected_expires_at_ms,
            renewed_expires_at_ms: request.renewal.requested_expires_at_ms,
            renewed_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical terminal-signing input for the renewal receipt.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_RENEWED_RECEIPT_SIGNING_DOMAIN.len() + 170);
        bytes.extend_from_slice(LEASE_RENEWED_RECEIPT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.admission_spend_id);
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.request_commitment);
        bytes.extend_from_slice(&self.previous_expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.renewed_expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.renewed_at_ms.to_be_bytes());
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

    /// Validates the transition and terminal signature.
    pub fn validate_and_verify(&self, node_key: &IdentityPublicKey) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Confirms that this receipt answers the exact authorized renewal.
    #[must_use]
    pub fn matches_renewal(&self, request: &BlindVaultBlindLeaseRenewalRequest) -> bool {
        self.version == request.renewal.version
            && self.admission_spend_id == request.admission.spend_id()
            && self.lease_id == request.renewal.lease_id
            && self.request_id == request.renewal.request_id
            && self.request_commitment == request.renewal.commitment()
            && self.previous_expires_at_ms == request.renewal.expected_expires_at_ms
            && self.renewed_expires_at_ms == request.renewal.requested_expires_at_ms
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("admission_spend_id", &self.admission_spend_id)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
        require_non_zero("request_commitment", &self.request_commitment)?;
        require_non_zero("node_id", &self.node_id)?;
        if self.renewed_at_ms == 0
            || self.previous_expires_at_ms <= self.renewed_at_ms
            || self.renewed_expires_at_ms <= self.previous_expires_at_ms
        {
            return Err(BlindVaultError::InvalidLeaseRenewalWindow);
        }
        Ok(())
    }
}

/// Source-owned state for one blind-authorized anonymous lease renewal.
///
/// [BLIND-VAULT-ONION-LEASE-RENEWAL 2026-08-28 by Codex] The blind token,
/// administration signature, and lease transition exist only in the encrypted
/// final layer. The consumed session retains commitments and a single-use
/// reply key, never the admission signature or administration secret.
pub struct BlindVaultOnionLeaseRenewalSession {
    expected_version: u16,
    expected_admission_spend_id: [u8; 32],
    expected_lease_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_request_commitment: [u8; 32],
    expected_previous_expires_at_ms: u64,
    expected_renewed_expires_at_ms: u64,
    reply_session: OnionReplySession,
}

impl BlindVaultOnionLeaseRenewalSession {
    /// Encodes one authorized renewal for the selected terminal.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultBlindLeaseRenewalRequest,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionLeaseRenewalError> {
        request.admission.validate_shape()?;
        request
            .renewal
            .validate_transition(now_ms, maximum_lease_ttl_ms)?;
        let expected_version = request.renewal.version;
        let expected_admission_spend_id = request.admission.spend_id();
        let expected_lease_id = request.renewal.lease_id;
        let expected_request_id = request.renewal.request_id;
        let expected_request_commitment = request.renewal.commitment();
        let expected_previous_expires_at_ms = request.renewal.expected_expires_at_ms;
        let expected_renewed_expires_at_ms = request.renewal.requested_expires_at_ms;
        let encoded_renewal =
            encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseRenewal(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_renewal,
        )?;
        let encoded_request = encode_onion_reply_request(&reply_request)?;
        Ok((
            encoded_request,
            Self {
                expected_version,
                expected_admission_spend_id,
                expected_lease_id,
                expected_request_id,
                expected_request_commitment,
                expected_previous_expires_at_ms,
                expected_renewed_expires_at_ms,
                reply_session,
            },
        ))
    }

    /// Opens and verifies the exact terminal-signed renewal receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultBlindLeaseRenewedReceipt, BlindVaultOnionLeaseRenewalError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::LeaseRenewal {
                return Err(BlindVaultOnionLeaseRenewalError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionLeaseRenewalError::TerminalFailure(
                failure.code(),
            ));
        }
        let BlindVaultFrame::BlindLeaseRenewed(receipt) = frame else {
            return Err(BlindVaultOnionLeaseRenewalError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionLeaseRenewalError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.admission_spend_id != self.expected_admission_spend_id
            || receipt.lease_id != self.expected_lease_id
            || receipt.request_id != self.expected_request_id
            || receipt.request_commitment != self.expected_request_commitment
            || receipt.previous_expires_at_ms != self.expected_previous_expires_at_ms
            || receipt.renewed_expires_at_ms != self.expected_renewed_expires_at_ms
        {
            return Err(BlindVaultOnionLeaseRenewalError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for blind-authorized anonymous lease renewal.
#[derive(Debug, Error)]
pub enum BlindVaultOnionLeaseRenewalError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion lease renewal frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion lease renewal reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion lease renewal terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// The decrypted workload response was not a renewal receipt.
    #[error("unexpected blind vault onion lease renewal response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact authorized renewal.
    #[error("blind vault onion lease renewal request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion lease renewal terminal identity")]
    InvalidTerminalIdentity,
}
