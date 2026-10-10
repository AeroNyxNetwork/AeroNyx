// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/lease_admission.rs
// ============================================
//! # Anonymous lease creation and admission
//!
//! Owns the self-authenticating `BlindVaultLeaseCreateRequest`, the V1
//! issuer-signed bearer `BlindVaultAdmissionTicket`, the additive RFC 9474
//! blind-issued `BlindVaultBlindAdmissionToken`, their atomic admission
//! requests, the terminal-signed acceptance receipt, and the source-owned
//! onion lease-admission session.
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
    ADMISSION_SIGNING_DOMAIN, BLIND_ADMISSION_MESSAGE_DOMAIN, BLIND_ADMISSION_SPEND_DOMAIN,
    BLIND_LEASE_ACCEPTED_SIGNING_DOMAIN, BLIND_VAULT_BLIND_ADMISSION_VERSION,
    BLIND_VAULT_PROTOCOL_VERSION, LEASE_SIGNING_DOMAIN,
};

/// RSA-2048 through RSA-4096 signatures are accepted by the wire contract.
pub const MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES: usize = 256;

/// Upper RSA signature bound prevents attacker-controlled allocation growth.
pub const MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES: usize = 512;

/// Anonymous lease metadata signed by its independent administration key.
///
/// Admission and quota policy are deliberately outside this structure. A node
/// must authenticate or rate-limit lease creation separately without adding an
/// account identity to the durable lease record.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseCreateRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Random identifier unique to one node replica and rotation epoch.
    pub lease_id: [u8; 32],
    /// Random idempotency identifier retained across creation retries.
    pub request_id: [u8; 16],
    /// Ed25519 key permitted to append immutable objects to this lease.
    pub write_verifying_key: [u8; 32],
    /// Separate Ed25519 key permitted to remove objects or retire this lease.
    pub admin_verifying_key: [u8; 32],
    /// SHA-256 of the random bearer capability used for private reads.
    pub read_capability_hash: [u8; 32],
    /// Absolute Unix timestamp in milliseconds after which the lease expires.
    pub expires_at_ms: u64,
    /// Signature by `admin_verifying_key` over all preceding fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseCreateRequest {
    /// Builds an unsigned anonymous lease request.
    #[must_use]
    pub fn new(
        lease_id: [u8; 32],
        request_id: [u8; 16],
        write_verifying_key: [u8; 32],
        admin_verifying_key: [u8; 32],
        read_capability_hash: [u8; 32],
        expires_at_ms: u64,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            request_id,
            write_verifying_key,
            admin_verifying_key,
            read_capability_hash,
            expires_at_ms,
            signature: [0; 64],
        }
    }

    /// Canonical lease-creation signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_SIGNING_DOMAIN.len() + 154);
        bytes.extend_from_slice(LEASE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.write_verifying_key);
        bytes.extend_from_slice(&self.admin_verifying_key);
        bytes.extend_from_slice(&self.read_capability_hash);
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        bytes
    }

    /// Signs the request with the lease administration key.
    pub fn sign(&mut self, admin_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        if self.admin_verifying_key != admin_key.public_key_bytes() {
            return Err(BlindVaultError::AdminIdentityMismatch);
        }
        self.signature = admin_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates anonymous lease fields and the self-authenticating admin key.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
    ) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
        require_non_zero("read_capability_hash", &self.read_capability_hash)?;
        if self.write_verifying_key == self.admin_verifying_key {
            return Err(BlindVaultError::LeaseKeyReuse);
        }
        let write_key = IdentityPublicKey::from_bytes(&self.write_verifying_key)
            .map_err(|_| BlindVaultError::InvalidPublicKey)?;
        let admin_key = IdentityPublicKey::from_bytes(&self.admin_verifying_key)
            .map_err(|_| BlindVaultError::InvalidPublicKey)?;
        // Parse both keys even though only the administration key signs this
        // request. This prevents an unusable lease from entering durable state.
        let _ = write_key;
        validate_future_deadline(now_ms, self.expires_at_ms, maximum_lease_ttl_ms)?;
        admin_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }
}

/// Short-lived bearer credential authorising one anonymous lease admission.
///
/// The issuer signs only random token material and coarse resource policy. It
/// must not attach an account, wallet, device, application namespace, or lease
/// identifier. Storage nodes persist only the spent `token_id` until expiry.
/// Version 1 is deliberately named a bearer ticket rather than a blind token:
/// unlinkable issuance requires a separately audited blind-signature or VOPRF
/// issuer and will use an additive protocol version.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultAdmissionTicket {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Cryptographically random one-time bearer identifier.
    pub token_id: [u8; 32],
    /// Ed25519 identity of the operator-approved admission issuer.
    pub issuer_id: [u8; 32],
    /// Earliest Unix millisecond at which redemption is valid.
    pub not_before_ms: u64,
    /// Unix millisecond after which redemption and spent-state retention end.
    pub expires_at_ms: u64,
    /// Maximum lease lifetime this credential permits.
    pub maximum_lease_ttl_ms: u64,
    /// Ed25519 signature by `issuer_id` over all preceding fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultAdmissionTicket {
    /// Builds an unsigned admission ticket from random bearer material.
    #[must_use]
    pub fn new(
        token_id: [u8; 32],
        issuer_id: [u8; 32],
        not_before_ms: u64,
        expires_at_ms: u64,
        maximum_lease_ttl_ms: u64,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            token_id,
            issuer_id,
            not_before_ms,
            expires_at_ms,
            maximum_lease_ttl_ms,
            signature: [0; 64],
        }
    }

    /// Canonical issuer-signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(ADMISSION_SIGNING_DOMAIN.len() + 90);
        bytes.extend_from_slice(ADMISSION_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.token_id);
        bytes.extend_from_slice(&self.issuer_id);
        bytes.extend_from_slice(&self.not_before_ms.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.maximum_lease_ttl_ms.to_be_bytes());
        bytes
    }

    /// Signs this bearer ticket with its declared issuer identity.
    pub fn sign(&mut self, issuer_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        if self.issuer_id != issuer_key.public_key_bytes() {
            return Err(BlindVaultError::AdmissionIssuerMismatch);
        }
        self.signature = issuer_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates bounded lifetime, resource policy, issuer binding, and
    /// signature. Operator issuer allowlisting remains server policy.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_ticket_lifetime_ms: u64,
        issuer_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("admission_token_id", &self.token_id)?;
        if self.issuer_id != issuer_key.to_bytes() {
            return Err(BlindVaultError::AdmissionIssuerMismatch);
        }
        if self.not_before_ms > now_ms {
            return Err(BlindVaultError::AdmissionNotYetValid);
        }
        let validity_window = self
            .expires_at_ms
            .checked_sub(self.not_before_ms)
            .ok_or(BlindVaultError::InvalidAdmissionPolicy)?;
        if validity_window == 0
            || validity_window > maximum_ticket_lifetime_ms
            || self.maximum_lease_ttl_ms == 0
        {
            return Err(BlindVaultError::InvalidAdmissionPolicy);
        }
        validate_future_deadline(now_ms, self.expires_at_ms, maximum_ticket_lifetime_ms)?;
        issuer_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }
}

/// Atomic wire request pairing one bearer admission with one anonymous lease.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseAdmissionRequest {
    /// Short-lived issuer-signed one-time credential.
    pub admission: BlindVaultAdmissionTicket,
    /// Self-authenticating random replica lease requested by the bearer.
    pub lease: BlindVaultLeaseCreateRequest,
}

impl BlindVaultLeaseAdmissionRequest {
    /// Validates both signatures and applies the narrower node/ticket lease
    /// lifetime without binding the two random identifiers in durable state.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        node_maximum_lease_ttl_ms: u64,
        maximum_ticket_lifetime_ms: u64,
        issuer_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.admission
            .validate_and_verify(now_ms, maximum_ticket_lifetime_ms, issuer_key)?;
        self.lease.validate_and_verify(
            now_ms,
            node_maximum_lease_ttl_ms.min(self.admission.maximum_lease_ttl_ms),
        )
    }
}

/// Unlinkable one-time admission credential finalized by an RFC 9474 client.
///
/// The issuer sees only a blinded message during issuance. The storage node
/// later receives these fields, verifies them against an operator-pinned RSA
/// epoch key, and cannot correlate redemption with the issuance transcript.
/// Resource limits and validity are intentionally absent: they are fixed by
/// the pinned issuer-key policy so a blind client cannot choose its own quota.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindAdmissionToken {
    /// Blind-admission credential version; independent of the outer frame.
    pub version: u16,
    /// SHA-256 fingerprint of the issuer's canonical RSA public-key DER.
    pub issuer_key_id: [u8; 32],
    /// Cryptographically random one-time token chosen before blinding.
    pub token_id: [u8; 32],
    /// RFC 9474 randomized-message value retained by the client.
    pub message_randomizer: [u8; 32],
    /// Finalized RSA-PSS signature over `message_bytes()`.
    pub signature: Vec<u8>,
}

impl BlindVaultBlindAdmissionToken {
    /// Builds a finalized token from client-owned blind-signature output.
    #[must_use]
    pub fn new(
        issuer_key_id: [u8; 32],
        token_id: [u8; 32],
        message_randomizer: [u8; 32],
        signature: Vec<u8>,
    ) -> Self {
        Self {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id,
            token_id,
            message_randomizer,
            signature,
        }
    }

    /// Domain-separated message blinded and signed by the external issuer.
    #[must_use]
    pub fn message_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(BLIND_ADMISSION_MESSAGE_DOMAIN.len() + 66);
        bytes.extend_from_slice(BLIND_ADMISSION_MESSAGE_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.issuer_key_id);
        bytes.extend_from_slice(&self.token_id);
        bytes
    }

    /// Opaque replay marker stored by a node until the issuer epoch expires.
    ///
    /// Hashing a domain, key fingerprint, and token avoids cross-scheme spend
    /// collisions without adding account or lease identifiers to durable state.
    #[must_use]
    pub fn spend_id(&self) -> [u8; 32] {
        let mut bytes = Vec::with_capacity(BLIND_ADMISSION_SPEND_DOMAIN.len() + 64);
        bytes.extend_from_slice(BLIND_ADMISSION_SPEND_DOMAIN);
        bytes.extend_from_slice(&self.issuer_key_id);
        bytes.extend_from_slice(&self.token_id);
        sha256(&bytes)
    }

    /// Validates allocation bounds and random identifiers before RSA work.
    pub fn validate_shape(&self) -> Result<(), BlindVaultError> {
        if self.version != BLIND_VAULT_BLIND_ADMISSION_VERSION {
            return Err(BlindVaultError::UnsupportedBlindAdmissionVersion(
                self.version,
            ));
        }
        require_non_zero("blind_admission_issuer_key_id", &self.issuer_key_id)?;
        require_non_zero("blind_admission_token_id", &self.token_id)?;
        require_non_zero(
            "blind_admission_message_randomizer",
            &self.message_randomizer,
        )?;
        if !(MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES..=MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES)
            .contains(&self.signature.len())
        {
            return Err(BlindVaultError::InvalidBlindAdmissionSignatureLength {
                actual: self.signature.len(),
            });
        }
        Ok(())
    }
}

/// Atomic V2 redemption pairing one unlinkable token with one random lease.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindLeaseAdmissionRequest {
    /// RFC 9474 finalized one-time credential.
    pub admission: BlindVaultBlindAdmissionToken,
    /// Self-authenticating random replica lease requested by the bearer.
    pub lease: BlindVaultLeaseCreateRequest,
}

/// Terminal-signed proof that one blind-issued lease request was accepted.
///
/// [BLIND-VAULT-ONION-ADMISSION 2026-08-28 by Codex] The receipt binds the
/// opaque credential spend marker and exact self-authenticating lease without
/// exposing the issuer key, token id, or lease authority to middle relays. It
/// is returned only inside a fixed-size encrypted onion response.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindLeaseAcceptedReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Domain-separated hash of the redeemed blind credential.
    pub admission_spend_id: [u8; 32],
    /// Replica-local lease accepted by the terminal.
    pub lease_id: [u8; 32],
    /// Original idempotency identifier from the signed lease request.
    pub request_id: [u8; 16],
    /// Exact bounded lease expiry accepted by the terminal.
    pub lease_expires_at_ms: u64,
    /// Acceptance time in Unix milliseconds.
    pub accepted_at_ms: u64,
    /// Descriptor identity of the accepting terminal node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultBlindLeaseAcceptedReceipt {
    /// Builds an unsigned receipt for one exact blind admission request.
    #[must_use]
    pub fn new(
        request: &BlindVaultBlindLeaseAdmissionRequest,
        accepted_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            admission_spend_id: request.admission.spend_id(),
            lease_id: request.lease.lease_id,
            request_id: request.lease.request_id,
            lease_expires_at_ms: request.lease.expires_at_ms,
            accepted_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical terminal-signing input for the admission receipt.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(BLIND_LEASE_ACCEPTED_SIGNING_DOMAIN.len() + 130);
        bytes.extend_from_slice(BLIND_LEASE_ACCEPTED_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.admission_spend_id);
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.lease_expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.accepted_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        bytes
    }

    /// Signs the receipt with the terminal descriptor identity.
    pub fn sign(&mut self, node_identity: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        if self.node_id != node_identity.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_identity.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates the receipt and verifies the accepting terminal signature.
    pub fn validate_and_verify(
        &self,
        terminal_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("admission_spend_id", &self.admission_spend_id)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("request_id", &self.request_id)?;
        require_non_zero("node_id", &self.node_id)?;
        if self.lease_expires_at_ms <= self.accepted_at_ms {
            return Err(BlindVaultError::InvalidReceiptWindow);
        }
        if self.node_id != terminal_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        terminal_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    /// Confirms that the receipt answers the exact blind admission request.
    #[must_use]
    pub fn matches_admission(&self, request: &BlindVaultBlindLeaseAdmissionRequest) -> bool {
        self.version == request.lease.version
            && self.admission_spend_id == request.admission.spend_id()
            && self.lease_id == request.lease.lease_id
            && self.request_id == request.lease.request_id
            && self.lease_expires_at_ms == request.lease.expires_at_ms
    }
}

/// Source-owned state for one anonymous blind-issued lease admission.
///
/// The finalized token and lease authorities exist only in the encrypted final
/// onion layer. Consuming the session on [`Self::open`] also consumes the
/// single-use reply private key, preventing a second response from being
/// accepted for the same route.
pub struct BlindVaultOnionLeaseAdmissionSession {
    expected_version: u16,
    expected_admission_spend_id: [u8; 32],
    expected_lease_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_lease_expires_at_ms: u64,
    reply_session: OnionReplySession,
}

impl BlindVaultOnionLeaseAdmissionSession {
    /// Encodes one V2 blind-issued admission for the selected terminal.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultBlindLeaseAdmissionRequest,
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionLeaseAdmissionError> {
        request.admission.validate_shape()?;
        request
            .lease
            .validate_and_verify(now_ms, maximum_lease_ttl_ms)?;
        let expected_version = request.lease.version;
        let expected_admission_spend_id = request.admission.spend_id();
        let expected_lease_id = request.lease.lease_id;
        let expected_request_id = request.lease.request_id;
        let expected_lease_expires_at_ms = request.lease.expires_at_ms;
        let encoded_admission =
            encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAdmission(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_admission,
        )?;
        let encoded_request = encode_onion_reply_request(&reply_request)?;
        Ok((
            encoded_request,
            Self {
                expected_version,
                expected_admission_spend_id,
                expected_lease_id,
                expected_request_id,
                expected_lease_expires_at_ms,
                reply_session,
            },
        ))
    }

    /// Opens and verifies the exact terminal-signed admission receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultBlindLeaseAcceptedReceipt, BlindVaultOnionLeaseAdmissionError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::LeaseAdmission {
                return Err(BlindVaultOnionLeaseAdmissionError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionLeaseAdmissionError::TerminalFailure(
                failure.code(),
            ));
        }
        let BlindVaultFrame::BlindLeaseAccepted(receipt) = frame else {
            return Err(BlindVaultOnionLeaseAdmissionError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionLeaseAdmissionError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.admission_spend_id != self.expected_admission_spend_id
            || receipt.lease_id != self.expected_lease_id
            || receipt.request_id != self.expected_request_id
            || receipt.lease_expires_at_ms != self.expected_lease_expires_at_ms
        {
            return Err(BlindVaultOnionLeaseAdmissionError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for anonymous blind-issued lease admission.
#[derive(Debug, Error)]
pub enum BlindVaultOnionLeaseAdmissionError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion lease admission frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion lease admission reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion lease admission terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// The decrypted workload response was not an admission receipt.
    #[error("unexpected blind vault onion lease admission response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact admission request.
    #[error("blind vault onion lease admission request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion lease admission terminal identity")]
    InvalidTerminalIdentity,
}

#[cfg(test)]
mod tests;
