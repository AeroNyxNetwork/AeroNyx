// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/lease_inventory.rs
// ============================================
//! # Private encrypted-object inventory commitments
//!
//! Owns the streaming per-replica inventory commitment builder and its
//! entry/summary values, the administration-key inventory request, the
//! terminal-signed inventory receipt, and the onion lease-inventory session.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
    require_non_zero, require_version, serde_bytes64, sha256, BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES,
    BLIND_VAULT_PROTOCOL_VERSION, LEASE_INVENTORY_RECEIPT_SIGNING_DOMAIN,
    LEASE_INVENTORY_REQUEST_COMMITMENT_DOMAIN, LEASE_INVENTORY_SET_COMMITMENT_DOMAIN,
    LEASE_INVENTORY_SET_END_DOMAIN, LEASE_INVENTORY_SIGNING_DOMAIN,
};

/// One canonical encrypted-object entry committed by a private inventory.
///
/// This type is deliberately not serializable as a protocol frame. It exists
/// only to let a source and terminal independently derive the same per-replica
/// commitment without publishing an object list.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultInventoryCommitmentEntry {
    /// Replica-local random object identifier.
    pub object_id: [u8; 32],
    /// SHA-256 commitment to the exact padded ciphertext bytes.
    pub ciphertext_commitment: [u8; 32],
    /// Object retention deadline committed by this inventory generation.
    pub expires_at_ms: u64,
    /// Exact padded ciphertext byte length.
    pub ciphertext_bytes: u64,
}

/// Aggregate result of one canonical inventory commitment pass.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultInventoryCommitmentSummary {
    /// Number of entries committed by the builder.
    pub object_count: u64,
    /// Total padded ciphertext bytes committed by the builder.
    pub ciphertext_bytes: u64,
    /// Domain-separated commitment to the ordered live object set.
    pub commitment: [u8; 32],
}

/// Streaming canonical inventory commitment builder.
///
/// [BLIND-VAULT-INVENTORY-COMMITMENT 2026-08-28 by Codex] Entries must arrive
/// in strict object-id order. Fixed-width framing plus an explicit end marker,
/// count, and byte total prevents ambiguity without retaining the full set in
/// memory. Different replicas remain unlinkable because their object IDs and
/// ciphertext wrappers are independently randomized.
pub struct BlindVaultInventoryCommitmentBuilder {
    hasher: Sha256,
    previous_object_id: Option<[u8; 32]>,
    object_count: u64,
    ciphertext_bytes: u64,
}

impl BlindVaultInventoryCommitmentBuilder {
    /// Starts one commitment for a replica-local lease.
    pub fn new(lease_id: [u8; 32]) -> Result<Self, BlindVaultError> {
        require_non_zero("lease_id", &lease_id)?;
        let mut hasher = Sha256::new();
        hasher.update(LEASE_INVENTORY_SET_COMMITMENT_DOMAIN);
        hasher.update(BLIND_VAULT_PROTOCOL_VERSION.to_be_bytes());
        hasher.update(lease_id);
        Ok(Self {
            hasher,
            previous_object_id: None,
            object_count: 0,
            ciphertext_bytes: 0,
        })
    }

    /// Commits one strictly ordered live object without retaining its metadata.
    pub fn push(
        &mut self,
        entry: BlindVaultInventoryCommitmentEntry,
    ) -> Result<(), BlindVaultError> {
        require_non_zero("object_id", &entry.object_id)?;
        require_non_zero("ciphertext_commitment", &entry.ciphertext_commitment)?;
        if entry.expires_at_ms == 0
            || !BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES.contains(
                &usize::try_from(entry.ciphertext_bytes)
                    .map_err(|_| BlindVaultError::InvalidInventoryEntry)?,
            )
        {
            return Err(BlindVaultError::InvalidInventoryEntry);
        }
        if self
            .previous_object_id
            .is_some_and(|previous| previous >= entry.object_id)
        {
            return Err(BlindVaultError::InvalidInventoryOrder);
        }
        self.object_count = self
            .object_count
            .checked_add(1)
            .ok_or(BlindVaultError::InventoryOverflow)?;
        self.ciphertext_bytes = self
            .ciphertext_bytes
            .checked_add(entry.ciphertext_bytes)
            .ok_or(BlindVaultError::InventoryOverflow)?;
        self.hasher.update(entry.object_id);
        self.hasher.update(entry.ciphertext_commitment);
        self.hasher.update(entry.expires_at_ms.to_be_bytes());
        self.hasher.update(entry.ciphertext_bytes.to_be_bytes());
        self.previous_object_id = Some(entry.object_id);
        Ok(())
    }

    /// Finalizes the exact ordered-set commitment and aggregate counters.
    #[must_use]
    pub fn finish(mut self) -> BlindVaultInventoryCommitmentSummary {
        self.hasher.update(LEASE_INVENTORY_SET_END_DOMAIN);
        self.hasher.update(self.object_count.to_be_bytes());
        self.hasher.update(self.ciphertext_bytes.to_be_bytes());
        BlindVaultInventoryCommitmentSummary {
            object_count: self.object_count,
            ciphertext_bytes: self.ciphertext_bytes,
            commitment: self.hasher.finalize().into(),
        }
    }
}

/// Administration-key request for one private replica inventory commitment.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseInventoryRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease whose live set is committed.
    pub lease_id: [u8; 32],
    /// Random request identifier binding the response to this observation.
    pub request_id: [u8; 16],
    /// Client request time in Unix milliseconds for bounded replay rejection.
    pub requested_at_ms: u64,
    /// Signature by the lease administration key.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseInventoryRequest {
    /// Builds an unsigned lease-inventory request.
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

    /// Canonical lease-inventory signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_INVENTORY_SIGNING_DOMAIN.len() + 58);
        bytes.extend_from_slice(LEASE_INVENTORY_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.requested_at_ms.to_be_bytes());
        bytes
    }

    /// Domain-separated commitment to the exact signed inventory request.
    #[must_use]
    pub fn commitment(&self) -> [u8; 32] {
        let signing_bytes = self.signing_bytes();
        let mut bytes = Vec::with_capacity(
            LEASE_INVENTORY_REQUEST_COMMITMENT_DOMAIN.len() + signing_bytes.len(),
        );
        bytes.extend_from_slice(LEASE_INVENTORY_REQUEST_COMMITMENT_DOMAIN);
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

/// Terminal-signed commitment to one coherent still-live replica inventory.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultLeaseInventoryReceipt {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease observed by the terminal.
    pub lease_id: [u8; 32],
    /// Request identifier copied from the signed query.
    pub request_id: [u8; 16],
    /// Commitment to the complete inventory-request semantics.
    pub request_commitment: [u8; 32],
    /// Current lease expiry observed with the inventory snapshot.
    pub expires_at_ms: u64,
    /// Number of still-live opaque objects in the committed set.
    pub live_object_count: u64,
    /// Total padded ciphertext bytes in the committed set.
    pub live_ciphertext_bytes: u64,
    /// Domain-separated commitment to the ordered live object set.
    pub inventory_commitment: [u8; 32],
    /// Terminal observation time in Unix milliseconds.
    pub observed_at_ms: u64,
    /// Descriptor identity of the observing terminal node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over the canonical receipt fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultLeaseInventoryReceipt {
    /// Builds an unsigned inventory receipt from one coherent node snapshot.
    #[must_use]
    pub fn new(
        request: &BlindVaultLeaseInventoryRequest,
        expires_at_ms: u64,
        summary: BlindVaultInventoryCommitmentSummary,
        observed_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: request.lease_id,
            request_id: request.request_id,
            request_commitment: request.commitment(),
            expires_at_ms,
            live_object_count: summary.object_count,
            live_ciphertext_bytes: summary.ciphertext_bytes,
            inventory_commitment: summary.commitment,
            observed_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical terminal-signing input for the inventory receipt.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(LEASE_INVENTORY_RECEIPT_SIGNING_DOMAIN.len() + 178);
        bytes.extend_from_slice(LEASE_INVENTORY_RECEIPT_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&self.request_id);
        bytes.extend_from_slice(&self.request_commitment);
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.live_object_count.to_be_bytes());
        bytes.extend_from_slice(&self.live_ciphertext_bytes.to_be_bytes());
        bytes.extend_from_slice(&self.inventory_commitment);
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

    /// Confirms that this receipt answers the exact signed inventory request.
    #[must_use]
    pub fn matches_inventory(&self, request: &BlindVaultLeaseInventoryRequest) -> bool {
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
        require_non_zero("inventory_commitment", &self.inventory_commitment)?;
        require_non_zero("node_id", &self.node_id)?;
        if self.observed_at_ms == 0
            || self.expires_at_ms <= self.observed_at_ms
            || (self.live_object_count == 0) != (self.live_ciphertext_bytes == 0)
        {
            return Err(BlindVaultError::InvalidInventorySummary);
        }
        Ok(())
    }
}

/// Source-owned state for one private replica inventory commitment.
///
/// [BLIND-VAULT-ONION-LEASE-INVENTORY 2026-08-28 by Codex] The object-set
/// commitment returns only through the single-use encrypted reply. Middle
/// relays never receive the lease ID, per-object commitments, aggregate usage,
/// or resulting inventory root.
pub struct BlindVaultOnionLeaseInventorySession {
    expected_version: u16,
    expected_lease_id: [u8; 32],
    expected_request_id: [u8; 16],
    expected_request_commitment: [u8; 32],
    reply_session: OnionReplySession,
}

impl BlindVaultOnionLeaseInventorySession {
    /// Encodes one administration-authorized inventory query for the terminal.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultLeaseInventoryRequest,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionLeaseInventoryError> {
        request.validate_freshness(now_ms, maximum_clock_skew_ms)?;
        let expected_version = request.version;
        let expected_lease_id = request.lease_id;
        let expected_request_id = request.request_id;
        let expected_request_commitment = request.commitment();
        let encoded_inventory =
            encode_blind_vault_frame(&BlindVaultFrame::LeaseInventory(request))?;
        // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Keep final
        // node identity and signed result inside the source-only reply; route
        // selection requires the matching path-wide descriptor features.
        let (reply_request, reply_session) = OnionReplySession::prepare_source_sealed(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_inventory,
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

    /// Opens and verifies the exact terminal-signed inventory receipt.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultLeaseInventoryReceipt, BlindVaultOnionLeaseInventoryError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::LeaseInventory {
                return Err(BlindVaultOnionLeaseInventoryError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionLeaseInventoryError::TerminalFailure(
                failure.code(),
            ));
        }
        let BlindVaultFrame::LeaseInventoryReceipt(receipt) = frame else {
            return Err(BlindVaultOnionLeaseInventoryError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionLeaseInventoryError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if receipt.version != self.expected_version
            || receipt.lease_id != self.expected_lease_id
            || receipt.request_id != self.expected_request_id
            || receipt.request_commitment != self.expected_request_commitment
        {
            return Err(BlindVaultOnionLeaseInventoryError::RequestMismatch);
        }
        Ok(receipt)
    }
}

/// Fail-closed source errors for private replica inventory commitments.
#[derive(Debug, Error)]
pub enum BlindVaultOnionLeaseInventoryError {
    /// The Blind Vault request or signed receipt violated its wire contract.
    #[error("blind vault onion lease inventory frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, request, identity, or signature checks.
    #[error("blind vault onion lease inventory reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion lease inventory terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// The decrypted workload response was not an inventory receipt.
    #[error("unexpected blind vault onion lease inventory response frame")]
    UnexpectedResponseFrame,
    /// The verified receipt did not answer the exact signed inventory request.
    #[error("blind vault onion lease inventory request mismatch")]
    RequestMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion lease inventory terminal identity")]
    InvalidTerminalIdentity,
}
