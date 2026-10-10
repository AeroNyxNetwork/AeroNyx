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
//! - Defines anonymous lease authority and signed object deletion contracts.
//! - Defines issuer-signed, short-lived bearer admission tickets without
//!   binding a storage lease to an account identity.
//! - Defines an additive RFC 9474 blind-issued admission credential whose
//!   redemption cannot be linked to its issuance transcript.
//! - Defines a node-signed issuer-epoch directory for authenticated key
//!   discovery and overlap-safe rotation.
//! - Defines an authority-signed issuer update for storage-node rotation.
//! - Defines bounded recovery request/page frames with node-signed ciphertext
//!   commitments and opaque continuation cursors.
//! - Prepares and verifies single-use onion recovery sessions without exposing
//!   read capabilities to entry or middle nodes.
//! - Defines administration-authorized, terminal-signed lease status proofs
//!   for private replica repair and renewal decisions.
//! - Defines streaming, per-replica inventory commitments so a source can
//!   verify its own manifest without linking independently wrapped replicas.
//! - Plans fail-closed replica renewal, reconciliation, retry, replacement,
//!   and provisioning from source-owned per-replica manifest expectations.
//! - Provides deterministic signing bytes and bounded binary wire framing.
//! - Enforces coarse ciphertext size classes to reduce content-size leakage.
//! - Keeps application domain separation inside client ciphertext and keys.
//!
//! ## Module Layout
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.
//! Every public item is re-exported from this module root, so all existing
//! `protocol::blind_vault::*` paths are unchanged.
//! - `blind_vault/delete.rs`: administration-key object deletion and receipts
//! - `blind_vault/error.rs`: shared validation and wire-codec error type
//! - `blind_vault/frame.rs`: frame constants, `BlindVaultFrame`, and the bounded codec
//! - `blind_vault/issuer_directory.rs`: blind-admission issuer epochs, directory, and updates
//! - `blind_vault/lease_admission.rs`: anonymous lease creation and V1/V2 admission
//! - `blind_vault/lease_inventory.rs`: private encrypted-object inventory commitments
//! - `blind_vault/lease_renewal.rs`: blind-authorized lease renewal
//! - `blind_vault/lease_retire.rs`: complete lease retirement
//! - `blind_vault/lease_status.rs`: private lease status observations
//! - `blind_vault/pull.rs`: encrypted-object recovery pages and onion pull
//! - `blind_vault/put.rs`: immutable ciphertext writes and storage receipts
//! - `blind_vault/redacted_debug.rs`: privacy-safe `Debug` for capability-bearing values
//! - `blind_vault/replica_plan.rs`: replica evidence and lifecycle planning
//! - `blind_vault/terminal_failure.rs`: encrypted terminal failure replies
//! - `blind_vault/test_support.rs`: shared test fixtures (test builds only)
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
//! - A v1 admission ticket remains a signed bearer credential. New clients
//!   should use the additive V2 blind-issued frame; never mutate kind 6.
//! - Do not change existing field order. Add a new frame version/kind instead.
//! - Media blobs use a separate bounded blob protocol; this object protocol is
//!   for padded metadata/message-event segments only.
//!
//! Last Modified: v1.22.0-ArchSplit - Split into focused child modules;
//! bodies unchanged.
//! v1.21.0-OnionPullSizeContract - Bound anonymous recovery to
//! the negotiated maximum fixed-size response class.
//! v1.20.0-PrivacySafeTypedDebug - Redacted direct formatting
//! for capability-, topology-, and commitment-bearing protocol values.
//! v1.19.0-PrivacySafeFrameDebug - Redacted top-level frames,
//! encrypted objects, bearer recovery requests, and storage receipts.
//! v1.18.0-PrivacySafeLifecycleDebug - Redacted replica action,
//! target, and plan topology from diagnostic formatting.
//! v1.17.0-BlindVaultEncryptedFailure - Added typed,
//! source-only terminal failure replies inside the fixed-size onion carrier.
//! v1.16.0-BlindVaultReplicaWorkflow - Exposed local-only
//! manifest expectation fields for exact, stale-plan-safe execution evidence.
//! v1.15.0-BlindVaultReplicaPlanner - Added source-owned,
//! manifest-bound replica evidence verification and lifecycle planning.
//! v1.14.0-BlindVaultOnionLeaseInventory - Added streaming,
//! private terminal-signed encrypted-object inventory commitments.
//! v1.13.0-BlindVaultOnionLeaseStatus - Added
//! administration-authorized, encrypted terminal-signed lease status observations.
//! v1.12.0-BlindVaultOnionLeaseRenewal - Added blind-authorized,
//! administration-key lease renewal with encrypted signed receipts.
//! v1.11.0-BlindVaultOnionLeaseRetire - Added request-bound,
//! administration-key lease retirement with encrypted aggregate receipts.
//! v1.10.0-BlindVaultOnionPutReceipt - Added bounded anonymous
//! 4 KiB writes with encrypted terminal-signed storage receipts.
//! v1.9.0-BlindVaultOnionAdmission - Added single-use anonymous
//! blind-issued lease admission with encrypted terminal-signed receipts.
//! v1.8.0-BlindVaultOnionDelete - Added request-bound anonymous
//! deletion sessions with terminal-signed receipt verification.
//! v1.7.0-BlindVaultOnionPull - Added client-owned anonymous
//! recovery session preparation and response verification.
//! v1.6.0-BlindVaultIssuerUpdate - Added a transport-independent
//! authority-signed runtime issuer update contract.
//! v1.5.0-BlindVaultIssuerDirectory - Added signed public
//! issuer-epoch discovery for safe blind-signing key rotation.
//! v1.4.0-BlindVaultBlindAdmission - Added an unlinkable
//! RFC 9474 redemption contract while preserving the V1 bearer frame.
//! v1.3.0-BlindVaultPull - Added bounded signed recovery pages
//! and per-frame allocation ceilings.
//! v1.2.0-BlindVaultAdmission - Added bounded issuer-signed
//! one-time bearer admission contracts.
//! v1.1.0-BlindVaultLease - Added anonymous lease authority,
//! administration-key deletion, and signed deletion receipts.
//! v1.0.0-BlindVaultWire - Initial durable object and signed receipt contract.
//! ============================================

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

mod delete;
mod error;
mod frame;
mod issuer_directory;
mod lease_admission;
mod lease_inventory;
mod lease_renewal;
mod lease_retire;
mod lease_status;
mod pull;
mod put;
mod redacted_debug;
mod replica_plan;
mod terminal_failure;
#[cfg(test)]
mod test_support;

pub use delete::{
    BlindVaultDeleteRequest, BlindVaultDeletedReceipt, BlindVaultOnionDeleteError,
    BlindVaultOnionDeleteSession,
};
pub use error::BlindVaultError;
pub use frame::{
    decode_blind_vault_frame, encode_blind_vault_frame, is_blind_vault_frame, BlindVaultFrame,
    MAX_BLIND_VAULT_FRAME_BYTES, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES,
    MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES,
};
pub use issuer_directory::{
    BlindVaultBlindIssuerDirectory, BlindVaultBlindIssuerEpoch, BlindVaultBlindIssuerUpdate,
    MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES, MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS,
    MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS,
};
pub use lease_admission::{
    BlindVaultAdmissionTicket, BlindVaultBlindAdmissionToken, BlindVaultBlindLeaseAcceptedReceipt,
    BlindVaultBlindLeaseAdmissionRequest, BlindVaultLeaseAdmissionRequest,
    BlindVaultLeaseCreateRequest, BlindVaultOnionLeaseAdmissionError,
    BlindVaultOnionLeaseAdmissionSession, MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
    MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
};
pub use lease_inventory::{
    BlindVaultInventoryCommitmentBuilder, BlindVaultInventoryCommitmentEntry,
    BlindVaultInventoryCommitmentSummary, BlindVaultLeaseInventoryReceipt,
    BlindVaultLeaseInventoryRequest, BlindVaultOnionLeaseInventoryError,
    BlindVaultOnionLeaseInventorySession,
};
pub use lease_renewal::{
    BlindVaultBlindLeaseRenewalRequest, BlindVaultBlindLeaseRenewedReceipt,
    BlindVaultLeaseRenewRequest, BlindVaultOnionLeaseRenewalError,
    BlindVaultOnionLeaseRenewalSession,
};
pub use lease_retire::{
    BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt, BlindVaultOnionLeaseRetireError,
    BlindVaultOnionLeaseRetireSession,
};
pub use lease_status::{
    BlindVaultLeaseStatusReceipt, BlindVaultLeaseStatusRequest, BlindVaultOnionLeaseStatusError,
    BlindVaultOnionLeaseStatusSession,
};
pub use pull::{
    BlindVaultOnionPullError, BlindVaultOnionPullSession, BlindVaultPullRequest,
    BlindVaultPullResponse, BlindVaultRecoveredObject, BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS,
    MAX_BLIND_VAULT_PULL_CURSOR_BYTES, MAX_BLIND_VAULT_PULL_OBJECTS,
};
pub use put::{
    BlindVaultOnionPutError, BlindVaultOnionPutSession, BlindVaultPutRequest,
    BlindVaultStoredReceipt,
};
pub use replica_plan::{
    BlindVaultManifestReplicaPlanner, BlindVaultReplicaAction, BlindVaultReplicaEvidence,
    BlindVaultReplicaEvidenceError, BlindVaultReplicaManifestExpectation, BlindVaultReplicaPlan,
    BlindVaultReplicaPlanError, BlindVaultReplicaPlanHealth, BlindVaultReplicaPlanner,
    BlindVaultReplicaPolicy, BlindVaultReplicaTarget, BlindVaultVerifiedReplicaInventory,
    MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS, MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS,
};
pub use terminal_failure::{
    BlindVaultTerminalFailure, BlindVaultTerminalFailureCode, BlindVaultTerminalOperation,
};

// ============================================
// Shared signing domains and protocol constants
// ============================================

/// [BLIND-VAULT-WIRE 2026-07-22 by Codex]
/// Domain separation prevents a valid chat/directory signature from being
/// replayed as blind-vault authorisation.
const PUT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Put-v1";
const RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-StoredReceipt-v1";
const LEASE_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Lease-v1";
const ADMISSION_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Admission-v1";
const BLIND_ADMISSION_MESSAGE_DOMAIN: &[u8] = b"AeroNyx-BlindVault-BlindAdmission-v2";
const BLIND_ADMISSION_SPEND_DOMAIN: &[u8] = b"AeroNyx-BlindVault-BlindSpend-v2";
const BLIND_LEASE_ACCEPTED_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-BlindLeaseAccepted-v1";
const BLIND_ISSUER_DIRECTORY_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-IssuerDirectory-v1";
const BLIND_ISSUER_UPDATE_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-IssuerUpdate-v1";
const PULL_RESPONSE_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-PullResponse-v1";
const DELETE_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Delete-v1";
const DELETE_RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-DeletedReceipt-v1";
const LEASE_RETIRE_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseRetire-v1";
const LEASE_RETIRE_REQUEST_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-LeaseRetire-RequestCommitment-v1";
const LEASE_RETIRED_RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseRetiredReceipt-v1";
const LEASE_RENEW_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseRenew-v1";
const LEASE_RENEW_REQUEST_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-LeaseRenew-RequestCommitment-v1";
const LEASE_RENEWED_RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseRenewedReceipt-v1";
const LEASE_STATUS_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseStatus-v1";
const LEASE_STATUS_REQUEST_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-LeaseStatus-RequestCommitment-v1";
const LEASE_STATUS_RECEIPT_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseStatusReceipt-v1";
const LEASE_INVENTORY_SIGNING_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseInventory-v1";
const LEASE_INVENTORY_REQUEST_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-LeaseInventory-RequestCommitment-v1";
const LEASE_INVENTORY_SET_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-LeaseInventory-SetCommitment-v1";
const LEASE_INVENTORY_SET_END_DOMAIN: &[u8] = b"AeroNyx-BlindVault-LeaseInventory-SetEnd-v1";
const LEASE_INVENTORY_RECEIPT_SIGNING_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-LeaseInventoryReceipt-v1";

/// Initial blind-vault wire version. This version is independent of the VPN
/// transport and legacy chat-envelope versions.
pub const BLIND_VAULT_PROTOCOL_VERSION: u16 = 1;

/// Unlinkable blind-admission credential version carried inside frame kind 9.
pub const BLIND_VAULT_BLIND_ADMISSION_VERSION: u16 = 2;

/// Padded ciphertext size classes accepted by protocol v1.
///
/// Clients should batch small events and pad encryption output to one of these
/// classes. Attachments and media must use the encrypted blob channel.
pub const BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES: [usize; 4] =
    [4 * 1024, 16 * 1024, 64 * 1024, 256 * 1024];

// ============================================
// Shared validation helpers
// ============================================

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

fn validate_future_deadline(
    now_ms: u64,
    deadline_ms: u64,
    maximum_lifetime_ms: u64,
) -> Result<(), BlindVaultError> {
    if deadline_ms <= now_ms {
        return Err(BlindVaultError::Expired);
    }
    if maximum_lifetime_ms == 0 || deadline_ms - now_ms > maximum_lifetime_ms {
        return Err(BlindVaultError::LifetimeTooLong);
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

// ============================================
// Serde helper for [u8; 64]
// ============================================

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
