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
//! Last Modified: v1.17.0-BlindVaultEncryptedFailure - Added typed,
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

use std::collections::BTreeSet;

use bincode::Options;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::crypto::keys::{IdentityKeyPair, IdentityPublicKey};

use super::onion_reply::{
    encode_onion_reply_request, OnionReplyError, OnionReplySession,
    ONION_REPLY_RESPONSE_SIZE_CLASSES,
};

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
const FRAME_MAGIC: [u8; 4] = *b"ANBV";
const FRAME_HEADER_BYTES: usize = 7;
const FRAME_KIND_PUT: u8 = 1;
const FRAME_KIND_STORED_RECEIPT: u8 = 2;
const FRAME_KIND_LEASE_CREATE: u8 = 3;
const FRAME_KIND_DELETE: u8 = 4;
const FRAME_KIND_DELETED_RECEIPT: u8 = 5;
const FRAME_KIND_LEASE_ADMISSION: u8 = 6;
const FRAME_KIND_PULL_REQUEST: u8 = 7;
const FRAME_KIND_PULL_RESPONSE: u8 = 8;
const FRAME_KIND_BLIND_LEASE_ADMISSION: u8 = 9;
const FRAME_KIND_BLIND_ISSUER_DIRECTORY: u8 = 10;
const FRAME_KIND_BLIND_LEASE_ACCEPTED: u8 = 11;
const FRAME_KIND_LEASE_RETIRE: u8 = 12;
const FRAME_KIND_LEASE_RETIRED_RECEIPT: u8 = 13;
const FRAME_KIND_BLIND_LEASE_RENEWAL: u8 = 14;
const FRAME_KIND_BLIND_LEASE_RENEWED: u8 = 15;
const FRAME_KIND_LEASE_STATUS: u8 = 16;
const FRAME_KIND_LEASE_STATUS_RECEIPT: u8 = 17;
const FRAME_KIND_LEASE_INVENTORY: u8 = 18;
const FRAME_KIND_LEASE_INVENTORY_RECEIPT: u8 = 19;
const FRAME_KIND_TERMINAL_FAILURE: u8 = 20;

/// Returns whether an opaque payload declares the Blind Vault wire format.
///
/// [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] This intentionally checks
/// only the fixed magic prefix. Callers must still use
/// [`decode_blind_vault_frame`] and fail closed when the remainder is malformed;
/// falling back to another protocol after seeing this prefix would create a
/// parser-confusion boundary at an onion terminal.
#[must_use]
pub fn is_blind_vault_frame(bytes: &[u8]) -> bool {
    bytes.starts_with(&FRAME_MAGIC)
}

/// Initial blind-vault wire version. This version is independent of the VPN
/// transport and legacy chat-envelope versions.
pub const BLIND_VAULT_PROTOCOL_VERSION: u16 = 1;

/// Unlinkable blind-admission credential version carried inside frame kind 9.
pub const BLIND_VAULT_BLIND_ADMISSION_VERSION: u16 = 2;

/// RSA-2048 through RSA-4096 signatures are accepted by the wire contract.
pub const MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES: usize = 256;

/// Upper RSA signature bound prevents attacker-controlled allocation growth.
pub const MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES: usize = 512;

/// Maximum rotating blind-admission keys advertised by one storage node.
pub const MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS: usize = 16;

/// Maximum replica members accepted by one local lifecycle plan.
pub const MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS: usize = 16;

/// Maximum canonical RSA public-key DER accepted in an issuer directory.
pub const MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES: usize = 800;

/// Maximum lifetime of one advertised issuer epoch.
pub const MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS: u64 = 31 * 24 * 60 * 60 * 1_000;

/// Maximum mutation/request frame. The largest v1 ciphertext class is 256 KiB;
/// the remaining space covers fixed metadata and framing.
pub const MAX_BLIND_VAULT_MUTATION_FRAME_BYTES: u64 = 272 * 1024;

/// Maximum objects returned by one signed recovery page.
pub const MAX_BLIND_VAULT_PULL_OBJECTS: usize = 16;

/// Maximum opaque server cursor carried by a recovery frame.
pub const MAX_BLIND_VAULT_PULL_CURSOR_BYTES: usize = 128;

/// Maximum signed pull-response frame: sixteen 256 KiB ciphertext classes plus
/// bounded metadata. Public request handlers must retain their narrower body
/// limits and must not apply this response ceiling to attacker-controlled puts.
pub const MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES: u64 = 5 * 1024 * 1024;

/// Absolute largest v1 frame accepted by the generic decoder.
pub const MAX_BLIND_VAULT_FRAME_BYTES: u64 = MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES;

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
    /// Client request to create one anonymous, replica-local lease.
    LeaseCreate(BlindVaultLeaseCreateRequest),
    /// Lease administrator request to remove one immutable object.
    Delete(BlindVaultDeleteRequest),
    /// Node proof that an exact object was removed or already absent.
    DeletedReceipt(BlindVaultDeletedReceipt),
    /// One-time bearer admission ticket plus a self-authenticating lease.
    LeaseAdmission(BlindVaultLeaseAdmissionRequest),
    /// Capability-authenticated request for one stable encrypted-object page.
    PullRequest(BlindVaultPullRequest),
    /// Node-signed stable encrypted-object page.
    PullResponse(BlindVaultPullResponse),
    /// RFC 9474 blind-issued one-time credential plus anonymous lease.
    BlindLeaseAdmission(BlindVaultBlindLeaseAdmissionRequest),
    /// Node-signed public blind-admission key epochs and coarse policy.
    BlindIssuerDirectory(BlindVaultBlindIssuerDirectory),
    /// Terminal-signed proof that one blind-issued lease was accepted.
    BlindLeaseAccepted(BlindVaultBlindLeaseAcceptedReceipt),
    /// Administration-key request to retire one complete replica lease.
    LeaseRetire(BlindVaultLeaseRetireRequest),
    /// Terminal-signed proof that one complete replica lease was retired.
    LeaseRetiredReceipt(BlindVaultLeaseRetiredReceipt),
    /// Blind-issued capacity authorization plus an administration-key renewal.
    BlindLeaseRenewal(BlindVaultBlindLeaseRenewalRequest),
    /// Terminal-signed proof that one authorized renewal was committed.
    BlindLeaseRenewed(BlindVaultBlindLeaseRenewedReceipt),
    /// Administration-key request for one encrypted replica-local status view.
    LeaseStatus(BlindVaultLeaseStatusRequest),
    /// Terminal-signed proof of one coherent lease status observation.
    LeaseStatusReceipt(BlindVaultLeaseStatusReceipt),
    /// Administration-key request for one private live-set commitment.
    LeaseInventory(BlindVaultLeaseInventoryRequest),
    /// Terminal-signed proof of one coherent encrypted-object inventory.
    LeaseInventoryReceipt(BlindVaultLeaseInventoryReceipt),
    /// Coarse workload failure visible only after source-side reply decryption.
    TerminalFailure(BlindVaultTerminalFailure),
}

/// Blind Vault operation answered by one encrypted terminal failure.
///
/// [BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] Stable one-byte
/// operation values keep the bincode body independent from Rust enum ordering.
/// They identify only the operation already bound into the signed onion reply;
/// lease, object, route, endpoint, and application metadata remain absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlindVaultTerminalOperation {
    /// Immutable ciphertext write.
    Put = 1,
    /// Capability-authenticated ciphertext recovery.
    Pull = 2,
    /// Administration-authorized object deletion.
    Delete = 3,
    /// Blind-issued anonymous lease admission.
    LeaseAdmission = 4,
    /// Blind-authorized lease renewal.
    LeaseRenewal = 5,
    /// Private lease status observation.
    LeaseStatus = 6,
    /// Private encrypted-object inventory observation.
    LeaseInventory = 7,
    /// Complete anonymous lease retirement.
    LeaseRetire = 8,
}

impl TryFrom<u8> for BlindVaultTerminalOperation {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Put),
            2 => Ok(Self::Pull),
            3 => Ok(Self::Delete),
            4 => Ok(Self::LeaseAdmission),
            5 => Ok(Self::LeaseRenewal),
            6 => Ok(Self::LeaseStatus),
            7 => Ok(Self::LeaseInventory),
            8 => Ok(Self::LeaseRetire),
            _ => Err("unknown blind vault terminal operation"),
        }
    }
}

impl Serialize for BlindVaultTerminalOperation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for BlindVaultTerminalOperation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

/// Stable source-action class for an encrypted terminal failure.
///
/// Codes are deliberately coarse. In particular, every capability, lease,
/// signature, cursor, object-state, and replay rejection collapses to
/// [`Self::Rejected`] so a client cannot turn the storage node into an oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlindVaultTerminalFailureCode {
    /// The request is invalid or permanently unauthorized; do not retry it.
    Rejected = 1,
    /// The selected replica lacks capacity; choose another audited terminal.
    Capacity = 2,
    /// The terminal is temporarily unavailable; bounded retry is permitted.
    Unavailable = 3,
    /// The selected inline response class cannot carry the requested result.
    ResponseTooLarge = 4,
}

impl BlindVaultTerminalFailureCode {
    /// Stable operator-safe label without request or storage details.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rejected => "rejected",
            Self::Capacity => "capacity",
            Self::Unavailable => "unavailable",
            Self::ResponseTooLarge => "response_too_large",
        }
    }
}

impl std::fmt::Display for BlindVaultTerminalFailureCode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl TryFrom<u8> for BlindVaultTerminalFailureCode {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Rejected),
            2 => Ok(Self::Capacity),
            3 => Ok(Self::Unavailable),
            4 => Ok(Self::ResponseTooLarge),
            _ => Err("unknown blind vault terminal failure code"),
        }
    }
}

impl Serialize for BlindVaultTerminalFailureCode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u8(*self as u8)
    }
}

impl<'de> Deserialize<'de> for BlindVaultTerminalFailureCode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        Self::try_from(value).map_err(serde::de::Error::custom)
    }
}

/// Authenticated workload failure carried inside a sealed onion reply.
///
/// The outer [`OnionReplySession`] verifies the terminal signature and binds
/// this body to the exact request commitment before a caller can inspect it.
/// No separate inner signature or request identifier is needed, and adding
/// either would create unnecessary correlation material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultTerminalFailure {
    operation: BlindVaultTerminalOperation,
    code: BlindVaultTerminalFailureCode,
}

impl BlindVaultTerminalFailure {
    /// Creates one minimal encrypted terminal failure.
    #[must_use]
    pub const fn new(
        operation: BlindVaultTerminalOperation,
        code: BlindVaultTerminalFailureCode,
    ) -> Self {
        Self { operation, code }
    }

    /// Operation already bound into the encrypted request and signed reply.
    #[must_use]
    pub const fn operation(&self) -> BlindVaultTerminalOperation {
        self.operation
    }

    /// Coarse source action; never a storage-engine or authorization detail.
    #[must_use]
    pub const fn code(&self) -> BlindVaultTerminalFailureCode {
        self.code
    }
}

/// Anonymous lease metadata signed by its independent administration key.
///
/// Admission and quota policy are deliberately outside this structure. A node
/// must authenticate or rate-limit lease creation separately without adding an
/// account identity to the durable lease record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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

/// One public RFC 9474 issuer key and its node-enforced coarse policy.
///
/// The key is public by design. No issuer URL, account scope, product tier, or
/// issuance transcript belongs in this storage-node directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindIssuerEpoch {
    /// Blind-admission scheme version accepted under this key.
    pub admission_version: u16,
    /// SHA-256 fingerprint of `public_key_der`.
    pub issuer_key_id: [u8; 32],
    /// Canonical SPKI DER for the RSA public key.
    pub public_key_der: Vec<u8>,
    /// Inclusive activation time in Unix milliseconds.
    pub not_before_ms: u64,
    /// Exclusive expiry time in Unix milliseconds.
    pub expires_at_ms: u64,
    /// Maximum anonymous lease lifetime authorized by this key epoch.
    pub max_lease_ttl_ms: u64,
}

impl BlindVaultBlindIssuerEpoch {
    /// Builds an epoch and derives its stable key fingerprint.
    #[must_use]
    pub fn new(
        public_key_der: Vec<u8>,
        not_before_ms: u64,
        expires_at_ms: u64,
        max_lease_ttl_ms: u64,
    ) -> Self {
        let issuer_key_id = sha256(&public_key_der);
        Self {
            admission_version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id,
            public_key_der,
            not_before_ms,
            expires_at_ms,
            max_lease_ttl_ms,
        }
    }

    fn validate_at(&self, generated_at_ms: u64) -> Result<(), BlindVaultError> {
        if self.admission_version != BLIND_VAULT_BLIND_ADMISSION_VERSION {
            return Err(BlindVaultError::UnsupportedBlindAdmissionVersion(
                self.admission_version,
            ));
        }
        require_non_zero("blind_issuer_key_id", &self.issuer_key_id)?;
        if self.public_key_der.is_empty()
            || self.public_key_der.len() > MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES
        {
            return Err(BlindVaultError::InvalidBlindIssuerKeyLength {
                actual: self.public_key_der.len(),
            });
        }
        if sha256(&self.public_key_der) != self.issuer_key_id {
            return Err(BlindVaultError::BlindIssuerKeyIdMismatch);
        }
        let epoch_lifetime_ms = self
            .expires_at_ms
            .checked_sub(self.not_before_ms)
            .ok_or(BlindVaultError::InvalidBlindIssuerEpochPolicy)?;
        if epoch_lifetime_ms == 0
            || epoch_lifetime_ms > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS
            || self.max_lease_ttl_ms == 0
            || self.expires_at_ms <= generated_at_ms
        {
            return Err(BlindVaultError::InvalidBlindIssuerEpochPolicy);
        }
        Ok(())
    }
}

/// Authenticated discovery response for blind-admission key rotation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindIssuerDirectory {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Response creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Descriptor identity of the responding storage node.
    pub node_id: [u8; 32],
    /// Strictly key-ID-sorted active and pre-announced future epochs.
    pub epochs: Vec<BlindVaultBlindIssuerEpoch>,
    /// Ed25519 signature by `node_id` over all directory fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultBlindIssuerDirectory {
    /// Builds an unsigned issuer directory.
    #[must_use]
    pub const fn new(
        generated_at_ms: u64,
        node_id: [u8; 32],
        epochs: Vec<BlindVaultBlindIssuerEpoch>,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            generated_at_ms,
            node_id,
            epochs,
            signature: [0; 64],
        }
    }

    /// Canonical key-directory signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let epoch_bytes = blind_issuer_epoch_bytes_capacity(&self.epochs);
        let mut bytes =
            Vec::with_capacity(BLIND_ISSUER_DIRECTORY_SIGNING_DOMAIN.len() + 44 + epoch_bytes);
        bytes.extend_from_slice(BLIND_ISSUER_DIRECTORY_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.generated_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        append_blind_issuer_epochs(&mut bytes, &self.epochs);
        bytes
    }

    /// Validates and signs the directory with the node descriptor identity.
    ///
    /// # Errors
    /// Returns an invariant error for malformed epochs or a node-identity
    /// mismatch.
    pub fn sign(&mut self, node_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Verifies bounds, freshness, expected node identity, and signature.
    ///
    /// # Errors
    /// Returns an invariant, freshness, identity, or signature error when the
    /// directory cannot be trusted.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_age_ms: u64,
        maximum_clock_skew_ms: u64,
        node_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if maximum_age_ms == 0
            || self.generated_at_ms > now_ms.saturating_add(maximum_clock_skew_ms)
            || now_ms.saturating_sub(self.generated_at_ms) > maximum_age_ms
        {
            return Err(BlindVaultError::IssuerDirectoryTimestampOutsideWindow);
        }
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("blind_issuer_directory_node_id", &self.node_id)?;
        if self.epochs.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
            return Err(BlindVaultError::TooManyBlindIssuerEpochs);
        }
        for epoch in &self.epochs {
            epoch.validate_at(self.generated_at_ms)?;
        }
        if self
            .epochs
            .windows(2)
            .any(|pair| pair[0].issuer_key_id >= pair[1].issuer_key_id)
        {
            return Err(BlindVaultError::BlindIssuerEpochOrderInvalid);
        }
        Ok(())
    }
}

/// [BLIND-VAULT-ISSUER-UPDATE 2026-07-23 by Codex]
/// Authority-signed public issuer generation accepted by storage nodes.
///
/// This object is transport-independent: a backend management channel, an
/// offline operator tool, or a future node synchronization layer may carry the
/// exact same signed bytes. It contains no issuer private key or issuance,
/// account, wallet, lease, storage, or client-network identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindIssuerUpdate {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Strictly increasing control-plane generation, starting at one.
    pub generation: u64,
    /// Update creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Ed25519 authority identity explicitly pinned by the node operator.
    pub authority_id: [u8; 32],
    /// Strictly key-ID-sorted active and pre-announced future epochs.
    pub epochs: Vec<BlindVaultBlindIssuerEpoch>,
    /// Ed25519 authority signature over every preceding field.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultBlindIssuerUpdate {
    /// Builds one unsigned issuer update.
    #[must_use]
    pub const fn new(
        generation: u64,
        generated_at_ms: u64,
        authority_id: [u8; 32],
        epochs: Vec<BlindVaultBlindIssuerEpoch>,
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            generation,
            generated_at_ms,
            authority_id,
            epochs,
            signature: [0; 64],
        }
    }

    /// Canonical authority-signing input.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let epoch_bytes = blind_issuer_epoch_bytes_capacity(&self.epochs);
        let mut bytes =
            Vec::with_capacity(BLIND_ISSUER_UPDATE_SIGNING_DOMAIN.len() + 52 + epoch_bytes);
        bytes.extend_from_slice(BLIND_ISSUER_UPDATE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.generation.to_be_bytes());
        bytes.extend_from_slice(&self.generated_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.authority_id);
        append_blind_issuer_epochs(&mut bytes, &self.epochs);
        bytes
    }

    /// Validates and signs this update with its declared authority identity.
    ///
    /// # Errors
    /// Returns a bounded invariant or identity error for malformed input.
    pub fn sign(&mut self, authority_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.authority_id != authority_key.public_key_bytes() {
            return Err(BlindVaultError::BlindIssuerAuthorityMismatch);
        }
        self.signature = authority_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Verifies bounds, freshness, pinned authority identity, and signature.
    ///
    /// # Errors
    /// Returns a fail-closed protocol error for stale, forged, or malformed
    /// updates.
    pub fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_age_ms: u64,
        maximum_clock_skew_ms: u64,
        authority_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if maximum_age_ms == 0
            || self.generated_at_ms > now_ms.saturating_add(maximum_clock_skew_ms)
            || now_ms.saturating_sub(self.generated_at_ms) > maximum_age_ms
        {
            return Err(BlindVaultError::BlindIssuerUpdateTimestampOutsideWindow);
        }
        if self.authority_id != authority_key.to_bytes() {
            return Err(BlindVaultError::BlindIssuerAuthorityMismatch);
        }
        authority_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("blind_issuer_update_authority_id", &self.authority_id)?;
        if self.generation == 0 {
            return Err(BlindVaultError::InvalidBlindIssuerUpdateGeneration);
        }
        if self.epochs.is_empty() {
            return Err(BlindVaultError::BlindIssuerUpdateHasNoEpochs);
        }
        if self.epochs.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
            return Err(BlindVaultError::TooManyBlindIssuerEpochs);
        }
        for epoch in &self.epochs {
            epoch.validate_at(self.generated_at_ms)?;
        }
        if self
            .epochs
            .windows(2)
            .any(|pair| pair[0].issuer_key_id >= pair[1].issuer_key_id)
        {
            return Err(BlindVaultError::BlindIssuerEpochOrderInvalid);
        }
        Ok(())
    }
}

fn blind_issuer_epoch_bytes_capacity(epochs: &[BlindVaultBlindIssuerEpoch]) -> usize {
    epochs
        .iter()
        .map(|epoch| 90usize.saturating_add(epoch.public_key_der.len()))
        .sum()
}

fn append_blind_issuer_epochs(bytes: &mut Vec<u8>, epochs: &[BlindVaultBlindIssuerEpoch]) {
    let epoch_count = u16::try_from(epochs.len()).unwrap_or(u16::MAX);
    bytes.extend_from_slice(&epoch_count.to_be_bytes());
    for epoch in epochs {
        bytes.extend_from_slice(&epoch.admission_version.to_be_bytes());
        bytes.extend_from_slice(&epoch.issuer_key_id);
        let der_length = u16::try_from(epoch.public_key_der.len()).unwrap_or(u16::MAX);
        bytes.extend_from_slice(&der_length.to_be_bytes());
        bytes.extend_from_slice(&epoch.public_key_der);
        bytes.extend_from_slice(&epoch.not_before_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.max_lease_ttl_ms.to_be_bytes());
    }
}

/// Capability-authenticated request for one stable recovery snapshot page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultPullRequest {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Random replica-local lease identifier.
    pub lease_id: [u8; 32],
    /// Random bearer read capability whose hash was committed at lease setup.
    pub read_capability: [u8; 32],
    /// Empty for the first page; otherwise a node-encrypted snapshot cursor.
    pub continuation_cursor: Vec<u8>,
    /// Requested page size, bounded again by node policy.
    pub limit: u16,
}

impl BlindVaultPullRequest {
    /// Validates fixed identifiers and protocol-wide recovery bounds.
    pub fn validate(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        require_non_zero("read_capability", &self.read_capability)?;
        if self.limit == 0 || usize::from(self.limit) > MAX_BLIND_VAULT_PULL_OBJECTS {
            return Err(BlindVaultError::InvalidPullLimit);
        }
        if self.continuation_cursor.len() > MAX_BLIND_VAULT_PULL_CURSOR_BYTES {
            return Err(BlindVaultError::InvalidPullCursorLength);
        }
        Ok(())
    }
}

/// One immutable ciphertext object returned by a Blind Vault recovery page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultRecoveredObject {
    /// Replica-local immutable object identifier.
    pub object_id: [u8; 32],
    /// Exact client ciphertext including its AEAD nonce, tag, and padding.
    pub ciphertext: Vec<u8>,
    /// SHA-256 commitment to the exact ciphertext bytes.
    pub ciphertext_commitment: [u8; 32],
    /// Object retention deadline in Unix milliseconds.
    pub expires_at_ms: u64,
}

impl BlindVaultRecoveredObject {
    fn validate_at(&self, generated_at_ms: u64) -> Result<(), BlindVaultError> {
        require_non_zero("object_id", &self.object_id)?;
        if !BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES.contains(&self.ciphertext.len()) {
            return Err(BlindVaultError::InvalidCiphertextSize {
                actual: self.ciphertext.len(),
            });
        }
        if sha256(&self.ciphertext) != self.ciphertext_commitment {
            return Err(BlindVaultError::CommitmentMismatch);
        }
        if self.expires_at_ms <= generated_at_ms {
            return Err(BlindVaultError::Expired);
        }
        Ok(())
    }
}

/// Node-signed bounded recovery page.
///
/// The signature authenticates ciphertext commitments rather than hashing the
/// multi-megabyte ciphertext a second time. Validation recomputes each stored
/// commitment first, so the signature remains bound to the exact bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultPullResponse {
    /// Independent Blind Vault protocol version.
    pub version: u16,
    /// Replica-local lease answered by this page.
    pub lease_id: [u8; 32],
    /// Bounded immutable ciphertext objects in snapshot order.
    pub objects: Vec<BlindVaultRecoveredObject>,
    /// Empty when this snapshot is complete; otherwise node-encrypted cursor.
    pub continuation_cursor: Vec<u8>,
    /// Node generation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Descriptor identity of the responding storage node.
    pub node_id: [u8; 32],
    /// Ed25519 signature by `node_id` over page commitments and metadata.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl BlindVaultPullResponse {
    /// Builds an unsigned recovery page.
    #[must_use]
    pub fn new(
        lease_id: [u8; 32],
        objects: Vec<BlindVaultRecoveredObject>,
        continuation_cursor: Vec<u8>,
        generated_at_ms: u64,
        node_id: [u8; 32],
    ) -> Self {
        Self {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id,
            objects,
            continuation_cursor,
            generated_at_ms,
            node_id,
            signature: [0; 64],
        }
    }

    /// Canonical signing input containing each already-validated ciphertext
    /// commitment, object metadata, and opaque continuation cursor.
    #[must_use]
    pub fn signing_bytes(&self) -> Vec<u8> {
        let object_bytes = self.objects.len().saturating_mul(76);
        let mut bytes = Vec::with_capacity(
            PULL_RESPONSE_SIGNING_DOMAIN.len() + 76 + object_bytes + self.continuation_cursor.len(),
        );
        bytes.extend_from_slice(PULL_RESPONSE_SIGNING_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.lease_id);
        bytes.extend_from_slice(&(self.objects.len() as u16).to_be_bytes());
        for object in &self.objects {
            bytes.extend_from_slice(&object.object_id);
            bytes.extend_from_slice(&(object.ciphertext.len() as u32).to_be_bytes());
            bytes.extend_from_slice(&object.ciphertext_commitment);
            bytes.extend_from_slice(&object.expires_at_ms.to_be_bytes());
        }
        bytes.extend_from_slice(&(self.continuation_cursor.len() as u16).to_be_bytes());
        bytes.extend_from_slice(&self.continuation_cursor);
        bytes.extend_from_slice(&self.generated_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.node_id);
        bytes
    }

    /// Validates page bounds and signs with the responding node identity.
    pub fn sign(&mut self, node_key: &IdentityKeyPair) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.public_key_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        self.signature = node_key.sign(&self.signing_bytes());
        Ok(())
    }

    /// Validates all ciphertext commitments, node identity, and page signature.
    pub fn validate_and_verify(&self, node_key: &IdentityPublicKey) -> Result<(), BlindVaultError> {
        self.validate_fields()?;
        if self.node_id != node_key.to_bytes() {
            return Err(BlindVaultError::NodeIdentityMismatch);
        }
        node_key
            .verify(&self.signing_bytes(), &self.signature)
            .map_err(|_| BlindVaultError::InvalidSignature)
    }

    fn validate_fields(&self) -> Result<(), BlindVaultError> {
        require_version(self.version)?;
        require_non_zero("lease_id", &self.lease_id)?;
        if self.objects.len() > MAX_BLIND_VAULT_PULL_OBJECTS {
            return Err(BlindVaultError::InvalidPullLimit);
        }
        if self.continuation_cursor.len() > MAX_BLIND_VAULT_PULL_CURSOR_BYTES {
            return Err(BlindVaultError::InvalidPullCursorLength);
        }
        for object in &self.objects {
            object.validate_at(self.generated_at_ms)?;
        }
        Ok(())
    }
}

/// Source-owned state for one anonymous inline Blind Vault recovery request.
///
/// [BLIND-VAULT-ONION-PULL-SESSION 2026-08-28 by Codex] Capability bytes are
/// retained only inside the encoded final onion payload; the session keeps a
/// request commitment and single-use reply key rather than a plaintext copy.
///
/// The encoded request returned by [`Self::prepare`] is the final payload for
/// `build_onion_envelope`; it must never be sent directly to a storage node.
/// The retained state contains the single-use reply private key and is consumed
/// when opening one response.
pub struct BlindVaultOnionPullSession {
    lease_id: [u8; 32],
    reply_session: OnionReplySession,
}

impl BlindVaultOnionPullSession {
    /// Encodes one capability-bearing pull for the final onion layer.
    pub fn prepare(
        route_id: [u8; 16],
        expected_terminal_node_id: [u8; 32],
        request: BlindVaultPullRequest,
    ) -> Result<(Vec<u8>, Self), BlindVaultOnionPullError> {
        request.validate()?;
        if request.limit != 1 {
            return Err(BlindVaultOnionPullError::UnsupportedInlinePullLimit);
        }
        let lease_id = request.lease_id;
        let encoded_pull = encode_blind_vault_frame(&BlindVaultFrame::PullRequest(request))?;
        let (reply_request, reply_session) = OnionReplySession::prepare(
            route_id,
            expected_terminal_node_id,
            ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            encoded_pull,
        )?;
        let encoded_request = encode_onion_reply_request(&reply_request)?;
        Ok((
            encoded_request,
            Self {
                lease_id,
                reply_session,
            },
        ))
    }

    /// Opens the terminal response and verifies the inner signed recovery page.
    pub fn open(
        self,
        encoded_response: &[u8],
    ) -> Result<BlindVaultPullResponse, BlindVaultOnionPullError> {
        let reply = self.reply_session.open(encoded_response)?;
        let frame = decode_blind_vault_frame(&reply.payload)?;
        if let BlindVaultFrame::TerminalFailure(failure) = &frame {
            if failure.operation() != BlindVaultTerminalOperation::Pull {
                return Err(BlindVaultOnionPullError::UnexpectedResponseFrame);
            }
            return Err(BlindVaultOnionPullError::TerminalFailure(failure.code()));
        }
        let BlindVaultFrame::PullResponse(response) = frame else {
            return Err(BlindVaultOnionPullError::UnexpectedResponseFrame);
        };
        let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
            .map_err(|_| BlindVaultOnionPullError::InvalidTerminalIdentity)?;
        response.validate_and_verify(&terminal_key)?;
        if response.lease_id != self.lease_id {
            return Err(BlindVaultOnionPullError::LeaseMismatch);
        }
        Ok(response)
    }
}

/// Fail-closed source errors for anonymous inline Blind Vault recovery.
#[derive(Debug, Error)]
pub enum BlindVaultOnionPullError {
    /// The Blind Vault request or signed response violated its wire contract.
    #[error("blind vault onion pull frame rejected")]
    BlindVault(#[from] BlindVaultError),
    /// The reply carrier failed key, route, identity, size, or signature checks.
    #[error("blind vault onion reply rejected")]
    OnionReply(#[from] OnionReplyError),
    /// The authenticated terminal returned a coarse encrypted failure.
    #[error("blind vault onion pull terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// Inline v1 deliberately returns at most one ciphertext object.
    #[error("blind vault onion pull requires limit 1")]
    UnsupportedInlinePullLimit,
    /// The decrypted workload response was not a Blind Vault pull page.
    #[error("unexpected blind vault onion response frame")]
    UnexpectedResponseFrame,
    /// The verified page belongs to another replica-local lease.
    #[error("blind vault onion pull lease mismatch")]
    LeaseMismatch,
    /// The verified outer terminal identity could not be reconstructed.
    #[error("invalid blind vault onion terminal identity")]
    InvalidTerminalIdentity,
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
        validate_future_deadline(now_ms, self.expires_at_ms, maximum_ttl_ms)
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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

/// Administration-key request to delete one opaque object.
///
/// Nodes retain only a bounded tombstone needed for idempotent retries; no
/// application deletion reason or account identifier belongs on this frame.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Administration-key request to retire one complete replica-local lease.
///
/// [BLIND-VAULT-LEASE-RETIRE 2026-08-28 by Codex] Retirement removes every
/// ciphertext object and object tombstone under the lease in one transaction.
/// No user identity, application reason, or cross-replica identifier belongs
/// in this request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Administration-key request to extend one live replica-local lease.
///
/// [BLIND-VAULT-LEASE-RENEWAL 2026-08-28 by Codex] The signed request binds
/// both the expected current expiry and requested next expiry. This makes a
/// stale or reordered renewal fail closed instead of silently extending a
/// different lease generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindVaultBlindLeaseRenewalRequest {
    /// New RFC 9474 one-time credential authorizing additional retention.
    pub admission: BlindVaultBlindAdmissionToken,
    /// Existing lease transition signed by its independent administration key.
    pub renewal: BlindVaultLeaseRenewRequest,
}

/// Terminal-signed proof that one blind-authorized renewal was committed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Administration-key request for one replica-local lease observation.
///
/// [BLIND-VAULT-LEASE-STATUS 2026-08-28 by Codex] The query is carried only
/// inside the encrypted terminal layer. It authorizes disclosure of aggregate
/// lease-local retention state to the administrator without publishing a
/// lease identifier, activity timestamp, or usage count to route relays.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// One canonical encrypted-object entry committed by a private inventory.
///
/// This type is deliberately not serializable as a protocol frame. It exists
/// only to let a source and terminal independently derive the same per-replica
/// commitment without publishing an object list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

/// Source-owned expected state for one independently wrapped replica.
///
/// This type is deliberately not serializable. Replica object identifiers and
/// commitments are local repair material and must never become public-chain,
/// discovery, or cross-replica correlation metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaManifestExpectation {
    node_id: [u8; 32],
    lease_id: [u8; 32],
    object_count: u64,
    ciphertext_bytes: u64,
    inventory_commitment: [u8; 32],
}

impl BlindVaultReplicaManifestExpectation {
    /// Creates one validated source-side replica expectation.
    pub fn new(
        node_id: [u8; 32],
        lease_id: [u8; 32],
        object_count: u64,
        ciphertext_bytes: u64,
        inventory_commitment: [u8; 32],
    ) -> Result<Self, BlindVaultReplicaEvidenceError> {
        require_non_zero("node_id", &node_id)?;
        require_non_zero("lease_id", &lease_id)?;
        require_non_zero("inventory_commitment", &inventory_commitment)?;
        if (object_count == 0) != (ciphertext_bytes == 0) {
            return Err(BlindVaultReplicaEvidenceError::InvalidExpectation);
        }
        Ok(Self {
            node_id,
            lease_id,
            object_count,
            ciphertext_bytes,
            inventory_commitment,
        })
    }

    /// Descriptor identity expected to sign this replica's observation.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Replica-local lease identifier.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Expected number of still-live encrypted objects.
    #[must_use]
    pub const fn object_count(&self) -> u64 {
        self.object_count
    }

    /// Expected total padded ciphertext bytes.
    #[must_use]
    pub const fn ciphertext_bytes(&self) -> u64 {
        self.ciphertext_bytes
    }

    /// Expected domain-separated root for this replica's own manifest.
    #[must_use]
    pub const fn inventory_commitment(&self) -> [u8; 32] {
        self.inventory_commitment
    }
}

/// Verified, freshness-bounded inventory evidence for one replica.
///
/// Construction requires the exact request, expected terminal identity, and
/// source-owned manifest. A valid but divergent receipt remains evidence: the
/// planner classifies it for reconciliation instead of confusing divergence
/// with an invalid signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedReplicaInventory {
    node_id: [u8; 32],
    lease_id: [u8; 32],
    expires_at_ms: u64,
    observed_at_ms: u64,
    expected_object_count: u64,
    observed_object_count: u64,
    expected_ciphertext_bytes: u64,
    observed_ciphertext_bytes: u64,
    expected_inventory_commitment: [u8; 32],
    observed_inventory_commitment: [u8; 32],
    matches_expected_manifest: bool,
}

impl BlindVaultVerifiedReplicaInventory {
    /// Verifies one terminal receipt against its exact request and local
    /// per-replica expectation.
    ///
    /// `maximum_receipt_age_ms` must be non-zero. Future observations are
    /// accepted only within `maximum_future_clock_skew_ms`. Lease expiry is not
    /// rejected here because the planner must convert signed expired evidence
    /// into an explicit replacement action.
    pub fn verify(
        receipt: &BlindVaultLeaseInventoryReceipt,
        request: &BlindVaultLeaseInventoryRequest,
        expectation: &BlindVaultReplicaManifestExpectation,
        now_ms: u64,
        maximum_receipt_age_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaEvidenceError> {
        if now_ms == 0 || maximum_receipt_age_ms == 0 {
            return Err(BlindVaultReplicaEvidenceError::InvalidFreshnessPolicy);
        }
        if request.lease_id != expectation.lease_id {
            return Err(BlindVaultReplicaEvidenceError::RequestMismatch);
        }
        if receipt.node_id != expectation.node_id {
            return Err(BlindVaultReplicaEvidenceError::TerminalIdentityMismatch);
        }
        let terminal_key = IdentityPublicKey::from_bytes(&expectation.node_id)
            .map_err(|_| BlindVaultReplicaEvidenceError::InvalidTerminalIdentity)?;
        receipt.validate_and_verify(&terminal_key)?;
        if !receipt.matches_inventory(request) {
            return Err(BlindVaultReplicaEvidenceError::RequestMismatch);
        }

        if receipt.observed_at_ms > now_ms {
            if receipt.observed_at_ms - now_ms > maximum_future_clock_skew_ms {
                return Err(BlindVaultReplicaEvidenceError::ReceiptFromFuture);
            }
        } else if now_ms - receipt.observed_at_ms > maximum_receipt_age_ms {
            return Err(BlindVaultReplicaEvidenceError::ReceiptStale);
        }

        let matches_expected_manifest = receipt.live_object_count == expectation.object_count
            && receipt.live_ciphertext_bytes == expectation.ciphertext_bytes
            && receipt.inventory_commitment == expectation.inventory_commitment;
        Ok(Self {
            node_id: expectation.node_id,
            lease_id: expectation.lease_id,
            expires_at_ms: receipt.expires_at_ms,
            observed_at_ms: receipt.observed_at_ms,
            expected_object_count: expectation.object_count,
            observed_object_count: receipt.live_object_count,
            expected_ciphertext_bytes: expectation.ciphertext_bytes,
            observed_ciphertext_bytes: receipt.live_ciphertext_bytes,
            expected_inventory_commitment: expectation.inventory_commitment,
            observed_inventory_commitment: receipt.inventory_commitment,
            matches_expected_manifest,
        })
    }

    /// Descriptor identity that signed the evidence.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Replica-local lease identifier.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Signed lease expiry observed at the terminal.
    #[must_use]
    pub const fn expires_at_ms(&self) -> u64 {
        self.expires_at_ms
    }

    /// Signed terminal observation time.
    #[must_use]
    pub const fn observed_at_ms(&self) -> u64 {
        self.observed_at_ms
    }

    /// Whether all signed aggregate fields match this replica's expectation.
    #[must_use]
    pub const fn matches_expected_manifest(&self) -> bool {
        self.matches_expected_manifest
    }

    /// Source-owned expected live object count used to bind repair evidence to
    /// the exact planner generation.
    #[must_use]
    pub const fn expected_object_count(&self) -> u64 {
        self.expected_object_count
    }

    /// Source-owned expected padded ciphertext bytes for this replica only.
    #[must_use]
    pub const fn expected_ciphertext_bytes(&self) -> u64 {
        self.expected_ciphertext_bytes
    }

    /// Source-owned per-replica manifest root. This remains local-only and is
    /// never serialized into discovery or public ledger state.
    // [BLIND-VAULT-REPLICA-WORKFLOW 2026-08-28 by Codex] Reconciliation must
    // bind to the planner's exact expectation, not merely the node and lease.
    #[must_use]
    pub const fn expected_inventory_commitment(&self) -> [u8; 32] {
        self.expected_inventory_commitment
    }
}

/// Most recent source-owned evidence for one intended replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaEvidence {
    /// A terminal-signed, request-bound, freshness-checked inventory.
    Observed(BlindVaultVerifiedReplicaInventory),
    /// No valid inventory could be obtained for this intended member.
    Unavailable {
        /// Expected descriptor identity.
        node_id: [u8; 32],
        /// Replica-local lease identifier.
        lease_id: [u8; 32],
        /// Consecutive completed observation attempts that failed.
        consecutive_failures: u16,
    },
}

impl BlindVaultReplicaEvidence {
    /// Expected descriptor identity for deterministic planning and deduplication.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        match self {
            Self::Observed(inventory) => inventory.node_id,
            Self::Unavailable { node_id, .. } => *node_id,
        }
    }

    /// Replica-local lease identifier for duplicate detection.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        match self {
            Self::Observed(inventory) => inventory.lease_id,
            Self::Unavailable { lease_id, .. } => *lease_id,
        }
    }
}

/// Source policy for maintaining independently wrapped replicas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaPolicy {
    /// Desired replica floor. Transitional extra replicas are permitted.
    pub target_replicas: u8,
    /// Minimum live replicas whose signed inventory matches local expectations.
    pub minimum_healthy_replicas: u8,
    /// Lead time before expiry at which a live lease should be renewed.
    pub renewal_lead_ms: u64,
    /// Consecutive failed observations before replacing a replica.
    pub replace_after_consecutive_failures: u16,
}

impl BlindVaultReplicaPolicy {
    fn validate(self) -> Result<(), BlindVaultReplicaPlanError> {
        let target_replicas = usize::from(self.target_replicas);
        if target_replicas == 0
            || target_replicas > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS
            || self.minimum_healthy_replicas == 0
            || self.minimum_healthy_replicas > self.target_replicas
            || self.renewal_lead_ms == 0
            || self.replace_after_consecutive_failures == 0
        {
            return Err(BlindVaultReplicaPlanError::InvalidPolicy);
        }
        Ok(())
    }
}

/// Aggregate safety state of a source's current replica set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaPlanHealth {
    /// Matching live evidence meets policy and no action is due.
    Healthy,
    /// Matching live evidence meets policy and only lease renewal is due.
    MaintenanceDue,
    /// Matching live evidence meets policy but repair or membership work is due.
    Degraded,
    /// Too few live, matching replicas exist to satisfy the configured quorum.
    QuorumUnavailable,
}

/// One declarative source-side lifecycle action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaAction {
    /// Renew one still-live lease before its current generation expires.
    RenewLease {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_expires_at_ms: u64,
    },
    /// Rebuild one divergent replica from this source's private manifest.
    ReconcileInventory {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_object_count: u64,
        observed_object_count: u64,
        expected_ciphertext_bytes: u64,
        observed_ciphertext_bytes: u64,
        expected_inventory_commitment: [u8; 32],
        observed_inventory_commitment: [u8; 32],
    },
    /// Retry a temporarily unavailable member without changing membership.
    RetryObservation {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    /// Replace an expired or repeatedly unreachable member.
    ReplaceReplica {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    /// Provision anonymous replicas until the configured floor is restored.
    ProvisionReplicas { count: u8 },
}

/// Deterministic, declarative result of one replica lifecycle evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaPlan {
    /// Safety state after evaluating the supplied evidence.
    pub health: BlindVaultReplicaPlanHealth,
    /// Number of intended members represented by the supplied evidence.
    pub configured_replicas: u8,
    /// Number of verified observations for leases that have not expired.
    pub live_verified_replicas: u8,
    /// Number of live observations matching their own expected manifests.
    pub live_matching_replicas: u8,
    /// Deterministic local actions; no repair source is selected here.
    pub actions: Vec<BlindVaultReplicaAction>,
}

/// Replaceable capability for source-owned replica lifecycle planning.
pub trait BlindVaultReplicaPlanner {
    /// Produces a deterministic plan from already verified or unavailable
    /// per-replica evidence.
    fn plan(
        &self,
        now_ms: u64,
        evidence: &[BlindVaultReplicaEvidence],
    ) -> Result<BlindVaultReplicaPlan, BlindVaultReplicaPlanError>;
}

/// Manifest-bound planner that never infers truth from a replica majority.
///
/// [BLIND-VAULT-REPLICA-PLANNER 2026-08-28 by Codex] Each node is checked only
/// against its own independently randomized expected manifest. The planner
/// intentionally does not select a repair source: if matching live evidence
/// falls below policy, callers receive `QuorumUnavailable` and must not treat
/// another replica's object identifiers as authoritative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultManifestReplicaPlanner {
    policy: BlindVaultReplicaPolicy,
}

impl BlindVaultManifestReplicaPlanner {
    /// Creates a planner after validating all policy invariants.
    pub fn new(policy: BlindVaultReplicaPolicy) -> Result<Self, BlindVaultReplicaPlanError> {
        policy.validate()?;
        Ok(Self { policy })
    }

    /// Returns the immutable policy used for planning.
    #[must_use]
    pub const fn policy(&self) -> BlindVaultReplicaPolicy {
        self.policy
    }
}

impl BlindVaultReplicaPlanner for BlindVaultManifestReplicaPlanner {
    fn plan(
        &self,
        now_ms: u64,
        evidence: &[BlindVaultReplicaEvidence],
    ) -> Result<BlindVaultReplicaPlan, BlindVaultReplicaPlanError> {
        self.policy.validate()?;
        if now_ms == 0 {
            return Err(BlindVaultReplicaPlanError::TimestampOutOfRange);
        }
        if evidence.len() > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS {
            return Err(BlindVaultReplicaPlanError::TooManyReplicas {
                actual: evidence.len(),
            });
        }
        let renewal_deadline = now_ms
            .checked_add(self.policy.renewal_lead_ms)
            .ok_or(BlindVaultReplicaPlanError::TimestampOutOfRange)?;

        let mut node_ids = BTreeSet::new();
        let mut lease_ids = BTreeSet::new();
        for member in evidence {
            let node_id = member.node_id();
            let lease_id = member.lease_id();
            if node_id == [0; 32] || lease_id == [0; 32] {
                return Err(BlindVaultReplicaPlanError::InvalidEvidenceIdentity);
            }
            if !node_ids.insert(node_id) {
                return Err(BlindVaultReplicaPlanError::DuplicateNode);
            }
            if !lease_ids.insert(lease_id) {
                return Err(BlindVaultReplicaPlanError::DuplicateLease);
            }
            if matches!(
                member,
                BlindVaultReplicaEvidence::Unavailable {
                    consecutive_failures: 0,
                    ..
                }
            ) {
                return Err(BlindVaultReplicaPlanError::InvalidUnavailableEvidence);
            }
        }

        let mut ordered = evidence.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|member| member.node_id());
        let mut live_verified_replicas = 0_u8;
        let mut live_matching_replicas = 0_u8;
        let mut actions = Vec::new();

        for member in ordered {
            match member {
                BlindVaultReplicaEvidence::Observed(inventory) => {
                    if inventory.expires_at_ms <= now_ms {
                        actions.push(BlindVaultReplicaAction::ReplaceReplica {
                            node_id: inventory.node_id,
                            lease_id: inventory.lease_id,
                        });
                        continue;
                    }
                    live_verified_replicas += 1;
                    if inventory.matches_expected_manifest {
                        live_matching_replicas += 1;
                    } else {
                        actions.push(BlindVaultReplicaAction::ReconcileInventory {
                            node_id: inventory.node_id,
                            lease_id: inventory.lease_id,
                            expected_object_count: inventory.expected_object_count,
                            observed_object_count: inventory.observed_object_count,
                            expected_ciphertext_bytes: inventory.expected_ciphertext_bytes,
                            observed_ciphertext_bytes: inventory.observed_ciphertext_bytes,
                            expected_inventory_commitment: inventory.expected_inventory_commitment,
                            observed_inventory_commitment: inventory.observed_inventory_commitment,
                        });
                    }
                    if inventory.expires_at_ms <= renewal_deadline {
                        actions.push(BlindVaultReplicaAction::RenewLease {
                            node_id: inventory.node_id,
                            lease_id: inventory.lease_id,
                            expected_expires_at_ms: inventory.expires_at_ms,
                        });
                    }
                }
                BlindVaultReplicaEvidence::Unavailable {
                    node_id,
                    lease_id,
                    consecutive_failures,
                } => {
                    if *consecutive_failures >= self.policy.replace_after_consecutive_failures {
                        actions.push(BlindVaultReplicaAction::ReplaceReplica {
                            node_id: *node_id,
                            lease_id: *lease_id,
                        });
                    } else {
                        actions.push(BlindVaultReplicaAction::RetryObservation {
                            node_id: *node_id,
                            lease_id: *lease_id,
                        });
                    }
                }
            }
        }

        if evidence.len() < usize::from(self.policy.target_replicas) {
            let missing = usize::from(self.policy.target_replicas) - evidence.len();
            actions.push(BlindVaultReplicaAction::ProvisionReplicas {
                count: u8::try_from(missing)
                    .map_err(|_| BlindVaultReplicaPlanError::TooManyReplicas { actual: missing })?,
            });
        }

        let health = if live_matching_replicas < self.policy.minimum_healthy_replicas {
            BlindVaultReplicaPlanHealth::QuorumUnavailable
        } else if actions.is_empty() {
            BlindVaultReplicaPlanHealth::Healthy
        } else if actions
            .iter()
            .all(|action| matches!(action, BlindVaultReplicaAction::RenewLease { .. }))
        {
            BlindVaultReplicaPlanHealth::MaintenanceDue
        } else {
            BlindVaultReplicaPlanHealth::Degraded
        };

        Ok(BlindVaultReplicaPlan {
            health,
            configured_replicas: u8::try_from(evidence.len()).map_err(|_| {
                BlindVaultReplicaPlanError::TooManyReplicas {
                    actual: evidence.len(),
                }
            })?,
            live_verified_replicas,
            live_matching_replicas,
            actions,
        })
    }
}

/// Fail-closed verification errors for private replica evidence.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaEvidenceError {
    /// The underlying inventory contract was malformed or unauthenticated.
    #[error("blind vault replica evidence violated the inventory protocol")]
    BlindVault(#[from] BlindVaultError),
    /// Expected object and byte aggregates were internally inconsistent.
    #[error("blind vault replica manifest expectation is invalid")]
    InvalidExpectation,
    /// Receipt terminal identity did not match the intended replica member.
    #[error("blind vault replica terminal identity mismatch")]
    TerminalIdentityMismatch,
    /// The expected terminal identity could not be reconstructed as Ed25519.
    #[error("invalid blind vault replica terminal identity")]
    InvalidTerminalIdentity,
    /// Receipt did not answer the exact request and replica-local lease.
    #[error("blind vault replica inventory request mismatch")]
    RequestMismatch,
    /// Caller supplied an unusable local freshness policy.
    #[error("blind vault replica evidence freshness policy is invalid")]
    InvalidFreshnessPolicy,
    /// Receipt observation exceeded allowed future clock skew.
    #[error("blind vault replica inventory receipt is from the future")]
    ReceiptFromFuture,
    /// Receipt observation was older than the configured evidence lifetime.
    #[error("blind vault replica inventory receipt is stale")]
    ReceiptStale,
}

/// Invalid policy or evidence-set errors for replica lifecycle planning.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaPlanError {
    /// Policy quorum, target, renewal, or replacement values were invalid.
    #[error("blind vault replica policy is invalid")]
    InvalidPolicy,
    /// Evidence exceeded the bounded local planning set.
    #[error("blind vault replica set contains too many members: {actual}")]
    TooManyReplicas { actual: usize },
    /// Two evidence entries declared the same terminal descriptor identity.
    #[error("blind vault replica set contains a duplicate node")]
    DuplicateNode,
    /// Two evidence entries declared the same replica-local lease.
    #[error("blind vault replica set contains a duplicate lease")]
    DuplicateLease,
    /// An evidence member used an all-zero terminal or lease identity.
    #[error("blind vault replica evidence identity is invalid")]
    InvalidEvidenceIdentity,
    /// Unavailable evidence must represent at least one completed failure.
    #[error("blind vault unavailable replica evidence is invalid")]
    InvalidUnavailableEvidence,
    /// Planning time or renewal-window arithmetic was not representable.
    #[error("blind vault replica planning timestamp is out of range")]
    TimestampOutOfRange,
}

/// Signed node proof that an opaque object was deleted or had already been
/// deleted under the same tombstone commitment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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
        let (reply_request, reply_session) = OnionReplySession::prepare(
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

/// Stable, bounded binary encoding with an explicit frame kind outside bincode.
/// This avoids depending on serde enum discriminants for future evolution.
pub fn encode_blind_vault_frame(frame: &BlindVaultFrame) -> Result<Vec<u8>, BlindVaultError> {
    let (kind, body) = match frame {
        BlindVaultFrame::Put(value) => (
            FRAME_KIND_PUT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::StoredReceipt(value) => (
            FRAME_KIND_STORED_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseCreate(value) => (
            FRAME_KIND_LEASE_CREATE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::Delete(value) => (
            FRAME_KIND_DELETE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::DeletedReceipt(value) => (
            FRAME_KIND_DELETED_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseAdmission(value) => (
            FRAME_KIND_LEASE_ADMISSION,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::PullRequest(value) => (
            FRAME_KIND_PULL_REQUEST,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::PullResponse(value) => (
            FRAME_KIND_PULL_RESPONSE,
            serialize_body(value, MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseAdmission(value) => (
            FRAME_KIND_BLIND_LEASE_ADMISSION,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindIssuerDirectory(value) => (
            FRAME_KIND_BLIND_ISSUER_DIRECTORY,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseAccepted(value) => (
            FRAME_KIND_BLIND_LEASE_ACCEPTED,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseRetire(value) => (
            FRAME_KIND_LEASE_RETIRE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseRetiredReceipt(value) => (
            FRAME_KIND_LEASE_RETIRED_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseRenewal(value) => (
            FRAME_KIND_BLIND_LEASE_RENEWAL,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::BlindLeaseRenewed(value) => (
            FRAME_KIND_BLIND_LEASE_RENEWED,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseStatus(value) => (
            FRAME_KIND_LEASE_STATUS,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseStatusReceipt(value) => (
            FRAME_KIND_LEASE_STATUS_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseInventory(value) => (
            FRAME_KIND_LEASE_INVENTORY,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::LeaseInventoryReceipt(value) => (
            FRAME_KIND_LEASE_INVENTORY_RECEIPT,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
        BlindVaultFrame::TerminalFailure(value) => (
            FRAME_KIND_TERMINAL_FAILURE,
            serialize_body(value, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)?,
        ),
    };

    let total = FRAME_HEADER_BYTES
        .checked_add(body.len())
        .ok_or(BlindVaultError::FrameTooLarge)?;
    if total as u64 > frame_limit_for_kind(kind)? {
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

    let kind = bytes[6];
    let frame_limit = frame_limit_for_kind(kind)?;
    if bytes.len() as u64 > frame_limit {
        return Err(BlindVaultError::FrameTooLarge);
    }
    let body = &bytes[FRAME_HEADER_BYTES..];
    match kind {
        FRAME_KIND_PUT => Ok(BlindVaultFrame::Put(deserialize_body(body, frame_limit)?)),
        FRAME_KIND_STORED_RECEIPT => Ok(BlindVaultFrame::StoredReceipt(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_CREATE => Ok(BlindVaultFrame::LeaseCreate(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_DELETE => Ok(BlindVaultFrame::Delete(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_DELETED_RECEIPT => Ok(BlindVaultFrame::DeletedReceipt(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_ADMISSION => Ok(BlindVaultFrame::LeaseAdmission(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_PULL_REQUEST => Ok(BlindVaultFrame::PullRequest(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_PULL_RESPONSE => Ok(BlindVaultFrame::PullResponse(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_BLIND_LEASE_ADMISSION => Ok(BlindVaultFrame::BlindLeaseAdmission(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_BLIND_ISSUER_DIRECTORY => Ok(BlindVaultFrame::BlindIssuerDirectory(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_BLIND_LEASE_ACCEPTED => Ok(BlindVaultFrame::BlindLeaseAccepted(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_LEASE_RETIRE => Ok(BlindVaultFrame::LeaseRetire(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_RETIRED_RECEIPT => Ok(BlindVaultFrame::LeaseRetiredReceipt(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_BLIND_LEASE_RENEWAL => Ok(BlindVaultFrame::BlindLeaseRenewal(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_BLIND_LEASE_RENEWED => Ok(BlindVaultFrame::BlindLeaseRenewed(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_STATUS => Ok(BlindVaultFrame::LeaseStatus(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_STATUS_RECEIPT => Ok(BlindVaultFrame::LeaseStatusReceipt(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_LEASE_INVENTORY => Ok(BlindVaultFrame::LeaseInventory(deserialize_body(
            body,
            frame_limit,
        )?)),
        FRAME_KIND_LEASE_INVENTORY_RECEIPT => Ok(BlindVaultFrame::LeaseInventoryReceipt(
            deserialize_body(body, frame_limit)?,
        )),
        FRAME_KIND_TERMINAL_FAILURE => Ok(BlindVaultFrame::TerminalFailure(deserialize_body(
            body,
            frame_limit,
        )?)),
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
    /// Embedded Ed25519 key bytes were not a valid public key.
    #[error("invalid blind-vault public key")]
    InvalidPublicKey,
    /// Lease reused one key for write and administration authority.
    #[error("lease write and administration keys must be distinct")]
    LeaseKeyReuse,
    /// Lease request signer did not match its declared administration key.
    #[error("lease administration identity does not match its signing key")]
    AdminIdentityMismatch,
    /// A signed mutation timestamp was outside the configured replay window.
    #[error("request timestamp is outside the accepted clock-skew window")]
    RequestTimestampOutsideWindow,
    /// A lease-retirement receipt carried inconsistent aggregate deletion data.
    #[error("lease retirement receipt summary is inconsistent")]
    InvalidRetirementSummary,
    /// A lease renewal did not strictly extend one currently live generation.
    #[error("lease renewal window is invalid")]
    InvalidLeaseRenewalWindow,
    /// A signed lease-status receipt carried impossible live-usage data.
    #[error("lease status receipt summary is inconsistent")]
    InvalidLeaseStatusSummary,
    /// A private inventory entry violated its fixed-width commitment contract.
    #[error("blind vault inventory entry is invalid")]
    InvalidInventoryEntry,
    /// Inventory entries were duplicated or not in canonical object-id order.
    #[error("blind vault inventory entries are not in strict object-id order")]
    InvalidInventoryOrder,
    /// Inventory aggregate counters exceeded their fixed-width representation.
    #[error("blind vault inventory aggregate overflowed")]
    InventoryOverflow,
    /// A signed inventory receipt carried impossible aggregate data.
    #[error("blind vault inventory receipt summary is inconsistent")]
    InvalidInventorySummary,
    /// Receipt signer did not match the declared descriptor identity.
    #[error("receipt node identity does not match its signing key")]
    NodeIdentityMismatch,
    /// Admission ticket signer did not match its declared issuer identity.
    #[error("admission issuer identity does not match its signing key")]
    AdmissionIssuerMismatch,
    /// Admission ticket cannot be redeemed before its validity window.
    #[error("admission ticket is not yet valid")]
    AdmissionNotYetValid,
    /// Admission ticket carried an invalid time or lease policy.
    #[error("admission ticket policy is invalid")]
    InvalidAdmissionPolicy,
    /// Blind-admission credential version is unsupported.
    #[error("unsupported blind-admission credential version {0}")]
    UnsupportedBlindAdmissionVersion(u16),
    /// Finalized RSA signature did not fit the RSA-2048 through RSA-4096 bound.
    #[error("blind-admission signature length {actual} is invalid")]
    InvalidBlindAdmissionSignatureLength {
        /// Received finalized signature length in bytes.
        actual: usize,
    },
    /// An advertised RSA public key exceeded the issuer-directory bound.
    #[error("blind issuer public-key DER length {actual} is invalid")]
    InvalidBlindIssuerKeyLength {
        /// Received canonical DER length in bytes.
        actual: usize,
    },
    /// Advertised RSA key bytes did not match their signed key fingerprint.
    #[error("blind issuer key fingerprint does not match public-key DER")]
    BlindIssuerKeyIdMismatch,
    /// An issuer epoch had an empty, expired, reversed, or overlong policy.
    #[error("blind issuer epoch policy is invalid")]
    InvalidBlindIssuerEpochPolicy,
    /// A signed directory exceeded the protocol-wide epoch count ceiling.
    #[error("blind issuer directory contains too many epochs")]
    TooManyBlindIssuerEpochs,
    /// Epochs were duplicated or not in canonical key-ID order.
    #[error("blind issuer epochs are not in strict key-ID order")]
    BlindIssuerEpochOrderInvalid,
    /// Signed issuer discovery data was stale or implausibly far in the future.
    #[error("blind issuer directory timestamp is outside the accepted window")]
    IssuerDirectoryTimestampOutsideWindow,
    /// An authority update used generation zero, which is reserved for local
    /// static bootstrap state.
    #[error("blind issuer update generation must start at one")]
    InvalidBlindIssuerUpdateGeneration,
    /// An authority update attempted to install no usable issuer epochs.
    #[error("blind issuer update contains no epochs")]
    BlindIssuerUpdateHasNoEpochs,
    /// Declared update authority did not match the pinned verification key.
    #[error("blind issuer update authority does not match its signing key")]
    BlindIssuerAuthorityMismatch,
    /// Authority update was stale or implausibly far in the future.
    #[error("blind issuer update timestamp is outside the accepted window")]
    BlindIssuerUpdateTimestampOutsideWindow,
    /// Pull page size was zero or exceeded the protocol-wide ceiling.
    #[error("blind-vault pull limit is invalid")]
    InvalidPullLimit,
    /// Opaque pull cursor exceeded the protocol-wide ceiling.
    #[error("blind-vault pull cursor length is invalid")]
    InvalidPullCursorLength,
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

fn frame_limit_for_kind(kind: u8) -> Result<u64, BlindVaultError> {
    match kind {
        FRAME_KIND_PUT
        | FRAME_KIND_STORED_RECEIPT
        | FRAME_KIND_LEASE_CREATE
        | FRAME_KIND_DELETE
        | FRAME_KIND_DELETED_RECEIPT
        | FRAME_KIND_LEASE_ADMISSION
        | FRAME_KIND_PULL_REQUEST
        | FRAME_KIND_BLIND_LEASE_ADMISSION
        | FRAME_KIND_BLIND_ISSUER_DIRECTORY
        | FRAME_KIND_BLIND_LEASE_ACCEPTED
        | FRAME_KIND_LEASE_RETIRE
        | FRAME_KIND_LEASE_RETIRED_RECEIPT
        | FRAME_KIND_BLIND_LEASE_RENEWAL
        | FRAME_KIND_BLIND_LEASE_RENEWED
        | FRAME_KIND_LEASE_STATUS
        | FRAME_KIND_LEASE_STATUS_RECEIPT
        | FRAME_KIND_LEASE_INVENTORY
        | FRAME_KIND_LEASE_INVENTORY_RECEIPT
        | FRAME_KIND_TERMINAL_FAILURE => Ok(MAX_BLIND_VAULT_MUTATION_FRAME_BYTES),
        FRAME_KIND_PULL_RESPONSE => Ok(MAX_BLIND_VAULT_PULL_RESPONSE_FRAME_BYTES),
        unknown => Err(BlindVaultError::UnknownFrameKind(unknown)),
    }
}

fn serialize_body<T: Serialize>(value: &T, limit: u64) -> Result<Vec<u8>, BlindVaultError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(limit)
        .serialize(value)
        .map_err(|_| BlindVaultError::Serialization)
}

fn deserialize_body<T>(bytes: &[u8], limit: u64) -> Result<T, BlindVaultError>
where
    T: for<'de> Deserialize<'de>,
{
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(limit)
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
    const MAX_TICKET_TTL_MS: u64 = 24 * 60 * 60 * 1_000;

    fn lease_key() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[7; 32]).expect("valid deterministic lease key")
    }

    fn node_key() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[9; 32]).expect("valid deterministic node key")
    }

    fn admin_key() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[11; 32]).expect("valid deterministic admin key")
    }

    fn admission_issuer_key() -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[15; 32]).expect("valid deterministic admission issuer key")
    }

    fn signed_lease() -> BlindVaultLeaseCreateRequest {
        let mut lease = BlindVaultLeaseCreateRequest::new(
            [1; 32],
            [8; 16],
            lease_key().public_key_bytes(),
            admin_key().public_key_bytes(),
            sha256(&[13; 32]),
            NOW_MS + 7 * 24 * 60 * 60 * 1_000,
        );
        lease.sign(&admin_key()).expect("matching admin key");
        lease
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

    fn signed_admission() -> BlindVaultAdmissionTicket {
        let issuer = admission_issuer_key();
        let mut ticket = BlindVaultAdmissionTicket::new(
            [17; 32],
            issuer.public_key_bytes(),
            NOW_MS - 1_000,
            NOW_MS + 60 * 60 * 1_000,
            14 * 24 * 60 * 60 * 1_000,
        );
        ticket.sign(&issuer).expect("matching admission issuer");
        ticket
    }

    fn signed_pull_response() -> BlindVaultPullResponse {
        let ciphertext = vec![0xB7; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[3]];
        let object = BlindVaultRecoveredObject {
            object_id: [31; 32],
            ciphertext_commitment: sha256(&ciphertext),
            ciphertext,
            expires_at_ms: NOW_MS + 24 * 60 * 60 * 1_000,
        };
        let second_ciphertext = vec![0xC3; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[3]];
        let second_object = BlindVaultRecoveredObject {
            object_id: [32; 32],
            ciphertext_commitment: sha256(&second_ciphertext),
            ciphertext: second_ciphertext,
            expires_at_ms: NOW_MS + 24 * 60 * 60 * 1_000,
        };
        let node = node_key();
        let mut response = BlindVaultPullResponse::new(
            [1; 32],
            vec![object, second_object],
            vec![1; 49],
            NOW_MS,
            node.public_key_bytes(),
        );
        response.sign(&node).expect("sign pull response");
        response
    }

    fn signed_blind_issuer_directory() -> BlindVaultBlindIssuerDirectory {
        let mut epochs = vec![
            BlindVaultBlindIssuerEpoch::new(
                vec![0x31; 64],
                NOW_MS - 60_000,
                NOW_MS + 24 * 60 * 60 * 1_000,
                7 * 24 * 60 * 60 * 1_000,
            ),
            BlindVaultBlindIssuerEpoch::new(
                vec![0x32; 72],
                NOW_MS + 12 * 60 * 60 * 1_000,
                NOW_MS + 2 * 24 * 60 * 60 * 1_000,
                7 * 24 * 60 * 60 * 1_000,
            ),
        ];
        epochs.sort_by_key(|epoch| epoch.issuer_key_id);
        let node = node_key();
        let mut directory =
            BlindVaultBlindIssuerDirectory::new(NOW_MS, node.public_key_bytes(), epochs);
        directory.sign(&node).expect("sign issuer directory");
        directory
    }

    // [BLIND-VAULT-ISSUER-UPDATE 2026-07-23 by Codex] The update signature
    // authenticates the complete canonical generation independently of the
    // transport that carries it to a storage node.
    fn signed_blind_issuer_update() -> BlindVaultBlindIssuerUpdate {
        let mut epochs = signed_blind_issuer_directory().epochs;
        epochs.sort_by_key(|epoch| epoch.issuer_key_id);
        let authority = admission_issuer_key();
        let mut update =
            BlindVaultBlindIssuerUpdate::new(1, NOW_MS, authority.public_key_bytes(), epochs);
        update.sign(&authority).expect("sign issuer update");
        update
    }

    #[test]
    fn signed_put_validates_and_round_trips() {
        let put = signed_put();
        put.validate_and_verify(NOW_MS, MAX_TTL_MS, &lease_key().public_key())
            .expect("valid put");

        let encoded =
            encode_blind_vault_frame(&BlindVaultFrame::Put(put.clone())).expect("encode put");
        assert!(is_blind_vault_frame(&encoded));
        let decoded = decode_blind_vault_frame(&encoded).expect("decode put");
        assert_eq!(decoded, BlindVaultFrame::Put(put));
    }

    #[test]
    fn blind_vault_magic_detection_does_not_accept_other_or_truncated_protocols() {
        // [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] Onion terminal
        // dispatch must distinguish a declared Blind Vault frame from legacy
        // chat bytes before decoding, without treating a short prefix as valid.
        assert!(!is_blind_vault_frame(b""));
        assert!(!is_blind_vault_frame(b"ANB"));
        assert!(!is_blind_vault_frame(b"ANMC\x00\x01"));
        assert!(is_blind_vault_frame(b"ANBV"));
        assert!(matches!(
            decode_blind_vault_frame(b"ANBV"),
            Err(BlindVaultError::TruncatedFrame)
        ));
    }

    #[test]
    fn anonymous_lease_is_self_authenticating_and_round_trips() {
        let lease = signed_lease();
        lease
            .validate_and_verify(NOW_MS, MAX_TTL_MS)
            .expect("valid anonymous lease");

        let encoded = encode_blind_vault_frame(&BlindVaultFrame::LeaseCreate(lease.clone()))
            .expect("encode lease");
        assert_eq!(
            decode_blind_vault_frame(&encoded).expect("decode lease"),
            BlindVaultFrame::LeaseCreate(lease)
        );
    }

    #[test]
    fn bearer_admission_validates_and_round_trips_without_identity_metadata() {
        let request = BlindVaultLeaseAdmissionRequest {
            admission: signed_admission(),
            lease: signed_lease(),
        };
        request
            .validate_and_verify(
                NOW_MS,
                MAX_TTL_MS,
                MAX_TICKET_TTL_MS,
                &admission_issuer_key().public_key(),
            )
            .expect("valid admission and lease");

        let encoded = encode_blind_vault_frame(&BlindVaultFrame::LeaseAdmission(request.clone()))
            .expect("encode admission");
        assert_eq!(
            decode_blind_vault_frame(&encoded).expect("decode admission"),
            BlindVaultFrame::LeaseAdmission(request)
        );
    }

    // [BLIND-VAULT-BLIND-ADMISSION 2026-07-23 by Codex] Core validates only
    // bounded wire shape; the server crate performs RFC 9474 verification with
    // an operator-pinned RSA epoch key.
    #[test]
    fn blind_admission_shape_is_domain_separated_and_round_trips_additively() {
        let admission = BlindVaultBlindAdmissionToken::new(
            [41; 32],
            [42; 32],
            [43; 32],
            vec![44; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
        );
        admission.validate_shape().expect("bounded blind token");
        assert!(admission
            .message_bytes()
            .starts_with(BLIND_ADMISSION_MESSAGE_DOMAIN));
        assert_ne!(admission.spend_id(), admission.token_id);

        let request = BlindVaultBlindLeaseAdmissionRequest {
            admission,
            lease: signed_lease(),
        };
        let encoded =
            encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAdmission(request.clone()))
                .expect("encode blind admission");
        assert_eq!(encoded[6], FRAME_KIND_BLIND_LEASE_ADMISSION);
        assert_eq!(
            decode_blind_vault_frame(&encoded).expect("decode blind admission"),
            BlindVaultFrame::BlindLeaseAdmission(request)
        );
    }

    #[test]
    fn blind_admission_rejects_zero_randomizer_and_unbounded_signature() {
        let zero_randomizer = BlindVaultBlindAdmissionToken::new(
            [45; 32],
            [46; 32],
            [0; 32],
            vec![47; MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES],
        );
        assert!(matches!(
            zero_randomizer.validate_shape(),
            Err(BlindVaultError::ZeroIdentifier(
                "blind_admission_message_randomizer"
            ))
        ));

        let oversized = BlindVaultBlindAdmissionToken::new(
            [45; 32],
            [46; 32],
            [48; 32],
            vec![49; MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES + 1],
        );
        assert_eq!(
            oversized.validate_shape(),
            Err(BlindVaultError::InvalidBlindAdmissionSignatureLength {
                actual: MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES + 1,
            })
        );
    }

    // [BLIND-VAULT-ISSUER-DIRECTORY 2026-07-23 by Codex] Clients authenticate
    // key rotation against the node descriptor identity before creating a
    // blinded token; the directory carries public policy only.
    #[test]
    fn signed_issuer_directory_is_fresh_canonical_and_round_trips() {
        let directory = signed_blind_issuer_directory();
        directory
            .validate_and_verify(NOW_MS + 1_000, 60_000, 5_000, &node_key().public_key())
            .expect("valid issuer directory");

        let encoded =
            encode_blind_vault_frame(&BlindVaultFrame::BlindIssuerDirectory(directory.clone()))
                .expect("encode issuer directory");
        assert_eq!(encoded[6], FRAME_KIND_BLIND_ISSUER_DIRECTORY);
        assert_eq!(
            decode_blind_vault_frame(&encoded).expect("decode issuer directory"),
            BlindVaultFrame::BlindIssuerDirectory(directory)
        );
    }

    #[test]
    fn issuer_directory_rejects_key_tampering_order_drift_and_staleness() {
        let mut key_tampered = signed_blind_issuer_directory();
        key_tampered.epochs[0].public_key_der[0] ^= 1;
        assert_eq!(
            key_tampered.validate_and_verify(NOW_MS, 60_000, 5_000, &node_key().public_key(),),
            Err(BlindVaultError::BlindIssuerKeyIdMismatch)
        );

        let mut reordered = signed_blind_issuer_directory();
        reordered.epochs.reverse();
        assert_eq!(
            reordered.validate_and_verify(NOW_MS, 60_000, 5_000, &node_key().public_key(),),
            Err(BlindVaultError::BlindIssuerEpochOrderInvalid)
        );

        let stale = signed_blind_issuer_directory();
        assert_eq!(
            stale.validate_and_verify(NOW_MS + 60_001, 60_000, 5_000, &node_key().public_key(),),
            Err(BlindVaultError::IssuerDirectoryTimestampOutsideWindow)
        );
    }

    #[test]
    fn authority_signed_issuer_update_is_canonical_and_fresh() {
        let update = signed_blind_issuer_update();
        update
            .validate_and_verify(
                NOW_MS + 1_000,
                60_000,
                5_000,
                &admission_issuer_key().public_key(),
            )
            .expect("valid authority-signed issuer update");

        let encoded = serialize_body(&update, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)
            .expect("serialize transport-independent update");
        let decoded: BlindVaultBlindIssuerUpdate =
            deserialize_body(&encoded, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)
                .expect("deserialize transport-independent update");
        assert_eq!(decoded, update);
    }

    #[test]
    fn issuer_update_rejects_forgery_wrong_authority_and_staleness() {
        let mut forged = signed_blind_issuer_update();
        forged.generation = 2;
        assert_eq!(
            forged
                .validate_and_verify(NOW_MS, 60_000, 5_000, &admission_issuer_key().public_key(),),
            Err(BlindVaultError::InvalidSignature)
        );

        let wrong_authority = IdentityKeyPair::from_bytes(&[29; 32]).expect("wrong authority key");
        let update = signed_blind_issuer_update();
        assert_eq!(
            update.validate_and_verify(NOW_MS, 60_000, 5_000, &wrong_authority.public_key(),),
            Err(BlindVaultError::BlindIssuerAuthorityMismatch)
        );
        assert_eq!(
            update.validate_and_verify(
                NOW_MS + 60_001,
                60_000,
                5_000,
                &admission_issuer_key().public_key(),
            ),
            Err(BlindVaultError::BlindIssuerUpdateTimestampOutsideWindow)
        );
    }

    #[test]
    fn issuer_update_rejects_reserved_generation_and_empty_epoch_set() {
        let authority = admission_issuer_key();
        let mut generation_zero = BlindVaultBlindIssuerUpdate::new(
            0,
            NOW_MS,
            authority.public_key_bytes(),
            signed_blind_issuer_directory().epochs,
        );
        assert_eq!(
            generation_zero.sign(&authority),
            Err(BlindVaultError::InvalidBlindIssuerUpdateGeneration)
        );

        let mut empty =
            BlindVaultBlindIssuerUpdate::new(1, NOW_MS, authority.public_key_bytes(), Vec::new());
        assert_eq!(
            empty.sign(&authority),
            Err(BlindVaultError::BlindIssuerUpdateHasNoEpochs)
        );
    }

    #[test]
    fn admission_rejects_future_window_and_overlong_lease() {
        let issuer = admission_issuer_key();
        let mut future = signed_admission();
        future.not_before_ms = NOW_MS + 1;
        future.sign(&issuer).expect("sign future ticket");
        assert_eq!(
            future.validate_and_verify(NOW_MS, MAX_TICKET_TTL_MS, &issuer.public_key()),
            Err(BlindVaultError::AdmissionNotYetValid)
        );

        let mut narrow = signed_admission();
        narrow.maximum_lease_ttl_ms = 60 * 60 * 1_000;
        narrow.sign(&issuer).expect("sign narrow ticket");
        let request = BlindVaultLeaseAdmissionRequest {
            admission: narrow,
            lease: signed_lease(),
        };
        assert_eq!(
            request.validate_and_verify(
                NOW_MS,
                MAX_TTL_MS,
                MAX_TICKET_TTL_MS,
                &issuer.public_key(),
            ),
            Err(BlindVaultError::LifetimeTooLong)
        );
    }

    #[test]
    fn signed_pull_page_round_trips_above_mutation_ceiling() {
        let request = BlindVaultPullRequest {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: [1; 32],
            read_capability: [33; 32],
            continuation_cursor: Vec::new(),
            limit: MAX_BLIND_VAULT_PULL_OBJECTS as u16,
        };
        request.validate().expect("valid pull request");
        let request_frame =
            encode_blind_vault_frame(&BlindVaultFrame::PullRequest(request.clone()))
                .expect("encode pull request");
        assert_eq!(
            decode_blind_vault_frame(&request_frame).expect("decode pull request"),
            BlindVaultFrame::PullRequest(request)
        );

        let response = signed_pull_response();
        response
            .validate_and_verify(&node_key().public_key())
            .expect("valid signed response");
        let encoded = encode_blind_vault_frame(&BlindVaultFrame::PullResponse(response.clone()))
            .expect("encode pull response");
        assert!(encoded.len() as u64 > MAX_BLIND_VAULT_MUTATION_FRAME_BYTES);
        assert_eq!(
            decode_blind_vault_frame(&encoded).expect("decode pull response"),
            BlindVaultFrame::PullResponse(response)
        );

        let mut wrong_kind = encoded;
        wrong_kind[6] = FRAME_KIND_PUT;
        assert_eq!(
            decode_blind_vault_frame(&wrong_kind),
            Err(BlindVaultError::FrameTooLarge)
        );
    }

    #[test]
    fn pull_page_rejects_ciphertext_and_cursor_tampering() {
        let mut response = signed_pull_response();
        response.objects[0].ciphertext[0] ^= 1;
        assert_eq!(
            response.validate_and_verify(&node_key().public_key()),
            Err(BlindVaultError::CommitmentMismatch)
        );

        let mut response = signed_pull_response();
        response.continuation_cursor[0] ^= 1;
        assert_eq!(
            response.validate_and_verify(&node_key().public_key()),
            Err(BlindVaultError::InvalidSignature)
        );
    }

    #[test]
    fn lease_rejects_write_and_admin_key_reuse() {
        let key = admin_key();
        let mut lease = BlindVaultLeaseCreateRequest::new(
            [1; 32],
            [8; 16],
            key.public_key_bytes(),
            key.public_key_bytes(),
            sha256(&[13; 32]),
            NOW_MS + 7 * 24 * 60 * 60 * 1_000,
        );
        lease.sign(&key).expect("matching admin key");
        assert_eq!(
            lease.validate_and_verify(NOW_MS, MAX_TTL_MS),
            Err(BlindVaultError::LeaseKeyReuse)
        );
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
    fn administration_delete_and_node_receipt_are_independently_signed() {
        let mut delete = BlindVaultDeleteRequest::new([1; 32], [2; 32], [6; 16], NOW_MS);
        delete.sign(&admin_key());
        delete
            .validate_and_verify(NOW_MS + 50, 1_000, &admin_key().public_key())
            .expect("valid admin deletion");

        let node = node_key();
        let mut receipt = BlindVaultDeletedReceipt::new(
            &delete,
            signed_put().ciphertext_commitment,
            NOW_MS + 100,
            node.public_key_bytes(),
        );
        receipt.sign(&node).expect("matching node key");
        receipt
            .validate_and_verify(&node.public_key())
            .expect("valid deletion receipt");
        assert!(receipt.matches_delete(&delete));

        let encoded = encode_blind_vault_frame(&BlindVaultFrame::DeletedReceipt(receipt.clone()))
            .expect("encode deletion receipt");
        assert_eq!(
            decode_blind_vault_frame(&encoded).expect("decode deletion receipt"),
            BlindVaultFrame::DeletedReceipt(receipt)
        );
    }

    #[test]
    fn deletion_timestamp_has_bounded_replay_window() {
        let mut delete = BlindVaultDeleteRequest::new([1; 32], [2; 32], [6; 16], NOW_MS - 1_001);
        delete.sign(&admin_key());
        assert_eq!(
            delete.validate_and_verify(NOW_MS, 1_000, &admin_key().public_key()),
            Err(BlindVaultError::RequestTimestampOutsideWindow)
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
