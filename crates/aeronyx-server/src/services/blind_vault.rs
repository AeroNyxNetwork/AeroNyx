// ============================================
// File: crates/aeronyx-server/src/services/blind_vault.rs
// ============================================
//! # Blind Vault Service
//!
//! ## Creation Reason
//! Implements durable, node-blind storage for encrypted contact-vault and
//! optional message-archive segments without reusing identity-indexed MemChain
//! records or receiver-indexed legacy chat queues.
//!
//! ## Main Functionality
//! - Self-authenticating anonymous lease provisioning.
//! - Atomic one-time bearer admission spend + lease creation.
//! - RFC 9474 blind-admission verification under rotating public epoch keys.
//! - Authority-authenticated installation of complete public issuer generations.
//! - Persistent monotonic runtime rotation of public issuer epochs.
//! - Deterministic public issuer-epoch snapshots for node-signed discovery.
//! - Immutable, idempotent ciphertext object persistence.
//! - Pre-transaction signature authentication with in-transaction authority
//!   revalidation for contention-safe write admission.
//! - Capability-gated bounded recovery pages with encrypted snapshot cursors.
//! - Administration-key object deletion with signed node receipts.
//! - Complete administration-key lease retirement with exact-retry receipts.
//! - Blind-authorized administration-key lease renewal with atomic token spend.
//! - Administration-authorized coherent live-lease status receipts.
//! - Streaming private inventory commitments over still-live ciphertext rows.
//! - Read-only per-lease administration authorization for explicit replica jobs.
//! - Transactional per-lease count/byte quotas and bounded expiry cleanup.
//! - Atomic node-wide live-lease and aggregate ciphertext capacity admission.
//! - Fail-closed physical disk reserve enforcement through a replaceable host
//!   capacity probe.
//! - Stable privacy-safe mutation failure classes for multi-hop retry policy.
//!
//! ## Dependencies
//! - `aeronyx_core::protocol::blind_vault`: stable signed wire contracts.
//! - `config_blind_vault.rs`: storage and retention policy.
//! - `rusqlite`: dedicated WAL database; never shares a MemChain/chat path.
//!
//! ## Main Logical Flow
//! 1. Verify an operator-pinned anonymous admission issuer, atomically consume
//!    the one-time bearer ticket, and provision a random replica/epoch lease.
//! 2. Validate put signatures with the lease write key and store ciphertext
//!    verbatim in one immediate transaction.
//! 3. Return a descriptor-identity-signed `BlindVaultStoredReceipt`.
//! 4. Authenticate recovery with a random bearer capability and return only a
//!    bounded page of still-live opaque objects.
//! 5. Validate deletion with the separate administration key, retain a bounded
//!    commitment-only tombstone, and sign a deletion receipt.
//! 6. Retire a complete lease atomically, retain only bounded request evidence,
//!    and return an encrypted-path-safe aggregate receipt.
//! 7. Spend a fresh blind credential and extend one live lease in the same
//!    transaction, preserving exact retry evidence without an account index.
//! 8. Observe expiry and still-live ciphertext usage under one storage lock,
//!    then return a terminal-signed private status receipt.
//! 9. Stream the canonical live object set into a per-replica commitment and
//!    return only its encrypted terminal-signed aggregate receipt.
//! 10. Verify explicit source-object replication authority without exposing
//!    the lease administration verifier or mutating Blind Vault state.
//! 11. Reject new leases or ciphertext before token spend/write when the
//!    operator's node-wide capacity commitment has been reached.
//! 12. Preserve the configured database-filesystem reserve while leaving
//!    recovery, deletion, retirement, and cleanup available under pressure.
//!
//! ## Privacy Invariant
//! This service has no account, wallet, sender, receiver, conversation,
//! namespace, content-type, vector, search-token, or social-edge columns. Logs
//! remain aggregate-only and must never include IDs, capabilities, ciphertext,
//! commitments, keys, paths selected by clients, or per-lease usage.
//!
//! ## Important Note For The Next Developer
//! - Every lease, including tests, must enter through V1 or V2 admission and
//!   the one-time spend table. Do not restore a direct provisioning bypass.
//! - V1 tickets remain linkable bearer credentials for compatibility. Prefer
//!   V2 blind admission for new integrations and never reuse issuance logs as
//!   storage-node policy input.
//! - Never accept mutable replacement under an existing object ID.
//! - Never publish object commitments or receipts into the Directory Chain.
//! - API handlers must run synchronous SQLite methods in `spawn_blocking`.
//! - Pull cursors must remain lease-bound AEAD ciphertext; never expose the
//!   internal SQLite sequence or accept a caller-provided raw sequence.
//! - Runtime issuer updates must be signed by a separately pinned authority,
//!   then remain monotonic, continuity-safe, and atomic across both SQLite
//!   persistence and in-process readers.
//!
//! Last Modified: v1.22.0-ReplicaJobClaims - Made the exact signed replica-job
//! claims one private typed value with an explicit V1 construction boundary.
//! v1.21.1-ReplicaJobNodeIdentityExclusion - Rejected source
//! lease administration keys that collide with the local node identity.
//! v1.21.0-ReplicaJobAuthorization - Added a dedicated,
//! read-only per-source-lease replication authorization capability.
//! v1.20.0-PrivacySafeServiceDiagnostics - Redacted opaque pull
//! values and nested host errors, disabled private-row Debug, and removed the
//! remaining production HMAC panic path.
//! v1.19.0-BlindVaultAdmissionReadiness - Added one aggregate,
//! fail-closed admission-readiness state for discovery and operations.
//! v1.18.0-BlindVaultDiskReserve - Added replaceable physical
//! capacity observation and fail-closed write watermarks.
//! v1.17.0-BlindVaultNodeCapacity - Added atomic node-wide live
//! lease and aggregate ciphertext capacity enforcement and status.
//! v1.16.0-BlindVaultLeaseInventory - Added streaming,
//! administration-authorized private encrypted-object inventory commitments.
//! v1.15.0-BlindVaultLeaseStatus - Added coherent,
//! administration-authorized live-lease status observations and receipts.
//! v1.14.0-BlindVaultLeaseRenewal - Added blind-authorized,
//! atomic live-lease renewal with exact request commitments and signed receipts.
//! v1.13.0-BlindVaultLeaseRetirement - Added atomic complete
//! lease deletion with exact request commitments and bounded retry evidence.
//! v1.12.0-BlindVaultAdmissionFailureClass - Added exhaustive,
//! privacy-safe anonymous admission retry classification.
//! v1.11.0-BlindVaultDeleteFailureClass - Added exhaustive,
//! privacy-safe anonymous deletion retry classification.
//! v1.10.0-BlindVaultPullFailureClass - Added exhaustive,
//! privacy-safe anonymous recovery retry classification.
//! v1.9.0-BlindVaultPutFailureClass - Separated permanent
//! request rejection, replica capacity, and retryable service availability
//! without exposing lease or object state.
//! v1.8.0-BlindVaultAuthPipeline - Moved client signature
//! verification outside SQLite write transactions and added authority
//! revalidation at the atomic mutation boundary.
//! v1.7.0-BlindVaultIssuerAuthority - Added pinned authority
//! verification and closed the unauthenticated runtime installer boundary.
//! v1.6.0-BlindVaultIssuerRuntime - Added persistent monotonic
//! public issuer-epoch rotation with rollback and continuity protection.
//! v1.5.0-BlindVaultAdmissionOnly - Removed the unused direct
//! lease-provisioning bypass and moved tests onto the production admission path.
//! v1.4.0-BlindVaultIssuerDirectory - Added deterministic
//! active/future issuer snapshots for authenticated key discovery.
//! v1.3.0-BlindVaultBlindAdmission - Added RFC 9474 V2
//! verification with key-bound policy and scheme-separated replay markers.
//! v1.2.0-BlindVaultAdmission - Added pinned issuer validation,
//! atomic one-time redemption, and bounded spend-marker cleanup.
//! v1.1.0-BlindVaultSnapshotCursor - Added lease-bound encrypted
//! snapshot cursors for stable recovery pagination.
//! v1.0.0-BlindVaultService - Initial transactional service.
//! ============================================

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::blind_vault::{
    BlindVaultBlindAdmissionToken, BlindVaultBlindIssuerDirectory, BlindVaultBlindIssuerEpoch,
    BlindVaultBlindIssuerUpdate, BlindVaultBlindLeaseAdmissionRequest,
    BlindVaultBlindLeaseRenewalRequest, BlindVaultBlindLeaseRenewedReceipt,
    BlindVaultDeleteRequest, BlindVaultDeletedReceipt, BlindVaultError,
    BlindVaultInventoryCommitmentBuilder, BlindVaultInventoryCommitmentEntry,
    BlindVaultInventoryCommitmentSummary, BlindVaultLeaseAdmissionRequest,
    BlindVaultLeaseCreateRequest, BlindVaultLeaseInventoryReceipt, BlindVaultLeaseInventoryRequest,
    BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt, BlindVaultLeaseStatusReceipt,
    BlindVaultLeaseStatusRequest, BlindVaultPutRequest, BlindVaultStoredReceipt,
};
use blind_rsa_signatures::{MessageRandomizer, PublicKeySha384PSSRandomized, Signature};
use chacha20poly1305::{
    aead::{Aead, NewAead, Payload},
    Key, XChaCha20Poly1305, XNonce,
};
use hmac::{Hmac, Mac};
use parking_lot::{Mutex, RwLock};
use rand::{rngs::OsRng, RngCore};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::config::BlindVaultConfig;

use super::blind_vault_capacity::{
    BlindVaultFilesystemCapacityProbe, SystemBlindVaultFilesystemCapacityProbe,
};

type HmacSha256 = Hmac<Sha256>;

const READ_AUTH_KEY_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReadAuth-Key-v1";
const READ_AUTH_TAG_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReadAuth-Tag-v1";
const PULL_CURSOR_KEY_DOMAIN: &[u8] = b"AeroNyx-BlindVault-PullCursor-Key-v1";
const PULL_CURSOR_AAD_DOMAIN: &[u8] = b"AeroNyx-BlindVault-PullCursor-AAD-v1";
const BLIND_ISSUER_SET_DIGEST_DOMAIN: &[u8] = b"AeroNyx-BlindVault-IssuerSet-Digest-v1";
const REPLICA_JOB_AUTHORIZATION_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaJobAuthorization-v1";
const REPLICA_JOB_CLAIMS_COMMITMENT_DOMAIN: &[u8] =
    b"AeroNyx-BlindVault-ReplicaJobClaimsCommitment-v1";
const REPLICA_JOB_AUTHORIZATION_VERSION_V1: u16 = 1;
// [BLIND-VAULT-REPLICA-AUTH 2026-09-01 by Codex] Keep client authority within
// a fixed ten-minute replay window. Longer lifecycle work must obtain a new
// per-lease authorization rather than inheriting node authority.
const MAX_REPLICA_JOB_AUTHORIZATION_TTL_MS: u64 = 10 * 60 * 1_000;
const PULL_CURSOR_VERSION: u8 = 1;
const PULL_CURSOR_NONCE_BYTES: usize = 24;
const PULL_CURSOR_PLAINTEXT_BYTES: usize = 16;
const PULL_CURSOR_TAG_BYTES: usize = 16;
const PULL_CURSOR_BYTES: usize =
    1 + PULL_CURSOR_NONCE_BYTES + PULL_CURSOR_PLAINTEXT_BYTES + PULL_CURSOR_TAG_BYTES;
const CLEANUP_OBJECT_BATCH: usize = 512;
const CLEANUP_LEASE_BATCH: usize = 128;
const CLEANUP_TOMBSTONE_BATCH: usize = 512;
const CLEANUP_ADMISSION_SPEND_BATCH: usize = 512;
// [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] Leave room for SQLite pages,
// indexes, and transaction metadata in addition to transient WAL/main-file
// duplication of a newly accepted ciphertext.
const FILESYSTEM_WRITE_OVERHEAD_BYTES: u64 = 64 * 1024;

/// Common authentication capability for private administration observations.
///
/// [BLIND-VAULT-OBSERVATION-TRAIT 2026-08-28 by Codex] Status and inventory
/// remain separate signed wire domains while sharing the same lease-authority,
/// freshness, and pre-SQLite verification pipeline.
trait BlindVaultAdminObservationRequest {
    fn lease_id(&self) -> &[u8; 32];

    fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError>;
}

impl BlindVaultAdminObservationRequest for BlindVaultLeaseStatusRequest {
    fn lease_id(&self) -> &[u8; 32] {
        &self.lease_id
    }

    fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        BlindVaultLeaseStatusRequest::validate_and_verify(
            self,
            now_ms,
            maximum_clock_skew_ms,
            admin_key,
        )
    }
}

impl BlindVaultAdminObservationRequest for BlindVaultLeaseInventoryRequest {
    fn lease_id(&self) -> &[u8; 32] {
        &self.lease_id
    }

    fn validate_and_verify(
        &self,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
        admin_key: &IdentityPublicKey,
    ) -> Result<(), BlindVaultError> {
        BlindVaultLeaseInventoryRequest::validate_and_verify(
            self,
            now_ms,
            maximum_clock_skew_ms,
            admin_key,
        )
    }
}

#[derive(Clone)]
struct BlindAdmissionIssuer {
    public_key: Arc<PublicKeySha384PSSRandomized>,
    not_before_ms: u64,
    expires_at_ms: u64,
    max_lease_ttl_ms: u64,
}

struct BlindAdmissionIssuerRuntime {
    generation: u64,
    digest: [u8; 32],
    updated_at_ms: u64,
    epochs: Vec<BlindVaultBlindIssuerEpoch>,
    issuers: HashMap<[u8; 32], BlindAdmissionIssuer>,
}

/// Result of idempotent anonymous lease provisioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultLeaseProvisionOutcome {
    /// A new lease row was committed.
    Created,
    /// The exact same signed lease was already present.
    Existing,
}

/// Result of an authenticated public issuer-directory installation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultIssuerInstallOutcome {
    /// A strictly newer generation was durably installed.
    Installed {
        /// Installed monotonic generation.
        generation: u64,
    },
    /// The exact same generation and canonical epoch set was already present.
    Unchanged {
        /// Existing monotonic generation.
        generation: u64,
    },
}

/// Aggregate-only public issuer runtime state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultIssuerRuntimeStatus {
    /// Monotonic directory generation; zero denotes static TOML bootstrap.
    pub generation: u64,
    /// Local installation time, or zero for the static bootstrap generation.
    pub updated_at_ms: u64,
    /// Number of epochs in the persisted canonical generation.
    pub epoch_count: usize,
    /// Number of epochs active at the supplied observation time.
    pub active_epoch_count: usize,
}

/// One opaque object returned to an authorised client.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultStoredObject {
    /// Replica-local object identifier.
    pub object_id: [u8; 32],
    /// Exact client ciphertext stored by the node.
    pub ciphertext: Vec<u8>,
    /// SHA-256 commitment bound by storage receipts.
    pub ciphertext_commitment: [u8; 32],
    /// Object retention deadline in Unix milliseconds.
    pub expires_at_ms: u64,
}

/// The one canonical set of client-authorized replica-job claims.
///
/// Fields stay private so a future coordinator must carry this typed value (or
/// its canonical commitment) instead of rebuilding parallel source, target,
/// and bundle arguments after verification.
///
/// [BLIND-VAULT-REPLICA-CLAIMS 2026-09-01 by Codex] This value is deliberately
/// neither cloneable nor independently constructible outside its versioned
/// authorization boundary.
pub(crate) struct BlindVaultReplicaJobClaimsV1 {
    version: u16,
    job_id: [u8; 16],
    source_lease_id: [u8; 32],
    source_object_id: [u8; 32],
    source_ciphertext_commitment: [u8; 32],
    target_node_id: [u8; 32],
    target_bundle_commitment: [u8; 32],
    authorized_at_ms: u64,
    expires_at_ms: u64,
}

impl BlindVaultReplicaJobClaimsV1 {
    /// Exact bytes signed by the existing source lease administration key.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(REPLICA_JOB_AUTHORIZATION_DOMAIN.len() + 194);
        bytes.extend_from_slice(REPLICA_JOB_AUTHORIZATION_DOMAIN);
        bytes.extend_from_slice(&self.version.to_be_bytes());
        bytes.extend_from_slice(&self.job_id);
        bytes.extend_from_slice(&self.source_lease_id);
        bytes.extend_from_slice(&self.source_object_id);
        bytes.extend_from_slice(&self.source_ciphertext_commitment);
        bytes.extend_from_slice(&self.target_node_id);
        bytes.extend_from_slice(&self.target_bundle_commitment);
        bytes.extend_from_slice(&self.authorized_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        bytes
    }

    /// Domain-separated identity for exact durable job-id retry comparison.
    #[allow(dead_code)]
    pub(crate) fn commitment(&self) -> [u8; 32] {
        let canonical = self.canonical_bytes();
        let mut hasher = Sha256::new();
        hasher.update(REPLICA_JOB_CLAIMS_COMMITMENT_DOMAIN);
        hasher.update((canonical.len() as u32).to_be_bytes());
        hasher.update(canonical);
        hasher.finalize().into()
    }

    /// Random client job identifier for future exact-retry staging.
    #[allow(dead_code)]
    pub(crate) const fn job_id(&self) -> [u8; 16] {
        self.job_id
    }

    /// Exact target whose canonical bundle was authorized.
    #[allow(dead_code)]
    pub(crate) const fn target_node_id(&self) -> [u8; 32] {
        self.target_node_id
    }

    /// Commitment to the only target bundle a coordinator may stage.
    #[allow(dead_code)]
    pub(crate) const fn target_bundle_commitment(&self) -> [u8; 32] {
        self.target_bundle_commitment
    }
}

/// Client-signed authority for one explicit source-object replication job.
///
/// This is source-local control data, not a Blind Vault frame or node-to-node
/// protocol value. The target bundle commitment binds independently wrapped
/// replica requests without making their identifiers comparable to this
/// source lease or object.
pub(crate) struct BlindVaultReplicaJobAuthorizationV1 {
    // [BLIND-VAULT-REPLICA-CLAIMS 2026-09-01 by Codex] Authorization owns the
    // sole typed claims value. Verification and future staging must borrow this
    // same value rather than accepting a second raw target/bundle context.
    claims: BlindVaultReplicaJobClaimsV1,
    signature: [u8; 64],
}

impl BlindVaultReplicaJobAuthorizationV1 {
    /// Decodes the fixed-width wire fields into one explicit V1 proof.
    ///
    /// Unknown versions fail before a typed claims value exists. A future API
    /// must pass the received version here rather than defaulting it to V1.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_wire_parts(
        wire_version: u16,
        job_id: [u8; 16],
        source_lease_id: [u8; 32],
        source_object_id: [u8; 32],
        source_ciphertext_commitment: [u8; 32],
        target_node_id: [u8; 32],
        target_bundle_commitment: [u8; 32],
        authorized_at_ms: u64,
        expires_at_ms: u64,
        signature: [u8; 64],
    ) -> Result<Self, BlindVaultReplicaJobAuthorizationError> {
        if wire_version != REPLICA_JOB_AUTHORIZATION_VERSION_V1 {
            return Err(BlindVaultReplicaJobAuthorizationError::Rejected);
        }
        Ok(Self {
            claims: BlindVaultReplicaJobClaimsV1 {
                version: wire_version,
                job_id,
                source_lease_id,
                source_object_id,
                source_ciphertext_commitment,
                target_node_id,
                target_bundle_commitment,
                authorized_at_ms,
                expires_at_ms,
            },
            signature,
        })
    }

    /// The exact claims future staging and dispatch must continue to use.
    #[allow(dead_code)]
    pub(crate) const fn claims(&self) -> &BlindVaultReplicaJobClaimsV1 {
        &self.claims
    }

    /// Canonical bytes signed only by the existing source lease admin key.
    #[must_use]
    pub(crate) fn signing_bytes(&self) -> Vec<u8> {
        self.claims.canonical_bytes()
    }

    /// Exact private bytes used to compare durable authorization retries.
    ///
    /// [BLIND-VAULT-REPLICA-COORDINATOR 2026-09-01 by Codex] The source
    /// coordinator persists this opaque value together with the canonical
    /// target bundle. It must never rebuild an authorization from parallel
    /// target or bundle arguments after verification.
    pub(crate) fn canonical_authorization_bytes(&self) -> Vec<u8> {
        let signing_bytes = self.signing_bytes();
        let mut bytes = Vec::with_capacity(signing_bytes.len() + self.signature.len());
        bytes.extend_from_slice(&signing_bytes);
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    /// Restores one exact canonical proof from the private durable job store.
    ///
    /// [BLIND-VAULT-REPLICA-JOB-STORE 2026-09-01 by Codex] Parsing is fixed
    /// width and domain/version checked. Cryptographic authority verification
    /// remains a separate read-only service capability immediately before use.
    pub(crate) fn from_canonical_authorization_bytes(
        bytes: &[u8],
    ) -> Result<Self, BlindVaultReplicaJobAuthorizationError> {
        const FIXED_CLAIMS_BYTES: usize = 2 + 16 + 32 * 5 + 8 * 2;
        const SIGNATURE_BYTES: usize = 64;
        let expected_len = REPLICA_JOB_AUTHORIZATION_DOMAIN
            .len()
            .checked_add(FIXED_CLAIMS_BYTES)
            .and_then(|length| length.checked_add(SIGNATURE_BYTES))
            .ok_or(BlindVaultReplicaJobAuthorizationError::Rejected)?;
        if bytes.len() != expected_len || !bytes.starts_with(REPLICA_JOB_AUTHORIZATION_DOMAIN) {
            return Err(BlindVaultReplicaJobAuthorizationError::Rejected);
        }

        let mut cursor = REPLICA_JOB_AUTHORIZATION_DOMAIN.len();
        let wire_version = u16::from_be_bytes(replica_authorization_take(bytes, &mut cursor)?);
        let job_id = replica_authorization_take(bytes, &mut cursor)?;
        let source_lease_id = replica_authorization_take(bytes, &mut cursor)?;
        let source_object_id = replica_authorization_take(bytes, &mut cursor)?;
        let source_ciphertext_commitment = replica_authorization_take(bytes, &mut cursor)?;
        let target_node_id = replica_authorization_take(bytes, &mut cursor)?;
        let target_bundle_commitment = replica_authorization_take(bytes, &mut cursor)?;
        let authorized_at_ms = u64::from_be_bytes(replica_authorization_take(bytes, &mut cursor)?);
        let expires_at_ms = u64::from_be_bytes(replica_authorization_take(bytes, &mut cursor)?);
        let signature = replica_authorization_take(bytes, &mut cursor)?;
        if cursor != bytes.len() {
            return Err(BlindVaultReplicaJobAuthorizationError::Rejected);
        }
        Self::from_wire_parts(
            wire_version,
            job_id,
            source_lease_id,
            source_object_id,
            source_ciphertext_commitment,
            target_node_id,
            target_bundle_commitment,
            authorized_at_ms,
            expires_at_ms,
            signature,
        )
    }

    fn validate_shape(&self, now_ms: u64) -> Result<(), BlindVaultReplicaJobAuthorizationError> {
        let claims = &self.claims;
        let lifetime = claims
            .expires_at_ms
            .checked_sub(claims.authorized_at_ms)
            .ok_or(BlindVaultReplicaJobAuthorizationError::Rejected)?;
        if claims.version != REPLICA_JOB_AUTHORIZATION_VERSION_V1
            || now_ms == 0
            || claims.job_id == [0; 16]
            || claims.source_lease_id == [0; 32]
            || claims.source_object_id == [0; 32]
            || claims.source_ciphertext_commitment == [0; 32]
            || claims.target_node_id == [0; 32]
            || claims.target_bundle_commitment == [0; 32]
            || claims.authorized_at_ms == 0
            || claims.authorized_at_ms > now_ms
            || claims.expires_at_ms <= now_ms
            || lifetime == 0
            || lifetime > MAX_REPLICA_JOB_AUTHORIZATION_TTL_MS
            || IdentityPublicKey::from_bytes(&claims.target_node_id).is_err()
        {
            return Err(BlindVaultReplicaJobAuthorizationError::Rejected);
        }
        Ok(())
    }
}

// [BLIND-VAULT-REPLICA-AUTH 2026-09-01 by Codex] Standard diagnostics expose
// neither source/target correlation identifiers nor opaque commitments.
impl fmt::Debug for BlindVaultReplicaJobAuthorizationV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaJobAuthorizationV1")
            .field("version", &self.claims.version)
            .field("private_fields", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Coarse result of source-lease replication authorization verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindVaultReplicaJobAuthorizationError {
    /// The same proof cannot authorize this source object and target bundle.
    Rejected,
    /// The local source authority could not be established safely.
    Unavailable,
}

impl fmt::Display for BlindVaultReplicaJobAuthorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Rejected => "blind vault replica job authorization rejected",
            Self::Unavailable => "blind vault replica job authorization unavailable",
        })
    }
}

impl fmt::Debug for BlindVaultReplicaJobAuthorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for BlindVaultReplicaJobAuthorizationError {}

// [BLIND-VAULT-OPAQUE-OBJECT-DIAGNOSTICS 2026-08-30 by Codex] Public Debug
// remains available for compatibility, but must never expose ciphertext or
// per-object correlation metadata from the node-blind storage boundary.
impl fmt::Debug for BlindVaultStoredObject {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultStoredObject")
            .field("object_id", &"<redacted>")
            .field("ciphertext", &"<redacted>")
            .field("ciphertext_commitment", &"<redacted>")
            .field("expires_at_ms", &"<redacted>")
            .finish()
    }
}

/// Bounded internal recovery page.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultPullPage {
    /// Still-live opaque objects in insertion order.
    pub objects: Vec<BlindVaultStoredObject>,
    /// Lease-bound encrypted cursor for the same stable recovery snapshot.
    pub continuation_cursor: Option<Vec<u8>>,
}

impl fmt::Debug for BlindVaultPullPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultPullPage")
            .field("objects", &"<redacted>")
            .field("continuation_cursor", &"<redacted>")
            .finish()
    }
}

/// Aggregate-only bounded maintenance result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultCleanupReport {
    /// Expired objects removed in this bounded run.
    pub objects_removed: u64,
    /// Expired leases removed in this bounded run.
    pub leases_removed: u64,
    /// Expired commitment-only tombstones removed in this bounded run.
    pub tombstones_removed: u64,
    /// Expired complete-lease retirement tombstones removed in this run.
    pub lease_tombstones_removed: u64,
    /// Expired exact-retry renewal markers removed in this bounded run.
    pub lease_renewals_removed: u64,
    /// Expired one-time admission spend markers removed in this bounded run.
    pub admission_spends_removed: u64,
}

/// Aggregate-only local service health snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultStatus {
    /// Whether the configured service is enabled.
    pub enabled: bool,
    /// Number of live anonymous replica leases.
    pub live_leases: u64,
    /// [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] Leases still consuming
    /// capacity, including expired rows awaiting bounded cleanup.
    pub committed_leases: u64,
    /// Number of live opaque objects.
    pub live_objects: u64,
    /// Total live ciphertext bytes.
    pub live_ciphertext_bytes: u64,
    /// Bytes still committed by live lease counters, including expired objects
    /// awaiting the next bounded cleanup pass.
    pub committed_ciphertext_bytes: u64,
    /// Operator-configured maximum simultaneous live leases.
    pub max_live_leases: u64,
    /// Remaining live lease admission capacity.
    pub remaining_live_leases: u64,
    /// Operator-configured aggregate ciphertext capacity.
    pub max_total_ciphertext_bytes: u64,
    /// Remaining aggregate ciphertext capacity.
    pub remaining_ciphertext_bytes: u64,
    /// [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] Operator-configured
    /// physical reserve. Zero means the backward-compatible probe policy is
    /// disabled.
    pub min_free_disk_bytes: u64,
    /// Latest bytes available to this process on the database filesystem.
    /// `None` means the policy is disabled or the probe could not observe it.
    pub available_disk_bytes: Option<u64>,
    /// Whether the physical reserve is disabled or currently satisfied.
    pub physical_capacity_ready: bool,
    /// Number of retained commitment-only deletion tombstones.
    pub tombstones: u64,
    /// Number of unexpired complete-lease retirement tombstones.
    pub lease_tombstones: u64,
    /// Number of unexpired exact-retry lease-renewal markers.
    pub lease_renewals: u64,
    /// Unexpired one-time admission spend markers retained for replay defence.
    pub retained_admission_spends: u64,
}

/// Aggregate reason a replica can or cannot accept a new anonymous lease.
///
/// The state intentionally carries no issuer, lease, object, path, or storage
/// amount. Discovery and operations may consume it without becoming a private
/// storage-state oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultAdmissionReadiness {
    /// Public admission policy, issuer material, and both capacity layers are
    /// currently ready.
    Ready,
    /// Public admission is disabled or no V1/V2 issuer is currently usable.
    PolicyUnavailable,
    /// The operator's retained lease or ciphertext commitment is exhausted.
    LogicalCapacityExhausted,
    /// The configured physical disk reserve cannot survive another admission.
    PhysicalCapacityExhausted,
    /// The enabled physical capacity probe could not establish a safe state.
    PhysicalCapacityUnknown,
}

impl BlindVaultAdmissionReadiness {
    /// Returns whether the node may advertise anonymous lease admission.
    #[must_use]
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// Fail-closed service errors. API handlers should map these to coarse public
/// buckets instead of returning database or capability details.
#[derive(thiserror::Error)]
pub enum BlindVaultServiceError {
    /// Service was not explicitly enabled.
    #[error("blind vault service is disabled")]
    Disabled,
    /// Dedicated SQLite operation failed.
    #[error("blind vault database error")]
    Sqlite(#[from] rusqlite::Error),
    /// Core protocol validation or signature verification failed.
    #[error("blind vault protocol validation failed")]
    Protocol(#[from] BlindVaultError),
    /// Database directory could not be created.
    #[error("blind vault database directory could not be created")]
    Filesystem,
    /// The enabled filesystem reserve could not be observed safely.
    #[error("blind vault filesystem capacity is unavailable")]
    FilesystemCapacityUnavailable,
    /// Lease does not exist.
    #[error("blind vault lease not found")]
    LeaseNotFound,
    /// Lease is no longer live.
    #[error("blind vault lease expired")]
    LeaseExpired,
    /// Existing lease ID or provisioning request has different authority data.
    #[error("blind vault lease conflicts with existing state")]
    LeaseConflict,
    /// Existing object ID has different immutable content.
    #[error("blind vault object conflicts with existing state")]
    ObjectConflict,
    /// Idempotency request ID was reused for a different object.
    #[error("blind vault request conflicts with existing state")]
    RequestConflict,
    /// Object was already deleted and its identifier cannot be reused.
    #[error("blind vault object was deleted")]
    ObjectDeleted,
    /// Per-lease object or byte quota would be exceeded.
    #[error("blind vault lease quota exceeded")]
    QuotaExceeded,
    /// Node-wide live lease or aggregate ciphertext capacity was reached.
    #[error("blind vault node capacity exhausted")]
    NodeCapacityExceeded,
    /// Read capability did not authorise this lease.
    #[error("blind vault read capability rejected")]
    ReadUnauthorized,
    /// Pull cursor could not be authenticated for this lease.
    #[error("blind vault pull cursor rejected")]
    InvalidPullCursor,
    /// Pull cursor could not be encrypted.
    #[error("blind vault pull cursor encryption failed")]
    PullCursorEncryptionFailed,
    /// Public lease admission is not explicitly enabled.
    #[error("blind vault public admission is disabled")]
    AdmissionUnavailable,
    /// Admission issuer is not pinned by this node operator.
    #[error("blind vault admission issuer rejected")]
    AdmissionIssuerRejected,
    /// Blind admission proof did not verify under its pinned epoch key.
    #[error("blind vault admission proof rejected")]
    AdmissionProofRejected,
    /// Admission ticket was already consumed by another lease.
    #[error("blind vault admission ticket already spent")]
    AdmissionSpent,
    /// Admission issuer configuration could not be parsed safely.
    #[error("blind vault admission issuer configuration is invalid")]
    AdmissionConfigurationInvalid,
    /// Runtime issuer update was not signed by a node-pinned authority.
    #[error("blind vault issuer update authority rejected")]
    IssuerDirectoryAuthorityRejected,
    /// Runtime issuer update was malformed, stale, or cryptographically invalid.
    #[error("blind vault issuer update signature rejected")]
    IssuerDirectoryUpdateRejected,
    /// Candidate issuer generation is older than the durable runtime state.
    #[error("blind vault issuer directory rollback rejected")]
    IssuerDirectoryRollback,
    /// Candidate reused a generation for different public issuer material.
    #[error("blind vault issuer directory generation conflicts")]
    IssuerDirectoryGenerationConflict,
    /// Candidate removed or changed an issuer epoch that is still valid.
    #[error("blind vault issuer directory breaks active epoch continuity")]
    IssuerDirectoryContinuity,
    /// Candidate has no issuer epoch active at installation time.
    #[error("blind vault issuer directory has no active epoch")]
    IssuerDirectoryNoActiveEpoch,
    /// Candidate issuer generation cannot be represented durably.
    #[error("blind vault issuer directory generation is outside the supported range")]
    IssuerDirectoryGenerationOutOfRange,
    /// Requested object is absent and has no retained tombstone.
    #[error("blind vault object not found")]
    ObjectNotFound,
    /// Durable row violated an internal fixed-size invariant.
    #[error("blind vault durable state is corrupt")]
    CorruptState,
    /// Timestamp could not be represented safely by SQLite.
    #[error("blind vault timestamp is outside the supported range")]
    TimestampOutOfRange,
}

// [BLIND-VAULT-SERVICE-ERROR-DIAGNOSTICS 2026-08-30 by Codex] Keep standard
// formatting at the coarse service boundary. `Error::source` still lets a
// trusted local adapter inspect nested SQLite or protocol errors explicitly.
impl fmt::Debug for BlindVaultServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

/// Privacy-safe retry class for an anonymous ciphertext Put failure.
///
/// This classification intentionally does not preserve lease, object,
/// signature, or database details. Multi-hop relays may use it to decide
/// whether another terminal is useful without becoming a storage-state oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultPutFailureClass {
    /// The signed request cannot succeed unchanged and must not be retried.
    Rejected,
    /// This replica cannot accept more ciphertext under the current lease.
    Capacity,
    /// The replica is disabled, unhealthy, or temporarily unavailable.
    Unavailable,
}

/// Privacy-safe retry class for an anonymous ciphertext Pull failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultPullFailureClass {
    /// The same capability-bound request cannot succeed unchanged.
    Rejected,
    /// Replica storage, cursor sealing, or local runtime is unavailable.
    Unavailable,
}

/// Privacy-safe retry class for an anonymous ciphertext Delete failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultDeleteFailureClass {
    /// The same signed administration request cannot succeed unchanged.
    Rejected,
    /// Replica storage or local runtime is unavailable.
    Unavailable,
}

/// Privacy-safe retry class for blind-issued anonymous lease admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultAdmissionFailureClass {
    /// The same credential and signed lease cannot succeed unchanged.
    Rejected,
    /// [BLIND-VAULT-NODE-CAPACITY 2026-08-28 by Codex] This replica has
    /// reached its operator-configured storage capacity.
    Capacity,
    /// Admission policy, replica storage, or local runtime is unavailable.
    Unavailable,
}

/// Privacy-safe retry class for complete anonymous lease retirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultLeaseRetireFailureClass {
    /// The same signed retirement cannot succeed unchanged.
    Rejected,
    /// Replica storage or local runtime is unavailable.
    Unavailable,
}

/// Privacy-safe retry class for blind-authorized lease renewal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultLeaseRenewFailureClass {
    /// The same credential and signed transition cannot succeed unchanged.
    Rejected,
    /// Admission policy, replica storage, or local runtime is unavailable.
    Unavailable,
}

/// Privacy-safe retry class for private lease-status observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultLeaseStatusFailureClass {
    /// The same signed status request cannot succeed unchanged.
    Rejected,
    /// Replica storage or local runtime is unavailable.
    Unavailable,
}

/// Privacy-safe retry class for private lease-inventory commitments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultLeaseInventoryFailureClass {
    /// The same signed inventory request cannot succeed unchanged.
    Rejected,
    /// Replica storage or local runtime is unavailable.
    Unavailable,
}

impl BlindVaultServiceError {
    /// Returns the coarse retry class for a Put failure.
    #[must_use]
    pub const fn put_failure_class(&self) -> BlindVaultPutFailureClass {
        // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] Keep this exhaustive:
        // adding a service error must force an explicit retry/privacy decision.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound
            | Self::TimestampOutOfRange => BlindVaultPutFailureClass::Rejected,
            Self::QuotaExceeded | Self::NodeCapacityExceeded => BlindVaultPutFailureClass::Capacity,
            Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState => BlindVaultPutFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for a Pull failure.
    #[must_use]
    pub const fn pull_failure_class(&self) -> BlindVaultPullFailureClass {
        // [BLIND-VAULT-PULL-RETRY-CLASS 2026-08-28 by Codex] Keep this match
        // exhaustive. New storage errors must receive an explicit privacy and
        // retry decision instead of silently becoming a remote state oracle.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultPullFailureClass::Rejected,
            Self::NodeCapacityExceeded
            | Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultPullFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for a Delete failure.
    #[must_use]
    pub const fn delete_failure_class(&self) -> BlindVaultDeleteFailureClass {
        // [BLIND-VAULT-DELETE-RETRY-CLASS 2026-08-28 by Codex] Keep this match
        // exhaustive so new storage errors cannot silently expose replica
        // object state or receive an accidental cross-node retry policy.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultDeleteFailureClass::Rejected,
            Self::NodeCapacityExceeded
            | Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultDeleteFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for a blind-issued admission failure.
    #[must_use]
    pub const fn admission_failure_class(&self) -> BlindVaultAdmissionFailureClass {
        // [BLIND-VAULT-ADMISSION-RETRY-CLASS 2026-08-28 by Codex] Keep this
        // exhaustive. An upstream relay must never learn whether rejection was
        // caused by a credential, issuer epoch, lease conflict, or replay.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultAdmissionFailureClass::Rejected,
            Self::NodeCapacityExceeded => BlindVaultAdmissionFailureClass::Capacity,
            Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultAdmissionFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for complete lease retirement.
    #[must_use]
    pub const fn lease_retire_failure_class(&self) -> BlindVaultLeaseRetireFailureClass {
        // [BLIND-VAULT-LEASE-RETIRE-FAILURE 2026-08-28 by Codex] Keep this
        // exhaustive so upstream relays never learn whether a lease was live,
        // already retired, expired, conflicting, or signed by the wrong key.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultLeaseRetireFailureClass::Rejected,
            Self::NodeCapacityExceeded
            | Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultLeaseRetireFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for blind-authorized lease renewal.
    #[must_use]
    pub const fn lease_renew_failure_class(&self) -> BlindVaultLeaseRenewFailureClass {
        // [BLIND-VAULT-LEASE-RENEW-FAILURE 2026-08-28 by Codex] Keep this
        // exhaustive so the relay cannot distinguish credential, lease,
        // expiry-generation, or administration-authority rejection.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultLeaseRenewFailureClass::Rejected,
            Self::NodeCapacityExceeded
            | Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultLeaseRenewFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for private lease-status observation.
    #[must_use]
    pub const fn lease_status_failure_class(&self) -> BlindVaultLeaseStatusFailureClass {
        // [BLIND-VAULT-LEASE-STATUS-FAILURE 2026-08-28 by Codex] Keep lease
        // existence, expiry, authority, and request validity indistinguishable
        // outside the encrypted terminal response boundary.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultLeaseStatusFailureClass::Rejected,
            Self::NodeCapacityExceeded
            | Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultLeaseStatusFailureClass::Unavailable,
        }
    }

    /// Returns the coarse retry class for private inventory commitment.
    #[must_use]
    pub const fn lease_inventory_failure_class(&self) -> BlindVaultLeaseInventoryFailureClass {
        // [BLIND-VAULT-INVENTORY-FAILURE 2026-08-28 by Codex] Object count,
        // commitment generation, lease state, and authority failures collapse
        // to the same outer classes; no inventory detail reaches a relay.
        match self {
            Self::Protocol(_)
            | Self::LeaseNotFound
            | Self::LeaseExpired
            | Self::LeaseConflict
            | Self::ObjectConflict
            | Self::RequestConflict
            | Self::ObjectDeleted
            | Self::QuotaExceeded
            | Self::ReadUnauthorized
            | Self::InvalidPullCursor
            | Self::AdmissionIssuerRejected
            | Self::AdmissionProofRejected
            | Self::AdmissionSpent
            | Self::ObjectNotFound => BlindVaultLeaseInventoryFailureClass::Rejected,
            Self::NodeCapacityExceeded
            | Self::Disabled
            | Self::Sqlite(_)
            | Self::Filesystem
            | Self::FilesystemCapacityUnavailable
            | Self::PullCursorEncryptionFailed
            | Self::AdmissionUnavailable
            | Self::AdmissionConfigurationInvalid
            | Self::IssuerDirectoryAuthorityRejected
            | Self::IssuerDirectoryUpdateRejected
            | Self::IssuerDirectoryRollback
            | Self::IssuerDirectoryGenerationConflict
            | Self::IssuerDirectoryContinuity
            | Self::IssuerDirectoryNoActiveEpoch
            | Self::IssuerDirectoryGenerationOutOfRange
            | Self::CorruptState
            | Self::TimestampOutOfRange => BlindVaultLeaseInventoryFailureClass::Unavailable,
        }
    }
}

/// Dedicated anonymous encrypted-object storage service.
pub struct BlindVaultService {
    config: BlindVaultConfig,
    connection: Mutex<Connection>,
    filesystem_capacity_probe: Arc<dyn BlindVaultFilesystemCapacityProbe>,
    filesystem_capacity_path: PathBuf,
    node_identity: IdentityKeyPair,
    read_auth_key: Zeroizing<[u8; 32]>,
    pull_cursor_key: Zeroizing<[u8; 32]>,
    admission_issuers: HashMap<[u8; 32], IdentityPublicKey>,
    blind_issuer_update_authorities: HashMap<[u8; 32], IdentityPublicKey>,
    blind_admission_issuers: RwLock<BlindAdmissionIssuerRuntime>,
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod admission;
mod filesystem_capacity;
mod lease_admin;
mod object_store;
mod vault_maintenance;
mod vault_rows;

use admission::blind_issuer_set_digest;
use admission::build_blind_issuer_runtime;
use admission::derive_node_key;
use admission::load_persisted_blind_issuer_runtime;
use admission::persist_blind_issuer_runtime;
use admission::replica_authorization_take;
use vault_rows::ensure_exact_lease_renewal;
use vault_rows::ensure_lease_request_available;
use vault_rows::ensure_node_admission_capacity;
use vault_rows::ensure_node_ciphertext_capacity;
use vault_rows::existing_lease_outcome;
use vault_rows::fixed_array;
use vault_rows::init_schema;
use vault_rows::insert_lease_row;
use vault_rows::load_active_lease_renewal;
use vault_rows::load_active_lease_retirement;
use vault_rows::load_existing_object;
use vault_rows::load_lease_object_usage;
use vault_rows::load_lease_provisioning;
use vault_rows::load_lease_runtime;
use vault_rows::load_live_lease_inventory;
use vault_rows::load_live_lease_status;
use vault_rows::load_replica_job_authority;
use vault_rows::non_negative_u64;
use vault_rows::select_expired_leases;
use vault_rows::select_expired_objects;
use vault_rows::sqlite_i64;

impl BlindVaultService {
    /// Opens the dedicated database, applies WAL safety settings, and audits the
    /// schema. The caller must only construct this service when config is valid.
    pub fn new(
        config: BlindVaultConfig,
        node_identity: IdentityKeyPair,
    ) -> Result<Self, BlindVaultServiceError> {
        Self::new_with_filesystem_capacity_probe(
            config,
            node_identity,
            Arc::new(SystemBlindVaultFilesystemCapacityProbe),
        )
    }

    /// Opens the service with an explicit physical-capacity capability.
    ///
    /// This constructor keeps production policy testable and permits a future
    /// container-volume probe without changing Blind Vault transaction logic.
    pub fn new_with_filesystem_capacity_probe(
        config: BlindVaultConfig,
        node_identity: IdentityKeyPair,
        filesystem_capacity_probe: Arc<dyn BlindVaultFilesystemCapacityProbe>,
    ) -> Result<Self, BlindVaultServiceError> {
        if !config.enabled {
            return Err(BlindVaultServiceError::Disabled);
        }
        let database_path = Path::new(&config.db_path);
        if let Some(parent) = database_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|_| BlindVaultServiceError::Filesystem)?;
            }
        }
        // [BLIND-VAULT-DISK-RESERVE 2026-08-28 by Codex] Probe the containing
        // filesystem, never the database file itself: a first startup may not
        // have created that file yet. Relative leaf paths belong to `.`.
        let filesystem_capacity_path = database_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let connection = Connection::open(&config.db_path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "NORMAL")?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        init_schema(&connection)?;

        let seed = Zeroizing::new(node_identity.to_bytes());
        let read_auth_key = derive_node_key(seed.as_ref(), READ_AUTH_KEY_DOMAIN)?;
        let pull_cursor_key = derive_node_key(seed.as_ref(), PULL_CURSOR_KEY_DOMAIN)?;
        let admission_issuers = config
            .admission_issuer_key_bytes()
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?
            .into_iter()
            .map(|bytes| {
                IdentityPublicKey::from_bytes(&bytes)
                    .map(|key| (bytes, key))
                    .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let blind_issuer_update_authorities = config
            .blind_issuer_update_authority_key_bytes()
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?
            .into_iter()
            .map(|bytes| {
                IdentityPublicKey::from_bytes(&bytes)
                    .map(|key| (bytes, key))
                    .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let bootstrap_epochs = config
            .blind_admission_issuer_materials()
            .map_err(|_| BlindVaultServiceError::AdmissionConfigurationInvalid)?
            .into_iter()
            .map(|material| {
                let epoch = BlindVaultBlindIssuerEpoch::new(
                    material.public_key_der,
                    material.not_before_ms,
                    material.expires_at_ms,
                    material.max_lease_ttl_ms,
                );
                if epoch.issuer_key_id != material.key_id {
                    return Err(BlindVaultServiceError::AdmissionConfigurationInvalid);
                }
                Ok(epoch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // [BLIND-VAULT-ISSUER-RUNTIME 2026-07-23 by Codex] Static TOML is the
        // backward-compatible generation-zero bootstrap only. Once an
        // authenticated newer set is installed, the durable SQLite snapshot
        // wins across restart so stale deployment files cannot roll it back.
        let bootstrap_runtime = build_blind_issuer_runtime(
            0,
            bootstrap_epochs,
            0,
            false,
            config.max_lease_ttl_ms(),
            &node_identity,
        )?;
        let blind_admission_issuers = load_persisted_blind_issuer_runtime(
            &connection,
            bootstrap_runtime,
            config.max_lease_ttl_ms(),
            &node_identity,
        )?;

        Ok(Self {
            config,
            connection: Mutex::new(connection),
            filesystem_capacity_probe,
            filesystem_capacity_path,
            node_identity,
            read_auth_key,
            pull_cursor_key,
            admission_issuers,
            blind_issuer_update_authorities,
            blind_admission_issuers: RwLock::new(blind_admission_issuers),
        })
    }
}

// [BLIND-VAULT-PRIVATE-ROW-DIAGNOSTICS 2026-08-30 by Codex] These private
// persistence rows intentionally do not implement Debug. They contain
// capability tags, correlation identifiers, commitments, keys, or per-lease
// usage that the service privacy invariant forbids in logs.
struct LeaseProvisioningRow {
    request_id: [u8; 16],
    write_verifying_key: [u8; 32],
    admin_verifying_key: [u8; 32],
    read_capability_tag: [u8; 32],
    expires_at_ms: u64,
}

#[derive(Clone, Copy)]
struct LeaseRuntimeRow {
    write_verifying_key: [u8; 32],
    admin_verifying_key: [u8; 32],
    expires_at_ms: u64,
    object_count: u64,
    byte_count: u64,
}

struct ReplicaJobAuthorityRow {
    admin_verifying_key: [u8; 32],
    lease_expires_at_ms: u64,
    ciphertext_commitment: [u8; 32],
    object_expires_at_ms: u64,
}

/// One coherent live-usage observation returned by a single SQLite statement.
#[derive(Clone, Copy)]
struct LeaseStatusObservation {
    admin_verifying_key: [u8; 32],
    expires_at_ms: u64,
    live_object_count: u64,
    live_ciphertext_bytes: u64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct LeaseObjectUsage {
    object_count: u64,
    ciphertext_bytes: u64,
}

struct ExistingObjectRow {
    request_id: [u8; 16],
    ciphertext_commitment: [u8; 32],
    created_at_ms: u64,
    expires_at_ms: u64,
}

struct ExpiredObjectRow {
    sequence: i64,
    lease_id: Vec<u8>,
    object_id: Vec<u8>,
    ciphertext_commitment: Vec<u8>,
    ciphertext_bytes: i64,
}

/// Bounded durable state retained only for exact retirement retries.
struct LeaseRetirementRow {
    request_id: [u8; 16],
    admin_verifying_key: [u8; 32],
    request_commitment: [u8; 32],
    retired_at_ms: u64,
    deleted_object_count: u64,
    deleted_ciphertext_bytes: u64,
}

/// Bounded durable state retained only for exact lease-renewal retries.
#[derive(Clone, Copy)]
struct LeaseRenewalRow {
    admission_spend_id: [u8; 32],
    request_commitment: [u8; 32],
    previous_expires_at_ms: u64,
    renewed_expires_at_ms: u64,
    renewed_at_ms: u64,
}

enum LeaseRetirementAuthoritySnapshot {
    Live { admin_verifying_key: [u8; 32] },
    Retired(LeaseRetirementRow),
}

/// Shared service pointer used by future Axum handlers and maintenance tasks.
pub type SharedBlindVaultService = Arc<BlindVaultService>;

#[cfg(test)]
mod tests {
    mod admission;
    mod lease;
    mod objects;
    mod other;

    use super::*;
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultAdmissionTicket, BlindVaultBlindAdmissionToken,
        BlindVaultBlindLeaseAdmissionRequest,
    };
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use blind_rsa_signatures::{DefaultRng, KeyPairSha384PSSRandomized};
    use tempfile::TempDir;

    use crate::config_blind_vault::BlindVaultBlindAdmissionIssuerConfig;

    const NOW_MS: u64 = 1_800_000_000_000;

    fn blind_issuer_epoch(
        key_pair: &KeyPairSha384PSSRandomized,
        not_before_ms: u64,
        expires_at_ms: u64,
        max_lease_ttl_ms: u64,
    ) -> BlindVaultBlindIssuerEpoch {
        BlindVaultBlindIssuerEpoch::new(
            key_pair.pk.to_der().expect("public key DER"),
            not_before_ms,
            expires_at_ms,
            max_lease_ttl_ms,
        )
    }

    struct Fixture {
        _directory: TempDir,
        service: BlindVaultService,
        write_key: IdentityKeyPair,
        admin_key: IdentityKeyPair,
        read_capability: [u8; 32],
        admission: BlindVaultAdmissionTicket,
        lease: BlindVaultLeaseCreateRequest,
    }

    impl Fixture {
        fn new(max_objects: u64, max_bytes: u64) -> Self {
            Self::new_with_admin_and_node_seeds(max_objects, max_bytes, [8; 32], [10; 32])
        }

        fn new_with_admin_and_node_seeds(
            max_objects: u64,
            max_bytes: u64,
            admin_seed: [u8; 32],
            node_seed: [u8; 32],
        ) -> Self {
            let directory = tempfile::tempdir().expect("temp directory");
            let issuer_key = IdentityKeyPair::from_bytes(&[6; 32]).expect("issuer key");
            let config = BlindVaultConfig {
                enabled: true,
                public_api_enabled: true,
                admission_issuer_public_keys: vec![hex::encode(issuer_key.public_key_bytes())],
                db_path: directory.path().join("vault.db").display().to_string(),
                max_objects_per_lease: max_objects,
                max_bytes_per_lease: max_bytes,
                ..BlindVaultConfig::default()
            };
            let write_key = IdentityKeyPair::from_bytes(&[7; 32]).expect("write key");
            let admin_key = IdentityKeyPair::from_bytes(&admin_seed).expect("admin key");
            let read_capability = [9; 32];
            let mut lease = BlindVaultLeaseCreateRequest::new(
                [1; 32],
                [2; 16],
                write_key.public_key_bytes(),
                admin_key.public_key_bytes(),
                Sha256::digest(read_capability).into(),
                NOW_MS + 7 * 24 * 60 * 60 * 1_000,
            );
            lease.sign(&admin_key).expect("sign lease");
            let mut admission = BlindVaultAdmissionTicket::new(
                [6; 32],
                issuer_key.public_key_bytes(),
                NOW_MS - 1_000,
                NOW_MS + 60 * 60 * 1_000,
                14 * 24 * 60 * 60 * 1_000,
            );
            admission.sign(&issuer_key).expect("sign admission");
            let node_key = IdentityKeyPair::from_bytes(&node_seed).expect("node key");
            let service = BlindVaultService::new(config, node_key).expect("service");
            Self {
                _directory: directory,
                service,
                write_key,
                admin_key,
                read_capability,
                admission,
                lease,
            }
        }

        fn provision(&self) {
            assert_eq!(
                self.service
                    .provision_lease_with_admission(
                        &self.admission_request(self.lease.clone()),
                        NOW_MS
                    )
                    .expect("provision lease"),
                BlindVaultLeaseProvisionOutcome::Created
            );
        }

        fn admission_request(
            &self,
            lease: BlindVaultLeaseCreateRequest,
        ) -> BlindVaultLeaseAdmissionRequest {
            BlindVaultLeaseAdmissionRequest {
                admission: self.admission.clone(),
                lease,
            }
        }

        fn put(&self, object_byte: u8, request_byte: u8) -> BlindVaultPutRequest {
            let mut put = BlindVaultPutRequest::new(
                [1; 32],
                [object_byte; 32],
                [request_byte; 16],
                vec![object_byte; 4 * 1024],
                NOW_MS + 24 * 60 * 60 * 1_000,
            );
            put.sign(&self.write_key);
            put
        }

        fn store_object(&self, object_byte: u8, request_byte: u8) -> BlindVaultPutRequest {
            let put = self.put(object_byte, request_byte);
            self.service
                .put(&put, NOW_MS + 1)
                .expect("store opaque object");
            put
        }

        fn replica_authorization(
            &self,
            put: &BlindVaultPutRequest,
        ) -> BlindVaultReplicaJobAuthorizationV1 {
            let target = IdentityKeyPair::from_bytes(&[31; 32]).expect("target node key");
            let mut authorization = BlindVaultReplicaJobAuthorizationV1::from_wire_parts(
                REPLICA_JOB_AUTHORIZATION_VERSION_V1,
                [40; 16],
                put.lease_id,
                put.object_id,
                put.ciphertext_commitment,
                target.public_key_bytes(),
                [41; 32],
                NOW_MS + 2,
                NOW_MS + 2 + MAX_REPLICA_JOB_AUTHORIZATION_TTL_MS / 2,
                [0; 64],
            )
            .expect("V1 replica authorization fixture");
            resign_replica_authorization(&mut authorization, &self.admin_key);
            authorization
        }
    }

    fn resign_replica_authorization(
        authorization: &mut BlindVaultReplicaJobAuthorizationV1,
        admin_key: &IdentityKeyPair,
    ) {
        authorization.signature = admin_key.sign(&authorization.signing_bytes());
    }

    // [BLIND-VAULT-REPLICA-AUTH 2026-09-01 by Codex] This snapshot proves the
    // verifier cannot spend admission authority, stage work, or change durable
    // lease/object/capacity counters. It deliberately never selects ciphertext.
    #[derive(PartialEq, Eq)]
    struct ReplicaAuthorizationMutationSnapshot {
        table_rows: [i64; 9],
        capacity: (i64, i64),
        lease_usage: (i64, i64, i64),
        object_authority: (Vec<u8>, i64),
        sqlite_total_changes: i64,
    }

    fn replica_authorization_mutation_snapshot(
        service: &BlindVaultService,
        lease_id: &[u8; 32],
        object_id: &[u8; 32],
    ) -> ReplicaAuthorizationMutationSnapshot {
        let connection = service.connection.lock();
        let count = |table: &str| -> i64 {
            connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .expect("table row count")
        };
        let capacity = connection
            .query_row(
                "SELECT committed_lease_count, committed_ciphertext_bytes
                 FROM blind_vault_capacity_state WHERE state_id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("capacity state");
        let lease_usage = connection
            .query_row(
                "SELECT object_count, byte_count, expires_at_ms
                 FROM blind_vault_leases WHERE lease_id = ?1",
                params![&lease_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("lease usage");
        let object_authority = connection
            .query_row(
                "SELECT ciphertext_commitment, expires_at_ms
                 FROM blind_vault_objects WHERE lease_id = ?1 AND object_id = ?2",
                params![&lease_id[..], &object_id[..]],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("object authority");
        let sqlite_total_changes = connection
            .query_row("SELECT total_changes()", [], |row| row.get(0))
            .expect("SQLite total changes");
        ReplicaAuthorizationMutationSnapshot {
            table_rows: [
                count("blind_vault_leases"),
                count("blind_vault_objects"),
                count("blind_vault_capacity_state"),
                count("blind_vault_tombstones"),
                count("blind_vault_lease_tombstones"),
                count("blind_vault_lease_renewals"),
                count("blind_vault_admission_spends"),
                count("blind_vault_blind_issuer_state"),
                count("blind_vault_blind_issuer_epochs"),
            ],
            capacity,
            lease_usage,
            object_authority,
            sqlite_total_changes,
        }
    }

    fn provisioned_object_fixture() -> (Fixture, BlindVaultPutRequest) {
        let fixture = Fixture::new(10, 1024 * 1024);
        fixture.provision();
        let put = fixture.store_object(42, 43);
        (fixture, put)
    }
}
