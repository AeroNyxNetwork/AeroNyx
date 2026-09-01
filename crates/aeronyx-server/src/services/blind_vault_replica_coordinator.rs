//! Source-local admission boundary for explicit Blind Vault replication jobs.
//!
//! # File Creation Notes
//! - Creation reason: bind a source-lease authorization to one exact target
//!   and one canonical, opaque terminal-effect bundle before durable staging.
//! - Main functionality: canonical bundle validation, immutable submission,
//!   exact job-id retry records, and fail-closed coordinator admission.
//! - Dependencies: consumes the existing Blind Vault V1 frame codec and the
//!   read-only per-lease verifier in `services::blind_vault`.
//!
//! # Main Logical Flow
//! 1. Decode and canonicalise the ordered target-local admission, put, and
//!    inventory frames without interpreting ciphertext.
//! 2. Bind the canonical bundle commitment to the one typed source authority.
//! 3. Verify immediately before exact-idempotent staging; do no network work.
//!
//! # Important Note For The Next Developer
//! - Ordinary Blind Vault `put` must never call this module implicitly.
//! - Do not add parallel target or bundle parameters after verification.
//! - This milestone does not send traffic. A production durable store and
//!   outbound transport must preserve the exact bytes exposed here.
//!
//! Last Modified: v1.0.0-ReplicaCoordinatorAdmission - Added the typed,
//! node-blind M12 admission boundary.
//! v1.1.0-DurableSingleJobGeneration - Added an atomic, versioned,
//! exact-idempotent single-active-job store with restart validation.

use super::blind_vault::{
    BlindVaultReplicaJobAuthorizationError, BlindVaultReplicaJobAuthorizationV1, BlindVaultService,
};
use aeronyx_core::crypto::IdentityPublicKey;
use aeronyx_core::protocol::blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultFrame,
};
use sha2::{Digest, Sha256};
use std::fmt;
#[cfg(unix)]
use std::path::Path;
use std::sync::Arc;
use zeroize::{Zeroize, Zeroizing};

#[cfg(unix)]
use super::blind_vault_replica_recovery_io::{PrivateAtomicRecoveryFile, PrivateRecoveryIoError};

const REPLICA_TARGET_BUNDLE_VERSION_V1: u16 = 1;
const REPLICA_TARGET_EFFECT_COUNT: usize = 3;
const REPLICA_TARGET_OBJECT_BYTES: usize = 4_096;
const REPLICA_TARGET_BUNDLE_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaTargetBundle-v1";
const REPLICA_STAGED_JOB_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaStagedJob-v1";
#[cfg(unix)]
const REPLICA_JOB_FILE_MAGIC: [u8; 4] = *b"AXVJ";
#[cfg(unix)]
const REPLICA_JOB_FILE_VERSION_V1: u16 = 1;
#[cfg(unix)]
const REPLICA_JOB_FILE_CHECKSUM_DOMAIN: &[u8] = b"AeroNyx-BlindVault-ReplicaJobFile-v1";
#[cfg(unix)]
const REPLICA_JOB_FILE_HEADER_BYTES: usize = 4 + 2 + 4;
#[cfg(unix)]
const REPLICA_JOB_FILE_CHECKSUM_BYTES: usize = 32;
#[cfg(unix)]
const MAX_REPLICA_JOB_AUTHORIZATION_BYTES: usize = 512;
const MAX_REPLICA_TARGET_BUNDLE_BYTES: usize = 32 * 1024;
#[cfg(unix)]
const REPLICA_JOB_BODY_FIXED_BYTES: usize = 16 + 32 * 3 + 4 + 4 + 32;
#[cfg(unix)]
const MAX_REPLICA_JOB_FILE_BYTES: usize = REPLICA_JOB_FILE_HEADER_BYTES
    + REPLICA_JOB_BODY_FIXED_BYTES
    + MAX_REPLICA_JOB_AUTHORIZATION_BYTES
    + MAX_REPLICA_TARGET_BUNDLE_BYTES
    + REPLICA_JOB_FILE_CHECKSUM_BYTES;

/// Validation policy supplied by the source runtime for one target bundle.
#[derive(Clone, Copy)]
pub(crate) struct BlindVaultReplicaTargetBundlePolicyV1 {
    now_ms: u64,
    maximum_lease_ttl_ms: u64,
    maximum_object_ttl_ms: u64,
    maximum_inventory_clock_skew_ms: u64,
}

impl BlindVaultReplicaTargetBundlePolicyV1 {
    /// Builds one explicit, bounded validation policy.
    pub(crate) fn new(
        now_ms: u64,
        maximum_lease_ttl_ms: u64,
        maximum_object_ttl_ms: u64,
        maximum_inventory_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaCoordinatorError> {
        if now_ms == 0
            || maximum_lease_ttl_ms == 0
            || maximum_object_ttl_ms == 0
            || maximum_inventory_clock_skew_ms == 0
        {
            return Err(BlindVaultReplicaCoordinatorError::Rejected);
        }
        Ok(Self {
            now_ms,
            maximum_lease_ttl_ms,
            maximum_object_ttl_ms,
            maximum_inventory_clock_skew_ms,
        })
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum BlindVaultReplicaTargetEffectPurpose {
    LeaseAdmission = 1,
    Put = 2,
    LeaseInventory = 3,
}

/// One canonical target-local bundle authorised by the source lease owner.
///
/// [BLIND-VAULT-REPLICA-COORDINATOR 2026-09-01 by Codex] Fields stay private;
/// callers may stage or dispatch only the exact canonical bytes and commitment
/// produced by this constructor.
pub(crate) struct BlindVaultReplicaTargetBundleV1 {
    target_node_id: [u8; 32],
    canonical_effects: [Vec<u8>; REPLICA_TARGET_EFFECT_COUNT],
    canonical_bytes: Vec<u8>,
    commitment: [u8; 32],
}

impl BlindVaultReplicaTargetBundleV1 {
    /// Validates and canonicalises the exact three-effect M12 bundle.
    pub(crate) fn from_wire_parts(
        wire_version: u16,
        target_node_id: [u8; 32],
        admission_frame: &[u8],
        put_frame: &[u8],
        inventory_frame: &[u8],
        policy: BlindVaultReplicaTargetBundlePolicyV1,
    ) -> Result<Self, BlindVaultReplicaCoordinatorError> {
        if wire_version != REPLICA_TARGET_BUNDLE_VERSION_V1
            || target_node_id == [0; 32]
            || IdentityPublicKey::from_bytes(&target_node_id).is_err()
        {
            return Err(BlindVaultReplicaCoordinatorError::Rejected);
        }

        let admission = match decode_blind_vault_frame(admission_frame)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?
        {
            BlindVaultFrame::BlindLeaseAdmission(request) => request,
            _ => return Err(BlindVaultReplicaCoordinatorError::Rejected),
        };
        admission
            .admission
            .validate_shape()
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;
        admission
            .lease
            .validate_and_verify(policy.now_ms, policy.maximum_lease_ttl_ms)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;

        let put = match decode_blind_vault_frame(put_frame)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?
        {
            BlindVaultFrame::Put(request) => request,
            _ => return Err(BlindVaultReplicaCoordinatorError::Rejected),
        };
        let write_key = IdentityPublicKey::from_bytes(&admission.lease.write_verifying_key)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;
        put.validate_and_verify(policy.now_ms, policy.maximum_object_ttl_ms, &write_key)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;

        let inventory = match decode_blind_vault_frame(inventory_frame)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?
        {
            BlindVaultFrame::LeaseInventory(request) => request,
            _ => return Err(BlindVaultReplicaCoordinatorError::Rejected),
        };
        let admin_key = IdentityPublicKey::from_bytes(&admission.lease.admin_verifying_key)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;
        inventory
            .validate_and_verify(
                policy.now_ms,
                policy.maximum_inventory_clock_skew_ms,
                &admin_key,
            )
            .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;

        if put.ciphertext.len() != REPLICA_TARGET_OBJECT_BYTES
            || put.lease_id != admission.lease.lease_id
            || inventory.lease_id != admission.lease.lease_id
            || put.expires_at_ms > admission.lease.expires_at_ms
        {
            return Err(BlindVaultReplicaCoordinatorError::Rejected);
        }

        let canonical_effects = [
            encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAdmission(admission))
                .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?,
            encode_blind_vault_frame(&BlindVaultFrame::Put(put))
                .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?,
            encode_blind_vault_frame(&BlindVaultFrame::LeaseInventory(inventory))
                .map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?,
        ];
        let canonical_bytes = canonical_bundle_bytes(target_node_id, &canonical_effects)?;
        if canonical_bytes.len() > MAX_REPLICA_TARGET_BUNDLE_BYTES {
            return Err(BlindVaultReplicaCoordinatorError::Rejected);
        }
        let commitment = Sha256::digest(&canonical_bytes).into();
        Ok(Self {
            target_node_id,
            canonical_effects,
            canonical_bytes,
            commitment,
        })
    }

    pub(crate) const fn target_node_id(&self) -> [u8; 32] {
        self.target_node_id
    }

    pub(crate) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    pub(crate) fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }

    pub(crate) fn canonical_effects(&self) -> &[Vec<u8>; REPLICA_TARGET_EFFECT_COUNT] {
        &self.canonical_effects
    }
}

impl Drop for BlindVaultReplicaTargetBundleV1 {
    fn drop(&mut self) {
        for effect in &mut self.canonical_effects {
            effect.zeroize();
        }
        self.canonical_bytes.zeroize();
        self.commitment.zeroize();
        self.target_node_id.zeroize();
    }
}

/// The only admissible source-authority and target-bundle pairing.
pub(crate) struct BlindVaultReplicaJobSubmissionV1 {
    authorization: BlindVaultReplicaJobAuthorizationV1,
    target_bundle: BlindVaultReplicaTargetBundleV1,
}

impl BlindVaultReplicaJobSubmissionV1 {
    pub(crate) fn new(
        authorization: BlindVaultReplicaJobAuthorizationV1,
        target_bundle: BlindVaultReplicaTargetBundleV1,
    ) -> Result<Self, BlindVaultReplicaCoordinatorError> {
        let claims = authorization.claims();
        if claims.target_node_id() != target_bundle.target_node_id()
            || claims.target_bundle_commitment() != target_bundle.commitment()
        {
            return Err(BlindVaultReplicaCoordinatorError::Rejected);
        }
        Ok(Self {
            authorization,
            target_bundle,
        })
    }

    fn authorization(&self) -> &BlindVaultReplicaJobAuthorizationV1 {
        &self.authorization
    }
}

/// Immutable source-private record handed to a durable exact-idempotent store.
pub(crate) struct BlindVaultReplicaStagedJobV1 {
    job_id: [u8; 16],
    claims_commitment: [u8; 32],
    target_node_id: [u8; 32],
    target_bundle_commitment: [u8; 32],
    canonical_authorization: Vec<u8>,
    canonical_target_bundle: Vec<u8>,
    record_commitment: [u8; 32],
}

impl BlindVaultReplicaStagedJobV1 {
    fn from_submission(submission: &BlindVaultReplicaJobSubmissionV1) -> Self {
        let claims = submission.authorization.claims();
        let canonical_authorization = submission.authorization.canonical_authorization_bytes();
        let canonical_target_bundle = submission.target_bundle.canonical_bytes().to_vec();
        let record_commitment =
            staged_job_commitment(&canonical_authorization, &canonical_target_bundle);
        Self {
            job_id: claims.job_id(),
            claims_commitment: claims.commitment(),
            target_node_id: claims.target_node_id(),
            target_bundle_commitment: claims.target_bundle_commitment(),
            canonical_authorization,
            canonical_target_bundle,
            record_commitment,
        }
    }

    pub(crate) const fn job_id(&self) -> [u8; 16] {
        self.job_id
    }

    pub(crate) const fn record_commitment(&self) -> [u8; 32] {
        self.record_commitment
    }

    fn exactly_matches(&self, other: &Self) -> bool {
        self.job_id == other.job_id
            && self.claims_commitment == other.claims_commitment
            && self.target_node_id == other.target_node_id
            && self.target_bundle_commitment == other.target_bundle_commitment
            && self.canonical_authorization == other.canonical_authorization
            && self.canonical_target_bundle == other.canonical_target_bundle
            && self.record_commitment == other.record_commitment
    }
}

impl Drop for BlindVaultReplicaStagedJobV1 {
    fn drop(&mut self) {
        self.job_id.zeroize();
        self.claims_commitment.zeroize();
        self.target_node_id.zeroize();
        self.target_bundle_commitment.zeroize();
        self.canonical_authorization.zeroize();
        self.canonical_target_bundle.zeroize();
        self.record_commitment.zeroize();
    }
}

/// Exact stage result for one durable random job identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlindVaultReplicaStageOutcome {
    Inserted,
    Existing,
    Conflict,
}

/// Coarse admission result safe to expose to an authenticated source client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlindVaultReplicaAdmissionOutcome {
    Inserted,
    Existing,
}

/// Replaceable source-lease authority verification capability.
pub(crate) trait BlindVaultReplicaAuthorizationVerifier: Send + Sync {
    fn verify(
        &self,
        authorization: &BlindVaultReplicaJobAuthorizationV1,
        now_ms: u64,
    ) -> Result<(), BlindVaultReplicaJobAuthorizationError>;
}

impl BlindVaultReplicaAuthorizationVerifier for BlindVaultService {
    fn verify(
        &self,
        authorization: &BlindVaultReplicaJobAuthorizationV1,
        now_ms: u64,
    ) -> Result<(), BlindVaultReplicaJobAuthorizationError> {
        self.verify_replica_job_authorization(authorization, now_ms)
    }
}

impl BlindVaultReplicaAuthorizationVerifier for Arc<BlindVaultService> {
    fn verify(
        &self,
        authorization: &BlindVaultReplicaJobAuthorizationV1,
        now_ms: u64,
    ) -> Result<(), BlindVaultReplicaJobAuthorizationError> {
        self.as_ref()
            .verify_replica_job_authorization(authorization, now_ms)
    }
}

/// Durable store contract: same id/same bytes is idempotent, any drift conflicts.
pub(crate) trait BlindVaultReplicaJobStore: Send + Sync {
    type Error: Send + Sync;

    fn load_active(&self) -> Result<Option<BlindVaultReplicaStagedJobV1>, Self::Error>;

    fn stage_exact(
        &self,
        record: &BlindVaultReplicaStagedJobV1,
    ) -> Result<BlindVaultReplicaStageOutcome, Self::Error>;
}

/// Source-local coordinator for admission only; it performs no network work.
pub(crate) struct BlindVaultReplicaCoordinator<V, S> {
    verifier: V,
    store: S,
}

/// Object-safe admission boundary consumed by the optional client API.
pub(crate) trait BlindVaultReplicaJobAdmission: Send + Sync {
    fn admit_v1(
        &self,
        submission: BlindVaultReplicaJobSubmissionV1,
        now_ms: u64,
    ) -> Result<BlindVaultReplicaAdmissionOutcome, BlindVaultReplicaCoordinatorError>;
}

impl<V, S> BlindVaultReplicaCoordinator<V, S>
where
    V: BlindVaultReplicaAuthorizationVerifier,
    S: BlindVaultReplicaJobStore,
{
    pub(crate) const fn new(verifier: V, store: S) -> Self {
        Self { verifier, store }
    }

    pub(crate) fn admit_v1(
        &self,
        submission: BlindVaultReplicaJobSubmissionV1,
        now_ms: u64,
    ) -> Result<BlindVaultReplicaAdmissionOutcome, BlindVaultReplicaCoordinatorError> {
        // [BLIND-VAULT-REPLICA-COORDINATOR 2026-09-01 by Codex] The first
        // verification rejects stale submissions before allocation. The second
        // is deliberately adjacent to durable staging and has no await,
        // callback, or network boundary between authority and use.
        self.verifier
            .verify(submission.authorization(), now_ms)
            .map_err(map_authorization_error)?;
        let staged = BlindVaultReplicaStagedJobV1::from_submission(&submission);
        self.verifier
            .verify(submission.authorization(), now_ms)
            .map_err(map_authorization_error)?;
        match self
            .store
            .stage_exact(&staged)
            .map_err(|_| BlindVaultReplicaCoordinatorError::Unavailable)?
        {
            BlindVaultReplicaStageOutcome::Inserted => {
                Ok(BlindVaultReplicaAdmissionOutcome::Inserted)
            }
            BlindVaultReplicaStageOutcome::Existing => {
                Ok(BlindVaultReplicaAdmissionOutcome::Existing)
            }
            BlindVaultReplicaStageOutcome::Conflict => {
                Err(BlindVaultReplicaCoordinatorError::Rejected)
            }
        }
    }
}

impl<V, S> BlindVaultReplicaJobAdmission for BlindVaultReplicaCoordinator<V, S>
where
    V: BlindVaultReplicaAuthorizationVerifier,
    S: BlindVaultReplicaJobStore,
{
    fn admit_v1(
        &self,
        submission: BlindVaultReplicaJobSubmissionV1,
        now_ms: u64,
    ) -> Result<BlindVaultReplicaAdmissionOutcome, BlindVaultReplicaCoordinatorError> {
        BlindVaultReplicaCoordinator::admit_v1(self, submission, now_ms)
    }
}

/// Coarse, privacy-safe admission failure.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindVaultReplicaCoordinatorError {
    Rejected,
    Unavailable,
}

impl fmt::Display for BlindVaultReplicaCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Rejected => "blind vault replica job rejected",
            Self::Unavailable => "blind vault replica job unavailable",
        })
    }
}

impl fmt::Debug for BlindVaultReplicaCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl std::error::Error for BlindVaultReplicaCoordinatorError {}

fn map_authorization_error(
    error: BlindVaultReplicaJobAuthorizationError,
) -> BlindVaultReplicaCoordinatorError {
    match error {
        BlindVaultReplicaJobAuthorizationError::Rejected => {
            BlindVaultReplicaCoordinatorError::Rejected
        }
        BlindVaultReplicaJobAuthorizationError::Unavailable => {
            BlindVaultReplicaCoordinatorError::Unavailable
        }
    }
}

fn canonical_bundle_bytes(
    target_node_id: [u8; 32],
    effects: &[Vec<u8>; REPLICA_TARGET_EFFECT_COUNT],
) -> Result<Vec<u8>, BlindVaultReplicaCoordinatorError> {
    let purposes = [
        BlindVaultReplicaTargetEffectPurpose::LeaseAdmission,
        BlindVaultReplicaTargetEffectPurpose::Put,
        BlindVaultReplicaTargetEffectPurpose::LeaseInventory,
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(REPLICA_TARGET_BUNDLE_DOMAIN);
    bytes.extend_from_slice(&REPLICA_TARGET_BUNDLE_VERSION_V1.to_be_bytes());
    bytes.extend_from_slice(&target_node_id);
    bytes.push(REPLICA_TARGET_EFFECT_COUNT as u8);
    for (ordinal, (purpose, effect)) in purposes.iter().zip(effects).enumerate() {
        let length =
            u32::try_from(effect.len()).map_err(|_| BlindVaultReplicaCoordinatorError::Rejected)?;
        bytes.push(ordinal as u8);
        bytes.push(*purpose as u8);
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(effect);
    }
    Ok(bytes)
}

fn staged_job_commitment(authorization: &[u8], target_bundle: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(REPLICA_STAGED_JOB_DOMAIN);
    hasher.update((authorization.len() as u32).to_be_bytes());
    hasher.update(authorization);
    hasher.update((target_bundle.len() as u32).to_be_bytes());
    hasher.update(target_bundle);
    hasher.finalize().into()
}

#[cfg(unix)]
/// Fail-closed local errors for the single-active-job durable store.
pub(crate) enum BlindVaultReplicaJobFileStoreError {
    Host(PrivateRecoveryIoError),
    CorruptState,
    TooLarge,
    Unavailable,
}

#[cfg(unix)]
impl fmt::Display for BlindVaultReplicaJobFileStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Host(_) | Self::Unavailable => "blind vault replica job store unavailable",
            Self::CorruptState => "blind vault replica job store is corrupt",
            Self::TooLarge => "blind vault replica job store exceeds its bound",
        })
    }
}

#[cfg(unix)]
impl fmt::Debug for BlindVaultReplicaJobFileStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

#[cfg(unix)]
impl std::error::Error for BlindVaultReplicaJobFileStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Host(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(unix)]
impl From<PrivateRecoveryIoError> for BlindVaultReplicaJobFileStoreError {
    fn from(error: PrivateRecoveryIoError) -> Self {
        Self::Host(error)
    }
}

/// Single-writer durable store for the intentionally single-active M12 job.
///
/// [BLIND-VAULT-REPLICA-JOB-STORE 2026-09-01 by Codex] The adapter reuses the
/// audited private atomic file host. A different job cannot replace an active
/// generation; exact same bytes are the only accepted retry.
#[cfg(unix)]
pub(crate) struct FileBlindVaultReplicaJobStore {
    file: std::sync::Mutex<PrivateAtomicRecoveryFile>,
}

#[cfg(unix)]
impl FileBlindVaultReplicaJobStore {
    pub(crate) fn open(directory: &Path) -> Result<Self, BlindVaultReplicaJobFileStoreError> {
        let file = PrivateAtomicRecoveryFile::open(directory)?;
        if let Some(bytes) = file.read(MAX_REPLICA_JOB_FILE_BYTES)? {
            let bytes = Zeroizing::new(bytes);
            decode_replica_job_file(bytes.as_slice())?;
        }
        Ok(Self {
            file: std::sync::Mutex::new(file),
        })
    }

    fn lock_file(
        &self,
    ) -> Result<
        std::sync::MutexGuard<'_, PrivateAtomicRecoveryFile>,
        BlindVaultReplicaJobFileStoreError,
    > {
        self.file
            .lock()
            .map_err(|_| BlindVaultReplicaJobFileStoreError::Unavailable)
    }

    fn load_from(
        file: &PrivateAtomicRecoveryFile,
    ) -> Result<Option<BlindVaultReplicaStagedJobV1>, BlindVaultReplicaJobFileStoreError> {
        let Some(bytes) = file.read(MAX_REPLICA_JOB_FILE_BYTES)? else {
            return Ok(None);
        };
        let bytes = Zeroizing::new(bytes);
        decode_replica_job_file(bytes.as_slice()).map(Some)
    }
}

#[cfg(unix)]
impl BlindVaultReplicaJobStore for FileBlindVaultReplicaJobStore {
    type Error = BlindVaultReplicaJobFileStoreError;

    fn load_active(&self) -> Result<Option<BlindVaultReplicaStagedJobV1>, Self::Error> {
        let file = self.lock_file()?;
        Self::load_from(&file)
    }

    fn stage_exact(
        &self,
        record: &BlindVaultReplicaStagedJobV1,
    ) -> Result<BlindVaultReplicaStageOutcome, Self::Error> {
        validate_staged_job(record)?;
        let file = self.lock_file()?;
        match Self::load_from(&file)? {
            None => {
                let encoded = Zeroizing::new(encode_replica_job_file(record)?);
                file.replace(encoded.as_slice(), MAX_REPLICA_JOB_FILE_BYTES)?;
                Ok(BlindVaultReplicaStageOutcome::Inserted)
            }
            Some(current) if current.exactly_matches(record) => {
                file.confirm_current_durable()?;
                Ok(BlindVaultReplicaStageOutcome::Existing)
            }
            Some(_) => Ok(BlindVaultReplicaStageOutcome::Conflict),
        }
    }
}

#[cfg(unix)]
fn validate_staged_job(
    record: &BlindVaultReplicaStagedJobV1,
) -> Result<(), BlindVaultReplicaJobFileStoreError> {
    if record.job_id == [0; 16]
        || record.claims_commitment == [0; 32]
        || record.target_node_id == [0; 32]
        || record.target_bundle_commitment == [0; 32]
        || record.record_commitment == [0; 32]
        || record.canonical_authorization.is_empty()
        || record.canonical_authorization.len() > MAX_REPLICA_JOB_AUTHORIZATION_BYTES
        || record.canonical_target_bundle.is_empty()
        || record.canonical_target_bundle.len() > MAX_REPLICA_TARGET_BUNDLE_BYTES
        || IdentityPublicKey::from_bytes(&record.target_node_id).is_err()
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }

    let authorization = BlindVaultReplicaJobAuthorizationV1::from_canonical_authorization_bytes(
        &record.canonical_authorization,
    )
    .map_err(|_| BlindVaultReplicaJobFileStoreError::CorruptState)?;
    let claims = authorization.claims();
    if authorization.canonical_authorization_bytes() != record.canonical_authorization
        || claims.job_id() != record.job_id
        || claims.commitment() != record.claims_commitment
        || claims.target_node_id() != record.target_node_id
        || claims.target_bundle_commitment() != record.target_bundle_commitment
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }

    validate_canonical_target_bundle(&record.canonical_target_bundle, record.target_node_id)?;
    if <[u8; 32]>::from(Sha256::digest(&record.canonical_target_bundle))
        != record.target_bundle_commitment
        || staged_job_commitment(
            &record.canonical_authorization,
            &record.canonical_target_bundle,
        ) != record.record_commitment
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }
    Ok(())
}

#[cfg(unix)]
fn validate_canonical_target_bundle(
    bytes: &[u8],
    expected_target_node_id: [u8; 32],
) -> Result<(), BlindVaultReplicaJobFileStoreError> {
    if bytes.len() > MAX_REPLICA_TARGET_BUNDLE_BYTES
        || !bytes.starts_with(REPLICA_TARGET_BUNDLE_DOMAIN)
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }
    let mut cursor = REPLICA_TARGET_BUNDLE_DOMAIN.len();
    let version = u16::from_be_bytes(replica_job_take(bytes, &mut cursor)?);
    let target_node_id = replica_job_take(bytes, &mut cursor)?;
    let effect_count = *bytes
        .get(cursor)
        .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?;
    cursor += 1;
    if version != REPLICA_TARGET_BUNDLE_VERSION_V1
        || target_node_id != expected_target_node_id
        || usize::from(effect_count) != REPLICA_TARGET_EFFECT_COUNT
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }

    let expected = [
        (BlindVaultReplicaTargetEffectPurpose::LeaseAdmission, 0_u8),
        (BlindVaultReplicaTargetEffectPurpose::Put, 1_u8),
        (BlindVaultReplicaTargetEffectPurpose::LeaseInventory, 2_u8),
    ];
    for (purpose, ordinal) in expected {
        let stored_ordinal = *bytes
            .get(cursor)
            .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?;
        cursor += 1;
        let stored_purpose = *bytes
            .get(cursor)
            .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?;
        cursor += 1;
        let effect_len = usize::try_from(u32::from_be_bytes(replica_job_take(bytes, &mut cursor)?))
            .map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
        let end = cursor
            .checked_add(effect_len)
            .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
        let effect = bytes
            .get(cursor..end)
            .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?;
        cursor = end;
        if stored_ordinal != ordinal || stored_purpose != purpose as u8 || effect.is_empty() {
            return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
        }
        let frame = decode_blind_vault_frame(effect)
            .map_err(|_| BlindVaultReplicaJobFileStoreError::CorruptState)?;
        let expected_kind = matches!(
            (purpose, &frame),
            (
                BlindVaultReplicaTargetEffectPurpose::LeaseAdmission,
                BlindVaultFrame::BlindLeaseAdmission(_)
            ) | (
                BlindVaultReplicaTargetEffectPurpose::Put,
                BlindVaultFrame::Put(_)
            ) | (
                BlindVaultReplicaTargetEffectPurpose::LeaseInventory,
                BlindVaultFrame::LeaseInventory(_)
            )
        );
        let canonical = encode_blind_vault_frame(&frame)
            .map_err(|_| BlindVaultReplicaJobFileStoreError::CorruptState)?;
        if !expected_kind || canonical != effect {
            return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
        }
    }
    if cursor != bytes.len() {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }
    Ok(())
}

#[cfg(unix)]
fn encode_replica_job_file(
    record: &BlindVaultReplicaStagedJobV1,
) -> Result<Vec<u8>, BlindVaultReplicaJobFileStoreError> {
    validate_staged_job(record)?;
    let authorization_len = u32::try_from(record.canonical_authorization.len())
        .map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let bundle_len = u32::try_from(record.canonical_target_bundle.len())
        .map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let body_len = REPLICA_JOB_BODY_FIXED_BYTES
        .checked_add(record.canonical_authorization.len())
        .and_then(|length| length.checked_add(record.canonical_target_bundle.len()))
        .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let body_len_u32 =
        u32::try_from(body_len).map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let total_len = REPLICA_JOB_FILE_HEADER_BYTES
        .checked_add(body_len)
        .and_then(|length| length.checked_add(REPLICA_JOB_FILE_CHECKSUM_BYTES))
        .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
    if total_len > MAX_REPLICA_JOB_FILE_BYTES {
        return Err(BlindVaultReplicaJobFileStoreError::TooLarge);
    }

    let mut encoded = Vec::with_capacity(total_len);
    encoded.extend_from_slice(&REPLICA_JOB_FILE_MAGIC);
    encoded.extend_from_slice(&REPLICA_JOB_FILE_VERSION_V1.to_be_bytes());
    encoded.extend_from_slice(&body_len_u32.to_be_bytes());
    encoded.extend_from_slice(&record.job_id);
    encoded.extend_from_slice(&record.claims_commitment);
    encoded.extend_from_slice(&record.target_node_id);
    encoded.extend_from_slice(&record.target_bundle_commitment);
    encoded.extend_from_slice(&authorization_len.to_be_bytes());
    encoded.extend_from_slice(&bundle_len.to_be_bytes());
    encoded.extend_from_slice(&record.canonical_authorization);
    encoded.extend_from_slice(&record.canonical_target_bundle);
    encoded.extend_from_slice(&record.record_commitment);
    let mut checksum = Sha256::new();
    checksum.update(REPLICA_JOB_FILE_CHECKSUM_DOMAIN);
    checksum.update(&encoded);
    encoded.extend_from_slice(&checksum.finalize());
    Ok(encoded)
}

#[cfg(unix)]
fn decode_replica_job_file(
    bytes: &[u8],
) -> Result<BlindVaultReplicaStagedJobV1, BlindVaultReplicaJobFileStoreError> {
    if bytes.len() > MAX_REPLICA_JOB_FILE_BYTES
        || bytes.len()
            < REPLICA_JOB_FILE_HEADER_BYTES
                + REPLICA_JOB_BODY_FIXED_BYTES
                + REPLICA_JOB_FILE_CHECKSUM_BYTES
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }
    let mut cursor = 0;
    let magic = replica_job_take(bytes, &mut cursor)?;
    let version = u16::from_be_bytes(replica_job_take(bytes, &mut cursor)?);
    let body_len = usize::try_from(u32::from_be_bytes(replica_job_take(bytes, &mut cursor)?))
        .map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let expected_total = REPLICA_JOB_FILE_HEADER_BYTES
        .checked_add(body_len)
        .and_then(|length| length.checked_add(REPLICA_JOB_FILE_CHECKSUM_BYTES))
        .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
    if magic != REPLICA_JOB_FILE_MAGIC
        || version != REPLICA_JOB_FILE_VERSION_V1
        || expected_total != bytes.len()
    {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }

    let checksum_offset = bytes
        .len()
        .checked_sub(REPLICA_JOB_FILE_CHECKSUM_BYTES)
        .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?;
    let stored_checksum: [u8; 32] = bytes[checksum_offset..]
        .try_into()
        .map_err(|_| BlindVaultReplicaJobFileStoreError::CorruptState)?;
    let mut checksum = Sha256::new();
    checksum.update(REPLICA_JOB_FILE_CHECKSUM_DOMAIN);
    checksum.update(&bytes[..checksum_offset]);
    if <[u8; 32]>::from(checksum.finalize()) != stored_checksum {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }

    let job_id = replica_job_take(bytes, &mut cursor)?;
    let claims_commitment = replica_job_take(bytes, &mut cursor)?;
    let target_node_id = replica_job_take(bytes, &mut cursor)?;
    let target_bundle_commitment = replica_job_take(bytes, &mut cursor)?;
    let authorization_len =
        usize::try_from(u32::from_be_bytes(replica_job_take(bytes, &mut cursor)?))
            .map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let bundle_len = usize::try_from(u32::from_be_bytes(replica_job_take(bytes, &mut cursor)?))
        .map_err(|_| BlindVaultReplicaJobFileStoreError::TooLarge)?;
    if authorization_len == 0
        || authorization_len > MAX_REPLICA_JOB_AUTHORIZATION_BYTES
        || bundle_len == 0
        || bundle_len > MAX_REPLICA_TARGET_BUNDLE_BYTES
    {
        return Err(BlindVaultReplicaJobFileStoreError::TooLarge);
    }
    let authorization_end = cursor
        .checked_add(authorization_len)
        .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let canonical_authorization = bytes
        .get(cursor..authorization_end)
        .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?
        .to_vec();
    cursor = authorization_end;
    let bundle_end = cursor
        .checked_add(bundle_len)
        .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let canonical_target_bundle = bytes
        .get(cursor..bundle_end)
        .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?
        .to_vec();
    cursor = bundle_end;
    let record_commitment = replica_job_take(bytes, &mut cursor)?;
    if cursor != checksum_offset {
        return Err(BlindVaultReplicaJobFileStoreError::CorruptState);
    }
    let record = BlindVaultReplicaStagedJobV1 {
        job_id,
        claims_commitment,
        target_node_id,
        target_bundle_commitment,
        canonical_authorization,
        canonical_target_bundle,
        record_commitment,
    };
    validate_staged_job(&record)?;
    Ok(record)
}

#[cfg(unix)]
fn replica_job_take<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], BlindVaultReplicaJobFileStoreError> {
    let end = cursor
        .checked_add(N)
        .ok_or(BlindVaultReplicaJobFileStoreError::TooLarge)?;
    let value = bytes
        .get(*cursor..end)
        .ok_or(BlindVaultReplicaJobFileStoreError::CorruptState)?
        .try_into()
        .map_err(|_| BlindVaultReplicaJobFileStoreError::CorruptState)?;
    *cursor = end;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultBlindAdmissionToken, BlindVaultBlindLeaseAdmissionRequest,
        BlindVaultLeaseCreateRequest, BlindVaultLeaseInventoryRequest, BlindVaultPutRequest,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    const NOW_MS: u64 = 2_000_000;

    struct Fixture {
        source_admin: IdentityKeyPair,
        target_write: IdentityKeyPair,
        target_node_id: [u8; 32],
        bundle: BlindVaultReplicaTargetBundleV1,
    }

    impl Fixture {
        fn new() -> Self {
            let write = IdentityKeyPair::from_bytes(&[11; 32]).expect("write key");
            let target_admin = IdentityKeyPair::from_bytes(&[12; 32]).expect("target admin");
            let source_admin = IdentityKeyPair::from_bytes(&[13; 32]).expect("source admin");
            let target = IdentityKeyPair::from_bytes(&[14; 32]).expect("target node");
            let lease_id = [21; 32];
            let lease_expiry = NOW_MS + 60_000;
            let mut lease = BlindVaultLeaseCreateRequest::new(
                lease_id,
                [22; 16],
                write.public_key_bytes(),
                target_admin.public_key_bytes(),
                [23; 32],
                lease_expiry,
            );
            lease.sign(&target_admin).expect("sign lease");
            let admission = BlindVaultBlindLeaseAdmissionRequest {
                admission: BlindVaultBlindAdmissionToken::new(
                    [24; 32],
                    [25; 32],
                    [26; 32],
                    vec![27; 256],
                ),
                lease,
            };
            let mut put = BlindVaultPutRequest::new(
                lease_id,
                [28; 32],
                [29; 16],
                vec![30; REPLICA_TARGET_OBJECT_BYTES],
                lease_expiry,
            );
            put.sign(&write);
            let mut inventory = BlindVaultLeaseInventoryRequest::new(lease_id, [31; 16], NOW_MS);
            inventory.sign(&target_admin);
            let admission =
                encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAdmission(admission))
                    .expect("encode admission");
            let put = encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode put");
            let inventory = encode_blind_vault_frame(&BlindVaultFrame::LeaseInventory(inventory))
                .expect("encode inventory");
            let policy =
                BlindVaultReplicaTargetBundlePolicyV1::new(NOW_MS, 120_000, 120_000, 1_000)
                    .expect("policy");
            let target_node_id = target.public_key_bytes();
            let bundle = BlindVaultReplicaTargetBundleV1::from_wire_parts(
                REPLICA_TARGET_BUNDLE_VERSION_V1,
                target_node_id,
                &admission,
                &put,
                &inventory,
                policy,
            )
            .expect("bundle");
            Self {
                source_admin,
                target_write: write,
                target_node_id,
                bundle,
            }
        }

        fn authorization(
            &self,
            job_id: [u8; 16],
            bundle_commitment: [u8; 32],
        ) -> BlindVaultReplicaJobAuthorizationV1 {
            let unsigned = BlindVaultReplicaJobAuthorizationV1::from_wire_parts(
                1,
                job_id,
                [41; 32],
                [42; 32],
                [43; 32],
                self.target_node_id,
                bundle_commitment,
                NOW_MS - 1,
                NOW_MS + 30_000,
                [0; 64],
            )
            .expect("unsigned authorization");
            let signature = self.source_admin.sign(&unsigned.signing_bytes());
            BlindVaultReplicaJobAuthorizationV1::from_wire_parts(
                1,
                job_id,
                [41; 32],
                [42; 32],
                [43; 32],
                self.target_node_id,
                bundle_commitment,
                NOW_MS - 1,
                NOW_MS + 30_000,
                signature,
            )
            .expect("authorization")
        }
    }

    struct CountingVerifier {
        calls: AtomicUsize,
        reject_on_call: Option<usize>,
    }

    impl CountingVerifier {
        fn accepting() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                reject_on_call: None,
            }
        }

        fn reject_on(call: usize) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                reject_on_call: Some(call),
            }
        }
    }

    impl BlindVaultReplicaAuthorizationVerifier for CountingVerifier {
        fn verify(
            &self,
            _authorization: &BlindVaultReplicaJobAuthorizationV1,
            _now_ms: u64,
        ) -> Result<(), BlindVaultReplicaJobAuthorizationError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.reject_on_call == Some(call) {
                Err(BlindVaultReplicaJobAuthorizationError::Rejected)
            } else {
                Ok(())
            }
        }
    }

    #[derive(Default)]
    struct ExactMemoryStore {
        records: Mutex<HashMap<[u8; 16], [u8; 32]>>,
        stage_calls: AtomicUsize,
    }

    impl BlindVaultReplicaJobStore for ExactMemoryStore {
        type Error = ();

        fn load_active(&self) -> Result<Option<BlindVaultReplicaStagedJobV1>, Self::Error> {
            Ok(None)
        }

        fn stage_exact(
            &self,
            record: &BlindVaultReplicaStagedJobV1,
        ) -> Result<BlindVaultReplicaStageOutcome, Self::Error> {
            self.stage_calls.fetch_add(1, Ordering::SeqCst);
            let mut records = self.records.lock().map_err(|_| ())?;
            Ok(match records.get(&record.job_id()) {
                None => {
                    records.insert(record.job_id(), record.record_commitment());
                    BlindVaultReplicaStageOutcome::Inserted
                }
                Some(commitment) if *commitment == record.record_commitment() => {
                    BlindVaultReplicaStageOutcome::Existing
                }
                Some(_) => BlindVaultReplicaStageOutcome::Conflict,
            })
        }
    }

    impl ExactMemoryStore {
        fn with_record(job_id: [u8; 16], commitment: [u8; 32]) -> Self {
            Self {
                records: Mutex::new(HashMap::from([(job_id, commitment)])),
                stage_calls: AtomicUsize::new(0),
            }
        }
    }

    #[test]
    fn canonical_bundle_binds_target_and_ordered_effects() {
        let fixture = Fixture::new();
        assert_eq!(fixture.bundle.target_node_id(), fixture.target_node_id);
        assert_ne!(fixture.bundle.commitment(), [0; 32]);
        assert_eq!(
            fixture.bundle.canonical_effects().len(),
            REPLICA_TARGET_EFFECT_COUNT
        );
        assert!(fixture
            .bundle
            .canonical_bytes()
            .starts_with(REPLICA_TARGET_BUNDLE_DOMAIN));
    }

    #[test]
    fn bundle_rejects_unknown_version_wrong_order_and_trailing_bytes() {
        let fixture = Fixture::new();
        let effects = fixture.bundle.canonical_effects();
        let policy = BlindVaultReplicaTargetBundlePolicyV1::new(NOW_MS, 120_000, 120_000, 1_000)
            .expect("policy");
        assert!(matches!(
            BlindVaultReplicaTargetBundleV1::from_wire_parts(
                2,
                fixture.target_node_id,
                &effects[0],
                &effects[1],
                &effects[2],
                policy,
            ),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
        assert!(matches!(
            BlindVaultReplicaTargetBundleV1::from_wire_parts(
                1,
                fixture.target_node_id,
                &effects[1],
                &effects[0],
                &effects[2],
                policy,
            ),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
        let mut trailing = effects[2].clone();
        trailing.push(0);
        assert!(matches!(
            BlindVaultReplicaTargetBundleV1::from_wire_parts(
                1,
                fixture.target_node_id,
                &effects[0],
                &effects[1],
                &trailing,
                policy,
            ),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
    }

    #[test]
    fn bundle_rejects_non_class_ciphertext_size() {
        let fixture = Fixture::new();
        let effects = fixture.bundle.canonical_effects();
        let mut put = match decode_blind_vault_frame(&effects[1]).expect("decode put") {
            BlindVaultFrame::Put(request) => request,
            _ => panic!("fixture put frame"),
        };
        put.ciphertext.pop();
        put.sign(&fixture.target_write);
        let short_put =
            encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode short put");
        let policy = BlindVaultReplicaTargetBundlePolicyV1::new(NOW_MS, 120_000, 120_000, 1_000)
            .expect("policy");

        assert!(matches!(
            BlindVaultReplicaTargetBundleV1::from_wire_parts(
                1,
                fixture.target_node_id,
                &effects[0],
                &short_put,
                &effects[2],
                policy,
            ),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
    }

    #[test]
    fn submission_rejects_target_bundle_substitution() {
        let fixture = Fixture::new();
        let wrong = fixture.authorization([51; 16], [52; 32]);
        assert!(matches!(
            BlindVaultReplicaJobSubmissionV1::new(wrong, fixture.bundle),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
    }

    #[test]
    fn coordinator_stages_exact_retry_once() {
        let fixture = Fixture::new();
        let authorization = fixture.authorization([61; 16], fixture.bundle.commitment());
        let submission = BlindVaultReplicaJobSubmissionV1::new(authorization, fixture.bundle)
            .expect("submission");
        let coordinator = BlindVaultReplicaCoordinator::new(
            CountingVerifier::accepting(),
            ExactMemoryStore::default(),
        );
        assert_eq!(
            coordinator.admit_v1(submission, NOW_MS).expect("insert"),
            BlindVaultReplicaAdmissionOutcome::Inserted
        );

        let fixture = Fixture::new();
        let authorization = fixture.authorization([61; 16], fixture.bundle.commitment());
        let retry = BlindVaultReplicaJobSubmissionV1::new(authorization, fixture.bundle)
            .expect("retry submission");
        assert_eq!(
            coordinator.admit_v1(retry, NOW_MS).expect("existing"),
            BlindVaultReplicaAdmissionOutcome::Existing
        );

        assert_eq!(coordinator.store.stage_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn coordinator_rejects_same_job_id_with_different_durable_record() {
        let fixture = Fixture::new();
        let job_id = [62; 16];
        let authorization = fixture.authorization(job_id, fixture.bundle.commitment());
        let submission = BlindVaultReplicaJobSubmissionV1::new(authorization, fixture.bundle)
            .expect("submission");
        let coordinator = BlindVaultReplicaCoordinator::new(
            CountingVerifier::accepting(),
            ExactMemoryStore::with_record(job_id, [99; 32]),
        );

        assert!(matches!(
            coordinator.admit_v1(submission, NOW_MS),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
        assert_eq!(coordinator.store.stage_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn second_authority_failure_has_zero_stage_side_effect() {
        let fixture = Fixture::new();
        let authorization = fixture.authorization([71; 16], fixture.bundle.commitment());
        let submission = BlindVaultReplicaJobSubmissionV1::new(authorization, fixture.bundle)
            .expect("submission");
        let coordinator = BlindVaultReplicaCoordinator::new(
            CountingVerifier::reject_on(2),
            ExactMemoryStore::default(),
        );
        assert!(matches!(
            coordinator.admit_v1(submission, NOW_MS),
            Err(BlindVaultReplicaCoordinatorError::Rejected)
        ));
        assert_eq!(coordinator.store.stage_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn diagnostics_are_coarse_and_redacted() {
        assert_eq!(
            format!("{:?}", BlindVaultReplicaCoordinatorError::Rejected),
            "blind vault replica job rejected"
        );
        assert_eq!(
            format!("{:?}", BlindVaultReplicaCoordinatorError::Unavailable),
            "blind vault replica job unavailable"
        );
    }

    #[cfg(unix)]
    #[test]
    fn file_store_reopens_exact_job_and_rejects_parallel_job() {
        let root = tempfile::tempdir().expect("temp root");
        let directory = root
            .path()
            .canonicalize()
            .expect("canonical root")
            .join("jobs");
        let first_fixture = Fixture::new();
        let first_authorization =
            first_fixture.authorization([81; 16], first_fixture.bundle.commitment());
        let first_submission =
            BlindVaultReplicaJobSubmissionV1::new(first_authorization, first_fixture.bundle)
                .expect("first submission");
        let first_record = BlindVaultReplicaStagedJobV1::from_submission(&first_submission);

        {
            let store = FileBlindVaultReplicaJobStore::open(&directory).expect("open store");
            assert_eq!(
                store.stage_exact(&first_record).expect("insert"),
                BlindVaultReplicaStageOutcome::Inserted
            );
            assert_eq!(
                store.stage_exact(&first_record).expect("exact retry"),
                BlindVaultReplicaStageOutcome::Existing
            );
            let loaded = store.load_active().expect("load").expect("active job");
            assert!(loaded.exactly_matches(&first_record));
        }

        let store = FileBlindVaultReplicaJobStore::open(&directory).expect("reopen store");
        let loaded = store.load_active().expect("reload").expect("active job");
        assert!(loaded.exactly_matches(&first_record));

        let second_fixture = Fixture::new();
        let second_authorization =
            second_fixture.authorization([82; 16], second_fixture.bundle.commitment());
        let second_submission =
            BlindVaultReplicaJobSubmissionV1::new(second_authorization, second_fixture.bundle)
                .expect("second submission");
        let second_record = BlindVaultReplicaStagedJobV1::from_submission(&second_submission);
        assert_eq!(
            store.stage_exact(&second_record).expect("active conflict"),
            BlindVaultReplicaStageOutcome::Conflict
        );
    }

    #[cfg(unix)]
    #[test]
    fn file_store_rejects_corrupt_generation_on_reopen() {
        use std::fs::OpenOptions;
        use std::io::{Read, Seek, SeekFrom, Write};

        let root = tempfile::tempdir().expect("temp root");
        let directory = root
            .path()
            .canonicalize()
            .expect("canonical root")
            .join("jobs");
        let fixture = Fixture::new();
        let authorization = fixture.authorization([83; 16], fixture.bundle.commitment());
        let submission = BlindVaultReplicaJobSubmissionV1::new(authorization, fixture.bundle)
            .expect("submission");
        let record = BlindVaultReplicaStagedJobV1::from_submission(&submission);
        {
            let store = FileBlindVaultReplicaJobStore::open(&directory).expect("open store");
            assert_eq!(
                store.stage_exact(&record).expect("insert"),
                BlindVaultReplicaStageOutcome::Inserted
            );
        }

        let state_path = directory.join("recovery-state-v1.bin");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(state_path)
            .expect("open state");
        let mut first_body_byte = [0_u8; 1];
        file.seek(SeekFrom::Start(REPLICA_JOB_FILE_HEADER_BYTES as u64))
            .expect("seek body");
        file.read_exact(&mut first_body_byte).expect("read body");
        first_body_byte[0] ^= 1;
        file.seek(SeekFrom::Start(REPLICA_JOB_FILE_HEADER_BYTES as u64))
            .expect("seek body again");
        file.write_all(&first_body_byte).expect("corrupt body");
        file.sync_all().expect("sync corruption");

        assert!(matches!(
            FileBlindVaultReplicaJobStore::open(&directory),
            Err(BlindVaultReplicaJobFileStoreError::CorruptState)
        ));
    }
}
