// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/persistence.rs
// ============================================
//! Storage-neutral durability contract for source-owned replica recovery.
//!
//! ## Creation Reason
//! Replica replacement and provisioning cross a local persistence boundary
//! before any network side effect may begin. Binding that boundary directly to
//! one filesystem, database, or mobile keystore would mix protocol state with
//! platform policy and make crash ordering difficult to audit.
//!
//! ## Main Functionality
//! - Defines explicit prepared, committed, and resolved durability phases.
//! - Provides bounded, redacted records for sealed snapshots and journals.
//! - Defines a replaceable recovery-store capability with atomic semantics.
//! - Carries independently protected sequence high-water marks on restore.
//! - Zeroizes owned sealed containers when restored state leaves scope.
//!
//! ## Dependencies
//! - `attempt_journal.rs`: prepared private attempt journal and fixed bounds.
//! - `snapshot.rs`: identity-sealed workflow snapshot and fixed bounds.
//! - `BlindVaultReplicaWorkId`: exact immutable workflow/work identity.
//!
//! ## Main Logical Flow
//! 1. Persist a prepared journal and journal high-water mark atomically.
//! 2. Commit the matching in-memory dispatch transition without network I/O.
//! 3. Persist the resulting snapshot and snapshot high-water mark atomically.
//! 4. Mark the exact journal committed in the same durable generation.
//! 5. Only then send the already prepared request; resolve after evidence.
//!
//! ## Important Note For The Next Developer
//! - Implementations must never return success before durable synchronization.
//! - Exact idempotent retries must re-confirm underlying durability; observing
//!   matching bytes alone is insufficient after a prior synchronization error.
//! - A filesystem adapter must fsync file and parent directory after rename.
//! - A database adapter must use one durable transaction for every method.
//! - High-water state needs stronger rollback protection than mutable records.
//! - Never log sealed bytes, workflow identity, target identity, or work ids.
//! - Partial, corrupt, reordered, or unsupported state must fail closed.
//!
//! Last Modified: v1.2.0-IdempotentDurabilityConfirmation - Required exact
//! retries to re-confirm host durability before returning success.
//! v1.1.0-PreparedAbort - Added exact safe cleanup for journals
//! proven to have stopped before the post-dispatch snapshot commit.
//! v1.0.0-RecoveryStoreContract - Initial storage-neutral
//! atomic persistence contract for restart-safe replica execution.
//! ============================================

use std::fmt;

use sha2::{Digest, Sha256};
use zeroize::Zeroize;

use super::{
    BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaWorkId,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES, MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES,
};

const SEALED_RECORD_COMMITMENT_DOMAIN: &[u8] = b"AeroNyx-BlindVault-Recovery-Sealed-Record-v1";

/// Durable relation between the current snapshot and private attempt journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaAttemptDurabilityPhase {
    /// No unresolved private mutating attempt belongs to the current snapshot.
    Resolved,
    /// The journal is durable but dispatch is not reflected by the snapshot.
    Prepared,
    /// The snapshot reflects dispatch and retains the exact matching journal.
    Committed,
}

/// Borrowed sealed workflow snapshot presented to a durability adapter.
pub struct BlindVaultReplicaSnapshotRecord<'a> {
    workflow_id: [u8; 16],
    snapshot_sequence: u64,
    sealed_snapshot: &'a [u8],
}

impl<'a> BlindVaultReplicaSnapshotRecord<'a> {
    /// Creates one bounded snapshot persistence record.
    ///
    /// [BLIND-VAULT-RECOVERY-STORE 2026-08-29 by Codex] Zero sequences,
    /// empty containers, and oversized containers never reach an adapter.
    pub fn new(
        workflow_id: [u8; 16],
        snapshot_sequence: u64,
        sealed_snapshot: &'a [u8],
    ) -> Option<Self> {
        if workflow_id == [0; 16]
            || snapshot_sequence == 0
            || sealed_snapshot.is_empty()
            || sealed_snapshot.len() > MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES
        {
            return None;
        }
        Some(Self {
            workflow_id,
            snapshot_sequence,
            sealed_snapshot,
        })
    }

    /// Builds a record from values already validated by the typed commit path.
    pub(super) const fn from_validated_parts(
        workflow_id: [u8; 16],
        snapshot_sequence: u64,
        sealed_snapshot: &'a [u8],
    ) -> Self {
        Self {
            workflow_id,
            snapshot_sequence,
            sealed_snapshot,
        }
    }

    #[must_use]
    pub const fn workflow_id(&self) -> [u8; 16] {
        self.workflow_id
    }

    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    #[must_use]
    pub const fn sealed_snapshot(&self) -> &'a [u8] {
        self.sealed_snapshot
    }

    #[must_use]
    pub fn sealed_commitment(&self) -> [u8; 32] {
        sealed_record_commitment(self.sealed_snapshot)
    }
}

impl fmt::Debug for BlindVaultReplicaSnapshotRecord<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaSnapshotRecord")
            .field("snapshot_sequence", &self.snapshot_sequence)
            .field("sealed_snapshot", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Borrowed exact private journal persisted before dispatch may be committed.
pub struct BlindVaultReplicaPreparedAttemptRecord<'a> {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    journal_sequence: u64,
    retain_until_ms: u64,
    sealed_journal: &'a [u8],
}

impl<'a> From<&'a BlindVaultReplicaPreparedAttemptJournal>
    for BlindVaultReplicaPreparedAttemptRecord<'a>
{
    fn from(prepared: &'a BlindVaultReplicaPreparedAttemptJournal) -> Self {
        Self {
            work_id: prepared.work_id(),
            attempt: prepared.attempt(),
            journal_sequence: prepared.journal_sequence(),
            retain_until_ms: prepared.retain_until_ms(),
            sealed_journal: prepared.sealed_journal(),
        }
    }
}

impl BlindVaultReplicaPreparedAttemptRecord<'_> {
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }

    #[must_use]
    pub const fn retain_until_ms(&self) -> u64 {
        self.retain_until_ms
    }

    #[must_use]
    pub const fn sealed_journal(&self) -> &[u8] {
        self.sealed_journal
    }

    #[must_use]
    pub fn sealed_commitment(&self) -> [u8; 32] {
        sealed_record_commitment(self.sealed_journal)
    }
}

impl fmt::Debug for BlindVaultReplicaPreparedAttemptRecord<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPreparedAttemptRecord")
            .field("journal_sequence", &self.journal_sequence)
            .field("attempt", &self.attempt)
            .field("sealed_journal", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Borrowed committed snapshot bound to the already durable private journal.
pub struct BlindVaultReplicaCommittedAttemptRecord<'a> {
    snapshot: BlindVaultReplicaSnapshotRecord<'a>,
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    journal_sequence: u64,
    journal_commitment: [u8; 32],
}

impl<'a> BlindVaultReplicaCommittedAttemptRecord<'a> {
    /// Binds the post-dispatch snapshot to the exact prepared journal.
    pub fn new(
        snapshot: BlindVaultReplicaSnapshotRecord<'a>,
        prepared: &BlindVaultReplicaPreparedAttemptJournal,
    ) -> Option<Self> {
        if snapshot.workflow_id() != prepared.work_id().workflow_id() {
            return None;
        }
        Some(Self {
            snapshot,
            work_id: prepared.work_id(),
            attempt: prepared.attempt(),
            journal_sequence: prepared.journal_sequence(),
            journal_commitment: sealed_record_commitment(prepared.sealed_journal()),
        })
    }

    /// Binds values already validated by the typed durable-dispatch path.
    pub(super) fn from_validated_parts(
        snapshot: BlindVaultReplicaSnapshotRecord<'a>,
        prepared: &BlindVaultReplicaPreparedAttemptJournal,
    ) -> Self {
        Self {
            snapshot,
            work_id: prepared.work_id(),
            attempt: prepared.attempt(),
            journal_sequence: prepared.journal_sequence(),
            journal_commitment: sealed_record_commitment(prepared.sealed_journal()),
        }
    }

    #[must_use]
    pub const fn snapshot(&self) -> &BlindVaultReplicaSnapshotRecord<'a> {
        &self.snapshot
    }

    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }

    #[must_use]
    pub const fn journal_commitment(&self) -> [u8; 32] {
        self.journal_commitment
    }
}

impl fmt::Debug for BlindVaultReplicaCommittedAttemptRecord<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCommittedAttemptRecord")
            .field("snapshot_sequence", &self.snapshot.snapshot_sequence)
            .field("journal_sequence", &self.journal_sequence)
            .field("attempt", &self.attempt)
            .field("sealed_records", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Validated sealed recovery generation loaded from durable storage.
pub struct BlindVaultReplicaRecoveryState {
    phase: BlindVaultReplicaAttemptDurabilityPhase,
    accepted_snapshot_sequence: u64,
    accepted_journal_sequence: u64,
    sealed_snapshot: Vec<u8>,
    sealed_attempt_journal: Option<Vec<u8>>,
}

impl BlindVaultReplicaRecoveryState {
    /// Validates one loaded generation before cryptographic opening begins.
    pub fn new(
        phase: BlindVaultReplicaAttemptDurabilityPhase,
        accepted_snapshot_sequence: u64,
        accepted_journal_sequence: u64,
        sealed_snapshot: Vec<u8>,
        sealed_attempt_journal: Option<Vec<u8>>,
    ) -> Option<Self> {
        if accepted_snapshot_sequence == 0
            || sealed_snapshot.is_empty()
            || sealed_snapshot.len() > MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES
        {
            return None;
        }
        let journal_shape_is_valid = match (&phase, &sealed_attempt_journal) {
            (BlindVaultReplicaAttemptDurabilityPhase::Resolved, None) => true,
            (
                BlindVaultReplicaAttemptDurabilityPhase::Prepared
                | BlindVaultReplicaAttemptDurabilityPhase::Committed,
                Some(journal),
            ) => {
                accepted_journal_sequence > 0
                    && !journal.is_empty()
                    && journal.len() <= MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES
            }
            _ => false,
        };
        if !journal_shape_is_valid {
            return None;
        }
        Some(Self {
            phase,
            accepted_snapshot_sequence,
            accepted_journal_sequence,
            sealed_snapshot,
            sealed_attempt_journal,
        })
    }

    #[must_use]
    pub const fn phase(&self) -> BlindVaultReplicaAttemptDurabilityPhase {
        self.phase
    }

    #[must_use]
    pub const fn accepted_snapshot_sequence(&self) -> u64 {
        self.accepted_snapshot_sequence
    }

    #[must_use]
    pub const fn accepted_journal_sequence(&self) -> u64 {
        self.accepted_journal_sequence
    }

    #[must_use]
    pub fn sealed_snapshot(&self) -> &[u8] {
        &self.sealed_snapshot
    }

    #[must_use]
    pub fn sealed_attempt_journal(&self) -> Option<&[u8]> {
        self.sealed_attempt_journal.as_deref()
    }

    /// Commitment used to bind an authenticated prepared-journal abort.
    #[must_use]
    pub fn sealed_attempt_journal_commitment(&self) -> Option<[u8; 32]> {
        self.sealed_attempt_journal
            .as_deref()
            .map(sealed_record_commitment)
    }

    /// Transfers sealed bytes to the recovery owner without cloning.
    #[must_use]
    pub fn into_sealed_records(mut self) -> (Vec<u8>, Option<Vec<u8>>) {
        (
            std::mem::take(&mut self.sealed_snapshot),
            self.sealed_attempt_journal.take(),
        )
    }
}

impl Drop for BlindVaultReplicaRecoveryState {
    fn drop(&mut self) {
        self.sealed_snapshot.zeroize();
        if let Some(journal) = self.sealed_attempt_journal.as_mut() {
            journal.zeroize();
        }
    }
}

impl fmt::Debug for BlindVaultReplicaRecoveryState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRecoveryState")
            .field("phase", &self.phase)
            .field(
                "accepted_snapshot_sequence",
                &self.accepted_snapshot_sequence,
            )
            .field("accepted_journal_sequence", &self.accepted_journal_sequence)
            .field("sealed_records", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Replaceable durable store for one source-owned workflow generation.
///
/// Every mutating method is an atomic durability boundary. Returning `Ok(())`
/// means content and metadata survive process and power loss according to the
/// adapter's documented storage guarantees. Implementations must reject
/// sequence rollback, conflicting journal commitments, partial generations,
/// symlinks, unsafe permissions, and unsupported formats.
pub trait BlindVaultReplicaRecoveryStore {
    type Error;

    /// Installs a normal snapshot only when no unresolved journal exists.
    ///
    /// Implementations must reject this method in `Prepared` or `Committed`
    /// phase; those transitions require the exact journal-bound methods below.
    fn persist_snapshot(
        &mut self,
        snapshot: &BlindVaultReplicaSnapshotRecord<'_>,
    ) -> Result<(), Self::Error>;

    /// Installs a prepared journal and advances journal high-water atomically.
    fn persist_prepared_attempt(
        &mut self,
        prepared: &BlindVaultReplicaPreparedAttemptRecord<'_>,
    ) -> Result<(), Self::Error>;

    /// Installs the post-dispatch snapshot and marks the exact journal committed.
    fn persist_committed_attempt(
        &mut self,
        committed: &BlindVaultReplicaCommittedAttemptRecord<'_>,
    ) -> Result<(), Self::Error>;

    /// Removes an exact authenticated journal that never reached commit.
    ///
    /// This method is valid only in `Prepared`. Implementations must preserve
    /// the accepted journal sequence as a rollback high-water mark and reject
    /// `Committed`: an ambiguous network effect must remain recoverable.
    fn abort_prepared_attempt(
        &mut self,
        journal_sequence: u64,
        journal_commitment: [u8; 32],
    ) -> Result<(), Self::Error>;

    /// Removes the exact resolved journal while atomically installing snapshot.
    fn resolve_attempt(
        &mut self,
        snapshot: &BlindVaultReplicaSnapshotRecord<'_>,
        journal_sequence: u64,
        journal_commitment: [u8; 32],
    ) -> Result<(), Self::Error>;

    /// Loads one internally consistent generation, or no initialized state.
    fn load_recovery_state(
        &mut self,
    ) -> Result<Option<BlindVaultReplicaRecoveryState>, Self::Error>;
}

pub(super) fn sealed_record_commitment(sealed: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SEALED_RECORD_COMMITMENT_DOMAIN);
    hasher.update((sealed.len() as u64).to_be_bytes());
    hasher.update(sealed);
    hasher.finalize().into()
}
