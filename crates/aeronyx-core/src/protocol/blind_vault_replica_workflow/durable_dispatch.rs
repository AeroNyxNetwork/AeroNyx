// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/durable_dispatch.rs
// ============================================
//! Typed durability ordering for one private replica dispatch attempt.
//!
//! ## Creation Reason
//! A correct repository is insufficient if orchestration may still send before
//! the journal and post-dispatch snapshot are durable. Marker types make the
//! safe ordering explicit without coupling protocol state to host I/O.
//!
//! ## Main Functionality
//! - Produces a persisted-journal marker only after the store returns success.
//! - Commits dispatch and seals its exact post-transition restart snapshot.
//! - Restores the in-memory state if snapshot sealing fails.
//! - Produces a durable dispatch permit only after atomic store publication.
//! - Redacts all markers and zeroizes the owned sealed snapshot on drop.
//!
//! ## Dependencies
//! - `attempt_journal.rs`: exact prepared journal and dispatch transition.
//! - `persistence.rs`: storage-neutral records and recovery-store capability.
//! - `snapshot.rs`: identity-sealed post-dispatch workflow snapshot.
//! - `IdentityKeyPair`: source-local snapshot key derivation identity.
//!
//! ## Main Logical Flow
//! 1. Persist the prepared journal and obtain `PersistedAttemptJournal`.
//! 2. Commit the exact workflow attempt without performing network I/O.
//! 3. Seal the new snapshot; roll back memory if local sealing fails.
//! 4. Persist the journal-bound snapshot and obtain `DurableAttemptDispatch`.
//! 5. Permit the adapter to send only while holding that final marker.
//!
//! ## Important Note For The Next Developer
//! - These markers represent local ordering, never network authorization.
//! - Do not add a constructor that bypasses successful store operations.
//! - Do not expose sealed bytes through Debug, telemetry, or error strings.
//! - Keep the low-level legacy methods for compatibility, but new adapters
//!   should use this path before every private mutating network attempt.
//!
//! Last Modified: v1.1.0-DurableResolutionBinding - Shared the exact prepared
//! journal with the sibling resolution domain without widening public access.
//! v1.0.0-DurableDispatchPermit - Initial fail-closed typed
//! journal, snapshot, and network-send durability sequence.
//! ============================================

use std::fmt;

use thiserror::Error;
use zeroize::Zeroize;

use super::{
    BlindVaultReplicaAttemptJournalError, BlindVaultReplicaCommittedAttemptRecord,
    BlindVaultReplicaExecution, BlindVaultReplicaPreparedAttemptJournal,
    BlindVaultReplicaPreparedAttemptRecord, BlindVaultReplicaRecoveryStore,
    BlindVaultReplicaSnapshotRecord, BlindVaultReplicaWorkId, BlindVaultReplicaWorkState,
    BlindVaultReplicaWorkflowError,
};
use crate::crypto::keys::IdentityKeyPair;

/// Prepared private journal proven durable by a recovery-store success.
pub struct BlindVaultReplicaPersistedAttemptJournal<'a> {
    prepared: &'a BlindVaultReplicaPreparedAttemptJournal,
}

/// Post-dispatch in-memory state plus its identity-sealed restart snapshot.
pub struct BlindVaultReplicaCommittedAttemptDispatch<'a> {
    // [BLIND-VAULT-DURABLE-RESOLUTION 2026-08-29 by Codex] Sibling-only
    // visibility lets resolution derive an exact binding after store success.
    pub(super) prepared: &'a BlindVaultReplicaPreparedAttemptJournal,
    workflow_id: [u8; 16],
    snapshot_sequence: u64,
    sealed_snapshot: Vec<u8>,
}

/// Marker permitting the exact already-prepared network request to be sent.
pub struct BlindVaultReplicaDurableAttemptDispatch<'a, 'b> {
    // [BLIND-VAULT-DURABLE-RESOLUTION 2026-08-29 by Codex] This remains
    // inaccessible outside the workflow module; callers receive a typed copy.
    pub(super) committed: &'b BlindVaultReplicaCommittedAttemptDispatch<'a>,
}

/// Local commit failures before any durable-send permit can exist.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaDurableDispatchError {
    #[error(transparent)]
    Attempt(#[from] BlindVaultReplicaAttemptJournalError),
    #[error("blind vault replica durable dispatch snapshot failed")]
    Snapshot(#[source] BlindVaultReplicaWorkflowError),
    #[error("blind vault replica durable dispatch state does not match")]
    StateMismatch,
}

impl BlindVaultReplicaPreparedAttemptJournal {
    /// Persists this exact prepared journal before dispatch may be committed.
    pub fn persist_for_dispatch<'a, Store>(
        &'a self,
        store: &mut Store,
    ) -> Result<BlindVaultReplicaPersistedAttemptJournal<'a>, Store::Error>
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let record = BlindVaultReplicaPreparedAttemptRecord::from(self);
        store.persist_prepared_attempt(&record)?;
        Ok(BlindVaultReplicaPersistedAttemptJournal { prepared: self })
    }
}

impl BlindVaultReplicaExecution {
    /// Commits a durably journaled attempt and seals its post-dispatch state.
    ///
    /// [BLIND-VAULT-DURABLE-DISPATCH 2026-08-29 by Codex] The transition is
    /// reversible until sealing succeeds. No store or network effect occurs
    /// inside this method, and a sealing error restores the prior work state.
    pub fn commit_persisted_attempt_dispatch<'a>(
        &mut self,
        identity: &IdentityKeyPair,
        persisted: BlindVaultReplicaPersistedAttemptJournal<'a>,
        snapshot_sequence: u64,
    ) -> Result<BlindVaultReplicaCommittedAttemptDispatch<'a>, BlindVaultReplicaDurableDispatchError>
    {
        let work_id = persisted.prepared.work_id();
        let previous_state = self
            .items
            .iter()
            .find(|item| item.id == work_id)
            .map(|item| item.state)
            .ok_or(BlindVaultReplicaDurableDispatchError::StateMismatch)?;
        self.commit_prepared_attempt_dispatch(persisted.prepared)?;

        let sealed_snapshot = match self.seal_restart_snapshot(identity, snapshot_sequence) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                restore_work_state(self, work_id, previous_state)?;
                return Err(BlindVaultReplicaDurableDispatchError::Snapshot(error));
            }
        };
        Ok(BlindVaultReplicaCommittedAttemptDispatch {
            prepared: persisted.prepared,
            workflow_id: self.workflow_id,
            snapshot_sequence,
            sealed_snapshot,
        })
    }
}

impl<'a> BlindVaultReplicaCommittedAttemptDispatch<'a> {
    /// Exact work item now waiting for terminal evidence.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.prepared.work_id()
    }

    /// Exact bounded attempt represented by the durable records.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.prepared.attempt()
    }

    /// Monotonic snapshot sequence installed after dispatch commit.
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    /// Persists the exact journal-bound snapshot before network send.
    pub fn persist_for_network_send<'b, Store>(
        &'b self,
        store: &mut Store,
    ) -> Result<BlindVaultReplicaDurableAttemptDispatch<'a, 'b>, Store::Error>
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let snapshot = BlindVaultReplicaSnapshotRecord::from_validated_parts(
            self.workflow_id,
            self.snapshot_sequence,
            &self.sealed_snapshot,
        );
        let committed =
            BlindVaultReplicaCommittedAttemptRecord::from_validated_parts(snapshot, self.prepared);
        store.persist_committed_attempt(&committed)?;
        Ok(BlindVaultReplicaDurableAttemptDispatch { committed: self })
    }
}

impl Drop for BlindVaultReplicaCommittedAttemptDispatch<'_> {
    fn drop(&mut self) {
        self.sealed_snapshot.zeroize();
    }
}

impl BlindVaultReplicaDurableAttemptDispatch<'_, '_> {
    /// Exact work item for the request now permitted to leave the source.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.committed.work_id()
    }

    /// Exact attempt for evidence correlation after sending.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.committed.attempt()
    }

    /// Snapshot sequence proving the post-dispatch state is durable.
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.committed.snapshot_sequence()
    }

    /// Journal sequence proving exact private continuation state is durable.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.committed.prepared.journal_sequence()
    }
}

impl fmt::Debug for BlindVaultReplicaPersistedAttemptJournal<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPersistedAttemptJournal")
            .field("attempt", &self.prepared.attempt())
            .field("journal_sequence", &self.prepared.journal_sequence())
            .field("sealed_journal", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaCommittedAttemptDispatch<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCommittedAttemptDispatch")
            .field("attempt", &self.prepared.attempt())
            .field("snapshot_sequence", &self.snapshot_sequence)
            .field("sealed_snapshot", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaDurableAttemptDispatch<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaDurableAttemptDispatch")
            .field("attempt", &self.attempt())
            .field("snapshot_sequence", &self.snapshot_sequence())
            .field("journal_sequence", &self.journal_sequence())
            .finish_non_exhaustive()
    }
}

fn restore_work_state(
    execution: &mut BlindVaultReplicaExecution,
    work_id: BlindVaultReplicaWorkId,
    previous_state: BlindVaultReplicaWorkState,
) -> Result<(), BlindVaultReplicaDurableDispatchError> {
    let item = execution
        .items
        .iter_mut()
        .find(|item| item.id == work_id)
        .ok_or(BlindVaultReplicaDurableDispatchError::StateMismatch)?;
    item.state = previous_state;
    Ok(())
}
