// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/durable_resolution.rs
// ============================================
//! Atomic durable resolution of one committed private replica attempt.
//!
//! ## Creation Reason
//! Persist-before-send ordering leaves a committed private journal until
//! authenticated terminal evidence is accepted. Updating only in-memory state
//! would lose that evidence transition on restart and either replay ambiguous
//! work or retain the journal forever.
//!
//! ## Main Functionality
//! - Defines an opaque exact binding for one committed attempt journal.
//! - Derives bindings from live durable-send permits and restored journals.
//! - Accepts verified evidence and seals the resulting workflow snapshot.
//! - Resolves the exact journal and snapshot in one recovery-store operation.
//! - Restores the prior in-memory state after sealing or persistence failure.
//!
//! ## Dependencies
//! - `durable_dispatch.rs`: live attempt proven durable before network send.
//! - `attempt_journal.rs`: authenticated continuation restored after restart.
//! - `persistence.rs`: atomic journal-resolution and snapshot store contract.
//! - `execution.rs`: evidence-gated workflow transition semantics.
//! - `snapshot.rs`: identity-bound restart snapshot cryptography.
//!
//! ## Main Logical Flow
//! 1. Obtain a committed binding from a durable send permit or opened journal.
//! 2. Require the current work item to await the exact bound attempt.
//! 3. Apply already verified terminal evidence to the in-memory workflow.
//! 4. Seal its post-evidence snapshot at a new monotonic sequence.
//! 5. Atomically persist that snapshot and remove only the exact journal.
//! 6. Roll memory back if any pre-resolution durability step fails.
//!
//! ## Important Note For The Next Developer
//! - A binding is local durability evidence, not terminal authorization.
//! - Never expose journal commitments, work identity, or targets in telemetry.
//! - Store success is the only point at which evidence resolution is durable.
//! - A failed store call may have an ambiguous host outcome; rollback plus an
//!   exact idempotent retry is required and is supported by the store contract.
//! - Do not split evidence acceptance and store resolution in new callers.
//!
//! Last Modified: v1.0.0-DurableAttemptResolution - Initial exact,
//! rollback-safe evidence resolution command.
//! ============================================

use std::{error::Error, fmt};

use zeroize::{Zeroize, Zeroizing};

use super::{
    persistence::sealed_record_commitment, BlindVaultReplicaActionEvidence,
    BlindVaultReplicaAttemptJournal, BlindVaultReplicaDurableAttemptDispatch,
    BlindVaultReplicaExecution, BlindVaultReplicaPreparedAttemptJournal,
    BlindVaultReplicaRecoveryStore, BlindVaultReplicaSnapshotRecord, BlindVaultReplicaWorkId,
    BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
};
use crate::crypto::keys::IdentityKeyPair;

/// Opaque exact identity of one committed private attempt journal.
pub struct BlindVaultReplicaCommittedAttemptBinding {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    journal_sequence: u64,
    journal_commitment: [u8; 32],
}

impl BlindVaultReplicaCommittedAttemptBinding {
    fn from_prepared(prepared: &BlindVaultReplicaPreparedAttemptJournal) -> Self {
        Self {
            work_id: prepared.work_id(),
            attempt: prepared.attempt(),
            journal_sequence: prepared.journal_sequence(),
            journal_commitment: sealed_record_commitment(prepared.sealed_journal()),
        }
    }

    fn from_opened(journal: &BlindVaultReplicaAttemptJournal) -> Self {
        Self {
            work_id: journal.work_id(),
            attempt: journal.attempt(),
            journal_sequence: journal.journal_sequence(),
            journal_commitment: journal.sealed_commitment(),
        }
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
}

impl Drop for BlindVaultReplicaCommittedAttemptBinding {
    fn drop(&mut self) {
        self.journal_commitment.zeroize();
    }
}

impl fmt::Debug for BlindVaultReplicaCommittedAttemptBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCommittedAttemptBinding")
            .field("attempt", &self.attempt)
            .field("journal_sequence", &self.journal_sequence)
            .field("binding", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Durable result after evidence and journal resolution become restart-safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaDurableResolution {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    snapshot_sequence: u64,
    journal_sequence: u64,
}

impl BlindVaultReplicaDurableResolution {
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }
}

/// Fail-closed errors before one attempt resolution becomes durable.
#[derive(Debug)]
pub enum BlindVaultReplicaDurableResolutionError<StoreError> {
    /// Evidence transition or snapshot sealing violated workflow semantics.
    Workflow(BlindVaultReplicaWorkflowError),
    /// Durable store did not confirm exact atomic resolution.
    Store(StoreError),
    /// Binding and current in-memory attempt did not match exactly.
    StateMismatch,
}

impl<StoreError> fmt::Display for BlindVaultReplicaDurableResolutionError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
            Self::Store(_) => {
                formatter.write_str("blind vault replica durable resolution store failed")
            }
            Self::StateMismatch => {
                formatter.write_str("blind vault replica resolution state does not match")
            }
        }
    }
}

impl<StoreError> Error for BlindVaultReplicaDurableResolutionError<StoreError>
where
    StoreError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Workflow(error) => Some(error),
            Self::Store(error) => Some(error),
            Self::StateMismatch => None,
        }
    }
}

impl BlindVaultReplicaDurableAttemptDispatch<'_, '_> {
    /// Captures the exact journal binding after durable send admission.
    #[must_use]
    pub fn committed_attempt_binding(&self) -> BlindVaultReplicaCommittedAttemptBinding {
        BlindVaultReplicaCommittedAttemptBinding::from_prepared(self.committed.prepared)
    }
}

impl BlindVaultReplicaAttemptJournal {
    /// Captures the exact journal binding after authenticated restart opening.
    #[must_use]
    pub fn committed_attempt_binding(&self) -> BlindVaultReplicaCommittedAttemptBinding {
        BlindVaultReplicaCommittedAttemptBinding::from_opened(self)
    }
}

impl BlindVaultReplicaExecution {
    /// Accepts evidence and atomically resolves its committed private journal.
    ///
    /// [BLIND-VAULT-DURABLE-RESOLUTION 2026-08-29 by Codex] Any failure before
    /// store success restores the exact prior work state. The caller may then
    /// retry the same idempotent resolution or reload durable state.
    pub fn accept_evidence_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        evidence: &BlindVaultReplicaActionEvidence,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let previous_state = self
            .items
            .iter()
            .find(|item| item.id == binding.work_id)
            .map(|item| item.state)
            .ok_or(BlindVaultReplicaDurableResolutionError::StateMismatch)?;
        if !matches!(
            previous_state,
            BlindVaultReplicaWorkState::AwaitingEvidence { attempt, .. }
                if attempt == binding.attempt
        ) {
            return Err(BlindVaultReplicaDurableResolutionError::StateMismatch);
        }

        self.accept_committed_attempt_evidence(binding.work_id, binding.attempt, evidence)
            .map_err(BlindVaultReplicaDurableResolutionError::Workflow)?;
        let sealed_snapshot = match self.seal_restart_snapshot(identity, snapshot_sequence) {
            Ok(snapshot) => Zeroizing::new(snapshot),
            Err(error) => {
                restore_work_state(self, binding.work_id, previous_state)?;
                return Err(BlindVaultReplicaDurableResolutionError::Workflow(error));
            }
        };
        let snapshot = BlindVaultReplicaSnapshotRecord::from_validated_parts(
            self.workflow_id,
            snapshot_sequence,
            sealed_snapshot.as_slice(),
        );
        if let Err(error) = store.resolve_attempt(
            &snapshot,
            binding.journal_sequence,
            binding.journal_commitment,
        ) {
            restore_work_state(self, binding.work_id, previous_state)?;
            return Err(BlindVaultReplicaDurableResolutionError::Store(error));
        }

        Ok(BlindVaultReplicaDurableResolution {
            work_id: binding.work_id,
            attempt: binding.attempt,
            snapshot_sequence,
            journal_sequence: binding.journal_sequence,
        })
    }
}

fn restore_work_state<StoreError>(
    execution: &mut BlindVaultReplicaExecution,
    work_id: BlindVaultReplicaWorkId,
    previous_state: BlindVaultReplicaWorkState,
) -> Result<(), BlindVaultReplicaDurableResolutionError<StoreError>> {
    let item = execution
        .items
        .iter_mut()
        .find(|item| item.id == work_id)
        .ok_or(BlindVaultReplicaDurableResolutionError::StateMismatch)?;
    item.state = previous_state;
    Ok(())
}
