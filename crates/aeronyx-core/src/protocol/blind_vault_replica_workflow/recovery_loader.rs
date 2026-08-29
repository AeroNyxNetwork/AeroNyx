// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/recovery_loader.rs
// ============================================
//! Authenticated loading of one durable replica workflow generation.
//!
//! ## Creation Reason
//! A storage adapter can prove that bytes form one atomic generation, but it
//! cannot decide whether the sealed snapshot, journal phase, and workflow
//! action form a safe runtime continuation. That decision belongs to the core
//! domain beside the snapshot and attempt-journal cryptography.
//!
//! ## Main Functionality
//! - Loads one storage-neutral recovery generation through the store trait.
//! - Opens the identity-sealed snapshot at the exact accepted high-water mark.
//! - Authenticates prepared journals without exposing continuation plaintext.
//! - Opens committed journals only for the exact ambiguous in-flight attempt.
//! - Rejects unsupported concurrent private attempts and every phase mismatch.
//!
//! ## Dependencies
//! - `persistence.rs`: atomic generation phases and recovery-store contract.
//! - `snapshot.rs`: identity-bound workflow restoration and rollback checks.
//! - `attempt_journal.rs`: prepared authentication and committed continuation.
//! - `recovery.rs`: action-specific private-journal recovery requirements.
//! - `IdentityKeyPair`: source identity that owns the sealed local generation.
//!
//! ## Main Logical Flow
//! 1. Load one internally consistent sealed generation from the host adapter.
//! 2. Open its snapshot and require exact equality with durable high-water.
//! 3. Match the durable phase to the only legal journal interpretation.
//! 4. Prove there is no unjournaled private in-flight attempt.
//! 5. Return a typed local recovery result without performing network I/O.
//!
//! ## Important Note For The Next Developer
//! - This loader is source-owned; never expose its values through node APIs.
//! - `Prepared` proves no dispatch commit and permits exact journal cleanup.
//! - `Committed` is ambiguous and must remain until terminal evidence resolves.
//! - Never relax exact sequence equality to a lower-bound comparison here.
//! - A single durable generation supports at most one private attempt journal.
//! - Loading, classification, and journal opening must remain side-effect free.
//!
//! Last Modified: v1.0.0-AuthenticatedRecoveryLoad - Initial phase-aware,
//! exact-high-water workflow recovery loader.
//! ============================================

use std::{error::Error, fmt};

use super::{
    BlindVaultReplicaAttemptDurabilityPhase, BlindVaultReplicaAttemptJournal,
    BlindVaultReplicaAttemptJournalError, BlindVaultReplicaAuthenticatedPreparedAttempt,
    BlindVaultReplicaExecution, BlindVaultReplicaRecoveryStore,
    BlindVaultReplicaRestartRecoveryTask, BlindVaultReplicaRestoredExecution,
    BlindVaultReplicaWorkflowError,
};
use crate::crypto::keys::IdentityKeyPair;

/// Authenticated runtime meaning of one durable recovery generation.
pub enum BlindVaultReplicaLoadedRecovery {
    /// No unresolved private attempt journal belongs to this snapshot.
    Resolved {
        restored: BlindVaultReplicaRestoredExecution,
    },
    /// A journal was durable, but dispatch was never committed.
    Prepared {
        restored: BlindVaultReplicaRestoredExecution,
        authenticated_attempt: BlindVaultReplicaAuthenticatedPreparedAttempt,
    },
    /// Dispatch was committed and the exact private continuation is available.
    Committed {
        restored: BlindVaultReplicaRestoredExecution,
        attempt_journal: BlindVaultReplicaAttemptJournal,
    },
}

impl BlindVaultReplicaLoadedRecovery {
    /// Borrows the authenticated workflow restored from the durable snapshot.
    #[must_use]
    pub const fn restored(&self) -> &BlindVaultReplicaRestoredExecution {
        match self {
            Self::Resolved { restored }
            | Self::Prepared { restored, .. }
            | Self::Committed { restored, .. } => restored,
        }
    }

    /// Durable phase whose complete invariants were authenticated.
    #[must_use]
    pub const fn phase(&self) -> BlindVaultReplicaAttemptDurabilityPhase {
        match self {
            Self::Resolved { .. } => BlindVaultReplicaAttemptDurabilityPhase::Resolved,
            Self::Prepared { .. } => BlindVaultReplicaAttemptDurabilityPhase::Prepared,
            Self::Committed { .. } => BlindVaultReplicaAttemptDurabilityPhase::Committed,
        }
    }

    /// Transfers prepared workflow state and its exact cleanup authority.
    #[must_use]
    pub fn into_prepared_recovery(
        self,
    ) -> Option<(
        BlindVaultReplicaRestoredExecution,
        BlindVaultReplicaAuthenticatedPreparedAttempt,
    )> {
        match self {
            Self::Prepared {
                restored,
                authenticated_attempt,
            } => Some((restored, authenticated_attempt)),
            Self::Resolved { .. } | Self::Committed { .. } => None,
        }
    }

    /// Transfers committed private continuation state to its runtime owner.
    #[must_use]
    pub fn into_committed_attempt(
        self,
    ) -> Option<(
        BlindVaultReplicaRestoredExecution,
        BlindVaultReplicaAttemptJournal,
    )> {
        match self {
            Self::Committed {
                restored,
                attempt_journal,
            } => Some((restored, attempt_journal)),
            Self::Resolved { .. } | Self::Prepared { .. } => None,
        }
    }
}

impl fmt::Debug for BlindVaultReplicaLoadedRecovery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaLoadedRecovery")
            .field("phase", &self.phase())
            .field("snapshot_sequence", &self.restored().snapshot_sequence())
            .field("private_recovery", &"<redacted>")
            .finish_non_exhaustive()
    }
}

/// Fail-closed boundary errors while interpreting durable recovery state.
#[derive(Debug)]
pub enum BlindVaultReplicaRecoveryLoadError<StoreError> {
    /// Host adapter could not load a complete durable generation.
    Store(StoreError),
    /// The sealed snapshot failed authentication or restored-state validation.
    Workflow(BlindVaultReplicaWorkflowError),
    /// The sealed private attempt journal failed authentication or binding.
    AttemptJournal(BlindVaultReplicaAttemptJournalError),
    /// Durable phase, high-water, or private-attempt cardinality conflicted.
    StateMismatch,
}

impl<StoreError> fmt::Display for BlindVaultReplicaRecoveryLoadError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(_) => formatter.write_str("blind vault replica recovery store failed"),
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
            Self::AttemptJournal(error) => fmt::Display::fmt(error, formatter),
            Self::StateMismatch => {
                formatter.write_str("blind vault replica recovery state does not match")
            }
        }
    }
}

impl<StoreError> Error for BlindVaultReplicaRecoveryLoadError<StoreError>
where
    StoreError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Workflow(error) => Some(error),
            Self::AttemptJournal(error) => Some(error),
            Self::StateMismatch => None,
        }
    }
}

/// Loads and authenticates one source-owned durable workflow generation.
///
/// [BLIND-VAULT-RECOVERY-LOADER 2026-08-29 by Codex] The adapter's accepted
/// sequences are treated as exact high-water values, not merely lower bounds.
/// This prevents a correctly encrypted but internally reordered generation
/// from entering runtime ownership.
pub fn load_blind_vault_replica_recovery<Store>(
    identity: &IdentityKeyPair,
    store: &mut Store,
    now_ms: u64,
) -> Result<Option<BlindVaultReplicaLoadedRecovery>, BlindVaultReplicaRecoveryLoadError<Store::Error>>
where
    Store: BlindVaultReplicaRecoveryStore,
{
    let Some(state) = store
        .load_recovery_state()
        .map_err(BlindVaultReplicaRecoveryLoadError::Store)?
    else {
        return Ok(None);
    };
    let phase = state.phase();
    let accepted_snapshot_sequence = state.accepted_snapshot_sequence();
    let accepted_journal_sequence = state.accepted_journal_sequence();
    let restored = BlindVaultReplicaExecution::open_restart_snapshot(
        identity,
        state.sealed_snapshot(),
        accepted_snapshot_sequence,
        now_ms,
    )
    .map_err(BlindVaultReplicaRecoveryLoadError::Workflow)?;
    if restored.snapshot_sequence() != accepted_snapshot_sequence {
        return Err(BlindVaultReplicaRecoveryLoadError::StateMismatch);
    }

    match phase {
        BlindVaultReplicaAttemptDurabilityPhase::Resolved => {
            require_private_recovery_count(&restored, now_ms, 0, None)?;
            Ok(Some(BlindVaultReplicaLoadedRecovery::Resolved { restored }))
        }
        BlindVaultReplicaAttemptDurabilityPhase::Prepared => {
            let sealed_journal = state
                .sealed_attempt_journal()
                .ok_or(BlindVaultReplicaRecoveryLoadError::StateMismatch)?;
            let authenticated_attempt = restored
                .authenticate_prepared_attempt_journal(
                    identity,
                    sealed_journal,
                    accepted_journal_sequence,
                    now_ms,
                )
                .map_err(BlindVaultReplicaRecoveryLoadError::AttemptJournal)?;
            if authenticated_attempt.journal_sequence() != accepted_journal_sequence {
                return Err(BlindVaultReplicaRecoveryLoadError::StateMismatch);
            }
            require_private_recovery_count(&restored, now_ms, 0, None)?;
            Ok(Some(BlindVaultReplicaLoadedRecovery::Prepared {
                restored,
                authenticated_attempt,
            }))
        }
        BlindVaultReplicaAttemptDurabilityPhase::Committed => {
            let sealed_journal = state
                .sealed_attempt_journal()
                .ok_or(BlindVaultReplicaRecoveryLoadError::StateMismatch)?;
            let attempt_journal = restored
                .open_attempt_journal(identity, sealed_journal, accepted_journal_sequence, now_ms)
                .map_err(BlindVaultReplicaRecoveryLoadError::AttemptJournal)?;
            if attempt_journal.journal_sequence() != accepted_journal_sequence {
                return Err(BlindVaultReplicaRecoveryLoadError::StateMismatch);
            }
            require_private_recovery_count(
                &restored,
                now_ms,
                1,
                Some((attempt_journal.work_id(), attempt_journal.attempt())),
            )?;
            Ok(Some(BlindVaultReplicaLoadedRecovery::Committed {
                restored,
                attempt_journal,
            }))
        }
    }
}

fn require_private_recovery_count<StoreError>(
    restored: &BlindVaultReplicaRestoredExecution,
    now_ms: u64,
    expected_count: usize,
    expected_attempt: Option<(super::BlindVaultReplicaWorkId, u8)>,
) -> Result<(), BlindVaultReplicaRecoveryLoadError<StoreError>> {
    let tasks = restored
        .restart_recovery_tasks(now_ms)
        .map_err(BlindVaultReplicaRecoveryLoadError::Workflow)?;
    let mut private_count = 0_usize;
    let mut expected_attempt_matched = expected_attempt.is_none();
    for task in tasks
        .iter()
        .filter(|task| task.kind().requires_private_attempt_journal())
    {
        private_count = private_count.saturating_add(1);
        if let Some((work_id, attempt)) = expected_attempt {
            expected_attempt_matched |= matches_private_attempt(task, work_id, attempt);
        }
    }
    if private_count != expected_count || !expected_attempt_matched {
        return Err(BlindVaultReplicaRecoveryLoadError::StateMismatch);
    }
    Ok(())
}

fn matches_private_attempt(
    task: &BlindVaultReplicaRestartRecoveryTask,
    work_id: super::BlindVaultReplicaWorkId,
    attempt: u8,
) -> bool {
    task.work_id() == work_id && task.attempt() == attempt
}
