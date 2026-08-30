// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/durable_snapshot.rs
// ============================================
//! Durable snapshot command for workflow states without private journals.
//!
//! ## Creation Reason
//! Initial workflow bootstrap and ordinary read-only/failure transitions need
//! restart-safe persistence before later dispatch work can rely on a current
//! snapshot. Requiring every adapter to manually compose sealing, record
//! construction, zeroization, and store publication invites ordering gaps.
//!
//! ## Main Functionality
//! - Seals one source-owned workflow at an explicit monotonic sequence.
//! - Publishes the bounded snapshot through the recovery-store capability.
//! - Confirms ambiguous publication with the exact same sealed generation.
//! - Returns a typed receipt only after the store confirms durable success.
//! - Zeroizes the temporary sealed container on every return path.
//! - Preserves store enforcement that unresolved private journals block writes.
//!
//! ## Dependencies
//! - `snapshot.rs`: identity-bound restart snapshot cryptography.
//! - `persistence.rs`: resolved-phase snapshot record and store contract.
//! - `IdentityKeyPair`: source-local identity that owns recovery state.
//!
//! ## Main Logical Flow
//! 1. Validate and seal current workflow state at the requested sequence.
//! 2. Wrap the sealed bytes in a bounded storage-neutral snapshot record.
//! 3. Ask the store to atomically install the resolved generation.
//! 4. Return the accepted sequence only after durable publication succeeds.
//! 5. Clear the temporary sealed container regardless of the result.
//!
//! ## Important Note For The Next Developer
//! - This command cannot resolve `Prepared` or `Committed` journals.
//! - Use durable resolution for evidence-bound private attempt completion.
//! - A double store failure leaves the result unknown; discard live state and
//!   use authenticated recovery before choosing another snapshot sequence.
//! - Never log source workflow ids, sealed bytes, paths, or store error detail.
//!
//! Last Modified: v1.1.0-AmbiguousSnapshotConfirmation - Replayed the exact
//! sealed generation once before reporting an unknown publication outcome.
//! v1.0.0-DurableWorkflowSnapshot - Initial reusable command
//! for restart-safe resolved workflow state.
//! ============================================

use std::{error::Error, fmt};

use zeroize::Zeroizing;

use super::{
    BlindVaultReplicaExecution, BlindVaultReplicaRecoveryStore, BlindVaultReplicaSnapshotRecord,
    BlindVaultReplicaWorkflowError,
};
use crate::crypto::keys::IdentityKeyPair;

/// Receipt proving the store accepted one exact workflow snapshot sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaDurableSnapshot {
    snapshot_sequence: u64,
}

impl BlindVaultReplicaDurableSnapshot {
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }
}

/// Fail-closed errors before a workflow snapshot becomes durable.
pub enum BlindVaultReplicaDurableSnapshotError<StoreError> {
    /// Snapshot sealing rejected invalid or oversized workflow state.
    Workflow(BlindVaultReplicaWorkflowError),
    /// Recovery store did not confirm durable publication.
    Store(StoreError),
    /// Both publication and exact idempotent confirmation failed.
    StoreOutcomeUnknown {
        publication: StoreError,
        confirmation: StoreError,
    },
}

impl<StoreError> fmt::Display for BlindVaultReplicaDurableSnapshotError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
            Self::Store(_) => {
                formatter.write_str("blind vault replica durable snapshot store failed")
            }
            Self::StoreOutcomeUnknown { .. } => {
                formatter.write_str("blind vault replica durable snapshot outcome is unknown")
            }
        }
    }
}

impl<StoreError> Error for BlindVaultReplicaDurableSnapshotError<StoreError>
where
    StoreError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Workflow(error) => Some(error),
            Self::Store(error) => Some(error),
            Self::StoreOutcomeUnknown { publication, .. } => Some(publication),
        }
    }
}

impl<StoreError> fmt::Debug for BlindVaultReplicaDurableSnapshotError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(_) => formatter.write_str("Workflow(<redacted>)"),
            Self::Store(_) => formatter.write_str("Store(<redacted>)"),
            Self::StoreOutcomeUnknown { .. } => {
                formatter.write_str("StoreOutcomeUnknown(<redacted>)")
            }
        }
    }
}

impl BlindVaultReplicaExecution {
    /// Seals and durably publishes current resolved workflow state.
    ///
    /// [BLIND-VAULT-DURABLE-SNAPSHOT 2026-08-29 by Codex] The receipt is
    /// created only after store success. Existing `Prepared` or `Committed`
    /// state remains protected because the store rejects this command there.
    pub fn persist_restart_snapshot_durably<Store>(
        &self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        snapshot_sequence: u64,
    ) -> Result<BlindVaultReplicaDurableSnapshot, BlindVaultReplicaDurableSnapshotError<Store::Error>>
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let sealed_snapshot = Zeroizing::new(
            self.seal_restart_snapshot(identity, snapshot_sequence)
                .map_err(BlindVaultReplicaDurableSnapshotError::Workflow)?,
        );
        let snapshot = BlindVaultReplicaSnapshotRecord::from_validated_parts(
            self.workflow_id,
            snapshot_sequence,
            sealed_snapshot.as_slice(),
        );
        if let Err(publication_error) = store.persist_snapshot(&snapshot) {
            // [BLIND-VAULT-SNAPSHOT-CONFIRMATION 2026-08-30 by Codex] Reuse
            // the same randomized sealed container. Re-sealing at the same
            // monotonic sequence would create a conflicting generation and
            // could not prove whether the first atomic replacement survived.
            if let Err(confirmation_error) = store.persist_snapshot(&snapshot) {
                return Err(BlindVaultReplicaDurableSnapshotError::StoreOutcomeUnknown {
                    publication: publication_error,
                    confirmation: confirmation_error,
                });
            }
        }
        Ok(BlindVaultReplicaDurableSnapshot { snapshot_sequence })
    }
}
