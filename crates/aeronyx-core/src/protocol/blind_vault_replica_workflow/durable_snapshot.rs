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
//! - Store errors may represent ambiguous host outcomes; callers should reload
//!   and compare the exact accepted sequence before choosing a new sequence.
//! - Never log source workflow ids, sealed bytes, paths, or store error detail.
//!
//! Last Modified: v1.0.0-DurableWorkflowSnapshot - Initial reusable command
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
#[derive(Debug)]
pub enum BlindVaultReplicaDurableSnapshotError<StoreError> {
    /// Snapshot sealing rejected invalid or oversized workflow state.
    Workflow(BlindVaultReplicaWorkflowError),
    /// Recovery store did not confirm durable publication.
    Store(StoreError),
}

impl<StoreError> fmt::Display for BlindVaultReplicaDurableSnapshotError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
            Self::Store(_) => {
                formatter.write_str("blind vault replica durable snapshot store failed")
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
        store
            .persist_snapshot(&snapshot)
            .map_err(BlindVaultReplicaDurableSnapshotError::Store)?;
        Ok(BlindVaultReplicaDurableSnapshot { snapshot_sequence })
    }
}
