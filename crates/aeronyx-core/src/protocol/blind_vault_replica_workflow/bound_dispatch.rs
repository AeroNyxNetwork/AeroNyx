// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch.rs
// ============================================
//! Durable marker pipeline for exact prepared terminal effects.
//!
//! ## Creation Reason
//! A bound continuation is useful only if orchestration cannot fall back to a
//! generic send permit after persistence. Dedicated marker types preserve the
//! exact effect commitment through every successful durability transition.
//!
//! ## Main Functionality
//! - Wraps the generic prepared journal with exact effect identity.
//! - Preserves the binding through prepared and committed store operations.
//! - Produces an ordered send sequence only after both records are durable.
//! - Redacts and zeroizes the binding commitment.
//!
//! ## Dependencies
//! - `bound_continuation.rs`: constructs the initial bound journal.
//! - `durable_dispatch.rs`: generic durability primitives.
//! - `prepared_effect.rs`: send-time payload matching.
//! - `persistence.rs`: storage-neutral recovery store contract.
//!
//! ## Main Logical Flow
//! 1. Persist the prepared bound journal and receive `PersistedBound`.
//! 2. Commit workflow dispatch and seal its restart snapshot.
//! 3. Persist the journal-bound snapshot and receive `DurableBound`.
//! 4. Consume that marker into one ordered terminal send sequence.
//!
//! ## Important Note For The Next Developer
//! - These markers prove local ordering, not remote node authorization.
//! - Do not add public constructors or accept commitment-only caller input.
//! - Keep the generic path for compatibility; new compound adapters use this.
//! - Network send remains forbidden until `into_terminal_send_sequence`.
//!
//! Last Modified: v1.0.0-BoundDurableDispatch - Initial effect-bound marker
//! pipeline from sealed journal through ordered network send capability.
//! ============================================

use std::fmt;

use zeroize::Zeroize;

use super::{
    BlindVaultReplicaCommittedAttemptDispatch, BlindVaultReplicaDurableAttemptDispatch,
    BlindVaultReplicaDurableDispatchError, BlindVaultReplicaExecution,
    BlindVaultReplicaPersistedAttemptJournal, BlindVaultReplicaPreparedAttemptJournal,
    BlindVaultReplicaPreparedEffectSet, BlindVaultReplicaRecoveryStore, BlindVaultReplicaWorkId,
};
use crate::crypto::keys::IdentityKeyPair;

mod send_sequence;

pub use send_sequence::{
    BlindVaultReplicaTerminalEffectTransport, BlindVaultReplicaTerminalSendContext,
    BlindVaultReplicaTerminalSendError, BlindVaultReplicaTerminalSendSequence,
};

/// Prepared private journal carrying one exact terminal-effect binding.
pub struct BlindVaultReplicaPreparedBoundAttemptJournal {
    journal: BlindVaultReplicaPreparedAttemptJournal,
    effect_set_commitment: [u8; 32],
}

/// Bound prepared journal proven durable by the recovery store.
pub struct BlindVaultReplicaPersistedBoundAttemptJournal<'a> {
    persisted: BlindVaultReplicaPersistedAttemptJournal<'a>,
    prepared: &'a BlindVaultReplicaPreparedBoundAttemptJournal,
}

/// Bound post-dispatch workflow state plus sealed restart snapshot.
pub struct BlindVaultReplicaCommittedBoundAttemptDispatch<'a> {
    committed: BlindVaultReplicaCommittedAttemptDispatch<'a>,
    prepared: &'a BlindVaultReplicaPreparedBoundAttemptJournal,
}

/// Final durability marker that can become one ordered send sequence.
pub struct BlindVaultReplicaDurableBoundAttemptDispatch<'a, 'b> {
    durable: BlindVaultReplicaDurableAttemptDispatch<'a, 'b>,
    prepared: &'a BlindVaultReplicaPreparedBoundAttemptJournal,
}

impl BlindVaultReplicaPreparedBoundAttemptJournal {
    pub(super) fn from_validated_parts(
        journal: BlindVaultReplicaPreparedAttemptJournal,
        effect_set: &BlindVaultReplicaPreparedEffectSet,
    ) -> Option<Self> {
        if journal.work_id() != effect_set.work_id()
            || journal.attempt() != effect_set.attempt()
            || journal.planned_dispatch_at_ms() != effect_set.planned_dispatch_at_ms()
            || journal.evidence_deadline_ms() != effect_set.evidence_deadline_ms()
        {
            return None;
        }
        Some(Self {
            journal,
            effect_set_commitment: effect_set.commitment(),
        })
    }

    /// Exact source-local work item represented by this prepared journal.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.journal.work_id()
    }

    /// Exact bounded workflow attempt represented by this journal.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.journal.attempt()
    }

    /// Source timestamp that the later dispatch transition must use.
    #[must_use]
    pub const fn planned_dispatch_at_ms(&self) -> u64 {
        self.journal.planned_dispatch_at_ms()
    }

    /// Policy-derived terminal evidence deadline for this attempt.
    #[must_use]
    pub const fn evidence_deadline_ms(&self) -> u64 {
        self.journal.evidence_deadline_ms()
    }

    /// Monotonic private-journal sequence persisted before dispatch.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal.journal_sequence()
    }

    /// Authenticated journal ciphertext; never log or send it to a node.
    #[must_use]
    pub fn sealed_journal(&self) -> &[u8] {
        self.journal.sealed_journal()
    }

    /// Persists this exact bound journal before workflow mutation.
    pub fn persist_for_dispatch<'a, Store>(
        &'a self,
        store: &mut Store,
    ) -> Result<BlindVaultReplicaPersistedBoundAttemptJournal<'a>, Store::Error>
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let persisted = self.journal.persist_for_dispatch(store)?;
        Ok(BlindVaultReplicaPersistedBoundAttemptJournal {
            persisted,
            prepared: self,
        })
    }

    fn matches_effect_set(&self, effect_set: &BlindVaultReplicaPreparedEffectSet) -> bool {
        self.work_id() == effect_set.work_id()
            && self.attempt() == effect_set.attempt()
            && self.planned_dispatch_at_ms() == effect_set.planned_dispatch_at_ms()
            && self.evidence_deadline_ms() == effect_set.evidence_deadline_ms()
            && self.effect_set_commitment == effect_set.commitment()
    }
}

impl Drop for BlindVaultReplicaPreparedBoundAttemptJournal {
    fn drop(&mut self) {
        self.effect_set_commitment.zeroize();
    }
}

impl BlindVaultReplicaExecution {
    /// Commits a durably persisted bound attempt and seals workflow state.
    ///
    /// [BLIND-VAULT-BOUND-DURABLE-DISPATCH 2026-08-29 by Codex] The generic
    /// primitive performs the only workflow mutation; this wrapper carries the
    /// exact effect identity forward without widening construction access.
    pub fn commit_persisted_bound_attempt_dispatch<'a>(
        &mut self,
        identity: &IdentityKeyPair,
        persisted: BlindVaultReplicaPersistedBoundAttemptJournal<'a>,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaCommittedBoundAttemptDispatch<'a>,
        BlindVaultReplicaDurableDispatchError,
    > {
        let prepared = persisted.prepared;
        let committed = self.commit_persisted_attempt_dispatch(
            identity,
            persisted.persisted,
            snapshot_sequence,
        )?;
        Ok(BlindVaultReplicaCommittedBoundAttemptDispatch {
            committed,
            prepared,
        })
    }
}

impl<'a> BlindVaultReplicaCommittedBoundAttemptDispatch<'a> {
    /// Exact work item now durably represented by the prepared journal.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.committed.work_id()
    }

    /// Exact workflow attempt now awaiting terminal evidence.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.committed.attempt()
    }

    /// Persists the post-dispatch snapshot before network send is possible.
    pub fn persist_for_network_send<'b, Store>(
        &'b self,
        store: &mut Store,
    ) -> Result<BlindVaultReplicaDurableBoundAttemptDispatch<'a, 'b>, Store::Error>
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let durable = self.committed.persist_for_network_send(store)?;
        Ok(BlindVaultReplicaDurableBoundAttemptDispatch {
            durable,
            prepared: self.prepared,
        })
    }
}

impl BlindVaultReplicaDurableBoundAttemptDispatch<'_, '_> {
    /// Consumes the final durability marker into one ordered send capability.
    pub fn into_terminal_send_sequence<'effects>(
        self,
        effect_set: &'effects BlindVaultReplicaPreparedEffectSet,
    ) -> Result<
        BlindVaultReplicaTerminalSendSequence<'effects>,
        BlindVaultReplicaDurableDispatchError,
    > {
        if !self.prepared.matches_effect_set(effect_set) {
            return Err(BlindVaultReplicaDurableDispatchError::StateMismatch);
        }
        Ok(BlindVaultReplicaTerminalSendSequence::from_durable_parts(
            effect_set,
            self.durable.snapshot_sequence(),
            self.durable.journal_sequence(),
        ))
    }
}

impl fmt::Debug for BlindVaultReplicaPreparedBoundAttemptJournal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPreparedBoundAttemptJournal")
            .field("attempt", &self.attempt())
            .field("journal_sequence", &self.journal_sequence())
            .field("effect_set_commitment", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaPersistedBoundAttemptJournal<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaPersistedBoundAttemptJournal")
            .field("attempt", &self.prepared.attempt())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaCommittedBoundAttemptDispatch<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCommittedBoundAttemptDispatch")
            .field("attempt", &self.attempt())
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaDurableBoundAttemptDispatch<'_, '_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaDurableBoundAttemptDispatch")
            .field("attempt", &self.durable.attempt())
            .field("snapshot_sequence", &self.durable.snapshot_sequence())
            .field("journal_sequence", &self.durable.journal_sequence())
            .finish_non_exhaustive()
    }
}
