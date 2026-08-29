// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       recovered_bound_attempt.rs
// ============================================
//! Committed-only recovery permit for exact bound terminal effects.
//!
//! ## Creation Reason
//! The authenticated recovery loader can distinguish Prepared from Committed,
//! but restored effects still need a typed capability proving that network
//! dispatch crossed both durable boundaries before process loss.
//!
//! ## Main Functionality
//! - Accepts only `BlindVaultReplicaLoadedRecovery::Committed`.
//! - Restores the identity-sealed effect/session composition.
//! - Binds accepted snapshot and journal high-water sequences into a permit.
//! - Consumes that permit into the same ordered send sequence as live dispatch.
//!
//! ## Dependencies
//! - `recovery_loader.rs`: authenticated durable phase classification.
//! - `bound_continuation.rs`: exact restored effects and reply sessions.
//! - `bound_dispatch/send_sequence.rs`: verified transport capability.
//!
//! ## Main Logical Flow
//! 1. Consume one authenticated loaded recovery result.
//! 2. Reject Resolved and Prepared phases.
//! 3. Decode the committed private journal against restored workflow state.
//! 4. Return restored workflow, private continuation, and one send permit.
//!
//! ## Important Note For The Next Developer
//! - Prepared recovery may only be aborted; it can never create this permit.
//! - Starting at effect zero intentionally replays exact idempotent requests.
//! - Terminal request IDs must remain stable across restart and retransmission.
//! - Never expose work identity or commitments through telemetry.
//!
//! Last Modified: v1.0.0-RecoveredBoundAttempt - Initial committed-only
//! restoration and ordered resend authorization.
//! ============================================

use std::fmt;

use thiserror::Error;
use zeroize::Zeroize;

use super::{
    BlindVaultReplicaBoundAttemptContinuation, BlindVaultReplicaBoundContinuationError,
    BlindVaultReplicaLoadedRecovery, BlindVaultReplicaPreparedEffectSet,
    BlindVaultReplicaRestoredExecution, BlindVaultReplicaTerminalSendSequence,
    BlindVaultReplicaWorkId,
};

/// Authenticated committed workflow, continuation, and resend authority.
pub struct BlindVaultReplicaRecoveredBoundAttempt {
    restored: BlindVaultReplicaRestoredExecution,
    continuation: BlindVaultReplicaBoundAttemptContinuation,
    send_permit: BlindVaultReplicaRecoveredSendPermit,
}

/// One exact committed-phase authority to reconstruct an ordered send sequence.
pub struct BlindVaultReplicaRecoveredSendPermit {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
    dispatched_at_ms: u64,
    evidence_deadline_ms: u64,
    snapshot_sequence: u64,
    journal_sequence: u64,
    effect_set_commitment: [u8; 32],
}

impl BlindVaultReplicaLoadedRecovery {
    /// Restores a payload-bound attempt only from authenticated Committed state.
    ///
    /// [BLIND-VAULT-RECOVERED-BOUND-ATTEMPT 2026-08-29 by Codex] Pattern
    /// matching the typed phase prevents prepared-journal cleanup authority
    /// from being confused with post-dispatch network-send authority.
    pub fn into_recovered_bound_attempt(
        self,
    ) -> Result<BlindVaultReplicaRecoveredBoundAttempt, BlindVaultReplicaRecoveredBoundAttemptError>
    {
        let Self::Committed {
            restored,
            attempt_journal,
        } = self
        else {
            return Err(BlindVaultReplicaRecoveredBoundAttemptError::NotCommitted);
        };
        let snapshot_sequence = restored.snapshot_sequence();
        let journal_sequence = attempt_journal.journal_sequence();
        let work_id = attempt_journal.work_id();
        let attempt = attempt_journal.attempt();
        let dispatched_at_ms = attempt_journal.dispatched_at_ms();
        let evidence_deadline_ms = attempt_journal.evidence_deadline_ms();
        let continuation = attempt_journal
            .into_bound_attempt_continuation(restored.execution())
            .map_err(BlindVaultReplicaRecoveredBoundAttemptError::Continuation)?;
        let send_permit = BlindVaultReplicaRecoveredSendPermit {
            work_id,
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
            snapshot_sequence,
            journal_sequence,
            effect_set_commitment: continuation.effect_set().commitment(),
        };
        Ok(BlindVaultReplicaRecoveredBoundAttempt {
            restored,
            continuation,
            send_permit,
        })
    }
}

impl BlindVaultReplicaRecoveredBoundAttempt {
    /// Authenticated restored workflow that remains awaiting evidence.
    #[must_use]
    pub const fn restored(&self) -> &BlindVaultReplicaRestoredExecution {
        &self.restored
    }

    /// Exact restored payload bindings and private reply sessions.
    #[must_use]
    pub const fn continuation(&self) -> &BlindVaultReplicaBoundAttemptContinuation {
        &self.continuation
    }

    /// Transfers workflow, continuation, and one resend permit to the runtime.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        BlindVaultReplicaRestoredExecution,
        BlindVaultReplicaBoundAttemptContinuation,
        BlindVaultReplicaRecoveredSendPermit,
    ) {
        (self.restored, self.continuation, self.send_permit)
    }
}

impl BlindVaultReplicaRecoveredSendPermit {
    /// Consumes committed recovery authority into an ordered resend sequence.
    pub fn into_terminal_send_sequence<'effects>(
        self,
        effect_set: &'effects BlindVaultReplicaPreparedEffectSet,
    ) -> Result<
        BlindVaultReplicaTerminalSendSequence<'effects>,
        BlindVaultReplicaRecoveredBoundAttemptError,
    > {
        if self.work_id != effect_set.work_id()
            || self.attempt != effect_set.attempt()
            || self.dispatched_at_ms != effect_set.planned_dispatch_at_ms()
            || self.evidence_deadline_ms != effect_set.evidence_deadline_ms()
            || self.effect_set_commitment != effect_set.commitment()
        {
            return Err(BlindVaultReplicaRecoveredBoundAttemptError::BindingMismatch);
        }
        Ok(BlindVaultReplicaTerminalSendSequence::from_durable_parts(
            effect_set,
            self.snapshot_sequence,
            self.journal_sequence,
        ))
    }

    /// Exact durable workflow snapshot high-water accepted by the loader.
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    /// Exact durable private-journal high-water accepted by the loader.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.journal_sequence
    }
}

impl Drop for BlindVaultReplicaRecoveredSendPermit {
    fn drop(&mut self) {
        self.effect_set_commitment.zeroize();
    }
}

impl fmt::Debug for BlindVaultReplicaRecoveredBoundAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRecoveredBoundAttempt")
            .field("snapshot_sequence", &self.restored.snapshot_sequence())
            .field("attempt", &self.continuation.effect_set().attempt())
            .field("private_continuation", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaRecoveredSendPermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRecoveredSendPermit")
            .field("attempt", &self.attempt)
            .field("snapshot_sequence", &self.snapshot_sequence)
            .field("journal_sequence", &self.journal_sequence)
            .field("effect_set_commitment", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

/// Fail-closed committed-bound recovery errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaRecoveredBoundAttemptError {
    /// Resolved or Prepared recovery cannot authorize a network resend.
    #[error("blind vault replica recovery phase is not committed")]
    NotCommitted,
    /// Identity-sealed effect/session continuation failed restoration.
    #[error(transparent)]
    Continuation(#[from] BlindVaultReplicaBoundContinuationError),
    /// Recovered send permit did not match the supplied effect set.
    #[error("blind vault replica recovered terminal effect binding mismatched")]
    BindingMismatch,
}
