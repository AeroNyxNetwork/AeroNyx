// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_continuation.rs
// ============================================
//! Restart-safe composition of prepared effects and private reply sessions.
//!
//! ## Creation Reason
//! Restoring reply keys without restoring the exact ordered request bindings
//! could let an adapter reuse valid sessions for different terminal payloads.
//! Both domains must cross the identity-sealed journal boundary together.
//!
//! ## Main Functionality
//! - Owns one prepared effect set and one private attempt continuation.
//! - Requires exactly one single-use reply session per terminal effect.
//! - Encodes both domains into one bounded journal-private container.
//! - Restores only against the authenticated in-flight workflow action.
//!
//! ## Dependencies
//! - `attempt_continuation.rs`: adapter state and reply-session ownership.
//! - `attempt_journal.rs`: identity-sealed, action-bound persistence.
//! - `prepared_effect.rs`: exact payload-blind send bindings.
//! - `execution.rs`: authenticated in-flight action contract.
//!
//! ## Main Logical Flow
//! 1. Build and validate ordered effects and matching reply sessions.
//! 2. Encode both into one private-state value and seal the attempt journal.
//! 3. After restart, authenticate the workflow and journal first.
//! 4. Decode, revalidate contract and commitment, then return typed ownership.
//!
//! ## Important Note For The Next Developer
//! - This private format is never a network frame, API value, or ledger entry.
//! - Session/effect cardinality is exact; do not permit implicit reuse.
//! - Never log encoded bytes, sessions, commitments, or adapter state.
//! - Generic journal APIs remain for compatibility but provide fewer guarantees.
//!
//! Last Modified: v1.0.0-BoundAttemptContinuation - Initial restart-safe
//! composition of ordered effects and single-use reply sessions.
//! ============================================

use std::fmt;

use thiserror::Error;
use zeroize::Zeroize;

use super::{
    BlindVaultReplicaAttemptContinuation, BlindVaultReplicaAttemptJournal,
    BlindVaultReplicaAttemptJournalError, BlindVaultReplicaExecution,
    BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaPreparedEffectError,
    BlindVaultReplicaPreparedEffectSet, BlindVaultReplicaWorkState,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES,
};
use crate::crypto::keys::IdentityKeyPair;

const BOUND_CONTINUATION_MAGIC: [u8; 4] = *b"AXBC";
const BOUND_CONTINUATION_VERSION_V1: u16 = 1;
const BOUND_CONTINUATION_HEADER_BYTES: usize = 4 + 2 + 4 + 4;

/// Exact prepared effects plus their private one-time response sessions.
pub struct BlindVaultReplicaBoundAttemptContinuation {
    effect_set: BlindVaultReplicaPreparedEffectSet,
    continuation: BlindVaultReplicaAttemptContinuation,
}

impl BlindVaultReplicaBoundAttemptContinuation {
    /// Creates a cardinality-safe attempt composition.
    ///
    /// [BLIND-VAULT-BOUND-CONTINUATION 2026-08-29 by Codex] One reply session
    /// per effect prevents response keys from being silently reused or shifted
    /// when a compound action is resumed after process loss.
    pub fn new(
        effect_set: BlindVaultReplicaPreparedEffectSet,
        continuation: BlindVaultReplicaAttemptContinuation,
    ) -> Result<Self, BlindVaultReplicaBoundContinuationError> {
        if effect_set.effects().len() != continuation.reply_session_count() {
            return Err(
                BlindVaultReplicaBoundContinuationError::SessionEffectCountMismatch {
                    effects: effect_set.effects().len(),
                    sessions: continuation.reply_session_count(),
                },
            );
        }
        Ok(Self {
            effect_set,
            continuation,
        })
    }

    /// Exact payload-blind terminal effects in dispatch order.
    #[must_use]
    pub const fn effect_set(&self) -> &BlindVaultReplicaPreparedEffectSet {
        &self.effect_set
    }

    /// Private adapter state and one-time response sessions.
    #[must_use]
    pub const fn continuation(&self) -> &BlindVaultReplicaAttemptContinuation {
        &self.continuation
    }

    /// Transfers both domains to the runtime owner.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        BlindVaultReplicaPreparedEffectSet,
        BlindVaultReplicaAttemptContinuation,
    ) {
        (self.effect_set, self.continuation)
    }

    fn encode_restart_state(&self) -> Result<Vec<u8>, BlindVaultReplicaBoundContinuationError> {
        let mut effect_binding = self.effect_set.encode_restart_binding();
        let mut continuation = self.continuation.encode_restart_state()?;
        let effect_length = u32::try_from(effect_binding.len())
            .map_err(|_| BlindVaultReplicaAttemptJournalError::TooLarge)?;
        let continuation_length = u32::try_from(continuation.len())
            .map_err(|_| BlindVaultReplicaAttemptJournalError::TooLarge)?;
        let total_length = BOUND_CONTINUATION_HEADER_BYTES
            .checked_add(effect_binding.len())
            .and_then(|length| length.checked_add(continuation.len()))
            .ok_or(BlindVaultReplicaAttemptJournalError::TooLarge)?;
        if total_length > MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES {
            effect_binding.zeroize();
            continuation.zeroize();
            return Err(BlindVaultReplicaAttemptJournalError::TooLarge.into());
        }
        let mut encoded = Vec::with_capacity(total_length);
        encoded.extend_from_slice(&BOUND_CONTINUATION_MAGIC);
        encoded.extend_from_slice(&BOUND_CONTINUATION_VERSION_V1.to_be_bytes());
        encoded.extend_from_slice(&effect_length.to_be_bytes());
        encoded.extend_from_slice(&continuation_length.to_be_bytes());
        encoded.extend_from_slice(&effect_binding);
        encoded.extend_from_slice(&continuation);
        effect_binding.zeroize();
        continuation.zeroize();
        Ok(encoded)
    }

    fn decode_restart_state(
        execution: &BlindVaultReplicaExecution,
        work_id: super::BlindVaultReplicaWorkId,
        attempt: u8,
        dispatched_at_ms: u64,
        evidence_deadline_ms: u64,
        encoded: &[u8],
    ) -> Result<Self, BlindVaultReplicaBoundContinuationError> {
        if encoded.len() < BOUND_CONTINUATION_HEADER_BYTES
            || encoded[..4] != BOUND_CONTINUATION_MAGIC
        {
            return Err(BlindVaultReplicaBoundContinuationError::Malformed);
        }
        if u16::from_be_bytes([encoded[4], encoded[5]]) != BOUND_CONTINUATION_VERSION_V1 {
            return Err(BlindVaultReplicaBoundContinuationError::VersionUnsupported);
        }
        let effect_length = read_u32_length(encoded, 6)?;
        let continuation_length = read_u32_length(encoded, 10)?;
        let expected_length = BOUND_CONTINUATION_HEADER_BYTES
            .checked_add(effect_length)
            .and_then(|length| length.checked_add(continuation_length))
            .ok_or(BlindVaultReplicaBoundContinuationError::Malformed)?;
        if encoded.len() != expected_length {
            return Err(BlindVaultReplicaBoundContinuationError::Malformed);
        }

        let item = execution
            .items()
            .iter()
            .find(|item| item.id() == work_id)
            .ok_or(BlindVaultReplicaBoundContinuationError::StateMismatch)?;
        let expected_state = BlindVaultReplicaWorkState::AwaitingEvidence {
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
        };
        if item.state() != expected_state {
            return Err(BlindVaultReplicaBoundContinuationError::StateMismatch);
        }
        let effect_end = BOUND_CONTINUATION_HEADER_BYTES + effect_length;
        let effect_set = BlindVaultReplicaPreparedEffectSet::decode_restart_binding(
            work_id,
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
            item.dispatch_contract(),
            &encoded[BOUND_CONTINUATION_HEADER_BYTES..effect_end],
        )?;
        let continuation = BlindVaultReplicaAttemptContinuation::decode_restart_state(
            &encoded[effect_end..expected_length],
        )?;
        Self::new(effect_set, continuation)
    }
}

impl fmt::Debug for BlindVaultReplicaBoundAttemptContinuation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaBoundAttemptContinuation")
            .field("attempt", &self.effect_set.attempt())
            .field("effect_count", &self.effect_set.effects().len())
            .field("private_continuation", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

/// Fail-closed composition and restoration errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaBoundContinuationError {
    /// Identity-sealed attempt journal preparation or continuation failed.
    #[error(transparent)]
    Attempt(#[from] BlindVaultReplicaAttemptJournalError),
    /// Prepared effect restoration or contract validation failed.
    #[error(transparent)]
    Effect(#[from] BlindVaultReplicaPreparedEffectError),
    /// One-time reply sessions did not map exactly to ordered effects.
    #[error("blind vault replica reply-session and terminal-effect counts differ")]
    SessionEffectCountMismatch { effects: usize, sessions: usize },
    /// The bound private container was truncated or internally inconsistent.
    #[error("blind vault replica bound attempt continuation is malformed")]
    Malformed,
    /// The bound private container uses an unsupported local format version.
    #[error("blind vault replica bound attempt continuation version is unsupported")]
    VersionUnsupported,
    /// Authenticated journal metadata did not match the supplied execution.
    #[error("blind vault replica bound attempt continuation state mismatched")]
    StateMismatch,
}

impl BlindVaultReplicaExecution {
    /// Seals an exact effect/session composition before dispatch is committed.
    pub fn prepare_bound_attempt_continuation_for_dispatch(
        &self,
        identity: &IdentityKeyPair,
        journal_sequence: u64,
        retain_until_ms: u64,
        bound: &BlindVaultReplicaBoundAttemptContinuation,
    ) -> Result<BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaBoundContinuationError>
    {
        let mut encoded = bound.encode_restart_state()?;
        let prepared = self.prepare_attempt_journal_for_dispatch(
            identity,
            bound.effect_set.work_id(),
            bound.effect_set.planned_dispatch_at_ms(),
            journal_sequence,
            retain_until_ms,
            &encoded,
        );
        encoded.zeroize();
        prepared.map_err(BlindVaultReplicaBoundContinuationError::from)
    }
}

impl BlindVaultReplicaAttemptJournal {
    /// Restores exact prepared sends after workflow and journal authentication.
    pub fn into_bound_attempt_continuation(
        self,
        execution: &BlindVaultReplicaExecution,
    ) -> Result<BlindVaultReplicaBoundAttemptContinuation, BlindVaultReplicaBoundContinuationError>
    {
        let work_id = self.work_id();
        let attempt = self.attempt();
        let dispatched_at_ms = self.dispatched_at_ms();
        let evidence_deadline_ms = self.evidence_deadline_ms();
        let mut encoded = self.into_private_state();
        let restored = BlindVaultReplicaBoundAttemptContinuation::decode_restart_state(
            execution,
            work_id,
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
            &encoded,
        );
        encoded.zeroize();
        restored
    }
}

fn read_u32_length(
    encoded: &[u8],
    offset: usize,
) -> Result<usize, BlindVaultReplicaBoundContinuationError> {
    let end = offset
        .checked_add(4)
        .ok_or(BlindVaultReplicaBoundContinuationError::Malformed)?;
    let bytes: [u8; 4] = encoded
        .get(offset..end)
        .ok_or(BlindVaultReplicaBoundContinuationError::Malformed)?
        .try_into()
        .map_err(|_| BlindVaultReplicaBoundContinuationError::Malformed)?;
    usize::try_from(u32::from_be_bytes(bytes))
        .map_err(|_| BlindVaultReplicaBoundContinuationError::Malformed)
}
