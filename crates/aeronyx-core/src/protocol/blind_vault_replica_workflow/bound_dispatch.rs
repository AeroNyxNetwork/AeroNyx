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
//! - Retains committed state across ambiguous store publication failures.
//! - Produces an ordered send sequence only after both records are durable.
//! - Gates old-lease retirement with workflow and terminal authorization.
//! - Composes replacement replies into one source-private lifecycle policy.
//! - Exposes one extensible, privacy-safe terminal failure classifier.
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
//! - Markers prove local ordering; retirement adds an exact terminal permit.
//! - Do not add public constructors or accept commitment-only caller input.
//! - Keep the generic path for compatibility; new compound adapters use this.
//! - Network send remains forbidden until `into_terminal_send_sequence`.
//!
//! Last Modified: v1.13.0-PrivacySafeActionDiagnostics - Standardized redacted
//! clock and private policy diagnostics across every action reply state machine.
//! v1.12.0-PrivacySafePolicyDiagnostics - Redacted generic
//! source-private policy errors at the request-bound verifier boundary.
//! v1.11.0-PrivacySafeTransportDiagnostics - Redacted generic
//! route, sender, transport, reply, and verifier diagnostic details.
//! v1.10.0-OwnedResolutionBinding - Exposed the exact opaque
//! committed binding needed to resolve an owned bound attempt after evidence.
//! v1.9.0-RecoverablePublication - Added consuming publication
//! permits whose errors retain the exact committed state for idempotent retry.
//! v1.8.0-TerminalFailureClassification - Exported the shared
//! bounded runtime failure classification boundary.
//! v1.7.0-RenewalReplyPolicy - Added exact lease-generation
//! compare-and-swap verification for single-effect renewal attempts.
//! v1.6.0-ObservationReplyPolicy - Added exact single-effect
//! fresh inventory verification for observation-retry attempts.
//! v1.5.0-ReconcileReplyPolicy - Added ordered write/delete and
//! post-mutation inventory verification for exact reconciliation attempts.
//! v1.4.0-ProvisioningReplyPolicy - Added bounded aggregate
//! admission/write/inventory verification for exact provisioning attempts.
//! v1.3.0-ReplacementReplyPolicy - Added a reusable typed
//! admission/write/inventory/retirement reply state machine.
//! v1.2.0-RetirementPermitGate - Required workflow and route
//! terminal authorization before old-lease retirement can reach transport.
//! v1.1.0-OwnedTerminalRuntime - Added one-step ownership
//! transfer from a durable bound attempt into a self-contained runtime.
//! v1.0.0-BoundDurableDispatch - Initial effect-bound marker pipeline from
//! sealed journal through ordered network send capability.
//! ============================================

use std::fmt;

use thiserror::Error;
use zeroize::Zeroize;

use super::{
    BlindVaultReplicaBoundAttemptContinuation, BlindVaultReplicaCommittedAttemptBinding,
    BlindVaultReplicaCommittedAttemptDispatch, BlindVaultReplicaDurableAttemptDispatch,
    BlindVaultReplicaDurableDispatchError, BlindVaultReplicaExecution,
    BlindVaultReplicaOwnedDurableAttemptDispatch, BlindVaultReplicaPersistedAttemptJournal,
    BlindVaultReplicaPreparedAttemptJournal, BlindVaultReplicaPreparedEffectSet,
    BlindVaultReplicaRecoveryStore, BlindVaultReplicaWorkId,
};
use crate::crypto::keys::IdentityKeyPair;

mod attempt_runtime;
mod observation_reply_policy;
mod onion_transport;
mod provisioning_reply_policy;
mod reconcile_reply_policy;
mod renewal_reply_policy;
mod replacement_reply_policy;
mod request_bound_verifier;
mod send_sequence;

pub use attempt_runtime::{
    BlindVaultReplicaTerminalAttemptError, BlindVaultReplicaTerminalAttemptRuntime,
    BlindVaultReplicaTerminalAttemptRuntimeBuildError, BlindVaultReplicaTerminalAttemptState,
    BlindVaultReplicaTerminalReplyVerifier, BlindVaultReplicaTerminalVerificationFailure,
};
pub use observation_reply_policy::{
    BlindVaultReplicaCompletedObservation, BlindVaultReplicaObservationReplyOutcome,
    BlindVaultReplicaObservationReplyPolicy, BlindVaultReplicaObservationReplyPolicyBuildError,
    BlindVaultReplicaObservationReplyPolicyError,
};
pub use onion_transport::{
    BlindVaultReplicaOnionDispatchPlan, BlindVaultReplicaOnionEnvelopeSender,
    BlindVaultReplicaOnionRouteProvider, BlindVaultReplicaVerifiedOnionTransport,
    BlindVaultReplicaVerifiedOnionTransportError,
};
pub use provisioning_reply_policy::{
    BlindVaultReplicaCompletedProvisioning, BlindVaultReplicaProvisioningReplyOutcome,
    BlindVaultReplicaProvisioningReplyPolicy, BlindVaultReplicaProvisioningReplyPolicyBuildError,
    BlindVaultReplicaProvisioningReplyPolicyError,
};
pub use reconcile_reply_policy::{
    BlindVaultReplicaCompletedReconciliation, BlindVaultReplicaReconcileReplyOutcome,
    BlindVaultReplicaReconcileReplyPolicy, BlindVaultReplicaReconcileReplyPolicyBuildError,
    BlindVaultReplicaReconcileReplyPolicyError,
};
pub use renewal_reply_policy::{
    BlindVaultReplicaCompletedRenewal, BlindVaultReplicaRenewalReplyOutcome,
    BlindVaultReplicaRenewalReplyPolicy, BlindVaultReplicaRenewalReplyPolicyBuildError,
    BlindVaultReplicaRenewalReplyPolicyError,
};
pub use replacement_reply_policy::{
    BlindVaultReplicaCompletedReplacement, BlindVaultReplicaReplacementAuthorizationError,
    BlindVaultReplicaReplacementPermitIssueError, BlindVaultReplicaReplacementReplyOutcome,
    BlindVaultReplicaReplacementReplyPolicy, BlindVaultReplicaReplacementReplyPolicyBuildError,
    BlindVaultReplicaReplacementReplyPolicyError,
    BlindVaultReplicaReplacementRetirementDispatchError,
};
pub use request_bound_verifier::{
    BlindVaultReplicaPrivateReplyPolicy, BlindVaultReplicaRequestBoundReply,
    BlindVaultReplicaRequestBoundReplyError, BlindVaultReplicaRequestBoundReplyVerifier,
    BlindVaultReplicaVerificationClock,
};
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
#[must_use = "committed bound dispatch must remain owned until publication succeeds"]
pub struct BlindVaultReplicaCommittedBoundAttemptDispatch<'a> {
    committed: BlindVaultReplicaCommittedAttemptDispatch<'a>,
    prepared: &'a BlindVaultReplicaPreparedBoundAttemptJournal,
}

/// Final durability marker that can become one ordered send sequence.
pub struct BlindVaultReplicaDurableBoundAttemptDispatch<'a, 'b> {
    durable: BlindVaultReplicaDurableAttemptDispatch<'a, 'b>,
    prepared: &'a BlindVaultReplicaPreparedBoundAttemptJournal,
}

/// Owned final durability marker for one exact bound terminal effect set.
///
/// [BLIND-VAULT-BOUND-RECOVERABLE-PUBLICATION 2026-08-30 by Codex] This
/// marker carries only source-private correlation and a commitment after the
/// sealed snapshot has been accepted. It can safely outlive the publication
/// call without borrowing the committed recovery payload.
#[must_use = "durable bound dispatch authority must reach an exact send path"]
pub struct BlindVaultReplicaOwnedDurableBoundAttemptDispatch {
    durable: BlindVaultReplicaOwnedDurableAttemptDispatch,
    planned_dispatch_at_ms: u64,
    evidence_deadline_ms: u64,
    effect_set_commitment: [u8; 32],
}

/// Recoverable failure to publish one committed bound dispatch generation.
///
/// The error owns the complete marker needed to retry the same journal-bound
/// snapshot. Callers may inspect the adapter error without gaining access to
/// private continuation state or sealed recovery bytes.
#[must_use = "publication failure retains committed state and must be retried or recovered"]
pub struct BlindVaultReplicaBoundCommitPublicationError<'a, StoreError> {
    committed: BlindVaultReplicaCommittedBoundAttemptDispatch<'a>,
    source: StoreError,
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

    /// Consumes this bound marker into owned authority or a retryable error.
    ///
    /// New orchestration should use this path so an ambiguous store error
    /// cannot discard the only in-process copy of the committed snapshot.
    pub fn publish_for_network_send<Store>(
        self,
        store: &mut Store,
    ) -> Result<
        BlindVaultReplicaOwnedDurableBoundAttemptDispatch,
        BlindVaultReplicaBoundCommitPublicationError<'a, Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        let Self {
            committed,
            prepared,
        } = self;
        match committed.publish_for_network_send(store) {
            Ok(durable) => Ok(BlindVaultReplicaOwnedDurableBoundAttemptDispatch {
                durable,
                planned_dispatch_at_ms: prepared.planned_dispatch_at_ms(),
                evidence_deadline_ms: prepared.evidence_deadline_ms(),
                effect_set_commitment: prepared.effect_set_commitment,
            }),
            Err(error) => {
                let (committed, source) = error.into_parts();
                Err(BlindVaultReplicaBoundCommitPublicationError {
                    committed: Self {
                        committed,
                        prepared,
                    },
                    source,
                })
            }
        }
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

    /// Consumes durable authority and complete private state into one owner.
    ///
    /// [BLIND-VAULT-OWNED-TERMINAL-RUNTIME 2026-08-29 by Codex] This path
    /// preserves the borrowed sequence API while giving long-lived adapters a
    /// non-self-referential runtime that owns effect bindings and sessions.
    pub fn into_terminal_attempt_runtime(
        self,
        bound: BlindVaultReplicaBoundAttemptContinuation,
    ) -> Result<BlindVaultReplicaTerminalAttemptRuntime<'static>, BlindVaultReplicaBoundRuntimeError>
    {
        if !self.prepared.matches_effect_set(bound.effect_set()) {
            return Err(BlindVaultReplicaDurableDispatchError::StateMismatch.into());
        }
        let snapshot_sequence = self.durable.snapshot_sequence();
        let journal_sequence = self.durable.journal_sequence();
        let (effect_set, continuation) = bound.into_parts();
        let send_sequence =
            BlindVaultReplicaTerminalSendSequence::<'static>::from_owned_durable_parts(
                effect_set,
                snapshot_sequence,
                journal_sequence,
            );
        BlindVaultReplicaTerminalAttemptRuntime::new(send_sequence, continuation)
            .map_err(BlindVaultReplicaBoundRuntimeError::from)
    }
}

impl BlindVaultReplicaOwnedDurableBoundAttemptDispatch {
    /// Exact source-local work item now permitted to reach transport.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.durable.work_id()
    }

    /// Exact bounded attempt represented by durable local state.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.durable.attempt()
    }

    /// Accepted post-dispatch snapshot sequence.
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.durable.snapshot_sequence()
    }

    /// Accepted private-journal sequence.
    #[must_use]
    pub const fn journal_sequence(&self) -> u64 {
        self.durable.journal_sequence()
    }

    /// Exact opaque journal binding required for durable terminal resolution.
    #[must_use]
    pub fn committed_attempt_binding(&self) -> BlindVaultReplicaCommittedAttemptBinding {
        self.durable.committed_attempt_binding()
    }

    /// Consumes owned durability authority into one ordered send capability.
    pub fn into_terminal_send_sequence<'effects>(
        self,
        effect_set: &'effects BlindVaultReplicaPreparedEffectSet,
    ) -> Result<
        BlindVaultReplicaTerminalSendSequence<'effects>,
        BlindVaultReplicaDurableDispatchError,
    > {
        if !self.matches_effect_set(effect_set) {
            return Err(BlindVaultReplicaDurableDispatchError::StateMismatch);
        }
        Ok(BlindVaultReplicaTerminalSendSequence::from_durable_parts(
            effect_set,
            self.snapshot_sequence(),
            self.journal_sequence(),
        ))
    }

    /// Consumes owned durability authority and private state into one runtime.
    pub fn into_terminal_attempt_runtime(
        self,
        bound: BlindVaultReplicaBoundAttemptContinuation,
    ) -> Result<BlindVaultReplicaTerminalAttemptRuntime<'static>, BlindVaultReplicaBoundRuntimeError>
    {
        if !self.matches_effect_set(bound.effect_set()) {
            return Err(BlindVaultReplicaDurableDispatchError::StateMismatch.into());
        }
        let snapshot_sequence = self.snapshot_sequence();
        let journal_sequence = self.journal_sequence();
        let (effect_set, continuation) = bound.into_parts();
        let send_sequence =
            BlindVaultReplicaTerminalSendSequence::<'static>::from_owned_durable_parts(
                effect_set,
                snapshot_sequence,
                journal_sequence,
            );
        BlindVaultReplicaTerminalAttemptRuntime::new(send_sequence, continuation)
            .map_err(BlindVaultReplicaBoundRuntimeError::from)
    }

    fn matches_effect_set(&self, effect_set: &BlindVaultReplicaPreparedEffectSet) -> bool {
        self.work_id() == effect_set.work_id()
            && self.attempt() == effect_set.attempt()
            && self.planned_dispatch_at_ms == effect_set.planned_dispatch_at_ms()
            && self.evidence_deadline_ms == effect_set.evidence_deadline_ms()
            && self.effect_set_commitment == effect_set.commitment()
    }
}

impl Drop for BlindVaultReplicaOwnedDurableBoundAttemptDispatch {
    fn drop(&mut self) {
        self.effect_set_commitment.zeroize();
    }
}

impl<'a, StoreError> BlindVaultReplicaBoundCommitPublicationError<'a, StoreError> {
    /// Coarse adapter error without exposing committed private state.
    #[must_use]
    pub const fn store_error(&self) -> &StoreError {
        &self.source
    }

    /// Retries the exact same committed bound generation idempotently.
    pub fn retry<Store>(
        self,
        store: &mut Store,
    ) -> Result<
        BlindVaultReplicaOwnedDurableBoundAttemptDispatch,
        BlindVaultReplicaBoundCommitPublicationError<'a, StoreError>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore<Error = StoreError>,
    {
        self.committed.publish_for_network_send(store)
    }
}

/// Failure while composing a durable bound attempt into one owned runtime.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaBoundRuntimeError {
    /// Durable marker and supplied effect/session binding did not match.
    #[error(transparent)]
    Durable(#[from] BlindVaultReplicaDurableDispatchError),
    /// Remaining effect/session ownership was internally inconsistent.
    #[error(transparent)]
    Runtime(#[from] BlindVaultReplicaTerminalAttemptRuntimeBuildError),
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

impl fmt::Debug for BlindVaultReplicaOwnedDurableBoundAttemptDispatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaOwnedDurableBoundAttemptDispatch")
            .field("attempt", &self.attempt())
            .field("snapshot_sequence", &self.snapshot_sequence())
            .field("journal_sequence", &self.journal_sequence())
            .field("work_id", &"<redacted>")
            .field("effect_set_commitment", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl<StoreError> fmt::Debug for BlindVaultReplicaBoundCommitPublicationError<'_, StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaBoundCommitPublicationError")
            .field("attempt", &self.committed.attempt())
            .field("committed_state", &"<retained>")
            .field("store_error", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl<StoreError> fmt::Display for BlindVaultReplicaBoundCommitPublicationError<'_, StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("blind vault replica committed bound publication failed")
    }
}

impl<StoreError> std::error::Error for BlindVaultReplicaBoundCommitPublicationError<'_, StoreError>
where
    StoreError: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
