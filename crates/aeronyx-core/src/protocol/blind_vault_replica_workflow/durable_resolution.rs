// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/durable_resolution.rs
// ============================================
//! Atomic durable resolution of one committed private replica attempt.
//!
//! ## Creation Reason
//! Persist-before-send ordering leaves a committed private journal until
//! authenticated terminal evidence or one bounded failure is accepted.
//! Updating only in-memory state would lose that transition on restart and
//! either replay ambiguous work or retain the journal forever.
//!
//! ## Main Functionality
//! - Defines an opaque exact binding for one committed attempt journal.
//! - Derives bindings from live durable-send permits and restored journals.
//! - Accepts verified evidence and seals the resulting workflow snapshot.
//! - Accepts typed completed-replacement capabilities without reopening proof.
//! - Accepts typed observation-retry completion capabilities.
//! - Accepts typed exact-generation renewal completion capabilities.
//! - Accepts typed aggregate-provisioning completion capabilities.
//! - Accepts typed inventory-reconciliation completion capabilities.
//! - Unifies all policy-issued capabilities under one closed action enum.
//! - Unifies verified completion and bounded failure under one resolution enum.
//! - Distills detailed terminal runtime errors into bounded attempt failures.
//! - Derives overflow-safe retry boundaries only for retryable failures.
//! - Records bounded failure and retry disposition through the same boundary.
//! - Resolves the exact journal and snapshot in one recovery-store operation.
//! - Reconciles ambiguous store errors through exact idempotent confirmation.
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
//! 3. Apply verified evidence or bounded failure to the in-memory workflow.
//! 4. Seal its post-resolution snapshot at a new monotonic sequence.
//! 5. Atomically persist that snapshot and remove only the exact journal.
//! 6. Roll memory back if any pre-resolution durability step fails.
//!
//! ## Important Note For The Next Developer
//! - A binding is local durability evidence, not terminal authorization.
//! - New adapters must prefer attempt-bound typed completion entry points.
//! - Never expose journal commitments, work identity, or targets in telemetry.
//! - Store success is the only point at which attempt resolution is durable.
//! - A failed store call retries the same exact local transition once; stores
//!   must re-confirm file/database durability before idempotent success.
//! - Do not split evidence acceptance and store resolution in new callers.
//!
//! Last Modified: v1.14.0-AmbiguousResolutionReconciliation - Replayed the
//! exact local transition once to confirm durability after ambiguous errors.
//! v1.13.0-OwnedResolutionBinding - Allowed owned durable send
//! permits to retain and derive the exact committed journal binding.
//! v1.12.0-RetryBoundaryDerivation - Added checked retry-delay
//! derivation that leaves permanent outcomes without invented schedules.
//! v1.11.0-TerminalFailureDistillation - Added the standard
//! detailed-runtime-error to bounded-attempt-failure conversion.
//! v1.10.0-UnifiedAttemptResolution - Added one closed adapter
//! outcome spanning verified completion and bounded failure.
//! v1.9.0-ReplyOutcomeConversion - Added idiomatic terminal
//! extraction from every action-specific reply outcome.
//! v1.8.0-CompletionBindingGate - Required every typed
//! completion to match the exact committed work id and attempt before mutation.
//! v1.7.0-UnifiedCompletedAction - Added one closed capability
//! enum and durable entry point spanning every planner action.
//! v1.6.0-DurableRenewalCompletion - Added a typed durable
//! boundary for exact-generation live lease renewal completion.
//! v1.5.0-DurableObservationCompletion - Added a typed durable
//! boundary for completed live inventory observation retries.
//! v1.4.0-DurableReconciliationCompletion - Added a typed
//! durable boundary for completed write/delete/inventory reconciliation.
//! v1.3.0-DurableProvisioningCompletion - Added a typed durable
//! boundary for complete aggregate provisioning reply policies.
//! v1.2.0-DurableReplacementCompletion - Added a typed durable
//! boundary for evidence emitted by the complete replacement reply policy.
//! v1.1.0-DurableFailureResolution - Added rollback-safe,
//! atomic failure recording and shared transition persistence.
//! v1.0.0-DurableAttemptResolution - Initial exact,
//! rollback-safe evidence resolution command.
//! ============================================

use std::{error::Error, fmt};

use zeroize::{Zeroize, Zeroizing};

use super::{
    persistence::sealed_record_commitment, BlindVaultReplicaActionEvidence,
    BlindVaultReplicaAttemptJournal, BlindVaultReplicaCompletedObservation,
    BlindVaultReplicaCompletedProvisioning, BlindVaultReplicaCompletedReconciliation,
    BlindVaultReplicaCompletedRenewal, BlindVaultReplicaCompletedReplacement,
    BlindVaultReplicaDispatchFailure, BlindVaultReplicaDurableAttemptDispatch,
    BlindVaultReplicaExecution, BlindVaultReplicaObservationReplyOutcome,
    BlindVaultReplicaOwnedDurableAttemptDispatch, BlindVaultReplicaPreparedAttemptJournal,
    BlindVaultReplicaProvisioningReplyOutcome, BlindVaultReplicaReconcileReplyOutcome,
    BlindVaultReplicaRecoveryStore, BlindVaultReplicaRenewalReplyOutcome,
    BlindVaultReplicaReplacementReplyOutcome, BlindVaultReplicaSnapshotRecord,
    BlindVaultReplicaTerminalAttemptError, BlindVaultReplicaTerminalVerificationFailure,
    BlindVaultReplicaWorkId, BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
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

/// Source-time and bounded disposition for one completed attempt failure.
///
/// The workflow remains the authority that validates timestamp ordering,
/// retryability, attempt exhaustion, and backoff. This value only keeps those
/// related inputs together across the durability boundary.
///
/// [BLIND-VAULT-ATTEMPT-FAILURE 2026-08-29 by Codex]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaAttemptFailure {
    failed_at_ms: u64,
    retry_not_before_ms: u64,
    failure: BlindVaultReplicaDispatchFailure,
}

impl BlindVaultReplicaAttemptFailure {
    /// Creates one candidate failure outcome for workflow validation.
    #[must_use]
    pub const fn new(
        failed_at_ms: u64,
        retry_not_before_ms: u64,
        failure: BlindVaultReplicaDispatchFailure,
    ) -> Self {
        Self {
            failed_at_ms,
            retry_not_before_ms,
            failure,
        }
    }

    /// Distills one detailed runtime error into durable privacy-safe state.
    ///
    /// [BLIND-VAULT-TERMINAL-FAILURE-DISTILLATION 2026-08-30 by Codex]
    /// Transport and verifier details remain source-local. The workflow still
    /// validates event ordering, retry backoff, and attempt exhaustion when
    /// this value crosses the atomic durable-resolution boundary.
    #[must_use]
    pub fn from_terminal_error<TransportError, VerificationError>(
        failed_at_ms: u64,
        retry_not_before_ms: u64,
        error: &BlindVaultReplicaTerminalAttemptError<TransportError, VerificationError>,
    ) -> Self
    where
        VerificationError: BlindVaultReplicaTerminalVerificationFailure,
    {
        Self::new(failed_at_ms, retry_not_before_ms, error.dispatch_failure())
    }

    /// Distills one runtime error and safely derives its retry boundary.
    ///
    /// Retryable failures use checked source-time addition. Permanent failures
    /// ignore the supplied delay and retain `failed_at_ms`, matching workflow
    /// semantics without requiring adapters to invent an unused schedule.
    pub fn from_terminal_error_with_retry_delay<TransportError, VerificationError>(
        failed_at_ms: u64,
        retry_delay_ms: u64,
        error: &BlindVaultReplicaTerminalAttemptError<TransportError, VerificationError>,
    ) -> Result<Self, BlindVaultReplicaWorkflowError>
    where
        VerificationError: BlindVaultReplicaTerminalVerificationFailure,
    {
        let failure = error.dispatch_failure();
        let retry_not_before_ms = if failure.is_retryable() {
            failed_at_ms
                .checked_add(retry_delay_ms)
                .ok_or(BlindVaultReplicaWorkflowError::TimestampOutOfRange)?
        } else {
            failed_at_ms
        };
        Ok(Self::new(failed_at_ms, retry_not_before_ms, failure))
    }

    /// Source time at which the attempt stopped awaiting evidence.
    #[must_use]
    pub const fn failed_at_ms(self) -> u64 {
        self.failed_at_ms
    }

    /// Earliest source time for a retry, when the outcome remains retryable.
    #[must_use]
    pub const fn retry_not_before_ms(self) -> u64 {
        self.retry_not_before_ms
    }

    /// Coarse privacy-safe failure persisted in workflow state.
    #[must_use]
    pub const fn failure(self) -> BlindVaultReplicaDispatchFailure {
        self.failure
    }
}

/// Durable result after an attempt outcome and journal resolution become safe.
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

/// Closed set of policy-issued capabilities that may complete planner work.
///
/// [BLIND-VAULT-COMPLETED-ACTION 2026-08-30 by Codex] Every variant carries
/// a capability whose fields are private and whose constructor is owned by its
/// exact reply policy. Adapters can unify persistence without accepting raw
/// receipts, caller-selected action labels, or unverified evidence.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaCompletedAction {
    /// Exact-generation lease renewal completed with a live new lease.
    Renewal(BlindVaultReplicaCompletedRenewal),
    /// Fresh live inventory restored observation of one replica.
    Observation(BlindVaultReplicaCompletedObservation),
    /// Ordered repair mutations ended in matching live inventory.
    Reconciliation(BlindVaultReplicaCompletedReconciliation),
    /// Distinct replacement became live before authorized old-lease retirement.
    Replacement(BlindVaultReplicaCompletedReplacement),
    /// Exact planner-authorized replica count became independently live.
    Provisioning(BlindVaultReplicaCompletedProvisioning),
}

impl BlindVaultReplicaCompletedAction {
    fn evidence(&self) -> &BlindVaultReplicaActionEvidence {
        match self {
            Self::Renewal(completed) => completed.evidence(),
            Self::Observation(completed) => completed.evidence(),
            Self::Reconciliation(completed) => completed.evidence(),
            Self::Replacement(completed) => completed.evidence(),
            Self::Provisioning(completed) => completed.evidence(),
        }
    }

    fn matches_attempt(&self, work_id: BlindVaultReplicaWorkId, attempt: u8) -> bool {
        match self {
            Self::Renewal(completed) => completed.matches_attempt(work_id, attempt),
            Self::Observation(completed) => completed.matches_attempt(work_id, attempt),
            Self::Reconciliation(completed) => completed.matches_attempt(work_id, attempt),
            Self::Replacement(completed) => completed.matches_attempt(work_id, attempt),
            Self::Provisioning(completed) => completed.matches_attempt(work_id, attempt),
        }
    }
}

impl fmt::Debug for BlindVaultReplicaCompletedAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Renewal(_) => "Renewal([REDACTED])",
            Self::Observation(_) => "Observation([REDACTED])",
            Self::Reconciliation(_) => "Reconciliation([REDACTED])",
            Self::Replacement(_) => "Replacement([REDACTED])",
            Self::Provisioning(_) => "Provisioning([REDACTED])",
        })
    }
}

impl From<BlindVaultReplicaCompletedRenewal> for BlindVaultReplicaCompletedAction {
    fn from(completed: BlindVaultReplicaCompletedRenewal) -> Self {
        Self::Renewal(completed)
    }
}

impl From<BlindVaultReplicaCompletedObservation> for BlindVaultReplicaCompletedAction {
    fn from(completed: BlindVaultReplicaCompletedObservation) -> Self {
        Self::Observation(completed)
    }
}

impl From<BlindVaultReplicaCompletedReconciliation> for BlindVaultReplicaCompletedAction {
    fn from(completed: BlindVaultReplicaCompletedReconciliation) -> Self {
        Self::Reconciliation(completed)
    }
}

impl From<BlindVaultReplicaCompletedReplacement> for BlindVaultReplicaCompletedAction {
    fn from(completed: BlindVaultReplicaCompletedReplacement) -> Self {
        Self::Replacement(completed)
    }
}

impl From<BlindVaultReplicaCompletedProvisioning> for BlindVaultReplicaCompletedAction {
    fn from(completed: BlindVaultReplicaCompletedProvisioning) -> Self {
        Self::Provisioning(completed)
    }
}

impl From<BlindVaultReplicaRenewalReplyOutcome> for BlindVaultReplicaCompletedAction {
    fn from(outcome: BlindVaultReplicaRenewalReplyOutcome) -> Self {
        let BlindVaultReplicaRenewalReplyOutcome::RenewalCompleted(completed) = outcome;
        Self::Renewal(completed)
    }
}

impl From<BlindVaultReplicaObservationReplyOutcome> for BlindVaultReplicaCompletedAction {
    fn from(outcome: BlindVaultReplicaObservationReplyOutcome) -> Self {
        let BlindVaultReplicaObservationReplyOutcome::ObservationCompleted(completed) = outcome;
        Self::Observation(completed)
    }
}

impl TryFrom<BlindVaultReplicaReconcileReplyOutcome> for BlindVaultReplicaCompletedAction {
    type Error = BlindVaultReplicaReconcileReplyOutcome;

    fn try_from(outcome: BlindVaultReplicaReconcileReplyOutcome) -> Result<Self, Self::Error> {
        match outcome {
            BlindVaultReplicaReconcileReplyOutcome::ReconciliationCompleted(completed) => {
                Ok(Self::Reconciliation(completed))
            }
            incomplete => Err(incomplete),
        }
    }
}

impl TryFrom<BlindVaultReplicaReplacementReplyOutcome> for BlindVaultReplicaCompletedAction {
    type Error = BlindVaultReplicaReplacementReplyOutcome;

    fn try_from(outcome: BlindVaultReplicaReplacementReplyOutcome) -> Result<Self, Self::Error> {
        match outcome {
            BlindVaultReplicaReplacementReplyOutcome::ReplacementCompleted(completed) => {
                Ok(Self::Replacement(completed))
            }
            incomplete => Err(incomplete),
        }
    }
}

impl TryFrom<BlindVaultReplicaProvisioningReplyOutcome> for BlindVaultReplicaCompletedAction {
    type Error = BlindVaultReplicaProvisioningReplyOutcome;

    fn try_from(outcome: BlindVaultReplicaProvisioningReplyOutcome) -> Result<Self, Self::Error> {
        match outcome {
            BlindVaultReplicaProvisioningReplyOutcome::ProvisioningCompleted(completed) => {
                Ok(Self::Provisioning(completed))
            }
            incomplete => Err(incomplete),
        }
    }
}

/// Closed terminal resolution accepted by the durable adapter boundary.
///
/// [BLIND-VAULT-ATTEMPT-RESOLUTION 2026-08-30 by Codex] Successful variants
/// can contain only reply-policy-issued completion capabilities. Failure
/// variants retain the source-owned timestamp and retry disposition that the
/// workflow validates before any snapshot or journal mutation is persisted.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaAttemptResolution {
    /// Exact authenticated reply policy completed the planner action.
    Completed(BlindVaultReplicaCompletedAction),
    /// Attempt ended with one bounded privacy-safe failure disposition.
    Failed(BlindVaultReplicaAttemptFailure),
}

impl BlindVaultReplicaAttemptResolution {
    /// Creates a bounded failed resolution from one detailed runtime error.
    pub fn failed_from_terminal_error<TransportError, VerificationError>(
        failed_at_ms: u64,
        retry_delay_ms: u64,
        error: &BlindVaultReplicaTerminalAttemptError<TransportError, VerificationError>,
    ) -> Result<Self, BlindVaultReplicaWorkflowError>
    where
        VerificationError: BlindVaultReplicaTerminalVerificationFailure,
    {
        BlindVaultReplicaAttemptFailure::from_terminal_error_with_retry_delay(
            failed_at_ms,
            retry_delay_ms,
            error,
        )
        .map(Self::Failed)
    }
}

impl fmt::Debug for BlindVaultReplicaAttemptResolution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed(completed) => {
                formatter.debug_tuple("Completed").field(completed).finish()
            }
            Self::Failed(_) => formatter.write_str("Failed([REDACTED])"),
        }
    }
}

impl From<BlindVaultReplicaCompletedAction> for BlindVaultReplicaAttemptResolution {
    fn from(completed: BlindVaultReplicaCompletedAction) -> Self {
        Self::Completed(completed)
    }
}

impl From<BlindVaultReplicaAttemptFailure> for BlindVaultReplicaAttemptResolution {
    fn from(failure: BlindVaultReplicaAttemptFailure) -> Self {
        Self::Failed(failure)
    }
}

/// Fail-closed errors before one attempt resolution becomes durable.
pub enum BlindVaultReplicaDurableResolutionError<StoreError> {
    /// Outcome transition or snapshot sealing violated workflow semantics.
    Workflow(BlindVaultReplicaWorkflowError),
    /// Durable store did not confirm exact atomic resolution.
    Store(StoreError),
    /// Both resolution and its exact idempotent confirmation failed.
    StoreOutcomeUnknown {
        resolution: StoreError,
        confirmation: StoreError,
    },
    /// Binding and current in-memory attempt did not match exactly.
    StateMismatch,
    /// Typed completion was produced by another work item or attempt.
    CompletionBindingMismatch,
}

impl<StoreError> fmt::Display for BlindVaultReplicaDurableResolutionError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
            Self::Store(_) => {
                formatter.write_str("blind vault replica durable resolution store failed")
            }
            Self::StoreOutcomeUnknown { .. } => {
                formatter.write_str("blind vault replica durable resolution outcome is unknown")
            }
            Self::StateMismatch => {
                formatter.write_str("blind vault replica resolution state does not match")
            }
            Self::CompletionBindingMismatch => {
                formatter.write_str("blind vault replica completion binding does not match attempt")
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
            Self::StoreOutcomeUnknown { resolution, .. } => Some(resolution),
            Self::StateMismatch | Self::CompletionBindingMismatch => None,
        }
    }
}

impl<StoreError> fmt::Debug for BlindVaultReplicaDurableResolutionError<StoreError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(_) => formatter.write_str("Workflow(<redacted>)"),
            Self::Store(_) => formatter.write_str("Store(<redacted>)"),
            Self::StoreOutcomeUnknown { .. } => {
                formatter.write_str("StoreOutcomeUnknown(<redacted>)")
            }
            Self::StateMismatch => formatter.write_str("StateMismatch"),
            Self::CompletionBindingMismatch => formatter.write_str("CompletionBindingMismatch"),
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

impl BlindVaultReplicaOwnedDurableAttemptDispatch {
    /// Captures the exact journal binding from owned durable send admission.
    #[must_use]
    pub fn committed_attempt_binding(&self) -> BlindVaultReplicaCommittedAttemptBinding {
        BlindVaultReplicaCommittedAttemptBinding {
            work_id: self.work_id(),
            attempt: self.attempt(),
            journal_sequence: self.journal_sequence(),
            journal_commitment: self.journal_commitment(),
        }
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
    /// Atomically resolves one committed attempt through a single adapter API.
    ///
    /// Completion remains capability-gated and attempt-bound. Failure remains
    /// workflow-validated for time ordering, retryability, exhaustion, and
    /// backoff. Both paths seal the resulting snapshot and resolve only the
    /// exact committed journal in one recovery-store operation.
    pub fn resolve_attempt_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        resolution: &BlindVaultReplicaAttemptResolution,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        match resolution {
            BlindVaultReplicaAttemptResolution::Completed(completed) => self
                .accept_completed_action_durably(
                    identity,
                    store,
                    binding,
                    completed,
                    snapshot_sequence,
                ),
            BlindVaultReplicaAttemptResolution::Failed(failure) => {
                self.record_failure_durably(identity, store, binding, *failure, snapshot_sequence)
            }
        }
    }

    /// Atomically resolves any reply-policy-issued completed action.
    ///
    /// This is the preferred generic adapter boundary. Action-specific methods
    /// remain available for source compatibility and narrower integrations.
    pub fn accept_completed_action_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completed: &BlindVaultReplicaCompletedAction,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.accept_attempt_bound_evidence_durably(
            identity,
            store,
            binding,
            completed.matches_attempt(binding.work_id, binding.attempt),
            completed.evidence(),
            snapshot_sequence,
        )
    }

    /// Atomically resolves one fully verified lease-renewal attempt.
    ///
    /// [BLIND-VAULT-DURABLE-RENEWAL-COMPLETION 2026-08-30 by Codex] The
    /// capability is constructible only after an exact node, lease, and prior
    /// expiry compare-and-swap renewal produces a still-live new lease.
    pub fn accept_completed_renewal_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completed: &BlindVaultReplicaCompletedRenewal,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.accept_attempt_bound_evidence_durably(
            identity,
            store,
            binding,
            completed.matches_attempt(binding.work_id, binding.attempt),
            completed.evidence(),
            snapshot_sequence,
        )
    }

    /// Atomically resolves one fully verified observation-retry attempt.
    ///
    /// [BLIND-VAULT-DURABLE-OBSERVATION-COMPLETION 2026-08-30 by Codex] The
    /// capability is constructible only after fresh live inventory verifies
    /// for the exact planner target. Valid divergence remains preserved in the
    /// evidence and will be classified by the mandatory fresh replan.
    pub fn accept_completed_observation_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completed: &BlindVaultReplicaCompletedObservation,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.accept_attempt_bound_evidence_durably(
            identity,
            store,
            binding,
            completed.matches_attempt(binding.work_id, binding.attempt),
            completed.evidence(),
            snapshot_sequence,
        )
    }

    /// Atomically resolves one fully verified inventory reconciliation attempt.
    ///
    /// [BLIND-VAULT-DURABLE-RECONCILIATION-COMPLETION 2026-08-30 by Codex]
    /// The capability is constructible only after ordered writes and deletes
    /// complete and one fresh terminal inventory matches the exact private
    /// manifest without predating any accepted mutation.
    pub fn accept_completed_reconciliation_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completed: &BlindVaultReplicaCompletedReconciliation,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.accept_attempt_bound_evidence_durably(
            identity,
            store,
            binding,
            completed.matches_attempt(binding.work_id, binding.attempt),
            completed.evidence(),
            snapshot_sequence,
        )
    }

    /// Atomically resolves one fully verified aggregate provisioning attempt.
    ///
    /// [BLIND-VAULT-DURABLE-PROVISIONING-COMPLETION 2026-08-30 by Codex] The
    /// capability is constructible only after the exact planner count completes
    /// ordered admission, writes, and matching live inventory verification
    /// without reusing any node or lease identity.
    pub fn accept_completed_provisioning_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completed: &BlindVaultReplicaCompletedProvisioning,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.accept_attempt_bound_evidence_durably(
            identity,
            store,
            binding,
            completed.matches_attempt(binding.work_id, binding.attempt),
            completed.evidence(),
            snapshot_sequence,
        )
    }

    /// Atomically resolves one fully verified replacement attempt.
    ///
    /// [BLIND-VAULT-DURABLE-REPLACEMENT-COMPLETION 2026-08-29 by Codex] The
    /// capability is constructible only after the replacement reply policy
    /// verifies admission, complete inventory, workflow retirement authority,
    /// and the old terminal's exact signed retirement receipt.
    pub fn accept_completed_replacement_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completed: &BlindVaultReplicaCompletedReplacement,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.accept_attempt_bound_evidence_durably(
            identity,
            store,
            binding,
            completed.matches_attempt(binding.work_id, binding.attempt),
            completed.evidence(),
            snapshot_sequence,
        )
    }

    fn accept_attempt_bound_evidence_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        completion_matches_binding: bool,
        evidence: &BlindVaultReplicaActionEvidence,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        if !completion_matches_binding {
            return Err(BlindVaultReplicaDurableResolutionError::CompletionBindingMismatch);
        }
        self.accept_evidence_durably(identity, store, binding, evidence, snapshot_sequence)
    }

    /// Low-level compatibility path for already verified action evidence.
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
        self.resolve_committed_attempt_durably(
            identity,
            store,
            binding,
            snapshot_sequence,
            |execution, work_id, attempt| {
                execution.accept_committed_attempt_evidence(work_id, attempt, evidence)
            },
        )
    }

    /// Records one bounded failure and atomically resolves its private journal.
    ///
    /// [BLIND-VAULT-DURABLE-FAILURE-RESOLUTION 2026-08-29 by Codex] A
    /// terminal operation may already have happened even when its reply is a
    /// failure or cannot be accepted. Persisting the typed workflow failure
    /// and deleting the exact committed journal must therefore be one store
    /// transition; otherwise restart can replay an already resolved attempt.
    pub fn record_failure_durably<Store>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        outcome: BlindVaultReplicaAttemptFailure,
        snapshot_sequence: u64,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
    {
        self.resolve_committed_attempt_durably(
            identity,
            store,
            binding,
            snapshot_sequence,
            |execution, work_id, _attempt| {
                execution.record_failure(
                    work_id,
                    outcome.failed_at_ms,
                    outcome.retry_not_before_ms,
                    outcome.failure,
                )
            },
        )
    }

    fn resolve_committed_attempt_durably<Store, Transition>(
        &mut self,
        identity: &IdentityKeyPair,
        store: &mut Store,
        binding: &BlindVaultReplicaCommittedAttemptBinding,
        snapshot_sequence: u64,
        transition: Transition,
    ) -> Result<
        BlindVaultReplicaDurableResolution,
        BlindVaultReplicaDurableResolutionError<Store::Error>,
    >
    where
        Store: BlindVaultReplicaRecoveryStore,
        Transition: FnOnce(
            &mut BlindVaultReplicaExecution,
            BlindVaultReplicaWorkId,
            u8,
        ) -> Result<(), BlindVaultReplicaWorkflowError>,
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

        transition(self, binding.work_id, binding.attempt)
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
        if let Err(resolution_error) = store.resolve_attempt(
            &snapshot,
            binding.journal_sequence,
            binding.journal_commitment,
        ) {
            // [BLIND-VAULT-RESOLUTION-RECONCILIATION 2026-08-30 by Codex]
            // Atomic replacement may have succeeded before a later directory
            // synchronization error surfaced. Replaying the exact transition
            // lets a compliant store either finish it or re-confirm durable
            // file/database state before returning success.
            if let Err(confirmation_error) = store.resolve_attempt(
                &snapshot,
                binding.journal_sequence,
                binding.journal_commitment,
            ) {
                restore_work_state(self, binding.work_id, previous_state)?;
                return Err(
                    BlindVaultReplicaDurableResolutionError::StoreOutcomeUnknown {
                        resolution: resolution_error,
                        confirmation: confirmation_error,
                    },
                );
            }
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
