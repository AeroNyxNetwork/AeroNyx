// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow.rs
// ============================================
//! # Blind Vault Replica Execution Workflow
//!
//! ## Creation Reason
//! The replica planner deliberately returns declarative actions without giving
//! storage nodes authority to move ciphertext between replicas. A source-side
//! workflow is required to authorize those actions, bound retries, verify node
//! evidence, and require a fresh plan before declaring convergence.
//!
//! ## Main Functionality
//! - Defines bounded source-owned execution domain types.
//! - Separates cryptographic evidence verification from state transitions.
//! - Seals exact private continuation state for ambiguous mutating attempts.
//! - Requires all work items to hold evidence before replanning.
//!
//! ## Dependencies
//! - `blind_vault_replica_workflow/attempt_continuation.rs`: reply sessions.
//! - `blind_vault_replica_workflow/attempt_journal.rs`: private attempt state.
//! - `blind_vault_replica_workflow/bound_continuation.rs`: exact effects/sessions.
//! - `blind_vault_replica_workflow/bound_dispatch.rs`: durable send markers.
//! - `blind_vault_replica_workflow/bound_dispatch/attempt_runtime.rs`: replies.
//! - `blind_vault_replica_workflow/bound_dispatch/request_bound_verifier.rs`:
//!   terminal request/receipt verification and private policy composition.
//! - `blind_vault_replica_workflow/durable_dispatch.rs`: ordered durability.
//! - `blind_vault_replica_workflow/durable_resolution.rs`: durable outcomes.
//! - `blind_vault_replica_workflow/durable_snapshot.rs`: resolved snapshots.
//! - `blind_vault_replica_workflow/evidence.rs`: receipt and inventory proof.
//! - `blind_vault_replica_workflow/execution.rs`: monotonic state machine.
//! - `blind_vault_replica_workflow/persistence.rs`: durable recovery boundary.
//! - `blind_vault_replica_workflow/prepared_effect.rs`: exact send bindings.
//! - `blind_vault_replica_workflow/recovery.rs`: restart recovery decisions.
//! - `blind_vault_replica_workflow/recovery_loader.rs`: durable phase loading.
//! - `blind_vault_replica_workflow/recovered_bound_attempt.rs`: resend permit.
//! - `blind_vault_replica_workflow/sealed_local.rs`: shared local AEAD container.
//! - `blind_vault_replica_workflow/snapshot.rs`: sealed local restart state.
//! - `protocol::blind_vault`: planner actions and terminal evidence.
//! - `protocol::onion`: descriptor-authenticated route failure disposition.
//!
//! ## Main Logical Flow
//! 1. The source creates a workflow from `BlindVaultReplicaPlan`.
//! 2. The user/client explicitly authorizes each planned action.
//! 3. The client starts one attempt and executes its typed encrypted-terminal
//!    dispatch contract; compound repair/replacement work uses ordered stages.
//!    Old replica retirement additionally requires a workflow-issued permit
//!    proving a distinct replacement is live in the current attempt.
//! 4. After restart, the source opens identity-sealed state and derives typed
//!    recovery tasks before any ambiguous operation can be repeated.
//! 5. The workflow durably accepts action-matching evidence or one bounded
//!    failure, atomically resolving the exact private attempt journal.
//! 6. Fresh inventories are planned again before declaring convergence.
//!
//! ## Important Note For The Next Developer
//! - This module is source-owned state, not a public wire/storage format.
//! - Never serialize it into discovery, the public ledger, or node telemetry.
//! - Never add ciphertext, capabilities, lease keys, owner IDs, or contacts.
//! - Persist the authenticated snapshot sequence in secure monotonic storage;
//!   accepting an older sequence can replay ambiguous network work.
//! - Persist replacement/provisioning attempt journals separately and sealed;
//!   their credentials do not belong in this workflow domain model.
//! - A node receipt proves one terminal operation, not whole-set convergence.
//! - Never retire an old replica from contract order alone; obtain
//!   `BlindVaultReplacementRetirementPermit` from the active execution.
//! - Distill accepted admission replies before waiting for inventory; do not
//!   retain one-time blind credentials across terminal stages.
//!
//! Last Modified: v1.71.0-TotalRuntimeStateGate - Removed the terminal runtime
//! panic branch in favor of exhaustive typed state handling.
//! v1.70.0-PrivacySafeFailureDiagnostics - Redacted exact
//! durable failure and retry timing from standard diagnostics.
//! v1.69.0-PrivacySafeStateDiagnostics - Redacted absolute
//! source timing from work-state and dispatch-readiness diagnostics.
//! v1.68.0-PrivacySafeRecoveryDiagnostics - Redacted recovery
//! task and durable-resolution identities from standard diagnostics.
//! v1.67.0-PrivacySafeActionDiagnostics - Standardized redacted
//! clock and private policy diagnostics across replica action state machines.
//! v1.66.0-PrivacySafeBoundaryDiagnostics - Redacted generic
//! source-policy and recovery-store diagnostics.
//! v1.65.0-PrivacySafeTransportDiagnostics - Redacted generic
//! route, transport, reply, and verification adapter diagnostics.
//! v1.64.0-DispatchDurabilityConfirmation - Added bounded exact
//! confirmation across prepared, committed, and prepared-abort transitions.
//! v1.63.0-SnapshotReconciliation - Reused one sealed snapshot
//! generation to confirm ambiguous normal-state publication outcomes.
//! v1.62.0-ResolutionReconciliation - Added exact idempotent
//! durability confirmation for ambiguous atomic attempt-resolution outcomes.
//! v1.61.0-OwnedResolutionBinding - Preserved exact opaque
//! journal resolution authority across the owned durable send path.
//! v1.60.0-RecoverablePublication - Added owned durable send
//! permits whose publication failures retain exact committed retry state.
//! v1.59.0-PrivacySafeDomainDebug - Replaced topology-bearing
//! derived diagnostics with bounded redacted workflow summaries.
//! v1.58.0-RetryBoundaryDerivation - Added overflow-safe retry
//! scheduling for detailed terminal runtime failures.
//! v1.57.0-TerminalFailureDistillation - Connected detailed
//! runtime failures to bounded durable attempt outcomes.
//! v1.56.0-TerminalFailureClassification - Standardized coarse,
//! privacy-safe failure mapping across terminal runtime adapters.
//! v1.55.0-UnifiedAttemptResolution - Added one closed durable
//! adapter outcome for verified completion and bounded failure.
//! v1.54.0-ReplyOutcomeConversion - Added standard typed
//! conversion from action reply outcomes into unified completion.
//! v1.53.0-CompletionBindingGate - Rejected typed completion
//! capabilities from any other work id or attempt before durable mutation.
//! v1.52.0-AttemptBoundCompletion - Preserved exact work and
//! attempt binding inside every policy-issued completion capability.
//! v1.51.0-UnifiedCompletedAction - Added one closed typed
//! capability enum and generic durable entry point for all planner actions.
//! v1.50.0-ReplyPolicyExports - Restored flat protocol exports
//! for every typed completion and reply-policy integration type.
//! v1.49.0-SingleEffectReplyContext - Centralized fail-closed
//! work, attempt, sequence, and authorization checks for single replies.
//! v1.48.0-DurableRenewalCompletion - Added typed atomic
//! resolution for policy-issued exact-generation renewal completion.
//! v1.47.0-RenewalReplyPolicy - Added exact lease-generation,
//! single-effect renewal reply verification.
//! v1.46.0-DurableObservationCompletion - Added typed atomic
//! resolution for policy-issued observation-retry completion.
//! v1.45.0-ObservationReplyPolicy - Added exact single-effect,
//! freshness-bounded observation-retry reply verification.
//! v1.44.0-DurableReconciliationCompletion - Added typed atomic
//! resolution for policy-issued inventory reconciliation completion.
//! v1.43.0-ReconcileReplyPolicy - Added an exact-action,
//! attempt-bound write/delete/inventory reconciliation reply state machine.
//! v1.42.0-ReplacementWriteLifetime - Bound replacement write
//! receipts to their exact signed admission lifecycle.
//! v1.41.0-DurableProvisioningCompletion - Added a typed atomic
//! resolution path for policy-issued aggregate provisioning completion.
//! v1.40.0-ProvisioningReplyPolicy - Added an exact-count,
//! attempt-bound aggregate admission/write/inventory reply state machine.
//! v1.39.0-ProvisioningLeaseLifetime - Required every newly
//! provisioned replica lease to remain live at aggregate evidence creation.
//! v1.38.0-AuthorizedRetirementDispatch - Joined active
//! workflow authority, permit-gated transport, and exact reply verification.
//! v1.37.0-ReplacementLeaseLifetime - Preserved and enforced
//! the verified replacement lease window through retirement completion.
//! v1.36.0-ReplacementPermitComposition - Bound policy and
//! runtime retirement to one active-execution-issued permit capability.
//! v1.35.0-DurableReplacementCompletion - Added a typed durable
//! path accepting only an unforgeable completed-policy capability.
//! v1.34.0-ReplacementReplyPolicy - Composed request-bound
//! replies into one fail-closed replacement lifecycle state machine.
//! v1.33.0-DistilledAdmissionEvidence - Added a credential-free
//! verified admission stage for sequential replacement/provision workflows.
//! v1.32.0-RetirementTransportPermit - Enforced replacement
//! retirement permits through ordered send and verified route selection.
//! v1.31.0-DurableTerminalOutcomes - Added bounded verifier
//! failure disposition and atomic committed-journal failure resolution.
//! v1.30.0-RequestBoundReplyVerifier - Verified exact signed
//! request/receipt pairs before source-private manifest policy acceptance.
//! v1.29.0-OwnedTerminalRuntime - Added direct live/recovery
//! ownership transfer without self-referential integration state.
//! v1.28.0-TerminalAttemptRuntime - Aligned durable effects,
//! one-time reply sessions, and workload verification under one cursor.
//! v1.27.0-VerifiedOnionEffectTransport - Composed ordered
//! effects with purpose-bound verified routes and opaque envelope I/O.
//! v1.26.0-RecoveredBoundAttempt - Added committed-only restart
//! authority for exact ordered terminal effect retransmission.
//! v1.25.0-BoundDurableDispatch - Carried exact effect identity
//! through durability markers into one ordered transport capability.
//! v1.24.0-BoundAttemptContinuation - Bound exact terminal
//! effects and one-time response sessions across restart recovery.
//! v1.23.0-PreparedTerminalEffects - Added payload-blind,
//! ordered terminal-effect binding before durable dispatch authorization.
//! v1.22.0-DurableWorkflowSnapshot - Added reusable bootstrap
//! and resolved-state persistence with explicit durable success semantics.
//! v1.21.0-DurableAttemptResolution - Added rollback-safe,
//! atomic evidence acceptance and committed-journal resolution.
//! v1.20.0-AuthenticatedRecoveryLoad - Added exact-high-water,
//! phase-aware loading of durable source workflow generations.
//! v1.19.0-PreparedRecoveryAuthentication - Required identity
//! and pre-dispatch workflow authentication before prepared-journal cleanup.
//! v1.18.0-DurableDispatchPermit - Added typed ordering from
//! durable prepared state through snapshot commit to network-send permission.
//! v1.17.0-RecoveryStoreContract - Added a storage-neutral,
//! fail-closed durability contract for snapshots and private attempt journals.
//! v1.16.0-TypedAttemptContinuation - Added restart-safe
//! ownership of adapter state and exact single-use onion reply sessions.
//! v1.15.0-PreparedAttemptJournal - Added a typed
//! persist-before-dispatch binding for private mutating attempts.
//! v1.14.0-PrivateAttemptJournal - Added action-bound sealed
//! continuation state for restart-safe replacement and provisioning attempts.
//! v1.13.0-IdentitySealedLocal - Centralized identity-bound
//! local persistence cryptography for snapshots and private attempt journals.
//! v1.12.0-RestartRecoveryPlan - Classified ambiguous restored
//! attempts into read-only observation or private-journal recovery paths.
//! v1.11.0-SealedRestartSnapshot - Added identity-bound,
//! authenticated local workflow persistence with fail-closed restoration.
//! v1.10.0-SourcePlanSummary - Retained the complete bounded
//! planner summary required for fail-closed restart validation.
//! v1.9.0-ReplacementRetirementPermit - Added an evidence-backed
//! permit that gates old-lease retirement behind a verified new replica.
//! v1.8.0-ReplicaDispatchContract - Made each action's required
//! onion terminal purposes and compound-stage order explicit for adapters.
//! v1.7.0-OnionRouteFailureDisposition - Mapped bounded route
//! admission failures into retryable discovery or permanent local outcomes.
//! v1.6.0-BlindVaultDispatchReadiness - Added one typed,
//! side-effect-free readiness contract shared with the dispatch transition.
//! v1.5.0-BlindVaultBoundedDispatch - Added source-owned global
//! dispatch limits, per-target single-flight, and ordered target dependencies.
//! v1.4.0-BlindVaultPlanShape - Reused the core plan invariant
//! at workflow creation and mandatory replan boundaries.
//! v1.3.0-BlindVaultTerminalAttemptBoundary - Removed retry
//! schedule requirements from exhausted terminal attempts.
//! v1.2.0-BlindVaultReplacementRetirement - Required verified
//! terminal retirement evidence before a replacement action can complete.
//! v1.1.0-BlindVaultFailureDisposition - Mapped authenticated
//! terminal failures into explicit retryable and permanent workflow states.
//! v1.0.0-BlindVaultReplicaWorkflow - Initial client-authorized,
//! evidence-gated replica execution state machine.
//! ============================================

mod attempt_continuation;
mod attempt_journal;
mod bound_continuation;
mod bound_dispatch;
mod durable_dispatch;
mod durable_resolution;
mod durable_snapshot;
mod evidence;
mod execution;
mod persistence;
mod prepared_effect;
mod recovered_bound_attempt;
mod recovery;
mod recovery_loader;
mod sealed_local;
mod snapshot;

pub use attempt_continuation::{
    BlindVaultReplicaAttemptContinuation, MAX_BLIND_VAULT_REPLICA_ATTEMPT_ADAPTER_STATE_BYTES,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_REPLY_SESSIONS,
};
pub use attempt_journal::{
    BlindVaultReplicaAttemptJournal, BlindVaultReplicaAttemptJournalError,
    BlindVaultReplicaAuthenticatedPreparedAttempt, BlindVaultReplicaPreparedAttemptJournal,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_BYTES,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_JOURNAL_RETENTION_MS,
    MAX_BLIND_VAULT_REPLICA_ATTEMPT_PRIVATE_STATE_BYTES,
};
pub use bound_continuation::{
    BlindVaultReplicaBoundAttemptContinuation, BlindVaultReplicaBoundContinuationError,
};
pub use bound_dispatch::{
    BlindVaultReplicaBoundCommitPublicationError, BlindVaultReplicaBoundRuntimeError,
    BlindVaultReplicaCommittedBoundAttemptDispatch, BlindVaultReplicaCompletedObservation,
    BlindVaultReplicaCompletedProvisioning, BlindVaultReplicaCompletedReconciliation,
    BlindVaultReplicaCompletedRenewal, BlindVaultReplicaCompletedReplacement,
    BlindVaultReplicaDurableBoundAttemptDispatch, BlindVaultReplicaObservationReplyOutcome,
    BlindVaultReplicaObservationReplyPolicy, BlindVaultReplicaObservationReplyPolicyBuildError,
    BlindVaultReplicaObservationReplyPolicyError, BlindVaultReplicaOnionDispatchPlan,
    BlindVaultReplicaOnionEnvelopeSender, BlindVaultReplicaOnionRouteProvider,
    BlindVaultReplicaOwnedDurableBoundAttemptDispatch,
    BlindVaultReplicaPersistedBoundAttemptJournal, BlindVaultReplicaPreparedBoundAttemptJournal,
    BlindVaultReplicaPrivateReplyPolicy, BlindVaultReplicaProvisioningReplyOutcome,
    BlindVaultReplicaProvisioningReplyPolicy, BlindVaultReplicaProvisioningReplyPolicyBuildError,
    BlindVaultReplicaProvisioningReplyPolicyError, BlindVaultReplicaReconcileReplyOutcome,
    BlindVaultReplicaReconcileReplyPolicy, BlindVaultReplicaReconcileReplyPolicyBuildError,
    BlindVaultReplicaReconcileReplyPolicyError, BlindVaultReplicaRenewalReplyOutcome,
    BlindVaultReplicaRenewalReplyPolicy, BlindVaultReplicaRenewalReplyPolicyBuildError,
    BlindVaultReplicaRenewalReplyPolicyError, BlindVaultReplicaReplacementAuthorizationError,
    BlindVaultReplicaReplacementPermitIssueError, BlindVaultReplicaReplacementReplyOutcome,
    BlindVaultReplicaReplacementReplyPolicy, BlindVaultReplicaReplacementReplyPolicyBuildError,
    BlindVaultReplicaReplacementReplyPolicyError,
    BlindVaultReplicaReplacementRetirementDispatchError, BlindVaultReplicaRequestBoundReply,
    BlindVaultReplicaRequestBoundReplyError, BlindVaultReplicaRequestBoundReplyVerifier,
    BlindVaultReplicaTerminalAttemptError, BlindVaultReplicaTerminalAttemptRuntime,
    BlindVaultReplicaTerminalAttemptRuntimeBuildError, BlindVaultReplicaTerminalAttemptState,
    BlindVaultReplicaTerminalEffectTransport, BlindVaultReplicaTerminalReplyVerifier,
    BlindVaultReplicaTerminalSendContext, BlindVaultReplicaTerminalSendError,
    BlindVaultReplicaTerminalSendSequence, BlindVaultReplicaTerminalVerificationFailure,
    BlindVaultReplicaVerificationClock, BlindVaultReplicaVerifiedOnionTransport,
    BlindVaultReplicaVerifiedOnionTransportError,
};
pub use durable_dispatch::{
    BlindVaultReplicaCommitPublicationError, BlindVaultReplicaCommittedAttemptDispatch,
    BlindVaultReplicaDurableAttemptDispatch, BlindVaultReplicaDurableDispatchError,
    BlindVaultReplicaOwnedDurableAttemptDispatch, BlindVaultReplicaPersistedAttemptJournal,
};
pub use durable_resolution::{
    BlindVaultReplicaAttemptFailure, BlindVaultReplicaAttemptResolution,
    BlindVaultReplicaCommittedAttemptBinding, BlindVaultReplicaCompletedAction,
    BlindVaultReplicaDurableResolution, BlindVaultReplicaDurableResolutionError,
};
pub use durable_snapshot::{
    BlindVaultReplicaDurableSnapshot, BlindVaultReplicaDurableSnapshotError,
};
pub use persistence::{
    BlindVaultReplicaAttemptDurabilityPhase, BlindVaultReplicaCommittedAttemptRecord,
    BlindVaultReplicaPreparedAttemptRecord, BlindVaultReplicaRecoveryState,
    BlindVaultReplicaRecoveryStore, BlindVaultReplicaSnapshotRecord,
};
pub use prepared_effect::{
    BlindVaultReplicaPreparedEffectError, BlindVaultReplicaPreparedEffectSet,
    BlindVaultReplicaTerminalEffect, MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECTS,
    MAX_BLIND_VAULT_REPLICA_TERMINAL_EFFECT_BYTES,
};
pub use recovered_bound_attempt::{
    BlindVaultReplicaRecoveredBoundAttempt, BlindVaultReplicaRecoveredBoundAttemptError,
    BlindVaultReplicaRecoveredSendPermit,
};
pub use recovery::{
    BlindVaultReplicaRestartRecoveryKind, BlindVaultReplicaRestartRecoveryTask,
    BlindVaultReplicaRestartRecoveryTiming,
};
pub use recovery_loader::{
    load_blind_vault_replica_recovery, BlindVaultReplicaLoadedRecovery,
    BlindVaultReplicaRecoveryLoadError,
};

use std::fmt;

use thiserror::Error;

use super::blind_vault::{
    BlindVaultError, BlindVaultReplicaAction, BlindVaultReplicaPlanHealth,
    BlindVaultTerminalFailureCode, MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS,
};
use super::onion::{OnionRouteFailureDisposition, OnionRoutePlanError};

/// At most two per-member actions plus one aggregate provisioning action can
/// be emitted by the current deterministic planner.
pub const MAX_BLIND_VAULT_REPLICA_WORK_ITEMS: usize = MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS;

/// Privacy-conservative default: one cross-replica operation in flight.
///
/// Apps that explicitly accept the timing-correlation and resource tradeoff
/// may choose a larger bounded value through `new_with_maximum_in_flight`.
pub const DEFAULT_BLIND_VAULT_REPLICA_MAXIMUM_IN_FLIGHT: u8 = 1;

/// Maximum identity-sealed source-local restart snapshot size.
pub const MAX_BLIND_VAULT_REPLICA_RESTART_SNAPSHOT_BYTES: usize = 64 * 1024;

/// Source-side retry and evidence timing policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaExecutionPolicy {
    /// Maximum dispatch attempts for the same immutable planned action.
    pub maximum_attempts: u8,
    /// Time allowed for terminal evidence after each dispatch.
    pub evidence_timeout_ms: u64,
}

impl BlindVaultReplicaExecutionPolicy {
    /// Creates a bounded execution policy.
    pub fn new(
        maximum_attempts: u8,
        evidence_timeout_ms: u64,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        let policy = Self {
            maximum_attempts,
            evidence_timeout_ms,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub(super) fn validate(self) -> Result<(), BlindVaultReplicaWorkflowError> {
        if self.maximum_attempts == 0 || self.evidence_timeout_ms == 0 {
            return Err(BlindVaultReplicaWorkflowError::InvalidPolicy);
        }
        Ok(())
    }
}

/// Stable source-local identity for one planned work item.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlindVaultReplicaWorkId {
    pub(super) workflow_id: [u8; 16],
    pub(super) sequence: u16,
}

impl fmt::Debug for BlindVaultReplicaWorkId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // [BLIND-VAULT-WORKFLOW-PRIVACY-SAFE-DEBUG 2026-08-30 by Codex]
        formatter
            .debug_struct("BlindVaultReplicaWorkId")
            .field("workflow_id", &"[REDACTED]")
            .field("sequence", &self.sequence)
            .finish()
    }
}

impl BlindVaultReplicaWorkId {
    /// Random workflow generation identifier supplied by the source.
    #[must_use]
    pub const fn workflow_id(&self) -> [u8; 16] {
        self.workflow_id
    }

    /// Deterministic action position within the planner output.
    #[must_use]
    pub const fn sequence(&self) -> u16 {
        self.sequence
    }
}

/// Bounded failure classes suitable for recovery decisions without persisting
/// endpoint URLs, payloads, or arbitrary server error strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaDispatchFailure {
    /// No usable route or terminal connection was available.
    TransportUnavailable,
    /// An authenticated terminal reported temporary local unavailability.
    TerminalUnavailable,
    /// The selected terminal rejected the authenticated operation.
    TerminalRejected,
    /// The terminal did not return acceptable evidence before the deadline.
    EvidenceTimeout,
    /// The source detected that this immutable plan generation became stale.
    StalePlan,
    /// The terminal lacked coarse storage capacity for the operation.
    CapacityUnavailable,
    /// Local privacy, routing, or replica policy rejected dispatch.
    PolicyRejected,
    /// Local authenticated request or onion-envelope construction failed.
    LocalConstructionFailed,
    /// The inline response class cannot carry the requested recovery result.
    InlineResponseUnsupported,
}

impl From<&OnionRoutePlanError> for BlindVaultReplicaDispatchFailure {
    /// Converts route admission into a bounded retry disposition.
    ///
    /// [ONION-ROUTE-FAILURE-DISPOSITION 2026-08-29 by Codex] Refreshable
    /// discovery surface failures remain retryable because the source may
    /// select a different current route. Structural policy violations and
    /// local construction failures block the immutable workflow generation.
    fn from(value: &OnionRoutePlanError) -> Self {
        match value.disposition() {
            OnionRouteFailureDisposition::RefreshRoute => Self::TransportUnavailable,
            OnionRouteFailureDisposition::PolicyRejected => Self::PolicyRejected,
            OnionRouteFailureDisposition::LocalConstructionFailed => Self::LocalConstructionFailed,
        }
    }
}

impl BlindVaultReplicaDispatchFailure {
    /// Whether the exact immutable action may be dispatched again.
    ///
    /// [BLIND-VAULT-FAILURE-DISPOSITION 2026-08-28 by Codex] Permanent
    /// protocol, policy, and stale-plan failures never consume repeated
    /// network attempts. Temporary transport, terminal, timeout, and capacity
    /// failures remain bounded by the workflow attempt budget.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::TransportUnavailable
                | Self::TerminalUnavailable
                | Self::EvidenceTimeout
                | Self::CapacityUnavailable
        )
    }
}

impl From<BlindVaultTerminalFailureCode> for BlindVaultReplicaDispatchFailure {
    fn from(value: BlindVaultTerminalFailureCode) -> Self {
        match value {
            BlindVaultTerminalFailureCode::Rejected => Self::TerminalRejected,
            BlindVaultTerminalFailureCode::Capacity => Self::CapacityUnavailable,
            BlindVaultTerminalFailureCode::Unavailable => Self::TerminalUnavailable,
            BlindVaultTerminalFailureCode::ResponseTooLarge => Self::InlineResponseUnsupported,
        }
    }
}

/// Monotonic state of one immutable planner action.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaWorkState {
    /// No network work is permitted until the source explicitly authorizes it.
    AwaitingAuthorization,
    /// The exact action is authorized and may be dispatched.
    Authorized { authorized_at_ms: u64 },
    /// A request was dispatched and must produce matching evidence by deadline.
    AwaitingEvidence {
        attempt: u8,
        dispatched_at_ms: u64,
        evidence_deadline_ms: u64,
    },
    /// Matching cryptographic evidence was accepted for this action.
    EvidenceAccepted { attempt: u8, verified_at_ms: u64 },
    /// The same immutable action may be retried after the bounded backoff.
    RetryableFailure {
        attempt: u8,
        failed_at_ms: u64,
        retry_not_before_ms: u64,
        failure: BlindVaultReplicaDispatchFailure,
    },
    /// A verified permanent failure blocked this immutable plan generation.
    PermanentFailure {
        attempt: u8,
        failed_at_ms: u64,
        failure: BlindVaultReplicaDispatchFailure,
    },
    /// The attempt budget is exhausted and the generation is blocked.
    Exhausted {
        attempt: u8,
        failed_at_ms: u64,
        failure: BlindVaultReplicaDispatchFailure,
    },
    /// The source explicitly cancelled the immutable action.
    Cancelled { cancelled_at_ms: u64 },
}

// [BLIND-VAULT-WORK-STATE-DIAGNOSTICS 2026-08-30 by Codex] Absolute source
// times can correlate an otherwise blind operation across process and host
// logs. Diagnostics retain the transition, attempt, and coarse failure only.
impl fmt::Debug for BlindVaultReplicaWorkState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AwaitingAuthorization => formatter.write_str("AwaitingAuthorization"),
            Self::Authorized { .. } => formatter
                .debug_struct("Authorized")
                .field("authorized_at_ms", &"<redacted>")
                .finish(),
            Self::AwaitingEvidence { attempt, .. } => formatter
                .debug_struct("AwaitingEvidence")
                .field("attempt", attempt)
                .field("source_timing", &"<redacted>")
                .finish(),
            Self::EvidenceAccepted { attempt, .. } => formatter
                .debug_struct("EvidenceAccepted")
                .field("attempt", attempt)
                .field("verified_at_ms", &"<redacted>")
                .finish(),
            Self::RetryableFailure {
                attempt, failure, ..
            } => formatter
                .debug_struct("RetryableFailure")
                .field("attempt", attempt)
                .field("source_timing", &"<redacted>")
                .field("failure", failure)
                .finish(),
            Self::PermanentFailure {
                attempt, failure, ..
            } => formatter
                .debug_struct("PermanentFailure")
                .field("attempt", attempt)
                .field("failed_at_ms", &"<redacted>")
                .field("failure", failure)
                .finish(),
            Self::Exhausted {
                attempt, failure, ..
            } => formatter
                .debug_struct("Exhausted")
                .field("attempt", attempt)
                .field("failed_at_ms", &"<redacted>")
                .field("failure", failure)
                .finish(),
            Self::Cancelled { .. } => formatter
                .debug_struct("Cancelled")
                .field("cancelled_at_ms", &"<redacted>")
                .finish(),
        }
    }
}

fn work_state_name(state: BlindVaultReplicaWorkState) -> &'static str {
    match state {
        BlindVaultReplicaWorkState::AwaitingAuthorization => "AwaitingAuthorization",
        BlindVaultReplicaWorkState::Authorized { .. } => "Authorized",
        BlindVaultReplicaWorkState::AwaitingEvidence { .. } => "AwaitingEvidence",
        BlindVaultReplicaWorkState::EvidenceAccepted { .. } => "EvidenceAccepted",
        BlindVaultReplicaWorkState::RetryableFailure { .. } => "RetryableFailure",
        BlindVaultReplicaWorkState::PermanentFailure { .. } => "PermanentFailure",
        BlindVaultReplicaWorkState::Exhausted { .. } => "Exhausted",
        BlindVaultReplicaWorkState::Cancelled { .. } => "Cancelled",
    }
}

/// One source-local action and its execution state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaWorkItem {
    pub(super) id: BlindVaultReplicaWorkId,
    pub(super) action: BlindVaultReplicaAction,
    pub(super) state: BlindVaultReplicaWorkState,
}

impl fmt::Debug for BlindVaultReplicaWorkItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // [BLIND-VAULT-WORKFLOW-PRIVACY-SAFE-DEBUG 2026-08-30 by Codex]
        // Diagnostics retain scheduling state without exposing topology,
        // source timing, workflow identity, lease identity, or commitments.
        formatter
            .debug_struct("BlindVaultReplicaWorkItem")
            .field("sequence", &self.id.sequence)
            .field("action", &self.action)
            .field("state", &work_state_name(self.state))
            .finish()
    }
}

impl BlindVaultReplicaWorkItem {
    /// Stable work identity for idempotent UI and persistence adapters.
    #[must_use]
    pub const fn id(&self) -> BlindVaultReplicaWorkId {
        self.id
    }

    /// Immutable planner action. Retries never mutate its target semantics.
    #[must_use]
    pub const fn action(&self) -> BlindVaultReplicaAction {
        self.action
    }

    /// Current fail-closed execution state.
    #[must_use]
    pub const fn state(&self) -> BlindVaultReplicaWorkState {
        self.state
    }

    /// Returns the immutable onion-purpose sequence for this action.
    #[must_use]
    pub const fn dispatch_contract(&self) -> BlindVaultReplicaDispatchContract {
        self.action.dispatch_contract()
    }
}

/// Aggregate execution phase derived from item states rather than stored as a
/// second, potentially inconsistent source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaExecutionPhase {
    /// The planner returned a healthy plan with no work.
    Converged,
    /// At least one immutable action still needs explicit authorization.
    AwaitingAuthorization,
    /// Authorized work is ready for encrypted terminal dispatch.
    ReadyToDispatch,
    /// At least one request is waiting for terminal evidence.
    AwaitingEvidence,
    /// Failed work is waiting for its retry boundary.
    RetryBackoff,
    /// Every action has evidence; fresh inventories and a new plan are due.
    AwaitingReplan,
    /// At least one action exhausted attempts or was cancelled.
    Blocked,
}

/// Side-effect-free dispatch readiness for one source-local work item.
///
/// [BLIND-VAULT-DISPATCH-READINESS 2026-08-29 by Codex] Product and agent
/// adapters can render or schedule work without invoking a mutating transition
/// merely to discover backoff, dependency, target, or capacity state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaDispatchReadiness {
    /// The exact immutable action may be dispatched now.
    Ready {
        attempt: u8,
        evidence_deadline_ms: u64,
    },
    /// The source has not explicitly authorized this action.
    AwaitingAuthorization,
    /// Retry is valid only at or after this source-local boundary.
    RetryBackoff { retry_not_before_ms: u64 },
    /// This action is already waiting for terminal evidence.
    AlreadyInFlight { evidence_deadline_ms: u64 },
    /// Another operation for the same terminal/lease is in flight.
    TargetInFlight,
    /// An earlier operation for the same terminal/lease lacks evidence.
    TargetDependencyPending,
    /// Unrelated work currently occupies the bounded dispatch capacity.
    CapacityReached { in_flight: u8, maximum: u8 },
    /// This work item is complete, cancelled, or permanently blocked.
    TerminalState,
}

// [BLIND-VAULT-DISPATCH-READINESS-DIAGNOSTICS 2026-08-30 by Codex] Preserve
// scheduling disposition without publishing exact deadline or backoff times.
impl fmt::Debug for BlindVaultReplicaDispatchReadiness {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready { attempt, .. } => formatter
                .debug_struct("Ready")
                .field("attempt", attempt)
                .field("evidence_deadline_ms", &"<redacted>")
                .finish(),
            Self::AwaitingAuthorization => formatter.write_str("AwaitingAuthorization"),
            Self::RetryBackoff { .. } => formatter
                .debug_struct("RetryBackoff")
                .field("retry_not_before_ms", &"<redacted>")
                .finish(),
            Self::AlreadyInFlight { .. } => formatter
                .debug_struct("AlreadyInFlight")
                .field("evidence_deadline_ms", &"<redacted>")
                .finish(),
            Self::TargetInFlight => formatter.write_str("TargetInFlight"),
            Self::TargetDependencyPending => formatter.write_str("TargetDependencyPending"),
            Self::CapacityReached { in_flight, maximum } => formatter
                .debug_struct("CapacityReached")
                .field("in_flight", in_flight)
                .field("maximum", maximum)
                .finish(),
            Self::TerminalState => formatter.write_str("TerminalState"),
        }
    }
}

/// Purpose-level network work required by one source-owned replica action.
///
/// [BLIND-VAULT-DISPATCH-CONTRACT 2026-08-29 by Codex] `dispatch()` opens one
/// bounded workflow attempt; it does not imply that every action is one HTTP
/// request. This contract gives Apps, SDKs, and agents the exact onion terminal
/// purposes and stage order while leaving ciphertext selection, credentials,
/// requests, and keys outside the workflow state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaDispatchContract {
    /// One terminal operation completes the action before evidence verification.
    SingleTerminalRequest {
        /// Purpose that the selected terminal's signed descriptor must support.
        purpose: OnionRoutePurpose,
    },
    /// Replay the source-owned private manifest delta, then verify inventory.
    ReconcileInventory {
        /// Used once per source-selected immutable object write.
        write_purpose: OnionRoutePurpose,
        /// Used once per source-selected surplus object deletion.
        delete_purpose: OnionRoutePurpose,
        /// Must run after all mutation receipts verify.
        verification_purpose: OnionRoutePurpose,
    },
    /// Admit, populate, and verify a new replica before retiring the old lease.
    ReplaceReplica {
        /// First stage: blind admission of one independently wrapped replica.
        admission_purpose: OnionRoutePurpose,
        /// Populate the new lease from the source-owned private manifest.
        write_purpose: OnionRoutePurpose,
        /// Prove the new terminal matches the complete private manifest.
        verification_purpose: OnionRoutePurpose,
        /// Final stage: complete retirement at the old terminal.
        retirement_purpose: OnionRoutePurpose,
    },
    /// Admit an exact bounded number of independently wrapped replicas.
    ProvisionReplicas {
        /// Purpose repeated independently for each intended replica.
        admission_purpose: OnionRoutePurpose,
        /// Populate each admitted lease from the private manifest.
        write_purpose: OnionRoutePurpose,
        /// Prove each new terminal matches its complete private manifest.
        verification_purpose: OnionRoutePurpose,
        /// Exact planner-authorized number of admissions.
        count: u8,
    },
}

impl BlindVaultReplicaAction {
    /// Derives the terminal-work contract after plan-shape validation.
    #[must_use]
    pub(super) const fn dispatch_contract(self) -> BlindVaultReplicaDispatchContract {
        match self {
            Self::RenewLease { .. } => BlindVaultReplicaDispatchContract::SingleTerminalRequest {
                purpose: OnionRoutePurpose::BlindVaultLeaseRenewal,
            },
            Self::ReconcileInventory { .. } => {
                BlindVaultReplicaDispatchContract::ReconcileInventory {
                    write_purpose: OnionRoutePurpose::BlindVaultPutReceipt,
                    delete_purpose: OnionRoutePurpose::BlindVaultDelete,
                    verification_purpose: OnionRoutePurpose::BlindVaultLeaseInventory,
                }
            }
            Self::RetryObservation { .. } => {
                BlindVaultReplicaDispatchContract::SingleTerminalRequest {
                    purpose: OnionRoutePurpose::BlindVaultLeaseInventory,
                }
            }
            Self::ReplaceReplica { .. } => BlindVaultReplicaDispatchContract::ReplaceReplica {
                admission_purpose: OnionRoutePurpose::BlindVaultLeaseAdmission,
                write_purpose: OnionRoutePurpose::BlindVaultPutReceipt,
                verification_purpose: OnionRoutePurpose::BlindVaultLeaseInventory,
                retirement_purpose: OnionRoutePurpose::BlindVaultLeaseRetire,
            },
            Self::ProvisionReplicas { count } => {
                BlindVaultReplicaDispatchContract::ProvisionReplicas {
                    admission_purpose: OnionRoutePurpose::BlindVaultLeaseAdmission,
                    write_purpose: OnionRoutePurpose::BlindVaultPutReceipt,
                    verification_purpose: OnionRoutePurpose::BlindVaultLeaseInventory,
                    count,
                }
            }
        }
    }
}

/// Outcome of comparing an evidence-complete generation with a fresh plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaConvergence {
    /// Fresh verified evidence produced a healthy plan with no actions.
    Converged,
    /// More source-authorized work is required under a new workflow generation.
    FollowUpRequired {
        health: BlindVaultReplicaPlanHealth,
        action_count: u16,
    },
}

/// Verified admission plus matching live inventory for one newly provisioned
/// replica. Private fields prevent callers from manufacturing this evidence.
///
/// [BLIND-VAULT-REPLACEMENT-LEASE-LIFETIME 2026-08-30 by Codex] The distilled
/// proof retains the signed expiry needed to prevent unsafe delayed retirement,
/// while Debug keeps replica topology identifiers redacted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedProvisionedReplica {
    pub(super) node_id: [u8; 32],
    pub(super) lease_id: [u8; 32],
    pub(super) lease_expires_at_ms: u64,
    pub(super) accepted_at_ms: u64,
    pub(super) observed_at_ms: u64,
}

// [BLIND-VAULT-EVIDENCE-DIAGNOSTICS 2026-08-30 by Codex] Verified lifecycle
// timestamps stay available through typed policy logic, but standard Debug
// must not turn them into correlation handles alongside terminal logs.
impl std::fmt::Debug for BlindVaultVerifiedProvisionedReplica {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultVerifiedProvisionedReplica")
            .field("lease_expires_at_ms", &"<redacted>")
            .field("accepted_at_ms", &"<redacted>")
            .field("observed_at_ms", &"<redacted>")
            .field("node_id", &"[REDACTED]")
            .field("lease_id", &"[REDACTED]")
            .finish()
    }
}

/// Credential-free proof that one anonymous replica admission was accepted.
///
/// [BLIND-VAULT-DISTILLED-ADMISSION 2026-08-29 by Codex] Sequential terminal
/// workflows must not retain the complete one-time blind admission credential
/// while waiting for a matching inventory receipt. This non-serializable value
/// keeps only signed lifecycle facts needed to finish replica verification.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedReplicaAdmission {
    pub(super) node_id: [u8; 32],
    pub(super) lease_id: [u8; 32],
    pub(super) lease_expires_at_ms: u64,
    pub(super) accepted_at_ms: u64,
}

impl std::fmt::Debug for BlindVaultVerifiedReplicaAdmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultVerifiedReplicaAdmission")
            .field("lease_expires_at_ms", &"<redacted>")
            .field("accepted_at_ms", &"<redacted>")
            .field("node_id", &"[REDACTED]")
            .field("lease_id", &"[REDACTED]")
            .finish()
    }
}

impl BlindVaultVerifiedReplicaAdmission {
    /// Descriptor identity that signed the admission receipt.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Newly admitted replica-local lease.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Exact lease expiry accepted by the terminal.
    #[must_use]
    pub const fn lease_expires_at_ms(&self) -> u64 {
        self.lease_expires_at_ms
    }

    /// Terminal-signed admission time.
    #[must_use]
    pub const fn accepted_at_ms(&self) -> u64 {
        self.accepted_at_ms
    }
}

/// Verified terminal retirement of one complete old replica lease.
///
/// Private fields prevent a caller from marking replacement complete using a
/// locally invented node or lease identifier. Construction is available only
/// through the receipt-verifying implementation in `evidence.rs`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedRetiredReplica {
    pub(super) node_id: [u8; 32],
    pub(super) lease_id: [u8; 32],
    pub(super) retired_at_ms: u64,
}

impl std::fmt::Debug for BlindVaultVerifiedRetiredReplica {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultVerifiedRetiredReplica")
            .field("retired_at_ms", &"<redacted>")
            .field("node_id", &"[REDACTED]")
            .field("lease_id", &"[REDACTED]")
            .finish()
    }
}

/// Evidence-backed permission to retire one replaced replica.
///
/// [BLIND-VAULT-REPLACEMENT-RETIREMENT-PERMIT 2026-08-29 by Codex] This value
/// can only be issued by an active workflow attempt after a distinct new
/// replica has produced a matching live inventory. It carries no ciphertext,
/// object identifier, manifest, credential, route, endpoint, or user identity.
///
/// [BLIND-VAULT-REPLACEMENT-LEASE-LIFETIME 2026-08-30 by Codex] The permit
/// carries the replacement expiry needed for fail-closed retries and redacts
/// all topology identifiers from Debug output.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplacementRetirementPermit {
    pub(super) work_id: BlindVaultReplicaWorkId,
    pub(super) attempt: u8,
    pub(super) replaced_node_id: [u8; 32],
    pub(super) replaced_lease_id: [u8; 32],
    pub(super) replacement_node_id: [u8; 32],
    pub(super) replacement_lease_id: [u8; 32],
    pub(super) replacement_expires_at_ms: u64,
    pub(super) authorized_at_ms: u64,
}

impl std::fmt::Debug for BlindVaultReplacementRetirementPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BlindVaultReplacementRetirementPermit")
            .field("attempt", &self.attempt)
            .field("replacement_expires_at_ms", &"<redacted>")
            .field("authorized_at_ms", &"<redacted>")
            .field("work_id", &"[REDACTED]")
            .field("replaced_node_id", &"[REDACTED]")
            .field("replaced_lease_id", &"[REDACTED]")
            .field("replacement_node_id", &"[REDACTED]")
            .field("replacement_lease_id", &"[REDACTED]")
            .finish()
    }
}

impl BlindVaultReplacementRetirementPermit {
    /// Workflow work item that authorized this retirement transition.
    #[must_use]
    pub const fn work_id(&self) -> BlindVaultReplicaWorkId {
        self.work_id
    }

    /// Active bounded attempt that admitted the verified replacement.
    #[must_use]
    pub const fn attempt(&self) -> u8 {
        self.attempt
    }

    /// Existing terminal descriptor identity authorized for retirement.
    #[must_use]
    pub const fn replaced_node_id(&self) -> [u8; 32] {
        self.replaced_node_id
    }

    /// Existing replica-local lease authorized for retirement.
    #[must_use]
    pub const fn replaced_lease_id(&self) -> [u8; 32] {
        self.replaced_lease_id
    }

    /// Distinct terminal that proved the live replacement.
    #[must_use]
    pub const fn replacement_node_id(&self) -> [u8; 32] {
        self.replacement_node_id
    }

    /// Distinct replica-local lease that proved the live replacement.
    #[must_use]
    pub const fn replacement_lease_id(&self) -> [u8; 32] {
        self.replacement_lease_id
    }

    /// Terminal-signed expiry bounding safe retirement and transport retry.
    #[must_use]
    pub const fn replacement_expires_at_ms(&self) -> u64 {
        self.replacement_expires_at_ms
    }

    /// Source time at which the active attempt authorized retirement.
    #[must_use]
    pub const fn authorized_at_ms(&self) -> u64 {
        self.authorized_at_ms
    }
}

impl BlindVaultVerifiedRetiredReplica {
    /// Descriptor identity that signed the complete retirement receipt.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Replica-local lease proven retired by the terminal.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Terminal-signed durable retirement time.
    #[must_use]
    pub const fn retired_at_ms(&self) -> u64 {
        self.retired_at_ms
    }
}

impl BlindVaultVerifiedProvisionedReplica {
    /// Newly provisioned terminal descriptor identity.
    #[must_use]
    pub const fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Newly provisioned replica-local lease.
    #[must_use]
    pub const fn lease_id(&self) -> [u8; 32] {
        self.lease_id
    }

    /// Terminal-signed lease expiry proven by matching live inventory.
    #[must_use]
    pub const fn lease_expires_at_ms(&self) -> u64 {
        self.lease_expires_at_ms
    }

    /// Terminal-signed admission time.
    #[must_use]
    pub const fn accepted_at_ms(&self) -> u64 {
        self.accepted_at_ms
    }

    /// Terminal-signed matching inventory time.
    #[must_use]
    pub const fn observed_at_ms(&self) -> u64 {
        self.observed_at_ms
    }
}

/// Action-specific evidence with private variants so only verification
/// constructors can produce an acceptable value.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaActionEvidence {
    pub(super) kind: BlindVaultReplicaActionEvidenceKind,
    pub(super) verified_at_ms: u64,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) enum BlindVaultReplicaActionEvidenceKind {
    LeaseRenewed {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        previous_expires_at_ms: u64,
    },
    InventoryReconciled {
        node_id: [u8; 32],
        lease_id: [u8; 32],
        expected_object_count: u64,
        expected_ciphertext_bytes: u64,
        expected_inventory_commitment: [u8; 32],
    },
    ObservationRecovered {
        node_id: [u8; 32],
        lease_id: [u8; 32],
    },
    ReplicaReplaced {
        replaced_node_id: [u8; 32],
        replaced_lease_id: [u8; 32],
    },
    ReplicaReplacedWithPermit {
        work_id: BlindVaultReplicaWorkId,
        attempt: u8,
        replaced_node_id: [u8; 32],
        replaced_lease_id: [u8; 32],
    },
    ReplicasProvisioned {
        count: u8,
    },
}

impl fmt::Debug for BlindVaultReplicaActionEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // [BLIND-VAULT-WORKFLOW-PRIVACY-SAFE-DEBUG 2026-08-30 by Codex]
        formatter
            .debug_struct("BlindVaultReplicaActionEvidence")
            .field("kind", &self.kind)
            .field("verified", &true)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BlindVaultReplicaActionEvidenceKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::LeaseRenewed { .. } => "LeaseRenewed",
            Self::InventoryReconciled { .. } => "InventoryReconciled",
            Self::ObservationRecovered { .. } => "ObservationRecovered",
            Self::ReplicaReplaced { .. } => "ReplicaReplaced",
            Self::ReplicaReplacedWithPermit { .. } => "ReplicaReplacedWithPermit",
            Self::ReplicasProvisioned { .. } => "ReplicasProvisioned",
        })
    }
}

/// Client-owned execution state for one immutable planner generation.
///
/// [BLIND-VAULT-REPLICA-WORKFLOW 2026-08-28 by Codex] The workflow stores no
/// cross-replica repair material. The App remains responsible for independently
/// wrapping data and constructing each encrypted terminal request.
#[derive(PartialEq, Eq)]
pub struct BlindVaultReplicaExecution {
    pub(super) workflow_id: [u8; 16],
    pub(super) created_at_ms: u64,
    pub(super) source_plan_health: BlindVaultReplicaPlanHealth,
    pub(super) source_configured_replicas: u8,
    pub(super) source_live_verified_replicas: u8,
    pub(super) source_live_matching_replicas: u8,
    pub(super) policy: BlindVaultReplicaExecutionPolicy,
    pub(super) maximum_in_flight: u8,
    pub(super) items: Vec<BlindVaultReplicaWorkItem>,
}

impl fmt::Debug for BlindVaultReplicaExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // [BLIND-VAULT-WORKFLOW-PRIVACY-SAFE-DEBUG 2026-08-30 by Codex]
        formatter
            .debug_struct("BlindVaultReplicaExecution")
            .field("phase", &self.phase())
            .field("plan_health", &self.source_plan_health)
            .field("configured_replicas", &self.source_configured_replicas)
            .field(
                "live_verified_replicas",
                &self.source_live_verified_replicas,
            )
            .field(
                "live_matching_replicas",
                &self.source_live_matching_replicas,
            )
            .field("item_count", &self.items.len())
            .field("in_flight_count", &self.in_flight_count())
            .field("workflow", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

/// Authenticated local workflow state recovered after restart.
///
/// [BLIND-VAULT-RESTART-ROLLBACK-GUARD 2026-08-29 by Codex] The persistence
/// adapter must advance its separately protected high-water mark to
/// `snapshot_sequence` after accepting this value. Keeping the sequence out of
/// `BlindVaultReplicaExecution` avoids confusing persistence order with the
/// immutable planner generation or network protocol state.
#[derive(PartialEq, Eq)]
pub struct BlindVaultReplicaRestoredExecution {
    pub(super) execution: BlindVaultReplicaExecution,
    pub(super) snapshot_sequence: u64,
}

impl fmt::Debug for BlindVaultReplicaRestoredExecution {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRestoredExecution")
            .field("snapshot_sequence", &self.snapshot_sequence)
            .field("execution", &self.execution)
            .finish()
    }
}

impl BlindVaultReplicaRestoredExecution {
    /// Authenticated monotonic sequence carried by the sealed snapshot.
    #[must_use]
    pub const fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    /// Borrows the restored source-owned workflow state.
    #[must_use]
    pub const fn execution(&self) -> &BlindVaultReplicaExecution {
        &self.execution
    }

    /// Transfers the restored workflow into its runtime owner.
    #[must_use]
    pub fn into_execution(self) -> BlindVaultReplicaExecution {
        self.execution
    }
}

/// Fail-closed source workflow errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindVaultReplicaWorkflowError {
    /// A signed Blind Vault request or receipt violated its protocol contract.
    #[error("blind vault replica workflow evidence violated the protocol")]
    BlindVault(#[from] BlindVaultError),
    /// Retry count or evidence timeout was zero.
    #[error("blind vault replica workflow policy is invalid")]
    InvalidPolicy,
    /// Dispatch concurrency was zero or exceeded the bounded work set.
    #[error("blind vault replica workflow dispatch limit is invalid")]
    InvalidDispatchLimit,
    /// Workflow generation identity was all zero.
    #[error("blind vault replica workflow id is invalid")]
    InvalidWorkflowId,
    /// Planner output exceeded the bounded work set.
    #[error("blind vault replica workflow has too many work items: {actual}")]
    TooManyWorkItems { actual: usize },
    /// Plan health and action set contradicted one another.
    #[error("blind vault replica plan is internally inconsistent")]
    InconsistentPlan,
    /// Event time was zero, reordered, or overflowed.
    #[error("blind vault replica workflow timestamp is out of range")]
    TimestampOutOfRange,
    /// Work identity did not belong to this workflow generation.
    #[error("blind vault replica work item was not found")]
    WorkItemNotFound,
    /// Requested transition was not valid from the current item state.
    #[error("blind vault replica work item transition is invalid")]
    InvalidTransition,
    /// Retry was attempted before its backoff boundary.
    #[error("blind vault replica work item retry is not ready")]
    RetryNotReady,
    /// Attempt arithmetic overflowed the bounded counter.
    #[error("blind vault replica work item attempt counter overflowed")]
    AttemptOverflow,
    /// Dispatch exceeded the configured attempt budget.
    #[error("blind vault replica work item attempt budget is exhausted")]
    AttemptBudgetExhausted,
    /// The workflow already has its configured number of in-flight actions.
    #[error("blind vault replica workflow dispatch capacity is reached")]
    DispatchCapacityReached,
    /// Another operation for the same terminal/lease is currently in flight.
    #[error("blind vault replica target already has an in-flight action")]
    TargetInFlight,
    /// An earlier action for the same terminal/lease lacks accepted evidence.
    #[error("blind vault replica target dependency is not complete")]
    TargetDependencyPending,
    /// Terminal evidence did not answer the exact request.
    #[error("blind vault replica evidence request mismatch")]
    EvidenceRequestMismatch,
    /// Verified evidence did not match the planned action kind or target.
    #[error("blind vault replica evidence action mismatch")]
    EvidenceActionMismatch,
    /// Evidence arrived after its dispatch or lease validity window.
    #[error("blind vault replica evidence expired")]
    EvidenceExpired,
    /// Terminal evidence exceeded the allowed future clock skew.
    #[error("blind vault replica evidence is from the future")]
    EvidenceFromFuture,
    /// Replacement completion omitted a verified old-lease retirement receipt.
    #[error("blind vault replica replacement requires terminal retirement evidence")]
    RetirementEvidenceRequired,
    /// Retirement was requested before a verified replacement was ready.
    #[error("blind vault replica replacement is not ready for retirement")]
    ReplacementRetirementNotReady,
    /// Receipt terminal descriptor identity was not valid Ed25519.
    #[error("blind vault replica terminal identity is invalid")]
    InvalidTerminalIdentity,
    /// Whole-set replanning is forbidden until all actions hold evidence.
    #[error("blind vault replica workflow is not ready for replanning")]
    ReplanNotReady,
    /// Local restart state exceeded its bounded encrypted container.
    #[error("blind vault replica restart snapshot is too large")]
    RestartSnapshotTooLarge,
    /// Local restart bytes did not match the versioned container contract.
    #[error("blind vault replica restart snapshot is malformed")]
    RestartSnapshotMalformed,
    /// Local restart bytes use an unsupported container version.
    #[error("blind vault replica restart snapshot version is unsupported")]
    RestartSnapshotVersionUnsupported,
    /// Snapshot sequence zero cannot participate in rollback protection.
    #[error("blind vault replica restart snapshot sequence is invalid")]
    RestartSnapshotSequenceInvalid,
    /// An authenticated snapshot is older than the protected high-water mark.
    #[error("blind vault replica restart snapshot rollback was detected")]
    RestartSnapshotRollbackDetected,
    /// Restart snapshot authentication or local key derivation failed.
    #[error("blind vault replica restart snapshot authentication failed")]
    RestartSnapshotAuthenticationFailed,
    /// Decrypted restart state violated workflow invariants.
    #[error("blind vault replica restart snapshot state is invalid")]
    RestartSnapshotStateInvalid,
}

pub(super) fn require_timestamp(timestamp_ms: u64) -> Result<(), BlindVaultReplicaWorkflowError> {
    if timestamp_ms == 0 {
        return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
    }
    Ok(())
}
