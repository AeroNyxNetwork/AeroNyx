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
//! - Requires all work items to hold evidence before replanning.
//!
//! ## Dependencies
//! - `blind_vault_replica_workflow/evidence.rs`: receipt and inventory proof.
//! - `blind_vault_replica_workflow/execution.rs`: monotonic state machine.
//! - `protocol::blind_vault`: planner actions and terminal evidence.
//! - `protocol::onion`: descriptor-authenticated route failure disposition.
//!
//! ## Main Logical Flow
//! 1. The source creates a workflow from `BlindVaultReplicaPlan`.
//! 2. The user/client explicitly authorizes each planned action.
//! 3. The client starts one attempt and executes its typed encrypted-terminal
//!    dispatch contract; compound repair/replacement work uses ordered stages.
//! 4. The workflow accepts only action-matching, verified terminal evidence.
//! 5. Fresh inventories are planned again before declaring convergence.
//!
//! ## Important Note For The Next Developer
//! - This module is source-owned state, not a public wire/storage format.
//! - Never serialize it into discovery, the public ledger, or node telemetry.
//! - Never add ciphertext, capabilities, lease keys, owner IDs, or contacts.
//! - A node receipt proves one terminal operation, not whole-set convergence.
//!
//! Last Modified: v1.8.0-ReplicaDispatchContract - Made each action's required
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

mod evidence;
mod execution;

use thiserror::Error;

use super::blind_vault::{
    BlindVaultError, BlindVaultReplicaAction, BlindVaultReplicaPlanHealth,
    BlindVaultTerminalFailureCode, MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS,
};
use super::onion::OnionRoutePlanError;

/// At most two per-member actions plus one aggregate provisioning action can
/// be emitted by the current deterministic planner.
pub const MAX_BLIND_VAULT_REPLICA_WORK_ITEMS: usize = MAX_BLIND_VAULT_REPLICA_PLAN_ACTIONS;

/// Privacy-conservative default: one cross-replica operation in flight.
///
/// Apps that explicitly accept the timing-correlation and resource tradeoff
/// may choose a larger bounded value through `new_with_maximum_in_flight`.
pub const DEFAULT_BLIND_VAULT_REPLICA_MAXIMUM_IN_FLIGHT: u8 = 1;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlindVaultReplicaWorkId {
    pub(super) workflow_id: [u8; 16],
    pub(super) sequence: u16,
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
        match value {
            OnionRoutePlanError::EmptyPath
            | OnionRoutePlanError::DescriptorRejected { .. }
            | OnionRoutePlanError::MissingCapability { .. }
            | OnionRoutePlanError::MissingProtocolFeature { .. }
            | OnionRoutePlanError::MissingX25519Kem { .. }
            | OnionRoutePlanError::MissingPublicEndpoint { .. }
            | OnionRoutePlanError::OutsideValidityWindow => Self::TransportUnavailable,
            OnionRoutePlanError::TooManyHops { .. }
            | OnionRoutePlanError::DuplicateNode { .. }
            | OnionRoutePlanError::SourceIncluded { .. } => Self::PolicyRejected,
            OnionRoutePlanError::SourceIdentityMismatch
            | OnionRoutePlanError::EnvelopeConstruction { .. } => Self::LocalConstructionFailed,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// One source-local action and its execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultReplicaWorkItem {
    pub(super) id: BlindVaultReplicaWorkId,
    pub(super) action: BlindVaultReplicaAction,
    pub(super) state: BlindVaultReplicaWorkState,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedProvisionedReplica {
    pub(super) node_id: [u8; 32],
    pub(super) lease_id: [u8; 32],
    pub(super) accepted_at_ms: u64,
    pub(super) observed_at_ms: u64,
}

/// Verified terminal retirement of one complete old replica lease.
///
/// Private fields prevent a caller from marking replacement complete using a
/// locally invented node or lease identifier. Construction is available only
/// through the receipt-verifying implementation in `evidence.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlindVaultVerifiedRetiredReplica {
    pub(super) node_id: [u8; 32],
    pub(super) lease_id: [u8; 32],
    pub(super) retired_at_ms: u64,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaActionEvidence {
    pub(super) kind: BlindVaultReplicaActionEvidenceKind,
    pub(super) verified_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    ReplicasProvisioned {
        count: u8,
    },
}

/// Client-owned execution state for one immutable planner generation.
///
/// [BLIND-VAULT-REPLICA-WORKFLOW 2026-08-28 by Codex] The workflow stores no
/// cross-replica repair material. The App remains responsible for independently
/// wrapping data and constructing each encrypted terminal request.
#[derive(Debug, PartialEq, Eq)]
pub struct BlindVaultReplicaExecution {
    pub(super) workflow_id: [u8; 16],
    pub(super) created_at_ms: u64,
    pub(super) source_plan_health: BlindVaultReplicaPlanHealth,
    pub(super) policy: BlindVaultReplicaExecutionPolicy,
    pub(super) maximum_in_flight: u8,
    pub(super) items: Vec<BlindVaultReplicaWorkItem>,
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
    /// Receipt terminal descriptor identity was not valid Ed25519.
    #[error("blind vault replica terminal identity is invalid")]
    InvalidTerminalIdentity,
    /// Whole-set replanning is forbidden until all actions hold evidence.
    #[error("blind vault replica workflow is not ready for replanning")]
    ReplanNotReady,
}

pub(super) fn require_timestamp(timestamp_ms: u64) -> Result<(), BlindVaultReplicaWorkflowError> {
    if timestamp_ms == 0 {
        return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
    }
    Ok(())
}
