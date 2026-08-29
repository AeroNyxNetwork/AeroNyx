// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/execution.rs
// ============================================
//! Monotonic client-owned execution state for one replica plan generation.
//!
//! [BLIND-VAULT-REPLICA-EXECUTION 2026-08-28 by Codex] Transport success is
//! intentionally not a terminal state. Every action must accept cryptographic
//! evidence, then a fresh planner pass must confirm whole-set convergence.

use super::{
    require_timestamp, BlindVaultReplacementRetirementPermit, BlindVaultReplicaActionEvidence,
    BlindVaultReplicaConvergence, BlindVaultReplicaDispatchFailure,
    BlindVaultReplicaDispatchReadiness, BlindVaultReplicaExecution,
    BlindVaultReplicaExecutionPhase, BlindVaultReplicaExecutionPolicy, BlindVaultReplicaWorkId,
    BlindVaultReplicaWorkItem, BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError,
    BlindVaultVerifiedProvisionedReplica, DEFAULT_BLIND_VAULT_REPLICA_MAXIMUM_IN_FLIGHT,
    MAX_BLIND_VAULT_REPLICA_WORK_ITEMS,
};
use crate::protocol::blind_vault::{
    BlindVaultReplicaAction, BlindVaultReplicaPlan, BlindVaultReplicaPlanHealth,
};

impl BlindVaultReplicaExecution {
    /// Creates one source-local execution generation from stable planner order.
    pub fn new(
        workflow_id: [u8; 16],
        created_at_ms: u64,
        policy: BlindVaultReplicaExecutionPolicy,
        plan: &BlindVaultReplicaPlan,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        Self::new_with_maximum_in_flight(
            workflow_id,
            created_at_ms,
            policy,
            DEFAULT_BLIND_VAULT_REPLICA_MAXIMUM_IN_FLIGHT,
            plan,
        )
    }

    /// Creates one generation with an explicit bounded dispatch limit.
    ///
    /// [BLIND-VAULT-BOUNDED-DISPATCH 2026-08-29 by Codex] The default remains
    /// sequential to reduce timing correlation across unrelated replicas.
    /// Higher limits are opt-in, bounded by the maximum work set, and never
    /// permit concurrent operations against the same terminal/lease.
    pub fn new_with_maximum_in_flight(
        workflow_id: [u8; 16],
        created_at_ms: u64,
        policy: BlindVaultReplicaExecutionPolicy,
        maximum_in_flight: u8,
        plan: &BlindVaultReplicaPlan,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(created_at_ms)?;
        policy.validate()?;
        if maximum_in_flight == 0
            || usize::from(maximum_in_flight) > MAX_BLIND_VAULT_REPLICA_WORK_ITEMS
        {
            return Err(BlindVaultReplicaWorkflowError::InvalidDispatchLimit);
        }
        if workflow_id == [0; 16] {
            return Err(BlindVaultReplicaWorkflowError::InvalidWorkflowId);
        }
        if plan.actions.len() > MAX_BLIND_VAULT_REPLICA_WORK_ITEMS {
            return Err(BlindVaultReplicaWorkflowError::TooManyWorkItems {
                actual: plan.actions.len(),
            });
        }
        plan.validate_shape()
            .map_err(|_| BlindVaultReplicaWorkflowError::InconsistentPlan)?;

        let mut items = Vec::with_capacity(plan.actions.len());
        for (sequence, action) in plan.actions.iter().copied().enumerate() {
            items.push(BlindVaultReplicaWorkItem {
                id: BlindVaultReplicaWorkId {
                    workflow_id,
                    sequence: u16::try_from(sequence).map_err(|_| {
                        BlindVaultReplicaWorkflowError::TooManyWorkItems {
                            actual: plan.actions.len(),
                        }
                    })?,
                },
                action,
                state: BlindVaultReplicaWorkState::AwaitingAuthorization,
            });
        }
        Ok(Self {
            workflow_id,
            created_at_ms,
            source_plan_health: plan.health,
            // [BLIND-VAULT-RESTART-PLAN-SUMMARY 2026-08-29 by Codex] Preserve
            // the complete bounded planner summary so a future restored state
            // can reconstruct and re-run `BlindVaultReplicaPlan::validate_shape`.
            source_configured_replicas: plan.configured_replicas,
            source_live_verified_replicas: plan.live_verified_replicas,
            source_live_matching_replicas: plan.live_matching_replicas,
            policy,
            maximum_in_flight,
            items,
        })
    }

    /// Random source-local workflow generation identifier.
    #[must_use]
    pub const fn workflow_id(&self) -> [u8; 16] {
        self.workflow_id
    }

    /// Source time at which the immutable plan became an execution generation.
    #[must_use]
    pub const fn created_at_ms(&self) -> u64 {
        self.created_at_ms
    }

    /// Health of the plan that produced this generation.
    #[must_use]
    pub const fn source_plan_health(&self) -> BlindVaultReplicaPlanHealth {
        self.source_plan_health
    }

    /// Intended replica membership represented by the source plan.
    #[must_use]
    pub const fn source_configured_replicas(&self) -> u8 {
        self.source_configured_replicas
    }

    /// Live terminal-signed observations represented by the source plan.
    #[must_use]
    pub const fn source_live_verified_replicas(&self) -> u8 {
        self.source_live_verified_replicas
    }

    /// Live observations matching their private source manifests.
    #[must_use]
    pub const fn source_live_matching_replicas(&self) -> u8 {
        self.source_live_matching_replicas
    }

    /// Maximum unrelated actions permitted to await evidence concurrently.
    #[must_use]
    pub const fn maximum_in_flight(&self) -> u8 {
        self.maximum_in_flight
    }

    /// Current number of dispatched actions awaiting cryptographic evidence.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| {
                matches!(
                    item.state,
                    BlindVaultReplicaWorkState::AwaitingEvidence { .. }
                )
            })
            .count()
    }

    /// Read-only work items for orchestration and user confirmation UI.
    #[must_use]
    pub fn items(&self) -> &[BlindVaultReplicaWorkItem] {
        &self.items
    }

    /// Derives aggregate phase from work item states.
    #[must_use]
    pub fn phase(&self) -> BlindVaultReplicaExecutionPhase {
        if self.items.is_empty() {
            return BlindVaultReplicaExecutionPhase::Converged;
        }
        if self.items.iter().any(|item| {
            matches!(
                item.state,
                BlindVaultReplicaWorkState::PermanentFailure { .. }
                    | BlindVaultReplicaWorkState::Exhausted { .. }
                    | BlindVaultReplicaWorkState::Cancelled { .. }
            )
        }) {
            return BlindVaultReplicaExecutionPhase::Blocked;
        }
        if self.items.iter().any(|item| {
            matches!(
                item.state,
                BlindVaultReplicaWorkState::AwaitingEvidence { .. }
            )
        }) {
            return BlindVaultReplicaExecutionPhase::AwaitingEvidence;
        }
        if self
            .items
            .iter()
            .any(|item| matches!(item.state, BlindVaultReplicaWorkState::Authorized { .. }))
        {
            return BlindVaultReplicaExecutionPhase::ReadyToDispatch;
        }
        if self.items.iter().any(|item| {
            matches!(
                item.state,
                BlindVaultReplicaWorkState::AwaitingAuthorization
            )
        }) {
            return BlindVaultReplicaExecutionPhase::AwaitingAuthorization;
        }
        if self.items.iter().any(|item| {
            matches!(
                item.state,
                BlindVaultReplicaWorkState::RetryableFailure { .. }
            )
        }) {
            return BlindVaultReplicaExecutionPhase::RetryBackoff;
        }
        BlindVaultReplicaExecutionPhase::AwaitingReplan
    }

    /// Explicitly authorizes one exact immutable action.
    pub fn authorize(
        &mut self,
        id: BlindVaultReplicaWorkId,
        authorized_at_ms: u64,
    ) -> Result<(), BlindVaultReplicaWorkflowError> {
        self.validate_event_time(authorized_at_ms)?;
        let item = self.item_mut(id)?;
        if item.state != BlindVaultReplicaWorkState::AwaitingAuthorization {
            return Err(BlindVaultReplicaWorkflowError::InvalidTransition);
        }
        item.state = BlindVaultReplicaWorkState::Authorized { authorized_at_ms };
        Ok(())
    }

    /// Records dispatch and opens a bounded evidence window. Retries keep the
    /// same work id and immutable action while incrementing only the attempt.
    pub fn dispatch(
        &mut self,
        id: BlindVaultReplicaWorkId,
        dispatched_at_ms: u64,
    ) -> Result<u8, BlindVaultReplicaWorkflowError> {
        let (attempt, evidence_deadline_ms) = match self.dispatch_readiness(id, dispatched_at_ms)? {
            BlindVaultReplicaDispatchReadiness::Ready {
                attempt,
                evidence_deadline_ms,
            } => (attempt, evidence_deadline_ms),
            BlindVaultReplicaDispatchReadiness::RetryBackoff { .. } => {
                return Err(BlindVaultReplicaWorkflowError::RetryNotReady)
            }
            BlindVaultReplicaDispatchReadiness::TargetInFlight => {
                return Err(BlindVaultReplicaWorkflowError::TargetInFlight)
            }
            BlindVaultReplicaDispatchReadiness::TargetDependencyPending => {
                return Err(BlindVaultReplicaWorkflowError::TargetDependencyPending)
            }
            BlindVaultReplicaDispatchReadiness::CapacityReached { .. } => {
                return Err(BlindVaultReplicaWorkflowError::DispatchCapacityReached)
            }
            BlindVaultReplicaDispatchReadiness::AwaitingAuthorization
            | BlindVaultReplicaDispatchReadiness::AlreadyInFlight { .. }
            | BlindVaultReplicaDispatchReadiness::TerminalState => {
                return Err(BlindVaultReplicaWorkflowError::InvalidTransition)
            }
        };
        let item_index = self.item_index(id)?;
        self.items[item_index].state = BlindVaultReplicaWorkState::AwaitingEvidence {
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
        };
        Ok(attempt)
    }

    /// Returns whether one exact action can be dispatched without mutation.
    #[must_use]
    pub fn dispatch_readiness(
        &self,
        id: BlindVaultReplicaWorkId,
        now_ms: u64,
    ) -> Result<BlindVaultReplicaDispatchReadiness, BlindVaultReplicaWorkflowError> {
        self.validate_event_time(now_ms)?;
        let item_index = self.item_index(id)?;
        let item = self.items[item_index];
        let attempt = match item.state {
            BlindVaultReplicaWorkState::AwaitingAuthorization => {
                return Ok(BlindVaultReplicaDispatchReadiness::AwaitingAuthorization)
            }
            BlindVaultReplicaWorkState::Authorized { authorized_at_ms } => {
                if now_ms < authorized_at_ms {
                    return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
                }
                1
            }
            BlindVaultReplicaWorkState::RetryableFailure {
                attempt,
                failed_at_ms,
                retry_not_before_ms,
                ..
            } => {
                if now_ms < failed_at_ms || now_ms < retry_not_before_ms {
                    return Ok(BlindVaultReplicaDispatchReadiness::RetryBackoff {
                        retry_not_before_ms,
                    });
                }
                attempt
                    .checked_add(1)
                    .ok_or(BlindVaultReplicaWorkflowError::AttemptOverflow)?
            }
            BlindVaultReplicaWorkState::AwaitingEvidence {
                evidence_deadline_ms,
                ..
            } => {
                return Ok(BlindVaultReplicaDispatchReadiness::AlreadyInFlight {
                    evidence_deadline_ms,
                })
            }
            BlindVaultReplicaWorkState::EvidenceAccepted { .. }
            | BlindVaultReplicaWorkState::PermanentFailure { .. }
            | BlindVaultReplicaWorkState::Exhausted { .. }
            | BlindVaultReplicaWorkState::Cancelled { .. } => {
                return Ok(BlindVaultReplicaDispatchReadiness::TerminalState)
            }
        };
        if attempt > self.policy.maximum_attempts {
            return Err(BlindVaultReplicaWorkflowError::AttemptBudgetExhausted);
        }

        if let Some(target) = item.action.target() {
            if self.items.iter().enumerate().any(|(index, active)| {
                index != item_index
                    && active.action.target() == Some(target)
                    && matches!(
                        active.state,
                        BlindVaultReplicaWorkState::AwaitingEvidence { .. }
                    )
            }) {
                return Ok(BlindVaultReplicaDispatchReadiness::TargetInFlight);
            }
            if self.items[..item_index].iter().any(|prior| {
                prior.action.target() == Some(target)
                    && !matches!(
                        prior.state,
                        BlindVaultReplicaWorkState::EvidenceAccepted { .. }
                    )
            }) {
                return Ok(BlindVaultReplicaDispatchReadiness::TargetDependencyPending);
            }
        }

        let in_flight = self.in_flight_count();
        if in_flight >= usize::from(self.maximum_in_flight) {
            return Ok(BlindVaultReplicaDispatchReadiness::CapacityReached {
                in_flight: u8::try_from(in_flight)
                    .map_err(|_| BlindVaultReplicaWorkflowError::InconsistentPlan)?,
                maximum: self.maximum_in_flight,
            });
        }
        let evidence_deadline_ms = now_ms
            .checked_add(self.policy.evidence_timeout_ms)
            .ok_or(BlindVaultReplicaWorkflowError::TimestampOutOfRange)?;
        Ok(BlindVaultReplicaDispatchReadiness::Ready {
            attempt,
            evidence_deadline_ms,
        })
    }

    /// Authorizes old-lease retirement only after the replacement is live.
    ///
    /// The replacement inventory must have been observed during the current
    /// dispatch attempt and this call must occur inside its evidence window.
    /// A retry therefore re-verifies the replacement instead of trusting stale
    /// process memory from an earlier failed attempt.
    ///
    /// [BLIND-VAULT-REPLACEMENT-RETIREMENT-PERMIT 2026-08-29 by Codex]
    pub fn replacement_retirement_permit(
        &self,
        id: BlindVaultReplicaWorkId,
        replacement: &BlindVaultVerifiedProvisionedReplica,
        authorized_at_ms: u64,
    ) -> Result<BlindVaultReplacementRetirementPermit, BlindVaultReplicaWorkflowError> {
        self.validate_event_time(authorized_at_ms)?;
        let item = self.items[self.item_index(id)?];
        let (attempt, dispatched_at_ms, evidence_deadline_ms) = match item.state {
            BlindVaultReplicaWorkState::AwaitingEvidence {
                attempt,
                dispatched_at_ms,
                evidence_deadline_ms,
            } => (attempt, dispatched_at_ms, evidence_deadline_ms),
            _ => return Err(BlindVaultReplicaWorkflowError::InvalidTransition),
        };
        let BlindVaultReplicaAction::ReplaceReplica {
            node_id: replaced_node_id,
            lease_id: replaced_lease_id,
        } = item.action
        else {
            return Err(BlindVaultReplicaWorkflowError::InvalidTransition);
        };
        if authorized_at_ms > evidence_deadline_ms
            || replacement.observed_at_ms < dispatched_at_ms
            || replacement.observed_at_ms > authorized_at_ms
            || replacement.accepted_at_ms > replacement.observed_at_ms
            || replacement.node_id == replaced_node_id
            || replacement.lease_id == replaced_lease_id
        {
            return Err(BlindVaultReplicaWorkflowError::ReplacementRetirementNotReady);
        }

        Ok(BlindVaultReplacementRetirementPermit {
            work_id: id,
            attempt,
            replaced_node_id,
            replaced_lease_id,
            replacement_node_id: replacement.node_id,
            replacement_lease_id: replacement.lease_id,
            authorized_at_ms,
        })
    }

    /// Accepts only verified evidence matching both the action kind and target.
    pub fn accept_evidence(
        &mut self,
        id: BlindVaultReplicaWorkId,
        evidence: &BlindVaultReplicaActionEvidence,
    ) -> Result<(), BlindVaultReplicaWorkflowError> {
        self.validate_event_time(evidence.verified_at_ms)?;
        let item = self.item_mut(id)?;
        let (attempt, dispatched_at_ms, evidence_deadline_ms) = match item.state {
            BlindVaultReplicaWorkState::AwaitingEvidence {
                attempt,
                dispatched_at_ms,
                evidence_deadline_ms,
            } => (attempt, dispatched_at_ms, evidence_deadline_ms),
            _ => return Err(BlindVaultReplicaWorkflowError::InvalidTransition),
        };
        if evidence.verified_at_ms < dispatched_at_ms {
            return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
        }
        if evidence.verified_at_ms > evidence_deadline_ms {
            return Err(BlindVaultReplicaWorkflowError::EvidenceExpired);
        }
        // Permit-gated replacement evidence is generation- and attempt-bound;
        // legacy evidence retains its existing action-only compatibility path.
        if !evidence.matches(&item.action) || !evidence.matches_attempt(id, attempt) {
            return Err(BlindVaultReplicaWorkflowError::EvidenceActionMismatch);
        }
        item.state = BlindVaultReplicaWorkState::EvidenceAccepted {
            attempt,
            verified_at_ms: evidence.verified_at_ms,
        };
        Ok(())
    }

    /// Records one typed failure without retrying permanent outcomes.
    ///
    /// `retry_not_before_ms` is required only for retryable failures. It is
    /// ignored for permanent failures so callers never need to invent a retry
    /// schedule for a request that must stop immediately.
    pub fn record_failure(
        &mut self,
        id: BlindVaultReplicaWorkId,
        failed_at_ms: u64,
        retry_not_before_ms: u64,
        failure: BlindVaultReplicaDispatchFailure,
    ) -> Result<(), BlindVaultReplicaWorkflowError> {
        self.validate_event_time(failed_at_ms)?;
        let maximum_attempts = self.policy.maximum_attempts;
        let item = self.item_mut(id)?;
        let (attempt, dispatched_at_ms) = match item.state {
            BlindVaultReplicaWorkState::AwaitingEvidence {
                attempt,
                dispatched_at_ms,
                ..
            } => (attempt, dispatched_at_ms),
            _ => return Err(BlindVaultReplicaWorkflowError::InvalidTransition),
        };
        if failed_at_ms < dispatched_at_ms {
            return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
        }
        item.state = if !failure.is_retryable() {
            // [BLIND-VAULT-PERMANENT-FAILURE 2026-08-28 by Codex] Rejection,
            // stale plans, local policy denials, and unsupported response
            // classes are terminal for this immutable generation.
            BlindVaultReplicaWorkState::PermanentFailure {
                attempt,
                failed_at_ms,
                failure,
            }
        } else if attempt >= maximum_attempts {
            // [BLIND-VAULT-TERMINAL-ATTEMPT-BOUNDARY 2026-08-28 by Codex]
            // Exhaustion has no future retry transition, so an absent or
            // overflowing retry schedule must not prevent recording the final
            // failure. This branch intentionally precedes backoff validation.
            BlindVaultReplicaWorkState::Exhausted {
                attempt,
                failed_at_ms,
                failure,
            }
        } else {
            if retry_not_before_ms < failed_at_ms {
                return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
            }
            BlindVaultReplicaWorkState::RetryableFailure {
                attempt,
                failed_at_ms,
                retry_not_before_ms,
                failure,
            }
        };
        Ok(())
    }

    /// Converts every overdue in-flight item into a bounded timeout failure.
    pub fn expire_overdue(
        &mut self,
        now_ms: u64,
        retry_delay_ms: u64,
    ) -> Result<usize, BlindVaultReplicaWorkflowError> {
        self.validate_event_time(now_ms)?;
        let maximum_attempts = self.policy.maximum_attempts;
        // Validate the one shared backoff boundary before mutating any item.
        // This preserves all-or-nothing expiry if time arithmetic overflows.
        let has_retryable_expiry = self.items.iter().any(|item| {
            matches!(
                item.state,
                BlindVaultReplicaWorkState::AwaitingEvidence {
                    attempt,
                    evidence_deadline_ms,
                    ..
                } if now_ms > evidence_deadline_ms && attempt < maximum_attempts
            )
        });
        let retry_not_before_ms = if has_retryable_expiry {
            now_ms
                .checked_add(retry_delay_ms)
                .ok_or(BlindVaultReplicaWorkflowError::TimestampOutOfRange)?
        } else {
            // No retryable item will consume this value; keeping it bounded
            // avoids an impossible-state panic inside the mutation loop.
            now_ms
        };
        let mut expired = 0;
        for item in &mut self.items {
            let BlindVaultReplicaWorkState::AwaitingEvidence {
                attempt,
                evidence_deadline_ms,
                ..
            } = item.state
            else {
                continue;
            };
            if now_ms <= evidence_deadline_ms {
                continue;
            }
            expired += 1;
            item.state = if attempt >= maximum_attempts {
                BlindVaultReplicaWorkState::Exhausted {
                    attempt,
                    failed_at_ms: now_ms,
                    failure: BlindVaultReplicaDispatchFailure::EvidenceTimeout,
                }
            } else {
                // Only a state that can actually retry needs a representable
                // future boundary. A batch containing only final attempts does
                // not evaluate `now_ms + retry_delay_ms` at all.
                BlindVaultReplicaWorkState::RetryableFailure {
                    attempt,
                    failed_at_ms: now_ms,
                    retry_not_before_ms,
                    failure: BlindVaultReplicaDispatchFailure::EvidenceTimeout,
                }
            };
        }
        Ok(expired)
    }

    /// Explicitly cancels non-terminal work and blocks this generation.
    pub fn cancel(
        &mut self,
        id: BlindVaultReplicaWorkId,
        cancelled_at_ms: u64,
    ) -> Result<(), BlindVaultReplicaWorkflowError> {
        self.validate_event_time(cancelled_at_ms)?;
        let item = self.item_mut(id)?;
        if matches!(
            item.state,
            BlindVaultReplicaWorkState::EvidenceAccepted { .. }
                | BlindVaultReplicaWorkState::PermanentFailure { .. }
                | BlindVaultReplicaWorkState::Exhausted { .. }
                | BlindVaultReplicaWorkState::Cancelled { .. }
        ) {
            return Err(BlindVaultReplicaWorkflowError::InvalidTransition);
        }
        item.state = BlindVaultReplicaWorkState::Cancelled { cancelled_at_ms };
        Ok(())
    }

    /// Evaluates whole-set convergence only after every action has accepted
    /// evidence. Follow-up work requires a new workflow id and authorization.
    pub fn evaluate_convergence(
        &self,
        next_plan: &BlindVaultReplicaPlan,
    ) -> Result<BlindVaultReplicaConvergence, BlindVaultReplicaWorkflowError> {
        if self.phase() != BlindVaultReplicaExecutionPhase::AwaitingReplan {
            return Err(BlindVaultReplicaWorkflowError::ReplanNotReady);
        }
        if next_plan.actions.len() > MAX_BLIND_VAULT_REPLICA_WORK_ITEMS {
            return Err(BlindVaultReplicaWorkflowError::TooManyWorkItems {
                actual: next_plan.actions.len(),
            });
        }
        next_plan
            .validate_shape()
            .map_err(|_| BlindVaultReplicaWorkflowError::InconsistentPlan)?;
        if next_plan.actions.is_empty() {
            return Ok(BlindVaultReplicaConvergence::Converged);
        }
        let action_count = u16::try_from(next_plan.actions.len()).map_err(|_| {
            BlindVaultReplicaWorkflowError::TooManyWorkItems {
                actual: next_plan.actions.len(),
            }
        })?;
        Ok(BlindVaultReplicaConvergence::FollowUpRequired {
            health: next_plan.health,
            action_count,
        })
    }

    fn validate_event_time(
        &self,
        event_time_ms: u64,
    ) -> Result<(), BlindVaultReplicaWorkflowError> {
        require_timestamp(event_time_ms)?;
        if event_time_ms < self.created_at_ms {
            return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
        }
        Ok(())
    }

    fn item_mut(
        &mut self,
        id: BlindVaultReplicaWorkId,
    ) -> Result<&mut BlindVaultReplicaWorkItem, BlindVaultReplicaWorkflowError> {
        let index = self.item_index(id)?;
        Ok(&mut self.items[index])
    }

    fn item_index(
        &self,
        id: BlindVaultReplicaWorkId,
    ) -> Result<usize, BlindVaultReplicaWorkflowError> {
        if id.workflow_id != self.workflow_id {
            return Err(BlindVaultReplicaWorkflowError::WorkItemNotFound);
        }
        self.items
            .get(usize::from(id.sequence))
            .filter(|item| item.id == id)
            .map(|_| usize::from(id.sequence))
            .ok_or(BlindVaultReplicaWorkflowError::WorkItemNotFound)
    }
}
