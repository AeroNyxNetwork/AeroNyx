// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/execution.rs
// ============================================
//! Monotonic client-owned execution state for one replica plan generation.
//!
//! [BLIND-VAULT-REPLICA-EXECUTION 2026-08-28 by Codex] Transport success is
//! intentionally not a terminal state. Every action must accept cryptographic
//! evidence, then a fresh planner pass must confirm whole-set convergence.

use super::{
    require_timestamp, BlindVaultReplicaActionEvidence, BlindVaultReplicaConvergence,
    BlindVaultReplicaDispatchFailure, BlindVaultReplicaExecution, BlindVaultReplicaExecutionPhase,
    BlindVaultReplicaExecutionPolicy, BlindVaultReplicaWorkId, BlindVaultReplicaWorkItem,
    BlindVaultReplicaWorkState, BlindVaultReplicaWorkflowError, MAX_BLIND_VAULT_REPLICA_WORK_ITEMS,
};
use crate::protocol::blind_vault::{BlindVaultReplicaPlan, BlindVaultReplicaPlanHealth};

impl BlindVaultReplicaExecution {
    /// Creates one source-local execution generation from stable planner order.
    pub fn new(
        workflow_id: [u8; 16],
        created_at_ms: u64,
        policy: BlindVaultReplicaExecutionPolicy,
        plan: &BlindVaultReplicaPlan,
    ) -> Result<Self, BlindVaultReplicaWorkflowError> {
        require_timestamp(created_at_ms)?;
        policy.validate()?;
        if workflow_id == [0; 16] {
            return Err(BlindVaultReplicaWorkflowError::InvalidWorkflowId);
        }
        if plan.actions.len() > MAX_BLIND_VAULT_REPLICA_WORK_ITEMS {
            return Err(BlindVaultReplicaWorkflowError::TooManyWorkItems {
                actual: plan.actions.len(),
            });
        }
        if plan.actions.is_empty() && plan.health != BlindVaultReplicaPlanHealth::Healthy {
            return Err(BlindVaultReplicaWorkflowError::InconsistentPlan);
        }

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
            policy,
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
        self.validate_event_time(dispatched_at_ms)?;
        let policy = self.policy;
        let item = self.item_mut(id)?;
        let attempt = match item.state {
            BlindVaultReplicaWorkState::Authorized { authorized_at_ms } => {
                if dispatched_at_ms < authorized_at_ms {
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
                if dispatched_at_ms < failed_at_ms || dispatched_at_ms < retry_not_before_ms {
                    return Err(BlindVaultReplicaWorkflowError::RetryNotReady);
                }
                attempt
                    .checked_add(1)
                    .ok_or(BlindVaultReplicaWorkflowError::AttemptOverflow)?
            }
            _ => return Err(BlindVaultReplicaWorkflowError::InvalidTransition),
        };
        if attempt > policy.maximum_attempts {
            return Err(BlindVaultReplicaWorkflowError::AttemptBudgetExhausted);
        }
        let evidence_deadline_ms = dispatched_at_ms
            .checked_add(policy.evidence_timeout_ms)
            .ok_or(BlindVaultReplicaWorkflowError::TimestampOutOfRange)?;
        item.state = BlindVaultReplicaWorkState::AwaitingEvidence {
            attempt,
            dispatched_at_ms,
            evidence_deadline_ms,
        };
        Ok(attempt)
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
        if !evidence.matches(&item.action) {
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
        } else if retry_not_before_ms < failed_at_ms {
            return Err(BlindVaultReplicaWorkflowError::TimestampOutOfRange);
        } else if attempt >= maximum_attempts {
            BlindVaultReplicaWorkState::Exhausted {
                attempt,
                failed_at_ms,
                failure,
            }
        } else {
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
        let retry_not_before_ms = now_ms
            .checked_add(retry_delay_ms)
            .ok_or(BlindVaultReplicaWorkflowError::TimestampOutOfRange)?;
        let maximum_attempts = self.policy.maximum_attempts;
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
        if next_plan.actions.is_empty() {
            if next_plan.health != BlindVaultReplicaPlanHealth::Healthy {
                return Err(BlindVaultReplicaWorkflowError::InconsistentPlan);
            }
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
        if id.workflow_id != self.workflow_id {
            return Err(BlindVaultReplicaWorkflowError::WorkItemNotFound);
        }
        self.items
            .get_mut(usize::from(id.sequence))
            .filter(|item| item.id == id)
            .ok_or(BlindVaultReplicaWorkflowError::WorkItemNotFound)
    }
}
