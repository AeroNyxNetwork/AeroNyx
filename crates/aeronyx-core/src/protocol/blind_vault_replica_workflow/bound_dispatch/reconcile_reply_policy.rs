// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/reconcile_reply_policy.rs
// ============================================
//! Source-private reply policy for one Blind Vault inventory reconciliation.
//!
//! ## Creation Reason
//! Reconciliation can contain zero or more writes, then zero or more deletes,
//! followed by one inventory observation. Adapter-managed receipt handling
//! could reorder stages or accept an inventory captured before the mutations.
//!
//! ## Main Functionality
//! - Binds one policy to an exact reconciliation work item and manifest.
//! - Accepts writes before deletes and permanently closes the write phase.
//! - Requires every signed mutation timestamp to advance monotonically.
//! - Requires final inventory to postdate all accepted mutations.
//! - Emits an unforgeable completion capability only for live matching state.
//! - Redacts work, terminal, lease, manifest, and receipt data from Debug.
//!
//! ## Dependencies
//! - `request_bound_verifier.rs`: exact signed request/reply pairs and clock.
//! - `evidence.rs`: matching live inventory and reconciliation evidence.
//! - `BlindVaultReplicaWorkItem`: immutable action target and expected state.
//!
//! ## Main Logical Flow
//! 1. Validate the source manifest against the immutable planner action.
//! 2. Bind the first reply to the configured work id and runtime attempt.
//! 3. Accept zero or more exact write receipts in monotonic time order.
//! 4. Accept zero or more exact delete receipts; no later write may pass.
//! 5. Verify one fresh inventory observed after all accepted mutations.
//! 6. Emit typed reconciliation completion for durable resolution.
//!
//! ## Important Note For The Next Developer
//! - An inventory-only replay is valid when another actor already reconciled.
//! - Never relax the mutation-to-inventory time ordering requirement.
//! - This policy is source-private and intentionally not serializable.
//! - Never expose object identifiers or private manifest values in telemetry.
//!
//! Last Modified: v1.3.0-RuntimeAttemptPredicate - Removed an invalid const
//! promise from the work-id comparison used by runtime recovery.
//! v1.2.0-PrivacySafeClockDiagnostics - Redacted generic clock
//! and source-private policy errors from standard diagnostics.
//! v1.1.0-AttemptBoundCompletion - Preserved exact policy
//! attempt binding inside the emitted durable completion capability.
//! v1.0.0-ReconcileReplyPolicy - Initial attempt-bound,
//! monotonic mutation and post-mutation inventory state machine.
//! ============================================

use std::{error::Error, fmt};

use super::super::{
    BlindVaultReplicaActionEvidence, BlindVaultReplicaWorkId, BlindVaultReplicaWorkItem,
    BlindVaultReplicaWorkflowError,
};
use super::request_bound_verifier::{
    BlindVaultReplicaPrivateReplyPolicy, BlindVaultReplicaRequestBoundReply,
    BlindVaultReplicaVerificationClock,
};
use super::send_sequence::BlindVaultReplicaTerminalSendContext;
use crate::protocol::blind_vault::{
    BlindVaultLeaseInventoryReceipt, BlindVaultLeaseInventoryRequest, BlindVaultReplicaAction,
    BlindVaultReplicaEvidenceError, BlindVaultReplicaManifestExpectation,
    BlindVaultVerifiedReplicaInventory,
};

/// Successful private transition for one inventory reconciliation attempt.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaReconcileReplyOutcome {
    /// One exact ciphertext write was durably acknowledged.
    ObjectStored { accepted_writes: u16 },
    /// One exact surplus ciphertext deletion was durably acknowledged.
    ObjectDeleted { accepted_deletes: u16 },
    /// Live post-mutation inventory exactly matched the source manifest.
    ReconciliationCompleted(BlindVaultReplicaCompletedReconciliation),
}

impl fmt::Debug for BlindVaultReplicaReconcileReplyOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObjectStored { accepted_writes } => formatter
                .debug_struct("ObjectStored")
                .field("accepted_writes", accepted_writes)
                .finish(),
            Self::ObjectDeleted { accepted_deletes } => formatter
                .debug_struct("ObjectDeleted")
                .field("accepted_deletes", accepted_deletes)
                .finish(),
            Self::ReconciliationCompleted(_) => {
                formatter.write_str("ReconciliationCompleted([REDACTED])")
            }
        }
    }
}

/// Unforgeable completion capability for one exact reconciliation action.
///
/// [BLIND-VAULT-COMPLETED-RECONCILIATION-CAPABILITY 2026-08-30 by Codex]
/// Only the full ordered policy can create this value after a post-mutation,
/// matching, still-live inventory has been cryptographically verified.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaCompletedReconciliation {
    evidence: BlindVaultReplicaActionEvidence,
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
}

impl BlindVaultReplicaCompletedReconciliation {
    pub(in crate::protocol::blind_vault_replica_workflow) const fn evidence(
        &self,
    ) -> &BlindVaultReplicaActionEvidence {
        &self.evidence
    }

    // [CORE-BUILD-BOUNDARY 2026-08-31 by Codex] This comparison is runtime
    // validation; derived PartialEq is not const on the supported toolchain.
    pub(in crate::protocol::blind_vault_replica_workflow) fn matches_attempt(
        &self,
        work_id: BlindVaultReplicaWorkId,
        attempt: u8,
    ) -> bool {
        self.work_id == work_id && self.attempt == attempt
    }
}

impl fmt::Debug for BlindVaultReplicaCompletedReconciliation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCompletedReconciliation")
            .field("evidence", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct BlindVaultReplicaReconcileAttemptBinding {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
}

impl BlindVaultReplicaReconcileAttemptBinding {
    fn from_context(context: BlindVaultReplicaTerminalSendContext) -> Self {
        Self {
            work_id: context.work_id(),
            attempt: context.attempt(),
        }
    }

    fn matches(self, context: BlindVaultReplicaTerminalSendContext) -> bool {
        self.work_id == context.work_id() && self.attempt == context.attempt()
    }
}

impl fmt::Debug for BlindVaultReplicaReconcileAttemptBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaReconcileAttemptBinding")
            .field("attempt", &self.attempt)
            .field("work_id", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlindVaultReplicaReconcileReplyState {
    Writing {
        binding: Option<BlindVaultReplicaReconcileAttemptBinding>,
        accepted_writes: u16,
        latest_mutation_at_ms: u64,
    },
    Deleting {
        binding: BlindVaultReplicaReconcileAttemptBinding,
        accepted_deletes: u16,
        latest_mutation_at_ms: u64,
    },
    Complete,
}

impl BlindVaultReplicaReconcileReplyState {
    const fn name(self) -> &'static str {
        match self {
            Self::Writing { .. } => "writing",
            Self::Deleting { .. } => "deleting",
            Self::Complete => "complete",
        }
    }
}

/// Ordered source-private verification policy for inventory reconciliation.
///
/// [BLIND-VAULT-RECONCILE-REPLY-POLICY 2026-08-30 by Codex] This state
/// machine closes the write stage once deletion begins and treats matching
/// post-mutation inventory as the only successful terminal condition.
pub struct BlindVaultReplicaReconcileReplyPolicy<Clock> {
    expected_work_id: BlindVaultReplicaWorkId,
    expectation: BlindVaultReplicaManifestExpectation,
    clock: Clock,
    maximum_receipt_age_ms: u64,
    maximum_future_clock_skew_ms: u64,
    state: BlindVaultReplicaReconcileReplyState,
}

impl<Clock> BlindVaultReplicaReconcileReplyPolicy<Clock> {
    /// Creates one policy after exact action/manifest compatibility checks.
    pub fn new(
        work_item: &BlindVaultReplicaWorkItem,
        expectation: BlindVaultReplicaManifestExpectation,
        clock: Clock,
        maximum_receipt_age_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaReconcileReplyPolicyBuildError> {
        let BlindVaultReplicaAction::ReconcileInventory {
            node_id,
            lease_id,
            expected_object_count,
            expected_ciphertext_bytes,
            expected_inventory_commitment,
            ..
        } = work_item.action()
        else {
            return Err(BlindVaultReplicaReconcileReplyPolicyBuildError::WrongAction);
        };
        if expectation.node_id() != node_id
            || expectation.lease_id() != lease_id
            || expectation.object_count() != expected_object_count
            || expectation.ciphertext_bytes() != expected_ciphertext_bytes
            || expectation.inventory_commitment() != expected_inventory_commitment
        {
            return Err(BlindVaultReplicaReconcileReplyPolicyBuildError::ExpectationMismatch);
        }
        if maximum_receipt_age_ms == 0 {
            return Err(BlindVaultReplicaReconcileReplyPolicyBuildError::InvalidFreshnessPolicy);
        }
        Ok(Self {
            expected_work_id: work_item.id(),
            expectation,
            clock,
            maximum_receipt_age_ms,
            maximum_future_clock_skew_ms,
            state: BlindVaultReplicaReconcileReplyState::Writing {
                binding: None,
                accepted_writes: 0,
                latest_mutation_at_ms: 0,
            },
        })
    }

    /// Whether matching post-mutation evidence has been emitted.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.state, BlindVaultReplicaReconcileReplyState::Complete)
    }
}

impl<Clock> fmt::Debug for BlindVaultReplicaReconcileReplyPolicy<Clock> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaReconcileReplyPolicy")
            .field("clock", &std::any::type_name::<Clock>())
            .field("state", &self.state.name())
            .field("work_id", &"[REDACTED]")
            .field("expectation", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl<Clock> BlindVaultReplicaPrivateReplyPolicy for BlindVaultReplicaReconcileReplyPolicy<Clock>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    type Output = BlindVaultReplicaReconcileReplyOutcome;
    type Error = BlindVaultReplicaReconcileReplyPolicyError<Clock::Error>;

    fn verify_private_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        _adapter_state: &[u8],
        reply: BlindVaultReplicaRequestBoundReply,
    ) -> Result<Self::Output, Self::Error> {
        match (self.state, reply) {
            (
                BlindVaultReplicaReconcileReplyState::Writing {
                    binding,
                    accepted_writes,
                    latest_mutation_at_ms,
                },
                BlindVaultReplicaRequestBoundReply::ObjectStored { receipt, .. },
            ) => {
                require_unrestricted_terminal(context)?;
                let binding = require_or_create_binding(self.expected_work_id, binding, context)?;
                require_target(&self.expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReconcileReplyPolicyError::Clock)?;
                require_mutation_time(
                    receipt.accepted_at_ms,
                    latest_mutation_at_ms,
                    now_ms,
                    self.maximum_future_clock_skew_ms,
                )?;
                if receipt.stored_until_ms <= now_ms {
                    return Err(BlindVaultReplicaReconcileReplyPolicyError::ReceiptOutsideWindow);
                }
                let accepted_writes = accepted_writes
                    .checked_add(1)
                    .ok_or(BlindVaultReplicaReconcileReplyPolicyError::MutationCountOverflow)?;
                self.state = BlindVaultReplicaReconcileReplyState::Writing {
                    binding: Some(binding),
                    accepted_writes,
                    latest_mutation_at_ms: receipt.accepted_at_ms,
                };
                Ok(BlindVaultReplicaReconcileReplyOutcome::ObjectStored { accepted_writes })
            }
            (
                BlindVaultReplicaReconcileReplyState::Writing {
                    binding,
                    latest_mutation_at_ms,
                    ..
                },
                BlindVaultReplicaRequestBoundReply::ObjectDeleted { receipt, .. },
            ) => {
                require_unrestricted_terminal(context)?;
                let binding = require_or_create_binding(self.expected_work_id, binding, context)?;
                require_target(&self.expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReconcileReplyPolicyError::Clock)?;
                require_mutation_time(
                    receipt.deleted_at_ms,
                    latest_mutation_at_ms,
                    now_ms,
                    self.maximum_future_clock_skew_ms,
                )?;
                self.state = BlindVaultReplicaReconcileReplyState::Deleting {
                    binding,
                    accepted_deletes: 1,
                    latest_mutation_at_ms: receipt.deleted_at_ms,
                };
                Ok(BlindVaultReplicaReconcileReplyOutcome::ObjectDeleted {
                    accepted_deletes: 1,
                })
            }
            (
                BlindVaultReplicaReconcileReplyState::Deleting {
                    binding,
                    accepted_deletes,
                    latest_mutation_at_ms,
                },
                BlindVaultReplicaRequestBoundReply::ObjectDeleted { receipt, .. },
            ) => {
                require_same_attempt(binding, context)?;
                require_unrestricted_terminal(context)?;
                require_target(&self.expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReconcileReplyPolicyError::Clock)?;
                require_mutation_time(
                    receipt.deleted_at_ms,
                    latest_mutation_at_ms,
                    now_ms,
                    self.maximum_future_clock_skew_ms,
                )?;
                let accepted_deletes = accepted_deletes
                    .checked_add(1)
                    .ok_or(BlindVaultReplicaReconcileReplyPolicyError::MutationCountOverflow)?;
                self.state = BlindVaultReplicaReconcileReplyState::Deleting {
                    binding,
                    accepted_deletes,
                    latest_mutation_at_ms: receipt.deleted_at_ms,
                };
                Ok(BlindVaultReplicaReconcileReplyOutcome::ObjectDeleted { accepted_deletes })
            }
            (
                BlindVaultReplicaReconcileReplyState::Writing {
                    binding,
                    latest_mutation_at_ms,
                    ..
                },
                BlindVaultReplicaRequestBoundReply::InventoryObserved { request, receipt },
            ) => self.complete_reconciliation(
                binding,
                latest_mutation_at_ms,
                context,
                request,
                receipt,
            ),
            (
                BlindVaultReplicaReconcileReplyState::Deleting {
                    binding,
                    latest_mutation_at_ms,
                    ..
                },
                BlindVaultReplicaRequestBoundReply::InventoryObserved { request, receipt },
            ) => self.complete_reconciliation(
                Some(binding),
                latest_mutation_at_ms,
                context,
                request,
                receipt,
            ),
            _ => Err(BlindVaultReplicaReconcileReplyPolicyError::StageMismatch),
        }
    }
}

impl<Clock> BlindVaultReplicaReconcileReplyPolicy<Clock>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    fn complete_reconciliation(
        &mut self,
        binding: Option<BlindVaultReplicaReconcileAttemptBinding>,
        latest_mutation_at_ms: u64,
        context: BlindVaultReplicaTerminalSendContext,
        request: BlindVaultLeaseInventoryRequest,
        receipt: BlindVaultLeaseInventoryReceipt,
    ) -> Result<
        BlindVaultReplicaReconcileReplyOutcome,
        BlindVaultReplicaReconcileReplyPolicyError<Clock::Error>,
    > {
        require_unrestricted_terminal(context)?;
        let binding = require_or_create_binding(self.expected_work_id, binding, context)?;
        let now_ms = self
            .clock
            .now_ms()
            .map_err(BlindVaultReplicaReconcileReplyPolicyError::Clock)?;
        let inventory = BlindVaultVerifiedReplicaInventory::verify(
            &receipt,
            &request,
            &self.expectation,
            now_ms,
            self.maximum_receipt_age_ms,
            self.maximum_future_clock_skew_ms,
        )
        .map_err(BlindVaultReplicaReconcileReplyPolicyError::Inventory)?;
        if inventory.observed_at_ms() < latest_mutation_at_ms {
            return Err(BlindVaultReplicaReconcileReplyPolicyError::InventoryPredatesMutation);
        }
        let evidence = BlindVaultReplicaActionEvidence::inventory_reconciled(inventory, now_ms)
            .map_err(BlindVaultReplicaReconcileReplyPolicyError::Workflow)?;
        self.state = BlindVaultReplicaReconcileReplyState::Complete;
        Ok(
            BlindVaultReplicaReconcileReplyOutcome::ReconciliationCompleted(
                BlindVaultReplicaCompletedReconciliation {
                    evidence,
                    work_id: binding.work_id,
                    attempt: binding.attempt,
                },
            ),
        )
    }
}

fn require_or_create_binding<ClockError>(
    expected_work_id: BlindVaultReplicaWorkId,
    binding: Option<BlindVaultReplicaReconcileAttemptBinding>,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<
    BlindVaultReplicaReconcileAttemptBinding,
    BlindVaultReplicaReconcileReplyPolicyError<ClockError>,
> {
    let candidate =
        binding.unwrap_or_else(|| BlindVaultReplicaReconcileAttemptBinding::from_context(context));
    if context.work_id() != expected_work_id || !candidate.matches(context) {
        return Err(BlindVaultReplicaReconcileReplyPolicyError::AttemptMismatch);
    }
    Ok(candidate)
}

fn require_same_attempt<ClockError>(
    binding: BlindVaultReplicaReconcileAttemptBinding,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaReconcileReplyPolicyError<ClockError>> {
    binding
        .matches(context)
        .then_some(())
        .ok_or(BlindVaultReplicaReconcileReplyPolicyError::AttemptMismatch)
}

fn require_unrestricted_terminal<ClockError>(
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaReconcileReplyPolicyError<ClockError>> {
    context
        .authorized_terminal_node_id()
        .is_none()
        .then_some(())
        .ok_or(BlindVaultReplicaReconcileReplyPolicyError::TerminalAuthorizationMismatch)
}

fn require_target<ClockError>(
    expectation: &BlindVaultReplicaManifestExpectation,
    node_id: [u8; 32],
    lease_id: [u8; 32],
) -> Result<(), BlindVaultReplicaReconcileReplyPolicyError<ClockError>> {
    (expectation.node_id() == node_id && expectation.lease_id() == lease_id)
        .then_some(())
        .ok_or(BlindVaultReplicaReconcileReplyPolicyError::TargetMismatch)
}

fn require_mutation_time<ClockError>(
    mutation_at_ms: u64,
    latest_mutation_at_ms: u64,
    now_ms: u64,
    maximum_future_clock_skew_ms: u64,
) -> Result<(), BlindVaultReplicaReconcileReplyPolicyError<ClockError>> {
    if now_ms == 0
        || mutation_at_ms == 0
        || mutation_at_ms < latest_mutation_at_ms
        || (mutation_at_ms > now_ms && mutation_at_ms - now_ms > maximum_future_clock_skew_ms)
    {
        return Err(BlindVaultReplicaReconcileReplyPolicyError::MutationTimeInvalid);
    }
    Ok(())
}

/// Reconciliation policy construction failure before any reply is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaReconcileReplyPolicyBuildError {
    /// The supplied work item was not inventory reconciliation.
    WrongAction,
    /// Private manifest expectation did not match the planner action.
    ExpectationMismatch,
    /// Inventory receipt freshness policy was zero.
    InvalidFreshnessPolicy,
}

impl fmt::Display for BlindVaultReplicaReconcileReplyPolicyBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongAction => formatter
                .write_str("blind vault reconciliation reply policy requires reconciliation work"),
            Self::ExpectationMismatch => formatter
                .write_str("blind vault reconciliation expectation mismatched planner action"),
            Self::InvalidFreshnessPolicy => {
                formatter.write_str("blind vault reconciliation freshness policy is invalid")
            }
        }
    }
}

impl Error for BlindVaultReplicaReconcileReplyPolicyBuildError {}

/// Fail-closed reconciliation reply or lifecycle transition failure.
pub enum BlindVaultReplicaReconcileReplyPolicyError<ClockError> {
    /// Source clock could not provide verification time.
    Clock(ClockError),
    /// Reply did not belong to the current reconciliation stage.
    StageMismatch,
    /// Reply work id or attempt did not match the bound action.
    AttemptMismatch,
    /// Reply node or lease did not match the private manifest expectation.
    TargetMismatch,
    /// Terminal authorization was unexpectedly installed for anonymous work.
    TerminalAuthorizationMismatch,
    /// Signed write receipt no longer represented live storage.
    ReceiptOutsideWindow,
    /// Signed mutation timestamp was zero, regressive, or too far in future.
    MutationTimeInvalid,
    /// The bounded in-memory mutation count exceeded its integer domain.
    MutationCountOverflow,
    /// Final inventory was captured before an accepted mutation.
    InventoryPredatesMutation,
    /// Inventory verification rejected source or terminal evidence.
    Inventory(BlindVaultReplicaEvidenceError),
    /// Reconciliation evidence violated the workflow lifecycle contract.
    Workflow(BlindVaultReplicaWorkflowError),
}

impl<ClockError> fmt::Display for BlindVaultReplicaReconcileReplyPolicyError<ClockError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(_) => formatter.write_str("blind vault source clock failed"),
            Self::StageMismatch => {
                formatter.write_str("blind vault reconciliation reply stage mismatched")
            }
            Self::AttemptMismatch => {
                formatter.write_str("blind vault reconciliation reply attempt mismatched")
            }
            Self::TargetMismatch => {
                formatter.write_str("blind vault reconciliation reply target mismatched")
            }
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("blind vault reconciliation terminal authorization mismatched")
            }
            Self::ReceiptOutsideWindow => {
                formatter.write_str("blind vault reconciliation receipt is outside its live window")
            }
            Self::MutationTimeInvalid => {
                formatter.write_str("blind vault reconciliation mutation time is invalid")
            }
            Self::MutationCountOverflow => {
                formatter.write_str("blind vault reconciliation mutation count overflowed")
            }
            Self::InventoryPredatesMutation => formatter
                .write_str("blind vault reconciliation inventory predates accepted mutation"),
            Self::Inventory(error) => fmt::Display::fmt(error, formatter),
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<ClockError> fmt::Debug for BlindVaultReplicaReconcileReplyPolicyError<ClockError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(_) => formatter.write_str("Clock(<redacted>)"),
            Self::StageMismatch => formatter.write_str("StageMismatch"),
            Self::AttemptMismatch => formatter.write_str("AttemptMismatch"),
            Self::TargetMismatch => formatter.write_str("TargetMismatch"),
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("TerminalAuthorizationMismatch")
            }
            Self::ReceiptOutsideWindow => formatter.write_str("ReceiptOutsideWindow"),
            Self::MutationTimeInvalid => formatter.write_str("MutationTimeInvalid"),
            Self::MutationCountOverflow => formatter.write_str("MutationCountOverflow"),
            Self::InventoryPredatesMutation => formatter.write_str("InventoryPredatesMutation"),
            Self::Inventory(_) => formatter.write_str("Inventory(<redacted>)"),
            Self::Workflow(_) => formatter.write_str("Workflow(<redacted>)"),
        }
    }
}

impl<ClockError> Error for BlindVaultReplicaReconcileReplyPolicyError<ClockError>
where
    ClockError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Clock(error) => Some(error),
            Self::Inventory(error) => Some(error),
            Self::Workflow(error) => Some(error),
            Self::StageMismatch
            | Self::AttemptMismatch
            | Self::TargetMismatch
            | Self::TerminalAuthorizationMismatch
            | Self::ReceiptOutsideWindow
            | Self::MutationTimeInvalid
            | Self::MutationCountOverflow
            | Self::InventoryPredatesMutation => None,
        }
    }
}
