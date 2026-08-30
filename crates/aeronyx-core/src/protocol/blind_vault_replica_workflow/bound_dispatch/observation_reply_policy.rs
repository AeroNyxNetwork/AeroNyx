// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/observation_reply_policy.rs
// ============================================
//! Source-private reply policy for one Blind Vault observation retry.
//!
//! ## Creation Reason
//! A signed inventory reply proves terminal state, but it does not by itself
//! prove that the reply belongs to the exact planner action and durable attempt
//! currently awaiting evidence. Observation recovery also must preserve valid
//! divergence so the next fresh plan can schedule reconciliation.
//!
//! ## Main Functionality
//! - Binds one policy to an exact `RetryObservation` work item.
//! - Requires one single-effect, unrestricted terminal attempt.
//! - Verifies fresh signed inventory against a private manifest expectation.
//! - Accepts valid matching or divergent inventory while requiring a live lease.
//! - Emits an unforgeable completion capability for durable resolution.
//! - Redacts work, node, lease, manifest, request, receipt, and evidence data.
//!
//! ## Dependencies
//! - `request_bound_verifier.rs`: exact signed request/reply and source clock.
//! - `evidence.rs`: live observation-recovery workflow evidence.
//! - `BlindVaultReplicaWorkItem`: immutable action identity and target.
//!
//! ## Main Logical Flow
//! 1. Validate the private expectation against the planner target.
//! 2. Require the exact work id, attempt shape, and anonymous terminal mode.
//! 3. Verify the exact terminal-signed inventory request and response.
//! 4. Preserve valid divergence as evidence rather than transport failure.
//! 5. Emit typed observation completion for atomic durable resolution.
//!
//! ## Important Note For The Next Developer
//! - Do not require a matching manifest in this policy.
//! - Divergent inventory is a successful observation and triggers fresh planning.
//! - This policy is source-private and intentionally not serializable.
//! - Never expose private expectations or inventory values in telemetry.
//!
//! Last Modified: v1.0.0-ObservationReplyPolicy - Initial exact-action,
//! single-effect, freshness-bounded observation recovery policy.
//! ============================================

use std::{error::Error, fmt};

use super::super::{
    BlindVaultReplicaActionEvidence, BlindVaultReplicaWorkId, BlindVaultReplicaWorkItem,
    BlindVaultReplicaWorkflowError,
};
use super::request_bound_verifier::{
    verify_single_effect_reply_context, BlindVaultReplicaPrivateReplyPolicy,
    BlindVaultReplicaRequestBoundReply, BlindVaultReplicaSingleEffectContextError,
    BlindVaultReplicaVerificationClock,
};
use super::send_sequence::BlindVaultReplicaTerminalSendContext;
use crate::protocol::blind_vault::{
    BlindVaultReplicaAction, BlindVaultReplicaEvidenceError, BlindVaultReplicaManifestExpectation,
    BlindVaultVerifiedReplicaInventory,
};

/// Successful private transition for one observation-retry attempt.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaObservationReplyOutcome {
    /// Fresh live inventory was verified for the exact retry target.
    ObservationCompleted(BlindVaultReplicaCompletedObservation),
}

impl fmt::Debug for BlindVaultReplicaObservationReplyOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObservationCompleted(_) => {
                formatter.write_str("ObservationCompleted([REDACTED])")
            }
        }
    }
}

/// Unforgeable completion capability for one exact observation retry.
///
/// [BLIND-VAULT-COMPLETED-OBSERVATION-CAPABILITY 2026-08-30 by Codex] Only
/// the full request-bound policy can create this value after live terminal
/// inventory has been verified for the exact planner target and attempt.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaCompletedObservation {
    evidence: BlindVaultReplicaActionEvidence,
}

impl BlindVaultReplicaCompletedObservation {
    pub(in crate::protocol::blind_vault_replica_workflow) const fn evidence(
        &self,
    ) -> &BlindVaultReplicaActionEvidence {
        &self.evidence
    }
}

impl fmt::Debug for BlindVaultReplicaCompletedObservation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCompletedObservation")
            .field("evidence", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlindVaultReplicaObservationReplyState {
    AwaitingInventory,
    Complete,
}

impl BlindVaultReplicaObservationReplyState {
    const fn name(self) -> &'static str {
        match self {
            Self::AwaitingInventory => "awaiting_inventory",
            Self::Complete => "complete",
        }
    }
}

/// Exact source-private verification policy for one observation retry.
///
/// [BLIND-VAULT-OBSERVATION-REPLY-POLICY 2026-08-30 by Codex] This policy
/// deliberately accepts valid divergent inventory. A successful observation
/// restores evidence availability; a mandatory fresh plan decides repair.
pub struct BlindVaultReplicaObservationReplyPolicy<Clock> {
    expected_work_id: BlindVaultReplicaWorkId,
    expectation: BlindVaultReplicaManifestExpectation,
    clock: Clock,
    maximum_receipt_age_ms: u64,
    maximum_future_clock_skew_ms: u64,
    state: BlindVaultReplicaObservationReplyState,
}

impl<Clock> BlindVaultReplicaObservationReplyPolicy<Clock> {
    /// Creates one policy after exact action/target compatibility checks.
    pub fn new(
        work_item: &BlindVaultReplicaWorkItem,
        expectation: BlindVaultReplicaManifestExpectation,
        clock: Clock,
        maximum_receipt_age_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaObservationReplyPolicyBuildError> {
        let BlindVaultReplicaAction::RetryObservation { node_id, lease_id } = work_item.action()
        else {
            return Err(BlindVaultReplicaObservationReplyPolicyBuildError::WrongAction);
        };
        if expectation.node_id() != node_id || expectation.lease_id() != lease_id {
            return Err(
                BlindVaultReplicaObservationReplyPolicyBuildError::ExpectationTargetMismatch,
            );
        }
        if maximum_receipt_age_ms == 0 {
            return Err(BlindVaultReplicaObservationReplyPolicyBuildError::InvalidFreshnessPolicy);
        }
        Ok(Self {
            expected_work_id: work_item.id(),
            expectation,
            clock,
            maximum_receipt_age_ms,
            maximum_future_clock_skew_ms,
            state: BlindVaultReplicaObservationReplyState::AwaitingInventory,
        })
    }

    /// Whether live observation evidence has been emitted.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.state, BlindVaultReplicaObservationReplyState::Complete)
    }
}

impl<Clock> fmt::Debug for BlindVaultReplicaObservationReplyPolicy<Clock> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaObservationReplyPolicy")
            .field("clock", &std::any::type_name::<Clock>())
            .field("state", &self.state.name())
            .field("work_id", &"[REDACTED]")
            .field("expectation", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl<Clock> BlindVaultReplicaPrivateReplyPolicy for BlindVaultReplicaObservationReplyPolicy<Clock>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    type Output = BlindVaultReplicaObservationReplyOutcome;
    type Error = BlindVaultReplicaObservationReplyPolicyError<Clock::Error>;

    fn verify_private_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        _adapter_state: &[u8],
        reply: BlindVaultReplicaRequestBoundReply,
    ) -> Result<Self::Output, Self::Error> {
        let BlindVaultReplicaObservationReplyState::AwaitingInventory = self.state else {
            return Err(BlindVaultReplicaObservationReplyPolicyError::StageMismatch);
        };
        let BlindVaultReplicaRequestBoundReply::InventoryObserved { request, receipt } = reply
        else {
            return Err(BlindVaultReplicaObservationReplyPolicyError::StageMismatch);
        };
        verify_single_effect_reply_context(self.expected_work_id, context)
            .map_err(map_single_effect_context_error)?;
        let now_ms = self
            .clock
            .now_ms()
            .map_err(BlindVaultReplicaObservationReplyPolicyError::Clock)?;
        let inventory = BlindVaultVerifiedReplicaInventory::verify(
            &receipt,
            &request,
            &self.expectation,
            now_ms,
            self.maximum_receipt_age_ms,
            self.maximum_future_clock_skew_ms,
        )
        .map_err(BlindVaultReplicaObservationReplyPolicyError::Inventory)?;
        let evidence = BlindVaultReplicaActionEvidence::observation_recovered(inventory, now_ms)
            .map_err(BlindVaultReplicaObservationReplyPolicyError::Workflow)?;
        self.state = BlindVaultReplicaObservationReplyState::Complete;
        Ok(
            BlindVaultReplicaObservationReplyOutcome::ObservationCompleted(
                BlindVaultReplicaCompletedObservation { evidence },
            ),
        )
    }
}

fn map_single_effect_context_error<ClockError>(
    error: BlindVaultReplicaSingleEffectContextError,
) -> BlindVaultReplicaObservationReplyPolicyError<ClockError> {
    match error {
        BlindVaultReplicaSingleEffectContextError::AttemptMismatch => {
            BlindVaultReplicaObservationReplyPolicyError::AttemptMismatch
        }
        BlindVaultReplicaSingleEffectContextError::SequenceMismatch => {
            BlindVaultReplicaObservationReplyPolicyError::SequenceMismatch
        }
        BlindVaultReplicaSingleEffectContextError::TerminalAuthorizationMismatch => {
            BlindVaultReplicaObservationReplyPolicyError::TerminalAuthorizationMismatch
        }
    }
}

/// Observation policy construction failure before any reply is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaObservationReplyPolicyBuildError {
    /// The supplied work item was not an observation retry.
    WrongAction,
    /// Private expectation did not target the planner node and lease.
    ExpectationTargetMismatch,
    /// Inventory receipt freshness policy was zero.
    InvalidFreshnessPolicy,
}

impl fmt::Display for BlindVaultReplicaObservationReplyPolicyBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongAction => formatter
                .write_str("blind vault observation reply policy requires observation work"),
            Self::ExpectationTargetMismatch => {
                formatter.write_str("blind vault observation expectation target mismatched action")
            }
            Self::InvalidFreshnessPolicy => {
                formatter.write_str("blind vault observation freshness policy is invalid")
            }
        }
    }
}

impl Error for BlindVaultReplicaObservationReplyPolicyBuildError {}

/// Fail-closed observation reply or lifecycle transition failure.
#[derive(Debug)]
pub enum BlindVaultReplicaObservationReplyPolicyError<ClockError> {
    /// Source clock could not provide verification time.
    Clock(ClockError),
    /// Reply did not belong to the awaiting-inventory stage.
    StageMismatch,
    /// Reply work id did not match the bound planner action.
    AttemptMismatch,
    /// Runtime did not represent exactly one terminal effect.
    SequenceMismatch,
    /// Terminal authorization was unexpectedly installed for anonymous work.
    TerminalAuthorizationMismatch,
    /// Inventory verification rejected source or terminal evidence.
    Inventory(BlindVaultReplicaEvidenceError),
    /// Observation evidence violated workflow lifecycle requirements.
    Workflow(BlindVaultReplicaWorkflowError),
}

impl<ClockError: fmt::Display> fmt::Display
    for BlindVaultReplicaObservationReplyPolicyError<ClockError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(error) => write!(formatter, "blind vault source clock failed: {error}"),
            Self::StageMismatch => {
                formatter.write_str("blind vault observation reply stage mismatched")
            }
            Self::AttemptMismatch => {
                formatter.write_str("blind vault observation reply attempt mismatched")
            }
            Self::SequenceMismatch => {
                formatter.write_str("blind vault observation reply sequence mismatched")
            }
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("blind vault observation terminal authorization mismatched")
            }
            Self::Inventory(error) => fmt::Display::fmt(error, formatter),
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<ClockError> Error for BlindVaultReplicaObservationReplyPolicyError<ClockError>
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
            | Self::SequenceMismatch
            | Self::TerminalAuthorizationMismatch => None,
        }
    }
}
