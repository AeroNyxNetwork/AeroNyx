// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/renewal_reply_policy.rs
// ============================================
//! Source-private reply policy for one Blind Vault lease renewal.
//!
//! ## Creation Reason
//! A valid renewal receipt proves one terminal transition but must also match
//! the exact lease generation observed by the planner. Accepting a receipt for
//! another node, lease, or previous expiry would break compare-and-swap renewal
//! semantics and could resolve the wrong source-owned work item.
//!
//! ## Main Functionality
//! - Binds one policy to an exact `RenewLease` work item.
//! - Requires one single-effect, unrestricted terminal attempt.
//! - Enforces exact node, lease, and previous-expiry action matching.
//! - Verifies the signed renewal request/receipt and live new lease window.
//! - Emits an unforgeable completion capability for durable resolution.
//! - Redacts work, lease lifecycle, request, receipt, and evidence data.
//!
//! ## Dependencies
//! - `request_bound_verifier.rs`: exact signed request/reply and source clock.
//! - `evidence.rs`: renewal lifecycle and lease-window verification.
//! - `BlindVaultReplicaWorkItem`: immutable renewal compare-and-swap target.
//!
//! ## Main Logical Flow
//! 1. Capture exact planner node, lease, and previous lease expiry.
//! 2. Require the exact work id and a one-effect anonymous terminal attempt.
//! 3. Reject any receipt outside the planner's compare-and-swap generation.
//! 4. Verify exact signed renewal evidence and a still-live new expiry.
//! 5. Emit typed renewal completion for atomic durable resolution.
//!
//! ## Important Note For The Next Developer
//! - `expected_expires_at_ms` is a security boundary, not telemetry metadata.
//! - Never accept a later lease generation as implicit success for this action.
//! - This policy is source-private and intentionally not serializable.
//! - Never expose lease timestamps, identifiers, or evidence in Debug output.
//!
//! Last Modified: v1.0.0-RenewalReplyPolicy - Initial exact-generation,
//! single-effect, live-lease renewal reply policy.
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
use crate::protocol::blind_vault::BlindVaultReplicaAction;

/// Successful private transition for one lease-renewal attempt.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaRenewalReplyOutcome {
    /// Exact compare-and-swap renewal completed with a still-live new lease.
    RenewalCompleted(BlindVaultReplicaCompletedRenewal),
}

impl fmt::Debug for BlindVaultReplicaRenewalReplyOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RenewalCompleted(_) => formatter.write_str("RenewalCompleted([REDACTED])"),
        }
    }
}

/// Unforgeable completion capability for one exact lease renewal.
///
/// [BLIND-VAULT-COMPLETED-RENEWAL-CAPABILITY 2026-08-30 by Codex] Only the
/// full request-bound policy can create this value after exact-generation
/// renewal evidence verifies and the renewed lease remains live.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaCompletedRenewal {
    evidence: BlindVaultReplicaActionEvidence,
}

impl BlindVaultReplicaCompletedRenewal {
    pub(in crate::protocol::blind_vault_replica_workflow) const fn evidence(
        &self,
    ) -> &BlindVaultReplicaActionEvidence {
        &self.evidence
    }
}

impl fmt::Debug for BlindVaultReplicaCompletedRenewal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCompletedRenewal")
            .field("evidence", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlindVaultReplicaRenewalReplyState {
    AwaitingRenewal,
    Complete,
}

impl BlindVaultReplicaRenewalReplyState {
    const fn name(self) -> &'static str {
        match self {
            Self::AwaitingRenewal => "awaiting_renewal",
            Self::Complete => "complete",
        }
    }
}

/// Exact source-private verification policy for one lease renewal.
///
/// [BLIND-VAULT-RENEWAL-REPLY-POLICY 2026-08-30 by Codex] Node, lease, and
/// prior expiry form one immutable compare-and-swap target; valid evidence for
/// any other lease generation fails closed.
pub struct BlindVaultReplicaRenewalReplyPolicy<Clock> {
    expected_work_id: BlindVaultReplicaWorkId,
    expected_node_id: [u8; 32],
    expected_lease_id: [u8; 32],
    expected_expires_at_ms: u64,
    clock: Clock,
    maximum_future_clock_skew_ms: u64,
    state: BlindVaultReplicaRenewalReplyState,
}

impl<Clock> BlindVaultReplicaRenewalReplyPolicy<Clock> {
    /// Creates one policy bound to an exact renewal work item.
    pub fn new(
        work_item: &BlindVaultReplicaWorkItem,
        clock: Clock,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaRenewalReplyPolicyBuildError> {
        let BlindVaultReplicaAction::RenewLease {
            node_id,
            lease_id,
            expected_expires_at_ms,
        } = work_item.action()
        else {
            return Err(BlindVaultReplicaRenewalReplyPolicyBuildError::WrongAction);
        };
        if expected_expires_at_ms == 0 {
            return Err(BlindVaultReplicaRenewalReplyPolicyBuildError::InvalidLeaseGeneration);
        }
        Ok(Self {
            expected_work_id: work_item.id(),
            expected_node_id: node_id,
            expected_lease_id: lease_id,
            expected_expires_at_ms,
            clock,
            maximum_future_clock_skew_ms,
            state: BlindVaultReplicaRenewalReplyState::AwaitingRenewal,
        })
    }

    /// Whether exact live renewal evidence has been emitted.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.state, BlindVaultReplicaRenewalReplyState::Complete)
    }
}

impl<Clock> fmt::Debug for BlindVaultReplicaRenewalReplyPolicy<Clock> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRenewalReplyPolicy")
            .field("clock", &std::any::type_name::<Clock>())
            .field("state", &self.state.name())
            .field("work_id", &"[REDACTED]")
            .field("target", &"[REDACTED]")
            .field("lease_generation", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl<Clock> BlindVaultReplicaPrivateReplyPolicy for BlindVaultReplicaRenewalReplyPolicy<Clock>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    type Output = BlindVaultReplicaRenewalReplyOutcome;
    type Error = BlindVaultReplicaRenewalReplyPolicyError<Clock::Error>;

    fn verify_private_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        _adapter_state: &[u8],
        reply: BlindVaultReplicaRequestBoundReply,
    ) -> Result<Self::Output, Self::Error> {
        let BlindVaultReplicaRenewalReplyState::AwaitingRenewal = self.state else {
            return Err(BlindVaultReplicaRenewalReplyPolicyError::StageMismatch);
        };
        let BlindVaultReplicaRequestBoundReply::LeaseRenewed { request, receipt } = reply else {
            return Err(BlindVaultReplicaRenewalReplyPolicyError::StageMismatch);
        };
        require_single_effect_attempt(self.expected_work_id, context)?;
        if receipt.node_id != self.expected_node_id
            || receipt.lease_id != self.expected_lease_id
            || receipt.previous_expires_at_ms != self.expected_expires_at_ms
        {
            return Err(BlindVaultReplicaRenewalReplyPolicyError::ActionMismatch);
        }
        let now_ms = self
            .clock
            .now_ms()
            .map_err(BlindVaultReplicaRenewalReplyPolicyError::Clock)?;
        let evidence = BlindVaultReplicaActionEvidence::verify_renewal(
            &request,
            &receipt,
            now_ms,
            self.maximum_future_clock_skew_ms,
        )
        .map_err(BlindVaultReplicaRenewalReplyPolicyError::Workflow)?;
        self.state = BlindVaultReplicaRenewalReplyState::Complete;
        Ok(BlindVaultReplicaRenewalReplyOutcome::RenewalCompleted(
            BlindVaultReplicaCompletedRenewal { evidence },
        ))
    }
}

fn require_single_effect_attempt<ClockError>(
    expected_work_id: BlindVaultReplicaWorkId,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaRenewalReplyPolicyError<ClockError>> {
    if context.work_id() != expected_work_id || context.attempt() == 0 {
        return Err(BlindVaultReplicaRenewalReplyPolicyError::AttemptMismatch);
    }
    if context.effect_index() != 0 || context.effect_count() != 1 {
        return Err(BlindVaultReplicaRenewalReplyPolicyError::SequenceMismatch);
    }
    if context.authorized_terminal_node_id().is_some() {
        return Err(BlindVaultReplicaRenewalReplyPolicyError::TerminalAuthorizationMismatch);
    }
    Ok(())
}

/// Renewal policy construction failure before any reply is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaRenewalReplyPolicyBuildError {
    /// The supplied work item was not lease renewal.
    WrongAction,
    /// Planner renewal generation had no valid prior expiry.
    InvalidLeaseGeneration,
}

impl fmt::Display for BlindVaultReplicaRenewalReplyPolicyBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongAction => {
                formatter.write_str("blind vault renewal reply policy requires lease-renewal work")
            }
            Self::InvalidLeaseGeneration => {
                formatter.write_str("blind vault renewal lease generation is invalid")
            }
        }
    }
}

impl Error for BlindVaultReplicaRenewalReplyPolicyBuildError {}

/// Fail-closed renewal reply or lifecycle transition failure.
#[derive(Debug)]
pub enum BlindVaultReplicaRenewalReplyPolicyError<ClockError> {
    /// Source clock could not provide verification time.
    Clock(ClockError),
    /// Reply did not belong to the awaiting-renewal stage.
    StageMismatch,
    /// Reply work id did not match the bound planner action.
    AttemptMismatch,
    /// Runtime did not represent exactly one terminal effect.
    SequenceMismatch,
    /// Reply target or prior expiry mismatched planner compare-and-swap state.
    ActionMismatch,
    /// Terminal authorization was unexpectedly installed for anonymous work.
    TerminalAuthorizationMismatch,
    /// Renewal evidence violated signature, request, time, or lease semantics.
    Workflow(BlindVaultReplicaWorkflowError),
}

impl<ClockError: fmt::Display> fmt::Display
    for BlindVaultReplicaRenewalReplyPolicyError<ClockError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(error) => write!(formatter, "blind vault source clock failed: {error}"),
            Self::StageMismatch => {
                formatter.write_str("blind vault renewal reply stage mismatched")
            }
            Self::AttemptMismatch => {
                formatter.write_str("blind vault renewal reply attempt mismatched")
            }
            Self::SequenceMismatch => {
                formatter.write_str("blind vault renewal reply sequence mismatched")
            }
            Self::ActionMismatch => {
                formatter.write_str("blind vault renewal reply action mismatched")
            }
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("blind vault renewal terminal authorization mismatched")
            }
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<ClockError> Error for BlindVaultReplicaRenewalReplyPolicyError<ClockError>
where
    ClockError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Clock(error) => Some(error),
            Self::Workflow(error) => Some(error),
            Self::StageMismatch
            | Self::AttemptMismatch
            | Self::SequenceMismatch
            | Self::ActionMismatch
            | Self::TerminalAuthorizationMismatch => None,
        }
    }
}
