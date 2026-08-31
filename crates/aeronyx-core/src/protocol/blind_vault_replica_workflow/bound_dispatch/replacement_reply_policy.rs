// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/replacement_reply_policy.rs
// ============================================
//! Source-private reply policy for one Blind Vault replacement attempt.
//!
//! ## Creation Reason
//! Replacement replies arrive across admission, zero or more writes,
//! inventory, and old-lease retirement. Leaving those transitions to adapters
//! allowed otherwise valid receipts to be combined across attempts or stages.
//!
//! ## Main Functionality
//! - Implements `BlindVaultReplicaPrivateReplyPolicy` for replacement work.
//! - Binds every reply to one durable work id and attempt.
//! - Retains only credential-free verified admission state between replies.
//! - Verifies replacement inventory against one source-owned manifest.
//! - Requires a workflow-issued permit before accepting retirement evidence.
//! - Composes permit issuance, runtime transport, and reply verification.
//! - Produces typed, non-serializable lifecycle outcomes for the source.
//! - Seals completed evidence behind an unforgeable replacement-only type.
//!
//! ## Dependencies
//! - `request_bound_verifier.rs`: exact signed request/reply pairs.
//! - `evidence.rs`: admission, inventory, retirement, and action evidence.
//! - `BlindVaultReplicaVerificationClock`: replaceable source time boundary.
//!
//! ## Main Logical Flow
//! 1. Accept and distill one exact replacement admission reply.
//! 2. Accept zero or more exact put receipts for that node and lease.
//! 3. Verify one fresh matching inventory and expose the replacement proof.
//! 4. Install a permit issued by the active workflow for that proof.
//! 5. Accept only the exact old-node retirement and emit action evidence.
//!
//! ## Important Note For The Next Developer
//! - This state machine is source-private and intentionally not serializable.
//! - Durable recovery replays the complete idempotent effect sequence from
//!   stage zero; recreate this policy instead of checkpointing partial state.
//! - Never add a transition that accepts retirement before verified inventory.
//! - Never expose request, receipt, manifest, node, or lease values in Debug.
//!
//! Last Modified: v1.8.0-RuntimeAttemptPredicate - Removed an invalid const
//! promise from the work-id comparison used by runtime recovery.
//! v1.7.0-PrivacySafePolicyDiagnostics - Redacted generic clock,
//! transport, and source-private policy errors from standard diagnostics.
//! v1.6.0-AttemptBoundCompletion - Preserved exact replacement
//! attempt binding inside the emitted durable completion capability.
//! v1.5.0-WriteLeaseLifetime - Bound replacement write receipts
//! to the current admission time and signed lease expiry.
//! v1.4.0-AuthorizedRetirementDispatch - Composed workflow
//! authority and the permit-gated runtime send behind one typed operation.
//! v1.3.0-RetryablePermitComposition - Revalidated and reused
//! the same live replacement permit after retryable transport failure.
//! v1.2.0-WorkflowPermitComposition - Issued and installed one
//! exact active-execution permit for both policy and runtime use.
//! v1.1.0-CompletedReplacementCapability - Restricted durable
//! replacement completion to evidence emitted by the full policy state machine.
//! v1.0.0-ReplacementReplyPolicy - Initial typed replacement
//! reply state machine and workflow-permit composition.
//! ============================================

use std::{error::Error, fmt};

use super::super::{
    BlindVaultReplacementRetirementPermit, BlindVaultReplicaActionEvidence,
    BlindVaultReplicaExecution, BlindVaultReplicaWorkId, BlindVaultReplicaWorkflowError,
    BlindVaultVerifiedProvisionedReplica, BlindVaultVerifiedReplicaAdmission,
    BlindVaultVerifiedRetiredReplica,
};
use super::attempt_runtime::{
    BlindVaultReplicaTerminalAttemptError, BlindVaultReplicaTerminalAttemptRuntime,
};
use super::request_bound_verifier::{
    BlindVaultReplicaPrivateReplyPolicy, BlindVaultReplicaRequestBoundReply,
    BlindVaultReplicaRequestBoundReplyError, BlindVaultReplicaRequestBoundReplyVerifier,
    BlindVaultReplicaVerificationClock,
};
use super::send_sequence::{
    BlindVaultReplicaTerminalEffectTransport, BlindVaultReplicaTerminalSendContext,
};
use crate::protocol::blind_vault::{
    BlindVaultReplicaEvidenceError, BlindVaultReplicaManifestExpectation,
    BlindVaultVerifiedReplicaInventory,
};
use crate::protocol::onion::OnionRoutePurpose;

/// Successful source-private transition produced by one terminal reply.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaReplacementReplyOutcome {
    /// Anonymous replacement lease was accepted and credential state dropped.
    AdmissionAccepted,
    /// One exact replacement ciphertext write was durably acknowledged.
    ObjectStored,
    /// Replacement admission and complete live manifest were verified.
    ReplacementVerified(BlindVaultVerifiedProvisionedReplica),
    /// Old lease was durably retired under workflow authority.
    ReplacementCompleted(BlindVaultReplicaCompletedReplacement),
}

impl fmt::Debug for BlindVaultReplicaReplacementReplyOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AdmissionAccepted => "AdmissionAccepted",
            Self::ObjectStored => "ObjectStored",
            Self::ReplacementVerified(_) => "ReplacementVerified([REDACTED])",
            Self::ReplacementCompleted(_) => "ReplacementCompleted([REDACTED])",
        })
    }
}

/// Unforgeable completion capability for one fully verified replacement.
///
/// [BLIND-VAULT-COMPLETED-REPLACEMENT-CAPABILITY 2026-08-29 by Codex]
/// Public construction is intentionally unavailable. The replacement policy
/// creates this value only after matching inventory, workflow permit, exact
/// terminal authorization, and signed old-lease retirement all succeed.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaCompletedReplacement {
    evidence: BlindVaultReplicaActionEvidence,
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
}

impl BlindVaultReplicaCompletedReplacement {
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

impl fmt::Debug for BlindVaultReplicaCompletedReplacement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCompletedReplacement")
            .field("evidence", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct BlindVaultReplicaReplacementAttemptBinding {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
}

impl BlindVaultReplicaReplacementAttemptBinding {
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

impl fmt::Debug for BlindVaultReplicaReplacementAttemptBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaReplacementAttemptBinding")
            .field("attempt", &self.attempt)
            .field("work_id", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlindVaultReplicaReplacementReplyState {
    AwaitingAdmission,
    Populating {
        binding: BlindVaultReplicaReplacementAttemptBinding,
        admission: BlindVaultVerifiedReplicaAdmission,
    },
    ReplacementVerified {
        binding: BlindVaultReplicaReplacementAttemptBinding,
        replacement: BlindVaultVerifiedProvisionedReplica,
    },
    RetirementAuthorized {
        binding: BlindVaultReplicaReplacementAttemptBinding,
        permit: BlindVaultReplacementRetirementPermit,
    },
    Complete,
}

impl BlindVaultReplicaReplacementReplyState {
    const fn name(self) -> &'static str {
        match self {
            Self::AwaitingAdmission => "awaiting_admission",
            Self::Populating { .. } => "populating",
            Self::ReplacementVerified { .. } => "replacement_verified",
            Self::RetirementAuthorized { .. } => "retirement_authorized",
            Self::Complete => "complete",
        }
    }
}

/// Source-private replacement verification policy and lifecycle accumulator.
pub struct BlindVaultReplicaReplacementReplyPolicy<Clock> {
    expectation: BlindVaultReplicaManifestExpectation,
    clock: Clock,
    maximum_lease_ttl_ms: u64,
    maximum_receipt_age_ms: u64,
    maximum_future_clock_skew_ms: u64,
    state: BlindVaultReplicaReplacementReplyState,
}

impl<Clock> BlindVaultReplicaReplacementReplyPolicy<Clock> {
    /// Creates one policy without reading time or terminal state.
    pub fn new(
        expectation: BlindVaultReplicaManifestExpectation,
        clock: Clock,
        maximum_lease_ttl_ms: u64,
        maximum_receipt_age_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaReplacementReplyPolicyBuildError> {
        if maximum_lease_ttl_ms == 0 || maximum_receipt_age_ms == 0 {
            return Err(BlindVaultReplicaReplacementReplyPolicyBuildError::InvalidFreshnessPolicy);
        }
        Ok(Self {
            expectation,
            clock,
            maximum_lease_ttl_ms,
            maximum_receipt_age_ms,
            maximum_future_clock_skew_ms,
            state: BlindVaultReplicaReplacementReplyState::AwaitingAdmission,
        })
    }

    /// Installs workflow authority after matching replacement inventory.
    ///
    /// [BLIND-VAULT-REPLACEMENT-POLICY-PERMIT 2026-08-29 by Codex] The permit
    /// must bind the same work attempt and the exact replacement proof emitted
    /// by this policy. A mismatch leaves the verified state unchanged.
    pub fn authorize_retirement(
        &mut self,
        permit: BlindVaultReplacementRetirementPermit,
    ) -> Result<(), BlindVaultReplicaReplacementAuthorizationError> {
        let BlindVaultReplicaReplacementReplyState::ReplacementVerified {
            binding,
            replacement,
        } = self.state
        else {
            return Err(BlindVaultReplicaReplacementAuthorizationError::StageMismatch);
        };
        if permit.work_id() != binding.work_id
            || permit.attempt() != binding.attempt
            || permit.replacement_node_id() != replacement.node_id()
            || permit.replacement_lease_id() != replacement.lease_id()
        {
            return Err(BlindVaultReplicaReplacementAuthorizationError::PermitMismatch);
        }
        self.state =
            BlindVaultReplicaReplacementReplyState::RetirementAuthorized { binding, permit };
        Ok(())
    }

    /// Issues or revalidates one exact active-workflow retirement permit.
    ///
    /// [BLIND-VAULT-REPLACEMENT-PERMIT-COMPOSITION 2026-08-30 by Codex]
    /// The returned copy is the same capability retained by this policy and
    /// must be passed to the runtime's permit-gated retirement send. A retry
    /// reuses that capability after the workflow confirms both its attempt and
    /// replacement lease remain live; callers never reconstruct authority.
    pub fn authorize_retirement_from_execution(
        &mut self,
        execution: &BlindVaultReplicaExecution,
        authorized_at_ms: u64,
    ) -> Result<BlindVaultReplacementRetirementPermit, BlindVaultReplicaReplacementPermitIssueError>
    {
        match self.state {
            BlindVaultReplicaReplacementReplyState::ReplacementVerified {
                binding,
                replacement,
            } => {
                let permit = execution
                    .replacement_retirement_permit(binding.work_id, &replacement, authorized_at_ms)
                    .map_err(BlindVaultReplicaReplacementPermitIssueError::Workflow)?;
                self.authorize_retirement(permit)
                    .map_err(BlindVaultReplicaReplacementPermitIssueError::Authorization)?;
                Ok(permit)
            }
            BlindVaultReplicaReplacementReplyState::RetirementAuthorized { permit, .. } => {
                execution
                    .validate_replacement_retirement_permit(&permit, authorized_at_ms)
                    .map_err(BlindVaultReplicaReplacementPermitIssueError::Workflow)?;
                Ok(permit)
            }
            _ => Err(BlindVaultReplicaReplacementPermitIssueError::Authorization(
                BlindVaultReplicaReplacementAuthorizationError::StageMismatch,
            )),
        }
    }

    /// Whether verified replacement and retirement evidence are complete.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.state, BlindVaultReplicaReplacementReplyState::Complete)
    }
}

impl<Clock>
    BlindVaultReplicaRequestBoundReplyVerifier<BlindVaultReplicaReplacementReplyPolicy<Clock>>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    /// Authorizes, sends, opens, and verifies one old-lease retirement.
    ///
    /// [BLIND-VAULT-AUTHORIZED-RETIREMENT-DISPATCH 2026-08-30 by Codex] This
    /// is the only high-assurance replacement path that joins active workflow
    /// authority to runtime transport. The exact permit installed in private
    /// policy state is passed directly into the ordered send; adapters cannot
    /// substitute another work item, attempt, replacement, or old terminal.
    pub fn send_authorized_retirement<Transport>(
        &mut self,
        execution: &BlindVaultReplicaExecution,
        runtime: &mut BlindVaultReplicaTerminalAttemptRuntime<'_>,
        transport: &mut Transport,
        purpose: OnionRoutePurpose,
        payload: &[u8],
        now_ms: u64,
    ) -> Result<
        BlindVaultReplicaReplacementReplyOutcome,
        BlindVaultReplicaReplacementRetirementDispatchError<Transport::Error, Clock::Error>,
    >
    where
        Transport: BlindVaultReplicaTerminalEffectTransport,
        Transport::Response: AsRef<[u8]>,
    {
        let permit = self
            .policy_mut()
            .authorize_retirement_from_execution(execution, now_ms)
            .map_err(BlindVaultReplicaReplacementRetirementDispatchError::Permit)?;
        runtime
            .send_next_with_retirement_permit(transport, self, purpose, payload, &permit)
            .map_err(BlindVaultReplicaReplacementRetirementDispatchError::Attempt)
    }
}

impl<Clock> fmt::Debug for BlindVaultReplicaReplacementReplyPolicy<Clock> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaReplacementReplyPolicy")
            .field("clock", &std::any::type_name::<Clock>())
            .field("state", &self.state.name())
            .field("expectation", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl<Clock> BlindVaultReplicaPrivateReplyPolicy for BlindVaultReplicaReplacementReplyPolicy<Clock>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    type Output = BlindVaultReplicaReplacementReplyOutcome;
    type Error = BlindVaultReplicaReplacementReplyPolicyError<Clock::Error>;

    fn verify_private_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        _adapter_state: &[u8],
        reply: BlindVaultReplicaRequestBoundReply,
    ) -> Result<Self::Output, Self::Error> {
        match (self.state, reply) {
            (
                BlindVaultReplicaReplacementReplyState::AwaitingAdmission,
                BlindVaultReplicaRequestBoundReply::LeaseAccepted { request, receipt },
            ) => {
                require_unrestricted_terminal(context)?;
                require_replacement_target(&self.expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReplacementReplyPolicyError::Clock)?;
                let admission = BlindVaultVerifiedReplicaAdmission::verify(
                    &request,
                    &receipt,
                    now_ms,
                    self.maximum_lease_ttl_ms,
                    self.maximum_future_clock_skew_ms,
                )
                .map_err(BlindVaultReplicaReplacementReplyPolicyError::Workflow)?;
                self.state = BlindVaultReplicaReplacementReplyState::Populating {
                    binding: BlindVaultReplicaReplacementAttemptBinding::from_context(context),
                    admission,
                };
                Ok(BlindVaultReplicaReplacementReplyOutcome::AdmissionAccepted)
            }
            (
                BlindVaultReplicaReplacementReplyState::Populating { binding, admission },
                BlindVaultReplicaRequestBoundReply::ObjectStored { receipt, .. },
            ) => {
                require_same_attempt(binding, context)?;
                require_unrestricted_terminal(context)?;
                require_replacement_target(&self.expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReplacementReplyPolicyError::Clock)?;
                if now_ms == 0
                    || receipt.stored_until_ms <= now_ms
                    // [BLIND-VAULT-REPLACEMENT-WRITE-LIFETIME 2026-08-30 by Codex]
                    // A valid write receipt must remain inside the exact
                    // terminal-signed admission lifecycle it is populating.
                    || receipt.stored_until_ms > admission.lease_expires_at_ms()
                    || receipt.accepted_at_ms < admission.accepted_at_ms()
                    || (receipt.accepted_at_ms > now_ms
                        && receipt.accepted_at_ms - now_ms > self.maximum_future_clock_skew_ms)
                {
                    return Err(BlindVaultReplicaReplacementReplyPolicyError::ReceiptOutsideWindow);
                }
                Ok(BlindVaultReplicaReplacementReplyOutcome::ObjectStored)
            }
            (
                BlindVaultReplicaReplacementReplyState::Populating { binding, admission },
                BlindVaultReplicaRequestBoundReply::InventoryObserved { request, receipt },
            ) => {
                require_same_attempt(binding, context)?;
                require_unrestricted_terminal(context)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReplacementReplyPolicyError::Clock)?;
                let inventory = BlindVaultVerifiedReplicaInventory::verify(
                    &receipt,
                    &request,
                    &self.expectation,
                    now_ms,
                    self.maximum_receipt_age_ms,
                    self.maximum_future_clock_skew_ms,
                )
                .map_err(BlindVaultReplicaReplacementReplyPolicyError::Inventory)?;
                let replacement = BlindVaultVerifiedProvisionedReplica::verify_admitted_inventory(
                    &admission, &inventory, now_ms,
                )
                .map_err(BlindVaultReplicaReplacementReplyPolicyError::Workflow)?;
                self.state = BlindVaultReplicaReplacementReplyState::ReplacementVerified {
                    binding,
                    replacement,
                };
                Ok(BlindVaultReplicaReplacementReplyOutcome::ReplacementVerified(replacement))
            }
            (
                BlindVaultReplicaReplacementReplyState::RetirementAuthorized { binding, permit },
                BlindVaultReplicaRequestBoundReply::LeaseRetired { request, receipt },
            ) => {
                require_same_attempt(binding, context)?;
                if context.authorized_terminal_node_id() != Some(permit.replaced_node_id()) {
                    return Err(
                        BlindVaultReplicaReplacementReplyPolicyError::TerminalAuthorizationMismatch,
                    );
                }
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaReplacementReplyPolicyError::Clock)?;
                let retirement = BlindVaultVerifiedRetiredReplica::verify(
                    permit.replaced_node_id(),
                    &request,
                    &receipt,
                    now_ms,
                    self.maximum_future_clock_skew_ms,
                )
                .map_err(BlindVaultReplicaReplacementReplyPolicyError::Workflow)?;
                let evidence =
                    BlindVaultReplicaActionEvidence::replica_replaced_with_retirement_permit(
                        &permit, retirement, now_ms,
                    )
                    .map_err(BlindVaultReplicaReplacementReplyPolicyError::Workflow)?;
                self.state = BlindVaultReplicaReplacementReplyState::Complete;
                Ok(
                    BlindVaultReplicaReplacementReplyOutcome::ReplacementCompleted(
                        BlindVaultReplicaCompletedReplacement {
                            evidence,
                            work_id: binding.work_id,
                            attempt: binding.attempt,
                        },
                    ),
                )
            }
            _ => Err(BlindVaultReplicaReplacementReplyPolicyError::StageMismatch),
        }
    }
}

fn require_same_attempt<ClockError>(
    binding: BlindVaultReplicaReplacementAttemptBinding,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaReplacementReplyPolicyError<ClockError>> {
    binding
        .matches(context)
        .then_some(())
        .ok_or(BlindVaultReplicaReplacementReplyPolicyError::AttemptMismatch)
}

fn require_unrestricted_terminal<ClockError>(
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaReplacementReplyPolicyError<ClockError>> {
    context
        .authorized_terminal_node_id()
        .is_none()
        .then_some(())
        .ok_or(BlindVaultReplicaReplacementReplyPolicyError::TerminalAuthorizationMismatch)
}

fn require_replacement_target<ClockError>(
    expectation: &BlindVaultReplicaManifestExpectation,
    node_id: [u8; 32],
    lease_id: [u8; 32],
) -> Result<(), BlindVaultReplicaReplacementReplyPolicyError<ClockError>> {
    (expectation.node_id() == node_id && expectation.lease_id() == lease_id)
        .then_some(())
        .ok_or(BlindVaultReplicaReplacementReplyPolicyError::ReplacementTargetMismatch)
}

/// Invalid replacement policy configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaReplacementReplyPolicyBuildError {
    /// Lease TTL and receipt age bounds must both be nonzero.
    InvalidFreshnessPolicy,
}

impl fmt::Display for BlindVaultReplicaReplacementReplyPolicyBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("blind vault replacement reply freshness policy is invalid")
    }
}

impl Error for BlindVaultReplicaReplacementReplyPolicyBuildError {}

/// Failure to install workflow retirement authority into verified state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaReplacementAuthorizationError {
    /// Replacement inventory has not been verified or authority was installed.
    StageMismatch,
    /// Permit did not bind this attempt and exact verified replacement.
    PermitMismatch,
}

impl fmt::Display for BlindVaultReplicaReplacementAuthorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StageMismatch => {
                formatter.write_str("blind vault replacement is not ready for retirement authority")
            }
            Self::PermitMismatch => {
                formatter.write_str("blind vault replacement workflow permit mismatched")
            }
        }
    }
}

impl Error for BlindVaultReplicaReplacementAuthorizationError {}

/// Active-workflow permit issuance or local policy installation failure.
#[derive(Debug, PartialEq, Eq)]
pub enum BlindVaultReplicaReplacementPermitIssueError {
    /// Active workflow rejected permit issuance or event time.
    Workflow(BlindVaultReplicaWorkflowError),
    /// Verified policy state rejected the issued capability.
    Authorization(BlindVaultReplicaReplacementAuthorizationError),
}

impl fmt::Display for BlindVaultReplicaReplacementPermitIssueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
            Self::Authorization(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl Error for BlindVaultReplicaReplacementPermitIssueError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Workflow(error) => Some(error),
            Self::Authorization(error) => Some(error),
        }
    }
}

/// Permit composition, runtime transport, or retirement-reply failure.
pub enum BlindVaultReplicaReplacementRetirementDispatchError<TransportError, ClockError> {
    /// Active execution could not issue or revalidate exact retirement authority.
    Permit(BlindVaultReplicaReplacementPermitIssueError),
    /// Ordered transport or exact terminal reply processing failed.
    Attempt(
        BlindVaultReplicaTerminalAttemptError<
            TransportError,
            BlindVaultReplicaRequestBoundReplyError<
                BlindVaultReplicaReplacementReplyPolicyError<ClockError>,
            >,
        >,
    ),
}

impl<TransportError, ClockError> fmt::Display
    for BlindVaultReplicaReplacementRetirementDispatchError<TransportError, ClockError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Permit(error) => fmt::Display::fmt(error, formatter),
            Self::Attempt(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<TransportError, ClockError> fmt::Debug
    for BlindVaultReplicaReplacementRetirementDispatchError<TransportError, ClockError>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Permit(_) => formatter.write_str("Permit(<redacted>)"),
            Self::Attempt(_) => formatter.write_str("Attempt(<redacted>)"),
        }
    }
}

impl<TransportError, ClockError> Error
    for BlindVaultReplicaReplacementRetirementDispatchError<TransportError, ClockError>
where
    TransportError: Error + 'static,
    ClockError: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Permit(error) => Some(error),
            Self::Attempt(error) => Some(error),
        }
    }
}

/// Fail-closed private replacement reply or lifecycle transition failure.
pub enum BlindVaultReplicaReplacementReplyPolicyError<ClockError> {
    /// Source clock could not provide verification time.
    Clock(ClockError),
    /// Reply did not belong to the current replacement stage.
    StageMismatch,
    /// Reply work id or attempt did not match the admitted replacement.
    AttemptMismatch,
    /// Reply node or lease did not match the source manifest expectation.
    ReplacementTargetMismatch,
    /// Terminal authorization was present at the wrong stage or mismatched.
    TerminalAuthorizationMismatch,
    /// Signed write receipt was no longer live or too far in the future.
    ReceiptOutsideWindow,
    /// Inventory verification rejected source or terminal evidence.
    Inventory(BlindVaultReplicaEvidenceError),
    /// Admission, provisioning, retirement, or action evidence was invalid.
    Workflow(BlindVaultReplicaWorkflowError),
}

impl<ClockError> fmt::Display for BlindVaultReplicaReplacementReplyPolicyError<ClockError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(_) => formatter.write_str("blind vault source clock failed"),
            Self::StageMismatch => {
                formatter.write_str("blind vault replacement reply stage mismatched")
            }
            Self::AttemptMismatch => {
                formatter.write_str("blind vault replacement reply attempt mismatched")
            }
            Self::ReplacementTargetMismatch => {
                formatter.write_str("blind vault replacement reply target mismatched")
            }
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("blind vault replacement terminal authorization mismatched")
            }
            Self::ReceiptOutsideWindow => {
                formatter.write_str("blind vault replacement receipt is outside its live window")
            }
            Self::Inventory(error) => fmt::Display::fmt(error, formatter),
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<ClockError> fmt::Debug for BlindVaultReplicaReplacementReplyPolicyError<ClockError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(_) => formatter.write_str("Clock(<redacted>)"),
            Self::StageMismatch => formatter.write_str("StageMismatch"),
            Self::AttemptMismatch => formatter.write_str("AttemptMismatch"),
            Self::ReplacementTargetMismatch => formatter.write_str("ReplacementTargetMismatch"),
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("TerminalAuthorizationMismatch")
            }
            Self::ReceiptOutsideWindow => formatter.write_str("ReceiptOutsideWindow"),
            Self::Inventory(_) => formatter.write_str("Inventory(<redacted>)"),
            Self::Workflow(_) => formatter.write_str("Workflow(<redacted>)"),
        }
    }
}

impl<ClockError> Error for BlindVaultReplicaReplacementReplyPolicyError<ClockError>
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
            | Self::ReplacementTargetMismatch
            | Self::TerminalAuthorizationMismatch
            | Self::ReceiptOutsideWindow => None,
        }
    }
}
