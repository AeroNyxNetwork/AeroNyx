// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/provisioning_reply_policy.rs
// ============================================
//! Source-private reply policy for one aggregate Blind Vault provisioning attempt.
//!
//! ## Creation Reason
//! Provisioning repeats admission, writes, and inventory verification for an
//! exact planner-authorized replica count. Adapter-managed accumulation could
//! mix valid replies across replica groups, attempts, or manifests.
//!
//! ## Main Functionality
//! - Binds one policy to an exact `ProvisionReplicas` work item and count.
//! - Enforces ordered admission/write/inventory stages for every new replica.
//! - Retains only credential-free admission and verified lifecycle evidence.
//! - Rejects duplicate node or lease identities across the aggregate action.
//! - Emits an unforgeable completion capability only while all leases are live.
//! - Redacts work, node, lease, manifest, request, and receipt data from Debug.
//!
//! ## Dependencies
//! - `request_bound_verifier.rs`: exact signed request/reply pairs and clock.
//! - `evidence.rs`: admission, inventory, and aggregate action evidence.
//! - `BlindVaultReplicaWorkItem`: immutable action identity and replica count.
//!
//! ## Main Logical Flow
//! 1. Require the first admission to bind the configured work id and attempt.
//! 2. Distill its signed admission into a credential-free lifecycle proof.
//! 3. Accept only writes for that replica's exact node and lease.
//! 4. Verify one fresh complete inventory against its private manifest.
//! 5. Repeat for the exact planner-authorized count without identity reuse.
//! 6. Emit aggregate evidence only if every verified lease remains live.
//!
//! ## Important Note For The Next Developer
//! - This source-private state is intentionally not serializable.
//! - Recovery replays the complete idempotent effect sequence from stage zero.
//! - Never permit interleaved replica groups or caller-selected group indexes.
//! - Never expose private expectations or replica topology in telemetry.
//!
//! Last Modified: v1.2.0-PrivacySafeClockDiagnostics - Redacted generic clock
//! and source-private policy errors from standard diagnostics.
//! v1.1.0-AttemptBoundCompletion - Preserved exact aggregate
//! attempt binding inside the emitted durable completion capability.
//! v1.0.0-ProvisioningReplyPolicy - Initial bounded aggregate
//! provisioning reply state machine and completion capability.
//! ============================================

use std::{collections::BTreeSet, error::Error, fmt};

use super::super::{
    BlindVaultReplicaActionEvidence, BlindVaultReplicaDispatchContract, BlindVaultReplicaWorkId,
    BlindVaultReplicaWorkItem, BlindVaultReplicaWorkflowError,
    BlindVaultVerifiedProvisionedReplica, BlindVaultVerifiedReplicaAdmission,
};
use super::request_bound_verifier::{
    BlindVaultReplicaPrivateReplyPolicy, BlindVaultReplicaRequestBoundReply,
    BlindVaultReplicaVerificationClock,
};
use super::send_sequence::BlindVaultReplicaTerminalSendContext;
use crate::protocol::blind_vault::{
    BlindVaultReplicaEvidenceError, BlindVaultReplicaManifestExpectation,
    BlindVaultVerifiedReplicaInventory, MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS,
};

/// Successful private transition for one aggregate provisioning attempt.
#[derive(Clone, PartialEq, Eq)]
pub enum BlindVaultReplicaProvisioningReplyOutcome {
    /// The current anonymous lease was accepted and credentials were dropped.
    AdmissionAccepted { replica_number: u8 },
    /// One exact ciphertext write was acknowledged for the current replica.
    ObjectStored { replica_number: u8 },
    /// One complete replica was verified and another group remains.
    ReplicaVerified { completed: u8, total: u8 },
    /// Every planner-authorized replica was independently verified and live.
    ProvisioningCompleted(BlindVaultReplicaCompletedProvisioning),
}

impl fmt::Debug for BlindVaultReplicaProvisioningReplyOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdmissionAccepted { replica_number } => formatter
                .debug_struct("AdmissionAccepted")
                .field("replica_number", replica_number)
                .finish(),
            Self::ObjectStored { replica_number } => formatter
                .debug_struct("ObjectStored")
                .field("replica_number", replica_number)
                .finish(),
            Self::ReplicaVerified { completed, total } => formatter
                .debug_struct("ReplicaVerified")
                .field("completed", completed)
                .field("total", total)
                .finish(),
            Self::ProvisioningCompleted(_) => {
                formatter.write_str("ProvisioningCompleted([REDACTED])")
            }
        }
    }
}

/// Unforgeable completion capability for one aggregate provisioning action.
///
/// [BLIND-VAULT-COMPLETED-PROVISIONING-CAPABILITY 2026-08-30 by Codex]
/// Construction follows the complete ordered reply policy and is unavailable
/// to adapters that merely possess individual valid replica proofs.
#[derive(Clone, PartialEq, Eq)]
pub struct BlindVaultReplicaCompletedProvisioning {
    evidence: BlindVaultReplicaActionEvidence,
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
}

impl BlindVaultReplicaCompletedProvisioning {
    pub(in crate::protocol::blind_vault_replica_workflow) const fn evidence(
        &self,
    ) -> &BlindVaultReplicaActionEvidence {
        &self.evidence
    }

    pub(in crate::protocol::blind_vault_replica_workflow) const fn matches_attempt(
        &self,
        work_id: BlindVaultReplicaWorkId,
        attempt: u8,
    ) -> bool {
        self.work_id == work_id && self.attempt == attempt
    }
}

impl fmt::Debug for BlindVaultReplicaCompletedProvisioning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaCompletedProvisioning")
            .field("evidence", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct BlindVaultReplicaProvisioningAttemptBinding {
    work_id: BlindVaultReplicaWorkId,
    attempt: u8,
}

impl BlindVaultReplicaProvisioningAttemptBinding {
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

impl fmt::Debug for BlindVaultReplicaProvisioningAttemptBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaProvisioningAttemptBinding")
            .field("attempt", &self.attempt)
            .field("work_id", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlindVaultReplicaProvisioningReplyState {
    AwaitingAdmission {
        binding: Option<BlindVaultReplicaProvisioningAttemptBinding>,
        replica_index: u8,
    },
    Populating {
        binding: BlindVaultReplicaProvisioningAttemptBinding,
        replica_index: u8,
        admission: BlindVaultVerifiedReplicaAdmission,
    },
    Complete,
}

impl BlindVaultReplicaProvisioningReplyState {
    const fn name(self) -> &'static str {
        match self {
            Self::AwaitingAdmission { .. } => "awaiting_admission",
            Self::Populating { .. } => "populating",
            Self::Complete => "complete",
        }
    }
}

/// Ordered private verification policy for one aggregate provisioning action.
pub struct BlindVaultReplicaProvisioningReplyPolicy<Clock> {
    expected_work_id: BlindVaultReplicaWorkId,
    expected_replicas: u8,
    expectations: Vec<BlindVaultReplicaManifestExpectation>,
    verified_replicas: Vec<BlindVaultVerifiedProvisionedReplica>,
    clock: Clock,
    maximum_lease_ttl_ms: u64,
    maximum_receipt_age_ms: u64,
    maximum_future_clock_skew_ms: u64,
    state: BlindVaultReplicaProvisioningReplyState,
}

impl<Clock> BlindVaultReplicaProvisioningReplyPolicy<Clock> {
    /// Creates one policy bound to an exact aggregate provisioning work item.
    pub fn new(
        work_item: &BlindVaultReplicaWorkItem,
        expectations: Vec<BlindVaultReplicaManifestExpectation>,
        clock: Clock,
        maximum_lease_ttl_ms: u64,
        maximum_receipt_age_ms: u64,
        maximum_future_clock_skew_ms: u64,
    ) -> Result<Self, BlindVaultReplicaProvisioningReplyPolicyBuildError> {
        let BlindVaultReplicaDispatchContract::ProvisionReplicas { count, .. } =
            work_item.dispatch_contract()
        else {
            return Err(BlindVaultReplicaProvisioningReplyPolicyBuildError::WrongAction);
        };
        if count == 0
            || usize::from(count) > MAX_BLIND_VAULT_REPLICA_PLAN_MEMBERS
            || expectations.len() != usize::from(count)
        {
            return Err(
                BlindVaultReplicaProvisioningReplyPolicyBuildError::ExpectationCountMismatch,
            );
        }
        if maximum_lease_ttl_ms == 0 || maximum_receipt_age_ms == 0 {
            return Err(BlindVaultReplicaProvisioningReplyPolicyBuildError::InvalidFreshnessPolicy);
        }
        let mut node_ids = BTreeSet::new();
        let mut lease_ids = BTreeSet::new();
        if expectations.iter().any(|expectation| {
            !node_ids.insert(expectation.node_id()) || !lease_ids.insert(expectation.lease_id())
        }) {
            return Err(BlindVaultReplicaProvisioningReplyPolicyBuildError::DuplicateExpectation);
        }
        Ok(Self {
            expected_work_id: work_item.id(),
            expected_replicas: count,
            expectations,
            verified_replicas: Vec::with_capacity(usize::from(count)),
            clock,
            maximum_lease_ttl_ms,
            maximum_receipt_age_ms,
            maximum_future_clock_skew_ms,
            state: BlindVaultReplicaProvisioningReplyState::AwaitingAdmission {
                binding: None,
                replica_index: 0,
            },
        })
    }

    /// Exact number of replicas authorized by the immutable planner action.
    #[must_use]
    pub const fn expected_replicas(&self) -> u8 {
        self.expected_replicas
    }

    /// Number of independently verified replicas accumulated so far.
    #[must_use]
    pub fn verified_replicas(&self) -> u8 {
        u8::try_from(self.verified_replicas.len()).unwrap_or(0)
    }

    /// Whether complete aggregate evidence has been emitted.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(
            self.state,
            BlindVaultReplicaProvisioningReplyState::Complete
        )
    }

    fn expectation(&self, replica_index: u8) -> Option<BlindVaultReplicaManifestExpectation> {
        self.expectations.get(usize::from(replica_index)).copied()
    }
}

impl<Clock> fmt::Debug for BlindVaultReplicaProvisioningReplyPolicy<Clock> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaProvisioningReplyPolicy")
            .field("clock", &std::any::type_name::<Clock>())
            .field("state", &self.state.name())
            .field("expected_replicas", &self.expectations.len())
            .field("verified_replicas", &self.verified_replicas.len())
            .field("work_id", &"[REDACTED]")
            .field("expectations", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl<Clock> BlindVaultReplicaPrivateReplyPolicy for BlindVaultReplicaProvisioningReplyPolicy<Clock>
where
    Clock: BlindVaultReplicaVerificationClock,
{
    type Output = BlindVaultReplicaProvisioningReplyOutcome;
    type Error = BlindVaultReplicaProvisioningReplyPolicyError<Clock::Error>;

    fn verify_private_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        _adapter_state: &[u8],
        reply: BlindVaultReplicaRequestBoundReply,
    ) -> Result<Self::Output, Self::Error> {
        match (self.state, reply) {
            (
                BlindVaultReplicaProvisioningReplyState::AwaitingAdmission {
                    binding,
                    replica_index,
                },
                BlindVaultReplicaRequestBoundReply::LeaseAccepted { request, receipt },
            ) => {
                require_unrestricted_terminal(context)?;
                let binding = require_or_create_binding(self.expected_work_id, binding, context)?;
                let expectation = self
                    .expectation(replica_index)
                    .ok_or(BlindVaultReplicaProvisioningReplyPolicyError::StageMismatch)?;
                require_replica_target(&expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Clock)?;
                let admission = BlindVaultVerifiedReplicaAdmission::verify(
                    &request,
                    &receipt,
                    now_ms,
                    self.maximum_lease_ttl_ms,
                    self.maximum_future_clock_skew_ms,
                )
                .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Workflow)?;
                self.state = BlindVaultReplicaProvisioningReplyState::Populating {
                    binding,
                    replica_index,
                    admission,
                };
                Ok(
                    BlindVaultReplicaProvisioningReplyOutcome::AdmissionAccepted {
                        replica_number: replica_index.saturating_add(1),
                    },
                )
            }
            (
                BlindVaultReplicaProvisioningReplyState::Populating {
                    binding,
                    replica_index,
                    admission,
                },
                BlindVaultReplicaRequestBoundReply::ObjectStored { receipt, .. },
            ) => {
                require_same_attempt(binding, context)?;
                require_unrestricted_terminal(context)?;
                let expectation = self
                    .expectation(replica_index)
                    .ok_or(BlindVaultReplicaProvisioningReplyPolicyError::StageMismatch)?;
                require_replica_target(&expectation, receipt.node_id, receipt.lease_id)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Clock)?;
                if now_ms == 0
                    || receipt.stored_until_ms <= now_ms
                    || receipt.stored_until_ms > admission.lease_expires_at_ms()
                    || receipt.accepted_at_ms < admission.accepted_at_ms()
                    || (receipt.accepted_at_ms > now_ms
                        && receipt.accepted_at_ms - now_ms > self.maximum_future_clock_skew_ms)
                {
                    return Err(
                        BlindVaultReplicaProvisioningReplyPolicyError::ReceiptOutsideWindow,
                    );
                }
                Ok(BlindVaultReplicaProvisioningReplyOutcome::ObjectStored {
                    replica_number: replica_index.saturating_add(1),
                })
            }
            (
                BlindVaultReplicaProvisioningReplyState::Populating {
                    binding,
                    replica_index,
                    admission,
                },
                BlindVaultReplicaRequestBoundReply::InventoryObserved { request, receipt },
            ) => {
                require_same_attempt(binding, context)?;
                require_unrestricted_terminal(context)?;
                let expectation = self
                    .expectation(replica_index)
                    .ok_or(BlindVaultReplicaProvisioningReplyPolicyError::StageMismatch)?;
                let now_ms = self
                    .clock
                    .now_ms()
                    .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Clock)?;
                let inventory = BlindVaultVerifiedReplicaInventory::verify(
                    &receipt,
                    &request,
                    &expectation,
                    now_ms,
                    self.maximum_receipt_age_ms,
                    self.maximum_future_clock_skew_ms,
                )
                .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Inventory)?;
                let replica = BlindVaultVerifiedProvisionedReplica::verify_admitted_inventory(
                    &admission, &inventory, now_ms,
                )
                .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Workflow)?;
                if self.verified_replicas.iter().any(|verified| {
                    verified.node_id() == replica.node_id()
                        || verified.lease_id() == replica.lease_id()
                }) {
                    return Err(BlindVaultReplicaProvisioningReplyPolicyError::DuplicateReplica);
                }

                let mut verified_replicas = self.verified_replicas.clone();
                verified_replicas.push(replica);
                let completed = verified_replicas.len() == self.expectations.len();
                if completed {
                    let evidence = BlindVaultReplicaActionEvidence::replicas_provisioned(
                        &verified_replicas,
                        self.expected_replicas(),
                        now_ms,
                    )
                    .map_err(BlindVaultReplicaProvisioningReplyPolicyError::Workflow)?;
                    self.verified_replicas = verified_replicas;
                    self.state = BlindVaultReplicaProvisioningReplyState::Complete;
                    Ok(
                        BlindVaultReplicaProvisioningReplyOutcome::ProvisioningCompleted(
                            BlindVaultReplicaCompletedProvisioning {
                                evidence,
                                work_id: binding.work_id,
                                attempt: binding.attempt,
                            },
                        ),
                    )
                } else {
                    self.verified_replicas = verified_replicas;
                    self.state = BlindVaultReplicaProvisioningReplyState::AwaitingAdmission {
                        binding: Some(binding),
                        replica_index: replica_index.saturating_add(1),
                    };
                    Ok(BlindVaultReplicaProvisioningReplyOutcome::ReplicaVerified {
                        completed: self.verified_replicas(),
                        total: self.expected_replicas(),
                    })
                }
            }
            _ => Err(BlindVaultReplicaProvisioningReplyPolicyError::StageMismatch),
        }
    }
}

fn require_or_create_binding<ClockError>(
    expected_work_id: BlindVaultReplicaWorkId,
    binding: Option<BlindVaultReplicaProvisioningAttemptBinding>,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<
    BlindVaultReplicaProvisioningAttemptBinding,
    BlindVaultReplicaProvisioningReplyPolicyError<ClockError>,
> {
    let candidate = binding
        .unwrap_or_else(|| BlindVaultReplicaProvisioningAttemptBinding::from_context(context));
    if context.work_id() != expected_work_id || !candidate.matches(context) {
        return Err(BlindVaultReplicaProvisioningReplyPolicyError::AttemptMismatch);
    }
    Ok(candidate)
}

fn require_same_attempt<ClockError>(
    binding: BlindVaultReplicaProvisioningAttemptBinding,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaProvisioningReplyPolicyError<ClockError>> {
    binding
        .matches(context)
        .then_some(())
        .ok_or(BlindVaultReplicaProvisioningReplyPolicyError::AttemptMismatch)
}

fn require_unrestricted_terminal<ClockError>(
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaProvisioningReplyPolicyError<ClockError>> {
    context
        .authorized_terminal_node_id()
        .is_none()
        .then_some(())
        .ok_or(BlindVaultReplicaProvisioningReplyPolicyError::TerminalAuthorizationMismatch)
}

fn require_replica_target<ClockError>(
    expectation: &BlindVaultReplicaManifestExpectation,
    node_id: [u8; 32],
    lease_id: [u8; 32],
) -> Result<(), BlindVaultReplicaProvisioningReplyPolicyError<ClockError>> {
    (expectation.node_id() == node_id && expectation.lease_id() == lease_id)
        .then_some(())
        .ok_or(BlindVaultReplicaProvisioningReplyPolicyError::ReplicaTargetMismatch)
}

/// Provisioning policy construction failure before any reply is accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlindVaultReplicaProvisioningReplyPolicyBuildError {
    /// The supplied work item was not aggregate provisioning.
    WrongAction,
    /// Private expectations did not match the bounded planner count.
    ExpectationCountMismatch,
    /// Lease or receipt freshness policy was zero.
    InvalidFreshnessPolicy,
    /// Two private expectations reused a node or lease identity.
    DuplicateExpectation,
}

impl fmt::Display for BlindVaultReplicaProvisioningReplyPolicyBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongAction => formatter
                .write_str("blind vault provisioning reply policy requires provisioning work"),
            Self::ExpectationCountMismatch => formatter
                .write_str("blind vault provisioning expectations do not match planner count"),
            Self::InvalidFreshnessPolicy => {
                formatter.write_str("blind vault provisioning freshness policy is invalid")
            }
            Self::DuplicateExpectation => {
                formatter.write_str("blind vault provisioning expectations reuse a node or lease")
            }
        }
    }
}

impl Error for BlindVaultReplicaProvisioningReplyPolicyBuildError {}

/// Fail-closed aggregate provisioning reply or lifecycle transition failure.
pub enum BlindVaultReplicaProvisioningReplyPolicyError<ClockError> {
    /// Source clock could not provide verification time.
    Clock(ClockError),
    /// Reply did not belong to the current replica group or stage.
    StageMismatch,
    /// Reply work id or attempt did not match the aggregate action.
    AttemptMismatch,
    /// Reply node or lease did not match the current private expectation.
    ReplicaTargetMismatch,
    /// Two independently provisioned groups reused a node or lease identity.
    DuplicateReplica,
    /// Terminal authorization was unexpectedly installed for anonymous work.
    TerminalAuthorizationMismatch,
    /// Signed write receipt was no longer live or too far in the future.
    ReceiptOutsideWindow,
    /// Inventory verification rejected source or terminal evidence.
    Inventory(BlindVaultReplicaEvidenceError),
    /// Admission, provisioning, or aggregate evidence was invalid.
    Workflow(BlindVaultReplicaWorkflowError),
}

impl<ClockError> fmt::Display for BlindVaultReplicaProvisioningReplyPolicyError<ClockError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(_) => formatter.write_str("blind vault source clock failed"),
            Self::StageMismatch => {
                formatter.write_str("blind vault provisioning reply stage mismatched")
            }
            Self::AttemptMismatch => {
                formatter.write_str("blind vault provisioning reply attempt mismatched")
            }
            Self::ReplicaTargetMismatch => {
                formatter.write_str("blind vault provisioning reply target mismatched")
            }
            Self::DuplicateReplica => {
                formatter.write_str("blind vault provisioning reused a node or lease")
            }
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("blind vault provisioning terminal authorization mismatched")
            }
            Self::ReceiptOutsideWindow => {
                formatter.write_str("blind vault provisioning receipt is outside its live window")
            }
            Self::Inventory(error) => fmt::Display::fmt(error, formatter),
            Self::Workflow(error) => fmt::Display::fmt(error, formatter),
        }
    }
}

impl<ClockError> fmt::Debug for BlindVaultReplicaProvisioningReplyPolicyError<ClockError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clock(_) => formatter.write_str("Clock(<redacted>)"),
            Self::StageMismatch => formatter.write_str("StageMismatch"),
            Self::AttemptMismatch => formatter.write_str("AttemptMismatch"),
            Self::ReplicaTargetMismatch => formatter.write_str("ReplicaTargetMismatch"),
            Self::DuplicateReplica => formatter.write_str("DuplicateReplica"),
            Self::TerminalAuthorizationMismatch => {
                formatter.write_str("TerminalAuthorizationMismatch")
            }
            Self::ReceiptOutsideWindow => formatter.write_str("ReceiptOutsideWindow"),
            Self::Inventory(_) => formatter.write_str("Inventory(<redacted>)"),
            Self::Workflow(_) => formatter.write_str("Workflow(<redacted>)"),
        }
    }
}

impl<ClockError> Error for BlindVaultReplicaProvisioningReplyPolicyError<ClockError>
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
            | Self::ReplicaTargetMismatch
            | Self::DuplicateReplica
            | Self::TerminalAuthorizationMismatch
            | Self::ReceiptOutsideWindow => None,
        }
    }
}
