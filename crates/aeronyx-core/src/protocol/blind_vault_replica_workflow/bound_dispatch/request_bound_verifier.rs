// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault_replica_workflow/
//       bound_dispatch/request_bound_verifier.rs
// ============================================
//! Request-bound verification for Blind Vault replica terminal replies.
//!
//! ## Creation Reason
//! Onion reply authentication proves route, request context, and terminal
//! identity, but a workflow must also verify the inner request/receipt pair.
//! Private manifest rules cannot be centralized because they must remain only
//! at the source, so protocol verification and private policy are composed.
//!
//! ## Main Functionality
//! - Decodes the exact committed onion request and decrypted terminal reply.
//! - Accepts only the six terminal purposes used by replica workflows.
//! - Verifies terminal signatures and exact request/receipt relationships.
//! - Maps authenticated terminal failures without exposing storage details.
//! - Delegates manifest, freshness, and lifecycle policy to a source adapter.
//! - Centralizes anonymous single-effect attempt-context verification.
//! - Exposes a source-only mutable policy boundary between verified stages.
//!
//! ## Dependencies
//! - `attempt_runtime.rs`: semantic reply-verifier trait and private state.
//! - `protocol::onion_reply`: exact reply-capable request decoder.
//! - `protocol::blind_vault`: workload frames and signed receipts.
//! - `IdentityPublicKey`: descriptor identity signature verification.
//!
//! ## Main Logical Flow
//! 1. Decode the exact request already matched by effect and reply commitments.
//! 2. Decode request and response Blind Vault frames under bounded codecs.
//! 3. Reject unsupported purposes, mismatched failures, or frame pairings.
//! 4. Verify the response signature against the onion-authenticated terminal.
//! 5. Verify that the signed response answers the exact request.
//! 6. Give the typed pair and private adapter state to source-owned policy.
//!
//! ## Important Note For The Next Developer
//! - Request-bound success is not complete workflow evidence by itself.
//! - Delete still requires the source's prior ciphertext commitment.
//! - Inventory still requires comparison with the source's private manifest.
//! - Never implement a permissive default private policy.
//! - Debug output must remain redacted because requests can contain ciphertext.
//!
//! Last Modified: v1.4.0-TerminalFailureClassification - Implemented the
//! shared bounded runtime failure classifier for request-bound errors.
//! v1.3.0-SingleEffectContext - Centralized exact work,
//! attempt, sequence, and terminal-authorization checks for single replies.
//! v1.2.0-SharedVerificationClock - Moved the replaceable
//! source-time boundary beside the common private reply-policy contract.
//! v1.1.0-StagedPolicyMutation - Allowed source adapters to
//! install workflow authority between verified replacement stages.
//! v1.0.0-RequestBoundReplyVerification - Initial protocol and
//! private-policy composition for replica workflow terminal replies.
//! ============================================

use std::fmt;

use thiserror::Error;

use super::super::{BlindVaultReplicaDispatchFailure, BlindVaultReplicaWorkId};
use super::attempt_runtime::{
    BlindVaultReplicaTerminalReplyVerifier, BlindVaultReplicaTerminalVerificationFailure,
};
use super::send_sequence::BlindVaultReplicaTerminalSendContext;
use crate::crypto::keys::IdentityPublicKey;
use crate::protocol::blind_vault::{
    decode_blind_vault_frame, BlindVaultBlindLeaseAcceptedReceipt,
    BlindVaultBlindLeaseAdmissionRequest, BlindVaultBlindLeaseRenewalRequest,
    BlindVaultBlindLeaseRenewedReceipt, BlindVaultDeleteRequest, BlindVaultDeletedReceipt,
    BlindVaultError, BlindVaultFrame, BlindVaultLeaseInventoryReceipt,
    BlindVaultLeaseInventoryRequest, BlindVaultLeaseRetireRequest, BlindVaultLeaseRetiredReceipt,
    BlindVaultPutRequest, BlindVaultStoredReceipt, BlindVaultTerminalFailureCode,
    BlindVaultTerminalOperation,
};
use crate::protocol::onion::OnionRoutePurpose;
use crate::protocol::onion_reply::{
    decode_onion_reply_request, OnionReplyError, OnionReplyPayload,
};

/// Replaceable source clock used by freshness-bounded private reply policies.
///
/// [BLIND-VAULT-SHARED-VERIFICATION-CLOCK 2026-08-30 by Codex] The clock
/// belongs to the common source-policy boundary, not any one lifecycle action.
pub trait BlindVaultReplicaVerificationClock {
    type Error;

    /// Returns nonzero Unix time in milliseconds.
    fn now_ms(&mut self) -> Result<u64, Self::Error>;
}

impl<Clock, ClockError> BlindVaultReplicaVerificationClock for Clock
where
    Clock: FnMut() -> Result<u64, ClockError>,
{
    type Error = ClockError;

    fn now_ms(&mut self) -> Result<u64, Self::Error> {
        self()
    }
}

/// Private classification for one invalid single-effect reply context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlindVaultReplicaSingleEffectContextError {
    /// Work identity was wrong or the runtime attempt was invalid.
    AttemptMismatch,
    /// Context did not describe the sole effect at index zero.
    SequenceMismatch,
    /// Anonymous single-effect work unexpectedly carried lifecycle authority.
    TerminalAuthorizationMismatch,
}

/// Verifies context shared by source-private single-terminal reply policies.
///
/// [BLIND-VAULT-SINGLE-EFFECT-REPLY-CONTEXT 2026-08-30 by Codex] Context is
/// created by the ordered runtime, but policies still fail closed so future
/// adapters cannot weaken work, attempt, sequence, or lifecycle boundaries.
pub(super) fn verify_single_effect_reply_context(
    expected_work_id: BlindVaultReplicaWorkId,
    context: BlindVaultReplicaTerminalSendContext,
) -> Result<(), BlindVaultReplicaSingleEffectContextError> {
    if context.work_id() != expected_work_id || context.attempt() == 0 {
        return Err(BlindVaultReplicaSingleEffectContextError::AttemptMismatch);
    }
    if context.effect_index() != 0 || context.effect_count() != 1 {
        return Err(BlindVaultReplicaSingleEffectContextError::SequenceMismatch);
    }
    if context.authorized_terminal_node_id().is_some() {
        return Err(BlindVaultReplicaSingleEffectContextError::TerminalAuthorizationMismatch);
    }
    Ok(())
}

/// Exact signed request/receipt pair authenticated at the common protocol layer.
pub enum BlindVaultReplicaRequestBoundReply {
    /// The terminal accepted the exact blind lease-admission request.
    LeaseAccepted {
        /// Exact admission request committed by the source continuation.
        request: BlindVaultBlindLeaseAdmissionRequest,
        /// Terminal-signed acceptance receipt bound to `request`.
        receipt: BlindVaultBlindLeaseAcceptedReceipt,
    },
    /// The terminal durably stored the exact encrypted object request.
    ObjectStored {
        /// Exact encrypted-object write committed by the source continuation.
        request: BlindVaultPutRequest,
        /// Terminal-signed storage receipt bound to `request`.
        receipt: BlindVaultStoredReceipt,
    },
    /// The terminal deleted the object identified by the exact request.
    ObjectDeleted {
        /// Exact delete request committed by the source continuation.
        request: BlindVaultDeleteRequest,
        /// Terminal-signed deletion receipt bound to `request`.
        receipt: BlindVaultDeletedReceipt,
    },
    /// The terminal retired the exact blind lease requested by the source.
    LeaseRetired {
        /// Exact retirement request committed by the source continuation.
        request: BlindVaultLeaseRetireRequest,
        /// Terminal-signed retirement receipt bound to `request`.
        receipt: BlindVaultLeaseRetiredReceipt,
    },
    /// The terminal renewed the exact blind lease requested by the source.
    LeaseRenewed {
        /// Exact renewal request committed by the source continuation.
        request: BlindVaultBlindLeaseRenewalRequest,
        /// Terminal-signed renewal receipt bound to `request`.
        receipt: BlindVaultBlindLeaseRenewedReceipt,
    },
    /// The terminal reported inventory for the exact blind lease query.
    InventoryObserved {
        /// Exact inventory request committed by the source continuation.
        request: BlindVaultLeaseInventoryRequest,
        /// Terminal-signed inventory receipt bound to `request`.
        receipt: BlindVaultLeaseInventoryReceipt,
    },
}

impl BlindVaultReplicaRequestBoundReply {
    /// Canonical purpose already matched by request, response, and route.
    #[must_use]
    pub const fn purpose(&self) -> OnionRoutePurpose {
        match self {
            Self::LeaseAccepted { .. } => OnionRoutePurpose::BlindVaultLeaseAdmission,
            Self::ObjectStored { .. } => OnionRoutePurpose::BlindVaultPutReceipt,
            Self::ObjectDeleted { .. } => OnionRoutePurpose::BlindVaultDelete,
            Self::LeaseRetired { .. } => OnionRoutePurpose::BlindVaultLeaseRetire,
            Self::LeaseRenewed { .. } => OnionRoutePurpose::BlindVaultLeaseRenewal,
            Self::InventoryObserved { .. } => OnionRoutePurpose::BlindVaultLeaseInventory,
        }
    }
}

impl fmt::Debug for BlindVaultReplicaRequestBoundReply {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRequestBoundReply")
            .field("purpose", &self.purpose())
            .field("request", &"[REDACTED]")
            .field("receipt", &"[REDACTED]")
            .finish()
    }
}

/// Source-owned private checks that complete request-bound verification.
pub trait BlindVaultReplicaPrivateReplyPolicy {
    type Output;
    type Error;

    /// Applies manifest, freshness, and action-lifecycle requirements.
    fn verify_private_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        adapter_state: &[u8],
        reply: BlindVaultReplicaRequestBoundReply,
    ) -> Result<Self::Output, Self::Error>;
}

/// Composition of common protocol verification and private source policy.
pub struct BlindVaultReplicaRequestBoundReplyVerifier<Policy> {
    policy: Policy,
}

impl<Policy> BlindVaultReplicaRequestBoundReplyVerifier<Policy> {
    /// Creates a verifier without decoding state or performing network work.
    pub const fn new(policy: Policy) -> Self {
        Self { policy }
    }

    /// Borrows the private policy for source-owned state inspection.
    #[must_use]
    pub const fn policy(&self) -> &Policy {
        &self.policy
    }

    /// Mutably borrows source policy between ordered reply stages.
    ///
    /// Replacement adapters use this boundary to install the workflow-issued
    /// retirement permit after matching replacement inventory is verified and
    /// before invoking the runtime's retirement send method.
    ///
    /// [BLIND-VAULT-STAGED-POLICY-MUTATION 2026-08-29 by Codex]
    #[must_use]
    pub fn policy_mut(&mut self) -> &mut Policy {
        &mut self.policy
    }

    /// Transfers the private policy back to its source adapter.
    #[must_use]
    pub fn into_policy(self) -> Policy {
        self.policy
    }
}

impl<Policy> BlindVaultReplicaTerminalReplyVerifier
    for BlindVaultReplicaRequestBoundReplyVerifier<Policy>
where
    Policy: BlindVaultReplicaPrivateReplyPolicy,
{
    type Output = Policy::Output;
    type Error = BlindVaultReplicaRequestBoundReplyError<Policy::Error>;

    fn verify_terminal_reply(
        &mut self,
        context: BlindVaultReplicaTerminalSendContext,
        purpose: OnionRoutePurpose,
        encoded_request: &[u8],
        adapter_state: &[u8],
        reply: OnionReplyPayload,
    ) -> Result<Self::Output, Self::Error> {
        let verified = verify_request_bound_reply(purpose, encoded_request, reply)?;
        self.policy
            .verify_private_reply(context, adapter_state, verified)
            .map_err(BlindVaultReplicaRequestBoundReplyError::Policy)
    }
}

impl<Policy> fmt::Debug for BlindVaultReplicaRequestBoundReplyVerifier<Policy> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BlindVaultReplicaRequestBoundReplyVerifier")
            .field("policy", &std::any::type_name::<Policy>())
            .finish_non_exhaustive()
    }
}

fn verify_request_bound_reply<PolicyError>(
    purpose: OnionRoutePurpose,
    encoded_request: &[u8],
    reply: OnionReplyPayload,
) -> Result<BlindVaultReplicaRequestBoundReply, BlindVaultReplicaRequestBoundReplyError<PolicyError>>
{
    let expected_operation = expected_terminal_operation(purpose)
        .ok_or(BlindVaultReplicaRequestBoundReplyError::UnsupportedPurpose)?;
    let onion_request = decode_onion_reply_request(encoded_request)?;
    let request_frame = decode_blind_vault_frame(&onion_request.payload)
        .map_err(BlindVaultReplicaRequestBoundReplyError::RequestFrame)?;
    if !request_frame_matches_purpose(purpose, &request_frame) {
        return Err(BlindVaultReplicaRequestBoundReplyError::RequestFrameMismatch);
    }
    let response_frame = decode_blind_vault_frame(&reply.payload)
        .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
    if let BlindVaultFrame::TerminalFailure(failure) = &response_frame {
        if failure.operation() != expected_operation {
            return Err(BlindVaultReplicaRequestBoundReplyError::TerminalOperationMismatch);
        }
        return Err(BlindVaultReplicaRequestBoundReplyError::TerminalFailure(
            failure.code(),
        ));
    }
    let terminal_key = IdentityPublicKey::from_bytes(&reply.terminal_node_id)
        .map_err(|_| BlindVaultReplicaRequestBoundReplyError::InvalidTerminalIdentity)?;

    // [BLIND-VAULT-REQUEST-BOUND-REPLY 2026-08-29 by Codex] Every accepted
    // branch verifies both the terminal signature and exact request relation.
    // No wildcard response branch may degrade an unknown frame into success.
    match (purpose, request_frame, response_frame) {
        (
            OnionRoutePurpose::BlindVaultLeaseAdmission,
            BlindVaultFrame::BlindLeaseAdmission(request),
            BlindVaultFrame::BlindLeaseAccepted(receipt),
        ) => {
            receipt
                .validate_and_verify(&terminal_key)
                .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
            require_match(receipt.matches_admission(&request))?;
            Ok(BlindVaultReplicaRequestBoundReply::LeaseAccepted { request, receipt })
        }
        (
            OnionRoutePurpose::BlindVaultPutReceipt,
            BlindVaultFrame::Put(request),
            BlindVaultFrame::StoredReceipt(receipt),
        ) => {
            receipt
                .validate_and_verify(&terminal_key)
                .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
            require_match(receipt.matches_put(&request))?;
            Ok(BlindVaultReplicaRequestBoundReply::ObjectStored { request, receipt })
        }
        (
            OnionRoutePurpose::BlindVaultDelete,
            BlindVaultFrame::Delete(request),
            BlindVaultFrame::DeletedReceipt(receipt),
        ) => {
            receipt
                .validate_and_verify(&terminal_key)
                .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
            require_match(receipt.matches_delete(&request))?;
            Ok(BlindVaultReplicaRequestBoundReply::ObjectDeleted { request, receipt })
        }
        (
            OnionRoutePurpose::BlindVaultLeaseRetire,
            BlindVaultFrame::LeaseRetire(request),
            BlindVaultFrame::LeaseRetiredReceipt(receipt),
        ) => {
            receipt
                .validate_and_verify(&terminal_key)
                .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
            require_match(receipt.matches_retire(&request))?;
            Ok(BlindVaultReplicaRequestBoundReply::LeaseRetired { request, receipt })
        }
        (
            OnionRoutePurpose::BlindVaultLeaseRenewal,
            BlindVaultFrame::BlindLeaseRenewal(request),
            BlindVaultFrame::BlindLeaseRenewed(receipt),
        ) => {
            receipt
                .validate_and_verify(&terminal_key)
                .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
            require_match(receipt.matches_renewal(&request))?;
            Ok(BlindVaultReplicaRequestBoundReply::LeaseRenewed { request, receipt })
        }
        (
            OnionRoutePurpose::BlindVaultLeaseInventory,
            BlindVaultFrame::LeaseInventory(request),
            BlindVaultFrame::LeaseInventoryReceipt(receipt),
        ) => {
            receipt
                .validate_and_verify(&terminal_key)
                .map_err(BlindVaultReplicaRequestBoundReplyError::ResponseFrame)?;
            require_match(receipt.matches_inventory(&request))?;
            Ok(BlindVaultReplicaRequestBoundReply::InventoryObserved { request, receipt })
        }
        _ => Err(BlindVaultReplicaRequestBoundReplyError::ResponseFrameMismatch),
    }
}

fn request_frame_matches_purpose(purpose: OnionRoutePurpose, frame: &BlindVaultFrame) -> bool {
    matches!(
        (purpose, frame),
        (
            OnionRoutePurpose::BlindVaultLeaseAdmission,
            BlindVaultFrame::BlindLeaseAdmission(_)
        ) | (
            OnionRoutePurpose::BlindVaultPutReceipt,
            BlindVaultFrame::Put(_)
        ) | (
            OnionRoutePurpose::BlindVaultDelete,
            BlindVaultFrame::Delete(_)
        ) | (
            OnionRoutePurpose::BlindVaultLeaseRetire,
            BlindVaultFrame::LeaseRetire(_)
        ) | (
            OnionRoutePurpose::BlindVaultLeaseRenewal,
            BlindVaultFrame::BlindLeaseRenewal(_)
        ) | (
            OnionRoutePurpose::BlindVaultLeaseInventory,
            BlindVaultFrame::LeaseInventory(_)
        )
    )
}

fn expected_terminal_operation(purpose: OnionRoutePurpose) -> Option<BlindVaultTerminalOperation> {
    match purpose {
        OnionRoutePurpose::BlindVaultLeaseAdmission => {
            Some(BlindVaultTerminalOperation::LeaseAdmission)
        }
        OnionRoutePurpose::BlindVaultPutReceipt => Some(BlindVaultTerminalOperation::Put),
        OnionRoutePurpose::BlindVaultDelete => Some(BlindVaultTerminalOperation::Delete),
        OnionRoutePurpose::BlindVaultLeaseRetire => Some(BlindVaultTerminalOperation::LeaseRetire),
        OnionRoutePurpose::BlindVaultLeaseRenewal => {
            Some(BlindVaultTerminalOperation::LeaseRenewal)
        }
        OnionRoutePurpose::BlindVaultLeaseInventory => {
            Some(BlindVaultTerminalOperation::LeaseInventory)
        }
        OnionRoutePurpose::MessageRelay
        | OnionRoutePurpose::BlindVaultPut
        | OnionRoutePurpose::BlindVaultPull
        | OnionRoutePurpose::BlindVaultLeaseStatus => None,
    }
}

fn require_match<PolicyError>(
    matches: bool,
) -> Result<(), BlindVaultReplicaRequestBoundReplyError<PolicyError>> {
    matches
        .then_some(())
        .ok_or(BlindVaultReplicaRequestBoundReplyError::RequestMismatch)
}

/// Common protocol failure or source-private policy rejection.
#[derive(Debug, Error)]
pub enum BlindVaultReplicaRequestBoundReplyError<PolicyError> {
    /// Encoded onion request was malformed or unsupported.
    #[error(transparent)]
    OnionReply(#[from] OnionReplyError),
    /// Persisted source request contained an invalid Blind Vault frame.
    #[error("blind vault request-bound source frame is invalid")]
    RequestFrame(#[source] BlindVaultError),
    /// Authenticated terminal response contained an invalid Blind Vault frame.
    #[error("blind vault request-bound terminal frame is invalid")]
    ResponseFrame(#[source] BlindVaultError),
    /// The outer authenticated terminal identity was not valid Ed25519.
    #[error("blind vault request-bound reply terminal identity is invalid")]
    InvalidTerminalIdentity,
    /// Purpose is not part of the replica workflow terminal contract.
    #[error("blind vault request-bound reply purpose is unsupported")]
    UnsupportedPurpose,
    /// Terminal failure operation did not match the exact route purpose.
    #[error("blind vault encrypted terminal failure operation mismatched")]
    TerminalOperationMismatch,
    /// Authenticated terminal returned one coarse operation failure.
    #[error("blind vault encrypted terminal failure: {0}")]
    TerminalFailure(BlindVaultTerminalFailureCode),
    /// Persisted source request frame did not implement its committed purpose.
    #[error("blind vault request-bound source frame mismatched its purpose")]
    RequestFrameMismatch,
    /// Terminal response frame did not implement the expected request pair.
    #[error("blind vault request-bound terminal frame mismatched its request")]
    ResponseFrameMismatch,
    /// Signed receipt did not answer the exact committed request.
    #[error("blind vault request-bound terminal receipt mismatched")]
    RequestMismatch,
    /// Source-private manifest or lifecycle policy rejected the signed pair.
    #[error("blind vault private reply policy rejected terminal outcome")]
    Policy(PolicyError),
}

impl<PolicyError> BlindVaultReplicaRequestBoundReplyError<PolicyError> {
    /// Maps detailed source-local verification failure into durable workflow
    /// state without persisting request, receipt, or policy error details.
    #[must_use]
    pub fn dispatch_failure(&self) -> BlindVaultReplicaDispatchFailure {
        match self {
            Self::TerminalFailure(code) => (*code).into(),
            Self::Policy(_) => BlindVaultReplicaDispatchFailure::PolicyRejected,
            Self::OnionReply(_)
            | Self::RequestFrame(_)
            | Self::UnsupportedPurpose
            | Self::RequestFrameMismatch => {
                BlindVaultReplicaDispatchFailure::LocalConstructionFailed
            }
            Self::ResponseFrame(_)
            | Self::InvalidTerminalIdentity
            | Self::TerminalOperationMismatch
            | Self::ResponseFrameMismatch
            | Self::RequestMismatch => BlindVaultReplicaDispatchFailure::TerminalRejected,
        }
    }
}

impl<PolicyError> BlindVaultReplicaTerminalVerificationFailure
    for BlindVaultReplicaRequestBoundReplyError<PolicyError>
{
    fn dispatch_failure(&self) -> BlindVaultReplicaDispatchFailure {
        BlindVaultReplicaRequestBoundReplyError::dispatch_failure(self)
    }
}
