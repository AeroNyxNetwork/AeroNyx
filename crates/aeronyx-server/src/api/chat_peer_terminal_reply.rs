// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer_terminal_reply.rs
// ============================================================================
//! # Blind Relay Terminal Reply Domain
//!
//! ## Creation Reason
//! Keeps reply-capable terminal workload execution out of the blind-relay HTTP
//! orchestrator while preserving the node-blind storage boundary.
//!
//! ## Main Functionality
//! - Accepts one decoded onion reply carrier at a terminal node.
//! - Restricts inline v1 recovery to one maximum-class Blind Vault object page.
//! - Executes immutable writes, capability-authenticated recovery, signed
//!   object deletion or complete lease retirement, and blind-issued admission
//!   without exposing any request to middle relays.
//! - Signs the recovery page, seals it to the source reply key, and returns
//!   only fixed-size opaque base64 bytes to the relay orchestrator.
//! - Seals coarse operation failures into the same fixed-size response so
//!   entry and middle relays cannot distinguish success from failure.
//!
//! ## Dependencies
//! - `aeronyx-core::protocol`: Blind Vault and onion reply wire contracts.
//! - `services::BlindVaultService`: replica-local encrypted object recovery.
//!
//! ## Main Logical Flow
//! 1. Decode and validate the reply carrier and its inner Blind Vault frame.
//! 2. Require `PullRequest(limit=1)` and its negotiated fixed response class.
//! 3. Read one stable snapshot page using the bearer read capability.
//! 4. Sign the exact ciphertext page with the terminal descriptor identity.
//! 5. Seal and base64-encode the fixed-size response for unchanged propagation.
//!
//! ## Important Note For The Next Developer
//! - Never log or return lease ids, capabilities, object ids, cursors, payloads,
//!   ciphertext commitments, node paths, or storage error strings.
//! - Keep the pull response class and peer ACK ceiling derived from core wire
//!   bounds. Every path hop must advertise the large-pull feature.
//! - Keep node fan-out out of this module; clients choose unrelated replicas
//!   and routes so one node cannot reconstruct a logical replica set.
//!
//! Last Modified: v1.10.0-BlindVaultPreparedEffect - Separates pure request
//! validation from read-only or mutating terminal execution.
//! v1.9.0-BlindVaultLargePull - Carried every v1 ciphertext
//! class through one negotiated maximum fixed-size response.
//! v1.8.0-BlindVaultEncryptedFailure - Moved valid-request
//! workload failures inside source-only authenticated onion replies.
//! v1.7.0-BlindVaultLeaseInventory - Added private streaming
//! inventory commitments with encrypted terminal-signed receipts.
//! v1.6.0-BlindVaultLeaseStatus - Added private
//! administration-authorized status observations with encrypted signed receipts.
//! v1.5.0-BlindVaultLeaseRenewal - Added blind-authorized
//! lease renewal with encrypted terminal-signed transition receipts.
//! v1.4.0-BlindVaultLeaseRetirement - Added complete anonymous
//! lease retirement with encrypted terminal-signed aggregate receipts.
//! v1.3.0-BlindVaultInlinePutReceipt - Added anonymous writes
//! with encrypted terminal-signed storage receipts and capacity classification.
//! v1.2.0-BlindVaultInlineAdmission - Added RFC 9474 blind-issued
//! lease admission with a terminal-signed, request-bound encrypted receipt.
//! v1.1.0-BlindVaultInlineDelete - Added anonymous deletion and
//! terminal-signed receipt replies through the same fixed-size carrier.
//! v1.0.0-BlindVaultInlinePull - Initial anonymous pull reply.
//! ============================================================================

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{
    decode_blind_vault_frame, decode_onion_reply_request, encode_blind_vault_frame,
    encode_onion_sealed_response, seal_onion_reply, BlindVaultBlindLeaseAcceptedReceipt,
    BlindVaultFrame, BlindVaultPullResponse, BlindVaultRecoveredObject, BlindVaultTerminalFailure,
    BlindVaultTerminalFailureCode, BlindVaultTerminalOperation, OnionReplyProofMode,
    OnionRoutePurpose, BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES,
    BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS, ONION_REPLY_RESPONSE_SIZE_CLASSES,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

use crate::services::{
    BlindVaultAdmissionFailureClass, BlindVaultDeleteFailureClass,
    BlindVaultLeaseInventoryFailureClass, BlindVaultLeaseRenewFailureClass,
    BlindVaultLeaseRetireFailureClass, BlindVaultLeaseStatusFailureClass,
    BlindVaultPullFailureClass, BlindVaultPutFailureClass, BlindVaultService,
    BlindVaultServiceError,
};

/// Coarse terminal-reply failure class used before encrypted response sealing.
///
/// Valid workload failures are converted into `BlindVaultTerminalFailure`.
/// Only malformed carriers and failures that prevent sealing escape to relay
/// orchestration through this internal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TerminalReplyFailure {
    /// The request, capability, cursor, or protocol frame is invalid.
    Rejected,
    /// The selected inline response class cannot carry the recovered page.
    ResponseTooLarge,
    /// The selected replica cannot accept more ciphertext under this lease.
    Capacity,
    /// Replica storage or response sealing is temporarily unavailable.
    Unavailable,
}

impl TerminalReplyFailure {
    const fn encrypted_code(self) -> BlindVaultTerminalFailureCode {
        match self {
            Self::Rejected => BlindVaultTerminalFailureCode::Rejected,
            Self::ResponseTooLarge => BlindVaultTerminalFailureCode::ResponseTooLarge,
            Self::Capacity => BlindVaultTerminalFailureCode::Capacity,
            Self::Unavailable => BlindVaultTerminalFailureCode::Unavailable,
        }
    }
}

/// Successfully sealed terminal reply with no plaintext metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TerminalReply {
    pub(super) purpose: OnionRoutePurpose,
    pub(super) proof_mode: OnionReplyProofMode,
    pub(super) opaque_response_b64: String,
}

/// External-effect class for one validated terminal reply operation.
///
/// [BLIND-VAULT-PREPARED-EFFECT 2026-08-30 by Codex] Relay recovery must arm a
/// durable route only for operations that can mutate replica state. Keeping
/// this decision next to the exhaustive frame match prevents a newly added
/// operation from silently inheriting the wrong cancellation semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TerminalReplyEffect {
    /// The operation observes one stable replica snapshot without mutating it.
    ReadOnly,
    /// The operation may commit a durable Blind Vault state transition.
    Mutation,
}

impl TerminalReplyEffect {
    pub(super) const fn requires_durable_guard(self) -> bool {
        matches!(self, Self::Mutation)
    }
}

/// Fully decoded terminal request ready for guarded execution.
///
/// This type intentionally does not implement `Debug`: both the reply carrier
/// and the Blind Vault frame contain private capabilities or ciphertext.
pub(super) struct PreparedTerminalReply {
    reply_request: aeronyx_core::protocol::OnionReplyRequest,
    frame: BlindVaultFrame,
    purpose: OnionRoutePurpose,
    operation: BlindVaultTerminalOperation,
    effect: TerminalReplyEffect,
}

impl PreparedTerminalReply {
    pub(super) const fn effect(&self) -> TerminalReplyEffect {
        self.effect
    }

    /// Executes a request whose wire shape and operation class were validated.
    pub(super) fn execute(
        self,
        vault: &BlindVaultService,
        terminal_identity: &IdentityKeyPair,
        route_id: [u8; 16],
        now_ms: u64,
    ) -> Result<TerminalReply, TerminalReplyFailure> {
        let Self {
            reply_request,
            frame,
            purpose,
            operation,
            effect: _,
        } = self;
        let response = execute_prepared_frame(vault, terminal_identity, frame, now_ms);

        // [BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] Once a valid
        // reply-capable workload request is identified, every workload failure
        // is sealed to the source. Upstream relays receive the same fixed-size
        // opaque success carrier and cannot use rejection/capacity as an oracle.
        let encoded_response = match response {
            Ok(encoded) => encoded,
            Err(failure) => encode_terminal_failure(operation, failure)?,
        };
        let sealed = match seal_onion_reply(
            route_id,
            &reply_request,
            &encoded_response,
            terminal_identity,
        ) {
            Ok(sealed) => sealed,
            Err(aeronyx_core::protocol::OnionReplyError::ResponsePayloadTooLarge) => {
                let failure =
                    encode_terminal_failure(operation, TerminalReplyFailure::ResponseTooLarge)?;
                seal_onion_reply(route_id, &reply_request, &failure, terminal_identity)
                    .map_err(|_| TerminalReplyFailure::Unavailable)?
            }
            Err(_) => return Err(TerminalReplyFailure::Unavailable),
        };
        let encoded_sealed =
            encode_onion_sealed_response(&sealed).map_err(|_| TerminalReplyFailure::Unavailable)?;
        Ok(TerminalReply {
            purpose,
            proof_mode: reply_request.proof_mode(),
            opaque_response_b64: BASE64.encode(encoded_sealed),
        })
    }
}

/// Decodes and classifies one reply-capable Blind Vault request without
/// reading or mutating replica storage.
pub(super) fn prepare_blind_vault_inline_reply(
    encoded_request: &[u8],
) -> Result<PreparedTerminalReply, TerminalReplyFailure> {
    let reply_request =
        decode_onion_reply_request(encoded_request).map_err(|_| TerminalReplyFailure::Rejected)?;
    let response_size_class = usize::try_from(reply_request.response_size_class)
        .map_err(|_| TerminalReplyFailure::Rejected)?;

    let frame = decode_blind_vault_frame(&reply_request.payload)
        .map_err(|_| TerminalReplyFailure::Rejected)?;
    let (purpose, operation, effect) = classify_prepared_frame(&frame)?;
    let is_pull = matches!(&frame, BlindVaultFrame::PullRequest(_));
    let current_size_class = if is_pull {
        BLIND_VAULT_ONION_PULL_RESPONSE_SIZE_CLASS
    } else {
        ONION_REPLY_RESPONSE_SIZE_CLASSES[0]
    };
    // [BLIND-VAULT-ONION-PULL-SIZE 2026-08-30 by Codex] Only Pull may reserve
    // the larger carrier. Retaining the old 8 KiB Pull class lets an upgraded
    // terminal serve legacy 4 KiB-only clients during a rolling deployment;
    // every other operation remains non-amplifying.
    let legacy_pull_size = is_pull && response_size_class == ONION_REPLY_RESPONSE_SIZE_CLASSES[0];
    if response_size_class != current_size_class && !legacy_pull_size {
        return Err(TerminalReplyFailure::Rejected);
    }

    // Reject operation-specific wire-shape errors before relay recovery marks
    // a route as having crossed an external-effect boundary.
    match &frame {
        BlindVaultFrame::Put(request)
            if request.ciphertext.len() != BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0] =>
        {
            return Err(TerminalReplyFailure::Rejected);
        }
        BlindVaultFrame::PullRequest(request) => {
            request
                .validate()
                .map_err(|_| TerminalReplyFailure::Rejected)?;
            if request.limit != 1 {
                return Err(TerminalReplyFailure::Rejected);
            }
        }
        _ => {}
    }

    Ok(PreparedTerminalReply {
        reply_request,
        frame,
        purpose,
        operation,
        effect,
    })
}

fn classify_prepared_frame(
    frame: &BlindVaultFrame,
) -> Result<
    (
        OnionRoutePurpose,
        BlindVaultTerminalOperation,
        TerminalReplyEffect,
    ),
    TerminalReplyFailure,
> {
    match frame {
        BlindVaultFrame::Put(_) => Ok((
            OnionRoutePurpose::BlindVaultPutReceipt,
            BlindVaultTerminalOperation::Put,
            TerminalReplyEffect::Mutation,
        )),
        BlindVaultFrame::PullRequest(_) => Ok((
            OnionRoutePurpose::BlindVaultPull,
            BlindVaultTerminalOperation::Pull,
            TerminalReplyEffect::ReadOnly,
        )),
        BlindVaultFrame::Delete(_) => Ok((
            OnionRoutePurpose::BlindVaultDelete,
            BlindVaultTerminalOperation::Delete,
            TerminalReplyEffect::Mutation,
        )),
        BlindVaultFrame::BlindLeaseAdmission(_) => Ok((
            OnionRoutePurpose::BlindVaultLeaseAdmission,
            BlindVaultTerminalOperation::LeaseAdmission,
            TerminalReplyEffect::Mutation,
        )),
        BlindVaultFrame::BlindLeaseRenewal(_) => Ok((
            OnionRoutePurpose::BlindVaultLeaseRenewal,
            BlindVaultTerminalOperation::LeaseRenewal,
            TerminalReplyEffect::Mutation,
        )),
        BlindVaultFrame::LeaseStatus(_) => Ok((
            OnionRoutePurpose::BlindVaultLeaseStatus,
            BlindVaultTerminalOperation::LeaseStatus,
            TerminalReplyEffect::ReadOnly,
        )),
        BlindVaultFrame::LeaseInventory(_) => Ok((
            OnionRoutePurpose::BlindVaultLeaseInventory,
            BlindVaultTerminalOperation::LeaseInventory,
            TerminalReplyEffect::ReadOnly,
        )),
        BlindVaultFrame::LeaseRetire(_) => Ok((
            OnionRoutePurpose::BlindVaultLeaseRetire,
            BlindVaultTerminalOperation::LeaseRetire,
            TerminalReplyEffect::Mutation,
        )),
        _ => Err(TerminalReplyFailure::Rejected),
    }
}

fn execute_prepared_frame(
    vault: &BlindVaultService,
    terminal_identity: &IdentityKeyPair,
    frame: BlindVaultFrame,
    now_ms: u64,
) -> Result<Vec<u8>, TerminalReplyFailure> {
    match frame {
        BlindVaultFrame::Put(put_request) => {
            let receipt = vault
                .put(&put_request, now_ms)
                .map_err(classify_put_failure)?;
            encode_blind_vault_frame(&BlindVaultFrame::StoredReceipt(receipt))
                .map_err(|_| TerminalReplyFailure::Unavailable)
        }
        BlindVaultFrame::PullRequest(pull_request) => {
            execute_pull_response(vault, terminal_identity, pull_request, now_ms)
        }
        BlindVaultFrame::Delete(delete_request) => vault
            .delete(&delete_request, now_ms)
            .map_err(classify_delete_failure)
            .and_then(|receipt| {
                encode_blind_vault_frame(&BlindVaultFrame::DeletedReceipt(receipt))
                    .map_err(|_| TerminalReplyFailure::Unavailable)
            }),
        BlindVaultFrame::BlindLeaseAdmission(admission_request) => (|| {
            vault
                .provision_lease_with_blind_admission(&admission_request, now_ms)
                .map_err(classify_admission_failure)?;
            let mut receipt = BlindVaultBlindLeaseAcceptedReceipt::new(
                &admission_request,
                now_ms,
                terminal_identity.public_key_bytes(),
            );
            receipt
                .sign(terminal_identity)
                .map_err(|_| TerminalReplyFailure::Unavailable)?;
            encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAccepted(receipt))
                .map_err(|_| TerminalReplyFailure::Unavailable)
        })(),
        BlindVaultFrame::BlindLeaseRenewal(renewal_request) => vault
            .renew_lease_with_blind_admission(&renewal_request, now_ms)
            .map_err(classify_lease_renew_failure)
            .and_then(|receipt| {
                encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseRenewed(receipt))
                    .map_err(|_| TerminalReplyFailure::Unavailable)
            }),
        BlindVaultFrame::LeaseStatus(status_request) => vault
            .lease_status(&status_request, now_ms)
            .map_err(classify_lease_status_failure)
            .and_then(|receipt| {
                encode_blind_vault_frame(&BlindVaultFrame::LeaseStatusReceipt(receipt))
                    .map_err(|_| TerminalReplyFailure::Unavailable)
            }),
        BlindVaultFrame::LeaseInventory(inventory_request) => vault
            .lease_inventory(&inventory_request, now_ms)
            .map_err(classify_lease_inventory_failure)
            .and_then(|receipt| {
                encode_blind_vault_frame(&BlindVaultFrame::LeaseInventoryReceipt(receipt))
                    .map_err(|_| TerminalReplyFailure::Unavailable)
            }),
        BlindVaultFrame::LeaseRetire(retire_request) => vault
            .retire_lease(&retire_request, now_ms)
            .map_err(classify_lease_retire_failure)
            .and_then(|receipt| {
                encode_blind_vault_frame(&BlindVaultFrame::LeaseRetiredReceipt(receipt))
                    .map_err(|_| TerminalReplyFailure::Unavailable)
            }),
        _ => Err(TerminalReplyFailure::Rejected),
    }
}

fn encode_terminal_failure(
    operation: BlindVaultTerminalOperation,
    failure: TerminalReplyFailure,
) -> Result<Vec<u8>, TerminalReplyFailure> {
    let failure = BlindVaultTerminalFailure::new(operation, failure.encrypted_code());
    encode_blind_vault_frame(&BlindVaultFrame::TerminalFailure(failure))
        .map_err(|_| TerminalReplyFailure::Unavailable)
}

fn execute_pull_response(
    vault: &BlindVaultService,
    terminal_identity: &IdentityKeyPair,
    pull_request: aeronyx_core::protocol::BlindVaultPullRequest,
    now_ms: u64,
) -> Result<Vec<u8>, TerminalReplyFailure> {
    pull_request
        .validate()
        .map_err(|_| TerminalReplyFailure::Rejected)?;
    if pull_request.limit != 1 {
        return Err(TerminalReplyFailure::Rejected);
    }

    let cursor = (!pull_request.continuation_cursor.is_empty())
        .then_some(pull_request.continuation_cursor.as_slice());
    let page = vault
        .pull_page(
            &pull_request.lease_id,
            &pull_request.read_capability,
            cursor,
            1,
            now_ms,
        )
        .map_err(classify_pull_failure)?;
    let objects = page
        .objects
        .into_iter()
        .map(|object| BlindVaultRecoveredObject {
            object_id: object.object_id,
            ciphertext: object.ciphertext,
            ciphertext_commitment: object.ciphertext_commitment,
            expires_at_ms: object.expires_at_ms,
        })
        .collect();
    let mut pull_response = BlindVaultPullResponse::new(
        pull_request.lease_id,
        objects,
        page.continuation_cursor.unwrap_or_default(),
        now_ms,
        terminal_identity.public_key_bytes(),
    );
    pull_response
        .sign(terminal_identity)
        .map_err(|_| TerminalReplyFailure::Unavailable)?;
    encode_blind_vault_frame(&BlindVaultFrame::PullResponse(pull_response))
        .map_err(|_| TerminalReplyFailure::ResponseTooLarge)
}

fn classify_pull_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-PULL-FAILURE-CLASS 2026-08-28 by Codex] Authentication and
    // object-state failures collapse to one encrypted source-only rejection.
    // Only local storage/runtime failures remain retryable after decryption.
    match error.pull_failure_class() {
        BlindVaultPullFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultPullFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_put_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-PUT-RECEIPT-FAILURE-CLASS 2026-08-28 by Codex] Preserve
    // capacity as a source-only route-selection signal while hiding every
    // lease, signature, object, and database detail from all upstream relays.
    match error.put_failure_class() {
        BlindVaultPutFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultPutFailureClass::Capacity => TerminalReplyFailure::Capacity,
        BlindVaultPutFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_delete_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-DELETE-FAILURE-CLASS 2026-08-28 by Codex] Only the source
    // learns permanent rejection versus temporary replica unavailability;
    // object existence and tombstone state remain indistinct.
    match error.delete_failure_class() {
        BlindVaultDeleteFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultDeleteFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_admission_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-ADMISSION-CAPACITY 2026-08-28 by Codex] Credential, issuer,
    // replay, and lease state remain one rejection. Only node-wide capacity is
    // actionable so the source can select another terminal without disclosing
    // any lease-local state to entry or middle relays.
    match error.admission_failure_class() {
        BlindVaultAdmissionFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultAdmissionFailureClass::Capacity => TerminalReplyFailure::Capacity,
        BlindVaultAdmissionFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_lease_retire_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-LEASE-RETIRE-FAILURE 2026-08-28 by Codex] Lease existence,
    // expiry, prior retirement, authority, and request conflicts collapse into
    // one permanent rejection inside the encrypted terminal boundary.
    match error.lease_retire_failure_class() {
        BlindVaultLeaseRetireFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultLeaseRetireFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_lease_renew_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-LEASE-RENEW-FAILURE 2026-08-28 by Codex] Credential,
    // generation, expiry, and lease-authority failures remain one permanent
    // rejection inside the encrypted terminal response boundary.
    match error.lease_renew_failure_class() {
        BlindVaultLeaseRenewFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultLeaseRenewFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_lease_status_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-LEASE-STATUS-FAILURE 2026-08-28 by Codex] Status callers
    // learn only permanent rejection versus replica unavailability after
    // decryption; lease existence, usage, and authority remain hidden.
    match error.lease_status_failure_class() {
        BlindVaultLeaseStatusFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultLeaseStatusFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_lease_inventory_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-INVENTORY-FAILURE 2026-08-28 by Codex] Only the source
    // learns permanent rejection versus local unavailability; the inventory
    // root, usage counters, lease state, and authority result stay encrypted.
    match error.lease_inventory_failure_class() {
        BlindVaultLeaseInventoryFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultLeaseInventoryFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}
