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
//! - Restricts inline v1 recovery to one 4 KiB-class Blind Vault object page.
//! - Executes immutable writes, capability-authenticated recovery, signed
//!   object deletion or complete lease retirement, and blind-issued admission
//!   without exposing any request to middle relays.
//! - Signs the recovery page, seals it to the source reply key, and returns
//!   only fixed-size opaque base64 bytes to the relay orchestrator.
//! - Maps internal failures to coarse retry classes without retaining secrets.
//!
//! ## Dependencies
//! - `aeronyx-core::protocol`: Blind Vault and onion reply wire contracts.
//! - `services::BlindVaultService`: replica-local encrypted object recovery.
//!
//! ## Main Logical Flow
//! 1. Decode and validate the reply carrier and its inner Blind Vault frame.
//! 2. Require `PullRequest(limit=1)` and the inline 8 KiB response class.
//! 3. Read one stable snapshot page using the bearer read capability.
//! 4. Sign the exact ciphertext page with the terminal descriptor identity.
//! 5. Seal and base64-encode the fixed-size response for unchanged propagation.
//!
//! ## Important Note For The Next Developer
//! - Never log or return lease ids, capabilities, object ids, cursors, payloads,
//!   ciphertext commitments, node paths, or storage error strings.
//! - Do not raise the inline size class without also raising and auditing the
//!   peer ACK response ceiling. Larger replies require a separate binary path.
//! - Keep node fan-out out of this module; clients choose unrelated replicas
//!   and routes so one node cannot reconstruct a logical replica set.
//!
//! Last Modified: v1.4.0-BlindVaultLeaseRetirement - Added complete anonymous
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
    BlindVaultFrame, BlindVaultPullResponse, BlindVaultRecoveredObject, OnionRoutePurpose,
    BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES, ONION_REPLY_RESPONSE_SIZE_CLASSES,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

use crate::services::{
    BlindVaultAdmissionFailureClass, BlindVaultDeleteFailureClass,
    BlindVaultLeaseRetireFailureClass, BlindVaultPullFailureClass, BlindVaultPutFailureClass,
    BlindVaultService, BlindVaultServiceError,
};

/// Coarse terminal-reply failure class consumed by blind-relay orchestration.
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

/// Successfully sealed terminal reply with no plaintext metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TerminalReply {
    pub(super) purpose: OnionRoutePurpose,
    pub(super) opaque_response_b64: String,
}

/// Executes one reply-capable Blind Vault request and seals its fixed-size ACK.
pub(super) fn execute_blind_vault_inline_reply(
    vault: &BlindVaultService,
    terminal_identity: &IdentityKeyPair,
    route_id: [u8; 16],
    encoded_request: &[u8],
    now_ms: u64,
) -> Result<TerminalReply, TerminalReplyFailure> {
    let reply_request =
        decode_onion_reply_request(encoded_request).map_err(|_| TerminalReplyFailure::Rejected)?;
    if usize::try_from(reply_request.response_size_class).ok()
        != Some(ONION_REPLY_RESPONSE_SIZE_CLASSES[0])
    {
        return Err(TerminalReplyFailure::Rejected);
    }

    let (purpose, encoded_response) = match decode_blind_vault_frame(&reply_request.payload)
        .map_err(|_| TerminalReplyFailure::Rejected)?
    {
        BlindVaultFrame::Put(put_request) => {
            if put_request.ciphertext.len() != BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0] {
                return Err(TerminalReplyFailure::Rejected);
            }
            let receipt = vault
                .put(&put_request, now_ms)
                .map_err(classify_put_failure)?;
            let encoded = encode_blind_vault_frame(&BlindVaultFrame::StoredReceipt(receipt))
                .map_err(|_| TerminalReplyFailure::Unavailable)?;
            (OnionRoutePurpose::BlindVaultPutReceipt, encoded)
        }
        BlindVaultFrame::PullRequest(pull_request) => (
            OnionRoutePurpose::BlindVaultPull,
            execute_pull_response(vault, terminal_identity, pull_request, now_ms)?,
        ),
        BlindVaultFrame::Delete(delete_request) => {
            let receipt = vault
                .delete(&delete_request, now_ms)
                .map_err(classify_delete_failure)?;
            let encoded = encode_blind_vault_frame(&BlindVaultFrame::DeletedReceipt(receipt))
                .map_err(|_| TerminalReplyFailure::Unavailable)?;
            (OnionRoutePurpose::BlindVaultDelete, encoded)
        }
        BlindVaultFrame::BlindLeaseAdmission(admission_request) => {
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
            let encoded = encode_blind_vault_frame(&BlindVaultFrame::BlindLeaseAccepted(receipt))
                .map_err(|_| TerminalReplyFailure::Unavailable)?;
            (OnionRoutePurpose::BlindVaultLeaseAdmission, encoded)
        }
        BlindVaultFrame::LeaseRetire(retire_request) => {
            let receipt = vault
                .retire_lease(&retire_request, now_ms)
                .map_err(classify_lease_retire_failure)?;
            let encoded = encode_blind_vault_frame(&BlindVaultFrame::LeaseRetiredReceipt(receipt))
                .map_err(|_| TerminalReplyFailure::Unavailable)?;
            (OnionRoutePurpose::BlindVaultLeaseRetire, encoded)
        }
        _ => return Err(TerminalReplyFailure::Rejected),
    };

    let sealed = seal_onion_reply(
        route_id,
        &reply_request,
        &encoded_response,
        terminal_identity,
    )
    .map_err(|error| match error {
        aeronyx_core::protocol::OnionReplyError::ResponsePayloadTooLarge => {
            TerminalReplyFailure::ResponseTooLarge
        }
        _ => TerminalReplyFailure::Unavailable,
    })?;
    let encoded_sealed =
        encode_onion_sealed_response(&sealed).map_err(|_| TerminalReplyFailure::Unavailable)?;
    Ok(TerminalReply {
        purpose,
        opaque_response_b64: BASE64.encode(encoded_sealed),
    })
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
    // object-state failures stay indistinguishable to upstream relays. Only
    // local storage/runtime failures remain retryable.
    match error.pull_failure_class() {
        BlindVaultPullFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultPullFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_put_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-PUT-RECEIPT-FAILURE-CLASS 2026-08-28 by Codex] Preserve
    // capacity as an actionable route-selection signal while hiding every
    // lease, signature, object, and database detail from upstream relays.
    match error.put_failure_class() {
        BlindVaultPutFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultPutFailureClass::Capacity => TerminalReplyFailure::Capacity,
        BlindVaultPutFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_delete_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-DELETE-FAILURE-CLASS 2026-08-28 by Codex] Upstream relays
    // learn only whether the signed request is final or this replica is
    // unavailable; object existence and tombstone state remain indistinct.
    match error.delete_failure_class() {
        BlindVaultDeleteFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultDeleteFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_admission_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-ADMISSION-FAILURE-CLASS 2026-08-28 by Codex] Credential,
    // issuer, replay, and lease state remain indistinguishable outside the
    // encrypted terminal response boundary.
    match error.admission_failure_class() {
        BlindVaultAdmissionFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultAdmissionFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}

fn classify_lease_retire_failure(error: BlindVaultServiceError) -> TerminalReplyFailure {
    // [BLIND-VAULT-LEASE-RETIRE-FAILURE 2026-08-28 by Codex] Lease existence,
    // expiry, prior retirement, authority, and request conflicts collapse into
    // one permanent rejection outside the encrypted terminal boundary.
    match error.lease_retire_failure_class() {
        BlindVaultLeaseRetireFailureClass::Rejected => TerminalReplyFailure::Rejected,
        BlindVaultLeaseRetireFailureClass::Unavailable => TerminalReplyFailure::Unavailable,
    }
}
