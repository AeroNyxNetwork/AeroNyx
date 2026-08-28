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
//! - Executes capability-authenticated Blind Vault recovery.
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
//! Last Modified: v1.0.0-BlindVaultInlinePull - Initial anonymous pull reply.
//! ============================================================================

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{
    decode_blind_vault_frame, decode_onion_reply_request, encode_blind_vault_frame,
    encode_onion_sealed_response, seal_onion_reply, BlindVaultFrame, BlindVaultPullResponse,
    BlindVaultRecoveredObject, ONION_REPLY_RESPONSE_SIZE_CLASSES,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

use crate::services::{BlindVaultPullFailureClass, BlindVaultService, BlindVaultServiceError};

/// Coarse terminal-reply failure class consumed by blind-relay orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TerminalReplyFailure {
    /// The request, capability, cursor, or protocol frame is invalid.
    Rejected,
    /// The selected inline response class cannot carry the recovered page.
    ResponseTooLarge,
    /// Replica storage or response sealing is temporarily unavailable.
    Unavailable,
}

/// Successfully sealed terminal reply with no plaintext metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TerminalReply {
    pub(super) opaque_response_b64: String,
}

/// Executes one inline Blind Vault pull and returns a fixed-size sealed reply.
pub(super) fn execute_blind_vault_inline_pull(
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

    let BlindVaultFrame::PullRequest(pull_request) =
        decode_blind_vault_frame(&reply_request.payload)
            .map_err(|_| TerminalReplyFailure::Rejected)?
    else {
        return Err(TerminalReplyFailure::Rejected);
    };
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
    let encoded_pull_response =
        encode_blind_vault_frame(&BlindVaultFrame::PullResponse(pull_response))
            .map_err(|_| TerminalReplyFailure::ResponseTooLarge)?;
    let sealed = seal_onion_reply(
        route_id,
        &reply_request,
        &encoded_pull_response,
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
        opaque_response_b64: BASE64.encode(encoded_sealed),
    })
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
