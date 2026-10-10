// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/frame_validation.rs
// ============================================
//! # Source terminal-frame validation
//!
//! Owns request-frame admission, the inner `TicketIssue` target-pin check, and
//! verification that a decoded terminal response matches its request and the
//! pinned responder before the journal may record it.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, AnonymousMailboxOperationV1,
    AnonymousMailboxPullResultV1, AnonymousMailboxTerminalFrameV1,
    AnonymousMailboxTerminalResponseV1,
};

use super::AnonymousMailboxSourceError;

pub(super) fn ensure_request_frame(frame: &[u8]) -> Result<(), AnonymousMailboxSourceError> {
    match decode_anonymous_mailbox_terminal_frame(frame)
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?
    {
        AnonymousMailboxTerminalFrameV1::LeaseCreate(_)
        | AnonymousMailboxTerminalFrameV1::Put(_)
        | AnonymousMailboxTerminalFrameV1::PullOne(_)
        | AnonymousMailboxTerminalFrameV1::Ack(_)
        | AnonymousMailboxTerminalFrameV1::TicketIssue(_) => Ok(()),
        _ => Err(AnonymousMailboxSourceError::Rejected),
    }
}

pub(super) fn ensure_ticket_target(
    frame: &[u8],
    target_node_id: &[u8; 32],
) -> Result<(), AnonymousMailboxSourceError> {
    let request = decode_anonymous_mailbox_terminal_frame(frame)
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    match request {
        AnonymousMailboxTerminalFrameV1::TicketIssue(request)
            if &request.target_node_id != target_node_id =>
        {
            Err(AnonymousMailboxSourceError::Rejected)
        }
        AnonymousMailboxTerminalFrameV1::LeaseCreate(_)
        | AnonymousMailboxTerminalFrameV1::Put(_)
        | AnonymousMailboxTerminalFrameV1::PullOne(_)
        | AnonymousMailboxTerminalFrameV1::Ack(_)
        | AnonymousMailboxTerminalFrameV1::TicketIssue(_) => Ok(()),
        _ => Err(AnonymousMailboxSourceError::Rejected),
    }
}

pub(super) fn verify_response(
    request: &[u8],
    response: &AnonymousMailboxTerminalFrameV1,
    target: &[u8; 32],
) -> Result<(), AnonymousMailboxSourceError> {
    let request = decode_anonymous_mailbox_terminal_frame(request)
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    match (request, response) {
        (
            AnonymousMailboxTerminalFrameV1::LeaseCreate(request),
            AnonymousMailboxTerminalFrameV1::LeaseCreateResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::LeaseCreate,
            request.admission.ticket_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::Put(request),
            AnonymousMailboxTerminalFrameV1::PutResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::Put,
            request.item_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::PullOne(request),
            AnonymousMailboxTerminalFrameV1::PullOneResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::PullOne,
            request.request_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::Ack(request),
            AnonymousMailboxTerminalFrameV1::AckResponse(response),
        ) => verify_standard_response(
            AnonymousMailboxOperationV1::Ack,
            request.request_id,
            request
                .request_commitment()
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?,
            response,
            target,
        ),
        (
            AnonymousMailboxTerminalFrameV1::TicketIssue(request),
            AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
        ) => response
            .verify_for_request(&request, target)
            .map_err(|_| AnonymousMailboxSourceError::Rejected),
        _ => Err(AnonymousMailboxSourceError::Rejected),
    }
}

fn verify_standard_response(
    operation: AnonymousMailboxOperationV1,
    request_id: [u8; 16],
    commitment: [u8; 32],
    response: &AnonymousMailboxTerminalResponseV1,
    target: &[u8; 32],
) -> Result<(), AnonymousMailboxSourceError> {
    response
        .verify_for_request(operation, &request_id, &commitment, target)
        .map_err(|_| AnonymousMailboxSourceError::Rejected)?;
    match operation {
        AnonymousMailboxOperationV1::LeaseCreate
        | AnonymousMailboxOperationV1::Put
        | AnonymousMailboxOperationV1::Ack => {
            if response.sealed_payload.is_empty() {
                Ok(())
            } else {
                Err(AnonymousMailboxSourceError::Rejected)
            }
        }
        AnonymousMailboxOperationV1::PullOne => {
            if response.sealed_payload.is_empty() {
                Ok(())
            } else {
                AnonymousMailboxPullResultV1::decode(&response.sealed_payload)
                    .map(|_| ())
                    .map_err(|_| AnonymousMailboxSourceError::Rejected)
            }
        }
        AnonymousMailboxOperationV1::TicketIssue => Err(AnonymousMailboxSourceError::Rejected),
    }
}

#[cfg(test)]
mod tests;
