// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/frame_validation/tests.rs
// ============================================
//! # Tests: source terminal-frame validation
//!
//! Unit tests for source-sealed response verification, moved from the former
//! inline `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use super::*;

use aeronyx_core::protocol::anonymous_mailbox::{
    encode_anonymous_mailbox_terminal_frame, AnonymousMailboxOutcomeV1,
    AnonymousMailboxSourceSealedResponseV1, AnonymousMailboxSourceTerminalCarrierV1,
    AnonymousMailboxTicketIssueResponseV1, AnonymousMailboxTicketIssueV1,
};

use super::super::test_support::{target, NOW};

#[test]
fn source_sealed_ticket_response_is_one_shot_and_request_bound() {
    let target = target();
    let request = AnonymousMailboxTicketIssueV1::new(
        [0x41; 16],
        [0x42; 16],
        target.public_key_bytes(),
        [0x43; 32],
        NOW,
        NOW + 30,
        0,
    )
    .expect("ticket request");
    let request_frame = encode_anonymous_mailbox_terminal_frame(
        &AnonymousMailboxTerminalFrameV1::TicketIssue(request.clone()),
    )
    .expect("request frame");
    let route_id = [0x44; 16];
    let (carrier, mut session) = AnonymousMailboxSourceTerminalCarrierV1::prepare(
        route_id,
        target.public_key_bytes(),
        request_frame.clone(),
    )
    .expect("carrier");
    let response = AnonymousMailboxTicketIssueResponseV1::signed(
        &request,
        AnonymousMailboxOutcomeV1::Rejected,
        None,
        NOW,
        &target,
    )
    .expect("response");
    let response_frame = encode_anonymous_mailbox_terminal_frame(
        &AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
    )
    .expect("response frame");
    let sealed = AnonymousMailboxSourceSealedResponseV1::seal(
        route_id,
        carrier.context_commitment(),
        target.public_key_bytes(),
        carrier.reply_public_key(),
        &response_frame,
        &target,
    )
    .and_then(|value| value.encode())
    .expect("seal");
    let opened = session.open(&sealed).expect("open once");
    let decoded = decode_anonymous_mailbox_terminal_frame(&opened).expect("decode");
    assert!(verify_response(&request_frame, &decoded, &target.public_key_bytes()).is_ok());
    assert!(
        session.open(&sealed).is_err(),
        "response key is consumed after the first open"
    );
}
