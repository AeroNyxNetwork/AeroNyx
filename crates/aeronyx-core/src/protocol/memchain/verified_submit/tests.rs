// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/verified_submit/tests.rs
// ============================================
//! # Tests: verified chat submit
//!
//! Unit tests for the verified chat submit request/response contract, moved from the former
//! `protocol::memchain::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use super::*;

use crate::protocol::chat::ChatContentType;
use crate::protocol::memchain::{decode_memchain, encode_memchain, MemChainMessage};

#[test]
fn verified_chat_submit_binds_exact_envelope_and_verifies_terminal_receipt() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let sender = IdentityKeyPair::generate();
    let terminal = IdentityKeyPair::generate();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_secs();
    let mut envelope = ChatEnvelope {
        message_id: [0xE1; 16],
        sender: sender.public_key_bytes(),
        receiver: [0xE2; 32],
        timestamp: now,
        ciphertext: vec![0xE3; 48],
        nonce: [0xE4; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());
    let request =
        ChatRelayVerifiedSubmitRequestV1::signed([0xE5; 16], envelope.clone(), now, &sender)
            .expect("sign verified submit");
    request
        .verify_authentication()
        .expect("verify exact submit request");
    request
        .verify_signatures_for_replay()
        .expect("verify both signed layers without freshness admission");

    // [VERIFIED-SUBMIT-STALE-REPLAY 2026-09-07 by Codex] The helper keeps
    // the timestamp inside the sender signature while deliberately leaving
    // new-effect freshness to the server admission boundary.
    let mut stale_signed = request.clone();
    stale_signed.request_timestamp = now.saturating_sub(61);
    stale_signed.signature = sender.sign(&stale_signed.signing_digest());
    stale_signed
        .verify_signatures_for_replay()
        .expect("stale exact request keeps both valid signatures");
    assert!(stale_signed.verify_authentication().is_err());
    let mut tampered_outer = stale_signed.clone();
    tampered_outer.signature[0] ^= 0x01;
    assert!(tampered_outer.verify_signatures_for_replay().is_err());
    let mut tampered_envelope = stale_signed;
    tampered_envelope.envelope.signature[0] ^= 0x01;
    assert!(tampered_envelope.verify_signatures_for_replay().is_err());

    let encoded = encode_memchain(&MemChainMessage::ChatRelayVerifiedSubmitV1(request.clone()))
        .expect("encode verified submit");
    let decoded = decode_memchain(&encoded[1..]).expect("decode verified submit");
    let MemChainMessage::ChatRelayVerifiedSubmitV1(decoded_request) = decoded else {
        panic!("expected verified submit request");
    };
    decoded_request
        .verify_authentication()
        .expect("verify decoded request");

    let payload = encode_envelope(&envelope).expect("encode terminal payload");
    let receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
        [0xE6; 16],
        &payload,
        OnionRoutePurpose::MessageRelay,
        now,
        &terminal,
    );
    let response = ChatRelayVerifiedSubmitResponseV1 {
        request_id: request.request_id,
        message_id: envelope.message_id,
        result: CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1,
        terminal_receipt: Some(receipt),
    };
    response
        .verify_terminal_receipt(&envelope, &terminal.public_key_bytes())
        .expect("verify terminal receipt");
    response
        .verify_terminal_receipt_for_request(&request, &terminal.public_key_bytes())
        .expect("verify response against exact request");
    response
        .validate_for_request(&request)
        .expect("correlate delivered response");

    let entry_retry = ChatRelayVerifiedSubmitResponseV1::from_evidence(
        request.request_id,
        request.envelope.message_id,
        false,
        true,
        None,
    );
    entry_retry
        .validate_for_request(&request)
        .expect("correlate non-onion response");

    let mut mismatched_request = request.clone();
    mismatched_request.request_id[0] ^= 0x01;
    assert!(response.validate_for_request(&mismatched_request).is_err());
    assert!(response
        .verify_terminal_receipt_for_request(&mismatched_request, &terminal.public_key_bytes(),)
        .is_err());

    let mut mismatched_message = response.clone();
    mismatched_message.message_id[0] ^= 0x01;
    assert!(mismatched_message.validate_for_request(&request).is_err());

    let mut substituted = envelope.clone();
    substituted.ciphertext[0] ^= 0x01;
    assert!(response
        .verify_terminal_receipt(&substituted, &terminal.public_key_bytes())
        .is_err());
    assert!(response
        .verify_terminal_receipt(&envelope, &[0xA7; 32])
        .is_err());

    let mut forged_request = request;
    forged_request.request_id[0] ^= 0x01;
    assert!(forged_request.verify_authentication().is_err());
}

#[test]
fn verified_chat_submit_rejects_receipt_result_mismatch() {
    let response = ChatRelayVerifiedSubmitResponseV1 {
        request_id: [0xF1; 16],
        message_id: [0xF2; 16],
        result: CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1,
        terminal_receipt: Some(BlindRelayDeliveryReceipt {
            version: 2,
            route_id: [0xF3; 16],
            payload_commitment: [0xF4; 32],
            terminal_node_id: [0xF5; 32],
            delivered_at: 1,
            disposition: 1,
            signature: [0xF6; 64],
        }),
    };
    assert!(response.validate_shape().is_err());
}

#[test]
fn verified_chat_submit_result_labels_are_closed_and_stable() {
    // [CHAT-VERIFIED-SUBMIT-RESULT-LABELS 2026-08-23 by Codex] Keep the
    // status vocabulary stable for nodeboard and SDK consumers while the
    // wire discriminants remain numeric.
    assert_eq!(
        chat_verified_submit_result_label(CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1),
        Some("onion_and_entry")
    );
    assert_eq!(
        chat_verified_submit_result_label(CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1),
        Some("onion_only")
    );
    assert_eq!(
        chat_verified_submit_result_label(CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1),
        Some("entry_retry")
    );
    assert_eq!(
        chat_verified_submit_result_label(CHAT_VERIFIED_SUBMIT_REJECTED_V1),
        Some("rejected")
    );
    assert_eq!(chat_verified_submit_result_label(u8::MAX), None);
}

#[test]
fn verified_chat_submit_outcome_mapping_is_closed_and_stable() {
    // [CHAT-VERIFIED-SUBMIT-OUTCOME-MAPPING 2026-08-23 by Codex] All four
    // independent evidence combinations have one stable wire result. This
    // keeps relay implementations from inventing open-text client states.
    assert_eq!(
        chat_verified_submit_result_for_outcomes(true, true),
        CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1
    );
    assert_eq!(
        chat_verified_submit_result_for_outcomes(true, false),
        CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1
    );
    assert_eq!(
        chat_verified_submit_result_for_outcomes(false, true),
        CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1
    );
    assert_eq!(
        chat_verified_submit_result_for_outcomes(false, false),
        CHAT_VERIFIED_SUBMIT_REJECTED_V1
    );
}

#[test]
fn verified_chat_submit_response_builder_is_fail_closed() {
    let receipt = BlindRelayDeliveryReceipt {
        version: 2,
        route_id: [0x61; 16],
        payload_commitment: [0x62; 32],
        terminal_node_id: [0x63; 32],
        delivered_at: 1_800_001_001,
        disposition: 1,
        signature: [0x64; 64],
    };

    // [CHAT-VERIFIED-SUBMIT-RESPONSE-EVIDENCE 2026-08-23 by Codex] When
    // terminal evidence is internally consistent, the response keeps the
    // receipt and stays shape-valid.
    let delivered = ChatRelayVerifiedSubmitResponseV1::from_evidence(
        [0x65; 16],
        [0x66; 16],
        true,
        true,
        Some(receipt.clone()),
    );
    assert_eq!(delivered.result, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1);
    assert!(delivered.terminal_receipt.is_some());
    delivered.validate_shape().expect("valid delivered shape");

    // If a caller claims onion delivery without receipt bytes, clients must
    // receive only entry-custody retry semantics.
    let missing_receipt =
        ChatRelayVerifiedSubmitResponseV1::from_evidence([0x67; 16], [0x68; 16], true, true, None);
    assert_eq!(missing_receipt.result, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1);
    assert!(missing_receipt.terminal_receipt.is_none());
    missing_receipt
        .validate_shape()
        .expect("valid missing receipt shape");

    // If a caller did not observe verified delivery, any stray receipt is
    // discarded before the response reaches the client.
    let inconsistent = ChatRelayVerifiedSubmitResponseV1::from_evidence(
        [0x69; 16],
        [0x6A; 16],
        false,
        false,
        Some(receipt),
    );
    assert_eq!(inconsistent.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(inconsistent.terminal_receipt.is_none());
    inconsistent
        .validate_shape()
        .expect("valid fail-closed shape");

    let rejected = ChatRelayVerifiedSubmitResponseV1::rejected([0x6B; 16], [0x6C; 16]);
    assert_eq!(rejected.result, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    assert!(rejected.terminal_receipt.is_none());
    rejected
        .validate_shape()
        .expect("valid rejected response shape");
}

#[test]
fn verified_chat_submit_route_id_is_retry_stable_and_path_bound() {
    let request_id = [0x91; 16];
    let source = [0x92; 32];
    let middle = [0x93; 32];
    let terminal = [0x94; 32];
    let route_id = chat_verified_submit_route_id(&request_id, &source, &middle, &terminal);

    // [CHAT-VERIFIED-SUBMIT-ROUTE-ID 2026-08-23 by Codex] Reusing the
    // exact signed request and path must hit the same blind-relay replay
    // key, while path or request changes must not collide.
    assert_eq!(
        route_id,
        chat_verified_submit_route_id(&request_id, &source, &middle, &terminal)
    );
    assert_ne!(
        route_id,
        chat_verified_submit_route_id(&request_id, &source, &[0x95; 32], &terminal)
    );
    assert_ne!(
        route_id,
        chat_verified_submit_route_id(&[0x96; 16], &source, &middle, &terminal)
    );
}
