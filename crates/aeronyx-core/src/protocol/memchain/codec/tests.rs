// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/codec/tests.rs
// ============================================
//! # Tests: bounded wire codec
//!
//! Unit tests for the bounded `MemChainMessage` codec, its magic byte, and the anonymous
//! mailbox canonical outer-byte gate, moved from the former
//! `protocol::memchain::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use super::*;

use crate::crypto::IdentityKeyPair;
use crate::protocol::anonymous_mailbox::{
    AnonymousMailboxRouteRequestV1, AnonymousMailboxRouteResponseV1,
};

// ============================================
// Tests
// ============================================

// ── Existing tests (preserved verbatim) ─────────────────────────────

#[test]
fn test_magic_does_not_collide_with_ip() {
    assert_ne!(MEMCHAIN_MAGIC >> 4, 4, "Must not collide with IPv4");
    assert_ne!(MEMCHAIN_MAGIC >> 4, 6, "Must not collide with IPv6");
}

#[test]
fn test_memchain_codec_preserves_wire_bytes_and_small_trailing_compatibility() {
    let msg = MemChainMessage::Ping {
        nonce: 0x0102_0304_0506_0708,
    };
    let encoded = encode_memchain(&msg).expect("bounded encode");
    assert_eq!(encoded[0], MEMCHAIN_MAGIC);
    assert_eq!(
        &encoded[1..],
        bincode::serialize(&msg)
            .expect("legacy wire encoding")
            .as_slice(),
        "bounded encoding must not alter existing MemChain wire bytes"
    );

    let mut payload = encoded[1..].to_vec();
    payload.extend_from_slice(&[0xAA, 0xBB]);
    match decode_memchain(&payload).expect("decode with legacy trailing bytes") {
        MemChainMessage::Ping { nonce } => {
            assert_eq!(nonce, 0x0102_0304_0506_0708);
        }
        other => panic!("Expected Ping, got {other:?}"),
    }
}

#[test]
fn test_memchain_codec_rejects_oversized_output_and_padded_input() {
    let oversized = MemChainMessage::ChatExpired {
        message_ids: vec![[0xCC; 16]; MAX_MEMCHAIN_PAYLOAD_BYTES as usize / 16 + 1],
        receiver: [0x42; 32],
    };
    assert!(
        encode_memchain(&oversized).is_err(),
        "sender must not create a MemChain frame that receivers reject"
    );

    let msg = MemChainMessage::Ping { nonce: 9 };
    let mut padded = bincode::serialize(&msg).expect("legacy wire encoding");
    padded.resize(MAX_MEMCHAIN_PAYLOAD_BYTES as usize + 1, 0);
    assert!(
        decode_memchain(&padded).is_err(),
        "ignored trailing padding must not bypass the complete input ceiling"
    );
}

fn anonymous_mailbox_route_fixtures() -> (
    AnonymousMailboxRouteRequestV1,
    AnonymousMailboxRouteResponseV1,
) {
    let source = IdentityKeyPair::from_bytes(&[0xE8; 32]).expect("source identity");
    let target = IdentityKeyPair::from_bytes(&[0xE9; 32]).expect("target identity");
    let request = AnonymousMailboxRouteRequestV1::signed(
        [0xEA; 16],
        target.public_key_bytes(),
        vec![0xEB; 96],
        1_800_000_100,
        &source,
    )
    .expect("route request");
    let response = AnonymousMailboxRouteResponseV1::signed(
        &request,
        crate::protocol::anonymous_mailbox::AnonymousMailboxOutcomeV1::Accepted,
        vec![0xEC; 80],
        1_800_000_101,
        &target,
    )
    .expect("route response");
    (request, response)
}

#[test]
fn anonymous_mailbox_request_rejects_trailing_outer_bytes() {
    let (request, _) = anonymous_mailbox_route_fixtures();
    let mut encoded = encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(request))
        .expect("encode request");
    encoded.push(0xA5);
    assert!(
        decode_memchain(&encoded[1..]).is_err(),
        "mailbox request must reject an otherwise valid frame with trailing bytes"
    );
}

#[test]
fn anonymous_mailbox_response_rejects_trailing_outer_bytes() {
    let (_, response) = anonymous_mailbox_route_fixtures();
    let mut encoded = encode_memchain(&MemChainMessage::AnonymousMailboxRouteResponseV1(response))
        .expect("encode response");
    encoded.push(0x5A);
    assert!(
        decode_memchain(&encoded[1..]).is_err(),
        "mailbox response must reject an otherwise valid frame with trailing bytes"
    );
}

#[test]
fn anonymous_mailbox_canonical_roundtrip_preserves_request_commitment() {
    let (request, response) = anonymous_mailbox_route_fixtures();
    let expected_commitment = request.request_commitment().expect("request commitment");

    let request_encoded =
        encode_memchain(&MemChainMessage::AnonymousMailboxRouteV1(request.clone()))
            .expect("encode request");
    let decoded_request = match decode_memchain(&request_encoded[1..]).expect("decode request") {
        MemChainMessage::AnonymousMailboxRouteV1(value) => value,
        other => panic!("expected anonymous mailbox request, got {other:?}"),
    };
    assert_eq!(decoded_request, request);
    assert_eq!(
        decoded_request
            .request_commitment()
            .expect("decoded request commitment"),
        expected_commitment
    );

    let response_encoded = encode_memchain(&MemChainMessage::AnonymousMailboxRouteResponseV1(
        response.clone(),
    ))
    .expect("encode response");
    let decoded_response = match decode_memchain(&response_encoded[1..]).expect("decode response") {
        MemChainMessage::AnonymousMailboxRouteResponseV1(value) => value,
        other => panic!("expected anonymous mailbox response, got {other:?}"),
    };
    assert_eq!(decoded_response, response);
    assert_eq!(decoded_response.request_commitment, expected_commitment);
}
