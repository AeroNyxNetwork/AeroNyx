// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/pull/tests.rs
// ============================================
//! # Tests: encrypted-object recovery
//!
//! Unit tests for encrypted-object recovery, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::*;

use crate::protocol::blind_vault::frame::FRAME_KIND_PUT;
use crate::protocol::blind_vault::test_support::{node_key, NOW_MS};
use crate::protocol::blind_vault::MAX_BLIND_VAULT_MUTATION_FRAME_BYTES;

fn signed_pull_response() -> BlindVaultPullResponse {
    let ciphertext = vec![0xB7; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[3]];
    let object = BlindVaultRecoveredObject {
        object_id: [31; 32],
        ciphertext_commitment: sha256(&ciphertext),
        ciphertext,
        expires_at_ms: NOW_MS + 24 * 60 * 60 * 1_000,
    };
    let second_ciphertext = vec![0xC3; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[3]];
    let second_object = BlindVaultRecoveredObject {
        object_id: [32; 32],
        ciphertext_commitment: sha256(&second_ciphertext),
        ciphertext: second_ciphertext,
        expires_at_ms: NOW_MS + 24 * 60 * 60 * 1_000,
    };
    let node = node_key();
    let mut response = BlindVaultPullResponse::new(
        [1; 32],
        vec![object, second_object],
        vec![1; 49],
        NOW_MS,
        node.public_key_bytes(),
    );
    response.sign(&node).expect("sign pull response");
    response
}

#[test]
fn signed_pull_page_round_trips_above_mutation_ceiling() {
    let request = BlindVaultPullRequest {
        version: BLIND_VAULT_PROTOCOL_VERSION,
        lease_id: [1; 32],
        read_capability: [33; 32],
        continuation_cursor: Vec::new(),
        limit: MAX_BLIND_VAULT_PULL_OBJECTS as u16,
    };
    request.validate().expect("valid pull request");
    let request_frame = encode_blind_vault_frame(&BlindVaultFrame::PullRequest(request.clone()))
        .expect("encode pull request");
    assert_eq!(
        decode_blind_vault_frame(&request_frame).expect("decode pull request"),
        BlindVaultFrame::PullRequest(request)
    );

    let response = signed_pull_response();
    response
        .validate_and_verify(&node_key().public_key())
        .expect("valid signed response");
    let encoded = encode_blind_vault_frame(&BlindVaultFrame::PullResponse(response.clone()))
        .expect("encode pull response");
    assert!(encoded.len() as u64 > MAX_BLIND_VAULT_MUTATION_FRAME_BYTES);
    assert_eq!(
        decode_blind_vault_frame(&encoded).expect("decode pull response"),
        BlindVaultFrame::PullResponse(response)
    );

    let mut wrong_kind = encoded;
    wrong_kind[6] = FRAME_KIND_PUT;
    assert_eq!(
        decode_blind_vault_frame(&wrong_kind),
        Err(BlindVaultError::FrameTooLarge)
    );
}

#[test]
fn pull_page_rejects_ciphertext_and_cursor_tampering() {
    let mut response = signed_pull_response();
    response.objects[0].ciphertext[0] ^= 1;
    assert_eq!(
        response.validate_and_verify(&node_key().public_key()),
        Err(BlindVaultError::CommitmentMismatch)
    );

    let mut response = signed_pull_response();
    response.continuation_cursor[0] ^= 1;
    assert_eq!(
        response.validate_and_verify(&node_key().public_key()),
        Err(BlindVaultError::InvalidSignature)
    );
}
