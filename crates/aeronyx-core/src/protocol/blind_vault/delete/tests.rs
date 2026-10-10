// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/delete/tests.rs
// ============================================
//! # Tests: object deletion
//!
//! Unit tests for object deletion, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::*;

use crate::protocol::blind_vault::test_support::{admin_key, node_key, signed_put, NOW_MS};

#[test]
fn administration_delete_and_node_receipt_are_independently_signed() {
    let mut delete = BlindVaultDeleteRequest::new([1; 32], [2; 32], [6; 16], NOW_MS);
    delete.sign(&admin_key());
    delete
        .validate_and_verify(NOW_MS + 50, 1_000, &admin_key().public_key())
        .expect("valid admin deletion");

    let node = node_key();
    let mut receipt = BlindVaultDeletedReceipt::new(
        &delete,
        signed_put().ciphertext_commitment,
        NOW_MS + 100,
        node.public_key_bytes(),
    );
    receipt.sign(&node).expect("matching node key");
    receipt
        .validate_and_verify(&node.public_key())
        .expect("valid deletion receipt");
    assert!(receipt.matches_delete(&delete));

    let encoded = encode_blind_vault_frame(&BlindVaultFrame::DeletedReceipt(receipt.clone()))
        .expect("encode deletion receipt");
    assert_eq!(
        decode_blind_vault_frame(&encoded).expect("decode deletion receipt"),
        BlindVaultFrame::DeletedReceipt(receipt)
    );
}

#[test]
fn deletion_timestamp_has_bounded_replay_window() {
    let mut delete = BlindVaultDeleteRequest::new([1; 32], [2; 32], [6; 16], NOW_MS - 1_001);
    delete.sign(&admin_key());
    assert_eq!(
        delete.validate_and_verify(NOW_MS, 1_000, &admin_key().public_key()),
        Err(BlindVaultError::RequestTimestampOutsideWindow)
    );
}
