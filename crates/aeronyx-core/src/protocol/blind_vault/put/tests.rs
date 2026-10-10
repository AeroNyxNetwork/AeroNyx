// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/put/tests.rs
// ============================================
//! # Tests: immutable ciphertext writes
//!
//! Unit tests for immutable ciphertext writes, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::*;

use crate::protocol::blind_vault::test_support::{
    lease_key, node_key, signed_put, MAX_TTL_MS, NOW_MS,
};
use crate::protocol::blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, is_blind_vault_frame, BlindVaultFrame,
};

#[test]
fn signed_put_validates_and_round_trips() {
    let put = signed_put();
    put.validate_and_verify(NOW_MS, MAX_TTL_MS, &lease_key().public_key())
        .expect("valid put");

    let encoded = encode_blind_vault_frame(&BlindVaultFrame::Put(put.clone())).expect("encode put");
    assert!(is_blind_vault_frame(&encoded));
    let decoded = decode_blind_vault_frame(&encoded).expect("decode put");
    assert_eq!(decoded, BlindVaultFrame::Put(put));
}

#[test]
fn ciphertext_tampering_breaks_commitment_before_signature_check() {
    let mut put = signed_put();
    put.ciphertext[0] ^= 0xFF;
    assert_eq!(
        put.validate_and_verify(NOW_MS, MAX_TTL_MS, &lease_key().public_key()),
        Err(BlindVaultError::CommitmentMismatch)
    );
}

#[test]
fn only_coarse_padded_size_classes_are_accepted() {
    let mut put = signed_put();
    put.ciphertext.pop();
    put.ciphertext_commitment = sha256(&put.ciphertext);
    put.sign(&lease_key());
    assert_eq!(
        put.validate(NOW_MS, MAX_TTL_MS),
        Err(BlindVaultError::InvalidCiphertextSize { actual: 4095 })
    );
}

#[test]
fn ttl_policy_is_enforced_without_exposing_retention_semantics() {
    let mut put = signed_put();
    put.expires_at_ms = NOW_MS + MAX_TTL_MS + 1;
    put.sign(&lease_key());
    assert_eq!(
        put.validate(NOW_MS, MAX_TTL_MS),
        Err(BlindVaultError::LifetimeTooLong)
    );
}

#[test]
fn stored_receipt_is_node_bound_and_matches_exact_put() {
    let put = signed_put();
    let key = node_key();
    let mut receipt = BlindVaultStoredReceipt::from_put(
        &put,
        NOW_MS + 100,
        put.expires_at_ms,
        key.public_key_bytes(),
    );
    receipt.sign(&key).expect("matching node identity");
    receipt
        .validate_and_verify(&key.public_key())
        .expect("valid receipt");
    assert!(receipt.matches_put(&put));

    receipt.ciphertext_commitment[0] ^= 1;
    assert_eq!(
        receipt.validate_and_verify(&key.public_key()),
        Err(BlindVaultError::InvalidSignature)
    );
    assert!(!receipt.matches_put(&put));
}

#[test]
fn receipt_cannot_be_signed_as_another_node() {
    let put = signed_put();
    let mut receipt =
        BlindVaultStoredReceipt::from_put(&put, NOW_MS + 100, put.expires_at_ms, [4; 32]);
    assert_eq!(
        receipt.sign(&node_key()),
        Err(BlindVaultError::NodeIdentityMismatch)
    );
}
