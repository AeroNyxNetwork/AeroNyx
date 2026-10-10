// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/frame/tests.rs
// ============================================
//! # Tests: Blind Vault binary frames
//!
//! Unit tests for Blind Vault binary frames, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::*;

use crate::protocol::blind_vault::test_support::signed_put;

#[test]
fn blind_vault_magic_detection_does_not_accept_other_or_truncated_protocols() {
    // [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] Onion terminal
    // dispatch must distinguish a declared Blind Vault frame from legacy
    // chat bytes before decoding, without treating a short prefix as valid.
    assert!(!is_blind_vault_frame(b""));
    assert!(!is_blind_vault_frame(b"ANB"));
    assert!(!is_blind_vault_frame(b"ANMC\x00\x01"));
    assert!(is_blind_vault_frame(b"ANBV"));
    assert!(matches!(
        decode_blind_vault_frame(b"ANBV"),
        Err(BlindVaultError::TruncatedFrame)
    ));
}

#[test]
fn decoder_rejects_unknown_kind_and_trailing_body_bytes() {
    let put = signed_put();
    let mut encoded = encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode put");
    encoded[6] = 0xFE;
    assert_eq!(
        decode_blind_vault_frame(&encoded),
        Err(BlindVaultError::UnknownFrameKind(0xFE))
    );

    let put = signed_put();
    let mut encoded = encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode put");
    encoded.push(0);
    assert_eq!(
        decode_blind_vault_frame(&encoded),
        Err(BlindVaultError::Deserialization)
    );
}
