// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/test_support.rs
// ============================================
//! # Blind Vault test fixtures
//!
//! Shared deterministic keys, clock constants, and the signed put fixture for
//! the per-module Blind Vault test suites, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use crate::crypto::keys::IdentityKeyPair;

use super::put::BlindVaultPutRequest;
use super::BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES;

pub(super) const NOW_MS: u64 = 1_800_000_000_000;
pub(super) const MAX_TTL_MS: u64 = 30 * 24 * 60 * 60 * 1_000;

pub(super) fn lease_key() -> IdentityKeyPair {
    IdentityKeyPair::from_bytes(&[7; 32]).expect("valid deterministic lease key")
}

pub(super) fn node_key() -> IdentityKeyPair {
    IdentityKeyPair::from_bytes(&[9; 32]).expect("valid deterministic node key")
}

pub(super) fn admin_key() -> IdentityKeyPair {
    IdentityKeyPair::from_bytes(&[11; 32]).expect("valid deterministic admin key")
}

pub(super) fn admission_issuer_key() -> IdentityKeyPair {
    IdentityKeyPair::from_bytes(&[15; 32]).expect("valid deterministic admission issuer key")
}

pub(super) fn signed_put() -> BlindVaultPutRequest {
    let mut put = BlindVaultPutRequest::new(
        [1; 32],
        [2; 32],
        [3; 16],
        vec![0xA5; BLIND_VAULT_CIPHERTEXT_SIZE_CLASSES[0]],
        NOW_MS + 24 * 60 * 60 * 1_000,
    );
    put.sign(&lease_key());
    put
}
