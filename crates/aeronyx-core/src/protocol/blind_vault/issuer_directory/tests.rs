// ============================================
// File: crates/aeronyx-core/src/protocol/blind_vault/issuer_directory/tests.rs
// ============================================
//! # Tests: blind-admission issuer directory and updates
//!
//! Unit tests for blind-admission issuer directory and updates, moved from the former
//! `protocol::blind_vault::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/blind_vault.rs`; bodies unchanged.

use super::*;

use crate::protocol::blind_vault::frame::{
    deserialize_body, serialize_body, FRAME_KIND_BLIND_ISSUER_DIRECTORY,
};
use crate::protocol::blind_vault::test_support::{admission_issuer_key, node_key, NOW_MS};
use crate::protocol::blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultFrame,
    MAX_BLIND_VAULT_MUTATION_FRAME_BYTES,
};

fn signed_blind_issuer_directory() -> BlindVaultBlindIssuerDirectory {
    let mut epochs = vec![
        BlindVaultBlindIssuerEpoch::new(
            vec![0x31; 64],
            NOW_MS - 60_000,
            NOW_MS + 24 * 60 * 60 * 1_000,
            7 * 24 * 60 * 60 * 1_000,
        ),
        BlindVaultBlindIssuerEpoch::new(
            vec![0x32; 72],
            NOW_MS + 12 * 60 * 60 * 1_000,
            NOW_MS + 2 * 24 * 60 * 60 * 1_000,
            7 * 24 * 60 * 60 * 1_000,
        ),
    ];
    epochs.sort_by_key(|epoch| epoch.issuer_key_id);
    let node = node_key();
    let mut directory =
        BlindVaultBlindIssuerDirectory::new(NOW_MS, node.public_key_bytes(), epochs);
    directory.sign(&node).expect("sign issuer directory");
    directory
}

// [BLIND-VAULT-ISSUER-UPDATE 2026-07-23 by Codex] The update signature
// authenticates the complete canonical generation independently of the
// transport that carries it to a storage node.
fn signed_blind_issuer_update() -> BlindVaultBlindIssuerUpdate {
    let mut epochs = signed_blind_issuer_directory().epochs;
    epochs.sort_by_key(|epoch| epoch.issuer_key_id);
    let authority = admission_issuer_key();
    let mut update =
        BlindVaultBlindIssuerUpdate::new(1, NOW_MS, authority.public_key_bytes(), epochs);
    update.sign(&authority).expect("sign issuer update");
    update
}

// [BLIND-VAULT-ISSUER-DIRECTORY 2026-07-23 by Codex] Clients authenticate
// key rotation against the node descriptor identity before creating a
// blinded token; the directory carries public policy only.
#[test]
fn signed_issuer_directory_is_fresh_canonical_and_round_trips() {
    let directory = signed_blind_issuer_directory();
    directory
        .validate_and_verify(NOW_MS + 1_000, 60_000, 5_000, &node_key().public_key())
        .expect("valid issuer directory");

    let encoded =
        encode_blind_vault_frame(&BlindVaultFrame::BlindIssuerDirectory(directory.clone()))
            .expect("encode issuer directory");
    assert_eq!(encoded[6], FRAME_KIND_BLIND_ISSUER_DIRECTORY);
    assert_eq!(
        decode_blind_vault_frame(&encoded).expect("decode issuer directory"),
        BlindVaultFrame::BlindIssuerDirectory(directory)
    );
}

#[test]
fn issuer_directory_rejects_key_tampering_order_drift_and_staleness() {
    let mut key_tampered = signed_blind_issuer_directory();
    key_tampered.epochs[0].public_key_der[0] ^= 1;
    assert_eq!(
        key_tampered.validate_and_verify(NOW_MS, 60_000, 5_000, &node_key().public_key(),),
        Err(BlindVaultError::BlindIssuerKeyIdMismatch)
    );

    let mut reordered = signed_blind_issuer_directory();
    reordered.epochs.reverse();
    assert_eq!(
        reordered.validate_and_verify(NOW_MS, 60_000, 5_000, &node_key().public_key(),),
        Err(BlindVaultError::BlindIssuerEpochOrderInvalid)
    );

    let stale = signed_blind_issuer_directory();
    assert_eq!(
        stale.validate_and_verify(NOW_MS + 60_001, 60_000, 5_000, &node_key().public_key(),),
        Err(BlindVaultError::IssuerDirectoryTimestampOutsideWindow)
    );
}

#[test]
fn authority_signed_issuer_update_is_canonical_and_fresh() {
    let update = signed_blind_issuer_update();
    update
        .validate_and_verify(
            NOW_MS + 1_000,
            60_000,
            5_000,
            &admission_issuer_key().public_key(),
        )
        .expect("valid authority-signed issuer update");

    let encoded = serialize_body(&update, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)
        .expect("serialize transport-independent update");
    let decoded: BlindVaultBlindIssuerUpdate =
        deserialize_body(&encoded, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES)
            .expect("deserialize transport-independent update");
    assert_eq!(decoded, update);
}

#[test]
fn issuer_update_rejects_forgery_wrong_authority_and_staleness() {
    let mut forged = signed_blind_issuer_update();
    forged.generation = 2;
    assert_eq!(
        forged.validate_and_verify(NOW_MS, 60_000, 5_000, &admission_issuer_key().public_key(),),
        Err(BlindVaultError::InvalidSignature)
    );

    let wrong_authority = IdentityKeyPair::from_bytes(&[29; 32]).expect("wrong authority key");
    let update = signed_blind_issuer_update();
    assert_eq!(
        update.validate_and_verify(NOW_MS, 60_000, 5_000, &wrong_authority.public_key(),),
        Err(BlindVaultError::BlindIssuerAuthorityMismatch)
    );
    assert_eq!(
        update.validate_and_verify(
            NOW_MS + 60_001,
            60_000,
            5_000,
            &admission_issuer_key().public_key(),
        ),
        Err(BlindVaultError::BlindIssuerUpdateTimestampOutsideWindow)
    );
}

#[test]
fn issuer_update_rejects_reserved_generation_and_empty_epoch_set() {
    let authority = admission_issuer_key();
    let mut generation_zero = BlindVaultBlindIssuerUpdate::new(
        0,
        NOW_MS,
        authority.public_key_bytes(),
        signed_blind_issuer_directory().epochs,
    );
    assert_eq!(
        generation_zero.sign(&authority),
        Err(BlindVaultError::InvalidBlindIssuerUpdateGeneration)
    );

    let mut empty =
        BlindVaultBlindIssuerUpdate::new(1, NOW_MS, authority.public_key_bytes(), Vec::new());
    assert_eq!(
        empty.sign(&authority),
        Err(BlindVaultError::BlindIssuerUpdateHasNoEpochs)
    );
}
