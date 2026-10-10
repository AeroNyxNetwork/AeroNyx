// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/witness_receipt/tests.rs
// ============================================
//! # Tests: Relay custody witness receipts
//!
//! Unit tests for the relay custody witness receipts, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::encode_custody_audit_anchor;

#[test]
fn relay_custody_witness_receipt_file_boundary_and_anchor_binding_are_exact() {
    let directory = tempfile::tempdir().expect("audit witness CLI directory");
    let receipt_path = directory.path().join("witness.bin");
    let producer = IdentityKeyPair::from_bytes(&[0x88; 32]).expect("anchor producer");
    let witness = IdentityKeyPair::from_bytes(&[0x89; 32]).expect("anchor witness");
    let anchor = CustodyAuditAnchorV1::signed(5, 65_541, 256 * 1024 * 1024, [0x8a; 32], &producer)
        .expect("sign anchor fixture");
    let anchor_frame = encode_custody_audit_anchor(&anchor).expect("encode anchor fixture");
    let anchor_sha256: [u8; 32] = Sha256::digest(&anchor_frame).into();
    let receipt = CustodyAuditWitnessReceiptV1::signed(
        producer.public_key_bytes(),
        5,
        anchor_sha256,
        1_787_200_100,
        5,
        anchor_sha256,
        CUSTODY_AUDIT_WITNESS_ADVANCED_V1,
        &witness,
    )
    .expect("sign witness fixture");
    let receipt_frame =
        encode_custody_audit_witness_receipt(&receipt).expect("encode witness fixture");

    write_new_relay_custody_artifact(
        &receipt_path,
        &receipt_frame,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )
    .expect("publish witness receipt");
    assert!(write_new_relay_custody_artifact(
        &receipt_path,
        &receipt_frame,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )
    .is_err());
    let loaded = read_bounded_relay_custody_artifact(
        &receipt_path,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )
    .expect("read witness receipt");
    let receipt_sha256: [u8; 32] = Sha256::digest(&loaded).into();
    let decoded = verify_relay_custody_witness_receipt_frame(
        &loaded,
        &receipt_sha256,
        &anchor,
        &anchor_sha256,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        5,
    )
    .expect("verify canonical witness receipt frame");
    decoded
        .verify_accepted_for_anchor(
            &anchor,
            &anchor_sha256,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            5,
        )
        .expect("verify witness receipt binding");

    let wrong_anchor_sha256 = [0x8b; 32];
    assert!(decoded
        .verify_accepted_for_anchor(
            &anchor,
            &wrong_anchor_sha256,
            &producer.public_key_bytes(),
            &witness.public_key_bytes(),
            5,
        )
        .is_err());
    assert!(verify_relay_custody_witness_receipt_frame(
        &loaded,
        &[0x8c; 32],
        &anchor,
        &anchor_sha256,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        5,
    )
    .is_err());

    let oversized_path = directory.path().join("oversized-witness.bin");
    std::fs::write(
        &oversized_path,
        vec![0u8; MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES + 1],
    )
    .expect("write oversized receipt fixture");
    assert!(read_bounded_relay_custody_artifact(
        &oversized_path,
        MAX_CUSTODY_AUDIT_WITNESS_RECEIPT_FRAME_BYTES,
        "audit witness receipt",
    )
    .is_err());
}
