// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/audit_anchor/tests.rs
// ============================================
//! # Tests: Relay custody audit anchors
//!
//! Unit tests for the relay custody audit anchors, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use aeronyx_core::crypto::IdentityKeyPair;

#[test]
fn relay_custody_anchor_file_boundary_is_exact_and_no_overwrite() {
    let directory = tempfile::tempdir().expect("audit anchor CLI directory");
    let path = directory.path().join("anchor.bin");
    let producer = IdentityKeyPair::from_bytes(&[0x85; 32]).expect("anchor producer");
    let anchor = CustodyAuditAnchorV1::signed(4, 65_540, 128 * 1024 * 1024, [0x86; 32], &producer)
        .expect("sign CLI anchor fixture");
    let frame = encode_custody_audit_anchor(&anchor).expect("encode CLI anchor fixture");
    let frame_sha256: [u8; 32] = Sha256::digest(&frame).into();

    write_new_relay_custody_anchor(&path, &frame).expect("publish exact CLI anchor");
    assert!(write_new_relay_custody_anchor(&path, &frame).is_err());
    let loaded = read_bounded_relay_custody_anchor(&path).expect("read bounded CLI anchor");
    let verified =
        verify_relay_custody_anchor_frame(&loaded, &frame_sha256, &producer.public_key_bytes(), 4)
            .expect("verify exact CLI anchor");
    assert_eq!(verified, anchor);

    let mut wrong_sha256 = frame_sha256;
    wrong_sha256[0] ^= 1;
    assert!(verify_relay_custody_anchor_frame(
        &loaded,
        &wrong_sha256,
        &producer.public_key_bytes(),
        4,
    )
    .is_err());
    assert!(verify_relay_custody_anchor_frame(
        &loaded,
        &frame_sha256,
        &producer.public_key_bytes(),
        5,
    )
    .is_err());

    let oversized_path = directory.path().join("oversized.bin");
    std::fs::write(
        &oversized_path,
        vec![0u8; MAX_CUSTODY_AUDIT_ANCHOR_FRAME_BYTES + 1],
    )
    .expect("write oversized CLI anchor fixture");
    assert!(read_bounded_relay_custody_anchor(&oversized_path).is_err());

    #[cfg(unix)]
    {
        let symlink_path = directory.path().join("anchor-link.bin");
        std::os::unix::fs::symlink(&path, &symlink_path)
            .expect("create audit anchor symlink fixture");
        assert!(read_bounded_relay_custody_anchor(&symlink_path).is_err());
    }
}
