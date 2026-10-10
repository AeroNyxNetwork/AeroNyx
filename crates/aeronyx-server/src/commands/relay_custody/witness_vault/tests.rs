// ============================================
// File: crates/aeronyx-server/src/commands/relay_custody/witness_vault/tests.rs
// ============================================
//! # Tests: Relay custody producer witness vault
//!
//! Unit tests for the relay custody producer witness vault, moved from the former
//! `main.rs` `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from aeronyx-server/src/main.rs; bodies unchanged.

use super::*;

use aeronyx_core::protocol::discovery::{NodeCapability, NodeDescriptor, SignedNodeDescriptor};

#[test]
fn custody_witness_vault_status_is_stable_and_fail_closed() {
    // [CUSTODY-WITNESS-VAULT-AUDIT 2026-08-17 by Codex] Monitoring labels
    // must not depend on count ordering: readiness wins only when the full
    // policy says so, while any unresolved adverse evidence is explicit.
    let collecting = CustodyAuditWitnessReceiptPolicyEvidence {
        configured: 1,
        missing: 1,
        minimum_verified: 1,
        ..CustodyAuditWitnessReceiptPolicyEvidence::default()
    };
    assert_eq!(
        custody_audit_witness_policy_status(&collecting),
        "collecting"
    );

    let adverse = CustodyAuditWitnessReceiptPolicyEvidence {
        configured: 1,
        fresh_verified: 1,
        adverse: 1,
        minimum_verified: 1,
        ..CustodyAuditWitnessReceiptPolicyEvidence::default()
    };
    assert_eq!(custody_audit_witness_policy_status(&adverse), "adverse");

    let inconsistent = CustodyAuditWitnessReceiptPolicyEvidence {
        configured: 1,
        fresh_verified: 1,
        accepted: 1,
        minimum_verified: 1,
        ..CustodyAuditWitnessReceiptPolicyEvidence::default()
    };
    assert_eq!(
        custody_audit_witness_policy_status(&inconsistent),
        "invalid"
    );

    let ready = CustodyAuditWitnessReceiptPolicyEvidence {
        configured: 1,
        fresh_verified: 1,
        accepted: 1,
        minimum_verified: 1,
        quorum_satisfied: true,
        quorum_valid_through: Some(1_100),
        ..CustodyAuditWitnessReceiptPolicyEvidence::default()
    };
    assert_eq!(custody_audit_witness_policy_status(&ready), "ready");
    assert_eq!(
        custody_audit_witness_renewal_fields(&ready, 1_000, 400),
        (Some(1_100), Some(100), 100, true)
    );
    assert_eq!(optional_u64_label(None, "s"), "unavailable");
    assert_eq!(optional_u64_label(Some(0), "s"), "0s");

    let threshold = CustodyAuditWitnessReceiptPolicyEvidence {
        configured: 2,
        fresh_verified: 1,
        accepted: 1,
        missing: 1,
        minimum_verified: 2,
        ..CustodyAuditWitnessReceiptPolicyEvidence::default()
    };
    assert_eq!(
        custody_audit_witness_policy_status(&threshold),
        "collecting"
    );
}

#[test]
fn custody_witness_operator_snapshot_imports_only_exact_pins() {
    // [CUSTODY-WITNESS-OPERATOR-COLLECT 2026-08-18 by Codex] A valid but
    // unrelated descriptor must not enter the ephemeral transport view;
    // a bad pinned signature must remain visible as aggregate rejection.
    let now = 1_700_000_100;
    let pinned_identity =
        IdentityKeyPair::from_bytes(&[0x91; 32]).expect("pinned witness identity");
    let unrelated_identity =
        IdentityKeyPair::from_bytes(&[0x92; 32]).expect("unrelated peer identity");
    let signed_descriptor = |identity: &IdentityKeyPair, endpoint: &str| {
        let mut descriptor =
            NodeDescriptor::new(identity.public_key_bytes(), 1, now - 10, now + 600, "0.1.0");
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor
            .capabilities
            .push(NodeCapability::EncryptedStorage);
        SignedNodeDescriptor::sign(descriptor, identity).expect("sign witness descriptor")
    };
    let pinned = signed_descriptor(&pinned_identity, "https://witness.example");
    let unrelated = signed_descriptor(&unrelated_identity, "https://unrelated.example");
    let snapshot = NodeBootstrapSnapshot::new(now, vec![unrelated, pinned.clone()]);
    let directory = tempfile::tempdir().expect("witness snapshot directory");
    let path = directory.path().join("snapshot.json");
    std::fs::write(
        &path,
        snapshot
            .to_json_pretty()
            .expect("serialize witness snapshot"),
    )
    .expect("write witness snapshot");

    let loaded =
        load_relay_custody_witness_snapshot(&path, &[pinned_identity.public_key_bytes()], 4, now)
            .expect("load pinned witness snapshot");
    assert_eq!(loaded.records, 2);
    assert_eq!(loaded.pinned_records, 1);
    assert_eq!(loaded.verified_records, 1);
    assert_eq!(loaded.rejected_records, 0);
    assert!(loaded
        .peer_store
        .get_valid(&pinned_identity.public_key_bytes(), now)
        .is_some());
    assert!(loaded
        .peer_store
        .get_valid(&unrelated_identity.public_key_bytes(), now)
        .is_none());

    let mut tampered = pinned;
    tampered.signature[0] ^= 0x01;
    let invalid_path = directory.path().join("invalid-snapshot.json");
    std::fs::write(
        &invalid_path,
        NodeBootstrapSnapshot::new(now, vec![tampered])
            .to_json_pretty()
            .expect("serialize invalid witness snapshot"),
    )
    .expect("write invalid witness snapshot");
    let rejected = load_relay_custody_witness_snapshot(
        &invalid_path,
        &[pinned_identity.public_key_bytes()],
        4,
        now,
    )
    .expect("audit invalid pinned snapshot");
    assert_eq!(rejected.pinned_records, 1);
    assert_eq!(rejected.verified_records, 0);
    assert_eq!(rejected.rejected_records, 1);
}
