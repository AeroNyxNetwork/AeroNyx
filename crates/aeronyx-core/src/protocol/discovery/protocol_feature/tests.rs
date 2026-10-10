// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/protocol_feature/tests.rs
// ============================================
//! # Tests: signed protocol-feature negotiation
//!
//! Unit tests for signed protocol-feature negotiation, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use crate::crypto::IdentityKeyPair;

use crate::protocol::discovery::test_support::descriptor_for;
use crate::protocol::discovery::{
    decode_discovery_message, encode_discovery_message, NodeDescriptor, NodeDiscoveryMessage,
    SignedNodeDescriptor, NODE_DESCRIPTOR_SCHEMA_VERSION,
};

#[test]
fn signed_protocol_features_preserve_schema_and_detect_downgrade() {
    let identity = IdentityKeyPair::generate();
    let failure_receipt = NodeProtocolFeature::BlindRelayFailureReceiptV1;
    let purpose_receipt = NodeProtocolFeature::PurposeBoundDeliveryReceiptV2;
    let direct_relay_auth = NodeProtocolFeature::DirectPeerRelayAuthV2;
    let direct_relay_receipt = NodeProtocolFeature::DirectPeerRelayReceiptV2;
    let direct_relay_target_binding = NodeProtocolFeature::DirectPeerRelayTargetBindingV3;
    let onion_reply = NodeProtocolFeature::OnionReplyV1;
    let onion_blind_admission = NodeProtocolFeature::OnionBlindLeaseAdmissionV1;
    let onion_put_receipt = NodeProtocolFeature::OnionBlindVaultPutReceiptV1;
    let onion_lease_retire = NodeProtocolFeature::OnionBlindVaultLeaseRetireV1;
    let onion_lease_renewal = NodeProtocolFeature::OnionBlindVaultLeaseRenewalV1;
    let onion_lease_status = NodeProtocolFeature::OnionBlindVaultLeaseStatusV1;
    let onion_lease_inventory = NodeProtocolFeature::OnionBlindVaultLeaseInventoryV1;
    let descriptor = descriptor_for(&identity).with_protocol_features([
        purpose_receipt,
        failure_receipt,
        direct_relay_auth,
        direct_relay_receipt,
        direct_relay_target_binding,
        onion_reply,
        onion_blind_admission,
        onion_put_receipt,
        onion_lease_retire,
        onion_lease_renewal,
        onion_lease_status,
        onion_lease_inventory,
        purpose_receipt,
    ]);

    assert_eq!(descriptor.schema_version, NODE_DESCRIPTOR_SCHEMA_VERSION);
    assert_eq!(
        descriptor.software_version,
        "test+anpf1-brfr1.anpf1-dpra2.anpf1-dprr2.anpf1-dprtb3.anpf1-obla1.anpf1-obli1.anpf1-oblr1.anpf1-obls1.anpf1-oblw1.anpf1-obpr1.anpf1-or1.anpf1-pbdr2"
    );
    assert!(descriptor.advertises_protocol_feature(failure_receipt));
    assert!(descriptor.advertises_protocol_feature(purpose_receipt));
    assert!(descriptor.advertises_protocol_feature(direct_relay_auth));
    assert!(descriptor.advertises_protocol_feature(direct_relay_receipt));
    assert!(descriptor.advertises_protocol_feature(direct_relay_target_binding));
    assert!(descriptor.advertises_protocol_feature(onion_reply));
    assert!(descriptor.advertises_protocol_feature(onion_blind_admission));
    assert!(descriptor.advertises_protocol_feature(onion_put_receipt));
    assert!(descriptor.advertises_protocol_feature(onion_lease_retire));
    assert!(descriptor.advertises_protocol_feature(onion_lease_renewal));
    assert!(descriptor.advertises_protocol_feature(onion_lease_status));
    assert!(descriptor.advertises_protocol_feature(onion_lease_inventory));

    let signed = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
    let encoded = encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
        descriptor: signed.clone(),
    })
    .unwrap();
    let decoded = decode_discovery_message(&encoded).unwrap();
    let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = decoded else {
        panic!("unexpected discovery message variant");
    };
    assert!(descriptor.verify_at(1_700_000_100).is_ok());
    assert!(descriptor
        .descriptor
        .advertises_protocol_feature(failure_receipt));
    assert!(descriptor
        .descriptor
        .advertises_protocol_feature(purpose_receipt));
    assert!(descriptor
        .descriptor
        .advertises_protocol_feature(direct_relay_auth));
    assert!(descriptor
        .descriptor
        .advertises_protocol_feature(direct_relay_receipt));
    assert!(descriptor
        .descriptor
        .advertises_protocol_feature(direct_relay_target_binding));
    assert!(descriptor
        .descriptor
        .advertises_protocol_feature(onion_reply));

    let mut stripped = signed;
    stripped.descriptor.software_version = "test".to_string();
    assert!(stripped.verify_at(1_700_000_100).is_err());
}

#[test]
fn anonymous_mailbox_feature_is_signed_and_append_only() {
    // [ANONYMOUS-MAILBOX-V1 2026-09-02 by Codex] The feature is one
    // fleet-wide capability claim, never a mailbox or receiver locator.
    let feature = NodeProtocolFeature::AnonymousMailboxV1;
    assert_eq!(feature.semver_build_token(), "anpf1-amb1");
    assert_eq!(NodeProtocolFeature::ALL[16], feature);

    let identity = IdentityKeyPair::from_bytes(&[0x9a; 32]).expect("identity");
    let descriptor = descriptor_for(&identity).with_protocol_features([feature]);
    assert!(descriptor.advertises_protocol_feature(feature));
    let signed = SignedNodeDescriptor::sign(descriptor, &identity).expect("sign descriptor");
    assert!(signed.verify_at(1_700_000_100).is_ok());
}

#[test]
fn identity_bound_tls_feature_is_signed_and_append_only() {
    // [NODE-TLS-BINDING 2026-10-10 by Claude] Appended after every
    // existing feature, so earlier tokens and their order are unchanged.
    let feature = NodeProtocolFeature::IdentityBoundTlsV1;
    assert_eq!(feature.semver_build_token(), "anpf1-ibt1");
    assert_eq!(NodeProtocolFeature::ALL.len(), 18);
    assert_eq!(NodeProtocolFeature::ALL[17], feature);

    let identity = IdentityKeyPair::from_bytes(&[0x9b; 32]).expect("identity");
    let descriptor = descriptor_for(&identity).with_protocol_features([feature]);
    assert!(descriptor.advertises_protocol_feature(feature));
    let signed = SignedNodeDescriptor::sign(descriptor, &identity).expect("sign descriptor");
    assert!(signed.verify_at(1_700_000_100).is_ok());
    let mut stripped = signed;
    stripped.descriptor.software_version = "0.1.0".to_string();
    assert!(stripped.verify_at(1_700_000_100).is_err());
}

#[test]
fn protocol_features_preserve_existing_semver_build_metadata() {
    let identity = IdentityKeyPair::generate();
    let descriptor = NodeDescriptor::new(
        identity.public_key_bytes(),
        1,
        1_700_000_000,
        1_700_003_600,
        "1.2.3-rc.1+git.abc",
    )
    .with_protocol_features([NodeProtocolFeature::BlindRelayFailureReceiptV1]);

    assert_eq!(
        descriptor.software_version,
        "1.2.3-rc.1+git.abc.anpf1-brfr1"
    );
}
