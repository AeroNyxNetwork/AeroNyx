// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/descriptor/tests.rs
// ============================================
//! # Tests: signed node descriptors
//!
//! Unit tests for signed node descriptors, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use bincode::Options;

use crate::protocol::discovery::test_support::descriptor_for;
use crate::protocol::discovery::{
    decode_discovery_message, encode_discovery_message, NodeDiscoveryMessage,
};

#[test]
fn test_signed_descriptor_roundtrip_verifies() {
    let kp = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();

    assert!(signed.verify_at(1_700_000_100).is_ok());
    assert_eq!(signed.node_id(), kp.public_key_bytes());
    assert_eq!(signed.sequence(), 7);
}

#[test]
#[allow(clippy::unwrap_used)]
fn signed_descriptor_canonical_codec_rejects_trailing_and_oversize() {
    let kp = IdentityKeyPair::from_bytes(&[0x5a; 32]).unwrap();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
    let encoded = signed.encode_canonical().unwrap();
    assert_eq!(
        SignedNodeDescriptor::decode_canonical(&encoded).unwrap(),
        signed
    );

    let mut trailing = encoded;
    trailing.push(0);
    assert!(SignedNodeDescriptor::decode_canonical(&trailing).is_err());
    assert!(
        SignedNodeDescriptor::decode_canonical(&vec![0; MAX_SIGNED_NODE_DESCRIPTOR_BYTES + 1])
            .is_err()
    );
}

#[test]
fn staged_capabilities_are_appended_and_signature_bound() {
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Existing enum positions are
    // part of the bincode wire contract. Appending the staged capability
    // keeps OnionMiddle at 4 and assigns the new carrier role to 5.
    assert_eq!(
        bincode::serialize(&NodeCapability::OnionMiddle).unwrap(),
        4u32.to_le_bytes()
    );
    assert_eq!(
        bincode::serialize(&NodeCapability::DirectoryMirrorCarrier).unwrap(),
        5u32.to_le_bytes()
    );
    assert_eq!(
        bincode::serialize(&NodeCapability::BlindVaultReplica).unwrap(),
        6u32.to_le_bytes()
    );

    let identity = IdentityKeyPair::generate();
    let mut descriptor = descriptor_for(&identity);
    descriptor
        .capabilities
        .push(NodeCapability::DirectoryMirrorCarrier);
    descriptor
        .capabilities
        .push(NodeCapability::BlindVaultReplica);
    let signed = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
    assert!(signed.verify_at(1_700_000_100).is_ok());

    let encoded =
        encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce { descriptor: signed })
            .unwrap();
    let decoded = decode_discovery_message(&encoded).unwrap();
    let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = decoded else {
        panic!("unexpected discovery message variant");
    };
    assert!(descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::DirectoryMirrorCarrier));
    assert!(descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::BlindVaultReplica));
    assert!(descriptor.verify_at(1_700_000_100).is_ok());
}

#[test]
fn test_descriptor_publishes_x25519_kem_key() {
    let kp = IdentityKeyPair::generate();
    let kem = kp.x25519_public_key_bytes();
    let descriptor = descriptor_for(&kp).with_x25519_kem(kem);
    assert_eq!(descriptor.schema_version, NODE_DESCRIPTOR_SCHEMA_VERSION);
    assert_eq!(descriptor.kem_alg, 1);
    assert_eq!(descriptor.x25519_kem_public(), Some(kem));

    // KEM key is covered by the signature and survives encode/decode.
    let signed = SignedNodeDescriptor::sign(descriptor, &kp).unwrap();
    assert!(signed.verify_at(1_700_000_100).is_ok());
    let bytes = encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
        descriptor: signed.clone(),
    })
    .unwrap();
    let decoded = decode_discovery_message(&bytes).unwrap();
    if let NodeDiscoveryMessage::DescriptorAnnounce { descriptor } = decoded {
        assert_eq!(descriptor.descriptor.x25519_kem_public(), Some(kem));
        assert!(descriptor.verify_at(1_700_000_100).is_ok());
    } else {
        panic!("unexpected discovery message variant");
    }
}

#[test]
fn test_descriptor_without_kem_reports_none() {
    let kp = IdentityKeyPair::generate();
    let descriptor = descriptor_for(&kp);
    assert_eq!(descriptor.kem_alg, 0);
    assert_eq!(descriptor.x25519_kem_public(), None);
}

#[test]
fn test_schema_v1_descriptor_without_kem_fields_still_verifies() {
    let kp = IdentityKeyPair::generate();
    let mut descriptor = descriptor_for(&kp);
    descriptor.schema_version = 1;
    descriptor.kem_alg = 0;
    descriptor.kem_public = [0u8; 32];
    let signature = kp.sign(&legacy_descriptor_v1_signing_bytes(&descriptor).unwrap());
    let signed = SignedNodeDescriptor {
        descriptor,
        signature,
    };

    let mut json = serde_json::to_value(&signed).unwrap();
    let descriptor_json = json
        .get_mut("descriptor")
        .and_then(serde_json::Value::as_object_mut)
        .expect("descriptor json object");
    descriptor_json.remove("kem_alg");
    descriptor_json.remove("kem_public");

    let decoded: SignedNodeDescriptor = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.descriptor.schema_version, 1);
    assert_eq!(decoded.descriptor.kem_alg, 0);
    assert_eq!(decoded.descriptor.kem_public, [0u8; 32]);
    assert_eq!(decoded.descriptor.x25519_kem_public(), None);
    assert!(decoded.verify_at(1_700_000_100).is_ok());
}

#[test]
fn test_tampered_descriptor_rejected() {
    let kp = IdentityKeyPair::generate();
    let mut signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
    signed.descriptor.sequence += 1;

    assert!(signed.verify_at(1_700_000_100).is_err());
}

#[test]
fn test_expired_descriptor_rejected() {
    let kp = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();

    assert!(signed.verify_at(1_700_004_000).is_err());
}

#[test]
fn test_signature_only_verification_keeps_expired_records_non_live() {
    let kp = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();

    assert!(signed.verify_at(1_700_004_000).is_err());
    assert!(signed.verify_signature().is_ok());

    let mut tampered = signed.clone();
    tampered.signature[0] ^= 0x01;
    assert!(tampered.verify_signature().is_err());
}

#[test]
fn test_descriptor_bincode_roundtrip() {
    let kp = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
    let bytes = bincode::options()
        .with_fixint_encoding()
        .serialize(&signed)
        .unwrap();
    let restored: SignedNodeDescriptor = bincode::options()
        .with_fixint_encoding()
        .deserialize(&bytes)
        .unwrap();

    assert_eq!(restored, signed);
    assert!(restored.verify_at(1_700_000_100).is_ok());
}
