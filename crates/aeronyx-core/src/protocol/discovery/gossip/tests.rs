// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/gossip/tests.rs
// ============================================
//! # Tests: bootstrap snapshots and discovery gossip messages
//!
//! Unit tests for bootstrap snapshots and discovery gossip messages, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use sha2::{Digest, Sha256};

use crate::crypto::IdentityKeyPair;
use crate::protocol::discovery_endpoint_attestation::{
    discovery_endpoint_evidence_commitment_v1, DiscoveryEndpointAttestationPurposeV1,
};
use crate::protocol::discovery_endpoint_proof::{
    canonical_public_endpoint_commitment, DiscoveryEndpointChallengeV1, DiscoveryEndpointProofV1,
};

use crate::protocol::discovery::test_support::descriptor_for;
use crate::protocol::discovery::{DirectoryCommitmentBlockV1, DirectoryDescriptorCommitmentV1};

const ENDPOINT_ATTESTATION_TEST_NOW: u64 = 1_780_000_000;

fn endpoint_attestation_frame() -> Vec<u8> {
    let observer = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
    let context = [0x33; 32];
    let endpoint = canonical_public_endpoint_commitment("8.8.8.8:51820").unwrap();
    let mut descriptor = descriptor_for(&subject);
    descriptor.public_endpoint = Some("8.8.8.8:51820".to_string());
    descriptor.issued_at = ENDPOINT_ATTESTATION_TEST_NOW - 60;
    descriptor.expires_at = ENDPOINT_ATTESTATION_TEST_NOW + 3_600;
    let descriptor = SignedNodeDescriptor::sign(descriptor, &subject).unwrap();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
    let challenge = DiscoveryEndpointChallengeV1::issue(
        subject.public_key_bytes(),
        commitment.descriptor_hash,
        endpoint,
        [0x44; 32],
        context,
        ENDPOINT_ATTESTATION_TEST_NOW,
        ENDPOINT_ATTESTATION_TEST_NOW + 120,
        &observer,
    )
    .unwrap();
    let proof = DiscoveryEndpointProofV1::respond(
        &challenge,
        &context,
        ENDPOINT_ATTESTATION_TEST_NOW + 1,
        &subject,
    )
    .unwrap();
    let evidence = discovery_endpoint_evidence_commitment_v1(&challenge, &proof);
    let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
        &descriptor,
        &challenge,
        &proof,
        context,
        DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
        ENDPOINT_ATTESTATION_TEST_NOW + 1,
        ENDPOINT_ATTESTATION_TEST_NOW + 3_601,
        &observer,
    )
    .unwrap();
    attestation
        .verify_at(
            ENDPOINT_ATTESTATION_TEST_NOW + 2,
            &observer.public_key_bytes(),
            &commitment,
            &endpoint,
            &evidence,
            &context,
            DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
        )
        .unwrap();
    attestation.encode()
}

#[test]
fn test_bootstrap_snapshot_json_roundtrip() {
    let kp = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
    let snapshot = NodeBootstrapSnapshot::new(1_700_000_010, vec![signed]);

    let json = snapshot.to_json_pretty().unwrap();
    let restored = NodeBootstrapSnapshot::from_json_bytes(&json).unwrap();

    assert_eq!(restored, snapshot);
    assert_eq!(restored.verified_count_at(1_700_000_100), 1);
}

#[test]
fn test_bootstrap_snapshot_rejects_unsupported_schema() {
    let snapshot = NodeBootstrapSnapshot {
        schema_version: NODE_BOOTSTRAP_SNAPSHOT_SCHEMA_VERSION + 1,
        generated_at: 1_700_000_010,
        peers: Vec::new(),
    };
    let json = serde_json::to_vec(&snapshot).unwrap();

    assert!(NodeBootstrapSnapshot::from_json_bytes(&json).is_err());
}

#[test]
fn test_bootstrap_snapshot_rejects_oversized_json() {
    let too_large = vec![b' '; MAX_BOOTSTRAP_SNAPSHOT_BYTES + 1];

    assert!(matches!(
        NodeBootstrapSnapshot::from_json_bytes(&too_large),
        Err(CoreError::MessageTooLarge { .. })
    ));
}

#[test]
fn test_discovery_message_snapshot_request_roundtrip() {
    let message = NodeDiscoveryMessage::SnapshotRequest {
        requested_at: 0x0102_0304_0506_0708,
        limit: Some(0x090a),
    };

    let bytes = encode_discovery_message(&message).unwrap();
    let decoded = decode_discovery_message(&bytes).unwrap();

    assert_eq!(decoded, message);
    assert_eq!(
        bytes,
        [
            0x00, 0x00, 0x00, 0x00, // enum variant
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // timestamp
            0x01, // Some
            0x0a, 0x09, // limit
        ],
        "the bounded codec must preserve the established discovery wire bytes"
    );

    let mut trailing = bytes;
    trailing.push(0);
    assert!(
        decode_discovery_message(&trailing).is_err(),
        "canonical discovery messages must reject trailing bytes"
    );

    let padded = vec![0; MAX_DISCOVERY_MESSAGE_BYTES as usize + 1];
    assert!(
        decode_discovery_message(&padded).is_err(),
        "the complete discovery input must obey the protocol ceiling"
    );
}

#[test]
fn test_discovery_message_snapshot_response_roundtrip() {
    let kp = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
    let snapshot = NodeBootstrapSnapshot::new(1_700_000_010, vec![signed]);
    let message = NodeDiscoveryMessage::SnapshotResponse { snapshot };

    let bytes = encode_discovery_message(&message).unwrap();
    let decoded = decode_discovery_message(&bytes).unwrap();

    assert_eq!(decoded, message);
}

#[test]
fn test_discovery_message_descriptor_announce_roundtrip() {
    let kp = IdentityKeyPair::generate();
    let descriptor = SignedNodeDescriptor::sign(descriptor_for(&kp), &kp).unwrap();
    let message = NodeDiscoveryMessage::DescriptorAnnounce { descriptor };

    let bytes = encode_discovery_message(&message).unwrap();
    let decoded = decode_discovery_message(&bytes).unwrap();

    assert_eq!(decoded, message);
}

#[test]
fn directory_descriptor_announce_is_append_only_and_roundtrips() {
    // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] The enum index is
    // part of the mixed-version bincode contract. Existing variants remain
    // 0/1/2 and the proof-carrying announcement is appended at index 3.
    let producer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
    let descriptor = SignedNodeDescriptor::sign(descriptor_for(&subject), &subject).unwrap();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![commitment],
        &producer,
    )
    .unwrap();
    let block_hash = block.hash();
    let proof =
        DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, 1_700_000_100)
            .unwrap();
    let message = NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
        producer: producer.public_key_bytes(),
        block_hash,
        descriptor_hash: proof.commitment.descriptor_hash,
        proof,
    };

    assert_eq!(
        &encode_discovery_message(&NodeDiscoveryMessage::SnapshotRequest {
            requested_at: 1,
            limit: None,
        })
        .unwrap()[..4],
        &0u32.to_le_bytes()
    );
    assert_eq!(
        &encode_discovery_message(&NodeDiscoveryMessage::SnapshotResponse {
            snapshot: NodeBootstrapSnapshot::new(1_700_000_100, Vec::new()),
        })
        .unwrap()[..4],
        &1u32.to_le_bytes()
    );
    assert_eq!(
        &encode_discovery_message(&NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: descriptor.clone(),
        })
        .unwrap()[..4],
        &2u32.to_le_bytes()
    );
    let encoded = encode_discovery_message(&message).unwrap();
    assert_eq!(&encoded[..4], &3u32.to_le_bytes());
    assert_eq!(decode_discovery_message(&encoded).unwrap(), message);
}

#[test]
fn endpoint_attestation_carrier_is_append_only_canonical_and_frozen() {
    // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex] The carrier is
    // append-only at discriminant 4 and admits exactly one canonical ADAT.
    let attestation_frame = endpoint_attestation_frame();
    assert_eq!(
        attestation_frame.len(),
        DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1
    );
    let message = NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { attestation_frame };
    let encoded = encode_discovery_message(&message).unwrap();
    assert_eq!(&encoded[..4], &4u32.to_le_bytes());
    assert_eq!(encoded.len(), 301);
    assert_eq!(decode_discovery_message(&encoded).unwrap(), message);
    assert_eq!(
        hex::encode(Sha256::digest(&encoded)),
        "7cdcf74ff0bfb4f6e61de021fc2c218b5603838a6d254b247d8f0a31eb9ff78e"
    );

    let mut trailing = encoded;
    trailing.push(0);
    assert!(decode_discovery_message(&trailing).is_err());
}

#[test]
fn endpoint_attestation_carrier_rejects_noncanonical_inner_frames() {
    let canonical = endpoint_attestation_frame();
    for malformed in [canonical[..canonical.len() - 1].to_vec(), {
        let mut bytes = canonical.clone();
        bytes.push(0);
        bytes
    }] {
        assert!(
            encode_discovery_message(&NodeDiscoveryMessage::EndpointEvidenceAttestationV1 {
                attestation_frame: malformed,
            })
            .is_err()
        );
    }

    for offset in [4usize, 5, 224] {
        let mut malformed = canonical.clone();
        malformed[offset] ^= 0x7f;
        assert!(
            encode_discovery_message(&NodeDiscoveryMessage::EndpointEvidenceAttestationV1 {
                attestation_frame: malformed,
            })
            .is_err()
        );
    }
}
