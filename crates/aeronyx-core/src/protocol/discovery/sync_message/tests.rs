// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/sync_message/tests.rs
// ============================================
//! # Tests: Directory Sync frames and their signing digests
//!
//! Unit tests for Directory Sync frames and their signing digests, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};

use crate::protocol::discovery::test_support::descriptor_for;
use crate::protocol::discovery::{
    directory_block_range_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_observation_certificate_request_signing_bytes,
    directory_observation_certificate_response_signing_bytes,
    directory_observation_witness_carrier_request_signing_bytes,
    directory_observation_witness_carrier_response_signing_bytes,
    directory_observation_witness_request_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_policy_anchor_request_signing_bytes, directory_policy_anchor_response_signing_bytes,
    directory_replica_block_range_request_signing_bytes,
    directory_replica_block_range_response_signing_bytes,
    directory_replica_descriptor_inclusion_proof_request_signing_bytes,
    directory_replica_descriptor_inclusion_proof_response_signing_bytes,
    directory_replica_descriptor_objects_request_signing_bytes,
    directory_replica_descriptor_objects_response_signing_bytes,
    directory_tip_request_signing_bytes, DirectoryDescriptorCommitmentV1,
    DirectoryObservationTipV1, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
};

#[test]
fn directory_sync_tip_frame_is_canonical_and_domain_bound() {
    let requester = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
    let request_id = [0x92; 16];
    let timestamp = 1_700_000_123;
    let signing_bytes = directory_tip_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester.public_key_bytes(),
        timestamp,
    );
    let message = DirectorySyncMessage::TipRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp: timestamp,
        signature: requester.sign(&signing_bytes),
    };

    let encoded = encode_directory_sync_message(&message).unwrap();
    assert_eq!(encoded.first().copied(), Some(DIRECTORY_SYNC_MAGIC));
    assert_eq!(decode_directory_sync_message(&encoded).unwrap(), message);
    assert_ne!(
        signing_bytes,
        directory_tip_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester.public_key_bytes(),
            timestamp + 1,
        )
    );

    let mut trailing = encoded;
    trailing.push(0);
    assert!(decode_directory_sync_message(&trailing).is_err());
}

#[test]
fn directory_sync_range_and_object_digests_bind_order_and_tip() {
    let producer = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
    let first_peer = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
    let second_peer = IdentityKeyPair::from_bytes(&[0x95; 32]).unwrap();
    let first = SignedNodeDescriptor::sign(descriptor_for(&first_peer), &first_peer).unwrap();
    let second = SignedNodeDescriptor::sign(descriptor_for(&second_peer), &second_peer).unwrap();
    let first_commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&first).unwrap();
    let second_commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&second).unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_200,
        [0u8; 32],
        vec![first_commitment, second_commitment],
        &producer,
    )
    .unwrap();
    let request_id = [0x96; 16];
    let forward = directory_block_range_response_signing_bytes(
        &request_id,
        &producer.public_key_bytes(),
        1_700_000_201,
        std::slice::from_ref(&block),
        false,
        1,
        &block.hash(),
    );
    let different_tip = directory_block_range_response_signing_bytes(
        &request_id,
        &producer.public_key_bytes(),
        1_700_000_201,
        std::slice::from_ref(&block),
        false,
        2,
        &block.hash(),
    );
    assert_ne!(forward, different_tip);

    let hashes = [
        first_commitment.descriptor_hash,
        second_commitment.descriptor_hash,
    ];
    let reversed = [hashes[1], hashes[0]];
    assert_ne!(
        directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &hashes,
            &request_id,
            &producer.public_key_bytes(),
            1_700_000_201,
        ),
        directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &reversed,
            &request_id,
            &producer.public_key_bytes(),
            1_700_000_201,
        )
    );
}

#[test]
fn test_directory_observation_witness_frames_are_canonical_and_bound() {
    let observer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x70; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        3,
        1_700_000_300,
        [0x75; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 8,
                tip_hash: [0x76; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 9,
                tip_hash: [0x77; 32],
            },
        ],
        [0x78; 32],
        &observer,
    )
    .unwrap();
    assert!(checkpoint
        .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_300)
        .is_ok());

    let request_id = [0x79; 16];
    let checkpoint_hash = checkpoint.hash();
    let request_digest = directory_observation_witness_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        1_700_000_301,
        &checkpoint_hash,
    );
    let request = DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: observer.public_key_bytes(),
        request_timestamp: 1_700_000_301,
        checkpoint: checkpoint.clone(),
        signature: observer.sign(&request_digest),
    };
    let request_frame = encode_directory_sync_message(&request).unwrap();
    let decoded = decode_directory_sync_message(&request_frame).unwrap();
    assert_eq!(decoded, request);

    let response_digest = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        checkpoint.sequence,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        1_700_000_302,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        observer: observer.public_key_bytes(),
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
        responder: witness.public_key_bytes(),
        response_timestamp: 1_700_000_302,
        outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        signature: witness.sign(&response_digest),
    };
    let response_frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        decode_directory_sync_message(&response_frame).unwrap(),
        response
    );
    let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        responder,
        signature,
        ..
    } = response
    else {
        unreachable!();
    };
    IdentityPublicKey::from_bytes(&responder)
        .unwrap()
        .verify(&response_digest, &signature)
        .unwrap();

    let carrier_request_id = [0x6f; 16];
    let witness_request_sha256: [u8; 32] = Sha256::digest(&request_frame).into();
    let carrier_request_digest = directory_observation_witness_carrier_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &carrier_request_id,
        &observer.public_key_bytes(),
        1_700_000_303,
        &witness.public_key_bytes(),
        &witness_request_sha256,
        u64::try_from(request_frame.len()).unwrap(),
    );
    let carrier_request = DirectorySyncMessage::ObservationCheckpointWitnessCarrierRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id: carrier_request_id,
        requester: observer.public_key_bytes(),
        request_timestamp: 1_700_000_303,
        witness: witness.public_key_bytes(),
        witness_request_sha256,
        witness_request_frame: request_frame.clone(),
        signature: observer.sign(&carrier_request_digest),
    };
    let carrier_request_frame = encode_directory_sync_message(&carrier_request).unwrap();
    assert_eq!(
        decode_directory_sync_message(&carrier_request_frame).unwrap(),
        carrier_request
    );

    let witness_response_sha256: [u8; 32] = Sha256::digest(&response_frame).into();
    let carrier_response_digest = directory_observation_witness_carrier_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &carrier_request_id,
        &observer.public_key_bytes(),
        &witness.public_key_bytes(),
        &carrier.public_key_bytes(),
        1_700_000_304,
        &witness_request_sha256,
        &witness_response_sha256,
        u64::try_from(response_frame.len()).unwrap(),
    );
    let carrier_response = DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id: carrier_request_id,
        requester: observer.public_key_bytes(),
        witness: witness.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: 1_700_000_304,
        witness_request_sha256,
        witness_response_sha256,
        witness_response_frame: response_frame,
        signature: carrier.sign(&carrier_response_digest),
    };
    let carrier_response_frame = encode_directory_sync_message(&carrier_response).unwrap();
    assert_eq!(
        decode_directory_sync_message(&carrier_response_frame).unwrap(),
        carrier_response
    );
    assert_ne!(
        carrier_request_digest,
        directory_observation_witness_carrier_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &carrier_request_id,
            &observer.public_key_bytes(),
            1_700_000_303,
            &witness.public_key_bytes(),
            &witness_request_sha256,
            u64::try_from(request_frame.len())
                .unwrap()
                .saturating_add(1),
        )
    );

    let altered = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        checkpoint.sequence,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        1_700_000_302,
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    );
    assert_ne!(response_digest, altered);

    let policy_request_id = [0x7a; 16];
    let policy_digest = [0x7b; 32];
    let policy_request_digest = directory_policy_anchor_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &policy_request_id,
        &observer.public_key_bytes(),
        1_700_000_303,
        1,
        &[0u8; 32],
        &policy_digest,
    );
    let policy_request = DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id: policy_request_id,
        requester: observer.public_key_bytes(),
        request_timestamp: 1_700_000_303,
        policy_epoch: 1,
        previous_policy_digest: [0u8; 32],
        policy_digest,
        signature: observer.sign(&policy_request_digest),
    };
    let encoded = encode_directory_sync_message(&policy_request).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        policy_request
    );

    let policy_response_digest = directory_policy_anchor_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &policy_request_id,
        &observer.public_key_bytes(),
        1,
        &policy_digest,
        &witness.public_key_bytes(),
        1_700_000_304,
        DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
    );
    let policy_response = DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id: policy_request_id,
        observer: observer.public_key_bytes(),
        policy_epoch: 1,
        policy_digest,
        responder: witness.public_key_bytes(),
        response_timestamp: 1_700_000_304,
        outcome: DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
        signature: witness.sign(&policy_response_digest),
    };
    let encoded = encode_directory_sync_message(&policy_response).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        policy_response
    );
    assert_ne!(
        policy_response_digest,
        directory_policy_anchor_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &policy_request_id,
            &observer.public_key_bytes(),
            1,
            &policy_digest,
            &witness.public_key_bytes(),
            1_700_000_304,
            DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
        )
    );
}

#[test]
fn test_directory_observation_certificate_exchange_frames_are_canonical_and_bound() {
    let requester = IdentityKeyPair::from_bytes(&[0x7c; 32]).unwrap();
    let responder = IdentityKeyPair::from_bytes(&[0x7d; 32]).unwrap();
    let request_id = [0x7e; 16];
    let request_timestamp = 1_700_000_305;
    let request_digest = directory_observation_certificate_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester.public_key_bytes(),
        request_timestamp,
    );
    let request = DirectorySyncMessage::ObservationCertificateRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp,
        signature: requester.sign(&request_digest),
    };
    let encoded = encode_directory_sync_message(&request).unwrap();
    assert_eq!(decode_directory_sync_message(&encoded).unwrap(), request);

    let certificate_frame = vec![0xa5; 96];
    let certificate_sha256: [u8; 32] = Sha256::digest(&certificate_frame).into();
    let response_timestamp = 1_700_000_306;
    let frame_bytes = u64::try_from(certificate_frame.len()).unwrap();
    let response_digest = directory_observation_certificate_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester.public_key_bytes(),
        &responder.public_key_bytes(),
        response_timestamp,
        &certificate_sha256,
        frame_bytes,
    );
    let response_signature = responder.sign(&response_digest);
    let response = DirectorySyncMessage::ObservationCertificateResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: requester.public_key_bytes(),
        responder: responder.public_key_bytes(),
        response_timestamp,
        certificate_sha256,
        certificate_frame: certificate_frame.clone(),
        signature: response_signature,
    };
    let encoded = encode_directory_sync_message(&response).unwrap();
    assert_eq!(decode_directory_sync_message(&encoded).unwrap(), response);
    IdentityPublicKey::from_bytes(&responder.public_key_bytes())
        .unwrap()
        .verify(&response_digest, &response_signature)
        .unwrap();

    assert_ne!(
        response_digest,
        directory_observation_certificate_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester.public_key_bytes(),
            &responder.public_key_bytes(),
            response_timestamp,
            &certificate_sha256,
            frame_bytes + 1,
        )
    );
}

#[test]
fn test_directory_replica_carrier_frames_are_canonical_and_fully_bound() {
    let requester = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
    let descriptor = SignedNodeDescriptor::sign(descriptor_for(&subject), &subject).unwrap();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_400,
        [0u8; 32],
        vec![commitment],
        &producer,
    )
    .unwrap();
    let request_id = [0x85; 16];

    let range_request_digest = directory_replica_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &producer.public_key_bytes(),
        1,
        1,
        &request_id,
        &requester.public_key_bytes(),
        1_700_000_401,
    );
    let range_request = DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer: producer.public_key_bytes(),
        from_height: 1,
        limit: 1,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp: 1_700_000_401,
        signature: requester.sign(&range_request_digest),
    };
    let encoded = encode_directory_sync_message(&range_request).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        range_request
    );

    let block_hash = block.hash();
    let blocks = vec![block];
    let range_response_digest = directory_replica_block_range_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        1_700_000_402,
        &blocks,
        false,
        1,
        &block_hash,
    );
    let range_response = DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: 1_700_000_402,
        blocks: blocks.clone(),
        has_more: false,
        tip_height: 1,
        tip_hash: block_hash,
        signature: carrier.sign(&range_response_digest),
    };
    let encoded = encode_directory_sync_message(&range_response).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        range_response
    );
    let altered_producer_digest = directory_replica_block_range_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &[0x86; 32],
        &carrier.public_key_bytes(),
        1_700_000_402,
        &blocks,
        false,
        1,
        &block_hash,
    );
    assert_ne!(range_response_digest, altered_producer_digest);

    let hashes = vec![commitment.descriptor_hash];
    let object_request_digest = directory_replica_descriptor_objects_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &producer.public_key_bytes(),
        &hashes,
        &request_id,
        &requester.public_key_bytes(),
        1_700_000_403,
    );
    let object_request = DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer: producer.public_key_bytes(),
        descriptor_hashes: hashes.clone(),
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp: 1_700_000_403,
        signature: requester.sign(&object_request_digest),
    };
    let encoded = encode_directory_sync_message(&object_request).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        object_request
    );

    let object_response_digest = directory_replica_descriptor_objects_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        1_700_000_404,
        &hashes,
    );
    let object_response = DirectorySyncMessage::ReplicaDescriptorObjectsResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: 1_700_000_404,
        descriptor_hashes: hashes,
        objects: vec![descriptor.clone()],
        signature: carrier.sign(&object_response_digest),
    };
    let encoded = encode_directory_sync_message(&object_response).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        object_response
    );

    let proof =
        DirectoryDescriptorInclusionProofV1::from_block_at(&blocks[0], &descriptor, 1_700_000_405)
            .unwrap();
    let proof_request_digest = directory_replica_descriptor_inclusion_proof_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &producer.public_key_bytes(),
        &block_hash,
        &commitment.descriptor_hash,
        &request_id,
        &requester.public_key_bytes(),
        1_700_000_405,
    );
    let proof_request = DirectorySyncMessage::ReplicaDescriptorInclusionProofRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer: producer.public_key_bytes(),
        block_hash,
        descriptor_hash: commitment.descriptor_hash,
        request_id,
        requester: requester.public_key_bytes(),
        request_timestamp: 1_700_000_405,
        signature: requester.sign(&proof_request_digest),
    };
    let encoded = encode_directory_sync_message(&proof_request).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        proof_request
    );

    let proof_response_digest = directory_replica_descriptor_inclusion_proof_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        1_700_000_406,
        &block_hash,
        &commitment.descriptor_hash,
        &proof,
    );
    let proof_response = DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: 1_700_000_406,
        block_hash,
        descriptor_hash: commitment.descriptor_hash,
        proof: proof.clone(),
        signature: carrier.sign(&proof_response_digest),
    };
    let encoded = encode_directory_sync_message(&proof_response).unwrap();
    assert_eq!(
        decode_directory_sync_message(&encoded).unwrap(),
        proof_response
    );
    assert_ne!(
        proof_response_digest,
        directory_replica_descriptor_inclusion_proof_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &producer.public_key_bytes(),
            &[0x87; 32],
            1_700_000_406,
            &block_hash,
            &commitment.descriptor_hash,
            &proof,
        )
    );
}
