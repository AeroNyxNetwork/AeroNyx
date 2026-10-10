// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/tests.rs
// ============================================
//! # Tests: directory replica store
//!
//! Shared fixtures (descriptors, blocks, signed response frames, certificate
//! and resolution builders) plus the topic test modules under `tests/`.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

mod mirror_schema;
mod observation;
mod policy;
mod quarantine;
mod remaining;
mod retry;
mod store;
mod sync;
mod witness;
use super::*;
use crate::services::DirectoryChainStore;
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::discovery::{
    directory_block_range_response_signing_bytes, encode_directory_sync_message, NodeDescriptor,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use tempfile::TempDir;

const NOW: u64 = 1_700_000_100;

struct DirectoryReplicaAuditObserverGuard;

impl Drop for DirectoryReplicaAuditObserverGuard {
    fn drop(&mut self) {
        set_directory_replica_audit_test_observer(None);
    }
}

fn observe_directory_replica_audit(
    observer: impl Fn(DirectoryReplicaAuditTestEvent) + 'static,
) -> DirectoryReplicaAuditObserverGuard {
    set_directory_replica_audit_test_observer(Some(Box::new(observer)));
    DirectoryReplicaAuditObserverGuard
}

fn descriptor(identity: &IdentityKeyPair, sequence: u64) -> SignedNodeDescriptor {
    SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            identity.public_key_bytes(),
            sequence,
            NOW - 10,
            NOW + 3_600,
            "replica-test",
        ),
        identity,
    )
    .unwrap()
}

fn block(
    producer: &IdentityKeyPair,
    height: u64,
    previous: [u8; 32],
    object: &SignedNodeDescriptor,
) -> DirectoryCommitmentBlockV1 {
    DirectoryCommitmentBlockV1::new_signed(
        height,
        NOW + height,
        previous,
        vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(object).unwrap()],
        producer,
    )
    .unwrap()
}

fn response_frame(
    producer: &IdentityKeyPair,
    blocks: Vec<DirectoryCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
    tip_hash: [u8; 32],
    request_id: [u8; 16],
) -> Vec<u8> {
    let responder = producer.public_key_bytes();
    let signing = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        NOW + 20,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    encode_directory_sync_message(&DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        responder,
        response_timestamp: NOW + 20,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature: producer.sign(&signing),
    })
    .unwrap()
}

fn carrier_response_frame(
    producer: &IdentityKeyPair,
    carrier: &IdentityKeyPair,
    blocks: Vec<DirectoryCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
    tip_hash: [u8; 32],
    request_id: [u8; 16],
) -> Vec<u8> {
    let producer_id = producer.public_key_bytes();
    let carrier_id = carrier.public_key_bytes();
    let signing = directory_replica_block_range_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer_id,
        &carrier_id,
        NOW + 20,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    encode_directory_sync_message(&DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer_id,
        carrier: carrier_id,
        response_timestamp: NOW + 20,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature: carrier.sign(&signing),
    })
    .unwrap()
}

fn accepted_observation_witness_response(
    observer: &IdentityKeyPair,
    witness: &IdentityKeyPair,
    checkpoint: &DirectoryObservationCheckpointV1,
    request_seed: u8,
) -> DirectorySyncMessage {
    let request_id = [request_seed; 16];
    let checkpoint_hash = checkpoint.hash();
    let responder = witness.public_key_bytes();
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        checkpoint.sequence,
        &checkpoint_hash,
        &responder,
        NOW + 22,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        observer: observer.public_key_bytes(),
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
        responder,
        response_timestamp: NOW + 22,
        outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        signature: witness.sign(&signing_bytes),
    }
}

fn portable_observation_certificate_fixture(
    observer: &IdentityKeyPair,
    witnesses: &[&IdentityKeyPair],
    checkpoint_sequence: u64,
    checkpoint_salt: u8,
    verified_at: u64,
) -> (
    Vec<u8>,
    [u8; 32],
    DirectoryObservationCertificateTrustPolicy,
) {
    let producer_a = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x42; 32]).unwrap();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        checkpoint_sequence,
        verified_at - 2,
        [checkpoint_salt; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: checkpoint_sequence + 10,
                tip_hash: [checkpoint_salt.wrapping_add(1); 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: checkpoint_sequence + 11,
                tip_hash: [checkpoint_salt.wrapping_add(2); 32],
            },
        ],
        [checkpoint_salt.wrapping_add(3); 32],
        observer,
    )
    .unwrap();
    let receipts = witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| {
            let request_seed = u8::try_from(index).unwrap().wrapping_add(0x50);
            let request_id = [request_seed; 16];
            let checkpoint_hash = checkpoint.hash();
            let responder = witness.public_key_bytes();
            let signing_bytes = directory_observation_witness_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &observer.public_key_bytes(),
                checkpoint.sequence,
                &checkpoint_hash,
                &responder,
                verified_at - 1,
                DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            );
            DirectoryObservationWitnessReceiptV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                observer: observer.public_key_bytes(),
                checkpoint_sequence: checkpoint.sequence,
                checkpoint_hash,
                responder,
                response_timestamp: verified_at - 1,
                outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                signature: witness.sign(&signing_bytes),
            }
        })
        .collect::<Vec<_>>();
    let certificate = DirectoryObservationCertificateV1::new_verified(
        checkpoint,
        u16::try_from(witnesses.len()).unwrap(),
        receipts,
        verified_at,
    )
    .unwrap();
    let frame = encode_directory_observation_certificate(&certificate).unwrap();
    let frame_sha256 = Sha256::digest(&frame).into();
    let trust_policy = DirectoryObservationCertificateTrustPolicy::new(
        observer.public_key_bytes(),
        witnesses
            .iter()
            .map(|witness| witness.public_key_bytes())
            .collect(),
        u16::try_from(witnesses.len()).unwrap(),
    )
    .unwrap();
    (frame, frame_sha256, trust_policy)
}

fn import_replica_block(
    store: &DirectoryReplicaStore,
    producer: &IdentityKeyPair,
    object: &SignedNodeDescriptor,
    replica_block: &DirectoryCommitmentBlockV1,
    request_id: [u8; 16],
) {
    let frame = response_frame(
        producer,
        vec![replica_block.clone()],
        false,
        replica_block.header.height,
        replica_block.hash(),
        request_id,
    );
    store
        .import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(replica_block),
            std::slice::from_ref(object),
            replica_block.header.height,
            replica_block.hash(),
            &frame,
            NOW + 20,
        )
        .unwrap();
}

fn resolution_command(
    resolver: &IdentityKeyPair,
    incident_digest: [u8; 32],
    tip: &DirectoryReplicaTip,
    command_id: [u8; 16],
    resolved_at: u64,
) -> DirectoryReplicaResolutionCommand {
    DirectoryReplicaResolutionCommand::sign(
        resolver,
        command_id,
        incident_digest,
        tip.producer,
        tip.tip_height,
        tip.tip_hash,
        tip.quarantine_kind.clone().unwrap(),
        tip.last_resolution_digest,
        resolved_at,
    )
    .unwrap()
}

fn frame_tip_hash(frame: &[u8]) -> [u8; 32] {
    let DirectorySyncMessage::BlockRangeResponseV1 { tip_hash, .. } =
        decode_directory_sync_message(frame).unwrap()
    else {
        unreachable!()
    };
    tip_hash
}
