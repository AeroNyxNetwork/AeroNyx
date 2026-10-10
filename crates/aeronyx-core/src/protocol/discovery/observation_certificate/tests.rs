// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/observation_certificate/tests.rs
// ============================================
//! # Tests: portable Directory observation certificates
//!
//! Unit tests for portable Directory observation certificates, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use crate::crypto::IdentityKeyPair;

use crate::protocol::discovery::{DirectoryObservationTipV1, AERONYX_DIRECTORY_MAINNET_CHAIN_ID};

fn accepted_observation_receipt(
    checkpoint: &DirectoryObservationCheckpointV1,
    witness: &IdentityKeyPair,
    request_id: [u8; 16],
    response_timestamp: u64,
) -> DirectoryObservationWitnessReceiptV1 {
    let checkpoint_hash = checkpoint.hash();
    let digest = directory_observation_witness_response_signing_bytes(
        &checkpoint.chain_id,
        &request_id,
        &checkpoint.observer,
        checkpoint.sequence,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        response_timestamp,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    DirectoryObservationWitnessReceiptV1 {
        chain_id: checkpoint.chain_id,
        request_id,
        observer: checkpoint.observer,
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
        responder: witness.public_key_bytes(),
        response_timestamp,
        outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        signature: witness.sign(&digest),
    }
}

#[test]
fn portable_observation_certificate_is_canonical_bounded_and_offline_verifiable() {
    // [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] This fixture
    // proves exact observer/witness signatures without introducing an
    // aggregator signature, vote, quorum, consensus, or finality claim.
    let observer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
    let witness_a = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
    let witness_b = IdentityKeyPair::from_bytes(&[0x85; 32]).unwrap();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        4,
        1_700_000_400,
        [0x86; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 18,
                tip_hash: [0x87; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 19,
                tip_hash: [0x88; 32],
            },
        ],
        [0x89; 32],
        &observer,
    )
    .unwrap();
    let receipt_a =
        accepted_observation_receipt(&checkpoint, &witness_a, [0x8a; 16], 1_700_000_401);
    let receipt_b =
        accepted_observation_receipt(&checkpoint, &witness_b, [0x8b; 16], 1_700_000_402);
    let certificate = DirectoryObservationCertificateV1::new_verified(
        checkpoint,
        2,
        vec![receipt_b, receipt_a],
        1_700_000_402,
    )
    .unwrap();

    assert!(certificate.receipts[0].responder < certificate.receipts[1].responder);
    assert!(certificate
        .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_402)
        .is_ok());
    let encoded = encode_directory_observation_certificate(&certificate).unwrap();
    assert_eq!(
        encoded.first().copied(),
        Some(DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC)
    );
    assert!(encoded.len() < MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES as usize);
    let decoded = decode_directory_observation_certificate(&encoded).unwrap();
    assert_eq!(decoded, certificate);
    assert_eq!(decoded.hash(), certificate.hash());
    assert_eq!(
        DirectoryObservationWitnessReceiptV1::from_sync_message(
            &decoded.receipts[0].to_sync_message()
        )
        .unwrap(),
        decoded.receipts[0]
    );

    let mut trailing = encoded;
    trailing.push(0);
    assert!(decode_directory_observation_certificate(&trailing).is_err());

    let mut oversized = vec![0u8; MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES as usize + 2];
    oversized[0] = DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC;
    assert!(decode_directory_observation_certificate(&oversized).is_err());
}

#[test]
fn portable_observation_certificate_rejects_partial_duplicate_and_tampered_receipts() {
    let observer = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        2,
        1_700_000_500,
        [0x95; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 2,
                tip_hash: [0x96; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 3,
                tip_hash: [0x97; 32],
            },
        ],
        [0x98; 32],
        &observer,
    )
    .unwrap();
    let receipt = accepted_observation_receipt(&checkpoint, &witness, [0x99; 16], 1_700_000_501);

    assert_eq!(
        DirectoryObservationCertificateV1::new_verified(
            checkpoint.clone(),
            2,
            vec![receipt],
            1_700_000_501,
        ),
        Err(DirectoryObservationCertificateValidationError::InvalidReceiptCount)
    );
    let duplicate = DirectoryObservationCertificateV1 {
        protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
        chain_id: checkpoint.chain_id,
        checkpoint: checkpoint.clone(),
        minimum_witnesses: 1,
        receipts: vec![receipt, receipt],
    };
    assert_eq!(
        duplicate.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_501),
        Err(DirectoryObservationCertificateValidationError::DuplicateWitness)
    );

    let mut tampered = receipt;
    tampered.signature[0] ^= 1;
    let tampered = DirectoryObservationCertificateV1 {
        protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
        chain_id: checkpoint.chain_id,
        checkpoint: checkpoint.clone(),
        minimum_witnesses: 1,
        receipts: vec![tampered],
    };
    assert_eq!(
        tampered.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_501),
        Err(DirectoryObservationCertificateValidationError::InvalidReceiptSignature)
    );

    let mut invalid_checkpoint = checkpoint;
    invalid_checkpoint.observer_signature[0] ^= 1;
    let invalid_checkpoint = DirectoryObservationCertificateV1 {
        protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
        chain_id: invalid_checkpoint.chain_id,
        checkpoint: invalid_checkpoint,
        minimum_witnesses: 1,
        receipts: vec![receipt],
    };
    assert_eq!(
        invalid_checkpoint.verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, 1_700_000_501),
        Err(DirectoryObservationCertificateValidationError::InvalidCheckpoint)
    );
}
