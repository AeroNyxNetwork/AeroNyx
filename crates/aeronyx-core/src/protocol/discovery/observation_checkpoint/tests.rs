// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/observation_checkpoint/tests.rs
// ============================================
//! # Tests: Directory observation checkpoints
//!
//! Unit tests for Directory observation checkpoints, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

#[test]
fn test_directory_observation_checkpoint_is_canonical_and_signed() {
    let observer = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
    let tips = vec![
        DirectoryObservationTipV1 {
            producer: producer_b.public_key_bytes(),
            tip_height: 12,
            tip_hash: [0xb2; 32],
        },
        DirectoryObservationTipV1 {
            producer: producer_a.public_key_bytes(),
            tip_height: 11,
            tip_hash: [0xa1; 32],
        },
    ];
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        2,
        tips.clone(),
        [0x44; 32],
        &observer,
    )
    .unwrap();

    assert!(checkpoint
        .verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        )
        .is_ok());
    assert!(checkpoint.producer_tips[0].producer < checkpoint.producer_tips[1].producer);
    let reordered = DirectoryObservationCheckpointV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        2,
        tips.into_iter().rev().collect(),
        [0x44; 32],
        &observer,
    )
    .unwrap();
    assert_eq!(checkpoint.hash(), reordered.hash());
    assert_eq!(checkpoint.observer_signature, reordered.observer_signature);
}

#[test]
fn test_directory_observation_checkpoint_rejects_tamper_and_invalid_history() {
    let observer = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x42; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x43; 32]).unwrap();
    let tips = vec![
        DirectoryObservationTipV1 {
            producer: producer_a.public_key_bytes(),
            tip_height: 3,
            tip_hash: [0x51; 32],
        },
        DirectoryObservationTipV1 {
            producer: producer_b.public_key_bytes(),
            tip_height: 4,
            tip_hash: [0x52; 32],
        },
    ];
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        2,
        1_700_000_200,
        [0x61; 32],
        2,
        tips,
        [0x62; 32],
        &observer,
    )
    .unwrap();

    let mut tampered = checkpoint.clone();
    tampered.observation_root[0] ^= 1;
    assert_eq!(
        tampered.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            2,
            &[0x61; 32],
            1_700_000_100,
            1_700_000_200,
        ),
        Err(DirectoryObservationCheckpointValidationError::InvalidSignature)
    );
    assert_eq!(
        checkpoint.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            3,
            &checkpoint.hash(),
            1_700_000_201,
            1_700_000_200,
        ),
        Err(DirectoryObservationCheckpointValidationError::InvalidPosition)
    );

    let mut noncanonical = checkpoint;
    noncanonical.producer_tips.reverse();
    assert_eq!(
        noncanonical.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            2,
            &[0x61; 32],
            1_700_000_100,
            1_700_000_200,
        ),
        Err(DirectoryObservationCheckpointValidationError::NonCanonicalProducerOrder)
    );
}
