// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/directory_block/tests.rs
// ============================================
//! # Tests: Directory Chain V1 commitment blocks
//!
//! Unit tests for Directory Chain V1 commitment blocks, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use bincode::Options;

use crate::protocol::discovery::test_support::descriptor_for;

// [PARALLEL-DIRECTORY-AUDIT 2026-10-10 by Claude] Audits now run
// verify_contents (in parallel) and verify_position (in order) instead of
// verify_at. The pair must accept exactly the blocks verify_at accepts.
#[test]
fn verify_contents_then_position_accepts_exactly_what_verify_at_accepts() {
    let producer = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
    let descriptor = SignedNodeDescriptor::sign(descriptor_for(&subject), &subject).unwrap();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
    let now = 1_700_000_100;
    let chain = AERONYX_DIRECTORY_MAINNET_CHAIN_ID;
    let genesis =
        DirectoryCommitmentBlockV1::new_signed(1, now, [0u8; 32], vec![commitment], &producer)
            .unwrap();
    let second = DirectoryCommitmentBlockV1::new_signed(
        2,
        now + 1,
        genesis.hash(),
        vec![commitment],
        &producer,
    )
    .unwrap();
    let mut bad_signature = second.clone();
    bad_signature.producer_signature[0] ^= 0x01;
    let mut wrong_chain = second.clone();
    wrong_chain.header.chain_id[0] ^= 0x01;
    let mut wrong_root = second.clone();
    wrong_root.header.commitment_root[0] ^= 0x01;
    let future = DirectoryCommitmentBlockV1::new_signed(
        2,
        now + MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS + 10,
        genesis.hash(),
        vec![commitment],
        &producer,
    )
    .unwrap();

    let cases: Vec<(&str, &DirectoryCommitmentBlockV1, u64, [u8; 32], u64, bool)> = vec![
        ("genesis", &genesis, 1, [0u8; 32], 0, true),
        ("second", &second, 2, genesis.hash(), now, true),
        ("wrong height", &second, 3, genesis.hash(), now, false),
        ("wrong previous hash", &second, 2, [0x07; 32], now, false),
        (
            "timestamp before predecessor",
            &second,
            2,
            genesis.hash(),
            now + 5,
            false,
        ),
        (
            "bad signature",
            &bad_signature,
            2,
            genesis.hash(),
            now,
            false,
        ),
        ("wrong chain", &wrong_chain, 2, genesis.hash(), now, false),
        (
            "wrong Merkle root",
            &wrong_root,
            2,
            genesis.hash(),
            now,
            false,
        ),
        (
            "too far in the future",
            &future,
            2,
            genesis.hash(),
            now,
            false,
        ),
    ];
    for (name, block, height, previous_hash, previous_timestamp, valid) in cases {
        let combined = block.verify_at(&chain, height, &previous_hash, previous_timestamp, now);
        let split = block
            .verify_contents(&chain, now)
            .and_then(|()| block.verify_position(height, &previous_hash, previous_timestamp));
        assert_eq!(combined.is_ok(), valid, "{name}: verify_at");
        assert_eq!(
            split.is_ok(),
            valid,
            "{name}: verify_contents + verify_position"
        );
    }
}

#[test]
fn test_directory_descriptor_commitment_binds_authenticated_descriptor() {
    let identity = IdentityKeyPair::generate();
    let signed = SignedNodeDescriptor::sign(descriptor_for(&identity), &identity).unwrap();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&signed).unwrap();

    assert_eq!(commitment.node_id, identity.public_key_bytes());
    assert_eq!(commitment.sequence, signed.sequence());
    assert!(commitment.matches_signed_descriptor(&signed).unwrap());
    assert_ne!(commitment.hash(), [0u8; 32]);

    let mut next_descriptor = descriptor_for(&identity);
    next_descriptor.sequence += 1;
    let next_signed = SignedNodeDescriptor::sign(next_descriptor, &identity).unwrap();
    assert!(!commitment.matches_signed_descriptor(&next_signed).unwrap());

    let mut forged = signed;
    forged.signature[0] ^= 0x01;
    assert!(commitment.matches_signed_descriptor(&forged).is_err());
}

#[test]
fn test_directory_block_is_deterministic_and_roundtrips() {
    let producer = IdentityKeyPair::generate();
    let first_identity = IdentityKeyPair::generate();
    let second_identity = IdentityKeyPair::generate();
    let first = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(descriptor_for(&first_identity), &first_identity).unwrap(),
    )
    .unwrap();
    let second = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(descriptor_for(&second_identity), &second_identity).unwrap(),
    )
    .unwrap();

    let forward = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![first, second],
        &producer,
    )
    .unwrap();
    let reverse = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![second, first],
        &producer,
    )
    .unwrap();

    assert_eq!(forward, reverse);
    assert_eq!(forward.header.commitment_count, 2);
    assert!(forward
        .verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        )
        .is_ok());
    let encoded = bincode::options()
        .with_fixint_encoding()
        .serialize(&forward)
        .unwrap();
    let decoded: DirectoryCommitmentBlockV1 = bincode::options()
        .with_fixint_encoding()
        .deserialize(&encoded)
        .unwrap();
    assert_eq!(decoded, forward);
    assert!(decoded.to_string().contains("height=1"));
}

#[test]
fn test_directory_block_verification_rejects_tampering() {
    let producer = IdentityKeyPair::generate();
    let node = IdentityKeyPair::generate();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(descriptor_for(&node), &node).unwrap(),
    )
    .unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![commitment],
        &producer,
    )
    .unwrap();

    let mut wrong_chain = block.clone();
    wrong_chain.header.chain_id[0] ^= 0x01;
    assert_eq!(
        wrong_chain.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        ),
        Err(DirectoryCommitmentValidationError::WrongChain)
    );

    let mut wrong_count = block.clone();
    wrong_count.header.commitment_count += 1;
    assert_eq!(
        wrong_count.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        ),
        Err(DirectoryCommitmentValidationError::CommitmentCountMismatch)
    );

    let mut wrong_root = block.clone();
    wrong_root.header.commitment_root[0] ^= 0x01;
    assert_eq!(
        wrong_root.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        ),
        Err(DirectoryCommitmentValidationError::InvalidMerkleRoot)
    );

    let mut wrong_signature = block;
    wrong_signature.producer_signature[0] ^= 0x01;
    assert_eq!(
        wrong_signature.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        ),
        Err(DirectoryCommitmentValidationError::InvalidSignature)
    );
}

#[test]
fn test_directory_block_rejects_invalid_and_unbounded_inputs() {
    let producer = IdentityKeyPair::generate();
    let node = IdentityKeyPair::generate();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(descriptor_for(&node), &node).unwrap(),
    )
    .unwrap();

    assert_eq!(
        DirectoryCommitmentBlockV1::new_signed(1, 1_700_000_100, [0u8; 32], Vec::new(), &producer,),
        Err(DirectoryCommitmentValidationError::EmptyBlock)
    );
    assert_eq!(
        DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![commitment, commitment],
            &producer,
        ),
        Err(DirectoryCommitmentValidationError::DuplicateCommitment)
    );
    assert_eq!(
        DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![commitment; MAX_DIRECTORY_COMMITMENTS_PER_BLOCK + 1],
            &producer,
        ),
        Err(DirectoryCommitmentValidationError::TooManyCommitments)
    );
    assert_eq!(
        DirectoryCommitmentBlockV1::new_signed(
            2,
            1_700_000_100,
            [0u8; 32],
            vec![commitment],
            &producer,
        ),
        Err(DirectoryCommitmentValidationError::InvalidPreviousHash)
    );
    assert_eq!(
        DirectoryCommitmentBlockV1::new_signed(1, 0, [0u8; 32], vec![commitment], &producer,),
        Err(DirectoryCommitmentValidationError::InvalidTimestamp)
    );

    let invalid = DirectoryDescriptorCommitmentV1 {
        node_id: [0u8; 32],
        ..commitment
    };
    assert_eq!(
        DirectoryCommitmentBlockV1::new_signed(
            1,
            1_700_000_100,
            [0u8; 32],
            vec![invalid],
            &producer,
        ),
        Err(DirectoryCommitmentValidationError::InvalidCommitment)
    );
}

#[test]
fn test_directory_block_chain_continuity_binds_height_hash_and_time() {
    let producer = IdentityKeyPair::generate();
    let first_node = IdentityKeyPair::generate();
    let second_node = IdentityKeyPair::generate();
    let first_commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(descriptor_for(&first_node), &first_node).unwrap(),
    )
    .unwrap();
    let mut second_descriptor = descriptor_for(&second_node);
    second_descriptor.sequence = 8;
    let second_commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(second_descriptor, &second_node).unwrap(),
    )
    .unwrap();
    let first = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![first_commitment],
        &producer,
    )
    .unwrap();
    let second = DirectoryCommitmentBlockV1::new_signed(
        2,
        1_700_000_101,
        first.hash(),
        vec![second_commitment],
        &producer,
    )
    .unwrap();

    assert!(second
        .verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            2,
            &first.hash(),
            first.header.timestamp,
            1_700_000_101,
        )
        .is_ok());
    assert_eq!(
        second.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            3,
            &first.hash(),
            first.header.timestamp,
            1_700_000_101,
        ),
        Err(DirectoryCommitmentValidationError::InvalidHeight)
    );

    let regressed = DirectoryCommitmentBlockV1::new_signed(
        2,
        1_700_000_099,
        first.hash(),
        vec![second_commitment],
        &producer,
    )
    .unwrap();
    assert_eq!(
        regressed.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            2,
            &first.hash(),
            first.header.timestamp,
            1_700_000_101,
        ),
        Err(DirectoryCommitmentValidationError::InvalidTimestamp)
    );

    let future = DirectoryCommitmentBlockV1::new_signed(
        2,
        1_700_000_222,
        first.hash(),
        vec![second_commitment],
        &producer,
    )
    .unwrap();
    assert_eq!(
        future.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            2,
            &first.hash(),
            first.header.timestamp,
            1_700_000_101,
        ),
        Err(DirectoryCommitmentValidationError::InvalidTimestamp)
    );
}

#[test]
fn test_directory_block_preserves_same_sequence_equivocation_evidence() {
    let producer = IdentityKeyPair::generate();
    let node = IdentityKeyPair::generate();
    let first_descriptor = descriptor_for(&node);
    let mut conflicting_descriptor = first_descriptor.clone();
    conflicting_descriptor.public_endpoint = Some("conflicting.example:443".to_string());
    let first = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(first_descriptor, &node).unwrap(),
    )
    .unwrap();
    let conflicting = DirectoryDescriptorCommitmentV1::from_signed_descriptor(
        &SignedNodeDescriptor::sign(conflicting_descriptor, &node).unwrap(),
    )
    .unwrap();

    assert_eq!(first.node_id, conflicting.node_id);
    assert_eq!(first.sequence, conflicting.sequence);
    assert_ne!(first.descriptor_hash, conflicting.descriptor_hash);
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![first, conflicting],
        &producer,
    )
    .unwrap();
    assert_eq!(block.commitments.len(), 2);
    assert!(block
        .verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            &[0u8; 32],
            0,
            1_700_000_100,
        )
        .is_ok());
}

#[test]
fn test_directory_block_v1_canonical_test_vector() {
    let producer = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
    let node = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
    let descriptor = SignedNodeDescriptor::sign(descriptor_for(&node), &node).unwrap();
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![commitment],
        &producer,
    )
    .unwrap();

    assert_eq!(
        hex::encode(commitment.descriptor_hash),
        "72d814f3d31e2a08d6f2003009cfa548be8e5fd05bc3ba38bb2285cea4432222"
    );
    assert_eq!(
        hex::encode(commitment.hash()),
        "fab10c677239ab88f615137654a4096aaa614b23b8eaea80bb898d1bf736d474"
    );
    assert_eq!(
        hex::encode(block.header.commitment_root),
        "fab10c677239ab88f615137654a4096aaa614b23b8eaea80bb898d1bf736d474"
    );
    assert_eq!(
        hex::encode(block.hash()),
        "51fc47f962be975d17e1f10e2ae9cc38201eea0e072f1bdb9bf3837ff2ad12c2"
    );
    assert_eq!(
        hex::encode(block.producer_signature),
        "8a5963474d6c0a6d94340593cbce67756b99e6a01919bde764c96d50fc57b092f479423b866b2c65036da8f2d2668c56d8c9b90782889e17a7ea2c34b4411e05"
    );
}
