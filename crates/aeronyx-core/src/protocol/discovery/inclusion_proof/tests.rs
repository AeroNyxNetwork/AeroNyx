// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/inclusion_proof/tests.rs
// ============================================
//! # Tests: Directory descriptor inclusion proofs
//!
//! Unit tests for Directory descriptor inclusion proofs, moved from the former
//! `protocol::discovery::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use super::*;

use crate::crypto::IdentityKeyPair;

use crate::protocol::discovery::test_support::descriptor_for;
use crate::protocol::discovery::AERONYX_DIRECTORY_MAINNET_CHAIN_ID;

#[test]
fn directory_descriptor_inclusion_proof_roundtrips_odd_tree() {
    let producer = IdentityKeyPair::generate();
    let signed_descriptors = (0u64..5)
        .map(|offset| {
            let identity = IdentityKeyPair::generate();
            let mut descriptor = descriptor_for(&identity);
            descriptor.sequence = descriptor.sequence.saturating_add(offset);
            SignedNodeDescriptor::sign(descriptor, &identity).unwrap()
        })
        .collect::<Vec<_>>();
    let commitments = signed_descriptors
        .iter()
        .map(|descriptor| {
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor).unwrap()
        })
        .collect::<Vec<_>>();
    let observed_at = 1_700_000_100;
    let block =
        DirectoryCommitmentBlockV1::new_signed(1, observed_at, [0u8; 32], commitments, &producer)
            .unwrap();
    let expected_block_hash = block.hash();

    for descriptor in &signed_descriptors {
        let proof =
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, descriptor, observed_at)
                .unwrap();
        assert_eq!(
            proof.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &producer.public_key_bytes(),
                &expected_block_hash,
                observed_at,
            ),
            Ok(())
        );
        assert_eq!(proof.block_hash(), expected_block_hash);
        assert!(proof.sibling_hashes.len() <= MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1);
    }
}

#[test]
fn directory_descriptor_inclusion_proof_rejects_wrong_trust_anchors_and_tampering() {
    let producer = IdentityKeyPair::generate();
    let first_identity = IdentityKeyPair::generate();
    let second_identity = IdentityKeyPair::generate();
    let first_signed =
        SignedNodeDescriptor::sign(descriptor_for(&first_identity), &first_identity).unwrap();
    let second_signed =
        SignedNodeDescriptor::sign(descriptor_for(&second_identity), &second_identity).unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&first_signed).unwrap(),
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&second_signed).unwrap(),
        ],
        &producer,
    )
    .unwrap();
    let block_hash = block.hash();
    let proof =
        DirectoryDescriptorInclusionProofV1::from_block_at(&block, &first_signed, 1_700_000_100)
            .unwrap();

    assert_eq!(
        proof.verify_at(
            &[0x41; 32],
            &producer.public_key_bytes(),
            &block_hash,
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::WrongChain)
    );
    assert_eq!(
        proof.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &[0x42; 32],
            &block_hash,
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::WrongProducer)
    );
    assert_eq!(
        proof.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &[0x43; 32],
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::WrongBlockHash)
    );

    let mut wrong_descriptor = proof.clone();
    wrong_descriptor.descriptor = second_signed;
    assert_eq!(
        wrong_descriptor.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &block_hash,
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::DescriptorMismatch)
    );

    let mut wrong_path = proof.clone();
    wrong_path.sibling_hashes[0][0] ^= 0x01;
    assert_eq!(
        wrong_path.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &block_hash,
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::InvalidMerkleProof)
    );

    let mut wrong_signature = proof.clone();
    wrong_signature.producer_signature[0] ^= 0x01;
    assert_eq!(
        wrong_signature.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &block_hash,
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::InvalidBlock)
    );

    let mut wrong_length = proof;
    wrong_length.sibling_hashes.clear();
    assert_eq!(
        wrong_length.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &block_hash,
            1_700_000_100,
        ),
        Err(DirectoryDescriptorInclusionProofError::InvalidProofLength)
    );
}

#[test]
fn directory_descriptor_inclusion_proof_is_bounded_and_requires_membership() {
    assert_eq!(
        directory_inclusion_proof_depth(MAX_DIRECTORY_COMMITMENTS_PER_BLOCK),
        MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1
    );

    let producer = IdentityKeyPair::generate();
    let included_identity = IdentityKeyPair::generate();
    let absent_identity = IdentityKeyPair::generate();
    let included =
        SignedNodeDescriptor::sign(descriptor_for(&included_identity), &included_identity).unwrap();
    let absent =
        SignedNodeDescriptor::sign(descriptor_for(&absent_identity), &absent_identity).unwrap();
    let block = DirectoryCommitmentBlockV1::new_signed(
        1,
        1_700_000_100,
        [0u8; 32],
        vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(&included).unwrap()],
        &producer,
    )
    .unwrap();

    assert_eq!(
        DirectoryDescriptorInclusionProofV1::from_block_at(&block, &absent, 1_700_000_100,),
        Err(DirectoryDescriptorInclusionProofError::DescriptorMismatch)
    );
}
