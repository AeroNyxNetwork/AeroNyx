// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/inclusion_proof.rs
// ============================================
//! # Directory descriptor inclusion proofs
//!
//! Owns compact producer-signed Merkle proofs that one authenticated
//! descriptor commitment is included in one exact Directory block, and the
//! header/depth checks used to verify them without the full block payload.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};

use crate::crypto::IdentityPublicKey;
use crate::ledger::{build_merkle_inclusion_proof, verify_merkle_inclusion_proof};

use super::descriptor::SignedNodeDescriptor;
use super::directory_block::{
    validate_directory_block_position, DirectoryCommitmentBlockV1, DirectoryCommitmentHeaderV1,
    DirectoryDescriptorCommitmentV1, DIRECTORY_COMMITMENT_BLOCK_VERSION_V1,
    MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
};
use super::{serde_bytes64, MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS};

/// Current compact Directory descriptor-inclusion proof contract version.
pub const DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1: u16 = 1;

/// Maximum sibling hashes for a 256-leaf Directory commitment tree.
///
/// [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] This is a wire and
/// allocation bound. The contract test fails if the block limit changes
/// without a reviewed proof-version update.
pub const MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1: usize = 8;

/// Compact proof that one authenticated descriptor commitment is included in
/// one exact producer-signed Directory block.
///
/// The proof intentionally carries no user, traffic, message, route, Memory
/// Chain, DNS, destination, or wallet data. It is useful only when the verifier
/// independently trusts `expected_producer` and `expected_block_hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryDescriptorInclusionProofV1 {
    /// Stable proof contract version.
    pub proof_version: u16,
    /// Exact producer-signed block header containing the commitment root.
    pub block_header: DirectoryCommitmentHeaderV1,
    /// Producer signature over `block_header.hash()`.
    #[serde(with = "serde_bytes64")]
    pub producer_signature: [u8; 64],
    /// Exact descriptor commitment used as the Merkle leaf.
    pub commitment: DirectoryDescriptorCommitmentV1,
    /// Zero-based commitment position in the canonical block payload.
    pub commitment_index: u32,
    /// Sibling hashes ordered from the leaf level toward the root.
    pub sibling_hashes: Vec<[u8; 32]>,
    /// Exact authenticated descriptor object bound by `commitment`.
    pub descriptor: SignedNodeDescriptor,
}

/// Fail-closed validation outcomes for a compact Directory inclusion proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryDescriptorInclusionProofError {
    /// The inclusion-proof contract version is unsupported.
    UnsupportedVersion,
    /// The proof belongs to another Directory chain.
    WrongChain,
    /// The producer differs from the verifier's independently pinned producer.
    WrongProducer,
    /// The signed block differs from the verifier's independently selected block.
    WrongBlockHash,
    /// The signed block header is structurally or cryptographically invalid.
    InvalidBlock,
    /// The descriptor object is malformed or has an invalid signature.
    InvalidDescriptor,
    /// The commitment does not bind the included descriptor object.
    DescriptorMismatch,
    /// The leaf index or declared block commitment count is invalid.
    InvalidPosition,
    /// The sibling path exceeds or differs from the exact tree depth.
    InvalidProofLength,
    /// The sibling path does not reconstruct the signed commitment root.
    InvalidMerkleProof,
}

impl std::fmt::Display for DirectoryDescriptorInclusionProofError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory inclusion proof version",
            Self::WrongChain => "directory inclusion proof belongs to another chain",
            Self::WrongProducer => "directory inclusion proof producer is not trusted",
            Self::WrongBlockHash => "directory inclusion proof block hash is not selected",
            Self::InvalidBlock => "directory inclusion proof block is invalid",
            Self::InvalidDescriptor => "directory inclusion proof descriptor is invalid",
            Self::DescriptorMismatch => {
                "directory inclusion proof commitment does not bind descriptor"
            }
            Self::InvalidPosition => "directory inclusion proof position is invalid",
            Self::InvalidProofLength => "directory inclusion proof path length is invalid",
            Self::InvalidMerkleProof => "directory inclusion proof path is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryDescriptorInclusionProofError {}

impl DirectoryDescriptorInclusionProofV1 {
    /// Builds a compact proof from one complete, valid Directory block.
    ///
    /// The constructor validates the block payload and signature at
    /// `observed_at`, authenticates `descriptor`, and requires its exact
    /// commitment to be present. Chain-continuity selection remains the
    /// caller's responsibility; this constructor cannot choose a canonical
    /// producer history.
    ///
    /// # Errors
    /// Returns a fail-closed proof error when the block, descriptor,
    /// commitment lookup, or bounded Merkle path is invalid.
    pub fn from_block_at(
        block: &DirectoryCommitmentBlockV1,
        descriptor: &SignedNodeDescriptor,
        observed_at: u64,
    ) -> Result<Self, DirectoryDescriptorInclusionProofError> {
        block
            .verify_at(
                &block.header.chain_id,
                block.header.height,
                &block.header.prev_block_hash,
                0,
                observed_at,
            )
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)?;
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidDescriptor)?;
        let commitment_index = block
            .commitments
            .binary_search(&commitment)
            .map_err(|_| DirectoryDescriptorInclusionProofError::DescriptorMismatch)?;
        let commitment_hashes = block
            .commitments
            .iter()
            .map(DirectoryDescriptorCommitmentV1::hash)
            .collect::<Vec<_>>();
        let sibling_hashes = build_merkle_inclusion_proof(&commitment_hashes, commitment_index)
            .ok_or(DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        if sibling_hashes.len() > MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1 {
            return Err(DirectoryDescriptorInclusionProofError::InvalidProofLength);
        }
        let commitment_index = u32::try_from(commitment_index)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        let proof = Self {
            proof_version: DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1,
            block_header: block.header.clone(),
            producer_signature: block.producer_signature,
            commitment,
            commitment_index,
            sibling_hashes,
            descriptor: descriptor.clone(),
        };
        proof.verify_at(
            &block.header.chain_id,
            &block.header.producer,
            &block.hash(),
            observed_at,
        )?;
        Ok(proof)
    }

    /// Verifies one proof against an independently selected producer and block.
    ///
    /// A successful result proves producer-signed inclusion only. It does not
    /// choose a canonical chain or establish voting, quorum, consensus,
    /// finality, transaction inclusion, or user activity.
    ///
    /// # Errors
    /// Returns a stable fail-closed proof error when any expected trust anchor,
    /// block signature, descriptor binding, position, or sibling hash fails.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        expected_producer: &[u8; 32],
        expected_block_hash: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), DirectoryDescriptorInclusionProofError> {
        if self.proof_version != DIRECTORY_DESCRIPTOR_INCLUSION_PROOF_VERSION_V1 {
            return Err(DirectoryDescriptorInclusionProofError::UnsupportedVersion);
        }
        if &self.block_header.chain_id != expected_chain_id {
            return Err(DirectoryDescriptorInclusionProofError::WrongChain);
        }
        if &self.block_header.producer != expected_producer {
            return Err(DirectoryDescriptorInclusionProofError::WrongProducer);
        }
        if &self.block_header.hash() != expected_block_hash {
            return Err(DirectoryDescriptorInclusionProofError::WrongBlockHash);
        }
        verify_directory_inclusion_header_at(
            &self.block_header,
            &self.producer_signature,
            observed_at,
        )?;
        let commitment_count = usize::try_from(self.block_header.commitment_count)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        let commitment_index = usize::try_from(self.commitment_index)
            .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
        if commitment_count == 0
            || commitment_count > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK
            || commitment_index >= commitment_count
        {
            return Err(DirectoryDescriptorInclusionProofError::InvalidPosition);
        }
        let expected_depth = directory_inclusion_proof_depth(commitment_count);
        if self.sibling_hashes.len() != expected_depth
            || self.sibling_hashes.len() > MAX_DIRECTORY_DESCRIPTOR_INCLUSION_SIBLINGS_V1
        {
            return Err(DirectoryDescriptorInclusionProofError::InvalidProofLength);
        }
        if !self.commitment.structurally_valid() {
            return Err(DirectoryDescriptorInclusionProofError::InvalidDescriptor);
        }
        match self.commitment.matches_signed_descriptor(&self.descriptor) {
            Ok(true) => {}
            Ok(false) => {
                return Err(DirectoryDescriptorInclusionProofError::DescriptorMismatch);
            }
            Err(_) => {
                return Err(DirectoryDescriptorInclusionProofError::InvalidDescriptor);
            }
        }
        if !verify_merkle_inclusion_proof(
            &self.block_header.commitment_root,
            &self.commitment.hash(),
            commitment_index,
            commitment_count,
            &self.sibling_hashes,
        ) {
            return Err(DirectoryDescriptorInclusionProofError::InvalidMerkleProof);
        }
        Ok(())
    }

    /// Returns the exact producer-signed block identity bound by this proof.
    #[must_use]
    pub fn block_hash(&self) -> [u8; 32] {
        self.block_header.hash()
    }
}

fn verify_directory_inclusion_header_at(
    header: &DirectoryCommitmentHeaderV1,
    producer_signature: &[u8; 64],
    observed_at: u64,
) -> Result<(), DirectoryDescriptorInclusionProofError> {
    if header.protocol_version != DIRECTORY_COMMITMENT_BLOCK_VERSION_V1 {
        return Err(DirectoryDescriptorInclusionProofError::InvalidBlock);
    }
    validate_directory_block_position(header.height, header.timestamp, &header.prev_block_hash, 0)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)?;
    if header.timestamp > observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS) {
        return Err(DirectoryDescriptorInclusionProofError::InvalidBlock);
    }
    let commitment_count = usize::try_from(header.commitment_count)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidPosition)?;
    if commitment_count == 0 || commitment_count > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK {
        return Err(DirectoryDescriptorInclusionProofError::InvalidPosition);
    }
    let producer = IdentityPublicKey::from_bytes(&header.producer)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)?;
    producer
        .verify(&header.hash(), producer_signature)
        .map_err(|_| DirectoryDescriptorInclusionProofError::InvalidBlock)
}

fn directory_inclusion_proof_depth(mut commitment_count: usize) -> usize {
    let mut depth = 0usize;
    while commitment_count > 1 {
        commitment_count = commitment_count.saturating_add(1) / 2;
        depth = depth.saturating_add(1);
    }
    depth
}

#[cfg(test)]
mod tests;
