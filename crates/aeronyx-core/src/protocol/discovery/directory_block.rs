// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/directory_block.rs
// ============================================
//! # Directory Chain V1 commitment blocks
//!
//! Owns authenticated descriptor commitments, the signed block header and
//! block, block construction, and the `verify_at` / `verify_contents` /
//! `verify_position` validation contract with its shared position and
//! payload checks.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};
use crate::error::CoreError;
use crate::ledger::merkle_root;

use super::descriptor::SignedNodeDescriptor;
use super::{
    serde_bytes64, AERONYX_DIRECTORY_MAINNET_CHAIN_ID, MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS,
};

/// First stable Directory Chain hashing and signature contract.
pub const DIRECTORY_COMMITMENT_BLOCK_VERSION_V1: u16 = 1;

/// Maximum descriptor commitments accepted in one directory block.
///
/// At 72 bytes of canonical commitment data per entry, this keeps the payload
/// bounded while matching the existing maximum discovery snapshot page size.
pub const MAX_DIRECTORY_COMMITMENTS_PER_BLOCK: usize = 256;

// ============================================
// Directory Chain V1
// ============================================

/// Opaque, content-addressed commitment to one authenticated node descriptor.
///
/// The commitment identifies the public node and monotonic descriptor sequence
/// needed for deterministic replay, while the digest binds the complete signed
/// descriptor. Endpoint, region, capacity, policy, and capability fields are
/// not duplicated into the directory block payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DirectoryDescriptorCommitmentV1 {
    /// Public Ed25519 identity of the node that signed the descriptor.
    pub node_id: [u8; 32],
    /// Monotonic sequence copied from the authenticated descriptor.
    pub sequence: u64,
    /// Domain-separated digest of descriptor signing bytes and signature.
    pub descriptor_hash: [u8; 32],
}

impl DirectoryDescriptorCommitmentV1 {
    /// Creates a commitment after verifying the descriptor schema and signature.
    ///
    /// Expiry is deliberately not checked here: an authenticated descriptor may
    /// remain part of immutable directory history after it stops being routeable.
    ///
    /// # Errors
    /// Returns a `CoreError` when the descriptor schema, key, signature, or
    /// canonical serialization is invalid.
    pub fn from_signed_descriptor(descriptor: &SignedNodeDescriptor) -> Result<Self, CoreError> {
        descriptor.verify_signature()?;
        Ok(Self {
            node_id: descriptor.node_id(),
            sequence: descriptor.sequence(),
            descriptor_hash: signed_descriptor_commitment_hash(descriptor)?,
        })
    }

    /// Returns the domain-separated Merkle leaf for this commitment.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryDescriptorCommitment-v1");
        hasher.update(self.node_id);
        hasher.update(self.sequence.to_le_bytes());
        hasher.update(self.descriptor_hash);
        hasher.finalize().into()
    }

    /// Checks whether this commitment binds the supplied signed descriptor.
    ///
    /// # Errors
    /// Returns a `CoreError` when the supplied descriptor is not authentic or
    /// cannot be canonically serialized.
    pub fn matches_signed_descriptor(
        &self,
        descriptor: &SignedNodeDescriptor,
    ) -> Result<bool, CoreError> {
        let candidate = Self::from_signed_descriptor(descriptor)?;
        Ok(self == &candidate)
    }

    pub(super) fn structurally_valid(&self) -> bool {
        self.node_id != [0u8; 32] && self.sequence > 0 && self.descriptor_hash != [0u8; 32]
    }
}

/// Computes the stable digest committed by [`DirectoryDescriptorCommitmentV1`].
///
/// The descriptor signature is included so the commitment proves exactly which
/// authenticated descriptor object was observed. A length prefix keeps the
/// canonical field boundary explicit for future schema versions.
fn signed_descriptor_commitment_hash(
    descriptor: &SignedNodeDescriptor,
) -> Result<[u8; 32], CoreError> {
    let signing_bytes = descriptor.descriptor.signing_bytes()?;
    let signing_bytes_len = u32::try_from(signing_bytes.len()).map_err(|_| {
        CoreError::malformed("signed node descriptor canonical bytes exceed u32 length")
    })?;
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-SignedNodeDescriptorCommitment-v1");
    hasher.update(signing_bytes_len.to_le_bytes());
    hasher.update(signing_bytes);
    hasher.update(descriptor.signature);
    Ok(hasher.finalize().into())
}

/// Canonical signed header for one Directory Chain block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryCommitmentHeaderV1 {
    /// Stable hashing and signature contract version.
    pub protocol_version: u16,
    /// Prevents replay between production, test, and private directories.
    pub chain_id: [u8; 32],
    /// One-based block height.
    pub height: u64,
    /// Producer timestamp in Unix epoch seconds.
    pub timestamp: u64,
    /// Hash of the previous V1 header, or all zeroes at height one.
    pub prev_block_hash: [u8; 32],
    /// Merkle root of canonically sorted descriptor commitment leaves.
    pub commitment_root: [u8; 32],
    /// Number of commitments carried by the block.
    pub commitment_count: u32,
    /// Ed25519 identity of the node producing this block.
    pub producer: [u8; 32],
}

impl DirectoryCommitmentHeaderV1 {
    /// Computes the domain-separated canonical block identity.
    ///
    /// Field order and little-endian integer encoding are stable protocol
    /// contracts and must not change within V1.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryCommitmentBlock-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.height.to_le_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.prev_block_hash);
        hasher.update(self.commitment_root);
        hasher.update(self.commitment_count.to_le_bytes());
        hasher.update(self.producer);
        hasher.finalize().into()
    }

    /// Returns the canonical block hash as lowercase hexadecimal.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash())
    }
}

/// Signed, hash-linked directory block containing no client or traffic data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryCommitmentBlockV1 {
    /// Signed chain header.
    pub header: DirectoryCommitmentHeaderV1,
    /// Canonically sorted descriptor commitments.
    pub commitments: Vec<DirectoryDescriptorCommitmentV1>,
    /// Ed25519 signature by `header.producer` over `header.hash()`.
    #[serde(with = "serde_bytes64")]
    pub producer_signature: [u8; 64],
}

/// Validation failures for the V1 Directory Chain contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryCommitmentValidationError {
    /// The block hashing/signature contract version is unsupported.
    UnsupportedVersion,
    /// The block belongs to another directory chain.
    WrongChain,
    /// The block height is zero or does not continue the expected chain.
    InvalidHeight,
    /// The previous block hash does not match the expected chain tip.
    InvalidPreviousHash,
    /// The block timestamp is zero or regresses behind its predecessor.
    InvalidTimestamp,
    /// A directory block must carry at least one commitment.
    EmptyBlock,
    /// The block exceeds the commitment count bound.
    TooManyCommitments,
    /// Header and payload commitment counts differ.
    CommitmentCountMismatch,
    /// A commitment contains a sentinel identity, sequence, or digest.
    InvalidCommitment,
    /// Commitments are not in canonical lexicographic order.
    NonCanonicalOrder,
    /// The same descriptor commitment appears more than once.
    DuplicateCommitment,
    /// The payload does not match the signed Merkle root.
    InvalidMerkleRoot,
    /// The producer public key is malformed.
    InvalidProducer,
    /// The producer signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for DirectoryCommitmentValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory block protocol version",
            Self::WrongChain => "directory block belongs to another chain",
            Self::InvalidHeight => "directory block height does not continue the chain",
            Self::InvalidPreviousHash => "directory block previous hash does not match the tip",
            Self::InvalidTimestamp => "directory block timestamp is invalid",
            Self::EmptyBlock => "directory block is empty",
            Self::TooManyCommitments => "directory block exceeds the commitment limit",
            Self::CommitmentCountMismatch => "directory header count does not match payload",
            Self::InvalidCommitment => "directory descriptor commitment is invalid",
            Self::NonCanonicalOrder => "directory commitments are not canonically ordered",
            Self::DuplicateCommitment => "directory descriptor commitment is duplicated",
            Self::InvalidMerkleRoot => "directory commitment Merkle root is invalid",
            Self::InvalidProducer => "directory block producer public key is invalid",
            Self::InvalidSignature => "directory block producer signature is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryCommitmentValidationError {}

impl DirectoryCommitmentBlockV1 {
    /// Builds and signs one deterministic production directory block.
    ///
    /// Input commitments are sorted before hashing. The constructor rejects
    /// empty, oversized, duplicated, sentinel, or impossible genesis inputs so
    /// invalid local blocks are never signed accidentally.
    ///
    /// # Errors
    /// Returns a [`DirectoryCommitmentValidationError`] for invalid block or
    /// commitment inputs.
    pub fn new_signed(
        height: u64,
        timestamp: u64,
        prev_block_hash: [u8; 32],
        mut commitments: Vec<DirectoryDescriptorCommitmentV1>,
        identity: &IdentityKeyPair,
    ) -> Result<Self, DirectoryCommitmentValidationError> {
        validate_directory_block_position(height, timestamp, &prev_block_hash, 0)?;
        commitments.sort_unstable();
        validate_directory_commitments(&commitments)?;
        let commitment_hashes = commitments
            .iter()
            .map(DirectoryDescriptorCommitmentV1::hash)
            .collect::<Vec<_>>();
        let commitment_count = u32::try_from(commitments.len())
            .map_err(|_| DirectoryCommitmentValidationError::TooManyCommitments)?;
        let header = DirectoryCommitmentHeaderV1 {
            protocol_version: DIRECTORY_COMMITMENT_BLOCK_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            height,
            timestamp,
            prev_block_hash,
            commitment_root: merkle_root(&commitment_hashes),
            commitment_count,
            producer: identity.public_key_bytes(),
        };
        let producer_signature = identity.sign(&header.hash());
        Ok(Self {
            header,
            commitments,
            producer_signature,
        })
    }

    /// Returns the canonical block identity.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.header.hash()
    }

    /// Validates contract, chain continuity, canonical payload, Merkle root,
    /// and producer authenticity.
    ///
    /// `previous_timestamp` is zero for genesis and the prior block timestamp
    /// otherwise. Equal timestamps are accepted to tolerate one-second clocks.
    /// `observed_at` is the verifier's current Unix time and enforces a bounded
    /// future-clock lead.
    ///
    /// # Errors
    /// Returns a [`DirectoryCommitmentValidationError`] when the block breaks
    /// the V1 contract, expected chain position, canonical payload, Merkle
    /// commitment, timestamp bound, producer identity, or signature.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        expected_height: u64,
        expected_prev_hash: &[u8; 32],
        previous_timestamp: u64,
        observed_at: u64,
    ) -> Result<(), DirectoryCommitmentValidationError> {
        // Same checks, same order as before the split: contract, position,
        // then payload and signature.
        self.verify_contract(expected_chain_id)?;
        self.verify_position(expected_height, expected_prev_hash, previous_timestamp)?;
        self.verify_payload(observed_at)
    }

    /// Validates everything that does not depend on the previous block:
    /// contract, future-clock bound, canonical payload, Merkle root and
    /// producer signature.
    ///
    /// [PARALLEL-DIRECTORY-AUDIT 2026-10-10 by Claude] Lets an audit verify the
    /// expensive, independent part of many blocks concurrently and then run
    /// [`Self::verify_position`] in chain order. `verify_contents` followed by
    /// `verify_position` accepts exactly the blocks [`Self::verify_at`] accepts.
    ///
    /// # Errors
    /// Returns a [`DirectoryCommitmentValidationError`] when the block breaks
    /// the V1 contract, timestamp bound, canonical payload, Merkle commitment,
    /// producer identity, or signature.
    pub fn verify_contents(
        &self,
        expected_chain_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), DirectoryCommitmentValidationError> {
        self.verify_contract(expected_chain_id)?;
        self.verify_payload(observed_at)
    }

    /// Validates the block's position after its predecessor: height,
    /// previous-block hash, genesis rules and non-decreasing timestamp.
    ///
    /// # Errors
    /// Returns a [`DirectoryCommitmentValidationError`] when the block does not
    /// directly follow the expected predecessor.
    pub fn verify_position(
        &self,
        expected_height: u64,
        expected_prev_hash: &[u8; 32],
        previous_timestamp: u64,
    ) -> Result<(), DirectoryCommitmentValidationError> {
        if self.header.height != expected_height {
            return Err(DirectoryCommitmentValidationError::InvalidHeight);
        }
        if &self.header.prev_block_hash != expected_prev_hash {
            return Err(DirectoryCommitmentValidationError::InvalidPreviousHash);
        }
        validate_directory_block_position(
            self.header.height,
            self.header.timestamp,
            &self.header.prev_block_hash,
            previous_timestamp,
        )
    }

    fn verify_contract(
        &self,
        expected_chain_id: &[u8; 32],
    ) -> Result<(), DirectoryCommitmentValidationError> {
        if self.header.protocol_version != DIRECTORY_COMMITMENT_BLOCK_VERSION_V1 {
            return Err(DirectoryCommitmentValidationError::UnsupportedVersion);
        }
        if &self.header.chain_id != expected_chain_id {
            return Err(DirectoryCommitmentValidationError::WrongChain);
        }
        Ok(())
    }

    fn verify_payload(&self, observed_at: u64) -> Result<(), DirectoryCommitmentValidationError> {
        if self.header.timestamp > observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryCommitmentValidationError::InvalidTimestamp);
        }
        validate_directory_commitments(&self.commitments)?;
        if self.header.commitment_count as usize != self.commitments.len() {
            return Err(DirectoryCommitmentValidationError::CommitmentCountMismatch);
        }
        let commitment_hashes = self
            .commitments
            .iter()
            .map(DirectoryDescriptorCommitmentV1::hash)
            .collect::<Vec<_>>();
        if merkle_root(&commitment_hashes) != self.header.commitment_root {
            return Err(DirectoryCommitmentValidationError::InvalidMerkleRoot);
        }
        let producer = IdentityPublicKey::from_bytes(&self.header.producer)
            .map_err(|_| DirectoryCommitmentValidationError::InvalidProducer)?;
        producer
            .verify(&self.header.hash(), &self.producer_signature)
            .map_err(|_| DirectoryCommitmentValidationError::InvalidSignature)
    }
}

impl std::fmt::Display for DirectoryCommitmentBlockV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "DirectoryCommitmentBlockV1(height={}, commitments={}, hash={}..)",
            self.header.height,
            self.commitments.len(),
            &self.header.hash_hex()[..8],
        )
    }
}

pub(super) fn validate_directory_block_position(
    height: u64,
    timestamp: u64,
    prev_block_hash: &[u8; 32],
    previous_timestamp: u64,
) -> Result<(), DirectoryCommitmentValidationError> {
    if height == 0 {
        return Err(DirectoryCommitmentValidationError::InvalidHeight);
    }
    let genesis_position_valid = if height == 1 {
        prev_block_hash == &[0u8; 32]
    } else {
        prev_block_hash != &[0u8; 32]
    };
    if !genesis_position_valid {
        return Err(DirectoryCommitmentValidationError::InvalidPreviousHash);
    }
    if timestamp == 0 || timestamp < previous_timestamp {
        return Err(DirectoryCommitmentValidationError::InvalidTimestamp);
    }
    Ok(())
}

fn validate_directory_commitments(
    commitments: &[DirectoryDescriptorCommitmentV1],
) -> Result<(), DirectoryCommitmentValidationError> {
    if commitments.is_empty() {
        return Err(DirectoryCommitmentValidationError::EmptyBlock);
    }
    if commitments.len() > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK {
        return Err(DirectoryCommitmentValidationError::TooManyCommitments);
    }
    if commitments.iter().any(|entry| !entry.structurally_valid()) {
        return Err(DirectoryCommitmentValidationError::InvalidCommitment);
    }
    if commitments.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(DirectoryCommitmentValidationError::NonCanonicalOrder);
    }
    if commitments.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(DirectoryCommitmentValidationError::DuplicateCommitment);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
