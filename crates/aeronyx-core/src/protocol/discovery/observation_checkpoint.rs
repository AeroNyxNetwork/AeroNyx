// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/observation_checkpoint.rs
// ============================================
//! # Directory observation checkpoints
//!
//! Owns observer-signed, hash-linked checkpoints binding one complete
//! configured producer-tip set to a recomputable overlap root, plus their
//! linked and standalone verification.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::{IdentityKeyPair, IdentityPublicKey};

use super::{
    serde_bytes64, AERONYX_DIRECTORY_MAINNET_CHAIN_ID, MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS,
};

/// First stable Directory observation-checkpoint hashing contract.
pub const DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1: u16 = 1;

/// Maximum producer tips bound into one observation checkpoint.
pub const MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1: usize = 16;

// ============================================
// Directory Observation Checkpoint V1
// ============================================

/// One exact producer prefix included in an observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DirectoryObservationTipV1 {
    /// Producer identity whose independently signed chain was observed.
    pub producer: [u8; 32],
    /// Accepted one-based prefix height.
    pub tip_height: u64,
    /// Producer-signed block hash at `tip_height`.
    pub tip_hash: [u8; 32],
}

impl DirectoryObservationTipV1 {
    fn structurally_valid(&self) -> bool {
        self.producer != [0u8; 32] && self.tip_height > 0 && self.tip_hash != [0u8; 32]
    }
}

/// Observer-signed evidence for one complete configured producer-tip set.
///
/// The overlap root is recomputable from retained producer blocks and public
/// descriptor commitments. This object records what one observer verified; it
/// is not consensus, finality, fork choice, or a quorum certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryObservationCheckpointV1 {
    /// Stable hashing/signature contract version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// One-based observer-local checkpoint sequence.
    pub sequence: u64,
    /// Observer timestamp in Unix epoch seconds.
    pub observed_at: u64,
    /// Prior checkpoint hash, or zero for sequence one.
    pub previous_checkpoint_hash: [u8; 32],
    /// Node identity that created this observation.
    pub observer: [u8; 32],
    /// Number of configured producers represented by this complete checkpoint.
    pub configured_producer_count: u16,
    /// Canonically sorted exact producer tips.
    pub producer_tips: Vec<DirectoryObservationTipV1>,
    /// Deterministic overlap root recomputed from the represented prefixes.
    pub observation_root: [u8; 32],
    /// Ed25519 observer signature over [`Self::hash`].
    #[serde(with = "serde_bytes64")]
    pub observer_signature: [u8; 64],
}

/// Validation failures for Directory Observation Checkpoint V1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationCheckpointValidationError {
    /// The hashing/signature contract version is unsupported.
    UnsupportedVersion,
    /// The checkpoint belongs to another Directory Chain.
    WrongChain,
    /// Sequence or prior-checkpoint linkage is invalid.
    InvalidPosition,
    /// Timestamp is zero, regressed, or too far in the future.
    InvalidTimestamp,
    /// The configured producer count is outside the V1 bound.
    InvalidProducerCount,
    /// A producer tip contains a sentinel or duplicates the observer.
    InvalidProducerTip,
    /// Producer tips are not in canonical ascending order.
    NonCanonicalProducerOrder,
    /// The same producer occurs more than once.
    DuplicateProducer,
    /// The overlap root uses the zero sentinel.
    InvalidObservationRoot,
    /// The observer public key is malformed.
    InvalidObserver,
    /// The observer signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for DirectoryObservationCheckpointValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory observation checkpoint version",
            Self::WrongChain => "directory observation checkpoint belongs to another chain",
            Self::InvalidPosition => "directory observation checkpoint position is invalid",
            Self::InvalidTimestamp => "directory observation checkpoint timestamp is invalid",
            Self::InvalidProducerCount => {
                "directory observation checkpoint producer count is invalid"
            }
            Self::InvalidProducerTip => "directory observation checkpoint producer tip is invalid",
            Self::NonCanonicalProducerOrder => {
                "directory observation checkpoint producers are not canonically ordered"
            }
            Self::DuplicateProducer => {
                "directory observation checkpoint contains a duplicate producer"
            }
            Self::InvalidObservationRoot => {
                "directory observation checkpoint overlap root is invalid"
            }
            Self::InvalidObserver => "directory observation checkpoint observer is invalid",
            Self::InvalidSignature => "directory observation checkpoint signature is invalid",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryObservationCheckpointValidationError {}

impl DirectoryObservationCheckpointV1 {
    /// Builds and signs one complete, canonical observation checkpoint.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCheckpointValidationError`] when sequence,
    /// timestamp, producer tips, root, or observer identity is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new_signed(
        sequence: u64,
        observed_at: u64,
        previous_checkpoint_hash: [u8; 32],
        configured_producer_count: u16,
        mut producer_tips: Vec<DirectoryObservationTipV1>,
        observation_root: [u8; 32],
        identity: &IdentityKeyPair,
    ) -> Result<Self, DirectoryObservationCheckpointValidationError> {
        producer_tips.sort_unstable();
        let mut checkpoint = Self {
            protocol_version: DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1,
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            sequence,
            observed_at,
            previous_checkpoint_hash,
            observer: identity.public_key_bytes(),
            configured_producer_count,
            producer_tips,
            observation_root,
            observer_signature: [0u8; 64],
        };
        checkpoint.validate_structure()?;
        checkpoint.observer_signature = identity.sign(&checkpoint.hash());
        Ok(checkpoint)
    }

    /// Computes the domain-separated canonical checkpoint identity.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationCheckpoint-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.sequence.to_le_bytes());
        hasher.update(self.observed_at.to_le_bytes());
        hasher.update(self.previous_checkpoint_hash);
        hasher.update(self.observer);
        hasher.update(self.configured_producer_count.to_le_bytes());
        hasher.update(
            u64::try_from(self.producer_tips.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for tip in &self.producer_tips {
            hasher.update(tip.producer);
            hasher.update(tip.tip_height.to_le_bytes());
            hasher.update(tip.tip_hash);
        }
        hasher.update(self.observation_root);
        hasher.finalize().into()
    }

    /// Verifies structure, position, timestamp, observer identity, and signature.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCheckpointValidationError`] on any
    /// contract, continuity, time, canonicalization, identity, or signature
    /// mismatch.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        expected_sequence: u64,
        expected_previous_hash: &[u8; 32],
        previous_observed_at: u64,
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCheckpointValidationError> {
        self.validate_structure()?;
        if &self.chain_id != expected_chain_id {
            return Err(DirectoryObservationCheckpointValidationError::WrongChain);
        }
        if self.sequence != expected_sequence
            || &self.previous_checkpoint_hash != expected_previous_hash
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidPosition);
        }
        if self.observed_at < previous_observed_at
            || self.observed_at
                > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidTimestamp);
        }
        IdentityPublicKey::from_bytes(&self.observer)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidObserver)?
            .verify(&self.hash(), &self.observer_signature)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidSignature)
    }

    /// Verifies a standalone checkpoint's structure, chain, time, and observer
    /// signature without claiming knowledge of the observer's prior sequence.
    ///
    /// An external witness uses this before independently recomputing every
    /// referenced producer prefix and the observation root from its own store.
    /// Sequence linkage remains the observer's local append-only invariant and
    /// is deliberately not inferred from a single transported checkpoint.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCheckpointValidationError`] when the
    /// checkpoint is malformed, belongs to another chain, is too far in the
    /// future, or has an invalid observer identity/signature.
    pub fn verify_standalone_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCheckpointValidationError> {
        self.validate_structure()?;
        if &self.chain_id != expected_chain_id {
            return Err(DirectoryObservationCheckpointValidationError::WrongChain);
        }
        if self.observed_at
            > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidTimestamp);
        }
        IdentityPublicKey::from_bytes(&self.observer)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidObserver)?
            .verify(&self.hash(), &self.observer_signature)
            .map_err(|_| DirectoryObservationCheckpointValidationError::InvalidSignature)
    }

    fn validate_structure(&self) -> Result<(), DirectoryObservationCheckpointValidationError> {
        if self.protocol_version != DIRECTORY_OBSERVATION_CHECKPOINT_VERSION_V1 {
            return Err(DirectoryObservationCheckpointValidationError::UnsupportedVersion);
        }
        let producer_count = usize::from(self.configured_producer_count);
        if !(2..=MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).contains(&producer_count)
            || producer_count != self.producer_tips.len()
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidProducerCount);
        }
        let genesis_position_valid = if self.sequence == 1 {
            self.previous_checkpoint_hash == [0u8; 32]
        } else {
            self.sequence > 1 && self.previous_checkpoint_hash != [0u8; 32]
        };
        if !genesis_position_valid {
            return Err(DirectoryObservationCheckpointValidationError::InvalidPosition);
        }
        if self.observed_at == 0 {
            return Err(DirectoryObservationCheckpointValidationError::InvalidTimestamp);
        }
        if self.observer == [0u8; 32]
            || self
                .producer_tips
                .iter()
                .any(|tip| !tip.structurally_valid() || tip.producer == self.observer)
        {
            return Err(DirectoryObservationCheckpointValidationError::InvalidProducerTip);
        }
        if self.producer_tips.windows(2).any(|tips| tips[0] > tips[1]) {
            return Err(DirectoryObservationCheckpointValidationError::NonCanonicalProducerOrder);
        }
        if self
            .producer_tips
            .windows(2)
            .any(|tips| tips[0].producer == tips[1].producer)
        {
            return Err(DirectoryObservationCheckpointValidationError::DuplicateProducer);
        }
        if self.observation_root == [0u8; 32] {
            return Err(DirectoryObservationCheckpointValidationError::InvalidObservationRoot);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
