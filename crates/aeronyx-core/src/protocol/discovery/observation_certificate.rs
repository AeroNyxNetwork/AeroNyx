// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/observation_certificate.rs
// ============================================
//! # Portable Directory observation certificates
//!
//! Owns standalone witness receipts, the portable observation certificate
//! that aggregates them for offline verification, and its bounded
//! magic-prefixed frame codec.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::crypto::IdentityPublicKey;
use crate::error::CoreError;
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

use super::observation_checkpoint::{
    DirectoryObservationCheckpointV1, MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1,
};
use super::sync_message::{DirectorySyncMessage, DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1};
use super::sync_signing::directory_observation_witness_response_signing_bytes;
use super::{serde_bytes64, MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS};

/// Stable portable observation-certificate contract version.
pub const DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1: u16 = 1;

/// One-byte discriminator prepended to portable observation certificates.
pub const DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC: u8 = 0xc7;

/// Maximum encoded certificate payload, excluding its magic byte.
///
/// The checkpoint and at most sixteen fixed-size receipts are substantially
/// smaller than this bound. The headroom preserves forward codec compatibility
/// without allowing attacker-controlled allocations.
const MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES: u64 = 64 * 1024;

/// Maximum complete portable observation-certificate frame size.
///
/// [PORTABLE-CERTIFICATE-VERIFIER 2026-07-26 by Codex] File, stdin, and
/// transport adapters must enforce this bound before allocating or decoding.
/// Keeping the complete-frame limit beside the canonical codec prevents each
/// adapter from inventing a subtly different allocation policy.
pub const MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES: usize =
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES as usize + 1;

// ============================================
// Portable Directory Observation Certificate V1
// ============================================

/// One independently signed accepted receipt carried by a portable certificate.
///
/// The receipt is a stable projection of
/// [`DirectorySyncMessage::ObservationCheckpointWitnessResponseV1`]. Keeping a
/// standalone representation lets offline verifiers validate a certificate
/// without interpreting an open-ended transport enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryObservationWitnessReceiptV1 {
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Original request identifier bound by the witness signature.
    pub request_id: [u8; 16],
    /// Observer identity copied from the witnessed checkpoint.
    pub observer: [u8; 32],
    /// Observer-local checkpoint sequence.
    pub checkpoint_sequence: u64,
    /// Exact canonical checkpoint hash evaluated by the witness.
    pub checkpoint_hash: [u8; 32],
    /// Independent witness identity.
    pub responder: [u8; 32],
    /// Witness response time in Unix epoch seconds.
    pub response_timestamp: u64,
    /// Stable witness outcome. Portable certificates accept only `accepted`.
    pub outcome: u8,
    /// Witness signature over every preceding field.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

impl DirectoryObservationWitnessReceiptV1 {
    /// Extracts a standalone receipt from the existing Directory Sync frame.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCertificateValidationError::InvalidReceiptContract`]
    /// when `message` is not an observation-witness response.
    pub fn from_sync_message(
        message: &DirectorySyncMessage,
    ) -> Result<Self, DirectoryObservationCertificateValidationError> {
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = message
        else {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptContract);
        };
        Ok(Self {
            chain_id: *chain_id,
            request_id: *request_id,
            observer: *observer,
            checkpoint_sequence: *checkpoint_sequence,
            checkpoint_hash: *checkpoint_hash,
            responder: *responder,
            response_timestamp: *response_timestamp,
            outcome: *outcome,
            signature: *signature,
        })
    }

    /// Recreates the backward-compatible Directory Sync response frame.
    #[must_use]
    pub const fn to_sync_message(self) -> DirectorySyncMessage {
        DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: self.chain_id,
            request_id: self.request_id,
            observer: self.observer,
            checkpoint_sequence: self.checkpoint_sequence,
            checkpoint_hash: self.checkpoint_hash,
            responder: self.responder,
            response_timestamp: self.response_timestamp,
            outcome: self.outcome,
            signature: self.signature,
        }
    }

    /// Computes a stable receipt identity including its witness signature.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &self.chain_id,
            &self.request_id,
            &self.observer,
            self.checkpoint_sequence,
            &self.checkpoint_hash,
            &self.responder,
            self.response_timestamp,
            self.outcome,
        );
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationWitnessReceipt-v1");
        hasher.update(signing_bytes);
        hasher.update(self.signature);
        hasher.finalize().into()
    }

    fn verify_for_checkpoint_at(
        &self,
        checkpoint: &DirectoryObservationCheckpointV1,
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCertificateValidationError> {
        let expected_checkpoint_hash = checkpoint.hash();
        if self.chain_id != checkpoint.chain_id
            || self.observer != checkpoint.observer
            || self.checkpoint_sequence != checkpoint.sequence
            || self.checkpoint_hash != expected_checkpoint_hash
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptContract);
        }
        if self.outcome != DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1
            || self.responder == [0u8; 32]
            || self.responder == checkpoint.observer
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptContract);
        }
        if self.response_timestamp < checkpoint.observed_at
            || self.response_timestamp
                > verifier_observed_at.saturating_add(MAX_DIRECTORY_BLOCK_FUTURE_SKEW_SECS)
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptTimestamp);
        }
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &self.chain_id,
            &self.request_id,
            &self.observer,
            self.checkpoint_sequence,
            &self.checkpoint_hash,
            &self.responder,
            self.response_timestamp,
            self.outcome,
        );
        IdentityPublicKey::from_bytes(&self.responder)
            .map_err(|_| DirectoryObservationCertificateValidationError::InvalidWitness)?
            .verify(&signing_bytes, &self.signature)
            .map_err(|_| DirectoryObservationCertificateValidationError::InvalidReceiptSignature)
    }
}

/// Portable, bounded evidence that independent nodes recomputed one checkpoint.
///
/// This package is not signed by an aggregator: every contained statement is
/// verified against the observer or witness identity that authored it. A
/// threshold records the exporting operator's evidence target only. It is not
/// validator membership, voting weight, fork choice, consensus, or finality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectoryObservationCertificateV1 {
    /// Stable certificate contract version.
    pub protocol_version: u16,
    /// Production Directory Chain identifier.
    pub chain_id: [u8; 32],
    /// Exact observer-signed checkpoint covered by every receipt.
    pub checkpoint: DirectoryObservationCheckpointV1,
    /// Minimum distinct receipts required by the exporting operator.
    pub minimum_witnesses: u16,
    /// Canonically responder-sorted accepted witness receipts.
    pub receipts: Vec<DirectoryObservationWitnessReceiptV1>,
}

/// Fail-closed portable observation-certificate validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationCertificateValidationError {
    /// The certificate contract version is unsupported.
    UnsupportedVersion,
    /// Certificate and checkpoint chain identifiers do not match the verifier.
    WrongChain,
    /// The embedded observer checkpoint is invalid.
    InvalidCheckpoint,
    /// The configured receipt threshold is zero or outside the protocol bound.
    InvalidThreshold,
    /// The receipt set is below threshold or outside the protocol bound.
    InvalidReceiptCount,
    /// Receipts are not sorted by witness identity.
    NonCanonicalWitnessOrder,
    /// The same witness appears more than once.
    DuplicateWitness,
    /// A receipt does not bind the exact accepted checkpoint.
    InvalidReceiptContract,
    /// A witness timestamp predates the checkpoint or is too far in the future.
    InvalidReceiptTimestamp,
    /// A witness public key is malformed.
    InvalidWitness,
    /// A receipt signature is invalid.
    InvalidReceiptSignature,
}

impl std::fmt::Display for DirectoryObservationCertificateValidationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::UnsupportedVersion => "unsupported directory observation certificate version",
            Self::WrongChain => "directory observation certificate belongs to another chain",
            Self::InvalidCheckpoint => "directory observation certificate checkpoint is invalid",
            Self::InvalidThreshold => {
                "directory observation certificate witness threshold is invalid"
            }
            Self::InvalidReceiptCount => {
                "directory observation certificate receipt count is invalid"
            }
            Self::NonCanonicalWitnessOrder => {
                "directory observation certificate witnesses are not canonically ordered"
            }
            Self::DuplicateWitness => {
                "directory observation certificate contains a duplicate witness"
            }
            Self::InvalidReceiptContract => {
                "directory observation certificate receipt contract is invalid"
            }
            Self::InvalidReceiptTimestamp => {
                "directory observation certificate receipt timestamp is invalid"
            }
            Self::InvalidWitness => "directory observation certificate witness identity is invalid",
            Self::InvalidReceiptSignature => {
                "directory observation certificate receipt signature is invalid"
            }
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for DirectoryObservationCertificateValidationError {}

impl DirectoryObservationCertificateV1 {
    /// Builds a canonical certificate and verifies every contained signature.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCertificateValidationError`] when the
    /// threshold, checkpoint, receipt binding, order, timestamp, identity, or
    /// signature is invalid.
    pub fn new_verified(
        checkpoint: DirectoryObservationCheckpointV1,
        minimum_witnesses: u16,
        mut receipts: Vec<DirectoryObservationWitnessReceiptV1>,
        verifier_observed_at: u64,
    ) -> Result<Self, DirectoryObservationCertificateValidationError> {
        receipts.sort_unstable_by_key(|receipt| receipt.responder);
        let certificate = Self {
            protocol_version: DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1,
            chain_id: checkpoint.chain_id,
            checkpoint,
            minimum_witnesses,
            receipts,
        };
        certificate.verify_at(&certificate.chain_id, verifier_observed_at)?;
        Ok(certificate)
    }

    /// Computes the stable identity of the checkpoint and exact receipt set.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-DirectoryObservationCertificate-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.checkpoint.hash());
        hasher.update(self.minimum_witnesses.to_le_bytes());
        hasher.update(
            u64::try_from(self.receipts.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        for receipt in &self.receipts {
            hasher.update(receipt.hash());
        }
        hasher.finalize().into()
    }

    /// Verifies the complete portable evidence package.
    ///
    /// # Errors
    /// Returns [`DirectoryObservationCertificateValidationError`] for any
    /// version, chain, checkpoint, threshold, canonical order, binding,
    /// timestamp, identity, or signature failure.
    pub fn verify_at(
        &self,
        expected_chain_id: &[u8; 32],
        verifier_observed_at: u64,
    ) -> Result<(), DirectoryObservationCertificateValidationError> {
        if self.protocol_version != DIRECTORY_OBSERVATION_CERTIFICATE_VERSION_V1 {
            return Err(DirectoryObservationCertificateValidationError::UnsupportedVersion);
        }
        if &self.chain_id != expected_chain_id || self.checkpoint.chain_id != self.chain_id {
            return Err(DirectoryObservationCertificateValidationError::WrongChain);
        }
        self.checkpoint
            .verify_standalone_at(expected_chain_id, verifier_observed_at)
            .map_err(|_| DirectoryObservationCertificateValidationError::InvalidCheckpoint)?;
        let threshold = usize::from(self.minimum_witnesses);
        if !(1..=MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).contains(&threshold) {
            return Err(DirectoryObservationCertificateValidationError::InvalidThreshold);
        }
        if self.receipts.len() < threshold
            || self.receipts.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryObservationCertificateValidationError::InvalidReceiptCount);
        }
        if self
            .receipts
            .windows(2)
            .any(|receipts| receipts[0].responder > receipts[1].responder)
        {
            return Err(DirectoryObservationCertificateValidationError::NonCanonicalWitnessOrder);
        }
        if self
            .receipts
            .windows(2)
            .any(|receipts| receipts[0].responder == receipts[1].responder)
        {
            return Err(DirectoryObservationCertificateValidationError::DuplicateWitness);
        }
        for receipt in &self.receipts {
            receipt.verify_for_checkpoint_at(&self.checkpoint, verifier_observed_at)?;
        }
        Ok(())
    }
}

/// Encodes one canonical bounded portable observation certificate.
///
/// Callers should run [`DirectoryObservationCertificateV1::verify_at`] with
/// their own current time before distributing a stored certificate. Encoding
/// itself enforces only the stable allocation bound.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization exceeds the bound
/// or fails.
pub fn encode_directory_observation_certificate(
    certificate: &DirectoryObservationCertificateV1,
) -> Result<Vec<u8>, CoreError> {
    let payload = encode_bincode_bounded(certificate, MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES)
        .map_err(|error| {
            CoreError::malformed(format!("directory observation certificate encode: {error}"))
        })?;
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one canonical bounded portable observation certificate.
///
/// Decoding does not establish trust. Call
/// [`DirectoryObservationCertificateV1::verify_at`] before using the evidence.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for a wrong magic byte, trailing data,
/// oversized payload, or malformed certificate.
pub fn decode_directory_observation_certificate(
    bytes: &[u8],
) -> Result<DirectoryObservationCertificateV1, CoreError> {
    if bytes.first().copied() != Some(DIRECTORY_OBSERVATION_CERTIFICATE_MAGIC) {
        return Err(CoreError::malformed(
            "directory observation certificate magic mismatch",
        ));
    }
    decode_bincode_bounded(
        &bytes[1..],
        MAX_DIRECTORY_OBSERVATION_CERTIFICATE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| {
        CoreError::malformed(format!("directory observation certificate decode: {error}"))
    })
}

#[cfg(test)]
mod tests;
