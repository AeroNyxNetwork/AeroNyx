// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/sync_message.rs
// ============================================
//! # Directory Sync V1 wire messages
//!
//! Owns the Directory Sync frame constants and outcome codes, the
//! append-only `DirectorySyncMessage` wire enum, and its bounded
//! magic-prefixed frame codec.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::protocol::codec::{decode_bincode_bounded, encode_bincode_bounded, TrailingBytesPolicy};

use super::descriptor::SignedNodeDescriptor;
use super::directory_block::DirectoryCommitmentBlockV1;
use super::inclusion_proof::DirectoryDescriptorInclusionProofV1;
use super::observation_checkpoint::DirectoryObservationCheckpointV1;
use super::serde_bytes64;

/// One-byte discriminator prepended to every Directory Sync V1 frame.
pub const DIRECTORY_SYNC_MAGIC: u8 = 0xd3;

/// Maximum encoded Directory Sync frame payload, excluding the magic byte.
const MAX_DIRECTORY_SYNC_MESSAGE_BYTES: u64 = 512 * 1024;

/// Maximum blocks returned by one Directory Sync range response.
pub const MAX_DIRECTORY_SYNC_BLOCKS_V1: u16 = 8;

/// Maximum content-addressed descriptors returned by one object response.
pub const MAX_DIRECTORY_SYNC_OBJECTS_V1: usize = 16;

/// Witness accepted the exact checkpoint after independent local recomputation.
pub const DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1: u8 = 1;
/// Witness lacks one or more exact retained producer prefixes.
pub const DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1: u8 = 2;
/// Witness has conflicting retained evidence or recomputed a different root.
pub const DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1: u8 = 3;

/// Witness durably retained the exact opaque observer policy head.
pub const DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1: u8 = 1;
/// Witness has a newer retained epoch and rejects observer rollback.
pub const DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1: u8 = 2;
/// Witness retained a different digest for the same observer epoch.
pub const DIRECTORY_POLICY_ANCHOR_CONFLICT_V1: u8 = 3;
/// Witness cannot connect the requested epoch to its retained policy head.
pub const DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1: u8 = 4;

// ============================================
// Directory Sync V1
// ============================================

/// Authenticated, bounded wire messages for one producer's Directory Chain.
///
/// A responder serves only the chain signed by its own node identity. Requests
/// are separately signed by an admitted peer. Descriptor objects remain public
/// node metadata; no user, route, traffic, or encrypted payload data belongs in
/// this protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DirectorySyncMessage {
    /// Requests the responder's current locally audited chain tip.
    TipRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over canonical tip-request bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns the responder's current locally audited chain tip.
    TipResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for this chain.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Current tip height, or zero for an empty chain.
        tip_height: u64,
        /// Current tip hash, or all zeroes for an empty chain.
        tip_hash: [u8; 32],
        /// Current tip block timestamp, or zero for an empty chain.
        tip_timestamp: u64,
        /// Responder signature over canonical tip-response bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests a contiguous bounded range from the responder's own chain.
    BlockRangeRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// First one-based block height requested.
        from_height: u64,
        /// Maximum number of blocks requested.
        limit: u16,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over canonical range-request bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns a contiguous bounded block range and a signed current tip.
    BlockRangeResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for every returned block.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Contiguous blocks in ascending height order.
        blocks: Vec<DirectoryCommitmentBlockV1>,
        /// Whether the signed tip extends beyond this page.
        has_more: bool,
        /// Current responder tip height.
        tip_height: u64,
        /// Current responder tip hash.
        tip_hash: [u8; 32],
        /// Responder signature over request binding, block hashes, and tip.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests exact content-addressed signed descriptor objects.
    DescriptorObjectsRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Descriptor commitment hashes, in required response order.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over canonical object-request bytes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns exact signed descriptor objects in requested hash order.
    DescriptorObjectsResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for the source chain.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Requested hashes in the exact order represented by `objects`.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Authenticated public node descriptors committed by those hashes.
        objects: Vec<SignedNodeDescriptor>,
        /// Responder signature over request binding and ordered object hashes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests an independent witness decision for one exact signed checkpoint.
    ///
    /// This variant is appended to preserve every existing bincode enum index.
    /// The responder must recompute producer-prefix evidence from its own
    /// replica store; validating only the observer signature is insufficient.
    ObservationCheckpointWitnessRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Requester identity; must equal `checkpoint.observer`.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Canonical observer-signed checkpoint to recompute independently.
        checkpoint: DirectoryObservationCheckpointV1,
        /// Requester signature binding the exact checkpoint hash.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one signed external decision for an exact checkpoint.
    ObservationCheckpointWitnessResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Observer identity copied from the witnessed checkpoint.
        observer: [u8; 32],
        /// Observer-local sequence copied from the witnessed checkpoint.
        checkpoint_sequence: u64,
        /// Exact canonical checkpoint hash evaluated by the witness.
        checkpoint_hash: [u8; 32],
        /// Independent witness identity.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// One stable `DIRECTORY_OBSERVATION_WITNESS_*_V1` outcome code.
        outcome: u8,
        /// Responder signature over every response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests a bounded producer range from an audited evidence carrier.
    ///
    /// This variant is appended to preserve every existing bincode enum index.
    /// Returned blocks remain signed by `producer`; `carrier` only transports
    /// evidence that it has already imported and audited.
    ReplicaBlockRangeRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Producer whose signed replica prefix is requested.
        producer: [u8; 32],
        /// First one-based block height requested.
        from_height: u64,
        /// Maximum number of blocks requested.
        limit: u16,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns a bounded producer-signed range through an audited carrier.
    ReplicaBlockRangeResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer identity carried by every returned block.
        producer: [u8; 32],
        /// Independent node transporting its audited replica evidence.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Contiguous producer-signed blocks in ascending height order.
        blocks: Vec<DirectoryCommitmentBlockV1>,
        /// Whether the audited producer tip extends beyond this page.
        has_more: bool,
        /// Audited producer tip height at the carrier.
        tip_height: u64,
        /// Audited producer tip hash at the carrier.
        tip_hash: [u8; 32],
        /// Carrier signature binding request, producer, block hashes, and tip.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests exact producer descriptor objects from an audited carrier.
    ReplicaDescriptorObjectsRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Producer namespace containing every requested descriptor object.
        producer: [u8; 32],
        /// Descriptor commitment hashes, in required response order.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns exact producer descriptor objects through an audited carrier.
    ReplicaDescriptorObjectsResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer namespace represented by `objects`.
        producer: [u8; 32],
        /// Independent node transporting its audited replica evidence.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Requested hashes in the exact order represented by `objects`.
        descriptor_hashes: Vec<[u8; 32]>,
        /// Signed public descriptors committed by the producer blocks.
        objects: Vec<SignedNodeDescriptor>,
        /// Carrier signature binding request, producer, and ordered hashes.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests durable external retention of one opaque witness-policy head.
    ///
    /// This variant is appended to preserve every existing bincode enum index.
    /// Policy member identities and endpoints are deliberately absent. The
    /// witness validates observer authentication and monotonic continuity, but
    /// does not approve the operator's policy or interpret its opaque digest.
    ObservationWitnessPolicyAnchorRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Node whose local witness policy is being externally anchored.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Positive observer-local policy epoch.
        policy_epoch: u64,
        /// Previous policy digest, or zero only for epoch one.
        previous_policy_digest: [u8; 32],
        /// Opaque digest of the observer-signed complete local policy object.
        policy_digest: [u8; 32],
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one signed external policy-head retention decision.
    ObservationWitnessPolicyAnchorResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Observer identity copied from the anchor request.
        observer: [u8; 32],
        /// Exact observer-local policy epoch evaluated by the witness.
        policy_epoch: u64,
        /// Exact opaque policy digest evaluated by the witness.
        policy_digest: [u8; 32],
        /// Independent witness identity.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// One stable `DIRECTORY_POLICY_ANCHOR_*_V1` outcome code.
        outcome: u8,
        /// Responder signature over every response field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests the responder's latest portable observation certificate.
    ///
    /// [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] This variant is appended to
    /// preserve every existing bincode enum index. The server must admit only
    /// authenticated pinned peers and must never expose this frame publicly.
    ObservationCertificateRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting pinned peer.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one exact portable observation certificate to a pinned peer.
    ///
    /// The responder authenticates transport only. Receivers must still verify
    /// the observer checkpoint, every witness receipt, local witness pins, and
    /// their own certificate-age policy before importing the evidence.
    ObservationCertificateResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Requester identity copied from the authenticated request.
        requester: [u8; 32],
        /// Pinned responder transporting its locally verified certificate.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// SHA-256 of the exact canonical certificate frame.
        certificate_sha256: [u8; 32],
        /// Canonical bounded portable observation-certificate frame.
        #[serde(with = "serde_bytes")]
        certificate_frame: Vec<u8>,
        /// Responder signature binding metadata, digest, and frame length.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Relays one exact observer-signed witness request through a carrier.
    ///
    /// [WITNESS-CARRIER 2026-07-26 by Codex] This variant is appended to
    /// preserve every existing bincode enum index. The carrier is transport
    /// only: it cannot alter the inner request, choose another witness, or
    /// produce an accepted witness receipt.
    ObservationCheckpointWitnessCarrierRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Random carrier-request identifier used for replay protection.
        request_id: [u8; 16],
        /// Observer that signed both this envelope and the inner request.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Exact pinned witness that must evaluate the inner request.
        witness: [u8; 32],
        /// SHA-256 of the exact canonical inner witness-request frame.
        witness_request_sha256: [u8; 32],
        /// Canonical observer-signed witness-request frame.
        #[serde(with = "serde_bytes")]
        witness_request_frame: Vec<u8>,
        /// Observer signature binding target, digest, and frame length.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one exact witness-signed response through a carrier.
    ///
    /// The carrier signature authenticates only the bounded transport
    /// envelope. Receivers must independently verify the inner response
    /// against the original observer request and the exact pinned witness.
    ObservationCheckpointWitnessCarrierResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Carrier-request identifier copied from the outer request.
        request_id: [u8; 16],
        /// Observer identity copied from the authenticated outer request.
        requester: [u8; 32],
        /// Exact witness that signed the inner response.
        witness: [u8; 32],
        /// Independent carrier transporting the exact response frame.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// SHA-256 of the exact inner request frame.
        witness_request_sha256: [u8; 32],
        /// SHA-256 of the exact inner response frame.
        witness_response_sha256: [u8; 32],
        /// Canonical witness-signed response frame.
        #[serde(with = "serde_bytes")]
        witness_response_frame: Vec<u8>,
        /// Carrier signature binding request, target, digests, and frame length.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests one compact proof for a descriptor in one exact selected block.
    ///
    /// [DIRECTORY-INCLUSION-PROOF 2026-07-27 by Codex] This variant is
    /// appended to preserve every existing bincode enum index. Server
    /// admission remains restricted to authenticated pinned peers.
    DescriptorInclusionProofRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Independently selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact content-addressed descriptor requested.
        descriptor_hash: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the requesting pinned node.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one producer-signed descriptor inclusion proof.
    ///
    /// The responder signature authenticates request/response transport. A
    /// receiver must separately call
    /// [`DirectoryDescriptorInclusionProofV1::verify_at`] with its own pinned
    /// producer and independently selected `block_hash`.
    DescriptorInclusionProofResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Producer and responder identity for the source block.
        responder: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Exact selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact requested descriptor content hash.
        descriptor_hash: [u8; 32],
        /// Compact producer-signed inclusion evidence.
        proof: DirectoryDescriptorInclusionProofV1,
        /// Responder signature binding request, proof digest, and response time.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Requests one exact producer descriptor proof from an audited carrier.
    ///
    /// [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] This variant is appended
    /// to preserve every existing bincode enum index. `producer` is the
    /// original block author; the responding carrier never replaces it.
    ReplicaDescriptorInclusionProofRequestV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Original producer whose signed replica block is selected.
        producer: [u8; 32],
        /// Independently selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact content-addressed descriptor requested.
        descriptor_hash: [u8; 32],
        /// Random request identifier used for replay protection.
        request_id: [u8; 16],
        /// Ed25519 identity of the authenticated recovery requester.
        requester: [u8; 32],
        /// Request creation time in Unix epoch seconds.
        request_timestamp: u64,
        /// Requester signature over every request field.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
    /// Returns one original producer-signed proof through an audited carrier.
    ///
    /// The carrier signature authenticates transport only. Receivers must
    /// independently verify `proof` against `producer` and `block_hash`.
    ReplicaDescriptorInclusionProofResponseV1 {
        /// Production Directory Chain identifier.
        chain_id: [u8; 32],
        /// Request identifier copied from the authenticated request.
        request_id: [u8; 16],
        /// Original producer whose signed block is represented by `proof`.
        producer: [u8; 32],
        /// Independent node transporting its audited replica evidence.
        carrier: [u8; 32],
        /// Response creation time in Unix epoch seconds.
        response_timestamp: u64,
        /// Exact selected producer-signed block hash.
        block_hash: [u8; 32],
        /// Exact requested descriptor content hash.
        descriptor_hash: [u8; 32],
        /// Compact original producer-signed inclusion evidence.
        proof: DirectoryDescriptorInclusionProofV1,
        /// Carrier signature binding request, producer, proof, and response.
        #[serde(with = "serde_bytes64")]
        signature: [u8; 64],
    },
}

/// Encodes a canonical bounded Directory Sync frame including its magic byte.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` when serialization fails.
pub fn encode_directory_sync_message(message: &DirectorySyncMessage) -> Result<Vec<u8>, CoreError> {
    let payload = encode_bincode_bounded(message, MAX_DIRECTORY_SYNC_MESSAGE_BYTES)
        .map_err(|error| CoreError::malformed(format!("directory sync encode: {error}")))?;
    let mut frame = Vec::with_capacity(payload.len() + 1);
    frame.push(DIRECTORY_SYNC_MAGIC);
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes one canonical bounded Directory Sync frame.
///
/// # Errors
/// Returns `CoreError::MalformedMessage` for a wrong magic byte, trailing data,
/// oversized payload, or malformed message.
pub fn decode_directory_sync_message(bytes: &[u8]) -> Result<DirectorySyncMessage, CoreError> {
    if bytes.first().copied() != Some(DIRECTORY_SYNC_MAGIC) {
        return Err(CoreError::malformed("directory sync magic mismatch"));
    }
    decode_bincode_bounded(
        &bytes[1..],
        MAX_DIRECTORY_SYNC_MESSAGE_BYTES,
        TrailingBytesPolicy::Reject,
    )
    .map_err(|error| CoreError::malformed(format!("directory sync decode: {error}")))
}

#[cfg(test)]
mod tests;
