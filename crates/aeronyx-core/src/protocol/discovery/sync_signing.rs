// ============================================
// File: crates/aeronyx-core/src/protocol/discovery/sync_signing.rs
// ============================================
//! # Directory Sync signing digests
//!
//! Owns the canonical domain-separated, length-prefixed digests signed by
//! every Directory Sync request and response frame.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from protocol/discovery.rs; bodies unchanged.

use sha2::{Digest, Sha256};

use super::directory_block::DirectoryCommitmentBlockV1;
use super::inclusion_proof::DirectoryDescriptorInclusionProofV1;

pub(super) fn directory_sync_signing_digest<'a>(
    domain: &[u8],
    fields: impl IntoIterator<Item = &'a [u8]>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for field in fields {
        hasher.update(u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().into()
}

/// Canonical digest signed by a Directory Sync tip request.
#[must_use]
pub fn directory_tip_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-TipRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a Directory Sync tip response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_tip_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    tip_height: u64,
    tip_hash: &[u8; 32],
    tip_timestamp: u64,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let tip_height = tip_height.to_le_bytes();
    let tip_timestamp = tip_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-TipResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            tip_height.as_slice(),
            tip_hash.as_slice(),
            tip_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a Directory Sync block-range request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_block_range_request_signing_bytes(
    chain_id: &[u8; 32],
    from_height: u64,
    limit: u16,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let from_height = from_height.to_le_bytes();
    let limit = limit.to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-BlockRangeRequest-v1",
        [
            chain_id.as_slice(),
            from_height.as_slice(),
            limit.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a Directory Sync block-range response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_block_range_response_signing_bytes(
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let block_count = u64::try_from(blocks.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let has_more = [u8::from(has_more)];
    let tip_height = tip_height.to_le_bytes();
    let block_hashes = blocks
        .iter()
        .map(DirectoryCommitmentBlockV1::hash)
        .collect::<Vec<_>>();
    let mut fields = Vec::<&[u8]>::with_capacity(block_hashes.len() + 7);
    fields.extend([
        request_id.as_slice(),
        responder.as_slice(),
        response_timestamp.as_slice(),
        block_count.as_slice(),
    ]);
    fields.extend(block_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        has_more.as_slice(),
        tip_height.as_slice(),
        tip_hash.as_slice(),
    ]);
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-BlockRangeResponse-v1", fields)
}

/// Canonical digest signed by a Directory Sync object request.
#[must_use]
pub fn directory_descriptor_objects_request_signing_bytes(
    chain_id: &[u8; 32],
    descriptor_hashes: &[[u8; 32]],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 5);
    fields.extend([chain_id.as_slice(), count.as_slice()]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        request_id.as_slice(),
        requester.as_slice(),
        request_timestamp.as_slice(),
    ]);
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ObjectsRequest-v1", fields)
}

/// Canonical digest signed by a Directory Sync object response.
#[must_use]
pub fn directory_descriptor_objects_response_signing_bytes(
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    descriptor_hashes: &[[u8; 32]],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 4);
    fields.extend([
        request_id.as_slice(),
        responder.as_slice(),
        response_timestamp.as_slice(),
        count.as_slice(),
    ]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ObjectsResponse-v1", fields)
}

/// Canonical digest signed by an exact descriptor-inclusion proof request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_descriptor_inclusion_proof_request_signing_bytes(
    chain_id: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-DescriptorInclusionProofRequest-v1",
        [
            chain_id.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

fn directory_descriptor_inclusion_proof_transport_digest(
    proof: &DirectoryDescriptorInclusionProofV1,
) -> [u8; 32] {
    let proof_version = proof.proof_version.to_le_bytes();
    let block_hash = proof.block_hash();
    let commitment_hash = proof.commitment.hash();
    let commitment_index = proof.commitment_index.to_le_bytes();
    let sibling_count = u64::try_from(proof.sibling_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(proof.sibling_hashes.len() + 6);
    fields.extend([
        proof_version.as_slice(),
        block_hash.as_slice(),
        proof.producer_signature.as_slice(),
        commitment_hash.as_slice(),
        commitment_index.as_slice(),
        sibling_count.as_slice(),
    ]);
    fields.extend(proof.sibling_hashes.iter().map(<[u8; 32]>::as_slice));
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-DescriptorInclusionProof-v1", fields)
}

/// Canonical digest signed by an exact descriptor-inclusion proof response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_descriptor_inclusion_proof_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    proof: &DirectoryDescriptorInclusionProofV1,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let proof_digest = directory_descriptor_inclusion_proof_transport_digest(proof);
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-DescriptorInclusionProofResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            proof_digest.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica descriptor-proof request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_descriptor_inclusion_proof_request_signing_bytes(
    chain_id: &[u8; 32],
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaDescriptorInclusionProofRequest-v1",
        [
            chain_id.as_slice(),
            producer.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica descriptor-proof response carrier.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_descriptor_inclusion_proof_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    producer: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    proof: &DirectoryDescriptorInclusionProofV1,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let proof_digest = directory_descriptor_inclusion_proof_transport_digest(proof);
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaDescriptorInclusionProofResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            producer.as_slice(),
            carrier.as_slice(),
            response_timestamp.as_slice(),
            block_hash.as_slice(),
            descriptor_hash.as_slice(),
            proof_digest.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica-carrier block-range request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_block_range_request_signing_bytes(
    chain_id: &[u8; 32],
    producer: &[u8; 32],
    from_height: u64,
    limit: u16,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let from_height = from_height.to_le_bytes();
    let limit = limit.to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaBlockRangeRequest-v1",
        [
            chain_id.as_slice(),
            producer.as_slice(),
            from_height.as_slice(),
            limit.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by a replica-carrier block-range response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_replica_block_range_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    producer: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let block_count = u64::try_from(blocks.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let has_more = [u8::from(has_more)];
    let tip_height = tip_height.to_le_bytes();
    let block_hashes = blocks
        .iter()
        .map(DirectoryCommitmentBlockV1::hash)
        .collect::<Vec<_>>();
    let mut fields = Vec::<&[u8]>::with_capacity(block_hashes.len() + 10);
    fields.extend([
        chain_id.as_slice(),
        request_id.as_slice(),
        producer.as_slice(),
        carrier.as_slice(),
        response_timestamp.as_slice(),
        block_count.as_slice(),
    ]);
    fields.extend(block_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        has_more.as_slice(),
        tip_height.as_slice(),
        tip_hash.as_slice(),
    ]);
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ReplicaBlockRangeResponse-v1",
        fields,
    )
}

/// Canonical digest signed by a replica-carrier object request.
#[must_use]
pub fn directory_replica_descriptor_objects_request_signing_bytes(
    chain_id: &[u8; 32],
    producer: &[u8; 32],
    descriptor_hashes: &[[u8; 32]],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let request_timestamp = request_timestamp.to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 7);
    fields.extend([chain_id.as_slice(), producer.as_slice(), count.as_slice()]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    fields.extend([
        request_id.as_slice(),
        requester.as_slice(),
        request_timestamp.as_slice(),
    ]);
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ReplicaObjectsRequest-v1", fields)
}

/// Canonical digest signed by a replica-carrier object response.
#[must_use]
pub fn directory_replica_descriptor_objects_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    producer: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    descriptor_hashes: &[[u8; 32]],
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let count = u64::try_from(descriptor_hashes.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes();
    let mut fields = Vec::<&[u8]>::with_capacity(descriptor_hashes.len() + 7);
    fields.extend([
        chain_id.as_slice(),
        request_id.as_slice(),
        producer.as_slice(),
        carrier.as_slice(),
        response_timestamp.as_slice(),
        count.as_slice(),
    ]);
    fields.extend(descriptor_hashes.iter().map(<[u8; 32]>::as_slice));
    directory_sync_signing_digest(b"AeroNyx-DirectorySync-ReplicaObjectsResponse-v1", fields)
}

/// Canonical digest signed by an observation-checkpoint witness request.
#[must_use]
pub fn directory_observation_witness_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    checkpoint_hash: &[u8; 32],
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
            checkpoint_hash.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-checkpoint witness response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_witness_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    observer: &[u8; 32],
    checkpoint_sequence: u64,
    checkpoint_hash: &[u8; 32],
    responder: &[u8; 32],
    response_timestamp: u64,
    outcome: u8,
) -> [u8; 32] {
    let checkpoint_sequence = checkpoint_sequence.to_le_bytes();
    let response_timestamp = response_timestamp.to_le_bytes();
    let outcome = [outcome];
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            observer.as_slice(),
            checkpoint_sequence.as_slice(),
            checkpoint_hash.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            outcome.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-witness carrier request.
///
/// The digest binds the exact inner frame by both SHA-256 and byte length. The
/// carrier must independently recompute both before forwarding the frame.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_witness_carrier_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    witness: &[u8; 32],
    witness_request_sha256: &[u8; 32],
    witness_request_frame_bytes: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    let witness_request_frame_bytes = witness_request_frame_bytes.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessCarrierRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
            witness.as_slice(),
            witness_request_sha256.as_slice(),
            witness_request_frame_bytes.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-witness carrier response.
///
/// A carrier authenticates bounded transport only. The caller must recompute
/// both frame digests and verify the inner witness response independently.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_witness_carrier_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    witness: &[u8; 32],
    carrier: &[u8; 32],
    response_timestamp: u64,
    witness_request_sha256: &[u8; 32],
    witness_response_sha256: &[u8; 32],
    witness_response_frame_bytes: u64,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let witness_response_frame_bytes = witness_response_frame_bytes.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationWitnessCarrierResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            witness.as_slice(),
            carrier.as_slice(),
            response_timestamp.as_slice(),
            witness_request_sha256.as_slice(),
            witness_response_sha256.as_slice(),
            witness_response_frame_bytes.as_slice(),
        ],
    )
}

/// Canonical digest signed by an opaque witness-policy anchor request.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_policy_anchor_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    policy_epoch: u64,
    previous_policy_digest: &[u8; 32],
    policy_digest: &[u8; 32],
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    let policy_epoch = policy_epoch.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-PolicyAnchorRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
            policy_epoch.as_slice(),
            previous_policy_digest.as_slice(),
            policy_digest.as_slice(),
        ],
    )
}

/// Canonical digest signed by an opaque witness-policy anchor response.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_policy_anchor_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    observer: &[u8; 32],
    policy_epoch: u64,
    policy_digest: &[u8; 32],
    responder: &[u8; 32],
    response_timestamp: u64,
    outcome: u8,
) -> [u8; 32] {
    let policy_epoch = policy_epoch.to_le_bytes();
    let response_timestamp = response_timestamp.to_le_bytes();
    let outcome = [outcome];
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-PolicyAnchorResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            observer.as_slice(),
            policy_epoch.as_slice(),
            policy_digest.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            outcome.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-certificate request.
#[must_use]
pub fn directory_observation_certificate_request_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> [u8; 32] {
    let request_timestamp = request_timestamp.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationCertificateRequest-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            request_timestamp.as_slice(),
        ],
    )
}

/// Canonical digest signed by an observation-certificate response.
///
/// The digest binds both the SHA-256 digest and exact byte length. The caller
/// must independently recompute the digest from `certificate_frame` before
/// accepting the responder signature.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn directory_observation_certificate_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    responder: &[u8; 32],
    response_timestamp: u64,
    certificate_sha256: &[u8; 32],
    certificate_frame_bytes: u64,
) -> [u8; 32] {
    let response_timestamp = response_timestamp.to_le_bytes();
    let certificate_frame_bytes = certificate_frame_bytes.to_le_bytes();
    directory_sync_signing_digest(
        b"AeroNyx-DirectorySync-ObservationCertificateResponse-v1",
        [
            chain_id.as_slice(),
            request_id.as_slice(),
            requester.as_slice(),
            responder.as_slice(),
            response_timestamp.as_slice(),
            certificate_sha256.as_slice(),
            certificate_frame_bytes.as_slice(),
        ],
    )
}
