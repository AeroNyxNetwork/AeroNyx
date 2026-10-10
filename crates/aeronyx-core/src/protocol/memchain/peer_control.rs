// ============================================
// File: crates/aeronyx-core/src/protocol/memchain/peer_control.rs
// ============================================
//! # Node-peer control-frame contracts
//!
//! Owns the contracts shared by the node-peer control frames: the fixed-size
//! checkpoint-certificate member, the verified delivery-anchor witness outcome
//! codes, and the canonical signing bytes and digests for block-range,
//! checkpoint, checkpoint-certificate, coordinator lease/release, delivery-anchor
//! witness, custody-audit witness, and coordinator handover requests/responses.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `protocol/memchain.rs`; bodies unchanged.

use serde::{Deserialize, Serialize};

use crate::ledger::{RecordCommitmentBlockV1, RecordCoordinatorHandoverV1};

use super::serde_bytes64;
#[cfg(doc)]
use super::MemChainMessage;

/// Witness atomically advanced from an older generation to this request.
pub const VERIFIED_DELIVERY_WITNESS_ADVANCED_V1: u8 = 0;
/// Witness already retained the same generation and digest.
pub const VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1: u8 = 1;
/// Request generation was below the witness's durable high-water mark.
pub const VERIFIED_DELIVERY_WITNESS_STALE_V1: u8 = 2;
/// Request reused the witness generation with a different digest.
pub const VERIFIED_DELIVERY_WITNESS_CONFLICT_V1: u8 = 3;
/// Request skipped one or more generations after this witness was established.
///
/// A witness must not advance on this outcome. Requiring contiguous updates
/// prevents a later, still correctly signed host snapshot from silently
/// replacing an unwitnessed intermediate generation.
pub const VERIFIED_DELIVERY_WITNESS_GAP_V1: u8 = 4;

/// Fixed-size representation of one historical signed checkpoint response.
///
/// The certificate response supplies the common `chain_id`. Reconstructing a
/// [`MemChainMessage::RecordChainCheckpointResponseV1`] from these fields
/// yields the exact frame whose SHA-256 digest is committed by the certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordCheckpointCertificateMemberV1 {
    /// Original request identifier signed by the witness.
    pub request_id: [u8; 16],
    /// Witness Ed25519 public key.
    pub responder: [u8; 32],
    /// Original response timestamp signed by the witness.
    pub response_timestamp: u64,
    /// Certified shared-prefix height.
    pub checkpoint_height: u64,
    /// Witness block hash at `checkpoint_height`.
    pub checkpoint_hash: [u8; 32],
    /// Witness tip height when it signed the response.
    pub tip_height: u64,
    /// Witness tip hash when it signed the response.
    pub tip_hash: [u8; 32],
    /// Original witness signature over the v1 checkpoint response fields.
    #[serde(with = "serde_bytes64")]
    pub signature: [u8; 64],
}

/// Canonical bytes signed by `RecordBlockRangeRequestV1.requester`.
#[must_use]
pub fn record_block_range_request_signing_bytes(
    chain_id: &[u8; 32],
    from_height: u64,
    limit: u16,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(140);
    bytes.extend_from_slice(b"AeroNyx-RecordBlockRangeRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&from_height.to_le_bytes());
    bytes.extend_from_slice(&limit.to_le_bytes());
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordBlockRangeResponseV1.responder`.
#[must_use]
pub fn record_block_range_response_signing_bytes(
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    blocks: &[RecordCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Vec<u8> {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    for block in blocks {
        hasher.update(block.hash());
    }
    let block_hashes_digest: [u8; 32] = hasher.finalize().into();

    let mut bytes = Vec::with_capacity(170);
    bytes.extend_from_slice(b"AeroNyx-RecordBlockRangeResponse-v1");
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&block_hashes_digest);
    bytes.push(u8::from(has_more));
    bytes.extend_from_slice(&tip_height.to_le_bytes());
    bytes.extend_from_slice(tip_hash);
    bytes
}

/// Canonical bytes signed by `RecordChainCheckpointRequestV1.requester`.
#[must_use]
pub fn record_chain_checkpoint_request_signing_bytes(
    chain_id: &[u8; 32],
    known_tip_height: u64,
    known_tip_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(166);
    bytes.extend_from_slice(b"AeroNyx-RecordChainCheckpointRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&known_tip_height.to_le_bytes());
    bytes.extend_from_slice(known_tip_hash);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordChainCheckpointResponseV1.responder`.
#[must_use]
pub fn record_chain_checkpoint_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    checkpoint_height: u64,
    checkpoint_hash: &[u8; 32],
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(214);
    bytes.extend_from_slice(b"AeroNyx-RecordChainCheckpointResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&checkpoint_height.to_le_bytes());
    bytes.extend_from_slice(checkpoint_hash);
    bytes.extend_from_slice(&tip_height.to_le_bytes());
    bytes.extend_from_slice(tip_hash);
    bytes
}

/// Canonical bytes signed by
/// `RecordCheckpointCertificateRequestV1.requester`.
#[must_use]
pub fn record_checkpoint_certificate_request_signing_bytes(
    chain_id: &[u8; 32],
    known_tip_height: u64,
    known_tip_hash: &[u8; 32],
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(178);
    bytes.extend_from_slice(b"AeroNyx-RecordCheckpointCertificateRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&known_tip_height.to_le_bytes());
    bytes.extend_from_slice(known_tip_hash);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by
/// `RecordCheckpointCertificateResponseV1.responder`.
///
/// `certificate_digest` already commits the exact independently signed member
/// frames, so the transport signature binds that digest rather than repeating
/// historical witness material in the signing preimage.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_checkpoint_certificate_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    checkpoint_height: u64,
    checkpoint_hash: &[u8; 32],
    certificate_digest: &[u8; 32],
    required_signers: u8,
    signer_count: u8,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(240);
    bytes.extend_from_slice(b"AeroNyx-RecordCheckpointCertificateResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&checkpoint_height.to_le_bytes());
    bytes.extend_from_slice(checkpoint_hash);
    bytes.extend_from_slice(certificate_digest);
    bytes.push(required_signers);
    bytes.push(signer_count);
    bytes
}

/// Computes the canonical v1 checkpoint-certificate digest.
///
/// Each tuple is `(witness_identity, SHA-256(exact signed checkpoint frame))`.
/// Sorting makes the digest independent of network arrival order while still
/// requiring distinct identities at the verifier and storage layers.
#[must_use]
pub fn record_checkpoint_certificate_digest_v1(
    chain_id: &[u8; 32],
    checkpoint_height: u64,
    checkpoint_hash: &[u8; 32],
    required_signers: usize,
    members: &[([u8; 32], [u8; 32])],
) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut ordered = members.to_vec();
    ordered.sort_unstable_by_key(|member| member.0);
    let mut digest = Sha256::new();
    digest.update(b"AERONYX_RECORD_CHECKPOINT_CERTIFICATE_V1");
    digest.update(chain_id);
    digest.update(checkpoint_height.to_be_bytes());
    digest.update(checkpoint_hash);
    digest.update((required_signers as u64).to_be_bytes());
    digest.update((ordered.len() as u64).to_be_bytes());
    for (responder, evidence_digest) in ordered {
        digest.update(responder);
        digest.update(evidence_digest);
    }
    digest.finalize().into()
}

/// Canonical bytes signed by `RecordCoordinatorLeaseRequestV1.coordinator`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_coordinator_lease_request_signing_bytes(
    chain_id: &[u8; 32],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    known_tip_height: u64,
    known_tip_hash: &[u8; 32],
    requested_ttl_secs: u32,
    request_id: &[u8; 16],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(224);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(&known_tip_height.to_le_bytes());
    bytes.extend_from_slice(known_tip_hash);
    bytes.extend_from_slice(&requested_ttl_secs.to_le_bytes());
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorLeaseResponseV1.witness`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_coordinator_lease_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    witness: &[u8; 32],
    response_timestamp: u64,
    lease_epoch: u64,
    lease_expires_at: u64,
    witness_tip_height: u64,
    witness_tip_hash: &[u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(260);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&lease_epoch.to_le_bytes());
    bytes.extend_from_slice(&lease_expires_at.to_le_bytes());
    bytes.extend_from_slice(&witness_tip_height.to_le_bytes());
    bytes.extend_from_slice(witness_tip_hash);
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorLeaseReleaseRequestV1.coordinator`.
#[must_use]
pub fn record_coordinator_lease_release_request_signing_bytes(
    chain_id: &[u8; 32],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    request_id: &[u8; 16],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseReleaseRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorLeaseReleaseResponseV1.witness`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn record_coordinator_lease_release_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    coordinator: &[u8; 32],
    instance_id: &[u8; 32],
    witness: &[u8; 32],
    released_at: u64,
    lease_epoch: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(224);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorLeaseReleaseResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(coordinator);
    bytes.extend_from_slice(instance_id);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&released_at.to_le_bytes());
    bytes.extend_from_slice(&lease_epoch.to_le_bytes());
    bytes
}

/// Canonical bytes signed by
/// `VerifiedDeliveryAnchorWitnessRequestV1.requester`.
#[must_use]
pub fn verified_delivery_anchor_witness_request_signing_bytes(
    requester: &[u8; 32],
    generation: u64,
    anchor_digest: &[u8; 32],
    request_id: &[u8; 16],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(b"AeroNyx-VerifiedDeliveryAnchorWitnessRequest-v1");
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&generation.to_le_bytes());
    bytes.extend_from_slice(anchor_digest);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Canonical bytes signed by
/// `VerifiedDeliveryAnchorWitnessResponseV1.witness`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn verified_delivery_anchor_witness_response_signing_bytes(
    request_id: &[u8; 16],
    requester: &[u8; 32],
    requested_generation: u64,
    requested_anchor_digest: &[u8; 32],
    witness: &[u8; 32],
    response_timestamp: u64,
    witness_generation: u64,
    witness_anchor_digest: &[u8; 32],
    outcome: u8,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(b"AeroNyx-VerifiedDeliveryAnchorWitnessResponse-v1");
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&requested_generation.to_le_bytes());
    bytes.extend_from_slice(requested_anchor_digest);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(&witness_generation.to_le_bytes());
    bytes.extend_from_slice(witness_anchor_digest);
    bytes.push(outcome);
    bytes
}

/// Canonical bytes signed by `CustodyAuditAnchorWitnessRequestV1.requester`.
///
/// The caller must supply the SHA-256 returned by
/// `custody_audit_anchor_frame_sha256`; transport code must never hash an
/// alternate representation of the nested anchor.
#[must_use]
pub fn custody_audit_anchor_witness_request_signing_bytes(
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
    anchor_frame_sha256: &[u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(b"AeroNyx-CustodyAuditAnchorWitnessRequest-v1");
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes.extend_from_slice(anchor_frame_sha256);
    bytes
}

/// Canonical bytes signed by `CustodyAuditAnchorWitnessResponseV1.witness`.
///
/// The nested receipt remains independently verifiable; this second signature
/// binds its canonical digest to the exact request and response timestamp.
#[must_use]
pub fn custody_audit_anchor_witness_response_signing_bytes(
    request_id: &[u8; 16],
    requester: &[u8; 32],
    witness: &[u8; 32],
    response_timestamp: u64,
    receipt_frame_sha256: &[u8; 32],
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(192);
    bytes.extend_from_slice(b"AeroNyx-CustodyAuditAnchorWitnessResponse-v1");
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(witness);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.extend_from_slice(receipt_frame_sha256);
    bytes
}

/// Canonical bytes signed by `RecordCoordinatorHandoverRequestV1.requester`.
#[must_use]
pub fn record_coordinator_handover_request_signing_bytes(
    chain_id: &[u8; 32],
    after_authority_epoch: u64,
    request_id: &[u8; 16],
    requester: &[u8; 32],
    request_timestamp: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(144);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorHandoverRequest-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&after_authority_epoch.to_le_bytes());
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(requester);
    bytes.extend_from_slice(&request_timestamp.to_le_bytes());
    bytes
}

/// Computes the fixed digest used to bind one optional handover proof.
fn record_coordinator_handover_envelope_digest_v1(
    handover: Option<&RecordCoordinatorHandoverV1>,
) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let Some(handover) = handover else {
        return [0u8; 32];
    };
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-RecordCoordinatorHandoverEnvelope-v1");
    hasher.update(handover.header.signing_digest());
    hasher.update(handover.previous_signature);
    hasher.update(handover.next_signature);
    hasher.finalize().into()
}

/// Canonical bytes signed by `RecordCoordinatorHandoverResponseV1.responder`.
#[must_use]
pub fn record_coordinator_handover_response_signing_bytes(
    chain_id: &[u8; 32],
    request_id: &[u8; 16],
    responder: &[u8; 32],
    response_timestamp: u64,
    handover: Option<&RecordCoordinatorHandoverV1>,
    latest_authority_epoch: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(184);
    bytes.extend_from_slice(b"AeroNyx-RecordCoordinatorHandoverResponse-v1");
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(request_id);
    bytes.extend_from_slice(responder);
    bytes.extend_from_slice(&response_timestamp.to_le_bytes());
    bytes.push(u8::from(handover.is_some()));
    bytes.extend_from_slice(&record_coordinator_handover_envelope_digest_v1(handover));
    bytes.extend_from_slice(&latest_authority_epoch.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests;
