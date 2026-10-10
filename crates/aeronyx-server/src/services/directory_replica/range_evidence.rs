// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/range_evidence.rs
// ============================================
//! # Range response evidence verification
//!
//! Owns verification of signed `BlockRangeResponseV1` evidence, page tip
//! contracts, exact descriptor-object checks, and the incident digest.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    decode_directory_sync_message, directory_block_range_response_signing_bytes,
    directory_replica_block_range_response_signing_bytes, encode_directory_sync_message, Digest,
    DirectoryCommitmentBlockV1, DirectoryDescriptorCommitmentV1, DirectoryRangeTipProvenance,
    DirectoryReplicaStoreError, DirectorySyncMessage, HashMap, HashSet, IdentityPublicKey,
    QuarantineIncident, Sha256, SignedNodeDescriptor, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    RESPONSE_TIMESTAMP_SKEW_SECS,
};

pub(super) fn verify_incident_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    verify_signed_range_response_evidence(frame, expected_producer).map(|_| ())
}

pub(super) struct VerifiedRangeResponseEvidence {
    tip_provenance: DirectoryRangeTipProvenance,
    response_timestamp: u64,
    blocks: Vec<DirectoryCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
    tip_hash: [u8; 32],
}

pub(super) fn verify_signed_range_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<VerifiedRangeResponseEvidence, DirectoryReplicaStoreError> {
    let message = decode_directory_sync_message(frame)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
    if encode_directory_sync_message(&message)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
        != frame
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence frame is not canonical".to_string(),
        ));
    }
    match message {
        DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id,
            request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID || responder != *expected_producer {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "range evidence belongs to another chain or producer".to_string(),
                ));
            }
            let signing_bytes = directory_block_range_response_signing_bytes(
                &request_id,
                &responder,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&responder)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "range evidence producer signature is invalid".to_string(),
                    )
                })?;
            Ok(VerifiedRangeResponseEvidence {
                tip_provenance: DirectoryRangeTipProvenance::ProducerSigned,
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
            })
        }
        DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id,
            request_id,
            producer,
            carrier,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                || producer != *expected_producer
                || carrier == [0u8; 32]
                || carrier == producer
                || blocks.iter().any(|block| block.header.producer != producer)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "carrier range evidence belongs to another chain or producer".to_string(),
                ));
            }
            let signing_bytes = directory_replica_block_range_response_signing_bytes(
                &chain_id,
                &request_id,
                &producer,
                &carrier,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&carrier)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "carrier range evidence signature is invalid".to_string(),
                    )
                })?;
            Ok(VerifiedRangeResponseEvidence {
                tip_provenance: DirectoryRangeTipProvenance::CarrierReported,
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
            })
        }
        _ => Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence is not a supported block-range response".to_string(),
        )),
    }
}

pub(super) fn verify_range_response_evidence(
    frame: &[u8],
    producer: &[u8; 32],
    expected_blocks: &[DirectoryCommitmentBlockV1],
    expected_tip_height: u64,
    expected_tip_hash: &[u8; 32],
    observed_at: u64,
) -> Result<VerifiedRangeResponseAdmission, DirectoryReplicaStoreError> {
    let verified = verify_signed_range_response_evidence(frame, producer)?;
    if verified.blocks != expected_blocks
        || verified.tip_height != expected_tip_height
        || verified.tip_hash != *expected_tip_hash
        || verified.response_timestamp.abs_diff(observed_at) > RESPONSE_TIMESTAMP_SKEW_SECS
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "signed range evidence does not match the import".to_string(),
        ));
    }
    Ok(VerifiedRangeResponseAdmission {
        has_more: verified.has_more,
        tip_provenance: verified.tip_provenance,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifiedRangeResponseAdmission {
    pub(super) has_more: bool,
    pub(super) tip_provenance: DirectoryRangeTipProvenance,
}

pub(super) fn validate_page_tip_contract(
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    if tip_height == 0 && *tip_hash != [0u8; 32] {
        return Err(DirectoryReplicaStoreError::Integrity(
            "empty advertised tip must use the zero hash".to_string(),
        ));
    }
    let Some(last) = blocks.last() else {
        if has_more {
            return Err(DirectoryReplicaStoreError::Integrity(
                "an empty response cannot advertise more pages".to_string(),
            ));
        }
        return Ok(());
    };
    if last.header.height > tip_height
        || (has_more && last.header.height >= tip_height)
        || (!has_more && (last.header.height != tip_height || last.hash() != *tip_hash))
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "range pagination fields contradict the signed tip".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn validate_exact_descriptor_objects<'a>(
    blocks: &[DirectoryCommitmentBlockV1],
    objects: &'a [SignedNodeDescriptor],
) -> Result<HashMap<[u8; 32], &'a SignedNodeDescriptor>, DirectoryReplicaStoreError> {
    let required = blocks
        .iter()
        .flat_map(|block| block.commitments.iter().map(|entry| entry.descriptor_hash))
        .collect::<Vec<_>>();
    let required_set = required.iter().copied().collect::<HashSet<_>>();
    if required_set.len() != required.len() || objects.len() != required.len() {
        return Err(DirectoryReplicaStoreError::Request(
            "descriptor objects must exactly cover unique page commitments".to_string(),
        ));
    }
    let mut mapped = HashMap::with_capacity(objects.len());
    for descriptor in objects {
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
        if !required_set.contains(&commitment.descriptor_hash)
            || mapped
                .insert(commitment.descriptor_hash, descriptor)
                .is_some()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "descriptor response contains an extra or duplicate object".to_string(),
            ));
        }
    }
    Ok(mapped)
}

pub(super) fn incident_digest(
    producer: &[u8; 32],
    subject_node_id: &[u8; 32],
    incident: &QuarantineIncident<'_>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaIncident-v1");
    hasher.update(producer);
    hasher.update(subject_node_id);
    hasher.update((incident.kind.len() as u64).to_le_bytes());
    hasher.update(incident.kind.as_bytes());
    hasher.update(incident.height.to_le_bytes());
    hasher.update(incident.local_hash);
    hasher.update(incident.remote_hash);
    hasher.update((incident.evidence_frame.len() as u64).to_le_bytes());
    hasher.update(incident.evidence_frame);
    hasher.finalize().into()
}
