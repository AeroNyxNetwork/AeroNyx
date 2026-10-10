// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/codec.rs
// ============================================
//! # Persisted payload codecs
//!
//! Owns the canonical encode/decode helpers for checkpoints, blocks, and
//! descriptor objects, plus bounded materialization of size-admitted blobs.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    nonnegative_i64_to_u64, DirectoryCommitmentBlockV1, DirectoryObservationCheckpointV1,
    DirectoryReplicaStoreError, Options, PersistedReplicaBlobKind, SignedNodeDescriptor,
    MAX_DIRECTORY_BLOCK_BYTES, MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES,
    MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES,
};
#[cfg(test)]
use super::{notify_directory_replica_audit_test_observer, DirectoryReplicaAuditTestEvent};

pub(super) fn encode_observation_checkpoint(
    checkpoint: &DirectoryObservationCheckpointV1,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES)
        .serialize(checkpoint)
        .map_err(|error| {
            DirectoryReplicaStoreError::Codec(format!(
                "encode directory observation checkpoint: {error}"
            ))
        })
}

pub(super) fn decode_observation_checkpoint(
    bytes: &[u8],
) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
    if bytes.is_empty()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX)
            > MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES
    {
        return Err(DirectoryReplicaStoreError::Codec(
            "directory observation checkpoint size is invalid".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES)
        .deserialize(bytes)
        .map_err(|error| {
            DirectoryReplicaStoreError::Codec(format!(
                "decode directory observation checkpoint: {error}"
            ))
        })
}

pub(super) fn encode_block(
    block: &DirectoryCommitmentBlockV1,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .serialize(block)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

pub(super) fn decode_block(
    bytes: &[u8],
) -> Result<DirectoryCommitmentBlockV1, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_DIRECTORY_BLOCK_BYTES) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica block exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

pub(super) fn encode_descriptor_object(
    descriptor: &SignedNodeDescriptor,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .serialize(descriptor)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

pub(super) fn decode_descriptor_object(
    bytes: &[u8],
) -> Result<SignedNodeDescriptor, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| {
        length > MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES
    }) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica descriptor object exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

/// Converts a SQL-size-admitted payload without ever accepting a missing,
/// oversized, or length-inconsistent durable value.
///
/// The supplying query must project `length(blob)` and
/// `CASE WHEN length(blob) <= ? THEN blob END`. SQLite can answer `length()`
/// for a BLOB without first returning the BLOB bytes; the `CASE` keeps an
/// oversized payload out of `row.get::<Vec<u8>>()` entirely. The codec limits
/// above remain an independent second layer after this storage admission.
pub(super) fn materialize_admitted_replica_blob(
    stored_length: Option<i64>,
    admitted_blob: Option<Vec<u8>>,
    kind: PersistedReplicaBlobKind,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    let stored_length = stored_length
        .ok_or_else(|| DirectoryReplicaStoreError::Integrity(kind.missing_message().to_string()))?;
    let stored_length = nonnegative_i64_to_u64(stored_length, kind.length_field())?;
    if stored_length > kind.max_bytes() {
        return Err(DirectoryReplicaStoreError::Codec(
            kind.oversized_message().to_string(),
        ));
    }
    let admitted_blob = admitted_blob
        .ok_or_else(|| DirectoryReplicaStoreError::Integrity(kind.missing_message().to_string()))?;
    let materialized_length = u64::try_from(admitted_blob.len()).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!(
            "{} exceeds platform bounds",
            kind.length_field()
        ))
    })?;
    if materialized_length != stored_length {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{} changed during materialization",
            kind.length_field()
        )));
    }
    #[cfg(test)]
    notify_directory_replica_audit_test_observer(DirectoryReplicaAuditTestEvent::BlobMaterialized(
        kind,
    ));
    Ok(admitted_blob)
}
