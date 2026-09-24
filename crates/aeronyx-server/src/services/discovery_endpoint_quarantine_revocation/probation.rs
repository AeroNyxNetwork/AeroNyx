// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_quarantine_revocation/probation.rs
// ============================================================================
//! Durable, non-routeable probation for one exact endpoint-promotion candidate.
//!
//! This module shares the revocation registry's SQLite transaction domain so
//! the final mutable-readiness check and probation write are atomic. It has no
//! peer-store, routing, API, server, configuration, or networking authority.
// [PERMISSIONLESS-ENDPOINT-PROMOTION-PROBATION 2026-09-24 by Codex] Probation
// retains an anti-rollback sequence floor and never activates a route.

use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, SignedNodeDescriptor, MAX_SIGNED_NODE_DESCRIPTOR_BYTES,
};
use aeronyx_core::protocol::discovery_endpoint_attestation::canonical_attested_public_endpoint_socket_v1;
use aeronyx_core::protocol::discovery_endpoint_proof::canonical_public_endpoint_commitment;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use sha2::{Digest, Sha256};

use super::{
    as_i64, as_u64, promotion_readiness_is_current_tx, promotion_readiness_shape_is_valid_at,
    DiscoveryEndpointPromotionReadiness, DiscoveryEndpointQuarantineRevocationConfig,
    DiscoveryEndpointQuarantineRevocationError,
    SqliteDiscoveryEndpointQuarantineRevocationRegistry,
};
use crate::services::discovery_endpoint_promotion_material::VerifiedPromotionMaterial;

const PROMOTION_PROBATION_RECORD_DOMAIN: &[u8] = b"AeroNyx/EndpointPromotionProbationRecordV1\0";

/// Mutation outcomes reveal no node, endpoint, descriptor, or commitment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointPromotionProbationOutcome {
    Inserted,
    Existing,
    Replaced,
    Stale,
    Conflict,
    Expired,
    Revoked,
    AtCapacity,
}

/// Coarse failures reveal no private probation material or database path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DiscoveryEndpointPromotionProbationError {
    #[error("endpoint promotion probation rejected")]
    Rejected,
    #[error("endpoint promotion probation schema unsupported")]
    UnsupportedSchema,
    #[error("endpoint promotion probation state corrupt")]
    Corrupt,
    #[error("endpoint promotion probation unavailable")]
    Unavailable,
}

/// Aggregate-only snapshot. No candidate enumeration is exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointPromotionProbationSnapshot {
    pub(crate) retained_candidates: usize,
}

/// Canonical private row material. Deliberately has no `Debug` implementation.
struct PreparedProbation {
    node_id: [u8; 32],
    descriptor_sequence: u64,
    descriptor_hash: [u8; 32],
    descriptor_bytes: Vec<u8>,
    endpoint_commitment: [u8; 32],
    group_commitment: [u8; 32],
    readiness_commitment: [u8; 32],
    admission_commitment: [u8; 32],
    positive_commitment: [u8; 32],
    challenge_id: [u8; 32],
    policy_epoch: u64,
    readiness_evaluated_at: u64,
    readiness_valid_until: u64,
    material_resolved_at: u64,
    material_valid_until: u64,
    record_commitment: [u8; 32],
}

#[derive(Clone, Copy)]
struct ExistingProbation {
    descriptor_sequence: u64,
    descriptor_hash: [u8; 32],
    record_commitment: [u8; 32],
}

impl SqliteDiscoveryEndpointQuarantineRevocationRegistry {
    /// Atomically retains one exact verified candidate in the private probation
    /// tier after a final current F.7 check in the same write transaction.
    pub(crate) fn retain_promotion_probation_at(
        &self,
        material: &VerifiedPromotionMaterial,
        readiness: &DiscoveryEndpointPromotionReadiness,
        now: u64,
    ) -> Result<DiscoveryEndpointPromotionProbationOutcome, DiscoveryEndpointPromotionProbationError>
    {
        let prepared = PreparedProbation::from_verified(material, readiness)?;
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;

        if let Some(existing) = load_existing(&tx, &prepared.node_id)? {
            if existing.record_commitment == prepared.record_commitment {
                if existing.descriptor_sequence != prepared.descriptor_sequence
                    || existing.descriptor_hash != prepared.descriptor_hash
                {
                    return Err(DiscoveryEndpointPromotionProbationError::Corrupt);
                }
                return finish(tx, DiscoveryEndpointPromotionProbationOutcome::Existing);
            }
            if prepared.descriptor_sequence < existing.descriptor_sequence {
                return finish(tx, DiscoveryEndpointPromotionProbationOutcome::Stale);
            }
            if prepared.descriptor_sequence == existing.descriptor_sequence {
                return finish(tx, DiscoveryEndpointPromotionProbationOutcome::Conflict);
            }
        }

        if now == 0
            || prepared.material_resolved_at > now
            || prepared.material_valid_until < now
            || material.descriptor().verify_at(now).is_err()
        {
            return finish(tx, DiscoveryEndpointPromotionProbationOutcome::Expired);
        }
        if !promotion_readiness_is_current_tx(&tx, readiness, now).map_err(map_parent_error)? {
            return finish(tx, DiscoveryEndpointPromotionProbationOutcome::Revoked);
        }

        let retained = load_probation_count(&tx)?;
        let replacing = load_existing(&tx, &prepared.node_id)?.is_some();
        if !replacing && retained >= self.config.max_states {
            return finish(tx, DiscoveryEndpointPromotionProbationOutcome::AtCapacity);
        }

        let descriptor_sequence = as_i64(prepared.descriptor_sequence).map_err(map_parent_error)?;
        let policy_epoch = as_i64(prepared.policy_epoch).map_err(map_parent_error)?;
        let readiness_evaluated_at =
            as_i64(prepared.readiness_evaluated_at).map_err(map_parent_error)?;
        let readiness_valid_until =
            as_i64(prepared.readiness_valid_until).map_err(map_parent_error)?;
        let material_resolved_at =
            as_i64(prepared.material_resolved_at).map_err(map_parent_error)?;
        let material_valid_until =
            as_i64(prepared.material_valid_until).map_err(map_parent_error)?;
        let admitted_at = as_i64(now).map_err(map_parent_error)?;

        if replacing {
            let changed = tx
                .execute(
                    "UPDATE discovery_endpoint_promotion_probation_v1 SET
                       descriptor_sequence=?1,descriptor_hash=?2,descriptor_bytes=?3,
                       endpoint_commitment=?4,group_commitment=?5,readiness_commitment=?6,
                       admission_commitment=?7,positive_commitment=?8,challenge_id=?9,
                       policy_epoch=?10,readiness_evaluated_at=?11,readiness_valid_until=?12,
                       material_resolved_at=?13,material_valid_until=?14,
                       record_commitment=?15,admitted_at=?16 WHERE node_id=?17",
                    params![
                        descriptor_sequence,
                        &prepared.descriptor_hash[..],
                        &prepared.descriptor_bytes,
                        &prepared.endpoint_commitment[..],
                        &prepared.group_commitment[..],
                        &prepared.readiness_commitment[..],
                        &prepared.admission_commitment[..],
                        &prepared.positive_commitment[..],
                        &prepared.challenge_id[..],
                        policy_epoch,
                        readiness_evaluated_at,
                        readiness_valid_until,
                        material_resolved_at,
                        material_valid_until,
                        &prepared.record_commitment[..],
                        admitted_at,
                        &prepared.node_id[..],
                    ],
                )
                .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
            if changed != 1 {
                return Err(DiscoveryEndpointPromotionProbationError::Corrupt);
            }
            finish(tx, DiscoveryEndpointPromotionProbationOutcome::Replaced)
        } else {
            tx.execute(
                "INSERT INTO discovery_endpoint_promotion_probation_v1(
                   descriptor_sequence,descriptor_hash,descriptor_bytes,endpoint_commitment,
                   group_commitment,readiness_commitment,admission_commitment,
                   positive_commitment,challenge_id,policy_epoch,readiness_evaluated_at,
                   readiness_valid_until,material_resolved_at,material_valid_until,
                   record_commitment,admitted_at,node_id
                 ) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
                params![
                    descriptor_sequence,
                    &prepared.descriptor_hash[..],
                    &prepared.descriptor_bytes,
                    &prepared.endpoint_commitment[..],
                    &prepared.group_commitment[..],
                    &prepared.readiness_commitment[..],
                    &prepared.admission_commitment[..],
                    &prepared.positive_commitment[..],
                    &prepared.challenge_id[..],
                    policy_epoch,
                    readiness_evaluated_at,
                    readiness_valid_until,
                    material_resolved_at,
                    material_valid_until,
                    &prepared.record_commitment[..],
                    admitted_at,
                    &prepared.node_id[..],
                ],
            )
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
            update_probation_count(
                &tx,
                retained
                    .checked_add(1)
                    .ok_or(DiscoveryEndpointPromotionProbationError::Corrupt)?,
            )?;
            finish(tx, DiscoveryEndpointPromotionProbationOutcome::Inserted)
        }
    }

    /// Exact, private currentness check for later activation design. This does
    /// not return descriptor bytes and cannot enumerate probation candidates.
    pub(crate) fn contains_current_promotion_probation_at(
        &self,
        material: &VerifiedPromotionMaterial,
        readiness: &DiscoveryEndpointPromotionReadiness,
        now: u64,
    ) -> Result<bool, DiscoveryEndpointPromotionProbationError> {
        let prepared = PreparedProbation::from_verified(material, readiness)?;
        if now == 0
            || prepared.material_resolved_at > now
            || prepared.material_valid_until < now
            || material.descriptor().verify_at(now).is_err()
        {
            return Ok(false);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
        let exact = matches!(
            load_existing(&tx, &prepared.node_id)?,
            Some(existing)
                if existing.descriptor_sequence == prepared.descriptor_sequence
                    && existing.descriptor_hash == prepared.descriptor_hash
                    && existing.record_commitment == prepared.record_commitment
        );
        let current = exact
            && promotion_readiness_is_current_tx(&tx, readiness, now).map_err(map_parent_error)?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
        Ok(current)
    }

    pub(crate) fn promotion_probation_snapshot(
        &self,
    ) -> Result<DiscoveryEndpointPromotionProbationSnapshot, DiscoveryEndpointPromotionProbationError>
    {
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
        Ok(DiscoveryEndpointPromotionProbationSnapshot {
            retained_candidates: load_probation_count_connection(&connection)?,
        })
    }
}

impl PreparedProbation {
    fn from_verified(
        material: &VerifiedPromotionMaterial,
        readiness: &DiscoveryEndpointPromotionReadiness,
    ) -> Result<Self, DiscoveryEndpointPromotionProbationError> {
        let descriptor = material.descriptor();
        let descriptor_bytes = descriptor
            .encode_canonical()
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Rejected)?;
        if descriptor_bytes.is_empty()
            || descriptor_bytes.len() > MAX_SIGNED_NODE_DESCRIPTOR_BYTES
            || descriptor.verify_signature().is_err()
        {
            return Err(DiscoveryEndpointPromotionProbationError::Rejected);
        }
        let descriptor_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                .map_err(|_| DiscoveryEndpointPromotionProbationError::Rejected)?;
        let endpoint = descriptor
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(DiscoveryEndpointPromotionProbationError::Rejected)?;
        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] ADAT and
        // the direct dialer bind the signed HTTP(S) endpoint's canonical
        // public socket, not its URL text. Keep the durable record identical.
        let endpoint_socket = canonical_attested_public_endpoint_socket_v1(endpoint)
            .map_err(|_| DiscoveryEndpointPromotionProbationError::Rejected)?;
        let endpoint_commitment =
            canonical_public_endpoint_commitment(&endpoint_socket.to_string())
                .map_err(|_| DiscoveryEndpointPromotionProbationError::Rejected)?;
        if descriptor_commitment != material.descriptor_commitment()
            || endpoint_commitment != material.endpoint_commitment()
            || material.group_commitment() != readiness.group_commitment
            || material.readiness_commitment() != readiness.readiness_commitment
            || descriptor_commitment.sequence != readiness.descriptor_sequence
            || descriptor_commitment.sequence == 0
            || descriptor_commitment.node_id.iter().all(|byte| *byte == 0)
            || descriptor_commitment
                .descriptor_hash
                .iter()
                .all(|byte| *byte == 0)
            || endpoint_commitment.iter().all(|byte| *byte == 0)
            || !descriptor.descriptor.policy.public_discovery
            || material.resolved_at() == 0
            || material.valid_until() < material.resolved_at()
            || descriptor.descriptor.issued_at > material.resolved_at()
            || descriptor.descriptor.expires_at < material.resolved_at()
            || material.valid_until() != readiness.valid_until.min(descriptor.descriptor.expires_at)
            || !promotion_readiness_shape_is_valid_at(readiness, material.resolved_at())
        {
            return Err(DiscoveryEndpointPromotionProbationError::Rejected);
        }
        let mut prepared = Self {
            node_id: descriptor_commitment.node_id,
            descriptor_sequence: descriptor_commitment.sequence,
            descriptor_hash: descriptor_commitment.descriptor_hash,
            descriptor_bytes,
            endpoint_commitment,
            group_commitment: readiness.group_commitment,
            readiness_commitment: readiness.readiness_commitment,
            admission_commitment: readiness.admission_commitment,
            positive_commitment: readiness.positive_commitment,
            challenge_id: readiness.challenge_id,
            policy_epoch: readiness.policy_epoch,
            readiness_evaluated_at: readiness.evaluated_at,
            readiness_valid_until: readiness.valid_until,
            material_resolved_at: material.resolved_at(),
            material_valid_until: material.valid_until(),
            record_commitment: [0; 32],
        };
        prepared.record_commitment = probation_record_commitment(&prepared);
        Ok(prepared)
    }
}

pub(super) fn initialize_schema_tx(
    tx: &Transaction<'_>,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    tx.execute_batch(
        "CREATE TABLE discovery_endpoint_promotion_probation_meta_v1(
           singleton INTEGER PRIMARY KEY CHECK(singleton=1),
           retained_candidates INTEGER NOT NULL CHECK(retained_candidates>=0)
         );
         INSERT INTO discovery_endpoint_promotion_probation_meta_v1 VALUES(1,0);
         CREATE TABLE discovery_endpoint_promotion_probation_v1(
           node_id BLOB PRIMARY KEY CHECK(length(node_id)=32),
           descriptor_sequence INTEGER NOT NULL CHECK(descriptor_sequence>0),
           descriptor_hash BLOB NOT NULL CHECK(length(descriptor_hash)=32),
           descriptor_bytes BLOB NOT NULL CHECK(length(descriptor_bytes)>0 AND length(descriptor_bytes)<=16384),
           endpoint_commitment BLOB NOT NULL CHECK(length(endpoint_commitment)=32),
           group_commitment BLOB NOT NULL CHECK(length(group_commitment)=32),
           readiness_commitment BLOB NOT NULL CHECK(length(readiness_commitment)=32),
           admission_commitment BLOB NOT NULL CHECK(length(admission_commitment)=32),
           positive_commitment BLOB NOT NULL CHECK(length(positive_commitment)=32),
           challenge_id BLOB NOT NULL CHECK(length(challenge_id)=32),
           policy_epoch INTEGER NOT NULL CHECK(policy_epoch>0),
           readiness_evaluated_at INTEGER NOT NULL CHECK(readiness_evaluated_at>0),
           readiness_valid_until INTEGER NOT NULL CHECK(readiness_valid_until>=readiness_evaluated_at),
           material_resolved_at INTEGER NOT NULL CHECK(material_resolved_at>0),
           material_valid_until INTEGER NOT NULL CHECK(material_valid_until>=material_resolved_at),
           record_commitment BLOB NOT NULL UNIQUE CHECK(length(record_commitment)=32),
           admitted_at INTEGER NOT NULL CHECK(admitted_at>0)
         );",
    )
    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)
}

pub(super) fn startup_audit(
    connection: &Connection,
    config: &DiscoveryEndpointQuarantineRevocationConfig,
) -> Result<(), DiscoveryEndpointQuarantineRevocationError> {
    let retained = load_probation_count_connection_parent(connection)?;
    let actual: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_promotion_probation_v1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let actual =
        usize::try_from(actual).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    if retained != actual || retained > config.max_states {
        return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
    }

    let mut statement = connection
        .prepare(
            "SELECT node_id,descriptor_sequence,descriptor_hash,length(descriptor_bytes),
                    descriptor_bytes,endpoint_commitment,group_commitment,readiness_commitment,
                    admission_commitment,positive_commitment,challenge_id,policy_epoch,
                    readiness_evaluated_at,readiness_valid_until,material_resolved_at,
                    material_valid_until,record_commitment,admitted_at
             FROM discovery_endpoint_promotion_probation_v1",
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    let mut rows = statement
        .query([])
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Unavailable)?
    {
        let descriptor_length: i64 = row
            .get(3)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        if descriptor_length <= 0
            || usize::try_from(descriptor_length)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?
                > MAX_SIGNED_NODE_DESCRIPTOR_BYTES
        {
            return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
        }
        let node_id = array32_parent(
            row.get(0)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let descriptor_sequence = as_u64(
            row.get(1)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let descriptor_hash = array32_parent(
            row.get(2)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let descriptor_bytes: Vec<u8> = row
            .get(4)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        let descriptor = SignedNodeDescriptor::decode_canonical(&descriptor_bytes)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        let endpoint = descriptor
            .descriptor
            .public_endpoint
            .as_deref()
            .ok_or(DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        let endpoint_commitment = array32_parent(
            row.get(5)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let readiness = DiscoveryEndpointPromotionReadiness {
            readiness_commitment: array32_parent(
                row.get(7)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            admission_commitment: array32_parent(
                row.get(8)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            positive_commitment: array32_parent(
                row.get(9)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            challenge_id: array32_parent(
                row.get(10)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            group_commitment: array32_parent(
                row.get(6)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            descriptor_sequence,
            policy_epoch: as_u64(
                row.get(11)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            evaluated_at: as_u64(
                row.get(12)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
            valid_until: as_u64(
                row.get(13)
                    .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
            )?,
        };
        let material_resolved_at = as_u64(
            row.get(14)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let material_valid_until = as_u64(
            row.get(15)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let record_commitment = array32_parent(
            row.get(16)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let admitted_at = as_u64(
            row.get(17)
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?,
        )?;
        let prepared = PreparedProbation {
            node_id,
            descriptor_sequence,
            descriptor_hash,
            descriptor_bytes,
            endpoint_commitment,
            group_commitment: readiness.group_commitment,
            readiness_commitment: readiness.readiness_commitment,
            admission_commitment: readiness.admission_commitment,
            positive_commitment: readiness.positive_commitment,
            challenge_id: readiness.challenge_id,
            policy_epoch: readiness.policy_epoch,
            readiness_evaluated_at: readiness.evaluated_at,
            readiness_valid_until: readiness.valid_until,
            material_resolved_at,
            material_valid_until,
            record_commitment,
        };
        let endpoint_socket = canonical_attested_public_endpoint_socket_v1(endpoint)
            .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        let canonical_endpoint_commitment =
            canonical_public_endpoint_commitment(&endpoint_socket.to_string())
                .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
        if commitment.node_id != node_id
            || commitment.sequence != descriptor_sequence
            || commitment.descriptor_hash != descriptor_hash
            || descriptor.descriptor.issued_at > material_resolved_at
            || descriptor.descriptor.expires_at < material_resolved_at
            || !descriptor.descriptor.policy.public_discovery
            || canonical_endpoint_commitment != endpoint_commitment
            || !promotion_readiness_shape_is_valid_at(&readiness, material_resolved_at)
            || material_valid_until != readiness.valid_until.min(descriptor.descriptor.expires_at)
            || admitted_at < material_resolved_at
            || record_commitment != probation_record_commitment(&prepared)
        {
            return Err(DiscoveryEndpointQuarantineRevocationError::Corrupt);
        }
    }
    Ok(())
}

fn load_existing(
    tx: &Transaction<'_>,
    node_id: &[u8; 32],
) -> Result<Option<ExistingProbation>, DiscoveryEndpointPromotionProbationError> {
    let raw = tx
        .query_row(
            "SELECT descriptor_sequence,descriptor_hash,record_commitment
             FROM discovery_endpoint_promotion_probation_v1 WHERE node_id=?1",
            params![&node_id[..]],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .optional()
        .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
    raw.map(|(sequence, descriptor_hash, record_commitment)| {
        Ok(ExistingProbation {
            descriptor_sequence: u64::try_from(sequence)
                .map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)?,
            descriptor_hash: array32(descriptor_hash)?,
            record_commitment: array32(record_commitment)?,
        })
    })
    .transpose()
}

fn load_probation_count(
    tx: &Transaction<'_>,
) -> Result<usize, DiscoveryEndpointPromotionProbationError> {
    let count: i64 = tx
        .query_row(
            "SELECT retained_candidates FROM discovery_endpoint_promotion_probation_meta_v1
             WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)
}

fn load_probation_count_connection(
    connection: &Connection,
) -> Result<usize, DiscoveryEndpointPromotionProbationError> {
    let count: i64 = connection
        .query_row(
            "SELECT retained_candidates FROM discovery_endpoint_promotion_probation_meta_v1
             WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)
}

fn load_probation_count_connection_parent(
    connection: &Connection,
) -> Result<usize, DiscoveryEndpointQuarantineRevocationError> {
    let count: i64 = connection
        .query_row(
            "SELECT retained_candidates FROM discovery_endpoint_promotion_probation_meta_v1
             WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)
}

fn update_probation_count(
    tx: &Transaction<'_>,
    count: usize,
) -> Result<(), DiscoveryEndpointPromotionProbationError> {
    let changed = tx
        .execute(
            "UPDATE discovery_endpoint_promotion_probation_meta_v1
             SET retained_candidates=?1 WHERE singleton=1",
            params![i64::try_from(count)
                .map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)?],
        )
        .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
    if changed != 1 {
        return Err(DiscoveryEndpointPromotionProbationError::Corrupt);
    }
    Ok(())
}

fn probation_record_commitment(prepared: &PreparedProbation) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(PROMOTION_PROBATION_RECORD_DOMAIN);
    hash.update(prepared.node_id);
    hash.update(prepared.descriptor_sequence.to_be_bytes());
    hash.update(prepared.descriptor_hash);
    hash.update(prepared.endpoint_commitment);
    hash.update(prepared.group_commitment);
    hash.update(prepared.readiness_commitment);
    hash.update(prepared.admission_commitment);
    hash.update(prepared.positive_commitment);
    hash.update(prepared.challenge_id);
    hash.update(prepared.policy_epoch.to_be_bytes());
    hash.update(prepared.readiness_evaluated_at.to_be_bytes());
    hash.update(prepared.readiness_valid_until.to_be_bytes());
    hash.update(prepared.material_resolved_at.to_be_bytes());
    hash.update(prepared.material_valid_until.to_be_bytes());
    hash.finalize().into()
}

fn finish(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointPromotionProbationOutcome,
) -> Result<DiscoveryEndpointPromotionProbationOutcome, DiscoveryEndpointPromotionProbationError> {
    tx.commit()
        .map_err(|_| DiscoveryEndpointPromotionProbationError::Unavailable)?;
    Ok(outcome)
}

fn array32(value: Vec<u8>) -> Result<[u8; 32], DiscoveryEndpointPromotionProbationError> {
    value
        .try_into()
        .map_err(|_| DiscoveryEndpointPromotionProbationError::Corrupt)
}

fn array32_parent(value: Vec<u8>) -> Result<[u8; 32], DiscoveryEndpointQuarantineRevocationError> {
    value
        .try_into()
        .map_err(|_| DiscoveryEndpointQuarantineRevocationError::Corrupt)
}

const fn map_parent_error(
    error: DiscoveryEndpointQuarantineRevocationError,
) -> DiscoveryEndpointPromotionProbationError {
    match error {
        DiscoveryEndpointQuarantineRevocationError::Rejected => {
            DiscoveryEndpointPromotionProbationError::Rejected
        }
        DiscoveryEndpointQuarantineRevocationError::UnsupportedSchema => {
            DiscoveryEndpointPromotionProbationError::UnsupportedSchema
        }
        DiscoveryEndpointQuarantineRevocationError::Corrupt => {
            DiscoveryEndpointPromotionProbationError::Corrupt
        }
        DiscoveryEndpointQuarantineRevocationError::Unavailable => {
            DiscoveryEndpointPromotionProbationError::Unavailable
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        NodeCapability, NodeDescriptor, NodePolicy, NodeProtocolFeature,
    };
    use aeronyx_core::protocol::discovery_endpoint_attestation::{
        DiscoveryEndpointAttestationPurposeV1, DiscoveryEndpointEvidenceAttestationV1,
    };
    use aeronyx_core::protocol::discovery_endpoint_proof::{
        DiscoveryEndpointChallengeV1, DiscoveryEndpointProofV1,
    };
    use tempfile::TempDir;

    use crate::services::discovery_endpoint_attestation_inbox::{
        DiscoveryEndpointAttestationInboxConfig, DiscoveryEndpointAttestationRecordOutcome,
        SqliteDiscoveryEndpointAttestationInbox, VerifiedDiscoveryEndpointAttestationV1,
    };
    use crate::services::discovery_endpoint_eligibility::{
        evaluate_endpoint_candidate, DiscoveryEndpointEligibilityPolicy,
        DiscoveryEndpointStakePolicyMode,
    };
    use crate::services::discovery_endpoint_promotion_material::{
        DiscoveryEndpointPromotionFeatureRequirement, DiscoveryEndpointPromotionMaterialResolver,
    };
    use crate::services::discovery_endpoint_quarantine::{
        quarantine_admission_commitment, DiscoveryEndpointQuarantineConfig,
        SqliteDiscoveryEndpointQuarantineRegistry,
    };
    use crate::services::discovery_endpoint_quarantine_observation::{
        DiscoveryEndpointObservationDirection, DiscoveryEndpointQuarantineChallengeOutcome,
        DiscoveryEndpointQuarantineEvidenceRequest,
        DiscoveryEndpointQuarantineEvidenceVerificationError,
        DiscoveryEndpointQuarantineEvidenceVerifier, DiscoveryEndpointQuarantineObservationConfig,
        DiscoveryEndpointSatisfiedQuarantineEvidence,
        SqliteDiscoveryEndpointQuarantineObservationRegistry,
    };
    use crate::services::discovery_endpoint_quarantine_revocation::{
        DiscoveryEndpointNegativeEvidenceVerificationError,
        DiscoveryEndpointNegativeEvidenceVerifier, DiscoveryEndpointNegativeObservationRequest,
        DiscoveryEndpointQuarantineNegativeOutcome, DiscoveryEndpointQuarantinePositiveOutcome,
    };

    const NOW: u64 = 2_000_000_000;

    struct AcceptObservation;

    impl DiscoveryEndpointQuarantineEvidenceVerifier for AcceptObservation {
        fn verify(
            &self,
            _request: &DiscoveryEndpointQuarantineEvidenceRequest,
        ) -> Result<(), DiscoveryEndpointQuarantineEvidenceVerificationError> {
            Ok(())
        }
    }

    struct AcceptNegative;

    impl DiscoveryEndpointNegativeEvidenceVerifier for AcceptNegative {
        fn verify(
            &self,
            _request: &DiscoveryEndpointNegativeObservationRequest,
        ) -> Result<(), DiscoveryEndpointNegativeEvidenceVerificationError> {
            Ok(())
        }
    }

    struct Candidate {
        material: VerifiedPromotionMaterial,
        readiness: DiscoveryEndpointPromotionReadiness,
        evidence: DiscoveryEndpointSatisfiedQuarantineEvidence,
    }

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external test root");
        TempDir::new_in("target/test-temp").expect("tempdir")
    }

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed identity")
    }

    fn config(
        directory: &TempDir,
        name: &str,
        max_states: usize,
    ) -> DiscoveryEndpointQuarantineRevocationConfig {
        DiscoveryEndpointQuarantineRevocationConfig {
            db_path: directory.path().join(name),
            max_states,
            max_negative_per_state: 4,
            negative_ttl_secs: 60,
            cleanup_batch_size: 8,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn candidate(
        directory: &TempDir,
        label: &str,
        target: &IdentityKeyPair,
        sequence: u64,
        endpoint: &str,
        policy_epoch: u64,
        seed: u8,
        registry: &SqliteDiscoveryEndpointQuarantineRevocationRegistry,
    ) -> Candidate {
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            sequence,
            NOW - 60,
            NOW + 600,
            "1.0.0",
        )
        .with_protocol_features([NodeProtocolFeature::AnonymousMailboxV1]);
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.capabilities.push(NodeCapability::ChatRelay);
        descriptor.policy = NodePolicy {
            public_discovery: true,
            ..NodePolicy::default()
        };
        let descriptor = SignedNodeDescriptor::sign(descriptor, target).expect("descriptor");
        let descriptor_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .expect("descriptor commitment");
        let endpoint_socket =
            canonical_attested_public_endpoint_socket_v1(endpoint).expect("public socket");
        let endpoint_commitment =
            canonical_public_endpoint_commitment(&endpoint_socket.to_string())
                .expect("endpoint commitment");
        let inbox = SqliteDiscoveryEndpointAttestationInbox::open(
            DiscoveryEndpointAttestationInboxConfig {
                db_path: directory
                    .path()
                    .join(format!("{label}-attestation.sqlite3")),
                max_entries: 8,
                max_logical_bytes: 8 * 1024,
                retention_ttl_secs: 180,
                cleanup_batch_size: 8,
            },
        )
        .expect("attestation inbox");
        for offset in [0_u8, 1] {
            let observer = key(seed.wrapping_add(0x10).wrapping_add(offset));
            let context = [seed.wrapping_add(0x20).wrapping_add(offset); 32];
            let observed_at = NOW + u64::from(offset);
            let challenge = DiscoveryEndpointChallengeV1::issue(
                target.public_key_bytes(),
                descriptor_pin.descriptor_hash,
                endpoint_commitment,
                [seed.wrapping_add(0x30).wrapping_add(offset); 32],
                context,
                observed_at,
                NOW + 120,
                &observer,
            )
            .expect("challenge");
            let proof =
                DiscoveryEndpointProofV1::respond(&challenge, &context, observed_at, target)
                    .expect("proof");
            let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
                &descriptor,
                &challenge,
                &proof,
                context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
                observed_at,
                NOW + 120,
                &observer,
            )
            .expect("attestation");
            let verified = VerifiedDiscoveryEndpointAttestationV1::verify(
                &attestation.encode(),
                observed_at,
                context,
            )
            .expect("verified attestation");
            assert_eq!(
                inbox
                    .record_verified_at(&verified, observed_at)
                    .expect("retain attestation"),
                DiscoveryEndpointAttestationRecordOutcome::Inserted
            );
        }
        let facts = inbox
            .candidate_facts_at(NOW + 2, 60, 8)
            .expect("candidate facts")
            .into_iter()
            .next()
            .expect("candidate facts present");
        let admission = evaluate_endpoint_candidate(
            &facts,
            DiscoveryEndpointEligibilityPolicy::new(
                2,
                60,
                policy_epoch,
                DiscoveryEndpointStakePolicyMode::Disabled,
            )
            .expect("policy"),
            NOW + 2,
            None,
        )
        .into_quarantine_admission()
        .expect("admission");
        let admission_commitment = quarantine_admission_commitment(&admission);
        let quarantine =
            SqliteDiscoveryEndpointQuarantineRegistry::open(DiscoveryEndpointQuarantineConfig {
                db_path: directory.path().join(format!("{label}-quarantine.sqlite3")),
                max_entries: 4,
                cleanup_batch_size: 4,
            })
            .expect("quarantine");
        quarantine
            .record_at(admission, NOW + 2)
            .expect("record admission");
        let fresh = quarantine
            .fresh_admission_at(admission_commitment, NOW + 2)
            .expect("fresh lookup")
            .expect("fresh admission");
        let observations = SqliteDiscoveryEndpointQuarantineObservationRegistry::open(
            DiscoveryEndpointQuarantineObservationConfig {
                db_path: directory
                    .path()
                    .join(format!("{label}-observation.sqlite3")),
                max_challenges: 4,
                max_attempts_per_challenge: 4,
                challenge_ttl_secs: 60,
                minimum_observation_span_secs: 5,
                cleanup_batch_size: 4,
            },
        )
        .expect("observations");
        let challenge = match observations
            .begin_at(
                fresh,
                [seed.wrapping_add(0x40); 32],
                [seed.wrapping_add(0x41); 32],
                NOW + 2,
                NOW + 2,
            )
            .expect("begin")
        {
            DiscoveryEndpointQuarantineChallengeOutcome::Issued(value) => value,
            other => panic!("unexpected challenge outcome {other:?}"),
        };
        observations
            .observe_at(
                challenge,
                [seed.wrapping_add(0x42); 32],
                [seed.wrapping_add(0x43); 32],
                DiscoveryEndpointObservationDirection::OutboundChallenge,
                [seed.wrapping_add(0x44); 32],
                NOW + 3,
                NOW + 3,
                Some(&AcceptObservation),
            )
            .expect("outbound");
        observations
            .observe_at(
                challenge,
                [seed.wrapping_add(0x45); 32],
                [seed.wrapping_add(0x46); 32],
                DiscoveryEndpointObservationDirection::InboundProof,
                [seed.wrapping_add(0x47); 32],
                NOW + 8,
                NOW + 8,
                Some(&AcceptObservation),
            )
            .expect("inbound");
        let evidence = observations
            .satisfied_evidence_at(challenge, NOW + 8)
            .expect("evidence lookup")
            .expect("evidence");
        assert_eq!(
            registry
                .retain_positive_at(evidence, NOW + 8)
                .expect("retain positive"),
            DiscoveryEndpointQuarantinePositiveOutcome::Retained
        );
        let readiness = registry
            .promotion_readiness_at(fresh, evidence, NOW + 8)
            .expect("readiness")
            .expect("ready");
        let material = DiscoveryEndpointPromotionMaterialResolver::new(&inbox, registry)
            .resolve_at(
                &readiness,
                &descriptor,
                DiscoveryEndpointPromotionFeatureRequirement::new(
                    Some(NodeCapability::ChatRelay),
                    Some(NodeProtocolFeature::AnonymousMailboxV1),
                ),
                NOW + 8,
            )
            .expect("resolve")
            .expect("material");
        Candidate {
            material,
            readiness,
            evidence,
        }
    }

    #[test]
    fn exact_retry_conflict_stale_replacement_restart_and_revocation_are_deterministic() {
        let directory = tempdir();
        let cfg = config(&directory, "probation.sqlite3", 16);
        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg.clone()).expect("open");
        let target = key(0x21);
        let first = candidate(
            &directory,
            "first",
            &target,
            7,
            "8.8.8.8:51820",
            1,
            0x21,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&first.material, &first.readiness, NOW + 8)
                .expect("insert"),
            DiscoveryEndpointPromotionProbationOutcome::Inserted
        );
        assert!(registry
            .contains_current_promotion_probation_at(&first.material, &first.readiness, NOW + 8)
            .expect("current"));
        assert_eq!(
            registry
                .retain_promotion_probation_at(&first.material, &first.readiness, NOW + 500)
                .expect("expired exact retry"),
            DiscoveryEndpointPromotionProbationOutcome::Existing
        );

        let conflict = candidate(
            &directory,
            "conflict",
            &target,
            7,
            "8.8.4.4:51820",
            1,
            0x31,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&conflict.material, &conflict.readiness, NOW + 8)
                .expect("conflict"),
            DiscoveryEndpointPromotionProbationOutcome::Conflict
        );
        let stale = candidate(
            &directory,
            "stale",
            &target,
            6,
            "1.1.1.1:51820",
            1,
            0x41,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&stale.material, &stale.readiness, NOW + 8)
                .expect("stale"),
            DiscoveryEndpointPromotionProbationOutcome::Stale
        );
        let replacement = candidate(
            &directory,
            "replacement",
            &target,
            8,
            "9.9.9.9:51820",
            1,
            0x51,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(
                    &replacement.material,
                    &replacement.readiness,
                    NOW + 8,
                )
                .expect("replace"),
            DiscoveryEndpointPromotionProbationOutcome::Replaced
        );
        assert!(!registry
            .contains_current_promotion_probation_at(&first.material, &first.readiness, NOW + 8)
            .expect("old no longer current"));
        assert_eq!(
            registry.promotion_probation_snapshot().expect("snapshot"),
            DiscoveryEndpointPromotionProbationSnapshot {
                retained_candidates: 1,
            }
        );
        drop(registry);

        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg).expect("restart");
        assert_eq!(
            registry
                .retain_promotion_probation_at(
                    &replacement.material,
                    &replacement.readiness,
                    NOW + 8,
                )
                .expect("restart retry"),
            DiscoveryEndpointPromotionProbationOutcome::Existing
        );
        assert_eq!(
            registry
                .record_negative_at(
                    replacement.evidence,
                    [0x91; 32],
                    [0x92; 32],
                    NOW + 9,
                    NOW + 9,
                    Some(&AcceptNegative),
                )
                .expect("revoke after promotion"),
            DiscoveryEndpointQuarantineNegativeOutcome::Revoked
        );
        assert!(!registry
            .contains_current_promotion_probation_at(
                &replacement.material,
                &replacement.readiness,
                NOW + 9,
            )
            .expect("revoked is inert"));
        assert_eq!(
            registry
                .retain_promotion_probation_at(
                    &replacement.material,
                    &replacement.readiness,
                    NOW + 9,
                )
                .expect("revoked exact retry"),
            DiscoveryEndpointPromotionProbationOutcome::Existing
        );
    }

    #[test]
    fn signed_https_endpoint_probation_reopens_and_rejects_stored_socket_mismatch() {
        let directory = tempdir();
        let cfg = config(&directory, "https-probation.sqlite3", 4);
        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg.clone()).expect("open");
        let item = candidate(
            &directory,
            "https-candidate",
            &key(0x91),
            7,
            "https://8.8.8.8:51820",
            1,
            0x91,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&item.material, &item.readiness, NOW + 8)
                .expect("retain signed HTTPS candidate"),
            DiscoveryEndpointPromotionProbationOutcome::Inserted,
        );
        drop(registry);
        let reopened =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg.clone()).expect("audit");
        assert!(reopened
            .contains_current_promotion_probation_at(&item.material, &item.readiness, NOW + 8)
            .expect("reopened exact probation"));
        drop(reopened);

        // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] The
        // persisted commitment must still bind the literal signed host+port
        // after restart; a same-length substitution is corruption.
        let wrong_socket =
            canonical_public_endpoint_commitment("8.8.8.8:51821").expect("different public port");
        let connection = Connection::open(&cfg.db_path).expect("database");
        connection
            .execute(
                "UPDATE discovery_endpoint_promotion_probation_v1 SET endpoint_commitment=?1",
                params![&wrong_socket[..]],
            )
            .expect("tamper commitment");
        drop(connection);
        assert!(matches!(
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg),
            Err(DiscoveryEndpointQuarantineRevocationError::Corrupt)
        ));
    }

    #[test]
    fn revocation_before_mutation_and_epoch_capacity_preserve_the_sequence_floor() {
        let directory = tempdir();
        let cfg = config(&directory, "capacity.sqlite3", 1);
        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg).expect("open");
        let first = candidate(
            &directory,
            "epoch-one",
            &key(0x61),
            1,
            "8.8.8.8:51820",
            1,
            0x61,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&first.material, &first.readiness, NOW + 8)
                .expect("first"),
            DiscoveryEndpointPromotionProbationOutcome::Inserted
        );
        let second = candidate(
            &directory,
            "epoch-two",
            &key(0x62),
            1,
            "1.1.1.1:51820",
            2,
            0x71,
            &registry,
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&second.material, &second.readiness, NOW + 8)
                .expect("probation remains full"),
            DiscoveryEndpointPromotionProbationOutcome::AtCapacity
        );
        assert_eq!(
            registry.promotion_probation_snapshot().expect("snapshot"),
            DiscoveryEndpointPromotionProbationSnapshot {
                retained_candidates: 1,
            }
        );
        assert_eq!(
            registry
                .record_negative_at(
                    second.evidence,
                    [0x93; 32],
                    [0x94; 32],
                    NOW + 9,
                    NOW + 9,
                    Some(&AcceptNegative),
                )
                .expect("revoke before probation"),
            DiscoveryEndpointQuarantineNegativeOutcome::Revoked
        );
        assert_eq!(
            registry
                .retain_promotion_probation_at(&second.material, &second.readiness, NOW + 9)
                .expect("revoked before mutation"),
            DiscoveryEndpointPromotionProbationOutcome::Revoked
        );
        assert_eq!(
            registry.promotion_probation_snapshot().expect("unchanged"),
            DiscoveryEndpointPromotionProbationSnapshot {
                retained_candidates: 1,
            }
        );
    }

    #[test]
    fn schema_migration_corruption_privacy_and_no_routeability_fail_closed() {
        let directory = tempdir();
        let migration_path = directory.path().join("migration.sqlite3");
        let connection = Connection::open(&migration_path).expect("v1 database");
        connection
            .execute_batch(
                "CREATE TABLE discovery_endpoint_quarantine_revocation_meta_v1(
                   singleton INTEGER PRIMARY KEY CHECK(singleton=1),states INTEGER NOT NULL,
                   negatives INTEGER NOT NULL,current_policy_epoch INTEGER NOT NULL
                 );
                 INSERT INTO discovery_endpoint_quarantine_revocation_meta_v1 VALUES(1,0,0,0);
                 CREATE TABLE discovery_endpoint_quarantine_policy_state_v1(
                   admission_commitment BLOB PRIMARY KEY CHECK(length(admission_commitment)=32),
                   positive_commitment BLOB NOT NULL UNIQUE CHECK(length(positive_commitment)=32),
                   challenge_id BLOB NOT NULL CHECK(length(challenge_id)=32),
                   policy_epoch INTEGER NOT NULL,valid_until INTEGER NOT NULL,
                   state INTEGER NOT NULL CHECK(state IN (1,2)),revoked_at INTEGER
                 );
                 CREATE INDEX discovery_endpoint_quarantine_policy_expiry_v1
                   ON discovery_endpoint_quarantine_policy_state_v1(valid_until,admission_commitment);
                 CREATE TABLE discovery_endpoint_quarantine_negative_v1(
                   evidence_id BLOB PRIMARY KEY CHECK(length(evidence_id)=32),
                   admission_commitment BLOB NOT NULL CHECK(length(admission_commitment)=32),
                   negative_commitment BLOB NOT NULL UNIQUE CHECK(length(negative_commitment)=32),
                   observer_context BLOB NOT NULL CHECK(length(observer_context)=32),
                   policy_epoch INTEGER NOT NULL,observed_at INTEGER NOT NULL,expires_at INTEGER NOT NULL,
                   FOREIGN KEY(admission_commitment) REFERENCES discovery_endpoint_quarantine_policy_state_v1(admission_commitment) ON DELETE CASCADE
                 );
                 CREATE INDEX discovery_endpoint_quarantine_negative_expiry_v1
                   ON discovery_endpoint_quarantine_negative_v1(expires_at,evidence_id);
                 PRAGMA user_version=1;",
            )
            .expect("v1 schema");
        drop(connection);
        let migrated = SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(
            DiscoveryEndpointQuarantineRevocationConfig {
                db_path: migration_path.clone(),
                max_states: 4,
                max_negative_per_state: 4,
                negative_ttl_secs: 60,
                cleanup_batch_size: 4,
            },
        )
        .expect("migrate");
        assert_eq!(
            migrated.promotion_probation_snapshot().expect("empty"),
            DiscoveryEndpointPromotionProbationSnapshot {
                retained_candidates: 0,
            }
        );
        drop(migrated);
        assert_eq!(
            Connection::open(&migration_path)
                .expect("read version")
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
                .expect("version"),
            2
        );

        let cfg = config(&directory, "corrupt.sqlite3", 4);
        let registry =
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg.clone()).expect("open");
        let item = candidate(
            &directory,
            "corrupt-item",
            &key(0x81),
            1,
            "8.8.8.8:51820",
            1,
            0x81,
            &registry,
        );
        registry
            .retain_promotion_probation_at(&item.material, &item.readiness, NOW + 8)
            .expect("insert");
        let debug = format!(
            "{:?}{:?}{:?}",
            DiscoveryEndpointPromotionProbationOutcome::Conflict,
            DiscoveryEndpointPromotionProbationError::Rejected,
            registry.promotion_probation_snapshot().expect("snapshot")
        );
        for secret in [
            item.material.descriptor_commitment().node_id,
            item.material.descriptor_commitment().descriptor_hash,
            item.material.endpoint_commitment(),
            item.material.group_commitment(),
            item.material.readiness_commitment(),
        ] {
            assert!(!debug.contains(&hex::encode(secret)));
        }
        assert!(!debug.contains("corrupt.sqlite3"));
        drop(registry);

        let connection = Connection::open(&cfg.db_path).expect("database");
        connection
            .execute(
                "UPDATE discovery_endpoint_promotion_probation_v1
                 SET readiness_commitment=?1",
                params![&[0xA5_u8; 32][..]],
            )
            .expect("tamper");
        drop(connection);
        assert!(matches!(
            SqliteDiscoveryEndpointQuarantineRevocationRegistry::open(cfg),
            Err(DiscoveryEndpointQuarantineRevocationError::Corrupt)
        ));

        let source = include_str!("probation.rs");
        for forbidden in [
            concat!("Peer", "Store"),
            concat!("upsert_", "verified"),
            concat!("route_", "candidates"),
            concat!("services::", "routing"),
            concat!("crate::", "server"),
            concat!("crate::", "api"),
        ] {
            assert!(!source.contains(forbidden), "forbidden symbol: {forbidden}");
        }
    }
}
