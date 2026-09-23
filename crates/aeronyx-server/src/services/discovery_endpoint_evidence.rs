// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_evidence.rs
// ============================================================================
//! Durable, bounded evidence for successfully verified public endpoints.
//!
//! Evidence is deliberately separate from peer discovery and routing. A row
//! proves only that this verifier observed one valid endpoint proof at a
//! particular time; it grants no promotion, advertisement, ranking, or route.
// [PERMISSIONLESS-ENDPOINT-EVIDENCE 2026-09-24 by Codex] This repository is
// intentionally evidence-only and has no peer-store or networking dependency.

use std::fmt;
#[cfg(unix)]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, SignedNodeDescriptor};
use aeronyx_core::protocol::discovery_endpoint_proof::{
    DiscoveryEndpointAuthenticatedTransportV2, DiscoveryEndpointChallengeV1,
    DiscoveryEndpointProofV1, DiscoveryEndpointTransportOperationV1,
    DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V2,
};
use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use sha2::{Digest, Sha256};

use super::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use super::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};
use super::chat_relay_mailbox::{prepare_private_sqlite_target, verify_private_file};

const SCHEMA_VERSION: i64 = 1;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const MAX_ENTRIES: usize = 65_536;
const MAX_RETENTION_TTL_SECS: u64 = 7 * 24 * 60 * 60;
const MAX_CLEANUP_BATCH: usize = 4_096;
const MAX_CHALLENGE_FRAME_BYTES: usize = 512;
const MAX_PROOF_FRAME_BYTES: usize = 512;
const ADEA_HEADER_BYTES: usize = 8;
const EVIDENCE_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointEvidenceV1\0";
const CONTEXT_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointEvidenceContextV1\0";
const PROOF_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointEvidenceProofV1\0";

/// Bounded durable-store policy.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointEvidenceStoreConfig {
    /// Dedicated private SQLite path.
    pub db_path: PathBuf,
    /// Maximum retained evidence rows.
    pub max_entries: usize,
    /// Retention after local observation, in seconds.
    pub retention_ttl_secs: u64,
    /// Maximum expired rows removed by one admission or cleanup call.
    pub cleanup_batch_size: usize,
}

impl fmt::Debug for DiscoveryEndpointEvidenceStoreConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointEvidenceStoreConfig")
            .field("max_entries", &self.max_entries)
            .field("retention_ttl_secs", &self.retention_ttl_secs)
            .field("cleanup_batch_size", &self.cleanup_batch_size)
            .finish_non_exhaustive()
    }
}

/// Coarse repository failures with no ids, endpoints, paths, or frame bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiscoveryEndpointEvidenceError {
    /// Configuration or candidate evidence was rejected.
    #[error("endpoint evidence rejected")]
    Rejected,
    /// The database contains another or unsupported schema.
    #[error("endpoint evidence schema unsupported")]
    UnsupportedSchema,
    /// Durable rows violate the frozen semantic contract.
    #[error("endpoint evidence store corrupt")]
    Corrupt,
    /// Storage could not be opened or committed safely.
    #[error("endpoint evidence store unavailable")]
    Unavailable,
}

/// Exact replay and capacity outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryEndpointEvidenceRecordOutcome {
    /// New evidence was durably inserted.
    Inserted,
    /// The exact evidence already exists.
    Existing,
    /// The request id is bound to different evidence.
    Conflict,
    /// The configured row bound is full after bounded cleanup.
    AtCapacity,
}

/// Privacy-safe aggregate repository state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoveryEndpointEvidenceSnapshot {
    /// Number of retained rows.
    pub retained: usize,
    /// Earliest retained expiry, if any.
    pub earliest_expires_at: Option<u64>,
}

#[derive(Clone, PartialEq, Eq)]
struct VerifiedEvidence {
    request_id: [u8; 32],
    evidence_commitment: [u8; 32],
    target_node_id: [u8; 32],
    descriptor_commitment: [u8; 32],
    descriptor_sequence: u64,
    endpoint_commitment: [u8; 32],
    challenge_commitment: [u8; 32],
    context_commitment: [u8; 32],
    challenger_node_id: [u8; 32],
    verifier_node_id: [u8; 32],
    proof_commitment: [u8; 32],
    observed_at: u64,
    expires_at: u64,
}

/// Dedicated durable endpoint-evidence repository.
pub struct SqliteDiscoveryEndpointEvidenceStore {
    config: DiscoveryEndpointEvidenceStoreConfig,
    verifier_node_id: [u8; 32],
    expected_context: [u8; 32],
    connection: Mutex<Connection>,
    #[cfg(unix)]
    _database_parent: File,
}

impl fmt::Debug for SqliteDiscoveryEndpointEvidenceStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteDiscoveryEndpointEvidenceStore")
            .field("max_entries", &self.config.max_entries)
            .field("retention_ttl_secs", &self.config.retention_ttl_secs)
            .finish_non_exhaustive()
    }
}

impl SqliteDiscoveryEndpointEvidenceStore {
    /// Opens and semantically audits one dedicated private database.
    pub fn open(
        config: DiscoveryEndpointEvidenceStoreConfig,
        verifier_node_id: [u8; 32],
        expected_context: [u8; 32],
    ) -> Result<Self, DiscoveryEndpointEvidenceError> {
        validate_config(&config, &verifier_node_id, &expected_context)?;
        let target = prepare_private_sqlite_target(&config.db_path)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "endpoint_evidence_startup")
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        initialize_schema(&mut connection)?;
        startup_audit(&connection, &config, verifier_node_id, expected_context)?;
        Ok(Self {
            config,
            verifier_node_id,
            expected_context,
            connection: Mutex::new(connection),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    /// Records one exact verified V2 proof. Exact replay precedes freshness and quota.
    pub fn record_verified_at(
        &self,
        transport_frame: &[u8],
        challenge_frame: &[u8],
        proof_frame: &[u8],
        observed_at: u64,
    ) -> Result<DiscoveryEndpointEvidenceRecordOutcome, DiscoveryEndpointEvidenceError> {
        if observed_at == 0
            || challenge_frame.len() > MAX_CHALLENGE_FRAME_BYTES
            || proof_frame.len() > MAX_PROOF_FRAME_BYTES
        {
            return Err(DiscoveryEndpointEvidenceError::Rejected);
        }
        let transport = DiscoveryEndpointAuthenticatedTransportV2::decode(transport_frame)
            .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
        let exact_commitment = evidence_commitment(transport_frame, challenge_frame, proof_frame);
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        if let Some(existing) = load_commitment(&tx, &transport.request_id())? {
            return finish(
                tx,
                if existing == exact_commitment {
                    DiscoveryEndpointEvidenceRecordOutcome::Existing
                } else {
                    DiscoveryEndpointEvidenceRecordOutcome::Conflict
                },
            );
        }
        cleanup_tx(&tx, observed_at, self.config.cleanup_batch_size)?;
        let count: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM discovery_endpoint_evidence_v1",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        if usize::try_from(count).map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?
            >= self.config.max_entries
        {
            return finish(tx, DiscoveryEndpointEvidenceRecordOutcome::AtCapacity);
        }
        let evidence = verify_evidence(
            &transport,
            transport_frame,
            challenge_frame,
            proof_frame,
            observed_at,
            self.config.retention_ttl_secs,
            self.verifier_node_id,
            self.expected_context,
        )?;
        insert_evidence(
            &tx,
            &evidence,
            transport_frame,
            challenge_frame,
            proof_frame,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        Ok(DiscoveryEndpointEvidenceRecordOutcome::Inserted)
    }

    /// Removes at most `limit` expired rows in one immediate transaction.
    pub fn cleanup_expired_at(
        &self,
        now: u64,
        limit: usize,
    ) -> Result<usize, DiscoveryEndpointEvidenceError> {
        if limit == 0 || limit > MAX_CLEANUP_BATCH {
            return Err(DiscoveryEndpointEvidenceError::Rejected);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        let removed = cleanup_tx(&tx, now, limit)?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        Ok(removed)
    }

    /// Returns aggregate-only state.
    pub fn snapshot(
        &self,
    ) -> Result<DiscoveryEndpointEvidenceSnapshot, DiscoveryEndpointEvidenceError> {
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        let (count, earliest): (i64, Option<i64>) = connection
            .query_row(
                "SELECT COUNT(*), MIN(expires_at) FROM discovery_endpoint_evidence_v1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        Ok(DiscoveryEndpointEvidenceSnapshot {
            retained: usize::try_from(count)
                .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            earliest_expires_at: earliest
                .map(|v| u64::try_from(v).map_err(|_| DiscoveryEndpointEvidenceError::Corrupt))
                .transpose()?,
        })
    }
}

fn verify_evidence(
    transport: &DiscoveryEndpointAuthenticatedTransportV2,
    transport_frame: &[u8],
    challenge_frame: &[u8],
    proof_frame: &[u8],
    observed_at: u64,
    retention_ttl_secs: u64,
    verifier_node_id: [u8; 32],
    expected_context: [u8; 32],
) -> Result<VerifiedEvidence, DiscoveryEndpointEvidenceError> {
    transport
        .verify_at(
            observed_at,
            DiscoveryEndpointTransportOperationV1::Verify,
            &transport.target_node_id(),
            &expected_context,
        )
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    let descriptor = SignedNodeDescriptor::decode_canonical(transport.descriptor_bytes())
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    descriptor
        .verify_at(observed_at)
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    let descriptor_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    let challenge = DiscoveryEndpointChallengeV1::decode(challenge_frame)
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    challenge
        .verify_at(observed_at, &expected_context)
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    let proof = DiscoveryEndpointProofV1::decode(proof_frame)
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    proof
        .verify_for_challenge(&challenge, observed_at, &expected_context)
        .map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?;
    validate_inner_verify(
        transport.inner_frame(),
        transport.request_id(),
        challenge_frame,
        proof_frame,
    )?;
    if challenge.challenger_node_id() != verifier_node_id
        || challenge.target_node_id() != transport.target_node_id()
        || challenge.descriptor_commitment() != transport.descriptor_commitment()
        || challenge.endpoint_commitment() != transport.endpoint_commitment()
        || descriptor_pin.descriptor_hash != transport.descriptor_commitment()
        || descriptor_pin.sequence != descriptor.sequence()
    {
        return Err(DiscoveryEndpointEvidenceError::Rejected);
    }
    let expires_at = observed_at
        .checked_add(retention_ttl_secs)
        .ok_or(DiscoveryEndpointEvidenceError::Rejected)?;
    Ok(VerifiedEvidence {
        request_id: transport.request_id(),
        evidence_commitment: evidence_commitment(transport_frame, challenge_frame, proof_frame),
        target_node_id: transport.target_node_id(),
        descriptor_commitment: transport.descriptor_commitment(),
        descriptor_sequence: descriptor.sequence(),
        endpoint_commitment: transport.endpoint_commitment(),
        challenge_commitment: challenge.commitment(),
        context_commitment: domain_hash(CONTEXT_DOMAIN, &expected_context),
        challenger_node_id: challenge.challenger_node_id(),
        verifier_node_id,
        proof_commitment: domain_hash(PROOF_DOMAIN, proof_frame),
        observed_at,
        expires_at,
    })
}

fn validate_inner_verify(
    inner: &[u8],
    request_id: [u8; 32],
    challenge: &[u8],
    proof: &[u8],
) -> Result<(), DiscoveryEndpointEvidenceError> {
    if inner.len() < ADEA_HEADER_BYTES + 36
        || &inner[..4] != b"ADEA"
        || inner[4] != 1
        || inner[5] != DiscoveryEndpointTransportOperationV1::Verify as u8
        || usize::from(u16::from_be_bytes([inner[6], inner[7]])) + ADEA_HEADER_BYTES != inner.len()
        || inner[8..40] != request_id
    {
        return Err(DiscoveryEndpointEvidenceError::Rejected);
    }
    let challenge_len = usize::from(u16::from_be_bytes([inner[40], inner[41]]));
    let challenge_end = 42usize
        .checked_add(challenge_len)
        .ok_or(DiscoveryEndpointEvidenceError::Rejected)?;
    let proof_len_end = challenge_end
        .checked_add(2)
        .ok_or(DiscoveryEndpointEvidenceError::Rejected)?;
    let proof_len_bytes = inner
        .get(challenge_end..proof_len_end)
        .ok_or(DiscoveryEndpointEvidenceError::Rejected)?;
    let proof_len = usize::from(u16::from_be_bytes([proof_len_bytes[0], proof_len_bytes[1]]));
    let proof_end = proof_len_end
        .checked_add(proof_len)
        .ok_or(DiscoveryEndpointEvidenceError::Rejected)?;
    if inner.get(42..challenge_end) != Some(challenge)
        || inner.get(proof_len_end..proof_end) != Some(proof)
        || proof_end != inner.len()
    {
        return Err(DiscoveryEndpointEvidenceError::Rejected);
    }
    Ok(())
}

fn validate_config(
    config: &DiscoveryEndpointEvidenceStoreConfig,
    verifier: &[u8; 32],
    context: &[u8; 32],
) -> Result<(), DiscoveryEndpointEvidenceError> {
    if config.db_path.as_os_str().is_empty()
        || config.db_path == Path::new(":memory:")
        || config.max_entries == 0
        || config.max_entries > MAX_ENTRIES
        || config.retention_ttl_secs == 0
        || config.retention_ttl_secs > MAX_RETENTION_TTL_SECS
        || config.cleanup_batch_size == 0
        || config.cleanup_batch_size > MAX_CLEANUP_BATCH
        || verifier.iter().all(|b| *b == 0)
        || context.iter().all(|b| *b == 0)
    {
        return Err(DiscoveryEndpointEvidenceError::Rejected);
    }
    Ok(())
}

fn initialize_schema(connection: &mut Connection) -> Result<(), DiscoveryEndpointEvidenceError> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    let version: i64 = tx
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    if version == 0 {
        let foreign: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
        if foreign != 0 {
            return Err(DiscoveryEndpointEvidenceError::UnsupportedSchema);
        }
        tx.execute_batch(
            "CREATE TABLE discovery_endpoint_evidence_v1 (
                request_id BLOB PRIMARY KEY CHECK(length(request_id)=32), evidence_commitment BLOB NOT NULL CHECK(length(evidence_commitment)=32),
                operation INTEGER NOT NULL CHECK(operation=2), target_node_id BLOB NOT NULL CHECK(length(target_node_id)=32), descriptor_commitment BLOB NOT NULL CHECK(length(descriptor_commitment)=32),
                descriptor_sequence INTEGER NOT NULL, endpoint_commitment BLOB NOT NULL CHECK(length(endpoint_commitment)=32), challenge_commitment BLOB NOT NULL CHECK(length(challenge_commitment)=32),
                context_commitment BLOB NOT NULL CHECK(length(context_commitment)=32), challenger_node_id BLOB NOT NULL CHECK(length(challenger_node_id)=32), verifier_node_id BLOB NOT NULL CHECK(length(verifier_node_id)=32),
                proof_commitment BLOB NOT NULL CHECK(length(proof_commitment)=32), observed_at INTEGER NOT NULL, expires_at INTEGER NOT NULL,
                transport_frame BLOB NOT NULL, challenge_frame BLOB NOT NULL, proof_frame BLOB NOT NULL
             );
             CREATE INDEX discovery_endpoint_evidence_expiry_v1 ON discovery_endpoint_evidence_v1(expires_at, request_id);
             PRAGMA user_version=1;"
        ).map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    } else if version != SCHEMA_VERSION {
        return Err(DiscoveryEndpointEvidenceError::UnsupportedSchema);
    }
    tx.commit()
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)
}

fn startup_audit(
    connection: &Connection,
    config: &DiscoveryEndpointEvidenceStoreConfig,
    verifier: [u8; 32],
    context: [u8; 32],
) -> Result<(), DiscoveryEndpointEvidenceError> {
    let count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_evidence_v1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    if usize::try_from(count).map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?
        > config.max_entries
    {
        return Err(DiscoveryEndpointEvidenceError::Corrupt);
    }
    let invalid_operations: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_evidence_v1 WHERE operation != 2",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    if invalid_operations != 0 {
        return Err(DiscoveryEndpointEvidenceError::Corrupt);
    }
    let mut statement = connection.prepare("SELECT request_id,evidence_commitment,target_node_id,descriptor_commitment,descriptor_sequence,endpoint_commitment,challenge_commitment,context_commitment,challenger_node_id,verifier_node_id,proof_commitment,observed_at,expires_at,LENGTH(transport_frame),LENGTH(challenge_frame),LENGTH(proof_frame),transport_frame,challenge_frame,proof_frame FROM discovery_endpoint_evidence_v1")
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    let mut rows = statement
        .query([])
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?
    {
        let transport_len: i64 = row
            .get(13)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let challenge_len: i64 = row
            .get(14)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let proof_len: i64 = row
            .get(15)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        if transport_len <= 0
            || usize::try_from(transport_len)
                .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?
                > DISCOVERY_ENDPOINT_TRANSPORT_MAX_FRAME_BYTES_V2
            || challenge_len <= 0
            || usize::try_from(challenge_len)
                .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?
                > MAX_CHALLENGE_FRAME_BYTES
            || proof_len <= 0
            || usize::try_from(proof_len).map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?
                > MAX_PROOF_FRAME_BYTES
        {
            return Err(DiscoveryEndpointEvidenceError::Corrupt);
        }
        let transport: Vec<u8> = row
            .get(16)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let challenge: Vec<u8> = row
            .get(17)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let proof: Vec<u8> = row
            .get(18)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let observed_at = as_u64(
            row.get(11)
                .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
        )?;
        let decoded = DiscoveryEndpointAuthenticatedTransportV2::decode(&transport)
            .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let evidence = verify_evidence(
            &decoded,
            &transport,
            &challenge,
            &proof,
            observed_at,
            config.retention_ttl_secs,
            verifier,
            context,
        )
        .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?;
        let stored = VerifiedEvidence {
            request_id: fixed(
                row.get::<_, Vec<u8>>(0)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            evidence_commitment: fixed(
                row.get::<_, Vec<u8>>(1)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            target_node_id: fixed(
                row.get::<_, Vec<u8>>(2)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            descriptor_commitment: fixed(
                row.get::<_, Vec<u8>>(3)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            descriptor_sequence: as_u64(
                row.get(4)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            endpoint_commitment: fixed(
                row.get::<_, Vec<u8>>(5)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            challenge_commitment: fixed(
                row.get::<_, Vec<u8>>(6)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            context_commitment: fixed(
                row.get::<_, Vec<u8>>(7)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            challenger_node_id: fixed(
                row.get::<_, Vec<u8>>(8)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            verifier_node_id: fixed(
                row.get::<_, Vec<u8>>(9)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            proof_commitment: fixed(
                row.get::<_, Vec<u8>>(10)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
            observed_at,
            expires_at: as_u64(
                row.get(12)
                    .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)?,
            )?,
        };
        if stored != evidence {
            return Err(DiscoveryEndpointEvidenceError::Corrupt);
        }
    }
    Ok(())
}

fn insert_evidence(
    tx: &Transaction<'_>,
    e: &VerifiedEvidence,
    transport: &[u8],
    challenge: &[u8],
    proof: &[u8],
) -> Result<(), DiscoveryEndpointEvidenceError> {
    tx.execute("INSERT INTO discovery_endpoint_evidence_v1 VALUES (?1,?2,2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)", params![
        &e.request_id[..], &e.evidence_commitment[..], &e.target_node_id[..], &e.descriptor_commitment[..], as_i64(e.descriptor_sequence)?, &e.endpoint_commitment[..],
        &e.challenge_commitment[..], &e.context_commitment[..], &e.challenger_node_id[..], &e.verifier_node_id[..], &e.proof_commitment[..], as_i64(e.observed_at)?, as_i64(e.expires_at)?, transport, challenge, proof
    ]).map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    Ok(())
}

fn load_commitment(
    tx: &Transaction<'_>,
    id: &[u8; 32],
) -> Result<Option<[u8; 32]>, DiscoveryEndpointEvidenceError> {
    tx.query_row(
        "SELECT evidence_commitment FROM discovery_endpoint_evidence_v1 WHERE request_id=?1",
        params![&id[..]],
        |row| row.get::<_, Vec<u8>>(0),
    )
    .optional()
    .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?
    .map(fixed)
    .transpose()
}

fn cleanup_tx(
    tx: &Transaction<'_>,
    now: u64,
    limit: usize,
) -> Result<usize, DiscoveryEndpointEvidenceError> {
    tx.execute("DELETE FROM discovery_endpoint_evidence_v1 WHERE request_id IN (SELECT request_id FROM discovery_endpoint_evidence_v1 WHERE expires_at < ?1 ORDER BY expires_at,request_id LIMIT ?2)", params![as_i64(now)?, i64::try_from(limit).map_err(|_| DiscoveryEndpointEvidenceError::Rejected)?])
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)
}

fn finish(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointEvidenceRecordOutcome,
) -> Result<DiscoveryEndpointEvidenceRecordOutcome, DiscoveryEndpointEvidenceError> {
    tx.commit()
        .map_err(|_| DiscoveryEndpointEvidenceError::Unavailable)?;
    Ok(outcome)
}

fn evidence_commitment(transport: &[u8], challenge: &[u8], proof: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(EVIDENCE_DOMAIN);
    for bytes in [transport, challenge, proof] {
        h.update((bytes.len() as u64).to_be_bytes());
        h.update(bytes);
    }
    h.finalize().into()
}
fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    h.finalize().into()
}
fn fixed<const N: usize>(v: Vec<u8>) -> Result<[u8; N], DiscoveryEndpointEvidenceError> {
    v.try_into()
        .map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)
}
fn as_i64(v: u64) -> Result<i64, DiscoveryEndpointEvidenceError> {
    i64::try_from(v).map_err(|_| DiscoveryEndpointEvidenceError::Rejected)
}
fn as_u64(v: i64) -> Result<u64, DiscoveryEndpointEvidenceError> {
    u64::try_from(v).map_err(|_| DiscoveryEndpointEvidenceError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::NodeDescriptor;
    use aeronyx_core::protocol::discovery_endpoint_proof::DiscoveryEndpointAuthenticatedTransportV1;
    use tempfile::TempDir;

    const NOW: u64 = 2_000_000_000;
    const ENDPOINT: &str = "8.8.8.8:51820";

    struct Fixture {
        transport: Vec<u8>,
        challenge: Vec<u8>,
        proof: Vec<u8>,
        verifier: [u8; 32],
        context: [u8; 32],
    }

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed key")
    }

    fn fixture(request_id: [u8; 32], nonce: [u8; 32]) -> Fixture {
        let verifier = key(3);
        let target = key(9);
        let context = [0x44; 32];
        let mut descriptor =
            NodeDescriptor::new(target.public_key_bytes(), 7, NOW - 10, NOW + 600, "1.0.0");
        descriptor.public_endpoint = Some(ENDPOINT.to_string());
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let pin =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).expect("pin");
        let endpoint_commitment =
            aeronyx_core::protocol::canonical_public_endpoint_commitment(ENDPOINT)
                .expect("endpoint");
        let challenge = DiscoveryEndpointChallengeV1::issue(
            target.public_key_bytes(),
            pin.descriptor_hash,
            endpoint_commitment,
            nonce,
            context,
            NOW,
            NOW + 60,
            &verifier,
        )
        .expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &context, NOW + 1, &target)
            .expect("proof");
        let challenge_frame = challenge.encode();
        let proof_frame = proof.encode();
        let inner = verify_inner(request_id, &challenge_frame, &proof_frame);
        let transport = DiscoveryEndpointAuthenticatedTransportV2::sign(
            DiscoveryEndpointTransportOperationV1::Verify,
            request_id,
            context,
            NOW,
            NOW + 60,
            &inner,
            &descriptor,
            &target,
        )
        .expect("transport")
        .encode();
        Fixture {
            transport,
            challenge: challenge_frame,
            proof: proof_frame,
            verifier: verifier.public_key_bytes(),
            context,
        }
    }

    fn verify_inner(request_id: [u8; 32], challenge: &[u8], proof: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&request_id);
        body.extend_from_slice(
            &u16::try_from(challenge.len())
                .expect("challenge length")
                .to_be_bytes(),
        );
        body.extend_from_slice(challenge);
        body.extend_from_slice(
            &u16::try_from(proof.len())
                .expect("proof length")
                .to_be_bytes(),
        );
        body.extend_from_slice(proof);
        let mut frame = b"ADEA".to_vec();
        frame.push(1);
        frame.push(DiscoveryEndpointTransportOperationV1::Verify as u8);
        frame.extend_from_slice(
            &u16::try_from(body.len())
                .expect("body length")
                .to_be_bytes(),
        );
        frame.extend_from_slice(&body);
        frame
    }

    fn config(dir: &TempDir, max_entries: usize) -> DiscoveryEndpointEvidenceStoreConfig {
        DiscoveryEndpointEvidenceStoreConfig {
            db_path: dir.path().join("evidence.sqlite3"),
            max_entries,
            retention_ttl_secs: 120,
            cleanup_batch_size: 8,
        }
    }

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external-disk test temp root");
        TempDir::new_in("target/test-temp").expect("external-disk tempdir")
    }

    #[test]
    fn durable_insert_exact_replay_conflict_and_restart_audit() {
        let dir = tempdir();
        let first = fixture([0x11; 32], [0x21; 32]);
        let store = SqliteDiscoveryEndpointEvidenceStore::open(
            config(&dir, 4),
            first.verifier,
            first.context,
        )
        .expect("open");
        assert_eq!(
            store
                .record_verified_at(&first.transport, &first.challenge, &first.proof, NOW + 2)
                .expect("insert"),
            DiscoveryEndpointEvidenceRecordOutcome::Inserted
        );
        assert_eq!(
            store
                .record_verified_at(&first.transport, &first.challenge, &first.proof, NOW + 1000)
                .expect("replay before freshness"),
            DiscoveryEndpointEvidenceRecordOutcome::Existing
        );
        let conflict = fixture([0x11; 32], [0x22; 32]);
        assert_eq!(
            store
                .record_verified_at(
                    &conflict.transport,
                    &conflict.challenge,
                    &conflict.proof,
                    NOW + 2
                )
                .expect("conflict"),
            DiscoveryEndpointEvidenceRecordOutcome::Conflict
        );
        assert_eq!(store.snapshot().expect("snapshot").retained, 1);
        drop(store);
        let reopened = SqliteDiscoveryEndpointEvidenceStore::open(
            config(&dir, 4),
            first.verifier,
            first.context,
        )
        .expect("reopen");
        assert_eq!(reopened.snapshot().expect("snapshot").retained, 1);
    }

    #[test]
    fn tamper_capacity_and_bounded_cleanup_fail_closed() {
        let dir = tempdir();
        let first = fixture([0x31; 32], [0x41; 32]);
        let store = SqliteDiscoveryEndpointEvidenceStore::open(
            config(&dir, 1),
            first.verifier,
            first.context,
        )
        .expect("open");
        let mut tampered = first.proof.clone();
        *tampered.last_mut().expect("signature") ^= 1;
        assert_eq!(
            store.record_verified_at(&first.transport, &first.challenge, &tampered, NOW + 2),
            Err(DiscoveryEndpointEvidenceError::Rejected)
        );
        assert_eq!(store.snapshot().expect("snapshot").retained, 0);
        assert_eq!(
            store
                .record_verified_at(&first.transport, &first.challenge, &first.proof, NOW + 2)
                .expect("insert"),
            DiscoveryEndpointEvidenceRecordOutcome::Inserted
        );
        let second = fixture([0x32; 32], [0x42; 32]);
        assert_eq!(
            store
                .record_verified_at(&second.transport, &second.challenge, &second.proof, NOW + 3)
                .expect("capacity"),
            DiscoveryEndpointEvidenceRecordOutcome::AtCapacity
        );
        assert_eq!(store.cleanup_expired_at(NOW + 123, 1).expect("cleanup"), 1);
        assert_eq!(
            store
                .record_verified_at(&second.transport, &second.challenge, &second.proof, NOW + 3)
                .expect("insert after cleanup"),
            DiscoveryEndpointEvidenceRecordOutcome::Inserted
        );
    }

    #[test]
    fn foreign_schema_and_corrupt_row_reject_without_repair() {
        let dir = tempdir();
        let path = dir.path().join("evidence.sqlite3");
        Connection::open(&path)
            .expect("foreign open")
            .execute("CREATE TABLE unrelated(value INTEGER)", [])
            .expect("foreign table");
        let fixture = fixture([0x51; 32], [0x61; 32]);
        assert!(matches!(
            SqliteDiscoveryEndpointEvidenceStore::open(
                config(&dir, 4),
                fixture.verifier,
                fixture.context
            ),
            Err(DiscoveryEndpointEvidenceError::UnsupportedSchema)
        ));
        let objects: i64 = Connection::open(&path)
            .expect("inspect")
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name='discovery_endpoint_evidence_v1'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(objects, 0);

        let clean = tempdir();
        let store = SqliteDiscoveryEndpointEvidenceStore::open(
            config(&clean, 4),
            fixture.verifier,
            fixture.context,
        )
        .expect("open");
        store
            .record_verified_at(
                &fixture.transport,
                &fixture.challenge,
                &fixture.proof,
                NOW + 2,
            )
            .expect("insert");
        drop(store);
        Connection::open(clean.path().join("evidence.sqlite3"))
            .expect("corrupt open")
            .execute(
                "UPDATE discovery_endpoint_evidence_v1 SET descriptor_sequence=99",
                [],
            )
            .expect("corrupt");
        assert!(matches!(
            SqliteDiscoveryEndpointEvidenceStore::open(
                config(&clean, 4),
                fixture.verifier,
                fixture.context
            ),
            Err(DiscoveryEndpointEvidenceError::Corrupt)
        ));
    }

    #[test]
    fn v1_trailing_wrong_context_and_invalid_config_reject_without_rows() {
        let dir = tempdir();
        let fixture = fixture([0x71; 32], [0x72; 32]);
        let store = SqliteDiscoveryEndpointEvidenceStore::open(
            config(&dir, 4),
            fixture.verifier,
            fixture.context,
        )
        .expect("open");
        let mut trailing = fixture.transport.clone();
        trailing.push(0);
        assert_eq!(
            store.record_verified_at(&trailing, &fixture.challenge, &fixture.proof, NOW + 2),
            Err(DiscoveryEndpointEvidenceError::Rejected)
        );

        let decoded =
            DiscoveryEndpointAuthenticatedTransportV2::decode(&fixture.transport).expect("v2");
        let target = key(9);
        let v1 = DiscoveryEndpointAuthenticatedTransportV1::sign(
            DiscoveryEndpointTransportOperationV1::Verify,
            decoded.request_id(),
            decoded.descriptor_commitment(),
            decoded.endpoint_commitment(),
            fixture.context,
            NOW,
            NOW + 60,
            decoded.inner_frame(),
            &target,
        )
        .expect("v1")
        .encode();
        assert_eq!(
            store.record_verified_at(&v1, &fixture.challenge, &fixture.proof, NOW + 2),
            Err(DiscoveryEndpointEvidenceError::Rejected)
        );

        let wrong_context = [0x45; 32];
        let wrong_store = SqliteDiscoveryEndpointEvidenceStore::open(
            DiscoveryEndpointEvidenceStoreConfig {
                db_path: dir.path().join("wrong.sqlite3"),
                ..config(&dir, 4)
            },
            fixture.verifier,
            wrong_context,
        )
        .expect("wrong-context store");
        assert_eq!(
            wrong_store.record_verified_at(
                &fixture.transport,
                &fixture.challenge,
                &fixture.proof,
                NOW + 2
            ),
            Err(DiscoveryEndpointEvidenceError::Rejected)
        );
        assert_eq!(store.snapshot().expect("snapshot").retained, 0);

        let invalid = DiscoveryEndpointEvidenceStoreConfig {
            max_entries: 0,
            db_path: dir.path().join("invalid.sqlite3"),
            ..config(&dir, 4)
        };
        assert!(matches!(
            SqliteDiscoveryEndpointEvidenceStore::open(invalid, fixture.verifier, fixture.context),
            Err(DiscoveryEndpointEvidenceError::Rejected)
        ));
    }
}
