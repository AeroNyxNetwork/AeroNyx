// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_attestation_inbox.rs
// ============================================================================
//! Durable quarantine for canonical endpoint-evidence attestations.
//!
//! This repository retains independently signed observations only. It has no
//! peer-store, promotion, ranking, routing, advertisement, or network authority.
// [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX 2026-09-24 by Codex] Keep
// canonical third-party observations in a dedicated bounded quarantine store;
// never merge verifier-local evidence with eligibility inputs.

use std::fmt;
#[cfg(unix)]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1;
use aeronyx_core::protocol::{
    DiscoveryEndpointAttestationPurposeV1, DiscoveryEndpointEvidenceAttestationV1,
    DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1,
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
const MAX_LOGICAL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_RETENTION_TTL_SECS: u64 = 7 * 24 * 60 * 60;
const MAX_CLEANUP_BATCH: usize = 4_096;
const SLOT_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointAttestationSlotV1\0";
const CANDIDATE_GROUP_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointAttestationCandidateGroupV1\0";

/// Bounded policy for one dedicated attestation inbox.
#[derive(Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointAttestationInboxConfig {
    /// Dedicated private database path.
    pub db_path: PathBuf,
    /// Maximum retained rows.
    pub max_entries: usize,
    /// Maximum sum of canonical frame bytes.
    pub max_logical_bytes: u64,
    /// Maximum local retention after admission.
    pub retention_ttl_secs: u64,
    /// Maximum expired rows removed by one transaction.
    pub cleanup_batch_size: usize,
}

impl fmt::Debug for DiscoveryEndpointAttestationInboxConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointAttestationInboxConfig")
            .field("max_entries", &self.max_entries)
            .field("max_logical_bytes", &self.max_logical_bytes)
            .field("retention_ttl_secs", &self.retention_ttl_secs)
            .field("cleanup_batch_size", &self.cleanup_batch_size)
            .finish_non_exhaustive()
    }
}

/// Coarse failures that disclose no identity, endpoint, commitment, or path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiscoveryEndpointAttestationInboxError {
    /// Configuration, canonical bytes, freshness, or context was rejected.
    #[error("endpoint attestation rejected")]
    Rejected,
    /// The dedicated database contains another or unknown schema.
    #[error("endpoint attestation inbox schema unsupported")]
    UnsupportedSchema,
    /// Durable rows or aggregate counters violate the frozen contract.
    #[error("endpoint attestation inbox corrupt")]
    Corrupt,
    /// Storage could not be opened, read, or committed safely.
    #[error("endpoint attestation inbox unavailable")]
    Unavailable,
}

/// Exact replay, slot conflict, and capacity result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryEndpointAttestationRecordOutcome {
    /// A new canonical attestation was durably inserted.
    Inserted,
    /// The exact attestation commitment was already retained.
    Existing,
    /// A fresh retained slot is bound to a different commitment.
    Conflict,
    /// Row or logical-byte capacity is exhausted.
    AtCapacity,
}

/// Identity-free aggregate eligibility input derived only from accepted rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryEndpointAttestationEligibilitySnapshot {
    /// All accepted rows still retained locally.
    pub retained_rows: usize,
    /// Retained rows whose signed interval contains the query time.
    pub fresh_rows: usize,
    /// Exact subject/descriptor/endpoint groups with fresh rows.
    pub candidate_groups: usize,
    /// Candidate groups whose member intervals overlap.
    pub groups_with_freshness_overlap: usize,
    /// Largest distinct-observer count among candidate groups.
    pub max_distinct_observers: usize,
    /// `(distinct observer count, number of exact subject-slot groups)`.
    pub groups_by_distinct_observers: Vec<(usize, usize)>,
}

/// Bounded, identity-free facts for one exact attestation candidate group.
// [PERMISSIONLESS-ENDPOINT-ELIGIBILITY 2026-09-24 by Codex] This projection
// intentionally keeps node, observer, and endpoint material inside SQLite.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointCandidateFacts {
    pub(crate) group_commitment: [u8; 32],
    pub(crate) descriptor_sequence: u64,
    pub(crate) distinct_observers: usize,
    pub(crate) overlap_started_at: u64,
    pub(crate) overlap_expires_at: u64,
    pub(crate) newest_observed_at: u64,
    pub(crate) newest_expires_at: u64,
}

impl fmt::Debug for DiscoveryEndpointCandidateFacts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointCandidateFacts")
            .field("descriptor_sequence", &self.descriptor_sequence)
            .field("distinct_observers", &self.distinct_observers)
            .field("overlap_started_at", &self.overlap_started_at)
            .field("overlap_expires_at", &self.overlap_expires_at)
            .field("newest_observed_at", &self.newest_observed_at)
            .field("newest_expires_at", &self.newest_expires_at)
            .finish_non_exhaustive()
    }
}

/// Canonical ADAT that passed the exact caller-selected context policy.
#[derive(Clone, PartialEq, Eq)]
pub struct VerifiedDiscoveryEndpointAttestationV1 {
    frame: Vec<u8>,
    value: DiscoveryEndpointEvidenceAttestationV1,
}

impl fmt::Debug for VerifiedDiscoveryEndpointAttestationV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedDiscoveryEndpointAttestationV1")
            .field("frame_bytes", &self.frame.len())
            .finish_non_exhaustive()
    }
}

impl VerifiedDiscoveryEndpointAttestationV1 {
    /// Decodes and verifies one canonical ADAT against its expected context.
    ///
    /// # Errors
    /// Returns a coarse rejection for malformed, unsupported, stale, wrongly
    /// signed, or context-mismatched input.
    pub fn verify(
        frame: &[u8],
        now: u64,
        expected_context: [u8; 32],
    ) -> Result<Self, DiscoveryEndpointAttestationInboxError> {
        if now == 0 || expected_context.iter().all(|byte| *byte == 0) {
            return Err(DiscoveryEndpointAttestationInboxError::Rejected);
        }
        let value = DiscoveryEndpointEvidenceAttestationV1::decode(frame)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)?;
        let descriptor = DirectoryDescriptorCommitmentV1 {
            node_id: value.subject_node_id(),
            sequence: value.descriptor_sequence(),
            descriptor_hash: value.descriptor_hash(),
        };
        value
            .verify_at(
                now,
                &value.observer_node_id(),
                &descriptor,
                &value.endpoint_commitment(),
                &value.evidence_commitment(),
                &expected_context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            )
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)?;
        if value.encode() != frame {
            return Err(DiscoveryEndpointAttestationInboxError::Rejected);
        }
        Ok(Self {
            frame: frame.to_vec(),
            value,
        })
    }

    fn commitment(&self) -> [u8; 32] {
        self.value.commitment()
    }

    fn slot_commitment(&self) -> [u8; 32] {
        slot_commitment(&self.value)
    }
}

/// Dedicated, bounded `SQLite` quarantine inbox.
pub struct SqliteDiscoveryEndpointAttestationInbox {
    config: DiscoveryEndpointAttestationInboxConfig,
    connection: Mutex<Connection>,
    #[cfg(unix)]
    _database_parent: File,
}

impl fmt::Debug for SqliteDiscoveryEndpointAttestationInbox {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteDiscoveryEndpointAttestationInbox")
            .field("max_entries", &self.config.max_entries)
            .field("max_logical_bytes", &self.config.max_logical_bytes)
            .finish_non_exhaustive()
    }
}

impl SqliteDiscoveryEndpointAttestationInbox {
    /// Opens and semantically audits one dedicated private database.
    ///
    /// # Errors
    /// Returns a coarse storage, schema, corruption, or configuration error.
    pub fn open(
        config: DiscoveryEndpointAttestationInboxConfig,
    ) -> Result<Self, DiscoveryEndpointAttestationInboxError> {
        validate_config(&config)?;
        let target = prepare_private_sqlite_target(&config.db_path)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "endpoint_attestation_inbox_startup")
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        initialize_schema(&mut connection)?;
        startup_audit(&connection, &config)?;
        Ok(Self {
            config,
            connection: Mutex::new(connection),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    /// Records a verified attestation. Exact replay is checked before time and quota.
    ///
    /// # Errors
    /// Returns a coarse rejection, corruption, or storage error.
    #[allow(clippy::significant_drop_tightening)]
    pub fn record_verified_at(
        &self,
        attestation: &VerifiedDiscoveryEndpointAttestationV1,
        admitted_at: u64,
    ) -> Result<DiscoveryEndpointAttestationRecordOutcome, DiscoveryEndpointAttestationInboxError>
    {
        if admitted_at == 0 {
            return Err(DiscoveryEndpointAttestationInboxError::Rejected);
        }
        let commitment = attestation.commitment();
        let slot = attestation.slot_commitment();
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;

        if commitment_exists(&tx, &commitment)? {
            return finish(tx, DiscoveryEndpointAttestationRecordOutcome::Existing);
        }
        // [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX 2026-09-24 by Codex]
        // Exact replay is the sole pre-freshness outcome. Every rejected new
        // frame exits before stale-slot cleanup or quota accounting mutates.
        let verified = VerifiedDiscoveryEndpointAttestationV1::verify(
            &attestation.frame,
            admitted_at,
            attestation.value.context(),
        )?;
        if let Some(retained_expires_at) = slot_expiry(&tx, &slot)? {
            if retained_expires_at >= admitted_at {
                return finish(tx, DiscoveryEndpointAttestationRecordOutcome::Conflict);
            }
            remove_slot(&tx, &slot)?;
        }
        cleanup_tx(&tx, admitted_at, self.config.cleanup_batch_size)?;
        let (rows, bytes) = load_meta(&tx)?;
        let frame_bytes = u64::try_from(attestation.frame.len())
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)?;
        if rows >= self.config.max_entries
            || bytes
                .checked_add(frame_bytes)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?
                > self.config.max_logical_bytes
        {
            return finish(tx, DiscoveryEndpointAttestationRecordOutcome::AtCapacity);
        }
        let retained_expires_at = verified.value.expires_at().min(
            admitted_at
                .checked_add(self.config.retention_ttl_secs)
                .ok_or(DiscoveryEndpointAttestationInboxError::Rejected)?,
        );
        insert_row(&tx, &verified, admitted_at, retained_expires_at)?;
        update_meta(
            &tx,
            rows.checked_add(1)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
            bytes
                .checked_add(frame_bytes)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        Ok(DiscoveryEndpointAttestationRecordOutcome::Inserted)
    }

    /// Removes at most `limit` expired rows in one immediate transaction.
    ///
    /// # Errors
    /// Returns a coarse rejection, corruption, or storage error.
    #[allow(clippy::significant_drop_tightening)]
    pub fn cleanup_expired_at(
        &self,
        now: u64,
        limit: usize,
    ) -> Result<usize, DiscoveryEndpointAttestationInboxError> {
        if now == 0 || limit == 0 || limit > MAX_CLEANUP_BATCH {
            return Err(DiscoveryEndpointAttestationInboxError::Rejected);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let removed = cleanup_tx(&tx, now, limit)?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        Ok(removed)
    }

    /// Returns aggregate-only multiplicity and freshness-overlap inputs.
    ///
    /// # Errors
    /// Returns a coarse rejection, corruption, or storage error.
    #[allow(clippy::significant_drop_tightening)]
    pub fn eligibility_snapshot_at(
        &self,
        now: u64,
    ) -> Result<
        DiscoveryEndpointAttestationEligibilitySnapshot,
        DiscoveryEndpointAttestationInboxError,
    > {
        if now == 0 {
            return Err(DiscoveryEndpointAttestationInboxError::Rejected);
        }
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let (retained_rows, _) = load_meta_connection(&connection)?;
        let fresh_rows = count_usize(
            &connection,
            "SELECT COUNT(*) FROM discovery_endpoint_attestation_inbox_v1 WHERE retained_expires_at>=?1 AND observed_at<=?1",
            now,
        )?;
        let mut statement = connection
            .prepare(
                "SELECT COUNT(DISTINCT observer_node_id),MAX(observed_at),MIN(retained_expires_at)
                 FROM discovery_endpoint_attestation_inbox_v1
                 WHERE retained_expires_at>=?1 AND observed_at<=?1
                 GROUP BY subject_node_id,descriptor_sequence,descriptor_hash,endpoint_commitment",
            )
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let mut rows = statement
            .query(params![as_i64(now)?])
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let mut histogram = std::collections::BTreeMap::<usize, usize>::new();
        let mut groups = 0usize;
        let mut overlap = 0usize;
        let mut max_observers = 0usize;
        while let Some(row) = rows
            .next()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?
        {
            let observers = usize::try_from(
                row.get::<_, i64>(0)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
            let overlap_start = as_u64(
                row.get(1)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            let overlap_end = as_u64(
                row.get(2)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            groups = groups
                .checked_add(1)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?;
            *histogram.entry(observers).or_default() = histogram
                .get(&observers)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?;
            max_observers = max_observers.max(observers);
            if overlap_start <= overlap_end {
                overlap = overlap
                    .checked_add(1)
                    .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?;
            }
        }
        Ok(DiscoveryEndpointAttestationEligibilitySnapshot {
            retained_rows,
            fresh_rows,
            candidate_groups: groups,
            groups_with_freshness_overlap: overlap,
            max_distinct_observers: max_observers,
            groups_by_distinct_observers: histogram.into_iter().collect(),
        })
    }

    /// Returns bounded factual projections for exact candidate groups.
    ///
    /// The age window is applied to every counted observation. This method
    /// does not decide eligibility, promotion, ranking, or routeability.
    pub(crate) fn candidate_facts_at(
        &self,
        now: u64,
        maximum_evidence_age_secs: u64,
        limit: usize,
    ) -> Result<Vec<DiscoveryEndpointCandidateFacts>, DiscoveryEndpointAttestationInboxError> {
        if now == 0
            || maximum_evidence_age_secs == 0
            || maximum_evidence_age_secs > MAX_RETENTION_TTL_SECS
            || limit == 0
            || limit > self.config.max_entries
        {
            return Err(DiscoveryEndpointAttestationInboxError::Rejected);
        }
        let cutoff = now.saturating_sub(maximum_evidence_age_secs);
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let mut statement = connection
            .prepare(
                "SELECT subject_node_id,descriptor_sequence,descriptor_hash,endpoint_commitment,
                        COUNT(DISTINCT observer_node_id),MAX(observed_at),
                        MIN(retained_expires_at),MAX(retained_expires_at)
                 FROM discovery_endpoint_attestation_inbox_v1
                 WHERE observed_at>=?1 AND observed_at<=?2
                 GROUP BY subject_node_id,descriptor_sequence,descriptor_hash,endpoint_commitment
                 ORDER BY subject_node_id,descriptor_sequence,descriptor_hash,endpoint_commitment
                 LIMIT ?3",
            )
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let mut rows = statement
            .query(params![
                as_i64(cutoff)?,
                as_i64(now)?,
                i64::try_from(limit)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)?
            ])
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        let mut facts = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?
        {
            let subject = array32(
                row.get(0)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            let descriptor_sequence = as_u64(
                row.get(1)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            let descriptor_hash = array32(
                row.get(2)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            let endpoint_commitment = array32(
                row.get(3)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            let newest_observed_at = as_u64(
                row.get(5)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            let overlap_expires_at = as_u64(
                row.get(6)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )?;
            facts.push(DiscoveryEndpointCandidateFacts {
                group_commitment: candidate_group_commitment(
                    &subject,
                    descriptor_sequence,
                    &descriptor_hash,
                    &endpoint_commitment,
                ),
                descriptor_sequence,
                distinct_observers: usize::try_from(
                    row.get::<_, i64>(4)
                        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
                )
                .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
                overlap_started_at: newest_observed_at,
                overlap_expires_at,
                newest_observed_at,
                newest_expires_at: as_u64(
                    row.get(7)
                        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
                )?,
            });
        }
        Ok(facts)
    }
}

fn validate_config(
    config: &DiscoveryEndpointAttestationInboxConfig,
) -> Result<(), DiscoveryEndpointAttestationInboxError> {
    if config.db_path.as_os_str().is_empty()
        || config.db_path == Path::new(":memory:")
        || config.max_entries == 0
        || config.max_entries > MAX_ENTRIES
        || config.max_logical_bytes < DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1 as u64
        || config.max_logical_bytes > MAX_LOGICAL_BYTES
        || config.retention_ttl_secs == 0
        || config.retention_ttl_secs > MAX_RETENTION_TTL_SECS
        || config.cleanup_batch_size == 0
        || config.cleanup_batch_size > MAX_CLEANUP_BATCH
    {
        return Err(DiscoveryEndpointAttestationInboxError::Rejected);
    }
    Ok(())
}

fn initialize_schema(
    connection: &mut Connection,
) -> Result<(), DiscoveryEndpointAttestationInboxError> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    let version: i64 = tx
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    if version == 0 {
        let foreign: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
        if foreign != 0 {
            return Err(DiscoveryEndpointAttestationInboxError::UnsupportedSchema);
        }
        tx.execute_batch(
            "CREATE TABLE discovery_endpoint_attestation_meta_v1(singleton INTEGER PRIMARY KEY CHECK(singleton=1),rows INTEGER NOT NULL,logical_bytes INTEGER NOT NULL);
             INSERT INTO discovery_endpoint_attestation_meta_v1 VALUES(1,0,0);
             CREATE TABLE discovery_endpoint_attestation_inbox_v1(
               attestation_commitment BLOB PRIMARY KEY CHECK(length(attestation_commitment)=32),
               slot_commitment BLOB NOT NULL UNIQUE CHECK(length(slot_commitment)=32),
               subject_node_id BLOB NOT NULL CHECK(length(subject_node_id)=32),descriptor_sequence INTEGER NOT NULL,
               descriptor_hash BLOB NOT NULL CHECK(length(descriptor_hash)=32),endpoint_commitment BLOB NOT NULL CHECK(length(endpoint_commitment)=32),
               evidence_commitment BLOB NOT NULL CHECK(length(evidence_commitment)=32),observer_node_id BLOB NOT NULL CHECK(length(observer_node_id)=32),
               observed_at INTEGER NOT NULL,signed_expires_at INTEGER NOT NULL,retained_expires_at INTEGER NOT NULL,admitted_at INTEGER NOT NULL,
               context BLOB NOT NULL CHECK(length(context)=32),frame BLOB NOT NULL CHECK(length(frame)=289));
             CREATE INDEX discovery_endpoint_attestation_expiry_v1 ON discovery_endpoint_attestation_inbox_v1(retained_expires_at,attestation_commitment);
             CREATE INDEX discovery_endpoint_attestation_group_v1 ON discovery_endpoint_attestation_inbox_v1(subject_node_id,descriptor_sequence,descriptor_hash,endpoint_commitment,observer_node_id);
             PRAGMA user_version=1;"
        ).map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    } else if version != SCHEMA_VERSION {
        return Err(DiscoveryEndpointAttestationInboxError::UnsupportedSchema);
    }
    tx.commit()
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)
}

#[allow(clippy::too_many_lines)]
fn startup_audit(
    connection: &Connection,
    config: &DiscoveryEndpointAttestationInboxConfig,
) -> Result<(), DiscoveryEndpointAttestationInboxError> {
    let (meta_rows, meta_bytes) = load_meta_connection(connection)?;
    let (actual_rows, actual_bytes): (i64, Option<i64>) = connection
        .query_row(
            "SELECT COUNT(*),SUM(LENGTH(frame)) FROM discovery_endpoint_attestation_inbox_v1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    let actual_rows = usize::try_from(actual_rows)
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
    let actual_bytes = as_u64(actual_bytes.unwrap_or(0))?;
    if meta_rows != actual_rows
        || meta_bytes != actual_bytes
        || actual_rows > config.max_entries
        || actual_bytes > config.max_logical_bytes
    {
        return Err(DiscoveryEndpointAttestationInboxError::Corrupt);
    }
    let mut statement = connection.prepare("SELECT attestation_commitment,slot_commitment,subject_node_id,descriptor_sequence,descriptor_hash,endpoint_commitment,evidence_commitment,observer_node_id,observed_at,signed_expires_at,retained_expires_at,admitted_at,context,LENGTH(frame),frame FROM discovery_endpoint_attestation_inbox_v1")
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    let mut rows = statement
        .query([])
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?
    {
        if row
            .get::<_, i64>(13)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?
            != i64::try_from(DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1)
                .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?
        {
            return Err(DiscoveryEndpointAttestationInboxError::Corrupt);
        }
        let frame: Vec<u8> = row
            .get(14)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
        let value = DiscoveryEndpointEvidenceAttestationV1::decode(&frame)
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
        let descriptor = DirectoryDescriptorCommitmentV1 {
            node_id: value.subject_node_id(),
            sequence: value.descriptor_sequence(),
            descriptor_hash: value.descriptor_hash(),
        };
        value
            .verify_at(
                value.observed_at(),
                &value.observer_node_id(),
                &descriptor,
                &value.endpoint_commitment(),
                &value.evidence_commitment(),
                &value.context(),
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            )
            .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
        let admitted_at = as_u64(
            row.get(11)
                .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
        )?;
        let retained = value.expires_at().min(
            admitted_at
                .checked_add(config.retention_ttl_secs)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
        );
        let expected = [
            value.commitment().to_vec(),
            slot_commitment(&value).to_vec(),
            value.subject_node_id().to_vec(),
            value.descriptor_hash().to_vec(),
            value.endpoint_commitment().to_vec(),
            value.evidence_commitment().to_vec(),
            value.observer_node_id().to_vec(),
            value.context().to_vec(),
        ];
        let stored = [0, 1, 2, 4, 5, 6, 7, 12]
            .map(|index| {
                row.get::<_, Vec<u8>>(index)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        if stored.as_slice() != expected
            || as_u64(
                row.get(3)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )? != value.descriptor_sequence()
            || as_u64(
                row.get(8)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )? != value.observed_at()
            || as_u64(
                row.get(9)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )? != value.expires_at()
            || as_u64(
                row.get(10)
                    .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
            )? != retained
        {
            return Err(DiscoveryEndpointAttestationInboxError::Corrupt);
        }
    }
    Ok(())
}

fn insert_row(
    tx: &Transaction<'_>,
    verified: &VerifiedDiscoveryEndpointAttestationV1,
    admitted_at: u64,
    retained_expires_at: u64,
) -> Result<(), DiscoveryEndpointAttestationInboxError> {
    let v = &verified.value;
    tx.execute("INSERT INTO discovery_endpoint_attestation_inbox_v1 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)", params![
        &verified.commitment()[..], &verified.slot_commitment()[..], &v.subject_node_id()[..], as_i64(v.descriptor_sequence())?, &v.descriptor_hash()[..], &v.endpoint_commitment()[..],
        &v.evidence_commitment()[..], &v.observer_node_id()[..], as_i64(v.observed_at())?, as_i64(v.expires_at())?, as_i64(retained_expires_at)?, as_i64(admitted_at)?, &v.context()[..], &verified.frame
    ]).map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    Ok(())
}

fn commitment_exists(
    tx: &Transaction<'_>,
    commitment: &[u8; 32],
) -> Result<bool, DiscoveryEndpointAttestationInboxError> {
    tx.query_row(
        "SELECT 1 FROM discovery_endpoint_attestation_inbox_v1 WHERE attestation_commitment=?1",
        params![&commitment[..]],
        |_| Ok(()),
    )
    .optional()
    .map(|v| v.is_some())
    .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)
}
fn slot_expiry(
    tx: &Transaction<'_>,
    slot: &[u8; 32],
) -> Result<Option<u64>, DiscoveryEndpointAttestationInboxError> {
    tx.query_row("SELECT retained_expires_at FROM discovery_endpoint_attestation_inbox_v1 WHERE slot_commitment=?1", params![&slot[..]], |row| row.get::<_,i64>(0)).optional()
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?.map(as_u64).transpose()
}
fn remove_slot(
    tx: &Transaction<'_>,
    slot: &[u8; 32],
) -> Result<(), DiscoveryEndpointAttestationInboxError> {
    let bytes: i64 = tx.query_row("SELECT LENGTH(frame) FROM discovery_endpoint_attestation_inbox_v1 WHERE slot_commitment=?1", params![&slot[..]], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
    tx.execute(
        "DELETE FROM discovery_endpoint_attestation_inbox_v1 WHERE slot_commitment=?1",
        params![&slot[..]],
    )
    .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    let (rows, total) = load_meta(tx)?;
    update_meta(
        tx,
        rows.checked_sub(1)
            .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
        total
            .checked_sub(as_u64(bytes)?)
            .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
    )
}
fn cleanup_tx(
    tx: &Transaction<'_>,
    now: u64,
    limit: usize,
) -> Result<usize, DiscoveryEndpointAttestationInboxError> {
    let (count, bytes): (i64, Option<i64>) = tx.query_row("SELECT COUNT(*),SUM(LENGTH(frame)) FROM discovery_endpoint_attestation_inbox_v1 WHERE attestation_commitment IN (SELECT attestation_commitment FROM discovery_endpoint_attestation_inbox_v1 WHERE retained_expires_at<?1 ORDER BY retained_expires_at,attestation_commitment LIMIT ?2)", params![as_i64(now)?, i64::try_from(limit).map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)?], |row| Ok((row.get(0)?,row.get(1)?)))
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    tx.execute("DELETE FROM discovery_endpoint_attestation_inbox_v1 WHERE attestation_commitment IN (SELECT attestation_commitment FROM discovery_endpoint_attestation_inbox_v1 WHERE retained_expires_at<?1 ORDER BY retained_expires_at,attestation_commitment LIMIT ?2)", params![as_i64(now)?, i64::try_from(limit).map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)?])
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    let removed =
        usize::try_from(count).map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
    if removed > 0 {
        let (rows, total) = load_meta(tx)?;
        update_meta(
            tx,
            rows.checked_sub(removed)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
            total
                .checked_sub(as_u64(bytes.unwrap_or(0))?)
                .ok_or(DiscoveryEndpointAttestationInboxError::Corrupt)?,
        )?;
    }
    Ok(removed)
}
fn load_meta(tx: &Transaction<'_>) -> Result<(usize, u64), DiscoveryEndpointAttestationInboxError> {
    let (rows, bytes): (i64, i64) = tx
        .query_row(
            "SELECT rows,logical_bytes FROM discovery_endpoint_attestation_meta_v1 WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
    checked_meta(rows, bytes)
}
fn load_meta_connection(
    connection: &Connection,
) -> Result<(usize, u64), DiscoveryEndpointAttestationInboxError> {
    let (rows, bytes): (i64, i64) = connection
        .query_row(
            "SELECT rows,logical_bytes FROM discovery_endpoint_attestation_meta_v1 WHERE singleton=1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?;
    checked_meta(rows, bytes)
}
fn checked_meta(
    rows: i64,
    bytes: i64,
) -> Result<(usize, u64), DiscoveryEndpointAttestationInboxError> {
    Ok((
        usize::try_from(rows).map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)?,
        as_u64(bytes)?,
    ))
}
fn update_meta(
    tx: &Transaction<'_>,
    rows: usize,
    bytes: u64,
) -> Result<(), DiscoveryEndpointAttestationInboxError> {
    tx.execute("UPDATE discovery_endpoint_attestation_meta_v1 SET rows=?1,logical_bytes=?2 WHERE singleton=1",params![i64::try_from(rows).map_err(|_|DiscoveryEndpointAttestationInboxError::Corrupt)?,as_i64(bytes)?]).map_err(|_|DiscoveryEndpointAttestationInboxError::Unavailable)?;
    Ok(())
}
fn count_usize(
    connection: &Connection,
    sql: &str,
    now: u64,
) -> Result<usize, DiscoveryEndpointAttestationInboxError> {
    let count: i64 = connection
        .query_row(sql, params![as_i64(now)?], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    usize::try_from(count).map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)
}
fn finish(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointAttestationRecordOutcome,
) -> Result<DiscoveryEndpointAttestationRecordOutcome, DiscoveryEndpointAttestationInboxError> {
    tx.commit()
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Unavailable)?;
    Ok(outcome)
}
fn slot_commitment(value: &DiscoveryEndpointEvidenceAttestationV1) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SLOT_DOMAIN);
    h.update(value.observer_node_id());
    h.update(value.subject_node_id());
    h.update(value.descriptor_sequence().to_be_bytes());
    h.update(value.descriptor_hash());
    h.update(value.endpoint_commitment());
    h.finalize().into()
}

fn candidate_group_commitment(
    subject_node_id: &[u8; 32],
    descriptor_sequence: u64,
    descriptor_hash: &[u8; 32],
    endpoint_commitment: &[u8; 32],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(CANDIDATE_GROUP_DOMAIN);
    h.update(subject_node_id);
    h.update(descriptor_sequence.to_be_bytes());
    h.update(descriptor_hash);
    h.update(endpoint_commitment);
    h.finalize().into()
}

fn array32(value: Vec<u8>) -> Result<[u8; 32], DiscoveryEndpointAttestationInboxError> {
    value
        .try_into()
        .map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)
}
fn as_i64(value: u64) -> Result<i64, DiscoveryEndpointAttestationInboxError> {
    i64::try_from(value).map_err(|_| DiscoveryEndpointAttestationInboxError::Rejected)
}
fn as_u64(value: i64) -> Result<u64, DiscoveryEndpointAttestationInboxError> {
    u64::try_from(value).map_err(|_| DiscoveryEndpointAttestationInboxError::Corrupt)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{NodeDescriptor, SignedNodeDescriptor};
    use aeronyx_core::protocol::discovery_endpoint_attestation::DiscoveryEndpointEvidenceAttestationV1;
    use aeronyx_core::protocol::discovery_endpoint_proof::{
        canonical_public_endpoint_commitment, DiscoveryEndpointChallengeV1,
        DiscoveryEndpointProofV1,
    };
    use tempfile::TempDir;

    const NOW: u64 = 2_000_000_000;
    const ENDPOINT: &str = "8.8.8.8:51820";

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed key")
    }

    struct Fixture {
        verified: VerifiedDiscoveryEndpointAttestationV1,
    }

    #[allow(clippy::too_many_arguments)]
    fn fixture(
        observer_seed: u8,
        target_seed: u8,
        nonce_seed: u8,
        context_seed: u8,
        observed_at: u64,
        expires_at: u64,
        descriptor_sequence: u64,
    ) -> Fixture {
        fixture_with_endpoint(
            observer_seed,
            target_seed,
            nonce_seed,
            context_seed,
            observed_at,
            expires_at,
            descriptor_sequence,
            ENDPOINT,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fixture_with_endpoint(
        observer_seed: u8,
        target_seed: u8,
        nonce_seed: u8,
        context_seed: u8,
        observed_at: u64,
        expires_at: u64,
        descriptor_sequence: u64,
        endpoint_text: &str,
    ) -> Fixture {
        let observer = key(observer_seed);
        let target = key(target_seed);
        let context = [context_seed; 32];
        let mut descriptor = NodeDescriptor::new(
            target.public_key_bytes(),
            descriptor_sequence,
            NOW - 100,
            NOW + 10_000,
            "1.0.0",
        );
        descriptor.public_endpoint = Some(endpoint_text.to_string());
        let descriptor = SignedNodeDescriptor::sign(descriptor, &target).expect("descriptor");
        let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .expect("descriptor pin");
        let endpoint = canonical_public_endpoint_commitment(endpoint_text).expect("endpoint");
        let challenge = DiscoveryEndpointChallengeV1::issue(
            target.public_key_bytes(),
            pin.descriptor_hash,
            endpoint,
            [nonce_seed; 32],
            context,
            observed_at,
            expires_at,
            &observer,
        )
        .expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &context, observed_at, &target)
            .expect("proof");
        let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
            &descriptor,
            &challenge,
            &proof,
            context,
            DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            observed_at,
            expires_at,
            &observer,
        )
        .expect("attestation");
        let frame = attestation.encode();
        Fixture {
            verified: VerifiedDiscoveryEndpointAttestationV1::verify(&frame, observed_at, context)
                .expect("verified"),
        }
    }

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external test root");
        TempDir::new_in("target/test-temp").expect("tempdir")
    }

    fn config(
        dir: &TempDir,
        max_entries: usize,
        max_bytes: u64,
    ) -> DiscoveryEndpointAttestationInboxConfig {
        DiscoveryEndpointAttestationInboxConfig {
            db_path: dir.path().join("attestations.sqlite3"),
            max_entries,
            max_logical_bytes: max_bytes,
            retention_ttl_secs: 120,
            cleanup_batch_size: 8,
        }
    }

    #[test]
    fn insert_exact_replay_restart_and_conflict_are_deterministic() {
        let dir = tempdir();
        let first = fixture(3, 9, 1, 7, NOW, NOW + 60, 4);
        let store =
            SqliteDiscoveryEndpointAttestationInbox::open(config(&dir, 8, 4096)).expect("open");
        assert_eq!(
            store
                .record_verified_at(&first.verified, NOW)
                .expect("insert"),
            DiscoveryEndpointAttestationRecordOutcome::Inserted
        );
        assert_eq!(
            store
                .record_verified_at(&first.verified, NOW + 1_000)
                .expect("exact replay before freshness"),
            DiscoveryEndpointAttestationRecordOutcome::Existing
        );
        let conflict = fixture(3, 9, 2, 7, NOW + 1, NOW + 61, 4);
        assert_eq!(
            store
                .record_verified_at(&conflict.verified, NOW + 2)
                .expect("conflict"),
            DiscoveryEndpointAttestationRecordOutcome::Conflict
        );
        let before = store.eligibility_snapshot_at(NOW + 2).expect("snapshot");
        assert_eq!(before.retained_rows, 1);
        assert_eq!(before.fresh_rows, 1);
        drop(store);
        let reopened =
            SqliteDiscoveryEndpointAttestationInbox::open(config(&dir, 8, 4096)).expect("restart");
        assert_eq!(
            reopened.eligibility_snapshot_at(NOW + 2).expect("snapshot"),
            before
        );
    }

    #[test]
    fn expired_slot_can_refresh_without_weakening_exact_replay() {
        let dir = tempdir();
        let store =
            SqliteDiscoveryEndpointAttestationInbox::open(config(&dir, 1, 4096)).expect("open");
        let first = fixture(3, 9, 1, 7, NOW, NOW + 10, 4);
        store
            .record_verified_at(&first.verified, NOW)
            .expect("first");
        assert_eq!(
            store
                .record_verified_at(&first.verified, NOW + 20)
                .expect("expired exact replay"),
            DiscoveryEndpointAttestationRecordOutcome::Existing
        );
        let refresh = fixture(3, 9, 2, 7, NOW + 11, NOW + 50, 4);
        assert_eq!(
            store
                .record_verified_at(&refresh.verified, NOW + 20)
                .expect("refresh"),
            DiscoveryEndpointAttestationRecordOutcome::Inserted
        );
        let snapshot = store.eligibility_snapshot_at(NOW + 20).expect("snapshot");
        assert_eq!(snapshot.retained_rows, 1);
        assert_eq!(snapshot.fresh_rows, 1);
    }

    #[test]
    fn aggregate_counts_distinct_observers_without_identity_projection() {
        let dir = tempdir();
        let store =
            SqliteDiscoveryEndpointAttestationInbox::open(config(&dir, 8, 8192)).expect("open");
        let first = fixture(3, 9, 1, 7, NOW, NOW + 60, 4);
        let second = fixture(4, 9, 2, 8, NOW + 1, NOW + 59, 4);
        store
            .record_verified_at(&first.verified, NOW)
            .expect("first");
        store
            .record_verified_at(&second.verified, NOW + 1)
            .expect("second");
        let snapshot = store.eligibility_snapshot_at(NOW + 2).expect("snapshot");
        assert_eq!(snapshot.retained_rows, 2);
        assert_eq!(snapshot.candidate_groups, 1);
        assert_eq!(snapshot.groups_with_freshness_overlap, 1);
        assert_eq!(snapshot.max_distinct_observers, 2);
        assert_eq!(snapshot.groups_by_distinct_observers, vec![(2, 1)]);
        let debug = format!("{snapshot:?}");
        for secret in [
            first.verified.value.subject_node_id(),
            first.verified.value.observer_node_id(),
            first.verified.value.endpoint_commitment(),
        ] {
            assert!(!debug.contains(&hex::encode(secret)));
        }
    }

    #[test]
    fn candidate_facts_are_bounded_deduplicated_and_exactly_grouped() {
        let dir = tempdir();
        let store =
            SqliteDiscoveryEndpointAttestationInbox::open(config(&dir, 8, 8192)).expect("open");
        let first = fixture(3, 9, 1, 7, NOW, NOW + 60, 4);
        let second = fixture(4, 9, 2, 8, NOW + 1, NOW + 59, 4);
        let other_sequence = fixture(5, 9, 3, 9, NOW + 1, NOW + 59, 5);
        let other_endpoint =
            fixture_with_endpoint(6, 9, 4, 10, NOW + 1, NOW + 59, 4, "8.8.4.4:51820");
        for item in [&first, &second, &other_sequence, &other_endpoint] {
            store
                .record_verified_at(&item.verified, item.verified.value.observed_at())
                .expect("insert");
        }
        assert_eq!(
            store
                .record_verified_at(&first.verified, NOW + 2)
                .expect("exact replay"),
            DiscoveryEndpointAttestationRecordOutcome::Existing
        );

        let facts = store
            .candidate_facts_at(NOW + 2, 30, 8)
            .expect("candidate facts");
        assert_eq!(facts.len(), 3);
        let mut observer_counts = facts
            .iter()
            .map(|fact| fact.distinct_observers)
            .collect::<Vec<_>>();
        observer_counts.sort_unstable();
        assert_eq!(observer_counts, vec![1, 1, 2]);
        assert_eq!(
            facts
                .iter()
                .map(|fact| fact.group_commitment)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        let exact = facts
            .iter()
            .find(|fact| fact.distinct_observers == 2)
            .expect("exact group");
        assert_eq!(exact.overlap_started_at, NOW + 1);
        assert_eq!(exact.overlap_expires_at, NOW + 59);
        let debug = format!("{facts:?}");
        for secret in [
            first.verified.value.subject_node_id(),
            first.verified.value.observer_node_id(),
            first.verified.value.endpoint_commitment(),
            facts[0].group_commitment,
        ] {
            assert!(!debug.contains(&hex::encode(secret)));
        }
        assert!(store.candidate_facts_at(NOW, 30, 0).is_err());
        assert!(store.candidate_facts_at(NOW, 30, 9).is_err());
    }

    #[test]
    fn quota_cleanup_and_rejected_attempts_do_not_mutate_meta() {
        let dir = tempdir();
        let store = SqliteDiscoveryEndpointAttestationInbox::open(config(
            &dir,
            1,
            DISCOVERY_ENDPOINT_ATTESTATION_FRAME_BYTES_V1 as u64,
        ))
        .expect("open");
        let first = fixture(3, 9, 1, 7, NOW, NOW + 30, 4);
        let second = fixture(4, 10, 2, 8, NOW, NOW + 60, 5);
        store
            .record_verified_at(&first.verified, NOW)
            .expect("first");
        assert_eq!(
            store
                .record_verified_at(&second.verified, NOW + 1)
                .expect("capacity"),
            DiscoveryEndpointAttestationRecordOutcome::AtCapacity
        );
        assert_eq!(
            store
                .eligibility_snapshot_at(NOW + 1)
                .expect("snapshot")
                .retained_rows,
            1
        );
        assert_eq!(store.cleanup_expired_at(NOW + 31, 1).expect("cleanup"), 1);
        assert_eq!(
            store
                .record_verified_at(&second.verified, NOW + 31)
                .expect("after cleanup"),
            DiscoveryEndpointAttestationRecordOutcome::Inserted
        );
    }

    #[test]
    fn malformed_foreign_schema_and_corrupt_restart_fail_closed() {
        let original = fixture(3, 9, 1, 7, NOW, NOW + 60, 4);
        let mut valid = original.verified.frame.clone();
        valid.push(0);
        assert_eq!(
            VerifiedDiscoveryEndpointAttestationV1::verify(&valid, NOW, [7; 32]),
            Err(DiscoveryEndpointAttestationInboxError::Rejected)
        );
        let mut unknown_version = original.verified.frame.clone();
        unknown_version[4] = 2;
        assert_eq!(
            VerifiedDiscoveryEndpointAttestationV1::verify(&unknown_version, NOW, [7; 32]),
            Err(DiscoveryEndpointAttestationInboxError::Rejected)
        );
        assert_eq!(
            VerifiedDiscoveryEndpointAttestationV1::verify(&original.verified.frame, NOW, [8; 32],),
            Err(DiscoveryEndpointAttestationInboxError::Rejected)
        );

        let foreign = tempdir();
        let foreign_path = foreign.path().join("attestations.sqlite3");
        Connection::open(&foreign_path)
            .expect("foreign open")
            .execute("CREATE TABLE unrelated(value INTEGER)", [])
            .expect("foreign schema");
        assert!(matches!(
            SqliteDiscoveryEndpointAttestationInbox::open(config(&foreign, 4, 4096)),
            Err(DiscoveryEndpointAttestationInboxError::UnsupportedSchema)
        ));
        let names: i64 = Connection::open(&foreign_path)
            .expect("inspect")
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name LIKE 'discovery_endpoint_attestation_%'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(names, 0);

        let corrupt = tempdir();
        let cfg = config(&corrupt, 4, 4096);
        let item = fixture(3, 9, 1, 7, NOW, NOW + 60, 4);
        let store = SqliteDiscoveryEndpointAttestationInbox::open(cfg.clone()).expect("open");
        store
            .record_verified_at(&item.verified, NOW)
            .expect("insert");
        drop(store);
        Connection::open(&cfg.db_path)
            .expect("tamper open")
            .execute(
                "UPDATE discovery_endpoint_attestation_meta_v1 SET logical_bytes=0",
                [],
            )
            .expect("tamper");
        assert!(matches!(
            SqliteDiscoveryEndpointAttestationInbox::open(cfg),
            Err(DiscoveryEndpointAttestationInboxError::Corrupt)
        ));
    }

    #[test]
    fn rejected_stale_new_attestation_has_zero_cleanup_or_meta_mutation() {
        let dir = tempdir();
        let store =
            SqliteDiscoveryEndpointAttestationInbox::open(config(&dir, 4, 4096)).expect("open");
        let retained = fixture(3, 9, 1, 7, NOW, NOW + 10, 4);
        store
            .record_verified_at(&retained.verified, NOW)
            .expect("insert");
        let stale_new = fixture(4, 10, 2, 8, NOW, NOW + 10, 5);
        assert_eq!(
            store.record_verified_at(&stale_new.verified, NOW + 20),
            Err(DiscoveryEndpointAttestationInboxError::Rejected)
        );
        let snapshot = store.eligibility_snapshot_at(NOW + 20).expect("snapshot");
        assert_eq!(snapshot.retained_rows, 1);
        assert_eq!(snapshot.fresh_rows, 0);
    }
}
