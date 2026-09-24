// ============================================================================
// File: crates/aeronyx-server/src/services/discovery_endpoint_quarantine.rs
// ============================================================================
//! Durable registry for endpoint candidates eligible only for quarantine.
//!
//! This repository accepts only evaluator-minted admission tokens. It has no
//! peer-store, descriptor, endpoint, ranking, routing, promotion, or network
//! dependency and cannot make a candidate routeable.
// [PERMISSIONLESS-ENDPOINT-QUARANTINE-ADMISSION 2026-09-24 by Codex] Preserve
// the one-way boundary from typed eligibility into isolated durable quarantine.

use std::fmt;
#[cfg(unix)]
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use sha2::{Digest, Sha256};

use super::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use super::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};
use super::chat_relay_mailbox::{prepare_private_sqlite_target, verify_private_file};
use super::discovery_endpoint_eligibility::DiscoveryEndpointQuarantineAdmission;

const SCHEMA_VERSION: i64 = 2;
const MINIMUM_SYNCHRONOUS_LEVEL: i64 = 2;
const MAX_ENTRIES: usize = 65_536;
const MAX_CLEANUP_BATCH: usize = 4_096;
const ADMISSION_DOMAIN: &[u8] = b"AeroNyx/DiscoveryEndpointQuarantineAdmissionV2\0";

/// Bounded policy for one dedicated quarantine registry.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineConfig {
    pub(crate) db_path: PathBuf,
    pub(crate) max_entries: usize,
    pub(crate) cleanup_batch_size: usize,
}

impl fmt::Debug for DiscoveryEndpointQuarantineConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointQuarantineConfig")
            .field("max_entries", &self.max_entries)
            .field("cleanup_batch_size", &self.cleanup_batch_size)
            .finish_non_exhaustive()
    }
}

/// Coarse failures that disclose no candidate commitment or path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DiscoveryEndpointQuarantineError {
    #[error("endpoint quarantine admission rejected")]
    Rejected,
    #[error("endpoint quarantine schema unsupported")]
    UnsupportedSchema,
    #[error("endpoint quarantine registry corrupt")]
    Corrupt,
    #[error("endpoint quarantine registry unavailable")]
    Unavailable,
}

/// Exact replay, slot conflict, and capacity result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiscoveryEndpointQuarantineRecordOutcome {
    Inserted,
    Existing,
    Conflict,
    Expired,
    AtCapacity,
}

/// Aggregate-only registry status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointQuarantineSnapshot {
    pub(crate) retained_candidates: usize,
    pub(crate) fresh_candidates: usize,
    pub(crate) maximum_valid_until: Option<u64>,
}

/// Fresh, opaque capability issued only by the durable quarantine registry.
// [PERMISSIONLESS-ENDPOINT-QUARANTINE-OBSERVATION 2026-09-24 by Codex] The
// observation policy receives no candidate identity, descriptor, or endpoint.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscoveryEndpointFreshQuarantineAdmission {
    admission_commitment: [u8; 32],
    group_commitment: [u8; 32],
    descriptor_sequence: u64,
    policy_version: u64,
    valid_until: u64,
}

impl DiscoveryEndpointFreshQuarantineAdmission {
    pub(crate) const fn admission_commitment(&self) -> [u8; 32] {
        self.admission_commitment
    }

    pub(crate) const fn group_commitment(&self) -> [u8; 32] {
        self.group_commitment
    }

    pub(crate) const fn descriptor_sequence(&self) -> u64 {
        self.descriptor_sequence
    }

    pub(crate) const fn policy_version(&self) -> u64 {
        self.policy_version
    }

    pub(crate) const fn valid_until(&self) -> u64 {
        self.valid_until
    }
}

impl fmt::Debug for DiscoveryEndpointFreshQuarantineAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiscoveryEndpointFreshQuarantineAdmission")
            .field("descriptor_sequence", &self.descriptor_sequence)
            .field("policy_version", &self.policy_version)
            .field("valid_until", &self.valid_until)
            .finish_non_exhaustive()
    }
}

/// Dedicated private registry with no routeability projection.
pub(crate) struct SqliteDiscoveryEndpointQuarantineRegistry {
    config: DiscoveryEndpointQuarantineConfig,
    connection: Mutex<Connection>,
    #[cfg(unix)]
    _database_parent: File,
}

impl fmt::Debug for SqliteDiscoveryEndpointQuarantineRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteDiscoveryEndpointQuarantineRegistry")
            .field("max_entries", &self.config.max_entries)
            .field("cleanup_batch_size", &self.config.cleanup_batch_size)
            .finish_non_exhaustive()
    }
}

impl SqliteDiscoveryEndpointQuarantineRegistry {
    pub(crate) fn open(
        config: DiscoveryEndpointQuarantineConfig,
    ) -> Result<Self, DiscoveryEndpointQuarantineError> {
        validate_config(&config)?;
        let target = prepare_private_sqlite_target(&config.db_path)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        verify_private_file(&target.resolved_path, false)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        restrict_private_sqlite_permissions(&target.resolved_path)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        verify_private_file(&target.resolved_path, true)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        verify_sqlite_physical_integrity(&connection, "endpoint_quarantine_startup")
            .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?;
        configure_full_durability(&connection, MINIMUM_SYNCHRONOUS_LEVEL)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        initialize_schema(&mut connection)?;
        startup_audit(&connection, &config)?;
        Ok(Self {
            config,
            connection: Mutex::new(connection),
            #[cfg(unix)]
            _database_parent: target.parent,
        })
    }

    /// Atomically admits an evaluator-minted token into quarantine.
    #[allow(clippy::significant_drop_tightening)]
    pub(crate) fn record_at(
        &self,
        admission: DiscoveryEndpointQuarantineAdmission,
        now: u64,
    ) -> Result<DiscoveryEndpointQuarantineRecordOutcome, DiscoveryEndpointQuarantineError> {
        if now == 0 {
            return Err(DiscoveryEndpointQuarantineError::Rejected);
        }
        let group = admission.group_commitment();
        let descriptor_sequence = admission.descriptor_sequence();
        let policy_version = admission.policy_version();
        let valid_until = admission.valid_until();
        let commitment =
            admission_commitment(&group, descriptor_sequence, policy_version, valid_until);
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;

        if admission_exists(&tx, &commitment)? {
            return finish(tx, DiscoveryEndpointQuarantineRecordOutcome::Existing);
        }
        if group.iter().all(|byte| *byte == 0) || policy_version == 0 {
            return finish(tx, DiscoveryEndpointQuarantineRecordOutcome::Conflict);
        }
        if valid_until < now {
            return finish(tx, DiscoveryEndpointQuarantineRecordOutcome::Expired);
        }
        if let Some(existing_expiry) = group_expiry(&tx, &group)? {
            if existing_expiry >= now {
                return finish(tx, DiscoveryEndpointQuarantineRecordOutcome::Conflict);
            }
            remove_group(&tx, &group)?;
        }
        cleanup_tx(&tx, now, self.config.cleanup_batch_size)?;
        let rows = load_meta(&tx)?;
        if rows >= self.config.max_entries {
            return finish(tx, DiscoveryEndpointQuarantineRecordOutcome::AtCapacity);
        }
        tx.execute(
            "INSERT INTO discovery_endpoint_quarantine_v1(
               group_commitment,admission_commitment,descriptor_sequence,policy_version,
               valid_until,admitted_at
             ) VALUES(?1,?2,?3,?4,?5,?6)",
            params![
                &group[..],
                &commitment[..],
                as_i64(descriptor_sequence)?,
                as_i64(policy_version)?,
                as_i64(valid_until)?,
                as_i64(now)?,
            ],
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        update_meta(
            &tx,
            rows.checked_add(1)
                .ok_or(DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantineRecordOutcome::Inserted)
    }

    pub(crate) fn cleanup_expired_at(
        &self,
        now: u64,
        limit: usize,
    ) -> Result<usize, DiscoveryEndpointQuarantineError> {
        if now == 0 || limit == 0 || limit > MAX_CLEANUP_BATCH {
            return Err(DiscoveryEndpointQuarantineError::Rejected);
        }
        let mut connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let tx = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let removed = cleanup_tx(&tx, now, limit)?;
        tx.commit()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        Ok(removed)
    }

    pub(crate) fn snapshot_at(
        &self,
        now: u64,
    ) -> Result<DiscoveryEndpointQuarantineSnapshot, DiscoveryEndpointQuarantineError> {
        if now == 0 {
            return Err(DiscoveryEndpointQuarantineError::Rejected);
        }
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let retained_candidates = load_meta_connection(&connection)?;
        let (fresh, maximum): (i64, Option<i64>) = connection
            .query_row(
                "SELECT COUNT(*),MAX(valid_until) FROM discovery_endpoint_quarantine_v1
                 WHERE valid_until>=?1",
                params![as_i64(now)?],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        Ok(DiscoveryEndpointQuarantineSnapshot {
            retained_candidates,
            fresh_candidates: usize::try_from(fresh)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
            maximum_valid_until: maximum.map(as_u64).transpose()?,
        })
    }

    /// Resolves one opaque commitment into a fresh typed capability.
    ///
    /// This is deliberately an exact lookup: it cannot enumerate candidates,
    /// reveal candidate identities, or project routeable peer state.
    pub(crate) fn fresh_admission_at(
        &self,
        exact_commitment: [u8; 32],
        now: u64,
    ) -> Result<Option<DiscoveryEndpointFreshQuarantineAdmission>, DiscoveryEndpointQuarantineError>
    {
        if now == 0 || exact_commitment.iter().all(|byte| *byte == 0) {
            return Err(DiscoveryEndpointQuarantineError::Rejected);
        }
        let connection = self
            .connection
            .lock()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let row = connection
            .query_row(
                "SELECT group_commitment,descriptor_sequence,policy_version,valid_until
                 FROM discovery_endpoint_quarantine_v1 WHERE admission_commitment=?1",
                params![&exact_commitment[..]],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        let Some((group, descriptor_sequence, policy_version, valid_until)) = row else {
            return Ok(None);
        };
        let group = array32(group)?;
        let descriptor_sequence = as_u64(descriptor_sequence)?;
        let policy_version = as_u64(policy_version)?;
        let valid_until = as_u64(valid_until)?;
        if group.iter().all(|byte| *byte == 0)
            || policy_version == 0
            || exact_commitment
                != admission_commitment(&group, descriptor_sequence, policy_version, valid_until)
        {
            return Err(DiscoveryEndpointQuarantineError::Corrupt);
        }
        if valid_until < now {
            return Ok(None);
        }
        Ok(Some(DiscoveryEndpointFreshQuarantineAdmission {
            admission_commitment: exact_commitment,
            group_commitment: group,
            descriptor_sequence,
            policy_version,
            valid_until,
        }))
    }
}

fn validate_config(
    config: &DiscoveryEndpointQuarantineConfig,
) -> Result<(), DiscoveryEndpointQuarantineError> {
    if config.db_path.as_os_str().is_empty()
        || config.db_path == Path::new(":memory:")
        || config.max_entries == 0
        || config.max_entries > MAX_ENTRIES
        || config.cleanup_batch_size == 0
        || config.cleanup_batch_size > MAX_CLEANUP_BATCH
    {
        return Err(DiscoveryEndpointQuarantineError::Rejected);
    }
    Ok(())
}

fn initialize_schema(connection: &mut Connection) -> Result<(), DiscoveryEndpointQuarantineError> {
    let tx = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    let version: i64 = tx
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    if version == 0 {
        let foreign: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
        if foreign != 0 {
            return Err(DiscoveryEndpointQuarantineError::UnsupportedSchema);
        }
        tx.execute_batch(
            "CREATE TABLE discovery_endpoint_quarantine_meta_v1(
               singleton INTEGER PRIMARY KEY CHECK(singleton=1),rows INTEGER NOT NULL
             );
             INSERT INTO discovery_endpoint_quarantine_meta_v1 VALUES(1,0);
             CREATE TABLE discovery_endpoint_quarantine_v1(
               group_commitment BLOB PRIMARY KEY CHECK(length(group_commitment)=32),
               admission_commitment BLOB NOT NULL UNIQUE CHECK(length(admission_commitment)=32),
               descriptor_sequence INTEGER NOT NULL,policy_version INTEGER NOT NULL,
               valid_until INTEGER NOT NULL,admitted_at INTEGER NOT NULL
             );
             CREATE INDEX discovery_endpoint_quarantine_expiry_v1
               ON discovery_endpoint_quarantine_v1(valid_until,group_commitment);
             PRAGMA user_version=2;",
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    } else if version == 1 {
        // [PERMISSIONLESS-ENDPOINT-PROMOTION-READINESS 2026-09-24 by Codex]
        // V1 rows cannot be promoted because they never persisted the signed
        // descriptor sequence. Discard them atomically instead of inventing it.
        tx.execute_batch(
            "DELETE FROM discovery_endpoint_quarantine_v1;
             UPDATE discovery_endpoint_quarantine_meta_v1 SET rows=0 WHERE singleton=1;
             ALTER TABLE discovery_endpoint_quarantine_v1
               ADD COLUMN descriptor_sequence INTEGER NOT NULL DEFAULT 0;
             PRAGMA user_version=2;",
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    } else if version != SCHEMA_VERSION {
        return Err(DiscoveryEndpointQuarantineError::UnsupportedSchema);
    }
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)
}

fn startup_audit(
    connection: &Connection,
    config: &DiscoveryEndpointQuarantineConfig,
) -> Result<(), DiscoveryEndpointQuarantineError> {
    let meta = load_meta_connection(connection)?;
    let actual: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_v1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    let actual = usize::try_from(actual).map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?;
    if meta != actual || actual > config.max_entries {
        return Err(DiscoveryEndpointQuarantineError::Corrupt);
    }
    let mut statement = connection
        .prepare(
            "SELECT group_commitment,admission_commitment,descriptor_sequence,policy_version,
                    valid_until,admitted_at
             FROM discovery_endpoint_quarantine_v1",
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    let mut rows = statement
        .query([])
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    while let Some(row) = rows
        .next()
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?
    {
        let group = array32(
            row.get(0)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        let stored = array32(
            row.get(1)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        let descriptor_sequence = as_u64(
            row.get(2)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        let policy = as_u64(
            row.get(3)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        let valid_until = as_u64(
            row.get(4)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        let admitted_at = as_u64(
            row.get(5)
                .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
        if group.iter().all(|byte| *byte == 0)
            || policy == 0
            || valid_until < admitted_at
            || stored != admission_commitment(&group, descriptor_sequence, policy, valid_until)
        {
            return Err(DiscoveryEndpointQuarantineError::Corrupt);
        }
    }
    Ok(())
}

fn admission_commitment(
    group: &[u8; 32],
    descriptor_sequence: u64,
    policy_version: u64,
    valid_until: u64,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(ADMISSION_DOMAIN);
    hash.update(group);
    hash.update(descriptor_sequence.to_be_bytes());
    hash.update(policy_version.to_be_bytes());
    hash.update(valid_until.to_be_bytes());
    hash.finalize().into()
}

/// Computes the opaque lookup key for an evaluator-minted admission.
pub(crate) fn quarantine_admission_commitment(
    admission: &DiscoveryEndpointQuarantineAdmission,
) -> [u8; 32] {
    admission_commitment(
        &admission.group_commitment(),
        admission.descriptor_sequence(),
        admission.policy_version(),
        admission.valid_until(),
    )
}

fn admission_exists(
    tx: &Transaction<'_>,
    commitment: &[u8; 32],
) -> Result<bool, DiscoveryEndpointQuarantineError> {
    tx.query_row(
        "SELECT 1 FROM discovery_endpoint_quarantine_v1 WHERE admission_commitment=?1",
        params![&commitment[..]],
        |_| Ok(()),
    )
    .optional()
    .map(|value| value.is_some())
    .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)
}

fn group_expiry(
    tx: &Transaction<'_>,
    group: &[u8; 32],
) -> Result<Option<u64>, DiscoveryEndpointQuarantineError> {
    tx.query_row(
        "SELECT valid_until FROM discovery_endpoint_quarantine_v1 WHERE group_commitment=?1",
        params![&group[..]],
        |row| row.get::<_, i64>(0),
    )
    .optional()
    .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?
    .map(as_u64)
    .transpose()
}

fn remove_group(
    tx: &Transaction<'_>,
    group: &[u8; 32],
) -> Result<(), DiscoveryEndpointQuarantineError> {
    let removed = tx
        .execute(
            "DELETE FROM discovery_endpoint_quarantine_v1 WHERE group_commitment=?1",
            params![&group[..]],
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    if removed != 1 {
        return Err(DiscoveryEndpointQuarantineError::Corrupt);
    }
    let rows = load_meta(tx)?;
    update_meta(
        tx,
        rows.checked_sub(1)
            .ok_or(DiscoveryEndpointQuarantineError::Corrupt)?,
    )
}

fn cleanup_tx(
    tx: &Transaction<'_>,
    now: u64,
    limit: usize,
) -> Result<usize, DiscoveryEndpointQuarantineError> {
    let count: i64 = tx
        .query_row(
            "SELECT COUNT(*) FROM discovery_endpoint_quarantine_v1
             WHERE group_commitment IN (
               SELECT group_commitment FROM discovery_endpoint_quarantine_v1
               WHERE valid_until<?1 ORDER BY valid_until,group_commitment LIMIT ?2
             )",
            params![
                as_i64(now)?,
                i64::try_from(limit).map_err(|_| DiscoveryEndpointQuarantineError::Rejected)?
            ],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    tx.execute(
        "DELETE FROM discovery_endpoint_quarantine_v1
         WHERE group_commitment IN (
           SELECT group_commitment FROM discovery_endpoint_quarantine_v1
           WHERE valid_until<?1 ORDER BY valid_until,group_commitment LIMIT ?2
         )",
        params![
            as_i64(now)?,
            i64::try_from(limit).map_err(|_| DiscoveryEndpointQuarantineError::Rejected)?
        ],
    )
    .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    let removed = usize::try_from(count).map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?;
    if removed > 0 {
        let rows = load_meta(tx)?;
        update_meta(
            tx,
            rows.checked_sub(removed)
                .ok_or(DiscoveryEndpointQuarantineError::Corrupt)?,
        )?;
    }
    Ok(removed)
}

fn load_meta(tx: &Transaction<'_>) -> Result<usize, DiscoveryEndpointQuarantineError> {
    let rows: i64 = tx
        .query_row(
            "SELECT rows FROM discovery_endpoint_quarantine_meta_v1 WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?;
    usize::try_from(rows).map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)
}

fn load_meta_connection(
    connection: &Connection,
) -> Result<usize, DiscoveryEndpointQuarantineError> {
    let rows: i64 = connection
        .query_row(
            "SELECT rows FROM discovery_endpoint_quarantine_meta_v1 WHERE singleton=1",
            [],
            |row| row.get(0),
        )
        .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?;
    usize::try_from(rows).map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)
}

fn update_meta(tx: &Transaction<'_>, rows: usize) -> Result<(), DiscoveryEndpointQuarantineError> {
    tx.execute(
        "UPDATE discovery_endpoint_quarantine_meta_v1 SET rows=?1 WHERE singleton=1",
        params![i64::try_from(rows).map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)?],
    )
    .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    Ok(())
}

fn finish(
    tx: Transaction<'_>,
    outcome: DiscoveryEndpointQuarantineRecordOutcome,
) -> Result<DiscoveryEndpointQuarantineRecordOutcome, DiscoveryEndpointQuarantineError> {
    tx.commit()
        .map_err(|_| DiscoveryEndpointQuarantineError::Unavailable)?;
    Ok(outcome)
}

fn array32(value: Vec<u8>) -> Result<[u8; 32], DiscoveryEndpointQuarantineError> {
    value
        .try_into()
        .map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)
}

fn as_i64(value: u64) -> Result<i64, DiscoveryEndpointQuarantineError> {
    i64::try_from(value).map_err(|_| DiscoveryEndpointQuarantineError::Rejected)
}

fn as_u64(value: i64) -> Result<u64, DiscoveryEndpointQuarantineError> {
    u64::try_from(value).map_err(|_| DiscoveryEndpointQuarantineError::Corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::discovery_endpoint_attestation_inbox::DiscoveryEndpointCandidateFacts;
    use crate::services::discovery_endpoint_eligibility::{
        evaluate_endpoint_candidate, DiscoveryEndpointEligibilityPolicy,
        DiscoveryEndpointStakePolicyMode,
    };
    use tempfile::TempDir;

    const NOW: u64 = 2_000_000_000;

    fn tempdir() -> TempDir {
        std::fs::create_dir_all("target/test-temp").expect("external test root");
        TempDir::new_in("target/test-temp").expect("tempdir")
    }

    fn config(dir: &TempDir, max_entries: usize) -> DiscoveryEndpointQuarantineConfig {
        DiscoveryEndpointQuarantineConfig {
            db_path: dir.path().join("endpoint-quarantine.sqlite3"),
            max_entries,
            cleanup_batch_size: 8,
        }
    }

    fn admission(
        group_seed: u8,
        policy_version: u64,
        valid_until: u64,
    ) -> DiscoveryEndpointQuarantineAdmission {
        let facts = DiscoveryEndpointCandidateFacts {
            group_commitment: [group_seed; 32],
            descriptor_sequence: 7,
            distinct_observers: 2,
            overlap_started_at: NOW - 1,
            overlap_expires_at: valid_until,
            newest_observed_at: NOW - 1,
            newest_expires_at: valid_until,
        };
        evaluate_endpoint_candidate(
            &facts,
            DiscoveryEndpointEligibilityPolicy::new(
                2,
                60,
                policy_version,
                DiscoveryEndpointStakePolicyMode::Disabled,
            )
            .expect("policy"),
            NOW,
            None,
        )
        .into_quarantine_admission()
        .expect("eligible admission")
    }

    #[test]
    fn exact_replay_conflict_expiry_and_restart_are_deterministic() {
        let directory = tempdir();
        let cfg = config(&directory, 4);
        let registry = SqliteDiscoveryEndpointQuarantineRegistry::open(cfg.clone()).expect("open");
        let first = admission(0x31, 1, NOW + 30);
        assert_eq!(
            registry.record_at(first, NOW).expect("insert"),
            DiscoveryEndpointQuarantineRecordOutcome::Inserted
        );
        assert_eq!(
            registry
                .record_at(first, NOW + 1_000)
                .expect("exact replay"),
            DiscoveryEndpointQuarantineRecordOutcome::Existing
        );
        let conflict = admission(0x31, 2, NOW + 40);
        assert_eq!(
            registry.record_at(conflict, NOW + 1).expect("conflict"),
            DiscoveryEndpointQuarantineRecordOutcome::Conflict
        );
        drop(registry);
        let reopened = SqliteDiscoveryEndpointQuarantineRegistry::open(cfg).expect("restart");
        assert_eq!(
            reopened.snapshot_at(NOW + 1).expect("snapshot"),
            DiscoveryEndpointQuarantineSnapshot {
                retained_candidates: 1,
                fresh_candidates: 1,
                maximum_valid_until: Some(NOW + 30),
            }
        );
        assert_eq!(
            reopened
                .record_at(conflict, NOW + 31)
                .expect("expired replacement"),
            DiscoveryEndpointQuarantineRecordOutcome::Inserted
        );
    }

    #[test]
    fn capacity_cleanup_and_missing_stake_fail_closed() {
        let directory = tempdir();
        let registry =
            SqliteDiscoveryEndpointQuarantineRegistry::open(config(&directory, 1)).expect("open");
        registry
            .record_at(admission(0x31, 1, NOW + 10), NOW)
            .expect("first");
        assert_eq!(
            registry
                .record_at(admission(0x32, 1, NOW + 20), NOW)
                .expect("capacity"),
            DiscoveryEndpointQuarantineRecordOutcome::AtCapacity
        );
        assert_eq!(
            registry.cleanup_expired_at(NOW + 11, 1).expect("cleanup"),
            1
        );
        assert_eq!(
            registry
                .record_at(admission(0x55, 1, NOW + 20), NOW + 21)
                .expect("expired admission"),
            DiscoveryEndpointQuarantineRecordOutcome::Expired
        );
        assert_eq!(
            registry
                .record_at(admission(0x32, 1, NOW + 20), NOW + 11)
                .expect("after cleanup"),
            DiscoveryEndpointQuarantineRecordOutcome::Inserted
        );

        let facts = DiscoveryEndpointCandidateFacts {
            group_commitment: [0x33; 32],
            descriptor_sequence: 8,
            distinct_observers: 2,
            overlap_started_at: NOW,
            overlap_expires_at: NOW + 20,
            newest_observed_at: NOW,
            newest_expires_at: NOW + 20,
        };
        let missing_stake = evaluate_endpoint_candidate(
            &facts,
            DiscoveryEndpointEligibilityPolicy::new(
                2,
                60,
                1,
                DiscoveryEndpointStakePolicyMode::Required,
            )
            .expect("policy"),
            NOW,
            None,
        );
        assert!(missing_stake.into_quarantine_admission().is_none());
        assert_eq!(
            registry
                .snapshot_at(NOW + 11)
                .expect("snapshot")
                .retained_candidates,
            1
        );
    }

    #[test]
    fn v1_rows_are_discarded_instead_of_receiving_an_invented_sequence() {
        let directory = tempdir();
        let cfg = config(&directory, 4);
        let legacy = rusqlite::Connection::open(&cfg.db_path).expect("legacy database");
        legacy
            .execute_batch(
                "CREATE TABLE discovery_endpoint_quarantine_meta_v1(
                   singleton INTEGER PRIMARY KEY CHECK(singleton=1),rows INTEGER NOT NULL
                 );
                 INSERT INTO discovery_endpoint_quarantine_meta_v1 VALUES(1,1);
                 CREATE TABLE discovery_endpoint_quarantine_v1(
                   group_commitment BLOB PRIMARY KEY CHECK(length(group_commitment)=32),
                   admission_commitment BLOB NOT NULL UNIQUE CHECK(length(admission_commitment)=32),
                   policy_version INTEGER NOT NULL,valid_until INTEGER NOT NULL,
                   admitted_at INTEGER NOT NULL
                 );
                 CREATE INDEX discovery_endpoint_quarantine_expiry_v1
                   ON discovery_endpoint_quarantine_v1(valid_until,group_commitment);
                 PRAGMA user_version=1;",
            )
            .expect("legacy schema");
        legacy
            .execute(
                "INSERT INTO discovery_endpoint_quarantine_v1 VALUES(?1,?2,1,?3,?4)",
                params![&[0x71_u8; 32][..], &[0x72_u8; 32][..], NOW + 20, NOW],
            )
            .expect("legacy row");
        drop(legacy);

        let registry =
            SqliteDiscoveryEndpointQuarantineRegistry::open(cfg.clone()).expect("migrate");
        assert_eq!(
            registry.snapshot_at(NOW).expect("snapshot"),
            DiscoveryEndpointQuarantineSnapshot {
                retained_candidates: 0,
                fresh_candidates: 0,
                maximum_valid_until: None,
            }
        );
        drop(registry);
        let migrated = rusqlite::Connection::open(&cfg.db_path).expect("migrated database");
        let version: i64 = migrated
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("schema version");
        let rows: i64 = migrated
            .query_row(
                "SELECT COUNT(*) FROM discovery_endpoint_quarantine_v1",
                [],
                |row| row.get(0),
            )
            .expect("rows");
        let sequence_column: i64 = migrated
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('discovery_endpoint_quarantine_v1')
                 WHERE name='descriptor_sequence'",
                [],
                |row| row.get(0),
            )
            .expect("sequence column");
        assert_eq!((version, rows, sequence_column), (2, 0, 1));
    }

    #[test]
    fn foreign_schema_corruption_and_debug_surfaces_fail_closed() {
        let foreign = tempdir();
        let foreign_config = config(&foreign, 4);
        rusqlite::Connection::open(&foreign_config.db_path)
            .expect("foreign database")
            .execute_batch("CREATE TABLE unrelated(value INTEGER);")
            .expect("foreign schema");
        assert!(matches!(
            SqliteDiscoveryEndpointQuarantineRegistry::open(foreign_config),
            Err(DiscoveryEndpointQuarantineError::UnsupportedSchema)
        ));

        let directory = tempdir();
        let cfg = config(&directory, 4);
        let registry = SqliteDiscoveryEndpointQuarantineRegistry::open(cfg.clone()).expect("open");
        registry
            .record_at(admission(0x44, 1, NOW + 20), NOW)
            .expect("insert");
        let debug = format!(
            "{registry:?}{:?}",
            registry.snapshot_at(NOW).expect("snapshot")
        );
        assert!(!debug.contains(&hex::encode([0x44; 32])));
        drop(registry);
        rusqlite::Connection::open(&cfg.db_path)
            .expect("database")
            .execute(
                "UPDATE discovery_endpoint_quarantine_v1 SET policy_version=2",
                [],
            )
            .expect("tamper");
        assert!(matches!(
            SqliteDiscoveryEndpointQuarantineRegistry::open(cfg),
            Err(DiscoveryEndpointQuarantineError::Corrupt)
        ));
    }
}
