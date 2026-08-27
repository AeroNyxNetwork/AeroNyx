// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay_backup_certification.rs
// ============================================================================
// Version: 1.0.0-RecoveryImageCertificationDomain
//
// Creation Reason:
//   [CHAT-RELAY-BACKUP-CERTIFICATION-DOMAIN 2026-08-27 by Codex] Extract the
//   read-only SQLite recovery-image trust boundary from `ChatRelayService`.
//
// Main Functionality:
//   - Normalizes an isolated SQLite recovery image to DELETE journal mode.
//   - Verifies bounded physical integrity before accepting an image.
//   - Verifies required schema sentinels and fixed replay-row shapes.
//   - Reconciles aggregate custody accounting against canonical rows.
//   - Verifies durable queue-sequence continuity and circuit checkpoints.
//
// Dependencies:
//   - chat_relay_direct_peer_circuit.rs: anonymous circuit checkpoint validator.
//   - chat_relay_error.rs: stable path-free failure contract.
//   - rusqlite: read-only recovery-image inspection and journal normalization.
//
// Main Logical Flow:
//   1. Normalize only the isolated image's journal mode before publication.
//   2. Run SQLite `quick_check(1)` and collapse physical failures safely.
//   3. Validate schema, replay rows, circuit state, accounting, and ordering.
//   4. Collapse all logical failures to one public backup-integrity bucket.
//
// Important Note for Next Developer:
//   - Keep certification independent from live service state and identifiers.
//   - Never weaken the logical failure collapse or expose private row content.
//   - Add new durable side-effect tables to this certification boundary before
//     treating a recovery image as complete.
//   - Preserve the exact ordering: physical integrity must precede logical SQL.
//
// Last Modified:
//   v1.0.0-RecoveryImageCertificationDomain - Initial focused extraction.
// ============================================================================

use rusqlite::{params, Connection, OptionalExtension};

use crate::services::chat_relay_direct_peer_circuit::{
    DirectPeerCircuitDomain, SqliteDirectPeerCircuitRepository,
    DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION, DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE,
};
use crate::services::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// One required SQLite schema feature and its exact installed version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecoveryImageSchemaRequirement {
    feature: &'static str,
    version: i64,
}

impl RecoveryImageSchemaRequirement {
    pub(crate) const fn new(feature: &'static str, version: i64) -> Self {
        Self { feature, version }
    }
}

/// Capability for normalizing and certifying an isolated recovery image.
pub(crate) trait BackupRecoveryImageCertification {
    /// Makes the offline image self-contained for immutable read-only replay.
    fn normalize_journal(&self, connection: &Connection) -> ChatRelayResult<()>;

    /// Proves the image is physically and logically fit for recovery.
    fn verify(&self, connection: &Connection, now: u64) -> ChatRelayResult<()>;
}

/// SQLite implementation of the relay recovery-image certification policy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SqliteBackupRecoveryImageCertifier {
    verified_submit_schema: RecoveryImageSchemaRequirement,
    blind_route_schema: RecoveryImageSchemaRequirement,
}

impl SqliteBackupRecoveryImageCertifier {
    pub(crate) const fn new(
        verified_submit_schema: RecoveryImageSchemaRequirement,
        blind_route_schema: RecoveryImageSchemaRequirement,
    ) -> Self {
        Self {
            verified_submit_schema,
            blind_route_schema,
        }
    }

    fn verify_logical_integrity(&self, connection: &Connection, now: u64) -> ChatRelayResult<()> {
        verify_schema_requirement(
            connection,
            RecoveryImageSchemaRequirement::new(
                DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE,
                DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION,
            ),
            "sqlite_backup_schema_sentinel",
        )?;
        verify_schema_requirement(
            connection,
            self.verified_submit_schema,
            "sqlite_backup_verified_submit_schema_sentinel",
        )?;
        verify_verified_submit_rows(connection)?;
        verify_schema_requirement(
            connection,
            self.blind_route_schema,
            "sqlite_backup_blind_route_schema_sentinel",
        )?;
        verify_blind_route_rows(connection)?;

        DirectPeerCircuitDomain::<SqliteDirectPeerCircuitRepository>::validate_checkpoint(
            connection, now,
        )?;
        verify_storage_accounting(connection)?;
        verify_queue_sequence(connection)
    }
}

impl BackupRecoveryImageCertification for SqliteBackupRecoveryImageCertifier {
    fn normalize_journal(&self, connection: &Connection) -> ChatRelayResult<()> {
        // [CHAT-RELAY-BACKUP-CERTIFICATION-DOMAIN 2026-08-27 by Codex]
        // SQLite may copy WAL mode into the isolated image. A later read-only
        // open can then create sidecars, violating immutable artifact replay.
        let journal_mode = connection
            .query_row("PRAGMA journal_mode=DELETE", [], |row| {
                row.get::<_, String>(0)
            })
            .map_err(|_| ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_journal_mode",
            })?;
        if !journal_mode.eq_ignore_ascii_case("delete") {
            return Err(ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_journal_mode",
            });
        }
        Ok(())
    }

    fn verify(&self, connection: &Connection, now: u64) -> ChatRelayResult<()> {
        verify_sqlite_physical_integrity(connection, "sqlite_backup_integrity")?;
        self.verify_logical_integrity(connection, now).map_err(|_| {
            ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_logical_integrity",
            }
        })
    }
}

/// Runs SQLite's bounded physical-integrity gate with a stable failure bucket.
pub(crate) fn verify_sqlite_physical_integrity(
    connection: &Connection,
    failure_field: &'static str,
) -> ChatRelayResult<()> {
    // [CHAT-RELAY-BACKUP-CERTIFICATION-DOMAIN 2026-08-27 by Codex]
    // `quick_check(1)` bounds returned findings while traversing the database.
    // SQLite errors and non-ok findings intentionally share one path-free error.
    let outcome = connection
        .query_row("PRAGMA quick_check(1)", [], |row| row.get::<_, String>(0))
        .map_err(|_| ChatRelayError::CorruptStoredData {
            field: failure_field,
        })?;
    if outcome != "ok" {
        return Err(ChatRelayError::CorruptStoredData {
            field: failure_field,
        });
    }
    Ok(())
}

fn verify_schema_requirement(
    connection: &Connection,
    requirement: RecoveryImageSchemaRequirement,
    failure_field: &'static str,
) -> ChatRelayResult<()> {
    let installed_version = connection
        .query_row(
            "SELECT schema_version
             FROM relay_schema_features
             WHERE feature = ?1",
            params![requirement.feature],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    if installed_version != Some(requirement.version) {
        return Err(ChatRelayError::CorruptStoredData {
            field: failure_field,
        });
    }
    Ok(())
}

fn verify_verified_submit_rows(connection: &Connection) -> ChatRelayResult<()> {
    let invalid_responses = connection.query_row(
        "SELECT COUNT(*) FROM relay_verified_submit_responses
         WHERE LENGTH(cache_key) != 32
            OR LENGTH(envelope_fingerprint) != 32
            OR LENGTH(response_nonce) != 24
            OR LENGTH(response_ciphertext) <= 16
            OR LENGTH(response_ciphertext) > 528
            OR completed_at < 0",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    if invalid_responses != 0 {
        return Err(ChatRelayError::CorruptStoredData {
            field: "sqlite_backup_verified_submit_rows",
        });
    }

    let invalid_reservations = connection.query_row(
        "SELECT COUNT(*) FROM relay_verified_submit_reservations
         WHERE LENGTH(cache_key) != 32
            OR LENGTH(envelope_fingerprint) != 32
            OR reserved_at < 0
            OR owner_epoch IS NULL
            OR TYPEOF(owner_epoch) != 'blob'
            OR LENGTH(owner_epoch) != 16
            OR TYPEOF(owner_acquired_at) != 'integer'
            OR owner_acquired_at < reserved_at",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    if invalid_reservations != 0 {
        return Err(ChatRelayError::CorruptStoredData {
            field: "sqlite_backup_verified_submit_reservations",
        });
    }
    Ok(())
}

fn verify_blind_route_rows(connection: &Connection) -> ChatRelayResult<()> {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Validate only fixed
    // shape here. AEAD authenticity remains node-secret-bound and is checked
    // when an exact response is recovered.
    let invalid_responses = connection.query_row(
        "SELECT COUNT(*) FROM relay_blind_route_responses
         WHERE LENGTH(cache_key) != 32
            OR LENGTH(request_fingerprint) != 32
            OR LENGTH(response_nonce) != 24
            OR LENGTH(response_ciphertext) <= 16
            OR LENGTH(response_ciphertext) > 2064
            OR completed_at < 0",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    let invalid_reservations = connection.query_row(
        "SELECT COUNT(*) FROM relay_blind_route_reservations
         WHERE LENGTH(cache_key) != 32
            OR LENGTH(request_fingerprint) != 32
            OR reserved_at < 0
            OR owner_epoch IS NULL
            OR TYPEOF(owner_epoch) != 'blob'
            OR LENGTH(owner_epoch) != 16
            OR TYPEOF(owner_acquired_at) != 'integer'
            OR owner_acquired_at < reserved_at
            OR (effect_started_at IS NOT NULL
                AND (TYPEOF(effect_started_at) != 'integer'
                     OR effect_started_at < reserved_at))",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    if invalid_responses != 0 || invalid_reservations != 0 {
        return Err(ChatRelayError::CorruptStoredData {
            field: "sqlite_backup_blind_route_rows",
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StorageUsageSnapshot {
    pending_messages: u64,
    pending_message_bytes: u64,
    pending_blobs: u64,
    pending_blob_bytes: u64,
}

fn verify_storage_accounting(connection: &Connection) -> ChatRelayResult<()> {
    if read_stored_usage(connection)? != read_canonical_usage(connection)? {
        return Err(ChatRelayError::CorruptStoredData {
            field: "sqlite_backup_storage_usage",
        });
    }
    Ok(())
}

fn read_stored_usage(connection: &Connection) -> ChatRelayResult<StorageUsageSnapshot> {
    let counters = connection.query_row(
        "SELECT
            pending_message_count,
            pending_message_bytes,
            pending_blob_count,
            pending_blob_bytes
         FROM relay_storage_usage
         WHERE singleton = 1",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        },
    )?;
    Ok(StorageUsageSnapshot {
        pending_messages: nonnegative(counters.0, "pending_message_count")?,
        pending_message_bytes: nonnegative(counters.1, "pending_message_bytes")?,
        pending_blobs: nonnegative(counters.2, "pending_blob_count")?,
        pending_blob_bytes: nonnegative(counters.3, "pending_blob_bytes")?,
    })
}

fn read_canonical_usage(connection: &Connection) -> ChatRelayResult<StorageUsageSnapshot> {
    let counters = connection.query_row(
        "SELECT
            (SELECT COUNT(*) FROM pending_messages WHERE status = 0),
            (SELECT COALESCE(SUM(LENGTH(envelope)), 0)
               FROM pending_messages WHERE status = 0),
            (SELECT COUNT(*) FROM pending_blobs),
            (SELECT COALESCE(SUM(size), 0) FROM pending_blobs)",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
            ))
        },
    )?;
    Ok(StorageUsageSnapshot {
        pending_messages: nonnegative(counters.0, "canonical_pending_message_count")?,
        pending_message_bytes: nonnegative(counters.1, "canonical_pending_message_bytes")?,
        pending_blobs: nonnegative(counters.2, "canonical_pending_blob_count")?,
        pending_blob_bytes: nonnegative(counters.3, "canonical_pending_blob_bytes")?,
    })
}

fn verify_queue_sequence(connection: &Connection) -> ChatRelayResult<()> {
    let (last_sequence, max_sequence, missing_sequences) = connection.query_row(
        "SELECT
            (SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1),
            (SELECT COALESCE(MAX(queue_sequence), 0) FROM pending_messages),
            (SELECT COUNT(*) FROM pending_messages WHERE queue_sequence IS NULL)",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        },
    )?;
    if last_sequence < 0
        || max_sequence < 0
        || missing_sequences != 0
        || last_sequence < max_sequence
    {
        return Err(ChatRelayError::CorruptStoredData {
            field: "sqlite_backup_queue_sequence",
        });
    }
    Ok(())
}

fn nonnegative(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}
