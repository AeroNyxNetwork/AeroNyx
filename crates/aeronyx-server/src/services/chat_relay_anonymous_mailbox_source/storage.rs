// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/storage.rs
// ============================================
//! # Source journal `SQLite` storage
//!
//! Owns the source journal schema bootstrap and v1-to-v2 migration, the
//! aggregate entry/byte accounting row, row-size and retention-phase invariants,
//! checked integer conversions, and the private `SQLite` hardening shims.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use std::path::Path;

use rusqlite::{params, Connection, TransactionBehavior};

use crate::services::chat_relay_backup_certification::verify_sqlite_physical_integrity;
use crate::services::chat_relay_backup_sqlite::{
    configure_full_durability, restrict_private_sqlite_permissions,
};
use crate::services::chat_relay_mailbox::AnonymousMailboxStoreError;

use super::{
    AnonymousMailboxSourceError, AnonymousMailboxSourcePhase, JOURNAL_AEAD_TAG_BYTES,
    MAX_JOURNAL_BODY_BYTES, MAX_JOURNAL_PROTECTED_STATE_BYTES,
};

// [ANONYMOUS-MAILBOX-SOURCE-RETENTION 2026-09-13 by Codex] Schema v2 gives
// terminal rows a durable replay deadline and exact aggregate accounting.
// Unresolved phases deliberately retain NULL forever and cannot be age-cleaned.
pub(super) const SOURCE_JOURNAL_SCHEMA_VERSION: i64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SourceJournalMeta {
    pub(super) entries: u64,
    pub(super) bytes: u64,
}

pub(super) fn initialize_or_verify_source_schema(
    connection: &mut Connection,
    migration_now: u64,
    terminal_retention_secs: u64,
) -> Result<(), AnonymousMailboxSourceError> {
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
    let version: i64 = transaction
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let foreign_before: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type IN ('table', 'index', 'view', 'trigger')
               AND name NOT LIKE 'sqlite_%'
               AND name NOT IN ('anonymous_mailbox_source_journal',
                                'anonymous_mailbox_source_meta')",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if foreign_before != 0 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    match version {
        0 => {
            let owned_before: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type IN ('table', 'index', 'view', 'trigger')
                       AND name NOT LIKE 'sqlite_%'",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            if owned_before != 0 {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            transaction
                .execute_batch(
                    "CREATE TABLE anonymous_mailbox_source_journal (
                        route_id BLOB PRIMARY KEY NOT NULL CHECK (length(route_id) = 16),
                        request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                        target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                        descriptor_commitment BLOB NOT NULL CHECK (length(descriptor_commitment) > 0),
                        body BLOB NOT NULL CHECK (length(body) > 0),
                        phase INTEGER NOT NULL CHECK (phase BETWEEN 1 AND 5),
                        retain_until INTEGER CHECK (retain_until IS NULL OR retain_until >= 0),
                        state_nonce BLOB NOT NULL CHECK (length(state_nonce) = 24),
                        protected_state BLOB NOT NULL CHECK (length(protected_state) >= 16)
                     );
                     CREATE TABLE anonymous_mailbox_source_meta (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        schema_version INTEGER NOT NULL CHECK (schema_version = 2),
                        total_entries INTEGER NOT NULL CHECK (total_entries >= 0),
                        total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0)
                     );
                     INSERT INTO anonymous_mailbox_source_meta VALUES (1, 2, 0, 0);
                     PRAGMA user_version = 2;",
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        }
        1 => {
            let legacy_tables: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'anonymous_mailbox_source_journal'",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            let unexpected_meta: i64 = transaction
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type = 'table' AND name = 'anonymous_mailbox_source_meta'",
                    [],
                    |row| row.get(0),
                )
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            if legacy_tables != 1 || unexpected_meta != 0 {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let retain_until = migration_now
                .checked_add(terminal_retention_secs)
                .filter(|value| *value <= i64::MAX as u64)
                .ok_or(AnonymousMailboxSourceError::Rejected)?;
            transaction
                .execute_batch(
                    "ALTER TABLE anonymous_mailbox_source_journal
                         ADD COLUMN retain_until INTEGER
                         CHECK (retain_until IS NULL OR retain_until >= 0);
                     CREATE TABLE anonymous_mailbox_source_meta (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        schema_version INTEGER NOT NULL CHECK (schema_version = 2),
                        total_entries INTEGER NOT NULL CHECK (total_entries >= 0),
                        total_bytes INTEGER NOT NULL CHECK (total_bytes >= 0)
                     );",
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            transaction
                .execute(
                    "UPDATE anonymous_mailbox_source_journal SET retain_until = ?1
                     WHERE phase IN (3, 5)",
                    params![source_i64(retain_until)?],
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            let observed = compute_source_meta(&transaction)?;
            transaction
                .execute(
                    "INSERT INTO anonymous_mailbox_source_meta
                     (singleton, schema_version, total_entries, total_bytes)
                     VALUES (1, 2, ?1, ?2)",
                    params![source_i64(observed.entries)?, source_i64(observed.bytes)?],
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            transaction
                .execute_batch("PRAGMA user_version = 2;")
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        }
        SOURCE_JOURNAL_SCHEMA_VERSION => {}
        _ => return Err(AnonymousMailboxSourceError::Corrupt),
    }
    let table_count: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND name IN ('anonymous_mailbox_source_journal',
                            'anonymous_mailbox_source_meta')",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    let owned_after: i64 = transaction
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type IN ('table', 'index', 'view', 'trigger')
               AND name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get(0),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    if table_count != 2 || owned_after != 2 {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    load_source_meta(&transaction)?;
    transaction
        .commit()
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

pub(super) fn source_i64(value: u64) -> Result<i64, AnonymousMailboxSourceError> {
    i64::try_from(value).map_err(|_| AnonymousMailboxSourceError::Rejected)
}

pub(super) fn source_u64(value: i64) -> Result<u64, AnonymousMailboxSourceError> {
    u64::try_from(value).map_err(|_| AnonymousMailboxSourceError::Corrupt)
}

pub(super) fn source_row_bytes(
    body_len: i64,
    protected_len: i64,
) -> Result<u64, AnonymousMailboxSourceError> {
    if body_len < 1
        || usize::try_from(body_len)
            .ok()
            .filter(|value| *value <= MAX_JOURNAL_BODY_BYTES)
            .is_none()
        || protected_len < JOURNAL_AEAD_TAG_BYTES as i64
        || usize::try_from(protected_len)
            .ok()
            .filter(|value| *value <= MAX_JOURNAL_PROTECTED_STATE_BYTES)
            .is_none()
    {
        return Err(AnonymousMailboxSourceError::Corrupt);
    }
    source_u64(body_len)?
        .checked_add(source_u64(protected_len)?)
        .ok_or(AnonymousMailboxSourceError::Corrupt)
}

pub(super) fn validate_phase_retention(
    phase: AnonymousMailboxSourcePhase,
    retain_until: Option<u64>,
) -> Result<(), AnonymousMailboxSourceError> {
    let terminal = matches!(
        phase,
        AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
    );
    if terminal == retain_until.is_some() {
        Ok(())
    } else {
        Err(AnonymousMailboxSourceError::Corrupt)
    }
}

pub(super) fn load_source_meta(
    connection: &Connection,
) -> Result<SourceJournalMeta, AnonymousMailboxSourceError> {
    connection
        .query_row(
            "SELECT schema_version, total_entries, total_bytes
             FROM anonymous_mailbox_source_meta WHERE singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)
        .and_then(|(version, entries, bytes)| {
            if version != SOURCE_JOURNAL_SCHEMA_VERSION {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            Ok(SourceJournalMeta {
                entries: source_u64(entries)?,
                bytes: source_u64(bytes)?,
            })
        })
}

pub(super) fn compute_source_meta(
    connection: &Connection,
) -> Result<SourceJournalMeta, AnonymousMailboxSourceError> {
    let (entries, bytes): (i64, i64) = connection
        .query_row(
            "SELECT COUNT(*), COALESCE(SUM(length(body) + length(protected_state)), 0)
             FROM anonymous_mailbox_source_journal",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
    Ok(SourceJournalMeta {
        entries: source_u64(entries)?,
        bytes: source_u64(bytes)?,
    })
}

pub(super) fn update_source_meta_exact(
    transaction: &rusqlite::Transaction<'_>,
    before: SourceJournalMeta,
    after: SourceJournalMeta,
) -> Result<(), AnonymousMailboxSourceError> {
    let updated = transaction
        .execute(
            "UPDATE anonymous_mailbox_source_meta
             SET total_entries = ?1, total_bytes = ?2
             WHERE singleton = 1 AND schema_version = 2
               AND total_entries = ?3 AND total_bytes = ?4",
            params![
                source_i64(after.entries)?,
                source_i64(after.bytes)?,
                source_i64(before.entries)?,
                source_i64(before.bytes)?
            ],
        )
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
    if updated == 1 {
        Ok(())
    } else {
        Err(AnonymousMailboxSourceError::Corrupt)
    }
}

pub(super) fn verify_source_sqlite_integrity(
    connection: &Connection,
) -> Result<(), AnonymousMailboxSourceError> {
    verify_sqlite_physical_integrity(connection, "anonymous_mailbox_source_startup_integrity")
        .map_err(|_| AnonymousMailboxSourceError::Corrupt)
}

pub(super) fn configure_source_full_durability(
    connection: &Connection,
) -> Result<(), AnonymousMailboxSourceError> {
    configure_full_durability(connection, 2)
        .map(|_| ())
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

pub(super) fn restrict_private_source_permissions(
    path: &Path,
) -> Result<(), AnonymousMailboxSourceError> {
    restrict_private_sqlite_permissions(path).map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

pub(super) fn map_private_sqlite_error(
    error: AnonymousMailboxStoreError,
) -> AnonymousMailboxSourceError {
    match error {
        AnonymousMailboxStoreError::Rejected => AnonymousMailboxSourceError::Rejected,
        AnonymousMailboxStoreError::Corrupt | AnonymousMailboxStoreError::UnsupportedSchema => {
            AnonymousMailboxSourceError::Corrupt
        }
        AnonymousMailboxStoreError::Disabled
        | AnonymousMailboxStoreError::Busy
        | AnonymousMailboxStoreError::Unavailable => AnonymousMailboxSourceError::Unavailable,
    }
}
