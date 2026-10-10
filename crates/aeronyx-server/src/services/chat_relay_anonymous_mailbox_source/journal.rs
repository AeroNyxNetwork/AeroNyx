// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/journal.rs
// ============================================
//! # Encrypted `SQLite` source journal
//!
//! Owns the durable record types and the `SQLite` implementation of the source
//! journal: private-inode open, startup audit, exact-replay insert, phase CAS
//! transitions, terminal retention cleanup, and AEAD sealing of persisted state.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::fs::File;

use aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1;
use chacha20poly1305::aead::{Aead, NewAead, Payload};
use chacha20poly1305::{Key, XChaCha20Poly1305, XNonce};
use parking_lot::Mutex;
use rand::rngs::OsRng;
use rand::RngCore;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};

use crate::config_chat_relay::{
    AnonymousMailboxSourceConfig, MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_RETENTION_SECS,
};
use crate::services::chat_relay_mailbox::{prepare_private_sqlite_target, verify_private_file};

#[cfg(test)]
use super::crash_drill::crash_after_source_journal_commit;
use super::state_codec::{
    decode_state, encode_state, fixed, journal_aad, source_body_commitment,
    source_request_commitment, validate_source_record_projection,
};
use super::storage::{
    compute_source_meta, configure_source_full_durability, initialize_or_verify_source_schema,
    load_source_meta, map_private_sqlite_error, restrict_private_source_permissions, source_i64,
    source_row_bytes, source_u64, update_source_meta_exact, validate_phase_retention,
    verify_source_sqlite_integrity, SourceJournalMeta,
};
use super::{
    AnonymousMailboxSourceError, AnonymousMailboxSourcePhase, JOURNAL_AEAD_TAG_BYTES,
    JOURNAL_STATE_BODY_COMMITMENT_BYTES, MAX_JOURNAL_BODY_BYTES, MAX_JOURNAL_CLEAR_STATE_BYTES,
    MAX_JOURNAL_PROTECTED_STATE_BYTES,
};

const JOURNAL_NONCE_BYTES: usize = 24;
const MAX_JOURNAL_DESCRIPTOR_BYTES: usize = 1024;

pub(super) struct SourceJournalRecord {
    pub(super) route_id: [u8; 16],
    pub(super) request_commitment: [u8; 32],
    pub(super) target_node_id: [u8; 32],
    pub(super) descriptor_commitment: DirectoryDescriptorCommitmentV1,
    pub(super) body: Vec<u8>,
    pub(super) phase: AnonymousMailboxSourcePhase,
    pub(super) retain_until: Option<u64>,
    pub(super) state: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnonymousMailboxSourceCleanupReport {
    pub(crate) rows_removed: u64,
    pub(crate) bytes_removed: u64,
}

/// SQLite implementation of the source journal. `open` is deliberately
/// synchronous: server startup invokes it through `spawn_blocking` before any
/// source route becomes reachable.
pub struct SqliteAnonymousMailboxSourceJournal {
    pub(super) connection: Mutex<Connection>,
    journal_key: [u8; 32],
    max_entries: usize,
    max_bytes: u64,
    terminal_retention_secs: u64,
    cleanup_batch_size: usize,
    #[cfg(unix)]
    _database_parent: Option<File>,
}

impl SqliteAnonymousMailboxSourceJournal {
    pub(crate) fn new(
        mut connection: Connection,
        journal_key: [u8; 32],
        config: &AnonymousMailboxSourceConfig,
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !source_config_is_valid(config) || journal_key == [0; 32] {
            return Err(AnonymousMailboxSourceError::Disabled);
        }
        initialize_or_verify_source_schema(
            &mut connection,
            source_now_secs()?,
            config.terminal_retention_secs,
        )?;
        let journal = Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
            terminal_retention_secs: config.terminal_retention_secs,
            cleanup_batch_size: config.cleanup_batch_size,
            #[cfg(unix)]
            _database_parent: None,
        };
        journal.audit_startup()?;
        Ok(journal)
    }

    /// Opens the production journal only after descriptor-relative ownership,
    /// link-count and mode validation. The source DB is never shared with
    /// receiver custody or chat relay state.
    pub(crate) fn open(
        config: AnonymousMailboxSourceConfig,
        journal_key: [u8; 32],
    ) -> Result<Self, AnonymousMailboxSourceError> {
        if !source_config_is_valid(&config)
            || config.db_path.is_empty()
            || config.db_path == ":memory:"
            || journal_key == [0; 32]
        {
            return Err(AnonymousMailboxSourceError::Rejected);
        }

        let target = prepare_private_sqlite_target(Path::new(&config.db_path))
            .map_err(map_private_sqlite_error)?;
        let mut flags = OpenFlags::SQLITE_OPEN_READ_WRITE;
        #[cfg(unix)]
        {
            flags |= OpenFlags::SQLITE_OPEN_NOFOLLOW;
        }
        let mut connection = Connection::open_with_flags(&target.resolved_path, flags)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        verify_private_file(&target.resolved_path, false).map_err(map_private_sqlite_error)?;
        restrict_private_source_permissions(&target.resolved_path)?;
        verify_private_file(&target.resolved_path, true).map_err(map_private_sqlite_error)?;
        connection
            .busy_timeout(Duration::from_secs(5))
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        verify_source_sqlite_integrity(&connection)?;
        configure_source_full_durability(&connection)?;
        connection
            .execute_batch("PRAGMA foreign_keys=ON; PRAGMA trusted_schema=OFF;")
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        initialize_or_verify_source_schema(
            &mut connection,
            source_now_secs()?,
            config.terminal_retention_secs,
        )?;
        let journal = Self {
            connection: Mutex::new(connection),
            journal_key,
            max_entries: config.max_journal_entries,
            max_bytes: config.max_journal_bytes,
            terminal_retention_secs: config.terminal_retention_secs,
            cleanup_batch_size: config.cleanup_batch_size,
            #[cfg(unix)]
            _database_parent: Some(target.parent),
        };
        journal.audit_startup()?;
        Ok(journal)
    }

    pub(super) fn load(
        &self,
        route_id: &[u8; 16],
    ) -> Result<Option<SourceJournalRecord>, AnonymousMailboxSourceError> {
        let connection = self.connection.lock();
        let lengths: Option<(i64, i64, i64, i64, i64, i64)> = connection
            .query_row(
                "SELECT length(request_commitment), length(target_node_id),
                        length(descriptor_commitment), length(body), length(state_nonce),
                        length(protected_state)
                 FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![route_id.as_slice()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let Some((request_len, target_len, descriptor_len, body_len, nonce_len, protected_len)) =
            lengths
        else {
            return Ok(None);
        };
        let stored_bytes = u64::try_from(body_len).ok().and_then(|body| {
            u64::try_from(protected_len)
                .ok()
                .and_then(|protected| body.checked_add(protected))
        });
        if request_len != 32
            || target_len != 32
            || descriptor_len < 1
            || usize::try_from(descriptor_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_DESCRIPTOR_BYTES)
                .is_none()
            || body_len < 1
            || usize::try_from(body_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_BODY_BYTES)
                .is_none()
            || nonce_len != JOURNAL_NONCE_BYTES as i64
            || protected_len < JOURNAL_AEAD_TAG_BYTES as i64
            || usize::try_from(protected_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_PROTECTED_STATE_BYTES)
                .is_none()
            || stored_bytes
                .filter(|value| *value <= self.max_bytes)
                .is_none()
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        connection
            .query_row(
                "SELECT request_commitment, target_node_id, descriptor_commitment, body, phase,
                        retain_until, state_nonce, protected_state
                 FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![route_id.as_slice()],
                |row| {
                    let request_commitment: Vec<u8> = row.get(0)?;
                    let target_node_id: Vec<u8> = row.get(1)?;
                    let descriptor: Vec<u8> = row.get(2)?;
                    let body: Vec<u8> = row.get(3)?;
                    let phase: i64 = row.get(4)?;
                    let retain_until: Option<i64> = row.get(5)?;
                    let nonce: Vec<u8> = row.get(6)?;
                    let protected: Vec<u8> = row.get(7)?;
                    Ok((
                        request_commitment,
                        target_node_id,
                        descriptor,
                        body,
                        phase,
                        retain_until,
                        nonce,
                        protected,
                    ))
                },
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?
            .map(
                |(request, target, descriptor, body, phase, retain_until, nonce, protected)| {
                    let request_commitment = fixed::<32>(&request)?;
                    let target_node_id = fixed::<32>(&target)?;
                    let descriptor_commitment = bincode::deserialize(&descriptor)
                        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
                    let phase = AnonymousMailboxSourcePhase::decode(phase)?;
                    let retain_until = retain_until.map(source_u64).transpose()?;
                    validate_phase_retention(phase, retain_until)?;
                    let state = self.open_state(
                        route_id,
                        &request_commitment,
                        &target_node_id,
                        &body,
                        &nonce,
                        &protected,
                    )?;
                    // [ANONYMOUS-MAILBOX-SOURCE-WIRING 2026-09-03 by Codex]
                    // The encrypted state binds every durable projection to
                    // the exact serialized carrier. Old rows lacking this
                    // commitment, body swaps, or descriptor/frame drift fail
                    // before a restart can release a network dispatch.
                    let decoded = decode_state(&state)?;
                    if decoded.body_commitment != source_body_commitment(&body)
                        || source_request_commitment(
                            route_id,
                            &target_node_id,
                            &descriptor_commitment,
                            &decoded.terminal_frame,
                        ) != request_commitment
                    {
                        return Err(AnonymousMailboxSourceError::Corrupt);
                    }
                    Ok(SourceJournalRecord {
                        route_id: *route_id,
                        request_commitment,
                        target_node_id,
                        descriptor_commitment,
                        body,
                        phase,
                        retain_until,
                        state,
                    })
                },
            )
            .transpose()
    }

    pub(super) fn insert_or_exact(
        &self,
        record: &SourceJournalRecord,
    ) -> Result<SourceJournalRecord, AnonymousMailboxSourceError> {
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let existing: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT request_commitment FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
                params![record.route_id.as_slice()],
                |row| row.get(0),
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if let Some(existing) = existing {
            transaction
                .commit()
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            if fixed::<32>(&existing)? != record.request_commitment {
                return Err(AnonymousMailboxSourceError::Conflict);
            }
            drop(connection);
            let loaded = self
                .load(&record.route_id)?
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
            if loaded.target_node_id != record.target_node_id
                || loaded.descriptor_commitment != record.descriptor_commitment
            {
                return Err(AnonymousMailboxSourceError::Conflict);
            }
            return Ok(loaded);
        }
        validate_source_record_projection(record)?;
        let meta = load_source_meta(&transaction)?;
        let (nonce, protected) = self.seal_state(
            &record.route_id,
            &record.request_commitment,
            &record.target_node_id,
            &record.state,
        )?;
        if record.body.is_empty() || record.body.len() > MAX_JOURNAL_BODY_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let incoming = record
            .body
            .len()
            .checked_add(protected.len())
            .ok_or(AnonymousMailboxSourceError::Unavailable)?;
        let incoming =
            u64::try_from(incoming).map_err(|_| AnonymousMailboxSourceError::Rejected)?;
        let updated_meta = SourceJournalMeta {
            entries: meta
                .entries
                .checked_add(1)
                .ok_or(AnonymousMailboxSourceError::Rejected)?,
            bytes: meta
                .bytes
                .checked_add(incoming)
                .ok_or(AnonymousMailboxSourceError::Rejected)?,
        };
        if updated_meta.entries
            > u64::try_from(self.max_entries).map_err(|_| AnonymousMailboxSourceError::Corrupt)?
            || updated_meta.bytes > self.max_bytes
        {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let descriptor = bincode::serialize(&record.descriptor_commitment)
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
        if descriptor.is_empty() || descriptor.len() > MAX_JOURNAL_DESCRIPTOR_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        transaction
            .execute(
                "INSERT INTO anonymous_mailbox_source_journal
                   (route_id, request_commitment, target_node_id, descriptor_commitment, body, phase,
                    retain_until, state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    record.route_id.as_slice(), record.request_commitment.as_slice(),
                    record.target_node_id.as_slice(), descriptor, record.body,
                    record.phase.code(), record.retain_until.map(source_i64).transpose()?, nonce,
                    protected
                ],
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        update_source_meta_exact(&transaction, meta, updated_meta)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        #[cfg(test)]
        crash_after_source_journal_commit(record.phase);
        Ok(SourceJournalRecord {
            route_id: record.route_id,
            request_commitment: record.request_commitment,
            target_node_id: record.target_node_id,
            descriptor_commitment: record.descriptor_commitment,
            body: record.body.clone(),
            phase: record.phase,
            retain_until: record.retain_until,
            state: record.state.clone(),
        })
    }

    pub(super) fn transition(
        &self,
        record: &SourceJournalRecord,
        expected: AnonymousMailboxSourcePhase,
        phase: AnonymousMailboxSourcePhase,
        state: Vec<u8>,
    ) -> Result<(), AnonymousMailboxSourceError> {
        self.transition_at(record, expected, phase, state, source_now_secs()?)
    }

    pub(super) fn transition_at(
        &self,
        record: &SourceJournalRecord,
        expected: AnonymousMailboxSourcePhase,
        phase: AnonymousMailboxSourcePhase,
        state: Vec<u8>,
        transitioned_at: u64,
    ) -> Result<(), AnonymousMailboxSourceError> {
        validate_source_record_projection(record)?;
        let desired_decoded = decode_state(&state)?;
        if desired_decoded.body_commitment != source_body_commitment(&record.body)
            || desired_decoded.terminal_frame != decode_state(&record.state)?.terminal_frame
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let desired = self.seal_state(
            &record.route_id,
            &record.request_commitment,
            &record.target_node_id,
            &state,
        )?;
        let compact_ambiguous = if phase == AnonymousMailboxSourcePhase::Completed {
            let decoded = decode_state(&state)?;
            if decoded.body_commitment != source_body_commitment(&record.body) {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let compact_state = encode_state(&record.body, &decoded.terminal_frame, None, None)?;
            Some(self.seal_state(
                &record.route_id,
                &record.request_commitment,
                &record.target_node_id,
                &compact_state,
            )?)
        } else {
            None
        };

        // [ANONYMOUS-MAILBOX-SOURCE-BOUNDS 2026-09-03 by Codex] The phase
        // check, replacement accounting, fallback selection, and update share
        // one write transaction. A valid but unretainable response consumes
        // the one-shot session into a compact Ambiguous state rather than
        // creating a row that load() must later reject as corrupt.
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let current: Option<(i64, i64, i64, Option<i64>)> = transaction
            .query_row(
                "SELECT phase, length(body), length(protected_state), retain_until
                 FROM anonymous_mailbox_source_journal
                 WHERE route_id = ?1 AND request_commitment = ?2 AND target_node_id = ?3",
                params![
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    record.target_node_id.as_slice()
                ],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let Some((current_phase, body_len, old_protected_len, current_retain_until)) = current
        else {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        };
        let current_phase = AnonymousMailboxSourcePhase::decode(current_phase)?;
        let current_retain_until = current_retain_until.map(source_u64).transpose()?;
        validate_phase_retention(current_phase, current_retain_until)?;
        if current_phase != expected {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        if body_len < 1
            || usize::try_from(body_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_BODY_BYTES)
                .is_none()
            || old_protected_len < JOURNAL_AEAD_TAG_BYTES as i64
            || usize::try_from(old_protected_len)
                .ok()
                .filter(|value| *value <= MAX_JOURNAL_PROTECTED_STATE_BYTES)
                .is_none()
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let meta = load_source_meta(&transaction)?;
        let old_row_bytes = u64::try_from(body_len)
            .ok()
            .and_then(|body| {
                u64::try_from(old_protected_len)
                    .ok()
                    .and_then(|protected| body.checked_add(protected))
            })
            .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        let retained_without_row = meta
            .bytes
            .checked_sub(old_row_bytes)
            .ok_or(AnonymousMailboxSourceError::Corrupt)?;

        let desired_row_bytes = u64::try_from(body_len)
            .ok()
            .and_then(|body| {
                u64::try_from(desired.1.len())
                    .ok()
                    .and_then(|protected| body.checked_add(protected))
            })
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        let desired_used = retained_without_row
            .checked_add(desired_row_bytes)
            .ok_or(AnonymousMailboxSourceError::Rejected)?;
        let (stored_phase, nonce, protected, fell_back) = if desired_used <= self.max_bytes {
            (phase, desired.0, desired.1, false)
        } else if let Some((nonce, protected)) = compact_ambiguous {
            let fallback_row_bytes = u64::try_from(body_len)
                .ok()
                .and_then(|body| {
                    u64::try_from(protected.len())
                        .ok()
                        .and_then(|protected| body.checked_add(protected))
                })
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
            if retained_without_row
                .checked_add(fallback_row_bytes)
                .filter(|value| *value <= self.max_bytes)
                .is_none()
            {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            (
                AnonymousMailboxSourcePhase::Ambiguous,
                nonce,
                protected,
                true,
            )
        } else {
            return Err(AnonymousMailboxSourceError::Rejected);
        };
        let retain_until = if matches!(
            stored_phase,
            AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
        ) {
            Some(
                transitioned_at
                    .checked_add(self.terminal_retention_secs)
                    .filter(|value| *value <= i64::MAX as u64)
                    .ok_or(AnonymousMailboxSourceError::Rejected)?,
            )
        } else {
            None
        };
        validate_phase_retention(stored_phase, retain_until)?;
        let updated_meta = SourceJournalMeta {
            entries: meta.entries,
            bytes: retained_without_row
                .checked_add(
                    u64::try_from(body_len)
                        .ok()
                        .and_then(|body| {
                            u64::try_from(protected.len())
                                .ok()
                                .and_then(|protected| body.checked_add(protected))
                        })
                        .ok_or(AnonymousMailboxSourceError::Corrupt)?,
                )
                .ok_or(AnonymousMailboxSourceError::Corrupt)?,
        };

        let updated = transaction
            .execute(
                "UPDATE anonymous_mailbox_source_journal
                 SET phase = ?1, retain_until = ?2, state_nonce = ?3, protected_state = ?4
                 WHERE route_id = ?5 AND request_commitment = ?6 AND target_node_id = ?7
                       AND phase = ?8",
                params![
                    stored_phase.code(),
                    retain_until.map(source_i64).transpose()?,
                    nonce,
                    protected,
                    record.route_id.as_slice(),
                    record.request_commitment.as_slice(),
                    record.target_node_id.as_slice(),
                    expected.code()
                ],
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if updated != 1 {
            return Err(AnonymousMailboxSourceError::Ambiguous);
        }
        update_source_meta_exact(&transaction, meta, updated_meta)?;
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        #[cfg(test)]
        crash_after_source_journal_commit(stored_phase);
        if fell_back {
            Err(AnonymousMailboxSourceError::Ambiguous)
        } else {
            Ok(())
        }
    }

    /// Reclaims only terminal source rows whose byte-identical replay window ended.
    ///
    /// Prepared, armed, and ambiguous rows are durable safety fences and are
    /// never selected, regardless of age. The aggregate report contains no
    /// route, target, commitment, path, or ciphertext information.
    pub(crate) fn cleanup_terminal_records(
        &self,
        now: u64,
    ) -> Result<AnonymousMailboxSourceCleanupReport, AnonymousMailboxSourceError> {
        let now_sql = source_i64(now)?;
        let limit = source_i64(
            u64::try_from(self.cleanup_batch_size)
                .map_err(|_| AnonymousMailboxSourceError::Rejected)?,
        )?;
        let mut connection = self.connection.lock();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        let before = load_source_meta(&transaction)?;
        let rows = {
            let mut statement = transaction
                .prepare(
                    "SELECT route_id, phase, retain_until, length(body), length(protected_state)
                     FROM anonymous_mailbox_source_journal
                     WHERE phase IN (3, 5) AND retain_until < ?1
                     ORDER BY retain_until, route_id LIMIT ?2",
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            let mapped = statement
                .query_map(params![now_sql, limit], |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                })
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            mapped
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?
        };
        let mut report = AnonymousMailboxSourceCleanupReport::default();
        for (route_id, phase, retain_until, body_len, protected_len) in rows {
            let route_id = fixed::<16>(&route_id)?;
            let phase = AnonymousMailboxSourcePhase::decode(phase)?;
            let retain_until = retain_until.map(source_u64).transpose()?;
            validate_phase_retention(phase, retain_until)?;
            if !matches!(
                phase,
                AnonymousMailboxSourcePhase::Completed | AnonymousMailboxSourcePhase::Rejected
            ) || retain_until.filter(|deadline| *deadline < now).is_none()
            {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let row_bytes = source_row_bytes(body_len, protected_len)?;
            let removed = transaction
                .execute(
                    "DELETE FROM anonymous_mailbox_source_journal
                     WHERE route_id = ?1 AND phase = ?2 AND retain_until = ?3",
                    params![
                        route_id.as_slice(),
                        phase.code(),
                        retain_until.map(source_i64).transpose()?
                    ],
                )
                .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
            if removed != 1 {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            report.rows_removed = report
                .rows_removed
                .checked_add(1)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
            report.bytes_removed = report
                .bytes_removed
                .checked_add(row_bytes)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        }
        let after = SourceJournalMeta {
            entries: before
                .entries
                .checked_sub(report.rows_removed)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?,
            bytes: before
                .bytes
                .checked_sub(report.bytes_removed)
                .ok_or(AnonymousMailboxSourceError::Corrupt)?,
        };
        if after != before {
            update_source_meta_exact(&transaction, before, after)?;
        }
        if compute_source_meta(&transaction)? != after {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        transaction
            .commit()
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        Ok(report)
    }

    pub(super) fn audit_startup(&self) -> Result<(), AnonymousMailboxSourceError> {
        let route_ids = {
            let connection = self.connection.lock();
            let stored = load_source_meta(&connection)?;
            let observed = compute_source_meta(&connection)?;
            if stored != observed
                || stored.entries
                    > u64::try_from(self.max_entries)
                        .map_err(|_| AnonymousMailboxSourceError::Corrupt)?
                || stored.bytes > self.max_bytes
            {
                return Err(AnonymousMailboxSourceError::Corrupt);
            }
            let mut statement = connection
                .prepare("SELECT route_id FROM anonymous_mailbox_source_journal ORDER BY route_id")
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            let rows = statement
                .query_map([], |row| row.get::<_, Vec<u8>>(0))
                .map_err(|_| AnonymousMailboxSourceError::Corrupt)?;
            rows.map(|row| {
                row.map_err(|_| AnonymousMailboxSourceError::Corrupt)
                    .and_then(|bytes| fixed::<16>(&bytes))
            })
            .collect::<Result<Vec<_>, _>>()?
        };
        for route_id in route_ids {
            self.load(&route_id)?
                .ok_or(AnonymousMailboxSourceError::Corrupt)?;
        }
        Ok(())
    }

    pub(super) fn seal_state(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        target: &[u8; 32],
        state: &[u8],
    ) -> Result<(Vec<u8>, Vec<u8>), AnonymousMailboxSourceError> {
        if state.len() > MAX_JOURNAL_CLEAR_STATE_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let decoded = decode_state(state)?;
        if decoded.body_commitment == [0; JOURNAL_STATE_BODY_COMMITMENT_BYTES] {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        let mut nonce = [0u8; JOURNAL_NONCE_BYTES];
        OsRng.fill_bytes(&mut nonce);
        let cipher = XChaCha20Poly1305::new(Key::from_slice(&self.journal_key));
        let protected = cipher
            .encrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: state,
                    aad: &journal_aad(
                        route_id,
                        request_commitment,
                        target,
                        &decoded.body_commitment,
                    ),
                },
            )
            .map_err(|_| AnonymousMailboxSourceError::Unavailable)?;
        if protected.len() > MAX_JOURNAL_PROTECTED_STATE_BYTES {
            return Err(AnonymousMailboxSourceError::Rejected);
        }
        Ok((nonce.to_vec(), protected))
    }

    pub(super) fn open_state(
        &self,
        route_id: &[u8; 16],
        request_commitment: &[u8; 32],
        target: &[u8; 32],
        body: &[u8],
        nonce: &[u8],
        protected: &[u8],
    ) -> Result<Vec<u8>, AnonymousMailboxSourceError> {
        if protected.len() < JOURNAL_AEAD_TAG_BYTES
            || protected.len() > MAX_JOURNAL_PROTECTED_STATE_BYTES
        {
            return Err(AnonymousMailboxSourceError::Corrupt);
        }
        let nonce = fixed::<JOURNAL_NONCE_BYTES>(nonce)?;
        XChaCha20Poly1305::new(Key::from_slice(&self.journal_key))
            .decrypt(
                XNonce::from_slice(&nonce),
                Payload {
                    msg: protected,
                    aad: &journal_aad(
                        route_id,
                        request_commitment,
                        target,
                        &source_body_commitment(body),
                    ),
                },
            )
            .map_err(|_| AnonymousMailboxSourceError::Corrupt)
    }
}

fn source_config_is_valid(config: &AnonymousMailboxSourceConfig) -> bool {
    config.enabled
        && config.max_journal_entries > 0
        && i64::try_from(config.max_journal_entries).is_ok()
        && config.max_journal_bytes > 0
        && config.max_journal_bytes <= i64::MAX as u64
        && config.max_in_flight > 0
        && config.request_timeout_secs > 0
        && config.terminal_retention_secs > 0
        && config.terminal_retention_secs <= MAX_ANONYMOUS_MAILBOX_SOURCE_TERMINAL_RETENTION_SECS
        && config.cleanup_batch_size > 0
        && config.cleanup_batch_size <= 4_096
}

fn source_now_secs() -> Result<u64, AnonymousMailboxSourceError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|_| AnonymousMailboxSourceError::Unavailable)
}

#[cfg(test)]
mod tests;
