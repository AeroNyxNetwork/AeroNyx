// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_cleanup.rs
// ============================================
// Version: 1.1.0-CleanupExecutionComposition
//
// Creation Reason:
//   [CHAT-RELAY-CLEANUP-DOMAIN 2026-08-25 by Codex] Extract bounded retention
//   policy, expired-row validation, notification generation, private replay
//   cleanup, and backlog detection from the oversized relay service.
//
// Modification Reason:
//   [CHAT-CLEANUP-EXECUTION-DOMAIN 2026-08-28 by Codex] Updated ownership after
//   transaction batching and partial-progress handling moved to an executor.
//
// Main Functionality:
//   - Models immutable cleanup policy and one run's stable TTL cutoffs.
//   - Defines typed cleanup targets instead of accepting table-name strings.
//   - Defines a replaceable repository trait for bounded durable operations.
//   - Validates signed pending envelopes before generating expiry controls.
//   - Composes durable quarantine into the same caller-owned transaction.
//
// Dependencies:
//   - `config_chat_relay.rs` supplies validated retention durations.
//   - `chat_relay_cleanup_execution.rs` owns locking, commits, and run budget.
//   - `chat_relay.rs` owns scheduling, telemetry, logs, and public APIs.
//   - `chat_relay_quarantine.rs` owns privacy-minimised poison-row evidence.
//   - `rusqlite` provides the production durable repository implementation.
//
// Main Logical Flow:
//   1. Freeze all retention cutoffs once at the start of a cleanup run.
//   2. Read and validate one bounded oldest-first pending-message batch.
//   3. Queue bounded expiry controls and quarantine corrupt rows atomically.
//   4. Delete bounded stale blobs, controls, and private replay evidence.
//   5. Report aggregate counts and whether another transaction is required.
//
// Important Note for Next Developer:
//   - The execution capability must commit or roll back the immediate transaction.
//   - Never log or expose message IDs, wallets, blobs, replay keys, or payloads.
//   - Cleanup targets are a closed enum; do not reintroduce dynamic SQL names.
//   - Out-of-range TTLs retain data rather than expiring potentially fresh rows.
//   - Preserve oldest-first limits and expiry notification chunking semantics.
//
// Last Modified:
//   v1.1.0-CleanupExecutionComposition - Documented executor ownership
//   v1.0.0-BoundedCleanupDomain - Initial policy/repository composition
// ============================================

use std::collections::HashMap;

use aeronyx_core::protocol::chat::decode_envelope;
use rusqlite::{params, Transaction};

use crate::config::ChatRelayConfig;

use super::chat_relay::{
    MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION, MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES,
};
// [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Cleanup depends directly on
// the typed failure boundary while protocol limits remain service-owned.
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};
use super::chat_relay_quarantine::{
    CorruptDurableRow, DurableQuarantineDomain, QUARANTINE_SOURCE_PENDING_MESSAGE,
};

/// Maximum expired pending-message rows handled by one transaction.
pub(crate) const CLEANUP_MESSAGE_BATCH_SIZE: usize = 1024;
/// Maximum expired encrypted blobs deleted by one transaction.
const CLEANUP_BLOB_BATCH_SIZE: usize = 128;
/// Maximum delivered or stale expiry controls deleted by one transaction.
const CLEANUP_NOTIFICATION_BATCH_SIZE: usize = 1024;
/// Maximum rows deleted from each private replay table per transaction.
const CLEANUP_PRIVATE_REPLAY_BATCH_SIZE: usize = 1024;
/// Maximum transactions executed by one scheduled maintenance run.
pub(crate) const CLEANUP_MAX_BATCHES_PER_RUN: usize = 8;

type ExpiredMessagesBySender = HashMap<[u8; 32], HashMap<[u8; 32], Vec<[u8; 16]>>>;

/// Immutable retention and batch policy for one relay process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelayCleanupPolicy {
    offline_ttl_secs: u64,
    notification_ttl_secs: u64,
    verified_submit_ttl_secs: u64,
    blind_route_ttl_secs: u64,
    message_batch_size: usize,
    blob_batch_size: usize,
    notification_batch_size: usize,
    private_replay_batch_size: usize,
}

impl RelayCleanupPolicy {
    pub(crate) fn new(
        config: &ChatRelayConfig,
        verified_submit_ttl_secs: u64,
        blind_route_ttl_secs: u64,
    ) -> Self {
        Self {
            offline_ttl_secs: config.offline_ttl_secs,
            notification_ttl_secs: config.expired_notification_ttl_secs,
            verified_submit_ttl_secs,
            blind_route_ttl_secs,
            message_batch_size: CLEANUP_MESSAGE_BATCH_SIZE,
            blob_batch_size: CLEANUP_BLOB_BATCH_SIZE,
            notification_batch_size: CLEANUP_NOTIFICATION_BATCH_SIZE,
            private_replay_batch_size: CLEANUP_PRIVATE_REPLAY_BATCH_SIZE,
        }
    }

    fn cutoffs(self, now: i64) -> RelayCleanupCutoffs {
        RelayCleanupCutoffs {
            pending: retention_cutoff(now, self.offline_ttl_secs),
            notification: retention_cutoff(now, self.notification_ttl_secs),
            verified_submit: retention_cutoff(now, self.verified_submit_ttl_secs),
            blind_route: retention_cutoff(now, self.blind_route_ttl_secs),
        }
    }
}

fn retention_cutoff(now: i64, ttl_secs: u64) -> i64 {
    // Configuration validation rejects values above i64::MAX. Retaining data
    // is the fail-closed behavior when a service is constructed directly.
    now.saturating_sub(i64::try_from(ttl_secs).unwrap_or(i64::MAX))
}

/// Stable cutoffs shared by every transaction in one cleanup run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelayCleanupCutoffs {
    pending: i64,
    notification: i64,
    verified_submit: i64,
    blind_route: i64,
}

/// Closed set of node-private replay stores eligible for retention cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrivateReplayCleanupTarget {
    VerifiedSubmitResponse,
    VerifiedSubmitReservation,
    BlindRouteResponse,
    BlindRouteReservation,
}

impl PrivateReplayCleanupTarget {
    fn delete_sql(self) -> &'static str {
        match self {
            Self::VerifiedSubmitResponse => {
                "DELETE FROM relay_verified_submit_responses
                 WHERE rowid IN (
                     SELECT rowid FROM relay_verified_submit_responses
                     WHERE completed_at < ?1
                     ORDER BY completed_at ASC, rowid ASC
                     LIMIT ?2
                 )"
            }
            Self::VerifiedSubmitReservation => {
                "DELETE FROM relay_verified_submit_reservations
                 WHERE rowid IN (
                     SELECT rowid FROM relay_verified_submit_reservations
                     WHERE reserved_at < ?1
                     ORDER BY reserved_at ASC, rowid ASC
                     LIMIT ?2
                 )"
            }
            Self::BlindRouteResponse => {
                "DELETE FROM relay_blind_route_responses
                 WHERE rowid IN (
                     SELECT rowid FROM relay_blind_route_responses
                     WHERE completed_at < ?1
                     ORDER BY completed_at ASC, rowid ASC
                     LIMIT ?2
                 )"
            }
            Self::BlindRouteReservation => {
                "DELETE FROM relay_blind_route_reservations
                 WHERE rowid IN (
                     SELECT rowid FROM relay_blind_route_reservations
                     WHERE reserved_at < ?1
                     ORDER BY reserved_at ASC, rowid ASC
                     LIMIT ?2
                 )"
            }
        }
    }
}

#[derive(Debug)]
struct ExpiredMessageRow {
    message_id: [u8; 16],
    sender: [u8; 32],
    receiver: [u8; 32],
}

#[derive(Debug)]
pub(crate) struct StoredExpiredMessageRow {
    rowid: i64,
    message_id: Vec<u8>,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    timestamp: i64,
    envelope: Vec<u8>,
    queue_sequence: Option<i64>,
}

#[derive(Debug, Default)]
struct ValidatedExpiredMessageBatch {
    valid_rows: Vec<ExpiredMessageRow>,
    corrupt_rows: Vec<CorruptDurableRow>,
    selected_rowids: Vec<i64>,
}

/// Aggregate result of one committed cleanup transaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CleanupBatchOutcome {
    pub(crate) expired_messages: usize,
    pub(crate) expired_blobs: usize,
    pub(crate) removed_notifications: usize,
    pub(crate) quarantined_pending_messages: usize,
    pub(crate) removed_quarantine_events: usize,
    pub(crate) removed_verified_submit_responses: usize,
    pub(crate) removed_verified_submit_reservations: usize,
    pub(crate) removed_blind_route_responses: usize,
    pub(crate) removed_blind_route_reservations: usize,
    pub(crate) retained_quarantine_events: usize,
    pub(crate) has_more: bool,
}

/// Aggregate result of one bounded multi-transaction maintenance run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CleanupRunSummary {
    pub(crate) expired_messages: usize,
    pub(crate) expired_blobs: usize,
    pub(crate) removed_notifications: usize,
    pub(crate) quarantined_pending_messages: usize,
    pub(crate) removed_quarantine_events: usize,
    pub(crate) removed_verified_submit_responses: usize,
    pub(crate) removed_verified_submit_reservations: usize,
    pub(crate) removed_blind_route_responses: usize,
    pub(crate) removed_blind_route_reservations: usize,
    pub(crate) retained_quarantine_events: usize,
    pub(crate) successful_batches: usize,
    pub(crate) backlog_deferred: bool,
}

impl CleanupRunSummary {
    pub(crate) fn absorb(&mut self, batch: CleanupBatchOutcome) {
        self.expired_messages = self.expired_messages.saturating_add(batch.expired_messages);
        self.expired_blobs = self.expired_blobs.saturating_add(batch.expired_blobs);
        self.removed_notifications = self
            .removed_notifications
            .saturating_add(batch.removed_notifications);
        self.quarantined_pending_messages = self
            .quarantined_pending_messages
            .saturating_add(batch.quarantined_pending_messages);
        self.removed_quarantine_events = self
            .removed_quarantine_events
            .saturating_add(batch.removed_quarantine_events);
        self.removed_verified_submit_responses = self
            .removed_verified_submit_responses
            .saturating_add(batch.removed_verified_submit_responses);
        self.removed_verified_submit_reservations = self
            .removed_verified_submit_reservations
            .saturating_add(batch.removed_verified_submit_reservations);
        self.removed_blind_route_responses = self
            .removed_blind_route_responses
            .saturating_add(batch.removed_blind_route_responses);
        self.removed_blind_route_reservations = self
            .removed_blind_route_reservations
            .saturating_add(batch.removed_blind_route_reservations);
        self.retained_quarantine_events = batch.retained_quarantine_events;
        self.successful_batches = self.successful_batches.saturating_add(1);
    }

    pub(crate) fn removed_anything(self) -> bool {
        self.expired_messages > 0
            || self.expired_blobs > 0
            || self.removed_notifications > 0
            || self.quarantined_pending_messages > 0
            || self.removed_quarantine_events > 0
            || self.removed_verified_submit_responses > 0
            || self.removed_verified_submit_reservations > 0
            || self.removed_blind_route_responses > 0
            || self.removed_blind_route_reservations > 0
    }
}

/// Replaceable persistence capability for bounded relay retention.
pub(crate) trait RelayCleanupRepository: Send + Sync {
    fn read_expired_messages(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredExpiredMessageRow>>;

    fn insert_expiry_notification(
        &self,
        tx: &Transaction<'_>,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        message_ids: &[u8],
        now: i64,
    ) -> ChatRelayResult<()>;

    fn delete_expired_messages(&self, tx: &Transaction<'_>, rowids: &[i64]) -> ChatRelayResult<()>;

    fn delete_expired_blobs(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize>;

    fn delete_stale_notifications(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize>;

    fn delete_stale_private_replay(
        &self,
        tx: &Transaction<'_>,
        target: PrivateReplayCleanupTarget,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize>;

    fn backlog_exists(
        &self,
        tx: &Transaction<'_>,
        cutoffs: RelayCleanupCutoffs,
    ) -> ChatRelayResult<bool>;
}

/// Production SQLite implementation for bounded relay retention.
pub(crate) struct SqliteRelayCleanupRepository;

impl RelayCleanupRepository for SqliteRelayCleanupRepository {
    fn read_expired_messages(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredExpiredMessageRow>> {
        let mut statement = tx.prepare(
            "SELECT rowid, message_id, sender, receiver, timestamp, envelope, queue_sequence
             FROM pending_messages
             WHERE status = 0 AND received_at < ?1
             ORDER BY received_at ASC, message_id ASC
             LIMIT ?2",
        )?;
        let rows = statement
            .query_map(params![cutoff, limit], |row| {
                Ok(StoredExpiredMessageRow {
                    rowid: row.get(0)?,
                    message_id: row.get(1)?,
                    sender: row.get(2)?,
                    receiver: row.get(3)?,
                    timestamp: row.get(4)?,
                    envelope: row.get(5)?,
                    queue_sequence: row.get(6)?,
                })
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    fn insert_expiry_notification(
        &self,
        tx: &Transaction<'_>,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        message_ids: &[u8],
        now: i64,
    ) -> ChatRelayResult<()> {
        tx.execute(
            "INSERT INTO expired_notifications
             (sender, receiver, message_ids, created_at, pushed)
             VALUES (?1, ?2, ?3, ?4, 0)",
            params![sender.as_slice(), receiver.as_slice(), message_ids, now],
        )?;
        Ok(())
    }

    fn delete_expired_messages(&self, tx: &Transaction<'_>, rowids: &[i64]) -> ChatRelayResult<()> {
        let mut statement = tx.prepare("DELETE FROM pending_messages WHERE rowid = ?1")?;
        let mut deleted = 0usize;
        for rowid in rowids {
            deleted = deleted.saturating_add(statement.execute(params![rowid])?);
        }
        if deleted != rowids.len() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_message_cleanup_count",
            });
        }
        Ok(())
    }

    fn delete_expired_blobs(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize> {
        Ok(tx.execute(
            "DELETE FROM pending_blobs
             WHERE rowid IN (
                 SELECT rowid FROM pending_blobs
                 WHERE received_at < ?1
                 ORDER BY received_at ASC, rowid ASC
                 LIMIT ?2
             )",
            params![cutoff, limit],
        )?)
    }

    fn delete_stale_notifications(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize> {
        Ok(tx.execute(
            "DELETE FROM expired_notifications
             WHERE id IN (
                 SELECT id FROM expired_notifications
                 WHERE pushed = 1 OR created_at < ?1
                 ORDER BY id ASC
                 LIMIT ?2
             )",
            params![cutoff, limit],
        )?)
    }

    fn delete_stale_private_replay(
        &self,
        tx: &Transaction<'_>,
        target: PrivateReplayCleanupTarget,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<usize> {
        Ok(tx.execute(target.delete_sql(), params![cutoff, limit])?)
    }

    fn backlog_exists(
        &self,
        tx: &Transaction<'_>,
        cutoffs: RelayCleanupCutoffs,
    ) -> ChatRelayResult<bool> {
        let has_more = tx.query_row(
            "SELECT
                 EXISTS(
                     SELECT 1 FROM pending_messages
                     WHERE status = 0 AND received_at < ?1
                 )
                 OR EXISTS(
                     SELECT 1 FROM pending_blobs WHERE received_at < ?1
                 )
                 OR EXISTS(
                     SELECT 1 FROM expired_notifications
                     WHERE pushed = 1 OR created_at < ?2
                 )
                 OR EXISTS(
                     SELECT 1 FROM relay_verified_submit_responses
                     WHERE completed_at < ?3
                 )
                 OR EXISTS(
                     SELECT 1 FROM relay_verified_submit_reservations
                     WHERE reserved_at < ?3
                 )
                 OR EXISTS(
                     SELECT 1 FROM relay_blind_route_responses
                     WHERE completed_at < ?4
                 )
                 OR EXISTS(
                     SELECT 1 FROM relay_blind_route_reservations
                     WHERE reserved_at < ?4
                 )",
            params![
                cutoffs.pending,
                cutoffs.notification,
                cutoffs.verified_submit,
                cutoffs.blind_route,
            ],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(has_more != 0)
    }
}

/// Composed cleanup policy, validation, and persistence capability.
pub(crate) struct RelayCleanupDomain<R = SqliteRelayCleanupRepository> {
    repository: R,
    policy: RelayCleanupPolicy,
}

impl RelayCleanupDomain<SqliteRelayCleanupRepository> {
    pub(crate) fn new(
        config: &ChatRelayConfig,
        verified_submit_ttl_secs: u64,
        blind_route_ttl_secs: u64,
    ) -> Self {
        Self::with_repository(
            SqliteRelayCleanupRepository,
            RelayCleanupPolicy::new(config, verified_submit_ttl_secs, blind_route_ttl_secs),
        )
    }
}

impl<R: RelayCleanupRepository> RelayCleanupDomain<R> {
    fn with_repository(repository: R, policy: RelayCleanupPolicy) -> Self {
        Self { repository, policy }
    }

    pub(crate) fn cutoffs(&self, now: i64) -> RelayCleanupCutoffs {
        self.policy.cutoffs(now)
    }

    pub(crate) fn run_batch(
        &self,
        tx: &Transaction<'_>,
        quarantine: &DurableQuarantineDomain,
        now: i64,
        cutoffs: RelayCleanupCutoffs,
    ) -> ChatRelayResult<CleanupBatchOutcome> {
        let expired_batch = self.load_and_validate_expired_messages(tx, cutoffs.pending)?;
        let expired_message_count = expired_batch.valid_rows.len();
        let quarantined_pending_messages = expired_batch.corrupt_rows.len();

        self.queue_expiry_notifications(tx, now, &expired_batch.valid_rows)?;
        quarantine.record(tx, now, &expired_batch.corrupt_rows)?;
        self.repository
            .delete_expired_messages(tx, &expired_batch.selected_rowids)?;

        let expired_blobs = self.repository.delete_expired_blobs(
            tx,
            cutoffs.pending,
            batch_limit(self.policy.blob_batch_size),
        )?;
        let removed_notifications = self.repository.delete_stale_notifications(
            tx,
            cutoffs.notification,
            batch_limit(self.policy.notification_batch_size),
        )?;
        let quarantine_maintenance = quarantine.maintain(tx, now)?;
        let replay_limit = batch_limit(self.policy.private_replay_batch_size);
        let removed_verified_submit_responses = self.repository.delete_stale_private_replay(
            tx,
            PrivateReplayCleanupTarget::VerifiedSubmitResponse,
            cutoffs.verified_submit,
            replay_limit,
        )?;
        let removed_verified_submit_reservations = self.repository.delete_stale_private_replay(
            tx,
            PrivateReplayCleanupTarget::VerifiedSubmitReservation,
            cutoffs.verified_submit,
            replay_limit,
        )?;
        let removed_blind_route_responses = self.repository.delete_stale_private_replay(
            tx,
            PrivateReplayCleanupTarget::BlindRouteResponse,
            cutoffs.blind_route,
            replay_limit,
        )?;
        let removed_blind_route_reservations = self.repository.delete_stale_private_replay(
            tx,
            PrivateReplayCleanupTarget::BlindRouteReservation,
            cutoffs.blind_route,
            replay_limit,
        )?;
        let has_more =
            self.repository.backlog_exists(tx, cutoffs)? || quarantine.backlog_exists(tx, now)?;

        Ok(CleanupBatchOutcome {
            expired_messages: expired_message_count,
            expired_blobs,
            removed_notifications,
            quarantined_pending_messages,
            removed_quarantine_events: quarantine_maintenance.removed_events,
            removed_verified_submit_responses,
            removed_verified_submit_reservations,
            removed_blind_route_responses,
            removed_blind_route_reservations,
            retained_quarantine_events: quarantine_maintenance.retained_events,
            has_more,
        })
    }

    fn load_and_validate_expired_messages(
        &self,
        tx: &Transaction<'_>,
        cutoff: i64,
    ) -> ChatRelayResult<ValidatedExpiredMessageBatch> {
        let stored_rows = self.repository.read_expired_messages(
            tx,
            cutoff,
            batch_limit(self.policy.message_batch_size),
        )?;
        let mut batch = ValidatedExpiredMessageBatch {
            valid_rows: Vec::with_capacity(stored_rows.len()),
            corrupt_rows: Vec::new(),
            selected_rowids: Vec::with_capacity(stored_rows.len()),
        };
        for row in stored_rows {
            batch.selected_rowids.push(row.rowid);
            match validate_expired_message_row(row) {
                Ok(valid) => batch.valid_rows.push(valid),
                Err(corrupt) => batch.corrupt_rows.push(corrupt),
            }
        }
        Ok(batch)
    }

    fn queue_expiry_notifications(
        &self,
        tx: &Transaction<'_>,
        now: i64,
        expired_rows: &[ExpiredMessageRow],
    ) -> ChatRelayResult<()> {
        let mut by_sender = ExpiredMessagesBySender::new();
        for row in expired_rows {
            by_sender
                .entry(row.sender)
                .or_default()
                .entry(row.receiver)
                .or_default()
                .push(row.message_id);
        }

        for (sender, by_receiver) in &by_sender {
            for (receiver, ids) in by_receiver {
                for ids_chunk in ids.chunks(MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION) {
                    let encoded = bincode::serialize(ids_chunk)?;
                    if encoded.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
                        return Err(ChatRelayError::CorruptStoredData {
                            field: "generated_expired_notification_payload_size",
                        });
                    }
                    self.repository
                        .insert_expiry_notification(tx, sender, receiver, &encoded, now)?;
                }
            }
        }
        Ok(())
    }
}

fn batch_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX)
}

fn validate_expired_message_row(
    row: StoredExpiredMessageRow,
) -> Result<ExpiredMessageRow, CorruptDurableRow> {
    let encoded_bytes = u64::try_from(row.envelope.len()).unwrap_or(u64::MAX);
    let corrupt = |reason| CorruptDurableRow {
        row_key: row.rowid,
        source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
        reason,
        encoded_bytes,
    };
    let message_id: [u8; 16] = row
        .message_id
        .try_into()
        .map_err(|_| corrupt("expired_message_id"))?;
    let sender: [u8; 32] = row
        .sender
        .try_into()
        .map_err(|_| corrupt("expired_message_sender"))?;
    let receiver: [u8; 32] = row
        .receiver
        .try_into()
        .map_err(|_| corrupt("expired_message_receiver"))?;
    let timestamp =
        u64::try_from(row.timestamp).map_err(|_| corrupt("expired_message_timestamp"))?;
    let envelope =
        decode_envelope(&row.envelope).map_err(|_| corrupt("expired_message_envelope"))?;
    if envelope.message_id != message_id {
        return Err(corrupt("expired_message_id_mismatch"));
    }
    if envelope.sender != sender {
        return Err(corrupt("expired_message_sender_mismatch"));
    }
    if envelope.receiver != receiver {
        return Err(corrupt("expired_message_receiver_mismatch"));
    }
    if envelope.timestamp != timestamp {
        return Err(corrupt("expired_message_timestamp_mismatch"));
    }
    envelope
        .verify_signature()
        .map_err(|_| corrupt("expired_message_signature"))?;
    match row.queue_sequence {
        Some(sequence) if sequence > 0 => {}
        _ => return Err(corrupt("expired_message_queue_sequence")),
    }
    Ok(ExpiredMessageRow {
        message_id,
        sender,
        receiver,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_ttl_retains_rows_fail_closed() {
        assert_eq!(
            retention_cutoff(42, u64::MAX),
            42_i64.saturating_sub(i64::MAX)
        );
    }

    #[test]
    fn private_replay_targets_are_closed_and_distinct() {
        let targets = [
            PrivateReplayCleanupTarget::VerifiedSubmitResponse,
            PrivateReplayCleanupTarget::VerifiedSubmitReservation,
            PrivateReplayCleanupTarget::BlindRouteResponse,
            PrivateReplayCleanupTarget::BlindRouteReservation,
        ];
        let statements = targets.map(PrivateReplayCleanupTarget::delete_sql);
        for (index, statement) in statements.iter().enumerate() {
            assert!(statement.starts_with("DELETE FROM relay_"));
            assert!(statements
                .iter()
                .enumerate()
                .all(|(other_index, other)| index == other_index || statement != other));
        }
    }
}
