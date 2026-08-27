// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_quarantine.rs
// ============================================
// Version: 1.0.0-DurableQuarantineDomain
//
// Creation Reason:
//   [CHAT-DURABLE-QUARANTINE-DOMAIN 2026-08-25 by Codex] Extract atomic,
//   privacy-minimised corrupt-row isolation from the oversized relay service.
//
// Main Functionality:
//   - Models corrupt durable rows and their exact source table as typed values.
//   - Defines a replaceable repository capability for quarantine persistence.
//   - Atomically records aggregate evidence before removing poison rows.
//   - Applies bounded TTL and capacity retention without storing identities.
//
// Dependencies:
//   - `config_chat_relay.rs` supplies the validated retention duration.
//   - `chat_relay.rs` owns maintenance telemetry and cleanup orchestration.
//   - `rusqlite` provides the production transactional repository.
//
// Main Logical Flow:
//   1. Group corrupt rows by stable source and reason without identity fields.
//   2. Insert aggregate evidence and delete the exact poisoned source rows.
//   3. Trim stale/overflow evidence in bounded batches in the same transaction.
//   4. Return aggregate counts for service-owned telemetry.
//
// Important Note for Next Developer:
//   - Never add message IDs, wallets, endpoints, ciphertext, or payloads here.
//   - Evidence insertion and source-row deletion must remain atomic.
//   - Cleanup callers must reuse the surrounding transaction via `record` and
//     `maintain`; opening a nested transaction would break expiry atomicity.
//   - Existing table names, reason buckets, and retention behavior are stable.
//
// Last Modified:
//   v1.0.0-DurableQuarantineDomain - Initial typed repository composition
// ============================================

use std::collections::HashMap;

use rusqlite::{params, Connection, Transaction, TransactionBehavior};

use crate::config::ChatRelayConfig;

// [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Quarantine persistence
// consumes typed failures directly instead of the relay service facade.
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Maximum privacy-minimised quarantine events removed by one transaction.
const CLEANUP_QUARANTINE_EVENT_BATCH_SIZE: usize = 1024;
/// Maximum retained de-identified corruption events.
pub(crate) const MAX_QUARANTINE_EVENTS: usize = 4096;

/// Stable source bucket for pending message corruption.
pub(crate) const QUARANTINE_SOURCE_PENDING_MESSAGE: &str = "pending_message";
/// Stable source bucket for expiry-notification corruption.
pub(crate) const QUARANTINE_SOURCE_EXPIRED_NOTIFICATION: &str = "expired_notification";

/// One malformed durable row represented without routing identities or content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CorruptDurableRow {
    pub(crate) row_key: i64,
    pub(crate) source_kind: &'static str,
    pub(crate) reason: &'static str,
    pub(crate) encoded_bytes: u64,
}

/// Durable source table from which poison rows must be removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuarantineRowTarget {
    PendingMessage,
    ExpiredNotification,
}

impl QuarantineRowTarget {
    fn source_kind(self) -> &'static str {
        match self {
            Self::PendingMessage => QUARANTINE_SOURCE_PENDING_MESSAGE,
            Self::ExpiredNotification => QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
        }
    }

    fn delete_sql(self) -> &'static str {
        match self {
            Self::PendingMessage => "DELETE FROM pending_messages WHERE rowid = ?1",
            Self::ExpiredNotification => "DELETE FROM expired_notifications WHERE id = ?1",
        }
    }

    fn delete_count_field(self) -> &'static str {
        match self {
            Self::PendingMessage => "pending_message_quarantine_delete_count",
            Self::ExpiredNotification => "expired_notification_quarantine_delete_count",
        }
    }
}

/// Aggregate result of one atomic poison-row replacement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct QuarantineReplaceOutcome {
    pub(crate) quarantined_rows: usize,
    pub(crate) removed_events: usize,
    pub(crate) retained_events: usize,
}

/// Aggregate result of bounded quarantine retention.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct QuarantineMaintenanceOutcome {
    pub(crate) removed_events: usize,
    pub(crate) retained_events: usize,
}

/// Immutable durable quarantine policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DurableQuarantinePolicy {
    retention_ttl_secs: u64,
}

impl From<&ChatRelayConfig> for DurableQuarantinePolicy {
    fn from(config: &ChatRelayConfig) -> Self {
        Self {
            retention_ttl_secs: config.expired_notification_ttl_secs,
        }
    }
}

/// Replaceable persistence capability for de-identified quarantine evidence.
pub(crate) trait DurableQuarantineRepository: Send + Sync {
    fn init_schema(&self, conn: &Connection) -> ChatRelayResult<()>;

    fn record(
        &self,
        tx: &Transaction<'_>,
        now: i64,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()>;

    fn delete_source_rows(
        &self,
        tx: &Transaction<'_>,
        target: QuarantineRowTarget,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()>;

    fn maintain(
        &self,
        tx: &Transaction<'_>,
        retention_cutoff: i64,
    ) -> ChatRelayResult<QuarantineMaintenanceOutcome>;

    fn retained_count(&self, conn: &Connection) -> ChatRelayResult<usize>;

    fn backlog_exists(&self, tx: &Transaction<'_>, retention_cutoff: i64) -> ChatRelayResult<bool>;
}

/// Production SQLite repository for de-identified quarantine evidence.
pub(crate) struct SqliteDurableQuarantineRepository;

impl DurableQuarantineRepository for SqliteDurableQuarantineRepository {
    fn init_schema(&self, conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_quarantine_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_kind     TEXT    NOT NULL,
                reason          TEXT    NOT NULL,
                row_count       INTEGER NOT NULL CHECK(row_count > 0),
                encoded_bytes   INTEGER NOT NULL CHECK(encoded_bytes >= 0),
                quarantined_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rqe_retention
                ON relay_quarantine_events(quarantined_at, id);
            ",
        )?;
        Ok(())
    }

    fn record(
        &self,
        tx: &Transaction<'_>,
        now: i64,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut aggregates: HashMap<(&'static str, &'static str), (u64, u64)> = HashMap::new();
        for row in rows {
            let aggregate = aggregates
                .entry((row.source_kind, row.reason))
                .or_insert((0, 0));
            aggregate.0 = aggregate.0.saturating_add(1);
            aggregate.1 = aggregate.1.saturating_add(row.encoded_bytes);
        }

        let mut statement = tx.prepare(
            "INSERT INTO relay_quarantine_events
             (source_kind, reason, row_count, encoded_bytes, quarantined_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for ((source_kind, reason), (row_count, encoded_bytes)) in aggregates {
            statement.execute(params![
                source_kind,
                reason,
                i64::try_from(row_count).unwrap_or(i64::MAX),
                i64::try_from(encoded_bytes).unwrap_or(i64::MAX),
                now,
            ])?;
        }
        Ok(())
    }

    fn delete_source_rows(
        &self,
        tx: &Transaction<'_>,
        target: QuarantineRowTarget,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut statement = tx.prepare(target.delete_sql())?;
        for row in rows {
            if statement.execute(params![row.row_key])? != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: target.delete_count_field(),
                });
            }
        }
        Ok(())
    }

    fn maintain(
        &self,
        tx: &Transaction<'_>,
        retention_cutoff: i64,
    ) -> ChatRelayResult<QuarantineMaintenanceOutcome> {
        let cleanup_limit = i64::try_from(CLEANUP_QUARANTINE_EVENT_BATCH_SIZE).unwrap_or(i64::MAX);
        let max_events = i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX);
        let removed_stale = tx.execute(
            "DELETE FROM relay_quarantine_events
             WHERE id IN (
                 SELECT id FROM relay_quarantine_events
                 WHERE quarantined_at < ?1
                 ORDER BY quarantined_at ASC, id ASC
                 LIMIT ?2
             )",
            params![retention_cutoff, cleanup_limit],
        )?;
        let removed_overflow = tx.execute(
            "DELETE FROM relay_quarantine_events
             WHERE id IN (
                 SELECT id FROM relay_quarantine_events
                 ORDER BY quarantined_at DESC, id DESC
                 LIMIT ?1 OFFSET ?2
             )",
            params![cleanup_limit, max_events],
        )?;
        Ok(QuarantineMaintenanceOutcome {
            removed_events: removed_stale.saturating_add(removed_overflow),
            retained_events: self.retained_count(tx)?,
        })
    }

    fn retained_count(&self, conn: &Connection) -> ChatRelayResult<usize> {
        let count = conn.query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
            row.get::<_, i64>(0)
        })?;
        Ok(usize::try_from(count.max(0)).unwrap_or(usize::MAX))
    }

    fn backlog_exists(&self, tx: &Transaction<'_>, retention_cutoff: i64) -> ChatRelayResult<bool> {
        let max_events = i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX);
        Ok(tx.query_row(
            "SELECT
                 EXISTS(
                     SELECT 1 FROM relay_quarantine_events
                     WHERE quarantined_at < ?1
                 )
                 OR (SELECT COUNT(*) FROM relay_quarantine_events) > ?2",
            params![retention_cutoff, max_events],
            |row| row.get::<_, i64>(0),
        )? != 0)
    }
}

/// Composed policy and persistence boundary for durable quarantine.
pub(crate) struct DurableQuarantineDomain<R = SqliteDurableQuarantineRepository> {
    repository: R,
    policy: DurableQuarantinePolicy,
}

impl DurableQuarantineDomain<SqliteDurableQuarantineRepository> {
    pub(crate) fn new(config: &ChatRelayConfig) -> Self {
        Self::with_repository(
            SqliteDurableQuarantineRepository,
            DurableQuarantinePolicy::from(config),
        )
    }
}

impl<R: DurableQuarantineRepository> DurableQuarantineDomain<R> {
    fn with_repository(repository: R, policy: DurableQuarantinePolicy) -> Self {
        Self { repository, policy }
    }

    pub(crate) fn init_schema(&self, conn: &Connection) -> ChatRelayResult<()> {
        self.repository.init_schema(conn)
    }

    pub(crate) fn retained_count(&self, conn: &Connection) -> ChatRelayResult<usize> {
        self.repository.retained_count(conn)
    }

    pub(crate) fn replace_rows(
        &self,
        conn: &mut Connection,
        target: QuarantineRowTarget,
        rows: &[CorruptDurableRow],
        now: u64,
    ) -> ChatRelayResult<QuarantineReplaceOutcome> {
        if rows.is_empty() {
            return Ok(QuarantineReplaceOutcome::default());
        }
        if rows
            .iter()
            .any(|row| row.source_kind != target.source_kind())
        {
            return Err(ChatRelayError::CorruptStoredData {
                field: "quarantine_source_target_mismatch",
            });
        }
        let now = i64::try_from(now).unwrap_or(i64::MAX);
        let transaction = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        self.record(&transaction, now, rows)?;
        self.repository
            .delete_source_rows(&transaction, target, rows)?;
        let maintenance = self.maintain(&transaction, now)?;
        transaction.commit()?;
        Ok(QuarantineReplaceOutcome {
            quarantined_rows: rows.len(),
            removed_events: maintenance.removed_events,
            retained_events: maintenance.retained_events,
        })
    }

    pub(crate) fn record(
        &self,
        tx: &Transaction<'_>,
        now: i64,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        self.repository.record(tx, now, rows)
    }

    pub(crate) fn maintain(
        &self,
        tx: &Transaction<'_>,
        now: i64,
    ) -> ChatRelayResult<QuarantineMaintenanceOutcome> {
        self.repository.maintain(tx, self.retention_cutoff(now))
    }

    pub(crate) fn backlog_exists(&self, tx: &Transaction<'_>, now: i64) -> ChatRelayResult<bool> {
        self.repository
            .backlog_exists(tx, self.retention_cutoff(now))
    }

    fn retention_cutoff(&self, now: i64) -> i64 {
        let ttl = i64::try_from(self.policy.retention_ttl_secs).unwrap_or(i64::MAX);
        now.saturating_sub(ttl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_quarantine_replaces_poison_row_with_identity_blind_evidence() {
        let mut connection = Connection::open_in_memory().expect("open quarantine database");
        connection
            .execute_batch(
                "CREATE TABLE pending_messages (payload BLOB NOT NULL);
                 INSERT INTO pending_messages (payload) VALUES (x'010203');",
            )
            .expect("seed poison row");
        let domain = DurableQuarantineDomain::with_repository(
            SqliteDurableQuarantineRepository,
            DurableQuarantinePolicy {
                retention_ttl_secs: 60,
            },
        );
        domain.init_schema(&connection).expect("initialize schema");

        let outcome = domain
            .replace_rows(
                &mut connection,
                QuarantineRowTarget::PendingMessage,
                &[CorruptDurableRow {
                    row_key: 1,
                    source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
                    reason: "invalid_envelope",
                    encoded_bytes: 3,
                }],
                100,
            )
            .expect("quarantine poison row");

        assert_eq!(outcome.quarantined_rows, 1);
        assert_eq!(outcome.retained_events, 1);
        let source_rows: i64 = connection
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count source rows");
        assert_eq!(source_rows, 0);
        let evidence: (String, String, i64, i64) = connection
            .query_row(
                "SELECT source_kind, reason, row_count, encoded_bytes
                 FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("read quarantine evidence");
        assert_eq!(
            evidence,
            (
                QUARANTINE_SOURCE_PENDING_MESSAGE.to_string(),
                "invalid_envelope".to_string(),
                1,
                3,
            )
        );
    }

    #[test]
    fn quarantine_target_mismatch_fails_before_source_mutation() {
        let mut connection = Connection::open_in_memory().expect("open quarantine database");
        connection
            .execute_batch(
                "CREATE TABLE pending_messages (payload BLOB NOT NULL);
                 INSERT INTO pending_messages (payload) VALUES (x'01');",
            )
            .expect("seed source row");
        let domain = DurableQuarantineDomain::with_repository(
            SqliteDurableQuarantineRepository,
            DurableQuarantinePolicy {
                retention_ttl_secs: 60,
            },
        );
        domain.init_schema(&connection).expect("initialize schema");

        assert!(matches!(
            domain.replace_rows(
                &mut connection,
                QuarantineRowTarget::PendingMessage,
                &[CorruptDurableRow {
                    row_key: 1,
                    source_kind: QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
                    reason: "wrong_source",
                    encoded_bytes: 1,
                }],
                100,
            ),
            Err(ChatRelayError::CorruptStoredData {
                field: "quarantine_source_target_mismatch"
            })
        ));
        let source_rows: i64 = connection
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count preserved rows");
        assert_eq!(source_rows, 1);
        assert_eq!(domain.retained_count(&connection).unwrap(), 0);
    }
}
