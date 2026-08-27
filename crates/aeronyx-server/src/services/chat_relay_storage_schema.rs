// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_storage_schema.rs
// ============================================
// Version: 1.0.0-StorageSchemaDomain
//
// Creation Reason:
//   [CHAT-RELAY-STORAGE-SCHEMA-DOMAIN 2026-08-27 by Codex] Extract encrypted
//   blob, expiry notification, and aggregate storage-accounting schema work
//   from the oversized relay orchestration service.
//
// Main Functionality:
//   - Defines a replaceable relay storage-schema capability.
//   - Installs encrypted blob and expiry-notification custody tables.
//   - Installs durable aggregate usage counters and maintenance triggers.
//   - Rebuilds the derived usage singleton from canonical custody rows.
//   - Rejects negative or non-representable accounting state fail closed.
//
// Dependencies:
//   - `chat_relay_error.rs` supplies typed fail-closed storage failures.
//   - `rusqlite` supplies durable schema and accounting operations.
//
// Main Logical Flow:
//   1. Install encrypted blob and expiry-notification custody tables.
//   2. Allow the relay composition root to install dependent schemas.
//   3. Install the aggregate usage table and row-maintenance triggers.
//   4. Read canonical active-message and blob counts and byte totals.
//   5. Replace the derived singleton only after all values validate.
//
// Important Note for Next Developer:
//   - Aggregate counters are derived state; custody rows remain canonical.
//   - Startup reconciliation must run after all custody tables are installed.
//   - Preserve status-zero as the only active pending-message accounting state.
//   - Never add identity, route, endpoint, or payload dimensions to usage data.
//   - Corrupt or overflowing accounting values must never become spare quota.
//
// Last Modified:
//   v1.0.0-StorageSchemaDomain - Initial composed SQLite schema capability
// ============================================

use rusqlite::{params, Connection};

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Capability that installs and reconciles relay custody accounting schema.
pub(crate) trait ChatRelayStorageSchemaMigration {
    fn install_custody_tables(&self, connection: &Connection) -> ChatRelayResult<()>;

    fn install_usage_accounting(&self, connection: &Connection) -> ChatRelayResult<()>;

    fn reconcile_usage(&self, connection: &Connection) -> ChatRelayResult<()>;
}

/// Production SQLite implementation of relay storage schema migration.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SqliteChatRelayStorageSchemaMigrator;

impl SqliteChatRelayStorageSchemaMigrator {
    pub(crate) const fn new() -> Self {
        Self
    }

    fn read_canonical_usage(connection: &Connection) -> ChatRelayResult<CanonicalStorageUsage> {
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
        Ok(CanonicalStorageUsage {
            pending_messages: nonnegative_sqlite_value(
                counters.0,
                "canonical_pending_message_count",
            )?,
            pending_message_bytes: nonnegative_sqlite_value(
                counters.1,
                "canonical_pending_message_bytes",
            )?,
            pending_blobs: nonnegative_sqlite_value(counters.2, "canonical_pending_blob_count")?,
            pending_blob_bytes: nonnegative_sqlite_value(
                counters.3,
                "canonical_pending_blob_bytes",
            )?,
        })
    }
}

impl ChatRelayStorageSchemaMigration for SqliteChatRelayStorageSchemaMigrator {
    fn install_custody_tables(&self, connection: &Connection) -> ChatRelayResult<()> {
        connection.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_blobs (
                blob_id      TEXT PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                data         BLOB     NOT NULL,
                size         INTEGER  NOT NULL,
                received_at  INTEGER  NOT NULL,
                downloaded   INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_pb_received_at
                ON pending_blobs(received_at);
            CREATE INDEX IF NOT EXISTS idx_pb_receiver
                ON pending_blobs(receiver);

            CREATE TABLE IF NOT EXISTS expired_notifications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                sender      BLOB(32) NOT NULL,
                receiver    BLOB(32) NOT NULL,
                message_ids BLOB     NOT NULL,
                created_at  INTEGER  NOT NULL,
                pushed      INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_en_sender_pushed
                ON expired_notifications(sender, pushed);
            CREATE INDEX IF NOT EXISTS idx_en_sender_pull_order
                ON expired_notifications(sender, pushed, created_at, id);
            CREATE INDEX IF NOT EXISTS idx_en_cleanup
                ON expired_notifications(pushed, created_at, id);
            ",
        )?;
        Ok(())
    }

    fn install_usage_accounting(&self, connection: &Connection) -> ChatRelayResult<()> {
        connection.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_storage_usage (
                singleton              INTEGER PRIMARY KEY CHECK(singleton = 1),
                pending_message_count  INTEGER NOT NULL CHECK(pending_message_count >= 0),
                pending_message_bytes  INTEGER NOT NULL CHECK(pending_message_bytes >= 0),
                pending_blob_count     INTEGER NOT NULL CHECK(pending_blob_count >= 0),
                pending_blob_bytes     INTEGER NOT NULL CHECK(pending_blob_bytes >= 0)
            );

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_insert
            AFTER INSERT ON pending_messages
            WHEN NEW.status = 0
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = pending_message_count + 1,
                    pending_message_bytes = pending_message_bytes + LENGTH(NEW.envelope)
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_delete
            AFTER DELETE ON pending_messages
            WHEN OLD.status = 0
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = MAX(0, pending_message_count - 1),
                    pending_message_bytes = MAX(
                        0,
                        pending_message_bytes - LENGTH(OLD.envelope)
                    )
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_message_usage_status
            AFTER UPDATE OF status ON pending_messages
            WHEN OLD.status != NEW.status
            BEGIN
                UPDATE relay_storage_usage
                SET pending_message_count = MAX(
                        0,
                        pending_message_count
                        + CASE
                            WHEN OLD.status = 0 AND NEW.status != 0 THEN -1
                            WHEN OLD.status != 0 AND NEW.status = 0 THEN 1
                            ELSE 0
                          END
                    ),
                    pending_message_bytes = MAX(
                        0,
                        pending_message_bytes
                        + CASE
                            WHEN OLD.status = 0 AND NEW.status != 0
                                THEN -LENGTH(OLD.envelope)
                            WHEN OLD.status != 0 AND NEW.status = 0
                                THEN LENGTH(NEW.envelope)
                            ELSE 0
                          END
                    )
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_blob_usage_insert
            AFTER INSERT ON pending_blobs
            BEGIN
                UPDATE relay_storage_usage
                SET pending_blob_count = pending_blob_count + 1,
                    pending_blob_bytes = pending_blob_bytes + NEW.size
                WHERE singleton = 1;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_relay_blob_usage_delete
            AFTER DELETE ON pending_blobs
            BEGIN
                UPDATE relay_storage_usage
                SET pending_blob_count = MAX(0, pending_blob_count - 1),
                    pending_blob_bytes = MAX(0, pending_blob_bytes - OLD.size)
                WHERE singleton = 1;
            END;
            ",
        )?;
        Ok(())
    }

    fn reconcile_usage(&self, connection: &Connection) -> ChatRelayResult<()> {
        // [CHAT-RELAY-STORAGE-SCHEMA-DOMAIN 2026-08-27 by Codex] Derived
        // counters are rebuilt from canonical rows after every startup.
        let usage = Self::read_canonical_usage(connection)?;
        connection.execute(
            "INSERT OR REPLACE INTO relay_storage_usage (
                singleton,
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             )
             VALUES (1, ?1, ?2, ?3, ?4)",
            params![
                sqlite_integer(usage.pending_messages, "pending_message_count")?,
                sqlite_integer(usage.pending_message_bytes, "pending_message_bytes")?,
                sqlite_integer(usage.pending_blobs, "pending_blob_count")?,
                sqlite_integer(usage.pending_blob_bytes, "pending_blob_bytes")?,
            ],
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CanonicalStorageUsage {
    pending_messages: u64,
    pending_message_bytes: u64,
    pending_blobs: u64,
    pending_blob_bytes: u64,
}

fn nonnegative_sqlite_value(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

fn sqlite_integer(value: u64, field: &'static str) -> ChatRelayResult<i64> {
    i64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}
