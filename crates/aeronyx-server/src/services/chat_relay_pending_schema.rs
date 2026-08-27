// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_schema.rs
// ============================================
// Version: 1.0.0-PendingMessageSchemaDomain
//
// Creation Reason:
//   [CHAT-RELAY-PENDING-SCHEMA-DOMAIN 2026-08-27 by Codex] Extract the
//   pending-message table installation and legacy queue-sequence migration
//   from the oversized relay orchestration service.
//
// Main Functionality:
//   - Defines a replaceable pending-message schema migration capability.
//   - Installs the durable pending-message table and read indexes.
//   - Adds queue sequencing to legacy databases in one immediate transaction.
//   - Preserves valid unique sequences and deterministically repairs the rest.
//   - Advances the durable sequence high-water mark before committing.
//
// Dependencies:
//   - `chat_relay_error.rs` supplies typed fail-closed storage failures.
//   - `rusqlite` supplies schema introspection and immediate transactions.
//
// Main Logical Flow:
//   1. Install the compatible base pending-message schema and indexes.
//   2. Open an immediate migration transaction.
//   3. Add the sequence column and singleton high-water table when required.
//   4. Retain valid unique sequences and assign deterministic replacements.
//   5. Install sequence indexes, advance the high-water mark, and commit.
//
// Important Note for Next Developer:
//   - Keep sequence repair and high-water advancement in one transaction.
//   - Existing positive unique sequence values must remain stable on restart.
//   - Invalid, missing, or duplicate values are repaired in rowid order.
//   - Sequence exhaustion and unexpected update counts must fail closed.
//   - Do not expose message, sender, receiver, or queue metadata in errors.
//
// Last Modified:
//   v1.0.0-PendingMessageSchemaDomain - Initial composed SQLite migrator
// ============================================

use std::collections::HashSet;

use rusqlite::{params, Connection, Transaction, TransactionBehavior};

use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Capability that brings pending-message custody to its required schema.
pub(crate) trait ChatRelayPendingSchemaMigration {
    fn migrate(&self, connection: &mut Connection) -> ChatRelayResult<()>;
}

/// Production SQLite implementation of pending-message schema migration.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SqliteChatRelayPendingSchemaMigrator;

impl SqliteChatRelayPendingSchemaMigrator {
    pub(crate) const fn new() -> Self {
        Self
    }

    fn install_base_schema(connection: &Connection) -> ChatRelayResult<()> {
        connection.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_messages (
                message_id   BLOB(16) PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                timestamp    INTEGER  NOT NULL,
                envelope     BLOB     NOT NULL,
                received_at  INTEGER  NOT NULL,
                status       INTEGER  NOT NULL DEFAULT 0,
                queue_sequence INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status
                ON pending_messages(receiver, status);
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status_message_id
                ON pending_messages(receiver, status, message_id);
            CREATE INDEX IF NOT EXISTS idx_pm_received_at
                ON pending_messages(received_at);
            CREATE INDEX IF NOT EXISTS idx_pm_cleanup
                ON pending_messages(status, received_at, message_id);
            ",
        )?;
        Ok(())
    }

    fn migrate_queue_sequence(connection: &mut Connection) -> ChatRelayResult<()> {
        // [CHAT-RELAY-PENDING-SCHEMA-DOMAIN 2026-08-27 by Codex] Keep legacy
        // repair, index installation, and high-water advancement atomic.
        let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        if !pending_message_column_exists(&tx, "queue_sequence")? {
            tx.execute(
                "ALTER TABLE pending_messages ADD COLUMN queue_sequence INTEGER",
                [],
            )?;
        }
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS relay_queue_sequence (
                singleton     INTEGER PRIMARY KEY CHECK(singleton = 1),
                last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0)
            );
            INSERT OR IGNORE INTO relay_queue_sequence (singleton, last_sequence)
            VALUES (1, 0);
            ",
        )?;

        let mut seen_sequences = HashSet::new();
        let mut max_sequence = 0_i64;
        let mut rowids_to_assign = Vec::new();
        {
            let mut stmt = tx.prepare(
                "SELECT rowid, queue_sequence
                 FROM pending_messages
                 ORDER BY rowid ASC",
            )?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<i64>>(1)?))
            })?;
            for row in rows {
                let (rowid, sequence) = row?;
                match sequence {
                    Some(sequence) if sequence > 0 && seen_sequences.insert(sequence) => {
                        max_sequence = max_sequence.max(sequence);
                    }
                    _ => rowids_to_assign.push(rowid),
                }
            }
        }

        let persisted_last = tx.query_row(
            "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        if persisted_last < 0 {
            return Err(ChatRelayError::CorruptStoredData {
                field: "relay_queue_sequence_negative",
            });
        }
        max_sequence = max_sequence.max(persisted_last);
        for rowid in rowids_to_assign {
            max_sequence = max_sequence
                .checked_add(1)
                .ok_or(ChatRelayError::QueueSequenceExhausted)?;
            if tx.execute(
                "UPDATE pending_messages SET queue_sequence = ?1 WHERE rowid = ?2",
                params![max_sequence, rowid],
            )? != 1
            {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "pending_message_sequence_backfill_count",
                });
            }
        }
        tx.execute(
            "UPDATE relay_queue_sequence
             SET last_sequence = ?1
             WHERE singleton = 1",
            params![max_sequence],
        )?;
        tx.execute_batch(
            "
            CREATE UNIQUE INDEX IF NOT EXISTS idx_pm_queue_sequence
                ON pending_messages(queue_sequence);
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_snapshot_v2
                ON pending_messages(receiver, status, queue_sequence, timestamp);
            ",
        )?;
        tx.commit()?;
        Ok(())
    }
}

impl ChatRelayPendingSchemaMigration for SqliteChatRelayPendingSchemaMigrator {
    fn migrate(&self, connection: &mut Connection) -> ChatRelayResult<()> {
        Self::install_base_schema(connection)?;
        Self::migrate_queue_sequence(connection)
    }
}

fn pending_message_column_exists(
    tx: &Transaction<'_>,
    expected_column: &str,
) -> ChatRelayResult<bool> {
    let mut stmt = tx.prepare("PRAGMA table_info(pending_messages)")?;
    let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for column in columns {
        if column? == expected_column {
            return Ok(true);
        }
    }
    Ok(false)
}
