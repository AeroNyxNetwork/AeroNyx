// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_expired_delivery.rs
// ============================================
// Version: 1.0.0-ExpiredDeliveryDomain
//
// Creation Reason:
//   [CHAT-EXPIRED-DELIVERY-DOMAIN 2026-08-25 by Codex] Extract expiry-control
//   reads, durable-row validation, pagination, and pushed-state writes from the
//   oversized relay service without changing its public API or SQLite schema.
//
// Main Functionality:
//   - Defines a replaceable expiry-notification repository trait.
//   - Reads one bounded, deterministically ordered sender page.
//   - Validates sender binding, receiver shape, payload bounds, and ID format.
//   - Marks a deduplicated delivered page as pushed in one transaction.
//
// Dependencies:
//   - `chat_relay.rs` owns connection locking, quarantine, telemetry, and API.
//   - `rusqlite` provides the production durable repository implementation.
//
// Main Logical Flow:
//   1. Read at most one page plus a look-ahead row for stable pagination.
//   2. Split valid notifications from privacy-minimised corruption evidence.
//   3. Return a typed page while the caller retains the connection lock.
//   4. After transport success, atomically mark unique notification IDs pushed.
//
// Important Note for Next Developer:
//   - Repository replacements must preserve sender filtering and stable order.
//   - This domain does not delete corrupt rows; the service owns quarantine.
//   - Mark-pushed is transport acknowledgement, not message-content consent.
//   - Never log sender/receiver keys, notification IDs, or serialized payloads.
//
// Last Modified:
//   v1.0.0-ExpiredDeliveryDomain - Initial delivery repository composition
// ============================================

use std::collections::HashSet;

use rusqlite::{params, Connection, TransactionBehavior};

use super::chat_relay::{
    ChatRelayResult, CorruptDurableRow, ExpiredNotification, MAX_EXPIRED_NOTIFICATIONS_PER_PULL,
    MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES, QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
};

/// Raw durable expiry-notification row returned by a repository.
#[derive(Debug, Clone)]
pub(crate) struct StoredExpiredNotificationRow {
    id: i64,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    message_ids_raw: Vec<u8>,
}

/// Validated expiry-notification page and quarantine candidates.
pub(crate) struct ExpiredNotificationPage {
    pub(crate) notifications: Vec<ExpiredNotification>,
    pub(crate) corrupt_rows: Vec<CorruptDurableRow>,
    pub(crate) has_more: bool,
}

/// Validated, deduplicated notification delivery acknowledgement.
pub(crate) struct ExpiredNotificationAckBatch {
    notification_ids: HashSet<i64>,
}

/// Replaceable persistence capability for expiry-notification delivery.
///
/// [CHAT-EXPIRED-DELIVERY-DOMAIN 2026-08-25 by Codex] Implementations perform
/// bounded durable operations only. Sender binding and corruption
/// classification remain in the composed domain and cannot be bypassed by a
/// replacement storage engine.
pub(crate) trait ExpiredNotificationRepository: Send + Sync {
    fn read_pending_rows(
        &self,
        conn: &Connection,
        sender: &[u8; 32],
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredExpiredNotificationRow>>;

    fn mark_pushed(
        &self,
        conn: &mut Connection,
        notification_ids: &HashSet<i64>,
    ) -> ChatRelayResult<()>;
}

/// Production SQLite repository for expiry-notification delivery.
pub(crate) struct SqliteExpiredNotificationRepository;

impl ExpiredNotificationRepository for SqliteExpiredNotificationRepository {
    fn read_pending_rows(
        &self,
        conn: &Connection,
        sender: &[u8; 32],
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredExpiredNotificationRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, sender, receiver, message_ids
             FROM expired_notifications
             WHERE sender = ?1 AND pushed = 0
             ORDER BY created_at ASC, id ASC
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![sender.as_slice(), limit], |row| {
                Ok(StoredExpiredNotificationRow {
                    id: row.get(0)?,
                    sender: row.get(1)?,
                    receiver: row.get(2)?,
                    message_ids_raw: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    fn mark_pushed(
        &self,
        conn: &mut Connection,
        notification_ids: &HashSet<i64>,
    ) -> ChatRelayResult<()> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for notification_id in notification_ids {
            tx.execute(
                "UPDATE expired_notifications SET pushed = 1 WHERE id = ?1",
                params![notification_id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }
}

/// Composed expiry-notification delivery domain.
pub(crate) struct ExpiredNotificationDelivery<R = SqliteExpiredNotificationRepository> {
    repository: R,
}

impl ExpiredNotificationDelivery<SqliteExpiredNotificationRepository> {
    pub(crate) fn new() -> Self {
        Self::with_repository(SqliteExpiredNotificationRepository)
    }
}

impl<R: ExpiredNotificationRepository> ExpiredNotificationDelivery<R> {
    fn with_repository(repository: R) -> Self {
        Self { repository }
    }

    pub(crate) fn read_page(
        &self,
        conn: &Connection,
        sender: &[u8; 32],
    ) -> ChatRelayResult<ExpiredNotificationPage> {
        let effective_limit = MAX_EXPIRED_NOTIFICATIONS_PER_PULL.saturating_add(1);
        let rows = self.repository.read_pending_rows(
            conn,
            sender,
            i64::try_from(effective_limit).unwrap_or(i64::MAX),
        )?;
        let raw_has_more = rows.len() == effective_limit;
        let mut notifications =
            Vec::with_capacity(rows.len().min(MAX_EXPIRED_NOTIFICATIONS_PER_PULL));
        let mut corrupt_rows = Vec::new();
        for row in rows {
            match validate_expired_notification_row(row, sender) {
                Ok(notification) => notifications.push(notification),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }

        let has_more = raw_has_more || notifications.len() > MAX_EXPIRED_NOTIFICATIONS_PER_PULL;
        notifications.truncate(MAX_EXPIRED_NOTIFICATIONS_PER_PULL);
        Ok(ExpiredNotificationPage {
            notifications,
            corrupt_rows,
            has_more,
        })
    }

    pub(crate) fn prepare_acknowledgement(
        &self,
        notification_ids: &[i64],
    ) -> Option<ExpiredNotificationAckBatch> {
        if notification_ids.is_empty() {
            return None;
        }
        Some(ExpiredNotificationAckBatch {
            notification_ids: notification_ids.iter().copied().collect(),
        })
    }

    pub(crate) fn mark_pushed(
        &self,
        conn: &mut Connection,
        batch: &ExpiredNotificationAckBatch,
    ) -> ChatRelayResult<()> {
        self.repository.mark_pushed(conn, &batch.notification_ids)
    }
}

fn validate_expired_notification_row(
    row: StoredExpiredNotificationRow,
    expected_sender: &[u8; 32],
) -> Result<ExpiredNotification, CorruptDurableRow> {
    let encoded_bytes = u64::try_from(row.message_ids_raw.len()).unwrap_or(u64::MAX);
    let corrupt = |reason| CorruptDurableRow {
        row_key: row.id,
        source_kind: QUARANTINE_SOURCE_EXPIRED_NOTIFICATION,
        reason,
        encoded_bytes,
    };
    if row.message_ids_raw.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
        return Err(corrupt("expired_notification_payload_size"));
    }
    let stored_sender: [u8; 32] = row
        .sender
        .try_into()
        .map_err(|_| corrupt("expired_notification_sender"))?;
    if stored_sender != *expected_sender {
        return Err(corrupt("expired_notification_sender_mismatch"));
    }
    let receiver: [u8; 32] = row
        .receiver
        .try_into()
        .map_err(|_| corrupt("expired_notification_receiver"))?;
    let notification = ExpiredNotification {
        id: row.id,
        sender: stored_sender,
        receiver,
        message_ids_raw: row.message_ids_raw,
    };
    notification
        .message_ids()
        .map_err(|_| corrupt("expired_notification_message_ids"))?;
    Ok(notification)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct StubRepository {
        rows: Vec<StoredExpiredNotificationRow>,
        acknowledgements: AtomicUsize,
    }

    impl ExpiredNotificationRepository for StubRepository {
        fn read_pending_rows(
            &self,
            _conn: &Connection,
            _sender: &[u8; 32],
            _limit: i64,
        ) -> ChatRelayResult<Vec<StoredExpiredNotificationRow>> {
            Ok(self.rows.clone())
        }

        fn mark_pushed(
            &self,
            _conn: &mut Connection,
            notification_ids: &HashSet<i64>,
        ) -> ChatRelayResult<()> {
            self.acknowledgements
                .fetch_add(notification_ids.len(), Ordering::Relaxed);
            Ok(())
        }
    }

    fn stored_row(id: i64, sender: Vec<u8>, receiver: Vec<u8>) -> StoredExpiredNotificationRow {
        StoredExpiredNotificationRow {
            id,
            sender,
            receiver,
            message_ids_raw: bincode::serialize(&vec![[0x31; 16]])
                .expect("encode test message IDs"),
        }
    }

    #[test]
    fn composed_delivery_validates_rows_and_deduplicates_acknowledgements() {
        let sender = [0x11; 32];
        let repository = StubRepository {
            rows: vec![
                stored_row(1, sender.to_vec(), vec![0x22; 32]),
                stored_row(2, sender.to_vec(), vec![0x23; 31]),
            ],
            acknowledgements: AtomicUsize::new(0),
        };
        let delivery = ExpiredNotificationDelivery::with_repository(repository);
        let mut conn = Connection::open_in_memory().expect("open test connection");

        let page = delivery
            .read_page(&conn, &sender)
            .expect("read composed page");
        assert_eq!(page.notifications.len(), 1);
        assert_eq!(page.corrupt_rows.len(), 1);
        assert_eq!(page.corrupt_rows[0].reason, "expired_notification_receiver");
        assert!(!page.has_more);

        assert!(delivery.prepare_acknowledgement(&[]).is_none());
        assert_eq!(
            delivery.repository.acknowledgements.load(Ordering::Relaxed),
            0
        );

        let batch = delivery
            .prepare_acknowledgement(&[7, 7, 9])
            .expect("prepare non-empty acknowledgement");
        delivery
            .mark_pushed(&mut conn, &batch)
            .expect("mark through stub");
        assert_eq!(
            delivery.repository.acknowledgements.load(Ordering::Relaxed),
            2
        );
    }
}
