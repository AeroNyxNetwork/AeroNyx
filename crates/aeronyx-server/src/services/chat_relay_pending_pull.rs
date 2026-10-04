// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_pull.rs
// ============================================
// Version: 1.3.0-PendingContractDependency
//
// Creation Reason:
//   [CHAT-PENDING-PULL-DOMAIN 2026-08-25 by Codex] Extract pending-message
//   reads and durable-row validation from the oversized relay service while
//   preserving both pull protocols, SQLite ordering, and quarantine behavior.
//
// Modification Reason:
//   [CHAT-PENDING-CONTRACT-DOMAIN 2026-08-28 by Codex] Depend directly on the
//   pending delivery contract instead of the central relay orchestrator.
//   [CHAT-PENDING-DELIVERY-DOMAIN 2026-08-28 by Codex] Updated ownership after
//   connection locking, quarantine, and final pagination moved to a coordinator.
//
// Main Functionality:
//   [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Reject invalid
//   durable BLOB shapes before copying them into Rust-owned page buffers.
//   - Defines a replaceable pending-pull repository trait.
//   - Implements legacy message-id and v2 sequence-ordered SQLite reads.
//   - Validates every denormalized durable row against its signed envelope.
//   - Returns typed valid/corrupt page results without performing side effects.
//
// Dependencies:
//   - `chat_relay_pending_delivery.rs` owns lock scope and final pagination.
//   - `chat_relay_quarantine.rs` owns typed corrupt-row evidence and isolation.
//   - `aeronyx-core` owns the bounded envelope codec and signature contract.
//
// Main Logical Flow:
//   1. Read a bounded raw page using the protocol-specific stable ordering.
//   2. Reconstruct and authenticate each opaque encrypted envelope.
//   3. Split valid messages from privacy-minimised corrupt-row evidence.
//   4. Return both sets to the delivery coordinator under its connection lock.
//
// Important Note for Next Developer:
//   - V1 must remain ordered by `message_id`; v2 must remain ordered by the
//     monotonic `queue_sequence` captured under a fixed snapshot ceiling.
//   - This domain intentionally does not delete or quarantine rows. The
//     delivery coordinator commits that side effect on the same connection.
//   - Never log row contents, message IDs, wallet keys, envelopes, or queries.
//   - Replacement repositories must preserve limits and deterministic order.
//
// Last Modified:
//   [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Shared SQL admission
//   preserves original byte evidence and both existing pagination contracts.
//   v1.3.0-PendingContractDependency - Removed orchestrator dependency
//   v1.2.0-PendingDeliveryComposition - Documented coordinator ownership
//   [CHAT-DURABLE-QUARANTINE-DOMAIN 2026-08-25 by Codex]
//   v1.1.0-DurableQuarantineBoundary - Consume shared typed corruption model
//   v1.0.0-PendingPullDomain - Initial repository/validation composition
// ============================================

use aeronyx_core::protocol::chat::{decode_envelope, MAX_CHAT_ENVELOPE_BYTES};
use rusqlite::{named_params, params, Connection, Row};

// [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Pull repositories consume the
// typed failure boundary directly while pending row models remain contract-owned.
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};
use super::chat_relay_pending_contract::PendingMessage;
use super::chat_relay_quarantine::{CorruptDurableRow, QUARANTINE_SOURCE_PENDING_MESSAGE};

#[derive(Debug, Clone)]
pub(crate) struct StoredPendingMessageRow {
    rowid: i64,
    message_id: Vec<u8>,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    timestamp: i64,
    envelope: Vec<u8>,
    // [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] A rejected SQL
    // projection carries only bounded evidence, never the rejected BLOB.
    projection_rejection: Option<CorruptDurableRow>,
}

#[derive(Debug, Clone)]
pub(crate) struct StoredSequencedPendingMessageRow {
    queue_sequence: i64,
    row: StoredPendingMessageRow,
}

/// Validated result of one legacy message-id-ordered repository read.
pub(crate) struct LegacyPendingPullPage {
    pub(crate) messages: Vec<PendingMessage>,
    pub(crate) corrupt_rows: Vec<CorruptDurableRow>,
    pub(crate) raw_has_more: bool,
}

/// Validated result of one v2 sequence-ordered snapshot repository read.
pub(crate) struct SnapshotPendingPullPage {
    pub(crate) messages: Vec<(u64, PendingMessage)>,
    pub(crate) corrupt_rows: Vec<CorruptDurableRow>,
    pub(crate) raw_has_more: bool,
    pub(crate) raw_max_sequence: Option<u64>,
}

/// Replaceable durable read capability for pending message pulls.
///
/// [CHAT-PENDING-PULL-DOMAIN 2026-08-25 by Codex] Repository implementations
/// return raw bounded rows only. Authentication and corruption classification
/// stay in the composed domain so storage engines cannot weaken validation.
pub(crate) trait PendingMessagePullRepository: Send + Sync {
    fn capture_snapshot_ceiling(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: i64,
    ) -> ChatRelayResult<u64>;

    fn read_legacy_rows(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: i64,
        cursor: &[u8; 16],
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredPendingMessageRow>>;

    fn read_snapshot_rows(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: i64,
        position: i64,
        ceiling: i64,
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredSequencedPendingMessageRow>>;
}

/// Production SQLite implementation of the pending-pull repository.
pub(crate) struct SqlitePendingMessagePullRepository;

// [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Both readers select
// only these guarded BLOB expressions, never an additional raw BLOB column.
// Named parameters keep the shared ceiling independent of query column/order
// differences. CAST is used only for scalar byte accounting (including TEXT),
// not as a way to admit non-BLOB data. SQLite's own pager/evaluation memory is
// outside this Rust projection bound; valid envelopes retain the core ceiling.
const BOUNDED_PENDING_ROW_PROJECTION: &str = "
    rowid, timestamp, length(CAST(envelope AS BLOB)),
    CASE WHEN typeof(message_id) = 'blob' AND length(message_id) = 16
         THEN message_id ELSE NULL END,
    CASE WHEN typeof(sender) = 'blob' AND length(sender) = 32
         THEN sender ELSE NULL END,
    CASE WHEN typeof(receiver) = 'blob' AND length(receiver) = 32
         THEN receiver ELSE NULL END,
    CASE WHEN typeof(envelope) = 'blob'
                   AND length(envelope) <= :max_envelope_bytes
         THEN envelope ELSE NULL END";

// [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Materialize only
// SQL-admitted fields. Preserve row identity and original byte length for the
// existing quarantine transaction; SQL/type/conversion failures propagate.
fn read_bounded_pending_row(row: &Row<'_>, base: usize) -> rusqlite::Result<StoredPendingMessageRow> {
    let rowid = row.get(base)?;
    let timestamp = row.get(base + 1)?;
    let encoded_bytes = match row.get::<_, Option<i64>>(base + 2)? {
        Some(length) => u64::try_from(length).map_err(|_| rusqlite::Error::InvalidQuery)?,
        None => 0,
    };
    let message_id: Option<Vec<u8>> = row.get(base + 3)?;
    let sender: Option<Vec<u8>> = row.get(base + 4)?;
    let receiver: Option<Vec<u8>> = row.get(base + 5)?;
    let envelope: Option<Vec<u8>> = row.get(base + 6)?;
    let reason = if message_id.is_none() {
        Some("pending_message_id")
    } else if sender.is_none() {
        Some("pending_message_sender")
    } else if receiver.is_none() {
        Some("pending_message_receiver")
    } else if envelope.is_none() {
        Some("pending_message_envelope")
    } else {
        None
    };
    Ok(StoredPendingMessageRow {
        rowid,
        message_id: message_id.unwrap_or_default(),
        sender: sender.unwrap_or_default(),
        receiver: receiver.unwrap_or_default(),
        timestamp,
        envelope: envelope.unwrap_or_default(),
        projection_rejection: reason.map(|reason| CorruptDurableRow {
            row_key: rowid,
            source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
            reason,
            encoded_bytes,
        }),
    })
}

impl PendingMessagePullRepository for SqlitePendingMessagePullRepository {
    fn capture_snapshot_ceiling(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: i64,
    ) -> ChatRelayResult<u64> {
        let ceiling = conn.query_row(
            "SELECT COALESCE(MAX(queue_sequence), 0)
             FROM pending_messages
             WHERE receiver = ?1
               AND status = 0
               AND timestamp > ?2
               AND queue_sequence > 0",
            params![receiver.as_slice(), after_timestamp],
            |row| row.get::<_, i64>(0),
        )?;
        u64::try_from(ceiling).map_err(|_| ChatRelayError::CorruptStoredData {
            field: "pending_message_snapshot_ceiling",
        })
    }

    fn read_legacy_rows(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: i64,
        cursor: &[u8; 16],
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredPendingMessageRow>> {
        // [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Project, do
        // not filter invalid rows: they still count toward LIMIT/quarantine.
        let sql = format!(
            "SELECT {BOUNDED_PENDING_ROW_PROJECTION}
             FROM pending_messages
             WHERE receiver = :receiver
               AND status = 0
               AND timestamp > :after_timestamp
               AND message_id > :cursor
             ORDER BY message_id ASC
             LIMIT :row_limit"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(
                named_params! {
                    ":receiver": receiver.as_slice(),
                    ":after_timestamp": after_timestamp,
                    ":cursor": cursor.as_slice(),
                    ":row_limit": limit,
                    ":max_envelope_bytes": MAX_CHAT_ENVELOPE_BYTES,
                },
                |row| read_bounded_pending_row(row, 0),
            )?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }

    fn read_snapshot_rows(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: i64,
        position: i64,
        ceiling: i64,
        limit: i64,
    ) -> ChatRelayResult<Vec<StoredSequencedPendingMessageRow>> {
        // [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Keep the
        // same sequence prefix, including rejected rows and the lookahead row.
        let sql = format!(
            "SELECT queue_sequence, {BOUNDED_PENDING_ROW_PROJECTION}
             FROM pending_messages
             WHERE receiver = :receiver
               AND status = 0
               AND timestamp > :after_timestamp
               AND queue_sequence > :position
               AND queue_sequence <= :ceiling
             ORDER BY queue_sequence ASC
             LIMIT :row_limit"
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(
                named_params! {
                    ":receiver": receiver.as_slice(),
                    ":after_timestamp": after_timestamp,
                    ":position": position,
                    ":ceiling": ceiling,
                    ":row_limit": limit,
                    ":max_envelope_bytes": MAX_CHAT_ENVELOPE_BYTES,
                },
                |row| {
                    Ok(StoredSequencedPendingMessageRow {
                        queue_sequence: row.get(0)?,
                        row: read_bounded_pending_row(row, 1)?,
                    })
                },
            )?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        Ok(rows)
    }
}

/// Composed pending-pull read and validation domain.
pub(crate) struct PendingMessagePullDomain<R = SqlitePendingMessagePullRepository> {
    repository: R,
}

impl PendingMessagePullDomain<SqlitePendingMessagePullRepository> {
    pub(crate) fn new() -> Self {
        Self::with_repository(SqlitePendingMessagePullRepository)
    }
}

impl<R: PendingMessagePullRepository> PendingMessagePullDomain<R> {
    fn with_repository(repository: R) -> Self {
        Self { repository }
    }

    pub(crate) fn capture_snapshot_ceiling(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: u64,
    ) -> ChatRelayResult<u64> {
        self.repository.capture_snapshot_ceiling(
            conn,
            receiver,
            sqlite_timestamp_filter(after_timestamp),
        )
    }

    pub(crate) fn read_legacy_page(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        page_limit: usize,
    ) -> ChatRelayResult<LegacyPendingPullPage> {
        let effective_limit = page_limit.saturating_add(1);
        let rows = self.repository.read_legacy_rows(
            conn,
            receiver,
            sqlite_timestamp_filter(after_timestamp),
            cursor,
            sqlite_limit(effective_limit),
        )?;
        let raw_has_more = rows.len() == effective_limit;
        let mut messages = Vec::with_capacity(rows.len().min(page_limit));
        let mut corrupt_rows = Vec::new();
        for row in rows {
            match validate_pending_message_row(row, receiver) {
                Ok(message) => messages.push(message),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }
        Ok(LegacyPendingPullPage {
            messages,
            corrupt_rows,
            raw_has_more,
        })
    }

    pub(crate) fn read_snapshot_page(
        &self,
        conn: &Connection,
        receiver: &[u8; 32],
        after_timestamp: u64,
        position: u64,
        ceiling: u64,
        page_limit: usize,
    ) -> ChatRelayResult<SnapshotPendingPullPage> {
        let query_position =
            i64::try_from(position).map_err(|_| ChatRelayError::InvalidPullCursor)?;
        let query_ceiling =
            i64::try_from(ceiling).map_err(|_| ChatRelayError::InvalidPullCursor)?;
        let effective_limit = page_limit.saturating_add(1);
        let rows = self.repository.read_snapshot_rows(
            conn,
            receiver,
            sqlite_timestamp_filter(after_timestamp),
            query_position,
            query_ceiling,
            sqlite_limit(effective_limit),
        )?;

        let raw_has_more = rows.len() == effective_limit;
        let raw_max_sequence = rows
            .last()
            .and_then(|row| u64::try_from(row.queue_sequence).ok());
        let mut messages = Vec::with_capacity(rows.len().min(page_limit));
        let mut corrupt_rows = Vec::new();
        for stored in rows {
            let sequence = match u64::try_from(stored.queue_sequence) {
                Ok(sequence) if sequence > 0 => sequence,
                _ => {
                    corrupt_rows.push(corrupt_sequence_row(&stored.row));
                    continue;
                }
            };
            match validate_pending_message_row(stored.row, receiver) {
                Ok(message) => messages.push((sequence, message)),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }
        Ok(SnapshotPendingPullPage {
            messages,
            corrupt_rows,
            raw_has_more,
            raw_max_sequence,
        })
    }
}

fn sqlite_timestamp_filter(timestamp: u64) -> i64 {
    i64::try_from(timestamp).unwrap_or(i64::MAX)
}

fn sqlite_limit(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX)
}

fn corrupt_sequence_row(row: &StoredPendingMessageRow) -> CorruptDurableRow {
    CorruptDurableRow {
        row_key: row.rowid,
        source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
        reason: "pending_message_queue_sequence",
        // [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Sequence
        // corruption retains priority without losing the unmaterialized size.
        encoded_bytes: row.projection_rejection.map_or_else(
            || u64::try_from(row.envelope.len()).unwrap_or(u64::MAX),
            |rejection| rejection.encoded_bytes,
        ),
    }
}

fn validate_pending_message_row(
    row: StoredPendingMessageRow,
    expected_receiver: &[u8; 32],
) -> Result<PendingMessage, CorruptDurableRow> {
    // [PENDING-PULL-BOUNDED-PROJECTION 2026-10-04 by Codex] Rejected
    // projections cannot be decoded or returned as custody. Accepted rows
    // still pass every existing codec, binding, and signature check below.
    if let Some(rejection) = row.projection_rejection {
        return Err(rejection);
    }
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
        .map_err(|_| corrupt("pending_message_id"))?;
    let stored_sender: [u8; 32] = row
        .sender
        .try_into()
        .map_err(|_| corrupt("pending_message_sender"))?;
    let stored_receiver: [u8; 32] = row
        .receiver
        .try_into()
        .map_err(|_| corrupt("pending_message_receiver"))?;
    let stored_timestamp =
        u64::try_from(row.timestamp).map_err(|_| corrupt("pending_message_timestamp"))?;
    if stored_receiver != *expected_receiver {
        return Err(corrupt("pending_message_receiver_mismatch"));
    }
    let envelope =
        decode_envelope(&row.envelope).map_err(|_| corrupt("pending_message_envelope"))?;
    if envelope.message_id != message_id {
        return Err(corrupt("pending_message_id_mismatch"));
    }
    if envelope.receiver != *expected_receiver {
        return Err(corrupt("pending_message_envelope_receiver_mismatch"));
    }
    if envelope.sender != stored_sender {
        return Err(corrupt("pending_message_sender_mismatch"));
    }
    if envelope.timestamp != stored_timestamp {
        return Err(corrupt("pending_message_timestamp_mismatch"));
    }
    envelope
        .verify_signature()
        .map_err(|_| corrupt("pending_message_signature"))?;
    Ok(PendingMessage {
        message_id,
        envelope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::{encode_envelope, ChatContentType, ChatEnvelope};

    struct StubRepository {
        ceiling: u64,
        legacy_rows: Vec<StoredPendingMessageRow>,
        snapshot_rows: Vec<StoredSequencedPendingMessageRow>,
    }

    impl PendingMessagePullRepository for StubRepository {
        fn capture_snapshot_ceiling(
            &self,
            _conn: &Connection,
            _receiver: &[u8; 32],
            _after_timestamp: i64,
        ) -> ChatRelayResult<u64> {
            Ok(self.ceiling)
        }

        fn read_legacy_rows(
            &self,
            _conn: &Connection,
            _receiver: &[u8; 32],
            _after_timestamp: i64,
            _cursor: &[u8; 16],
            _limit: i64,
        ) -> ChatRelayResult<Vec<StoredPendingMessageRow>> {
            Ok(self.legacy_rows.clone())
        }

        fn read_snapshot_rows(
            &self,
            _conn: &Connection,
            _receiver: &[u8; 32],
            _after_timestamp: i64,
            _position: i64,
            _ceiling: i64,
            _limit: i64,
        ) -> ChatRelayResult<Vec<StoredSequencedPendingMessageRow>> {
            Ok(self.snapshot_rows.clone())
        }
    }

    fn stored_row(rowid: i64, envelope: &ChatEnvelope) -> StoredPendingMessageRow {
        StoredPendingMessageRow {
            rowid,
            message_id: envelope.message_id.to_vec(),
            sender: envelope.sender.to_vec(),
            receiver: envelope.receiver.to_vec(),
            timestamp: i64::try_from(envelope.timestamp).expect("test timestamp fits SQLite"),
            envelope: encode_envelope(envelope).expect("encode test envelope"),
            projection_rejection: None,
        }
    }

    fn signed_envelope(receiver: [u8; 32]) -> ChatEnvelope {
        let identity = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [0x41; 16],
            sender: identity.public_key_bytes(),
            receiver,
            timestamp: 1_800_000_000,
            ciphertext: b"opaque-ciphertext".to_vec(),
            nonce: [0x42; 24],
            content_type: ChatContentType::Text,
            signature: [0; 64],
        };
        envelope.signature = identity.sign(&envelope.sign_data());
        envelope
    }

    #[test]
    fn composed_domain_validates_rows_and_classifies_sequence_corruption() {
        let receiver = [0x43; 32];
        let envelope = signed_envelope(receiver);
        let valid = stored_row(1, &envelope);
        let mut corrupt = stored_row(2, &envelope);
        corrupt.sender = vec![0x44; 31];
        let repository = StubRepository {
            ceiling: 2,
            legacy_rows: vec![valid.clone(), corrupt.clone()],
            snapshot_rows: vec![
                StoredSequencedPendingMessageRow {
                    queue_sequence: 1,
                    row: valid,
                },
                StoredSequencedPendingMessageRow {
                    queue_sequence: 0,
                    row: corrupt,
                },
            ],
        };
        let domain = PendingMessagePullDomain::with_repository(repository);
        let conn = Connection::open_in_memory().expect("open test connection");

        assert_eq!(
            domain
                .capture_snapshot_ceiling(&conn, &receiver, 0)
                .expect("capture stub ceiling"),
            2
        );
        let legacy = domain
            .read_legacy_page(&conn, &receiver, 0, &[0; 16], 10)
            .expect("read legacy page");
        assert_eq!(legacy.messages.len(), 1);
        assert_eq!(legacy.corrupt_rows.len(), 1);
        assert_eq!(legacy.corrupt_rows[0].reason, "pending_message_sender");

        let snapshot = domain
            .read_snapshot_page(&conn, &receiver, 0, 0, 2, 10)
            .expect("read snapshot page");
        assert_eq!(snapshot.messages.len(), 1);
        assert_eq!(snapshot.messages[0].0, 1);
        assert_eq!(snapshot.corrupt_rows.len(), 1);
        assert_eq!(
            snapshot.corrupt_rows[0].reason,
            "pending_message_queue_sequence"
        );
    }
}
