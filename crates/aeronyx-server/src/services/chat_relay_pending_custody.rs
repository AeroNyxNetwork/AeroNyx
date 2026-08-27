// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_pending_custody.rs
// ============================================
// Version: 1.0.0-PendingCustodyDomain
//
// Creation Reason:
//   [CHAT-PENDING-CUSTODY-DOMAIN 2026-08-25 by Codex] Extract durable offline
//   message writes and receiver-bound acknowledgements from the oversized
//   relay service without changing its public API or SQLite schema.
//
// Main Functionality:
//   - Defines the pending-message custody policy as a domain value.
//   - Defines a replaceable repository trait for durable store and ACK writes.
//   - Implements idempotence, quotas, monotonic sequence allocation, and ACKs.
//   - Keeps ciphertext opaque and returns only aggregate write outcomes.
//
// Dependencies:
//   - `config_chat_relay.rs` supplies validated custody limits.
//   - `chat_relay.rs` owns the connection lock, API surface, and safe logging.
//   - `aeronyx-core` owns the signed encrypted envelope wire codec.
//
// Main Logical Flow:
//   1. Validate the envelope against the configured item-size boundary.
//   2. Encode it and enter one immediate SQLite transaction.
//   3. Resolve exact retry/conflict before evaluating durable quotas.
//   4. Allocate a sequence and insert, or delete receiver-bound ACK rows.
//
// Important Note for Next Developer:
//   - Exact retries must succeed before quota checks and consume no sequence.
//   - Sequence allocation and insertion must remain in the same transaction.
//   - ACK deletion must always bind both message ID and receiver identity.
//   - Never log or expose message IDs, wallet keys, envelopes, or ciphertext.
//
// Last Modified:
//   v1.0.0-PendingCustodyDomain - Initial custody repository composition
// ============================================

use std::collections::HashSet;

use aeronyx_core::protocol::chat::{encode_envelope, ChatEnvelope};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

use crate::config::ChatRelayConfig;

use super::chat_relay::MAX_CHAT_ACK_MESSAGE_IDS;
// [CHAT-RELAY-ERROR-DOMAIN 2026-08-27 by Codex] Custody repositories consume
// typed failures directly while the public ACK ceiling stays service-owned.
use super::chat_relay_error::{ChatRelayError, ChatRelayResult};

/// Immutable limits governing one node's pending-message custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PendingMessageCustodyPolicy {
    max_message_size: usize,
    max_pending_per_wallet: usize,
    max_pending_messages_total: usize,
    max_pending_message_bytes_total: u64,
}

impl From<&ChatRelayConfig> for PendingMessageCustodyPolicy {
    fn from(config: &ChatRelayConfig) -> Self {
        Self {
            max_message_size: config.max_message_size,
            max_pending_per_wallet: config.max_pending_per_wallet,
            max_pending_messages_total: config.max_pending_messages_total,
            max_pending_message_bytes_total: config.max_pending_message_bytes_total,
        }
    }
}

/// Owned, already-encoded write model passed to the storage repository.
#[derive(Debug)]
pub(crate) struct PendingMessageWrite {
    message_id: [u8; 16],
    sender: [u8; 32],
    receiver: [u8; 32],
    timestamp: i64,
    envelope: Vec<u8>,
    received_at: i64,
}

/// Validated and deduplicated receiver-bound acknowledgement command.
pub(crate) struct PendingMessageAckBatch {
    message_ids: HashSet<[u8; 16]>,
}

/// Coarse result of a durable custody write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingMessageStoreOutcome {
    /// A new encrypted envelope was committed.
    Stored { encoded_bytes: u64 },
    /// An exact already-durable retry was accepted without mutation.
    AlreadyStored,
}

/// Replaceable persistence capability for pending-message custody.
///
/// [CHAT-PENDING-CUSTODY-DOMAIN 2026-08-25 by Codex] The domain owns input
/// validation and policy. Implementations own atomic durable state changes and
/// must preserve receiver binding and retry idempotence.
pub(crate) trait PendingMessageCustodyRepository: Send + Sync {
    fn store(
        &self,
        conn: &mut Connection,
        write: PendingMessageWrite,
        policy: PendingMessageCustodyPolicy,
    ) -> ChatRelayResult<PendingMessageStoreOutcome>;

    fn acknowledge(
        &self,
        conn: &mut Connection,
        message_ids: &HashSet<[u8; 16]>,
        receiver: &[u8; 32],
    ) -> ChatRelayResult<usize>;
}

/// Production SQLite repository for pending-message custody.
pub(crate) struct SqlitePendingMessageCustodyRepository;

impl PendingMessageCustodyRepository for SqlitePendingMessageCustodyRepository {
    fn store(
        &self,
        conn: &mut Connection,
        write: PendingMessageWrite,
        policy: PendingMessageCustodyPolicy,
    ) -> ChatRelayResult<PendingMessageStoreOutcome> {
        let encoded_bytes = u64::try_from(write.envelope.len()).unwrap_or(u64::MAX);
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Retry identity is resolved before quotas so an already-durable write
        // remains successful even while the node or mailbox is full.
        let existing_envelope = tx
            .query_row(
                "SELECT envelope FROM pending_messages WHERE message_id = ?1",
                params![write.message_id.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        if let Some(existing_envelope) = existing_envelope {
            if existing_envelope == write.envelope {
                tx.commit()?;
                return Ok(PendingMessageStoreOutcome::AlreadyStored);
            }
            return Err(ChatRelayError::MessageIdConflict);
        }

        let (pending_messages, pending_message_bytes) = read_pending_usage(&tx)?;
        if pending_messages >= u64::try_from(policy.max_pending_messages_total).unwrap_or(u64::MAX)
        {
            return Err(ChatRelayError::PendingMessageQueueFull {
                current: usize::try_from(pending_messages).unwrap_or(usize::MAX),
                limit: policy.max_pending_messages_total,
            });
        }
        if pending_message_bytes.saturating_add(encoded_bytes)
            > policy.max_pending_message_bytes_total
        {
            return Err(ChatRelayError::PendingMessageBytesExceeded {
                current: pending_message_bytes,
                incoming: encoded_bytes,
                limit: policy.max_pending_message_bytes_total,
            });
        }

        let mailbox_count = tx.query_row(
            "SELECT COUNT(*) FROM pending_messages WHERE receiver = ?1 AND status = 0",
            params![write.receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )?;
        let mailbox_count = usize::try_from(mailbox_count.max(0)).unwrap_or(usize::MAX);
        if mailbox_count >= policy.max_pending_per_wallet {
            return Err(ChatRelayError::MailboxFull {
                current: mailbox_count,
                limit: policy.max_pending_per_wallet,
            });
        }

        let queue_sequence = allocate_queue_sequence(&tx)?;
        tx.execute(
            "INSERT INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status,
              queue_sequence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0, ?7)",
            params![
                write.message_id.as_slice(),
                write.sender.as_slice(),
                write.receiver.as_slice(),
                write.timestamp,
                write.envelope,
                write.received_at,
                queue_sequence,
            ],
        )?;
        tx.commit()?;
        Ok(PendingMessageStoreOutcome::Stored { encoded_bytes })
    }

    fn acknowledge(
        &self,
        conn: &mut Connection,
        message_ids: &HashSet<[u8; 16]>,
        receiver: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut deleted = 0usize;
        for message_id in message_ids {
            deleted = deleted.saturating_add(tx.execute(
                "DELETE FROM pending_messages
                 WHERE message_id = ?1 AND receiver = ?2",
                params![message_id.as_slice(), receiver.as_slice()],
            )?);
        }
        tx.commit()?;
        Ok(deleted)
    }
}

/// Composed pending-message write and acknowledgement domain.
pub(crate) struct PendingMessageCustodyDomain<R = SqlitePendingMessageCustodyRepository> {
    repository: R,
    policy: PendingMessageCustodyPolicy,
}

impl PendingMessageCustodyDomain<SqlitePendingMessageCustodyRepository> {
    pub(crate) fn new(config: &ChatRelayConfig) -> Self {
        Self::with_repository(
            SqlitePendingMessageCustodyRepository,
            PendingMessageCustodyPolicy::from(config),
        )
    }
}

impl<R: PendingMessageCustodyRepository> PendingMessageCustodyDomain<R> {
    fn with_repository(repository: R, policy: PendingMessageCustodyPolicy) -> Self {
        Self { repository, policy }
    }

    pub(crate) fn prepare_store(
        &self,
        envelope: &ChatEnvelope,
        received_at: u64,
    ) -> ChatRelayResult<PendingMessageWrite> {
        if envelope.ciphertext.len() > self.policy.max_message_size {
            return Err(ChatRelayError::MessageTooLarge {
                size: envelope.ciphertext.len(),
                limit: self.policy.max_message_size,
            });
        }
        let timestamp =
            i64::try_from(envelope.timestamp).map_err(|_| ChatRelayError::TimestampOutOfRange)?;
        Ok(PendingMessageWrite {
            message_id: envelope.message_id,
            sender: envelope.sender,
            receiver: envelope.receiver,
            timestamp,
            envelope: encode_envelope(envelope)?,
            received_at: i64::try_from(received_at).unwrap_or(i64::MAX),
        })
    }

    pub(crate) fn store(
        &self,
        conn: &mut Connection,
        write: PendingMessageWrite,
    ) -> ChatRelayResult<PendingMessageStoreOutcome> {
        self.repository.store(conn, write, self.policy)
    }

    pub(crate) fn prepare_acknowledgement(
        &self,
        message_ids: &[[u8; 16]],
    ) -> ChatRelayResult<Option<PendingMessageAckBatch>> {
        if message_ids.is_empty() {
            return Ok(None);
        }
        if message_ids.len() > MAX_CHAT_ACK_MESSAGE_IDS {
            return Err(ChatRelayError::AckBatchTooLarge {
                size: message_ids.len(),
                limit: MAX_CHAT_ACK_MESSAGE_IDS,
            });
        }
        Ok(Some(PendingMessageAckBatch {
            message_ids: message_ids.iter().copied().collect(),
        }))
    }

    pub(crate) fn acknowledge(
        &self,
        conn: &mut Connection,
        batch: &PendingMessageAckBatch,
        receiver: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        self.repository
            .acknowledge(conn, &batch.message_ids, receiver)
    }
}

fn read_pending_usage(conn: &Connection) -> ChatRelayResult<(u64, u64)> {
    let counters = conn.query_row(
        "SELECT pending_message_count, pending_message_bytes
         FROM relay_storage_usage
         WHERE singleton = 1",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;
    Ok((
        nonnegative_counter(counters.0, "pending_message_count")?,
        nonnegative_counter(counters.1, "pending_message_bytes")?,
    ))
}

fn nonnegative_counter(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

pub(crate) fn allocate_queue_sequence(tx: &Transaction<'_>) -> ChatRelayResult<i64> {
    let updated = tx.execute(
        "UPDATE relay_queue_sequence
         SET last_sequence = last_sequence + 1
         WHERE singleton = 1 AND last_sequence < ?1",
        params![i64::MAX],
    )?;
    if updated != 1 {
        return Err(ChatRelayError::QueueSequenceExhausted);
    }
    let sequence = tx.query_row(
        "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
        [],
        |row| row.get::<_, i64>(0),
    )?;
    if sequence <= 0 {
        return Err(ChatRelayError::CorruptStoredData {
            field: "relay_queue_sequence_nonpositive",
        });
    }
    Ok(sequence)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct StubRepository {
        stores: AtomicUsize,
        acknowledgements: AtomicUsize,
    }

    impl PendingMessageCustodyRepository for StubRepository {
        fn store(
            &self,
            _conn: &mut Connection,
            _write: PendingMessageWrite,
            _policy: PendingMessageCustodyPolicy,
        ) -> ChatRelayResult<PendingMessageStoreOutcome> {
            self.stores.fetch_add(1, Ordering::Relaxed);
            Ok(PendingMessageStoreOutcome::Stored { encoded_bytes: 1 })
        }

        fn acknowledge(
            &self,
            _conn: &mut Connection,
            message_ids: &HashSet<[u8; 16]>,
            _receiver: &[u8; 32],
        ) -> ChatRelayResult<usize> {
            self.acknowledgements.fetch_add(1, Ordering::Relaxed);
            Ok(message_ids.len())
        }
    }

    fn test_policy(max_message_size: usize) -> PendingMessageCustodyPolicy {
        PendingMessageCustodyPolicy {
            max_message_size,
            max_pending_per_wallet: 10,
            max_pending_messages_total: 100,
            max_pending_message_bytes_total: 1_000,
        }
    }

    fn test_envelope(ciphertext: Vec<u8>) -> ChatEnvelope {
        ChatEnvelope {
            message_id: [0x11; 16],
            sender: [0x12; 32],
            receiver: [0x13; 32],
            timestamp: 42,
            ciphertext,
            nonce: [0x14; 24],
            content_type: aeronyx_core::protocol::chat::ChatContentType::Text,
            signature: [0x15; 64],
        }
    }

    #[test]
    fn composed_domain_validates_before_repository_and_deduplicates_ack_ids() {
        let repository = StubRepository {
            stores: AtomicUsize::new(0),
            acknowledgements: AtomicUsize::new(0),
        };
        let domain = PendingMessageCustodyDomain::with_repository(repository, test_policy(4));
        let mut conn = Connection::open_in_memory().expect("open test connection");

        assert!(matches!(
            domain.prepare_store(&test_envelope(vec![0; 5]), 1),
            Err(ChatRelayError::MessageTooLarge { size: 5, limit: 4 })
        ));
        assert_eq!(domain.repository.stores.load(Ordering::Relaxed), 0);

        let write = domain
            .prepare_store(&test_envelope(vec![0; 4]), 1)
            .expect("prepare bounded store");
        assert_eq!(
            domain.store(&mut conn, write).expect("store through stub"),
            PendingMessageStoreOutcome::Stored { encoded_bytes: 1 }
        );
        assert_eq!(domain.repository.stores.load(Ordering::Relaxed), 1);

        assert!(domain
            .prepare_acknowledgement(&[])
            .expect("prepare empty ACK batch")
            .is_none());
        assert_eq!(
            domain.repository.acknowledgements.load(Ordering::Relaxed),
            0
        );

        let ids = [[0x21; 16], [0x21; 16], [0x22; 16]];
        let batch = domain
            .prepare_acknowledgement(&ids)
            .expect("prepare ACK batch")
            .expect("non-empty ACK batch");
        assert_eq!(
            domain
                .acknowledge(&mut conn, &batch, &[0x23; 32])
                .expect("acknowledge unique IDs"),
            2
        );
        assert_eq!(
            domain.repository.acknowledgements.load(Ordering::Relaxed),
            1
        );
    }
}
