// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_blob_custody.rs
// ============================================
// Version: 1.0.0-BlobCustodyDomain
//
// Creation Reason:
//   [CHAT-BLOB-CUSTODY-DOMAIN 2026-08-25 by Codex] Extract opaque encrypted
//   blob identity, quotas, persistence, retrieval, and authorized deletion from
//   the oversized relay service without changing public APIs or SQLite schema.
//
// Main Functionality:
//   - Defines immutable blob-custody limits as a domain policy.
//   - Derives node-bound, opaque blob identifiers with HMAC-SHA256.
//   - Defines a replaceable repository trait for put, get, and sender deletion.
//   - Enforces idempotence and node/receiver quotas in immediate transactions.
//
// Dependencies:
//   - `config_chat_relay.rs` supplies validated encrypted-blob limits.
//   - `chat_relay.rs` owns connection locking, public APIs, and safe telemetry.
//   - `rusqlite` provides the production durable repository implementation.
//
// Main Logical Flow:
//   1. Validate size and derive a node-secret-bound identifier before locking.
//   2. Resolve an existing identifier before evaluating durable quotas.
//   3. Insert opaque bytes atomically, or retrieve and mark them downloaded.
//   4. Delete only when the authenticated requester matches the stored sender.
//
// Important Note for Next Developer:
//   - Exact identifier retries must remain successful before quota checks.
//   - Blob bytes are encrypted client payloads; never parse or log their data.
//   - Download intentionally uses the unguessable blob ID as a capability.
//   - Sender-bound deletion and aggregate-only service logs must be preserved.
//
// Last Modified:
//   v1.0.0-BlobCustodyDomain - Initial blob repository composition
// ============================================

use hmac::{Hmac, Mac};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use sha2::Sha256;

use crate::config::ChatRelayConfig;

use super::chat_relay::{ChatRelayError, ChatRelayResult};

type HmacSha256 = Hmac<Sha256>;

/// Immutable limits governing one node's encrypted blob custody.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EncryptedBlobCustodyPolicy {
    max_blob_size: usize,
    max_blobs_per_receiver: usize,
    max_pending_blobs_total: usize,
    max_pending_blob_bytes_total: u64,
}

impl From<&ChatRelayConfig> for EncryptedBlobCustodyPolicy {
    fn from(config: &ChatRelayConfig) -> Self {
        Self {
            max_blob_size: config.max_blob_size,
            max_blobs_per_receiver: config.max_blobs_per_receiver,
            max_pending_blobs_total: config.max_pending_blobs_total,
            max_pending_blob_bytes_total: config.max_pending_blob_bytes_total,
        }
    }
}

/// Validated borrowed write model passed to the durable repository.
pub(crate) struct EncryptedBlobWrite<'a> {
    blob_id: String,
    sender: &'a [u8; 32],
    receiver: &'a [u8; 32],
    data: &'a [u8],
    received_at: i64,
}

/// Coarse result of an idempotent encrypted-blob write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EncryptedBlobStoreOutcome {
    /// A new opaque encrypted object was committed.
    Stored { blob_id: String, size: usize },
    /// The derived identifier was already durable and was accepted unchanged.
    AlreadyStored { blob_id: String },
}

impl EncryptedBlobStoreOutcome {
    pub(crate) fn blob_id(&self) -> &str {
        match self {
            Self::Stored { blob_id, .. } | Self::AlreadyStored { blob_id } => blob_id,
        }
    }
}

/// Result of a sender-bound encrypted-blob deletion attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EncryptedBlobDeleteOutcome {
    Deleted,
    Unauthorized,
    NotFound,
}

/// Replaceable persistence capability for encrypted blob custody.
///
/// [CHAT-BLOB-CUSTODY-DOMAIN 2026-08-25 by Codex] Repository implementations
/// own atomic storage changes. Identity derivation, input validation, policy,
/// and public error mapping remain in the composed domain.
pub(crate) trait EncryptedBlobCustodyRepository: Send + Sync {
    fn put(
        &self,
        conn: &mut Connection,
        write: EncryptedBlobWrite<'_>,
        policy: EncryptedBlobCustodyPolicy,
    ) -> ChatRelayResult<EncryptedBlobStoreOutcome>;

    fn get(&self, conn: &Connection, blob_id: &str) -> ChatRelayResult<Option<Vec<u8>>>;

    fn delete(
        &self,
        conn: &Connection,
        blob_id: &str,
        requester: &[u8; 32],
    ) -> ChatRelayResult<EncryptedBlobDeleteOutcome>;
}

/// Production SQLite repository for encrypted blob custody.
pub(crate) struct SqliteEncryptedBlobCustodyRepository;

impl EncryptedBlobCustodyRepository for SqliteEncryptedBlobCustodyRepository {
    fn put(
        &self,
        conn: &mut Connection,
        write: EncryptedBlobWrite<'_>,
        policy: EncryptedBlobCustodyPolicy,
    ) -> ChatRelayResult<EncryptedBlobStoreOutcome> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Existing opaque identities are resolved before quotas so retries
        // remain successful while the node or target receiver is at capacity.
        let duplicate = tx
            .query_row(
                "SELECT 1 FROM pending_blobs WHERE blob_id = ?1",
                params![&write.blob_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if duplicate {
            tx.commit()?;
            return Ok(EncryptedBlobStoreOutcome::AlreadyStored {
                blob_id: write.blob_id,
            });
        }

        let (pending_blobs, pending_blob_bytes) = read_blob_usage(&tx)?;
        if pending_blobs >= u64::try_from(policy.max_pending_blobs_total).unwrap_or(u64::MAX) {
            return Err(ChatRelayError::PendingBlobStoreFull {
                current: usize::try_from(pending_blobs).unwrap_or(usize::MAX),
                limit: policy.max_pending_blobs_total,
            });
        }
        let incoming_bytes = u64::try_from(write.data.len()).unwrap_or(u64::MAX);
        if pending_blob_bytes.saturating_add(incoming_bytes) > policy.max_pending_blob_bytes_total {
            return Err(ChatRelayError::PendingBlobBytesExceeded {
                current: pending_blob_bytes,
                incoming: incoming_bytes,
                limit: policy.max_pending_blob_bytes_total,
            });
        }

        let receiver_count = tx.query_row(
            "SELECT COUNT(*) FROM pending_blobs WHERE receiver = ?1",
            params![write.receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )?;
        let receiver_count = usize::try_from(receiver_count.max(0)).unwrap_or(usize::MAX);
        if receiver_count >= policy.max_blobs_per_receiver {
            return Err(ChatRelayError::BlobQuotaExceeded {
                current: receiver_count,
                limit: policy.max_blobs_per_receiver,
            });
        }

        tx.execute(
            "INSERT INTO pending_blobs
             (blob_id, sender, receiver, data, size, received_at, downloaded)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![
                &write.blob_id,
                write.sender.as_slice(),
                write.receiver.as_slice(),
                write.data,
                i64::try_from(write.data.len()).unwrap_or(i64::MAX),
                write.received_at,
            ],
        )?;
        tx.commit()?;
        Ok(EncryptedBlobStoreOutcome::Stored {
            blob_id: write.blob_id,
            size: write.data.len(),
        })
    }

    fn get(&self, conn: &Connection, blob_id: &str) -> ChatRelayResult<Option<Vec<u8>>> {
        let data = conn
            .query_row(
                "SELECT data FROM pending_blobs WHERE blob_id = ?1",
                params![blob_id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        if data.is_some() {
            // Downloaded is a cleanup hint. Preserve successful capability
            // reads even if this non-critical hint cannot be updated.
            let _ = conn.execute(
                "UPDATE pending_blobs SET downloaded = 1 WHERE blob_id = ?1",
                params![blob_id],
            );
        }
        Ok(data)
    }

    fn delete(
        &self,
        conn: &Connection,
        blob_id: &str,
        requester: &[u8; 32],
    ) -> ChatRelayResult<EncryptedBlobDeleteOutcome> {
        let deleted = conn.execute(
            "DELETE FROM pending_blobs WHERE blob_id = ?1 AND sender = ?2",
            params![blob_id, requester.as_slice()],
        )?;
        if deleted == 1 {
            return Ok(EncryptedBlobDeleteOutcome::Deleted);
        }
        let exists = conn
            .query_row(
                "SELECT 1 FROM pending_blobs WHERE blob_id = ?1",
                params![blob_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        Ok(if exists {
            EncryptedBlobDeleteOutcome::Unauthorized
        } else {
            EncryptedBlobDeleteOutcome::NotFound
        })
    }
}

/// Composed encrypted-blob identity, policy, and persistence domain.
pub(crate) struct EncryptedBlobCustodyDomain<R = SqliteEncryptedBlobCustodyRepository> {
    repository: R,
    policy: EncryptedBlobCustodyPolicy,
    node_secret: [u8; 32],
}

impl EncryptedBlobCustodyDomain<SqliteEncryptedBlobCustodyRepository> {
    pub(crate) fn new(node_secret: [u8; 32], config: &ChatRelayConfig) -> Self {
        Self::with_repository(
            SqliteEncryptedBlobCustodyRepository,
            EncryptedBlobCustodyPolicy::from(config),
            node_secret,
        )
    }
}

impl<R: EncryptedBlobCustodyRepository> EncryptedBlobCustodyDomain<R> {
    fn with_repository(
        repository: R,
        policy: EncryptedBlobCustodyPolicy,
        node_secret: [u8; 32],
    ) -> Self {
        Self {
            repository,
            policy,
            node_secret,
        }
    }

    pub(crate) fn compute_blob_id(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        file_hash: &[u8; 32],
    ) -> String {
        let mut mac =
            HmacSha256::new_from_slice(&self.node_secret).expect("HMAC accepts any key length");
        mac.update(sender);
        mac.update(receiver);
        mac.update(file_hash);
        let result = mac.finalize().into_bytes();
        hex::encode(&result[..16])
    }

    pub(crate) fn prepare_put<'a>(
        &self,
        sender: &'a [u8; 32],
        receiver: &'a [u8; 32],
        data: &'a [u8],
        file_hash: &[u8; 32],
        received_at: u64,
    ) -> ChatRelayResult<EncryptedBlobWrite<'a>> {
        if data.len() > self.policy.max_blob_size {
            return Err(ChatRelayError::BlobTooLarge {
                size: data.len(),
                limit: self.policy.max_blob_size,
            });
        }
        Ok(EncryptedBlobWrite {
            blob_id: self.compute_blob_id(sender, receiver, file_hash),
            sender,
            receiver,
            data,
            received_at: i64::try_from(received_at).unwrap_or(i64::MAX),
        })
    }

    pub(crate) fn put(
        &self,
        conn: &mut Connection,
        write: EncryptedBlobWrite<'_>,
    ) -> ChatRelayResult<EncryptedBlobStoreOutcome> {
        self.repository.put(conn, write, self.policy)
    }

    pub(crate) fn get(&self, conn: &Connection, blob_id: &str) -> ChatRelayResult<Vec<u8>> {
        self.repository
            .get(conn, blob_id)?
            .ok_or_else(|| ChatRelayError::BlobNotFound {
                blob_id: blob_id.to_string(),
            })
    }

    pub(crate) fn delete(
        &self,
        conn: &Connection,
        blob_id: &str,
        requester: &[u8; 32],
    ) -> ChatRelayResult<()> {
        match self.repository.delete(conn, blob_id, requester)? {
            EncryptedBlobDeleteOutcome::Deleted => Ok(()),
            EncryptedBlobDeleteOutcome::Unauthorized => Err(ChatRelayError::Unauthorized),
            EncryptedBlobDeleteOutcome::NotFound => Err(ChatRelayError::BlobNotFound {
                blob_id: blob_id.to_string(),
            }),
        }
    }
}

fn read_blob_usage(conn: &Connection) -> ChatRelayResult<(u64, u64)> {
    let counters = conn.query_row(
        "SELECT pending_blob_count, pending_blob_bytes
         FROM relay_storage_usage
         WHERE singleton = 1",
        [],
        |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
    )?;
    Ok((
        nonnegative_counter(counters.0, "pending_blob_count")?,
        nonnegative_counter(counters.1, "pending_blob_bytes")?,
    ))
}

fn nonnegative_counter(value: i64, field: &'static str) -> ChatRelayResult<u64> {
    u64::try_from(value).map_err(|_| ChatRelayError::CorruptStoredData { field })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct StubRepository {
        puts: AtomicUsize,
    }

    impl EncryptedBlobCustodyRepository for StubRepository {
        fn put(
            &self,
            _conn: &mut Connection,
            write: EncryptedBlobWrite<'_>,
            _policy: EncryptedBlobCustodyPolicy,
        ) -> ChatRelayResult<EncryptedBlobStoreOutcome> {
            self.puts.fetch_add(1, Ordering::Relaxed);
            Ok(EncryptedBlobStoreOutcome::Stored {
                blob_id: write.blob_id,
                size: write.data.len(),
            })
        }

        fn get(&self, _conn: &Connection, _blob_id: &str) -> ChatRelayResult<Option<Vec<u8>>> {
            Ok(None)
        }

        fn delete(
            &self,
            _conn: &Connection,
            _blob_id: &str,
            _requester: &[u8; 32],
        ) -> ChatRelayResult<EncryptedBlobDeleteOutcome> {
            Ok(EncryptedBlobDeleteOutcome::NotFound)
        }
    }

    fn test_policy(max_blob_size: usize) -> EncryptedBlobCustodyPolicy {
        EncryptedBlobCustodyPolicy {
            max_blob_size,
            max_blobs_per_receiver: 10,
            max_pending_blobs_total: 100,
            max_pending_blob_bytes_total: 1_000,
        }
    }

    #[test]
    fn composed_blob_custody_validates_before_repository_and_binds_identity() {
        let repository = StubRepository {
            puts: AtomicUsize::new(0),
        };
        let domain =
            EncryptedBlobCustodyDomain::with_repository(repository, test_policy(4), [0x41; 32]);
        let sender = [0x11; 32];
        let receiver = [0x12; 32];
        let hash = [0x13; 32];
        let mut conn = Connection::open_in_memory().expect("open test connection");

        assert!(matches!(
            domain.prepare_put(&sender, &receiver, &[0; 5], &hash, 1),
            Err(ChatRelayError::BlobTooLarge { size: 5, limit: 4 })
        ));
        assert_eq!(domain.repository.puts.load(Ordering::Relaxed), 0);

        let first_id = domain.compute_blob_id(&sender, &receiver, &hash);
        let second_id = domain.compute_blob_id(&sender, &[0x14; 32], &hash);
        assert_ne!(first_id, second_id);

        let write = domain
            .prepare_put(&sender, &receiver, &[0; 4], &hash, 1)
            .expect("prepare bounded blob");
        let outcome = domain.put(&mut conn, write).expect("put through stub");
        assert_eq!(outcome.blob_id(), first_id);
        assert_eq!(domain.repository.puts.load(Ordering::Relaxed), 1);
    }
}
