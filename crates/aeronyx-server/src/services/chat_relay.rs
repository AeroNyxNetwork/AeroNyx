// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay.rs
// ============================================================================
// Version: 1.3.0-Sovereign
//
// Modification Reason:
//   v1.3.0-Sovereign — Added WalletRouteCache field to ChatRelayService.
//   The route cache decouples wallet identity from session key, enabling
//   per-message signature-based authentication for all chat operations.
//   Also added dedup_cache as an Arc<Mutex<LruCache>> for the online-path
//   deduplication that will be used by the new handler in server.rs.
//   v1.3.1-Maintenance — Removed stale imports after the chat relay schema and
//   wallet-route integration stabilized. No database schema or API behavior changed.
//   v1.4.0-PeerRelayHealth — Added privacy-safe node-to-node relay health
//   counters for heartbeat/nodeboard diagnostics.
//   v1.5.0-GlobalStorageQuotas — Added transactionally maintained node-wide
//   message/blob usage and hard count/byte ceilings.
//
// Main Functionality:
//   - ChatRelayService: Central service managing all chat relay state
//   - Message store/pull/ack: SQLite-backed pending message queue
//   - Blob store/get/delete: SQLite-backed encrypted media cache
//   - TTL cleanup: Expires pending messages and blobs, queues notifications
//   - Expired notifications: Queued ChatExpired delivery for offline senders
//   - Online deduplication: LRU cache prevents duplicate delivery (online path)
//   - WalletRouteCache: In-memory wallet → session routing (v1.3.0-Sovereign)
//   - Peer relay health: aggregate outbound/inbound node-to-node relay status
//   - Durable queue quotas: per-receiver and node-wide count/byte ceilings
//
// Dependencies:
//   - aeronyx-core/src/protocol/chat.rs: ChatEnvelope, encode_envelope, decode_envelope
//   - aeronyx-server/src/config_chat_relay.rs: ChatRelayConfig
//   - crates/aeronyx-server/src/services/chat_relay/wallet_routes.rs: WalletRouteCache
//
// Main Logical Flow:
//   ChatRelayService::new():
//     1. Open/create SQLite database
//     2. Set WAL + NORMAL pragmas
//     3. init_schema() creates tables if missing
//     4. Initialise MessageDedup (online-path LRU)
//     5. Initialise WalletRouteCache (in-memory, empty on startup)
//
// ⚠️ Important Notes for Next Developer:
//   - wallet_routes is Arc<WalletRouteCache> so server.rs can hold a separate
//     Arc clone for the cleanup background task without borrowing ChatRelayService.
//   - All SQLite operations use parking_lot::Mutex<Connection>. Do NOT call
//     SQLite methods while holding another lock.
//   - ack_messages deletes only WHERE receiver = receiver_wallet.
//   - run_cleanup is synchronous — call from spawn_blocking or a sync task.
//   - node_secret is HKDF-derived from Ed25519 private key; stable across restarts.
//   - `relay_storage_usage` is rebuilt from canonical rows at startup, then
//     maintained only by SQLite triggers in the same write transaction.
//   - Logs must remain aggregate-only. Do not log message IDs, wallet prefixes,
//     blob IDs, sender/receiver keys, payload bytes, or endpoint/session IDs.
//
// Last Modified:
//   v1.5.0-GlobalStorageQuotas — Durable global quotas, enforced message size,
//     and route-safe logging
//   v1.4.0-PeerRelayHealth — Added node-to-node relay health status snapshot
//   v1.1.0-ChatRelay — Initial implementation
//   v1.3.0-Sovereign — Added wallet_routes: Arc<WalletRouteCache> field
//   v1.3.1-Maintenance — Removed stale imports; behavior unchanged
// ============================================================================

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use hmac::{Hmac, Mac};
use parking_lot::{Mutex, RwLock};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use tracing::{debug, info};

use aeronyx_core::protocol::chat::{decode_envelope, encode_envelope, ChatEnvelope};

use crate::config::ChatRelayConfig;
use crate::services::wallet_routes::WalletRouteCache;

// ============================================
// Type aliases
// ============================================

type HmacSha256 = Hmac<Sha256>;

// ============================================
// Peer relay health status
// ============================================

/// Privacy-safe node-to-node encrypted chat relay health snapshot.
///
/// This structure intentionally contains only aggregate counters and stable
/// reason buckets. It must not include message IDs, wallet IDs, client IPs,
/// destinations, DNS contents, URLs, chat plaintext, ciphertext, private keys,
/// voucher secrets, or per-user traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayPeerStatus {
    /// Whether chat relay is enabled in local config.
    pub enabled: bool,
    /// Total outbound peer relay attempts.
    pub outbound_attempted_total: u64,
    /// Total outbound peer relay requests accepted by peer nodes.
    pub outbound_accepted_total: u64,
    /// Total outbound peer relay requests that failed or were rejected.
    pub outbound_failed_total: u64,
    /// Total outbound fanout rounds observed.
    pub outbound_rounds: u64,
    /// Number of peers attempted in the last outbound fanout round.
    pub last_outbound_attempted: u64,
    /// Number of peers that accepted the last outbound fanout round.
    pub last_outbound_accepted: u64,
    /// Number of peers that failed the last outbound fanout round.
    pub last_outbound_failed: u64,
    /// Last outbound health bucket: healthy, degraded, failed, idle.
    pub last_outbound_status: Option<String>,
    /// Privacy-safe reason bucket for the last outbound relay failure.
    pub last_outbound_failure_reason: Option<String>,
    /// Consecutive outbound rounds with zero accepted peer relays.
    pub consecutive_outbound_failures: u64,
    /// Timestamp of the last outbound round with at least one accepted peer.
    pub last_outbound_success_at: Option<u64>,
    /// Timestamp of the last outbound fanout round.
    pub last_outbound_at: Option<u64>,
    /// Total inbound peer relay envelopes accepted for local processing.
    pub inbound_accepted_total: u64,
    /// Total inbound duplicate envelopes ignored idempotently.
    pub inbound_duplicate_total: u64,
    /// Total inbound envelopes delivered to online local sessions.
    pub inbound_delivered_online_total: u64,
    /// Total inbound envelopes stored in the local pending queue.
    pub inbound_stored_pending_total: u64,
    /// Total inbound peer relay requests rejected by local validation/storage.
    pub inbound_rejected_total: u64,
    /// Last inbound status bucket: accepted, duplicate, rejected.
    pub last_inbound_status: Option<String>,
    /// Privacy-safe reason bucket for the last inbound rejection.
    pub last_inbound_failure_reason: Option<String>,
    /// Timestamp of the last inbound peer relay request processed.
    pub last_inbound_at: Option<u64>,
}

impl ChatRelayPeerStatus {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            outbound_attempted_total: 0,
            outbound_accepted_total: 0,
            outbound_failed_total: 0,
            outbound_rounds: 0,
            last_outbound_attempted: 0,
            last_outbound_accepted: 0,
            last_outbound_failed: 0,
            last_outbound_status: None,
            last_outbound_failure_reason: None,
            consecutive_outbound_failures: 0,
            last_outbound_success_at: None,
            last_outbound_at: None,
            inbound_accepted_total: 0,
            inbound_duplicate_total: 0,
            inbound_delivered_online_total: 0,
            inbound_stored_pending_total: 0,
            inbound_rejected_total: 0,
            last_inbound_status: None,
            last_inbound_failure_reason: None,
            last_inbound_at: None,
        }
    }
}

// ============================================
// Error type
// ============================================

/// Errors produced by `ChatRelayService`.
#[derive(Debug, thiserror::Error)]
pub enum ChatRelayError {
    /// SQLite schema, query, transaction, or persistence failure.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Envelope or notification serialization failure.
    #[error("Serialization error: {0}")]
    Serialize(#[from] bincode::Error),

    /// Encrypted message ciphertext exceeds the configured item ceiling.
    #[error("Message too large: {size} bytes (limit {limit})")]
    MessageTooLarge {
        /// Incoming ciphertext bytes.
        size: usize,
        /// Configured ciphertext byte ceiling.
        limit: usize,
    },

    /// One receiver already holds the configured maximum pending messages.
    #[error("Mailbox full: receiver has {current} pending messages (limit {limit})")]
    MailboxFull {
        /// Current pending rows for the receiver.
        current: usize,
        /// Configured per-receiver row ceiling.
        limit: usize,
    },

    /// Node-wide pending message count is at capacity.
    #[error("Pending message queue full: {current} messages (limit {limit})")]
    PendingMessageQueueFull {
        /// Current active pending rows on the node.
        current: usize,
        /// Configured node-wide pending row ceiling.
        limit: usize,
    },

    /// Adding a message would exceed node-wide pending encoded bytes.
    #[error("Pending message byte quota exceeded: {current} + {incoming} bytes (limit {limit})")]
    PendingMessageBytesExceeded {
        /// Current encoded pending bytes.
        current: u64,
        /// Encoded bytes required by the incoming envelope.
        incoming: u64,
        /// Configured node-wide encoded byte ceiling.
        limit: u64,
    },

    /// One receiver already holds the configured maximum encrypted blobs.
    #[error("Blob quota exceeded: receiver has {current} pending blobs (limit {limit})")]
    BlobQuotaExceeded {
        /// Current blob rows for the receiver.
        current: usize,
        /// Configured per-receiver blob ceiling.
        limit: usize,
    },

    /// Node-wide encrypted blob count is at capacity.
    #[error("Pending blob store full: {current} blobs (limit {limit})")]
    PendingBlobStoreFull {
        /// Current retained blob rows on the node.
        current: usize,
        /// Configured node-wide blob row ceiling.
        limit: usize,
    },

    /// Adding an encrypted blob would exceed node-wide retained blob bytes.
    #[error("Pending blob byte quota exceeded: {current} + {incoming} bytes (limit {limit})")]
    PendingBlobBytesExceeded {
        /// Current retained encrypted blob bytes.
        current: u64,
        /// Incoming encrypted blob bytes.
        incoming: u64,
        /// Configured node-wide encrypted blob byte ceiling.
        limit: u64,
    },

    /// One encrypted blob exceeds the configured item ceiling.
    #[error("Blob too large: {size} bytes (limit {limit})")]
    BlobTooLarge {
        /// Incoming encrypted blob bytes.
        size: usize,
        /// Configured encrypted blob byte ceiling.
        limit: usize,
    },

    /// The opaque blob identifier does not resolve to a retained object.
    #[error("Blob not found: {blob_id}")]
    BlobNotFound {
        /// Opaque HMAC-derived identifier supplied by the caller.
        blob_id: String,
    },

    /// The authenticated caller is not allowed to mutate the object.
    #[error("Unauthorized: sender mismatch")]
    Unauthorized,
}

impl ChatRelayError {
    /// Returns a stable aggregate-only diagnostics bucket.
    #[must_use]
    pub const fn reason_bucket(&self) -> &'static str {
        match self {
            Self::Sqlite(_) => "sqlite_error",
            Self::Serialize(_) => "serialization_error",
            Self::MessageTooLarge { .. } => "message_too_large",
            Self::MailboxFull { .. } => "mailbox_full",
            Self::PendingMessageQueueFull { .. } => "pending_message_count_quota",
            Self::PendingMessageBytesExceeded { .. } => "pending_message_byte_quota",
            Self::BlobQuotaExceeded { .. } => "receiver_blob_quota",
            Self::PendingBlobStoreFull { .. } => "pending_blob_count_quota",
            Self::PendingBlobBytesExceeded { .. } => "pending_blob_byte_quota",
            Self::BlobTooLarge { .. } => "blob_too_large",
            Self::BlobNotFound { .. } => "blob_not_found",
            Self::Unauthorized => "unauthorized",
        }
    }

    /// Whether retrying without queue cleanup or operator action cannot help.
    #[must_use]
    pub const fn is_capacity_exhausted(&self) -> bool {
        matches!(
            self,
            Self::MailboxFull { .. }
                | Self::PendingMessageQueueFull { .. }
                | Self::PendingMessageBytesExceeded { .. }
                | Self::BlobQuotaExceeded { .. }
                | Self::PendingBlobStoreFull { .. }
                | Self::PendingBlobBytesExceeded { .. }
        )
    }
}

pub type ChatRelayResult<T> = Result<T, ChatRelayError>;

/// Aggregate durable relay usage with no user or routing identifiers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRelayStorageUsage {
    /// Active pending message rows.
    pub pending_messages: u64,
    /// Encoded bytes held by active pending messages.
    pub pending_message_bytes: u64,
    /// Pending encrypted blob rows.
    pub pending_blobs: u64,
    /// Encrypted blob bytes retained by the node.
    pub pending_blob_bytes: u64,
}

// ============================================
// Pending message row (returned from pull)
// ============================================

/// A pending offline message retrieved from the store.
#[derive(Debug)]
pub struct PendingMessage {
    /// Opaque client-generated message identifier used for ACK pagination.
    pub message_id: [u8; 16],
    /// Signed end-to-end encrypted envelope; relay code must not inspect its ciphertext.
    pub envelope: ChatEnvelope,
}

// ============================================
// Expired notification row
// ============================================

/// A queued `ChatExpired` notification for an offline sender.
#[derive(Debug)]
pub struct ExpiredNotification {
    /// Local notification row identifier.
    pub id: i64,
    /// Original sender public key used only for authenticated delivery lookup.
    pub sender: [u8; 32],
    /// Original receiver public key returned inside the encrypted client flow.
    pub receiver: [u8; 32],
    /// bincode-serialised `Vec<[u8; 16]>`
    pub message_ids_raw: Vec<u8>,
}

impl ExpiredNotification {
    /// Deserialise the stored message IDs.
    pub fn message_ids(&self) -> ChatRelayResult<Vec<[u8; 16]>> {
        Ok(bincode::deserialize(&self.message_ids_raw)?)
    }
}

// ============================================
// Minimal LRU for online-path deduplication
// ============================================

/// Fixed-capacity LRU cache for `message_id` deduplication on the online path.
struct MessageDedup {
    map: DashMap<[u8; 16], u64>,
    capacity: usize,
    seq: AtomicU64,
}

impl MessageDedup {
    fn new(capacity: usize) -> Self {
        Self {
            map: DashMap::with_capacity(capacity),
            capacity,
            seq: AtomicU64::new(0),
        }
    }

    /// Returns `true` if the message_id was already seen (duplicate).
    fn check_and_insert(&self, message_id: &[u8; 16]) -> bool {
        if self.map.contains_key(message_id) {
            return true;
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        self.map.insert(*message_id, seq);

        if self.map.len() > self.capacity {
            let oldest_key = self.map.iter().min_by_key(|e| *e.value()).map(|e| *e.key());
            if let Some(k) = oldest_key {
                self.map.remove(&k);
            }
        }
        false
    }
}

// ============================================
// ChatRelayService
// ============================================

/// Central service for zero-knowledge P2P chat relay.
///
/// ## v1.3.0-Sovereign additions
/// - `wallet_routes`: Arc-wrapped WalletRouteCache for wallet→session routing.
///   Exposed as a public Arc so `server.rs` can hold an independent clone for
///   the background cleanup task and for passing into message handlers.
pub struct ChatRelayService {
    config: ChatRelayConfig,
    conn: Mutex<Connection>,
    node_secret: [u8; 32],
    dedup: MessageDedup,
    peer_status: RwLock<ChatRelayPeerStatus>,
    /// In-memory wallet → session routing table.
    ///
    /// Arc so the cleanup task and each handler can hold independent references
    /// without borrowing the whole ChatRelayService.
    pub wallet_routes: Arc<WalletRouteCache>,
}

impl ChatRelayService {
    /// Creates a new `ChatRelayService`, opening (or creating) the SQLite database.
    pub fn new(config: ChatRelayConfig, node_secret: [u8; 32]) -> ChatRelayResult<Self> {
        if let Some(parent) = std::path::Path::new(&config.db_path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    rusqlite::Error::SqliteFailure(
                        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                        Some(format!("Cannot create dir {}: {}", parent.display(), e)),
                    )
                })?;
            }
        }

        let conn = Connection::open(&config.db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        let dedup_capacity = config.dedup_lru_capacity;
        let relay_enabled = config.enabled;
        let svc = Self {
            config,
            conn: Mutex::new(conn),
            node_secret,
            dedup: MessageDedup::new(dedup_capacity),
            peer_status: RwLock::new(ChatRelayPeerStatus::new(relay_enabled)),
            // v1.3.0-Sovereign: initialise empty route cache
            wallet_routes: Arc::new(WalletRouteCache::new()),
        };

        svc.init_schema()?;
        info!(
            "[CHAT_RELAY] Service initialised (db: {})",
            svc.config.db_path
        );
        Ok(svc)
    }

    // ============================================
    // Schema initialisation
    // ============================================

    fn init_schema(&self) -> ChatRelayResult<()> {
        let conn = self.conn.lock();
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS pending_messages (
                message_id   BLOB(16) PRIMARY KEY,
                sender       BLOB(32) NOT NULL,
                receiver     BLOB(32) NOT NULL,
                timestamp    INTEGER  NOT NULL,
                envelope     BLOB     NOT NULL,
                received_at  INTEGER  NOT NULL,
                status       INTEGER  NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_pm_receiver_status
                ON pending_messages(receiver, status);
            CREATE INDEX IF NOT EXISTS idx_pm_received_at
                ON pending_messages(received_at);

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
        // Reconcile from canonical rows at every startup. This makes upgrades
        // and restored databases deterministic even if an older process never
        // maintained the aggregate usage row.
        conn.execute(
            "INSERT OR REPLACE INTO relay_storage_usage (
                singleton,
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             )
             SELECT
                1,
                (SELECT COUNT(*) FROM pending_messages WHERE status = 0),
                (SELECT COALESCE(SUM(LENGTH(envelope)), 0)
                   FROM pending_messages WHERE status = 0),
                (SELECT COUNT(*) FROM pending_blobs),
                (SELECT COALESCE(SUM(size), 0) FROM pending_blobs)",
            [],
        )?;
        drop(conn);
        Ok(())
    }

    // ============================================
    // Blob ID derivation
    // ============================================

    pub fn compute_blob_id(
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

    // ============================================
    // Online-path deduplication
    // ============================================

    /// Returns `true` if this `message_id` has already been forwarded on the
    /// online path (duplicate detection for live sessions).
    pub fn is_online_duplicate(&self, message_id: &[u8; 16]) -> bool {
        self.dedup.check_and_insert(message_id)
    }

    // ============================================
    // Message store / pull / ack
    // ============================================

    fn read_storage_usage(conn: &Connection) -> ChatRelayResult<ChatRelayStorageUsage> {
        conn.query_row(
            "SELECT
                pending_message_count,
                pending_message_bytes,
                pending_blob_count,
                pending_blob_bytes
             FROM relay_storage_usage
             WHERE singleton = 1",
            [],
            |row| {
                Ok(ChatRelayStorageUsage {
                    pending_messages: nonnegative_sqlite_counter(row.get(0)?),
                    pending_message_bytes: nonnegative_sqlite_counter(row.get(1)?),
                    pending_blobs: nonnegative_sqlite_counter(row.get(2)?),
                    pending_blob_bytes: nonnegative_sqlite_counter(row.get(3)?),
                })
            },
        )
        .map_err(ChatRelayError::from)
    }

    /// Stores a pending offline message for a receiver that is not currently online.
    ///
    /// # Errors
    ///
    /// Returns an item-size or durable-capacity error before insertion, or a
    /// serialization/SQLite error if encoding or the atomic write fails.
    pub fn store_pending(&self, envelope: &ChatEnvelope) -> ChatRelayResult<()> {
        if envelope.ciphertext.len() > self.config.max_message_size {
            return Err(ChatRelayError::MessageTooLarge {
                size: envelope.ciphertext.len(),
                limit: self.config.max_message_size,
            });
        }

        let now = now_secs();
        let receiver = envelope.receiver;
        let envelope_bytes = encode_envelope(envelope)?;
        let incoming_bytes = envelope_bytes.len() as u64;

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Idempotence is checked before every quota. A retry of an already
        // durable message must succeed even while the queue is at capacity.
        let duplicate = tx
            .query_row(
                "SELECT 1 FROM pending_messages WHERE message_id = ?1",
                params![envelope.message_id.as_slice()],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if duplicate {
            tx.commit()?;
            return Ok(());
        }

        let usage = Self::read_storage_usage(&tx)?;
        if usage.pending_messages >= self.config.max_pending_messages_total as u64 {
            return Err(ChatRelayError::PendingMessageQueueFull {
                current: usize::try_from(usage.pending_messages).unwrap_or(usize::MAX),
                limit: self.config.max_pending_messages_total,
            });
        }
        if usage.pending_message_bytes.saturating_add(incoming_bytes)
            > self.config.max_pending_message_bytes_total
        {
            return Err(ChatRelayError::PendingMessageBytesExceeded {
                current: usage.pending_message_bytes,
                incoming: incoming_bytes,
                limit: self.config.max_pending_message_bytes_total,
            });
        }

        let count = tx.query_row(
            "SELECT COUNT(*) FROM pending_messages WHERE receiver = ? AND status = 0",
            params![receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )?;
        let count = usize::try_from(count.max(0)).unwrap_or(usize::MAX);

        if count >= self.config.max_pending_per_wallet {
            return Err(ChatRelayError::MailboxFull {
                current: count,
                limit: self.config.max_pending_per_wallet,
            });
        }

        tx.execute(
            "INSERT OR IGNORE INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![
                envelope.message_id.as_slice(),
                envelope.sender.as_slice(),
                envelope.receiver.as_slice(),
                envelope.timestamp as i64,
                envelope_bytes,
                now as i64,
            ],
        )?;
        tx.commit()?;
        drop(conn);

        debug!(
            encoded_bytes = incoming_bytes,
            "[CHAT_RELAY] Message stored pending"
        );
        Ok(())
    }

    /// Retrieves a page of pending messages for the given receiver wallet.
    pub fn pull_pending(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        limit: u32,
    ) -> ChatRelayResult<(Vec<PendingMessage>, bool)> {
        let effective_limit = (limit.min(100) as usize) + 1;

        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT message_id, envelope FROM pending_messages
             WHERE receiver = ?1
               AND status = 0
               AND timestamp > ?2
               AND message_id > ?3
             ORDER BY timestamp ASC, message_id ASC
             LIMIT ?4",
        )?;

        let rows: Vec<(Vec<u8>, Vec<u8>)> = stmt
            .query_map(
                params![
                    receiver.as_slice(),
                    after_timestamp as i64,
                    cursor.as_slice(),
                    effective_limit as i64,
                ],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);
        drop(conn);

        let has_more = rows.len() == effective_limit;
        let page = &rows[..rows.len().min(effective_limit - 1)];

        let mut messages = Vec::with_capacity(page.len());
        for (id_bytes, env_bytes) in page {
            let mut message_id = [0u8; 16];
            if id_bytes.len() == 16 {
                message_id.copy_from_slice(id_bytes);
            }
            let envelope = decode_envelope(env_bytes)?;
            messages.push(PendingMessage {
                message_id,
                envelope,
            });
        }

        Ok((messages, has_more))
    }

    /// Acknowledges delivery of a batch of messages, deleting them from the store.
    ///
    /// Only deletes rows where `receiver = receiver_wallet`.
    pub fn ack_messages(
        &self,
        message_ids: &[[u8; 16]],
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        if message_ids.is_empty() {
            return Ok(0);
        }

        let conn = self.conn.lock();
        let mut deleted = 0usize;

        for mid in message_ids {
            let n = conn.execute(
                "DELETE FROM pending_messages
                 WHERE message_id = ?1 AND receiver = ?2",
                params![mid.as_slice(), receiver_wallet.as_slice()],
            )?;
            deleted += n;
        }
        drop(conn);

        debug!(count = deleted, "[CHAT_RELAY] Messages ACKed and deleted");
        Ok(deleted)
    }

    // ============================================
    // Blob store / get / delete
    // ============================================

    pub fn put_blob(
        &self,
        sender: &[u8; 32],
        receiver: &[u8; 32],
        data: &[u8],
        file_hash: &[u8; 32],
    ) -> ChatRelayResult<String> {
        if data.len() > self.config.max_blob_size {
            return Err(ChatRelayError::BlobTooLarge {
                size: data.len(),
                limit: self.config.max_blob_size,
            });
        }

        let blob_id = self.compute_blob_id(sender, receiver, file_hash);
        let incoming_bytes = data.len() as u64;
        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Return the stable content-derived ID before quota checks when the
        // encrypted object is already present. Retries remain idempotent even
        // while the blob store is full.
        let duplicate = tx
            .query_row(
                "SELECT 1 FROM pending_blobs WHERE blob_id = ?1",
                params![&blob_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        if duplicate {
            tx.commit()?;
            return Ok(blob_id);
        }

        let usage = Self::read_storage_usage(&tx)?;
        if usage.pending_blobs >= self.config.max_pending_blobs_total as u64 {
            return Err(ChatRelayError::PendingBlobStoreFull {
                current: usize::try_from(usage.pending_blobs).unwrap_or(usize::MAX),
                limit: self.config.max_pending_blobs_total,
            });
        }
        if usage.pending_blob_bytes.saturating_add(incoming_bytes)
            > self.config.max_pending_blob_bytes_total
        {
            return Err(ChatRelayError::PendingBlobBytesExceeded {
                current: usage.pending_blob_bytes,
                incoming: incoming_bytes,
                limit: self.config.max_pending_blob_bytes_total,
            });
        }

        let count = tx.query_row(
            "SELECT COUNT(*) FROM pending_blobs WHERE receiver = ?",
            params![receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )?;
        let count = usize::try_from(count.max(0)).unwrap_or(usize::MAX);

        if count >= self.config.max_blobs_per_receiver {
            return Err(ChatRelayError::BlobQuotaExceeded {
                current: count,
                limit: self.config.max_blobs_per_receiver,
            });
        }

        let now = now_secs();

        tx.execute(
            "INSERT OR IGNORE INTO pending_blobs
             (blob_id, sender, receiver, data, size, received_at, downloaded)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![
                &blob_id,
                sender.as_slice(),
                receiver.as_slice(),
                data,
                data.len() as i64,
                now as i64,
            ],
        )?;
        tx.commit()?;
        drop(conn);

        info!(size = data.len(), "[CHAT_RELAY] Encrypted blob stored");
        Ok(blob_id)
    }

    pub fn get_blob(&self, blob_id: &str) -> ChatRelayResult<Vec<u8>> {
        let conn = self.conn.lock();

        let data: Option<Vec<u8>> = conn
            .query_row(
                "SELECT data FROM pending_blobs WHERE blob_id = ?",
                params![blob_id],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;

        match data {
            None => {
                drop(conn);
                Err(ChatRelayError::BlobNotFound {
                    blob_id: blob_id.to_string(),
                })
            }
            Some(bytes) => {
                let _ = conn.execute(
                    "UPDATE pending_blobs SET downloaded = 1 WHERE blob_id = ?",
                    params![blob_id],
                );
                drop(conn);
                debug!(size = bytes.len(), "[CHAT_RELAY] Encrypted blob retrieved");
                Ok(bytes)
            }
        }
    }

    pub fn delete_blob(&self, blob_id: &str, requester: &[u8; 32]) -> ChatRelayResult<()> {
        let conn = self.conn.lock();

        let deleted = conn.execute(
            "DELETE FROM pending_blobs WHERE blob_id = ?1 AND sender = ?2",
            params![blob_id, requester.as_slice()],
        )?;

        if deleted == 1 {
            drop(conn);
            info!("[CHAT_RELAY] Encrypted blob deleted by authorized sender");
            return Ok(());
        }

        let exists: bool = conn
            .query_row(
                "SELECT 1 FROM pending_blobs WHERE blob_id = ?",
                params![blob_id],
                |_| Ok(true),
            )
            .optional()?
            .unwrap_or(false);
        drop(conn);

        if exists {
            Err(ChatRelayError::Unauthorized)
        } else {
            Err(ChatRelayError::BlobNotFound {
                blob_id: blob_id.to_string(),
            })
        }
    }

    // ============================================
    // Expired notifications
    // ============================================

    pub fn get_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<Vec<ExpiredNotification>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, sender, receiver, message_ids
             FROM expired_notifications
             WHERE sender = ? AND pushed = 0
             ORDER BY created_at ASC",
        )?;

        let rows = stmt
            .query_map(params![sender.as_slice()], |row| {
                Ok(ExpiredNotification {
                    id: row.get::<_, i64>(0)?,
                    sender: {
                        let b: Vec<u8> = row.get(1)?;
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b);
                        arr
                    },
                    receiver: {
                        let b: Vec<u8> = row.get(2)?;
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b);
                        arr
                    },
                    message_ids_raw: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);
        drop(conn);

        Ok(rows)
    }

    pub fn mark_notifications_pushed(&self, ids: &[i64]) -> ChatRelayResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let conn = self.conn.lock();
        for id in ids {
            conn.execute(
                "UPDATE expired_notifications SET pushed = 1 WHERE id = ?",
                params![id],
            )?;
        }
        drop(conn);
        Ok(())
    }

    // ============================================
    // TTL cleanup
    // ============================================

    /// Runs one TTL cleanup cycle (synchronous — call from spawn_blocking).
    ///
    /// All mutations run inside a single SQLite IMMEDIATE transaction.
    /// Returns `(expired_messages, expired_blobs)`.
    pub fn run_cleanup(&self) -> ChatRelayResult<(usize, usize)> {
        let now = now_secs() as i64;
        let ttl = self.config.offline_ttl_secs as i64;
        let notif_ttl = self.config.expired_notification_ttl_secs as i64;
        let cutoff = now - ttl;
        let notif_cutoff = now - notif_ttl;

        let conn = self.conn.lock();
        conn.execute_batch("BEGIN IMMEDIATE")?;

        let result: ChatRelayResult<(usize, usize)> = (|| {
            let mut stmt = conn.prepare(
                "SELECT message_id, sender, receiver FROM pending_messages
                 WHERE status = 0 AND received_at < ?",
            )?;

            #[derive(Debug)]
            struct ExpiredRow {
                message_id: [u8; 16],
                sender: [u8; 32],
                receiver: [u8; 32],
            }

            let expired_rows: Vec<ExpiredRow> = stmt
                .query_map(params![cutoff], |row| {
                    let mid_b: Vec<u8> = row.get(0)?;
                    let sender_b: Vec<u8> = row.get(1)?;
                    let receiver_b: Vec<u8> = row.get(2)?;
                    Ok((mid_b, sender_b, receiver_b))
                })?
                .filter_map(|r| r.ok())
                .filter_map(|(mid_b, sender_b, receiver_b)| {
                    if mid_b.len() != 16 || sender_b.len() != 32 || receiver_b.len() != 32 {
                        return None;
                    }
                    let mut mid = [0u8; 16];
                    let mut sender = [0u8; 32];
                    let mut receiver = [0u8; 32];
                    mid.copy_from_slice(&mid_b);
                    sender.copy_from_slice(&sender_b);
                    receiver.copy_from_slice(&receiver_b);
                    Some(ExpiredRow {
                        message_id: mid,
                        sender,
                        receiver,
                    })
                })
                .collect();

            let expired_message_count = expired_rows.len();

            use std::collections::HashMap;
            let mut by_sender: HashMap<[u8; 32], HashMap<[u8; 32], Vec<[u8; 16]>>> = HashMap::new();
            for row in &expired_rows {
                by_sender
                    .entry(row.sender)
                    .or_default()
                    .entry(row.receiver)
                    .or_default()
                    .push(row.message_id);
            }

            for (sender, by_receiver) in &by_sender {
                for (receiver, ids) in by_receiver {
                    let ids_bytes = bincode::serialize(ids)?;
                    conn.execute(
                        "INSERT INTO expired_notifications
                         (sender, receiver, message_ids, created_at, pushed)
                         VALUES (?1, ?2, ?3, ?4, 0)",
                        params![sender.as_slice(), receiver.as_slice(), ids_bytes, now],
                    )?;
                }
            }

            for row in &expired_rows {
                conn.execute(
                    "UPDATE pending_messages SET status = 2 WHERE message_id = ?",
                    params![row.message_id.as_slice()],
                )?;
            }

            conn.execute("DELETE FROM pending_messages WHERE status IN (1, 2)", [])?;

            let expired_blobs = conn.execute(
                "DELETE FROM pending_blobs WHERE received_at < ?",
                params![cutoff],
            )?;

            conn.execute(
                "DELETE FROM expired_notifications WHERE pushed = 1 OR created_at < ?",
                params![notif_cutoff],
            )?;

            Ok((expired_message_count, expired_blobs))
        })();

        match &result {
            Ok(_) => {
                conn.execute_batch("COMMIT")?;
            }
            Err(_) => {
                let _ = conn.execute_batch("ROLLBACK");
            }
        }
        drop(conn);

        if let Ok((expired_message_count, expired_blobs)) = &result {
            if *expired_message_count > 0 || *expired_blobs > 0 {
                info!(
                    expired_messages = expired_message_count,
                    expired_blobs = expired_blobs,
                    "[CHAT_RELAY] Cleanup complete"
                );
            } else {
                debug!("[CHAT_RELAY] Cleanup: nothing to expire");
            }
        }

        result
    }

    // ============================================
    // Peer relay health
    // ============================================

    /// Records an outbound node-to-node encrypted chat relay fanout round.
    ///
    /// The failure reason must be a stable bucket such as
    /// `peer_relay_request_timeout` or `peer_relay_http_503`; do not pass peer
    /// URLs, message IDs, wallet IDs, client IPs, or payload-derived data.
    pub fn record_peer_relay_outbound(
        &self,
        now: u64,
        attempted: usize,
        accepted: usize,
        failure_reason: Option<String>,
    ) {
        let failed = attempted.saturating_sub(accepted);
        let status_bucket = if attempted == 0 && failure_reason.is_some() {
            "failed"
        } else if attempted == 0 {
            "idle"
        } else if accepted == attempted {
            "healthy"
        } else if accepted > 0 {
            "degraded"
        } else {
            "failed"
        };
        let failure_reason = if failed > 0 || (attempted == 0 && status_bucket == "failed") {
            Some(failure_reason.unwrap_or_else(|| "unknown".to_string()))
        } else {
            None
        };

        let mut status = self.peer_status.write();
        status.outbound_attempted_total = status
            .outbound_attempted_total
            .saturating_add(attempted as u64);
        status.outbound_accepted_total = status
            .outbound_accepted_total
            .saturating_add(accepted as u64);
        status.outbound_failed_total = status.outbound_failed_total.saturating_add(failed as u64);
        status.outbound_rounds = status.outbound_rounds.saturating_add(1);
        status.last_outbound_attempted = attempted as u64;
        status.last_outbound_accepted = accepted as u64;
        status.last_outbound_failed = failed as u64;
        status.last_outbound_status = Some(status_bucket.to_string());
        status.last_outbound_failure_reason = failure_reason;
        status.last_outbound_at = Some(now);

        if accepted > 0 {
            status.consecutive_outbound_failures = 0;
            status.last_outbound_success_at = Some(now);
        } else if attempted > 0 || status.last_outbound_failure_reason.is_some() {
            status.consecutive_outbound_failures =
                status.consecutive_outbound_failures.saturating_add(1);
        }
    }

    /// Records an accepted inbound peer relay request.
    pub fn record_peer_relay_inbound_accepted(
        &self,
        now: u64,
        duplicate: bool,
        delivered_online: usize,
        stored_pending: bool,
    ) {
        let mut status = self.peer_status.write();
        status.inbound_accepted_total = status.inbound_accepted_total.saturating_add(1);
        if duplicate {
            status.inbound_duplicate_total = status.inbound_duplicate_total.saturating_add(1);
        }
        status.inbound_delivered_online_total = status
            .inbound_delivered_online_total
            .saturating_add(delivered_online as u64);
        if stored_pending {
            status.inbound_stored_pending_total =
                status.inbound_stored_pending_total.saturating_add(1);
        }
        status.last_inbound_status = Some(if duplicate { "duplicate" } else { "accepted" }.into());
        status.last_inbound_failure_reason = None;
        status.last_inbound_at = Some(now);
    }

    /// Records a rejected inbound peer relay request with a stable reason bucket.
    pub fn record_peer_relay_inbound_rejected(&self, now: u64, reason: impl Into<String>) {
        let mut status = self.peer_status.write();
        status.inbound_rejected_total = status.inbound_rejected_total.saturating_add(1);
        status.last_inbound_status = Some("rejected".to_string());
        status.last_inbound_failure_reason = Some(reason.into());
        status.last_inbound_at = Some(now);
    }

    // ============================================
    // Accessors
    // ============================================

    #[must_use]
    pub fn config(&self) -> &ChatRelayConfig {
        &self.config
    }

    /// Returns a privacy-safe node-to-node relay health snapshot.
    #[must_use]
    pub fn peer_status(&self) -> ChatRelayPeerStatus {
        self.peer_status.read().clone()
    }

    /// Returns aggregate durable queue usage maintained by `SQLite` triggers.
    ///
    /// The result contains no message, wallet, sender, receiver, route, or
    /// payload identifiers and is safe for operator-capacity telemetry.
    ///
    /// # Errors
    ///
    /// Returns a SQLite error if the reconciled singleton usage row cannot be
    /// read. Callers must treat that as unavailable telemetry, not zero usage.
    pub fn storage_usage(&self) -> ChatRelayResult<ChatRelayStorageUsage> {
        let conn = self.conn.lock();
        Self::read_storage_usage(&conn)
    }
}

fn nonnegative_sqlite_counter(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

// ============================================
// Helpers
// ============================================

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Derives a stable 32-byte node secret from the node's Ed25519 private key.
pub fn derive_node_secret(ed25519_sk_bytes: &[u8; 32]) -> [u8; 32] {
    use hkdf::Hkdf;
    let hk = Hkdf::<Sha256>::new(Some(b"aeronyx-chat-relay-v1"), ed25519_sk_bytes);
    let mut okm = [0u8; 32];
    hk.expand(b"", &mut okm)
        .expect("HKDF expand with 32-byte output always succeeds");
    okm
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_common::types::SessionId;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::ChatContentType;
    use sha2::{Digest, Sha256};
    use std::net::SocketAddr;

    fn test_config() -> ChatRelayConfig {
        ChatRelayConfig {
            enabled: true,
            db_path: ":memory:".to_string(),
            offline_ttl_secs: 259_200,
            max_pending_per_wallet: 5,
            max_pending_messages_total: 100,
            max_pending_message_bytes_total: 1024 * 1024,
            max_message_size: 65_536,
            max_blob_size: 1_024,
            max_blobs_per_receiver: 3,
            max_pending_blobs_total: 10,
            max_pending_blob_bytes_total: 10 * 1024,
            cleanup_interval_secs: 60,
            dedup_lru_capacity: 10,
            expired_notification_ttl_secs: 604_800,
        }
    }

    fn make_service() -> ChatRelayService {
        make_service_with_config(test_config())
    }

    fn make_service_with_config(config: ChatRelayConfig) -> ChatRelayService {
        let secret = derive_node_secret(&[0x42u8; 32]);
        ChatRelayService::new(config, secret).expect("init")
    }

    #[test]
    fn test_peer_relay_outbound_health_tracks_failure_and_recovery() {
        let svc = make_service();

        svc.record_peer_relay_outbound(
            1_800_000_010,
            2,
            1,
            Some("peer_relay_request_timeout".to_string()),
        );
        let status = svc.peer_status();
        assert_eq!(status.last_outbound_status.as_deref(), Some("degraded"));
        assert_eq!(status.last_outbound_attempted, 2);
        assert_eq!(status.last_outbound_accepted, 1);
        assert_eq!(status.last_outbound_failed, 1);
        assert_eq!(status.consecutive_outbound_failures, 0);
        assert_eq!(status.last_outbound_success_at, Some(1_800_000_010));

        svc.record_peer_relay_outbound(
            1_800_000_020,
            1,
            0,
            Some("peer_relay_http_503".to_string()),
        );
        let status = svc.peer_status();
        assert_eq!(status.last_outbound_status.as_deref(), Some("failed"));
        assert_eq!(
            status.last_outbound_failure_reason.as_deref(),
            Some("peer_relay_http_503")
        );
        assert_eq!(status.consecutive_outbound_failures, 1);

        svc.record_peer_relay_outbound(1_800_000_030, 1, 1, None);
        let status = svc.peer_status();
        assert_eq!(status.last_outbound_status.as_deref(), Some("healthy"));
        assert_eq!(status.last_outbound_failure_reason, None);
        assert_eq!(status.consecutive_outbound_failures, 0);
        assert_eq!(status.last_outbound_success_at, Some(1_800_000_030));
    }

    #[test]
    fn test_peer_relay_inbound_health_tracks_accept_and_reject() {
        let svc = make_service();

        svc.record_peer_relay_inbound_accepted(1_800_000_010, false, 0, true);
        let status = svc.peer_status();
        assert_eq!(status.inbound_accepted_total, 1);
        assert_eq!(status.inbound_stored_pending_total, 1);
        assert_eq!(status.last_inbound_status.as_deref(), Some("accepted"));
        assert_eq!(status.last_inbound_failure_reason, None);

        svc.record_peer_relay_inbound_accepted(1_800_000_020, true, 0, false);
        let status = svc.peer_status();
        assert_eq!(status.inbound_accepted_total, 2);
        assert_eq!(status.inbound_duplicate_total, 1);
        assert_eq!(status.last_inbound_status.as_deref(), Some("duplicate"));

        svc.record_peer_relay_inbound_rejected(1_800_000_030, "invalid_signature");
        let status = svc.peer_status();
        assert_eq!(status.inbound_rejected_total, 1);
        assert_eq!(status.last_inbound_status.as_deref(), Some("rejected"));
        assert_eq!(
            status.last_inbound_failure_reason.as_deref(),
            Some("invalid_signature")
        );
    }

    fn make_envelope(kp: &IdentityKeyPair, receiver: [u8; 32]) -> ChatEnvelope {
        let mut env = ChatEnvelope {
            message_id: rand::random(),
            sender: kp.public_key_bytes(),
            receiver,
            timestamp: now_secs(),
            ciphertext: b"encrypted".to_vec(),
            nonce: [0x02; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        let data = env.sign_data();
        env.signature = kp.sign(&data);
        env
    }

    fn make_session() -> SessionId {
        SessionId::from_bytes(&rand::random::<[u8; 16]>())
            .expect("random bytes form valid SessionId")
    }

    fn make_addr(port: u16) -> SocketAddr {
        format!("127.0.0.1:{}", port).parse().unwrap()
    }

    // ── Schema init ──────────────────────────────────────────────────────

    #[test]
    fn test_service_init() {
        let svc = make_service();
        let (m, b) = svc.run_cleanup().expect("cleanup");
        assert_eq!(m, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn chat_relay_logs_stay_free_of_routing_identifiers() {
        let source = include_str!("chat_relay.rs");
        let message_log = concat!("id = %hex::", "encode(envelope.message_id)");
        let receiver_log = concat!("receiver = %hex::", "encode");
        let sender_log = concat!("sender = %hex::", "encode");
        let blob_log = concat!("blob_id", " = %");

        for forbidden in [message_log, receiver_log, sender_log, blob_log] {
            assert!(
                !source.contains(forbidden),
                "relay logs must not expose stable routing identifiers"
            );
        }
    }

    // ── v1.3.0: wallet_routes field accessible ───────────────────────────

    #[test]
    fn test_wallet_routes_field_accessible() {
        let svc = make_service();
        let wallet = [0xAAu8; 32];
        let sid = make_session();
        let addr = make_addr(9000);

        // announce via the public field
        svc.wallet_routes.announce(&wallet, sid.clone(), addr);

        let results = svc.wallet_routes.lookup(&wallet);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, sid);
    }

    #[test]
    fn test_wallet_routes_arc_clone_shares_state() {
        let svc = make_service();
        let routes_clone = Arc::clone(&svc.wallet_routes);

        let wallet = [0xBBu8; 32];
        let sid = make_session();

        // announce via original
        svc.wallet_routes
            .announce(&wallet, sid.clone(), make_addr(9001));

        // lookup via clone — must see the same entry
        let results = routes_clone.lookup(&wallet);
        assert_eq!(
            results.len(),
            1,
            "Arc clone must share the same underlying cache"
        );
    }

    // ── store → pull → ack (preserved) ───────────────────────────────────

    #[test]
    fn test_store_pull_ack_roundtrip() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);
        let mid = env.message_id;

        svc.store_pending(&env).expect("store");
        let usage = svc.storage_usage().expect("usage after store");
        assert_eq!(usage.pending_messages, 1);
        assert!(usage.pending_message_bytes > 0);

        let (msgs, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
        assert!(!has_more);
        assert_eq!(msgs[0].message_id, mid);

        let deleted = svc.ack_messages(&[mid], &receiver).expect("ack");
        assert_eq!(deleted, 1);
        let usage = svc.storage_usage().expect("usage after ack");
        assert_eq!(usage.pending_messages, 0);
        assert_eq!(usage.pending_message_bytes, 0);

        let (msgs2, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull2");
        assert!(msgs2.is_empty());
    }

    #[test]
    fn test_store_duplicate_ignored() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);

        svc.store_pending(&env).expect("first store");
        svc.store_pending(&env)
            .expect("duplicate store — should not error");

        let (msgs, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn test_store_enforces_configured_ciphertext_size_limit() {
        let mut config = test_config();
        config.max_message_size = 4;
        let svc = make_service_with_config(config);
        let kp = IdentityKeyPair::generate();
        let envelope = make_envelope(&kp, [0x10; 32]);

        assert!(matches!(
            svc.store_pending(&envelope),
            Err(ChatRelayError::MessageTooLarge { size: 9, limit: 4 })
        ));
        assert_eq!(
            svc.storage_usage().unwrap(),
            ChatRelayStorageUsage::default()
        );
    }

    #[test]
    fn test_global_message_count_quota_preserves_duplicate_idempotence() {
        let mut config = test_config();
        config.max_pending_messages_total = 1;
        let svc = make_service_with_config(config);
        let kp = IdentityKeyPair::generate();
        let first = make_envelope(&kp, [0x11; 32]);

        svc.store_pending(&first).expect("first store");
        svc.store_pending(&first)
            .expect("duplicate remains successful at global capacity");

        let second = make_envelope(&kp, [0x22; 32]);
        assert!(matches!(
            svc.store_pending(&second),
            Err(ChatRelayError::PendingMessageQueueFull { .. })
        ));
        assert_eq!(svc.storage_usage().unwrap().pending_messages, 1);
    }

    #[test]
    fn test_global_message_byte_quota_spans_distinct_receivers() {
        let kp = IdentityKeyPair::generate();
        let first = make_envelope(&kp, [0x31; 32]);
        let encoded_bytes = encode_envelope(&first).unwrap().len() as u64;
        let mut config = test_config();
        config.max_pending_message_bytes_total = encoded_bytes;
        let svc = make_service_with_config(config);

        svc.store_pending(&first).expect("first store");
        let second = make_envelope(&kp, [0x32; 32]);
        assert!(matches!(
            svc.store_pending(&second),
            Err(ChatRelayError::PendingMessageBytesExceeded { .. })
        ));
    }

    #[test]
    fn test_storage_usage_reconciles_from_canonical_rows_on_restart() {
        let path = std::env::temp_dir().join(format!(
            "aeronyx-chat-relay-usage-{}.db",
            rand::random::<u64>()
        ));
        let mut config = test_config();
        config.db_path = path.to_string_lossy().into_owned();
        let kp = IdentityKeyPair::generate();
        let envelope = make_envelope(&kp, [0x61; 32]);

        {
            let svc = make_service_with_config(config.clone());
            svc.store_pending(&envelope).expect("store before restart");
        }
        {
            let conn = Connection::open(&path).expect("open usage database");
            conn.execute(
                "UPDATE relay_storage_usage
                 SET pending_message_count = 0, pending_message_bytes = 0
                 WHERE singleton = 1",
                [],
            )
            .expect("tamper derived usage row");
        }

        let restarted = make_service_with_config(config);
        let usage = restarted.storage_usage().expect("reconciled usage");
        assert_eq!(usage.pending_messages, 1);
        assert_eq!(
            usage.pending_message_bytes,
            encode_envelope(&envelope).unwrap().len() as u64
        );
        drop(restarted);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}-wal", path.display()));
        let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    }

    #[test]
    fn test_mailbox_full_rejected() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];

        for _ in 0..5 {
            let env = make_envelope(&kp, receiver);
            svc.store_pending(&env).expect("store");
        }

        let env6 = make_envelope(&kp, receiver);
        let result = svc.store_pending(&env6);
        assert!(matches!(result, Err(ChatRelayError::MailboxFull { .. })));
    }

    #[test]
    fn test_ack_wrong_receiver_cannot_delete() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);
        let mid = env.message_id;

        svc.store_pending(&env).expect("store");

        let wrong_receiver = [0xCCu8; 32];
        let deleted = svc.ack_messages(&[mid], &wrong_receiver).expect("ack");
        assert_eq!(deleted, 0, "Wrong receiver must not delete messages");

        let (msgs, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
    }

    // ── Pagination (preserved) ───────────────────────────────────────────

    #[test]
    fn test_pull_pagination() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];

        for _ in 0..5 {
            let env = make_envelope(&kp, receiver);
            svc.store_pending(&env).expect("store");
        }

        let (page1, has_more1) = svc.pull_pending(&receiver, 0, &[0u8; 16], 3).expect("p1");
        assert_eq!(page1.len(), 3);
        assert!(has_more1);

        let cursor = page1.last().unwrap().message_id;
        let (page2, has_more2) = svc.pull_pending(&receiver, 0, &cursor, 3).expect("p2");
        assert_eq!(page2.len(), 2);
        assert!(!has_more2);
    }

    // ── Blob (preserved) ─────────────────────────────────────────────────

    #[test]
    fn test_blob_put_get_delete() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = b"encrypted_image_bytes";
        let file_hash: [u8; 32] = Sha256::digest(data).into();

        let blob_id = svc
            .put_blob(&sender, &receiver, data, &file_hash)
            .expect("put");
        assert_eq!(blob_id.len(), 32);
        let usage = svc.storage_usage().expect("usage after blob put");
        assert_eq!(usage.pending_blobs, 1);
        assert_eq!(usage.pending_blob_bytes, data.len() as u64);

        let fetched = svc.get_blob(&blob_id).expect("get");
        assert_eq!(fetched, data);

        svc.delete_blob(&blob_id, &sender).expect("delete");
        let usage = svc.storage_usage().expect("usage after blob delete");
        assert_eq!(usage.pending_blobs, 0);
        assert_eq!(usage.pending_blob_bytes, 0);
        assert!(matches!(
            svc.get_blob(&blob_id),
            Err(ChatRelayError::BlobNotFound { .. })
        ));
    }

    #[test]
    fn test_blob_too_large_rejected() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = vec![0u8; 2048];
        let file_hash: [u8; 32] = Sha256::digest(&data).into();

        let result = svc.put_blob(&sender, &receiver, &data, &file_hash);
        assert!(matches!(result, Err(ChatRelayError::BlobTooLarge { .. })));
    }

    #[test]
    fn test_global_blob_count_quota_preserves_duplicate_idempotence() {
        let mut config = test_config();
        config.max_pending_blobs_total = 1;
        let svc = make_service_with_config(config);
        let sender = [0x41; 32];
        let first_receiver = [0x42; 32];
        let first_data = b"first encrypted blob";
        let first_hash: [u8; 32] = Sha256::digest(first_data).into();

        let first_id = svc
            .put_blob(&sender, &first_receiver, first_data, &first_hash)
            .expect("first put");
        let duplicate_id = svc
            .put_blob(&sender, &first_receiver, first_data, &first_hash)
            .expect("duplicate remains successful at global capacity");
        assert_eq!(duplicate_id, first_id);

        let second_data = b"second encrypted blob";
        let second_hash: [u8; 32] = Sha256::digest(second_data).into();
        assert!(matches!(
            svc.put_blob(&sender, &[0x43; 32], second_data, &second_hash),
            Err(ChatRelayError::PendingBlobStoreFull { .. })
        ));
        assert_eq!(svc.storage_usage().unwrap().pending_blobs, 1);
    }

    #[test]
    fn test_global_blob_byte_quota_spans_distinct_receivers() {
        let data = b"bounded encrypted blob";
        let mut config = test_config();
        config.max_pending_blob_bytes_total = data.len() as u64;
        let svc = make_service_with_config(config);
        let sender = [0x51; 32];
        let first_hash: [u8; 32] = Sha256::digest(data).into();

        svc.put_blob(&sender, &[0x52; 32], data, &first_hash)
            .expect("first put");
        let second_hash: [u8; 32] = Sha256::digest(b"different hash").into();
        assert!(matches!(
            svc.put_blob(&sender, &[0x53; 32], data, &second_hash),
            Err(ChatRelayError::PendingBlobBytesExceeded { .. })
        ));
    }

    #[test]
    fn test_blob_delete_wrong_sender_rejected() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = b"file";
        let file_hash: [u8; 32] = Sha256::digest(data).into();

        let blob_id = svc
            .put_blob(&sender, &receiver, data, &file_hash)
            .expect("put");
        let wrong = [0xCCu8; 32];
        assert!(matches!(
            svc.delete_blob(&blob_id, &wrong),
            Err(ChatRelayError::Unauthorized)
        ));
    }

    // ── Online dedup (preserved) ─────────────────────────────────────────

    #[test]
    fn test_online_dedup() {
        let svc = make_service();
        let id = [0x01u8; 16];
        assert!(!svc.is_online_duplicate(&id));
        assert!(svc.is_online_duplicate(&id));
    }

    // ── TTL cleanup (preserved) ──────────────────────────────────────────

    #[test]
    fn test_cleanup_does_not_touch_fresh_messages() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);

        svc.store_pending(&env).expect("store");
        let (expired, blobs) = svc.run_cleanup().expect("cleanup");
        assert_eq!(expired, 0);
        assert_eq!(blobs, 0);

        let (msgs, _) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull");
        assert_eq!(msgs.len(), 1);
    }

    // ── node_secret derivation (preserved) ───────────────────────────────

    #[test]
    fn test_derive_node_secret_deterministic() {
        let sk = [0x42u8; 32];
        let s1 = derive_node_secret(&sk);
        let s2 = derive_node_secret(&sk);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_derive_node_secret_different_keys() {
        let s1 = derive_node_secret(&[0x01u8; 32]);
        let s2 = derive_node_secret(&[0x02u8; 32]);
        assert_ne!(s1, s2);
    }
}
