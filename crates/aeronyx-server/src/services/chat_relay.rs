// ============================================================================
// File: crates/aeronyx-server/src/services/chat_relay.rs
// ============================================================================
// Version: 1.9.0-DurableQuarantine
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
//   v1.6.0-OfflineControlReliability — Made ACK and notification batches
//   bounded and atomic, surfaced corrupt durable rows instead of skipping or
//   panicking, and split expiry notifications into transport-safe chunks.
//   v1.7.0-MaintenanceRuntime — Added aggregate cleanup execution evidence
//   and aligned durable pull ordering with the existing message-id cursor.
//   v1.8.0-BoundedMaintenance — Split retention cleanup into bounded SQLite
//   transactions and exposed deferred-backlog evidence.
//   v1.9.0-DurableQuarantine — Added privacy-minimised corrupt-row tombstones,
//   poison-row isolation, complete durable-envelope consistency checks, and
//   atomic concurrent online deduplication.
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
//   - Maintenance telemetry: aggregate TTL cleanup, batch, and backlog evidence
//   - Durable quarantine: bounded de-identified evidence for corrupt relay rows
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
//   - Offline control batches are protocol-bounded. Do not remove their limits.
//     Malformed rows must be atomically replaced by de-identified quarantine
//     evidence; never silently skip them or copy raw routing metadata.
//   - Pending-message pages are ordered by message_id because the v1 wire
//     cursor contains only message_id. Chronological display belongs client-side.
//   - Retention cleanup is batch-bounded. Do not replace it with an unbounded
//     SELECT/DELETE or hold the SQLite connection across multiple batches.
//   - Quarantine events must remain de-identified. Never persist message IDs,
//     sender/receiver keys, ciphertext, endpoints, or raw durable rows there.
//
// Last Modified:
//   v1.9.0-DurableQuarantine — Poison-row isolation, private tombstones,
//     durable metadata/signature validation, and atomic live-path deduplication
//   v1.8.0-BoundedMaintenance — Bounded transactions and backlog observability
//   v1.7.0-MaintenanceRuntime — Runtime cleanup evidence and cursor-safe paging
//   v1.6.0-OfflineControlReliability — Atomic bounded ACK/expiry control flow
//   v1.5.0-GlobalStorageQuotas — Durable global quotas, enforced message size,
//     and route-safe logging
//   v1.4.0-PeerRelayHealth — Added node-to-node relay health status snapshot
//   v1.1.0-ChatRelay — Initial implementation
//   v1.3.0-Sovereign — Added wallet_routes: Arc<WalletRouteCache> field
//   v1.3.1-Maintenance — Removed stale imports; behavior unchanged
// ============================================================================

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::{mapref::entry::Entry, DashMap};
use hmac::{Hmac, Mac};
use parking_lot::{Mutex, RwLock};
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::Sha256;

use tracing::{debug, info, warn};

use aeronyx_core::protocol::chat::{decode_envelope, encode_envelope, ChatEnvelope};

use crate::config::ChatRelayConfig;
use crate::services::wallet_routes::WalletRouteCache;

// ============================================
// Type aliases
// ============================================

type HmacSha256 = Hmac<Sha256>;

/// Maximum IDs accepted in one authenticated `ChatAck` frame.
pub const MAX_CHAT_ACK_MESSAGE_IDS: usize = 100;
/// Maximum IDs encoded into one `ChatExpired` frame.
const MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION: usize = 32;
/// Maximum notification rows offered during one authenticated pull.
const MAX_EXPIRED_NOTIFICATIONS_PER_PULL: usize = 16;
/// Defensive ceiling for one persisted bincode notification payload.
const MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES: usize = 1024;
/// Maximum expired message rows processed by one `SQLite` transaction.
const CLEANUP_MESSAGE_BATCH_SIZE: usize = 1024;
/// Maximum expired encrypted blobs deleted by one `SQLite` transaction.
const CLEANUP_BLOB_BATCH_SIZE: usize = 128;
/// Maximum delivered/stale notification rows deleted by one transaction.
const CLEANUP_NOTIFICATION_BATCH_SIZE: usize = 1024;
/// Maximum privacy-minimised quarantine events removed by one transaction.
const CLEANUP_QUARANTINE_EVENT_BATCH_SIZE: usize = 1024;
/// Maximum cleanup transactions executed by one scheduled maintenance run.
const CLEANUP_MAX_BATCHES_PER_RUN: usize = 8;
/// Maximum retained de-identified corruption events.
const MAX_QUARANTINE_EVENTS: usize = 4096;
const QUARANTINE_SOURCE_PENDING_MESSAGE: &str = "pending_message";
const QUARANTINE_SOURCE_EXPIRED_NOTIFICATION: &str = "expired_notification";

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

    /// One authenticated ACK frame exceeds the protocol processing ceiling.
    #[error("ACK batch too large: {size} message IDs (limit {limit})")]
    AckBatchTooLarge {
        /// Number of IDs supplied by the authenticated caller.
        size: usize,
        /// Protocol-defined processing ceiling.
        limit: usize,
    },

    /// Durable relay data violates a fixed-size or bounded storage invariant.
    #[error("Corrupt stored relay data: {field}")]
    CorruptStoredData {
        /// Stable aggregate-only field bucket; never include stored values.
        field: &'static str,
    },

    /// A client-supplied timestamp cannot be represented by SQLite INTEGER.
    #[error("Message timestamp is outside the supported range")]
    TimestampOutOfRange,

    /// An existing durable row uses the same ID for a different signed envelope.
    #[error("Message ID conflicts with an existing durable envelope")]
    MessageIdConflict,

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
            Self::AckBatchTooLarge { .. } => "ack_batch_too_large",
            Self::CorruptStoredData { .. } => "corrupt_stored_data",
            Self::TimestampOutOfRange => "timestamp_out_of_range",
            Self::MessageIdConflict => "message_id_conflict",
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

/// Aggregate TTL maintenance evidence safe for heartbeat and node health APIs.
///
/// This snapshot intentionally excludes message IDs, wallet keys, routes,
/// endpoints, payloads, and per-user counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatRelayMaintenanceStatus {
    /// Total cleanup attempts, including failed transactions.
    pub cleanup_runs_total: u64,
    /// Cleanup attempts that returned an error.
    pub cleanup_failures_total: u64,
    /// Successfully committed bounded cleanup transactions.
    pub cleanup_batches_total: u64,
    /// Runs that reached their transaction budget with work still pending.
    pub cleanup_backlog_deferred_total: u64,
    /// Pending message rows removed by successfully committed batches.
    pub expired_messages_total: u64,
    /// Encrypted blob rows removed by successfully committed batches.
    pub expired_blobs_total: u64,
    /// Delivered or stale expiry-notification rows removed by committed batches.
    pub expired_notifications_removed_total: u64,
    /// Corrupt pending-message rows atomically isolated from active delivery.
    pub quarantined_pending_messages_total: u64,
    /// Corrupt expiry-notification rows atomically isolated from delivery.
    pub quarantined_expired_notifications_total: u64,
    /// De-identified quarantine event rows removed by bounded retention.
    pub quarantine_events_removed_total: u64,
    /// Current durable de-identified quarantine event rows.
    pub quarantine_events_retained: u64,
    /// Unix timestamp of the most recent poison-row isolation.
    pub last_quarantine_at: Option<u64>,
    /// Unix timestamp of the most recent cleanup attempt.
    pub last_cleanup_at: Option<u64>,
    /// Stable state bucket: `succeeded` or `failed`.
    pub last_cleanup_status: Option<String>,
    /// Stable aggregate failure bucket from [`ChatRelayError::reason_bucket`].
    pub last_cleanup_failure_reason: Option<String>,
    /// Number of successfully committed transactions in the latest run.
    pub last_cleanup_batches: u64,
    /// Whether the latest run deferred remaining work to the next timer tick.
    pub last_cleanup_backlog_deferred: bool,
    /// Corrupt pending-message rows isolated by the latest cleanup run.
    pub last_cleanup_quarantined_pending_messages: u64,
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

#[derive(Debug)]
struct ExpiredMessageRow {
    message_id: [u8; 16],
    sender: [u8; 32],
    receiver: [u8; 32],
}

type ExpiredMessagesBySender = HashMap<[u8; 32], HashMap<[u8; 32], Vec<[u8; 16]>>>;
#[derive(Debug)]
struct StoredExpiredNotificationRow {
    id: i64,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    message_ids_raw: Vec<u8>,
}

#[derive(Debug)]
struct StoredPendingMessageRow {
    rowid: i64,
    message_id: Vec<u8>,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    timestamp: i64,
    envelope: Vec<u8>,
}

#[derive(Debug)]
struct StoredExpiredMessageRow {
    rowid: i64,
    message_id: Vec<u8>,
    sender: Vec<u8>,
    receiver: Vec<u8>,
    timestamp: i64,
    envelope: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct CorruptDurableRow {
    row_key: i64,
    source_kind: &'static str,
    reason: &'static str,
    encoded_bytes: u64,
}

#[derive(Debug, Default)]
struct ValidatedExpiredMessageBatch {
    valid_rows: Vec<ExpiredMessageRow>,
    corrupt_rows: Vec<CorruptDurableRow>,
    selected_rowids: Vec<i64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CleanupBatchOutcome {
    expired_messages: usize,
    expired_blobs: usize,
    removed_notifications: usize,
    quarantined_pending_messages: usize,
    removed_quarantine_events: usize,
    retained_quarantine_events: usize,
    has_more: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct CleanupRunSummary {
    expired_messages: usize,
    expired_blobs: usize,
    removed_notifications: usize,
    quarantined_pending_messages: usize,
    removed_quarantine_events: usize,
    retained_quarantine_events: usize,
    successful_batches: usize,
    backlog_deferred: bool,
}

impl CleanupRunSummary {
    fn absorb(&mut self, batch: CleanupBatchOutcome) {
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
        self.retained_quarantine_events = batch.retained_quarantine_events;
        self.successful_batches = self.successful_batches.saturating_add(1);
    }

    fn removed_anything(self) -> bool {
        self.expired_messages > 0
            || self.expired_blobs > 0
            || self.removed_notifications > 0
            || self.quarantined_pending_messages > 0
            || self.removed_quarantine_events > 0
    }
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
        if self.message_ids_raw.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_payload_size",
            });
        }
        let message_ids: Vec<[u8; 16]> = bincode::deserialize(&self.message_ids_raw)?;
        if message_ids.is_empty() || message_ids.len() > MAX_EXPIRED_MESSAGE_IDS_PER_NOTIFICATION {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_notification_message_count",
            });
        }
        Ok(message_ids)
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
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        match self.map.entry(*message_id) {
            Entry::Occupied(_) => return true,
            Entry::Vacant(entry) => {
                entry.insert(seq);
            }
        }

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
    maintenance_status: RwLock<ChatRelayMaintenanceStatus>,
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
        // A short bounded wait absorbs transient locks from an operator backup
        // or diagnostic reader without allowing relay requests to hang forever.
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        let dedup_capacity = config.dedup_lru_capacity;
        let relay_enabled = config.enabled;
        let svc = Self {
            config,
            conn: Mutex::new(conn),
            node_secret,
            dedup: MessageDedup::new(dedup_capacity),
            peer_status: RwLock::new(ChatRelayPeerStatus::new(relay_enabled)),
            maintenance_status: RwLock::new(ChatRelayMaintenanceStatus::default()),
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
        Self::init_pending_message_schema(&conn)?;
        Self::init_blob_and_notification_schema(&conn)?;
        Self::init_quarantine_schema(&conn)?;
        Self::init_usage_schema(&conn)?;
        Self::reconcile_storage_usage(&conn)?;
        let retained_quarantine_events =
            conn.query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
                row.get::<_, i64>(0)
            })?;
        drop(conn);
        self.maintenance_status.write().quarantine_events_retained =
            nonnegative_sqlite_counter(retained_quarantine_events);
        Ok(())
    }

    fn init_pending_message_schema(conn: &Connection) -> ChatRelayResult<()> {
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

    fn init_blob_and_notification_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
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

    fn init_quarantine_schema(conn: &Connection) -> ChatRelayResult<()> {
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

    fn init_usage_schema(conn: &Connection) -> ChatRelayResult<()> {
        conn.execute_batch(
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

    fn reconcile_storage_usage(conn: &Connection) -> ChatRelayResult<()> {
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

    fn quarantine_retention_cutoff(&self, now: i64) -> i64 {
        let ttl = i64::try_from(self.config.expired_notification_ttl_secs).unwrap_or(i64::MAX);
        now.saturating_sub(ttl)
    }

    fn insert_quarantine_events(
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

        let mut stmt = tx.prepare(
            "INSERT INTO relay_quarantine_events
             (source_kind, reason, row_count, encoded_bytes, quarantined_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for ((source_kind, reason), (row_count, encoded_bytes)) in aggregates {
            stmt.execute(params![
                source_kind,
                reason,
                i64::try_from(row_count).unwrap_or(i64::MAX),
                i64::try_from(encoded_bytes).unwrap_or(i64::MAX),
                now,
            ])?;
        }
        Ok(())
    }

    fn delete_pending_rows_by_rowid(
        tx: &Transaction<'_>,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut stmt = tx.prepare("DELETE FROM pending_messages WHERE rowid = ?1")?;
        for row in rows {
            if stmt.execute(params![row.row_key])? != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "pending_message_quarantine_delete_count",
                });
            }
        }
        Ok(())
    }

    fn delete_notification_rows_by_id(
        tx: &Transaction<'_>,
        rows: &[CorruptDurableRow],
    ) -> ChatRelayResult<()> {
        let mut stmt = tx.prepare("DELETE FROM expired_notifications WHERE id = ?1")?;
        for row in rows {
            if stmt.execute(params![row.row_key])? != 1 {
                return Err(ChatRelayError::CorruptStoredData {
                    field: "expired_notification_quarantine_delete_count",
                });
            }
        }
        Ok(())
    }

    fn trim_quarantine_events(
        tx: &Transaction<'_>,
        retention_cutoff: i64,
    ) -> ChatRelayResult<usize> {
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
        Ok(removed_stale.saturating_add(removed_overflow))
    }

    fn quarantine_event_count(tx: &Transaction<'_>) -> ChatRelayResult<usize> {
        let count = tx.query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
            row.get::<_, i64>(0)
        })?;
        Ok(usize::try_from(count.max(0)).unwrap_or(usize::MAX))
    }

    fn record_pull_quarantine(
        &self,
        now: u64,
        pending_messages: usize,
        expired_notifications: usize,
        removed_events: usize,
        retained_events: usize,
    ) {
        let mut status = self.maintenance_status.write();
        status.quarantined_pending_messages_total = status
            .quarantined_pending_messages_total
            .saturating_add(u64::try_from(pending_messages).unwrap_or(u64::MAX));
        status.quarantined_expired_notifications_total = status
            .quarantined_expired_notifications_total
            .saturating_add(u64::try_from(expired_notifications).unwrap_or(u64::MAX));
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(u64::try_from(removed_events).unwrap_or(u64::MAX));
        status.quarantine_events_retained = u64::try_from(retained_events).unwrap_or(u64::MAX);
        status.last_quarantine_at = Some(now);
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
        let received_at = i64::try_from(now).unwrap_or(i64::MAX);
        let envelope_timestamp =
            i64::try_from(envelope.timestamp).map_err(|_| ChatRelayError::TimestampOutOfRange)?;
        let receiver = envelope.receiver;
        let envelope_bytes = encode_envelope(envelope)?;
        let incoming_bytes = u64::try_from(envelope_bytes.len()).unwrap_or(u64::MAX);

        let mut conn = self.conn.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        // Idempotence is checked before every quota. A retry of an already
        // durable message must succeed even while the queue is at capacity.
        let existing_envelope = tx
            .query_row(
                "SELECT envelope FROM pending_messages WHERE message_id = ?1",
                params![envelope.message_id.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        if let Some(existing_envelope) = existing_envelope {
            if existing_envelope == envelope_bytes {
                tx.commit()?;
                return Ok(());
            }
            return Err(ChatRelayError::MessageIdConflict);
        }

        let usage = Self::read_storage_usage(&tx)?;
        if usage.pending_messages
            >= u64::try_from(self.config.max_pending_messages_total).unwrap_or(u64::MAX)
        {
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
                envelope_timestamp,
                envelope_bytes,
                received_at,
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

    fn validate_pending_message_row(
        row: StoredPendingMessageRow,
        expected_receiver: &[u8; 32],
    ) -> Result<PendingMessage, CorruptDurableRow> {
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

    /// Retrieves a page of pending messages for the given receiver wallet.
    ///
    /// The v1 wire cursor contains only `message_id`, so rows must be ordered
    /// by that same key. Ordering by timestamp first can permanently skip a
    /// later row whose random ID sorts below the previous page's cursor.
    ///
    /// # Errors
    ///
    /// Corrupt rows are atomically replaced by de-identified quarantine events
    /// so one poison row cannot permanently block a receiver's mailbox.
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending(
        &self,
        receiver: &[u8; 32],
        after_timestamp: u64,
        cursor: &[u8; 16],
        limit: u32,
    ) -> ChatRelayResult<(Vec<PendingMessage>, bool)> {
        let page_limit = usize::try_from(limit.clamp(1, 100)).unwrap_or(100);
        let effective_limit = page_limit.saturating_add(1);
        let query_after_timestamp = i64::try_from(after_timestamp).unwrap_or(i64::MAX);
        let query_limit = i64::try_from(effective_limit).unwrap_or(i64::MAX);

        let mut conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT rowid, message_id, sender, receiver, timestamp, envelope
             FROM pending_messages
             WHERE receiver = ?1
               AND status = 0
               AND timestamp > ?2
               AND message_id > ?3
             ORDER BY message_id ASC
             LIMIT ?4",
        )?;

        let rows: Vec<StoredPendingMessageRow> = stmt
            .query_map(
                params![
                    receiver.as_slice(),
                    query_after_timestamp,
                    cursor.as_slice(),
                    query_limit,
                ],
                |row| {
                    Ok(StoredPendingMessageRow {
                        rowid: row.get(0)?,
                        message_id: row.get(1)?,
                        sender: row.get(2)?,
                        receiver: row.get(3)?,
                        timestamp: row.get(4)?,
                        envelope: row.get(5)?,
                    })
                },
            )?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);
        let raw_has_more = rows.len() == effective_limit;
        let mut messages = Vec::with_capacity(rows.len().min(page_limit));
        let mut corrupt_rows = Vec::new();
        for row in rows {
            match Self::validate_pending_message_row(row, receiver) {
                Ok(message) => messages.push(message),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }

        if corrupt_rows.is_empty() {
            drop(conn);
        } else {
            let quarantine_now = now_secs();
            let quarantine_now_i64 = i64::try_from(quarantine_now).unwrap_or(i64::MAX);
            let retention_cutoff = self.quarantine_retention_cutoff(quarantine_now_i64);
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            Self::insert_quarantine_events(&tx, quarantine_now_i64, &corrupt_rows)?;
            Self::delete_pending_rows_by_rowid(&tx, &corrupt_rows)?;
            let removed_events = Self::trim_quarantine_events(&tx, retention_cutoff)?;
            let retained_events = Self::quarantine_event_count(&tx)?;
            tx.commit()?;
            drop(conn);

            self.record_pull_quarantine(
                quarantine_now,
                corrupt_rows.len(),
                0,
                removed_events,
                retained_events,
            );
            warn!(
                quarantined_pending_messages = corrupt_rows.len(),
                "[CHAT_RELAY] Corrupt pending rows isolated during pull"
            );
        }

        let has_more = raw_has_more || messages.len() > page_limit;
        messages.truncate(page_limit);
        Ok((messages, has_more))
    }

    /// Acknowledges delivery of a batch of messages, deleting them from the store.
    ///
    /// Only deletes rows where `receiver = receiver_wallet`.
    ///
    /// # Errors
    ///
    /// Returns an oversized-batch or `SQLite` error. The transaction is atomic.
    pub fn ack_messages(
        &self,
        message_ids: &[[u8; 16]],
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        if message_ids.is_empty() {
            return Ok(0);
        }
        if message_ids.len() > MAX_CHAT_ACK_MESSAGE_IDS {
            return Err(ChatRelayError::AckBatchTooLarge {
                size: message_ids.len(),
                limit: MAX_CHAT_ACK_MESSAGE_IDS,
            });
        }

        let unique_ids: HashSet<[u8; 16]> = message_ids.iter().copied().collect();
        let deleted =
            Self::ack_messages_transaction(&mut self.conn.lock(), &unique_ids, receiver_wallet)?;

        debug!(count = deleted, "[CHAT_RELAY] Messages ACKed and deleted");
        Ok(deleted)
    }

    fn ack_messages_transaction(
        conn: &mut Connection,
        unique_ids: &HashSet<[u8; 16]>,
        receiver_wallet: &[u8; 32],
    ) -> ChatRelayResult<usize> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut deleted = 0usize;

        for mid in unique_ids {
            let n = tx.execute(
                "DELETE FROM pending_messages
                 WHERE message_id = ?1 AND receiver = ?2",
                params![mid.as_slice(), receiver_wallet.as_slice()],
            )?;
            deleted += n;
        }
        tx.commit()?;
        Ok(deleted)
    }

    // ============================================
    // Blob store / get / delete
    // ============================================

    /// Stores one opaque encrypted blob under node-wide and receiver quotas.
    ///
    /// # Errors
    ///
    /// Returns an item-size, capacity, serialization, or `SQLite` error.
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
        let incoming_bytes = u64::try_from(data.len()).unwrap_or(u64::MAX);
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
        if usage.pending_blobs
            >= u64::try_from(self.config.max_pending_blobs_total).unwrap_or(u64::MAX)
        {
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
        let received_at = i64::try_from(now).unwrap_or(i64::MAX);
        let stored_size = i64::try_from(data.len()).unwrap_or(i64::MAX);

        tx.execute(
            "INSERT OR IGNORE INTO pending_blobs
             (blob_id, sender, receiver, data, size, received_at, downloaded)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![
                &blob_id,
                sender.as_slice(),
                receiver.as_slice(),
                data,
                stored_size,
                received_at,
            ],
        )?;
        tx.commit()?;
        drop(conn);

        info!(size = data.len(), "[CHAT_RELAY] Encrypted blob stored");
        Ok(blob_id)
    }

    /// Retrieves an opaque encrypted blob by its HMAC-derived identifier.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error or [`ChatRelayError::BlobNotFound`].
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

    /// Deletes an encrypted blob when requested by its original sender.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite`, not-found, or authorization error.
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

    /// Retrieves one bounded page of expiry notifications for a sender.
    ///
    /// The extra row is used only to compute `has_more`; it is never returned.
    /// Invalid durable rows are atomically replaced by de-identified quarantine
    /// evidence so one poison row cannot permanently block sender control flow.
    ///
    /// # Errors
    ///
    /// Returns a storage error if reading or quarantine persistence fails.
    pub fn pull_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<(Vec<ExpiredNotification>, bool)> {
        let effective_limit = MAX_EXPIRED_NOTIFICATIONS_PER_PULL + 1;
        let query_limit = i64::try_from(effective_limit).unwrap_or(i64::MAX);
        let mut conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, sender, receiver, message_ids
             FROM expired_notifications
             WHERE sender = ?1 AND pushed = 0
             ORDER BY created_at ASC, id ASC
             LIMIT ?2",
        )?;

        let rows: Vec<StoredExpiredNotificationRow> = stmt
            .query_map(params![sender.as_slice(), query_limit], |row| {
                Ok(StoredExpiredNotificationRow {
                    id: row.get(0)?,
                    sender: row.get(1)?,
                    receiver: row.get(2)?,
                    message_ids_raw: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);

        let raw_has_more = rows.len() == effective_limit;
        let mut notifications =
            Vec::with_capacity(rows.len().min(MAX_EXPIRED_NOTIFICATIONS_PER_PULL));
        let mut corrupt_rows = Vec::new();
        for row in rows {
            match Self::validate_expired_notification_row(row, sender) {
                Ok(notification) => notifications.push(notification),
                Err(corrupt) => corrupt_rows.push(corrupt),
            }
        }

        if corrupt_rows.is_empty() {
            drop(conn);
        } else {
            let quarantine_now = now_secs();
            let quarantine_now_i64 = i64::try_from(quarantine_now).unwrap_or(i64::MAX);
            let retention_cutoff = self.quarantine_retention_cutoff(quarantine_now_i64);
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            Self::insert_quarantine_events(&tx, quarantine_now_i64, &corrupt_rows)?;
            Self::delete_notification_rows_by_id(&tx, &corrupt_rows)?;
            let removed_events = Self::trim_quarantine_events(&tx, retention_cutoff)?;
            let retained_events = Self::quarantine_event_count(&tx)?;
            tx.commit()?;
            drop(conn);

            self.record_pull_quarantine(
                quarantine_now,
                0,
                corrupt_rows.len(),
                removed_events,
                retained_events,
            );
            warn!(
                quarantined_expired_notifications = corrupt_rows.len(),
                "[CHAT_RELAY] Corrupt expiry notifications isolated during pull"
            );
        }

        let has_more = raw_has_more || notifications.len() > MAX_EXPIRED_NOTIFICATIONS_PER_PULL;
        notifications.truncate(MAX_EXPIRED_NOTIFICATIONS_PER_PULL);
        Ok((notifications, has_more))
    }

    /// Compatibility wrapper for callers that do not consume pagination yet.
    ///
    /// New runtime code should use [`Self::pull_pending_notifications`] and
    /// propagate its `has_more` flag.
    ///
    /// # Errors
    ///
    /// Returns a storage, decoding, or durable-data integrity error.
    pub fn get_pending_notifications(
        &self,
        sender: &[u8; 32],
    ) -> ChatRelayResult<Vec<ExpiredNotification>> {
        self.pull_pending_notifications(sender)
            .map(|(notifications, _)| notifications)
    }

    /// Atomically marks a successfully written notification page as pushed.
    ///
    /// # Errors
    ///
    /// Returns a `SQLite` error and rolls back the whole page on failure.
    pub fn mark_notifications_pushed(&self, ids: &[i64]) -> ChatRelayResult<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let unique_ids: HashSet<i64> = ids.iter().copied().collect();
        Self::mark_notifications_pushed_transaction(&mut self.conn.lock(), &unique_ids)
    }

    fn mark_notifications_pushed_transaction(
        conn: &mut Connection,
        unique_ids: &HashSet<i64>,
    ) -> ChatRelayResult<()> {
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        for id in unique_ids {
            tx.execute(
                "UPDATE expired_notifications SET pushed = 1 WHERE id = ?",
                params![id],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    // ============================================
    // TTL cleanup
    // ============================================

    /// Runs one TTL cleanup cycle (synchronous — call from `spawn_blocking`).
    ///
    /// Mutations run in a bounded sequence of `SQLite` IMMEDIATE transactions.
    /// Each committed batch releases the connection before the next begins.
    /// Returns `(expired_messages, expired_blobs)`.
    ///
    /// # Errors
    ///
    /// Returns a storage, serialization, or durable-data integrity error. A
    /// failed batch is rolled back and counted in maintenance evidence. Earlier
    /// committed batches remain durable and are included in aggregate counters.
    pub fn run_cleanup(&self) -> ChatRelayResult<(usize, usize)> {
        self.run_cleanup_with_batch_budget(CLEANUP_MAX_BATCHES_PER_RUN)
    }

    fn run_cleanup_with_batch_budget(&self, max_batches: usize) -> ChatRelayResult<(usize, usize)> {
        let now = now_secs();
        let cleanup_now = i64::try_from(now).unwrap_or(i64::MAX);
        let (summary, failure) = self.run_cleanup_at(cleanup_now, max_batches.max(1));

        self.record_cleanup_run(now, summary, failure.as_ref());

        let Some(error) = failure else {
            return Ok((summary.expired_messages, summary.expired_blobs));
        };
        Err(error)
    }

    fn record_cleanup_run(
        &self,
        now: u64,
        summary: CleanupRunSummary,
        failure: Option<&ChatRelayError>,
    ) {
        let mut status = self.maintenance_status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_batches_total = status
            .cleanup_batches_total
            .saturating_add(u64::try_from(summary.successful_batches).unwrap_or(u64::MAX));
        if summary.backlog_deferred {
            status.cleanup_backlog_deferred_total =
                status.cleanup_backlog_deferred_total.saturating_add(1);
        }
        status.expired_messages_total = status
            .expired_messages_total
            .saturating_add(u64::try_from(summary.expired_messages).unwrap_or(u64::MAX));
        status.expired_blobs_total = status
            .expired_blobs_total
            .saturating_add(u64::try_from(summary.expired_blobs).unwrap_or(u64::MAX));
        status.expired_notifications_removed_total = status
            .expired_notifications_removed_total
            .saturating_add(u64::try_from(summary.removed_notifications).unwrap_or(u64::MAX));
        status.quarantined_pending_messages_total =
            status.quarantined_pending_messages_total.saturating_add(
                u64::try_from(summary.quarantined_pending_messages).unwrap_or(u64::MAX),
            );
        status.quarantine_events_removed_total = status
            .quarantine_events_removed_total
            .saturating_add(u64::try_from(summary.removed_quarantine_events).unwrap_or(u64::MAX));
        if summary.successful_batches > 0 {
            status.quarantine_events_retained =
                u64::try_from(summary.retained_quarantine_events).unwrap_or(u64::MAX);
        }
        if summary.quarantined_pending_messages > 0 {
            status.last_quarantine_at = Some(now);
        }
        status.last_cleanup_at = Some(now);
        status.last_cleanup_batches = u64::try_from(summary.successful_batches).unwrap_or(u64::MAX);
        status.last_cleanup_backlog_deferred = summary.backlog_deferred;
        status.last_cleanup_quarantined_pending_messages =
            u64::try_from(summary.quarantined_pending_messages).unwrap_or(u64::MAX);
        match failure {
            None => {
                status.last_cleanup_status = Some("succeeded".to_string());
                status.last_cleanup_failure_reason = None;
            }
            Some(error) => {
                status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
                status.last_cleanup_status = Some("failed".to_string());
                status.last_cleanup_failure_reason = Some(error.reason_bucket().to_string());
            }
        }
    }

    fn run_cleanup_at(
        &self,
        now: i64,
        max_batches: usize,
    ) -> (CleanupRunSummary, Option<ChatRelayError>) {
        // Configuration validation rejects values above i64::MAX. These
        // fallbacks keep direct service construction fail-closed as well:
        // an out-of-range TTL retains data instead of expiring fresh rows.
        let ttl = i64::try_from(self.config.offline_ttl_secs).unwrap_or(i64::MAX);
        let notif_ttl =
            i64::try_from(self.config.expired_notification_ttl_secs).unwrap_or(i64::MAX);
        let cutoff = now.saturating_sub(ttl);
        let notif_cutoff = now.saturating_sub(notif_ttl);

        let mut summary = CleanupRunSummary::default();
        let mut failure = None;
        for batch_index in 0..max_batches {
            let batch_result = {
                let mut conn = self.conn.lock();
                Self::run_cleanup_transaction(&mut conn, now, cutoff, notif_cutoff)
            };
            match batch_result {
                Ok(batch) => {
                    let has_more = batch.has_more;
                    summary.absorb(batch);
                    if !has_more {
                        break;
                    }
                    if batch_index + 1 == max_batches {
                        summary.backlog_deferred = true;
                    }
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        if summary.removed_anything() || summary.backlog_deferred {
            info!(
                expired_messages = summary.expired_messages,
                expired_blobs = summary.expired_blobs,
                removed_notifications = summary.removed_notifications,
                quarantined_pending_messages = summary.quarantined_pending_messages,
                removed_quarantine_events = summary.removed_quarantine_events,
                retained_quarantine_events = summary.retained_quarantine_events,
                committed_batches = summary.successful_batches,
                backlog_deferred = summary.backlog_deferred,
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Bounded cleanup run complete"
            );
        } else {
            debug!(
                cleanup_failed = failure.is_some(),
                "[CHAT_RELAY] Cleanup: nothing to expire"
            );
        }
        if summary.quarantined_pending_messages > 0 {
            warn!(
                quarantined_pending_messages = summary.quarantined_pending_messages,
                "[CHAT_RELAY] Corrupt pending rows isolated during cleanup"
            );
        }

        (summary, failure)
    }

    fn run_cleanup_transaction(
        conn: &mut Connection,
        now: i64,
        cutoff: i64,
        notif_cutoff: i64,
    ) -> ChatRelayResult<CleanupBatchOutcome> {
        let message_limit = i64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(i64::MAX);
        let blob_limit = i64::try_from(CLEANUP_BLOB_BATCH_SIZE).unwrap_or(i64::MAX);
        let notification_limit = i64::try_from(CLEANUP_NOTIFICATION_BATCH_SIZE).unwrap_or(i64::MAX);
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let transaction_result: ChatRelayResult<CleanupBatchOutcome> = (|| {
            let expired_batch = Self::load_expired_message_batch(&tx, cutoff, message_limit)?;
            let expired_message_count = expired_batch.valid_rows.len();
            let quarantined_pending_messages = expired_batch.corrupt_rows.len();
            Self::queue_expiry_notifications(&tx, now, &expired_batch.valid_rows)?;
            Self::insert_quarantine_events(&tx, now, &expired_batch.corrupt_rows)?;
            Self::delete_expired_message_batch(&tx, &expired_batch.selected_rowids)?;
            let expired_blobs = Self::delete_expired_blob_batch(&tx, cutoff, blob_limit)?;
            let removed_notifications =
                Self::delete_stale_notification_batch(&tx, notif_cutoff, notification_limit)?;
            let removed_quarantine_events = Self::trim_quarantine_events(&tx, notif_cutoff)?;
            let retained_quarantine_events = Self::quarantine_event_count(&tx)?;
            let has_more = Self::cleanup_backlog_exists(&tx, cutoff, notif_cutoff)?;

            Ok(CleanupBatchOutcome {
                expired_messages: expired_message_count,
                expired_blobs,
                removed_notifications,
                quarantined_pending_messages,
                removed_quarantine_events,
                retained_quarantine_events,
                has_more,
            })
        })();

        match transaction_result {
            Ok(counts) => {
                tx.commit()?;
                Ok(counts)
            }
            Err(error) => Err(error),
        }
    }

    fn load_expired_message_batch(
        tx: &Transaction<'_>,
        cutoff: i64,
        limit: i64,
    ) -> ChatRelayResult<ValidatedExpiredMessageBatch> {
        let mut stmt = tx.prepare(
            "SELECT rowid, message_id, sender, receiver, timestamp, envelope
             FROM pending_messages
             WHERE status = 0 AND received_at < ?1
             ORDER BY received_at ASC, message_id ASC
             LIMIT ?2",
        )?;
        let stored_rows: Vec<StoredExpiredMessageRow> = stmt
            .query_map(params![cutoff, limit], |row| {
                Ok(StoredExpiredMessageRow {
                    rowid: row.get(0)?,
                    message_id: row.get(1)?,
                    sender: row.get(2)?,
                    receiver: row.get(3)?,
                    timestamp: row.get(4)?,
                    envelope: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, rusqlite::Error>>()?;
        drop(stmt);

        let mut batch = ValidatedExpiredMessageBatch {
            valid_rows: Vec::with_capacity(stored_rows.len()),
            corrupt_rows: Vec::new(),
            selected_rowids: Vec::with_capacity(stored_rows.len()),
        };
        for row in stored_rows {
            batch.selected_rowids.push(row.rowid);
            let encoded_bytes = u64::try_from(row.envelope.len()).unwrap_or(u64::MAX);
            let corrupt = |reason| CorruptDurableRow {
                row_key: row.rowid,
                source_kind: QUARANTINE_SOURCE_PENDING_MESSAGE,
                reason,
                encoded_bytes,
            };
            let parsed = (|| {
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
                let timestamp = u64::try_from(row.timestamp)
                    .map_err(|_| corrupt("expired_message_timestamp"))?;
                let envelope = decode_envelope(&row.envelope)
                    .map_err(|_| corrupt("expired_message_envelope"))?;
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
                Ok::<ExpiredMessageRow, CorruptDurableRow>(ExpiredMessageRow {
                    message_id,
                    sender,
                    receiver,
                })
            })();
            match parsed {
                Ok(valid) => batch.valid_rows.push(valid),
                Err(corrupt) => batch.corrupt_rows.push(corrupt),
            }
        }
        Ok(batch)
    }

    fn queue_expiry_notifications(
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
                    let ids_bytes = bincode::serialize(ids_chunk)?;
                    if ids_bytes.len() > MAX_EXPIRED_NOTIFICATION_ENCODED_BYTES {
                        return Err(ChatRelayError::CorruptStoredData {
                            field: "generated_expired_notification_payload_size",
                        });
                    }
                    tx.execute(
                        "INSERT INTO expired_notifications
                         (sender, receiver, message_ids, created_at, pushed)
                         VALUES (?1, ?2, ?3, ?4, 0)",
                        params![sender.as_slice(), receiver.as_slice(), ids_bytes, now],
                    )?;
                }
            }
        }
        Ok(())
    }

    fn delete_expired_message_batch(
        tx: &Transaction<'_>,
        selected_rowids: &[i64],
    ) -> ChatRelayResult<()> {
        let mut stmt = tx.prepare("DELETE FROM pending_messages WHERE rowid = ?1")?;
        let mut deleted = 0usize;
        for rowid in selected_rowids {
            deleted = deleted.saturating_add(stmt.execute(params![rowid])?);
        }
        if deleted != selected_rowids.len() {
            return Err(ChatRelayError::CorruptStoredData {
                field: "expired_message_cleanup_count",
            });
        }
        Ok(())
    }

    fn delete_expired_blob_batch(
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

    fn delete_stale_notification_batch(
        tx: &Transaction<'_>,
        notif_cutoff: i64,
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
            params![notif_cutoff, limit],
        )?)
    }

    fn cleanup_backlog_exists(
        tx: &Transaction<'_>,
        cutoff: i64,
        notif_cutoff: i64,
    ) -> ChatRelayResult<bool> {
        let message_has_more = tx.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pending_messages
                 WHERE status = 0 AND received_at < ?1
             )",
            params![cutoff],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let blob_has_more = tx.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM pending_blobs WHERE received_at < ?1
             )",
            params![cutoff],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let notification_has_more = tx.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM expired_notifications
                 WHERE pushed = 1 OR created_at < ?1
             )",
            params![notif_cutoff],
            |row| row.get::<_, i64>(0),
        )? != 0;
        let max_quarantine_events = i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX);
        let quarantine_has_more = tx.query_row(
            "SELECT
                 EXISTS(
                     SELECT 1 FROM relay_quarantine_events
                     WHERE quarantined_at < ?1
                 )
                 OR (SELECT COUNT(*) FROM relay_quarantine_events) > ?2",
            params![notif_cutoff, max_quarantine_events],
            |row| row.get::<_, i64>(0),
        )? != 0;

        Ok(message_has_more || blob_has_more || notification_has_more || quarantine_has_more)
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

    /// Returns aggregate TTL cleanup execution evidence.
    #[must_use]
    pub fn maintenance_status(&self) -> ChatRelayMaintenanceStatus {
        self.maintenance_status.read().clone()
    }

    /// Records a blocking-worker failure that occurred outside `run_cleanup`.
    ///
    /// Tokio join failures are deliberately converted to stable buckets so a
    /// heartbeat never exposes panic payloads or other runtime internals.
    pub(crate) fn record_maintenance_worker_failure(&self, reason: &'static str) {
        let mut status = self.maintenance_status.write();
        status.cleanup_runs_total = status.cleanup_runs_total.saturating_add(1);
        status.cleanup_failures_total = status.cleanup_failures_total.saturating_add(1);
        status.last_cleanup_at = Some(now_secs());
        status.last_cleanup_status = Some("failed".to_string());
        status.last_cleanup_failure_reason = Some(reason.to_string());
        status.last_cleanup_batches = 0;
        status.last_cleanup_backlog_deferred = false;
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
    use std::sync::Barrier;

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

    fn insert_expired_pending_rows(svc: &ChatRelayService, count: usize, prefix: u8) {
        let mut conn = svc.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .expect("start bulk pending insert");
        {
            let mut stmt = tx
                .prepare(
                    "INSERT INTO pending_messages
                     (message_id, sender, receiver, timestamp, envelope, received_at, status)
                     VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                )
                .expect("prepare bulk pending insert");
            for sequence in 0..count {
                let mut message_id = [0u8; 16];
                message_id[0] = prefix;
                message_id[8..]
                    .copy_from_slice(&u64::try_from(sequence).unwrap_or(u64::MAX).to_be_bytes());
                let envelope = ChatEnvelope {
                    message_id,
                    sender: [0xA2u8; 32],
                    receiver: [0xA3u8; 32],
                    timestamp: 0,
                    ciphertext: vec![0xA4],
                    nonce: [0u8; 24],
                    content_type: ChatContentType::System,
                    signature: [0u8; 64],
                };
                let encoded_envelope = encode_envelope(&envelope).expect("encode expired envelope");
                stmt.execute(params![
                    message_id.as_slice(),
                    envelope.sender.as_slice(),
                    envelope.receiver.as_slice(),
                    encoded_envelope,
                ])
                .expect("insert expired pending row");
            }
        }
        tx.commit().expect("commit bulk pending insert");
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
    fn test_pull_isolates_malformed_row_and_delivers_valid_message() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBEu8; 32];
        let envelope = make_envelope(&kp, receiver);
        let expected_message_id = envelope.message_id;
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 1, ?4, 1, 0)",
                params![
                    [0x01u8; 15].as_slice(),
                    kp.public_key_bytes().as_slice(),
                    receiver.as_slice(),
                    [0xFFu8].as_slice(),
                ],
            )
            .expect("insert malformed pending row");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull with poison-row isolation");
        assert_eq!(messages.len(), 1);
        assert!(!has_more);
        assert_eq!(messages[0].message_id, expected_message_id);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);

        let status = svc.maintenance_status();
        assert_eq!(status.quarantined_pending_messages_total, 1);
        assert_eq!(status.quarantine_events_retained, 1);
        let event: (String, String, i64) = svc
            .conn
            .lock()
            .query_row(
                "SELECT source_kind, reason, row_count FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read pending quarantine event");
        assert_eq!(event.0, QUARANTINE_SOURCE_PENDING_MESSAGE);
        assert_eq!(event.1, "pending_message_id");
        assert_eq!(event.2, 1);
    }

    #[test]
    fn test_pull_quarantines_message_id_envelope_mismatch() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBFu8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET message_id = ?1 WHERE message_id = ?2",
                params![[0xFEu8; 16].as_slice(), envelope.message_id.as_slice()],
            )
            .expect("tamper durable message id");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull mismatched durable row");
        assert!(messages.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read mismatch reason");
        assert_eq!(reason, "pending_message_id_mismatch");
    }

    #[test]
    fn test_pull_quarantines_stored_sender_mismatch_before_delivery() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC2u8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET sender = ?1 WHERE message_id = ?2",
                params![[0xF1u8; 32].as_slice(), envelope.message_id.as_slice()],
            )
            .expect("tamper durable sender");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull mismatched durable sender");
        assert!(messages.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read mismatch reason");
        assert_eq!(reason, "pending_message_sender_mismatch");
    }

    #[test]
    fn test_pull_quarantines_invalid_durable_signature() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC3u8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store valid message");
        let mut tampered = envelope.clone();
        tampered.signature[0] ^= 0xFF;
        let tampered_bytes = encode_envelope(&tampered).expect("encode tampered envelope");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages SET envelope = ?1 WHERE message_id = ?2",
                params![tampered_bytes, envelope.message_id.as_slice()],
            )
            .expect("tamper durable signature");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull invalid durable signature");
        assert!(messages.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read signature reason");
        assert_eq!(reason, "pending_message_signature");
    }

    #[test]
    fn test_store_rejects_timestamp_outside_sqlite_domain() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBCu8; 32];
        let mut envelope = make_envelope(&kp, receiver);
        envelope.timestamp = u64::MAX;
        envelope.signature = kp.sign(&envelope.sign_data());

        let error = svc
            .store_pending(&envelope)
            .expect_err("out-of-range timestamp must be rejected");
        assert!(matches!(error, ChatRelayError::TimestampOutOfRange));
        assert_eq!(error.reason_bucket(), "timestamp_out_of_range");
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);
    }

    #[test]
    fn test_pull_out_of_range_timestamp_fails_closed() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBDu8; 32];
        let mut envelope = make_envelope(&kp, receiver);
        envelope.timestamp = 1;
        envelope.signature = kp.sign(&envelope.sign_data());
        svc.store_pending(&envelope).expect("store pending message");

        let (messages, has_more) = svc
            .pull_pending(&receiver, u64::MAX, &[0u8; 16], 50)
            .expect("bounded pull");
        assert!(messages.is_empty());
        assert!(!has_more);
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
    fn test_store_rejects_message_id_conflict_without_replacing_original() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC0u8; 32];
        let original = make_envelope(&kp, receiver);
        let mut conflict = make_envelope(&kp, receiver);
        conflict.message_id = original.message_id;
        conflict.ciphertext = b"different ciphertext".to_vec();
        conflict.signature = kp.sign(&conflict.sign_data());

        svc.store_pending(&original).expect("store original");
        let error = svc
            .store_pending(&conflict)
            .expect_err("conflicting message id must fail");
        assert!(matches!(error, ChatRelayError::MessageIdConflict));
        assert_eq!(error.reason_bucket(), "message_id_conflict");
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 50)
            .expect("pull original");
        assert_eq!(messages.len(), 1);
        assert!(!has_more);
        assert_eq!(messages[0].envelope.ciphertext, original.ciphertext);
    }

    #[test]
    fn test_pull_zero_limit_makes_progress_with_minimum_page() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xC1u8; 32];
        let envelope = make_envelope(&kp, receiver);
        svc.store_pending(&envelope).expect("store message");

        let (messages, has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 0)
            .expect("zero limit pull");
        assert_eq!(messages.len(), 1);
        assert!(!has_more);
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

    #[test]
    fn test_ack_batch_is_atomic_and_deduplicated() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBD; 32];
        let first = make_envelope(&kp, receiver);
        let second = make_envelope(&kp, receiver);

        svc.store_pending(&first).expect("store first");
        svc.store_pending(&second).expect("store second");
        let deleted = svc
            .ack_messages(
                &[first.message_id, first.message_id, second.message_id],
                &receiver,
            )
            .expect("deduplicated ACK");

        assert_eq!(deleted, 2);
        assert_eq!(
            svc.storage_usage().expect("usage after ACK"),
            ChatRelayStorageUsage::default()
        );
    }

    #[test]
    fn test_ack_batch_above_protocol_ceiling_is_rejected() {
        let svc = make_service();
        let ids = vec![[0x11; 16]; MAX_CHAT_ACK_MESSAGE_IDS + 1];

        assert!(matches!(
            svc.ack_messages(&ids, &[0xBE; 32]),
            Err(ChatRelayError::AckBatchTooLarge {
                size,
                limit: MAX_CHAT_ACK_MESSAGE_IDS,
            }) if size == MAX_CHAT_ACK_MESSAGE_IDS + 1
        ));
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

    #[test]
    fn test_pull_cursor_does_not_skip_rows_across_timestamps() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBC; 32];
        let fixtures = [
            (100, [0xF0; 16]),
            (200, [0xE0; 16]),
            (300, [0xD0; 16]),
            (400, [0xC0; 16]),
        ];

        for (timestamp, message_id) in fixtures {
            let mut envelope = make_envelope(&kp, receiver);
            envelope.timestamp = timestamp;
            envelope.message_id = message_id;
            envelope.signature = kp.sign(&envelope.sign_data());
            svc.store_pending(&envelope).expect("store ordered fixture");
        }

        let (first_page, first_has_more) = svc
            .pull_pending(&receiver, 0, &[0u8; 16], 2)
            .expect("first cursor page");
        assert_eq!(first_page.len(), 2);
        assert!(first_has_more);

        let cursor = first_page.last().expect("first page cursor").message_id;
        let (second_page, second_has_more) = svc
            .pull_pending(&receiver, 0, &cursor, 2)
            .expect("second cursor page");
        assert_eq!(second_page.len(), 2);
        assert!(!second_has_more);

        let actual: HashSet<[u8; 16]> = first_page
            .iter()
            .chain(&second_page)
            .map(|message| message.message_id)
            .collect();
        let expected: HashSet<[u8; 16]> = fixtures
            .into_iter()
            .map(|(_, message_id)| message_id)
            .collect();
        assert_eq!(actual, expected);
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

    #[test]
    fn test_online_dedup_is_atomic_under_concurrency() {
        const WORKERS: usize = 16;
        let dedup = Arc::new(MessageDedup::new(32));
        let barrier = Arc::new(Barrier::new(WORKERS));
        let id = [0x02u8; 16];
        let handles: Vec<_> = (0..WORKERS)
            .map(|_| {
                let dedup = Arc::clone(&dedup);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    dedup.check_and_insert(&id)
                })
            })
            .collect();

        let duplicate_results = handles
            .into_iter()
            .map(|handle| handle.join().expect("dedup worker must not panic"))
            .collect::<Vec<_>>();
        assert_eq!(
            duplicate_results
                .iter()
                .filter(|is_duplicate| !**is_duplicate)
                .count(),
            1,
            "exactly one concurrent caller must win first delivery"
        );
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

    #[test]
    fn test_cleanup_chunks_expiry_notifications_and_reconciles_usage() {
        let mut config = test_config();
        config.max_pending_per_wallet = 100;
        let svc = make_service_with_config(config);
        let kp = IdentityKeyPair::generate();
        let sender = kp.public_key_bytes();
        let receiver = [0xC1; 32];
        let mut expected_ids = HashSet::new();

        for _ in 0..70 {
            let envelope = make_envelope(&kp, receiver);
            expected_ids.insert(envelope.message_id);
            svc.store_pending(&envelope)
                .expect("store expiring message");
        }
        svc.conn
            .lock()
            .execute("UPDATE pending_messages SET received_at = 0", [])
            .expect("age pending messages");

        let (expired, blobs) = svc.run_cleanup().expect("cleanup expired messages");
        assert_eq!(expired, 70);
        assert_eq!(blobs, 0);
        assert_eq!(
            svc.storage_usage().expect("usage after cleanup"),
            ChatRelayStorageUsage::default()
        );

        let (notifications, has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("pull expiry notifications");
        assert!(!has_more);
        assert_eq!(notifications.len(), 3);

        let mut chunk_lengths = Vec::new();
        let mut actual_ids = HashSet::new();
        for notification in &notifications {
            assert_eq!(notification.sender, sender);
            assert_eq!(notification.receiver, receiver);
            let ids = notification.message_ids().expect("decode notification");
            chunk_lengths.push(ids.len());
            actual_ids.extend(ids);
        }
        chunk_lengths.sort_unstable();
        assert_eq!(chunk_lengths, vec![6, 32, 32]);
        assert_eq!(actual_ids, expected_ids);

        let pending_rows: i64 = svc
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count pending rows");
        assert_eq!(pending_rows, 0);
    }

    #[test]
    fn test_cleanup_quarantines_malformed_row_without_blocking() {
        let svc = make_service();
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                params![
                    [0xA1u8; 15].as_slice(),
                    [0xA2u8; 32].as_slice(),
                    [0xA3u8; 32].as_slice(),
                    [0xA4u8].as_slice(),
                ],
            )
            .expect("insert malformed durable row");

        assert_eq!(svc.run_cleanup().expect("quarantine cleanup"), (0, 0));
        assert_eq!(
            svc.storage_usage().expect("usage after quarantine"),
            ChatRelayStorageUsage::default()
        );

        let status = svc.maintenance_status();
        assert_eq!(status.cleanup_runs_total, 1);
        assert_eq!(status.cleanup_failures_total, 0);
        assert_eq!(status.cleanup_batches_total, 1);
        assert_eq!(status.quarantined_pending_messages_total, 1);
        assert_eq!(status.quarantine_events_retained, 1);
        assert_eq!(status.last_cleanup_quarantined_pending_messages, 1);
        assert!(status.last_quarantine_at.is_some());
        assert_eq!(status.last_cleanup_status.as_deref(), Some("succeeded"));

        let conn = svc.conn.lock();
        let pending_rows: i64 = conn
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count pending rows");
        assert_eq!(pending_rows, 0);
        let event: (String, String, i64, i64) = conn
            .query_row(
                "SELECT source_kind, reason, row_count, encoded_bytes
                 FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("read de-identified quarantine event");
        assert_eq!(event.0, QUARANTINE_SOURCE_PENDING_MESSAGE);
        assert_eq!(event.1, "expired_message_id");
        assert_eq!(event.2, 1);
        assert!(event.3 > 0);

        let mut schema_stmt = conn
            .prepare("PRAGMA table_info(relay_quarantine_events)")
            .expect("prepare quarantine schema query");
        let columns: Vec<String> = schema_stmt
            .query_map([], |row| row.get(1))
            .expect("query quarantine columns")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect quarantine columns");
        for forbidden in ["message_id", "sender", "receiver", "envelope", "ciphertext"] {
            assert!(!columns.iter().any(|column| column == forbidden));
        }
    }

    #[test]
    fn test_cleanup_does_not_notify_tampered_stored_sender() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xD4u8; 32];
        let envelope = make_envelope(&kp, receiver);
        let tampered_sender = [0xD5u8; 32];
        svc.store_pending(&envelope).expect("store valid message");
        svc.conn
            .lock()
            .execute(
                "UPDATE pending_messages
                 SET sender = ?1, received_at = 0
                 WHERE message_id = ?2",
                params![tampered_sender.as_slice(), envelope.message_id.as_slice()],
            )
            .expect("tamper expired message sender");

        assert_eq!(svc.run_cleanup().expect("cleanup tampered sender"), (0, 0));
        let (notifications, has_more) = svc
            .pull_pending_notifications(&tampered_sender)
            .expect("pull attacker notifications");
        assert!(notifications.is_empty());
        assert!(!has_more);
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

        let reason: String = svc
            .conn
            .lock()
            .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("read cleanup mismatch reason");
        assert_eq!(reason, "expired_message_sender_mismatch");
    }

    #[test]
    fn test_cleanup_defers_backlog_at_batch_budget_and_recovers_next_run() {
        let svc = make_service();
        insert_expired_pending_rows(&svc, CLEANUP_MESSAGE_BATCH_SIZE + 1, 0x10);

        let (first_expired, first_blobs) = svc
            .run_cleanup_with_batch_budget(1)
            .expect("first bounded cleanup");
        assert_eq!(first_expired, CLEANUP_MESSAGE_BATCH_SIZE);
        assert_eq!(first_blobs, 0);
        assert_eq!(
            svc.storage_usage().expect("first usage").pending_messages,
            1
        );

        let deferred = svc.maintenance_status();
        assert_eq!(deferred.cleanup_runs_total, 1);
        assert_eq!(deferred.cleanup_batches_total, 1);
        assert_eq!(deferred.cleanup_backlog_deferred_total, 1);
        assert_eq!(
            deferred.expired_messages_total,
            u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(u64::MAX)
        );
        assert_eq!(deferred.last_cleanup_batches, 1);
        assert!(deferred.last_cleanup_backlog_deferred);

        let (second_expired, second_blobs) = svc
            .run_cleanup_with_batch_budget(1)
            .expect("second bounded cleanup");
        assert_eq!(second_expired, 1);
        assert_eq!(second_blobs, 0);
        assert_eq!(
            svc.storage_usage().expect("second usage").pending_messages,
            0
        );

        let recovered = svc.maintenance_status();
        assert_eq!(recovered.cleanup_runs_total, 2);
        assert_eq!(recovered.cleanup_batches_total, 2);
        assert_eq!(recovered.cleanup_backlog_deferred_total, 1);
        assert_eq!(
            recovered.expired_messages_total,
            u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE + 1).unwrap_or(u64::MAX)
        );
        assert!(!recovered.last_cleanup_backlog_deferred);
    }

    #[test]
    fn test_cleanup_isolates_trailing_poison_row_after_committed_batch() {
        let svc = make_service();
        insert_expired_pending_rows(&svc, CLEANUP_MESSAGE_BATCH_SIZE, 0x10);
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                params![
                    [0xF0u8; 15].as_slice(),
                    [0xA2u8; 32].as_slice(),
                    [0xA3u8; 32].as_slice(),
                    [0xA4u8].as_slice(),
                ],
            )
            .expect("insert malformed trailing row");

        let (expired, blobs) = svc
            .run_cleanup_with_batch_budget(2)
            .expect("bounded cleanup with poison-row isolation");
        assert_eq!(expired, CLEANUP_MESSAGE_BATCH_SIZE);
        assert_eq!(blobs, 0);

        let status = svc.maintenance_status();
        assert_eq!(status.cleanup_runs_total, 1);
        assert_eq!(status.cleanup_failures_total, 0);
        assert_eq!(status.cleanup_batches_total, 2);
        assert_eq!(
            status.expired_messages_total,
            u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(u64::MAX)
        );
        assert_eq!(status.quarantined_pending_messages_total, 1);
        assert_eq!(status.last_cleanup_batches, 2);
        assert_eq!(status.last_cleanup_status.as_deref(), Some("succeeded"));
        assert_eq!(
            svc.storage_usage()
                .expect("remaining usage")
                .pending_messages,
            0
        );
    }

    #[test]
    fn test_quarantine_persistence_failure_rolls_back_source_deletion() {
        let svc = make_service();
        svc.conn
            .lock()
            .execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
                params![
                    [0xA1u8; 15].as_slice(),
                    [0xA2u8; 32].as_slice(),
                    [0xA3u8; 32].as_slice(),
                    [0xA4u8].as_slice(),
                ],
            )
            .expect("insert malformed durable row");
        svc.conn
            .lock()
            .execute("DROP TABLE relay_quarantine_events", [])
            .expect("simulate quarantine persistence failure");

        assert!(matches!(svc.run_cleanup(), Err(ChatRelayError::Sqlite(_))));
        assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);
        let pending_rows: i64 = svc
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
                row.get(0)
            })
            .expect("count retained source rows");
        assert_eq!(pending_rows, 1);
        let status = svc.maintenance_status();
        assert_eq!(status.cleanup_failures_total, 1);
        assert_eq!(status.quarantined_pending_messages_total, 0);
    }

    #[test]
    fn test_quarantine_event_store_enforces_hard_retention_cap() {
        let svc = make_service();
        {
            let mut conn = svc.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("start quarantine event insert");
            let mut stmt = tx
                .prepare(
                    "INSERT INTO relay_quarantine_events
                     (source_kind, reason, row_count, encoded_bytes, quarantined_at)
                     VALUES (?1, 'test_reason', 1, 1, ?2)",
                )
                .expect("prepare quarantine event insert");
            for _ in 0..=MAX_QUARANTINE_EVENTS {
                stmt.execute(params![QUARANTINE_SOURCE_PENDING_MESSAGE, i64::MAX])
                    .expect("insert quarantine event");
            }
            drop(stmt);
            tx.commit().expect("commit quarantine events");
        }

        svc.run_cleanup_with_batch_budget(1)
            .expect("trim quarantine event overflow");
        let retained: i64 = svc
            .conn
            .lock()
            .query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
                row.get(0)
            })
            .expect("count bounded quarantine events");
        assert_eq!(
            retained,
            i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX)
        );
        let status = svc.maintenance_status();
        assert_eq!(status.quarantine_events_removed_total, 1);
        assert_eq!(
            status.quarantine_events_retained,
            u64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(u64::MAX)
        );
        assert!(!status.last_cleanup_backlog_deferred);
    }

    #[test]
    fn test_cleanup_out_of_range_ttl_fails_closed() {
        let mut config = test_config();
        config.offline_ttl_secs = u64::MAX;
        let svc = ChatRelayService::new(config, [0x42; 32]).expect("service");
        let kp = IdentityKeyPair::generate();
        let receiver = [0xB4; 32];
        let envelope = make_envelope(&kp, receiver);

        svc.store_pending(&envelope).expect("store pending message");
        let (expired, _) = svc.run_cleanup().expect("cleanup");

        assert_eq!(expired, 0);
        assert_eq!(
            svc.storage_usage().expect("storage usage").pending_messages,
            1
        );
    }

    #[test]
    fn test_maintenance_status_deserializes_older_snapshot() {
        let status: ChatRelayMaintenanceStatus = serde_json::from_value(serde_json::json!({
            "cleanup_runs_total": 7,
            "last_cleanup_status": "succeeded"
        }))
        .expect("deserialize backward-compatible maintenance snapshot");

        assert_eq!(status.cleanup_runs_total, 7);
        assert_eq!(status.cleanup_batches_total, 0);
        assert_eq!(status.quarantined_pending_messages_total, 0);
        assert_eq!(status.quarantine_events_retained, 0);
        assert!(!status.last_cleanup_backlog_deferred);
    }

    #[test]
    fn test_expiry_notification_pull_is_bounded_and_pageable() {
        let svc = make_service();
        let sender = [0xD1; 32];
        let receiver = [0xD2u8; 32];
        {
            let mut conn = svc.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("start notification insert");
            for created_at in 0..17i64 {
                let ids = bincode::serialize(&vec![[created_at as u8; 16]])
                    .expect("serialize notification");
                tx.execute(
                    "INSERT INTO expired_notifications
                     (sender, receiver, message_ids, created_at, pushed)
                     VALUES (?1, ?2, ?3, ?4, 0)",
                    params![sender.as_slice(), receiver.as_slice(), ids, created_at],
                )
                .expect("insert notification");
            }
            tx.commit().expect("commit notifications");
        }

        let (first_page, first_has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("first notification page");
        assert_eq!(first_page.len(), MAX_EXPIRED_NOTIFICATIONS_PER_PULL);
        assert!(first_has_more);
        let first_ids: Vec<i64> = first_page
            .iter()
            .map(|notification| notification.id)
            .collect();
        svc.mark_notifications_pushed(&first_ids)
            .expect("mark first page");

        let (second_page, second_has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("second notification page");
        assert_eq!(second_page.len(), 1);
        assert!(!second_has_more);
    }

    #[test]
    fn test_malformed_expiry_notification_isolated_without_blocking_valid_rows() {
        let svc = make_service();
        let sender = [0xE1; 32];
        let valid_receiver = [0xE4; 32];
        let ids = bincode::serialize(&vec![[0xE2; 16]]).expect("serialize notification");
        {
            let mut conn = svc.conn.lock();
            let tx = conn
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .expect("start mixed notification transaction");
            tx.execute(
                "INSERT INTO expired_notifications
                 (sender, receiver, message_ids, created_at, pushed)
                 VALUES (?1, ?2, ?3, 0, 0)",
                params![sender.as_slice(), [0xE3u8; 31].as_slice(), &ids],
            )
            .expect("insert malformed notification");
            tx.execute(
                "INSERT INTO expired_notifications
                 (sender, receiver, message_ids, created_at, pushed)
                 VALUES (?1, ?2, ?3, 1, 0)",
                params![sender.as_slice(), valid_receiver.as_slice(), ids],
            )
            .expect("insert valid notification");
            tx.commit().expect("commit mixed notifications");
        }

        let (notifications, has_more) = svc
            .pull_pending_notifications(&sender)
            .expect("pull must isolate poison row");
        assert_eq!(notifications.len(), 1);
        assert!(!has_more);
        assert_eq!(notifications[0].receiver, valid_receiver);

        let status = svc.maintenance_status();
        assert_eq!(status.quarantined_expired_notifications_total, 1);
        assert_eq!(status.quarantine_events_retained, 1);
        assert!(status.last_quarantine_at.is_some());

        let conn = svc.conn.lock();
        let remaining: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM expired_notifications WHERE pushed = 0",
                [],
                |row| row.get(0),
            )
            .expect("count valid notification");
        assert_eq!(remaining, 1);
        let event: (String, String, i64) = conn
            .query_row(
                "SELECT source_kind, reason, row_count FROM relay_quarantine_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("read notification quarantine event");
        assert_eq!(event.0, QUARANTINE_SOURCE_EXPIRED_NOTIFICATION);
        assert_eq!(event.1, "expired_notification_receiver");
        assert_eq!(event.2, 1);
    }

    #[test]
    fn expired_notifications_are_wired_to_authenticated_chat_pull() {
        let source = include_str!("../server.rs");
        assert!(source.contains("relay.pull_pending_notifications(&wallet)"));
        assert!(source.contains("Self::push_expired_notifications("));
        assert!(source.contains("has_more |= notification_has_more || !delivery_complete"));
        assert!(source.contains("self.spawn_chat_relay_cleanup_task(Arc::clone(relay))"));
        assert!(source.contains("tokio::task::spawn_blocking(move || cleanup_relay.run_cleanup())"));
        assert!(source.contains("tokio::time::MissedTickBehavior::Skip"));
        assert!(source.contains("relay.record_maintenance_worker_failure(reason)"));
        assert!(source.contains("\"maintenance\": relay.maintenance_status()"));
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
