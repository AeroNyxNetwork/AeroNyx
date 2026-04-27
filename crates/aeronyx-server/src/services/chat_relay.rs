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
//
// Main Functionality:
//   - ChatRelayService: Central service managing all chat relay state
//   - Message store/pull/ack: SQLite-backed pending message queue
//   - Blob store/get/delete: SQLite-backed encrypted media cache
//   - TTL cleanup: Expires pending messages and blobs, queues notifications
//   - Expired notifications: Queued ChatExpired delivery for offline senders
//   - Online deduplication: LRU cache prevents duplicate delivery (online path)
//   - WalletRouteCache: In-memory wallet → session routing (v1.3.0-Sovereign)
//
// Dependencies:
//   - aeronyx-core/src/protocol/chat.rs: ChatEnvelope, encode_envelope, decode_envelope
//   - aeronyx-server/src/config.rs: ChatRelayConfig
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
//
// Last Modified:
//   v1.1.0-ChatRelay — Initial implementation
//   v1.3.0-Sovereign — Added wallet_routes: Arc<WalletRouteCache> field
// ============================================================================

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use sha2::Sha256;
use rusqlite::{params, Connection, OptionalExtension};

use tracing::{debug, info, warn};

use aeronyx_core::protocol::chat::{decode_envelope, encode_envelope, ChatEnvelope};

use crate::config::ChatRelayConfig;
use crate::services::wallet_routes::WalletRouteCache;

// ============================================
// Type aliases
// ============================================

type HmacSha256 = Hmac<Sha256>;

// ============================================
// Error type
// ============================================

/// Errors produced by `ChatRelayService`.
#[derive(Debug, thiserror::Error)]
pub enum ChatRelayError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialize(#[from] bincode::Error),

    #[error("Mailbox full: receiver has {current} pending messages (limit {limit})")]
    MailboxFull { current: usize, limit: usize },

    #[error("Blob quota exceeded: receiver has {current} pending blobs (limit {limit})")]
    BlobQuotaExceeded { current: usize, limit: usize },

    #[error("Blob too large: {size} bytes (limit {limit})")]
    BlobTooLarge { size: usize, limit: usize },

    #[error("Blob not found: {blob_id}")]
    BlobNotFound { blob_id: String },

    #[error("Unauthorized: sender mismatch")]
    Unauthorized,
}

pub type ChatRelayResult<T> = Result<T, ChatRelayError>;

// ============================================
// Pending message row (returned from pull)
// ============================================

/// A pending offline message retrieved from the store.
#[derive(Debug)]
pub struct PendingMessage {
    pub message_id: [u8; 16],
    pub envelope: ChatEnvelope,
}

// ============================================
// Expired notification row
// ============================================

/// A queued `ChatExpired` notification for an offline sender.
#[derive(Debug)]
pub struct ExpiredNotification {
    pub id: i64,
    pub sender: [u8; 32],
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
            let oldest_key = self.map
                .iter()
                .min_by_key(|e| *e.value())
                .map(|e| *e.key());
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
        let svc = Self {
            config,
            conn: Mutex::new(conn),
            node_secret,
            dedup: MessageDedup::new(dedup_capacity),
            // v1.3.0-Sovereign: initialise empty route cache
            wallet_routes: Arc::new(WalletRouteCache::new()),
        };

        svc.init_schema()?;
        info!("[CHAT_RELAY] Service initialised (db: {})", svc.config.db_path);
        Ok(svc)
    }

    // ============================================
    // Schema initialisation
    // ============================================

    fn init_schema(&self) -> ChatRelayResult<()> {
        let conn = self.conn.lock();
        conn.execute_batch("
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
        ")?;
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
        let mut mac = HmacSha256::new_from_slice(&self.node_secret)
            .expect("HMAC accepts any key length");
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

    /// Stores a pending offline message for a receiver that is not currently online.
    pub fn store_pending(&self, envelope: &ChatEnvelope) -> ChatRelayResult<()> {
        let now = now_secs();
        let receiver = envelope.receiver;

        let conn = self.conn.lock();

        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM pending_messages WHERE receiver = ? AND status = 0",
            params![receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )? as usize;

        if count >= self.config.max_pending_per_wallet {
            return Err(ChatRelayError::MailboxFull {
                current: count,
                limit: self.config.max_pending_per_wallet,
            });
        }

        let envelope_bytes = encode_envelope(envelope)?;

        conn.execute(
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

        debug!(
            id = %hex::encode(envelope.message_id),
            receiver = %hex::encode(&receiver[..4]),
            "[CHAT_RELAY] Message stored (pending)"
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

        let has_more = rows.len() == effective_limit;
        let page = &rows[..rows.len().min(effective_limit - 1)];

        let mut messages = Vec::with_capacity(page.len());
        for (id_bytes, env_bytes) in page {
            let mut message_id = [0u8; 16];
            if id_bytes.len() == 16 {
                message_id.copy_from_slice(id_bytes);
            }
            let envelope = decode_envelope(env_bytes)?;
            messages.push(PendingMessage { message_id, envelope });
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

        debug!(
            count = deleted,
            receiver = %hex::encode(&receiver_wallet[..4]),
            "[CHAT_RELAY] Messages ACKed and deleted"
        );
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

        let conn = self.conn.lock();

        let count: usize = conn.query_row(
            "SELECT COUNT(*) FROM pending_blobs WHERE receiver = ?",
            params![receiver.as_slice()],
            |row| row.get::<_, i64>(0),
        )? as usize;

        if count >= self.config.max_blobs_per_receiver {
            return Err(ChatRelayError::BlobQuotaExceeded {
                current: count,
                limit: self.config.max_blobs_per_receiver,
            });
        }

        let blob_id = self.compute_blob_id(sender, receiver, file_hash);
        let now = now_secs();

        conn.execute(
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

        info!(
            blob_id = %blob_id,
            size = data.len(),
            sender = %hex::encode(&sender[..4]),
            "[CHAT_RELAY] Blob stored"
        );
        Ok(blob_id)
    }

    pub fn get_blob(&self, blob_id: &str) -> ChatRelayResult<Vec<u8>> {
        let conn = self.conn.lock();

        let data: Option<Vec<u8>> = conn.query_row(
            "SELECT data FROM pending_blobs WHERE blob_id = ?",
            params![blob_id],
            |row| row.get::<_, Vec<u8>>(0),
        ).optional()?;

        match data {
            None => Err(ChatRelayError::BlobNotFound { blob_id: blob_id.to_string() }),
            Some(bytes) => {
                let _ = conn.execute(
                    "UPDATE pending_blobs SET downloaded = 1 WHERE blob_id = ?",
                    params![blob_id],
                );
                debug!(blob_id = %blob_id, "[CHAT_RELAY] Blob retrieved");
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
            info!(blob_id = %blob_id, "[CHAT_RELAY] Blob deleted by sender");
            return Ok(());
        }

        let exists: bool = conn.query_row(
            "SELECT 1 FROM pending_blobs WHERE blob_id = ?",
            params![blob_id],
            |_| Ok(true),
        ).optional()?.unwrap_or(false);

        if exists {
            Err(ChatRelayError::Unauthorized)
        } else {
            Err(ChatRelayError::BlobNotFound { blob_id: blob_id.to_string() })
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

        let rows = stmt.query_map(params![sender.as_slice()], |row| {
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

        Ok(rows)
    }

    pub fn mark_notifications_pushed(&self, ids: &[i64]) -> ChatRelayResult<()> {
        if ids.is_empty() { return Ok(()); }
        let conn = self.conn.lock();
        for id in ids {
            conn.execute(
                "UPDATE expired_notifications SET pushed = 1 WHERE id = ?",
                params![id],
            )?;
        }
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
                    Some(ExpiredRow { message_id: mid, sender, receiver })
                })
                .collect();

            let expired_message_count = expired_rows.len();

            use std::collections::HashMap;
            let mut by_sender: HashMap<[u8; 32], HashMap<[u8; 32], Vec<[u8; 16]>>> =
                HashMap::new();
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

            let expired_blobs =
                conn.execute("DELETE FROM pending_blobs WHERE received_at < ?", params![cutoff])?;

            conn.execute(
                "DELETE FROM expired_notifications WHERE pushed = 1 OR created_at < ?",
                params![notif_cutoff],
            )?;

            Ok((expired_message_count, expired_blobs))
        })();

        match &result {
            Ok(_) => { conn.execute_batch("COMMIT")?; }
            Err(_) => { let _ = conn.execute_batch("ROLLBACK"); }
        }

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
    // Accessors
    // ============================================

    #[must_use]
    pub fn config(&self) -> &ChatRelayConfig {
        &self.config
    }
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
    use sha2::{Digest, Sha256};
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::ChatContentType;
    use std::net::SocketAddr;
    use aeronyx_common::types::SessionId;

    fn test_config() -> ChatRelayConfig {
        ChatRelayConfig {
            enabled: true,
            db_path: ":memory:".to_string(),
            offline_ttl_secs: 259_200,
            max_pending_per_wallet: 5,
            max_message_size: 65_536,
            max_blob_size: 1_024,
            max_blobs_per_receiver: 3,
            cleanup_interval_secs: 60,
            dedup_lru_capacity: 10,
            expired_notification_ttl_secs: 604_800,
        }
    }

    fn make_service() -> ChatRelayService {
        let secret = derive_node_secret(&[0x42u8; 32]);
        ChatRelayService::new(test_config(), secret).expect("init")
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
        svc.wallet_routes.announce(&wallet, sid.clone(), make_addr(9001));

        // lookup via clone — must see the same entry
        let results = routes_clone.lookup(&wallet);
        assert_eq!(results.len(), 1, "Arc clone must share the same underlying cache");
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

        let (msgs, has_more) = svc.pull_pending(&receiver, 0, &[0u8; 16], 50).expect("pull");
        assert_eq!(msgs.len(), 1);
        assert!(!has_more);
        assert_eq!(msgs[0].message_id, mid);

        let deleted = svc.ack_messages(&[mid], &receiver).expect("ack");
        assert_eq!(deleted, 1);

        let (msgs2, _) = svc.pull_pending(&receiver, 0, &[0u8; 16], 50).expect("pull2");
        assert!(msgs2.is_empty());
    }

    #[test]
    fn test_store_duplicate_ignored() {
        let svc = make_service();
        let kp = IdentityKeyPair::generate();
        let receiver = [0xBBu8; 32];
        let env = make_envelope(&kp, receiver);

        svc.store_pending(&env).expect("first store");
        svc.store_pending(&env).expect("duplicate store — should not error");

        let (msgs, _) = svc.pull_pending(&receiver, 0, &[0u8; 16], 50).expect("pull");
        assert_eq!(msgs.len(), 1);
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

        let (msgs, _) = svc.pull_pending(&receiver, 0, &[0u8; 16], 50).expect("pull");
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

        let blob_id = svc.put_blob(&sender, &receiver, data, &file_hash).expect("put");
        assert_eq!(blob_id.len(), 32);

        let fetched = svc.get_blob(&blob_id).expect("get");
        assert_eq!(fetched, data);

        svc.delete_blob(&blob_id, &sender).expect("delete");
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
    fn test_blob_delete_wrong_sender_rejected() {
        let svc = make_service();
        let sender = [0xAAu8; 32];
        let receiver = [0xBBu8; 32];
        let data = b"file";
        let file_hash: [u8; 32] = Sha256::digest(data).into();

        let blob_id = svc.put_blob(&sender, &receiver, data, &file_hash).expect("put");
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

        let (msgs, _) = svc.pull_pending(&receiver, 0, &[0u8; 16], 50).expect("pull");
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
