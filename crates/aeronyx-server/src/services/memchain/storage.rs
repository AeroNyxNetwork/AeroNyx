// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage.rs
// ============================================
//! # MemoryStorage — SQLite 持久化存储引擎
//! # MemoryStorage — SQLite Persistent Storage Engine
//!
//! ## Creation Reason
//! MemChain v2.0 的唯一主存储引擎，替代旧的 MemPool + AofWriter 双引擎。
//! 使用 rusqlite (bundled) + WAL 模式，提供索引查询、事务安全、生命周期管理。
//!
//! The sole primary storage engine for MemChain v2.0. Uses rusqlite (bundled)
//! with WAL mode for indexed queries, crash-safe transactions, and lifecycle mgmt.
//!
//! ## v2.1 Changes (vs v1.0.0)
//! - `embedding` 列改为 BLOB（f32 LE 序列化），去掉 `encrypted_embedding`
//! - `access_count` 改为 INTEGER (u32)
//! - `layer = 3` 表示 Archive（独立层级，不是 status）
//! - `drain_episodes_for_compaction` 改为将 layer 从 Episode 变更为 Archive
//!   （而非标记 status=Archived），使归档记忆仍可以极低权重被召回
//! - 新增 `get_records_needing_embedding()` 用于启动时重建向量索引
//!
//! ## Schema (v2)
//! ```sql
//! CREATE TABLE records (
//!     record_id         BLOB(32) PRIMARY KEY,
//!     owner             BLOB(32) NOT NULL,
//!     timestamp         INTEGER NOT NULL,
//!     layer             INTEGER NOT NULL,  -- 0=Identity,1=Knowledge,2=Episode,3=Archive
//!     topic_tags        TEXT NOT NULL,      -- JSON array
//!     source_ai         TEXT NOT NULL,
//!     status            INTEGER NOT NULL DEFAULT 0,  -- 0=Active,1=Superseded,2=Revoked
//!     supersedes        BLOB(32),
//!     encrypted_content BLOB NOT NULL,
//!     embedding         BLOB,              -- f32 LE bytes, NULL if no embedding
//!     signature         BLOB(64) NOT NULL,
//!     access_count      INTEGER NOT NULL DEFAULT 0,
//!     created_at        INTEGER NOT NULL
//! );
//! ```
//!
//! ## Thread Safety
//! `rusqlite::Connection` behind `tokio::sync::Mutex`. Phase 2+ can use r2d2 pooling.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Schema migrations: use `ALTER TABLE` in `migrate()`, NEVER drop tables.
//! - `record_id` is PRIMARY KEY. Duplicate inserts use `INSERT OR IGNORE`.
//! - `embedding` is stored as raw f32 little-endian bytes. 384-dim = 1536 bytes.
//! - Miner 压实: 改 `layer` 从 Episode→Archive（不是改 status），保持 status=Active。
//! - `created_at` is local insertion time, not the record's `timestamp`.
//!
//! ## Last Modified
//! v1.0.0 - Initial SQLite storage engine
//! v2.1.0 - 🌟 4-layer support, plaintext embedding BLOB, compaction via layer change

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord, RecordStatus};

// ============================================
// RawLog Encryption (D3)
// ============================================

/// Derive a rawlog encryption key from the owner's Ed25519 private key.
///
/// Uses HKDF-SHA256 with salt="memchain-rawlog", info="v1".
/// Output: 32-byte key suitable for ChaCha20-Poly1305.
pub fn derive_rawlog_key(owner_secret: &[u8; 32]) -> [u8; 32] {
    use sha2::Sha256;
    use hkdf::Hkdf;
    let hk = Hkdf::<Sha256>::new(Some(b"memchain-rawlog"), owner_secret);
    let mut key = [0u8; 32];
    hk.expand(b"v1", &mut key).expect("HKDF expand should not fail for 32 bytes");
    key
}

/// Encrypt content bytes for rawlog storage.
/// Format: nonce(12) || ciphertext(len + 16 tag)
fn encrypt_rawlog_content(key: &[u8; 32], plaintext: &[u8]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, NewAead};

    let cipher_key = Key::from_slice(key);
    let cipher = ChaCha20Poly1305::new(cipher_key);

    // Generate random 12-byte nonce
    let mut nonce_bytes = [0u8; 12];
    use rand::RngCore;
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext)
        .map_err(|e| format!("ChaCha20 encrypt: {}", e))?;

    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt content bytes from rawlog storage.
/// Input format: nonce(12) || ciphertext(len + 16 tag)
fn decrypt_rawlog_content(key: &[u8; 32], stored: &[u8]) -> Result<Vec<u8>, String> {
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use chacha20poly1305::aead::{Aead, NewAead};

    if stored.len() < 12 + 16 {
        return Err("Ciphertext too short".into());
    }

    let cipher_key = Key::from_slice(key);
    let cipher = ChaCha20Poly1305::new(cipher_key);
    let nonce = Nonce::from_slice(&stored[..12]);
    let ciphertext = &stored[12..];

    cipher.decrypt(nonce, ciphertext)
        .map_err(|e| format!("ChaCha20 decrypt: {}", e))
}

// ============================================
// RawLogRow — query result struct
// ============================================

/// A row from the raw_logs table.
#[derive(Debug, Clone)]
pub struct RawLogRow {
    pub log_id: i64,
    pub session_id: String,
    pub turn_index: i64,
    pub role: String,
    pub content: Vec<u8>,
    pub encrypted: i64,
    pub recall_context: Option<String>,
    pub extractable: Option<i64>,
    pub feedback_signal: Option<i64>,
}

/// LRU 缓存容量（条目数）— 单用户 1000 条约 2MB 内存
/// LRU cache capacity — 1000 records ≈ 2MB for a single user
const LRU_CACHE_CAPACITY: usize = 1000;

// ============================================
// Constants
// ============================================

/// Schema 版本号，增加时需要添加迁移逻辑
const SCHEMA_VERSION: u32 = 2;

/// 分页查询默认页大小
const DEFAULT_PAGE_SIZE: usize = 100;

// ============================================
// Embedding 序列化辅助
// Embedding serialization helpers
// ============================================

/// 将 `Vec<f32>` 序列化为 little-endian 字节序列
/// Serialize `Vec<f32>` to little-endian byte sequence
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// 从 little-endian 字节序列反序列化为 `Vec<f32>`
/// Deserialize little-endian bytes to `Vec<f32>`
fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            f32::from_le_bytes(buf)
        })
        .collect()
}

// ============================================
// StorageStats — 聚合统计
// ============================================

/// 存储引擎聚合统计 / Aggregate storage statistics
#[derive(Debug, Clone, serde::Serialize)]
pub struct StorageStats {
    /// 数据库中所有记录数（含所有 status）
    pub total_records: u64,
    /// 仅 status=Active 的记录数
    pub active_records: u64,
    /// 按 Layer 分布（仅 Active）/ Per-layer distribution (Active only)
    pub by_layer: LayerCounts,
    /// 加密内容总字节数（Active only）
    pub content_bytes: u64,
    /// 有 embedding 的记录数
    pub records_with_embedding: u64,
    /// 本次会话插入数 / Session inserts
    pub session_inserts: u64,
    /// 本次会话拒绝数 / Session rejects
    pub session_rejects: u64,
}

/// 按层级的活跃记录计数 / Per-layer active record counts
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LayerCounts {
    pub identity: u64,
    pub knowledge: u64,
    pub episode: u64,
    pub archive: u64,
}

// ============================================
// MemoryStorage
// ============================================

/// SQLite 持久化存储引擎
/// SQLite persistent storage engine for MemoryRecord
pub struct MemoryStorage {
    conn: TokioMutex<Connection>,
    total_inserted: AtomicU64,
    total_rejected: AtomicU64,
    /// LRU 缓存：最近访问的 MemoryRecord，减少 recall 时的 SQLite IO
    /// LRU cache: recently accessed records to reduce SQLite IO during recall
    cache: RwLock<LruCache>,
}

/// 简易 LRU 缓存（基于 LinkedHashMap 语义的 Vec + HashMap）
/// 对于 1000 条的规模，HashMap 查找 + Vec 淘汰足够高效。
/// Phase 2+ 可替换为 `lru` crate 的专业实现。
struct LruCache {
    map: HashMap<[u8; 32], (usize, MemoryRecord)>, // record_id → (order, record)
    order_counter: usize,
    capacity: usize,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self { map: HashMap::with_capacity(capacity), order_counter: 0, capacity }
    }

    fn get(&mut self, id: &[u8; 32]) -> Option<&MemoryRecord> {
        if let Some(entry) = self.map.get_mut(id) {
            self.order_counter += 1;
            entry.0 = self.order_counter; // 刷新访问顺序
            Some(&entry.1)
        } else {
            None
        }
    }

    fn put(&mut self, record: MemoryRecord) {
        let id = record.record_id;
        self.order_counter += 1;
        self.map.insert(id, (self.order_counter, record));

        // 超容量时淘汰最久未访问的
        if self.map.len() > self.capacity {
            if let Some((&evict_id, _)) = self.map.iter().min_by_key(|(_, (ord, _))| *ord) {
                self.map.remove(&evict_id);
            }
        }
    }

    fn invalidate(&mut self, id: &[u8; 32]) {
        self.map.remove(id);
    }

    fn clear(&mut self) {
        self.map.clear();
        self.order_counter = 0;
    }
}

impl MemoryStorage {
    /// 打开或创建 SQLite 数据库
    /// Open or create the SQLite database at the given path
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();

        let conn = if path.to_str() == Some(":memory:") {
            Connection::open_in_memory()
        } else {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        format!("Failed to create DB directory '{}': {}", parent.display(), e)
                    })?;
                }
            }
            Connection::open(path)
        }
        .map_err(|e| format!("Failed to open SQLite: {}", e))?;

        // 性能优化 pragmas / Performance pragmas
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -8000;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;"
        ).map_err(|e| format!("Failed to set pragmas: {}", e))?;

        Self::create_schema(&conn)?;
        Self::maybe_migrate(&conn)?;

        info!(path = %path.display(), "[STORAGE] ✅ SQLite database opened (schema v{})", SCHEMA_VERSION);

        Ok(Self {
            conn: TokioMutex::new(conn),
            total_inserted: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            cache: RwLock::new(LruCache::new(LRU_CACHE_CAPACITY)),
        })
    }

    fn create_schema(conn: &Connection) -> Result<(), String> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS records (
                record_id         BLOB PRIMARY KEY,
                owner             BLOB NOT NULL,
                timestamp         INTEGER NOT NULL,
                layer             INTEGER NOT NULL,
                topic_tags        TEXT NOT NULL DEFAULT '[]',
                source_ai         TEXT NOT NULL DEFAULT '',
                status            INTEGER NOT NULL DEFAULT 0,
                supersedes        BLOB,
                encrypted_content BLOB NOT NULL DEFAULT x'',
                embedding         BLOB,
                embedding_model   TEXT NOT NULL DEFAULT '',
                embedding_dim     INTEGER NOT NULL DEFAULT 0,
                signature         BLOB NOT NULL,
                access_count      INTEGER NOT NULL DEFAULT 0,
                created_at        INTEGER NOT NULL,
                archived_at       INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_owner
                ON records(owner);
            CREATE INDEX IF NOT EXISTS idx_owner_layer_status
                ON records(owner, layer, status);
            CREATE INDEX IF NOT EXISTS idx_status_layer
                ON records(status, layer);
            CREATE INDEX IF NOT EXISTS idx_timestamp
                ON records(timestamp);

            -- Raw conversation logs (new in v4, but created in initial schema
            -- so fresh installs get it immediately)
            CREATE TABLE IF NOT EXISTS raw_logs (
                log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                turn_index      INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         BLOB NOT NULL,
                source_ai       TEXT NOT NULL DEFAULT '',
                recall_context  TEXT DEFAULT NULL,
                extractable     INTEGER DEFAULT 1,
                feedback_signal INTEGER DEFAULT NULL,
                encrypted       INTEGER DEFAULT 0,
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rawlogs_session
                ON raw_logs(session_id, turn_index);
            CREATE INDEX IF NOT EXISTS idx_rawlogs_feedback
                ON raw_logs(feedback_signal);

            -- Memory co-occurrence graph edges
            CREATE TABLE IF NOT EXISTS memory_edges (
                source_id   BLOB NOT NULL,
                target_id   BLOB NOT NULL,
                edge_type   TEXT NOT NULL DEFAULT 'co_occurred',
                weight      REAL NOT NULL DEFAULT 1.0,
                created_at  INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);

            -- Per-user MVF learned weights (9 dim × 3 arrays = 108 bytes)
            CREATE TABLE IF NOT EXISTS user_weights (
                owner       BLOB PRIMARY KEY,
                weights     BLOB NOT NULL,
                version     INTEGER NOT NULL DEFAULT 0,
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );

            -- Feedback events for SGD training
            CREATE TABLE IF NOT EXISTS memory_feedback (
                feedback_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                owner         BLOB NOT NULL,
                memory_id     BLOB NOT NULL,
                session_id    TEXT NOT NULL,
                turn_index    INTEGER NOT NULL,
                signal        INTEGER NOT NULL,
                features      BLOB,
                prediction    REAL,
                created_at    INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_owner ON memory_feedback(owner);
            CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);

            CREATE TABLE IF NOT EXISTS chain_state (
                key   TEXT PRIMARY KEY,
                value BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );"
        ).map_err(|e| format!("Schema creation failed: {}", e))?;

        let existing: Option<u32> = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
            .optional()
            .map_err(|e| format!("Read schema version: {}", e))?;

        if existing.is_none() {
            conn.execute("INSERT INTO schema_version (version) VALUES (?1)", params![SCHEMA_VERSION])
                .map_err(|e| format!("Insert schema version: {}", e))?;
        }

        Ok(())
    }

    /// Schema 迁移 — v1 → v2 的差量
    /// Schema migration — delta from v1 to v2
    fn maybe_migrate(conn: &Connection) -> Result<(), String> {
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
            .unwrap_or(1);

        if current < 2 {
            info!("[STORAGE] Migrating schema v{} → v2", current);

            // v1 had `encrypted_embedding BLOB`; v2 renames to `embedding BLOB`
            // SQLite doesn't support RENAME COLUMN before 3.25, so we check if
            // the old column exists and add the new one if missing.
            let has_embedding: bool = conn
                .prepare("SELECT embedding FROM records LIMIT 0")
                .is_ok();

            if !has_embedding {
                // Old schema — add `embedding` column
                let _ = conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN embedding BLOB;"
                );
                info!("[STORAGE] Added `embedding` column");
            }

            conn.execute(
                "UPDATE schema_version SET version = ?1",
                params![SCHEMA_VERSION],
            ).map_err(|e| format!("Update schema version: {}", e))?;

            info!("[STORAGE] ✅ Migration to v2 complete");
        }

        Ok(())
    }

    // ========================================
    // Insert — 插入记录
    // ========================================

    /// 验证哈希完整性后插入记录（幂等：重复 ID 静默跳过）
    /// Validate hash integrity and insert record (idempotent: duplicate ID silently ignored)
    ///
    /// # Arguments
    /// * `record` - 要插入的记忆记录
    /// * `embedding_model` - embedding 模型标识（如 "minilm-l6-v2"）
    pub async fn insert(&self, record: &MemoryRecord, embedding_model: &str) -> bool {
        if !record.verify_id() {
            warn!(
                record_id = hex::encode(record.record_id),
                "[STORAGE] ❌ Rejected: hash mismatch"
            );
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let tags_json = serde_json::to_string(&record.topic_tags)
            .unwrap_or_else(|_| "[]".to_string());

        let embedding_blob: Option<Vec<u8>> = if record.has_embedding() {
            Some(embedding_to_bytes(&record.embedding))
        } else {
            None
        };

        let embedding_dim = record.embedding_dim() as i64;

        let conn = self.conn.lock().await;

        let result = conn.execute(
            "INSERT OR IGNORE INTO records (
                record_id, owner, timestamp, layer, topic_tags, source_ai,
                status, supersedes, encrypted_content, embedding,
                embedding_model, embedding_dim,
                signature, access_count, created_at
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            params![
                record.record_id.as_slice(),
                record.owner.as_slice(),
                record.timestamp as i64,
                record.layer as u8 as i64,
                tags_json,
                record.source_ai,
                record.status as u8 as i64,
                record.supersedes.as_ref().map(|s| s.as_slice()),
                record.encrypted_content.as_slice(),
                embedding_blob.as_deref(),
                embedding_model,
                embedding_dim,
                record.signature.as_slice(),
                record.access_count as i64,
                now,
            ],
        );

        match result {
            Ok(changes) if changes > 0 => {
                self.total_inserted.fetch_add(1, Ordering::Relaxed);
                // 回填 LRU 缓存
                self.cache.write().put(record.clone());
                debug!(
                    record_id = hex::encode(record.record_id),
                    layer = %record.layer,
                    "[STORAGE] ✅ Inserted"
                );
                true
            }
            Ok(_) => {
                debug!(record_id = hex::encode(record.record_id), "[STORAGE] Duplicate, skipped");
                false
            }
            Err(e) => {
                error!(record_id = hex::encode(record.record_id), error = %e, "[STORAGE] ❌ Insert failed");
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    // ========================================
    // Query — 查询
    // ========================================

    /// 按 record_id 查找单条记录（LRU 缓存优先）
    /// Lookup by record_id (LRU cache first, then SQLite fallback)
    pub async fn get(&self, record_id: &[u8; 32]) -> Option<MemoryRecord> {
        // 先查缓存
        {
            let mut cache = self.cache.write();
            if let Some(record) = cache.get(record_id) {
                return Some(record.clone());
            }
        }

        // 缓存未命中 → 查 SQLite
        let conn = self.conn.lock().await;
        let result = conn.query_row(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count
             FROM records WHERE record_id = ?1",
            params![record_id.as_slice()],
            |row| Self::row_to_record(row),
        )
        .optional()
        .unwrap_or_else(|e| { error!(error=%e, "[STORAGE] Query failed"); None });

        // 回填缓存
        if let Some(ref record) = result {
            self.cache.write().put(record.clone());
        }

        result
    }

    /// 查询某 owner 的 Active 记录，可选 layer 过滤
    /// Query active records for an owner, with optional layer filter
    pub async fn get_active_records(
        &self,
        owner: &[u8; 32],
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;

        if let Some(l) = layer {
            Self::query_rows(&conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,
                        signature,access_count
                 FROM records
                 WHERE owner=?1 AND status=0 AND layer=?2
                 ORDER BY timestamp DESC LIMIT ?3",
                params![owner.as_slice(), l as u8 as i64, limit as i64],
            )
        } else {
            Self::query_rows(&conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,
                        signature,access_count
                 FROM records
                 WHERE owner=?1 AND status=0
                 ORDER BY timestamp DESC LIMIT ?2",
                params![owner.as_slice(), limit as i64],
            )
        }
    }

    /// 按 owner + timestamp 查询（用于 P2P 同步）
    pub async fn query_by_owner_after(
        &self,
        owner: &[u8; 32],
        after_timestamp: u64,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        Self::query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count
             FROM records
             WHERE owner=?1 AND timestamp>?2
             ORDER BY timestamp ASC LIMIT ?3",
            params![owner.as_slice(), after_timestamp as i64, DEFAULT_PAGE_SIZE as i64],
        )
    }

    /// 获取所有有 embedding 的 Active 记录（启动时重建向量索引用）
    /// Get all active records with embeddings for vector index rebuild on startup.
    ///
    /// Returns `(MemoryRecord, embedding_model)` tuples so the caller can
    /// insert each vector into the correct partition.
    pub async fn get_records_with_embedding(&self, owner: &[u8; 32]) -> Vec<(MemoryRecord, String)> {
        let conn = self.conn.lock().await;

        let mut stmt = match conn.prepare(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count, embedding_model
             FROM records
             WHERE owner=?1 AND status=0 AND embedding IS NOT NULL
             ORDER BY timestamp DESC"
        ) {
            Ok(s) => s,
            Err(e) => { error!(error=%e, "[STORAGE] Prepare failed"); return Vec::new(); }
        };

        stmt.query_map(params![owner.as_slice()], |row| {
            let record = Self::row_to_record(row)?;
            let model: String = row.get(12)?;
            Ok((record, model))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    // ========================================
    // Lifecycle — 生命周期管理
    // ========================================

    /// 更新记录的 status / Update record status
    pub async fn update_status(&self, record_id: &[u8; 32], new_status: RecordStatus) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1 WHERE record_id=?2",
            params![new_status as u8 as i64, record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => { debug!(record_id=hex::encode(record_id), %new_status, "[STORAGE] ✅ Status updated"); true }
            Ok(_) => { warn!(record_id=hex::encode(record_id), "[STORAGE] Not found for status update"); false }
            Err(e) => { error!(error=%e, "[STORAGE] Status update failed"); false }
        }
    }

    /// 撤销记忆：标记 Revoked + 擦除加密内容和 embedding
    /// Revoke memory: mark Revoked + erase encrypted content and embedding
    pub async fn revoke(&self, record_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1, encrypted_content=x'', embedding=NULL
             WHERE record_id=?2",
            params![RecordStatus::Revoked as u8 as i64, record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                // 缓存失效
                self.cache.write().invalidate(record_id);
                info!(record_id=hex::encode(record_id), "[STORAGE] 🗑️ Revoked");
                true
            }
            Ok(_) => { warn!(record_id=hex::encode(record_id), "[STORAGE] Not found for revoke"); false }
            Err(e) => { error!(error=%e, "[STORAGE] Revoke failed"); false }
        }
    }

    /// 递增访问计数（召回命中时）
    /// Increment access count (on recall hit)
    pub async fn increment_access(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET access_count=access_count+1 WHERE record_id=?1",
            params![record_id.as_slice()],
        );
    }

    // ========================================
    // Miner 支持 — 记忆压实
    // Miner Support — Memory Compaction
    // ========================================

    /// 统计某 layer 的 Active 记录数 / Count active records for a layer
    pub async fn count_by_layer(&self, layer: MemoryLayer) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM records WHERE status=0 AND layer=?1",
            params![layer as u8 as i64],
            |row| row.get::<_, i64>(0),
        ).unwrap_or(0) as u64
    }

    /// 将 Episode 记录压实为 Archive（改 layer，保持 status=Active）。
    /// Compact episodes to archive: change layer to Archive, keep status=Active.
    ///
    /// 这是 v2.1 的关键改动：Archive 是独立 Layer，不是 Status。
    /// 压实后的记忆仍以极低权重（0.05）参与召回。
    ///
    /// This is the key v2.1 change: Archive is a separate Layer, not Status.
    /// Compacted memories still participate in recall at very low weight (0.05).
    pub async fn compact_episodes_to_archive(
        &self,
        owner: &[u8; 32],
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;

        // Step 1: 选出最老的 Active Episode
        let records = Self::query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count
             FROM records
             WHERE owner=?1 AND status=0 AND layer=?2
             ORDER BY timestamp ASC LIMIT ?3",
            params![
                owner.as_slice(),
                MemoryLayer::Episode as u8 as i64,
                limit as i64,
            ],
        );

        if records.is_empty() {
            return records;
        }

        // Step 2: 事务内批量改 layer → Archive
        if conn.execute_batch("BEGIN TRANSACTION").is_err() {
            return Vec::new();
        }

        for r in &records {
            let now_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            if let Err(e) = conn.execute(
                "UPDATE records SET layer=?1, archived_at=?2 WHERE record_id=?3",
                params![
                    MemoryLayer::Archive as u8 as i64,
                    now_ts,
                    r.record_id.as_slice(),
                ],
            ) {
                error!(error=%e, "[STORAGE] ❌ Compact update failed, rolling back");
                let _ = conn.execute_batch("ROLLBACK");
                return Vec::new();
            }
        }

        if conn.execute_batch("COMMIT").is_err() {
            return Vec::new();
        }

        info!(count = records.len(), "[STORAGE] ⛏️ Episodes compacted to Archive layer");
        records
    }

    // ========================================
    // Chain State — 区块链状态
    // ========================================

    pub async fn set_chain_state(&self, block_hash: &[u8; 32], height: u64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT OR REPLACE INTO chain_state (key,value) VALUES ('last_block_hash',?1)",
            params![block_hash.as_slice()],
        );
        let _ = conn.execute(
            "INSERT OR REPLACE INTO chain_state (key,value) VALUES ('last_block_height',?1)",
            params![height.to_le_bytes().as_slice()],
        );
    }

    pub async fn last_block_hash(&self) -> [u8; 32] {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_hash'",
            [], |row| {
                let blob: Vec<u8> = row.get(0)?;
                let mut h = [0u8; 32];
                if blob.len() == 32 { h.copy_from_slice(&blob); }
                Ok(h)
            },
        ).unwrap_or([0u8; 32])
    }

    pub async fn last_block_height(&self) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_height'",
            [], |row| {
                let blob: Vec<u8> = row.get(0)?;
                if blob.len() == 8 {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&blob);
                    Ok(u64::from_le_bytes(b))
                } else { Ok(0u64) }
            },
        ).unwrap_or(0)
    }

    // ========================================
    // Statistics — 统计
    // ========================================

    pub async fn stats(&self) -> StorageStats {
        let conn = self.conn.lock().await;

        let q = |sql: &str, p: &[&dyn rusqlite::ToSql]| -> u64 {
            conn.query_row(sql, p, |r| r.get::<_, i64>(0)).unwrap_or(0) as u64
        };

        let total = q("SELECT COUNT(*) FROM records", &[]);
        let active = q("SELECT COUNT(*) FROM records WHERE status=0", &[]);

        StorageStats {
            total_records: total,
            active_records: active,
            by_layer: LayerCounts {
                identity:  q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=0", &[]),
                knowledge: q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=1", &[]),
                episode:   q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=2", &[]),
                archive:   q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=3", &[]),
            },
            content_bytes: q("SELECT COALESCE(SUM(LENGTH(encrypted_content)),0) FROM records WHERE status=0", &[]),
            records_with_embedding: q("SELECT COUNT(*) FROM records WHERE status=0 AND embedding IS NOT NULL", &[]),
            session_inserts: self.total_inserted.load(Ordering::Relaxed),
            session_rejects: self.total_rejected.load(Ordering::Relaxed),
        }
    }

    pub async fn count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM records", [], |r| r.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }

    pub fn total_inserted(&self) -> u64 { self.total_inserted.load(Ordering::Relaxed) }
    pub fn total_rejected(&self) -> u64 { self.total_rejected.load(Ordering::Relaxed) }

    // ========================================
    // RawLog Operations
    // ========================================

    /// Insert a raw conversation log entry.
    ///
    /// If `rawlog_key` is provided, content is encrypted with XChaCha20-Poly1305
    /// before storage. The `encrypted` flag is set accordingly.
    pub async fn insert_raw_log(
        &self,
        session_id: &str,
        turn_index: i64,
        role: &str,
        content: &str,
        source_ai: &str,
        recall_context: Option<&str>,
        extractable: i64,
        feedback_signal: Option<i64>,
        rawlog_key: Option<&[u8; 32]>,
    ) -> Result<i64, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let (stored_content, encrypted_flag): (Vec<u8>, i64) = match rawlog_key {
            Some(key) => {
                match encrypt_rawlog_content(key, content.as_bytes()) {
                    Ok(ciphertext) => (ciphertext, 1),
                    Err(e) => {
                        warn!("[STORAGE] RawLog encryption failed, storing plaintext: {}", e);
                        (content.as_bytes().to_vec(), 0)
                    }
                }
            }
            None => (content.as_bytes().to_vec(), 0),
        };

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO raw_logs (session_id, turn_index, role, content, source_ai,
                recall_context, extractable, feedback_signal, encrypted, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                session_id, turn_index, role, stored_content,
                source_ai, recall_context, extractable,
                feedback_signal, encrypted_flag, now,
            ],
        ).map_err(|e| format!("RawLog insert: {}", e))?;

        let log_id = conn.last_insert_rowid();
        Ok(log_id)
    }

    /// Read and decrypt a raw log entry's content.
    pub async fn read_rawlog_content(
        &self,
        log_id: i64,
        rawlog_key: Option<&[u8; 32]>,
    ) -> Option<String> {
        let conn = self.conn.lock().await;
        let row: Option<(Vec<u8>, i64)> = conn.query_row(
            "SELECT content, encrypted FROM raw_logs WHERE log_id = ?1",
            params![log_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional().ok()?;

        let (content_bytes, encrypted) = row?;

        if encrypted == 0 {
            String::from_utf8(content_bytes).ok()
        } else if let Some(key) = rawlog_key {
            decrypt_rawlog_content(key, &content_bytes).ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
        } else {
            warn!("[STORAGE] Encrypted rawlog but no key provided");
            None
        }
    }

    /// Update feedback_signal for a raw_log entry.
    pub async fn update_rawlog_feedback(&self, log_id: i64, signal: i64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE raw_logs SET feedback_signal = ?1 WHERE log_id = ?2",
            params![signal, log_id],
        );
    }

    /// Increment positive_feedback on a record.
    pub async fn increment_positive_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET positive_feedback = positive_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        );
    }

    /// Increment negative_feedback on a record.
    pub async fn increment_negative_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET negative_feedback = negative_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        );
    }

    /// Insert a feedback event for SGD training.
    pub async fn insert_feedback(
        &self,
        owner: &[u8; 32],
        memory_id: &[u8; 32],
        session_id: &str,
        turn_index: i64,
        signal: i64,
        features: Option<&[f32; 9]>,
        prediction: Option<f32>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let features_blob: Option<Vec<u8>> = features.map(|f| {
            f.iter().flat_map(|v| v.to_le_bytes()).collect()
        });

        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO memory_feedback (owner, memory_id, session_id, turn_index,
                signal, features, prediction, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                owner.as_slice(), memory_id.as_slice(), session_id,
                turn_index, signal, features_blob.as_deref(),
                prediction, now,
            ],
        );
    }

    /// Query unprocessed raw_logs for Miner feedback detection.
    pub async fn get_unprocessed_rawlogs(
        &self,
        limit: usize,
    ) -> Vec<RawLogRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT log_id, session_id, turn_index, role, content, encrypted,
                    recall_context, extractable, feedback_signal
             FROM raw_logs
             WHERE feedback_signal IS NULL
             ORDER BY session_id, turn_index
             LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![limit as i64], |row| {
            Ok(RawLogRow {
                log_id: row.get(0)?,
                session_id: row.get(1)?,
                turn_index: row.get(2)?,
                role: row.get(3)?,
                content: row.get(4)?,
                encrypted: row.get(5)?,
                recall_context: row.get(6)?,
                extractable: row.get(7)?,
                feedback_signal: row.get(8)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Query records that need embedding backfill.
    pub async fn get_records_needing_embedding(&self, limit: usize) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        Self::query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count
             FROM records
             WHERE embedding IS NULL AND status = 0
             LIMIT ?1",
            params![limit as i64],
        )
    }

    /// Load MVF user weights from database.
    pub async fn load_user_weights(&self, owner: &[u8; 32]) -> Option<Vec<u8>> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT weights FROM user_weights WHERE owner = ?1",
            params![owner.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        ).optional().ok()?
    }

    /// Save MVF user weights to database.
    pub async fn save_user_weights(&self, owner: &[u8; 32], weights_blob: &[u8], version: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO user_weights (owner, weights, version, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(owner) DO UPDATE SET weights=?2, version=?3, updated_at=?4",
            params![owner.as_slice(), weights_blob, version as i64, now],
        );
    }

    /// Mark a record as superseded by another.
    pub async fn supersede_record(&self, old_id: &[u8; 32], new_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        let r1 = conn.execute(
            "UPDATE records SET status = 1 WHERE record_id = ?1",
            params![old_id.as_slice()],
        );
        let r2 = conn.execute(
            "UPDATE records SET supersedes = ?1 WHERE record_id = ?2",
            params![old_id.as_slice(), new_id.as_slice()],
        );
        r1.is_ok() && r2.is_ok()
    }

    /// Get records with _correction tag for Miner Step 0.6.
    pub async fn get_correction_records(&self) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        Self::query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count
             FROM records
             WHERE topic_tags LIKE '%_correction%' AND status = 0",
            [],
        )
    }

    /// Update topic_tags for a record (remove _correction marker).
    pub async fn update_topic_tags(&self, record_id: &[u8; 32], tags: &[String]) {
        let json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2",
            params![json, record_id.as_slice()],
        );
    }

    /// Get recent feedback events for baseline calculation.
    pub async fn get_recent_feedback(&self, limit: usize) -> Vec<(i64, f32)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT signal, prediction FROM memory_feedback
             ORDER BY created_at DESC LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1).unwrap_or(0.0)))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    // ========================================
    // Private helpers
    // ========================================

    /// 通用查询辅助：执行 SQL 返回 Vec<MemoryRecord>
    fn query_rows(conn: &Connection, sql: &str, p: impl rusqlite::Params) -> Vec<MemoryRecord> {
        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(e) => { error!(error=%e, "[STORAGE] Prepare failed"); return Vec::new(); }
        };
        stmt.query_map(p, |row| Self::row_to_record(row))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// SQLite row → MemoryRecord
    fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
        let record_id_blob: Vec<u8> = row.get(0)?;
        let owner_blob: Vec<u8> = row.get(1)?;
        let timestamp: i64 = row.get(2)?;
        let layer_val: i64 = row.get(3)?;
        let tags_json: String = row.get(4)?;
        let source_ai: String = row.get(5)?;
        let status_val: i64 = row.get(6)?;
        let supersedes_blob: Option<Vec<u8>> = row.get(7)?;
        let encrypted_content: Vec<u8> = row.get(8)?;
        let embedding_blob: Option<Vec<u8>> = row.get(9)?;
        let signature_blob: Vec<u8> = row.get(10)?;
        let access_count: i64 = row.get(11)?;

        let mut record_id = [0u8; 32];
        if record_id_blob.len() == 32 { record_id.copy_from_slice(&record_id_blob); }

        let mut owner = [0u8; 32];
        if owner_blob.len() == 32 { owner.copy_from_slice(&owner_blob); }

        let mut signature = [0u8; 64];
        if signature_blob.len() == 64 { signature.copy_from_slice(&signature_blob); }

        let supersedes = supersedes_blob.and_then(|b| {
            if b.len() == 32 { let mut a = [0u8; 32]; a.copy_from_slice(&b); Some(a) }
            else { None }
        });

        let embedding = embedding_blob
            .map(|b| bytes_to_embedding(&b))
            .unwrap_or_default();

        Ok(MemoryRecord {
            record_id,
            owner,
            timestamp: timestamp as u64,
            layer: MemoryLayer::from_u8(layer_val as u8).unwrap_or(MemoryLayer::Episode),
            topic_tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            source_ai,
            status: RecordStatus::from_u8(status_val as u8).unwrap_or(RecordStatus::Active),
            supersedes,
            encrypted_content,
            embedding,
            signature,
            access_count: access_count as u32,
        })
    }
}

impl std::fmt::Debug for MemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStorage")
            .field("inserted", &self.total_inserted())
            .field("rejected", &self.total_rejected())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rec(ts: u64, layer: MemoryLayer, src: &str) -> MemoryRecord {
        MemoryRecord::new(
            [0xAA; 32], ts, layer,
            vec!["test".into()], src.into(),
            b"encrypted_data".to_vec(),
            vec![0.1, 0.2, 0.3],
        )
    }

    fn make_rec_owner(ts: u64, owner: [u8; 32], layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new(
            owner, ts, layer,
            vec!["test".into()], "ai".into(),
            format!("content_{}", ts).into_bytes(),
            vec![0.5; 4],
        )
    }

    #[tokio::test]
    async fn test_open_in_memory() {
        let s = MemoryStorage::open(":memory:").unwrap();
        assert_eq!(s.count().await, 0);
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        let id = r.record_id;

        assert!(s.insert(&r, "minilm").await);
        assert!(!s.insert(&r, "minilm").await, "duplicate");

        let got = s.get(&id).await.unwrap();
        assert_eq!(got.source_ai, "test");
        assert_eq!(got.layer, MemoryLayer::Episode);
        assert_eq!(got.embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(s.count().await, 1);
    }

    #[tokio::test]
    async fn test_reject_invalid_hash() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let mut r = make_rec(100, MemoryLayer::Episode, "t");
        r.record_id = [0xFF; 32];
        assert!(!s.insert(&r, "m").await);
        assert_eq!(s.total_rejected(), 1);
    }

    #[tokio::test]
    async fn test_get_active_records() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let o = [0xAA; 32];
        s.insert(&make_rec_owner(100, o, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, o, MemoryLayer::Knowledge), "m").await;
        s.insert(&make_rec_owner(300, o, MemoryLayer::Archive), "m").await;

        let all = s.get_active_records(&o, None, 100).await;
        assert_eq!(all.len(), 3);

        let ep = s.get_active_records(&o, Some(MemoryLayer::Episode), 100).await;
        assert_eq!(ep.len(), 1);

        let ar = s.get_active_records(&o, Some(MemoryLayer::Archive), 100).await;
        assert_eq!(ar.len(), 1);
    }

    #[tokio::test]
    async fn test_revoke() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "t");
        let id = r.record_id;
        s.insert(&r, "m").await;
        assert!(s.revoke(&id).await);

        let got = s.get(&id).await.unwrap();
        assert_eq!(got.status, RecordStatus::Revoked);
        assert!(got.encrypted_content.is_empty());
        assert!(got.embedding.is_empty());
    }

    #[tokio::test]
    async fn test_compact_episodes_to_archive() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let o = [0xAA; 32];
        s.insert(&make_rec_owner(100, o, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, o, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(300, o, MemoryLayer::Knowledge), "m").await;

        let compacted = s.compact_episodes_to_archive(&o, 10).await;
        assert_eq!(compacted.len(), 2);

        // 压实后 Episode 数为 0，Archive 数为 2
        assert_eq!(s.count_by_layer(MemoryLayer::Episode).await, 0);
        assert_eq!(s.count_by_layer(MemoryLayer::Archive).await, 2);

        // 但它们仍然是 Active 的！（可被低权重召回）
        let active = s.get_active_records(&o, Some(MemoryLayer::Archive), 100).await;
        assert_eq!(active.len(), 2);

        // Knowledge 不受影响
        assert_eq!(s.count_by_layer(MemoryLayer::Knowledge).await, 1);
    }

    #[tokio::test]
    async fn test_increment_access() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "t");
        let id = r.record_id;
        s.insert(&r, "m").await;
        s.increment_access(&id).await;
        s.increment_access(&id).await;
        let got = s.get(&id).await.unwrap();
        assert_eq!(got.access_count, 2);
    }

    #[tokio::test]
    async fn test_chain_state() {
        let s = MemoryStorage::open(":memory:").unwrap();
        assert_eq!(s.last_block_hash().await, [0u8; 32]);
        s.set_chain_state(&[0xBB; 32], 42).await;
        assert_eq!(s.last_block_hash().await, [0xBB; 32]);
        assert_eq!(s.last_block_height().await, 42);
    }

    #[tokio::test]
    async fn test_stats() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let o = [0xAA; 32];
        s.insert(&make_rec_owner(100, o, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, o, MemoryLayer::Identity), "m").await;

        let st = s.stats().await;
        assert_eq!(st.total_records, 2);
        assert_eq!(st.active_records, 2);
        assert_eq!(st.by_layer.episode, 1);
        assert_eq!(st.by_layer.identity, 1);
        assert_eq!(st.records_with_embedding, 2);
    }

    #[tokio::test]
    async fn test_embedding_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.14159, 0.0, f32::MIN, f32::MAX];
        let bytes = embedding_to_bytes(&original);
        let restored = bytes_to_embedding(&bytes);
        assert_eq!(original, restored);
    }

    #[tokio::test]
    async fn test_no_embedding_stored_as_null() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let r = MemoryRecord::new(
            [0xAA; 32], 100, MemoryLayer::Episode,
            vec![], "t".into(), b"c".to_vec(),
            vec![], // 空 embedding
        );
        s.insert(&r).await;
        let got = s.get(&r.record_id).await.unwrap();
        assert!(got.embedding.is_empty());
        assert!(!got.has_embedding());
    }

    #[tokio::test]
    async fn test_get_records_with_embedding() {
        let s = MemoryStorage::open(":memory:").unwrap();
        let o = [0xAA; 32];

        // 有 embedding
        s.insert(&make_rec_owner(100, o, MemoryLayer::Episode)).await;
        // 无 embedding
        let mut no_emb = make_rec_owner(200, o, MemoryLayer::Knowledge);
        no_emb.embedding = vec![];
        // 需要重新计算 record_id 因为改了字段…但 embedding 不在哈希里，所以不需要
        s.insert(&no_emb).await;

        let with = s.get_records_with_embedding(&o).await;
        assert_eq!(with.len(), 1);
        assert_eq!(with[0].timestamp, 100);
    }
}
