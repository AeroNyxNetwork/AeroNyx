// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage.rs
// ============================================
//! # MemoryStorage — SQLite Core (Schema, CRUD, LRU Cache)
//!
//! ## Creation Reason
//!
//! ## Split Structure (v2.2.0)
//! storage.rs was split into 3 files for maintainability:
//! - `storage.rs` (THIS FILE) — struct, open, schema, migration, core CRUD, LRU
//! - `storage_crypto.rs` — all encryption/decryption functions
//! - `storage_ops.rs` — rawlog, feedback, chain state, stats, miner, overview
//!
//! All 3 files impl on the same `MemoryStorage` struct.
//! External API is unchanged — mod.rs re-exports everything.
//!
//! ## Schema History
//! - v1: Initial schema (records only)
//! - v2: Added embedding column to records
//! - v4: Added positive_feedback, negative_feedback, conflict_with to records
//!        Added memory_edges, user_weights, memory_feedback, chain_state tables
//! - v5 (v2.4.0-GraphCognition): Three-layer cognitive graph schema:
//!   - episodes: Episode layer (complete original conversations, non-lossy)
//!   - entities: Semantic Entity layer nodes (GLiNER-extracted)
//!   - knowledge_edges: Semantic Entity layer edges (with temporal validity)
//!   - episode_edges: Bridge layer (Episode ↔ Entity bidirectional links)
//!   - communities: Community layer (label propagation auto-clustering)
//!   - projects: Project table (Community specialization for code projects)
//!   - sessions: Conversation session metadata + summaries
//!   - artifacts: Code/document artifacts with version chains
//!   - records ALTER: added project_id, session_id, episode_id columns
//!   - memory_edges data migrated to knowledge_edges (relation_type = 'RELATED_TO')
//!
//! ## Thread Safety
//! `rusqlite::Connection` behind `tokio::sync::Mutex`. Phase 2+ can use r2d2 pooling.
//!
//! ⚠️ Important Note for Next Developer:
//! - Schema migrations: use `ALTER TABLE` in `maybe_migrate()`, NEVER drop tables.
//! - `record_id` is PRIMARY KEY. Duplicate inserts use `INSERT OR IGNORE`.
//! - `embedding` stored as raw f32 LE bytes. 384-dim = 1536 bytes.
//! - `query_rows()` is `&self` — it needs `self.record_key` for decryption.
//! - LRU cache stores PLAINTEXT records (decrypted) for fast reads.
//! - v4 migration is additive-only (ALTER TABLE ADD COLUMN).
//! - v5 migration creates 8 new tables, ALTERs records, migrates memory_edges data.
//!   All additive — no data loss on upgrade. Rollback: new tables are simply ignored
//!   by v2.3.0 code (it doesn't query them).
//! - Rawlog key migration clears old raw_logs on first run after key fix.
//! - episodes.encrypted_content uses same ChaCha20 encryption as records.encrypted_content.
//! - knowledge_edges.valid_until = NULL means "currently valid".
//!   Query pattern: WHERE valid_until IS NULL (current state).
//!
//! ## Last Modified
//! v1.0.0 - Initial SQLite storage engine
//! v2.1.0 - 4-layer, plaintext embedding BLOB, compaction via layer change
//! v2.1.0+MVF - Schema v4, feedback columns, content dedup
//! v2.1.0+MVF+Encryption - Record encryption, rawlog key fix
//! v2.2.0 - 🌟 Split into storage.rs + storage_crypto.rs + storage_ops.rs
//! v2.4.0-GraphCognition - 🌟 Schema v5: Three-layer cognitive graph (8 new tables,
//!   records ALTER, memory_edges migration)

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord, RecordStatus};

use super::storage_crypto::{encrypt_record_content, decrypt_record_content};

// ============================================
// Constants
// ============================================

/// Current schema version.
/// v4 → v5 migration adds cognitive graph tables.
const SCHEMA_VERSION: u32 = 5;

const LRU_CACHE_CAPACITY: usize = 1000;
const DEFAULT_PAGE_SIZE: usize = 100;

// ============================================
// Embedding helpers
// ============================================

pub(crate) fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub(crate) fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        f32::from_le_bytes(buf)
    }).collect()
}

// ============================================
// StorageStats / LayerCounts
// ============================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct StorageStats {
    pub total_records: u64,
    pub active_records: u64,
    pub by_layer: LayerCounts,
    pub content_bytes: u64,
    pub records_with_embedding: u64,
    pub session_inserts: u64,
    pub session_rejects: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LayerCounts {
    pub identity: u64,
    pub knowledge: u64,
    pub episode: u64,
    pub archive: u64,
}

// ============================================
// RawLogRow
// ============================================

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

// ============================================
// LRU Cache
// ============================================

pub(crate) struct LruCache {
    map: HashMap<[u8; 32], (usize, MemoryRecord)>,
    order_counter: usize,
    capacity: usize,
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self { map: HashMap::with_capacity(capacity), order_counter: 0, capacity }
    }

    pub fn get(&mut self, id: &[u8; 32]) -> Option<&MemoryRecord> {
        if let Some(entry) = self.map.get_mut(id) {
            self.order_counter += 1;
            entry.0 = self.order_counter;
            Some(&entry.1)
        } else {
            None
        }
    }

    pub fn put(&mut self, record: MemoryRecord) {
        let id = record.record_id;
        self.order_counter += 1;
        self.map.insert(id, (self.order_counter, record));
        if self.map.len() > self.capacity {
            if let Some((&evict_id, _)) = self.map.iter().min_by_key(|(_, (ord, _))| *ord) {
                self.map.remove(&evict_id);
            }
        }
    }

    pub fn invalidate(&mut self, id: &[u8; 32]) {
        self.map.remove(id);
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order_counter = 0;
    }
}

// ============================================
// MemoryStorage
// ============================================

pub struct MemoryStorage {
    pub(crate) conn: TokioMutex<Connection>,
    pub(crate) total_inserted: AtomicU64,
    pub(crate) total_rejected: AtomicU64,
    pub(crate) cache: RwLock<LruCache>,
    pub(crate) record_key: Option<[u8; 32]>,
}

impl MemoryStorage {
    pub fn open(path: impl AsRef<Path>, record_key: Option<[u8; 32]>) -> Result<Self, String> {
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

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -8000;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;"
        ).map_err(|e| format!("Failed to set pragmas: {}", e))?;

        Self::create_schema(&conn)?;
        Self::maybe_migrate(&conn)?;

        let mode = if record_key.is_some() { "encrypted" } else { "plaintext" };
        info!(path = %path.display(), mode = mode, "[STORAGE] ✅ SQLite opened (schema v{})", SCHEMA_VERSION);

        Ok(Self {
            conn: TokioMutex::new(conn),
            total_inserted: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            cache: RwLock::new(LruCache::new(LRU_CACHE_CAPACITY)),
            record_key,
        })
    }

    fn create_schema(conn: &Connection) -> Result<(), String> {
        // ── v1-v4: Original tables (records, raw_logs, memory_edges, etc.) ──
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS records (
                record_id           BLOB PRIMARY KEY,
                owner               BLOB NOT NULL,
                timestamp           INTEGER NOT NULL,
                layer               INTEGER NOT NULL,
                topic_tags          TEXT NOT NULL DEFAULT '[]',
                source_ai           TEXT NOT NULL DEFAULT '',
                status              INTEGER NOT NULL DEFAULT 0,
                supersedes          BLOB,
                encrypted_content   BLOB NOT NULL DEFAULT x'',
                embedding           BLOB,
                embedding_model     TEXT NOT NULL DEFAULT '',
                embedding_dim       INTEGER NOT NULL DEFAULT 0,
                signature           BLOB NOT NULL,
                access_count        INTEGER NOT NULL DEFAULT 0,
                created_at          INTEGER NOT NULL,
                archived_at         INTEGER,
                positive_feedback   INTEGER NOT NULL DEFAULT 0,
                negative_feedback   INTEGER NOT NULL DEFAULT 0,
                conflict_with       BLOB,
                project_id          TEXT,
                session_id          TEXT,
                episode_id          TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_owner ON records(owner);
            CREATE INDEX IF NOT EXISTS idx_owner_layer_status ON records(owner, layer, status);
            CREATE INDEX IF NOT EXISTS idx_status_layer ON records(status, layer);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_records_project ON records(project_id, timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
            CREATE INDEX IF NOT EXISTS idx_records_episode ON records(episode_id);

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
            CREATE INDEX IF NOT EXISTS idx_rawlogs_session ON raw_logs(session_id, turn_index);
            CREATE INDEX IF NOT EXISTS idx_rawlogs_feedback ON raw_logs(feedback_signal);

            CREATE TABLE IF NOT EXISTS memory_edges (
                source_id BLOB NOT NULL, target_id BLOB NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'co_occurred',
                weight REAL NOT NULL DEFAULT 1.0, created_at INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);

            CREATE TABLE IF NOT EXISTS user_weights (
                owner BLOB PRIMARY KEY, weights BLOB NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner BLOB NOT NULL, memory_id BLOB NOT NULL,
                session_id TEXT NOT NULL, turn_index INTEGER NOT NULL,
                signal INTEGER NOT NULL, features BLOB, prediction REAL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_owner ON memory_feedback(owner);
            CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);

            CREATE TABLE IF NOT EXISTS chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);"
        ).map_err(|e| format!("Schema creation failed (base tables): {}", e))?;

        // ── v5 (v2.4.0): Three-layer cognitive graph tables ──
        conn.execute_batch(
            "-- Episode layer: complete original conversations (non-lossy)
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                episode_type        TEXT NOT NULL,
                source              TEXT NOT NULL,
                session_id          TEXT,
                encrypted_content   BLOB NOT NULL,
                content_hash        TEXT NOT NULL,
                embedding           BLOB,
                token_count         INTEGER,
                created_at          INTEGER NOT NULL,
                ingested_at         INTEGER NOT NULL,
                metadata_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_owner ON episodes(owner, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(owner, episode_type);

            -- Semantic Entity layer: GLiNER-extracted entity nodes
            CREATE TABLE IF NOT EXISTS entities (
                entity_id           TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                name_normalized     TEXT NOT NULL,
                entity_type         TEXT NOT NULL,
                description         TEXT,
                embedding           BLOB,
                community_id        TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL,
                mention_count       INTEGER DEFAULT 1,
                metadata_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entities_owner ON entities(owner);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(owner, entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(owner, name_normalized);
            CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(owner, community_id);

            -- Semantic Entity layer: temporal knowledge edges
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                edge_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                owner               BLOB NOT NULL,
                source_id           TEXT NOT NULL,
                target_id           TEXT NOT NULL,
                relation_type       TEXT NOT NULL,
                fact_text           TEXT,
                weight              REAL DEFAULT 1.0,
                confidence          REAL DEFAULT 1.0,
                embedding           BLOB,
                valid_from          INTEGER NOT NULL,
                valid_until         INTEGER,
                episode_id          TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kedges_source ON knowledge_edges(owner, source_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_kedges_target ON knowledge_edges(owner, target_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_kedges_valid ON knowledge_edges(owner, valid_until);
            CREATE INDEX IF NOT EXISTS idx_kedges_episode ON knowledge_edges(episode_id);

            -- Bridge layer: Episode ↔ Entity bidirectional links
            CREATE TABLE IF NOT EXISTS episode_edges (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                owner               BLOB NOT NULL,
                episode_id          TEXT NOT NULL,
                entity_id           TEXT NOT NULL,
                role                TEXT NOT NULL,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ep_edges_episode ON episode_edges(episode_id);
            CREATE INDEX IF NOT EXISTS idx_ep_edges_entity ON episode_edges(entity_id);

            -- Community layer: auto-clustering via label propagation
            CREATE TABLE IF NOT EXISTS communities (
                community_id        TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                summary             TEXT,
                description         TEXT,
                entity_count        INTEGER DEFAULT 0,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_communities_owner ON communities(owner);

            -- Projects: Community specialization for code projects
            CREATE TABLE IF NOT EXISTS projects (
                project_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                status              TEXT DEFAULT 'active',
                community_id        TEXT NOT NULL,
                summary             TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL,
                last_active_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner, status);
            CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(owner, last_active_at DESC);

            -- Sessions: conversation session metadata
            CREATE TABLE IF NOT EXISTS sessions (
                session_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                project_id          TEXT,
                session_type        TEXT DEFAULT 'chat',
                started_at          INTEGER NOT NULL,
                ended_at            INTEGER,
                turn_count          INTEGER DEFAULT 0,
                summary             TEXT,
                key_decisions       TEXT,
                files_touched       TEXT,
                entities_extracted  INTEGER DEFAULT 0,
                summary_generated   INTEGER DEFAULT 0,
                artifacts_extracted INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_pending ON sessions(entities_extracted, summary_generated);

            -- Artifacts: code/document artifacts with version chains
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id         TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                session_id          TEXT NOT NULL,
                project_id          TEXT,
                artifact_type       TEXT NOT NULL,
                filename            TEXT,
                language            TEXT,
                version             INTEGER DEFAULT 1,
                parent_id           TEXT,
                encrypted_content   BLOB NOT NULL,
                content_hash        TEXT NOT NULL,
                embedding           BLOB,
                line_count          INTEGER,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id, filename, version DESC);
            CREATE INDEX IF NOT EXISTS idx_artifacts_file ON artifacts(owner, filename, version DESC);"
        ).map_err(|e| format!("Schema creation failed (v5 cognitive graph tables): {}", e))?;

        // Insert schema version if not present
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

    fn maybe_migrate(conn: &Connection) -> Result<(), String> {
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
            .unwrap_or(1);

        // v1 → v2: embedding column
        if current < 2 {
            info!("[STORAGE] Migrating schema v{} → v2", current);
            let has_embedding: bool = conn.prepare("SELECT embedding FROM records LIMIT 0").is_ok();
            if !has_embedding {
                let _ = conn.execute_batch("ALTER TABLE records ADD COLUMN embedding BLOB;");
                info!("[STORAGE] Added `embedding` column");
            }
            conn.execute("UPDATE schema_version SET version = 2", [])
                .map_err(|e| format!("Update schema version to v2: {}", e))?;
            info!("[STORAGE] ✅ Migration to v2 complete");
        }

        // v2 → v4: MVF feedback + conflict
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
            .unwrap_or(2);

        if current < 4 {
            info!("[STORAGE] Migrating schema v{} → v4", current);

            if !conn.prepare("SELECT positive_feedback FROM records LIMIT 0").is_ok() {
                conn.execute_batch("ALTER TABLE records ADD COLUMN positive_feedback INTEGER NOT NULL DEFAULT 0;")
                    .map_err(|e| format!("Add positive_feedback: {}", e))?;
            }
            if !conn.prepare("SELECT negative_feedback FROM records LIMIT 0").is_ok() {
                conn.execute_batch("ALTER TABLE records ADD COLUMN negative_feedback INTEGER NOT NULL DEFAULT 0;")
                    .map_err(|e| format!("Add negative_feedback: {}", e))?;
            }
            if !conn.prepare("SELECT conflict_with FROM records LIMIT 0").is_ok() {
                conn.execute_batch("ALTER TABLE records ADD COLUMN conflict_with BLOB;")
                    .map_err(|e| format!("Add conflict_with: {}", e))?;
            }

            conn.execute("UPDATE schema_version SET version = 4", [])
                .map_err(|e| format!("Update schema version to v4: {}", e))?;
            info!("[STORAGE] ✅ Migration to v4 complete");
        }

        // v4 → v5 (v2.4.0-GraphCognition): Three-layer cognitive graph
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0))
            .unwrap_or(4);

        if current < 5 {
            info!("[STORAGE] Migrating schema v{} → v5 (cognitive graph)", current);

            // 5a: ALTER records — add project_id, session_id, episode_id
            //     These link existing records to the new cognitive graph.
            if !conn.prepare("SELECT project_id FROM records LIMIT 0").is_ok() {
                conn.execute_batch("ALTER TABLE records ADD COLUMN project_id TEXT;")
                    .map_err(|e| format!("Add project_id to records: {}", e))?;
                info!("[STORAGE] Added records.project_id");
            }
            if !conn.prepare("SELECT session_id FROM records LIMIT 0").is_ok() {
                conn.execute_batch("ALTER TABLE records ADD COLUMN session_id TEXT;")
                    .map_err(|e| format!("Add session_id to records: {}", e))?;
                info!("[STORAGE] Added records.session_id");
            }
            if !conn.prepare("SELECT episode_id FROM records LIMIT 0").is_ok() {
                conn.execute_batch("ALTER TABLE records ADD COLUMN episode_id TEXT;")
                    .map_err(|e| format!("Add episode_id to records: {}", e))?;
                info!("[STORAGE] Added records.episode_id");
            }

            // 5b: Create indexes for new records columns
            let _ = conn.execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_records_project ON records(project_id, timestamp DESC);
                 CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
                 CREATE INDEX IF NOT EXISTS idx_records_episode ON records(episode_id);"
            );

            // 5c: Create v5 tables (IF NOT EXISTS — safe if create_schema already ran)
            // This handles the case where the DB file is new (create_schema creates them)
            // vs upgrade from v4 (create_schema may not have created them if the file
            // already existed before the code update).
            //
            // Note: The full CREATE TABLE statements are in create_schema().
            // Here we just ensure they exist for the migration path where
            // create_schema() ran on the OLD code without these tables.
            let v5_tables = [
                "episodes", "entities", "knowledge_edges", "episode_edges",
                "communities", "projects", "sessions", "artifacts",
            ];
            for table in &v5_tables {
                let exists: bool = conn.query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![table],
                    |row| row.get::<_, i64>(0),
                ).unwrap_or(0) > 0;

                if !exists {
                    warn!(
                        "[STORAGE] v5 table '{}' missing after create_schema — \
                         this is expected when upgrading from v2.3.0. \
                         Table will be created by create_schema on next restart.",
                        table
                    );
                    // The tables are created by create_schema() which runs BEFORE
                    // maybe_migrate(). If we get here, something is unusual.
                    // We don't re-create them here to avoid duplicating the
                    // CREATE TABLE SQL. The next server restart will fix it.
                }
            }

            // 5d: Migrate memory_edges → knowledge_edges
            //     Existing co-occurrence edges become RELATED_TO knowledge edges.
            //     This preserves the graph structure from v2.3.0.
            {
                let migrated: bool = conn.query_row(
                    "SELECT value FROM chain_state WHERE key = 'memory_edges_migrated_v5'",
                    [], |row| { let v: Vec<u8> = row.get(0)?; Ok(v == b"1") },
                ).unwrap_or(false);

                if !migrated {
                    let edge_count: i64 = conn.query_row(
                        "SELECT COUNT(*) FROM memory_edges", [], |row| row.get(0),
                    ).unwrap_or(0);

                    if edge_count > 0 {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64;

                        // Check if knowledge_edges table exists before attempting migration
                        let ke_exists: bool = conn.query_row(
                            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='knowledge_edges'",
                            [],
                            |row| row.get::<_, i64>(0),
                        ).unwrap_or(0) > 0;

                        if ke_exists {
                            match conn.execute(
                                "INSERT OR IGNORE INTO knowledge_edges
                                    (owner, source_id, target_id, relation_type, weight,
                                     confidence, valid_from, created_at, updated_at)
                                 SELECT
                                    source_id, hex(source_id), hex(target_id), 'RELATED_TO', weight,
                                    1.0, created_at, ?1, ?1
                                 FROM memory_edges",
                                params![now],
                            ) {
                                Ok(migrated_count) => {
                                    info!(
                                        count = migrated_count,
                                        "[STORAGE] Migrated memory_edges → knowledge_edges"
                                    );
                                }
                                Err(e) => {
                                    warn!(
                                        error = %e,
                                        "[STORAGE] memory_edges migration failed (non-fatal, edges preserved)"
                                    );
                                }
                            }
                        } else {
                            warn!("[STORAGE] knowledge_edges table not found, skipping memory_edges migration");
                        }
                    }

                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('memory_edges_migrated_v5', ?1)",
                        params![b"1".as_slice()],
                    );
                    info!("[STORAGE] ✅ memory_edges migration marker set");
                }
            }

            // 5e: Update schema version to v5
            conn.execute("UPDATE schema_version SET version = ?1", params![SCHEMA_VERSION])
                .map_err(|e| format!("Update schema version to v5: {}", e))?;
            info!("[STORAGE] ✅ Migration to v5 (cognitive graph) complete");
        }

        // Rawlog key migration: clear old raw_logs encrypted with public key
        {
            let migrated: bool = conn.query_row(
                "SELECT value FROM chain_state WHERE key = 'rawlog_key_migrated'",
                [], |row| { let v: Vec<u8> = row.get(0)?; Ok(v == b"1") },
            ).unwrap_or(false);

            if !migrated {
                let count: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM raw_logs", [], |row| row.get(0),
                ).unwrap_or(0);

                if count > 0 {
                    match conn.execute("DELETE FROM raw_logs", []) {
                        Ok(deleted) => info!(deleted = deleted,
                            "[STORAGE] 🔑 Cleared old raw_logs (rawlog key migrated to private key)"),
                        Err(e) => warn!(error = %e, "[STORAGE] Failed to clear old raw_logs"),
                    }
                    let _ = conn.execute("DELETE FROM sqlite_sequence WHERE name = 'raw_logs'", []);
                }

                let _ = conn.execute(
                    "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('rawlog_key_migrated', ?1)",
                    params![b"1".as_slice()],
                );
                info!("[STORAGE] ✅ Rawlog key migration marker set");
            }
        }

        Ok(())
    }

    // ========================================
    // Insert
    // ========================================

    pub async fn insert(&self, record: &MemoryRecord, embedding_model: &str) -> bool {
        if !record.verify_id() {
            warn!(record_id = hex::encode(record.record_id), "[STORAGE] ❌ Rejected: hash mismatch");
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let tags_json = serde_json::to_string(&record.topic_tags).unwrap_or_else(|_| "[]".to_string());
        let embedding_blob: Option<Vec<u8>> = if record.has_embedding() {
            Some(embedding_to_bytes(&record.embedding))
        } else { None };
        let embedding_dim = record.embedding_dim() as i64;
        let conflict_with_blob: Option<Vec<u8>> = record.conflict_with.map(|c| c.to_vec());

        let stored_content: Vec<u8> = if let Some(ref key) = self.record_key {
            if record.encrypted_content.is_empty() {
                record.encrypted_content.clone()
            } else {
                match encrypt_record_content(key, &record.encrypted_content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        warn!("[STORAGE] Record encryption failed, storing plaintext: {}", e);
                        record.encrypted_content.clone()
                    }
                }
            }
        } else {
            record.encrypted_content.clone()
        };

        let conn = self.conn.lock().await;
        let result = conn.execute(
            "INSERT OR IGNORE INTO records (
                record_id, owner, timestamp, layer, topic_tags, source_ai,
                status, supersedes, encrypted_content, embedding,
                embedding_model, embedding_dim, signature, access_count, created_at,
                positive_feedback, negative_feedback, conflict_with
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18)",
            params![
                record.record_id.as_slice(), record.owner.as_slice(),
                record.timestamp as i64, record.layer as u8 as i64,
                tags_json, record.source_ai,
                record.status as u8 as i64,
                record.supersedes.as_ref().map(|s| s.as_slice()),
                stored_content.as_slice(), embedding_blob.as_deref(),
                embedding_model, embedding_dim,
                record.signature.as_slice(), record.access_count as i64, now,
                record.positive_feedback as i64, record.negative_feedback as i64,
                conflict_with_blob.as_deref(),
            ],
        );

        match result {
            Ok(changes) if changes > 0 => {
                self.total_inserted.fetch_add(1, Ordering::Relaxed);
                self.cache.write().put(record.clone());
                debug!(record_id = hex::encode(record.record_id), layer = %record.layer, "[STORAGE] ✅ Inserted");
                true
            }
            Ok(_) => { debug!(record_id = hex::encode(record.record_id), "[STORAGE] Duplicate, skipped"); false }
            Err(e) => {
                error!(record_id = hex::encode(record.record_id), error = %e, "[STORAGE] ❌ Insert failed");
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    // ========================================
    // Query
    // ========================================

    pub async fn get(&self, record_id: &[u8; 32]) -> Option<MemoryRecord> {
        {
            let mut cache = self.cache.write();
            if let Some(record) = cache.get(record_id) {
                return Some(record.clone());
            }
        }
        let conn = self.conn.lock().await;
        let rk = self.record_key;
        let result = conn.query_row(
            Self::SELECT_RECORD_COLS,
            params![record_id.as_slice()],
            |row| Self::row_to_record(row, rk.as_ref()),
        ).optional().unwrap_or_else(|e| { error!(error=%e, "[STORAGE] Query failed"); None });

        if let Some(ref record) = result {
            self.cache.write().put(record.clone());
        }
        result
    }

    pub async fn get_active_records(
        &self, owner: &[u8; 32], layer: Option<MemoryLayer>, limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;
        if let Some(l) = layer {
            self.query_rows(&conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,signature,access_count,
                        positive_feedback,negative_feedback,conflict_with
                 FROM records WHERE owner=?1 AND status=0 AND layer=?2
                 ORDER BY timestamp DESC LIMIT ?3",
                params![owner.as_slice(), l as u8 as i64, limit as i64])
        } else {
            self.query_rows(&conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,signature,access_count,
                        positive_feedback,negative_feedback,conflict_with
                 FROM records WHERE owner=?1 AND status=0
                 ORDER BY timestamp DESC LIMIT ?2",
                params![owner.as_slice(), limit as i64])
        }
    }

    pub async fn query_by_owner_after(&self, owner: &[u8; 32], after_timestamp: u64) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE owner=?1 AND timestamp>?2
             ORDER BY timestamp ASC LIMIT ?3",
            params![owner.as_slice(), after_timestamp as i64, DEFAULT_PAGE_SIZE as i64])
    }

    pub async fn get_records_with_embedding(&self, owner: &[u8; 32]) -> Vec<(MemoryRecord, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,embedding_model
             FROM records WHERE owner=?1 AND status=0 AND embedding IS NOT NULL
             ORDER BY timestamp DESC"
        ) {
            Ok(s) => s,
            Err(e) => { error!(error=%e, "[STORAGE] Prepare failed"); return Vec::new(); }
        };
        let rk = self.record_key;
        stmt.query_map(params![owner.as_slice()], |row| {
            let record = Self::row_to_record(row, rk.as_ref())?;
            let model: String = row.get(15)?;
            Ok((record, model))
        }).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    // ========================================
    // Lifecycle
    // ========================================

    pub async fn update_status(&self, record_id: &[u8; 32], new_status: RecordStatus) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute("UPDATE records SET status=?1 WHERE record_id=?2",
            params![new_status as u8 as i64, record_id.as_slice()]) {
            Ok(n) if n > 0 => { debug!(record_id=hex::encode(record_id), %new_status, "[STORAGE] ✅ Status updated"); true }
            Ok(_) => { warn!(record_id=hex::encode(record_id), "[STORAGE] Not found for status update"); false }
            Err(e) => { error!(error=%e, "[STORAGE] Status update failed"); false }
        }
    }

    pub async fn revoke(&self, record_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1, encrypted_content=x'', embedding=NULL WHERE record_id=?2",
            params![RecordStatus::Revoked as u8 as i64, record_id.as_slice()]) {
            Ok(n) if n > 0 => {
                self.cache.write().invalidate(record_id);
                info!(record_id=hex::encode(record_id), "[STORAGE] 🗑️ Revoked"); true
            }
            Ok(_) => { warn!(record_id=hex::encode(record_id), "[STORAGE] Not found for revoke"); false }
            Err(e) => { error!(error=%e, "[STORAGE] Revoke failed"); false }
        }
    }

    pub async fn increment_access(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET access_count=access_count+1 WHERE record_id=?1",
            params![record_id.as_slice()]);
    }

    /// Acquire the inner SQLite connection lock.
    pub async fn conn_lock(&self) -> tokio::sync::MutexGuard<'_, Connection> {
        self.conn.lock().await
    }

    pub fn total_inserted(&self) -> u64 { self.total_inserted.load(Ordering::Relaxed) }
    pub fn total_rejected(&self) -> u64 { self.total_rejected.load(Ordering::Relaxed) }

    // ========================================
    // Private helpers
    // ========================================

    const SELECT_RECORD_COLS: &'static str =
        "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                status,supersedes,encrypted_content,embedding,signature,access_count,
                positive_feedback,negative_feedback,conflict_with
         FROM records WHERE record_id = ?1";

    pub(crate) fn query_rows(&self, conn: &Connection, sql: &str, p: impl rusqlite::Params) -> Vec<MemoryRecord> {
        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(e) => { error!(error=%e, "[STORAGE] Prepare failed"); return Vec::new(); }
        };
        let rk = self.record_key;
        stmt.query_map(p, |row| Self::row_to_record(row, rk.as_ref()))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn row_to_record(row: &rusqlite::Row<'_>, record_key: Option<&[u8; 32]>) -> rusqlite::Result<MemoryRecord> {
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

        let decrypted_content = if let Some(key) = record_key {
            if encrypted_content.len() >= 28 {
                match decrypt_record_content(key, &encrypted_content) {
                    Ok(plain) => plain,
                    Err(_) => encrypted_content,
                }
            } else { encrypted_content }
        } else { encrypted_content };

        let signature_blob: Vec<u8> = row.get(10)?;
        let access_count: i64 = row.get(11)?;
        let positive_feedback: i64 = row.get(12).unwrap_or(0);
        let negative_feedback: i64 = row.get(13).unwrap_or(0);
        let conflict_with_blob: Option<Vec<u8>> = row.get(14).unwrap_or(None);

        let mut record_id = [0u8; 32];
        if record_id_blob.len() == 32 { record_id.copy_from_slice(&record_id_blob); }
        let mut owner = [0u8; 32];
        if owner_blob.len() == 32 { owner.copy_from_slice(&owner_blob); }
        let mut signature = [0u8; 64];
        if signature_blob.len() == 64 { signature.copy_from_slice(&signature_blob); }

        let supersedes = supersedes_blob.and_then(|b| {
            if b.len() == 32 { let mut a = [0u8; 32]; a.copy_from_slice(&b); Some(a) } else { None }
        });
        let conflict_with = conflict_with_blob.and_then(|b| {
            if b.len() == 32 { let mut a = [0u8; 32]; a.copy_from_slice(&b); Some(a) } else { None }
        });
        let embedding = embedding_blob.map(|b| bytes_to_embedding(&b)).unwrap_or_default();

        Ok(MemoryRecord {
            record_id, owner, timestamp: timestamp as u64,
            layer: MemoryLayer::from_u8(layer_val as u8).unwrap_or(MemoryLayer::Episode),
            topic_tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            source_ai,
            status: RecordStatus::from_u8(status_val as u8).unwrap_or(RecordStatus::Active),
            supersedes, encrypted_content: decrypted_content, embedding, signature,
            access_count: access_count as u32,
            positive_feedback: positive_feedback as u32,
            negative_feedback: negative_feedback as u32,
            conflict_with,
        })
    }
}

impl std::fmt::Debug for MemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStorage")
            .field("inserted", &self.total_inserted())
            .field("rejected", &self.total_rejected())
            .field("encrypted", &self.record_key.is_some())
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
        MemoryRecord::new([0xAA; 32], ts, layer, vec!["test".into()], src.into(),
            b"encrypted_data".to_vec(), vec![0.1, 0.2, 0.3])
    }

    fn make_rec_owner(ts: u64, owner: [u8; 32], layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new(owner, ts, layer, vec!["test".into()], "ai".into(),
            format!("content_{}", ts).into_bytes(), vec![0.5; 4])
    }

    // ========================================
    // Existing tests (preserved from v2.3.0)
    // ========================================

    #[tokio::test]
    async fn test_open_in_memory() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        assert_eq!(s.count().await, 0);
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        let id = r.record_id;
        assert!(s.insert(&r, "minilm").await);
        assert!(!s.insert(&r, "minilm").await);
        let got = s.get(&id).await.unwrap();
        assert_eq!(got.source_ai, "test");
        assert_eq!(got.layer, MemoryLayer::Episode);
        assert_eq!(got.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_reject_invalid_hash() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let mut r = make_rec(100, MemoryLayer::Episode, "t");
        r.record_id = [0xFF; 32];
        assert!(!s.insert(&r, "m").await);
        assert_eq!(s.total_rejected(), 1);
    }

    #[tokio::test]
    async fn test_get_active_records() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let o = [0xAA; 32];
        s.insert(&make_rec_owner(100, o, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, o, MemoryLayer::Knowledge), "m").await;
        s.insert(&make_rec_owner(300, o, MemoryLayer::Archive), "m").await;
        assert_eq!(s.get_active_records(&o, None, 100).await.len(), 3);
        assert_eq!(s.get_active_records(&o, Some(MemoryLayer::Episode), 100).await.len(), 1);
    }

    #[tokio::test]
    async fn test_revoke() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "t");
        s.insert(&r, "m").await;
        assert!(s.revoke(&r.record_id).await);
        let got = s.get(&r.record_id).await.unwrap();
        assert_eq!(got.status, RecordStatus::Revoked);
        assert!(got.encrypted_content.is_empty());
    }

    #[tokio::test]
    async fn test_encrypted_insert_and_get() {
        use super::super::storage_crypto::derive_record_key;
        let key = derive_record_key(&[0x42; 32]);
        let s = MemoryStorage::open(":memory:", Some(key)).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        s.insert(&r, "m").await;
        s.cache.write().clear();
        let got = s.get(&r.record_id).await.unwrap();
        assert_eq!(got.encrypted_content, b"encrypted_data");
    }

    // ========================================
    // v2.4.0: Schema v5 Tests
    // ========================================

    #[tokio::test]
    async fn test_schema_version_is_5() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;
        let v: u32 = conn.query_row(
            "SELECT version FROM schema_version LIMIT 1", [], |r| r.get(0)
        ).unwrap();
        assert_eq!(v, 5);
    }

    #[tokio::test]
    async fn test_v5_tables_exist() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let expected_tables = [
            "episodes", "entities", "knowledge_edges", "episode_edges",
            "communities", "projects", "sessions", "artifacts",
        ];

        for table in &expected_tables {
            let exists: bool = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                params![table],
                |row| row.get::<_, i64>(0),
            ).unwrap() > 0;

            assert!(exists, "Table '{}' should exist in schema v5", table);
        }
    }

    #[tokio::test]
    async fn test_v5_records_has_new_columns() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        // Verify new columns exist by selecting them
        let result = conn.prepare(
            "SELECT project_id, session_id, episode_id FROM records LIMIT 0"
        );
        assert!(result.is_ok(), "records should have project_id, session_id, episode_id columns");
    }

    #[tokio::test]
    async fn test_v5_episodes_table_schema() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        // Insert a test episode
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result = conn.execute(
            "INSERT INTO episodes (episode_id, owner, episode_type, source,
                session_id, encrypted_content, content_hash, token_count,
                created_at, ingested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                "ep_001", [0xAAu8; 32].as_slice(), "conversation", "test",
                "session_001", b"encrypted".as_slice(), "hash123", 100i64,
                now, now,
            ],
        );
        assert!(result.is_ok(), "Episode insert should succeed: {:?}", result.err());

        // Verify we can read it back
        let ep_type: String = conn.query_row(
            "SELECT episode_type FROM episodes WHERE episode_id = 'ep_001'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(ep_type, "conversation");
    }

    #[tokio::test]
    async fn test_v5_entities_table_schema() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result = conn.execute(
            "INSERT INTO entities (entity_id, owner, name, name_normalized,
                entity_type, description, community_id, created_at, updated_at, mention_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                "ent_jwt", [0xAAu8; 32].as_slice(), "JWT", "jwt",
                "technology", "JSON Web Token", Option::<String>::None,
                now, now, 3i64,
            ],
        );
        assert!(result.is_ok(), "Entity insert should succeed: {:?}", result.err());

        let mention_count: i64 = conn.query_row(
            "SELECT mention_count FROM entities WHERE entity_id = 'ent_jwt'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(mention_count, 3);
    }

    #[tokio::test]
    async fn test_v5_knowledge_edges_temporal() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Insert a currently valid edge
        conn.execute(
            "INSERT INTO knowledge_edges (owner, source_id, target_id, relation_type,
                fact_text, weight, confidence, valid_from, valid_until, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9, ?10)",
            params![
                [0xAAu8; 32].as_slice(), "ent_auth", "ent_jwt", "USES",
                "auth module uses JWT", 1.0f64, 0.95f64,
                now, now, now,
            ],
        ).unwrap();

        // Insert an invalidated edge
        conn.execute(
            "INSERT INTO knowledge_edges (owner, source_id, target_id, relation_type,
                fact_text, weight, confidence, valid_from, valid_until, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                [0xAAu8; 32].as_slice(), "ent_auth", "ent_basic", "USES",
                "auth module uses Basic Auth", 1.0f64, 0.9f64,
                now - 86400, now, now - 86400, now,
            ],
        ).unwrap();

        // Query only currently valid edges
        let valid_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM knowledge_edges WHERE valid_until IS NULL",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(valid_count, 1);

        // Query all edges (including invalidated)
        let total_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM knowledge_edges",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(total_count, 2);
    }

    #[tokio::test]
    async fn test_v5_episode_edges_bidirectional() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Link episode to entity
        conn.execute(
            "INSERT INTO episode_edges (owner, episode_id, entity_id, role, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![[0xAAu8; 32].as_slice(), "ep_001", "ent_jwt", "mentioned", now],
        ).unwrap();

        // Forward: Episode → Entities
        let entity_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM episode_edges WHERE episode_id = 'ep_001'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(entity_count, 1);

        // Reverse: Entity → Episodes
        let episode_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM episode_edges WHERE entity_id = 'ent_jwt'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(episode_count, 1);
    }

    #[tokio::test]
    async fn test_v5_sessions_and_projects() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Create community
        conn.execute(
            "INSERT INTO communities (community_id, owner, name, summary, entity_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params!["comm_1", [0xAAu8; 32].as_slice(), "Project B", "Auth system", 5i64, now, now],
        ).unwrap();

        // Create project from community
        conn.execute(
            "INSERT INTO projects (project_id, owner, name, status, community_id, summary,
                created_at, updated_at, last_active_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params!["comm_1", [0xAAu8; 32].as_slice(), "Project B", "active", "comm_1",
                "Auth system project", now, now, now],
        ).unwrap();

        // Create session linked to project
        conn.execute(
            "INSERT INTO sessions (session_id, owner, project_id, session_type,
                started_at, turn_count, summary)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params!["sess_001", [0xAAu8; 32].as_slice(), "comm_1", "code",
                now, 15i64, "Implemented JWT auth"],
        ).unwrap();

        // Query sessions for project
        let session_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE project_id = 'comm_1'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(session_count, 1);

        // Query active projects
        let project_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM projects WHERE owner = ?1 AND status = 'active'",
            params![[0xAAu8; 32].as_slice()],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(project_count, 1);
    }

    #[tokio::test]
    async fn test_v5_artifacts_version_chain() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Version 1
        conn.execute(
            "INSERT INTO artifacts (artifact_id, owner, session_id, artifact_type,
                filename, language, version, parent_id, encrypted_content, content_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9, ?10)",
            params![
                "art_v1", [0xAAu8; 32].as_slice(), "sess_001", "code",
                "auth.rs", "rust", 1i64,
                b"fn auth() {}".as_slice(), "hash_v1", now,
            ],
        ).unwrap();

        // Version 2 (parent = v1)
        conn.execute(
            "INSERT INTO artifacts (artifact_id, owner, session_id, artifact_type,
                filename, language, version, parent_id, encrypted_content, content_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                "art_v2", [0xAAu8; 32].as_slice(), "sess_002", "code",
                "auth.rs", "rust", 2i64, "art_v1",
                b"fn auth() { jwt() }".as_slice(), "hash_v2", now + 100,
            ],
        ).unwrap();

        // Query latest version
        let latest_version: i64 = conn.query_row(
            "SELECT MAX(version) FROM artifacts WHERE owner = ?1 AND filename = 'auth.rs'",
            params![[0xAAu8; 32].as_slice()],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(latest_version, 2);

        // Query version chain
        let parent: Option<String> = conn.query_row(
            "SELECT parent_id FROM artifacts WHERE artifact_id = 'art_v2'",
            [], |row| row.get(0),
        ).unwrap();
        assert_eq!(parent, Some("art_v1".to_string()));
    }

    #[tokio::test]
    async fn test_v5_backward_compat_insert_without_new_cols() {
        // Ensure existing insert() works without providing new columns
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        assert!(s.insert(&r, "minilm").await);

        // The new columns should be NULL
        let conn = s.conn.lock().await;
        let (pid, sid, eid): (Option<String>, Option<String>, Option<String>) = conn.query_row(
            "SELECT project_id, session_id, episode_id FROM records WHERE record_id = ?1",
            params![r.record_id.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        ).unwrap();
        assert!(pid.is_none());
        assert!(sid.is_none());
        assert!(eid.is_none());
    }
}
