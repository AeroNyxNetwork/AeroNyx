// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_ops.rs
// ============================================
//! # Storage Operations — Core MemoryStorage Methods
//!
//! ## Creation Reason
//! Extracted from storage.rs to reduce file size. Contains all extended
//! operations that are NOT core CRUD: rawlog ops, feedback ops, chain state,
//! statistics, miner support, and overview queries.
//!
//! ## Main Functionality
//! - RawLog: insert_raw_log, read_rawlog_content, update_rawlog_feedback,
//!   get_unprocessed_rawlogs, get_rawlogs_for_session (→ storage_miner.rs)
//! - Feedback: increment_positive/negative_feedback, set_conflict_with,
//!   insert_feedback, get_recent_feedback
//! - Chain State: set_chain_state, last_block_hash, last_block_height
//! - Stats: stats(), count(), total_inserted/rejected
//! - Miner: count_by_layer, compact_episodes_to_archive, get_records_needing_embedding,
//!   get_correction_records, update_topic_tags, supersede_record
//! - MVF Weights: load_user_weights, save_user_weights
//! - Content Dedup: has_active_content
//! - v2.2.0: get_embedding_model, get_overview
//! - v2.3.0: count_distinct_owners, owner_exists (Phase 1 remote storage capacity check)
//! - v2.5.3+Isolation: get_active_records_by_context (project_id context filter for /recall)
//!
//! ## Split Architecture (v2.4.0+Search)
//! This file was split into three files to reduce size:
//!   - storage_ops.rs   (this file) — core/legacy ops (RawLog, Feedback, Stats, Miner, etc.)
//!   - storage_graph.rs             — cognitive graph CRUD (Episodes, Entities, Edges, etc.)
//!   - storage_miner.rs             — Miner step support (get_rawlogs_for_session, merge_entities, etc.)
//!
//! All public types and methods remain accessible via their respective modules.
//! Callers use `use super::storage_ops::{OverviewData, OverviewRecord}` etc. — unchanged.
//!
//! ## Dependencies
//! - storage.rs — MemoryStorage struct, LruCache, schema, core CRUD
//! - storage_crypto.rs — encrypt/decrypt functions for rawlog and content
//!
//! ⚠️ Important Note for Next Developer:
//! - All methods here access self.conn (TokioMutex<Connection>) and self.cache (RwLock<LruCache>)
//! - When adding new query methods, use self.query_rows() for SELECT queries that
//!   return MemoryRecord (it handles record_key decryption transparently)
//! - For raw SQL that reads encrypted_content directly (like get_overview), you MUST
//!   manually decrypt using self.record_key
//! - count_distinct_owners() and owner_exists() are used by the MPI auth middleware
//!   for max_remote_owners capacity checks. They must be fast (indexed queries).
//! - Cognitive graph methods have been moved to storage_graph.rs.
//! - Miner Step support methods have been moved to storage_miner.rs.
//! - get_active_records_by_context() uses LEFT JOIN records→sessions to check
//!   project_id on EITHER the record directly OR via its session. Records inserted
//!   via /remember directly (no session) are matched by records.project_id only.
//!
//! ## Modification History
//! v2.2.0               - 🌟 Extracted from storage.rs; added get_embedding_model, get_overview
//! v2.3.0+RemoteStorage - 🌟 Added count_distinct_owners(), owner_exists()
//! v2.4.0-GraphCognition - 🌟 Added full CRUD for cognitive graph tables
//!   (now in storage_graph.rs) + Miner Step support (now in storage_miner.rs)
//! v2.4.0+Search        - 🌟 Split into storage_ops.rs / storage_graph.rs / storage_miner.rs
//! v2.5.3+Isolation     - 🌟 Added get_active_records_by_context() for /recall context filter
//!
//! ## Last Modified
//! v2.5.3+Isolation - 🌟 get_active_records_by_context() + 2 tests

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::{debug, error, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord, RecordStatus};

use super::storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow, embedding_to_bytes};
use super::storage_crypto::{
    encrypt_rawlog_content, decrypt_rawlog_content,
    encrypt_record_content, decrypt_record_content,
};

// ============================================
// Overview Types (v2.2.0)
// ============================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct OverviewRecord {
    pub record_id: String,
    pub content: String,
    pub topic_tags: Vec<String>,
    pub timestamp: u64,
    pub access_count: u32,
    pub positive_feedback: u32,
    pub negative_feedback: u32,
    pub source_ai: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OverviewData {
    pub by_layer: HashMap<String, u64>,
    pub recent_by_layer: HashMap<String, Vec<OverviewRecord>>,
    pub last_memory_at: u64,
}

// ============================================
// impl MemoryStorage — RawLog Operations
// ============================================

impl MemoryStorage {
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

    pub async fn update_rawlog_feedback(&self, log_id: i64, signal: i64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE raw_logs SET feedback_signal = ?1 WHERE log_id = ?2",
            params![signal, log_id],
        );
    }

    pub async fn get_unprocessed_rawlogs(&self, limit: usize) -> Vec<RawLogRow> {
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
}

// ============================================
// impl MemoryStorage — Feedback Operations
// ============================================

impl MemoryStorage {
    pub async fn increment_positive_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET positive_feedback = positive_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(record_id = hex::encode(record_id), "[STORAGE] positive_feedback incremented");
                self.cache.write().invalidate(record_id);
            }
            Ok(_) => {
                warn!(record_id = hex::encode(record_id), "[STORAGE] positive_feedback: record not found");
            }
            Err(e) => {
                error!(
                    record_id = hex::encode(record_id), error = %e,
                    "[STORAGE] ❌ positive_feedback increment failed (schema migration needed?)"
                );
            }
        }
    }

    pub async fn increment_negative_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET negative_feedback = negative_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(record_id = hex::encode(record_id), "[STORAGE] negative_feedback incremented");
                self.cache.write().invalidate(record_id);
            }
            Ok(_) => {
                warn!(record_id = hex::encode(record_id), "[STORAGE] negative_feedback: record not found");
            }
            Err(e) => {
                error!(
                    record_id = hex::encode(record_id), error = %e,
                    "[STORAGE] ❌ negative_feedback increment failed (schema migration needed?)"
                );
            }
        }
    }

    pub async fn set_conflict_with(&self, record_id: &[u8; 32], conflict_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET conflict_with = ?1 WHERE record_id = ?2",
            params![conflict_id.as_slice(), record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(record_id = hex::encode(record_id), conflict = hex::encode(conflict_id), "[STORAGE] conflict_with set");
                self.cache.write().invalidate(record_id);
                true
            }
            Ok(_) => { warn!(record_id = hex::encode(record_id), "[STORAGE] conflict_with: record not found"); false }
            Err(e) => { error!(error = %e, "[STORAGE] ❌ set_conflict_with failed"); false }
        }
    }

    pub async fn insert_feedback(
        &self, owner: &[u8; 32], memory_id: &[u8; 32], session_id: &str,
        turn_index: i64, signal: i64, features: Option<&[f32; 9]>, prediction: Option<f32>,
    ) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let features_blob: Option<Vec<u8>> = features.map(|f| f.iter().flat_map(|v| v.to_le_bytes()).collect());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO memory_feedback (owner, memory_id, session_id, turn_index,
                signal, features, prediction, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                owner.as_slice(), memory_id.as_slice(), session_id,
                turn_index, signal, features_blob.as_deref(), prediction, now,
            ],
        );
    }

    pub async fn get_recent_feedback(&self, limit: usize) -> Vec<(i64, f32)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT signal, prediction FROM memory_feedback ORDER BY created_at DESC LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1).unwrap_or(0.0)))
        }).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — Chain State
// ============================================

impl MemoryStorage {
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
}

// ============================================
// impl MemoryStorage — Statistics
// ============================================

impl MemoryStorage {
    pub async fn stats(&self) -> StorageStats {
        let conn = self.conn.lock().await;
        let q = |sql: &str, p: &[&dyn rusqlite::ToSql]| -> u64 {
            conn.query_row(sql, p, |r| r.get::<_, i64>(0)).unwrap_or(0) as u64
        };
        let total = q("SELECT COUNT(*) FROM records", &[]);
        let active = q("SELECT COUNT(*) FROM records WHERE status=0", &[]);
        StorageStats {
            total_records: total, active_records: active,
            by_layer: LayerCounts {
                identity:  q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=0", &[]),
                knowledge: q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=1", &[]),
                episode:   q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=2", &[]),
                archive:   q("SELECT COUNT(*) FROM records WHERE status=0 AND layer=3", &[]),
            },
            content_bytes: q("SELECT COALESCE(SUM(LENGTH(encrypted_content)),0) FROM records WHERE status=0", &[]),
            records_with_embedding: q("SELECT COUNT(*) FROM records WHERE status=0 AND embedding IS NOT NULL", &[]),
            session_inserts: self.total_inserted(),
            session_rejects: self.total_rejected(),
        }
    }

    pub async fn count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM records", [], |r| r.get::<_, i64>(0)).unwrap_or(0) as usize
    }
}

// ============================================
// impl MemoryStorage — Miner Support
// ============================================

impl MemoryStorage {
    pub async fn count_by_layer(&self, layer: MemoryLayer) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM records WHERE status=0 AND layer=?1",
            params![layer as u8 as i64], |row| row.get::<_, i64>(0),
        ).unwrap_or(0) as u64
    }

    pub async fn compact_episodes_to_archive(&self, owner: &[u8; 32], limit: usize) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        let records = self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE owner=?1 AND status=0 AND layer=?2 ORDER BY timestamp ASC LIMIT ?3",
            params![owner.as_slice(), MemoryLayer::Episode as u8 as i64, limit as i64],
        );
        if records.is_empty() { return records; }
        if conn.execute_batch("BEGIN TRANSACTION").is_err() { return Vec::new(); }
        for r in &records {
            let now_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
            if let Err(e) = conn.execute(
                "UPDATE records SET layer=?1, archived_at=?2 WHERE record_id=?3",
                params![MemoryLayer::Archive as u8 as i64, now_ts, r.record_id.as_slice()],
            ) {
                error!(error=%e, "[STORAGE] ❌ Compact update failed, rolling back");
                let _ = conn.execute_batch("ROLLBACK");
                return Vec::new();
            }
        }
        if conn.execute_batch("COMMIT").is_err() { return Vec::new(); }
        info!(count = records.len(), "[STORAGE] ⛏️ Episodes compacted to Archive layer");
        records
    }

    pub async fn get_records_needing_embedding(&self, limit: usize) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE embedding IS NULL AND status = 0 LIMIT ?1",
            params![limit as i64],
        )
    }

    pub async fn get_correction_records(&self) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE topic_tags LIKE '%_correction%' AND status = 0", [],
        )
    }

    pub async fn update_topic_tags(&self, record_id: &[u8; 32], tags: &[String]) {
        let json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2",
            params![json, record_id.as_slice()],
        );
    }

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

    pub async fn load_user_weights(&self, owner: &[u8; 32]) -> Option<Vec<u8>> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT weights FROM user_weights WHERE owner = ?1",
            params![owner.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        ).optional().ok()?
    }

    pub async fn save_user_weights(&self, owner: &[u8; 32], weights_blob: &[u8], version: u64) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO user_weights (owner, weights, version, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(owner) DO UPDATE SET weights=?2, version=?3, updated_at=?4",
            params![owner.as_slice(), weights_blob, version as i64, now],
        );
    }
}

// ============================================
// impl MemoryStorage — Content Dedup
// ============================================

impl MemoryStorage {
    pub async fn has_active_content(&self, owner: &[u8; 32], content: &[u8]) -> bool {
        let compare_content: Vec<u8> = if let Some(ref key) = self.record_key {
            if content.is_empty() { content.to_vec() }
            else {
                match encrypt_record_content(key, content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        warn!("[STORAGE] has_active_content encryption failed: {}", e);
                        content.to_vec()
                    }
                }
            }
        } else { content.to_vec() };
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM records WHERE owner = ?1 AND encrypted_content = ?2 AND status = 0 LIMIT 1",
            params![owner.as_slice(), compare_content.as_slice()],
            |row| row.get(0),
        ).unwrap_or(0);
        count > 0
    }
}

// ============================================
// impl MemoryStorage — v2.2.0 MemExplorer
// ============================================

impl MemoryStorage {
    pub async fn get_embedding_model(&self, record_id: &[u8; 32]) -> Option<String> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT embedding_model FROM records WHERE record_id = ?1",
            params![record_id.as_slice()],
            |row| row.get::<_, String>(0),
        ).optional().unwrap_or(None)
    }

    pub async fn get_overview(&self, owner: &[u8; 32], per_layer_limit: usize) -> OverviewData {
        let conn = self.conn.lock().await;
        let limit = per_layer_limit.min(50).max(1);
        let mut by_layer = HashMap::new();
        let layer_names = [(0i64, "identity"), (1, "knowledge"), (2, "episode"), (3, "archive")];
        for (layer_val, layer_name) in &layer_names {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM records WHERE owner = ?1 AND status = 0 AND layer = ?2",
                params![owner.as_slice(), layer_val],
                |row| row.get(0),
            ).unwrap_or(0);
            by_layer.insert(layer_name.to_string(), count as u64);
        }
        let mut recent_by_layer = HashMap::new();
        let rk = self.record_key;
        for (layer_val, layer_name) in &layer_names {
            let mut stmt = match conn.prepare(
                "SELECT record_id, encrypted_content, topic_tags, timestamp,
                        access_count, positive_feedback, negative_feedback, source_ai
                 FROM records WHERE owner = ?1 AND layer = ?2 AND status = 0
                 ORDER BY timestamp DESC LIMIT ?3"
            ) { Ok(s) => s, Err(_) => continue };
            let records: Vec<OverviewRecord> = stmt.query_map(
                params![owner.as_slice(), layer_val, limit as i64],
                |row| {
                    let rid_blob: Vec<u8> = row.get(0)?;
                    let raw_content: Vec<u8> = row.get(1)?;
                    let tags_json: String = row.get(2)?;
                    let timestamp: i64 = row.get(3)?;
                    let access_count: i64 = row.get(4)?;
                    let pos_fb: i64 = row.get(5)?;
                    let neg_fb: i64 = row.get(6)?;
                    let source_ai: String = row.get(7)?;
                    let content = if let Some(key) = rk.as_ref() {
                        if raw_content.len() >= 28 {
                            match decrypt_record_content(key, &raw_content) {
                                Ok(plain) => String::from_utf8_lossy(&plain).to_string(),
                                Err(_) => String::from_utf8_lossy(&raw_content).to_string(),
                            }
                        } else { String::from_utf8_lossy(&raw_content).to_string() }
                    } else { String::from_utf8_lossy(&raw_content).to_string() };
                    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                    Ok(OverviewRecord {
                        record_id: hex::encode(&rid_blob), content, topic_tags: tags,
                        timestamp: timestamp as u64, access_count: access_count as u32,
                        positive_feedback: pos_fb as u32, negative_feedback: neg_fb as u32,
                        source_ai,
                    })
                },
            ).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default();
            recent_by_layer.insert(layer_name.to_string(), records);
        }
        let last_memory_at: i64 = conn.query_row(
            "SELECT COALESCE(MAX(timestamp), 0) FROM records WHERE owner = ?1 AND status = 0",
            params![owner.as_slice()],
            |row| row.get(0),
        ).unwrap_or(0);
        OverviewData { by_layer, recent_by_layer, last_memory_at: last_memory_at as u64 }
    }
}

// ============================================
// impl MemoryStorage — v2.3.0 Remote Storage
// ============================================

impl MemoryStorage {
    pub async fn count_distinct_owners(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT COUNT(DISTINCT owner) FROM records",
            [],
            |row| row.get::<_, i64>(0),
        ).unwrap_or(0) as usize
    }

    pub async fn owner_exists(&self, owner: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM records WHERE owner = ?1 LIMIT 1)",
            params![owner.as_slice()],
            |row| row.get::<_, bool>(0),
        ).unwrap_or(false)
    }
}

// ============================================
// impl MemoryStorage — v2.5.3+Isolation: Context Filter
// ============================================

impl MemoryStorage {
    /// Get active records filtered by project_id (context isolation).
    ///
    /// Matches records via TWO paths:
    /// 1. `records.project_id = project_id` — direct tag (set by future /remember context param)
    /// 2. `records.session_id → sessions.project_id = project_id` — via session association
    ///    (set by /log handler when `context` field is provided, stored by upsert_session)
    ///
    /// Records inserted via `/remember` directly without a session use path 1 only.
    /// Records inserted via `/log` with `context` field use path 2.
    ///
    /// ## Usage
    /// Called by `recall_handler.rs` Step 4.1 when `RecallRequest.context` is
    /// a non-"all" value. The caller builds a HashSet of returned record_ids
    /// and filters `scored` with `retain()`.
    ///
    /// ## Performance
    /// LEFT JOIN on session_id. For best performance, ensure an index exists on
    /// `records.session_id` and `sessions.project_id`. With typical MemChain
    /// data sizes (< 100k records per user) this is fast without extra indexes.
    ///
    /// v2.5.3+Isolation
    pub async fn get_active_records_by_context(
        &self,
        owner: &[u8; 32],
        project_id: &str,
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;

        if let Some(l) = layer {
            self.query_rows(&conn,
                "SELECT r.record_id, r.owner, r.timestamp, r.layer, r.topic_tags, r.source_ai,
                        r.status, r.supersedes, r.encrypted_content, r.embedding, r.signature,
                        r.access_count, r.positive_feedback, r.negative_feedback, r.conflict_with
                 FROM records r
                 LEFT JOIN sessions s ON r.session_id = s.session_id
                 WHERE r.owner = ?1
                   AND r.status = 0
                   AND r.layer = ?2
                   AND (r.project_id = ?3 OR s.project_id = ?3)
                 ORDER BY r.timestamp DESC
                 LIMIT ?4",
                params![owner.as_slice(), l as u8 as i64, project_id, limit as i64],
            )
        } else {
            self.query_rows(&conn,
                "SELECT r.record_id, r.owner, r.timestamp, r.layer, r.topic_tags, r.source_ai,
                        r.status, r.supersedes, r.encrypted_content, r.embedding, r.signature,
                        r.access_count, r.positive_feedback, r.negative_feedback, r.conflict_with
                 FROM records r
                 LEFT JOIN sessions s ON r.session_id = s.session_id
                 WHERE r.owner = ?1
                   AND r.status = 0
                   AND (r.project_id = ?2 OR s.project_id = ?2)
                 ORDER BY r.timestamp DESC
                 LIMIT ?3",
                params![owner.as_slice(), project_id, limit as i64],
            )
        }
    }
}

// ============================================
// Tests
// ============================================
// v2.4.0+Search: Cognitive graph tests moved to storage_graph.rs.
//   Miner step support tests moved to storage_miner.rs.

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::ledger::MemoryRecord;

    fn make_rec_owner(ts: u64, owner: [u8; 32], layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new(owner, ts, layer, vec!["test".into()], "ai".into(),
            format!("content_{}", ts).into_bytes(), vec![0.5; 4])
    }

    #[tokio::test]
    async fn test_has_active_content() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xBB; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Episode);
        s.insert(&r, "m").await;
        assert!(s.has_active_content(&owner, &r.encrypted_content).await);
        assert!(!s.has_active_content(&owner, b"nonexistent").await);
        assert!(!s.has_active_content(&[0xCC; 32], &r.encrypted_content).await);
    }

    #[tokio::test]
    async fn test_insert_raw_log_plaintext() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let log_id = s.insert_raw_log("s1", 0, "user", "hello", "test", None, 1, None, None).await.unwrap();
        assert!(log_id > 0);
        let content = s.read_rawlog_content(log_id, None).await.unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_insert_raw_log_encrypted() {
        use super::super::storage_crypto::derive_rawlog_key;
        let rlk = derive_rawlog_key(&[0x42; 32]);
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let log_id = s.insert_raw_log("s1", 0, "user", "secret", "test", None, 1, None, Some(&rlk)).await.unwrap();
        let content = s.read_rawlog_content(log_id, Some(&rlk)).await.unwrap();
        assert_eq!(content, "secret");
        assert!(s.read_rawlog_content(log_id, None).await.is_none());
    }

    #[tokio::test]
    async fn test_feedback_operations() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Episode);
        let rid = r.record_id;
        s.insert(&r, "m").await;
        s.increment_positive_feedback(&rid).await;
        s.increment_negative_feedback(&rid).await;
        s.increment_negative_feedback(&rid).await;
        let got = s.get(&rid).await.unwrap();
        assert_eq!(got.positive_feedback, 1);
        assert_eq!(got.negative_feedback, 2);
    }

    #[tokio::test]
    async fn test_stats() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, owner, MemoryLayer::Identity), "m").await;
        let stats = s.stats().await;
        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.active_records, 2);
        assert_eq!(stats.by_layer.episode, 1);
        assert_eq!(stats.by_layer.identity, 1);
    }

    #[tokio::test]
    async fn test_get_embedding_model() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec_owner(100, [0xAA; 32], MemoryLayer::Episode);
        let rid = r.record_id;
        s.insert(&r, "minilm-l6-v2").await;
        assert_eq!(s.get_embedding_model(&rid).await, Some("minilm-l6-v2".into()));
    }

    #[tokio::test]
    async fn test_get_overview() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, owner, MemoryLayer::Identity), "m").await;
        s.insert(&make_rec_owner(300, owner, MemoryLayer::Knowledge), "m").await;
        let ov = s.get_overview(&owner, 20).await;
        assert_eq!(*ov.by_layer.get("episode").unwrap(), 1);
        assert_eq!(*ov.by_layer.get("identity").unwrap(), 1);
        assert_eq!(*ov.by_layer.get("knowledge").unwrap(), 1);
        assert_eq!(ov.last_memory_at, 300);
    }

    #[tokio::test]
    async fn test_count_distinct_owners_empty() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        assert_eq!(s.count_distinct_owners().await, 0);
    }

    #[tokio::test]
    async fn test_count_distinct_owners_multiple() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert(&make_rec_owner(100, [0xAA; 32], MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(200, [0xBB; 32], MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(300, [0xCC; 32], MemoryLayer::Episode), "m").await;
        s.insert(&make_rec_owner(400, [0xAA; 32], MemoryLayer::Identity), "m").await;
        assert_eq!(s.count_distinct_owners().await, 3);
    }

    #[tokio::test]
    async fn test_owner_exists_after_insert() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m").await;
        assert!(s.owner_exists(&owner).await);
        assert!(!s.owner_exists(&[0xBB; 32]).await);
    }

    // ── v2.5.3+Isolation: get_active_records_by_context tests ──

    #[tokio::test]
    async fn test_get_active_records_by_context_via_session() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;

        // Register a session with project_id = "work"
        s.upsert_session("sess_work", &owner, Some("work"), "chat", now, 2).await.unwrap();

        // Insert two records linked to the work session
        let mut r1 = make_rec_owner(100, owner, MemoryLayer::Episode);
        // Manually set session_id on the record row after insert
        s.insert(&r1, "m").await;
        {
            let conn = s.conn_lock().await;
            let _ = conn.execute(
                "UPDATE records SET session_id = 'sess_work' WHERE record_id = ?1",
                params![r1.record_id.as_slice()],
            );
        }

        // Insert a record with no session (personal — untagged)
        let r2 = make_rec_owner(200, owner, MemoryLayer::Episode);
        s.insert(&r2, "m").await;

        // Query context = "work" should return only r1
        let results = s.get_active_records_by_context(&owner, "work", None, 100).await;
        assert_eq!(results.len(), 1, "Only the work-tagged record should be returned");
        assert_eq!(results[0].record_id, r1.record_id);

        // Query context = "personal" should return nothing (no records tagged personal)
        let personal = s.get_active_records_by_context(&owner, "personal", None, 100).await;
        assert!(personal.is_empty(), "No personal records exist");
    }

    #[tokio::test]
    async fn test_get_active_records_by_context_no_cross_owner() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner_a = [0xAA; 32];
        let owner_b = [0xBB; 32];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;

        s.upsert_session("sess_a", &owner_a, Some("work"), "chat", now, 1).await.unwrap();

        let r = make_rec_owner(100, owner_a, MemoryLayer::Episode);
        s.insert(&r, "m").await;
        {
            let conn = s.conn_lock().await;
            let _ = conn.execute(
                "UPDATE records SET session_id = 'sess_a' WHERE record_id = ?1",
                params![r.record_id.as_slice()],
            );
        }

        // owner_b querying "work" must see nothing (different owner)
        let results = s.get_active_records_by_context(&owner_b, "work", None, 100).await;
        assert!(results.is_empty(), "Cross-owner isolation must be enforced");
    }
}
