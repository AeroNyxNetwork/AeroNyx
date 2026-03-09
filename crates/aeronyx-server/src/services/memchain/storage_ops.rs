// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_ops.rs
// ============================================
//! # Storage Operations — Extended MemoryStorage Methods
//!
//! ## Creation Reason
//! Extracted from storage.rs to reduce file size. Contains all extended
//! operations that are NOT core CRUD: rawlog ops, feedback ops, chain state,
//! statistics, miner support, and overview queries.
//!
//! ## Main Functionality
//! - RawLog: insert_raw_log, read_rawlog_content, update_rawlog_feedback,
//!   get_unprocessed_rawlogs
//! - Feedback: increment_positive/negative_feedback, set_conflict_with,
//!   insert_feedback, get_recent_feedback
//! - Chain State: set_chain_state, last_block_hash, last_block_height
//! - Stats: stats(), count(), total_inserted/rejected
//! - Miner: count_by_layer, compact_episodes_to_archive, get_records_needing_embedding,
//!   get_correction_records, update_topic_tags, supersede_record
//! - MVF Weights: load_user_weights, save_user_weights
//! - Content Dedup: has_active_content
//! - v2.2.0: get_embedding_model, get_overview
//!
//! ## Architecture
//! This file uses `impl MemoryStorage` blocks to add methods to the struct
//! defined in storage.rs. Rust allows multiple impl blocks across files
//! within the same crate.
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
//!
//! ## Last Modified
//! v2.2.0 - 🌟 Extracted from storage.rs; added get_embedding_model, get_overview

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::{debug, error, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord, RecordStatus};

use super::storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow};
use super::storage_crypto::{
    encrypt_rawlog_content, decrypt_rawlog_content,
    encrypt_record_content, decrypt_record_content,
};

// ============================================
// Overview Types (v2.2.0)
// ============================================

/// A lightweight record for the overview endpoint.
/// Does not include embedding data (too large for list views).
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

/// Aggregated overview data returned by get_overview().
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
    /// Insert a raw conversation log entry.
    ///
    /// If `rawlog_key` is provided, content is encrypted with ChaCha20-Poly1305
    /// (random nonce) before storage. The `encrypted` flag is set accordingly.
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

    /// Query unprocessed raw_logs for Miner feedback detection.
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
    /// Increment positive_feedback on a record.
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

    /// Increment negative_feedback on a record.
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

    /// Set the conflict_with field on a record (for MVF φ₈ feature).
    pub async fn set_conflict_with(&self, record_id: &[u8; 32], conflict_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET conflict_with = ?1 WHERE record_id = ?2",
            params![conflict_id.as_slice(), record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    conflict = hex::encode(conflict_id),
                    "[STORAGE] conflict_with set"
                );
                self.cache.write().invalidate(record_id);
                true
            }
            Ok(_) => {
                warn!(record_id = hex::encode(record_id), "[STORAGE] conflict_with: record not found");
                false
            }
            Err(e) => {
                error!(error = %e, "[STORAGE] ❌ set_conflict_with failed");
                false
            }
        }
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
            session_inserts: self.total_inserted(),
            session_rejects: self.total_rejected(),
        }
    }

    pub async fn count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM records", [], |r| r.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }
}

// ============================================
// impl MemoryStorage — Miner Support
// ============================================

impl MemoryStorage {
    /// Count active records for a layer.
    pub async fn count_by_layer(&self, layer: MemoryLayer) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM records WHERE status=0 AND layer=?1",
            params![layer as u8 as i64],
            |row| row.get::<_, i64>(0),
        ).unwrap_or(0) as u64
    }

    /// Compact episodes to archive: change layer to Archive, keep status=Active.
    pub async fn compact_episodes_to_archive(
        &self,
        owner: &[u8; 32],
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;

        let records = self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
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

    /// Query records that need embedding backfill.
    pub async fn get_records_needing_embedding(&self, limit: usize) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records
             WHERE embedding IS NULL AND status = 0
             LIMIT ?1",
            params![limit as i64],
        )
    }

    /// Get records with _correction tag for Miner Step 0.6.
    pub async fn get_correction_records(&self) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(&conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,
                    signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records
             WHERE topic_tags LIKE '%_correction%' AND status = 0",
            [],
        )
    }

    /// Update topic_tags for a record.
    pub async fn update_topic_tags(&self, record_id: &[u8; 32], tags: &[String]) {
        let json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2",
            params![json, record_id.as_slice()],
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
}

// ============================================
// impl MemoryStorage — Content Dedup
// ============================================

impl MemoryStorage {
    /// Check if an active record with the same content already exists for this owner.
    ///
    /// When record_key is set, encrypts input before comparison (deterministic
    /// encryption guarantees same plaintext → same ciphertext).
    pub async fn has_active_content(&self, owner: &[u8; 32], content: &[u8]) -> bool {
        let compare_content: Vec<u8> = if let Some(ref key) = self.record_key {
            if content.is_empty() {
                content.to_vec()
            } else {
                match encrypt_record_content(key, content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        warn!("[STORAGE] has_active_content encryption failed: {}", e);
                        content.to_vec()
                    }
                }
            }
        } else {
            content.to_vec()
        };

        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM records
             WHERE owner = ?1 AND encrypted_content = ?2 AND status = 0
             LIMIT 1",
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
    /// Get the embedding_model for a record.
    ///
    /// MemoryRecord struct does not include embedding_model (it's storage metadata),
    /// so this requires a separate query.
    pub async fn get_embedding_model(&self, record_id: &[u8; 32]) -> Option<String> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT embedding_model FROM records WHERE record_id = ?1",
            params![record_id.as_slice()],
            |row| row.get::<_, String>(0),
        ).optional().unwrap_or(None)
    }

    /// Get overview data: per-layer counts + recent records per layer.
    ///
    /// Returns up to 20 records per layer (80 total max), sorted by timestamp DESC.
    /// Content is decrypted transparently if record_key is set.
    pub async fn get_overview(&self, owner: &[u8; 32], per_layer_limit: usize) -> OverviewData {
        let conn = self.conn.lock().await;
        let limit = per_layer_limit.min(50).max(1);

        // 1. Per-layer counts
        let mut by_layer = HashMap::new();
        let layer_names = [
            (0i64, "identity"),
            (1, "knowledge"),
            (2, "episode"),
            (3, "archive"),
        ];

        for (layer_val, layer_name) in &layer_names {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM records WHERE owner = ?1 AND status = 0 AND layer = ?2",
                params![owner.as_slice(), layer_val],
                |row| row.get(0),
            ).unwrap_or(0);
            by_layer.insert(layer_name.to_string(), count as u64);
        }

        // 2. Recent records per layer (with decryption)
        let mut recent_by_layer = HashMap::new();
        let rk = self.record_key;

        for (layer_val, layer_name) in &layer_names {
            let mut stmt = match conn.prepare(
                "SELECT record_id, encrypted_content, topic_tags, timestamp,
                        access_count, positive_feedback, negative_feedback, source_ai
                 FROM records
                 WHERE owner = ?1 AND layer = ?2 AND status = 0
                 ORDER BY timestamp DESC
                 LIMIT ?3"
            ) {
                Ok(s) => s,
                Err(_) => continue,
            };

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

                    // Decrypt content if record_key is set
                    let content = if let Some(key) = rk.as_ref() {
                        if raw_content.len() >= 28 {
                            match decrypt_record_content(key, &raw_content) {
                                Ok(plain) => String::from_utf8_lossy(&plain).to_string(),
                                Err(_) => String::from_utf8_lossy(&raw_content).to_string(),
                            }
                        } else {
                            String::from_utf8_lossy(&raw_content).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&raw_content).to_string()
                    };

                    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();

                    Ok(OverviewRecord {
                        record_id: hex::encode(&rid_blob),
                        content,
                        topic_tags: tags,
                        timestamp: timestamp as u64,
                        access_count: access_count as u32,
                        positive_feedback: pos_fb as u32,
                        negative_feedback: neg_fb as u32,
                        source_ai,
                    })
                },
            )
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

            recent_by_layer.insert(layer_name.to_string(), records);
        }

        // 3. Last memory timestamp
        let last_memory_at: i64 = conn.query_row(
            "SELECT COALESCE(MAX(timestamp), 0) FROM records WHERE owner = ?1 AND status = 0",
            params![owner.as_slice()],
            |row| row.get(0),
        ).unwrap_or(0);

        OverviewData {
            by_layer,
            recent_by_layer,
            last_memory_at: last_memory_at as u64,
        }
    }
}
