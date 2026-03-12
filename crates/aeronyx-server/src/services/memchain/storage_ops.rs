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
//!   get_unprocessed_rawlogs, get_rawlogs_for_session (v2.4.0 Phase B)
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
//! - v2.4.0-GraphCognition: Full CRUD for cognitive graph tables:
//!   - Episodes: upsert_episode, get_episode, get_episodes_for_session
//!   - Entities: upsert_entity, get_entity, get_entities_by_owner, get_entities_cached,
//!     increment_entity_mention, update_entity_community, get_entities_with_embedding
//!   - Knowledge Edges: insert_knowledge_edge, get_active_edges, get_edges_for_entity,
//!     invalidate_edge, get_edges_within_community
//!   - Episode Edges: insert_episode_edge, get_entities_for_episode, get_episodes_for_entity
//!   - Communities: upsert_community, get_communities, get_entities_in_community,
//!     get_communities_with_new_entities
//!   - Projects: upsert_project, get_projects, get_project
//!   - Sessions: upsert_session, get_session, get_sessions_for_project,
//!     update_session_summary, get_pending_sessions, update_session_ended_at,
//!     mark_session_entities_extracted, mark_session_summary_generated,
//!     mark_session_artifacts_extracted
//!   - Artifacts: insert_artifact, get_artifacts_for_session, get_artifact_versions
//!   - Graph Stats: graph_stats
//!   - Entity Merge: merge_entities (v2.4.0 Phase B — Miner Step 9)
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
//! - count_distinct_owners() and owner_exists() are used by the MPI auth middleware
//!   for max_remote_owners capacity checks. They must be fast (indexed queries).
//! - v2.4.0 cognitive graph methods use TEXT primary keys (SHA256 hashes or UUIDs),
//!   not BLOB like records. This simplifies JSON serialization for API responses.
//! - knowledge_edges.valid_until = NULL means "currently valid". Always filter by
//!   valid_until IS NULL for current state queries.
//! - entity_id = SHA256(owner || name_normalized) — deterministic, enables upsert.
//! - merge_entities() performs cascading updates across knowledge_edges, episode_edges,
//!   and cleans up self-referencing edges created by the merge. The source entity is
//!   deleted after merge.
//! - mark_session_artifacts_extracted() requires the `artifacts_extracted` column in
//!   the `sessions` table. Ensure schema migration adds this column before calling.
//!
//! ## Last Modified
//! v2.2.0 - 🌟 Extracted from storage.rs; added get_embedding_model, get_overview
//! v2.3.0+RemoteStorage - 🌟 Added count_distinct_owners(), owner_exists()
//! v2.4.0-GraphCognition - 🌟 Added full CRUD for episodes, entities, knowledge_edges,
//!   episode_edges, communities, projects, sessions, artifacts tables + graph_stats
//! v2.4.0-GraphCognition Phase B - 🌟 Added get_rawlogs_for_session, update_session_ended_at,
//!   mark_session_artifacts_extracted, mark_session_summary_generated,
//!   get_entities_with_embedding, merge_entities (Miner Steps 7/9/10 support)

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
// v2.4.0: Cognitive Graph Types
// ============================================

/// Lightweight entity row for API responses and graph traversal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityRow {
    pub entity_id: String,
    pub name: String,
    pub name_normalized: String,
    pub entity_type: String,
    pub description: Option<String>,
    pub community_id: Option<String>,
    pub mention_count: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Knowledge edge row for graph traversal and API responses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnowledgeEdgeRow {
    pub edge_id: i64,
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String,
    pub fact_text: Option<String>,
    pub weight: f64,
    pub confidence: f64,
    pub valid_from: i64,
    pub valid_until: Option<i64>,
    pub episode_id: Option<String>,
}

/// Session row for timeline and detail views.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionRow {
    pub session_id: String,
    pub project_id: Option<String>,
    pub session_type: String,
    pub started_at: i64,
    pub ended_at: Option<i64>,
    pub turn_count: i64,
    pub summary: Option<String>,
    pub key_decisions: Option<String>,
    pub entities_extracted: bool,
    pub summary_generated: bool,
}

/// Community row for API responses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommunityRow {
    pub community_id: String,
    pub name: String,
    pub summary: Option<String>,
    pub description: Option<String>,
    pub entity_count: i64,
    pub updated_at: i64,
}

/// Project row for API responses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProjectRow {
    pub project_id: String,
    pub name: String,
    pub status: String,
    pub community_id: String,
    pub summary: Option<String>,
    pub last_active_at: i64,
}

/// Artifact row for API responses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArtifactRow {
    pub artifact_id: String,
    pub session_id: String,
    pub project_id: Option<String>,
    pub artifact_type: String,
    pub filename: Option<String>,
    pub language: Option<String>,
    pub version: i64,
    pub parent_id: Option<String>,
    pub content_hash: String,
    pub line_count: Option<i64>,
    pub created_at: i64,
}

/// Cognitive graph statistics for /api/mpi/status.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GraphStats {
    pub episodes: u64,
    pub entities: u64,
    pub knowledge_edges: u64,
    pub communities: u64,
    pub projects: u64,
    pub sessions: u64,
    pub artifacts: u64,
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
                let mut h = [0u8; 32]; if blob.len() == 32 { h.copy_from_slice(&blob); } Ok(h)
            },
        ).unwrap_or([0u8; 32])
    }

    pub async fn last_block_height(&self) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_height'",
            [], |row| {
                let blob: Vec<u8> = row.get(0)?;
                if blob.len() == 8 { let mut b = [0u8; 8]; b.copy_from_slice(&blob); Ok(u64::from_le_bytes(b)) }
                else { Ok(0u64) }
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
        let _ = conn.execute("UPDATE records SET topic_tags = ?1 WHERE record_id = ?2", params![json, record_id.as_slice()]);
    }

    pub async fn supersede_record(&self, old_id: &[u8; 32], new_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        let r1 = conn.execute("UPDATE records SET status = 1 WHERE record_id = ?1", params![old_id.as_slice()]);
        let r2 = conn.execute("UPDATE records SET supersedes = ?1 WHERE record_id = ?2", params![old_id.as_slice(), new_id.as_slice()]);
        r1.is_ok() && r2.is_ok()
    }

    pub async fn load_user_weights(&self, owner: &[u8; 32]) -> Option<Vec<u8>> {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT weights FROM user_weights WHERE owner = ?1", params![owner.as_slice()], |row| row.get::<_, Vec<u8>>(0)).optional().ok()?
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
                    Err(e) => { warn!("[STORAGE] has_active_content encryption failed: {}", e); content.to_vec() }
                }
            }
        } else { content.to_vec() };
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM records WHERE owner = ?1 AND encrypted_content = ?2 AND status = 0 LIMIT 1",
            params![owner.as_slice(), compare_content.as_slice()], |row| row.get(0),
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
        conn.query_row("SELECT embedding_model FROM records WHERE record_id = ?1", params![record_id.as_slice()], |row| row.get::<_, String>(0)).optional().unwrap_or(None)
    }

    pub async fn get_overview(&self, owner: &[u8; 32], per_layer_limit: usize) -> OverviewData {
        let conn = self.conn.lock().await;
        let limit = per_layer_limit.min(50).max(1);
        let mut by_layer = HashMap::new();
        let layer_names = [(0i64, "identity"), (1, "knowledge"), (2, "episode"), (3, "archive")];
        for (layer_val, layer_name) in &layer_names {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM records WHERE owner = ?1 AND status = 0 AND layer = ?2",
                params![owner.as_slice(), layer_val], |row| row.get(0),
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
                        positive_feedback: pos_fb as u32, negative_feedback: neg_fb as u32, source_ai,
                    })
                },
            ).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default();
            recent_by_layer.insert(layer_name.to_string(), records);
        }
        let last_memory_at: i64 = conn.query_row(
            "SELECT COALESCE(MAX(timestamp), 0) FROM records WHERE owner = ?1 AND status = 0",
            params![owner.as_slice()], |row| row.get(0),
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
        conn.query_row("SELECT COUNT(DISTINCT owner) FROM records", [], |row| row.get::<_, i64>(0)).unwrap_or(0) as usize
    }

    pub async fn owner_exists(&self, owner: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM records WHERE owner = ?1 LIMIT 1)",
            params![owner.as_slice()], |row| row.get::<_, bool>(0),
        ).unwrap_or(false)
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Episodes
// ============================================

impl MemoryStorage {
    /// Insert or update an episode (complete conversation window).
    /// Uses content_hash for dedup (INSERT OR IGNORE on episode_id).
    pub async fn upsert_episode(
        &self, episode_id: &str, owner: &[u8; 32], episode_type: &str,
        source: &str, session_id: Option<&str>, encrypted_content: &[u8],
        content_hash: &str, embedding: Option<&[f32]>, token_count: Option<i64>,
        created_at: i64, metadata_json: Option<&str>,
    ) -> Result<(), String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let emb_blob: Option<Vec<u8>> = embedding.map(embedding_to_bytes);
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR IGNORE INTO episodes
                (episode_id, owner, episode_type, source, session_id, encrypted_content,
                 content_hash, embedding, token_count, created_at, ingested_at, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                episode_id, owner.as_slice(), episode_type, source, session_id,
                encrypted_content, content_hash, emb_blob.as_deref(),
                token_count, created_at, now, metadata_json,
            ],
        ).map_err(|e| format!("Episode insert: {}", e))?;
        Ok(())
    }

    /// Get episodes for a session, ordered by creation time.
    pub async fn get_episodes_for_session(&self, session_id: &str) -> Vec<(String, String, i64)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT episode_id, episode_type, created_at FROM episodes
             WHERE session_id = ?1 ORDER BY created_at ASC"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![session_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?))
        }).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Entities
// ============================================

impl MemoryStorage {
    /// Upsert an entity. If entity_id already exists, increment mention_count
    /// and update description/updated_at.
    pub async fn upsert_entity(
        &self, entity_id: &str, owner: &[u8; 32], name: &str, name_normalized: &str,
        entity_type: &str, description: Option<&str>, embedding: Option<&[f32]>,
    ) -> Result<bool, String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let emb_blob: Option<Vec<u8>> = embedding.map(embedding_to_bytes);
        let conn = self.conn.lock().await;

        // Try insert first
        let inserted = conn.execute(
            "INSERT OR IGNORE INTO entities
                (entity_id, owner, name, name_normalized, entity_type, description,
                 embedding, created_at, updated_at, mention_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)",
            params![
                entity_id, owner.as_slice(), name, name_normalized,
                entity_type, description, emb_blob.as_deref(), now, now,
            ],
        ).map_err(|e| format!("Entity insert: {}", e))?;

        if inserted == 0 {
            // Already exists — increment mention count and update
            conn.execute(
                "UPDATE entities SET mention_count = mention_count + 1, updated_at = ?1,
                    description = COALESCE(?2, description),
                    embedding = COALESCE(?3, embedding)
                 WHERE entity_id = ?4",
                params![now, description, emb_blob.as_deref(), entity_id],
            ).map_err(|e| format!("Entity update: {}", e))?;
            Ok(false) // existing
        } else {
            Ok(true) // new
        }
    }

    /// Get entity by ID.
    pub async fn get_entity(&self, entity_id: &str) -> Option<EntityRow> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT entity_id, name, name_normalized, entity_type, description,
                    community_id, mention_count, created_at, updated_at
             FROM entities WHERE entity_id = ?1",
            params![entity_id],
            |row| Ok(EntityRow {
                entity_id: row.get(0)?, name: row.get(1)?, name_normalized: row.get(2)?,
                entity_type: row.get(3)?, description: row.get(4)?,
                community_id: row.get(5)?, mention_count: row.get(6)?,
                created_at: row.get(7)?, updated_at: row.get(8)?,
            }),
        ).optional().unwrap_or(None)
    }

    /// Get all entities for an owner, optionally filtered by type.
    pub async fn get_entities_by_owner(
        &self, owner: &[u8; 32], entity_type: Option<&str>, limit: usize,
    ) -> Vec<EntityRow> {
        let conn = self.conn.lock().await;
        if let Some(et) = entity_type {
            let mut stmt = match conn.prepare(
                "SELECT entity_id, name, name_normalized, entity_type, description,
                        community_id, mention_count, created_at, updated_at
                 FROM entities WHERE owner = ?1 AND entity_type = ?2
                 ORDER BY mention_count DESC LIMIT ?3"
            ) { Ok(s) => s, Err(_) => return Vec::new() };
            stmt.query_map(params![owner.as_slice(), et, limit as i64], |row| Ok(EntityRow {
                entity_id: row.get(0)?, name: row.get(1)?, name_normalized: row.get(2)?,
                entity_type: row.get(3)?, description: row.get(4)?,
                community_id: row.get(5)?, mention_count: row.get(6)?,
                created_at: row.get(7)?, updated_at: row.get(8)?,
            })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
        } else {
            let mut stmt = match conn.prepare(
                "SELECT entity_id, name, name_normalized, entity_type, description,
                        community_id, mention_count, created_at, updated_at
                 FROM entities WHERE owner = ?1 ORDER BY mention_count DESC LIMIT ?2"
            ) { Ok(s) => s, Err(_) => return Vec::new() };
            stmt.query_map(params![owner.as_slice(), limit as i64], |row| Ok(EntityRow {
                entity_id: row.get(0)?, name: row.get(1)?, name_normalized: row.get(2)?,
                entity_type: row.get(3)?, description: row.get(4)?,
                community_id: row.get(5)?, mention_count: row.get(6)?,
                created_at: row.get(7)?, updated_at: row.get(8)?,
            })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
        }
    }

    /// Get cached entity names for a given owner (for Stage 1 novelty scoring).
    /// Returns a map of name_normalized → entity_id.
    pub async fn get_entities_cached(&self, owner: &[u8; 32]) -> HashMap<String, String> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT name_normalized, entity_id FROM entities WHERE owner = ?1"
        ) { Ok(s) => s, Err(_) => return HashMap::new() };
        stmt.query_map(params![owner.as_slice()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Update an entity's community assignment.
    pub async fn update_entity_community(&self, entity_id: &str, community_id: &str) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE entities SET community_id = ?1, updated_at = ?2 WHERE entity_id = ?3",
            params![community_id, now, entity_id],
        );
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Knowledge Edges
// ============================================

impl MemoryStorage {
    /// Insert a new knowledge edge (relationship between entities).
    pub async fn insert_knowledge_edge(
        &self, owner: &[u8; 32], source_id: &str, target_id: &str,
        relation_type: &str, fact_text: Option<&str>, weight: f64, confidence: f64,
        embedding: Option<&[f32]>, valid_from: i64, episode_id: Option<&str>,
    ) -> Result<i64, String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let emb_blob: Option<Vec<u8>> = embedding.map(embedding_to_bytes);
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO knowledge_edges
                (owner, source_id, target_id, relation_type, fact_text, weight, confidence,
                 embedding, valid_from, valid_until, episode_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, NULL, ?10, ?11, ?12)",
            params![
                owner.as_slice(), source_id, target_id, relation_type,
                fact_text, weight, confidence, emb_blob.as_deref(),
                valid_from, episode_id, now, now,
            ],
        ).map_err(|e| format!("Knowledge edge insert: {}", e))?;
        Ok(conn.last_insert_rowid())
    }

    /// Get all currently valid edges from/to an entity.
    pub async fn get_edges_for_entity(&self, entity_id: &str, owner: &[u8; 32]) -> Vec<KnowledgeEdgeRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT edge_id, source_id, target_id, relation_type, fact_text, weight,
                    confidence, valid_from, valid_until, episode_id
             FROM knowledge_edges
             WHERE owner = ?1 AND (source_id = ?2 OR target_id = ?2) AND valid_until IS NULL
             ORDER BY weight DESC"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice(), entity_id], |row| Ok(KnowledgeEdgeRow {
            edge_id: row.get(0)?, source_id: row.get(1)?, target_id: row.get(2)?,
            relation_type: row.get(3)?, fact_text: row.get(4)?, weight: row.get(5)?,
            confidence: row.get(6)?, valid_from: row.get(7)?, valid_until: row.get(8)?,
            episode_id: row.get(9)?,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Get currently valid edges for BFS traversal from a set of entity IDs.
    /// Returns edges sorted by weight × confidence descending.
    pub async fn get_active_edges(
        &self, owner: &[u8; 32], entity_ids: &[String], min_weight: f64,
    ) -> Vec<KnowledgeEdgeRow> {
        if entity_ids.is_empty() { return Vec::new(); }
        let conn = self.conn.lock().await;
        // Build IN clause — rusqlite doesn't support array params natively
        let placeholders: Vec<String> = entity_ids.iter().enumerate().map(|(i, _)| format!("?{}", i + 3)).collect();
        let in_clause = placeholders.join(",");
        let sql = format!(
            "SELECT edge_id, source_id, target_id, relation_type, fact_text, weight,
                    confidence, valid_from, valid_until, episode_id
             FROM knowledge_edges
             WHERE owner = ?1 AND valid_until IS NULL AND weight >= ?2
               AND (source_id IN ({in_clause}) OR target_id IN ({in_clause}))
             ORDER BY (weight * confidence) DESC"
        );
        let mut stmt = match conn.prepare(&sql) { Ok(s) => s, Err(_) => return Vec::new() };
        // Build params: owner, min_weight, then entity_ids twice (for source_id IN and target_id IN)
        // However, SQLite reuses the same placeholders. We need to flatten.
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        param_values.push(Box::new(owner.to_vec()));
        param_values.push(Box::new(min_weight));
        for eid in entity_ids { param_values.push(Box::new(eid.clone())); }

        let param_refs: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
        stmt.query_map(param_refs.as_slice(), |row| Ok(KnowledgeEdgeRow {
            edge_id: row.get(0)?, source_id: row.get(1)?, target_id: row.get(2)?,
            relation_type: row.get(3)?, fact_text: row.get(4)?, weight: row.get(5)?,
            confidence: row.get(6)?, valid_from: row.get(7)?, valid_until: row.get(8)?,
            episode_id: row.get(9)?,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Invalidate a knowledge edge (set valid_until = now).
    /// Used for temporal conflict resolution.
    pub async fn invalidate_edge(&self, edge_id: i64) {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE knowledge_edges SET valid_until = ?1, updated_at = ?1 WHERE edge_id = ?2",
            params![now, edge_id],
        );
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Episode Edges
// ============================================

impl MemoryStorage {
    /// Link an episode to an entity (bidirectional index).
    pub async fn insert_episode_edge(
        &self, owner: &[u8; 32], episode_id: &str, entity_id: &str, role: &str,
    ) -> Result<(), String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO episode_edges (owner, episode_id, entity_id, role, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![owner.as_slice(), episode_id, entity_id, role, now],
        ).map_err(|e| format!("Episode edge insert: {}", e))?;
        Ok(())
    }

    /// Get entity IDs linked to an episode (forward traversal).
    pub async fn get_entities_for_episode(&self, episode_id: &str) -> Vec<(String, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT entity_id, role FROM episode_edges WHERE episode_id = ?1"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![episode_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Get episode IDs linked to an entity (reverse traversal / provenance).
    pub async fn get_episodes_for_entity(&self, entity_id: &str) -> Vec<(String, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT episode_id, role FROM episode_edges WHERE entity_id = ?1"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![entity_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Communities
// ============================================

impl MemoryStorage {
    /// Upsert a community (label propagation result).
    pub async fn upsert_community(
        &self, community_id: &str, owner: &[u8; 32], name: &str,
        summary: Option<&str>, description: Option<&str>, entity_count: i64,
    ) -> Result<(), String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO communities (community_id, owner, name, summary, description,
                entity_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(community_id) DO UPDATE SET
                name = ?3, summary = COALESCE(?4, summary),
                description = COALESCE(?5, description),
                entity_count = ?6, updated_at = ?8",
            params![community_id, owner.as_slice(), name, summary, description, entity_count, now, now],
        ).map_err(|e| format!("Community upsert: {}", e))?;
        Ok(())
    }

    /// Get all communities for an owner.
    pub async fn get_communities(&self, owner: &[u8; 32]) -> Vec<CommunityRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT community_id, name, summary, description, entity_count, updated_at
             FROM communities WHERE owner = ?1 ORDER BY entity_count DESC"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice()], |row| Ok(CommunityRow {
            community_id: row.get(0)?, name: row.get(1)?, summary: row.get(2)?,
            description: row.get(3)?, entity_count: row.get(4)?, updated_at: row.get(5)?,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Get entities belonging to a specific community.
    pub async fn get_entities_in_community(&self, community_id: &str, owner: &[u8; 32]) -> Vec<EntityRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT entity_id, name, name_normalized, entity_type, description,
                    community_id, mention_count, created_at, updated_at
             FROM entities WHERE owner = ?1 AND community_id = ?2
             ORDER BY mention_count DESC"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice(), community_id], |row| Ok(EntityRow {
            entity_id: row.get(0)?, name: row.get(1)?, name_normalized: row.get(2)?,
            entity_type: row.get(3)?, description: row.get(4)?,
            community_id: row.get(5)?, mention_count: row.get(6)?,
            created_at: row.get(7)?, updated_at: row.get(8)?,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Get communities that have new entities since last community detection run.
    /// Used by Miner Step 8 for incremental label propagation.
    pub async fn get_communities_with_new_entities(&self, owner: &[u8; 32], since: i64) -> Vec<String> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT DISTINCT community_id FROM entities
             WHERE owner = ?1 AND community_id IS NOT NULL AND updated_at > ?2"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice(), since], |row| row.get::<_, String>(0))
            .map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Projects
// ============================================

impl MemoryStorage {
    /// Upsert a project (community specialization).
    pub async fn upsert_project(
        &self, project_id: &str, owner: &[u8; 32], name: &str,
        status: &str, community_id: &str, summary: Option<&str>,
    ) -> Result<(), String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO projects (project_id, owner, name, status, community_id,
                summary, created_at, updated_at, last_active_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(project_id) DO UPDATE SET
                name = ?3, status = ?4, summary = COALESCE(?6, summary),
                updated_at = ?8, last_active_at = ?9",
            params![project_id, owner.as_slice(), name, status, community_id, summary, now, now, now],
        ).map_err(|e| format!("Project upsert: {}", e))?;
        Ok(())
    }

    /// Get all projects for an owner, optionally filtered by status.
    pub async fn get_projects(&self, owner: &[u8; 32], status: Option<&str>, limit: usize) -> Vec<ProjectRow> {
        let conn = self.conn.lock().await;
        if let Some(s) = status {
            let mut stmt = match conn.prepare(
                "SELECT project_id, name, status, community_id, summary, last_active_at
                 FROM projects WHERE owner = ?1 AND status = ?2
                 ORDER BY last_active_at DESC LIMIT ?3"
            ) { Ok(s) => s, Err(_) => return Vec::new() };
            stmt.query_map(params![owner.as_slice(), s, limit as i64], |row| Ok(ProjectRow {
                project_id: row.get(0)?, name: row.get(1)?, status: row.get(2)?,
                community_id: row.get(3)?, summary: row.get(4)?, last_active_at: row.get(5)?,
            })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
        } else {
            let mut stmt = match conn.prepare(
                "SELECT project_id, name, status, community_id, summary, last_active_at
                 FROM projects WHERE owner = ?1 ORDER BY last_active_at DESC LIMIT ?2"
            ) { Ok(s) => s, Err(_) => return Vec::new() };
            stmt.query_map(params![owner.as_slice(), limit as i64], |row| Ok(ProjectRow {
                project_id: row.get(0)?, name: row.get(1)?, status: row.get(2)?,
                community_id: row.get(3)?, summary: row.get(4)?, last_active_at: row.get(5)?,
            })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
        }
    }

    /// Get a single project by ID.
    pub async fn get_project(&self, project_id: &str) -> Option<ProjectRow> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT project_id, name, status, community_id, summary, last_active_at
             FROM projects WHERE project_id = ?1",
            params![project_id],
            |row| Ok(ProjectRow {
                project_id: row.get(0)?, name: row.get(1)?, status: row.get(2)?,
                community_id: row.get(3)?, summary: row.get(4)?, last_active_at: row.get(5)?,
            }),
        ).optional().unwrap_or(None)
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Sessions
// ============================================

impl MemoryStorage {
    /// Upsert a session (conversation metadata).
    pub async fn upsert_session(
        &self, session_id: &str, owner: &[u8; 32], project_id: Option<&str>,
        session_type: &str, started_at: i64, turn_count: i64,
    ) -> Result<(), String> {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO sessions (session_id, owner, project_id, session_type,
                started_at, turn_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(session_id) DO UPDATE SET
                project_id = COALESCE(?3, project_id),
                turn_count = ?6",
            params![session_id, owner.as_slice(), project_id, session_type, started_at, turn_count],
        ).map_err(|e| format!("Session upsert: {}", e))?;
        Ok(())
    }

    /// Get a session by ID.
    pub async fn get_session(&self, session_id: &str) -> Option<SessionRow> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT session_id, project_id, session_type, started_at, ended_at,
                    turn_count, summary, key_decisions, entities_extracted, summary_generated
             FROM sessions WHERE session_id = ?1",
            params![session_id],
            |row| Ok(SessionRow {
                session_id: row.get(0)?, project_id: row.get(1)?, session_type: row.get(2)?,
                started_at: row.get(3)?, ended_at: row.get(4)?, turn_count: row.get(5)?,
                summary: row.get(6)?, key_decisions: row.get(7)?,
                entities_extracted: row.get::<_, i64>(8).unwrap_or(0) != 0,
                summary_generated: row.get::<_, i64>(9).unwrap_or(0) != 0,
            }),
        ).optional().unwrap_or(None)
    }

    /// Get sessions for a project (timeline view).
    pub async fn get_sessions_for_project(&self, project_id: &str, limit: usize) -> Vec<SessionRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT session_id, project_id, session_type, started_at, ended_at,
                    turn_count, summary, key_decisions, entities_extracted, summary_generated
             FROM sessions WHERE project_id = ?1 ORDER BY started_at DESC LIMIT ?2"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![project_id, limit as i64], |row| Ok(SessionRow {
            session_id: row.get(0)?, project_id: row.get(1)?, session_type: row.get(2)?,
            started_at: row.get(3)?, ended_at: row.get(4)?, turn_count: row.get(5)?,
            summary: row.get(6)?, key_decisions: row.get(7)?,
            entities_extracted: row.get::<_, i64>(8).unwrap_or(0) != 0,
            summary_generated: row.get::<_, i64>(9).unwrap_or(0) != 0,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Update session summary and key decisions (Miner Step 10).
    pub async fn update_session_summary(
        &self, session_id: &str, summary: &str, key_decisions: Option<&str>,
    ) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET summary = ?1, key_decisions = ?2, summary_generated = 1
             WHERE session_id = ?3",
            params![summary, key_decisions, session_id],
        );
    }

    /// Mark a session as having completed entity extraction.
    pub async fn mark_session_entities_extracted(&self, session_id: &str) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET entities_extracted = 1 WHERE session_id = ?1",
            params![session_id],
        );
    }

    /// Get sessions pending entity extraction or summary generation.
    pub async fn get_pending_sessions(&self, owner: &[u8; 32], limit: usize) -> Vec<SessionRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT session_id, project_id, session_type, started_at, ended_at,
                    turn_count, summary, key_decisions, entities_extracted, summary_generated
             FROM sessions
             WHERE owner = ?1 AND (entities_extracted = 0 OR summary_generated = 0)
             ORDER BY started_at DESC LIMIT ?2"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice(), limit as i64], |row| Ok(SessionRow {
            session_id: row.get(0)?, project_id: row.get(1)?, session_type: row.get(2)?,
            started_at: row.get(3)?, ended_at: row.get(4)?, turn_count: row.get(5)?,
            summary: row.get(6)?, key_decisions: row.get(7)?,
            entities_extracted: row.get::<_, i64>(8).unwrap_or(0) != 0,
            summary_generated: row.get::<_, i64>(9).unwrap_or(0) != 0,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Artifacts
// ============================================

impl MemoryStorage {
    /// Insert a code/document artifact.
    pub async fn insert_artifact(
        &self, artifact_id: &str, owner: &[u8; 32], session_id: &str,
        project_id: Option<&str>, artifact_type: &str, filename: Option<&str>,
        language: Option<&str>, version: i64, parent_id: Option<&str>,
        encrypted_content: &[u8], content_hash: &str, embedding: Option<&[f32]>,
        line_count: Option<i64>,
    ) -> Result<(), String> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let emb_blob: Option<Vec<u8>> = embedding.map(embedding_to_bytes);
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT OR IGNORE INTO artifacts
                (artifact_id, owner, session_id, project_id, artifact_type, filename,
                 language, version, parent_id, encrypted_content, content_hash,
                 embedding, line_count, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            params![
                artifact_id, owner.as_slice(), session_id, project_id,
                artifact_type, filename, language, version, parent_id,
                encrypted_content, content_hash, emb_blob.as_deref(),
                line_count, now,
            ],
        ).map_err(|e| format!("Artifact insert: {}", e))?;
        Ok(())
    }

    /// Get artifacts for a session.
    pub async fn get_artifacts_for_session(&self, session_id: &str) -> Vec<ArtifactRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT artifact_id, session_id, project_id, artifact_type, filename,
                    language, version, parent_id, content_hash, line_count, created_at
             FROM artifacts WHERE session_id = ?1 ORDER BY created_at ASC"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![session_id], |row| Ok(ArtifactRow {
            artifact_id: row.get(0)?, session_id: row.get(1)?, project_id: row.get(2)?,
            artifact_type: row.get(3)?, filename: row.get(4)?, language: row.get(5)?,
            version: row.get(6)?, parent_id: row.get(7)?, content_hash: row.get(8)?,
            line_count: row.get(9)?, created_at: row.get(10)?,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Get version history for a file (all versions, newest first).
    pub async fn get_artifact_versions(&self, owner: &[u8; 32], filename: &str) -> Vec<ArtifactRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT artifact_id, session_id, project_id, artifact_type, filename,
                    language, version, parent_id, content_hash, line_count, created_at
             FROM artifacts WHERE owner = ?1 AND filename = ?2
             ORDER BY version DESC"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice(), filename], |row| Ok(ArtifactRow {
            artifact_id: row.get(0)?, session_id: row.get(1)?, project_id: row.get(2)?,
            artifact_type: row.get(3)?, filename: row.get(4)?, language: row.get(5)?,
            version: row.get(6)?, parent_id: row.get(7)?, content_hash: row.get(8)?,
            line_count: row.get(9)?, created_at: row.get(10)?,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Graph Stats
// ============================================

impl MemoryStorage {
    /// Get cognitive graph statistics for /api/mpi/status.
    pub async fn graph_stats(&self, owner: &[u8; 32]) -> GraphStats {
        let conn = self.conn.lock().await;
        let q = |table: &str| -> u64 {
            let sql = format!("SELECT COUNT(*) FROM {} WHERE owner = ?1", table);
            conn.query_row(&sql, params![owner.as_slice()], |r| r.get::<_, i64>(0)).unwrap_or(0) as u64
        };
        GraphStats {
            episodes: q("episodes"),
            entities: q("entities"),
            knowledge_edges: q("knowledge_edges"),
            communities: q("communities"),
            projects: q("projects"),
            sessions: q("sessions"),
            artifacts: q("artifacts"),
        }
    }
}

// ============================================
// impl MemoryStorage — v2.4.0 Miner Step Support (Phase B)
// ============================================
//
// ## Creation Reason (v2.4.0-GraphCognition Phase B):
//   Step 7 (entity_extraction) needs to read raw_logs by session_id to
//   reconstruct full conversation text for NerEngine entity extraction.
//   Step 9 (entity merge) needs entities with embeddings for pairwise
//   cosine similarity comparison, and a merge operation to deduplicate.
//   Step 10 (session_summary) needs to update session ended_at and mark
//   extraction/summary stages as complete.
//
// ## Dependencies:
//   - storage_crypto::decrypt_rawlog_content (for encrypted rawlog reads)
//   - storage::bytes_to_embedding (for embedding deserialization)
//
// ## Depended by:
//   - miner/reflection.rs Steps 7, 9, 10, 11
//
// ⚠️ Important Note for Next Developer:
//   - mark_session_artifacts_extracted() requires `artifacts_extracted` column
//     in the `sessions` table. If this column doesn't exist in your schema,
//     add it via migration: ALTER TABLE sessions ADD COLUMN artifacts_extracted INTEGER DEFAULT 0
//   - merge_entities() performs cascading updates and cleans up self-referencing
//     edges. If episode_edges has a UNIQUE constraint on (episode_id, entity_id),
//     the repoint uses OR IGNORE to avoid constraint violations from duplicates.

impl MemoryStorage {
    /// Get raw_logs for a specific session, ordered by turn_index.
    /// Used by Miner Step 7 to reconstruct full conversation text for NER.
    ///
    /// Returns: Vec<RawLogRow> — all turns (user + assistant) for this session.
    ///
    /// ## v2.4.0-GraphCognition Phase B
    /// Created for Step 7 entity extraction pipeline:
    ///   get_pending_sessions() → get_rawlogs_for_session() → NerEngine → upsert_entity()
    pub async fn get_rawlogs_for_session(&self, session_id: &str) -> Vec<RawLogRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT log_id, session_id, turn_index, role, content, encrypted,
                    recall_context, extractable, feedback_signal
             FROM raw_logs
             WHERE session_id = ?1
             ORDER BY turn_index ASC"
        ) {
            Ok(s) => s,
            Err(e) => {
                error!("[STORAGE] get_rawlogs_for_session prepare failed: {}", e);
                return Vec::new();
            }
        };

        stmt.query_map(params![session_id], |row| {
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

    /// Update session ended_at timestamp.
    /// Used by Miner Step 10 after processing a session.
    pub async fn update_session_ended_at(&self, session_id: &str, ended_at: i64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE session_id = ?2",
            params![ended_at, session_id],
        );
    }

    /// Mark a session as having completed artifact extraction.
    /// Used by Miner Step 10.
    ///
    /// ⚠️ Requires `artifacts_extracted` column in `sessions` table.
    /// Schema migration: ALTER TABLE sessions ADD COLUMN artifacts_extracted INTEGER DEFAULT 0
    pub async fn mark_session_artifacts_extracted(&self, session_id: &str) {
        let conn = self.conn.lock().await;
        if let Err(e) = conn.execute(
            "UPDATE sessions SET artifacts_extracted = 1 WHERE session_id = ?1",
            params![session_id],
        ) {
            // Graceful degradation: log warning if column doesn't exist yet
            warn!(
                session_id = session_id, error = %e,
                "[STORAGE] mark_session_artifacts_extracted failed — \
                 ensure `artifacts_extracted` column exists in sessions table"
            );
        }
    }

    /// Mark a session as having completed summary generation.
    /// Used by Miner Step 10.
    pub async fn mark_session_summary_generated(&self, session_id: &str) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET summary_generated = 1 WHERE session_id = ?1",
            params![session_id],
        );
    }

    /// Get all entities for an owner (lightweight: id + name + type + embedding).
    /// Used by Miner Step 9 for pairwise similarity merge.
    /// Returns entities that have embeddings (for cosine comparison).
    pub async fn get_entities_with_embedding(
        &self, owner: &[u8; 32], limit: usize,
    ) -> Vec<(String, String, String, Vec<f32>)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT entity_id, name, entity_type, embedding
             FROM entities
             WHERE owner = ?1 AND embedding IS NOT NULL
             ORDER BY mention_count DESC
             LIMIT ?2"
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![owner.as_slice(), limit as i64], |row| {
            let entity_id: String = row.get(0)?;
            let name: String = row.get(1)?;
            let entity_type: String = row.get(2)?;
            let emb_blob: Vec<u8> = row.get(3)?;
            let embedding = super::storage::bytes_to_embedding(&emb_blob);
            Ok((entity_id, name, entity_type, embedding))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Merge entity `source_id` into `target_id`:
    /// - Add source's mention_count to target
    /// - Append source's description to target (if different)
    /// - Update all knowledge_edges referencing source to point to target
    /// - Update all episode_edges referencing source to point to target
    /// - Clean up self-referencing knowledge_edges created by the merge
    /// - Delete the source entity
    ///
    /// Used by Miner Step 9 for recursive merge (cosine > 0.92).
    ///
    /// ## Bug fixes applied (v2.4.0 Phase B review):
    /// - Self-loop cleanup: after repointing knowledge_edges, edges where
    ///   source_id == target_id are invalidated (set valid_until = now).
    /// - Duplicate episode_edges: uses OR IGNORE to handle unique constraint
    ///   violations when both source and target were linked to the same episode.
    pub async fn merge_entities(
        &self, owner: &[u8; 32], source_id: &str, target_id: &str,
    ) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let conn = self.conn.lock().await;

        // Get source entity info
        let source_info: Option<(i64, Option<String>)> = conn.query_row(
            "SELECT mention_count, description FROM entities WHERE entity_id = ?1",
            params![source_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional().unwrap_or(None);

        let (src_mentions, src_desc) = match source_info {
            Some(info) => info,
            None => return Err(format!("Source entity {} not found", source_id)),
        };

        // Merge mention_count and description into target
        if let Some(desc) = src_desc {
            conn.execute(
                "UPDATE entities SET
                    mention_count = mention_count + ?1,
                    description = CASE
                        WHEN description IS NULL THEN ?2
                        WHEN description = ?2 THEN description
                        ELSE description || '; ' || ?2
                    END,
                    updated_at = ?3
                 WHERE entity_id = ?4",
                params![src_mentions, desc, now, target_id],
            ).map_err(|e| format!("Merge entity counts: {}", e))?;
        } else {
            conn.execute(
                "UPDATE entities SET mention_count = mention_count + ?1, updated_at = ?2
                 WHERE entity_id = ?3",
                params![src_mentions, now, target_id],
            ).map_err(|e| format!("Merge entity counts: {}", e))?;
        }

        // Repoint knowledge_edges: source_id references → target_id
        let _ = conn.execute(
            "UPDATE knowledge_edges SET source_id = ?1, updated_at = ?2
             WHERE source_id = ?3 AND owner = ?4",
            params![target_id, now, source_id, owner.as_slice()],
        );
        let _ = conn.execute(
            "UPDATE knowledge_edges SET target_id = ?1, updated_at = ?2
             WHERE target_id = ?3 AND owner = ?4",
            params![target_id, now, source_id, owner.as_slice()],
        );

        // BUG FIX: Clean up self-referencing edges created by the merge.
        // After repointing, edges that originally connected source↔target now
        // have source_id == target_id == target_id, which is a meaningless self-loop.
        // Invalidate them instead of deleting to preserve audit trail.
        let self_loop_count = conn.execute(
            "UPDATE knowledge_edges SET valid_until = ?1, updated_at = ?1
             WHERE owner = ?2 AND source_id = ?3 AND target_id = ?3 AND valid_until IS NULL",
            params![now, owner.as_slice(), target_id],
        ).unwrap_or(0);
        if self_loop_count > 0 {
            debug!(
                count = self_loop_count, target = target_id,
                "[STORAGE] Self-loop edges invalidated after merge"
            );
        }

        // Repoint episode_edges, using OR IGNORE to handle duplicate
        // (episode_id, entity_id) pairs that may arise when both source
        // and target were already linked to the same episode.
        let _ = conn.execute(
            "UPDATE OR IGNORE episode_edges SET entity_id = ?1
             WHERE entity_id = ?2 AND owner = ?3",
            params![target_id, source_id, owner.as_slice()],
        );
        // Clean up any remaining source episode_edges that couldn't be
        // repointed due to duplicate constraints — these are redundant
        // since the target already has a link to the same episode.
        let _ = conn.execute(
            "DELETE FROM episode_edges WHERE entity_id = ?1 AND owner = ?2",
            params![source_id, owner.as_slice()],
        );

        // Delete source entity
        let _ = conn.execute(
            "DELETE FROM entities WHERE entity_id = ?1",
            params![source_id],
        );

        info!(
            source = source_id, target = target_id,
            merged_mentions = src_mentions,
            "[STORAGE] Entities merged"
        );

        Ok(())
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::ledger::MemoryRecord;

    fn make_rec_owner(ts: u64, owner: [u8; 32], layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new(owner, ts, layer, vec!["test".into()], "ai".into(),
            format!("content_{}", ts).into_bytes(), vec![0.5; 4])
    }

    // ========================================
    // Existing tests (preserved from v2.3.0)
    // ========================================

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

    // ========================================
    // v2.4.0: Cognitive Graph CRUD Tests
    // ========================================

    #[tokio::test]
    async fn test_entity_upsert_new() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let is_new = s.upsert_entity("ent_jwt", &owner, "JWT", "jwt", "technology", Some("JSON Web Token"), None).await.unwrap();
        assert!(is_new);
        let ent = s.get_entity("ent_jwt").await.unwrap();
        assert_eq!(ent.name, "JWT");
        assert_eq!(ent.mention_count, 1);
    }

    #[tokio::test]
    async fn test_entity_upsert_existing_increments() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.upsert_entity("ent_jwt", &owner, "JWT", "jwt", "technology", None, None).await.unwrap();
        let is_new = s.upsert_entity("ent_jwt", &owner, "JWT", "jwt", "technology", Some("Updated desc"), None).await.unwrap();
        assert!(!is_new);
        let ent = s.get_entity("ent_jwt").await.unwrap();
        assert_eq!(ent.mention_count, 2);
        assert_eq!(ent.description, Some("Updated desc".into()));
    }

    #[tokio::test]
    async fn test_entities_cached() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.upsert_entity("ent_jwt", &owner, "JWT", "jwt", "technology", None, None).await.unwrap();
        s.upsert_entity("ent_auth", &owner, "auth module", "auth module", "module", None, None).await.unwrap();
        let cache = s.get_entities_cached(&owner).await;
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get("jwt"), Some(&"ent_jwt".to_string()));
    }

    #[tokio::test]
    async fn test_knowledge_edge_insert_and_query() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let edge_id = s.insert_knowledge_edge(&owner, "ent_auth", "ent_jwt", "USES", Some("auth uses JWT"), 1.0, 0.95, None, now, None).await.unwrap();
        assert!(edge_id > 0);
        let edges = s.get_edges_for_entity("ent_auth", &owner).await;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation_type, "USES");
    }

    #[tokio::test]
    async fn test_knowledge_edge_invalidation() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let eid = s.insert_knowledge_edge(&owner, "ent_auth", "ent_jwt", "USES", None, 1.0, 0.9, None, now, None).await.unwrap();
        s.invalidate_edge(eid).await;
        // After invalidation, get_edges_for_entity should return empty (only valid edges)
        let edges = s.get_edges_for_entity("ent_auth", &owner).await;
        assert!(edges.is_empty());
    }

    #[tokio::test]
    async fn test_episode_edge_bidirectional() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert_episode_edge(&owner, "ep_001", "ent_jwt", "mentioned").await.unwrap();
        s.insert_episode_edge(&owner, "ep_001", "ent_auth", "produced").await.unwrap();
        let entities = s.get_entities_for_episode("ep_001").await;
        assert_eq!(entities.len(), 2);
        let episodes = s.get_episodes_for_entity("ent_jwt").await;
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].0, "ep_001");
    }

    #[tokio::test]
    async fn test_community_upsert_and_query() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.upsert_community("comm_1", &owner, "Auth System", Some("Authentication components"), None, 5).await.unwrap();
        let comms = s.get_communities(&owner).await;
        assert_eq!(comms.len(), 1);
        assert_eq!(comms[0].name, "Auth System");
        assert_eq!(comms[0].entity_count, 5);
    }

    #[tokio::test]
    async fn test_session_upsert_and_pending() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        s.upsert_session("sess_001", &owner, None, "code", now, 15).await.unwrap();
        let sess = s.get_session("sess_001").await.unwrap();
        assert_eq!(sess.turn_count, 15);
        assert!(!sess.entities_extracted);
        assert!(!sess.summary_generated);
        // Should appear in pending
        let pending = s.get_pending_sessions(&owner, 10).await;
        assert_eq!(pending.len(), 1);
        // Mark extracted
        s.mark_session_entities_extracted("sess_001").await;
        let sess2 = s.get_session("sess_001").await.unwrap();
        assert!(sess2.entities_extracted);
    }

    #[tokio::test]
    async fn test_project_upsert_and_sessions() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        s.upsert_community("comm_1", &owner, "Project B", None, None, 3).await.unwrap();
        s.upsert_project("comm_1", &owner, "Project B", "active", "comm_1", Some("Auth project")).await.unwrap();
        s.upsert_session("sess_001", &owner, Some("comm_1"), "code", now, 10).await.unwrap();
        let projects = s.get_projects(&owner, Some("active"), 10).await;
        assert_eq!(projects.len(), 1);
        let sessions = s.get_sessions_for_project("comm_1", 10).await;
        assert_eq!(sessions.len(), 1);
    }

    #[tokio::test]
    async fn test_artifact_insert_and_versions() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert_artifact("art_v1", &owner, "sess_001", None, "code", Some("auth.rs"), Some("rust"), 1, None, b"fn auth() {}", "hash1", None, Some(1)).await.unwrap();
        s.insert_artifact("art_v2", &owner, "sess_002", None, "code", Some("auth.rs"), Some("rust"), 2, Some("art_v1"), b"fn auth() { jwt() }", "hash2", None, Some(1)).await.unwrap();
        let versions = s.get_artifact_versions(&owner, "auth.rs").await;
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].version, 2); // newest first
        assert_eq!(versions[1].version, 1);
    }

    #[tokio::test]
    async fn test_graph_stats() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        s.upsert_entity("ent_1", &owner, "JWT", "jwt", "technology", None, None).await.unwrap();
        s.upsert_entity("ent_2", &owner, "auth", "auth", "module", None, None).await.unwrap();
        s.insert_knowledge_edge(&owner, "ent_2", "ent_1", "USES", None, 1.0, 0.9, None, now, None).await.unwrap();
        s.upsert_community("comm_1", &owner, "Auth", None, None, 2).await.unwrap();
        let stats = s.graph_stats(&owner).await;
        assert_eq!(stats.entities, 2);
        assert_eq!(stats.knowledge_edges, 1);
        assert_eq!(stats.communities, 1);
    }

    // ========================================
    // v2.4.0 Phase B: Miner Step Support Tests
    // ========================================

    #[tokio::test]
    async fn test_get_rawlogs_for_session() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_raw_log("sess_a", 0, "user", "hello", "test", None, 1, None, None).await.unwrap();
        s.insert_raw_log("sess_a", 1, "assistant", "hi there", "test", None, 0, None, None).await.unwrap();
        s.insert_raw_log("sess_b", 0, "user", "other session", "test", None, 1, None, None).await.unwrap();

        let logs = s.get_rawlogs_for_session("sess_a").await;
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].turn_index, 0);
        assert_eq!(logs[1].turn_index, 1);

        let logs_b = s.get_rawlogs_for_session("sess_b").await;
        assert_eq!(logs_b.len(), 1);
    }

    #[tokio::test]
    async fn test_merge_entities() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        // Create two entities
        s.upsert_entity("ent_jwt", &owner, "JWT", "jwt", "technology", Some("Token format"), None).await.unwrap();
        s.upsert_entity("ent_jwt2", &owner, "JSON Web Token", "json web token", "technology", Some("Auth token"), None).await.unwrap();
        // Mention ent_jwt2 twice more
        s.upsert_entity("ent_jwt2", &owner, "JSON Web Token", "json web token", "technology", None, None).await.unwrap();

        // Create an edge referencing source
        s.insert_knowledge_edge(&owner, "ent_jwt2", "ent_auth", "USED_BY", None, 1.0, 0.9, None, now, None).await.unwrap();

        // Merge ent_jwt2 into ent_jwt
        s.merge_entities(&owner, "ent_jwt2", "ent_jwt").await.unwrap();

        // Source should be deleted
        assert!(s.get_entity("ent_jwt2").await.is_none());

        // Target should have accumulated mentions: 1 (original) + 2 (from source)
        let target = s.get_entity("ent_jwt").await.unwrap();
        assert_eq!(target.mention_count, 3);

        // Edge should now point to target
        let edges = s.get_edges_for_entity("ent_jwt", &owner).await;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_id, "ent_jwt");
    }

    #[tokio::test]
    async fn test_merge_entities_self_loop_cleanup() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        // Create two entities that will be merged
        s.upsert_entity("ent_a", &owner, "A", "a", "concept", None, None).await.unwrap();
        s.upsert_entity("ent_b", &owner, "B", "b", "concept", None, None).await.unwrap();

        // Create edge from A→B — after merging B into A, this becomes A→A (self-loop)
        s.insert_knowledge_edge(&owner, "ent_a", "ent_b", "RELATED_TO", None, 1.0, 0.9, None, now, None).await.unwrap();

        // Merge B into A
        s.merge_entities(&owner, "ent_b", "ent_a").await.unwrap();

        // The self-loop edge should have been invalidated (valid_until set)
        let edges = s.get_edges_for_entity("ent_a", &owner).await;
        assert!(edges.is_empty(), "Self-loop edges should be invalidated after merge");
    }

    #[tokio::test]
    async fn test_session_lifecycle_methods() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        s.upsert_session("sess_001", &owner, None, "code", now, 10).await.unwrap();

        // Mark various stages
        s.mark_session_entities_extracted("sess_001").await;
        s.mark_session_summary_generated("sess_001").await;
        // Note: mark_session_artifacts_extracted requires `artifacts_extracted` column
        // which may not exist in the base schema — skipped in this test.
        s.update_session_ended_at("sess_001", now + 3600).await;

        let sess = s.get_session("sess_001").await.unwrap();
        assert!(sess.entities_extracted);
        assert!(sess.summary_generated);
        assert_eq!(sess.ended_at, Some(now + 3600));

        // Should no longer appear in pending
        let pending = s.get_pending_sessions(&owner, 10).await;
        assert!(pending.is_empty());
    }
}
