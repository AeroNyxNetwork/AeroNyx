// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_miner.rs
// ============================================
//! # Storage Miner — Miner Step Support Methods + Entity Timeline
//!
//! ## File Creation/Modification Notes
//! ============================================
//! Creation Reason: Split from storage_ops.rs to group all Miner pipeline
//!   support methods (Steps 7/9/10/11) and entity timeline API endpoint.
//! Modification Reason:
//!   v2.5.2+Pagination     — get_entity_timeline gains `offset: usize` param
//!   v2.5.3+ArtifactChain  — New methods for DEV-TASK-001:
//!     get_latest_artifact_version() — Miner Step 10 version chain lookup
//!     get_artifact_with_content()   — GET /artifacts/:id real implementation
//!     search_artifacts_by_filename() — GET /artifacts/search endpoint
//!     ArtifactWithContent struct    — return type for get_artifact_with_content
//! Main Functionality:
//!   - get_rawlogs_for_session       → Miner Step 7 (NER entity extraction)
//!   - get_entities_with_embedding   → Miner Step 9 (pairwise cosine merge)
//!   - merge_entities                → Miner Step 9 (deduplicate similar entities)
//!   - update_session_ended_at       → Miner Step 10
//!   - mark_session_summary_generated / mark_session_artifacts_extracted → Step 10
//!   - EntityTimelineEntry + get_entity_timeline → /entities/:id/timeline
//!   - get_latest_artifact_version   → Miner Step 10 (v2.5.3+ArtifactChain)
//!   - get_artifact_with_content     → GET /artifacts/:id (v2.5.3+ArtifactChain)
//!   - search_artifacts_by_filename  → GET /artifacts/search (v2.5.3+ArtifactChain)
//! Dependencies:
//!   - storage.rs — MemoryStorage struct, RawLogRow, bytes_to_embedding
//!   - storage_crypto.rs — decrypt_rawlog_content_pub
//! Depended by:
//!   - miner/reflection.rs — Steps 7, 9, 10, 11
//!   - api/mpi_graph_handlers.rs — entity timeline + artifact endpoints
//!
//! ⚠️ Important Note for Next Developer:
//! - mark_session_artifacts_extracted() requires `artifacts_extracted` column in
//!   sessions table. Schema v5 migration. If missing:
//!   ALTER TABLE sessions ADD COLUMN artifacts_extracted INTEGER DEFAULT 0
//! - merge_entities() performs cascading updates and self-loop cleanup.
//!   episode_edges uses OR IGNORE for unique constraint on (episode_id, entity_id).
//! - get_entity_timeline() offset param is v2.5.2+Pagination — must be passed
//!   from the handler; internal callers (mpi_context_inject) pass 0.
//! - get_artifact_with_content() stores plaintext (Step 10 stores code_content
//!   directly). Future encryption: decrypt here using record_key pattern.
//! - search_artifacts_by_filename() escapes LIKE special chars (%, _, \) to
//!   prevent injection via filename pattern parameter.
//! - The core logic of this file cannot be deleted or significantly modified.
//! - Maintain interface compatibility with reflection.rs and mpi_graph_handlers.rs.
//!
//! ## Modification History
//! v2.4.0-GraphCognition Phase B - 🌟 get_rawlogs_for_session, update_session_ended_at,
//!   mark_session_artifacts_extracted, mark_session_summary_generated,
//!   get_entities_with_embedding, merge_entities
//! v2.4.0+Search    - 🌟 EntityTimelineEntry + get_entity_timeline()
//! v2.5.2+Pagination - 🌟 get_entity_timeline gains offset param
//! v2.5.3+ArtifactChain - 🌟 get_latest_artifact_version, get_artifact_with_content,
//!   search_artifacts_by_filename, ArtifactWithContent struct; 4 new tests
//!
//! ## Last Modified
//! v2.5.3+ArtifactChain - 🌟 DEV-TASK-001 artifact chain methods

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::{debug, info, warn};

use super::storage::{MemoryStorage, RawLogRow};

// ============================================
// v2.4.0+Search: Entity Timeline Types
// ============================================

/// Entity timeline entry for GET /api/mpi/entities/:id/timeline.
///
/// Represents a single event in an entity's history across all sessions:
/// - "mentioned"            → entity appeared in a session conversation
/// - "relation_created"     → a knowledge edge was created from/to this entity
/// - "relation_invalidated" → a knowledge edge was invalidated (temporal conflict)
///
/// ⚠️ Relation events have `session_id = ""` because knowledge_edges don't
/// directly reference sessions. Future improvement: trace via episode_id.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EntityTimelineEntry {
    pub session_id: String,
    pub session_title: Option<String>,
    pub project_name: Option<String>,
    pub started_at: i64,
    /// "mentioned" | "relation_created" | "relation_invalidated"
    pub event_type: String,
    /// Human-readable description of the event.
    pub detail: String,
    pub relation_type: Option<String>,
    pub related_entity: Option<String>,
}

// ============================================
// v2.5.3+ArtifactChain: Artifact with Content
// ============================================

/// Artifact row with decrypted content.
///
/// Returned by `get_artifact_with_content()`.
/// Content is always UTF-8 text (code artifacts store plaintext).
///
/// ## Encryption note
/// Step 10 currently stores `code_content.as_bytes()` directly (plaintext).
/// Future: if encryption is added, decrypt here using the record_key pattern.
///
/// v2.5.3+ArtifactChain
#[derive(Debug, Clone, serde::Serialize)]
pub struct ArtifactWithContent {
    pub artifact_id: String,
    pub session_id: String,
    pub project_id: Option<String>,
    pub artifact_type: String,
    pub filename: Option<String>,
    pub language: Option<String>,
    pub version: i64,
    pub parent_id: Option<String>,
    /// Decrypted plaintext code content.
    pub content: String,
    pub content_hash: String,
    pub line_count: Option<i64>,
    pub created_at: i64,
}

// ============================================
// impl MemoryStorage — Miner Step Support (Phase B)
// ============================================

impl MemoryStorage {
    /// Get raw_logs for a specific session, ordered by turn_index.
    /// Used by Miner Step 7 to reconstruct full conversation text for NER.
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
                warn!("[STORAGE] get_rawlogs_for_session prepare failed: {}", e);
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

    /// Update session ended_at timestamp. Used by Miner Step 10.
    pub async fn update_session_ended_at(&self, session_id: &str, ended_at: i64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE session_id = ?2",
            params![ended_at, session_id],
        );
    }

    /// Mark a session as having completed artifact extraction. Used by Miner Step 10.
    ///
    /// ⚠️ Requires `artifacts_extracted` column in `sessions` table.
    pub async fn mark_session_artifacts_extracted(&self, session_id: &str) {
        let conn = self.conn.lock().await;
        if let Err(e) = conn.execute(
            "UPDATE sessions SET artifacts_extracted = 1 WHERE session_id = ?1",
            params![session_id],
        ) {
            warn!(
                session_id = session_id, error = %e,
                "[STORAGE] mark_session_artifacts_extracted failed — \
                 ensure `artifacts_extracted` column exists in sessions table"
            );
        }
    }

    /// Mark a session as having completed summary generation. Used by Miner Step 10.
    pub async fn mark_session_summary_generated(&self, session_id: &str) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET summary_generated = 1 WHERE session_id = ?1",
            params![session_id],
        );
    }

    /// Get entities with embeddings for pairwise cosine merge.
    /// Used by Miner Step 9. Returns (entity_id, name, type, embedding).
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

    /// Merge entity `source_id` into `target_id`.
    ///
    /// Operations:
    /// 1. Add source's mention_count to target
    /// 2. Append source's description to target (if different)
    /// 3. Repoint all knowledge_edges source→target
    /// 4. Invalidate self-loops created by the merge
    /// 5. Repoint episode_edges (OR IGNORE for unique constraint)
    /// 6. Delete orphaned source episode_edges
    /// 7. Delete the source entity
    pub async fn merge_entities(
        &self, owner: &[u8; 32], source_id: &str, target_id: &str,
    ) -> Result<(), String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let conn = self.conn.lock().await;

        let source_info: Option<(i64, Option<String>)> = conn.query_row(
            "SELECT mention_count, description FROM entities WHERE entity_id = ?1",
            params![source_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional().unwrap_or(None);

        let (src_mentions, src_desc) = match source_info {
            Some(info) => info,
            None => return Err(format!("Source entity {} not found", source_id)),
        };

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

        // Invalidate self-loops created by the merge (source↔target → target↔target)
        let self_loop_count = conn.execute(
            "UPDATE knowledge_edges SET valid_until = ?1, updated_at = ?1
             WHERE owner = ?2 AND source_id = ?3 AND target_id = ?3 AND valid_until IS NULL",
            params![now, owner.as_slice(), target_id],
        ).unwrap_or(0);
        if self_loop_count > 0 {
            debug!(count = self_loop_count, target = target_id,
                "[STORAGE] Self-loop edges invalidated after merge");
        }

        let _ = conn.execute(
            "UPDATE OR IGNORE episode_edges SET entity_id = ?1
             WHERE entity_id = ?2 AND owner = ?3",
            params![target_id, source_id, owner.as_slice()],
        );
        let _ = conn.execute(
            "DELETE FROM episode_edges WHERE entity_id = ?1 AND owner = ?2",
            params![source_id, owner.as_slice()],
        );
        let _ = conn.execute(
            "DELETE FROM entities WHERE entity_id = ?1",
            params![source_id],
        );

        info!(source = source_id, target = target_id,
            merged_mentions = src_mentions, "[STORAGE] Entities merged");

        Ok(())
    }
}

// ============================================
// impl MemoryStorage — Entity Timeline (v2.4.0+Search)
// ============================================

impl MemoryStorage {
    /// Get timeline of events for an entity across all sessions.
    ///
    /// Returns events sorted by started_at ASC (chronological).
    ///
    /// ## v2.5.2+Pagination
    /// `offset` parameter added. Pass `0` for non-paginated callers
    /// (e.g. mpi_context_inject internal call).
    ///
    /// ## Event Types
    /// - "mentioned"            → entity appeared in a session
    /// - "relation_created"     → knowledge edge was created
    /// - "relation_invalidated" → knowledge edge was invalidated
    ///
    /// ⚠️ Relation events have `session_id = ""` — knowledge_edges don't
    /// directly reference sessions. TODO Phase C: trace via episode_id.
    pub async fn get_entity_timeline(
        &self,
        entity_id: &str,
        owner: &[u8; 32],
        limit: usize,
        offset: usize,
    ) -> Vec<EntityTimelineEntry> {
        let conn = self.conn.lock().await;
        let mut events: Vec<EntityTimelineEntry> = Vec::new();

        // ── Part 1: Mentions ──
        {
            let mut stmt = match conn.prepare(
                "SELECT DISTINCT s.session_id, s.title, s.started_at, s.project_id, ee.role
                 FROM episode_edges ee
                 JOIN episodes e ON e.episode_id = ee.episode_id
                 JOIN sessions s ON s.session_id = e.session_id
                 WHERE ee.entity_id = ?1 AND ee.owner = ?2
                 ORDER BY s.started_at ASC
                 LIMIT ?3 OFFSET ?4"
            ) {
                Ok(s) => s,
                Err(_) => return events,
            };

            let rows: Vec<(String, Option<String>, i64, Option<String>, String)> = stmt
                .query_map(
                    params![entity_id, owner.as_slice(), limit as i64, offset as i64],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
                )
                .map(|r| r.filter_map(|x| x.ok()).collect())
                .unwrap_or_default();

            for (sid, title, started_at, project_id, role) in rows {
                let project_name: Option<String> = project_id.and_then(|pid| {
                    conn.query_row(
                        "SELECT name FROM projects WHERE project_id = ?1",
                        params![pid], |row| row.get(0),
                    ).ok()
                });

                events.push(EntityTimelineEntry {
                    session_id: sid,
                    session_title: title,
                    project_name,
                    started_at,
                    event_type: "mentioned".to_string(),
                    detail: format!("Role: {}", role),
                    relation_type: None,
                    related_entity: None,
                });
            }
        }

        // ── Part 2: Relations ──
        {
            let mut stmt = match conn.prepare(
                "SELECT ke.relation_type, ke.fact_text, ke.valid_from, ke.valid_until,
                        ke.source_id, ke.target_id,
                        COALESCE(es.name, ke.source_id) as src_name,
                        COALESCE(et.name, ke.target_id) as tgt_name
                 FROM knowledge_edges ke
                 LEFT JOIN entities es ON es.entity_id = ke.source_id
                 LEFT JOIN entities et ON et.entity_id = ke.target_id
                 WHERE ke.owner = ?1 AND (ke.source_id = ?2 OR ke.target_id = ?2)
                 ORDER BY ke.valid_from ASC
                 LIMIT ?3"
            ) {
                Ok(s) => s,
                Err(_) => return events,
            };

            let rows: Vec<(String, Option<String>, i64, Option<i64>, String, String, String, String)> = stmt
                .query_map(
                    params![owner.as_slice(), entity_id, limit as i64],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                              row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?))
                )
                .map(|r| r.filter_map(|x| x.ok()).collect())
                .unwrap_or_default();

            for (rel_type, fact_text, valid_from, valid_until,
                 src_id, _tgt_id, src_name, tgt_name) in rows
            {
                let (other_name, direction) = if src_id == entity_id {
                    (tgt_name, format!("→ {}", rel_type))
                } else {
                    (src_name, format!("{} ←", rel_type))
                };

                let event_type = if valid_until.is_some() {
                    "relation_invalidated"
                } else {
                    "relation_created"
                };

                let detail = fact_text.unwrap_or_else(|| {
                    format!("{} {}", direction, other_name)
                });

                events.push(EntityTimelineEntry {
                    session_id: String::new(),
                    session_title: None,
                    project_name: None,
                    started_at: valid_from,
                    event_type: event_type.to_string(),
                    detail,
                    relation_type: Some(rel_type),
                    related_entity: Some(other_name),
                });
            }
        }

        events.sort_by_key(|e| e.started_at);
        events
    }
}

// ============================================
// impl MemoryStorage — v2.5.3+ArtifactChain
// ============================================

impl MemoryStorage {
    /// Get the latest artifact version for a given owner + filename.
    ///
    /// Returns `(version, artifact_id)`:
    /// - `(0, None)` if no artifact with this filename exists yet (first version)
    /// - `(n, Some(id))` if version n exists (next insert should use version n+1)
    ///
    /// Used by Miner Step 10 to build version chains across sessions.
    ///
    /// v2.5.3+ArtifactChain
    pub async fn get_latest_artifact_version(
        &self,
        owner: &[u8; 32],
        filename: &str,
    ) -> (i64, Option<String>) {
        let conn = self.conn.lock().await;
        let result: Option<(i64, String)> = conn.query_row(
            "SELECT version, artifact_id FROM artifacts
             WHERE owner = ?1 AND filename = ?2
             ORDER BY version DESC
             LIMIT 1",
            params![owner.as_slice(), filename],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional().unwrap_or(None);

        match result {
            Some((v, id)) => (v, Some(id)),
            None => (0, None),
        }
    }

    /// Get artifact metadata + decrypted content by artifact_id.
    ///
    /// Returns `None` if artifact not found or owner mismatch.
    /// Content is returned as UTF-8 string (code is always text).
    ///
    /// Used by `GET /api/mpi/artifacts/:id` endpoint.
    ///
    /// v2.5.3+ArtifactChain
    pub async fn get_artifact_with_content(
        &self,
        artifact_id: &str,
        owner: &[u8; 32],
    ) -> Option<ArtifactWithContent> {
        let conn = self.conn.lock().await;

        conn.query_row(
            "SELECT artifact_id, session_id, project_id, artifact_type,
                    filename, language, version, parent_id,
                    encrypted_content, content_hash, line_count, created_at
             FROM artifacts
             WHERE artifact_id = ?1 AND owner = ?2",
            params![artifact_id, owner.as_slice()],
            |row| {
                let raw_content: Vec<u8> = row.get(8)?;
                // Step 10 stores code_content.as_bytes() directly (plaintext).
                // Future: if encryption is added, decrypt here using record_key.
                let content = String::from_utf8(raw_content)
                    .unwrap_or_else(|_| String::from("[binary content]"));

                Ok(ArtifactWithContent {
                    artifact_id: row.get(0)?,
                    session_id: row.get(1)?,
                    project_id: row.get(2)?,
                    artifact_type: row.get(3)?,
                    filename: row.get(4)?,
                    language: row.get(5)?,
                    version: row.get(6)?,
                    parent_id: row.get(7)?,
                    content,
                    content_hash: row.get(9)?,
                    line_count: row.get(10)?,
                    created_at: row.get(11)?,
                })
            },
        ).optional().unwrap_or(None)
    }

    /// Search artifacts by filename pattern (LIKE %pattern%) for an owner.
    ///
    /// Returns all versions matching the filename, ordered by filename ASC,
    /// version DESC (latest version first within each file).
    ///
    /// Used by `GET /api/mpi/artifacts/search?filename=auth.rs`.
    ///
    /// ## LIKE injection prevention
    /// Escapes `%`, `_`, `\` in the pattern before building the LIKE clause.
    ///
    /// v2.5.3+ArtifactChain
    pub async fn search_artifacts_by_filename(
        &self,
        owner: &[u8; 32],
        filename_pattern: &str,
        limit: usize,
        offset: usize,
    ) -> Vec<crate::services::memchain::ArtifactRow> {
        if filename_pattern.trim().is_empty() {
            return Vec::new();
        }

        // Escape LIKE special chars to prevent injection via filename param
        let escaped = filename_pattern
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_");
        let pattern = format!("%{}%", escaped.to_lowercase());

        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT artifact_id, session_id, project_id, artifact_type, filename,
                    language, version, parent_id, content_hash, line_count, created_at
             FROM artifacts
             WHERE owner = ?1 AND lower(COALESCE(filename, '')) LIKE ?2 ESCAPE '\\'
             ORDER BY filename ASC, version DESC
             LIMIT ?3 OFFSET ?4"
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("[STORAGE] search_artifacts_by_filename prepare failed: {}", e);
                return Vec::new();
            }
        };

        stmt.query_map(
            params![owner.as_slice(), pattern, limit as i64, offset as i64],
            |row| Ok(crate::services::memchain::ArtifactRow {
                artifact_id: row.get(0)?,
                session_id: row.get(1)?,
                project_id: row.get(2)?,
                artifact_type: row.get(3)?,
                filename: row.get(4)?,
                language: row.get(5)?,
                version: row.get(6)?,
                parent_id: row.get(7)?,
                content_hash: row.get(8)?,
                line_count: row.get(9)?,
                created_at: row.get(10)?,
            }),
        )
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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

        s.upsert_entity("ent_jwt", &owner, "JWT", "jwt", "technology", Some("Token format"), None).await.unwrap();
        s.upsert_entity("ent_jwt2", &owner, "JSON Web Token", "json web token", "technology", Some("Auth token"), None).await.unwrap();
        s.upsert_entity("ent_jwt2", &owner, "JSON Web Token", "json web token", "technology", None, None).await.unwrap();

        s.insert_knowledge_edge(&owner, "ent_jwt2", "ent_auth", "USED_BY", None, 1.0, 0.9, None, now, None).await.unwrap();

        s.merge_entities(&owner, "ent_jwt2", "ent_jwt").await.unwrap();

        assert!(s.get_entity("ent_jwt2").await.is_none());

        let target = s.get_entity("ent_jwt").await.unwrap();
        assert_eq!(target.mention_count, 3);

        let edges = s.get_edges_for_entity("ent_jwt", &owner).await;
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source_id, "ent_jwt");
    }

    #[tokio::test]
    async fn test_merge_entities_self_loop_cleanup() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        s.upsert_entity("ent_a", &owner, "A", "a", "concept", None, None).await.unwrap();
        s.upsert_entity("ent_b", &owner, "B", "b", "concept", None, None).await.unwrap();

        s.insert_knowledge_edge(&owner, "ent_a", "ent_b", "RELATED_TO", None, 1.0, 0.9, None, now, None).await.unwrap();

        s.merge_entities(&owner, "ent_b", "ent_a").await.unwrap();

        let edges = s.get_edges_for_entity("ent_a", &owner).await;
        assert!(edges.is_empty(), "Self-loop edges should be invalidated after merge");
    }

    #[tokio::test]
    async fn test_session_lifecycle_methods() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        s.upsert_session("sess_001", &owner, None, "code", now, 10).await.unwrap();

        s.mark_session_entities_extracted("sess_001").await;
        s.mark_session_summary_generated("sess_001").await;
        s.update_session_ended_at("sess_001", now + 3600).await;

        let sess = s.get_session("sess_001").await.unwrap();
        assert!(sess.entities_extracted);
        assert!(sess.summary_generated);
        assert_eq!(sess.ended_at, Some(now + 3600));

        let pending = s.get_pending_sessions(&owner, 10).await;
        assert!(pending.is_empty());
    }

    // ── v2.5.3+ArtifactChain tests ──

    #[tokio::test]
    async fn test_get_latest_artifact_version_empty() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        // No artifacts exist → should return (0, None)
        let (v, id) = s.get_latest_artifact_version(&owner, "auth.rs").await;
        assert_eq!(v, 0);
        assert!(id.is_none());
    }

    #[tokio::test]
    async fn test_get_latest_artifact_version_chain() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        s.insert_artifact(
            "art_v1", &owner, "sess_001", None, "code",
            Some("auth.rs"), Some("rust"), 1, None,
            b"fn auth() {}", "hash1", None, Some(1),
        ).await.unwrap();

        let (v, id) = s.get_latest_artifact_version(&owner, "auth.rs").await;
        assert_eq!(v, 1);
        assert_eq!(id, Some("art_v1".to_string()));

        s.insert_artifact(
            "art_v2", &owner, "sess_002", None, "code",
            Some("auth.rs"), Some("rust"), 2, Some("art_v1"),
            b"fn auth() { jwt() }", "hash2", None, Some(1),
        ).await.unwrap();

        let (v2, id2) = s.get_latest_artifact_version(&owner, "auth.rs").await;
        assert_eq!(v2, 2);
        assert_eq!(id2, Some("art_v2".to_string()));
    }

    #[tokio::test]
    async fn test_get_artifact_with_content() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        s.insert_artifact(
            "art_001", &owner, "sess_001", None, "code",
            Some("main.rs"), Some("rust"), 1, None,
            b"fn main() { println!(\"hello\"); }", "abc123",
            None, Some(1),
        ).await.unwrap();

        let result = s.get_artifact_with_content("art_001", &owner).await;
        assert!(result.is_some());
        let art = result.unwrap();
        assert_eq!(art.filename, Some("main.rs".to_string()));
        assert_eq!(art.version, 1);
        assert!(art.content.contains("println"));

        // Other owner must not access
        let other_owner = [0xBB; 32];
        assert!(s.get_artifact_with_content("art_001", &other_owner).await.is_none());
    }

    #[tokio::test]
    async fn test_search_artifacts_by_filename() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];

        s.insert_artifact("art_1", &owner, "s1", None, "code",
            Some("auth.rs"), Some("rust"), 1, None, b"v1", "h1", None, None).await.unwrap();
        s.insert_artifact("art_2", &owner, "s2", None, "code",
            Some("auth.rs"), Some("rust"), 2, Some("art_1"), b"v2", "h2", None, None).await.unwrap();
        s.insert_artifact("art_3", &owner, "s3", None, "code",
            Some("main.rs"), Some("rust"), 1, None, b"main", "h3", None, None).await.unwrap();

        // "auth" matches auth.rs (both versions)
        let results = s.search_artifacts_by_filename(&owner, "auth", 10, 0).await;
        assert_eq!(results.len(), 2);

        // "main" matches only main.rs
        let results2 = s.search_artifacts_by_filename(&owner, "main", 10, 0).await;
        assert_eq!(results2.len(), 1);
        assert_eq!(results2[0].filename, Some("main.rs".to_string()));

        // offset beyond results → empty
        let results3 = s.search_artifacts_by_filename(&owner, "auth", 10, 99).await;
        assert!(results3.is_empty());

        // empty pattern → empty (fast-path guard)
        let results4 = s.search_artifacts_by_filename(&owner, "", 10, 0).await;
        assert!(results4.is_empty());
    }
}
