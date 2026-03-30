// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_graph.rs
// ============================================
//! # Storage Graph — Cognitive Graph CRUD Operations
//!
//! ## Creation Reason (v2.4.0+Search)
//! Split out from storage_ops.rs (which was too large) to improve
//! maintainability. Contains all CRUD for the v2.4.0 cognitive graph tables.
//!
//! ## Modification Reason (v2.5.2+Pagination)
//! Added `offset: usize` parameter to `get_sessions_for_project` and
//! `get_entity_timeline` to support cursor-free pagination for API consumers.
//! All internal callers that do not paginate pass `offset = 0`.
//!
//! ## Main Functionality
//! - Episodes: upsert_episode, get_episode, get_episodes_for_session
//! - Entities: upsert_entity, get_entity, get_entities_by_owner, get_entities_cached,
//!   update_entity_community
//! - Knowledge Edges: insert_knowledge_edge, get_active_edges, get_edges_for_entity,
//!   invalidate_edge, get_edges_within_community
//! - Episode Edges: insert_episode_edge, get_entities_for_episode, get_episodes_for_entity
//! - Communities: upsert_community, get_communities, get_entities_in_community,
//!   get_communities_with_new_entities
//! - Projects: upsert_project, get_projects, get_project
//! - Sessions: upsert_session, get_session, get_sessions_for_project (paginated),
//!   update_session_summary (v2.4.0+Search: + title param), get_pending_sessions
//! - Entity Timeline: get_entity_timeline (paginated) — see also storage_miner.rs
//! - Artifacts: insert_artifact, get_artifacts_for_session, get_artifact_versions
//! - Graph Stats: graph_stats
//!
//! ## Architecture
//! This file uses `impl MemoryStorage` blocks (same as storage_ops.rs).
//! Rust allows multiple impl blocks across files within the same crate.
//!
//! ## Key Type Definitions
//! All public types (EntityRow, SessionRow, KnowledgeEdgeRow, etc.) are defined
//! here and re-exported via memchain/mod.rs for use by API handlers.
//!
//! ## Calling Relationships
//! - mpi_graph_handlers.rs::mpi_project_timeline()  → get_sessions_for_project(..., params.offset)
//! - mpi_graph_handlers.rs::mpi_entity_timeline()   → get_entity_timeline(..., params.offset)
//! - mpi_graph_handlers.rs::mpi_context_inject()    → get_sessions_for_project(..., 0)
//!
//! ⚠️ Important Notes for Next Developer:
//! - knowledge_edges.valid_until = NULL means "currently valid".
//!   Always filter by valid_until IS NULL for current state queries.
//! - entity_id = SHA256(owner || name_normalized) — deterministic, enables upsert.
//! - v2.4.0+Search: SessionRow now includes `title: Option<String>`.
//!   update_session_summary() now accepts a 4th param `title: Option<&str>`.
//!   Both changes require a `sessions.title` column — added in Schema v5 migration.
//! - sessions.title is COALESCE-guarded: passing None preserves the existing title.
//! - v2.5.2+Pagination: get_sessions_for_project and get_entity_timeline now take
//!   `offset: usize`. ALL call sites must be updated. Internal non-paginating callers
//!   must pass `0` explicitly. Do NOT remove the offset parameter.
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - Extracted from storage_ops.rs (was inline).
//!   Full CRUD for episodes, entities, knowledge_edges, episode_edges,
//!   communities, projects, sessions, artifacts tables + graph_stats.
//! v2.4.0+Search - Split into dedicated file (storage_graph.rs).
//!   Added SessionRow.title field.
//!   Updated update_session_summary() to accept title: Option<&str>.
//!   Added EntityTimelineEntry struct (see storage_miner.rs for get_entity_timeline).
//! v2.5.2+Pagination - Added offset param to get_sessions_for_project and
//!   get_entity_timeline. SQL updated to LIMIT ?N OFFSET ?M pattern.
//!   mpi_context_inject internal call passes offset=0 (no pagination).
// ============================================

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::warn;

use super::storage::{MemoryStorage, embedding_to_bytes};

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
///
/// ## v2.4.0+Search
/// Added `title: Option<String>` — a short human-readable label generated
/// by Miner Step 10. Populated by update_session_summary() with title param.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionRow {
    pub session_id: String,
    pub project_id: Option<String>,
    pub session_type: String,
    /// Human-readable session title (v2.4.0+Search).
    /// Examples: "Project Alpha: JWT, auth module" / "RS256 migration plan..."
    /// None until Miner Step 10 processes the session.
    pub title: Option<String>,
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

/// Entity timeline entry — one event in an entity's history.
/// Defined here; populated by get_entity_timeline (storage_miner.rs or below).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntityTimelineEntry {
    pub entity_id: String,
    pub event_type: String,
    pub event_time: i64,
    pub session_id: Option<String>,
    pub episode_id: Option<String>,
    pub detail: Option<String>,
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
            conn.execute(
                "UPDATE entities SET mention_count = mention_count + 1, updated_at = ?1,
                    description = COALESCE(?2, description),
                    embedding = COALESCE(?3, embedding)
                 WHERE entity_id = ?4",
                params![now, description, emb_blob.as_deref(), entity_id],
            ).map_err(|e| format!("Entity update: {}", e))?;
            Ok(false)
        } else {
            Ok(true)
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
        let placeholders: Vec<String> = entity_ids.iter().enumerate()
            .map(|(i, _)| format!("?{}", i + 3))
            .collect();
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
    ///
    /// ## v2.4.0+Search
    /// SELECT now includes `title` (column index 3).
    pub async fn get_session(&self, session_id: &str) -> Option<SessionRow> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT session_id, project_id, session_type, title, started_at, ended_at,
                    turn_count, summary, key_decisions, entities_extracted, summary_generated
             FROM sessions WHERE session_id = ?1",
            params![session_id],
            |row| Ok(SessionRow {
                session_id: row.get(0)?, project_id: row.get(1)?, session_type: row.get(2)?,
                title: row.get(3)?,
                started_at: row.get(4)?, ended_at: row.get(5)?, turn_count: row.get(6)?,
                summary: row.get(7)?, key_decisions: row.get(8)?,
                entities_extracted: row.get::<_, i64>(9).unwrap_or(0) != 0,
                summary_generated: row.get::<_, i64>(10).unwrap_or(0) != 0,
            }),
        ).optional().unwrap_or(None)
    }

    /// Get sessions for a project (timeline view), with pagination support.
    ///
    /// ## v2.4.0+Search
    /// SELECT now includes `title` (column index 3).
    ///
    /// ## v2.5.2+Pagination
    /// Added `offset: usize` parameter. SQL updated to `LIMIT ?2 OFFSET ?3`.
    ///
    /// ⚠️ Call sites:
    /// - mpi_project_timeline() → pass `params.offset`
    /// - mpi_context_inject()   → pass `0` (no pagination needed)
    pub async fn get_sessions_for_project(
        &self, project_id: &str, limit: usize, offset: usize,
    ) -> Vec<SessionRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT session_id, project_id, session_type, title, started_at, ended_at,
                    turn_count, summary, key_decisions, entities_extracted, summary_generated
             FROM sessions WHERE project_id = ?1
             ORDER BY started_at DESC LIMIT ?2 OFFSET ?3"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![project_id, limit as i64, offset as i64], |row| Ok(SessionRow {
            session_id: row.get(0)?, project_id: row.get(1)?, session_type: row.get(2)?,
            title: row.get(3)?,
            started_at: row.get(4)?, ended_at: row.get(5)?, turn_count: row.get(6)?,
            summary: row.get(7)?, key_decisions: row.get(8)?,
            entities_extracted: row.get::<_, i64>(9).unwrap_or(0) != 0,
            summary_generated: row.get::<_, i64>(10).unwrap_or(0) != 0,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }

    /// Update session summary, title, and key decisions (Miner Step 10).
    ///
    /// ## v2.4.0+Search: Added title parameter
    /// Title is a short human-readable label for the session.
    /// When title is None, the existing title is preserved (COALESCE).
    ///
    /// ⚠️ Callers in reflection.rs must pass Some(&title) — not None —
    ///    if a title was generated. Passing None leaves an existing title intact.
    pub async fn update_session_summary(
        &self, session_id: &str, summary: &str, key_decisions: Option<&str>,
        title: Option<&str>,
    ) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE sessions SET summary = ?1, key_decisions = ?2, summary_generated = 1,
                title = COALESCE(?4, title)
             WHERE session_id = ?3",
            params![summary, key_decisions, session_id, title],
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
    ///
    /// ## v2.4.0+Search
    /// SELECT now includes `title` (column index 3).
    pub async fn get_pending_sessions(&self, owner: &[u8; 32], limit: usize) -> Vec<SessionRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT session_id, project_id, session_type, title, started_at, ended_at,
                    turn_count, summary, key_decisions, entities_extracted, summary_generated
             FROM sessions
             WHERE owner = ?1 AND (entities_extracted = 0 OR summary_generated = 0)
             ORDER BY started_at DESC LIMIT ?2"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(params![owner.as_slice(), limit as i64], |row| Ok(SessionRow {
            session_id: row.get(0)?, project_id: row.get(1)?, session_type: row.get(2)?,
            title: row.get(3)?,
            started_at: row.get(4)?, ended_at: row.get(5)?, turn_count: row.get(6)?,
            summary: row.get(7)?, key_decisions: row.get(8)?,
            entities_extracted: row.get::<_, i64>(9).unwrap_or(0) != 0,
            summary_generated: row.get::<_, i64>(10).unwrap_or(0) != 0,
        })).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — v2.4.0+Search Entity Timeline
// ============================================

impl MemoryStorage {
    /// Get timeline events for a specific entity, ordered by event_time ASC.
    ///
    /// ## v2.5.2+Pagination
    /// Added `offset: usize` parameter. SQL updated to `LIMIT ?4 OFFSET ?5`.
    ///
    /// ⚠️ If the entity_timeline table lives in storage_miner.rs, move this
    ///    method there and keep the signature identical.
    ///
    /// Call sites:
    /// - mpi_entity_timeline() → pass `params.offset`
    pub async fn get_entity_timeline(
        &self, entity_id: &str, owner: &[u8; 32], limit: usize, offset: usize,
    ) -> Vec<EntityTimelineEntry> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT entity_id, event_type, event_time, session_id, episode_id, detail
             FROM entity_timeline
             WHERE entity_id = ?1 AND owner = ?2
             ORDER BY event_time ASC LIMIT ?3 OFFSET ?4"
        ) { Ok(s) => s, Err(_) => return Vec::new() };
        stmt.query_map(
            params![entity_id, owner.as_slice(), limit as i64, offset as i64],
            |row| Ok(EntityTimelineEntry {
                entity_id: row.get(0)?,
                event_type: row.get(1)?,
                event_time: row.get(2)?,
                session_id: row.get(3)?,
                episode_id: row.get(4)?,
                detail: row.get(5)?,
            }),
        ).map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default()
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
// Tests — v2.4.0 Cognitive Graph CRUD
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let eid = s.insert_knowledge_edge(&owner, "ent_auth", "ent_jwt", "USES", Some("auth uses JWT"), 1.0, 0.95, None, now, None).await.unwrap();
        assert!(eid > 0);
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
    async fn test_session_upsert_and_title() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        s.upsert_session("sess_001", &owner, None, "code", now, 15).await.unwrap();
        // Title should be None initially
        let sess = s.get_session("sess_001").await.unwrap();
        assert!(sess.title.is_none());
        // Set title via update_session_summary
        s.update_session_summary("sess_001", "Topics: JWT", None, Some("Project Alpha: JWT")).await;
        let sess2 = s.get_session("sess_001").await.unwrap();
        assert_eq!(sess2.title, Some("Project Alpha: JWT".into()));
        // Passing None preserves the existing title
        s.update_session_summary("sess_001", "Topics: JWT, RS256", None, None).await;
        let sess3 = s.get_session("sess_001").await.unwrap();
        assert_eq!(sess3.title, Some("Project Alpha: JWT".into()));
    }

    #[tokio::test]
    async fn test_session_pending() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        s.upsert_session("sess_001", &owner, None, "code", now, 15).await.unwrap();
        let pending = s.get_pending_sessions(&owner, 10).await;
        assert_eq!(pending.len(), 1);
        s.mark_session_entities_extracted("sess_001").await;
        let sess = s.get_session("sess_001").await.unwrap();
        assert!(sess.entities_extracted);
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
        // v2.5.2+Pagination: pass explicit offset=0
        let sessions = s.get_sessions_for_project("comm_1", 10, 0).await;
        assert_eq!(sessions.len(), 1);
    }

    #[tokio::test]
    async fn test_sessions_for_project_pagination() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        s.upsert_community("comm_1", &owner, "Proj", None, None, 1).await.unwrap();
        for i in 0..5u64 {
            s.upsert_session(&format!("sess_{i:03}"), &owner, Some("comm_1"), "code", now + i as i64, 1).await.unwrap();
        }
        let page0 = s.get_sessions_for_project("comm_1", 3, 0).await;
        let page1 = s.get_sessions_for_project("comm_1", 3, 3).await;
        assert_eq!(page0.len(), 3);
        assert_eq!(page1.len(), 2);
        // No overlap between pages
        let ids0: std::collections::HashSet<_> = page0.iter().map(|s| &s.session_id).collect();
        let ids1: std::collections::HashSet<_> = page1.iter().map(|s| &s.session_id).collect();
        assert!(ids0.is_disjoint(&ids1));
    }

    #[tokio::test]
    async fn test_artifact_insert_and_versions() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert_artifact("art_v1", &owner, "sess_001", None, "code", Some("auth.rs"), Some("rust"), 1, None, b"fn auth() {}", "hash1", None, Some(1)).await.unwrap();
        s.insert_artifact("art_v2", &owner, "sess_002", None, "code", Some("auth.rs"), Some("rust"), 2, Some("art_v1"), b"fn auth() { jwt() }", "hash2", None, Some(1)).await.unwrap();
        let versions = s.get_artifact_versions(&owner, "auth.rs").await;
        assert_eq!(versions.len(), 2);
        assert_eq!(versions[0].version, 2);
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
}
