// ============================================
// File: crates/aeronyx-server/src/api/mpi_graph_handlers.rs
// ============================================
//! # MPI Graph Handlers — v2.4.0 Cognitive Graph Endpoints (11 new)
//!
//! ## Creation Reason (v2.4.0 Split)
//! Extracted from mpi.rs to keep the v2.4.0 cognitive graph endpoints
//! separate from the original handlers. All 11 endpoints are read-only
//! GET requests, all behind unified_auth_middleware.
//!
//! ## Modification Reason (v2.5.2+Pagination)
//! Added `offset: usize` to `ListParams` for cursor-free pagination.
//! - mpi_project_timeline() passes `params.offset` to get_sessions_for_project()
//! - mpi_entity_timeline() passes `params.offset` to get_entity_timeline()
//! - mpi_context_inject() passes `0` (internal call, no pagination)
//! Response JSON for timeline endpoints now includes a `pagination` block.
//!
//! ## Endpoints
//! - GET /api/mpi/projects — List projects
//! - GET /api/mpi/projects/:id — Project detail
//! - GET /api/mpi/projects/:id/timeline — Session summaries by date (paginated)
//! - GET /api/mpi/sessions/:id — Session detail
//! - GET /api/mpi/sessions/:id/conversation — Full conversation (decrypted turns)
//! - GET /api/mpi/sessions/:id/artifacts — Session artifacts
//! - GET /api/mpi/artifacts/:id — Artifact detail
//! - GET /api/mpi/artifacts/:id/versions — Artifact version history
//! - GET /api/mpi/entities/:id — Entity detail + relationships
//! - GET /api/mpi/entities/:id/graph — Entity 1-2 hop BFS subgraph
//! - GET /api/mpi/entities/:id/timeline — Entity event timeline (paginated)
//! - GET /api/mpi/communities — List communities
//!
//! ## Dependencies
//! - super::mpi::{MpiState, extract_owner, default_list_limit} — shared types/helpers
//! - storage_graph.rs — CRUD methods for cognitive graph tables (v2.5.2+Pagination)
//! - storage_crypto.rs — decrypt_rawlog_content_pub for conversation replay
//! - graph.rs — bfs_traverse() for entity subgraph
//!
//! ## v2.4.0+Conversation: Session Conversation Replay
//! The `/sessions/:id/conversation` endpoint now returns full decrypted
//! conversation turns (user + assistant messages) for Local auth requests.
//!
//! Data flow:
//! ```text
//! GET /sessions/:id/conversation
//!   → storage.get_rawlogs_for_session(session_id)  // RawLogRow with encrypted content
//!   → for each row: decrypt_rawlog_content_pub(rawlog_key, content)
//!   → return JSON array of {turn_index, role, content, created_at}
//! ```
//!
//! Security:
//! - Local requests: server-side decryption using rawlog_key derived from
//!   Ed25519 private key (state.rawlog_key). Full plaintext returned.
//! - Remote requests: rawlog_key is not available for remote owners.
//!   Returns raw_log metadata without decrypted content (content field = null).
//!   Remote clients must decrypt using their own rawlog_key.
//!
//! ⚠️ Important Notes for Next Developer:
//! - All endpoints extract owner from AuthenticatedOwner (via extract_owner)
//! - artifact_detail and artifact_versions are partial stubs (Phase D to complete)
//! - session_conversation uses state.rawlog_key for decryption — this key is
//!   derived from the LOCAL node's identity private key. It can ONLY decrypt
//!   rawlogs belonging to the local owner. Remote owners' rawlogs cannot be
//!   decrypted server-side (they were encrypted with a different key).
//! - entity_graph calls graph::bfs_traverse() which queries knowledge_edges
//! - The conversation endpoint also returns session metadata (summary, project)
//!   alongside the turns, so the UI can show context without extra API calls.
//! - v2.5.2+Pagination: ListParams now has `offset: usize`. All timeline
//!   handlers pass this to the storage layer. mpi_context_inject passes 0.
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - Initial implementation (11 endpoints)
//! v2.4.0+Conversation   - session_conversation now returns decrypted turns
//!   via rawlog_key server-side decryption. Added ConversationTurn type.
//!   Remote requests get metadata only (content=null).
//! v2.5.2+Pagination     - ListParams gains `offset` field.
//!   mpi_project_timeline: passes params.offset + emits pagination block.
//!   mpi_entity_timeline:  passes params.offset + emits pagination block.
//!   mpi_context_inject:   passes 0 for get_sessions_for_project (no pagination).
// ============================================

use std::sync::Arc;

use axum::{extract::{State, Path, Query}, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::services::memchain::graph;
use crate::services::memchain::storage_crypto::decrypt_rawlog_content_pub;

use super::mpi::{MpiState, extract_owner, default_list_limit};

// ============================================
// Shared Query Params
// ============================================

/// Shared list/pagination query params used across timeline and list endpoints.
///
/// ## v2.5.2+Pagination
/// Added `offset: usize` (default 0). Use with `limit` for page-based traversal:
///   - Page 1: `?limit=20&offset=0`
///   - Page 2: `?limit=20&offset=20`
/// If the response `pagination.has_more` is true, increment offset by limit
/// to fetch the next page.
#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    pub status: Option<String>,
    /// Pagination offset (default 0). v2.5.2+Pagination.
    #[serde(default)]
    pub offset: usize,
}

// ============================================
// Conversation Types (v2.4.0+Conversation)
// ============================================

/// A single conversation turn (user or assistant message).
///
/// Returned by `/sessions/:id/conversation` endpoint.
/// Content is decrypted server-side for Local requests,
/// null for Remote requests (client must decrypt).
#[derive(Debug, Clone, Serialize)]
pub struct ConversationTurn {
    /// Turn index within the session (0-based, ordered by conversation flow).
    pub turn_index: i64,
    /// Role: "user" or "assistant".
    pub role: String,
    /// Decrypted message content. None if decryption failed or Remote request.
    pub content: Option<String>,
    /// Whether this turn's content is encrypted (true = content is null, client must decrypt).
    pub encrypted: bool,
}

// ============================================
// Projects
// ============================================

/// `GET /api/mpi/projects` — List projects for the authenticated owner.
pub async fn mpi_projects(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let projects = state.storage.get_projects(
        &owner,
        params.status.as_deref(),
        params.limit.min(100),
    ).await;

    debug!(count = projects.len(), "[MPI] GET /projects");
    (StatusCode::OK, Json(serde_json::json!({"projects": projects})))
}

/// `GET /api/mpi/projects/:id` — Project detail.
pub async fn mpi_project_detail(
    State(state): State<Arc<MpiState>>,
    Path(project_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    match state.storage.get_project(&project_id).await {
        Some(p) => (StatusCode::OK, Json(serde_json::json!(p))).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "project not found"}))).into_response(),
    }
}

/// `GET /api/mpi/projects/:id/timeline` — Session summaries by date for a project.
///
/// ## v2.5.2+Pagination
/// Supports `?limit=N&offset=M` query params.
/// Response includes a `pagination` block:
/// ```json
/// "pagination": { "limit": 20, "offset": 0, "has_more": true }
/// ```
pub async fn mpi_project_timeline(
    State(state): State<Arc<MpiState>>,
    Path(project_id): Path<String>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    let limit = params.limit.min(100);
    let offset = params.offset;

    let sessions = state.storage.get_sessions_for_project(
        &project_id,
        limit,
        offset,
    ).await;

    debug!(
        project = %project_id,
        sessions = sessions.len(),
        offset = offset,
        "[MPI] GET /projects/:id/timeline"
    );
    (StatusCode::OK, Json(serde_json::json!({
        "project_id": project_id,
        "sessions": sessions,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_more": sessions.len() == limit,
        }
    })))
}

// ============================================
// Sessions
// ============================================

/// `GET /api/mpi/sessions/:id` — Session detail (summary + key decisions).
pub async fn mpi_session_detail(
    State(state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    match state.storage.get_session(&session_id).await {
        Some(s) => (StatusCode::OK, Json(serde_json::json!(s))).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "session not found"}))).into_response(),
    }
}

/// `GET /api/mpi/sessions/:id/conversation` — Full conversation replay.
///
/// Returns the complete conversation turns (user + assistant messages) for a session.
///
/// ## Local requests (Bearer token / open access)
/// Server-side decrypts raw_logs using the node's rawlog_key, returns plaintext
/// content for each turn. This is the primary use case for the MemExplorer UI.
///
/// ## Remote requests (Ed25519 signature auth)
/// The node's rawlog_key cannot decrypt remote owners' rawlogs (different key).
/// Returns turns with `content: null` and `encrypted: true`. The remote client
/// must decrypt using their own rawlog_key.
///
/// ## Response format
/// ```json
/// {
///   "session_id": "sess_alpha_001",
///   "session": { /* SessionRow metadata */ },
///   "turns": [
///     {"turn_index": 0, "role": "user", "content": "...", "encrypted": false},
///     {"turn_index": 1, "role": "assistant", "content": "...", "encrypted": false}
///   ],
///   "turn_count": 2
/// }
/// ```
pub async fn mpi_session_conversation(
    State(state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req);
    let is_remote = auth.is_remote();

    // Fetch session metadata (summary, project, etc.) for context
    let session_meta = state.storage.get_session(&session_id).await;

    // Fetch raw conversation turns
    let raw_logs = state.storage.get_rawlogs_for_session(&session_id).await;

    if raw_logs.is_empty() {
        // No raw_logs found — might be an old session before /log was implemented,
        // or the session_id is invalid. Fall back to episode metadata.
        let episodes = state.storage.get_episodes_for_session(&session_id).await;

        debug!(
            session = %session_id,
            episodes = episodes.len(),
            "[MPI] GET /sessions/:id/conversation (no raw_logs, fallback to episodes)"
        );

        return (StatusCode::OK, Json(serde_json::json!({
            "session_id": session_id,
            "session": session_meta,
            "turns": serde_json::Value::Null,
            "episodes": episodes,
            "turn_count": 0,
            "note": "No raw conversation logs available for this session. Showing episode metadata only."
        }))).into_response();
    }

    // Decrypt turns.
    // For Local requests: use state.rawlog_key to decrypt.
    // For Remote requests: cannot decrypt (different owner's key), return encrypted flag.
    let rawlog_key = if !is_remote {
        state.rawlog_key.as_ref()
    } else {
        None
    };

    let mut turns: Vec<ConversationTurn> = Vec::with_capacity(raw_logs.len());
    let mut decrypt_failures = 0u32;

    for log in &raw_logs {
        let (content, encrypted) = if log.encrypted == 1 {
            // Content is encrypted
            if let Some(key) = rawlog_key {
                // Local request — attempt server-side decryption
                match decrypt_rawlog_content_pub(key, &log.content) {
                    Ok(plaintext_bytes) => {
                        match String::from_utf8(plaintext_bytes) {
                            Ok(text) => (Some(text), false),
                            Err(_) => {
                                warn!(
                                    session = %session_id,
                                    turn = log.turn_index,
                                    "[MPI] Decrypted content is not valid UTF-8"
                                );
                                decrypt_failures += 1;
                                (None, true)
                            }
                        }
                    }
                    Err(e) => {
                        // Decryption failed — key mismatch or corrupt data.
                        // Can happen if rawlog_key was rotated or data is corrupted.
                        warn!(
                            session = %session_id,
                            turn = log.turn_index,
                            error = %e,
                            "[MPI] RawLog decryption failed"
                        );
                        decrypt_failures += 1;
                        (None, true)
                    }
                }
            } else {
                // Remote request or no rawlog_key — return as encrypted
                (None, true)
            }
        } else {
            // Content is plaintext (unencrypted rawlog)
            match String::from_utf8(log.content.clone()) {
                Ok(text) => (Some(text), false),
                Err(_) => (None, true),
            }
        };

        turns.push(ConversationTurn {
            turn_index: log.turn_index,
            role: log.role.clone(),
            content,
            encrypted,
        });
    }

    let turn_count = turns.len();

    if decrypt_failures > 0 {
        warn!(
            session = %session_id,
            failures = decrypt_failures,
            total = turn_count,
            "[MPI] Some turns could not be decrypted"
        );
    }

    debug!(
        session = %session_id,
        turns = turn_count,
        decrypted = turn_count as u32 - decrypt_failures,
        remote = is_remote,
        "[MPI] GET /sessions/:id/conversation"
    );

    (StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id,
        "session": session_meta,
        "turns": turns,
        "turn_count": turn_count,
    }))).into_response()
}

/// `GET /api/mpi/sessions/:id/artifacts` — Artifacts linked to a session.
pub async fn mpi_session_artifacts(
    State(state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    let artifacts = state.storage.get_artifacts_for_session(&session_id).await;

    debug!(session = %session_id, artifacts = artifacts.len(), "[MPI] GET /sessions/:id/artifacts");
    (StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id,
        "artifacts": artifacts,
    })))
}

// ============================================
// Artifacts
// ============================================

/// `GET /api/mpi/artifacts/:id` — Artifact detail (encrypted content).
///
/// TODO (Phase D): Load full artifact content from artifacts table.
/// Currently returns metadata only.
pub async fn mpi_artifact_detail(
    State(_state): State<Arc<MpiState>>,
    Path(artifact_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();

    // TODO: Implement full artifact content retrieval
    // let artifact = state.storage.get_artifact(&artifact_id).await;
    debug!(artifact = %artifact_id, "[MPI] GET /artifacts/:id (stub)");

    (StatusCode::OK, Json(serde_json::json!({
        "artifact_id": artifact_id,
        "status": "stub — full content retrieval in Phase D",
    })))
}

/// `GET /api/mpi/artifacts/:id/versions` — Artifact version history.
///
/// TODO (Phase D): Need to resolve artifact_id → filename first,
/// then query get_artifact_versions(owner, filename).
pub async fn mpi_artifact_versions(
    State(_state): State<Arc<MpiState>>,
    Path(artifact_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();

    // TODO: Look up filename from artifact_id, then query versions
    debug!(artifact = %artifact_id, "[MPI] GET /artifacts/:id/versions (stub)");

    let versions: Vec<crate::services::memchain::ArtifactRow> = Vec::new();
    (StatusCode::OK, Json(serde_json::json!({
        "artifact_id": artifact_id,
        "versions": versions,
    })))
}

// ============================================
// Entities
// ============================================

/// `GET /api/mpi/entities/:id` — Entity detail + relationship edges.
pub async fn mpi_entity_detail(
    State(state): State<Arc<MpiState>>,
    Path(entity_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();

    match state.storage.get_entity(&entity_id).await {
        Some(entity) => {
            let edges = state.storage.get_edges_for_entity(&entity_id, &owner).await;
            let provenance = state.storage.get_episodes_for_entity(&entity_id).await;

            debug!(entity = %entity_id, edges = edges.len(), "[MPI] GET /entities/:id");
            (StatusCode::OK, Json(serde_json::json!({
                "entity": entity,
                "edges": edges,
                "provenance": provenance,
            }))).into_response()
        }
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "entity not found",
        }))).into_response(),
    }
}

/// `GET /api/mpi/entities/:id/graph` — Entity 1-2 hop subgraph via BFS.
///
/// Returns the BFS traversal result from graph.rs::bfs_traverse().
/// Includes the root entity (depth 0) and all reachable entities
/// within 2 hops through currently valid knowledge_edges.
pub async fn mpi_entity_graph(
    State(state): State<Arc<MpiState>>,
    Path(entity_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();

    let conn = state.storage.conn_lock().await;
    let nodes = graph::bfs_traverse(
        &conn, &owner, &[entity_id.clone()],
        2,    // max_depth (configurable in Phase D)
        20,   // max_nodes_per_hop
        0.3,  // min_edge_weight
    );
    drop(conn);

    let node_json: Vec<serde_json::Value> = nodes.iter().map(|n| serde_json::json!({
        "entity_id": n.entity_id,
        "depth": n.depth,
        "weight": n.weight,
        "via_relation": n.via_relation,
    })).collect();

    debug!(root = %entity_id, nodes = node_json.len(), "[MPI] GET /entities/:id/graph");
    (StatusCode::OK, Json(serde_json::json!({
        "root": entity_id,
        "nodes": node_json,
    })))
}

/// `GET /api/mpi/entities` — List entities for the authenticated owner.
pub async fn mpi_entities_list(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let entities = state.storage.get_entities_by_owner(&owner, None, 200).await;
    Json(serde_json::json!({ "entities": entities, "total": entities.len() }))
}

// ============================================
// Search (v2.4.0+Search)
// ============================================

/// Query params for search endpoint.
#[derive(Debug, Deserialize)]
pub struct SearchParams {
    /// Search query string (natural language).
    pub q: String,
    /// Maximum results to return (default 20, max 100).
    #[serde(default = "default_list_limit")]
    pub limit: usize,
}

/// `GET /api/mpi/search?q=token+bucket&limit=20` — Human-facing search.
///
/// Returns search results grouped by session, with highlighted snippets
/// showing where the match occurred. Each result includes session metadata
/// (title, project, date) for navigation context.
///
/// ## Response Format
/// ```json
/// {
///   "query": "token bucket",
///   "results": [
///     {
///       "session_id": "sess_alpha_002",
///       "session_title": "Project Alpha: RS256 migration + rate limiting",
///       "project_name": "Project Alpha",
///       "started_at": 1710400000,
///       "hits": [
///         {"source_type": "record", "snippet": "...use <mark>token bucket</mark>...", "score": 2.8}
///       ],
///       "best_score": 2.8
///     }
///   ],
///   "total_results": 1
/// }
/// ```
pub async fn mpi_search(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<SearchParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let limit = params.limit.min(100).max(1);

    if params.q.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "query parameter 'q' is required and cannot be empty"
        }))).into_response();
    }

    // Step 1: FTS5 snippet search
    let hits = state.storage.search_with_snippets(&params.q, &owner, limit).await;

    // Step 2: Group by session with metadata enrichment
    let groups = state.storage.group_hits_by_session(&hits).await;

    let total_results: usize = groups.iter().map(|g| g.hits.len()).sum();

    debug!(
        query = %params.q,
        groups = groups.len(),
        total_hits = total_results,
        "[MPI] GET /search"
    );

    (StatusCode::OK, Json(serde_json::json!({
        "query": params.q,
        "results": groups,
        "total_results": total_results,
    }))).into_response()
}

// ============================================
// Entity Timeline (v2.4.0+Search)
// ============================================

/// `GET /api/mpi/entities/:id/timeline` — Entity event timeline.
///
/// Returns a chronological list of events for an entity across all sessions:
/// - "mentioned": entity appeared in a conversation
/// - "relation_created": a new relationship was established
/// - "relation_invalidated": an old relationship was superseded
///
/// Used by the frontend to show how a concept/tool/person evolved over time.
///
/// ## v2.5.2+Pagination
/// Supports `?limit=N&offset=M`. Response includes a `pagination` block.
///
/// ## Response Format
/// ```json
/// {
///   "entity": { "name": "auth module", "type": "module" },
///   "timeline": [...],
///   "pagination": { "limit": 50, "offset": 0, "has_more": false }
/// }
/// ```
pub async fn mpi_entity_timeline(
    State(state): State<Arc<MpiState>>,
    Path(entity_id): Path<String>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let limit = params.limit.min(200).max(1);
    let offset = params.offset;

    // Get entity metadata
    let entity = match state.storage.get_entity(&entity_id).await {
        Some(e) => e,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "entity not found"
        }))).into_response(),
    };

    // Get timeline events (v2.5.2+Pagination: pass offset)
    let timeline = state.storage.get_entity_timeline(&entity_id, &owner, limit, offset).await;

    debug!(
        entity = %entity_id,
        events = timeline.len(),
        offset = offset,
        "[MPI] GET /entities/:id/timeline"
    );

    (StatusCode::OK, Json(serde_json::json!({
        "entity": {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "mention_count": entity.mention_count,
        },
        "timeline": timeline,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_more": timeline.len() == limit,
        }
    }))).into_response()
}

// ============================================
// Context Injection (v2.4.0+Context)
// ============================================

/// Query params for context injection endpoint.
#[derive(Debug, Deserialize)]
pub struct ContextInjectParams {
    /// Optional project ID to scope context to a specific project.
    /// When None, returns context from the most recently active project.
    pub project_id: Option<String>,
    /// Maximum tokens for the formatted context (default 500).
    #[serde(default = "default_context_max_tokens")]
    pub max_tokens: usize,
    /// Number of recent sessions to include (default 3).
    #[serde(default = "default_context_sessions")]
    pub recent_sessions: usize,
}

fn default_context_max_tokens() -> usize { 500 }
fn default_context_sessions() -> usize { 3 }

/// `GET /api/mpi/context/inject` — Auto-inject project context for new sessions.
///
/// Called by the plugin at session start to prime the AI with relevant memory.
/// Returns a pre-formatted markdown string ready for system prompt injection.
///
/// ## Usage Flow
/// ```text
/// Plugin: user opens a new conversation
///   → GET /api/mpi/context/inject?project_id=comm_alpha&max_tokens=500
///   → receives formatted_context markdown
///   → injects into system prompt: "## Project Context\n{formatted_context}"
///   → AI naturally "remembers" recent work
/// ```
///
/// ## v2.5.2+Pagination
/// Internal call to get_sessions_for_project passes offset=0 (no pagination needed
/// here — we always want the N most recent sessions for context injection).
///
/// ## Response
/// ```json
/// {
///   "project": {"name": "Project Alpha", "status": "active"},
///   "recent_sessions": [...],
///   "key_entities": [...],
///   "formatted_context": "# Project Alpha\n\n## Recent Sessions\n...",
///   "token_estimate": 320
/// }
/// ```
pub async fn mpi_context_inject(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<ContextInjectParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let max_tokens = params.max_tokens.min(2000).max(100);
    let recent_count = params.recent_sessions.min(10).max(1);

    // Resolve project: explicit ID or most recently active
    let project = if let Some(ref pid) = params.project_id {
        state.storage.get_project(pid).await
    } else {
        let projects = state.storage.get_projects(&owner, Some("active"), 1).await;
        projects.into_iter().next()
    };

    let project_name = project.as_ref().map(|p| p.name.clone());
    let project_id = project.as_ref().map(|p| p.project_id.clone());

    // Get recent sessions for this project (or all sessions if no project).
    // offset=0: context injection always wants the N most recent — no pagination.
    let recent_sessions = if let Some(ref pid) = project_id {
        state.storage.get_sessions_for_project(pid, recent_count, 0).await
    } else {
        // No project → get most recent sessions for this owner
        state.storage.get_pending_sessions(&owner, recent_count).await
    };

    // Get key entities (top by mention count)
    let key_entities = if let Some(ref pid) = project_id {
        state.storage.get_entities_in_community(pid, &owner).await
    } else {
        state.storage.get_entities_by_owner(&owner, None, 10).await
    };

    // Build formatted context markdown
    let mut ctx = String::new();
    let mut token_est: usize = 0;

    if let Some(ref name) = project_name {
        ctx.push_str(&format!("# {}\n\n", name));
        token_est += name.len() / 4 + 5;
    }

    // Recent sessions
    if !recent_sessions.is_empty() {
        ctx.push_str("## Recent Sessions\n");
        token_est += 5;

        for sess in &recent_sessions {
            if token_est >= max_tokens { break; }

            let title = sess.title.as_deref()
                .or(sess.summary.as_deref())
                .unwrap_or(&sess.session_id);

            // Format date from timestamp.
            // Frontend is responsible for locale-aware formatting;
            // we emit the raw Unix timestamp for maximum compatibility.
            let date_str = format!("ts:{}", sess.started_at);

            let line = format!("- **{}** ({})\n", title, date_str);
            token_est += line.len() / 4;
            ctx.push_str(&line);

            if let Some(ref summary) = sess.summary {
                let summary_line = format!("  {}\n", summary);
                token_est += summary_line.len() / 4;
                if token_est < max_tokens {
                    ctx.push_str(&summary_line);
                }
            }
        }
        ctx.push('\n');
    }

    // Key entities
    if !key_entities.is_empty() && token_est < max_tokens {
        ctx.push_str("## Key Concepts\n");
        token_est += 5;

        let entity_names: Vec<String> = key_entities.iter()
            .take(10)
            .filter(|e| {
                token_est += e.name.len() / 4 + 5;
                token_est < max_tokens
            })
            .map(|e| format!("{} ({})", e.name, e.entity_type))
            .collect();

        if !entity_names.is_empty() {
            ctx.push_str(&entity_names.join(", "));
            ctx.push('\n');
        }
    }

    // Build response JSON
    let sessions_json: Vec<serde_json::Value> = recent_sessions.iter()
        .map(|s| serde_json::json!({
            "session_id": s.session_id,
            "title": s.title,
            "started_at": s.started_at,
            "summary": s.summary,
            "turn_count": s.turn_count,
        }))
        .collect();

    let entities_json: Vec<serde_json::Value> = key_entities.iter()
        .take(10)
        .map(|e| serde_json::json!({
            "name": e.name,
            "type": e.entity_type,
            "mentions": e.mention_count,
        }))
        .collect();

    debug!(
        project = project_name.as_deref().unwrap_or("none"),
        sessions = sessions_json.len(),
        entities = entities_json.len(),
        tokens = token_est,
        "[MPI] GET /context/inject"
    );

    (StatusCode::OK, Json(serde_json::json!({
        "project": project.as_ref().map(|p| serde_json::json!({
            "project_id": p.project_id,
            "name": p.name,
            "status": p.status,
        })),
        "recent_sessions": sessions_json,
        "key_entities": entities_json,
        "formatted_context": ctx.trim(),
        "token_estimate": token_est,
    }))).into_response()
}

// ============================================
// Communities
// ============================================

/// `GET /api/mpi/communities` — List communities for the authenticated owner.
pub async fn mpi_communities(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let communities = state.storage.get_communities(&owner).await;

    debug!(count = communities.len(), "[MPI] GET /communities");
    (StatusCode::OK, Json(serde_json::json!({
        "communities": communities,
    })))
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mpi::{build_mpi_router, MpiState, BaselineSnapshot};
    use crate::services::memchain::{MemoryStorage, VectorIndex};
    use crate::services::memchain::mvf::WeightVector;
    use crate::services::memchain::storage_crypto::derive_rawlog_key;
    use aeronyx_core::crypto::IdentityKeyPair;
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::AtomicBool;
    use parking_lot::RwLock;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn make_state() -> Arc<MpiState> {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let vi = Arc::new(VectorIndex::new());
        let id = IdentityKeyPair::generate();
        let ok = id.public_key_bytes();
        let rlk = derive_rawlog_key(&id.to_bytes());
        Arc::new(MpiState {
            storage, vector_index: vi, identity: id,
            identity_cache: RwLock::new(HashMap::new()),
            index_ready: AtomicBool::new(true),
            user_weights: Arc::new(RwLock::new(HashMap::new())),
            mvf_alpha: 0.0, mvf_enabled: false,
            session_embeddings: RwLock::new(HashMap::new()),
            mvf_baseline: RwLock::new(None),
            owner_key: ok, api_secret: None, embed_engine: None,
            allow_remote_storage: false, max_remote_owners: 0,
            ner_engine: None, graph_enabled: false, entropy_filter_enabled: false,
            reranker_engine: None,
            rawlog_key: Some(rlk),
        })
    }

    #[tokio::test]
    async fn test_projects_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/projects").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["projects"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_communities_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/communities").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["communities"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_entity_not_found() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/entities/nonexistent").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_session_not_found() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/sessions/nonexistent").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_project_not_found() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/projects/nonexistent").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_entity_graph_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/entities/some_entity/graph").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["root"], "some_entity");
        assert!(json["nodes"].as_array().is_some());
    }

    #[tokio::test]
    async fn test_session_conversation_no_rawlogs() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/sessions/s1/conversation").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["session_id"], "s1");
        assert_eq!(json["turn_count"], 0);
        assert!(json["note"].is_string());
    }

    #[tokio::test]
    async fn test_session_conversation_plaintext_turns() {
        let state = make_state().await;
        state.storage.insert_raw_log(
            "sess_test", 0, "user", "Hello, let's discuss auth", "test",
            None, 1, None, None,
        ).await.unwrap();
        state.storage.insert_raw_log(
            "sess_test", 1, "assistant", "Sure! What auth method?", "test",
            None, 0, None, None,
        ).await.unwrap();
        state.storage.insert_raw_log(
            "sess_test", 2, "user", "RS256 with ring crate", "test",
            None, 1, None, None,
        ).await.unwrap();

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/sessions/sess_test/conversation")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["session_id"], "sess_test");
        assert_eq!(json["turn_count"], 3);

        let turns = json["turns"].as_array().unwrap();
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0]["turn_index"], 0);
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(turns[0]["content"], "Hello, let's discuss auth");
        assert_eq!(turns[0]["encrypted"], false);
        assert_eq!(turns[1]["role"], "assistant");
        assert_eq!(turns[1]["content"], "Sure! What auth method?");
        assert_eq!(turns[2]["content"], "RS256 with ring crate");
    }

    #[tokio::test]
    async fn test_session_conversation_encrypted_turns() {
        let state = make_state().await;
        let rlk = state.rawlog_key.unwrap();

        state.storage.insert_raw_log(
            "sess_enc", 0, "user", "Secret project discussion", "test",
            None, 1, None, Some(&rlk),
        ).await.unwrap();
        state.storage.insert_raw_log(
            "sess_enc", 1, "assistant", "Understood, proceeding with encryption", "test",
            None, 0, None, Some(&rlk),
        ).await.unwrap();

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/sessions/sess_enc/conversation")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["turn_count"], 2);
        let turns = json["turns"].as_array().unwrap();
        assert_eq!(turns[0]["content"], "Secret project discussion");
        assert_eq!(turns[0]["encrypted"], false);
        assert_eq!(turns[1]["content"], "Understood, proceeding with encryption");
        assert_eq!(turns[1]["encrypted"], false);
    }

    #[tokio::test]
    async fn test_session_artifacts_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/sessions/s1/artifacts").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_artifact_detail_stub() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/artifacts/art1").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_artifact_versions_stub() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/artifacts/art1/versions").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_project_timeline_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/projects/proj1/timeline").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["project_id"], "proj1");
        // v2.5.2+Pagination: pagination block must be present
        assert!(json["pagination"].is_object());
        assert_eq!(json["pagination"]["offset"], 0);
    }

    #[tokio::test]
    async fn test_project_timeline_pagination_params() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/projects/proj1/timeline?limit=5&offset=10")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["pagination"]["limit"], 5);
        assert_eq!(json["pagination"]["offset"], 10);
        assert_eq!(json["pagination"]["has_more"], false);
    }

    // ── v2.4.0+Search: Search endpoint tests ──

    #[tokio::test]
    async fn test_search_empty_query_rejected() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/search?q=")
            .body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_no_results() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/search?q=nonexistent_term_xyz")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["query"], "nonexistent_term_xyz");
        assert_eq!(json["total_results"], 0);
    }

    #[tokio::test]
    async fn test_search_with_indexed_content() {
        let state = make_state().await;
        let owner = state.owner_key;
        state.storage.fts_index_record(
            &[0x01; 32], &owner,
            "token bucket rate limiting at 100 requests per minute", "",
        ).await;

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/search?q=token+bucket")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["total_results"].as_u64().unwrap() > 0);
    }

    // ── v2.4.0+Search: Entity timeline tests ──

    #[tokio::test]
    async fn test_entity_timeline_not_found() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/entities/nonexistent/timeline")
            .body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_entity_timeline_empty_events() {
        let state = make_state().await;
        let owner = state.owner_key;
        state.storage.upsert_entity(
            "ent_test", &owner, "TestEntity", "testentity", "concept", None, None,
        ).await.unwrap();

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/entities/ent_test/timeline")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["entity"]["name"], "TestEntity");
        assert!(json["timeline"].as_array().unwrap().is_empty());
        // v2.5.2+Pagination: pagination block must be present
        assert!(json["pagination"].is_object());
        assert_eq!(json["pagination"]["offset"], 0);
    }

    #[tokio::test]
    async fn test_entity_timeline_pagination_params() {
        let state = make_state().await;
        let owner = state.owner_key;
        state.storage.upsert_entity(
            "ent_pg", &owner, "PgEntity", "pgentity", "concept", None, None,
        ).await.unwrap();

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/entities/ent_pg/timeline?limit=10&offset=20")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["pagination"]["limit"], 10);
        assert_eq!(json["pagination"]["offset"], 20);
        assert_eq!(json["pagination"]["has_more"], false);
    }
}
