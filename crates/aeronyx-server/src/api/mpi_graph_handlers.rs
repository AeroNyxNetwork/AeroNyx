// ============================================
// File: crates/aeronyx-server/src/api/mpi_graph_handlers.rs
// ============================================
//! # MPI Graph Handlers — v2.4.0 Cognitive Graph Endpoints
//!
//! ## Modification History
//! v2.4.0-GraphCognition - Initial 11 endpoints extracted from mpi.rs
//! v2.4.0+Conversation   - session_conversation decrypted turns
//! v2.5.2+Pagination     - ListParams gains `offset`; timeline pagination blocks
//! v1.0.1-SaaSFix       - BREAKING: all state.storage direct accesses replaced
//!                         with Extension<Arc<MemoryStorage>> extraction.
//!                         mpi_artifacts_search added (was registered in router
//!                         but missing from this file — caused compile error).
//!                         make_state() in tests updated to include all MpiState
//!                         fields added by v1.0.1 (pool_max_connections,
//!                         pool_idle_timeout_secs, mode, and SaaS None fields).
//!
//! ## Endpoints
//! - GET /api/mpi/projects
//! - GET /api/mpi/projects/:id
//! - GET /api/mpi/projects/:id/timeline
//! - GET /api/mpi/sessions/:id
//! - GET /api/mpi/sessions/:id/conversation
//! - GET /api/mpi/sessions/:id/artifacts
//! - GET /api/mpi/artifacts/search          <- v2.5.3 (was missing, added here)
//! - GET /api/mpi/artifacts/:id
//! - GET /api/mpi/artifacts/:id/versions
//! - GET /api/mpi/entities/:id
//! - GET /api/mpi/entities/:id/graph
//! - GET /api/mpi/entities/:id/timeline
//! - GET /api/mpi/entities
//! - GET /api/mpi/communities
//! - GET /api/mpi/search
//! - GET /api/mpi/context/inject
//!
//! ## SaaS Compatibility (v1.0.1-SaaSFix)
//! All state.storage accesses replaced with Extension extraction.
//! mpi_entity_graph uses conn_lock() — gated behind state.mode == Local
//! to avoid blocking the async runtime in SaaS mode with a shared conn.
//! In SaaS mode entity graph falls back to empty nodes (stub behavior,
//! full SaaS graph support is a future TODO).
//!
//! ⚠️ Important Notes for Next Developer:
//! - /artifacts/search MUST be registered before /artifacts/:id in the router
//!   to prevent "search" being captured as the :id path parameter.
//! - mpi_artifact_detail and mpi_artifact_versions are still stubs (Phase D).
//! - conn_lock() in mpi_entity_graph is Local-mode only; SaaS returns empty.
//! - make_state() in tests must stay in sync with MpiState struct fields.
//!
//! ## Last Modified
//! v1.0.1-SaaSFix - Extension-based storage; mpi_artifacts_search added;
//!                  make_state updated; conn_lock Local-mode guard added.
// ============================================

use std::sync::Arc;

use axum::{extract::{State, Path, Query}, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::services::memchain::{MemoryStorage, VectorIndex};
use crate::services::memchain::graph;
use crate::services::memchain::storage_crypto::decrypt_rawlog_content_pub;

use super::mpi::{MpiState, Mode, extract_owner, default_list_limit};

// ============================================
// Shared Query Params
// ============================================

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
// Conversation Types
// ============================================

#[derive(Debug, Clone, Serialize)]
pub struct ConversationTurn {
    pub turn_index: i64,
    pub role: String,
    pub content: Option<String>,
    pub encrypted: bool,
}

// ============================================
// Projects
// ============================================

pub async fn mpi_projects(
    State(_state): State<Arc<MpiState>>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let projects = storage.get_projects(
        &owner, params.status.as_deref(), params.limit.min(100),
    ).await;

    debug!(count = projects.len(), "[MPI] GET /projects");
    (StatusCode::OK, Json(serde_json::json!({"projects": projects})))
}

pub async fn mpi_project_detail(
    State(_state): State<Arc<MpiState>>,
    Path(project_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    match storage.get_project(&project_id, &owner).await {
        Some(p) => (StatusCode::OK, Json(serde_json::json!(p))).into_response(),
        None => (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"project not found"}))).into_response(),
    }
}

pub async fn mpi_project_timeline(
    State(_state): State<Arc<MpiState>>,
    Path(project_id): Path<String>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let limit = params.limit.min(100);
    let offset = params.offset;

    let sessions = storage.get_sessions_for_project(
        &project_id, limit, offset, &owner,
    ).await;

    debug!(project = %project_id, sessions = sessions.len(), offset, "[MPI] GET /projects/:id/timeline");
    (StatusCode::OK, Json(serde_json::json!({
        "project_id": project_id,
        "sessions": sessions,
        "pagination": { "limit": limit, "offset": offset, "has_more": sessions.len() == limit }
    })))
}

// ============================================
// Sessions
// ============================================

pub async fn mpi_session_detail(
    State(_state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    match storage.get_session(&session_id, &owner).await {
        Some(s) => (StatusCode::OK, Json(serde_json::json!(s))).into_response(),
        None => (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"session not found"}))).into_response(),
    }
}

pub async fn mpi_session_conversation(
    State(state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req);
    let owner = auth.owner_bytes();
    let is_remote = auth.is_remote();

    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    // Verify ownership — return 404 (not 403) to avoid leaking session existence.
    let session_meta = storage.get_session(&session_id, &owner).await;
    if session_meta.is_none() {
        return (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"session not found"}))).into_response();
    }

    let raw_logs = storage.get_rawlogs_for_session(&session_id).await;

    if raw_logs.is_empty() {
        let episodes = storage.get_episodes_for_session(&session_id).await;
        debug!(session = %session_id, episodes = episodes.len(),
            "[MPI] GET /sessions/:id/conversation (no raw_logs, fallback)");
        return (StatusCode::OK, Json(serde_json::json!({
            "session_id": session_id,
            "session": session_meta,
            "turns": serde_json::Value::Null,
            "episodes": episodes,
            "turn_count": 0,
            "note": "No raw conversation logs available. Showing episode metadata only."
        }))).into_response();
    }

    let rawlog_key = if !is_remote { state.rawlog_key.as_ref() } else { None };

    let mut turns: Vec<ConversationTurn> = Vec::with_capacity(raw_logs.len());
    let mut decrypt_failures = 0u32;

    for log in &raw_logs {
        let (content, encrypted) = if log.encrypted == 1 {
            if let Some(key) = rawlog_key {
                match decrypt_rawlog_content_pub(key, &log.content) {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(text) => (Some(text), false),
                        Err(_) => { decrypt_failures += 1; (None, true) }
                    },
                    Err(e) => {
                        warn!(session = %session_id, turn = log.turn_index,
                            error = %e, "[MPI] RawLog decryption failed");
                        decrypt_failures += 1;
                        (None, true)
                    }
                }
            } else {
                (None, true)
            }
        } else {
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

    if decrypt_failures > 0 {
        warn!(session = %session_id, failures = decrypt_failures,
            total = turns.len(), "[MPI] Some turns could not be decrypted");
    }

    debug!(session = %session_id, turns = turns.len(), remote = is_remote,
        "[MPI] GET /sessions/:id/conversation");

    (StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id,
        "session": session_meta,
        "turns": turns,
        "turn_count": turns.len(),
    }))).into_response()
}

pub async fn mpi_session_artifacts(
    State(_state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let artifacts = storage.get_artifacts_for_session(&session_id, &owner).await;
    debug!(session = %session_id, artifacts = artifacts.len(), "[MPI] GET /sessions/:id/artifacts");
    (StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id, "artifacts": artifacts,
    })))
}

// ============================================
// Artifacts
// ============================================

/// Query params for artifact search.
#[derive(Debug, Deserialize)]
pub struct ArtifactSearchParams {
    pub q: Option<String>,
    pub session_id: Option<String>,
    pub language: Option<String>,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

/// `GET /api/mpi/artifacts/search` — Search artifacts by content, session, or language.
///
/// v2.5.3+ArtifactChain. MUST be registered before /artifacts/:id in the router.
pub async fn mpi_artifacts_search(
    State(_state): State<Arc<MpiState>>,
    Query(params): Query<ArtifactSearchParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let limit = params.limit.min(100).max(1);
    let offset = params.offset;

    let artifacts = storage.search_artifacts(
        &owner,
        params.q.as_deref(),
        params.session_id.as_deref(),
        params.language.as_deref(),
        limit,
        offset,
    ).await;

    debug!(
        owner = &hex::encode(owner)[..8],
        q = params.q.as_deref().unwrap_or(""),
        results = artifacts.len(),
        "[MPI] GET /artifacts/search"
    );

    (StatusCode::OK, Json(serde_json::json!({
        "artifacts": artifacts,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "has_more": artifacts.len() == limit,
        }
    })))
}

/// `GET /api/mpi/artifacts/:id` — Artifact detail. (stub, Phase D)
pub async fn mpi_artifact_detail(
    State(_state): State<Arc<MpiState>>,
    Path(artifact_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    debug!(artifact = %artifact_id, "[MPI] GET /artifacts/:id (stub)");
    (StatusCode::OK, Json(serde_json::json!({
        "artifact_id": artifact_id,
        "status": "stub - full content retrieval in Phase D",
    })))
}

/// `GET /api/mpi/artifacts/:id/versions` — Artifact version history. (stub, Phase D)
pub async fn mpi_artifact_versions(
    State(_state): State<Arc<MpiState>>,
    Path(artifact_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    debug!(artifact = %artifact_id, "[MPI] GET /artifacts/:id/versions (stub)");
    let versions: Vec<crate::services::memchain::ArtifactRow> = Vec::new();
    (StatusCode::OK, Json(serde_json::json!({
        "artifact_id": artifact_id, "versions": versions,
    })))
}

// ============================================
// Entities
// ============================================

pub async fn mpi_entity_detail(
    State(_state): State<Arc<MpiState>>,
    Path(entity_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    match storage.get_entity(&entity_id).await {
        Some(entity) => {
            let edges = storage.get_edges_for_entity(&entity_id, &owner).await;
            let provenance = storage.get_episodes_for_entity(&entity_id).await;
            debug!(entity = %entity_id, edges = edges.len(), "[MPI] GET /entities/:id");
            (StatusCode::OK, Json(serde_json::json!({
                "entity": entity, "edges": edges, "provenance": provenance,
            }))).into_response()
        }
        None => (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"entity not found"}))).into_response(),
    }
}

/// `GET /api/mpi/entities/:id/graph` — 1-2 hop BFS subgraph.
///
/// conn_lock() is only called in Local mode. In SaaS mode, each user has
/// an isolated SQLite DB and conn_lock() on the Extension storage would be
/// correct but the graph query requires a synchronous rusqlite call inside
/// spawn_blocking which is not wired here yet. Returns empty nodes in SaaS
/// mode as a safe stub until full async graph support is added.
pub async fn mpi_entity_graph(
    State(state): State<Arc<MpiState>>,
    Path(entity_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let node_json: Vec<serde_json::Value> = if state.mode == Mode::Local {
        let conn = storage.conn_lock().await;
        let nodes = graph::bfs_traverse(
            &conn, &owner, &[entity_id.clone()],
            2, 20, 0.3,
        );
        drop(conn);
        nodes.iter().map(|n| serde_json::json!({
            "entity_id": n.entity_id,
            "depth": n.depth,
            "weight": n.weight,
            "via_relation": n.via_relation,
        })).collect()
    } else {
        // SaaS: full async graph BFS not yet implemented. Return empty.
        Vec::new()
    };

    debug!(root = %entity_id, nodes = node_json.len(), "[MPI] GET /entities/:id/graph");
    (StatusCode::OK, Json(serde_json::json!({
        "root": entity_id, "nodes": node_json,
    })))
}

pub async fn mpi_entities_list(
    State(_state): State<Arc<MpiState>>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let limit = params.limit.min(200).max(1);
    let offset = params.offset;

    let entities = storage.get_entities_by_owner(&owner, None, limit).await;

    (StatusCode::OK, Json(serde_json::json!({
        "entities": entities,
        "total": entities.len(),
        "pagination": { "limit": limit, "offset": offset, "has_more": entities.len() == limit }
    })))
}

// ============================================
// Search
// ============================================

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: String,
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    pub since: Option<i64>,
    pub until: Option<i64>,
    #[serde(rename = "type")]
    pub record_type: Option<String>,
}

pub async fn mpi_search(
    State(_state): State<Arc<MpiState>>,
    Query(params): Query<SearchParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let limit = params.limit.min(100).max(1);

    if params.q.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "query parameter 'q' is required and cannot be empty"
        }))).into_response();
    }

    let hits = storage.search_with_snippets(&params.q, &owner, limit).await;
    let groups = storage.group_hits_by_session(&hits).await;
    let total_results: usize = groups.iter().map(|g| g.hits.len()).sum();

    debug!(query = %params.q, groups = groups.len(), total = total_results, "[MPI] GET /search");
    (StatusCode::OK, Json(serde_json::json!({
        "query": params.q,
        "results": groups,
        "total_results": total_results,
    }))).into_response()
}

// ============================================
// Entity Timeline
// ============================================

pub async fn mpi_entity_timeline(
    State(_state): State<Arc<MpiState>>,
    Path(entity_id): Path<String>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let limit = params.limit.min(200).max(1);
    let offset = params.offset;

    let entity = match storage.get_entity(&entity_id).await {
        Some(e) => e,
        None => return (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"entity not found"}))).into_response(),
    };

    let timeline = storage.get_entity_timeline(&entity_id, &owner, limit, offset).await;

    debug!(entity = %entity_id, events = timeline.len(), offset, "[MPI] GET /entities/:id/timeline");
    (StatusCode::OK, Json(serde_json::json!({
        "entity": {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "mention_count": entity.mention_count,
        },
        "timeline": timeline,
        "pagination": { "limit": limit, "offset": offset, "has_more": timeline.len() == limit }
    }))).into_response()
}

// ============================================
// Context Injection
// ============================================

#[derive(Debug, Deserialize)]
pub struct ContextInjectParams {
    pub project_id: Option<String>,
    #[serde(default = "default_context_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_context_sessions")]
    pub recent_sessions: usize,
}

fn default_context_max_tokens() -> usize { 500 }
fn default_context_sessions()   -> usize { 3 }

pub async fn mpi_context_inject(
    State(_state): State<Arc<MpiState>>,
    Query(params): Query<ContextInjectParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let max_tokens  = params.max_tokens.min(2000).max(100);
    let recent_count = params.recent_sessions.min(10).max(1);

    let project = if let Some(ref pid) = params.project_id {
        storage.get_project(pid, &owner).await
    } else {
        let projects = storage.get_projects(&owner, Some("active"), 1).await;
        projects.into_iter().next()
    };

    let project_name = project.as_ref().map(|p| p.name.clone());
    let project_id   = project.as_ref().map(|p| p.project_id.clone());

    let recent_sessions = if let Some(ref pid) = project_id {
        storage.get_sessions_for_project(pid, recent_count, 0, &owner).await
    } else {
        storage.get_pending_sessions(&owner, recent_count).await
    };

    let key_entities = if let Some(ref pid) = project_id {
        storage.get_entities_in_community(pid, &owner).await
    } else {
        storage.get_entities_by_owner(&owner, None, 10).await
    };

    let mut ctx = String::new();
    let mut token_est: usize = 0;

    if let Some(ref name) = project_name {
        ctx.push_str(&format!("# {}\n\n", name));
        token_est += name.len() / 4 + 5;
    }

    if !recent_sessions.is_empty() {
        ctx.push_str("## Recent Sessions\n");
        token_est += 5;
        for sess in &recent_sessions {
            if token_est >= max_tokens { break; }
            let title = sess.title.as_deref()
                .or(sess.summary.as_deref())
                .unwrap_or(&sess.session_id);
            let line = format!("- **{}** (ts:{})\n", title, sess.started_at);
            token_est += line.len() / 4;
            ctx.push_str(&line);
            if let Some(ref summary) = sess.summary {
                let sl = format!("  {}\n", summary);
                token_est += sl.len() / 4;
                if token_est < max_tokens { ctx.push_str(&sl); }
            }
        }
        ctx.push('\n');
    }

    if !key_entities.is_empty() && token_est < max_tokens {
        ctx.push_str("## Key Concepts\n");
        token_est += 5;
        let names: Vec<String> = key_entities.iter().take(10)
            .filter(|e| { token_est += e.name.len() / 4 + 5; token_est < max_tokens })
            .map(|e| format!("{} ({})", e.name, e.entity_type))
            .collect();
        if !names.is_empty() {
            ctx.push_str(&names.join(", "));
            ctx.push('\n');
        }
    }

    let sessions_json: Vec<serde_json::Value> = recent_sessions.iter()
        .map(|s| serde_json::json!({
            "session_id": s.session_id, "title": s.title,
            "started_at": s.started_at, "summary": s.summary,
            "turn_count": s.turn_count,
        }))
        .collect();

    let entities_json: Vec<serde_json::Value> = key_entities.iter().take(10)
        .map(|e| serde_json::json!({
            "name": e.name, "type": e.entity_type, "mentions": e.mention_count,
        }))
        .collect();

    debug!(
        project = project_name.as_deref().unwrap_or("none"),
        sessions = sessions_json.len(), entities = entities_json.len(),
        tokens = token_est, "[MPI] GET /context/inject"
    );

    (StatusCode::OK, Json(serde_json::json!({
        "project": project.as_ref().map(|p| serde_json::json!({
            "project_id": p.project_id, "name": p.name, "status": p.status,
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

pub async fn mpi_communities(
    State(_state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let owner = extract_owner(&req).owner_bytes();
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let communities = storage.get_communities(&owner).await;
    debug!(count = communities.len(), "[MPI] GET /communities");
    (StatusCode::OK, Json(serde_json::json!({"communities": communities})))
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mpi::{build_mpi_router, MpiState, Mode};
    use crate::services::memchain::{MemoryStorage, VectorIndex};
    use crate::services::memchain::storage_crypto::derive_rawlog_key;
    use aeronyx_core::crypto::IdentityKeyPair;
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::AtomicBool;
    use parking_lot::RwLock;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// Build a minimal Local-mode MpiState for testing.
    ///
    /// Must be kept in sync with MpiState struct fields. Updated in v1.0.1
    /// to include pool_max_connections, pool_idle_timeout_secs, mode, and
    /// all SaaS fields (set to None for Local mode).
    async fn make_state() -> Arc<MpiState> {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let vi = Arc::new(VectorIndex::new());
        let identity = IdentityKeyPair::generate();
        let owner_key = identity.public_key_bytes();
        let rawlog_key = derive_rawlog_key(&identity.to_bytes());

        Arc::new(MpiState {
            mode: Mode::Local,
            storage: Some(Arc::clone(&storage)),
            vector_index: Some(Arc::clone(&vi)),
            identity,
            identity_cache: RwLock::new(HashMap::new()),
            index_ready: AtomicBool::new(true),
            user_weights: Arc::new(RwLock::new(HashMap::new())),
            mvf_alpha: 0.0,
            mvf_enabled: false,
            session_embeddings: RwLock::new(HashMap::new()),
            mvf_baseline: RwLock::new(None),
            owner_key,
            api_secret: None,
            embed_engine: None,
            allow_remote_storage: false,
            max_remote_owners: 0,
            ner_engine: None,
            graph_enabled: false,
            entropy_filter_enabled: false,
            reranker_engine: None,
            rawlog_key: Some(rawlog_key),
            llm_router: None,
            // SaaS fields — None in Local mode.
            storage_pool: None,
            vector_pool: None,
            volume_router: None,
            system_db: None,
            jwt_secret: None,
            token_ttl_secs: 86400,
            // Pool config fields (v1.0.1-Fix MT2).
            pool_max_connections: 0,
            pool_idle_timeout_secs: 0,
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
    async fn test_entity_graph_local_mode() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/entities/some_entity/graph").body(Body::empty()).unwrap();
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
        let req = Request::builder()
            .uri("/api/mpi/sessions/s1/conversation").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["session_id"], "s1");
        assert_eq!(json["turn_count"], 0);
    }

    #[tokio::test]
    async fn test_session_conversation_plaintext_turns() {
        let state = make_state().await;
        state.storage.as_ref().unwrap().insert_raw_log(
            "sess_test", 0, "user", "Hello auth discussion", "test", None, 1, None, None,
        ).await.unwrap();
        state.storage.as_ref().unwrap().insert_raw_log(
            "sess_test", 1, "assistant", "What auth method?", "test", None, 0, None, None,
        ).await.unwrap();

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/sessions/sess_test/conversation").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["turn_count"], 2);
        let turns = json["turns"].as_array().unwrap();
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(turns[0]["content"], "Hello auth discussion");
    }

    #[tokio::test]
    async fn test_project_timeline_pagination() {
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
    }

    #[tokio::test]
    async fn test_search_empty_query_rejected() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/search?q=").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_no_results() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/search?q=nonexistent_xyz").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["total_results"], 0);
    }

    #[tokio::test]
    async fn test_artifacts_search_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/artifacts/search?q=fn+main")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["artifacts"].as_array().unwrap().is_empty());
        assert!(json["pagination"].is_object());
    }

    #[tokio::test]
    async fn test_artifact_detail_stub() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/artifacts/art1").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_artifact_versions_stub() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/artifacts/art1/versions").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_entity_timeline_not_found() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/entities/nonexistent/timeline").body(Body::empty()).unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_entity_timeline_empty_events() {
        let state = make_state().await;
        let owner = state.owner_key;
        state.storage.as_ref().unwrap().upsert_entity(
            "ent_test", &owner, "TestEntity", "testentity", "concept", None, None,
        ).await.unwrap();

        let app = build_mpi_router(state);
        let req = Request::builder()
            .uri("/api/mpi/entities/ent_test/timeline").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["entity"]["name"], "TestEntity");
        assert!(json["timeline"].as_array().unwrap().is_empty());
        assert!(json["pagination"].is_object());
    }

    #[tokio::test]
    async fn test_session_owner_isolation() {
        let state = make_state().await;
        let owner = state.owner_key;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        state.storage.as_ref().unwrap()
            .upsert_session("sess_a", &owner, None, "code", now, 5).await.unwrap();

        // owner matches — should succeed
        let app_a = build_mpi_router(state.clone());
        let req = Request::builder().uri("/api/mpi/sessions/sess_a").body(Body::empty()).unwrap();
        assert_eq!(app_a.oneshot(req).await.unwrap().status(), StatusCode::OK);

        // Different identity — no sess_a in their DB -> 404
        let state_b = make_state().await;
        let app_b = build_mpi_router(state_b);
        let req2 = Request::builder().uri("/api/mpi/sessions/sess_a").body(Body::empty()).unwrap();
        assert_eq!(app_b.oneshot(req2).await.unwrap().status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_conversation_owner_isolation() {
        let state_a = make_state().await;
        let owner_a = state_a.owner_key;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        state_a.storage.as_ref().unwrap()
            .upsert_session("sess_private", &owner_a, None, "code", now, 2).await.unwrap();

        let state_b = make_state().await;
        let app_b = build_mpi_router(state_b);
        let req = Request::builder()
            .uri("/api/mpi/sessions/sess_private/conversation")
            .body(Body::empty()).unwrap();
        assert_eq!(app_b.oneshot(req).await.unwrap().status(), StatusCode::NOT_FOUND);
    }
}
