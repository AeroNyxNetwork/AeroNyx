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
//! ## Endpoints
//! - GET /api/mpi/projects — List projects
//! - GET /api/mpi/projects/:id — Project detail
//! - GET /api/mpi/projects/:id/timeline — Session summaries by date
//! - GET /api/mpi/sessions/:id — Session detail
//! - GET /api/mpi/sessions/:id/conversation — Full conversation (episodes)
//! - GET /api/mpi/sessions/:id/artifacts — Session artifacts
//! - GET /api/mpi/artifacts/:id — Artifact detail
//! - GET /api/mpi/artifacts/:id/versions — Artifact version history
//! - GET /api/mpi/entities/:id — Entity detail + relationships
//! - GET /api/mpi/entities/:id/graph — Entity 1-2 hop BFS subgraph
//! - GET /api/mpi/communities — List communities
//!
//! ## Dependencies
//! - super::mpi::{MpiState, extract_owner, default_list_limit} — shared types/helpers
//! - storage_ops.rs — CRUD methods for cognitive graph tables
//! - graph.rs — bfs_traverse() for entity subgraph
//!
//! ⚠️ Important Note for Next Developer:
//! - All endpoints extract owner from AuthenticatedOwner (via extract_owner)
//! - artifact_detail and artifact_versions are partial stubs (Phase D to complete)
//! - session_conversation returns episode metadata, not decrypted content
//!   (client must decrypt using ChaCha20 key)
//! - entity_graph calls graph::bfs_traverse() which queries knowledge_edges
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - 🌟 Initial implementation (11 endpoints)

use std::sync::Arc;

use axum::{extract::{State, Path, Query}, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use serde::Deserialize;
use tracing::debug;

use crate::services::memchain::graph;

use super::mpi::{MpiState, extract_owner, default_list_limit};

// ============================================
// Shared Query Params
// ============================================

#[derive(Debug, Deserialize)]
pub struct ListParams {
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    pub status: Option<String>,
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
pub async fn mpi_project_timeline(
    State(state): State<Arc<MpiState>>,
    Path(project_id): Path<String>,
    Query(params): Query<ListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    let sessions = state.storage.get_sessions_for_project(
        &project_id,
        params.limit.min(100),
    ).await;

    debug!(project = %project_id, sessions = sessions.len(), "[MPI] GET /projects/:id/timeline");
    (StatusCode::OK, Json(serde_json::json!({
        "project_id": project_id,
        "sessions": sessions,
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

/// `GET /api/mpi/sessions/:id/conversation` — Full conversation (episodes).
///
/// Returns episode metadata (episode_id, type, created_at).
/// Encrypted content must be fetched separately and decrypted client-side.
/// In Remote mode, returns ciphertext — plugin decryptContent() handles it.
pub async fn mpi_session_conversation(
    State(state): State<Arc<MpiState>>,
    Path(session_id): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _owner = extract_owner(&req).owner_bytes();
    let episodes = state.storage.get_episodes_for_session(&session_id).await;

    debug!(session = %session_id, episodes = episodes.len(), "[MPI] GET /sessions/:id/conversation");
    (StatusCode::OK, Json(serde_json::json!({
        "session_id": session_id,
        "episodes": episodes,
    })))
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

    let versions: Vec<crate::services::memchain::storage_ops::ArtifactRow> = Vec::new();
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
        // Should have at least the seed node (if it exists) or empty
        assert!(json["nodes"].as_array().is_some());
    }

    #[tokio::test]
    async fn test_session_conversation_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/sessions/s1/conversation").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["session_id"], "s1");
    }

    #[tokio::test]
    async fn test_session_artifacts_empty() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/sessions/s1/artifacts").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_artifact_detail_stub() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/artifacts/art1").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_artifact_versions_stub() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/artifacts/art1/versions").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
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
    }
}
