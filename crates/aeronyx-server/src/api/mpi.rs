// ============================================
// File: crates/aeronyx-server/src/api/mpi.rs
// ============================================
//! # MPI — Memory Protocol Interface (Core Module)
//!
//! ## File Split
//! - `mpi.rs` (THIS FILE) — MpiState, auth middleware, router, helpers
//! - `mpi_handlers.rs`       — remember, forget, status, embed, record, overview,
//!                             patch record, provenance (v2.5.2)
//! - `recall_handler.rs`     — hybrid recall pipeline
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints
//! - `log_handler.rs`        — /log ingestion
//! - `supernode_handlers.rs` — v2.5.0 SuperNode management
//!
//! ⚠️ Important Note for Next Developer:
//! - unified_auth_middleware MUST be applied to all routes via route_layer.
//!   AuthenticatedOwner is injected into extensions here; handlers depend on it.
//! - Remote auth path consumes the request body for signature verification,
//!   then reconstructs the request. Any body size > 1MB is rejected (BAD_REQUEST).
//! - .patch() on a MethodRouter does NOT require importing patch from axum::routing.
//!   axum::routing::{get, post} is sufficient — .patch() is a MethodRouter method.
//! - Route registration order matters for path param routes:
//!   `/record/:record_id/provenance` must be registered AFTER `/record/:record_id`
//!   to prevent `:record_id` from greedily matching "xxxx/provenance".
//!   Current order is correct — do not reorder.
//! - Endpoint count in the info! log must be kept in sync with actual route count.
//!
//! ## Modification History
//! v2.1.0                   - MPI endpoints with MVF fusion scoring
//! v2.1.0+MVF+Auth          - Bearer token auth middleware
//! v2.2.0                   - Added record/:id, records/overview, embed endpoints
//! v2.3.0+RemoteStorage     - Unified auth middleware (Bearer + Ed25519)
//! v2.4.0-GraphCognition    - 🌟 Split into files; MpiState + graph extensions
//! v2.4.0+Reranker          - 🌟 MpiState.reranker_engine
//! v2.4.0+Conversation      - 🌟 MpiState.rawlog_key
//! v2.5.0+SuperNode Phase D - 🌟 MpiState.llm_router; 6 supernode routes; 23→29
//! v2.5.2+Provenance        - 🌟 PATCH /record/:id + GET /record/:id/provenance;
//!                            29→31 endpoints. No new files — routes added to
//!                            existing mpi_handlers.rs.
//!
//! ## Last Modified
//! v2.5.2+Provenance - 🌟 +2 routes (patch record, provenance); 29→31

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use axum::middleware::{self, Next};
use axum::routing::{get, post};
use axum::Router;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use tracing::{debug, info, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{MemoryStorage, VectorIndex};
use crate::services::memchain::mvf::WeightVector;
use crate::services::memchain::EmbedEngine;
use crate::services::memchain::NerEngine;
use crate::services::memchain::RerankerEngine;
use crate::services::memchain::LlmRouter;

use super::mpi_handlers;
use super::mpi_graph_handlers;
use super::supernode_handlers;

// ============================================
// Constants
// ============================================

pub(crate) const AUTH_TIMESTAMP_TOLERANCE_SECS: u64 = 300;

// ============================================
// AuthenticatedOwner
// ============================================

#[derive(Debug, Clone)]
pub enum AuthenticatedOwner {
    Local  { owner: [u8; 32] },
    Remote { owner: [u8; 32], owner_hex: String },
}

impl AuthenticatedOwner {
    #[must_use]
    pub fn owner_bytes(&self) -> [u8; 32] {
        match self {
            Self::Local  { owner }      => *owner,
            Self::Remote { owner, .. }  => *owner,
        }
    }

    #[must_use]
    pub fn owner_hex(&self) -> String {
        match self {
            Self::Local  { owner }          => hex::encode(owner),
            Self::Remote { owner_hex, .. }  => owner_hex.clone(),
        }
    }

    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }
}

// ============================================
// MpiState
// ============================================

pub struct MpiState {
    pub storage: Arc<MemoryStorage>,
    pub vector_index: Arc<VectorIndex>,
    pub identity: IdentityKeyPair,
    pub identity_cache: RwLock<HashMap<String, Vec<MemoryRecord>>>,
    pub index_ready: AtomicBool,
    pub user_weights: Arc<RwLock<HashMap<String, WeightVector>>>,
    pub mvf_alpha: f32,
    pub mvf_enabled: bool,
    pub session_embeddings: RwLock<HashMap<String, VecDeque<Vec<f32>>>>,
    pub mvf_baseline: RwLock<Option<BaselineSnapshot>>,
    pub owner_key: [u8; 32],
    pub api_secret: Option<String>,
    pub embed_engine: Option<Arc<EmbedEngine>>,
    /// v2.3.0: Remote storage config.
    pub allow_remote_storage: bool,
    pub max_remote_owners: usize,
    /// v2.4.0: NER engine (GLiNER).
    pub ner_engine: Option<Arc<NerEngine>>,
    /// v2.4.0: Knowledge graph config.
    pub graph_enabled: bool,
    pub entropy_filter_enabled: bool,
    /// v2.4.0+Reranker: Cross-encoder reranker.
    pub reranker_engine: Option<Arc<RerankerEngine>>,
    /// v2.4.0+Conversation: RawLog decryption key (PRIVATE key derived).
    pub rawlog_key: Option<[u8; 32]>,
    /// v2.5.0+SuperNode: LLM router for cognitive task dispatch.
    ///
    /// When Some, TaskWorker polls cognitive_tasks and dispatches to providers.
    /// When None, SuperNode is disabled — /supernode/* returns 404.
    pub llm_router: Option<Arc<LlmRouter>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub positive_rate: f32,
    pub sample_size: usize,
    pub frozen_at: i64,
}

// ============================================
// Shared Helpers
// ============================================

pub(crate) fn extract_owner<B>(req: &Request<B>) -> &AuthenticatedOwner {
    req.extensions()
        .get::<AuthenticatedOwner>()
        .expect("[BUG] AuthenticatedOwner not set — unified_auth_middleware must be applied")
}

pub(crate) fn parse_layer(s: &str) -> Option<MemoryLayer> {
    match s.to_lowercase().as_str() {
        "identity"  => Some(MemoryLayer::Identity),
        "knowledge" => Some(MemoryLayer::Knowledge),
        "episode"   => Some(MemoryLayer::Episode),
        "archive"   => Some(MemoryLayer::Archive),
        _ => None,
    }
}

pub(crate) fn estimate_tokens(text: &str) -> usize {
    let len = text.len();
    if len == 0 { return 0; }
    let sample_len = len.min(100);
    let ascii = text.as_bytes()[..sample_len].iter().filter(|b| b.is_ascii()).count();
    if (ascii as f64 / sample_len as f64) > 0.80 { (len + 3) / 4 }
    else { (len * 2 / 3).max(1) }
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub(crate) fn default_layer()         -> String { "episode".into() }
pub(crate) fn default_source()        -> String { "unknown".into() }
pub(crate) fn default_model()         -> String { "default".into() }
pub(crate) fn default_top_k()         -> usize  { 10 }
pub(crate) fn default_token_budget()  -> usize  { 4000 }
pub(crate) fn default_include_graph() -> bool   { true }
pub(crate) fn default_list_limit()    -> usize  { 20 }

// ============================================
// Unified Auth Middleware
// ============================================

async fn unified_auth_middleware(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    if req.headers().contains_key("x-memchain-signature") {
        return handle_remote_auth(state, req, next).await;
    }
    handle_local_auth(state, req, next).await
}

async fn handle_remote_auth(
    state: Arc<MpiState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    let pubkey_hex = match req.headers()
        .get("x-memchain-publickey").and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"missing X-MemChain-PublicKey header"}
        ))).into_response(),
    };
    let timestamp_str = match req.headers()
        .get("x-memchain-timestamp").and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"missing X-MemChain-Timestamp header"}
        ))).into_response(),
    };
    let signature_hex = match req.headers()
        .get("x-memchain-signature").and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"missing X-MemChain-Signature header"}
        ))).into_response(),
    };

    let pubkey_bytes = match hex::decode(&pubkey_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8;32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"invalid X-MemChain-PublicKey"}
        ))).into_response(),
    };

    let identity_pubkey = match IdentityPublicKey::from_bytes(&pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"invalid Ed25519 public key"}
        ))).into_response(),
    };

    let timestamp: u64 = match timestamp_str.parse() {
        Ok(ts) => ts,
        Err(_) => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"invalid X-MemChain-Timestamp"}
        ))).into_response(),
    };

    let now = now_secs();
    let drift = if now > timestamp { now - timestamp } else { timestamp - now };
    if drift > AUTH_TIMESTAMP_TOLERANCE_SECS {
        return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({
            "error": "timestamp expired or clock drift too large",
            "server_time": now, "request_time": timestamp,
            "tolerance_secs": AUTH_TIMESTAMP_TOLERANCE_SECS,
        }))).into_response();
    }

    if !state.allow_remote_storage {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!(
            {"error":"this node does not accept remote storage requests"}
        ))).into_response();
    }

    let sig_bytes = match hex::decode(&signature_hex) {
        Ok(b) if b.len() == 64 => { let mut a = [0u8;64]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"invalid X-MemChain-Signature"}
        ))).into_response(),
    };

    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!(
            {"error":"failed to read request body"}
        ))).into_response(),
    };

    let body_hash = Sha256::digest(&body_bytes);
    let mut msg_hasher = Sha256::new();
    msg_hasher.update(timestamp_str.as_bytes());
    msg_hasher.update(method.as_bytes());
    msg_hasher.update(path.as_bytes());
    msg_hasher.update(&body_hash);
    let signed_msg = msg_hasher.finalize();

    if identity_pubkey.verify(&signed_msg, &sig_bytes).is_err() {
        warn!("[MPI_AUTH] Ed25519 sig verification failed for {}", pubkey_hex);
        return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
            {"error":"signature verification failed"}
        ))).into_response();
    }

    if state.max_remote_owners > 0 {
        let current = state.storage.count_distinct_owners().await;
        let remote = current.saturating_sub(1);
        let exists = state.storage.owner_exists(&pubkey_bytes).await;
        if !exists && remote >= state.max_remote_owners {
            warn!("[MPI_AUTH] Remote capacity reached: {}/{}", remote, state.max_remote_owners);
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "this node has reached maximum remote user capacity",
                "max_remote_owners": state.max_remote_owners,
            }))).into_response();
        }
    }

    debug!("[MPI_AUTH] Remote OK: {} ({} {})", &pubkey_hex[..8], method, path);

    let auth = AuthenticatedOwner::Remote { owner: pubkey_bytes, owner_hex: pubkey_hex };
    let mut req = Request::from_parts(parts, axum::body::Body::from(body_bytes));
    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

async fn handle_local_auth(
    state: Arc<MpiState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    let expected = match &state.api_secret {
        Some(s) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    };

    if let Some(expected) = expected {
        let auth_header = req.headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        let token = match auth_header {
            Some(h) if h.starts_with("Bearer ") => &h[7..],
            Some(h) if h.starts_with("bearer ") => &h[7..],
            _ => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
                {"error":"missing Authorization: Bearer <token>"}
            ))).into_response(),
        };

        let valid = token.len() == expected.len()
            && token.as_bytes().iter().zip(expected.as_bytes())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0;

        if !valid {
            warn!("[MPI_AUTH] Invalid Bearer token");
            return (StatusCode::UNAUTHORIZED, Json(serde_json::json!(
                {"error":"invalid token"}
            ))).into_response();
        }
    }

    let auth = AuthenticatedOwner::Local { owner: state.owner_key };
    let mut req = req;
    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

// ============================================
// Router — 31 endpoints (v2.5.2+Provenance)
// ============================================

pub fn build_mpi_router(state: Arc<MpiState>) -> Router {
    let router = Router::new()
        // ── Core endpoints (mpi_handlers.rs) ──
        .route("/api/mpi/remember",          post(mpi_handlers::mpi_remember))
        .route("/api/mpi/recall",            post(super::recall_handler::mpi_recall))
        .route("/api/mpi/recall/detail",     post(super::recall_handler::mpi_recall_detail))
        .route("/api/mpi/forget",            post(mpi_handlers::mpi_forget))
        .route("/api/mpi/status",            get(mpi_handlers::mpi_status))
        .route("/api/mpi/embed",             post(mpi_handlers::mpi_embed))
        // v2.5.2+Provenance: GET + PATCH on same path; provenance route registered after.
        // ⚠️ Order matters: /record/:id/provenance must come AFTER /record/:id
        //    to prevent `:record_id` greedily matching "xxxx/provenance".
        .route("/api/mpi/record/:record_id",
            get(mpi_handlers::mpi_get_record)
                .patch(mpi_handlers::mpi_patch_record))          // ← v2.5.2 PATCH
        .route("/api/mpi/record/:record_id/provenance",
            get(mpi_handlers::mpi_record_provenance))            // ← v2.5.2 GET provenance
        .route("/api/mpi/records/overview",  get(mpi_handlers::mpi_records_overview))
        // ── /log (log_handler.rs) ──
        .route("/api/mpi/log",               post(crate::api::log_handler::mpi_log))
        // ── v2.4.0: Cognitive graph (mpi_graph_handlers.rs) ──
        .route("/api/mpi/projects",                    get(mpi_graph_handlers::mpi_projects))
        .route("/api/mpi/projects/:id",                get(mpi_graph_handlers::mpi_project_detail))
        .route("/api/mpi/projects/:id/timeline",       get(mpi_graph_handlers::mpi_project_timeline))
        .route("/api/mpi/sessions/:id",                get(mpi_graph_handlers::mpi_session_detail))
        .route("/api/mpi/sessions/:id/conversation",   get(mpi_graph_handlers::mpi_session_conversation))
        .route("/api/mpi/sessions/:id/artifacts",      get(mpi_graph_handlers::mpi_session_artifacts))
        .route("/api/mpi/artifacts/:id",               get(mpi_graph_handlers::mpi_artifact_detail))
        .route("/api/mpi/artifacts/:id/versions",      get(mpi_graph_handlers::mpi_artifact_versions))
        .route("/api/mpi/entities/:id",                get(mpi_graph_handlers::mpi_entity_detail))
        .route("/api/mpi/entities",                    get(mpi_graph_handlers::mpi_entities_list))
        .route("/api/mpi/entities/:id/graph",          get(mpi_graph_handlers::mpi_entity_graph))
        .route("/api/mpi/communities",                 get(mpi_graph_handlers::mpi_communities))
        .route("/api/mpi/search",                      get(mpi_graph_handlers::mpi_search))
        .route("/api/mpi/entities/:id/timeline",       get(mpi_graph_handlers::mpi_entity_timeline))
        .route("/api/mpi/context/inject",              get(mpi_graph_handlers::mpi_context_inject))
        // ── v2.5.0+SuperNode: Task queue management (supernode_handlers.rs) ──
        .route("/api/mpi/supernode/tasks",             get(supernode_handlers::supernode_list_tasks))
        .route("/api/mpi/supernode/tasks/:id",         get(supernode_handlers::supernode_task_detail))
        .route("/api/mpi/supernode/tasks/:id/retry",   post(supernode_handlers::supernode_retry_task))
        .route("/api/mpi/supernode/tasks/:id/cancel",  post(supernode_handlers::supernode_cancel_task))
        .route("/api/mpi/supernode/usage",             get(supernode_handlers::supernode_usage))
        .route("/api/mpi/supernode/health",            get(supernode_handlers::supernode_health));

    info!(
        "[MPI] Router: 31 endpoints (bearer={}, remote={}, ner={}, graph={}, reranker={}, supernode={})",
        state.api_secret.as_ref().map_or(false, |s| !s.is_empty()),
        state.allow_remote_storage,
        state.ner_engine.is_some(),
        state.graph_enabled,
        state.reranker_engine.is_some(),
        state.llm_router.is_some(),
    );

    router
        .route_layer(middleware::from_fn_with_state(state.clone(), unified_auth_middleware))
        .with_state(state)
}
