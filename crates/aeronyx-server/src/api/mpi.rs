// ============================================
// File: crates/aeronyx-server/src/api/mpi.rs
// ============================================
//! # MPI — Memory Protocol Interface (v2.4.0 — Core Module)
//!
//! ## File Split (v2.4.0)
//! mpi.rs was split into 3 files for maintainability:
//! - `mpi.rs` (THIS FILE) — MpiState, AuthenticatedOwner, auth middleware, router,
//!   helpers, request/response types. This is the entry point; mod.rs re-exports from here.
//! - `mpi_handlers.rs` — Original 8 endpoint handlers (remember, recall, forget,
//!   status, embed, record, overview) + their tests
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints (11 new) + their tests
//!
//! All 3 files impl handlers that share MpiState via Arc<MpiState>.
//! External API is unchanged — api/mod.rs re-exports {build_mpi_router, MpiState, BaselineSnapshot}.
//!
//! ## Main Functionality
//! - MpiState: shared state for all 19 MPI endpoints
//! - AuthenticatedOwner: request extension for owner identification
//! - Unified auth middleware: Bearer token (local) + Ed25519 signature (remote)
//! - Router construction with all 19 routes + auth middleware layer
//! - Shared helper functions used by handlers in all 3 files
//!
//! ⚠️ Important Note for Next Developer:
//! - AuthenticatedOwner is the SINGLE source of truth for "who is making this request"
//! - NEVER use state.owner_key directly in handlers — always use the extension
//! - mpi_handlers.rs and mpi_graph_handlers.rs import types from this file via `super::mpi::`
//! - When adding new endpoints, register the route in build_mpi_router() here,
//!   but implement the handler in the appropriate handlers file
//!
//! ## Last Modified
//! v2.1.0 - MPI endpoints with MVF fusion scoring
//! v2.1.0+MVF+Auth - Added Bearer token auth middleware
//! v2.2.0 - Added record/:id, records/overview, embed endpoints
//! v2.3.0+RemoteStorage - Unified auth middleware (Bearer + Ed25519)
//! v2.4.0-GraphCognition - 🌟 Split into 3 files; MpiState extended with
//!   ner_engine, graph_enabled, entropy_filter_enabled; 11 new graph endpoints;
//!   hybrid recall pipeline; status extended with graph_stats

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

// Sibling handler modules
use super::mpi_handlers;
use super::mpi_graph_handlers;

// ============================================
// Constants
// ============================================

/// Maximum allowed clock skew for Ed25519 signature timestamps (seconds).
pub(crate) const AUTH_TIMESTAMP_TOLERANCE_SECS: u64 = 300;

// ============================================
// AuthenticatedOwner (v2.3.0)
// ============================================

/// Represents the authenticated identity making the current request.
///
/// Inserted as a request extension by the unified auth middleware.
/// All handlers MUST extract this instead of using `state.owner_key` directly.
#[derive(Debug, Clone)]
pub enum AuthenticatedOwner {
    /// Local node operator (Bearer token auth or open access).
    Local { owner: [u8; 32] },
    /// Remote user (Ed25519 signature auth).
    Remote { owner: [u8; 32], owner_hex: String },
}

impl AuthenticatedOwner {
    /// Get the 32-byte owner key for storage operations.
    #[must_use]
    pub fn owner_bytes(&self) -> [u8; 32] {
        match self {
            Self::Local { owner } => *owner,
            Self::Remote { owner, .. } => *owner,
        }
    }

    /// Get the hex-encoded owner key.
    #[must_use]
    pub fn owner_hex(&self) -> String {
        match self {
            Self::Local { owner } => hex::encode(owner),
            Self::Remote { owner_hex, .. } => owner_hex.clone(),
        }
    }

    /// Whether this is a remote (Ed25519 signature) authenticated request.
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }
}

// ============================================
// Shared State
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
    /// v2.3.0: Whether this node accepts remote storage requests.
    pub allow_remote_storage: bool,
    /// v2.3.0: Maximum number of distinct remote owners this node serves.
    pub max_remote_owners: usize,
    /// v2.4.0: Local NER engine (GLiNER via ONNX Runtime).
    pub ner_engine: Option<Arc<NerEngine>>,
    /// v2.4.0: Whether knowledge graph traversal is enabled in recall.
    pub graph_enabled: bool,
    /// v2.4.0: Whether entropy filtering is enabled on /log ingestion.
    pub entropy_filter_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub positive_rate: f32,
    pub sample_size: usize,
    pub frozen_at: i64,
}

// ============================================
// Shared Helpers (used by all handler files)
// ============================================

/// Extract AuthenticatedOwner from request extensions.
/// Panics if the auth middleware was not applied (programming error).
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
    let ascii_count = text.as_bytes()[..sample_len].iter().filter(|b| b.is_ascii()).count();
    if (ascii_count as f64 / sample_len as f64) > 0.80 {
        (len + 3) / 4
    } else {
        (len * 2 / 3).max(1)
    }
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

pub(crate) fn default_layer() -> String { "episode".into() }
pub(crate) fn default_source() -> String { "unknown".into() }
pub(crate) fn default_model() -> String { "default".into() }
pub(crate) fn default_top_k() -> usize { 10 }
pub(crate) fn default_token_budget() -> usize { 4000 }
pub(crate) fn default_include_graph() -> bool { true }
pub(crate) fn default_list_limit() -> usize { 20 }

// ============================================
// Unified Auth Middleware (v2.3.0)
// ============================================

/// Unified authentication middleware for all MPI endpoints.
/// Handles Ed25519 signature (remote) and Bearer token (local) in one pass.
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

/// Handle Ed25519 signature authentication for remote users.
async fn handle_remote_auth(
    state: Arc<MpiState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    let pubkey_hex = match req.headers().get("x-memchain-publickey").and_then(|v| v.to_str().ok()) {
        Some(v) => v.to_string(),
        None => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "missing X-MemChain-PublicKey header"}))).into_response(),
    };
    let timestamp_str = match req.headers().get("x-memchain-timestamp").and_then(|v| v.to_str().ok()) {
        Some(v) => v.to_string(),
        None => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "missing X-MemChain-Timestamp header"}))).into_response(),
    };
    let signature_hex = match req.headers().get("x-memchain-signature").and_then(|v| v.to_str().ok()) {
        Some(v) => v.to_string(),
        None => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "missing X-MemChain-Signature header"}))).into_response(),
    };

    let pubkey_bytes = match hex::decode(&pubkey_hex) {
        Ok(b) if b.len() == 32 => { let mut arr = [0u8; 32]; arr.copy_from_slice(&b); arr }
        _ => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "invalid X-MemChain-PublicKey: expected 64 hex chars"}))).into_response(),
    };

    let identity_pubkey = match IdentityPublicKey::from_bytes(&pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "invalid Ed25519 public key"}))).into_response(),
    };

    let timestamp: u64 = match timestamp_str.parse() {
        Ok(ts) => ts,
        Err(_) => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "invalid X-MemChain-Timestamp"}))).into_response(),
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
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "this node does not accept remote storage requests"}))).into_response();
    }

    let signature_bytes = match hex::decode(&signature_hex) {
        Ok(b) if b.len() == 64 => { let mut arr = [0u8; 64]; arr.copy_from_slice(&b); arr }
        _ => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "invalid X-MemChain-Signature: expected 128 hex chars"}))).into_response(),
    };

    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "failed to read request body"}))).into_response(),
    };

    let body_hash = Sha256::digest(&body_bytes);
    let mut message_hasher = Sha256::new();
    message_hasher.update(timestamp_str.as_bytes());
    message_hasher.update(method.as_bytes());
    message_hasher.update(path.as_bytes());
    message_hasher.update(&body_hash);
    let signed_message = message_hasher.finalize();

    if identity_pubkey.verify(&signed_message, &signature_bytes).is_err() {
        warn!("[MPI_AUTH] Ed25519 signature verification failed for {}", pubkey_hex);
        return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "signature verification failed"}))).into_response();
    }

    if state.max_remote_owners > 0 {
        let current_owners = state.storage.count_distinct_owners().await;
        let remote_count = current_owners.saturating_sub(1);
        let owner_exists = state.storage.owner_exists(&pubkey_bytes).await;
        if !owner_exists && remote_count >= state.max_remote_owners {
            warn!("[MPI_AUTH] Remote owner capacity reached: {}/{}", remote_count, state.max_remote_owners);
            return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                "error": "this node has reached maximum remote user capacity",
                "max_remote_owners": state.max_remote_owners,
            }))).into_response();
        }
    }

    debug!("[MPI_AUTH] Remote auth OK: {} (method={}, path={})", &pubkey_hex[..8], method, path);

    let auth = AuthenticatedOwner::Remote { owner: pubkey_bytes, owner_hex: pubkey_hex };
    let mut req = Request::from_parts(parts, axum::body::Body::from(body_bytes));
    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

/// Handle Bearer token authentication for local node operator.
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
            _ => return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "missing Authorization: Bearer <token>"}))).into_response(),
        };

        let token_bytes = token.as_bytes();
        let expected_bytes = expected.as_bytes();
        let valid = if token_bytes.len() == expected_bytes.len() {
            token_bytes.iter().zip(expected_bytes.iter()).fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
        } else { false };

        if !valid {
            warn!("[MPI_AUTH] Invalid Bearer token from request");
            return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({"error": "invalid token"}))).into_response();
        }
    }

    let auth = AuthenticatedOwner::Local { owner: state.owner_key };
    let mut req = req;
    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

// ============================================
// Router (v2.4.0: 19 endpoints across 3 files)
// ============================================

/// Build the MPI router with unified auth middleware.
///
/// Routes are registered here but handlers live in sibling files:
/// - mpi_handlers.rs: remember, recall, forget, status, embed, record, overview
/// - mpi_graph_handlers.rs: projects, sessions, artifacts, entities, communities
/// - log_handler.rs: /log (already separate since v2.1.0)
pub fn build_mpi_router(state: Arc<MpiState>) -> Router {
    let router = Router::new()
        // ── Original endpoints (mpi_handlers.rs) ──
        .route("/api/mpi/remember", post(mpi_handlers::mpi_remember))
        .route("/api/mpi/recall", post(mpi_handlers::mpi_recall))
        .route("/api/mpi/forget", post(mpi_handlers::mpi_forget))
        .route("/api/mpi/status", get(mpi_handlers::mpi_status))
        .route("/api/mpi/embed", post(mpi_handlers::mpi_embed))
        .route("/api/mpi/record/:record_id", get(mpi_handlers::mpi_get_record))
        .route("/api/mpi/records/overview", get(mpi_handlers::mpi_records_overview))
        // ── /log (log_handler.rs — separate since v2.1.0) ──
        .route("/api/mpi/log", post(crate::api::log_handler::mpi_log))
        // ── v2.4.0: Cognitive graph endpoints (mpi_graph_handlers.rs) ──
        .route("/api/mpi/projects", get(mpi_graph_handlers::mpi_projects))
        .route("/api/mpi/projects/:id", get(mpi_graph_handlers::mpi_project_detail))
        .route("/api/mpi/projects/:id/timeline", get(mpi_graph_handlers::mpi_project_timeline))
        .route("/api/mpi/sessions/:id", get(mpi_graph_handlers::mpi_session_detail))
        .route("/api/mpi/sessions/:id/conversation", get(mpi_graph_handlers::mpi_session_conversation))
        .route("/api/mpi/sessions/:id/artifacts", get(mpi_graph_handlers::mpi_session_artifacts))
        .route("/api/mpi/artifacts/:id", get(mpi_graph_handlers::mpi_artifact_detail))
        .route("/api/mpi/artifacts/:id/versions", get(mpi_graph_handlers::mpi_artifact_versions))
        .route("/api/mpi/entities/:id", get(mpi_graph_handlers::mpi_entity_detail))
        .route("/api/mpi/entities/:id/graph", get(mpi_graph_handlers::mpi_entity_graph))
        .route("/api/mpi/communities", get(mpi_graph_handlers::mpi_communities));

    info!(
        "[MPI] Router: 19 endpoints (bearer={}, remote={}, ner={}, graph={})",
        state.api_secret.as_ref().map_or(false, |s| !s.is_empty()),
        state.allow_remote_storage,
        state.ner_engine.is_some(),
        state.graph_enabled,
    );

    router
        .route_layer(middleware::from_fn_with_state(state.clone(), unified_auth_middleware))
        .with_state(state)
}
