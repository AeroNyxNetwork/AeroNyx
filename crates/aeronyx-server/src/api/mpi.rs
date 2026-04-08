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
//! - `auth.rs`               — v1.0.0-MultiTenant JWT token issuance
//! - `admin_handlers.rs`     — v1.0.0-MultiTenant Admin endpoints
//!
//! ## Modification History
//! v2.1.0                   - MPI endpoints with MVF fusion scoring
//! v2.1.0+MVF+Auth          - Bearer token auth middleware
//! v2.2.0                   - Added record/:id, records/overview, embed endpoints
//! v2.3.0+RemoteStorage     - Unified auth middleware (Bearer + Ed25519)
//! v2.4.0-GraphCognition    - Split into files; MpiState + graph extensions
//! v2.4.0+Reranker          - MpiState.reranker_engine
//! v2.4.0+Conversation      - MpiState.rawlog_key
//! v2.5.0+SuperNode Phase D - MpiState.llm_router; 6 supernode routes; 23→29
//! v2.5.2+Provenance        - PATCH /record/:id + GET /record/:id/provenance;
//!                            29→31 endpoints.
//! v1.0.0-MultiTenant       - MpiState extended for SaaS mode:
//!                            +mode, +storage_pool, +vector_pool, +volume_router,
//!                            +system_db, +jwt_secret, +token_ttl_secs.
//!                            unified_auth_middleware: added SaaS JWT branch.
//!                            build_mpi_router: POST /api/auth/token (SaaS only,
//!                            registered BEFORE middleware layer).
//!                            Admin routes registered in SaaS mode.
//!                            32 endpoints total (SaaS) / 31 (Local).
//!
//! ⚠️ Important Note for Next Developer:
//! - unified_auth_middleware MUST be applied to all routes via route_layer.
//!   AuthenticatedOwner is injected into extensions here; handlers depend on it.
//! - In SaaS mode, storage + vector_index extensions are ALSO injected by the
//!   middleware from StoragePool / VectorIndexPool. Handlers use Extension<>
//!   extractors to get them — they do NOT access MpiState.storage directly.
//! - In Local mode, storage + vector_index come from MpiState directly (unchanged).
//! - /api/auth/token is registered OUTSIDE the middleware layer — it is the
//!   auth entry point and cannot require a JWT to issue a JWT.
//! - Mode::Local storage/vector_index fields remain Some(). In Mode::Saas they
//!   are None. Handlers that use Extension<Arc<MemoryStorage>> work in both modes.
//! - Route registration order: /record/:id/provenance AFTER /record/:id.
//!   /artifacts/search BEFORE /artifacts/:id (see v2.5.3+ArtifactChain).
//! - owner_key field: in SaaS mode this is [0u8; 32] (placeholder).
//!   Never use it directly in SaaS mode — use AuthenticatedOwner from extensions.
//! - SaaS mode admin routes use api_secret Bearer token (same as local mode),
//!   NOT the user JWT. Admin is the operator, not an end-user.
//!
//! ## Last Modified
//! v2.5.2+Provenance  - +2 routes (patch record, provenance); 29→31
//! v2.5.3+ArtifactChain - +1 route (/artifacts/search); 31→32
//! v1.0.0-MultiTenant - MpiState SaaS fields; SaaS auth branch; auth + admin routes

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
// v1.0.0-MultiTenant: SaaS mode pools and infrastructure
use crate::services::memchain::{StoragePool, VectorIndexPool, VolumeRouter, SystemDb};

use super::mpi_handlers;
use super::mpi_graph_handlers;
use super::supernode_handlers;
use super::auth::{AuthState, verify_jwt, parse_pubkey_hex, issue_token};

// ============================================
// Constants
// ============================================

pub(crate) const AUTH_TIMESTAMP_TOLERANCE_SECS: u64 = 300;

// ============================================
// Mode Enum
// ============================================

/// Server operating mode.
///
/// - `Local`: single-user, api_secret Bearer auth, fixed owner_key.
///   All existing behavior preserved unchanged.
/// - `Saas`: multi-user, JWT auth, per-user StoragePool + VectorIndexPool.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Local,
    Saas,
}

// ============================================
// AuthenticatedOwner
// ============================================

#[derive(Debug, Clone)]
pub enum AuthenticatedOwner {
    Local  { owner: [u8; 32] },
    Remote { owner: [u8; 32], owner_hex: String },
    /// v1.0.0-MultiTenant: SaaS JWT-authenticated user.
    Saas   { owner: [u8; 32], owner_hex: String },
}

impl AuthenticatedOwner {
    #[must_use]
    pub fn owner_bytes(&self) -> [u8; 32] {
        match self {
            Self::Local  { owner }      => *owner,
            Self::Remote { owner, .. }  => *owner,
            Self::Saas   { owner, .. }  => *owner,
        }
    }

    #[must_use]
    pub fn owner_hex(&self) -> String {
        match self {
            Self::Local  { owner }          => hex::encode(owner),
            Self::Remote { owner_hex, .. }  => owner_hex.clone(),
            Self::Saas   { owner_hex, .. }  => owner_hex.clone(),
        }
    }

    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }

    #[must_use]
    pub fn is_saas(&self) -> bool {
        matches!(self, Self::Saas { .. })
    }
}

// ============================================
// MpiState
// ============================================

pub struct MpiState {
    // ── Operating mode ──────────────────────────────────────────────
    pub mode: Mode,

    // ── Local mode: single-user storage (Some in Local, None in Saas) ──
    /// Single-user MemoryStorage instance (Local mode only).
    pub storage: Option<Arc<MemoryStorage>>,
    /// Single-user VectorIndex instance (Local mode only).
    pub vector_index: Option<Arc<VectorIndex>>,

    // ── Shared fields (both modes) ───────────────────────────────────
    pub identity: IdentityKeyPair,
    pub identity_cache: RwLock<HashMap<String, Vec<MemoryRecord>>>,
    pub index_ready: AtomicBool,
    pub user_weights: Arc<RwLock<HashMap<String, WeightVector>>>,
    pub mvf_alpha: f32,
    pub mvf_enabled: bool,
    pub session_embeddings: RwLock<HashMap<String, VecDeque<Vec<f32>>>>,
    pub mvf_baseline: RwLock<Option<BaselineSnapshot>>,
    /// Local mode owner public key. In SaaS mode this is [0u8; 32] (placeholder).
    /// ⚠️ Never read this in SaaS mode — use AuthenticatedOwner from extensions.
    pub owner_key: [u8; 32],
    /// Local mode API secret (Bearer token auth).
    pub api_secret: Option<String>,
    pub embed_engine: Option<Arc<EmbedEngine>>,
    pub allow_remote_storage: bool,
    pub max_remote_owners: usize,
    pub ner_engine: Option<Arc<NerEngine>>,
    pub graph_enabled: bool,
    pub entropy_filter_enabled: bool,
    pub reranker_engine: Option<Arc<RerankerEngine>>,
    pub rawlog_key: Option<[u8; 32]>,
    pub llm_router: Option<Arc<LlmRouter>>,

    // ── SaaS mode: multi-user pools (Some in Saas, None in Local) ───
    /// Per-user MemoryStorage connection pool (SaaS mode only).
    pub storage_pool: Option<Arc<StoragePool>>,
    /// Per-user VectorIndex pool (SaaS mode only).
    pub vector_pool: Option<Arc<VectorIndexPool>>,
    /// Volume router for disk assignment (SaaS mode only).
    pub volume_router: Option<Arc<VolumeRouter>>,
    /// Global metadata DB (SaaS mode only).
    pub system_db: Option<Arc<SystemDb>>,
    /// JWT signing secret (SaaS mode only).
    pub jwt_secret: Option<String>,
    /// JWT token TTL in seconds (default 86400 = 24h).
    pub token_ttl_secs: u64,
}

// ── Backward-compatible constructor helpers ──────────────────────────

impl MpiState {
    /// Build a Local-mode MpiState with the same signature as pre-MultiTenant code.
    ///
    /// All SaaS fields are set to None. This is the ONLY constructor that should
    /// be called from server.rs for Local mode — it ensures all new fields are
    /// correctly zero-initialized.
    #[allow(clippy::too_many_arguments)]
    pub fn local(
        storage: Arc<MemoryStorage>,
        vector_index: Arc<VectorIndex>,
        identity: IdentityKeyPair,
        identity_cache: RwLock<HashMap<String, Vec<MemoryRecord>>>,
        index_ready: AtomicBool,
        user_weights: Arc<RwLock<HashMap<String, WeightVector>>>,
        mvf_alpha: f32,
        mvf_enabled: bool,
        session_embeddings: RwLock<HashMap<String, VecDeque<Vec<f32>>>>,
        mvf_baseline: RwLock<Option<BaselineSnapshot>>,
        owner_key: [u8; 32],
        api_secret: Option<String>,
        embed_engine: Option<Arc<EmbedEngine>>,
        allow_remote_storage: bool,
        max_remote_owners: usize,
        ner_engine: Option<Arc<NerEngine>>,
        graph_enabled: bool,
        entropy_filter_enabled: bool,
        reranker_engine: Option<Arc<RerankerEngine>>,
        rawlog_key: Option<[u8; 32]>,
        llm_router: Option<Arc<LlmRouter>>,
    ) -> Self {
        Self {
            mode: Mode::Local,
            storage: Some(storage),
            vector_index: Some(vector_index),
            identity,
            identity_cache,
            index_ready,
            user_weights,
            mvf_alpha,
            mvf_enabled,
            session_embeddings,
            mvf_baseline,
            owner_key,
            api_secret,
            embed_engine,
            allow_remote_storage,
            max_remote_owners,
            ner_engine,
            graph_enabled,
            entropy_filter_enabled,
            reranker_engine,
            rawlog_key,
            llm_router,
            // SaaS fields — None in Local mode
            storage_pool: None,
            vector_pool: None,
            volume_router: None,
            system_db: None,
            jwt_secret: None,
            token_ttl_secs: 86400,
        }
    }
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
    match state.mode {
        Mode::Local => {
            // Check for remote Ed25519 auth first (X-MemChain-Signature header).
            if req.headers().contains_key("x-memchain-signature") {
                return handle_remote_auth(state, req, next).await;
            }
            handle_local_auth(state, req, next).await
        }
        Mode::Saas => {
            handle_saas_jwt_auth(state, req, next).await
        }
    }
}

// ── SaaS: JWT Auth ────────────────────────────────────────────────────

async fn handle_saas_jwt_auth(
    state: Arc<MpiState>,
    mut req: Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    // ── 1. Extract Bearer JWT ─────────────────────────────────────────
    let token = match extract_bearer_token(req.headers()) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "missing Authorization: Bearer <token>" })),
            )
                .into_response();
        }
    };

    // ── 2. Verify JWT signature + expiry ─────────────────────────────
    let jwt_secret = match &state.jwt_secret {
        Some(s) => s.as_str(),
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
                .into_response();
        }
    };

    let claims = match verify_jwt(token, jwt_secret) {
        Ok(c) => c,
        Err(e) => {
            use jsonwebtoken::errors::ErrorKind;
            let msg = match e.kind() {
                ErrorKind::ExpiredSignature => "token expired",
                ErrorKind::InvalidSignature | ErrorKind::InvalidToken => "invalid token",
                _ => "invalid token claims",
            };
            warn!(error = %e, "[MPI_AUTH_SAAS] JWT verification failed");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": msg })),
            )
                .into_response();
        }
    };

    // ── 3. Extract owner pubkey from sub (single source of truth) ────
    let owner = match parse_pubkey_hex(&claims.sub) {
        Ok(b) => b,
        Err(e) => {
            warn!(sub = &claims.sub[..8.min(claims.sub.len())], "[MPI_AUTH_SAAS] Invalid sub claim");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": format!("invalid token claims: {}", e) })),
            )
                .into_response();
        }
    };

    // ── 4. Get or create per-user Storage from pool ───────────────────
    let pool = match &state.storage_pool {
        Some(p) => p,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
                .into_response();
        }
    };

    let storage = match pool.get_or_create(&owner).await {
        Ok(s) => s,
        Err(e) => {
            // Do not leak volume details to the client.
            tracing::error!(error = %e, "[MPI_AUTH_SAAS] StoragePool error");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
                .into_response();
        }
    };

    // ── 5. Get or create per-user VectorIndex from pool ───────────────
    let vpool = match &state.vector_pool {
        Some(p) => p,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
                .into_response();
        }
    };

    let vector_index = match vpool.get_or_create(&owner) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(error = %e, "[MPI_AUTH_SAAS] VectorIndexPool error");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "internal error" })),
            )
                .into_response();
        }
    };

    // ── 6. Update last-active timestamp (Miner scheduling) ───────────
    if let Some(ref sys_db) = state.system_db {
        let db = Arc::clone(sys_db);
        let owner_copy = owner;
        // Fire-and-forget: do not block the request on this write.
        tokio::spawn(async move {
            let _ = db.update_last_active(&owner_copy).await;
        });
    }

    // ── 7. Inject extensions (handlers use Extension<> extractors) ────
    let auth = AuthenticatedOwner::Saas {
        owner,
        owner_hex: claims.sub.clone(),
    };
    req.extensions_mut().insert(auth);
    req.extensions_mut().insert(storage);
    req.extensions_mut().insert(vector_index);

    debug!(
        owner = &claims.sub[..8],
        "[MPI_AUTH_SAAS] Authenticated"
    );

    next.run(req).await.into_response()
}

// ── Local: Remote Ed25519 Auth (unchanged from v2.3.0) ───────────────

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

    // Check remote capacity against the single-user storage.
    if let Some(ref storage) = state.storage {
        if state.max_remote_owners > 0 {
            let current = storage.count_distinct_owners().await;
            let remote = current.saturating_sub(1);
            let exists = storage.owner_exists(&pubkey_bytes).await;
            if !exists && remote >= state.max_remote_owners {
                warn!("[MPI_AUTH] Remote capacity reached: {}/{}", remote, state.max_remote_owners);
                return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({
                    "error": "this node has reached maximum remote user capacity",
                    "max_remote_owners": state.max_remote_owners,
                }))).into_response();
            }
        }
    }

    debug!("[MPI_AUTH] Remote OK: {} ({} {})", &pubkey_hex[..8], method, path);

    let auth = AuthenticatedOwner::Remote { owner: pubkey_bytes, owner_hex: pubkey_hex };
    let mut req = Request::from_parts(parts, axum::body::Body::from(body_bytes));
    req.extensions_mut().insert(auth);

    // Local mode: inject the single-user storage + vector_index as extensions
    // so that handlers can use the same Extension<> extractor pattern in both modes.
    if let (Some(ref st), Some(ref vi)) = (&state.storage, &state.vector_index) {
        req.extensions_mut().insert(Arc::clone(st));
        req.extensions_mut().insert(Arc::clone(vi));
    }

    next.run(req).await.into_response()
}

// ── Local: Bearer Token Auth (unchanged from v2.1.0+MVF+Auth) ────────

async fn handle_local_auth(
    state: Arc<MpiState>,
    mut req: Request<axum::body::Body>,
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

    // Local mode: inject storage + vector_index as extensions for handler
    // consistency (handlers use Extension<> in both Local and SaaS modes).
    if let (Some(ref st), Some(ref vi)) = (&state.storage, &state.vector_index) {
        req.extensions_mut().insert(Arc::clone(st));
        req.extensions_mut().insert(Arc::clone(vi));
    }

    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

// ── Helper: extract Bearer token from Authorization header ───────────

fn extract_bearer_token(
    headers: &axum::http::HeaderMap,
) -> Option<&str> {
    let header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())?;

    if header.starts_with("Bearer ") {
        Some(&header[7..])
    } else if header.starts_with("bearer ") {
        Some(&header[7..])
    } else {
        None
    }
}

// ============================================
// Admin Auth Middleware (SaaS mode)
// ============================================

/// Admin endpoint auth: validates api_secret Bearer token.
///
/// Admin endpoints are operator-only (not end-user). They use the same
/// api_secret mechanism as Local mode, NOT the user JWT system.
async fn admin_auth_middleware(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    let expected = match &state.api_secret {
        Some(s) if !s.is_empty() => s.as_str(),
        _ => {
            // No api_secret configured — admin endpoints are open.
            // This is acceptable for development but warn in production.
            warn!("[ADMIN_AUTH] No api_secret configured — admin endpoints are unprotected");
            return next.run(req).await.into_response();
        }
    };

    let token = match extract_bearer_token(req.headers()) {
        Some(t) => t,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "missing Authorization: Bearer <admin_secret>" })),
            )
                .into_response();
        }
    };

    let valid = token.len() == expected.len()
        && token.as_bytes().iter().zip(expected.as_bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0;

    if !valid {
        warn!("[ADMIN_AUTH] Invalid admin token");
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "invalid admin token" })),
        )
            .into_response();
    }

    next.run(req).await.into_response()
}

// ============================================
// Router
// ============================================

/// Build the complete MPI router.
///
/// ## Local mode (31 endpoints)
/// All existing routes unchanged. Auth endpoint NOT registered.
///
/// ## SaaS mode (32 MPI endpoints + 1 auth + 4 admin = 37 total)
/// - `POST /api/auth/token`: registered OUTSIDE the auth middleware layer
/// - Admin routes: registered under separate admin_auth_middleware
/// - All 31 existing MPI routes unchanged (now use Extension<> for storage)
pub fn build_mpi_router(state: Arc<MpiState>) -> Router {
    let is_saas = state.mode == Mode::Saas;

    // ── Core MPI routes (under unified_auth_middleware) ───────────────
    let mpi_routes = Router::new()
        // Core endpoints (mpi_handlers.rs)
        .route("/api/mpi/remember",          post(mpi_handlers::mpi_remember))
        .route("/api/mpi/recall",            post(super::recall_handler::mpi_recall))
        .route("/api/mpi/recall/detail",     post(super::recall_handler::mpi_recall_detail))
        .route("/api/mpi/forget",            post(mpi_handlers::mpi_forget))
        .route("/api/mpi/status",            get(mpi_handlers::mpi_status))
        .route("/api/mpi/embed",             post(mpi_handlers::mpi_embed))
        // v2.5.2+Provenance: GET + PATCH; provenance AFTER plain :record_id
        // ⚠️ /record/:id/provenance MUST be registered after /record/:id
        .route("/api/mpi/record/:record_id",
            get(mpi_handlers::mpi_get_record)
                .patch(mpi_handlers::mpi_patch_record))
        .route("/api/mpi/record/:record_id/provenance",
            get(mpi_handlers::mpi_record_provenance))
        .route("/api/mpi/records/overview",  get(mpi_handlers::mpi_records_overview))
        // /log (log_handler.rs)
        .route("/api/mpi/log",               post(crate::api::log_handler::mpi_log))
        // v2.4.0: Cognitive graph (mpi_graph_handlers.rs)
        .route("/api/mpi/projects",                    get(mpi_graph_handlers::mpi_projects))
        .route("/api/mpi/projects/:id",                get(mpi_graph_handlers::mpi_project_detail))
        .route("/api/mpi/projects/:id/timeline",       get(mpi_graph_handlers::mpi_project_timeline))
        .route("/api/mpi/sessions/:id",                get(mpi_graph_handlers::mpi_session_detail))
        .route("/api/mpi/sessions/:id/conversation",   get(mpi_graph_handlers::mpi_session_conversation))
        .route("/api/mpi/sessions/:id/artifacts",      get(mpi_graph_handlers::mpi_session_artifacts))
        // ⚠️ v2.5.3+ArtifactChain: /artifacts/search MUST be before /artifacts/:id
        //    to prevent "search" being captured as the :id path parameter.
        .route("/api/mpi/artifacts/search",            get(mpi_graph_handlers::mpi_artifacts_search))
        .route("/api/mpi/artifacts/:id",               get(mpi_graph_handlers::mpi_artifact_detail))
        .route("/api/mpi/artifacts/:id/versions",      get(mpi_graph_handlers::mpi_artifact_versions))
        .route("/api/mpi/entities/:id",                get(mpi_graph_handlers::mpi_entity_detail))
        .route("/api/mpi/entities",                    get(mpi_graph_handlers::mpi_entities_list))
        .route("/api/mpi/entities/:id/graph",          get(mpi_graph_handlers::mpi_entity_graph))
        .route("/api/mpi/communities",                 get(mpi_graph_handlers::mpi_communities))
        .route("/api/mpi/search",                      get(mpi_graph_handlers::mpi_search))
        .route("/api/mpi/entities/:id/timeline",       get(mpi_graph_handlers::mpi_entity_timeline))
        .route("/api/mpi/context/inject",              get(mpi_graph_handlers::mpi_context_inject))
        // v2.5.0+SuperNode: Task queue management (supernode_handlers.rs)
        .route("/api/mpi/supernode/tasks",             get(supernode_handlers::supernode_list_tasks))
        .route("/api/mpi/supernode/tasks/:id",         get(supernode_handlers::supernode_task_detail))
        .route("/api/mpi/supernode/tasks/:id/retry",   post(supernode_handlers::supernode_retry_task))
        .route("/api/mpi/supernode/tasks/:id/cancel",  post(supernode_handlers::supernode_cancel_task))
        .route("/api/mpi/supernode/usage",             get(supernode_handlers::supernode_usage))
        .route("/api/mpi/supernode/health",            get(supernode_handlers::supernode_health))
        // Apply unified auth middleware to all MPI routes.
        .route_layer(middleware::from_fn_with_state(state.clone(), unified_auth_middleware))
        .with_state(Arc::clone(&state));

    let router = Router::new().merge(mpi_routes);

    // ── SaaS-only routes ──────────────────────────────────────────────
    let router = if is_saas {
        // Auth endpoint: OUTSIDE the middleware layer (no JWT required to get a JWT).
        let auth_state = {
            let jwt_secret = state.jwt_secret.clone().unwrap_or_default();
            let ttl = state.token_ttl_secs;
            AuthState { jwt_secret, token_ttl_secs: ttl }
        };

        let auth_routes = Router::new()
            .route("/api/auth/token", post(issue_token))
            .with_state(auth_state);

        // Admin routes: separate admin_auth_middleware (api_secret, not JWT).
        let admin_routes = Router::new()
            .route("/api/admin/volumes",         get(super::admin_handlers::admin_volumes))
            .route("/api/admin/volumes/reload",  post(super::admin_handlers::admin_volumes_reload))
            .route("/api/admin/pool/stats",      get(super::admin_handlers::admin_pool_stats))
            .route("/api/admin/usage",           get(super::admin_handlers::admin_usage))
            .route_layer(middleware::from_fn_with_state(state.clone(), admin_auth_middleware))
            .with_state(Arc::clone(&state));

        router.merge(auth_routes).merge(admin_routes)
    } else {
        router
    };

    info!(
        mode = ?state.mode,
        endpoints = if is_saas { "32 MPI + 1 auth + 4 admin" } else { "31 MPI" },
        bearer = state.api_secret.as_ref().map_or(false, |s| !s.is_empty()),
        remote = state.allow_remote_storage,
        ner = state.ner_engine.is_some(),
        graph = state.graph_enabled,
        reranker = state.reranker_engine.is_some(),
        supernode = state.llm_router.is_some(),
        saas_pools = state.storage_pool.is_some(),
        "[MPI] Router initialized"
    );

    router
}
