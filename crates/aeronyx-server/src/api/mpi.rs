// ============================================
// File: crates/aeronyx-server/src/api/mpi.rs
// ============================================
//! # MPI — Memory Protocol Interface (v2.3.0 — Phase 1 Remote Storage)
//!
//! Endpoints: remember, recall, forget, status, log, embed, record/:id, records/overview
//! recall hot path target: < 50ms
//!
//! ## Modification Reason (v2.3.0+RemoteStorage)
//! Phase 1 Remote Storage Gateway:
//! - Unified auth middleware: Bearer token (local) + Ed25519 signature (remote)
//! - All handlers extract `owner` from request extension (not state.owner_key)
//! - Owner isolation: remote users can only access their own records
//! - max_remote_owners capacity check for remote users
//! - mpi_get_record and mpi_forget now enforce owner verification (security fix)
//!
//! ## Previous Modifications
//! v2.1.0+MVF+Auth — Added Bearer token authentication middleware
//! v2.2.0 — Added record/:id and records/overview endpoints, WS MPI Proxy
//!
//! ## Main Functionality
//! - 8 MPI endpoints: remember, recall, forget, status, log, embed, record/:id, records/overview
//! - Dual auth: Bearer token (local owner) + Ed25519 signature (remote owner)
//! - MVF 9-dim scoring with real feedback data (φ₄, φ₈ from Schema v4)
//! - Identity forced injection on recall (per-owner)
//! - Session centroid tracking for φ₇
//! - Owner isolation for all data operations
//!
//! ## Dependencies
//! - MpiState shared across all handlers via Arc
//! - storage.rs for SQLite operations
//! - vector.rs for similarity search
//! - mvf.rs for scoring
//! - graph.rs for co-occurrence
//! - log_handler.rs for /log endpoint
//! - config.rs for api_secret, allow_remote_storage, max_remote_owners
//! - aeronyx_core::crypto::IdentityPublicKey for Ed25519 signature verification
//!
//! ## Main Logical Flow
//! 1. Request arrives → unified_auth_middleware determines auth type:
//!    a. X-MemChain-Signature header present → Ed25519 signature verification
//!       → owner = signer's public key bytes (remote user)
//!    b. Authorization: Bearer header → constant-time token comparison
//!       → owner = state.owner_key (local node operator)
//!    c. No auth configured + no signature → open access with state.owner_key
//! 2. AuthenticatedOwner inserted as request extension
//! 3. Route to handler (remember/recall/forget/status/log/embed/record/overview)
//! 4. Handler extracts AuthenticatedOwner from extension
//! 5. Response returned
//!
//! ## Security Model (v2.3.0)
//! - Remote users authenticate via Ed25519 signature over:
//!   SHA256(timestamp || method || path || SHA256(body))
//! - Timestamp must be within ±300 seconds of server time (replay protection)
//! - Remote user's public key becomes their `owner` key (data isolation)
//! - Remote users CANNOT access node operator's data or other remote users' data
//! - Node operator CANNOT decrypt remote user's record content (client-side encryption)
//!   but CAN see embeddings (required for vector search)
//! - mpi_get_record and mpi_forget enforce owner == record.owner
//!
//! ⚠️ Important Note for Next Developer:
//! - AuthenticatedOwner is the SINGLE source of truth for "who is making this request"
//! - NEVER use state.owner_key directly in handlers — always use the extension
//! - The unified auth middleware handles both local and remote auth in one pass
//! - Ed25519 signature verification uses aeronyx_core::crypto::IdentityPublicKey
//! - Timestamp skew tolerance (AUTH_TIMESTAMP_TOLERANCE_SECS) prevents replay attacks
//!   but also means server clock must be reasonably accurate
//! - /log endpoint is EXCLUDED from remote access (security: rule engine runs locally only)
//!
//! ## Last Modified
//! v2.1.0 - MPI endpoints with MVF fusion scoring
//! v2.1.0+MVF - Fixed hardcoded feedback/conflict in compute_features()
//! v2.1.0+MVF+Auth - Added Bearer token auth middleware
//! v2.2.0 - Added record/:id, records/overview, embed endpoints
//! v2.3.0+RemoteStorage - 🌟 Phase 1: Unified auth middleware (Bearer + Ed25519),
//!   AuthenticatedOwner extension, owner isolation in all handlers,
//!   owner verification in get_record and forget (security fix),
//!   max_remote_owners capacity check, /log restricted to local-only

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::extract::Path;
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

use crate::services::memchain::{MemoryStorage, VectorIndex, compute_recall_score};
use crate::services::memchain::mvf::{self, WeightVector};
use crate::services::memchain::graph;
use crate::services::memchain::EmbedEngine;

// ============================================
// Constants
// ============================================

/// Maximum allowed clock skew for Ed25519 signature timestamps (seconds).
/// Remote requests with timestamps older than this are rejected (replay protection).
/// 300 seconds (5 minutes) allows for reasonable clock drift between client and server.
const AUTH_TIMESTAMP_TOLERANCE_SECS: u64 = 300;

// ============================================
// AuthenticatedOwner (v2.3.0)
// ============================================

/// Represents the authenticated identity making the current request.
///
/// Inserted as a request extension by the unified auth middleware.
/// All handlers MUST extract this instead of using `state.owner_key` directly.
///
/// ## Variants
/// - `Local`: Authenticated via Bearer token → owner is the node operator
/// - `Remote`: Authenticated via Ed25519 signature → owner is the signer's public key
///
/// ## Usage in Handlers
/// ```rust,ignore
/// let auth = req.extensions().get::<AuthenticatedOwner>()
///     .expect("auth middleware must set AuthenticatedOwner");
/// let owner = auth.owner_bytes();
/// ```
#[derive(Debug, Clone)]
pub enum AuthenticatedOwner {
    /// Local node operator (Bearer token auth or open access).
    /// Owner key is `state.owner_key` (the node's Ed25519 public key).
    Local {
        owner: [u8; 32],
    },
    /// Remote user (Ed25519 signature auth).
    /// Owner key is the signer's Ed25519 public key bytes.
    Remote {
        owner: [u8; 32],
        /// Hex-encoded public key (cached to avoid repeated hex::encode)
        owner_hex: String,
    },
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
    /// MPI Bearer token secret for API authentication (local requests).
    /// When Some (non-empty), local requests must include `Authorization: Bearer <secret>`.
    /// When None, local auth is disabled (backward compatible, loopback-only mitigates).
    pub api_secret: Option<String>,
    /// Local embedding engine (MiniLM-L6-v2 via ONNX Runtime).
    /// When None, `/api/mpi/embed` returns 503 and Miner falls back to OpenClaw Gateway.
    pub embed_engine: Option<Arc<EmbedEngine>>,
    /// v2.3.0: Whether this node accepts remote storage requests (Ed25519 signature auth).
    /// When false, only local Bearer token auth is accepted.
    pub allow_remote_storage: bool,
    /// v2.3.0: Maximum number of distinct remote owners this node serves.
    /// 0 = unlimited. Checked during auth middleware for remote requests.
    pub max_remote_owners: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub positive_rate: f32,
    pub sample_size: usize,
    pub frozen_at: i64,
}

// ============================================
// Unified Auth Middleware (v2.3.0)
// ============================================

/// Unified authentication middleware for MPI endpoints.
///
/// Handles two authentication methods in priority order:
///
/// 1. **Ed25519 Signature** (remote users):
///    Headers: X-MemChain-PublicKey, X-MemChain-Timestamp, X-MemChain-Signature
///    - Verifies Ed25519 signature over SHA256(timestamp || method || path || SHA256(body))
///    - Checks timestamp within ±300s tolerance (replay protection)
///    - Checks allow_remote_storage config
///    - Checks max_remote_owners capacity
///    - Sets AuthenticatedOwner::Remote with signer's public key as owner
///
/// 2. **Bearer Token** (local node operator):
///    Header: Authorization: Bearer <token>
///    - Constant-time comparison against configured api_secret
///    - Sets AuthenticatedOwner::Local with state.owner_key
///
/// 3. **No Auth** (backward compatible):
///    - If no api_secret configured and no signature headers → pass through
///    - Sets AuthenticatedOwner::Local with state.owner_key
///
/// ## Request Extension
/// On success, inserts `AuthenticatedOwner` into request extensions.
/// All handlers MUST extract this extension to determine the owner.
///
/// ## Security Notes
/// - Ed25519 signature check is attempted first (if headers present)
/// - Bearer token uses constant-time comparison (timing attack prevention)
/// - Body is consumed and re-inserted for signature verification
/// - /log endpoint has additional local-only restriction in its handler
async fn unified_auth_middleware(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    // Check if this is an Ed25519 signed request (remote user)
    let has_signature = req.headers().contains_key("x-memchain-signature");

    if has_signature {
        return handle_remote_auth(state, req, next).await;
    }

    // Fall through to Bearer token auth (local) or open access
    handle_local_auth(state, req, next).await
}

/// Handle Ed25519 signature authentication for remote users.
///
/// ## Signature Format
/// The client signs: `SHA256(timestamp_str || method || path || SHA256(body))`
///
/// Where:
/// - `timestamp_str`: decimal string of Unix timestamp (same as X-MemChain-Timestamp header)
/// - `method`: HTTP method string (e.g., "POST", "GET")
/// - `path`: Request URI path (e.g., "/api/mpi/remember")
/// - `body`: Raw request body bytes (empty for GET requests)
///
/// ## Verification Steps
/// 1. Extract and validate all 3 headers (public key, timestamp, signature)
/// 2. Parse public key (64 hex chars → 32 bytes → IdentityPublicKey)
/// 3. Check timestamp within tolerance (replay protection)
/// 4. Check allow_remote_storage config
/// 5. Reconstruct signed message: SHA256(timestamp || method || path || SHA256(body))
/// 6. Verify Ed25519 signature
/// 7. Check max_remote_owners capacity
/// 8. Insert AuthenticatedOwner::Remote into request extensions
async fn handle_remote_auth(
    state: Arc<MpiState>,
    req: Request<axum::body::Body>,
    next: Next,
) -> axum::response::Response {
    // --- Step 1: Extract headers ---
    let pubkey_hex = match req.headers().get("x-memchain-publickey")
        .and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => {
            debug!("[MPI_AUTH] Missing X-MemChain-PublicKey header");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "missing X-MemChain-PublicKey header"})),
            ).into_response();
        }
    };

    let timestamp_str = match req.headers().get("x-memchain-timestamp")
        .and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => {
            debug!("[MPI_AUTH] Missing X-MemChain-Timestamp header");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "missing X-MemChain-Timestamp header"})),
            ).into_response();
        }
    };

    let signature_hex = match req.headers().get("x-memchain-signature")
        .and_then(|v| v.to_str().ok())
    {
        Some(v) => v.to_string(),
        None => {
            debug!("[MPI_AUTH] Missing X-MemChain-Signature header");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "missing X-MemChain-Signature header"})),
            ).into_response();
        }
    };

    // --- Step 2: Parse public key ---
    let pubkey_bytes = match hex::decode(&pubkey_hex) {
        Ok(b) if b.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            arr
        }
        _ => {
            debug!("[MPI_AUTH] Invalid public key format: {}", pubkey_hex);
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "invalid X-MemChain-PublicKey: expected 64 hex chars"})),
            ).into_response();
        }
    };

    let identity_pubkey: IdentityPublicKey = match IdentityPublicKey::from_bytes(&pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => {
            debug!("[MPI_AUTH] Invalid Ed25519 public key");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "invalid Ed25519 public key"})),
            ).into_response();
        }
    };

    // --- Step 3: Check timestamp tolerance (replay protection) ---
    let timestamp: u64 = match timestamp_str.parse() {
        Ok(ts) => ts,
        Err(_) => {
            debug!("[MPI_AUTH] Invalid timestamp: {}", timestamp_str);
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "invalid X-MemChain-Timestamp"})),
            ).into_response();
        }
    };

    let now = now_secs();
    let drift = if now > timestamp { now - timestamp } else { timestamp - now };
    if drift > AUTH_TIMESTAMP_TOLERANCE_SECS {
        debug!(
            "[MPI_AUTH] Timestamp drift too large: {}s (tolerance: {}s)",
            drift, AUTH_TIMESTAMP_TOLERANCE_SECS
        );
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "timestamp expired or clock drift too large",
                "server_time": now,
                "request_time": timestamp,
                "tolerance_secs": AUTH_TIMESTAMP_TOLERANCE_SECS,
            })),
        ).into_response();
    }

    // --- Step 4: Check allow_remote_storage ---
    if !state.allow_remote_storage {
        debug!("[MPI_AUTH] Remote storage not enabled, rejecting signed request");
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "this node does not accept remote storage requests"})),
        ).into_response();
    }

    // --- Step 5: Parse signature ---
    let signature_bytes = match hex::decode(&signature_hex) {
        Ok(b) if b.len() == 64 => {
            let mut arr = [0u8; 64];
            arr.copy_from_slice(&b);
            arr
        }
        _ => {
            debug!("[MPI_AUTH] Invalid signature format");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "invalid X-MemChain-Signature: expected 128 hex chars"})),
            ).into_response();
        }
    };

    // --- Step 6: Reconstruct signed message and verify ---
    // Collect method and path before consuming body
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();

    // Consume body for signature verification
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            warn!("[MPI_AUTH] Failed to read request body for signature verification");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "failed to read request body"})),
            ).into_response();
        }
    };

    // Compute: SHA256(timestamp || method || path || SHA256(body))
    let body_hash = Sha256::digest(&body_bytes);
    let mut message_hasher = Sha256::new();
    message_hasher.update(timestamp_str.as_bytes());
    message_hasher.update(method.as_bytes());
    message_hasher.update(path.as_bytes());
    message_hasher.update(&body_hash);
    let signed_message = message_hasher.finalize();

    // Verify Ed25519 signature
    if let Err(_) = identity_pubkey.verify(&signed_message, &signature_bytes) {
        warn!("[MPI_AUTH] Ed25519 signature verification failed for {}", pubkey_hex);
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": "signature verification failed"})),
        ).into_response();
    }

    // --- Step 7: Check max_remote_owners capacity ---
    // Only check if max_remote_owners > 0 (0 = unlimited)
    if state.max_remote_owners > 0 {
        let current_owners = state.storage.count_distinct_owners().await;
        // Subtract 1 for the local owner (node operator) which always exists
        let remote_count = current_owners.saturating_sub(1);
        // Allow if this owner already exists (not a new owner)
        let owner_exists = state.storage.owner_exists(&pubkey_bytes).await;
        if !owner_exists && remote_count >= state.max_remote_owners {
            warn!(
                "[MPI_AUTH] Remote owner capacity reached: {}/{} (rejecting {})",
                remote_count, state.max_remote_owners, pubkey_hex
            );
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "this node has reached maximum remote user capacity",
                    "max_remote_owners": state.max_remote_owners,
                })),
            ).into_response();
        }
    }

    debug!(
        "[MPI_AUTH] Remote auth OK: {} (method={}, path={})",
        &pubkey_hex[..8], method, path
    );

    // --- Step 8: Reconstruct request with AuthenticatedOwner extension ---
    let auth = AuthenticatedOwner::Remote {
        owner: pubkey_bytes,
        owner_hex: pubkey_hex,
    };

    let mut req = Request::from_parts(parts, axum::body::Body::from(body_bytes));
    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

/// Handle Bearer token authentication for local node operator.
///
/// If no api_secret is configured, passes through as open access (backward compatible).
/// In both cases, sets AuthenticatedOwner::Local with state.owner_key.
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
        // Bearer token required
        let auth_header = req.headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());

        let token = match auth_header {
            Some(h) if h.starts_with("Bearer ") => &h[7..],
            Some(h) if h.starts_with("bearer ") => &h[7..],
            _ => {
                debug!("[MPI_AUTH] Missing or malformed Authorization header");
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({"error": "missing Authorization: Bearer <token>"})),
                ).into_response();
            }
        };

        // Constant-time comparison to prevent timing attacks.
        let token_bytes = token.as_bytes();
        let expected_bytes = expected.as_bytes();
        let valid = if token_bytes.len() == expected_bytes.len() {
            token_bytes.iter()
                .zip(expected_bytes.iter())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b)) == 0
        } else {
            false
        };

        if !valid {
            warn!("[MPI_AUTH] Invalid Bearer token from request");
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "invalid token"})),
            ).into_response();
        }
    }

    // Bearer token valid (or no auth required) → local owner
    let auth = AuthenticatedOwner::Local {
        owner: state.owner_key,
    };
    let mut req = req;
    req.extensions_mut().insert(auth);
    next.run(req).await.into_response()
}

// ============================================
// Helpers
// ============================================

/// Extract AuthenticatedOwner from request extensions.
///
/// Panics if the auth middleware was not applied (programming error).
/// All handlers behind the unified_auth_middleware can safely call this.
fn extract_owner<B>(req: &Request<B>) -> &AuthenticatedOwner {
    req.extensions()
        .get::<AuthenticatedOwner>()
        .expect("[BUG] AuthenticatedOwner not set — unified_auth_middleware must be applied")
}

fn parse_layer(s: &str) -> Option<MemoryLayer> {
    match s.to_lowercase().as_str() {
        "identity"  => Some(MemoryLayer::Identity),
        "knowledge" => Some(MemoryLayer::Knowledge),
        "episode"   => Some(MemoryLayer::Episode),
        "archive"   => Some(MemoryLayer::Archive),
        _ => None,
    }
}

fn estimate_tokens(text: &str) -> usize {
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

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ============================================
// POST /api/mpi/remember
// ============================================

#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub content: String,
    #[serde(default = "default_layer")]
    pub layer: String,
    #[serde(default)]
    pub topic_tags: Vec<String>,
    #[serde(default = "default_source")]
    pub source_ai: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default = "default_model")]
    pub embedding_model: String,
}

fn default_layer() -> String { "episode".into() }
fn default_source() -> String { "unknown".into() }
fn default_model() -> String { "default".into() }

#[derive(Debug, Serialize)]
pub struct RememberResponse {
    pub record_id: String,
    pub status: String,
    pub duplicate_of: Option<String>,
}

/// `POST /api/mpi/remember` — Store a new memory record.
///
/// ## v2.3.0 Changes
/// - Owner is extracted from AuthenticatedOwner extension (not state.owner_key)
/// - For remote users, content is expected to be client-encrypted (opaque bytes)
/// - Embeddings are stored in plaintext regardless (needed for vector search)
pub async fn mpi_remember(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();
    let owner_hex = auth.owner_hex();

    // Parse body
    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))),
    };
    let req_body: RememberRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))),
    };

    if req_body.content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"content empty"})));
    }
    let layer = match parse_layer(&req_body.layer) {
        Some(l) => l,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid layer"}))),
    };

    let ts = now_secs();

    // Dedup check
    if !req_body.embedding.is_empty() {
        let dedup = state.vector_index.check_duplicate(
            &req_body.embedding, &owner, &req_body.embedding_model, layer, ts,
        );
        if dedup.is_duplicate {
            let dup_hex = hex::encode(dedup.existing_id.unwrap_or([0; 32]));
            return (StatusCode::OK, Json(serde_json::json!(RememberResponse {
                record_id: dup_hex.clone(), status: "duplicate".into(), duplicate_of: Some(dup_hex),
            })));
        }
    }

    let encrypted_content = req_body.content.as_bytes().to_vec();
    let mut record = MemoryRecord::new(
        owner, ts, layer, req_body.topic_tags, req_body.source_ai,
        encrypted_content, req_body.embedding.clone(),
    );
    record.signature = state.identity.sign(&record.record_id);
    let rid_hex = record.id_hex();

    // SQLite first (source of truth)
    if !state.storage.insert(&record, &req_body.embedding_model).await {
        return (StatusCode::CONFLICT, Json(serde_json::json!({"error":"exists","record_id":rid_hex})));
    }

    // Vector index (after SQLite success)
    if !req_body.embedding.is_empty() {
        state.vector_index.upsert(
            record.record_id, req_body.embedding, layer, ts, &owner, &req_body.embedding_model,
        );
    }

    // Identity cache (after SQLite success)
    if layer == MemoryLayer::Identity {
        let mut cache = state.identity_cache.write();
        cache.entry(owner_hex.clone()).or_default().push(record.clone());
    }

    info!(id = %rid_hex, layer = %layer, owner = %&owner_hex[..8], "[MPI_REMEMBER] Stored");
    (StatusCode::CREATED, Json(serde_json::json!(RememberResponse {
        record_id: rid_hex, status: "created".into(), duplicate_of: None,
    })))
}

// ============================================
// POST /api/mpi/recall
// ============================================

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default = "default_model")]
    pub embedding_model: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub layer: Option<String>,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
    pub time_hint: Option<TimeHint>,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimeHint { pub start: i64, pub end: i64 }

fn default_top_k() -> usize { 10 }
fn default_token_budget() -> usize { 4000 }

#[derive(Debug, Serialize)]
pub struct RecalledMemory {
    pub record_id: String,
    pub layer: String,
    pub score: f64,
    pub content: String,
    pub topic_tags: Vec<String>,
    pub source_ai: String,
    pub timestamp: u64,
    pub access_count: u32,
    #[serde(default)]
    pub proactive: bool,
}

#[derive(Debug, Serialize)]
pub struct RecallResponse {
    pub memories: Vec<RecalledMemory>,
    pub total_candidates: usize,
    pub token_estimate: usize,
}

/// `POST /api/mpi/recall` — Semantic search + MVF scoring.
///
/// ## v2.3.0 Changes
/// - Owner extracted from AuthenticatedOwner (supports multi-owner isolation)
/// - Identity cache and vector search scoped to authenticated owner
pub async fn mpi_recall(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();
    let owner_hex = auth.owner_hex();

    // Parse body
    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let req_body: RecallRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };

    let now = now_secs();
    let layer_filter = req_body.layer.as_deref().and_then(parse_layer);
    let top_k = req_body.top_k.min(100).max(1);

    // Session centroid for phi_7
    let session_centroid: Option<Vec<f32>> = if let Some(ref sid) = req_body.session_id {
        if !req_body.embedding.is_empty() {
            let mut map = state.session_embeddings.write();
            let buf = map.entry(sid.clone()).or_insert_with(|| VecDeque::with_capacity(5));
            if buf.len() >= 5 { buf.pop_front(); }
            buf.push_back(req_body.embedding.clone());
            let dim = buf[0].len();
            let mut centroid = vec![0.0f32; dim];
            for emb in buf.iter() {
                for (i, &v) in emb.iter().enumerate() { if i < dim { centroid[i] += v; } }
            }
            let n = buf.len() as f32;
            for v in centroid.iter_mut() { *v /= n; }
            Some(centroid)
        } else { None }
    } else { None };

    let mut memories: Vec<RecalledMemory> = Vec::new();
    let mut total_tokens = 0usize;
    let mut seen_ids: Vec<[u8; 32]> = Vec::new();

    // Step 1: Identity forced injection (scoped to authenticated owner)
    {
        let cache = state.identity_cache.read();
        let id_recs = cache.get(&owner_hex).cloned().unwrap_or_default();
        drop(cache);

        for r in &id_recs {
            if !r.is_active() { continue; }
            let content = String::from_utf8_lossy(&r.encrypted_content).to_string();
            let tokens = estimate_tokens(&content);
            if total_tokens + tokens > req_body.token_budget && !memories.is_empty() { break; }
            total_tokens += tokens;
            seen_ids.push(r.record_id);
            memories.push(RecalledMemory {
                record_id: r.id_hex(), layer: r.layer.to_string(),
                score: r.layer.recall_weight() + 1.0, content,
                topic_tags: r.topic_tags.clone(), source_ai: r.source_ai.clone(),
                timestamp: r.timestamp, access_count: r.access_count, proactive: false,
            });
            let st = Arc::clone(&state.storage);
            let rid = r.record_id;
            tokio::spawn(async move { st.increment_access(&rid).await; });
        }
    }

    // Step 2: Vector search (scoped to authenticated owner)
    let idx_ready = state.index_ready.load(std::sync::atomic::Ordering::Relaxed);
    let search = if !req_body.embedding.is_empty() && idx_ready {
        state.vector_index.search_filtered(
            &req_body.embedding, &owner, &req_body.embedding_model, layer_filter, top_k * 3, 0.0,
        )
    } else { Vec::new() };

    let total_candidates = search.len() + seen_ids.len();

    // Graph max degree
    let max_degree = {
        let conn = state.storage.conn_lock().await;
        graph::get_max_degree(&conn, &owner)
    };

    let time_hint_tuple = req_body.time_hint.as_ref().map(|th| (th.start, th.end));

    // Step 3+4: Load + score
    let mut scored: Vec<(MemoryRecord, f64)> = Vec::new();

    for sr in &search {
        if seen_ids.contains(&sr.record_id) { continue; }
        if let Some(record) = state.storage.get(&sr.record_id).await {
            if !record.is_active() { continue; }
            // v2.3.0: Verify record belongs to authenticated owner
            if record.owner != owner { continue; }

            let v_old = compute_recall_score(
                sr.similarity, record.timestamp, now, record.access_count, record.layer,
            );
            let time_bonus: f64 = match &req_body.time_hint {
                Some(th) if (record.timestamp as i64) >= th.start && (record.timestamp as i64) <= th.end => 0.15,
                _ => 0.0,
            };
            let v_old_h = v_old + time_bonus;

            let final_score = if state.mvf_enabled {
                let gd = { let c = state.storage.conn_lock().await; graph::get_degree(&c, &record.record_id) };
                let cs = session_centroid.as_ref()
                    .filter(|c| record.has_embedding() && c.len() == record.embedding.len())
                    .map(|c| crate::services::memchain::cosine_similarity(c, &record.embedding))
                    .unwrap_or(0.0);

                let phi = mvf::compute_features(
                    sr.similarity,
                    record.layer as u8,
                    record.timestamp,
                    now,
                    record.access_count,
                    record.positive_feedback,
                    record.negative_feedback,
                    record.has_conflict(),
                    time_hint_tuple,
                    cs,
                    gd,
                    max_degree,
                );
                let mut uw = { state.user_weights.read().get(&owner_hex).cloned().unwrap_or_else(mvf::default_weights) };
                let pn = mvf::normalize(&phi, &mut uw);
                let vm = mvf::compute_value(&uw, &pn);
                { state.user_weights.write().insert(owner_hex.clone(), uw); }
                mvf::fuse_scores(vm, v_old_h, state.mvf_alpha)
            } else { v_old_h };

            scored.push((record, final_score));
        }
    }

    // Fallback: no search results
    if search.is_empty() {
        let recent = state.storage.get_active_records(&owner, layer_filter, top_k).await;
        for r in recent {
            if seen_ids.contains(&r.record_id) { continue; }
            let s = compute_recall_score(1.0, r.timestamp, now, r.access_count, r.layer);
            scored.push((r, s));
        }
    }

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step 5: Token budget
    let mut returned_ids = seen_ids.clone();
    for (r, score) in &scored {
        let content = String::from_utf8_lossy(&r.encrypted_content).to_string();
        let tokens = estimate_tokens(&content);
        if total_tokens + tokens > req_body.token_budget && !memories.is_empty() { break; }
        let proactive = r.layer == MemoryLayer::Identity
            && search.iter().find(|sr| sr.record_id == r.record_id).map_or(false, |sr| sr.similarity > 0.3);
        total_tokens += tokens;
        returned_ids.push(r.record_id);
        memories.push(RecalledMemory {
            record_id: r.id_hex(), layer: r.layer.to_string(), score: *score, content,
            topic_tags: r.topic_tags.clone(), source_ai: r.source_ai.clone(),
            timestamp: r.timestamp, access_count: r.access_count, proactive,
        });
        let st = Arc::clone(&state.storage);
        let rid = r.record_id;
        tokio::spawn(async move { st.increment_access(&rid).await; });
    }
    memories.truncate(top_k);

    // Async graph update
    {
        let st = Arc::clone(&state.storage);
        let ids = returned_ids;
        tokio::spawn(async move {
            let c = st.conn_lock().await;
            let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
            graph::update_cooccurrence(&c, &ids, n);
        });
    }

    (StatusCode::OK, Json(serde_json::json!(RecallResponse {
        memories, total_candidates, token_estimate: total_tokens,
    }))).into_response()
}

// ============================================
// POST /api/mpi/forget
// ============================================

#[derive(Debug, Deserialize)]
pub struct ForgetRequest { pub record_id: String }

#[derive(Debug, Serialize)]
pub struct ForgetResponse { pub status: String, pub record_id: String }

/// `POST /api/mpi/forget` — Delete (revoke) a memory record.
///
/// ## v2.3.0 Changes (Security Fix)
/// - Now verifies record.owner == authenticated owner before revoking
/// - Prevents remote users from deleting other users' records
/// - Prevents remote users from deleting node operator's records
pub async fn mpi_forget(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    // Parse body
    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))),
    };
    let req_body: ForgetRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))),
    };

    let rid = match hex::decode(&req_body.record_id) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8;32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"bad record_id"}))),
    };

    // v2.3.0: Verify ownership before revoking
    if let Some(record) = state.storage.get(&rid).await {
        if record.owner != owner {
            warn!(
                "[MPI_FORGET] Owner mismatch: record owner={}, request owner={}",
                hex::encode(record.owner), auth.owner_hex()
            );
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({
                "error": "access denied: record belongs to another owner"
            })));
        }
    } else {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: req_body.record_id })));
    }

    if !state.storage.revoke(&rid).await {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: req_body.record_id })));
    }

    state.vector_index.remove(&rid);
    {
        let oh = auth.owner_hex();
        let mut c = state.identity_cache.write();
        if let Some(e) = c.get_mut(&oh) { e.retain(|r| r.record_id != rid); }
    }

    (StatusCode::OK, Json(serde_json::json!(ForgetResponse {
        status:"revoked".into(), record_id: req_body.record_id })))
}

// ============================================
// GET /api/mpi/status
// ============================================

#[derive(Debug, Serialize)]
pub struct MpiStatusResponse {
    pub memchain_enabled: bool,
    pub mode: String,
    pub stats: crate::services::memchain::StorageStats,
    pub vector_index_total: usize,
    pub vector_partitions: usize,
    pub last_block_height: u64,
    pub index_ready: bool,
    pub embed_ready: bool,
    pub embed_dim: Option<usize>,
    pub mvf: MvfMetrics,
    /// v2.3.0: Whether remote storage is enabled on this node
    pub remote_storage_enabled: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MvfMetrics {
    pub enabled: bool,
    pub alpha: f32,
    pub total_positive_feedback: u64,
    pub total_negative_feedback: u64,
    pub baseline_positive_rate: Option<f32>,
    pub baseline_sample_size: Option<usize>,
    pub mvf_positive_rate: Option<f32>,
    pub mvf_sample_size: Option<usize>,
    pub lift: Option<f32>,
    pub weights_version: u64,
}

/// `GET /api/mpi/status` — System status endpoint.
///
/// ## v2.3.0 Changes
/// - Added `remote_storage_enabled` field
/// - Feedback stats scoped to authenticated owner
pub async fn mpi_status(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner_hex = auth.owner_hex();

    let stats = state.storage.stats().await;
    let height = state.storage.last_block_height().await;
    let wv = { state.user_weights.read().get(&owner_hex).map(|w| w.version).unwrap_or(0) };
    let fb = state.storage.get_recent_feedback(500).await;
    let total_pos = fb.iter().filter(|(s,_)| *s == 1).count() as u64;
    let total_neg = fb.iter().filter(|(s,_)| *s == -1).count() as u64;
    let mvf_rate = if !fb.is_empty() { Some(total_pos as f32 / fb.len() as f32) } else { None };
    let mvf_sample = if !fb.is_empty() { Some(fb.len()) } else { None };
    let baseline = state.mvf_baseline.read().clone();
    let lift = match (&baseline, mvf_rate) {
        (Some(b), Some(m)) if b.positive_rate > 0.0 => Some((m - b.positive_rate) / b.positive_rate),
        _ => None,
    };

    (StatusCode::OK, Json(serde_json::json!(MpiStatusResponse {
        memchain_enabled: true, mode: "local".into(), stats,
        vector_index_total: state.vector_index.total_vectors(),
        vector_partitions: state.vector_index.partition_count(),
        last_block_height: height,
        index_ready: state.index_ready.load(std::sync::atomic::Ordering::Relaxed),
        embed_ready: state.embed_engine.is_some(),
        embed_dim: state.embed_engine.as_ref().map(|e| e.dim()),
        mvf: MvfMetrics {
            enabled: state.mvf_enabled, alpha: state.mvf_alpha,
            total_positive_feedback: total_pos, total_negative_feedback: total_neg,
            baseline_positive_rate: baseline.as_ref().map(|b| b.positive_rate),
            baseline_sample_size: baseline.as_ref().map(|b| b.sample_size),
            mvf_positive_rate: mvf_rate, mvf_sample_size: mvf_sample,
            lift, weights_version: wv,
        },
        remote_storage_enabled: state.allow_remote_storage,
    })))
}

// ============================================
// GET /api/mpi/record/:record_id (v2.2.0)
// ============================================

#[derive(Debug, Serialize)]
pub struct RecordDetailResponse {
    pub record_id: String,
    pub layer: String,
    pub content: String,
    pub topic_tags: Vec<String>,
    pub source_ai: String,
    pub timestamp: u64,
    pub access_count: u32,
    pub positive_feedback: u32,
    pub negative_feedback: u32,
    pub has_conflict: bool,
    pub embedding_model: String,
    pub has_embedding: bool,
    pub status: String,
}

/// `GET /api/mpi/record/:record_id` — Get single record details.
///
/// Used by the MemExplorer frontend for record detail/edit views.
///
/// ## v2.3.0 Changes (Security Fix)
/// - Now verifies record.owner == authenticated owner before returning data
/// - Prevents remote users from reading other users' records
/// - Prevents remote users from reading node operator's records
pub async fn mpi_get_record(
    State(state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let rid = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid record_id format"}))).into_response(),
    };

    let record = match state.storage.get(&rid).await {
        Some(r) => r,
        None => return (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "record not found"}))).into_response(),
    };

    // v2.3.0: Verify ownership
    if record.owner != owner {
        return (StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": "access denied: record belongs to another owner"}))).into_response();
    }

    let embedding_model = state.storage.get_embedding_model(&rid).await
        .unwrap_or_else(|| String::new());

    let content = String::from_utf8_lossy(&record.encrypted_content).to_string();

    (StatusCode::OK, Json(serde_json::json!(RecordDetailResponse {
        record_id: record_id_hex,
        layer: record.layer.to_string(),
        content,
        topic_tags: record.topic_tags.clone(),
        source_ai: record.source_ai.clone(),
        timestamp: record.timestamp,
        access_count: record.access_count,
        positive_feedback: record.positive_feedback,
        negative_feedback: record.negative_feedback,
        has_conflict: record.has_conflict(),
        embedding_model,
        has_embedding: record.has_embedding(),
        status: if record.is_active() { "active" } else { "revoked" }.to_string(),
    }))).into_response()
}

// ============================================
// GET /api/mpi/records/overview (v2.2.0)
// ============================================

#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    pub total: u64,
    pub by_layer: std::collections::HashMap<String, u64>,
    pub recent_by_layer: std::collections::HashMap<String, Vec<crate::services::memchain::OverviewRecord>>,
    pub last_memory_at: u64,
    pub embed_ready: bool,
    pub embed_dim: Option<usize>,
}

/// `GET /api/mpi/records/overview` — Memory overview for MemExplorer frontend.
///
/// ## v2.3.0 Changes
/// - Owner extracted from AuthenticatedOwner (scoped to authenticated user's records)
pub async fn mpi_records_overview(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let overview = state.storage.get_overview(&owner, 20).await;
    let total: u64 = overview.by_layer.values().sum();

    (StatusCode::OK, Json(serde_json::json!(OverviewResponse {
        total,
        by_layer: overview.by_layer,
        recent_by_layer: overview.recent_by_layer,
        last_memory_at: overview.last_memory_at,
        embed_ready: state.embed_engine.is_some(),
        embed_dim: state.embed_engine.as_ref().map(|e| e.dim()),
    })))
}

// ============================================
// POST /api/mpi/embed
// ============================================

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
    #[serde(default = "default_model")]
    pub model: String,
}

#[derive(Debug, Serialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub dim: usize,
}

/// `POST /api/mpi/embed` — Generate embeddings locally using MiniLM-L6-v2.
///
/// This endpoint does not store any data, so owner isolation is not relevant.
/// Auth is still required (part of the global middleware).
pub async fn mpi_embed(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    // embed doesn't need owner, but auth middleware is still applied
    let _auth = extract_owner(&req);

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))),
    };
    let req_body: EmbedRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))),
    };

    let engine = match &state.embed_engine {
        Some(e) => e,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "local embed engine not available",
                    "hint": "ensure model files exist at configured embed_model_path"
                })),
            );
        }
    };

    if req_body.texts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "texts array is empty"})),
        );
    }

    if req_body.texts.len() > 100 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "batch too large",
                "max": 100,
                "got": req_body.texts.len()
            })),
        );
    }

    let text_refs: Vec<&str> = req_body.texts.iter().map(|s| s.as_str()).collect();

    match engine.embed_batch(&text_refs) {
        Ok(embeddings) => {
            let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
            debug!(
                batch = embeddings.len(),
                dim = dim,
                "[MPI_EMBED] Generated embeddings"
            );
            (
                StatusCode::OK,
                Json(serde_json::json!(EmbedResponse {
                    embeddings,
                    model: "minilm-l6-v2".into(),
                    dim,
                })),
            )
        }
        Err(e) => {
            warn!(error = %e, "[MPI_EMBED] Inference failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("embed inference failed: {}", e)})),
            )
        }
    }
}

// ============================================
// Router
// ============================================

/// Build the MPI router with unified auth middleware.
///
/// ## Auth Behavior (v2.3.0)
/// The unified auth middleware is ALWAYS applied to ALL endpoints.
/// It handles three scenarios:
/// 1. Ed25519 signature headers → remote user auth → AuthenticatedOwner::Remote
/// 2. Bearer token → local node operator auth → AuthenticatedOwner::Local
/// 3. No auth configured + no signature → open access → AuthenticatedOwner::Local
///
/// ## /log Endpoint Restriction
/// The /log endpoint only accepts local (Bearer token) auth.
/// Remote users cannot submit logs (rule engine runs locally only).
/// This restriction is enforced in the log_handler, not in the middleware,
/// to keep the middleware logic clean and simple.
pub fn build_mpi_router(state: Arc<MpiState>) -> Router {
    let router = Router::new()
        .route("/api/mpi/remember", post(mpi_remember))
        .route("/api/mpi/recall", post(mpi_recall))
        .route("/api/mpi/forget", post(mpi_forget))
        .route("/api/mpi/status", get(mpi_status))
        .route("/api/mpi/embed", post(mpi_embed))
        .route("/api/mpi/log", post(crate::api::log_handler::mpi_log))
        // v2.2.0 MemExplorer
        .route("/api/mpi/record/:record_id", get(mpi_get_record))
        .route("/api/mpi/records/overview", get(mpi_records_overview));

    info!(
        "[MPI] Unified auth middleware enabled (bearer={}, remote_storage={})",
        state.api_secret.as_ref().map_or(false, |s| !s.is_empty()),
        state.allow_remote_storage,
    );

    router
        .route_layer(middleware::from_fn_with_state(state.clone(), unified_auth_middleware))
        .with_state(state)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn make_state() -> Arc<MpiState> {
        make_state_with_options(None, false, 0).await
    }

    async fn make_state_with_secret(secret: Option<String>) -> Arc<MpiState> {
        make_state_with_options(secret, false, 0).await
    }

    async fn make_state_with_options(
        secret: Option<String>,
        allow_remote: bool,
        max_remote: usize,
    ) -> Arc<MpiState> {
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
            owner_key: ok,
            api_secret: secret,
            embed_engine: None,
            allow_remote_storage: allow_remote,
            max_remote_owners: max_remote,
        })
    }

    /// Helper: create Ed25519 signed request headers for remote auth testing.
    fn sign_request(
        identity: &IdentityKeyPair,
        method: &str,
        path: &str,
        body: &[u8],
        timestamp: u64,
    ) -> Vec<(&'static str, String)> {
        let body_hash = Sha256::digest(body);
        let mut msg_hasher = Sha256::new();
        msg_hasher.update(timestamp.to_string().as_bytes());
        msg_hasher.update(method.as_bytes());
        msg_hasher.update(path.as_bytes());
        msg_hasher.update(&body_hash);
        let signed_msg = msg_hasher.finalize();
        let signature = identity.sign(&signed_msg);

        vec![
            ("x-memchain-publickey", hex::encode(identity.public_key_bytes())),
            ("x-memchain-timestamp", timestamp.to_string()),
            ("x-memchain-signature", hex::encode(signature)),
        ]
    }

    #[tokio::test]
    async fn test_remember_and_recall() {
        let s = make_state().await;
        let app = build_mpi_router(s);

        let req = Request::builder().method("POST").uri("/api/mpi/remember")
            .header("content-type","application/json")
            .body(Body::from(r#"{"content":"User likes Rust","layer":"identity","embedding":[0.1,0.2,0.3],"embedding_model":"t"}"#)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let req = Request::builder().method("POST").uri("/api/mpi/recall")
            .header("content-type","application/json")
            .body(Body::from(r#"{"embedding":[0.1,0.2,0.3],"embedding_model":"t","top_k":5}"#)).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/status").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // ========================================
    // Auth Middleware Tests (v2.1.0+MVF+Auth)
    // ========================================

    #[tokio::test]
    async fn test_auth_no_secret_passes() {
        let s = make_state_with_secret(None).await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/status").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_empty_secret_passes() {
        let s = make_state_with_secret(Some(String::new())).await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/status").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_missing_header_rejects() {
        let s = make_state_with_secret(Some("test-secret-at-least-16".into())).await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/status").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_wrong_token_rejects() {
        let s = make_state_with_secret(Some("test-secret-at-least-16".into())).await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/status")
            .header("Authorization", "Bearer wrong-token-value")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_correct_token_passes() {
        let secret = "test-secret-at-least-16".to_string();
        let s = make_state_with_secret(Some(secret.clone())).await;
        let app = build_mpi_router(s);
        let req = Request::builder()
            .uri("/api/mpi/status")
            .header("Authorization", format!("Bearer {}", secret))
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_protects_remember_endpoint() {
        let secret = "my-super-secret-key-1234".to_string();
        let s = make_state_with_secret(Some(secret.clone())).await;
        let app = build_mpi_router(s);

        // Without auth → 401
        let req = Request::builder().method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"content":"test","layer":"episode"}"#)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // With auth → 201
        let req = Request::builder().method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .header("Authorization", format!("Bearer {}", secret))
            .body(Body::from(r#"{"content":"test","layer":"episode"}"#)).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    // ========================================
    // Embed Endpoint Tests (v2.1.0+Embed)
    // ========================================

    #[tokio::test]
    async fn test_embed_returns_503_when_no_engine() {
        let s = make_state().await;
        let app = build_mpi_router(s);

        let req = Request::builder().method("POST").uri("/api/mpi/embed")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"texts":["hello world"]}"#)).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_embed_rejects_empty_texts() {
        let s = make_state().await;
        let app = build_mpi_router(s);

        let req = Request::builder().method("POST").uri("/api/mpi/embed")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"texts":[]}"#)).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // 503 because engine is None (checked before empty validation)
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    // ========================================
    // v2.3.0: Remote Auth Tests
    // ========================================

    #[tokio::test]
    async fn test_remote_auth_rejected_when_disabled() {
        // allow_remote_storage = false (default)
        let s = make_state().await;
        let app = build_mpi_router(s);

        let remote_user = IdentityKeyPair::generate();
        let ts = now_secs();
        let headers = sign_request(&remote_user, "GET", "/api/mpi/status", &[], ts);

        let mut builder = Request::builder().uri("/api/mpi/status");
        for (k, v) in &headers {
            builder = builder.header(*k, v.as_str());
        }
        let req = builder.body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_remote_auth_accepted_when_enabled() {
        let s = make_state_with_options(None, true, 0).await;
        let app = build_mpi_router(s);

        let remote_user = IdentityKeyPair::generate();
        let ts = now_secs();
        let headers = sign_request(&remote_user, "GET", "/api/mpi/status", &[], ts);

        let mut builder = Request::builder().uri("/api/mpi/status");
        for (k, v) in &headers {
            builder = builder.header(*k, v.as_str());
        }
        let req = builder.body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_remote_auth_expired_timestamp_rejected() {
        let s = make_state_with_options(None, true, 0).await;
        let app = build_mpi_router(s);

        let remote_user = IdentityKeyPair::generate();
        // Timestamp 10 minutes ago (beyond 5 minute tolerance)
        let ts = now_secs() - 600;
        let headers = sign_request(&remote_user, "GET", "/api/mpi/status", &[], ts);

        let mut builder = Request::builder().uri("/api/mpi/status");
        for (k, v) in &headers {
            builder = builder.header(*k, v.as_str());
        }
        let req = builder.body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_remote_auth_wrong_signature_rejected() {
        let s = make_state_with_options(None, true, 0).await;
        let app = build_mpi_router(s);

        let remote_user = IdentityKeyPair::generate();
        let ts = now_secs();

        // Sign with correct key but tamper the signature
        let mut headers = sign_request(&remote_user, "GET", "/api/mpi/status", &[], ts);
        // Replace signature with garbage
        headers[2].1 = "00".repeat(64);

        let mut builder = Request::builder().uri("/api/mpi/status");
        for (k, v) in &headers {
            builder = builder.header(*k, v.as_str());
        }
        let req = builder.body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_remote_remember_owner_isolation() {
        let s = make_state_with_options(None, true, 0).await;
        let app = build_mpi_router(s);

        // Remote user stores a record
        let remote_user = IdentityKeyPair::generate();
        let body = r#"{"content":"remote secret","layer":"episode"}"#;
        let ts = now_secs();
        let headers = sign_request(&remote_user, "POST", "/api/mpi/remember", body.as_bytes(), ts);

        let mut builder = Request::builder().method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json");
        for (k, v) in &headers {
            builder = builder.header(*k, v.as_str());
        }
        let req = builder.body(Body::from(body)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Local user (no auth) should get empty overview (different owner)
        let req = Request::builder().uri("/api/mpi/records/overview")
            .body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // The local owner's overview should not contain the remote user's record
    }

    #[tokio::test]
    async fn test_remote_forget_owner_verification() {
        let s = make_state_with_options(None, true, 0).await;
        let app = build_mpi_router(s.clone());

        // Local user stores a record
        let body_store = r#"{"content":"local secret","layer":"episode"}"#;
        let req = Request::builder().method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(body_store)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let resp_body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let resp_json: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        let record_id = resp_json["record_id"].as_str().unwrap().to_string();

        // Remote user tries to forget local user's record → should be FORBIDDEN
        let remote_user = IdentityKeyPair::generate();
        let forget_body = format!(r#"{{"record_id":"{}"}}"#, record_id);
        let ts = now_secs();
        let headers = sign_request(&remote_user, "POST", "/api/mpi/forget", forget_body.as_bytes(), ts);

        let mut builder = Request::builder().method("POST").uri("/api/mpi/forget")
            .header("content-type", "application/json");
        for (k, v) in &headers {
            builder = builder.header(*k, v.as_str());
        }
        let req = builder.body(Body::from(forget_body)).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }
}
