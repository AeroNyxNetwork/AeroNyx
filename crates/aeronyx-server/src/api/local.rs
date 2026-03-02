// ============================================
// File: crates/aeronyx-server/src/api/local.rs
// ============================================
//! # Local Agent API — Axum HTTP Server
//!
//! ## Creation Reason
//! Provides HTTP endpoints for local AI Agents to interact with the
//! MemChain ledger. This is the "front door" for OpenClaw and other
//! tools to write and read memory Facts.
//!
//! ## Main Functionality
//! - `start_api_server()` — spawn the axum server as a tokio task
//! - `POST /api/fact` — create a new Fact (JSON → sign → MemPool → AOF)
//! - `GET /api/facts?n=10` — query recent Facts from MemPool
//! - `GET /api/status` — MemChain health check (pool size, AOF writes)
//!
//! ## Request / Response Formats
//!
//! ### POST /api/fact
//! ```json
//! // Request
//! {
//!   "subject": "user.preference",
//!   "predicate": "favorite_language",
//!   "object": "Rust"
//! }
//!
//! // Response (201 Created)
//! {
//!   "fact_id": "a1b2c3d4...",
//!   "timestamp": 1700000000,
//!   "subject": "user.preference",
//!   "predicate": "favorite_language",
//!   "object": "Rust",
//!   "origin": "ab12cd34...",
//!   "signature": "ef56..."
//! }
//! ```
//!
//! ### GET /api/facts?n=10
//! ```json
//! // Response (200 OK)
//! {
//!   "count": 2,
//!   "facts": [
//!     { "fact_id": "...", "subject": "...", ... },
//!     { "fact_id": "...", "subject": "...", ... }
//!   ]
//! }
//! ```
//!
//! ### GET /api/status
//! ```json
//! {
//!   "memchain_enabled": true,
//!   "mode": "local",
//!   "mempool_count": 42,
//!   "mempool_total_accepted": 100,
//!   "mempool_total_rejected": 3,
//!   "aof_writes": 97
//! }
//! ```
//!
//! ## Dependencies
//! - `axum` for HTTP routing
//! - `aeronyx_core::crypto::IdentityKeyPair` for Ed25519 signing
//! - `aeronyx_core::ledger::Fact` for data model
//! - `MemPool` and `AofWriter` from services
//!
//! ## ⚠️ Important Note for Next Developer
//! - All handlers are non-blocking (async, no `block_in_place`).
//! - AofWriter is behind `TokioMutex` — hold the lock briefly.
//! - Fact signing uses the server's own IdentityKeyPair.
//! - Future: add `POST /api/broadcast` to trigger P2P push.
//! - Future: add authentication (API key / JWT) if binding to non-loopback.
//!
//! ## Last Modified
//! v0.3.0 - 🌟 Initial API implementation for MemChain Phase 1

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex as TokioMutex;
use tracing::{error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::ledger::Fact;

use crate::services::memchain::{AofWriter, MemPool};

// ============================================
// Shared State
// ============================================

/// Shared application state injected into all axum handlers.
///
/// Wrapped in `Arc` and passed via `axum::extract::State`.
pub struct ApiState {
    /// In-memory Fact pool (thread-safe via DashMap).
    pub mempool: Arc<MemPool>,
    /// Append-only file writer (needs exclusive access via Mutex).
    pub aof_writer: Arc<TokioMutex<AofWriter>>,
    /// Server identity for signing new Facts.
    pub identity: IdentityKeyPair,
}

// ============================================
// Request / Response Types
// ============================================

/// Request body for `POST /api/fact`.
#[derive(Debug, Deserialize)]
pub struct CreateFactRequest {
    /// Subject of the triple (e.g. "user.preference").
    pub subject: String,
    /// Predicate / relation (e.g. "favorite_language").
    pub predicate: String,
    /// Object / value (e.g. "Rust").
    pub object: String,
}

/// Response for a single Fact (JSON-friendly representation).
#[derive(Debug, Serialize)]
pub struct FactResponse {
    /// Hex-encoded SHA-256 content hash.
    pub fact_id: String,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
    /// Subject.
    pub subject: String,
    /// Predicate.
    pub predicate: String,
    /// Object.
    pub object: String,
    /// Hex-encoded Ed25519 public key of the signer.
    pub origin: String,
    /// Hex-encoded Ed25519 signature.
    pub signature: String,
}

impl From<&Fact> for FactResponse {
    fn from(f: &Fact) -> Self {
        Self {
            fact_id: hex::encode(f.fact_id),
            timestamp: f.timestamp,
            subject: f.subject.clone(),
            predicate: f.predicate.clone(),
            object: f.object.clone(),
            origin: hex::encode(f.origin),
            signature: hex::encode(f.signature),
        }
    }
}

/// Response for `GET /api/facts`.
#[derive(Debug, Serialize)]
pub struct FactsListResponse {
    /// Number of facts returned.
    pub count: usize,
    /// Facts in reverse-chronological order.
    pub facts: Vec<FactResponse>,
}

/// Query parameters for `GET /api/facts`.
#[derive(Debug, Deserialize)]
pub struct FactsQuery {
    /// Number of recent facts to return (default: 10, max: 1000).
    #[serde(default = "default_facts_limit")]
    pub n: usize,
}

fn default_facts_limit() -> usize {
    10
}

/// Response for `GET /api/status`.
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    /// Whether MemChain is enabled.
    pub memchain_enabled: bool,
    /// Current mode ("local" or "p2p").
    pub mode: String,
    /// Number of facts currently in MemPool.
    pub mempool_count: usize,
    /// Total facts ever accepted.
    pub mempool_total_accepted: u64,
    /// Total facts ever rejected.
    pub mempool_total_rejected: u64,
    /// Total facts written to AOF in this session.
    pub aof_writes: u64,
}

// ============================================
// Route Handlers
// ============================================

/// `POST /api/fact` — Create and store a new Fact.
///
/// 1. Parse JSON request body.
/// 2. Generate timestamp and compute content hash.
/// 3. Sign the hash with server's Ed25519 key.
/// 4. Add to MemPool (hash validation + dedup).
/// 5. Persist to AOF.
/// 6. Return the signed Fact as JSON.
async fn create_fact(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<CreateFactRequest>,
) -> impl IntoResponse {
    // Validate input
    if req.subject.is_empty() || req.predicate.is_empty() || req.object.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "subject, predicate, and object must be non-empty"
            })),
        );
    }

    // Generate timestamp
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build Fact with auto-computed hash
    let mut fact = Fact::new(timestamp, req.subject, req.predicate, req.object);

    // Sign: signature = Ed25519(private_key, fact_id)
    fact.origin = state.identity.public_key_bytes();
    fact.signature = state.identity.sign(&fact.fact_id);

    let fact_id_hex = fact.id_hex();

    // Store in MemPool
    if !state.mempool.add_fact(fact.clone()) {
        // Duplicate or invalid — shouldn't happen for fresh facts
        warn!(
            fact_id = %fact_id_hex,
            "[API] Fact rejected by MemPool (duplicate or invalid hash)"
        );
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "Fact already exists or hash validation failed",
                "fact_id": fact_id_hex
            })),
        );
    }

    // Persist to AOF
    {
        let mut writer = state.aof_writer.lock().await;
        if let Err(e) = writer.append_fact(&fact).await {
            error!(
                fact_id = %fact_id_hex,
                error = %e,
                "[API] ❌ AOF write failed"
            );
            // Fact is in MemPool but not persisted — log but don't fail the request
            // (it will be persisted on next write or recovered from MemPool)
        }
    }

    // TODO (Phase 2): If P2P mode is enabled, broadcast the Fact:
    // 1. Encode: encode_memchain(&MemChainMessage::BroadcastFact(fact.clone()))
    // 2. For each connected session: encrypt with session key → send via UDP

    info!(
        fact_id = %fact_id_hex,
        pool_size = state.mempool.count(),
        "[API] ✅ Fact created and stored"
    );

    let response = FactResponse::from(&fact);
    (StatusCode::CREATED, Json(serde_json::json!(response)))
}

/// `GET /api/facts?n=10` — Query recent Facts.
async fn list_facts(
    State(state): State<Arc<ApiState>>,
    Query(params): Query<FactsQuery>,
) -> impl IntoResponse {
    // Clamp limit
    let limit = params.n.min(1000).max(1);

    let facts = state.mempool.recent(limit);
    let responses: Vec<FactResponse> = facts.iter().map(FactResponse::from).collect();

    let body = FactsListResponse {
        count: responses.len(),
        facts: responses,
    };

    (StatusCode::OK, Json(body))
}

/// `GET /api/status` — MemChain health check.
async fn status(
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    let aof_writes = {
        let writer = state.aof_writer.lock().await;
        writer.write_count()
    };

    let body = StatusResponse {
        memchain_enabled: true,
        mode: "local".to_string(),
        mempool_count: state.mempool.count(),
        mempool_total_accepted: state.mempool.total_accepted(),
        mempool_total_rejected: state.mempool.total_rejected(),
        aof_writes,
    };

    (StatusCode::OK, Json(body))
}

// ============================================
// Server Startup
// ============================================

/// Builds the axum Router with all MemChain API routes.
fn build_router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/api/fact", post(create_fact))
        .route("/api/facts", get(list_facts))
        .route("/api/status", get(status))
        .with_state(state)
}

/// Starts the MemChain Agent API HTTP server.
///
/// This function spawns a tokio task that runs the axum server.
/// It respects the shutdown signal by using `axum::serve` with
/// `with_graceful_shutdown`.
///
/// # Arguments
/// * `listen_addr` - Socket address to bind (e.g. `127.0.0.1:8421`)
/// * `mempool` - Shared MemPool instance
/// * `aof_writer` - Shared AofWriter instance
/// * `identity` - Server's Ed25519 identity for signing Facts
/// * `shutdown_rx` - Broadcast receiver for graceful shutdown
///
/// # Returns
/// `JoinHandle` for the API server task.
pub fn start_api_server(
    listen_addr: SocketAddr,
    mempool: Arc<MemPool>,
    aof_writer: Arc<TokioMutex<AofWriter>>,
    identity: IdentityKeyPair,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) -> tokio::task::JoinHandle<()> {
    let state = Arc::new(ApiState {
        mempool,
        aof_writer,
        identity,
    });

    let app = build_router(state);

    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(listen_addr).await {
            Ok(l) => {
                info!(
                    "[API] ✅ MemChain Agent API listening on http://{}",
                    listen_addr
                );
                l
            }
            Err(e) => {
                error!(
                    "[API] ❌ Failed to bind API server to {}: {}",
                    listen_addr, e
                );
                return;
            }
        };

        let server = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
                info!("[API] Received shutdown signal, stopping API server");
            });

        if let Err(e) = server.await {
            error!("[API] ❌ API server error: {}", e);
        }

        info!("[API] API server stopped");
    })
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn make_test_state() -> Arc<ApiState> {
        Arc::new(ApiState {
            mempool: Arc::new(MemPool::new()),
            aof_writer: Arc::new(TokioMutex::new(
                // We can't easily create AofWriter in tests without a file,
                // so we'll test routes that don't need AOF separately.
                // For integration tests, use tempfile.
                unreachable!("AOF not used in unit route tests"),
            )),
            identity: IdentityKeyPair::generate(),
        })
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        // Create a minimal state without AOF (status reads write_count via lock)
        let mempool = Arc::new(MemPool::new());
        // Add a test fact
        let fact = Fact::new(100, "s".into(), "p".into(), "o".into());
        mempool.add_fact(fact);

        // For status test, we need a real AOF
        let dir = tempfile::tempdir().unwrap();
        let aof_path = dir.path().join(".memchain");
        let aof = AofWriter::open(&aof_path).await.unwrap();

        let state = Arc::new(ApiState {
            mempool,
            aof_writer: Arc::new(TokioMutex::new(aof)),
            identity: IdentityKeyPair::generate(),
        });

        let app = build_router(state);

        let req = Request::builder()
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["memchain_enabled"], true);
        assert_eq!(json["mempool_count"], 1);
    }

    #[tokio::test]
    async fn test_create_and_list_facts() {
        let dir = tempfile::tempdir().unwrap();
        let aof_path = dir.path().join(".memchain");
        let aof = AofWriter::open(&aof_path).await.unwrap();

        let state = Arc::new(ApiState {
            mempool: Arc::new(MemPool::new()),
            aof_writer: Arc::new(TokioMutex::new(aof)),
            identity: IdentityKeyPair::generate(),
        });

        let app = build_router(state);

        // POST a new fact
        let req = Request::builder()
            .method("POST")
            .uri("/api/fact")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"subject":"user","predicate":"likes","object":"Rust"}"#))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["subject"], "user");
        assert!(!json["fact_id"].as_str().unwrap().is_empty());
        assert!(!json["signature"].as_str().unwrap().is_empty());

        // GET facts
        let req = Request::builder()
            .uri("/api/facts?n=5")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["count"], 1);
        assert_eq!(json["facts"][0]["subject"], "user");
    }

    #[tokio::test]
    async fn test_create_fact_empty_fields_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let aof_path = dir.path().join(".memchain");
        let aof = AofWriter::open(&aof_path).await.unwrap();

        let state = Arc::new(ApiState {
            mempool: Arc::new(MemPool::new()),
            aof_writer: Arc::new(TokioMutex::new(aof)),
            identity: IdentityKeyPair::generate(),
        });

        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/api/fact")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"subject":"","predicate":"likes","object":"Rust"}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
