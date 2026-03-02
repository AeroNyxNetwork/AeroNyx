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
//! ## Modification Reason
//! - 🌟 v0.4.0: Added P2P broadcast on `POST /api/fact` — after storing
//!   locally, the Fact is encrypted and sent to all connected sessions.
//! - 🌟 v0.4.0: Added `POST /api/sync` endpoint to trigger P2P catch-up
//!   (broadcasts a SyncRequest to all connected peers).
//! - 🌟 v0.4.0: `ApiState` extended with `sessions`, `udp`, `crypto`,
//!   and `memchain_config` for P2P operations.
//!
//! ## Main Functionality
//! - `start_api_server()` — spawn the axum server as a tokio task
//! - `POST /api/fact` — create a new Fact (JSON → sign → MemPool → AOF → broadcast)
//! - `GET /api/facts?n=10` — query recent Facts from MemPool
//! - `GET /api/status` — MemChain health check (pool size, AOF writes)
//! - `POST /api/sync` — 🌟 trigger P2P SyncRequest broadcast
//!
//! ## Request / Response Formats
//!
//! ### POST /api/fact
//! ```json
//! // Request
//! { "subject": "user.preference", "predicate": "favorite_language", "object": "Rust" }
//! // Response (201 Created)
//! { "fact_id": "a1b2c3d4...", "timestamp": 1700000000, "subject": "...", ... }
//! ```
//!
//! ### POST /api/sync
//! ```json
//! // Request: empty body
//! // Response (200 OK)
//! { "status": "sync_requested", "last_known_hash": "abc123...", "peers_contacted": 5 }
//! ```
//!
//! ## Dependencies
//! - `axum` for HTTP routing
//! - `aeronyx_core::crypto::IdentityKeyPair` for Ed25519 signing
//! - `aeronyx_core::crypto::transport::{TransportCrypto, DefaultTransportCrypto}` for encryption
//! - `aeronyx_core::ledger::Fact` for data model
//! - `aeronyx_core::protocol::memchain::encode_memchain` for 0xAE encoding
//! - `MemPool` and `AofWriter` from services
//! - `SessionManager` and `UdpTransport` for P2P
//!
//! ## ⚠️ Important Note for Next Developer
//! - All handlers are non-blocking (async, no `block_in_place`).
//! - AofWriter is behind `TokioMutex` — hold the lock briefly.
//! - Fact signing uses the server's own IdentityKeyPair.
//! - P2P broadcast failures for individual sessions are logged but
//!   do NOT fail the API response.
//! - Broadcast is fire-and-forget via `tokio::spawn`.
//!
//! ## Last Modified
//! v0.3.0 - 🌟 Initial API implementation for MemChain Phase 1
//! v0.4.0 - 🌟 P2P broadcast on fact creation + POST /api/sync

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
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::ledger::Fact;
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::DataPacket;
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;

use crate::config::MemChainConfig;
use crate::services::memchain::{AofWriter, MemPool};
use crate::services::SessionManager;

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
    /// 🌟 Session manager for enumerating connected peers.
    pub sessions: Arc<SessionManager>,
    /// 🌟 UDP transport for sending P2P messages.
    pub udp: Arc<UdpTransport>,
    /// 🌟 Transport crypto for encrypting outbound messages.
    pub crypto: DefaultTransportCrypto,
    /// 🌟 MemChain config (for mode check).
    pub memchain_config: MemChainConfig,
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

/// Response for `POST /api/sync`.
#[derive(Debug, Serialize)]
pub struct SyncResponse {
    /// Status message.
    pub status: String,
    /// Last known hash (hex), or null if pool is empty.
    pub last_known_hash: Option<String>,
    /// Number of peers the sync request was sent to.
    pub peers_contacted: usize,
}

// ============================================
// P2P Broadcast Helper
// ============================================

/// Encrypts a MemChain message and sends it to all connected sessions.
///
/// This is fire-and-forget: individual session failures are logged but
/// do not propagate errors. The function is designed to be called from
/// a `tokio::spawn` context.
///
/// # Arguments
/// * `msg` - The MemChainMessage to broadcast
/// * `sessions` - Session manager to enumerate peers
/// * `udp` - UDP transport for sending
/// * `crypto` - Transport crypto for per-session encryption
async fn broadcast_to_all_sessions(
    msg: MemChainMessage,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    crypto: DefaultTransportCrypto,
) -> usize {
    // Encode the MemChain message with 0xAE prefix
    let plaintext = match encode_memchain(&msg) {
        Ok(p) => p,
        Err(e) => {
            error!("[API_BROADCAST] ❌ Failed to encode MemChain message: {}", e);
            return 0;
        }
    };

    let all_sessions = sessions.all_sessions();
    let mut sent_count = 0usize;

    for session in &all_sessions {
        if !session.is_established() {
            continue;
        }

        // Get next TX counter for this session
        let counter = session.next_tx_counter();

        // Encrypt plaintext with session's key
        let encrypted_len = plaintext.len() + ENCRYPTION_OVERHEAD;
        let mut encrypted = vec![0u8; encrypted_len];
        let actual_len = match crypto.encrypt(
            &session.session_key,
            counter,
            session.id.as_bytes(),
            &plaintext,
            &mut encrypted,
        ) {
            Ok(len) => len,
            Err(e) => {
                warn!(
                    session_id = %session.id,
                    error = %e,
                    "[API_BROADCAST] ⚠️ Encryption failed, skipping session"
                );
                continue;
            }
        };
        encrypted.truncate(actual_len);

        // Build DataPacket
        let data_packet = DataPacket::new(
            *session.id.as_bytes(),
            counter,
            encrypted,
        );

        let packet_bytes = encode_data_packet(&data_packet).to_vec();

        // Send via UDP
        if let Err(e) = udp.send(&packet_bytes, &session.client_endpoint).await {
            warn!(
                session_id = %session.id,
                endpoint = %session.client_endpoint,
                error = %e,
                "[API_BROADCAST] ⚠️ UDP send failed, skipping session"
            );
            continue;
        }

        sent_count += 1;
        debug!(
            session_id = %session.id,
            "[API_BROADCAST] ✅ Sent to session"
        );
    }

    sent_count
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
/// 6. 🌟 If P2P mode: broadcast to all connected sessions.
/// 7. Return the signed Fact as JSON.
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
        }
    }

    // 🌟 P2P broadcast (if enabled)
    if state.memchain_config.is_p2p() {
        let msg = MemChainMessage::BroadcastFact(fact.clone());
        let sessions = Arc::clone(&state.sessions);
        let udp = Arc::clone(&state.udp);
        let crypto = state.crypto.clone();

        // Fire-and-forget: don't block the HTTP response
        tokio::spawn(async move {
            let count = broadcast_to_all_sessions(msg, sessions, udp, crypto).await;
            info!(
                fact_id = %fact_id_hex,
                peers_sent = count,
                "[API] 📡 BroadcastFact sent to peers"
            );
        });
    }

    info!(
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

    let mode_str = if state.memchain_config.is_p2p() {
        "p2p"
    } else {
        "local"
    };

    let body = StatusResponse {
        memchain_enabled: true,
        mode: mode_str.to_string(),
        mempool_count: state.mempool.count(),
        mempool_total_accepted: state.mempool.total_accepted(),
        mempool_total_rejected: state.mempool.total_rejected(),
        aof_writes,
    };

    (StatusCode::OK, Json(body))
}

/// `POST /api/sync` — 🌟 Trigger P2P catch-up.
///
/// Broadcasts a `SyncRequest` to all connected sessions, asking them
/// to send any Facts this node is missing.
///
/// The `last_known_hash` is the `fact_id` of the most recently inserted
/// Fact in the local MemPool. If the pool is empty, `[0u8; 32]` is sent
/// (meaning "give me everything").
async fn trigger_sync(
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    if !state.memchain_config.is_p2p() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "P2P sync is only available in p2p mode. Current mode: local"
            })),
        );
    }

    // Determine last known hash
    let last_hash = state.mempool.last_fact_id().unwrap_or([0u8; 32]);
    let last_hash_hex = if last_hash == [0u8; 32] {
        None
    } else {
        Some(hex::encode(last_hash))
    };

    // Build SyncRequest
    let msg = MemChainMessage::SyncRequest {
        last_known_hash: last_hash,
    };

    let sessions = Arc::clone(&state.sessions);
    let udp = Arc::clone(&state.udp);
    let crypto = state.crypto.clone();

    // Broadcast SyncRequest to all peers
    let peers_contacted = broadcast_to_all_sessions(
        msg,
        sessions,
        udp,
        crypto,
    ).await;

    info!(
        last_known_hash = ?last_hash_hex,
        peers_contacted = peers_contacted,
        "[API] 📡 SyncRequest broadcast complete"
    );

    let body = SyncResponse {
        status: "sync_requested".to_string(),
        last_known_hash: last_hash_hex,
        peers_contacted,
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
        .route("/api/sync", post(trigger_sync))
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
/// * `sessions` - 🌟 Session manager for P2P broadcast
/// * `udp` - 🌟 UDP transport for sending P2P messages
/// * `memchain_config` - 🌟 MemChain configuration (mode, trust)
/// * `shutdown_rx` - Broadcast receiver for graceful shutdown
///
/// # Returns
/// `JoinHandle` for the API server task.
pub fn start_api_server(
    listen_addr: SocketAddr,
    mempool: Arc<MemPool>,
    aof_writer: Arc<TokioMutex<AofWriter>>,
    identity: IdentityKeyPair,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    memchain_config: MemChainConfig,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) -> tokio::task::JoinHandle<()> {
    let state = Arc::new(ApiState {
        mempool,
        aof_writer,
        identity,
        sessions,
        udp,
        crypto: DefaultTransportCrypto::new(),
        memchain_config,
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

    /// Helper to build a test state with real AOF (needed for write tests).
    /// Sessions / UDP / crypto are stubbed out since P2P is not tested here.
    async fn make_test_state_with_aof(aof_path: &std::path::Path) -> Arc<ApiState> {
        let aof = AofWriter::open(aof_path).await.unwrap();

        // We need a real UdpTransport for the state struct, but won't
        // actually send anything in local-mode tests.
        let udp = Arc::new(
            UdpTransport::bind_addr("127.0.0.1:0".parse().unwrap())
                .await
                .unwrap()
        );

        Arc::new(ApiState {
            mempool: Arc::new(MemPool::new()),
            aof_writer: Arc::new(TokioMutex::new(aof)),
            identity: IdentityKeyPair::generate(),
            sessions: Arc::new(SessionManager::new(100, std::time::Duration::from_secs(300))),
            udp,
            crypto: DefaultTransportCrypto::new(),
            memchain_config: MemChainConfig::default(), // local mode
        })
    }

    #[tokio::test]
    async fn test_status_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let aof_path = dir.path().join(".memchain");
        let state = make_test_state_with_aof(&aof_path).await;

        // Add a test fact
        let fact = Fact::new(100, "s".into(), "p".into(), "o".into());
        state.mempool.add_fact(fact);

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
        assert_eq!(json["mode"], "local");
    }

    #[tokio::test]
    async fn test_create_and_list_facts() {
        let dir = tempfile::tempdir().unwrap();
        let aof_path = dir.path().join(".memchain");
        let state = make_test_state_with_aof(&aof_path).await;

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
        let state = make_test_state_with_aof(&aof_path).await;

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

    #[tokio::test]
    async fn test_sync_rejected_in_local_mode() {
        let dir = tempfile::tempdir().unwrap();
        let aof_path = dir.path().join(".memchain");
        let state = make_test_state_with_aof(&aof_path).await;

        let app = build_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/api/sync")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}
