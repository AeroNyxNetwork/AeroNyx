// ============================================
// File: crates/aeronyx-server/src/api/local.rs
// ============================================
//! # Local Agent API — Axum HTTP Server (MPI + Legacy)
//!
//! ## Creation Reason
//! Provides HTTP endpoints for local AI Agents to interact with the
//! MemChain ledger. This is the "front door" for OpenClaw and other
//! tools to write and read memory.
//!
//! ## Modification Reason
//! - 🌟 v0.3.0: Initial API implementation (POST /api/fact, GET /api/facts)
//! - 🌟 v0.4.0: Added P2P broadcast on `POST /api/fact` + `POST /api/sync`
//! - 🌟 v1.0.0: **Major refactor** — Added MPI (Memory Protocol Interface)
//!   endpoints for the intelligent AI memory engine:
//!   - `POST /api/mpi/remember` — store a new memory (dedup + encrypt + persist)
//!   - `POST /api/mpi/recall` — semantic recall with composite scoring
//!   - `POST /api/mpi/forget` — revoke a memory (tombstone + erase content)
//!   - `GET  /api/mpi/status` — memory statistics and layer distribution
//!   Legacy endpoints (`/api/fact`, `/api/facts`, `/api/status`, `/api/sync`)
//!   are preserved for backward compatibility with existing P2P flows.
//!
//! ## MPI Request / Response Formats
//!
//! ### POST /api/mpi/remember
//! ```json
//! // Request
//! {
//!   "content": "User prefers dark mode and Rust programming",
//!   "layer": "identity",          // "identity" | "knowledge" | "episode"
//!   "topic_tags": ["preference", "programming"],
//!   "source_ai": "openclaw-v1",
//!   "embedding": [0.1, 0.2, ...]  // optional f32 vector
//! }
//! // Response (201 Created)
//! {
//!   "record_id": "a1b2c3d4...",
//!   "status": "created",
//!   "duplicate_of": null           // or hex ID if dedup detected
//! }
//! ```
//!
//! ### POST /api/mpi/recall
//! ```json
//! // Request
//! {
//!   "query": "What programming language does the user prefer?",
//!   "embedding": [0.1, 0.2, ...], // required f32 vector
//!   "top_k": 5,                   // optional, default 10
//!   "layer": null,                 // optional layer filter
//!   "token_budget": 2000           // optional, max tokens in response
//! }
//! // Response (200 OK)
//! {
//!   "memories": [
//!     {
//!       "record_id": "a1b2c3d4...",
//!       "layer": "identity",
//!       "score": 0.92,
//!       "content": "User prefers dark mode and Rust programming",
//!       "topic_tags": ["preference"],
//!       "source_ai": "openclaw-v1",
//!       "timestamp": 1700000000
//!     }
//!   ],
//!   "total_candidates": 42,
//!   "token_estimate": 156
//! }
//! ```
//!
//! ### POST /api/mpi/forget
//! ```json
//! // Request
//! { "record_id": "a1b2c3d4..." }
//! // Response (200 OK)
//! { "status": "revoked", "record_id": "a1b2c3d4..." }
//! ```
//!
//! ## Dependencies
//! - `axum` for HTTP routing
//! - `MemoryStorage` for SQLite persistence
//! - `VectorIndex` for semantic search
//! - `IdentityKeyPair` for Ed25519 signing
//! - Legacy: `MemPool`, `AofWriter` for Fact-based endpoints
//!
//! ## ⚠️ Important Note for Next Developer
//! - All handlers are non-blocking (async).
//! - MPI endpoints use `MemoryStorage` (SQLite) + `VectorIndex`.
//! - Legacy endpoints still use `MemPool` + `AofWriter`.
//! - The `owner` field for MPI operations uses the server's identity
//!   public key. In future phases, clients will provide their own wallet key.
//! - `embedding` in remember/recall is provided by the AI agent (e.g. OpenClaw).
//!   The server does NOT generate embeddings itself.
//! - P2P broadcast of MemoryRecords uses `BroadcastRecord` message type.
//!
//! ## Last Modified
//! v0.3.0 - Initial API implementation for MemChain Phase 1
//! v0.4.0 - P2P broadcast on fact creation + POST /api/sync
//! v1.0.0 - 🌟 MPI endpoints (remember/recall/forget/status)

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
#[allow(deprecated)]
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

/// Shared application state for legacy Fact-based endpoints.
///
/// The legacy API is deprecated — new integrations should use the MPI
/// endpoints in `api/mpi.rs`. This struct is preserved to keep
/// `/api/fact`, `/api/facts`, `/api/status`, `/api/sync` working
/// for backward compatibility with older AI agents.
#[deprecated(since = "2.1.0", note = "Use MPI endpoints (/api/mpi/*) via api::mpi::MpiState")]
pub struct ApiState {
    // Legacy state (Fact-based)
    pub mempool: Arc<MemPool>,
    pub aof_writer: Arc<TokioMutex<AofWriter>>,

    // Shared
    pub identity: IdentityKeyPair,
    pub sessions: Arc<SessionManager>,
    pub udp: Arc<UdpTransport>,
    pub crypto: DefaultTransportCrypto,
    pub memchain_config: MemChainConfig,
}

// ============================================
// Legacy Request / Response Types
// ============================================

/// Request body for `POST /api/fact` (legacy).
#[derive(Debug, Deserialize)]
pub struct CreateFactRequest {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Response for a single Fact (legacy).
#[derive(Debug, Serialize)]
pub struct FactResponse {
    pub fact_id: String,
    pub timestamp: u64,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub origin: String,
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

#[derive(Debug, Serialize)]
pub struct FactsListResponse {
    pub count: usize,
    pub facts: Vec<FactResponse>,
}

#[derive(Debug, Deserialize)]
pub struct FactsQuery {
    #[serde(default = "default_facts_limit")]
    pub n: usize,
}

fn default_facts_limit() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub memchain_enabled: bool,
    pub mode: String,
    pub mempool_count: usize,
    pub mempool_total_accepted: u64,
    pub mempool_total_rejected: u64,
    pub aof_writes: u64,
}

#[derive(Debug, Serialize)]
pub struct SyncResponse {
    pub status: String,
    pub last_known_hash: Option<String>,
    pub peers_contacted: usize,
}

// ============================================
// Legacy Route Handlers
// ============================================

async fn broadcast_to_all_sessions(
    msg: MemChainMessage,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    crypto: DefaultTransportCrypto,
) -> usize {
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

        let counter = session.next_tx_counter();
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

        let data_packet = DataPacket::new(
            *session.id.as_bytes(),
            counter,
            encrypted,
        );
        let packet_bytes = encode_data_packet(&data_packet).to_vec();

        if let Err(e) = udp.send(&packet_bytes, &session.client_endpoint).await {
            warn!(
                session_id = %session.id,
                error = %e,
                "[API_BROADCAST] ⚠️ UDP send failed, skipping session"
            );
            continue;
        }

        sent_count += 1;
    }

    sent_count
}

// ============================================
// P2P Broadcast Helper
// ============================================

/// `POST /api/fact` — Create and store a new Fact (legacy).
async fn create_fact(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<CreateFactRequest>,
) -> impl IntoResponse {
    if req.subject.is_empty() || req.predicate.is_empty() || req.object.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "subject, predicate, and object must be non-empty"
            })),
        );
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut fact = Fact::new(timestamp, req.subject, req.predicate, req.object);
    fact.origin = state.identity.public_key_bytes();
    fact.signature = state.identity.sign(&fact.fact_id);

    let fact_id_hex = fact.id_hex();

    if !state.mempool.add_fact(fact.clone()) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": "Fact already exists or hash validation failed",
                "fact_id": fact_id_hex
            })),
        );
    }

    {
        let mut writer = state.aof_writer.lock().await;
        if let Err(e) = writer.append_fact(&fact).await {
            error!(fact_id = %fact_id_hex, error = %e, "[API] ❌ AOF write failed");
        }
    }

    if state.memchain_config.is_p2p() {
        let msg = MemChainMessage::BroadcastFact(fact.clone());
        let sessions = Arc::clone(&state.sessions);
        let udp = Arc::clone(&state.udp);
        let crypto = state.crypto.clone();
        tokio::spawn(async move {
            let count = broadcast_to_all_sessions(msg, sessions, udp, crypto).await;
            debug!(fact_id = %fact_id_hex, peers_sent = count, "[API] Legacy BroadcastFact sent");
        });
    }

    let response = FactResponse::from(&fact);
    (StatusCode::CREATED, Json(serde_json::json!(response)))
}

/// `GET /api/facts?n=10` — Query recent Facts (legacy).
async fn list_facts(
    State(state): State<Arc<ApiState>>,
    Query(params): Query<FactsQuery>,
) -> impl IntoResponse {
    let limit = params.n.min(1000).max(1);
    let facts = state.mempool.recent(limit);
    let responses: Vec<FactResponse> = facts.iter().map(FactResponse::from).collect();

    (StatusCode::OK, Json(FactsListResponse {
        count: responses.len(),
        facts: responses,
    }))
}

/// `GET /api/status` — Legacy health check.
async fn status(
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    let aof_writes = {
        let writer = state.aof_writer.lock().await;
        writer.write_count()
    };

    let mode_str = if state.memchain_config.is_p2p() { "p2p" } else { "local" };

    (StatusCode::OK, Json(StatusResponse {
        memchain_enabled: true,
        mode: mode_str.to_string(),
        mempool_count: state.mempool.count(),
        mempool_total_accepted: state.mempool.total_accepted(),
        mempool_total_rejected: state.mempool.total_rejected(),
        aof_writes,
    }))
}

/// `POST /api/sync` — Trigger P2P catch-up (legacy).
async fn trigger_sync(
    State(state): State<Arc<ApiState>>,
) -> impl IntoResponse {
    if !state.memchain_config.is_p2p() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "P2P sync is only available in p2p mode"
            })),
        );
    }

    let last_hash = state.mempool.last_fact_id().unwrap_or([0u8; 32]);
    let last_hash_hex = if last_hash == [0u8; 32] { None } else { Some(hex::encode(last_hash)) };

    let msg = MemChainMessage::SyncRequest { last_known_hash: last_hash };
    let peers_contacted = broadcast_to_all_sessions(
        msg,
        Arc::clone(&state.sessions),
        Arc::clone(&state.udp),
        state.crypto.clone(),
    ).await;

    (StatusCode::OK, Json(serde_json::json!(SyncResponse {
        status: "sync_requested".to_string(),
        last_known_hash: last_hash_hex,
        peers_contacted,
    })))
}

// ============================================
// Router + Server Startup
// ============================================

/// Builds the legacy Axum Router (Fact-based endpoints only).
/// MPI routes are in `api/mpi.rs` and mounted separately in `server.rs`.
#[allow(deprecated)]
fn build_legacy_router(state: Arc<ApiState>) -> Router {
    Router::new()
        .route("/api/fact", post(create_fact))
        .route("/api/facts", get(list_facts))
        .route("/api/status", get(status))
        .route("/api/sync", post(trigger_sync))
        .with_state(state)
}

/// Starts the legacy MemChain Agent API HTTP server.
///
/// **Deprecated**: New code should use `api::mpi::build_mpi_router` instead.
/// This function is preserved for backward compatibility with older AI agents
/// that still use `/api/fact` and `/api/facts`.
#[allow(deprecated)]
pub fn start_legacy_api_server(
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

    let app = build_legacy_router(state);

    tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(listen_addr).await {
            Ok(l) => {
                info!("[LEGACY_API] Listening on http://{}", listen_addr);
                l
            }
            Err(e) => {
                error!("[LEGACY_API] Bind failed: {}", e);
                return;
            }
        };

        let server = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.recv().await;
            });

        if let Err(e) = server.await {
            error!("[LEGACY_API] Error: {}", e);
        }
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
    use tower::ServiceExt;

    #[allow(deprecated)]
    async fn make_test_state(aof_path: &std::path::Path) -> Arc<ApiState> {
        let aof = AofWriter::open(aof_path).await.unwrap();

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
            memchain_config: MemChainConfig::default(),
        })
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_legacy_status_endpoint() {
        let dir = tempfile::tempdir().unwrap();
        let state = make_test_state(&dir.path().join(".memchain")).await;

        let fact = Fact::new(100, "s".into(), "p".into(), "o".into());
        state.mempool.add_fact(fact);

        let app = build_legacy_router(state);

        let req = Request::builder()
            .uri("/api/status")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[allow(deprecated)]
    async fn test_legacy_create_and_list_facts() {
        let dir = tempfile::tempdir().unwrap();
        let state = make_test_state(&dir.path().join(".memchain")).await;
        let app = build_legacy_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/api/fact")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"subject":"user","predicate":"likes","object":"Rust"}"#))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let req = Request::builder()
            .uri("/api/facts?n=5")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
