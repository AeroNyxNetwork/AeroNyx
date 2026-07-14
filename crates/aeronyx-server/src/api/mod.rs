// ============================================
// File: crates/aeronyx-server/src/api/mod.rs
// ============================================
//! # MemChain Local API
//!
//! ## Creation Reason
//! Provides a local HTTP API for trusted node-local clients to read and write
//! memory Facts into the MemChain ledger. The API binds to loopback by default
//! (`127.0.0.1:8421`) and is NOT exposed to the public network.
//!
//! ## v2.4.0 File Split
//! mpi.rs was split into 3 files for maintainability:
//! - `mpi.rs` — MpiState, AuthenticatedOwner, auth middleware, router, helpers
//! - `mpi_handlers.rs` — Original 7 endpoint handlers (remember, recall, forget,
//!   status, embed, record, overview)
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints (11 new)
//!
//! External API is UNCHANGED — this file re-exports the same symbols:
//! `{build_mpi_router, MpiState, BaselineSnapshot}`
//!
//! server.rs, log_handler.rs, and ws_client.rs all import from
//! `crate::api::mpi::` — their code does NOT need to change.
//!
//! ## Submodules
//! - [`mpi`]: Core MPI types, auth, router (entry point)
//! - [`mpi_handlers`]: Original endpoint handlers (remember, recall, etc.)
//! - [`mpi_graph_handlers`]: v2.4.0 cognitive graph endpoints
//! - [`recall_handler`]: Hybrid recall pipeline (vector + BM25 + graph + RRF)
//! - [`log_handler`]: /log endpoint with rule engine + entropy filter + privacy tags
//! - [`supernode_handlers`]: v2.5.0 SuperNode management endpoints
//! - [`auth`]: v1.0.0-MultiTenant JWT token issuance for SaaS mode
//! - [`admin_handlers`]: v1.0.0-MultiTenant Admin endpoints (volumes, pool, usage)
//! - [`local`]: Legacy Axum router (deprecated)
//! - [`voice`]: v1.0.0-Voice Peer virtual IP resolution for UDP direct-connect
//! - [`discovery`]: v0.1.0 Discovery snapshot/gossip endpoints
//! - [`chat_peer`]: v0.1.0 node-to-node encrypted chat envelope relay
//! - [`memchain_peer`]: v2.7.0 signed node-to-node commitment block ranges
//!
//! ⚠️ Important Note for Next Developer:
//! - When adding new ORIGINAL-style endpoints → add to mpi_handlers.rs
//! - When adding new GRAPH/COGNITIVE endpoints → add to mpi_graph_handlers.rs
//! - When adding new SUPERNODE endpoints → add to supernode_handlers.rs
//! - When adding new ADMIN endpoints → add to admin_handlers.rs
//! - Register all routes in mpi.rs::build_mpi_router() regardless of which file
//!   the handler lives in
//! - Re-exports below MUST stay in sync — server.rs depends on them
//! - auth.rs and admin_handlers.rs are SaaS-mode only but always compiled.
//!   The routes are conditionally registered in build_mpi_router() based on mode.
//! - voice.rs injects its own Arc<SessionManager> State independently of MpiState.
//!   It is merged into the combined API router in server.rs::start_combined_api().
//! - memchain_peer.rs is a public node-peer surface, not a client memory API.
//!   It must keep PeerStore admission and return commitments only.
//! - Every outbound peer response must be read through the bounded helpers in
//!   this module. `Content-Length` is advisory; the streaming byte count is the
//!   authoritative memory boundary and peer-controlled bodies are never logged.
//!
//! ## Last Modified
//! v0.3.0 - Initial Agent API for MemChain Phase 1
//! v0.4.0 - Extended for Phase 3: P2P broadcast + POST /api/sync
//! v2.4.0-GraphCognition - Split mpi.rs into 3 files; added mpi_handlers,
//!   mpi_graph_handlers, recall_handler submodules
//! v2.4.0+Privacy - log_handler updated with privacy tag stripping
//! v2.5.0+SuperNode Phase D - Added supernode_handlers submodule
//! v1.0.0-MultiTenant - Added auth + admin_handlers submodules for SaaS mode;
//!   MpiState extended with Mode enum + SaaS pool fields;
//!   build_mpi_router conditionally registers auth + admin routes in SaaS mode.
//! v1.0.0-Voice - Added voice submodule:
//!   GET /api/peer-virtual-ip?pubkey=<hex> → { online, virtual_ip, last_seen }
//!   Two-pass lookup: wallet_index (O(1)) → all_sessions fallback (O(n)).
//!   No auth required (virtual IP is network-layer routing info, not PII).
//! v0.1.0-DiscoveryAPI - Added discovery submodule:
//!   GET /api/discovery/snapshot and POST /api/discovery/gossip.
//! v0.1.0-ChatPeerRelay - Added chat_peer submodule:
//!   POST /api/chat/peer/relay for inter-node encrypted envelope relay.
//! v2.7.0-BlockSync - Added authenticated `/api/memchain/peer/block-range`.
//! v2.7.19-PublicApiBounds - Centralized bounded peer HTTP response decoding.

use serde::de::DeserializeOwned;

/// Privacy-safe failure classes for bounded responses from untrusted peers.
///
/// Deliberately avoid carrying response bodies or parser details: callers may
/// expose these reasons through health telemetry, and peer-controlled content
/// must never become an accidental logging channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundedHttpResponseError {
    /// The declared or streamed response exceeded its protocol ceiling.
    TooLarge,
    /// The response stream failed before a complete bounded body was read.
    BodyRead,
    /// The bounded body did not match the expected JSON response schema.
    JsonDecode,
}

impl BoundedHttpResponseError {
    /// Returns a stable privacy-safe telemetry bucket.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::TooLarge => "response_too_large",
            Self::BodyRead => "response_body_read_failed",
            Self::JsonDecode => "response_json_decode_failed",
        }
    }
}

impl std::fmt::Display for BoundedHttpResponseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Relay acknowledgements contain only booleans, counters, and reason codes.
pub(crate) const PEER_ACK_RESPONSE_MAX_BYTES: usize = 16 * 1024;

/// Reads an untrusted HTTP response without allowing a peer to grow the
/// process heap without bound.
///
/// `Content-Length` is only an early rejection. The streaming check remains
/// authoritative because the header may be absent, incorrect, or refer to
/// compressed bytes.
pub(crate) async fn read_bounded_http_response(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, BoundedHttpResponseError> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(BoundedHttpResponseError::TooLarge);
    }

    let initial_capacity = response
        .content_length()
        .unwrap_or_default()
        .min(max_bytes as u64) as usize;
    let mut body = Vec::with_capacity(initial_capacity);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| BoundedHttpResponseError::BodyRead)?
    {
        if chunk.len() > max_bytes.saturating_sub(body.len()) {
            return Err(BoundedHttpResponseError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Decodes one schema-checked JSON response after enforcing its byte ceiling.
pub(crate) async fn decode_bounded_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<T, BoundedHttpResponseError> {
    let body = read_bounded_http_response(response, max_bytes).await?;
    serde_json::from_slice(&body).map_err(|_| BoundedHttpResponseError::JsonDecode)
}

// ── Core MPI module (state, auth, router) ──
pub mod mpi;
// ── Handler modules ──
pub mod mpi_graph_handlers;
pub mod mpi_handlers;
pub mod recall_handler;
// ── /log endpoint ──
pub mod log_handler;
// ── v2.5.0+SuperNode: Task queue management + monitoring ──
pub mod supernode_handlers;
// ── v1.0.0-MultiTenant: JWT token issuance (SaaS mode only, always compiled) ──
pub mod auth;
// ── v1.0.0-MultiTenant: Admin endpoints (SaaS mode only, always compiled) ──
pub mod admin_handlers;
// ── Legacy API (deprecated) ──
pub mod local;
// ── v1.0.0-Voice: Peer virtual IP resolution for UDP direct-connect routing ──
pub mod chat_peer;
pub mod discovery;
pub mod memchain_peer;
pub mod voice;
pub mod vpn_health;

// ── Re-exports (unchanged from v2.3.0 — external callers unaffected) ──
pub use mpi::{build_mpi_router, BaselineSnapshot, MpiState};
// v1.0.0-MultiTenant: export Mode for server.rs SaaS init branch
#[allow(deprecated)]
pub use local::start_legacy_api_server;
pub use mpi::Mode;
