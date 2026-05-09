// ============================================
// File: crates/aeronyx-server/src/api/voice.rs
// ============================================
//! # Voice API
//!
//! ## Creation Reason
//! AeroNyx Voice needs to resolve a peer's P2P identity public key to its
//! VPN virtual IP before establishing a UDP direct-connect voice call,
//! allowing clients to bypass the relay server and send packets directly.
//!
//! ## Main Functionality
//! - `GET /api/peer-virtual-ip?pubkey=<64-char hex>`
//!   Resolves the specified Ed25519 public key (32 bytes) to an active session,
//!   returning its VPN virtual IP (100.64.0.x), online status, and last-seen
//!   timestamp.
//!
//! ## Dependencies
//! - `crate::services::SessionManager` — two-pass session lookup:
//!   1. `get_by_wallet()` — O(1) DashMap lookup (P2P-mode clients)
//!   2. `all_sessions().find()` — O(n) linear scan fallback (pure-VPN clients)
//! - `axum` — HTTP handler + Router
//! - Mounted by `server.rs::start_combined_api()` via `.merge()`
//!
//! ## Two-Pass Lookup Design
//!
//! ### Why two passes?
//! Clients fall into two categories:
//! - **P2P mode** (desktop auto, mobile after first P2P message):
//!   Sends `DeviceRegister` → recorded in `wallet_index` → fast O(1) lookup.
//! - **Pure VPN mode** (iOS/Android browsing only, no P2P chat):
//!   Only performs WireGuard handshake, NEVER sends `DeviceRegister`.
//!   `wallet_index` has no entry for these clients.
//!
//! Using only `get_by_wallet()` would return `online: false` for pure-VPN
//! clients even though they have an active session and valid virtual IP,
//! causing voice call UDP direct-connect to fail.
//!
//! ### Resolution
//! Pass 1 — `wallet_index` (O(1)):
//!   For P2P-mode clients. Hit rate is near 100%. Returns immediately on hit.
//!
//! Pass 2 — `all_sessions()` linear scan (O(n)):
//!   Fallback for pure-VPN clients. Compares `session.client_public_key` bytes
//!   directly. n is bounded by max_sessions (typically <= 1000). Acceptable
//!   because Voice API is called at most once per call setup, not per packet.
//!
//! ## Main Logical Flow
//! 1. Parse query param `pubkey` (64-char hex -> [u8; 32]); bad format -> offline
//! 2. Pass 1: `SessionManager::get_by_wallet(&pubkey_bytes)` (O(1))
//! 3. Pass 2: `all_sessions().find()` linear scan (O(n) fallback)
//! 4. Session found -> compute `last_seen`, return online=true + virtual_ip
//!    - `last_seen` = current unix seconds - `session.idle_time().as_secs()`
//!    - `saturating_sub` guards against integer underflow
//! 5. Not found -> return online=false, virtual_ip=null, last_seen=null
//!
//! ## Auth
//! No authentication required. Virtual IPs are network-layer routing
//! information, not user PII.
//!
//! ## Client-Side Staleness Check
//! A `get_by_wallet()` hit does not guarantee the session is truly active
//! (it may have just timed out but not yet been evicted by `cleanup_expired`).
//! Clients should apply a secondary check on `last_seen`:
//! ```dart
//! if (now - lastSeen > 120) return null; // treat as offline if idle > 2 min
//! ```
//!
//! ## Response Format
//! ```json
//! // Online
//! { "online": true,  "virtual_ip": "100.64.0.3", "last_seen": 1746614400 }
//! // Offline / not found / bad pubkey
//! { "online": false, "virtual_ip": null,          "last_seen": null }
//! ```
//!
//! ## ⚠️ Important Notes for Next Developer
//! - `wallet_index` key is the P2P identity public key (Ed25519, 32 bytes),
//!   written by the client's `DeviceRegister` message; identical to
//!   `session.client_public_key`.
//! - `get_by_wallet()` returns the most recently registered session (last device).
//!   Sufficient for voice calls; use `get_all_by_wallet()` for multi-device broadcast.
//! - `session.idle_time()` returns a Duration (time since last activity).
//!   Subtract from current unix time to get a timestamp.
//! - This router does NOT share MpiState. It injects Arc<SessionManager>
//!   as its own independent axum State.
//! - Pass 2 accesses `client_public_key.to_bytes()` directly rather than
//!   `wallet_bytes()` — functionally identical, avoids one extra method call.
//!
//! ## Last Modified
//! v1.0.0 - Initial implementation for AeroNyx Voice UDP direct-connect routing.
//!   Two-pass lookup: wallet_index O(1) + all_sessions O(n) fallback.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower::limit::RateLimitLayer;

use crate::services::SessionManager;

// ── Rate limit: 30 requests per 60 seconds per server instance.
// Voice API is called once per call setup — 30 req/min is generous
// for legitimate use and blocks enumeration attacks.
const RATE_LIMIT_REQUESTS: u64 = 30;
const RATE_LIMIT_WINDOW_SECS: u64 = 60;

// ============================================
// Request / Response Types
// ============================================

/// Query parameters for `GET /api/peer-virtual-ip`.
#[derive(Debug, Deserialize)]
pub struct PeerVirtualIpQuery {
    /// 64-character lowercase hex string representing the peer's
    /// Ed25519 P2P identity public key (32 bytes).
    pubkey: String,
}

/// Response body for `GET /api/peer-virtual-ip`.
#[derive(Debug, Serialize)]
pub struct PeerVirtualIpResponse {
    /// Whether the peer currently has an active VPN session.
    online: bool,

    /// The peer's VPN virtual IP (e.g. "100.64.0.3"), or null if offline.
    virtual_ip: Option<String>,

    /// Unix timestamp (seconds) of the peer's last packet activity,
    /// or null if offline.
    ///
    /// Clients MUST treat the peer as offline when:
    ///   `now_unix - last_seen > 120`
    /// because `cleanup_expired` runs every 60 s and a just-timed-out
    /// session may still appear in `wallet_index` until the next sweep.
    last_seen: Option<u64>,
}

impl PeerVirtualIpResponse {
    /// Constructs an offline / not-found response.
    #[inline]
    fn offline() -> Self {
        Self {
            online: false,
            virtual_ip: None,
            last_seen: None,
        }
    }
}

/// Error response body (used for validation failures only).
/// Rate limit errors are returned as plain 429 by Tower middleware.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: &'static str,
}

// ============================================
// Handler
// ============================================

/// `GET /api/peer-virtual-ip`
///
/// Resolves a peer's P2P identity public key to its current VPN virtual IP.
///
/// ## Lookup Strategy (Two-pass)
///
/// **Pass 1** — `wallet_index` O(1):
/// Serves P2P-mode clients whose `DeviceRegister` has already been processed.
/// Returns immediately on hit; covers ~100% of desktop and post-first-message
/// mobile clients.
///
/// **Pass 2** — `all_sessions()` O(n) linear scan:
/// Fallback for pure-VPN clients (iOS/Android in VPN-only mode, or mobile
/// before the first P2P message). Compares `session.client_public_key`
/// bytes directly against the requested pubkey. n is bounded by
/// `max_sessions` (typically <= 1000), and this endpoint is low-frequency
/// (called once per call setup), so O(n) is acceptable.
///
/// # Query Parameters
/// - `pubkey`: 64-char lowercase hex string (Ed25519 public key, 32 bytes)
///
/// # Responses
/// - Online  → `{ online: true,  virtual_ip: "100.64.0.x", last_seen: <unix_ts> }`
/// - Offline → `{ online: false, virtual_ip: null,          last_seen: null }`
/// - Bad key → `{ online: false, virtual_ip: null,          last_seen: null }`
pub async fn peer_virtual_ip_handler(
    State(sessions): State<Arc<SessionManager>>,
    Query(params): Query<PeerVirtualIpQuery>,
) -> Json<PeerVirtualIpResponse> {
    // ── Step 1: Decode hex pubkey → [u8; 32] ─────────────────────────────
    // Malformed hex or wrong length: return offline without leaking error details.
    let pubkey_bytes: [u8; 32] = match hex::decode(&params.pubkey) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => return Json(PeerVirtualIpResponse::offline()),
    };

    // ── Step 2: Pass 1 — wallet_index O(1) ───────────────────────────────
    // Fast path for P2P-mode clients (DeviceRegister already received).
    let session = if let Some(s) = sessions.get_by_wallet(&pubkey_bytes) {
        s
    } else {
        // ── Step 3: Pass 2 — full session scan O(n) ──────────────────────
        // Fallback for pure-VPN clients that never send DeviceRegister.
        // Iterates all active sessions and compares client_public_key bytes.
        match sessions
            .all_sessions()
            .into_iter()
            .find(|s| s.client_public_key.to_bytes() == pubkey_bytes)
        {
            Some(s) => s,
            None => return Json(PeerVirtualIpResponse::offline()),
        }
    };

    // ── Step 4: Compute last_seen unix timestamp ──────────────────────────
    // last_seen = now_unix_secs - idle_secs
    // idle_time() returns Duration since last activity (AtomicInstant::elapsed).
    // saturating_sub prevents underflow in pathological clock conditions.
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let last_seen = now_unix.saturating_sub(session.idle_time().as_secs());

    Json(PeerVirtualIpResponse {
        online: true,
        virtual_ip: Some(session.virtual_ip.to_string()),
        last_seen: Some(last_seen),
    })
}

// ============================================
// Router Builder
// ============================================

/// Builds the Voice API router.
///
/// Registers:
/// - `GET /api/peer-virtual-ip` → [`peer_virtual_ip_handler`]
///
/// ## Security layers applied
/// - **Rate limit**: 30 requests / 60 s (Tower `RateLimitLayer`).
///   Blocks enumeration attacks that sweep pubkeys to map online users.
///   Returns HTTP 429 when exceeded; Tower handles this automatically.
///
/// Called from `server.rs::start_combined_api()` and merged into the
/// main axum app via `.merge(build_voice_router(sessions))`.
///
/// This router does NOT share `MpiState`. It injects `Arc<SessionManager>`
/// as its own independent axum `State`, keeping the two routers fully isolated.
///
/// # Arguments
/// - `sessions`: shared `SessionManager` reference, injected as axum State.
pub fn build_voice_router(sessions: Arc<SessionManager>) -> Router {
    Router::new()
        .route("/api/peer-virtual-ip", get(peer_virtual_ip_handler))
        // Rate limit: 30 req / 60 s across all callers on this server instance.
        // Tower RateLimitLayer is a token-bucket applied to the entire service,
        // not per-IP. Per-IP limiting would require tower_governor (not in deps);
        // this global limit is sufficient to block bulk enumeration.
        .layer(RateLimitLayer::new(
            RATE_LIMIT_REQUESTS,
            std::time::Duration::from_secs(RATE_LIMIT_WINDOW_SECS),
        ))
        .with_state(sessions)
}
