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
//! ## Two-Pass Lookup Design
//!
//! ### Why two passes?
//! - **P2P mode** clients send `DeviceRegister` → recorded in `wallet_index`
//!   → O(1) lookup via `get_by_wallet()`.
//! - **Pure VPN mode** clients (iOS/Android, no P2P chat) never send
//!   `DeviceRegister` → not in `wallet_index` → need O(n) fallback via
//!   `all_sessions().find()`.
//!
//! ## Rate Limiting
//! Global token-bucket: 30 requests / 60 seconds across all callers.
//! Blocks enumeration attacks that sweep pubkeys to map online users.
//!
//! Implementation uses a `DashMap`-backed counter (no external middleware)
//! because `tower::limit::RateLimitLayer` produces a `RateLimit<Route>` that
//! does not implement `Clone`, which axum's `Router::layer` requires.
//! The DashMap approach is lock-free and Clone-safe.
//!
//! ## Auth
//! No authentication required. Virtual IPs are network-layer routing
//! information, not user PII.
//!
//! ## Client-Side Staleness Check
//! A `get_by_wallet()` hit does not guarantee the session is truly active.
//! Clients should apply a secondary check on `last_seen`:
//! ```dart
//! if (now - lastSeen > 120) return null; // treat as offline
//! ```
//!
//! ## ⚠️ Important Notes for Next Developer
//! - `wallet_index` key is the P2P identity public key (Ed25519, 32 bytes).
//! - `get_by_wallet()` returns the most recently registered session (last device).
//! - `session.idle_time()` returns a Duration — subtract from unix time for timestamp.
//! - This router uses `VoiceState` (sessions + rate_limiter) as its axum State,
//!   independent of MpiState. The two routers are fully isolated.
//! - Do NOT replace the DashMap rate limiter with `tower::limit::RateLimitLayer`
//!   — it breaks axum's Clone requirement on Router::layer.
//!
//! ## Last Modified
//! v2.7.14-RustdocQuality - Marked router composition pseudocode as a
//!   non-standalone Rustdoc example.
//! v1.0.0 - Initial implementation for AeroNyx Voice UDP direct-connect routing.
//!   Two-pass lookup: wallet_index O(1) + all_sessions O(n) fallback.
//!   DashMap-based global rate limiter (30 req / 60 s).

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::services::SessionManager;

// ============================================
// Rate Limiter
// ============================================

/// Rate limit: 30 requests per 60-second window (global, not per-IP).
/// Sufficient to block bulk enumeration; legitimate voice clients call
/// this endpoint at most once per call setup.
const RATE_LIMIT_REQUESTS: u64 = 30;
const RATE_LIMIT_WINDOW_SECS: u64 = 60;

/// Global token-bucket rate limiter backed by DashMap.
///
/// Tracks (window_start_unix_secs, request_count) under a single
/// "global" key. Resets automatically when the window expires.
///
/// ## Why not tower::limit::RateLimitLayer?
/// `RateLimit<Route>` does not implement `Clone`, which axum's
/// `Router::layer` requires. This DashMap approach is lock-free
/// and satisfies axum's Clone constraint.
#[derive(Debug, Clone)]
struct RateLimiter {
    state: Arc<DashMap<&'static str, (u64, u64)>>,
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            state: Arc::new(DashMap::new()),
        }
    }

    /// Returns `true` if the request is allowed, `false` if limit exceeded.
    ///
    /// Thread-safe: DashMap entry lock is held only during the counter update.
    fn check(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut entry = self.state.entry("global").or_insert((now, 0));
        let (window_start, count) = entry.value_mut();

        // Reset window if the current window has expired.
        if now.saturating_sub(*window_start) >= RATE_LIMIT_WINDOW_SECS {
            *window_start = now;
            *count = 0;
        }

        if *count >= RATE_LIMIT_REQUESTS {
            return false;
        }

        *count += 1;
        true
    }
}

// ============================================
// Combined Router State
// ============================================

/// Axum state for the Voice API router.
///
/// Bundles `SessionManager` and `RateLimiter` into a single `Clone`-able
/// state, because axum `Router` supports only one State type per router.
/// Both fields are cheaply cloneable (`Arc` / `Arc<DashMap>`).
#[derive(Clone)]
pub struct VoiceState {
    sessions: Arc<SessionManager>,
    rate_limiter: RateLimiter,
}

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
    #[inline]
    fn offline() -> Self {
        Self {
            online: false,
            virtual_ip: None,
            last_seen: None,
        }
    }
}

/// Error response body for HTTP 429 Too Many Requests.
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
/// ## Rate limiting
/// Returns HTTP 429 with `{"error":"rate limit exceeded"}` when the global
/// limit (30 req / 60 s) is exceeded.
///
/// ## Lookup Strategy (Two-pass)
///
/// **Pass 1** — `wallet_index` O(1):
/// P2P-mode clients whose `DeviceRegister` has been processed. Covers ~100%
/// of desktop and post-first-message mobile clients.
///
/// **Pass 2** — `all_sessions()` O(n) linear scan:
/// Fallback for pure-VPN clients that never send `DeviceRegister`.
/// n ≤ max_sessions (typically ≤ 1000). Acceptable for low-frequency
/// voice-setup calls (once per call, not per packet).
///
/// # Query Parameters
/// - `pubkey`: 64-char lowercase hex string (Ed25519 public key, 32 bytes)
///
/// # Responses
/// - Online      → `{ online: true,  virtual_ip: "100.64.0.x", last_seen: <ts> }`
/// - Offline     → `{ online: false, virtual_ip: null,          last_seen: null }`
/// - Bad pubkey  → `{ online: false, virtual_ip: null,          last_seen: null }`
/// - Rate limit  → HTTP 429 `{ error: "rate limit exceeded" }`
pub async fn peer_virtual_ip_handler(
    State(state): State<VoiceState>,
    Query(params): Query<PeerVirtualIpQuery>,
) -> Result<Json<PeerVirtualIpResponse>, (StatusCode, Json<ErrorResponse>)> {
    // ── Step 1: Rate limit check ──────────────────────────────────────────
    if !state.rate_limiter.check() {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: "rate limit exceeded",
            }),
        ));
    }

    // ── Step 2: Validate and decode hex pubkey → [u8; 32] ────────────────
    // Reject anything that is not exactly 64 hex characters.
    // Return offline (not an error) to avoid leaking structural information.
    if params.pubkey.len() != 64 {
        return Ok(Json(PeerVirtualIpResponse::offline()));
    }

    let pubkey_bytes: [u8; 32] = match hex::decode(&params.pubkey) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => return Ok(Json(PeerVirtualIpResponse::offline())),
    };

    // ── Step 3: Pass 1 — wallet_index O(1) ───────────────────────────────
    let session = if let Some(s) = state.sessions.get_by_wallet(&pubkey_bytes) {
        s
    } else {
        // ── Step 4: Pass 2 — full session scan O(n) ──────────────────────
        match state
            .sessions
            .all_sessions()
            .into_iter()
            .find(|s| s.client_public_key.to_bytes() == pubkey_bytes)
        {
            Some(s) => s,
            None => return Ok(Json(PeerVirtualIpResponse::offline())),
        }
    };

    // ── Step 5: Compute last_seen unix timestamp ──────────────────────────
    let now_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let last_seen = now_unix.saturating_sub(session.idle_time().as_secs());

    Ok(Json(PeerVirtualIpResponse {
        online: true,
        virtual_ip: Some(session.virtual_ip.to_string()),
        last_seen: Some(last_seen),
    }))
}

// ============================================
// Router Builder
// ============================================

/// Builds the Voice API router.
///
/// Registers:
/// - `GET /api/peer-virtual-ip` → [`peer_virtual_ip_handler`]
///
/// Uses `VoiceState` (sessions + rate_limiter) as axum State.
/// Does NOT share `MpiState` — the two routers are fully isolated.
///
/// Called from `server.rs::start_combined_api()`:
/// ```rust,ignore
/// let app = build_mpi_router(mpi_state)
///     .merge(build_voice_router(sessions));
/// ```
///
/// # Arguments
/// - `sessions`: shared `SessionManager`, injected into `VoiceState`.
pub fn build_voice_router(sessions: Arc<SessionManager>) -> Router {
    let state = VoiceState {
        sessions,
        rate_limiter: RateLimiter::new(),
    };
    Router::new()
        .route("/api/peer-virtual-ip", get(peer_virtual_ip_handler))
        .with_state(state)
}
