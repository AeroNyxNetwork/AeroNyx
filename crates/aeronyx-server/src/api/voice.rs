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
//! Global fixed window: 30 requests / 60 seconds across all callers.
//! Blocks enumeration attacks that sweep pubkeys to map online users.
//!
//! Implementation uses one monotonic `parking_lot::Mutex` window because
//! `tower::limit::RateLimitLayer` produces a `RateLimit<Route>` that does not
//! implement `Clone`, which axum's `Router::layer` requires. The window is
//! shared by cloned routers and is independent of wall-clock corrections.
//!
//! ## Auth
//! This legacy lookup has no application-layer authentication and is exposed
//! only on the node API listeners. Virtual IP and activity time are routing
//! metadata, so future contact-gated rendezvous must replace this endpoint
//! before it is exposed outside the VPN trust boundary.
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
//! - Do NOT replace the shared rate limiter with `tower::limit::RateLimitLayer`
//!   — it breaks axum's Clone requirement on Router::layer.
//!
//! ## Last Modified
//! v2.8.13-VoiceMonotonicRateLimit - Replaced the single-key DashMap wall-clock
//!   limiter with one clone-safe monotonic window and deterministic tests.
//! v2.7.14-RustdocQuality - Marked router composition pseudocode as a
//!   non-standalone Rustdoc example.
//! v1.0.0 - Initial implementation for AeroNyx Voice UDP direct-connect routing.
//!   Two-pass lookup: wallet_index O(1) + all_sessions O(n) fallback.
//!   Global fixed-window rate limiter (30 req / 60 s).

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use parking_lot::Mutex;
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

/// Mutable state for the global fixed-window rate limiter.
///
/// `Instant` is deliberately process-local and monotonic. System clock
/// corrections therefore cannot extend or prematurely reset the window.
#[derive(Debug)]
struct RateLimitWindow {
    started_at: Instant,
    request_count: u64,
}

/// Global fixed-window rate limiter shared by every cloned Voice router.
///
/// ## Why not tower::limit::RateLimitLayer?
/// `RateLimit<Route>` does not implement `Clone`, which axum's
/// `Router::layer` requires. One mutex is sufficient because there is exactly
/// one global window and the critical section performs no allocation or I/O.
#[derive(Debug, Clone)]
struct RateLimiter {
    state: Arc<Mutex<RateLimitWindow>>,
}

impl RateLimiter {
    fn new() -> Self {
        Self::new_at(Instant::now())
    }

    fn new_at(started_at: Instant) -> Self {
        Self {
            state: Arc::new(Mutex::new(RateLimitWindow {
                started_at,
                request_count: 0,
            })),
        }
    }

    /// Returns `true` if the request is allowed, `false` if limit exceeded.
    ///
    /// Thread-safe: the window lock is held only during the counter update.
    fn check(&self) -> bool {
        self.check_at(Instant::now())
    }

    // [VOICE-RATE-MONOTONIC 2026-08-12 by Codex] Keep time injectable for
    // deterministic boundary tests while production always supplies Instant.
    fn check_at(&self, now: Instant) -> bool {
        let mut state = self.state.lock();
        let window_elapsed = now
            .checked_duration_since(state.started_at)
            .is_some_and(|elapsed| elapsed >= Duration::from_secs(RATE_LIMIT_WINDOW_SECS));
        if window_elapsed {
            state.started_at = now;
            state.request_count = 0;
        }

        if state.request_count >= RATE_LIMIT_REQUESTS {
            return false;
        }

        state.request_count += 1;
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
/// Both fields are cheaply cloneable (`Arc` / `Arc<Mutex<_>>`).
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
async fn peer_virtual_ip_handler(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_enforces_exact_window_and_resets_at_boundary() {
        let started_at = Instant::now();
        let limiter = RateLimiter::new_at(started_at);

        for _ in 0..RATE_LIMIT_REQUESTS {
            assert!(limiter.check_at(started_at));
        }
        assert!(!limiter.check_at(started_at));
        assert!(!limiter.check_at(started_at + Duration::from_secs(RATE_LIMIT_WINDOW_SECS - 1)));
        assert!(limiter.check_at(started_at + Duration::from_secs(RATE_LIMIT_WINDOW_SECS)));
    }

    #[test]
    fn rate_limiter_clones_share_capacity_and_ignore_earlier_instants() {
        let started_at = Instant::now();
        let limiter = RateLimiter::new_at(started_at);
        let clone = limiter.clone();

        for _ in 0..RATE_LIMIT_REQUESTS {
            assert!(limiter.check_at(started_at));
        }
        let earlier = started_at
            .checked_sub(Duration::from_secs(1))
            .unwrap_or(started_at);
        assert!(!clone.check_at(earlier));
        assert!(clone.check_at(started_at + Duration::from_secs(RATE_LIMIT_WINDOW_SECS)));
    }
}
