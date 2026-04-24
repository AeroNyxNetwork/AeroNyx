// ============================================
// File: crates/aeronyx-server/src/services/session.rs
// ============================================
//! # Session Management Service
//!
//! ## Modification Reason
//! - Fixed replay detection to use sliding window algorithm instead of
//!   strict increment.
//! - 🌟 v1.1.0-ChatRelay: Added `wallet_index` reverse lookup (wallet → SessionId)
//!   for zero-knowledge chat relay.
//! - 🌟 v1.2.0-MultiDevice: `wallet_index` changed from
//!   `DashMap<[u8;32], SessionId>` to `DashMap<[u8;32], Vec<DeviceEntry>>`.
//!   Added `DeviceId` type alias, `DeviceEntry` struct, and three new methods:
//!   `register_device`, `remove_device`, `get_all_by_wallet`.
//!   `get_by_wallet` now returns the most recently registered session
//!   (last entry in the Vec) to preserve backward-compatible single-device
//!   routing for callers that only need one session.
//!
//! - 🔴 **v1.2.1-ReplayWindowFix (CRITICAL BUG FIX)**:
//!   The previous `ReplayWindow::new()` initialised `highest_seen = u64::MAX`
//!   as a "no packets seen yet" sentinel. This was fundamentally broken:
//!   the `counter > self.highest_seen` check in `check_and_record()` could
//!   NEVER be true (no u64 value is greater than u64::MAX), so the
//!   `AcceptAndAdvance` branch was unreachable. `highest_seen` stayed at
//!   `u64::MAX` forever, the bitmap's clear-bit logic never ran, and after
//!   the first 2048 packets the bitmap wrapped around (2049 % 2048 == 1)
//!   causing every legitimate packet to be misclassified as a replay.
//!
//!   Fix: Replace the sentinel with an explicit `has_seen_any: bool` flag.
//!   The first packet unconditionally initialises the window regardless of
//!   its counter value. All subsequent packets follow the normal sliding
//!   window rules.
//!
//!   Discovered via production diagnostic logs showing:
//!   `counter=N highest_before=18446744073709551615 result=Accept`
//!   for every packet — `highest_seen` was never being updated.
//!
//!   This bug affected every macOS/iOS VPN connection. Counter values
//!   1..=2047 would succeed, then the 2049th packet would be rejected
//!   (2049 % 2048 = 1, bit already set by counter=1). Connections appeared
//!   "connected" but all traffic was silently dropped on the server.
//!
//! ## Multi-Device Design (v1.2.0)
//! ```text
//! wallet_index: DashMap<[u8; 32], Vec<DeviceEntry>>
//!
//! One wallet can have multiple simultaneous active sessions (one per device).
//!
//! DeviceEntry { device_id: [u8; 16], session_id: SessionId }
//!
//! register_device(wallet, device_id, session_id):
//!   - Removes any stale entry with the same device_id (reconnect case).
//!   - Appends the new entry.
//!
//! remove_device(session_id):
//!   - Scans all wallet vectors, removes entries matching the session_id.
//!   - Removes the wallet key entirely if the vector becomes empty.
//!
//! get_by_wallet(wallet):
//!   - Returns the Arc<Session> for the LAST registered device (most recent).
//!   - Used for single-target routing (backward compatible).
//!
//! get_all_by_wallet(wallet):
//!   - Returns Arc<Session> for ALL active devices of a wallet.
//!   - Used by ChatRelayService for multi-device broadcast.
//! ```
//!
//! ## ⚠️ Important Notes for Next Developer
//! - `DeviceId` is `[u8; 16]` — generated once on install, persisted on device.
//! - `register_device` must be called from the server's `DeviceRegister` handler
//!   AFTER the session is already in `sessions` map (create() is called first).
//! - `remove_device` is called from `remove()` — do NOT call it separately.
//! - `get_all_by_wallet` returns a Vec; empty Vec means wallet is offline.
//! - The two-map update (sessions + wallet_index) is not strictly atomic.
//!   Chat routing tolerates a brief gap between them.
//! - Window size of 2048 allows ~100ms of reordering at 20k pps.
//! - 🔴 DO NOT revert `has_seen_any` back to a u64 sentinel. The sentinel
//!   approach is fundamentally incompatible with the `counter > highest_seen`
//!   comparison because any real counter value can be a valid "first packet".
//!
//! ## Last Modified
//! v1.2.1-ReplayWindowFix
//!   - `ReplayWindow::new()` no longer uses u64::MAX sentinel
//!   - Added `has_seen_any: bool` flag
//!   - `check_and_record()` first-packet branch initialises unconditionally
//!   - `window_base()` returns 0 when `has_seen_any == false`
//!   - Removed `test_replay_window_counter_zero_is_accept_and_advance`
//!     ambiguity (was testing the buggy sentinel path); replaced with an
//!     explicit "first packet with arbitrary counter" test.
//!   - All other v1.2.0 behaviour preserved verbatim.
//!
//! v1.2.0-MultiDevice
//!   - `wallet_index` type: `DashMap<[u8;32], SessionId>`
//!     → `DashMap<[u8;32], Vec<DeviceEntry>>`
//!   - Added `DeviceId` type alias
//!   - Added `DeviceEntry` struct
//!   - Added `register_device()`, `remove_device()` (replaces old wallet_index
//!     cleanup in `remove()`), `get_all_by_wallet()`
//!   - `get_by_wallet()` returns last entry (most recent device)
//!   - `wallet_index_count()` returns total device-entry count across all wallets
//!   - All existing tests preserved verbatim
//!   - New multi-device tests appended

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use tracing::{debug, info, trace};

use aeronyx_common::time::AtomicInstant;
use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::keys::{IdentityPublicKey, SessionKey};

use crate::error::{Result, ServerError};

// ============================================
// Replay Window Constants
// ============================================

const REPLAY_WINDOW_SIZE: u64 = 2048;
const BITMAP_WORDS: usize = (REPLAY_WINDOW_SIZE / 64) as usize;

// ============================================
// 🌟 v1.2.0-MultiDevice: DeviceId type + DeviceEntry
// ============================================

/// Stable random identifier for a physical device.
///
/// Generated once on first install, persisted in FlutterSecureStorage.
/// Survives VPN reconnects — the same phone always has the same DeviceId.
/// `[u8; 16]` matches the `message_id` and `cursor` field sizes used
/// elsewhere in the protocol for consistency.
pub type DeviceId = [u8; 16];

/// Associates one device with one active session.
///
/// Stored inside `wallet_index` Vec — one entry per connected device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceEntry {
    /// Stable device identifier (persisted on the client side).
    pub device_id: DeviceId,
    /// The current active SessionId for this device.
    pub session_id: SessionId,
}

// ============================================
// Replay Window (v1.2.1: sentinel bug fix)
// ============================================

/// Outcome of checking a packet counter against the replay window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayCheckResult {
    /// Counter is in-window and unseen — packet accepted, bit set.
    Accept,
    /// Counter is the new highest — packet accepted, window advanced.
    AcceptAndAdvance,
    /// Counter is in-window but already seen — reject as replay.
    Replay,
    /// Counter is older than the window base — reject as too old.
    TooOld,
}

/// WireGuard-style sliding replay window.
///
/// # Window Semantics
/// - Window size: 2048 packets.
/// - Accepts any counter `c` where `c > highest_seen`
///   (sets new highest, clears stale bits).
/// - Accepts any counter `c` where `window_base <= c <= highest_seen`
///   and the bitmap bit for `c` is zero (marks bit as seen).
/// - Rejects `c < window_base` as `TooOld`.
/// - Rejects `c` with an already-set bit as `Replay`.
///
/// # First Packet Handling (v1.2.1)
/// The first packet after construction is always accepted unconditionally
/// via the `has_seen_any` flag. This is critical: the previous
/// implementation used `u64::MAX` as a "no packets seen" sentinel, which
/// broke the entire algorithm because `counter > u64::MAX` is never true.
pub struct ReplayWindow {
    /// Highest counter value seen so far. Meaningful only when
    /// `has_seen_any == true`.
    highest_seen: u64,

    /// 🔴 v1.2.1: Explicit flag indicating whether any packet has been
    /// accepted yet. Replaces the broken `u64::MAX` sentinel.
    has_seen_any: bool,

    /// Bitmap tracking which counters within the current window have been
    /// seen. Each bit position is `counter % REPLAY_WINDOW_SIZE`.
    bitmap: [u64; BITMAP_WORDS],
}

impl ReplayWindow {
    /// Creates a new, empty replay window.
    #[must_use]
    pub fn new() -> Self {
        Self {
            highest_seen: 0,
            has_seen_any: false,
            bitmap: [0u64; BITMAP_WORDS],
        }
    }

    /// Checks whether `counter` is acceptable and, if so, records it.
    ///
    /// Returns the classification — the caller is responsible for
    /// translating `Accept`/`AcceptAndAdvance` into "packet allowed" and
    /// `Replay`/`TooOld` into "packet rejected".
    pub fn check_and_record(&mut self, counter: u64) -> ReplayCheckResult {
        // 🔴 v1.2.1 First-packet fast path.
        //
        // The first packet ever seen on this session initialises the
        // window. Its counter value is arbitrary (clients may start from
        // 0, 1, or any other value depending on implementation history),
        // so we unconditionally accept and advance here.
        if !self.has_seen_any {
            self.has_seen_any = true;
            self.highest_seen = counter;
            self.set_bit(counter);
            return ReplayCheckResult::AcceptAndAdvance;
        }

        // ── Normal path (has_seen_any == true) ──────────────────────────

        if counter > self.highest_seen {
            // New highest counter — advance window and clear stale bits.
            let advance = counter - self.highest_seen;
            if advance >= REPLAY_WINDOW_SIZE {
                // Jump larger than window: clear everything.
                self.bitmap = [0u64; BITMAP_WORDS];
            } else {
                // Clear bits for positions that are now outside the window
                // (counters between old highest+1 and new highest, exclusive
                // of the new highest itself — that one gets set below).
                for i in 1..=advance {
                    let old_counter = self.highest_seen + i;
                    self.clear_bit(old_counter);
                }
            }
            self.highest_seen = counter;
            self.set_bit(counter);
            return ReplayCheckResult::AcceptAndAdvance;
        }

        // counter <= highest_seen — could be in-window retransmit,
        // a genuine replay, or a too-old straggler.

        let window_base = if self.highest_seen >= REPLAY_WINDOW_SIZE - 1 {
            self.highest_seen - REPLAY_WINDOW_SIZE + 1
        } else {
            0
        };

        if counter < window_base {
            return ReplayCheckResult::TooOld;
        }
        if self.get_bit(counter) {
            return ReplayCheckResult::Replay;
        }
        self.set_bit(counter);
        ReplayCheckResult::Accept
    }

    /// Checks whether the bitmap bit for `counter` is set.
    #[inline]
    fn get_bit(&self, counter: u64) -> bool {
        let bit_index = (counter % REPLAY_WINDOW_SIZE) as usize;
        let word_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        (self.bitmap[word_index] & (1u64 << bit_offset)) != 0
    }

    /// Sets the bitmap bit for `counter`.
    #[inline]
    fn set_bit(&mut self, counter: u64) {
        let bit_index = (counter % REPLAY_WINDOW_SIZE) as usize;
        let word_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        self.bitmap[word_index] |= 1u64 << bit_offset;
    }

    /// Clears the bitmap bit for `counter`.
    #[inline]
    fn clear_bit(&mut self, counter: u64) {
        let bit_index = (counter % REPLAY_WINDOW_SIZE) as usize;
        let word_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        self.bitmap[word_index] &= !(1u64 << bit_offset);
    }

    /// Returns the highest counter value accepted so far.
    /// Meaningful only when at least one packet has been accepted.
    #[must_use]
    pub fn highest_seen(&self) -> u64 {
        self.highest_seen
    }

    /// Returns the lower bound (inclusive) of the current window.
    /// Returns `0` when no packets have been seen yet.
    #[must_use]
    pub fn window_base(&self) -> u64 {
        if !self.has_seen_any {
            return 0;
        }
        if self.highest_seen >= REPLAY_WINDOW_SIZE - 1 {
            self.highest_seen - REPLAY_WINDOW_SIZE + 1
        } else {
            0
        }
    }

    /// Returns true if at least one packet has been accepted.
    /// Primarily useful for tests and diagnostics.
    #[must_use]
    pub fn has_seen_any(&self) -> bool {
        self.has_seen_any
    }
}

impl Default for ReplayWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ReplayWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplayWindow")
            .field("has_seen_any", &self.has_seen_any)
            .field("highest_seen", &self.highest_seen)
            .field("window_base", &self.window_base())
            .finish()
    }
}

// ============================================
// Session State (unchanged)
// ============================================

/// Lifecycle state of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is active and processing traffic.
    Established,
    /// Session is in the process of closing.
    Closing,
    /// Session has been closed.
    Closed,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Established => write!(f, "Established"),
            Self::Closing => write!(f, "Closing"),
            Self::Closed => write!(f, "Closed"),
        }
    }
}

// ============================================
// Session Statistics (unchanged)
// ============================================

/// Per-session packet/byte counters and rejection stats.
#[derive(Debug, Default)]
pub struct SessionStats {
    /// Total bytes received from the client (decrypted payload length).
    pub bytes_rx: AtomicU64,
    /// Total bytes transmitted to the client (plaintext length before encryption).
    pub bytes_tx: AtomicU64,
    /// Total packets received from the client.
    pub packets_rx: AtomicU64,
    /// Total packets transmitted to the client.
    pub packets_tx: AtomicU64,
    /// Packets rejected because the counter had already been seen.
    pub replays_rejected: AtomicU64,
    /// Packets rejected because the counter was below the window base.
    pub too_old_rejected: AtomicU64,
}

impl SessionStats {
    /// Records a successful receive of `bytes` bytes.
    pub fn record_rx(&self, bytes: u64) {
        self.bytes_rx.fetch_add(bytes, Ordering::Relaxed);
        self.packets_rx.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a successful transmit of `bytes` bytes.
    pub fn record_tx(&self, bytes: u64) {
        self.bytes_tx.fetch_add(bytes, Ordering::Relaxed);
        self.packets_tx.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the replay-rejection counter.
    pub fn record_replay_rejected(&self) {
        self.replays_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Increments the too-old-rejection counter.
    pub fn record_too_old_rejected(&self) {
        self.too_old_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns a point-in-time snapshot of all counters.
    #[must_use]
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            bytes_rx: self.bytes_rx.load(Ordering::Relaxed),
            bytes_tx: self.bytes_tx.load(Ordering::Relaxed),
            packets_rx: self.packets_rx.load(Ordering::Relaxed),
            packets_tx: self.packets_tx.load(Ordering::Relaxed),
            replays_rejected: self.replays_rejected.load(Ordering::Relaxed),
            too_old_rejected: self.too_old_rejected.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of [`SessionStats`].
#[derive(Debug, Clone, Copy)]
pub struct StatsSnapshot {
    /// Bytes received.
    pub bytes_rx: u64,
    /// Bytes transmitted.
    pub bytes_tx: u64,
    /// Packets received.
    pub packets_rx: u64,
    /// Packets transmitted.
    pub packets_tx: u64,
    /// Packets rejected as replays.
    pub replays_rejected: u64,
    /// Packets rejected as too old.
    pub too_old_rejected: u64,
}

// ============================================
// Session (unchanged)
// ============================================

/// A single authenticated client session.
pub struct Session {
    /// Unique identifier assigned at handshake time.
    pub id: SessionId,
    /// Current lifecycle state.
    state: RwLock<SessionState>,
    /// Client's long-term Ed25519 identity public key.
    pub client_public_key: IdentityPublicKey,
    /// Derived symmetric session key for XChaCha20-Poly1305.
    pub session_key: SessionKey,
    /// Virtual IPv4 address assigned to this client inside the tunnel.
    pub virtual_ip: Ipv4Addr,
    /// Client's UDP endpoint (source address at handshake time).
    pub client_endpoint: SocketAddr,
    /// Time the session was created.
    pub created_at: std::time::Instant,
    /// Time of the most recent activity, used for idle-timeout cleanup.
    pub last_activity: AtomicInstant,
    /// Monotonic counter used for outbound (server → client) packets.
    pub tx_counter: AtomicU64,
    /// Replay window guarding inbound (client → server) packets.
    replay_window: Mutex<ReplayWindow>,
    /// Last accepted inbound counter (mirrors replay window's highest_seen).
    pub rx_counter: AtomicU64,
    /// Per-session traffic and rejection statistics.
    pub stats: SessionStats,
}

impl Session {
    /// Creates a new established session.
    #[must_use]
    pub fn new(
        id: SessionId,
        client_public_key: IdentityPublicKey,
        session_key: SessionKey,
        virtual_ip: Ipv4Addr,
        client_endpoint: SocketAddr,
    ) -> Self {
        let now = std::time::Instant::now();
        Self {
            id,
            state: RwLock::new(SessionState::Established),
            client_public_key,
            session_key,
            virtual_ip,
            client_endpoint,
            created_at: now,
            last_activity: AtomicInstant::from_instant(now),
            tx_counter: AtomicU64::new(0),
            replay_window: Mutex::new(ReplayWindow::new()),
            rx_counter: AtomicU64::new(0),
            stats: SessionStats::default(),
        }
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        *self.state.read()
    }

    /// Sets the lifecycle state.
    pub fn set_state(&self, state: SessionState) {
        *self.state.write() = state;
    }

    /// Returns true if the session is in the `Established` state.
    #[must_use]
    pub fn is_established(&self) -> bool {
        self.state() == SessionState::Established
    }

    /// Updates `last_activity` to the current instant.
    pub fn touch(&self) {
        self.last_activity.store(std::time::Instant::now());
    }

    /// Returns the time elapsed since the most recent activity.
    #[must_use]
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Returns true if the session has been idle longer than `timeout`.
    #[must_use]
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.idle_time() > timeout
    }

    /// Atomically returns the next outbound counter value.
    #[must_use]
    pub fn next_tx_counter(&self) -> u64 {
        self.tx_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Validates an inbound counter against the replay window.
    ///
    /// Returns `true` if the packet should be accepted (either a brand-new
    /// highest counter or an unseen in-window retransmit), and `false`
    /// otherwise (replay or too-old). Rejection statistics are updated
    /// internally.
    pub fn validate_rx_counter(&self, counter: u64) -> bool {
        let mut window = self.replay_window.lock();
        let result = window.check_and_record(counter);
        if matches!(result, ReplayCheckResult::AcceptAndAdvance) {
            self.rx_counter.store(counter, Ordering::SeqCst);
        }
        match result {
            ReplayCheckResult::Accept | ReplayCheckResult::AcceptAndAdvance => {
                trace!(
                    session_id = %self.id,
                    counter,
                    highest = window.highest_seen(),
                    "Counter accepted"
                );
                true
            }
            ReplayCheckResult::Replay => {
                self.stats.record_replay_rejected();
                debug!(
                    session_id = %self.id,
                    counter,
                    highest = window.highest_seen(),
                    "Replay detected"
                );
                false
            }
            ReplayCheckResult::TooOld => {
                self.stats.record_too_old_rejected();
                debug!(
                    session_id = %self.id,
                    counter,
                    window_base = window.window_base(),
                    "Counter too old"
                );
                false
            }
        }
    }

    /// Returns `(window_base, highest_seen)` for diagnostics.
    #[must_use]
    pub fn replay_window_info(&self) -> (u64, u64) {
        let window = self.replay_window.lock();
        (window.window_base(), window.highest_seen())
    }

    /// Returns the client's wallet (identity) public key as raw bytes.
    #[must_use]
    pub fn wallet_bytes(&self) -> [u8; 32] {
        self.client_public_key.to_bytes()
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (window_base, highest_seen) = self.replay_window_info();
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("state", &self.state())
            .field("virtual_ip", &self.virtual_ip)
            .field("client_endpoint", &self.client_endpoint)
            .field("idle_time", &self.idle_time())
            .field(
                "replay_window",
                &format!("[{}..{}]", window_base, highest_seen),
            )
            .finish_non_exhaustive()
    }
}

// ============================================
// Session Manager — v1.2.0-MultiDevice
// ============================================

/// Manages all active sessions with multi-device wallet reverse lookup.
///
/// ## v1.2.0-MultiDevice Change
/// `wallet_index` maps `[u8; 32]` wallet bytes to `Vec<DeviceEntry>`,
/// where each `DeviceEntry` holds a `(DeviceId, SessionId)` pair.
///
/// This allows `ChatRelayService` to broadcast to ALL active devices of
/// a given wallet via `get_all_by_wallet()`.
///
/// ### Invariants
/// - `sessions` is the source of truth. `wallet_index` is a secondary index.
/// - `create()` inserts only into `sessions`. Callers MUST call
///   `register_device()` separately after the client sends `DeviceRegister`.
/// - `remove()` calls `remove_device_by_session()` internally — do NOT
///   call it externally.
/// - `get_by_wallet()` returns the LAST registered device (most recent).
///   Use `get_all_by_wallet()` for broadcast.
pub struct SessionManager {
    sessions: DashMap<SessionId, Arc<Session>>,

    // 🌟 v1.2.0-MultiDevice: wallet → Vec<DeviceEntry>
    //
    // Vec order = registration order. Last entry = most recently registered device.
    // Max entries per wallet is bounded by max_sessions (global limit).
    wallet_index: DashMap<[u8; 32], Vec<DeviceEntry>>,

    max_sessions: usize,
    session_timeout: Duration,
}

impl SessionManager {
    /// Creates a new session manager with the given capacity and idle timeout.
    #[must_use]
    pub fn new(max_sessions: usize, session_timeout: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            wallet_index: DashMap::new(),
            max_sessions,
            session_timeout,
        }
    }

    /// Creates and registers a new session.
    ///
    /// Does NOT insert into `wallet_index`. The caller must wait for the
    /// client to send a `DeviceRegister` message, then call `register_device()`.
    ///
    /// # Errors
    /// Returns `SessionLimitReached` if max sessions exceeded.
    pub fn create(
        &self,
        session_id: SessionId,
        client_public_key: IdentityPublicKey,
        session_key: SessionKey,
        virtual_ip: Ipv4Addr,
        client_endpoint: SocketAddr,
    ) -> Result<Arc<Session>> {
        if self.sessions.len() >= self.max_sessions {
            return Err(ServerError::SessionLimitReached {
                limit: self.max_sessions,
            });
        }

        let session = Arc::new(Session::new(
            session_id.clone(),
            client_public_key,
            session_key,
            virtual_ip,
            client_endpoint,
        ));

        self.sessions.insert(session_id.clone(), Arc::clone(&session));

        info!(
            session_id = %session_id,
            virtual_ip = %virtual_ip,
            client = %client_endpoint,
            "Session created (device registration pending)"
        );

        Ok(session)
    }

    /// 🌟 v1.2.0-MultiDevice: Register a device under its wallet.
    ///
    /// Called when the node receives a `DeviceRegister` message from the client.
    /// The session MUST already exist in `sessions` (created by `create()`).
    ///
    /// If the same `device_id` already has an entry (same device reconnect),
    /// the old entry is replaced with the new `session_id`.
    ///
    /// # Arguments
    /// * `wallet` - The wallet public key bytes (from verified session)
    /// * `device_id` - Stable device identifier from the client
    /// * `session_id` - The current active session for this device
    pub fn register_device(
        &self,
        wallet: &[u8; 32],
        device_id: DeviceId,
        session_id: SessionId,
    ) {
        let mut entry = self.wallet_index.entry(*wallet).or_default();

        // Remove stale entry for the same device_id (reconnect case).
        entry.retain(|e| e.device_id != device_id);

        // Append the new entry (becomes the "most recent" device).
        entry.push(DeviceEntry {
            device_id,
            session_id: session_id.clone(),
        });

        debug!(
            wallet = %hex::encode(&wallet[..4]),
            device_id = %hex::encode(device_id),
            session_id = %session_id,
            device_count = entry.len(),
            "Device registered"
        );
    }

    /// Returns the session with the given id, if any.
    #[must_use]
    pub fn get(&self, id: &SessionId) -> Option<Arc<Session>> {
        self.sessions.get(id).map(|r| Arc::clone(r.value()))
    }

    /// Returns the session with the given id, or `SessionNotFound`.
    pub fn get_or_error(&self, id: &SessionId) -> Result<Arc<Session>> {
        self.get(id)
            .ok_or_else(|| ServerError::SessionNotFound(id.clone()))
    }

    /// 🌟 v1.2.0-MultiDevice: Look up the most recently registered session
    /// for a wallet (backward-compatible single-device routing).
    ///
    /// Returns `None` if the wallet has no active devices.
    #[must_use]
    pub fn get_by_wallet(&self, wallet: &[u8; 32]) -> Option<Arc<Session>> {
        self.wallet_index.get(wallet).and_then(|entries| {
            // Last entry = most recently registered device.
            entries.last().and_then(|e| {
                self.sessions
                    .get(&e.session_id)
                    .map(|s| Arc::clone(s.value()))
            })
        })
    }

    /// 🌟 v1.2.0-MultiDevice: Return ALL active sessions for a wallet.
    ///
    /// Used by `ChatRelayService` to broadcast a message to every device
    /// the wallet currently has connected. Returns an empty Vec if the
    /// wallet is fully offline.
    #[must_use]
    pub fn get_all_by_wallet(&self, wallet: &[u8; 32]) -> Vec<Arc<Session>> {
        let Some(entries) = self.wallet_index.get(wallet) else {
            return Vec::new();
        };

        entries
            .iter()
            .filter_map(|e| {
                self.sessions
                    .get(&e.session_id)
                    .map(|s| Arc::clone(s.value()))
            })
            .collect()
    }

    /// Remove a session and clean up its wallet_index entry.
    ///
    /// Calls `remove_device_by_session()` internally to keep
    /// `wallet_index` consistent.
    pub fn remove(&self, id: &SessionId) -> Option<Arc<Session>> {
        let removed = self.sessions.remove(id).map(|(_, s)| s);

        if let Some(ref session) = removed {
            session.set_state(SessionState::Closed);
            self.remove_device_by_session(id, &session.wallet_bytes());

            let stats = session.stats.snapshot();
            info!(
                session_id = %id,
                virtual_ip = %session.virtual_ip,
                packets_rx = stats.packets_rx,
                packets_tx = stats.packets_tx,
                replays_rejected = stats.replays_rejected,
                "Session removed"
            );
        }

        removed
    }

    /// 🌟 v1.2.0-MultiDevice: Remove all wallet_index entries for a session.
    ///
    /// Scans the wallet's Vec and removes any `DeviceEntry` whose
    /// `session_id` matches `id`. If the Vec becomes empty, the wallet
    /// key is removed entirely.
    ///
    /// Called internally by `remove()` — do NOT call externally.
    fn remove_device_by_session(&self, id: &SessionId, wallet: &[u8; 32]) {
        let mut should_remove_wallet = false;

        if let Some(mut entries) = self.wallet_index.get_mut(wallet) {
            entries.retain(|e| &e.session_id != id);
            if entries.is_empty() {
                should_remove_wallet = true;
            } else {
                debug!(
                    session_id = %id,
                    wallet = %hex::encode(&wallet[..4]),
                    remaining_devices = entries.len(),
                    "Device unregistered (other devices still active)"
                );
            }
        }

        if should_remove_wallet {
            self.wallet_index.remove(wallet);
            debug!(
                session_id = %id,
                wallet = %hex::encode(&wallet[..4]),
                "wallet_index entry removed (no more devices)"
            );
        }
    }

    /// Marks a session as closing and then removes it.
    pub fn close(&self, id: &SessionId) {
        if let Some(session) = self.get(id) {
            session.set_state(SessionState::Closing);
        }
        self.remove(id);
    }

    /// Returns the number of currently active sessions.
    #[must_use]
    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns true if there are no active sessions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Clean up expired sessions and their wallet_index entries.
    pub fn cleanup_expired(&self) -> Vec<(SessionId, Ipv4Addr)> {
        let mut expired = Vec::new();

        for entry in self.sessions.iter() {
            let session = entry.value();
            if session.is_expired(self.session_timeout) {
                expired.push((session.id.clone(), session.virtual_ip));
            }
        }

        for (id, ip) in &expired {
            debug!(session_id = %id, virtual_ip = %ip, "Session expired");
            self.remove(id);
        }

        if !expired.is_empty() {
            info!(
                count = expired.len(),
                wallet_index_wallets = self.wallet_index.len(),
                "Cleaned up expired sessions"
            );
        }

        expired
    }

    /// Returns an owned snapshot of all active sessions.
    #[must_use]
    pub fn all_sessions(&self) -> Vec<Arc<Session>> {
        self.sessions
            .iter()
            .map(|r| Arc::clone(r.value()))
            .collect()
    }

    /// Updates `last_activity` for the session with the given id (if it exists).
    pub fn touch(&self, id: &SessionId) {
        if let Some(session) = self.get(id) {
            session.touch();
        }
    }

    /// Returns total number of device entries across all wallets.
    ///
    /// In normal operation this equals `count()`.
    /// A discrepancy indicates stale entries (bug in cleanup logic).
    #[must_use]
    pub fn wallet_index_count(&self) -> usize {
        self.wallet_index.iter().map(|e| e.value().len()).sum()
    }

    /// Returns the number of distinct wallets in the index.
    #[must_use]
    pub fn wallet_count(&self) -> usize {
        self.wallet_index.len()
    }
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager")
            .field("sessions", &self.count())
            .field("wallet_count", &self.wallet_count())
            .field("device_entries", &self.wallet_index_count())
            .field("max_sessions", &self.max_sessions)
            .field("session_timeout", &self.session_timeout)
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;

    // ── ReplayWindow Tests ───────────────────────────────────────────────

    #[test]
    fn test_replay_window_sequential() {
        let mut window = ReplayWindow::new();
        for i in 1..=100 {
            let result = window.check_and_record(i);
            assert!(
                matches!(result, ReplayCheckResult::AcceptAndAdvance),
                "Counter {} should be accepted",
                i
            );
        }
        assert_eq!(window.highest_seen(), 100);
    }

    #[test]
    fn test_replay_window_replay_detection() {
        let mut window = ReplayWindow::new();
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::Replay
        ));
    }

    #[test]
    fn test_replay_window_out_of_order() {
        let mut window = ReplayWindow::new();
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert!(matches!(
            window.check_and_record(95),
            ReplayCheckResult::Accept
        ));
        assert!(matches!(
            window.check_and_record(99),
            ReplayCheckResult::Accept
        ));
        assert!(matches!(
            window.check_and_record(95),
            ReplayCheckResult::Replay
        ));
    }

    #[test]
    fn test_replay_window_too_old() {
        let mut window = ReplayWindow::new();
        window.check_and_record(3000);
        assert!(matches!(
            window.check_and_record(500),
            ReplayCheckResult::TooOld
        ));
        assert!(matches!(
            window.check_and_record(953),
            ReplayCheckResult::Accept
        ));
    }

    /// 🔴 v1.2.1: First packet is accepted regardless of its counter value.
    /// This test replaces the ambiguous `test_replay_window_counter_zero_is_accept_and_advance`
    /// which was silently testing the buggy u64::MAX sentinel path.
    #[test]
    fn test_replay_window_first_packet_accepted_with_any_counter() {
        // counter=0
        let mut w = ReplayWindow::new();
        assert!(matches!(
            w.check_and_record(0),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert_eq!(w.highest_seen(), 0);
        assert!(w.has_seen_any());

        // counter=1
        let mut w = ReplayWindow::new();
        assert!(matches!(
            w.check_and_record(1),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert_eq!(w.highest_seen(), 1);

        // counter=2049 (the historically problematic value)
        let mut w = ReplayWindow::new();
        assert!(matches!(
            w.check_and_record(2049),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert_eq!(w.highest_seen(), 2049);

        // Large counter
        let mut w = ReplayWindow::new();
        assert!(matches!(
            w.check_and_record(u64::MAX - 1),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert_eq!(w.highest_seen(), u64::MAX - 1);
    }

    /// 🔴 v1.2.1 regression test: the original bug manifested as every
    /// packet appearing to be accepted (result=Accept) but `highest_seen`
    /// never advancing past the sentinel value. The bitmap would then
    /// wrap around at counter 2049 and falsely report replays.
    #[test]
    fn test_replay_window_highest_seen_advances_from_first_packet() {
        let mut w = ReplayWindow::new();

        // First packet must establish highest_seen.
        assert!(!w.has_seen_any());
        w.check_and_record(1);
        assert!(w.has_seen_any());
        assert_eq!(w.highest_seen(), 1);

        // Subsequent packets must keep advancing it.
        w.check_and_record(2);
        assert_eq!(w.highest_seen(), 2);

        // Run through 3000 sequential counters and verify every single one
        // is AcceptAndAdvance — this is the exact scenario that was
        // previously failing at counter 2049.
        for c in 3..=3000u64 {
            let result = w.check_and_record(c);
            assert!(
                matches!(result, ReplayCheckResult::AcceptAndAdvance),
                "counter {} must be AcceptAndAdvance, got {:?}",
                c,
                result
            );
        }
        assert_eq!(w.highest_seen(), 3000);
    }

    #[test]
    fn test_replay_window_large_jump() {
        let mut window = ReplayWindow::new();
        window.check_and_record(100);
        assert!(matches!(
            window.check_and_record(10000),
            ReplayCheckResult::AcceptAndAdvance
        ));
        assert_eq!(window.highest_seen(), 10000);
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::TooOld
        ));
    }

    #[test]
    fn test_replay_window_stress() {
        let mut window = ReplayWindow::new();
        let mut counters: Vec<u64> = (1..=10000).collect();
        for i in (0..counters.len() - 1).step_by(10) {
            counters.swap(i, i + 1);
        }
        let mut accepted = 0u64;
        let mut rejected = 0u64;
        for counter in counters {
            match window.check_and_record(counter) {
                ReplayCheckResult::Accept | ReplayCheckResult::AcceptAndAdvance => accepted += 1,
                _ => rejected += 1,
            }
        }
        assert_eq!(accepted, 10000);
        assert_eq!(rejected, 0);
    }

    // ── Session Tests (preserved verbatim) ───────────────────────────────

    fn create_test_session() -> Session {
        let identity = IdentityKeyPair::generate();
        Session::new(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        )
    }

    #[test]
    fn test_session_creation() {
        let session = create_test_session();
        assert!(session.is_established());
    }

    #[test]
    fn test_session_tx_counters() {
        let session = create_test_session();
        assert_eq!(session.next_tx_counter(), 0);
        assert_eq!(session.next_tx_counter(), 1);
        assert_eq!(session.next_tx_counter(), 2);
    }

    #[test]
    fn test_session_rx_counter_validation() {
        let session = create_test_session();
        assert!(session.validate_rx_counter(1));
        assert!(session.validate_rx_counter(2));
        assert!(session.validate_rx_counter(3));
        assert!(!session.validate_rx_counter(2));
        assert!(session.validate_rx_counter(5));
        assert!(session.validate_rx_counter(4));
        assert!(!session.validate_rx_counter(4));
    }

    #[test]
    fn test_session_rx_counter_out_of_order() {
        let session = create_test_session();
        assert!(session.validate_rx_counter(100));
        assert!(session.validate_rx_counter(95));
        assert!(session.validate_rx_counter(96));
        assert!(!session.validate_rx_counter(95));
        assert!(!session.validate_rx_counter(100));
    }

    #[test]
    fn test_session_wallet_bytes() {
        let identity = IdentityKeyPair::generate();
        let session = Session::new(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        );
        assert_eq!(session.wallet_bytes(), identity.public_key_bytes());
    }

    // ── SessionManager Tests (preserved verbatim) ────────────────────────

    fn make_manager() -> SessionManager {
        SessionManager::new(100, Duration::from_secs(300))
    }

    #[test]
    fn test_session_manager_create() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let session_id = SessionId::generate();

        let session = manager
            .create(
                session_id.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:12345".parse().unwrap(),
            )
            .unwrap();

        assert_eq!(manager.count(), 1);
        assert_eq!(session.id, session_id);

        let retrieved = manager.get(&session.id).unwrap();
        assert_eq!(retrieved.virtual_ip, Ipv4Addr::new(100, 64, 0, 2));
    }

    #[test]
    fn test_session_manager_limit() {
        let manager = SessionManager::new(2, Duration::from_secs(300));
        let identity = IdentityKeyPair::generate();

        manager
            .create(
                SessionId::generate(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:12345".parse().unwrap(),
            )
            .unwrap();
        manager
            .create(
                SessionId::generate(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 3),
                "127.0.0.1:12346".parse().unwrap(),
            )
            .unwrap();

        let result = manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 4),
            "127.0.0.1:12347".parse().unwrap(),
        );
        assert!(matches!(
            result,
            Err(ServerError::SessionLimitReached { .. })
        ));
    }

    #[test]
    fn test_session_stats_tracking() {
        let session = create_test_session();
        session.validate_rx_counter(100);
        session.validate_rx_counter(100);
        session.validate_rx_counter(100);
        let stats = session.stats.snapshot();
        assert_eq!(stats.replays_rejected, 2);
    }

    // ── 🌟 v1.2.0-MultiDevice: wallet_index Tests ───────────────────────

    fn make_device_id(seed: u8) -> DeviceId {
        [seed; 16]
    }

    /// create() no longer inserts into wallet_index.
    /// wallet_index is only populated by register_device().
    #[test]
    fn test_create_does_not_populate_wallet_index() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();

        manager
            .create(
                SessionId::generate(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:12345".parse().unwrap(),
            )
            .unwrap();

        // wallet_index must be empty until register_device is called
        assert_eq!(manager.wallet_index_count(), 0);
        assert!(manager.get_by_wallet(&identity.public_key_bytes()).is_none());
    }

    #[test]
    fn test_register_device_and_get_by_wallet() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();
        let sid = SessionId::generate();

        manager
            .create(
                sid.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:12345".parse().unwrap(),
            )
            .unwrap();

        manager.register_device(&wallet, make_device_id(0x01), sid.clone());

        let found = manager.get_by_wallet(&wallet);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, sid);
        assert_eq!(manager.wallet_index_count(), 1);
    }

    #[test]
    fn test_get_all_by_wallet_multiple_devices() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();

        // Device A
        let sid_a = SessionId::generate();
        manager
            .create(
                sid_a.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x01; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:11111".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, make_device_id(0xAA), sid_a.clone());

        // Device B (same wallet)
        let sid_b = SessionId::generate();
        manager
            .create(
                sid_b.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x02; 32]),
                Ipv4Addr::new(100, 64, 0, 3),
                "127.0.0.1:22222".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, make_device_id(0xBB), sid_b.clone());

        let all = manager.get_all_by_wallet(&wallet);
        assert_eq!(all.len(), 2, "Both devices must be returned");

        let ids: Vec<_> = all.iter().map(|s| s.id.clone()).collect();
        assert!(ids.contains(&sid_a));
        assert!(ids.contains(&sid_b));

        // get_by_wallet returns the most recently registered (sid_b)
        let last = manager.get_by_wallet(&wallet).unwrap();
        assert_eq!(
            last.id, sid_b,
            "get_by_wallet must return most recent device"
        );

        assert_eq!(manager.wallet_index_count(), 2);
        assert_eq!(manager.wallet_count(), 1);
    }

    #[test]
    fn test_remove_one_device_leaves_other() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();

        let sid_a = SessionId::generate();
        manager
            .create(
                sid_a.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x01; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:11111".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, make_device_id(0xAA), sid_a.clone());

        let sid_b = SessionId::generate();
        manager
            .create(
                sid_b.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x02; 32]),
                Ipv4Addr::new(100, 64, 0, 3),
                "127.0.0.1:22222".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, make_device_id(0xBB), sid_b.clone());

        // Remove device A
        manager.remove(&sid_a);

        // Device B must still be reachable
        let all = manager.get_all_by_wallet(&wallet);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, sid_b);
        assert_eq!(manager.wallet_index_count(), 1);
        assert_eq!(manager.wallet_count(), 1);
    }

    #[test]
    fn test_remove_last_device_clears_wallet_index() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();
        let sid = SessionId::generate();

        manager
            .create(
                sid.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:12345".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, make_device_id(0x01), sid.clone());

        manager.remove(&sid);

        assert!(manager.get_by_wallet(&wallet).is_none());
        assert_eq!(manager.wallet_index_count(), 0);
        assert_eq!(manager.wallet_count(), 0);
    }

    #[test]
    fn test_same_device_reconnect_replaces_session() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();
        let device_id = make_device_id(0x42);

        // First connection
        let sid_old = SessionId::generate();
        manager
            .create(
                sid_old.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x01; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:11111".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, device_id, sid_old.clone());
        assert_eq!(manager.wallet_index_count(), 1);

        // Same device reconnects with new session
        let sid_new = SessionId::generate();
        manager
            .create(
                sid_new.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x02; 32]),
                Ipv4Addr::new(100, 64, 0, 3),
                "127.0.0.1:22222".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, device_id, sid_new.clone());

        // Still only one entry (old replaced, not accumulated)
        assert_eq!(manager.wallet_index_count(), 1);
        let found = manager.get_by_wallet(&wallet).unwrap();
        assert_eq!(found.id, sid_new);
    }

    #[test]
    fn test_get_all_by_wallet_offline_returns_empty() {
        let manager = make_manager();
        let wallet = [0xFFu8; 32];
        assert!(manager.get_all_by_wallet(&wallet).is_empty());
    }

    #[test]
    fn test_cleanup_expired_removes_device_entries() {
        let manager = SessionManager::new(100, Duration::from_millis(1));
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();
        let sid = SessionId::generate();

        manager
            .create(
                sid.clone(),
                identity.public_key(),
                SessionKey::from_bytes([0x42; 32]),
                Ipv4Addr::new(100, 64, 0, 2),
                "127.0.0.1:12345".parse().unwrap(),
            )
            .unwrap();
        manager.register_device(&wallet, make_device_id(0x01), sid);

        std::thread::sleep(Duration::from_millis(10));
        manager.cleanup_expired();

        assert!(manager.get_all_by_wallet(&wallet).is_empty());
        assert_eq!(manager.wallet_index_count(), 0);
    }

    #[test]
    fn test_wallet_index_count_matches_total_devices() {
        let manager = make_manager();

        for i in 0u8..3 {
            let identity = IdentityKeyPair::generate();
            let wallet = identity.public_key_bytes();

            // Register 2 devices per wallet
            for j in 0u8..2 {
                let sid = SessionId::generate();
                manager
                    .create(
                        sid.clone(),
                        identity.public_key(),
                        SessionKey::from_bytes([i * 10 + j; 32]),
                        Ipv4Addr::new(100, 64, 0, (i * 10 + j + 2) as u32),
                        format!("127.0.0.1:{}", 10000u16 + (i as u16) * 10 + j as u16)
                            .parse()
                            .unwrap(),
                    )
                    .unwrap();
                manager.register_device(&wallet, make_device_id(i * 10 + j), sid);
            }
        }

        assert_eq!(manager.count(), 6); // 3 wallets × 2 devices
        assert_eq!(manager.wallet_count(), 3); // 3 distinct wallets
        assert_eq!(manager.wallet_index_count(), 6); // 6 device entries total
    }
}
