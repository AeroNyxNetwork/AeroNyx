// ============================================
// File: crates/aeronyx-server/src/services/session.rs
// ============================================
//! # Session Management Service
//!
//! ## Creation Reason
//! Manages the lifecycle of client sessions, including creation,
//! lookup, timeout handling, and cleanup.
//!
//! ## Modification Reason
//! - Fixed replay detection to use sliding window algorithm instead of
//!   strict increment. UDP packets can arrive out-of-order, so we need
//!   to tolerate reordering within a window while still detecting replays.
//! - 🌟 v1.1.0-ChatRelay: Added `wallet_index` reverse lookup to
//!   `SessionManager` for zero-knowledge chat relay. The index maps
//!   `wallet_bytes([u8; 32]) → SessionId`, allowing `ChatRelayService`
//!   to find the active session for a receiver wallet in O(1) without
//!   scanning all sessions.
//!
//! ## Main Functionality
//! - `Session`: Session data structure
//! - `SessionManager`: Session lifecycle management
//! - `SessionState`: Session state machine
//! - `ReplayWindow`: Sliding window anti-replay (NEW)
//! - Thread-safe concurrent access
//!
//! ## Session Lifecycle
//! ```text
//! ┌──────────┐     handshake      ┌─────────────┐
//! │  (none)  │ ─────────────────► │ Established │
//! └──────────┘                    └──────┬──────┘
//!                                        │
//!                      ┌─────────────────┼─────────────────┐
//!                      │                 │                 │
//!                      ▼                 ▼                 ▼
//!                  timeout           explicit          error
//!                      │              close              │
//!                      │                 │               │
//!                      └────────┬────────┴───────────────┘
//!                               │
//!                               ▼
//!                         ┌──────────┐
//!                         │  Closed  │
//!                         └──────────┘
//! ```
//!
//! ## Replay Detection Algorithm (Sliding Window)
//! ```text
//! Window size: 2048 packets
//!
//!     ◄─────────── WINDOW_SIZE ───────────►
//!     ┌─────────────────────────────────────┐
//!     │  bitmap (2048 bits = 256 bytes)     │
//!     └─────────────────────────────────────┘
//!     ▲                                     ▲
//!     │                                     │
//!  window_base                         highest_seen
//!  (oldest valid)                      (newest seen)
//!
//! Accept conditions:
//! 1. counter > highest_seen → Accept, advance window
//! 2. counter >= window_base AND not in bitmap → Accept, mark in bitmap
//! 3. counter < window_base → Reject (too old)
//! 4. counter in bitmap → Reject (replay)
//! ```
//!
//! ## wallet_index Design (v1.1.0-ChatRelay)
//! ```text
//! wallet_index: DashMap<[u8; 32], SessionId>
//!
//! ┌─────────────────────────────────────────────────────┐
//! │  wallet_bytes  →  SessionId                         │
//! │  [u8; 32]          Arc<Session> (via sessions map)  │
//! └─────────────────────────────────────────────────────┘
//!
//! Lifecycle:
//!  create()  → insert wallet_bytes → SessionId
//!  remove()  → remove wallet_bytes entry (if SessionId matches)
//!
//! MVP constraint: one wallet = one session.
//! If the same wallet reconnects, the new SessionId overwrites the old.
//! The old Session is NOT force-closed here — it will expire naturally
//! via the cleanup task. ChatRelayService always routes to the LATEST session.
//!
//! Phase 2: change to DashMap<[u8; 32], Vec<SessionId>> for multi-device.
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Sessions are stored in a DashMap for concurrent access
//! - Session cleanup releases IP and removes routes
//! - Counters are atomic for lock-free packet handling
//! - Always use SessionManager, not direct Session access
//! - ReplayWindow uses parking_lot::Mutex for performance
//! - Window size of 2048 allows ~100ms of reordering at 20k pps
//! - wallet_index entries are keyed by raw [u8; 32] bytes from
//!   IdentityPublicKey::to_bytes() — NOT hex strings (saves allocation)
//! - remove() checks SessionId before deleting from wallet_index to avoid
//!   race condition where a new session for the same wallet was already
//!   registered before the old one is removed
//!
//! ## Last Modified
//! v0.1.0 - Initial session management
//! v0.1.1 - Fixed replay detection with sliding window algorithm
//! v1.1.0-ChatRelay - 🌟 Added wallet_index reverse lookup to SessionManager
//!                        for zero-knowledge P2P chat relay routing

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

/// Size of the replay window in packets.
/// Allows reordering of up to 2048 packets.
/// At 20,000 pps, this covers ~100ms of network jitter.
const REPLAY_WINDOW_SIZE: u64 = 2048;

/// Size of bitmap in u64 words (2048 / 64 = 32)
const BITMAP_WORDS: usize = (REPLAY_WINDOW_SIZE / 64) as usize;

// ============================================
// Replay Window (Sliding Window Anti-Replay)
// ============================================

/// Result of replay check operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayCheckResult {
    /// Packet is valid and has been recorded.
    Accept,
    /// Packet counter is ahead of window, window advanced.
    AcceptAndAdvance,
    /// Packet is a replay (already seen).
    Replay,
    /// Packet is too old (before window).
    TooOld,
}

/// Sliding window replay detection.
///
/// Uses a bitmap to track which counters have been seen within the window.
/// This allows out-of-order packet delivery while still detecting replays.
///
/// # Algorithm
/// - Maintains `highest_seen` counter and a bitmap of size `REPLAY_WINDOW_SIZE`
/// - Window covers counters from `(highest_seen - WINDOW_SIZE + 1)` to `highest_seen`
/// - Packets ahead of window: accept and advance window
/// - Packets within window: check bitmap, accept if not seen
/// - Packets before window: reject as too old
pub struct ReplayWindow {
    /// Highest counter value seen so far.
    highest_seen: u64,
    /// Bitmap tracking seen counters within the window.
    /// Bit at position (counter % WINDOW_SIZE) indicates if counter was seen.
    bitmap: [u64; BITMAP_WORDS],
}

impl ReplayWindow {
    /// Creates a new replay window starting at counter 0.
    ///
    /// `highest_seen` is initialised to `u64::MAX` as a sentinel so that
    /// the very first packet (counter=0) always takes the `AcceptAndAdvance`
    /// path and correctly updates `rx_counter`. Without this, counter=0 would
    /// return `Accept` instead of `AcceptAndAdvance`, leaving `rx_counter`
    /// un-updated after the first packet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            // Sentinel: any real counter (0..=u64::MAX-1) is > MAX wraps to
            // AcceptAndAdvance on the very first call.
            highest_seen: u64::MAX,
            bitmap: [0u64; BITMAP_WORDS],
        }
    }

    /// Checks if a counter is valid and records it if so.
    ///
    /// # Arguments
    /// * `counter` - The packet counter to check
    ///
    /// # Returns
    /// - `AcceptAndAdvance` if counter is ahead of window (window advanced)
    /// - `Accept` if counter is within window and not seen before
    /// - `Replay` if counter was already seen
    /// - `TooOld` if counter is before the window
    pub fn check_and_record(&mut self, counter: u64) -> ReplayCheckResult {
        // Special case: first packet or counter is ahead of window
        if counter > self.highest_seen {
            let advance = counter - self.highest_seen;

            if advance >= REPLAY_WINDOW_SIZE {
                // Counter is way ahead - clear entire bitmap
                self.bitmap = [0u64; BITMAP_WORDS];
            } else {
                // Clear bits for counters that are now outside the window
                for i in 1..=advance {
                    let old_counter = self.highest_seen + i;
                    self.clear_bit(old_counter);
                }
            }

            self.highest_seen = counter;
            self.set_bit(counter);
            return ReplayCheckResult::AcceptAndAdvance;
        }

        // Calculate window base (oldest valid counter).
        // Guard against u64::MAX sentinel — before any real packet the base is 0.
        let window_base = if self.highest_seen == u64::MAX {
            0
        } else if self.highest_seen >= REPLAY_WINDOW_SIZE - 1 {
            self.highest_seen - REPLAY_WINDOW_SIZE + 1
        } else {
            0
        };

        // Counter is before the window - too old
        if counter < window_base {
            return ReplayCheckResult::TooOld;
        }

        // Counter is within window - check if already seen
        if self.get_bit(counter) {
            return ReplayCheckResult::Replay;
        }

        // Mark as seen and accept
        self.set_bit(counter);
        ReplayCheckResult::Accept
    }

    /// Gets the bit for a counter in the bitmap.
    #[inline]
    fn get_bit(&self, counter: u64) -> bool {
        let bit_index = (counter % REPLAY_WINDOW_SIZE) as usize;
        let word_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        (self.bitmap[word_index] & (1u64 << bit_offset)) != 0
    }

    /// Sets the bit for a counter in the bitmap.
    #[inline]
    fn set_bit(&mut self, counter: u64) {
        let bit_index = (counter % REPLAY_WINDOW_SIZE) as usize;
        let word_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        self.bitmap[word_index] |= 1u64 << bit_offset;
    }

    /// Clears the bit for a counter in the bitmap.
    #[inline]
    fn clear_bit(&mut self, counter: u64) {
        let bit_index = (counter % REPLAY_WINDOW_SIZE) as usize;
        let word_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        self.bitmap[word_index] &= !(1u64 << bit_offset);
    }

    /// Returns the highest seen counter.
    #[must_use]
    pub fn highest_seen(&self) -> u64 {
        self.highest_seen
    }

    /// Returns the window base (oldest valid counter).
    ///
    /// Returns 0 when the window has not yet received any real packet
    /// (i.e. `highest_seen` is still the `u64::MAX` sentinel).
    #[must_use]
    pub fn window_base(&self) -> u64 {
        if self.highest_seen == u64::MAX {
            return 0;
        }
        if self.highest_seen >= REPLAY_WINDOW_SIZE - 1 {
            self.highest_seen - REPLAY_WINDOW_SIZE + 1
        } else {
            0
        }
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
            .field("highest_seen", &self.highest_seen)
            .field("window_base", &self.window_base())
            .finish()
    }
}

// ============================================
// Session State
// ============================================

/// Session state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Established,
    Closing,
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
// Session Statistics
// ============================================

/// Session statistics.
#[derive(Debug, Default)]
pub struct SessionStats {
    pub bytes_rx: AtomicU64,
    pub bytes_tx: AtomicU64,
    pub packets_rx: AtomicU64,
    pub packets_tx: AtomicU64,
    pub replays_rejected: AtomicU64,
    pub too_old_rejected: AtomicU64,
}

impl SessionStats {
    pub fn record_rx(&self, bytes: u64) {
        self.bytes_rx.fetch_add(bytes, Ordering::Relaxed);
        self.packets_rx.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_tx(&self, bytes: u64) {
        self.bytes_tx.fetch_add(bytes, Ordering::Relaxed);
        self.packets_tx.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replay_rejected(&self) {
        self.replays_rejected.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_too_old_rejected(&self) {
        self.too_old_rejected.fetch_add(1, Ordering::Relaxed);
    }

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

#[derive(Debug, Clone, Copy)]
pub struct StatsSnapshot {
    pub bytes_rx: u64,
    pub bytes_tx: u64,
    pub packets_rx: u64,
    pub packets_tx: u64,
    pub replays_rejected: u64,
    pub too_old_rejected: u64,
}

// ============================================
// Session
// ============================================

/// Represents an active client session.
pub struct Session {
    pub id: SessionId,
    state: RwLock<SessionState>,
    pub client_public_key: IdentityPublicKey,
    pub session_key: SessionKey,
    pub virtual_ip: Ipv4Addr,
    pub client_endpoint: SocketAddr,
    pub created_at: std::time::Instant,
    pub last_activity: AtomicInstant,
    pub tx_counter: AtomicU64,
    /// Replay window for RX counter validation (replaces simple rx_counter)
    replay_window: Mutex<ReplayWindow>,
    /// Kept for backward compatibility - reflects highest seen counter
    pub rx_counter: AtomicU64,
    pub stats: SessionStats,
}

impl Session {
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

    #[must_use]
    pub fn state(&self) -> SessionState {
        *self.state.read()
    }

    pub fn set_state(&self, state: SessionState) {
        *self.state.write() = state;
    }

    #[must_use]
    pub fn is_established(&self) -> bool {
        self.state() == SessionState::Established
    }

    pub fn touch(&self) {
        self.last_activity.store(std::time::Instant::now());
    }

    #[must_use]
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }

    #[must_use]
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.idle_time() > timeout
    }

    #[must_use]
    pub fn next_tx_counter(&self) -> u64 {
        self.tx_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Validates an RX counter using sliding window replay detection.
    ///
    /// # Arguments
    /// * `counter` - The packet counter to validate
    ///
    /// # Returns
    /// `true` if the counter is valid (not a replay, not too old)
    /// `false` if the counter should be rejected
    ///
    /// # Algorithm
    /// Uses a sliding window of 2048 packets to allow out-of-order delivery
    /// while still detecting replays. This is similar to WireGuard's approach.
    pub fn validate_rx_counter(&self, counter: u64) -> bool {
        let mut window = self.replay_window.lock();
        let result = window.check_and_record(counter);

        // Update rx_counter for backward compatibility (reflects highest seen)
        if matches!(result, ReplayCheckResult::AcceptAndAdvance) {
            self.rx_counter.store(counter, Ordering::SeqCst);
        }

        match result {
            ReplayCheckResult::Accept | ReplayCheckResult::AcceptAndAdvance => {
                trace!(
                    session_id = %self.id,
                    counter = counter,
                    highest = window.highest_seen(),
                    "Counter accepted"
                );
                true
            }
            ReplayCheckResult::Replay => {
                self.stats.record_replay_rejected();
                debug!(
                    session_id = %self.id,
                    counter = counter,
                    highest = window.highest_seen(),
                    "Replay detected - counter already seen"
                );
                false
            }
            ReplayCheckResult::TooOld => {
                self.stats.record_too_old_rejected();
                debug!(
                    session_id = %self.id,
                    counter = counter,
                    window_base = window.window_base(),
                    highest = window.highest_seen(),
                    "Counter too old - outside window"
                );
                false
            }
        }
    }

    /// Returns the current replay window statistics.
    #[must_use]
    pub fn replay_window_info(&self) -> (u64, u64) {
        let window = self.replay_window.lock();
        (window.window_base(), window.highest_seen())
    }

    /// Returns the wallet address bytes for this session.
    ///
    /// Convenience wrapper over `client_public_key.to_bytes()` used by
    /// `SessionManager::wallet_index` operations and `ChatRelayService`.
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
            .field("replay_window", &format!("[{}..{}]", window_base, highest_seen))
            .finish_non_exhaustive()
    }
}

// ============================================
// Session Manager
// ============================================

/// Manages all active sessions with wallet-address reverse lookup.
///
/// ## v1.1.0-ChatRelay Addition
/// `wallet_index` maps `[u8; 32]` wallet bytes to `SessionId`, enabling
/// `ChatRelayService::get_session_by_wallet()` to route incoming chat
/// messages to the correct session in O(1) without iterating all sessions.
///
/// ### Invariants
/// - Every entry in `wallet_index` has a corresponding entry in `sessions`.
/// - `create()` always inserts into both maps atomically (from the caller's
///   perspective — no external lock needed because DashMap sharding handles
///   individual operations; the two-map update is not strictly atomic but
///   the window is tiny and chat routing tolerates a brief gap).
/// - `remove()` checks that the `wallet_index` entry still points to the
///   *same* `SessionId` before deleting, avoiding a race where a new session
///   for the same wallet was registered between the `sessions.remove()` call
///   and the `wallet_index.remove()` call.
pub struct SessionManager {
    sessions: DashMap<SessionId, Arc<Session>>,

    // 🌟 v1.1.0-ChatRelay: wallet address → SessionId reverse index.
    //
    // Keyed by raw [u8; 32] wallet bytes (from IdentityPublicKey::to_bytes()).
    // MVP: one wallet → one active SessionId (last-writer-wins on reconnect).
    // Phase 2: change value type to Vec<SessionId> for multi-device support.
    wallet_index: DashMap<[u8; 32], SessionId>,

    max_sessions: usize,
    session_timeout: Duration,
}

impl SessionManager {
    #[must_use]
    pub fn new(max_sessions: usize, session_timeout: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            wallet_index: DashMap::new(),
            max_sessions,
            session_timeout,
        }
    }

    /// Creates and registers a new session with a specific session ID.
    ///
    /// Also inserts the wallet → SessionId mapping into `wallet_index`.
    /// If the same wallet reconnects, the index entry is overwritten with
    /// the new SessionId (MVP: last-writer-wins).
    ///
    /// # Arguments
    /// * `session_id` - The session ID to use (must match the one sent to client)
    /// * `client_public_key` - Client's identity public key
    /// * `session_key` - Derived session key for encryption
    /// * `virtual_ip` - Assigned virtual IP address
    /// * `client_endpoint` - Client's UDP endpoint
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

        // Extract wallet bytes before inserting (avoids borrow after move)
        let wallet_bytes = session.wallet_bytes();

        self.sessions.insert(session_id.clone(), Arc::clone(&session));

        // 🌟 v1.1.0-ChatRelay: register wallet → SessionId.
        //
        // Use insert() which is a single CAS-like operation on the DashMap shard,
        // minimising (but not fully eliminating — two separate maps) the race window
        // vs. the sessions.insert() above. In the MVP single-UDP-loop architecture
        // sessions are created serially, so concurrent creation of the same wallet
        // is not possible in practice. Phase 2 multi-device support should use a
        // purpose-built atomic (wallet, session_ids) structure.
        self.wallet_index.insert(wallet_bytes, session_id.clone());

        info!(
            session_id = %session_id,
            virtual_ip = %virtual_ip,
            client = %client_endpoint,
            wallet = %hex::encode(&wallet_bytes[..4]),  // log only first 4 bytes
            "Session created"
        );

        Ok(session)
    }

    #[must_use]
    pub fn get(&self, id: &SessionId) -> Option<Arc<Session>> {
        self.sessions.get(id).map(|r| Arc::clone(r.value()))
    }

    pub fn get_or_error(&self, id: &SessionId) -> Result<Arc<Session>> {
        self.get(id).ok_or_else(|| ServerError::SessionNotFound(id.clone()))
    }

    /// 🌟 v1.1.0-ChatRelay: Look up an active session by wallet address.
    ///
    /// Used by `ChatRelayService` to find the online session for a message
    /// receiver, enabling immediate forwarding without scanning all sessions.
    ///
    /// Returns `None` if no active session exists for the given wallet
    /// (receiver is offline → message should be stored for later delivery).
    ///
    /// # Arguments
    /// * `wallet` - The 32-byte Ed25519 public key (wallet address) to look up
    #[must_use]
    pub fn get_by_wallet(&self, wallet: &[u8; 32]) -> Option<Arc<Session>> {
        self.wallet_index
            .get(wallet)
            .and_then(|sid| self.sessions.get(sid.value()).map(|s| Arc::clone(s.value())))
    }

    pub fn remove(&self, id: &SessionId) -> Option<Arc<Session>> {
        let removed = self.sessions.remove(id).map(|(_, s)| s);

        if let Some(ref session) = removed {
            session.set_state(SessionState::Closed);

            // 🌟 v1.1.0-ChatRelay: clean up wallet_index.
            //
            // Safety check: only remove the wallet_index entry if it still
            // points to THIS session's ID. If the same wallet reconnected
            // before we got here, the index already points to the new session
            // and we must NOT delete it.
            let wallet_bytes = session.wallet_bytes();
            let should_remove = self.wallet_index
                .get(&wallet_bytes)
                .map(|sid| *sid.value() == *id)
                .unwrap_or(false);

            if should_remove {
                self.wallet_index.remove(&wallet_bytes);
                debug!(
                    session_id = %id,
                    wallet = %hex::encode(&wallet_bytes[..4]),
                    "wallet_index entry removed"
                );
            }

            let stats = session.stats.snapshot();
            info!(
                session_id = %id,
                virtual_ip = %session.virtual_ip,
                packets_rx = stats.packets_rx,
                packets_tx = stats.packets_tx,
                replays_rejected = stats.replays_rejected,
                too_old_rejected = stats.too_old_rejected,
                "Session removed"
            );
        }

        removed
    }

    pub fn close(&self, id: &SessionId) {
        if let Some(session) = self.get(id) {
            session.set_state(SessionState::Closing);
        }
        self.remove(id);
    }

    #[must_use]
    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Cleans up expired sessions.
    ///
    /// Also removes corresponding `wallet_index` entries for expired sessions.
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
            self.remove(id); // wallet_index cleanup happens inside remove()
        }

        if !expired.is_empty() {
            info!(
                count = expired.len(),
                wallet_index_size = self.wallet_index.len(),
                "Cleaned up expired sessions"
            );
        }

        expired
    }

    #[must_use]
    pub fn all_sessions(&self) -> Vec<Arc<Session>> {
        self.sessions.iter().map(|r| Arc::clone(r.value())).collect()
    }

    pub fn touch(&self, id: &SessionId) {
        if let Some(session) = self.get(id) {
            session.touch();
        }
    }

    /// Returns the current number of entries in the wallet index.
    ///
    /// Used for diagnostics. Should equal `count()` in normal operation.
    /// A discrepancy indicates a bug in index maintenance.
    #[must_use]
    pub fn wallet_index_count(&self) -> usize {
        self.wallet_index.len()
    }
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager")
            .field("sessions", &self.count())
            .field("wallet_index", &self.wallet_index_count())
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

    // ========================================
    // ReplayWindow Tests (preserved verbatim)
    // ========================================

    #[test]
    fn test_replay_window_sequential() {
        let mut window = ReplayWindow::new();
        for i in 1..=100 {
            let result = window.check_and_record(i);
            assert!(
                matches!(result, ReplayCheckResult::AcceptAndAdvance),
                "Counter {} should be accepted", i
            );
        }
        assert_eq!(window.highest_seen(), 100);
    }

    #[test]
    fn test_replay_window_replay_detection() {
        let mut window = ReplayWindow::new();
        assert!(matches!(window.check_and_record(100), ReplayCheckResult::AcceptAndAdvance));
        assert!(matches!(window.check_and_record(100), ReplayCheckResult::Replay));
    }

    #[test]
    fn test_replay_window_out_of_order() {
        let mut window = ReplayWindow::new();
        assert!(matches!(window.check_and_record(100), ReplayCheckResult::AcceptAndAdvance));
        assert!(matches!(window.check_and_record(95), ReplayCheckResult::Accept));
        assert!(matches!(window.check_and_record(99), ReplayCheckResult::Accept));
        assert!(matches!(window.check_and_record(95), ReplayCheckResult::Replay));
    }

    #[test]
    fn test_replay_window_too_old() {
        let mut window = ReplayWindow::new();
        // Advance window to 3000
        window.check_and_record(3000);
        // window_base = 3000 - 2048 + 1 = 953
        assert!(matches!(window.check_and_record(500), ReplayCheckResult::TooOld));
        // Packet exactly at window_base = 953 should still be valid
        assert!(matches!(window.check_and_record(953), ReplayCheckResult::Accept));
    }

    #[test]
    fn test_replay_window_counter_zero_is_accept_and_advance() {
        // Regression: counter=0 must return AcceptAndAdvance (not Accept)
        // so that rx_counter is updated on the very first packet.
        let mut window = ReplayWindow::new();
        let result = window.check_and_record(0);
        assert!(
            matches!(result, ReplayCheckResult::AcceptAndAdvance),
            "counter=0 on a fresh window must return AcceptAndAdvance, got {:?}", result
        );
        assert_eq!(window.highest_seen(), 0);
    }

    #[test]
    fn test_replay_window_large_jump() {
        let mut window = ReplayWindow::new();
        window.check_and_record(100);
        assert!(matches!(window.check_and_record(10000), ReplayCheckResult::AcceptAndAdvance));
        assert_eq!(window.highest_seen(), 10000);
        assert!(matches!(window.check_and_record(100), ReplayCheckResult::TooOld));
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

    // ========================================
    // Session Tests (preserved verbatim)
    // ========================================

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
        assert!(session.validate_rx_counter(97));
        assert!(session.validate_rx_counter(101));
        assert!(session.validate_rx_counter(102));
        assert!(!session.validate_rx_counter(95));
        assert!(!session.validate_rx_counter(100));
        assert!(!session.validate_rx_counter(101));
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

    // ========================================
    // SessionManager Tests (preserved + new)
    // ========================================

    fn make_manager() -> SessionManager {
        SessionManager::new(100, Duration::from_secs(300))
    }

    #[test]
    fn test_session_manager_create() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let session_id = SessionId::generate();

        let session = manager.create(
            session_id.clone(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();

        assert_eq!(manager.count(), 1);
        assert_eq!(session.id, session_id);

        let retrieved = manager.get(&session.id).unwrap();
        assert_eq!(retrieved.virtual_ip, Ipv4Addr::new(100, 64, 0, 2));
    }

    #[test]
    fn test_session_manager_limit() {
        let manager = SessionManager::new(2, Duration::from_secs(300));
        let identity = IdentityKeyPair::generate();

        manager.create(
            SessionId::generate(), identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2), "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();

        manager.create(
            SessionId::generate(), identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 3), "127.0.0.1:12346".parse().unwrap(),
        ).unwrap();

        let result = manager.create(
            SessionId::generate(), identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 4), "127.0.0.1:12347".parse().unwrap(),
        );

        assert!(matches!(result, Err(ServerError::SessionLimitReached { .. })));
    }

    #[test]
    fn test_session_stats_tracking() {
        let session = create_test_session();
        session.validate_rx_counter(100);
        session.validate_rx_counter(100); // Replay
        session.validate_rx_counter(100); // Replay
        let stats = session.stats.snapshot();
        assert_eq!(stats.replays_rejected, 2);
    }

    // ========================================
    // 🌟 v1.1.0-ChatRelay: wallet_index Tests
    // ========================================

    #[test]
    fn test_get_by_wallet_returns_session() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();

        manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();

        let wallet = identity.public_key_bytes();
        let found = manager.get_by_wallet(&wallet);
        assert!(found.is_some(), "Should find session by wallet");
        assert_eq!(found.unwrap().wallet_bytes(), wallet);
    }

    #[test]
    fn test_get_by_wallet_returns_none_when_not_registered() {
        let manager = make_manager();
        let random_wallet = [0xFFu8; 32];
        assert!(manager.get_by_wallet(&random_wallet).is_none());
    }

    #[test]
    fn test_wallet_index_cleaned_on_remove() {
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();

        let session = manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();

        assert!(manager.get_by_wallet(&wallet).is_some());
        assert_eq!(manager.wallet_index_count(), 1);

        manager.remove(&session.id);

        assert!(manager.get_by_wallet(&wallet).is_none());
        assert_eq!(manager.wallet_index_count(), 0);
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_wallet_index_not_removed_on_reconnect_race() {
        // Simulates the reconnect race condition:
        // 1. Alice connects → session_A registered
        // 2. Alice reconnects → session_B registered (overwrites wallet_index)
        // 3. session_A is removed → wallet_index must NOT be cleared
        //    because it now points to session_B
        let manager = make_manager();
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();

        // Session A
        let session_a = manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x01; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:11111".parse().unwrap(),
        ).unwrap();

        // Session B (same wallet, new connection — overwrites wallet_index)
        let session_b = manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x02; 32]),
            Ipv4Addr::new(100, 64, 0, 3),
            "127.0.0.1:22222".parse().unwrap(),
        ).unwrap();

        // wallet_index now points to session_B
        let found = manager.get_by_wallet(&wallet).unwrap();
        assert_eq!(found.id, session_b.id, "wallet_index should point to session_B");

        // Remove session_A — wallet_index must NOT be cleared
        manager.remove(&session_a.id);

        let still_found = manager.get_by_wallet(&wallet);
        assert!(still_found.is_some(), "wallet_index must survive session_A removal");
        assert_eq!(
            still_found.unwrap().id, session_b.id,
            "wallet_index must still point to session_B"
        );
        assert_eq!(manager.wallet_index_count(), 1);
    }

    #[test]
    fn test_wallet_index_count_matches_session_count() {
        let manager = make_manager();

        // Each unique wallet adds one entry to wallet_index
        for i in 0..5u8 {
            let identity = IdentityKeyPair::generate();
            manager.create(
                SessionId::generate(),
                identity.public_key(),
                SessionKey::from_bytes([i; 32]),
                Ipv4Addr::new(100, 64, 0, 2 + i as u32),
                format!("127.0.0.1:{}", 10000 + i as u16).parse().unwrap(),
            ).unwrap();
        }

        assert_eq!(manager.count(), 5);
        assert_eq!(manager.wallet_index_count(), 5);
    }

    #[test]
    fn test_get_by_wallet_after_cleanup_expired() {
        let manager = SessionManager::new(100, Duration::from_millis(1));
        let identity = IdentityKeyPair::generate();
        let wallet = identity.public_key_bytes();

        manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();

        // Wait for session to expire
        std::thread::sleep(Duration::from_millis(10));
        manager.cleanup_expired();

        assert!(manager.get_by_wallet(&wallet).is_none());
        assert_eq!(manager.wallet_index_count(), 0);
    }
}
