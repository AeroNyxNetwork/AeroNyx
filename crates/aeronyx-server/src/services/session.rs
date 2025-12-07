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
//! Fixed replay detection to use sliding window algorithm instead of
//! strict increment. UDP packets can arrive out-of-order, so we need
//! to tolerate reordering within a window while still detecting replays.
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
//! ## ⚠️ Important Note for Next Developer
//! - Sessions are stored in a DashMap for concurrent access
//! - Session cleanup releases IP and removes routes
//! - Counters are atomic for lock-free packet handling
//! - Always use SessionManager, not direct Session access
//! - ReplayWindow uses parking_lot::Mutex for performance
//! - Window size of 2048 allows ~100ms of reordering at 20k pps
//!
//! ## Last Modified
//! v0.1.0 - Initial session management
//! v0.1.1 - Fixed replay detection with sliding window algorithm

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
    #[must_use]
    pub fn new() -> Self {
        Self {
            highest_seen: 0,
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

        // Calculate window base (oldest valid counter)
        let window_base = if self.highest_seen >= REPLAY_WINDOW_SIZE - 1 {
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
    #[must_use]
    pub fn window_base(&self) -> u64 {
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

/// Manages all active sessions.
pub struct SessionManager {
    sessions: DashMap<SessionId, Arc<Session>>,
    max_sessions: usize,
    session_timeout: Duration,
}

impl SessionManager {
    #[must_use]
    pub fn new(max_sessions: usize, session_timeout: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions,
            session_timeout,
        }
    }

    /// Creates and registers a new session with a specific session ID.
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

        self.sessions.insert(session_id.clone(), Arc::clone(&session));

        info!(
            session_id = %session_id,
            virtual_ip = %virtual_ip,
            client = %client_endpoint,
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

    pub fn remove(&self, id: &SessionId) -> Option<Arc<Session>> {
        let removed = self.sessions.remove(id).map(|(_, s)| s);
        
        if let Some(ref session) = removed {
            session.set_state(SessionState::Closed);
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
            info!("Cleaned up {} expired sessions", expired.len());
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
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager")
            .field("sessions", &self.count())
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
    // ReplayWindow Tests
    // ========================================

    #[test]
    fn test_replay_window_sequential() {
        let mut window = ReplayWindow::new();
        
        // Sequential packets should all be accepted
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
        
        // Accept first packet
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::AcceptAndAdvance
        ));
        
        // Same counter should be rejected as replay
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::Replay
        ));
    }

    #[test]
    fn test_replay_window_out_of_order() {
        let mut window = ReplayWindow::new();
        
        // Accept packet 100
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::AcceptAndAdvance
        ));
        
        // Accept packet 95 (within window, not seen)
        assert!(matches!(
            window.check_and_record(95),
            ReplayCheckResult::Accept
        ));
        
        // Accept packet 99 (within window, not seen)
        assert!(matches!(
            window.check_and_record(99),
            ReplayCheckResult::Accept
        ));
        
        // Reject packet 95 again (replay)
        assert!(matches!(
            window.check_and_record(95),
            ReplayCheckResult::Replay
        ));
    }

    #[test]
    fn test_replay_window_too_old() {
        let mut window = ReplayWindow::new();
        
        // Advance window to 3000
        window.check_and_record(3000);
        
        // Packet at 500 should be too old (before window_base)
        // Window base = 3000 - 2048 + 1 = 953
        assert!(matches!(
            window.check_and_record(500),
            ReplayCheckResult::TooOld
        ));
        
        // Packet at 953 should still be valid (exactly at window base)
        assert!(matches!(
            window.check_and_record(953),
            ReplayCheckResult::Accept
        ));
    }

    #[test]
    fn test_replay_window_large_jump() {
        let mut window = ReplayWindow::new();
        
        // Start at 100
        window.check_and_record(100);
        
        // Jump to 10000 (clears entire bitmap)
        assert!(matches!(
            window.check_and_record(10000),
            ReplayCheckResult::AcceptAndAdvance
        ));
        
        assert_eq!(window.highest_seen(), 10000);
        
        // Old counter 100 should now be too old
        assert!(matches!(
            window.check_and_record(100),
            ReplayCheckResult::TooOld
        ));
    }

    #[test]
    fn test_replay_window_stress() {
        let mut window = ReplayWindow::new();
        
        // Simulate realistic traffic with some reordering
        let mut counters: Vec<u64> = (1..=10000).collect();
        
        // Shuffle to simulate network reordering (but keep mostly in order)
        // Swap adjacent pairs occasionally
        for i in (0..counters.len() - 1).step_by(10) {
            counters.swap(i, i + 1);
        }
        
        let mut accepted = 0u64;
        let mut rejected = 0u64;
        
        for counter in counters {
            match window.check_and_record(counter) {
                ReplayCheckResult::Accept | ReplayCheckResult::AcceptAndAdvance => {
                    accepted += 1;
                }
                _ => {
                    rejected += 1;
                }
            }
        }
        
        // All packets should be accepted (reordering is within window)
        assert_eq!(accepted, 10000);
        assert_eq!(rejected, 0);
    }

    // ========================================
    // Session Tests
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
        
        // Sequential should work
        assert!(session.validate_rx_counter(1));
        assert!(session.validate_rx_counter(2));
        assert!(session.validate_rx_counter(3));
        
        // Replay should fail
        assert!(!session.validate_rx_counter(2));
        
        // Out of order within window should work
        assert!(session.validate_rx_counter(5));
        assert!(session.validate_rx_counter(4)); // Still within window
        
        // But replay of 4 should fail
        assert!(!session.validate_rx_counter(4));
    }

    #[test]
    fn test_session_rx_counter_out_of_order() {
        let session = create_test_session();
        
        // Accept 100
        assert!(session.validate_rx_counter(100));
        
        // Accept 95, 96, 97 (out of order but within window)
        assert!(session.validate_rx_counter(95));
        assert!(session.validate_rx_counter(96));
        assert!(session.validate_rx_counter(97));
        
        // Accept 101, 102
        assert!(session.validate_rx_counter(101));
        assert!(session.validate_rx_counter(102));
        
        // All these should be replays now
        assert!(!session.validate_rx_counter(95));
        assert!(!session.validate_rx_counter(100));
        assert!(!session.validate_rx_counter(101));
    }

    // ========================================
    // SessionManager Tests
    // ========================================

    #[test]
    fn test_session_manager_create() {
        let manager = SessionManager::new(100, Duration::from_secs(300));
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
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();
        
        manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 3),
            "127.0.0.1:12346".parse().unwrap(),
        ).unwrap();
        
        let result = manager.create(
            SessionId::generate(),
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 4),
            "127.0.0.1:12347".parse().unwrap(),
        );
        
        assert!(matches!(result, Err(ServerError::SessionLimitReached { .. })));
    }

    #[test]
    fn test_session_stats_tracking() {
        let session = create_test_session();
        
        // Trigger some replay rejections
        session.validate_rx_counter(100);
        session.validate_rx_counter(100); // Replay
        session.validate_rx_counter(100); // Replay
        
        let stats = session.stats.snapshot();
        assert_eq!(stats.replays_rejected, 2);
    }
}
