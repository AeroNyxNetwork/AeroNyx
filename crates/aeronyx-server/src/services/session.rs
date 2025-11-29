// ============================================
// File: crates/aeronyx-server/src/services/session.rs
// ============================================
//! # Session Management Service
//!
//! ## Creation Reason
//! Manages the lifecycle of client sessions, including creation,
//! lookup, timeout handling, and cleanup.
//!
//! ## Main Functionality
//! - `Session`: Session data structure
//! - `SessionManager`: Session lifecycle management
//! - `SessionState`: Session state machine
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
//! ## ⚠️ Important Note for Next Developer
//! - Sessions are stored in a DashMap for concurrent access
//! - Session cleanup releases IP and removes routes
//! - Counters are atomic for lock-free packet handling
//! - Always use SessionManager, not direct Session access
//!
//! ## Last Modified
//! v0.1.0 - Initial session management

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use aeronyx_common::time::AtomicInstant;
use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::keys::{IdentityPublicKey, SessionKey};

use crate::error::{Result, ServerError};

// ============================================
// SessionState
// ============================================

/// Session state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is fully established and active.
    Established,
    /// Session is being closed.
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
// SessionStats
// ============================================

/// Session statistics.
#[derive(Debug, Default)]
pub struct SessionStats {
    /// Bytes received from client.
    pub bytes_rx: AtomicU64,
    /// Bytes sent to client.
    pub bytes_tx: AtomicU64,
    /// Packets received from client.
    pub packets_rx: AtomicU64,
    /// Packets sent to client.
    pub packets_tx: AtomicU64,
}

impl SessionStats {
    /// Records received data.
    pub fn record_rx(&self, bytes: u64) {
        self.bytes_rx.fetch_add(bytes, Ordering::Relaxed);
        self.packets_rx.fetch_add(1, Ordering::Relaxed);
    }

    /// Records sent data.
    pub fn record_tx(&self, bytes: u64) {
        self.bytes_tx.fetch_add(bytes, Ordering::Relaxed);
        self.packets_tx.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns current statistics snapshot.
    #[must_use]
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            bytes_rx: self.bytes_rx.load(Ordering::Relaxed),
            bytes_tx: self.bytes_tx.load(Ordering::Relaxed),
            packets_rx: self.packets_rx.load(Ordering::Relaxed),
            packets_tx: self.packets_tx.load(Ordering::Relaxed),
        }
    }
}

/// Statistics snapshot (non-atomic copy).
#[derive(Debug, Clone, Copy)]
pub struct StatsSnapshot {
    /// Bytes received.
    pub bytes_rx: u64,
    /// Bytes sent.
    pub bytes_tx: u64,
    /// Packets received.
    pub packets_rx: u64,
    /// Packets sent.
    pub packets_tx: u64,
}

// ============================================
// Session
// ============================================

/// Represents an active client session.
///
/// # Thread Safety
/// - Atomic counters for packet handling
/// - RwLock for state changes
/// - AtomicInstant for activity tracking
pub struct Session {
    /// Unique session identifier.
    pub id: SessionId,
    /// Current session state.
    state: RwLock<SessionState>,
    /// Client's identity public key.
    pub client_public_key: IdentityPublicKey,
    /// Derived session encryption key.
    pub session_key: SessionKey,
    /// Virtual IP assigned to client.
    pub virtual_ip: Ipv4Addr,
    /// Client's UDP endpoint.
    pub client_endpoint: SocketAddr,
    /// Session creation time.
    pub created_at: std::time::Instant,
    /// Last activity time (atomic for concurrent updates).
    pub last_activity: AtomicInstant,
    /// Transmit counter (server → client).
    pub tx_counter: AtomicU64,
    /// Receive counter (client → server).
    pub rx_counter: AtomicU64,
    /// Session statistics.
    pub stats: SessionStats,
}

impl Session {
    /// Creates a new session.
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
            rx_counter: AtomicU64::new(0),
            stats: SessionStats::default(),
        }
    }

    /// Returns the current session state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        *self.state.read()
    }

    /// Sets the session state.
    pub fn set_state(&self, state: SessionState) {
        *self.state.write() = state;
    }

    /// Returns `true` if the session is established.
    #[must_use]
    pub fn is_established(&self) -> bool {
        self.state() == SessionState::Established
    }

    /// Updates the last activity timestamp.
    pub fn touch(&self) {
        self.last_activity.store(std::time::Instant::now());
    }

    /// Returns the duration since last activity.
    #[must_use]
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Checks if the session has exceeded the timeout.
    #[must_use]
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.idle_time() > timeout
    }

    /// Gets the next transmit counter value.
    #[must_use]
    pub fn next_tx_counter(&self) -> u64 {
        self.tx_counter.fetch_add(1, Ordering::SeqCst)
    }

    /// Validates and updates the receive counter.
    ///
    /// # Arguments
    /// * `counter` - The counter value from the received packet
    ///
    /// # Returns
    /// `true` if the counter is valid (greater than previous), `false` otherwise.
    pub fn validate_rx_counter(&self, counter: u64) -> bool {
        let current = self.rx_counter.load(Ordering::SeqCst);
        if counter > current {
            self.rx_counter.store(counter, Ordering::SeqCst);
            true
        } else {
            false
        }
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("state", &self.state())
            .field("virtual_ip", &self.virtual_ip)
            .field("client_endpoint", &self.client_endpoint)
            .field("idle_time", &self.idle_time())
            .finish_non_exhaustive()
    }
}

// ============================================
// SessionManager
// ============================================

/// Manages all active sessions.
///
/// # Thread Safety
/// Uses DashMap for concurrent access without global locks.
pub struct SessionManager {
    /// Active sessions by ID.
    sessions: DashMap<SessionId, Arc<Session>>,
    /// Maximum allowed sessions.
    max_sessions: usize,
    /// Session timeout duration.
    session_timeout: Duration,
}

impl SessionManager {
    /// Creates a new session manager.
    ///
    /// # Arguments
    /// * `max_sessions` - Maximum concurrent sessions
    /// * `session_timeout` - Session inactivity timeout
    #[must_use]
    pub fn new(max_sessions: usize, session_timeout: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions,
            session_timeout,
        }
    }

    /// Creates and registers a new session.
    ///
    /// # Arguments
    /// * `client_public_key` - Client's identity public key
    /// * `session_key` - Derived session encryption key
    /// * `virtual_ip` - Virtual IP assigned to client
    /// * `client_endpoint` - Client's UDP address
    ///
    /// # Returns
    /// The created session wrapped in Arc.
    ///
    /// # Errors
    /// - `SessionLimitReached`: If max sessions exceeded
    pub fn create(
        &self,
        client_public_key: IdentityPublicKey,
        session_key: SessionKey,
        virtual_ip: Ipv4Addr,
        client_endpoint: SocketAddr,
    ) -> Result<Arc<Session>> {
        // Check session limit
        if self.sessions.len() >= self.max_sessions {
            return Err(ServerError::SessionLimitReached {
                limit: self.max_sessions,
            });
        }

        let session_id = SessionId::generate();
        let session = Arc::new(Session::new(
            session_id,
            client_public_key,
            session_key,
            virtual_ip,
            client_endpoint,
        ));

        self.sessions.insert(session_id, Arc::clone(&session));

        info!(
            session_id = %session_id,
            virtual_ip = %virtual_ip,
            client = %client_endpoint,
            "Session created"
        );

        Ok(session)
    }

    /// Gets a session by ID.
    #[must_use]
    pub fn get(&self, id: &SessionId) -> Option<Arc<Session>> {
        self.sessions.get(id).map(|r| Arc::clone(r.value()))
    }

    /// Gets a session by ID, returning error if not found.
    pub fn get_or_error(&self, id: &SessionId) -> Result<Arc<Session>> {
        self.get(id).ok_or_else(|| ServerError::SessionNotFound(*id))
    }

    /// Removes a session.
    ///
    /// # Returns
    /// The removed session, if it existed.
    pub fn remove(&self, id: &SessionId) -> Option<Arc<Session>> {
        let removed = self.sessions.remove(id).map(|(_, s)| s);
        
        if let Some(ref session) = removed {
            session.set_state(SessionState::Closed);
            info!(
                session_id = %id,
                virtual_ip = %session.virtual_ip,
                "Session removed"
            );
        }

        removed
    }

    /// Closes a session gracefully.
    pub fn close(&self, id: &SessionId) {
        if let Some(session) = self.get(id) {
            session.set_state(SessionState::Closing);
        }
        self.remove(id);
    }

    /// Returns the number of active sessions.
    #[must_use]
    pub fn count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns `true` if the manager has no sessions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Cleans up expired sessions.
    ///
    /// # Returns
    /// List of expired session IDs and their virtual IPs.
    pub fn cleanup_expired(&self) -> Vec<(SessionId, Ipv4Addr)> {
        let mut expired = Vec::new();

        // Collect expired sessions
        for entry in self.sessions.iter() {
            let session = entry.value();
            if session.is_expired(self.session_timeout) {
                expired.push((session.id, session.virtual_ip));
            }
        }

        // Remove expired sessions
        for (id, ip) in &expired {
            debug!(session_id = %id, virtual_ip = %ip, "Session expired");
            self.remove(id);
        }

        if !expired.is_empty() {
            info!("Cleaned up {} expired sessions", expired.len());
        }

        expired
    }

    /// Returns all active sessions.
    #[must_use]
    pub fn all_sessions(&self) -> Vec<Arc<Session>> {
        self.sessions.iter().map(|r| Arc::clone(r.value())).collect()
    }

    /// Updates activity timestamp for a session.
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
        assert_eq!(session.tx_counter.load(Ordering::SeqCst), 0);
        assert_eq!(session.rx_counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_session_counters() {
        let session = create_test_session();
        
        // TX counter
        assert_eq!(session.next_tx_counter(), 0);
        assert_eq!(session.next_tx_counter(), 1);
        assert_eq!(session.next_tx_counter(), 2);
        
        // RX counter validation
        assert!(session.validate_rx_counter(1));
        assert!(!session.validate_rx_counter(1)); // Same counter should fail
        assert!(!session.validate_rx_counter(0)); // Lower counter should fail
        assert!(session.validate_rx_counter(5)); // Higher counter should succeed
    }

    #[test]
    fn test_session_state() {
        let session = create_test_session();
        
        assert_eq!(session.state(), SessionState::Established);
        
        session.set_state(SessionState::Closing);
        assert_eq!(session.state(), SessionState::Closing);
        assert!(!session.is_established());
    }

    #[test]
    fn test_session_manager_create() {
        let manager = SessionManager::new(100, Duration::from_secs(300));
        let identity = IdentityKeyPair::generate();
        
        let session = manager.create(
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();
        
        assert_eq!(manager.count(), 1);
        
        // Should be able to retrieve it
        let retrieved = manager.get(&session.id).unwrap();
        assert_eq!(retrieved.virtual_ip, Ipv4Addr::new(100, 64, 0, 2));
    }

    #[test]
    fn test_session_manager_limit() {
        let manager = SessionManager::new(2, Duration::from_secs(300));
        let identity = IdentityKeyPair::generate();
        
        // Create 2 sessions (at limit)
        manager.create(
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();
        
        manager.create(
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 3),
            "127.0.0.1:12346".parse().unwrap(),
        ).unwrap();
        
        // Third should fail
        let result = manager.create(
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 4),
            "127.0.0.1:12347".parse().unwrap(),
        );
        
        assert!(matches!(result, Err(ServerError::SessionLimitReached { .. })));
    }

    #[test]
    fn test_session_manager_remove() {
        let manager = SessionManager::new(100, Duration::from_secs(300));
        let identity = IdentityKeyPair::generate();
        
        let session = manager.create(
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();
        
        let id = session.id;
        assert_eq!(manager.count(), 1);
        
        manager.remove(&id);
        assert_eq!(manager.count(), 0);
        assert!(manager.get(&id).is_none());
    }
}
