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
use tracing::{debug, info};

use aeronyx_common::time::AtomicInstant;
use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::keys::{IdentityPublicKey, SessionKey};

use crate::error::{Result, ServerError};

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

/// Session statistics.
#[derive(Debug, Default)]
pub struct SessionStats {
    pub bytes_rx: AtomicU64,
    pub bytes_tx: AtomicU64,
    pub packets_rx: AtomicU64,
    pub packets_tx: AtomicU64,
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

#[derive(Debug, Clone, Copy)]
pub struct StatsSnapshot {
    pub bytes_rx: u64,
    pub bytes_tx: u64,
    pub packets_rx: u64,
    pub packets_tx: u64,
}

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
        session_id: SessionId,  // ← 修改：接受外部传入的 SessionId
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

        // 使用传入的 session_id，而不是生成新的
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
            info!(
                session_id = %id,
                virtual_ip = %session.virtual_ip,
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
    }

    #[test]
    fn test_session_counters() {
        let session = create_test_session();
        
        assert_eq!(session.next_tx_counter(), 0);
        assert_eq!(session.next_tx_counter(), 1);
        
        assert!(session.validate_rx_counter(1));
        assert!(!session.validate_rx_counter(1));
        assert!(session.validate_rx_counter(5));
    }

    #[test]
    fn test_session_manager_create() {
        let manager = SessionManager::new(100, Duration::from_secs(300));
        let identity = IdentityKeyPair::generate();
        let session_id = SessionId::generate();  // ← 修改：先生成 ID
        
        let session = manager.create(
            session_id.clone(),  // ← 修改：传入 ID
            identity.public_key(),
            SessionKey::from_bytes([0x42; 32]),
            Ipv4Addr::new(100, 64, 0, 2),
            "127.0.0.1:12345".parse().unwrap(),
        ).unwrap();
        
        assert_eq!(manager.count(), 1);
        assert_eq!(session.id, session_id);  // ← 新增：验证 ID 一致
        
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
}
