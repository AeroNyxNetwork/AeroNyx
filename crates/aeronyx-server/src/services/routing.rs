// ============================================
// File: crates/aeronyx-server/src/services/routing.rs
// ============================================
//! # Routing Service
//!
//! ## Creation Reason
//! Maps virtual IP addresses to sessions for packet routing,
//! enabling efficient lookups during data transfer.
//!
//! ## Main Functionality
//! - `RoutingService`: Virtual IP to session mapping
//! - Add/remove route entries
//! - Fast O(1) lookups using DashMap
//!
//! ## Routing Table Structure
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                 Routing Table                        │
//! ├─────────────────┬───────────────────────────────────┤
//! │  Virtual IP     │          Session ID               │
//! ├─────────────────┼───────────────────────────────────┤
//! │  100.64.0.2     │  abc123...                        │
//! │  100.64.0.3     │  def456...                        │
//! │  100.64.0.4     │  ghi789...                        │
//! └─────────────────┴───────────────────────────────────┘
//! ```
//!
//! ## Usage
//! ```ignore
//! let routing = RoutingService::new();
//!
//! // Register route when session is created
//! routing.add_route(virtual_ip, session_id);
//!
//! // Look up session for incoming packet
//! if let Some(session_id) = routing.lookup(destination_ip) {
//!     // Forward packet to session
//! }
//!
//! // Remove route when session closes
//! routing.remove_route(virtual_ip);
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Routes must be cleaned up when sessions close
//! - One virtual IP can only map to one session
//! - Thread-safe via DashMap
//!
//! ## Last Modified
//! v0.1.0 - Initial routing service

use std::net::Ipv4Addr;

use dashmap::DashMap;
use tracing::{debug, warn};

use aeronyx_common::types::SessionId;

use crate::error::{Result, ServerError};

// ============================================
// RoutingService
// ============================================

/// Virtual IP to session routing service.
///
/// # Thread Safety
/// Uses DashMap for lock-free concurrent access.
///
/// # Performance
/// - Lookup: O(1) average
/// - Insert: O(1) average
/// - Remove: O(1) average
pub struct RoutingService {
    /// Virtual IP → Session ID mapping.
    routes: DashMap<Ipv4Addr, SessionId>,
}

impl RoutingService {
    /// Creates a new routing service.
    #[must_use]
    pub fn new() -> Self {
        Self {
            routes: DashMap::new(),
        }
    }

    /// Adds a route mapping.
    ///
    /// # Arguments
    /// * `virtual_ip` - The virtual IP address
    /// * `session_id` - The session to route to
    ///
    /// # Returns
    /// The previous session ID if one was already mapped.
    pub fn add_route(&self, virtual_ip: Ipv4Addr, session_id: SessionId) -> Option<SessionId> {
        let previous = self.routes.insert(virtual_ip, session_id);
        
        if let Some(prev) = previous {
            warn!(
                virtual_ip = %virtual_ip,
                old_session = %prev,
                new_session = %session_id,
                "Route replaced"
            );
        } else {
            debug!(
                virtual_ip = %virtual_ip,
                session_id = %session_id,
                "Route added"
            );
        }

        previous
    }

    /// Removes a route mapping.
    ///
    /// # Arguments
    /// * `virtual_ip` - The virtual IP to remove
    ///
    /// # Returns
    /// The session ID that was mapped, if any.
    pub fn remove_route(&self, virtual_ip: Ipv4Addr) -> Option<SessionId> {
        let removed = self.routes.remove(&virtual_ip).map(|(_, id)| id);
        
        if let Some(id) = removed {
            debug!(virtual_ip = %virtual_ip, session_id = %id, "Route removed");
        }

        removed
    }

    /// Looks up the session for a virtual IP.
    ///
    /// # Arguments
    /// * `virtual_ip` - The destination virtual IP
    ///
    /// # Returns
    /// The session ID if a route exists.
    #[must_use]
    pub fn lookup(&self, virtual_ip: Ipv4Addr) -> Option<SessionId> {
        self.routes.get(&virtual_ip).map(|r| *r.value())
    }

    /// Looks up the session, returning error if not found.
    pub fn lookup_or_error(&self, virtual_ip: Ipv4Addr) -> Result<SessionId> {
        self.lookup(virtual_ip).ok_or_else(|| ServerError::NoRoute {
            destination: virtual_ip,
        })
    }

    /// Checks if a route exists for the given IP.
    #[must_use]
    pub fn has_route(&self, virtual_ip: Ipv4Addr) -> bool {
        self.routes.contains_key(&virtual_ip)
    }

    /// Returns the number of routes.
    #[must_use]
    pub fn count(&self) -> usize {
        self.routes.len()
    }

    /// Returns `true` if there are no routes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Clears all routes.
    pub fn clear(&self) {
        self.routes.clear();
        debug!("All routes cleared");
    }

    /// Returns all routes as a vector of (IP, SessionId) pairs.
    #[must_use]
    pub fn all_routes(&self) -> Vec<(Ipv4Addr, SessionId)> {
        self.routes
            .iter()
            .map(|r| (*r.key(), *r.value()))
            .collect()
    }

    /// Removes all routes for a specific session.
    ///
    /// # Returns
    /// The number of routes removed.
    pub fn remove_session_routes(&self, session_id: &SessionId) -> usize {
        let to_remove: Vec<Ipv4Addr> = self
            .routes
            .iter()
            .filter(|r| r.value() == session_id)
            .map(|r| *r.key())
            .collect();

        let count = to_remove.len();
        for ip in to_remove {
            self.routes.remove(&ip);
        }

        if count > 0 {
            debug!(session_id = %session_id, count, "Removed session routes");
        }

        count
    }
}

impl Default for RoutingService {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RoutingService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingService")
            .field("routes", &self.count())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_lookup_route() {
        let routing = RoutingService::new();
        let session_id = SessionId::generate();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        // Add route
        let prev = routing.add_route(ip, session_id);
        assert!(prev.is_none());

        // Lookup should succeed
        let found = routing.lookup(ip);
        assert_eq!(found, Some(session_id));

        // Has route should return true
        assert!(routing.has_route(ip));
    }

    #[test]
    fn test_lookup_nonexistent() {
        let routing = RoutingService::new();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        // Lookup should return None
        assert!(routing.lookup(ip).is_none());
        assert!(!routing.has_route(ip));

        // Error variant
        let result = routing.lookup_or_error(ip);
        assert!(matches!(result, Err(ServerError::NoRoute { .. })));
    }

    #[test]
    fn test_remove_route() {
        let routing = RoutingService::new();
        let session_id = SessionId::generate();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        routing.add_route(ip, session_id);
        assert_eq!(routing.count(), 1);

        let removed = routing.remove_route(ip);
        assert_eq!(removed, Some(session_id));
        assert_eq!(routing.count(), 0);
        assert!(!routing.has_route(ip));
    }

    #[test]
    fn test_replace_route() {
        let routing = RoutingService::new();
        let session1 = SessionId::generate();
        let session2 = SessionId::generate();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        // Add first route
        routing.add_route(ip, session1);
        
        // Replace with second
        let prev = routing.add_route(ip, session2);
        assert_eq!(prev, Some(session1));

        // Should now point to session2
        assert_eq!(routing.lookup(ip), Some(session2));
    }

    #[test]
    fn test_multiple_routes() {
        let routing = RoutingService::new();
        
        let s1 = SessionId::generate();
        let s2 = SessionId::generate();
        let s3 = SessionId::generate();

        routing.add_route(Ipv4Addr::new(100, 64, 0, 2), s1);
        routing.add_route(Ipv4Addr::new(100, 64, 0, 3), s2);
        routing.add_route(Ipv4Addr::new(100, 64, 0, 4), s3);

        assert_eq!(routing.count(), 3);
        assert_eq!(routing.lookup(Ipv4Addr::new(100, 64, 0, 2)), Some(s1));
        assert_eq!(routing.lookup(Ipv4Addr::new(100, 64, 0, 3)), Some(s2));
        assert_eq!(routing.lookup(Ipv4Addr::new(100, 64, 0, 4)), Some(s3));
    }

    #[test]
    fn test_remove_session_routes() {
        let routing = RoutingService::new();
        
        let s1 = SessionId::generate();
        let s2 = SessionId::generate();

        // Session 1 has 2 routes (unusual but possible)
        routing.add_route(Ipv4Addr::new(100, 64, 0, 2), s1);
        routing.add_route(Ipv4Addr::new(100, 64, 0, 3), s1);
        
        // Session 2 has 1 route
        routing.add_route(Ipv4Addr::new(100, 64, 0, 4), s2);

        assert_eq!(routing.count(), 3);

        // Remove all routes for session 1
        let removed = routing.remove_session_routes(&s1);
        assert_eq!(removed, 2);
        assert_eq!(routing.count(), 1);
        
        // Session 2's route should still exist
        assert!(routing.has_route(Ipv4Addr::new(100, 64, 0, 4)));
    }

    #[test]
    fn test_all_routes() {
        let routing = RoutingService::new();
        
        let s1 = SessionId::generate();
        let s2 = SessionId::generate();

        routing.add_route(Ipv4Addr::new(100, 64, 0, 2), s1);
        routing.add_route(Ipv4Addr::new(100, 64, 0, 3), s2);

        let routes = routing.all_routes();
        assert_eq!(routes.len(), 2);
    }

    #[test]
    fn test_clear() {
        let routing = RoutingService::new();
        
        routing.add_route(Ipv4Addr::new(100, 64, 0, 2), SessionId::generate());
        routing.add_route(Ipv4Addr::new(100, 64, 0, 3), SessionId::generate());

        assert_eq!(routing.count(), 2);

        routing.clear();
        assert_eq!(routing.count(), 0);
        assert!(routing.is_empty());
    }
}
