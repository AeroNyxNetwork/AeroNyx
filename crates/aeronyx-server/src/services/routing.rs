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
use tracing::debug;

use aeronyx_common::types::SessionId;

use crate::error::{Result, ServerError};

/// Virtual IP to session routing service.
pub struct RoutingService {
    routes: DashMap<Ipv4Addr, SessionId>,
}

impl RoutingService {
    #[must_use]
    pub fn new() -> Self {
        Self {
            routes: DashMap::new(),
        }
    }

    /// Adds a route mapping.
    pub fn add_route(&self, virtual_ip: Ipv4Addr, session_id: SessionId) -> Option<SessionId> {
        let previous = self.routes.insert(virtual_ip, session_id.clone());
        
        if previous.is_some() {
            debug!(
                virtual_ip = %virtual_ip,
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
    pub fn remove_route(&self, virtual_ip: Ipv4Addr) -> Option<SessionId> {
        let removed = self.routes.remove(&virtual_ip).map(|(_, id)| id);
        
        if let Some(ref id) = removed {
            debug!(virtual_ip = %virtual_ip, session_id = %id, "Route removed");
        }

        removed
    }

    /// Looks up the session for a virtual IP.
    #[must_use]
    pub fn lookup(&self, virtual_ip: Ipv4Addr) -> Option<SessionId> {
        self.routes.get(&virtual_ip).map(|r| r.value().clone())
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
            .map(|r| (*r.key(), r.value().clone()))
            .collect()
    }

    /// Removes all routes for a specific session.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_lookup_route() {
        let routing = RoutingService::new();
        let session_id = SessionId::generate();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        let prev = routing.add_route(ip, session_id.clone());
        assert!(prev.is_none());

        let found = routing.lookup(ip);
        assert_eq!(found, Some(session_id));
        assert!(routing.has_route(ip));
    }

    #[test]
    fn test_lookup_nonexistent() {
        let routing = RoutingService::new();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        assert!(routing.lookup(ip).is_none());
        assert!(!routing.has_route(ip));
    }

    #[test]
    fn test_remove_route() {
        let routing = RoutingService::new();
        let session_id = SessionId::generate();
        let ip = Ipv4Addr::new(100, 64, 0, 2);

        routing.add_route(ip, session_id.clone());
        assert_eq!(routing.count(), 1);

        let removed = routing.remove_route(ip);
        assert_eq!(removed, Some(session_id));
        assert_eq!(routing.count(), 0);
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
