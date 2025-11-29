// ============================================
// File: crates/aeronyx-server/src/services/handshake.rs
// ============================================
//! # Handshake Service
//!
//! ## Creation Reason
//! Orchestrates the handshake process, coordinating between
//! cryptographic operations and session management.
//!
//! ## Main Functionality
//! - `HandshakeService`: High-level handshake orchestration
//! - Validates and processes ClientHello messages
//! - Creates sessions and generates ServerHello responses
//! - Integrates with IP pool and routing services
//!
//! ## Handshake Flow
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                    HandshakeService                          │
//! ├──────────────────────────────────────────────────────────────┤
//! │  1. Receive ClientHello                                      │
//! │     │                                                        │
//! │     ▼                                                        │
//! │  2. Verify signature (HandshakeCrypto)                       │
//! │     │                                                        │
//! │     ▼                                                        │
//! │  3. Allocate virtual IP (IpPoolService)                      │
//! │     │                                                        │
//! │     ▼                                                        │
//! │  4. Process handshake, derive keys (HandshakeCrypto)         │
//! │     │                                                        │
//! │     ▼                                                        │
//! │  5. Create session (SessionManager)                          │
//! │     │                                                        │
//! │     ▼                                                        │
//! │  6. Register route (RoutingService)                          │
//! │     │                                                        │
//! │     ▼                                                        │
//! │  7. Return ServerHello                                       │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Error Handling
//! - Signature failures: Drop packet (potential attack)
//! - IP exhaustion: Return error, client can retry
//! - Session limit: Return error, client can retry
//!
//! ## ⚠️ Important Note for Next Developer
//! - Handshake must be atomic (all or nothing)
//! - On failure, release any allocated resources
//! - Rate limiting should be implemented at transport layer
//!
//! ## Last Modified
//! v0.1.0 - Initial handshake service

use std::net::SocketAddr;
use std::sync::Arc;

use tracing::{debug, info, warn};

use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::handshake::{DefaultHandshakeCrypto, HandshakeCrypto};
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{ClientHello, ServerHello};

use crate::error::{Result, ServerError};
use crate::services::{IpPoolService, RoutingService, Session, SessionManager};

// ============================================
// HandshakeResult
// ============================================

/// Result of a successful handshake.
pub struct HandshakeResult {
    /// The created session.
    pub session: Arc<Session>,
    /// ServerHello response to send to client.
    pub response: ServerHello,
}

// ============================================
// HandshakeService
// ============================================

/// High-level handshake orchestration service.
///
/// # Responsibilities
/// - Coordinate cryptographic operations
/// - Manage resource allocation (IP, session)
/// - Ensure atomic handshake (rollback on failure)
pub struct HandshakeService {
    /// Cryptographic operations.
    crypto: DefaultHandshakeCrypto,
    /// IP address pool.
    ip_pool: Arc<IpPoolService>,
    /// Session manager.
    sessions: Arc<SessionManager>,
    /// Routing service.
    routing: Arc<RoutingService>,
}

impl HandshakeService {
    /// Creates a new handshake service.
    ///
    /// # Arguments
    /// * `server_identity` - Server's long-term identity key pair
    /// * `ip_pool` - IP address pool service
    /// * `sessions` - Session manager
    /// * `routing` - Routing service
    pub fn new(
        server_identity: IdentityKeyPair,
        ip_pool: Arc<IpPoolService>,
        sessions: Arc<SessionManager>,
        routing: Arc<RoutingService>,
    ) -> Self {
        let crypto = DefaultHandshakeCrypto::new(server_identity);
        Self {
            crypto,
            ip_pool,
            sessions,
            routing,
        }
    }

    /// Processes a ClientHello and creates a session.
    ///
    /// # Arguments
    /// * `client_hello` - The client's handshake message
    /// * `client_addr` - Client's UDP address
    ///
    /// # Returns
    /// The handshake result containing session and ServerHello.
    ///
    /// # Errors
    /// - Signature verification failure
    /// - IP pool exhausted
    /// - Session limit reached
    ///
    /// # Atomicity
    /// If any step fails, all allocated resources are released.
    pub fn process(
        &self,
        client_hello: &ClientHello,
        client_addr: SocketAddr,
    ) -> Result<HandshakeResult> {
        debug!(
            client = %client_addr,
            "Processing handshake"
        );

        // Step 1: Verify the ClientHello signature
        self.crypto.verify_client_hello(client_hello).map_err(|e| {
            warn!(client = %client_addr, error = %e, "Handshake signature verification failed");
            e
        })?;

        debug!(client = %client_addr, "ClientHello signature verified");

        // Step 2: Allocate virtual IP
        let virtual_ip = self.ip_pool.allocate().map_err(|e| {
            warn!(client = %client_addr, "IP allocation failed: {}", e);
            e
        })?;

        debug!(client = %client_addr, virtual_ip = %virtual_ip, "IP allocated");

        // Step 3: Process cryptographic handshake
        let session_id_bytes = SessionId::generate();
        let (server_hello, session_key) = match self.crypto.process_handshake(
            client_hello,
            virtual_ip.octets(),
            *session_id_bytes.as_bytes(),
        ) {
            Ok(result) => result,
            Err(e) => {
                // Rollback: release IP
                self.ip_pool.release(virtual_ip);
                warn!(client = %client_addr, error = %e, "Handshake crypto failed");
                return Err(e.into());
            }
        };

        // Step 4: Create session
        let client_public_key = aeronyx_core::crypto::keys::IdentityPublicKey::from_bytes(
            &client_hello.client_public_key,
        )
        .map_err(|e| {
            self.ip_pool.release(virtual_ip);
            ServerError::session_creation_failed(format!("Invalid client public key: {}", e))
        })?;

        let session = match self.sessions.create(
            client_public_key,
            session_key,
            virtual_ip,
            client_addr,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Rollback: release IP
                self.ip_pool.release(virtual_ip);
                warn!(client = %client_addr, error = %e, "Session creation failed");
                return Err(e);
            }
        };

        // Step 5: Register route
        self.routing.add_route(virtual_ip, session.id);

        info!(
            client = %client_addr,
            session_id = %session.id,
            virtual_ip = %virtual_ip,
            "Handshake completed successfully"
        );

        Ok(HandshakeResult {
            session,
            response: server_hello,
        })
    }

    /// Cleans up resources for a failed or closed session.
    ///
    /// # Arguments
    /// * `session_id` - The session to clean up
    /// * `virtual_ip` - The virtual IP to release
    pub fn cleanup(&self, session_id: &SessionId, virtual_ip: std::net::Ipv4Addr) {
        debug!(
            session_id = %session_id,
            virtual_ip = %virtual_ip,
            "Cleaning up session resources"
        );

        self.routing.remove_route(virtual_ip);
        self.ip_pool.release(virtual_ip);
        self.sessions.remove(session_id);
    }

    /// Returns the server's public key for clients to verify.
    #[must_use]
    pub fn server_public_key(&self) -> aeronyx_core::crypto::keys::IdentityPublicKey {
        self.crypto.public_key()
    }
}

impl std::fmt::Debug for HandshakeService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HandshakeService")
            .field("server_public_key", &self.server_public_key())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use aeronyx_core::crypto::handshake::create_client_hello;
    use aeronyx_core::crypto::EphemeralKeyPair;
    use aeronyx_core::protocol::CURRENT_PROTOCOL_VERSION;
    use std::net::Ipv4Addr;

    fn create_test_services() -> (
        Arc<IpPoolService>,
        Arc<SessionManager>,
        Arc<RoutingService>,
    ) {
        let ip_pool = Arc::new(
            IpPoolService::new(
                Ipv4Addr::new(100, 64, 0, 0),
                24,
                Ipv4Addr::new(100, 64, 0, 1),
            )
            .unwrap(),
        );
        let sessions = Arc::new(SessionManager::new(100, Duration::from_secs(300)));
        let routing = Arc::new(RoutingService::new());

        (ip_pool, sessions, routing)
    }

    #[test]
    fn test_successful_handshake() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing) = create_test_services();

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
        );

        // Client creates handshake
        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        // Process handshake
        let result = service.process(&client_hello, client_addr).unwrap();

        // Verify results
        assert!(result.session.is_established());
        assert_eq!(result.session.client_endpoint, client_addr);
        
        // Verify IP was allocated
        assert!(ip_pool.is_allocated(result.session.virtual_ip));
        
        // Verify route was added
        assert!(routing.has_route(result.session.virtual_ip));
        
        // Verify session exists
        assert_eq!(sessions.count(), 1);
    }

    #[test]
    fn test_invalid_signature_rejected() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing) = create_test_services();

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
        );

        // Create handshake with corrupted signature
        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let mut client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );
        client_hello.signature[0] ^= 0xFF; // Corrupt signature

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        // Should fail
        let result = service.process(&client_hello, client_addr);
        assert!(result.is_err());

        // No resources should be allocated
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn test_cleanup() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing) = create_test_services();

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
        );

        // Create a session
        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );

        let result = service
            .process(&client_hello, "127.0.0.1:12345".parse().unwrap())
            .unwrap();

        let session_id = result.session.id;
        let virtual_ip = result.session.virtual_ip;

        // Verify resources are allocated
        assert!(ip_pool.is_allocated(virtual_ip));
        assert!(routing.has_route(virtual_ip));
        assert_eq!(sessions.count(), 1);

        // Cleanup
        service.cleanup(&session_id, virtual_ip);

        // Verify resources are released
        assert!(!ip_pool.is_allocated(virtual_ip));
        assert!(!routing.has_route(virtual_ip));
        assert_eq!(sessions.count(), 0);
    }
}
