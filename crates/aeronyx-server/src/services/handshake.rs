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

use aeronyx_core::crypto::handshake::{DefaultHandshakeCrypto, HandshakeCrypto};
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{ClientHello, ServerHello};

use crate::error::{Result, ServerError};
use crate::services::{IpPoolService, RoutingService, Session, SessionManager};

/// Result of a successful handshake.
pub struct HandshakeResult {
    pub session: Arc<Session>,
    pub response: ServerHello,
}

/// High-level handshake orchestration service.
pub struct HandshakeService {
    crypto: DefaultHandshakeCrypto,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    routing: Arc<RoutingService>,
}

impl HandshakeService {
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
        let session_id_bytes = aeronyx_common::SessionId::generate();
        let (server_hello, session_key) = match self.crypto.process_handshake(
            client_hello,
            virtual_ip.octets(),
            *session_id_bytes.as_bytes(),
        ) {
            Ok(result) => result,
            Err(e) => {
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
                self.ip_pool.release(virtual_ip);
                warn!(client = %client_addr, error = %e, "Session creation failed");
                return Err(e);
            }
        };

        // Step 5: Register route
        self.routing.add_route(virtual_ip, session.id.clone());

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
    pub fn cleanup(&self, session_id: &aeronyx_common::SessionId, virtual_ip: std::net::Ipv4Addr) {
        debug!(
            session_id = %session_id,
            virtual_ip = %virtual_ip,
            "Cleaning up session resources"
        );

        self.routing.remove_route(virtual_ip);
        self.ip_pool.release(virtual_ip);
        self.sessions.remove(session_id);
    }

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

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        let result = service.process(&client_hello, client_addr).unwrap();

        assert!(result.session.is_established());
        assert_eq!(result.session.client_endpoint, client_addr);
        assert!(ip_pool.is_allocated(result.session.virtual_ip));
        assert!(routing.has_route(result.session.virtual_ip));
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

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let mut client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );
        client_hello.signature[0] ^= 0xFF;

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        let result = service.process(&client_hello, client_addr);
        assert!(result.is_err());

        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }
}
