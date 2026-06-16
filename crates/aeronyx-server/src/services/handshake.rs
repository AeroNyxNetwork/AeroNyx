// ============================================
// File: crates/aeronyx-server/src/services/handshake.rs
// ============================================
// Version: 1.0.0-Membership
//
// Modification Reason:
//   Injected Arc<DenyList> into HandshakeService.
//   process() now checks deny list before allocating IP or creating session.
//   Denied wallets receive an immediate error without consuming resources.
//   This prevents the 30-second reconnect loop where a quota-exceeded or
//   no-premium-access wallet reconnects before the next heartbeat fires.
//
// What changed:
//   - HandshakeService struct: added `deny_list: Arc<DenyList>`
//   - HandshakeService::new(): added `deny_list` parameter
//   - process(): Step 0 (new) — deny list check before any resource alloc
//   - All other steps unchanged
//   - Tests: added test_denied_wallet_rejected
//
// Main Logical Flow:
//   0. Check deny list → if denied, return WalletDenied immediately
//   1. Verify ClientHello signature
//   2. Allocate virtual IP
//   3. Generate session ID
//   4. Process cryptographic handshake (key exchange)
//   5. Create session
//   6. Register route
//   7. Return ServerHello
//
// ⚠️ Important Notes for Next Developer:
//   - Deny list check is BEFORE signature verification (Step 0 before Step 1).
//     Rationale: deny list check is O(1) DashMap lookup — cheaper than Ed25519
//     verify. Saves CPU on repeated reconnect attempts from denied wallets.
//   - wallet_hex is derived from client_hello.client_public_key (not yet a
//     full Session), so hex::encode is called once here. Acceptable since
//     this is not the hot path.
//   - ServerError::WalletDenied must be added to error.rs if not present.
//     Caller (server.rs UDP task) sends 0xFF RESET on any Err.
//
// Last Modified:
//   v0.1.0          - Initial handshake service
//   v1.0.0-Membership - Added DenyList check (Step 0)
// ============================================

use std::net::SocketAddr;
use std::sync::Arc;

use tracing::{debug, info, warn};

use aeronyx_core::crypto::handshake::{DefaultHandshakeCrypto, HandshakeCrypto};
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::{ClientHello, ServerHello};

use crate::error::{Result, ServerError};
use crate::services::deny_list::DenyList;
use crate::services::{IpPoolService, NodePolicyRuntime, RoutingService, Session, SessionManager};

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
    /// v1.0.0-Membership: deny list checked before any resource allocation.
    deny_list: Arc<DenyList>,
    /// Operator policy from nodeboard Settings.
    policy: Arc<NodePolicyRuntime>,
}

impl HandshakeService {
    pub fn new(
        server_identity: IdentityKeyPair,
        ip_pool: Arc<IpPoolService>,
        sessions: Arc<SessionManager>,
        routing: Arc<RoutingService>,
        deny_list: Arc<DenyList>,
        policy: Arc<NodePolicyRuntime>,
    ) -> Self {
        let crypto = DefaultHandshakeCrypto::new(server_identity);
        Self {
            crypto,
            ip_pool,
            sessions,
            routing,
            deny_list,
            policy,
        }
    }

    /// Processes a ClientHello and creates a session.
    ///
    /// ## Step 0 (v1.0.0-Membership)
    /// Checks the deny list before any resource allocation. Denied wallets
    /// receive an immediate error without consuming IP or session slots.
    /// This prevents the 30-second reconnect loop for quota-exceeded and
    /// no-premium-access wallets.
    pub fn process(
        &self,
        client_hello: &ClientHello,
        client_addr: SocketAddr,
    ) -> Result<HandshakeResult> {
        debug!(client = %client_addr, "Processing handshake");

        if let Err(reason) = self.policy.validate_new_session(self.sessions.count()) {
            warn!(
                client = %client_addr,
                reason = reason,
                "[NODE_POLICY] Handshake rejected"
            );
            return Err(ServerError::node_policy_rejected(reason));
        }

        // ── Step 0: Deny list check (v1.0.0-Membership) ──────────────────
        // Derive wallet hex from the public key in the ClientHello.
        // O(1) DashMap lookup — cheaper than Ed25519 verify below.
        let wallet_hex = hex::encode(&client_hello.client_public_key);
        if self.deny_list.is_denied(&wallet_hex) {
            let reason = self
                .deny_list
                .deny_reason(&wallet_hex)
                .map(|r| r.to_string())
                .unwrap_or_else(|| "denied".to_string());
            warn!(
                client = %client_addr,
                wallet = %&wallet_hex[..8],
                reason = %reason,
                "[HANDSHAKE] Wallet on deny list — rejected"
            );
            return Err(ServerError::WalletDenied { reason });
        }

        // ── Step 1: Verify ClientHello signature ──────────────────────────
        self.crypto.verify_client_hello(client_hello).map_err(|e| {
            warn!(client = %client_addr, error = %e, "Handshake signature verification failed");
            e
        })?;

        debug!(client = %client_addr, "ClientHello signature verified");

        // ── Step 2: Allocate virtual IP ───────────────────────────────────
        let virtual_ip = self.ip_pool.allocate().map_err(|e| {
            warn!(client = %client_addr, "IP allocation failed: {}", e);
            e
        })?;

        debug!(client = %client_addr, virtual_ip = %virtual_ip, "IP allocated");

        // ── Step 3: Generate session ID ───────────────────────────────────
        let session_id = aeronyx_common::SessionId::generate();

        debug!(client = %client_addr, session_id = %session_id, "Generated session ID");

        // ── Step 4: Cryptographic handshake ───────────────────────────────
        let (server_hello, session_key) = match self.crypto.process_handshake(
            client_hello,
            virtual_ip.octets(),
            *session_id.as_bytes(),
        ) {
            Ok(result) => result,
            Err(e) => {
                self.ip_pool.release(virtual_ip);
                warn!(client = %client_addr, error = %e, "Handshake crypto failed");
                return Err(e.into());
            }
        };

        // ── Step 5: Create session ────────────────────────────────────────
        let client_public_key = aeronyx_core::crypto::keys::IdentityPublicKey::from_bytes(
            &client_hello.client_public_key,
        )
        .map_err(|e| {
            self.ip_pool.release(virtual_ip);
            ServerError::session_creation_failed(format!("Invalid client public key: {}", e))
        })?;

        let session = match self.sessions.create(
            session_id.clone(),
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

        // ── Step 6: Register route ────────────────────────────────────────
        self.routing.add_route(virtual_ip, session.id.clone());

        info!(
            client     = %client_addr,
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
        debug!(session_id = %session_id, virtual_ip = %virtual_ip, "Cleaning up session resources");
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

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::deny_list::DenyReason;
    use crate::services::NodePolicySnapshot;
    use aeronyx_core::crypto::handshake::create_client_hello;
    use aeronyx_core::crypto::EphemeralKeyPair;
    use aeronyx_core::protocol::CURRENT_PROTOCOL_VERSION;
    use std::net::Ipv4Addr;
    use std::time::Duration;

    fn create_test_services() -> (
        Arc<IpPoolService>,
        Arc<SessionManager>,
        Arc<RoutingService>,
        Arc<DenyList>,
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
        let deny_list = Arc::new(DenyList::new());
        (ip_pool, sessions, routing, deny_list)
    }

    #[test]
    fn test_successful_handshake() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
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

        assert_eq!(
            result.session.id.as_bytes(),
            &result.response.session_id,
            "Session ID mismatch between Session and ServerHello!"
        );
        assert!(result.session.is_established());
        assert_eq!(result.session.client_endpoint, client_addr);
        assert!(ip_pool.is_allocated(result.session.virtual_ip));
        assert!(routing.has_route(result.session.virtual_ip));
        assert_eq!(sessions.count(), 1);
    }

    #[test]
    fn test_invalid_signature_rejected() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
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

    #[test]
    fn test_denied_wallet_rejected_before_ip_alloc() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );

        // Add wallet to deny list before handshake.
        let wallet_hex = hex::encode(client_identity.public_key_bytes());
        deny_list.add(&wallet_hex, DenyReason::QuotaExceeded);

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        let result = service.process(&client_hello, client_addr);

        assert!(result.is_err(), "Denied wallet must be rejected");
        // No resources consumed.
        assert_eq!(
            ip_pool.allocated_count(),
            0,
            "IP must not be allocated for denied wallet"
        );
        assert_eq!(
            sessions.count(),
            0,
            "Session must not be created for denied wallet"
        );
        assert!(
            routing.is_empty(),
            "Route must not be registered for denied wallet"
        );
    }

    #[test]
    fn test_removed_from_deny_list_can_connect() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );

        let wallet_hex = hex::encode(client_identity.public_key_bytes());
        deny_list.add(&wallet_hex, DenyReason::NoPremiumAccess);

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
            Arc::clone(&deny_list),
            Arc::new(NodePolicyRuntime::default()),
        );

        // Denied.
        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        assert!(service.process(&client_hello, client_addr).is_err());

        // Remove from deny list (simulating tier upgrade).
        deny_list.remove(&wallet_hex);

        // Now allowed.
        let result = service.process(&client_hello, client_addr);
        assert!(
            result.is_ok(),
            "Wallet removed from deny list must be allowed"
        );
        assert_eq!(sessions.count(), 1);
    }

    #[test]
    fn test_maintenance_policy_rejects_before_ip_alloc() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let policy = Arc::new(NodePolicyRuntime::default());
        policy.update(NodePolicySnapshot {
            maintenance_mode: true,
            ..NodePolicySnapshot::default()
        });

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
            deny_list,
            policy,
        );

        let client_identity = IdentityKeyPair::generate();
        let client_ephemeral = EphemeralKeyPair::generate();
        let client_hello = create_client_hello(
            &client_identity,
            client_ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );
        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        let result = service.process(&client_hello, client_addr);
        assert!(matches!(
            result,
            Err(ServerError::NodePolicyRejected { .. })
        ));
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn test_policy_max_sessions_rejects_before_local_limit() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let policy = Arc::new(NodePolicyRuntime::default());
        policy.update(NodePolicySnapshot {
            max_sessions: 1,
            ..NodePolicySnapshot::default()
        });

        let service = HandshakeService::new(
            server_identity,
            ip_pool.clone(),
            sessions.clone(),
            routing.clone(),
            deny_list,
            policy,
        );

        for index in 0..2 {
            let client_identity = IdentityKeyPair::generate();
            let client_ephemeral = EphemeralKeyPair::generate();
            let client_hello = create_client_hello(
                &client_identity,
                client_ephemeral.public_key_bytes(),
                CURRENT_PROTOCOL_VERSION,
            );
            let client_addr: SocketAddr = format!("127.0.0.1:{}", 12345 + index).parse().unwrap();
            let result = service.process(&client_hello, client_addr);
            if index == 0 {
                assert!(result.is_ok());
            } else {
                assert!(matches!(
                    result,
                    Err(ServerError::NodePolicyRejected { .. })
                ));
            }
        }

        assert_eq!(sessions.count(), 1);
    }
}
