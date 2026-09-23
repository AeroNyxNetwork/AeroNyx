// ============================================
// File: crates/aeronyx-server/src/services/handshake.rs
// ============================================
// Version: 1.2.0-Admission
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
//   - v1.0.1 privacy hardening: handshake logs no longer persist client IPs,
//     session IDs, wallet prefixes, or virtual IP assignments.
//   - v1.1.0 cleanup hardening: route/IP release is conditional on session
//     ownership, making delayed or duplicated cleanup safe after address reuse.
//   - [POLICY-ADMISSION 2026-07-29 by Codex] Final policy validation and
//     session resource admission are serialized so concurrent handshakes
//     cannot exceed the operator's dynamic max_sessions limit.
//   - [PROTOCOL-V2-ADMISSION-HOTFIX 2026-09-13 by Codex] V1 and V2 now share
//     deny/policy admission and the same mutation fence; identity eviction is
//     performed only after the claimed client key authenticates successfully.
//   - [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] V2 diagnostics expose
//     only fixed outcome classes and aggregate counters, never client/socket,
//     key, signature, session, or virtual-address metadata.
//   - [V2-HANDSHAKE-CLOSED-VERSION-ADMISSION 2026-09-21 by Codex] Reject
//     unknown protocol versions before policy, cryptography, or resource state.
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
//   - [HANDSHAKE-CLEANUP 2026-07-29 by Codex] Never release a virtual IP from
//     session cleanup unless the route or removed session proves ownership.
//     Retry-safe cleanup must not tear down a replacement session.
//
// Last Modified:
//   v0.1.0          - Initial handshake service
//   v1.0.0-Membership - Added DenyList check (Step 0)
//   v1.0.1-PrivacyLogs - Redacted handshake/session correlation metadata
//   v1.1.0-OwnershipCleanup - Conditional route removal and IP release
//   v1.2.0-Admission - Made dynamic policy admission atomic across handshakes
//   v1.3.0-AuthenticatedAdmission - Unified V1/V2 gates and post-auth eviction
//   v1.3.1-V2LogPrivacy - Redacted V2 handshake correlation metadata
//   v1.3.2-ClosedVersionAdmission - Fail closed on every non-V1/V2 hello
// ============================================

use std::net::SocketAddr;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::{debug, info, warn};

use aeronyx_core::crypto::handshake::{DefaultHandshakeCrypto, HandshakeCrypto};
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::error::CoreError;
use aeronyx_core::protocol::version::{
    classify_supported_protocol_version, SupportedProtocolVersion,
};
use aeronyx_core::protocol::{ClientHello, ServerHello, CURRENT_PROTOCOL_VERSION};

use crate::error::{Result, ServerError};
use crate::services::deny_list::DenyList;
use crate::services::{IpPoolService, NodePolicyRuntime, RoutingService, Session, SessionManager};

/// Maximum live transport sessions admitted for one authenticated identity.
const MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY: usize = 4;

/// Result of a successful handshake.
pub struct HandshakeResult {
    pub session: Arc<Session>,
    pub response: ServerHello,
}

/// High-level handshake orchestration service.
pub struct HandshakeService {
    // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] Keep the cryptographic
    // boundary replaceable in same-module tests so every otherwise-unreachable
    // fail-closed diagnostic branch can be exercised without weakening the
    // production implementation.
    crypto: Box<dyn HandshakeCrypto>,
    ip_pool: Arc<IpPoolService>,
    sessions: Arc<SessionManager>,
    routing: Arc<RoutingService>,
    /// v1.0.0-Membership: deny list checked before any resource allocation.
    deny_list: Arc<DenyList>,
    /// Operator policy from nodeboard Settings.
    policy: Arc<NodePolicyRuntime>,
    /// Serializes the final policy check with IP/session/route admission.
    ///
    /// Signature verification remains outside this lock, so invalid or
    /// attacker-controlled handshakes cannot monopolize the admission path.
    admission_lock: Mutex<()>,
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
        let crypto = Box::new(DefaultHandshakeCrypto::new(server_identity));
        Self {
            crypto,
            ip_pool,
            sessions,
            routing,
            deny_list,
            policy,
            admission_lock: Mutex::new(()),
        }
    }

    #[cfg(test)]
    fn new_with_crypto(
        crypto: Box<dyn HandshakeCrypto>,
        ip_pool: Arc<IpPoolService>,
        sessions: Arc<SessionManager>,
        routing: Arc<RoutingService>,
        deny_list: Arc<DenyList>,
        policy: Arc<NodePolicyRuntime>,
    ) -> Self {
        Self {
            crypto,
            ip_pool,
            sessions,
            routing,
            deny_list,
            policy,
            admission_lock: Mutex::new(()),
        }
    }

    /// Processes a ClientHello and creates a session.
    ///
    /// ## Step 0 (v1.0.0-Membership)
    /// Checks the deny list before any resource allocation. Denied wallets
    /// receive an immediate error without consuming IP or session slots.
    /// This prevents the 30-second reconnect loop for quota-exceeded and
    /// no-premium-access wallets.
    /// `extension` is the byte string that followed the fixed hello on the
    /// wire (the voucher block, or empty). A v0x02 signature covers it.
    pub fn process(
        &self,
        client_hello: &ClientHello,
        extension: &[u8],
        client_addr: SocketAddr,
    ) -> Result<HandshakeResult> {
        // [VPN-PROTOCOL-VERSION-POLICY 2026-09-23 by Codex] Classify through
        // the core policy before policy, cryptography, or resource mutation.
        // Exhaustive dispatch prevents a newly admitted version from silently
        // inheriting the legacy symmetric-key path.
        let supported_version = match classify_supported_protocol_version(client_hello.version) {
            Some(supported_version) => supported_version,
            None => {
                let got = client_hello.version;
                return Err(CoreError::UnsupportedVersion {
                    got,
                    expected: CURRENT_PROTOCOL_VERSION,
                }
                .into());
            }
        };
        self.validate_candidate(client_hello)?;
        match supported_version {
            SupportedProtocolVersion::V2 => {
                return self.process_v2(client_hello, extension, client_addr);
            }
            SupportedProtocolVersion::V1 => {}
        }
        debug!("Processing handshake");

        // ── Step 1: Verify ClientHello signature ──────────────────────────
        self.crypto.verify_client_hello(client_hello).map_err(|e| {
            warn!(error = %e, "Handshake signature verification failed");
            e
        })?;

        debug!("ClientHello signature verified");

        // [POLICY-ADMISSION 2026-07-29 by Codex] The earlier validation is a
        // cheap fast rejection. Re-check while holding the admission lock so
        // multiple valid handshakes cannot all observe the same free slot and
        // overshoot nodeboard's dynamic max_sessions policy.
        let _admission_guard = self.admission_lock.lock();
        self.validate_candidate(client_hello)?;

        // ── Step 2: Allocate virtual IP ───────────────────────────────────
        let virtual_ip = self.ip_pool.allocate().map_err(|e| {
            warn!("IP allocation failed: {}", e);
            e
        })?;

        debug!("IP allocated for handshake");

        // ── Step 3: Generate session ID ───────────────────────────────────
        let session_id = aeronyx_common::SessionId::generate();

        debug!("Generated session ID");

        // ── Step 4: Cryptographic handshake ───────────────────────────────
        let (server_hello, session_key) = match self.crypto.process_handshake(
            client_hello,
            virtual_ip.octets(),
            *session_id.as_bytes(),
        ) {
            Ok(result) => result,
            Err(e) => {
                self.ip_pool.release(virtual_ip);
                warn!(error = %e, "Handshake crypto failed");
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

        let admission = match self.sessions.create_authenticated(
            session_id.clone(),
            client_public_key,
            session_key,
            virtual_ip,
            client_addr,
            MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY,
        ) {
            Ok(admission) => admission,
            Err(e) => {
                self.ip_pool.release(virtual_ip);
                warn!(error = %e, "Session creation failed");
                return Err(e);
            }
        };

        // ── Step 6: Register route ────────────────────────────────────────
        for termination in &admission.evicted {
            self.routing
                .remove_route_for_session(termination.virtual_ip, &termination.session_id);
        }
        self.routing
            .add_route(virtual_ip, admission.session.id.clone());
        admission
            .session
            .stage_admission_evictions(admission.evicted);

        info!("Handshake completed successfully");

        Ok(HandshakeResult {
            session: admission.session,
            response: server_hello,
        })
    }

    /// Shared, coarse admission gate used before signature work and again
    /// under `admission_lock` immediately before resource mutation.
    fn validate_candidate(&self, client_hello: &ClientHello) -> Result<()> {
        // [PROTOCOL-V2-ADMISSION-HOTFIX 2026-09-13 by Codex] Version dispatch
        // must not bypass operator policy or membership denial. Keeping this
        // helper common also preserves v0x01 behavior during mixed rollout.
        if let Err(reason) = self.policy.validate_new_session(self.sessions.count()) {
            warn!(reason = reason, "[NODE_POLICY] Handshake rejected");
            return Err(ServerError::node_policy_rejected(reason));
        }

        let wallet_hex = hex::encode(client_hello.client_public_key);
        if self.deny_list.is_denied(&wallet_hex) {
            let reason = self
                .deny_list
                .deny_reason(&wallet_hex)
                .map(|reason| reason.to_string())
                .unwrap_or_else(|| "denied".to_string());
            warn!(reason = %reason, "[HANDSHAKE] Wallet on deny list — rejected");
            return Err(ServerError::WalletDenied { reason });
        }
        Ok(())
    }

    /// Cleans up resources for a failed or closed session.
    /// v0x02: transcript-bound hellos, per-direction keys.
    fn process_v2(
        &self,
        client_hello: &ClientHello,
        extension: &[u8],
        client_addr: SocketAddr,
    ) -> Result<HandshakeResult> {
        self.crypto
            .verify_client_hello_v2(client_hello, extension)
            .map_err(|e| {
                // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] The returned
                // typed error remains intact, but an attacker-controlled hello
                // must not turn socket, key, signature, or timestamp material
                // into persistent diagnostics.
                warn!(
                    event = "handshake_v2_rejected",
                    reason = "authentication",
                    "Handshake v2 rejected"
                );
                e
            })?;
        let _admission_guard = self.admission_lock.lock();
        self.validate_candidate(client_hello)?;
        let virtual_ip = self.ip_pool.allocate().map_err(|e| {
            warn!(
                event = "handshake_v2_rejected",
                reason = "ip_capacity",
                "Handshake v2 rejected"
            );
            e
        })?;
        let session_id = aeronyx_common::SessionId::generate();
        let (server_hello, keys) = match self.crypto.process_handshake_v2(
            client_hello,
            extension,
            virtual_ip.octets(),
            *session_id.as_bytes(),
        ) {
            Ok(result) => result,
            Err(e) => {
                self.ip_pool.release(virtual_ip);
                warn!(
                    event = "handshake_v2_rejected",
                    reason = "crypto_state",
                    "Handshake v2 rejected"
                );
                return Err(e.into());
            }
        };
        let client_public_key = aeronyx_core::crypto::keys::IdentityPublicKey::from_bytes(
            &client_hello.client_public_key,
        )
        .map_err(|e| {
            self.ip_pool.release(virtual_ip);
            ServerError::session_creation_failed(format!("Invalid client public key: {}", e))
        })?;
        let admission = match self.sessions.create_v2_authenticated(
            session_id.clone(),
            client_public_key,
            keys,
            virtual_ip,
            client_addr,
            MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY,
        ) {
            Ok(admission) => admission,
            Err(e) => {
                self.ip_pool.release(virtual_ip);
                warn!(
                    event = "handshake_v2_rejected",
                    reason = "session_admission",
                    "Handshake v2 rejected"
                );
                return Err(e);
            }
        };
        let evicted_sessions = admission.evicted.len();
        for termination in &admission.evicted {
            self.routing
                .remove_route_for_session(termination.virtual_ip, &termination.session_id);
        }
        self.routing
            .add_route(virtual_ip, admission.session.id.clone());
        admission
            .session
            .stage_admission_evictions(admission.evicted);
        info!(
            event = "handshake_v2_accepted",
            protocol_version = aeronyx_core::protocol::PROTOCOL_VERSION_V2,
            active_sessions = self.sessions.count(),
            evicted_sessions,
            "Handshake v2 completed successfully"
        );
        Ok(HandshakeResult {
            session: admission.session,
            response: server_hello,
        })
    }

    pub fn cleanup(&self, session_id: &aeronyx_common::SessionId, virtual_ip: std::net::Ipv4Addr) {
        debug!("Cleaning up session resources");
        let removed_route = self
            .routing
            .remove_route_for_session(virtual_ip, session_id)
            .is_some();
        let removed_session = self.sessions.remove(session_id);
        let removed_session_owned_ip = removed_session
            .as_ref()
            .is_some_and(|session| session.virtual_ip == virtual_ip);

        // A surviving route proves that the IP now belongs to another session.
        // When neither source confirms ownership, this is duplicate/stale
        // cleanup and must be a no-op rather than releasing a reused address.
        if (removed_route || removed_session_owned_ip) && !self.routing.has_route(virtual_ip) {
            self.ip_pool.release(virtual_ip);
        } else {
            debug!("Session cleanup skipped unowned or replacement virtual IP");
        }
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
    use aeronyx_core::crypto::handshake::{create_client_hello, create_client_hello_v2};
    use aeronyx_core::crypto::kdf::SessionKeys;
    use aeronyx_core::crypto::keys::{IdentityPublicKey, SessionKey};
    use aeronyx_core::crypto::EphemeralKeyPair;
    // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] These legacy fixtures
    // exercise the signed v1 control path even after v2 becomes the default.
    use aeronyx_core::protocol::PROTOCOL_VERSION_V1;
    use aeronyx_core::CoreError;
    use std::io::{self, Write};
    use std::net::Ipv4Addr;
    use std::sync::{Barrier, Mutex as StdMutex};
    use std::thread;
    use std::time::Duration;
    use tracing_subscriber::{fmt::MakeWriter, prelude::*};

    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<StdMutex<Vec<u8>>>);

    struct CapturedLogWriter(Arc<StdMutex<Vec<u8>>>);

    impl Write for CapturedLogWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .expect("captured log mutex")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'writer> MakeWriter<'writer> for CapturedLogs {
        type Writer = CapturedLogWriter;

        fn make_writer(&'writer self) -> Self::Writer {
            CapturedLogWriter(Arc::clone(&self.0))
        }
    }

    fn capture_info_logs<T>(operation: impl FnOnce() -> T) -> (T, String) {
        let captured = CapturedLogs::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .without_time()
            .with_writer(captured.clone())
            .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
                metadata.level() <= &tracing::Level::INFO
                    && metadata.target().ends_with("services::handshake")
            }));
        let subscriber = tracing_subscriber::registry().with(layer);
        let result = tracing::subscriber::with_default(subscriber, operation);
        let logs = String::from_utf8(captured.0.lock().expect("captured log mutex").clone())
            .expect("captured logs are UTF-8");
        (result, logs)
    }

    fn assert_log_omits(logs: &str, forbidden: &[String]) {
        for value in forbidden {
            assert!(
                !value.is_empty() && !logs.contains(value),
                "sensitive value leaked into logs: {value} in {logs}"
            );
        }
    }

    fn assert_v2_log_schema_is_coarse(logs: &str) {
        for forbidden_field in [
            "client=",
            "client_addr",
            "endpoint",
            "session_id",
            "virtual_ip",
            "public_key",
            "signature",
            "route_id",
            "message_id",
            "payload",
            "nonce",
            "key=",
            "error=",
        ] {
            assert!(
                !logs.contains(forbidden_field),
                "forbidden diagnostic field {forbidden_field} in {logs}"
            );
        }
    }

    struct FailingV2Crypto {
        inner: DefaultHandshakeCrypto,
    }

    impl HandshakeCrypto for FailingV2Crypto {
        fn public_key(&self) -> IdentityPublicKey {
            self.inner.public_key()
        }

        fn verify_client_hello(&self, message: &ClientHello) -> aeronyx_core::Result<()> {
            self.inner.verify_client_hello(message)
        }

        fn process_handshake(
            &self,
            client_hello: &ClientHello,
            assigned_ip: [u8; 4],
            session_id: [u8; 16],
        ) -> aeronyx_core::Result<(ServerHello, SessionKey)> {
            self.inner
                .process_handshake(client_hello, assigned_ip, session_id)
        }

        fn verify_client_hello_v2(
            &self,
            message: &ClientHello,
            extension: &[u8],
        ) -> aeronyx_core::Result<()> {
            self.inner.verify_client_hello_v2(message, extension)
        }

        fn process_handshake_v2(
            &self,
            _client_hello: &ClientHello,
            _extension: &[u8],
            _assigned_ip: [u8; 4],
            _session_id: [u8; 16],
        ) -> aeronyx_core::Result<(ServerHello, SessionKeys)> {
            Err(CoreError::KeyDerivation {
                reason: "SENSITIVE_CRYPTO_SENTINEL".to_string(),
            })
        }
    }

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

    fn create_v2_hello(identity: &IdentityKeyPair) -> ClientHello {
        create_client_hello_v2(
            identity,
            EphemeralKeyPair::generate().public_key_bytes(),
            &[],
        )
    }

    #[test]
    fn v2_success_logs_only_fixed_classes_and_aggregate_counters() {
        // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] Capture a real signed
        // v2 success and prove that every correlation value stays out of the
        // formatted event while coarse operational counters remain available.
        let server_identity = IdentityKeyPair::generate();
        let server_public = server_identity.public_key_bytes();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new(
            server_identity,
            ip_pool,
            sessions,
            routing,
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let client_identity = IdentityKeyPair::generate();
        let hello = create_v2_hello(&client_identity);
        let client_addr: SocketAddr = "198.51.100.77:45678".parse().unwrap();

        let (result, logs) = capture_info_logs(|| {
            service
                .process(&hello, &[], client_addr)
                .expect("signed v2 handshake")
        });

        assert!(logs.contains("handshake_v2_accepted"));
        assert!(logs.contains("protocol_version=2"));
        assert!(logs.contains("active_sessions=1"));
        assert!(logs.contains("evicted_sessions=0"));
        assert_v2_log_schema_is_coarse(&logs);
        assert_log_omits(
            &logs,
            &[
                client_addr.to_string(),
                hex::encode(client_identity.public_key_bytes()),
                hex::encode(server_public),
                hex::encode(hello.signature),
                result.session.id.to_string(),
                result.session.virtual_ip.to_string(),
            ],
        );
    }

    #[test]
    fn v2_rejection_logs_only_fixed_authentication_class() {
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let client_identity = IdentityKeyPair::generate();
        let mut forged = create_v2_hello(&client_identity);
        forged.signature = [0xA5; 64];
        let client_addr: SocketAddr = "203.0.113.91:54321".parse().unwrap();

        let (result, logs) = capture_info_logs(|| service.process(&forged, &[], client_addr));

        assert!(result.is_err());
        assert!(logs.contains("handshake_v2_rejected"));
        assert!(logs.contains("reason=\"authentication\""));
        assert_v2_log_schema_is_coarse(&logs);
        assert_log_omits(
            &logs,
            &[
                client_addr.to_string(),
                hex::encode(client_identity.public_key_bytes()),
                hex::encode(forged.signature),
                forged.timestamp.to_string(),
            ],
        );
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn v2_allocation_failure_logs_only_capacity_class() {
        let ip_pool = Arc::new(
            IpPoolService::new(
                Ipv4Addr::new(100, 64, 0, 0),
                30,
                Ipv4Addr::new(100, 64, 0, 1),
            )
            .unwrap(),
        );
        assert_eq!(ip_pool.allocate().unwrap(), Ipv4Addr::new(100, 64, 0, 2));
        let sessions = Arc::new(SessionManager::new(4, Duration::from_secs(300)));
        let routing = Arc::new(RoutingService::new());
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            Arc::new(DenyList::new()),
            Arc::new(NodePolicyRuntime::default()),
        );
        let identity = IdentityKeyPair::generate();
        let hello = create_v2_hello(&identity);
        let client_addr: SocketAddr = "198.51.100.88:46000".parse().unwrap();

        let (result, logs) = capture_info_logs(|| service.process(&hello, &[], client_addr));

        assert!(matches!(result, Err(ServerError::IpPoolExhausted)));
        assert!(logs.contains("handshake_v2_rejected"));
        assert!(logs.contains("reason=\"ip_capacity\""));
        assert!(!logs.contains("IP pool exhausted"));
        assert_v2_log_schema_is_coarse(&logs);
        assert_log_omits(
            &logs,
            &[
                client_addr.to_string(),
                hex::encode(identity.public_key_bytes()),
                hex::encode(hello.signature),
            ],
        );
        assert_eq!(ip_pool.allocated_count(), 1);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn v2_crypto_failure_logs_only_crypto_state_class() {
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new_with_crypto(
            Box::new(FailingV2Crypto {
                inner: DefaultHandshakeCrypto::new(IdentityKeyPair::generate()),
            }),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let identity = IdentityKeyPair::generate();
        let hello = create_v2_hello(&identity);
        let client_addr: SocketAddr = "203.0.113.92:47000".parse().unwrap();

        let (result, logs) = capture_info_logs(|| service.process(&hello, &[], client_addr));

        assert!(result.is_err());
        assert!(logs.contains("handshake_v2_rejected"));
        assert!(logs.contains("reason=\"crypto_state\""));
        assert!(!logs.contains("SENSITIVE_CRYPTO_SENTINEL"));
        assert_v2_log_schema_is_coarse(&logs);
        assert_log_omits(
            &logs,
            &[
                client_addr.to_string(),
                hex::encode(identity.public_key_bytes()),
                hex::encode(hello.signature),
                Ipv4Addr::new(100, 64, 0, 2).to_string(),
            ],
        );
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn v2_session_rejection_logs_only_admission_class() {
        let ip_pool = Arc::new(
            IpPoolService::new(
                Ipv4Addr::new(100, 64, 0, 0),
                24,
                Ipv4Addr::new(100, 64, 0, 1),
            )
            .unwrap(),
        );
        let sessions = Arc::new(SessionManager::new(0, Duration::from_secs(300)));
        let routing = Arc::new(RoutingService::new());
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            Arc::new(DenyList::new()),
            Arc::new(NodePolicyRuntime::default()),
        );
        let identity = IdentityKeyPair::generate();
        let hello = create_v2_hello(&identity);
        let client_addr: SocketAddr = "192.0.2.93:48000".parse().unwrap();

        let (result, logs) = capture_info_logs(|| service.process(&hello, &[], client_addr));

        assert!(matches!(
            result,
            Err(ServerError::SessionLimitReached { limit: 0 })
        ));
        assert!(logs.contains("handshake_v2_rejected"));
        assert!(logs.contains("reason=\"session_admission\""));
        assert!(!logs.contains("Session limit reached"));
        assert_v2_log_schema_is_coarse(&logs);
        assert_log_omits(
            &logs,
            &[
                client_addr.to_string(),
                hex::encode(identity.public_key_bytes()),
                hex::encode(hello.signature),
                Ipv4Addr::new(100, 64, 0, 2).to_string(),
            ],
        );
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn legacy_success_and_rejection_keep_privacy_safe_diagnostics() {
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            ip_pool,
            sessions,
            routing,
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let client_identity = IdentityKeyPair::generate();
        let client_addr: SocketAddr = "192.0.2.44:40404".parse().unwrap();
        let hello = create_client_hello(
            &client_identity,
            EphemeralKeyPair::generate().public_key_bytes(),
            PROTOCOL_VERSION_V1,
        );
        let (accepted, accepted_logs) =
            capture_info_logs(|| service.process(&hello, &[], client_addr));
        let accepted = accepted.expect("signed v1 handshake");
        assert!(accepted_logs.contains("Handshake completed successfully"));
        assert_log_omits(
            &accepted_logs,
            &[
                client_addr.to_string(),
                hex::encode(client_identity.public_key_bytes()),
                hex::encode(hello.signature),
                accepted.session.id.to_string(),
                accepted.session.virtual_ip.to_string(),
            ],
        );

        let mut forged = hello;
        forged.signature = [0x5A; 64];
        let rejected_addr: SocketAddr = "192.0.2.45:40405".parse().unwrap();
        let (rejected, rejected_logs) =
            capture_info_logs(|| service.process(&forged, &[], rejected_addr));
        assert!(rejected.is_err());
        assert!(rejected_logs.contains("Handshake signature verification failed"));
        assert_log_omits(
            &rejected_logs,
            &[
                rejected_addr.to_string(),
                hex::encode(client_identity.public_key_bytes()),
                hex::encode(forged.signature),
            ],
        );
    }

    #[test]
    fn unsupported_signed_versions_fail_before_every_admission_mutation() {
        // [V2-HANDSHAKE-CLOSED-VERSION-ADMISSION 2026-09-21 by Codex]
        // A valid signature over an unknown version used to fall through to
        // v1. Exercise values on both sides of the frozen {1,2} set and prove
        // the rejection neither reserves an IP nor creates a session/route.
        for version in [0, 3, u8::MAX] {
            let (ip_pool, sessions, routing, deny_list) = create_test_services();
            let service = HandshakeService::new(
                IdentityKeyPair::generate(),
                Arc::clone(&ip_pool),
                Arc::clone(&sessions),
                Arc::clone(&routing),
                deny_list,
                Arc::new(NodePolicyRuntime::default()),
            );
            let identity = IdentityKeyPair::generate();
            let hello = create_client_hello(
                &identity,
                EphemeralKeyPair::generate().public_key_bytes(),
                version,
            );
            let client_addr: SocketAddr = "192.0.2.99:40999".parse().unwrap();

            let result = service.process(&hello, &[], client_addr);
            assert!(matches!(
                result,
                Err(ServerError::Core(CoreError::UnsupportedVersion {
                    got,
                    expected: CURRENT_PROTOCOL_VERSION,
                })) if got == version
            ));
            assert_eq!(ip_pool.allocated_count(), 0);
            assert_eq!(sessions.count(), 0);
            assert_eq!(routing.count(), 0);
        }
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
            PROTOCOL_VERSION_V1,
        );

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        let result = service.process(&client_hello, &[], client_addr).unwrap();

        assert_eq!(
            result.session.id.as_bytes(),
            &result.response.session_id,
            "Session ID mismatch between Session and ServerHello!"
        );
        assert!(result.session.is_established());
        assert_eq!(result.session.endpoint(), client_addr);
        assert!(ip_pool.is_allocated(result.session.virtual_ip));
        assert!(routing.has_route(result.session.virtual_ip));
        assert_eq!(sessions.count(), 1);
    }

    #[test]
    fn test_stale_cleanup_preserves_replacement_session() {
        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new(
            server_identity,
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        let first_identity = IdentityKeyPair::generate();
        let first_hello = create_client_hello(
            &first_identity,
            EphemeralKeyPair::generate().public_key_bytes(),
            PROTOCOL_VERSION_V1,
        );
        let first = service.process(&first_hello, &[], client_addr).unwrap();
        let old_session_id = first.session.id.clone();
        let reused_ip = first.session.virtual_ip;

        service.cleanup(&old_session_id, reused_ip);
        assert!(!ip_pool.is_allocated(reused_ip));
        assert!(routing.lookup(reused_ip).is_none());
        assert!(sessions.get(&old_session_id).is_none());

        let replacement_identity = IdentityKeyPair::generate();
        let replacement_hello = create_client_hello(
            &replacement_identity,
            EphemeralKeyPair::generate().public_key_bytes(),
            PROTOCOL_VERSION_V1,
        );
        let replacement = service
            .process(&replacement_hello, &[], "127.0.0.1:12346".parse().unwrap())
            .unwrap();
        assert_eq!(replacement.session.virtual_ip, reused_ip);

        // A delayed retry for the old cleanup must not remove or release the
        // route now owned by the replacement session.
        service.cleanup(&old_session_id, reused_ip);

        assert!(ip_pool.is_allocated(reused_ip));
        assert_eq!(
            routing.lookup(reused_ip),
            Some(replacement.session.id.clone())
        );
        assert!(sessions.get(&replacement.session.id).is_some());
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
            PROTOCOL_VERSION_V1,
        );
        client_hello.signature[0] ^= 0xFF;

        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();
        let result = service.process(&client_hello, &[], client_addr);
        assert!(result.is_err());
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
    }

    #[test]
    fn invalid_v2_signature_cannot_evict_claimed_identity_sessions() {
        // [PROTOCOL-V2-ADMISSION-HOTFIX 2026-09-13 by Codex] The identity cap
        // is downstream of signature verification and reads the authenticated
        // primary session map, never the untrusted hello or wallet index.
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let victim = IdentityKeyPair::generate();
        let mut retained = Vec::new();
        for offset in 0..MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY {
            let result = service
                .process(
                    &create_v2_hello(&victim),
                    &[],
                    SocketAddr::from(([127, 0, 0, 1], 24_000 + offset as u16)),
                )
                .expect("seed authenticated victim session");
            retained.push(result.session.id.clone());
        }

        let attacker = IdentityKeyPair::generate();
        let mut forged = create_v2_hello(&attacker);
        forged.client_public_key = victim.public_key_bytes();
        let before_allocated = ip_pool.allocated_count();
        let rejected = service.process(
            &forged,
            &[],
            "127.0.0.1:24999".parse().expect("forged endpoint"),
        );

        assert!(rejected.is_err());
        assert_eq!(sessions.count(), retained.len());
        assert_eq!(routing.count(), retained.len());
        assert_eq!(ip_pool.allocated_count(), before_allocated);
        assert!(retained.iter().all(|id| sessions.get(id).is_some()));
    }

    #[test]
    fn fifth_authenticated_v2_session_atomically_replaces_oldest_identity_session() {
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            ip_pool,
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );
        let identity = IdentityKeyPair::generate();
        let mut admitted = Vec::new();
        for offset in 0..MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY {
            admitted.push(
                service
                    .process(
                        &create_v2_hello(&identity),
                        &[],
                        SocketAddr::from(([127, 0, 0, 1], 25_000 + offset as u16)),
                    )
                    .expect("seed identity session"),
            );
        }
        let oldest = admitted[0].session.id.clone();

        let replacement = service
            .process(
                &create_v2_hello(&identity),
                &[],
                "127.0.0.1:25999".parse().expect("replacement endpoint"),
            )
            .expect("admit authenticated replacement");

        let evicted = replacement.session.take_admission_evictions();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].session_id, oldest);
        assert!(sessions.get(&oldest).is_none());
        assert!(sessions.get(&replacement.session.id).is_some());
        assert_eq!(sessions.count(), MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY);
        assert_eq!(routing.count(), MAX_AUTHENTICATED_SESSIONS_PER_IDENTITY);
        assert!(routing.lookup(evicted[0].virtual_ip).is_none());
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
            PROTOCOL_VERSION_V1,
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
        let result = service.process(&client_hello, &[], client_addr);

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
    fn denied_v2_identity_is_rejected_before_resource_allocation() {
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let identity = IdentityKeyPair::generate();
        deny_list.add(
            &hex::encode(identity.public_key_bytes()),
            DenyReason::QuotaExceeded,
        );
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            Arc::new(NodePolicyRuntime::default()),
        );

        let result = service.process(
            &create_v2_hello(&identity),
            &[],
            "127.0.0.1:26000".parse().expect("client endpoint"),
        );

        assert!(matches!(result, Err(ServerError::WalletDenied { .. })));
        assert_eq!(ip_pool.allocated_count(), 0);
        assert_eq!(sessions.count(), 0);
        assert!(routing.is_empty());
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
            PROTOCOL_VERSION_V1,
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
        assert!(service.process(&client_hello, &[], client_addr).is_err());

        // Remove from deny list (simulating tier upgrade).
        deny_list.remove(&wallet_hex);

        // Now allowed.
        let result = service.process(&client_hello, &[], client_addr);
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
            PROTOCOL_VERSION_V1,
        );
        let client_addr: SocketAddr = "127.0.0.1:12345".parse().unwrap();

        let result = service.process(&client_hello, &[], client_addr);
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
                PROTOCOL_VERSION_V1,
            );
            let client_addr: SocketAddr = format!("127.0.0.1:{}", 12345 + index).parse().unwrap();
            let result = service.process(&client_hello, &[], client_addr);
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

    #[test]
    fn v2_respects_dynamic_policy_capacity_without_partial_allocation() {
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let policy = Arc::new(NodePolicyRuntime::default());
        policy.update(NodePolicySnapshot {
            max_sessions: 1,
            ..NodePolicySnapshot::default()
        });
        let service = HandshakeService::new(
            IdentityKeyPair::generate(),
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            policy,
        );
        let first = IdentityKeyPair::generate();
        service
            .process(
                &create_v2_hello(&first),
                &[],
                "127.0.0.1:27000".parse().expect("first endpoint"),
            )
            .expect("first v2 admission");
        let allocated = ip_pool.allocated_count();
        let second = IdentityKeyPair::generate();

        let rejected = service.process(
            &create_v2_hello(&second),
            &[],
            "127.0.0.1:27001".parse().expect("second endpoint"),
        );

        assert!(matches!(
            rejected,
            Err(ServerError::NodePolicyRejected { .. })
        ));
        assert_eq!(ip_pool.allocated_count(), allocated);
        assert_eq!(sessions.count(), 1);
        assert_eq!(routing.count(), 1);
    }

    #[test]
    fn test_concurrent_handshakes_respect_policy_max_sessions() {
        const ATTEMPTS: usize = 24;
        const POLICY_LIMIT: u32 = 4;

        let server_identity = IdentityKeyPair::generate();
        let (ip_pool, sessions, routing, deny_list) = create_test_services();
        let policy = Arc::new(NodePolicyRuntime::default());
        policy.update(NodePolicySnapshot {
            max_sessions: POLICY_LIMIT,
            ..NodePolicySnapshot::default()
        });
        let service = Arc::new(HandshakeService::new(
            server_identity,
            Arc::clone(&ip_pool),
            Arc::clone(&sessions),
            Arc::clone(&routing),
            deny_list,
            policy,
        ));
        let start = Arc::new(Barrier::new(ATTEMPTS));

        let handles: Vec<_> = (0..ATTEMPTS)
            .map(|index| {
                let service = Arc::clone(&service);
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    let client_identity = IdentityKeyPair::generate();
                    let client_ephemeral = EphemeralKeyPair::generate();
                    let client_hello = create_client_hello(
                        &client_identity,
                        client_ephemeral.public_key_bytes(),
                        PROTOCOL_VERSION_V1,
                    );
                    let port = 20_000u16 + u16::try_from(index).expect("test index fits u16");
                    let client_addr = SocketAddr::from(([127, 0, 0, 1], port));
                    start.wait();
                    service.process(&client_hello, &[], client_addr).is_ok()
                })
            })
            .collect();

        let successes = handles
            .into_iter()
            .map(|handle| handle.join().expect("handshake worker must not panic"))
            .filter(|succeeded| *succeeded)
            .count();

        assert_eq!(successes, POLICY_LIMIT as usize);
        assert_eq!(sessions.count(), POLICY_LIMIT as usize);
        assert_eq!(ip_pool.allocated_count(), POLICY_LIMIT as usize);
        assert_eq!(routing.count(), POLICY_LIMIT as usize);
    }
}
