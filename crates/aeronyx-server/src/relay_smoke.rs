// ============================================
// File: crates/aeronyx-server/src/relay_smoke.rs
// ============================================
//! # Authenticated live relay smoke client
//!
//! ## Creation Reason
//! Provides a bounded, host-local operator command that proves a real client
//! can complete the production UDP handshake, submit one E2E ciphertext through
//! the authenticated `ChatRelay` path, observe a verified multi-hop terminal
//! receipt, and complete the entry-node mailbox pull/ACK lifecycle.
//!
//! ## Main Functionality
//! - Pins the `ServerHello` identity to the node key configured on this host.
//! - Reuses the production `ClientHello`, transport AEAD, and `MemChain` codecs.
//! - Sends random E2E ciphertext between two ephemeral identities.
//! - Requires an idle, healthy, two-hop-ready node before execution.
//! - Requires verified FULL encrypted-custody durability before protocol traffic.
//! - Verifies a request-bound terminal receipt, exact mailbox bytes, E2E
//!   decryption, ACK deletion, and aggregate delivery-accounting consistency.
//!
//! ## Dependencies
//! - `aeronyx-core`: production handshake, key, transport, chat, and codec APIs.
//! - `main.rs`: CLI parsing, local config lookup, and aggregate-only output.
//! - `/api/vpn/health`: host-local readiness and receipt evidence contract.
//!
//! ## Main Logical Flow
//! 1. Verify the host-local health surface is idle and two-hop ready.
//! 2. Open a pinned sender VPN session and submit one signed E2E envelope.
//! 3. Verify the private response and signed terminal receipt bind the exact
//!    ephemeral request, route, purpose, and opaque envelope bytes.
//! 4. Open the ephemeral receiver session, pull the exact envelope, decrypt it,
//!    ACK it on the entry node, and prove the entry mailbox is empty.
//!
//! ## Important Notes for Next Developer
//! - The command must remain host-local and must never accept real user keys.
//! - Never print identities, message IDs, endpoints, nonces, payloads, or keys.
//! - Terminal replicas are TTL-managed today; entry-node ACK does not delete a
//!   terminal replica and the report must preserve that limitation.
//! - This is real protocol traffic, not a synthetic counter mutation.
//! - [CHAT-RELAY-DURABILITY-PREFLIGHT 2026-08-16 by Codex] Missing or
//!   unverified custody durability must fail before ephemeral sessions exist.
//! - [RELAY-SMOKE-LOOPBACK-SOURCE 2026-08-25 by Codex] Bind the UDP client to
//!   the matching loopback family; this command is host-local by contract.
//!
//! Last Modified: v1.3.0-RequestBoundTerminalProof - Requires the encrypted
//! verified-submit response and exact terminal receipt before aggregate health
//! may be used as a secondary consistency observation.
//!
//! Previous: v1.2.0-LoopbackSourceBinding - Uses an explicit loopback source
//! address for portable host-local UDP handshake behavior.
// ============================================

use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use base64::Engine;
use futures::StreamExt;
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::net::UdpSocket;
use tokio::time::{sleep, timeout_at, Instant as TokioInstant};

use aeronyx_core::crypto::handshake::{create_client_hello, verify_server_hello};
use aeronyx_core::crypto::kdf::derive_session_key;
use aeronyx_core::crypto::transport::{decrypt_packet, encrypt_packet};
use aeronyx_core::crypto::{E2eSession, EphemeralKeyPair, IdentityKeyPair, SessionKey};
use aeronyx_core::protocol::codec::{
    decode_data_packet, decode_server_hello, encode_client_hello, encode_data_packet,
};
use aeronyx_core::protocol::memchain::{
    ChatRelayVerifiedSubmitRequestV1, ChatRelayVerifiedSubmitResponseV1,
    CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1,
    CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1, CHAT_VERIFIED_SUBMIT_REJECTED_V1,
};
use aeronyx_core::protocol::{
    decode_memchain, encode_memchain, ChatContentType, ChatEnvelope, DataPacket, MemChainMessage,
    CURRENT_PROTOCOL_VERSION, DOMAIN_CHAT_ACK, DOMAIN_CHAT_PULL_V2, DOMAIN_SESSION_CLOSE_V1,
    MEMCHAIN_MAGIC,
};

const MIN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_TIMEOUT: Duration = Duration::from_secs(120);
const HEALTH_BODY_LIMIT: usize = 1024 * 1024;
const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);
const ACK_SETTLE_INTERVAL: Duration = Duration::from_millis(100);
const SMOKE_PLAINTEXT_BYTES: usize = 48;
// [SESSION-TERMINATION 2026-08-15 by Codex] Cleanup has its own short budget
// after the proof deadline and duplicates the tiny UDP close frame once.
const SESSION_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
const SESSION_CLOSE_REDUNDANCY: usize = 2;

/// Host-local options for one authenticated live relay smoke run.
#[derive(Debug, Clone)]
pub struct RelaySmokeOptions {
    pub server_addr: SocketAddr,
    pub health_url: String,
    pub expected_server_key: [u8; 32],
    pub timeout: Duration,
}

impl RelaySmokeOptions {
    fn validate(&self) -> Result<reqwest::Url> {
        // [LIVE-RELAY-SMOKE 2026-08-15 by Codex] Keep the first production
        // smoke host-local. This makes aggregate receipt attribution honest
        // and prevents this privileged operator command becoming a generic
        // remote scanner or SSRF primitive.
        anyhow::ensure!(
            self.server_addr.ip().is_loopback(),
            "relay smoke server must use a loopback address"
        );
        anyhow::ensure!(
            (MIN_TIMEOUT..=MAX_TIMEOUT).contains(&self.timeout),
            "relay smoke timeout must be between 5 and 120 seconds"
        );

        let url = reqwest::Url::parse(&self.health_url).context("invalid health URL")?;
        anyhow::ensure!(
            url.scheme() == "http",
            "health URL must use host-local HTTP"
        );
        anyhow::ensure!(
            url.username().is_empty() && url.password().is_none(),
            "health URL must not contain credentials"
        );
        anyhow::ensure!(
            url.query().is_none() && url.fragment().is_none(),
            "health URL must not contain a query or fragment"
        );
        let host = url
            .host_str()
            .context("health URL must contain a host")?
            .parse::<IpAddr>()
            .context("health URL host must be an IP address")?;
        anyhow::ensure!(host.is_loopback(), "health URL must use a loopback address");
        Ok(url)
    }
}

/// Aggregate-only report emitted by a successful smoke run.
#[derive(Debug, Serialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct RelaySmokeReport {
    pub status: &'static str,
    pub transport: &'static str,
    pub terminal_receipt_observed: bool,
    pub verified_client_deliveries_before: u64,
    pub verified_client_deliveries_after: u64,
    pub entry_mailbox_round_trip_verified: bool,
    pub entry_mailbox_ack_verified: bool,
    pub e2e_ciphertext_verified: bool,
    pub ephemeral_sessions_created: u8,
    pub session_cleanup: &'static str,
    pub terminal_replica_cleanup: &'static str,
    pub evidence_scope: &'static str,
    pub elapsed_ms: u64,
    pub privacy_boundary: &'static str,
}

#[derive(Debug, Deserialize)]
struct HealthSnapshot {
    status: String,
    active_sessions: u64,
    privacy_protocol_health: PrivacyProtocolHealth,
    discovery_status: DiscoveryStatus,
    #[serde(default)]
    chat_relay_status: Option<ChatRelayHealth>,
}

#[derive(Debug, Deserialize)]
struct ChatRelayHealth {
    #[serde(default)]
    configured_enabled: bool,
    #[serde(default)]
    runtime_ready: bool,
    peer_relay: ChatRelayOutboundStatus,
}

#[derive(Debug, Deserialize)]
struct ChatRelayOutboundStatus {
    /// Verified aggregate custody durability from the Rust relay service.
    #[serde(default)]
    custody_durability: Option<ChatRelayCustodyDurabilityStatus>,
    /// [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] Optional preserves
    /// readable failure output against an older local binary, but never falls
    /// back to the ambiguous aggregate/direct relay status.
    #[serde(default)]
    authenticated_onion_outbound: Option<ChatRelayRouteStatus>,
}

#[derive(Debug, Deserialize)]
struct ChatRelayCustodyDurabilityStatus {
    state: String,
    full_durability_verified: bool,
    synchronous_level: Option<u8>,
}

#[derive(Debug, Deserialize)]
struct ChatRelayRouteStatus {
    rounds: u64,
    last_attempted: u64,
    last_accepted: u64,
    last_failed: u64,
    last_status: Option<String>,
    last_failure_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PrivacyProtocolHealth {
    failed_checks: u64,
}

#[derive(Debug, Deserialize)]
struct DiscoveryStatus {
    peer_store: HealthPeerStore,
}

#[derive(Debug, Deserialize)]
struct HealthPeerStore {
    blind_relay_quality: BlindRelayQuality,
    peer_quorum: PeerQuorum,
    route_governance: RouteGovernance,
    network_story: NetworkStory,
}

#[derive(Debug, Deserialize)]
struct BlindRelayQuality {
    verified_client_onion_deliveries: u64,
    delivery_receipt_capable_peers: u64,
    authenticated_delivery_path_ready: bool,
    authenticated_delivery_path_reason: String,
}

#[derive(Debug, Deserialize)]
struct PeerQuorum {
    quorum_ready: bool,
}

#[derive(Debug, Deserialize)]
struct RouteGovernance {
    route_pool_ready: bool,
}

#[derive(Debug, Deserialize)]
struct NetworkStory {
    chat_two_hop_onion_ready: bool,
}

impl HealthSnapshot {
    const fn verified_client_deliveries(&self) -> u64 {
        self.discovery_status
            .peer_store
            .blind_relay_quality
            .verified_client_onion_deliveries
    }

    fn authenticated_onion_outbound_rounds(&self) -> Option<u64> {
        self.chat_relay_status
            .as_ref()
            .and_then(|health| health.peer_relay.authenticated_onion_outbound.as_ref())
            .map(|status| status.rounds)
    }

    fn authenticated_onion_outbound_diagnostic(&self, baseline_rounds: Option<u64>) -> String {
        // [RELAY-SMOKE-DIAGNOSTIC-ALLOWLIST 2026-08-31 by Codex] Treat the
        // host-local health document as untrusted at the final output boundary.
        // The relay service normally publishes typed aggregate buckets, but a
        // mixed-version or corrupted runtime must not make this privileged CLI
        // echo an endpoint, route id, message id, or payload-adjacent string.
        let Some(status) = self
            .chat_relay_status
            .as_ref()
            .and_then(|health| health.peer_relay.authenticated_onion_outbound.as_ref())
        else {
            return "authenticated_onion_status=unavailable".to_string();
        };
        if baseline_rounds.is_some_and(|baseline| status.rounds <= baseline) {
            return "authenticated_onion_status=no_new_round".to_string();
        }
        format!(
            "authenticated_onion_status={}, failure_reason={}, attempted={}, accepted={}, failed={}",
            safe_authenticated_onion_status(status.last_status.as_deref()),
            safe_authenticated_onion_failure(status.last_failure_reason.as_deref()),
            status.last_attempted,
            status.last_accepted,
            status.last_failed,
        )
    }

    fn ensure_idle_two_hop_ready(&self) -> Result<()> {
        anyhow::ensure!(self.status == "ok", "node health is not ready");
        anyhow::ensure!(
            self.privacy_protocol_health.failed_checks == 0,
            "node health has failed checks"
        );
        anyhow::ensure!(
            self.active_sessions == 0,
            "relay smoke requires zero active sessions for evidence attribution"
        );
        anyhow::ensure!(
            self.discovery_status.peer_store.peer_quorum.quorum_ready,
            "peer quorum is not ready"
        );
        anyhow::ensure!(
            self.discovery_status
                .peer_store
                .route_governance
                .route_pool_ready,
            "route pool is not ready"
        );
        anyhow::ensure!(
            self.discovery_status
                .peer_store
                .network_story
                .chat_two_hop_onion_ready,
            "two-hop onion route is not ready"
        );
        let path = &self.discovery_status.peer_store.blind_relay_quality;
        // [AUTHENTICATED-RELAY-PATH-DIAGNOSTICS 2026-08-15 by Codex] Report
        // the production selector's stable failure bucket before the
        // defensive aggregate-count invariant. The path gate already proves
        // that distinct middle and terminal peers exist, while its reason is
        // actionable for operators and never exposes peer identities.
        anyhow::ensure!(
            path.authenticated_delivery_path_ready,
            "authenticated delivery path is not ready: {}",
            AuthenticatedDeliveryPathFailure::from_reason(&path.authenticated_delivery_path_reason)
                .as_str()
        );
        anyhow::ensure!(
            path.delivery_receipt_capable_peers >= 2,
            "authenticated delivery path reported ready with fewer than two delivery-receipt-capable peers"
        );
        // [RELAY-SMOKE-RUNTIME-GATE 2026-08-15 by Codex] Discovery readiness
        // does not prove the local encrypted mailbox/relay runtime exists.
        // Fail before creating ephemeral VPN sessions when the service failed
        // initialization or an older running binary lacks route-class proof.
        let relay = self
            .chat_relay_status
            .as_ref()
            .context("chat relay health is unavailable")?;
        anyhow::ensure!(
            relay.configured_enabled,
            "chat relay is disabled in node configuration"
        );
        anyhow::ensure!(relay.runtime_ready, "chat relay runtime is not ready");
        let durability = relay.peer_relay.custody_durability.as_ref().context(
            "encrypted custody durability is unavailable; restart the node with the current binary",
        )?;
        anyhow::ensure!(
            durability.state == "full"
                && durability.full_durability_verified
                && durability.synchronous_level.is_some_and(|level| level >= 2),
            "encrypted custody durability is not FULL-or-stronger"
        );
        anyhow::ensure!(
            relay.peer_relay.authenticated_onion_outbound.is_some(),
            "authenticated onion relay health is unavailable; restart the node with the current binary"
        );
        Ok(())
    }
}

// [RELAY-SMOKE-TYPED-PATH-FAILURE 2026-08-31 by Codex] Keep the preflight
// failure surface closed even when a mixed-version or corrupted host-local
// health response supplies an arbitrary string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthenticatedDeliveryPathFailure {
    NoReceiptCapableTerminal,
    NoNetworkDiverseReceiptPath,
    NoReceiptCapableMiddle,
    Unknown,
}

impl AuthenticatedDeliveryPathFailure {
    fn from_reason(reason: &str) -> Self {
        match reason {
            "no_receipt_capable_terminal" => Self::NoReceiptCapableTerminal,
            "no_network_diverse_receipt_path" => Self::NoNetworkDiverseReceiptPath,
            "no_receipt_capable_middle" => Self::NoReceiptCapableMiddle,
            _ => Self::Unknown,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::NoReceiptCapableTerminal => "no_receipt_capable_terminal",
            Self::NoNetworkDiverseReceiptPath => "no_network_diverse_receipt_path",
            Self::NoReceiptCapableMiddle => "no_receipt_capable_middle",
            Self::Unknown => "unknown",
        }
    }
}

fn safe_authenticated_onion_status(status: Option<&str>) -> &'static str {
    match status {
        None => "unobserved",
        Some("healthy") => "healthy",
        Some("degraded") => "degraded",
        Some("failed") => "failed",
        Some("idle") => "idle",
        Some(_) => "unknown",
    }
}

// [RELAY-SMOKE-TYPED-OUTBOUND-FAILURE 2026-08-31 by Codex] Store only
// explicit static literals in the accepted variant. This makes the privacy
// boundary compile-time visible and cannot accidentally return the borrowed
// health-document string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthenticatedOnionFailure {
    None,
    Allowlisted(&'static str),
    HttpStatus,
    Unknown,
}

static AUTHENTICATED_ONION_FAILURE_ALLOWLIST: &[&str] = &[
    "peer_http_client_unavailable",
    "no_receipt_capable_terminal",
    "no_network_diverse_receipt_path",
    "no_receipt_capable_middle",
    "onion_terminal_selection_changed",
    "onion_terminal_diversity_exhausted",
    "onion_middle_candidate_unavailable",
    "onion_middle_endpoint_missing",
    "onion_middle_endpoint_invalid",
    "onion_request_build_failed",
    "onion_payload_encoding_failed",
    "onion_route_refresh_required",
    "onion_route_policy_rejected",
    "onion_route_local_construction_failed",
    "onion_delivery_receipt_rejected",
    "onion_delivery_receipt_verifier_unavailable",
    "onion_delivery_route_surface_changed",
    "onion_delivery_ack_response_too_large",
    "onion_delivery_ack_response_body_read_failed",
    "onion_delivery_ack_response_json_decode_failed",
    "onion_delivery_request_timeout",
    "onion_delivery_request_connect",
    "onion_delivery_request_http_status",
    "onion_delivery_request_decode",
    "onion_delivery_request_body",
    "onion_delivery_request_request",
    "onion_delivery_request_unknown",
    "unknown",
];

impl AuthenticatedOnionFailure {
    fn from_reason(reason: Option<&str>) -> Self {
        let Some(reason) = reason else {
            return Self::None;
        };
        if let Some(safe) = AUTHENTICATED_ONION_FAILURE_ALLOWLIST
            .iter()
            .copied()
            .find(|safe| *safe == reason)
        {
            return Self::Allowlisted(safe);
        }
        if is_bounded_http_status_bucket(reason, "onion_delivery_http_")
            || is_bounded_http_status_bucket(reason, "onion_delivery_request_http_")
        {
            Self::HttpStatus
        } else {
            Self::Unknown
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Allowlisted(reason) => reason,
            Self::HttpStatus => "onion_delivery_http_status",
            Self::Unknown => "unknown",
        }
    }
}

fn safe_authenticated_onion_failure(reason: Option<&str>) -> &'static str {
    AuthenticatedOnionFailure::from_reason(reason).as_str()
}

fn is_bounded_http_status_bucket(reason: &str, prefix: &str) -> bool {
    let Some(status) = reason.strip_prefix(prefix) else {
        return false;
    };
    status.len() == 3
        && status.bytes().all(|byte| byte.is_ascii_digit())
        && status
            .parse::<u16>()
            .is_ok_and(|status| (100..=599).contains(&status))
}

struct HealthClient {
    client: reqwest::Client,
    url: reqwest::Url,
}

impl HealthClient {
    fn new(url: reqwest::Url, request_timeout: Duration) -> Result<Self> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(2))
            .redirect(reqwest::redirect::Policy::none())
            .timeout(request_timeout)
            .build()
            .map_err(|_| anyhow::anyhow!("failed to initialize health client"))?;
        Ok(Self { client, url })
    }

    async fn fetch_unbounded(&self) -> Result<HealthSnapshot> {
        let response = self
            .client
            .get(self.url.clone())
            .send()
            .await
            .map_err(|_| anyhow::anyhow!("health request failed"))?;
        anyhow::ensure!(
            response.status().is_success(),
            "health request was rejected"
        );
        if let Some(length) = response.content_length() {
            anyhow::ensure!(
                length <= HEALTH_BODY_LIMIT as u64,
                "health response exceeds size limit"
            );
        }

        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| anyhow::anyhow!("health response read failed"))?;
            let next_len = body
                .len()
                .checked_add(chunk.len())
                .context("health response length overflow")?;
            anyhow::ensure!(
                next_len <= HEALTH_BODY_LIMIT,
                "health response exceeds size limit"
            );
            body.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&body).context("health response contract is invalid")
    }

    async fn fetch_until(&self, deadline: TokioInstant) -> Result<HealthSnapshot> {
        // [SESSION-TERMINATION 2026-08-15 by Codex] The HTTP client's own
        // timeout is only defense in depth. Every proof and cleanup read must
        // obey its phase deadline so cancellation is never needed for control.
        timeout_at(deadline, self.fetch_unbounded())
            .await
            .map_err(|_| anyhow::anyhow!("health request exceeded phase deadline"))?
    }

    async fn wait_for_aggregate_delivery_consistency(
        &self,
        baseline: u64,
        baseline_outbound_rounds: Option<u64>,
        deadline: TokioInstant,
    ) -> Result<HealthSnapshot> {
        // [RELAY-SMOKE-REQUEST-BOUND-PROOF 2026-08-31 by Codex] This aggregate
        // counter is only a health-accounting consistency gate. The caller
        // must already hold the exact encrypted verified-submit response and
        // request-bound terminal receipt before entering this loop.
        loop {
            let snapshot = self.fetch_until(deadline).await?;
            anyhow::ensure!(
                snapshot.active_sessions == 1,
                "concurrent session activity prevents receipt attribution"
            );
            let current = snapshot.verified_client_deliveries();
            anyhow::ensure!(
                current >= baseline,
                "verified client delivery counter regressed"
            );
            if current > baseline {
                return Ok(snapshot);
            }
            let diagnostic =
                snapshot.authenticated_onion_outbound_diagnostic(baseline_outbound_rounds);
            timeout_at(deadline, sleep(HEALTH_POLL_INTERVAL))
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "verified terminal receipt was not observed before timeout ({diagnostic})"
                    )
                })?;
        }
    }

    async fn wait_for_active_sessions(&self, expected: u64, deadline: TokioInstant) -> Result<()> {
        loop {
            let snapshot = self.fetch_until(deadline).await?;
            if snapshot.active_sessions == expected {
                return Ok(());
            }
            timeout_at(deadline, sleep(HEALTH_POLL_INTERVAL))
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "explicit smoke session cleanup was not observed before timeout"
                    )
                })?;
        }
    }
}

struct RelaySmokeClient {
    socket: UdpSocket,
    identity: IdentityKeyPair,
    session_id: [u8; 16],
    session_key: SessionKey,
    next_tx_counter: u64,
    highest_server_counter: Option<u64>,
}

impl RelaySmokeClient {
    async fn connect(
        server_addr: SocketAddr,
        expected_server_key: &[u8; 32],
        identity: IdentityKeyPair,
        deadline: TokioInstant,
    ) -> Result<Self> {
        // [RELAY-SMOKE-LOOPBACK-SOURCE 2026-08-25 by Codex] Validation makes
        // the smoke path host-local. Bind the same address family explicitly;
        // wildcard UDP sources can fail loopback route selection on macOS.
        let bind_addr = match server_addr {
            SocketAddr::V4(_) => "127.0.0.1:0",
            SocketAddr::V6(_) => "[::1]:0",
        };
        let socket = UdpSocket::bind(bind_addr)
            .await
            .map_err(|_| anyhow::anyhow!("failed to bind smoke UDP socket"))?;
        socket
            .connect(server_addr)
            .await
            .map_err(|_| anyhow::anyhow!("failed to connect smoke UDP socket"))?;

        let ephemeral = EphemeralKeyPair::generate();
        let hello = create_client_hello(
            &identity,
            ephemeral.public_key_bytes(),
            CURRENT_PROTOCOL_VERSION,
        );
        let hello_bytes = encode_client_hello(&hello);
        let sent = timeout_at(deadline, socket.send(&hello_bytes))
            .await
            .map_err(|_| anyhow::anyhow!("ClientHello send timed out"))?
            .map_err(|_| anyhow::anyhow!("ClientHello send failed"))?;
        anyhow::ensure!(sent == hello_bytes.len(), "ClientHello send was incomplete");

        let mut response = [0u8; 151];
        let received = timeout_at(deadline, socket.recv(&mut response))
            .await
            .map_err(|_| anyhow::anyhow!("ServerHello timed out"))?
            .map_err(|_| anyhow::anyhow!("ServerHello receive failed"))?;
        anyhow::ensure!(received == 150, "ServerHello has an invalid wire length");
        let server_hello =
            decode_server_hello(&response[..received]).context("ServerHello decode failed")?;
        anyhow::ensure!(
            server_hello.server_public_key == *expected_server_key,
            "ServerHello identity pin mismatch"
        );
        anyhow::ensure!(
            server_hello.version == CURRENT_PROTOCOL_VERSION,
            "ServerHello protocol version mismatch"
        );
        verify_server_hello(&server_hello, &identity.public_key_bytes())
            .context("ServerHello signature verification failed")?;

        let shared_secret = ephemeral.exchange(&server_hello.server_ephemeral_key);
        let session_key = derive_session_key(
            &shared_secret,
            &identity.public_key_bytes(),
            &server_hello.server_public_key,
        )
        .context("session key derivation failed")?;

        Ok(Self {
            socket,
            identity,
            session_id: server_hello.session_id,
            session_key,
            next_tx_counter: 0,
            highest_server_counter: None,
        })
    }

    async fn send_memchain(
        &mut self,
        message: &MemChainMessage,
        deadline: TokioInstant,
    ) -> Result<()> {
        let plaintext = encode_memchain(message).context("MemChain encode failed")?;
        let counter = self.next_tx_counter;
        self.next_tx_counter = self
            .next_tx_counter
            .checked_add(1)
            .context("client transport counter exhausted")?;
        let encrypted = encrypt_packet(&self.session_key, counter, &self.session_id, &plaintext)
            .context("client transport encryption failed")?;
        let packet = DataPacket::new(self.session_id, counter, encrypted);
        let bytes = encode_data_packet(&packet);
        let sent = timeout_at(deadline, self.socket.send(&bytes))
            .await
            .map_err(|_| anyhow::anyhow!("encrypted frame send timed out"))?
            .map_err(|_| anyhow::anyhow!("encrypted frame send failed"))?;
        anyhow::ensure!(sent == bytes.len(), "encrypted frame send was incomplete");
        Ok(())
    }

    async fn send_graceful_close(&mut self, deadline: TokioInstant) -> Result<()> {
        let request = build_session_close(&self.identity, self.session_id)?;
        // [SESSION-TERMINATION 2026-08-15 by Codex] Duplicate the small UDP
        // control frame with fresh transport counters. Server removal is
        // idempotent; this only reduces the chance that a lost datagram leaves
        // the smoke session waiting for the normal liveness timeout.
        for _ in 0..SESSION_CLOSE_REDUNDANCY {
            self.send_memchain(&request, deadline).await?;
        }
        Ok(())
    }

    async fn receive_memchain(&mut self, deadline: TokioInstant) -> Result<MemChainMessage> {
        // [LIVE-RELAY-SMOKE 2026-08-15 by Codex] Keep the maximum UDP receive
        // buffer on the heap so this async future remains small when nested in
        // the pull/ACK orchestration futures.
        let mut buffer = vec![0u8; 65_535].into_boxed_slice();
        loop {
            let received = timeout_at(deadline, self.socket.recv(&mut buffer))
                .await
                .map_err(|_| anyhow::anyhow!("encrypted response timed out"))?
                .map_err(|_| anyhow::anyhow!("encrypted response receive failed"))?;
            anyhow::ensure!(
                !(received == 1 && buffer[0] == 0xff),
                "server reset the smoke session"
            );
            let Ok(packet) = decode_data_packet(&buffer[..received]) else {
                continue;
            };
            if packet.session_id != self.session_id {
                continue;
            }
            if self
                .highest_server_counter
                .is_some_and(|highest| packet.counter <= highest)
            {
                continue;
            }
            let plaintext = decrypt_packet(
                &self.session_key,
                packet.counter,
                &self.session_id,
                &packet.encrypted_payload,
            )
            .context("server transport decryption failed")?;
            self.highest_server_counter = Some(packet.counter);
            if plaintext.first().copied() != Some(MEMCHAIN_MAGIC) {
                continue;
            }
            return decode_memchain(&plaintext[1..]).context("server MemChain response is invalid");
        }
    }
}

fn unix_now() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs())
}

fn sign_domain(identity: &IdentityKeyPair, domain: &str, payload_slices: &[&[u8]]) -> [u8; 64] {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    for slice in payload_slices {
        hasher.update(slice);
    }
    let digest: [u8; 32] = hasher.finalize().into();
    identity.sign(&digest)
}

fn build_pull(identity: &IdentityKeyPair) -> Result<MemChainMessage> {
    let wallet = identity.public_key_bytes();
    let after_timestamp = 0u64;
    let cursor = Vec::new();
    let cursor_len = 0u16;
    let limit = 10u32;
    let request_timestamp = unix_now()?;
    let after_bytes = after_timestamp.to_le_bytes();
    let cursor_len_bytes = cursor_len.to_le_bytes();
    let limit_bytes = limit.to_le_bytes();
    let timestamp_bytes = request_timestamp.to_le_bytes();
    let signature = sign_domain(
        identity,
        DOMAIN_CHAT_PULL_V2,
        &[
            wallet.as_ref(),
            after_bytes.as_ref(),
            cursor_len_bytes.as_ref(),
            cursor.as_slice(),
            limit_bytes.as_ref(),
            timestamp_bytes.as_ref(),
        ],
    );
    Ok(MemChainMessage::ChatPullV2 {
        wallet,
        after_timestamp,
        cursor,
        limit,
        request_timestamp,
        signature,
    })
}

fn build_ack(identity: &IdentityKeyPair, message_id: [u8; 16]) -> Result<MemChainMessage> {
    let wallet = identity.public_key_bytes();
    let ack_timestamp = unix_now()?;
    let mut ids_hasher = Sha256::new();
    ids_hasher.update(message_id);
    let ids_hash: [u8; 32] = ids_hasher.finalize().into();
    let timestamp_bytes = ack_timestamp.to_le_bytes();
    let signature = sign_domain(
        identity,
        DOMAIN_CHAT_ACK,
        &[wallet.as_ref(), timestamp_bytes.as_ref(), ids_hash.as_ref()],
    );
    Ok(MemChainMessage::ChatAck {
        message_ids: vec![message_id],
        wallet,
        ack_timestamp,
        signature,
    })
}

fn build_session_close(
    identity: &IdentityKeyPair,
    session_id: [u8; 16],
) -> Result<MemChainMessage> {
    let close_timestamp = unix_now()?;
    let timestamp_bytes = close_timestamp.to_le_bytes();
    // [SESSION-TERMINATION 2026-08-15 by Codex] The close request is signed by
    // the same identity used in ClientHello and bound to the exact negotiated
    // session. The server additionally verifies the encrypted outer session.
    let signature = sign_domain(
        identity,
        DOMAIN_SESSION_CLOSE_V1,
        &[session_id.as_ref(), timestamp_bytes.as_ref()],
    );
    Ok(MemChainMessage::SessionCloseV1 {
        session_id,
        close_timestamp,
        signature,
    })
}

fn build_smoke_envelope(
    sender: &IdentityKeyPair,
    receiver: &IdentityKeyPair,
) -> Result<(ChatEnvelope, Vec<u8>, E2eSession)> {
    let receiver_x25519 = receiver.x25519_public_key_bytes();
    let (sender_e2e, sender_x25519) = sender.e2e_handshake(&receiver_x25519);
    let (receiver_e2e, _) = receiver.e2e_handshake(&sender_x25519);

    let mut plaintext = vec![0u8; SMOKE_PLAINTEXT_BYTES];
    OsRng.fill_bytes(&mut plaintext);
    let mut nonce = [0u8; 24];
    OsRng.fill_bytes(&mut nonce);
    let ciphertext = sender_e2e
        .encrypt_raw(&plaintext, &nonce)
        .context("smoke E2E encryption failed")?;
    let mut message_id = [0u8; 16];
    OsRng.fill_bytes(&mut message_id);
    let mut envelope = ChatEnvelope {
        message_id,
        sender: sender.public_key_bytes(),
        receiver: receiver.public_key_bytes(),
        timestamp: unix_now()?,
        ciphertext,
        nonce,
        content_type: ChatContentType::System,
        signature: [0u8; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());
    Ok((envelope, plaintext, receiver_e2e))
}

fn build_verified_submit_request(
    sender: &IdentityKeyPair,
    envelope: ChatEnvelope,
) -> Result<ChatRelayVerifiedSubmitRequestV1> {
    let mut request_id = [0u8; 16];
    OsRng.fill_bytes(&mut request_id);
    ChatRelayVerifiedSubmitRequestV1::signed(request_id, envelope, unix_now()?, sender)
        .context("verified submit request construction failed")
}

/// Coarse failure vocabulary for one private request-bound terminal proof.
///
/// [RELAY-SMOKE-REQUEST-BOUND-PROOF 2026-08-31 by Codex] These outcomes never
/// retain or render request ids, message ids, routes, node identities, receipt
/// bytes, payload commitments, ciphertext, nonces, or endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestBoundTerminalProofFailure {
    ResponseContractInvalid,
    ResponseMismatch,
    ResponseRejected,
    EntryOnly,
    OnionOnly,
    TerminalReceiptInvalid,
    ResponseUnavailableAmbiguous,
    TimeoutAmbiguous,
}

impl RequestBoundTerminalProofFailure {
    const fn reason(self) -> &'static str {
        match self {
            Self::ResponseContractInvalid => "verified submit response contract is invalid",
            Self::ResponseMismatch => "verified submit response does not match this request",
            Self::ResponseRejected => "verified submit response rejected the request",
            Self::EntryOnly => "verified submit accepted entry custody without terminal delivery",
            Self::OnionOnly => "verified submit proved terminal delivery without entry custody",
            Self::TerminalReceiptInvalid => "verified submit terminal receipt is invalid",
            Self::ResponseUnavailableAmbiguous => {
                "verified submit response transport failed; delivery is ambiguous and must not be retried automatically"
            }
            Self::TimeoutAmbiguous => {
                "verified submit timed out; delivery is ambiguous and must not be retried automatically"
            }
        }
    }
}

fn validate_request_bound_terminal_proof(
    request: &ChatRelayVerifiedSubmitRequestV1,
    response: &ChatRelayVerifiedSubmitResponseV1,
) -> std::result::Result<(), RequestBoundTerminalProofFailure> {
    // [RELAY-SMOKE-TYPED-OUTCOMES 2026-08-31 by Codex] Keep every incomplete
    // protocol result distinct and fail closed. In particular, neither global
    // health movement nor one half of the delivery/custody contract may fall
    // through to the exact terminal-receipt verifier.
    response
        .validate_shape()
        .map_err(|_| RequestBoundTerminalProofFailure::ResponseContractInvalid)?;
    if response.request_id != request.request_id
        || response.message_id != request.envelope.message_id
    {
        return Err(RequestBoundTerminalProofFailure::ResponseMismatch);
    }
    match response.result {
        CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1 => {}
        CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1 => {
            return Err(RequestBoundTerminalProofFailure::OnionOnly);
        }
        CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1 => {
            return Err(RequestBoundTerminalProofFailure::EntryOnly);
        }
        CHAT_VERIFIED_SUBMIT_REJECTED_V1 => {
            return Err(RequestBoundTerminalProofFailure::ResponseRejected);
        }
        _ => return Err(RequestBoundTerminalProofFailure::ResponseContractInvalid),
    }
    let receipt = response
        .terminal_receipt
        .as_ref()
        .ok_or(RequestBoundTerminalProofFailure::ResponseContractInvalid)?;
    // The pinned entry already verified this terminal against its current
    // signed peer store. Passing the receipt identity here rechecks signature,
    // route, MessageRelay purpose, and exact envelope bytes for corruption; it
    // is not treated as an independent directory trust root.
    response
        .verify_terminal_receipt_for_request(request, &receipt.terminal_node_id)
        .map_err(|_| RequestBoundTerminalProofFailure::TerminalReceiptInvalid)
}

fn classify_terminal_proof_receive_failure(
    deadline: TokioInstant,
    observed_at: TokioInstant,
) -> RequestBoundTerminalProofFailure {
    if observed_at >= deadline {
        RequestBoundTerminalProofFailure::TimeoutAmbiguous
    } else {
        RequestBoundTerminalProofFailure::ResponseUnavailableAmbiguous
    }
}

fn exact_envelope_match(left: &ChatEnvelope, right: &ChatEnvelope) -> bool {
    left.message_id == right.message_id
        && left.sender == right.sender
        && left.receiver == right.receiver
        && left.timestamp == right.timestamp
        && left.ciphertext == right.ciphertext
        && left.nonce == right.nonce
        && left.content_type == right.content_type
        && left.signature == right.signature
}

async fn receive_request_bound_terminal_proof(
    client: &mut RelaySmokeClient,
    request: &ChatRelayVerifiedSubmitRequestV1,
    deadline: TokioInstant,
) -> std::result::Result<(), RequestBoundTerminalProofFailure> {
    loop {
        let response = match client.receive_memchain(deadline).await {
            Ok(response) => response,
            Err(_) => {
                // [RELAY-SMOKE-AMBIGUOUS-TIMEOUT 2026-08-31 by Codex] The one
                // submit may have reached entry or terminal before its private
                // response became unavailable. Preserve timeout separately,
                // but mark both outcomes ambiguous; never resend the request or
                // consult aggregate health as fallback.
                return Err(classify_terminal_proof_receive_failure(
                    deadline,
                    TokioInstant::now(),
                ));
            }
        };
        if let MemChainMessage::ChatRelayVerifiedSubmitResponseV1(response) = response {
            return validate_request_bound_terminal_proof(request, &response);
        }
    }
}

async fn pull_page(
    client: &mut RelaySmokeClient,
    deadline: TokioInstant,
) -> Result<Vec<ChatEnvelope>> {
    let request = build_pull(&client.identity)?;
    client.send_memchain(&request, deadline).await?;
    loop {
        if let MemChainMessage::ChatPullResponseV2 { envelopes, .. } =
            client.receive_memchain(deadline).await?
        {
            return Ok(envelopes);
        }
    }
}

async fn confirm_entry_ack(
    receiver: &mut RelaySmokeClient,
    message_id: [u8; 16],
    deadline: TokioInstant,
) -> Result<()> {
    let ack = build_ack(&receiver.identity, message_id)?;
    receiver.send_memchain(&ack, deadline).await?;
    loop {
        anyhow::ensure!(
            TokioInstant::now() < deadline,
            "entry mailbox ACK was not observed before timeout"
        );
        timeout_at(deadline, sleep(ACK_SETTLE_INTERVAL))
            .await
            .map_err(|_| anyhow::anyhow!("entry mailbox ACK was not observed before timeout"))?;
        let envelopes = pull_page(receiver, deadline).await?;
        if envelopes
            .iter()
            .all(|envelope| envelope.message_id != message_id)
        {
            return Ok(());
        }
    }
}

async fn cleanup_ephemeral_sessions(
    sender: &mut Option<RelaySmokeClient>,
    receiver: &mut Option<RelaySmokeClient>,
    health: &HealthClient,
    baseline_active_sessions: u64,
) -> Result<()> {
    let cleanup_deadline = TokioInstant::now() + SESSION_CLEANUP_TIMEOUT;
    // [SESSION-TERMINATION 2026-08-15 by Codex] Always attempt every known
    // session even if one socket write fails. Aggregate health is the final
    // authority because a prior redundant frame may already have succeeded.
    if let Some(client) = receiver.as_mut() {
        let _ = client.send_graceful_close(cleanup_deadline).await;
    }
    if let Some(client) = sender.as_mut() {
        let _ = client.send_graceful_close(cleanup_deadline).await;
    }
    health
        .wait_for_active_sessions(baseline_active_sessions, cleanup_deadline)
        .await
}

/// Executes one real authenticated relay smoke run within one proof deadline.
///
/// A separate bounded cleanup phase always runs after the transaction. This is
/// intentionally not implemented with an outer cancellation timeout: dropping
/// the future would also drop the session keys required for graceful cleanup.
pub async fn run(options: RelaySmokeOptions) -> Result<RelaySmokeReport> {
    let started = Instant::now();
    let health_url = options.validate()?;
    let health = HealthClient::new(health_url, options.timeout)?;
    let deadline = TokioInstant::now() + options.timeout;
    let baseline = health.fetch_until(deadline).await?;
    baseline.ensure_idle_two_hop_ready()?;
    let baseline_active_sessions = baseline.active_sessions;
    let deliveries_before = baseline.verified_client_deliveries();
    let outbound_rounds_before = baseline.authenticated_onion_outbound_rounds();

    let sender_identity = IdentityKeyPair::generate();
    let receiver_identity = IdentityKeyPair::generate();
    let (expected_envelope, expected_plaintext, receiver_e2e) =
        build_smoke_envelope(&sender_identity, &receiver_identity)?;
    let verified_submit_request =
        build_verified_submit_request(&sender_identity, expected_envelope.clone())?;
    let mut sender = None;
    let mut receiver = None;

    let transaction: Result<u64> = async {
        sender = Some(
            RelaySmokeClient::connect(
                options.server_addr,
                &options.expected_server_key,
                sender_identity,
                deadline,
            )
            .await?,
        );
        sender
            .as_mut()
            .context("sender session was not retained")?
            .send_memchain(
                &MemChainMessage::ChatRelayVerifiedSubmitV1(verified_submit_request.clone()),
                deadline,
            )
            .await?;
        receive_request_bound_terminal_proof(
            sender.as_mut().context("sender session was not retained")?,
            &verified_submit_request,
            deadline,
        )
        .await
        .map_err(|failure| anyhow::anyhow!(failure.reason()))?;
        let consistency_health = health
            .wait_for_aggregate_delivery_consistency(
                deliveries_before,
                outbound_rounds_before,
                deadline,
            )
            .await?;

        receiver = Some(
            RelaySmokeClient::connect(
                options.server_addr,
                &options.expected_server_key,
                receiver_identity,
                deadline,
            )
            .await?,
        );
        let receiver_client = receiver
            .as_mut()
            .context("receiver session was not retained")?;
        let envelopes = pull_page(receiver_client, deadline).await?;
        let delivered = envelopes
            .iter()
            .find(|envelope| envelope.message_id == expected_envelope.message_id)
            .context("entry mailbox did not return the smoke envelope")?;
        anyhow::ensure!(
            exact_envelope_match(delivered, &expected_envelope),
            "entry mailbox envelope differs from the signed input"
        );
        delivered
            .verify_signature()
            .context("entry mailbox envelope signature is invalid")?;
        let decrypted = receiver_e2e
            .decrypt_raw(&delivered.ciphertext, &delivered.nonce)
            .context("smoke E2E decryption failed")?;
        anyhow::ensure!(
            decrypted == expected_plaintext,
            "smoke E2E plaintext verification failed"
        );
        confirm_entry_ack(receiver_client, expected_envelope.message_id, deadline).await?;

        let final_health = health.fetch_until(deadline).await?;
        anyhow::ensure!(
            final_health.active_sessions == 2,
            "concurrent session activity prevents final smoke attribution"
        );
        anyhow::ensure!(
            final_health.verified_client_deliveries()
                >= consistency_health.verified_client_deliveries(),
            "verified client delivery counter regressed after mailbox ACK"
        );
        Ok(final_health.verified_client_deliveries())
    }
    .await;

    let cleanup = cleanup_ephemeral_sessions(
        &mut sender,
        &mut receiver,
        &health,
        baseline_active_sessions,
    )
    .await;
    let deliveries_after = match (transaction, cleanup) {
        (Ok(deliveries_after), Ok(())) => deliveries_after,
        (Err(primary), Ok(())) => return Err(primary),
        (Ok(_), Err(cleanup_error)) => {
            return Err(cleanup_error.context("relay proof passed but session cleanup failed"));
        }
        (Err(primary), Err(cleanup_error)) => {
            return Err(primary.context(format!(
                "smoke session cleanup also failed: {cleanup_error}"
            )));
        }
    };

    Ok(RelaySmokeReport {
        status: "passed",
        transport: "authenticated_udp",
        terminal_receipt_observed: true,
        verified_client_deliveries_before: deliveries_before,
        verified_client_deliveries_after: deliveries_after,
        entry_mailbox_round_trip_verified: true,
        entry_mailbox_ack_verified: true,
        e2e_ciphertext_verified: true,
        ephemeral_sessions_created: 2,
        session_cleanup: "explicit_authenticated_close",
        terminal_replica_cleanup: "node_ttl_managed",
        evidence_scope: "request_bound_terminal_receipt_plus_aggregate_consistency_plus_exact_entry_mailbox_round_trip",
        elapsed_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        privacy_boundary: "aggregate smoke outcomes only; no identities, message ids, endpoints, nonces, ciphertext, plaintext, session ids, or keys",
    })
}

#[derive(Deserialize)]
struct PublicKeyFile {
    public_key: String,
}

/// Loads only the configured public identity field used to pin `ServerHello`.
pub async fn load_expected_server_public_key(path: &Path) -> Result<[u8; 32]> {
    let content = tokio::fs::read_to_string(path)
        .await
        .context("failed to read node key file")?;
    let key_file: PublicKeyFile =
        serde_json::from_str(&content).context("node key file is invalid")?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(key_file.public_key)
        .context("node public key encoding is invalid")?;
    let key: [u8; 32] = decoded
        .try_into()
        .map_err(|_| anyhow::anyhow!("node public key must decode to 32 bytes"))?;
    aeronyx_core::crypto::IdentityPublicKey::from_bytes(&key)
        .context("node public key is invalid")?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::handshake::DefaultHandshakeCrypto;
    use aeronyx_core::crypto::HandshakeCrypto;
    use aeronyx_core::protocol::chat::BlindRelayDeliveryReceipt;
    use aeronyx_core::protocol::codec::decode_client_hello;
    use aeronyx_core::protocol::{encode_envelope, OnionRoutePurpose};

    #[test]
    fn smoke_envelope_is_signed_and_e2e_decryptable() {
        let sender = IdentityKeyPair::generate();
        let receiver = IdentityKeyPair::generate();
        let (envelope, plaintext, receiver_e2e) =
            build_smoke_envelope(&sender, &receiver).expect("build smoke envelope");

        envelope.verify_signature().expect("verify envelope");
        let decrypted = receiver_e2e
            .decrypt_raw(&envelope.ciphertext, &envelope.nonce)
            .expect("decrypt envelope");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn request_bound_terminal_proof_accepts_exact_ephemeral_envelope() {
        // [RELAY-SMOKE-REQUEST-BOUND-PROOF 2026-08-31 by Codex] Exercise the
        // exact protocol evidence used by the live smoke without logging any
        // request, route, message, identity, or payload material.
        let sender = IdentityKeyPair::generate();
        let receiver = IdentityKeyPair::generate();
        let terminal = IdentityKeyPair::generate();
        let (envelope, _, _) =
            build_smoke_envelope(&sender, &receiver).expect("build smoke envelope");
        let request = build_verified_submit_request(&sender, envelope.clone())
            .expect("build verified submit request");
        request
            .verify_authentication()
            .expect("verify submit request");
        let payload = encode_envelope(&envelope).expect("encode envelope");
        let receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
            [0x61; 16],
            &payload,
            OnionRoutePurpose::MessageRelay,
            unix_now().expect("current time"),
            &terminal,
        );
        let response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            true,
            true,
            Some(receipt),
        );

        assert_eq!(
            validate_request_bound_terminal_proof(&request, &response),
            Ok(())
        );
    }

    #[test]
    fn request_bound_terminal_proof_rejects_unrelated_or_partial_evidence() {
        // [RELAY-SMOKE-REQUEST-BOUND-PROOF 2026-08-31 by Codex] Neither a
        // correlated entry-only result nor a valid signature over different
        // opaque bytes may satisfy terminal delivery for this smoke request.
        let sender = IdentityKeyPair::generate();
        let receiver = IdentityKeyPair::generate();
        let terminal = IdentityKeyPair::generate();
        let (envelope, _, _) =
            build_smoke_envelope(&sender, &receiver).expect("build smoke envelope");
        let request = build_verified_submit_request(&sender, envelope.clone())
            .expect("build verified submit request");
        let payload = encode_envelope(&envelope).expect("encode envelope");
        let exact_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
            [0x62; 16],
            &payload,
            OnionRoutePurpose::MessageRelay,
            unix_now().expect("current time"),
            &terminal,
        );
        let exact_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            true,
            true,
            Some(exact_receipt.clone()),
        );

        let mut unrelated_response = exact_response.clone();
        unrelated_response.request_id[0] ^= 0xff;
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &unrelated_response),
            Err(RequestBoundTerminalProofFailure::ResponseMismatch)
        );

        let mut malformed_response = exact_response.clone();
        malformed_response.result = u8::MAX;
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &malformed_response),
            Err(RequestBoundTerminalProofFailure::ResponseContractInvalid)
        );

        let mut different_envelope = envelope.clone();
        different_envelope.ciphertext[0] ^= 0xff;
        let different_payload =
            encode_envelope(&different_envelope).expect("encode different envelope");
        let different_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
            [0x63; 16],
            &different_payload,
            OnionRoutePurpose::MessageRelay,
            unix_now().expect("current time"),
            &terminal,
        );
        let different_payload_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            true,
            true,
            Some(different_receipt),
        );
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &different_payload_response),
            Err(RequestBoundTerminalProofFailure::TerminalReceiptInvalid)
        );

        let different_purpose_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
            [0x64; 16],
            &payload,
            OnionRoutePurpose::BlindVaultPut,
            unix_now().expect("current time"),
            &terminal,
        );
        let different_purpose_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            true,
            true,
            Some(different_purpose_receipt),
        );
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &different_purpose_response),
            Err(RequestBoundTerminalProofFailure::TerminalReceiptInvalid)
        );

        let mut changed_route_receipt = exact_receipt.clone();
        changed_route_receipt.route_id[0] ^= 0xff;
        let changed_route_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            true,
            true,
            Some(changed_route_receipt),
        );
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &changed_route_response),
            Err(RequestBoundTerminalProofFailure::TerminalReceiptInvalid)
        );

        let entry_only_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            false,
            true,
            None,
        );
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &entry_only_response),
            Err(RequestBoundTerminalProofFailure::EntryOnly)
        );

        let onion_only_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
            request.request_id,
            envelope.message_id,
            true,
            false,
            Some(exact_receipt),
        );
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &onion_only_response),
            Err(RequestBoundTerminalProofFailure::OnionOnly)
        );

        let rejected_response =
            ChatRelayVerifiedSubmitResponseV1::rejected(request.request_id, envelope.message_id);
        assert_eq!(
            validate_request_bound_terminal_proof(&request, &rejected_response),
            Err(RequestBoundTerminalProofFailure::ResponseRejected)
        );
    }

    #[test]
    fn request_bound_terminal_proof_timeout_is_typed_ambiguous_and_non_retryable() {
        // [RELAY-SMOKE-AMBIGUOUS-TIMEOUT 2026-08-31 by Codex] Test the pure
        // boundary used after the single submit. The static outcome explicitly
        // forbids automatic resend and contains no request or receipt material.
        let observed_at = TokioInstant::now();
        let timeout = classify_terminal_proof_receive_failure(observed_at, observed_at);
        assert_eq!(timeout, RequestBoundTerminalProofFailure::TimeoutAmbiguous);
        assert_eq!(
            timeout.reason(),
            "verified submit timed out; delivery is ambiguous and must not be retried automatically"
        );

        let future_deadline = observed_at + Duration::from_secs(1);
        assert_eq!(
            classify_terminal_proof_receive_failure(future_deadline, observed_at),
            RequestBoundTerminalProofFailure::ResponseUnavailableAmbiguous
        );
    }

    #[test]
    fn graceful_close_signature_binds_exact_session() {
        let identity = IdentityKeyPair::generate();
        let session_id = [0x91; 16];
        let request = build_session_close(&identity, session_id).expect("build close request");
        let MemChainMessage::SessionCloseV1 {
            session_id: signed_session_id,
            close_timestamp,
            signature,
        } = request
        else {
            panic!("expected SessionCloseV1");
        };
        let timestamp_bytes = close_timestamp.to_le_bytes();
        assert!(aeronyx_core::protocol::verify_signed_message(
            DOMAIN_SESSION_CLOSE_V1,
            &[signed_session_id.as_ref(), timestamp_bytes.as_ref()],
            &identity.public_key_bytes(),
            &signature,
            close_timestamp,
        )
        .is_ok());
        // [SESSION-TERMINATION 2026-08-15 by Codex] A valid signature cannot
        // be moved to a different encrypted session during the time window.
        let other_session_id = [0x92; 16];
        assert!(aeronyx_core::protocol::verify_signed_message(
            DOMAIN_SESSION_CLOSE_V1,
            &[other_session_id.as_ref(), timestamp_bytes.as_ref()],
            &identity.public_key_bytes(),
            &signature,
            close_timestamp,
        )
        .is_err());
    }

    #[test]
    fn relay_failure_diagnostic_uses_only_aggregate_buckets() {
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 1,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 4,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": true,
                        "authenticated_delivery_path_reason": "authenticated_receipt_path_ready"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            },
            "chat_relay_status": {
                "peer_relay": {
                    "authenticated_onion_outbound": {
                        "rounds": 7,
                        "last_attempted": 2,
                        "last_accepted": 0,
                        "last_failed": 2,
                        "last_status": "failed",
                        "last_failure_reason": "onion_delivery_receipt_rejected"
                    }
                }
            }
        }))
        .expect("parse health fixture");

        // [RELAY-HEALTH-DIAGNOSTICS 2026-08-15 by Codex] The rendered failure
        // is operationally useful but cannot identify a route or user.
        assert_eq!(
            snapshot.authenticated_onion_outbound_diagnostic(Some(6)),
            "authenticated_onion_status=failed, failure_reason=onion_delivery_receipt_rejected, attempted=2, accepted=0, failed=2"
        );
        assert_eq!(
            snapshot.authenticated_onion_outbound_diagnostic(Some(7)),
            "authenticated_onion_status=no_new_round"
        );
    }

    #[test]
    fn relay_failure_diagnostic_redacts_untrusted_health_strings() {
        // [RELAY-SMOKE-DIAGNOSTIC-ALLOWLIST 2026-08-31 by Codex] A local
        // mixed-version response is still an input boundary. Prove that values
        // shaped like endpoint or route material never reach operator output.
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 1,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 4,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": true,
                        "authenticated_delivery_path_reason": "authenticated_receipt_path_ready"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            },
            "chat_relay_status": {
                "peer_relay": {
                    "authenticated_onion_outbound": {
                        "rounds": 8,
                        "last_attempted": 1,
                        "last_accepted": 0,
                        "last_failed": 1,
                        "last_status": "endpoint=http://sensitive.invalid",
                        "last_failure_reason": "route_id=00112233445566778899aabbccddeeff"
                    }
                }
            }
        }))
        .expect("parse health fixture");

        let diagnostic = snapshot.authenticated_onion_outbound_diagnostic(Some(7));
        assert_eq!(
            diagnostic,
            "authenticated_onion_status=unknown, failure_reason=unknown, attempted=1, accepted=0, failed=1"
        );
        assert!(!diagnostic.contains("sensitive.invalid"));
        assert!(!diagnostic.contains("00112233445566778899aabbccddeeff"));
    }

    #[test]
    fn relay_failure_diagnostic_coarsens_dynamic_http_status() {
        // [RELAY-SMOKE-DIAGNOSTIC-ALLOWLIST 2026-08-31 by Codex] Both
        // producer prefixes accept only bounded status codes and collapse to
        // one route-independent operator bucket.
        assert_eq!(
            safe_authenticated_onion_failure(Some("onion_delivery_http_503")),
            "onion_delivery_http_status"
        );
        assert_eq!(
            safe_authenticated_onion_failure(Some("onion_delivery_request_http_429")),
            "onion_delivery_http_status"
        );
        assert_eq!(
            safe_authenticated_onion_failure(Some("onion_delivery_http_999")),
            "unknown"
        );
    }

    #[test]
    fn idle_two_hop_preflight_rejects_existing_sessions() {
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 1,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": true,
                        "authenticated_delivery_path_reason": "authenticated_receipt_path_ready"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            }
        }))
        .expect("health fixture");

        assert!(snapshot.ensure_idle_two_hop_ready().is_err());
    }

    #[test]
    fn idle_two_hop_preflight_rejects_unpairable_receipt_peers() {
        // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] A raw peer
        // count must not start a destructive live smoke when the production
        // selector cannot build a network-diverse middle/terminal pair.
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 0,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": false,
                        "authenticated_delivery_path_reason": "no_network_diverse_receipt_path"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            }
        }))
        .expect("health fixture");

        let error = snapshot
            .ensure_idle_two_hop_ready()
            .expect_err("unpairable peers must fail closed");
        assert!(error
            .to_string()
            .contains("no_network_diverse_receipt_path"));
    }

    #[test]
    fn idle_two_hop_preflight_preserves_zero_peer_path_reason() {
        // [AUTHENTICATED-RELAY-PATH-DIAGNOSTICS 2026-08-15 by Codex] A cold
        // restart intentionally clears process-local receipt evidence. Keep
        // the privacy-safe selector reason visible so rollout work is not
        // mistaken for a generic peer-count failure.
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 0,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 0,
                        "authenticated_delivery_path_ready": false,
                        "authenticated_delivery_path_reason": "no_receipt_capable_terminal"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            }
        }))
        .expect("health fixture");

        let error = snapshot
            .ensure_idle_two_hop_ready()
            .expect_err("zero receipt peers must fail closed");
        assert!(error.to_string().contains("no_receipt_capable_terminal"));
    }

    #[test]
    fn idle_two_hop_preflight_redacts_untrusted_path_reason() {
        // [RELAY-SMOKE-TYPED-PATH-FAILURE 2026-08-31 by Codex] The host-local
        // preflight must not echo endpoint or route-shaped material from an
        // untrusted health response before any live session is created.
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 0,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": false,
                        "authenticated_delivery_path_reason": "endpoint=http://sensitive.invalid/route/00112233"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            }
        }))
        .expect("health fixture");

        let error = snapshot
            .ensure_idle_two_hop_ready()
            .expect_err("untrusted path reason must fail closed");
        assert_eq!(
            error.to_string(),
            "authenticated delivery path is not ready: unknown"
        );
        assert!(!error.to_string().contains("sensitive.invalid"));
        assert!(!error.to_string().contains("00112233"));
    }

    #[test]
    fn idle_two_hop_preflight_rejects_missing_relay_runtime() {
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 0,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": true,
                        "authenticated_delivery_path_reason": "authenticated_receipt_path_ready"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            },
            "chat_relay_status": {
                "configured_enabled": true,
                "runtime_ready": false,
                "peer_relay": {
                    "authenticated_onion_outbound": {
                        "rounds": 0,
                        "last_attempted": 0,
                        "last_accepted": 0,
                        "last_failed": 0,
                        "last_status": null,
                        "last_failure_reason": null
                    }
                }
            }
        }))
        .expect("health fixture");

        let error = snapshot
            .ensure_idle_two_hop_ready()
            .expect_err("missing relay runtime must fail before creating sessions");
        assert!(error.to_string().contains("runtime is not ready"));
    }

    #[test]
    fn idle_two_hop_preflight_rejects_unverified_custody_durability() {
        // [CHAT-RELAY-DURABILITY-PREFLIGHT 2026-08-16 by Codex] A rolling
        // upgrade may expose otherwise healthy relay fields from an older
        // binary. It must not create smoke-test sessions without FULL evidence.
        let mut fixture = serde_json::json!({
            "status": "ok",
            "active_sessions": 0,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": true,
                        "authenticated_delivery_path_reason": "authenticated_receipt_path_ready"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            },
            "chat_relay_status": {
                "configured_enabled": true,
                "runtime_ready": true,
                "peer_relay": {
                    "authenticated_onion_outbound": {
                        "rounds": 0,
                        "last_attempted": 0,
                        "last_accepted": 0,
                        "last_failed": 0,
                        "last_status": null,
                        "last_failure_reason": null
                    }
                }
            }
        });
        let snapshot: HealthSnapshot =
            serde_json::from_value(fixture.clone()).expect("health fixture");

        let error = snapshot
            .ensure_idle_two_hop_ready()
            .expect_err("missing custody durability must fail before creating sessions");
        assert!(error
            .to_string()
            .contains("custody durability is unavailable"));

        fixture["chat_relay_status"]["peer_relay"]["custody_durability"] = serde_json::json!({
            "state": "normal",
            "full_durability_verified": false,
            "synchronous_level": 1
        });
        let downgraded: HealthSnapshot =
            serde_json::from_value(fixture).expect("downgraded health fixture");
        let error = downgraded
            .ensure_idle_two_hop_ready()
            .expect_err("NORMAL custody must fail before creating sessions");
        assert!(error.to_string().contains("not FULL-or-stronger"));
    }

    #[test]
    fn idle_two_hop_preflight_requires_route_class_health_contract() {
        let snapshot: HealthSnapshot = serde_json::from_value(serde_json::json!({
            "status": "ok",
            "active_sessions": 0,
            "privacy_protocol_health": { "failed_checks": 0 },
            "discovery_status": {
                "peer_store": {
                    "blind_relay_quality": {
                        "verified_client_onion_deliveries": 0,
                        "delivery_receipt_capable_peers": 2,
                        "authenticated_delivery_path_ready": true,
                        "authenticated_delivery_path_reason": "authenticated_receipt_path_ready"
                    },
                    "peer_quorum": { "quorum_ready": true },
                    "route_governance": { "route_pool_ready": true },
                    "network_story": { "chat_two_hop_onion_ready": true }
                }
            },
            "chat_relay_status": {
                "configured_enabled": true,
                "runtime_ready": true,
                "peer_relay": {
                    "custody_durability": {
                        "state": "full",
                        "full_durability_verified": true,
                        "synchronous_level": 2
                    }
                }
            }
        }))
        .expect("health fixture");

        let error = snapshot
            .ensure_idle_two_hop_ready()
            .expect_err("ambiguous relay status must fail before creating sessions");
        assert!(error
            .to_string()
            .contains("authenticated onion relay health is unavailable"));
    }

    #[tokio::test]
    async fn udp_handshake_pins_server_and_derives_same_session_key() {
        let server_socket = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("bind server socket");
        let server_addr = server_socket.local_addr().expect("server address");
        let server_identity = IdentityKeyPair::generate();
        let expected_server_key = server_identity.public_key_bytes();
        let server_task = tokio::spawn(async move {
            let mut buffer = [0u8; 256];
            let (received, peer) = server_socket
                .recv_from(&mut buffer)
                .await
                .expect("receive ClientHello");
            let hello = decode_client_hello(&buffer[..received]).expect("decode ClientHello");
            let crypto = DefaultHandshakeCrypto::new(server_identity);
            crypto
                .verify_client_hello(&hello)
                .expect("verify ClientHello");
            let (response, session_key) = crypto
                .process_handshake(&hello, [100, 64, 0, 2], [0x41; 16])
                .expect("process handshake");
            let bytes = aeronyx_core::protocol::codec::encode_server_hello(&response);
            server_socket
                .send_to(&bytes, peer)
                .await
                .expect("send ServerHello");
            session_key
        });

        let client = RelaySmokeClient::connect(
            server_addr,
            &expected_server_key,
            IdentityKeyPair::generate(),
            TokioInstant::now() + Duration::from_secs(2),
        )
        .await
        .expect("connect smoke client");
        let server_session_key = server_task.await.expect("join server task");

        assert_eq!(client.session_key.as_bytes(), server_session_key.as_bytes());
        assert_eq!(client.session_id, [0x41; 16]);
    }
}
