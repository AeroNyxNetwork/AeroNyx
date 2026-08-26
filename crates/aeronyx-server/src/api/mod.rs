// ============================================
// File: crates/aeronyx-server/src/api/mod.rs
// ============================================
//! # MemChain Local API
//!
//! ## Creation Reason
//! Provides a local HTTP API for trusted node-local clients to read and write
//! memory Facts into the MemChain ledger. The API binds to loopback by default
//! (`127.0.0.1:8421`) and is NOT exposed to the public network.
//!
//! ## v2.4.0 File Split
//! mpi.rs was split into 3 files for maintainability:
//! - `mpi.rs` — MpiState, AuthenticatedOwner, auth middleware, router, helpers
//! - `mpi_handlers.rs` — Original 7 endpoint handlers (remember, recall, forget,
//!   status, embed, record, overview)
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints (11 new)
//!
//! External API is UNCHANGED — this file re-exports the same symbols:
//! `{build_mpi_router, MpiState, BaselineSnapshot}`
//!
//! server.rs, log_handler.rs, and ws_client.rs all import from
//! `crate::api::mpi::` — their code does NOT need to change.
//!
//! ## Submodules
//! - [`mpi`]: Core MPI types, auth, router (entry point)
//! - [`mpi_handlers`]: Original endpoint handlers (remember, recall, etc.)
//! - [`mpi_graph_handlers`]: v2.4.0 cognitive graph endpoints
//! - [`recall_handler`]: Hybrid recall pipeline (vector + BM25 + graph + RRF)
//! - [`log_handler`]: /log endpoint with rule engine + entropy filter + privacy tags
//! - [`supernode_handlers`]: v2.5.0 SuperNode management endpoints
//! - [`auth`]: v1.0.0-MultiTenant JWT token issuance for SaaS mode
//! - [`admin_handlers`]: v1.0.0-MultiTenant Admin endpoints (volumes, pool, usage)
//! - [`local`]: Legacy Axum router (deprecated)
//! - [`voice`]: v1.0.0-Voice Peer virtual IP resolution for UDP direct-connect
//! - [`chat_handlers`]: client/VPN-only encrypted media blob transfer
//! - [`discovery`]: v0.1.0 Discovery snapshot/gossip endpoints
//! - [`directory_chain_peer`]: signed bounded Directory Chain peer transport
//! - [`directory_replica_status`]: privacy-tiered replica health endpoint
//! - [`directory_replica_sync`]: bounded concurrent outbound replica coordinator
//! - [`chat_peer`]: v0.1.0 node-to-node encrypted chat envelope relay
//! - `chat_peer_admission`: private direct-peer admission and ACK replay domain
//! - `chat_peer_abuse_guard`: private blind-relay abuse-control domain
//! - `chat_peer_observer`: private aggregate forward observation capability
//! - `chat_peer_replay`: private generation-fenced blind-route replay domain
//! - `chat_peer_retry`: private payload-blind forwarding retry policy domain
//! - `chat_peer_response`: private receipt and response decision domain
//! - `chat_peer_transport`: private bounded blind-relay HTTP transport domain
//! - [`blind_vault`]: node-blind encrypted object lease/store/recovery routes
//! - [`memchain_peer`]: v2.7.0 signed node-to-node commitment block ranges
//!
//! ⚠️ Important Note for Next Developer:
//! - When adding new ORIGINAL-style endpoints → add to mpi_handlers.rs
//! - When adding new GRAPH/COGNITIVE endpoints → add to mpi_graph_handlers.rs
//! - When adding new SUPERNODE endpoints → add to supernode_handlers.rs
//! - When adding new ADMIN endpoints → add to admin_handlers.rs
//! - Register all routes in mpi.rs::build_mpi_router() regardless of which file
//!   the handler lives in
//! - Re-exports below MUST stay in sync — server.rs depends on them
//! - auth.rs and admin_handlers.rs are SaaS-mode only but always compiled.
//!   The routes are conditionally registered in build_mpi_router() based on mode.
//! - voice.rs injects its own Arc<SessionManager> State independently of MpiState.
//!   It is merged into the combined API router in server.rs::start_combined_api().
//! - memchain_peer.rs is a public node-peer surface, not a client memory API.
//!   It must keep PeerStore admission and return commitments only.
//! - chat_handlers.rs is a client media surface. Mount it only on loopback/VPN
//!   listeners; never merge it into the public node-peer router.
//! - Every outbound peer response must be read through the bounded helpers in
//!   this module. `Content-Length` is advisory; the streaming byte count is the
//!   authoritative memory boundary and peer-controlled bodies are never logged.
//! - Public request handlers that buffer or hash attacker-controlled bodies
//!   must acquire `InFlightRequestGuard` before extraction. Keep independent
//!   counters for workloads that should not starve each other.
//! - [CHAT-PEER-ADMISSION-DOMAIN 2026-08-26 by Codex] Direct peer admission
//!   policy is composed privately; keep user and route data out of its keys.
//! - [CHAT-PEER-ABUSE-DOMAIN 2026-08-26 by Codex] Blind-relay abuse policy is
//!   composed privately; keep payload, user, route, endpoint, and IP data out.
//! - [BLIND-REPLAY-DOMAIN 2026-08-26 by Codex] Process-local route ownership
//!   is generation-fenced; stale leases must never mutate a newer owner.
//! - [BLIND-RETRY-DOMAIN 2026-08-26 by Codex] Retry policy may use only coarse
//!   transport state and signed route metadata, never payload or user data.
//! - [BLIND-TRANSPORT-DOMAIN 2026-08-26 by Codex] Outbound HTTP adapters own
//!   bounded response decoding but no routing, receipt, or health decisions.
//! - [BLIND-RESPONSE-DOMAIN 2026-08-26 by Codex] Response policy validates
//!   receipts and returns decisions but owns no I/O, clocks, logs, or storage.
//! - [BLIND-FORWARD-OBSERVER 2026-08-26 by Codex] Forward observations are
//!   write-only aggregate effects and never influence relay control decisions.
//!
//! ## Last Modified
//! v0.3.0 - Initial Agent API for MemChain Phase 1
//! v0.4.0 - Extended for Phase 3: P2P broadcast + POST /api/sync
//! v2.4.0-GraphCognition - Split mpi.rs into 3 files; added mpi_handlers,
//!   mpi_graph_handlers, recall_handler submodules
//! v2.4.0+Privacy - log_handler updated with privacy tag stripping
//! v2.5.0+SuperNode Phase D - Added supernode_handlers submodule
//! v1.0.0-MultiTenant - Added auth + admin_handlers submodules for SaaS mode;
//!   MpiState extended with Mode enum + SaaS pool fields;
//!   build_mpi_router conditionally registers auth + admin routes in SaaS mode.
//! v1.0.0-Voice - Added voice submodule:
//!   GET /api/peer-virtual-ip?pubkey=<hex> → { online, virtual_ip, last_seen }
//!   Two-pass lookup: wallet_index (O(1)) → all_sessions fallback (O(n)).
//!   No auth required (virtual IP is network-layer routing info, not PII).
//! v0.1.0-DiscoveryAPI - Added discovery submodule:
//!   GET /api/discovery/snapshot and POST /api/discovery/gossip.
//! v0.1.0-ChatPeerRelay - Added chat_peer submodule:
//!   POST /api/chat/peer/relay for inter-node encrypted envelope relay.
//! v2.7.0-BlockSync - Added authenticated `/api/memchain/peer/block-range`.
//! v2.7.19-PublicApiBounds - Centralized bounded peer HTTP response decoding.
//! v2.7.20-PublicApiBackpressure - Centralized lock-free RAII request permits.
//! v2.7.21-ChatBlobWiring - Compile the encrypted client blob API module.
//! v2.8.24-DirectorySyncServing - Compile authenticated Directory Chain peer routes.
//! v2.8.29-DirectoryReplicaCoordinator - Split replica scheduling from server lifecycle.
//! v1.0.0-BlindVaultApi - Added bounded binary Blind Vault client routes.
//! v2.8.30-PeerEndpointPolicy - Centralized canonical peer URL parsing and
//!   public-IP-only SSRF protection for permissionless outbound transports.
//! v2.8.31-ChatPeerAdmissionDomain - Split direct-peer fairness and exact ACK
//!   replay ownership from public HTTP orchestration.
//! v2.8.32-ChatPeerAbuseDomain - Split blind-relay rate, quarantine, and fixed
//!   identity-capacity ownership from public HTTP orchestration.
//! v2.8.33-ChatPeerReplayDomain - Split and generation-fence process-local
//!   blind-route replay ownership.
//! v2.8.34-ChatPeerReplayCodec - Move versioned durable ACK storage rules into
//!   the replay domain while retaining legacy sealed-row reads.
//! v2.8.35-ChatPeerRetryDomain - Split payload-blind retry decisions from HTTP
//!   forwarding, telemetry, and route-health effects.
//! v2.8.36-ChatPeerTransportDomain - Split bounded reqwest I/O from blind-relay
//!   routing, receipt validation, retry policy, and route-health effects.
//! v2.8.37-ChatPeerResponseDomain - Split receipt verification and response
//!   interpretation from asynchronous forwarding and observability effects.
//! v2.8.38-ChatPeerObserverDomain - Split aggregate retry and route-health
//!   persistence from forwarding control behind a write-only observer trait.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use serde::de::DeserializeOwned;

/// Structural failures while deriving a canonical outbound peer URL.
///
/// The type intentionally carries no attacker-controlled endpoint text so it
/// remains safe to map into public health and operator telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeerEndpointUrlError {
    /// The endpoint was empty after trimming.
    Missing,
    /// The endpoint was not a credential-free HTTP(S) URL with a host.
    Invalid,
}

/// Builds one canonical HTTP(S) URL for an outbound peer protocol route.
///
/// [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] Centralizing this parser prevents
/// discovery, `MemChain`, and future peer transports from disagreeing about
/// credentials, paths, queries, fragments, host casing, or default ports.
/// This function validates URL structure only. Permissionless descriptors
/// must additionally pass [`peer_endpoint_is_public_ip`] before transport.
pub(crate) fn canonical_peer_http_url(
    endpoint: &str,
    path: &str,
) -> Result<reqwest::Url, PeerEndpointUrlError> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return Err(PeerEndpointUrlError::Missing);
    }
    let normalized = if endpoint.contains("://") {
        endpoint.to_string()
    } else {
        format!("http://{endpoint}")
    };
    let mut url = reqwest::Url::parse(&normalized).map_err(|_| PeerEndpointUrlError::Invalid)?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
    {
        return Err(PeerEndpointUrlError::Invalid);
    }
    url.set_path(path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

/// Starts a fail-closed client builder for permissionless peer transports.
///
/// [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] Callers add their own timeout and
/// pool limits, while this shared base makes proxy inheritance and redirects
/// impossible to re-enable accidentally on discovery, relay, onion, or
/// `MemChain` traffic.
pub(crate) fn privacy_safe_peer_http_client_builder() -> reqwest::ClientBuilder {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
}

/// Accepts only public IP literals for permissionless outbound peer traffic.
///
/// A descriptor signature authenticates the advertiser, not the destination's
/// safety for this host. Domain names are excluded to prevent DNS rebinding;
/// loopback, private, link-local, CGNAT, benchmark, documentation, multicast,
/// and reserved ranges are rejected as well.
pub(crate) fn peer_endpoint_is_public_ip(endpoint: &str) -> bool {
    let Some(address) = peer_endpoint_ip_literal(endpoint) else {
        return false;
    };
    match address {
        IpAddr::V4(address) => ipv4_is_public_unicast(address),
        IpAddr::V6(address) => ipv6_is_public_unicast(address),
    }
}

/// Localhost-only seam for integration tests that bind ephemeral listeners.
///
/// Production peer transports never call this function. Tests still exercise
/// the same canonical parser while the public-address policy has independent
/// regression coverage in [`peer_endpoint_is_public_ip`].
#[cfg(test)]
pub(crate) fn peer_endpoint_is_loopback_ip(endpoint: &str) -> bool {
    peer_endpoint_ip_literal(endpoint).is_some_and(|address| address.is_loopback())
}

fn peer_endpoint_ip_literal(endpoint: &str) -> Option<IpAddr> {
    let url = canonical_peer_http_url(endpoint, "/").ok()?;
    let host = url.host_str()?;
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    host.parse().ok()
}

fn ipv4_is_public_unicast(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224)
}

fn ipv6_is_public_unicast(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4() {
        return ipv4_is_public_unicast(mapped);
    }
    let segments = address.segments();
    (segments[0] & 0xe000) == 0x2000 && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
}

/// One lock-free permit for a bounded class of in-flight public requests.
///
/// The counter is shared by cloned Axum state. Acquisition uses
/// compare-and-exchange so concurrent requests never overshoot the limit,
/// and `Drop` releases the permit on every return path. This type deliberately
/// does not own a semaphore wait queue: public callers receive immediate
/// backpressure instead of retaining request bodies while waiting for memory.
pub(crate) struct InFlightRequestGuard {
    counter: Arc<AtomicUsize>,
}

impl InFlightRequestGuard {
    /// Attempts to reserve one in-flight slot without blocking.
    pub(crate) fn try_acquire(counter: &Arc<AtomicUsize>, limit: usize) -> Option<Self> {
        let counter = Arc::clone(counter);
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Self { counter }),
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for InFlightRequestGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Privacy-safe failure classes for bounded responses from untrusted peers.
///
/// Deliberately avoid carrying response bodies or parser details: callers may
/// expose these reasons through health telemetry, and peer-controlled content
/// must never become an accidental logging channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundedHttpResponseError {
    /// The declared or streamed response exceeded its protocol ceiling.
    TooLarge,
    /// The response stream failed before a complete bounded body was read.
    BodyRead,
    /// The bounded body did not match the expected JSON response schema.
    JsonDecode,
}

impl BoundedHttpResponseError {
    /// Returns a stable privacy-safe telemetry bucket.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::TooLarge => "response_too_large",
            Self::BodyRead => "response_body_read_failed",
            Self::JsonDecode => "response_json_decode_failed",
        }
    }
}

impl std::fmt::Display for BoundedHttpResponseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Relay acknowledgements contain only booleans, counters, and reason codes.
pub(crate) const PEER_ACK_RESPONSE_MAX_BYTES: usize = 16 * 1024;

/// Reads an untrusted HTTP response without allowing a peer to grow the
/// process heap without bound.
///
/// `Content-Length` is only an early rejection. The streaming check remains
/// authoritative because the header may be absent, incorrect, or refer to
/// compressed bytes.
pub(crate) async fn read_bounded_http_response(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, BoundedHttpResponseError> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(BoundedHttpResponseError::TooLarge);
    }

    let initial_capacity = response
        .content_length()
        .unwrap_or_default()
        .min(max_bytes as u64) as usize;
    let mut body = Vec::with_capacity(initial_capacity);
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| BoundedHttpResponseError::BodyRead)?
    {
        if chunk.len() > max_bytes.saturating_sub(body.len()) {
            return Err(BoundedHttpResponseError::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Decodes one schema-checked JSON response after enforcing its byte ceiling.
pub(crate) async fn decode_bounded_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<T, BoundedHttpResponseError> {
    let body = read_bounded_http_response(response, max_bytes).await?;
    serde_json::from_slice(&body).map_err(|_| BoundedHttpResponseError::JsonDecode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_flight_request_guard_is_bounded_and_releases_on_drop() {
        let counter = Arc::new(AtomicUsize::new(0));

        let first = InFlightRequestGuard::try_acquire(&counter, 2).expect("first permit");
        let second = InFlightRequestGuard::try_acquire(&counter, 2).expect("second permit");
        assert_eq!(counter.load(Ordering::Acquire), 2);
        assert!(InFlightRequestGuard::try_acquire(&counter, 2).is_none());

        drop(first);
        assert_eq!(counter.load(Ordering::Acquire), 1);
        let replacement =
            InFlightRequestGuard::try_acquire(&counter, 2).expect("replacement permit");
        drop((second, replacement));
        assert_eq!(counter.load(Ordering::Acquire), 0);
    }

    #[test]
    fn in_flight_request_guard_rejects_zero_capacity() {
        let counter = Arc::new(AtomicUsize::new(0));
        assert!(InFlightRequestGuard::try_acquire(&counter, 0).is_none());
        assert_eq!(counter.load(Ordering::Acquire), 0);
    }

    #[test]
    fn canonical_peer_http_url_strips_untrusted_url_components() -> Result<(), PeerEndpointUrlError>
    {
        let url = canonical_peer_http_url(
            " HTTPS://Node.Example:443/untrusted/path?token=secret#fragment ",
            "/api/discovery/gossip",
        )?;
        assert_eq!(url.as_str(), "https://node.example/api/discovery/gossip");
        assert_eq!(
            canonical_peer_http_url("  ", "/api/discovery/gossip"),
            Err(PeerEndpointUrlError::Missing)
        );
        for endpoint in [
            "ftp://8.8.8.8",
            "https://user@8.8.8.8",
            "https://user:password@8.8.8.8",
            "http://",
        ] {
            assert_eq!(
                canonical_peer_http_url(endpoint, "/api/discovery/gossip"),
                Err(PeerEndpointUrlError::Invalid),
                "unexpectedly accepted {endpoint}"
            );
        }
        Ok(())
    }

    #[test]
    fn permissionless_peer_endpoint_rejects_ssrf_targets() {
        assert!(peer_endpoint_is_public_ip("http://8.8.8.8:8422"));
        assert!(peer_endpoint_is_public_ip(
            "https://[2606:4700:4700::1111]:8422"
        ));
        for endpoint in [
            "http://127.0.0.1:8422",
            "http://127.1:8422",
            "http://2130706433:8422",
            "http://0x7f000001:8422",
            "http://017700000001:8422",
            "http://10.0.0.1:8422",
            "http://100.64.0.1:8422",
            "http://169.254.1.1:8422",
            "http://172.16.0.1:8422",
            "http://192.168.1.1:8422",
            "http://198.18.0.1:8422",
            "http://203.0.113.1:8422",
            "http://node.example:8422",
            "http://[::1]:8422",
            "http://[::ffff:127.0.0.1]:8422",
            "http://[fc00::1]:8422",
            "http://[fe80::1]:8422",
            "http://[2001:db8::1]:8422",
        ] {
            assert!(
                !peer_endpoint_is_public_ip(endpoint),
                "unexpectedly accepted {endpoint}"
            );
        }
    }
}

// ── Core MPI module (state, auth, router) ──
pub mod mpi;
// ── Handler modules ──
pub mod mpi_graph_handlers;
pub mod mpi_handlers;
pub mod recall_handler;
// ── /log endpoint ──
pub mod log_handler;
// ── v2.5.0+SuperNode: Task queue management + monitoring ──
pub mod supernode_handlers;
// ── v1.0.0-MultiTenant: JWT token issuance (SaaS mode only, always compiled) ──
pub mod auth;
// ── v1.0.0-MultiTenant: Admin endpoints (SaaS mode only, always compiled) ──
pub mod admin_handlers;
// ── Legacy API (deprecated) ──
pub mod local;
// ── v1.0.0-Voice: Peer virtual IP resolution for UDP direct-connect routing ──
pub mod blind_vault;
pub mod chat_handlers;
pub mod chat_peer;
mod chat_peer_abuse_guard;
mod chat_peer_admission;
mod chat_peer_observer;
mod chat_peer_replay;
mod chat_peer_response;
mod chat_peer_retry;
mod chat_peer_transport;
pub mod directory_chain_peer;
pub mod directory_replica_status;
pub mod directory_replica_sync;
pub mod discovery;
pub mod memchain_peer;
pub mod voice;
pub mod vpn_health;

// ── Re-exports (unchanged from v2.3.0 — external callers unaffected) ──
pub use mpi::{build_mpi_router, BaselineSnapshot, MpiState};
// v1.0.0-MultiTenant: export Mode for server.rs SaaS init branch
#[allow(deprecated)]
pub use local::start_legacy_api_server;
pub use mpi::Mode;
