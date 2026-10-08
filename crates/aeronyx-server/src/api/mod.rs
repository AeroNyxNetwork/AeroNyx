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
//! - `discovery_endpoint_verification`: unmounted authenticated Stage C adapter
//! - [`directory_chain_peer`]: signed bounded Directory Chain peer transport
//! - [`directory_replica_status`]: privacy-tiered replica health endpoint
//! - [`directory_replica_sync`]: bounded concurrent outbound replica coordinator
//! - [`chat_peer`]: v0.1.0 node-to-node encrypted chat envelope relay
//! - `chat_anonymous_mailbox_source`: authenticated VPN-only exact-target
//!   anonymous-mailbox source composition; never mount on peer/public routers
//! - `chat_peer_admission`: private direct-peer admission and ACK replay domain
//! - `chat_peer_abuse_guard`: private blind-relay abuse-control domain
//! - `chat_peer_observer`: private aggregate forward observation capability
//! - `chat_peer_replay`: private generation-fenced blind-route replay domain
//! - `chat_peer_retry`: private payload-blind forwarding retry policy domain
//! - `chat_peer_response`: private receipt and response decision domain
//! - `chat_peer_transport`: private bounded blind-relay HTTP transport domain
//! - `chat_peer_terminal_reply`: private fixed-size terminal response domain
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
use sha2::{Digest, Sha256};

// [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Shared by source and
// adjacent-hop HTTP owners. A response retains its already-acquired permit,
// never allocates another slot, and never changes a durable task deadline.
pub(crate) const REVERSE_ONION_RESPONSE_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);
pub(crate) const REVERSE_ONION_RESPONSE_CHUNK_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, Copy)]
pub(crate) enum ReverseOnionResponseError { Busy, Unavailable }

pub(crate) struct ReverseOnionResponseRegistry {
    responses: std::sync::Mutex<Vec<std::sync::Weak<ReverseOnionResponseState>>>,
    max_responses: usize,
    changed: tokio::sync::Notify,
}

impl ReverseOnionResponseRegistry {
    pub(crate) fn new(max_responses: usize) -> Self {
        Self { responses: std::sync::Mutex::new(Vec::new()),
            max_responses, changed: tokio::sync::Notify::new() }
    }

    pub(crate) fn bound_body(&self, body: axum::body::Body,
        permit: impl Send + Sync + 'static, max_bytes: usize,
    ) -> Result<axum::body::Body, ReverseOnionResponseError> {
        self.expire();
        let mut responses = self.responses.lock()
            .map_err(|_| ReverseOnionResponseError::Unavailable)?;
        if max_bytes == 0 || responses.len() >= self.max_responses {
            return Err(ReverseOnionResponseError::Busy);
        }
        let deadline = tokio::time::Instant::now() + REVERSE_ONION_RESPONSE_TIMEOUT;
        let state = Arc::new(ReverseOnionResponseState {
            data: std::sync::Mutex::new(Some(ReverseOnionResponseData {
                stream: body.into_data_stream(), pending: axum::body::Bytes::new(),
                remaining: max_bytes, _permit: Box::new(permit),
            })),
            deadline, expired: std::sync::atomic::AtomicBool::new(false),
            waker: futures::task::AtomicWaker::new(),
        });
        responses.push(Arc::downgrade(&state));
        self.changed.notify_one();
        Ok(axum::body::Body::from_stream(ReverseOnionResponseStream {
            state, deadline: Box::pin(tokio::time::sleep_until(deadline)), terminated: false,
        }))
    }

    // Weak entries cannot retain abandoned bodies. Admission, polling and drain
    // reclaim the same idle allocation; no unbounded background timers are added.
    pub(crate) fn expire(&self) -> Option<tokio::time::Instant> {
        let now = tokio::time::Instant::now();
        let mut earliest = None;
        let mut responses = self.responses.lock().unwrap_or_else(|error| error.into_inner());
        responses.retain(|entry| {
            let Some(state) = entry.upgrade() else { return false; };
            if now >= state.deadline { state.finish(true); }
            if !state.active() { return false; }
            earliest = Some(earliest.map_or(state.deadline,
                |at: tokio::time::Instant| at.min(state.deadline)));
            true
        });
        earliest
    }

    // Caller closes intake first. Cancellation leaves bodies and their original
    // deadlines registered, so a later drain still observes accepted work.
    pub(crate) async fn drain(&self, permits: Arc<tokio::sync::Semaphore>, capacity: u32) {
        let all = permits.acquire_many_owned(capacity);
        tokio::pin!(all);
        loop {
            let deadline = self.expire().unwrap_or_else(||
                tokio::time::Instant::now() + REVERSE_ONION_RESPONSE_TIMEOUT);
            tokio::select! {
                _all = &mut all => return,
                _ = self.changed.notified() => {},
                _ = tokio::time::sleep_until(deadline) => {},
            }
        }
    }
}

struct ReverseOnionResponseData {
    stream: axum::body::BodyDataStream,
    pending: axum::body::Bytes,
    remaining: usize,
    _permit: Box<dyn Send + Sync>,
}

struct ReverseOnionResponseState {
    data: std::sync::Mutex<Option<ReverseOnionResponseData>>,
    deadline: tokio::time::Instant,
    expired: std::sync::atomic::AtomicBool,
    waker: futures::task::AtomicWaker,
}

impl ReverseOnionResponseState {
    fn finish(&self, expired: bool) {
        if expired { self.expired.store(true, Ordering::Release); }
        let data = self.data.lock().unwrap_or_else(|error| error.into_inner()).take();
        // Drop the large frame and permit outside the lock, before waking HTTP.
        drop(data);
        self.waker.wake();
    }

    fn active(&self) -> bool {
        self.data.lock().unwrap_or_else(|error| error.into_inner()).is_some()
    }
}

struct ReverseOnionResponseStream {
    state: Arc<ReverseOnionResponseState>,
    deadline: std::pin::Pin<Box<tokio::time::Sleep>>,
    terminated: bool,
}

impl futures::Stream for ReverseOnionResponseStream {
    type Item = Result<axum::body::Bytes, axum::Error>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        use std::future::Future;
        use futures::Stream;
        if self.terminated { return std::task::Poll::Ready(None); }
        self.state.waker.register(cx.waker());
        if self.deadline.as_mut().poll(cx).is_ready() { self.state.finish(true); }
        let state = Arc::clone(&self.state);
        let mut slot = state.data.lock().unwrap_or_else(|error| error.into_inner());
        let Some(data) = slot.as_mut() else {
            drop(slot);
            self.terminated = true;
            if state.expired.load(Ordering::Acquire) {
                return std::task::Poll::Ready(Some(Err(axum::Error::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut, "reverse onion response delivery expired",
                )))));
            }
            return std::task::Poll::Ready(None);
        };
        // Empty frames must not let a malformed stream monopolize the executor.
        for _ in 0..8 {
            if !data.pending.is_empty() {
                let size = data.pending.len().min(REVERSE_ONION_RESPONSE_CHUNK_BYTES);
                // A shared Bytes slice would keep the entire allocation alive
                // in a slow socket after expiry. Detach only this bounded chunk.
                let chunk = axum::body::Bytes::copy_from_slice(&data.pending[..size]);
                data.pending = if size == data.pending.len() { axum::body::Bytes::new() }
                    else { data.pending.slice(size..) };
                return std::task::Poll::Ready(Some(Ok(chunk)));
            }
            match std::pin::Pin::new(&mut data.stream).poll_next(cx) {
                std::task::Poll::Pending => return std::task::Poll::Pending,
                std::task::Poll::Ready(Some(Ok(bytes))) if bytes.len() <= data.remaining => {
                    data.remaining -= bytes.len();
                    data.pending = bytes;
                }
                std::task::Poll::Ready(Some(Ok(_))) => {
                    drop(slot);
                    state.finish(false);
                    self.terminated = true;
                    return std::task::Poll::Ready(Some(Err(axum::Error::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidData, "reverse onion response exceeds bound",
                    )))));
                }
                std::task::Poll::Ready(Some(Err(error))) => {
                    drop(slot);
                    state.finish(false);
                    self.terminated = true;
                    return std::task::Poll::Ready(Some(Err(error)));
                }
                std::task::Poll::Ready(None) => {
                    drop(slot);
                    state.finish(false);
                    self.terminated = true;
                    return std::task::Poll::Ready(None);
                }
            }
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    }
}

impl Drop for ReverseOnionResponseStream {
    fn drop(&mut self) { self.state.finish(false); }
}

// [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] This timeout covers
// only pre-effect request buffering, never DB work, execution or result drain.
pub(crate) const REVERSE_ONION_REQUEST_BODY_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(10);

// [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Call only after the
// route's DefaultBodyLimit and bounded admission have been installed. Reuse
// Axum's Bytes extractor/rejections so per-route byte caps and 413 responses
// stay intact. Preserve the original headers/auth/permit extensions for the
// handler, without cloning or buffering attacker data outside that admission.
pub(crate) async fn buffer_reverse_onion_request(
    mut request: axum::extract::Request,
) -> Result<axum::extract::Request, axum::response::Response> {
    use axum::extract::FromRequest;
    use axum::response::IntoResponse;
    let body = std::mem::replace(request.body_mut(), axum::body::Body::empty());
    let mut buffering = axum::extract::Request::new(body);
    *buffering.extensions_mut() = request.extensions().clone();
    match tokio::time::timeout(REVERSE_ONION_REQUEST_BODY_TIMEOUT,
        axum::body::Bytes::from_request(buffering, &())).await {
        Ok(Ok(bytes)) => {
            *request.body_mut() = axum::body::Body::from(bytes);
            Ok(request)
        }
        Ok(Err(rejection)) => Err(rejection.into_response()),
        Err(_) => Err(axum::http::StatusCode::REQUEST_TIMEOUT.into_response()),
    }
}

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

/// [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] A route-specific client
/// retains the URL hostname for TLS SNI/certificate verification while
/// forcing the connector to the public addresses resolved for this request.
#[derive(Clone)]
pub(crate) struct PinnedPeerHttpTarget {
    pub(crate) client: reqwest::Client,
    pub(crate) url: reqwest::Url,
}

// [PHALA-DNS-SEED-PIN 2026-10-07 by Codex] Operator-configured HTTPS DNS
// bootstrap targets use the same public-answer pinning as attested peers.
pub(crate) fn peer_http_target_requires_dns_pin(url: &reqwest::Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    url.scheme() == "https"
        && host
            .trim_start_matches('[')
            .trim_end_matches(']')
            .parse::<IpAddr>()
            .is_err()
}

/// Reverse-onion alone may use a signed HTTPS DNS endpoint. Ordinary
/// permissionless transports remain restricted to IP literals.
// [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex]
pub(crate) fn reverse_onion_endpoint_supported(endpoint: &str) -> bool {
    let Ok(url) = canonical_peer_http_url(endpoint, "/") else {
        return false;
    };
    // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] Relay acknowledgements
    // retire durable claims/results, so this private transport must always
    // authenticate the peer and response body with TLS, including IP routes.
    if url.scheme() != "https" || url.port() == Some(0) {
        return false;
    }
    if peer_endpoint_is_public_ip(endpoint) {
        return true;
    }
    is_public_dns_https_url(&url)
}

// [REVERSE-ONION-ORIGIN-BINDING 2026-10-06 by Codex] Bind durable exact-frame
// retries to the signed transport origin, while excluding path and DNS answers.
pub(crate) fn reverse_onion_origin_commitment(
    endpoint: &str,
) -> Result<[u8; 32], PeerEndpointUrlError> {
    let url = canonical_peer_http_url(endpoint, "/")?;
    if url.scheme() != "https" || !reverse_onion_endpoint_supported(endpoint) {
        return Err(PeerEndpointUrlError::Invalid);
    }
    let host = url.host_str().ok_or(PeerEndpointUrlError::Invalid)?;
    let port = url.port_or_known_default().ok_or(PeerEndpointUrlError::Invalid)?;
    let host_len = u16::try_from(host.len()).map_err(|_| PeerEndpointUrlError::Invalid)?;
    let mut digest = Sha256::new();
    digest.update(b"AeroNyx-ReverseOnion-TransportOrigin-v1");
    digest.update((url.scheme().len() as u16).to_be_bytes());
    digest.update(url.scheme().as_bytes());
    digest.update(host_len.to_be_bytes());
    digest.update(host.as_bytes());
    digest.update(port.to_be_bytes());
    Ok(digest.finalize().into())
}

// [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Discovery refresh and
// exact-frame transport share the same public HTTPS origin contract. A signed
// identity renewal cannot authorize a new host, port, or cleartext scheme.
pub(crate) fn reverse_onion_pinned_origin(
    endpoint: &str,
    expected: [u8; 32],
) -> Result<[u8; 32], PeerEndpointUrlError> {
    let origin = reverse_onion_origin_commitment(endpoint)?;
    if origin != expected {
        return Err(PeerEndpointUrlError::Invalid);
    }
    Ok(origin)
}

pub(crate) fn reverse_onion_same_origin(left: &str, right: &str) -> bool {
    let Ok(expected) = reverse_onion_origin_commitment(left) else {
        return false;
    };
    reverse_onion_pinned_origin(right, expected).is_ok()
}

/// Resolves and pins one peer target before any durable send boundary.
/// DNS rebinding cannot change the destination after validation; mixed public
/// and private answer sets fail closed. Rustls still verifies the original
/// HTTPS hostname, with proxy inheritance and redirects disabled.
// [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex]
pub(crate) async fn resolve_pinned_peer_http_target(
    url: reqwest::Url,
    timeout: std::time::Duration,
) -> Result<PinnedPeerHttpTarget, PeerEndpointUrlError> {
    // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] This resolver is private
    // to reverse onion; accepting HTTP here would make relay receipts spoofable.
    if timeout.is_zero() || url.scheme() != "https"
        || !url.username().is_empty() || url.password().is_some()
        || url.query().is_some() || url.fragment().is_some()
        || url.port() == Some(0)
    {
        return Err(PeerEndpointUrlError::Invalid);
    }
    let host = url.host_str().ok_or(PeerEndpointUrlError::Invalid)?;
    let ip = host.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>().ok();
    let mut builder = privacy_safe_peer_http_client_builder()
        .connect_timeout(timeout)
        .timeout(timeout);
    if let Some(ip) = ip {
        if !ip_is_public_unicast(ip) {
            return Err(PeerEndpointUrlError::Invalid);
        }
    } else {
        if !is_public_dns_https_url(&url) {
            return Err(PeerEndpointUrlError::Invalid);
        }
        let port = url.port_or_known_default().ok_or(PeerEndpointUrlError::Invalid)?;
        let lookup = tokio::time::timeout(timeout, tokio::net::lookup_host((host, port)))
            .await
            .map_err(|_| PeerEndpointUrlError::Invalid)?
            .map_err(|_| PeerEndpointUrlError::Invalid)?;
        let addresses = validated_public_dns_addresses(
            lookup, port,
        ).ok_or(PeerEndpointUrlError::Invalid)?;
        builder = builder.resolve_to_addrs(host, &addresses);
    }
    let client = builder.build().map_err(|_| PeerEndpointUrlError::Invalid)?;
    Ok(PinnedPeerHttpTarget { client, url })
}

// [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] Validate the complete DNS
// answer set before constructing a client; filtering unsafe answers would
// leave resolver ordering as a destination-selection side channel.
fn validated_public_dns_addresses(
    addresses: impl IntoIterator<Item = std::net::SocketAddr>,
    port: u16,
) -> Option<Vec<std::net::SocketAddr>> {
    let mut pinned = Vec::new();
    for address in addresses {
        if !ip_is_public_unicast(address.ip()) {
            return None;
        }
        let address = std::net::SocketAddr::new(address.ip(), port);
        if !pinned.contains(&address) {
            if pinned.len() == 16 {
                return None;
            }
            pinned.push(address);
        }
    }
    (!pinned.is_empty()).then_some(pinned)
}

fn is_public_dns_https_url(url: &reqwest::Url) -> bool {
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    if url.port() == Some(0) {
        return false;
    }
    let Some(host) = url.host_str() else { return false; };
    let host = host.to_ascii_lowercase();
    if host.parse::<IpAddr>().is_ok()
        || !host.contains('.')
        || host.starts_with('.')
        || host.ends_with('.')
        || host.len() > 253
        // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Only canonical
        // DNS labels may reach resolution; URL parsing alone allows labels
        // such as underscores and leading hyphens.
        || host.split('.').any(|label| {
            label.is_empty()
                || label.len() > 63
                || label.starts_with('-')
                || label.ends_with('-')
                || !label.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        })
        || [".localhost", ".local", ".internal", ".test", ".invalid", ".example", ".onion"]
            .iter().any(|suffix| host.ends_with(suffix))
    {
        return false;
    }
    true
}

fn ip_is_public_unicast(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => ipv4_is_public_unicast(address),
        IpAddr::V6(address) => ipv6_is_public_unicast(address),
    }
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
    ip_is_public_unicast(address)
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
    // [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] A global-unicast
    // prefix alone is insufficient: special-purpose 2001::/23, 6to4's
    // embedded-IPv4 2002::/16, and documentation 3fff::/20 are not valid
    // externally selected peer destinations.
    (segments[0] & 0xe000) == 0x2000
        && !(segments[0] == 0x2001 && segments[1] <= 0x01ff)
        // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex] Documentation
        // 2001:db8::/32 is outside 2001::/23 and needs its own exclusion.
        && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
        && segments[0] != 0x2002
        && !(segments[0] == 0x3fff && (segments[1] & 0xfff0) == 0)
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

/// Relay acknowledgements without opaque terminal data remain compact.
pub(crate) const PEER_ACK_RESPONSE_MAX_BYTES: usize = 16 * 1024;

/// Headroom for bounded Blind Relay JSON fields and signed hop-local receipts.
const BLIND_RELAY_ACK_METADATA_MAX_BYTES: usize = PEER_ACK_RESPONSE_MAX_BYTES;

/// Maximum base64 length of the protocol-bounded opaque terminal response.
const BLIND_RELAY_ACK_OPAQUE_RESPONSE_MAX_BYTES: usize =
    aeronyx_core::protocol::MAX_ONION_SEALED_RESPONSE_BASE64_BYTES;

/// Maximum Blind Relay acknowledgement accepted by its bounded decoder.
///
/// [BLIND-VAULT-LARGE-PULL-TRANSPORT 2026-08-30 by Codex] Derive this ceiling
/// from the core wire bound without widening unrelated peer control responses.
/// The stream reader enforces the resulting cap before JSON decoding.
pub(crate) const BLIND_RELAY_ACK_RESPONSE_MAX_BYTES: usize =
    BLIND_RELAY_ACK_OPAQUE_RESPONSE_MAX_BYTES + BLIND_RELAY_ACK_METADATA_MAX_BYTES;

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
    fn only_https_dns_peer_targets_require_resolution_pinning() {
        assert!(peer_http_target_requires_dns_pin(
            &reqwest::Url::parse("https://seed.example.net/api/discovery/gossip").unwrap()
        ));
        assert!(!peer_http_target_requires_dns_pin(
            &reqwest::Url::parse("https://8.8.8.8/api/discovery/gossip").unwrap()
        ));
        assert!(!peer_http_target_requires_dns_pin(
            &reqwest::Url::parse("https://[2606:4700:4700::1111]/api/discovery/gossip").unwrap()
        ));
        assert!(!peer_http_target_requires_dns_pin(
            &reqwest::Url::parse("http://seed.example.net/api/discovery/gossip").unwrap()
        ));
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

    #[test]
    fn reverse_onion_hostname_policy_is_narrow_and_dns_answers_fail_closed() {
        assert!(reverse_onion_endpoint_supported("https://relay.example.net:443"));
        // [PHALA-GATEWAY-ORIGIN-REGRESSION 2026-10-06 by Codex] Phala's
        // app-port origin must retain HTTPS hostname validation for onion
        // transport; this is not permission for generic peer DNS traffic.
        assert!(reverse_onion_endpoint_supported(
            "https://1e598a2f983dd80c413627e0b50d91905f3f48be-8422.dstack-prod5.phala.network"
        ));
        assert!(reverse_onion_endpoint_supported("https://8.8.8.8:8422"));
        for endpoint in [
            "http://relay.example.net:8422",
            "http://8.8.8.8:8422",
            "https://localhost",
            "https://relay.localhost",
            "https://relay.internal",
            "https://relay.onion",
            "https://a.b.invalid",
            "http://10.0.0.1:8422",
            "https://8.8.8.8:0",
            // [PHALA-RENDER-DNS-PARITY 2026-10-07 by Codex]
            "https://relay..example.net",
            "https://-relay.example.net",
            "https://relay-.example.net",
            "https://relay_name.example.net",
            "https://[2001:db8::1]",
            "https://[ff0e::1]",
        ] {
            assert!(!reverse_onion_endpoint_supported(endpoint), "{endpoint}");
        }
        let oversized_label = format!("https://{}.example.net", "a".repeat(64));
        assert!(!reverse_onion_endpoint_supported(&oversized_label));

        let public = "8.8.8.8:0".parse().expect("socket address");
        assert_eq!(validated_public_dns_addresses([public], 443), Some(vec![
            "8.8.8.8:443".parse().expect("pinned address"),
        ]));
        let mixed = [
            public,
            "127.0.0.1:0".parse().expect("socket address"),
        ];
        assert!(validated_public_dns_addresses(mixed, 443).is_none());
        assert!(validated_public_dns_addresses([], 443).is_none());
        let overbound = (1..=17).map(|last| {
            format!("8.8.8.{last}:0").parse().expect("socket address")
        });
        assert!(validated_public_dns_addresses(overbound, 443).is_none());
    }

    // [REVERSE-ONION-ORIGIN-BINDING 2026-10-06 by Codex] Canonical URL
    // decoration and DNS rotation do not change origin, but authority changes do.
    #[test]
    fn reverse_onion_origin_commitment_tracks_scheme_host_and_effective_port() {
        let canonical = reverse_onion_origin_commitment("https://Relay.Example.net:443/a?x=1").unwrap();
        assert_eq!(canonical, reverse_onion_origin_commitment(" HTTPS://relay.example.net/other ").unwrap());
        assert_ne!(canonical, reverse_onion_origin_commitment("https://other.example.net").unwrap());
        assert_ne!(canonical, reverse_onion_origin_commitment("https://relay.example.net:8443").unwrap());
        // [PHALA-GATEWAY-ORIGIN-REGRESSION 2026-10-06 by Codex] The origin
        // persisted with a Phala onion lease remains hostname-bound, not IP-bound.
        let phala = reverse_onion_origin_commitment(
            "https://1e598a2f983dd80c413627e0b50d91905f3f48be-8422.dstack-prod5.phala.network",
        )
        .unwrap();
        assert_eq!(
            phala,
            reverse_onion_origin_commitment(
                // [PHALA-ORIGIN-FIXTURE-REPAIR 2026-10-08 by Codex] Same
                // hostname, not the distinct phala.net origin.
                " HTTPS://1E598A2F983DD80C413627E0B50D91905F3F48BE-8422.DSTACK-PROD5.PHALA.NETWORK:443/other"
            )
            .unwrap()
        );
        assert_ne!(canonical, phala);
        assert!(reverse_onion_origin_commitment("http://relay.example.net").is_err());
        assert!(reverse_onion_origin_commitment("https://relay.internal").is_err());
    }

    // [PHALA-PRIVATE-EGRESS-ORIGIN 2026-10-07 by Codex] Authored, not run.
    #[test]
    fn private_origin_comparison_rejects_invalid_and_rotated_targets() {
        let pinned = "https://relay.example.net";
        let origin = reverse_onion_origin_commitment(pinned).unwrap();
        let renewed = "https://RELAY.example.net:443/api/discovery/gossip";
        assert_eq!(reverse_onion_pinned_origin(renewed, origin).unwrap(), origin);
        assert!(reverse_onion_same_origin(pinned, renewed));
        for candidate in [
            "https://other.example.net", "https://relay.example.net:8443",
            "http://relay.example.net", "https://relay.internal",
            "https://127.0.0.1", "https://user@relay.example.net",
            "https://relay.example.net:0", "not a URL",
        ] {
            assert!(reverse_onion_pinned_origin(candidate, origin).is_err(), "{candidate}");
            assert!(!reverse_onion_same_origin(pinned, candidate), "{candidate}");
        }
        assert!(!reverse_onion_same_origin("http://relay.example.net", "http://relay.example.net"));
        assert!(!reverse_onion_same_origin("https://relay.internal", "https://relay.internal"));
    }

    // [REVERSE-ONION-HTTPS-ONLY 2026-10-05 by Codex] Reject cleartext before
    // DNS or connection setup, even when a future caller skips endpoint policy.
    #[tokio::test]
    async fn pinned_reverse_onion_resolver_rejects_http_before_network_io() {
        let url = reqwest::Url::parse("http://8.8.8.8:8422/").unwrap();
        assert!(matches!(
            resolve_pinned_peer_http_target(url, std::time::Duration::from_secs(1)).await,
            Err(PeerEndpointUrlError::Invalid),
        ));
    }

    // [REVERSE-ONION-PINNED-HOST 2026-10-05 by Codex] Calibrate the shared
    // address gate with special-purpose ranges and an ordinary global unicast.
    #[test]
    fn public_ipv6_gate_rejects_special_purpose_and_tunnel_ranges() {
        for address in [
            "2001:0:1::1",
            "2001:2::1",
            "2001:db8::1",
            "2002:c000:0201::1",
            "3fff::1",
        ] {
            let address = address.parse().expect("IPv6 address");
            assert!(!ipv6_is_public_unicast(address), "unexpectedly accepted {address}");
        }
        assert!(ipv6_is_public_unicast(
            "2606:4700:4700::1111".parse().expect("global IPv6 address")
        ));
    }
}

// ── Core MPI module (state, auth, router) ──
pub mod mpi;
// [RECIPIENT-STARTUP-WIRING 2026-10-04 by Codex] Compile private capabilities;
// registration here mounts no HTTP route. Queue handlers remain unmounted.
pub(crate) mod reverse_onion;
pub(crate) mod reverse_onion_terminal;
// [REVERSE-ONION-SOURCE-API 2026-10-05 by Codex] VPN/MPI source-only entry.
pub(crate) mod reverse_onion_source;
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
pub mod chat_anonymous_mailbox_source;
pub mod chat_handlers;
pub mod chat_peer;
mod chat_peer_abuse_guard;
mod chat_peer_admission;
mod chat_peer_anonymous_mailbox;
mod chat_peer_observer;
mod chat_peer_replay;
mod chat_peer_response;
mod chat_peer_retry;
mod chat_peer_terminal_reply;
mod chat_peer_transport;
pub mod directory_chain_peer;
pub mod directory_replica_status;
pub mod directory_replica_sync;
pub mod discovery;
// [AUTHENTICATED-ENDPOINT-PROOF-ADAPTER 2026-09-24 by Codex] Its composition
// items remain crate-private and unmounted until verified middleware owns identity.
pub mod discovery_endpoint_verification;
pub mod memchain_peer;
// [PERMISSIONLESS-ENDPOINT-PROOF 2026-09-24 by Codex] Keep public candidate
// authentication composition separate from local and peer API surfaces.
pub(crate) mod public_node_router;
pub mod voice;
pub mod vpn_health;

// ── Re-exports (unchanged from v2.3.0 — external callers unaffected) ──
pub use mpi::{build_mpi_router, BaselineSnapshot, MpiState};
// v1.0.0-MultiTenant: export Mode for server.rs SaaS init branch
#[allow(deprecated)]
pub use local::start_legacy_api_server;
pub use mpi::Mode;
