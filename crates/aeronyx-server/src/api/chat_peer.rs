// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer.rs
// ============================================================================
//! # Inter-Node Encrypted Chat Relay API
//!
//! ## Creation Reason
//! Phase 9 connects AeroNyx node discovery to real encrypted message movement.
//! Discovery tells a node which peers advertise `NodeCapability::ChatRelay`;
//! this module exposes the receiving side for those peers.
//!
//! ## Main Functionality
//! - `POST /api/chat/peer/relay`: accepts a signed `ChatEnvelope` from another
//!   AeroNyx node
//! - `POST /api/chat/peer/blind-relay`: accepts a signed `BlindRelayEnvelope`
//!   and forwards only opaque encrypted bytes toward `next_hop`
//! - Verifies the envelope signature before doing any delivery or storage
//! - Delivers to locally online receiver devices when possible
//! - Falls back to the existing SQLite pending queue when the receiver is
//!   offline or all local routes are stale
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/chat.rs: `ChatEnvelope`, `BlindRelayEnvelope`,
//!   and bounded envelope encoding
//! - aeronyx-core/src/protocol/memchain.rs: wraps envelope for client delivery
//! - aeronyx-server/src/services/chat_relay.rs: pending queue and dedup logic
//! - aeronyx-server/src/services/peer_store.rs: verified node descriptors for
//!   next-hop routing
//! - aeronyx-server/src/services/session.rs: active receiver sessions
//! - aeronyx-transport/src/udp.rs: encrypted client packet send path
//!
//! ## Main Logical Flow
//! 1. Peer node posts an already end-to-end encrypted `ChatEnvelope`
//! 2. This node checks size and sender signature
//! 3. Duplicate message IDs are ignored idempotently
//! 4. Online receiver sessions get the envelope through the existing encrypted
//!    client transport
//! 5. Offline receivers keep the existing pending-message fallback
//! 6. Blind relay requests verify the previous-hop signature, decrement TTL,
//!    re-sign with this node key, and POST to the verified `next_hop`
//!
//! ## Important Note for Next Developer
//! - Never decrypt, inspect, log, store, or report ciphertext contents.
//! - Do not add client public IPs, destination domains, DNS contents, URLs,
//!   browsing history, voucher secrets, private keys, or wallet-level traffic
//!   analytics to this endpoint.
//! - The endpoint is node-to-node plumbing only. Client wire format remains
//!   `MemChainMessage::ChatRelay(ChatEnvelope)`.
//! - Blind relay keeps the relay invariant: route_id / next_hop / ttl /
//!   encrypted_blob / timestamp / signature are handled as routing metadata;
//!   encrypted_blob stays opaque and must not be parsed.
//! - Blind relay rejects immediate self/previous-hop loops using only node-level
//!   route metadata, preserving the "blind relay" invariant while preparing
//!   for future controlled multi-hop/onion routing.
//! - Blind relay keeps a bounded local `route_id` replay cache. The cache is
//!   in-memory only, stores no payload/peer endpoint/user data, and prevents a
//!   repeated encrypted route frame from being forwarded twice by this node.
//! - Blind relay applies previous-hop rate limiting and short quarantine using
//!   only node-level metadata. This protects commercial nodes from relay abuse
//!   while preserving the invariant that encrypted blobs are never parsed.
//! - Blind relay reports privacy-safe previous-hop health buckets to PeerStore
//!   so nodeboard can show protection status without route ids, endpoints,
//!   encrypted blobs, or user metadata.
//! - Blind relay only forwards to peers that explicitly advertise
//!   `NodeCapability::ChatRelay`; valid discovery peers without that capability
//!   are treated as unavailable routes.
//! - Blind relay validates next-hop ACK bodies before marking a route forward
//!   successful. HTTP 2xx with `accepted=false` or an unreadable ACK is treated
//!   as `forward_failed`, preserving delivery correctness without logging
//!   route ids, endpoints, encrypted blobs, or user metadata.
//! - Blind relay tests cover rejected and malformed next-hop ACKs so future
//!   routing work cannot accidentally count a bad HTTP 200 response as
//!   successful encrypted message movement.
//! - Blind relay tests cover unresponsive next-hop endpoints so timeout
//!   failures stay retryable, aggregate-only, and never leak endpoint URLs into
//!   audit status.
//! - Blind relay rejects stale or too-far-in-the-future routing timestamps so
//!   old opaque route frames cannot be replayed indefinitely. This uses only
//!   envelope routing metadata and never inspects encrypted blob contents.
//! - Blind relay supports an optional `onward_envelope` for controlled two-hop
//!   experiments. A middle hop accepts an outer frame addressed to itself, then
//!   forwards only the already-opaque onward frame to its next node without
//!   parsing the encrypted blob or learning any user-level receiver identity.
//! - Blind relay forwards only to peers with fresh routeability evidence.
//!   Signed descriptors prove identity/capability, but routeability probes
//!   prove the next hop can actually receive encrypted relay work.
//! - True onion middle-hop recovery may forward to a fresh signed terminal
//!   descriptor before routeability is proven, but it still refuses route-health
//!   quarantined peers. This breaks cold-start proof deadlocks without relaxing
//!   the ordinary blind relay forwarding gate.
//! - Onion terminal hops must successfully hand the peeled `ChatEnvelope` to
//!   the existing chat relay store-and-forward path before ACKing the previous
//!   hop. A successful peel alone is not enough to claim real encrypted message
//!   movement.
//! - Duplicate route IDs are treated as idempotent replay drops, not previous-hop
//!   abuse. Lost ACK retries must not quarantine an otherwise healthy relay.
//! - Relay logs are route-safe: they must not include message IDs, receiver
//!   prefixes, endpoint URLs, raw transport errors, route IDs, encrypted blobs,
//!   or payload-derived strings. Use stable reason buckets only.
//!
//! ## Last Modified
//! v0.23.0-RouteSafeRelayLogs - Remove user/route-adjacent values and raw transport errors from chat peer logs
//! v0.22.0-BlindRelayDuplicateIdempotence - Keep duplicate route drops out of previous-hop quarantine scoring
//! v0.21.0-OnionMiddleRouteabilityRecovery - Allow true onion middle recovery through fresh signed descriptors unless route-quarantined
//! v0.20.0-OnionTerminalDeliveryAck - Require terminal onion delivery before accepted ACK
//! v0.19.0-BlindRelayDescriptorHint - Allow signed next-hop descriptor hints for controlled two-hop proofs
//! v0.18.0-BlindRelayRouteabilityGate - Require fresh routeability evidence before next-hop forwarding
//! v0.17.0-BlindRelayOnwardEnvelope - Add optional two-hop middle-hop forwarding
//! v0.16.0-BlindRelayTimestampFreshness - Reject stale/future opaque route frames
//! v0.15.0-BlindRelayTimeoutTest - Cover unresponsive next-hop retry exhaustion
//! v0.14.0-BlindRelayMalformedAckTest - Cover malformed 2xx next-hop ACK as forward_failed
//! v0.13.0-BlindRelayAckValidation - Require accepted next-hop ACK before route success
//! v0.12.0-BlindRelayCapabilityGate - Require next hop to advertise ChatRelay before forwarding
//! v0.11.0-PeerHealthSummary - Report previous-hop abuse buckets to PeerStore
//! v0.10.0-BlindRelayAbuseGuard - Add previous-hop rate limit and quarantine
//! v0.9.0-BlindRelayReplayGuard - Drop duplicate route_id frames idempotently
//! v0.8.0-BlindRelayLoopGuard - Reject immediate self/previous-hop relay loops
//! v0.7.0-BlindRelayRetryStats - Report retry recovery/exhaustion to PeerStore status
//! v0.6.0-BlindRelayRetryJitter - Retry transient next-hop blind relay failures with privacy-safe jitter
//! v0.5.0-BlindRelayRouteHealth - Feed next-hop success/failure back into PeerStore scoring
//! v0.4.0-BlindRelayBackpressure - Added blind relay in-flight pressure gate
//! v0.3.0-BlindRelayEndpoint - Added node-to-node opaque blind relay endpoint
//! v0.2.0-PeerRelayHealth - Record inbound peer relay health counters
//! v0.1.0-DiscoveryPhase9 - Initial inter-node encrypted chat relay endpoint
// ============================================================================

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use aeronyx_core::crypto::transport::{
    DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD,
};
use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::{
    decode_envelope, encode_blind_relay_envelope, encode_envelope, BlindRelayEnvelope, ChatEnvelope,
};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::discovery::SignedNodeDescriptor;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{is_onion_blob, try_open_onion_layer};
use aeronyx_core::protocol::{DataPacket, NodeCapability};
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tracing::{debug, warn};

use crate::services::peer_store::PeerStore;
use crate::services::{ChatRelayService, Session, SessionManager};

// ============================================
// Constants
// ============================================

/// Maximum bincode-encoded envelope bytes accepted from another node.
///
/// This mirrors the protocol decode limit and protects the JSON endpoint from
/// carrying huge opaque payloads. Encrypted files should use blob storage, not
/// the peer envelope relay path.
const MAX_PEER_CHAT_ENVELOPE_BYTES: usize = 128 * 1024;

/// Maximum concurrent blind relay requests handled by this process.
///
/// Blind relay is intentionally opaque and can carry large encrypted blobs, so
/// it needs a hard in-flight cap before future multi-hop routing increases the
/// possible fanout. This is local backpressure only; callers should retry with
/// jitter at the transport/client layer.
const MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS: usize = 64;

/// Maximum attempts for a single next-hop blind relay POST.
///
/// The relay still treats `encrypted_blob` as opaque. Retry decisions are based
/// only on transport status buckets such as timeout/connect/5xx.
const MAX_BLIND_RELAY_FORWARD_ATTEMPTS: usize = 3;

/// Lower bound for transient next-hop retry jitter.
const BLIND_RELAY_RETRY_BASE_MS: u64 = 25;

/// Extra deterministic jitter window used to avoid retry herds.
const BLIND_RELAY_RETRY_JITTER_MS: u64 = 35;

/// Maximum route ids retained by one node for blind relay replay suppression.
///
/// The value is deliberately small and local-only: it prevents immediate
/// replay amplification without becoming a durable route history.
const MAX_BLIND_RELAY_SEEN_ROUTES: usize = 8192;

/// Replay cache horizon for blind relay route ids.
const BLIND_RELAY_ROUTE_REPLAY_WINDOW_SECS: u64 = 10 * 60;

/// Per previous-hop accepted relay attempts allowed in the short window.
const BLIND_RELAY_PREVIOUS_HOP_RATE_LIMIT: u32 = 120;

/// Sliding window for previous-hop relay rate limiting.
const BLIND_RELAY_PREVIOUS_HOP_RATE_WINDOW_SECS: u64 = 60;

/// Privacy-safe failure score that puts one previous-hop node into quarantine.
const BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD: u32 = 12;

/// Failure score decay horizon before a previous-hop gets a clean bucket.
const BLIND_RELAY_PREVIOUS_HOP_FAILURE_WINDOW_SECS: u64 = 5 * 60;

/// Short local quarantine for noisy previous-hop nodes.
const BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS: u64 = 5 * 60;

/// Maximum previous-hop abuse buckets retained by this process.
const MAX_BLIND_RELAY_PREVIOUS_HOP_BUCKETS: usize = 4096;

/// Maximum accepted age for an opaque blind-relay routing frame.
///
/// This is intentionally based only on `BlindRelayEnvelope.timestamp`, a signed
/// routing metadata field. It does not inspect or derive anything from the
/// encrypted blob, preserving the blind relay invariant while reducing replay
/// risk for commercial node operators.
const BLIND_RELAY_MAX_ENVELOPE_AGE_SECS: u64 = 10 * 60;

/// Small clock-skew allowance for peers whose clocks run slightly ahead.
const BLIND_RELAY_MAX_FUTURE_SKEW_SECS: u64 = 120;

// ============================================
// State / Request / Response Types
// ============================================

#[derive(Clone)]
struct ChatPeerState {
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_relay_in_flight: Arc<AtomicUsize>,
    blind_relay_seen_routes: Arc<Mutex<BlindRelayRouteReplayCache>>,
    blind_relay_abuse_guard: Arc<Mutex<BlindRelayAbuseGuard>>,
}

#[derive(Debug, Default)]
struct BlindRelayRouteReplayCache {
    seen: HashMap<[u8; 16], u64>,
    order: VecDeque<[u8; 16]>,
}

impl BlindRelayRouteReplayCache {
    fn observe(&mut self, route_id: [u8; 16], now: u64) -> bool {
        self.evict_expired(now);
        if self.seen.contains_key(&route_id) {
            return true;
        }

        self.seen.insert(route_id, now);
        self.order.push_back(route_id);
        self.evict_over_capacity();
        false
    }

    fn evict_expired(&mut self, now: u64) {
        while let Some(route_id) = self.order.front().copied() {
            let Some(first_seen_at) = self.seen.get(&route_id).copied() else {
                self.order.pop_front();
                continue;
            };
            if now.saturating_sub(first_seen_at) <= BLIND_RELAY_ROUTE_REPLAY_WINDOW_SECS {
                break;
            }
            self.order.pop_front();
            self.seen.remove(&route_id);
        }
    }

    fn evict_over_capacity(&mut self) {
        while self.seen.len() > MAX_BLIND_RELAY_SEEN_ROUTES {
            if let Some(route_id) = self.order.pop_front() {
                self.seen.remove(&route_id);
            } else {
                break;
            }
        }
    }

    fn forget(&mut self, route_id: &[u8; 16]) {
        self.seen.remove(route_id);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlindRelayAbuseDecision {
    Allowed,
    RateLimited { quarantine_until: u64 },
    Quarantined { quarantine_until: u64 },
}

#[derive(Debug, Default)]
struct BlindRelayPreviousHopBucket {
    rate_window_start: u64,
    rate_count: u32,
    failure_window_start: u64,
    failure_score: u32,
    quarantine_until: Option<u64>,
    last_seen_at: u64,
}

#[derive(Debug, Default)]
struct BlindRelayAbuseGuard {
    buckets: HashMap<[u8; 32], BlindRelayPreviousHopBucket>,
    order: VecDeque<[u8; 32]>,
}

impl BlindRelayAbuseGuard {
    fn observe_request(&mut self, previous_hop: [u8; 32], now: u64) -> BlindRelayAbuseDecision {
        self.evict_idle(now);
        let bucket = self.bucket_mut(previous_hop, now);
        bucket.last_seen_at = now;

        if bucket
            .quarantine_until
            .is_some_and(|quarantine_until| now < quarantine_until)
        {
            return BlindRelayAbuseDecision::Quarantined {
                quarantine_until: bucket.quarantine_until.unwrap_or(now),
            };
        }
        if now.saturating_sub(bucket.rate_window_start) > BLIND_RELAY_PREVIOUS_HOP_RATE_WINDOW_SECS
        {
            bucket.rate_window_start = now;
            bucket.rate_count = 0;
        }

        bucket.rate_count = bucket.rate_count.saturating_add(1);
        if bucket.rate_count > BLIND_RELAY_PREVIOUS_HOP_RATE_LIMIT {
            let quarantine_until = now + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS;
            bucket.quarantine_until = Some(quarantine_until);
            return BlindRelayAbuseDecision::RateLimited { quarantine_until };
        }

        BlindRelayAbuseDecision::Allowed
    }

    fn record_failure(&mut self, previous_hop: [u8; 32], now: u64) -> Option<u64> {
        let bucket = self.bucket_mut(previous_hop, now);
        bucket.last_seen_at = now;
        if now.saturating_sub(bucket.failure_window_start)
            > BLIND_RELAY_PREVIOUS_HOP_FAILURE_WINDOW_SECS
        {
            bucket.failure_window_start = now;
            bucket.failure_score = 0;
        }

        bucket.failure_score = bucket.failure_score.saturating_add(1);
        if bucket.failure_score >= BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD {
            let quarantine_until = now + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS;
            bucket.quarantine_until = Some(quarantine_until);
            bucket.failure_score = 0;
            return Some(quarantine_until);
        }
        None
    }

    fn record_success(&mut self, previous_hop: [u8; 32], now: u64) {
        if let Some(bucket) = self.buckets.get_mut(&previous_hop) {
            bucket.last_seen_at = now;
            if now.saturating_sub(bucket.failure_window_start)
                > BLIND_RELAY_PREVIOUS_HOP_FAILURE_WINDOW_SECS
            {
                bucket.failure_window_start = now;
                bucket.failure_score = 0;
            }
        }
    }

    fn bucket_mut(&mut self, previous_hop: [u8; 32], now: u64) -> &mut BlindRelayPreviousHopBucket {
        if !self.buckets.contains_key(&previous_hop) {
            self.order.push_back(previous_hop);
        }
        self.evict_over_capacity();
        self.buckets
            .entry(previous_hop)
            .or_insert_with(|| BlindRelayPreviousHopBucket {
                rate_window_start: now,
                failure_window_start: now,
                last_seen_at: now,
                ..BlindRelayPreviousHopBucket::default()
            })
    }

    fn evict_idle(&mut self, now: u64) {
        let retention_secs =
            BLIND_RELAY_PREVIOUS_HOP_FAILURE_WINDOW_SECS + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS;
        while let Some(previous_hop) = self.order.front().copied() {
            let Some(bucket) = self.buckets.get(&previous_hop) else {
                self.order.pop_front();
                continue;
            };
            let quarantine_active = bucket
                .quarantine_until
                .is_some_and(|quarantine_until| now < quarantine_until);
            if quarantine_active || now.saturating_sub(bucket.last_seen_at) <= retention_secs {
                break;
            }
            self.order.pop_front();
            self.buckets.remove(&previous_hop);
        }
    }

    fn evict_over_capacity(&mut self) {
        while self.buckets.len() >= MAX_BLIND_RELAY_PREVIOUS_HOP_BUCKETS {
            if let Some(previous_hop) = self.order.pop_front() {
                self.buckets.remove(&previous_hop);
            } else {
                break;
            }
        }
    }
}

/// Node-to-node encrypted envelope relay request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerChatRelayRequest {
    /// End-to-end encrypted, sender-signed chat envelope.
    pub envelope: ChatEnvelope,
}

/// Node-to-node encrypted envelope relay response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerChatRelayResponse {
    /// Whether this node accepted the envelope as valid relay work.
    pub accepted: bool,
    /// Whether the message id was already seen on this node.
    pub duplicate: bool,
    /// Number of local online receiver sessions reached.
    pub delivered_online: usize,
    /// Whether the envelope was stored in the local pending queue.
    pub stored_pending: bool,
}

/// Node-to-node blind relay request.
///
/// `previous_hop_node_id` is transport/auth context for signature
/// verification. It is intentionally outside `BlindRelayEnvelope` so the
/// envelope itself remains the minimal route metadata set documented in
/// aeronyx-core. Do not add user ids, receiver wallet ids, domains, URLs, DNS
/// contents, or payload-derived information here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerBlindRelayRequest {
    /// Opaque encrypted relay envelope. `encrypted_blob` must not be parsed.
    pub envelope: BlindRelayEnvelope,
    /// Ed25519 node id that signed this hop.
    pub previous_hop_node_id: [u8; 32],
    /// Optional already-opaque next routing frame for a no-exit middle hop.
    ///
    /// This field is absent for existing single-hop requests. When present,
    /// only the node named by `envelope.next_hop` may use it, and it must still
    /// verify/re-sign the onward frame as normal blind relay work. Do not place
    /// user identifiers, plaintext, DNS data, domains, URLs, packet payloads,
    /// or full route/social-graph metadata here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onward_envelope: Option<BlindRelayEnvelope>,
    /// Optional signed descriptor for the onward `next_hop`.
    ///
    /// This is used by controlled two-hop path proofs when the middle node has
    /// not yet warmed its local routeability cache for the terminal node. The
    /// descriptor is still node-control-plane metadata: it must verify, match
    /// the onward `next_hop`, and advertise `ChatRelay`. It must never carry
    /// user ids, receiver identities, plaintext, DNS data, domains, URLs,
    /// packet payloads, route ids, or social-graph metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onward_descriptor_hint: Option<SignedNodeDescriptor>,
}

/// Node-to-node blind relay response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerBlindRelayResponse {
    /// Whether this node accepted the request as valid relay work.
    pub accepted: bool,
    /// Whether this node is the requested next hop.
    pub terminal: bool,
    /// Whether this node forwarded the opaque envelope to another node.
    pub forwarded: bool,
    /// Remaining TTL observed or forwarded by this node.
    pub ttl_remaining: u8,
    /// Privacy-safe coarse result bucket for nodeboard/audits.
    pub reason: Option<String>,
}

#[derive(Debug, thiserror::Error)]
enum ChatPeerRelayError {
    #[error("chat relay disabled")]
    RelayUnavailable,

    #[error("invalid envelope signature")]
    InvalidSignature,

    #[error("envelope too large: {size} bytes")]
    EnvelopeTooLarge { size: usize },

    #[error("envelope serialization failed")]
    Serialization,

    #[error("pending store failed")]
    StoreFailed,
}

#[derive(Debug, thiserror::Error)]
enum BlindRelayError {
    #[error("invalid previous hop public key")]
    InvalidPreviousHop,

    #[error("invalid blind envelope signature")]
    InvalidSignature,

    #[error("blind envelope too large")]
    EnvelopeTooLarge,

    #[error("ttl exhausted")]
    TtlExhausted,

    #[error("blind envelope timestamp expired")]
    TimestampExpired,

    #[error("blind envelope timestamp is too far in the future")]
    TimestampInFuture,

    #[error("previous hop rate limited")]
    RateLimited,

    #[error("previous hop quarantined")]
    Quarantined,

    #[error("blind relay route loop detected")]
    RouteLoop,

    #[error("next hop not found")]
    NoRoute,

    #[error("next hop endpoint missing or invalid")]
    InvalidEndpoint,

    #[error("blind relay forward failed")]
    ForwardFailed,

    #[error("onion layer peel failed")]
    OnionPeelFailed,
}

impl BlindRelayError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidPreviousHop
            | Self::InvalidSignature
            | Self::EnvelopeTooLarge
            | Self::TtlExhausted
            | Self::TimestampExpired
            | Self::TimestampInFuture
            | Self::RouteLoop
            | Self::OnionPeelFailed => StatusCode::BAD_REQUEST,
            Self::RateLimited | Self::Quarantined => StatusCode::TOO_MANY_REQUESTS,
            Self::NoRoute | Self::InvalidEndpoint => StatusCode::BAD_GATEWAY,
            Self::ForwardFailed => StatusCode::BAD_GATEWAY,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::InvalidPreviousHop => "invalid_previous_hop",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge => "envelope_too_large",
            Self::TtlExhausted => "ttl_exhausted",
            Self::TimestampExpired => "timestamp_expired",
            Self::TimestampInFuture => "timestamp_in_future",
            Self::RateLimited => "rate_limited",
            Self::Quarantined => "quarantined",
            Self::RouteLoop => "route_loop",
            Self::NoRoute => "no_route",
            Self::InvalidEndpoint => "invalid_endpoint",
            Self::ForwardFailed => "forward_failed",
            Self::OnionPeelFailed => "onion_peel_failed",
        }
    }
}

impl ChatPeerRelayError {
    fn status_code(&self) -> StatusCode {
        match self {
            Self::RelayUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            Self::InvalidSignature | Self::EnvelopeTooLarge { .. } | Self::Serialization => {
                StatusCode::BAD_REQUEST
            }
            Self::StoreFailed => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::RelayUnavailable => "relay_unavailable",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge { .. } => "envelope_too_large",
            Self::Serialization => "envelope_serialization_failed",
            Self::StoreFailed => "store_pending_failed",
        }
    }
}

// ============================================
// Router
// ============================================

/// Builds node-to-node encrypted chat relay routes.
pub fn build_chat_peer_router(
    chat_relay: Option<Arc<ChatRelayService>>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
) -> Router {
    let state = ChatPeerState {
        chat_relay,
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
        blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
    };

    Router::new()
        .route("/api/chat/peer/relay", post(peer_relay_handler))
        .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
        .with_state(state)
}

// ============================================
// Handlers
// ============================================

async fn peer_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerChatRelayRequest>,
) -> impl IntoResponse {
    match process_peer_relay(state, request.envelope).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            error.status_code(),
            Json(PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            }),
        )
            .into_response(),
    }
}

async fn peer_blind_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerBlindRelayRequest>,
) -> impl IntoResponse {
    let Some(_in_flight) = BlindRelayInFlightGuard::try_acquire(&state) else {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "backpressure");
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(PeerBlindRelayResponse {
                accepted: false,
                terminal: false,
                forwarded: false,
                ttl_remaining: 0,
                reason: Some("backpressure".to_string()),
            }),
        )
            .into_response();
    };

    match process_peer_blind_relay(state, request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            error.status_code(),
            Json(PeerBlindRelayResponse {
                accepted: false,
                terminal: false,
                forwarded: false,
                ttl_remaining: 0,
                reason: Some(error.reason_bucket().to_string()),
            }),
        )
            .into_response(),
    }
}

struct BlindRelayInFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl BlindRelayInFlightGuard {
    fn try_acquire(state: &ChatPeerState) -> Option<Self> {
        let counter = Arc::clone(&state.blind_relay_in_flight);
        let mut current = counter.load(Ordering::Relaxed);
        loop {
            if current >= MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(Self { counter }),
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for BlindRelayInFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn process_peer_relay(
    state: ChatPeerState,
    envelope: ChatEnvelope,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let now = now_secs();
    if let Err(error) = validate_peer_envelope(&envelope) {
        if let Some(relay) = state.chat_relay.as_ref() {
            relay.record_peer_relay_inbound_rejected(now, error.reason_bucket());
        }
        return Err(error);
    }

    let Some(relay) = state.chat_relay else {
        return Err(ChatPeerRelayError::RelayUnavailable);
    };

    if relay.is_online_duplicate(&envelope.message_id) {
        debug!("[CHAT_PEER] Duplicate peer envelope ignored");
        relay.record_peer_relay_inbound_accepted(now, true, 0, false);
        return Ok(PeerChatRelayResponse {
            accepted: true,
            duplicate: true,
            delivered_online: 0,
            stored_pending: false,
        });
    }

    let target_sessions = state.sessions.get_all_by_wallet(&envelope.receiver);
    let mut delivered_online = 0usize;

    for session in target_sessions {
        if send_envelope_to_session(&envelope, &session, &state.udp).await {
            delivered_online += 1;
        }
    }

    let stored_pending = if delivered_online == 0 {
        relay.store_pending(&envelope).map_err(|_error| {
            warn!(
                reason = "store_pending_failed",
                "[CHAT_PEER] Failed to store peer envelope for offline receiver"
            );
            relay.record_peer_relay_inbound_rejected(now, "store_pending_failed");
            ChatPeerRelayError::StoreFailed
        })?;
        true
    } else {
        false
    };

    relay.record_peer_relay_inbound_accepted(now, false, delivered_online, stored_pending);

    Ok(PeerChatRelayResponse {
        accepted: true,
        duplicate: false,
        delivered_online,
        stored_pending,
    })
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn process_peer_blind_relay(
    state: ChatPeerState,
    request: PeerBlindRelayRequest,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let now = now_secs();
    let previous_hop_node_id = request.previous_hop_node_id;
    let onward_descriptor_hint = request.onward_descriptor_hint;
    let envelope = request.envelope;

    check_blind_relay_previous_hop_allowed(&state, previous_hop_node_id, now)?;

    validate_blind_relay_envelope(&envelope, &previous_hop_node_id, now).map_err(|error| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, error.reason_bucket());
        error
    })?;

    let self_node_id = state.node_identity.public_key_bytes();
    if previous_hop_node_id == self_node_id {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "self_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if envelope.next_hop == self_node_id {
        // Onion routing v1: if the opaque blob is an onion layer addressed to
        // this node, peel exactly one layer and either deliver locally
        // (terminal) or forward the inner layer to the revealed next hop. Legacy
        // opaque blobs (no onion magic) fall through to the existing behavior.
        if is_onion_blob(&envelope.encrypted_blob) {
            return process_onion_blind_relay(state, previous_hop_node_id, envelope, now).await;
        }
        if let Some(onward_envelope) = request.onward_envelope {
            return process_onion_middle_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                onward_envelope,
                onward_descriptor_hint,
                now,
            )
            .await;
        }
        if observe_blind_relay_route(&state, envelope.route_id, now) {
            return Ok(duplicate_blind_relay_response(
                &state,
                previous_hop_node_id,
                now,
                envelope.ttl,
            ));
        }
        record_blind_relay_previous_hop_success(&state, previous_hop_node_id, now);
        state.peer_store.record_blind_relay_terminal(
            now,
            envelope.ttl,
            envelope.encrypted_blob.len(),
        );
        return Ok(PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: envelope.ttl,
            reason: Some("terminal_next_hop".to_string()),
        });
    }

    if envelope.next_hop == previous_hop_node_id {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if !envelope.can_forward() {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = envelope.next_hop;
    let (descriptor, used_descriptor_hint) = resolve_blind_relay_next_hop_descriptor(
        &state,
        &next_hop,
        now,
        onward_descriptor_hint.as_ref(),
    )
    .ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        BlindRelayError::NoRoute
    })?;
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        state.peer_store.record_route_forward_failure(
            &next_hop,
            now,
            "missing_chat_relay_capability",
        );
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        state
            .peer_store
            .record_route_forward_failure(&next_hop, now, "routeability_not_ready");
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            state
                .peer_store
                .record_route_forward_failure(&next_hop, now, "missing_endpoint");
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        state
            .peer_store
            .record_route_forward_failure(&next_hop, now, "invalid_endpoint");
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    if observe_blind_relay_route(&state, envelope.route_id, now) {
        return Ok(duplicate_blind_relay_response(
            &state,
            previous_hop_node_id,
            now,
            envelope.ttl,
        ));
    }

    let forwarded_envelope = envelope
        .decremented_ttl()
        .ok_or(BlindRelayError::TtlExhausted)?
        .sign_with(state.node_identity.as_ref());
    let forwarded_onward_envelope = request
        .onward_envelope
        .map(|envelope| envelope.sign_with(state.node_identity.as_ref()));
    let forwarded_onward_descriptor_hint = onward_descriptor_hint;
    let ttl_remaining = forwarded_envelope.ttl;

    if let Err(error) = forward_blind_relay_with_retry(
        &state,
        &url,
        next_hop,
        PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: self_node_id,
            onward_envelope: forwarded_onward_envelope,
            onward_descriptor_hint: forwarded_onward_descriptor_hint,
        },
        now,
    )
    .await
    {
        forget_blind_relay_route(&state, &envelope.route_id);
        return Err(error);
    }

    state
        .peer_store
        .record_route_forward_success(&next_hop, now);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id, now);
    state
        .peer_store
        .record_blind_relay_forwarded(now, ttl_remaining);

    Ok(PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining,
        reason: Some("forwarded".to_string()),
    })
}

/// Onion routing v1 — this node is the addressed hop and the opaque blob is an
/// onion layer. Peel exactly one layer with the node's rotating onion key(s),
/// then either deliver locally (terminal hop) or forward the revealed inner
/// layer to the next hop (entry/middle hop).
///
/// Privacy invariant: a relay learns only the previous hop (transport auth) and
/// the immediate next hop (from its own peeled layer). It never sees the
/// original source, the final destination, or the payload. The onion secret
/// keys (current + previous within the rotation grace window, see
/// services::onion_keys) are never logged.
async fn process_onion_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    envelope: BlindRelayEnvelope,
    now: u64,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let self_node_id = state.node_identity.public_key_bytes();

    // Per-route replay/dedup, identical to the opaque terminal/forward paths.
    if observe_blind_relay_route(&state, envelope.route_id, now) {
        return Ok(duplicate_blind_relay_response(
            &state,
            previous_hop_node_id,
            now,
            envelope.ttl,
        ));
    }

    // Peel exactly one onion layer with the node's rotating onion key(s): the
    // current key, plus the previous key while it is within the rotation grace
    // window (forward secrecy — see services::onion_keys). A failure yields a
    // coarse bucket only, never a payload leak.
    let onion_secrets = crate::services::onion_keys::peel_secrets(now);
    let peel = match try_open_onion_layer(&envelope.encrypted_blob, &onion_secrets) {
        Ok(peel) => peel,
        Err(_) => {
            forget_blind_relay_route(&state, &envelope.route_id);
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "onion_peel_failed");
            return Err(BlindRelayError::OnionPeelFailed);
        }
    };

    match peel.next_hop {
        // Terminal hop: `inner` is the delivered payload (a ChatEnvelope).
        None => {
            let inner_envelope = decode_envelope(&peel.inner).map_err(|_| {
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "onion_terminal_decode_failed",
                );
                BlindRelayError::OnionPeelFailed
            })?;

            // Deliver via the existing zero-knowledge chat relay store-and-forward.
            // This is a hard ACK boundary for Milestone 2: a terminal onion hop
            // may only return accepted=true after the existing relay path has
            // either delivered online or stored the encrypted envelope pending.
            // The coarse rejection bucket avoids leaking payload or receiver data.
            if let Err(error) = process_peer_relay(state.clone(), inner_envelope).await {
                forget_blind_relay_route(&state, &envelope.route_id);
                debug!(
                    reason = error.reason_bucket(),
                    "[BLIND_RELAY] Onion terminal delivery failed"
                );
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "onion_terminal_relay_failed",
                );
                return Err(BlindRelayError::ForwardFailed);
            }

            record_blind_relay_previous_hop_success(&state, previous_hop_node_id, now);
            state.peer_store.record_blind_relay_terminal(
                now,
                envelope.ttl,
                envelope.encrypted_blob.len(),
            );
            Ok(PeerBlindRelayResponse {
                accepted: true,
                terminal: true,
                forwarded: false,
                ttl_remaining: envelope.ttl,
                reason: Some("onion_terminal_delivered".to_string()),
            })
        }
        // Entry/middle hop: forward the inner layer to the revealed next hop.
        Some(next_hop) => {
            if next_hop == self_node_id || next_hop == previous_hop_node_id {
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
                return Err(BlindRelayError::RouteLoop);
            }
            if !is_onion_blob(&peel.inner) {
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "onion_inner_not_layer",
                );
                return Err(BlindRelayError::OnionPeelFailed);
            }
            if !envelope.can_forward() {
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
                return Err(BlindRelayError::TtlExhausted);
            }

            let descriptor = state.peer_store.get_valid(&next_hop, now).ok_or_else(|| {
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                BlindRelayError::NoRoute
            })?;
            if !descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            {
                state.peer_store.record_route_forward_failure(
                    &next_hop,
                    now,
                    "missing_chat_relay_capability",
                );
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }
            // True onion routes are the recovery/proof path for a small mesh:
            // after restart a fresh signed peer may not yet have routeability
            // evidence, and refusing the proof attempt creates a deadlock. Keep
            // the hard stop for peers under local route quarantine; forward
            // errors below will still feed route health without logging payloads.
            if state.peer_store.is_route_quarantined_now(&next_hop, now) {
                state
                    .peer_store
                    .record_route_forward_failure(&next_hop, now, "route_quarantined");
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }

            let endpoint = descriptor
                .descriptor
                .public_endpoint
                .as_deref()
                .ok_or_else(|| {
                    state.peer_store.record_route_forward_failure(
                        &next_hop,
                        now,
                        "missing_endpoint",
                    );
                    forget_blind_relay_route(&state, &envelope.route_id);
                    reject_blind_relay_previous_hop(
                        &state,
                        previous_hop_node_id,
                        now,
                        "missing_endpoint",
                    );
                    BlindRelayError::InvalidEndpoint
                })?;
            let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
                state
                    .peer_store
                    .record_route_forward_failure(&next_hop, now, "invalid_endpoint");
                forget_blind_relay_route(&state, &envelope.route_id);
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "invalid_endpoint",
                );
                BlindRelayError::InvalidEndpoint
            })?;

            // Fresh envelope carrying the peeled inner layer onward. Re-signed by
            // this node; next_hop is addressed to the revealed relay.
            let forwarded_envelope = BlindRelayEnvelope {
                route_id: envelope.route_id,
                next_hop,
                ttl: envelope.ttl.saturating_sub(1),
                encrypted_blob: peel.inner,
                timestamp: now,
                signature: [0u8; 64],
            }
            .sign_with(state.node_identity.as_ref());
            let ttl_remaining = forwarded_envelope.ttl;

            if let Err(error) = forward_blind_relay_with_retry(
                &state,
                &url,
                next_hop,
                PeerBlindRelayRequest {
                    envelope: forwarded_envelope,
                    previous_hop_node_id: self_node_id,
                    onward_envelope: None,
                    onward_descriptor_hint: None,
                },
                now,
            )
            .await
            {
                forget_blind_relay_route(&state, &envelope.route_id);
                return Err(error);
            }

            state
                .peer_store
                .record_route_forward_success(&next_hop, now);
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id, now);
            state
                .peer_store
                .record_blind_relay_forwarded(now, ttl_remaining);

            Ok(PeerBlindRelayResponse {
                accepted: true,
                terminal: false,
                forwarded: true,
                ttl_remaining,
                reason: Some("onion_forwarded".to_string()),
            })
        }
    }
}

async fn process_onion_middle_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    outer_envelope: BlindRelayEnvelope,
    onward_envelope: BlindRelayEnvelope,
    onward_descriptor_hint: Option<SignedNodeDescriptor>,
    now: u64,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    validate_blind_relay_envelope(&onward_envelope, &previous_hop_node_id, now).map_err(
        |error| {
            reject_blind_relay_previous_hop(
                &state,
                previous_hop_node_id,
                now,
                error.reason_bucket(),
            );
            error
        },
    )?;

    let self_node_id = state.node_identity.public_key_bytes();
    if onward_envelope.next_hop == self_node_id || onward_envelope.next_hop == previous_hop_node_id
    {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if !onward_envelope.can_forward() {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = onward_envelope.next_hop;
    let (descriptor, used_descriptor_hint) = resolve_blind_relay_next_hop_descriptor(
        &state,
        &next_hop,
        now,
        onward_descriptor_hint.as_ref(),
    )
    .ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        BlindRelayError::NoRoute
    })?;
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        state.peer_store.record_route_forward_failure(
            &next_hop,
            now,
            "missing_chat_relay_capability",
        );
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        state
            .peer_store
            .record_route_forward_failure(&next_hop, now, "routeability_not_ready");
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            state
                .peer_store
                .record_route_forward_failure(&next_hop, now, "missing_endpoint");
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        state
            .peer_store
            .record_route_forward_failure(&next_hop, now, "invalid_endpoint");
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    if observe_blind_relay_route(&state, outer_envelope.route_id, now) {
        return Ok(duplicate_blind_relay_response(
            &state,
            previous_hop_node_id,
            now,
            outer_envelope.ttl,
        ));
    }

    let forwarded_envelope = onward_envelope
        .decremented_ttl()
        .ok_or(BlindRelayError::TtlExhausted)?
        .sign_with(state.node_identity.as_ref());
    let ttl_remaining = forwarded_envelope.ttl;

    if let Err(error) = forward_blind_relay_with_retry(
        &state,
        &url,
        next_hop,
        PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: self_node_id,
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
        now,
    )
    .await
    {
        forget_blind_relay_route(&state, &outer_envelope.route_id);
        return Err(error);
    }

    state
        .peer_store
        .record_route_forward_success(&next_hop, now);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id, now);
    state
        .peer_store
        .record_blind_relay_forwarded(now, ttl_remaining);

    Ok(PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining,
        reason: Some("onion_middle_forwarded".to_string()),
    })
}

fn resolve_blind_relay_next_hop_descriptor(
    state: &ChatPeerState,
    next_hop: &[u8; 32],
    now: u64,
    descriptor_hint: Option<&SignedNodeDescriptor>,
) -> Option<(SignedNodeDescriptor, bool)> {
    if let Some(descriptor) = state.peer_store.get_valid(next_hop, now) {
        return Some((descriptor, false));
    }

    let descriptor = descriptor_hint?;
    if descriptor.node_id() != *next_hop {
        return None;
    }
    if descriptor.verify_at(now).is_err() {
        return None;
    }
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        return None;
    }
    Some((descriptor.clone(), true))
}

fn observe_blind_relay_route(state: &ChatPeerState, route_id: [u8; 16], now: u64) -> bool {
    let mut seen_routes = state
        .blind_relay_seen_routes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    seen_routes.observe(route_id, now)
}

fn forget_blind_relay_route(state: &ChatPeerState, route_id: &[u8; 16]) {
    let mut seen_routes = state
        .blind_relay_seen_routes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    seen_routes.forget(route_id);
}

fn check_blind_relay_previous_hop_allowed(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
) -> Result<(), BlindRelayError> {
    let decision = {
        let mut abuse_guard = state
            .blind_relay_abuse_guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        abuse_guard.observe_request(previous_hop, now)
    };

    match decision {
        BlindRelayAbuseDecision::Allowed => Ok(()),
        BlindRelayAbuseDecision::RateLimited { quarantine_until } => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "rate_limited");
            state
                .peer_store
                .record_blind_relay_quarantine_started(now, "rate_limit");
            state
                .peer_store
                .record_peer_relay_rejection(&previous_hop, now, "rate_limited");
            state.peer_store.record_peer_relay_quarantine_started(
                &previous_hop,
                now,
                quarantine_until,
                "rate_limit",
            );
            Err(BlindRelayError::RateLimited)
        }
        BlindRelayAbuseDecision::Quarantined { quarantine_until } => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "quarantined");
            state
                .peer_store
                .record_peer_relay_rejection(&previous_hop, now, "quarantined");
            state.peer_store.record_peer_relay_quarantine_started(
                &previous_hop,
                now,
                quarantine_until,
                "still_quarantined",
            );
            Err(BlindRelayError::Quarantined)
        }
    }
}

fn reject_blind_relay_previous_hop(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
    reason: &'static str,
) {
    state.peer_store.record_blind_relay_rejected(now, reason);
    state
        .peer_store
        .record_peer_relay_rejection(&previous_hop, now, reason);
    if !blind_relay_reason_counts_toward_quarantine(reason) {
        return;
    }

    let quarantine_until = {
        let mut abuse_guard = state
            .blind_relay_abuse_guard
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        abuse_guard.record_failure(previous_hop, now)
    };
    if let Some(quarantine_until) = quarantine_until {
        state
            .peer_store
            .record_blind_relay_quarantine_started(now, "failure_threshold");
        state.peer_store.record_peer_relay_quarantine_started(
            &previous_hop,
            now,
            quarantine_until,
            "failure_threshold",
        );
    }
}

fn record_blind_relay_previous_hop_success(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
) {
    let mut abuse_guard = state
        .blind_relay_abuse_guard
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    abuse_guard.record_success(previous_hop, now);
}

fn blind_relay_reason_counts_toward_quarantine(reason: &str) -> bool {
    matches!(
        reason,
        "invalid_previous_hop" | "invalid_signature" | "self_loop" | "route_loop" | "ttl_exhausted"
    )
}

fn duplicate_blind_relay_response(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
    ttl_remaining: u8,
) -> PeerBlindRelayResponse {
    // Duplicate route IDs are idempotent replay drops. They are still counted
    // in the aggregate blind relay replay bucket, but they must not poison the
    // previous-hop health score: a healthy peer may retry after losing our ACK.
    state
        .peer_store
        .record_blind_relay_rejected(now, "duplicate_route");
    record_blind_relay_previous_hop_success(state, previous_hop, now);
    PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: false,
        ttl_remaining,
        reason: Some("duplicate_route".to_string()),
    }
}

async fn forward_blind_relay_with_retry(
    state: &ChatPeerState,
    url: &str,
    next_hop: [u8; 32],
    request: PeerBlindRelayRequest,
    now: u64,
) -> Result<(), BlindRelayError> {
    for attempt in 1..=MAX_BLIND_RELAY_FORWARD_ATTEMPTS {
        match state.http_client.post(url).json(&request).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<PeerBlindRelayResponse>().await {
                    Ok(ack) if ack.accepted => {
                        if attempt > 1 {
                            state
                                .peer_store
                                .record_blind_relay_retry_succeeded(now, attempt);
                        }
                        return Ok(());
                    }
                    Ok(_ack) => {
                        debug!(
                            attempt,
                            "[BLIND_RELAY] Next-hop ACK rejected opaque relay envelope"
                        );
                    }
                    Err(_error) => {
                        debug!(
                            attempt,
                            reason = "ack_decode_failed",
                            "[BLIND_RELAY] Next-hop ACK decode failed"
                        );
                    }
                }

                if attempt > 1 {
                    state.peer_store.record_blind_relay_retry_exhausted(
                        now,
                        attempt,
                        "forward_failed",
                    );
                }
                state
                    .peer_store
                    .record_route_forward_failure(&next_hop, now, "forward_failed");
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "forward_failed");
                return Err(BlindRelayError::ForwardFailed);
            }
            Ok(response) => {
                let status = response.status();
                let reason = format!("http_{}", status.as_u16());
                if attempt < MAX_BLIND_RELAY_FORWARD_ATTEMPTS
                    && is_retryable_blind_relay_status(status)
                {
                    state
                        .peer_store
                        .record_blind_relay_retry_attempt(now, &reason);
                    debug!(
                        attempt,
                        status = %status,
                        "[BLIND_RELAY] Next-hop returned retryable status"
                    );
                    sleep(blind_relay_retry_delay(
                        &request.envelope.route_id,
                        &next_hop,
                        attempt,
                    ))
                    .await;
                    continue;
                }

                debug!(
                    attempt,
                    status = %status,
                    "[BLIND_RELAY] Next-hop returned non-success"
                );
                if attempt > 1 {
                    state
                        .peer_store
                        .record_blind_relay_retry_exhausted(now, attempt, &reason);
                }
                state
                    .peer_store
                    .record_route_forward_failure(&next_hop, now, reason.clone());
                state.peer_store.record_blind_relay_rejected(now, reason);
                return Err(BlindRelayError::ForwardFailed);
            }
            Err(error) => {
                let reason = classify_reqwest_error("blind_relay_request", &error);
                if attempt < MAX_BLIND_RELAY_FORWARD_ATTEMPTS && is_retryable_reqwest_error(&error)
                {
                    state
                        .peer_store
                        .record_blind_relay_retry_attempt(now, &reason);
                    debug!(
                        attempt,
                        reason = %reason,
                        "[BLIND_RELAY] Next-hop forward failed; retrying"
                    );
                    sleep(blind_relay_retry_delay(
                        &request.envelope.route_id,
                        &next_hop,
                        attempt,
                    ))
                    .await;
                    continue;
                }

                debug!(
                    attempt,
                    reason = %reason,
                    "[BLIND_RELAY] Next-hop forward failed"
                );
                if attempt > 1 {
                    state
                        .peer_store
                        .record_blind_relay_retry_exhausted(now, attempt, &reason);
                }
                state
                    .peer_store
                    .record_route_forward_failure(&next_hop, now, reason.clone());
                state.peer_store.record_blind_relay_rejected(now, reason);
                return Err(BlindRelayError::ForwardFailed);
            }
        }
    }

    Err(BlindRelayError::ForwardFailed)
}

fn is_retryable_blind_relay_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_retryable_reqwest_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request()
}

fn blind_relay_retry_delay(route_id: &[u8; 16], next_hop: &[u8; 32], attempt: usize) -> Duration {
    let mut seed = attempt as u64;
    for byte in route_id.iter().chain(next_hop.iter()) {
        seed = seed.wrapping_mul(31).wrapping_add(u64::from(*byte));
    }
    let jitter = seed % (BLIND_RELAY_RETRY_JITTER_MS + 1);
    Duration::from_millis(BLIND_RELAY_RETRY_BASE_MS + jitter)
}

fn validate_blind_relay_envelope(
    envelope: &BlindRelayEnvelope,
    previous_hop_node_id: &[u8; 32],
    now: u64,
) -> Result<(), BlindRelayError> {
    let previous_hop = IdentityPublicKey::from_bytes(previous_hop_node_id)
        .map_err(|_| BlindRelayError::InvalidPreviousHop)?;
    envelope
        .verify_signature_from(&previous_hop)
        .map_err(|_| BlindRelayError::InvalidSignature)?;
    validate_blind_relay_timestamp(envelope.timestamp, now)?;
    encode_blind_relay_envelope(envelope).map_err(|_| BlindRelayError::EnvelopeTooLarge)?;
    Ok(())
}

fn validate_blind_relay_timestamp(timestamp: u64, now: u64) -> Result<(), BlindRelayError> {
    if timestamp > now.saturating_add(BLIND_RELAY_MAX_FUTURE_SKEW_SECS) {
        return Err(BlindRelayError::TimestampInFuture);
    }
    if now.saturating_sub(timestamp) > BLIND_RELAY_MAX_ENVELOPE_AGE_SECS {
        return Err(BlindRelayError::TimestampExpired);
    }
    Ok(())
}

fn blind_peer_relay_url(endpoint: &str) -> Option<String> {
    let endpoint = endpoint.trim().trim_end_matches('/');
    if endpoint.is_empty() {
        return None;
    }
    let base = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("http://{endpoint}")
    };
    Some(format!("{base}/api/chat/peer/blind-relay"))
}

fn classify_reqwest_error(phase: &str, error: &reqwest::Error) -> String {
    if error.is_timeout() {
        return format!("{phase}_timeout");
    }
    if error.is_connect() {
        return format!("{phase}_connect");
    }
    if error.is_status() {
        if let Some(status) = error.status() {
            return format!("{phase}_http_{}", status.as_u16());
        }
        return format!("{phase}_http_status");
    }
    if error.is_decode() {
        return format!("{phase}_decode");
    }
    if error.is_body() {
        return format!("{phase}_body");
    }
    if error.is_request() {
        return format!("{phase}_request");
    }
    format!("{phase}_unknown")
}

fn validate_peer_envelope(envelope: &ChatEnvelope) -> Result<(), ChatPeerRelayError> {
    envelope
        .verify_signature()
        .map_err(|_| ChatPeerRelayError::InvalidSignature)?;

    let encoded = encode_envelope(envelope).map_err(|_| ChatPeerRelayError::Serialization)?;
    if encoded.len() > MAX_PEER_CHAT_ENVELOPE_BYTES {
        return Err(ChatPeerRelayError::EnvelopeTooLarge {
            size: encoded.len(),
        });
    }

    Ok(())
}

async fn send_envelope_to_session(
    envelope: &ChatEnvelope,
    session: &Arc<Session>,
    udp: &Arc<UdpTransport>,
) -> bool {
    let msg = MemChainMessage::ChatRelay(envelope.clone());
    let plaintext = match encode_memchain(&msg) {
        Ok(plaintext) => plaintext,
        Err(_error) => {
            warn!(
                reason = "encode_client_relay_message_failed",
                "[CHAT_PEER] Failed to encode client relay message"
            );
            return false;
        }
    };

    let crypto = DefaultTransportCrypto::new();
    let counter = session.next_tx_counter();
    let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
    let len = match crypto.encrypt(
        &session.session_key,
        counter,
        session.id.as_bytes(),
        &plaintext,
        &mut encrypted,
    ) {
        Ok(len) => len,
        Err(_error) => {
            warn!(
                reason = "encrypt_client_relay_message_failed",
                "[CHAT_PEER] Failed to encrypt client relay message"
            );
            return false;
        }
    };
    encrypted.truncate(len);

    let packet = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
    let bytes = encode_data_packet(&packet).to_vec();
    udp.send(&bytes, &session.client_endpoint).await.is_ok()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::chat::ChatContentType;
    use aeronyx_core::protocol::{
        NodeCapability, NodeCapacity, NodeDescriptor, SignedNodeDescriptor,
    };
    use aeronyx_transport::UdpTransport;
    use axum::body::Body;
    use axum::http::Request;
    use axum::response::IntoResponse;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    use crate::config::ChatRelayConfig;

    fn signed_envelope() -> ChatEnvelope {
        let kp = IdentityKeyPair::generate();
        let mut envelope = ChatEnvelope {
            message_id: [0x11u8; 16],
            sender: kp.public_key_bytes(),
            receiver: [0x22u8; 32],
            timestamp: 1_800_000_000,
            ciphertext: b"opaque encrypted payload".to_vec(),
            nonce: [0x33u8; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        envelope.signature = kp.sign(&envelope.sign_data());
        envelope
    }

    fn test_chat_config(path: String) -> ChatRelayConfig {
        ChatRelayConfig {
            enabled: true,
            db_path: path,
            ..ChatRelayConfig::default()
        }
    }

    fn temp_chat_relay(label: &str) -> (Arc<ChatRelayService>, std::path::PathBuf) {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("aeronyx-{label}-{unique}.db"));
        let relay = Arc::new(
            ChatRelayService::new(
                test_chat_config(path.to_string_lossy().to_string()),
                [7u8; 32],
            )
            .unwrap(),
        );
        (relay, path)
    }

    fn signed_chat_relay_peer_descriptor_for(
        peer_identity: &IdentityKeyPair,
        endpoint: String,
        sequence: u64,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        signed_peer_descriptor_for(
            peer_identity,
            endpoint,
            sequence,
            expires_at,
            vec![NodeCapability::ChatRelay],
        )
    }

    fn signed_peer_descriptor_for(
        peer_identity: &IdentityKeyPair,
        endpoint: String,
        sequence: u64,
        expires_at: u64,
        capabilities: Vec<NodeCapability>,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            peer_identity.public_key_bytes(),
            sequence,
            sequence,
            expires_at,
            "test-chat-peer",
        );
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = capabilities;
        descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        SignedNodeDescriptor::sign(descriptor, peer_identity).unwrap()
    }

    #[test]
    fn validate_peer_envelope_rejects_tampered_ciphertext() {
        let mut envelope = signed_envelope();
        envelope.ciphertext.push(0x44);

        assert!(matches!(
            validate_peer_envelope(&envelope),
            Err(ChatPeerRelayError::InvalidSignature)
        ));
    }

    #[test]
    fn chat_peer_logs_stay_route_safe() {
        let source = include_str!("chat_peer.rs");
        let message_id_log_pattern = concat!("id = %hex::encode(envelope.", "message_id)");
        let receiver_log_pattern = concat!("receiver = %hex::encode(&envelope.", "receiver");
        let raw_error_log_pattern = concat!("error = %", "error");

        assert!(
            !source.contains(message_id_log_pattern),
            "message ids must not be logged by the relay"
        );
        assert!(
            !source.contains(receiver_log_pattern),
            "receiver prefixes must not be logged by the relay"
        );
        assert!(
            !source.contains(raw_error_log_pattern),
            "raw errors may contain endpoint URLs or route-adjacent context"
        );
        assert!(
            source.contains("reason = \"ack_decode_failed\""),
            "ACK decode failures should use a stable reason bucket"
        );
        assert!(
            source.contains("reason = \"store_pending_failed\""),
            "store failures should use a stable reason bucket"
        );
    }

    #[tokio::test]
    async fn peer_relay_endpoint_stores_offline_receiver_message() {
        let (relay, path) = temp_chat_relay("chat-peer");
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let http_client = Arc::new(reqwest::Client::new());
        let envelope = signed_envelope();
        let receiver = envelope.receiver;

        let app = build_chat_peer_router(
            Some(Arc::clone(&relay)),
            sessions,
            udp,
            peer_store,
            node_identity,
            http_client,
        );
        let body = serde_json::to_vec(&PeerChatRelayRequest { envelope }).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("pending message should be readable");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn blind_relay_endpoint_terminal_accepts_opaque_blob_without_parsing() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let http_client = Arc::new(reqwest::Client::new());
        let opaque_blob = br#"{"looks_like":"json","must_not_be_parsed":true}"#.to_vec();
        let now = now_secs();
        let envelope = BlindRelayEnvelope {
            route_id: [0x41u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: opaque_blob,
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let app = build_chat_peer_router(
            None,
            sessions,
            udp,
            Arc::clone(&peer_store),
            node_identity,
            http_client,
        );
        let body = serde_json::to_vec(&PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        })
        .unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();

        assert!(parsed.accepted);
        assert!(parsed.terminal);
        assert!(!parsed.forwarded);
        assert_eq!(parsed.ttl_remaining, 2);
        let blind_stats = peer_store.status(now + 10).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 0);
        assert!(peer_store
            .recent_audit_events()
            .iter()
            .any(|event| event.action == "blind_relay_terminal"));
    }

    #[tokio::test]
    async fn blind_relay_rejects_stale_timestamp_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let now = now_secs();
        let envelope = BlindRelayEnvelope {
            route_id: [0x42u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: br#"{"opaque":"old route frame"}"#.to_vec(),
            timestamp: now - BLIND_RELAY_MAX_ENVELOPE_AGE_SECS - 1,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::TimestampExpired)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "timestamp_expired"
        }));
    }

    #[tokio::test]
    async fn blind_relay_rejects_future_timestamp_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let now = now_secs();
        let envelope = BlindRelayEnvelope {
            route_id: [0x43u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: br#"{"opaque":"future route frame"}"#.to_vec(),
            timestamp: now + BLIND_RELAY_MAX_FUTURE_SKEW_SECS + 1,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::TimestampInFuture)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "timestamp_in_future"
        }));
    }

    #[tokio::test]
    async fn onion_terminal_layer_is_peeled_and_delivered() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let (relay, path) = temp_chat_relay("onion-terminal");
        let state = ChatPeerState {
            chat_relay: Some(Arc::clone(&relay)),
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let now = now_secs();

        // Single-hop onion addressed to this node; inner payload is a ChatEnvelope.
        let delivered_envelope = signed_envelope();
        let receiver = delivered_envelope.receiver;
        let inner = encode_envelope(&delivered_envelope).unwrap();
        let hop = OnionHop {
            node_id: node_identity.public_key_bytes(),
            // Build to the node's CURRENT rotating onion key — what the handler
            // peels with (not the identity-derived key).
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let envelope = build_onion_envelope(&[hop], &inner, [0x55u8; 16], 4, now, &source).unwrap();
        assert!(is_onion_blob(&envelope.encrypted_blob));

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        assert!(result.terminal);
        assert!(!result.forwarded);
        assert_eq!(result.reason.as_deref(), Some("onion_terminal_delivered"));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.rejected, 0);
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("terminal onion delivery should enter pending relay queue");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn onion_terminal_requires_chat_relay_delivery_before_ack() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let seen_routes = Arc::new(Mutex::new(BlindRelayRouteReplayCache::default()));
        let abuse_guard = Arc::new(Mutex::new(BlindRelayAbuseGuard::default()));
        let failed_state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::clone(&seen_routes),
            blind_relay_abuse_guard: Arc::clone(&abuse_guard),
        };
        let now = now_secs();

        let delivered_envelope = signed_envelope();
        let receiver = delivered_envelope.receiver;
        let inner = encode_envelope(&delivered_envelope).unwrap();
        let hop = OnionHop {
            node_id: node_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let envelope = build_onion_envelope(&[hop], &inner, [0x56u8; 16], 4, now, &source).unwrap();

        let result = process_peer_blind_relay(
            failed_state,
            PeerBlindRelayRequest {
                envelope: envelope.clone(),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.rejected, 1);

        let (relay, path) = temp_chat_relay("onion-terminal-retry-after-relay-failure");
        let retry_state = ChatPeerState {
            chat_relay: Some(Arc::clone(&relay)),
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::clone(&seen_routes),
            blind_relay_abuse_guard: Arc::clone(&abuse_guard),
        };

        // A terminal delivery failure must release the route id from the
        // replay cache. Otherwise a transient ChatRelay outage would make the
        // sender's retry look like a duplicate replay and permanently strand
        // the E2E-encrypted message.
        let retry = process_peer_blind_relay(
            retry_state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("terminal delivery retry should not be blocked by replay cache");

        assert!(retry.terminal);
        assert_eq!(retry.reason.as_deref(), Some("onion_terminal_delivered"));
        let (messages, has_more) = relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("retry should store the terminal onion payload");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);

        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn onion_layer_with_wrong_node_key_is_rejected() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let wrong_target = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let now = now_secs();

        // Layer is sealed to a different node's KEM key, but addressed (next_hop)
        // to this node — peel must fail without leaking anything.
        let inner = encode_envelope(&signed_envelope()).unwrap();
        let sealed_for_wrong = OnionHop {
            node_id: node_identity.public_key_bytes(),
            kem_pub: wrong_target.x25519_public_key_bytes(),
        };
        let envelope =
            build_onion_envelope(&[sealed_for_wrong], &inner, [0x56u8; 16], 4, now, &source)
                .unwrap();

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::OnionPeelFailed)));
        let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.rejected, 1);
    }

    #[tokio::test]
    async fn onion_middle_allows_fresh_signed_peer_before_routeability_probe() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, is_onion_blob, OnionHop};

        let terminal_requests: Arc<Mutex<Vec<PeerBlindRelayRequest>>> =
            Arc::new(Mutex::new(Vec::new()));
        let terminal_requests_for_route = Arc::clone(&terminal_requests);
        let terminal_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let terminal_requests_for_request = Arc::clone(&terminal_requests_for_route);
                async move {
                    terminal_requests_for_request.lock().unwrap().push(request);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 2,
                        reason: Some("terminal_next_hop".to_string()),
                    })
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, terminal_app).await.unwrap();
        });

        let now = now_secs();
        let source = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = IdentityKeyPair::generate();
        let terminal_node_id = terminal_identity.public_key_bytes();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    &terminal_identity,
                    terminal_endpoint,
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };

        // Build a true two-layer onion. The middle hop can peel only the outer
        // layer and must forward the remaining opaque onion blob without knowing
        // the final payload or the terminal's user-level receiver.
        let inner = encode_envelope(&signed_envelope()).unwrap();
        let middle_hop = OnionHop {
            node_id: middle_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let terminal_hop = OnionHop {
            node_id: terminal_node_id,
            kem_pub: terminal_identity.x25519_public_key_bytes(),
        };
        let envelope = build_onion_envelope(
            &[middle_hop, terminal_hop],
            &inner,
            [0x66u8; 16],
            4,
            now,
            &source,
        )
        .unwrap();

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(response.reason.as_deref(), Some("onion_forwarded"));

        let terminal_requests = terminal_requests.lock().unwrap();
        assert_eq!(terminal_requests.len(), 1);
        let terminal_request = &terminal_requests[0];
        assert_eq!(
            terminal_request.previous_hop_node_id,
            middle_identity.public_key_bytes()
        );
        assert_eq!(terminal_request.envelope.next_hop, terminal_node_id);
        assert!(is_onion_blob(&terminal_request.envelope.encrypted_blob));
        assert!(terminal_request.onward_envelope.is_none());

        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&terminal_node_id[..4]))
            .expect("terminal route row should remain visible");
        assert!(route_row.routeability_ready);
        assert_eq!(route_row.routeability_state, "reachable");
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);
    }

    #[tokio::test]
    async fn two_hop_onion_relay_delivers_real_ciphertext_payload_to_terminal_store() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, open_onion_layer, OnionHop};

        let now = now_secs();
        let source = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = IdentityKeyPair::generate();
        let terminal_node_id = terminal_identity.public_key_bytes();
        let terminal_secret = terminal_identity.to_x25519().0;

        let chat_sender = IdentityKeyPair::generate();
        let route_id = [0x7au8; 16];
        let receiver = [0x8bu8; 32];
        let mut delivered_envelope = ChatEnvelope {
            message_id: route_id,
            sender: chat_sender.public_key_bytes(),
            receiver,
            timestamp: now,
            ciphertext: b"real e2e ciphertext payload carried through two hops".to_vec(),
            nonce: [0x9cu8; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        delivered_envelope.signature = chat_sender.sign(&delivered_envelope.sign_data());
        let encoded_chat = encode_envelope(&delivered_envelope).unwrap();

        let (terminal_relay, terminal_db_path) = temp_chat_relay("two-hop-onion-terminal-store");
        let terminal_relay_for_route = Arc::clone(&terminal_relay);
        let terminal_previous_hops: Arc<Mutex<Vec<[u8; 32]>>> = Arc::new(Mutex::new(Vec::new()));
        let terminal_previous_hops_for_route = Arc::clone(&terminal_previous_hops);

        let terminal_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let terminal_relay_for_request = Arc::clone(&terminal_relay_for_route);
                let terminal_previous_hops_for_request =
                    Arc::clone(&terminal_previous_hops_for_route);
                async move {
                    if validate_blind_relay_envelope(
                        &request.envelope,
                        &request.previous_hop_node_id,
                        now_secs(),
                    )
                    .is_err()
                    {
                        return StatusCode::BAD_REQUEST.into_response();
                    }

                    let peel = match open_onion_layer(
                        &request.envelope.encrypted_blob,
                        &terminal_secret,
                    ) {
                        Ok(peel) => peel,
                        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                    };
                    if peel.next_hop.is_some() {
                        return StatusCode::BAD_REQUEST.into_response();
                    }

                    let inner = match decode_envelope(&peel.inner) {
                        Ok(envelope) => envelope,
                        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                    };
                    if validate_peer_envelope(&inner).is_err() {
                        return StatusCode::BAD_REQUEST.into_response();
                    }
                    if terminal_relay_for_request.store_pending(&inner).is_err() {
                        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                    }
                    terminal_previous_hops_for_request
                        .lock()
                        .unwrap()
                        .push(request.previous_hop_node_id);

                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: request.envelope.ttl,
                        reason: Some("onion_terminal_delivered".to_string()),
                    })
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, terminal_app).await.unwrap();
        });

        let mut terminal_descriptor = NodeDescriptor::new(
            terminal_node_id,
            now,
            now,
            now + 300,
            "test-terminal-onion-peer",
        )
        .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
        terminal_descriptor.public_endpoint = Some(terminal_endpoint);
        terminal_descriptor.capabilities = vec![NodeCapability::ChatRelay];
        terminal_descriptor.capacity = NodeCapacity {
            max_sessions: 32,
            max_bps: None,
            max_pps: None,
        };
        let terminal_descriptor =
            SignedNodeDescriptor::sign(terminal_descriptor, &terminal_identity).unwrap();

        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(terminal_descriptor, now, "gossip_snapshot")
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };

        let middle_hop = OnionHop {
            node_id: middle_identity.public_key_bytes(),
            kem_pub: crate::services::onion_keys::current_public_key(),
        };
        let terminal_hop = OnionHop {
            node_id: terminal_node_id,
            kem_pub: terminal_identity.x25519_public_key_bytes(),
        };
        let envelope = build_onion_envelope(
            &[middle_hop, terminal_hop],
            &encoded_chat,
            route_id,
            2,
            now,
            &source,
        )
        .unwrap();

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("middle hop should forward real onion payload to terminal");

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(response.ttl_remaining, 1);
        assert_eq!(response.reason.as_deref(), Some("onion_forwarded"));

        let previous_hops = terminal_previous_hops.lock().unwrap();
        assert_eq!(
            previous_hops.as_slice(),
            &[middle_identity.public_key_bytes()]
        );
        drop(previous_hops);

        let (messages, has_more) = terminal_relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .expect("terminal should store the delivered E2E envelope");
        assert!(!has_more);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_id, delivered_envelope.message_id);
        assert_eq!(
            messages[0].envelope.ciphertext,
            delivered_envelope.ciphertext
        );
        assert_eq!(messages[0].envelope.nonce, delivered_envelope.nonce);
        assert_eq!(messages[0].envelope.sender, delivered_envelope.sender);
        assert_eq!(messages[0].envelope.receiver, delivered_envelope.receiver);

        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);

        let _ = std::fs::remove_file(terminal_db_path);
    }

    #[tokio::test]
    async fn blind_relay_in_flight_guard_enforces_backpressure_limit() {
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store,
            node_identity: Arc::new(IdentityKeyPair::generate()),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };

        assert!(BlindRelayInFlightGuard::try_acquire(&state).is_none());
    }

    #[tokio::test]
    async fn blind_relay_rejects_immediate_previous_hop_loop_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x44u8; 16],
            next_hop: previous_hop.public_key_bytes(),
            ttl: 2,
            encrypted_blob: br#"{"opaque":"must_not_be_parsed"}"#.to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::RouteLoop)));
        let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.loop_detected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "route_loop"
        }));
    }

    #[test]
    fn blind_relay_replay_cache_forgets_failed_route_ids() {
        let mut cache = BlindRelayRouteReplayCache::default();
        let route_id = [0x46u8; 16];

        assert!(!cache.observe(route_id, 1_800_000_000));
        assert!(cache.observe(route_id, 1_800_000_001));
        cache.forget(&route_id);
        assert!(!cache.observe(route_id, 1_800_000_002));
    }

    #[test]
    fn blind_relay_abuse_guard_rate_limits_previous_hop_without_payload_data() {
        let mut guard = BlindRelayAbuseGuard::default();
        let previous_hop = [0x52u8; 32];
        let now = 1_800_000_000;

        for _ in 0..BLIND_RELAY_PREVIOUS_HOP_RATE_LIMIT {
            assert_eq!(
                guard.observe_request(previous_hop, now),
                BlindRelayAbuseDecision::Allowed
            );
        }

        assert_eq!(
            guard.observe_request(previous_hop, now),
            BlindRelayAbuseDecision::RateLimited {
                quarantine_until: now + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
        assert_eq!(
            guard.observe_request(previous_hop, now + 1),
            BlindRelayAbuseDecision::Quarantined {
                quarantine_until: now + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
    }

    #[test]
    fn blind_relay_abuse_guard_quarantines_repeated_bad_previous_hop() {
        let mut guard = BlindRelayAbuseGuard::default();
        let previous_hop = [0x53u8; 32];
        let now = 1_800_000_000;

        for offset in 0..(BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD - 1) {
            assert_eq!(
                guard.record_failure(previous_hop, now + u64::from(offset)),
                None
            );
        }

        let quarantine_at = now + u64::from(BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD);
        assert_eq!(
            guard.record_failure(previous_hop, quarantine_at),
            Some(quarantine_at + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS)
        );
        assert_eq!(
            guard.observe_request(previous_hop, quarantine_at + 1),
            BlindRelayAbuseDecision::Quarantined {
                quarantine_until: quarantine_at + BLIND_RELAY_PREVIOUS_HOP_QUARANTINE_SECS
            }
        );
    }

    #[tokio::test]
    async fn blind_relay_drops_duplicate_route_id_without_forwarding_again() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x45u8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted replay candidate".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let first = process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: envelope.clone(),
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();
        let duplicate = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        assert!(first.terminal);
        assert!(!duplicate.terminal);
        assert!(!duplicate.forwarded);
        assert_eq!(duplicate.reason.as_deref(), Some("duplicate_route"));
        let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
        assert_eq!(blind_stats.received, 2);
        assert_eq!(blind_stats.terminal, 1);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.replay_dropped, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "duplicate_route"
        }));
        assert!(!blind_relay_reason_counts_toward_quarantine(
            "duplicate_route"
        ));
    }

    #[tokio::test]
    async fn blind_relay_forward_retries_transient_next_hop_failure() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    let attempt = attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    if attempt == 0 {
                        StatusCode::SERVICE_UNAVAILABLE.into_response()
                    } else {
                        Json(PeerBlindRelayResponse {
                            accepted: true,
                            terminal: true,
                            forwarded: false,
                            ttl_remaining: 1,
                            reason: Some("terminal_next_hop".to_string()),
                        })
                        .into_response()
                    }
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x42u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);
        assert_eq!(blind_stats.retry_attempted, 1);
        assert_eq!(blind_stats.retry_succeeded, 1);
        assert_eq!(blind_stats.retry_exhausted, 0);
        assert!(peer_store
            .recent_audit_events()
            .iter()
            .any(|event| { event.action == "blind_relay_retry" && event.outcome == "accepted" }));
    }

    #[tokio::test]
    async fn blind_relay_middle_hop_forwards_onward_envelope_without_payload_inspection() {
        let terminal_requests: Arc<Mutex<Vec<PeerBlindRelayRequest>>> =
            Arc::new(Mutex::new(Vec::new()));
        let terminal_requests_for_route = Arc::clone(&terminal_requests);
        let terminal_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let terminal_requests_for_request = Arc::clone(&terminal_requests_for_route);
                async move {
                    terminal_requests_for_request.lock().unwrap().push(request);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 0,
                        reason: Some("terminal_next_hop".to_string()),
                    })
                    .into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, terminal_app).await.unwrap();
        });

        let now = now_secs();
        let entry_identity = IdentityKeyPair::generate();
        let middle_identity = Arc::new(IdentityKeyPair::generate());
        let terminal_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    &terminal_identity,
                    terminal_endpoint,
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&terminal_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&middle_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let outer_envelope = BlindRelayEnvelope {
            route_id: [0x62u8; 16],
            next_hop: middle_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque middle-hop carrier; do not parse".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&entry_identity);
        let onward_envelope = BlindRelayEnvelope {
            route_id: [0x63u8; 16],
            next_hop: terminal_identity.public_key_bytes(),
            ttl: 1,
            encrypted_blob: b"opaque terminal relay blob; do not parse".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&entry_identity);

        let response = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope: outer_envelope,
                previous_hop_node_id: entry_identity.public_key_bytes(),
                onward_envelope: Some(onward_envelope),
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap();

        server.abort();

        assert!(response.accepted);
        assert!(response.forwarded);
        assert!(!response.terminal);
        assert_eq!(response.reason.as_deref(), Some("onion_middle_forwarded"));

        let terminal_requests = terminal_requests.lock().unwrap();
        assert_eq!(terminal_requests.len(), 1);
        let terminal_request = &terminal_requests[0];
        assert_eq!(
            terminal_request.previous_hop_node_id,
            middle_identity.public_key_bytes()
        );
        assert_eq!(
            terminal_request.envelope.next_hop,
            terminal_identity.public_key_bytes()
        );
        assert_eq!(terminal_request.envelope.ttl, 0);
        assert!(terminal_request.onward_envelope.is_none());
        let middle_public =
            IdentityPublicKey::from_bytes(&middle_identity.public_key_bytes()).unwrap();
        assert!(terminal_request
            .envelope
            .verify_signature_from(&middle_public)
            .is_ok());
        assert_eq!(
            terminal_request.envelope.encrypted_blob,
            b"opaque terminal relay blob; do not parse".to_vec()
        );

        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 1);
        assert_eq!(blind_stats.rejected, 0);
    }

    #[tokio::test]
    async fn blind_relay_forward_requires_accepted_next_hop_ack() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerBlindRelayResponse {
                        accepted: false,
                        terminal: false,
                        forwarded: false,
                        ttl_remaining: 1,
                        reason: Some("relay_unavailable".to_string()),
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x56u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "forward_failed"
        }));
    }

    #[tokio::test]
    async fn blind_relay_forward_rejects_malformed_success_ack() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    (StatusCode::OK, "not-a-peer-blind-relay-ack").into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x57u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert_eq!(blind_stats.retry_attempted, 0);
        assert_eq!(blind_stats.retry_exhausted, 0);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "forward_failed"
                && !event.detail.contains("not-a-peer-blind-relay-ack")
        }));
    }

    #[tokio::test]
    async fn blind_relay_requires_next_hop_chat_relay_capability() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::OK.into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_peer_descriptor_for(
                    &next_hop_identity,
                    endpoint,
                    now,
                    now + 300,
                    vec![NodeCapability::PrivacyRelay],
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x54u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::NoRoute)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 0);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.no_route, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "no_route"
        }));
    }

    #[tokio::test]
    async fn blind_relay_requires_routeability_evidence_before_forwarding() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 1,
                        reason: Some("terminal_next_hop".to_string()),
                    })
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let next_hop_node_id = next_hop_identity.public_key_bytes();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x59u8; 16],
            next_hop: next_hop_node_id,
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::NoRoute)));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 0);
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.no_route, 1);
        let route_status = peer_store.route_candidate_status(now + 5);
        let route_row = route_status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
            .expect("chat relay row should remain visible");
        assert_eq!(route_row.routeability_state, "unreachable");
        assert!(!route_row.routeability_ready);
        assert_eq!(
            route_row.last_route_failure_reason.as_deref(),
            Some("routeability_not_ready")
        );
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_route_health"
                && event.outcome == "rejected"
                && event.detail.contains("reason=routeability_not_ready")
                && !event.detail.contains("opaque encrypted relay bytes")
        }));
    }

    #[tokio::test]
    async fn blind_relay_forward_reports_retry_exhaustion_without_payload_data() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_route = Arc::clone(&attempts);
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(move |Json(_request): Json<PeerBlindRelayRequest>| {
                let attempts_for_request = Arc::clone(&attempts_for_route);
                async move {
                    attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::SERVICE_UNAVAILABLE.into_response()
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x43u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        assert_eq!(
            attempts.load(AtomicOrdering::SeqCst),
            MAX_BLIND_RELAY_FORWARD_ATTEMPTS
        );
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert_eq!(
            blind_stats.retry_attempted,
            (MAX_BLIND_RELAY_FORWARD_ATTEMPTS - 1) as u64
        );
        assert_eq!(blind_stats.retry_succeeded, 0);
        assert_eq!(blind_stats.retry_exhausted, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_retry"
                && event.outcome == "rejected"
                && !event.detail.contains("opaque encrypted relay bytes")
        }));
    }

    #[tokio::test]
    async fn blind_relay_forward_retries_timeout_without_endpoint_leak() {
        let next_hop_app = Router::new().route(
            "/api/chat/peer/blind-relay",
            post(
                move |Json(_request): Json<PeerBlindRelayRequest>| async move {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: true,
                        forwarded: false,
                        ttl_remaining: 1,
                        reason: Some("terminal_next_hop".to_string()),
                    })
                },
            ),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, next_hop_app).await.unwrap();
        });

        let now = now_secs();
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let next_hop_identity = IdentityKeyPair::generate();
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(
                signed_chat_relay_peer_descriptor_for(
                    &next_hop_identity,
                    endpoint.clone(),
                    now,
                    now + 300,
                ),
                now,
                "gossip_snapshot",
            )
            .unwrap();
        peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

        let state = ChatPeerState {
            chat_relay: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity,
            http_client: Arc::new(
                reqwest::Client::builder()
                    .timeout(std::time::Duration::from_millis(30))
                    .build()
                    .unwrap(),
            ),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id: [0x58u8; 16],
            next_hop: next_hop_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let result = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        server.abort();

        assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
        let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert_eq!(blind_stats.forward_failed, 1);
        assert_eq!(
            blind_stats.retry_attempted,
            (MAX_BLIND_RELAY_FORWARD_ATTEMPTS - 1) as u64
        );
        assert_eq!(blind_stats.retry_succeeded, 0);
        assert_eq!(blind_stats.retry_exhausted, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_retry"
                && event.outcome == "scheduled"
                && event
                    .detail
                    .contains("reason_bucket=blind_relay_request_timeout")
                && !event.detail.contains(&endpoint)
        }));
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_retry"
                && event.outcome == "rejected"
                && event
                    .detail
                    .contains("reason_bucket=blind_relay_request_timeout")
                && !event.detail.contains(&endpoint)
                && !event.detail.contains("opaque encrypted relay bytes")
        }));
    }
}
