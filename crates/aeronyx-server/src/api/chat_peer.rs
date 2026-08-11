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
//! - Returns a terminal-signed receipt bound to the exact opaque payload after
//!   successful onion terminal store-and-forward; middle hops only propagate it
//! - [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Signs receipt v2 with a
//!   purpose-separated opaque payload commitment after terminal acceptance;
//!   v1 remains signature-verifiable for rolling-upgrade relay compatibility
//! - [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] Accepts a bounded,
//!   signed Blind Vault `Put` frame as an alternative onion terminal payload,
//!   reusing the existing anonymous lease, quota, TTL, and idempotency service
//! - [MULTIHOP-RECEIPT-VALIDATION 2026-08-01 by Codex] Validates terminal ACKs
//!   against the immediate next hop while allowing a forwarded ACK to carry a
//!   valid downstream terminal receipt through three-hop and longer paths
//! - [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Carries the actual
//!   response-observation time out of retry handling so receipt freshness and
//!   route-health evidence never reuse a stale request-ingress timestamp
//! - [DURABLE-TERMINAL-REPLAY-WINDOW 2026-08-11 by Codex] Starts terminal
//!   receipt and replay retention at durable acceptance, while generation-tags
//!   replay queue entries so a forgotten route cannot evict a newer reuse
//! - [REPLAY-GENERATION-COMPACTION 2026-08-11 by Codex] Uses unique local
//!   generations instead of second-resolution timestamps for replay eviction,
//!   and compacts stale generations under a strict memory bound
//! - [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] Distinguishes in-flight route
//!   retries from completed delivery replays and retains the exact bounded ACK,
//!   so a lost response cannot erase a terminal delivery receipt or create a
//!   false acceptance while the original attempt is still unresolved
//! - [RELAY-ROUTE-RAII 2026-08-11 by Codex] Owns every newly admitted route
//!   through an RAII lease so cancellation, shutdown, and future early-return
//!   paths release in-flight replay state unless a durable ACK is committed
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/chat.rs: `ChatEnvelope`, `BlindRelayEnvelope`,
//!   and bounded envelope encoding
//! - aeronyx-core/src/protocol/memchain.rs: wraps envelope for client delivery
//! - aeronyx-server/src/services/chat_relay.rs: pending queue and dedup logic
//! - aeronyx-server/src/services/blind_vault.rs: anonymous encrypted-object
//!   persistence for receiver-independent store-and-forward
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
//! - Route-specific request ceilings and concurrency gates must wrap the Axum
//!   handlers. Putting them inside a `Json<T>` handler is too late because an
//!   attacker can consume memory and parser work before the guard executes.
//! - Next-hop acknowledgement bodies are untrusted and must use the shared
//!   bounded decoder. Never call `Response::json()` directly on peer traffic.
//! - Durable queue count/byte exhaustion is a retryable capacity condition,
//!   not an internal server fault. Preserve the service's privacy-safe reason
//!   bucket while returning HTTP 503 to the previous hop.
//! - Delivery receipts authenticate terminal acceptance only. They must not add
//!   sender, receiver, endpoint, online-state, mailbox-state, or payload-size
//!   fields. Intermediates verify route, freshness, and signature, but only the
//!   source knows the complete route and final payload commitment and can
//!   therefore enforce the final terminal/payload binding.
//! - A successful downstream ACK must describe exactly one completed action:
//!   terminal acceptance or onward forwarding. `accepted=true` without either
//!   action is not delivery evidence and must never improve route reputation.
//! - Blind Vault's detailed storage receipt contains stable replica-local lease
//!   metadata and therefore must not be exposed in a multi-hop JSON ACK. The
//!   existing delivery receipt is signed only after `BlindVaultService::put`
//!   succeeds and is bound to the exact encoded Put frame; the source can prove
//!   replica acceptance without revealing the lease or object to middle hops.
//! - Blind Vault terminal failures expose only permanent rejection, replica
//!   capacity, or temporary unavailability. Never forward service errors.
//! - Receipt v2 purpose separation must stay inside the opaque commitment.
//!   Do not add a clear workload label to the propagated ACK: low-cardinality
//!   route purpose would become visible metadata at every middle hop.
//! - [ROUTE-SUCCESS-SURFACE-BINDING 2026-08-10 by Codex] Successful next-hop
//!   forwarding must be recorded against the exact descriptor used to build
//!   the request URL; a concurrent endpoint/KEM rotation must fail closed.
//! - [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Retry completion,
//!   receipt freshness, previous-hop success, and route-health evidence must
//!   use one projected response-observation time returned by the forwarder.
//! - [DURABLE-TERMINAL-REPLAY-WINDOW 2026-08-11 by Codex] Replay eviction must
//!   compare the queued generation with the current route entry;
//!   terminal receipts and replay retention begin only after durable success.
//! - [REPLAY-GENERATION-COMPACTION 2026-08-11 by Codex] A replay queue entry
//!   must carry a unique process-local generation. Timestamps are not unique:
//!   fail/retry can reuse one route id within the same second. Keep the queue
//!   bounded independently from the live route map to prevent stale-entry DoS.
//!
//! ## Last Modified
//! v0.36.0-RelayAckStateMachine - Require successful downstream ACKs to prove
//! exactly one terminal or forwarding disposition before recording success
//! v0.35.0-ReplayGenerationCompaction - Make same-second route reuse safe and
//! bound stale replay generations independently from live route capacity
//! v0.34.0-DurableTerminalReplayWindow - Bind terminal receipt time to durable
//! acceptance and make replay-cache eviction generation-safe
//! v0.33.0-RelayResponseObservationTime - Bind retry ACK validation and route
//! evidence to response time rather than request-ingress time
//! v0.32.0-RouteSuccessSurfaceBinding - Bound next-hop success observations to
//! the exact signed descriptor used for each opaque forward
//! v0.31.0-PurposeBoundReceipt - Sign terminal workload into opaque receipt v2 commitments
//! v0.30.0-BlindVaultRetryClass - Stop retrying permanently invalid anonymous
//! writes while preserving coarse capacity and availability failover signals
//! v0.29.0-BlindVaultOnionDispatch - Persist signed anonymous Blind Vault Put
//! frames at onion terminals without exposing lease/object metadata in ACKs
//! v0.28.0-MultihopReceiptValidation - Keep direct-terminal signer checks while
//! accepting valid downstream terminal receipts propagated through longer paths
//! v0.27.0-PeerEndpointPolicy - Enforce canonical public-IP-only next-hop URLs
//! v0.26.0-SignedDeliveryReceipt - Sign exact terminal payload acceptance and propagate verified receipts
//! v0.25.0-DurableQueueCapacity - Classify global pending-store quota exhaustion
//! v0.24.0-PublicRequestBounds - Bound peer bodies and concurrency before JSON extraction
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
    sync::{atomic::AtomicUsize, Arc, Mutex},
    time::{Duration, Instant},
};

use aeronyx_core::crypto::transport::{
    DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD,
};
use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::{
    decode_envelope, encode_blind_relay_envelope, encode_envelope, BlindRelayDeliveryReceipt,
    BlindRelayEnvelope, ChatEnvelope,
};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::discovery::SignedNodeDescriptor;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::onion::{is_onion_blob, try_open_onion_layer, OnionRoutePurpose};
use aeronyx_core::protocol::{
    decode_blind_vault_frame, is_blind_vault_frame, BlindVaultFrame, DataPacket, NodeCapability,
};
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::time::sleep;
use tracing::{debug, warn};

use crate::api::{
    canonical_peer_http_url, decode_bounded_json_response, peer_endpoint_is_public_ip,
    InFlightRequestGuard, PEER_ACK_RESPONSE_MAX_BYTES,
};
use crate::services::chat_relay::ChatRelayError;
use crate::services::peer_store::PeerStore;
use crate::services::{
    BlindVaultPutFailureClass, BlindVaultServiceError, ChatRelayService, Session, SessionManager,
    SharedBlindVaultService,
};

// ============================================
// Constants
// ============================================

/// Maximum bincode-encoded envelope bytes accepted from another node.
///
/// This mirrors the protocol decode limit and protects the JSON endpoint from
/// carrying huge opaque payloads. Encrypted files should use blob storage, not
/// the peer envelope relay path.
const MAX_PEER_CHAT_ENVELOPE_BYTES: usize = 128 * 1024;

/// Maximum ordinary peer-relay JSON body accepted before deserialization.
///
/// `ChatEnvelope` has a 128 KiB binary ceiling, while JSON byte arrays are
/// substantially larger. This transport allowance preserves valid envelopes
/// without allowing an untrusted peer to allocate an unbounded request body.
const PEER_CHAT_REQUEST_BODY_MAX_BYTES: usize = 768 * 1024;

/// Maximum blind-relay JSON body accepted before deserialization.
///
/// A two-hop request can contain two independently bounded 192 KiB opaque
/// blobs plus a signed descriptor. JSON byte arrays can expand to roughly four
/// characters per byte, so 2 MiB is the narrow safe ceiling for that contract.
const PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES: usize = 2 * 1024 * 1024;

/// Maximum ordinary peer-relay requests allowed in parser/handler execution.
const MAX_IN_FLIGHT_PEER_CHAT_REQUESTS: usize = 64;

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

/// Maximum live and stale generations retained by the replay eviction queue.
const MAX_BLIND_RELAY_REPLAY_QUEUE_GENERATIONS: usize = MAX_BLIND_RELAY_SEEN_ROUTES * 2;

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
/// Terminal delivery receipts are short-lived acknowledgements, not durable tokens.
const BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS: u64 = 120;
/// Delivery receipt future skew is intentionally tighter than relay-frame skew.
const BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS: u64 = 30;

// ============================================
// State / Request / Response Types
// ============================================

#[derive(Clone)]
struct ChatPeerState {
    chat_relay: Option<Arc<ChatRelayService>>,
    /// Optional anonymous ciphertext store used only for declared Blind Vault
    /// terminal frames. Absence is fail-closed and never falls back to chat.
    blind_vault: Option<SharedBlindVaultService>,
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,
    peer_store: Arc<PeerStore>,
    node_identity: Arc<IdentityKeyPair>,
    http_client: Arc<reqwest::Client>,
    blind_relay_in_flight: Arc<AtomicUsize>,
    blind_relay_seen_routes: Arc<Mutex<BlindRelayRouteReplayCache>>,
    blind_relay_abuse_guard: Arc<Mutex<BlindRelayAbuseGuard>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BlindRelayRouteReplayState {
    InFlight,
    Completed(PeerBlindRelayResponse),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BlindRelayRouteReplayEntry {
    observed_at: u64,
    generation: u64,
    state: BlindRelayRouteReplayState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BlindRelayRouteReplayDecision {
    New,
    InFlight,
    Completed(PeerBlindRelayResponse),
    Saturated,
}

enum BlindRelayRouteStart {
    Acquired(BlindRelayRouteLease),
    Completed(PeerBlindRelayResponse),
}

/// Owns one in-flight route until its durable outcome is published.
///
/// [RELAY-ROUTE-RAII 2026-08-11 by Codex] Axum request futures may be dropped
/// during shutdown or transport cancellation. Releasing from `Drop` prevents
/// a cancelled owner from pinning the route in `InFlight` until replay expiry.
/// Successful paths consume the lease through `complete`, atomically replacing
/// the in-flight marker with the exact bounded ACK before disarming cleanup.
struct BlindRelayRouteLease {
    seen_routes: Arc<Mutex<BlindRelayRouteReplayCache>>,
    route_id: [u8; 16],
    active: bool,
}

impl BlindRelayRouteLease {
    fn new(seen_routes: Arc<Mutex<BlindRelayRouteReplayCache>>, route_id: [u8; 16]) -> Self {
        Self {
            seen_routes,
            route_id,
            active: true,
        }
    }

    fn complete(mut self, now: u64, response: PeerBlindRelayResponse) {
        let mut seen_routes = self
            .seen_routes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        seen_routes.complete(&self.route_id, now, response);
        self.active = false;
    }
}

impl Drop for BlindRelayRouteLease {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut seen_routes = self
            .seen_routes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        seen_routes.forget(&self.route_id);
    }
}

#[derive(Default)]
struct BlindRelayRouteReplayCache {
    seen: HashMap<[u8; 16], BlindRelayRouteReplayEntry>,
    order: VecDeque<([u8; 16], u64)>,
    monotonic_observed_at: u64,
    generation_counter: u64,
}

impl BlindRelayRouteReplayCache {
    fn observe(&mut self, route_id: [u8; 16], now: u64) -> BlindRelayRouteReplayDecision {
        let now = self.normalize_observation_time(now);
        self.evict_expired(now);
        if let Some(entry) = self.seen.get(&route_id) {
            return match &entry.state {
                BlindRelayRouteReplayState::InFlight => BlindRelayRouteReplayDecision::InFlight,
                BlindRelayRouteReplayState::Completed(response) => {
                    BlindRelayRouteReplayDecision::Completed(response.clone())
                }
            };
        }

        let generation = self.allocate_generation();
        self.seen.insert(
            route_id,
            BlindRelayRouteReplayEntry {
                observed_at: now,
                generation,
                state: BlindRelayRouteReplayState::InFlight,
            },
        );
        self.order.push_back((route_id, generation));
        let retained = self.evict_over_capacity(route_id, generation);
        self.compact_stale_generations_if_needed();
        if retained {
            BlindRelayRouteReplayDecision::New
        } else {
            BlindRelayRouteReplayDecision::Saturated
        }
    }

    /// Moves one accepted route's replay horizon to its completion boundary.
    ///
    /// [DURABLE-TERMINAL-REPLAY-WINDOW 2026-08-11 by Codex] The old queue
    /// entry remains as a harmless stale generation. Eviction removes a route
    /// only when the queued generation still matches the current map value.
    fn complete(&mut self, route_id: &[u8; 16], now: u64, response: PeerBlindRelayResponse) {
        let now = self.normalize_observation_time(now);
        let Some(current) = self.seen.get(route_id).cloned() else {
            return;
        };
        if current.observed_at >= now {
            if let Some(entry) = self.seen.get_mut(route_id) {
                entry.state = BlindRelayRouteReplayState::Completed(response);
            }
            return;
        }
        let generation = self.allocate_generation();
        self.seen.insert(
            *route_id,
            BlindRelayRouteReplayEntry {
                observed_at: now,
                generation,
                state: BlindRelayRouteReplayState::Completed(response),
            },
        );
        self.order.push_back((*route_id, generation));
        self.evict_expired(now);
        // Completion never increases the number of live map entries.
        self.compact_stale_generations_if_needed();
    }

    fn normalize_observation_time(&mut self, now: u64) -> u64 {
        self.monotonic_observed_at = self.monotonic_observed_at.max(now);
        self.monotonic_observed_at
    }

    fn allocate_generation(&mut self) -> u64 {
        // Generation zero is reserved for the `Default` state. Exhausting the
        // full u64 space in one process is infeasible; skipping zero also keeps
        // wrap behavior deterministic in fuzzing and model tests.
        self.generation_counter = self.generation_counter.wrapping_add(1);
        if self.generation_counter == 0 {
            self.generation_counter = 1;
        }
        self.generation_counter
    }

    fn evict_expired(&mut self, now: u64) {
        while let Some((route_id, queued_generation)) = self.order.front().copied() {
            let Some(current) = self.seen.get(&route_id) else {
                self.order.pop_front();
                continue;
            };
            if current.generation != queued_generation {
                self.order.pop_front();
                continue;
            }
            if now.saturating_sub(current.observed_at) <= BLIND_RELAY_ROUTE_REPLAY_WINDOW_SECS {
                break;
            }
            self.order.pop_front();
            self.seen.remove(&route_id);
        }
    }

    fn evict_over_capacity(&mut self, new_route_id: [u8; 16], new_generation: u64) -> bool {
        while self.seen.len() > MAX_BLIND_RELAY_SEEN_ROUTES {
            // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] Never evict an
            // unresolved route to admit newer work: doing so permits a retry
            // to execute concurrently and forward the same ciphertext twice.
            // Completed ACKs are bounded and safe to evict oldest-first.
            let scan_limit = self.order.len();
            let mut evicted_completed = false;
            for _ in 0..scan_limit {
                let Some((route_id, queued_generation)) = self.order.pop_front() else {
                    break;
                };
                let Some(entry) = self.seen.get(&route_id) else {
                    continue;
                };
                if entry.generation != queued_generation {
                    continue;
                }
                if matches!(entry.state, BlindRelayRouteReplayState::Completed(_)) {
                    self.seen.remove(&route_id);
                    evicted_completed = true;
                    break;
                }
                self.order.push_back((route_id, queued_generation));
            }
            if !evicted_completed {
                self.seen.remove(&new_route_id);
                self.order.retain(|(route_id, generation)| {
                    *route_id != new_route_id || *generation != new_generation
                });
                return false;
            }
        }
        true
    }

    fn forget(&mut self, route_id: &[u8; 16]) {
        self.seen.remove(route_id);
        self.compact_stale_generations_if_needed();
    }

    fn compact_stale_generations_if_needed(&mut self) {
        if self.order.len() <= MAX_BLIND_RELAY_REPLAY_QUEUE_GENERATIONS {
            return;
        }

        let seen = &self.seen;
        self.order.retain(|(route_id, generation)| {
            seen.get(route_id)
                .is_some_and(|entry| entry.generation == *generation)
        });
        debug_assert!(self.order.len() <= self.seen.len());
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
    /// Optional terminal-signed proof bound to the exact delivered payload.
    ///
    /// Older nodes omit this field. Intermediate nodes may propagate it but
    /// must not infer sender, receiver, online state, or payload contents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_receipt: Option<BlindRelayDeliveryReceipt>,
}

/// Internal result of one accepted next-hop relay round.
///
/// The observation timestamp is deliberately process-local and never enters
/// the peer wire contract. Callers use it for every success-side state write,
/// keeping receipt verification and route evidence on one clock snapshot.
struct BlindRelayForwardOutcome {
    response: PeerBlindRelayResponse,
    observed_at: u64,
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

    #[error("pending store capacity exhausted")]
    PendingCapacity,
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

    #[error("blind relay route is still in flight")]
    RouteInFlight,

    #[error("blind relay replay cache capacity exhausted")]
    ReplayCapacity,

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

    #[error("onion terminal payload rejected")]
    OnionTerminalPayloadRejected,

    #[error("onion terminal replica capacity exhausted")]
    OnionTerminalCapacityExhausted,

    #[error("downstream blind relay rejected request")]
    DownstreamRejected,
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
            | Self::OnionPeelFailed
            | Self::OnionTerminalPayloadRejected
            | Self::DownstreamRejected => StatusCode::BAD_REQUEST,
            Self::OnionTerminalCapacityExhausted | Self::RouteInFlight | Self::ReplayCapacity => {
                StatusCode::SERVICE_UNAVAILABLE
            }
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
            Self::RouteInFlight => "route_in_flight",
            Self::ReplayCapacity => "replay_capacity",
            Self::RouteLoop => "route_loop",
            Self::NoRoute => "no_route",
            Self::InvalidEndpoint => "invalid_endpoint",
            Self::ForwardFailed => "forward_failed",
            Self::OnionPeelFailed => "onion_peel_failed",
            Self::OnionTerminalPayloadRejected => "onion_terminal_payload_rejected",
            Self::OnionTerminalCapacityExhausted => "onion_terminal_capacity_exhausted",
            Self::DownstreamRejected => "downstream_rejected",
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
            Self::PendingCapacity => StatusCode::SERVICE_UNAVAILABLE,
            Self::StoreFailed => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    fn reason_bucket(&self) -> &'static str {
        match self {
            Self::RelayUnavailable => "relay_unavailable",
            Self::InvalidSignature => "invalid_signature",
            Self::EnvelopeTooLarge { .. } => "envelope_too_large",
            Self::Serialization => "envelope_serialization_failed",
            Self::PendingCapacity => "pending_capacity_exhausted",
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
    blind_vault: Option<SharedBlindVaultService>,
) -> Router {
    let state = ChatPeerState {
        chat_relay,
        blind_vault,
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
        blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
    };
    let peer_relay_in_flight = Arc::new(AtomicUsize::new(0));

    let peer_relay_router = Router::new()
        .route("/api/chat/peer/relay", post(peer_relay_handler))
        .route_layer(middleware::from_fn_with_state(
            peer_relay_in_flight,
            peer_relay_request_gate,
        ))
        .layer(DefaultBodyLimit::max(PEER_CHAT_REQUEST_BODY_MAX_BYTES));
    let blind_relay_router = Router::new()
        .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            peer_blind_relay_request_gate,
        ))
        .layer(DefaultBodyLimit::max(
            PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES,
        ));

    peer_relay_router
        .merge(blind_relay_router)
        .with_state(state)
}

// ============================================
// Handlers
// ============================================

/// Applies ordinary peer-relay backpressure before Axum reads a JSON body.
async fn peer_relay_request_gate(
    State(counter): State<Arc<AtomicUsize>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(_in_flight) =
        InFlightRequestGuard::try_acquire(&counter, MAX_IN_FLIGHT_PEER_CHAT_REQUESTS)
    else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            }),
        )
            .into_response();
    };

    next.run(request).await
}

/// Applies blind-relay backpressure before Axum reads a JSON body.
async fn peer_blind_relay_request_gate(
    State(state): State<ChatPeerState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(_in_flight) = InFlightRequestGuard::try_acquire(
        &state.blind_relay_in_flight,
        MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS,
    ) else {
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
                delivery_receipt: None,
            }),
        )
            .into_response();
    };

    next.run(request).await
}

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
                delivery_receipt: None,
            }),
        )
            .into_response(),
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
        relay.store_pending(&envelope).map_err(|error| {
            let reason = error.reason_bucket();
            warn!(
                reason,
                "[CHAT_PEER] Failed to store peer envelope for offline receiver"
            );
            relay.record_peer_relay_inbound_rejected(now, reason);
            map_pending_store_error(&error)
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

fn map_pending_store_error(error: &ChatRelayError) -> ChatPeerRelayError {
    match error {
        ChatRelayError::MessageTooLarge { size, .. } => {
            ChatPeerRelayError::EnvelopeTooLarge { size: *size }
        }
        error if error.is_capacity_exhausted() => ChatPeerRelayError::PendingCapacity,
        _ => ChatPeerRelayError::StoreFailed,
    }
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
    let route_started_at = Instant::now();
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
            return process_onion_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                now,
                &route_started_at,
            )
            .await;
        }
        if let Some(onward_envelope) = request.onward_envelope {
            return process_onion_middle_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                onward_envelope,
                onward_descriptor_hint,
                now,
                &route_started_at,
            )
            .await;
        }
        let route_lease =
            match begin_blind_relay_route(&state, envelope.route_id, previous_hop_node_id, now)? {
                BlindRelayRouteStart::Acquired(lease) => lease,
                BlindRelayRouteStart::Completed(response) => return Ok(response),
            };
        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: envelope.ttl,
            reason: Some("terminal_next_hop".to_string()),
            delivery_receipt: None,
        };
        route_lease.complete(now, response.clone());
        record_blind_relay_previous_hop_success(&state, previous_hop_node_id, now);
        state.peer_store.record_blind_relay_terminal(
            now,
            envelope.ttl,
            envelope.encrypted_blob.len(),
        );
        return Ok(response);
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
        // [ROUTE-HEALTH-REMOTE-POISONING 2026-08-11 by Codex] A remote
        // previous hop can choose this target. Preflight rejection is therefore
        // not next-hop failure evidence; only a real outbound request may
        // mutate that peer's route health. The requester is rejected below.
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let route_lease =
        match begin_blind_relay_route(&state, envelope.route_id, previous_hop_node_id, now)? {
            BlindRelayRouteStart::Acquired(lease) => lease,
            BlindRelayRouteStart::Completed(response) => return Ok(response),
        };

    let forwarded_envelope = envelope
        .decremented_ttl()
        .ok_or(BlindRelayError::TtlExhausted)?
        .sign_with(state.node_identity.as_ref());
    let forwarded_onward_envelope = request
        .onward_envelope
        .map(|envelope| envelope.sign_with(state.node_identity.as_ref()));
    let forwarded_onward_descriptor_hint = onward_descriptor_hint;
    let ttl_remaining = forwarded_envelope.ttl;

    let forward_started_at = blind_relay_response_observed_at(now, &route_started_at);
    let observed_at = match forward_blind_relay_with_retry(
        &state,
        &url,
        &descriptor,
        PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: self_node_id,
            onward_envelope: forwarded_onward_envelope,
            onward_descriptor_hint: forwarded_onward_descriptor_hint,
        },
        forward_started_at,
    )
    .await
    {
        Ok(outcome) => outcome.observed_at,
        Err(error) => return Err(error),
    };

    let response = PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining,
        reason: Some("forwarded".to_string()),
        delivery_receipt: None,
    };
    route_lease.complete(observed_at, response.clone());
    let _ = state
        .peer_store
        .record_route_forward_success_for_descriptor(&descriptor, observed_at);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id, observed_at);
    state
        .peer_store
        .record_blind_relay_forwarded(observed_at, ttl_remaining);

    Ok(response)
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
    route_started_at: &Instant,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let self_node_id = state.node_identity.public_key_bytes();

    // Per-route replay/dedup, identical to the opaque terminal/forward paths.
    let route_lease =
        match begin_blind_relay_route(&state, envelope.route_id, previous_hop_node_id, now)? {
            BlindRelayRouteStart::Acquired(lease) => lease,
            BlindRelayRouteStart::Completed(response) => return Ok(response),
        };

    // Peel exactly one onion layer with the node's rotating onion key(s): the
    // current key, plus the previous key while it is within the rotation grace
    // window (forward secrecy — see services::onion_keys). A failure yields a
    // coarse bucket only, never a payload leak.
    let onion_secrets = crate::services::onion_keys::peel_secrets(now);
    let peel = match try_open_onion_layer(&envelope.encrypted_blob, &onion_secrets) {
        Ok(peel) => peel,
        Err(_) => {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "onion_peel_failed");
            return Err(BlindRelayError::OnionPeelFailed);
        }
    };

    match peel.next_hop {
        // Terminal hop: `inner` is either a legacy ChatEnvelope or one signed
        // Blind Vault Put frame. The fixed protocol magic selects the parser;
        // declared-but-malformed Blind Vault bytes never fall back to chat.
        None => {
            // [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] This is the hard
            // terminal ACK boundary for both delivery models. A successful
            // peel is insufficient: chat must reach its pending queue, while a
            // Blind Vault Put must pass its signature, lease, quota, TTL,
            // idempotency, and durable SQLite transaction before we sign ACK.
            let route_purpose = match deliver_onion_terminal_payload(&state, &peel.inner, now).await
            {
                Ok(purpose) => purpose,
                Err(error) => {
                    let failed_at = blind_relay_response_observed_at(now, route_started_at);
                    debug!(
                        reason = error.reason_bucket(),
                        "[BLIND_RELAY] Onion terminal delivery failed"
                    );
                    reject_blind_relay_previous_hop(
                        &state,
                        previous_hop_node_id,
                        failed_at,
                        "onion_terminal_delivery_failed",
                    );
                    return Err(error);
                }
            };

            let accepted_at = blind_relay_response_observed_at(now, route_started_at);
            // [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Sign v2 only after
            // the selected terminal workload has crossed its durable acceptance
            // boundary. The purpose is committed with the opaque payload hash,
            // not returned as relay-visible metadata.
            let delivery_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
                envelope.route_id,
                &peel.inner,
                route_purpose,
                accepted_at,
                state.node_identity.as_ref(),
            );
            let response = PeerBlindRelayResponse {
                accepted: true,
                terminal: true,
                forwarded: false,
                ttl_remaining: envelope.ttl,
                reason: Some("onion_terminal_delivered".to_string()),
                delivery_receipt: Some(delivery_receipt),
            };
            route_lease.complete(accepted_at, response.clone());
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id, accepted_at);
            state.peer_store.record_blind_relay_terminal(
                accepted_at,
                envelope.ttl,
                envelope.encrypted_blob.len(),
            );
            Ok(response)
        }
        // Entry/middle hop: forward the inner layer to the revealed next hop.
        Some(next_hop) => {
            if next_hop == self_node_id || next_hop == previous_hop_node_id {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
                return Err(BlindRelayError::RouteLoop);
            }
            if !is_onion_blob(&peel.inner) {
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "onion_inner_not_layer",
                );
                return Err(BlindRelayError::OnionPeelFailed);
            }
            if !envelope.can_forward() {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
                return Err(BlindRelayError::TtlExhausted);
            }

            let descriptor = state.peer_store.get_valid(&next_hop, now).ok_or_else(|| {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                BlindRelayError::NoRoute
            })?;
            if !descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }
            // True onion routes are the recovery/proof path for a small mesh:
            // after restart a fresh signed peer may not yet have routeability
            // evidence, and refusing the proof attempt creates a deadlock. Keep
            // the hard stop for peers under local route quarantine; forward
            // errors below will still feed route health without logging payloads.
            if state.peer_store.is_route_quarantined_now(&next_hop, now) {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }

            let endpoint = descriptor
                .descriptor
                .public_endpoint
                .as_deref()
                .ok_or_else(|| {
                    reject_blind_relay_previous_hop(
                        &state,
                        previous_hop_node_id,
                        now,
                        "missing_endpoint",
                    );
                    BlindRelayError::InvalidEndpoint
                })?;
            let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
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

            let forward_started_at = blind_relay_response_observed_at(now, route_started_at);
            let next_hop_forward = match forward_blind_relay_with_retry(
                &state,
                &url,
                &descriptor,
                PeerBlindRelayRequest {
                    envelope: forwarded_envelope,
                    previous_hop_node_id: self_node_id,
                    onward_envelope: None,
                    onward_descriptor_hint: None,
                },
                forward_started_at,
            )
            .await
            {
                Ok(ack) => ack,
                Err(error) => return Err(error),
            };
            let observed_at = next_hop_forward.observed_at;
            let next_hop_ack = next_hop_forward.response;

            let response = PeerBlindRelayResponse {
                accepted: true,
                terminal: false,
                forwarded: true,
                ttl_remaining,
                reason: Some("onion_forwarded".to_string()),
                delivery_receipt: next_hop_ack.delivery_receipt,
            };
            route_lease.complete(observed_at, response.clone());
            let _ = state
                .peer_store
                .record_route_forward_success_for_descriptor(&descriptor, observed_at);
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id, observed_at);
            state
                .peer_store
                .record_blind_relay_forwarded(observed_at, ttl_remaining);

            Ok(response)
        }
    }
}

/// Persists one terminal onion payload without widening what middle hops can
/// observe. The terminal necessarily learns the replica-local Blind Vault
/// lease used for storage, but neither the sender/receiver identity nor the
/// ciphertext plaintext exists in the Put frame.
async fn deliver_onion_terminal_payload(
    state: &ChatPeerState,
    payload: &[u8],
    now_secs: u64,
) -> Result<OnionRoutePurpose, BlindRelayError> {
    if is_blind_vault_frame(payload) {
        let frame = decode_blind_vault_frame(payload)
            .map_err(|_| BlindRelayError::OnionTerminalPayloadRejected)?;
        let BlindVaultFrame::Put(request) = frame else {
            // Lease admission, pull, delete, issuer, and response frames retain
            // their dedicated bounded client API. The relay path is append-only.
            return Err(BlindRelayError::OnionTerminalPayloadRejected);
        };
        let vault = state
            .blind_vault
            .as_ref()
            .ok_or(BlindRelayError::ForwardFailed)?;
        vault
            .put(&request, now_secs.saturating_mul(1_000))
            .map_err(|error| map_blind_vault_put_error(&error))?;
        return Ok(OnionRoutePurpose::BlindVaultPut);
    }

    let envelope =
        decode_envelope(payload).map_err(|_| BlindRelayError::OnionTerminalPayloadRejected)?;
    process_peer_relay(state.clone(), envelope)
        .await
        .map(|_| OnionRoutePurpose::MessageRelay)
        .map_err(|_| BlindRelayError::ForwardFailed)
}

fn map_blind_vault_put_error(error: &BlindVaultServiceError) -> BlindRelayError {
    // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] The relay must make a
    // useful retry decision without forwarding replica-local state. Every
    // authorization, signature, lease, and object conflict shares one bucket.
    match error.put_failure_class() {
        BlindVaultPutFailureClass::Rejected => BlindRelayError::OnionTerminalPayloadRejected,
        BlindVaultPutFailureClass::Capacity => BlindRelayError::OnionTerminalCapacityExhausted,
        BlindVaultPutFailureClass::Unavailable => BlindRelayError::ForwardFailed,
    }
}

async fn process_onion_middle_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    outer_envelope: BlindRelayEnvelope,
    onward_envelope: BlindRelayEnvelope,
    onward_descriptor_hint: Option<SignedNodeDescriptor>,
    now: u64,
    route_started_at: &Instant,
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
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let route_lease = match begin_blind_relay_route(
        &state,
        outer_envelope.route_id,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => return Ok(response),
    };

    let forwarded_envelope = onward_envelope
        .decremented_ttl()
        .ok_or(BlindRelayError::TtlExhausted)?
        .sign_with(state.node_identity.as_ref());
    let ttl_remaining = forwarded_envelope.ttl;

    let forward_started_at = blind_relay_response_observed_at(now, route_started_at);
    let next_hop_forward = match forward_blind_relay_with_retry(
        &state,
        &url,
        &descriptor,
        PeerBlindRelayRequest {
            envelope: forwarded_envelope,
            previous_hop_node_id: self_node_id,
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
        forward_started_at,
    )
    .await
    {
        Ok(ack) => ack,
        Err(error) => return Err(error),
    };
    let observed_at = next_hop_forward.observed_at;
    let next_hop_ack = next_hop_forward.response;

    let response = PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining,
        reason: Some("onion_middle_forwarded".to_string()),
        delivery_receipt: next_hop_ack.delivery_receipt,
    };
    route_lease.complete(observed_at, response.clone());
    let _ = state
        .peer_store
        .record_route_forward_success_for_descriptor(&descriptor, observed_at);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id, observed_at);
    state
        .peer_store
        .record_blind_relay_forwarded(observed_at, ttl_remaining);

    Ok(response)
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

fn begin_blind_relay_route(
    state: &ChatPeerState,
    route_id: [u8; 16],
    previous_hop: [u8; 32],
    now: u64,
) -> Result<BlindRelayRouteStart, BlindRelayError> {
    let mut seen_routes = state
        .blind_relay_seen_routes
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let decision = seen_routes.observe(route_id, now);
    drop(seen_routes);

    match decision {
        BlindRelayRouteReplayDecision::New => Ok(BlindRelayRouteStart::Acquired(
            BlindRelayRouteLease::new(Arc::clone(&state.blind_relay_seen_routes), route_id),
        )),
        BlindRelayRouteReplayDecision::InFlight => {
            // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] An unresolved first
            // attempt is not proof of acceptance. Return a retryable status and
            // leave previous-hop health unchanged; a later retry will either
            // replay the durable result or own a fresh attempt after failure.
            state
                .peer_store
                .record_blind_relay_rejected(now, "route_in_flight");
            Err(BlindRelayError::RouteInFlight)
        }
        BlindRelayRouteReplayDecision::Saturated => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "replay_capacity");
            Err(BlindRelayError::ReplayCapacity)
        }
        BlindRelayRouteReplayDecision::Completed(response) => {
            // ACK-loss retries receive the exact bounded success response,
            // including any terminal-signed receipt. No payload is retained.
            state
                .peer_store
                .record_blind_relay_rejected(now, "duplicate_route");
            record_blind_relay_previous_hop_success(state, previous_hop, now);
            Ok(BlindRelayRouteStart::Completed(response))
        }
    }
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

/// Validates the downstream success state and receipt surface visible at this hop.
///
/// [MULTIHOP-RECEIPT-VALIDATION 2026-08-01 by Codex] A direct terminal ACK must
/// be signed by `immediate_next_hop`. A forwarded ACK may carry a receipt from a
/// deeper terminal, so requiring that terminal to equal the immediate next hop
/// incorrectly rejects every path longer than two relay nodes. This hop still
/// verifies the route id, freshness, disposition, and Ed25519 signature. The
/// source must additionally call `verify_expected` with the final payload
/// commitment and terminal selected when it built the onion path.
fn validate_downstream_delivery_receipt(
    ack: &PeerBlindRelayResponse,
    route_id: &[u8; 16],
    immediate_next_hop: &[u8; 32],
    observed_at: u64,
) -> Result<(), &'static str> {
    // [RELAY-ACK-STATE-MACHINE 2026-08-11 by Codex] Treat the response as a
    // protocol state transition, not a collection of independent booleans.
    // A malicious peer previously could return `accepted=true` with neither
    // terminal delivery nor forwarding and still receive route-success credit
    // whenever it omitted the optional legacy receipt. XOR keeps old receipt-
    // less terminal/forwarded ACKs compatible while rejecting false evidence.
    if !ack.accepted {
        return Err("ack_not_accepted");
    }
    if ack.terminal == ack.forwarded {
        return Err("invalid_ack_shape");
    }

    let Some(receipt) = ack.delivery_receipt.as_ref() else {
        // Legacy peers may omit receipts. Keep wire compatibility; callers
        // separately expose that this is not verified client-delivery evidence.
        return Ok(());
    };

    if &receipt.route_id != route_id {
        return Err("receipt_route_mismatch");
    }
    if receipt.delivered_at
        > observed_at.saturating_add(BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS)
    {
        return Err("receipt_timestamp_in_future");
    }
    if observed_at.saturating_sub(receipt.delivered_at) > BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS
    {
        return Err("receipt_timestamp_expired");
    }
    receipt
        .verify_signature()
        .map_err(|_| "receipt_signature_invalid")?;

    if ack.terminal && &receipt.terminal_node_id != immediate_next_hop {
        return Err("terminal_receipt_signer_mismatch");
    }

    Ok(())
}

/// Projects a monotonic request duration onto the caller's Unix timestamp.
///
/// [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Tests and recovery
/// probes inject a stable Unix base. `Instant` supplies elapsed time without
/// depending on wall-clock jumps, while saturation keeps failure handling
/// total even near the integer boundary.
fn blind_relay_response_observed_at(started_at: u64, started: &Instant) -> u64 {
    started_at.saturating_add(started.elapsed().as_secs())
}

async fn forward_blind_relay_with_retry(
    state: &ChatPeerState,
    url: &str,
    descriptor: &SignedNodeDescriptor,
    request: PeerBlindRelayRequest,
    now: u64,
) -> Result<BlindRelayForwardOutcome, BlindRelayError> {
    // [ROUTE-FAILURE-SURFACE-BINDING 2026-08-11 by Codex] Keep the exact
    // descriptor that selected `url` through every retry. A delayed response
    // can then update health only if the signed route surface is still current.
    let next_hop = descriptor.node_id();
    let request_started_at = Instant::now();
    for attempt in 1..=MAX_BLIND_RELAY_FORWARD_ATTEMPTS {
        match state.http_client.post(url).json(&request).send().await {
            Ok(response) if response.status().is_success() => {
                let ack = decode_bounded_json_response::<PeerBlindRelayResponse>(
                    response,
                    PEER_ACK_RESPONSE_MAX_BYTES,
                )
                .await;
                let observed_at = blind_relay_response_observed_at(now, &request_started_at);
                match ack {
                    Ok(ack) if ack.accepted => {
                        if let Err(reason) = validate_downstream_delivery_receipt(
                            &ack,
                            &request.envelope.route_id,
                            &next_hop,
                            observed_at,
                        ) {
                            debug!(
                                attempt,
                                reason,
                                "[BLIND_RELAY] Next-hop delivery receipt verification failed"
                            );
                            let _ = state
                                .peer_store
                                .record_route_forward_failure_for_descriptor(
                                    descriptor,
                                    observed_at,
                                    "delivery_receipt_invalid",
                                );
                            state.peer_store.record_blind_relay_rejected(
                                observed_at,
                                "delivery_receipt_invalid",
                            );
                            return Err(BlindRelayError::ForwardFailed);
                        }
                        if attempt > 1 {
                            state
                                .peer_store
                                .record_blind_relay_retry_succeeded(observed_at, attempt);
                        }
                        return Ok(BlindRelayForwardOutcome {
                            response: ack,
                            observed_at,
                        });
                    }
                    Ok(_ack) => {
                        debug!(
                            attempt,
                            "[BLIND_RELAY] Next-hop ACK rejected opaque relay envelope"
                        );
                    }
                    Err(error) => {
                        debug!(
                            attempt,
                            reason = error.as_str(),
                            "[BLIND_RELAY] Next-hop ACK decode failed"
                        );
                    }
                }

                if attempt > 1 {
                    state.peer_store.record_blind_relay_retry_exhausted(
                        observed_at,
                        attempt,
                        "forward_failed",
                    );
                }
                let _ = state
                    .peer_store
                    .record_route_forward_failure_for_descriptor(
                        descriptor,
                        observed_at,
                        "forward_failed",
                    );
                state
                    .peer_store
                    .record_blind_relay_rejected(observed_at, "forward_failed");
                return Err(BlindRelayError::ForwardFailed);
            }
            Ok(response) => {
                let status = response.status();
                let reason = format!("http_{}", status.as_u16());
                let declared_reason = if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                    decode_bounded_json_response::<PeerBlindRelayResponse>(
                        response,
                        PEER_ACK_RESPONSE_MAX_BYTES,
                    )
                    .await
                    .ok()
                    .and_then(|body| body.reason)
                } else {
                    None
                };
                let observed_at = blind_relay_response_observed_at(now, &request_started_at);
                if let Some(error) =
                    non_retryable_downstream_error(status, declared_reason.as_deref())
                {
                    // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] Preserve
                    // only transport-level retry semantics across middle hops.
                    // Only one bounded stable capacity reason is recognized;
                    // no lease/object detail or arbitrary downstream error is
                    // propagated.
                    debug!(
                        attempt,
                        status = %status,
                        "[BLIND_RELAY] Next-hop returned terminal non-retryable status"
                    );
                    let _ = state
                        .peer_store
                        .record_route_forward_failure_for_descriptor(
                            descriptor,
                            observed_at,
                            error.reason_bucket(),
                        );
                    state
                        .peer_store
                        .record_blind_relay_rejected(observed_at, error.reason_bucket());
                    return Err(error);
                }
                if attempt < MAX_BLIND_RELAY_FORWARD_ATTEMPTS
                    && is_retryable_blind_relay_status(status)
                {
                    state
                        .peer_store
                        .record_blind_relay_retry_attempt(observed_at, &reason);
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
                    state.peer_store.record_blind_relay_retry_exhausted(
                        observed_at,
                        attempt,
                        &reason,
                    );
                }
                let _ = state
                    .peer_store
                    .record_route_forward_failure_for_descriptor(
                        descriptor,
                        observed_at,
                        reason.clone(),
                    );
                state
                    .peer_store
                    .record_blind_relay_rejected(observed_at, reason);
                return Err(BlindRelayError::ForwardFailed);
            }
            Err(error) => {
                let observed_at = blind_relay_response_observed_at(now, &request_started_at);
                let reason = classify_reqwest_error("blind_relay_request", &error);
                if attempt < MAX_BLIND_RELAY_FORWARD_ATTEMPTS && is_retryable_reqwest_error(&error)
                {
                    state
                        .peer_store
                        .record_blind_relay_retry_attempt(observed_at, &reason);
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
                    state.peer_store.record_blind_relay_retry_exhausted(
                        observed_at,
                        attempt,
                        &reason,
                    );
                }
                let _ = state
                    .peer_store
                    .record_route_forward_failure_for_descriptor(
                        descriptor,
                        observed_at,
                        reason.clone(),
                    );
                state
                    .peer_store
                    .record_blind_relay_rejected(observed_at, reason);
                return Err(BlindRelayError::ForwardFailed);
            }
        }
    }

    Err(BlindRelayError::ForwardFailed)
}

fn non_retryable_downstream_error(
    status: reqwest::StatusCode,
    declared_reason: Option<&str>,
) -> Option<BlindRelayError> {
    if status == reqwest::StatusCode::SERVICE_UNAVAILABLE
        && declared_reason == Some("onion_terminal_capacity_exhausted")
    {
        // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] Only the exact bounded
        // peer error contract marks deterministic capacity. An unknown proxy
        // 503 remains retryable instead of being misclassified as lease state.
        return Some(BlindRelayError::OnionTerminalCapacityExhausted);
    }
    if status.is_client_error() && status != reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Some(BlindRelayError::DownstreamRejected);
    }
    None
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
    // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] A next-hop descriptor is
    // permissionless input. Its signature cannot authorize localhost, private
    // networks, metadata services, DNS rebinding, or URL-controlled paths.
    if !peer_endpoint_is_public_ip(endpoint) {
        #[cfg(not(test))]
        return None;
        #[cfg(test)]
        if !crate::api::peer_endpoint_is_loopback_ip(endpoint) {
            return None;
        }
    }
    canonical_peer_http_url(endpoint, "/api/chat/peer/blind-relay")
        .ok()
        .map(|url| url.to_string())
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
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultAdmissionTicket, BlindVaultLeaseAdmissionRequest,
    };
    use aeronyx_core::protocol::chat::ChatContentType;
    use aeronyx_core::protocol::{
        encode_blind_vault_frame, BlindVaultFrame, BlindVaultLeaseCreateRequest,
        BlindVaultPutRequest, NodeCapability, NodeCapacity, NodeDescriptor, SignedNodeDescriptor,
    };
    use aeronyx_transport::UdpTransport;
    use axum::body::Body;
    use axum::http::Request;
    use axum::response::IntoResponse;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tokio::net::TcpListener;
    use tower::ServiceExt;

    use sha2::{Digest, Sha256};

    use crate::config::{BlindVaultConfig, ChatRelayConfig};
    use crate::services::{BlindVaultLeaseProvisionOutcome, BlindVaultService};

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

    #[test]
    fn delayed_receipt_validation_uses_response_observation_time() {
        // [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] A valid ACK
        // produced after a long request must be compared with response time,
        // not the stale ingress timestamp. Backdating only the monotonic test
        // clock keeps this regression deterministic and fast.
        let started_at = 1_800_000_000u64;
        let request_started_at = Instant::now() - Duration::from_secs(31);
        let observed_at = blind_relay_response_observed_at(started_at, &request_started_at);
        assert!(observed_at >= started_at + 31);

        let terminal = IdentityKeyPair::generate();
        let terminal_node_id = terminal.public_key_bytes();
        let route_id = [0x81u8; 16];
        let payload = b"opaque delayed message relay payload";
        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted_for_purpose(
                route_id,
                payload,
                OnionRoutePurpose::MessageRelay,
                started_at + 31,
                &terminal,
            )),
        };

        assert_eq!(
            validate_downstream_delivery_receipt(&ack, &route_id, &terminal_node_id, started_at,),
            Err("receipt_timestamp_in_future")
        );
        assert!(validate_downstream_delivery_receipt(
            &ack,
            &route_id,
            &terminal_node_id,
            observed_at,
        )
        .is_ok());
    }

    #[test]
    fn three_hop_forwarded_ack_accepts_downstream_terminal_receipt() {
        let now = 1_800_000_100;
        let route_id = [0xa1; 16];
        let immediate_middle = IdentityKeyPair::generate();
        let downstream_terminal = IdentityKeyPair::generate();
        assert_ne!(
            immediate_middle.public_key_bytes(),
            downstream_terminal.public_key_bytes()
        );

        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("onion_forwarded".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
                route_id,
                [0xb2; 32],
                now,
                &downstream_terminal,
            )),
        };

        validate_downstream_delivery_receipt(
            &ack,
            &route_id,
            &immediate_middle.public_key_bytes(),
            now,
        )
        .expect("an intermediate ACK may propagate a deeper terminal receipt");
    }

    #[test]
    fn direct_terminal_ack_requires_immediate_next_hop_receipt_signer() {
        let now = 1_800_000_100;
        let route_id = [0xc3; 16];
        let immediate_terminal = IdentityKeyPair::generate();
        let wrong_terminal = IdentityKeyPair::generate();
        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
                route_id,
                [0xd4; 32],
                now,
                &wrong_terminal,
            )),
        };

        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack,
                &route_id,
                &immediate_terminal.public_key_bytes(),
                now,
            ),
            Err("terminal_receipt_signer_mismatch")
        );
    }

    #[test]
    fn downstream_success_ack_requires_exactly_one_delivery_disposition() {
        // [RELAY-ACK-STATE-MACHINE 2026-08-11 by Codex] Receipt-less ACKs stay
        // valid for mixed-version peers only when their state proves one real
        // terminal or forwarding action. No-op and contradictory success
        // shapes must fail before they can become route-health evidence.
        let now = 1_800_000_100;
        let route_id = [0xd5; 16];
        let immediate_next_hop = IdentityKeyPair::generate().public_key_bytes();
        let ack = |accepted, terminal, forwarded| PeerBlindRelayResponse {
            accepted,
            terminal,
            forwarded,
            ttl_remaining: 1,
            reason: None,
            delivery_receipt: None,
        };

        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack(true, false, false),
                &route_id,
                &immediate_next_hop,
                now,
            ),
            Err("invalid_ack_shape")
        );
        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack(true, true, true),
                &route_id,
                &immediate_next_hop,
                now,
            ),
            Err("invalid_ack_shape")
        );
        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack(false, false, false),
                &route_id,
                &immediate_next_hop,
                now,
            ),
            Err("ack_not_accepted")
        );
        assert!(validate_downstream_delivery_receipt(
            &ack(true, true, false),
            &route_id,
            &immediate_next_hop,
            now,
        )
        .is_ok());
        assert!(validate_downstream_delivery_receipt(
            &ack(true, false, true),
            &route_id,
            &immediate_next_hop,
            now,
        )
        .is_ok());
    }

    #[test]
    fn forwarded_ack_rejects_tampered_downstream_receipt() {
        let now = 1_800_000_100;
        let route_id = [0xe5; 16];
        let immediate_middle = IdentityKeyPair::generate();
        let downstream_terminal = IdentityKeyPair::generate();
        let mut receipt =
            BlindRelayDeliveryReceipt::accepted(route_id, [0xf6; 32], now, &downstream_terminal);
        receipt.payload_commitment[0] ^= 0xff;
        let ack = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("onion_forwarded".to_string()),
            delivery_receipt: Some(receipt),
        };

        assert_eq!(
            validate_downstream_delivery_receipt(
                &ack,
                &route_id,
                &immediate_middle.public_key_bytes(),
                now,
            ),
            Err("receipt_signature_invalid")
        );
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

    fn temp_blind_vault_with_put(
        node_identity: &IdentityKeyPair,
        now_ms: u64,
    ) -> (
        tempfile::TempDir,
        Arc<BlindVaultService>,
        BlindVaultPutRequest,
    ) {
        // [BLIND-VAULT-ONION-DISPATCH 2026-08-10 by Codex] Use the production
        // admission and mutation pipeline in relay tests. Bypassing lease
        // provisioning would miss signature, quota, expiry, and authority
        // regressions at the protocol boundary this feature is meant to join.
        let directory = tempfile::tempdir().expect("blind vault temp directory");
        let issuer = IdentityKeyPair::generate();
        let config = BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            admission_issuer_public_keys: vec![hex::encode(issuer.public_key_bytes())],
            db_path: directory
                .path()
                .join("blind-vault.db")
                .display()
                .to_string(),
            ..BlindVaultConfig::default()
        };
        let service = Arc::new(
            BlindVaultService::new(config, node_identity.clone()).expect("blind vault service"),
        );
        let write_key = IdentityKeyPair::generate();
        let admin_key = IdentityKeyPair::generate();
        let lease_id = [0x61; 32];
        let mut lease = BlindVaultLeaseCreateRequest::new(
            lease_id,
            [0x62; 16],
            write_key.public_key_bytes(),
            admin_key.public_key_bytes(),
            Sha256::digest([0x63; 32]).into(),
            now_ms + 24 * 60 * 60 * 1_000,
        );
        lease.sign(&admin_key).expect("sign anonymous lease");
        let mut admission = BlindVaultAdmissionTicket::new(
            [0x64; 32],
            issuer.public_key_bytes(),
            now_ms.saturating_sub(1_000),
            now_ms + 60 * 60 * 1_000,
            2 * 24 * 60 * 60 * 1_000,
        );
        admission.sign(&issuer).expect("sign admission ticket");
        assert_eq!(
            service
                .provision_lease_with_admission(
                    &BlindVaultLeaseAdmissionRequest { admission, lease },
                    now_ms,
                )
                .expect("provision anonymous lease"),
            BlindVaultLeaseProvisionOutcome::Created
        );

        let mut put = BlindVaultPutRequest::new(
            lease_id,
            [0x65; 32],
            [0x66; 16],
            vec![0xa5; 4 * 1024],
            now_ms + 60 * 60 * 1_000,
        );
        put.sign(&write_key);
        (directory, service, put)
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
            source.contains("reason = error.as_str()"),
            "ACK decode failures should use bounded stable reason buckets"
        );
        assert!(
            source.contains("let reason = error.reason_bucket()"),
            "store failures should use service-owned stable reason buckets"
        );
    }

    #[test]
    fn pending_store_capacity_maps_to_retryable_service_unavailable() {
        let capacity = ChatRelayError::PendingMessageQueueFull {
            current: 100,
            limit: 100,
        };
        let mapped = map_pending_store_error(&capacity);

        assert!(matches!(mapped, ChatPeerRelayError::PendingCapacity));
        assert_eq!(mapped.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(mapped.reason_bucket(), "pending_capacity_exhausted");

        let storage = ChatRelayError::Sqlite(rusqlite::Error::InvalidQuery);
        assert!(matches!(
            map_pending_store_error(&storage),
            ChatPeerRelayError::StoreFailed
        ));

        let oversized = ChatRelayError::MessageTooLarge {
            size: 65_537,
            limit: 65_536,
        };
        assert!(matches!(
            map_pending_store_error(&oversized),
            ChatPeerRelayError::EnvelopeTooLarge { size: 65_537 }
        ));
    }

    #[test]
    fn blind_vault_put_errors_preserve_retry_semantics_without_state_details() {
        assert!(matches!(
            map_blind_vault_put_error(&BlindVaultServiceError::LeaseNotFound),
            BlindRelayError::OnionTerminalPayloadRejected
        ));
        assert!(matches!(
            map_blind_vault_put_error(&BlindVaultServiceError::QuotaExceeded),
            BlindRelayError::OnionTerminalCapacityExhausted
        ));
        assert!(matches!(
            map_blind_vault_put_error(&BlindVaultServiceError::Disabled),
            BlindRelayError::ForwardFailed
        ));
        assert_eq!(
            BlindRelayError::OnionTerminalCapacityExhausted.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            BlindRelayError::RouteInFlight.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            BlindRelayError::RouteInFlight.reason_bucket(),
            "route_in_flight"
        );
        assert_eq!(
            BlindRelayError::ReplayCapacity.status_code(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            BlindRelayError::ReplayCapacity.reason_bucket(),
            "replay_capacity"
        );
        assert!(matches!(
            non_retryable_downstream_error(reqwest::StatusCode::BAD_REQUEST, None),
            Some(BlindRelayError::DownstreamRejected)
        ));
        assert!(matches!(
            non_retryable_downstream_error(
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                Some("onion_terminal_capacity_exhausted")
            ),
            Some(BlindRelayError::OnionTerminalCapacityExhausted)
        ));
        assert!(non_retryable_downstream_error(
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            Some("proxy_unavailable")
        )
        .is_none());
        assert!(
            non_retryable_downstream_error(reqwest::StatusCode::TOO_MANY_REQUESTS, None).is_none()
        );
        assert!(non_retryable_downstream_error(reqwest::StatusCode::BAD_GATEWAY, None).is_none());
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
            None,
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
    async fn peer_routes_reject_oversized_bodies_before_json_deserialization() {
        let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
        let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
        let peer_store = Arc::new(PeerStore::new());
        let app = build_chat_peer_router(
            None,
            sessions,
            udp,
            Arc::clone(&peer_store),
            Arc::new(IdentityKeyPair::generate()),
            Arc::new(reqwest::Client::new()),
            None,
        );

        let peer_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/relay")
                    .header("content-type", "application/json")
                    .body(Body::from(vec![b' '; PEER_CHAT_REQUEST_BODY_MAX_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        let blind_response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from(vec![
                        b' ';
                        PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES + 1
                    ]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(peer_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(blind_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let blind_stats = peer_store.status(now_secs()).runtime.blind_relay;
        assert_eq!(blind_stats.received, 0, "oversized body reached handler");
        assert_eq!(blind_stats.rejected, 0, "oversized body reached handler");
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
            None,
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
            blind_vault: None,
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
            blind_vault: None,
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
            blind_vault: None,
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
        result
            .delivery_receipt
            .as_ref()
            .expect("terminal onion delivery must return a signed receipt")
            .verify_expected_for_purpose(
                &[0x55u8; 16],
                &inner,
                OnionRoutePurpose::MessageRelay,
                &node_identity.public_key_bytes(),
            )
            .expect("receipt must bind the exact terminal payload, purpose, and node");
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
    async fn onion_terminal_persists_anonymous_blind_vault_put_idempotently() {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

        let source = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let now = now_secs();
        let now_ms = now.saturating_mul(1_000);
        let (_directory, vault, put) = temp_blind_vault_with_put(node_identity.as_ref(), now_ms);
        let encoded_put =
            encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode vault put");
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: Some(Arc::clone(&vault)),
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };

        let make_envelope = |route_id| {
            build_onion_envelope(
                &[OnionHop {
                    node_id: node_identity.public_key_bytes(),
                    kem_pub: crate::services::onion_keys::current_public_key(),
                }],
                &encoded_put,
                route_id,
                4,
                now,
                &source,
            )
            .expect("build vault terminal onion")
        };
        let first_route = [0x67; 16];
        let result = process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: make_envelope(first_route),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("signed anonymous put should reach Blind Vault");

        assert!(result.accepted);
        assert!(result.terminal);
        assert!(!result.forwarded);
        assert_eq!(result.reason.as_deref(), Some("onion_terminal_delivered"));
        result
            .delivery_receipt
            .as_ref()
            .expect("vault acceptance must return a route-safe terminal receipt")
            .verify_expected_for_purpose(
                &first_route,
                &encoded_put,
                OnionRoutePurpose::BlindVaultPut,
                &node_identity.public_key_bytes(),
            )
            .expect("receipt must bind exact encoded put and purpose without exposing metadata");
        let public_ack = serde_json::to_string(&result).expect("serialize terminal ACK");
        assert!(!public_ack.contains("lease_id"));
        assert!(!public_ack.contains("object_id"));
        assert!(!public_ack.contains("ciphertext"));

        let status = vault.status(now_ms + 1).expect("vault status after put");
        assert_eq!(status.live_objects, 1);
        assert_eq!(status.live_ciphertext_bytes, 4 * 1024);

        // A source may rebuild a route after losing the first ACK. Blind Vault
        // handles the exact Put idempotently even when the relay route differs.
        process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: make_envelope([0x68; 16]),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .expect("same immutable put through a fresh route should be idempotent");
        assert_eq!(
            vault
                .status(now_ms + 2)
                .expect("vault status after retry")
                .live_objects,
            1
        );

        assert!(matches!(
            deliver_onion_terminal_payload(&state, b"ANBV", now).await,
            Err(BlindRelayError::OnionTerminalPayloadRejected)
        ));
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
            blind_vault: None,
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
            blind_vault: None,
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
            blind_vault: None,
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
                        delivery_receipt: None,
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
            blind_vault: None,
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
        let terminal_receipt_identity = terminal_identity.clone();
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
                let terminal_receipt_identity = terminal_receipt_identity.clone();
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
                    let delivery_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
                        request.envelope.route_id,
                        &peel.inner,
                        OnionRoutePurpose::MessageRelay,
                        now_secs(),
                        &terminal_receipt_identity,
                    );
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
                        delivery_receipt: Some(delivery_receipt),
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
            blind_vault: None,
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
        response
            .delivery_receipt
            .as_ref()
            .expect("middle hop must propagate the terminal receipt")
            .verify_expected_for_purpose(
                &route_id,
                &encoded_chat,
                OnionRoutePurpose::MessageRelay,
                &terminal_node_id,
            )
            .expect("propagated receipt must retain terminal purpose binding");

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
    async fn peer_request_in_flight_guard_enforces_backpressure_limit() {
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store,
            node_identity: Arc::new(IdentityKeyPair::generate()),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS)),
            blind_relay_seen_routes: Arc::new(Mutex::new(BlindRelayRouteReplayCache::default())),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };

        assert!(InFlightRequestGuard::try_acquire(
            &state.blind_relay_in_flight,
            MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS,
        )
        .is_none());

        let app = Router::new()
            .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
            .route_layer(middleware::from_fn_with_state(
                state.clone(),
                peer_blind_relay_request_gate,
            ))
            .with_state(state);
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat/peer/blind-relay")
                    .header("content-type", "application/json")
                    .body(Body::from("not-json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn blind_relay_rejects_immediate_previous_hop_loop_without_parsing_blob() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
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

        assert_eq!(
            cache.observe(route_id, 1_800_000_000),
            BlindRelayRouteReplayDecision::New
        );
        assert_eq!(
            cache.observe(route_id, 1_800_000_001),
            BlindRelayRouteReplayDecision::InFlight
        );
        cache.forget(&route_id);
        assert_eq!(
            cache.observe(route_id, 1_800_000_002),
            BlindRelayRouteReplayDecision::New
        );
    }

    #[test]
    fn blind_relay_route_lease_releases_cancellation_and_commits_exact_ack() {
        // [RELAY-ROUTE-RAII 2026-08-11 by Codex] Dropping an async request
        // owner must release its in-flight claim. A successful owner must
        // instead publish the exact bounded response for ACK-loss replay.
        let seen_routes = Arc::new(Mutex::new(BlindRelayRouteReplayCache::default()));
        let route_id = [0x48u8; 16];
        let started_at = 1_800_000_000;

        assert_eq!(
            seen_routes.lock().unwrap().observe(route_id, started_at),
            BlindRelayRouteReplayDecision::New
        );
        drop(BlindRelayRouteLease::new(
            Arc::clone(&seen_routes),
            route_id,
        ));
        assert_eq!(
            seen_routes
                .lock()
                .unwrap()
                .observe(route_id, started_at + 1),
            BlindRelayRouteReplayDecision::New
        );

        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("forwarded".to_string()),
            delivery_receipt: None,
        };
        BlindRelayRouteLease::new(Arc::clone(&seen_routes), route_id)
            .complete(started_at + 2, response.clone());
        assert_eq!(
            seen_routes
                .lock()
                .unwrap()
                .observe(route_id, started_at + 3),
            BlindRelayRouteReplayDecision::Completed(response)
        );
    }

    #[test]
    fn blind_relay_replay_cache_eviction_preserves_in_flight_generations() {
        // [DURABLE-TERMINAL-REPLAY-WINDOW 2026-08-11 by Codex] A failed route
        // leaves a stale queue generation after `forget`. Reusing that route id
        // must not let capacity eviction remove its newer live generation.
        // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] Capacity pressure evicts
        // the oldest completed ACK instead of either unresolved route.
        let mut cache = BlindRelayRouteReplayCache::default();
        let reused_route = [0x91u8; 16];
        let older_live_route = [0x92u8; 16];
        let now = 1_800_000_000;
        let completed_response = PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining: 1,
            reason: Some("forwarded".to_string()),
            delivery_receipt: None,
        };

        assert_eq!(
            cache.observe(reused_route, now),
            BlindRelayRouteReplayDecision::New
        );
        assert_eq!(
            cache.observe(older_live_route, now),
            BlindRelayRouteReplayDecision::New
        );
        cache.forget(&reused_route);
        assert_eq!(
            cache.observe(reused_route, now),
            BlindRelayRouteReplayDecision::New
        );

        for sequence in 0..MAX_BLIND_RELAY_SEEN_ROUTES.saturating_sub(1) {
            let mut route_id = [0x93u8; 16];
            route_id[..8].copy_from_slice(&(sequence as u64).to_be_bytes());
            assert_eq!(
                cache.observe(route_id, now),
                BlindRelayRouteReplayDecision::New
            );
            cache.complete(&route_id, now, completed_response.clone());
        }

        assert_eq!(
            cache.observe(reused_route, now),
            BlindRelayRouteReplayDecision::InFlight
        );
        assert_eq!(
            cache.observe(older_live_route, now),
            BlindRelayRouteReplayDecision::InFlight
        );
        let mut oldest_completed_route = [0x93u8; 16];
        oldest_completed_route[..8].copy_from_slice(&0u64.to_be_bytes());
        assert_eq!(
            cache.observe(oldest_completed_route, now),
            BlindRelayRouteReplayDecision::New
        );
    }

    #[test]
    fn blind_relay_replay_cache_fails_closed_when_all_routes_are_in_flight() {
        let mut cache = BlindRelayRouteReplayCache::default();
        let now = 1_800_000_000;

        for sequence in 0..MAX_BLIND_RELAY_SEEN_ROUTES {
            let mut route_id = [0x97u8; 16];
            route_id[..8].copy_from_slice(&(sequence as u64).to_be_bytes());
            assert_eq!(
                cache.observe(route_id, now),
                BlindRelayRouteReplayDecision::New
            );
        }

        let saturated_route = [0x98u8; 16];
        assert_eq!(
            cache.observe(saturated_route, now),
            BlindRelayRouteReplayDecision::Saturated
        );
        assert_eq!(cache.seen.len(), MAX_BLIND_RELAY_SEEN_ROUTES);

        let mut first_route = [0x97u8; 16];
        first_route[..8].copy_from_slice(&0u64.to_be_bytes());
        assert_eq!(
            cache.observe(first_route, now),
            BlindRelayRouteReplayDecision::InFlight
        );
        cache.forget(&first_route);
        assert_eq!(
            cache.observe(saturated_route, now),
            BlindRelayRouteReplayDecision::New
        );
    }

    #[test]
    fn blind_relay_replay_cache_bounds_same_second_failed_generations() {
        // [REPLAY-GENERATION-COMPACTION 2026-08-11 by Codex] Second-resolution
        // timestamps cannot distinguish a failed attempt from its immediate
        // retry. Unique generations preserve the retry while bounded
        // compaction prevents an active queue prefix from retaining unlimited
        // stale attempts.
        let mut cache = BlindRelayRouteReplayCache::default();
        let live_route = [0x95u8; 16];
        let retried_route = [0x96u8; 16];
        let now = 1_800_000_000;

        assert_eq!(
            cache.observe(live_route, now),
            BlindRelayRouteReplayDecision::New
        );
        for _ in 0..=MAX_BLIND_RELAY_REPLAY_QUEUE_GENERATIONS {
            assert_eq!(
                cache.observe(retried_route, now),
                BlindRelayRouteReplayDecision::New
            );
            cache.forget(&retried_route);
        }

        assert!(cache.order.len() <= MAX_BLIND_RELAY_REPLAY_QUEUE_GENERATIONS);
        assert_eq!(
            cache.observe(live_route, now),
            BlindRelayRouteReplayDecision::InFlight
        );
        assert_eq!(
            cache.observe(retried_route, now),
            BlindRelayRouteReplayDecision::New
        );
        assert_eq!(
            cache.observe(retried_route, now),
            BlindRelayRouteReplayDecision::InFlight
        );
    }

    #[test]
    fn blind_relay_replay_cache_starts_window_at_route_completion() {
        let mut cache = BlindRelayRouteReplayCache::default();
        let route_id = [0x94u8; 16];
        let started_at = 1_800_000_000;
        let completed_at = started_at + BLIND_RELAY_ROUTE_REPLAY_WINDOW_SECS;
        let terminal_identity = IdentityKeyPair::generate();
        let delivery_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
            route_id,
            b"opaque terminal payload",
            OnionRoutePurpose::MessageRelay,
            completed_at,
            &terminal_identity,
        );

        let response = PeerBlindRelayResponse {
            accepted: true,
            terminal: true,
            forwarded: false,
            ttl_remaining: 1,
            reason: Some("onion_terminal_delivered".to_string()),
            delivery_receipt: Some(delivery_receipt),
        };

        assert_eq!(
            cache.observe(route_id, started_at),
            BlindRelayRouteReplayDecision::New
        );
        cache.complete(&route_id, completed_at, response.clone());
        assert_eq!(
            cache.observe(route_id, completed_at + 1),
            BlindRelayRouteReplayDecision::Completed(response)
        );
        assert_eq!(
            cache.observe(
                route_id,
                completed_at + BLIND_RELAY_ROUTE_REPLAY_WINDOW_SECS + 1,
            ),
            BlindRelayRouteReplayDecision::New
        );
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
    async fn blind_relay_replays_completed_response_without_forwarding_again() {
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
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
        assert_eq!(duplicate, first);
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
    async fn blind_relay_in_flight_duplicate_never_returns_false_acceptance() {
        // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] A concurrent retry must
        // remain retryable until the owner attempt publishes a durable result.
        // Returning an accepted replay here could lose the route if that owner
        // subsequently fails.
        let previous_hop = IdentityKeyPair::generate();
        let node_identity = Arc::new(IdentityKeyPair::generate());
        let route_id = [0x47u8; 16];
        let mut replay_cache = BlindRelayRouteReplayCache::default();
        assert_eq!(
            replay_cache.observe(route_id, now_secs()),
            BlindRelayRouteReplayDecision::New
        );
        let peer_store = Arc::new(PeerStore::new());
        let state = ChatPeerState {
            chat_relay: None,
            blind_vault: None,
            sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
            udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
            peer_store: Arc::clone(&peer_store),
            node_identity: Arc::clone(&node_identity),
            http_client: Arc::new(reqwest::Client::new()),
            blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
            blind_relay_seen_routes: Arc::new(Mutex::new(replay_cache)),
            blind_relay_abuse_guard: Arc::new(Mutex::new(BlindRelayAbuseGuard::default())),
        };
        let envelope = BlindRelayEnvelope {
            route_id,
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque concurrent replay candidate".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop);

        let error = process_peer_blind_relay(
            state,
            PeerBlindRelayRequest {
                envelope,
                previous_hop_node_id: previous_hop.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(matches!(error, BlindRelayError::RouteInFlight));
        let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
        assert_eq!(blind_stats.terminal, 0);
        assert_eq!(blind_stats.forwarded, 0);
        assert_eq!(blind_stats.rejected, 1);
        assert!(peer_store.recent_audit_events().iter().any(|event| {
            event.action == "blind_relay_forward"
                && event.outcome == "rejected"
                && event.detail == "route_in_flight"
        }));
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
                            delivery_receipt: None,
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
            blind_vault: None,
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
                        delivery_receipt: None,
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
            blind_vault: None,
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
                        delivery_receipt: None,
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
            blind_vault: None,
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
            blind_vault: None,
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
            blind_vault: None,
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
                        delivery_receipt: None,
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
            blind_vault: None,
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
        // [ROUTE-HEALTH-REMOTE-POISONING 2026-08-11 by Codex] This request was
        // rejected before any outbound attempt. It must not let an untrusted
        // previous hop manufacture failure or quarantine evidence for a peer.
        assert_eq!(route_row.routeability_state, "unknown");
        assert!(!route_row.routeability_ready);
        assert_eq!(route_row.route_failure_count, 0);
        assert_eq!(route_row.route_consecutive_failures, 0);
        assert!(route_row.last_route_failure_reason.is_none());
        assert!(!route_row.route_quarantined);
        assert!(peer_store.recent_audit_events().iter().all(|event| {
            event.action != "blind_relay_route_health"
                || !event.detail.contains("reason=routeability_not_ready")
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
            blind_vault: None,
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
                        delivery_receipt: None,
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
            blind_vault: None,
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
