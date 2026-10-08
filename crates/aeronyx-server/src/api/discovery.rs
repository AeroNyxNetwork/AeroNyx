// ============================================================================
// File: crates/aeronyx-server/src/api/discovery.rs
// ============================================================================
//! # Discovery API
//!
//! ## Creation Reason
//! Exposes a minimal HTTP entry point for decentralized AeroNyx node discovery
//! so nodes can exchange signed descriptors without relying on the centralized
//! management backend.
//!
//! ## Main Functionality
//! - `POST /api/discovery/join`: accepts one canonical binary self-signed node
//!   descriptor into a bounded, non-routeable permissionless candidate lane
//! - `GET /api/discovery/snapshot`: returns a JSON bootstrap snapshot of
//!   verified descriptors from the local `PeerStore`
//! - `POST /api/discovery/gossip`: accepts a JSON `NodeDiscoveryMessage`,
//!   applies descriptor/snapshot updates, verifies proof announcements against
//!   an audited local Directory replica, and returns a snapshot response for
//!   request messages
//! - `GET /api/discovery/status`: returns aggregate peer-store status, local
//!   capability readiness, and compact discovery readiness for dashboards
//! - `GET /api/discovery/summary`: returns a compact public-safe protocol
//!   foundation summary for app, website, backend aggregation, and AI runbooks,
//!   including aggregate route-governance readiness and non-authoritative
//!   transport feature negotiation without route metadata
//! - [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] Publishes one privacy-safe
//!   recovery-anchor aggregate and requires an external witness to protect the
//!   exact cache generation before restart continuity can become ready.
//! - `GET /api/discovery/public-card`: returns the smallest product-facing
//!   protocol health card for website, Nodeboard first-level views, and apps
//! - [ONION-CANDIDATE-PROOF 2026-07-31 by Codex] Returns each onion candidate's
//!   original signed node descriptor so App/SDK path builders can independently
//!   verify identity, capability, endpoint, capacity, and rotating KEM metadata
//! - [DISCOVERY-RATE-LIMIT-RECOVERY 2026-07-30 by Codex] Keeps permissionless
//!   gossip admission usable after an unrelated panic while the process-local
//!   rate-limit lock is held.
//! - [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex] Publishes independent,
//!   aggregate-only three-hop runtime proof maturity without selected routes.
//! - [THREE-HOP-FEATURE-NEGOTIATION 2026-08-02 by Codex] Advertises whether
//!   this runtime can validate a terminal delivery receipt propagated through
//!   more than one middle relay, allowing safe mixed-version probe selection.
//! - [ONION-PATH-ADMISSION 2026-08-02 by Codex] Fails closed when a requested
//!   multi-hop path has enough candidates but lacks stable runtime proof or
//!   restart-continuity evidence, with an explicit lower-hop fallback.
//! - [ONION-CAPABILITY-GATE 2026-08-02 by Codex] Requires every public onion
//!   candidate to advertise both `ChatRelay` and `OnionMiddle`, preventing a
//!   single-hop-only relay from being counted toward a multi-hop route.
//! - [ONION-NETWORK-DIVERSITY 2026-08-03 by Codex] Requires a pairwise
//!   network-diverse candidate subset before multi-hop admission can become
//!   ready, reusing the Rust path planner's fail-closed endpoint policy.
//! - [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] Preserves a lower-ranked
//!   network-diverse subset before applying a small client response limit, so
//!   healthier collocated relays cannot hide an otherwise valid onion path.
//! - [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] Production routers inject
//!   the local node id and exclude candidates sharing the entry node's coarse
//!   endpoint network identity before multi-hop readiness is evaluated.
//! - [ROUTE-DOMAIN-CERTIFICATE-INGRESS 2026-08-03 by Codex] Accepts a tightly
//!   bounded portable certificate frame from any transport sender, then admits
//!   it only under the node's exact host-local subject/domain pins and pinned
//!   independent-attestor quorum.
//! - [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Separates ordinary encrypted
//!   message terminals from admitted Blind Vault ciphertext replicas without
//!   changing the legacy candidate query or exposing a selected route.
//! - [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Advertises v2
//!   purpose-bound terminal receipt support as an unsigned bootstrap hint;
//!   route authority still requires a fresh cryptographically verified receipt.
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/discovery.rs: message and snapshot types
//! - aeronyx-server/src/services/peer_store.rs: verification, anti-rollback,
//!   and snapshot export logic
//! - axum: router and JSON extraction/response
//!
//! ## Main Logical Flow
//! 1. Snapshot requests read valid descriptors from `PeerStore`
//! 2. Gossip messages are applied through `PeerStore::apply_discovery_message`
//! 3. Incoming data never bypasses descriptor signature verification
//! 4. Directory-proof gossip additionally requires exact local replica evidence
//! 5. Response reports import counts and optionally includes a snapshot response
//!
//! ## Important Note for Next Developer
//! - Do not add client public IPs, packet payloads, destinations, DNS contents,
//!   domains, URLs, browsing history, voucher secrets, private keys, or
//!   wallet-level traffic to these endpoints.
//! - This API exchanges only signed node descriptors and aggregate import
//!   counts. It is not an encrypted message relay endpoint.
//! - Public exit remains disabled by default at descriptor policy level.
//! - Security decisions are recorded as privacy-safe aggregate audit events in
//!   `PeerStoreStatus.recent_audit_events`.
//! - `DiscoveryLocalCapabilityStatus` reports only local configuration,
//!   runtime service readiness, and endpoint readiness; it must not include
//!   node ids, route ids, client data, peer endpoints, payloads, or
//!   wallet-level information.
//! - `discovery_readiness_status_value()` is the shared compact status contract
//!   used by both public/local discovery status and backend heartbeat reports.
//! - `DiscoverySummaryResponse` is intentionally smaller than
//!   `DiscoveryStatusResponse`; keep it aggregate-only so public/product
//!   surfaces never need to parse full peer diagnostics.
//! - `DiscoveryPublicCardResponse` is smaller again. It is the contract for
//!   top-level UX surfaces that should show confidence and readiness, not raw
//!   engineering diagnostics.
//! - The gossip request body ceiling must remain outside the JSON handler so
//!   oversized untrusted input is rejected before allocation/deserialization.
//!   Keep `DISCOVERY_REQUEST_BODY_MAX_BYTES` aligned with protocol limits.
//! - The global gossip limiter must use a non-poisoning mutex. A poisoned
//!   process-local lock must never turn one recovered panic into a permanent
//!   discovery outage.
//! - Protocol feature fields are unsigned compatibility hints only. They may
//!   suppress an optional probe, but must never grant route trust or replace
//!   terminal signature and payload-commitment verification.
//! - `multihop_delivery_receipt_v1` means the node understands propagated
//!   receipt framing. `purpose_bound_delivery_receipt_v2` means current
//!   terminals sign workload-separated commitments. Never infer v2 from v1.
//! - Candidate count alone must never make a multi-hop path ready. Keep
//!   `requested_path_ready` gated by the matching two-hop or three-hop runtime
//!   proof and signed restart-continuity decision.
//! - An onion candidate must advertise both `ChatRelay` and `OnionMiddle` in
//!   its verified signed descriptor. Apply client limits only after capability,
//!   routeability, KEM, and endpoint filtering so valid lower-ranked relays are
//!   not hidden by ineligible peers.
//! - Candidate count is not network diversity. Multi-hop admission must find a
//!   pairwise-diverse subset using the same coarse IPv4 /24, IPv6 /48, and DNS
//!   hostname policy as the internal Rust path planner. This does not prove
//!   distinct operators or autonomous systems.
//! - Apply a client response limit only after preserving a network-diverse
//!   requested-hop subset (or a safe two-hop fallback). This endpoint prepares
//!   an eligible pool; the client still chooses the actual weighted-random
//!   route and must independently verify every signed descriptor.
//! - Production callers must use `build_discovery_router_with_local_entry` so
//!   candidate anti-affinity includes the entry node itself. Legacy builders
//!   remain for compatibility and explicitly report that this gate is absent.
//! - Route-domain certificate transport is permissionless but not trusted:
//!   signatures, exact local subject/domain pins, expiry, and local quorum are
//!   the authority. Never log or return subject ids, attestors, domain tokens,
//!   signatures, certificate hashes, or selected routes.
//! - A purpose-aware response is only a candidate contract. For
//!   `blind_vault_put`, at least one candidate in the complete diverse subset
//!   must carry `BlindVaultReplica` in its original signed descriptor. Never
//!   infer terminal eligibility from a flattened JSON field alone.
//! - [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Purpose parsing and specialized
//!   terminal capability semantics live in `aeronyx-core`. This server owns
//!   only live admission policy and must not fork the shared wire contract.
//! - External witness status is generation-bound. A `verified` result from an
//!   older cache generation must never authorize current proof continuity,
//!   even during the short interval between local persistence and witnessing.
//!
//! ## Last Modified
//! v0.63.0-OpenNodeAdmission - [OPEN-NODE-ADMISSION 2026-09-24 by Codex]
//! Added canonical allowlist-independent Stage-A node admission without route
//! or economic authority
//! v0.62.0-PublicRuntimeEventProjection - Projected public blind-relay runtime
//! events through a closed aggregate allowlist without exporting audit detail
//! v0.61.0-OnionCandidateExclusionTelemetry - Add k-anonymous aggregate
//! candidate-exclusion buckets without changing route admission or ordering
//! v0.60.0-CoreVerifiedRouteContract - Shared the core route hop ceiling and
//! forwarding-capability contract with source-side onion construction
//! v0.59.0-BlindVaultEncryptedFailureNegotiation - Required signed support for
//! source-only terminal failures across every reply-capable vault purpose
//! v0.58.0-BlindVaultRuntimeAdvertisement - Added aggregate runtime readiness
//! and signed capability consistency for anonymous storage replicas
//! v0.57.0-OnionLeaseInventoryTerminalContract - Added feature-gated private
//! encrypted-object inventory commitments
//! v0.56.0-OnionLeaseStatusTerminalContract - Added feature-gated private
//! administration-authorized lease status observations
//! v0.55.0-OnionLeaseRenewalTerminalContract - Added feature-gated blind
//! lease renewal through encrypted terminal replies
//! v0.54.0-OnionLeaseRetireTerminalContract - Added feature-gated complete
//! lease retirement through encrypted terminal replies
//! v0.53.0-OnionPutReceiptTerminalContract - Added feature-gated anonymous
//! writes with terminal-signed encrypted receipts
//! v0.52.0-OnionBlindAdmissionTerminalContract - Added feature-gated blind
//! lease admission through encrypted terminal replies
//! v0.51.0-OnionDeleteTerminalContract - Added signed reply-capable anonymous
//! deletion terminal admission
//! v0.50.0-OnionReplyTerminalContract - Require signed reply-protocol support
//! when selecting anonymous Blind Vault recovery terminals
//! v0.49.0-RecoveryAnchorStatus - Added exact-generation recovery observability
//! and closed the post-persistence stale-witness readiness window
//! v0.48.0-PurposeBoundReceiptNegotiation - Advertise v2 receipt semantics
//! separately from legacy multi-hop receipt framing
//! v0.47.0-CoreRoutePurposeContract - Consumed the shared onion purpose
//! protocol contract and advertised its canonical values for negotiation
//! v0.46.0-OnionRoutePurpose - Added fail-closed, terminal-capability-aware
//! candidate admission for anonymous Blind Vault ciphertext writes
//! v0.45.0-RouteDomainCertificateIngress - Added bounded, rate-limited,
//! verifier-local certificate admission without publishing trust metadata
//! v0.44.0-PinnedRouteDomainAdmission - Added optional fail-closed multi-hop
//! admission using operator-audited opaque route-domain assignments
//! v0.43.0-OnionEntryAntiAffinity - Exclude candidates collocated with the
//! local entry node without exposing the local node id or endpoint.
//! v0.42.0-OnionDiversityAwarePool - Preserve a valid lower-ranked diverse
//! subset when producing a client-limited public candidate pool.
//! v0.41.0-OnionNetworkDiversity - Gate multi-hop candidate readiness on a
//! pairwise network-diverse subset without exposing the selected path.
//! v0.40.0-OnionCapabilityGate - Require signed ChatRelay + OnionMiddle
//! capability and apply candidate limits after all eligibility filters.
//! v0.39.0-OnionPathAdmission - Gate requested multi-hop candidate plans on
//! stable matching runtime proof and signed restart continuity.
//! v0.38.0-PathProofRollbackAnchor - Require local recovery-anchor and optional
//! external-witness readiness before signed proof continuity becomes ready
//! v0.37.0-ThreeHopSignedRecovery - Expose aggregate signed persistence and
//! warm-restart continuity without presenting it as consensus or user traffic.
//! v0.36.0-ThreeHopFeatureNegotiation - Advertise multihop terminal-receipt
//! compatibility so new entries do not penalize legacy middle relays.
//! v0.35.0-ThreeHopRuntimeProof - Added compact independent three-hop onion
//! message-delivery proof status to the public discovery summary.
//! v0.34.0-OnionCandidateSignedProof - Preserve the verified signed descriptor
//! in each public onion candidate and publish the client verification contract.
//! v0.33.0-DiscoveryRateLimitRecovery - Prevent one panic from permanently
//! poisoning the permissionless gossip admission limiter.
//! v0.32.0-DirectoryGossipNegotiation - Advertise additive public transport
//! feature hints so mixed-version peers can avoid unsupported proof frames
//! v0.31.0-DirectoryAuthenticatedGossipAdmission - Gate proof announcements on
//! exact audited local Directory replica evidence before PeerStore import
//! v0.30.0-VerifiedDeliveryPeerGate - Keep public real-relay readiness gated by two currently verified receipt-capable peers
//! v0.29.0-PublicCardRealRelayEvidence - Prefer verified client delivery receipts over synthetic proof labels
//! v0.28.0-VerifiedClientRelayEvidence - Expose aggregate terminal-signed App onion delivery readiness
//! v0.27.0-ProofRestartContinuity - Gate onion admission on verified or durably signed proof stability
//! v0.26.0-RelayEvidenceTruthfulness - Expose origin-neutral accepted relay readiness without claiming user traffic
//! v0.25.0-BoundedGossipBody - Reject oversized gossip before JSON deserialization
//! v0.24.0-DiscoveryPublicCard - Add product-facing public protocol card endpoint
//! v0.23.0-RouteGovernanceHeartbeatReadiness - Add compact route governance to discovery readiness
//! v0.22.0-RouteGovernanceSummary - Add compact route-quality governance to public summary
//! v0.21.0-BlindRelayRuntimeObservability - Add unified blind relay runtime view for nodeboard/backend
//! v0.20.0-OnionRelayAdmissionWarmupDetail - Expose stability-window progress without route metadata
//! v0.19.0-OnionRelayAdmissionContract - Add aggregate admission score and warmup contract
//! v0.18.0-OnionCandidatePoolHealth - Expose aggregate onion candidate pool health for App/nodeboard decisions
//! v0.17.0-DiscoverySummaryProofStabilityWindow - Expose two-hop proof stability and circuit-breaker fields
//! v0.16.0-DiscoverySummaryRestartSurvivableProof - Expose strict restart-survivable two-hop proof readiness
//! v0.15.0-OnionCandidatesFallbackContract - Add explicit two-hop readiness and fallback fields
//! v0.14.0-DiscoverySummaryRecoveredProofStatus - Treat recent message-delivery proof as recovered ready evidence
//! v0.13.0-OnionCandidatesContract - Add explicit client-facing onion candidate contract metadata
//! v0.12.0-DiscoverySummaryContractVersion - Add explicit public summary contract version
//! v0.11.0-DiscoverySummaryProofQuality - Expose privacy-safe two-hop proof quality buckets
//! v0.10.0-DiscoverySummaryEndpoint - Add compact privacy-safe protocol summary endpoint
//! v0.9.3-OnionCandidatesRouteabilityGate - Only expose fresh routeable onion candidates to clients
//! v0.9.2-BlindRelayFreshnessGuard - Expose timestamp rejection aggregate in compact readiness
//! v0.9.1-BlindRelayReadinessReason - Expose privacy-safe relay readiness reason
//! v0.9.0-ProtocolFoundationSummary - Add product-facing privacy protocol foundation readiness
//! v0.8.1-BlindRelayProbeFreshness - Include synthetic probe age in readiness
//! v0.8.0-BlindRelayProbeReadiness - Include synthetic blind relay probe counters in readiness
//! v0.7.0-DiscoveryReadinessStatus - Share compact discovery readiness with status endpoint
//! v0.6.0-RuntimeRelayAdvertisementGate - Gate ChatRelay advertisement on service runtime readiness
//! v0.5.0-LocalCapabilityStatus - Report ChatRelay/blind relay readiness self-check
//! v0.4.0-DiscoveryAuditLog - Added audit events for rate-limit/policy decisions
//! v0.3.0-DiscoveryPhase10-11 - Added status endpoint and inbound safety policy
//! v0.2.0-DiscoveryPhase6 - Public gossip response type for outbound sync
//! v0.1.0-DiscoveryPhase5 - Initial discovery snapshot/gossip HTTP API
// ============================================================================

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::discovery::{
    decode_route_domain_attestation_certificate,
    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES, MAX_SIGNED_NODE_DESCRIPTOR_BYTES,
    // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] Use the defining module.
    PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1,
};
use aeronyx_core::protocol::{
    DiscoveryEndpointEvidenceAttestationV1, NodeBootstrapSnapshot, NodeCapability,
    NodeDiscoveryMessage, NodeProtocolFeature, OnionRoutePurpose,
    PhalaNodeAttestationResponseV1, SignedNodeDescriptor,
    MAX_VERIFIED_ONION_ROUTE_HOPS, ONION_FORWARD_HOP_REQUIRED_CAPABILITIES,
    ONION_ROUTE_PURPOSE_VALUES, PHALA_NODE_ATTESTATION_CONTRACT_VERSION_V1,
    PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V0, PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V1,
    PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1,
    PHALA_NODE_ATTESTATION_VERIFICATION_NOTE_V1,
    PHALA_PRIVATE_RECIPIENT_ATTESTATION_CONTRACT_VERSION_V1,
};
use aeronyx_core::protocol::{
    phala_node_attestation_report_data_v1,
    phala_private_onion_authorization_sha256_v1,
    phala_private_recipient_attestation_report_data_v1,
};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use parking_lot::Mutex;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

use crate::api::directory_replica_sync::admit_directory_gossip_descriptor;
use crate::api::public_node_router::public_endpoint_flow_context;
use crate::config::DiscoveryConfig;
use crate::config_reverse_onion::ReverseOnionQueueConfig;
use crate::services::peer_store::PermissionlessNodeAdmissionOutcome;
use crate::services::{
    DirectoryReplicaStore, DiscoveryEndpointAttestationInboxError,
    DiscoveryEndpointAttestationRecordOutcome, PeerStore, PeerStoreImportReport, PeerStoreStatus,
    RouteDomainCertificateImportError, SqliteDiscoveryEndpointAttestationInbox,
    VerifiedDiscoveryEndpointAttestationV1,
};

// ============================================
// State / Request / Response Types
// ============================================

const ONION_CANDIDATES_CONTRACT_VERSION: &str = "onion_candidates.v1";
/// Maximum JSON gossip request accepted before Axum deserializes it.
///
/// The signed discovery protocol has a 512 KiB binary/message ceiling. JSON
/// encoding adds overhead, so this public HTTP boundary deliberately allows
/// 1 MiB while still preventing unbounded allocation from untrusted peers.
const DISCOVERY_REQUEST_BODY_MAX_BYTES: usize = 1024 * 1024;
const ROUTE_DOMAIN_CERTIFICATE_RATE_LIMIT_PER_MINUTE: u32 = 60;
const PHALA_NODE_ATTESTATION_RATE_LIMIT_PER_MINUTE: u32 = 12;
const PHALA_NODE_ATTESTATION_TIMEOUT_SECS: u64 = 8;
// [PHALA-ATTESTATION-CONCURRENCY 2026-10-06 by Codex] A minute-window limit
// still permits a full burst; bound simultaneous work sent to the local agent.
const PHALA_NODE_ATTESTATION_MAX_IN_FLIGHT: usize = 2;
// [PHALA-ATTESTATION-PROCESS-LIMIT 2026-10-06 by Codex] The discovery router
// is mounted on more than one listener; share one limiter across all of them.
static PHALA_NODE_ATTESTATION_PERMITS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
// [PHALA-ATTESTATION-PROCESS-LIMIT 2026-10-06 by Codex] Apply the request
// budget process-wide too; listener-specific counters multiply the allowance.
static PHALA_NODE_ATTESTATION_RATE_LIMIT: OnceLock<Mutex<RateLimitState>> = OnceLock::new();

fn phala_node_attestation_permits() -> Arc<tokio::sync::Semaphore> {
    Arc::clone(PHALA_NODE_ATTESTATION_PERMITS.get_or_init(|| {
        Arc::new(tokio::sync::Semaphore::new(
            PHALA_NODE_ATTESTATION_MAX_IN_FLIGHT,
        ))
    }))
}

fn phala_node_attestation_rate_limit() -> &'static Mutex<RateLimitState> {
    PHALA_NODE_ATTESTATION_RATE_LIMIT.get_or_init(|| Mutex::new(RateLimitState::new()))
}

// [PHALA-QUOTE-RESPONSE-OWNERSHIP 2026-10-08 by Codex] The process-wide
// quote permit follows its large response buffer until EOS/drop/expiry. A
// weak registry allows admission and the owned server sweeper to reclaim an
// unpolled buffer without retaining it after HTTP cancellation.
const PHALA_QUOTE_RESPONSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const PHALA_QUOTE_RESPONSE_CHUNK_BYTES: usize = 16 * 1024;
const PHALA_QUOTE_RESPONSE_MAX_BYTES: usize = PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 * 2 + 16_384;
static PHALA_QUOTE_RESPONSES: OnceLock<Arc<PhalaQuoteResponseRegistry>> = OnceLock::new();

#[derive(Default)]
struct PhalaQuoteResponseRegistry {
    responses: Mutex<Vec<std::sync::Weak<PhalaQuoteResponseState>>>,
}

pub(crate) struct PhalaAttestationDeliveryOwner {
    closed: std::sync::atomic::AtomicBool,
    registry: Arc<PhalaQuoteResponseRegistry>,
}

impl std::fmt::Debug for PhalaAttestationDeliveryOwner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("PhalaAttestationDeliveryOwner").field("stopped", &self.is_stopped()).finish()
    }
}

impl Default for PhalaAttestationDeliveryOwner {
    fn default() -> Self {
        Self { closed: std::sync::atomic::AtomicBool::new(false),
            registry: Arc::clone(PHALA_QUOTE_RESPONSES.get_or_init(|| Arc::new(PhalaQuoteResponseRegistry::default()))) }
    }
}

impl PhalaAttestationDeliveryOwner {
    pub(crate) fn is_stopped(&self) -> bool {
        self.closed.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub(crate) fn stop(&self) {
        self.closed.store(true, std::sync::atomic::Ordering::SeqCst);
        self.expire_buffers();
    }

    pub(crate) fn expire_buffers(&self) {
        let now = tokio::time::Instant::now();
        self.registry.responses.lock().retain(|weak| {
            let Some(state) = weak.upgrade() else { return false; };
            if state.owner.is_stopped() || now >= state.deadline { state.finish(true); }
            let active = state.data.lock().is_some();
            active
        });
    }

    fn response_body(self: &Arc<Self>, bytes: Vec<u8>, permit: tokio::sync::OwnedSemaphorePermit) -> Result<axum::body::Body, ()> {
        self.expire_buffers();
        let mut responses = self.registry.responses.lock();
        // Registration and stop's sweep share this lock. Stop either rejects
        // this handoff or sees the registered buffer; it cannot miss both.
        if self.is_stopped() || bytes.len() > PHALA_QUOTE_RESPONSE_MAX_BYTES { return Err(()); }
        let state = Arc::new(PhalaQuoteResponseState {
            data: Mutex::new(Some(PhalaQuoteResponseData { bytes, offset: 0, _permit: permit })),
            deadline: tokio::time::Instant::now() + PHALA_QUOTE_RESPONSE_TIMEOUT,
            expired: std::sync::atomic::AtomicBool::new(false),
            owner: Arc::clone(self), waker: futures::task::AtomicWaker::new(),
        });
        responses.push(Arc::downgrade(&state));
        Ok(axum::body::Body::from_stream(PhalaQuoteResponseStream { state, terminated: false }))
    }
}

struct PhalaQuoteResponseData {
    bytes: Vec<u8>,
    offset: usize,
    _permit: tokio::sync::OwnedSemaphorePermit,
}

struct PhalaQuoteResponseState {
    data: Mutex<Option<PhalaQuoteResponseData>>,
    deadline: tokio::time::Instant,
    expired: std::sync::atomic::AtomicBool,
    owner: Arc<PhalaAttestationDeliveryOwner>,
    waker: futures::task::AtomicWaker,
}

impl PhalaQuoteResponseState {
    fn finish(&self, expired: bool) {
        if expired { self.expired.store(true, std::sync::atomic::Ordering::SeqCst); }
        let data = self.data.lock().take();
        drop(data);
        self.waker.wake();
    }
}

struct PhalaQuoteResponseStream {
    state: Arc<PhalaQuoteResponseState>,
    terminated: bool,
}

impl futures::Stream for PhalaQuoteResponseStream {
    type Item = Result<Bytes, io::Error>;

    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        if self.terminated { return Poll::Ready(None); }
        self.state.waker.register(cx.waker());
        if self.state.owner.is_stopped() || tokio::time::Instant::now() >= self.state.deadline { self.state.finish(true); }
        let state = Arc::clone(&self.state);
        let mut slot = state.data.lock();
        if let Some(data) = slot.as_mut() {
            if data.offset < data.bytes.len() {
                let end = data.offset.saturating_add(PHALA_QUOTE_RESPONSE_CHUNK_BYTES).min(data.bytes.len());
                // Never lend a Bytes slice of the whole quote to a slow socket.
                let chunk = Bytes::copy_from_slice(&data.bytes[data.offset..end]);
                data.offset = end;
                return Poll::Ready(Some(Ok(chunk)));
            }
        }
        drop(slot);
        state.finish(false);
        self.terminated = true;
        if state.expired.load(std::sync::atomic::Ordering::SeqCst) {
            Poll::Ready(Some(Err(io::Error::new(io::ErrorKind::TimedOut, "attestation response delivery stopped"))))
        } else { Poll::Ready(None) }
    }
}

impl Drop for PhalaQuoteResponseStream {
    fn drop(&mut self) { self.state.finish(false); }
}
const ONION_CANDIDATES_SOURCE: &str = "rust_discovery_onion_candidates";
const ONION_CANDIDATES_SELECTION_POLICY: &str =
    "fresh_routeable_signed_chat_relays_with_kem_public_key";
// [VERIFIED-ONION-ROUTE 2026-08-29 by Codex] Candidate JSON and the core
// source-side planner share one forwarding-role and maximum-hop contract.
const ONION_REQUIRED_CAPABILITIES: [NodeCapability; 2] = ONION_FORWARD_HOP_REQUIRED_CAPABILITIES;
const ONION_CANDIDATES_REFRESH_AFTER_SECONDS: u64 = 300;
const ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS: u64 = 1_800;
const ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES: usize = 2;
const ONION_CANDIDATES_MAX_CLIENT_HOPS: u8 = MAX_VERIFIED_ONION_ROUTE_HOPS as u8;
const ONION_CANDIDATE_EXCLUSION_TELEMETRY_CONTRACT_VERSION: &str = "onion_candidate_exclusions.v1";
/// Minimum independent observations disclosed by one exclusion bucket.
const ONION_CANDIDATE_EXCLUSION_MIN_BUCKET_SIZE: usize = 3;
const ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS: u64 = 3;
const ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT: u8 = 80;
const DISCOVERY_PUBLIC_CARD_CONTRACT_VERSION: &str = "discovery_public_card.v1";
const DISCOVERY_PUBLIC_CARD_SOURCE: &str = "rust_discovery_public_card";

// [PHALA-NODE-ATTESTATION-API 2026-10-06 by Codex] The public endpoint is
// opt-in, challenge-bound, size/rate bounded, and returns opaque evidence only.
#[derive(Debug, Deserialize)]
struct PhalaNodeAttestationQuery {
    nonce: String,
    #[serde(default)]
    recipient_node_id: Option<String>,
}

// [PHALA-DSTACK-V0-FALLBACK 2026-10-06 by Codex] The evidence format is
// explicit so callers never parse legacy GetQuote JSON as a v1 attestation.
#[derive(Debug, Serialize)]
struct PhalaNodeAttestationError {
    error: &'static str,
}

#[derive(Debug, Deserialize)]
struct DstackAttestBody {
    attestation: String,
}

// [PHALA-DSTACK-V0-FALLBACK 2026-10-06 by Codex]
#[derive(Debug, Deserialize)]
struct DstackV0QuoteBody {
    quote: String,
    event_log: serde_json::Value,
    report_data: String,
}

#[derive(Clone)]
struct DiscoveryApiState {
    peer_store: Arc<PeerStore>,
    /// Local entry identity used only to resolve its signed public descriptor
    /// and enforce coarse route anti-affinity. Never serialize this value.
    local_node_id: Option<[u8; 32]>,
    /// Audited local Directory replica used only as an admission trust anchor.
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    /// Optional durable ADAT quarantine. It has no peer promotion authority.
    endpoint_attestation_inbox: Option<Arc<SqliteDiscoveryEndpointAttestationInbox>>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    rate_limit: Arc<Mutex<RateLimitState>>,
    node_admission_rate_limit: Arc<Mutex<RateLimitState>>,
    route_domain_certificate_rate_limit: Arc<Mutex<RateLimitState>>,
}

/// API-facing discovery safety policy.
#[derive(Debug, Clone)]
pub struct DiscoveryApiPolicy {
    max_snapshot_limit: usize,
    gossip_rate_limit_per_minute: u32,
    allowed_peer_ids: HashSet<String>,
    denied_peer_ids: HashSet<String>,
    /// Operator-audited local assignments. Opaque values are process-only and
    /// must never be serialized by discovery APIs.
    pinned_route_domains: HashMap<String, String>,
    require_pinned_route_domains_for_multi_hop: bool,
    phala_attestation_socket_path: Option<String>,
    phala_attestation_allow_legacy_v0: bool,
    phala_private_recipient_node_id: Option<[u8; 32]>,
    // [PHALA-QUOTE-RESPONSE-OWNERSHIP 2026-10-08 by Codex] Policy clones
    // on local/public listeners share one server lifetime, not a new budget.
    phala_attestation_delivery: Arc<PhalaAttestationDeliveryOwner>,
}

impl DiscoveryApiPolicy {
    /// Builds policy from server discovery config.
    #[must_use]
    pub fn from_config(config: &DiscoveryConfig) -> Self {
        Self {
            max_snapshot_limit: config.max_snapshot_limit,
            gossip_rate_limit_per_minute: config.gossip_rate_limit_per_minute,
            allowed_peer_ids: normalize_peer_ids(&config.allowed_peer_ids),
            denied_peer_ids: normalize_peer_ids(&config.denied_peer_ids),
            pinned_route_domains: config
                .pinned_route_domains
                .iter()
                .map(|(node_id, domain)| {
                    (
                        node_id.trim().to_ascii_lowercase(),
                        domain.trim().to_ascii_lowercase(),
                    )
                })
                .collect(),
            require_pinned_route_domains_for_multi_hop: config
                .require_pinned_route_domains_for_multi_hop,
            phala_attestation_socket_path: config.phala_attestation_socket_path.clone(),
            phala_attestation_allow_legacy_v0: config.phala_attestation_allow_legacy_v0,
            phala_private_recipient_node_id: None,
            phala_attestation_delivery: Arc::new(PhalaAttestationDeliveryOwner::default()),
        }
    }

    pub(crate) fn phala_attestation_delivery_owner(&self) -> Arc<PhalaAttestationDeliveryOwner> {
        Arc::clone(&self.phala_attestation_delivery)
    }

    // [PHALA-QUEUE-RECOVERY-GATE 2026-10-06 by Codex] Derive quote authority
    // from the same live queue policy used by signed discovery advertisement.
    pub(crate) fn with_phala_private_recipient_queue(
        mut self,
        queue: &ReverseOnionQueueConfig,
    ) -> Self {
        let recipient_node_id = (queue.permits_new_claims()
            && queue.recipient_node_ids.len() == 1)
            .then(|| queue.recipient_node_ids[0].as_str());
        self.phala_private_recipient_node_id =
            recipient_node_id.and_then(parse_phala_recipient_node_id);
        self
    }

    fn route_domain_certificate_rate_limit_per_minute(&self) -> u32 {
        self.gossip_rate_limit_per_minute
            .clamp(1, ROUTE_DOMAIN_CERTIFICATE_RATE_LIMIT_PER_MINUTE)
    }

    fn snapshot_limit(&self, requested: Option<usize>) -> usize {
        requested
            .unwrap_or(self.max_snapshot_limit)
            .min(self.max_snapshot_limit)
    }

    fn message_allowed(
        &self,
        message: &NodeDiscoveryMessage,
        has_local_identity: bool,
    ) -> bool {
        match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => true,
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => {
                self.node_allowed(&descriptor.node_id())
            }
            NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { proof, .. } => {
                self.node_allowed(&proof.descriptor.node_id())
            }
            NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { attestation_frame } => {
                DiscoveryEndpointEvidenceAttestationV1::decode(attestation_frame)
                    .map(|attestation| self.node_allowed(&attestation.subject_node_id()))
                    .unwrap_or(false)
            }
            NodeDiscoveryMessage::PrivateOnionRecipientAuthorizationV1 { authorization, .. } => {
                has_local_identity
                    && authorization.relay_node_id() != authorization.recipient_node_id()
                    && self.node_allowed(&authorization.relay_node_id())
                    && self.node_allowed(&authorization.recipient_node_id())
            }
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => snapshot
                .peers
                .iter()
                .all(|descriptor| self.node_allowed(&descriptor.node_id())),
        }
    }

    fn node_allowed(&self, node_id: &[u8; 32]) -> bool {
        let node_id = hex::encode(node_id);
        if self.denied_peer_ids.contains(&node_id) {
            return false;
        }
        self.allowed_peer_ids.is_empty() || self.allowed_peer_ids.contains(&node_id)
    }

    fn node_denied(&self, node_id: &[u8; 32]) -> bool {
        self.denied_peer_ids.contains(&hex::encode(node_id))
    }

    fn pinned_route_domain(&self, node_id: &[u8; 32]) -> Option<&str> {
        self.pinned_route_domains
            .get(&hex::encode(node_id))
            .map(String::as_str)
    }
}

/// Aggregate-only response for permissionless Stage-A node admission.
#[derive(Debug, Clone, Copy, Serialize)]
struct OpenNodeAdmissionResponse {
    accepted: bool,
    status: &'static str,
    route_authority: bool,
    economic_admission: &'static str,
}

impl OpenNodeAdmissionResponse {
    const fn new(accepted: bool, status: &'static str) -> Self {
        Self {
            accepted,
            status,
            route_authority: false,
            economic_admission: "reserved_future_eth_projection_not_enforced",
        }
    }
}

impl Default for DiscoveryApiPolicy {
    fn default() -> Self {
        Self {
            max_snapshot_limit: DiscoveryConfig::default_max_snapshot_limit(),
            gossip_rate_limit_per_minute: DiscoveryConfig::default_gossip_rate_limit_per_minute(),
            allowed_peer_ids: HashSet::new(),
            denied_peer_ids: HashSet::new(),
            pinned_route_domains: HashMap::new(),
            require_pinned_route_domains_for_multi_hop: false,
            phala_attestation_socket_path: None,
            phala_attestation_allow_legacy_v0: false,
            phala_attestation_delivery: Arc::new(PhalaAttestationDeliveryOwner::default()),
            phala_private_recipient_node_id: None,
        }
    }
}

fn normalize_peer_ids(peer_ids: &[String]) -> HashSet<String> {
    peer_ids
        .iter()
        .map(|peer_id| peer_id.trim().to_ascii_lowercase())
        .collect()
}

#[derive(Debug)]
struct RateLimitState {
    window_minute: u64,
    used: u32,
}

impl RateLimitState {
    fn new() -> Self {
        Self {
            window_minute: 0,
            used: 0,
        }
    }

    fn allow(&mut self, now: u64, limit: u32) -> bool {
        let window_minute = now / 60;
        if self.window_minute != window_minute {
            self.window_minute = window_minute;
            self.used = 0;
        }
        if self.used >= limit {
            return false;
        }
        self.used += 1;
        true
    }
}

// [PHALA-NODE-ATTESTATION-API 2026-10-06 by Codex] This route asks the local
// dstack agent for a challenge-bound quote; it deliberately does not decide
// whether that quote, compose, app identity, or TCB is trusted.
async fn phala_node_attestation_handler(
    State(state): State<DiscoveryApiState>,
    Query(query): Query<PhalaNodeAttestationQuery>,
) -> impl IntoResponse {
    let Some(socket_path) = state.policy.phala_attestation_socket_path.as_deref() else {
        return (
            StatusCode::NOT_FOUND,
            Json(PhalaNodeAttestationError {
                error: "attestation_unavailable",
            }),
        )
            .into_response();
    };
    let Some(node_id) = state.local_node_id else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(PhalaNodeAttestationError {
                error: "attestation_unavailable",
            }),
        )
            .into_response();
    };
    let Some(nonce) = parse_phala_attestation_nonce(&query.nonce) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(PhalaNodeAttestationError {
                error: "invalid_nonce",
            }),
        )
            .into_response();
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    if !phala_node_attestation_rate_limit()
        .lock()
        .allow(now, PHALA_NODE_ATTESTATION_RATE_LIMIT_PER_MINUTE)
    {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(PhalaNodeAttestationError {
                error: "rate_limited",
            }),
        )
            .into_response();
    }
    // [PHALA-ATTESTATION-CONCURRENCY 2026-10-06 by Codex] Do not queue
    // anonymous quote requests behind the guest agent. Bound its concurrent
    // work and let callers retry after a short, explicit overload response.
    // [PHALA-QUOTE-RESPONSE-OWNERSHIP 2026-10-08 by Codex] Admission
    // reclaims expired buffers even for embedded routers without a sweeper.
    let delivery_owner = state.policy.phala_attestation_delivery_owner();
    delivery_owner.expire_buffers();
    if delivery_owner.is_stopped() {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(PhalaNodeAttestationError { error: "attestation_unavailable" })).into_response();
    }
    let permits = phala_node_attestation_permits();
    let Ok(attestation_permit) = permits.try_acquire_owned()
    else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(PhalaNodeAttestationError {
                error: "rate_limited",
            }),
        )
            .into_response();
    };

    let recipient_authority = match query.recipient_node_id.as_deref() {
        Some(value) => match parse_phala_recipient_node_id(value) {
            Some(recipient) => {
                if !phala_private_recipient_pin_matches(&state.policy, &recipient) {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(PhalaNodeAttestationError {
                            error: "attestation_unavailable",
                        }),
                    )
                        .into_response();
                }
                let authority = state.peer_store.current_private_onion_authority_snapshot(
                    &node_id,
                    &recipient,
                    now,
                );
                let Some((_, _, authorization)) = authority else {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(PhalaNodeAttestationError {
                            error: "attestation_unavailable",
                        }),
                    )
                        .into_response();
                };
                let Ok(digest) = phala_private_onion_authorization_sha256_v1(&authorization) else {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(PhalaNodeAttestationError {
                            error: "attestation_unavailable",
                        }),
                    )
                        .into_response();
                };
                // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex]
                Some((recipient, digest))
            }
            None => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(PhalaNodeAttestationError {
                        error: "invalid_recipient_node_id",
                    }),
                )
                    .into_response();
            }
        },
        None => None,
    };

    // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] A recipient
    // quote requires the relay's exact current signed R/P/grant snapshot.
    // This endpoint transports evidence; it does not appraise TCB or prove
    // that the recipient key itself is held in the TEE.
    let report_data = match recipient_authority.as_ref() {
        Some((recipient, authorization_sha256)) => phala_private_recipient_attestation_report_data_v1(
            &node_id,
            recipient,
            authorization_sha256,
            &nonce,
        ),
        None => phala_node_attestation_report_data_v1(&node_id, &nonce),
    };
    let expected_report_data = {
        let mut padded = [0_u8; 64];
        padded[..report_data.len()].copy_from_slice(&report_data);
        hex::encode(padded)
    };
    let request = serde_json::json!({"report_data": hex::encode(report_data)});
    let request = match serde_json::to_vec(&request) {
        Ok(request) => request,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(PhalaNodeAttestationError {
                    error: "attestation_unavailable",
                }),
            )
                .into_response();
        }
    };

    let fetched = tokio::time::timeout(
        std::time::Duration::from_secs(PHALA_NODE_ATTESTATION_TIMEOUT_SECS),
        fetch_dstack_attestation(
            socket_path,
            &request,
            &expected_report_data,
            state.policy.phala_attestation_allow_legacy_v0,
        ),
    )
    .await;
    let (attestation_format, attestation) = match fetched {
        Ok(Ok(attestation)) => attestation,
        Ok(Err(_)) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(PhalaNodeAttestationError {
                    error: "attestation_unavailable",
                }),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(PhalaNodeAttestationError {
                    error: "attestation_timeout",
                }),
            )
                .into_response();
        }
    };

    // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] Quote
    // generation awaits the guest agent; do not return recipient-bound
    // evidence if its signed R/P/grant authority expired or was withdrawn
    // while that request was in flight.
    if let Some((recipient, expected_authorization_sha256)) = recipient_authority {
        let response_now = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => duration.as_secs(),
            Err(_) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(PhalaNodeAttestationError {
                        error: "attestation_unavailable",
                    }),
                )
                    .into_response();
            }
        };
        let current_authorization_sha256 = state
            .peer_store
            .current_private_onion_authority_snapshot(&node_id, &recipient, response_now)
            .and_then(|(_, _, authorization)| {
                phala_private_onion_authorization_sha256_v1(&authorization).ok()
            });
        if response_now < now
            || current_authorization_sha256 != Some(expected_authorization_sha256)
        {
            return (
                StatusCode::NOT_FOUND,
                Json(PhalaNodeAttestationError {
                    error: "attestation_unavailable",
                }),
            )
                .into_response();
        }
    }

    let response = PhalaNodeAttestationResponseV1 {
        contract_version: if recipient_authority.is_some() {
            PHALA_PRIVATE_RECIPIENT_ATTESTATION_CONTRACT_VERSION_V1.into()
        } else {
            PHALA_NODE_ATTESTATION_CONTRACT_VERSION_V1.into()
        },
        node_id: hex::encode(node_id),
        recipient_node_id: recipient_authority.map(|(recipient, _)| hex::encode(recipient)),
        authorization_sha256: recipient_authority
            .map(|(_, digest)| hex::encode(digest)),
        nonce: hex::encode(nonce),
        expected_report_data,
        attestation_format: attestation_format.into(),
        attestation: hex::encode(attestation),
        verification: PHALA_NODE_ATTESTATION_VERIFICATION_NOTE_V1.into(),
    };
    let recipient_binding = recipient_authority
        .as_ref()
        .map(|(recipient, digest)| (recipient, digest));
    if response
        .validate_for(&node_id, &nonce, recipient_binding)
        .is_err()
    {
        return (
            StatusCode::BAD_GATEWAY,
            Json(PhalaNodeAttestationError {
                error: "attestation_unavailable",
            }),
        )
            .into_response();
    }
    // [PHALA-QUOTE-RESPONSE-OWNERSHIP 2026-10-08 by Codex] Serialization
    // and delivery remain inside the same two-slot process admission bound.
    let Ok(bytes) = serde_json::to_vec(&response) else {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(PhalaNodeAttestationError { error: "attestation_unavailable" })).into_response();
    };
    let length = bytes.len();
    let Ok(body) = delivery_owner.response_body(bytes, attestation_permit) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(PhalaNodeAttestationError { error: "attestation_unavailable" })).into_response();
    };
    let mut response = Response::new(body);
    response.headers_mut().insert(axum::http::header::CONTENT_TYPE, axum::http::HeaderValue::from_static("application/json"));
    if let Ok(length) = axum::http::HeaderValue::from_str(&length.to_string()) {
        response.headers_mut().insert(axum::http::header::CONTENT_LENGTH, length);
    }
    response
}

fn parse_phala_attestation_nonce(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let decoded = hex::decode(value).ok()?;
    let nonce: [u8; 32] = decoded.try_into().ok()?;
    (nonce.iter().any(|byte| *byte != 0)).then_some(nonce)
}

// [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] Recipient IDs
// are canonical nonzero Ed25519 public keys, never endpoint input.
fn parse_phala_recipient_node_id(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    let decoded: [u8; 32] = hex::decode(value).ok()?.try_into().ok()?;
    (decoded != [0; 32]).then_some(decoded)
}

fn phala_private_recipient_pin_matches(
    policy: &DiscoveryApiPolicy,
    requested: &[u8; 32],
) -> bool {
    policy.phala_private_recipient_node_id.as_ref() == Some(requested)
}

async fn fetch_dstack_attestation(
    socket_path: &str,
    request_body: &[u8],
    expected_report_data: &str,
    allow_legacy_v0: bool,
) -> io::Result<(&'static str, Vec<u8>)> {
    if socket_path.len() > 4096 || !Path::new(socket_path).is_absolute() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid socket path"));
    }
    let v1_response = fetch_dstack_json(socket_path, "/v1/Attest", request_body).await?;
    if v1_response.status == 200 {
        return parse_dstack_v1_attestation(&v1_response.body, expected_report_data)
            .map(|attestation| (PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V1, attestation));
    }
    if !allow_legacy_v0 || !is_dstack_v1_mount_missing(&v1_response) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "agent rejected request"));
    }

    // [PHALA-DSTACK-V0-FALLBACK 2026-10-06 by Codex] Fall back only when the
    // documented v1-unmounted 404 is observed; never downgrade on timeouts,
    // malformed responses, method-level errors, or quote-generation failures.
    let v0_response = fetch_dstack_json(socket_path, "/GetQuote", request_body).await?;
    if v0_response.status != 200 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "legacy quote request rejected"));
    }
    parse_dstack_v0_quote(&v0_response.body, expected_report_data)
        .map(|opaque_json| (PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V0, opaque_json))
}

// [PHALA-DSTACK-V0-FALLBACK 2026-10-06 by Codex]
struct DstackHttpResponse {
    status: u16,
    content_type: Option<String>,
    body: Vec<u8>,
}

async fn fetch_dstack_json(
    socket_path: &str,
    path: &str,
    request_body: &[u8],
) -> io::Result<DstackHttpResponse> {
    let mut socket = UnixStream::connect(socket_path).await?;
    let header = format!(
        "POST {path} HTTP/1.1\r\nHost: dstack\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        request_body.len()
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(request_body).await?;
    socket.shutdown().await?;

    let mut response = Vec::new();
    socket
        .take((PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1 + 1) as u64)
        .read_to_end(&mut response)
        .await?;
    parse_dstack_http_response(&response)
}

fn parse_dstack_http_response(response: &[u8]) -> io::Result<DstackHttpResponse> {
    if response.len() > PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "response too large"));
    }
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed response"))?;
    let headers = std::str::from_utf8(&response[..separator])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "malformed headers"))?;
    let mut lines = headers.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let mut status_parts = status_line.split_ascii_whitespace();
    let version = status_parts.next().unwrap_or_default();
    let status = status_parts
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .filter(|_| version == "HTTP/1.1" || version == "HTTP/1.0")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed status line"))?;
    let mut content_length = None;
    let mut content_type = None;
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed headers"))?;
        if name.eq_ignore_ascii_case("transfer-encoding")
            || name.eq_ignore_ascii_case("content-encoding")
        {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported encoding"));
        }
        if name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "duplicate length"));
            }
            content_length = Some(
                value
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid length"))?,
            );
        } else if name.eq_ignore_ascii_case("content-type") {
            if content_type.is_some() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "duplicate content type"));
            }
            content_type = Some(value.trim().to_ascii_lowercase());
        }
    }
    let body = &response[separator + 4..];
    let json_content_type = content_type
        .as_deref()
        .and_then(|value| value.split(';').next())
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"));
    if content_length != Some(body.len()) || (status == 200 && !json_content_type) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid body framing"));
    }
    Ok(DstackHttpResponse {
        status,
        content_type,
        body: body.to_vec(),
    })
}

fn parse_dstack_v1_attestation(
    body: &[u8],
    expected_report_data: &str,
) -> io::Result<Vec<u8>> {
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Unknown JSON
    // members cannot hide ambiguous evidence from downstream consumers.
    crate::services::memchain::validate_phala_json(body, PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid attestation response"))?;
    let decoded: DstackAttestBody = serde_json::from_slice(body)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid attestation response"))?;
    let attestation = hex::decode(decoded.attestation)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid attestation encoding"))?;
    if attestation.is_empty() || attestation.len() > PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid attestation size"));
    }
    // [PHALA-DSTACK-V1-ACI-EVIDENCE 2026-10-06 by Codex] Decode the complete
    // normative v1 envelope, not just its outer tag, before advertising it.
    let (report_data, _) = dstack_v1_aci_evidence(&attestation)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid v1 attestation evidence"))?;
    if hex::encode(report_data) != expected_report_data {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "v1 report data mismatch"));
    }
    Ok(attestation)
}

// [PHALA-DSTACK-V1-ACI-EVIDENCE 2026-10-06 by Codex] dstack v1 is named
// MessagePack with binary quote/event bytes. aci-verify accepts a different
// JSON view, so convert only the authenticated schema fields explicitly.
fn dstack_v1_aci_evidence(attestation: &[u8]) -> Result<(Vec<u8>, serde_json::Value), &'static str> {
    let mut cursor = std::io::Cursor::new(attestation);
    let mut decoder = rmp_serde::Deserializer::new(&mut cursor);
    let envelope = DstackVersionedAttestationEnvelope::deserialize(&mut decoder)
        .map_err(|_| "invalid dstack v1 messagepack")?;
    drop(decoder);
    if cursor.position() != attestation.len() as u64 {
        return Err("trailing dstack v1 messagepack bytes");
    }
    if envelope.version != 1 || envelope.platform.kind != "tdx"
        || !matches!(envelope.stack.kind.as_str(), "dstack" | "dstack-pod") {
        return Err("unsupported dstack v1 evidence");
    }

    let DstackVersionedAttestationPlatformData { quote, event_log } = envelope.platform.data;
    let DstackVersionedAttestationStackData { report_data, config } = envelope.stack.data;
    if quote.0.is_empty()
        || quote.0.len() > PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1
        || report_data.0.len() != 64
    {
        return Err("invalid dstack v1 component data");
    }
    let report_data = report_data.0;
    let event_log = event_log
        .into_iter()
        .map(|event| {
            serde_json::json!({
                "imr": event.imr,
                "event_type": event.event_type,
                "digest": hex::encode(event.digest.0),
                "event": event.event,
                "event_payload": hex::encode(event.event_payload.0),
            })
        })
        .collect::<Vec<_>>();
    let evidence = serde_json::json!({
        "quote": hex::encode(quote.0),
        "quote_report_data": hex::encode(&report_data),
        "event_log": serde_json::to_string(&event_log).map_err(|_| "invalid dstack event log")?,
        "app_compose": config,
    });
    Ok((report_data, evidence))
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct DstackVersionedAttestationEnvelope {
    version: u64,
    platform: DstackVersionedAttestationPlatform,
    stack: DstackVersionedAttestationStack,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct DstackVersionedAttestationPlatform {
    kind: String,
    data: DstackVersionedAttestationPlatformData,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct DstackVersionedAttestationPlatformData {
    quote: DstackByteField,
    event_log: Vec<DstackVersionedTdxEvent>,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct DstackVersionedTdxEvent {
    imr: u32,
    event_type: u32,
    digest: DstackByteField,
    event: String,
    event_payload: DstackByteField,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct DstackVersionedAttestationStack {
    kind: String,
    data: DstackVersionedAttestationStackData,
}

#[derive(Deserialize)]
#[cfg_attr(test, derive(Serialize))]
struct DstackVersionedAttestationStackData {
    report_data: DstackByteField,
    config: String,
}

// [PHALA-DSTACK-MESSAGEPACK-BYTES 2026-10-06 by Codex] dstack emits binary
// MessagePack tokens for byte fields, while older/test producers may encode
// them as integer arrays. Accept both wire forms without adding a dependency.
struct DstackByteField(Vec<u8>);

#[cfg(test)]
impl Serialize for DstackByteField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for DstackByteField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ByteFieldVisitor;

        impl<'de> serde::de::Visitor<'de> for ByteFieldVisitor {
            type Value = Vec<u8>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a MessagePack binary value or byte sequence")
            }

            fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(value.to_vec())
            }

            fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(value)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut bytes = Vec::new();
                while let Some(byte) = sequence.next_element::<u8>()? {
                    bytes.push(byte);
                }
                Ok(bytes)
            }
        }

        deserializer.deserialize_any(ByteFieldVisitor).map(Self)
    }
}

fn is_dstack_v1_mount_missing(response: &DstackHttpResponse) -> bool {
    if response.status != 404 {
        return false;
    }
    let media_type = response
        .content_type
        .as_deref()
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    matches!(media_type, Some("text/plain" | "text/html"))
        && serde_json::from_slice::<serde_json::Value>(&response.body).is_err()
}

fn parse_dstack_v0_quote(body: &[u8], expected_report_data: &str) -> io::Result<Vec<u8>> {
    if body.is_empty() || body.len() > PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "legacy quote response too large"));
    }
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] Preserve legacy
    // evidence bytes, but reject duplicate members before passing them on.
    crate::services::memchain::validate_phala_json(body, PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid legacy quote response"))?;
    let decoded: DstackV0QuoteBody = serde_json::from_slice(body)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid legacy quote response"))?;
    let report_data = decoded.report_data.strip_prefix("0x").unwrap_or(&decoded.report_data);
    // [PHALA-DSTACK-V0-HEX-PREFIX 2026-10-06 by Codex] v0 encodes `bytes` as
    // hex and accepts an optional 0x prefix on input.
    let quote_hex = decoded.quote.strip_prefix("0x").unwrap_or(&decoded.quote);
    let quote_bytes = hex::decode(quote_hex)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid legacy quote encoding"))?;
    if !decoded.event_log.is_array()
        || quote_bytes.is_empty()
        || quote_bytes.len() > PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1
        || !report_data.eq_ignore_ascii_case(expected_report_data)
    {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid legacy quote evidence"));
    }
    // Preserve exact evidence bytes; callers distinguish v0 JSON from v1
    // MessagePack through the additive `attestation_format` field.
    Ok(body.to_vec())
}

#[derive(Debug, Deserialize)]
struct SnapshotQuery {
    limit: Option<usize>,
    public_only: Option<bool>,
}

/// Query for the onion relay candidate endpoint.
#[derive(Debug, Deserialize)]
struct OnionCandidatesQuery {
    limit: Option<usize>,
    /// Optional route purpose. Omitted keeps the historical message-relay
    /// contract. Unknown values fail closed instead of silently downgrading a
    /// storage request into ordinary message delivery.
    purpose: Option<String>,
    /// Optional product privacy mode requested by the client.
    ///
    /// Stable values are `standard`, `enhanced`, and `high`. Unknown values
    /// fall back to `enhanced` so older clients and AI agents get the existing
    /// two-hop behavior instead of accidentally downgrading privacy.
    privacy_mode: Option<String>,
    /// Optional explicit relay-hop count requested by advanced clients.
    ///
    /// Values are clamped to 1..=3. The local node serving this endpoint is the
    /// entry context, so this count means remote relay hops returned from the
    /// candidate pool, not total network nodes.
    hops: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnionPrivacyMode {
    Standard,
    Enhanced,
    High,
}

/// Internal fail-closed gates for the path requested by an App or SDK.
///
/// [ONION-PATH-ADMISSION 2026-08-02 by Codex] Candidate availability,
/// runtime delivery proof, and restart continuity are deliberately separate.
/// This keeps a populated descriptor pool from being mistaken for an actually
/// exercised multi-hop transport. The structure contains aggregate booleans
/// only and must never carry selected relays or route metadata.
#[derive(Debug, Clone, Copy)]
struct OnionRequestedPathGates {
    purpose_supported: bool,
    terminal_capability_ready: bool,
    candidate_pool_ready: bool,
    network_diversity_required: bool,
    network_diversity_ready: bool,
    pinned_route_domain_required: bool,
    pinned_route_domain_ready: bool,
    runtime_proof_required: bool,
    runtime_proof_ready: bool,
    restart_continuity_required: bool,
    restart_continuity_ready: bool,
}

#[derive(Debug, Clone, Copy)]
struct OnionPurposeAdmission {
    supported: bool,
    terminal_capability_ready: bool,
}

#[derive(Debug, Clone, Copy)]
struct OnionRequirementGate {
    required: bool,
    ready: bool,
}

/// [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Named admission inputs prevent
/// purpose and policy booleans from being reordered accidentally at the
/// two-hop and three-hop call sites.
#[derive(Debug, Clone, Copy)]
struct OnionCandidateAdmissionInput {
    purpose: OnionPurposeAdmission,
    candidate_pool_ready: bool,
    network_diversity_ready: bool,
    pinned_route_domain: OnionRequirementGate,
}

impl OnionRequestedPathGates {
    const fn ready(self) -> bool {
        self.purpose_supported
            && self.terminal_capability_ready
            && self.candidate_pool_ready
            && (!self.network_diversity_required || self.network_diversity_ready)
            && (!self.pinned_route_domain_required || self.pinned_route_domain_ready)
            && (!self.runtime_proof_required || self.runtime_proof_ready)
            && (!self.restart_continuity_required || self.restart_continuity_ready)
    }
}

impl OnionPrivacyMode {
    fn from_query(value: Option<&str>) -> Self {
        match value
            .unwrap_or("enhanced")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "standard" | "fast" | "low_latency" | "low-latency" => Self::Standard,
            "high" | "maximum" | "max" => Self::High,
            _ => Self::Enhanced,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Enhanced => "enhanced",
            Self::High => "high",
        }
    }

    const fn default_hops(self) -> u8 {
        match self {
            Self::Standard => 1,
            Self::Enhanced => 2,
            Self::High => 3,
        }
    }
}

/// One onion-routing relay candidate: the signed, public node discovery
/// metadata a client needs to build an onion layer addressed to this hop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionRelayCandidate {
    /// Relay Ed25519 node id, hex-encoded.
    pub node_id: String,
    /// KEM algorithm id (1 = X25519; 2 = X-Wing, reserved).
    pub kem_alg: u8,
    /// Relay KEM public key, hex-encoded — build the onion layer against this.
    pub kem_public: String,
    /// Public control-plane endpoint for node-to-node relay traffic.
    pub public_endpoint: String,
    /// Advertised capability flags (lets the client pick middle vs exit hops).
    pub capabilities: Vec<NodeCapability>,
    /// Relative selection weight for client-side weighted random path building.
    ///
    /// Higher-ranked, healthier candidates receive a higher bucket. Clients
    /// should still sample randomly within the eligible pool so traffic does
    /// not collapse onto the first listed relay.
    pub selection_weight: u16,
    /// Optional public region hint from the signed descriptor.
    pub region: Option<String>,
    /// Coarse max session capacity advertised by the peer.
    pub max_sessions: u32,
    /// Optional bandwidth policy advertised by the peer.
    pub max_bps: Option<u64>,
    /// Optional packet-rate policy advertised by the peer.
    pub max_pps: Option<u64>,
    /// Original Ed25519-signed descriptor accepted by the local `PeerStore`.
    ///
    /// [ONION-CANDIDATE-PROOF 2026-07-31 by Codex] The flattened fields above
    /// remain for backward compatibility. Security-sensitive App/SDK path
    /// builders must independently call the protocol-equivalent of
    /// `SignedNodeDescriptor::verify_at(generated_at)` and then derive node id,
    /// KEM key, endpoint, capabilities, capacity, and region from this object.
    /// A mismatch between a flattened field and this descriptor must reject the
    /// candidate rather than silently trusting the API projection.
    pub signed_descriptor: SignedNodeDescriptor,
}

/// K-anonymous aggregate observations explaining why public descriptors did
/// not reach the onion candidate pool.
///
/// [ONION-CANDIDATE-EXCLUSION-TELEMETRY 2026-08-31 by Codex] Every positive
/// bucket smaller than the fixed disclosure threshold is omitted. The
/// diagnostic pass never returns identities, endpoints, keys, routes, hashes,
/// timestamps, or enough bucket totals to reconstruct a suppressed value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnionCandidateExclusionTelemetry {
    /// Independently versioned additive telemetry contract.
    pub contract_version: String,
    /// Whether every coarse bucket is publishable under the privacy contract.
    pub status: OnionCandidateExclusionTelemetryStatus,
    /// Fixed k-anonymity threshold applied independently to every bucket.
    pub minimum_bucket_size: usize,
    /// First matching exclusion reason is counted at most once per descriptor.
    pub bucket_semantics: String,
    /// Protected aggregate observations. Missing members were suppressed.
    pub buckets: OnionCandidateExclusionBuckets,
    /// Explicit privacy boundary for operators and downstream agents.
    pub privacy_boundary: String,
}

/// Closed telemetry disclosure outcomes serialized as stable snake-case values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnionCandidateExclusionTelemetryStatus {
    /// Every bucket is either zero or large enough to publish.
    Ready,
    /// At least one positive bucket or internal observation was suppressed.
    Partial,
    /// The bounded descriptor sample itself is smaller than k.
    SuppressedSmallSample,
}

/// Optional counts in the versioned candidate-exclusion contract.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnionCandidateExclusionBuckets {
    /// Missing relay capabilities or required signed protocol features.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_or_feature: Option<usize>,
    /// Missing or expired local routeability evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routeability_unknown_or_stale: Option<usize>,
    /// Recent route failure or active route quarantine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routeability_failed_or_quarantined: Option<usize>,
    /// Missing signed KEM material or public endpoint metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing_kem_or_endpoint: Option<usize>,
    /// Entry collocation, self exclusion, or pinned-domain policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anti_affinity_or_policy: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct OnionCandidateExclusionCounts {
    capability_or_feature: usize,
    routeability_unknown_or_stale: usize,
    routeability_failed_or_quarantined: usize,
    missing_kem_or_endpoint: usize,
    anti_affinity_or_policy: usize,
    unclassified: usize,
}

impl OnionCandidateExclusionCounts {
    fn protected_bucket(count: usize, sample_adequate: bool) -> Option<usize> {
        (sample_adequate && (count == 0 || count >= ONION_CANDIDATE_EXCLUSION_MIN_BUCKET_SIZE))
            .then_some(count)
    }

    fn into_telemetry(self, observed_descriptors: usize) -> OnionCandidateExclusionTelemetry {
        let sample_adequate = observed_descriptors >= ONION_CANDIDATE_EXCLUSION_MIN_BUCKET_SIZE;
        let suppressed_bucket = [
            self.capability_or_feature,
            self.routeability_unknown_or_stale,
            self.routeability_failed_or_quarantined,
            self.missing_kem_or_endpoint,
            self.anti_affinity_or_policy,
        ]
        .into_iter()
        .any(|count| count > 0 && count < ONION_CANDIDATE_EXCLUSION_MIN_BUCKET_SIZE);
        let status = if !sample_adequate {
            OnionCandidateExclusionTelemetryStatus::SuppressedSmallSample
        } else if suppressed_bucket || self.unclassified > 0 {
            OnionCandidateExclusionTelemetryStatus::Partial
        } else {
            OnionCandidateExclusionTelemetryStatus::Ready
        };

        OnionCandidateExclusionTelemetry {
            contract_version: ONION_CANDIDATE_EXCLUSION_TELEMETRY_CONTRACT_VERSION.to_string(),
            status,
            minimum_bucket_size: ONION_CANDIDATE_EXCLUSION_MIN_BUCKET_SIZE,
            bucket_semantics:
                "bounded_public_descriptors; first_matching_reason; positive_counts_below_k_omitted"
                    .to_string(),
            buckets: OnionCandidateExclusionBuckets {
                capability_or_feature: Self::protected_bucket(
                    self.capability_or_feature,
                    sample_adequate,
                ),
                routeability_unknown_or_stale: Self::protected_bucket(
                    self.routeability_unknown_or_stale,
                    sample_adequate,
                ),
                routeability_failed_or_quarantined: Self::protected_bucket(
                    self.routeability_failed_or_quarantined,
                    sample_adequate,
                ),
                missing_kem_or_endpoint: Self::protected_bucket(
                    self.missing_kem_or_endpoint,
                    sample_adequate,
                ),
                anti_affinity_or_policy: Self::protected_bucket(
                    self.anti_affinity_or_policy,
                    sample_adequate,
                ),
            },
            privacy_boundary: "aggregate exclusion buckets only; no peer identifiers or prefixes, endpoints, public keys, routes, hashes, timestamps, payloads, or reconstructable small-sample details".to_string(),
        }
    }
}

/// Signed terminal requirements for one onion workload.
///
/// [ONION-TERMINAL-CONTRACT 2026-08-28 by Codex] Coarse capabilities describe
/// the node role while signed protocol features describe the exact response
/// contract. Keeping both in one value prevents candidate filtering, bounded
/// pool preservation, and route-readiness checks from drifting apart.
#[derive(Clone, Copy)]
struct OnionTerminalRequirement {
    capability: Option<NodeCapability>,
    protocol_features: &'static [NodeProtocolFeature],
}

impl Default for OnionTerminalRequirement {
    fn default() -> Self {
        Self {
            capability: None,
            protocol_features: &[],
        }
    }
}

impl OnionTerminalRequirement {
    fn for_purpose(purpose: Option<OnionRoutePurpose>) -> Self {
        Self {
            capability: purpose.and_then(OnionRoutePurpose::specialized_terminal_capability),
            // [ONION-TERMINAL-FEATURE-CONTRACT 2026-08-28 by Codex] The exact
            // signed wire contract is owned by the core purpose model; this
            // server adapter only applies it to discovered candidates.
            protocol_features: purpose
                .map_or(&[], |purpose| purpose.required_terminal_protocol_features()),
        }
    }

    fn is_specialized(self) -> bool {
        self.capability.is_some() || !self.protocol_features.is_empty()
    }

    fn matches(self, candidate: &OnionRelayCandidate) -> bool {
        let descriptor = &candidate.signed_descriptor.descriptor;
        self.capability
            .map_or(true, |required| descriptor.capabilities.contains(&required))
            && self
                .protocol_features
                .iter()
                .all(|required| descriptor.advertises_protocol_feature(*required))
    }
}

fn default_onion_route_purpose() -> String {
    OnionRoutePurpose::MessageRelay.as_str().to_string()
}

const fn default_true() -> bool {
    true
}

/// Response for `GET /api/discovery/onion-candidates`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionCandidatesResponse {
    /// Unix timestamp when the candidate set was generated.
    pub generated_at: u64,
    /// Stable JSON contract version for App, SDK, and AI-agent path builders.
    pub contract_version: String,
    /// Stable source label for downstream telemetry/runbooks.
    pub source: String,
    /// Signed capabilities every returned onion candidate must advertise.
    ///
    /// [ONION-CAPABILITY-GATE 2026-08-02 by Codex] This additive field makes
    /// the multi-hop eligibility contract machine-readable while preserving
    /// the existing `onion_candidates.v1` response for older clients.
    #[serde(default = "onion_required_capabilities")]
    pub required_capabilities: Vec<NodeCapability>,
    /// Normalized terminal purpose requested by the client.
    #[serde(default = "default_onion_route_purpose")]
    pub requested_purpose: String,
    /// Whether the requested purpose is implemented by this contract.
    #[serde(default = "default_true")]
    pub requested_purpose_supported: bool,
    /// Signed capabilities required of the terminal selected by the client.
    /// Middle relays need only `required_capabilities`.
    #[serde(default = "onion_required_capabilities")]
    pub terminal_required_capabilities: Vec<NodeCapability>,
    /// Signed SemVer build-metadata feature tokens required from the terminal.
    ///
    /// Clients must verify the original signed descriptor and require every
    /// token in this list. An empty list means the requested purpose has no
    /// additional fine-grained terminal feature beyond its capabilities.
    #[serde(default)]
    pub terminal_required_protocol_features: Vec<String>,
    /// Signed SemVer feature tokens required from every selected route hop.
    ///
    /// [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] Clients must verify
    /// these tokens on original signed descriptors before building a path.
    /// Missing support must fail closed or fall back to a legacy one-way mode;
    /// it must never silently expose a v2 terminal identity.
    #[serde(default)]
    pub path_required_protocol_features: Vec<String>,
    /// Number of returned candidates whose original signed descriptor satisfies
    /// the complete terminal capability contract.
    #[serde(default)]
    pub terminal_candidate_count: usize,
    /// Whether at least one returned candidate can serve as the requested
    /// terminal. Older responses default true for backward compatibility.
    #[serde(default = "default_true")]
    pub requested_terminal_capability_ready: bool,
    /// Number of candidates returned.
    pub count: usize,
    /// Versioned, k-anonymous candidate-exclusion diagnostics.
    ///
    /// Missing on older responses and omitted when deserializing/re-serializing
    /// a legacy response, preserving the v1 candidate contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_exclusion_telemetry: Option<OnionCandidateExclusionTelemetry>,
    /// Minimum unique candidates required for a client-planned two-hop path.
    ///
    /// The entry node is the local node serving this endpoint, so clients need
    /// at least two other fresh routeable relays: one middle hop and one
    /// terminal hop. If fewer are available, clients should fall back to the
    /// standard encrypted relay path.
    pub min_candidates_for_two_hop: usize,
    /// Whether this response contains enough fresh routeable candidates for a
    /// controlled two-hop path attempt.
    pub two_hop_ready: bool,
    /// Product privacy mode requested by the client after normalization.
    pub requested_privacy_mode: String,
    /// Number of remote relay hops requested after normalization.
    pub requested_hops: u8,
    /// Number of fresh routeable candidates required for the requested hop count.
    pub min_candidates_for_requested_hops: usize,
    /// Whether this candidate set can satisfy the requested hop count.
    pub requested_path_ready: bool,
    /// Whether enough distinct routeable candidates exist for the requested
    /// hop count before runtime proof gates are applied.
    pub requested_candidate_pool_ready: bool,
    /// Whether the requested path requires coarse endpoint-network diversity.
    #[serde(default)]
    pub requested_network_diversity_required: bool,
    /// Whether a pairwise network-diverse candidate subset exists for the
    /// requested hop count.
    #[serde(default)]
    pub requested_network_diversity_ready: bool,
    /// Whether candidates were also checked against the local entry node's
    /// coarse endpoint network identity.
    #[serde(default)]
    pub local_entry_network_diversity_enforced: bool,
    /// Whether complete operator-pinned route-domain coverage is required for
    /// this requested multi-hop path.
    #[serde(default)]
    pub requested_pinned_route_domain_required: bool,
    /// Whether the local entry and a complete remote-hop subset have distinct,
    /// operator-audited opaque route-domain assignments.
    #[serde(default)]
    pub requested_pinned_route_domain_ready: bool,
    /// Whether the local entry resolved to an operator-pinned route domain.
    #[serde(default)]
    pub local_entry_pinned_route_domain_enforced: bool,
    /// Whether this requested path requires matching runtime delivery proof.
    pub requested_runtime_proof_required: bool,
    /// Whether the matching runtime delivery proof gate currently passes.
    pub requested_runtime_proof_ready: bool,
    /// Whether this requested path requires signed restart continuity.
    pub requested_restart_continuity_required: bool,
    /// Whether the matching signed restart-continuity gate currently passes.
    pub requested_restart_continuity_ready: bool,
    /// Best hop count the client can safely attempt from this response.
    pub recommended_hops: u8,
    /// Whether the requested path cannot currently be satisfied and the client
    /// must follow `route_plan` as a lower-hop or standard encrypted fallback.
    pub fallback_required: bool,
    /// Aggregate requested-path maturity bucket.
    ///
    /// Stable values include `ready`, `unsupported_purpose`, `terminal_limited`,
    /// `diversity_limited`, `proof_warming`, `continuity_warming`, `warming`,
    /// `empty`, or `client_limited`.
    /// This lets App, nodeboard, backend aggregation, and AI-agent runbooks
    /// distinguish a usable pool from a partial pool without inspecting
    /// individual relay metadata.
    pub pool_status: String,
    /// Privacy-safe route plan recommendation for clients.
    ///
    /// Stable values include `three_hop_onion_path`, `two_hop_onion_path`,
    /// `single_hop_encrypted_relay`, `standard_relay_fallback`,
    /// `defer_specialized_delivery`, or `reject_unsupported_purpose`.
    /// The server never returns route ids, selected path ids, receiver
    /// identities, payload metadata, or client information here.
    pub route_plan: String,
    /// Stable privacy-safe reason bucket for fallback decisions.
    ///
    /// This must never include node ids, endpoint URLs, route ids, receiver
    /// identities, encrypted payloads, client IPs, DNS contents, destinations,
    /// Memory Chain plaintext, voucher secrets, private keys, wallet-level
    /// traffic, or social graph metadata.
    pub fallback_reason: String,
    /// Stable privacy-safe readiness reason for product surfaces.
    pub readiness_reason: String,
    /// Short operator/client action that does not expose route metadata.
    pub next_action: String,
    /// Privacy-safe route selection policy used to build this candidate set.
    pub selection_policy: String,
    /// Stable verification rule for each candidate's public metadata.
    ///
    /// This field is intentionally explicit so mixed-version App/SDK clients
    /// can distinguish independently verifiable candidates from legacy
    /// projections without inferring support from optional JSON members.
    pub candidate_verification: String,
    /// Stable strategy clients should use when choosing among candidates.
    pub path_selection_strategy: String,
    /// Coarse endpoint anti-affinity policy enforced before multi-hop admission.
    #[serde(default)]
    pub network_diversity_policy: String,
    /// Operator-pinned route-domain policy. Opaque assignment values are never
    /// included in this public response.
    #[serde(default)]
    pub pinned_route_domain_policy: String,
    /// Privacy-safe region diversity policy for client-side path builders.
    pub region_diversity_policy: String,
    /// Product-facing rule: users choose a privacy level, not raw node ids.
    pub user_choice_policy: String,
    /// Recommended client refresh interval for this candidate set.
    pub refresh_after_seconds: u64,
    /// Maximum routeability age accepted by this endpoint before a candidate is
    /// hidden. Clients should refresh before this value and must tolerate an
    /// empty candidate set by falling back to the standard relay path.
    pub routeability_stale_after_seconds: u64,
    /// Health-ranked onion relay candidates; each advertises a KEM key and a
    /// reachable public endpoint.
    pub candidates: Vec<OnionRelayCandidate>,
    /// Explicit privacy boundary for downstream consumers.
    pub privacy_boundary: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GossipResponse {
    pub applied: PeerStoreImportReport,
    pub response: Option<NodeDiscoveryMessage>,
}

/// Stable, identity-blind result of one route-domain certificate submission.
#[derive(Debug, Serialize)]
struct RouteDomainCertificateImportResponse {
    /// Whether the frame and its locally pinned attestor quorum were accepted.
    accepted: bool,
    /// Whether this request inserted/replaced evidence instead of being idempotent.
    stored: bool,
    /// Stable aggregate outcome with no subject, attestor, token, or hash.
    status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct DiscoveryStatusResponse {
    generated_at: u64,
    peer_store: PeerStoreStatus,
    policy: DiscoveryPolicyStatus,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    discovery_readiness: serde_json::Value,
    /// Unified privacy-safe runtime view for nodeboard/backend.
    ///
    /// This duplicates selected aggregate counters from `peer_store` into a
    /// stable product-facing shape. It must not include endpoints, route IDs,
    /// encrypted payloads, receiver identities, client IPs, DNS contents,
    /// destinations, Memory Chain plaintext, private keys, wallet-level
    /// traffic, or social graph metadata.
    blind_relay_runtime: serde_json::Value,
    /// Aggregate recovery-anchor and external-witness readiness.
    ///
    /// [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] This exposes only local
    /// generation numbers, status buckets, and bounded counts. It must never
    /// contain anchor digests, signatures, witness identities/endpoints,
    /// routes, peers, clients, messages, or payload metadata.
    recovery_anchor: serde_json::Value,
}

/// Compact public-safe discovery summary.
///
/// This is the preferred response for app, website, backend aggregation, and
/// AI-agent runbooks that only need protocol health storytelling. It must not
/// include signed descriptors, full node ids, endpoint URLs, route ids,
/// encrypted payloads, receiver identities, client public IPs, DNS contents,
/// destinations, Memory Chain plaintext, voucher secrets, private keys,
/// wallet-level traffic, or social graph metadata.
#[derive(Debug, Serialize)]
pub struct DiscoverySummaryResponse {
    /// Unix timestamp when the summary was generated.
    generated_at: u64,
    /// Stable public JSON contract version for backend, nodeboard, website,
    /// app, and AI-agent consumers.
    contract_version: &'static str,
    /// Stable summary source label.
    source: &'static str,
    /// Non-authoritative transport feature hints for mixed-version peers.
    ///
    /// [DIRECTORY-GOSSIP-NEGOTIATION 2026-07-27 by Codex] These booleans only
    /// suppress unsupported optional frames. They must never grant descriptor,
    /// replica, witness, checkpoint, policy, consensus, or routing authority.
    protocol_features: serde_json::Value,
    /// Product-facing current protocol status bucket.
    status: String,
    /// Product-facing current protocol stage bucket.
    stage: String,
    /// Short display headline safe for public surfaces.
    headline: String,
    /// Local capability readiness without route/user metadata.
    local_capability: serde_json::Value,
    /// Verified peer mesh aggregate without descriptors or endpoints.
    peer_mesh: serde_json::Value,
    /// Route governance aggregate without endpoints, selected paths, or payload data.
    route_governance: serde_json::Value,
    /// Blind relay aggregate runtime/probe evidence without payload metadata.
    blind_relay: serde_json::Value,
    /// Product-facing blind relay runtime counters and last safe event buckets.
    blind_relay_runtime: serde_json::Value,
    /// Bounded two-hop path proof aggregate without route reconstruction data.
    two_hop_path_proof: serde_json::Value,
    /// Bounded runtime-only three-hop proof aggregate without route data.
    three_hop_path_proof: serde_json::Value,
    /// Aggregate permissionless relay-pool admission gate without route data.
    onion_relay_admission: serde_json::Value,
    /// Aggregate exact-generation recovery protection without secret material.
    recovery_anchor: serde_json::Value,
    /// Actionable next step for operators and AI runbooks.
    next_action: String,
    /// Explicit invariant for downstream UI and AI-agent consumers.
    privacy_invariant: &'static str,
    /// Explicit privacy boundary for downstream UI/API consumers.
    privacy_boundary: &'static str,
}

/// Minimal product-facing protocol card.
///
/// This response is intentionally smaller than `DiscoverySummaryResponse`.
/// Use it for website home modules, Nodeboard first-level cards, App "Privacy
/// Network" surfaces, and AI-agent runbooks that need a quick answer to:
/// "Is this node participating in the blind AeroNyx privacy protocol right
/// now?" It exposes only aggregate readiness and counters. Do not add peer
/// endpoints, full node ids, route ids, selected hops, receiver identifiers,
/// encrypted payload metadata, DNS contents, destinations, client public IPs,
/// Memory Chain plaintext, private keys, wallet-level traffic, or social graph
/// metadata.
#[derive(Debug, Serialize)]
pub struct DiscoveryPublicCardResponse {
    /// Unix timestamp when the card was generated.
    generated_at: u64,
    /// Stable public JSON contract version for website, Nodeboard, app, and
    /// AI-agent consumers.
    contract_version: &'static str,
    /// Stable source label for downstream aggregation.
    source: &'static str,
    /// Product-facing protocol health bucket.
    status: String,
    /// Product-facing protocol stage bucket.
    stage: String,
    /// Short display headline safe for public surfaces.
    headline: String,
    /// Human-readable health label for first-level UI cards.
    health_label: &'static str,
    /// Compact confidence score derived from aggregate readiness checks.
    confidence_percent: u8,
    /// Three top-level cards intended for primary product surfaces.
    cards: serde_json::Value,
    /// Additional compact readiness signals for badges and detail links.
    signals: serde_json::Value,
    /// UI guidance that keeps first-level pages focused and avoids diagnostic overload.
    display_policy: serde_json::Value,
    /// Actionable next step for operators and AI runbooks.
    next_action: String,
    /// Explicit invariant for downstream UI and AI-agent consumers.
    privacy_invariant: &'static str,
    /// Explicit privacy boundary for downstream UI/API consumers.
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct DiscoveryPolicyStatus {
    max_snapshot_limit: usize,
    gossip_rate_limit_per_minute: u32,
    allow_list_enabled: bool,
    allowed_peer_count: usize,
    denied_peer_count: usize,
    pinned_route_domain_count: usize,
    require_pinned_route_domains_for_multi_hop: bool,
    snapshot_default_public_only: bool,
    private_descriptors_hidden_by_default: bool,
}

/// [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Typed aggregate
/// observation used to avoid order-dependent boolean parameters at the
/// discovery/Blind Vault boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiscoveryBlindVaultCapabilityObservation {
    /// Whether the operator enabled signed replica advertisement.
    pub configured: bool,
    /// Whether current policy, issuer, and capacity admission is ready.
    pub runtime_ready: bool,
    /// Whether the signed self descriptor carries the replica capability.
    pub advertised: bool,
}

/// Privacy-safe local protocol capability readiness.
///
/// This object is intentionally small and aggregate-only. It tells operators
/// whether the node configuration, runtime relay service, public peer API
/// endpoint, and advertised descriptor capabilities agree with each other,
/// without exposing route ids, peer endpoints, client addresses, payloads, or
/// user identifiers.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryLocalCapabilityStatus {
    /// Whether `[memchain.chat_relay].enabled` is true.
    pub chat_relay_configured: bool,
    /// Whether this process has the public discovery/peer API listener and a
    /// public endpoint configured, which is required by peer relay routes.
    pub blind_relay_endpoint_ready: bool,
    /// Whether `ChatRelayService` initialized successfully at runtime.
    ///
    /// This prevents the node from advertising `NodeCapability::ChatRelay`
    /// when configuration is enabled but the backing relay service failed to
    /// start, for example because SQLite or the relay DB path is unavailable.
    pub chat_relay_runtime_ready: bool,
    /// Whether the self descriptor advertises `NodeCapability::ChatRelay`.
    pub advertised_chat_relay_capability: bool,
    /// Whether it is safe for this node to advertise `ChatRelay`.
    pub safe_to_advertise_chat_relay: bool,
    /// [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Whether the
    /// operator configured this node to advertise anonymous storage.
    pub blind_vault_replica_configured: bool,
    /// Whether current policy, issuer state, and logical/physical capacity can
    /// admit a new anonymous lease.
    pub blind_vault_runtime_ready: bool,
    /// Whether the signed self descriptor currently carries the replica
    /// capability.
    pub advertised_blind_vault_replica_capability: bool,
    /// Whether the complete relay, endpoint, configuration, and storage
    /// runtime surface can safely advertise the replica capability.
    pub safe_to_advertise_blind_vault_replica: bool,
    /// Whether actual Blind Vault advertisement equals runtime expectation.
    pub blind_vault_capability_consistent: bool,
    /// Stable aggregate blockers for anonymous storage advertisement.
    pub blind_vault_advertisement_blockers: Vec<&'static str>,
    /// Whether config, endpoint readiness, and advertised capability agree.
    pub capability_config_consistent: bool,
    /// Stable privacy-safe reason buckets that block ChatRelay advertisement.
    pub advertisement_blockers: Vec<&'static str>,
    /// Stable operator-facing status: `ready`, `disabled`, or `misconfigured`.
    pub status: &'static str,
    /// Short remediation-oriented detail safe for public discovery status.
    pub detail: &'static str,
}

impl DiscoveryLocalCapabilityStatus {
    /// Builds a privacy-safe readiness summary for local discovery status.
    #[must_use]
    pub fn new(
        chat_relay_configured: bool,
        blind_relay_endpoint_ready: bool,
        chat_relay_runtime_ready: bool,
        advertised_chat_relay_capability: bool,
    ) -> Self {
        Self::new_with_blind_vault(
            chat_relay_configured,
            blind_relay_endpoint_ready,
            chat_relay_runtime_ready,
            advertised_chat_relay_capability,
            DiscoveryBlindVaultCapabilityObservation::default(),
        )
    }

    /// Builds local capability status with observed anonymous-storage state.
    #[must_use]
    pub fn new_with_blind_vault(
        chat_relay_configured: bool,
        blind_relay_endpoint_ready: bool,
        chat_relay_runtime_ready: bool,
        advertised_chat_relay_capability: bool,
        blind_vault: DiscoveryBlindVaultCapabilityObservation,
    ) -> Self {
        let DiscoveryBlindVaultCapabilityObservation {
            configured: blind_vault_replica_configured,
            runtime_ready: blind_vault_runtime_ready,
            advertised: advertised_blind_vault_replica_capability,
        } = blind_vault;
        let safe_to_advertise_chat_relay =
            chat_relay_configured && blind_relay_endpoint_ready && chat_relay_runtime_ready;
        let expected_advertisement = safe_to_advertise_chat_relay;
        let capability_config_consistent =
            advertised_chat_relay_capability == expected_advertisement;
        let mut advertisement_blockers = Vec::new();
        if !chat_relay_configured {
            advertisement_blockers.push("chat_relay_disabled");
        }
        if !blind_relay_endpoint_ready {
            advertisement_blockers.push("public_peer_api_not_ready");
        }
        if chat_relay_configured && !chat_relay_runtime_ready {
            advertisement_blockers.push("chat_relay_runtime_not_ready");
        }
        let safe_to_advertise_blind_vault_replica = blind_vault_replica_configured
            && safe_to_advertise_chat_relay
            && blind_vault_runtime_ready;
        let blind_vault_capability_consistent =
            advertised_blind_vault_replica_capability == safe_to_advertise_blind_vault_replica;
        let mut blind_vault_advertisement_blockers = Vec::new();
        if !blind_vault_replica_configured {
            blind_vault_advertisement_blockers.push("blind_vault_replica_disabled");
        } else {
            if !blind_relay_endpoint_ready {
                blind_vault_advertisement_blockers.push("public_peer_api_not_ready");
            }
            if !chat_relay_configured || !chat_relay_runtime_ready {
                blind_vault_advertisement_blockers.push("chat_relay_runtime_not_ready");
            }
            if !blind_vault_runtime_ready {
                blind_vault_advertisement_blockers.push("blind_vault_admission_not_ready");
            }
        }
        let (status, detail) = if !capability_config_consistent {
            (
                "misconfigured",
                "chat relay capability advertisement does not match config, endpoint, and runtime readiness",
            )
        } else if advertised_chat_relay_capability {
            (
                "ready",
                "chat relay runtime and blind relay peer endpoint are configured and advertised",
            )
        } else if chat_relay_configured && !chat_relay_runtime_ready {
            (
                "misconfigured",
                "chat relay is enabled but the runtime relay service is not ready",
            )
        } else if chat_relay_configured {
            (
                "misconfigured",
                "chat relay is enabled but public peer API endpoint is not ready",
            )
        } else {
            (
                "disabled",
                "chat relay is disabled; blind relay endpoint remains available for discovery API plumbing",
            )
        };

        Self {
            chat_relay_configured,
            blind_relay_endpoint_ready,
            chat_relay_runtime_ready,
            advertised_chat_relay_capability,
            safe_to_advertise_chat_relay,
            blind_vault_replica_configured,
            blind_vault_runtime_ready,
            advertised_blind_vault_replica_capability,
            safe_to_advertise_blind_vault_replica,
            blind_vault_capability_consistent,
            blind_vault_advertisement_blockers,
            capability_config_consistent,
            advertisement_blockers,
            status,
            detail,
        }
    }
}

impl Default for DiscoveryLocalCapabilityStatus {
    fn default() -> Self {
        Self::new(false, false, false, false)
    }
}

/// Internal aggregate proof-continuity decision shared by admission and the
/// public summary. It deliberately carries only authentication/status buckets
/// and counts already present in `PeerStoreBootstrapStatus`.
struct PathProofRestartContinuity {
    peer_recovery_configured: bool,
    authenticated_restore_ready: bool,
    signed_persistence_ready: bool,
    ready: bool,
    source: &'static str,
    authentication: String,
    rollback_protection: String,
    external_witness: String,
    external_witness_required: bool,
    restored: u64,
    persisted: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExternalWitnessRecoveryAdmission {
    ready: bool,
    adverse_evidence: bool,
    generation_aligned: bool,
}

// ============================================
// Router
// ============================================

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod gossip_ingress;
mod onion_candidates;
mod status_card;

use gossip_ingress::apply_gossip_message;
use gossip_ingress::gossip_handler;
use gossip_ingress::open_node_admission_handler;
use gossip_ingress::persist_or_discard_endpoint_attestation;
use gossip_ingress::route_domain_certificate_handler;
use gossip_ingress::snapshot_handler;
use onion_candidates::normalize_requested_hops;
use onion_candidates::onion_candidate_exclusion_counts;
use onion_candidates::onion_candidate_fallback_reason;
#[cfg(test)]
use onion_candidates::onion_candidate_network_diverse_subset_indices;
#[cfg(test)]
use onion_candidates::onion_candidate_network_diversity_ready;
use onion_candidates::onion_candidate_next_action;
use onion_candidates::onion_candidate_pool_status;
use onion_candidates::onion_candidate_readiness_reason;
#[cfg(test)]
use onion_candidates::onion_candidate_route_diverse_subset_indices;
use onion_candidates::onion_candidate_route_diverse_subset_indices_for_terminal;
use onion_candidates::onion_candidate_route_diversity_ready_for_terminal;
use onion_candidates::onion_candidate_route_plan;
use onion_candidates::onion_candidate_selection_weight;
use onion_candidates::onion_candidate_supports_specialized_terminal;
use onion_candidates::onion_candidates_are_route_diverse;
use onion_candidates::onion_candidates_handler;
use onion_candidates::onion_path_required_protocol_features;
use onion_candidates::onion_requested_path_gates;
use onion_candidates::onion_required_capabilities;
use onion_candidates::onion_route_purpose_from_query;
use onion_candidates::onion_route_purpose_name;
use onion_candidates::onion_terminal_candidate_matches;
use onion_candidates::onion_terminal_required_capabilities;
use onion_candidates::onion_terminal_required_protocol_features;
use onion_candidates::recommended_onion_hops;
#[cfg(test)]
use onion_candidates::select_onion_candidate_response_pool;
#[cfg(test)]
use onion_candidates::select_onion_candidate_response_pool_with_policy;
use onion_candidates::select_onion_candidate_response_pool_with_policy_and_terminal;
pub use status_card::blind_relay_runtime_status_value;
pub use status_card::discovery_public_card_response;
pub use status_card::discovery_readiness_status_value;
pub use status_card::discovery_summary_response;
use status_card::external_witness_recovery_admission;
use status_card::latest_blind_relay_event_value;
use status_card::now_secs;
pub use status_card::onion_relay_admission_status_value;
use status_card::path_proof_restart_continuity;
use status_card::public_blind_relay_event_reason_bucket;
use status_card::public_card_handler;
use status_card::public_status_audit_event_is_publishable;
use status_card::recovery_anchor_protection_ready;
pub use status_card::recovery_anchor_status_value;
use status_card::sanitize_public_peer_store_status;
use status_card::status_handler;
use status_card::summary_handler;
use status_card::three_hop_proof_restart_continuity;
use status_card::two_hop_proof_restart_continuity;

// [ARCH-SPLIT-VERIFY 2026-10-02 by Codex] Keep router documentation on its public builder.
/// Builds the discovery API router.
pub fn build_discovery_router(peer_store: Arc<PeerStore>, policy: DiscoveryApiPolicy) -> Router {
    build_discovery_router_with_local_status(
        peer_store,
        policy,
        DiscoveryLocalCapabilityStatus::default(),
    )
}

/// Builds the discovery API router with local capability readiness status.
pub fn build_discovery_router_with_local_status(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
) -> Router {
    build_discovery_router_with_local_status_and_directory_admission(
        peer_store,
        policy,
        local_capabilities,
        None,
    )
}

/// Builds the discovery API router with capability status and optional
/// Directory-authenticated gossip admission.
///
/// [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] Keeping this as an
/// additive builder preserves every existing caller while allowing production
/// nodes to inject their already audited replica store.
pub fn build_discovery_router_with_local_status_and_directory_admission(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
) -> Router {
    build_discovery_router_state(
        peer_store,
        policy,
        local_capabilities,
        directory_replica_store,
        None,
        None,
    )
}

/// Builds the production discovery router with local-entry anti-affinity.
///
/// [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] The local id is process
/// context only. The handler resolves the already-public signed descriptor and
/// filters collocated route candidates without returning the local id,
/// endpoint, selected route, or any client metadata.
pub fn build_discovery_router_with_local_entry(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    local_node_id: [u8; 32],
) -> Router {
    build_discovery_router_with_local_entry_and_attestation_inbox(
        peer_store,
        policy,
        local_capabilities,
        directory_replica_store,
        local_node_id,
        None,
    )
}

/// Builds the production discovery router with optional durable ADAT quarantine.
///
/// Existing builders delegate with no inbox so default-off behavior remains
/// canonical verify-and-discard with no filesystem or task side effect.
pub fn build_discovery_router_with_local_entry_and_attestation_inbox(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    local_node_id: [u8; 32],
    endpoint_attestation_inbox: Option<Arc<SqliteDiscoveryEndpointAttestationInbox>>,
) -> Router {
    build_discovery_router_state(
        peer_store,
        policy,
        local_capabilities,
        directory_replica_store,
        Some(local_node_id),
        endpoint_attestation_inbox,
    )
}

fn build_discovery_router_state(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    local_node_id: Option<[u8; 32]>,
    endpoint_attestation_inbox: Option<Arc<SqliteDiscoveryEndpointAttestationInbox>>,
) -> Router {
    // [PERMISSIONLESS-DISCOVERY-CANDIDATES 2026-09-14 by Codex] The public
    // gossip surface has no endpoint-possession proof. Enable the PeerStore's
    // bounded candidate mode before exposing it so legacy self-signed gossip
    // cannot consume verified-live routing capacity.
    peer_store.enable_untrusted_discovery_candidate_mode();
    let state = DiscoveryApiState {
        peer_store,
        local_node_id,
        directory_replica_store,
        endpoint_attestation_inbox,
        policy,
        local_capabilities,
        rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
        node_admission_rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
        route_domain_certificate_rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
    };
    Router::new()
        .route(
            "/api/discovery/join",
            post(open_node_admission_handler)
                .layer(DefaultBodyLimit::max(MAX_SIGNED_NODE_DESCRIPTOR_BYTES)),
        )
        .route("/api/discovery/snapshot", get(snapshot_handler))
        .route("/api/discovery/gossip", post(gossip_handler))
        .route(
            "/api/discovery/route-domain-certificate",
            post(route_domain_certificate_handler).layer(DefaultBodyLimit::max(
                MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES,
            )),
        )
        .route("/api/discovery/status", get(status_handler))
        .route(
            "/api/discovery/phala-attestation",
            get(phala_node_attestation_handler)
                .layer(middleware::from_fn(no_store_phala_attestation_response)),
        )
        .route("/api/discovery/summary", get(summary_handler))
        .route("/api/discovery/public-card", get(public_card_handler))
        .route(
            "/api/discovery/onion-candidates",
            get(onion_candidates_handler),
        )
        .layer(DefaultBodyLimit::max(DISCOVERY_REQUEST_BODY_MAX_BYTES))
        .with_state(state)
}

// [PHALA-ATTESTATION-NO-STORE 2026-10-06 by Codex] Quote responses are bound
// to a caller nonce and must never be replayed by an HTTP cache.
async fn no_store_phala_attestation_response(
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store, private"),
    );
    response.headers_mut().insert(
        axum::http::header::PRAGMA,
        axum::http::HeaderValue::from_static("no-cache"),
    );
    response
}

// [PHALA-PEER-QUOTE-APPRAISAL 2026-10-06 by Codex] Retrieve and appraise a
// peer's nonce-bound dstack v1 evidence. Caller supplies only locally pinned
// trust roots; descriptor-advertised features and response metadata never
// expand these allowlists.
// [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Both discovery
// owners share this budget, including detached blocking verification children.
pub(crate) const PHALA_APPRAISAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
const PHALA_PCCS_MAX_BODY_BYTES: usize = 1024 * 1024;
const PHALA_PCCS_MAX_HEADER_BYTES: usize = 32 * 1024;
static PHALA_APPRAISAL_PERMITS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();

// [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex] Both the
// verifier result and its route cache bound descriptor expiry to the original
// challenge's monotonic age. A wall-clock rollback must not extend that window.
pub(crate) fn phala_appraisal_effective_time(
    challenged_at: u64, now: u64, elapsed: std::time::Duration, max_age_secs: u64,
) -> Option<u64> {
    if now.checked_sub(challenged_at)? > max_age_secs
        || elapsed > std::time::Duration::from_secs(max_age_secs)
    {
        return None;
    }
    Some(now.max(challenged_at.checked_add(elapsed.as_secs())?))
}

// [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] This process-local
// result is created only after QVL, nonce, event-log and measurement appraisal.
// It is not serializable, caller-supplied, or proof of recipient key residency.
pub(crate) struct VerifiedPhalaPeerAttestation {
    descriptor_commitment: [u8; 32],
    challenged_at: u64,
    verified_at: u64,
    started: std::time::Instant,
}

impl VerifiedPhalaPeerAttestation {
    pub(crate) fn cache_time_for(
        &self, descriptor: &SignedNodeDescriptor, now: u64, max_age_secs: u64,
    ) -> Option<(u64, std::time::Instant)> {
        let commitment = aeronyx_core::protocol::discovery::signed_descriptor_commitment_hash(descriptor).ok()?;
        // [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex]
        let effective_now = phala_appraisal_effective_time(self.challenged_at, now, self.started.elapsed(), max_age_secs)?;
        (commitment == self.descriptor_commitment
            && self.challenged_at <= self.verified_at && self.verified_at <= now
            && effective_now == now
            && descriptor.verify_at(self.challenged_at).is_ok()
            && descriptor.verify_at(self.verified_at).is_ok()
            && descriptor.verify_at(effective_now).is_ok()).then_some((self.challenged_at, self.started))
    }

    #[cfg(test)]
    pub(crate) fn synthetic_for_test(descriptor: &SignedNodeDescriptor, challenged_at: u64, verified_at: u64) -> Self {
        Self { descriptor_commitment: aeronyx_core::protocol::discovery::signed_descriptor_commitment_hash(descriptor).unwrap(),
            challenged_at, verified_at, started: std::time::Instant::now() }
    }
}

// [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] The QVL SCALE
// decoder allocates declared byte-vector lengths before reading their bodies.
// Validate only those framing/allocation bounds first; QVL remains the parser
// and cryptographic verifier. No untrusted length may allocate beyond its slice.
pub(crate) fn guard_phala_quote_allocations(quote: &[u8]) -> Result<(), &'static str> {
    fn data<'a>(bytes: &'a [u8], offset: &mut usize, width: usize) -> Result<&'a [u8], &'static str> {
        let end = offset.checked_add(width).ok_or("peer_quote_malformed")?;
        let encoded = bytes.get(*offset..end).ok_or("peer_quote_malformed")?;
        let length = match width {
            2 => usize::from(u16::from_le_bytes(encoded.try_into().map_err(|_| "peer_quote_malformed")?)),
            4 => usize::try_from(u32::from_le_bytes(encoded.try_into().map_err(|_| "peer_quote_malformed")?))
                .map_err(|_| "peer_quote_malformed")?,
            _ => return Err("peer_quote_malformed"),
        };
        let data_end = end.checked_add(length).ok_or("peer_quote_malformed")?;
        let body = bytes.get(end..data_end).ok_or("peer_quote_malformed")?;
        *offset = data_end;
        Ok(body)
    }
    if quote.len() > PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 || quote.len() < 48 {
        return Err("peer_quote_malformed");
    }
    let version = u16::from_le_bytes([quote[0], quote[1]]);
    let tee = u32::from_le_bytes(quote[4..8].try_into().map_err(|_| "peer_quote_malformed")?);
    let mut offset = match (version, tee) {
        (3, 0) | (4, 0) => 48 + 384,
        (4, 0x81) => 48 + 584,
        (5, _) => {
            let body = quote.get(48..54).ok_or("peer_quote_malformed")?;
            let kind = u16::from_le_bytes([body[0], body[1]]);
            let size = match kind { 1 => 384, 2 => 584, 3 => 648, _ => return Err("peer_quote_malformed") };
            if u32::from_le_bytes(body[2..6].try_into().map_err(|_| "peer_quote_malformed")?) != size as u32 {
                return Err("peer_quote_malformed");
            }
            54 + size
        }
        _ => return Err("peer_quote_malformed"),
    };
    let auth = data(quote, &mut offset, 4)?;
    if offset != quote.len() { return Err("peer_quote_malformed"); }
    let (certification, mut inner_offset) = if version == 3 {
        (auth, 576)
    } else {
        let mut outer_offset = 130; // signature + attestation key + cert type
        let certification = data(auth, &mut outer_offset, 4)?;
        if outer_offset != auth.len() { return Err("peer_quote_malformed"); }
        (certification, 448) // QE report + QE signature
    };
    data(certification, &mut inner_offset, 2)?; // QE authentication bytes
    inner_offset = inner_offset.checked_add(2).ok_or("peer_quote_malformed")?; // inner cert type
    data(certification, &mut inner_offset, 4)?;
    if inner_offset != certification.len() { return Err("peer_quote_malformed"); }
    Ok(())
}

// [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Only these official
// PCCS resources are reachable. No certificate CRL URL, redirect or ambient
// proxy may choose a destination. Query fields are added after DNS pinning.
fn phala_pccs_url(
    target: &super::PinnedPeerHttpTarget, path: &str, query: &[(&str, String)],
) -> Result<reqwest::Url, &'static str> {
    if !super::reverse_onion_same_origin(target.url.as_str(), dcap_qvl::PHALA_PCCS_URL)
        || !matches!(path, "/sgx/certification/v4/pckcert" | "/sgx/certification/v4/pckcrl"
            | "/sgx/certification/v4/rootcacrl" | "/tdx/certification/v4/tcb"
            | "/tdx/certification/v4/qe/identity") {
        return Err("peer_collateral_target_rejected");
    }
    let mut url = target.url.clone();
    url.set_path(path);
    url.set_query(None);
    if !query.is_empty() {
        url.query_pairs_mut().extend_pairs(query.iter().map(|(name, value)| (*name, value.as_str())));
    }
    Ok(url)
}

async fn fetch_phala_pccs_resource(
    target: &super::PinnedPeerHttpTarget, path: &str, query: &[(&str, String)],
) -> Result<(reqwest::header::HeaderMap, Vec<u8>), &'static str> {
    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] The same pinned
    // client serves every resource, with per-response streaming limits.
    let url = phala_pccs_url(target, path, query)?;
    let response = target.client.get(url.clone()).send().await.map_err(|_| "peer_collateral_unavailable")?;
    if !response.status().is_success() || response.url() != &url {
        return Err("peer_collateral_unavailable");
    }
    if response.content_length().is_some_and(|length| length > PHALA_PCCS_MAX_BODY_BYTES as u64)
        || response.headers().iter().map(|(name, value)| name.as_str().len().saturating_add(value.as_bytes().len()))
            .try_fold(0usize, |total, length| total.checked_add(length)).is_none_or(|total| total > PHALA_PCCS_MAX_HEADER_BYTES) {
        return Err("peer_collateral_oversized");
    }
    let headers = response.headers().clone();
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| "peer_collateral_unavailable")?;
        if body.len().saturating_add(chunk.len()) > PHALA_PCCS_MAX_BODY_BYTES { return Err("peer_collateral_oversized"); }
        body.extend_from_slice(&chunk);
    }
    if body.is_empty() { return Err("peer_collateral_malformed"); }
    Ok((headers, body))
}

fn phala_pccs_header(headers: &reqwest::header::HeaderMap, name: &str) -> Result<String, &'static str> {
    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] A percent-
    // encoded certificate header stays bounded before and after decoding.
    let encoded = headers.get(name).ok_or("peer_collateral_malformed")?;
    if encoded.as_bytes().len() > PHALA_PCCS_MAX_HEADER_BYTES { return Err("peer_collateral_oversized"); }
    percent_encoding::percent_decode(encoded.as_bytes()).decode_utf8()
        .map(|value| value.into_owned()).map_err(|_| "peer_collateral_malformed")
}

fn phala_pccs_signed_payload(body: &[u8], field: &str) -> Result<(String, Vec<u8>), &'static str> {
    // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Preserve signed
    // bytes within our restrictive Intel profile. Pinned Rust QVL verifies
    // these strings directly, unlike Intel C++'s DOM-to-Writer step; do not
    // generalize whitespace-only extraction to every valid JSON spelling.
    #[derive(serde::Deserialize)]
    struct SignedCollateral<'a> {
        #[serde(rename = "tcbInfo", borrow)]
        tcb_info: Option<&'a serde_json::value::RawValue>,
        #[serde(rename = "enclaveIdentity", borrow)]
        enclave_identity: Option<&'a serde_json::value::RawValue>,
        signature: String,
    }
    if body.len() > PHALA_PCCS_MAX_BODY_BYTES { return Err("peer_collateral_oversized"); }
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] RawValue must
    // preserve signed lexemes, not conceal duplicate members inside them.
    crate::services::memchain::validate_phala_json(body, PHALA_PCCS_MAX_BODY_BYTES)
        .map_err(|_| "peer_collateral_malformed")?;
    // Typed deserialization rejects duplicate known fields, including escaped
    // aliases. RawValue validates JSON before the whitespace-only pass below.
    let envelope: SignedCollateral<'_> = serde_json::from_slice(body)
        .map_err(|_| "peer_collateral_malformed")?;
    let payload = match (field, envelope.tcb_info, envelope.enclave_identity) {
        ("tcbInfo", Some(payload), None) | ("enclaveIdentity", None, Some(payload)) => payload.get(),
        _ => return Err("peer_collateral_malformed"),
    };
    if !payload.starts_with('{') || envelope.signature.len() != 128 {
        return Err("peer_collateral_malformed");
    }
    crate::services::memchain::validate_phala_intel_signed_json(payload.as_bytes(), PHALA_PCCS_MAX_BODY_BYTES)?;
    let signature = hex::decode(envelope.signature).map_err(|_| "peer_collateral_malformed")?;
    let mut compact = Vec::with_capacity(payload.len());
    let mut quoted = false;
    let mut escaped = false;
    for byte in payload.bytes() {
        if quoted {
            compact.push(byte);
            if escaped { escaped = false; }
            else if byte == b'\\' { escaped = true; }
            else if byte == b'"' { quoted = false; }
        } else if byte == b'"' {
            quoted = true;
            compact.push(byte);
        } else if !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            compact.push(byte);
        }
    }
    let payload = String::from_utf8(compact).map_err(|_| "peer_collateral_malformed")?;
    Ok((payload, signature))
}

// [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] The injected/offline
// boundary must not bypass the profile enforced by the bounded downloader.
// Length is only framing; QVL still validates P-256 scalars/keys and trust.
pub(crate) fn validate_phala_collateral_signed_inputs(collateral: &dcap_qvl::QuoteCollateralV3) -> Result<(), &'static str> {
    for (payload, signature) in [
        (&collateral.tcb_info, &collateral.tcb_info_signature),
        (&collateral.qe_identity, &collateral.qe_identity_signature),
    ] {
        crate::services::memchain::validate_phala_intel_signed_json(payload.as_bytes(), PHALA_PCCS_MAX_BODY_BYTES)?;
        if signature.len() != 64 { return Err("peer_collateral_malformed"); }
    }
    Ok(())
}

pub(crate) async fn fetch_bounded_phala_collateral(quote: &[u8]) -> Result<dcap_qvl::QuoteCollateralV3, &'static str> {
    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Keep the SDK's
    // quote metadata/parser and offline verifier, not its unbounded downloader.
    guard_phala_quote_allocations(quote)?;
    let mut parsed = dcap_qvl::quote::Quote::parse(quote).map_err(|_| "peer_quote_malformed")?;
    if parsed.report.as_td10().is_none() { return Err("peer_quote_not_tdx"); }
    let target = super::resolve_pinned_peer_http_target(
        reqwest::Url::parse(dcap_qvl::PHALA_PCCS_URL).map_err(|_| "peer_collateral_target_rejected")?,
        std::time::Duration::from_secs(8),
    ).await.map_err(|_| "peer_collateral_target_rejected")?;
    let chain = match parsed.inner_cert_type() {
        5 => std::str::from_utf8(parsed.inner_cert_data()).map_err(|_| "peer_quote_malformed")?.to_owned(),
        2 | 3 => {
            let params = parsed.encrypted_ppid_params().map_err(|_| "peer_quote_malformed")?;
            let query = [("qeid", hex::encode_upper(parsed.qeid())), ("encrypted_ppid", hex::encode_upper(&params.encrypted_ppid)),
                ("cpusvn", hex::encode_upper(params.cpusvn)), ("pcesvn", hex::encode_upper(params.pcesvn.to_le_bytes())),
                ("pceid", hex::encode_upper(params.pceid))];
            let (headers, leaf) = fetch_phala_pccs_resource(&target, "/sgx/certification/v4/pckcert", &query).await?;
            if let Some(tcbm) = headers.get("SGX-TCBm") {
                let tcbm = hex::decode(tcbm.as_bytes()).map_err(|_| "peer_collateral_malformed")?;
                if tcbm.len() != 18 || tcbm[..16] != params.cpusvn || tcbm[16..] != params.pcesvn.to_le_bytes() {
                    return Err("peer_collateral_tcb_mismatch");
                }
            }
            let issuer = phala_pccs_header(&headers, "SGX-PCK-Certificate-Issuer-Chain")?;
            let leaf = String::from_utf8(leaf).map_err(|_| "peer_collateral_malformed")?;
            format!("{leaf}\n{issuer}")
        }
        _ => return Err("peer_quote_malformed"),
    };
    // Public Quote helpers extract FMSPC/CA only from type-5 chains. A fetched
    // chain is metadata input only; the original quote is verified unchanged.
    let certificate = match &mut parsed.auth_data {
        dcap_qvl::quote::AuthData::V3(data) => &mut data.certification_data,
        dcap_qvl::quote::AuthData::V4(data) => &mut data.qe_report_data.certification_data,
    };
    certificate.cert_type = 5;
    certificate.body.data = chain.as_bytes().to_vec();
    let fmspc = hex::encode_upper(parsed.fmspc().map_err(|_| "peer_quote_malformed")?);
    let ca = parsed.ca().map_err(|_| "peer_quote_malformed")?;
    let (headers, pck_crl) = fetch_phala_pccs_resource(&target, "/sgx/certification/v4/pckcrl",
        &[("ca", ca.to_owned()), ("encoding", "der".to_owned())]).await?;
    let pck_crl_issuer_chain = phala_pccs_header(&headers, "SGX-PCK-CRL-Issuer-Chain")?;
    let (headers, tcb) = fetch_phala_pccs_resource(&target, "/tdx/certification/v4/tcb", &[("fmspc", fmspc)]).await?;
    let tcb_info_issuer_chain = phala_pccs_header(&headers, "SGX-TCB-Info-Issuer-Chain")
        .or_else(|_| phala_pccs_header(&headers, "TCB-Info-Issuer-Chain"))?;
    let (tcb_info, tcb_info_signature) = phala_pccs_signed_payload(&tcb, "tcbInfo")?;
    let (headers, qe) = fetch_phala_pccs_resource(&target, "/tdx/certification/v4/qe/identity", &[("update", "standard".to_owned())]).await?;
    let qe_identity_issuer_chain = phala_pccs_header(&headers, "SGX-Enclave-Identity-Issuer-Chain")?;
    let (qe_identity, qe_identity_signature) = phala_pccs_signed_payload(&qe, "enclaveIdentity")?;
    // No unverified certificate-selected CRL URL fallback, including HTTP.
    let (_, root) = fetch_phala_pccs_resource(&target, "/sgx/certification/v4/rootcacrl", &[]).await?;
    let root_ca_crl = hex::decode(&root).map_err(|_| "peer_collateral_malformed")?;
    let collateral = dcap_qvl::QuoteCollateralV3 { pck_crl_issuer_chain, root_ca_crl, pck_crl,
        tcb_info_issuer_chain, tcb_info, tcb_info_signature, qe_identity_issuer_chain,
        qe_identity, qe_identity_signature, pck_certificate_chain: Some(chain) };
    validate_phala_collateral_signed_inputs(&collateral)?;
    Ok(collateral)
}

pub(crate) async fn verify_phala_peer_attestation(
    descriptor: &SignedNodeDescriptor,
    trusted_app_ids: &[String],
    trusted_compose_hashes: &[String],
) -> Result<VerifiedPhalaPeerAttestation, &'static str> {
    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] A caller may
    // cancel its waiter, but a blocking QVL child retains this shared permit.
    with_phala_appraisal_budget(|permit| {
        verify_phala_peer_attestation_owned(descriptor, trusted_app_ids, trusted_compose_hashes, permit, false)
    }).await?
}

// [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] The narrower
// PeerStore-minted target may name a locally pinned non-public relay. General
// promotion/peer callers retain the original public-discovery requirement.
pub(crate) async fn verify_phala_peer_appraisal_target(
    target: &crate::services::peer_store::PhalaPeerAppraisalTarget,
    trusted_app_ids: &[String], trusted_compose_hashes: &[String],
) -> Result<VerifiedPhalaPeerAttestation, &'static str> {
    // [PHALA-APPRAISAL-EGRESS-PIN 2026-10-07 by Codex] Recheck the
    // immutable configured origin before any permit, DNS or evidence request.
    // A descriptor signature does not authorize private-role endpoint rotation.
    if !target.transport_origin_is_permitted() {
        return Err("peer_endpoint_rejected");
    }
    with_phala_appraisal_budget(|permit| {
        verify_phala_peer_attestation_owned(target.descriptor(), trusted_app_ids,
            trusted_compose_hashes, permit, target.permits_private_relay())
    }).await?
}

// [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] Private scope only
// removes public discovery membership, never signatures, role or feature gates.
fn phala_peer_appraisal_descriptor_eligible(
    descriptor: &SignedNodeDescriptor, now: u64, pinned_relay: bool,
) -> bool {
    descriptor.verify_at(now).is_ok()
        && (descriptor.descriptor.policy.public_discovery || pinned_relay)
        && descriptor.descriptor.capabilities.contains(&NodeCapability::ChatRelay)
        && descriptor.descriptor.advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1)
}

// [PHALA-ACI-BOUNDED-VERIFICATION 2026-10-07 by Codex] Peer discovery,
// ACI identity and receipt crypto share one process-wide capacity ceiling.
// The callback moves this permit into any blocking child it starts.
pub(crate) async fn with_phala_appraisal_budget<T, F: std::future::Future<Output = T>>(
    appraisal: impl FnOnce(Arc<tokio::sync::OwnedSemaphorePermit>) -> F,
) -> Result<T, &'static str> {
    let permits = PHALA_APPRAISAL_PERMITS.get_or_init(|| Arc::new(tokio::sync::Semaphore::new(2)));
    let permit = Arc::new(Arc::clone(permits).try_acquire_owned().map_err(|_| "peer_appraisal_busy")?);
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(PHALA_APPRAISAL_TIMEOUT, appraisal(permit))
        .await.map_err(|_| "peer_appraisal_timeout")?;
    // Tokio may poll a ready inner future before observing its timer.
    if started.elapsed() > PHALA_APPRAISAL_TIMEOUT { return Err("peer_appraisal_timeout"); }
    Ok(result)
}

// [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Keep ownership
// with the blocking child even if its async waiter times out or is aborted.
pub(crate) async fn run_phala_appraisal_crypto<T: Send + 'static>(
    permit: Arc<tokio::sync::OwnedSemaphorePermit>,
    verify: impl FnOnce() -> T + Send + 'static,
) -> Result<T, &'static str> {
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        verify()
    }).await.map_err(|_| "peer_appraisal_unavailable")
}

async fn verify_phala_peer_attestation_owned(
    descriptor: &SignedNodeDescriptor,
    trusted_app_ids: &[String],
    trusted_compose_hashes: &[String],
    permit: Arc<tokio::sync::OwnedSemaphorePermit>,
    pinned_relay: bool,
) -> Result<VerifiedPhalaPeerAttestation, &'static str> {
    use aeronyx_core::protocol::discovery::{
        PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V1, PhalaNodeAttestationResponseV1,
    };

    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Count DNS,
    // transport and crypto time from before the challenge, not completion.
    let started = std::time::Instant::now();
    let challenged_at = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_| "clock_unavailable")?.as_secs();
    let descriptor_commitment = aeronyx_core::protocol::discovery::signed_descriptor_commitment_hash(descriptor)
        .map_err(|_| "peer_not_eligible")?;
    if !phala_peer_appraisal_descriptor_eligible(descriptor, challenged_at, pinned_relay)
        || trusted_app_ids.is_empty()
        || trusted_compose_hashes.is_empty()
    {
        return Err("peer_not_eligible");
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or("peer_endpoint_missing")?;
    if !crate::api::reverse_onion_endpoint_supported(endpoint) {
        return Err("peer_endpoint_not_https");
    }
    let mut url = reqwest::Url::parse(endpoint).map_err(|_| "peer_endpoint_invalid")?;
    url.set_path("/api/discovery/phala-attestation");
    url.set_query(None);
    url.set_fragment(None);

    let mut nonce = [0u8; aeronyx_core::protocol::discovery::PHALA_NODE_ATTESTATION_NONCE_BYTES_V1];
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    if nonce == [0; aeronyx_core::protocol::discovery::PHALA_NODE_ATTESTATION_NONCE_BYTES_V1] {
        return Err("nonce_generation_failed");
    }
    // Resolve only the authority/path form accepted by the shared pinned
    // resolver, then add the fixed nonce query after DNS pinning.
    let mut target = crate::api::resolve_pinned_peer_http_target(
        url,
        std::time::Duration::from_secs(8),
    )
    .await
    .map_err(|_| "peer_endpoint_rejected")?;
    target
        .url
        .query_pairs_mut()
        .append_pair("nonce", &hex::encode(nonce));
    let response = target
        .client
        .get(target.url)
        .send()
        .await
        .map_err(|_| "peer_transport_failed")?;
    if !response.status().is_success()
        || response
            .content_length()
            .is_some_and(|length| length > (PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 * 2 + 16_384) as u64)
    {
        return Err("peer_attestation_unavailable");
    }
    let max_body = PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 * 2 + 16_384;
    let mut stream = response.bytes_stream();
    let mut body = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| "peer_transport_failed")?;
        if body.len().saturating_add(chunk.len()) > max_body {
            return Err("peer_attestation_oversized");
        }
        body.extend_from_slice(&chunk);
    }
    // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] No consumer may
    // choose different evidence from duplicate or escaped-alias member names.
    crate::services::memchain::validate_phala_json(&body, max_body)
        .map_err(|_| "peer_attestation_malformed")?;
    let evidence: PhalaNodeAttestationResponseV1 =
        serde_json::from_slice(&body).map_err(|_| "peer_attestation_malformed")?;
    let node_id = descriptor.node_id();
    evidence
        .validate_for(&node_id, &nonce, None)
        .map_err(|_| "peer_attestation_binding_invalid")?;
    if evidence.attestation_format != PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V1 {
        return Err("peer_attestation_format_unsupported");
    }

    let attestation_bytes = hex::decode(&evidence.attestation)
        .map_err(|_| "peer_attestation_encoding_invalid")?;
    let (stack_report_data, aci_evidence) = dstack_v1_aci_evidence(&attestation_bytes)
        .map_err(|_| "peer_attestation_format_invalid")?;
    if hex::encode(stack_report_data) != evidence.expected_report_data {
        return Err("peer_stack_report_data_mismatch");
    }
    let quote = aci_verify::quote::quote_bytes(&aci_evidence)
        .map_err(|_| "peer_quote_malformed")?;
    let collateral = fetch_bounded_phala_collateral(&quote).await?;
    let trusted_app_ids = trusted_app_ids.to_vec();
    let trusted_compose_hashes = trusted_compose_hashes.to_vec();
    // A dropped JoinHandle does not cancel blocking crypto. Keep its permit
    // inside the child until it really exits; never monopolize an async worker.
    run_phala_appraisal_crypto(permit, move || {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "clock_unavailable")?
            .as_secs();
        if now < challenged_at || started.elapsed() > PHALA_APPRAISAL_TIMEOUT {
            return Err("peer_appraisal_timeout");
        }
        // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Match the
        // offline ACI boundary immediately before the only verifier path.
        validate_phala_collateral_signed_inputs(&collateral)?;
        let verified = dcap_qvl::verify::rustcrypto::verify(&quote, &collateral, now)
            .map_err(|_| "peer_quote_invalid")?;
        if verified.status != "UpToDate" {
            return Err("peer_tcb_not_up_to_date");
        }
        let td_report = verified.report.as_td10().ok_or("peer_quote_not_tdx")?;
        let expected_report_data = aeronyx_core::protocol::phala_node_attestation_report_data_v1(
            &node_id, &nonce,
        );
        // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] The official
        // helper receives the digest and itself enforces the full padded slot.
        aci_verify::quote::quote_binds_report_data(
            &aci_evidence, &td_report.report_data, expected_report_data,
        ).map_err(|_| "peer_quote_report_data_mismatch")?;
        let event_log = aci_verify::dstack::verify_dstack_event_log(
            &aci_evidence, Some(&td_report.rt_mr3),
        ).map_err(|_| "peer_event_log_invalid")?;
        let compose_hash = aci_verify::dstack::verify_dstack_compose_measurement(
            &aci_evidence, &event_log,
        ).map_err(|_| "peer_compose_measurement_invalid")?;
        let compose_hash = format!("sha256:{compose_hash}");
        let app_id = aci_verify::dstack::dstack_app_id(&event_log)
            .map_err(|_| "peer_app_id_missing")?;
        let app_id = canonical_phala_app_id_pin(&app_id).ok_or("peer_app_id_invalid")?;
        if !trusted_compose_hashes.iter().any(|allowed| allowed == &compose_hash)
            || !trusted_app_ids.iter().any(|allowed| allowed == &app_id)
        {
            return Err("peer_measurement_untrusted");
        }
        let verified_at = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_| "clock_unavailable")?.as_secs();
        if verified_at < now || started.elapsed() > PHALA_APPRAISAL_TIMEOUT {
            return Err("peer_appraisal_timeout");
        }
        Ok(VerifiedPhalaPeerAttestation { descriptor_commitment, challenged_at, verified_at, started })
    }).await?
}

// [PHALA-APP-ID-PIN-FORMAT 2026-10-06 by Codex] ACI's dstack helper returns
// measured app-id bytes; operator pins use its canonical `0x` lowercase-hex
// representation, never lossy UTF-8 or a display label.
fn canonical_phala_app_id_pin(app_id: &[u8]) -> Option<String> {
    (!app_id.is_empty() && app_id.len() <= 64)
        .then(|| format!("0x{}", hex::encode(app_id)))
}

#[derive(Clone, Copy)]
struct OnionTerminalSubsetPolicy<'a> {
    route_policy: &'a DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
}

/// Full-identity membership used to sanitize prefix-only public status rows.
///
/// [PUBLIC-DISCOVERY-PROJECTION 2026-09-01 by Codex] A short prefix is never
/// treated as identity evidence on its own. A prefix is publishable only when
/// it maps to exactly one currently valid full public node id and exactly one
/// matching public row in the complete peer summary. Collisions, expired rows,
/// private rows, and projection-only unknowns therefore fail closed.
struct PublicStatusProjectionMembership {
    unambiguous_public_prefixes: HashSet<String>,
}

impl PublicStatusProjectionMembership {
    fn from_status(status: &PeerStoreStatus, public_descriptors: &[SignedNodeDescriptor]) -> Self {
        let public_node_ids: HashSet<[u8; 32]> = public_descriptors
            .iter()
            .map(SignedNodeDescriptor::node_id)
            .collect();
        let mut public_prefix_counts: HashMap<String, usize> = HashMap::new();
        for node_id in public_node_ids {
            *public_prefix_counts
                .entry(hex::encode(&node_id[..4]))
                .or_default() += 1;
        }

        let mut summary_prefix_counts: HashMap<String, (usize, usize)> = HashMap::new();
        for peer in &status.peer_summary.peers {
            let counts = summary_prefix_counts
                .entry(peer.node_id_prefix.clone())
                .or_default();
            counts.0 += 1;
            if peer.public_discovery {
                counts.1 += 1;
            }
        }

        let unambiguous_public_prefixes = public_prefix_counts
            .into_iter()
            .filter_map(|(prefix, full_public_id_count)| {
                let summary_counts = summary_prefix_counts.get(&prefix).copied();
                (full_public_id_count == 1 && summary_counts == Some((1, 1))).then_some(prefix)
            })
            .collect();
        Self {
            unambiguous_public_prefixes,
        }
    }

    fn permits_prefix(&self, prefix: &str) -> bool {
        self.unambiguous_public_prefixes.contains(prefix)
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    mod gossip_ingress;
    mod onion_candidates;
    mod other;
    mod status_card;

    use super::*;
    use crate::services::DiscoveryEndpointAttestationInboxConfig;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        directory_block_range_response_signing_bytes, encode_directory_sync_message,
        encode_route_domain_attestation_certificate, DirectoryCommitmentBlockV1,
        DirectoryDescriptorCommitmentV1, DirectoryDescriptorInclusionProofV1, DirectorySyncMessage,
        RouteDomainAttestationCertificateV1, RouteDomainAttestationV1,
        AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    };
    use aeronyx_core::protocol::{
        canonical_public_endpoint_commitment, discovery_endpoint_evidence_commitment_v1,
        DiscoveryEndpointAttestationPurposeV1, DiscoveryEndpointChallengeV1,
        DiscoveryEndpointProofV1, NodeCapability, NodeCapacity, NodeDescriptor, NodePolicy,
        SignedNodeDescriptor,
    };
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use tempfile::TempDir;
    use tower::ServiceExt;

    // [PHALA-PINNED-RELAY-APPRAISAL 2026-10-07 by Codex] Authored only:
    // private selection scope changes membership, not signature/role checks.
    #[test]
    fn phala_private_appraisal_scope_retains_signed_role_and_expiry_gates() {
        let now = now_secs();
        let key = IdentityKeyPair::from_bytes(&[185; 32]).unwrap();
        let mut base = NodeDescriptor::new(key.public_key_bytes(), 1, now - 1, now + 900, "test");
        base.public_endpoint = Some("https://private-relay.example".into());
        base.policy.public_discovery = false;
        base.capabilities = vec![NodeCapability::ChatRelay];
        let body = base.clone().with_protocol_features([NodeProtocolFeature::PhalaNodeAttestationV1]);
        let descriptor = SignedNodeDescriptor::sign(body.clone(), &key).unwrap();
        assert!(!phala_peer_appraisal_descriptor_eligible(&descriptor, now, false));
        assert!(phala_peer_appraisal_descriptor_eligible(&descriptor, now, true));
        assert!(!phala_peer_appraisal_descriptor_eligible(&descriptor, now + 901, true));
        let legacy = SignedNodeDescriptor::sign(base, &key).unwrap();
        assert!(!phala_peer_appraisal_descriptor_eligible(&legacy, now, true));
        let mut no_role = body;
        no_role.capabilities.clear();
        let no_role = SignedNodeDescriptor::sign(no_role, &key).unwrap();
        assert!(!phala_peer_appraisal_descriptor_eligible(&no_role, now, true));
        let mut tampered = descriptor;
        tampered.descriptor.sequence += 1;
        assert!(!phala_peer_appraisal_descriptor_eligible(&tampered, now, true));
    }

    // [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex] Authored
    // only: deterministic age projections, not cryptographic quote acceptance.
    #[test]
    fn phala_appraisal_time_rejects_rollback_expiry_and_projection_overflow() {
        let started = std::time::Duration::ZERO;
        assert!(phala_appraisal_effective_time(100, 101, started, 300).unwrap() >= 101);
        assert!(phala_appraisal_effective_time(100, 99, started, 300).is_none());
        assert!(phala_appraisal_effective_time(100, 401, started, 300).is_none());
        let elapsed = std::time::Duration::from_secs(11);
        assert!(phala_appraisal_effective_time(100, 101, elapsed, 10).is_none());
        assert!(phala_appraisal_effective_time(u64::MAX - 5, u64::MAX - 4, elapsed, 300).is_none());
        assert!(phala_appraisal_effective_time(100, 101, elapsed, 300).unwrap() >= 111);
    }

    // [PHALA-APPRAISAL-OBSERVATION-FLOOR 2026-10-07 by Codex]
    #[test]
    fn phala_verified_result_cannot_publish_after_monotonic_descriptor_expiry() {
        let now = now_secs();
        let descriptor = signed_routeable_chat_descriptor(1, now + 10, "https://phala-peer.example");
        let mut result = VerifiedPhalaPeerAttestation::synthetic_for_test(&descriptor, now, now);
        assert!(result.cache_time_for(&descriptor, now, 300).is_some());
        assert!(result.cache_time_for(&descriptor, now - 1, 300).is_none());
        result.started = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(11)).unwrap();
        assert!(descriptor.verify_at(now + 1).is_ok());
        assert!(result.cache_time_for(&descriptor, now + 1, 300).is_none());
        let still_valid = signed_routeable_chat_descriptor(1, now + 300, "https://phala-peer.example");
        let mut result = VerifiedPhalaPeerAttestation::synthetic_for_test(&still_valid, now, now);
        result.started = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(11)).unwrap();
        assert!(still_valid.verify_at(now + 1).is_ok());
        assert!(result.cache_time_for(&still_valid, now + 1, 300).is_none());
        assert!(result.cache_time_for(&still_valid, now + 60, 300).is_some());
    }

    // [PHALA-BOUNDED-PEER-APPRAISAL 2026-10-07 by Codex] Synthetic
    // framing only, never cryptographic acceptance or live TEE evidence.
    fn framed_phala_quote(version: u16, body_kind: u16) -> (Vec<u8>, Vec<(usize, usize)>) {
        let size = match body_kind { 1 => 384, 2 => 584, 3 => 648, _ => unreachable!() };
        let mut quote = vec![0; 48];
        quote[..2].copy_from_slice(&version.to_le_bytes());
        quote[4..8].copy_from_slice(&(if body_kind == 1 { 0u32 } else { 0x81u32 }).to_le_bytes());
        if version == 5 {
            quote.extend_from_slice(&body_kind.to_le_bytes());
            quote.extend_from_slice(&(size as u32).to_le_bytes());
        }
        quote.resize(quote.len() + size, 0);
        let auth_length_at = quote.len();
        let mut inner = vec![0; if version == 3 { 576 } else { 448 }];
        let qe_length_at = inner.len();
        inner.extend_from_slice(&0u16.to_le_bytes());
        inner.extend_from_slice(&5u16.to_le_bytes());
        let certificate_length_at = inner.len();
        inner.extend_from_slice(&3u32.to_le_bytes());
        inner.extend_from_slice(b"pem");
        let inner_at;
        let mut lengths = vec![(auth_length_at, 4)];
        let auth = if version == 3 {
            inner_at = auth_length_at + 4;
            inner
        } else {
            let mut auth = vec![0; 128];
            auth.extend_from_slice(&6u16.to_le_bytes());
            lengths.push((auth_length_at + 4 + auth.len(), 4));
            auth.extend_from_slice(&(inner.len() as u32).to_le_bytes());
            inner_at = auth_length_at + 4 + auth.len();
            auth.extend_from_slice(&inner);
            auth
        };
        lengths.push((inner_at + qe_length_at, 2));
        lengths.push((inner_at + certificate_length_at, 4));
        quote.extend_from_slice(&(auth.len() as u32).to_le_bytes());
        quote.extend_from_slice(&auth);
        (quote, lengths)
    }

    #[test]
    fn phala_quote_guard_bounds_every_sdk_vector_before_allocation() {
        for (version, kind) in [(3, 1), (4, 1), (4, 2), (5, 1), (5, 2), (5, 3)] {
            let (quote, lengths) = framed_phala_quote(version, kind);
            assert!(guard_phala_quote_allocations(&quote).is_ok());
            assert!(dcap_qvl::quote::Quote::parse(&quote).is_ok());
            for (at, width) in lengths {
                let mut corrupted = quote.clone();
                corrupted[at..at + width].fill(0xff);
                assert!(guard_phala_quote_allocations(&corrupted).is_err());
            }
            for cut in 0..quote.len() {
                assert!(guard_phala_quote_allocations(&quote[..cut]).is_err());
            }
            let mut trailing = quote.clone();
            trailing.push(0);
            assert!(guard_phala_quote_allocations(&trailing).is_err());
            if version == 5 {
                let mut wrong_size = quote.clone();
                wrong_size[50..54].fill(0xff);
                assert!(guard_phala_quote_allocations(&wrong_size).is_err());
            }
        }
        let (mut unsupported, _) = framed_phala_quote(4, 2);
        unsupported[..2].copy_from_slice(&6u16.to_le_bytes());
        assert!(guard_phala_quote_allocations(&unsupported).is_err());
        assert!(guard_phala_quote_allocations(&vec![0; PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 + 1]).is_err());
    }

    #[test]
    fn phala_collateral_target_cannot_follow_certificate_urls() {
        let mut target = super::super::PinnedPeerHttpTarget {
            client: reqwest::Client::new(),
            url: reqwest::Url::parse(dcap_qvl::PHALA_PCCS_URL).unwrap(),
        };
        let path = "/tdx/certification/v4/tcb";
        assert!(phala_pccs_url(&target, "/sgx/certification/v4/rootcacrl", &[]).unwrap().query().is_none());
        let value = "x&url=http://127.0.0.1/private".to_owned();
        let url = phala_pccs_url(&target, path, &[("fmspc", value.clone())]).unwrap();
        assert!(super::super::reverse_onion_same_origin(url.as_str(), dcap_qvl::PHALA_PCCS_URL));
        assert_eq!(url.path(), path);
        let pairs = url.query_pairs().collect::<Vec<_>>();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "fmspc");
        assert_eq!(pairs[0].1, value);
        for rejected in ["http://127.0.0.1/crl", "//127.0.0.1/crl", "/unknown", "/tdx/certification/v4/tcb/../crl"] {
            assert!(phala_pccs_url(&target, rejected, &[]).is_err());
        }
        for rejected in ["http://pccs.phala.network", "https://pccs.phala.network:444", "https://127.0.0.1"] {
            target.url = reqwest::Url::parse(rejected).unwrap();
            assert!(phala_pccs_url(&target, path, &[]).is_err());
        }
    }

    #[test]
    fn phala_collateral_fields_are_bounded_and_structured() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("issuer", "line%0Asecond%20line".parse().unwrap());
        assert_eq!(phala_pccs_header(&headers, "issuer").unwrap(), "line\nsecond line");
        assert!(phala_pccs_header(&headers, "missing").is_err());
        headers.insert("issuer", "%ff".parse().unwrap());
        assert!(phala_pccs_header(&headers, "issuer").is_err());
        headers.insert("issuer", "a".repeat(PHALA_PCCS_MAX_HEADER_BYTES + 1).parse().unwrap());
        assert!(phala_pccs_header(&headers, "issuer").is_err());
        let mut payload = serde_json::json!({"tcbInfo": {"version": 3}, "signature": "ab".repeat(64)});
        assert_eq!(phala_pccs_signed_payload(&serde_json::to_vec(&payload).unwrap(), "tcbInfo").unwrap().1, vec![0xab; 64]);
        for invalid in [serde_json::json!(null), serde_json::json!("invalid")] {
            payload["tcbInfo"] = invalid;
            assert!(phala_pccs_signed_payload(&serde_json::to_vec(&payload).unwrap(), "tcbInfo").is_err());
        }
        payload["tcbInfo"] = serde_json::json!({});
        for invalid in ["ab".repeat(63), "gg".repeat(64)] {
            payload["signature"] = invalid.into();
            assert!(phala_pccs_signed_payload(&serde_json::to_vec(&payload).unwrap(), "tcbInfo").is_err());
        }
        assert!(phala_pccs_signed_payload(&vec![b' '; PHALA_PCCS_MAX_BODY_BYTES + 1], "tcbInfo").is_err());
    }

    // [PHALA-COLLATERAL-SIGNED-BYTES 2026-10-07 by Codex] Authored only:
    // byte preservation tests, not synthetic claims of valid Intel signatures.
    #[test]
    fn phala_collateral_keeps_supported_signed_lexemes_and_only_removes_json_whitespace() {
        // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Literal
        // backslash-u is not a Unicode escape; supported lexemes stay exact.
        let raw = r#"{ "z" : "\\u0061/", "space" : " a b ",
            "slashes" : "\\\" quoted \\\\", "number" : 100,
            "array" : [ true, null, { "x" : "\t\n" } ] }"#;
        let expected = r#"{"z":"\\u0061/","space":" a b ","slashes":"\\\" quoted \\\\","number":100,"array":[true,null,{"x":"\t\n"}]}"#;
        for field in ["tcbInfo", "enclaveIdentity"] {
            let body = format!("{{\"{field}\":{raw},\"signature\":\"{}\"}}", "ab".repeat(64));
            let (signed, signature) = phala_pccs_signed_payload(body.as_bytes(), field).unwrap();
            assert_eq!(signed, expected);
            assert_eq!(signature, vec![0xab; 64]);
            // Supported spellings agree for this fixture; this is not proof
            // that reconstructing arbitrary JSON preserves signed bytes.
            let reconstructed: serde_json::Value = serde_json::from_str(raw).unwrap();
            assert_eq!(signed, reconstructed.to_string());
        }
    }

    // [PHALA-INTEL-LEXICAL-PROFILE 2026-10-08 by Codex] Authored only:
    // valid JSON spellings must not reach either verifier through PCCS.
    #[test]
    fn phala_collateral_pccs_profile_rejects_unsupported_signed_subtrees() {
        for raw in [
            r#"{"x":"\u0008"}"#, r#"{"\u0078":"ok"}"#,
            r#"{"a":[{"x":"a\/b"}]}"#, r#"{"x":1.00e+02}"#,
            r#"{"x":-0}"#, r#"{"x":9007199254740992}"#,
        ] {
            for field in ["tcbInfo", "enclaveIdentity"] {
                let body = format!("{{\"{field}\":{raw},\"signature\":\"{}\"}}", "ab".repeat(64));
                assert!(crate::services::memchain::validate_phala_json(body.as_bytes(), PHALA_PCCS_MAX_BODY_BYTES).is_ok());
                assert_eq!(phala_pccs_signed_payload(body.as_bytes(), field), Err("peer_collateral_unsupported_form"));
            }
        }
        // Unrelated envelope members do not impose this profile on general
        // JSON. Duplicate names there are still rejected by the broad guard.
        let body = format!(r#"{{"tcbInfo":{{}},"extra":"\u0061\/","signature":"{}"}}"#, "ab".repeat(64));
        assert_eq!(phala_pccs_signed_payload(body.as_bytes(), "tcbInfo").unwrap().0, "{}");
    }

    #[test]
    fn phala_collateral_rejects_ambiguous_envelopes_before_qvl() {
        // [PHALA-COLLATERAL-SIGNED-BYTES 2026-10-07 by Codex] No known
        // field may select a different signed object through duplicate keys.
        let signature = "ab".repeat(64);
        for body in [
            format!(r#"{{"tcbInfo":{{}},"tcbInfo":{{"version":3}},"signature":"{signature}"}}"#),
            format!(r#"{{"tcbInfo":{{}},"\u0074cbInfo":{{}},"signature":"{signature}"}}"#),
            format!(r#"{{"tcbInfo":{{}},"signature":"{signature}","signature":"{signature}"}}"#),
            format!(r#"{{"tcbInfo":{{}},"enclaveIdentity":{{}},"signature":"{signature}"}}"#),
            format!(r#"{{"tcbInfo":[],"signature":"{signature}"}}"#),
            // [PHALA-UNIQUE-EVIDENCE-JSON 2026-10-07 by Codex] RawValue
            // preserves input, so its nested/unknown members need the guard too.
            format!(r#"{{"tcbInfo":{{"x":1,"x":2}},"signature":"{signature}"}}"#),
            format!(r#"{{"tcbInfo":{{}},"unknown":{{"x":1,"\u0078":2}},"signature":"{signature}"}}"#),
            format!(r#"{{"tcbInfo":{{}},"signature":"{signature}"}} trailing"#),
        ] {
            assert!(phala_pccs_signed_payload(body.as_bytes(), "tcbInfo").is_err());
        }
        let body = format!(r#"{{"tcbInfo":{{}},"signature":"{signature}"}}"#);
        for field in ["enclaveIdentity", "unknown"] {
            assert!(phala_pccs_signed_payload(body.as_bytes(), field).is_err());
        }
    }

    #[tokio::test]
    async fn phala_appraisal_cancelled_waiter_retains_blocking_child_permit() {
        let permits = Arc::new(tokio::sync::Semaphore::new(1));
        let permit = Arc::new(Arc::clone(&permits).try_acquire_owned().unwrap());
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(run_phala_appraisal_crypto(permit, move || {
            let _ = started_tx.send(());
            let _ = release_rx.blocking_recv();
            Ok::<(), &'static str>(())
        }));
        started_rx.await.unwrap();
        assert!(Arc::clone(&permits).try_acquire_owned().is_err());
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(Arc::clone(&permits).try_acquire_owned().is_err());
        release_tx.send(()).unwrap();
        let _recovered = tokio::time::timeout(std::time::Duration::from_secs(2), permits.acquire_owned()).await.unwrap().unwrap();
    }

    // [PHALA-NODE-ATTESTATION-API 2026-10-06 by Codex] Calibrate the nonce
    // boundary with accepted, malformed, predictable, and wrong-length inputs.
    #[test]
    fn phala_node_attestation_nonce_is_exact_hex_and_nonzero() {
        assert_eq!(
            parse_phala_attestation_nonce(&"ab".repeat(32)),
            Some([0xab; 32])
        );
        assert!(parse_phala_attestation_nonce(&"00".repeat(32)).is_none());
        assert!(parse_phala_attestation_nonce(&"gg".repeat(32)).is_none());
        assert!(parse_phala_attestation_nonce(&"ab".repeat(31)).is_none());
    }

    // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] Reject zero,
    // malformed, and wrong-length recipient public identities.
    #[test]
    fn phala_recipient_node_id_is_exact_nonzero_hex() {
        assert_eq!(parse_phala_recipient_node_id(&"ab".repeat(32)), Some([0xab; 32]));
        assert!(parse_phala_recipient_node_id(&"00".repeat(32)).is_none());
        assert!(parse_phala_recipient_node_id(&"gg".repeat(32)).is_none());
        assert!(parse_phala_recipient_node_id(&"ab".repeat(31)).is_none());
    }

    // [PHALA-QUEUE-RECOVERY-GATE 2026-10-06 by Codex] Calibrate quote
    // authority against live, recovery-only, and ambiguous queue configs.
    #[test]
    fn phala_private_recipient_quote_policy_requires_exact_queue_pin() {
        let discovery = crate::config::DiscoveryConfig::default();
        let pinned = "ab".repeat(32);
        let mut queue = ReverseOnionQueueConfig::default();
        queue.enabled = true;
        queue.recipient_node_ids = vec![pinned];
        let allowed = DiscoveryApiPolicy::from_config(&discovery)
            .with_phala_private_recipient_queue(&queue);
        assert_eq!(allowed.phala_private_recipient_node_id, Some([0xab; 32]));
        assert!(phala_private_recipient_pin_matches(&allowed, &[0xab; 32]));
        assert!(!phala_private_recipient_pin_matches(&allowed, &[0xcd; 32]));

        queue.recovery_only = true;
        let recovery = DiscoveryApiPolicy::from_config(&discovery)
            .with_phala_private_recipient_queue(&queue);
        assert_eq!(recovery.phala_private_recipient_node_id, None);
        assert!(!phala_private_recipient_pin_matches(&recovery, &[0xab; 32]));

        queue.recovery_only = false;
        queue.recipient_node_ids.push("cd".repeat(32));
        let ambiguous = DiscoveryApiPolicy::from_config(&discovery)
            .with_phala_private_recipient_queue(&queue);
        assert_eq!(ambiguous.phala_private_recipient_node_id, None);
        assert!(!phala_private_recipient_pin_matches(&ambiguous, &[0xab; 32]));
    }

    // [PHALA-QUOTE-ROUTE-GATE 2026-10-06 by Codex] Exercise the mounted
    // handler, not just its policy helper: recovery-only must reject before
    // attempting the configured guest socket.
    #[tokio::test]
    async fn phala_recovery_queue_rejects_recipient_quote_before_guest_io() {
        let identity = IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap();
        let mut discovery = crate::config::DiscoveryConfig::default();
        discovery.phala_attestation_socket_path =
            Some("/var/run/aeronyx/no-such-dstack.sock".into());
        let mut queue = ReverseOnionQueueConfig::default();
        queue.enabled = true;
        queue.recovery_only = true;
        queue.recipient_node_ids = vec![hex::encode([0x76; 32])];
        let policy = DiscoveryApiPolicy::from_config(&discovery)
            .with_phala_private_recipient_queue(&queue);
        let app = build_discovery_router_with_local_entry(
            Arc::new(PeerStore::new()),
            policy,
            DiscoveryLocalCapabilityStatus::default(),
            None,
            identity.public_key_bytes(),
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!(
                        "/api/discovery/phala-attestation?nonce={}&recipient_node_id={}",
                        "77".repeat(32),
                        hex::encode([0x76; 32]),
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] Preserve the
    // existing node-only response shape; recipient binding is additive.
    #[test]
    fn phala_attestation_response_keeps_v1_shape_and_names_bound_recipient() {
        let base = PhalaNodeAttestationResponseV1 {
            contract_version: PHALA_NODE_ATTESTATION_CONTRACT_VERSION_V1.into(),
            node_id: "11".repeat(32),
            recipient_node_id: None,
            authorization_sha256: None,
            nonce: "22".repeat(32),
            expected_report_data: "33".repeat(64),
            attestation_format: PHALA_NODE_ATTESTATION_FORMAT_DSTACK_V1.into(),
            attestation: "aa".into(),
            verification: PHALA_NODE_ATTESTATION_VERIFICATION_NOTE_V1.into(),
        };
        let legacy = serde_json::to_value(&base).unwrap();
        assert_eq!(legacy["contract_version"], "phala_node_attestation.v1");
        assert!(legacy.get("recipient_node_id").is_none());

        let bound = PhalaNodeAttestationResponseV1 {
            contract_version: PHALA_PRIVATE_RECIPIENT_ATTESTATION_CONTRACT_VERSION_V1.into(),
            recipient_node_id: Some("44".repeat(32)),
            authorization_sha256: Some("55".repeat(32)),
            ..base
        };
        let bound = serde_json::to_value(bound).unwrap();
        assert_eq!(
            bound["contract_version"],
            "phala_private_recipient_attestation.v1",
        );
        assert_eq!(bound["recipient_node_id"], "44".repeat(32));
        assert_eq!(bound["authorization_sha256"], "55".repeat(32));
    }

    // [PHALA-ATTESTATION-NO-STORE 2026-10-06 by Codex] Authored, not run:
    // success and upstream-error responses must not be cached for a nonce.
    #[tokio::test]
    async fn phala_attestation_response_is_non_cacheable() {
        let app = Router::new()
            .route("/quote", get(|| async { StatusCode::OK }))
            .route("/error", get(|| async { StatusCode::BAD_GATEWAY }))
            .layer(middleware::from_fn(no_store_phala_attestation_response));
        for (path, status) in [
            ("/quote", StatusCode::OK),
            ("/error", StatusCode::BAD_GATEWAY),
        ] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .body(axum::body::Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(
                response.headers().get(axum::http::header::CACHE_CONTROL)
                    .and_then(|value| value.to_str().ok()),
                Some("no-store, private"),
            );
            assert_eq!(
                response.headers().get(axum::http::header::PRAGMA)
                    .and_then(|value| value.to_str().ok()),
                Some("no-cache"),
            );
        }
    }

    #[test]
    fn dstack_response_parser_rejects_ambiguous_http_framing() {
        let report_data = vec![0x11; 64];
        let expected_report_data = hex::encode(&report_data);
        // [PHALA-DSTACK-MESSAGEPACK-BYTES 2026-10-06 by Codex] Emit actual
        // MessagePack `bin` tokens for all dstack byte fields, matching the
        // guest API wire type rather than JSON-like integer arrays.
        let attestation = rmp_serde::to_vec_named(&DstackVersionedAttestationEnvelope {
            version: 1,
            platform: DstackVersionedAttestationPlatform {
                kind: "tdx".into(),
                data: DstackVersionedAttestationPlatformData {
                    quote: DstackByteField(vec![0xaa, 0xbb]),
                    event_log: vec![DstackVersionedTdxEvent {
                        imr: 3,
                        event_type: 134_217_729,
                        digest: DstackByteField(vec![1, 2]),
                        event: "app-id".into(),
                        event_payload: DstackByteField(vec![3, 4]),
                    }],
                },
            },
            stack: DstackVersionedAttestationStack {
                kind: "dstack".into(),
                data: DstackVersionedAttestationStackData {
                    report_data: DstackByteField(report_data.clone()),
                    config: "{}".into(),
                },
            },
        })
        .unwrap();
        let body = serde_json::to_vec(&serde_json::json!({
            "attestation": hex::encode(&attestation)
        }))
        .unwrap();
        let valid = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            std::str::from_utf8(&body).unwrap()
        );
        assert_eq!(
            parse_dstack_http_response(valid.as_bytes()).unwrap().status,
            200
        );
        assert_eq!(
            parse_dstack_v1_attestation(&body, &expected_report_data).unwrap(),
            attestation,
        );
        // [PHALA-DSTACK-V1-ACI-EVIDENCE 2026-10-06 by Codex] Calibrate the
        // MessagePack-to-ACI conversion and failure cases without running it.
        let (decoded_report_data, aci_evidence) = dstack_v1_aci_evidence(&attestation).unwrap();
        assert_eq!(decoded_report_data, report_data);
        assert_eq!(aci_evidence["quote"], "aabb");
        assert_eq!(aci_evidence["quote_report_data"], expected_report_data);
        assert_eq!(aci_evidence["app_compose"], "{}");
        let events: Vec<serde_json::Value> = serde_json::from_str(
            aci_evidence["event_log"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(events[0]["digest"], "0102");
        assert_eq!(events[0]["event_payload"], "0304");
        assert_eq!(canonical_phala_app_id_pin(&[0x12, 0xab]), Some("0x12ab".into()));
        assert_eq!(canonical_phala_app_id_pin(&[]), None);
        let legacy_byte_sequence = rmp_serde::to_vec(&vec![1_u8, 2]).unwrap();
        assert_eq!(
            rmp_serde::from_slice::<DstackByteField>(&legacy_byte_sequence)
                .unwrap()
                .0,
            vec![1, 2],
        );
        assert!(parse_dstack_v1_attestation(&body, &"22".repeat(64)).is_err());

        // [PHALA-DSTACK-V1-FORMAT-CHECK 2026-10-06 by Codex] A v0 SCALE
        // prefix and a wrong envelope version must not be mislabeled as v1.
        assert!(parse_dstack_v1_attestation(
            br#"{"attestation":"00"}"#,
            &expected_report_data,
        )
        .is_err());
        // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] json! accepts
        // the byte-vector expression, not Rust array-repeat syntax.
        let wrong_version = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 2,
            "platform": { "kind": "tdx", "data": {} },
            "stack": { "kind": "dstack", "data": { "report_data": vec![17_u8; 64] } }
        }))
        .unwrap();
        let wrong_version = serde_json::to_vec(&serde_json::json!({
            "attestation": hex::encode(wrong_version)
        }))
        .unwrap();
        assert!(parse_dstack_v1_attestation(&wrong_version, &expected_report_data).is_err());
        let unsupported_platform = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "platform": { "kind": "gcp-tdx", "data": {
                "quote": [0xaa], "event_log": [], "tpm_quote": {}
            } },
            "stack": { "kind": "dstack", "data": {
                "report_data": report_data, "runtime_events": [], "config": "{}"
            } }
        })).unwrap();
        assert!(dstack_v1_aci_evidence(&unsupported_platform).is_err());
        let mut trailing = attestation.clone();
        trailing.push(0);
        assert!(dstack_v1_aci_evidence(&trailing).is_err());

        let duplicate_length = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body.len(),
            std::str::from_utf8(&body).unwrap()
        );
        assert!(parse_dstack_http_response(duplicate_length.as_bytes()).is_err());
        let chunked = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n";
        assert!(parse_dstack_http_response(chunked).is_err());

        let expanded_body = format!(
            r#"{{"attestation":"{}"}}"#,
            "00".repeat(PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1)
        );
        let expanded_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            expanded_body.len(), expanded_body
        );
        assert!(expanded_response.len() <= PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1);
        assert!(parse_dstack_http_response(expanded_response.as_bytes()).is_ok());

        let oversized_response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1,
            "x".repeat(PHALA_NODE_ATTESTATION_MAX_GUEST_HTTP_RESPONSE_BYTES_V1)
        );
        assert!(parse_dstack_http_response(oversized_response.as_bytes()).is_err());
    }

    // [PHALA-DSTACK-V0-FALLBACK 2026-10-06 by Codex] Legacy responses are
    // accepted only with a valid quote shape and exact padded report-data echo.
    #[test]
    fn dstack_legacy_quote_requires_exact_report_data_and_mount_missing_404() {
        let report_data = "ab".repeat(64);
        let body = format!(
            r#"{{"quote":"aa","event_log":[],"report_data":"{report_data}"}}"#
        );
        assert_eq!(
            parse_dstack_v0_quote(body.as_bytes(), &report_data).unwrap(),
            body.as_bytes()
        );
        assert!(parse_dstack_v0_quote(body.as_bytes(), &"cd".repeat(64)).is_err());
        let empty_quote = format!(
            r#"{{"quote":"0x","event_log":[],"report_data":"{report_data}"}}"#
        );
        assert!(parse_dstack_v0_quote(empty_quote.as_bytes(), &report_data).is_err());
        let oversized_body = format!(
            "{}{}",
            body,
            " ".repeat(PHALA_NODE_ATTESTATION_MAX_EVIDENCE_BYTES_V1 + 1 - body.len())
        );
        assert!(parse_dstack_v0_quote(oversized_body.as_bytes(), &report_data).is_err());
        let prefixed_quote = format!(
            r#"{{"quote":"0xaa","event_log":[],"report_data":"0x{report_data}"}}"#
        );
        assert!(parse_dstack_v0_quote(prefixed_quote.as_bytes(), &report_data).is_ok());

        let missing = DstackHttpResponse {
            status: 404,
            content_type: Some("text/plain; charset=utf-8".into()),
            body: b"not found".to_vec(),
        };
        assert!(is_dstack_v1_mount_missing(&missing));
        let method_error = DstackHttpResponse {
            status: 404,
            content_type: Some("application/json".into()),
            body: br#"{"error":"Service not found: Attest"}"#.to_vec(),
        };
        assert!(!is_dstack_v1_mount_missing(&method_error));
        let other_json_404 = DstackHttpResponse {
            status: 404,
            content_type: Some("application/json".into()),
            body: br#"{"error":"temporary failure"}"#.to_vec(),
        };
        assert!(!is_dstack_v1_mount_missing(&other_json_404));
        let untyped_404 = DstackHttpResponse {
            status: 404,
            content_type: None,
            body: b"not found".to_vec(),
        };
        assert!(!is_dstack_v1_mount_missing(&untyped_404));
    }

    fn signed_descriptor() -> aeronyx_core::protocol::SignedNodeDescriptor {
        let kp = IdentityKeyPair::generate();
        let now = now_secs();
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 64,
            max_bps: None,
            max_pps: None,
        };
        aeronyx_core::protocol::SignedNodeDescriptor::sign(descriptor, &kp).unwrap()
    }

    fn open_node_descriptor(
        identity: &IdentityKeyPair,
        sequence: u64,
        endpoint: &str,
    ) -> SignedNodeDescriptor {
        let now = now_secs();
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            sequence,
            now.saturating_sub(1),
            now + 600,
            "1.0.0+anpf1-brsr1",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, identity).unwrap()
    }

    async fn post_open_node(app: Router, body: Vec<u8>) -> axum::response::Response {
        app.oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/join")
                .header("content-type", "application/octet-stream")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    fn endpoint_attestation_message(now: u64) -> NodeDiscoveryMessage {
        endpoint_attestation_message_with(now, 0x72, 0x73, 9)
    }

    fn endpoint_attestation_message_with(
        now: u64,
        subject_seed: u8,
        nonce_seed: u8,
        descriptor_sequence: u64,
    ) -> NodeDiscoveryMessage {
        let observer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[subject_seed; 32]).unwrap();
        let context = public_endpoint_flow_context(observer.public_key_bytes());
        let endpoint = canonical_public_endpoint_commitment("8.8.8.8:51820").unwrap();
        let mut descriptor = NodeDescriptor::new(
            subject.public_key_bytes(),
            descriptor_sequence,
            now.saturating_sub(1),
            now + 600,
            "endpoint-attestation-api-test",
        );
        descriptor.public_endpoint = Some("8.8.8.8:51820".to_string());
        let descriptor = SignedNodeDescriptor::sign(descriptor, &subject).unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let challenge = DiscoveryEndpointChallengeV1::issue(
            subject.public_key_bytes(),
            commitment.descriptor_hash,
            endpoint,
            [nonce_seed; 32],
            context,
            now,
            now + 120,
            &observer,
        )
        .unwrap();
        let proof =
            DiscoveryEndpointProofV1::respond(&challenge, &context, now + 1, &subject).unwrap();
        let evidence = discovery_endpoint_evidence_commitment_v1(&challenge, &proof);
        let attestation = DiscoveryEndpointEvidenceAttestationV1::issue_from_verified_proof(
            &descriptor,
            &challenge,
            &proof,
            context,
            DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            now + 1,
            now + 601,
            &observer,
        )
        .unwrap();
        attestation
            .verify_at(
                now + 2,
                &observer.public_key_bytes(),
                &commitment,
                &endpoint,
                &evidence,
                &context,
                DiscoveryEndpointAttestationPurposeV1::EndpointPossessionObservation,
            )
            .unwrap();
        NodeDiscoveryMessage::EndpointEvidenceAttestationV1 {
            attestation_frame: attestation.encode(),
        }
    }

    fn route_domain_certificate_for(
        subject_node_id: [u8; 32],
        route_domain: [u8; 16],
        now: u64,
        attestors: &[&IdentityKeyPair],
    ) -> RouteDomainAttestationCertificateV1 {
        let statements = attestors
            .iter()
            .enumerate()
            .map(|(index, attestor)| {
                RouteDomainAttestationV1::new_signed(
                    subject_node_id,
                    route_domain,
                    now.saturating_sub(2)
                        + u64::try_from(index).expect("bounded test attestor count"),
                    now + 600,
                    attestor,
                )
                .unwrap()
            })
            .collect();
        RouteDomainAttestationCertificateV1::new_verified(
            subject_node_id,
            route_domain,
            statements,
            now,
        )
        .unwrap()
    }

    fn directory_gossip_fixture(
        now: u64,
    ) -> (
        Arc<DirectoryReplicaStore>,
        NodeDiscoveryMessage,
        SignedNodeDescriptor,
    ) {
        let producer = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
        let descriptor = SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                subject.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now + 600,
                "directory-gossip-api-test",
            ),
            &subject,
        )
        .unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let block =
            DirectoryCommitmentBlockV1::new_signed(1, now, [0u8; 32], vec![commitment], &producer)
                .unwrap();
        let block_hash = block.hash();
        let proof =
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, now).unwrap();
        let descriptor_hash = proof.commitment.descriptor_hash;
        let request_id = [0x94; 16];
        let blocks = vec![block.clone()];
        let signing_bytes = directory_block_range_response_signing_bytes(
            &request_id,
            &producer.public_key_bytes(),
            now,
            &blocks,
            false,
            block.header.height,
            &block_hash,
        );
        let response = DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            responder: producer.public_key_bytes(),
            response_timestamp: now,
            blocks,
            has_more: false,
            tip_height: block.header.height,
            tip_hash: block_hash,
            signature: producer.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&response).unwrap();
        let (replica_store, _) =
            DirectoryReplicaStore::open(":memory:", local.public_key_bytes(), now).unwrap();
        replica_store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&block),
                std::slice::from_ref(&descriptor),
                block.header.height,
                block_hash,
                &frame,
                now,
            )
            .unwrap();
        let message = NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
            producer: producer.public_key_bytes(),
            block_hash,
            descriptor_hash,
            proof,
        };
        (Arc::new(replica_store), message, descriptor)
    }

    fn signed_routeable_chat_descriptor(
        sequence: u64,
        expires_at: u64,
        endpoint: &str,
    ) -> SignedNodeDescriptor {
        signed_routeable_chat_descriptor_with_capabilities(sequence, expires_at, endpoint, &[])
    }

    fn signed_routeable_chat_descriptor_with_capabilities(
        sequence: u64,
        expires_at: u64,
        endpoint: &str,
        additional_capabilities: &[NodeCapability],
    ) -> SignedNodeDescriptor {
        let kp = IdentityKeyPair::generate();
        let issued_at = now_secs().saturating_sub(1);
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            sequence,
            issued_at,
            expires_at,
            "test",
        )
        .with_x25519_kem(kp.x25519_public_key_bytes());
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.capabilities = vec![
            NodeCapability::PrivacyRelay,
            NodeCapability::ChatRelay,
            NodeCapability::OnionMiddle,
        ];
        for capability in additional_capabilities {
            if !descriptor.capabilities.contains(capability) {
                descriptor.capabilities.push(*capability);
            }
        }
        descriptor.capacity = NodeCapacity {
            max_sessions: 128,
            max_bps: Some(500_000_000),
            max_pps: None,
        };
        descriptor.policy = NodePolicy::default();
        SignedNodeDescriptor::sign(descriptor, &kp).unwrap()
    }

    fn onion_candidate_for_test(endpoint: &str, rank: usize) -> OnionRelayCandidate {
        onion_candidate_for_test_with_capabilities(endpoint, rank, &[])
    }

    fn onion_candidate_for_test_with_capabilities(
        endpoint: &str,
        rank: usize,
        additional_capabilities: &[NodeCapability],
    ) -> OnionRelayCandidate {
        let signed_descriptor = signed_routeable_chat_descriptor_with_capabilities(
            1,
            now_secs() + 300,
            endpoint,
            additional_capabilities,
        );
        let descriptor = &signed_descriptor.descriptor;
        OnionRelayCandidate {
            node_id: hex::encode(signed_descriptor.node_id()),
            kem_alg: descriptor.kem_alg,
            kem_public: hex::encode(descriptor.x25519_kem_public().unwrap()),
            public_endpoint: descriptor.public_endpoint.clone().unwrap(),
            capabilities: descriptor.capabilities.clone(),
            selection_weight: onion_candidate_selection_weight(rank),
            region: descriptor.policy.region.clone(),
            max_sessions: descriptor.capacity.max_sessions,
            max_bps: descriptor.capacity.max_bps,
            max_pps: descriptor.capacity.max_pps,
            signed_descriptor,
        }
    }

    // [ONION-CANDIDATE-EXCLUSION-TELEMETRY 2026-08-31 by Codex] Deterministic
    // identities let the tests exercise every coarse gate without serializing
    // any identity, endpoint, or routeability evidence into the telemetry.
    fn signed_candidate_exclusion_descriptor(
        seed: u8,
        now: u64,
        endpoint: Option<&str>,
        capabilities: &[NodeCapability],
        with_kem: bool,
    ) -> SignedNodeDescriptor {
        let keypair = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let mut descriptor = NodeDescriptor::new(
            keypair.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "candidate-exclusion-test",
        );
        descriptor.public_endpoint = endpoint.map(ToString::to_string);
        descriptor.capabilities = capabilities.to_vec();
        descriptor.capacity = NodeCapacity {
            max_sessions: 128,
            max_bps: Some(500_000_000),
            max_pps: None,
        };
        descriptor.policy = NodePolicy::default();
        if with_kem {
            descriptor = descriptor.with_x25519_kem(keypair.x25519_public_key_bytes());
        }
        SignedNodeDescriptor::sign(descriptor, &keypair).unwrap()
    }

    /// Records enough fresh aggregate evidence for the requested synthetic
    /// path depth and marks that stable window as durably persisted.
    ///
    /// [ONION-PATH-ADMISSION 2026-08-02 by Codex] Tests must establish the
    /// same proof + restart-continuity contract used by production instead of
    /// treating descriptor count as transport readiness.
    fn record_stable_runtime_path_proof(store: &PeerStore, now: u64, hops: u8) {
        store.configure_bootstrap_status(true, true, true, 2);
        let first_at = now.saturating_sub(10);
        for offset in 0..ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS {
            let proof_at = first_at.saturating_add(offset);
            if hops >= 3 {
                store.record_blind_relay_three_hop_probe_result_with_context(
                    proof_at,
                    true,
                    "onion_terminal_delivered",
                    3,
                    1,
                    3,
                    2,
                );
            } else {
                store.record_blind_relay_two_hop_probe_result_with_context(
                    proof_at,
                    true,
                    "onion_terminal_delivered",
                    2,
                    1,
                    2,
                    1,
                );
            }
        }
    }

    fn record_stable_path_proof(store: &PeerStore, now: u64, hops: u8) {
        record_stable_runtime_path_proof(store, now, hops);
        let stability_proofs = usize::try_from(ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS)
            .expect("stability proof count must fit usize");
        let persisted_at = now.saturating_sub(1);
        store.record_cache_save_status(persisted_at, "success", "snapshot_persisted");
        if hops >= 3 {
            store.record_three_hop_proof_cache_persisted(persisted_at, stability_proofs, true);
        } else {
            store.record_two_hop_proof_cache_persisted(persisted_at, stability_proofs, true);
        }
    }
}
