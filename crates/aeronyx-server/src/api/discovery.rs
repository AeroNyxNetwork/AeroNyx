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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::discovery::{
    decode_route_domain_attestation_certificate,
    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES,
};
use aeronyx_core::protocol::{
    NodeBootstrapSnapshot, NodeCapability, NodeDiscoveryMessage, NodeProtocolFeature,
    OnionRoutePurpose, SignedNodeDescriptor, ONION_ROUTE_PURPOSE_VALUES,
};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::api::directory_replica_sync::admit_directory_gossip_descriptor;
use crate::config::DiscoveryConfig;
use crate::services::{
    DirectoryReplicaStore, PeerStore, PeerStoreImportReport, PeerStoreStatus,
    RouteDomainCertificateImportError,
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
const ONION_CANDIDATES_SOURCE: &str = "rust_discovery_onion_candidates";
const ONION_CANDIDATES_SELECTION_POLICY: &str =
    "fresh_routeable_signed_chat_relays_with_kem_public_key";
const ONION_REQUIRED_CAPABILITIES: [NodeCapability; 2] =
    [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
const ONION_CANDIDATES_REFRESH_AFTER_SECONDS: u64 = 300;
const ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS: u64 = 1_800;
const ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES: usize = 2;
const ONION_CANDIDATES_MAX_CLIENT_HOPS: u8 = 3;
const ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS: u64 = 3;
const ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT: u8 = 80;
const DISCOVERY_PUBLIC_CARD_CONTRACT_VERSION: &str = "discovery_public_card.v1";
const DISCOVERY_PUBLIC_CARD_SOURCE: &str = "rust_discovery_public_card";

#[derive(Clone)]
struct DiscoveryApiState {
    peer_store: Arc<PeerStore>,
    /// Local entry identity used only to resolve its signed public descriptor
    /// and enforce coarse route anti-affinity. Never serialize this value.
    local_node_id: Option<[u8; 32]>,
    /// Audited local Directory replica used only as an admission trust anchor.
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    rate_limit: Arc<Mutex<RateLimitState>>,
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
        }
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

    fn message_allowed(&self, message: &NodeDiscoveryMessage) -> bool {
        match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => true,
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => {
                self.node_allowed(&descriptor.node_id())
            }
            NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { proof, .. } => {
                self.node_allowed(&proof.descriptor.node_id())
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

    fn pinned_route_domain(&self, node_id: &[u8; 32]) -> Option<&str> {
        self.pinned_route_domains
            .get(&hex::encode(node_id))
            .map(String::as_str)
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

fn onion_required_capabilities() -> Vec<NodeCapability> {
    ONION_REQUIRED_CAPABILITIES.to_vec()
}

/// Resolves an optional query value without turning an unknown explicit value
/// into the default message workload.
fn onion_route_purpose_from_query(value: Option<&str>) -> Option<OnionRoutePurpose> {
    match value {
        Some(value) => OnionRoutePurpose::from_wire_value(value),
        None => Some(OnionRoutePurpose::MessageRelay),
    }
}

fn onion_route_purpose_name(purpose: Option<OnionRoutePurpose>) -> &'static str {
    purpose.map_or("unsupported", OnionRoutePurpose::as_str)
}

fn onion_terminal_required_capabilities(purpose: Option<OnionRoutePurpose>) -> Vec<NodeCapability> {
    let Some(purpose) = purpose else {
        return Vec::new();
    };
    let mut capabilities = onion_required_capabilities();
    if let Some(capability) = purpose.specialized_terminal_capability() {
        capabilities.push(capability);
    }
    capabilities
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
            protocol_features: match purpose {
                Some(OnionRoutePurpose::BlindVaultPull | OnionRoutePurpose::BlindVaultDelete) => {
                    // [ONION-BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex]
                    // A generic reply carrier alone does not promise that
                    // workload failures remain hidden from upstream relays.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                Some(OnionRoutePurpose::BlindVaultLeaseAdmission) => {
                    // [ONION-BLIND-LEASE-ADMISSION 2026-08-28 by Codex] Both
                    // signed tokens are required: the generic reply carrier
                    // and explicit execution of this sensitive workload.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindLeaseAdmissionV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                Some(OnionRoutePurpose::BlindVaultPutReceipt) => {
                    // [ONION-BLIND-VAULT-PUT-RECEIPT 2026-08-28 by Codex]
                    // Keep receipt-capable writes distinct from the compatible
                    // one-way Put path during rolling fleet upgrades.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindVaultPutReceiptV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                Some(OnionRoutePurpose::BlindVaultLeaseRetire) => {
                    // [ONION-BLIND-VAULT-LEASE-RETIRE 2026-08-28 by Codex]
                    // Destructive lease-wide mutation requires an exact signed
                    // workload feature in addition to the generic reply path.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindVaultLeaseRetireV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                Some(OnionRoutePurpose::BlindVaultLeaseRenewal) => {
                    // [ONION-BLIND-VAULT-LEASE-RENEWAL 2026-08-28 by Codex]
                    // Capacity-authorized expiry mutation is separately
                    // negotiated so mixed-version routes fail closed.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindVaultLeaseRenewalV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                Some(OnionRoutePurpose::BlindVaultLeaseStatus) => {
                    // [ONION-BLIND-VAULT-LEASE-STATUS 2026-08-28 by Codex]
                    // Private status observations require an exact terminal
                    // feature so older nodes cannot be selected speculatively.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindVaultLeaseStatusV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                Some(OnionRoutePurpose::BlindVaultLeaseInventory) => {
                    // [ONION-BLIND-VAULT-INVENTORY 2026-08-28 by Codex]
                    // Inventory commitments require exact support because a
                    // status-only terminal cannot prove an object set.
                    &[
                        NodeProtocolFeature::OnionReplyV1,
                        NodeProtocolFeature::OnionBlindVaultLeaseInventoryV1,
                        NodeProtocolFeature::OnionBlindVaultEncryptedFailureV1,
                    ]
                }
                _ => &[],
            },
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

fn onion_terminal_candidate_matches(
    purpose: Option<OnionRoutePurpose>,
    candidate: &OnionRelayCandidate,
) -> bool {
    purpose.is_some()
        && ONION_REQUIRED_CAPABILITIES.iter().all(|required| {
            candidate
                .signed_descriptor
                .descriptor
                .capabilities
                .contains(required)
        })
        && OnionTerminalRequirement::for_purpose(purpose).matches(candidate)
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

fn recovery_anchor_protection_ready(protection: Option<&str>) -> bool {
    matches!(protection, Some("anchored" | "cache_ahead"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExternalWitnessRecoveryAdmission {
    ready: bool,
    adverse_evidence: bool,
    generation_aligned: bool,
}

fn external_witness_recovery_admission(
    status: &str,
    required: bool,
    cache_generation: u64,
    witness_generation: u64,
) -> ExternalWitnessRecoveryAdmission {
    // [EXTERNAL-WITNESS-ADVERSE-GATE 2026-08-21 by Codex] Optional witnessing
    // makes transport availability advisory; it never makes authenticated
    // rollback, conflict, or generation-gap evidence advisory. Keep this one
    // decision shared by recovery health and path-proof admission so an
    // operator surface cannot fail closed while the data plane stays eligible.
    let adverse_evidence = matches!(status, "rollback_detected" | "conflict" | "gap");
    let generation_aligned = cache_generation != 0 && witness_generation == cache_generation;
    let ready = !adverse_evidence && (!required || (status == "verified" && generation_aligned));

    ExternalWitnessRecoveryAdmission {
        ready,
        adverse_evidence,
        generation_aligned,
    }
}

/// Builds an aggregate view of the local recovery anchor and its witnesses.
///
/// [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] This contract deliberately
/// excludes anchor digests, signatures, file paths, witness identities and
/// endpoints, peer identifiers, routes, messages, clients, and payload data.
/// A witness is ready only when it verified the exact cache generation now
/// represented by the restored or newly persisted local state.
#[must_use]
pub fn recovery_anchor_status_value(status: &PeerStoreStatus) -> serde_json::Value {
    let bootstrap = &status.bootstrap;
    let cache_generation = bootstrap.last_client_delivery_cache_generation;
    let witness_generation = bootstrap.last_client_delivery_witness_generation;
    let witness_status = bootstrap
        .last_client_delivery_witness_status
        .as_deref()
        .unwrap_or("not_observed");
    let witness_required = bootstrap.last_client_delivery_witness_required;
    let witness_admission = external_witness_recovery_admission(
        witness_status,
        witness_required,
        cache_generation,
        witness_generation,
    );
    let routeability_protection = bootstrap
        .last_routeability_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let two_hop_protection = bootstrap
        .last_two_hop_proof_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let three_hop_protection = bootstrap
        .last_three_hop_proof_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let delivery_protection = bootstrap
        .last_client_delivery_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let local_anchor_ready = cache_generation != 0
        && [
            Some(routeability_protection),
            Some(two_hop_protection),
            Some(three_hop_protection),
            Some(delivery_protection),
        ]
        .into_iter()
        .all(recovery_anchor_protection_ready);
    let ready_for_restore = local_anchor_ready && witness_admission.ready;
    let adverse_local_evidence = [
        routeability_protection,
        two_hop_protection,
        three_hop_protection,
        delivery_protection,
    ]
    .into_iter()
    .any(|protection| {
        matches!(
            protection,
            "anchor_invalid" | "anchor_conflict" | "rollback_detected"
        )
    });
    let status_bucket = if cache_generation == 0 {
        "idle"
    } else if ready_for_restore {
        "ready"
    } else if adverse_local_evidence || witness_admission.adverse_evidence || witness_required {
        "blocked"
    } else {
        "attention"
    };
    let next_action = match status_bucket {
        "idle" => "persist the first signed peer-cache recovery generation",
        "ready" => "continue bounded persistence and exact-generation witnessing",
        "blocked" if witness_admission.adverse_evidence => {
            "reject restored readiness and inspect the authenticated witness failure bucket"
        }
        "blocked" if witness_required && !witness_admission.generation_aligned => {
            "obtain the required witness quorum for the current cache generation"
        }
        "blocked" => "inspect aggregate rollback or witness failure buckets before restore",
        _ => "complete local recovery-anchor protection before relying on restored readiness",
    };

    serde_json::json!({
        "contract_version": "recovery_anchor.v1",
        "status": status_bucket,
        "ready_for_restore": ready_for_restore,
        "cache_generation": cache_generation,
        "local_anchor": {
            "ready": local_anchor_ready,
            "routeability": routeability_protection,
            "two_hop_proof": two_hop_protection,
            "three_hop_proof": three_hop_protection,
            "aggregate_delivery": delivery_protection,
        },
        "external_witness": {
            "status": witness_status,
            "required": witness_required,
            "ready": witness_admission.ready,
            "adverse_evidence": witness_admission.adverse_evidence,
            "generation": witness_generation,
            "generation_aligned": witness_admission.generation_aligned,
            "minimum_verified": bootstrap.last_client_delivery_witness_minimum_verified,
            "configured": bootstrap.last_client_delivery_witness_configured,
            "attempted": bootstrap.last_client_delivery_witness_attempted,
            "verified": bootstrap.last_client_delivery_witness_verified,
            "accepted": bootstrap.last_client_delivery_witness_advanced
                .saturating_add(bootstrap.last_client_delivery_witness_idempotent),
            "adverse": bootstrap.last_client_delivery_witness_stale
                .saturating_add(bootstrap.last_client_delivery_witness_conflicts)
                .saturating_add(bootstrap.last_client_delivery_witness_gaps),
            "failed": bootstrap.last_client_delivery_witness_failed,
        },
        "rollback_boundary": "signed_sections_plus_monotonic_local_anchor_with_optional_exact_generation_external_witness",
        "next_action": next_action,
        "privacy_boundary": "aggregate recovery control state only; no anchor digests, signatures, file paths, witness identities or endpoints, peer ids, routes, messages, clients, or payload metadata",
    })
}

fn path_proof_restart_continuity(
    status: &PeerStoreStatus,
    generated_at: u64,
    stale_after_seconds: u64,
    authentication: Option<&str>,
    rollback_protection: Option<&str>,
    restored_stability_ready: bool,
    restored_at: Option<u64>,
    restored: u64,
    persisted_stability_ready: bool,
    persisted_at: Option<u64>,
    persisted: u64,
) -> PathProofRestartContinuity {
    let authentication = authentication.unwrap_or("not_observed").to_string();
    let rollback_protection = rollback_protection.unwrap_or("not_observed").to_string();
    let external_witness = status
        .bootstrap
        .last_client_delivery_witness_status
        .as_deref()
        .unwrap_or("not_observed")
        .to_string();
    let external_witness_required = status.bootstrap.last_client_delivery_witness_required;
    let rollback_protection_ready =
        matches!(rollback_protection.as_str(), "anchored" | "cache_ahead");
    // [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] Persistence updates the
    // local cache generation before the post-write witness round completes.
    // Never let the prior generation's `verified` bucket authorize this short
    // interval or any later state restored from a mismatched generation.
    let external_witness_admission = external_witness_recovery_admission(
        &external_witness,
        external_witness_required,
        status.bootstrap.last_client_delivery_cache_generation,
        status.bootstrap.last_client_delivery_witness_generation,
    );
    let restore_evidence_fresh = restored_at
        .map(|at| at <= generated_at && generated_at.saturating_sub(at) <= stale_after_seconds)
        .unwrap_or(false);
    let persistence_evidence_fresh = persisted_at
        .map(|at| at <= generated_at && generated_at.saturating_sub(at) <= stale_after_seconds)
        .unwrap_or(false);
    // [PATH-PROOF-ROLLBACK-ANCHOR 2026-08-02 by Codex] A valid section
    // signature proves authorship, not freshness. Restart continuity therefore
    // also requires the monotonic local anchor and, when configured as a
    // startup gate, the existing opaque external witness quorum.
    let authenticated_restore_ready = authentication == "verified"
        && rollback_protection_ready
        && external_witness_admission.ready
        && restored_stability_ready
        && restore_evidence_fresh;
    let signed_persistence_ready = rollback_protection_ready
        && external_witness_admission.ready
        && persisted_stability_ready
        && persistence_evidence_fresh;
    let peer_recovery_configured = status.peer_quorum.restart_recovery_configured;
    let ready = authenticated_restore_ready || signed_persistence_ready;
    let source = match (authenticated_restore_ready, signed_persistence_ready) {
        (true, true) => "verified_restore_and_signed_persistence",
        (true, false) => "verified_restore",
        (false, true) => "signed_persistence",
        (false, false) if authentication == "signature_invalid" => "restore_signature_invalid",
        (false, false) if authentication == "identity_unavailable" => {
            "restore_identity_unavailable"
        }
        (false, false) if authentication == "legacy_descriptor_only" => "legacy_cache",
        (false, false) if !rollback_protection_ready => "rollback_protection_not_ready",
        (false, false) if !external_witness_admission.ready => "external_witness_not_ready",
        _ => "not_ready",
    };

    PathProofRestartContinuity {
        peer_recovery_configured,
        authenticated_restore_ready,
        signed_persistence_ready,
        ready,
        source,
        authentication,
        rollback_protection,
        external_witness,
        external_witness_required,
        restored,
        persisted,
    }
}

fn two_hop_proof_restart_continuity(status: &PeerStoreStatus) -> PathProofRestartContinuity {
    let proof = &status.two_hop_path_proof_history;
    let bootstrap = &status.bootstrap;
    path_proof_restart_continuity(
        status,
        proof.generated_at,
        proof.stale_after_seconds,
        bootstrap.last_two_hop_proof_cache_authentication.as_deref(),
        bootstrap
            .last_two_hop_proof_cache_rollback_protection
            .as_deref(),
        bootstrap.last_two_hop_proof_cache_restored_stability_ready,
        bootstrap.last_two_hop_proof_cache_at,
        bootstrap.last_two_hop_proof_cache_restored,
        bootstrap.last_two_hop_proof_cache_persisted_stability_ready,
        bootstrap.last_two_hop_proof_cache_persisted_at,
        bootstrap.last_two_hop_proof_cache_persisted,
    )
}

fn three_hop_proof_restart_continuity(status: &PeerStoreStatus) -> PathProofRestartContinuity {
    let proof = &status.three_hop_path_proof_history;
    let bootstrap = &status.bootstrap;
    path_proof_restart_continuity(
        status,
        proof.generated_at,
        proof.stale_after_seconds,
        bootstrap
            .last_three_hop_proof_cache_authentication
            .as_deref(),
        bootstrap
            .last_three_hop_proof_cache_rollback_protection
            .as_deref(),
        bootstrap.last_three_hop_proof_cache_restored_stability_ready,
        bootstrap.last_three_hop_proof_cache_at,
        bootstrap.last_three_hop_proof_cache_restored,
        bootstrap.last_three_hop_proof_cache_persisted_stability_ready,
        bootstrap.last_three_hop_proof_cache_persisted_at,
        bootstrap.last_three_hop_proof_cache_persisted,
    )
}

/// Builds the aggregate relay-pool admission contract.
///
/// This is the Rust-side source of truth for nodeboard, backend aggregation,
/// website counters, and AI runbooks that need to know whether this node is
/// mature enough to participate in the permissionless onion relay pool. It is
/// deliberately aggregate-only: it exposes gate booleans, counts, score, and
/// stable reason buckets, but never endpoints, route IDs, selected hops,
/// receiver keys, encrypted payloads, client IPs, DNS, Memory Chain plaintext,
/// private keys, wallet-level traffic, or social graph metadata.
#[must_use]
pub fn onion_relay_admission_status_value(
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> serde_json::Value {
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let proof = &status.two_hop_path_proof_history;
    let local_relay_ready = local_capabilities.safe_to_advertise_chat_relay;
    let route_pool_ready = peer_quorum.routeable_chat_relays
        >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        && peer_quorum.routeable_onion_middle_hops >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;
    let recent_path_proof_ready = proof.proof_ready && !proof.failure_streak_active;
    let stable_path_proof_ready = proof.stability_ready && !proof.failure_circuit_breaker_active;
    let proof_restart_continuity = two_hop_proof_restart_continuity(status);
    let peer_restart_recovery_ready = proof_restart_continuity.peer_recovery_configured;
    let proof_restart_continuity_ready = proof_restart_continuity.ready;
    let restart_recovery_ready = peer_restart_recovery_ready && proof_restart_continuity_ready;
    let stability_remaining_attempts =
        ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS.saturating_sub(proof.stability_window_attempted);
    let checks_total = 5u8;
    let checks_passed = [
        local_relay_ready,
        route_pool_ready,
        recent_path_proof_ready,
        stable_path_proof_ready,
        restart_recovery_ready,
    ]
    .into_iter()
    .filter(|ready| *ready)
    .count() as u8;
    let admission_score_percent =
        ((u16::from(checks_passed) * 100) / u16::from(checks_total)).min(100) as u8;
    let admission_ready = checks_passed == checks_total;
    let attention = proof.failure_circuit_breaker_active
        || proof.failure_streak_active
        || local_capabilities.status == "misconfigured";
    let admission_status = if !local_capabilities.chat_relay_configured {
        "disabled"
    } else if admission_ready {
        "eligible"
    } else if attention {
        "attention"
    } else {
        "warming"
    };
    let warmup_stage = if !local_relay_ready {
        "local_relay"
    } else if !route_pool_ready {
        "route_pool"
    } else if !recent_path_proof_ready {
        "path_proof"
    } else if !stable_path_proof_ready {
        "stability_window"
    } else if !peer_restart_recovery_ready {
        "restart_recovery"
    } else if !proof_restart_continuity_ready {
        "proof_restart_continuity"
    } else {
        "eligible"
    };
    let mut admission_blockers = Vec::new();
    if !local_relay_ready {
        admission_blockers.push("local_relay_not_ready");
    }
    if !route_pool_ready {
        admission_blockers.push("route_pool_not_ready");
    }
    if !recent_path_proof_ready {
        admission_blockers.push("recent_path_proof_not_ready");
    }
    if !stable_path_proof_ready {
        admission_blockers.push("stable_path_proof_not_ready");
    }
    if !peer_restart_recovery_ready {
        admission_blockers.push("restart_recovery_not_ready");
    }
    if !proof_restart_continuity_ready {
        admission_blockers.push("proof_restart_continuity_not_ready");
    }
    let warmup_hint = match warmup_stage {
        "eligible" => "node is eligible for client-selected two-hop onion relay paths".to_string(),
        "local_relay" => {
            "align ChatRelay config, runtime, public peer API, and advertised capability"
                .to_string()
        }
        "route_pool" => {
            "wait for at least two fresh routeable ChatRelay and OnionMiddle peers".to_string()
        }
        "path_proof" => "wait for a fresh accepted entry-middle-terminal proof".to_string(),
        "stability_window" => format!(
            "collect {stability_remaining_attempts} more recent two-hop proof sample(s) and keep success rate at or above {ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT}%"
        ),
        "restart_recovery" => {
            "configure peer cache or seed endpoints before treating admission as restart-resilient"
                .to_string()
        }
        "proof_restart_continuity" => {
            "persist or restore a fresh independently signed stable proof window"
                .to_string()
        }
        _ => "continue warming relay admission gates".to_string(),
    };

    let mut admission = serde_json::json!({
        "status": admission_status,
        "eligible": admission_ready,
        "permissionless": true,
        "admission_score_percent": admission_score_percent,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "admission_blockers": admission_blockers,
        "warmup_stage": warmup_stage,
        "warmup_hint": warmup_hint,
        "local_relay_ready": local_relay_ready,
        "route_pool_ready": route_pool_ready,
        "recent_path_proof_ready": recent_path_proof_ready,
        "stable_path_proof_ready": stable_path_proof_ready,
        "restart_recovery_ready": restart_recovery_ready,
        "routeable_chat_relays": peer_quorum.routeable_chat_relays,
        "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
        "min_routeable_chat_relays": ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        "min_routeable_onion_middle_hops": ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        "two_hop_stability_status": &proof.stability_status,
        "two_hop_stability_ready": proof.stability_ready,
        "two_hop_stability_window_size": proof.stability_window_size,
        "two_hop_stability_window_attempted": proof.stability_window_attempted,
        "two_hop_stability_window_succeeded": proof.stability_window_succeeded,
        "two_hop_stability_window_failed": proof.stability_window_failed,
        "two_hop_stability_min_attempts": ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS,
        "two_hop_stability_remaining_attempts": stability_remaining_attempts,
        "two_hop_stability_success_percent": proof.stability_success_percent,
        "two_hop_stability_success_threshold_percent": ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT,
        "latest_path_proof_age_seconds": proof.latest_age_seconds,
        "latest_success_age_seconds": proof.latest_success_age_seconds,
        "latest_message_delivery_age_seconds": proof.latest_message_delivery_age_seconds,
        "failure_circuit_breaker_active": proof.failure_circuit_breaker_active,
        "failure_streak_active": proof.failure_streak_active,
        "routeability_stale_after_seconds": ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        "refresh_after_seconds": ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
        "probe_cadence_policy": "recovery_cadence_until_stability_window_ready_then_low_frequency",
        "client_route_policy": "client_selected_two_hop_onion_when_eligible",
        "network_story_status": &network_story.status,
        "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        "privacy_boundary": "aggregate onion relay admission gates only; no node endpoints, route ids, selected hops, receiver keys, encrypted payloads, client IPs, DNS contents, destinations, Memory Chain plaintext, private keys, wallet-level traffic, or social graph metadata",
    });
    let continuity_fields = serde_json::json!({
        "peer_restart_recovery_ready": peer_restart_recovery_ready,
        "proof_restart_continuity_ready": proof_restart_continuity_ready,
        "proof_restart_continuity_source": proof_restart_continuity.source,
        "proof_cache_authentication": proof_restart_continuity.authentication,
        "proof_cache_rollback_protection": proof_restart_continuity.rollback_protection,
        "proof_cache_external_witness": proof_restart_continuity.external_witness,
        "proof_cache_external_witness_required": proof_restart_continuity.external_witness_required,
        "proof_cache_authenticated_restore_ready": proof_restart_continuity.authenticated_restore_ready,
        "proof_cache_signed_persistence_ready": proof_restart_continuity.signed_persistence_ready,
        "proof_cache_restored_events": proof_restart_continuity.restored,
        "proof_cache_persisted_events": proof_restart_continuity.persisted,
    });

    // [DISCOVERY-PANIC-CONTAINMENT 2026-08-12 by Codex] Keep the established
    // flat response contract without assuming either untyped JSON value has an
    // object shape. A future schema refactor now emits a diagnostic and returns
    // the intact base contract instead of panicking in the request path.
    match (&mut admission, continuity_fields) {
        (serde_json::Value::Object(admission_fields), serde_json::Value::Object(fields)) => {
            admission_fields.extend(fields);
        }
        _ => {
            tracing::error!(
                "onion relay admission continuity fields could not be merged into JSON object"
            );
        }
    }
    admission
}

/// Builds the compact aggregate discovery readiness contract.
///
/// This helper intentionally mirrors only privacy-safe, operator-facing fields
/// from `PeerStoreStatus` and `DiscoveryLocalCapabilityStatus`. It is used by
/// both `/api/discovery/status` and backend heartbeat payloads so nodeboard,
/// public website surfaces, and AI runbooks can depend on one stable JSON shape
/// without parsing the full internal peer store object.
#[must_use]
pub fn discovery_readiness_status_value(
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> serde_json::Value {
    let onion_relay_admission = onion_relay_admission_status_value(status, local_capabilities);
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let route_governance = &status.route_governance;
    let recent_message_delivery_ready = status
        .two_hop_path_proof_history
        .recent_message_delivery_ready
        && !status.two_hop_path_proof_history.failure_streak_active;
    let local_relay_ready = local_capabilities.safe_to_advertise_chat_relay;
    let peer_mesh_ready = peer_quorum.quorum_ready;
    let blind_relay_ready = blind_relay_quality.runtime_ready
        && (blind_relay_quality.quality_ready || recent_message_delivery_ready);
    let two_hop_path_ready = network_story.chat_two_hop_onion_ready
        || blind_relay_quality.two_hop_probe_ready
        || recent_message_delivery_ready;
    let restart_recovery_ready = peer_quorum.restart_recovery_configured;
    let checks_total = 4u8;
    let checks_passed = [
        local_relay_ready,
        peer_mesh_ready,
        blind_relay_ready,
        restart_recovery_ready,
    ]
    .into_iter()
    .filter(|ready| *ready)
    .count() as u8;
    let foundation_status = if checks_passed == checks_total {
        "ready"
    } else if blind_relay_ready && peer_quorum.valid_peers >= peer_quorum.min_valid_peers {
        "live"
    } else if peer_quorum.valid_peers > 0 || blind_relay_quality.runtime_ready {
        "forming"
    } else if !local_capabilities.chat_relay_configured {
        "disabled"
    } else {
        "pending"
    };
    let foundation_stage = if two_hop_path_ready {
        "two_hop_path_ready"
    } else if network_story.chat_single_hop_ready || blind_relay_ready {
        "single_hop_relay_ready"
    } else if peer_quorum.valid_peers > 0 {
        "verified_peer_view"
    } else {
        "bootstrap"
    };
    let foundation_headline = match foundation_status {
        "ready" => "AeroNyx privacy protocol foundation is live",
        "live" => "AeroNyx privacy protocol has live relay evidence",
        "forming" => "AeroNyx nodes are forming a verified relay mesh",
        "disabled" => "AeroNyx privacy protocol discovery is not enabled",
        _ => "AeroNyx privacy protocol is waiting for live peer evidence",
    };
    let foundation_next_action = match foundation_status {
        "ready" => "monitor peer freshness, blind relay probe age, and restart recovery",
        "live" if !restart_recovery_ready => {
            "configure peer cache or seed endpoints before treating relay state as restart-resilient"
        }
        "live" => "wait for peer quorum to become fully ready",
        "forming" => "add or recover verified peers and routeable relay candidates",
        "disabled" => {
            "enable discovery and chat relay capability before advertising protocol readiness"
        }
        _ => "wait for verified peer discovery and the first blind relay runtime check",
    };

    serde_json::json!({
        "protocol_foundation": {
            "status": foundation_status,
            "stage": foundation_stage,
            "headline": foundation_headline,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "local_relay_ready": local_relay_ready,
            "peer_mesh_ready": peer_mesh_ready,
            "blind_relay_ready": blind_relay_ready,
            "restart_recovery_ready": restart_recovery_ready,
            "single_hop_relay_ready": network_story.chat_single_hop_ready,
            "two_hop_onion_ready": two_hop_path_ready,
            "two_hop_path_proof_ready": blind_relay_quality.two_hop_probe_ready,
            "two_hop_message_delivery_ready": status
                .two_hop_path_proof_history
                .message_delivery_ready,
            "two_hop_recent_message_delivery_ready": status
                .two_hop_path_proof_history
                .recent_message_delivery_ready,
            "two_hop_message_delivery_evidence_mode": &status
                .two_hop_path_proof_history
                .message_delivery_evidence_mode,
            "two_hop_probe_attempted": blind_relay_quality.two_hop_probe_attempted,
            "two_hop_probe_succeeded": blind_relay_quality.two_hop_probe_succeeded,
            "two_hop_probe_failed": blind_relay_quality.two_hop_probe_failed,
            "last_two_hop_probe_age_seconds": blind_relay_quality.last_two_hop_probe_age_seconds,
            "last_two_hop_message_delivery_age_seconds": status
                .two_hop_path_proof_history
                .latest_message_delivery_age_seconds,
            "verified_peer_count": peer_quorum.valid_peers,
            "routeable_relay_count": peer_quorum.routeable_chat_relays,
            "last_probe_age_seconds": blind_relay_quality.last_probe_age_seconds,
            "relay_evidence_mode": &blind_relay_quality.evidence_mode,
            "relay_readiness_reason": &blind_relay_quality.readiness_reason,
            "timestamp_rejected": blind_relay_quality.timestamp_rejected,
            "real_relay_ready": blind_relay_quality.real_relay_ready,
            "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
            "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
            "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
            "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
            "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
            "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
            "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
            "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
            "next_action": foundation_next_action,
        },
        "chat_relay_capability": {
            "status": local_capabilities.status,
            "chat_relay_configured": local_capabilities.chat_relay_configured,
            "blind_relay_endpoint_ready": local_capabilities.blind_relay_endpoint_ready,
            "chat_relay_runtime_ready": local_capabilities.chat_relay_runtime_ready,
            "advertised_chat_relay_capability": local_capabilities.advertised_chat_relay_capability,
            "safe_to_advertise_chat_relay": local_capabilities.safe_to_advertise_chat_relay,
            "capability_config_consistent": local_capabilities.capability_config_consistent,
            "advertisement_blockers": &local_capabilities.advertisement_blockers,
            "detail": local_capabilities.detail,
        },
        // [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Keep
        // anonymous storage readiness independent from the required chat relay
        // foundation so an optional full replica does not distort relay SLOs.
        "blind_vault_replica_capability": {
            "configured": local_capabilities.blind_vault_replica_configured,
            "runtime_ready": local_capabilities.blind_vault_runtime_ready,
            "advertised": local_capabilities.advertised_blind_vault_replica_capability,
            "safe_to_advertise": local_capabilities.safe_to_advertise_blind_vault_replica,
            "capability_consistent": local_capabilities.blind_vault_capability_consistent,
            "advertisement_blockers": &local_capabilities.blind_vault_advertisement_blockers,
        },
        "peer_quorum": {
            "status": &peer_quorum.status,
            "quorum_ready": peer_quorum.quorum_ready,
            "valid_peers": peer_quorum.valid_peers,
            "healthy_peers": peer_quorum.healthy_peers,
            "stale_peers": peer_quorum.stale_peers,
            "routeable_chat_relays": peer_quorum.routeable_chat_relays,
            "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
            "restart_recovery_configured": peer_quorum.restart_recovery_configured,
            "relay_foundation_ready": peer_quorum.relay_foundation_ready,
            "next_action": &peer_quorum.next_action,
        },
        "network_story": {
            "status": &network_story.status,
            "headline": &network_story.headline,
            "chat_single_hop_ready": network_story.chat_single_hop_ready,
            "chat_two_hop_onion_ready": network_story.chat_two_hop_onion_ready,
            "routeable_chat_relays": network_story.routeable_chat_relays,
            "routeable_onion_middle_hops": network_story.routeable_onion_middle_hops,
        },
        "route_governance": {
            "contract_version": &route_governance.contract_version,
            "status": &route_governance.status,
            "route_pool_ready": route_governance.route_pool_ready,
            "quality_ready": route_governance.quality_ready,
            "candidates_total": route_governance.candidates_total,
            "routeable_total": route_governance.routeable_total,
            "routeable_chat_relays": route_governance.routeable_chat_relays,
            "routeable_onion_middle_hops": route_governance.routeable_onion_middle_hops,
            "routeable_privacy_relays": route_governance.routeable_privacy_relays,
            "quarantined_total": route_governance.quarantined_total,
            "failing_total": route_governance.failing_total,
            "degraded_total": route_governance.degraded_total,
            "unknown_routeability_total": route_governance.unknown_routeability_total,
            "stale_routeability_total": route_governance.stale_routeability_total,
            "unreachable_total": route_governance.unreachable_total,
            "best_score": route_governance.best_score,
            "worst_score": route_governance.worst_score,
            "average_score": route_governance.average_score,
            "chat_single_hop_ready": route_governance.chat_single_hop_ready,
            "chat_two_hop_onion_ready": route_governance.chat_two_hop_onion_ready,
            "quarantine_threshold": route_governance.quarantine_threshold,
            "quarantine_seconds": route_governance.quarantine_seconds,
            "routeability_stale_after_seconds": route_governance.routeability_stale_after_seconds,
            "next_action": &route_governance.next_action,
        },
        "onion_relay_admission": onion_relay_admission,
        "blind_relay_runtime": {
            "status": &blind_relay_quality.status,
            "runtime_ready": blind_relay_quality.runtime_ready,
            "quality_ready": blind_relay_quality.quality_ready,
            "real_relay_ready": blind_relay_quality.real_relay_ready,
            "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
            "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
            "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
            "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
            "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
            "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
            "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
            "evidence_mode": &blind_relay_quality.evidence_mode,
            "readiness_reason": &blind_relay_quality.readiness_reason,
            "accepted_total": blind_relay_quality.accepted_total,
            "forward_failed": blind_relay_quality.forward_failed,
            "retry_exhausted": blind_relay_quality.retry_exhausted,
            "backpressure_dropped": blind_relay_quality.backpressure_dropped,
            "probe_attempted": blind_relay_quality.probe_attempted,
            "probe_succeeded": blind_relay_quality.probe_succeeded,
            "probe_failed": blind_relay_quality.probe_failed,
            "two_hop_probe_ready": blind_relay_quality.two_hop_probe_ready,
            "two_hop_probe_attempted": blind_relay_quality.two_hop_probe_attempted,
            "two_hop_probe_succeeded": blind_relay_quality.two_hop_probe_succeeded,
            "two_hop_probe_failed": blind_relay_quality.two_hop_probe_failed,
            "timestamp_rejected": blind_relay_quality.timestamp_rejected,
            "protection_active": blind_relay_quality.protection_active,
            "accepted_percent": blind_relay_quality.accepted_percent,
            "last_event_age_seconds": blind_relay_quality.last_event_age_seconds,
            "last_probe_age_seconds": blind_relay_quality.last_probe_age_seconds,
            "last_two_hop_probe_age_seconds": blind_relay_quality.last_two_hop_probe_age_seconds,
            "next_action": &blind_relay_quality.next_action,
        },
        "source": "rust_discovery_readiness",
        "privacy_boundary": "aggregate discovery readiness only; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, or wallet-level traffic",
    })
}

/// Builds the product-facing blind relay runtime observability contract.
///
/// This view intentionally mirrors only aggregate counters and stable event
/// buckets from `PeerStoreStatus`. It exists so nodeboard, backend aggregation,
/// public website status, and AI runbooks can show whether a node is actually
/// participating in the encrypted relay network without reconstructing routes.
/// Never add endpoints, full node IDs, route IDs, encrypted blobs, receiver
/// identities, client IPs, DNS contents, destinations, Memory Chain plaintext,
/// private keys, wallet-level traffic, or social graph metadata here.
#[must_use]
pub fn blind_relay_runtime_status_value(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> serde_json::Value {
    let stats = &status.runtime.blind_relay;
    let quality = &status.blind_relay_quality;
    let proof = &status.two_hop_path_proof_history;
    let peer_quorum = &status.peer_quorum;
    let route_pool_ready = peer_quorum.routeable_chat_relays
        >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        && peer_quorum.routeable_onion_middle_hops >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;

    let mut value = serde_json::json!({
        "generated_at": generated_at,
        "contract_version": "blind_relay_runtime.v1",
        "source": "rust_blind_relay_runtime",
        "status": &quality.status,
        "runtime_ready": quality.runtime_ready,
        "quality_ready": quality.quality_ready,
        "real_relay_ready": quality.real_relay_ready,
        "verified_client_onion_deliveries": quality.verified_client_onion_deliveries,
        "last_verified_client_onion_delivery_age_seconds": quality.last_verified_client_onion_delivery_age_seconds,
        "delivery_receipt_capable_peers": quality.delivery_receipt_capable_peers,
        "accepted_relay_ready": quality.accepted_relay_ready,
        "synthetic_probe_ready": quality.synthetic_probe_ready,
        "evidence_mode": &quality.evidence_mode,
        "readiness_reason": &quality.readiness_reason,
        "onion_candidates": {
            "two_hop_ready": route_pool_ready,
            "routeable_chat_relays": peer_quorum.routeable_chat_relays,
            "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
            "min_candidates_for_two_hop": ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
            "selection_policy": ONION_CANDIDATES_SELECTION_POLICY,
            "refresh_after_seconds": ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
            "routeability_stale_after_seconds": ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        },
        "relay_counters": {
            "received": stats.received,
            "accepted_total": quality.accepted_total,
            "verified_client_onion_deliveries": stats.verified_client_onion_deliveries,
            "terminal_delivered_count": stats.terminal,
            "middle_forwarded_count": stats.forwarded,
            "rejected": stats.rejected,
            "route_ttl_exhausted": stats.ttl_exhausted,
            "forward_failed": stats.forward_failed,
            "retry_attempted": stats.retry_attempted,
            "retry_succeeded": stats.retry_succeeded,
            "retry_exhausted": stats.retry_exhausted,
            "backpressure_dropped": stats.backpressure_dropped,
            "timestamp_rejected": stats.timestamp_rejected,
            "replay_dropped": stats.replay_dropped,
            "loop_detected": stats.loop_detected,
            "rate_limited": stats.rate_limited,
            "quarantined": stats.quarantined,
        },
        "proof_counters": {
            "proof_ready": proof.proof_ready,
            "message_delivery_ready": proof.message_delivery_ready,
            "recent_message_delivery_ready": proof.recent_message_delivery_ready,
            "message_delivery_evidence_mode": &proof.message_delivery_evidence_mode,
            "proof_accepted": proof.succeeded,
            "proof_rejected": proof.failed,
            "proof_attempted": proof.attempted,
            "message_delivery_successes": proof.message_delivery_successes,
            "success_percent": proof.success_percent,
            "stability_ready": proof.stability_ready,
            "stability_status": &proof.stability_status,
            "stability_window_attempted": proof.stability_window_attempted,
            "stability_window_succeeded": proof.stability_window_succeeded,
            "stability_window_failed": proof.stability_window_failed,
            "failure_streak_active": proof.failure_streak_active,
            "failure_circuit_breaker_active": proof.failure_circuit_breaker_active,
            "latest_outcome": &proof.latest_outcome,
            "latest_reason_bucket": &proof.latest_reason_bucket,
            "latest_age_seconds": proof.latest_age_seconds,
            "latest_success_age_seconds": proof.latest_success_age_seconds,
            "latest_failure_age_seconds": proof.latest_failure_age_seconds,
            "latest_message_delivery_age_seconds": proof.latest_message_delivery_age_seconds,
            "proof_scope": &proof.proof_scope,
        },
        "probe_counters": {
            "single_hop_attempted": stats.probe_attempted,
            "single_hop_succeeded": stats.probe_succeeded,
            "single_hop_failed": stats.probe_failed,
            "two_hop_attempted": stats.two_hop_probe_attempted,
            "two_hop_succeeded": stats.two_hop_probe_succeeded,
            "two_hop_failed": stats.two_hop_probe_failed,
            "last_probe_age_seconds": quality.last_probe_age_seconds,
            "last_two_hop_probe_age_seconds": quality.last_two_hop_probe_age_seconds,
        },
        "last_successful_blind_relay": latest_blind_relay_event_value(status, generated_at, true),
        "last_failed_blind_relay": latest_blind_relay_event_value(status, generated_at, false),
        "local_capability": {
            "status": local_capabilities.status,
            "chat_relay_configured": local_capabilities.chat_relay_configured,
            "blind_relay_endpoint_ready": local_capabilities.blind_relay_endpoint_ready,
            "chat_relay_runtime_ready": local_capabilities.chat_relay_runtime_ready,
            "safe_to_advertise_chat_relay": local_capabilities.safe_to_advertise_chat_relay,
        },
        "last_event_age_seconds": quality.last_event_age_seconds,
        "last_accepted_age_seconds": quality.last_accepted_age_seconds,
        "accepted_percent": quality.accepted_percent,
        "next_action": &quality.next_action,
        "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        "privacy_boundary": "aggregate blind relay runtime counters only; no node endpoints, route ids, selected hops, receiver keys, encrypted payloads, client IPs, DNS contents, destinations, Memory Chain plaintext, private keys, wallet-level traffic, or social graph metadata",
    });
    // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Insert these
    // fields after the legacy macro expansion so this already-large stable
    // contract does not require a crate-wide recursion-limit increase.
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "authenticated_delivery_path_ready".to_string(),
            quality.authenticated_delivery_path_ready.into(),
        );
        object.insert(
            "authenticated_delivery_path_reason".to_string(),
            quality.authenticated_delivery_path_reason.clone().into(),
        );
    }
    value
}

fn latest_blind_relay_event_value(
    status: &PeerStoreStatus,
    generated_at: u64,
    successful: bool,
) -> serde_json::Value {
    let event = status.recent_audit_events.iter().rev().find(|event| {
        event.action.starts_with("blind_relay")
            && if successful {
                event.outcome == "accepted"
            } else {
                event.outcome == "rejected" || event.outcome == "limited"
            }
    });

    match event {
        Some(event) => serde_json::json!({
            "at": event.at,
            "age_seconds": generated_at.saturating_sub(event.at),
            "action": &event.action,
            "outcome": &event.outcome,
            "reason_bucket": &event.detail,
        }),
        None => serde_json::Value::Null,
    }
}

/// Builds the compact public-safe discovery summary response.
///
/// Keep this helper intentionally narrow. `/api/discovery/status` remains the
/// operator/debug payload, while `/api/discovery/summary` is the small contract
/// for product surfaces that should not receive descriptors, endpoints, full
/// peer ids, route ids, or encrypted payload metadata.
#[must_use]
pub fn discovery_summary_response(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> DiscoverySummaryResponse {
    let readiness = discovery_readiness_status_value(status, local_capabilities);
    let protocol_foundation = &readiness["protocol_foundation"];
    let onion_relay_admission = onion_relay_admission_status_value(status, local_capabilities);
    let blind_relay_runtime =
        blind_relay_runtime_status_value(generated_at, status, local_capabilities);
    let recovery_anchor = recovery_anchor_status_value(status);
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let two_hop_history = &status.two_hop_path_proof_history;
    let three_hop_history = &status.three_hop_path_proof_history;
    let proof_restart_continuity = two_hop_proof_restart_continuity(status);
    let three_hop_restart_continuity = three_hop_proof_restart_continuity(status);
    let two_hop_restart_survivable_ready = two_hop_history.recent_message_delivery_ready
        && peer_quorum.quorum_ready
        && proof_restart_continuity.peer_recovery_configured
        && proof_restart_continuity.ready;
    let two_hop_restart_recovery_basis = if two_hop_restart_survivable_ready {
        "message_delivery_proof_with_verified_restart_continuity"
    } else if !two_hop_history.recent_message_delivery_ready {
        "waiting_for_fresh_message_delivery_proof"
    } else if !peer_quorum.quorum_ready {
        "waiting_for_peer_quorum"
    } else if !proof_restart_continuity.peer_recovery_configured {
        "restart_recovery_not_configured"
    } else {
        "proof_restart_continuity_not_ready"
    };

    let status_bucket = protocol_foundation["status"]
        .as_str()
        .unwrap_or("forming")
        .to_string();
    let stage_bucket = protocol_foundation["stage"]
        .as_str()
        .unwrap_or("bootstrap")
        .to_string();
    let headline = protocol_foundation["headline"]
        .as_str()
        .unwrap_or("AeroNyx nodes are forming a verified relay mesh")
        .to_string();
    let next_action = protocol_foundation["next_action"]
        .as_str()
        .unwrap_or("monitor verified peer discovery and relay path proof freshness")
        .to_string();
    let mut two_hop_path_proof = serde_json::Map::new();
    two_hop_path_proof.insert(
        "status".to_string(),
        serde_json::json!(&two_hop_history.status),
    );
    two_hop_path_proof.insert(
        "freshness_bucket".to_string(),
        serde_json::json!(&two_hop_history.freshness_bucket),
    );
    two_hop_path_proof.insert(
        "proof_ready".to_string(),
        serde_json::json!(two_hop_history.proof_ready),
    );
    two_hop_path_proof.insert(
        "recent_success_ready".to_string(),
        serde_json::json!(two_hop_history.recent_success_ready),
    );
    two_hop_path_proof.insert(
        "message_delivery_ready".to_string(),
        serde_json::json!(two_hop_history.message_delivery_ready),
    );
    two_hop_path_proof.insert(
        "recent_message_delivery_ready".to_string(),
        serde_json::json!(two_hop_history.recent_message_delivery_ready),
    );
    two_hop_path_proof.insert(
        "message_delivery_evidence_mode".to_string(),
        serde_json::json!(&two_hop_history.message_delivery_evidence_mode),
    );
    two_hop_path_proof.insert(
        "failure_streak_active".to_string(),
        serde_json::json!(two_hop_history.failure_streak_active),
    );
    two_hop_path_proof.insert(
        "retained_events".to_string(),
        serde_json::json!(two_hop_history.retained_events),
    );
    two_hop_path_proof.insert(
        "attempted".to_string(),
        serde_json::json!(two_hop_history.attempted),
    );
    two_hop_path_proof.insert(
        "succeeded".to_string(),
        serde_json::json!(two_hop_history.succeeded),
    );
    two_hop_path_proof.insert(
        "message_delivery_successes".to_string(),
        serde_json::json!(two_hop_history.message_delivery_successes),
    );
    two_hop_path_proof.insert(
        "failed".to_string(),
        serde_json::json!(two_hop_history.failed),
    );
    two_hop_path_proof.insert(
        "success_percent".to_string(),
        serde_json::json!(two_hop_history.success_percent),
    );
    two_hop_path_proof.insert(
        "stability_window_size".to_string(),
        serde_json::json!(two_hop_history.stability_window_size),
    );
    two_hop_path_proof.insert(
        "stability_window_attempted".to_string(),
        serde_json::json!(two_hop_history.stability_window_attempted),
    );
    two_hop_path_proof.insert(
        "stability_window_succeeded".to_string(),
        serde_json::json!(two_hop_history.stability_window_succeeded),
    );
    two_hop_path_proof.insert(
        "stability_window_failed".to_string(),
        serde_json::json!(two_hop_history.stability_window_failed),
    );
    two_hop_path_proof.insert(
        "stability_success_percent".to_string(),
        serde_json::json!(two_hop_history.stability_success_percent),
    );
    two_hop_path_proof.insert(
        "stability_status".to_string(),
        serde_json::json!(&two_hop_history.stability_status),
    );
    two_hop_path_proof.insert(
        "stability_ready".to_string(),
        serde_json::json!(two_hop_history.stability_ready),
    );
    two_hop_path_proof.insert(
        "failure_circuit_breaker_threshold".to_string(),
        serde_json::json!(two_hop_history.failure_circuit_breaker_threshold),
    );
    two_hop_path_proof.insert(
        "failure_circuit_breaker_active".to_string(),
        serde_json::json!(two_hop_history.failure_circuit_breaker_active),
    );
    two_hop_path_proof.insert(
        "latest_age_bucket".to_string(),
        serde_json::json!(&two_hop_history.latest_age_bucket),
    );
    two_hop_path_proof.insert(
        "latest_outcome".to_string(),
        serde_json::json!(&two_hop_history.latest_outcome),
    );
    two_hop_path_proof.insert(
        "latest_reason_bucket".to_string(),
        serde_json::json!(&two_hop_history.latest_reason_bucket),
    );
    two_hop_path_proof.insert(
        "latest_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_age_seconds),
    );
    two_hop_path_proof.insert(
        "latest_success_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_success_age_seconds),
    );
    two_hop_path_proof.insert(
        "latest_failure_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_failure_age_seconds),
    );
    two_hop_path_proof.insert(
        "latest_message_delivery_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_message_delivery_age_seconds),
    );
    two_hop_path_proof.insert(
        "consecutive_successes".to_string(),
        serde_json::json!(two_hop_history.consecutive_successes),
    );
    two_hop_path_proof.insert(
        "consecutive_failures".to_string(),
        serde_json::json!(two_hop_history.consecutive_failures),
    );
    two_hop_path_proof.insert(
        "consecutive_message_delivery_successes".to_string(),
        serde_json::json!(two_hop_history.consecutive_message_delivery_successes),
    );
    two_hop_path_proof.insert(
        "path_shape_counts".to_string(),
        serde_json::json!(&two_hop_history.path_shape_counts),
    );
    two_hop_path_proof.insert(
        "candidate_pool_counts".to_string(),
        serde_json::json!(&two_hop_history.candidate_pool_counts),
    );
    two_hop_path_proof.insert(
        "ttl_shape_counts".to_string(),
        serde_json::json!(&two_hop_history.ttl_shape_counts),
    );
    two_hop_path_proof.insert(
        "proof_scope".to_string(),
        serde_json::json!(&two_hop_history.proof_scope),
    );
    two_hop_path_proof.insert(
        "proof_scope_counts".to_string(),
        serde_json::json!(&two_hop_history.proof_scope_counts),
    );
    two_hop_path_proof.insert(
        "restart_recovery_configured".to_string(),
        serde_json::json!(peer_quorum.restart_recovery_configured),
    );
    two_hop_path_proof.insert(
        "peer_quorum_ready".to_string(),
        serde_json::json!(peer_quorum.quorum_ready),
    );
    two_hop_path_proof.insert(
        "restart_survivable_ready".to_string(),
        serde_json::json!(two_hop_restart_survivable_ready),
    );
    two_hop_path_proof.insert(
        "proof_restart_continuity_ready".to_string(),
        serde_json::json!(proof_restart_continuity.ready),
    );
    two_hop_path_proof.insert(
        "proof_restart_continuity_source".to_string(),
        serde_json::json!(proof_restart_continuity.source),
    );
    two_hop_path_proof.insert(
        "proof_cache_authentication".to_string(),
        serde_json::json!(proof_restart_continuity.authentication),
    );
    two_hop_path_proof.insert(
        "proof_cache_rollback_protection".to_string(),
        serde_json::json!(proof_restart_continuity.rollback_protection),
    );
    two_hop_path_proof.insert(
        "proof_cache_external_witness".to_string(),
        serde_json::json!(proof_restart_continuity.external_witness),
    );
    two_hop_path_proof.insert(
        "proof_cache_external_witness_required".to_string(),
        serde_json::json!(proof_restart_continuity.external_witness_required),
    );
    two_hop_path_proof.insert(
        "proof_cache_restored_events".to_string(),
        serde_json::json!(proof_restart_continuity.restored),
    );
    two_hop_path_proof.insert(
        "proof_cache_persisted_events".to_string(),
        serde_json::json!(proof_restart_continuity.persisted),
    );
    two_hop_path_proof.insert(
        "restart_recovery_basis".to_string(),
        serde_json::json!(two_hop_restart_recovery_basis),
    );
    two_hop_path_proof.insert(
        "stale_after_seconds".to_string(),
        serde_json::json!(two_hop_history.stale_after_seconds),
    );
    two_hop_path_proof.insert(
        "next_action".to_string(),
        serde_json::json!(&two_hop_history.next_action),
    );

    DiscoverySummaryResponse {
        generated_at,
        contract_version: "discovery_summary.v1",
        source: "rust_discovery_summary",
        protocol_features: serde_json::json!({
            "legacy_descriptor_gossip_v1": true,
            "directory_descriptor_proof_gossip_v1": true,
            // [THREE-HOP-FEATURE-NEGOTIATION 2026-08-02 by Codex] This is an
            // unsigned transport hint only. A successful path still requires
            // the terminal's signed, route-bound delivery receipt.
            "multihop_delivery_receipt_v1": true,
            // [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Keep v2
            // separate from v1 so a legacy relay cannot be selected for a
            // workload-bound proof merely because it understands ACK framing.
            "purpose_bound_delivery_receipt_v2": true,
            // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Canonical values come
            // from aeronyx-core so all implementations negotiate one contract.
            "onion_route_purpose_v1": true,
            "onion_route_purposes": ONION_ROUTE_PURPOSE_VALUES,
            // Operational hint only; route selection trusts the matching
            // token in each terminal's signed descriptor.
            "blind_vault_encrypted_terminal_failure_v1": true,
        }),
        status: status_bucket,
        stage: stage_bucket,
        headline,
        local_capability: serde_json::json!({
            "status": local_capabilities.status,
            "chat_relay_configured": local_capabilities.chat_relay_configured,
            "blind_relay_endpoint_ready": local_capabilities.blind_relay_endpoint_ready,
            "chat_relay_runtime_ready": local_capabilities.chat_relay_runtime_ready,
            "safe_to_advertise_chat_relay": local_capabilities.safe_to_advertise_chat_relay,
            "capability_config_consistent": local_capabilities.capability_config_consistent,
            "advertisement_blockers": &local_capabilities.advertisement_blockers,
            "blind_vault_replica_configured": local_capabilities.blind_vault_replica_configured,
            "blind_vault_runtime_ready": local_capabilities.blind_vault_runtime_ready,
            "advertised_blind_vault_replica_capability": local_capabilities.advertised_blind_vault_replica_capability,
            "safe_to_advertise_blind_vault_replica": local_capabilities.safe_to_advertise_blind_vault_replica,
            "blind_vault_capability_consistent": local_capabilities.blind_vault_capability_consistent,
            "blind_vault_advertisement_blockers": &local_capabilities.blind_vault_advertisement_blockers,
        }),
        peer_mesh: serde_json::json!({
            "status": &peer_quorum.status,
            "quorum_ready": peer_quorum.quorum_ready,
            "valid_peers": peer_quorum.valid_peers,
            "healthy_peers": peer_quorum.healthy_peers,
            "stale_peers": peer_quorum.stale_peers,
            "min_valid_peers": peer_quorum.min_valid_peers,
            "routeable_chat_relays": peer_quorum.routeable_chat_relays,
            "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
            "restart_recovery_configured": peer_quorum.restart_recovery_configured,
            "relay_foundation_ready": peer_quorum.relay_foundation_ready,
            "network_story_status": &network_story.status,
            "chat_single_hop_ready": network_story.chat_single_hop_ready,
            "chat_two_hop_onion_ready": network_story.chat_two_hop_onion_ready,
        }),
        route_governance: serde_json::json!(&status.route_governance),
        blind_relay: serde_json::json!({
            "status": &blind_relay_quality.status,
            "runtime_ready": blind_relay_quality.runtime_ready,
            "quality_ready": blind_relay_quality.quality_ready,
            "real_relay_ready": blind_relay_quality.real_relay_ready,
            "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
            "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
            "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
            "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
            "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
            "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
            "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
            "evidence_mode": &blind_relay_quality.evidence_mode,
            "readiness_reason": &blind_relay_quality.readiness_reason,
            "accepted_total": blind_relay_quality.accepted_total,
            "forward_failed": blind_relay_quality.forward_failed,
            "timestamp_rejected": blind_relay_quality.timestamp_rejected,
            "last_event_age_seconds": blind_relay_quality.last_event_age_seconds,
            "last_probe_age_seconds": blind_relay_quality.last_probe_age_seconds,
            "next_action": &blind_relay_quality.next_action,
        }),
        blind_relay_runtime,
        two_hop_path_proof: serde_json::Value::Object(two_hop_path_proof),
        three_hop_path_proof: serde_json::json!({
            "status": &three_hop_history.status,
            "freshness_bucket": &three_hop_history.freshness_bucket,
            "proof_ready": three_hop_history.proof_ready,
            "recent_success_ready": three_hop_history.recent_success_ready,
            "message_delivery_ready": three_hop_history.message_delivery_ready,
            "recent_message_delivery_ready": three_hop_history.recent_message_delivery_ready,
            "attempted": three_hop_history.attempted,
            "succeeded": three_hop_history.succeeded,
            "failed": three_hop_history.failed,
            "success_percent": three_hop_history.success_percent,
            "latest_age_bucket": &three_hop_history.latest_age_bucket,
            "latest_reason_bucket": &three_hop_history.latest_reason_bucket,
            "latest_message_delivery_age_seconds": three_hop_history.latest_message_delivery_age_seconds,
            "path_shape_counts": &three_hop_history.path_shape_counts,
            "ttl_shape_counts": &three_hop_history.ttl_shape_counts,
            "proof_scope": &three_hop_history.proof_scope,
            "persistence": "signed_local_cache_with_runtime_revalidation",
            "proof_restart_continuity_ready": three_hop_restart_continuity.ready,
            "proof_restart_continuity_source": three_hop_restart_continuity.source,
            "proof_cache_authentication": three_hop_restart_continuity.authentication,
            "proof_cache_rollback_protection": three_hop_restart_continuity.rollback_protection,
            "proof_cache_external_witness": three_hop_restart_continuity.external_witness,
            "proof_cache_external_witness_required": three_hop_restart_continuity.external_witness_required,
            "proof_cache_restored_events": three_hop_restart_continuity.restored,
            "proof_cache_persisted_events": three_hop_restart_continuity.persisted,
            "rollback_boundary": "signed_section_plus_monotonic_local_anchor; whole_host_rollback_is_fail_closed_when_external_witness_is_required",
            "privacy_boundary": &three_hop_history.privacy_boundary,
        }),
        onion_relay_admission,
        recovery_anchor,
        next_action,
        privacy_invariant: "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        privacy_boundary: "aggregate discovery summary only; no signed descriptors, full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, wallet-level traffic, or social graph metadata",
    }
}

/// Builds the smallest product-facing discovery card response.
///
/// `/api/discovery/public-card` is for first-level UX surfaces: website home
/// cards, Nodeboard overview, App status modules, and AI-agent quick health
/// checks. It intentionally compresses detailed route governance into a few
/// stable readiness signals so product surfaces can feel trustworthy without
/// leaking route metadata or overwhelming users with raw diagnostics.
#[must_use]
pub fn discovery_public_card_response(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> DiscoveryPublicCardResponse {
    let readiness = discovery_readiness_status_value(status, local_capabilities);
    let protocol_foundation = &readiness["protocol_foundation"];
    let onion_relay_admission = onion_relay_admission_status_value(status, local_capabilities);
    let peer_quorum = &status.peer_quorum;
    let blind_relay_quality = &status.blind_relay_quality;
    let relay_stats = &status.runtime.blind_relay;
    let two_hop_history = &status.two_hop_path_proof_history;
    let real_delivery_evidence_seen = blind_relay_quality.verified_client_onion_deliveries > 0;
    let message_delivery_proof_ready =
        blind_relay_quality.real_relay_ready || two_hop_history.proof_ready;
    let message_delivery_ready =
        blind_relay_quality.real_relay_ready || two_hop_history.message_delivery_ready;
    let latest_delivery_proof_age_seconds = if real_delivery_evidence_seen {
        blind_relay_quality.last_verified_client_onion_delivery_age_seconds
    } else {
        two_hop_history.latest_age_seconds
    };

    let status_bucket = protocol_foundation["status"]
        .as_str()
        .unwrap_or("forming")
        .to_string();
    let stage_bucket = protocol_foundation["stage"]
        .as_str()
        .unwrap_or("bootstrap")
        .to_string();
    let headline = protocol_foundation["headline"]
        .as_str()
        .unwrap_or("AeroNyx nodes are forming a verified relay mesh")
        .to_string();
    let next_action = protocol_foundation["next_action"]
        .as_str()
        .unwrap_or("monitor verified peer discovery and relay path proof freshness")
        .to_string();
    let checks_passed = protocol_foundation["checks_passed"].as_u64().unwrap_or(0);
    let checks_total = protocol_foundation["checks_total"]
        .as_u64()
        .unwrap_or(4)
        .max(1);
    let foundation_confidence = ((checks_passed * 100) / checks_total).min(100) as u8;
    let admission_score = onion_relay_admission["admission_score_percent"]
        .as_u64()
        .unwrap_or(0)
        .min(100) as u8;
    let confidence_percent =
        ((u16::from(foundation_confidence) * 2 + u16::from(admission_score)) / 3).min(100) as u8;
    let health_label = match status_bucket.as_str() {
        "ready" => "Live protocol",
        "live" => "Relay evidence live",
        "forming" => "Mesh forming",
        "disabled" => "Not advertising",
        "pending" => "Waiting for peers",
        _ => "Protocol warming",
    };
    let two_hop_ready = protocol_foundation["two_hop_onion_ready"]
        .as_bool()
        .unwrap_or(false);

    DiscoveryPublicCardResponse {
        generated_at,
        contract_version: DISCOVERY_PUBLIC_CARD_CONTRACT_VERSION,
        source: DISCOVERY_PUBLIC_CARD_SOURCE,
        status: status_bucket,
        stage: stage_bucket,
        headline,
        health_label,
        confidence_percent,
        cards: serde_json::json!({
            "protocol_health": {
                "label": "AeroNyx Privacy Protocol",
                "status": protocol_foundation["status"],
                "stage": protocol_foundation["stage"],
                "confidence_percent": confidence_percent,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "two_hop_onion_ready": two_hop_ready,
                "restart_recovery_ready": protocol_foundation["restart_recovery_ready"],
            },
            "verified_mesh": {
                "label": "Verified Node Mesh",
                "status": &peer_quorum.status,
                "healthy_peers": peer_quorum.healthy_peers,
                "valid_peers": peer_quorum.valid_peers,
                "routeable_relays": peer_quorum.routeable_chat_relays,
                "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
                "restart_recovery_configured": peer_quorum.restart_recovery_configured,
            },
            "blind_relay": {
                "label": "Blind Relay",
                "status": &blind_relay_quality.status,
                "runtime_ready": blind_relay_quality.runtime_ready,
                "real_relay_ready": blind_relay_quality.real_relay_ready,
                "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
                "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
                "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
                "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
                "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
                "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
                "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
                "proof_ready": message_delivery_proof_ready,
                "message_delivery_ready": message_delivery_ready,
                "message_delivery_evidence_mode": &blind_relay_quality.evidence_mode,
                "latest_proof_age_seconds": latest_delivery_proof_age_seconds,
                "terminal_delivered_count": relay_stats.terminal,
                "middle_forwarded_count": relay_stats.forwarded,
            }
        }),
        signals: serde_json::json!({
            "local_relay_ready": protocol_foundation["local_relay_ready"],
            "peer_mesh_ready": protocol_foundation["peer_mesh_ready"],
            "blind_relay_ready": protocol_foundation["blind_relay_ready"],
            "two_hop_onion_ready": two_hop_ready,
            "onion_admission_status": onion_relay_admission["status"],
            "onion_admission_eligible": onion_relay_admission["eligible"],
            "onion_admission_score_percent": admission_score,
            "onion_warmup_stage": onion_relay_admission["warmup_stage"],
            "stable_path_proof_ready": onion_relay_admission["stable_path_proof_ready"],
            "failure_circuit_breaker_active": two_hop_history.failure_circuit_breaker_active,
            "failure_streak_active": two_hop_history.failure_streak_active,
            "latest_path_proof_outcome": &two_hop_history.latest_outcome,
            "latest_path_proof_reason_bucket": &two_hop_history.latest_reason_bucket,
            "latest_path_proof_age_bucket": &two_hop_history.latest_age_bucket,
            "permissionless_node_admission": true,
        }),
        display_policy: serde_json::json!({
            "primary_surface": "show_protocol_health_verified_mesh_and_blind_relay",
            "detail_surface": "link_to_discovery_summary_or_nodeboard_detail_for_diagnostics",
            "recommended_cards": ["protocol_health", "verified_mesh", "blind_relay"],
            "avoid_first_level_fields": [
                "signed_descriptors",
                "raw_route_metadata",
                "raw_audit_events",
                "raw_peer_diagnostics",
                "path_selection_details"
            ],
        }),
        next_action,
        privacy_invariant: "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        privacy_boundary: "public protocol card aggregates only; no signed descriptors, full node ids, endpoint URLs, route ids, selected hops, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, wallet-level traffic, or social graph metadata",
    }
}

// ============================================
// Router
// ============================================

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
    build_discovery_router_state(
        peer_store,
        policy,
        local_capabilities,
        directory_replica_store,
        Some(local_node_id),
    )
}

fn build_discovery_router_state(
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    directory_replica_store: Option<Arc<DirectoryReplicaStore>>,
    local_node_id: Option<[u8; 32]>,
) -> Router {
    let state = DiscoveryApiState {
        peer_store,
        local_node_id,
        directory_replica_store,
        policy,
        local_capabilities,
        rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
        route_domain_certificate_rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
    };
    Router::new()
        .route("/api/discovery/snapshot", get(snapshot_handler))
        .route("/api/discovery/gossip", post(gossip_handler))
        .route(
            "/api/discovery/route-domain-certificate",
            post(route_domain_certificate_handler).layer(DefaultBodyLimit::max(
                MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES,
            )),
        )
        .route("/api/discovery/status", get(status_handler))
        .route("/api/discovery/summary", get(summary_handler))
        .route("/api/discovery/public-card", get(public_card_handler))
        .route(
            "/api/discovery/onion-candidates",
            get(onion_candidates_handler),
        )
        .layer(DefaultBodyLimit::max(DISCOVERY_REQUEST_BODY_MAX_BYTES))
        .with_state(state)
}

// ============================================
// Handlers
// ============================================

async fn snapshot_handler(
    State(state): State<DiscoveryApiState>,
    Query(query): Query<SnapshotQuery>,
) -> Json<NodeBootstrapSnapshot> {
    let now = now_secs();
    let limit = state.policy.snapshot_limit(query.limit);
    Json(state.peer_store.export_bootstrap_snapshot(
        now,
        now,
        query.public_only.unwrap_or(true),
        Some(limit),
    ))
}

/// `GET /api/discovery/onion-candidates` — health-ranked onion relay candidates
/// for client-side path selection.
///
/// Each candidate advertises a KEM public key (so the client can build an onion
/// layer addressed to it) and a reachable public endpoint. Only signed, public
/// node discovery metadata is exposed — never client traffic, route ids, or
/// payloads. Candidates without a KEM key or a public endpoint are filtered out
/// (they cannot serve as an onion hop). Candidates also need fresh routeability
/// evidence from local probes or successful forwards; signed descriptors prove
/// identity/capability, but they do not prove the endpoint is currently usable.
/// Because the KEM key rotates on the relay's onion-key schedule, clients should
/// fetch fresh candidates rather than caching keys for long periods.
async fn onion_candidates_handler(
    State(state): State<DiscoveryApiState>,
    Query(query): Query<OnionCandidatesQuery>,
) -> Json<OnionCandidatesResponse> {
    let now = now_secs();
    let limit = state.policy.snapshot_limit(query.limit);
    let requested_purpose = onion_route_purpose_from_query(query.purpose.as_deref());
    let terminal_requirement = OnionTerminalRequirement::for_purpose(requested_purpose);
    let requested_privacy_mode = OnionPrivacyMode::from_query(query.privacy_mode.as_deref());
    let requested_hops = normalize_requested_hops(requested_privacy_mode, query.hops);
    let pinned_route_domain_required =
        requested_hops >= 2 && state.policy.require_pinned_route_domains_for_multi_hop;
    let local_pinned_route_domain = state
        .local_node_id
        .and_then(|node_id| state.policy.pinned_route_domain(&node_id));
    let local_descriptor = state
        .local_node_id
        .and_then(|node_id| state.peer_store.get_valid(&node_id, now));
    // [ONION-CAPABILITY-GATE 2026-08-02 by Codex] Query the bounded policy
    // pool first, then apply every onion-hop eligibility rule before the
    // client limit and ranking. Limiting earlier can let ineligible high-rank
    // ChatRelay peers hide valid OnionMiddle relays below them.
    let eligible_candidates: Vec<OnionRelayCandidate> = state
        .peer_store
        .route_candidates_with_capability(
            NodeCapability::ChatRelay,
            now,
            state.policy.max_snapshot_limit,
        )
        .into_iter()
        .filter_map(|descriptor| {
            let node_id = descriptor.node_id();
            if state.local_node_id == Some(node_id) {
                return None;
            }
            if !descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::OnionMiddle)
            {
                return None;
            }
            if !state.peer_store.is_routeable_now(&node_id, now) {
                return None;
            }
            let kem_public = descriptor.descriptor.x25519_kem_public()?;
            let public_endpoint = descriptor.descriptor.public_endpoint.clone()?;
            // [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] A first remote
            // hop collocated with the entry weakens the route before pairwise
            // candidate diversity is considered. Missing/malformed entry or
            // candidate endpoints fail this production gate closed.
            if local_descriptor.as_ref().is_some_and(|local_descriptor| {
                !PeerStore::route_endpoints_are_network_diverse(local_descriptor, &descriptor)
            }) {
                return None;
            }
            // [PINNED-ROUTE-DOMAINS 2026-08-03 by Codex] Known same-domain
            // entry/candidate pairs are excluded even before strict rollout.
            // Strict mode additionally requires complete remote coverage; the
            // local-entry coverage check remains a separate fail-closed gate.
            let candidate_route_domain = state.policy.pinned_route_domain(&node_id);
            if local_pinned_route_domain
                .zip(candidate_route_domain)
                .is_some_and(|(local, candidate)| local == candidate)
            {
                return None;
            }
            if pinned_route_domain_required && candidate_route_domain.is_none() {
                return None;
            }
            Some((descriptor, kem_public, public_endpoint))
        })
        .enumerate()
        .map(|(rank, (descriptor, kem_public, public_endpoint))| {
            let capacity = descriptor.descriptor.capacity.clone();
            OnionRelayCandidate {
                node_id: hex::encode(descriptor.node_id()),
                kem_alg: descriptor.descriptor.kem_alg,
                kem_public: hex::encode(kem_public),
                public_endpoint,
                capabilities: descriptor.descriptor.capabilities.clone(),
                selection_weight: onion_candidate_selection_weight(rank),
                region: descriptor.descriptor.policy.region.clone(),
                max_sessions: capacity.max_sessions,
                max_bps: capacity.max_bps,
                max_pps: capacity.max_pps,
                signed_descriptor: descriptor,
            }
        })
        .collect();
    // [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] The health-ranked
    // weights above belong to the full eligible pool and remain unchanged.
    // Preserve a diverse requested path before applying a small response
    // limit, then let the client independently choose the actual route.
    let eligible_candidate_count = eligible_candidates.len();
    let candidates = select_onion_candidate_response_pool_with_policy_and_terminal(
        eligible_candidates,
        limit,
        requested_hops as usize,
        &state.policy,
        pinned_route_domain_required,
        terminal_requirement,
    );
    let two_hop_ready = candidates.len() >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;
    let min_candidates_for_requested_hops = requested_hops as usize;
    let terminal_candidate_count = candidates
        .iter()
        .filter(|candidate| onion_terminal_candidate_matches(requested_purpose, candidate))
        .count();
    let requested_terminal_capability_ready = terminal_candidate_count > 0;
    // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] The legacy message purpose has
    // no terminal role beyond the base candidate contract. Its empty-pool
    // state must therefore remain `no_routeable_candidates`; only a purpose
    // with an additional signed terminal capability uses the new terminal
    // readiness gate.
    let terminal_capability_gate_ready =
        !terminal_requirement.is_specialized() || requested_terminal_capability_ready;
    let purpose_admission = OnionPurposeAdmission {
        supported: requested_purpose.is_some(),
        terminal_capability_ready: terminal_capability_gate_ready,
    };
    // [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] Legacy builders have no
    // entry context and retain their historical pairwise behavior. Production
    // builders inject an id and fail multi-hop readiness closed until its
    // signed descriptor can be resolved and used for entry anti-affinity.
    let local_entry_context_ready = state.local_node_id.is_none() || local_descriptor.is_some();
    let requested_network_diversity_ready = local_entry_context_ready
        && onion_candidate_route_diversity_ready_for_terminal(
            &candidates,
            min_candidates_for_requested_hops,
            &DiscoveryApiPolicy::default(),
            false,
            terminal_requirement,
        );
    let local_entry_pinned_route_domain_enforced =
        state.local_node_id.is_some() && local_pinned_route_domain.is_some();
    let requested_pinned_route_domain_ready = !pinned_route_domain_required
        || (local_entry_pinned_route_domain_enforced
            && onion_candidate_route_diversity_ready_for_terminal(
                &candidates,
                min_candidates_for_requested_hops,
                &state.policy,
                true,
                terminal_requirement,
            ));
    let peer_status = state.peer_store.status(now);
    let requested_path_gates = onion_requested_path_gates(
        &peer_status,
        requested_hops,
        OnionCandidateAdmissionInput {
            purpose: purpose_admission,
            candidate_pool_ready: limit >= min_candidates_for_requested_hops
                && eligible_candidate_count >= min_candidates_for_requested_hops,
            network_diversity_ready: requested_network_diversity_ready,
            pinned_route_domain: OnionRequirementGate {
                required: pinned_route_domain_required,
                ready: requested_pinned_route_domain_ready,
            },
        },
    );
    let requested_path_ready = requested_path_gates.ready();
    let two_hop_network_diversity_ready = local_entry_context_ready
        && onion_candidate_route_diversity_ready_for_terminal(
            &candidates,
            ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
            &DiscoveryApiPolicy::default(),
            false,
            terminal_requirement,
        );
    let two_hop_fallback_ready = requested_hops > 2
        && onion_requested_path_gates(
            &peer_status,
            2,
            OnionCandidateAdmissionInput {
                purpose: purpose_admission,
                candidate_pool_ready: limit >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
                    && eligible_candidate_count >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
                network_diversity_ready: two_hop_network_diversity_ready,
                pinned_route_domain: OnionRequirementGate {
                    required: state.policy.require_pinned_route_domains_for_multi_hop,
                    ready: !state.policy.require_pinned_route_domains_for_multi_hop
                        || (local_entry_pinned_route_domain_enforced
                            && onion_candidate_route_diversity_ready_for_terminal(
                                &candidates,
                                ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
                                &state.policy,
                                true,
                                terminal_requirement,
                            )),
                },
            },
        )
        .ready();
    let recommended_hops = if requested_purpose.is_some() && terminal_capability_gate_ready {
        recommended_onion_hops(
            candidates.len(),
            requested_hops,
            requested_path_ready,
            two_hop_fallback_ready,
        )
    } else {
        0
    };
    let fallback_reason = onion_candidate_fallback_reason(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
        requested_path_gates,
    );
    let pool_status = onion_candidate_pool_status(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
        requested_path_gates,
    );
    let route_plan = onion_candidate_route_plan(
        requested_path_ready,
        requested_hops,
        recommended_hops,
        fallback_reason,
    );
    let readiness_reason = onion_candidate_readiness_reason(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
        requested_path_gates,
    );
    let next_action =
        onion_candidate_next_action(requested_path_ready, recommended_hops, fallback_reason);

    Json(OnionCandidatesResponse {
        generated_at: now,
        contract_version: ONION_CANDIDATES_CONTRACT_VERSION.to_string(),
        source: ONION_CANDIDATES_SOURCE.to_string(),
        required_capabilities: onion_required_capabilities(),
        requested_purpose: onion_route_purpose_name(requested_purpose).to_string(),
        requested_purpose_supported: requested_purpose.is_some(),
        terminal_required_capabilities: onion_terminal_required_capabilities(requested_purpose),
        terminal_candidate_count,
        requested_terminal_capability_ready,
        count: candidates.len(),
        min_candidates_for_two_hop: ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        two_hop_ready,
        requested_privacy_mode: requested_privacy_mode.as_str().to_string(),
        requested_hops,
        min_candidates_for_requested_hops,
        requested_path_ready,
        requested_candidate_pool_ready: requested_path_gates.candidate_pool_ready,
        requested_network_diversity_required: requested_path_gates.network_diversity_required,
        requested_network_diversity_ready: requested_path_gates.network_diversity_ready,
        local_entry_network_diversity_enforced: local_descriptor.is_some(),
        requested_pinned_route_domain_required: requested_path_gates
            .pinned_route_domain_required,
        requested_pinned_route_domain_ready: requested_path_gates.pinned_route_domain_ready,
        local_entry_pinned_route_domain_enforced,
        requested_runtime_proof_required: requested_path_gates.runtime_proof_required,
        requested_runtime_proof_ready: requested_path_gates.runtime_proof_ready,
        requested_restart_continuity_required: requested_path_gates
            .restart_continuity_required,
        requested_restart_continuity_ready: requested_path_gates.restart_continuity_ready,
        recommended_hops,
        fallback_required: !requested_path_ready,
        pool_status: pool_status.to_string(),
        route_plan: route_plan.to_string(),
        fallback_reason: fallback_reason.to_string(),
        readiness_reason: readiness_reason.to_string(),
        next_action: next_action.to_string(),
        selection_policy: ONION_CANDIDATES_SELECTION_POLICY.to_string(),
        candidate_verification: "signed_node_descriptor_ed25519_v2".to_string(),
        path_selection_strategy: "weighted_random_health_ranked_distinct_hops".to_string(),
        network_diversity_policy: match (state.local_node_id.is_some(), local_descriptor.is_some()) {
            (_, true) => "required_against_local_entry_and_pairwise_ipv4_24_ipv6_48_or_distinct_dns_hostnames; not_operator_or_as_proof",
            (true, false) => "local_entry_descriptor_unavailable_fail_closed; required_pairwise_ipv4_24_ipv6_48_or_distinct_dns_hostnames; not_operator_or_as_proof",
            (false, false) => "required_pairwise_ipv4_24_ipv6_48_or_distinct_dns_hostnames; legacy_local_entry_context_unavailable; not_operator_or_as_proof",
        }
        .to_string(),
        pinned_route_domain_policy: if state
            .policy
            .require_pinned_route_domains_for_multi_hop
        {
            "required_for_multi_hop; operator_audited_local_opaque_assignments; distinct_entry_and_remote_domains; not_permissionless_consensus_as_proof_or_sybil_resistance"
        } else if state.policy.pinned_route_domains.is_empty() {
            "disabled_backward_compatible; coarse_endpoint_anti_affinity_only"
        } else {
            "best_effort_known_same_domain_exclusion; incomplete_assignments_allowed; not_permissionless_consensus_as_proof_or_sybil_resistance"
        }
        .to_string(),
        region_diversity_policy:
            "prefer_distinct_regions_when_available_without_exposing_selected_route".to_string(),
        user_choice_policy:
            "users_choose_privacy_mode; clients select distinct routeable relays automatically"
                .to_string(),
        refresh_after_seconds: ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
        routeability_stale_after_seconds: ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        candidates,
        privacy_boundary: "fresh routeable signed node discovery metadata with the original public descriptor proof (node id, KEM public key, public endpoint, capabilities, capacity, and region); no client IPs, route ids, encrypted payloads, receiver identities, DNS contents, destinations, voucher secrets, private keys, wallet-level traffic, or social graph metadata".to_string(),
    })
}

fn normalize_requested_hops(mode: OnionPrivacyMode, requested: Option<u8>) -> u8 {
    requested
        .unwrap_or_else(|| mode.default_hops())
        .clamp(1, ONION_CANDIDATES_MAX_CLIENT_HOPS)
}

fn onion_requested_path_gates(
    status: &PeerStoreStatus,
    requested_hops: u8,
    admission: OnionCandidateAdmissionInput,
) -> OnionRequestedPathGates {
    let network_diversity_required = requested_hops >= 2;
    let runtime_proof_required = requested_hops >= 2;
    let restart_continuity_required = requested_hops >= 2;
    let (runtime_proof_ready, restart_continuity_ready) = match requested_hops {
        3.. => {
            let proof = &status.three_hop_path_proof_history;
            let continuity = three_hop_proof_restart_continuity(status);
            (
                proof.recent_message_delivery_ready
                    && proof.stability_ready
                    && !proof.failure_streak_active
                    && !proof.failure_circuit_breaker_active,
                continuity.peer_recovery_configured && continuity.ready,
            )
        }
        2 => {
            let proof = &status.two_hop_path_proof_history;
            let continuity = two_hop_proof_restart_continuity(status);
            (
                proof.recent_message_delivery_ready
                    && proof.stability_ready
                    && !proof.failure_streak_active
                    && !proof.failure_circuit_breaker_active,
                continuity.peer_recovery_configured && continuity.ready,
            )
        }
        _ => (true, true),
    };

    OnionRequestedPathGates {
        purpose_supported: admission.purpose.supported,
        terminal_capability_ready: admission.purpose.terminal_capability_ready,
        candidate_pool_ready: admission.candidate_pool_ready,
        network_diversity_required,
        network_diversity_ready: admission.network_diversity_ready,
        pinned_route_domain_required: admission.pinned_route_domain.required,
        pinned_route_domain_ready: admission.pinned_route_domain.ready,
        runtime_proof_required,
        runtime_proof_ready,
        restart_continuity_required,
        restart_continuity_ready,
    }
}

/// Returns whether the public candidate set contains a pairwise-diverse path.
///
/// [ONION-NETWORK-DIVERSITY 2026-08-03 by Codex] This bounded backtracking
/// search shares the internal path planner's endpoint anti-affinity rule. It
/// returns only one aggregate decision and never exposes the selected subset.
fn onion_candidates_are_route_diverse(
    left: &OnionRelayCandidate,
    right: &OnionRelayCandidate,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
) -> bool {
    if !PeerStore::route_endpoints_are_network_diverse(
        &left.signed_descriptor,
        &right.signed_descriptor,
    ) {
        return false;
    }

    match (
        policy.pinned_route_domain(&left.signed_descriptor.node_id()),
        policy.pinned_route_domain(&right.signed_descriptor.node_id()),
    ) {
        (Some(left), Some(right)) => left != right,
        _ => !require_pinned_route_domains,
    }
}

#[cfg(test)]
fn onion_candidate_route_diverse_subset_indices(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
) -> Option<Vec<usize>> {
    onion_candidate_route_diverse_subset_indices_for_terminal(
        candidates,
        required_hops,
        policy,
        require_pinned_route_domains,
        OnionTerminalRequirement::default(),
    )
}

fn onion_candidate_supports_specialized_terminal(
    candidate: &OnionRelayCandidate,
    terminal_requirement: OnionTerminalRequirement,
) -> bool {
    terminal_requirement.matches(candidate)
}

#[derive(Clone, Copy)]
struct OnionTerminalSubsetPolicy<'a> {
    route_policy: &'a DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
}

/// Finds a diverse candidate subset that includes the required terminal role.
///
/// [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] A generic diverse subset is not
/// sufficient for encrypted storage: the terminal-capable node must itself be
/// inside that subset. This remains bounded by the existing candidate limit
/// and returns indices only to the in-process pool constructor.
fn onion_candidate_route_diverse_subset_indices_for_terminal(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
) -> Option<Vec<usize>> {
    fn search(
        candidates: &[OnionRelayCandidate],
        start: usize,
        remaining: usize,
        selected: &mut Vec<usize>,
        subset_policy: OnionTerminalSubsetPolicy<'_>,
        terminal_satisfied: bool,
    ) -> bool {
        if remaining == 0 {
            return terminal_satisfied;
        }
        if candidates.len().saturating_sub(start) < remaining {
            return false;
        }
        if !terminal_satisfied
            && !candidates[start..].iter().any(|candidate| {
                onion_candidate_supports_specialized_terminal(
                    candidate,
                    subset_policy.terminal_requirement,
                )
            })
        {
            return false;
        }

        for index in start..candidates.len() {
            let candidate = &candidates[index];
            if subset_policy.require_pinned_route_domains
                && subset_policy
                    .route_policy
                    .pinned_route_domain(&candidate.signed_descriptor.node_id())
                    .is_none()
            {
                continue;
            }
            if selected.iter().any(|selected_index| {
                !onion_candidates_are_route_diverse(
                    candidate,
                    &candidates[*selected_index],
                    subset_policy.route_policy,
                    subset_policy.require_pinned_route_domains,
                )
            }) {
                continue;
            }
            let candidate_satisfies_terminal = onion_candidate_supports_specialized_terminal(
                candidate,
                subset_policy.terminal_requirement,
            );
            selected.push(index);
            if search(
                candidates,
                index + 1,
                remaining - 1,
                selected,
                subset_policy,
                terminal_satisfied || candidate_satisfies_terminal,
            ) {
                return true;
            }
            selected.pop();
        }
        false
    }

    let subset_policy = OnionTerminalSubsetPolicy {
        route_policy: policy,
        require_pinned_route_domains,
        terminal_requirement,
    };
    if required_hops == 0 {
        return (!terminal_requirement.is_specialized()).then(Vec::new);
    }
    if candidates.len() < required_hops {
        return None;
    }
    let mut selected = Vec::with_capacity(required_hops);
    if search(
        candidates,
        0,
        required_hops,
        &mut selected,
        subset_policy,
        !terminal_requirement.is_specialized(),
    ) {
        Some(selected)
    } else {
        None
    }
}

#[cfg(test)]
fn onion_candidate_network_diverse_subset_indices(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
) -> Option<Vec<usize>> {
    onion_candidate_route_diverse_subset_indices(
        candidates,
        required_hops,
        &DiscoveryApiPolicy::default(),
        false,
    )
}

#[cfg(test)]
fn onion_candidate_network_diversity_ready(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
) -> bool {
    onion_candidate_network_diverse_subset_indices(candidates, required_hops).is_some()
}

fn onion_candidate_route_diversity_ready_for_terminal(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
) -> bool {
    onion_candidate_route_diverse_subset_indices_for_terminal(
        candidates,
        required_hops,
        policy,
        require_pinned_route_domains,
        terminal_requirement,
    )
    .is_some()
}

/// Produces the bounded public pool without hiding a valid diverse path.
///
/// [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] Ranking remains the
/// health-derived ordering/weight from the full eligible pool. When the first
/// `limit` entries are collocated, one lower-ranked candidate may be promoted
/// into the response only to preserve a pairwise-diverse requested path. For a
/// three-hop request that is not diverse-ready, a diverse two-hop subset is
/// preserved as the safe fallback. This is pool construction, not server-side
/// route selection; no chosen path, route id, or client metadata is exposed.
#[cfg(test)]
fn select_onion_candidate_response_pool(
    candidates: Vec<OnionRelayCandidate>,
    limit: usize,
    requested_hops: usize,
) -> Vec<OnionRelayCandidate> {
    select_onion_candidate_response_pool_with_policy(
        candidates,
        limit,
        requested_hops,
        &DiscoveryApiPolicy::default(),
        false,
    )
}

#[cfg(test)]
fn select_onion_candidate_response_pool_with_policy(
    candidates: Vec<OnionRelayCandidate>,
    limit: usize,
    requested_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
) -> Vec<OnionRelayCandidate> {
    select_onion_candidate_response_pool_with_policy_and_terminal(
        candidates,
        limit,
        requested_hops,
        policy,
        require_pinned_route_domains,
        OnionTerminalRequirement::default(),
    )
}

fn select_onion_candidate_response_pool_with_policy_and_terminal(
    candidates: Vec<OnionRelayCandidate>,
    limit: usize,
    requested_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
) -> Vec<OnionRelayCandidate> {
    if limit == 0 {
        return Vec::new();
    }
    if candidates.len() <= limit && !require_pinned_route_domains {
        return candidates;
    }

    let requested_subset = ((requested_hops >= 2 || terminal_requirement.is_specialized())
        && limit >= requested_hops)
        .then(|| {
            onion_candidate_route_diverse_subset_indices_for_terminal(
                &candidates,
                requested_hops,
                policy,
                require_pinned_route_domains,
                terminal_requirement,
            )
        })
        .flatten();
    let fallback_subset = (requested_hops > ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        && limit >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES)
        .then(|| {
            onion_candidate_route_diverse_subset_indices_for_terminal(
                &candidates,
                ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
                policy,
                require_pinned_route_domains,
                terminal_requirement,
            )
        })
        .flatten();
    // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Preserve one healthy
    // specialized terminal even when the requested multi-hop subset is not
    // yet mature. This keeps purpose readiness observable under a small client
    // limit; route gates still defer delivery until the requested path is safe.
    let terminal_fallback = terminal_requirement
        .is_specialized()
        .then(|| {
            candidates
                .iter()
                .position(|candidate| {
                    onion_candidate_supports_specialized_terminal(candidate, terminal_requirement)
                        && (!require_pinned_route_domains
                            || policy
                                .pinned_route_domain(&candidate.signed_descriptor.node_id())
                                .is_some())
                })
                .map(|index| vec![index])
        })
        .flatten();
    let preferred_indices = requested_subset
        .or(fallback_subset)
        .or(terminal_fallback)
        .unwrap_or_default();

    let mut selected = Vec::with_capacity(limit.min(candidates.len()));
    let mut included = vec![false; candidates.len()];
    for index in preferred_indices {
        if selected.len() >= limit {
            break;
        }
        if let Some(candidate) = candidates.get(index) {
            included[index] = true;
            selected.push(candidate.clone());
        }
    }
    for (index, candidate) in candidates.into_iter().enumerate() {
        if selected.len() >= limit {
            break;
        }
        if !included[index]
            && (!require_pinned_route_domains
                || (policy
                    .pinned_route_domain(&candidate.signed_descriptor.node_id())
                    .is_some()
                    && selected.iter().all(|selected_candidate| {
                        onion_candidates_are_route_diverse(
                            &candidate,
                            selected_candidate,
                            policy,
                            true,
                        )
                    })))
        {
            selected.push(candidate);
        }
    }
    selected
}

fn recommended_onion_hops(
    candidate_count: usize,
    requested_hops: u8,
    requested_path_ready: bool,
    two_hop_fallback_ready: bool,
) -> u8 {
    let candidate_recommendation =
        (candidate_count.min(requested_hops as usize) as u8).min(ONION_CANDIDATES_MAX_CLIENT_HOPS);
    if requested_path_ready {
        candidate_recommendation
    } else if requested_hops > 2 && two_hop_fallback_ready {
        2
    } else {
        candidate_recommendation.min(1)
    }
}

fn onion_candidate_selection_weight(rank: usize) -> u16 {
    1_000u16
        .saturating_sub((rank as u16).saturating_mul(100))
        .max(100)
}

fn onion_candidate_pool_status(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
    gates: OnionRequestedPathGates,
) -> &'static str {
    if !gates.purpose_supported {
        "unsupported_purpose"
    } else if !gates.terminal_capability_ready {
        "terminal_limited"
    } else if limit < required_candidates {
        "client_limited"
    } else if gates.pinned_route_domain_required && !gates.pinned_route_domain_ready {
        "routing_domain_limited"
    } else if !gates.candidate_pool_ready {
        if candidate_count == 0 {
            "empty"
        } else {
            "warming"
        }
    } else if gates.network_diversity_required && !gates.network_diversity_ready {
        "diversity_limited"
    } else if gates.runtime_proof_required && !gates.runtime_proof_ready {
        "proof_warming"
    } else if gates.restart_continuity_required && !gates.restart_continuity_ready {
        "continuity_warming"
    } else {
        "ready"
    }
}

fn onion_candidate_route_plan(
    requested_path_ready: bool,
    requested_hops: u8,
    recommended_hops: u8,
    fallback_reason: &str,
) -> &'static str {
    if fallback_reason == "unsupported_route_purpose" {
        "reject_unsupported_purpose"
    } else if fallback_reason == "requested_terminal_capability_not_ready" {
        "defer_specialized_delivery"
    } else if !requested_path_ready && requested_hops > 2 && recommended_hops == 2 {
        "two_hop_onion_path"
    } else if !requested_path_ready {
        "standard_relay_fallback"
    } else if recommended_hops >= 3 {
        "three_hop_onion_path"
    } else if recommended_hops == 2 {
        "two_hop_onion_path"
    } else if recommended_hops == 1 {
        "single_hop_encrypted_relay"
    } else {
        "standard_relay_fallback"
    }
}

fn onion_candidate_fallback_reason(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
    gates: OnionRequestedPathGates,
) -> &'static str {
    if !gates.purpose_supported {
        "unsupported_route_purpose"
    } else if !gates.terminal_capability_ready {
        "requested_terminal_capability_not_ready"
    } else if limit < required_candidates {
        if required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES {
            "client_limit_below_two_hop_minimum"
        } else {
            "client_limit_below_requested_hops"
        }
    } else if gates.pinned_route_domain_required && !gates.pinned_route_domain_ready {
        "requested_path_pinned_route_domain_not_ready"
    } else if !gates.candidate_pool_ready && candidate_count == 0 {
        "no_routeable_candidates"
    } else if !gates.candidate_pool_ready
        && candidate_count == 1
        && required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
    {
        "single_routeable_candidate"
    } else if !gates.candidate_pool_ready {
        "insufficient_routeable_candidates"
    } else if gates.network_diversity_required && !gates.network_diversity_ready {
        "requested_path_network_diversity_not_ready"
    } else if gates.runtime_proof_required && !gates.runtime_proof_ready {
        "requested_path_runtime_proof_not_ready"
    } else if gates.restart_continuity_required && !gates.restart_continuity_ready {
        "requested_path_restart_continuity_not_ready"
    } else {
        "ready"
    }
}

fn onion_candidate_readiness_reason(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
    gates: OnionRequestedPathGates,
) -> &'static str {
    match onion_candidate_fallback_reason(candidate_count, limit, required_candidates, gates) {
        "ready" => {
            if required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES {
                "two_hop_candidate_pool_ready"
            } else {
                "requested_onion_candidate_pool_ready"
            }
        }
        "unsupported_route_purpose" => "requested_route_purpose_is_not_supported",
        "requested_terminal_capability_not_ready" => {
            "waiting_for_routeable_signed_terminal_capability"
        }
        "client_limit_below_two_hop_minimum" => "client_limit_blocks_two_hop_pool",
        "client_limit_below_requested_hops" => "client_limit_blocks_requested_hops",
        "no_routeable_candidates" => "waiting_for_routeable_kem_relays",
        "single_routeable_candidate" => "waiting_for_second_routeable_kem_relay",
        "requested_path_pinned_route_domain_not_ready" => {
            "waiting_for_operator_audited_route_domain_coverage"
        }
        "requested_path_network_diversity_not_ready" => "waiting_for_network_diverse_onion_relays",
        "requested_path_runtime_proof_not_ready" => {
            "waiting_for_stable_requested_path_runtime_proof"
        }
        "requested_path_restart_continuity_not_ready" => {
            "waiting_for_requested_path_restart_continuity"
        }
        _ => "waiting_for_more_routeable_kem_relays",
    }
}

fn onion_candidate_next_action(
    requested_path_ready: bool,
    recommended_hops: u8,
    fallback_reason: &str,
) -> &'static str {
    if fallback_reason == "unsupported_route_purpose" {
        "reject the unsupported route purpose without sending the payload"
    } else if fallback_reason == "requested_terminal_capability_not_ready" {
        "keep the ciphertext queued locally and refresh until an admitted terminal is routeable"
    } else if requested_path_ready {
        "build a weighted-random onion path with fresh distinct candidates"
    } else if recommended_hops == 2
        && fallback_reason == "requested_path_pinned_route_domain_not_ready"
    {
        "use the audited two-hop fallback until a distinct pinned third route domain is available"
    } else if fallback_reason == "requested_path_pinned_route_domain_not_ready" {
        "use standard encrypted relay fallback until pinned route-domain coverage is complete"
    } else if recommended_hops == 2
        && fallback_reason == "requested_path_network_diversity_not_ready"
    {
        "use the network-diverse two-hop fallback until a diverse third hop is available"
    } else if recommended_hops == 2 {
        "use the mature two-hop onion fallback while requested path evidence warms"
    } else if fallback_reason == "client_limit_below_requested_hops"
        || fallback_reason == "client_limit_below_two_hop_minimum"
    {
        "increase candidate limit or use standard encrypted relay fallback"
    } else {
        "use standard encrypted relay fallback and refresh candidate pool later"
    }
}

/// Imports one portable route-domain certificate under exact host-local pins.
///
/// [ROUTE-DOMAIN-CERTIFICATE-INGRESS 2026-08-03 by Codex] The transport sender
/// is intentionally not authority: any peer may carry the bounded frame, while
/// only signatures from locally pinned independent attestors count. Responses
/// and audit events remain identity-blind.
async fn route_domain_certificate_handler(
    State(state): State<DiscoveryApiState>,
    body: Bytes,
) -> impl IntoResponse {
    let now = now_secs();
    let rate_limit = state
        .policy
        .route_domain_certificate_rate_limit_per_minute();
    if !state
        .route_domain_certificate_rate_limit
        .lock()
        .allow(now, rate_limit)
    {
        state.peer_store.record_audit_event(
            now,
            "route_domain_certificate_import",
            "rate_limited",
            "reason=bounded_ingress_limit",
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(RouteDomainCertificateImportResponse {
                accepted: false,
                stored: false,
                status: "rate_limited",
            }),
        )
            .into_response();
    }

    let certificate = match decode_route_domain_attestation_certificate(&body) {
        Ok(certificate) => certificate,
        Err(_) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=malformed_frame",
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "malformed_certificate",
                }),
            )
                .into_response();
        }
    };

    match state
        .peer_store
        .import_route_domain_attestation_certificate(certificate, now)
    {
        Ok(stored) => {
            let status = if stored { "stored" } else { "already_present" };
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "accepted",
                format!("result={status}"),
            );
            (
                StatusCode::OK,
                Json(RouteDomainCertificateImportResponse {
                    accepted: true,
                    stored,
                    status,
                }),
            )
                .into_response()
        }
        Err(RouteDomainCertificateImportError::Rejected) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=local_policy_verification_failed",
            );
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "certificate_rejected",
                }),
            )
                .into_response()
        }
        Err(RouteDomainCertificateImportError::Stale) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=stale_evidence",
            );
            (
                StatusCode::CONFLICT,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "stale_certificate",
                }),
            )
                .into_response()
        }
        Err(RouteDomainCertificateImportError::CapacityExceeded) => {
            state.peer_store.record_audit_event(
                now,
                "route_domain_certificate_import",
                "rejected",
                "reason=bounded_cache_capacity",
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(RouteDomainCertificateImportResponse {
                    accepted: false,
                    stored: false,
                    status: "certificate_capacity_reached",
                }),
            )
                .into_response()
        }
    }
}

async fn gossip_handler(
    State(state): State<DiscoveryApiState>,
    Json(message): Json<NodeDiscoveryMessage>,
) -> impl IntoResponse {
    let now = now_secs();
    if !state
        .rate_limit
        .lock()
        .allow(now, state.policy.gossip_rate_limit_per_minute)
    {
        state.peer_store.record_rate_limited(
            now,
            format!(
                "global_limit_per_minute={}",
                state.policy.gossip_rate_limit_per_minute
            ),
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(GossipResponse {
                applied: PeerStoreImportReport::empty(),
                response: None,
            }),
        )
            .into_response();
    }

    if !state.policy.message_allowed(&message) {
        state.peer_store.record_policy_rejected(
            now,
            format!(
                "allow_list_enabled={} allowed_peer_count={} denied_peer_count={}",
                !state.policy.allowed_peer_ids.is_empty(),
                state.policy.allowed_peer_ids.len(),
                state.policy.denied_peer_ids.len()
            ),
        );
        return (
            StatusCode::FORBIDDEN,
            Json(GossipResponse {
                applied: PeerStoreImportReport::empty(),
                response: None,
            }),
        )
            .into_response();
    }

    let (admission_status, applied) = apply_gossip_message(&state, &message, now);
    state.peer_store.mark_gossip_at(now);
    if admission_status != StatusCode::OK {
        return (
            admission_status,
            Json(GossipResponse {
                applied,
                response: None,
            }),
        )
            .into_response();
    }
    let response = match message {
        NodeDiscoveryMessage::SnapshotRequest { limit, .. } => {
            Some(state.peer_store.build_snapshot_response(
                now,
                now,
                true,
                Some(state.policy.snapshot_limit(limit.map(usize::from))),
            ))
        }
        NodeDiscoveryMessage::SnapshotResponse { .. }
        | NodeDiscoveryMessage::DescriptorAnnounce { .. }
        | NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => None,
    };

    (StatusCode::OK, Json(GossipResponse { applied, response })).into_response()
}

fn apply_gossip_message(
    state: &DiscoveryApiState,
    message: &NodeDiscoveryMessage,
    now: u64,
) -> (StatusCode, PeerStoreImportReport) {
    let NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
        producer,
        block_hash,
        descriptor_hash,
        proof,
    } = message
    else {
        return (
            StatusCode::OK,
            state.peer_store.apply_discovery_message(message, now),
        );
    };

    // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] A node without an
    // audited replica cannot authenticate this stronger gossip contract. Do
    // not silently downgrade it to signature-only DescriptorAnnounce.
    let Some(replica_store) = state.directory_replica_store.as_deref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            state.peer_store.record_rejected_directory_proof_import(now),
        );
    };
    match admit_directory_gossip_descriptor(
        replica_store,
        &state.peer_store,
        proof,
        producer,
        block_hash,
        descriptor_hash,
        now,
    ) {
        Ok(report) => (StatusCode::OK, report),
        Err(_) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            state.peer_store.record_rejected_directory_proof_import(now),
        ),
    }
}

async fn status_handler(State(state): State<DiscoveryApiState>) -> Json<DiscoveryStatusResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    let discovery_readiness = discovery_readiness_status_value(&peer_store, &local_capabilities);
    let blind_relay_runtime =
        blind_relay_runtime_status_value(now, &peer_store, &local_capabilities);
    let recovery_anchor = recovery_anchor_status_value(&peer_store);
    Json(DiscoveryStatusResponse {
        generated_at: now,
        peer_store,
        policy: DiscoveryPolicyStatus {
            max_snapshot_limit: state.policy.max_snapshot_limit,
            gossip_rate_limit_per_minute: state.policy.gossip_rate_limit_per_minute,
            allow_list_enabled: !state.policy.allowed_peer_ids.is_empty(),
            allowed_peer_count: state.policy.allowed_peer_ids.len(),
            denied_peer_count: state.policy.denied_peer_ids.len(),
            pinned_route_domain_count: state.policy.pinned_route_domains.len(),
            require_pinned_route_domains_for_multi_hop: state
                .policy
                .require_pinned_route_domains_for_multi_hop,
            snapshot_default_public_only: true,
            private_descriptors_hidden_by_default: true,
        },
        local_capabilities,
        discovery_readiness,
        blind_relay_runtime,
        recovery_anchor,
    })
}

async fn summary_handler(State(state): State<DiscoveryApiState>) -> Json<DiscoverySummaryResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    Json(discovery_summary_response(
        now,
        &peer_store,
        &local_capabilities,
    ))
}

async fn public_card_handler(
    State(state): State<DiscoveryApiState>,
) -> Json<DiscoveryPublicCardResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    Json(discovery_public_card_response(
        now,
        &peer_store,
        &local_capabilities,
    ))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        directory_block_range_response_signing_bytes, encode_directory_sync_message,
        encode_route_domain_attestation_certificate, DirectoryCommitmentBlockV1,
        DirectoryDescriptorCommitmentV1, DirectoryDescriptorInclusionProofV1, DirectorySyncMessage,
        RouteDomainAttestationCertificateV1, RouteDomainAttestationV1,
        AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    };
    use aeronyx_core::protocol::{
        NodeCapability, NodeCapacity, NodeDescriptor, NodePolicy, SignedNodeDescriptor,
    };
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt;

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

    #[tokio::test]
    async fn test_snapshot_endpoint_returns_snapshot() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/snapshot?limit=10")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_onion_candidates_endpoint_exposes_routeable_kem_relays() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();

        // (a) Routeable ChatRelay + OnionMiddle advertising a KEM key and
        // endpoint -> included.
        let kp = IdentityKeyPair::generate();
        let kem = kp.x25519_public_key_bytes();
        let mut included = NodeDescriptor::new(
            kp.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        included.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
        included.public_endpoint = Some("relay.example:443".to_string());
        let included = included.with_x25519_kem(kem);
        let included = aeronyx_core::protocol::SignedNodeDescriptor::sign(included, &kp).unwrap();
        let want_node_id = hex::encode(included.node_id());
        let included_node_id = included.node_id();
        store.upsert_verified(included, now).unwrap();
        store.record_route_forward_success(&included_node_id, now);

        // (b) ChatRelay + OnionMiddle WITHOUT a KEM key -> filtered out.
        let kp2 = IdentityKeyPair::generate();
        let mut no_kem = NodeDescriptor::new(
            kp2.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        no_kem.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
        no_kem.public_endpoint = Some("nokem.example:443".to_string());
        let no_kem = aeronyx_core::protocol::SignedNodeDescriptor::sign(no_kem, &kp2).unwrap();
        store.upsert_verified(no_kem, now).unwrap();

        // (c) KEM-bearing ChatRelay + OnionMiddle without routeability
        // evidence -> filtered out. This keeps clients from building paths
        // through unknown peers while allowing probes to keep learning.
        let kp3 = IdentityKeyPair::generate();
        let mut unknown = NodeDescriptor::new(
            kp3.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        unknown.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
        unknown.public_endpoint = Some("unknown.example:443".to_string());
        let unknown = unknown.with_x25519_kem(kp3.x25519_public_key_bytes());
        let unknown = aeronyx_core::protocol::SignedNodeDescriptor::sign(unknown, &kp3).unwrap();
        store.upsert_verified(unknown, now).unwrap();

        // (d) Routeable KEM-bearing ChatRelay without OnionMiddle -> filtered
        // out. It can serve a standard encrypted relay path, but must never be
        // counted as a blind multi-hop onion relay.
        let kp4 = IdentityKeyPair::generate();
        let mut single_hop_only = NodeDescriptor::new(
            kp4.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        single_hop_only.capabilities = vec![NodeCapability::ChatRelay];
        single_hop_only.public_endpoint = Some("single-hop.example:443".to_string());
        // [ONION-CAPABILITY-GATE 2026-08-02 by Codex] Give the ineligible
        // relay a higher route-capacity score. With `limit=1`, this proves the
        // limit is applied after capability filtering rather than before it.
        single_hop_only.capacity = NodeCapacity {
            max_sessions: 10_000,
            max_bps: Some(10_000_000_000),
            max_pps: Some(1_000_000),
        };
        let single_hop_only = single_hop_only.with_x25519_kem(kp4.x25519_public_key_bytes());
        let single_hop_only =
            aeronyx_core::protocol::SignedNodeDescriptor::sign(single_hop_only, &kp4).unwrap();
        let single_hop_node_id = single_hop_only.node_id();
        store.upsert_verified(single_hop_only, now).unwrap();
        store.record_route_forward_success(&single_hop_node_id, now);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?limit=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        // Only the routeable KEM-bearing relay is exposed, with its KEM key for the client.
        assert_eq!(parsed.contract_version, ONION_CANDIDATES_CONTRACT_VERSION);
        assert_eq!(parsed.source, ONION_CANDIDATES_SOURCE);
        assert_eq!(
            parsed.required_capabilities,
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle]
        );
        assert_eq!(parsed.requested_purpose, "message_relay");
        assert!(parsed.requested_purpose_supported);
        assert_eq!(
            parsed.terminal_required_capabilities,
            vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle]
        );
        assert_eq!(parsed.terminal_candidate_count, 1);
        assert!(parsed.requested_terminal_capability_ready);
        assert_eq!(parsed.selection_policy, ONION_CANDIDATES_SELECTION_POLICY);
        assert_eq!(
            parsed.candidate_verification,
            "signed_node_descriptor_ed25519_v2"
        );
        assert_eq!(
            parsed.refresh_after_seconds,
            ONION_CANDIDATES_REFRESH_AFTER_SECONDS
        );
        assert_eq!(
            parsed.routeability_stale_after_seconds,
            ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS
        );
        assert_eq!(parsed.count, 1);
        assert_eq!(
            parsed.min_candidates_for_two_hop,
            ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        );
        assert_eq!(parsed.requested_privacy_mode, "enhanced");
        assert_eq!(parsed.requested_hops, 2);
        assert_eq!(parsed.min_candidates_for_requested_hops, 2);
        assert!(!parsed.requested_path_ready);
        assert!(!parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_runtime_proof_required);
        assert!(!parsed.requested_runtime_proof_ready);
        assert!(parsed.requested_restart_continuity_required);
        assert!(!parsed.requested_restart_continuity_ready);
        assert_eq!(parsed.recommended_hops, 1);
        assert!(!parsed.two_hop_ready);
        assert!(parsed.fallback_required);
        assert_eq!(parsed.pool_status, "client_limited");
        assert_eq!(parsed.route_plan, "standard_relay_fallback");
        assert_eq!(parsed.fallback_reason, "client_limit_below_two_hop_minimum");
        assert_eq!(parsed.readiness_reason, "client_limit_blocks_two_hop_pool");
        assert_eq!(
            parsed.next_action,
            "increase candidate limit or use standard encrypted relay fallback"
        );
        assert_eq!(
            parsed.path_selection_strategy,
            "weighted_random_health_ranked_distinct_hops"
        );
        assert_eq!(
            parsed.region_diversity_policy,
            "prefer_distinct_regions_when_available_without_exposing_selected_route"
        );
        assert!(parsed.user_choice_policy.contains("privacy_mode"));
        assert_eq!(parsed.candidates.len(), 1);
        let candidate = &parsed.candidates[0];
        assert_eq!(candidate.node_id, want_node_id);
        assert_eq!(candidate.kem_alg, 1);
        assert_eq!(candidate.kem_public, hex::encode(kem));
        assert_eq!(candidate.public_endpoint, "relay.example:443");
        assert!(candidate.capabilities.contains(&NodeCapability::ChatRelay));
        assert!(candidate
            .capabilities
            .contains(&NodeCapability::OnionMiddle));
        assert_eq!(candidate.selection_weight, 1_000);
        assert_eq!(candidate.region, None);
        assert_eq!(candidate.max_sessions, 0);
        assert!(candidate
            .signed_descriptor
            .verify_at(parsed.generated_at)
            .is_ok());
        assert_eq!(
            candidate.node_id,
            hex::encode(candidate.signed_descriptor.node_id())
        );
        assert_eq!(
            candidate.kem_alg,
            candidate.signed_descriptor.descriptor.kem_alg
        );
        assert_eq!(
            candidate.kem_public,
            hex::encode(
                candidate
                    .signed_descriptor
                    .descriptor
                    .x25519_kem_public()
                    .expect("candidate proof must carry its projected X25519 KEM key")
            )
        );
        assert_eq!(
            Some(candidate.public_endpoint.as_str()),
            candidate
                .signed_descriptor
                .descriptor
                .public_endpoint
                .as_deref()
        );
        assert_eq!(
            candidate.capabilities,
            candidate.signed_descriptor.descriptor.capabilities
        );
        assert_eq!(
            candidate.max_sessions,
            candidate.signed_descriptor.descriptor.capacity.max_sessions
        );
        assert_eq!(
            candidate.max_bps,
            candidate.signed_descriptor.descriptor.capacity.max_bps
        );
        assert_eq!(
            candidate.max_pps,
            candidate.signed_descriptor.descriptor.capacity.max_pps
        );
        assert_eq!(
            candidate.region,
            candidate.signed_descriptor.descriptor.policy.region
        );
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert!(encoded["candidates"][0]["signed_descriptor"].is_object());
        assert_eq!(
            encoded["candidate_verification"],
            "signed_node_descriptor_ed25519_v2"
        );
        assert!(parsed.privacy_boundary.contains("fresh routeable"));
        assert!(parsed.privacy_boundary.contains("descriptor proof"));
    }

    #[tokio::test]
    async fn test_onion_candidates_endpoint_marks_two_hop_ready_when_pool_is_sufficient() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
        let second_node_id = second.node_id();

        store.upsert_verified(first, now).unwrap();
        store.upsert_verified(second, now).unwrap();
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);
        record_stable_path_proof(store.as_ref(), now, 2);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed.count, 2);
        assert_eq!(
            parsed.min_candidates_for_two_hop,
            ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        );
        assert_eq!(parsed.requested_privacy_mode, "enhanced");
        assert_eq!(parsed.requested_hops, 2);
        assert_eq!(parsed.min_candidates_for_requested_hops, 2);
        assert!(parsed.requested_path_ready);
        assert!(parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_runtime_proof_required);
        assert!(parsed.requested_runtime_proof_ready);
        assert!(parsed.requested_restart_continuity_required);
        assert!(parsed.requested_restart_continuity_ready);
        assert_eq!(parsed.recommended_hops, 2);
        assert!(parsed.two_hop_ready);
        assert!(!parsed.fallback_required);
        assert_eq!(parsed.pool_status, "ready");
        assert_eq!(parsed.route_plan, "two_hop_onion_path");
        assert_eq!(parsed.fallback_reason, "ready");
        assert_eq!(parsed.readiness_reason, "two_hop_candidate_pool_ready");
        assert_eq!(
            parsed.next_action,
            "build a weighted-random onion path with fresh distinct candidates"
        );
        assert_eq!(parsed.candidates[0].selection_weight, 1_000);
        assert_eq!(parsed.candidates[1].selection_weight, 900);
    }

    #[tokio::test]
    async fn test_onion_candidates_endpoint_marks_client_limit_fallback() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
        let second_node_id = second.node_id();

        store.upsert_verified(first, now).unwrap();
        store.upsert_verified(second, now).unwrap();
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?limit=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.requested_privacy_mode, "enhanced");
        assert_eq!(parsed.requested_hops, 2);
        assert_eq!(parsed.min_candidates_for_requested_hops, 2);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 1);
        assert!(!parsed.two_hop_ready);
        assert!(parsed.fallback_required);
        assert_eq!(parsed.pool_status, "client_limited");
        assert_eq!(parsed.route_plan, "standard_relay_fallback");
        assert_eq!(parsed.fallback_reason, "client_limit_below_two_hop_minimum");
        assert_eq!(parsed.readiness_reason, "client_limit_blocks_two_hop_pool");
        assert_eq!(
            parsed.next_action,
            "increase candidate limit or use standard encrypted relay fallback"
        );
    }

    #[tokio::test]
    async fn test_two_hop_candidates_wait_for_signed_restart_continuity() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
        let second_node_id = second.node_id();

        store.upsert_verified(first, now).unwrap();
        store.upsert_verified(second, now).unwrap();
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);
        record_stable_runtime_path_proof(store.as_ref(), now, 2);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert!(parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_runtime_proof_ready);
        assert!(!parsed.requested_restart_continuity_ready);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 1);
        assert!(parsed.fallback_required);
        assert_eq!(parsed.pool_status, "continuity_warming");
        assert_eq!(parsed.route_plan, "standard_relay_fallback");
        assert_eq!(
            parsed.fallback_reason,
            "requested_path_restart_continuity_not_ready"
        );
        assert_eq!(
            parsed.readiness_reason,
            "waiting_for_requested_path_restart_continuity"
        );
    }

    #[tokio::test]
    async fn test_onion_candidates_endpoint_supports_high_privacy_three_hop_policy() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
        let second_node_id = second.node_id();
        let third = signed_routeable_chat_descriptor(1, now + 300, "https://relay-three.example");
        let third_node_id = third.node_id();

        store.upsert_verified(first, now).unwrap();
        store.upsert_verified(second, now).unwrap();
        store.upsert_verified(third, now).unwrap();
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);
        store.record_route_forward_success(&third_node_id, now);
        record_stable_path_proof(store.as_ref(), now, 3);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed.requested_privacy_mode, "high");
        assert_eq!(parsed.requested_hops, 3);
        assert_eq!(parsed.min_candidates_for_requested_hops, 3);
        assert!(parsed.requested_path_ready);
        assert!(parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_runtime_proof_required);
        assert!(parsed.requested_runtime_proof_ready);
        assert!(parsed.requested_restart_continuity_required);
        assert!(parsed.requested_restart_continuity_ready);
        assert_eq!(parsed.recommended_hops, 3);
        assert!(parsed.two_hop_ready);
        assert!(!parsed.fallback_required);
        assert_eq!(parsed.pool_status, "ready");
        assert_eq!(parsed.route_plan, "three_hop_onion_path");
        assert_eq!(parsed.fallback_reason, "ready");
        assert_eq!(
            parsed.readiness_reason,
            "requested_onion_candidate_pool_ready"
        );
        assert_eq!(
            parsed.next_action,
            "build a weighted-random onion path with fresh distinct candidates"
        );
        assert_eq!(parsed.candidates.len(), 3);
        assert_eq!(parsed.candidates[0].selection_weight, 1_000);
        assert_eq!(parsed.candidates[1].selection_weight, 900);
        assert_eq!(parsed.candidates[2].selection_weight, 800);
    }

    #[tokio::test]
    async fn test_high_privacy_candidates_require_pairwise_network_diversity() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422");
        let second_node_id = second.node_id();
        let collocated_third =
            signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.20:8422");
        let third_node_id = collocated_third.node_id();

        store.upsert_verified(first, now).unwrap();
        store.upsert_verified(second, now).unwrap();
        store.upsert_verified(collocated_third, now).unwrap();
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);
        store.record_route_forward_success(&third_node_id, now);
        record_stable_path_proof(store.as_ref(), now, 2);
        record_stable_path_proof(store.as_ref(), now, 3);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        // [ONION-NETWORK-DIVERSITY 2026-08-03 by Codex] Candidate count and
        // mature proof are insufficient when two of three hops share an IPv4
        // /24. The already-diverse two-hop subset remains a safe fallback.
        assert_eq!(parsed.count, 3);
        assert!(parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_network_diversity_required);
        assert!(!parsed.requested_network_diversity_ready);
        assert!(parsed.requested_runtime_proof_ready);
        assert!(parsed.requested_restart_continuity_ready);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 2);
        assert!(parsed.fallback_required);
        assert_eq!(parsed.pool_status, "diversity_limited");
        assert_eq!(parsed.route_plan, "two_hop_onion_path");
        assert_eq!(
            parsed.fallback_reason,
            "requested_path_network_diversity_not_ready"
        );
        assert_eq!(
            parsed.readiness_reason,
            "waiting_for_network_diverse_onion_relays"
        );
        assert_eq!(
            parsed.next_action,
            "use the network-diverse two-hop fallback until a diverse third hop is available"
        );
        assert!(parsed.network_diversity_policy.contains("ipv4_24"));
        assert!(parsed
            .network_diversity_policy
            .contains("not_operator_or_as_proof"));
    }

    #[test]
    fn test_bounded_onion_pool_preserves_lower_rank_network_diverse_candidate() {
        let candidates = vec![
            onion_candidate_for_test("https://192.0.2.10:8422", 0),
            onion_candidate_for_test("https://192.0.2.20:8422", 1),
            onion_candidate_for_test("https://198.51.100.10:8422", 2),
            onion_candidate_for_test("https://203.0.113.10:8422", 3),
        ];

        let selected = select_onion_candidate_response_pool(candidates, 3, 3);

        // [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] The first three
        // ranked entries cannot form a diverse path because two share an IPv4
        // /24. The bounded pool must retain the fourth candidate, preserve its
        // original health weight, and exclude one collocated relay.
        assert_eq!(selected.len(), 3);
        assert!(onion_candidate_network_diversity_ready(&selected, 3));
        assert!(selected
            .iter()
            .any(|candidate| candidate.public_endpoint == "https://203.0.113.10:8422"));
        assert!(selected
            .iter()
            .any(|candidate| candidate.selection_weight == 700));
        assert_eq!(
            selected
                .iter()
                .filter(|candidate| candidate.public_endpoint.starts_with("https://192.0.2."))
                .count(),
            1
        );
    }

    #[test]
    fn test_bounded_onion_pool_preserves_signed_specialized_terminal() {
        let candidates = vec![
            onion_candidate_for_test("https://192.0.2.10:8422", 0),
            onion_candidate_for_test("https://198.51.100.10:8422", 1),
            onion_candidate_for_test_with_capabilities(
                "https://203.0.113.10:8422",
                2,
                &[NodeCapability::BlindVaultReplica],
            ),
        ];

        let selected = select_onion_candidate_response_pool_with_policy_and_terminal(
            candidates,
            2,
            2,
            &DiscoveryApiPolicy::default(),
            false,
            OnionTerminalRequirement {
                capability: Some(NodeCapability::BlindVaultReplica),
                protocol_features: &[],
            },
        );

        // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] A healthier generic pair
        // must not hide the only signed storage terminal under a small limit.
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().any(|candidate| {
            candidate.public_endpoint == "https://203.0.113.10:8422"
                && candidate
                    .signed_descriptor
                    .descriptor
                    .capabilities
                    .contains(&NodeCapability::BlindVaultReplica)
        }));
        assert!(onion_candidate_route_diversity_ready_for_terminal(
            &selected,
            2,
            &DiscoveryApiPolicy::default(),
            false,
            Some(NodeCapability::BlindVaultReplica),
        ));
    }

    #[tokio::test]
    async fn test_onion_candidates_route_purpose_fails_closed_without_terminal() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let descriptor =
            signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422");
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?purpose=blind_vault_put&hops=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.requested_purpose, "blind_vault_put");
        assert!(parsed.requested_purpose_supported);
        assert!(parsed
            .terminal_required_capabilities
            .contains(&NodeCapability::BlindVaultReplica));
        assert_eq!(parsed.terminal_candidate_count, 0);
        assert!(!parsed.requested_terminal_capability_ready);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 0);
        assert_eq!(parsed.pool_status, "terminal_limited");
        assert_eq!(parsed.route_plan, "defer_specialized_delivery");
        assert_eq!(
            parsed.fallback_reason,
            "requested_terminal_capability_not_ready"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?purpose=unknown_storage_mode")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.requested_purpose, "unsupported");
        assert!(!parsed.requested_purpose_supported);
        assert!(parsed.terminal_required_capabilities.is_empty());
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 0);
        assert_eq!(parsed.pool_status, "unsupported_purpose");
        assert_eq!(parsed.route_plan, "reject_unsupported_purpose");
        assert_eq!(parsed.fallback_reason, "unsupported_route_purpose");

        let empty_app =
            build_discovery_router(Arc::new(PeerStore::new()), DiscoveryApiPolicy::default());
        let response = empty_app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();
        // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] The additive purpose gate
        // must preserve the legacy message-relay diagnosis for an empty pool.
        assert_eq!(parsed.requested_purpose, "message_relay");
        assert!(!parsed.requested_terminal_capability_ready);
        assert_eq!(parsed.pool_status, "empty");
        assert_eq!(parsed.fallback_reason, "no_routeable_candidates");
    }

    #[tokio::test]
    async fn test_production_onion_pool_excludes_candidate_collocated_with_entry() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let local = signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.5:8422");
        let local_node_id = local.node_id();
        store.upsert_verified(local, now).unwrap();

        let remotes = [
            signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://203.0.113.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://198.18.0.10:8422"),
        ];
        for remote in remotes {
            let node_id = remote.node_id();
            store.upsert_verified(remote, now).unwrap();
            store.record_route_forward_success(&node_id, now);
        }
        record_stable_path_proof(store.as_ref(), now, 3);

        let app = build_discovery_router_with_local_entry(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::default(),
            None,
            local_node_id,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        // [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] The collocated
        // remote is valid and routeable, but a production pool must remove it
        // before count, diversity, and requested-path readiness are computed.
        assert_eq!(parsed.count, 3);
        assert!(parsed.local_entry_network_diversity_enforced);
        assert!(parsed.requested_network_diversity_ready);
        assert!(parsed.requested_path_ready);
        assert!(parsed
            .network_diversity_policy
            .contains("against_local_entry"));
        assert!(parsed
            .candidates
            .iter()
            .all(|candidate| !candidate.public_endpoint.starts_with("https://192.0.2.")));
    }

    #[tokio::test]
    async fn test_strict_pinned_route_domains_preserve_distinct_high_privacy_pool() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let local = signed_routeable_chat_descriptor(1, now + 300, "https://10.0.0.5:8422");
        let local_node_id = local.node_id();
        store.upsert_verified(local, now).unwrap();

        let remotes = [
            signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://203.0.113.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://198.18.0.10:8422"),
        ];
        let remote_node_ids = remotes
            .iter()
            .map(SignedNodeDescriptor::node_id)
            .collect::<Vec<_>>();
        for remote in remotes {
            let node_id = remote.node_id();
            store.upsert_verified(remote, now).unwrap();
            store.record_route_forward_success(&node_id, now);
        }
        record_stable_path_proof(store.as_ref(), now, 3);

        let mut config = DiscoveryConfig::default();
        config.require_pinned_route_domains_for_multi_hop = true;
        config.pinned_route_domains.insert(
            hex::encode(local_node_id),
            "11111111111111111111111111111111".to_string(),
        );
        for (node_id, domain) in remote_node_ids.iter().copied().zip([
            "22222222222222222222222222222222",
            "22222222222222222222222222222222",
            "33333333333333333333333333333333",
            "44444444444444444444444444444444",
        ]) {
            config
                .pinned_route_domains
                .insert(hex::encode(node_id), domain.to_string());
        }
        let policy = DiscoveryApiPolicy::from_config(&config);

        let app = build_discovery_router_with_local_entry(
            store,
            policy,
            DiscoveryLocalCapabilityStatus::default(),
            None,
            local_node_id,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=4")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        // [PINNED-ROUTE-DOMAINS 2026-08-03 by Codex] Four routeable remote
        // nodes exist, but two share one audited failure domain. Strict mode
        // emits only a pairwise-distinct three-hop pool, keeps the path ready,
        // and never discloses the opaque local assignment tokens.
        assert_eq!(parsed.count, 3);
        assert!(parsed.requested_pinned_route_domain_required);
        assert!(parsed.requested_pinned_route_domain_ready);
        assert!(parsed.local_entry_pinned_route_domain_enforced);
        assert!(parsed.requested_network_diversity_ready);
        assert!(parsed.requested_path_ready);
        assert_eq!(parsed.route_plan, "three_hop_onion_path");
        assert!(parsed
            .pinned_route_domain_policy
            .contains("operator_audited_local_opaque_assignments"));
        assert_eq!(
            parsed
                .candidates
                .iter()
                .filter(|candidate| {
                    candidate.node_id == hex::encode(remote_node_ids[0])
                        || candidate.node_id == hex::encode(remote_node_ids[1])
                })
                .count(),
            1
        );
        let encoded = String::from_utf8(body.to_vec()).unwrap();
        assert!(!encoded.contains("11111111111111111111111111111111"));
        assert!(!encoded.contains("22222222222222222222222222222222"));
    }

    #[tokio::test]
    async fn test_strict_pinned_route_domains_fail_closed_without_local_entry_assignment() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let local = signed_routeable_chat_descriptor(1, now + 300, "https://10.0.1.5:8422");
        let local_node_id = local.node_id();
        store.upsert_verified(local, now).unwrap();

        let remotes = [
            signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422"),
            signed_routeable_chat_descriptor(1, now + 300, "https://203.0.113.10:8422"),
        ];
        let mut config = DiscoveryConfig::default();
        config.require_pinned_route_domains_for_multi_hop = true;
        for (remote, domain) in remotes.into_iter().zip([
            "55555555555555555555555555555555",
            "66666666666666666666666666666666",
            "77777777777777777777777777777777",
        ]) {
            let node_id = remote.node_id();
            config
                .pinned_route_domains
                .insert(hex::encode(node_id), domain.to_string());
            store.upsert_verified(remote, now).unwrap();
            store.record_route_forward_success(&node_id, now);
        }
        record_stable_path_proof(store.as_ref(), now, 3);

        let app = build_discovery_router_with_local_entry(
            store,
            DiscoveryApiPolicy::from_config(&config),
            DiscoveryLocalCapabilityStatus::default(),
            None,
            local_node_id,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed.count, 3);
        assert!(parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_pinned_route_domain_required);
        assert!(!parsed.local_entry_pinned_route_domain_enforced);
        assert!(!parsed.requested_pinned_route_domain_ready);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 1);
        assert_eq!(parsed.pool_status, "routing_domain_limited");
        assert_eq!(
            parsed.fallback_reason,
            "requested_path_pinned_route_domain_not_ready"
        );
        assert_eq!(
            parsed.readiness_reason,
            "waiting_for_operator_audited_route_domain_coverage"
        );
        assert_eq!(parsed.route_plan, "standard_relay_fallback");
    }

    #[tokio::test]
    async fn test_production_onion_pool_fails_closed_without_entry_descriptor() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        for endpoint in [
            "https://192.0.2.10:8422",
            "https://198.51.100.10:8422",
            "https://203.0.113.10:8422",
        ] {
            let remote = signed_routeable_chat_descriptor(1, now + 300, endpoint);
            let node_id = remote.node_id();
            store.upsert_verified(remote, now).unwrap();
            store.record_route_forward_success(&node_id, now);
        }
        record_stable_path_proof(store.as_ref(), now, 3);

        let missing_local_node_id = IdentityKeyPair::generate().public_key_bytes();
        let app = build_discovery_router_with_local_entry(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::default(),
            None,
            missing_local_node_id,
        );
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed.count, 3);
        assert!(parsed.requested_candidate_pool_ready);
        assert!(!parsed.local_entry_network_diversity_enforced);
        assert!(!parsed.requested_network_diversity_ready);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.pool_status, "diversity_limited");
        assert_eq!(
            parsed.fallback_reason,
            "requested_path_network_diversity_not_ready"
        );
        assert!(parsed
            .network_diversity_policy
            .contains("local_entry_descriptor_unavailable_fail_closed"));
    }

    #[tokio::test]
    async fn test_high_privacy_candidates_fall_back_until_three_hop_proof_is_mature() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
        let second_node_id = second.node_id();
        let third = signed_routeable_chat_descriptor(1, now + 300, "https://relay-three.example");
        let third_node_id = third.node_id();

        store.upsert_verified(first, now).unwrap();
        store.upsert_verified(second, now).unwrap();
        store.upsert_verified(third, now).unwrap();
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);
        store.record_route_forward_success(&third_node_id, now);
        record_stable_path_proof(store.as_ref(), now, 2);

        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed.count, 3);
        assert_eq!(parsed.requested_hops, 3);
        assert!(parsed.requested_candidate_pool_ready);
        assert!(parsed.requested_runtime_proof_required);
        assert!(!parsed.requested_runtime_proof_ready);
        assert!(parsed.requested_restart_continuity_required);
        assert!(!parsed.requested_restart_continuity_ready);
        assert!(!parsed.requested_path_ready);
        assert_eq!(parsed.recommended_hops, 2);
        assert!(parsed.fallback_required);
        assert_eq!(parsed.pool_status, "proof_warming");
        assert_eq!(parsed.route_plan, "two_hop_onion_path");
        assert_eq!(
            parsed.fallback_reason,
            "requested_path_runtime_proof_not_ready"
        );
        assert_eq!(
            parsed.readiness_reason,
            "waiting_for_stable_requested_path_runtime_proof"
        );
        assert_eq!(
            parsed.next_action,
            "use the mature two-hop onion fallback while requested path evidence warms"
        );
    }

    #[tokio::test]
    async fn test_gossip_snapshot_request_returns_response() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());
        let body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
            requested_at: now_secs(),
            limit: Some(1),
        })
        .unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn route_domain_certificate_ingress_is_verified_and_idempotent() {
        // [ROUTE-DOMAIN-CERTIFICATE-INGRESS 2026-08-03 by Codex] The HTTP
        // sender has no authority; only the configured certificate quorum can
        // move the process-local route gate.
        let now = now_secs();
        let store = Arc::new(PeerStore::new());
        let subject = IdentityKeyPair::generate();
        let attestor_a = IdentityKeyPair::generate();
        let attestor_b = IdentityKeyPair::generate();
        let route_domain = [0x61; 16];
        store
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &[attestor_a.public_key_bytes(), attestor_b.public_key_bytes()],
                2,
                true,
            )
            .unwrap();
        let certificate = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now,
            &[&attestor_a, &attestor_b],
        );
        let body = encode_route_domain_attestation_certificate(&certificate).unwrap();
        let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

        let first = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/route-domain-certificate")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first_body = axum::body::to_bytes(first.into_body(), usize::MAX)
            .await
            .unwrap();
        let first_json: serde_json::Value = serde_json::from_slice(&first_body).unwrap();
        assert_eq!(first_json["accepted"], true);
        assert_eq!(first_json["stored"], true);
        assert_eq!(first_json["status"], "stored");

        let duplicate = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/route-domain-certificate")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(duplicate.status(), StatusCode::OK);
        let duplicate_body = axum::body::to_bytes(duplicate.into_body(), usize::MAX)
            .await
            .unwrap();
        let duplicate_json: serde_json::Value = serde_json::from_slice(&duplicate_body).unwrap();
        assert_eq!(duplicate_json["accepted"], true);
        assert_eq!(duplicate_json["stored"], false);
        assert_eq!(duplicate_json["status"], "already_present");
    }

    #[tokio::test]
    async fn route_domain_certificate_ingress_rejects_untrusted_malformed_and_oversized_frames() {
        let now = now_secs();
        let store = Arc::new(PeerStore::new());
        let subject = IdentityKeyPair::generate();
        let attestor_a = IdentityKeyPair::generate();
        let attestor_b = IdentityKeyPair::generate();
        let untrusted = IdentityKeyPair::generate();
        let route_domain = [0x62; 16];
        store
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &[attestor_a.public_key_bytes(), attestor_b.public_key_bytes()],
                2,
                true,
            )
            .unwrap();
        let untrusted_certificate = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now,
            &[&attestor_a, &untrusted],
        );
        let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

        let untrusted_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/route-domain-certificate")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(
                        encode_route_domain_attestation_certificate(&untrusted_certificate)
                            .unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            untrusted_response.status(),
            StatusCode::UNPROCESSABLE_ENTITY
        );

        let malformed_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/route-domain-certificate")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![0u8; 8]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(malformed_response.status(), StatusCode::BAD_REQUEST);

        let oversized_response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/route-domain-certificate")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(vec![
                        0u8;
                        MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES
                            + 1
                    ]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(oversized_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn route_domain_certificate_ingress_has_an_isolated_rate_budget() {
        let now = now_secs();
        let store = Arc::new(PeerStore::new());
        let subject = IdentityKeyPair::generate();
        let attestor = IdentityKeyPair::generate();
        let route_domain = [0x63; 16];
        store
            .configure_route_domain_attestor_policy(
                &[(subject.public_key_bytes(), route_domain)],
                &[attestor.public_key_bytes()],
                1,
                true,
            )
            .unwrap();
        let certificate = route_domain_certificate_for(
            subject.public_key_bytes(),
            route_domain,
            now,
            &[&attestor],
        );
        let body = encode_route_domain_attestation_certificate(&certificate).unwrap();
        let mut policy = DiscoveryApiPolicy::default();
        policy.gossip_rate_limit_per_minute = 1;
        let app = build_discovery_router(Arc::clone(&store), policy);

        for expected in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/api/discovery/route-domain-certificate")
                        .header("content-type", "application/octet-stream")
                        .body(Body::from(body.clone()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
        }

        let gossip_body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
            requested_at: now,
            limit: Some(1),
        })
        .unwrap();
        let gossip_response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(gossip_body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            gossip_response.status(),
            StatusCode::OK,
            "certificate abuse must not exhaust the gossip recovery budget"
        );
    }

    #[tokio::test]
    async fn gossip_rejects_oversized_body_before_deserialization() {
        let store = Arc::new(PeerStore::new());
        let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(vec![b' '; DISCOVERY_REQUEST_BODY_MAX_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(store.len(), 0, "oversized gossip must not reach PeerStore");
    }

    #[tokio::test]
    async fn test_gossip_descriptor_announce_imports_peer() {
        let store = Arc::new(PeerStore::new());
        let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
        let body = serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed_descriptor(),
        })
        .unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn directory_gossip_imports_only_against_local_replica_anchor() {
        let now = now_secs();
        let (replica_store, message, descriptor) = directory_gossip_fixture(now);
        let store = Arc::new(PeerStore::new());
        let app = build_discovery_router_with_local_status_and_directory_admission(
            Arc::clone(&store),
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::default(),
            Some(replica_store),
        );
        let body = serde_json::to_vec(&message).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
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
        let parsed: GossipResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed.applied.inserted, 1);
        assert_eq!(
            store.get_valid(&descriptor.node_id(), now),
            Some(descriptor)
        );
    }

    #[tokio::test]
    async fn directory_gossip_fails_closed_without_local_replica_or_anchor() {
        let now = now_secs();
        let (_, message, _) = directory_gossip_fixture(now);
        let store_without_replica = Arc::new(PeerStore::new());
        let app_without_replica = build_discovery_router(
            Arc::clone(&store_without_replica),
            DiscoveryApiPolicy::default(),
        );
        let body = serde_json::to_vec(&message).unwrap();
        let response = app_without_replica
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(store_without_replica.len(), 0);

        let empty_local = IdentityKeyPair::from_bytes(&[0x95; 32]).unwrap();
        let (empty_replica, _) =
            DirectoryReplicaStore::open(":memory:", empty_local.public_key_bytes(), now).unwrap();
        let store_without_anchor = Arc::new(PeerStore::new());
        let app_without_anchor = build_discovery_router_with_local_status_and_directory_admission(
            Arc::clone(&store_without_anchor),
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::default(),
            Some(Arc::new(empty_replica)),
        );
        let response = app_without_anchor
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&message).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(store_without_anchor.len(), 0);
        assert_eq!(
            store_without_anchor.status(now).runtime.rejected,
            1,
            "proof rejection must remain visible as an aggregate only"
        );
    }

    #[tokio::test]
    async fn test_status_endpoint_returns_peer_store_status() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let app = build_discovery_router(store, DiscoveryApiPolicy::default());

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status_endpoint_returns_local_capability_status() {
        let store = Arc::new(PeerStore::new());
        let app = build_discovery_router_with_local_status(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::new(true, true, true, true),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            parsed["local_capabilities"]["status"].as_str(),
            Some("ready")
        );
        assert_eq!(
            parsed["local_capabilities"]["chat_relay_configured"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["blind_relay_endpoint_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["chat_relay_runtime_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["safe_to_advertise_chat_relay"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["advertised_chat_relay_capability"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["local_capabilities"]["capability_config_consistent"].as_bool(),
            Some(true)
        );
    }

    #[tokio::test]
    async fn test_status_endpoint_returns_compact_discovery_readiness_without_private_metadata() {
        let store = Arc::new(PeerStore::new());
        store.record_blind_relay_forwarded(now_secs(), 1);
        let app = build_discovery_router_with_local_status(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::new(true, true, true, true),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(
            parsed["discovery_readiness"]["chat_relay_capability"]["status"].as_str(),
            Some("ready")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["status"].as_str(),
            Some("forming")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["stage"].as_str(),
            Some("single_hop_relay_ready")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["checks_total"].as_u64(),
            Some(4)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["checks_passed"].as_u64(),
            Some(2)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["blind_relay_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["relay_evidence_mode"].as_str(),
            Some("opaque_relay_acceptance")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["relay_readiness_reason"].as_str(),
            Some("opaque_relay_acceptance_observed")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["timestamp_rejected"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["real_relay_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["accepted_relay_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["synthetic_probe_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["two_hop_path_proof_ready"]
                .as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["two_hop_probe_succeeded"]
                .as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["privacy_invariant"].as_str(),
            Some("blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status")
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["status"].as_str(),
            Some("ready")
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["runtime_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["evidence_mode"].as_str(),
            Some("opaque_relay_acceptance")
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["readiness_reason"].as_str(),
            Some("opaque_relay_acceptance_observed")
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["real_relay_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["accepted_relay_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["synthetic_probe_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["two_hop_probe_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["accepted_total"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["timestamp_rejected"].as_u64(),
            Some(0)
        );

        let serialized = serde_json::to_string(&parsed["discovery_readiness"]).unwrap();
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("payload_b64"));
        assert!(!serialized.contains("client_ip"));
        assert_eq!(
            parsed["recovery_anchor"]["contract_version"].as_str(),
            Some("recovery_anchor.v1")
        );
        assert_eq!(parsed["recovery_anchor"]["status"].as_str(), Some("idle"));
    }

    #[tokio::test]
    async fn test_summary_endpoint_returns_public_safe_protocol_summary() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        store.record_blind_relay_forwarded(now, 1);
        store.record_blind_relay_two_hop_probe_result_with_context(
            now,
            true,
            "onion_terminal_delivered",
            4,
            3,
            2,
            1,
        );
        store.record_blind_relay_three_hop_probe_result_with_context(
            now,
            true,
            "onion_terminal_delivered",
            3,
            2,
            3,
            2,
        );
        let app = build_discovery_router_with_local_status(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::new(true, true, true, true),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/summary")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["source"].as_str(), Some("rust_discovery_summary"));
        assert_eq!(
            parsed["contract_version"].as_str(),
            Some("discovery_summary.v1")
        );
        assert_eq!(
            parsed["protocol_features"]["legacy_descriptor_gossip_v1"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["protocol_features"]["directory_descriptor_proof_gossip_v1"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["protocol_features"]["multihop_delivery_receipt_v1"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["protocol_features"]["purpose_bound_delivery_receipt_v2"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["protocol_features"]["onion_route_purpose_v1"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["protocol_features"]["onion_route_purposes"],
            serde_json::json!([
                "message_relay",
                "blind_vault_put",
                "blind_vault_pull",
                "blind_vault_delete"
            ])
        );
        assert_eq!(
            parsed["recovery_anchor"]["contract_version"].as_str(),
            Some("recovery_anchor.v1")
        );
        assert_eq!(parsed["local_capability"]["status"].as_str(), Some("ready"));
        assert_eq!(
            parsed["onion_relay_admission"]["status"].as_str(),
            Some("warming")
        );
        assert_eq!(
            parsed["onion_relay_admission"]["admission_score_percent"].as_u64(),
            Some(40)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["warmup_stage"].as_str(),
            Some("route_pool")
        );
        assert_eq!(
            parsed["onion_relay_admission"]["local_relay_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["recent_path_proof_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["route_pool_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(parsed["blind_relay"]["runtime_ready"].as_bool(), Some(true));
        assert_eq!(
            parsed["blind_relay"]["evidence_mode"].as_str(),
            Some("opaque_relay_acceptance")
        );
        assert_eq!(
            parsed["blind_relay"]["readiness_reason"].as_str(),
            Some("opaque_relay_acceptance_observed")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["proof_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["message_delivery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["recent_message_delivery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["message_delivery_evidence_mode"].as_str(),
            Some("synthetic_onion_message_delivery_probe")
        );
        assert_eq!(parsed["two_hop_path_proof"]["succeeded"].as_u64(), Some(1));
        assert_eq!(
            parsed["three_hop_path_proof"]["proof_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["three_hop_path_proof"]["message_delivery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["three_hop_path_proof"]["succeeded"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["three_hop_path_proof"]["path_shape_counts"]["entry_middle_middle_terminal"]
                .as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["three_hop_path_proof"]["persistence"].as_str(),
            Some("signed_local_cache_with_runtime_revalidation")
        );
        assert_eq!(
            parsed["three_hop_path_proof"]["proof_cache_rollback_protection"].as_str(),
            Some("not_observed")
        );
        assert_eq!(
            parsed["three_hop_path_proof"]["proof_cache_external_witness_required"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["message_delivery_successes"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_window_attempted"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_window_succeeded"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_window_failed"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_success_percent"].as_u64(),
            Some(100)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_status"].as_str(),
            Some("warming_up")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["failure_circuit_breaker_active"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["latest_age_bucket"].as_str(),
            Some("fresh")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["latest_reason_bucket"].as_str(),
            Some("onion_terminal_delivered")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["proof_scope"].as_str(),
            Some("message_delivery")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["proof_scope_counts"]["message_delivery"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["consecutive_message_delivery_successes"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["latest_message_delivery_age_seconds"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_recovery_configured"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["peer_quorum_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_survivable_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_recovery_basis"].as_str(),
            Some("waiting_for_peer_quorum")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["path_shape_counts"]["entry_middle_terminal"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["candidate_pool_counts"]["forming"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["ttl_shape_counts"]["entry_ttl_2_onward_ttl_1"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["route_governance"]["contract_version"].as_str(),
            Some("route_governance.v1")
        );
        assert_eq!(
            parsed["route_governance"]["status"].as_str(),
            Some("forming")
        );
        assert_eq!(
            parsed["route_governance"]["route_pool_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["route_governance"]["quality_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["route_governance"]["candidates_total"].as_u64(),
            Some(0)
        );
        assert_eq!(parsed["stage"].as_str(), Some("two_hop_path_ready"));
        assert_eq!(
            parsed["privacy_invariant"].as_str(),
            Some("blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status")
        );

        let serialized = serde_json::to_string(&parsed).unwrap();
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("payload_b64"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("client_ip"));
        assert!(!serialized.contains("receiver_pubkey"));
        assert!(!serialized.contains("public_endpoint"));
        assert!(!serialized.contains("selected_hop"));
    }

    #[test]
    fn test_onion_admission_requires_and_accepts_signed_proof_persistence() {
        let store = PeerStore::new();
        let now = now_secs();
        let first =
            signed_routeable_chat_descriptor(1, now + 1_000, "https://continuity-one.example");
        let first_node_id = first.node_id();
        let second =
            signed_routeable_chat_descriptor(1, now + 1_000, "https://continuity-two.example");
        let second_node_id = second.node_id();

        store.configure_bootstrap_status(true, true, true, 2);
        store
            .upsert_verified_from_source(first, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(second, now, "gossip_snapshot")
            .unwrap();
        store.record_gossip_round(now + 1, 2, 2, 2, None);
        store.record_route_forward_success(&first_node_id, now + 2);
        store.record_route_forward_success(&second_node_id, now + 3);
        for offset in 4..=6 {
            store.record_blind_relay_two_hop_probe_result_with_context(
                now + offset,
                true,
                "onion_terminal_delivered",
                2,
                1,
                2,
                1,
            );
        }

        let local_capabilities = DiscoveryLocalCapabilityStatus::new(true, true, true, true);
        store.record_two_hop_proof_cache_persisted(now + 60, 3, true);
        let before_persist = store.status(now + 7);
        let before_admission =
            onion_relay_admission_status_value(&before_persist, &local_capabilities);
        assert!(before_persist.two_hop_path_proof_history.stability_ready);
        assert_eq!(before_admission["status"].as_str(), Some("warming"));
        assert_eq!(
            before_admission["warmup_stage"].as_str(),
            Some("proof_restart_continuity")
        );
        assert_eq!(
            before_admission["admission_blockers"][0].as_str(),
            Some("proof_restart_continuity_not_ready")
        );
        assert_eq!(
            before_admission["proof_cache_signed_persistence_ready"].as_bool(),
            Some(false)
        );

        store.record_cache_save_status(now + 8, "success", "snapshot_persisted");
        store.record_two_hop_proof_cache_persisted(now + 8, 3, true);
        // [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] Production persistence
        // records the aggregate generation in the same successful cache round.
        store.record_client_delivery_cache_persisted(now + 8, 0, 1);
        let after_persist = store.status(now + 8);
        let after_admission =
            onion_relay_admission_status_value(&after_persist, &local_capabilities);

        assert_eq!(after_admission["status"].as_str(), Some("eligible"));
        assert_eq!(after_admission["eligible"].as_bool(), Some(true));
        assert_eq!(
            after_admission["restart_recovery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            after_admission["proof_restart_continuity_source"].as_str(),
            Some("signed_persistence")
        );
        assert_eq!(
            after_admission["proof_cache_signed_persistence_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            after_admission["proof_cache_rollback_protection"].as_str(),
            Some("anchored")
        );

        let summary = discovery_summary_response(now + 8, &after_persist, &local_capabilities);
        assert_eq!(
            summary.two_hop_path_proof["restart_survivable_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            summary.two_hop_path_proof["restart_recovery_basis"].as_str(),
            Some("message_delivery_proof_with_verified_restart_continuity")
        );

        store.record_client_delivery_witness_round(
            now + 9,
            1,
            true,
            1,
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                failed: 1,
                ..Default::default()
            },
        );
        let witness_blocked =
            onion_relay_admission_status_value(&store.status(now + 9), &local_capabilities);
        assert_eq!(witness_blocked["status"].as_str(), Some("warming"));
        assert_eq!(
            witness_blocked["proof_restart_continuity_source"].as_str(),
            Some("external_witness_not_ready")
        );
        assert_eq!(
            witness_blocked["proof_cache_external_witness"].as_str(),
            Some("unavailable")
        );

        store.record_client_delivery_witness_round(
            now + 10,
            1,
            true,
            1,
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                idempotent: 1,
                ..Default::default()
            },
        );
        let witness_verified =
            onion_relay_admission_status_value(&store.status(now + 10), &local_capabilities);
        assert_eq!(witness_verified["status"].as_str(), Some("eligible"));
        assert_eq!(
            witness_verified["proof_cache_external_witness"].as_str(),
            Some("verified")
        );
    }

    #[test]
    fn test_recovery_anchor_status_requires_exact_witness_generation() {
        let store = PeerStore::new();
        let now = now_secs();
        store.record_routeability_cache_rollback_protection(now, 2, "anchored");
        store.record_two_hop_proof_cache_persisted(now, 3, true);
        store.record_three_hop_proof_cache_persisted(now, 3, true);
        store.record_client_delivery_cache_persisted(now, 2, 2);
        store.record_client_delivery_witness_round(
            now,
            1,
            true,
            1,
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                idempotent: 1,
                ..Default::default()
            },
        );

        let mismatched_status = store.status(now + 1);
        let mismatched_anchor = recovery_anchor_status_value(&mismatched_status);
        assert_eq!(mismatched_anchor["status"].as_str(), Some("blocked"));
        assert_eq!(
            mismatched_anchor["local_anchor"]["ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            mismatched_anchor["external_witness"]["status"].as_str(),
            Some("verified")
        );
        assert_eq!(
            mismatched_anchor["external_witness"]["generation_aligned"].as_bool(),
            Some(false)
        );
        assert_eq!(
            mismatched_anchor["external_witness"]["ready"].as_bool(),
            Some(false)
        );
        assert!(!two_hop_proof_restart_continuity(&mismatched_status).ready);

        store.record_client_delivery_witness_round(
            now + 2,
            2,
            true,
            1,
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                idempotent: 1,
                ..Default::default()
            },
        );
        let aligned_status = store.status(now + 2);
        let aligned_anchor = recovery_anchor_status_value(&aligned_status);
        assert_eq!(aligned_anchor["status"].as_str(), Some("ready"));
        assert_eq!(
            aligned_anchor["external_witness"]["generation_aligned"].as_bool(),
            Some(true)
        );
        assert_eq!(
            aligned_anchor["external_witness"]["ready"].as_bool(),
            Some(true)
        );
        assert!(two_hop_proof_restart_continuity(&aligned_status).ready);
    }

    #[test]
    fn test_optional_external_witness_adverse_evidence_blocks_recovery() {
        let store = PeerStore::new();
        let now = now_secs();
        store.record_routeability_cache_rollback_protection(now, 1, "anchored");
        store.record_two_hop_proof_cache_persisted(now, 3, true);
        store.record_three_hop_proof_cache_persisted(now, 3, true);
        store.record_client_delivery_cache_persisted(now, 0, 1);

        // [EXTERNAL-WITNESS-ADVERSE-GATE 2026-08-21 by Codex] An optional
        // witness may be unavailable without becoming an availability
        // dependency. Once a valid witness reports rollback, conflict, or a
        // generation gap, however, both recovery and relay admission must
        // reject the anchored state even though strict quorum was not enabled.
        store.record_client_delivery_witness_round(
            now + 1,
            1,
            false,
            1,
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                failed: 1,
                ..Default::default()
            },
        );
        let unavailable_status = store.status(now + 1);
        let unavailable_anchor = recovery_anchor_status_value(&unavailable_status);
        assert_eq!(unavailable_anchor["status"].as_str(), Some("ready"));
        assert_eq!(
            unavailable_anchor["external_witness"]["required"].as_bool(),
            Some(false)
        );
        assert_eq!(
            unavailable_anchor["external_witness"]["ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            unavailable_anchor["external_witness"]["adverse_evidence"].as_bool(),
            Some(false)
        );
        assert!(two_hop_proof_restart_continuity(&unavailable_status).ready);

        let adverse_rounds = [
            (
                "rollback_detected",
                crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                    configured: 1,
                    attempted: 1,
                    verified: 1,
                    stale: 1,
                    ..Default::default()
                },
            ),
            (
                "conflict",
                crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                    configured: 1,
                    attempted: 1,
                    verified: 1,
                    conflicts: 1,
                    ..Default::default()
                },
            ),
            (
                "gap",
                crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                    configured: 1,
                    attempted: 1,
                    verified: 1,
                    gaps: 1,
                    ..Default::default()
                },
            ),
        ];

        for (offset, (expected_status, round)) in adverse_rounds.into_iter().enumerate() {
            let observed_at = now + offset as u64 + 2;
            store.record_client_delivery_witness_round(observed_at, 1, false, 1, round);
            let status = store.status(observed_at);
            let anchor = recovery_anchor_status_value(&status);

            assert_eq!(
                anchor["external_witness"]["status"].as_str(),
                Some(expected_status)
            );
            assert_eq!(anchor["status"].as_str(), Some("blocked"));
            assert_eq!(anchor["ready_for_restore"].as_bool(), Some(false));
            assert_eq!(
                anchor["external_witness"]["adverse_evidence"].as_bool(),
                Some(true)
            );
            assert_eq!(anchor["external_witness"]["ready"].as_bool(), Some(false));
            let continuity = two_hop_proof_restart_continuity(&status);
            assert!(!continuity.ready);
            assert_eq!(continuity.source, "external_witness_not_ready");
        }
    }

    #[tokio::test]
    async fn test_public_card_endpoint_returns_minimal_product_protocol_card() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let middle = signed_routeable_chat_descriptor(1, now + 300, "https://middle.example");
        let middle_node_id = middle.node_id();
        let terminal = signed_routeable_chat_descriptor(1, now + 300, "https://terminal.example");
        let terminal_node_id = terminal.node_id();

        store.upsert_verified(middle, now).unwrap();
        store.upsert_verified(terminal, now).unwrap();
        // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Public
        // `real_relay_ready` now requires current routeability in addition to
        // purpose-bound receipt evidence and network-diverse endpoints.
        store.record_route_forward_success(&middle_node_id, now);
        store.record_route_forward_success(&terminal_node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&middle_node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&terminal_node_id, now);
        store.record_blind_relay_terminal(now, 2, 128);
        store.record_blind_relay_forwarded(now, 1);
        store.record_blind_relay_two_hop_probe_result_with_context(
            now,
            true,
            "onion_terminal_delivered",
            4,
            3,
            2,
            1,
        );
        store.record_verified_client_onion_delivery(now);
        let app = build_discovery_router_with_local_status(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::new(true, true, true, true),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/public-card")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(
            parsed["source"].as_str(),
            Some(DISCOVERY_PUBLIC_CARD_SOURCE)
        );
        assert_eq!(
            parsed["contract_version"].as_str(),
            Some(DISCOVERY_PUBLIC_CARD_CONTRACT_VERSION)
        );
        assert_eq!(
            parsed["cards"]["protocol_health"]["label"].as_str(),
            Some("AeroNyx Privacy Protocol")
        );
        assert_eq!(
            parsed["cards"]["verified_mesh"]["label"].as_str(),
            Some("Verified Node Mesh")
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["label"].as_str(),
            Some("Blind Relay")
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["terminal_delivered_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["middle_forwarded_count"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["real_relay_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["verified_client_onion_deliveries"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["proof_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["message_delivery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["cards"]["blind_relay"]["message_delivery_evidence_mode"].as_str(),
            Some("verified_client_onion_delivery_receipt")
        );
        assert_eq!(
            parsed["signals"]["permissionless_node_admission"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["signals"]["latest_path_proof_reason_bucket"].as_str(),
            Some("onion_terminal_delivered")
        );
        assert_eq!(
            parsed["display_policy"]["primary_surface"].as_str(),
            Some("show_protocol_health_verified_mesh_and_blind_relay")
        );
        assert_eq!(
            parsed["privacy_invariant"].as_str(),
            Some("blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status")
        );

        let serialized = serde_json::to_string(&parsed).unwrap();
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("payload_b64"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("client_ip"));
        assert!(!serialized.contains("receiver_pubkey"));
        assert!(!serialized.contains("public_endpoint"));
        assert!(!serialized.contains("selected_hop"));
    }

    #[tokio::test]
    async fn test_summary_status_recovers_when_latest_two_hop_message_delivery_is_ready() {
        let store = Arc::new(PeerStore::new());
        let now = now_secs();
        let first = signed_routeable_chat_descriptor(1, now + 1_000, "https://peer-one.example");
        let first_node_id = first.node_id();
        let second = signed_routeable_chat_descriptor(1, now + 1_000, "https://peer-two.example");
        let second_node_id = second.node_id();

        store.configure_bootstrap_status(true, true, true, 2);
        store
            .upsert_verified_from_source(first, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(second, now, "gossip_snapshot")
            .unwrap();
        store.record_gossip_round(now, 2, 2, 2, None);
        store.record_route_forward_success(&first_node_id, now);
        store.record_route_forward_success(&second_node_id, now);

        for _ in 0..6 {
            store.record_blind_relay_two_hop_probe_result_with_context(
                now,
                false,
                "request_error",
                2,
                1,
                2,
                1,
            );
        }
        store.record_blind_relay_two_hop_probe_result_with_context(
            now,
            true,
            "onion_terminal_delivered",
            2,
            1,
            2,
            1,
        );

        let app = build_discovery_router_with_local_status(
            store,
            DiscoveryApiPolicy::default(),
            DiscoveryLocalCapabilityStatus::new(true, true, true, true),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/summary")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["status"].as_str(), Some("ready"));
        assert_eq!(parsed["stage"].as_str(), Some("two_hop_path_ready"));
        assert_eq!(
            parsed["blind_relay"]["evidence_mode"].as_str(),
            Some("synthetic_onion_message_delivery_probe")
        );
        assert_eq!(
            parsed["blind_relay"]["readiness_reason"].as_str(),
            Some("synthetic_onion_message_delivery_probe_ready")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["latest_reason_bucket"].as_str(),
            Some("onion_terminal_delivered")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["recent_message_delivery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["message_delivery_evidence_mode"].as_str(),
            Some("synthetic_onion_message_delivery_probe")
        );
        assert_eq!(
            parsed["peer_mesh"]["chat_two_hop_onion_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["status"].as_str(),
            Some("warming")
        );
        assert_eq!(
            parsed["onion_relay_admission"]["admission_score_percent"].as_u64(),
            Some(60)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["warmup_stage"].as_str(),
            Some("stability_window")
        );
        assert_eq!(
            parsed["onion_relay_admission"]["admission_blockers"][0].as_str(),
            Some("stable_path_proof_not_ready")
        );
        assert_eq!(
            parsed["onion_relay_admission"]["route_pool_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["restart_recovery_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["peer_restart_recovery_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["proof_restart_continuity_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["stable_path_proof_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["two_hop_stability_window_attempted"].as_u64(),
            Some(7)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["two_hop_stability_window_succeeded"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["two_hop_stability_min_attempts"].as_u64(),
            Some(3)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["two_hop_stability_remaining_attempts"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["two_hop_stability_success_threshold_percent"].as_u64(),
            Some(80)
        );
        assert_eq!(
            parsed["onion_relay_admission"]["probe_cadence_policy"].as_str(),
            Some("recovery_cadence_until_stability_window_ready_then_low_frequency")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_recovery_configured"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["peer_quorum_ready"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_survivable_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_recovery_basis"].as_str(),
            Some("proof_restart_continuity_not_ready")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_window_attempted"].as_u64(),
            Some(7)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_window_succeeded"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_window_failed"].as_u64(),
            Some(6)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_status"].as_str(),
            Some("degraded")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["stability_ready"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["failure_circuit_breaker_active"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["latest_age_bucket"].as_str(),
            Some("fresh")
        );

        let serialized = serde_json::to_string(&parsed).unwrap();
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("payload_b64"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("client_ip"));
        assert!(!serialized.contains("receiver_pubkey"));
        assert!(!serialized.contains("public_endpoint"));
        assert!(!serialized.contains("selected_hop"));
    }

    #[tokio::test]
    async fn test_snapshot_endpoint_caps_requested_limit() {
        let store = Arc::new(PeerStore::new());
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        store
            .upsert_verified(signed_descriptor(), now_secs())
            .unwrap();
        let mut policy = DiscoveryApiPolicy::default();
        policy.max_snapshot_limit = 1;
        let app = build_discovery_router(store, policy);

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/api/discovery/snapshot?limit=50")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let snapshot: NodeBootstrapSnapshot = serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot.peers.len(), 1);
    }

    #[tokio::test]
    async fn test_gossip_denies_blocked_descriptor() {
        let store = Arc::new(PeerStore::new());
        let descriptor = signed_descriptor();
        let mut policy = DiscoveryApiPolicy::default();
        policy
            .denied_peer_ids
            .insert(hex::encode(descriptor.node_id()));
        let app = build_discovery_router(Arc::clone(&store), policy);
        let body =
            serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce { descriptor }).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(store.len(), 0);
        let status = store.status(now_secs());
        assert_eq!(status.runtime.policy_rejected, 1);
        assert_eq!(
            status
                .recent_audit_events
                .last()
                .map(|event| event.action.as_str()),
            Some("gossip_policy_rejected")
        );
    }

    #[tokio::test]
    async fn gossip_rate_limiter_recovers_after_lock_owner_panic() {
        // [DISCOVERY-RATE-LIMIT-RECOVERY 2026-07-30 by Codex] The historical
        // std::sync::Mutex became permanently poisoned here, causing every
        // later gossip request to panic at lock().expect().
        let rate_limit = Arc::new(Mutex::new(RateLimitState::new()));
        let panic_lock = Arc::clone(&rate_limit);
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _guard = panic_lock.lock();
            panic!("test-only-discovery-rate-limit-panic");
        }));
        assert!(panic_result.is_err());

        let state = DiscoveryApiState {
            peer_store: Arc::new(PeerStore::new()),
            local_node_id: None,
            directory_replica_store: None,
            policy: DiscoveryApiPolicy::default(),
            local_capabilities: DiscoveryLocalCapabilityStatus::default(),
            rate_limit,
            route_domain_certificate_rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
        };
        let response = gossip_handler(
            State(state),
            Json(NodeDiscoveryMessage::SnapshotRequest {
                requested_at: now_secs(),
                limit: Some(1),
            }),
        )
        .await
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_gossip_rate_limit_rejects_excess_requests() {
        let store = Arc::new(PeerStore::new());
        let mut policy = DiscoveryApiPolicy::default();
        policy.gossip_rate_limit_per_minute = 1;
        let app = build_discovery_router(Arc::clone(&store), policy);

        for expected_status in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
            let body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
                requested_at: now_secs(),
                limit: Some(1),
            })
            .unwrap();
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method(Method::POST)
                        .uri("/api/discovery/gossip")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status);
        }

        let status = store.status(now_secs());
        assert_eq!(status.runtime.rate_limited, 1);
        assert_eq!(
            status
                .recent_audit_events
                .last()
                .map(|event| event.action.as_str()),
            Some("gossip_rate_limited")
        );
    }
}
