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
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::discovery::{
    decode_route_domain_attestation_certificate,
    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES, MAX_SIGNED_NODE_DESCRIPTOR_BYTES,
};
use aeronyx_core::protocol::{
    DiscoveryEndpointEvidenceAttestationV1, NodeBootstrapSnapshot, NodeCapability,
    NodeDiscoveryMessage, NodeProtocolFeature, OnionRoutePurpose, SignedNodeDescriptor,
    MAX_VERIFIED_ONION_ROUTE_HOPS, ONION_FORWARD_HOP_REQUIRED_CAPABILITIES,
    ONION_ROUTE_PURPOSE_VALUES,
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
use crate::api::public_node_router::public_endpoint_flow_context;
use crate::config::DiscoveryConfig;
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
            NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { attestation_frame } => {
                DiscoveryEndpointEvidenceAttestationV1::decode(attestation_frame)
                    .map(|attestation| self.node_allowed(&attestation.subject_node_id()))
                    .unwrap_or(false)
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
        .route("/api/discovery/summary", get(summary_handler))
        .route("/api/discovery/public-card", get(public_card_handler))
        .route(
            "/api/discovery/onion-candidates",
            get(onion_candidates_handler),
        )
        .layer(DefaultBodyLimit::max(DISCOVERY_REQUEST_BODY_MAX_BYTES))
        .with_state(state)
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
