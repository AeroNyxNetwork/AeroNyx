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
//!   applies descriptor/snapshot updates, and returns a snapshot response for
//!   request messages
//! - `GET /api/discovery/status`: returns aggregate peer-store status, local
//!   capability readiness, and compact discovery readiness for dashboards
//! - `GET /api/discovery/summary`: returns a compact public-safe protocol
//!   foundation summary for app, website, backend aggregation, and AI runbooks,
//!   including aggregate route-governance readiness without route metadata
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
//! 4. Response reports import counts and optionally includes a snapshot response
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
//!
//! ## Last Modified
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

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::{NodeBootstrapSnapshot, NodeCapability, NodeDiscoveryMessage};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

use crate::config::DiscoveryConfig;
use crate::services::{PeerStore, PeerStoreImportReport, PeerStoreStatus};

// ============================================
// State / Request / Response Types
// ============================================

const ONION_CANDIDATES_CONTRACT_VERSION: &str = "onion_candidates.v1";
const ONION_CANDIDATES_SOURCE: &str = "rust_discovery_onion_candidates";
const ONION_CANDIDATES_SELECTION_POLICY: &str =
    "fresh_routeable_signed_chat_relays_with_kem_public_key";
const ONION_CANDIDATES_REFRESH_AFTER_SECONDS: u64 = 300;
const ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS: u64 = 1_800;
const ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES: usize = 2;
const ONION_CANDIDATES_MAX_CLIENT_HOPS: u8 = 3;
const ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS: u64 = 3;
const ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT: u8 = 80;

#[derive(Clone)]
struct DiscoveryApiState {
    peer_store: Arc<PeerStore>,
    policy: DiscoveryApiPolicy,
    local_capabilities: DiscoveryLocalCapabilityStatus,
    rate_limit: Arc<Mutex<RateLimitState>>,
}

/// API-facing discovery safety policy.
#[derive(Debug, Clone)]
pub struct DiscoveryApiPolicy {
    max_snapshot_limit: usize,
    gossip_rate_limit_per_minute: u32,
    allowed_peer_ids: HashSet<String>,
    denied_peer_ids: HashSet<String>,
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
        }
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
}

impl Default for DiscoveryApiPolicy {
    fn default() -> Self {
        Self {
            max_snapshot_limit: DiscoveryConfig::default_max_snapshot_limit(),
            gossip_rate_limit_per_minute: DiscoveryConfig::default_gossip_rate_limit_per_minute(),
            allowed_peer_ids: HashSet::new(),
            denied_peer_ids: HashSet::new(),
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

    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Enhanced => "enhanced",
            Self::High => "high",
        }
    }

    fn default_hops(self) -> u8 {
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
    /// Best hop count the client can safely attempt from this response.
    pub recommended_hops: u8,
    /// Whether the client should fall back to the standard encrypted relay path
    /// before attempting to build an onion envelope.
    pub fallback_required: bool,
    /// Aggregate candidate-pool maturity bucket.
    ///
    /// Stable values are `ready`, `warming`, `empty`, or `client_limited`.
    /// This lets App, nodeboard, backend aggregation, and AI-agent runbooks
    /// distinguish a usable pool from a partial pool without inspecting
    /// individual relay metadata.
    pub pool_status: String,
    /// Privacy-safe route plan recommendation for clients.
    ///
    /// Stable values are `two_hop_onion_path` or `standard_relay_fallback`.
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
    /// Stable strategy clients should use when choosing among candidates.
    pub path_selection_strategy: String,
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
    /// Aggregate permissionless relay-pool admission gate without route data.
    onion_relay_admission: serde_json::Value,
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
    snapshot_default_public_only: bool,
    private_descriptors_hidden_by_default: bool,
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
    let restart_recovery_ready = peer_quorum.restart_recovery_configured;
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
    } else if !restart_recovery_ready {
        "restart_recovery"
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
    if !restart_recovery_ready {
        admission_blockers.push("restart_recovery_not_ready");
    }
    let warmup_hint = match warmup_stage {
        "eligible" => "node is eligible for client-selected two-hop onion relay paths".to_string(),
        "local_relay" => {
            "align ChatRelay config, runtime, public peer API, and advertised capability".to_string()
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
        _ => "continue warming relay admission gates".to_string(),
    };

    serde_json::json!({
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
    })
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
        "disabled" => "enable discovery and chat relay capability before advertising protocol readiness",
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

    serde_json::json!({
        "generated_at": generated_at,
        "contract_version": "blind_relay_runtime.v1",
        "source": "rust_blind_relay_runtime",
        "status": &quality.status,
        "runtime_ready": quality.runtime_ready,
        "quality_ready": quality.quality_ready,
        "real_relay_ready": quality.real_relay_ready,
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
    })
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
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let two_hop_history = &status.two_hop_path_proof_history;
    let two_hop_restart_survivable_ready = two_hop_history.recent_message_delivery_ready
        && peer_quorum.quorum_ready
        && peer_quorum.restart_recovery_configured;
    let two_hop_restart_recovery_basis = if two_hop_restart_survivable_ready {
        "message_delivery_proof_with_restart_recovery"
    } else if !two_hop_history.recent_message_delivery_ready {
        "waiting_for_fresh_message_delivery_proof"
    } else if !peer_quorum.quorum_ready {
        "waiting_for_peer_quorum"
    } else {
        "restart_recovery_not_configured"
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
        onion_relay_admission,
        next_action,
        privacy_invariant: "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        privacy_boundary: "aggregate discovery summary only; no signed descriptors, full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, wallet-level traffic, or social graph metadata",
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
    let state = DiscoveryApiState {
        peer_store,
        policy,
        local_capabilities,
        rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
    };
    Router::new()
        .route("/api/discovery/snapshot", get(snapshot_handler))
        .route("/api/discovery/gossip", post(gossip_handler))
        .route("/api/discovery/status", get(status_handler))
        .route("/api/discovery/summary", get(summary_handler))
        .route(
            "/api/discovery/onion-candidates",
            get(onion_candidates_handler),
        )
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
    let requested_privacy_mode = OnionPrivacyMode::from_query(query.privacy_mode.as_deref());
    let requested_hops = normalize_requested_hops(requested_privacy_mode, query.hops);
    let candidates: Vec<OnionRelayCandidate> = state
        .peer_store
        .route_candidates_with_capability(NodeCapability::ChatRelay, now, limit)
        .into_iter()
        .enumerate()
        .filter_map(|descriptor| {
            let (rank, descriptor) = descriptor;
            let node_id = descriptor.node_id();
            if !state.peer_store.is_routeable_now(&node_id, now) {
                return None;
            }
            let kem_public = descriptor.descriptor.x25519_kem_public()?;
            let public_endpoint = descriptor.descriptor.public_endpoint.clone()?;
            let capacity = descriptor.descriptor.capacity.clone();
            Some(OnionRelayCandidate {
                node_id: hex::encode(node_id),
                kem_alg: descriptor.descriptor.kem_alg,
                kem_public: hex::encode(kem_public),
                public_endpoint,
                capabilities: descriptor.descriptor.capabilities.clone(),
                selection_weight: onion_candidate_selection_weight(rank),
                region: descriptor.descriptor.policy.region.clone(),
                max_sessions: capacity.max_sessions,
                max_bps: capacity.max_bps,
                max_pps: capacity.max_pps,
            })
        })
        .collect();
    let two_hop_ready = candidates.len() >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;
    let min_candidates_for_requested_hops = requested_hops as usize;
    let requested_path_ready = candidates.len() >= min_candidates_for_requested_hops;
    let recommended_hops = recommended_onion_hops(candidates.len(), requested_hops);
    let fallback_reason =
        onion_candidate_fallback_reason(candidates.len(), limit, min_candidates_for_requested_hops);
    let pool_status =
        onion_candidate_pool_status(candidates.len(), limit, min_candidates_for_requested_hops);
    let route_plan = onion_candidate_route_plan(requested_path_ready, recommended_hops);
    let readiness_reason = onion_candidate_readiness_reason(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
    );
    let next_action = onion_candidate_next_action(requested_path_ready, fallback_reason);

    Json(OnionCandidatesResponse {
        generated_at: now,
        contract_version: ONION_CANDIDATES_CONTRACT_VERSION.to_string(),
        source: ONION_CANDIDATES_SOURCE.to_string(),
        count: candidates.len(),
        min_candidates_for_two_hop: ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        two_hop_ready,
        requested_privacy_mode: requested_privacy_mode.as_str().to_string(),
        requested_hops,
        min_candidates_for_requested_hops,
        requested_path_ready,
        recommended_hops,
        fallback_required: !requested_path_ready,
        pool_status: pool_status.to_string(),
        route_plan: route_plan.to_string(),
        fallback_reason: fallback_reason.to_string(),
        readiness_reason: readiness_reason.to_string(),
        next_action: next_action.to_string(),
        selection_policy: ONION_CANDIDATES_SELECTION_POLICY.to_string(),
        path_selection_strategy: "weighted_random_health_ranked_distinct_hops".to_string(),
        region_diversity_policy:
            "prefer_distinct_regions_when_available_without_exposing_selected_route".to_string(),
        user_choice_policy:
            "users_choose_privacy_mode; clients select distinct routeable relays automatically"
                .to_string(),
        refresh_after_seconds: ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
        routeability_stale_after_seconds: ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        candidates,
        privacy_boundary: "fresh routeable signed node discovery metadata only (node id, KEM public key, public endpoint, capabilities); no client IPs, route ids, encrypted payloads, receiver identities, DNS contents, destinations, voucher secrets, private keys, or wallet-level traffic".to_string(),
    })
}

fn normalize_requested_hops(mode: OnionPrivacyMode, requested: Option<u8>) -> u8 {
    requested
        .unwrap_or_else(|| mode.default_hops())
        .clamp(1, ONION_CANDIDATES_MAX_CLIENT_HOPS)
}

fn recommended_onion_hops(candidate_count: usize, requested_hops: u8) -> u8 {
    (candidate_count.min(requested_hops as usize) as u8).min(ONION_CANDIDATES_MAX_CLIENT_HOPS)
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
) -> &'static str {
    if candidate_count >= required_candidates {
        "ready"
    } else if limit < required_candidates {
        "client_limited"
    } else if candidate_count == 0 {
        "empty"
    } else {
        "warming"
    }
}

fn onion_candidate_route_plan(requested_path_ready: bool, recommended_hops: u8) -> &'static str {
    if !requested_path_ready {
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
) -> &'static str {
    if candidate_count >= required_candidates {
        "ready"
    } else if limit < required_candidates {
        if required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES {
            "client_limit_below_two_hop_minimum"
        } else {
            "client_limit_below_requested_hops"
        }
    } else if candidate_count == 0 {
        "no_routeable_candidates"
    } else if candidate_count == 1 && required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
    {
        "single_routeable_candidate"
    } else {
        "insufficient_routeable_candidates"
    }
}

fn onion_candidate_readiness_reason(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
) -> &'static str {
    match onion_candidate_fallback_reason(candidate_count, limit, required_candidates) {
        "ready" => {
            if required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES {
                "two_hop_candidate_pool_ready"
            } else {
                "requested_onion_candidate_pool_ready"
            }
        }
        "client_limit_below_two_hop_minimum" => "client_limit_blocks_two_hop_pool",
        "client_limit_below_requested_hops" => "client_limit_blocks_requested_hops",
        "no_routeable_candidates" => "waiting_for_routeable_kem_relays",
        "single_routeable_candidate" => "waiting_for_second_routeable_kem_relay",
        _ => "waiting_for_more_routeable_kem_relays",
    }
}

fn onion_candidate_next_action(requested_path_ready: bool, fallback_reason: &str) -> &'static str {
    if requested_path_ready {
        "build a weighted-random onion path with fresh distinct candidates"
    } else if fallback_reason == "client_limit_below_requested_hops"
        || fallback_reason == "client_limit_below_two_hop_minimum"
    {
        "increase candidate limit or use standard encrypted relay fallback"
    } else {
        "use standard encrypted relay fallback and refresh candidate pool later"
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
        .expect("discovery rate limiter poisoned")
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

    let applied = state.peer_store.apply_discovery_message(&message, now);
    state.peer_store.mark_gossip_at(now);
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
        | NodeDiscoveryMessage::DescriptorAnnounce { .. } => None,
    };

    (StatusCode::OK, Json(GossipResponse { applied, response })).into_response()
}

async fn status_handler(State(state): State<DiscoveryApiState>) -> Json<DiscoveryStatusResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    let discovery_readiness = discovery_readiness_status_value(&peer_store, &local_capabilities);
    let blind_relay_runtime =
        blind_relay_runtime_status_value(now, &peer_store, &local_capabilities);
    Json(DiscoveryStatusResponse {
        generated_at: now,
        peer_store,
        policy: DiscoveryPolicyStatus {
            max_snapshot_limit: state.policy.max_snapshot_limit,
            gossip_rate_limit_per_minute: state.policy.gossip_rate_limit_per_minute,
            allow_list_enabled: !state.policy.allowed_peer_ids.is_empty(),
            allowed_peer_count: state.policy.allowed_peer_ids.len(),
            denied_peer_count: state.policy.denied_peer_ids.len(),
            snapshot_default_public_only: true,
            private_descriptors_hidden_by_default: true,
        },
        local_capabilities,
        discovery_readiness,
        blind_relay_runtime,
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

    fn signed_routeable_chat_descriptor(
        sequence: u64,
        expires_at: u64,
        endpoint: &str,
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
        descriptor.capacity = NodeCapacity {
            max_sessions: 128,
            max_bps: Some(500_000_000),
            max_pps: None,
        };
        descriptor.policy = NodePolicy::default();
        SignedNodeDescriptor::sign(descriptor, &kp).unwrap()
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

        // (a) Routeable ChatRelay relay advertising a KEM key + endpoint -> included.
        let kp = IdentityKeyPair::generate();
        let kem = kp.x25519_public_key_bytes();
        let mut included = NodeDescriptor::new(
            kp.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        included.capabilities = vec![NodeCapability::ChatRelay];
        included.public_endpoint = Some("relay.example:443".to_string());
        let included = included.with_x25519_kem(kem);
        let included = aeronyx_core::protocol::SignedNodeDescriptor::sign(included, &kp).unwrap();
        let want_node_id = hex::encode(included.node_id());
        let included_node_id = included.node_id();
        store.upsert_verified(included, now).unwrap();
        store.record_route_forward_success(&included_node_id, now);

        // (b) ChatRelay relay WITHOUT a KEM key -> filtered out (cannot be a hop).
        let kp2 = IdentityKeyPair::generate();
        let mut no_kem = NodeDescriptor::new(
            kp2.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        no_kem.capabilities = vec![NodeCapability::ChatRelay];
        no_kem.public_endpoint = Some("nokem.example:443".to_string());
        let no_kem = aeronyx_core::protocol::SignedNodeDescriptor::sign(no_kem, &kp2).unwrap();
        store.upsert_verified(no_kem, now).unwrap();

        // (c) KEM-bearing ChatRelay without routeability evidence -> filtered
        // out. This keeps clients from building paths through unknown peers
        // while allowing internal probes to continue learning about them.
        let kp3 = IdentityKeyPair::generate();
        let mut unknown = NodeDescriptor::new(
            kp3.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 300,
            "test",
        );
        unknown.capabilities = vec![NodeCapability::ChatRelay];
        unknown.public_endpoint = Some("unknown.example:443".to_string());
        let unknown = unknown.with_x25519_kem(kp3.x25519_public_key_bytes());
        let unknown = aeronyx_core::protocol::SignedNodeDescriptor::sign(unknown, &kp3).unwrap();
        store.upsert_verified(unknown, now).unwrap();

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

        // Only the routeable KEM-bearing relay is exposed, with its KEM key for the client.
        assert_eq!(parsed.contract_version, ONION_CANDIDATES_CONTRACT_VERSION);
        assert_eq!(parsed.source, ONION_CANDIDATES_SOURCE);
        assert_eq!(parsed.selection_policy, ONION_CANDIDATES_SELECTION_POLICY);
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
        assert_eq!(parsed.recommended_hops, 1);
        assert!(!parsed.two_hop_ready);
        assert!(parsed.fallback_required);
        assert_eq!(parsed.pool_status, "warming");
        assert_eq!(parsed.route_plan, "standard_relay_fallback");
        assert_eq!(parsed.fallback_reason, "single_routeable_candidate");
        assert_eq!(
            parsed.readiness_reason,
            "waiting_for_second_routeable_kem_relay"
        );
        assert_eq!(
            parsed.next_action,
            "use standard encrypted relay fallback and refresh candidate pool later"
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
        assert_eq!(candidate.selection_weight, 1_000);
        assert_eq!(candidate.region, None);
        assert_eq!(candidate.max_sessions, 0);
        assert!(parsed.privacy_boundary.contains("fresh routeable"));
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
            Some("real_relay_traffic")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["relay_readiness_reason"].as_str(),
            Some("real_relay_observed")
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["timestamp_rejected"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["discovery_readiness"]["protocol_foundation"]["real_relay_ready"].as_bool(),
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
            Some("real_relay_traffic")
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["readiness_reason"].as_str(),
            Some("real_relay_observed")
        );
        assert_eq!(
            parsed["discovery_readiness"]["blind_relay_runtime"]["real_relay_ready"].as_bool(),
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
        assert_eq!(parsed["two_hop_path_proof"]["succeeded"].as_u64(), Some(1));
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
        store.record_gossip_round(now + 1, 2, 2, 2, None);
        store.record_route_forward_success(&first_node_id, now + 2);
        store.record_route_forward_success(&second_node_id, now + 3);

        for offset in 4..10 {
            store.record_blind_relay_two_hop_probe_result_with_context(
                now + offset,
                false,
                "request_error",
                2,
                1,
                2,
                1,
            );
        }
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + 10,
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
            parsed["two_hop_path_proof"]["latest_reason_bucket"].as_str(),
            Some("onion_terminal_delivered")
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["recent_message_delivery_ready"].as_bool(),
            Some(true)
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
            Some(80)
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
            Some(true)
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
            Some(true)
        );
        assert_eq!(
            parsed["two_hop_path_proof"]["restart_recovery_basis"].as_str(),
            Some("message_delivery_proof_with_restart_recovery")
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
