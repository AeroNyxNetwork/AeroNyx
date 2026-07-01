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
//!   foundation summary for app, website, backend aggregation, and AI runbooks
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
}

/// Response for `GET /api/discovery/onion-candidates`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnionCandidatesResponse {
    /// Unix timestamp when the candidate set was generated.
    pub generated_at: u64,
    /// Number of candidates returned.
    pub count: usize,
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
    /// Blind relay aggregate runtime/probe evidence without payload metadata.
    blind_relay: serde_json::Value,
    /// Bounded two-hop path proof aggregate without route reconstruction data.
    two_hop_path_proof: serde_json::Value,
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
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let local_relay_ready = local_capabilities.safe_to_advertise_chat_relay;
    let peer_mesh_ready = peer_quorum.quorum_ready;
    let blind_relay_ready = blind_relay_quality.runtime_ready && blind_relay_quality.quality_ready;
    let two_hop_path_ready =
        network_story.chat_two_hop_onion_ready || blind_relay_quality.two_hop_probe_ready;
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
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let two_hop_history = &status.two_hop_path_proof_history;

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
        two_hop_path_proof: serde_json::json!({
            "status": &two_hop_history.status,
            "freshness_bucket": &two_hop_history.freshness_bucket,
            "proof_ready": two_hop_history.proof_ready,
            "recent_success_ready": two_hop_history.recent_success_ready,
            "message_delivery_ready": two_hop_history.message_delivery_ready,
            "recent_message_delivery_ready": two_hop_history.recent_message_delivery_ready,
            "failure_streak_active": two_hop_history.failure_streak_active,
            "retained_events": two_hop_history.retained_events,
            "attempted": two_hop_history.attempted,
            "succeeded": two_hop_history.succeeded,
            "message_delivery_successes": two_hop_history.message_delivery_successes,
            "failed": two_hop_history.failed,
            "success_percent": two_hop_history.success_percent,
            "latest_outcome": &two_hop_history.latest_outcome,
            "latest_reason_bucket": &two_hop_history.latest_reason_bucket,
            "latest_age_seconds": two_hop_history.latest_age_seconds,
            "latest_success_age_seconds": two_hop_history.latest_success_age_seconds,
            "latest_failure_age_seconds": two_hop_history.latest_failure_age_seconds,
            "latest_message_delivery_age_seconds": two_hop_history.latest_message_delivery_age_seconds,
            "consecutive_successes": two_hop_history.consecutive_successes,
            "consecutive_failures": two_hop_history.consecutive_failures,
            "consecutive_message_delivery_successes": two_hop_history.consecutive_message_delivery_successes,
            "path_shape_counts": &two_hop_history.path_shape_counts,
            "candidate_pool_counts": &two_hop_history.candidate_pool_counts,
            "ttl_shape_counts": &two_hop_history.ttl_shape_counts,
            "proof_scope": &two_hop_history.proof_scope,
            "proof_scope_counts": &two_hop_history.proof_scope_counts,
            "stale_after_seconds": two_hop_history.stale_after_seconds,
            "next_action": &two_hop_history.next_action,
        }),
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
    let candidates: Vec<OnionRelayCandidate> = state
        .peer_store
        .route_candidates_with_capability(NodeCapability::ChatRelay, now, limit)
        .into_iter()
        .filter_map(|descriptor| {
            let node_id = descriptor.node_id();
            if !state.peer_store.is_routeable_now(&node_id, now) {
                return None;
            }
            let kem_public = descriptor.descriptor.x25519_kem_public()?;
            let public_endpoint = descriptor.descriptor.public_endpoint.clone()?;
            Some(OnionRelayCandidate {
                node_id: hex::encode(node_id),
                kem_alg: descriptor.descriptor.kem_alg,
                kem_public: hex::encode(kem_public),
                public_endpoint,
                capabilities: descriptor.descriptor.capabilities.clone(),
            })
        })
        .collect();

    Json(OnionCandidatesResponse {
        generated_at: now,
        count: candidates.len(),
        candidates,
        privacy_boundary: "fresh routeable signed node discovery metadata only (node id, KEM public key, public endpoint, capabilities); no client IPs, route ids, encrypted payloads, receiver identities, DNS contents, destinations, voucher secrets, private keys, or wallet-level traffic".to_string(),
    })
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
    use aeronyx_core::protocol::{NodeCapability, NodeCapacity, NodeDescriptor};
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
        assert_eq!(parsed.count, 1);
        assert_eq!(parsed.candidates.len(), 1);
        let candidate = &parsed.candidates[0];
        assert_eq!(candidate.node_id, want_node_id);
        assert_eq!(candidate.kem_alg, 1);
        assert_eq!(candidate.kem_public, hex::encode(kem));
        assert_eq!(candidate.public_endpoint, "relay.example:443");
        assert!(candidate.capabilities.contains(&NodeCapability::ChatRelay));
        assert!(parsed.privacy_boundary.contains("fresh routeable"));
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
