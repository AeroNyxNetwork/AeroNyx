// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store.rs
// ============================================================================
//! # Peer Store
//!
//! ## Creation Reason
//! Stores verified AeroNyx node descriptors in memory as the first foundation
//! for decentralized node discovery, encrypted message relay, and future
//! gossip synchronization.
//!
//! ## Main Functionality
//! - `PeerStore`: thread-safe map of node_id to `SignedNodeDescriptor`
//! - `upsert_verified()`: verifies signature/expiry before storing
//! - Sequence protection: older descriptors cannot overwrite newer ones
//! - Capability queries: find peers that advertise a required protocol role
//! - Expiry cleanup and monitoring snapshots
//! - Bootstrap snapshot loading with per-descriptor import reporting
//! - Discovery gossip message application and snapshot response generation
//! - Privacy-safe discovery audit events for rate-limit, policy, import, and
//!   snapshot export operations
//! - Bootstrap/cache/gossip runtime status for nodeboard diagnostics
//! - Seed endpoint recovery counters without exposing seed endpoint values
//! - Discovery stability summary for operator health gates and nodeboard
//! - Effective discovery recovery status so stale bootstrap-file warnings do
//!   not mask a later successful seed-gossip recovery path
//! - Commercial peer metadata summary: source, last_seen, TTL, capabilities,
//!   and health bucket for nodeboard capacity and stale-peer visibility
//! - Gossip scheduler visibility for jitter/backpressure diagnostics without
//!   exposing seed endpoint values or peer URLs
//! - Expired-peer cleanup counters so stale descriptor eviction is observable
//!   without exposing peer endpoints or user traffic metadata
//! - Health-ranked route candidates for blind relay preparation, using only
//!   node-level signed descriptor metadata and never encrypted payload content
//! - Blind relay runtime counters and drop reason buckets for nodeboard,
//!   without exposing encrypted payloads, peer endpoint URLs, or user metadata
//! - Per-peer node-to-node route health feedback so failed next hops are
//!   naturally deprioritized without exposing payloads or full peer endpoints
//! - Exclude-list route candidate selection so server internals can remove
//!   self or already-used hops before applying fanout/path limits
//!
//! ## Dependencies
//! - aeronyx-core/src/protocol/discovery.rs: descriptor and capability types
//! - parking_lot::RwLock: same locking style used by other server services
//! - std::collections::HashMap: small in-memory map for Phase 1
//!
//! ## Main Logical Flow
//! 1. Caller receives or builds a `SignedNodeDescriptor`
//! 2. Caller passes it to `upsert_verified(now)`
//! 3. Store verifies signature and descriptor validity window
//! 4. Store rejects stale sequence numbers for the same node
//! 5. Verified descriptors become available for future peer selection
//! 6. Bootstrap snapshots can hydrate the store without trusting unsigned data
//! 7. Gossip message handlers reuse the same verification and anti-rollback path
//!
//! ## Important Note for Next Developer
//! - This store keeps verified descriptors in memory, while optional peer-cache
//!   persistence and seed gossip provide restart recovery. Do not treat a
//!   currently healthy in-memory peer view as commercially resilient unless
//!   `PeerStoreStabilityStatus.restart_recovery_configured` is true.
//! - Do not store client-level traffic, wallet traffic, DNS contents, packet
//!   payloads, browsing history, voucher secrets, or private keys here.
//! - Do not use this as public-exit authorization. `allows_public_exit` stays
//!   false by default and must be governed by a separate reviewed policy.
//!
//! ## Last Modified
//! v0.18.0-RouteCandidateExclusion - Added exclude-before-limit route selection
//! v0.17.0-RouteHealthFeedback - Added per-peer blind relay success/failure scoring
//! v0.16.0-BlindRelayRuntimeStats - Added blind relay drop reason counters
//! v0.13.0-GossipBackpressureStatus - Added outbound gossip jitter/backpressure status
//! v0.14.0-ExpiredPeerCleanupStats - Added expired peer cleanup counters/audit
//! v0.15.0-RouteCandidateScoring - Added health-ranked peer route candidates
//! v0.12.0-CommercialPeerSummary - Added source/TTL/health/capability peer summary
//! v0.11.0-DiscoveryRecoveryStatus - Added effective recovery status for nodeboard
//! v0.10.0-DiscoveryRestartRecovery - Gate relay foundation on restart recovery
//! v0.9.0-DiscoveryStability - Added aggregate discovery stability summary
//! v0.8.0-DiscoveryGossipHealth - Added outbound gossip health summary fields
//! v0.7.0-DiscoverySeedStatus - Added privacy-safe seed endpoint recovery counters
//! v0.6.0-DiscoveryBootstrapStatus - Added bootstrap/cache/gossip status snapshot
//! v0.5.0-DiscoveryAuditLog - Added privacy-safe discovery audit ring buffer
//! v0.4.0-DiscoverySafetyStatus - Added capacity limit, runtime stats, status API support
//! v0.1.0-DiscoveryPhase1 - Initial verified in-memory peer store
//! v0.2.0-DiscoveryPhase2 - Added bootstrap snapshot import reporting
//! v0.3.0-DiscoveryPhase4 - Added discovery gossip apply/export helpers
// ============================================================================

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use aeronyx_core::protocol::discovery::{
    NodeBootstrapSnapshot, NodeCapability, NodeDiscoveryMessage, SignedNodeDescriptor,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

const DISCOVERY_GOSSIP_STALE_AFTER_SECS: u64 = 900;
const DISCOVERY_GOSSIP_FAILURE_ATTENTION_THRESHOLD: u64 = 3;
const PEER_DESCRIPTOR_STALE_WINDOW_SECS: u64 = 300;
const PEER_ROUTE_LAST_SEEN_FRESH_SECS: u64 = 300;
const PEER_ROUTE_LAST_SEEN_ACCEPTABLE_SECS: u64 = 900;
const PEER_ROUTE_LAST_SEEN_STALE_SECS: u64 = 1_800;
const PEER_ROUTE_RECENT_FAILURE_SECS: u64 = 600;
const PEER_ROUTE_STATUS_LIMIT: usize = 8;

// ============================================
// PeerStoreError
// ============================================

/// Errors returned by `PeerStore` operations.
#[derive(Debug, thiserror::Error)]
pub enum PeerStoreError {
    /// Descriptor failed signature, schema, or validity-window verification.
    #[error("descriptor verification failed")]
    VerificationFailed,
    /// Descriptor sequence is older than the descriptor already stored.
    #[error("stale descriptor sequence: current={current}, incoming={incoming}")]
    StaleSequence {
        /// Current stored sequence.
        current: u64,
        /// Incoming descriptor sequence.
        incoming: u64,
    },
    /// Store is at its configured maximum peer capacity.
    #[error("peer store capacity exceeded: max_peers={max_peers}")]
    CapacityExceeded {
        /// Configured maximum peer count.
        max_peers: usize,
    },
}

// ============================================
// PeerStoreSnapshot
// ============================================

/// Lightweight monitoring snapshot for dashboards and health checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreSnapshot {
    /// Total descriptors currently stored.
    pub total_peers: usize,
    /// Peers whose descriptors are valid at the snapshot time.
    pub valid_peers: usize,
    /// Peers advertising public discovery.
    pub public_peers: usize,
    /// Peers that allow public exit behavior.
    pub public_exit_peers: usize,
}

// ============================================
// PeerStoreRuntimeStats / PeerStoreStatus
// ============================================

/// Maximum number of discovery audit events retained in memory.
///
/// The audit log is intentionally bounded because this process may run on
/// small operator nodes. It is diagnostic evidence, not a durable ledger.
const MAX_AUDIT_EVENTS: usize = 64;

/// Privacy-safe discovery control-plane audit event.
///
/// This structure deliberately excludes client IPs, destinations, DNS
/// contents, packet payloads, chat plaintext, voucher secrets, private keys,
/// wallet-level traffic, and full peer public keys. It records only aggregate
/// discovery control-plane decisions needed by operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreAuditEvent {
    /// Unix timestamp when the event was recorded.
    pub at: u64,
    /// Short machine-readable action name.
    pub action: String,
    /// Outcome bucket such as `accepted`, `rejected`, or `limited`.
    pub outcome: String,
    /// Human-readable aggregate detail with counts or policy scope only.
    pub detail: String,
}

/// Runtime status for discovery bootstrap, peer-cache persistence, and gossip.
///
/// All fields are aggregate control-plane state. They must not contain client
/// identifiers, traffic metadata, DNS contents, packet payloads, chat
/// plaintext, voucher secrets, private keys, or wallet-level traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreBootstrapStatus {
    /// Whether discovery bootstrap is enabled in local config.
    pub enabled: bool,
    /// Whether peer cache persistence is configured.
    pub peer_cache_configured: bool,
    /// Whether outbound gossip is enabled in local config.
    pub gossip_enabled: bool,
    /// Number of configured discovery seed endpoints.
    ///
    /// The endpoint values themselves are intentionally omitted from status and
    /// heartbeat payloads. Operators only need this aggregate to diagnose
    /// whether seed recovery is configured.
    pub seed_endpoints_configured: u64,
    /// Last bootstrap/cache source kind observed.
    pub last_source_kind: Option<String>,
    /// Last bootstrap/cache source status: success, failed, missing, skipped.
    pub last_source_status: Option<String>,
    /// Import report detail for the last bootstrap/cache source.
    pub last_source_detail: Option<String>,
    /// Timestamp of the last bootstrap/cache source event.
    pub last_source_at: Option<u64>,
    /// Effective discovery recovery status after combining source load,
    /// self-descriptor, and successful seed/peer gossip evidence.
    ///
    /// This deliberately complements `last_source_status` instead of
    /// replacing it: operators can still see that a static bootstrap file was
    /// stale, while nodeboard can show that discovery recovered via live
    /// gossip without exposing peer URLs or descriptors.
    pub recovery_status: Option<String>,
    /// Privacy-safe aggregate detail for `recovery_status`.
    pub recovery_detail: Option<String>,
    /// Timestamp of the last effective recovery evidence.
    pub recovery_at: Option<u64>,
    /// Self descriptor registration status.
    pub self_descriptor_status: Option<String>,
    /// Timestamp of the last self descriptor event.
    pub self_descriptor_at: Option<u64>,
    /// Last peer-cache save status.
    pub last_cache_save_status: Option<String>,
    /// Last peer-cache save detail.
    pub last_cache_save_detail: Option<String>,
    /// Timestamp of the last peer-cache save attempt.
    pub last_cache_save_at: Option<u64>,
    /// Number of peers attempted in the last outbound gossip round.
    pub last_gossip_attempted: u64,
    /// Number of configured seed endpoints attempted in the last gossip round.
    pub last_gossip_seed_attempted: u64,
    /// Number of peers successfully contacted in the last outbound gossip round.
    pub last_gossip_succeeded: u64,
    /// Number of peers that failed in the last outbound gossip round.
    pub last_gossip_failed: u64,
    /// Health bucket for the last outbound gossip round: healthy, degraded, failed, idle.
    pub last_gossip_status: Option<String>,
    /// Stable privacy-safe reason bucket for the last outbound gossip failure.
    pub last_gossip_failure_reason: Option<String>,
    /// Consecutive outbound gossip rounds with zero successful peer contacts.
    pub consecutive_gossip_failures: u64,
    /// Timestamp of the last outbound gossip round with at least one success.
    pub last_gossip_success_at: Option<u64>,
    /// Timestamp of the last outbound gossip round.
    pub last_gossip_round_at: Option<u64>,
    /// Whether outbound gossip is currently reducing fanout after failures.
    pub gossip_backpressure_active: bool,
    /// Delay planned before the next outbound gossip round.
    pub next_gossip_delay_seconds: Option<u64>,
    /// Jitter offset applied to the next gossip delay. May be negative.
    pub next_gossip_jitter_seconds: i64,
    /// Timestamp when the current gossip schedule was calculated.
    pub last_gossip_schedule_at: Option<u64>,
}

impl Default for PeerStoreBootstrapStatus {
    fn default() -> Self {
        Self {
            enabled: false,
            peer_cache_configured: false,
            gossip_enabled: false,
            seed_endpoints_configured: 0,
            last_source_kind: None,
            last_source_status: None,
            last_source_detail: None,
            last_source_at: None,
            recovery_status: None,
            recovery_detail: None,
            recovery_at: None,
            self_descriptor_status: None,
            self_descriptor_at: None,
            last_cache_save_status: None,
            last_cache_save_detail: None,
            last_cache_save_at: None,
            last_gossip_attempted: 0,
            last_gossip_seed_attempted: 0,
            last_gossip_succeeded: 0,
            last_gossip_failed: 0,
            last_gossip_status: None,
            last_gossip_failure_reason: None,
            consecutive_gossip_failures: 0,
            last_gossip_success_at: None,
            last_gossip_round_at: None,
            gossip_backpressure_active: false,
            next_gossip_delay_seconds: None,
            next_gossip_jitter_seconds: 0,
            last_gossip_schedule_at: None,
        }
    }
}

/// Operator-facing stability summary derived from verified PeerStore state.
///
/// This is deliberately a compact aggregate contract. It never includes peer
/// URLs, full peer public keys, client IPs, destinations, DNS contents, packet
/// payloads, chat plaintext, voucher secrets, private keys, wallet-level
/// traffic, or per-user traffic. The goal is to let nodeboard and backend
/// health checks decide whether discovery is ready for future blind relay work
/// without reconstructing policy from raw counters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreStabilityStatus {
    /// Stable health bucket: disabled, pending, healthy, degraded, stale, failed.
    pub health: String,
    /// Whether this node has enough fresh aggregate discovery state to be used
    /// as a foundation for later multi-hop / blind relay development.
    pub relay_foundation_ready: bool,
    /// Privacy-safe operator-facing detail.
    pub detail: String,
    /// Privacy-safe next action for nodeboard / AI runbooks.
    pub next_action: String,
    /// Age of the last successful outbound gossip round, when known.
    pub last_gossip_success_age_seconds: Option<u64>,
    /// Age of the last outbound gossip round, when known.
    pub last_gossip_round_age_seconds: Option<u64>,
    /// Whether discovery seed recovery is configured.
    pub seed_recovery_configured: bool,
    /// Configured stale window for gossip freshness checks.
    pub stale_after_seconds: u64,
    /// Whether this node has at least one configured recovery path after restart.
    pub restart_recovery_configured: bool,
}

/// Cumulative runtime counters for nodeboard and operator diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRuntimeStats {
    /// Total descriptors seen through verified import paths.
    pub total_imported: u64,
    /// Total descriptors inserted or upgraded.
    pub inserted: u64,
    /// Total descriptors ignored because sequence was unchanged.
    pub unchanged: u64,
    /// Total descriptors rejected because they were stale.
    pub stale: u64,
    /// Total descriptors rejected because verification or expiry failed.
    pub rejected: u64,
    /// Total descriptors rejected because max_peers was reached.
    pub capacity_rejected: u64,
    /// Total inbound messages rejected by allow/deny policy.
    pub policy_rejected: u64,
    /// Total inbound gossip requests rejected by rate limiting.
    pub rate_limited: u64,
    /// Total expired descriptors removed by cleanup.
    pub expired_removed: u64,
    /// Opaque node-to-node blind relay counters and drop reason buckets.
    pub blind_relay: PeerStoreBlindRelayStats,
    /// Unix timestamp of the last descriptor import attempt.
    pub last_import_at: Option<u64>,
    /// Unix timestamp of the last gossip exchange observed by this node.
    pub last_gossip_at: Option<u64>,
    /// Unix timestamp of the last exported snapshot.
    pub last_snapshot_at: Option<u64>,
    /// Unix timestamp of the last cleanup that removed at least one expired peer.
    pub last_cleanup_at: Option<u64>,
}

/// Opaque blind relay counters exposed to nodeboard.
///
/// These counters are deliberately coarse. They never include route ids,
/// previous-hop ids, full next-hop ids, peer endpoint URLs, encrypted blobs,
/// message bodies, client IPs, DNS contents, destinations, voucher secrets, or
/// wallet-level traffic. They exist to prove the relay is healthy and to make
/// pressure/drop reasons actionable for operators.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreBlindRelayStats {
    /// Total blind relay HTTP requests that reached routing logic.
    pub received: u64,
    /// Requests where this node was the requested next hop.
    pub terminal: u64,
    /// Requests forwarded to another verified node descriptor.
    pub forwarded: u64,
    /// Requests rejected before terminal handling or next-hop forwarding.
    pub rejected: u64,
    /// Requests rejected because the endpoint was under local backpressure.
    pub backpressure_dropped: u64,
    /// Requests rejected because the previous-hop key or signature was invalid.
    pub invalid_signature: u64,
    /// Requests rejected because the signed envelope exceeded size limits.
    pub envelope_too_large: u64,
    /// Requests rejected because TTL was already exhausted.
    pub ttl_exhausted: u64,
    /// Requests rejected because `next_hop` was not in verified PeerStore.
    pub no_route: u64,
    /// Requests rejected because the next hop had no usable public endpoint.
    pub invalid_endpoint: u64,
    /// Requests rejected because forwarding to next hop failed.
    pub forward_failed: u64,
    /// Unix timestamp of the last blind relay event.
    pub last_event_at: Option<u64>,
}

/// Combined peer store status payload for nodeboard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreStatus {
    /// Point-in-time peer counts.
    pub snapshot: PeerStoreSnapshot,
    /// Cumulative runtime counters.
    pub runtime: PeerStoreRuntimeStats,
    /// Configured maximum peer count.
    pub max_peers: Option<usize>,
    /// Recent privacy-safe discovery control-plane audit events.
    pub recent_audit_events: Vec<PeerStoreAuditEvent>,
    /// Bootstrap/cache/gossip runtime status.
    pub bootstrap: PeerStoreBootstrapStatus,
    /// Aggregate discovery stability summary for health gates and nodeboard.
    pub stability: PeerStoreStabilityStatus,
    /// Commercial peer summary for nodeboard capacity and stale-peer panels.
    ///
    /// This contains only node-level signed descriptor metadata and aggregate
    /// source counters. It must never include client IPs, payloads, DNS,
    /// destinations, wallet-level traffic, or chat/message content.
    pub peer_summary: PeerStorePeerSummaryStatus,
    /// Privacy-safe health-ranked route candidates for future blind relay paths.
    ///
    /// Candidate rows intentionally omit full peer ids and public endpoints.
    /// Server internals can use `route_candidates_with_capability()` when they
    /// need signed descriptors for actual node-to-node transport.
    pub route_candidates: PeerStoreRouteCandidateStatus,
}

// ============================================
// Commercial peer metadata summary
// ============================================

#[derive(Debug, Clone)]
struct PeerRuntimeMetadata {
    source: String,
    first_seen_at: u64,
    last_seen_at: u64,
    last_sequence: u64,
    imported_count: u64,
}

#[derive(Debug, Clone, Default)]
struct PeerRouteHealth {
    success_count: u64,
    failure_count: u64,
    consecutive_failures: u64,
    last_success_at: Option<u64>,
    last_failure_at: Option<u64>,
    last_failure_reason: Option<String>,
}

/// Privacy-safe per-peer nodeboard row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStorePeerSummary {
    /// Short node id prefix for operator debugging without exposing full keys.
    pub node_id_prefix: String,
    /// Last import source bucket: self, file, url, cache, cache_backup,
    /// gossip_snapshot, gossip_announce, or unknown.
    pub source: String,
    /// Last descriptor sequence seen for this node.
    pub sequence: u64,
    /// Number of accepted imports/upgrades observed for this peer in this process.
    pub imported_count: u64,
    /// Unix timestamp when this peer was first seen by this process.
    pub first_seen_at: u64,
    /// Unix timestamp when this peer was most recently observed.
    pub last_seen_at: u64,
    /// Age of the last observation at status generation time.
    pub last_seen_age_seconds: u64,
    /// Descriptor expiry timestamp.
    pub expires_at: u64,
    /// Remaining descriptor TTL in seconds, if still valid.
    pub ttl_remaining_seconds: Option<u64>,
    /// Stable health bucket: healthy, stale, expired.
    pub health: String,
    /// Public capability labels advertised by the signed descriptor.
    pub capabilities: Vec<String>,
    /// Whether the peer advertises a public discovery endpoint.
    pub endpoint_advertised: bool,
    /// Whether the peer is included in public discovery snapshots.
    pub public_discovery: bool,
    /// Optional region hint from the signed descriptor policy.
    pub region: Option<String>,
}

/// Aggregate peer summary attached to heartbeat/nodeboard discovery status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStorePeerSummaryStatus {
    /// Number of descriptors currently retained by PeerStore.
    pub total_peers: usize,
    /// Number of descriptors that verify at status generation time.
    pub valid_peers: usize,
    /// Valid peers with descriptor TTL comfortably above the stale window.
    pub healthy_peers: usize,
    /// Valid peers whose descriptor TTL is close to expiry.
    pub stale_peers: usize,
    /// Retained descriptors that no longer verify at status generation time.
    pub expired_peers: usize,
    /// Number of peers advertising privacy relay capability.
    pub privacy_relay_peers: usize,
    /// Number of peers advertising encrypted chat relay capability.
    pub chat_relay_peers: usize,
    /// Number of peers advertising encrypted storage capability.
    pub encrypted_storage_peers: usize,
    /// Number of peers advertising agent relay capability.
    pub agent_relay_peers: usize,
    /// Number of peers advertising future onion middle-hop capability.
    pub onion_middle_peers: usize,
    /// Counts by coarse source bucket, never raw peer URLs.
    pub source_counts: BTreeMap<String, usize>,
    /// Privacy-safe per-peer rows for operator UI.
    pub peers: Vec<PeerStorePeerSummary>,
}

// ============================================
// Route candidate summary
// ============================================

/// Privacy-safe route candidate row for nodeboard and health reporting.
///
/// The score is derived from signed node descriptor metadata plus local
/// discovery observation age. It never uses chat plaintext, encrypted blob
/// contents, packet payloads, DNS data, client public IPs, voucher secrets,
/// private keys, or wallet-level traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteCandidate {
    /// Short node id prefix for operator debugging without exposing full keys.
    pub node_id_prefix: String,
    /// Capability requested for this route list.
    pub capability: String,
    /// Stable score bucket used for sorting candidates.
    pub score: i64,
    /// Last import source bucket such as cache, gossip_snapshot, or gossip_announce.
    pub source: String,
    /// Stable health bucket: healthy or stale.
    pub health: String,
    /// Local route health bucket from recent blind relay forward attempts.
    ///
    /// This is node-level control-plane feedback only. It never includes route
    /// ids, endpoint URLs, encrypted blobs, receivers, client IPs, or content.
    pub route_health: String,
    /// Recent route failure count used to penalize unstable next hops.
    pub route_failure_count: u64,
    /// Consecutive route failures since the last successful forward.
    pub route_consecutive_failures: u64,
    /// Last successful node-to-node relay forward to this candidate.
    pub last_route_success_at: Option<u64>,
    /// Last failed node-to-node relay forward to this candidate.
    pub last_route_failure_at: Option<u64>,
    /// Coarse failure reason bucket from the last failed forward.
    pub last_route_failure_reason: Option<String>,
    /// Age of the last observation at status generation time.
    pub last_seen_age_seconds: u64,
    /// Remaining descriptor TTL in seconds.
    pub ttl_remaining_seconds: Option<u64>,
    /// Whether a public node-to-node endpoint exists.
    pub endpoint_advertised: bool,
    /// Whether this descriptor is public-discovery visible.
    pub public_discovery: bool,
    /// Optional region hint from the signed descriptor policy.
    pub region: Option<String>,
    /// Coarse max session capacity advertised by the peer.
    pub max_sessions: u32,
    /// Optional bandwidth policy advertised by the peer.
    pub max_bps: Option<u64>,
    /// Optional packet-rate policy advertised by the peer.
    pub max_pps: Option<u64>,
}

/// Health-ranked candidate lists used by nodeboard before blind relay rollout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteCandidateStatus {
    /// Unix timestamp when the candidate lists were generated.
    pub generated_at: u64,
    /// Candidates for privacy protocol packet relay.
    pub privacy_relay: Vec<PeerStoreRouteCandidate>,
    /// Candidates for E2E encrypted chat envelope relay.
    pub chat_relay: Vec<PeerStoreRouteCandidate>,
    /// Candidates for future no-exit onion middle-hop relay.
    pub onion_middle: Vec<PeerStoreRouteCandidate>,
}

#[derive(Debug, Clone)]
struct ScoredPeerRouteCandidate {
    descriptor: SignedNodeDescriptor,
    summary: PeerStoreRouteCandidate,
}

struct PeerStoreCounters {
    total_imported: AtomicU64,
    inserted: AtomicU64,
    unchanged: AtomicU64,
    stale: AtomicU64,
    rejected: AtomicU64,
    capacity_rejected: AtomicU64,
    policy_rejected: AtomicU64,
    rate_limited: AtomicU64,
    expired_removed: AtomicU64,
    blind_relay_received: AtomicU64,
    blind_relay_terminal: AtomicU64,
    blind_relay_forwarded: AtomicU64,
    blind_relay_rejected: AtomicU64,
    blind_relay_backpressure_dropped: AtomicU64,
    blind_relay_invalid_signature: AtomicU64,
    blind_relay_envelope_too_large: AtomicU64,
    blind_relay_ttl_exhausted: AtomicU64,
    blind_relay_no_route: AtomicU64,
    blind_relay_invalid_endpoint: AtomicU64,
    blind_relay_forward_failed: AtomicU64,
    last_import_at: AtomicU64,
    last_gossip_at: AtomicU64,
    last_snapshot_at: AtomicU64,
    last_cleanup_at: AtomicU64,
    last_blind_relay_at: AtomicU64,
}

impl PeerStoreCounters {
    fn new() -> Self {
        Self {
            total_imported: AtomicU64::new(0),
            inserted: AtomicU64::new(0),
            unchanged: AtomicU64::new(0),
            stale: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
            capacity_rejected: AtomicU64::new(0),
            policy_rejected: AtomicU64::new(0),
            rate_limited: AtomicU64::new(0),
            expired_removed: AtomicU64::new(0),
            blind_relay_received: AtomicU64::new(0),
            blind_relay_terminal: AtomicU64::new(0),
            blind_relay_forwarded: AtomicU64::new(0),
            blind_relay_rejected: AtomicU64::new(0),
            blind_relay_backpressure_dropped: AtomicU64::new(0),
            blind_relay_invalid_signature: AtomicU64::new(0),
            blind_relay_envelope_too_large: AtomicU64::new(0),
            blind_relay_ttl_exhausted: AtomicU64::new(0),
            blind_relay_no_route: AtomicU64::new(0),
            blind_relay_invalid_endpoint: AtomicU64::new(0),
            blind_relay_forward_failed: AtomicU64::new(0),
            last_import_at: AtomicU64::new(0),
            last_gossip_at: AtomicU64::new(0),
            last_snapshot_at: AtomicU64::new(0),
            last_cleanup_at: AtomicU64::new(0),
            last_blind_relay_at: AtomicU64::new(0),
        }
    }

    fn optional_ts(value: u64) -> Option<u64> {
        (value > 0).then_some(value)
    }

    fn snapshot(&self) -> PeerStoreRuntimeStats {
        PeerStoreRuntimeStats {
            total_imported: self.total_imported.load(Ordering::Relaxed),
            inserted: self.inserted.load(Ordering::Relaxed),
            unchanged: self.unchanged.load(Ordering::Relaxed),
            stale: self.stale.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            capacity_rejected: self.capacity_rejected.load(Ordering::Relaxed),
            policy_rejected: self.policy_rejected.load(Ordering::Relaxed),
            rate_limited: self.rate_limited.load(Ordering::Relaxed),
            expired_removed: self.expired_removed.load(Ordering::Relaxed),
            blind_relay: PeerStoreBlindRelayStats {
                received: self.blind_relay_received.load(Ordering::Relaxed),
                terminal: self.blind_relay_terminal.load(Ordering::Relaxed),
                forwarded: self.blind_relay_forwarded.load(Ordering::Relaxed),
                rejected: self.blind_relay_rejected.load(Ordering::Relaxed),
                backpressure_dropped: self
                    .blind_relay_backpressure_dropped
                    .load(Ordering::Relaxed),
                invalid_signature: self.blind_relay_invalid_signature.load(Ordering::Relaxed),
                envelope_too_large: self.blind_relay_envelope_too_large.load(Ordering::Relaxed),
                ttl_exhausted: self.blind_relay_ttl_exhausted.load(Ordering::Relaxed),
                no_route: self.blind_relay_no_route.load(Ordering::Relaxed),
                invalid_endpoint: self.blind_relay_invalid_endpoint.load(Ordering::Relaxed),
                forward_failed: self.blind_relay_forward_failed.load(Ordering::Relaxed),
                last_event_at: Self::optional_ts(self.last_blind_relay_at.load(Ordering::Relaxed)),
            },
            last_import_at: Self::optional_ts(self.last_import_at.load(Ordering::Relaxed)),
            last_gossip_at: Self::optional_ts(self.last_gossip_at.load(Ordering::Relaxed)),
            last_snapshot_at: Self::optional_ts(self.last_snapshot_at.load(Ordering::Relaxed)),
            last_cleanup_at: Self::optional_ts(self.last_cleanup_at.load(Ordering::Relaxed)),
        }
    }
}

// ============================================
// PeerStoreImportReport
// ============================================

/// Result summary for bootstrap snapshot imports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreImportReport {
    /// Number of descriptors present in the snapshot.
    pub total: usize,
    /// Number of descriptors inserted or upgraded.
    pub inserted: usize,
    /// Number of descriptors already present with the same sequence.
    pub unchanged: usize,
    /// Number of descriptors rejected because they were older than stored data.
    pub stale: usize,
    /// Number of descriptors rejected because verification or expiry failed.
    pub rejected: usize,
}

impl PeerStoreImportReport {
    /// Empty import report.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            total: 0,
            inserted: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        }
    }

    /// Returns true when at least one peer was inserted or upgraded.
    #[must_use]
    pub const fn changed(&self) -> bool {
        self.inserted > 0
    }
}

// ============================================
// PeerStore
// ============================================

/// In-memory verified descriptor store for known AeroNyx nodes.
pub struct PeerStore {
    peers: RwLock<HashMap<[u8; 32], SignedNodeDescriptor>>,
    peer_runtime: RwLock<HashMap<[u8; 32], PeerRuntimeMetadata>>,
    route_health: RwLock<HashMap<[u8; 32], PeerRouteHealth>>,
    max_peers: RwLock<Option<usize>>,
    counters: PeerStoreCounters,
    audit_events: RwLock<VecDeque<PeerStoreAuditEvent>>,
    bootstrap_status: RwLock<PeerStoreBootstrapStatus>,
}

impl PeerStore {
    /// Creates an empty peer store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            peer_runtime: RwLock::new(HashMap::new()),
            route_health: RwLock::new(HashMap::new()),
            max_peers: RwLock::new(None),
            counters: PeerStoreCounters::new(),
            audit_events: RwLock::new(VecDeque::with_capacity(MAX_AUDIT_EVENTS)),
            bootstrap_status: RwLock::new(PeerStoreBootstrapStatus::default()),
        }
    }

    /// Creates an empty peer store with a maximum descriptor capacity.
    #[must_use]
    pub fn with_max_peers(max_peers: usize) -> Self {
        let store = Self::new();
        store.set_max_peers(Some(max_peers));
        store
    }

    /// Updates the maximum peer capacity.
    pub fn set_max_peers(&self, max_peers: Option<usize>) {
        *self.max_peers.write() = max_peers;
    }

    /// Returns the configured maximum peer capacity.
    #[must_use]
    pub fn max_peers(&self) -> Option<usize> {
        *self.max_peers.read()
    }

    /// Records discovery bootstrap feature flags from local config.
    pub fn configure_bootstrap_status(
        &self,
        enabled: bool,
        peer_cache_configured: bool,
        gossip_enabled: bool,
        seed_endpoints_configured: usize,
    ) {
        let mut status = self.bootstrap_status.write();
        status.enabled = enabled;
        status.peer_cache_configured = peer_cache_configured;
        status.gossip_enabled = gossip_enabled;
        status.seed_endpoints_configured = seed_endpoints_configured as u64;
    }

    /// Records a bootstrap or peer-cache load source result.
    pub fn record_bootstrap_source(
        &self,
        now: u64,
        source_kind: impl Into<String>,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_kind = source_kind.into();
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.last_source_kind = Some(source_kind.clone());
            status.last_source_status = Some(source_status.clone());
            status.last_source_detail = Some(detail.clone());
            status.last_source_at = Some(now);
            status.recovery_status = Some(source_status.clone());
            status.recovery_detail = Some(format!("source_kind={source_kind} {detail}"));
            status.recovery_at = Some(now);
        }
        self.record_audit_event(
            now,
            "bootstrap_source",
            source_status,
            format!("kind={source_kind} {detail}"),
        );
    }

    /// Records self descriptor registration status.
    pub fn record_self_descriptor_status(
        &self,
        now: u64,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.self_descriptor_status = Some(source_status.clone());
            status.self_descriptor_at = Some(now);
        }
        self.record_audit_event(now, "self_descriptor", source_status, detail);
    }

    /// Records peer-cache save status.
    pub fn record_cache_save_status(
        &self,
        now: u64,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.last_cache_save_status = Some(source_status.clone());
            status.last_cache_save_detail = Some(detail.clone());
            status.last_cache_save_at = Some(now);
        }
        self.record_audit_event(now, "peer_cache_save", source_status, detail);
    }

    /// Records outbound gossip round result.
    pub fn record_gossip_round(
        &self,
        now: u64,
        attempted: usize,
        succeeded: usize,
        seed_attempted: usize,
        failure_reason: Option<String>,
    ) {
        let failed = attempted.saturating_sub(succeeded);
        let status_bucket = if attempted == 0 && failure_reason.is_some() {
            "failed"
        } else if attempted == 0 {
            "idle"
        } else if succeeded == attempted {
            "healthy"
        } else if succeeded > 0 {
            "degraded"
        } else {
            "failed"
        };
        let failure_reason = if failed > 0 || (attempted == 0 && status_bucket == "failed") {
            Some(failure_reason.unwrap_or_else(|| "unknown".to_string()))
        } else {
            None
        };
        let consecutive_gossip_failures;
        {
            let mut status = self.bootstrap_status.write();
            if succeeded > 0 {
                status.consecutive_gossip_failures = 0;
                status.last_gossip_success_at = Some(now);
                status.recovery_status = Some("success".to_string());
                status.recovery_detail = Some(format!(
                    "gossip_recovered attempted={attempted} succeeded={succeeded} seed_attempted={seed_attempted}"
                ));
                status.recovery_at = Some(now);
            } else if attempted > 0 || failure_reason.is_some() {
                status.consecutive_gossip_failures =
                    status.consecutive_gossip_failures.saturating_add(1);
            }
            consecutive_gossip_failures = status.consecutive_gossip_failures;
            status.last_gossip_attempted = attempted as u64;
            status.last_gossip_seed_attempted = seed_attempted as u64;
            status.last_gossip_succeeded = succeeded as u64;
            status.last_gossip_failed = failed as u64;
            status.last_gossip_status = Some(status_bucket.to_string());
            status.last_gossip_failure_reason = failure_reason.clone();
            status.last_gossip_round_at = Some(now);
        }
        let outcome = match status_bucket {
            "healthy" => "accepted",
            "idle" => "ignored",
            _ => "warning",
        };
        let reason_detail = failure_reason
            .as_deref()
            .map(|reason| format!(" failure_reason={reason}"))
            .unwrap_or_default();
        self.record_audit_event(
            now,
            "outbound_gossip_round",
            outcome,
            format!(
                "attempted={attempted} succeeded={succeeded} failed={failed} seed_attempted={seed_attempted} status={status_bucket} consecutive_failures={consecutive_gossip_failures}{reason_detail}"
            ),
        );
    }

    /// Records the next outbound gossip schedule.
    ///
    /// This is intentionally aggregate scheduler state only. It never stores
    /// peer URLs, seed URLs, client identifiers, ciphertext, plaintext, packet
    /// payloads, voucher secrets, private keys, or wallet-level traffic.
    pub fn record_gossip_schedule(
        &self,
        now: u64,
        backpressure_active: bool,
        next_delay_seconds: u64,
        jitter_seconds: i64,
    ) {
        {
            let mut status = self.bootstrap_status.write();
            status.gossip_backpressure_active = backpressure_active;
            status.next_gossip_delay_seconds = Some(next_delay_seconds);
            status.next_gossip_jitter_seconds = jitter_seconds;
            status.last_gossip_schedule_at = Some(now);
        }

        if backpressure_active {
            self.record_audit_event(
                now,
                "outbound_gossip_backpressure",
                "limited",
                format!("next_delay_seconds={next_delay_seconds} jitter_seconds={jitter_seconds}"),
            );
        }
    }

    /// Returns the current consecutive outbound gossip failure count.
    ///
    /// The scheduler uses this aggregate counter to decide whether to reduce
    /// fanout. It is exposed as a method so task code does not need to inspect
    /// the full status snapshot or any peer-level state.
    pub fn consecutive_gossip_failures(&self) -> u64 {
        self.bootstrap_status.read().consecutive_gossip_failures
    }

    /// Verifies and inserts or refreshes a descriptor.
    ///
    /// Returns `Ok(true)` when the store changed, `Ok(false)` when the same
    /// sequence was already present, and an error when the descriptor is
    /// invalid or would roll the node backward to an older sequence.
    pub fn upsert_verified(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
    ) -> Result<bool, PeerStoreError> {
        self.upsert_verified_from_source(descriptor, now, "unknown")
    }

    /// Verifies and inserts or refreshes a descriptor with a source bucket.
    ///
    /// Source buckets are intentionally coarse (`cache`, `gossip_snapshot`,
    /// `self`, etc.). They let nodeboard distinguish restart recovery from
    /// live discovery without exposing peer URLs or transport metadata.
    pub fn upsert_verified_from_source(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: impl Into<String>,
    ) -> Result<bool, PeerStoreError> {
        descriptor
            .verify_at(now)
            .map_err(|_| PeerStoreError::VerificationFailed)?;

        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
        let source = source.into();
        let mut peers = self.peers.write();

        let is_existing_peer = if let Some(existing) = peers.get(&node_id) {
            let current = existing.sequence();
            if incoming_sequence < current {
                return Err(PeerStoreError::StaleSequence {
                    current,
                    incoming: incoming_sequence,
                });
            }
            if incoming_sequence == current {
                drop(peers);
                self.record_peer_runtime(&descriptor, now, source, false);
                return Ok(false);
            }
            true
        } else {
            false
        };

        if let Some(max_peers) = *self.max_peers.read() {
            if !is_existing_peer && peers.len() >= max_peers {
                self.counters
                    .capacity_rejected
                    .fetch_add(1, Ordering::Relaxed);
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }

        peers.insert(node_id, descriptor.clone());
        drop(peers);
        self.record_peer_runtime(&descriptor, now, source, true);
        Ok(true)
    }

    fn record_peer_runtime(
        &self,
        descriptor: &SignedNodeDescriptor,
        now: u64,
        source: String,
        inserted_or_upgraded: bool,
    ) {
        let node_id = descriptor.node_id();
        let mut metadata = self.peer_runtime.write();
        let entry = metadata
            .entry(node_id)
            .or_insert_with(|| PeerRuntimeMetadata {
                source: source.clone(),
                first_seen_at: now,
                last_seen_at: now,
                last_sequence: descriptor.sequence(),
                imported_count: 0,
            });
        entry.source = source;
        entry.last_seen_at = now;
        entry.last_sequence = descriptor.sequence();
        if inserted_or_upgraded {
            entry.imported_count = entry.imported_count.saturating_add(1);
        }
    }

    /// Imports all descriptors from a validated bootstrap snapshot.
    ///
    /// Invalid descriptors are counted and skipped. This lets a node keep using
    /// the healthy part of a bootstrap snapshot while surfacing corruption or
    /// expiry through the returned report.
    pub fn load_bootstrap_snapshot(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
    ) -> PeerStoreImportReport {
        self.load_bootstrap_snapshot_from_source(snapshot, now, "unknown")
    }

    /// Imports descriptors from a snapshot and tags runtime metadata source.
    pub fn load_bootstrap_snapshot_from_source(
        &self,
        snapshot: &NodeBootstrapSnapshot,
        now: u64,
        source: impl Into<String>,
    ) -> PeerStoreImportReport {
        let source = source.into();
        let mut report = PeerStoreImportReport {
            total: snapshot.peers.len(),
            inserted: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };

        for descriptor in &snapshot.peers {
            match self.upsert_verified_from_source(descriptor.clone(), now, source.clone()) {
                Ok(true) => report.inserted += 1,
                Ok(false) => report.unchanged += 1,
                Err(PeerStoreError::StaleSequence { .. }) => report.stale += 1,
                Err(PeerStoreError::VerificationFailed)
                | Err(PeerStoreError::CapacityExceeded { .. }) => report.rejected += 1,
            }
        }

        self.record_import_report(&report, now);
        report
    }

    /// Applies a discovery gossip message to this store.
    ///
    /// Snapshot requests are read-only and return an empty report; callers can
    /// use `build_snapshot_response()` to construct the actual response.
    pub fn apply_discovery_message(
        &self,
        message: &NodeDiscoveryMessage,
        now: u64,
    ) -> PeerStoreImportReport {
        match message {
            NodeDiscoveryMessage::SnapshotRequest { .. } => PeerStoreImportReport::empty(),
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => {
                self.load_bootstrap_snapshot_from_source(snapshot, now, "gossip_snapshot")
            }
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => {
                let mut report = PeerStoreImportReport {
                    total: 1,
                    inserted: 0,
                    unchanged: 0,
                    stale: 0,
                    rejected: 0,
                };
                match self.upsert_verified_from_source(descriptor.clone(), now, "gossip_announce") {
                    Ok(true) => report.inserted = 1,
                    Ok(false) => report.unchanged = 1,
                    Err(PeerStoreError::StaleSequence { .. }) => report.stale = 1,
                    Err(PeerStoreError::VerificationFailed)
                    | Err(PeerStoreError::CapacityExceeded { .. }) => report.rejected = 1,
                }
                self.record_import_report(&report, now);
                report
            }
        }
    }

    fn record_import_report(&self, report: &PeerStoreImportReport, now: u64) {
        if report.total == 0 {
            return;
        }

        self.counters
            .total_imported
            .fetch_add(report.total as u64, Ordering::Relaxed);
        self.counters
            .inserted
            .fetch_add(report.inserted as u64, Ordering::Relaxed);
        self.counters
            .unchanged
            .fetch_add(report.unchanged as u64, Ordering::Relaxed);
        self.counters
            .stale
            .fetch_add(report.stale as u64, Ordering::Relaxed);
        self.counters
            .rejected
            .fetch_add(report.rejected as u64, Ordering::Relaxed);
        self.counters.last_import_at.store(now, Ordering::Relaxed);

        let outcome = if report.rejected > 0 || report.stale > 0 {
            "warning"
        } else if report.inserted > 0 || report.unchanged > 0 {
            "accepted"
        } else {
            "ignored"
        };
        self.record_audit_event(
            now,
            "descriptor_import",
            outcome,
            format!(
                "total={} inserted={} unchanged={} stale={} rejected={}",
                report.total, report.inserted, report.unchanged, report.stale, report.rejected
            ),
        );
    }

    /// Records a discovery gossip exchange timestamp.
    pub fn mark_gossip_at(&self, now: u64) {
        self.counters.last_gossip_at.store(now, Ordering::Relaxed);
    }

    /// Records an allow/deny policy rejection.
    pub fn record_policy_rejected(&self, now: u64, detail: impl Into<String>) {
        self.counters
            .policy_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.record_audit_event(now, "gossip_policy_rejected", "rejected", detail);
    }

    /// Records a rate-limited inbound request.
    pub fn record_rate_limited(&self, now: u64, detail: impl Into<String>) {
        self.counters.rate_limited.fetch_add(1, Ordering::Relaxed);
        self.record_audit_event(now, "gossip_rate_limited", "limited", detail);
    }

    /// Records a blind relay request that terminates at this node.
    ///
    /// Only aggregate routing facts are recorded. The route id, previous-hop
    /// id, full next-hop id, endpoint URL, encrypted blob bytes, and any
    /// payload-derived metadata remain outside PeerStore status.
    pub fn record_blind_relay_terminal(&self, now: u64, ttl_remaining: u8, blob_bytes: usize) {
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_terminal
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_terminal",
            "accepted",
            format!("ttl_remaining={ttl_remaining} encrypted_blob_bytes={blob_bytes}"),
        );
    }

    /// Records a blind relay request forwarded to the next verified node.
    pub fn record_blind_relay_forwarded(&self, now: u64, ttl_remaining: u8) {
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_forwarded
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_forward",
            "accepted",
            format!("ttl_remaining={ttl_remaining} encrypted_blob_bytes=opaque"),
        );
    }

    /// Records a blind relay rejection with a stable privacy-safe reason.
    pub fn record_blind_relay_rejected(&self, now: u64, reason: impl AsRef<str>) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);

        match reason {
            "backpressure" => {
                self.counters
                    .blind_relay_backpressure_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
            "invalid_previous_hop" | "invalid_signature" => {
                self.counters
                    .blind_relay_invalid_signature
                    .fetch_add(1, Ordering::Relaxed);
            }
            "envelope_too_large" => {
                self.counters
                    .blind_relay_envelope_too_large
                    .fetch_add(1, Ordering::Relaxed);
            }
            "ttl_exhausted" => {
                self.counters
                    .blind_relay_ttl_exhausted
                    .fetch_add(1, Ordering::Relaxed);
            }
            "no_route" => {
                self.counters
                    .blind_relay_no_route
                    .fetch_add(1, Ordering::Relaxed);
            }
            "missing_endpoint" | "invalid_endpoint" => {
                self.counters
                    .blind_relay_invalid_endpoint
                    .fetch_add(1, Ordering::Relaxed);
            }
            "request_failed" | "forward_failed" => {
                self.counters
                    .blind_relay_forward_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            reason if reason.starts_with("http_") => {
                self.counters
                    .blind_relay_forward_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        self.record_audit_event(now, "blind_relay_forward", "rejected", reason.to_string());
    }

    /// Records successful opaque node-to-node forwarding to a verified next hop.
    ///
    /// The key is retained only inside this process for route health scoring.
    /// Public status exposes only a short prefix and aggregate counters. Never
    /// pass route ids, encrypted blobs, client identifiers, endpoint URLs, or
    /// payload-derived details into this method.
    pub fn record_route_forward_success(&self, node_id: &[u8; 32], now: u64) {
        let mut route_health = self.route_health.write();
        let health = route_health.entry(*node_id).or_default();
        health.success_count = health.success_count.saturating_add(1);
        health.consecutive_failures = 0;
        health.last_success_at = Some(now);
        self.record_audit_event(
            now,
            "blind_relay_route_health",
            "accepted",
            format!("node_prefix={} result=success", hex::encode(&node_id[..4])),
        );
    }

    /// Records failed opaque node-to-node forwarding to a verified next hop.
    ///
    /// The reason must be a stable coarse bucket such as `request_failed` or
    /// `http_502`; no endpoint URL, route id, ciphertext, receiver, or client
    /// traffic metadata may be recorded here.
    pub fn record_route_forward_failure(
        &self,
        node_id: &[u8; 32],
        now: u64,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        let mut route_health = self.route_health.write();
        let health = route_health.entry(*node_id).or_default();
        health.failure_count = health.failure_count.saturating_add(1);
        health.consecutive_failures = health.consecutive_failures.saturating_add(1);
        health.last_failure_at = Some(now);
        health.last_failure_reason = Some(reason.clone());
        self.record_audit_event(
            now,
            "blind_relay_route_health",
            "rejected",
            format!(
                "node_prefix={} result=failure reason={reason}",
                hex::encode(&node_id[..4])
            ),
        );
    }

    /// Records a privacy-safe discovery control-plane audit event.
    ///
    /// Events are bounded and newest-last. Full peer public keys, client
    /// identifiers, traffic metadata, and payload-derived information must not
    /// be passed into this method.
    pub fn record_audit_event(
        &self,
        now: u64,
        action: impl Into<String>,
        outcome: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let mut events = self.audit_events.write();
        if events.len() >= MAX_AUDIT_EVENTS {
            events.pop_front();
        }
        events.push_back(PeerStoreAuditEvent {
            at: now,
            action: action.into(),
            outcome: outcome.into(),
            detail: detail.into(),
        });
    }

    /// Returns newest discovery audit events in chronological order.
    #[must_use]
    pub fn recent_audit_events(&self) -> Vec<PeerStoreAuditEvent> {
        self.audit_events.read().iter().cloned().collect()
    }

    /// Exports valid descriptors as a bootstrap snapshot for gossip response.
    ///
    /// When `public_only` is true, descriptors with `public_discovery=false`
    /// are excluded. `limit` caps the number of descriptors returned.
    #[must_use]
    pub fn export_bootstrap_snapshot(
        &self,
        generated_at: u64,
        now: u64,
        public_only: bool,
        limit: Option<usize>,
    ) -> NodeBootstrapSnapshot {
        self.counters
            .last_snapshot_at
            .store(generated_at, Ordering::Relaxed);
        let mut descriptors: Vec<SignedNodeDescriptor> = self
            .peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| !public_only || descriptor.descriptor.policy.public_discovery)
            .cloned()
            .collect();

        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        if let Some(limit) = limit {
            descriptors.truncate(limit);
        }

        self.record_audit_event(
            generated_at,
            "snapshot_export",
            "accepted",
            format!(
                "public_only={} limit={} exported={}",
                public_only,
                limit.map_or_else(|| "none".to_string(), |value| value.to_string()),
                descriptors.len()
            ),
        );

        NodeBootstrapSnapshot::new(generated_at, descriptors)
    }

    /// Builds a discovery snapshot response from current valid peers.
    #[must_use]
    pub fn build_snapshot_response(
        &self,
        generated_at: u64,
        now: u64,
        public_only: bool,
        limit: Option<usize>,
    ) -> NodeDiscoveryMessage {
        NodeDiscoveryMessage::SnapshotResponse {
            snapshot: self.export_bootstrap_snapshot(generated_at, now, public_only, limit),
        }
    }

    /// Returns a descriptor for a node id if present and valid at `now`.
    #[must_use]
    pub fn get_valid(&self, node_id: &[u8; 32], now: u64) -> Option<SignedNodeDescriptor> {
        self.peers
            .read()
            .get(node_id)
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .cloned()
    }

    /// Returns valid descriptors that advertise a capability.
    #[must_use]
    pub fn peers_with_capability(
        &self,
        capability: NodeCapability,
        now: u64,
    ) -> Vec<SignedNodeDescriptor> {
        self.peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_at(now).is_ok())
            .filter(|descriptor| descriptor.descriptor.capabilities.contains(&capability))
            .cloned()
            .collect()
    }

    /// Returns health-ranked route candidates for a capability.
    ///
    /// This method is the server-internal companion to the privacy-safe
    /// `PeerStoreStatus.route_candidates` payload. It sorts only by signed
    /// descriptor metadata and local discovery observation age. It never reads
    /// or derives from encrypted chat/media blobs, packet payloads, DNS data,
    /// destinations, client public IPs, voucher secrets, private keys, or
    /// wallet-level traffic.
    #[must_use]
    pub fn route_candidates_with_capability(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
    ) -> Vec<SignedNodeDescriptor> {
        self.route_candidates_with_capability_excluding(capability, now, limit, &[])
    }

    /// Returns health-ranked route candidates after excluding specific node ids.
    ///
    /// Exclusion happens before the limit is applied. This matters for fanout
    /// and future controlled multi-hop planning: self, already-used hops, or
    /// policy-excluded peers must not consume the limited candidate budget.
    ///
    /// The selection still uses only signed node descriptor metadata plus local
    /// node-to-node route health. It never reads encrypted blobs, plaintext,
    /// client IPs, destinations, DNS contents, voucher secrets, private keys,
    /// or wallet-level traffic.
    #[must_use]
    pub fn route_candidates_with_capability_excluding(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: usize,
        excluded_node_ids: &[[u8; 32]],
    ) -> Vec<SignedNodeDescriptor> {
        self.scored_route_candidates(capability, now, None)
            .into_iter()
            .filter(|candidate| {
                let node_id = candidate.descriptor.node_id();
                !excluded_node_ids
                    .iter()
                    .any(|excluded| *excluded == node_id)
            })
            .take(limit)
            .map(|candidate| candidate.descriptor)
            .collect()
    }

    /// Removes descriptors that are no longer valid at `now`.
    pub fn cleanup_expired(&self, now: u64) -> usize {
        let mut peers = self.peers.write();
        let before = peers.len();
        let mut removed = Vec::new();
        peers.retain(|node_id, descriptor| {
            let keep = descriptor.verify_at(now).is_ok();
            if !keep {
                removed.push(*node_id);
            }
            keep
        });
        drop(peers);
        if !removed.is_empty() {
            let mut metadata = self.peer_runtime.write();
            for node_id in &removed {
                metadata.remove(node_id);
            }
            let mut route_health = self.route_health.write();
            for node_id in &removed {
                route_health.remove(node_id);
            }
        }
        let removed_count = before.saturating_sub(self.peers.read().len());
        if removed_count > 0 {
            self.counters
                .expired_removed
                .fetch_add(removed_count as u64, Ordering::Relaxed);
            self.counters.last_cleanup_at.store(now, Ordering::Relaxed);
            self.record_audit_event(
                now,
                "expired_peer_cleanup",
                "accepted",
                format!("removed={removed_count}"),
            );
        }
        removed_count
    }

    /// Returns a monitoring snapshot.
    #[must_use]
    pub fn snapshot(&self, now: u64) -> PeerStoreSnapshot {
        let peers = self.peers.read();
        let mut valid_peers = 0usize;
        let mut public_peers = 0usize;
        let mut public_exit_peers = 0usize;

        for descriptor in peers.values() {
            if descriptor.verify_at(now).is_ok() {
                valid_peers += 1;
                if descriptor.descriptor.policy.public_discovery {
                    public_peers += 1;
                }
                if descriptor.descriptor.policy.allows_public_exit {
                    public_exit_peers += 1;
                }
            }
        }

        PeerStoreSnapshot {
            total_peers: peers.len(),
            valid_peers,
            public_peers,
            public_exit_peers,
        }
    }

    fn optional_age(now: u64, timestamp: Option<u64>) -> Option<u64> {
        timestamp.map(|value| now.saturating_sub(value))
    }

    fn stability(
        snapshot: &PeerStoreSnapshot,
        bootstrap: &PeerStoreBootstrapStatus,
        now: u64,
    ) -> PeerStoreStabilityStatus {
        let last_gossip_success_age_seconds =
            Self::optional_age(now, bootstrap.last_gossip_success_at);
        let last_gossip_round_age_seconds = Self::optional_age(now, bootstrap.last_gossip_round_at);
        let seed_recovery_configured = bootstrap.seed_endpoints_configured > 0;
        let restart_recovery_configured =
            seed_recovery_configured || bootstrap.peer_cache_configured;
        let has_minimum_peer_view = snapshot.valid_peers >= 2;
        let gossip_success_is_fresh = !bootstrap.gossip_enabled
            || last_gossip_success_age_seconds
                .map(|age| age <= DISCOVERY_GOSSIP_STALE_AFTER_SECS)
                .unwrap_or(false);
        let repeated_gossip_failure =
            bootstrap.consecutive_gossip_failures >= DISCOVERY_GOSSIP_FAILURE_ATTENTION_THRESHOLD;
        let last_gossip_failed = bootstrap.last_gossip_status.as_deref() == Some("failed");
        let last_gossip_stale = bootstrap.gossip_enabled
            && last_gossip_success_age_seconds
                .map(|age| age > DISCOVERY_GOSSIP_STALE_AFTER_SECS)
                .unwrap_or(false);

        let (health, relay_foundation_ready, detail, next_action) = if !bootstrap.enabled {
            (
                "disabled",
                false,
                "Discovery is disabled in local configuration.",
                "Enable discovery and configure signed seed or peer bootstrap before relying on peer discovery.",
            )
        } else if snapshot.valid_peers == 0 {
            (
                "pending",
                false,
                "No valid signed AeroNyx peers are currently in PeerStore.",
                "Confirm local descriptor registration, seed endpoints, peer cache, and inbound discovery reachability.",
            )
        } else if bootstrap.gossip_enabled && repeated_gossip_failure {
            (
                "failed",
                false,
                "Outbound discovery gossip has failed for multiple consecutive rounds.",
                "Check peer reachability, seed recovery, firewall rules, and discovery endpoint health.",
            )
        } else if last_gossip_stale {
            (
                "stale",
                false,
                "PeerStore has valid peers, but the last successful outbound gossip is stale.",
                "Wait for a fresh gossip success or inspect seed recovery and peer endpoint connectivity.",
            )
        } else if bootstrap.gossip_enabled && last_gossip_failed {
            (
                "degraded",
                false,
                "The latest outbound gossip round failed, but the consecutive failure threshold has not been reached.",
                "Monitor the next gossip round and inspect the privacy-safe failure bucket if it repeats.",
            )
        } else if !has_minimum_peer_view {
            (
                "degraded",
                false,
                "PeerStore has only one valid signed peer, so the multi-node relay foundation is incomplete.",
                "Add or recover at least one additional signed AeroNyx peer before testing relay paths.",
            )
        } else if !restart_recovery_configured {
            (
                "degraded",
                false,
                "PeerStore has fresh peers, but no seed recovery or peer cache is configured for restart resilience.",
                "Configure discovery seed endpoints or peer_cache_path before treating this node as a stable relay foundation.",
            )
        } else if gossip_success_is_fresh {
            (
                "healthy",
                true,
                "PeerStore has multiple valid signed peers and discovery gossip is fresh enough for relay foundation checks.",
                "Continue monitoring gossip freshness, rejected descriptors, and peer-cache persistence.",
            )
        } else {
            (
                "pending",
                false,
                "Discovery is enabled and peers exist, but no successful outbound gossip has been observed yet.",
                "Wait for the first successful gossip round or verify configured seed endpoints.",
            )
        };

        PeerStoreStabilityStatus {
            health: health.to_string(),
            relay_foundation_ready,
            detail: detail.to_string(),
            next_action: next_action.to_string(),
            last_gossip_success_age_seconds,
            last_gossip_round_age_seconds,
            seed_recovery_configured,
            stale_after_seconds: DISCOVERY_GOSSIP_STALE_AFTER_SECS,
            restart_recovery_configured,
        }
    }

    /// Returns nodeboard-friendly peer store status.
    #[must_use]
    pub fn status(&self, now: u64) -> PeerStoreStatus {
        let snapshot = self.snapshot(now);
        let bootstrap = self.bootstrap_status.read().clone();
        let stability = Self::stability(&snapshot, &bootstrap, now);
        let peer_summary = self.peer_summary(now);
        let route_candidates = self.route_candidate_status(now);

        PeerStoreStatus {
            snapshot,
            runtime: self.counters.snapshot(),
            max_peers: self.max_peers(),
            recent_audit_events: self.recent_audit_events(),
            bootstrap,
            stability,
            peer_summary,
            route_candidates,
        }
    }

    fn capability_label(capability: NodeCapability) -> &'static str {
        match capability {
            NodeCapability::PrivacyRelay => "privacy_relay",
            NodeCapability::ChatRelay => "chat_relay",
            NodeCapability::EncryptedStorage => "encrypted_storage",
            NodeCapability::AgentRelay => "agent_relay",
            NodeCapability::OnionMiddle => "onion_middle",
        }
    }

    fn descriptor_health(
        descriptor: &SignedNodeDescriptor,
        now: u64,
    ) -> (&'static str, Option<u64>) {
        if descriptor.verify_at(now).is_err() {
            return ("expired", None);
        }
        let remaining = descriptor.descriptor.expires_at.saturating_sub(now);
        if remaining <= PEER_DESCRIPTOR_STALE_WINDOW_SECS {
            ("stale", Some(remaining))
        } else {
            ("healthy", Some(remaining))
        }
    }

    fn source_score(source: &str) -> i64 {
        match source {
            "self" => 30,
            "gossip_announce" | "gossip_snapshot" => 25,
            "cache" | "cache_backup" => 18,
            "url" | "file" => 12,
            _ => 0,
        }
    }

    fn last_seen_score(last_seen_age_seconds: u64) -> i64 {
        if last_seen_age_seconds <= PEER_ROUTE_LAST_SEEN_FRESH_SECS {
            20
        } else if last_seen_age_seconds <= PEER_ROUTE_LAST_SEEN_ACCEPTABLE_SECS {
            10
        } else if last_seen_age_seconds <= PEER_ROUTE_LAST_SEEN_STALE_SECS {
            0
        } else {
            -20
        }
    }

    fn ttl_score(ttl_remaining_seconds: Option<u64>) -> i64 {
        ttl_remaining_seconds
            .map(|ttl| (ttl / PEER_DESCRIPTOR_STALE_WINDOW_SECS).min(20) as i64)
            .unwrap_or(-50)
    }

    fn capacity_score(descriptor: &SignedNodeDescriptor) -> i64 {
        let sessions = (descriptor.descriptor.capacity.max_sessions / 32).min(20) as i64;
        let bps = descriptor
            .descriptor
            .capacity
            .max_bps
            .map(|value| (value / 100_000_000).min(10) as i64)
            .unwrap_or(0);
        let pps = descriptor
            .descriptor
            .capacity
            .max_pps
            .map(|value| (value / 10_000).min(10) as i64)
            .unwrap_or(0);
        sessions + bps + pps
    }

    fn route_health_bucket_and_score(
        route_health: Option<&PeerRouteHealth>,
        now: u64,
    ) -> (&'static str, i64) {
        let Some(route_health) = route_health else {
            return ("unknown", 0);
        };
        let recent_failure = route_health
            .last_failure_at
            .map(|failed_at| now.saturating_sub(failed_at) <= PEER_ROUTE_RECENT_FAILURE_SECS)
            .unwrap_or(false);
        let success_after_failure =
            match (route_health.last_success_at, route_health.last_failure_at) {
                (Some(success_at), Some(failure_at)) => success_at >= failure_at,
                (Some(_), None) => true,
                _ => false,
            };

        if recent_failure && !success_after_failure {
            let penalty = (route_health.consecutive_failures.min(4) as i64) * 35;
            if route_health.consecutive_failures >= 3 {
                ("failing", -140)
            } else {
                ("degraded", -penalty)
            }
        } else if route_health.success_count > 0 {
            ("healthy", 8)
        } else {
            ("unknown", 0)
        }
    }

    fn route_score(
        descriptor: &SignedNodeDescriptor,
        source: &str,
        health: &str,
        last_seen_age_seconds: u64,
        ttl_remaining_seconds: Option<u64>,
        route_health_score: i64,
    ) -> i64 {
        let health_score = match health {
            "healthy" => 100,
            "stale" => 40,
            _ => -100,
        };
        health_score
            + Self::source_score(source)
            + Self::last_seen_score(last_seen_age_seconds)
            + Self::ttl_score(ttl_remaining_seconds)
            + Self::capacity_score(descriptor)
            + route_health_score
    }

    fn scored_route_candidates(
        &self,
        capability: NodeCapability,
        now: u64,
        limit: Option<usize>,
    ) -> Vec<ScoredPeerRouteCandidate> {
        let peers = self.peers.read();
        let metadata = self.peer_runtime.read();
        let route_health = self.route_health.read();
        let capability_label = Self::capability_label(capability).to_string();
        let mut candidates = Vec::new();

        for (node_id, descriptor) in peers.iter() {
            if descriptor.verify_at(now).is_err()
                || !descriptor.descriptor.capabilities.contains(&capability)
                || descriptor.descriptor.public_endpoint.is_none()
            {
                continue;
            }

            let meta = metadata.get(node_id);
            let source = meta
                .map(|value| value.source.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let last_seen_age_seconds = meta
                .map(|value| now.saturating_sub(value.last_seen_at))
                .unwrap_or(u64::MAX);
            let (health, ttl_remaining_seconds) = Self::descriptor_health(descriptor, now);
            if health == "expired" {
                continue;
            }
            let route_health_entry = route_health.get(node_id);
            let (route_health_bucket, route_health_score) =
                Self::route_health_bucket_and_score(route_health_entry, now);
            let score = Self::route_score(
                descriptor,
                &source,
                health,
                last_seen_age_seconds,
                ttl_remaining_seconds,
                route_health_score,
            );

            candidates.push(ScoredPeerRouteCandidate {
                descriptor: descriptor.clone(),
                summary: PeerStoreRouteCandidate {
                    node_id_prefix: hex::encode(&node_id[..4]),
                    capability: capability_label.clone(),
                    score,
                    source,
                    health: health.to_string(),
                    route_health: route_health_bucket.to_string(),
                    route_failure_count: route_health_entry
                        .map(|value| value.failure_count)
                        .unwrap_or(0),
                    route_consecutive_failures: route_health_entry
                        .map(|value| value.consecutive_failures)
                        .unwrap_or(0),
                    last_route_success_at: route_health_entry
                        .and_then(|value| value.last_success_at),
                    last_route_failure_at: route_health_entry
                        .and_then(|value| value.last_failure_at),
                    last_route_failure_reason: route_health_entry
                        .and_then(|value| value.last_failure_reason.clone()),
                    last_seen_age_seconds,
                    ttl_remaining_seconds,
                    endpoint_advertised: true,
                    public_discovery: descriptor.descriptor.policy.public_discovery,
                    region: descriptor.descriptor.policy.region.clone(),
                    max_sessions: descriptor.descriptor.capacity.max_sessions,
                    max_bps: descriptor.descriptor.capacity.max_bps,
                    max_pps: descriptor.descriptor.capacity.max_pps,
                },
            });
        }

        candidates.sort_by(|a, b| {
            b.summary
                .score
                .cmp(&a.summary.score)
                .then_with(|| {
                    a.summary
                        .last_seen_age_seconds
                        .cmp(&b.summary.last_seen_age_seconds)
                })
                .then_with(|| {
                    b.summary
                        .ttl_remaining_seconds
                        .unwrap_or(0)
                        .cmp(&a.summary.ttl_remaining_seconds.unwrap_or(0))
                })
                .then_with(|| a.descriptor.node_id().cmp(&b.descriptor.node_id()))
        });

        if let Some(limit) = limit {
            candidates.truncate(limit);
        }
        candidates
    }

    fn route_candidate_summaries(
        &self,
        capability: NodeCapability,
        now: u64,
    ) -> Vec<PeerStoreRouteCandidate> {
        self.scored_route_candidates(capability, now, Some(PEER_ROUTE_STATUS_LIMIT))
            .into_iter()
            .map(|candidate| candidate.summary)
            .collect()
    }

    /// Builds privacy-safe route candidate lists for nodeboard.
    #[must_use]
    pub fn route_candidate_status(&self, now: u64) -> PeerStoreRouteCandidateStatus {
        PeerStoreRouteCandidateStatus {
            generated_at: now,
            privacy_relay: self.route_candidate_summaries(NodeCapability::PrivacyRelay, now),
            chat_relay: self.route_candidate_summaries(NodeCapability::ChatRelay, now),
            onion_middle: self.route_candidate_summaries(NodeCapability::OnionMiddle, now),
        }
    }

    /// Builds a commercial peer summary for heartbeat/nodeboard.
    #[must_use]
    pub fn peer_summary(&self, now: u64) -> PeerStorePeerSummaryStatus {
        let peers = self.peers.read();
        let metadata = self.peer_runtime.read();
        let mut source_counts = BTreeMap::new();
        let mut rows = Vec::with_capacity(peers.len().min(64));
        let mut healthy_peers = 0usize;
        let mut stale_peers = 0usize;
        let mut expired_peers = 0usize;
        let mut valid_peers = 0usize;
        let mut privacy_relay_peers = 0usize;
        let mut chat_relay_peers = 0usize;
        let mut encrypted_storage_peers = 0usize;
        let mut agent_relay_peers = 0usize;
        let mut onion_middle_peers = 0usize;

        for (node_id, descriptor) in peers.iter() {
            let meta = metadata.get(node_id);
            let source = meta
                .map(|value| value.source.clone())
                .unwrap_or_else(|| "unknown".to_string());
            *source_counts.entry(source.clone()).or_insert(0) += 1;

            let (health, ttl_remaining_seconds) = Self::descriptor_health(descriptor, now);
            match health {
                "healthy" => {
                    healthy_peers += 1;
                    valid_peers += 1;
                }
                "stale" => {
                    stale_peers += 1;
                    valid_peers += 1;
                }
                _ => expired_peers += 1,
            }

            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::PrivacyRelay)
            {
                privacy_relay_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            {
                chat_relay_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::EncryptedStorage)
            {
                encrypted_storage_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::AgentRelay)
            {
                agent_relay_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::OnionMiddle)
            {
                onion_middle_peers += 1;
            }

            rows.push(PeerStorePeerSummary {
                node_id_prefix: hex::encode(&node_id[..4]),
                source,
                sequence: descriptor.sequence(),
                imported_count: meta.map(|value| value.imported_count).unwrap_or(0),
                first_seen_at: meta.map(|value| value.first_seen_at).unwrap_or(now),
                last_seen_at: meta.map(|value| value.last_seen_at).unwrap_or(now),
                last_seen_age_seconds: meta
                    .map(|value| now.saturating_sub(value.last_seen_at))
                    .unwrap_or(0),
                expires_at: descriptor.descriptor.expires_at,
                ttl_remaining_seconds,
                health: health.to_string(),
                capabilities: descriptor
                    .descriptor
                    .capabilities
                    .iter()
                    .copied()
                    .map(Self::capability_label)
                    .map(str::to_string)
                    .collect(),
                endpoint_advertised: descriptor.descriptor.public_endpoint.is_some(),
                public_discovery: descriptor.descriptor.policy.public_discovery,
                region: descriptor.descriptor.policy.region.clone(),
            });
        }

        rows.sort_by(|a, b| {
            a.health
                .cmp(&b.health)
                .then_with(|| a.source.cmp(&b.source))
                .then_with(|| a.node_id_prefix.cmp(&b.node_id_prefix))
        });

        PeerStorePeerSummaryStatus {
            total_peers: peers.len(),
            valid_peers,
            healthy_peers,
            stale_peers,
            expired_peers,
            privacy_relay_peers,
            chat_relay_peers,
            encrypted_storage_peers,
            agent_relay_peers,
            onion_middle_peers,
            source_counts,
            peers: rows,
        }
    }

    /// Returns the number of stored descriptors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.peers.read().len()
    }

    /// Returns true when the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for PeerStore {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PeerStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerStore")
            .field("peers", &self.len())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        NodeCapacity, NodeDescriptor, NodePolicy, SignedNodeDescriptor,
    };

    fn signed_descriptor(sequence: u64, expires_at: u64) -> SignedNodeDescriptor {
        let kp = IdentityKeyPair::generate();
        signed_descriptor_for(&kp, sequence, expires_at)
    }

    fn signed_descriptor_for(
        kp: &IdentityKeyPair,
        sequence: u64,
        expires_at: u64,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            sequence,
            1_700_000_000,
            expires_at,
            "test",
        );
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        descriptor.capacity = NodeCapacity {
            max_sessions: 128,
            max_bps: Some(500_000_000),
            max_pps: None,
        };
        descriptor.policy = NodePolicy::default();
        SignedNodeDescriptor::sign(descriptor, kp).unwrap()
    }

    #[test]
    fn test_upsert_verified_stores_descriptor() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_001_000);

        assert_eq!(
            store.upsert_verified(descriptor, 1_700_000_100).unwrap(),
            true
        );
        assert_eq!(store.len(), 1);
        assert_eq!(store.snapshot(1_700_000_100).valid_peers, 1);
    }

    #[test]
    fn test_same_sequence_is_idempotent() {
        let store = PeerStore::new();
        let kp = IdentityKeyPair::generate();
        let descriptor = signed_descriptor_for(&kp, 1, 1_700_001_000);
        let same = signed_descriptor_for(&kp, 1, 1_700_001_000);

        assert_eq!(
            store
                .upsert_verified(descriptor, 1_700_000_100)
                .expect("first insert"),
            true
        );
        assert_eq!(
            store
                .upsert_verified(same, 1_700_000_100)
                .expect("same sequence"),
            false
        );
    }

    #[test]
    fn test_stale_sequence_rejected() {
        let store = PeerStore::new();
        let kp = IdentityKeyPair::generate();
        let newer = signed_descriptor_for(&kp, 2, 1_700_001_000);
        let older = signed_descriptor_for(&kp, 1, 1_700_001_000);

        store.upsert_verified(newer, 1_700_000_100).unwrap();
        let err = store.upsert_verified(older, 1_700_000_100).unwrap_err();

        assert!(matches!(err, PeerStoreError::StaleSequence { .. }));
    }

    #[test]
    fn test_expired_descriptor_rejected() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_000_050);

        let err = store
            .upsert_verified(descriptor, 1_700_000_100)
            .unwrap_err();
        assert!(matches!(err, PeerStoreError::VerificationFailed));
        assert!(store.is_empty());
    }

    #[test]
    fn test_max_peers_rejects_new_descriptor_but_allows_existing_update() {
        let store = PeerStore::with_max_peers(1);
        let kp = IdentityKeyPair::generate();
        let first = signed_descriptor_for(&kp, 1, 1_700_001_000);
        let first_update = signed_descriptor_for(&kp, 2, 1_700_001_000);
        let second = signed_descriptor(1, 1_700_001_000);

        assert!(store.upsert_verified(first, 1_700_000_100).unwrap());
        assert!(store.upsert_verified(first_update, 1_700_000_100).unwrap());

        let err = store.upsert_verified(second, 1_700_000_100).unwrap_err();
        assert!(matches!(err, PeerStoreError::CapacityExceeded { .. }));
        assert_eq!(store.len(), 1);
        assert_eq!(store.status(1_700_000_100).runtime.capacity_rejected, 1);
    }

    #[test]
    fn test_capability_query_returns_only_valid_matching_peers() {
        let store = PeerStore::new();
        let matching = signed_descriptor(1, 1_700_001_000);
        let mut non_matching = signed_descriptor(1, 1_700_001_000);
        let kp = IdentityKeyPair::generate();
        non_matching.descriptor.node_id = kp.public_key_bytes();
        non_matching.descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
        non_matching = SignedNodeDescriptor::sign(non_matching.descriptor, &kp).unwrap();

        store.upsert_verified(matching, 1_700_000_100).unwrap();
        store.upsert_verified(non_matching, 1_700_000_100).unwrap();

        let peers = store.peers_with_capability(NodeCapability::ChatRelay, 1_700_000_100);
        assert_eq!(peers.len(), 1);
    }

    #[test]
    fn test_route_candidates_rank_healthy_endpoint_peers_without_payload_data() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let preferred_kp = IdentityKeyPair::generate();
        let stale_kp = IdentityKeyPair::generate();
        let no_endpoint_kp = IdentityKeyPair::generate();

        let mut preferred = signed_descriptor_for(&preferred_kp, 1, now + 2_000);
        preferred.descriptor.public_endpoint = Some("https://preferred.example".to_string());
        preferred.descriptor.capacity.max_sessions = 512;
        preferred = SignedNodeDescriptor::sign(preferred.descriptor, &preferred_kp).unwrap();

        let mut stale = signed_descriptor_for(&stale_kp, 1, now + 90);
        stale.descriptor.public_endpoint = Some("https://stale.example".to_string());
        stale.descriptor.capacity.max_sessions = 64;
        stale = SignedNodeDescriptor::sign(stale.descriptor, &stale_kp).unwrap();

        let no_endpoint = signed_descriptor_for(&no_endpoint_kp, 1, now + 2_000);

        store
            .upsert_verified_from_source(preferred.clone(), now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(stale.clone(), now, "gossip_snapshot")
            .unwrap();
        store
            .upsert_verified_from_source(no_endpoint, now, "gossip_announce")
            .unwrap();

        let candidates = store.route_candidates_with_capability(NodeCapability::ChatRelay, now, 8);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].node_id(), preferred.node_id());
        assert_eq!(candidates[1].node_id(), stale.node_id());

        let status = store.route_candidate_status(now);
        assert_eq!(status.chat_relay.len(), 2);
        assert_eq!(status.chat_relay[0].health, "healthy");
        assert_eq!(status.chat_relay[1].health, "stale");
        assert!(status.chat_relay[0].endpoint_advertised);
        assert!(status.chat_relay[0].score > status.chat_relay[1].score);
    }

    #[test]
    fn test_route_candidates_deprioritize_recent_forward_failures_without_payload_data() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let healthy_kp = IdentityKeyPair::generate();
        let failing_kp = IdentityKeyPair::generate();

        let mut healthy = signed_descriptor_for(&healthy_kp, 1, now + 2_000);
        healthy.descriptor.public_endpoint = Some("https://healthy.example".to_string());
        healthy = SignedNodeDescriptor::sign(healthy.descriptor, &healthy_kp).unwrap();

        let mut failing = signed_descriptor_for(&failing_kp, 1, now + 2_000);
        failing.descriptor.public_endpoint = Some("https://failing.example".to_string());
        failing = SignedNodeDescriptor::sign(failing.descriptor, &failing_kp).unwrap();
        let failing_node_id = failing.node_id();
        let failing_prefix = hex::encode(&failing_node_id[..4]);

        store
            .upsert_verified_from_source(healthy.clone(), now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(failing.clone(), now, "gossip_announce")
            .unwrap();

        store.record_route_forward_failure(&failing_node_id, now + 1, "request_failed");
        store.record_route_forward_failure(&failing_node_id, now + 2, "request_failed");
        store.record_route_forward_failure(&failing_node_id, now + 3, "http_502");

        let candidates =
            store.route_candidates_with_capability(NodeCapability::ChatRelay, now + 4, 8);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].node_id(), healthy.node_id());
        assert_eq!(candidates[1].node_id(), failing_node_id);

        let status = store.route_candidate_status(now + 4);
        let failing_row = status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == failing_prefix)
            .expect("failing peer should remain visible as a degraded candidate");
        assert_eq!(failing_row.route_health, "failing");
        assert_eq!(failing_row.route_failure_count, 3);
        assert_eq!(failing_row.route_consecutive_failures, 3);
        assert_eq!(
            failing_row.last_route_failure_reason.as_deref(),
            Some("http_502")
        );

        let status_json = serde_json::to_string(&status).unwrap();
        assert!(!status_json.contains("failing.example"));
        assert!(!status_json.contains(&hex::encode(failing_node_id)));
        assert!(!status_json.contains("encrypted_blob"));

        store.record_route_forward_success(&failing_node_id, now + 5);
        let recovered = store.route_candidate_status(now + 6);
        let recovered_row = recovered
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == failing_prefix)
            .expect("recovered peer should still be reported");
        assert_eq!(recovered_row.route_health, "healthy");
        assert_eq!(recovered_row.route_consecutive_failures, 0);
        assert_eq!(recovered_row.last_route_success_at, Some(now + 5));
    }

    #[test]
    fn test_route_candidates_apply_exclusion_before_limit_for_self_filtering() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let self_kp = IdentityKeyPair::generate();
        let peer_kp = IdentityKeyPair::generate();

        let mut self_descriptor = signed_descriptor_for(&self_kp, 1, now + 2_000);
        self_descriptor.descriptor.public_endpoint = Some("https://self.example".to_string());
        self_descriptor.descriptor.capacity.max_sessions = 1024;
        self_descriptor = SignedNodeDescriptor::sign(self_descriptor.descriptor, &self_kp).unwrap();
        let self_node_id = self_descriptor.node_id();

        let mut peer_descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
        peer_descriptor.descriptor.public_endpoint = Some("https://peer.example".to_string());
        peer_descriptor.descriptor.capacity.max_sessions = 32;
        peer_descriptor = SignedNodeDescriptor::sign(peer_descriptor.descriptor, &peer_kp).unwrap();
        let peer_node_id = peer_descriptor.node_id();

        store
            .upsert_verified_from_source(self_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(peer_descriptor, now, "gossip_announce")
            .unwrap();

        let candidates = store.route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            1,
            &[self_node_id],
        );

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].node_id(), peer_node_id);
    }

    #[test]
    fn test_cleanup_expired_removes_old_peers() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_001_000);

        store.upsert_verified(descriptor, 1_700_000_100).unwrap();
        assert_eq!(store.cleanup_expired(1_700_002_000), 1);
        assert!(store.is_empty());

        let status = store.status(1_700_002_001);
        assert_eq!(status.runtime.expired_removed, 1);
        assert_eq!(status.runtime.last_cleanup_at, Some(1_700_002_000));
        assert!(status.recent_audit_events.iter().any(|event| {
            event.action == "expired_peer_cleanup"
                && event.outcome == "accepted"
                && event.detail == "removed=1"
        }));
    }

    #[test]
    fn test_load_bootstrap_snapshot_reports_inserted_and_rejected() {
        let store = PeerStore::new();
        let valid = signed_descriptor(1, 1_700_001_000);
        let expired = signed_descriptor(1, 1_700_000_050);
        let snapshot = NodeBootstrapSnapshot::new(1_700_000_010, vec![valid, expired]);

        let report = store.load_bootstrap_snapshot(&snapshot, 1_700_000_100);

        assert_eq!(
            report,
            PeerStoreImportReport {
                total: 2,
                inserted: 1,
                unchanged: 0,
                stale: 0,
                rejected: 1,
            }
        );
        assert!(report.changed());
        assert_eq!(store.len(), 1);
        let status = store.status(1_700_000_100);
        assert_eq!(status.runtime.total_imported, 2);
        assert_eq!(status.runtime.inserted, 1);
        assert_eq!(status.runtime.rejected, 1);
        assert_eq!(status.runtime.last_import_at, Some(1_700_000_100));
    }

    #[test]
    fn test_load_bootstrap_snapshot_reports_unchanged_and_stale() {
        let store = PeerStore::new();
        let kp = IdentityKeyPair::generate();
        let newer = signed_descriptor_for(&kp, 2, 1_700_001_000);
        let same = signed_descriptor_for(&kp, 2, 1_700_001_000);
        let older = signed_descriptor_for(&kp, 1, 1_700_001_000);

        store.load_bootstrap_snapshot(
            &NodeBootstrapSnapshot::new(1_700_000_010, vec![newer]),
            1_700_000_100,
        );
        let report = store.load_bootstrap_snapshot(
            &NodeBootstrapSnapshot::new(1_700_000_020, vec![same, older]),
            1_700_000_100,
        );

        assert_eq!(
            report,
            PeerStoreImportReport {
                total: 2,
                inserted: 0,
                unchanged: 1,
                stale: 1,
                rejected: 0,
            }
        );
        assert!(!report.changed());
    }

    #[test]
    fn test_export_bootstrap_snapshot_filters_private_peers() {
        let store = PeerStore::new();
        let public = signed_descriptor(1, 1_700_001_000);
        let private_key = IdentityKeyPair::generate();
        let mut private = signed_descriptor_for(&private_key, 1, 1_700_001_000);
        private.descriptor.policy.public_discovery = false;
        private = SignedNodeDescriptor::sign(private.descriptor, &private_key).unwrap();

        store.upsert_verified(public, 1_700_000_100).unwrap();
        store.upsert_verified(private, 1_700_000_100).unwrap();

        let snapshot = store.export_bootstrap_snapshot(1_700_000_200, 1_700_000_100, true, None);
        assert_eq!(snapshot.peers.len(), 1);
        assert!(snapshot.peers[0].descriptor.policy.public_discovery);
    }

    #[test]
    fn test_build_snapshot_response_honors_limit() {
        let store = PeerStore::new();
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
            .unwrap();
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
            .unwrap();

        let message = store.build_snapshot_response(1_700_000_200, 1_700_000_100, true, Some(1));

        match message {
            NodeDiscoveryMessage::SnapshotResponse { snapshot } => {
                assert_eq!(snapshot.peers.len(), 1);
            }
            _ => panic!("expected snapshot response"),
        }
        assert_eq!(
            store.status(1_700_000_100).runtime.last_snapshot_at,
            Some(1_700_000_200)
        );
    }

    #[test]
    fn test_apply_descriptor_announce_imports_peer() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_001_000);
        let message = NodeDiscoveryMessage::DescriptorAnnounce { descriptor };

        let report = store.apply_discovery_message(&message, 1_700_000_100);

        assert_eq!(report.inserted, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_apply_snapshot_response_imports_peers() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_001_000);
        let message = NodeDiscoveryMessage::SnapshotResponse {
            snapshot: NodeBootstrapSnapshot::new(1_700_000_200, vec![descriptor]),
        };

        let report = store.apply_discovery_message(&message, 1_700_000_100);

        assert_eq!(report.inserted, 1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.status(1_700_000_100).runtime.inserted, 1);
    }

    #[test]
    fn test_runtime_rejection_counters_are_recorded() {
        let store = PeerStore::new();

        store.record_policy_rejected(1_700_000_300, "allow_list_enabled=true");
        store.record_rate_limited(1_700_000_301, "global_limit_per_minute=1");
        store.mark_gossip_at(1_700_000_333);

        let status = store.status(1_700_000_400);
        assert_eq!(status.runtime.policy_rejected, 1);
        assert_eq!(status.runtime.rate_limited, 1);
        assert_eq!(status.runtime.last_gossip_at, Some(1_700_000_333));
        assert_eq!(status.recent_audit_events.len(), 2);
        assert_eq!(
            status.recent_audit_events[0].action,
            "gossip_policy_rejected"
        );
        assert_eq!(status.recent_audit_events[1].action, "gossip_rate_limited");
    }

    #[test]
    fn test_audit_log_is_bounded() {
        let store = PeerStore::new();

        for i in 0..70 {
            store.record_audit_event(1_700_000_000 + i, "snapshot_export", "accepted", "test");
        }

        let events = store.recent_audit_events();
        assert_eq!(events.len(), MAX_AUDIT_EVENTS);
        assert_eq!(events[0].at, 1_700_000_006);
        assert_eq!(events[MAX_AUDIT_EVENTS - 1].at, 1_700_000_069);
    }

    #[test]
    fn test_bootstrap_status_is_recorded() {
        let store = PeerStore::new();

        store.configure_bootstrap_status(true, true, true, 2);
        store.record_bootstrap_source(
            1_700_000_010,
            "cache",
            "success",
            "total=1 inserted=1 unchanged=0 stale=0 rejected=0",
        );
        store.record_self_descriptor_status(1_700_000_011, "success", "registered");
        store.record_cache_save_status(1_700_000_012, "success", "exported=1");
        store.record_gossip_schedule(1_700_000_012, true, 180, -7);
        store.record_gossip_round(
            1_700_000_013,
            2,
            1,
            1,
            Some("snapshot_request_timeout".to_string()),
        );

        let status = store.status(1_700_000_100);
        assert!(status.bootstrap.enabled);
        assert!(status.bootstrap.peer_cache_configured);
        assert!(status.bootstrap.gossip_enabled);
        assert_eq!(status.bootstrap.seed_endpoints_configured, 2);
        assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.self_descriptor_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.last_cache_save_status.as_deref(),
            Some("success")
        );
        assert_eq!(status.bootstrap.last_gossip_attempted, 2);
        assert_eq!(status.bootstrap.last_gossip_seed_attempted, 1);
        assert_eq!(status.bootstrap.last_gossip_succeeded, 1);
        assert_eq!(status.bootstrap.last_gossip_failed, 1);
        assert_eq!(
            status.bootstrap.last_gossip_status.as_deref(),
            Some("degraded")
        );
        assert_eq!(
            status.bootstrap.last_gossip_failure_reason.as_deref(),
            Some("snapshot_request_timeout")
        );
        assert_eq!(status.bootstrap.consecutive_gossip_failures, 0);
        assert_eq!(status.bootstrap.last_gossip_success_at, Some(1_700_000_013));
        assert!(status.bootstrap.gossip_backpressure_active);
        assert_eq!(status.bootstrap.next_gossip_delay_seconds, Some(180));
        assert_eq!(status.bootstrap.next_gossip_jitter_seconds, -7);
        assert_eq!(
            status.bootstrap.last_gossip_schedule_at,
            Some(1_700_000_012)
        );
        assert!(status
            .recent_audit_events
            .iter()
            .any(|event| event.action == "outbound_gossip_round"));
        assert!(status
            .recent_audit_events
            .iter()
            .any(|event| event.action == "outbound_gossip_backpressure"));
    }

    #[test]
    fn test_gossip_round_tracks_consecutive_failures() {
        let store = PeerStore::new();

        store.record_gossip_round(
            1_700_000_020,
            1,
            0,
            1,
            Some("announce_request_connect".to_string()),
        );
        store.record_gossip_round(
            1_700_000_030,
            1,
            0,
            1,
            Some("snapshot_request_timeout".to_string()),
        );

        let status = store.status(1_700_000_040);
        assert_eq!(
            status.bootstrap.last_gossip_status.as_deref(),
            Some("failed")
        );
        assert_eq!(
            status.bootstrap.last_gossip_failure_reason.as_deref(),
            Some("snapshot_request_timeout")
        );
        assert_eq!(status.bootstrap.consecutive_gossip_failures, 2);
        assert_eq!(status.bootstrap.last_gossip_success_at, None);

        store.record_gossip_round(1_700_000_050, 1, 1, 1, None);
        let status = store.status(1_700_000_060);
        assert_eq!(
            status.bootstrap.last_gossip_status.as_deref(),
            Some("healthy")
        );
        assert_eq!(status.bootstrap.last_gossip_failure_reason, None);
        assert_eq!(status.bootstrap.consecutive_gossip_failures, 0);
        assert_eq!(status.bootstrap.last_gossip_success_at, Some(1_700_000_050));
    }

    #[test]
    fn test_gossip_schedule_status_tracks_backpressure_without_peer_details() {
        let store = PeerStore::new();

        store.record_gossip_schedule(1_700_000_020, true, 240, 12);
        assert_eq!(store.consecutive_gossip_failures(), 0);

        let status = store.status(1_700_000_030);
        assert!(status.bootstrap.gossip_backpressure_active);
        assert_eq!(status.bootstrap.next_gossip_delay_seconds, Some(240));
        assert_eq!(status.bootstrap.next_gossip_jitter_seconds, 12);
        assert_eq!(
            status.bootstrap.last_gossip_schedule_at,
            Some(1_700_000_020)
        );
        assert!(status.recent_audit_events.iter().any(|event| {
            event.action == "outbound_gossip_backpressure"
                && event.outcome == "limited"
                && !event.detail.contains("http")
        }));
    }

    #[test]
    fn test_blind_relay_runtime_stats_track_drop_reasons_without_payload_data() {
        let store = PeerStore::new();

        store.record_blind_relay_terminal(1_700_000_010, 2, 128);
        store.record_blind_relay_forwarded(1_700_000_011, 1);
        store.record_blind_relay_rejected(1_700_000_012, "backpressure");
        store.record_blind_relay_rejected(1_700_000_013, "invalid_signature");
        store.record_blind_relay_rejected(1_700_000_014, "ttl_exhausted");
        store.record_blind_relay_rejected(1_700_000_015, "no_route");
        store.record_blind_relay_rejected(1_700_000_016, "missing_endpoint");
        store.record_blind_relay_rejected(1_700_000_017, "http_502");

        let status = store.status(1_700_000_020);
        let stats = status.runtime.blind_relay;

        assert_eq!(stats.received, 8);
        assert_eq!(stats.terminal, 1);
        assert_eq!(stats.forwarded, 1);
        assert_eq!(stats.rejected, 6);
        assert_eq!(stats.backpressure_dropped, 1);
        assert_eq!(stats.invalid_signature, 1);
        assert_eq!(stats.ttl_exhausted, 1);
        assert_eq!(stats.no_route, 1);
        assert_eq!(stats.invalid_endpoint, 1);
        assert_eq!(stats.forward_failed, 1);
        assert_eq!(stats.last_event_at, Some(1_700_000_017));
        assert!(status
            .recent_audit_events
            .iter()
            .all(|event| !event.detail.contains("route_id")));
        assert!(status
            .recent_audit_events
            .iter()
            .all(|event| !event.detail.contains("encrypted_blob=")));
    }

    #[test]
    fn test_gossip_success_sets_effective_recovery_status_without_hiding_source_warning() {
        let store = PeerStore::new();

        store.record_bootstrap_source(
            1_700_000_010,
            "file",
            "warning",
            "total=1 inserted=0 unchanged=0 stale=0 rejected=1",
        );
        store.record_gossip_round(1_700_000_050, 2, 2, 1, None);

        let status = store.status(1_700_000_060);
        assert_eq!(
            status.bootstrap.last_source_status.as_deref(),
            Some("warning")
        );
        assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("file"));
        assert_eq!(status.bootstrap.recovery_status.as_deref(), Some("success"));
        assert_eq!(
            status.bootstrap.recovery_detail.as_deref(),
            Some("gossip_recovered attempted=2 succeeded=2 seed_attempted=1")
        );
        assert_eq!(status.bootstrap.recovery_at, Some(1_700_000_050));
    }

    #[test]
    fn test_peer_summary_tracks_source_ttl_health_and_capabilities() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let healthy = signed_descriptor(1, now + 1_000);
        let stale = signed_descriptor(1, now + 120);

        store
            .upsert_verified_from_source(healthy.clone(), now, "cache")
            .unwrap();
        store
            .upsert_verified_from_source(stale.clone(), now, "gossip_snapshot")
            .unwrap();
        store
            .upsert_verified_from_source(healthy, now + 20, "gossip_announce")
            .unwrap();

        let status = store.status(now + 30);

        assert_eq!(status.peer_summary.total_peers, 2);
        assert_eq!(status.peer_summary.valid_peers, 2);
        assert_eq!(status.peer_summary.healthy_peers, 1);
        assert_eq!(status.peer_summary.stale_peers, 1);
        assert_eq!(status.peer_summary.expired_peers, 0);
        assert_eq!(status.peer_summary.chat_relay_peers, 2);
        assert_eq!(status.peer_summary.privacy_relay_peers, 2);
        assert_eq!(
            status.peer_summary.source_counts.get("gossip_announce"),
            Some(&1)
        );
        assert_eq!(
            status.peer_summary.source_counts.get("gossip_snapshot"),
            Some(&1)
        );
        assert!(status
            .peer_summary
            .peers
            .iter()
            .any(|peer| peer.health == "stale" && peer.ttl_remaining_seconds == Some(90)));
        assert!(status
            .peer_summary
            .peers
            .iter()
            .all(|peer| peer.capabilities.contains(&"chat_relay".to_string())));
    }

    #[test]
    fn test_stability_marks_ready_when_peer_view_and_gossip_are_fresh() {
        let store = PeerStore::new();
        store.configure_bootstrap_status(true, true, true, 2);
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
            .unwrap();
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
            .unwrap();
        store.record_gossip_round(1_700_000_120, 2, 2, 1, None);

        let status = store.status(1_700_000_180);

        assert_eq!(status.stability.health, "healthy");
        assert!(status.stability.relay_foundation_ready);
        assert_eq!(status.stability.last_gossip_success_age_seconds, Some(60));
        assert_eq!(status.stability.last_gossip_round_age_seconds, Some(60));
        assert!(status.stability.seed_recovery_configured);
        assert!(status.stability.restart_recovery_configured);
    }

    #[test]
    fn test_stability_blocks_after_repeated_gossip_failures() {
        let store = PeerStore::new();
        store.configure_bootstrap_status(true, true, true, 1);
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
            .unwrap();
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
            .unwrap();

        for now in [1_700_000_120, 1_700_000_180, 1_700_000_240] {
            store.record_gossip_round(now, 2, 0, 1, Some("snapshot_request_timeout".to_string()));
        }

        let status = store.status(1_700_000_300);

        assert_eq!(status.stability.health, "failed");
        assert!(!status.stability.relay_foundation_ready);
        assert_eq!(status.bootstrap.consecutive_gossip_failures, 3);
        assert_eq!(status.stability.last_gossip_round_age_seconds, Some(60));
        assert_eq!(status.stability.last_gossip_success_age_seconds, None);
    }

    #[test]
    fn test_stability_marks_gossip_success_as_stale() {
        let store = PeerStore::new();
        store.configure_bootstrap_status(true, true, true, 1);
        store
            .upsert_verified(signed_descriptor(1, 1_700_010_000), 1_700_000_100)
            .unwrap();
        store
            .upsert_verified(signed_descriptor(1, 1_700_010_000), 1_700_000_100)
            .unwrap();
        store.record_gossip_round(1_700_000_120, 2, 2, 1, None);

        let status = store.status(1_700_001_100);

        assert_eq!(status.stability.health, "stale");
        assert!(!status.stability.relay_foundation_ready);
        assert_eq!(status.stability.last_gossip_success_age_seconds, Some(980));
        assert_eq!(
            status.stability.stale_after_seconds,
            DISCOVERY_GOSSIP_STALE_AFTER_SECS
        );
    }

    #[test]
    fn test_stability_requires_restart_recovery_for_relay_foundation() {
        let store = PeerStore::new();
        store.configure_bootstrap_status(true, false, true, 0);
        let now = 1_700_000_180;
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
            .unwrap();
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
            .unwrap();
        store.record_gossip_round(now - 60, 2, 2, 0, None);

        let status = store.status(now);

        assert_eq!(status.stability.health, "degraded");
        assert!(!status.stability.relay_foundation_ready);
        assert!(!status.stability.seed_recovery_configured);
        assert!(!status.stability.restart_recovery_configured);
        assert!(status.stability.next_action.contains("peer_cache_path"));
    }

    #[test]
    fn test_stability_accepts_peer_cache_as_restart_recovery() {
        let store = PeerStore::new();
        store.configure_bootstrap_status(true, true, true, 0);
        let now = 1_700_000_180;
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
            .unwrap();
        store
            .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
            .unwrap();
        store.record_gossip_round(now - 60, 2, 2, 0, None);

        let status = store.status(now);

        assert_eq!(status.stability.health, "healthy");
        assert!(status.stability.relay_foundation_ready);
        assert!(!status.stability.seed_recovery_configured);
        assert!(status.stability.restart_recovery_configured);
    }
}
