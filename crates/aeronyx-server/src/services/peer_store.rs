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
//! - Blind relay audit size buckets so exact encrypted blob sizes do not become
//!   traffic fingerprints in nodeboard or heartbeat diagnostics
//! - Per-peer node-to-node route health feedback so failed next hops are
//!   naturally deprioritized without exposing payloads or full peer endpoints
//! - Exclude-list route candidate selection so server internals can remove
//!   self or already-used hops before applying fanout/path limits
//! - Controlled route path planning for future multi-hop/onion relay, using
//!   only descriptor metadata and route health while exposing only safe prefixes
//! - Blind relay retry counters so nodeboard can distinguish transient
//!   next-hop recovery from final forwarding failures without payload metadata
//! - Startup self-check status so operators can see whether discovery has the
//!   cache, gossip, self-advertisement, and public endpoint wiring needed for
//!   commercial restart recovery without exposing endpoint values
//! - Blind relay loop-detection counters so immediate self/previous-hop loops
//!   are visible as aggregate drop reasons before future multi-hop rollout
//! - Blind relay replay-drop counters for duplicate route_id frames, without
//!   exposing route ids, previous hops, next hops, endpoints, or payload data
//! - Blind relay abuse-guard counters for previous-hop rate limiting and
//!   short quarantine without exposing peer identities or route metadata
//! - Per-peer health summary for operators, using only signed node metadata,
//!   gossip/import observation buckets, route-health counters, and relay
//!   protection buckets without payload or route reconstruction data
//! - Peer-cache startup recovery evidence that records cache/backup load
//!   status separately from generic bootstrap source status, so nodeboard can
//!   diagnose restart recovery without exposing cache paths or peer endpoints
//! - Network story status that converts peer summary, route candidates, and
//!   discovery stability into a product-facing aggregate readiness bucket for
//!   app/nodeboard/website surfaces without exposing endpoints or user data
//! - Privacy-safe recent peer lifecycle events so operators can understand
//!   whether peers are being inserted, refreshed, rejected, or expired without
//!   exposing full node IDs, endpoints, route IDs, payloads, or user metadata
//! - Peer quorum readiness summary that tells operators whether the verified
//!   peer view has enough fresh, routeable, restart-survivable peers for future
//!   multi-hop work without claiming global consensus or exposing endpoints
//! - Route-level failure quarantine so repeated opaque next-hop failures stop
//!   being selected for live relay paths while remaining visible to operators
//! - Blind relay transport failure buckets count as forward failures so
//!   nodeboard and public health surfaces do not under-report unresponsive
//!   next-hop relay paths
//! - Blind relay quality summary converts opaque runtime counters into a
//!   privacy-safe readiness bucket for nodeboard, website, and AI runbooks
//! - Blind relay synthetic probe counters and last-probe age provide
//!   low-frequency route readiness evidence without inflating real encrypted
//!   message traffic totals
//! - Blind relay evidence mode separates real relay traffic from synthetic
//!   probes so public surfaces do not overstate user-message movement
//! - Bounded two-hop path proof history records recent entry -> middle ->
//!   terminal proof outcomes for nodeboard/public status without exposing node
//!   IDs, route IDs, endpoints, payloads, or social graph metadata
//! - Blind relay readiness reason gives operators a stable privacy-safe bucket
//!   for why the relay path is ready, probe-only, degraded, protected, or idle
//! - Blind relay timestamp freshness counters show stale/future route-frame
//!   protection without exposing route ids, peer endpoints, payloads, or users
//! - Routeability evidence separates advertised endpoints from actually
//!   reachable relay paths, so quorum and network-story readiness cannot be
//!   inflated by unprobed peers
//! - Expired peers are downgraded and retained instead of deleted so local
//!   peer history survives cleanup and restart without being treated as live
//! - Blind relay forwarding can query routeability readiness directly, keeping
//!   node-to-node encrypted routing tied to fresh probe/forward evidence
//! - Heartbeat can export a bounded signed peer-record snapshot so centralized
//!   coordination can verify peer records instead of trusting derived counters
//! - Two-hop path proof history exposes privacy-safe freshness buckets and
//!   latest success/failure ages so UI surfaces can distinguish fresh,
//!   stale, failed, and forming proof states without reconstructing routes
//! - Two-hop path proof quality context records only coarse path/candidate/TTL
//!   buckets so operators can verify relay maturity without seeing node IDs,
//!   endpoints, route IDs, encrypted blobs, or social graph edges
//! - Two-hop path proof scope separates synthetic control-plane reachability
//!   from real message-delivery proof so public surfaces never overstate App
//!   chat delivery before true terminal store-and-forward evidence exists
//! - Two-hop message-delivery readiness exposes fresh terminal ChatRelay proof
//!   as its own aggregate gate, so App/nodeboard/backend can distinguish
//!   onion route reachability from actual store-and-forward delivery evidence
//! - Onion middle-hop recovery can distinguish route quarantine from ordinary
//!   unknown routeability, allowing cold-start proof attempts without sending
//!   through peers that are actively isolated by local route health policy
//! - Network story readiness treats a fresh successful two-hop path proof as
//!   onion-ready evidence, preventing proven delivery paths from being hidden by
//!   conservative local route-candidate planning after restart
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
//! v0.49.0-TwoHopProbeReasonBuckets - Bucket runtime two-hop blind relay probe errors
//! v0.48.0-BlindRelayFreshnessGate - Require fresh accepted/probe evidence before reporting blind relay ready
//! v0.47.0-TwoHopProofBackedStory - Let fresh two-hop path proof promote local network story to onion_ready
//! v0.46.0-OnionMiddleRouteabilityRecovery - Exposed route quarantine checks for onion middle cold-start recovery
//! v0.45.0-TwoHopMessageDeliveryReadiness - Added aggregate freshness/streak gates for message-delivery proof
//! v0.44.0-TwoHopOnionDeliveryScope - Mark synthetic onion terminal delivery as message-delivery proof
//! v0.43.0-TwoHopProofScope - Added explicit control-plane proof scope for synthetic two-hop probes
//! v0.42.0-TwoHopProofQualityContext - Added privacy-safe path/candidate/TTL buckets
//! v0.41.0-TwoHopProofFreshness - Added freshness bucket and latest success/failure ages
//! v0.40.0-TwoHopPathProofCounters - Added aggregate two-hop blind relay path proof counters
//! v0.39.0-SignedPeerRecordsHeartbeat - Added bounded verifiable peer-record snapshot for heartbeat
//! v0.38.0-RouteabilityForwardGate - Exposed routeability readiness helper for blind relay next-hop selection
//! v0.37.0-ExpiredPeerRetention - Downgrade expired signed peers instead of deleting local peer state
//! v0.36.0-RouteabilityEvidence - Require fresh probe/forward evidence for route-ready peer status
//! v0.35.4-BlindRelayTimestampProtection - Count stale/future route-frame protection
//! v0.35.3-BlindRelayReadinessReason - Added privacy-safe readiness reason bucket
//! v0.35.2-BlindRelayEvidenceMode - Distinguish real relay traffic from synthetic probe evidence
//! v0.35.1-BlindRelayProbeAge - Expose synthetic probe age separately from real relay event age
//! v0.35.0-BlindRelayProbeStats - Added privacy-safe blind relay synthetic probe counters
//! v0.34.0-BlindRelayQualityStatus - Added aggregate blind relay quality summary
//! v0.33.0-BlindRelayTransportFailureStats - Count transport buckets as forward failures
//! v0.32.0-PeerRouteFailureQuarantine - Added route-level next-hop quarantine after repeated failures
//! v0.31.0-PeerQuorumReadiness - Added privacy-safe peer quorum readiness summary
//! v0.30.0-PeerLifecycleEvents - Added privacy-safe recent peer discovery lifecycle events
//! v0.29.0-NetworkStoryAttentionPriority - Make degraded discovery stability outrank peer-view marketing status
//! v0.28.0-NetworkStoryStatus - Added product-facing aggregate discovery readiness story
//! v0.27.0-PeerCacheRecoveryEvidence - Added peer-cache load evidence and restart recovery sources
//! v0.26.0-PeerHealthSummary - Added privacy-safe per-peer health summary
//! v0.25.0-BlindRelayAbuseGuard - Added aggregate rate-limit/quarantine counters
//! v0.24.0-BlindRelayReplayGuard - Added privacy-safe duplicate route drop counter
//! v0.23.0-BlindRelayLoopGuard - Added privacy-safe blind relay loop drop counter
//! v0.22.0-DiscoveryStartupSelfCheck - Added privacy-safe discovery startup self-check status
//! v0.21.0-BlindRelayRetryStats - Added privacy-safe blind relay retry observability
//! v0.20.0-BlindRelaySizeBuckets - Bucket blind relay encrypted blob sizes in audit events
//! v0.19.0-ControlledRoutePathPlanner - Added privacy-safe multi-hop path planning foundation
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
const PEER_ROUTEABILITY_STALE_AFTER_SECS: u64 = 1_800;
const PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD: u64 = 3;
const PEER_ROUTE_FAILURE_QUARANTINE_SECS: u64 = 300;
const PEER_ROUTE_STATUS_LIMIT: usize = 8;
const PEER_HEALTH_STATUS_LIMIT: usize = 64;
const PEER_QUORUM_MIN_VALID_PEERS: usize = 2;
const PEER_QUORUM_MIN_ROUTEABLE_CHAT_RELAYS: usize = 1;

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
const MAX_PEER_EVENTS: usize = 64;
const MAX_TWO_HOP_PATH_PROOF_EVENTS: usize = 32;

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

/// Privacy-safe two-hop relay path proof event.
///
/// This is local protocol-health evidence for nodeboard, public website
/// aggregation, and AI runbooks. It deliberately records only coarse proof
/// outcome buckets. It must never include node ids, endpoint URLs, route ids,
/// encrypted blobs, receiver identities, client IPs, DNS contents,
/// destinations, Memory Chain plaintext, voucher secrets, wallet-level
/// traffic, or social graph metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreTwoHopPathProofEvent {
    /// Unix timestamp when the proof result was recorded.
    pub at: u64,
    /// Stable outcome bucket: accepted or rejected.
    pub outcome: String,
    /// Stable privacy-safe reason bucket.
    pub reason_bucket: String,
    /// Stable evidence mode for downstream status copy.
    ///
    /// Synthetic two-hop probes use `synthetic_two_hop_control_probe` because
    /// they prove route-control reachability, not App chat payload delivery.
    pub evidence_mode: String,
    /// Stable proof scope: control_plane today, message_delivery in future
    /// true terminal store-and-forward proof events.
    #[serde(default)]
    pub proof_scope: String,
    /// Planned relay path shape for this proof.
    pub path_shape: String,
    /// Number of relay hops proven by this event.
    pub hop_count: u8,
    /// Stable route policy bucket used by the proof planner.
    #[serde(default)]
    pub path_policy: String,
    /// Coarse count bucket for routeable middle-hop candidates.
    #[serde(default)]
    pub middle_candidate_bucket: String,
    /// Coarse count bucket for routeable terminal-hop candidates.
    #[serde(default)]
    pub terminal_candidate_bucket: String,
    /// Coarse TTL shape, never a route id or per-peer path.
    #[serde(default)]
    pub ttl_shape: String,
}

/// Bounded privacy-safe history for recent two-hop relay path proofs.
///
/// This is not a durable ledger and not user traffic accounting. It is a
/// rolling operator view over synthetic protocol-health probes so nodeboard and
/// the public website can show whether the blind relay fabric is repeatedly
/// proving entry -> middle -> terminal reachability while preserving the
/// blind-node invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreTwoHopPathProofHistory {
    /// Unix timestamp when this summary was generated.
    pub generated_at: u64,
    /// Stable readiness bucket: forming, ready, stale, attention, or idle.
    pub status: String,
    /// Stable freshness bucket for UI and runbooks: forming, fresh_success,
    /// stale_success, recent_failure, or no_success.
    ///
    /// This is derived only from bounded local proof outcomes and coarse age
    /// windows. It must never encode node IDs, route IDs, endpoints, payloads,
    /// receiver identities, or social graph information.
    pub freshness_bucket: String,
    /// Whether the latest retained proof is a fresh accepted two-hop path.
    pub proof_ready: bool,
    /// Whether the latest proof success is still within the routeability window.
    pub recent_success_ready: bool,
    /// Whether the latest retained proof is a fresh terminal message-delivery proof.
    ///
    /// This is stricter than `proof_ready`: control-plane probes can prove
    /// entry -> middle -> terminal reachability, while this gate requires the
    /// terminal hop to accept the opaque payload into ChatRelay store-and-forward.
    #[serde(default)]
    pub message_delivery_ready: bool,
    /// Whether any retained terminal message-delivery proof is still fresh.
    #[serde(default)]
    pub recent_message_delivery_ready: bool,
    /// Whether the latest retained proof ended in one or more failures.
    pub failure_streak_active: bool,
    /// Maximum number of recent proof events retained by this process.
    pub window_size: usize,
    /// Number of retained proof events in this summary.
    pub retained_events: usize,
    /// Retained proof attempts in the bounded window.
    pub attempted: u64,
    /// Retained accepted proofs in the bounded window.
    pub succeeded: u64,
    /// Retained accepted proofs whose scope is terminal message delivery.
    #[serde(default)]
    pub message_delivery_successes: u64,
    /// Retained rejected proofs in the bounded window.
    pub failed: u64,
    /// Success percentage over retained events.
    pub success_percent: u8,
    /// Latest proof outcome, when any retained event exists.
    pub latest_outcome: Option<String>,
    /// Latest proof reason bucket, when any retained event exists.
    pub latest_reason_bucket: Option<String>,
    /// Seconds since the latest retained proof event.
    pub latest_age_seconds: Option<u64>,
    /// Seconds since the latest retained accepted proof event.
    pub latest_success_age_seconds: Option<u64>,
    /// Seconds since the latest retained rejected proof event.
    pub latest_failure_age_seconds: Option<u64>,
    /// Seconds since the latest retained terminal message-delivery proof.
    #[serde(default)]
    pub latest_message_delivery_age_seconds: Option<u64>,
    /// Consecutive accepted proofs ending at the latest event.
    pub consecutive_successes: u64,
    /// Consecutive rejected proofs ending at the latest event.
    pub consecutive_failures: u64,
    /// Consecutive accepted terminal message-delivery proofs ending at latest event.
    #[serde(default)]
    pub consecutive_message_delivery_successes: u64,
    /// Aggregate reason-bucket counts in the retained window.
    ///
    /// This exposes only stable buckets derived from
    /// `two_hop_path_proof_reason_bucket`; it must never include raw errors,
    /// endpoints, route ids, node ids, encrypted blobs, receiver identities, or
    /// other route metadata.
    #[serde(default)]
    pub reason_bucket_counts: BTreeMap<String, u64>,
    /// Aggregate rejected proof reason-bucket counts in the retained window.
    ///
    /// Accepted proofs are intentionally excluded so nodeboard can explain
    /// recent failures without parsing the retained event list or leaking
    /// private route metadata.
    #[serde(default)]
    pub failure_reason_bucket_counts: BTreeMap<String, u64>,
    /// Aggregate proof path-shape counts in the retained window.
    pub path_shape_counts: BTreeMap<String, u64>,
    /// Aggregate candidate-pool quality buckets in the retained window.
    pub candidate_pool_counts: BTreeMap<String, u64>,
    /// Aggregate TTL-shape counts in the retained window.
    pub ttl_shape_counts: BTreeMap<String, u64>,
    /// Aggregate proof-scope counts in the retained window.
    ///
    /// Current synthetic two-hop probes are `control_plane` only. Future
    /// terminal store-and-forward probes or real App traffic can add
    /// `message_delivery` without changing existing counters.
    #[serde(default)]
    pub proof_scope_counts: BTreeMap<String, u64>,
    /// Seconds after which a retained successful proof is considered stale.
    pub stale_after_seconds: u64,
    /// Dominant proof scope represented by the latest retained event.
    #[serde(default)]
    pub proof_scope: String,
    /// Privacy-safe next action for nodeboard, website, and AI runbooks.
    pub next_action: String,
    /// Retained events in chronological order.
    pub events: Vec<PeerStoreTwoHopPathProofEvent>,
    /// Explicit invariant for downstream UI and AI-agent consumers.
    pub privacy_invariant: String,
    /// Explicit privacy boundary for downstream UI and API consumers.
    pub privacy_boundary: String,
}

/// Privacy-safe peer discovery lifecycle event.
///
/// This is separate from the generic audit log because nodeboard and backend
/// status pages need an easy way to explain peer discovery motion: inserted,
/// refreshed, rejected, or expired. It intentionally exposes only a short node
/// prefix plus stable reason/source buckets. It must never include endpoint
/// URLs, full public keys, route ids, encrypted blobs, receiver identities,
/// client IPs, destinations, DNS contents, voucher secrets, private keys,
/// wallet-level traffic, or plaintext content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStorePeerEvent {
    /// Unix timestamp when the peer lifecycle event was recorded.
    pub at: u64,
    /// Stable event bucket: peer_inserted, peer_upgraded, peer_refreshed,
    /// peer_rejected, or peer_expired.
    pub event: String,
    /// Outcome bucket: accepted, ignored, rejected, or expired.
    pub outcome: String,
    /// Coarse import/source bucket such as self, cache, gossip_snapshot, or gossip_announce.
    pub source: String,
    /// Short node id prefix for operator debugging without exposing full keys.
    pub node_id_prefix: String,
    /// Descriptor sequence observed with this event, when available.
    pub sequence: Option<u64>,
    /// Stable reason bucket for rejected/expired events.
    pub reason: Option<String>,
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
    /// Last peer-cache startup load source bucket: cache or cache_backup.
    ///
    /// This is separated from `last_source_kind` because generic bootstrap
    /// sources may include file/url/config events. Operators need a stable
    /// restart-recovery signal without exposing the cache path or peer
    /// endpoints.
    pub last_cache_load_source: Option<String>,
    /// Last peer-cache startup load status: success, warning, failed, missing.
    pub last_cache_load_status: Option<String>,
    /// Last peer-cache startup load detail with aggregate import counts only.
    pub last_cache_load_detail: Option<String>,
    /// Timestamp of the last peer-cache startup load attempt.
    pub last_cache_load_at: Option<u64>,
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
    /// Startup discovery readiness bucket: skipped, ready, or warning.
    ///
    /// This is recorded once during server startup from local configuration so
    /// nodeboard can distinguish "discovery is healthy" from "discovery is
    /// running but missing commercial recovery paths". It never stores config
    /// values such as file paths, peer URLs, seed URLs, or public endpoints.
    pub startup_self_check_status: Option<String>,
    /// Privacy-safe aggregate detail for the startup readiness bucket.
    pub startup_self_check_detail: Option<String>,
    /// Unix timestamp when startup readiness was evaluated.
    pub startup_self_check_at: Option<u64>,
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
            last_cache_load_source: None,
            last_cache_load_status: None,
            last_cache_load_detail: None,
            last_cache_load_at: None,
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
            startup_self_check_status: None,
            startup_self_check_detail: None,
            startup_self_check_at: None,
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
    /// Configured restart recovery source buckets, such as `seed_endpoints`
    /// and `peer_cache`.
    ///
    /// This is configuration evidence only. It deliberately does not expose
    /// seed URLs, cache paths, peer URLs, public keys, client IPs, destinations,
    /// DNS contents, packet payloads, voucher secrets, private keys, or
    /// wallet-level traffic.
    pub restart_recovery_sources: Vec<String>,
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
    /// Legacy counter for expired descriptors physically removed by cleanup.
    ///
    /// New code should prefer `expired_degraded`. This field is retained for
    /// backward-compatible nodeboard/API consumers.
    pub expired_removed: u64,
    /// Total expired signed descriptors downgraded and retained locally.
    pub expired_degraded: u64,
    /// Opaque node-to-node blind relay counters and drop reason buckets.
    pub blind_relay: PeerStoreBlindRelayStats,
    /// Unix timestamp of the last descriptor import attempt.
    pub last_import_at: Option<u64>,
    /// Unix timestamp of the last gossip exchange observed by this node.
    pub last_gossip_at: Option<u64>,
    /// Unix timestamp of the last exported snapshot.
    pub last_snapshot_at: Option<u64>,
    /// Unix timestamp of the last cleanup that downgraded or removed at least one expired peer.
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
    /// Requests rejected because route metadata would immediately loop.
    pub loop_detected: u64,
    /// Requests dropped because this node already observed the route id.
    pub replay_dropped: u64,
    /// Requests rejected because signed routing timestamps were stale or too far ahead.
    ///
    /// This is a coarse replay/freshness protection counter. It must not be
    /// expanded into route ids, exact timestamps, previous-hop ids, endpoint
    /// URLs, encrypted blobs, receiver identities, or user metadata.
    pub timestamp_rejected: u64,
    /// Requests rejected by local previous-hop rate limiting.
    pub rate_limited: u64,
    /// Requests rejected while the previous-hop bucket was quarantined.
    pub quarantined: u64,
    /// Number of short previous-hop quarantines started by the abuse guard.
    pub quarantine_started: u64,
    /// Retry sleeps scheduled for transient next-hop failures.
    pub retry_attempted: u64,
    /// Blind relay forwards that succeeded after at least one retry.
    pub retry_succeeded: u64,
    /// Blind relay forwards that still failed after retry attempts were exhausted.
    pub retry_exhausted: u64,
    /// Low-frequency synthetic blind relay probes attempted by this node.
    ///
    /// Probe counters are not user traffic and must not be added to encrypted
    /// message, packet, or payload byte totals.
    pub probe_attempted: u64,
    /// Synthetic probes accepted by a verified next-hop blind relay endpoint.
    pub probe_succeeded: u64,
    /// Synthetic probes rejected or failed at transport/ACK validation.
    pub probe_failed: u64,
    /// Unix timestamp of the last synthetic route readiness probe.
    ///
    /// This is not user traffic and must not be used to infer encrypted
    /// message, packet, or payload byte activity.
    pub last_probe_at: Option<u64>,
    /// Low-frequency synthetic entry -> middle -> terminal path proofs attempted.
    ///
    /// This is aggregate protocol evidence only. It deliberately does not
    /// expose the selected path, endpoint URLs, route ids, node ids, encrypted
    /// blobs, receiver identities, client IPs, DNS contents, destinations,
    /// Memory Chain plaintext, voucher secrets, private keys, wallet-level
    /// traffic, or social graph metadata.
    pub two_hop_probe_attempted: u64,
    /// Two-hop path proofs accepted by the middle hop and terminal hop chain.
    pub two_hop_probe_succeeded: u64,
    /// Two-hop path proofs rejected or failed at route planning, transport, or ACK validation.
    pub two_hop_probe_failed: u64,
    /// Unix timestamp of the last two-hop synthetic path proof.
    pub last_two_hop_probe_at: Option<u64>,
    /// Unix timestamp of the last accepted terminal or forwarded blind relay work.
    ///
    /// This is aggregate freshness evidence only. It must never be joined with
    /// route ids, endpoint URLs, peer ids, encrypted blobs, receiver identities,
    /// client IPs, DNS contents, voucher secrets, private keys, wallet-level
    /// traffic, plaintext, or social graph metadata.
    pub last_accepted_at: Option<u64>,
    /// Unix timestamp of the last blind relay event.
    pub last_event_at: Option<u64>,
}

/// Aggregate quality bucket for blind relay operations.
///
/// This summary combines cumulative in-process counters with freshness-gated
/// readiness. Cumulative totals remain historical evidence, while `*_ready`
/// booleans require recent accepted work or synthetic probes. It gives
/// nodeboard, public website status, and AI runbooks a stable operator signal
/// without exposing route ids, peer endpoints, previous/next hops, encrypted
/// blobs, receiver identities, client IPs, DNS contents, destinations, voucher
/// secrets, private keys, wallet-level traffic, or plaintext.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreBlindRelayQualityStatus {
    /// Unix timestamp when this summary was generated.
    pub generated_at: u64,
    /// Stable bucket: idle, observing, stale, ready, protecting, degraded, or attention.
    pub status: String,
    /// Whether this process has fresh accepted terminal/forwarded work or fresh probe evidence.
    pub runtime_ready: bool,
    /// Whether fresh successful terminal/forwarded/probe evidence exists without final transport failures.
    pub quality_ready: bool,
    /// Whether this process has fresh accepted real terminal or forwarded relay work.
    pub real_relay_ready: bool,
    /// Whether this process has fresh successful synthetic route probe evidence.
    pub synthetic_probe_ready: bool,
    /// Stable evidence bucket: idle, real_relay_traffic,
    /// synthetic_two_hop_control_probe, synthetic_probe, probe_failed, or
    /// real_relay_attempted.
    pub evidence_mode: String,
    /// Stable proof scope for UI copy: message_delivery, control_plane,
    /// single_hop_control_plane, attempted, or none.
    ///
    /// This prevents dashboards from presenting synthetic control-plane
    /// reachability as completed App chat delivery.
    #[serde(default)]
    pub proof_scope: String,
    /// Stable readiness reason bucket for operators and public status surfaces.
    ///
    /// Values are intentionally coarse and must never encode peer ids,
    /// endpoints, route ids, encrypted blob sizes, receiver identities, or
    /// social graph hints.
    pub readiness_reason: String,
    /// Accepted terminal plus forwarded requests.
    pub accepted_total: u64,
    /// Unix timestamp of the last accepted terminal/forwarded blind relay work.
    ///
    /// This is an aggregate liveness marker only. It must never be expanded
    /// into route ids, endpoint URLs, node ids, encrypted blobs, receiver
    /// identities, client IPs, DNS contents, voucher secrets, private keys,
    /// wallet-level traffic, plaintext, or social graph metadata.
    pub last_accepted_at: Option<u64>,
    /// Rejected requests counted as transport/next-hop forwarding failures.
    pub forward_failed: u64,
    /// Retry attempts that were exhausted without a successful next-hop ACK.
    pub retry_exhausted: u64,
    /// Requests dropped by local backpressure.
    pub backpressure_dropped: u64,
    /// Low-frequency synthetic route probes attempted by this process.
    pub probe_attempted: u64,
    /// Synthetic route probes accepted by verified next-hop blind relay endpoints.
    pub probe_succeeded: u64,
    /// Synthetic route probes that failed without exposing endpoint or route data.
    pub probe_failed: u64,
    /// Whether this process has successful synthetic two-hop path proof evidence.
    pub two_hop_probe_ready: bool,
    /// Low-frequency synthetic entry -> middle -> terminal path proofs attempted.
    pub two_hop_probe_attempted: u64,
    /// Synthetic two-hop path proofs accepted by the relay chain.
    pub two_hop_probe_succeeded: u64,
    /// Synthetic two-hop path proofs that failed without exposing endpoint or route data.
    pub two_hop_probe_failed: u64,
    /// Seconds since the last two-hop synthetic path proof, when known.
    pub last_two_hop_probe_age_seconds: Option<u64>,
    /// Stale or future-dated opaque route frames rejected by the freshness guard.
    ///
    /// This is an aggregate protection counter only. Do not expand it into
    /// route ids, exact timestamps, previous-hop ids, endpoints, encrypted
    /// payloads, receiver identities, client IPs, DNS contents, Memory Chain
    /// plaintext, or social graph edges.
    pub timestamp_rejected: u64,
    /// Whether abuse protection counters have fired in this process.
    pub protection_active: bool,
    /// Percentage of received requests accepted as terminal or forwarded.
    pub accepted_percent: u8,
    /// Seconds since the last blind relay event, when known.
    pub last_event_age_seconds: Option<u64>,
    /// Seconds since the last accepted terminal/forwarded blind relay work.
    ///
    /// Dashboards should use this together with `runtime_ready` to avoid
    /// presenting stale historical relay evidence as current route readiness.
    pub last_accepted_age_seconds: Option<u64>,
    /// Seconds since the last synthetic route readiness probe, when known.
    ///
    /// This is separate from real relay event age so dashboards can show probe
    /// freshness without implying user traffic occurred.
    pub last_probe_age_seconds: Option<u64>,
    /// Operator-facing detail with aggregate counters only.
    pub detail: String,
    /// Privacy-safe next action for nodeboard / AI runbooks.
    pub next_action: String,
    /// Explicit privacy boundary for downstream UI and API consumers.
    pub privacy_boundary: String,
}

/// Bounded signed peer records exported for heartbeat verification.
///
/// This payload is intentionally separate from `PeerStoreStatus`: nodeboard
/// and website surfaces should keep using aggregate summaries, while the
/// centralized coordination server can verify each signed descriptor before
/// accepting peer-discovery claims. Records contain node-level discovery
/// metadata only. They must never include client IPs, route ids, encrypted
/// payloads, receiver identities, DNS contents, voucher secrets, private keys,
/// wallet-level traffic, or plaintext.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreSignedPeerRecordsStatus {
    /// Unix timestamp when this signed snapshot was generated.
    pub generated_at: u64,
    /// Stable source label for downstream ingestion.
    pub source: String,
    /// Total descriptors currently retained in this PeerStore.
    pub total_retained_records: usize,
    /// Retained descriptors that verify at `generated_at`.
    pub valid_signed_records: usize,
    /// Valid signed descriptors exported after applying `limit`.
    pub exported_signed_records: usize,
    /// Export limit applied to the signed record snapshot.
    pub limit: Option<usize>,
    /// Verifiable signed descriptor snapshot.
    pub records: NodeBootstrapSnapshot,
    /// Explicit verification rule for the central server and future agents.
    pub verification_rule: String,
    /// Explicit privacy boundary for downstream consumers.
    pub privacy_boundary: String,
}

/// Combined peer store status payload for nodeboard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreStatus {
    /// Point-in-time peer counts.
    pub snapshot: PeerStoreSnapshot,
    /// Cumulative runtime counters.
    pub runtime: PeerStoreRuntimeStats,
    /// Aggregate blind relay runtime quality for dashboards and runbooks.
    pub blind_relay_quality: PeerStoreBlindRelayQualityStatus,
    /// Bounded privacy-safe proof history for recent two-hop relay checks.
    ///
    /// This lets dashboards show repeated protocol evidence instead of a
    /// single ready bit, while preserving the blind relay privacy boundary.
    pub two_hop_path_proof_history: PeerStoreTwoHopPathProofHistory,
    /// Configured maximum peer count.
    pub max_peers: Option<usize>,
    /// Recent privacy-safe discovery control-plane audit events.
    pub recent_audit_events: Vec<PeerStoreAuditEvent>,
    /// Recent privacy-safe peer discovery lifecycle events.
    ///
    /// These events are meant for nodeboard/app surfaces that need to show
    /// concrete peer discovery motion while preserving the blind-node privacy
    /// invariant. Rows contain only short node prefixes and reason buckets.
    pub recent_peer_events: Vec<PeerStorePeerEvent>,
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
    /// Privacy-safe per-peer health summary for nodeboard security panels.
    ///
    /// Rows use only node-level descriptor/runtime buckets and short prefixes.
    /// They never include route ids, endpoint URLs, encrypted blobs, receiver
    /// identities, client IPs, destinations, DNS contents, voucher secrets,
    /// private keys, wallet-level traffic, or plaintext content.
    pub peer_health_summary: PeerStorePeerHealthStatus,
    /// Privacy-safe quorum readiness for peer discovery.
    ///
    /// This is not chain consensus. It is an operator-facing readiness summary
    /// derived from verified descriptors, route candidates, and restart
    /// recovery status. It never exposes full node ids, endpoint URLs, route
    /// ids, encrypted payloads, receiver identities, client IPs, destinations,
    /// DNS contents, voucher secrets, private keys, wallet-level traffic, or
    /// plaintext.
    pub peer_quorum: PeerStorePeerQuorumStatus,
    /// Product-facing aggregate story for app/nodeboard/website surfaces.
    ///
    /// This is derived from existing discovery status objects and must remain
    /// aggregate-only: no full node ids, endpoint URLs, route ids, encrypted
    /// payloads, receiver identities, client IPs, destinations, DNS contents,
    /// voucher secrets, private keys, wallet-level traffic, or plaintext.
    pub network_story: PeerStoreNetworkStoryStatus,
}

/// Privacy-safe peer quorum readiness for future multi-hop work.
///
/// The word "quorum" here is intentionally scoped to this node's verified peer
/// view. It is not public-chain consensus, not ledger finality, and not a vote
/// over user traffic. The goal is to help nodeboard and automated runbooks tell
/// whether a node has enough fresh routeable peer state to continue with
/// encrypted relay and future onion-shaped path work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStorePeerQuorumStatus {
    /// Unix timestamp when this summary was generated.
    pub generated_at: u64,
    /// Stable bucket: disabled, forming, peer_view_ready, route_ready, or attention.
    pub status: String,
    /// Whether the local verified peer view meets the minimum readiness gates.
    pub quorum_ready: bool,
    /// Minimum valid descriptors required by this readiness gate.
    pub min_valid_peers: usize,
    /// Minimum routeable chat relay candidates required by this readiness gate.
    pub min_routeable_chat_relays: usize,
    /// Valid descriptors at generation time.
    pub valid_peers: usize,
    /// Healthy valid descriptors at generation time.
    pub healthy_peers: usize,
    /// Stale-but-valid descriptors at generation time.
    pub stale_peers: usize,
    /// Routeable encrypted chat relay candidates.
    pub routeable_chat_relays: usize,
    /// Routeable future onion middle-hop candidates.
    pub routeable_onion_middle_hops: usize,
    /// Healthy valid descriptor percentage, rounded down.
    pub healthy_ratio_percent: u8,
    /// Whether peer cache or seed recovery is configured for restart survival.
    pub restart_recovery_configured: bool,
    /// Whether discovery stability says the relay foundation is ready.
    pub relay_foundation_ready: bool,
    /// Short operator-facing detail with aggregate counts only.
    pub detail: String,
    /// Privacy-safe next action for nodeboard / AI runbooks.
    pub next_action: String,
    /// Explicit privacy boundary for downstream UI and API consumers.
    pub privacy_boundary: String,
}

/// Product-facing aggregate discovery readiness story.
///
/// This is the compact state a user or investor can understand: whether the
/// node has discovered other protocol nodes, whether routeable encrypted-chat
/// relay peers exist, whether a future two-hop onion-shaped path can be planned,
/// and whether restart recovery is configured. It deliberately reuses only
/// aggregate node-control-plane status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreNetworkStoryStatus {
    /// Unix timestamp when the story was generated.
    pub generated_at: u64,
    /// Stable bucket: disabled, discovering, peer_view_ready, relay_ready,
    /// onion_ready, or attention.
    pub status: String,
    /// Short display headline for dashboards and website stats.
    pub headline: String,
    /// Operator-facing detail with aggregate counts only.
    pub detail: String,
    /// Total retained node descriptors.
    pub discovered_nodes: usize,
    /// Valid node descriptors at generation time.
    pub valid_nodes: usize,
    /// Retained descriptors advertising chat relay capability.
    pub chat_relay_nodes: usize,
    /// Retained descriptors advertising future onion middle-hop capability.
    pub onion_middle_nodes: usize,
    /// Health-ranked chat relay route candidates with public endpoints.
    pub routeable_chat_relays: usize,
    /// Health-ranked onion middle-hop route candidates with public endpoints.
    pub routeable_onion_middle_hops: usize,
    /// Whether a one-hop encrypted chat relay path can be planned.
    pub chat_single_hop_ready: bool,
    /// Whether a two-hop onion-shaped path can be planned.
    pub chat_two_hop_onion_ready: bool,
    /// Whether peer cache or seed recovery is configured for restart survival.
    pub restart_recovery_configured: bool,
    /// Whether discovery stability says the relay foundation is ready.
    pub relay_foundation_ready: bool,
    /// Explicit privacy boundary for downstream UI and API consumers.
    pub privacy_boundary: String,
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
    expired_degraded_at: Option<u64>,
}

#[derive(Debug, Clone, Default)]
struct PeerRouteHealth {
    success_count: u64,
    failure_count: u64,
    consecutive_failures: u64,
    last_success_at: Option<u64>,
    last_failure_at: Option<u64>,
    last_failure_reason: Option<String>,
    quarantine_count: u64,
    quarantine_until: Option<u64>,
    last_quarantine_at: Option<u64>,
    last_quarantine_reason: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct PeerRelayProtectionHealth {
    rejection_count: u64,
    quarantine_count: u64,
    quarantine_until: Option<u64>,
    last_rejection_at: Option<u64>,
    last_rejection_reason: Option<String>,
    last_quarantine_at: Option<u64>,
    last_quarantine_reason: Option<String>,
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
// Peer health summary
// ============================================

/// Privacy-safe per-peer health row for nodeboard security and capacity pages.
///
/// This row is intentionally diagnostic-only. It joins signed descriptor
/// freshness, local gossip/import observation buckets, node-to-node route
/// counters, and relay protection buckets. It must never contain route ids,
/// full node ids, endpoint URLs, encrypted blobs, receiver identities, client
/// IPs, destinations, DNS contents, voucher secrets, private keys,
/// wallet-level traffic, or plaintext content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStorePeerHealth {
    /// Short node id prefix for operator debugging without exposing full keys.
    pub node_id_prefix: String,
    /// Overall node-level health bucket: healthy, stale, degraded, failing,
    /// quarantined, or expired.
    pub health: String,
    /// Signed descriptor health bucket: healthy, stale, or expired.
    pub descriptor_health: String,
    /// Last import/source bucket such as gossip_snapshot, gossip_announce,
    /// cache, file, url, self, or unknown.
    pub source: String,
    /// Last successful gossip/import observation for this peer, when the latest
    /// accepted source was live gossip.
    pub last_successful_gossip_at: Option<u64>,
    /// Age of `last_successful_gossip_at`.
    pub last_successful_gossip_age_seconds: Option<u64>,
    /// Last time this process accepted or refreshed this peer descriptor from
    /// any privacy-safe source.
    pub last_seen_at: u64,
    /// Age of the last accepted/refreshed descriptor observation.
    pub last_seen_age_seconds: u64,
    /// Route health bucket from opaque node-to-node forwarding attempts.
    pub route_health: String,
    /// Routeability bucket derived from fresh opaque probe/forward evidence.
    ///
    /// Stable values are `unknown`, `reachable`, `unreachable`, `stale`, and
    /// `quarantined`. This is endpoint-readiness evidence only; it never
    /// exposes endpoint URLs, route ids, encrypted payloads, or user metadata.
    pub routeability_state: String,
    /// Whether this peer has fresh successful routeability evidence.
    pub routeability_ready: bool,
    /// Last opaque routeability probe or forward timestamp.
    pub last_routeability_probe_at: Option<u64>,
    /// Age of the last opaque routeability probe or forward timestamp.
    pub last_routeability_probe_age_seconds: Option<u64>,
    /// Successful opaque node-to-node forwards to this peer.
    pub route_success_count: u64,
    /// Failed opaque node-to-node forwards to this peer.
    pub route_failure_count: u64,
    /// Consecutive failed opaque node-to-node forwards since last success.
    pub route_consecutive_failures: u64,
    /// Last successful opaque node-to-node forward timestamp.
    pub last_route_success_at: Option<u64>,
    /// Last failed opaque node-to-node forward timestamp.
    pub last_route_failure_at: Option<u64>,
    /// Coarse reason bucket for the last opaque route failure.
    pub last_route_failure_reason: Option<String>,
    /// Whether this peer is temporarily suppressed as a next hop after
    /// repeated opaque route failures.
    pub route_quarantined: bool,
    /// Remaining route-level suppression window in seconds, when active.
    pub route_quarantine_remaining_seconds: Option<u64>,
    /// Number of route-level quarantine windows started for this peer.
    pub route_quarantine_count: u64,
    /// Last route-level quarantine timestamp.
    pub last_route_quarantine_at: Option<u64>,
    /// Coarse reason bucket for the last route-level quarantine.
    pub last_route_quarantine_reason: Option<String>,
    /// Local relay-protection rejection count for this peer as previous hop.
    pub relay_rejection_count: u64,
    /// Number of short local relay-protection quarantines started.
    pub relay_quarantine_count: u64,
    /// Whether this peer is currently quarantined as a previous hop.
    pub relay_quarantined: bool,
    /// Remaining local quarantine time in seconds, when quarantined.
    pub relay_quarantine_remaining_seconds: Option<u64>,
    /// Last relay-protection rejection timestamp.
    pub last_relay_rejection_at: Option<u64>,
    /// Coarse reason bucket for the last relay-protection rejection.
    pub last_relay_rejection_reason: Option<String>,
    /// Last relay-protection quarantine timestamp.
    pub last_relay_quarantine_at: Option<u64>,
    /// Coarse reason bucket for the last relay-protection quarantine.
    pub last_relay_quarantine_reason: Option<String>,
}

/// Aggregate peer health summary exposed in discovery status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStorePeerHealthStatus {
    /// Unix timestamp when the summary was generated.
    pub generated_at: u64,
    /// Total retained peer descriptors considered by this summary.
    pub total_peers: usize,
    /// Peers with no current descriptor, route, or relay-protection attention bucket.
    pub healthy_peers: usize,
    /// Peers that need operator attention but are not hard failing.
    pub degraded_peers: usize,
    /// Peers with repeated recent route failures.
    pub failing_peers: usize,
    /// Peers currently under short local relay-protection quarantine.
    pub quarantined_peers: usize,
    /// Privacy-safe per-peer health rows, capped for nodeboard.
    pub peers: Vec<PeerStorePeerHealth>,
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
    /// Endpoint routeability state derived from fresh probe or forward evidence.
    ///
    /// A signed descriptor with a public endpoint is only `unknown` until this
    /// node observes a successful blind-relay probe or real opaque forward.
    /// Public readiness surfaces must count only `routeability_ready=true`.
    pub routeability_state: String,
    /// Whether this candidate is safe to count as currently routeable.
    pub routeability_ready: bool,
    /// Last opaque routeability probe or forward timestamp.
    pub last_routeability_probe_at: Option<u64>,
    /// Age of the last opaque routeability probe or forward timestamp.
    pub last_routeability_probe_age_seconds: Option<u64>,
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
    /// Whether route selection is temporarily suppressing this peer after
    /// repeated opaque next-hop failures.
    pub route_quarantined: bool,
    /// Remaining route-suppression window in seconds, when active.
    pub route_quarantine_remaining_seconds: Option<u64>,
    /// Number of route-level quarantine windows started for this peer.
    pub route_quarantine_count: u64,
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
    /// Privacy-safe previews for controlled routes that future relay layers can use.
    pub planned_paths: PeerStoreRoutePathStatus,
}

// ============================================
// Route path planning summary
// ============================================

/// Privacy-safe hop preview for a planned route path.
///
/// This row intentionally mirrors only route-control metadata. It never
/// contains full node ids, endpoint URLs, route ids, encrypted blobs, receiver
/// identities, client IPs, destinations, DNS contents, voucher secrets, private
/// keys, or wallet-level traffic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRoutePathHop {
    /// Hop position in the planned route, starting at zero.
    pub hop_index: usize,
    /// Capability required for this hop.
    pub capability: String,
    /// Short node id prefix for operator debugging without exposing full keys.
    pub node_id_prefix: String,
    /// Route score inherited from the candidate scorer.
    pub score: i64,
    /// Descriptor health bucket at planning time.
    pub health: String,
    /// Local route health bucket at planning time.
    pub route_health: String,
    /// Age of the last observation at status generation time.
    pub last_seen_age_seconds: u64,
    /// Remaining descriptor TTL in seconds.
    pub ttl_remaining_seconds: Option<u64>,
    /// Optional region hint from signed descriptor policy.
    pub region: Option<String>,
}

/// Privacy-safe preview for one controlled route plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRoutePathPlan {
    /// Stable plan label for nodeboard and health checks.
    pub label: String,
    /// Required capability sequence for this plan.
    pub required_capabilities: Vec<String>,
    /// Whether every requested hop was selected.
    pub complete: bool,
    /// Number of selected hops.
    pub hop_count: usize,
    /// Selected hop previews, with no full keys or endpoints.
    pub hops: Vec<PeerStoreRoutePathHop>,
}

/// Controlled route previews for future multi-hop relay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRoutePathStatus {
    /// One-hop encrypted chat relay path.
    pub chat_single_hop: PeerStoreRoutePathPlan,
    /// Two-hop path shaped for future onion relay: middle hop then chat relay.
    pub chat_two_hop_onion_ready: PeerStoreRoutePathPlan,
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
    expired_degraded: AtomicU64,
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
    blind_relay_loop_detected: AtomicU64,
    blind_relay_replay_dropped: AtomicU64,
    blind_relay_timestamp_rejected: AtomicU64,
    blind_relay_rate_limited: AtomicU64,
    blind_relay_quarantined: AtomicU64,
    blind_relay_quarantine_started: AtomicU64,
    blind_relay_retry_attempted: AtomicU64,
    blind_relay_retry_succeeded: AtomicU64,
    blind_relay_retry_exhausted: AtomicU64,
    blind_relay_probe_attempted: AtomicU64,
    blind_relay_probe_succeeded: AtomicU64,
    blind_relay_probe_failed: AtomicU64,
    last_blind_relay_probe_at: AtomicU64,
    blind_relay_two_hop_probe_attempted: AtomicU64,
    blind_relay_two_hop_probe_succeeded: AtomicU64,
    blind_relay_two_hop_probe_failed: AtomicU64,
    last_blind_relay_two_hop_probe_at: AtomicU64,
    last_blind_relay_accepted_at: AtomicU64,
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
            expired_degraded: AtomicU64::new(0),
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
            blind_relay_loop_detected: AtomicU64::new(0),
            blind_relay_replay_dropped: AtomicU64::new(0),
            blind_relay_timestamp_rejected: AtomicU64::new(0),
            blind_relay_rate_limited: AtomicU64::new(0),
            blind_relay_quarantined: AtomicU64::new(0),
            blind_relay_quarantine_started: AtomicU64::new(0),
            blind_relay_retry_attempted: AtomicU64::new(0),
            blind_relay_retry_succeeded: AtomicU64::new(0),
            blind_relay_retry_exhausted: AtomicU64::new(0),
            blind_relay_probe_attempted: AtomicU64::new(0),
            blind_relay_probe_succeeded: AtomicU64::new(0),
            blind_relay_probe_failed: AtomicU64::new(0),
            last_blind_relay_probe_at: AtomicU64::new(0),
            blind_relay_two_hop_probe_attempted: AtomicU64::new(0),
            blind_relay_two_hop_probe_succeeded: AtomicU64::new(0),
            blind_relay_two_hop_probe_failed: AtomicU64::new(0),
            last_blind_relay_two_hop_probe_at: AtomicU64::new(0),
            last_blind_relay_accepted_at: AtomicU64::new(0),
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
            expired_degraded: self.expired_degraded.load(Ordering::Relaxed),
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
                loop_detected: self.blind_relay_loop_detected.load(Ordering::Relaxed),
                replay_dropped: self.blind_relay_replay_dropped.load(Ordering::Relaxed),
                timestamp_rejected: self.blind_relay_timestamp_rejected.load(Ordering::Relaxed),
                rate_limited: self.blind_relay_rate_limited.load(Ordering::Relaxed),
                quarantined: self.blind_relay_quarantined.load(Ordering::Relaxed),
                quarantine_started: self.blind_relay_quarantine_started.load(Ordering::Relaxed),
                retry_attempted: self.blind_relay_retry_attempted.load(Ordering::Relaxed),
                retry_succeeded: self.blind_relay_retry_succeeded.load(Ordering::Relaxed),
                retry_exhausted: self.blind_relay_retry_exhausted.load(Ordering::Relaxed),
                probe_attempted: self.blind_relay_probe_attempted.load(Ordering::Relaxed),
                probe_succeeded: self.blind_relay_probe_succeeded.load(Ordering::Relaxed),
                probe_failed: self.blind_relay_probe_failed.load(Ordering::Relaxed),
                last_probe_at: Self::optional_ts(
                    self.last_blind_relay_probe_at.load(Ordering::Relaxed),
                ),
                two_hop_probe_attempted: self
                    .blind_relay_two_hop_probe_attempted
                    .load(Ordering::Relaxed),
                two_hop_probe_succeeded: self
                    .blind_relay_two_hop_probe_succeeded
                    .load(Ordering::Relaxed),
                two_hop_probe_failed: self
                    .blind_relay_two_hop_probe_failed
                    .load(Ordering::Relaxed),
                last_two_hop_probe_at: Self::optional_ts(
                    self.last_blind_relay_two_hop_probe_at
                        .load(Ordering::Relaxed),
                ),
                last_accepted_at: Self::optional_ts(
                    self.last_blind_relay_accepted_at.load(Ordering::Relaxed),
                ),
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
    relay_protection_health: RwLock<HashMap<[u8; 32], PeerRelayProtectionHealth>>,
    max_peers: RwLock<Option<usize>>,
    counters: PeerStoreCounters,
    audit_events: RwLock<VecDeque<PeerStoreAuditEvent>>,
    peer_events: RwLock<VecDeque<PeerStorePeerEvent>>,
    two_hop_path_proof_events: RwLock<VecDeque<PeerStoreTwoHopPathProofEvent>>,
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
            relay_protection_health: RwLock::new(HashMap::new()),
            max_peers: RwLock::new(None),
            counters: PeerStoreCounters::new(),
            audit_events: RwLock::new(VecDeque::with_capacity(MAX_AUDIT_EVENTS)),
            peer_events: RwLock::new(VecDeque::with_capacity(MAX_PEER_EVENTS)),
            two_hop_path_proof_events: RwLock::new(VecDeque::with_capacity(
                MAX_TWO_HOP_PATH_PROOF_EVENTS,
            )),
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

    /// Records startup discovery readiness without exposing endpoint values.
    ///
    /// `detail` must be a stable bucket list such as
    /// `missing=peer_cache_path,seed_endpoints`; callers must not include raw
    /// file paths, public endpoints, seed URLs, peer IDs, client IPs,
    /// destinations, DNS contents, packet payloads, chat plaintext, voucher
    /// secrets, private keys, or wallet-level traffic.
    pub fn record_startup_self_check(
        &self,
        now: u64,
        status_bucket: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let status_bucket = status_bucket.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.startup_self_check_status = Some(status_bucket.clone());
            status.startup_self_check_detail = Some(detail.clone());
            status.startup_self_check_at = Some(now);
        }
        let outcome = match status_bucket.as_str() {
            "ready" => "accepted",
            "skipped" => "ignored",
            _ => "warning",
        };
        self.record_audit_event(now, "startup_self_check", outcome, detail);
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
            if matches!(source_kind.as_str(), "cache" | "cache_backup") {
                status.last_cache_load_source = Some(source_kind.clone());
                status.last_cache_load_status = Some(source_status.clone());
                status.last_cache_load_detail = Some(detail.clone());
                status.last_cache_load_at = Some(now);
            }
        }
        self.record_audit_event(
            now,
            "bootstrap_source",
            source_status,
            format!("kind={source_kind} {detail}"),
        );
    }

    /// Records peer-cache startup load status without exposing local paths.
    ///
    /// This method exists for callers that need to update cache recovery
    /// evidence without changing the generic bootstrap source state. Most
    /// bootstrap imports should continue using `record_bootstrap_source()`,
    /// which already mirrors cache/cache_backup events into these fields.
    pub fn record_peer_cache_load_status(
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
            status.last_cache_load_source = Some(source_kind.clone());
            status.last_cache_load_status = Some(source_status.clone());
            status.last_cache_load_detail = Some(detail.clone());
            status.last_cache_load_at = Some(now);
        }
        self.record_audit_event(
            now,
            "peer_cache_load",
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
        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
        let source = source.into();
        if descriptor.verify_at(now).is_err() {
            self.record_peer_event(
                now,
                "peer_rejected",
                "rejected",
                source,
                &node_id,
                Some(incoming_sequence),
                Some("verification_failed"),
            );
            return Err(PeerStoreError::VerificationFailed);
        }

        let mut peers = self.peers.write();

        let is_existing_peer = if let Some(existing) = peers.get(&node_id) {
            let current = existing.sequence();
            if incoming_sequence < current {
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("stale_sequence"),
                );
                return Err(PeerStoreError::StaleSequence {
                    current,
                    incoming: incoming_sequence,
                });
            }
            if incoming_sequence == current {
                drop(peers);
                self.record_peer_runtime(&descriptor, now, source.clone(), false);
                self.record_peer_event(
                    now,
                    "peer_refreshed",
                    "ignored",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("same_sequence"),
                );
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
                drop(peers);
                self.record_peer_event(
                    now,
                    "peer_rejected",
                    "rejected",
                    source,
                    &node_id,
                    Some(incoming_sequence),
                    Some("capacity_exceeded"),
                );
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }

        peers.insert(node_id, descriptor.clone());
        drop(peers);
        self.record_peer_runtime(&descriptor, now, source.clone(), true);
        self.record_peer_event(
            now,
            if is_existing_peer {
                "peer_upgraded"
            } else {
                "peer_inserted"
            },
            "accepted",
            source,
            &node_id,
            Some(incoming_sequence),
            None,
        );
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
                expired_degraded_at: None,
            });
        entry.source = source;
        entry.last_seen_at = now;
        entry.last_sequence = descriptor.sequence();
        entry.expired_degraded_at = None;
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

    /// Imports descriptors from a local peer-cache snapshot.
    ///
    /// Unlike gossip/bootstrap imports, peer-cache recovery may retain
    /// expired-but-authentic signed descriptors so operators do not lose local
    /// peer history after restart. Retained expired records are never counted
    /// as valid, exported to public gossip, or selected as routeable peers
    /// until a fresh signed descriptor is received and `verify_at(now)` passes.
    pub fn load_peer_cache_snapshot_from_source(
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
            if descriptor.verify_at(now).is_ok() {
                match self.upsert_verified_from_source(descriptor.clone(), now, source.clone()) {
                    Ok(true) => report.inserted += 1,
                    Ok(false) => report.unchanged += 1,
                    Err(PeerStoreError::StaleSequence { .. }) => report.stale += 1,
                    Err(PeerStoreError::VerificationFailed)
                    | Err(PeerStoreError::CapacityExceeded { .. }) => report.rejected += 1,
                }
                continue;
            }

            match self.retain_signed_expired_from_cache(descriptor.clone(), now, source.clone()) {
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

    fn retain_signed_expired_from_cache(
        &self,
        descriptor: SignedNodeDescriptor,
        now: u64,
        source: String,
    ) -> Result<bool, PeerStoreError> {
        if descriptor.verify_signature().is_err() || descriptor.descriptor.is_valid_at(now) {
            return Err(PeerStoreError::VerificationFailed);
        }

        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
        let mut peers = self.peers.write();
        let existing_sequence = peers.get(&node_id).map(SignedNodeDescriptor::sequence);

        if let Some(current) = existing_sequence {
            if incoming_sequence < current {
                return Err(PeerStoreError::StaleSequence {
                    current,
                    incoming: incoming_sequence,
                });
            }
        } else if let Some(max_peers) = self.max_peers() {
            if peers.len() >= max_peers {
                return Err(PeerStoreError::CapacityExceeded { max_peers });
            }
        }

        let changed = existing_sequence != Some(incoming_sequence);
        if changed {
            peers.insert(node_id, descriptor.clone());
        }
        drop(peers);

        {
            let mut metadata = self.peer_runtime.write();
            let entry = metadata
                .entry(node_id)
                .or_insert_with(|| PeerRuntimeMetadata {
                    source: source.clone(),
                    first_seen_at: now,
                    last_seen_at: now,
                    last_sequence: incoming_sequence,
                    imported_count: 0,
                    expired_degraded_at: Some(now),
                });
            entry.source = source.clone();
            entry.last_seen_at = now;
            entry.last_sequence = incoming_sequence;
            entry.expired_degraded_at = Some(now);
            if changed {
                entry.imported_count = entry.imported_count.saturating_add(1);
            }
        }

        self.record_peer_event(
            now,
            "peer_expired",
            "retained",
            source,
            &node_id,
            Some(incoming_sequence),
            Some("signature_valid_descriptor_expired"),
        );
        Ok(changed)
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
    /// id, full next-hop id, endpoint URL, exact encrypted blob bytes, and any
    /// payload-derived metadata remain outside PeerStore status.
    pub fn record_blind_relay_terminal(&self, now: u64, ttl_remaining: u8, blob_bytes: usize) {
        self.counters
            .blind_relay_received
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .blind_relay_terminal
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_accepted_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_terminal",
            "accepted",
            format!(
                "ttl_remaining={ttl_remaining} encrypted_blob_size_bucket={}",
                Self::blind_relay_blob_size_bucket(blob_bytes)
            ),
        );
    }

    fn blind_relay_blob_size_bucket(blob_bytes: usize) -> &'static str {
        match blob_bytes {
            0..=4_096 => "lte_4kb",
            4_097..=65_536 => "lte_64kb",
            65_537..=262_144 => "lte_256kb",
            262_145..=1_048_576 => "lte_1mb",
            _ => "gt_1mb",
        }
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
            .last_blind_relay_accepted_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_forward",
            "accepted",
            format!("ttl_remaining={ttl_remaining} encrypted_blob_size_bucket=opaque"),
        );
    }

    /// Records a scheduled retry after a transient blind relay forward failure.
    ///
    /// This is intentionally aggregate-only. The reason bucket may be `http_503`,
    /// `blind_relay_request_timeout`, or similar transport state, but callers
    /// must not pass route ids, full peer ids, endpoint URLs, encrypted blobs,
    /// receiver identities, client IPs, DNS contents, voucher secrets, or
    /// payload-derived metadata.
    pub fn record_blind_relay_retry_attempt(&self, now: u64, reason: impl AsRef<str>) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_retry_attempted
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_retry",
            "scheduled",
            format!("reason_bucket={reason}"),
        );
    }

    /// Records that a blind relay forward succeeded after retrying.
    pub fn record_blind_relay_retry_succeeded(&self, now: u64, attempts: usize) {
        self.counters
            .blind_relay_retry_succeeded
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_accepted_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_retry",
            "accepted",
            format!("attempts={attempts}"),
        );
    }

    /// Records that retry attempts were exhausted before the final forward failed.
    pub fn record_blind_relay_retry_exhausted(
        &self,
        now: u64,
        attempts: usize,
        reason: impl AsRef<str>,
    ) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_retry_exhausted
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_retry",
            "rejected",
            format!("attempts={attempts} reason_bucket={reason}"),
        );
    }

    /// Records the result of a low-frequency synthetic blind relay route probe.
    ///
    /// Probe traffic is operator readiness evidence, not user traffic. It must
    /// never be added to encrypted message totals, packet totals, payload byte
    /// totals, billing, rewards, or user-facing usage claims. The detail is a
    /// stable reason bucket only; callers must not pass endpoint URLs, full
    /// node ids, route ids, encrypted blobs, receiver identities, client IPs,
    /// DNS contents, destinations, voucher secrets, wallet-level metadata, or
    /// plaintext.
    pub fn record_blind_relay_probe_result(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
    ) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_probe_attempted
            .fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.counters
                .blind_relay_probe_succeeded
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters
                .blind_relay_probe_failed
                .fetch_add(1, Ordering::Relaxed);
        }
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_probe_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_probe",
            if accepted { "accepted" } else { "rejected" },
            format!("reason_bucket={reason}"),
        );
    }

    /// Records a low-frequency synthetic two-hop blind relay path proof.
    ///
    /// This is protocol-health evidence only, not user traffic. Callers must
    /// pass stable reason buckets only. Never include path members, endpoint
    /// URLs, route ids, node ids, encrypted blobs, receiver identities, client
    /// IPs, DNS contents, destinations, Memory Chain plaintext, voucher
    /// secrets, private keys, wallet-level traffic, or social graph metadata.
    pub fn record_blind_relay_two_hop_probe_result(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
    ) {
        self.record_blind_relay_two_hop_probe_result_with_context(
            now, accepted, reason, 0, 0, 2, 1,
        );
    }

    /// Records a two-hop synthetic proof with privacy-safe route quality context.
    ///
    /// The context is intentionally bucketed before it enters history or audit
    /// output. It must never include node ids, endpoints, route ids, encrypted
    /// blobs, receiver identities, client IPs, DNS contents, domains, URLs,
    /// Memory Chain plaintext, voucher secrets, wallet-level traffic, or
    /// social graph metadata.
    pub fn record_blind_relay_two_hop_probe_result_with_context(
        &self,
        now: u64,
        accepted: bool,
        reason: impl AsRef<str>,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
        entry_ttl: u8,
        onward_ttl: u8,
    ) {
        let reason = reason.as_ref();
        self.counters
            .blind_relay_two_hop_probe_attempted
            .fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.counters
                .blind_relay_two_hop_probe_succeeded
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters
                .blind_relay_two_hop_probe_failed
                .fetch_add(1, Ordering::Relaxed);
        }
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.counters
            .last_blind_relay_two_hop_probe_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(
            now,
            "blind_relay_two_hop_probe",
            if accepted { "accepted" } else { "rejected" },
            format!(
                "reason_bucket={}; middle_candidates={}; terminal_candidates={}; ttl_shape={}",
                Self::two_hop_path_proof_reason_bucket(reason),
                Self::two_hop_candidate_count_bucket(middle_candidate_count),
                Self::two_hop_candidate_count_bucket(terminal_candidate_count),
                Self::two_hop_ttl_shape(entry_ttl, onward_ttl),
            ),
        );
        self.record_two_hop_path_proof_event(
            now,
            accepted,
            reason,
            middle_candidate_count,
            terminal_candidate_count,
            entry_ttl,
            onward_ttl,
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
            "self_loop" | "route_loop" => {
                self.counters
                    .blind_relay_loop_detected
                    .fetch_add(1, Ordering::Relaxed);
            }
            "duplicate_route" => {
                self.counters
                    .blind_relay_replay_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
            "timestamp_expired" | "timestamp_in_future" => {
                self.counters
                    .blind_relay_timestamp_rejected
                    .fetch_add(1, Ordering::Relaxed);
            }
            "rate_limited" => {
                self.counters
                    .blind_relay_rate_limited
                    .fetch_add(1, Ordering::Relaxed);
            }
            "quarantined" => {
                self.counters
                    .blind_relay_quarantined
                    .fetch_add(1, Ordering::Relaxed);
            }
            reason if reason.starts_with("http_") || reason.starts_with("blind_relay_request_") => {
                self.counters
                    .blind_relay_forward_failed
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        self.record_audit_event(now, "blind_relay_forward", "rejected", reason.to_string());
    }

    /// Records that the local blind relay abuse guard started a short
    /// previous-hop quarantine.
    ///
    /// The detail is a stable bucket such as `rate_limit` or
    /// `failure_threshold`; callers must not pass node ids, route ids, endpoint
    /// values, encrypted blobs, wallet ids, or payload-derived details.
    pub fn record_blind_relay_quarantine_started(&self, now: u64, detail: impl Into<String>) {
        self.counters
            .blind_relay_quarantine_started
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .last_blind_relay_at
            .store(now, Ordering::Relaxed);
        self.record_audit_event(now, "blind_relay_quarantine", "limited", detail);
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
        health.quarantine_until = None;
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
        let starts_new_quarantine = health.consecutive_failures
            >= PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD
            && health
                .quarantine_until
                .map(|quarantine_until| now >= quarantine_until)
                .unwrap_or(true);
        if starts_new_quarantine {
            health.quarantine_count = health.quarantine_count.saturating_add(1);
            health.quarantine_until = Some(now.saturating_add(PEER_ROUTE_FAILURE_QUARANTINE_SECS));
            health.last_quarantine_at = Some(now);
            health.last_quarantine_reason = Some("consecutive_route_failures".to_string());
        }
        self.record_audit_event(
            now,
            "blind_relay_route_health",
            "rejected",
            format!(
                "node_prefix={} result=failure reason={reason} consecutive_failures={}",
                hex::encode(&node_id[..4]),
                health.consecutive_failures
            ),
        );
        if starts_new_quarantine {
            self.record_audit_event(
                now,
                "blind_relay_route_quarantine",
                "limited",
                format!(
                    "node_prefix={} reason_bucket=consecutive_route_failures duration_seconds={}",
                    hex::encode(&node_id[..4]),
                    PEER_ROUTE_FAILURE_QUARANTINE_SECS
                ),
            );
        }
    }

    /// Records a relay-protection rejection for a previous-hop peer.
    ///
    /// This is peer-level control-plane health only. Status exposes a short
    /// node prefix, counters, and coarse reason buckets; it never includes full
    /// node ids, route ids, endpoint URLs, encrypted blobs, receiver identities,
    /// client IPs, DNS contents, voucher secrets, private keys, wallet-level
    /// traffic, or payload-derived details.
    pub fn record_peer_relay_rejection(
        &self,
        node_id: &[u8; 32],
        now: u64,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        let mut relay_health = self.relay_protection_health.write();
        let health = relay_health.entry(*node_id).or_default();
        health.rejection_count = health.rejection_count.saturating_add(1);
        health.last_rejection_at = Some(now);
        health.last_rejection_reason = Some(reason.clone());
        self.record_audit_event(
            now,
            "blind_relay_peer_protection",
            "rejected",
            format!(
                "node_prefix={} reason_bucket={reason}",
                hex::encode(&node_id[..4])
            ),
        );
    }

    /// Records that relay protection started a short quarantine for a peer.
    ///
    /// `quarantine_until` is a local control-plane timestamp. It is exposed as
    /// remaining seconds only; callers must not pass route ids, endpoint URLs,
    /// encrypted blobs, receivers, client traffic metadata, wallet ids, or
    /// plaintext-derived values into this method.
    pub fn record_peer_relay_quarantine_started(
        &self,
        node_id: &[u8; 32],
        now: u64,
        quarantine_until: u64,
        reason: impl Into<String>,
    ) {
        let reason = reason.into();
        let mut relay_health = self.relay_protection_health.write();
        let health = relay_health.entry(*node_id).or_default();
        health.quarantine_count = health.quarantine_count.saturating_add(1);
        health.quarantine_until = Some(quarantine_until);
        health.last_quarantine_at = Some(now);
        health.last_quarantine_reason = Some(reason.clone());
        self.record_audit_event(
            now,
            "blind_relay_peer_quarantine",
            "limited",
            format!(
                "node_prefix={} reason_bucket={reason}",
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

    fn record_two_hop_path_proof_event(
        &self,
        now: u64,
        accepted: bool,
        reason: &str,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
        entry_ttl: u8,
        onward_ttl: u8,
    ) {
        let mut events = self.two_hop_path_proof_events.write();
        if events.len() >= MAX_TWO_HOP_PATH_PROOF_EVENTS {
            events.pop_front();
        }
        events.push_back(PeerStoreTwoHopPathProofEvent {
            at: now,
            outcome: if accepted { "accepted" } else { "rejected" }.to_string(),
            reason_bucket: Self::two_hop_path_proof_reason_bucket(reason),
            evidence_mode: Self::two_hop_path_proof_evidence_mode(reason).to_string(),
            proof_scope: Self::two_hop_path_proof_scope(reason).to_string(),
            path_shape: "entry_middle_terminal".to_string(),
            hop_count: 2,
            path_policy: "distinct_middle_terminal".to_string(),
            middle_candidate_bucket: Self::two_hop_candidate_count_bucket(middle_candidate_count)
                .to_string(),
            terminal_candidate_bucket: Self::two_hop_candidate_count_bucket(
                terminal_candidate_count,
            )
            .to_string(),
            ttl_shape: Self::two_hop_ttl_shape(entry_ttl, onward_ttl),
        });
    }

    fn two_hop_path_proof_evidence_mode(reason: &str) -> &'static str {
        let bucket = Self::two_hop_path_proof_reason_bucket(reason);
        match bucket.as_str() {
            "onion_terminal_delivered" => "real_relay_traffic",
            _ => "synthetic_two_hop_control_probe",
        }
    }

    fn two_hop_path_proof_scope(reason: &str) -> &'static str {
        let bucket = Self::two_hop_path_proof_reason_bucket(reason);
        match bucket.as_str() {
            "onion_terminal_delivered" => "message_delivery",
            _ => "control_plane",
        }
    }

    fn two_hop_candidate_count_bucket(count: usize) -> &'static str {
        match count {
            0 => "none",
            1 => "one",
            2..=3 => "few",
            4..=8 => "healthy",
            _ => "deep",
        }
    }

    fn two_hop_ttl_shape(entry_ttl: u8, onward_ttl: u8) -> String {
        match (entry_ttl, onward_ttl) {
            (2, 1) => "entry_ttl_2_onward_ttl_1".to_string(),
            (entry, onward) => format!("entry_ttl_{entry}_onward_ttl_{onward}"),
        }
    }

    fn two_hop_path_proof_reason_bucket(reason: &str) -> String {
        match reason {
            "accepted" => "accepted".to_string(),
            "onion_terminal_delivered" => "onion_terminal_delivered".to_string(),
            "onion_ack_rejected" => "onion_ack_rejected".to_string(),
            "onion_ack_decode" => "onion_ack_decode".to_string(),
            "onion_kem_unavailable" => "onion_kem_unavailable".to_string(),
            "ack_rejected" => "ack_rejected".to_string(),
            "ack_decode" => "ack_decode".to_string(),
            "no_distinct_path" => "no_distinct_path".to_string(),
            "middle_missing_endpoint" => "middle_missing_endpoint".to_string(),
            "middle_invalid_endpoint" => "middle_invalid_endpoint".to_string(),
            value if value.starts_with("onion_http_") => "onion_http_error".to_string(),
            value if value.starts_with("http_") => "http_error".to_string(),
            value if value.starts_with("two_hop_onion_delivery_probe_") => {
                "onion_request_error".to_string()
            }
            value if value.starts_with("two_hop_blind_relay_probe_") => "request_error".to_string(),
            value if value.starts_with("two_hop_blind_relay_probe_request_") => {
                "request_error".to_string()
            }
            _ => "unknown".to_string(),
        }
    }

    fn two_hop_path_proof_history(&self, now: u64) -> PeerStoreTwoHopPathProofHistory {
        let events = self
            .two_hop_path_proof_events
            .read()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let attempted = events.len() as u64;
        let succeeded = events
            .iter()
            .filter(|event| event.outcome == "accepted")
            .count() as u64;
        let message_delivery_successes = events
            .iter()
            .filter(|event| event.outcome == "accepted" && event.proof_scope == "message_delivery")
            .count() as u64;
        let failed = attempted.saturating_sub(succeeded);
        let success_percent = if attempted == 0 {
            0
        } else {
            ((succeeded.saturating_mul(100)) / attempted).min(100) as u8
        };
        let latest = events.last();
        let latest_age_seconds = latest.map(|event| now.saturating_sub(event.at));
        let latest_success_age_seconds = events
            .iter()
            .rev()
            .find(|event| event.outcome == "accepted")
            .map(|event| now.saturating_sub(event.at));
        let latest_failure_age_seconds = events
            .iter()
            .rev()
            .find(|event| event.outcome == "rejected")
            .map(|event| now.saturating_sub(event.at));
        let latest_message_delivery_age_seconds = events
            .iter()
            .rev()
            .find(|event| event.outcome == "accepted" && event.proof_scope == "message_delivery")
            .map(|event| now.saturating_sub(event.at));
        let consecutive_successes = Self::count_trailing_two_hop_outcomes(&events, "accepted");
        let consecutive_failures = Self::count_trailing_two_hop_outcomes(&events, "rejected");
        let consecutive_message_delivery_successes =
            Self::count_trailing_two_hop_message_delivery_successes(&events);
        let reason_bucket_counts =
            Self::count_two_hop_event_buckets(&events, |event| event.reason_bucket.as_str());
        let failure_events = events
            .iter()
            .filter(|event| event.outcome == "rejected")
            .cloned()
            .collect::<Vec<_>>();
        let failure_reason_bucket_counts =
            Self::count_two_hop_event_buckets(&failure_events, |event| {
                event.reason_bucket.as_str()
            });
        let path_shape_counts =
            Self::count_two_hop_event_buckets(&events, |event| event.path_shape.as_str());
        let candidate_pool_counts = Self::count_two_hop_event_buckets(&events, |event| {
            if event.middle_candidate_bucket.is_empty()
                || event.terminal_candidate_bucket.is_empty()
                || event.middle_candidate_bucket == "none"
                || event.terminal_candidate_bucket == "none"
            {
                "incomplete"
            } else if event.middle_candidate_bucket == "one"
                || event.terminal_candidate_bucket == "one"
            {
                "thin"
            } else if event.middle_candidate_bucket == "few"
                || event.terminal_candidate_bucket == "few"
            {
                "forming"
            } else {
                "healthy"
            }
        });
        let ttl_shape_counts =
            Self::count_two_hop_event_buckets(&events, |event| event.ttl_shape.as_str());
        let proof_scope_counts = Self::count_two_hop_event_buckets(&events, |event| {
            if event.proof_scope.is_empty() {
                "unknown"
            } else {
                event.proof_scope.as_str()
            }
        });
        let recent_success_ready = latest
            .map(|event| {
                event.outcome == "accepted"
                    && now.saturating_sub(event.at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS
            })
            .unwrap_or(false);
        let message_delivery_ready = latest
            .map(|event| {
                event.outcome == "accepted"
                    && event.proof_scope == "message_delivery"
                    && now.saturating_sub(event.at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS
            })
            .unwrap_or(false);
        let recent_message_delivery_ready = latest_message_delivery_age_seconds
            .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
            .unwrap_or(false);
        let failure_streak_active = consecutive_failures > 0;
        let (status, proof_ready, next_action) = if attempted == 0 {
            (
                "forming",
                false,
                "wait for the first synthetic two-hop path proof",
            )
        } else if recent_success_ready {
            (
                "ready",
                true,
                "continue monitoring repeated two-hop path proof freshness",
            )
        } else if failure_streak_active {
            (
                "attention",
                false,
                "inspect recent routeability, middle-hop endpoint, and terminal-hop proof buckets",
            )
        } else if succeeded > 0 {
            (
                "stale",
                false,
                "wait for a fresh two-hop path proof or verify peer routeability gossip",
            )
        } else {
            (
                "idle",
                false,
                "wait for route candidates before advertising two-hop path proof readiness",
            )
        };
        let freshness_bucket = if attempted == 0 {
            "forming"
        } else if recent_success_ready {
            "fresh_success"
        } else if failure_streak_active {
            "recent_failure"
        } else if latest_success_age_seconds.is_some() {
            "stale_success"
        } else {
            "no_success"
        };

        PeerStoreTwoHopPathProofHistory {
            generated_at: now,
            status: status.to_string(),
            freshness_bucket: freshness_bucket.to_string(),
            proof_ready,
            recent_success_ready,
            message_delivery_ready,
            recent_message_delivery_ready,
            failure_streak_active,
            window_size: MAX_TWO_HOP_PATH_PROOF_EVENTS,
            retained_events: events.len(),
            attempted,
            succeeded,
            message_delivery_successes,
            failed,
            success_percent,
            latest_outcome: latest.map(|event| event.outcome.clone()),
            latest_reason_bucket: latest.map(|event| event.reason_bucket.clone()),
            latest_age_seconds,
            latest_success_age_seconds,
            latest_failure_age_seconds,
            latest_message_delivery_age_seconds,
            consecutive_successes,
            consecutive_failures,
            consecutive_message_delivery_successes,
            reason_bucket_counts,
            failure_reason_bucket_counts,
            path_shape_counts,
            candidate_pool_counts,
            ttl_shape_counts,
            proof_scope_counts,
            stale_after_seconds: PEER_ROUTEABILITY_STALE_AFTER_SECS,
            proof_scope: latest
                .map(|event| {
                    if event.proof_scope.is_empty() {
                        "unknown".to_string()
                    } else {
                        event.proof_scope.clone()
                    }
                })
                .unwrap_or_else(|| "none".to_string()),
            next_action: next_action.to_string(),
            events,
            privacy_invariant: "blind_nodes_route_only_opaque_ciphertext".to_string(),
            privacy_boundary: "bounded synthetic two-hop proof history only; no node IDs, endpoints, route IDs, encrypted payloads, receiver identities, client IPs, DNS contents, domains, URLs, Memory Chain plaintext, voucher secrets, wallet-level traffic, or social graph edges".to_string(),
        }
    }

    fn count_trailing_two_hop_outcomes(
        events: &[PeerStoreTwoHopPathProofEvent],
        outcome: &str,
    ) -> u64 {
        events
            .iter()
            .rev()
            .take_while(|event| event.outcome == outcome)
            .count() as u64
    }

    fn count_trailing_two_hop_message_delivery_successes(
        events: &[PeerStoreTwoHopPathProofEvent],
    ) -> u64 {
        events
            .iter()
            .rev()
            .take_while(|event| {
                event.outcome == "accepted" && event.proof_scope == "message_delivery"
            })
            .count() as u64
    }

    fn count_two_hop_event_buckets<'a>(
        events: &'a [PeerStoreTwoHopPathProofEvent],
        bucket: impl Fn(&'a PeerStoreTwoHopPathProofEvent) -> &'a str,
    ) -> BTreeMap<String, u64> {
        let mut counts = BTreeMap::new();
        for event in events {
            let key = bucket(event);
            if key.is_empty() {
                continue;
            }
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        counts
    }

    fn record_peer_event(
        &self,
        now: u64,
        event: impl Into<String>,
        outcome: impl Into<String>,
        source: impl Into<String>,
        node_id: &[u8; 32],
        sequence: Option<u64>,
        reason: Option<&str>,
    ) {
        let mut events = self.peer_events.write();
        if events.len() >= MAX_PEER_EVENTS {
            events.pop_front();
        }
        events.push_back(PeerStorePeerEvent {
            at: now,
            event: event.into(),
            outcome: outcome.into(),
            source: source.into(),
            node_id_prefix: Self::node_id_prefix(node_id),
            sequence,
            reason: reason.map(str::to_string),
        });
    }

    fn node_id_prefix(node_id: &[u8; 32]) -> String {
        hex::encode(&node_id[..4])
    }

    /// Returns newest discovery audit events in chronological order.
    #[must_use]
    pub fn recent_audit_events(&self) -> Vec<PeerStoreAuditEvent> {
        self.audit_events.read().iter().cloned().collect()
    }

    /// Returns recent peer lifecycle events in chronological order.
    #[must_use]
    pub fn recent_peer_events(&self) -> Vec<PeerStorePeerEvent> {
        self.peer_events.read().iter().cloned().collect()
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

    /// Exports a bounded verifiable peer-record snapshot for heartbeat.
    ///
    /// Heartbeat consumers should verify every descriptor in `records` using
    /// `SignedNodeDescriptor::verify_at(generated_at)` before accepting peer
    /// claims. This method deliberately exports signed node-level discovery
    /// metadata, not client/session traffic. It filters out expired or invalid
    /// descriptors and does not include retained expired cache history.
    #[must_use]
    pub fn export_signed_peer_records_for_heartbeat(
        &self,
        generated_at: u64,
        limit: Option<usize>,
    ) -> PeerStoreSignedPeerRecordsStatus {
        let peers = self.peers.read();
        let total_retained_records = peers.len();
        let mut descriptors: Vec<SignedNodeDescriptor> = peers
            .values()
            .filter(|descriptor| descriptor.verify_at(generated_at).is_ok())
            .cloned()
            .collect();
        drop(peers);

        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        let valid_signed_records = descriptors.len();
        if let Some(limit) = limit {
            descriptors.truncate(limit);
        }
        let exported_signed_records = descriptors.len();

        self.record_audit_event(
            generated_at,
            "heartbeat_signed_peer_records_export",
            "accepted",
            format!(
                "retained={} valid={} exported={} limit={}",
                total_retained_records,
                valid_signed_records,
                exported_signed_records,
                limit.map_or_else(|| "none".to_string(), |value| value.to_string())
            ),
        );

        PeerStoreSignedPeerRecordsStatus {
            generated_at,
            source: "rust_peer_store_signed_descriptors".to_string(),
            total_retained_records,
            valid_signed_records,
            exported_signed_records,
            limit,
            records: NodeBootstrapSnapshot::new(generated_at, descriptors),
            verification_rule:
                "verify each records.peers[] with SignedNodeDescriptor::verify_at(generated_at)"
                    .to_string(),
            privacy_boundary: "signed node discovery descriptors only; may include node public keys and public endpoints, but no client IPs, route ids, encrypted payloads, receiver identities, DNS contents, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    /// Exports a local peer-cache snapshot, including expired signed peers.
    ///
    /// This snapshot is for local restart recovery only. Public gossip and
    /// bootstrap responses must continue to use `export_bootstrap_snapshot()`,
    /// which filters to descriptors that are valid at `now`. Cache consumers
    /// must call `load_peer_cache_snapshot_from_source()` so expired records
    /// are retained only after signature verification and remain non-routeable.
    #[must_use]
    pub fn export_peer_cache_snapshot(&self, generated_at: u64) -> NodeBootstrapSnapshot {
        self.counters
            .last_snapshot_at
            .store(generated_at, Ordering::Relaxed);
        let mut valid = 0usize;
        let mut expired = 0usize;
        let mut descriptors: Vec<SignedNodeDescriptor> = self
            .peers
            .read()
            .values()
            .filter(|descriptor| descriptor.verify_signature().is_ok())
            .inspect(|descriptor| {
                if descriptor.descriptor.is_valid_at(generated_at) {
                    valid += 1;
                } else {
                    expired += 1;
                }
            })
            .cloned()
            .collect();

        descriptors.sort_by_key(|descriptor| (descriptor.node_id(), descriptor.sequence()));
        self.record_audit_event(
            generated_at,
            "peer_cache_export",
            "accepted",
            format!(
                "retained={} valid={} expired={}",
                descriptors.len(),
                valid,
                expired
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
        self.scored_route_candidates(capability, now, None, false)
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

    /// Plans a complete controlled route for the requested capability sequence.
    ///
    /// The planner selects one unique healthy-ranked peer for each requested
    /// capability, excluding self or already-used hops before each hop is
    /// chosen. It returns `None` unless the full path can be satisfied. This
    /// keeps future multi-hop/onion relay callers from accidentally using a
    /// partial route that weakens the intended privacy boundary.
    ///
    /// This method never reads or derives from encrypted chat/media blobs,
    /// packet payloads, DNS data, destinations, client public IPs, voucher
    /// secrets, private keys, wallet-level traffic, or plaintext content.
    #[must_use]
    pub fn route_path_with_capabilities_excluding(
        &self,
        capabilities: &[NodeCapability],
        now: u64,
        excluded_node_ids: &[[u8; 32]],
    ) -> Option<Vec<SignedNodeDescriptor>> {
        let mut excluded = excluded_node_ids.to_vec();
        let mut path = Vec::with_capacity(capabilities.len());

        for capability in capabilities {
            let next = self
                .scored_route_candidates(*capability, now, None, false)
                .into_iter()
                .find(|candidate| {
                    let node_id = candidate.descriptor.node_id();
                    !excluded.iter().any(|excluded| *excluded == node_id)
                })?;
            let node_id = next.descriptor.node_id();
            excluded.push(node_id);
            path.push(next.descriptor);
        }

        Some(path)
    }

    /// Downgrades descriptors that are no longer valid at `now`.
    ///
    /// Expired peers are retained as signed, non-routeable local history so a
    /// node does not forget known peers during cleanup or restart. Validity
    /// gates elsewhere (`verify_at(now)`, route candidates, gossip export)
    /// still prevent expired descriptors from being counted as live peers.
    pub fn cleanup_expired(&self, now: u64) -> usize {
        let expired: Vec<([u8; 32], u64)> = self
            .peers
            .read()
            .iter()
            .filter(|(_, descriptor)| descriptor.verify_at(now).is_err())
            .map(|(node_id, descriptor)| (*node_id, descriptor.sequence()))
            .collect();

        let mut newly_degraded = Vec::new();
        if !expired.is_empty() {
            let mut metadata = self.peer_runtime.write();
            for (node_id, sequence) in expired {
                let entry = metadata
                    .entry(node_id)
                    .or_insert_with(|| PeerRuntimeMetadata {
                        source: "unknown".to_string(),
                        first_seen_at: now,
                        last_seen_at: now,
                        last_sequence: sequence,
                        imported_count: 0,
                        expired_degraded_at: None,
                    });
                if entry.expired_degraded_at.is_none() {
                    entry.expired_degraded_at = Some(now);
                    newly_degraded.push((node_id, sequence));
                }
            }
        }

        let degraded_count = newly_degraded.len();
        if degraded_count > 0 {
            self.counters
                .expired_degraded
                .fetch_add(degraded_count as u64, Ordering::Relaxed);
            self.counters.last_cleanup_at.store(now, Ordering::Relaxed);
            for (node_id, sequence) in &newly_degraded {
                self.record_peer_event(
                    now,
                    "peer_expired",
                    "degraded",
                    "cleanup",
                    node_id,
                    Some(*sequence),
                    Some("descriptor_expired_retained"),
                );
            }
            self.record_audit_event(
                now,
                "expired_peer_cleanup",
                "accepted",
                format!(
                    "degraded={degraded_count} retained_total={} removed=0",
                    self.peers.read().len()
                ),
            );
        }
        degraded_count
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
        let mut restart_recovery_sources = Vec::new();
        if seed_recovery_configured {
            restart_recovery_sources.push("seed_endpoints".to_string());
        }
        if bootstrap.peer_cache_configured {
            restart_recovery_sources.push("peer_cache".to_string());
        }
        let restart_recovery_configured = !restart_recovery_sources.is_empty();
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
            restart_recovery_sources,
        }
    }

    /// Returns nodeboard-friendly peer store status.
    #[must_use]
    pub fn status(&self, now: u64) -> PeerStoreStatus {
        let snapshot = self.snapshot(now);
        let bootstrap = self.bootstrap_status.read().clone();
        let runtime = self.counters.snapshot();
        let blind_relay_quality = Self::blind_relay_quality_status(now, &runtime.blind_relay);
        let two_hop_path_proof_history = self.two_hop_path_proof_history(now);
        let stability = Self::stability(&snapshot, &bootstrap, now);
        let peer_summary = self.peer_summary(now);
        let route_candidates = self.route_candidate_status(now);
        let peer_health_summary = self.peer_health_summary(now);
        let peer_quorum =
            Self::peer_quorum_status(now, &stability, &peer_summary, &route_candidates);
        let network_story = Self::network_story_status(
            now,
            &stability,
            &peer_summary,
            &route_candidates,
            &two_hop_path_proof_history,
        );

        PeerStoreStatus {
            snapshot,
            runtime,
            blind_relay_quality,
            two_hop_path_proof_history,
            max_peers: self.max_peers(),
            recent_audit_events: self.recent_audit_events(),
            recent_peer_events: self.recent_peer_events(),
            bootstrap,
            stability,
            peer_summary,
            route_candidates,
            peer_health_summary,
            peer_quorum,
            network_story,
        }
    }

    fn blind_relay_quality_status(
        now: u64,
        stats: &PeerStoreBlindRelayStats,
    ) -> PeerStoreBlindRelayQualityStatus {
        let accepted_total = stats.terminal.saturating_add(stats.forwarded);
        let last_event_age_seconds = stats
            .last_event_at
            .map(|last_event_at| now.saturating_sub(last_event_at));
        let last_accepted_age_seconds = stats
            .last_accepted_at
            .map(|last_accepted_at| now.saturating_sub(last_accepted_at));
        let last_probe_age_seconds = stats
            .last_probe_at
            .map(|last_probe_at| now.saturating_sub(last_probe_at));
        let last_two_hop_probe_age_seconds = stats
            .last_two_hop_probe_at
            .map(|last_probe_at| now.saturating_sub(last_probe_at));
        let accepted_evidence_seen = accepted_total > 0;
        let probe_evidence_seen = stats.probe_succeeded > 0;
        let two_hop_probe_evidence_seen = stats.two_hop_probe_succeeded > 0;
        let real_relay_ready = accepted_evidence_seen
            && last_accepted_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let probe_ready = probe_evidence_seen
            && last_probe_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let two_hop_probe_ready = two_hop_probe_evidence_seen
            && last_two_hop_probe_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let synthetic_probe_ready = probe_ready || two_hop_probe_ready;
        let runtime_ready = real_relay_ready || synthetic_probe_ready;
        let stale_success_evidence =
            (accepted_evidence_seen || probe_evidence_seen || two_hop_probe_evidence_seen)
                && !runtime_ready;
        let evidence_mode = if accepted_evidence_seen {
            "real_relay_traffic"
        } else if two_hop_probe_evidence_seen {
            "synthetic_two_hop_control_probe"
        } else if probe_evidence_seen {
            "synthetic_probe"
        } else if stats.two_hop_probe_attempted > 0 {
            "two_hop_probe_failed"
        } else if stats.probe_attempted > 0 {
            "probe_failed"
        } else if stats.received > 0 {
            "real_relay_attempted"
        } else {
            "idle"
        };
        let proof_scope = if accepted_evidence_seen {
            "message_delivery"
        } else if two_hop_probe_evidence_seen || stats.two_hop_probe_attempted > 0 {
            "control_plane"
        } else if probe_evidence_seen || stats.probe_attempted > 0 {
            "single_hop_control_plane"
        } else if stats.received > 0 {
            "attempted"
        } else {
            "none"
        };
        let protection_active = stats.rate_limited > 0
            || stats.quarantined > 0
            || stats.quarantine_started > 0
            || stats.replay_dropped > 0
            || stats.loop_detected > 0
            || stats.timestamp_rejected > 0
            || stats.invalid_signature > 0;
        let transport_attention = stats.forward_failed > 0
            || stats.retry_exhausted > 0
            || stats.backpressure_dropped > 0
            || stats.probe_failed > 0;
        let quality_ready = runtime_ready && !transport_attention;

        let status = if stats.received == 0
            && stats.probe_attempted == 0
            && stats.two_hop_probe_attempted == 0
        {
            "idle"
        } else if stats.retry_exhausted > 0 || stats.backpressure_dropped > 0 {
            "attention"
        } else if stats.forward_failed > 0 || stats.probe_failed > 0 {
            "degraded"
        } else if protection_active {
            "protecting"
        } else if runtime_ready {
            "ready"
        } else if stale_success_evidence {
            "stale"
        } else {
            "observing"
        };

        let readiness_reason = if real_relay_ready && !transport_attention && !protection_active {
            "real_relay_observed"
        } else if real_relay_ready && transport_attention {
            "real_relay_transport_attention"
        } else if real_relay_ready && protection_active {
            "real_relay_protection_active"
        } else if two_hop_probe_ready && !transport_attention && !protection_active {
            "synthetic_two_hop_control_probe_ready"
        } else if synthetic_probe_ready && !transport_attention && !protection_active {
            "synthetic_probe_ready"
        } else if stats.two_hop_probe_attempted > 0 && stats.two_hop_probe_succeeded == 0 {
            "synthetic_two_hop_control_probe_failed"
        } else if stats.probe_attempted > 0 && stats.probe_succeeded == 0 {
            "synthetic_probe_failed"
        } else if transport_attention {
            "transport_attention"
        } else if protection_active {
            "protection_active"
        } else if accepted_evidence_seen && !real_relay_ready {
            "real_relay_stale"
        } else if two_hop_probe_evidence_seen && !two_hop_probe_ready {
            "synthetic_two_hop_control_probe_stale"
        } else if probe_evidence_seen && !probe_ready {
            "synthetic_probe_stale"
        } else if stats.received > 0 {
            "real_relay_attempted"
        } else {
            "idle_waiting_for_relay"
        };

        let accepted_percent = if stats.received == 0 {
            0
        } else {
            ((accepted_total.saturating_mul(100)) / stats.received).min(100) as u8
        };
        let next_action = match status {
            "idle" => "wait for encrypted blind relay traffic or run a synthetic relay probe",
            "attention" => {
                "inspect next-hop reachability, retry exhaustion, and local relay backpressure"
            }
            "degraded" => "inspect route candidates and next-hop transport health",
            "protecting" => {
                "review aggregate abuse-guard buckets while preserving blind relay metadata"
            }
            "ready" if real_relay_ready => {
                "blind relay runtime has accepted encrypted relay work"
            }
            "ready" if two_hop_probe_ready => {
                "two-hop control-plane path proof succeeded; wait for real message-delivery evidence"
            }
            "ready" => "synthetic relay probe succeeded; wait for real encrypted relay traffic",
            "stale" => "refresh relay readiness with a new accepted encrypted relay event or synthetic path proof",
            _ => "observe additional relay traffic before declaring runtime quality ready",
        };

        let detail = format!(
            "received={} accepted_total={} terminal={} forwarded={} rejected={} forward_failed={} retry_attempted={} retry_succeeded={} retry_exhausted={} backpressure_dropped={} probe_attempted={} probe_succeeded={} probe_failed={} two_hop_probe_attempted={} two_hop_probe_succeeded={} two_hop_probe_failed={} timestamp_rejected={} real_relay_ready={} synthetic_probe_ready={} two_hop_probe_ready={} evidence_mode={} proof_scope={} readiness_reason={} protection_active={} accepted_percent={} stale_after_seconds={} last_event_age_seconds={} last_accepted_age_seconds={} last_probe_age_seconds={} last_two_hop_probe_age_seconds={}",
            stats.received,
            accepted_total,
            stats.terminal,
            stats.forwarded,
            stats.rejected,
            stats.forward_failed,
            stats.retry_attempted,
            stats.retry_succeeded,
            stats.retry_exhausted,
            stats.backpressure_dropped,
            stats.probe_attempted,
            stats.probe_succeeded,
            stats.probe_failed,
            stats.two_hop_probe_attempted,
            stats.two_hop_probe_succeeded,
            stats.two_hop_probe_failed,
            stats.timestamp_rejected,
            real_relay_ready,
            synthetic_probe_ready,
            two_hop_probe_ready,
            evidence_mode,
            proof_scope,
            readiness_reason,
            protection_active,
            accepted_percent,
            PEER_ROUTEABILITY_STALE_AFTER_SECS,
            last_event_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_accepted_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_probe_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_two_hop_probe_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );

        PeerStoreBlindRelayQualityStatus {
            generated_at: now,
            status: status.to_string(),
            runtime_ready,
            quality_ready,
            real_relay_ready,
            synthetic_probe_ready,
            evidence_mode: evidence_mode.to_string(),
            proof_scope: proof_scope.to_string(),
            readiness_reason: readiness_reason.to_string(),
            accepted_total,
            last_accepted_at: stats.last_accepted_at,
            forward_failed: stats.forward_failed,
            retry_exhausted: stats.retry_exhausted,
            backpressure_dropped: stats.backpressure_dropped,
            probe_attempted: stats.probe_attempted,
            probe_succeeded: stats.probe_succeeded,
            probe_failed: stats.probe_failed,
            two_hop_probe_ready,
            two_hop_probe_attempted: stats.two_hop_probe_attempted,
            two_hop_probe_succeeded: stats.two_hop_probe_succeeded,
            two_hop_probe_failed: stats.two_hop_probe_failed,
            last_two_hop_probe_age_seconds,
            timestamp_rejected: stats.timestamp_rejected,
            protection_active,
            accepted_percent,
            last_event_age_seconds,
            last_accepted_age_seconds,
            last_probe_age_seconds,
            detail,
            next_action: next_action.to_string(),
            privacy_boundary: "aggregate blind relay runtime counters only; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client IPs, DNS contents, destinations, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    fn peer_quorum_status(
        now: u64,
        stability: &PeerStoreStabilityStatus,
        peer_summary: &PeerStorePeerSummaryStatus,
        route_candidates: &PeerStoreRouteCandidateStatus,
    ) -> PeerStorePeerQuorumStatus {
        let routeable_chat_relays = route_candidates
            .chat_relay
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();
        let routeable_onion_middle_hops = route_candidates
            .onion_middle
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();
        let healthy_ratio_percent = if peer_summary.valid_peers == 0 {
            0
        } else {
            ((peer_summary.healthy_peers * 100) / peer_summary.valid_peers).min(100) as u8
        };
        let enough_valid_peers = peer_summary.valid_peers >= PEER_QUORUM_MIN_VALID_PEERS;
        let enough_routeable_chat_relays =
            routeable_chat_relays >= PEER_QUORUM_MIN_ROUTEABLE_CHAT_RELAYS;
        let stability_needs_attention =
            matches!(stability.health.as_str(), "failed" | "degraded" | "stale");
        let quorum_ready = !stability_needs_attention
            && enough_valid_peers
            && enough_routeable_chat_relays
            && stability.restart_recovery_configured
            && stability.relay_foundation_ready;

        let status = if stability.health == "disabled" {
            "disabled"
        } else if stability_needs_attention {
            "attention"
        } else if !enough_valid_peers {
            "forming"
        } else if enough_routeable_chat_relays {
            "route_ready"
        } else {
            "peer_view_ready"
        };

        let next_action = match status {
            "disabled" => "enable discovery and configure peer recovery before testing multi-hop routing",
            "attention" => "restore discovery stability before relying on the peer view",
            "forming" => "add seed endpoints, peer cache, or live gossip until the verified peer view reaches quorum",
            "peer_view_ready" if route_candidates.chat_relay.iter().any(|peer| peer.endpoint_advertised) => {
                "wait for a fresh successful routeability probe before declaring encrypted relay readiness"
            }
            "peer_view_ready" => "ensure at least one verified peer advertises a public chat relay endpoint",
            _ if !stability.restart_recovery_configured => {
                "configure peer cache or seed endpoints so peer quorum survives restart"
            }
            _ => "peer quorum is ready for controlled encrypted relay experiments",
        };

        let detail = format!(
            "valid_peers={} healthy_peers={} stale_peers={} routeable_chat_relays={} routeable_onion_middle_hops={} restart_recovery_configured={} relay_foundation_ready={}",
            peer_summary.valid_peers,
            peer_summary.healthy_peers,
            peer_summary.stale_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            stability.restart_recovery_configured,
            stability.relay_foundation_ready
        );

        PeerStorePeerQuorumStatus {
            generated_at: now,
            status: status.to_string(),
            quorum_ready,
            min_valid_peers: PEER_QUORUM_MIN_VALID_PEERS,
            min_routeable_chat_relays: PEER_QUORUM_MIN_ROUTEABLE_CHAT_RELAYS,
            valid_peers: peer_summary.valid_peers,
            healthy_peers: peer_summary.healthy_peers,
            stale_peers: peer_summary.stale_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            healthy_ratio_percent,
            restart_recovery_configured: stability.restart_recovery_configured,
            relay_foundation_ready: stability.relay_foundation_ready,
            detail,
            next_action: next_action.to_string(),
            privacy_boundary: "aggregate local peer-view readiness only; not public-chain consensus; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client IPs, destinations, DNS contents, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    fn network_story_status(
        now: u64,
        stability: &PeerStoreStabilityStatus,
        peer_summary: &PeerStorePeerSummaryStatus,
        route_candidates: &PeerStoreRouteCandidateStatus,
        two_hop_path_proof_history: &PeerStoreTwoHopPathProofHistory,
    ) -> PeerStoreNetworkStoryStatus {
        let chat_single_hop_ready = route_candidates.planned_paths.chat_single_hop.complete;
        let planned_two_hop_onion_ready = route_candidates
            .planned_paths
            .chat_two_hop_onion_ready
            .complete;
        let proof_backed_two_hop_onion_ready = two_hop_path_proof_history.recent_success_ready
            && !two_hop_path_proof_history.failure_streak_active;
        let chat_two_hop_onion_ready =
            planned_two_hop_onion_ready || proof_backed_two_hop_onion_ready;
        let routeable_chat_relays = route_candidates
            .chat_relay
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();
        let routeable_onion_middle_hops = route_candidates
            .onion_middle
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();

        let stability_needs_attention =
            matches!(stability.health.as_str(), "failed" | "degraded" | "stale");

        let status = if stability.health == "disabled" {
            "disabled"
        } else if stability_needs_attention {
            "attention"
        } else if chat_two_hop_onion_ready {
            "onion_ready"
        } else if chat_single_hop_ready {
            "relay_ready"
        } else if peer_summary.valid_peers > 0 {
            "peer_view_ready"
        } else {
            "discovering"
        };

        let headline = match status {
            "onion_ready" => "Discovery can plan a two-hop privacy path",
            "relay_ready" => "Discovery can route encrypted relay traffic",
            "peer_view_ready" => "Discovery has a verified peer view",
            "disabled" => "Discovery is disabled",
            "attention" => "Discovery needs operator attention",
            _ => "Discovery is learning the network",
        };

        let detail = format!(
            "valid_nodes={} chat_relay_nodes={} onion_middle_nodes={} routeable_chat_relays={} routeable_onion_middle_hops={} two_hop_path_proof_recent={} restart_recovery_configured={} relay_foundation_ready={}",
            peer_summary.valid_peers,
            peer_summary.chat_relay_peers,
            peer_summary.onion_middle_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            proof_backed_two_hop_onion_ready,
            stability.restart_recovery_configured,
            stability.relay_foundation_ready
        );

        PeerStoreNetworkStoryStatus {
            generated_at: now,
            status: status.to_string(),
            headline: headline.to_string(),
            detail,
            discovered_nodes: peer_summary.total_peers,
            valid_nodes: peer_summary.valid_peers,
            chat_relay_nodes: peer_summary.chat_relay_peers,
            onion_middle_nodes: peer_summary.onion_middle_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            chat_single_hop_ready,
            chat_two_hop_onion_ready,
            restart_recovery_configured: stability.restart_recovery_configured,
            relay_foundation_ready: stability.relay_foundation_ready,
            privacy_boundary: "aggregate node discovery status only; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client IPs, destinations, DNS contents, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
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
        if Self::route_quarantine_remaining_seconds(route_health, now).is_some() {
            return ("quarantined", -300);
        }
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

    fn routeability_probe_at(route_health: Option<&PeerRouteHealth>) -> Option<u64> {
        route_health.and_then(
            |health| match (health.last_success_at, health.last_failure_at) {
                (Some(success_at), Some(failure_at)) => Some(success_at.max(failure_at)),
                (Some(success_at), None) => Some(success_at),
                (None, Some(failure_at)) => Some(failure_at),
                (None, None) => None,
            },
        )
    }

    fn routeability_state_and_ready(
        route_health: Option<&PeerRouteHealth>,
        now: u64,
    ) -> (&'static str, bool) {
        let Some(route_health) = route_health else {
            return ("unknown", false);
        };
        if Self::route_quarantine_remaining_seconds(route_health, now).is_some() {
            return ("quarantined", false);
        }

        match (route_health.last_success_at, route_health.last_failure_at) {
            (Some(success_at), Some(failure_at)) if failure_at > success_at => {
                if now.saturating_sub(failure_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS {
                    ("unreachable", false)
                } else {
                    ("stale", false)
                }
            }
            (Some(success_at), _) => {
                if now.saturating_sub(success_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS {
                    ("reachable", true)
                } else {
                    ("stale", false)
                }
            }
            (None, Some(failure_at)) => {
                if now.saturating_sub(failure_at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS {
                    ("unreachable", false)
                } else {
                    ("stale", false)
                }
            }
            (None, None) => ("unknown", false),
        }
    }

    /// Returns whether a peer has fresh routeability evidence at `now`.
    ///
    /// This is intentionally stricter than descriptor validity. A peer may be
    /// signed, unexpired, and capability-compatible while still being unknown,
    /// stale, unreachable, or quarantined from the routing layer's point of
    /// view. Blind relay forwarding must use this helper so encrypted envelopes
    /// are sent only to nodes with fresh successful probe/forward evidence.
    #[must_use]
    pub fn is_routeable_now(&self, node_id: &[u8; 32], now: u64) -> bool {
        let route_health = self.route_health.read();
        let (_, ready) = Self::routeability_state_and_ready(route_health.get(node_id), now);
        ready
    }

    /// Returns whether a peer is currently isolated by route-health quarantine.
    ///
    /// This is narrower than `is_routeable_now`: unknown or stale peers return
    /// false here. Onion middle recovery uses this helper to allow a fresh signed
    /// descriptor to prove itself after restart while still refusing peers that
    /// recently crossed the local failure threshold.
    #[must_use]
    pub fn is_route_quarantined_now(&self, node_id: &[u8; 32], now: u64) -> bool {
        let route_health = self.route_health.read();
        route_health
            .get(node_id)
            .and_then(|value| Self::route_quarantine_remaining_seconds(value, now))
            .is_some()
    }

    fn route_quarantine_remaining_seconds(route_health: &PeerRouteHealth, now: u64) -> Option<u64> {
        route_health.quarantine_until.and_then(|quarantine_until| {
            (now < quarantine_until).then_some(quarantine_until.saturating_sub(now))
        })
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
        include_route_quarantined: bool,
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
            let (routeability_state, routeability_ready) =
                Self::routeability_state_and_ready(route_health_entry, now);
            let last_routeability_probe_at = Self::routeability_probe_at(route_health_entry);
            let last_routeability_probe_age_seconds =
                last_routeability_probe_at.map(|probe_at| now.saturating_sub(probe_at));
            if !include_route_quarantined && route_health_bucket == "quarantined" {
                continue;
            }
            let route_quarantine_remaining_seconds = route_health_entry
                .and_then(|value| Self::route_quarantine_remaining_seconds(value, now));
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
                    routeability_state: routeability_state.to_string(),
                    routeability_ready,
                    last_routeability_probe_at,
                    last_routeability_probe_age_seconds,
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
                    route_quarantined: route_quarantine_remaining_seconds.is_some(),
                    route_quarantine_remaining_seconds,
                    route_quarantine_count: route_health_entry
                        .map(|value| value.quarantine_count)
                        .unwrap_or(0),
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
        self.scored_route_candidates(capability, now, Some(PEER_ROUTE_STATUS_LIMIT), true)
            .into_iter()
            .map(|candidate| candidate.summary)
            .collect()
    }

    fn route_path_plan_preview(
        &self,
        label: &str,
        capabilities: &[NodeCapability],
        now: u64,
    ) -> PeerStoreRoutePathPlan {
        let mut excluded = Vec::with_capacity(capabilities.len());
        let mut hops = Vec::with_capacity(capabilities.len());

        for capability in capabilities {
            let Some(next) = self
                .scored_route_candidates(*capability, now, None, false)
                .into_iter()
                .filter(|candidate| candidate.summary.routeability_ready)
                .find(|candidate| {
                    let node_id = candidate.descriptor.node_id();
                    !excluded.iter().any(|excluded| *excluded == node_id)
                })
            else {
                break;
            };

            let node_id = next.descriptor.node_id();
            excluded.push(node_id);
            hops.push(PeerStoreRoutePathHop {
                hop_index: hops.len(),
                capability: Self::capability_label(*capability).to_string(),
                node_id_prefix: next.summary.node_id_prefix,
                score: next.summary.score,
                health: next.summary.health,
                route_health: next.summary.route_health,
                last_seen_age_seconds: next.summary.last_seen_age_seconds,
                ttl_remaining_seconds: next.summary.ttl_remaining_seconds,
                region: next.summary.region,
            });
        }

        PeerStoreRoutePathPlan {
            label: label.to_string(),
            required_capabilities: capabilities
                .iter()
                .map(|capability| Self::capability_label(*capability).to_string())
                .collect(),
            complete: hops.len() == capabilities.len(),
            hop_count: hops.len(),
            hops,
        }
    }

    fn route_path_status(&self, now: u64) -> PeerStoreRoutePathStatus {
        PeerStoreRoutePathStatus {
            chat_single_hop: self.route_path_plan_preview(
                "chat_single_hop",
                &[NodeCapability::ChatRelay],
                now,
            ),
            chat_two_hop_onion_ready: self.route_path_plan_preview(
                "chat_two_hop_onion_ready",
                &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
                now,
            ),
        }
    }

    /// Builds privacy-safe route candidate lists for nodeboard.
    #[must_use]
    pub fn route_candidate_status(&self, now: u64) -> PeerStoreRouteCandidateStatus {
        PeerStoreRouteCandidateStatus {
            generated_at: now,
            privacy_relay: self.route_candidate_summaries(NodeCapability::PrivacyRelay, now),
            chat_relay: self.route_candidate_summaries(NodeCapability::ChatRelay, now),
            onion_middle: self.route_candidate_summaries(NodeCapability::OnionMiddle, now),
            planned_paths: self.route_path_status(now),
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

    fn source_is_live_gossip(source: &str) -> bool {
        matches!(source, "gossip_snapshot" | "gossip_announce")
    }

    fn peer_health_bucket(
        descriptor_health: &str,
        route_health: &str,
        relay_quarantined: bool,
        relay_rejection_count: u64,
    ) -> &'static str {
        if relay_quarantined || route_health == "quarantined" {
            return "quarantined";
        }
        if descriptor_health == "expired" {
            return "expired";
        }
        if route_health == "failing" {
            return "failing";
        }
        if descriptor_health == "stale" || route_health == "degraded" || relay_rejection_count > 0 {
            return "degraded";
        }
        "healthy"
    }

    /// Builds a privacy-safe per-peer health summary for nodeboard.
    ///
    /// The summary intentionally combines only node-level control-plane state:
    /// signed descriptor freshness, import/gossip source buckets, local opaque
    /// route-health counters, and relay-protection buckets. It never includes
    /// endpoint URLs, route ids, encrypted blobs, receiver identities, client
    /// IPs, destinations, DNS contents, voucher secrets, private keys,
    /// wallet-level traffic, or plaintext content.
    #[must_use]
    pub fn peer_health_summary(&self, now: u64) -> PeerStorePeerHealthStatus {
        let peers = self.peers.read();
        let metadata = self.peer_runtime.read();
        let route_health = self.route_health.read();
        let relay_health = self.relay_protection_health.read();

        let mut rows = Vec::with_capacity(peers.len().min(PEER_HEALTH_STATUS_LIMIT));
        let mut healthy_peers = 0usize;
        let mut degraded_peers = 0usize;
        let mut failing_peers = 0usize;
        let mut quarantined_peers = 0usize;

        for (node_id, descriptor) in peers.iter() {
            let meta = metadata.get(node_id);
            let source = meta
                .map(|value| value.source.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let last_seen_at = meta.map(|value| value.last_seen_at).unwrap_or(now);
            let last_seen_age_seconds = now.saturating_sub(last_seen_at);
            let last_successful_gossip_at = meta.and_then(|value| {
                Self::source_is_live_gossip(&value.source).then_some(value.last_seen_at)
            });
            let last_successful_gossip_age_seconds =
                last_successful_gossip_at.map(|value| now.saturating_sub(value));

            let (descriptor_health, _) = Self::descriptor_health(descriptor, now);
            let route_health_entry = route_health.get(node_id);
            let (route_health_bucket, _) =
                Self::route_health_bucket_and_score(route_health_entry, now);
            let (routeability_state, routeability_ready) =
                Self::routeability_state_and_ready(route_health_entry, now);
            let last_routeability_probe_at = Self::routeability_probe_at(route_health_entry);
            let last_routeability_probe_age_seconds =
                last_routeability_probe_at.map(|probe_at| now.saturating_sub(probe_at));
            let route_quarantine_remaining_seconds = route_health_entry
                .and_then(|value| Self::route_quarantine_remaining_seconds(value, now));
            let relay_health_entry = relay_health.get(node_id);
            let relay_quarantine_remaining_seconds = relay_health_entry
                .and_then(|value| value.quarantine_until)
                .and_then(|quarantine_until| {
                    (now < quarantine_until).then_some(quarantine_until.saturating_sub(now))
                });
            let relay_quarantined = relay_quarantine_remaining_seconds.is_some();
            let relay_rejection_count = relay_health_entry
                .map(|value| value.rejection_count)
                .unwrap_or(0);
            let health = Self::peer_health_bucket(
                descriptor_health,
                route_health_bucket,
                relay_quarantined,
                relay_rejection_count,
            );

            match health {
                "quarantined" => quarantined_peers += 1,
                "failing" => failing_peers += 1,
                "degraded" | "expired" => degraded_peers += 1,
                _ => healthy_peers += 1,
            }

            rows.push(PeerStorePeerHealth {
                node_id_prefix: hex::encode(&node_id[..4]),
                health: health.to_string(),
                descriptor_health: descriptor_health.to_string(),
                source,
                last_successful_gossip_at,
                last_successful_gossip_age_seconds,
                last_seen_at,
                last_seen_age_seconds,
                route_health: route_health_bucket.to_string(),
                routeability_state: routeability_state.to_string(),
                routeability_ready,
                last_routeability_probe_at,
                last_routeability_probe_age_seconds,
                route_success_count: route_health_entry
                    .map(|value| value.success_count)
                    .unwrap_or(0),
                route_failure_count: route_health_entry
                    .map(|value| value.failure_count)
                    .unwrap_or(0),
                route_consecutive_failures: route_health_entry
                    .map(|value| value.consecutive_failures)
                    .unwrap_or(0),
                last_route_success_at: route_health_entry.and_then(|value| value.last_success_at),
                last_route_failure_at: route_health_entry.and_then(|value| value.last_failure_at),
                last_route_failure_reason: route_health_entry
                    .and_then(|value| value.last_failure_reason.clone()),
                route_quarantined: route_quarantine_remaining_seconds.is_some(),
                route_quarantine_remaining_seconds,
                route_quarantine_count: route_health_entry
                    .map(|value| value.quarantine_count)
                    .unwrap_or(0),
                last_route_quarantine_at: route_health_entry
                    .and_then(|value| value.last_quarantine_at),
                last_route_quarantine_reason: route_health_entry
                    .and_then(|value| value.last_quarantine_reason.clone()),
                relay_rejection_count,
                relay_quarantine_count: relay_health_entry
                    .map(|value| value.quarantine_count)
                    .unwrap_or(0),
                relay_quarantined,
                relay_quarantine_remaining_seconds,
                last_relay_rejection_at: relay_health_entry
                    .and_then(|value| value.last_rejection_at),
                last_relay_rejection_reason: relay_health_entry
                    .and_then(|value| value.last_rejection_reason.clone()),
                last_relay_quarantine_at: relay_health_entry
                    .and_then(|value| value.last_quarantine_at),
                last_relay_quarantine_reason: relay_health_entry
                    .and_then(|value| value.last_quarantine_reason.clone()),
            });
        }

        rows.sort_by(|a, b| {
            peer_health_rank(&a.health)
                .cmp(&peer_health_rank(&b.health))
                .then_with(|| {
                    b.route_consecutive_failures
                        .cmp(&a.route_consecutive_failures)
                })
                .then_with(|| b.relay_rejection_count.cmp(&a.relay_rejection_count))
                .then_with(|| a.last_seen_age_seconds.cmp(&b.last_seen_age_seconds))
                .then_with(|| a.node_id_prefix.cmp(&b.node_id_prefix))
        });
        rows.truncate(PEER_HEALTH_STATUS_LIMIT);

        PeerStorePeerHealthStatus {
            generated_at: now,
            total_peers: peers.len(),
            healthy_peers,
            degraded_peers,
            failing_peers,
            quarantined_peers,
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

fn peer_health_rank(health: &str) -> u8 {
    match health {
        "quarantined" => 0,
        "failing" => 1,
        "degraded" => 2,
        "expired" => 3,
        "stale" => 4,
        _ => 5,
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
        assert_eq!(status.chat_relay[0].routeability_state, "unknown");
        assert!(!status.chat_relay[0].routeability_ready);
        assert!(status.chat_relay[0].score > status.chat_relay[1].score);
    }

    #[test]
    fn test_is_routeable_now_requires_fresh_successful_route_evidence() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let peer_kp = IdentityKeyPair::generate();
        let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
        descriptor.descriptor.public_endpoint = Some("https://routeable.example".to_string());
        descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
        let node_id = descriptor.node_id();

        store
            .upsert_verified_from_source(descriptor, now, "gossip_announce")
            .unwrap();

        assert!(!store.is_routeable_now(&node_id, now));

        store.record_route_forward_failure(&node_id, now + 1, "request_failed");
        assert!(!store.is_routeable_now(&node_id, now + 2));

        store.record_route_forward_success(&node_id, now + 3);
        assert!(store.is_routeable_now(&node_id, now + 4));

        assert!(!store.is_routeable_now(&node_id, now + 3 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1));
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
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].node_id(), healthy.node_id());

        let status = store.route_candidate_status(now + 4);
        let failing_row = status
            .chat_relay
            .iter()
            .find(|row| row.node_id_prefix == failing_prefix)
            .expect("failing peer should remain visible as a quarantined candidate");
        assert_eq!(failing_row.route_health, "quarantined");
        assert_eq!(failing_row.routeability_state, "quarantined");
        assert!(!failing_row.routeability_ready);
        assert!(failing_row.route_quarantined);
        assert_eq!(failing_row.route_quarantine_count, 1);
        assert_eq!(
            failing_row.route_quarantine_remaining_seconds,
            Some(PEER_ROUTE_FAILURE_QUARANTINE_SECS - 1)
        );
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
        assert_eq!(recovered_row.routeability_state, "reachable");
        assert!(recovered_row.routeability_ready);
        assert_eq!(recovered_row.last_routeability_probe_at, Some(now + 5));
        assert!(!recovered_row.route_quarantined);
        assert_eq!(recovered_row.route_quarantine_remaining_seconds, None);
        assert_eq!(recovered_row.route_consecutive_failures, 0);
        assert_eq!(recovered_row.last_route_success_at, Some(now + 5));
    }

    #[test]
    fn test_peer_health_summary_reports_gossip_route_and_quarantine_without_payload_data() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let peer_kp = IdentityKeyPair::generate();
        let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
        descriptor.descriptor.public_endpoint = Some("https://peer-health.example".to_string());
        descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
        let node_id = descriptor.node_id();
        let node_prefix = hex::encode(&node_id[..4]);

        store
            .upsert_verified_from_source(descriptor, now, "gossip_announce")
            .unwrap();
        store.record_route_forward_failure(&node_id, now + 1, "request_failed");
        store.record_route_forward_failure(&node_id, now + 2, "http_502");
        store.record_peer_relay_rejection(&node_id, now + 3, "duplicate_route");
        store.record_peer_relay_quarantine_started(
            &node_id,
            now + 4,
            now + 304,
            "failure_threshold",
        );

        let status = store.status(now + 10).peer_health_summary;
        assert_eq!(status.total_peers, 1);
        assert_eq!(status.quarantined_peers, 1);
        let row = status.peers.first().expect("peer health row");
        assert_eq!(row.node_id_prefix, node_prefix);
        assert_eq!(row.health, "quarantined");
        assert_eq!(row.source, "gossip_announce");
        assert_eq!(row.last_successful_gossip_at, Some(now));
        assert_eq!(row.last_successful_gossip_age_seconds, Some(10));
        assert_eq!(row.route_failure_count, 2);
        assert_eq!(row.route_consecutive_failures, 2);
        assert_eq!(row.routeability_state, "unreachable");
        assert!(!row.routeability_ready);
        assert_eq!(row.last_routeability_probe_at, Some(now + 2));
        assert_eq!(row.last_routeability_probe_age_seconds, Some(8));
        assert_eq!(row.last_route_failure_reason.as_deref(), Some("http_502"));
        assert!(row.relay_quarantined);
        assert_eq!(row.relay_rejection_count, 1);
        assert_eq!(row.relay_quarantine_count, 1);
        assert_eq!(row.relay_quarantine_remaining_seconds, Some(294));
        assert_eq!(
            row.last_relay_rejection_reason.as_deref(),
            Some("duplicate_route")
        );
        assert_eq!(
            row.last_relay_quarantine_reason.as_deref(),
            Some("failure_threshold")
        );

        let status_json = serde_json::to_string(&status).unwrap();
        assert!(!status_json.contains("peer-health.example"));
        assert!(!status_json.contains(&hex::encode(node_id)));
        assert!(!status_json.contains("encrypted_blob"));
        assert!(!status_json.contains("route_id"));
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
    fn test_route_path_planner_selects_unique_hops_without_payload_data() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let self_kp = IdentityKeyPair::generate();
        let shared_kp = IdentityKeyPair::generate();
        let middle_kp = IdentityKeyPair::generate();
        let relay_kp = IdentityKeyPair::generate();

        let mut self_descriptor = signed_descriptor_for(&self_kp, 1, now + 2_000);
        self_descriptor.descriptor.capabilities =
            vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay];
        self_descriptor.descriptor.public_endpoint = Some("https://self.example".to_string());
        self_descriptor.descriptor.capacity.max_sessions = 4096;
        self_descriptor = SignedNodeDescriptor::sign(self_descriptor.descriptor, &self_kp).unwrap();
        let self_node_id = self_descriptor.node_id();

        let mut shared_descriptor = signed_descriptor_for(&shared_kp, 1, now + 2_000);
        shared_descriptor.descriptor.capabilities =
            vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay];
        shared_descriptor.descriptor.public_endpoint = Some("https://shared.example".to_string());
        shared_descriptor.descriptor.capacity.max_sessions = 2048;
        shared_descriptor =
            SignedNodeDescriptor::sign(shared_descriptor.descriptor, &shared_kp).unwrap();
        let shared_node_id = shared_descriptor.node_id();

        let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint = Some("https://middle.example".to_string());
        middle_descriptor.descriptor.capacity.max_sessions = 512;
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();

        let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
        relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
        relay_descriptor.descriptor.public_endpoint = Some("https://relay.example".to_string());
        relay_descriptor.descriptor.capacity.max_sessions = 256;
        relay_descriptor =
            SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
        let relay_node_id = relay_descriptor.node_id();

        store
            .upsert_verified_from_source(self_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(shared_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
            .unwrap();

        let path = store
            .route_path_with_capabilities_excluding(
                &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
                now,
                &[self_node_id],
            )
            .expect("two-hop path should be available");

        assert_eq!(path.len(), 2);
        assert_eq!(path[0].node_id(), shared_node_id);
        assert_eq!(path[1].node_id(), relay_node_id);
        assert_ne!(path[0].node_id(), path[1].node_id());
        assert!(!path.iter().any(|hop| hop.node_id() == self_node_id));
    }

    #[test]
    fn test_route_path_status_is_privacy_safe_and_marks_incomplete_paths() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let middle_kp = IdentityKeyPair::generate();

        let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint =
            Some("https://private-middle.example".to_string());
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
        let middle_node_id = middle_descriptor.node_id();

        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();

        let status = store.route_candidate_status(now);
        assert!(!status.planned_paths.chat_single_hop.complete);
        assert_eq!(status.planned_paths.chat_single_hop.hop_count, 0);
        assert!(!status.planned_paths.chat_two_hop_onion_ready.complete);
        assert_eq!(status.planned_paths.chat_two_hop_onion_ready.hop_count, 0);
        assert_eq!(status.onion_middle.len(), 1);
        assert_eq!(status.onion_middle[0].routeability_state, "unknown");
        assert!(!status.onion_middle[0].routeability_ready);

        let status_json = serde_json::to_string(&status).unwrap();
        assert!(!status_json.contains("private-middle.example"));
        assert!(!status_json.contains(&hex::encode(middle_node_id)));
        assert!(!status_json.contains("encrypted_blob"));
        assert!(!status_json.contains("receiver_pubkey"));
    }

    #[test]
    fn test_route_path_status_does_not_count_quarantined_peers_as_ready() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let middle_kp = IdentityKeyPair::generate();
        let relay_kp = IdentityKeyPair::generate();

        let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint =
            Some("https://quarantined-middle.example".to_string());
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
        let middle_node_id = middle_descriptor.node_id();

        let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
        relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
        relay_descriptor.descriptor.public_endpoint =
            Some("https://quarantined-relay.example".to_string());
        relay_descriptor =
            SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
        let relay_node_id = relay_descriptor.node_id();

        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
            .unwrap();

        for node_id in [&middle_node_id, &relay_node_id] {
            store.record_route_forward_success(node_id, now + 1);
            store.record_route_forward_failure(node_id, now + 2, "request_failed");
            store.record_route_forward_failure(node_id, now + 3, "request_failed");
            store.record_route_forward_failure(node_id, now + 4, "http_502");
        }

        let status = store.route_candidate_status(now + 5);
        assert!(!status.planned_paths.chat_single_hop.complete);
        assert_eq!(status.planned_paths.chat_single_hop.hop_count, 0);
        assert!(!status.planned_paths.chat_two_hop_onion_ready.complete);
        assert_eq!(status.planned_paths.chat_two_hop_onion_ready.hop_count, 0);
        assert_eq!(status.chat_relay[0].route_health, "quarantined");
        assert!(status.chat_relay[0].route_quarantined);
        assert_eq!(status.onion_middle[0].route_health, "quarantined");
        assert!(status.onion_middle[0].route_quarantined);

        let status_json = serde_json::to_string(&status).unwrap();
        assert!(!status_json.contains("quarantined-middle.example"));
        assert!(!status_json.contains("quarantined-relay.example"));
        assert!(!status_json.contains(&hex::encode(middle_node_id)));
        assert!(!status_json.contains(&hex::encode(relay_node_id)));
        assert!(!status_json.contains("encrypted_blob"));
        assert!(!status_json.contains("receiver_pubkey"));
    }

    #[test]
    fn test_route_quarantine_expiry_allows_reprobe_without_marking_ready() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let peer_kp = IdentityKeyPair::generate();
        let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
        descriptor.descriptor.public_endpoint =
            Some("https://reprobe-after-quarantine.example".to_string());
        descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
        let node_id = descriptor.node_id();
        let node_prefix = hex::encode(&node_id[..4]);

        store
            .upsert_verified_from_source(descriptor.clone(), now, "gossip_announce")
            .unwrap();

        store.record_route_forward_success(&node_id, now + 1);
        store.record_route_forward_failure(&node_id, now + 2, "request_failed");
        store.record_route_forward_failure(&node_id, now + 3, "request_failed");
        store.record_route_forward_failure(&node_id, now + 4, "http_502");

        let quarantined_candidates =
            store.route_candidates_with_capability(NodeCapability::ChatRelay, now + 5, 8);
        assert!(quarantined_candidates.is_empty());

        let after_quarantine = now + 4 + PEER_ROUTE_FAILURE_QUARANTINE_SECS + 1;
        let reprobe_candidates =
            store.route_candidates_with_capability(NodeCapability::ChatRelay, after_quarantine, 8);
        assert_eq!(reprobe_candidates.len(), 1);
        assert_eq!(reprobe_candidates[0].node_id(), descriptor.node_id());

        let status = store.route_candidate_status(after_quarantine);
        let row = status
            .chat_relay
            .iter()
            .find(|candidate| candidate.node_id_prefix == node_prefix)
            .expect("expired quarantine peer should be visible for reprobe");
        assert_eq!(row.route_health, "failing");
        assert_eq!(row.routeability_state, "unreachable");
        assert!(!row.routeability_ready);
        assert!(!row.route_quarantined);
        assert_eq!(row.route_quarantine_remaining_seconds, None);
        assert_eq!(row.route_quarantine_count, 1);

        store.record_route_forward_success(&node_id, after_quarantine + 1);
        let recovered = store.route_candidate_status(after_quarantine + 2);
        let recovered_row = recovered
            .chat_relay
            .iter()
            .find(|candidate| candidate.node_id_prefix == node_prefix)
            .expect("successful reprobe should keep peer visible");
        assert_eq!(recovered_row.route_health, "healthy");
        assert_eq!(recovered_row.routeability_state, "reachable");
        assert!(recovered_row.routeability_ready);
        assert_eq!(
            recovered_row.last_routeability_probe_at,
            Some(after_quarantine + 1)
        );

        let status_json = serde_json::to_string(&recovered).unwrap();
        assert!(!status_json.contains("reprobe-after-quarantine.example"));
        assert!(!status_json.contains(&hex::encode(node_id)));
        assert!(!status_json.contains("encrypted_blob"));
        assert!(!status_json.contains("receiver_pubkey"));
    }

    #[test]
    fn test_network_story_reports_onion_ready_without_endpoint_or_full_node_id() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        store.configure_bootstrap_status(true, true, true, 2);
        store.record_gossip_round(now + 20, 2, 2, 1, None);

        let middle_kp = IdentityKeyPair::generate();
        let relay_kp = IdentityKeyPair::generate();

        let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint =
            Some("https://story-middle.example".to_string());
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
        let middle_node_id = middle_descriptor.node_id();

        let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
        relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
        relay_descriptor.descriptor.public_endpoint =
            Some("https://story-relay.example".to_string());
        relay_descriptor =
            SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
        let relay_node_id = relay_descriptor.node_id();

        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
            .unwrap();
        store.record_route_forward_success(&middle_node_id, now + 25);
        store.record_route_forward_success(&relay_node_id, now + 26);

        let story = store.status(now + 30).network_story;

        assert_eq!(story.status, "onion_ready");
        assert!(story.chat_single_hop_ready);
        assert!(story.chat_two_hop_onion_ready);
        assert_eq!(story.valid_nodes, 2);
        assert_eq!(story.routeable_chat_relays, 1);
        assert_eq!(story.routeable_onion_middle_hops, 1);
        assert!(story.relay_foundation_ready);
        assert!(story.restart_recovery_configured);

        let story_json = serde_json::to_string(&story).unwrap();
        assert!(!story_json.contains("story-middle.example"));
        assert!(!story_json.contains("story-relay.example"));
        assert!(!story_json.contains(&hex::encode(middle_node_id)));
        assert!(!story_json.contains(&hex::encode(relay_node_id)));
        assert!(!story_json.contains("encrypted_blob"));
        assert!(!story_json.contains("receiver_pubkey"));
    }

    #[test]
    fn test_network_story_uses_fresh_two_hop_path_proof_as_onion_ready_evidence() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        store.configure_bootstrap_status(true, true, true, 2);
        store.record_gossip_round(now + 20, 2, 2, 1, None);

        let middle_kp = IdentityKeyPair::generate();
        let relay_kp = IdentityKeyPair::generate();

        let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint =
            Some("https://proof-middle.example".to_string());
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();

        let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
        relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
        relay_descriptor.descriptor.public_endpoint =
            Some("https://proof-relay.example".to_string());
        relay_descriptor =
            SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();

        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
            .unwrap();
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + 25,
            true,
            "onion_terminal_delivered",
            1,
            1,
            2,
            1,
        );

        let story = store.status(now + 30).network_story;

        assert_eq!(story.status, "onion_ready");
        assert!(story.chat_two_hop_onion_ready);
        assert_eq!(story.routeable_chat_relays, 0);
        assert_eq!(story.routeable_onion_middle_hops, 0);
        assert!(story.detail.contains("two_hop_path_proof_recent=true"));
        let story_json = serde_json::to_string(&story).unwrap();
        assert!(!story_json.contains("proof-middle.example"));
        assert!(!story_json.contains("proof-relay.example"));
    }

    #[test]
    fn test_network_story_does_not_use_stale_two_hop_path_proof_as_onion_ready_evidence() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        store.configure_bootstrap_status(true, true, true, 2);
        store.record_gossip_round(now + 20, 2, 2, 1, None);

        let middle_kp = IdentityKeyPair::generate();
        let relay_kp = IdentityKeyPair::generate();

        let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 4_000);
        middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
        middle_descriptor.descriptor.public_endpoint =
            Some("https://stale-proof-middle.example".to_string());
        middle_descriptor =
            SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
        let middle_node_id = middle_descriptor.node_id();

        let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 4_000);
        relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
        relay_descriptor.descriptor.public_endpoint =
            Some("https://stale-proof-relay.example".to_string());
        relay_descriptor =
            SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
        let relay_node_id = relay_descriptor.node_id();

        store
            .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
            .unwrap();
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + 25,
            true,
            "onion_terminal_delivered",
            1,
            1,
            2,
            1,
        );

        let stale_now = now + 25 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1;
        store.record_gossip_round(stale_now - 10, 2, 2, 1, None);
        let status = store.status(stale_now);
        let history = status.two_hop_path_proof_history;
        let story = status.network_story;

        assert_eq!(history.status, "stale");
        assert_eq!(history.freshness_bucket, "stale_success");
        assert!(!history.recent_success_ready);
        assert_eq!(story.status, "peer_view_ready");
        assert!(!story.chat_two_hop_onion_ready);
        assert!(story.detail.contains("two_hop_path_proof_recent=false"));

        let story_json = serde_json::to_string(&story).unwrap();
        assert!(!story_json.contains("stale-proof-middle.example"));
        assert!(!story_json.contains("stale-proof-relay.example"));
        assert!(!story_json.contains(&hex::encode(middle_node_id)));
        assert!(!story_json.contains(&hex::encode(relay_node_id)));
        assert!(!story_json.contains("encrypted_blob"));
        assert!(!story_json.contains("receiver_pubkey"));
    }

    #[test]
    fn test_network_story_attention_overrides_peer_view_when_recovery_is_missing() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        store.configure_bootstrap_status(true, false, false, 0);

        let first_kp = IdentityKeyPair::generate();
        let second_kp = IdentityKeyPair::generate();

        let mut first_descriptor = signed_descriptor_for(&first_kp, 1, now + 2_000);
        first_descriptor.descriptor.public_endpoint =
            Some("https://recovery-missing-a.example".to_string());
        first_descriptor = SignedNodeDescriptor::sign(first_descriptor.descriptor, &first_kp)
            .expect("descriptor should sign");
        let first_node_id = first_descriptor.node_id();

        let mut second_descriptor = signed_descriptor_for(&second_kp, 1, now + 2_000);
        second_descriptor.descriptor.public_endpoint =
            Some("https://recovery-missing-b.example".to_string());
        second_descriptor = SignedNodeDescriptor::sign(second_descriptor.descriptor, &second_kp)
            .expect("descriptor should sign");
        let second_node_id = second_descriptor.node_id();

        store
            .upsert_verified_from_source(first_descriptor, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(second_descriptor, now, "gossip_announce")
            .unwrap();

        let story = store.status(now + 30).network_story;

        assert_eq!(story.status, "attention");
        assert_eq!(story.valid_nodes, 2);
        assert!(!story.relay_foundation_ready);
        assert!(!story.restart_recovery_configured);

        let story_json = serde_json::to_string(&story).unwrap();
        assert!(!story_json.contains("recovery-missing-a.example"));
        assert!(!story_json.contains("recovery-missing-b.example"));
        assert!(!story_json.contains(&hex::encode(first_node_id)));
        assert!(!story_json.contains(&hex::encode(second_node_id)));
        assert!(!story_json.contains("encrypted_blob"));
        assert!(!story_json.contains("receiver_pubkey"));
    }

    #[test]
    fn test_cleanup_expired_degrades_and_retains_old_peers() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_001_000);

        store.upsert_verified(descriptor, 1_700_000_100).unwrap();
        assert_eq!(store.cleanup_expired(1_700_002_000), 1);
        assert_eq!(store.len(), 1);
        assert_eq!(store.snapshot(1_700_002_001).valid_peers, 0);
        assert_eq!(store.cleanup_expired(1_700_002_100), 0);

        let status = store.status(1_700_002_001);
        assert_eq!(status.runtime.expired_removed, 0);
        assert_eq!(status.runtime.expired_degraded, 1);
        assert_eq!(status.runtime.last_cleanup_at, Some(1_700_002_000));
        assert_eq!(status.peer_summary.expired_peers, 1);
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_expired"
                && event.outcome == "degraded"
                && event.source == "cleanup"
                && event.reason.as_deref() == Some("descriptor_expired_retained")
        }));
        assert!(status.recent_audit_events.iter().any(|event| {
            event.action == "expired_peer_cleanup"
                && event.outcome == "accepted"
                && event.detail.contains("degraded=1")
                && event.detail.contains("removed=0")
        }));

        let public_snapshot =
            store.export_bootstrap_snapshot(1_700_002_002, 1_700_002_002, false, None);
        assert_eq!(public_snapshot.peers.len(), 0);
        let cache_snapshot = store.export_peer_cache_snapshot(1_700_002_002);
        assert_eq!(cache_snapshot.peers.len(), 1);
    }

    #[test]
    fn test_peer_cache_snapshot_retains_expired_signed_records_without_making_them_live() {
        let store = PeerStore::new();
        let expired = signed_descriptor(1, 1_700_001_000);
        let node_id = expired.node_id();
        let snapshot = NodeBootstrapSnapshot::new(1_700_002_000, vec![expired.clone()]);

        let report = store.load_peer_cache_snapshot_from_source(&snapshot, 1_700_002_000, "cache");

        assert_eq!(
            report,
            PeerStoreImportReport {
                total: 1,
                inserted: 1,
                unchanged: 0,
                stale: 0,
                rejected: 0,
            }
        );
        assert_eq!(store.len(), 1);
        assert!(store.get_valid(&node_id, 1_700_002_000).is_none());
        let status = store.status(1_700_002_000);
        assert_eq!(status.snapshot.valid_peers, 0);
        assert_eq!(status.peer_summary.expired_peers, 1);
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_expired"
                && event.outcome == "retained"
                && event.source == "cache"
                && event.reason.as_deref() == Some("signature_valid_descriptor_expired")
        }));

        let mut tampered = expired;
        tampered.signature[0] ^= 0x01;
        let rejected = store.load_peer_cache_snapshot_from_source(
            &NodeBootstrapSnapshot::new(1_700_002_010, vec![tampered]),
            1_700_002_010,
            "cache",
        );
        assert_eq!(rejected.rejected, 1);
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
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_inserted"
                && event.outcome == "accepted"
                && event.source == "unknown"
                && event.sequence == Some(1)
        }));
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_rejected"
                && event.outcome == "rejected"
                && event.source == "unknown"
                && event.reason.as_deref() == Some("verification_failed")
        }));
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
        let status = store.status(1_700_000_100);
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_refreshed"
                && event.outcome == "ignored"
                && event.source == "unknown"
                && event.reason.as_deref() == Some("same_sequence")
        }));
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_rejected"
                && event.outcome == "rejected"
                && event.source == "unknown"
                && event.reason.as_deref() == Some("stale_sequence")
        }));
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
    fn test_heartbeat_signed_peer_records_export_only_verifiable_live_records() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let valid = signed_descriptor(1, 1_700_001_000);
        let valid_node_id = valid.node_id();
        let expired = signed_descriptor(1, 1_699_999_999);
        let mut tampered = signed_descriptor(1, 1_700_001_000);
        tampered.signature[0] ^= 0x01;

        store.upsert_verified(valid, now).unwrap();
        store.peers.write().insert(expired.node_id(), expired);
        store.peers.write().insert(tampered.node_id(), tampered);

        let signed_records = store.export_signed_peer_records_for_heartbeat(now, Some(8));

        assert_eq!(signed_records.total_retained_records, 3);
        assert_eq!(signed_records.valid_signed_records, 1);
        assert_eq!(signed_records.exported_signed_records, 1);
        assert_eq!(signed_records.records.generated_at, now);
        assert_eq!(signed_records.records.peers.len(), 1);
        assert_eq!(signed_records.records.peers[0].node_id(), valid_node_id);
        assert!(signed_records.records.peers[0].verify_at(now).is_ok());
        assert!(signed_records
            .verification_rule
            .contains("SignedNodeDescriptor::verify_at"));
        assert!(signed_records
            .privacy_boundary
            .contains("signed node discovery descriptors only"));
        assert!(store.recent_audit_events().iter().any(|event| {
            event.action == "heartbeat_signed_peer_records_export"
                && event.outcome == "accepted"
                && event.detail.contains("retained=3")
                && event.detail.contains("valid=1")
                && event.detail.contains("exported=1")
        }));
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
        let status = store.status(1_700_000_100);
        assert!(status.recent_peer_events.iter().any(|event| {
            event.event == "peer_inserted"
                && event.outcome == "accepted"
                && event.source == "gossip_announce"
                && event.sequence == Some(1)
                && event.reason.is_none()
        }));
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
    fn startup_self_check_status_is_recorded_without_config_values() {
        let store = PeerStore::new();

        store.record_startup_self_check(
            1_700_000_350,
            "warning",
            "missing=peer_cache_path,seed_endpoints,public_endpoint",
        );

        let status = store.status(1_700_000_400);
        assert_eq!(
            status.bootstrap.startup_self_check_status.as_deref(),
            Some("warning")
        );
        assert_eq!(
            status.bootstrap.startup_self_check_detail.as_deref(),
            Some("missing=peer_cache_path,seed_endpoints,public_endpoint")
        );
        assert_eq!(status.bootstrap.startup_self_check_at, Some(1_700_000_350));
        assert!(status.recent_audit_events.iter().any(|event| {
            event.action == "startup_self_check"
                && event.outcome == "warning"
                && !event.detail.contains("https://")
                && !event.detail.contains("/root/")
        }));
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
        store.record_blind_relay_rejected(1_700_000_018, "route_loop");
        store.record_blind_relay_rejected(1_700_000_019, "duplicate_route");
        store.record_blind_relay_rejected(1_700_000_020, "rate_limited");
        store.record_blind_relay_rejected(1_700_000_021, "quarantined");
        store.record_blind_relay_rejected(1_700_000_022, "blind_relay_request_timeout");
        store.record_blind_relay_rejected(1_700_000_023, "timestamp_expired");
        store.record_blind_relay_rejected(1_700_000_024, "timestamp_in_future");
        store.record_blind_relay_quarantine_started(1_700_000_025, "failure_threshold");

        let status = store.status(1_700_000_026);
        let stats = status.runtime.blind_relay;

        assert_eq!(stats.received, 15);
        assert_eq!(stats.terminal, 1);
        assert_eq!(stats.forwarded, 1);
        assert_eq!(stats.rejected, 13);
        assert_eq!(stats.backpressure_dropped, 1);
        assert_eq!(stats.invalid_signature, 1);
        assert_eq!(stats.ttl_exhausted, 1);
        assert_eq!(stats.no_route, 1);
        assert_eq!(stats.invalid_endpoint, 1);
        assert_eq!(stats.forward_failed, 2);
        assert_eq!(stats.loop_detected, 1);
        assert_eq!(stats.replay_dropped, 1);
        assert_eq!(stats.timestamp_rejected, 2);
        assert_eq!(stats.rate_limited, 1);
        assert_eq!(stats.quarantined, 1);
        assert_eq!(stats.quarantine_started, 1);
        assert_eq!(stats.last_event_at, Some(1_700_000_025));
        assert!(status
            .recent_audit_events
            .iter()
            .all(|event| !event.detail.contains("route_id")));
        assert!(status
            .recent_audit_events
            .iter()
            .all(|event| !event.detail.contains("encrypted_blob=")));
        assert!(status
            .recent_audit_events
            .iter()
            .all(|event| !event.detail.contains("encrypted_blob_bytes")));
        assert!(status.recent_audit_events.iter().any(|event| {
            event.action == "blind_relay_terminal"
                && event.detail.contains("encrypted_blob_size_bucket=lte_4kb")
        }));
    }

    #[test]
    fn test_blind_relay_quality_reports_ready_without_private_metadata() {
        let store = PeerStore::new();

        store.record_blind_relay_terminal(1_700_000_010, 2, 128);
        store.record_blind_relay_forwarded(1_700_000_011, 1);

        let quality = store.status(1_700_000_021).blind_relay_quality;

        assert_eq!(quality.status, "ready");
        assert!(quality.runtime_ready);
        assert!(quality.quality_ready);
        assert!(quality.real_relay_ready);
        assert!(!quality.synthetic_probe_ready);
        assert_eq!(quality.evidence_mode, "real_relay_traffic");
        assert_eq!(quality.proof_scope, "message_delivery");
        assert_eq!(quality.readiness_reason, "real_relay_observed");
        assert_eq!(quality.accepted_total, 2);
        assert_eq!(quality.accepted_percent, 100);
        assert_eq!(quality.last_accepted_at, Some(1_700_000_011));
        assert_eq!(quality.last_event_age_seconds, Some(10));
        assert_eq!(quality.last_accepted_age_seconds, Some(10));
        assert_eq!(quality.last_probe_age_seconds, None);
        assert!(!quality.detail.contains("https://"));
        assert!(!quality.detail.contains("route_id"));
        assert!(!quality.detail.contains("encrypted_blob"));
        assert!(!quality.detail.contains("payload"));
        assert!(quality
            .privacy_boundary
            .contains("aggregate blind relay runtime counters only"));
    }

    #[test]
    fn test_blind_relay_quality_does_not_stay_ready_after_stale_real_relay_evidence() {
        let store = PeerStore::new();

        store.record_blind_relay_terminal(1_700_000_010, 2, 128);
        store.record_blind_relay_forwarded(1_700_000_011, 1);

        let quality = store
            .status(1_700_000_011 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
            .blind_relay_quality;

        assert_eq!(quality.status, "stale");
        assert!(!quality.runtime_ready);
        assert!(!quality.quality_ready);
        assert!(!quality.real_relay_ready);
        assert!(!quality.synthetic_probe_ready);
        assert_eq!(quality.evidence_mode, "real_relay_traffic");
        assert_eq!(quality.proof_scope, "message_delivery");
        assert_eq!(quality.readiness_reason, "real_relay_stale");
        assert_eq!(quality.accepted_total, 2);
        assert_eq!(quality.last_accepted_at, Some(1_700_000_011));
        assert_eq!(
            quality.last_accepted_age_seconds,
            Some(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        );
        assert!(quality.next_action.contains("refresh relay readiness"));
        assert!(quality.detail.contains("stale_after_seconds=1800"));
        assert!(!quality.detail.contains("https://"));
        assert!(!quality.detail.contains("route_id"));
        assert!(!quality.detail.contains("encrypted_blob"));
        assert!(!quality.detail.contains("payload"));
    }

    #[test]
    fn test_two_hop_blind_relay_probe_reports_path_proof_without_private_metadata() {
        let store = PeerStore::new();

        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_010,
            true,
            "accepted",
            4,
            3,
            2,
            1,
        );

        let status = store.status(1_700_000_025);
        let stats = status.runtime.blind_relay;
        let quality = status.blind_relay_quality;

        assert_eq!(stats.two_hop_probe_attempted, 1);
        assert_eq!(stats.two_hop_probe_succeeded, 1);
        assert_eq!(stats.two_hop_probe_failed, 0);
        assert_eq!(stats.last_two_hop_probe_at, Some(1_700_000_010));
        assert_eq!(quality.status, "ready");
        assert!(quality.runtime_ready);
        assert!(quality.quality_ready);
        assert!(!quality.real_relay_ready);
        assert!(quality.synthetic_probe_ready);
        assert!(quality.two_hop_probe_ready);
        assert_eq!(quality.evidence_mode, "synthetic_two_hop_control_probe");
        assert_eq!(quality.proof_scope, "control_plane");
        assert_eq!(
            quality.readiness_reason,
            "synthetic_two_hop_control_probe_ready"
        );
        assert!(quality.detail.contains("proof_scope=control_plane"));
        assert!(quality
            .next_action
            .contains("real message-delivery evidence"));
        assert_eq!(quality.last_two_hop_probe_age_seconds, Some(15));
        assert_eq!(status.two_hop_path_proof_history.attempted, 1);
        assert_eq!(status.two_hop_path_proof_history.succeeded, 1);
        assert_eq!(status.two_hop_path_proof_history.failed, 0);
        assert_eq!(status.two_hop_path_proof_history.status, "ready");
        assert_eq!(
            status.two_hop_path_proof_history.freshness_bucket,
            "fresh_success"
        );
        assert!(status.two_hop_path_proof_history.proof_ready);
        assert!(status.two_hop_path_proof_history.recent_success_ready);
        assert!(!status.two_hop_path_proof_history.message_delivery_ready);
        assert!(
            !status
                .two_hop_path_proof_history
                .recent_message_delivery_ready
        );
        assert_eq!(
            status.two_hop_path_proof_history.message_delivery_successes,
            0
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .latest_message_delivery_age_seconds,
            None
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .consecutive_message_delivery_successes,
            0
        );
        assert!(!status.two_hop_path_proof_history.failure_streak_active);
        assert_eq!(
            status.two_hop_path_proof_history.stale_after_seconds,
            PEER_ROUTEABILITY_STALE_AFTER_SECS
        );
        assert_eq!(status.two_hop_path_proof_history.success_percent, 100);
        assert_eq!(
            status.two_hop_path_proof_history.latest_outcome.as_deref(),
            Some("accepted")
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .latest_reason_bucket
                .as_deref(),
            Some("accepted")
        );
        assert_eq!(
            status.two_hop_path_proof_history.latest_age_seconds,
            Some(15)
        );
        assert_eq!(
            status.two_hop_path_proof_history.latest_success_age_seconds,
            Some(15)
        );
        assert_eq!(
            status.two_hop_path_proof_history.latest_failure_age_seconds,
            None
        );
        assert_eq!(
            status.two_hop_path_proof_history.proof_scope,
            "control_plane"
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .proof_scope_counts
                .get("control_plane"),
            Some(&1)
        );
        assert_eq!(status.two_hop_path_proof_history.consecutive_successes, 1);
        assert_eq!(status.two_hop_path_proof_history.consecutive_failures, 0);
        assert_eq!(status.two_hop_path_proof_history.events.len(), 1);
        assert_eq!(
            status.two_hop_path_proof_history.events[0].path_shape,
            "entry_middle_terminal"
        );
        assert_eq!(status.two_hop_path_proof_history.events[0].hop_count, 2);
        assert_eq!(
            status.two_hop_path_proof_history.events[0].path_policy,
            "distinct_middle_terminal"
        );
        assert_eq!(
            status.two_hop_path_proof_history.events[0].middle_candidate_bucket,
            "healthy"
        );
        assert_eq!(
            status.two_hop_path_proof_history.events[0].terminal_candidate_bucket,
            "few"
        );
        assert_eq!(
            status.two_hop_path_proof_history.events[0].ttl_shape,
            "entry_ttl_2_onward_ttl_1"
        );
        assert_eq!(
            status.two_hop_path_proof_history.events[0].evidence_mode,
            "synthetic_two_hop_control_probe"
        );
        assert_eq!(
            status.two_hop_path_proof_history.events[0].proof_scope,
            "control_plane"
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .path_shape_counts
                .get("entry_middle_terminal"),
            Some(&1)
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .candidate_pool_counts
                .get("forming"),
            Some(&1)
        );
        assert_eq!(
            status
                .two_hop_path_proof_history
                .ttl_shape_counts
                .get("entry_ttl_2_onward_ttl_1"),
            Some(&1)
        );
        assert!(status.recent_audit_events.iter().any(|event| {
            event.action == "blind_relay_two_hop_probe"
                && event.outcome == "accepted"
                && event.detail.contains("reason_bucket=accepted")
                && event.detail.contains("middle_candidates=healthy")
                && event.detail.contains("terminal_candidates=few")
                && event.detail.contains("ttl_shape=entry_ttl_2_onward_ttl_1")
        }));
        let audit_detail = status
            .recent_audit_events
            .iter()
            .find(|event| event.action == "blind_relay_two_hop_probe")
            .map(|event| event.detail.as_str())
            .unwrap_or_default();
        assert!(!audit_detail.contains("endpoint"));
        assert!(!audit_detail.contains("node_id"));
        assert!(!audit_detail.contains("route_id"));
        assert!(!audit_detail.contains("payload"));

        assert!(!quality.detail.contains("route_id"));
        assert!(!quality.detail.contains("endpoint"));
        assert!(!quality.detail.contains("node_id"));
        assert!(!quality.detail.contains("encrypted_blob"));
        assert!(!quality.detail.contains("payload"));
        assert!(!quality.detail.contains("client_ip"));
        assert!(!status
            .two_hop_path_proof_history
            .privacy_boundary
            .contains("route_id"));
        assert!(status
            .two_hop_path_proof_history
            .privacy_boundary
            .contains("no node IDs"));
    }

    #[test]
    fn test_blind_relay_quality_does_not_keep_ready_from_stale_two_hop_probe() {
        let store = PeerStore::new();

        store.record_blind_relay_two_hop_probe_result(1_700_000_010, true, "accepted");

        let quality = store
            .status(1_700_000_010 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
            .blind_relay_quality;

        assert_eq!(quality.status, "stale");
        assert!(!quality.runtime_ready);
        assert!(!quality.quality_ready);
        assert!(!quality.real_relay_ready);
        assert!(!quality.synthetic_probe_ready);
        assert!(!quality.two_hop_probe_ready);
        assert_eq!(quality.evidence_mode, "synthetic_two_hop_control_probe");
        assert_eq!(quality.proof_scope, "control_plane");
        assert_eq!(
            quality.readiness_reason,
            "synthetic_two_hop_control_probe_stale"
        );
        assert_eq!(
            quality.last_two_hop_probe_age_seconds,
            Some(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        );
        assert_eq!(quality.last_accepted_age_seconds, None);
        assert!(quality.next_action.contains("synthetic path proof"));
        assert!(!quality.detail.contains("https://"));
        assert!(!quality.detail.contains("route_id"));
        assert!(!quality.detail.contains("encrypted_blob"));
        assert!(!quality.detail.contains("payload"));
    }

    #[test]
    fn test_two_hop_onion_delivery_probe_reports_message_delivery_scope() {
        let store = PeerStore::new();

        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_010,
            true,
            "onion_terminal_delivered",
            3,
            2,
            2,
            1,
        );

        let status = store.status(1_700_000_015);
        let history = status.two_hop_path_proof_history;

        assert_eq!(
            history.latest_reason_bucket.as_deref(),
            Some("onion_terminal_delivered")
        );
        assert_eq!(history.proof_scope, "message_delivery");
        assert_eq!(history.proof_scope_counts.get("message_delivery"), Some(&1));
        assert!(history.proof_ready);
        assert!(history.recent_success_ready);
        assert!(history.message_delivery_ready);
        assert!(history.recent_message_delivery_ready);
        assert_eq!(history.message_delivery_successes, 1);
        assert_eq!(history.latest_message_delivery_age_seconds, Some(5));
        assert_eq!(history.consecutive_message_delivery_successes, 1);
        assert_eq!(history.events.len(), 1);
        assert_eq!(history.events[0].evidence_mode, "real_relay_traffic");
        assert_eq!(history.events[0].proof_scope, "message_delivery");
        assert!(!history.privacy_boundary.contains("route_id="));
        assert!(!history.privacy_boundary.contains("receiver="));
        assert!(!history.privacy_boundary.contains("encrypted_blob="));
    }

    #[test]
    fn test_two_hop_message_delivery_readiness_distinguishes_latest_from_recent() {
        let store = PeerStore::new();

        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_010,
            true,
            "onion_terminal_delivered",
            3,
            3,
            2,
            1,
        );
        store.record_blind_relay_two_hop_probe_result_with_context(
            1_700_000_020,
            true,
            "accepted",
            3,
            3,
            2,
            1,
        );

        let history = store.status(1_700_000_030).two_hop_path_proof_history;

        assert!(history.proof_ready);
        assert!(history.recent_success_ready);
        assert!(!history.message_delivery_ready);
        assert!(history.recent_message_delivery_ready);
        assert_eq!(history.message_delivery_successes, 1);
        assert_eq!(history.latest_message_delivery_age_seconds, Some(20));
        assert_eq!(history.consecutive_successes, 2);
        assert_eq!(history.consecutive_message_delivery_successes, 0);
        assert_eq!(history.proof_scope, "control_plane");
        assert_eq!(history.proof_scope_counts.get("message_delivery"), Some(&1));
        assert_eq!(history.proof_scope_counts.get("control_plane"), Some(&1));

        let serialized = serde_json::to_string(&history).expect("history serializes");
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("endpoint="));
        assert!(!serialized.contains("https://"));
        assert!(!serialized.contains("http://"));
        assert!(!serialized.contains("receiver="));
        assert!(!serialized.contains("payload="));
        assert!(!serialized.contains("node_id"));
    }

    #[test]
    fn test_two_hop_path_proof_history_buckets_failures_without_private_metadata() {
        let store = PeerStore::new();

        store.record_blind_relay_two_hop_probe_result(1_700_000_010, false, "http_502");
        store.record_blind_relay_two_hop_probe_result(1_700_000_020, false, "endpoint://leak");
        store.record_blind_relay_two_hop_probe_result(
            1_700_000_030,
            false,
            "two_hop_blind_relay_probe_timeout",
        );
        store.record_blind_relay_two_hop_probe_result(1_700_000_040, true, "accepted");

        let status = store.status(1_700_000_055);
        let history = status.two_hop_path_proof_history;

        assert_eq!(history.window_size, MAX_TWO_HOP_PATH_PROOF_EVENTS);
        assert_eq!(history.retained_events, 4);
        assert_eq!(history.attempted, 4);
        assert_eq!(history.succeeded, 1);
        assert_eq!(history.message_delivery_successes, 0);
        assert_eq!(history.failed, 3);
        assert_eq!(history.status, "ready");
        assert_eq!(history.freshness_bucket, "fresh_success");
        assert!(history.proof_ready);
        assert!(history.recent_success_ready);
        assert!(!history.message_delivery_ready);
        assert!(!history.recent_message_delivery_ready);
        assert!(!history.failure_streak_active);
        assert_eq!(history.success_percent, 25);
        assert_eq!(history.latest_outcome.as_deref(), Some("accepted"));
        assert_eq!(history.latest_reason_bucket.as_deref(), Some("accepted"));
        assert_eq!(history.latest_age_seconds, Some(15));
        assert_eq!(history.latest_success_age_seconds, Some(15));
        assert_eq!(history.latest_failure_age_seconds, Some(25));
        assert_eq!(history.latest_message_delivery_age_seconds, None);
        assert_eq!(history.consecutive_successes, 1);
        assert_eq!(history.consecutive_failures, 0);
        assert_eq!(history.consecutive_message_delivery_successes, 0);
        assert_eq!(history.events[0].reason_bucket, "http_error");
        assert_eq!(history.events[1].reason_bucket, "unknown");
        assert_eq!(history.events[2].reason_bucket, "request_error");
        assert_eq!(history.events[3].reason_bucket, "accepted");
        assert_eq!(history.reason_bucket_counts.get("http_error"), Some(&1));
        assert_eq!(history.reason_bucket_counts.get("unknown"), Some(&1));
        assert_eq!(history.reason_bucket_counts.get("request_error"), Some(&1));
        assert_eq!(history.reason_bucket_counts.get("accepted"), Some(&1));
        assert_eq!(
            history.failure_reason_bucket_counts.get("http_error"),
            Some(&1)
        );
        assert_eq!(
            history.failure_reason_bucket_counts.get("unknown"),
            Some(&1)
        );
        assert_eq!(
            history.failure_reason_bucket_counts.get("request_error"),
            Some(&1)
        );
        assert_eq!(history.failure_reason_bucket_counts.get("accepted"), None);
        assert_eq!(
            history.path_shape_counts.get("entry_middle_terminal"),
            Some(&4)
        );
        assert_eq!(history.candidate_pool_counts.get("incomplete"), Some(&4));
        assert_eq!(
            history.ttl_shape_counts.get("entry_ttl_2_onward_ttl_1"),
            Some(&4)
        );

        let serialized = serde_json::to_string(&history).expect("history serializes");
        assert!(!serialized.contains("endpoint://leak"));
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("node_id"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("payload="));
        assert!(!serialized.contains("client_ip"));
    }

    #[test]
    fn test_two_hop_path_proof_history_marks_attention_and_stale_states() {
        let failing_store = PeerStore::new();
        failing_store.record_blind_relay_two_hop_probe_result(1_700_000_010, false, "ack_rejected");

        let failing_history = failing_store
            .status(1_700_000_020)
            .two_hop_path_proof_history;
        assert_eq!(failing_history.status, "attention");
        assert_eq!(failing_history.freshness_bucket, "recent_failure");
        assert!(!failing_history.proof_ready);
        assert!(!failing_history.recent_success_ready);
        assert!(failing_history.failure_streak_active);
        assert_eq!(failing_history.latest_success_age_seconds, None);
        assert_eq!(failing_history.latest_failure_age_seconds, Some(10));
        assert_eq!(failing_history.consecutive_failures, 1);
        assert!(failing_history.next_action.contains("routeability"));

        let stale_store = PeerStore::new();
        stale_store.record_blind_relay_two_hop_probe_result(1_700_000_010, true, "accepted");
        let stale_history = stale_store
            .status(1_700_000_010 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
            .two_hop_path_proof_history;
        assert_eq!(stale_history.status, "stale");
        assert_eq!(stale_history.freshness_bucket, "stale_success");
        assert!(!stale_history.proof_ready);
        assert!(!stale_history.recent_success_ready);
        assert!(!stale_history.failure_streak_active);
        assert_eq!(
            stale_history.latest_success_age_seconds,
            Some(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        );
        assert_eq!(stale_history.latest_failure_age_seconds, None);
        assert_eq!(stale_history.consecutive_successes, 1);
        assert!(stale_history
            .next_action
            .contains("fresh two-hop path proof"));
    }

    #[test]
    fn test_blind_relay_quality_marks_timestamp_replay_protection_active() {
        let store = PeerStore::new();

        store.record_blind_relay_rejected(1_700_000_010, "timestamp_expired");

        let status = store.status(1_700_000_020);
        let stats = status.runtime.blind_relay;
        let quality = status.blind_relay_quality;

        assert_eq!(stats.timestamp_rejected, 1);
        assert_eq!(quality.status, "protecting");
        assert!(!quality.runtime_ready);
        assert!(!quality.quality_ready);
        assert!(!quality.real_relay_ready);
        assert_eq!(quality.evidence_mode, "real_relay_attempted");
        assert_eq!(quality.readiness_reason, "protection_active");
        assert_eq!(quality.timestamp_rejected, 1);
        assert!(quality.protection_active);
        assert_eq!(quality.last_event_age_seconds, Some(10));
        assert!(!quality.detail.contains("route_id"));
        assert!(!quality.detail.contains("endpoint"));
        assert!(!quality.detail.contains("encrypted_blob"));
        assert!(!quality.detail.contains("payload"));
    }

    #[test]
    fn test_blind_relay_quality_surfaces_transport_attention_without_endpoint_data() {
        let store = PeerStore::new();

        store.record_blind_relay_forwarded(1_700_000_010, 1);
        store.record_blind_relay_retry_attempt(1_700_000_011, "blind_relay_request_timeout");
        store.record_blind_relay_retry_exhausted(1_700_000_012, 2, "blind_relay_request_timeout");
        store.record_blind_relay_rejected(1_700_000_013, "blind_relay_request_timeout");

        let quality = store.status(1_700_000_018).blind_relay_quality;

        assert_eq!(quality.status, "attention");
        assert!(quality.runtime_ready);
        assert!(!quality.quality_ready);
        assert!(quality.real_relay_ready);
        assert_eq!(quality.evidence_mode, "real_relay_traffic");
        assert_eq!(quality.proof_scope, "message_delivery");
        assert_eq!(quality.readiness_reason, "real_relay_transport_attention");
        assert_eq!(quality.forward_failed, 1);
        assert_eq!(quality.retry_exhausted, 1);
        assert_eq!(quality.last_event_age_seconds, Some(5));
        assert!(quality.next_action.contains("next-hop reachability"));
        assert!(!quality.detail.contains("https://"));
        assert!(!quality.detail.contains("endpoint"));
        assert!(!quality.detail.contains("route_id"));
        assert!(!quality.detail.contains("encrypted_blob"));
        assert!(!quality.detail.contains("payload"));
    }

    #[test]
    fn test_blind_relay_probe_quality_does_not_inflate_real_traffic_counters() {
        let store = PeerStore::new();

        store.record_blind_relay_probe_result(1_700_000_010, true, "accepted");

        let status = store.status(1_700_000_020);
        let stats = status.runtime.blind_relay;
        let quality = status.blind_relay_quality;

        assert_eq!(stats.received, 0);
        assert_eq!(stats.terminal, 0);
        assert_eq!(stats.forwarded, 0);
        assert_eq!(stats.probe_attempted, 1);
        assert_eq!(stats.probe_succeeded, 1);
        assert_eq!(stats.probe_failed, 0);
        assert_eq!(stats.last_probe_at, Some(1_700_000_010));
        assert_eq!(quality.accepted_total, 0);
        assert!(quality.runtime_ready);
        assert!(quality.quality_ready);
        assert!(!quality.real_relay_ready);
        assert!(quality.synthetic_probe_ready);
        assert_eq!(quality.evidence_mode, "synthetic_probe");
        assert_eq!(quality.proof_scope, "single_hop_control_plane");
        assert_eq!(quality.readiness_reason, "synthetic_probe_ready");
        assert_eq!(quality.probe_attempted, 1);
        assert_eq!(quality.probe_succeeded, 1);
        assert_eq!(quality.probe_failed, 0);
        assert_eq!(quality.timestamp_rejected, 0);
        assert_eq!(quality.last_probe_age_seconds, Some(10));
        assert!(quality.detail.contains("last_probe_age_seconds=10"));
        assert!(quality.detail.contains("evidence_mode=synthetic_probe"));
        assert!(quality
            .detail
            .contains("readiness_reason=synthetic_probe_ready"));
        assert!(quality
            .next_action
            .contains("wait for real encrypted relay traffic"));

        let serialized = serde_json::to_string(&quality).unwrap();
        assert!(!serialized.contains("route_id"));
        assert!(!serialized.contains("encrypted_blob"));
        assert!(!serialized.contains("payload_b64"));
        assert!(!serialized.contains("client_ip"));
        assert!(!serialized.contains("http://"));
        assert!(!serialized.contains("https://"));
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
    fn test_peer_cache_load_evidence_is_separate_from_generic_recovery_status() {
        let store = PeerStore::new();

        store.record_bootstrap_source(1_700_000_010, "cache", "failed", "json_rejected");
        store.record_bootstrap_source(
            1_700_000_011,
            "cache_backup",
            "success",
            "total=2 inserted=2 unchanged=0 stale=0 rejected=0",
        );
        store.record_gossip_round(1_700_000_050, 2, 2, 1, None);

        let status = store.status(1_700_000_060);
        assert_eq!(
            status.bootstrap.last_cache_load_source.as_deref(),
            Some("cache_backup")
        );
        assert_eq!(
            status.bootstrap.last_cache_load_status.as_deref(),
            Some("success")
        );
        assert_eq!(
            status.bootstrap.last_cache_load_detail.as_deref(),
            Some("total=2 inserted=2 unchanged=0 stale=0 rejected=0")
        );
        assert_eq!(status.bootstrap.last_cache_load_at, Some(1_700_000_011));
        assert_eq!(status.bootstrap.recovery_status.as_deref(), Some("success"));
        assert_eq!(
            status.bootstrap.recovery_detail.as_deref(),
            Some("gossip_recovered attempted=2 succeeded=2 seed_attempted=1")
        );
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
    fn test_peer_quorum_reports_peer_view_ready_without_routeable_endpoint() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        store.configure_bootstrap_status(true, true, true, 2);
        store
            .upsert_verified_from_source(signed_descriptor(1, now + 1_000), now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(signed_descriptor(1, now + 1_000), now, "gossip_snapshot")
            .unwrap();
        store.record_gossip_round(now + 20, 2, 2, 1, None);

        let status = store.status(now + 60);

        assert_eq!(status.peer_quorum.status, "peer_view_ready");
        assert!(!status.peer_quorum.quorum_ready);
        assert_eq!(status.peer_quorum.valid_peers, 2);
        assert_eq!(status.peer_quorum.routeable_chat_relays, 0);
        assert_eq!(status.peer_quorum.healthy_ratio_percent, 100);
        assert!(status
            .peer_quorum
            .next_action
            .contains("public chat relay endpoint"));
    }

    #[test]
    fn test_peer_quorum_ready_requires_fresh_routeable_restart_recoverable_peers() {
        let store = PeerStore::new();
        let now = 1_700_000_100;
        let first_kp = IdentityKeyPair::generate();
        let second_kp = IdentityKeyPair::generate();

        let mut first = signed_descriptor_for(&first_kp, 1, now + 1_000);
        first.descriptor.public_endpoint = Some("https://peer-one.example".to_string());
        first = SignedNodeDescriptor::sign(first.descriptor, &first_kp).unwrap();
        let first_node_id = first.node_id();

        let mut second = signed_descriptor_for(&second_kp, 1, now + 1_000);
        second.descriptor.public_endpoint = Some("https://peer-two.example".to_string());
        second = SignedNodeDescriptor::sign(second.descriptor, &second_kp).unwrap();
        let second_node_id = second.node_id();

        store.configure_bootstrap_status(true, true, true, 2);
        store
            .upsert_verified_from_source(first, now, "gossip_announce")
            .unwrap();
        store
            .upsert_verified_from_source(second, now, "gossip_snapshot")
            .unwrap();
        store.record_gossip_round(now + 20, 2, 2, 1, None);
        store.record_route_forward_success(&first_node_id, now + 30);
        store.record_route_forward_success(&second_node_id, now + 31);

        let status = store.status(now + 60);

        assert_eq!(status.peer_quorum.status, "route_ready");
        assert!(status.peer_quorum.quorum_ready);
        assert_eq!(status.peer_quorum.min_valid_peers, 2);
        assert_eq!(status.peer_quorum.valid_peers, 2);
        assert_eq!(status.peer_quorum.healthy_peers, 2);
        assert_eq!(status.peer_quorum.routeable_chat_relays, 2);
        assert!(status.peer_quorum.restart_recovery_configured);
        assert!(status.peer_quorum.relay_foundation_ready);
        assert!(status
            .peer_quorum
            .privacy_boundary
            .contains("not public-chain consensus"));
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
        assert_eq!(
            status.stability.restart_recovery_sources,
            vec!["seed_endpoints".to_string(), "peer_cache".to_string()]
        );
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
        assert!(status.stability.restart_recovery_sources.is_empty());
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
        assert_eq!(
            status.stability.restart_recovery_sources,
            vec!["peer_cache".to_string()]
        );
    }
}
