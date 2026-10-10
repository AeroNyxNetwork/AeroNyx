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
//! - Signature-verified expired descriptor lookup for operator-pinned startup
//!   evidence only; expired records remain forbidden for routing and liveness
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
//! - Directory proof-gossip convergence status with bounded, mutually
//!   exclusive rejection buckets and no peer, endpoint, producer, block, or
//!   descriptor dimensions
//! - A side-effect-free internal view of every valid public endpoint identity
//!   for fail-closed duplicate-endpoint handling during outbound gossip
//! - Expired-peer cleanup counters so stale descriptor eviction is observable
//!   without exposing peer endpoints or user traffic metadata
//! - Health-ranked route candidates for blind relay preparation, using only
//!   node-level signed descriptor metadata and never encrypted payload content
//! - [ROUTE-DOMAIN-ATTESTED-SELECTION 2026-08-03 by Codex] Optional fail-closed
//!   multi-hop admission under host-local opaque route-domain pins and a
//!   portable independent-attestor quorum, while preserving direct single-hop
//!   compatibility and never publishing trust identities or domain tokens
//! - [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] Deterministic,
//!   bounded host-cache export and verifier-local restart revalidation for
//!   current portable route-domain certificates
//! - Blind relay runtime counters and drop reason buckets for nodeboard,
//!   without exposing encrypted payloads, peer endpoint URLs, or user metadata
//! - [SIGNED-FAILURE-RECEIPT 2026-08-11 by Codex] Counts invalid hop-local
//!   failure receipts as forward failures without retaining signed material
//! - [SIGNED-PROTOCOL-FEATURES 2026-08-11 by Codex] Binds recognized signed
//!   wire-feature advertisements into route-surface evidence without changing
//!   hashes for legacy descriptors that advertise no feature token
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
//! - [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] Blind relay
//!   abuse-guard counters cover aggregate parser-front admission plus verified
//!   previous-hop fairness and quarantine without identity or route dimensions
//! - Per-peer health summary for operators, using only signed node metadata,
//!   gossip/import observation buckets, route-health counters, and relay
//!   protection buckets without payload or route reconstruction data
//! - Peer-cache startup recovery evidence that records cache/backup load
//!   status separately from generic bootstrap source status, so nodeboard can
//!   diagnose restart recovery without exposing cache paths or peer endpoints
//! - [PEER-CACHE-PERSISTENCE-SPLIT 2026-09-25 by Codex] Local cache load,
//!   export, recovery evidence, and dirty notification live in the private
//!   `cache_persistence` module; public PeerStore methods remain unchanged
//! - [PEER-STORE-ROUTE-SELECTION-SPLIT 2026-09-25 by Codex] Exact peer lookup,
//!   routeability scoring, and strict multi-hop path selection live in a
//!   private module with the same fail-closed target and capability checks
//! - [PEER-STORE-STATUS-SPLIT 2026-09-25 by Codex] Read-only, privacy-safe
//!   health, route and network status projections live in the private
//!   `status` module without changing live routing or lock acquisition order
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
//!   low-frequency route readiness evidence tracked separately from any claim
//!   about App/user encrypted traffic
//! - Blind relay evidence semantics separate opaque accepted relay work from
//!   synthetic probes without claiming that unclassified work is App/user traffic
//! - Bounded two-hop path proof history records recent entry -> middle ->
//!   terminal proof outcomes for nodeboard/public status without exposing node
//!   IDs, route IDs, endpoints, payloads, or social graph metadata
//! - [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex] Keeps an independent
//!   bounded entry -> middle -> middle -> terminal runtime proof history so a
//!   three-hop failure cannot overwrite mature two-hop readiness
//! - [PATH-PROOF-CLOCK-GUARD 2026-08-03 by Codex] Fails closed when retained
//!   path-proof evidence is future-dated relative to the current node clock,
//!   preventing clock rollback from turning future evidence into fresh proof
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
//!   from synthetic terminal store-and-forward proof so public surfaces never
//!   present node-generated checks as App/user chat delivery
//! - Mixed-version two-hop ACKs remain control-plane compatibility evidence;
//!   only terminal-signed receipts can enter message-delivery proof history
//! - Two-hop message-delivery readiness exposes fresh synthetic terminal
//!   ChatRelay proof as its own aggregate gate, so App/nodeboard/backend can
//!   distinguish onion reachability from tested store-and-forward capability
//! - Onion middle-hop recovery can distinguish route quarantine from ordinary
//!   unknown routeability, allowing cold-start proof attempts without sending
//!   through peers that are actively isolated by local route health policy
//! - Network story readiness treats a fresh successful two-hop path proof as
//!   onion-ready evidence, preventing proven delivery paths from being hidden by
//!   conservative local route-candidate planning after restart
//! - Two-hop path proof stability windows expose recent success rate, proof-age
//!   buckets, and failure-circuit-breaker state so nodeboard/backend can
//!   distinguish a single fresh proof from a repeated stable encrypted route
//!   foundation without exposing route metadata
//! - Route governance summary condenses routeability, scoring, quarantine,
//!   and degradation state into one aggregate nodeboard/backend contract
//!   without exposing endpoints, route IDs, payloads, or peer graph data
//! - Local peer-cache snapshots can retain descriptor-bound successful
//!   routeability evidence across a short process restart while rejecting
//!   stale, future-dated, mismatched, quarantined, or unsigned state
//! - Routeability evidence follows the signed route surface across ordinary
//!   descriptor sequence/TTL refreshes, but endpoint, capability, discovery
//!   visibility, or onion KEM changes fail closed until a new direct success
//! - External delivery-cache witness rounds expose aggregate continuity status
//!   and can clear only restored delivery readiness before listeners, without
//!   retaining witness identities, opaque digests, or traffic metadata
//! - [PURPOSE-BOUND-RECEIPT-EVIDENCE 2026-08-10 by Codex] Keeps v2
//!   purpose-bound receipt interoperability as process-local, freshness-bounded
//!   evidence; legacy v1 receipt framing never authorizes App onion routes
//! - [RECEIPT-EVIDENCE-LIFECYCLE 2026-08-10 by Codex] Binds v2 receipt
//!   authority to the current signed route surface and excludes invalid peers
//!   from readiness counts, candidate selection, and capability queries
//! - [RECEIPT-EVIDENCE-SURFACE-BINDING 2026-08-10 by Codex] Stores the signed
//!   route-surface fingerprint beside every v2 receipt observation so a
//!   concurrent endpoint/KEM/capability rotation cannot inherit old authority
//! - [CLIENT-DELIVERY-ATOMIC-ROUTE-EVIDENCE 2026-08-11 by Codex] Commits both
//!   hop capabilities, both route successes, and the aggregate real-delivery
//!   counter only while one coherent signed two-hop snapshot remains current
//! - [PEER-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Admits route failures,
//!   blind-relay rejections, and quarantine events through closed reason
//!   vocabularies before they can affect reputation or public diagnostics
//! - [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Persists only active,
//!   signed-route-bound quarantine windows in the host-local signed peer cache
//!   so a process restart cannot revive a currently isolated route
//! - [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Reports only fixed
//!   aggregate recovery-anchor decisions for routeability/quarantine and uses
//!   the same closed vocabulary for local cache-rejection audit evidence
//! - [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] Revokes the complete
//!   anchor-v3 restart-readiness bundle on adverse external witness evidence
//!   while retaining verified descriptors for bounded fresh probing
//! - [EXTERNAL-WITNESS-GENERATION-BINDING 2026-08-21 by Codex] Exposes the
//!   aggregate restored cache generation so startup witnessing cannot protect
//!   a different local anchor generation after an interrupted atomic update
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
//! 8. Directory-proof gossip is admitted only by a caller that has already
//!    matched the proof against an audited local replica anchor
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
//! - Path-proof readiness must use only events at or before the current node
//!   time. Future-dated retained evidence is an aggregate clock-health signal,
//!   never fresh relay proof.
//! - Recovery-anchor v3 authorizes route-state restore. Older anchors remain
//!   parseable, but must never authorize routeability or quarantine recovery.
//! - External witness rejection is applied only during startup, before public
//!   listeners. Do not reuse that bulk reset as a runtime route-health tool.
//!
//! ## Last Modified
//! v0.92.0-RouteSelectionSplit - Isolate live route candidate ranking and
//! exact-target selection without changing public PeerStore APIs or policy
//! v0.91.0-StatusProjectionSplit - Isolate read-only aggregate node status
//! while preserving all existing inherent methods and privacy boundaries
//! v0.90.0-OpenNodeAdmission - [OPEN-NODE-ADMISSION 2026-09-24 by Codex]
//! Added a canonical, bounded, permissionless descriptor candidate boundary
//! with strict endpoint/shape checks and no route or economic authority
//! v0.89.0-ExpiredCacheSequenceFencing - Rejects conflicting authentic expired
//! cache descriptors that reuse one node identity and sequence after restart
//! v0.88.0-BlindRelayGlobalAdmission - Broadened the existing privacy-safe
//! rate-limit aggregate to include parser-front identity-rotation protection
//! v0.87.0-ExternalWitnessGenerationBinding - Bound startup witness decisions
//! to the exact aggregate generation currently represented by restored state
//! v0.86.0-ExternalWitnessRouteGate - Applied adverse external v3 witness
//! evidence to all restored readiness sections without deleting descriptors
//! v0.85.0-RouteStateRollbackAnchor - Bound routeability and quarantine to the
//! monotonic signed v3 recovery anchor with privacy-safe operator status
//! v0.84.0-RouteQuarantineRecovery - Added signed, expiry-bounded restart
//! recovery for active route quarantine without retaining failure details
//! v0.83.0-PeerHealthReasonBoundary - Replaced open-text route-health and
//! relay-protection diagnostics with compatibility-preserving reason admission
//! v0.82.0-CustodyWitnessAdmission - Isolated custody requester pins from
//! permissionless discovery and verified-delivery witness authority
//! v0.81.0-SignedProtocolFeatures - Bound negotiated response contracts into
//! route-surface fingerprints while preserving legacy cache compatibility
//! v0.80.0-SignedFailureReceipt - Classify invalid authenticated failure ACKs
//! without storing route, receipt, endpoint, or payload material
//! v0.79.0-ClientDeliveryAtomicRouteEvidence - Made real two-hop receipt
//! evidence an all-or-nothing signed-route state transition
//! v0.78.0-RouteSuccessSurfaceBinding - Bound successful forward evidence to
//! the exact signed route surface used for the outbound request
//! v0.77.0-ReceiptEvidenceSurfaceBinding - Bound every purpose-separated v2
//! receipt observation to the exact signed route surface used by the caller
//! v0.76.0-ReceiptEvidenceLifecycle - Revoked purpose-bound receipt authority
//! on route-surface rotation and removed invalid peers from readiness counts
//! v0.75.0-PurposeBoundReceiptEvidence - Made the v2-only route authority
//! explicit while preserving public status and legacy method compatibility
//! v0.74.0-BlindVaultReplicaCapability - Bound the append-only anonymous
//! ciphertext replica capability into routeability evidence fingerprints
//! v0.73.0-RouteDomainCertificateRecovery - Persist only currently valid
//! route-domain certificates and reverify each record after restart
//! v0.72.0-PathProofClockGuard - Ignore future-dated path proofs and fail
//! readiness closed until the node clock catches up
//! v0.71.0-PathProofRollbackAnchor - Track independent local-anchor decisions
//! for signed two-hop and three-hop proof recovery without retaining digests
//! v0.70.0-ThreeHopSignedRecovery - Added independently signed, route-pool-
//! bound three-hop aggregate proof persistence and warm-restart recovery
//! v0.69.0-ThreeHopRuntimeProof - Added independent privacy-safe three-hop
//! message-delivery proof history while preserving the existing two-hop wire contract
//! v0.68.0-TwoHopProbeOutcome - Prevented legacy control ACKs from being
//! classified as terminal message-delivery evidence
//! v0.67.0-DiscoveryIdentityAmbiguity - Added a complete, lightweight
//! valid-public endpoint identity view for fail-closed gossip hint resolution
//! v0.66.0-DiscoveryGossipIsolation - Distinguish failed proof capability
//! negotiation from a valid legacy-only peer in aggregate health
//! v0.65.0-DirectoryProofGossipReliability - Added privacy-safe convergence,
//! fallback, and rejection-bucket status for authenticated descriptor gossip
//! v0.64.0-DirectoryAuthenticatedGossipAdmission - Added a shared single-peer
//! import report path and fail-closed handling for unverified proof gossip
//! v0.63.0-DirectoryMirrorCarrierCapability - Added the signed mirror-carrier
//! role to route-surface fingerprints and privacy-safe capability labels
//! v0.62.0-VerifiedDeliveryWitnessAdmission - Added fail-closed bilateral requester pins for witness writes
//! v0.61.0-VerifiedDeliveryExternalWitness - Added aggregate external witness status and startup fail-closed clearing
//! v0.60.0-VerifiedDeliveryRollbackAnchor - Track signed cache generations and fail closed on local delivery-evidence rollback
//! v0.59.0-VerifiedDeliveryRestartContinuity - Persist only signed aggregate client-delivery evidence and require fresh receipt-capable peers after restart
//! v0.58.0-RouteNetworkAntiAffinity - Require routeable peers and distinct endpoint network identities for multi-hop paths
//! v0.57.0-SignedDeliveryReceipts - Track freshness-bounded receipt-capable routes and verified client onion delivery evidence
//! v0.56.0-ProofRestartContinuity - Expose authenticated restore and signed proof persistence evidence
//! v0.55.0-TwoHopProofCache - Added signed bounded warm-restart proof history
//! v0.54.0-RelayEvidenceTruthfulness - Stopped classifying unlabelled blind-relay acceptance as real user traffic
//! v0.53.0-RouteSurfaceEvidence - Bound route health to stable signed routing fields across descriptor refreshes
//! v0.52.0-RouteabilityCacheEvidence - Added bounded descriptor-bound warm-restart route evidence
//! v0.51.0-RouteGovernanceSummary - Added aggregate route-quality governance contract
//! v0.50.0-TwoHopProofStabilityWindow - Added privacy-safe stability window and failure circuit breaker fields
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

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeBootstrapSnapshot, NodeCapability, NodeDiscoveryMessage,
    NodeProtocolFeature, RouteDomainAttestationCertificateV1, SignedNodeDescriptor,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Notify;

use super::discovery_endpoint_promotion_material::VerifiedPromotionMaterial;

mod cache_persistence;
mod permissionless_promotion;
use permissionless_promotion::PermissionlessPromotionGate;
mod route_domain_certificates;
mod route_selection;
mod status;
use route_domain_certificates::PeerStoreRouteDomainAttestorPolicy;
pub use route_domain_certificates::{
    PeerStoreRouteDomainCertificateCacheReport, RouteDomainAttestorPolicyError,
    RouteDomainCertificateImportError,
};

const DISCOVERY_GOSSIP_STALE_AFTER_SECS: u64 = 900;
const DISCOVERY_GOSSIP_FAILURE_ATTENTION_THRESHOLD: u64 = 3;
const PEER_DESCRIPTOR_STALE_WINDOW_SECS: u64 = 300;
const PEER_ROUTE_LAST_SEEN_FRESH_SECS: u64 = 300;
const PEER_ROUTE_LAST_SEEN_ACCEPTABLE_SECS: u64 = 900;
const PEER_ROUTE_LAST_SEEN_STALE_SECS: u64 = 1_800;
const PEER_ROUTE_RECENT_FAILURE_SECS: u64 = 600;
const PEER_ROUTEABILITY_STALE_AFTER_SECS: u64 = 1_800;
/// Local peer-cache routeability evidence schema understood by this node.
///
/// Version 2 signs active route-quarantine evidence together with successful
/// routeability evidence. Readers continue accepting signed version 1 caches.
pub const ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION: u16 = 2;
/// Previous routeability-only cache schema accepted during rolling upgrades.
pub const ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION: u16 = 1;
/// Active route-quarantine section schema coupled to routeability cache v2.
pub const ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION: u16 = 1;
/// Local peer-cache two-hop proof history schema understood by this node.
pub const TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION: u16 = 1;
/// Local peer-cache three-hop proof history schema understood by this node.
pub const THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION: u16 = 1;
/// Local peer-cache aggregate verified-client delivery schema understood by this node.
pub const VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION: u16 = 2;
/// Local peer-cache route-domain certificate schema understood by this node.
pub const ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION: u16 = 1;
/// Maximum route-domain certificates accepted from one local recovery cache.
pub const ROUTE_DOMAIN_CERTIFICATE_CACHE_MAX_ENTRIES: usize = 4_096;
const ROUTEABILITY_EVIDENCE_KIND_EXACT_DESCRIPTOR: &str = "direct_opaque_route_success";
const ROUTEABILITY_EVIDENCE_KIND_ROUTE_SURFACE: &str = "direct_opaque_route_surface_success";
const PEER_ROUTEABILITY_CACHE_MAX_ENTRIES: usize = 4_096;
const PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD: u64 = 3;
const PEER_ROUTE_FAILURE_QUARANTINE_SECS: u64 = 300;
const PEER_ROUTE_RECOVERY_PROBE_AFTER_SECS: u64 = 60;
const PEER_ROUTE_STATUS_LIMIT: usize = 8;
const PEER_HEALTH_STATUS_LIMIT: usize = 64;
const PEER_QUORUM_MIN_VALID_PEERS: usize = 2;
const PEER_QUORUM_MIN_ROUTEABLE_CHAT_RELAYS: usize = 1;
const TWO_HOP_DELIVERY_RECEIPT_MIN_CAPABLE_PEERS: usize = 2;
/// Maximum terminal replicas considered by authenticated App chat delivery.
pub(crate) const AUTHENTICATED_CHAT_TERMINAL_FANOUT_LIMIT: usize = 3;
/// Maximum middle-hop candidates inspected for each authenticated terminal.
pub(crate) const AUTHENTICATED_CHAT_MIDDLE_CANDIDATE_LIMIT: usize = 8;
const TWO_HOP_PATH_POLICY_NETWORK_DIVERSE: &str = "distinct_node_and_network_prefix";
/// Stage-A hard ceiling for self-signed descriptors that have not completed a
/// separate endpoint-possession proof. This is intentionally independent of
/// the verified-live peer capacity.
const UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY: usize = 256;
/// A legacy snapshot may not monopolize the candidate lane in one request.
const UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE: usize = 16;
/// A self-signed descriptor cannot reserve candidate capacity for longer than
/// this receiver-local horizon.
const UNTRUSTED_DISCOVERY_MAX_LIFETIME_SECS: u64 = 7_200;
/// Defensive bound retained explicitly even though `verify_at(now)` rejects a
/// not-yet-valid descriptor today.
const UNTRUSTED_DISCOVERY_MAX_FUTURE_SKEW_SECS: u64 = 300;
/// Bounded exact-replay memory after an expired candidate releases its slot.
const UNTRUSTED_DISCOVERY_TOMBSTONE_TTL_SECS: u64 = 900;
const UNTRUSTED_DISCOVERY_TOMBSTONE_CAPACITY: usize = UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY;

/// Privacy-safe result of applying authenticated client relay path policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AuthenticatedDeliveryPathReadiness {
    pub ready: bool,
    pub reason: &'static str,
}

/// Internal selector for signed path-proof cache sections.
///
/// [THREE-HOP-SIGNED-RECOVERY 2026-08-02 by Codex] Keeping path shape,
/// validation, route-pool gates, and status mutation behind one selector
/// prevents the two-hop and three-hop recovery contracts from drifting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathProofCacheKind {
    TwoHop,
    ThreeHop,
}

impl PathProofCacheKind {
    const fn schema_version(self) -> u16 {
        match self {
            Self::TwoHop => TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
            Self::ThreeHop => THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION,
        }
    }

    const fn hop_count(self) -> u8 {
        match self {
            Self::TwoHop => 2,
            Self::ThreeHop => 3,
        }
    }

    const fn path_shape(self) -> &'static str {
        match self {
            Self::TwoHop => "entry_middle_terminal",
            Self::ThreeHop => "entry_middle_middle_terminal",
        }
    }

    const fn ttl_shape(self) -> &'static str {
        match self {
            Self::TwoHop => "entry_ttl_2_onward_ttl_1",
            Self::ThreeHop => "entry_ttl_3_onward_ttl_2",
        }
    }

    const fn control_evidence_mode(self) -> &'static str {
        match self {
            Self::TwoHop => "synthetic_two_hop_control_probe",
            Self::ThreeHop => "synthetic_three_hop_control_probe",
        }
    }

    const fn audit_prefix(self) -> &'static str {
        match self {
            Self::TwoHop => "two_hop_proof_cache",
            Self::ThreeHop => "three_hop_proof_cache",
        }
    }
}

/// Coarse endpoint identity used only to prevent obviously collocated hops.
///
/// IP endpoints are grouped by IPv4 /24 or IPv6 /48. DNS endpoints use their
/// normalized exact hostname because this process does not perform mutable DNS
/// resolution during route planning. This is deliberately not presented as
/// ASN or operator diversity proof.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EndpointNetworkIdentity {
    Ipv4([u8; 3]),
    Ipv6([u8; 6]),
    Dns(String),
}

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
/// Warm-restart proof history is intentionally smaller than runtime history.
/// Eight events are enough for the existing stability window while bounding
/// disk exposure and preventing a local cache from becoming a traffic ledger.
const TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES: usize = 8;
const TWO_HOP_PATH_PROOF_STABILITY_WINDOW_EVENTS: usize = 8;
const TWO_HOP_PATH_PROOF_STABILITY_MIN_ATTEMPTS: u64 = 3;
const TWO_HOP_PATH_PROOF_STABILITY_SUCCESS_PERCENT: u8 = 80;
const TWO_HOP_PATH_PROOF_FAILURE_CIRCUIT_BREAKER_THRESHOLD: u64 = 3;

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
    /// Control-only checks use `synthetic_two_hop_control_probe`; onion checks
    /// that reach terminal store-and-forward use
    /// `synthetic_onion_message_delivery_probe`. Neither is App/user traffic.
    pub evidence_mode: String,
    /// Stable proof scope: `control_plane` or `message_delivery`.
    ///
    /// `message_delivery` means a synthetic encrypted ChatEnvelope reached the
    /// terminal store-and-forward boundary. It must not be presented as a user
    /// message, active conversation, or production traffic volume.
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
/// blind-node invariant. A separately signed, freshness-bounded subset may be
/// restored after a warm restart; stale history is discarded.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreTwoHopPathProofHistory {
    /// Unix timestamp when this summary was generated.
    pub generated_at: u64,
    /// Stable readiness bucket: forming, ready, stale, attention, or idle.
    pub status: String,
    /// Stable freshness bucket for UI and runbooks: forming, fresh_success,
    /// stale_success, recent_failure, future_ignored, or no_success.
    ///
    /// This is derived only from bounded local proof outcomes and coarse age
    /// windows. It must never encode node IDs, route IDs, endpoints, payloads,
    /// receiver identities, or social graph information.
    pub freshness_bucket: String,
    /// Whether the latest retained proof is a fresh accepted two-hop path.
    pub proof_ready: bool,
    /// Whether the latest proof success is still within the routeability window.
    pub recent_success_ready: bool,
    /// Whether the latest proof is a fresh synthetic terminal delivery check.
    ///
    /// This is stricter than `proof_ready`: control-plane probes can prove
    /// entry -> middle -> terminal reachability, while this gate requires the
    /// terminal hop to accept a synthetic opaque payload into ChatRelay
    /// store-and-forward. It does not prove that a user message was delivered.
    #[serde(default)]
    pub message_delivery_ready: bool,
    /// Whether any retained terminal message-delivery proof is still fresh.
    #[serde(default)]
    pub recent_message_delivery_ready: bool,
    /// Evidence mode of the newest accepted terminal delivery proof.
    ///
    /// Today this is `synthetic_onion_message_delivery_probe`; `none` means no
    /// such proof is retained. Consumers must inspect this field before
    /// describing `message_delivery_ready` in product copy.
    #[serde(default)]
    pub message_delivery_evidence_mode: String,
    /// Whether the latest retained proof ended in one or more failures.
    pub failure_streak_active: bool,
    /// Maximum number of recent proof events retained by this process.
    pub window_size: usize,
    /// Number of retained proof events in this summary.
    pub retained_events: usize,
    /// Retained events ignored because their timestamps are ahead of `generated_at`.
    ///
    /// [PATH-PROOF-CLOCK-GUARD 2026-08-03 by Codex] This is an aggregate
    /// clock-health counter only. It never contains route or peer dimensions.
    #[serde(default)]
    pub future_events_ignored: u64,
    /// Retained proof attempts in the bounded window.
    pub attempted: u64,
    /// Retained accepted proofs in the bounded window.
    pub succeeded: u64,
    /// Retained accepted synthetic proofs whose scope is terminal delivery.
    #[serde(default)]
    pub message_delivery_successes: u64,
    /// Retained rejected proofs in the bounded window.
    pub failed: u64,
    /// Success percentage over retained events.
    pub success_percent: u8,
    /// Number of latest retained events considered for stability scoring.
    ///
    /// This window is intentionally aggregate-only. It must never expose
    /// individual route ids, endpoint URLs, node ids, encrypted blobs, receiver
    /// identities, client IPs, DNS contents, destinations, Memory Chain
    /// plaintext, wallet traffic, or social graph edges.
    #[serde(default)]
    pub stability_window_size: usize,
    /// Attempts in the stability scoring window.
    #[serde(default)]
    pub stability_window_attempted: u64,
    /// Accepted proofs in the stability scoring window.
    #[serde(default)]
    pub stability_window_succeeded: u64,
    /// Rejected proofs in the stability scoring window.
    #[serde(default)]
    pub stability_window_failed: u64,
    /// Success percentage over the stability scoring window.
    #[serde(default)]
    pub stability_success_percent: u8,
    /// Stable maturity bucket: forming, warming_up, stable, degraded, stale,
    /// failing, circuit_breaker, or clock_attention.
    #[serde(default)]
    pub stability_status: String,
    /// Whether the recent proof window is mature enough for product surfaces to
    /// describe the route foundation as repeatedly stable.
    #[serde(default)]
    pub stability_ready: bool,
    /// Consecutive rejected proof threshold for the local circuit-breaker bucket.
    #[serde(default)]
    pub failure_circuit_breaker_threshold: u64,
    /// Whether recent consecutive proof failures crossed the local
    /// circuit-breaker threshold.
    #[serde(default)]
    pub failure_circuit_breaker_active: bool,
    /// Coarse latest proof age bucket: none, fresh, acceptable, aging, or stale.
    #[serde(default)]
    pub latest_age_bucket: String,
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
    /// Synthetic route-only checks use `control_plane`; synthetic onion checks
    /// that enter terminal ChatRelay use `message_delivery`. This map does not
    /// classify App/user traffic.
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

/// Aggregate result of one outbound Directory proof-gossip round.
///
/// [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] This input contract is
/// deliberately dimensionless. Callers may report only process-wide counts;
/// peer ids, endpoints, producers, descriptors, blocks, proofs, routes,
/// messages, payloads, clients, and traffic metadata are forbidden.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeerStoreDirectoryProofGossipRound {
    /// Peer summaries checked for explicit proof-gossip support.
    pub capability_checked: usize,
    /// Peers that explicitly advertised proof-gossip support.
    pub capable: usize,
    /// Capable peers sent at least one proof frame.
    pub peers_attempted: usize,
    /// Total proof frames sent, including bounded fallback frames.
    pub frames_attempted: usize,
    /// Peers that accepted at least one audited proof.
    pub accepted: usize,
    /// Proof frames rejected because exact local replica evidence was absent.
    pub evidence_rejected: usize,
    /// Peers whose audited replica admission service was unavailable.
    pub replica_unavailable: usize,
    /// Peers that rate-limited the optional proof frame.
    pub rate_limited: usize,
    /// Peers that returned another non-success protocol status.
    pub protocol_rejected: usize,
    /// Peers whose optional proof request failed at the transport layer.
    pub transport_failed: usize,
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
    /// Routeability cache restore status: restored, partial, empty, or rejected.
    #[serde(default)]
    pub last_routeability_cache_status: Option<String>,
    /// Number of descriptor-bound routeability records restored at startup.
    #[serde(default)]
    pub last_routeability_cache_restored: u64,
    /// Number of routeability cache records rejected by freshness or binding checks.
    #[serde(default)]
    pub last_routeability_cache_rejected: u64,
    /// Timestamp of the last routeability cache restore attempt.
    #[serde(default)]
    pub last_routeability_cache_at: Option<u64>,
    /// Local recovery-anchor result for routeability plus active quarantine.
    ///
    /// Stable buckets match the other signed recovery sections. No digest,
    /// peer identity, endpoint, route, or failure detail is exported.
    #[serde(default)]
    pub last_routeability_cache_rollback_protection: Option<String>,
    /// Signed two-hop proof cache restore status: restored, partial, empty, or rejected.
    #[serde(default)]
    pub last_two_hop_proof_cache_status: Option<String>,
    /// Authentication result for the last parsed two-hop proof cache section.
    ///
    /// Stable buckets are `verified`, `legacy_descriptor_only`,
    /// `signature_invalid`, and `identity_unavailable`. Keeping this separate
    /// from the aggregate source detail lets admission fail closed without
    /// parsing log text.
    #[serde(default)]
    pub last_two_hop_proof_cache_authentication: Option<String>,
    /// Number of fresh synthetic proof events restored at startup.
    #[serde(default)]
    pub last_two_hop_proof_cache_restored: u64,
    /// Whether the restored events independently reconstructed a mature fresh
    /// stability window at restore time.
    #[serde(default)]
    pub last_two_hop_proof_cache_restored_stability_ready: bool,
    /// Number of proof events rejected by authentication, schema, or freshness checks.
    #[serde(default)]
    pub last_two_hop_proof_cache_rejected: u64,
    /// Timestamp of the last two-hop proof cache restore attempt.
    #[serde(default)]
    pub last_two_hop_proof_cache_at: Option<u64>,
    /// Number of fresh proof events included in the latest successfully
    /// persisted, independently signed cache section during this process.
    #[serde(default)]
    pub last_two_hop_proof_cache_persisted: u64,
    /// Whether that exact persisted proof section contained a mature fresh
    /// stability window.
    #[serde(default)]
    pub last_two_hop_proof_cache_persisted_stability_ready: bool,
    /// Timestamp of the latest successful signed proof-cache persistence.
    #[serde(default)]
    pub last_two_hop_proof_cache_persisted_at: Option<u64>,
    /// Local recovery-anchor result for the latest two-hop proof section.
    ///
    /// [PATH-PROOF-ROLLBACK-ANCHOR 2026-08-02 by Codex] Stable buckets mirror
    /// the aggregate delivery anchor without retaining proof digests or paths.
    #[serde(default)]
    pub last_two_hop_proof_cache_rollback_protection: Option<String>,
    /// Signed three-hop proof cache restore status: restored, partial, empty, or rejected.
    #[serde(default)]
    pub last_three_hop_proof_cache_status: Option<String>,
    /// Authentication result for the independently signed three-hop section.
    #[serde(default)]
    pub last_three_hop_proof_cache_authentication: Option<String>,
    /// Number of fresh three-hop synthetic proof events restored at startup.
    #[serde(default)]
    pub last_three_hop_proof_cache_restored: u64,
    /// Whether restored three-hop events reconstruct a mature fresh window.
    #[serde(default)]
    pub last_three_hop_proof_cache_restored_stability_ready: bool,
    /// Number of three-hop events rejected by authentication or validation.
    #[serde(default)]
    pub last_three_hop_proof_cache_rejected: u64,
    /// Timestamp of the latest three-hop proof-cache restore attempt.
    #[serde(default)]
    pub last_three_hop_proof_cache_at: Option<u64>,
    /// Number of fresh three-hop events in the latest durable signed snapshot.
    #[serde(default)]
    pub last_three_hop_proof_cache_persisted: u64,
    /// Whether the latest persisted three-hop section held a mature window.
    #[serde(default)]
    pub last_three_hop_proof_cache_persisted_stability_ready: bool,
    /// Timestamp of the latest successful signed three-hop cache persistence.
    #[serde(default)]
    pub last_three_hop_proof_cache_persisted_at: Option<u64>,
    /// Local recovery-anchor result for the latest three-hop proof section.
    #[serde(default)]
    pub last_three_hop_proof_cache_rollback_protection: Option<String>,
    /// Signed aggregate client-delivery cache restore status.
    ///
    /// Stable buckets are `restored`, `empty`, and `rejected`. The section
    /// never contains route, peer, sender, receiver, message, or payload data.
    #[serde(default)]
    pub last_client_delivery_cache_status: Option<String>,
    /// Authentication result for the last aggregate client-delivery section.
    #[serde(default)]
    pub last_client_delivery_cache_authentication: Option<String>,
    /// Aggregate delivery count restored from the signed local cache.
    #[serde(default)]
    pub last_client_delivery_cache_restored: u64,
    /// Timestamp of the last aggregate client-delivery restore attempt.
    #[serde(default)]
    pub last_client_delivery_cache_at: Option<u64>,
    /// Aggregate delivery count included in the latest durable cache write.
    #[serde(default)]
    pub last_client_delivery_cache_persisted: u64,
    /// Timestamp of the latest durable aggregate client-delivery cache write.
    #[serde(default)]
    pub last_client_delivery_cache_persisted_at: Option<u64>,
    /// Monotonic generation of the latest evaluated aggregate delivery section.
    #[serde(default)]
    pub last_client_delivery_cache_generation: u64,
    /// Local rollback-protection status for the latest aggregate delivery section.
    ///
    /// Stable buckets are `anchored`, `cache_ahead`, `legacy_unanchored`,
    /// `anchor_missing`, `anchor_invalid`, `anchor_conflict`,
    /// `rollback_detected`, and `not_checked`.
    #[serde(default)]
    pub last_client_delivery_cache_rollback_protection: Option<String>,
    /// External cache-anchor witness result for the latest evaluated generation.
    ///
    /// Stable buckets are `disabled`, `verified`, `partial`, `unavailable`,
    /// `rollback_detected`, `conflict`, and `gap`. All accompanying fields are
    /// aggregate-only and omit witness identities, endpoints, and digests.
    #[serde(default)]
    pub last_client_delivery_witness_status: Option<String>,
    /// Timestamp of the latest external witness evaluation.
    #[serde(default)]
    pub last_client_delivery_witness_checked_at: Option<u64>,
    /// Local signed cache generation evaluated by the latest witness round.
    #[serde(default)]
    pub last_client_delivery_witness_generation: u64,
    /// Whether availability below threshold rejects restored delivery evidence.
    #[serde(default)]
    pub last_client_delivery_witness_required: bool,
    /// Minimum valid signed responses configured for the latest round.
    #[serde(default)]
    pub last_client_delivery_witness_minimum_verified: u64,
    /// Distinct operator-pinned witnesses in the bounded round.
    #[serde(default)]
    pub last_client_delivery_witness_configured: u64,
    /// Witness HTTP requests attempted in the bounded round.
    #[serde(default)]
    pub last_client_delivery_witness_attempted: u64,
    /// Cryptographically valid signed witness responses.
    #[serde(default)]
    pub last_client_delivery_witness_verified: u64,
    /// Witnesses that advanced their durable high-water mark.
    #[serde(default)]
    pub last_client_delivery_witness_advanced: u64,
    /// Witnesses already holding the exact generation and opaque digest.
    #[serde(default)]
    pub last_client_delivery_witness_idempotent: u64,
    /// Witnesses proving the local generation is below their high-water mark.
    #[serde(default)]
    pub last_client_delivery_witness_stale: u64,
    /// Witnesses proving a different digest reused the same generation.
    #[serde(default)]
    pub last_client_delivery_witness_conflicts: u64,
    /// Witnesses refusing a discontinuous generation advance.
    #[serde(default)]
    pub last_client_delivery_witness_gaps: u64,
    /// Admission, endpoint, transport, decoding, or signature failures.
    #[serde(default)]
    pub last_client_delivery_witness_failed: u64,
    /// Last Directory proof-gossip convergence bucket.
    ///
    /// Stable values are `idle`, `legacy_only`, `converged`, `partial`,
    /// `evidence_diverged`, and `degraded`.
    #[serde(default)]
    pub last_directory_proof_gossip_status: Option<String>,
    /// Peer summaries checked for explicit proof-gossip support.
    #[serde(default)]
    pub last_directory_proof_gossip_capability_checked: u64,
    /// Peers that explicitly advertised proof-gossip support.
    #[serde(default)]
    pub last_directory_proof_gossip_capable: u64,
    /// Capable peers sent at least one proof frame.
    #[serde(default)]
    pub last_directory_proof_gossip_peers_attempted: u64,
    /// Total proof frames sent, including bounded fallback frames.
    #[serde(default)]
    pub last_directory_proof_gossip_frames_attempted: u64,
    /// Bounded fallback proof frames sent after an evidence miss.
    #[serde(default)]
    pub last_directory_proof_gossip_fallback_frames_attempted: u64,
    /// Peers that accepted at least one audited proof.
    #[serde(default)]
    pub last_directory_proof_gossip_accepted: u64,
    /// Percentage of capable peers accepting at least one proof in the round.
    #[serde(default)]
    pub last_directory_proof_gossip_acceptance_percent: u64,
    /// Proof frames rejected because exact local replica evidence was absent.
    #[serde(default)]
    pub last_directory_proof_gossip_evidence_rejected: u64,
    /// Peers whose audited replica admission service was unavailable.
    #[serde(default)]
    pub last_directory_proof_gossip_replica_unavailable: u64,
    /// Peers that rate-limited the optional proof frame.
    #[serde(default)]
    pub last_directory_proof_gossip_rate_limited: u64,
    /// Peers that returned another non-success protocol status.
    #[serde(default)]
    pub last_directory_proof_gossip_protocol_rejected: u64,
    /// Peers whose optional proof request failed at the transport layer.
    #[serde(default)]
    pub last_directory_proof_gossip_transport_failed: u64,
    /// Consecutive capable rounds with no accepted Directory proof.
    #[serde(default)]
    pub consecutive_directory_proof_gossip_zero_acceptance_rounds: u64,
    /// Timestamp of the latest round with at least one accepted proof.
    #[serde(default)]
    pub last_directory_proof_gossip_success_at: Option<u64>,
    /// Timestamp of the latest Directory proof-gossip observation.
    #[serde(default)]
    pub last_directory_proof_gossip_round_at: Option<u64>,
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
            last_routeability_cache_status: None,
            last_routeability_cache_restored: 0,
            last_routeability_cache_rejected: 0,
            last_routeability_cache_at: None,
            last_routeability_cache_rollback_protection: None,
            last_two_hop_proof_cache_status: None,
            last_two_hop_proof_cache_authentication: None,
            last_two_hop_proof_cache_restored: 0,
            last_two_hop_proof_cache_restored_stability_ready: false,
            last_two_hop_proof_cache_rejected: 0,
            last_two_hop_proof_cache_at: None,
            last_two_hop_proof_cache_persisted: 0,
            last_two_hop_proof_cache_persisted_stability_ready: false,
            last_two_hop_proof_cache_persisted_at: None,
            last_two_hop_proof_cache_rollback_protection: None,
            last_three_hop_proof_cache_status: None,
            last_three_hop_proof_cache_authentication: None,
            last_three_hop_proof_cache_restored: 0,
            last_three_hop_proof_cache_restored_stability_ready: false,
            last_three_hop_proof_cache_rejected: 0,
            last_three_hop_proof_cache_at: None,
            last_three_hop_proof_cache_persisted: 0,
            last_three_hop_proof_cache_persisted_stability_ready: false,
            last_three_hop_proof_cache_persisted_at: None,
            last_three_hop_proof_cache_rollback_protection: None,
            last_client_delivery_cache_status: None,
            last_client_delivery_cache_authentication: None,
            last_client_delivery_cache_restored: 0,
            last_client_delivery_cache_at: None,
            last_client_delivery_cache_persisted: 0,
            last_client_delivery_cache_persisted_at: None,
            last_client_delivery_cache_generation: 0,
            last_client_delivery_cache_rollback_protection: None,
            last_client_delivery_witness_status: None,
            last_client_delivery_witness_checked_at: None,
            last_client_delivery_witness_generation: 0,
            last_client_delivery_witness_required: false,
            last_client_delivery_witness_minimum_verified: 0,
            last_client_delivery_witness_configured: 0,
            last_client_delivery_witness_attempted: 0,
            last_client_delivery_witness_verified: 0,
            last_client_delivery_witness_advanced: 0,
            last_client_delivery_witness_idempotent: 0,
            last_client_delivery_witness_stale: 0,
            last_client_delivery_witness_conflicts: 0,
            last_client_delivery_witness_gaps: 0,
            last_client_delivery_witness_failed: 0,
            last_directory_proof_gossip_status: None,
            last_directory_proof_gossip_capability_checked: 0,
            last_directory_proof_gossip_capable: 0,
            last_directory_proof_gossip_peers_attempted: 0,
            last_directory_proof_gossip_frames_attempted: 0,
            last_directory_proof_gossip_fallback_frames_attempted: 0,
            last_directory_proof_gossip_accepted: 0,
            last_directory_proof_gossip_acceptance_percent: 0,
            last_directory_proof_gossip_evidence_rejected: 0,
            last_directory_proof_gossip_replica_unavailable: 0,
            last_directory_proof_gossip_rate_limited: 0,
            last_directory_proof_gossip_protocol_rejected: 0,
            last_directory_proof_gossip_transport_failed: 0,
            consecutive_directory_proof_gossip_zero_acceptance_rounds: 0,
            last_directory_proof_gossip_success_at: None,
            last_directory_proof_gossip_round_at: None,
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
    /// Total descriptors processed through live or bounded-candidate import paths.
    pub total_imported: u64,
    /// Total descriptors inserted or upgraded.
    pub inserted: u64,
    /// Total self-signed descriptors retained in the bounded, non-routeable
    /// candidate lane. This is aggregate-only and grants no peer authority.
    pub candidate_admitted: u64,
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
    /// Requests rejected by local aggregate or verified previous-hop admission.
    ///
    /// [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] This remains one
    /// process aggregate. It must not gain node, IP, route, endpoint, user,
    /// receiver, message, or ciphertext-derived dimensions.
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
    /// Authenticated App/client-originated onion deliveries whose exact opaque
    /// terminal payload was acknowledged by a valid terminal-node signature.
    ///
    /// This aggregate may be restored from the node's independently signed
    /// local cache. It must never be joined with route ids, peer ids, receiver
    /// identities, or message metadata.
    #[serde(default)]
    pub verified_client_onion_deliveries: u64,
    /// Unix timestamp of the last verified client-originated delivery receipt.
    #[serde(default)]
    pub last_verified_client_onion_delivery_at: Option<u64>,
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
    /// Whether a fresh terminal-signed receipt proves an authenticated
    /// App/client-originated opaque payload reached terminal store-and-forward.
    pub real_relay_ready: bool,
    /// Aggregate verified App/client-originated onion deliveries observed by
    /// this node, including fresh independently signed local-cache recovery.
    #[serde(default)]
    pub verified_client_onion_deliveries: u64,
    /// Seconds since the last verified client delivery receipt, when known.
    #[serde(default)]
    pub last_verified_client_onion_delivery_age_seconds: Option<u64>,
    /// Number of fresh peers proven to carry purpose-bound v2 delivery receipts.
    #[serde(default)]
    pub delivery_receipt_capable_peers: usize,
    /// Whether current receipt-capable route surfaces contain at least one
    /// policy-compliant, network-diverse middle/terminal pair.
    ///
    /// [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] This is
    /// intentionally stronger than `delivery_receipt_capable_peers >= 2`.
    /// It applies the same capability, routeability, attestation, exclusion,
    /// fanout, and network-diversity gates as authenticated client relay.
    #[serde(default)]
    pub authenticated_delivery_path_ready: bool,
    /// Stable aggregate reason for authenticated delivery path readiness.
    /// Never encode peer ids, endpoints, route ids, or message metadata here.
    #[serde(default)]
    pub authenticated_delivery_path_reason: String,
    /// Whether this process has fresh accepted terminal or forwarded relay work.
    ///
    /// This is intentionally origin-neutral: accepted work may be App traffic,
    /// node-generated synthetic delivery probes, or another opaque protocol
    /// payload. Consumers must not present it as verified user traffic.
    #[serde(default)]
    pub accepted_relay_ready: bool,
    /// Whether this process has fresh successful synthetic route probe evidence.
    pub synthetic_probe_ready: bool,
    /// Stable evidence bucket: idle, opaque_relay_acceptance,
    /// synthetic_onion_message_delivery_probe,
    /// synthetic_two_hop_control_probe, synthetic_probe, probe_failed, or
    /// opaque_relay_attempted.
    pub evidence_mode: String,
    /// Stable proof scope for UI copy: relay_acceptance, control_plane,
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
    /// This is separate from accepted opaque relay age so dashboards can show
    /// probe freshness without implying user traffic occurred.
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
    /// Bounded privacy-safe proof history for recent three-hop checks.
    ///
    /// [THREE-HOP-SIGNED-RECOVERY 2026-08-02 by Codex] This intentionally stays
    /// separate from two-hop admission. Fresh aggregate events may be restored
    /// from an independently signed local cache after the current route pool is
    /// revalidated. They are not consensus, App traffic, or a disclosed route.
    #[serde(default)]
    pub three_hop_path_proof_history: PeerStoreTwoHopPathProofHistory,
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
    /// Aggregate route governance summary for nodeboard/backend health cards.
    ///
    /// This compresses candidate lists, routeability evidence, and quarantine
    /// state into one public-safe contract. It intentionally does not expose
    /// endpoint URLs, full node IDs, route IDs, selected paths, encrypted
    /// payloads, receiver identities, client IPs, DNS, destinations, voucher
    /// secrets, private keys, wallet-level traffic, or social graph metadata.
    pub route_governance: PeerStoreRouteGovernanceStatus,
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
    /// Fingerprint of the signed route surface used by the latest success.
    ///
    /// This lets routine sequence/TTL refreshes retain health while endpoint,
    /// capability, public-discovery, or KEM changes invalidate it fail-closed.
    last_success_route_fingerprint_sha256: Option<String>,
    last_failure_at: Option<u64>,
    last_failure_reason: Option<String>,
    quarantine_count: u64,
    quarantine_until: Option<u64>,
    last_quarantine_at: Option<u64>,
    last_quarantine_reason: Option<String>,
}

/// A reason admitted into node reputation, aggregate counters, or audit state.
///
/// [PEER-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Public compatibility
/// recorders still accept strings, but open text ends here. Keeping this value
/// private prevents future mutation code from accidentally persisting endpoint
/// values, peer-controlled response text, route identifiers, or payload-derived
/// details. Unknown and malformed buckets collapse to one stable value.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PrivacySafePeerHealthReason(String);

impl PrivacySafePeerHealthReason {
    const UNKNOWN: &'static str = "unknown";

    fn route_failure(reason: &str) -> Self {
        Self::admit(reason, is_route_failure_reason)
    }

    fn blind_relay_rejection(reason: &str) -> Self {
        Self::admit(reason, is_blind_relay_rejection_reason)
    }

    fn peer_relay_rejection(reason: &str) -> Self {
        Self::admit(reason, is_peer_relay_rejection_reason)
    }

    fn quarantine(reason: &str) -> Self {
        Self::admit(reason, is_quarantine_reason)
    }

    fn admit(reason: &str, predicate: fn(&str) -> bool) -> Self {
        if predicate(reason) {
            Self(reason.to_string())
        } else {
            Self(Self::UNKNOWN.to_string())
        }
    }

    fn as_str(&self) -> &str {
        &self.0
    }

    fn into_inner(self) -> String {
        self.0
    }
}

fn is_http_status_reason(reason: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| {
        let Some(status) = reason.strip_prefix(prefix) else {
            return false;
        };
        status.len() == 3
            && status.bytes().all(|byte| byte.is_ascii_digit())
            && status
                .parse::<u16>()
                .is_ok_and(|status| (100..=599).contains(&status))
    })
}

fn is_bounded_response_reason(reason: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "ack_",
        "onion_ack_",
        "onion_delivery_ack_",
        "peer_relay_ack_",
    ];
    const SUFFIXES: &[&str] = &[
        "response_too_large",
        "response_body_read_failed",
        "response_json_decode_failed",
    ];

    PREFIXES.iter().any(|prefix| {
        reason
            .strip_prefix(prefix)
            .is_some_and(|suffix| SUFFIXES.contains(&suffix))
    })
}

fn is_reqwest_failure_reason(reason: &str) -> bool {
    const PHASES: &[&str] = &[
        "blind_relay_probe",
        "two_hop_onion_delivery_probe",
        "two_hop_blind_relay_probe",
        "three_hop_onion_delivery_probe",
        "blind_relay_request",
        "onion_delivery_request",
        "peer_relay_request",
    ];
    const SUFFIXES: &[&str] = &[
        "timeout",
        "connect",
        "http_status",
        "decode",
        "body",
        "request",
        "unknown",
    ];

    PHASES.iter().any(|phase| {
        let Some(suffix) = reason
            .strip_prefix(phase)
            .and_then(|suffix| suffix.strip_prefix('_'))
        else {
            return false;
        };
        SUFFIXES.contains(&suffix) || is_http_status_reason(suffix, &["http_"])
    })
}

fn is_route_failure_reason(reason: &str) -> bool {
    matches!(
        reason,
        "missing_endpoint"
            | "invalid_endpoint"
            | "ack_rejected"
            | "onion_ack_rejected"
            | "delivery_receipt_invalid"
            | "delivery_receipt_rejected"
            | "failure_receipt_downgrade"
            | "failure_receipt_invalid"
            | "forward_failed"
            | "request_failed"
            | "invalid_previous_hop"
            | "invalid_signature"
            | "envelope_too_large"
            | "ttl_exhausted"
            | "timestamp_expired"
            | "timestamp_in_future"
            | "rate_limited"
            | "quarantined"
            | "route_in_flight"
            | "replay_capacity"
            // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Closed buckets
            // expose restart/replay protection health without route dimensions.
            | "replay_conflict"
            | "replay_response_expired"
            | "replay_protection_unavailable"
            | "route_loop"
            | "no_route"
            | "onion_peel_failed"
            | "onion_terminal_payload_rejected"
            | "onion_terminal_capacity_exhausted"
            | "downstream_rejected"
            | "peer_relay_target_auth_encode_failed"
            | "peer_relay_auth_encode_failed"
            | "peer_relay_ack_rejected"
            | "peer_relay_receipt_request_missing"
            | "peer_relay_receipt_missing"
            | "peer_relay_receipt_version_invalid"
            | "peer_relay_receipt_binding_invalid"
            | "peer_relay_receipt_signature_invalid"
    ) || is_http_status_reason(
        reason,
        &[
            "http_",
            "onion_http_",
            "onion_delivery_http_",
            "peer_relay_http_",
        ],
    ) || is_bounded_response_reason(reason)
        || is_reqwest_failure_reason(reason)
}

fn is_blind_relay_rejection_reason(reason: &str) -> bool {
    matches!(
        reason,
        "backpressure"
            | "self_loop"
            | "duplicate_route"
            | "onion_inner_not_layer"
            | "onion_terminal_delivery_failed"
            | "relay_unavailable"
            | "envelope_serialization_failed"
            | "pending_capacity_exhausted"
            | "store_pending_failed"
    ) || is_route_failure_reason(reason)
}

fn is_peer_relay_rejection_reason(reason: &str) -> bool {
    matches!(reason, "duplicate_route") || is_blind_relay_rejection_reason(reason)
}

fn is_quarantine_reason(reason: &str) -> bool {
    matches!(
        reason,
        "rate_limit" | "failure_threshold" | "still_quarantined"
    )
}

/// Process-local proof that one signed route surface carried a valid,
/// purpose-bound version-2 terminal receipt.
///
/// The fingerprint contains no plaintext, route id, endpoint string, payload
/// commitment, sender, receiver, or social-graph edge. It prevents receipt
/// authority from crossing an endpoint, capability, policy, or KEM rotation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PurposeBoundDeliveryReceiptEvidence {
    observed_at: u64,
    route_surface_fingerprint_sha256: String,
}

/// Signed-route-surface-bound successful routeability evidence stored only in
/// the local peer cache for warm restart recovery.
///
/// This deliberately excludes endpoints, route ids, payloads, receiver ids,
/// client metadata, failure history, quarantine state, and social graph data.
/// New records survive sequence/TTL-only descriptor refreshes because they bind
/// to the signed fields that affect routing. Legacy exact-descriptor records
/// remain readable. Endpoint, capability, discovery visibility, or KEM changes
/// invalidate both runtime and cached success. Startup direct probes still run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteabilityCacheEvidence {
    /// Full node id encoded as lowercase hex for exact local descriptor lookup.
    pub node_id_hex: String,
    /// Descriptor sequence current when this evidence was exported.
    ///
    /// Route-surface evidence permits a newer sequence only when the signed
    /// route surface is unchanged. Legacy exact-descriptor evidence requires
    /// an exact sequence match.
    pub descriptor_sequence: u64,
    /// SHA-256 binding over either the stable signed route surface (current)
    /// or canonical descriptor bytes plus signature (legacy).
    pub descriptor_fingerprint_sha256: String,
    /// Unix timestamp of the last successful direct opaque probe or forward.
    pub last_success_at: u64,
    /// Stable evidence kind identifying route-surface or legacy exact binding.
    pub evidence_kind: String,
}

/// Aggregate result of restoring descriptor-bound routeability evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteabilityCacheRestoreReport {
    /// Number of cache evidence records supplied, including bounded overflow.
    pub total: usize,
    /// Number of fresh records restored into local route-health state.
    pub restored: usize,
    /// Number rejected by schema, descriptor binding, validity, or freshness checks.
    pub rejected: usize,
}

/// Minimal active route quarantine retained in the signed host-local cache.
///
/// [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] The record intentionally
/// excludes failure reasons, counters, endpoint values, routes, payloads, and
/// user metadata. Its only authority is to preserve an already active local
/// isolation window across a short process restart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteQuarantineCacheEvidence {
    /// Full node id encoded as lowercase hex for local descriptor binding.
    pub node_id_hex: String,
    /// Descriptor sequence current when the quarantine evidence was exported.
    pub descriptor_sequence: u64,
    /// SHA-256 of the stable signed route surface currently under isolation.
    pub route_surface_fingerprint_sha256: String,
    /// Unix timestamp when this local quarantine window began.
    pub quarantined_at: u64,
    /// Unix timestamp when this local quarantine window expires.
    pub quarantine_until: u64,
}

/// Aggregate result of restoring active route-quarantine cache evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteQuarantineCacheRestoreReport {
    /// Number of records supplied, including bounded overflow.
    pub total: usize,
    /// Number of fresh records restored into local route health.
    pub restored: usize,
    /// Number rejected by bounds, time, descriptor, or route-surface checks.
    pub rejected: usize,
}

/// Aggregate result of restoring signed, privacy-safe two-hop proof history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreTwoHopProofCacheRestoreReport {
    /// Number of cache events supplied, including bounded overflow.
    pub total: usize,
    /// Number of fresh, schema-valid events restored into local probe history.
    pub restored: usize,
    /// Number rejected by bounds, freshness, route-pool, or field checks.
    pub rejected: usize,
}

/// Privacy-safe aggregate client delivery evidence stored in the signed local
/// peer cache for restart continuity.
///
/// A node creates this record only after validating a terminal signature bound
/// to the exact route and opaque payload. Those correlatable proof inputs are
/// deliberately discarded. The persisted form contains only a cumulative
/// count and the most recent verification time, so an operator cannot recover
/// a route, peer pair, sender, receiver, message id, or payload from the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreVerifiedClientDeliveryCacheEvidence {
    /// Aggregate verified deliveries observed by this node.
    pub verified_deliveries: u64,
    /// Unix timestamp of the most recent verified terminal receipt.
    pub last_verified_at: u64,
}

/// Aggregate result of restoring signed client-delivery evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreVerifiedClientDeliveryCacheRestoreReport {
    /// Whether the cache contained an evidence record.
    pub present: bool,
    /// Whether the record passed schema, signature, freshness, and route-pool checks.
    pub restored: bool,
    /// Aggregate count restored when `restored` is true.
    pub restored_deliveries: u64,
}

/// Aggregate-only input for one external delivery-cache witness status update.
///
/// The API layer verifies every response before constructing this value. It
/// deliberately has no fields for node identities, endpoints, anchor digests,
/// delivery timestamps, routes, message ids, payloads, or client metadata.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PeerStoreVerifiedDeliveryWitnessRound {
    /// Distinct configured witnesses after hard bounding.
    pub configured: u64,
    /// Witness HTTP requests attempted.
    pub attempted: u64,
    /// Cryptographically valid signed responses.
    pub verified: u64,
    /// Durable high-water advances.
    pub advanced: u64,
    /// Exact already-durable observations.
    pub idempotent: u64,
    /// Responses proving local rollback.
    pub stale: u64,
    /// Responses proving same-generation digest conflict.
    pub conflicts: u64,
    /// Responses refusing a discontinuous advance.
    pub gaps: u64,
    /// Admission, transport, or verification failures.
    pub failed: u64,
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
// Route governance summary
// ============================================

/// Aggregate route governance summary for nodeboard, backend, and AI runbooks.
///
/// This is a compact product contract built from route candidates and opaque
/// node-to-node route health evidence. It deliberately avoids exposing full
/// node IDs, endpoint URLs, route IDs, selected paths, encrypted payloads,
/// receiver identities, client IPs, DNS, destinations, voucher secrets,
/// private keys, wallet-level traffic, plaintext, or social graph metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerStoreRouteGovernanceStatus {
    /// Unix timestamp when this summary was generated.
    pub generated_at: u64,
    /// Stable JSON contract version for dashboards and backend ingestion.
    pub contract_version: String,
    /// Stable source label for downstream schema routing.
    pub source: String,
    /// Stable status bucket: forming, healthy, or attention.
    pub status: String,
    /// Whether this node has a complete candidate pool for a two-hop privacy path.
    pub route_pool_ready: bool,
    /// Whether the route pool is ready and has no local quality attention buckets.
    pub quality_ready: bool,
    /// Total candidate rows considered across privacy, chat, and onion roles.
    pub candidates_total: usize,
    /// Candidates with fresh routeability evidence and no active route quarantine.
    pub routeable_total: usize,
    /// Routeable candidates that can receive encrypted chat relay traffic.
    pub routeable_chat_relays: usize,
    /// Routeable candidates that can act as onion middle hops.
    pub routeable_onion_middle_hops: usize,
    /// Routeable candidates that can relay privacy protocol packets.
    pub routeable_privacy_relays: usize,
    /// Candidates currently suppressed by local route-health quarantine.
    pub quarantined_total: usize,
    /// Candidates with repeated opaque route failures but no active quarantine.
    pub failing_total: usize,
    /// Candidates with some opaque route failures but not enough to be failing.
    pub degraded_total: usize,
    /// Candidates that advertise an endpoint but have no fresh routeability evidence yet.
    pub unknown_routeability_total: usize,
    /// Candidates whose last successful routeability evidence is stale.
    pub stale_routeability_total: usize,
    /// Candidates whose latest routeability state is unreachable.
    pub unreachable_total: usize,
    /// Best candidate score observed in the bounded candidate summary.
    pub best_score: Option<i64>,
    /// Worst candidate score observed in the bounded candidate summary.
    pub worst_score: Option<i64>,
    /// Average candidate score rounded toward zero.
    pub average_score: Option<i64>,
    /// Whether the planned one-hop chat relay path is complete.
    pub chat_single_hop_ready: bool,
    /// Whether the planned two-hop onion-shaped path is complete.
    pub chat_two_hop_onion_ready: bool,
    /// Consecutive opaque route failures needed before local quarantine starts.
    pub quarantine_threshold: u64,
    /// Local route quarantine window in seconds.
    pub quarantine_seconds: u64,
    /// Fresh routeability evidence window in seconds.
    pub routeability_stale_after_seconds: u64,
    /// Human-readable aggregate detail for logs and runbooks.
    pub detail: String,
    /// Operator or automation next action.
    pub next_action: String,
    /// Explicit privacy boundary for downstream consumers.
    pub privacy_boundary: String,
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
    candidate_admitted: AtomicU64,
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
    verified_client_onion_deliveries: AtomicU64,
    last_verified_client_onion_delivery_at: AtomicU64,
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
            candidate_admitted: AtomicU64::new(0),
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
            verified_client_onion_deliveries: AtomicU64::new(0),
            last_verified_client_onion_delivery_at: AtomicU64::new(0),
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
            candidate_admitted: self.candidate_admitted.load(Ordering::Relaxed),
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
                verified_client_onion_deliveries: self
                    .verified_client_onion_deliveries
                    .load(Ordering::Relaxed),
                last_verified_client_onion_delivery_at: Self::optional_ts(
                    self.last_verified_client_onion_delivery_at
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
    /// Number of self-signed descriptors retained only as non-routeable
    /// candidates. They never grant live peer or routing authority.
    #[serde(default)]
    pub candidates: usize,
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
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        }
    }

    /// Returns true when local import state changed.
    #[must_use]
    pub const fn changed(&self) -> bool {
        self.inserted > 0 || self.candidates > 0
    }
}

/// Bounded state for an unauthenticated descriptor that passed only its own
/// signature and local lifetime checks.
#[derive(Debug, Clone)]
struct UntrustedDiscoveryCandidate {
    descriptor: SignedNodeDescriptor,
    commitment: [u8; 32],
}

/// Exact, short-lived replay memory for an expired candidate. The map is
/// bounded and stores no endpoint, payload, or transport identity.
#[derive(Debug, Clone, Copy)]
struct UntrustedDiscoveryTombstone {
    sequence: u64,
    commitment: [u8; 32],
    expires_at: u64,
}

#[derive(Default)]
struct UntrustedDiscoveryCandidateState {
    candidates: HashMap<[u8; 32], UntrustedDiscoveryCandidate>,
    tombstones: HashMap<[u8; 32], UntrustedDiscoveryTombstone>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateAdmissionOutcome {
    Candidate,
    Unchanged,
    Stale,
    Conflict,
    Saturated,
    Rejected,
}

/// Coarse result of permissionless open-node admission.
///
/// The result intentionally contains no node identity, endpoint, descriptor,
/// signature, commitment, or transport metadata. Admission grants only a
/// bounded Stage-A candidate slot; endpoint possession and route authority
/// remain separate reviewed transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PermissionlessNodeAdmissionOutcome {
    /// A new or higher-sequence descriptor entered the candidate lane.
    Admitted,
    /// The exact same signed descriptor was already retained.
    ExactReplay,
    /// The descriptor sequence rolled back relative to retained state.
    Stale,
    /// The same identity and sequence were presented with different content.
    Conflict,
    /// The independent candidate lane is at its hard capacity.
    Saturated,
    /// Canonical, cryptographic, temporal, or endpoint checks failed.
    Rejected,
}

// ============================================
// PeerStore
// ============================================

/// In-memory verified descriptor store for known AeroNyx nodes.
pub struct PeerStore {
    peers: RwLock<HashMap<[u8; 32], SignedNodeDescriptor>>,
    // [PERMISSIONLESS-DISCOVERY-CANDIDATES 2026-09-14 by Codex] A descriptor
    // learned through legacy, unauthenticated gossip is evidence of only its
    // own signature. Keep it outside `peers` until a later endpoint-possession
    // protocol can promote it; candidate exhaustion must never consume live
    // routing capacity.
    untrusted_discovery_candidates: RwLock<UntrustedDiscoveryCandidateState>,
    untrusted_discovery_candidate_mode: AtomicBool,
    permissionless_promotions: RwLock<HashMap<[u8; 32], PermissionlessPromotionGate>>,
    permissionless_promotion_generation: AtomicU64,
    permissionless_candidate_round: AtomicU64,
    verified_delivery_witness_requesters: RwLock<HashSet<[u8; 32]>>,
    custody_audit_witness_requesters: RwLock<HashSet<[u8; 32]>>,
    peer_runtime: RwLock<HashMap<[u8; 32], PeerRuntimeMetadata>>,
    route_health: RwLock<HashMap<[u8; 32], PeerRouteHealth>>,
    relay_protection_health: RwLock<HashMap<[u8; 32], PeerRelayProtectionHealth>>,
    purpose_bound_delivery_receipt_capability:
        RwLock<HashMap<[u8; 32], PurposeBoundDeliveryReceiptEvidence>>,
    route_domain_attestor_policy: RwLock<PeerStoreRouteDomainAttestorPolicy>,
    route_domain_certificates: RwLock<HashMap<[u8; 32], RouteDomainAttestationCertificateV1>>,
    #[cfg(test)]
    route_domain_import_test_gate:
        Option<std::sync::Arc<route_domain_certificates::RouteDomainImportTestGate>>,
    max_peers: RwLock<Option<usize>>,
    counters: PeerStoreCounters,
    audit_events: RwLock<VecDeque<PeerStoreAuditEvent>>,
    peer_events: RwLock<VecDeque<PeerStorePeerEvent>>,
    two_hop_path_proof_events: RwLock<VecDeque<PeerStoreTwoHopPathProofEvent>>,
    three_hop_path_proof_events: RwLock<VecDeque<PeerStoreTwoHopPathProofEvent>>,
    bootstrap_status: RwLock<PeerStoreBootstrapStatus>,
    peer_cache_dirty: AtomicBool,
    peer_cache_notify: Notify,
    /// [NODE-TLS-BINDING 2026-10-10 by Claude] Whether this store publishes
    /// the process-wide identity-bound TLS directory (production store only).
    identity_tls_directory_enabled: AtomicBool,
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod admission;
mod blind_relay_stats;
mod bootstrap_snapshot;
mod gossip_status;
mod path_proof_log;
mod route_evidence;
mod verified_upsert;

/// Refreshes the identity-bound TLS directory when a peer-map writer returns.
///
/// [NODE-TLS-BINDING 2026-10-10 by Claude] Declared first in each writer, so
/// it drops last: after the write lock is released, on every return path.
struct IdentityTlsDirectoryRefresh<'a>(&'a PeerStore);

impl Drop for IdentityTlsDirectoryRefresh<'_> {
    fn drop(&mut self) {
        self.0.refresh_identity_tls_directory();
    }
}

impl PeerStore {
    /// Creates an empty peer store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            untrusted_discovery_candidates: RwLock::new(UntrustedDiscoveryCandidateState::default()),
            untrusted_discovery_candidate_mode: AtomicBool::new(false),
            permissionless_promotions: RwLock::new(HashMap::new()),
            permissionless_promotion_generation: AtomicU64::new(0),
            permissionless_candidate_round: AtomicU64::new(0),
            verified_delivery_witness_requesters: RwLock::new(HashSet::new()),
            custody_audit_witness_requesters: RwLock::new(HashSet::new()),
            peer_runtime: RwLock::new(HashMap::new()),
            route_health: RwLock::new(HashMap::new()),
            relay_protection_health: RwLock::new(HashMap::new()),
            purpose_bound_delivery_receipt_capability: RwLock::new(HashMap::new()),
            route_domain_attestor_policy: RwLock::new(PeerStoreRouteDomainAttestorPolicy::default()),
            route_domain_certificates: RwLock::new(HashMap::new()),
            #[cfg(test)]
            route_domain_import_test_gate: None,
            max_peers: RwLock::new(None),
            counters: PeerStoreCounters::new(),
            audit_events: RwLock::new(VecDeque::with_capacity(MAX_AUDIT_EVENTS)),
            peer_events: RwLock::new(VecDeque::with_capacity(MAX_PEER_EVENTS)),
            two_hop_path_proof_events: RwLock::new(VecDeque::with_capacity(
                MAX_TWO_HOP_PATH_PROOF_EVENTS,
            )),
            three_hop_path_proof_events: RwLock::new(VecDeque::with_capacity(
                MAX_TWO_HOP_PATH_PROOF_EVENTS,
            )),
            bootstrap_status: RwLock::new(PeerStoreBootstrapStatus::default()),
            peer_cache_dirty: AtomicBool::new(false),
            peer_cache_notify: Notify::new(),
            identity_tls_directory_enabled: AtomicBool::new(false),
        }
    }

    /// Makes this store the source of the process-wide identity-bound TLS
    /// directory used by every outbound peer URL.
    ///
    /// [NODE-TLS-BINDING 2026-10-10 by Claude] Called once by the server for
    /// its discovery store. Other stores (tests, operator tools) never publish,
    /// so they cannot change how this process reaches its peers.
    pub fn enable_identity_tls_directory(&self) {
        self.identity_tls_directory_enabled
            .store(true, Ordering::SeqCst);
        self.refresh_identity_tls_directory();
    }

    /// Rebuilds the directory from the current peer map, if enabled.
    pub(crate) fn refresh_identity_tls_directory(&self) {
        if !self.identity_tls_directory_enabled.load(Ordering::SeqCst) {
            return;
        }
        let directory = {
            let peers = self.peers.read();
            crate::api::peer_tls::PeerTlsDirectory::from_descriptors(peers.values())
        };
        crate::api::peer_tls::install_directory(directory);
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

    const fn purpose_bound_delivery_receipt_evidence_is_fresh(at: u64, now: u64) -> bool {
        at <= now && now.saturating_sub(at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS
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
    mod blind_relay;
    mod gossip_bootstrap;
    mod path_proof;
    mod remaining;
    mod routeability;
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        NodeCapacity, NodeDescriptor, NodePolicy, SignedNodeDescriptor,
    };
    use std::sync::{Arc, Barrier};

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

    fn permissionless_descriptor_for(
        kp: &IdentityKeyPair,
        sequence: u64,
        now: u64,
        endpoint: &str,
    ) -> SignedNodeDescriptor {
        let mut descriptor = NodeDescriptor::new(
            kp.public_key_bytes(),
            sequence,
            now.saturating_sub(1),
            now + 600,
            "1.0.0+anpf1-brsr1",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, kp).unwrap()
    }
}
