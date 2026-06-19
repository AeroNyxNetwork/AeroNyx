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

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use aeronyx_core::protocol::discovery::{
    NodeBootstrapSnapshot, NodeCapability, NodeDiscoveryMessage, SignedNodeDescriptor,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

const DISCOVERY_GOSSIP_STALE_AFTER_SECS: u64 = 900;
const DISCOVERY_GOSSIP_FAILURE_ATTENTION_THRESHOLD: u64 = 3;

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
    /// Unix timestamp of the last descriptor import attempt.
    pub last_import_at: Option<u64>,
    /// Unix timestamp of the last gossip exchange observed by this node.
    pub last_gossip_at: Option<u64>,
    /// Unix timestamp of the last exported snapshot.
    pub last_snapshot_at: Option<u64>,
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
    last_import_at: AtomicU64,
    last_gossip_at: AtomicU64,
    last_snapshot_at: AtomicU64,
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
            last_import_at: AtomicU64::new(0),
            last_gossip_at: AtomicU64::new(0),
            last_snapshot_at: AtomicU64::new(0),
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
            last_import_at: Self::optional_ts(self.last_import_at.load(Ordering::Relaxed)),
            last_gossip_at: Self::optional_ts(self.last_gossip_at.load(Ordering::Relaxed)),
            last_snapshot_at: Self::optional_ts(self.last_snapshot_at.load(Ordering::Relaxed)),
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
        descriptor
            .verify_at(now)
            .map_err(|_| PeerStoreError::VerificationFailed)?;

        let node_id = descriptor.node_id();
        let incoming_sequence = descriptor.sequence();
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

        peers.insert(node_id, descriptor);
        Ok(true)
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
        let mut report = PeerStoreImportReport {
            total: snapshot.peers.len(),
            inserted: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        };

        for descriptor in &snapshot.peers {
            match self.upsert_verified(descriptor.clone(), now) {
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
                self.load_bootstrap_snapshot(snapshot, now)
            }
            NodeDiscoveryMessage::DescriptorAnnounce { descriptor } => {
                let mut report = PeerStoreImportReport {
                    total: 1,
                    inserted: 0,
                    unchanged: 0,
                    stale: 0,
                    rejected: 0,
                };
                match self.upsert_verified(descriptor.clone(), now) {
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

    /// Removes descriptors that are no longer valid at `now`.
    pub fn cleanup_expired(&self, now: u64) -> usize {
        let mut peers = self.peers.write();
        let before = peers.len();
        peers.retain(|_node_id, descriptor| descriptor.verify_at(now).is_ok());
        before - peers.len()
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

        PeerStoreStatus {
            snapshot,
            runtime: self.counters.snapshot(),
            max_peers: self.max_peers(),
            recent_audit_events: self.recent_audit_events(),
            bootstrap,
            stability,
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
    fn test_cleanup_expired_removes_old_peers() {
        let store = PeerStore::new();
        let descriptor = signed_descriptor(1, 1_700_001_000);

        store.upsert_verified(descriptor, 1_700_000_100).unwrap();
        assert_eq!(store.cleanup_expired(1_700_002_000), 1);
        assert!(store.is_empty());
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
        assert!(status
            .recent_audit_events
            .iter()
            .any(|event| event.action == "outbound_gossip_round"));
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
