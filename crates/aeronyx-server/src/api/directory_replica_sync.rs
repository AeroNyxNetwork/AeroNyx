// ============================================
// File: crates/aeronyx-server/src/api/directory_replica_sync.rs
// ============================================
//! # Directory Replica Synchronization Coordinator
//!
//! ## Creation Reason
//! Directory replica scheduling originally lived inside `server.rs`, mixing
//! server lifecycle wiring with outbound transport, catch-up policy, telemetry,
//! and per-producer failure isolation. That made the startup path difficult to
//! audit and caused one slow pinned producer to delay every producer after it.
//!
//! ## Main Functionality
//! - Owns the hardened outbound HTTP client used for Directory Sync V1 pulls.
//! - Starts the first synchronization round after a short deterministic jitter.
//! - Synchronizes independent producers concurrently with a strict fan-out cap.
//! - Applies a producer-local round deadline and exponential failure backoff.
//! - Restores audited retry boundaries before the first post-restart request.
//! - Persists failure/skip scheduling without blocking the async runtime.
//! - Preserves producer-local page and request budgets on every round.
//! - Records only bounded, privacy-safe synchronization observations.
//! - Persists one signed observation checkpoint only after every pinned
//!   producer reaches its authenticated remote tip in the same round.
//! - Requests independently recomputed signed witness receipts for the newest
//!   mature, forward-moving local checkpoint and persists accepted receipts
//!   idempotently.
//! - Classifies every witness attempt into a closed privacy-safe outcome enum
//!   and persists aggregate diagnostics without peer-identifying metadata.
//! - Learns endpoint-level witness unavailability against the authenticated
//!   descriptor sequence so rolling upgrades do not inflate transport faults.
//! - Witnesses only checkpoints older than one complete synchronization
//!   interval, preventing asymmetric schedulers from chasing a moving head.
//! - Continues witnessing a mature checkpoint until the configured number of
//!   current pinned peers have independently recomputed it, while skipping
//!   pins whose canonical receipts are already durable.
//! - Anchors the current opaque witness-policy head with current pinned peers,
//!   skipping peers whose canonical policy receipt is already durable.
//! - Tries the producer directly first, then uses another pinned node as an
//!   audited evidence carrier only for bounded availability/admission failures.
//! - Requests up to eight contiguous blocks per page while the peer-side
//!   commitment cap preserves the original hydration/body budget.
//! - Cancels an in-flight round when server shutdown is requested.
//! - Optionally mirrors bounded multi-page prefixes from a rotating, bounded
//!   set of valid public discovery peers, using direct-first bounded carrier
//!   recovery without adding any mirror or carrier to authority checkpoints.
//!
//! ## Calling Relationships
//! - `server.rs` constructs this coordinator after the replica store is audited.
//! - `directory_chain_peer.rs` independently serves authenticated inbound pulls.
//! - `directory_replica_status.rs` exposes bounded scheduler observations.
//! - `services/directory_replica.rs` owns durable data and runtime observations.
//!
//! ## Main Logical Flow
//! 1. Validate constructor inputs and build a redirect-free bounded HTTP client.
//! 2. Derive a stable 5-15 second startup delay from the local public identity.
//! 3. On each tick, run at most four producer synchronization futures at once.
//! 4. Skip producer-local retries whose bounded backoff window is still active.
//! 5. Pull directly; before any trusted range exists, availability failures may
//!    fall back to a pinned carrier while cryptographic failures stop closed.
//! 6. Pull pages until the request budget or 45-second deadline is exhausted.
//! 7. Persist failures, and let a successful import atomically clear backoff.
//! 8. If every producer reaches its signed tip, append an idempotent local
//!    observation checkpoint from a blocking worker.
//! 9. After one complete synchronization interval, ask not-yet-recorded pinned
//!    peers to independently recompute the next forward mature checkpoint
//!    below its configured corroboration target; persist only canonical accepted
//!    receipts, never trust an unavailable or conflicting result.
//! 10. Treat an explicitly unsupported witness endpoint as peer unavailability
//!     and retry only after that peer publishes a newer signed descriptor.
//! 11. Persist bounded aggregate witness outcomes and mirror the current
//!     process round into runtime telemetry without retaining witness identity.
//! 12. Stop the complete round immediately when shutdown wins the select.
//! 13. Ask missing current pins to retain the opaque current policy head and
//!     persist only exact accepted signed receipts.
//! 14. Select verified public mirror candidates, exclude self and authority
//!     pins, and catch each selection up within a strict page, request, and
//!     wall-clock budget. Try the producer directly before at most two public
//!     carriers. Every imported block remains signed by the original producer;
//!     a carrier signs only the response envelope and never gains authority.
//!
//! ## Privacy Invariant
//! The coordinator never logs or retains endpoints, full producer identities,
//! response bodies, descriptor hashes, routes, selected hops, client metadata,
//! packet/chat payloads, Memory Chain records, DNS contents, destinations,
//! private keys, wallet traffic, or social graph metadata.
//!
//! ## Important Note for Next Developer
//! - Do not remove the producer-local request budget when increasing concurrency.
//! - Keep the fan-out cap small; pinned producers are independent trust domains.
//! - The deterministic startup delay is part of restart-storm protection.
//! - Stable failure reason buckets may be exposed by the status API. Never place
//!   peer-controlled strings, endpoints, or response bodies in those reasons.
//! - Witness receipts are external recomputation evidence, not votes, quorum,
//!   fork choice, consensus, or finality.
//! - Never use carrier fallback after a noncanonical, wrong-producer, invalid
//!   signature, or descriptor-hash response; these are security failures.
//! - Never feed permissionless mirror membership into checkpoints, witnesses,
//!   policy anchors, fork choice, consensus, voting, or finality.
//! - Mirror carrier recovery is one level only. Never recursively fetch from a
//!   carrier while serving a recovery request.
//!
//! ## Last Modified
//! `v0.16.0-MirrorBoundedCatchUp` - Added truthful converged/catching-up
//! outcomes and bounded multi-page mirror synchronization.
//! `v0.15.2-MirrorRecoveryDeadline` - Allowed audited public carriers to
//! complete within the bounded producer round.
//! `v0.15.1-MirrorRecoveryDiagnostics` - Added privacy-safe carrier failure diagnostics.
//! `v0.15.0-MirrorRecovery` - Added direct-first bounded public carrier recovery.
//! `v0.14.0-FullNodeMirror` - Added bounded rotating non-authoritative mirror pulls.
//! `v0.13.0-DirectoryPolicyHeadAnchor` - Added bounded external policy-head anchor rounds.
//! `v0.12.0-DirectoryBoundedColdCatchUp` - Raised the sparse-page cold catch-up cap while preserving the per-peer request budget.
//! `v0.11.0-DirectoryWitnessThreshold` - Added retryable pinned-witness corroboration targets.
//! `v0.10.0-DirectoryMatureWitnessScheduling` - Added one-interval mature unwitnessed checkpoint targeting.
//! `v0.9.0-DirectoryWitnessCapabilityNegotiation` - Added descriptor-sequence-scoped witness capability probing.
//! `v0.8.0-DirectoryWitnessOutcomeTelemetry` - Added typed witness outcomes and audited aggregate diagnostics.
//! `v0.7.2-DirectoryRoundBudgetAlignment` - Aligned outbound catch-up work with the existing inbound identity limit.
//! `v0.7.1-DirectoryBoundedMultiBlockCatchUp` - Raised bounded page width without raising commitment/request ceilings.
//! `v0.7.0-DirectoryEvidenceCarrier` - Added direct-first audited carrier fallback and dual-layer verification.
//! `v0.6.0-DirectoryObservationWitness` - Added bounded external recomputation rounds and durable receipts.
//! `v0.5.0-DirectoryObservationCheckpoints` - Added all-producer round gating
//! and signed, idempotent checkpoint persistence after authenticated catch-up.
//! `v0.4.0-DirectoryReplicaDurableBackoff` - Restored audited `SQLite` retry state
//! at startup and persisted failure/skip updates through blocking workers.
//! v0.3.0-DirectoryReplicaBackoff - Added producer-local round deadlines,
//! exponential retry backoff, and bounded retry scheduling telemetry.
//! v0.2.0-DirectoryReplicaClient - Owns outbound Directory Sync request,
//! verification, hydration, and import in addition to scheduling.
//! v0.1.0-DirectoryReplicaCoordinator - Extracted bounded concurrent scheduling
//! from `server.rs` and added deterministic startup synchronization jitter.
// ============================================

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_request_signing_bytes,
    directory_block_range_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_descriptor_objects_response_signing_bytes,
    directory_observation_witness_request_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_policy_anchor_request_signing_bytes, directory_policy_anchor_response_signing_bytes,
    directory_replica_block_range_request_signing_bytes,
    directory_replica_block_range_response_signing_bytes,
    directory_replica_descriptor_objects_request_signing_bytes,
    directory_replica_descriptor_objects_response_signing_bytes, encode_directory_sync_message,
    DirectoryCommitmentBlockV1, DirectoryObservationCheckpointV1, DirectorySyncMessage,
    SignedNodeDescriptor, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1, DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1, DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
    DIRECTORY_POLICY_ANCHOR_CONFLICT_V1, DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
    DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_SYNC_BLOCKS_V1, MAX_DIRECTORY_SYNC_OBJECTS_V1,
};
use futures::{stream, StreamExt};
use parking_lot::Mutex;
use rand::RngCore;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::api::memchain_peer::{commitment_peer_endpoint_is_public, commitment_peer_url};
use crate::api::{read_bounded_http_response, BoundedHttpResponseError};
use crate::services::directory_replica::{
    DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS, DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES,
};
use crate::services::{
    DirectoryObservationWitnessOutcome, DirectoryReplicaImportReport, DirectoryReplicaStore,
    DirectoryReplicaStoreError, DirectoryReplicaSyncRuntime, PeerStore,
};

/// Maximum pinned producers synchronized concurrently by one node.
pub(crate) const DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS: usize = 4;
/// Hard wall-clock ceiling for one producer within a synchronization round.
pub(crate) const DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS: u64 = 45;
/// TCP establishment remains short so unreachable peers fail over promptly.
const DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS: u64 = 3;
/// A verified carrier may audit thousands of retained blocks before exporting
/// one page. Keep the request bounded but leave enough time for that audit;
/// the independent producer-round deadline still caps the complete operation.
const DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS: u64 = 10;
/// Maximum producer-local retry delay after repeated consecutive failures.
pub(crate) const DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS: u64 =
    DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS;
/// Minimum delay before the first synchronization round after startup.
const DIRECTORY_SYNC_STARTUP_DELAY_MIN_SECS: u64 = 5;
/// Inclusive startup jitter span: 5 + (identity byte modulo 11) = 5-15 seconds.
const DIRECTORY_SYNC_STARTUP_DELAY_SPAN_SECS: u64 = 11;
/// Bounded retry cadence while at least one pinned producer is still catching up.
pub(crate) const DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS: u64 = 60;
/// Accepted signed response clock skew in either direction.
const DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS: u64 = 60;
/// External witnesses receive one complete producer-sync interval to catch up.
pub(crate) const DIRECTORY_OBSERVATION_WITNESS_MATURITY_INTERVALS: u64 = 1;
/// Hard response ceiling shared with the core Directory Sync decoder.
const MAX_DIRECTORY_SYNC_RESPONSE_BODY_BYTES: usize = 512 * 1024;
/// Multi-block pages accelerate cold catch-up. Peer handlers cap each returned
/// page to one block's maximum aggregate commitment budget, so hydration keeps
/// the same body and request ceiling as the original one-block transport.
const OUTBOUND_BLOCKS_PER_PAGE: u16 = MAX_DIRECTORY_SYNC_BLOCKS_V1;
/// One failed direct range, one carrier range, and bounded object chunks.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE: u32 =
    2 + MAX_DIRECTORY_COMMITMENTS_PER_BLOCK.div_ceil(MAX_DIRECTORY_SYNC_OBJECTS_V1) as u32;
/// Hard producer-local page cap for one low-frequency synchronization round.
/// Up to eight exceptionally sparse pages are permitted. The independent
/// worst-case request budget normally stops the common block-plus-object path
/// after seven pages and leaves capacity under the inbound identity budget for
/// witness and control traffic.
pub(crate) const DIRECTORY_SYNC_MAX_PAGES_PER_ROUND: u32 = 8;
/// Matches, but never exceeds, the inbound 30 requests/minute identity budget.
/// Worst-case pages consume the complete round; ordinary sparse blocks can
/// use the remaining budget without crossing the peer admission ceiling.
pub(crate) const DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND: u32 = 30;
/// Permissionless mirror work is intentionally below authority fan-out limits.
const DIRECTORY_MIRROR_MAX_ATTEMPTS_PER_ROUND: usize = 8;
/// [MIRROR-CATCHUP 2026-07-24 by Codex] A permissionless producer may advance
/// several authenticated pages per selection, but never consume the larger
/// pinned-authority budget in one round.
pub(crate) const DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND: u32 = 4;
/// Successful direct/carrier range and object hydration requests are bounded
/// independently from the 45-second wall-clock deadline.
pub(crate) const DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND: u32 = 24;
/// One direct mirror failure may try at most two independent public carriers.
const DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE: usize = 2;
/// One direct failure and one unsuccessful recovery carrier can precede the
/// existing worst-case successful carrier page.
const DIRECTORY_MIRROR_MAX_REQUESTS_PER_PAGE: u32 = DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE + 1;
/// Keep carrier choice stable within a round while avoiding permanent affinity.
const DIRECTORY_MIRROR_RECOVERY_ROTATION_SECS: u64 = 5 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryMirrorPullSource {
    DirectProducer,
    PublicCarrier,
}

#[derive(Debug)]
struct DirectoryMirrorPullFailure {
    reason: String,
    recovery_attempted: bool,
}

/// Aggregate result for one selected permissionless producer.
///
/// This deliberately carries no producer, carrier, endpoint, hash, or route.
/// A producer can make durable progress without yet reaching the signed tip;
/// that state must be reported as catching up instead of healthy/converged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct DirectoryMirrorProducerRoundOutcome {
    pages_succeeded: u32,
    requests_sent: u32,
    converged: bool,
    failed: bool,
}

/// Result of one authenticated outbound replica synchronization page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectorySyncPullOutcome {
    /// Durable replica import result.
    pub import: DirectoryReplicaImportReport,
    /// Whether the signed remote tip extends beyond this page.
    pub has_more: bool,
    /// Signed remote tip height observed in this round.
    pub remote_tip_height: u64,
    /// Signed remote tip hash observed in this round.
    pub remote_tip_hash: [u8; 32],
    /// HTTP requests consumed by this successful page and object hydration.
    pub requests_made: u32,
}

fn directory_sync_outcome_is_checkpoint_complete(outcome: &DirectorySyncPullOutcome) -> bool {
    !outcome.has_more
        && outcome.import.tip_height == outcome.remote_tip_height
        && outcome.import.tip_hash == outcome.remote_tip_hash
}

/// Whether another page can be requested without violating the conservative
/// worst-case request budget.
#[must_use]
pub(crate) const fn should_continue_directory_replica_catch_up(
    pages_completed: u32,
    requests_used: u32,
    has_more: bool,
) -> bool {
    has_more
        && pages_completed < DIRECTORY_SYNC_MAX_PAGES_PER_ROUND
        && requests_used.saturating_add(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
            <= DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND
}

const fn directory_sync_next_round_delay(
    configured_interval: Duration,
    all_producers_synchronized: bool,
) -> Duration {
    if all_producers_synchronized {
        configured_interval
    } else {
        let catch_up_interval = Duration::from_secs(DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS);
        if configured_interval.as_secs() < catch_up_interval.as_secs() {
            configured_interval
        } else {
            catch_up_interval
        }
    }
}

fn directory_sync_request_count_for_objects(object_count: usize) -> u32 {
    let object_requests = object_count.div_ceil(MAX_DIRECTORY_SYNC_OBJECTS_V1);
    1u32.saturating_add(u32::try_from(object_requests).unwrap_or(u32::MAX))
}

/// Returns the retry delay after a consecutive producer failure.
///
/// The first failure is retried on the next ordinary tick. Later failures skip
/// 1, 3, 7, then at most 15 nominal intervals before the hard delay cap.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
fn directory_sync_failure_backoff_delay_secs(interval_secs: u64, consecutive_failures: u64) -> u64 {
    if consecutive_failures <= 1 {
        return 0;
    }
    let exponent = consecutive_failures.saturating_sub(1).min(4) as u32;
    let multiplier = (1u64 << exponent).saturating_sub(1);
    interval_secs
        .saturating_mul(multiplier)
        .min(DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS)
}

struct DirectoryRangePage {
    blocks: Vec<DirectoryCommitmentBlockV1>,
    has_more: bool,
    remote_tip_height: u64,
    remote_tip_hash: [u8; 32],
    signed_response: Vec<u8>,
}

/// Process-local negative capability cache for the optional witness endpoint.
///
/// A negative observation is scoped to the exact sequence of an authenticated
/// node descriptor. A software upgrade publishes a newer signed sequence and
/// therefore becomes probeable without a timer, version-string comparison, or
/// mutable operator override. The cache never grants trust: every successful
/// response still passes the complete canonical frame and signature checks.
#[derive(Debug, Default)]
struct DirectoryWitnessCapabilityCache {
    unsupported_descriptor_sequences: Mutex<HashMap<[u8; 32], u64>>,
}

impl DirectoryWitnessCapabilityCache {
    fn should_attempt(&self, witness: &[u8; 32], descriptor_sequence: u64) -> bool {
        match self.unsupported_descriptor_sequences.lock().get(witness) {
            Some(unsupported_sequence) => *unsupported_sequence != descriptor_sequence,
            None => true,
        }
    }

    fn record_unsupported(&self, witness: [u8; 32], descriptor_sequence: u64) {
        self.unsupported_descriptor_sequences
            .lock()
            .insert(witness, descriptor_sequence);
    }

    fn record_supported(&self, witness: &[u8; 32]) {
        self.unsupported_descriptor_sequences.lock().remove(witness);
    }
}

/// Typed result boundary for untrusted peer HTTP exchange.
///
/// Keeping the status code typed until the caller applies operation-specific
/// policy prevents string parsing from becoming part of capability negotiation.
/// The type deliberately carries no URL, response body, peer identity, or
/// request material because failures can flow into operator telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryFramePostError {
    Transport,
    HttpStatus(u16),
    Response(BoundedHttpResponseError),
}

impl DirectoryFramePostError {
    const fn witness_capability_unavailable(self) -> bool {
        matches!(self, Self::HttpStatus(404 | 405 | 501))
    }

    fn stable_reason(self, operation: &str) -> String {
        match self {
            Self::Transport => format!("directory_{operation}_transport_failed"),
            Self::HttpStatus(status) => {
                format!("directory_{operation}_http_status_{status}")
            }
            Self::Response(error) => format!("directory_{operation}_{}", error.as_str()),
        }
    }
}

/// Immutable authority and mirror policy for one synchronization coordinator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectoryReplicaSyncPolicy {
    /// Minimum independent pinned witnesses for observation evidence.
    pub(crate) witness_min_verified: usize,
    /// Enables non-authoritative permissionless producer mirroring.
    pub(crate) full_node_mirror_enabled: bool,
    /// Durable namespace ceiling for permissionless mirror producers.
    pub(crate) full_node_mirror_max_producers: usize,
}

/// Coordinates bounded synchronization for operator-pinned Directory producers.
pub struct DirectoryReplicaSyncCoordinator {
    peers: Arc<[[u8; 32]]>,
    interval: Duration,
    store: Arc<DirectoryReplicaStore>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    client: reqwest::Client,
    witness_capabilities: DirectoryWitnessCapabilityCache,
    policy_anchor_capabilities: DirectoryWitnessCapabilityCache,
    witness_min_verified: usize,
    restored_retry_states: usize,
    full_node_mirror_enabled: bool,
    full_node_mirror_max_producers: usize,
    mirror_round_cursor: AtomicU64,
}

impl DirectoryReplicaSyncCoordinator {
    /// Builds a coordinator and its hardened, redirect-free HTTP client.
    ///
    /// # Errors
    /// Returns a stable reason when the configured interval or producer set is
    /// empty, or when the HTTP client cannot be initialized.
    pub fn new(
        peers: Vec<[u8; 32]>,
        interval_secs: u64,
        store: Arc<DirectoryReplicaStore>,
        runtime: Arc<DirectoryReplicaSyncRuntime>,
        peer_store: Arc<PeerStore>,
        identity: Arc<IdentityKeyPair>,
        witness_min_verified: usize,
    ) -> Result<Self, &'static str> {
        Self::new_with_policy(
            peers,
            interval_secs,
            store,
            runtime,
            peer_store,
            identity,
            DirectoryReplicaSyncPolicy {
                witness_min_verified,
                full_node_mirror_enabled: false,
                full_node_mirror_max_producers: 32,
            },
        )
    }

    /// Builds a coordinator with explicit authority and mirror policy.
    pub(crate) fn new_with_policy(
        peers: Vec<[u8; 32]>,
        interval_secs: u64,
        store: Arc<DirectoryReplicaStore>,
        runtime: Arc<DirectoryReplicaSyncRuntime>,
        peer_store: Arc<PeerStore>,
        identity: Arc<IdentityKeyPair>,
        policy: DirectoryReplicaSyncPolicy,
    ) -> Result<Self, &'static str> {
        let DirectoryReplicaSyncPolicy {
            witness_min_verified,
            full_node_mirror_enabled,
            full_node_mirror_max_producers,
        } = policy;
        if peers.is_empty() && !full_node_mirror_enabled {
            return Err("directory_sync_no_producers_or_mirror_mode");
        }
        if interval_secs == 0 {
            return Err("directory_sync_interval_invalid");
        }
        if !peers.is_empty()
            && (witness_min_verified == 0 || witness_min_verified > peers.len())
        {
            return Err("directory_observation_witness_threshold_invalid");
        }
        if full_node_mirror_enabled
            && !(1..=crate::services::directory_replica::MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS)
                .contains(&full_node_mirror_max_producers)
        {
            return Err("directory_full_node_mirror_capacity_invalid");
        }
        store
            .promote_pinned_producers(&peers)
            .map_err(|_| "directory_mirror_authority_promotion_failed")?;
        if full_node_mirror_enabled {
            store
                .ensure_mirror_capacity(full_node_mirror_max_producers)
                .map_err(|error| match error {
                    DirectoryReplicaStoreError::MirrorCapacity => {
                        "directory_mirror_registry_exceeds_configured_capacity"
                    }
                    _ => "directory_mirror_capacity_audit_failed",
                })?;
        }
        runtime.register_producers(&peers);
        let retry_states = store
            .retry_states()
            .map_err(|_| "directory_sync_retry_state_restore_failed")?
            .into_iter()
            .filter(|state| peers.contains(&state.producer))
            .collect::<Vec<_>>();
        runtime.restore_retry_states(&retry_states);
        let restored_retry_states = retry_states.len();
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS))
            .redirect(reqwest::redirect::Policy::none())
            .pool_max_idle_per_host(1)
            .build()
            .map_err(|_| "directory_sync_http_client_initialization_failed")?;
        Ok(Self {
            peers: peers.into(),
            interval: Duration::from_secs(interval_secs),
            store,
            runtime,
            peer_store,
            identity,
            client,
            witness_capabilities: DirectoryWitnessCapabilityCache::default(),
            policy_anchor_capabilities: DirectoryWitnessCapabilityCache::default(),
            witness_min_verified,
            restored_retry_states,
            full_node_mirror_enabled,
            full_node_mirror_max_producers,
            mirror_round_cursor: AtomicU64::new(0),
        })
    }

    /// Spawns the coordinator lifecycle task.
    #[must_use]
    pub fn spawn(self, mut shutdown_rx: broadcast::Receiver<()>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let startup_delay = Duration::from_secs(directory_sync_startup_delay_secs(
                &self.identity.public_key_bytes(),
            ));
            info!(
                pinned_producers = self.peers.len(),
                max_concurrent_producers = DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS,
                startup_delay_secs = startup_delay.as_secs(),
                interval_secs = self.interval.as_secs(),
                catch_up_interval_secs = DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS,
                restored_retry_states = self.restored_retry_states,
                full_node_mirror_enabled = self.full_node_mirror_enabled,
                full_node_mirror_capacity = self.full_node_mirror_max_producers,
                "[DIRECTORY_REPLICA] Synchronization coordinator started"
            );

            let mut next_delay = startup_delay;
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    () = tokio::time::sleep(next_delay) => {}
                }
                let round = self.synchronize_round();
                let all_producers_synchronized = tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    complete = round => complete,
                };
                next_delay =
                    directory_sync_next_round_delay(self.interval, all_producers_synchronized);
            }
            info!("[DIRECTORY_REPLICA] Synchronization coordinator stopped");
        })
    }

    async fn synchronize_round(&self) -> bool {
        let outcomes = stream::iter(self.peers.iter().copied())
            .map(|producer| async move { self.synchronize_producer(producer).await })
            .buffer_unordered(DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS)
            .collect::<Vec<_>>()
            .await;
        let all_producers_synchronized =
            outcomes.len() == self.peers.len() && outcomes.iter().all(|complete| *complete);
        // Policy rollback detection must not wait for replica convergence. A
        // temporarily unavailable producer cannot be allowed to suppress the
        // independent external high-water check after a host rollback.
        if !self.peers.is_empty() {
            self.anchor_current_observation_witness_policy().await;
        }
        if !self.peers.is_empty() && all_producers_synchronized {
            self.persist_observation_checkpoint().await;
        }
        if self.full_node_mirror_enabled {
            self.synchronize_full_node_mirrors().await;
        }
        all_producers_synchronized
    }

    async fn synchronize_full_node_mirrors(&self) {
        let now = unix_now_secs();
        let retained = {
            let store = Arc::clone(&self.store);
            let Ok(Ok(producers)) =
                tokio::task::spawn_blocking(move || store.mirror_producer_ids()).await
            else {
                self.runtime
                    .record_full_node_mirror_round(0, 0, 0, now);
                warn!(
                    reason = "directory_mirror_registry_read_failed",
                    "[DIRECTORY_REPLICA] Full-node Mirror round skipped"
                );
                return;
            };
            producers
        };
        let retained_set = retained.iter().copied().collect::<HashSet<_>>();
        let pinned = self.peers.iter().copied().collect::<HashSet<_>>();
        let local = self.identity.public_key_bytes();
        let mut candidates = self
            .peer_store
            .valid_public_descriptors(now, usize::MAX)
            .into_iter()
            .filter(|descriptor| {
                let node_id = descriptor.node_id();
                node_id != local
                    && !pinned.contains(&node_id)
                    && descriptor
                        .descriptor
                        .public_endpoint
                        .as_deref()
                        .is_some_and(commitment_peer_endpoint_is_public)
            })
            .map(|descriptor| (descriptor.node_id(), descriptor.sequence()))
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(node_id, _)| (!retained_set.contains(node_id), *node_id));
        let candidate_count = candidates.len();
        let open_slots = self
            .full_node_mirror_max_producers
            .saturating_sub(retained_set.len());
        let mut new_selected = 0usize;
        candidates.retain(|(node_id, _)| {
            if retained_set.contains(node_id) {
                true
            } else if new_selected < open_slots {
                new_selected = new_selected.saturating_add(1);
                true
            } else {
                false
            }
        });
        candidates.truncate(self.full_node_mirror_max_producers);
        if candidates.is_empty() {
            self.runtime
                .record_full_node_mirror_round(candidate_count, 0, 0, now);
            return;
        }
        let cursor = usize::try_from(self.mirror_round_cursor.fetch_add(1, Ordering::Relaxed))
            .unwrap_or(0)
            % candidates.len();
        candidates.rotate_left(cursor);
        candidates.truncate(DIRECTORY_MIRROR_MAX_ATTEMPTS_PER_ROUND);
        let selected = candidates.len();
        let outcomes = stream::iter(candidates)
            .map(|(producer, descriptor_sequence)| async move {
                self.synchronize_full_node_mirror(producer, descriptor_sequence)
                    .await
            })
            .buffer_unordered(DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS)
            .collect::<Vec<_>>()
            .await;
        let converged = outcomes.iter().filter(|outcome| outcome.converged).count();
        let catching_up = outcomes
            .iter()
            .filter(|outcome| !outcome.converged && !outcome.failed)
            .count();
        let failed = outcomes.iter().filter(|outcome| outcome.failed).count();
        let pages_succeeded = outcomes.iter().fold(0u64, |total, outcome| {
            total.saturating_add(u64::from(outcome.pages_succeeded))
        });
        let requests_sent = outcomes.iter().fold(0u64, |total, outcome| {
            total.saturating_add(u64::from(outcome.requests_sent))
        });
        self.runtime.record_full_node_mirror_catch_up_round(
            candidate_count,
            selected,
            converged,
            catching_up,
            failed,
            pages_succeeded,
            requests_sent,
            unix_now_secs(),
        );
        debug!(
            candidates = candidate_count,
            selected,
            converged,
            catching_up,
            failed,
            pages_succeeded,
            requests_sent,
            retained = retained_set.len(),
            capacity = self.full_node_mirror_max_producers,
            "[DIRECTORY_REPLICA] Full-node Mirror round completed"
        );
    }

    async fn synchronize_full_node_mirror(
        &self,
        producer: [u8; 32],
        descriptor_sequence: u64,
    ) -> DirectoryMirrorProducerRoundOutcome {
        // [MIRROR-CATCHUP 2026-07-24 by Codex] Use one absolute deadline for
        // every page so a slow carrier cannot multiply the producer budget.
        // Completed page metrics remain available if a later page times out.
        let deadline = tokio::time::Instant::now()
            + Duration::from_secs(DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS);
        let mut round = DirectoryMirrorProducerRoundOutcome::default();
        loop {
            let result = tokio::time::timeout_at(
                deadline,
                pull_directory_chain_mirror_page_with_recovery(
                    Arc::clone(&self.store),
                    self.peer_store.as_ref(),
                    self.identity.as_ref(),
                    &producer,
                    descriptor_sequence,
                    self.full_node_mirror_max_producers,
                    &self.client,
                ),
            )
            .await;
            let (outcome, source) = match result {
                Ok(Ok(value)) => value,
                Ok(Err(failure)) => {
                    if failure.recovery_attempted {
                        self.runtime
                            .record_full_node_mirror_recovery(false, unix_now_secs());
                    }
                    debug!(
                        reason = failure.reason,
                        recovery_attempted = failure.recovery_attempted,
                        pages_succeeded = round.pages_succeeded,
                        requests_sent = round.requests_sent,
                        "[DIRECTORY_REPLICA] Full-node Mirror pull rejected"
                    );
                    round.failed = true;
                    return round;
                }
                Err(_) => {
                    debug!(
                        reason = "directory_mirror_producer_round_timeout",
                        pages_succeeded = round.pages_succeeded,
                        requests_sent = round.requests_sent,
                        "[DIRECTORY_REPLICA] Full-node Mirror catch-up deadline reached"
                    );
                    round.failed = true;
                    return round;
                }
            };
            round.pages_succeeded = round.pages_succeeded.saturating_add(1);
            round.requests_sent = round.requests_sent.saturating_add(outcome.requests_made);
            if source == DirectoryMirrorPullSource::PublicCarrier {
                self.runtime
                    .record_full_node_mirror_recovery(true, unix_now_secs());
            }
            if directory_sync_outcome_is_checkpoint_complete(&outcome) {
                round.converged = true;
                return round;
            }
            if !outcome.has_more {
                warn!(
                    reason = "directory_mirror_terminal_page_not_converged",
                    pages_succeeded = round.pages_succeeded,
                    requests_sent = round.requests_sent,
                    "[DIRECTORY_REPLICA] Full-node Mirror terminal page failed convergence"
                );
                round.failed = true;
                return round;
            }
            if !should_continue_directory_mirror_catch_up(
                round.pages_succeeded,
                round.requests_sent,
                outcome.has_more,
            ) {
                return round;
            }
        }
    }

    async fn synchronize_producer(&self, producer: [u8; 32]) -> bool {
        let now = unix_now_secs();
        if let Some(retry_at) = self.runtime.deferred_retry_until(&producer, now) {
            let retry_state_durable = self.persist_retry_skip(producer, now).await;
            self.runtime.record_backoff_skip(producer);
            debug!(
                retry_after_secs = retry_at.saturating_sub(now),
                retry_state_durable,
                "[DIRECTORY_REPLICA] Producer synchronization deferred by backoff"
            );
            return false;
        }
        let Ok(complete) = tokio::time::timeout(
            Duration::from_secs(DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS),
            self.synchronize_producer_pages(producer),
        )
        .await
        else {
            self.record_producer_failure(producer, "directory_producer_round_timeout", None, None)
                .await;
            return false;
        };
        complete
    }

    async fn synchronize_producer_pages(&self, producer: [u8; 32]) -> bool {
        let mut pages_completed = 0u32;
        let mut requests_used = 0u32;
        loop {
            self.runtime.record_attempt(producer, unix_now_secs());
            match pull_directory_chain_page_with_carriers(
                Arc::clone(&self.store),
                &self.peer_store,
                self.identity.as_ref(),
                &producer,
                self.peers.as_ref(),
                &self.client,
            )
            .await
            {
                Ok(outcome) => {
                    pages_completed = pages_completed.saturating_add(1);
                    requests_used = requests_used.saturating_add(outcome.requests_made);
                    self.runtime.record_success(
                        producer,
                        unix_now_secs(),
                        outcome.import.tip_height,
                        outcome.remote_tip_height,
                        outcome.has_more,
                        outcome.import.blocks_inserted,
                        outcome.import.commitments_inserted,
                        outcome.requests_made,
                    );
                    debug!(
                        blocks_inserted = outcome.import.blocks_inserted,
                        commitments_inserted = outcome.import.commitments_inserted,
                        blocks_already_present = outcome.import.blocks_already_present,
                        descriptor_equivocations = outcome.import.descriptor_equivocations,
                        replica_tip_height = outcome.import.tip_height,
                        remote_tip_height = outcome.remote_tip_height,
                        has_more = outcome.has_more,
                        pages_completed,
                        requests_used,
                        "[DIRECTORY_REPLICA] Authenticated bounded page synchronized"
                    );
                    if !should_continue_directory_replica_catch_up(
                        pages_completed,
                        requests_used,
                        outcome.has_more,
                    ) {
                        return directory_sync_outcome_is_checkpoint_complete(&outcome);
                    }
                }
                Err(reason) => {
                    self.record_producer_failure(
                        producer,
                        &reason,
                        Some(pages_completed),
                        Some(requests_used),
                    )
                    .await;
                    return false;
                }
            }
        }
    }

    async fn persist_observation_checkpoint(&self) {
        let store = Arc::clone(&self.store);
        let peers = Arc::clone(&self.peers);
        let identity = Arc::clone(&self.identity);
        let observed_at = unix_now_secs();
        match tokio::task::spawn_blocking(move || {
            store.append_observation_checkpoint(peers.as_ref(), identity.as_ref(), observed_at)
        })
        .await
        {
            Ok(Ok(report)) => {
                debug!(
                    appended = report.appended,
                    sequence = report.sequence,
                    producer_count = report.producer_count,
                    "[DIRECTORY_REPLICA] Complete observation checkpoint evaluated"
                );
                self.witness_mature_observation_checkpoint().await;
            }
            Ok(Err(_)) | Err(_) => {
                warn!(
                    reason = "directory_observation_checkpoint_persist_failed",
                    "[DIRECTORY_REPLICA] Complete observation checkpoint rejected"
                );
            }
        }
    }

    async fn witness_mature_observation_checkpoint(&self) {
        let store = Arc::clone(&self.store);
        let eligible_witnesses = Arc::clone(&self.peers);
        let minimum_witnesses = self.witness_min_verified;
        let observed_at = unix_now_secs();
        let maturity_delay_secs = self
            .interval
            .as_secs()
            .saturating_mul(DIRECTORY_OBSERVATION_WITNESS_MATURITY_INTERVALS);
        let matured_before = observed_at.saturating_sub(maturity_delay_secs);
        if matured_before == 0 {
            return;
        }
        let target = match tokio::task::spawn_blocking(move || {
            store.next_audited_mature_observation_checkpoint_below_witness_threshold(
                matured_before,
                observed_at,
                minimum_witnesses,
                eligible_witnesses.as_ref(),
            )
        })
        .await
        {
            Ok(Ok(Some(target))) => target,
            Ok(Ok(None)) => return,
            Ok(Err(_)) | Err(_) => {
                warn!(
                    reason = "directory_observation_checkpoint_audit_failed",
                    "[DIRECTORY_REPLICA] External witness round skipped"
                );
                return;
            }
        };
        let checkpoint = target.checkpoint.clone();
        debug!(
            checkpoint_sequence = checkpoint.sequence,
            checkpoint_age_seconds = observed_at.saturating_sub(checkpoint.observed_at),
            maturity_delay_secs,
            retained_pinned_witnesses = target.witnessed_by.len(),
            minimum_witnesses = target.minimum_witnesses,
            "[DIRECTORY_REPLICA] Mature checkpoint below witness target selected"
        );
        let outcomes = stream::iter(
            self.peers
                .iter()
                .copied()
                .filter(|witness| !target.witnessed_by.contains(witness)),
        )
        .map(|witness| {
            let checkpoint = checkpoint.clone();
            async move {
                request_observation_checkpoint_witness(
                    Arc::clone(&self.store),
                    self.peer_store.as_ref(),
                    self.identity.as_ref(),
                    &witness,
                    &self.client,
                    &self.witness_capabilities,
                    checkpoint,
                )
                .await
            }
        })
        .buffer_unordered(DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS)
        .collect::<Vec<_>>()
        .await;
        self.record_witness_outcome_round(checkpoint.sequence, outcomes)
            .await;
    }

    async fn anchor_current_observation_witness_policy(&self) {
        let store = Arc::clone(&self.store);
        let eligible_witnesses = Arc::clone(&self.peers);
        let observed_at = unix_now_secs();
        let anchor = match tokio::task::spawn_blocking(move || {
            let Some(anchor) = store.current_observation_witness_policy_anchor()? else {
                return Ok::<_, DirectoryReplicaStoreError>(None);
            };
            let witnessed = store.verified_observation_witness_policy_anchor_witnesses_for_pins(
                anchor.epoch,
                &anchor.policy_digest,
                eligible_witnesses.as_ref(),
                observed_at,
            )?;
            Ok(Some((anchor, witnessed)))
        })
        .await
        {
            Ok(Ok(Some(anchor))) => anchor,
            Ok(Ok(None)) => return,
            Ok(Err(_)) | Err(_) => {
                warn!(
                    reason = "directory_observation_policy_anchor_audit_failed",
                    "[DIRECTORY_REPLICA] Policy-head anchor round skipped"
                );
                return;
            }
        };
        if anchor.1.len() >= self.witness_min_verified {
            return;
        }
        let outcomes = stream::iter(
            self.peers
                .iter()
                .copied()
                .filter(|witness| !anchor.1.contains(witness)),
        )
        .map(|witness| async move {
            request_observation_policy_anchor(
                Arc::clone(&self.store),
                self.peer_store.as_ref(),
                self.identity.as_ref(),
                &witness,
                &self.client,
                &self.policy_anchor_capabilities,
                anchor.0,
            )
            .await
        })
        .buffer_unordered(DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS)
        .collect::<Vec<_>>()
        .await;
        debug!(
            policy_epoch = anchor.0.epoch,
            attempted_witnesses = outcomes.len(),
            accepted =
                witness_outcome_count(&outcomes, DirectoryObservationWitnessOutcome::Accepted),
            "[DIRECTORY_REPLICA] Opaque policy-head anchor round completed"
        );
    }

    async fn record_witness_outcome_round(
        &self,
        checkpoint_sequence: u64,
        outcomes: Vec<DirectoryObservationWitnessOutcome>,
    ) {
        let completed_at = unix_now_secs();
        let durable_store = Arc::clone(&self.store);
        let durable_outcomes = outcomes.clone();
        let telemetry_durable = tokio::task::spawn_blocking(move || {
            durable_store.persist_observation_witness_outcome_round(
                checkpoint_sequence,
                completed_at,
                &durable_outcomes,
            )
        })
        .await
        .is_ok_and(|result| result.is_ok());
        self.runtime.record_observation_witness_round(
            checkpoint_sequence,
            completed_at,
            &outcomes,
            telemetry_durable,
        );
        if !telemetry_durable {
            warn!(
                reason = "directory_observation_witness_telemetry_persist_failed",
                "[DIRECTORY_REPLICA] Witness outcome aggregate was not durable"
            );
        }
        let accepted =
            witness_outcome_count(&outcomes, DirectoryObservationWitnessOutcome::Accepted);
        let evidence_unavailable = witness_outcome_count(
            &outcomes,
            DirectoryObservationWitnessOutcome::EvidenceUnavailable,
        );
        let evidence_conflict = witness_outcome_count(
            &outcomes,
            DirectoryObservationWitnessOutcome::EvidenceConflict,
        );
        let peer_unavailable = witness_outcome_count(
            &outcomes,
            DirectoryObservationWitnessOutcome::PeerUnavailable,
        );
        let transport_failures = witness_outcome_count(
            &outcomes,
            DirectoryObservationWitnessOutcome::TransportFailure,
        );
        let verification_failures = witness_outcome_count(
            &outcomes,
            DirectoryObservationWitnessOutcome::VerificationFailure,
        );
        let persistence_failures = witness_outcome_count(
            &outcomes,
            DirectoryObservationWitnessOutcome::PersistenceFailure,
        );
        debug!(
            checkpoint_sequence,
            attempted_witnesses = outcomes.len(),
            accepted,
            evidence_unavailable,
            evidence_conflict,
            peer_unavailable,
            transport_failures,
            verification_failures,
            persistence_failures,
            telemetry_durable,
            "[DIRECTORY_REPLICA] Bounded observation checkpoint witness round completed"
        );
    }

    async fn record_producer_failure(
        &self,
        producer: [u8; 32],
        reason: &str,
        pages_completed: Option<u32>,
        requests_used: Option<u32>,
    ) {
        let failed_at = unix_now_secs();
        let consecutive_failures = self
            .runtime
            .consecutive_failures(&producer)
            .saturating_add(1)
            .min(DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES);
        let retry_delay_secs = directory_sync_failure_backoff_delay_secs(
            self.interval.as_secs(),
            consecutive_failures,
        );
        let retry_not_before =
            (retry_delay_secs > 0).then(|| failed_at.saturating_add(retry_delay_secs));
        let store = Arc::clone(&self.store);
        let durable_reason = reason.to_string();
        let retry_state_durable = tokio::task::spawn_blocking(move || {
            store.persist_retry_failure(
                producer,
                consecutive_failures,
                retry_not_before,
                failed_at,
                &durable_reason,
            )
        })
        .await
        .is_ok_and(|result| result.is_ok());
        self.runtime
            .record_failure(producer, failed_at, reason, retry_not_before);
        warn!(
            reason = %reason,
            consecutive_failures,
            retry_delay_secs,
            retry_state_durable,
            pages_completed = ?pages_completed,
            requests_used = ?requests_used,
            "[DIRECTORY_REPLICA] Pinned producer sync round rejected"
        );
    }

    async fn persist_retry_skip(&self, producer: [u8; 32], skipped_at: u64) -> bool {
        let store = Arc::clone(&self.store);
        let durable =
            tokio::task::spawn_blocking(move || store.persist_retry_skip(producer, skipped_at))
                .await
                .is_ok_and(|result| result.is_ok());
        if !durable {
            warn!(
                reason = "directory_retry_skip_persist_failed",
                "[DIRECTORY_REPLICA] Durable retry skip update rejected"
            );
        }
        durable
    }
}

async fn request_observation_policy_anchor(
    store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness: &[u8; 32],
    client: &reqwest::Client,
    capability_cache: &DirectoryWitnessCapabilityCache,
    anchor: crate::services::directory_replica::DirectoryObservationWitnessPolicyAnchor,
) -> DirectoryObservationWitnessOutcome {
    let request_timestamp = unix_now_secs();
    let Some(descriptor) = peer_store.get_valid(witness, request_timestamp) else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    if !commitment_peer_endpoint_is_public(endpoint) {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    }
    let descriptor_sequence = descriptor.sequence();
    if !capability_cache.should_attempt(witness, descriptor_sequence) {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    }
    let Ok(url) = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/observation-policy-anchor",
    ) else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_policy_anchor_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester,
        request_timestamp,
        anchor.epoch,
        &anchor.previous_policy_digest,
        &anchor.policy_digest,
    );
    let request = DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester,
        request_timestamp,
        policy_epoch: anchor.epoch,
        previous_policy_digest: anchor.previous_policy_digest,
        policy_digest: anchor.policy_digest,
        signature: identity.sign(&signing_bytes),
    };
    let Ok(frame) = encode_directory_sync_message(&request) else {
        return DirectoryObservationWitnessOutcome::VerificationFailure;
    };
    let response = match post_directory_frame_typed(client, url, frame).await {
        Ok(response) => {
            capability_cache.record_supported(witness);
            response
        }
        Err(error) if error.witness_capability_unavailable() => {
            capability_cache.record_unsupported(*witness, descriptor_sequence);
            return DirectoryObservationWitnessOutcome::PeerUnavailable;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::TransportFailure,
    };
    let verified = match verify_observation_policy_anchor_response(
        &response,
        &request_id,
        &requester,
        witness,
        request_timestamp,
        anchor.epoch,
        &anchor.policy_digest,
    ) {
        Ok(response) => response,
        Err(reason) if reason == "observation_policy_anchor_rollback" => {
            return DirectoryObservationWitnessOutcome::EvidenceConflict;
        }
        Err(reason) if reason == "observation_policy_anchor_conflict" => {
            return DirectoryObservationWitnessOutcome::EvidenceConflict;
        }
        Err(reason) if reason == "observation_policy_anchor_history_gap" => {
            return DirectoryObservationWitnessOutcome::EvidenceUnavailable;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::VerificationFailure,
    };
    let durable = tokio::task::spawn_blocking(move || {
        store.persist_observation_witness_policy_anchor_receipt(&verified, unix_now_secs())
    })
    .await
    .is_ok_and(|result| result.is_ok());
    if durable {
        DirectoryObservationWitnessOutcome::Accepted
    } else {
        DirectoryObservationWitnessOutcome::PersistenceFailure
    }
}

pub(crate) fn verify_observation_policy_anchor_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_observer: &[u8; 32],
    expected_witness: &[u8; 32],
    request_timestamp: u64,
    expected_policy_epoch: u64,
    expected_policy_digest: &[u8; 32],
) -> Result<DirectorySyncMessage, String> {
    let response = decode_directory_sync_message(frame)
        .map_err(|_| "observation_policy_anchor_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&response)
        .map_err(|_| "observation_policy_anchor_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("observation_policy_anchor_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
        chain_id,
        request_id,
        observer,
        policy_epoch,
        policy_digest,
        responder,
        response_timestamp,
        outcome,
        signature,
    } = &response
    else {
        return Err("observation_policy_anchor_response_unexpected_message".to_string());
    };
    if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != expected_request_id
        || observer != expected_observer
        || responder != expected_witness
        || *policy_epoch != expected_policy_epoch
        || policy_digest != expected_policy_digest
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || ![
            DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
            DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
            DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
        ]
        .contains(outcome)
    {
        return Err("observation_policy_anchor_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_policy_anchor_response_signing_bytes(
        chain_id,
        request_id,
        observer,
        *policy_epoch,
        policy_digest,
        responder,
        *response_timestamp,
        *outcome,
    );
    IdentityPublicKey::from_bytes(responder)
        .and_then(|key| key.verify(&signing_bytes, signature))
        .map_err(|_| "observation_policy_anchor_response_invalid_signature".to_string())?;
    match *outcome {
        DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1 => Ok(response),
        DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1 => {
            Err("observation_policy_anchor_rollback".to_string())
        }
        DIRECTORY_POLICY_ANCHOR_CONFLICT_V1 => {
            Err("observation_policy_anchor_conflict".to_string())
        }
        DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1 => {
            Err("observation_policy_anchor_history_gap".to_string())
        }
        _ => Err("observation_policy_anchor_response_outcome_invalid".to_string()),
    }
}

async fn request_observation_checkpoint_witness(
    store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness: &[u8; 32],
    client: &reqwest::Client,
    capability_cache: &DirectoryWitnessCapabilityCache,
    checkpoint: DirectoryObservationCheckpointV1,
) -> DirectoryObservationWitnessOutcome {
    let request_timestamp = unix_now_secs();
    let Some(descriptor) = peer_store.get_valid(witness, request_timestamp) else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    if !commitment_peer_endpoint_is_public(endpoint) {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    }
    let descriptor_sequence = descriptor.sequence();
    if !capability_cache.should_attempt(witness, descriptor_sequence) {
        debug!(
            reason = "directory_observation_witness_capability_cached_unavailable",
            "[DIRECTORY_REPLICA] Witness request skipped for unchanged signed descriptor"
        );
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    }
    let Ok(url) = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/observation-checkpoint-witness",
    ) else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let checkpoint_sequence = checkpoint.sequence;
    let checkpoint_hash = checkpoint.hash();
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester,
        request_timestamp,
        &checkpoint_hash,
    );
    let request = DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester,
        request_timestamp,
        checkpoint,
        signature: identity.sign(&signing_bytes),
    };
    let Ok(frame) = encode_directory_sync_message(&request) else {
        return DirectoryObservationWitnessOutcome::VerificationFailure;
    };
    let response = match post_directory_frame_typed(client, url, frame).await {
        Ok(response) => {
            capability_cache.record_supported(witness);
            response
        }
        Err(error) if error.witness_capability_unavailable() => {
            capability_cache.record_unsupported(*witness, descriptor_sequence);
            debug!(
                reason = "directory_observation_witness_capability_unavailable",
                http_status = match error {
                    DirectoryFramePostError::HttpStatus(status) => status,
                    _ => 0,
                },
                "[DIRECTORY_REPLICA] Peer descriptor does not currently expose witness service"
            );
            return DirectoryObservationWitnessOutcome::PeerUnavailable;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::TransportFailure,
    };
    let verified = match verify_observation_witness_response(
        &response,
        &request_id,
        &requester,
        witness,
        request_timestamp,
        checkpoint_sequence,
        &checkpoint_hash,
    ) {
        Ok(verified) => verified,
        Err(reason) if reason == "observation_witness_evidence_unavailable" => {
            return DirectoryObservationWitnessOutcome::EvidenceUnavailable;
        }
        Err(reason) if reason == "observation_witness_evidence_conflict" => {
            return DirectoryObservationWitnessOutcome::EvidenceConflict;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::VerificationFailure,
    };
    let durable = tokio::task::spawn_blocking(move || {
        store.persist_observation_checkpoint_witness(&verified, unix_now_secs())
    })
    .await
    .is_ok_and(|result| result.is_ok());
    if durable {
        DirectoryObservationWitnessOutcome::Accepted
    } else {
        DirectoryObservationWitnessOutcome::PersistenceFailure
    }
}

fn witness_outcome_count(
    outcomes: &[DirectoryObservationWitnessOutcome],
    expected: DirectoryObservationWitnessOutcome,
) -> usize {
    outcomes
        .iter()
        .filter(|outcome| **outcome == expected)
        .count()
}

pub(crate) fn verify_observation_witness_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_observer: &[u8; 32],
    expected_witness: &[u8; 32],
    request_timestamp: u64,
    expected_checkpoint_sequence: u64,
    expected_checkpoint_hash: &[u8; 32],
) -> Result<DirectorySyncMessage, String> {
    let response = decode_directory_sync_message(frame)
        .map_err(|_| "observation_witness_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&response)
        .map_err(|_| "observation_witness_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("observation_witness_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id,
        request_id,
        observer,
        checkpoint_sequence,
        checkpoint_hash,
        responder,
        response_timestamp,
        outcome,
        signature,
    } = &response
    else {
        return Err("observation_witness_response_unexpected_message".to_string());
    };
    if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != expected_request_id
        || observer != expected_observer
        || responder != expected_witness
        || *checkpoint_sequence != expected_checkpoint_sequence
        || checkpoint_hash != expected_checkpoint_hash
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || ![
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
        ]
        .contains(outcome)
    {
        return Err("observation_witness_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        chain_id,
        request_id,
        observer,
        *checkpoint_sequence,
        checkpoint_hash,
        responder,
        *response_timestamp,
        *outcome,
    );
    IdentityPublicKey::from_bytes(responder)
        .and_then(|key| key.verify(&signing_bytes, signature))
        .map_err(|_| "observation_witness_response_invalid_signature".to_string())?;
    match *outcome {
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1 => Ok(response),
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1 => {
            Err("observation_witness_evidence_unavailable".to_string())
        }
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1 => {
            Err("observation_witness_evidence_conflict".to_string())
        }
        _ => Err("observation_witness_response_outcome_invalid".to_string()),
    }
}

/// Pulls, verifies, hydrates, and atomically imports one pinned producer page.
///
/// The producer must have a current signed descriptor in `PeerStore`, and its
/// endpoint must be a public IP literal. Every response is canonicalized and
/// signature-verified before the blocking atomic import begins.
///
/// # Errors
/// Returns a stable privacy-safe reason code for unavailable descriptors,
/// unsafe endpoints, transport/status/body failures, invalid signed responses,
/// missing objects, replica integrity failures, or durable quarantine.
pub async fn pull_directory_chain_page(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "replica_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("producer_quarantined".to_string());
    }
    let (range_url, object_url) =
        directory_sync_peer_urls(peer_store, producer, request_timestamp)?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "replica_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_block_page(
        identity,
        producer,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_descriptor_objects(
        identity,
        producer,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_range_page(replica_store, *producer, page, objects, requests_made).await
}

/// Pulls one signed page into the bounded non-authoritative mirror set.
///
/// The producer is always tried first. Only availability/admission failures may
/// enter the bounded carrier path; canonical, signature, producer-binding,
/// descriptor, hash-chain, and durable integrity failures stop immediately.
async fn pull_directory_chain_mirror_page_with_recovery(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    client: &reqwest::Client,
) -> Result<(DirectorySyncPullOutcome, DirectoryMirrorPullSource), DirectoryMirrorPullFailure> {
    match pull_directory_chain_mirror_page(
        Arc::clone(&replica_store),
        peer_store,
        identity,
        producer,
        descriptor_sequence,
        max_mirror_producers,
        client,
    )
    .await
    {
        Ok(outcome) => Ok((outcome, DirectoryMirrorPullSource::DirectProducer)),
        Err(reason) if directory_mirror_failure_allows_recovery(&reason) => {
            // [MIRROR-CATCHUP 2026-07-24 by Codex] Conservatively reserve one
            // request for the direct attempt even when endpoint validation may
            // have failed before transport. Each retryable carrier failure can
            // consume at most its range request before another carrier is used.
            let mut prior_requests = 1u32;
            let carriers = directory_mirror_recovery_carriers(
                peer_store,
                producer,
                &identity.public_key_bytes(),
                unix_now_secs(),
            );
            for carrier in carriers {
                match pull_directory_chain_mirror_page_via_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    descriptor_sequence,
                    max_mirror_producers,
                    &carrier,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        return Ok((outcome, DirectoryMirrorPullSource::PublicCarrier));
                    }
                    Err(carrier_reason)
                        if directory_mirror_failure_allows_recovery(&carrier_reason) =>
                    {
                        prior_requests = prior_requests.saturating_add(1);
                        debug!(
                            reason = carrier_reason,
                            "[DIRECTORY_REPLICA] Full-node Mirror recovery carrier unavailable"
                        );
                    }
                    Err(carrier_reason) => {
                        return Err(DirectoryMirrorPullFailure {
                            reason: carrier_reason,
                            recovery_attempted: true,
                        });
                    }
                }
            }
            Err(DirectoryMirrorPullFailure {
                reason: "directory_mirror_recovery_exhausted".to_string(),
                recovery_attempted: true,
            })
        }
        Err(reason) => Err(DirectoryMirrorPullFailure {
            reason,
            recovery_attempted: false,
        }),
    }
}

/// Pulls one direct signed page into the bounded non-authoritative mirror set.
///
/// The exact discovery descriptor sequence selected for this attempt must still
/// be current and public when URLs are derived. This function performs no
/// fallback and never alters configured checkpoint/witness authority membership.
async fn pull_directory_chain_mirror_page(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "directory_mirror_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("directory_mirror_producer_quarantined".to_string());
    }
    let (range_url, object_url) = directory_mirror_peer_urls(
        peer_store,
        producer,
        descriptor_sequence,
        request_timestamp,
    )?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "directory_mirror_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_block_page(
        identity,
        producer,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_descriptor_objects(
        identity,
        producer,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_mirror_range_page(
        replica_store,
        *producer,
        descriptor_sequence,
        max_mirror_producers,
        page,
        objects,
        requests_made,
    )
    .await
}

// Each argument is a distinct authenticated protocol boundary. Grouping them
// into an opaque context would make producer/carrier confusion easier during
// security review, so keep the identities and bounded retention policy explicit.
#[allow(clippy::too_many_arguments)]
async fn pull_directory_chain_mirror_page_via_carrier(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    carrier: &[u8; 32],
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "directory_mirror_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("directory_mirror_producer_quarantined".to_string());
    }
    let (range_url, object_url) =
        directory_mirror_recovery_carrier_urls(peer_store, carrier, request_timestamp)?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "directory_mirror_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_replica_block_page(
        identity,
        producer,
        carrier,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_replica_descriptor_objects(
        identity,
        producer,
        carrier,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_mirror_range_page(
        replica_store,
        *producer,
        descriptor_sequence,
        max_mirror_producers,
        page,
        objects,
        requests_made,
    )
    .await
}

async fn pull_directory_chain_page_with_carriers(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carriers: &[[u8; 32]],
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    match pull_directory_chain_page(
        Arc::clone(&replica_store),
        peer_store,
        identity,
        producer,
        client,
    )
    .await
    {
        Ok(outcome) => Ok(outcome),
        Err(reason) if directory_sync_failure_allows_carrier_fallback(&reason) => {
            for carrier in carriers.iter().copied().filter(|candidate| {
                candidate != producer && *candidate != identity.public_key_bytes()
            }) {
                match pull_directory_chain_page_via_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    &carrier,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        // Direct-first consumed at most one failed range request;
                        // reserving it even for descriptor failures is conservative.
                        outcome.requests_made = outcome.requests_made.saturating_add(1);
                        debug!(
                            requests_made = outcome.requests_made,
                            "[DIRECTORY_REPLICA] Audited carrier recovered producer evidence"
                        );
                        return Ok(outcome);
                    }
                    Err(carrier_reason)
                        if directory_sync_failure_allows_carrier_fallback(&carrier_reason) => {}
                    Err(carrier_reason) => return Err(carrier_reason),
                }
            }
            Err("directory_carrier_fallback_exhausted".to_string())
        }
        Err(reason) => Err(reason),
    }
}

fn directory_sync_failure_allows_carrier_fallback(reason: &str) -> bool {
    if matches!(
        reason,
        "pinned_directory_peer_unavailable"
            | "pinned_directory_peer_missing_endpoint"
            | "pinned_directory_peer_unsafe_endpoint"
            | "pinned_directory_peer_invalid_endpoint"
    ) || reason == "directory_range_transport_failed"
        || reason == "directory_replica_range_transport_failed"
    {
        return true;
    }
    for prefix in [
        "directory_range_http_status_",
        "directory_replica_range_http_status_",
    ] {
        let Some(status) = reason
            .strip_prefix(prefix)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        return matches!(status, 403 | 404 | 408 | 429) || status >= 500;
    }
    false
}

fn directory_mirror_failure_allows_recovery(reason: &str) -> bool {
    if matches!(
        reason,
        "directory_range_transport_failed"
            | "directory_objects_transport_failed"
            | "directory_replica_range_transport_failed"
            | "directory_replica_objects_transport_failed"
            | "directory_mirror_recovery_carrier_unavailable"
            | "directory_mirror_recovery_carrier_not_public"
            | "directory_mirror_recovery_carrier_missing_endpoint"
            | "directory_mirror_recovery_carrier_unsafe_endpoint"
            | "directory_mirror_recovery_carrier_invalid_endpoint"
    ) {
        return true;
    }
    for prefix in [
        "directory_range_http_status_",
        "directory_objects_http_status_",
        "directory_replica_range_http_status_",
        "directory_replica_objects_http_status_",
    ] {
        let Some(status) = reason
            .strip_prefix(prefix)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        return matches!(status, 403 | 404 | 408 | 429) || status >= 500;
    }
    false
}

fn should_continue_directory_mirror_catch_up(
    pages_completed: u32,
    requests_used: u32,
    has_more: bool,
) -> bool {
    has_more
        && pages_completed < DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND
        && requests_used.saturating_add(DIRECTORY_MIRROR_MAX_REQUESTS_PER_PAGE)
            <= DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND
}

fn directory_mirror_recovery_carriers(
    peer_store: &PeerStore,
    producer: &[u8; 32],
    requester: &[u8; 32],
    now: u64,
) -> Vec<[u8; 32]> {
    let mut carriers = peer_store
        .valid_public_descriptors(now, usize::MAX)
        .into_iter()
        .filter(|descriptor| {
            let node_id = descriptor.node_id();
            node_id != *producer
                && node_id != *requester
                && descriptor.descriptor.policy.public_discovery
                && descriptor
                    .descriptor
                    .public_endpoint
                    .as_deref()
                    .is_some_and(commitment_peer_endpoint_is_public)
        })
        .map(|descriptor| descriptor.node_id())
        .collect::<Vec<_>>();
    carriers.sort_unstable();
    carriers.dedup();
    if !carriers.is_empty() {
        let producer_seed = u64::from_be_bytes(producer[..8].try_into().unwrap_or([0u8; 8]));
        let requester_seed = u64::from_be_bytes(requester[..8].try_into().unwrap_or([0u8; 8]));
        let epoch_seed = now / DIRECTORY_MIRROR_RECOVERY_ROTATION_SECS;
        let cursor = usize::try_from(producer_seed ^ requester_seed ^ epoch_seed).unwrap_or(0)
            % carriers.len();
        carriers.rotate_left(cursor);
        carriers.truncate(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE);
    }
    carriers
}

async fn pull_directory_chain_page_via_carrier(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "replica_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("producer_quarantined".to_string());
    }
    let (range_url, object_url) =
        directory_replica_carrier_urls(peer_store, carrier, request_timestamp)?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "replica_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_replica_block_page(
        identity,
        producer,
        carrier,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_replica_descriptor_objects(
        identity,
        producer,
        carrier,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_range_page(replica_store, *producer, page, objects, requests_made).await
}

fn directory_sync_peer_urls(
    peer_store: &PeerStore,
    producer: &[u8; 32],
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(producer, request_timestamp)
        .ok_or_else(|| "pinned_directory_peer_unavailable".to_string())?;
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_directory_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("pinned_directory_peer_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(endpoint, "/api/discovery/peer/directory/block-range")
        .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    let object_url =
        commitment_peer_url(endpoint, "/api/discovery/peer/directory/descriptor-objects")
            .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

fn directory_mirror_peer_urls(
    peer_store: &PeerStore,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(producer, request_timestamp)
        .ok_or_else(|| "directory_mirror_peer_unavailable".to_string())?;
    if descriptor.sequence() != descriptor_sequence
        || !descriptor.descriptor.policy.public_discovery
    {
        return Err("directory_mirror_descriptor_changed".to_string());
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_mirror_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_mirror_peer_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(endpoint, "/api/discovery/peer/directory/block-range")
        .map_err(|_| "directory_mirror_peer_invalid_endpoint".to_string())?;
    let object_url =
        commitment_peer_url(endpoint, "/api/discovery/peer/directory/descriptor-objects")
            .map_err(|_| "directory_mirror_peer_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

fn directory_replica_carrier_urls(
    peer_store: &PeerStore,
    carrier: &[u8; 32],
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(carrier, request_timestamp)
        .ok_or_else(|| "pinned_directory_peer_unavailable".to_string())?;
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_directory_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("pinned_directory_peer_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-block-range",
    )
    .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    let object_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-descriptor-objects",
    )
    .map_err(|_| "pinned_directory_peer_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

fn directory_mirror_recovery_carrier_urls(
    peer_store: &PeerStore,
    carrier: &[u8; 32],
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(carrier, request_timestamp)
        .ok_or_else(|| "directory_mirror_recovery_carrier_unavailable".to_string())?;
    if !descriptor.descriptor.policy.public_discovery {
        return Err("directory_mirror_recovery_carrier_not_public".to_string());
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_mirror_recovery_carrier_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_mirror_recovery_carrier_unsafe_endpoint".to_string());
    }
    let range_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-block-range",
    )
    .map_err(|_| "directory_mirror_recovery_carrier_invalid_endpoint".to_string())?;
    let object_url = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-descriptor-objects",
    )
    .map_err(|_| "directory_mirror_recovery_carrier_invalid_endpoint".to_string())?;
    Ok((range_url, object_url))
}

async fn request_directory_block_page(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    client: &reqwest::Client,
    range_url: reqwest::Url,
    from_height: u64,
    request_timestamp: u64,
) -> Result<DirectoryRangePage, String> {
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        from_height,
        OUTBOUND_BLOCKS_PER_PAGE,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::BlockRangeRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        from_height,
        limit: OUTBOUND_BLOCKS_PER_PAGE,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request)
        .map_err(|_| "directory_range_request_encode_failed".to_string())?;
    let signed_response = post_directory_frame(client, range_url, frame, "range").await?;
    let (blocks, has_more, remote_tip_height, remote_tip_hash) = verify_block_range_response(
        &signed_response,
        &request_id,
        producer,
        from_height,
        request_timestamp,
    )?;
    Ok(DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    })
}

async fn request_directory_replica_block_page(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
    range_url: reqwest::Url,
    from_height: u64,
    request_timestamp: u64,
) -> Result<DirectoryRangePage, String> {
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_replica_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer,
        from_height,
        OUTBOUND_BLOCKS_PER_PAGE,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer: *producer,
        from_height,
        limit: OUTBOUND_BLOCKS_PER_PAGE,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request)
        .map_err(|_| "directory_replica_range_request_encode_failed".to_string())?;
    let signed_response = post_directory_frame(client, range_url, frame, "replica_range").await?;
    let (blocks, has_more, remote_tip_height, remote_tip_hash) =
        verify_replica_block_range_response(
            &signed_response,
            &request_id,
            producer,
            carrier,
            from_height,
            request_timestamp,
        )?;
    Ok(DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    })
}

async fn hydrate_directory_descriptor_objects(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    client: &reqwest::Client,
    object_url: reqwest::Url,
    requester: &[u8; 32],
    blocks: &[DirectoryCommitmentBlockV1],
) -> Result<(Vec<SignedNodeDescriptor>, u32), String> {
    let descriptor_hashes = blocks
        .iter()
        .flat_map(|block| {
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash)
        })
        .collect::<Vec<_>>();
    let requests_made = directory_sync_request_count_for_objects(descriptor_hashes.len());
    let mut objects = Vec::with_capacity(descriptor_hashes.len());
    for hashes in descriptor_hashes.chunks(MAX_DIRECTORY_SYNC_OBJECTS_V1) {
        let request_timestamp = unix_now_secs();
        let mut request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut request_id);
        let signing_bytes = directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            hashes,
            &request_id,
            requester,
            request_timestamp,
        );
        let request = DirectorySyncMessage::DescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            descriptor_hashes: hashes.to_vec(),
            request_id,
            requester: *requester,
            request_timestamp,
            signature: identity.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&request)
            .map_err(|_| "directory_object_request_encode_failed".to_string())?;
        let response = post_directory_frame(client, object_url.clone(), frame, "objects").await?;
        let mut verified = verify_descriptor_objects_response(
            &response,
            &request_id,
            producer,
            hashes,
            request_timestamp,
        )?;
        objects.append(&mut verified);
    }
    Ok((objects, requests_made))
}

async fn hydrate_directory_replica_descriptor_objects(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
    object_url: reqwest::Url,
    requester: &[u8; 32],
    blocks: &[DirectoryCommitmentBlockV1],
) -> Result<(Vec<SignedNodeDescriptor>, u32), String> {
    let descriptor_hashes = blocks
        .iter()
        .flat_map(|block| {
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash)
        })
        .collect::<Vec<_>>();
    let requests_made = directory_sync_request_count_for_objects(descriptor_hashes.len());
    let mut objects = Vec::with_capacity(descriptor_hashes.len());
    for hashes in descriptor_hashes.chunks(MAX_DIRECTORY_SYNC_OBJECTS_V1) {
        let request_timestamp = unix_now_secs();
        let mut request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut request_id);
        let signing_bytes = directory_replica_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer,
            hashes,
            &request_id,
            requester,
            request_timestamp,
        );
        let request = DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: *producer,
            descriptor_hashes: hashes.to_vec(),
            request_id,
            requester: *requester,
            request_timestamp,
            signature: identity.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&request)
            .map_err(|_| "directory_replica_object_request_encode_failed".to_string())?;
        let response =
            post_directory_frame(client, object_url.clone(), frame, "replica_objects").await?;
        let mut verified = verify_replica_descriptor_objects_response(
            &response,
            &request_id,
            producer,
            carrier,
            hashes,
            request_timestamp,
        )?;
        objects.append(&mut verified);
    }
    Ok((objects, requests_made))
}

async fn import_directory_range_page(
    replica_store: Arc<DirectoryReplicaStore>,
    producer: [u8; 32],
    page: DirectoryRangePage,
    objects: Vec<SignedNodeDescriptor>,
    requests_made: u32,
) -> Result<DirectorySyncPullOutcome, String> {
    let DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    } = page;
    let import = tokio::task::spawn_blocking(move || {
        replica_store.import_verified_page(
            producer,
            &blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &signed_response,
            unix_now_secs(),
        )
    })
    .await
    .map_err(|_| "directory_replica_import_task_failed".to_string())?
    .map_err(|error| match error {
        DirectoryReplicaStoreError::Quarantined(_) => "producer_quarantined".to_string(),
        _ => "directory_replica_import_rejected".to_string(),
    })?;
    Ok(DirectorySyncPullOutcome {
        import,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        requests_made,
    })
}

async fn import_directory_mirror_range_page(
    replica_store: Arc<DirectoryReplicaStore>,
    producer: [u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    page: DirectoryRangePage,
    objects: Vec<SignedNodeDescriptor>,
    requests_made: u32,
) -> Result<DirectorySyncPullOutcome, String> {
    let DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    } = page;
    let import = tokio::task::spawn_blocking(move || {
        replica_store.import_verified_mirror_page(
            producer,
            descriptor_sequence,
            max_mirror_producers,
            &blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &signed_response,
            unix_now_secs(),
        )
    })
    .await
    .map_err(|_| "directory_mirror_import_task_failed".to_string())?
    .map_err(|error| match error {
        DirectoryReplicaStoreError::MirrorCapacity => "directory_mirror_capacity_full".to_string(),
        DirectoryReplicaStoreError::Quarantined(_) => {
            "directory_mirror_producer_quarantined".to_string()
        }
        _ => "directory_mirror_import_rejected".to_string(),
    })?;
    Ok(DirectorySyncPullOutcome {
        import,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        requests_made,
    })
}

async fn post_directory_frame(
    client: &reqwest::Client,
    url: reqwest::Url,
    frame: Vec<u8>,
    operation: &'static str,
) -> Result<Vec<u8>, String> {
    post_directory_frame_typed(client, url, frame)
        .await
        .map_err(|error| error.stable_reason(operation))
}

async fn post_directory_frame_typed(
    client: &reqwest::Client,
    url: reqwest::Url,
    frame: Vec<u8>,
) -> Result<Vec<u8>, DirectoryFramePostError> {
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|_| DirectoryFramePostError::Transport)?;
    if !response.status().is_success() {
        return Err(DirectoryFramePostError::HttpStatus(
            response.status().as_u16(),
        ));
    }
    read_bounded_http_response(response, MAX_DIRECTORY_SYNC_RESPONSE_BODY_BYTES)
        .await
        .map_err(DirectoryFramePostError::Response)
}

pub(crate) fn verify_block_range_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_from_height: u64,
    request_timestamp: u64,
) -> Result<
    (
        Vec<aeronyx_core::protocol::discovery::DirectoryCommitmentBlockV1>,
        bool,
        u64,
        [u8; 32],
    ),
    String,
> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_range_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_range_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_range_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("directory_range_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || responder != *expected_producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || blocks.len() > usize::from(OUTBOUND_BLOCKS_PER_PAGE)
        || blocks
            .first()
            .is_some_and(|block| block.header.height != expected_from_height)
        || blocks
            .iter()
            .any(|block| block.header.producer != *expected_producer)
    {
        return Err("directory_range_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_range_response_invalid_signature".to_string())?;
    Ok((blocks, has_more, tip_height, tip_hash))
}

pub(crate) fn verify_replica_block_range_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_carrier: &[u8; 32],
    expected_from_height: u64,
    request_timestamp: u64,
) -> Result<(Vec<DirectoryCommitmentBlockV1>, bool, u64, [u8; 32]), String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_replica_range_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_replica_range_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_replica_range_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("directory_replica_range_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || producer != *expected_producer
        || carrier != *expected_carrier
        || carrier == producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || blocks.len() > usize::from(OUTBOUND_BLOCKS_PER_PAGE)
        || blocks
            .first()
            .is_some_and(|block| block.header.height != expected_from_height)
        || blocks
            .iter()
            .any(|block| block.header.producer != *expected_producer)
    {
        return Err("directory_replica_range_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_replica_block_range_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&carrier)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_replica_range_response_invalid_signature".to_string())?;
    Ok((blocks, has_more, tip_height, tip_hash))
}

pub(crate) fn verify_descriptor_objects_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_hashes: &[[u8; 32]],
    request_timestamp: u64,
) -> Result<Vec<aeronyx_core::protocol::discovery::SignedNodeDescriptor>, String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_object_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_object_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_object_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::DescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature,
    } = message
    else {
        return Err("directory_object_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || responder != *expected_producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || descriptor_hashes != expected_hashes
        || objects.len() != expected_hashes.len()
    {
        return Err("directory_object_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_descriptor_objects_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &descriptor_hashes,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_object_response_invalid_signature".to_string())?;
    for (expected_hash, object) in expected_hashes.iter().zip(&objects) {
        let commitment = aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            object,
        )
        .map_err(|_| "directory_object_response_invalid_descriptor".to_string())?;
        if commitment.descriptor_hash != *expected_hash {
            return Err("directory_object_response_hash_mismatch".to_string());
        }
    }
    Ok(objects)
}

pub(crate) fn verify_replica_descriptor_objects_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_carrier: &[u8; 32],
    expected_hashes: &[[u8; 32]],
    request_timestamp: u64,
) -> Result<Vec<SignedNodeDescriptor>, String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_replica_object_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_replica_object_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_replica_object_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ReplicaDescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature,
    } = message
    else {
        return Err("directory_replica_object_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || producer != *expected_producer
        || carrier != *expected_carrier
        || carrier == producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || descriptor_hashes != expected_hashes
        || objects.len() != expected_hashes.len()
    {
        return Err("directory_replica_object_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_replica_descriptor_objects_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &descriptor_hashes,
    );
    IdentityPublicKey::from_bytes(&carrier)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_replica_object_response_invalid_signature".to_string())?;
    for (expected_hash, object) in expected_hashes.iter().zip(&objects) {
        let commitment = aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            object,
        )
        .map_err(|_| "directory_replica_object_response_invalid_descriptor".to_string())?;
        if commitment.descriptor_hash != *expected_hash {
            return Err("directory_replica_object_response_hash_mismatch".to_string());
        }
    }
    Ok(objects)
}

#[must_use]
const fn directory_sync_startup_delay_secs(local_node_id: &[u8; 32]) -> u64 {
    DIRECTORY_SYNC_STARTUP_DELAY_MIN_SECS
        + (local_node_id[0] as u64 % DIRECTORY_SYNC_STARTUP_DELAY_SPAN_SECS)
}

fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const TEST_NOW: u64 = 1_700_000_000;

    #[test]
    fn startup_delay_is_stable_bounded_and_identity_spread() {
        assert_eq!(directory_sync_startup_delay_secs(&[0u8; 32]), 5);
        assert_eq!(directory_sync_startup_delay_secs(&[10u8; 32]), 15);
        assert_eq!(directory_sync_startup_delay_secs(&[11u8; 32]), 5);
        assert_eq!(directory_sync_startup_delay_secs(&[255u8; 32]), 7);
    }

    #[test]
    fn concurrency_cap_remains_small_and_nonzero() {
        assert!((1..=4).contains(&DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS));
        assert!((1..120).contains(&DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS));
        assert_eq!(OUTBOUND_BLOCKS_PER_PAGE, MAX_DIRECTORY_SYNC_BLOCKS_V1);
    }

    #[test]
    fn repeated_failures_use_bounded_exponential_backoff() {
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 0), 0);
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 1), 0);
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 2), 120);
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 3), 360);
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 4), 840);
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 5), 1_800);
        assert_eq!(directory_sync_failure_backoff_delay_secs(120, 99), 1_800);
    }

    #[test]
    fn coordinator_restores_retry_state_for_configured_producers_only() {
        let temp = TempDir::new().unwrap();
        let local = Arc::new(IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap());
        let configured = IdentityKeyPair::from_bytes(&[0xd2; 32])
            .unwrap()
            .public_key_bytes();
        let retired = IdentityKeyPair::from_bytes(&[0xd3; 32])
            .unwrap()
            .public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            TEST_NOW,
        )
        .unwrap();
        for producer in [configured, retired] {
            store
                .persist_retry_failure(
                    producer,
                    2,
                    Some(TEST_NOW + 300),
                    TEST_NOW,
                    "directory_range_transport_failed",
                )
                .unwrap();
        }
        let store = Arc::new(store);
        let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
        assert_eq!(
            DirectoryReplicaSyncCoordinator::new_with_policy(
                vec![configured],
                120,
                Arc::clone(&store),
                Arc::clone(&runtime),
                Arc::new(PeerStore::new()),
                Arc::clone(&local),
                DirectoryReplicaSyncPolicy {
                    witness_min_verified: 0,
                    full_node_mirror_enabled: false,
                    full_node_mirror_max_producers: 32,
                },
            )
            .err(),
            Some("directory_observation_witness_threshold_invalid")
        );
        assert_eq!(
            DirectoryReplicaSyncCoordinator::new_with_policy(
                vec![configured],
                120,
                Arc::clone(&store),
                Arc::clone(&runtime),
                Arc::new(PeerStore::new()),
                Arc::clone(&local),
                DirectoryReplicaSyncPolicy {
                    witness_min_verified: 2,
                    full_node_mirror_enabled: false,
                    full_node_mirror_max_producers: 32,
                },
            )
            .err(),
            Some("directory_observation_witness_threshold_invalid")
        );
        let coordinator = DirectoryReplicaSyncCoordinator::new_with_policy(
            vec![configured],
            120,
            store,
            Arc::clone(&runtime),
            Arc::new(PeerStore::new()),
            local,
            DirectoryReplicaSyncPolicy {
                witness_min_verified: 1,
                full_node_mirror_enabled: false,
                full_node_mirror_max_producers: 32,
            },
        )
        .unwrap();

        assert_eq!(coordinator.restored_retry_states, 1);
        let restored = runtime.snapshot();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].producer, configured);
        assert_eq!(restored[0].consecutive_failures, 2);
        assert_eq!(restored[0].retry_not_before, Some(TEST_NOW + 300));
    }

    #[test]
    fn catch_up_budget_allows_small_pages_but_reserves_worst_case_headroom() {
        assert_eq!(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE, 18);
        assert_eq!(directory_sync_request_count_for_objects(0), 1);
        assert_eq!(directory_sync_request_count_for_objects(1), 2);
        assert_eq!(directory_sync_request_count_for_objects(16), 2);
        assert_eq!(directory_sync_request_count_for_objects(17), 3);
        assert_eq!(directory_sync_request_count_for_objects(256), 17);
        assert!(should_continue_directory_replica_catch_up(1, 2, true));
        assert!(should_continue_directory_replica_catch_up(4, 8, true));
        assert!(should_continue_directory_replica_catch_up(6, 12, true));
        assert!(!should_continue_directory_replica_catch_up(7, 14, true));
        assert!(!should_continue_directory_replica_catch_up(8, 16, true));
        assert!(should_continue_directory_replica_catch_up(1, 12, true));
        assert!(!should_continue_directory_replica_catch_up(1, 13, true));
        assert!(!should_continue_directory_replica_catch_up(1, 2, false));
    }

    #[test]
    fn incomplete_rounds_use_bounded_catch_up_cadence() {
        let configured = Duration::from_secs(120);
        assert_eq!(
            directory_sync_next_round_delay(configured, true),
            configured
        );
        assert_eq!(
            directory_sync_next_round_delay(configured, false),
            Duration::from_secs(DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS)
        );
        let already_fast = Duration::from_secs(30);
        assert_eq!(
            directory_sync_next_round_delay(already_fast, false),
            already_fast
        );
    }

    #[test]
    fn carrier_fallback_is_limited_to_availability_and_admission_failures() {
        for reason in [
            "pinned_directory_peer_unavailable",
            "pinned_directory_peer_missing_endpoint",
            "directory_range_transport_failed",
            "directory_range_http_status_403",
            "directory_range_http_status_404",
            "directory_range_http_status_408",
            "directory_range_http_status_429",
            "directory_range_http_status_500",
            "directory_replica_range_http_status_503",
        ] {
            assert!(directory_sync_failure_allows_carrier_fallback(reason));
        }
        for reason in [
            "directory_range_response_noncanonical",
            "directory_range_response_invalid_signature",
            "directory_range_response_contract_mismatch",
            "directory_object_response_hash_mismatch",
            "directory_range_http_status_400",
            "directory_range_http_status_401",
            "directory_range_http_status_409",
        ] {
            assert!(!directory_sync_failure_allows_carrier_fallback(reason));
        }
    }

    #[test]
    fn mirror_recovery_is_bounded_and_rejects_security_failures() {
        assert_eq!(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE, 2);
        let recovery_carrier_count =
            u64::try_from(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE)
                .expect("bounded recovery carrier count fits u64");
        assert!(
            DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS
                * (1 + recovery_carrier_count)
                < DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS
        );
        for reason in [
            "directory_range_transport_failed",
            "directory_objects_transport_failed",
            "directory_range_http_status_404",
            "directory_replica_range_http_status_429",
            "directory_replica_objects_http_status_503",
            "directory_mirror_recovery_carrier_unavailable",
        ] {
            assert!(directory_mirror_failure_allows_recovery(reason));
        }
        for reason in [
            "directory_mirror_descriptor_changed",
            "directory_range_response_noncanonical",
            "directory_range_response_invalid_signature",
            "directory_replica_range_response_contract_mismatch",
            "directory_replica_range_response_invalid_signature",
            "directory_replica_object_response_hash_mismatch",
            "directory_mirror_import_rejected",
            "directory_range_http_status_400",
            "directory_range_http_status_401",
            "directory_range_http_status_409",
        ] {
            assert!(!directory_mirror_failure_allows_recovery(reason));
        }
    }

    #[test]
    fn mirror_catch_up_stops_at_page_request_or_convergence_boundaries() {
        // [MIRROR-CATCHUP 2026-07-24 by Codex] Permissionless work must remain
        // strictly below the pinned producer budget.
        assert!(DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND < DIRECTORY_SYNC_MAX_PAGES_PER_ROUND);
        assert!(
            DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND
                < DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND
        );
        assert!(
            DIRECTORY_MIRROR_MAX_REQUESTS_PER_PAGE
                <= DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND
        );
        assert!(should_continue_directory_mirror_catch_up(1, 1, true));
        assert!(should_continue_directory_mirror_catch_up(1, 5, true));
        assert!(!should_continue_directory_mirror_catch_up(1, 1, false));
        assert!(!should_continue_directory_mirror_catch_up(
            DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND,
            1,
            true
        ));
        assert!(!should_continue_directory_mirror_catch_up(
            1,
            6,
            true
        ));
    }

    #[test]
    fn mirror_recovery_carrier_selection_excludes_participants_and_is_deterministic() {
        let now = unix_now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
        let requester = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
        let store = PeerStore::new();
        let mut expected_excluded = HashSet::new();
        expected_excluded.insert(producer.public_key_bytes());
        expected_excluded.insert(requester.public_key_bytes());

        for seed in [0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7] {
            let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
            let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
                identity.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now + 600,
                "mirror-recovery-test",
            );
            descriptor.policy.public_discovery = true;
            descriptor.public_endpoint = Some(format!("http://8.8.8.{seed}:8422"));
            store
                .upsert_verified_from_source(
                    SignedNodeDescriptor::sign(descriptor, &identity).unwrap(),
                    now,
                    "directory_mirror_recovery_test",
                )
                .unwrap();
        }

        let selected = directory_mirror_recovery_carriers(
            &store,
            &producer.public_key_bytes(),
            &requester.public_key_bytes(),
            now,
        );
        assert_eq!(selected.len(), DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE);
        assert!(selected
            .iter()
            .all(|candidate| !expected_excluded.contains(candidate)));
        assert_eq!(
            selected,
            directory_mirror_recovery_carriers(
                &store,
                &producer.public_key_bytes(),
                &requester.public_key_bytes(),
                now,
            )
        );
    }

    #[test]
    fn carrier_range_response_verification_binds_producer_carrier_and_signature() {
        let producer = IdentityKeyPair::from_bytes(&[0xf1; 32]).unwrap();
        let carrier = IdentityKeyPair::from_bytes(&[0xf2; 32]).unwrap();
        let other = IdentityKeyPair::from_bytes(&[0xf3; 32]).unwrap();
        let request_id = [0xf4; 16];
        let now = unix_now_secs();
        let blocks = Vec::new();
        let signing_bytes = directory_replica_block_range_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            now,
            &blocks,
            false,
            0,
            &[0u8; 32],
        );
        let response = DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            producer: producer.public_key_bytes(),
            carrier: carrier.public_key_bytes(),
            response_timestamp: now,
            blocks,
            has_more: false,
            tip_height: 0,
            tip_hash: [0u8; 32],
            signature: carrier.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&response).unwrap();
        assert_eq!(
            verify_replica_block_range_response(
                &frame,
                &request_id,
                &producer.public_key_bytes(),
                &carrier.public_key_bytes(),
                1,
                now,
            )
            .unwrap(),
            (Vec::new(), false, 0, [0u8; 32])
        );
        assert_eq!(
            verify_replica_block_range_response(
                &frame,
                &request_id,
                &other.public_key_bytes(),
                &carrier.public_key_bytes(),
                1,
                now,
            )
            .unwrap_err(),
            "directory_replica_range_response_contract_mismatch"
        );
        assert_eq!(
            verify_replica_block_range_response(
                &frame,
                &request_id,
                &producer.public_key_bytes(),
                &other.public_key_bytes(),
                1,
                now,
            )
            .unwrap_err(),
            "directory_replica_range_response_contract_mismatch"
        );

        let mut tampered = response;
        let DirectorySyncMessage::ReplicaBlockRangeResponseV1 { signature, .. } = &mut tampered
        else {
            unreachable!();
        };
        signature[0] ^= 1;
        assert_eq!(
            verify_replica_block_range_response(
                &encode_directory_sync_message(&tampered).unwrap(),
                &request_id,
                &producer.public_key_bytes(),
                &carrier.public_key_bytes(),
                1,
                now,
            )
            .unwrap_err(),
            "directory_replica_range_response_invalid_signature"
        );
    }

    #[test]
    fn policy_anchor_response_verification_binds_the_complete_statement() {
        let observer = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
        let other_witness = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
        let request_id = [0xa4; 16];
        let policy_epoch = 7;
        let policy_digest = [0xa5; 32];
        let now = unix_now_secs();
        let response = |outcome: u8| {
            let signing_bytes = directory_policy_anchor_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &observer.public_key_bytes(),
                policy_epoch,
                &policy_digest,
                &witness.public_key_bytes(),
                now,
                outcome,
            );
            DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                observer: observer.public_key_bytes(),
                policy_epoch,
                policy_digest,
                responder: witness.public_key_bytes(),
                response_timestamp: now,
                outcome,
                signature: witness.sign(&signing_bytes),
            }
        };

        let accepted = response(DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1);
        let frame = encode_directory_sync_message(&accepted).unwrap();
        assert_eq!(
            verify_observation_policy_anchor_response(
                &frame,
                &request_id,
                &observer.public_key_bytes(),
                &witness.public_key_bytes(),
                now,
                policy_epoch,
                &policy_digest,
            )
            .unwrap(),
            accepted
        );
        assert_eq!(
            verify_observation_policy_anchor_response(
                &frame,
                &request_id,
                &observer.public_key_bytes(),
                &other_witness.public_key_bytes(),
                now,
                policy_epoch,
                &policy_digest,
            )
            .unwrap_err(),
            "observation_policy_anchor_response_contract_mismatch"
        );
        assert_eq!(
            verify_observation_policy_anchor_response(
                &frame,
                &request_id,
                &observer.public_key_bytes(),
                &witness.public_key_bytes(),
                now,
                policy_epoch,
                &[0xff; 32],
            )
            .unwrap_err(),
            "observation_policy_anchor_response_contract_mismatch"
        );

        let rollback =
            encode_directory_sync_message(&response(DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1)).unwrap();
        assert_eq!(
            verify_observation_policy_anchor_response(
                &rollback,
                &request_id,
                &observer.public_key_bytes(),
                &witness.public_key_bytes(),
                now,
                policy_epoch,
                &policy_digest,
            )
            .unwrap_err(),
            "observation_policy_anchor_rollback"
        );

        let mut tampered = response(DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1);
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 { signature, .. } =
            &mut tampered
        else {
            unreachable!();
        };
        signature[0] ^= 1;
        assert_eq!(
            verify_observation_policy_anchor_response(
                &encode_directory_sync_message(&tampered).unwrap(),
                &request_id,
                &observer.public_key_bytes(),
                &witness.public_key_bytes(),
                now,
                policy_epoch,
                &policy_digest,
            )
            .unwrap_err(),
            "observation_policy_anchor_response_invalid_signature"
        );
    }

    #[test]
    fn checkpoint_requires_exact_authenticated_remote_tip() {
        let complete = DirectorySyncPullOutcome {
            import: DirectoryReplicaImportReport {
                blocks_inserted: 1,
                blocks_already_present: 0,
                commitments_inserted: 4,
                descriptor_equivocations: 0,
                tip_height: 9,
                tip_hash: [0x41; 32],
            },
            has_more: false,
            remote_tip_height: 9,
            remote_tip_hash: [0x41; 32],
            requests_made: 2,
        };
        assert!(directory_sync_outcome_is_checkpoint_complete(&complete));

        let mut catching_up = complete;
        catching_up.has_more = true;
        assert!(!directory_sync_outcome_is_checkpoint_complete(&catching_up));
        let mut stale_height = complete;
        stale_height.remote_tip_height = 10;
        assert!(!directory_sync_outcome_is_checkpoint_complete(
            &stale_height
        ));
        let mut wrong_hash = complete;
        wrong_hash.remote_tip_hash = [0x42; 32];
        assert!(!directory_sync_outcome_is_checkpoint_complete(&wrong_hash));
    }

    #[test]
    fn observation_witness_response_verification_is_exact_and_fail_closed() {
        let observer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
        let request_id = [0xe3; 16];
        let checkpoint_hash = [0xe4; 32];
        let now = unix_now_secs();
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            7,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            now,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: 7,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp: now,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&response).unwrap();
        assert_eq!(
            verify_observation_witness_response(
                &frame,
                &request_id,
                &observer.public_key_bytes(),
                &witness.public_key_bytes(),
                now,
                7,
                &checkpoint_hash,
            )
            .unwrap(),
            response
        );

        let unavailable_signing = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            7,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            now,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
        );
        let unavailable = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: 7,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp: now,
            outcome: DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
            signature: witness.sign(&unavailable_signing),
        };
        assert_eq!(
            verify_observation_witness_response(
                &encode_directory_sync_message(&unavailable).unwrap(),
                &request_id,
                &observer.public_key_bytes(),
                &witness.public_key_bytes(),
                now,
                7,
                &checkpoint_hash,
            )
            .unwrap_err(),
            "observation_witness_evidence_unavailable"
        );

        let mut tampered = frame;
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        assert!(verify_observation_witness_response(
            &tampered,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            7,
            &checkpoint_hash,
        )
        .is_err());
    }

    #[test]
    fn witness_capability_cache_is_scoped_to_authenticated_descriptor_sequence() {
        let cache = DirectoryWitnessCapabilityCache::default();
        let witness = [0x91; 32];

        assert!(cache.should_attempt(&witness, 7));
        cache.record_unsupported(witness, 7);
        assert!(!cache.should_attempt(&witness, 7));
        assert!(cache.should_attempt(&witness, 8));

        cache.record_supported(&witness);
        assert!(cache.should_attempt(&witness, 7));
    }

    #[test]
    fn witness_capability_http_statuses_are_narrow_and_typed() {
        for status in [404, 405, 501] {
            assert!(DirectoryFramePostError::HttpStatus(status).witness_capability_unavailable());
        }
        for status in [400, 401, 403, 409, 429, 500, 503] {
            assert!(!DirectoryFramePostError::HttpStatus(status).witness_capability_unavailable());
        }
        assert_eq!(
            DirectoryFramePostError::HttpStatus(404).stable_reason("range"),
            "directory_range_http_status_404"
        );
        assert_eq!(
            DirectoryFramePostError::Response(BoundedHttpResponseError::TooLarge)
                .stable_reason("objects"),
            "directory_objects_response_too_large"
        );
    }
}
