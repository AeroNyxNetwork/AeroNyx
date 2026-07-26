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
//! - Prioritizes fresh routeable recovery carriers and uses signed region hints
//!   only as a best-effort same-tier fault-domain diversity signal.
//! - Prefers an explicit signed Directory Mirror carrier capability while
//!   retaining separately measured unadvertised compatibility fallback during
//!   staged fleet rollout.
//! - Runs a bounded operator-only carrier smoke against one retained anchor
//!   without direct-producer fallback, replica import, or authority mutation.
//! - Allows pinned producers to recover through bounded explicitly advertised
//!   permissionless carriers after direct and pinned-carrier availability
//!   failures, without granting those carriers authority.
//! - Runs an operator-only cold-bootstrap smoke that replays a bounded
//!   multi-page producer-signed genesis prefix through rotating explicit
//!   carriers into a fresh in-memory store.
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
//! 15. Prefer fresh routeability evidence, rotate equally healthy candidates,
//!     and avoid repeating a signed region hint when an equally healthy
//!     alternative exists. Region hints never prove operator or ASN diversity.
//! 16. On explicit operator request, prove bounded multi-page cold recovery in
//!     an isolated in-memory replica. Rotate the first carrier between pages,
//!     retry only availability failures, and preserve an already verified
//!     multi-page prefix if a later page becomes unavailable. Stop closed on
//!     cryptographic or import failures, then discard the replica without
//!     touching the live store.
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
//! - A carrier availability failure may select another authenticated carrier.
//!   A signature, chain, commitment, noncanonical, or import failure must stop
//!   the isolated recovery immediately; never use failover to hide bad evidence.
//! - Once at least two pages form an audited producer-signed genesis prefix, a
//!   later availability failure may end the smoke as a verified partial-prefix
//!   result. It must never claim the observed remote tip was reached.
//!
//! ## Last Modified
//! `v0.22.1-CarrierPartialPrefix` - Preserved and fully audited a verified
//! multi-page prefix when a later carrier page becomes unavailable.
//! `v0.22.0-CarrierMultiPageRecovery` - Extended isolated cold bootstrap to a
//! bounded multi-page prefix with carrier rotation, availability-only failover,
//! conservative request accounting, and a complete post-import store audit.
//! `v0.21.0-CarrierColdBootstrap` - Added bounded explicit-carrier recovery for
//! pinned producers and an isolated carrier-assisted cold-bootstrap release gate.
//! `v0.20.0-ReadOnlyCarrierSmoke` - Added explicit-carrier-only retained-anchor
//! verification for release gates and post-upgrade operator checks.
//! `v0.19.0-SignedMirrorCarrierSelection` - Preferred signed carrier
//! advertisements and separated unadvertised compatibility fallback telemetry.
//! `v0.18.0-MirrorCarrierCapabilityMemory` - Added bounded descriptor-sequence-scoped
//! negative capability memory for recovery carriers.
//! `v0.17.0-MirrorSourceDiversity` - Added routeability/freshness-aware carrier
//! ordering, best-effort signed-region diversity, aggregate selection data, and
//! explicit proxy bypass for authenticated node-to-node synchronization.
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

use std::collections::{HashMap, HashSet, VecDeque};
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
    NodeCapability, SignedNodeDescriptor, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1, DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1, DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
    DIRECTORY_POLICY_ANCHOR_CONFLICT_V1, DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
    DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_SYNC_BLOCKS_V1, MAX_DIRECTORY_SYNC_OBJECTS_V1,
};
use futures::{stream, StreamExt};
use parking_lot::Mutex;
use rand::RngCore;
use serde::Serialize;
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
/// Peer protocol errors are fixed ASCII codes. Keep non-success reads tiny so
/// an unauthenticated endpoint cannot turn diagnostics into a memory sink.
const MAX_DIRECTORY_SYNC_ERROR_BODY_BYTES: usize = 128;
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
/// [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] Bound operator-pinned
/// carrier attempts before permissionless explicit carriers are considered.
/// Direct + two pinned + two explicit carrier range attempts, followed by one
/// successful worst-case hydration page, remains below the 30-request budget.
const DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE: usize = 2;
/// One direct failure and one unsuccessful recovery carrier can precede the
/// existing worst-case successful carrier page.
const DIRECTORY_MIRROR_MAX_REQUESTS_PER_PAGE: u32 = DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE + 1;
/// Keep carrier choice stable within a round while avoiding permanent affinity.
const DIRECTORY_MIRROR_RECOVERY_ROTATION_SECS: u64 = 5 * 60;
/// Recently issued descriptors are preferred within the same routeability tier.
const DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS: u64 = 10 * 60;
/// Valid but older descriptors remain fallback candidates after fresher peers.
const DIRECTORY_MIRROR_RECOVERY_AGING_DESCRIPTOR_SECS: u64 = 30 * 60;
/// [MIRROR-CAPABILITY 2026-07-24 by Codex] Bound process-local negative
/// capability memory under permissionless descriptor churn. A newer signed
/// descriptor sequence is always eligible without waiting for a timer.
const DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES: usize = 256;
/// A manual smoke remains bounded even when the mirror registry is full.
const DIRECTORY_MIRROR_CARRIER_SMOKE_MAX_PRODUCERS: usize = 2;
/// An isolated smoke checks only a bounded number of configured producers.
const DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PRODUCERS: usize = 2;
/// [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Three pages prove
/// continuation beyond genesis without turning an operator smoke into an
/// unbounded full-chain download.
const DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES: u32 = 3;
/// Failed carrier attempts are charged at the complete worst-case page cost.
/// The smoke shares the existing pinned-producer round ceiling and records the
/// exact range/object requests consumed before a failed carrier is replaced.
const DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET: u32 =
    DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryCarrierRecoveryDisposition {
    RetryAvailabilityFailure,
    StopClosed,
}

fn directory_carrier_recovery_disposition(reason: &str) -> DirectoryCarrierRecoveryDisposition {
    if directory_mirror_failure_allows_recovery(reason) {
        DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure
    } else {
        DirectoryCarrierRecoveryDisposition::StopClosed
    }
}

/// Internal failure carrying conservative network-request accounting.
///
/// [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] This is intentionally
/// private and never serialized: peer-controlled reasons remain mapped to
/// stable privacy-safe buckets before leaving the coordinator.
#[derive(Debug)]
struct DirectoryCarrierPullFailure {
    reason: String,
    requests_made: u32,
}

impl DirectoryCarrierPullFailure {
    fn new(reason: String, requests_made: u32) -> Self {
        Self {
            reason,
            requests_made,
        }
    }
}

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

/// Internal carrier candidate used only during one bounded recovery selection.
///
/// The signed region is an untrusted availability hint. It must never be
/// interpreted as proof of a distinct operator, ASN, jurisdiction, or identity.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryMirrorRecoveryCarrierCandidate {
    node_id: [u8; 32],
    descriptor_sequence: u64,
    explicitly_advertised: bool,
    routeable: bool,
    freshness_rank: u8,
    rotation_rank: usize,
    signed_region_hint: Option<String>,
}

impl DirectoryMirrorRecoveryCarrierCandidate {
    const fn availability_tier(&self) -> (u8, u8, u8) {
        // [MIRROR-CAPABILITY 2026-07-24 by Codex] Local reachability remains
        // stronger than self-reported metadata. Within the same reachability
        // tier, an authenticated capability is preferred before freshness.
        (
            (!self.routeable) as u8,
            (!self.explicitly_advertised) as u8,
            self.freshness_rank,
        )
    }
}

/// Descriptor-bound carrier selected for one authenticated recovery attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectoryMirrorRecoveryCarrier {
    node_id: [u8; 32],
    descriptor_sequence: u64,
}

/// Privacy-safe result of one bounded recovery-carrier selection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct DirectoryMirrorRecoveryCarrierSelection {
    carriers: Vec<DirectoryMirrorRecoveryCarrier>,
    candidate_count: u64,
    routeable_candidate_count: u64,
    explicitly_advertised_candidate_count: u64,
    unadvertised_compatibility_candidate_count: u64,
    capability_cached_unavailable_count: u64,
    selected_routeable_count: u64,
    selected_explicitly_advertised_count: u64,
    selected_unadvertised_compatibility_count: u64,
    selected_region_hint_count: u64,
    distinct_selected_region_hint_count: u64,
}

/// Privacy-safe result of one read-only signed carrier verification.
///
/// [MIRROR-CARRIER-SMOKE 2026-07-25 by Codex] This contract intentionally
/// omits producer/carrier identities, endpoints, region hints, hashes, block
/// timestamps, descriptor contents, and route order. It is safe to show in
/// local operator tooling but must remain off the public discovery listener.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct DirectoryMirrorCarrierSmokeReport {
    pub success: bool,
    pub status: &'static str,
    pub contract_version: &'static str,
    pub source: &'static str,
    pub scope: &'static str,
    pub retained_producers: u64,
    pub eligible_retained_producers: u64,
    pub explicit_carrier_candidates: u64,
    pub selected_routeable_carriers: u64,
    pub attempted_carriers: u64,
    pub verified_blocks: u64,
    pub verified_descriptor_objects: u64,
    pub carrier_signature_verified: bool,
    pub producer_evidence_verified: bool,
    pub local_anchor_verified: bool,
    pub storage_effect: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
    pub privacy_invariant: &'static str,
    pub privacy_boundary: &'static str,
}

impl DirectoryMirrorCarrierSmokeReport {
    fn pending() -> Self {
        Self {
            success: false,
            status: "unavailable",
            contract_version: "directory_mirror_carrier_smoke.v1",
            source: "rust_local_operator_smoke",
            scope: "local_or_vpn_operator_api_only",
            retained_producers: 0,
            eligible_retained_producers: 0,
            explicit_carrier_candidates: 0,
            selected_routeable_carriers: 0,
            attempted_carriers: 0,
            verified_blocks: 0,
            verified_descriptor_objects: 0,
            carrier_signature_verified: false,
            producer_evidence_verified: false,
            local_anchor_verified: false,
            storage_effect: "none_read_only",
            failure_reason: None,
            retry_after_seconds: None,
            privacy_invariant:
                "carriers transport signed public protocol evidence but gain no authority",
            privacy_boundary:
                "aggregate verification status only; no producer or carrier identities, endpoints, regions, hashes, descriptors, routes, selected hops, payloads, client IPs, destinations, DNS contents, Memory Chain records, private keys, wallet traffic, or social graph metadata",
        }
    }

    pub(crate) fn unavailable(reason: &'static str) -> Self {
        let mut report = Self::pending();
        report.failure_reason = Some(reason);
        report
    }

    pub(crate) fn busy() -> Self {
        let mut report = Self::unavailable("smoke_in_progress");
        report.status = "busy";
        report
    }

    pub(crate) fn cooldown(retry_after_seconds: u64) -> Self {
        let mut report = Self::unavailable("smoke_cooldown");
        report.status = "cooldown";
        report.retry_after_seconds = Some(retry_after_seconds);
        report
    }
}

/// Aggregate result of replaying a pinned producer's signed genesis prefix
/// through explicit public carriers into a fresh in-memory replica.
///
/// [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] The report proves that a
/// node with no retained producer state can establish and continue a bounded
/// producer-signed prefix without contacting that producer. It exposes no
/// selected identities, endpoints, hashes, descriptors, routes, payloads, or
/// user metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct DirectoryCarrierColdBootstrapSmokeReport {
    pub success: bool,
    pub status: &'static str,
    pub contract_version: &'static str,
    pub source: &'static str,
    pub scope: &'static str,
    pub configured_producers: u64,
    pub eligible_producers: u64,
    pub explicit_carrier_candidates: u64,
    pub selected_routeable_carriers: u64,
    pub attempted_carriers: u64,
    pub availability_failovers: u64,
    pub distinct_successful_carriers: u64,
    pub pages_imported: u64,
    pub requests_used: u64,
    pub request_budget: u64,
    pub imported_blocks: u64,
    pub imported_commitments: u64,
    pub bootstrapped_tip_height: u64,
    pub multi_page_prefix_verified: bool,
    pub reached_observed_remote_tip: bool,
    pub carrier_signature_verified: bool,
    pub producer_chain_verified: bool,
    pub genesis_anchor_verified: bool,
    pub isolated_store_audit_verified: bool,
    pub live_store_effect: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
    pub authority_boundary: &'static str,
    pub privacy_boundary: &'static str,
}

impl DirectoryCarrierColdBootstrapSmokeReport {
    fn pending(configured_producers: usize) -> Self {
        Self {
            success: false,
            status: "unavailable",
            contract_version: "directory_carrier_cold_bootstrap_smoke.v1",
            source: "rust_isolated_memory_replica_smoke",
            scope: "local_or_vpn_operator_api_only",
            configured_producers: u64::try_from(configured_producers).unwrap_or(u64::MAX),
            eligible_producers: 0,
            explicit_carrier_candidates: 0,
            selected_routeable_carriers: 0,
            attempted_carriers: 0,
            availability_failovers: 0,
            distinct_successful_carriers: 0,
            pages_imported: 0,
            requests_used: 0,
            request_budget: u64::from(
                DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET,
            ),
            imported_blocks: 0,
            imported_commitments: 0,
            bootstrapped_tip_height: 0,
            multi_page_prefix_verified: false,
            reached_observed_remote_tip: false,
            carrier_signature_verified: false,
            producer_chain_verified: false,
            genesis_anchor_verified: false,
            isolated_store_audit_verified: false,
            live_store_effect: "none_isolated_memory_store_only",
            failure_reason: None,
            retry_after_seconds: None,
            authority_boundary:
                "operator_pins_the_producer_identity; carriers_transport_but_never_author_blocks",
            privacy_boundary:
                "aggregate cold-bootstrap verification only; no producer or carrier identities, endpoints, hashes, descriptors, routes, selected hops, payloads, client IPs, destinations, DNS contents, Memory Chain records, private keys, wallet traffic, or social graph metadata",
        }
    }

    pub(crate) fn unavailable(configured_producers: usize, reason: &'static str) -> Self {
        let mut report = Self::pending(configured_producers);
        report.failure_reason = Some(reason);
        report
    }

    pub(crate) fn busy(configured_producers: usize) -> Self {
        let mut report = Self::unavailable(configured_producers, "smoke_in_progress");
        report.status = "busy";
        report
    }

    pub(crate) fn cooldown(configured_producers: usize, retry_after_seconds: u64) -> Self {
        let mut report = Self::unavailable(configured_producers, "smoke_cooldown");
        report.status = "cooldown";
        report.retry_after_seconds = Some(retry_after_seconds);
        report
    }
}

struct DirectoryMirrorCarrierSmokeAttemptContext<'a> {
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &'a PeerStore,
    identity: &'a IdentityKeyPair,
    client: &'a reqwest::Client,
    producer: [u8; 32],
    retained_tip_height: u64,
    requester: [u8; 32],
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

/// Whether the isolated carrier cold-bootstrap smoke may request another page.
///
/// [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Reserve a complete
/// worst-case page before continuing. The attempt loop applies the same check
/// before every carrier, so an availability failure can never overrun the
/// operator smoke budget.
const fn should_continue_directory_carrier_cold_bootstrap(
    pages_completed: u32,
    requests_used: u32,
    has_more: bool,
) -> bool {
    has_more
        && pages_completed < DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES
        && requests_used.saturating_add(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
            <= DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET
}

/// Whether the isolated store contains enough pages to prove multi-page
/// third-party recovery after a later availability-only failure.
const fn directory_carrier_cold_bootstrap_prefix_ready(pages_completed: u32) -> bool {
    pages_completed >= 2
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

/// Bounded negative capability cache for optional mirror-carrier endpoints.
///
/// [MIRROR-CAPABILITY 2026-07-24 by Codex] Only explicit endpoint absence
/// (`404`, `405`, or `501`) is cached. Transport failures, admission pressure,
/// and every cryptographic or canonical verification failure remain uncached.
/// Entries are bound to the exact authenticated descriptor sequence so a
/// software upgrade becomes immediately probeable. The FIFO is process-local,
/// bounded, and never exported as peer identity or reputation.
#[derive(Debug, Default)]
struct DirectoryMirrorCarrierCapabilityCache {
    state: Mutex<DirectoryMirrorCarrierCapabilityCacheState>,
}

#[derive(Debug, Default)]
struct DirectoryMirrorCarrierCapabilityCacheState {
    unsupported_descriptor_sequences: HashMap<[u8; 32], u64>,
    insertion_order: VecDeque<([u8; 32], u64)>,
}

impl DirectoryMirrorCarrierCapabilityCache {
    fn should_attempt(&self, carrier: &[u8; 32], descriptor_sequence: u64) -> bool {
        match self
            .state
            .lock()
            .unsupported_descriptor_sequences
            .get(carrier)
        {
            Some(unsupported_sequence) => *unsupported_sequence != descriptor_sequence,
            None => true,
        }
    }

    fn record_unsupported(&self, carrier: [u8; 32], descriptor_sequence: u64) {
        let mut state = self.state.lock();
        state
            .insertion_order
            .retain(|(existing, _)| *existing != carrier);
        if !state
            .unsupported_descriptor_sequences
            .contains_key(&carrier)
        {
            while state.unsupported_descriptor_sequences.len()
                >= DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES
            {
                let Some((oldest, oldest_sequence)) = state.insertion_order.pop_front() else {
                    state.unsupported_descriptor_sequences.clear();
                    break;
                };
                if state
                    .unsupported_descriptor_sequences
                    .get(&oldest)
                    .is_some_and(|current| *current == oldest_sequence)
                {
                    state.unsupported_descriptor_sequences.remove(&oldest);
                }
            }
        }
        state
            .unsupported_descriptor_sequences
            .insert(carrier, descriptor_sequence);
        state
            .insertion_order
            .push_back((carrier, descriptor_sequence));
    }

    fn record_supported(&self, carrier: &[u8; 32]) {
        let mut state = self.state.lock();
        state.unsupported_descriptor_sequences.remove(carrier);
        state
            .insertion_order
            .retain(|(existing, _)| existing != carrier);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.state.lock().unsupported_descriptor_sequences.len()
    }
}

/// Typed result boundary for untrusted peer HTTP exchange.
///
/// Keeping the status code typed until the caller applies operation-specific
/// policy prevents string parsing from becoming part of capability negotiation.
/// The type deliberately carries no URL, response body, peer identity, or
/// request material because failures can flow into operator telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryPeerErrorCode {
    ReplicaNotFound,
    ReplicaRangeNotRetained,
    MirrorReplicaNotRetained,
    ReplicaObjectNotFound,
}

impl DirectoryPeerErrorCode {
    fn parse(body: &[u8]) -> Option<Self> {
        match body {
            b"replica_not_found" => Some(Self::ReplicaNotFound),
            b"replica_range_not_retained" => Some(Self::ReplicaRangeNotRetained),
            b"mirror_replica_not_retained" => Some(Self::MirrorReplicaNotRetained),
            b"replica_object_not_found" => Some(Self::ReplicaObjectNotFound),
            _ => None,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::ReplicaNotFound => "replica_not_found",
            Self::ReplicaRangeNotRetained => "replica_range_not_retained",
            Self::MirrorReplicaNotRetained => "mirror_replica_not_retained",
            Self::ReplicaObjectNotFound => "replica_object_not_found",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryFramePostError {
    Transport,
    HttpStatus {
        status: u16,
        peer_code: Option<DirectoryPeerErrorCode>,
    },
    Response(BoundedHttpResponseError),
}

impl DirectoryFramePostError {
    const fn witness_capability_unavailable(self) -> bool {
        matches!(
            self,
            Self::HttpStatus {
                status: 404 | 405 | 501,
                peer_code: None
            }
        )
    }

    fn stable_reason(self, operation: &str) -> String {
        match self {
            Self::Transport => format!("directory_{operation}_transport_failed"),
            Self::HttpStatus {
                peer_code: Some(peer_code),
                ..
            } => {
                format!("directory_{operation}_peer_{}", peer_code.as_str())
            }
            Self::HttpStatus {
                status,
                peer_code: None,
            } => {
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
    mirror_carrier_capabilities: DirectoryMirrorCarrierCapabilityCache,
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
            // [MIRROR-DIVERSITY 2026-07-24 by Codex] Directory Sync endpoints
            // are authenticated direct node relationships. Inheriting host
            // HTTP(S)_PROXY settings would add an unreviewed metadata observer
            // and common failure domain despite the signed frame boundary.
            .no_proxy()
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
            mirror_carrier_capabilities: DirectoryMirrorCarrierCapabilityCache::default(),
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
                    self.runtime.as_ref(),
                    self.peer_store.as_ref(),
                    &self.mirror_carrier_capabilities,
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
                &self.mirror_carrier_capabilities,
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
                    DirectoryFramePostError::HttpStatus { status, .. } => status,
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
    runtime: &DirectoryReplicaSyncRuntime,
    peer_store: &PeerStore,
    carrier_capabilities: &DirectoryMirrorCarrierCapabilityCache,
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
            let carrier_selection = directory_mirror_recovery_carriers(
                peer_store,
                carrier_capabilities,
                producer,
                &identity.public_key_bytes(),
                unix_now_secs(),
            );
            // [MIRROR-DIVERSITY 2026-07-24 by Codex] Persist only aggregate
            // selection properties. Carrier identities, endpoints, regions,
            // producer identities, and route order never enter telemetry.
            runtime.record_full_node_mirror_carrier_selection(
                carrier_selection.candidate_count,
                carrier_selection.routeable_candidate_count,
                carrier_selection.explicitly_advertised_candidate_count,
                carrier_selection.unadvertised_compatibility_candidate_count,
                carrier_selection.capability_cached_unavailable_count,
                u64::try_from(carrier_selection.carriers.len()).unwrap_or(u64::MAX),
                carrier_selection.selected_routeable_count,
                carrier_selection.selected_explicitly_advertised_count,
                carrier_selection.selected_unadvertised_compatibility_count,
                carrier_selection.selected_region_hint_count,
                carrier_selection.distinct_selected_region_hint_count,
            );
            for carrier in carrier_selection.carriers {
                match pull_directory_chain_mirror_page_via_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    descriptor_sequence,
                    max_mirror_producers,
                    &carrier.node_id,
                    carrier.descriptor_sequence,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        carrier_capabilities.record_supported(&carrier.node_id);
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        return Ok((outcome, DirectoryMirrorPullSource::PublicCarrier));
                    }
                    Err(carrier_reason)
                        if directory_mirror_failure_allows_recovery(&carrier_reason) =>
                    {
                        if directory_mirror_carrier_capability_unavailable(&carrier_reason) {
                            carrier_capabilities
                                .record_unsupported(carrier.node_id, carrier.descriptor_sequence);
                        }
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
    carrier_descriptor_sequence: u64,
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "directory_mirror_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("directory_mirror_producer_quarantined".to_string());
    }
    let (range_url, object_url) = directory_mirror_recovery_carrier_urls(
        peer_store,
        carrier,
        carrier_descriptor_sequence,
        request_timestamp,
    )?;
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

/// Verifies that at least one explicit signed mirror carrier can return one
/// locally retained producer anchor and its exact descriptor objects.
///
/// No direct producer request is attempted. The returned evidence is audited
/// against the local retained mirror and then discarded without import.
pub(crate) async fn run_directory_mirror_carrier_smoke(
    replica_store: Option<Arc<DirectoryReplicaStore>>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: Option<&reqwest::Client>,
) -> DirectoryMirrorCarrierSmokeReport {
    let Some(replica_store) = replica_store else {
        return DirectoryMirrorCarrierSmokeReport::unavailable("replica_store_disabled");
    };
    let Some(client) = client else {
        return DirectoryMirrorCarrierSmokeReport::unavailable(
            "smoke_http_client_initialization_failed",
        );
    };
    let Ok(producers) = replica_store.mirror_producer_ids() else {
        return DirectoryMirrorCarrierSmokeReport::unavailable(
            "retained_mirror_registry_audit_failed",
        );
    };
    let mut report = DirectoryMirrorCarrierSmokeReport::pending();
    report.retained_producers = u64::try_from(producers.len()).unwrap_or(u64::MAX);
    if producers.is_empty() {
        report.failure_reason = Some("no_retained_mirror_producers");
        return report;
    }

    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let requester = identity.public_key_bytes();
    let mut last_failure = "no_retained_mirror_evidence";
    for producer in producers
        .into_iter()
        .take(DIRECTORY_MIRROR_CARRIER_SMOKE_MAX_PRODUCERS)
    {
        let tip = match replica_store.producer_tip(&producer) {
            Ok(tip) if tip.tip_height > 0 && !tip.quarantined => tip,
            Ok(_) => continue,
            Err(_) => {
                last_failure = "retained_mirror_tip_audit_failed";
                continue;
            }
        };
        report.eligible_retained_producers =
            report.eligible_retained_producers.saturating_add(1);
        let selection = directory_mirror_recovery_carriers_with_requirement(
            peer_store,
            &capability_cache,
            &producer,
            &requester,
            unix_now_secs(),
            true,
        );
        report.explicit_carrier_candidates = report
            .explicit_carrier_candidates
            .max(selection.explicitly_advertised_candidate_count);
        report.selected_routeable_carriers = report
            .selected_routeable_carriers
            .max(selection.selected_routeable_count);
        if selection.carriers.is_empty() {
            last_failure = "no_explicit_carrier_candidates";
            continue;
        }

        for carrier in selection.carriers {
            report.attempted_carriers = report.attempted_carriers.saturating_add(1);
            let context = DirectoryMirrorCarrierSmokeAttemptContext {
                replica_store: Arc::clone(&replica_store),
                peer_store,
                identity,
                client,
                producer,
                retained_tip_height: tip.tip_height,
                requester,
            };
            match verify_directory_mirror_carrier_smoke_candidate(&context, carrier).await {
                Ok((verified_blocks, verified_descriptor_objects)) => {
                    report.success = true;
                    report.status = "verified";
                    report.verified_blocks = verified_blocks;
                    report.verified_descriptor_objects = verified_descriptor_objects;
                    report.carrier_signature_verified = true;
                    report.producer_evidence_verified = true;
                    report.local_anchor_verified = true;
                    report.failure_reason = None;
                    return report;
                }
                Err((reason, carrier_signature_verified)) => {
                    report.carrier_signature_verified |= carrier_signature_verified;
                    last_failure = reason;
                }
            }
        }
    }
    report.failure_reason = Some(last_failure);
    report
}

/// Cold-bootstraps one operator-pinned producer through an explicit carrier.
///
/// The target producer is never contacted. Every producer attempt gets one new
/// SQLite `:memory:` store, starts at height one, imports up to three bounded
/// pages, rotates the first carrier between pages, and runs the complete
/// replica audit before the store is dropped. Only transport/availability
/// failures may try another carrier; evidence or import failures stop closed.
pub(crate) async fn run_directory_carrier_cold_bootstrap_smoke(
    configured_producers: &[[u8; 32]],
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: Option<&reqwest::Client>,
) -> DirectoryCarrierColdBootstrapSmokeReport {
    let configured_count = configured_producers.len();
    let Some(client) = client else {
        return DirectoryCarrierColdBootstrapSmokeReport::unavailable(
            configured_count,
            "smoke_http_client_initialization_failed",
        );
    };
    let requester = identity.public_key_bytes();
    let mut producers = configured_producers
        .iter()
        .copied()
        .filter(|producer| *producer != [0u8; 32] && *producer != requester)
        .collect::<Vec<_>>();
    producers.sort_unstable();
    producers.dedup();
    let mut report = DirectoryCarrierColdBootstrapSmokeReport::pending(configured_count);
    if producers.is_empty() {
        report.failure_reason = Some("no_configured_producers");
        return report;
    }

    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let mut last_failure = "no_cold_bootstrap_evidence";
    for producer in producers
        .into_iter()
        .take(DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PRODUCERS)
    {
        report.eligible_producers = report.eligible_producers.saturating_add(1);
        let selection = directory_mirror_recovery_carriers_with_requirement(
            peer_store,
            &capability_cache,
            &producer,
            &requester,
            unix_now_secs(),
            true,
        );
        report.explicit_carrier_candidates = report
            .explicit_carrier_candidates
            .max(selection.explicitly_advertised_candidate_count);
        report.selected_routeable_carriers = report
            .selected_routeable_carriers
            .max(selection.selected_routeable_count);
        if selection.carriers.is_empty() {
            last_failure = "no_explicit_carrier_candidates";
            continue;
        }

        let local_node_id = requester;
        let isolated_store = match tokio::task::spawn_blocking(move || {
            DirectoryReplicaStore::open(":memory:", local_node_id, unix_now_secs())
                .map(|(store, _)| Arc::new(store))
        })
        .await
        {
            Ok(Ok(store)) => store,
            _ => {
                last_failure = "isolated_store_initialization_failed";
                continue;
            }
        };
        let carriers = selection.carriers;
        let mut pages_imported = 0u32;
        let mut requests_used = 0u32;
        let mut imported_blocks = 0u64;
        let mut imported_commitments = 0u64;
        let mut successful_carriers = HashSet::new();
        let mut last_outcome = None;
        let mut producer_attempt_failed = false;

        while pages_imported < DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES {
            let first_carrier = usize::try_from(pages_imported)
                .unwrap_or(0)
                .checked_rem(carriers.len())
                .unwrap_or(0);
            let mut page_outcome = None;

            for offset in 0..carriers.len() {
                if requests_used.saturating_add(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
                    > DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET
                {
                    last_failure = "smoke_request_budget_exhausted";
                    break;
                }
                let carrier = carriers[(first_carrier + offset) % carriers.len()];
                report.attempted_carriers = report.attempted_carriers.saturating_add(1);
                match pull_directory_chain_pinned_page_via_discovered_carrier(
                    Arc::clone(&isolated_store),
                    peer_store,
                    identity,
                    &producer,
                    carrier,
                    client,
                )
                .await
                {
                    Ok(outcome) => {
                        requests_used = requests_used.saturating_add(outcome.requests_made);
                        successful_carriers.insert(carrier.node_id);
                        page_outcome = Some(outcome);
                        break;
                    }
                    Err(failure)
                        if directory_carrier_recovery_disposition(&failure.reason)
                            == DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure =>
                    {
                        // [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex]
                        // The tracked pull reports every range/object request
                        // consumed before failure. A complete page is still
                        // reserved before the next carrier is attempted.
                        requests_used = requests_used.saturating_add(failure.requests_made);
                        report.availability_failovers =
                            report.availability_failovers.saturating_add(1);
                        last_failure =
                            directory_mirror_carrier_smoke_failure_bucket(&failure.reason);
                    }
                    Err(failure) => {
                        requests_used = requests_used.saturating_add(failure.requests_made);
                        report.requests_used = report.requests_used.max(u64::from(requests_used));
                        report.failure_reason = Some(
                            directory_mirror_carrier_smoke_failure_bucket(&failure.reason),
                        );
                        return report;
                    }
                }
            }

            let Some(outcome) = page_outcome else {
                // [CARRIER-PARTIAL-PREFIX 2026-07-26 by Codex] A carrier may
                // retain only a bounded prefix or become unavailable after
                // serving earlier pages. Two or more imported pages already
                // prove multi-page third-party recovery once the isolated
                // store passes its full chain audit below. Availability may
                // stop extension, but must not erase verified evidence.
                producer_attempt_failed =
                    !directory_carrier_cold_bootstrap_prefix_ready(pages_imported);
                break;
            };
            pages_imported = pages_imported.saturating_add(1);
            imported_blocks = imported_blocks.saturating_add(outcome.import.blocks_inserted);
            imported_commitments =
                imported_commitments.saturating_add(outcome.import.commitments_inserted);
            report.carrier_signature_verified = true;
            last_outcome = Some(outcome);

            if !should_continue_directory_carrier_cold_bootstrap(
                pages_imported,
                requests_used,
                outcome.has_more,
            ) {
                break;
            }
        }

        report.pages_imported = report.pages_imported.max(u64::from(pages_imported));
        report.requests_used = report.requests_used.max(u64::from(requests_used));
        if producer_attempt_failed {
            continue;
        }
        let Some(last_outcome) = last_outcome else {
            last_failure = "no_cold_bootstrap_evidence";
            continue;
        };
        if !directory_carrier_cold_bootstrap_prefix_ready(pages_imported) {
            last_failure = "insufficient_multi_page_evidence";
            continue;
        }

        let verification_store = Arc::clone(&isolated_store);
        let verification = tokio::task::spawn_blocking(move || {
            let tip = verification_store.producer_tip(&producer)?;
            let audit = verification_store.audit(unix_now_secs())?;
            let mirrors = verification_store.mirror_producer_ids()?;
            if tip.tip_height == 0
                || tip.quarantined
                || audit.producers != 1
                || audit.mirror_producers != 0
                || audit.quarantined_producers != 0
                || audit.blocks != tip.tip_height
                || !mirrors.is_empty()
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "isolated cold-bootstrap audit contract mismatch".to_string(),
                ));
            }
            Ok((tip.tip_height, audit.blocks, audit.commitments))
        })
        .await;
        let Ok(Ok((tip_height, audited_blocks, audited_commitments))) = verification else {
            last_failure = "isolated_store_audit_failed";
            continue;
        };
        if imported_blocks == 0
            || last_outcome.import.tip_height != tip_height
            || audited_blocks != imported_blocks
            || audited_commitments != imported_commitments
        {
            last_failure = "isolated_store_import_contract_mismatch";
            continue;
        }

        report.success = true;
        report.status = "verified";
        report.distinct_successful_carriers =
            u64::try_from(successful_carriers.len()).unwrap_or(u64::MAX);
        report.imported_blocks = imported_blocks;
        report.imported_commitments = imported_commitments;
        report.bootstrapped_tip_height = tip_height;
        report.multi_page_prefix_verified = true;
        report.reached_observed_remote_tip =
            directory_sync_outcome_is_checkpoint_complete(&last_outcome);
        report.producer_chain_verified = true;
        report.genesis_anchor_verified = true;
        report.isolated_store_audit_verified = true;
        report.failure_reason = None;
        return report;
    }
    report.failure_reason = Some(last_failure);
    report
}

async fn verify_directory_mirror_carrier_smoke_candidate(
    context: &DirectoryMirrorCarrierSmokeAttemptContext<'_>,
    carrier: DirectoryMirrorRecoveryCarrier,
) -> Result<(u64, u64), (&'static str, bool)> {
    let request_timestamp = unix_now_secs();
    let (range_url, object_url) = directory_mirror_recovery_carrier_urls(
        context.peer_store,
        &carrier.node_id,
        carrier.descriptor_sequence,
        request_timestamp,
    )
    .map_err(|reason| {
        (
            directory_mirror_carrier_smoke_failure_bucket(&reason),
            false,
        )
    })?;
    let page = request_directory_replica_block_page(
        context.identity,
        &context.producer,
        &carrier.node_id,
        context.client,
        range_url,
        context.retained_tip_height,
        request_timestamp,
    )
    .await
    .map_err(|reason| {
        (
            directory_mirror_carrier_smoke_failure_bucket(&reason),
            false,
        )
    })?;
    let (objects, _) = hydrate_directory_replica_descriptor_objects(
        context.identity,
        &context.producer,
        &carrier.node_id,
        context.client,
        object_url,
        &context.requester,
        &page.blocks,
    )
    .await
    .map_err(|reason| {
        (
            directory_mirror_carrier_smoke_failure_bucket(&reason),
            true,
        )
    })?;
    let DirectoryRangePage {
        blocks,
        has_more: _,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    } = page;
    let store = Arc::clone(&context.replica_store);
    let producer = context.producer;
    let retained_tip_height = context.retained_tip_height;
    tokio::task::spawn_blocking(move || {
        store.verify_retained_carrier_page(
            producer,
            retained_tip_height,
            &blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &signed_response,
            unix_now_secs(),
        )
    })
    .await
    .map_err(|_| ("carrier_verification_task_failed", true))?
    .map_err(|_| ("carrier_evidence_rejected", true))
}

fn directory_mirror_carrier_smoke_failure_bucket(reason: &str) -> &'static str {
    if reason.contains("_transport_failed")
        || reason.contains("_http_status_")
        || reason.contains("_peer_replica_")
        || reason.contains("_peer_mirror_")
        || reason.contains("_carrier_unavailable")
        || reason.contains("_carrier_descriptor_changed")
        || reason.contains("_carrier_not_public")
        || reason.contains("_carrier_missing_endpoint")
        || reason.contains("_carrier_unsafe_endpoint")
        || reason.contains("_carrier_invalid_endpoint")
    {
        "carrier_unavailable"
    } else if reason.contains("_response_")
        || reason.contains("_invalid_")
        || reason.contains("_hash_mismatch")
        || reason.contains("_noncanonical")
        || reason.contains("_contract_mismatch")
    {
        "carrier_evidence_rejected"
    } else {
        "carrier_request_failed"
    }
}

async fn pull_directory_chain_page_with_carriers(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carriers: &[[u8; 32]],
    carrier_capabilities: &DirectoryMirrorCarrierCapabilityCache,
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
            // [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] Bound every
            // availability fallback and account conservatively for the failed
            // direct range even when endpoint validation consumed no request.
            let requester = identity.public_key_bytes();
            let mut prior_requests = 1u32;
            let mut attempted = HashSet::new();
            for carrier in carriers
                .iter()
                .copied()
                .filter(|candidate| candidate != producer && *candidate != requester)
                .take(DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE)
            {
                attempted.insert(carrier);
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
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        debug!(
                            requests_made = outcome.requests_made,
                            "[DIRECTORY_REPLICA] Pinned carrier recovered producer evidence"
                        );
                        return Ok(outcome);
                    }
                    Err(carrier_reason)
                        if directory_sync_failure_allows_carrier_fallback(&carrier_reason) =>
                    {
                        prior_requests = prior_requests.saturating_add(1);
                    }
                    Err(carrier_reason) => return Err(carrier_reason),
                }
            }

            let selection = directory_mirror_recovery_carriers_with_requirement(
                peer_store,
                carrier_capabilities,
                producer,
                &requester,
                unix_now_secs(),
                true,
            );
            for carrier in selection.carriers {
                if !attempted.insert(carrier.node_id) {
                    continue;
                }
                match pull_directory_chain_pinned_page_via_discovered_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    carrier,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        carrier_capabilities.record_supported(&carrier.node_id);
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        debug!(
                            requests_made = outcome.requests_made,
                            "[DIRECTORY_REPLICA] Explicit public carrier cold-recovered pinned producer evidence"
                        );
                        return Ok(outcome);
                    }
                    Err(carrier_failure)
                        if directory_mirror_failure_allows_recovery(&carrier_failure.reason) =>
                    {
                        if directory_mirror_carrier_capability_unavailable(&carrier_failure.reason)
                        {
                            carrier_capabilities
                                .record_unsupported(carrier.node_id, carrier.descriptor_sequence);
                        }
                        prior_requests =
                            prior_requests.saturating_add(carrier_failure.requests_made);
                    }
                    Err(carrier_failure) => return Err(carrier_failure.reason),
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
        || reason == "directory_replica_objects_transport_failed"
        || reason == "directory_replica_range_peer_replica_not_found"
        || reason == "directory_replica_range_peer_replica_range_not_retained"
        || reason == "directory_replica_range_peer_mirror_replica_not_retained"
        || reason == "directory_replica_objects_peer_replica_object_not_found"
        || reason == "directory_replica_objects_peer_mirror_replica_not_retained"
    {
        return true;
    }
    for prefix in [
        "directory_range_http_status_",
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

fn directory_mirror_failure_allows_recovery(reason: &str) -> bool {
    if matches!(
        reason,
        "directory_range_transport_failed"
            | "directory_objects_transport_failed"
            | "directory_replica_range_transport_failed"
            | "directory_replica_objects_transport_failed"
            | "directory_replica_range_peer_replica_not_found"
            | "directory_replica_range_peer_replica_range_not_retained"
            | "directory_replica_range_peer_mirror_replica_not_retained"
            | "directory_replica_objects_peer_replica_object_not_found"
            | "directory_replica_objects_peer_mirror_replica_not_retained"
            | "directory_mirror_recovery_carrier_unavailable"
            | "directory_mirror_recovery_carrier_descriptor_changed"
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
        return matches!(status, 403 | 404 | 405 | 408 | 429) || status >= 500;
    }
    false
}

fn directory_mirror_carrier_capability_unavailable(reason: &str) -> bool {
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Cache only explicit absence of
    // optional replica-carrier endpoints. A direct producer endpoint, generic
    // transport error, overload response, or invalid signed frame must not
    // suppress a future carrier attempt.
    for prefix in [
        "directory_replica_range_http_status_",
        "directory_replica_objects_http_status_",
    ] {
        let Some(status) = reason
            .strip_prefix(prefix)
            .and_then(|value| value.parse::<u16>().ok())
        else {
            continue;
        };
        return matches!(status, 404 | 405 | 501);
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
    capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    producer: &[u8; 32],
    requester: &[u8; 32],
    now: u64,
) -> DirectoryMirrorRecoveryCarrierSelection {
    directory_mirror_recovery_carriers_with_requirement(
        peer_store,
        capability_cache,
        producer,
        requester,
        now,
        false,
    )
}

fn directory_mirror_recovery_carriers_with_requirement(
    peer_store: &PeerStore,
    capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    producer: &[u8; 32],
    requester: &[u8; 32],
    now: u64,
    require_explicitly_advertised: bool,
) -> DirectoryMirrorRecoveryCarrierSelection {
    let mut descriptors = peer_store
        .valid_public_descriptors(now, usize::MAX)
        .into_iter()
        .filter(|descriptor| {
            let node_id = descriptor.node_id();
            node_id != *producer
                && node_id != *requester
                && !peer_store.is_route_quarantined_now(&node_id, now)
                && descriptor.descriptor.policy.public_discovery
                && descriptor
                    .descriptor
                    .public_endpoint
                    .as_deref()
                    .is_some_and(commitment_peer_endpoint_is_public)
                && (!require_explicitly_advertised
                    || descriptor
                        .descriptor
                        .capabilities
                        .contains(&NodeCapability::DirectoryMirrorCarrier))
        })
        .collect::<Vec<_>>();
    descriptors.sort_by_key(SignedNodeDescriptor::node_id);
    descriptors.dedup_by_key(|descriptor| descriptor.node_id());
    if descriptors.is_empty() {
        return DirectoryMirrorRecoveryCarrierSelection::default();
    }

    let producer_seed = u64::from_be_bytes(producer[..8].try_into().unwrap_or([0u8; 8]));
    let requester_seed = u64::from_be_bytes(requester[..8].try_into().unwrap_or([0u8; 8]));
    let epoch_seed = now / DIRECTORY_MIRROR_RECOVERY_ROTATION_SECS;
    let cursor = usize::try_from(producer_seed ^ requester_seed ^ epoch_seed).unwrap_or(0)
        % descriptors.len();
    descriptors.rotate_left(cursor);

    // [MIRROR-DIVERSITY 2026-07-24 by Codex] Routeability is local observed
    // evidence and therefore outranks self-reported descriptor metadata.
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Within that local-evidence tier,
    // signed carrier capability outranks separately measured unadvertised
    // compatibility fallback.
    // Freshness is bucketed to avoid permanent affinity to tiny timestamp
    // differences; deterministic rotation remains the tie-breaker.
    let candidate_count = u64::try_from(descriptors.len()).unwrap_or(u64::MAX);
    let routeable_candidate_count = u64::try_from(
        descriptors
            .iter()
            .filter(|descriptor| peer_store.is_routeable_now(&descriptor.node_id(), now))
            .count(),
    )
    .unwrap_or(u64::MAX);
    let explicitly_advertised_candidate_count = u64::try_from(
        descriptors
            .iter()
            .filter(|descriptor| {
                descriptor
                    .descriptor
                    .capabilities
                    .contains(&NodeCapability::DirectoryMirrorCarrier)
            })
            .count(),
    )
    .unwrap_or(u64::MAX);
    let unadvertised_compatibility_candidate_count =
        candidate_count.saturating_sub(explicitly_advertised_candidate_count);
    let capability_cached_unavailable_count = u64::try_from(
        descriptors
            .iter()
            .filter(|descriptor| {
                !capability_cache.should_attempt(&descriptor.node_id(), descriptor.sequence())
            })
            .count(),
    )
    .unwrap_or(u64::MAX);

    let mut candidates = descriptors
        .into_iter()
        .filter(|descriptor| {
            capability_cache.should_attempt(&descriptor.node_id(), descriptor.sequence())
        })
        .enumerate()
        .map(|(rotation_rank, descriptor)| {
            let issued_at = descriptor.descriptor.issued_at;
            let age = now.checked_sub(issued_at);
            let freshness_rank = match age {
                Some(age) if age <= DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS => 0,
                Some(age) if age <= DIRECTORY_MIRROR_RECOVERY_AGING_DESCRIPTOR_SECS => 1,
                Some(_) => 2,
                None => 3,
            };
            let signed_region_hint = descriptor
                .descriptor
                .policy
                .region
                .as_deref()
                .map(str::trim)
                .filter(|region| !region.is_empty())
                .map(str::to_ascii_lowercase);
            let explicitly_advertised = descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::DirectoryMirrorCarrier);
            let node_id = descriptor.node_id();
            DirectoryMirrorRecoveryCarrierCandidate {
                node_id,
                descriptor_sequence: descriptor.sequence(),
                explicitly_advertised,
                routeable: peer_store.is_routeable_now(&node_id, now),
                freshness_rank,
                rotation_rank,
                signed_region_hint,
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|candidate| {
        let (routeability_rank, capability_rank, freshness_rank) =
            candidate.availability_tier();
        (
            routeability_rank,
            capability_rank,
            freshness_rank,
            candidate.rotation_rank,
        )
    });

    let mut selected = Vec::with_capacity(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE);
    let mut selected_regions = HashSet::new();
    while !candidates.is_empty() && selected.len() < DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE
    {
        let best_tier = candidates[0].availability_tier();
        let position = candidates
            .iter()
            .position(|candidate| {
                candidate.availability_tier() == best_tier
                    && candidate
                        .signed_region_hint
                        .as_ref()
                        .is_some_and(|region| !selected_regions.contains(region))
            })
            .or_else(|| {
                candidates
                    .iter()
                    .position(|candidate| candidate.availability_tier() == best_tier)
            })
            .unwrap_or(0);
        let candidate = candidates.remove(position);
        if let Some(region) = candidate.signed_region_hint.as_ref() {
            selected_regions.insert(region.clone());
        }
        selected.push(candidate);
    }

    DirectoryMirrorRecoveryCarrierSelection {
        carriers: selected
            .iter()
            .map(|candidate| DirectoryMirrorRecoveryCarrier {
                node_id: candidate.node_id,
                descriptor_sequence: candidate.descriptor_sequence,
            })
            .collect(),
        candidate_count,
        routeable_candidate_count,
        explicitly_advertised_candidate_count,
        unadvertised_compatibility_candidate_count,
        capability_cached_unavailable_count,
        selected_routeable_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| candidate.routeable)
                .count(),
        )
        .unwrap_or(u64::MAX),
        selected_explicitly_advertised_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| candidate.explicitly_advertised)
                .count(),
        )
        .unwrap_or(u64::MAX),
        selected_unadvertised_compatibility_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| !candidate.explicitly_advertised)
                .count(),
        )
        .unwrap_or(u64::MAX),
        selected_region_hint_count: u64::try_from(
            selected
                .iter()
                .filter(|candidate| candidate.signed_region_hint.is_some())
                .count(),
        )
        .unwrap_or(u64::MAX),
        distinct_selected_region_hint_count: u64::try_from(selected_regions.len())
            .unwrap_or(u64::MAX),
    }
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

/// Pull one pinned-producer page through a permissionless carrier whose exact
/// signed descriptor sequence advertised `DirectoryMirrorCarrier`.
///
/// [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] The carrier signs only the
/// transport envelope. `import_directory_range_page` independently verifies
/// the pinned producer's block signatures, genesis/hash chain, advertised tip,
/// and exact descriptor commitments before an atomic import.
async fn pull_directory_chain_pinned_page_via_discovered_carrier(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: DirectoryMirrorRecoveryCarrier,
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, DirectoryCarrierPullFailure> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| DirectoryCarrierPullFailure::new("replica_tip_unavailable".to_string(), 0))?;
    if local_tip.quarantined {
        return Err(DirectoryCarrierPullFailure::new(
            "producer_quarantined".to_string(),
            0,
        ));
    }
    let (range_url, object_url) = directory_mirror_recovery_carrier_urls(
        peer_store,
        &carrier.node_id,
        carrier.descriptor_sequence,
        request_timestamp,
    )
    .map_err(|reason| DirectoryCarrierPullFailure::new(reason, 0))?;
    let from_height = local_tip.tip_height.checked_add(1).ok_or_else(|| {
        DirectoryCarrierPullFailure::new("replica_height_exhausted".to_string(), 0)
    })?;
    let requester = identity.public_key_bytes();
    let page = request_directory_replica_block_page(
        identity,
        producer,
        &carrier.node_id,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await
    .map_err(|reason| DirectoryCarrierPullFailure::new(reason, 1))?;
    let (objects, requests_made) = hydrate_directory_replica_descriptor_objects_tracked(
        identity,
        producer,
        &carrier.node_id,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_range_page(replica_store, *producer, page, objects, requests_made)
        .await
        .map_err(|reason| DirectoryCarrierPullFailure::new(reason, requests_made))
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
    descriptor_sequence: u64,
    request_timestamp: u64,
) -> Result<(reqwest::Url, reqwest::Url), String> {
    let descriptor = peer_store
        .get_valid(carrier, request_timestamp)
        .ok_or_else(|| "directory_mirror_recovery_carrier_unavailable".to_string())?;
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] Bind endpoint derivation to the
    // same authenticated descriptor sequence selected by capability policy.
    // A concurrent descriptor change retries through a fresh selection rather
    // than probing or caching an endpoint under the wrong sequence.
    if descriptor.sequence() != descriptor_sequence {
        return Err("directory_mirror_recovery_carrier_descriptor_changed".to_string());
    }
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
    hydrate_directory_replica_descriptor_objects_tracked(
        identity, producer, carrier, client, object_url, requester, blocks,
    )
    .await
    .map_err(|failure| failure.reason)
}

#[allow(clippy::too_many_arguments)]
async fn hydrate_directory_replica_descriptor_objects_tracked(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
    object_url: reqwest::Url,
    requester: &[u8; 32],
    blocks: &[DirectoryCommitmentBlockV1],
) -> Result<(Vec<SignedNodeDescriptor>, u32), DirectoryCarrierPullFailure> {
    // [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Count the successful
    // range plus each object request at its dispatch boundary so carrier
    // failover cannot reset or understate the operator smoke budget.
    let descriptor_hashes = blocks
        .iter()
        .flat_map(|block| {
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash)
        })
        .collect::<Vec<_>>();
    // The successful range request has already been consumed before hydration.
    let mut requests_made = 1u32;
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
        let frame = encode_directory_sync_message(&request).map_err(|_| {
            DirectoryCarrierPullFailure::new(
                "directory_replica_object_request_encode_failed".to_string(),
                requests_made,
            )
        })?;
        requests_made = requests_made.saturating_add(1);
        let response = post_directory_frame(client, object_url.clone(), frame, "replica_objects")
            .await
            .map_err(|reason| DirectoryCarrierPullFailure::new(reason, requests_made))?;
        let mut verified = verify_replica_descriptor_objects_response(
            &response,
            &request_id,
            producer,
            carrier,
            hashes,
            request_timestamp,
        )
        .map_err(|reason| DirectoryCarrierPullFailure::new(reason, requests_made))?;
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
        let status = response.status().as_u16();
        // [MIRROR-CAPABILITY 2026-07-24 by Codex] A 404 can mean either that
        // an optional route is absent or that this carrier has not retained
        // the requested producer/range yet. Read only the tiny fixed protocol
        // code so temporary lag is never cached as missing software support.
        let peer_code =
            read_bounded_http_response(response, MAX_DIRECTORY_SYNC_ERROR_BODY_BYTES)
                .await
                .ok()
                .and_then(|body| DirectoryPeerErrorCode::parse(&body));
        return Err(DirectoryFramePostError::HttpStatus { status, peer_code });
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
    use aeronyx_core::protocol::discovery::{
        DirectoryDescriptorCommitmentV1, NodeDescriptor,
    };
    use axum::{http::StatusCode, routing::post, Router};
    use tempfile::TempDir;

    const TEST_NOW: u64 = 1_700_000_000;

    fn carrier_hydration_test_context(
    ) -> (
        IdentityKeyPair,
        IdentityKeyPair,
        IdentityKeyPair,
        DirectoryCommitmentBlockV1,
    ) {
        let requester = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let carrier = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
        let descriptor = SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                subject.public_key_bytes(),
                1,
                TEST_NOW - 10,
                TEST_NOW + 3_600,
                "carrier-hydration-test",
            ),
            &subject,
        )
        .unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            1,
            TEST_NOW,
            [0u8; 32],
            vec![commitment],
            &producer,
        )
        .unwrap();
        (requester, producer, carrier, block)
    }

    async fn carrier_hydration_test_endpoint(
        status: StatusCode,
        body: Vec<u8>,
    ) -> (reqwest::Url, JoinHandle<()>) {
        let app = Router::new().route(
            "/",
            post(move || {
                let body = body.clone();
                async move { (status, body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (
            reqwest::Url::parse(&format!("http://{address}/")).unwrap(),
            server,
        )
    }

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
            "directory_replica_objects_transport_failed",
            "directory_replica_range_peer_replica_range_not_retained",
            "directory_replica_objects_peer_replica_object_not_found",
            "directory_replica_objects_http_status_503",
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
    fn pinned_and_explicit_carrier_recovery_stays_inside_request_budget() {
        let worst_case_requests = 1usize
            .saturating_add(DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE)
            .saturating_add(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE)
            .saturating_add(
                usize::try_from(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
                    .expect("request bound fits usize"),
            );
        assert!(
            worst_case_requests
                <= usize::try_from(DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND)
                    .expect("round request budget fits usize")
        );
        assert_eq!(DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE, 2);
        assert_eq!(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE, 2);
    }

    #[test]
    fn carrier_cold_bootstrap_multi_page_budget_is_bounded() {
        // [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Every attempt
        // reserves a full worst-case page while accounting the exact requests
        // already consumed. Sparse pages can prove multi-page continuation;
        // dense pages stop before crossing the normal producer-round ceiling.
        assert_eq!(DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES, 3);
        assert_eq!(
            DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET,
            DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND
        );
        assert!(should_continue_directory_carrier_cold_bootstrap(1, 1, true));
        assert!(should_continue_directory_carrier_cold_bootstrap(2, 2, true));
        assert!(!should_continue_directory_carrier_cold_bootstrap(
            DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES,
            0,
            true,
        ));
        assert!(!should_continue_directory_carrier_cold_bootstrap(
            1, 0, false,
        ));
        assert!(!should_continue_directory_carrier_cold_bootstrap(
            1,
            DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET,
            true,
        ));
        assert!(!directory_carrier_cold_bootstrap_prefix_ready(1));
        assert!(directory_carrier_cold_bootstrap_prefix_ready(2));
        assert!(directory_carrier_cold_bootstrap_prefix_ready(3));
    }

    #[test]
    fn carrier_cold_bootstrap_retries_only_availability_failures() {
        for reason in [
            "directory_replica_range_transport_failed",
            "directory_replica_objects_transport_failed",
            "directory_replica_range_http_status_503",
            "directory_replica_range_peer_replica_range_not_retained",
            "directory_mirror_recovery_carrier_descriptor_changed",
        ] {
            assert_eq!(
                directory_carrier_recovery_disposition(reason),
                DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure
            );
        }
        for reason in [
            "directory_replica_range_response_noncanonical",
            "directory_replica_range_response_invalid_signature",
            "directory_replica_object_response_hash_mismatch",
            "directory_replica_import_rejected",
            "producer_quarantined",
        ] {
            assert_eq!(
                directory_carrier_recovery_disposition(reason),
                DirectoryCarrierRecoveryDisposition::StopClosed
            );
        }
    }

    #[tokio::test]
    async fn carrier_hydration_availability_failure_preserves_request_count() {
        let (requester, producer, carrier, block) = carrier_hydration_test_context();
        let (object_url, server) =
            carrier_hydration_test_endpoint(StatusCode::SERVICE_UNAVAILABLE, Vec::new()).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let failure = hydrate_directory_replica_descriptor_objects_tracked(
            &requester,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            &client,
            object_url,
            &requester.public_key_bytes(),
            &[block],
        )
        .await
        .unwrap_err();
        server.abort();

        // One already-successful range plus one dispatched object request.
        assert_eq!(failure.requests_made, 2);
        assert_eq!(
            directory_carrier_recovery_disposition(&failure.reason),
            DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure
        );
    }

    #[tokio::test]
    async fn carrier_hydration_corruption_stops_closed_without_losing_request_count() {
        let (requester, producer, carrier, block) = carrier_hydration_test_context();
        let (object_url, server) =
            carrier_hydration_test_endpoint(StatusCode::OK, b"corrupt-frame".to_vec()).await;
        let client = reqwest::Client::builder().no_proxy().build().unwrap();

        let failure = hydrate_directory_replica_descriptor_objects_tracked(
            &requester,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            &client,
            object_url,
            &requester.public_key_bytes(),
            &[block],
        )
        .await
        .unwrap_err();
        server.abort();

        assert_eq!(failure.requests_made, 2);
        assert_eq!(
            directory_carrier_recovery_disposition(&failure.reason),
            DirectoryCarrierRecoveryDisposition::StopClosed
        );
    }

    #[tokio::test]
    async fn cold_bootstrap_smoke_fails_closed_without_transport() {
        let local = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0xc2; 32])
            .unwrap()
            .public_key_bytes();
        let report = run_directory_carrier_cold_bootstrap_smoke(
            &[producer],
            &PeerStore::new(),
            &local,
            None,
        )
        .await;

        assert!(!report.success);
        assert_eq!(
            report.failure_reason,
            Some("smoke_http_client_initialization_failed")
        );
        assert_eq!(report.live_store_effect, "none_isolated_memory_store_only");
        assert_eq!(report.pages_imported, 0);
        assert_eq!(
            report.request_budget,
            u64::from(DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET)
        );
        assert!(!report.multi_page_prefix_verified);
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
            "directory_replica_range_http_status_405",
            "directory_replica_range_http_status_429",
            "directory_replica_objects_http_status_503",
            "directory_replica_range_peer_replica_range_not_retained",
            "directory_replica_objects_peer_replica_object_not_found",
            "directory_mirror_recovery_carrier_unavailable",
            "directory_mirror_recovery_carrier_descriptor_changed",
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
        let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
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

        let selection = directory_mirror_recovery_carriers(
            &store,
            &capability_cache,
            &producer.public_key_bytes(),
            &requester.public_key_bytes(),
            now,
        );
        assert_eq!(
            selection.carriers.len(),
            DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE
        );
        assert_eq!(selection.explicitly_advertised_candidate_count, 0);
        assert_eq!(selection.unadvertised_compatibility_candidate_count, 5);
        assert_eq!(selection.selected_explicitly_advertised_count, 0);
        assert_eq!(selection.selected_unadvertised_compatibility_count, 2);
        assert!(selection
            .carriers
            .iter()
            .all(|candidate| !expected_excluded.contains(&candidate.node_id)));
        assert_eq!(
            selection,
            directory_mirror_recovery_carriers(
                &store,
                &capability_cache,
                &producer.public_key_bytes(),
                &requester.public_key_bytes(),
                now,
            )
        );
    }

    #[test]
    fn mirror_recovery_carrier_selection_prefers_live_diverse_fresh_peers() {
        let now = unix_now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap();
        let requester = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
        let store = PeerStore::new();
        let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
        let mut fresh_routeable = HashSet::new();

        // [MIRROR-DIVERSITY 2026-07-24 by Codex] Three fresh routeable peers
        // include two identical signed region hints. The second selected peer
        // must use the different hint without sacrificing routeability or
        // descriptor freshness. These hints are not operator/ASN proof.
        for (seed, issued_at, region, routeable, advertised) in [
            (0xd3, now - 1, Some("region-a"), true, true),
            (0xd4, now - 2, Some("REGION-A"), true, false),
            (0xd5, now - 3, Some("region-b"), true, true),
            (
                0xd6,
                now - DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS - 1,
                Some("region-c"),
                true,
                true,
            ),
            (0xd7, now - 4, Some("region-d"), false, true),
        ] {
            let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
            let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
                identity.public_key_bytes(),
                1,
                issued_at,
                now + 600,
                "mirror-diversity-test",
            );
            descriptor.policy.public_discovery = true;
            descriptor.policy.region = region.map(str::to_string);
            descriptor.public_endpoint = Some(format!("http://8.8.8.{seed}:8422"));
            if advertised {
                descriptor
                    .capabilities
                    .push(NodeCapability::DirectoryMirrorCarrier);
            }
            store
                .upsert_verified_from_source(
                    SignedNodeDescriptor::sign(descriptor, &identity).unwrap(),
                    now,
                    "directory_mirror_diversity_test",
                )
                .unwrap();
            if routeable {
                store.record_route_forward_success(&identity.public_key_bytes(), now);
            }
            if routeable
                && issued_at >= now.saturating_sub(DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS)
            {
                fresh_routeable.insert(identity.public_key_bytes());
            }
        }

        let quarantined = IdentityKeyPair::from_bytes(&[0xd8; 32]).unwrap();
        let mut quarantined_descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
            quarantined.public_key_bytes(),
            1,
            now - 1,
            now + 600,
            "mirror-diversity-test",
        );
        quarantined_descriptor.policy.public_discovery = true;
        quarantined_descriptor.policy.region = Some("region-e".to_string());
        quarantined_descriptor.public_endpoint = Some("http://8.8.8.216:8422".to_string());
        store
            .upsert_verified_from_source(
                SignedNodeDescriptor::sign(quarantined_descriptor, &quarantined).unwrap(),
                now,
                "directory_mirror_diversity_test",
            )
            .unwrap();
        for _ in 0..3 {
            store.record_route_forward_failure(
                &quarantined.public_key_bytes(),
                now,
                "request_failed",
            );
        }

        let selection = directory_mirror_recovery_carriers(
            &store,
            &capability_cache,
            &producer.public_key_bytes(),
            &requester.public_key_bytes(),
            now,
        );
        assert_eq!(selection.candidate_count, 5);
        assert_eq!(selection.routeable_candidate_count, 4);
        assert_eq!(selection.explicitly_advertised_candidate_count, 4);
        assert_eq!(selection.unadvertised_compatibility_candidate_count, 1);
        assert_eq!(selection.carriers.len(), 2);
        assert_eq!(selection.selected_routeable_count, 2);
        assert_eq!(selection.selected_explicitly_advertised_count, 2);
        assert_eq!(selection.selected_unadvertised_compatibility_count, 0);
        assert_eq!(selection.selected_region_hint_count, 2);
        assert_eq!(selection.distinct_selected_region_hint_count, 2);
        assert!(selection
            .carriers
            .iter()
            .all(|candidate| fresh_routeable.contains(&candidate.node_id)));
        assert!(!selection
            .carriers
            .iter()
            .any(|candidate| candidate.node_id == quarantined.public_key_bytes()));

        // [MIRROR-CARRIER-SMOKE 2026-07-25 by Codex] Manual verification must
        // never silently fall back to an unadvertised compatibility carrier.
        let smoke_selection = directory_mirror_recovery_carriers_with_requirement(
            &store,
            &capability_cache,
            &producer.public_key_bytes(),
            &requester.public_key_bytes(),
            now,
            true,
        );
        assert_eq!(smoke_selection.candidate_count, 4);
        assert_eq!(smoke_selection.explicitly_advertised_candidate_count, 4);
        assert_eq!(smoke_selection.unadvertised_compatibility_candidate_count, 0);
        assert_eq!(smoke_selection.selected_explicitly_advertised_count, 2);
        assert_eq!(smoke_selection.selected_unadvertised_compatibility_count, 0);
    }

    #[test]
    fn mirror_recovery_selection_skips_only_the_cached_descriptor_sequence() {
        let now = unix_now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
        let requester = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
        let store = PeerStore::new();
        let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
        let mut carriers = Vec::new();

        for seed in [0xc3, 0xc4, 0xc5] {
            let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
            let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
                identity.public_key_bytes(),
                7,
                now.saturating_sub(1),
                now + 600,
                "mirror-capability-test",
            );
            descriptor.policy.public_discovery = true;
            descriptor.public_endpoint = Some(format!("http://8.8.8.{seed}:8422"));
            store
                .upsert_verified_from_source(
                    SignedNodeDescriptor::sign(descriptor, &identity).unwrap(),
                    now,
                    "directory_mirror_capability_test",
                )
                .unwrap();
            store.record_route_forward_success(&identity.public_key_bytes(), now);
            carriers.push(identity.public_key_bytes());
        }

        capability_cache.record_unsupported(carriers[0], 7);
        let selection = directory_mirror_recovery_carriers(
            &store,
            &capability_cache,
            &producer.public_key_bytes(),
            &requester.public_key_bytes(),
            now,
        );
        assert_eq!(selection.candidate_count, 3);
        assert_eq!(selection.explicitly_advertised_candidate_count, 0);
        assert_eq!(selection.unadvertised_compatibility_candidate_count, 3);
        assert_eq!(selection.capability_cached_unavailable_count, 1);
        assert_eq!(selection.carriers.len(), 2);
        assert_eq!(selection.selected_unadvertised_compatibility_count, 2);
        assert!(!selection
            .carriers
            .iter()
            .any(|candidate| candidate.node_id == carriers[0]));
        assert!(capability_cache.should_attempt(&carriers[0], 8));
    }

    #[test]
    fn carrier_smoke_failures_collapse_to_privacy_safe_buckets() {
        assert_eq!(
            directory_mirror_carrier_smoke_failure_bucket(
                "directory_replica_range_http_status_503"
            ),
            "carrier_unavailable"
        );
        assert_eq!(
            directory_mirror_carrier_smoke_failure_bucket(
                "directory_replica_object_response_invalid_signature"
            ),
            "carrier_evidence_rejected"
        );
        assert_eq!(
            directory_mirror_carrier_smoke_failure_bucket(
                "directory_replica_range_request_encode_failed"
            ),
            "carrier_request_failed"
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
    fn mirror_carrier_capability_cache_is_bounded_and_failure_specific() {
        let cache = DirectoryMirrorCarrierCapabilityCache::default();
        for index in 0..=DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES {
            let mut carrier = [0u8; 32];
            carrier[..8].copy_from_slice(
                &u64::try_from(index)
                    .expect("test cache index fits u64")
                    .to_be_bytes(),
            );
            cache.record_unsupported(carrier, 1);
        }
        let mut oldest = [0u8; 32];
        oldest[..8].copy_from_slice(&0u64.to_be_bytes());
        let mut newest = [0u8; 32];
        newest[..8].copy_from_slice(
            &u64::try_from(DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES)
                .expect("test cache capacity fits u64")
                .to_be_bytes(),
        );
        assert_eq!(
            cache.len(),
            DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES
        );
        assert!(cache.should_attempt(&oldest, 1));
        assert!(!cache.should_attempt(&newest, 1));
        assert!(cache.should_attempt(&newest, 2));

        for reason in [
            "directory_replica_range_http_status_404",
            "directory_replica_range_http_status_405",
            "directory_replica_objects_http_status_501",
        ] {
            assert!(directory_mirror_carrier_capability_unavailable(reason));
        }
        for reason in [
            "directory_range_http_status_404",
            "directory_replica_range_transport_failed",
            "directory_replica_range_http_status_403",
            "directory_replica_range_http_status_408",
            "directory_replica_range_http_status_429",
            "directory_replica_range_http_status_500",
            "directory_replica_range_response_invalid_signature",
            "directory_replica_range_peer_replica_not_found",
            "directory_replica_range_peer_replica_range_not_retained",
            "directory_replica_objects_peer_replica_object_not_found",
        ] {
            assert!(!directory_mirror_carrier_capability_unavailable(reason));
        }
    }

    #[test]
    fn mirror_carrier_endpoint_is_bound_to_selected_descriptor_sequence() {
        let now = unix_now_secs();
        let store = PeerStore::new();
        let carrier = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
            carrier.public_key_bytes(),
            7,
            now.saturating_sub(1),
            now + 600,
            "mirror-capability-sequence-test",
        );
        descriptor.policy.public_discovery = true;
        descriptor.public_endpoint = Some("http://8.8.8.146:8422".to_string());
        store
            .upsert_verified_from_source(
                SignedNodeDescriptor::sign(descriptor, &carrier).unwrap(),
                now,
                "directory_mirror_capability_sequence_test",
            )
            .unwrap();

        assert!(directory_mirror_recovery_carrier_urls(
            &store,
            &carrier.public_key_bytes(),
            7,
            now,
        )
        .is_ok());
        assert_eq!(
            directory_mirror_recovery_carrier_urls(
                &store,
                &carrier.public_key_bytes(),
                8,
                now,
            )
            .unwrap_err(),
            "directory_mirror_recovery_carrier_descriptor_changed"
        );
    }

    #[test]
    fn witness_capability_http_statuses_are_narrow_and_typed() {
        for status in [404, 405, 501] {
            assert!(DirectoryFramePostError::HttpStatus {
                status,
                peer_code: None,
            }
            .witness_capability_unavailable());
        }
        for status in [400, 401, 403, 409, 429, 500, 503] {
            assert!(!DirectoryFramePostError::HttpStatus {
                status,
                peer_code: None,
            }
            .witness_capability_unavailable());
        }
        assert_eq!(
            DirectoryFramePostError::HttpStatus {
                status: 404,
                peer_code: None,
            }
            .stable_reason("range"),
            "directory_range_http_status_404"
        );
        let lagging_carrier = DirectoryFramePostError::HttpStatus {
            status: 404,
            peer_code: Some(DirectoryPeerErrorCode::ReplicaRangeNotRetained),
        };
        assert!(!lagging_carrier.witness_capability_unavailable());
        assert_eq!(
            lagging_carrier.stable_reason("replica_range"),
            "directory_replica_range_peer_replica_range_not_retained"
        );
        assert_eq!(
            DirectoryFramePostError::Response(BoundedHttpResponseError::TooLarge)
                .stable_reason("objects"),
            "directory_objects_response_too_large"
        );
    }
}
