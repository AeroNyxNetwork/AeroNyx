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
//! - [WITNESS-CATCHUP 2026-07-26 by Codex] Advances a bounded batch of distinct
//!   mature checkpoints per synchronization round so restart backlog converges
//!   instead of remaining permanently one-for-one with newly appended heads.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Keeps witness requests direct-first,
//!   then tries at most two operator-pinned, explicitly advertised carriers
//!   only after bounded availability failures.
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
//! - [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] Resumes retained mirror
//!   prefixes through current admitted carriers after producer discovery
//!   expiry, while only direct producer responses can declare convergence.
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
//! - Provides an operator-triggered pinned-peer certificate pull primitive
//!   with redirect-free transport and exact response/certificate verification.
//! - [REPLICA-PROOF-RECOVERY 2026-07-27 by Codex] Fetches an exact descriptor
//!   inclusion proof direct from its producer first, then tries at most two
//!   explicitly advertised carriers only for typed availability failures.
//! - [DIRECTORY-PEER-ADMISSION 2026-07-27 by Codex] Admits a proven descriptor
//!   into `PeerStore` only after an exact locally retained replica proof matches
//!   the independently verified network proof.
//! - [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] Applies the same local
//!   replica anchor to proof-carrying discovery gossip before PeerStore import.
//! - [PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex] Accepts the server's
//!   process-lifetime hardened Directory transport while preserving the
//!   constructor that builds an equivalent client for tests and embedders.
//! - [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] Exports the canonical
//!   Directory sync connect/request deadlines so the injected production
//!   profile cannot drift from standalone constructors and timeout tests.
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
//!    peers to independently recompute up to four distinct forward mature
//!    checkpoints below their configured corroboration target; persist only
//!    canonical accepted receipts, never trust an unavailable or conflicting
//!    result, and stop the batch if the selector does not advance.
//! 10. Treat an explicitly unsupported witness endpoint as peer unavailability
//!     and retry only after that peer publishes a newer signed descriptor.
//! 11. If direct witness transport is unavailable, try at most two current,
//!     operator-pinned `DirectoryMirrorCarrier` descriptors. Each attempt uses
//!     a fresh inner request id; target rejection or invalid evidence stops
//!     closed, while the exact witness signature remains mandatory.
//! 12. Persist bounded aggregate witness outcomes and mirror the current
//!     process round into runtime telemetry without retaining witness identity.
//! 13. Stop the complete round immediately when shutdown wins the select.
//! 14. Ask missing current pins to retain the opaque current policy head and
//!     persist only exact accepted signed receipts.
//! 15. Select verified public mirror candidates, exclude self and authority
//!     pins, and catch each selection up within a strict page, request, and
//!     wall-clock budget. Try the producer directly before at most two public
//!     carriers. Every imported block remains signed by the original producer;
//!     a carrier signs only the response envelope and never gains authority.
//! 16. Prefer fresh routeability evidence, rotate equally healthy candidates,
//!     and avoid repeating a signed region hint when an equally healthy
//!     alternative exists. Region hints never prove operator or ASN diversity.
//! 17. On explicit operator request, prove bounded multi-page cold recovery in
//!     an isolated in-memory replica. Rotate the first carrier between pages,
//!     retry only availability failures, and preserve an already verified
//!     multi-page prefix if a later page becomes unavailable. Stop closed on
//!     cryptographic or import failures, then discard the replica without
//!     touching the live store.
//! 18. On explicit operator request, fetch one portable observation certificate
//!     from an expected pinned identity and verify request binding, responder
//!     signature, freshness, frame size, and SHA-256 before local policy import.
//! 19. For one independently selected producer/block/descriptor tuple, request
//!     the compact inclusion proof directly, then use bounded carrier recovery
//!     only when transport or route admission is unavailable. Verify the
//!     producer proof and carrier envelope independently.
//! 20. Before proof retrieval, require the exact producer/block/descriptor
//!     anchor in the audited local replica. After retrieval, re-audit and match
//!     the deterministic proof byte-for-byte before normal PeerStore admission.
//! 21. For proof-carrying discovery gossip, ignore sender authority and admit
//!     only after the exact deterministic proof matches the local replica.
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
//! - [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] A valid transport response is
//!   not trust by itself. Importers must additionally verify the certificate's
//!   observer, witness signatures, current local pins, threshold, and age.
//! - [WITNESS-CATCHUP 2026-07-26 by Codex] Keep catch-up sequential across
//!   checkpoint sequences and concurrent only across missing pinned witnesses.
//!   Never retry the same sequence twice in one batch or remove the hard batch
//!   ceiling; those rules bound request amplification during peer failure.
//! - [WITNESS-CARRIER 2026-07-26 by Codex] Carrier fallback is availability
//!   recovery, not trust recovery. Select only current explicit carrier
//!   capabilities that are also local operator pins, never recurse, and stop
//!   on any target rejection, noncanonical frame, contract, or signature fault.
//!   Keep the dedicated 16 KiB inner and 32 KiB outer response ceilings.
//! - [REPLICA-PROOF-RECOVERY 2026-07-27 by Codex] A semantic producer
//!   `proof_not_found`, noncanonical frame, contract mismatch, bad signature,
//!   or invalid proof is not retryable. Carrier failover must never conceal
//!   contradictory evidence or choose the trusted producer/block for a caller.
//! - [DIRECTORY-PEER-ADMISSION 2026-07-27 by Codex] A transport-authenticated
//!   proof is not a local trust anchor. Keep both local replica audits around
//!   the network request, exact-proof equality, and `PeerStore` anti-rollback.
//! - [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] Coordinator HTTP
//!   outcomes must remain task-scoped and aggregate. Never attach peer,
//!   producer, carrier, endpoint, URL, status code, request, or frame labels.
//!
//! ## Last Modified
//! `v0.32.0-DirectoryMirrorProvenance` - Separated direct producer completion
//! from carrier-verified prefixes and resumed retained mirrors after expiry.
//! `v0.31.0-DirectoryTransportTelemetry` - Added task-scoped, mutually
//! exclusive process transport outcomes without changing stable failure codes.
//! `v0.30.0-RoleSpecificTransportBudgets` - Restored one canonical 10-second
//! replica request deadline while preserving the separate operator budget.
//! `v0.29.0-ProcessLifetimePeerTransport` - Reused the server-owned Directory
//! HTTP pool so synchronization cannot silently disappear after startup.
//! `v0.28.0-DirectoryAuthenticatedGossipAdmission` - Added sender-neutral
//! proof-gossip admission against exact audited local replica evidence.
//! `v0.27.0-DirectoryAuthenticatedPeerAdmission` - Added locally anchored proof
//! admission into `PeerStore` with preflight/postflight audits and stable errors.
//! `v0.26.0-ReplicaProofRecovery` - Added direct-first, at-most-two explicit
//! carrier recovery for exact descriptor inclusion proofs with dual verification.
//! `v0.25.0-BoundedWitnessCarrierRecovery` - Added direct-first, at-most-two,
//! pinned explicit-carrier witness recovery with exact target verification.
//! `v0.24.0-BoundedWitnessCatchUp` - Added a four-checkpoint mature witness
//! catch-up budget with strict sequence advancement and no same-round retry.
//! `v0.23.0-AuthenticatedCertificateExchange` - Added hardened pinned-source
//! portable observation-certificate pull and exact response verification.
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
    directory_descriptor_inclusion_proof_request_signing_bytes,
    directory_descriptor_inclusion_proof_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_descriptor_objects_response_signing_bytes,
    directory_observation_certificate_request_signing_bytes,
    directory_observation_certificate_response_signing_bytes,
    directory_observation_witness_carrier_request_signing_bytes,
    directory_observation_witness_carrier_response_signing_bytes,
    directory_observation_witness_request_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_policy_anchor_request_signing_bytes, directory_policy_anchor_response_signing_bytes,
    directory_replica_block_range_request_signing_bytes,
    directory_replica_block_range_response_signing_bytes,
    directory_replica_descriptor_inclusion_proof_request_signing_bytes,
    directory_replica_descriptor_inclusion_proof_response_signing_bytes,
    directory_replica_descriptor_objects_request_signing_bytes,
    directory_replica_descriptor_objects_response_signing_bytes, encode_directory_sync_message,
    DirectoryCommitmentBlockV1, DirectoryDescriptorInclusionProofV1,
    DirectoryObservationCheckpointV1, DirectorySyncMessage, NodeCapability, SignedNodeDescriptor,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
    DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1, DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
    DIRECTORY_POLICY_ANCHOR_CONFLICT_V1, DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
    DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES, MAX_DIRECTORY_SYNC_BLOCKS_V1,
    MAX_DIRECTORY_SYNC_OBJECTS_V1,
};
use futures::{stream, StreamExt};
use parking_lot::Mutex;
use rand::RngCore;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::api::memchain_peer::{commitment_peer_endpoint_is_public, commitment_peer_url};
use crate::api::{
    privacy_safe_peer_http_client_builder, read_bounded_http_response, BoundedHttpResponseError,
};
use crate::services::directory_replica::{
    DirectoryReplicaTransportOutcome, DirectoryRetainedMirrorCursor,
    DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS, DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES,
};
use crate::services::{
    DirectoryObservationWitnessOutcome, DirectoryReplicaImportReport, DirectoryReplicaStore,
    DirectoryReplicaStoreError, DirectoryReplicaSyncRuntime, PeerStore, PeerStoreError,
    PeerStoreImportReport,
};

/// Maximum pinned producers synchronized concurrently by one node.
pub(crate) const DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS: usize = 4;
/// Hard wall-clock ceiling for one producer within a synchronization round.
pub(crate) const DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS: u64 = 45;
/// TCP establishment remains short so unreachable peers fail over promptly.
///
/// [PEER-TRANSPORT-BUDGETS 2026-07-28 by Codex] This crate-visible value is
/// also consumed by the production process-lifetime transport profile.
pub(crate) const DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS: u64 = 3;
/// A verified carrier may audit thousands of retained blocks before exporting
/// one page. Keep the request bounded but leave enough time for that audit;
/// the independent producer-round deadline still caps the complete operation.
/// Production and standalone constructors consume this same value.
pub(crate) const DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS: u64 = 10;

// [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] Only futures polled
// inside a coordinator synchronization round can update the Directory sync
// profile counters. Operator smokes and standalone protocol helpers therefore
// cannot be misclassified as production synchronization traffic.
tokio::task_local! {
    static DIRECTORY_SYNC_TRANSPORT_RUNTIME: Arc<DirectoryReplicaSyncRuntime>;
}
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
/// [WITNESS-CATCHUP 2026-07-26 by Codex] Maximum distinct mature checkpoints
/// witnessed after one producer-sync round. Four closes a restart backlog while
/// keeping two-pin deployments at or below eight witness requests per round.
pub(crate) const DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND: usize = 4;
/// [WITNESS-CARRIER 2026-07-26 by Codex] Maximum authenticated transport
/// carriers tried after one exact witness direct path has an availability-only
/// failure. Cryptographic, canonical, admission, and evidence failures stop.
pub(crate) const DIRECTORY_OBSERVATION_WITNESS_RECOVERY_MAX_CARRIERS: usize = 2;
/// Exact witness responses are tiny. The carrier enforces this inner-frame
/// ceiling before signing its transport envelope.
const MAX_DIRECTORY_OBSERVATION_WITNESS_CARRIER_INNER_RESPONSE_BODY_BYTES: usize = 16 * 1024;
/// Bounds the signed outer carrier envelope independently from general
/// Directory block/object responses.
const MAX_DIRECTORY_OBSERVATION_WITNESS_CARRIER_RESPONSE_BODY_BYTES: usize = 32 * 1024;
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

/// Exact authenticated certificate bytes returned by one pinned source.
///
/// [CERTIFICATE-EXCHANGE 2026-07-26 by Codex] This proves only the transport
/// source and exact bytes. Callers must apply local certificate trust and age
/// policy before importing the frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedDirectoryObservationCertificate {
    /// Exact canonical portable certificate frame authenticated by the source.
    pub frame: Vec<u8>,
    /// SHA-256 of `frame`, recomputed before the response signature is trusted.
    pub certificate_sha256: [u8; 32],
    /// Expected pinned node identity that signed the transport response.
    pub source: [u8; 32],
    /// Authenticated response creation time in Unix epoch seconds.
    pub response_timestamp: u64,
}

/// Authenticated transport source for one recovered descriptor proof.
///
/// This intentionally omits peer identity and endpoint metadata. The proof
/// itself remains bound to the original producer and selected block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryDescriptorProofTransport {
    /// The original producer returned and transport-signed the proof.
    DirectProducer,
    /// An audited replica carrier transported the original producer proof.
    ReplicaCarrier,
}

/// Fully verified result of one bounded descriptor-proof recovery operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedDirectoryDescriptorProof {
    /// Original producer-signed descriptor inclusion proof.
    pub proof: DirectoryDescriptorInclusionProofV1,
    /// Privacy-safe transport class used for the successful response.
    pub transport: DirectoryDescriptorProofTransport,
    /// Whether a direct producer request was attempted.
    pub direct_attempted: bool,
    /// Number of bounded carriers attempted before success.
    pub carrier_attempts: u8,
}

/// Privacy-safe outcome of one locally anchored `PeerStore` admission.
///
/// The result deliberately omits node identity, producer identity, hashes,
/// endpoint, carrier, route, and proof contents. `inserted == false` means the
/// exact same descriptor sequence was already present and verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryAuthenticatedPeerAdmission {
    /// Whether `PeerStore` inserted or upgraded the proven descriptor.
    pub inserted: bool,
    /// Authenticated transport class used to recover the proof.
    pub transport: DirectoryDescriptorProofTransport,
    /// Whether the direct producer transport was attempted.
    pub direct_attempted: bool,
    /// Number of bounded replica carriers attempted before success.
    pub carrier_attempts: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryCarrierRecoveryDisposition {
    RetryAvailabilityFailure,
    StopClosed,
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

impl DirectoryMirrorPullSource {
    const fn authenticates_producer_tip(self) -> bool {
        matches!(self, Self::DirectProducer)
    }
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
    /// Whether the authenticated responder reports more pages.
    pub has_more: bool,
    /// Transport-authenticated reported tip height observed in this round.
    pub remote_tip_height: u64,
    /// Transport-authenticated reported tip hash observed in this round.
    pub remote_tip_hash: [u8; 32],
    /// HTTP requests consumed by this successful page and object hydration.
    pub requests_made: u32,
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
    ProofNotFound,
    ReplicaDescriptorProofNotFound,
    WitnessTargetUnavailable,
    WitnessTargetCapabilityUnavailable,
    WitnessTargetRejected,
    WitnessTargetInvalidResponse,
}

impl DirectoryPeerErrorCode {
    fn parse(body: &[u8]) -> Option<Self> {
        match body {
            b"replica_not_found" => Some(Self::ReplicaNotFound),
            b"replica_range_not_retained" => Some(Self::ReplicaRangeNotRetained),
            b"mirror_replica_not_retained" => Some(Self::MirrorReplicaNotRetained),
            b"replica_object_not_found" => Some(Self::ReplicaObjectNotFound),
            b"proof_not_found" => Some(Self::ProofNotFound),
            b"replica_descriptor_proof_not_found" => Some(Self::ReplicaDescriptorProofNotFound),
            b"witness_target_unavailable" => Some(Self::WitnessTargetUnavailable),
            b"witness_target_capability_unavailable" => {
                Some(Self::WitnessTargetCapabilityUnavailable)
            }
            b"witness_target_rejected" => Some(Self::WitnessTargetRejected),
            b"witness_target_invalid_response" => Some(Self::WitnessTargetInvalidResponse),
            _ => None,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::ReplicaNotFound => "replica_not_found",
            Self::ReplicaRangeNotRetained => "replica_range_not_retained",
            Self::MirrorReplicaNotRetained => "mirror_replica_not_retained",
            Self::ReplicaObjectNotFound => "replica_object_not_found",
            Self::ProofNotFound => "proof_not_found",
            Self::ReplicaDescriptorProofNotFound => "replica_descriptor_proof_not_found",
            Self::WitnessTargetUnavailable => "witness_target_unavailable",
            Self::WitnessTargetCapabilityUnavailable => "witness_target_capability_unavailable",
            Self::WitnessTargetRejected => "witness_target_rejected",
            Self::WitnessTargetInvalidResponse => "witness_target_invalid_response",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryTransportFailure {
    Preflight,
    ConnectTimeout,
    RequestTimeout,
    Connect,
    Request,
}

impl DirectoryTransportFailure {
    fn from_reqwest(error: &reqwest::Error) -> Self {
        match (error.is_connect(), error.is_timeout()) {
            (true, true) => Self::ConnectTimeout,
            (false, true) => Self::RequestTimeout,
            (true, false) => Self::Connect,
            (false, false) => Self::Request,
        }
    }

    const fn outcome(self) -> Option<DirectoryReplicaTransportOutcome> {
        match self {
            Self::Preflight => None,
            Self::ConnectTimeout => Some(DirectoryReplicaTransportOutcome::ConnectTimeout),
            Self::RequestTimeout => Some(DirectoryReplicaTransportOutcome::RequestTimeout),
            Self::Connect => Some(DirectoryReplicaTransportOutcome::ConnectFailure),
            Self::Request => Some(DirectoryReplicaTransportOutcome::RequestFailure),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryFramePostError {
    Transport(DirectoryTransportFailure),
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
            Self::Transport(_) => format!("directory_{operation}_transport_failed"),
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

#[derive(Debug)]
enum DirectoryDescriptorProofRequestError {
    Post(DirectoryFramePostError),
    FailClosed(String),
}

impl DirectoryDescriptorProofRequestError {
    fn stable_reason(self, operation: &str) -> String {
        match self {
            Self::Post(error) => error.stable_reason(operation),
            Self::FailClosed(reason) => reason,
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

/// Runtime dependencies owned outside one synchronization coordinator.
///
/// [PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex] Keeping resources separate
/// from authority policy makes ownership explicit and prevents constructor
/// growth whenever another process-lifetime service is injected.
pub(crate) struct DirectoryReplicaSyncResources {
    /// Audited local replica store.
    pub(crate) store: Arc<DirectoryReplicaStore>,
    /// Shared status and retry runtime.
    pub(crate) runtime: Arc<DirectoryReplicaSyncRuntime>,
    /// Verified descriptor and route store.
    pub(crate) peer_store: Arc<PeerStore>,
    /// Local signing identity.
    pub(crate) identity: Arc<IdentityKeyPair>,
    /// Hardened process-lifetime Directory transport.
    pub(crate) client: reqwest::Client,
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
    witness_carrier_capabilities: DirectoryMirrorCarrierCapabilityCache,
    policy_anchor_capabilities: DirectoryWitnessCapabilityCache,
    mirror_carrier_capabilities: DirectoryMirrorCarrierCapabilityCache,
    witness_min_verified: usize,
    restored_retry_states: usize,
    full_node_mirror_enabled: bool,
    full_node_mirror_max_producers: usize,
    mirror_round_cursor: AtomicU64,
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod carrier_recovery;
mod chain_pull;
mod descriptor_admission;
mod directory_transport;
mod observation_witness;
mod sync_schedule;

use carrier_recovery::directory_carrier_recovery_disposition;
use carrier_recovery::directory_mirror_carrier_capability_unavailable;
use carrier_recovery::directory_mirror_carrier_smoke_failure_bucket;
use carrier_recovery::directory_mirror_failure_allows_recovery;
use carrier_recovery::directory_mirror_recovery_carriers;
use carrier_recovery::directory_mirror_recovery_carriers_with_policy;
use carrier_recovery::directory_mirror_recovery_carriers_with_requirement;
use carrier_recovery::directory_sync_failure_allows_carrier_fallback;
pub(crate) use carrier_recovery::run_directory_carrier_cold_bootstrap_smoke;
pub(crate) use carrier_recovery::run_directory_mirror_carrier_smoke;
use carrier_recovery::verify_directory_mirror_carrier_smoke_candidate;
use chain_pull::hydrate_directory_descriptor_objects;
use chain_pull::hydrate_directory_replica_descriptor_objects;
use chain_pull::hydrate_directory_replica_descriptor_objects_tracked;
use chain_pull::import_directory_mirror_range_page;
use chain_pull::import_directory_range_page;
use chain_pull::pull_directory_chain_mirror_page;
use chain_pull::pull_directory_chain_mirror_page_via_carrier;
use chain_pull::pull_directory_chain_mirror_page_with_recovery;
pub use chain_pull::pull_directory_chain_page;
use chain_pull::pull_directory_chain_page_via_carrier;
use chain_pull::pull_directory_chain_page_with_carriers;
use chain_pull::pull_directory_chain_pinned_page_via_discovered_carrier;
use chain_pull::request_directory_block_page;
use chain_pull::request_directory_replica_block_page;
pub(crate) use chain_pull::verify_block_range_response;
pub(crate) use chain_pull::verify_descriptor_objects_response;
pub(crate) use chain_pull::verify_replica_block_range_response;
pub(crate) use chain_pull::verify_replica_descriptor_objects_response;
pub use descriptor_admission::admit_directory_authenticated_descriptor;
pub use descriptor_admission::admit_directory_gossip_descriptor;
use descriptor_admission::directory_authenticated_peer_store_error;
use descriptor_admission::directory_authenticated_transport_summary_valid;
use descriptor_admission::directory_descriptor_inclusion_proof_url;
use descriptor_admission::directory_descriptor_proof_carrier_capability_unavailable;
use descriptor_admission::directory_descriptor_proof_carrier_post_allows_next;
use descriptor_admission::directory_descriptor_proof_direct_post_allows_recovery;
use descriptor_admission::directory_descriptor_proof_direct_url_allows_recovery;
use descriptor_admission::directory_replica_descriptor_inclusion_proof_url;
pub use descriptor_admission::fetch_and_admit_directory_authenticated_descriptor;
pub use descriptor_admission::fetch_directory_descriptor_inclusion_proof_with_recovery;
use descriptor_admission::locally_audited_directory_descriptor_proof;
use descriptor_admission::request_directory_descriptor_inclusion_proof;
use descriptor_admission::request_directory_replica_descriptor_inclusion_proof;
pub(crate) use descriptor_admission::verify_descriptor_inclusion_proof_response;
use descriptor_admission::verify_locally_anchored_directory_descriptor_proof;
pub(crate) use descriptor_admission::verify_replica_descriptor_inclusion_proof_response;
pub use directory_transport::build_directory_certificate_exchange_http_client;
use directory_transport::build_hardened_directory_http_client;
use directory_transport::directory_mirror_peer_urls;
use directory_transport::directory_mirror_recovery_carrier_urls;
use directory_transport::directory_replica_carrier_urls;
use directory_transport::directory_sync_peer_urls;
use directory_transport::post_directory_frame;
use directory_transport::post_directory_frame_typed;
use directory_transport::post_directory_frame_typed_with_response_limit;
use directory_transport::record_directory_sync_transport_outcome;
use directory_transport::unix_now_secs;
use observation_witness::build_observation_witness_request;
use observation_witness::directory_observation_witness_carrier_url;
use observation_witness::directory_observation_witness_recovery_carriers;
pub use observation_witness::fetch_authenticated_observation_certificate;
use observation_witness::observation_witness_carrier_failure_allows_next;
use observation_witness::observation_witness_failure_allows_carrier;
use observation_witness::observation_witness_unavailable_recovery_outcome;
use observation_witness::request_observation_checkpoint_witness;
use observation_witness::request_observation_policy_anchor;
use observation_witness::request_observation_witness_via_carrier;
use observation_witness::should_attempt_observation_witness_catch_up;
use observation_witness::verify_and_persist_observation_witness_response;
pub(crate) use observation_witness::verify_observation_certificate_response;
pub(crate) use observation_witness::verify_observation_policy_anchor_response;
use observation_witness::verify_observation_witness_carrier_response;
pub(crate) use observation_witness::verify_observation_witness_response;
use observation_witness::witness_outcome_count;
use sync_schedule::directory_carrier_cold_bootstrap_prefix_ready;
use sync_schedule::directory_full_node_mirror_candidates;
use sync_schedule::directory_sync_failure_backoff_delay_secs;
use sync_schedule::directory_sync_next_round_delay;
use sync_schedule::directory_sync_outcome_is_checkpoint_complete;
use sync_schedule::directory_sync_request_count_for_objects;
use sync_schedule::directory_sync_startup_delay_secs;
use sync_schedule::should_continue_directory_carrier_cold_bootstrap;
use sync_schedule::should_continue_directory_mirror_catch_up;
pub(crate) use sync_schedule::should_continue_directory_replica_catch_up;

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
        // [PEER-TRANSPORT-RUNTIME 2026-07-28 by Codex] Preserve the public
        // constructor for embedders/tests while production server startup can
        // inject its process-lifetime directory transport below.
        let client = build_hardened_directory_http_client()?;
        Self::new_with_policy_and_resources(
            peers,
            interval_secs,
            DirectoryReplicaSyncResources {
                store,
                runtime,
                peer_store,
                identity,
                client,
            },
            policy,
        )
    }

    /// Builds a coordinator around process-lifetime runtime resources.
    ///
    /// Server startup uses this path so one failed transport profile blocks
    /// startup instead of silently disabling Directory Replica synchronization.
    pub(crate) fn new_with_policy_and_resources(
        peers: Vec<[u8; 32]>,
        interval_secs: u64,
        resources: DirectoryReplicaSyncResources,
        policy: DirectoryReplicaSyncPolicy,
    ) -> Result<Self, &'static str> {
        let DirectoryReplicaSyncResources {
            store,
            runtime,
            peer_store,
            identity,
            client,
        } = resources;
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
        if !peers.is_empty() && (witness_min_verified == 0 || witness_min_verified > peers.len()) {
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
        Ok(Self {
            peers: peers.into(),
            interval: Duration::from_secs(interval_secs),
            store,
            runtime,
            peer_store,
            identity,
            client,
            witness_capabilities: DirectoryWitnessCapabilityCache::default(),
            witness_carrier_capabilities: DirectoryMirrorCarrierCapabilityCache::default(),
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
                witness_catch_up_checkpoints_per_round =
                    DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND,
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
        // [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] Scope the
        // recorder around the complete coordinator round. Futures created by
        // bounded `buffer_unordered` collections are polled inside this task,
        // so deep transport helpers remain observable without threading an
        // ambient metrics argument through protocol-verification functions.
        DIRECTORY_SYNC_TRANSPORT_RUNTIME
            .scope(
                Arc::clone(&self.runtime),
                self.synchronize_round_with_transport_telemetry(),
            )
            .await
    }

    async fn synchronize_round_with_transport_telemetry(&self) -> bool {
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
            let Ok(Ok(cursors)) =
                tokio::task::spawn_blocking(move || store.retained_mirror_cursors()).await
            else {
                self.runtime.record_full_node_mirror_round(0, 0, 0, now);
                warn!(
                    reason = "directory_mirror_registry_read_failed",
                    "[DIRECTORY_REPLICA] Full-node Mirror round skipped"
                );
                return;
            };
            cursors
        };
        let pinned = self.peers.iter().copied().collect::<HashSet<_>>();
        let local = self.identity.public_key_bytes();
        let retained = retained
            .into_iter()
            .filter(|cursor| {
                if cursor.producer == local || pinned.contains(&cursor.producer) {
                    return false;
                }
                self.peer_store
                    .get_valid(&cursor.producer, now)
                    .map(|descriptor| {
                        descriptor.descriptor.policy.public_discovery
                            && descriptor
                                .descriptor
                                .public_endpoint
                                .as_deref()
                                .is_some_and(commitment_peer_endpoint_is_public)
                    })
                    // No live descriptor is the retained-expiry recovery case.
                    // A present live descriptor must still satisfy its current
                    // signed public-discovery policy.
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();
        let retained_count = retained.len();
        let live_candidates = self
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
        let mut candidates = directory_full_node_mirror_candidates(
            &retained,
            live_candidates,
            self.full_node_mirror_max_producers,
        );
        let candidate_count = candidates.len();
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
            retained = retained_count,
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
            if directory_sync_outcome_is_checkpoint_complete(&outcome, source) {
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
                Ok((outcome, source)) => {
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
                        return directory_sync_outcome_is_checkpoint_complete(&outcome, source);
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
                self.witness_mature_observation_checkpoints().await;
            }
            Ok(Err(_)) | Err(_) => {
                warn!(
                    reason = "directory_observation_checkpoint_persist_failed",
                    "[DIRECTORY_REPLICA] Complete observation checkpoint rejected"
                );
            }
        }
    }

    async fn witness_mature_observation_checkpoints(&self) {
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
        let mut checkpoints_attempted = 0usize;
        let mut previous_sequence = None;
        loop {
            let store = Arc::clone(&self.store);
            let eligible_witnesses = Arc::clone(&self.peers);
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
                Ok(Ok(None)) => break,
                Ok(Err(_)) | Err(_) => {
                    warn!(
                        reason = "directory_observation_checkpoint_audit_failed",
                        checkpoints_attempted,
                        "[DIRECTORY_REPLICA] External witness catch-up batch stopped"
                    );
                    break;
                }
            };
            let checkpoint = target.checkpoint.clone();
            if !should_attempt_observation_witness_catch_up(
                checkpoints_attempted,
                previous_sequence,
                checkpoint.sequence,
            ) {
                debug!(
                    checkpoints_attempted,
                    previous_sequence = ?previous_sequence,
                    candidate_sequence = checkpoint.sequence,
                    catch_up_budget =
                        DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND,
                    "[DIRECTORY_REPLICA] External witness catch-up batch bounded"
                );
                break;
            }
            debug!(
                checkpoint_sequence = checkpoint.sequence,
                checkpoint_age_seconds = observed_at.saturating_sub(checkpoint.observed_at),
                maturity_delay_secs,
                retained_pinned_witnesses = target.witnessed_by.len(),
                minimum_witnesses = target.minimum_witnesses,
                checkpoints_attempted,
                catch_up_budget = DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND,
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
                        self.peers.as_ref(),
                        &self.client,
                        &self.witness_capabilities,
                        &self.witness_carrier_capabilities,
                        self.runtime.as_ref(),
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
            checkpoints_attempted = checkpoints_attempted.saturating_add(1);
            previous_sequence = Some(checkpoint.sequence);
        }
        debug!(
            checkpoints_attempted,
            last_checkpoint_sequence = ?previous_sequence,
            catch_up_budget = DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND,
            "[DIRECTORY_REPLICA] External witness catch-up batch completed"
        );
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

#[derive(Debug, Clone)]
struct ObservationWitnessRequest {
    request_id: [u8; 16],
    requester: [u8; 32],
    request_timestamp: u64,
    checkpoint_sequence: u64,
    checkpoint_hash: [u8; 32],
    frame: Vec<u8>,
}

#[cfg(test)]
mod tests {
    mod carrier_recovery;
    mod chain_pull;
    mod observation_witness;
    mod other;
    mod sync_schedule;

    use super::*;
    use aeronyx_core::protocol::discovery::{DirectoryDescriptorCommitmentV1, NodeDescriptor};
    use axum::{http::StatusCode, routing::post, Router};
    use tempfile::TempDir;

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    const TEST_NOW: u64 = 1_700_000_000;

    fn carrier_hydration_test_context() -> (
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

    fn descriptor_proof_test_context() -> (
        IdentityKeyPair,
        IdentityKeyPair,
        SignedNodeDescriptor,
        DirectoryCommitmentBlockV1,
        DirectoryDescriptorInclusionProofV1,
    ) {
        let now = unix_now_secs();
        let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let carrier = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
        let descriptor = SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                subject.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now + 600,
                "descriptor-proof-recovery-test",
            ),
            &subject,
        )
        .unwrap();
        let commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap();
        let block =
            DirectoryCommitmentBlockV1::new_signed(1, now, [0u8; 32], vec![commitment], &producer)
                .unwrap();
        let proof =
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, now).unwrap();
        (producer, carrier, descriptor, block, proof)
    }

    fn descriptor_proof_replica_store(
        producer: &IdentityKeyPair,
        descriptor: &SignedNodeDescriptor,
        block: &DirectoryCommitmentBlockV1,
        observed_at: u64,
    ) -> DirectoryReplicaStore {
        let local = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
        let request_id = [0x75; 16];
        let blocks = vec![block.clone()];
        let block_hash = block.hash();
        let signing_bytes = directory_block_range_response_signing_bytes(
            &request_id,
            &producer.public_key_bytes(),
            observed_at,
            &blocks,
            false,
            block.header.height,
            &block_hash,
        );
        let response = DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            responder: producer.public_key_bytes(),
            response_timestamp: observed_at,
            blocks,
            has_more: false,
            tip_height: block.header.height,
            tip_hash: block_hash,
            signature: producer.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&response).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(":memory:", local.public_key_bytes(), observed_at).unwrap();
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(block),
                std::slice::from_ref(descriptor),
                block.header.height,
                block_hash,
                &frame,
                observed_at,
            )
            .unwrap();
        store
    }

    async fn carrier_hydration_test_endpoint(
        status: StatusCode,
        body: Vec<u8>,
    ) -> TestResult<(reqwest::Url, JoinHandle<std::io::Result<()>>)> {
        let app = Router::new().route(
            "/",
            post(move || {
                let body = body.clone();
                async move { (status, body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move { axum::serve(listener, app).await });
        Ok((reqwest::Url::parse(&format!("http://{address}/"))?, server))
    }
}
