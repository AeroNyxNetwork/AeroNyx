// ============================================
// File: crates/aeronyx-server/src/api/memchain_peer.rs
// ============================================
//! # MemChain Node Peer API — Commitment Block Synchronisation
//!
//! ## Creation Reason
//! Block Sync v1 needs a node-to-node transport that is separate from the VPN
//! client tunnel and from public discovery metadata. Reusing either surface
//! would let ordinary clients enumerate commitments or couple ledger catch-up
//! to unrelated descriptor gossip.
//!
//! ## Main Functionality
//! - `POST /api/memchain/peer/block-range`
//! - `POST /api/memchain/peer/block-announce`
//! - `POST /api/memchain/peer/checkpoint`
//! - `POST /api/memchain/peer/checkpoint-certificate`
//! - `POST /api/memchain/peer/coordinator-lease`
//! - `POST /api/memchain/peer/coordinator-lease/release`
//! - Bincode `MemChainMessage` request/response with the existing magic byte.
//! - Signed discovery-peer admission, timestamp freshness, stateful-request
//!   replay protection, shared per-peer rate limiting, and bounded pagination.
//! - Idempotent signed tip hints may be retried within that shared rate limit;
//!   they only coalesce a follower wake-up and never mutate canonical state.
//! - Coordinator delivery uses a three-peer, three-attempt in-memory retry
//!   queue with bounded exponential backoff for transport and transient HTTP
//!   failures; every retry revalidates the latest signed peer endpoint.
//! - Response signing that binds request id, block order, pagination, and tip.
//! - Default-off follower pull from one configured coordinator identity.
//! - Best-effort signed tip announcements that only wake the existing verified
//!   follower pull; an announcement can never append or select a chain.
//! - Whole-page signature, proposer, continuity, fork, and rollback validation
//!   followed by one atomic SQLite page append.
//! - Signed tip/checkpoint comparison that distinguishes lag from a fork.
//! - Durable bounded storage of the exact verified checkpoint response before
//!   follower convergence can be reported.
//! - Bounded coordinator witness rounds that collect signed peer checkpoints
//!   as evidence without treating peer count as consensus or fork choice.
//! - Operator-pinned divergent checkpoints become durable storage incidents;
//!   the verified relation still reaches startup/runtime policy unchanged.
//! - Direction-isolated checkpoint telemetry: serving a requester updates only
//!   service counters and cannot manufacture local convergence or divergence.
//! - Audit-gated block pages assembled from one SQLite snapshot and
//!   canonically reverified before the node signs a response.
//! - Fixed-size certificate exchange between admitted peers. Imported members
//!   must still belong to the receiver's operator-pinned witness set.
//! - Last-hop public-IP validation on every outbound commitment request so a
//!   rotated signed descriptor cannot redirect the node into private services.
//! - Default-off, signed short-lived coordinator leases persisted by followers
//!   for cross-host duplicate-writer fencing.
//!
//! ## Calling Relationships
//! - Mounted by `server.rs` on the public node peer listener and local operator
//!   listener when Local-mode `MemoryStorage` is available.
//! - Reads peer pages through
//!   `MemoryStorage::get_verified_record_commitment_block_page`.
//! - Uses `PeerStore::get_valid` as the node admission boundary.
//! - Uses canonical signing bytes from `aeronyx_core::protocol::memchain`.
//! - `server.rs` runs the optional low-frequency follower and coordinator
//!   witness schedulers.
//! - Coordinator startup may restrict reconciliation to explicit operator-
//!   pinned witness identities before opening transport/API listeners.
//!
//! ## Privacy Invariant
//! This API returns only signed commitment blocks. It never returns memory
//! records, ciphertext, owners, tags, embeddings, client IPs, destinations,
//! routes, endpoints, or social graph metadata.
//!
//! ## Important Note for Next Developer
//! - Do not mount this handler without `PeerStore` admission.
//! - Do not add a JSON/debug response containing raw commitments.
//! - Do not increase body/page limits without memory and abuse testing.
//! - Sealed payload replication requires a separate owner-authorised protocol.
//! - Never fall back from the pinned coordinator to an arbitrary discovered
//!   peer. Block Sync v1 is authenticated replication, not consensus.
//! - A block announcement is an untrusted scheduling hint even after its
//!   signature is verified. It must never bypass page/checkpoint validation,
//!   failure backoff, rollback protection, or the pinned coordinator policy.
//! - Do not put the deterministic block-header hash into the stateful replay
//!   cache. A follower must be able to retry the exact signed hint after a
//!   transient pull failure; rate limiting and the capacity-one notifier bound
//!   that idempotent wake-up without weakening other anti-replay checks.
//! - Never retry permanent `4xx` or protocol-incompatible receipts. Retry work
//!   must remain process-local, bounded to pinned peers, cancellable on task
//!   shutdown, and unable to delay or roll back canonical block production.
//! - Checkpoint proof establishes what a peer signed; it is not a majority,
//!   finality, leader-election, or longest-chain consensus rule.
//! - The latest bounded round summary is aggregate operational evidence only;
//!   its counts must never become voting weight or a fork-choice input.
//! - Coordinator witness failures or divergence evidence must never mutate the
//!   canonical chain; they are operator evidence until consensus is designed.
//! - Only explicit operator pins may turn signed checkpoint evidence into a
//!   startup gate. Permissionless discovery peers remain evidence-only.
//! - A trusted divergent-prefix incident must not be converted into a generic
//!   transport failure: callers need the verified divergence to fail closed.
//! - Never derive outbound checkpoint state from an inbound request. The peer
//!   controls its requested height/hash, so those values are not local evidence.
//! - Never sign a range assembled from separate block/tip reads or from a
//!   missing/stale process audit baseline.
//! - Imported certificates are post-startup evidence only. Never let a replayed
//!   bundle satisfy the live startup witness threshold.
//! - Revalidate the resolved signed endpoint inside every pull helper. Candidate
//!   filtering alone is vulnerable to concurrent descriptor replacement.
//! - Coordinator leases require every configured witness grant. Do not describe
//!   them as permissionless consensus, Byzantine finality, or fork choice.
//!
//! ## Last Modified
//! v2.8.17-TipRetryQueue - Added bounded transient-failure delivery retries.
//! v2.8.16-IdempotentTipRetry - Allowed bounded retry of signed follower wake-ups.
//! v2.8.15-AnnouncementReceipts - Classified exact accepted, stale, and failed receipts.
//! v2.8.14-SyncObservability - Added privacy-safe authenticated announcement dispositions.
//! v2.8.13-EventDrivenFollower - Added authenticated coalesced tip wake-ups.
//! v2.8.11-CoordinatorLeaseRelease - Added authenticated graceful lease handover.
//! v2.8.10-CoordinatorLease - Added durable follower lease grants and verified client.
//! v2.8.8-EndpointSSRFGuard - Enforced final-hop public endpoint validation.
//! v2.8.7-CertificateExchange - Added admitted fixed-size certificate exchange.
//! v2.8.6-CheckpointCertificate - Require distinct pinned witnesses for certificate rounds.
//! v2.8.5-TrustedDivergenceHalt - Preserve verified divergence after sticky incident creation.
//! v2.8.3-WitnessDivergence - Exposed crate-local reconciliation for startup tests.
//! v2.8.4-WitnessEquivocation - Retain and reject conflicting pinned-witness claims.
//! v2.8.2-AdversarialFollower - Added signed malicious-page regression coverage.
//! v2.7.18-VerifiedRangeSnapshot - Sign only snapshot-consistent audited pages.
//! v2.7.17-AtomicBlockPage - Commit each verified follower page atomically.
//! v2.7.15-ExternalWitnessGuard - Added identity-pinned reconciliation.
//! v2.7.5-CheckpointProof - Signed cross-node checkpoint reconciliation.
//! v2.7.6-EvidenceVault - Fail-closed durable verified checkpoint evidence.
//! v2.7.8-CoordinatorWitness - Bounded non-consensus witness reconciliation.
//! v2.7.10-CheckpointDirectionIsolation - Isolated inbound service telemetry.
//! v2.7.12-WitnessRoundEvidence - Persist aggregate bounded-round runtime state.
//! v2.7.1-BlockFollower - Pinned coordinator pull and fail-closed page verification.
//! v2.7.0-BlockSync - Initial signed node-blind block range protocol.

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use futures::StreamExt;
use rand::RngCore;
use reqwest::Url;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{
    RecordCommitmentBlockV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH,
    MAX_RECORD_COMMITMENTS_PER_BLOCK, RECORD_COMMITMENT_BLOCK_VERSION_V1,
};
use aeronyx_core::protocol::memchain::{
    decode_memchain, encode_memchain, record_block_range_request_signing_bytes,
    record_block_range_response_signing_bytes, record_chain_checkpoint_request_signing_bytes,
    record_chain_checkpoint_response_signing_bytes, record_checkpoint_certificate_digest_v1,
    record_checkpoint_certificate_request_signing_bytes,
    record_checkpoint_certificate_response_signing_bytes,
    record_coordinator_lease_release_request_signing_bytes,
    record_coordinator_lease_release_response_signing_bytes,
    record_coordinator_lease_request_signing_bytes,
    record_coordinator_lease_response_signing_bytes, MemChainMessage,
    RecordCheckpointCertificateMemberV1, MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1,
    MAX_COORDINATOR_LEASE_TTL_SECS_V1, MEMCHAIN_MAGIC, MIN_COORDINATOR_LEASE_TTL_SECS_V1,
};
use aeronyx_core::protocol::NodeCapability;
use sha2::{Digest, Sha256};

use crate::services::memchain::storage_ops::{
    RecordCommitmentCheckpointEvidencePersistOutcome, RecordCoordinatorLeaseGrantOutcome,
    RecordCoordinatorLeaseReleaseOutcome,
};
use crate::services::memchain::{MemoryStorage, RecordCommitmentAnnouncementDisposition};
use crate::services::PeerStore;

const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 512 * 1024;
const MAX_BLOCKS_PER_RESPONSE: usize = 16;
const MAX_BLOCKS_PER_RESPONSE_WIRE: u16 = 16;
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
const REPLAY_RETENTION_SECS: u64 = 120;
const MAX_PINNED_WITNESSES_PER_ROUND: usize = 3;
const TIP_ANNOUNCEMENT_MAX_ATTEMPTS: usize = 3;
const TIP_ANNOUNCEMENT_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);

/// Aggregate result of one bounded follower pull.
///
/// No record ids, block hashes, peer endpoint, or memory metadata are exposed
/// so callers can log this structure without widening the privacy boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentSyncPageOutcome {
    /// Newly persisted blocks from this response page.
    pub inserted: usize,
    /// Blocks already present because another valid catch-up won the race.
    pub already_present: usize,
    /// Whether the signed coordinator tip extends beyond this page.
    pub has_more: bool,
    /// Privacy-safe height of the coordinator's signed chain tip.
    pub remote_tip_height: u64,
}

/// One independently verified coordinator lease grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCoordinatorLeaseGrant {
    /// Signed witness lease epoch.
    pub lease_epoch: u64,
    /// Signed witness wall-clock expiry.
    pub lease_expires_at: u64,
    /// Conservative duration between signed response time and expiry.
    pub valid_for_secs: u64,
}

/// One independently verified graceful lease release acknowledgement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCoordinatorLeaseRelease {
    /// Released witness lease generation.
    pub lease_epoch: u64,
    /// Signed witness release timestamp.
    pub released_at: u64,
}

/// Relationship proven by one valid signed checkpoint response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitmentCheckpointRelation {
    /// Both peers signed the same tip height and hash.
    Converged,
    /// The responder extends the requester's verified chain prefix.
    RemoteAhead,
    /// The responder is behind but shares its full verified prefix.
    RemoteBehind,
    /// The signed chains disagree at the shorter peer's tip.
    Diverged,
}

impl CommitmentCheckpointRelation {
    /// Stable privacy-safe status value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::RemoteAhead => "remote_ahead",
            Self::RemoteBehind => "remote_behind",
            Self::Diverged => "diverged",
        }
    }
}

/// Aggregate result of a cryptographically verified checkpoint response.
///
/// The evidence digest identifies the exact signed response for an operator
/// evidence vault without putting peer identities, hashes, or signatures into
/// logs, status APIs, or heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCheckpointOutcome {
    /// Proven relationship between the two verified chains.
    pub relation: CommitmentCheckpointRelation,
    /// Requester's tip height at proof construction.
    pub local_tip_height: u64,
    /// Responder's signed tip height.
    pub remote_tip_height: u64,
    /// Height at which the shared-prefix comparison was made.
    pub checkpoint_height: u64,
    /// SHA-256 digest of the complete signed response frame.
    pub evidence_digest: [u8; 32],
}

/// Privacy-safe aggregate result of one bounded coordinator witness round.
///
/// Counts establish only how many signed observations were collected. They do
/// not represent votes, quorum, finality, peer trust weight, or fork choice.
/// Peer identities, endpoints, hashes, signatures, and request ids remain in
/// the local evidence vault and are deliberately absent from this structure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitmentReconciliationOutcome {
    /// Valid discovered peers eligible for checkpoint observation.
    pub eligible_witnesses: usize,
    /// Peers contacted after applying the per-round bound.
    pub attempted: usize,
    /// Responses that passed identity, freshness, signature, and chain checks.
    pub verified: usize,
    /// Verified peers at the same height and hash.
    pub converged: usize,
    /// Verified peers extending the local chain prefix.
    pub remote_ahead: usize,
    /// Verified peers behind the local tip on the same prefix.
    pub remote_behind: usize,
    /// Verified peers signing a different hash at the shared height.
    pub diverged: usize,
    /// Attempts that did not establish durable signed evidence.
    pub failed: usize,
    /// Distinct certifiable pinned-witness frames in this exact round.
    pub certificate_signers: usize,
    /// Threshold requested for an immutable certificate; zero when disabled.
    pub certificate_required_signers: usize,
    /// Whether the current local tip has a re-audited immutable certificate.
    pub certificate_persisted: bool,
    /// Whether certificate persistence or its full re-audit failed.
    pub certificate_persistence_failed: bool,
}

/// Privacy-safe result of importing one independently verified certificate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitmentCertificateImportOutcome {
    /// Certified local height; hashes and witness identities remain private.
    pub checkpoint_height: u64,
    /// Distinct pinned witnesses represented by exact signed frames.
    pub signer_count: usize,
    /// Threshold embedded in the immutable certificate.
    pub required_signers: usize,
    /// Whether storage contains a fully re-audited certificate afterward.
    pub persisted: bool,
}

/// Aggregate delivery result for one best-effort commitment tip announcement.
///
/// Peer identities, endpoints, hashes, and timing remain intentionally absent
/// so the result is safe for operational logs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommitmentTipAnnouncementOutcome {
    /// Audited local tip height actually encoded in the outbound frame.
    pub announced_height: u64,
    /// Distinct operator-pinned peers considered in this bounded round.
    pub attempted: usize,
    /// Peers that accepted or coalesced the wake-up hint.
    pub accepted: usize,
    /// Peers already at or above the announced height.
    pub stale: usize,
    /// Missing, unsafe, unreachable, or incompatible peers.
    pub failed: usize,
    /// Additional HTTP attempts after an initial transient delivery failure.
    pub retries_attempted: usize,
    /// Peers that returned a terminal accepted/stale receipt after a retry.
    pub retries_succeeded: usize,
    /// Peers still transiently failing after the bounded retry budget.
    pub retries_exhausted: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitmentTipAnnouncementDelivery {
    Accepted,
    Stale,
    RetryableFailure,
    PermanentFailure,
}

// Keep the receipt contract at the HTTP wire boundary. Axum and Reqwest may
// resolve different `http` crate versions, but the protocol status codes are
// stable and must not depend on either transport library's Rust type.
fn classify_commitment_tip_announcement_status(status: u16) -> CommitmentTipAnnouncementDelivery {
    match status {
        202 => CommitmentTipAnnouncementDelivery::Accepted,
        204 => CommitmentTipAnnouncementDelivery::Stale,
        408 | 425 | 500..=599 => CommitmentTipAnnouncementDelivery::RetryableFailure,
        _ => CommitmentTipAnnouncementDelivery::PermanentFailure,
    }
}

#[derive(Debug, Clone, Copy)]
struct CommitmentTipAnnouncementRetryPolicy {
    max_attempts: usize,
    base_delay: Duration,
}

const TIP_ANNOUNCEMENT_RETRY_POLICY: CommitmentTipAnnouncementRetryPolicy =
    CommitmentTipAnnouncementRetryPolicy {
        max_attempts: TIP_ANNOUNCEMENT_MAX_ATTEMPTS,
        base_delay: TIP_ANNOUNCEMENT_RETRY_BASE_DELAY,
    };

#[derive(Debug)]
struct VerifiedCommitmentPage {
    blocks: Vec<RecordCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
}

#[derive(Debug)]
struct VerifiedCertificateMember {
    observed_at: u64,
    remote_tip_height: u64,
    evidence_digest: [u8; 32],
    frame: Vec<u8>,
}

#[derive(Debug)]
struct VerifiedCheckpointCertificate {
    checkpoint_height: u64,
    required_signers: usize,
    members: Vec<VerifiedCertificateMember>,
}

#[derive(Clone)]
struct MemChainPeerState {
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    guard: Arc<Mutex<PeerRequestGuard>>,
    lease_authorized_coordinator: Option<[u8; 32]>,
    block_announce_notifier: Option<mpsc::Sender<u64>>,
}

#[derive(Debug, Default)]
struct PeerRequestGuard {
    rate_windows: HashMap<[u8; 32], PeerRateWindow>,
    seen_requests: HashMap<([u8; 32], [u8; 16]), u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PeerRateWindow {
    minute: u64,
    used: u32,
}

impl PeerRequestGuard {
    fn prune_replay_requests(&mut self, now: u64) {
        self.seen_requests
            .retain(|_, seen_at| now.saturating_sub(*seen_at) <= REPLAY_RETENTION_SECS);
    }

    fn admit_rate_limited(&mut self, requester: [u8; 32], now: u64) -> bool {
        let minute = now / 60;
        self.rate_windows
            .retain(|_, window| window.minute >= minute.saturating_sub(1));
        let window = self.rate_windows.entry(requester).or_default();
        if window.minute != minute {
            *window = PeerRateWindow { minute, used: 0 };
        }
        if window.used >= MAX_REQUESTS_PER_PEER_PER_MINUTE {
            return false;
        }
        window.used += 1;
        true
    }

    /// Admits a stateful request exactly once inside the replay-retention window.
    fn admit(&mut self, requester: [u8; 32], request_id: [u8; 16], now: u64) -> bool {
        self.prune_replay_requests(now);
        if !self.admit_rate_limited(requester, now) {
            return false;
        }
        // Rejected replay attempts consume the same abuse budget as valid
        // requests; otherwise one signed frame could bypass the rate cap.
        if self.seen_requests.contains_key(&(requester, request_id)) {
            return false;
        }
        self.seen_requests.insert((requester, request_id), now);
        true
    }

    /// Admits an authenticated, idempotent scheduling hint within the shared cap.
    fn admit_idempotent_hint(&mut self, requester: [u8; 32], now: u64) -> bool {
        self.prune_replay_requests(now);
        self.admit_rate_limited(requester, now)
    }
}

/// Builds the signed node-to-node commitment block sync router.
#[must_use]
pub fn build_memchain_peer_router(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
) -> Router {
    build_memchain_peer_router_with_runtime(storage, peer_store, identity, None, None)
}

/// Builds the peer router with an optional follower-side lease trust root.
///
/// `lease_authorized_coordinator` must be the follower's explicitly pinned
/// Block Sync coordinator. `None` keeps the new endpoint fail-closed while all
/// existing block/checkpoint routes remain wire-compatible.
#[must_use]
pub fn build_memchain_peer_router_with_coordinator_lease(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    lease_authorized_coordinator: Option<[u8; 32]>,
) -> Router {
    build_memchain_peer_router_with_runtime(
        storage,
        peer_store,
        identity,
        lease_authorized_coordinator,
        None,
    )
}

/// Builds the peer router with follower lease and event-driven sync runtime.
///
/// The same explicitly pinned coordinator identity authorizes lease requests
/// and block announcements. The notifier is bounded by the caller; this
/// handler uses only `try_send`, so public traffic cannot stall the HTTP task.
#[must_use]
pub fn build_memchain_peer_router_with_runtime(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    lease_authorized_coordinator: Option<[u8; 32]>,
    block_announce_notifier: Option<mpsc::Sender<u64>>,
) -> Router {
    let state = MemChainPeerState {
        storage,
        peer_store,
        identity,
        guard: Arc::new(Mutex::new(PeerRequestGuard::default())),
        lease_authorized_coordinator,
        block_announce_notifier,
    };
    Router::new()
        .route(
            "/api/memchain/peer/block-announce",
            post(block_announce_handler),
        )
        .route("/api/memchain/peer/block-range", post(block_range_handler))
        .route("/api/memchain/peer/checkpoint", post(checkpoint_handler))
        .route(
            "/api/memchain/peer/checkpoint-certificate",
            post(checkpoint_certificate_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-lease",
            post(coordinator_lease_handler),
        )
        .route(
            "/api/memchain/peer/coordinator-lease/release",
            post(coordinator_lease_release_handler),
        )
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .with_state(state)
}

async fn verified_local_commitment_tip(storage: &MemoryStorage) -> Result<(u64, [u8; 32]), String> {
    let (_, checkpoint_hash, tip_height, tip_hash) = storage
        .record_commitment_chain_checkpoint(u64::MAX)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if checkpoint_hash != tip_hash {
        return Err("local_checkpoint_tip_mismatch".to_string());
    }
    Ok((tip_height, tip_hash))
}

/// Announces the current audited tip to a bounded set of operator-pinned peers.
///
/// Delivery is advisory and best effort. Followers independently authenticate
/// the pinned coordinator and then run the ordinary signed page/checkpoint
/// pull, so accepting this frame never changes their canonical chain.
///
/// # Errors
///
/// Returns an error only when the local audited tip cannot be loaded or encoded.
/// Individual peer failures are represented in the privacy-safe aggregate.
pub async fn announce_current_record_commitment_tip(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
) -> Result<CommitmentTipAnnouncementOutcome, String> {
    announce_current_record_commitment_tip_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn announce_current_record_commitment_tip_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    endpoint_allowed: &F,
) -> Result<CommitmentTipAnnouncementOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        endpoint_allowed,
        TIP_ANNOUNCEMENT_RETRY_POLICY,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    endpoint_allowed: &F,
    retry_policy: CommitmentTipAnnouncementRetryPolicy,
) -> Result<CommitmentTipAnnouncementOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let (tip_height, _) = verified_local_commitment_tip(storage).await?;
    if tip_height == 0 {
        return Err("local_commitment_tip_empty".to_string());
    }
    let page = storage
        .get_verified_record_commitment_block_page(tip_height, 1)
        .await
        .map_err(|_| "local_commitment_tip_unavailable".to_string())?;
    let block = page
        .blocks
        .into_iter()
        .next()
        .filter(|block| block.header.height == tip_height)
        .ok_or_else(|| "local_commitment_tip_unavailable".to_string())?;
    if block.header.proposer != identity.public_key_bytes() {
        return Err("local_commitment_tip_not_self_proposed".to_string());
    }
    let frame = encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
        header: block.header,
        proposer_signature: block.proposer_signature,
    })
    .map_err(|_| "tip_announcement_encode_failed".to_string())?;

    let self_node_id = identity.public_key_bytes();
    let mut distinct = HashSet::new();
    let mut outcome = CommitmentTipAnnouncementOutcome {
        announced_height: tip_height,
        ..CommitmentTipAnnouncementOutcome::default()
    };
    let mut pending = pinned_peer_ids
        .iter()
        .copied()
        .filter(|peer_id| *peer_id != self_node_id && distinct.insert(*peer_id))
        .take(MAX_PINNED_WITNESSES_PER_ROUND)
        .collect::<Vec<_>>();
    outcome.attempted = pending.len();

    // Keep the queue hard-bounded even if a future internal caller supplies a
    // malformed policy. Production uses exactly three attempts.
    let max_attempts = retry_policy.max_attempts.clamp(1, 8);
    let mut attempt_number = 1usize;
    while !pending.is_empty() {
        if attempt_number > 1 {
            let shift = u32::try_from(attempt_number.saturating_sub(2))
                .unwrap_or(u32::MAX)
                .min(7);
            let multiplier = 1u32 << shift;
            tokio::time::sleep(retry_policy.base_delay.saturating_mul(multiplier)).await;
            outcome.retries_attempted = outcome.retries_attempted.saturating_add(pending.len());
        }

        let deliveries = futures::stream::iter(pending.into_iter())
            .map(|peer_id| {
                let frame = frame.clone();
                async move {
                    let delivery = deliver_commitment_tip_announcement(
                        peer_store,
                        client,
                        peer_id,
                        frame,
                        endpoint_allowed,
                    )
                    .await;
                    (peer_id, delivery)
                }
            })
            .buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)
            .collect::<Vec<_>>()
            .await;
        let mut retry_queue = Vec::with_capacity(deliveries.len());
        for (peer_id, delivery) in deliveries {
            match delivery {
                CommitmentTipAnnouncementDelivery::Accepted => {
                    outcome.accepted = outcome.accepted.saturating_add(1);
                    if attempt_number > 1 {
                        outcome.retries_succeeded = outcome.retries_succeeded.saturating_add(1);
                    }
                }
                CommitmentTipAnnouncementDelivery::Stale => {
                    outcome.stale = outcome.stale.saturating_add(1);
                    if attempt_number > 1 {
                        outcome.retries_succeeded = outcome.retries_succeeded.saturating_add(1);
                    }
                }
                CommitmentTipAnnouncementDelivery::RetryableFailure
                    if attempt_number < max_attempts =>
                {
                    retry_queue.push(peer_id);
                }
                CommitmentTipAnnouncementDelivery::RetryableFailure => {
                    outcome.failed = outcome.failed.saturating_add(1);
                    outcome.retries_exhausted = outcome.retries_exhausted.saturating_add(1);
                }
                CommitmentTipAnnouncementDelivery::PermanentFailure => {
                    outcome.failed = outcome.failed.saturating_add(1);
                }
            }
        }
        pending = retry_queue;
        attempt_number = attempt_number.saturating_add(1);
    }
    Ok(outcome)
}

async fn deliver_commitment_tip_announcement<F>(
    peer_store: &PeerStore,
    client: &reqwest::Client,
    peer_id: [u8; 32],
    frame: Vec<u8>,
    endpoint_allowed: &F,
) -> CommitmentTipAnnouncementDelivery
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let Some(peer) = peer_store.get_valid(&peer_id, now_secs()) else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    if !endpoint_allowed(endpoint) {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    }
    let Ok(url) = commitment_block_announce_url(endpoint) else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    match client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
    {
        Ok(response) => classify_commitment_tip_announcement_status(response.status().as_u16()),
        Err(_) => CommitmentTipAnnouncementDelivery::RetryableFailure,
    }
}

async fn block_announce_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordBlockAnnounceV1 {
        header,
        proposer_signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let Some(authorized_coordinator) = state.lease_authorized_coordinator else {
        return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled");
    };
    if header.proposer != authorized_coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "coordinator_not_pinned");
    }
    let now = now_secs();
    if header.protocol_version != RECORD_COMMITMENT_BLOCK_VERSION_V1
        || header.chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || header.height == 0
        || header.timestamp == 0
        || header.timestamp > now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS)
        || header.record_count == 0
        || header.record_count as usize > MAX_RECORD_COMMITMENTS_PER_BLOCK
        || (header.height == 1 && header.prev_block_hash != GENESIS_PREV_HASH)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_block_announcement");
    }
    if state.peer_store.get_valid(&header.proposer, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let header_hash = header.hash();
    let signature_valid = IdentityPublicKey::from_bytes(&header.proposer)
        .and_then(|key| key.verify(&header_hash, &proposer_signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state
        .guard
        .lock()
        .await
        .admit_idempotent_hint(header.proposer, now)
    {
        // Keep the established wire error for older coordinators even though
        // authenticated tip hints are now rejected only by the shared rate cap.
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let (local_tip_height, _) = match verified_local_commitment_tip(&state.storage).await {
        Ok(tip) => tip,
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified"),
    };
    if header.height <= local_tip_height {
        state.storage.record_commitment_sync_announcement(
            now,
            header.height,
            RecordCommitmentAnnouncementDisposition::Stale,
        );
        return StatusCode::NO_CONTENT.into_response();
    }
    let Some(notifier) = state.block_announce_notifier.as_ref() else {
        state.storage.record_commitment_sync_announcement(
            now,
            header.height,
            RecordCommitmentAnnouncementDisposition::Unavailable,
        );
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "sync_notifier_unavailable");
    };
    match notifier.try_send(header.height) {
        Ok(()) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Accepted,
            );
            debug!(
                announced_height = header.height,
                local_tip_height, "[MEMCHAIN_BLOCK] Authenticated follower wake-up accepted"
            );
            StatusCode::ACCEPTED.into_response()
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Coalesced,
            );
            debug!(
                announced_height = header.height,
                local_tip_height, "[MEMCHAIN_BLOCK] Authenticated follower wake-up coalesced"
            );
            StatusCode::ACCEPTED.into_response()
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Unavailable,
            );
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "sync_notifier_unavailable")
        }
    }
}

/// Obtains and verifies one signed chain-checkpoint comparison from the pinned
/// coordinator. The response proves peer attestation, not network consensus.
///
/// # Errors
///
/// Returns a stable privacy-safe code when the local audited tip is
/// unavailable, the pinned peer cannot be reached, its response is invalid,
/// or durable checkpoint evidence cannot be stored.
pub async fn pull_record_commitment_checkpoint(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentCheckpointOutcome, String> {
    pull_record_commitment_checkpoint_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        false,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn pull_record_commitment_checkpoint_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    track_trusted_witness_incidents: bool,
    endpoint_allowed: &F,
) -> Result<CommitmentCheckpointOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let coordinator = peer_store
        .get_valid(coordinator_node_id, request_timestamp)
        .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = coordinator
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("pinned_coordinator_unsafe_endpoint".to_string());
    }
    let url = commitment_checkpoint_url(endpoint)?;

    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("checkpoint_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "checkpoint_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let observed_at = now_secs();
    let outcome = verify_record_commitment_checkpoint(
        storage,
        &body,
        &request_id,
        coordinator_node_id,
        (known_tip_height, known_tip_hash),
        observed_at,
    )
    .await?;
    let persist_outcome = storage
        .persist_record_commitment_checkpoint_evidence_with_witness_policy(
            observed_at,
            outcome.relation.as_str(),
            outcome.local_tip_height,
            outcome.remote_tip_height,
            outcome.checkpoint_height,
            &outcome.evidence_digest,
            &body,
            track_trusted_witness_incidents,
        )
        .await
        .map_err(|_| "checkpoint_evidence_persist_failed".to_string())?;
    if persist_outcome == RecordCommitmentCheckpointEvidencePersistOutcome::EquivocationDetected {
        return Err("checkpoint_witness_equivocation".to_string());
    }
    Ok(outcome)
}

/// Requests and verifies one short-lived lease from an operator-pinned witness.
///
/// The response authorizes only `instance_id` and the exact audited local tip.
/// It does not expose the current holder when contended and does not establish
/// permissionless consensus, fork choice, or finality.
///
/// # Errors
///
/// Returns a stable privacy-safe code for invalid policy, peer admission,
/// unsafe endpoint, contention, transport failure, stale response, tip
/// mismatch, or signature failure.
pub async fn request_record_commitment_coordinator_lease(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
    client: &reqwest::Client,
) -> Result<CommitmentCoordinatorLeaseGrant, String> {
    request_record_commitment_coordinator_lease_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        witness_node_id,
        instance_id,
        requested_ttl_secs,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Releases one previously acquired witness lease during graceful shutdown.
///
/// A failed or partial release is safe: the unreleased witnesses retain their
/// short expiry and the next process remains fail-closed until it can acquire
/// every configured grant.
pub async fn release_record_commitment_coordinator_lease(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentCoordinatorLeaseRelease, String> {
    release_record_commitment_coordinator_lease_with_endpoint_policy(
        peer_store,
        identity,
        witness_node_id,
        instance_id,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn release_record_commitment_coordinator_lease_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCoordinatorLeaseRelease, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let witness = peer_store
        .get_valid(witness_node_id, request_timestamp)
        .ok_or_else(|| "lease_release_witness_unavailable".to_string())?;
    let endpoint = witness
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "lease_release_witness_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("lease_release_witness_unsafe_endpoint".to_string());
    }
    let url = commitment_coordinator_lease_release_url(endpoint)?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let coordinator = identity.public_key_bytes();
    let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &coordinator,
        instance_id,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        coordinator,
        instance_id: *instance_id,
        request_id,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "lease_release_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("lease_release", &error))?;
    if response.status().as_u16() == StatusCode::CONFLICT.as_u16() {
        return Err("lease_release_not_holder".to_string());
    }
    if !response.status().is_success() {
        return Err(format!(
            "lease_release_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    verify_record_commitment_coordinator_lease_release_response(
        &body,
        &request_id,
        &coordinator,
        instance_id,
        witness_node_id,
        now_secs(),
    )
}

#[allow(clippy::too_many_arguments)]
async fn request_record_commitment_coordinator_lease_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCoordinatorLeaseGrant, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
        .contains(&requested_ttl_secs)
    {
        return Err("lease_policy_invalid".to_string());
    }
    let request_timestamp = now_secs();
    let witness = peer_store
        .get_valid(witness_node_id, request_timestamp)
        .ok_or_else(|| "lease_witness_unavailable".to_string())?;
    let endpoint = witness
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "lease_witness_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("lease_witness_unsafe_endpoint".to_string());
    }
    let url = commitment_coordinator_lease_url(endpoint)?;
    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let coordinator = identity.public_key_bytes();
    let signing_bytes = record_coordinator_lease_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &coordinator,
        instance_id,
        known_tip_height,
        &known_tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        coordinator,
        instance_id: *instance_id,
        known_tip_height,
        known_tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "lease_request_encode_failed".to_string())?;
    let request_started = Instant::now();
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("lease_request", &error))?;
    if response.status().as_u16() == StatusCode::CONFLICT.as_u16() {
        return Err("lease_contended".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("lease_http_status_{}", response.status().as_u16()));
    }
    let body = read_bounded_response(response).await?;
    let mut grant = verify_record_commitment_coordinator_lease_response(
        &body,
        &request_id,
        &coordinator,
        instance_id,
        witness_node_id,
        (known_tip_height, known_tip_hash),
        requested_ttl_secs,
        now_secs(),
    )?;
    grant.valid_for_secs = grant
        .valid_for_secs
        .saturating_sub(request_started.elapsed().as_secs());
    if grant.valid_for_secs == 0 {
        return Err("lease_expired_in_transit".to_string());
    }
    Ok(grant)
}

/// Pulls and imports one current-tip certificate from an admitted peer.
///
/// The serving peer is transport only. Every historical member must verify as
/// an exact signed checkpoint frame from a distinct identity in
/// `allowed_witnesses`. `minimum_required_signers` is the receiver's current
/// operator policy and cannot be downgraded by the serving peer. Callers must
/// use this after startup; a replayed bundle never replaces the live startup
/// witness round.
///
/// # Errors
///
/// Returns a stable privacy-safe code when peer admission, transport, outer
/// freshness, member signatures, operator pinning, digest, or durable storage
/// verification fails.
pub async fn pull_record_commitment_checkpoint_certificate(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
) -> Result<CommitmentCertificateImportOutcome, String> {
    pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn pull_record_commitment_checkpoint_certificate_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCertificateImportOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&minimum_required_signers)
        || allowed_witnesses.len() < minimum_required_signers
    {
        return Err("certificate_policy_invalid".to_string());
    }
    let request_timestamp = now_secs();
    let source = peer_store
        .get_valid(source_node_id, request_timestamp)
        .ok_or_else(|| "certificate_source_unavailable".to_string())?;
    let endpoint = source
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "certificate_source_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("certificate_source_unsafe_endpoint".to_string());
    }
    let url = commitment_checkpoint_certificate_url(endpoint)?;
    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    if known_tip_height == 0 {
        return Err("certificate_local_tip_unavailable".to_string());
    }

    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_checkpoint_certificate_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("certificate_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "certificate_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let verified = verify_checkpoint_certificate_response(
        &body,
        &request_id,
        source_node_id,
        (known_tip_height, known_tip_hash),
        allowed_witnesses,
        minimum_required_signers,
        now_secs(),
    )?;

    let mut evidence_digests = Vec::with_capacity(verified.members.len());
    for member in &verified.members {
        let relation = if member.remote_tip_height == verified.checkpoint_height {
            "converged"
        } else {
            "remote_ahead"
        };
        let persist_outcome = storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                member.observed_at,
                relation,
                verified.checkpoint_height,
                member.remote_tip_height,
                verified.checkpoint_height,
                &member.evidence_digest,
                &member.frame,
                true,
            )
            .await
            .map_err(|_| "certificate_member_persist_failed".to_string())?;
        if persist_outcome != RecordCommitmentCheckpointEvidencePersistOutcome::Stored {
            return Err("certificate_member_security_incident".to_string());
        }
        evidence_digests.push(member.evidence_digest);
    }
    let persisted = storage
        .persist_record_commitment_checkpoint_certificate(
            now_secs(),
            verified.required_signers,
            allowed_witnesses,
            &evidence_digests,
        )
        .await
        .map_err(|_| "certificate_persist_failed".to_string())?;
    Ok(CommitmentCertificateImportOutcome {
        checkpoint_height: verified.checkpoint_height,
        signer_count: verified.members.len(),
        required_signers: verified.required_signers,
        persisted,
    })
}

/// Collects a bounded set of signed checkpoint observations from discovered
/// encrypted-storage peers.
///
/// This is evidence collection for the current single-writer Block Sync v1
/// architecture, not distributed consensus. The function never adopts a
/// remote chain, changes the coordinator, selects a longest chain, or derives
/// truth from peer count. Every accepted response is independently verified
/// and durably stored by `pull_record_commitment_checkpoint`.
pub async fn reconcile_record_commitment_witnesses(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    max_witnesses: usize,
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        max_witnesses,
        commitment_peer_endpoint_is_public,
    )
    .await
}

/// Collects checkpoints only from explicit operator-pinned identities.
///
/// This is the trust boundary used by the coordinator startup guard. Signed
/// discovery still resolves endpoint rotation, but no unpinned permissionless
/// peer can become startup authority merely by advertising a capability.
pub async fn reconcile_record_commitment_pinned_witnesses(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        2,
    )
    .await
}

/// Collects pinned witness proofs and attempts one immutable certificate using
/// the operator's configured minimum. Values below two preserve legacy
/// one-witness startup behavior but cannot be represented as a multi-witness
/// certificate.
pub async fn reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_certificate_signers: usize,
) -> CommitmentReconciliationOutcome {
    reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        minimum_certificate_signers,
        commitment_peer_endpoint_is_public,
    )
    .await
}

async fn reconcile_record_commitment_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    max_witnesses: usize,
    endpoint_allowed: F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync,
{
    let now = now_secs();
    let self_node_id = identity.public_key_bytes();
    let mut candidates: Vec<_> = peer_store
        .peers_with_capability(NodeCapability::EncryptedStorage, now)
        .into_iter()
        .filter(|candidate| candidate.descriptor.node_id != self_node_id)
        .filter(|candidate| {
            candidate
                .descriptor
                .public_endpoint
                .as_deref()
                .is_some_and(&endpoint_allowed)
        })
        .collect();
    candidates.sort_by_key(|candidate| candidate.descriptor.node_id);
    if !candidates.is_empty() {
        // Rotate the deterministic signed-descriptor order so a larger network
        // does not permanently starve peers beyond the per-round fan-out cap.
        // The selector is local scheduling state only and is never reported.
        let node_selector = u64::from_be_bytes(
            self_node_id[..8]
                .try_into()
                .expect("fixed identity prefix length"),
        );
        let offset =
            usize::try_from((node_selector ^ (now / 300)) % candidates.len() as u64).unwrap_or(0);
        candidates.rotate_left(offset);
    }

    let candidate_ids = candidates
        .into_iter()
        .map(|candidate| candidate.descriptor.node_id)
        .collect();
    reconcile_record_commitment_candidate_ids(
        storage,
        peer_store,
        identity,
        client,
        candidate_ids,
        max_witnesses,
        false,
        None,
        &endpoint_allowed,
    )
    .await
}

pub(crate) async fn reconcile_record_commitment_pinned_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    minimum_certificate_signers: usize,
    endpoint_allowed: F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync,
{
    let now = now_secs();
    let self_node_id = identity.public_key_bytes();
    let mut candidate_ids = Vec::with_capacity(witness_node_ids.len());
    for node_id in witness_node_ids {
        if *node_id == self_node_id || candidate_ids.contains(node_id) {
            continue;
        }
        let Some(peer) = peer_store.get_valid(node_id, now) else {
            continue;
        };
        if peer
            .descriptor
            .public_endpoint
            .as_deref()
            .is_some_and(&endpoint_allowed)
        {
            candidate_ids.push(*node_id);
        }
    }

    reconcile_record_commitment_candidate_ids(
        storage,
        peer_store,
        identity,
        client,
        candidate_ids,
        witness_node_ids.len().min(MAX_PINNED_WITNESSES_PER_ROUND),
        true,
        Some(minimum_certificate_signers),
        &endpoint_allowed,
    )
    .await
}

async fn reconcile_record_commitment_candidate_ids<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    mut candidate_ids: Vec<[u8; 32]>,
    max_witnesses: usize,
    track_trusted_witness_incidents: bool,
    minimum_certificate_signers: Option<usize>,
    endpoint_allowed: &F,
) -> CommitmentReconciliationOutcome
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    // Preserve operator order while ensuring one identity can contribute at
    // most one request, one result, and one certificate member. This remains
    // defense in depth even when config parsing already rejects duplicate IDs.
    let mut distinct_candidates = HashSet::with_capacity(candidate_ids.len());
    candidate_ids.retain(|candidate| distinct_candidates.insert(*candidate));
    let eligible_witnesses = candidate_ids.len();
    candidate_ids.truncate(max_witnesses);
    let attempted = candidate_ids.len();
    let mut outcome = CommitmentReconciliationOutcome {
        eligible_witnesses,
        attempted,
        ..CommitmentReconciliationOutcome::default()
    };
    let certificate_witnesses = candidate_ids.clone();
    let mut verified = Vec::with_capacity(attempted);
    let mut certificate_evidence = Vec::with_capacity(attempted);

    for candidate_node_id in candidate_ids {
        match pull_record_commitment_checkpoint_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &candidate_node_id,
            client,
            track_trusted_witness_incidents,
            endpoint_allowed,
        )
        .await
        {
            Ok(proof) => {
                outcome.verified = outcome.verified.saturating_add(1);
                match proof.relation {
                    CommitmentCheckpointRelation::Converged => {
                        outcome.converged = outcome.converged.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::RemoteAhead => {
                        outcome.remote_ahead = outcome.remote_ahead.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::RemoteBehind => {
                        outcome.remote_behind = outcome.remote_behind.saturating_add(1);
                    }
                    CommitmentCheckpointRelation::Diverged => {
                        outcome.diverged = outcome.diverged.saturating_add(1);
                    }
                }
                if matches!(
                    proof.relation,
                    CommitmentCheckpointRelation::Converged
                        | CommitmentCheckpointRelation::RemoteAhead
                ) && proof.checkpoint_height == proof.local_tip_height
                {
                    certificate_evidence.push(proof.evidence_digest);
                }
                verified.push(proof);
            }
            Err(_) => {
                outcome.failed = outcome.failed.saturating_add(1);
            }
        }
    }

    let completed_at = now_secs();
    if let Some(configured_threshold) = minimum_certificate_signers {
        let threshold = configured_threshold
            .max(2)
            .min(MAX_PINNED_WITNESSES_PER_ROUND);
        outcome.certificate_signers = certificate_evidence.len();
        outcome.certificate_required_signers = threshold;
        if certificate_evidence.len() >= threshold {
            match storage
                .persist_record_commitment_checkpoint_certificate(
                    completed_at,
                    threshold,
                    &certificate_witnesses,
                    &certificate_evidence,
                )
                .await
            {
                Ok(persisted) => outcome.certificate_persisted = persisted,
                Err(_) => outcome.certificate_persistence_failed = true,
            }
        }
    }
    for _ in 0..outcome.failed {
        storage.record_commitment_checkpoint_failure(completed_at);
    }
    // Record valid proofs after failures and from least to most severe. A
    // partial transport failure must not hide valid evidence, while a signed
    // divergence must remain the final operator-visible state for the round.
    verified.sort_by_key(|proof| checkpoint_relation_priority(proof.relation));
    for proof in verified {
        storage.record_commitment_checkpoint_verified(
            completed_at,
            proof.relation.as_str(),
            proof.local_tip_height,
            proof.remote_tip_height,
        );
    }
    storage.record_commitment_checkpoint_witness_round(
        completed_at,
        outcome.eligible_witnesses,
        outcome.attempted,
        outcome.verified,
        outcome.failed,
        outcome.converged,
        outcome.remote_ahead,
        outcome.remote_behind,
        outcome.diverged,
    );

    outcome
}

/// Accepts only public IP literals for permissionless witness traffic.
///
/// A signed descriptor proves who advertised an endpoint, not that the target
/// is safe for this host to contact. Domain names are deliberately excluded to
/// prevent DNS rebinding; loopback, private, link-local, CGNAT, benchmark,
/// documentation, multicast, and reserved ranges are also rejected.
fn commitment_peer_endpoint_is_public(endpoint: &str) -> bool {
    let Ok(url) = commitment_checkpoint_url(endpoint) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(host);
    let Ok(address) = host.parse::<IpAddr>() else {
        return false;
    };
    match address {
        IpAddr::V4(address) => ipv4_is_public_unicast(address),
        IpAddr::V6(address) => ipv6_is_public_unicast(address),
    }
}

fn ipv4_is_public_unicast(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224)
}

fn ipv6_is_public_unicast(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4() {
        return ipv4_is_public_unicast(mapped);
    }
    let segments = address.segments();
    (segments[0] & 0xe000) == 0x2000 && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
}

const fn checkpoint_relation_priority(relation: CommitmentCheckpointRelation) -> u8 {
    match relation {
        CommitmentCheckpointRelation::RemoteBehind => 0,
        CommitmentCheckpointRelation::Converged => 1,
        CommitmentCheckpointRelation::RemoteAhead => 2,
        CommitmentCheckpointRelation::Diverged => 3,
    }
}

/// Pulls, verifies, and atomically appends one bounded commitment-block page.
///
/// The coordinator identity is supplied by validated operator configuration.
/// Discovery is used only to resolve that exact identity's current signed
/// endpoint; this function never selects or falls back to another peer.
///
/// # Errors
///
/// Returns a stable privacy-safe code when the local audited tip is
/// unavailable, the pinned peer cannot be reached, the signed page is invalid,
/// or the atomic local append fails closed.
pub async fn pull_record_commitment_page(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentSyncPageOutcome, String> {
    pull_record_commitment_page_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

async fn pull_record_commitment_page_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentSyncPageOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let coordinator = peer_store
        .get_valid(coordinator_node_id, request_timestamp)
        .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = coordinator
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("pinned_coordinator_unsafe_endpoint".to_string());
    }
    let url = commitment_block_range_url(endpoint)?;

    let local_tip = verified_local_commitment_tip(storage).await?;
    let from_height = local_tip.0.saturating_add(1).max(1);
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let limit = MAX_BLOCKS_PER_RESPONSE_WIRE;
    let signing_bytes = record_block_range_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height,
        limit,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("request", &error))?;
    if !response.status().is_success() {
        return Err(format!("http_status_{}", response.status().as_u16()));
    }
    let body = read_bounded_response(response).await?;
    let page = verify_record_commitment_page(
        &body,
        &request_id,
        coordinator_node_id,
        local_tip,
        now_secs(),
    )?;

    let append = storage
        .append_record_commitment_blocks_atomic(&page.blocks, Some(coordinator_node_id))
        .await
        .map_err(|_| "storage_append_rejected".to_string())?;

    Ok(CommitmentSyncPageOutcome {
        inserted: append.inserted,
        already_present: append.already_present,
        has_more: page.has_more,
        remote_tip_height: page.tip_height,
    })
}

fn commitment_block_range_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/block-range")
}

fn commitment_block_announce_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/block-announce")
}

fn commitment_checkpoint_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint")
}

fn commitment_checkpoint_certificate_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint-certificate")
}

fn commitment_coordinator_lease_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-lease")
}

fn commitment_coordinator_lease_release_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/coordinator-lease/release")
}

fn commitment_peer_url(endpoint: &str, path: &str) -> Result<Url, String> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return Err("pinned_coordinator_missing_endpoint".to_string());
    }
    if endpoint.contains("://")
        && !endpoint.starts_with("http://")
        && !endpoint.starts_with("https://")
    {
        return Err("pinned_coordinator_invalid_endpoint".to_string());
    }
    let normalized = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.to_string()
    } else {
        format!("http://{endpoint}")
    };
    let mut url = Url::parse(&normalized).map_err(|_| "pinned_coordinator_invalid_endpoint")?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.host_str().is_none()
    {
        return Err("pinned_coordinator_invalid_endpoint".to_string());
    }
    url.set_path(path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

#[allow(clippy::too_many_arguments)]
fn verify_record_commitment_coordinator_lease_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_coordinator: &[u8; 32],
    expected_instance_id: &[u8; 32],
    expected_witness: &[u8; 32],
    expected_tip: (u64, [u8; 32]),
    requested_ttl_secs: u32,
    now: u64,
) -> Result<CommitmentCoordinatorLeaseGrant, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_lease_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_lease_frame")?;
    let canonical = encode_memchain(&response).map_err(|_| "invalid_lease_frame")?;
    if canonical != body {
        return Err("noncanonical_lease_frame".to_string());
    }
    let MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height,
        witness_tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_lease_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("lease_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("lease_request_mismatch".to_string());
    }
    if coordinator != *expected_coordinator || instance_id != *expected_instance_id {
        return Err("lease_instance_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("lease_witness_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_lease_response".to_string());
    }
    if (witness_tip_height, witness_tip_hash) != expected_tip {
        return Err("lease_tip_mismatch".to_string());
    }
    if lease_epoch == 0 || lease_expires_at <= now {
        return Err("lease_expiry_invalid".to_string());
    }
    let valid_for_secs = lease_expires_at
        .checked_sub(response_timestamp)
        .ok_or_else(|| "lease_expiry_invalid".to_string())?;
    // The signed remainder can be slightly shorter than the minimum request
    // TTL when persistence and response signing cross a second boundary.
    if valid_for_secs == 0 || valid_for_secs > u64::from(requested_ttl_secs) {
        return Err("lease_duration_invalid".to_string());
    }
    let signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height,
        &witness_tip_hash,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_lease_signature".to_string())?;
    Ok(CommitmentCoordinatorLeaseGrant {
        lease_epoch,
        lease_expires_at,
        valid_for_secs,
    })
}

fn verify_record_commitment_coordinator_lease_release_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_coordinator: &[u8; 32],
    expected_instance_id: &[u8; 32],
    expected_witness: &[u8; 32],
    now: u64,
) -> Result<CommitmentCoordinatorLeaseRelease, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_lease_release_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_lease_release_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_lease_release_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_lease_release_frame".to_string());
    }
    let MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        released_at,
        lease_epoch,
        signature,
    } = response
    else {
        return Err("unexpected_lease_release_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("lease_release_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("lease_release_request_mismatch".to_string());
    }
    if coordinator != *expected_coordinator || instance_id != *expected_instance_id {
        return Err("lease_release_instance_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("lease_release_witness_mismatch".to_string());
    }
    if lease_epoch == 0 || now.abs_diff(released_at) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("lease_release_timestamp_invalid".to_string());
    }
    let signing_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        released_at,
        lease_epoch,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_lease_release_signature".to_string())?;
    Ok(CommitmentCoordinatorLeaseRelease {
        lease_epoch,
        released_at,
    })
}

fn verify_checkpoint_certificate_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    now: u64,
) -> Result<VerifiedCheckpointCertificate, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_certificate_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_certificate_frame")?;
    let canonical = encode_memchain(&response).map_err(|_| "invalid_certificate_frame")?;
    if canonical != body {
        return Err("noncanonical_certificate_frame".to_string());
    }
    let MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        certificate_digest,
        required_signers,
        members,
        signature,
    } = response
    else {
        return Err("unexpected_certificate_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("certificate_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("certificate_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("certificate_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_certificate_response".to_string());
    }
    if checkpoint_height == 0 || checkpoint_height != local_tip.0 || checkpoint_hash != local_tip.1
    {
        return Err("certificate_local_tip_mismatch".to_string());
    }
    let signer_count = members.iter().flatten().count();
    let required_signers = usize::from(required_signers);
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&required_signers)
        || signer_count < required_signers
        || signer_count > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
    {
        return Err("certificate_threshold_invalid".to_string());
    }
    if required_signers < minimum_required_signers {
        return Err("certificate_threshold_below_policy".to_string());
    }
    let signing_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        &certificate_digest,
        required_signers as u8,
        signer_count as u8,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_certificate_response_signature".to_string())?;

    let mut saw_empty_slot = false;
    let mut previous_responder = None;
    let mut digest_members = Vec::with_capacity(signer_count);
    let mut verified_members = Vec::with_capacity(signer_count);
    for slot in members {
        let Some(member) = slot else {
            saw_empty_slot = true;
            continue;
        };
        if saw_empty_slot {
            return Err("certificate_members_not_packed".to_string());
        }
        if previous_responder.is_some_and(|previous| previous >= member.responder) {
            return Err("certificate_members_not_distinct_sorted".to_string());
        }
        previous_responder = Some(member.responder);
        if !allowed_witnesses.contains(&member.responder) {
            return Err("certificate_member_not_pinned".to_string());
        }
        if member.response_timestamp > now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS) {
            return Err("certificate_member_timestamp_invalid".to_string());
        }
        if member.checkpoint_height != checkpoint_height
            || member.checkpoint_hash != checkpoint_hash
            || member.tip_height < checkpoint_height
            || (member.tip_height == checkpoint_height && member.tip_hash != checkpoint_hash)
        {
            return Err("certificate_member_claim_invalid".to_string());
        }
        let member_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &chain_id,
            &member.request_id,
            &member.responder,
            member.response_timestamp,
            member.checkpoint_height,
            &member.checkpoint_hash,
            member.tip_height,
            &member.tip_hash,
        );
        IdentityPublicKey::from_bytes(&member.responder)
            .and_then(|key| key.verify(&member_signing_bytes, &member.signature))
            .map_err(|_| "invalid_certificate_member_signature".to_string())?;
        let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id,
            request_id: member.request_id,
            responder: member.responder,
            response_timestamp: member.response_timestamp,
            checkpoint_height: member.checkpoint_height,
            checkpoint_hash: member.checkpoint_hash,
            tip_height: member.tip_height,
            tip_hash: member.tip_hash,
            signature: member.signature,
        })
        .map_err(|_| "certificate_member_encode_failed".to_string())?;
        let evidence_digest: [u8; 32] = Sha256::digest(&frame).into();
        digest_members.push((member.responder, evidence_digest));
        verified_members.push(VerifiedCertificateMember {
            observed_at: member.response_timestamp,
            remote_tip_height: member.tip_height,
            evidence_digest,
            frame,
        });
    }
    let computed_digest = record_checkpoint_certificate_digest_v1(
        &chain_id,
        checkpoint_height,
        &checkpoint_hash,
        required_signers,
        &digest_members,
    );
    if computed_digest != certificate_digest {
        return Err("certificate_digest_mismatch".to_string());
    }
    Ok(VerifiedCheckpointCertificate {
        checkpoint_height,
        required_signers,
        members: verified_members,
    })
}

async fn verify_record_commitment_checkpoint(
    storage: &MemoryStorage,
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    now: u64,
) -> Result<CommitmentCheckpointOutcome, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_checkpoint_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_checkpoint_frame")?;
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_checkpoint_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("checkpoint_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("checkpoint_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("checkpoint_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_checkpoint_response".to_string());
    }
    let response_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&response_signing_bytes, &signature))
        .map_err(|_| "invalid_checkpoint_signature".to_string())?;

    if tip_height == 0 && tip_hash != GENESIS_PREV_HASH {
        return Err("invalid_checkpoint_genesis".to_string());
    }
    let expected_checkpoint_height = local_tip.0.min(tip_height);
    if checkpoint_height != expected_checkpoint_height {
        return Err("checkpoint_height_mismatch".to_string());
    }
    if checkpoint_height == tip_height && checkpoint_hash != tip_hash {
        return Err("checkpoint_tip_inconsistent".to_string());
    }
    let (resolved_height, local_checkpoint_hash, _, _) = storage
        .record_commitment_chain_checkpoint(checkpoint_height)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if resolved_height != checkpoint_height {
        return Err("local_checkpoint_height_mismatch".to_string());
    }

    let relation = if local_checkpoint_hash != checkpoint_hash {
        CommitmentCheckpointRelation::Diverged
    } else if local_tip.0 == tip_height {
        CommitmentCheckpointRelation::Converged
    } else if local_tip.0 < tip_height {
        CommitmentCheckpointRelation::RemoteAhead
    } else {
        CommitmentCheckpointRelation::RemoteBehind
    };
    let evidence_digest: [u8; 32] = Sha256::digest(body).into();
    Ok(CommitmentCheckpointOutcome {
        relation,
        local_tip_height: local_tip.0,
        remote_tip_height: tip_height,
        checkpoint_height,
        evidence_digest,
    })
}

async fn read_bounded_response(response: reqwest::Response) -> Result<Vec<u8>, String> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BODY_BYTES as u64)
    {
        return Err("response_too_large".to_string());
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| classify_http_error("response_body", &error))?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BODY_BYTES {
            return Err("response_too_large".to_string());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn verify_record_commitment_page(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    now: u64,
) -> Result<VerifiedCommitmentPage, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_response_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_response_frame")?;
    let MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_response_message".to_string());
    };

    if request_id != *expected_request_id {
        return Err("response_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("response_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_response".to_string());
    }
    if blocks.len() > MAX_BLOCKS_PER_RESPONSE {
        return Err("response_page_too_large".to_string());
    }
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&response_signing_bytes, &signature))
        .map_err(|_| "invalid_response_signature".to_string())?;

    let (local_height, local_hash) = local_tip;
    if local_height == 0 && local_hash != GENESIS_PREV_HASH {
        return Err("invalid_local_genesis".to_string());
    }
    if tip_height < local_height {
        return Err("coordinator_rollback_detected".to_string());
    }
    if blocks.is_empty() {
        if has_more || tip_height != local_height || tip_hash != local_hash {
            return Err("empty_page_tip_mismatch".to_string());
        }
        return Ok(VerifiedCommitmentPage {
            blocks,
            has_more,
            tip_height,
        });
    }
    if tip_height <= local_height {
        return Err("unexpected_blocks_at_current_tip".to_string());
    }

    let mut expected_height = local_height.saturating_add(1);
    let mut expected_prev_hash = local_hash;
    for block in &blocks {
        if block.header.proposer != *expected_responder {
            return Err("unexpected_block_proposer".to_string());
        }
        block
            .verify(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                expected_height,
                &expected_prev_hash,
            )
            .map_err(|_| "commitment_chain_verification_failed".to_string())?;
        expected_height = expected_height.saturating_add(1);
        expected_prev_hash = block.hash();
    }

    let page_tip_height = expected_height.saturating_sub(1);
    let expected_has_more = page_tip_height < tip_height;
    if has_more != expected_has_more {
        return Err("pagination_state_mismatch".to_string());
    }
    if !has_more && (tip_height != page_tip_height || tip_hash != expected_prev_hash) {
        return Err("terminal_tip_mismatch".to_string());
    }

    Ok(VerifiedCommitmentPage {
        blocks,
        has_more,
        tip_height,
    })
}

fn classify_http_error(phase: &str, error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timeout"
    } else if error.is_connect() {
        "connect"
    } else if error.is_body() {
        "body"
    } else if error.is_decode() {
        "decode"
    } else if error.is_request() {
        "request"
    } else {
        "unknown"
    };
    format!("{phase}_{kind}")
}

fn checkpoint_certificate_member_from_frame(
    frame: &[u8],
    expected_chain_id: &[u8; 32],
) -> Result<RecordCheckpointCertificateMemberV1, String> {
    if frame.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("certificate_member_frame_invalid".to_string());
    }
    let message =
        decode_memchain(&frame[1..]).map_err(|_| "certificate_member_frame_invalid".to_string())?;
    let canonical =
        encode_memchain(&message).map_err(|_| "certificate_member_frame_invalid".to_string())?;
    if canonical != frame {
        return Err("certificate_member_frame_noncanonical".to_string());
    }
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("certificate_member_frame_unexpected".to_string());
    };
    if chain_id != *expected_chain_id {
        return Err("certificate_member_chain_mismatch".to_string());
    }
    Ok(RecordCheckpointCertificateMemberV1 {
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    })
}

async fn checkpoint_certificate_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || known_tip_height == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_certificate_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_checkpoint_certificate_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let bundle = match state
        .storage
        .record_commitment_checkpoint_certificate_bundle(known_tip_height, &known_tip_hash)
        .await
    {
        Ok(Some(bundle)) => bundle,
        Ok(None) => return protocol_error(StatusCode::NOT_FOUND, "certificate_unavailable"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited certificate export");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
        }
    };
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&bundle.required_signers)
        || bundle.member_frames.len() < bundle.required_signers
        || bundle.member_frames.len() > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
    {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
    }
    let mut members = [None; MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1];
    for (slot, frame) in members.iter_mut().zip(bundle.member_frames.iter()) {
        *slot = match checkpoint_certificate_member_from_frame(frame, &chain_id) {
            Ok(member) => Some(member),
            Err(error) => {
                warn!(error = %error, "[MEMCHAIN_BLOCK] Refused invalid certificate member");
                return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
            }
        };
    }
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        bundle.checkpoint_height,
        &bundle.checkpoint_hash,
        &bundle.certificate_digest,
        bundle.required_signers as u8,
        bundle.member_frames.len() as u8,
    );
    let response = MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height: bundle.checkpoint_height,
        checkpoint_hash: bundle.checkpoint_hash,
        certificate_digest: bundle.certificate_digest,
        required_signers: bundle.required_signers as u8,
        members,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode certificate response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        checkpoint_height = bundle.checkpoint_height,
        signer_count = bundle.member_frames.len(),
        "[MEMCHAIN_BLOCK] Served authenticated checkpoint certificate"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_lease_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id,
        coordinator,
        instance_id,
        known_tip_height,
        known_tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || instance_id.iter().all(|byte| *byte == 0)
        || !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
            .contains(&requested_ttl_secs)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_request");
    }
    if state.lease_authorized_coordinator != Some(coordinator) {
        return protocol_error(StatusCode::FORBIDDEN, "unauthorized_coordinator");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&coordinator, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_coordinator_lease_request_signing_bytes(
        &chain_id,
        &coordinator,
        &instance_id,
        known_tip_height,
        &known_tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&coordinator)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(coordinator, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let witness_tip = match verified_local_commitment_tip(&state.storage).await {
        Ok(tip) => tip,
        Err(_) => {
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "witness_tip_unavailable");
        }
    };
    if witness_tip != (known_tip_height, known_tip_hash) {
        return protocol_error(StatusCode::CONFLICT, "lease_tip_mismatch");
    }
    let grant = match state
        .storage
        .grant_record_commitment_coordinator_lease(
            &chain_id,
            &coordinator,
            &instance_id,
            known_tip_height,
            &known_tip_hash,
            now,
            requested_ttl_secs,
        )
        .await
    {
        Ok(RecordCoordinatorLeaseGrantOutcome::Granted {
            lease_epoch,
            lease_expires_at,
        }) => (lease_epoch, lease_expires_at),
        Ok(RecordCoordinatorLeaseGrantOutcome::TipMismatch) => {
            return protocol_error(StatusCode::CONFLICT, "lease_tip_mismatch");
        }
        Ok(RecordCoordinatorLeaseGrantOutcome::Contended) => {
            return protocol_error(StatusCode::CONFLICT, "lease_contended");
        }
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Coordinator lease persistence failed");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "lease_persist_failed");
        }
    };
    let witness = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        response_timestamp,
        grant.0,
        grant.1,
        witness_tip.0,
        &witness_tip.1,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        response_timestamp,
        lease_epoch: grant.0,
        lease_expires_at: grant.1,
        witness_tip_height: witness_tip.0,
        witness_tip_hash: witness_tip.1,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        lease_epoch = grant.0,
        lease_ttl_secs = requested_ttl_secs,
        "[MEMCHAIN_BLOCK] Granted authenticated coordinator lease"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn coordinator_lease_release_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id,
        coordinator,
        instance_id,
        request_id,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || instance_id.iter().all(|byte| *byte == 0) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_lease_release_request");
    }
    if state.lease_authorized_coordinator != Some(coordinator) {
        return protocol_error(StatusCode::FORBIDDEN, "unauthorized_coordinator");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&coordinator, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
        &chain_id,
        &coordinator,
        &instance_id,
        &request_id,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&coordinator)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(coordinator, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let (lease_epoch, released_at) = match state
        .storage
        .release_record_commitment_coordinator_lease(&chain_id, &coordinator, &instance_id, now)
        .await
    {
        Ok(RecordCoordinatorLeaseReleaseOutcome::Released {
            lease_epoch,
            released_at,
        }) => (lease_epoch, released_at),
        Ok(RecordCoordinatorLeaseReleaseOutcome::NotHolder) => {
            return protocol_error(StatusCode::CONFLICT, "lease_release_not_holder");
        }
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Coordinator lease release persistence failed");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "lease_release_persist_failed",
            );
        }
    };
    let witness = state.identity.public_key_bytes();
    let response_signing_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        released_at,
        lease_epoch,
    );
    let response = MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        released_at,
        lease_epoch,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(_) => return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error"),
    };
    debug!(
        lease_epoch,
        "[MEMCHAIN_BLOCK] Released authenticated coordinator lease"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn checkpoint_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || (known_tip_height == 0 && known_tip_hash != GENESIS_PREV_HASH)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_checkpoint_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let signature_valid = IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (checkpoint_height, checkpoint_hash, tip_height, tip_hash) = match state
        .storage
        .record_commitment_chain_checkpoint(known_tip_height)
        .await
    {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited checkpoint proof");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified");
        }
    };
    let relation = if known_tip_height > tip_height {
        "served"
    } else if known_tip_hash != checkpoint_hash {
        "diverged"
    } else if known_tip_height == tip_height {
        "converged"
    } else {
        "remote_behind"
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode checkpoint response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    // Count only a response that was successfully constructed. The inbound
    // request's relation/heights remain requester-controlled debug context and
    // cannot overwrite this node's outbound checkpoint evidence.
    state.storage.record_commitment_checkpoint_served(now);
    debug!(
        relation,
        checkpoint_height, tip_height, "[MEMCHAIN_BLOCK] Served authenticated chain checkpoint"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

async fn block_range_handler(State(state): State<MemChainPeerState>, body: Bytes) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id,
        from_height,
        limit,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || from_height == 0 || limit == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    let signature_valid = IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let page_limit = usize::from(limit).min(MAX_BLOCKS_PER_RESPONSE);
    let page = match state
        .storage
        .get_verified_record_commitment_block_page(from_height, page_limit)
        .await
    {
        Ok(page) => page,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unverified block range");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified");
        }
    };
    let blocks = page.blocks;
    let tip_height = page.tip_height;
    let tip_hash = page.tip_hash;
    let page_tip = blocks.last().map_or_else(
        || from_height.saturating_sub(1),
        |block| block.header.height,
    );
    let has_more = page_tip < tip_height;
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode block range response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        blocks = match &response {
            MemChainMessage::RecordBlockRangeResponseV1 { blocks, .. } => blocks.len(),
            _ => 0,
        },
        has_more, tip_height, "[MEMCHAIN_BLOCK] Served authenticated commitment range"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, axum::Json(serde_json::json!({ "error": code }))).into_response()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    use aeronyx_core::protocol::{NodeDescriptor, NodeDiscoveryMessage, SignedNodeDescriptor};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn admit_peer(
        peer_store: &PeerStore,
        identity: &IdentityKeyPair,
        endpoint: Option<String>,
        now: u64,
    ) {
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "memchain-sync-test",
        );
        descriptor.public_endpoint = endpoint;
        descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
        let descriptor = SignedNodeDescriptor::sign(descriptor, identity).unwrap();
        let import = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(import.inserted, 1);
    }

    fn allow_test_endpoint(_endpoint: &str) -> bool {
        true
    }

    fn block_announce_frame(block: &RecordCommitmentBlockV1) -> Vec<u8> {
        encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
            header: block.header.clone(),
            proposer_signature: block.proposer_signature,
        })
        .unwrap()
    }

    #[test]
    fn tip_announcement_status_contract_is_exact() {
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::ACCEPTED.as_u16()),
            CommitmentTipAnnouncementDelivery::Accepted
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::NO_CONTENT.as_u16()),
            CommitmentTipAnnouncementDelivery::Stale
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::OK.as_u16()),
            CommitmentTipAnnouncementDelivery::PermanentFailure
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::SERVICE_UNAVAILABLE.as_u16()),
            CommitmentTipAnnouncementDelivery::RetryableFailure
        );
        assert_eq!(
            classify_commitment_tip_announcement_status(StatusCode::TOO_MANY_REQUESTS.as_u16()),
            CommitmentTipAnnouncementDelivery::PermanentFailure
        );
    }

    #[tokio::test]
    async fn pinned_block_announcement_wakes_follower_without_mutating_chain() {
        let now = now_secs();
        let follower = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        storage.configure_record_commitment_sync(false, true);
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            Arc::clone(&storage),
            peer_store,
            follower,
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x31; 32]],
            &coordinator,
        );
        let frame = block_announce_frame(&block);

        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let accepted = storage.record_commitment_sync_status();
        assert_eq!(
            accepted.last_announcement_result.as_deref(),
            Some("accepted")
        );
        assert_eq!(accepted.announcements_accepted_total, 1);

        let next_block = RecordCommitmentBlockV1::new_signed(
            2,
            now,
            block.header.hash(),
            vec![[0x32; 32]],
            &coordinator,
        );
        let coalesced = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(block_announce_frame(&next_block)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(coalesced.status(), StatusCode::ACCEPTED);
        let coalesced_status = storage.record_commitment_sync_status();
        assert_eq!(
            coalesced_status.last_announcement_result.as_deref(),
            Some("coalesced")
        );
        assert_eq!(coalesced_status.last_announced_height, Some(2));
        assert_eq!(coalesced_status.announcements_accepted_total, 1);
        assert_eq!(coalesced_status.announcements_coalesced_total, 1);
        assert_eq!(notifications.recv().await, Some(1));
        assert_eq!(storage.record_commitment_chain_tip().await.0, 0);

        let retry = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(retry.status(), StatusCode::ACCEPTED);
        assert_eq!(notifications.recv().await, Some(1));
        let retry_status = storage.record_commitment_sync_status();
        assert_eq!(
            retry_status.last_announcement_result.as_deref(),
            Some("accepted")
        );
        assert_eq!(retry_status.last_announced_height, Some(2));
        assert_eq!(retry_status.announcements_accepted_total, 2);
        assert_eq!(retry_status.announcements_coalesced_total, 1);
        assert_eq!(storage.record_commitment_chain_tip().await.0, 0);
    }

    #[tokio::test]
    async fn block_announcement_rejects_unpinned_or_invalid_proposer() {
        let now = now_secs();
        let follower = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let unpinned = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        admit_peer(&peer_store, &unpinned, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            storage,
            peer_store,
            follower,
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let unpinned_block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x32; 32]],
            &unpinned,
        );
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .body(Body::from(block_announce_frame(&unpinned_block)))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert!(notifications.try_recv().is_err());

        let coordinator_block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x34; 32]],
            &coordinator,
        );
        let coordinator_block_hash = coordinator_block.hash();
        let invalid_signature = encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
            header: coordinator_block.header,
            proposer_signature: unpinned.sign(&coordinator_block_hash),
        })
        .unwrap();
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-announce")
                    .body(Body::from(invalid_signature))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(notifications.try_recv().is_err());
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_reaches_pinned_follower_runtime() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = Arc::new(IdentityKeyPair::generate());
        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x33; 32]],
            &coordinator,
        );
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let follower_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        follower_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let follower_peers = Arc::new(PeerStore::new());
        admit_peer(&follower_peers, &coordinator, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            follower_storage,
            follower_peers,
            Arc::clone(&follower),
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(outcome.announced_height, 1);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.stale, 0);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.retries_attempted, 0);
        assert_eq!(outcome.retries_succeeded, 0);
        assert_eq!(outcome.retries_exhausted, 0);
        assert_eq!(notifications.recv().await, Some(1));

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_reports_current_follower_as_stale() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = Arc::new(IdentityKeyPair::generate());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x34; 32]],
            &coordinator,
        );

        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let follower_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        follower_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        follower_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let follower_peers = Arc::new(PeerStore::new());
        admit_peer(&follower_peers, &coordinator, None, now);
        let (notifier, mut notifications) = mpsc::channel(1);
        let router = build_memchain_peer_router_with_runtime(
            follower_storage,
            follower_peers,
            Arc::clone(&follower),
            Some(coordinator.public_key_bytes()),
            Some(notifier),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(outcome.announced_height, 1);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.stale, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.retries_attempted, 0);
        assert_eq!(outcome.retries_succeeded, 0);
        assert_eq!(outcome.retries_exhausted, 0);
        assert!(notifications.try_recv().is_err());

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_retries_transient_failure_to_success() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = IdentityKeyPair::generate();
        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x35; 32]],
            &coordinator,
        );
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let handler_attempts = Arc::clone(&attempts);
        let endpoint_checks = AtomicUsize::new(0);
        let endpoint_policy = |_endpoint: &str| {
            endpoint_checks.fetch_add(1, Ordering::SeqCst);
            true
        };
        let router = Router::new().route(
            "/api/memchain/peer/block-announce",
            post(move || {
                let attempts = Arc::clone(&handler_attempts);
                async move {
                    if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        StatusCode::SERVICE_UNAVAILABLE
                    } else {
                        StatusCode::ACCEPTED
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &endpoint_policy,
            CommitmentTipAnnouncementRetryPolicy {
                max_attempts: 3,
                base_delay: Duration::ZERO,
            },
        )
        .await
        .unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(endpoint_checks.load(Ordering::SeqCst), 2);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.retries_attempted, 1);
        assert_eq!(outcome.retries_succeeded, 1);
        assert_eq!(outcome.retries_exhausted, 0);

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_tip_announcement_stops_after_retry_budget() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let follower = IdentityKeyPair::generate();
        let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x36; 32]],
            &coordinator,
        );
        source_storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        source_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let attempts = Arc::new(AtomicUsize::new(0));
        let handler_attempts = Arc::clone(&attempts);
        let router = Router::new().route(
            "/api/memchain/peer/block-announce",
            post(move || {
                let attempts = Arc::clone(&handler_attempts);
                async move {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let source_peers = PeerStore::new();
        admit_peer(
            &source_peers,
            &follower,
            Some(format!("http://{address}")),
            now,
        );
        let outcome = announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
            &source_storage,
            &source_peers,
            &coordinator,
            &reqwest::Client::new(),
            &[follower.public_key_bytes()],
            &allow_test_endpoint,
            CommitmentTipAnnouncementRetryPolicy {
                max_attempts: 3,
                base_delay: Duration::ZERO,
            },
        )
        .await
        .unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert_eq!(outcome.attempted, 1);
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.failed, 1);
        assert_eq!(outcome.retries_attempted, 2);
        assert_eq!(outcome.retries_succeeded, 0);
        assert_eq!(outcome.retries_exhausted, 1);

        server.abort();
        let _ = server.await;
    }

    #[allow(clippy::too_many_arguments)]
    fn coordinator_lease_request_frame(
        coordinator: &IdentityKeyPair,
        instance_id: [u8; 32],
        tip_height: u64,
        tip_hash: [u8; 32],
        ttl_secs: u32,
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let coordinator_id = coordinator.public_key_bytes();
        let signing_bytes = record_coordinator_lease_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &coordinator_id,
            &instance_id,
            tip_height,
            &tip_hash,
            ttl_secs,
            &request_id,
            request_timestamp,
        );
        encode_memchain(&MemChainMessage::RecordCoordinatorLeaseRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            coordinator: coordinator_id,
            instance_id,
            known_tip_height: tip_height,
            known_tip_hash: tip_hash,
            requested_ttl_secs: ttl_secs,
            request_id,
            request_timestamp,
            signature: coordinator.sign(&signing_bytes),
        })
        .unwrap()
    }

    fn coordinator_lease_release_request_frame(
        coordinator: &IdentityKeyPair,
        instance_id: [u8; 32],
        request_id: [u8; 16],
        request_timestamp: u64,
    ) -> Vec<u8> {
        let coordinator_id = coordinator.public_key_bytes();
        let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &coordinator_id,
            &instance_id,
            &request_id,
            request_timestamp,
        );
        encode_memchain(&MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            coordinator: coordinator_id,
            instance_id,
            request_id,
            request_timestamp,
            signature: coordinator.sign(&signing_bytes),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn coordinator_lease_endpoint_grants_renews_and_rejects_competing_instance() {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            Arc::clone(&storage),
            peer_store,
            Arc::clone(&witness),
            Some(coordinator.public_key_bytes()),
        );
        let first_instance = [0x71; 32];
        let first_request_id = [0x72; 16];
        let first_frame = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            first_request_id,
            now,
        );
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(first_frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let grant = verify_record_commitment_coordinator_lease_response(
            &body,
            &first_request_id,
            &coordinator.public_key_bytes(),
            &first_instance,
            &witness.public_key_bytes(),
            (0, GENESIS_PREV_HASH),
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            now,
        )
        .unwrap();
        assert_eq!(grant.lease_epoch, 1);

        let renewal = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x73; 16],
            now,
        );
        let renewed = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(renewal))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(renewed.status(), StatusCode::OK);

        let competing = coordinator_lease_request_frame(
            &coordinator,
            [0x74; 32],
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x75; 16],
            now,
        );
        let rejected = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(competing))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn coordinator_lease_endpoint_releases_exact_holder_and_hands_over_immediately() {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            storage,
            peer_store,
            Arc::clone(&witness),
            Some(coordinator.public_key_bytes()),
        );
        let first_instance = [0x76; 32];
        let second_instance = [0x77; 32];
        let acquire = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x78; 16],
            now,
        );
        let acquired = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(acquire))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(acquired.status(), StatusCode::OK);

        let wrong_release =
            coordinator_lease_release_request_frame(&coordinator, second_instance, [0x79; 16], now);
        let wrong_release_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease/release")
                    .body(Body::from(wrong_release))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_release_response.status(), StatusCode::CONFLICT);

        let release_request_id = [0x7A; 16];
        let release_frame = coordinator_lease_release_request_frame(
            &coordinator,
            first_instance,
            release_request_id,
            now,
        );
        let released = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease/release")
                    .body(Body::from(release_frame.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(released.status(), StatusCode::OK);
        let body = axum::body::to_bytes(released.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let release_ack = verify_record_commitment_coordinator_lease_release_response(
            &body,
            &release_request_id,
            &coordinator.public_key_bytes(),
            &first_instance,
            &witness.public_key_bytes(),
            now,
        )
        .unwrap();
        assert_eq!(release_ack.lease_epoch, 1);

        let replay = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease/release")
                    .body(Body::from(release_frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::TOO_MANY_REQUESTS);

        let delayed_renewal = coordinator_lease_request_frame(
            &coordinator,
            first_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x7B; 16],
            now,
        );
        let delayed = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(delayed_renewal))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(delayed.status(), StatusCode::CONFLICT);

        let takeover = coordinator_lease_request_frame(
            &coordinator,
            second_instance,
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x7C; 16],
            now,
        );
        let takeover_response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(takeover))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(takeover_response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(takeover_response.into_body(), MAX_RESPONSE_BODY_BYTES)
            .await
            .unwrap();
        let grant = verify_record_commitment_coordinator_lease_response(
            &body,
            &[0x7C; 16],
            &coordinator.public_key_bytes(),
            &second_instance,
            &witness.public_key_bytes(),
            (0, GENESIS_PREV_HASH),
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            now,
        )
        .unwrap();
        assert_eq!(grant.lease_epoch, 2);
    }

    #[tokio::test]
    async fn coordinator_lease_endpoint_rejects_unpinned_invalid_and_wrong_tip_requests() {
        let now = now_secs();
        let witness = Arc::new(IdentityKeyPair::generate());
        let coordinator = IdentityKeyPair::generate();
        let unpinned = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        storage.audit_record_commitment_chain().await.unwrap();
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &coordinator, None, now);
        admit_peer(&peer_store, &unpinned, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            storage,
            peer_store,
            witness,
            Some(coordinator.public_key_bytes()),
        );

        let unpinned_frame = coordinator_lease_request_frame(
            &unpinned,
            [0x81; 32],
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x82; 16],
            now,
        );
        let unpinned_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(unpinned_frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unpinned_response.status(), StatusCode::FORBIDDEN);

        let mut invalid_signature = coordinator_lease_request_frame(
            &coordinator,
            [0x83; 32],
            0,
            GENESIS_PREV_HASH,
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x84; 16],
            now,
        );
        let last = invalid_signature.len() - 1;
        invalid_signature[last] ^= 0x01;
        let signature_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(invalid_signature))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(signature_response.status(), StatusCode::UNAUTHORIZED);

        let wrong_tip = coordinator_lease_request_frame(
            &coordinator,
            [0x85; 32],
            1,
            [0x86; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            [0x87; 16],
            now,
        );
        let tip_response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-lease")
                    .body(Body::from(wrong_tip))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(tip_response.status(), StatusCode::CONFLICT);
    }

    #[test]
    fn coordinator_lease_response_accepts_processing_time_remainder() {
        let coordinator = IdentityKeyPair::generate();
        let witness = IdentityKeyPair::generate();
        let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
        let request_id = [0x88; 16];
        let instance_id = [0x89; 32];
        let response_timestamp = 10_001;
        let lease_expires_at = 10_060;
        let signing_bytes = record_coordinator_lease_response_signing_bytes(
            &chain_id,
            &request_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &witness.public_key_bytes(),
            response_timestamp,
            1,
            lease_expires_at,
            0,
            &GENESIS_PREV_HASH,
        );
        let frame = encode_memchain(&MemChainMessage::RecordCoordinatorLeaseResponseV1 {
            chain_id,
            request_id,
            coordinator: coordinator.public_key_bytes(),
            instance_id,
            witness: witness.public_key_bytes(),
            response_timestamp,
            lease_epoch: 1,
            lease_expires_at,
            witness_tip_height: 0,
            witness_tip_hash: GENESIS_PREV_HASH,
            signature: witness.sign(&signing_bytes),
        })
        .unwrap();

        let grant = verify_record_commitment_coordinator_lease_response(
            &frame,
            &request_id,
            &coordinator.public_key_bytes(),
            &instance_id,
            &witness.public_key_bytes(),
            (0, GENESIS_PREV_HASH),
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            response_timestamp,
        )
        .unwrap();
        assert_eq!(grant.valid_for_secs, 59);
    }

    #[tokio::test]
    async fn coordinator_lease_client_verifies_grant_release_and_immediate_handover() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let witness = Arc::new(IdentityKeyPair::generate());
        let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        witness_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let witness_peers = Arc::new(PeerStore::new());
        admit_peer(&witness_peers, &coordinator, None, now);
        let router = build_memchain_peer_router_with_coordinator_lease(
            witness_storage,
            witness_peers,
            Arc::clone(&witness),
            Some(coordinator.public_key_bytes()),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let coordinator_storage = MemoryStorage::open(":memory:", None).unwrap();
        coordinator_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();
        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &witness,
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap();
        let grant = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_storage,
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x91; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(grant.lease_epoch, 1);
        assert!(grant.valid_for_secs > 0);

        let error = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_storage,
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x92; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "lease_contended");

        let release = release_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x91; 32],
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(release.lease_epoch, 1);

        let takeover = request_record_commitment_coordinator_lease_with_endpoint_policy(
            &coordinator_storage,
            &coordinator_peers,
            &coordinator,
            &witness.public_key_bytes(),
            &[0x92; 32],
            MIN_COORDINATOR_LEASE_TTL_SECS_V1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(takeover.lease_epoch, 2);
        server.abort();
    }

    fn signed_block_page_frame(
        signer: &IdentityKeyPair,
        request_id: [u8; 16],
        response_timestamp: u64,
        blocks: Vec<RecordCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
    ) -> Vec<u8> {
        let responder = signer.public_key_bytes();
        let signing_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            response_timestamp,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: signer.sign(&signing_bytes),
        })
        .unwrap()
    }

    #[test]
    fn peer_guard_rejects_replay_and_enforces_rate_limit() {
        let peer = [0x11; 32];
        let now = 1_700_000_000;
        let mut guard = PeerRequestGuard::default();
        assert!(guard.admit(peer, [0x01; 16], now));
        assert!(!guard.admit(peer, [0x01; 16], now));
        for value in 2..MAX_REQUESTS_PER_PEER_PER_MINUTE {
            let mut request_id = [0u8; 16];
            request_id[..4].copy_from_slice(&value.to_le_bytes());
            assert!(guard.admit(peer, request_id, now));
        }
        assert!(!guard.admit(peer, [0xFF; 16], now));
        assert!(guard.admit(peer, [0xFF; 16], now + 61));
    }

    #[test]
    fn peer_guard_allows_idempotent_hint_retries_within_shared_rate_limit() {
        let peer = [0x22; 32];
        let now = 1_700_000_000;
        let mut guard = PeerRequestGuard::default();

        for _ in 0..MAX_REQUESTS_PER_PEER_PER_MINUTE {
            assert!(guard.admit_idempotent_hint(peer, now));
        }
        assert!(!guard.admit_idempotent_hint(peer, now));
        assert!(guard.admit_idempotent_hint(peer, now + 61));
    }

    #[test]
    fn commitment_range_url_is_bounded_to_the_peer_api_path() {
        let url = commitment_block_range_url("https://node.example/ignored?secret=no").unwrap();
        assert_eq!(
            url.as_str(),
            "https://node.example/api/memchain/peer/block-range"
        );
        assert!(commitment_block_range_url("ftp://node.example").is_err());
        assert!(commitment_block_range_url("https://user@node.example").is_err());
        assert_eq!(
            commitment_checkpoint_url("node.example:9281/path")
                .unwrap()
                .as_str(),
            "http://node.example:9281/api/memchain/peer/checkpoint"
        );
        assert_eq!(
            commitment_checkpoint_certificate_url("node.example:9281/path")
                .unwrap()
                .as_str(),
            "http://node.example:9281/api/memchain/peer/checkpoint-certificate"
        );
    }

    #[test]
    fn commitment_peer_endpoint_rejects_ssrf_targets() {
        assert!(commitment_peer_endpoint_is_public("http://8.8.8.8:8422"));
        assert!(commitment_peer_endpoint_is_public(
            "https://[2606:4700:4700::1111]:8422"
        ));
        for endpoint in [
            "http://127.0.0.1:8422",
            "http://127.1:8422",
            "http://2130706433:8422",
            "http://0x7f000001:8422",
            "http://017700000001:8422",
            "http://10.0.0.1:8422",
            "http://100.64.0.1:8422",
            "http://169.254.1.1:8422",
            "http://172.16.0.1:8422",
            "http://192.168.1.1:8422",
            "http://198.18.0.1:8422",
            "http://203.0.113.1:8422",
            "http://node.example:8422",
            "http://[::1]:8422",
            "http://[::ffff:127.0.0.1]:8422",
            "http://[fc00::1]:8422",
            "http://[fe80::1]:8422",
            "http://[2001:db8::1]:8422",
        ] {
            assert!(
                !commitment_peer_endpoint_is_public(endpoint),
                "unexpectedly accepted {endpoint}"
            );
        }
    }

    #[tokio::test]
    async fn outbound_commitment_pulls_reject_private_descriptor_targets() {
        let now = now_secs();
        let local_identity = IdentityKeyPair::generate();
        let remote_identity = IdentityKeyPair::generate();
        let peer_store = PeerStore::new();
        admit_peer(
            &peer_store,
            &remote_identity,
            Some("http://169.254.169.254/latest/meta-data".to_string()),
            now,
        );
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let remote_id = remote_identity.public_key_bytes();

        let checkpoint_error = pull_record_commitment_checkpoint(
            &storage,
            &peer_store,
            &local_identity,
            &remote_id,
            &client,
        )
        .await
        .unwrap_err();
        assert_eq!(checkpoint_error, "pinned_coordinator_unsafe_endpoint");

        let page_error = pull_record_commitment_page(
            &storage,
            &peer_store,
            &local_identity,
            &remote_id,
            &client,
        )
        .await
        .unwrap_err();
        assert_eq!(page_error, "pinned_coordinator_unsafe_endpoint");

        let certificate_error = pull_record_commitment_checkpoint_certificate(
            &storage,
            &peer_store,
            &local_identity,
            &remote_id,
            &[remote_id, IdentityKeyPair::generate().public_key_bytes()],
            2,
            &client,
        )
        .await
        .unwrap_err();
        assert_eq!(certificate_error, "certificate_source_unsafe_endpoint");
    }

    #[tokio::test]
    async fn witness_reconciliation_rechecks_endpoint_after_selection() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let now = now_secs();
        let local_identity = IdentityKeyPair::generate();
        let remote_identity = IdentityKeyPair::generate();
        let peer_store = PeerStore::new();
        admit_peer(
            &peer_store,
            &remote_identity,
            Some("http://8.8.8.8:8422".to_string()),
            now,
        );
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let checks = AtomicUsize::new(0);
        let round = reconcile_record_commitment_witnesses_with_endpoint_policy(
            &storage,
            &peer_store,
            &local_identity,
            &client,
            1,
            |_endpoint| checks.fetch_add(1, Ordering::SeqCst) == 0,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 1);
        assert_eq!(round.attempted, 1);
        assert_eq!(round.verified, 0);
        assert_eq!(round.failed, 1);
        assert!(checks.load(Ordering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn checkpoint_endpoint_refuses_to_sign_an_unaudited_chain() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &requester_identity, None, now);

        let request_id = [0x91; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_chain_checkpoint_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            0,
            &GENESIS_PREV_HASH,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            known_tip_height: 0,
            known_tip_hash: GENESIS_PREV_HASH,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let response = build_memchain_peer_router(storage, peer_store, responder_identity)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/checkpoint")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn block_range_endpoint_refuses_to_sign_an_unaudited_chain() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let peer_store = Arc::new(PeerStore::new());
        admit_peer(&peer_store, &requester_identity, None, now);

        let request_id = [0x92; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_block_range_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            1,
            MAX_BLOCKS_PER_RESPONSE_WIRE,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: MAX_BLOCKS_PER_RESPONSE_WIRE,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let response = build_memchain_peer_router(storage, peer_store, responder_identity)
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-range")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn response_verification_rejects_blocks_from_an_unpinned_proposer() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let other_writer = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x44; 32]],
            &other_writer,
        );
        let request_id = [0x33; 16];
        let responder = coordinator.public_key_bytes();
        let blocks = vec![block.clone()];
        let signing_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            &blocks,
            false,
            1,
            &block.hash(),
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp: now,
            blocks,
            has_more: false,
            tip_height: 1,
            tip_hash: block.hash(),
            signature: coordinator.sign(&signing_bytes),
        })
        .unwrap();

        let error = verify_record_commitment_page(
            &frame,
            &request_id,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err();
        assert_eq!(error, "unexpected_block_proposer");
    }

    #[test]
    fn response_verification_rejects_coordinator_rollback_and_fork() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let responder = coordinator.public_key_bytes();
        let request_id = [0x61; 16];
        let local_tip = (1, [0x71; 32]);

        let rollback_signing = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            &[],
            false,
            0,
            &GENESIS_PREV_HASH,
        );
        let rollback_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp: now,
            blocks: Vec::new(),
            has_more: false,
            tip_height: 0,
            tip_hash: GENESIS_PREV_HASH,
            signature: coordinator.sign(&rollback_signing),
        })
        .unwrap();
        assert_eq!(
            verify_record_commitment_page(
                &rollback_frame,
                &request_id,
                &responder,
                local_tip,
                now,
            )
            .unwrap_err(),
            "coordinator_rollback_detected"
        );

        let forked =
            RecordCommitmentBlockV1::new_signed(2, now, [0x72; 32], vec![[0x73; 32]], &coordinator);
        let fork_blocks = vec![forked.clone()];
        let fork_signing = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            &fork_blocks,
            false,
            2,
            &forked.hash(),
        );
        let fork_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
            request_id,
            responder,
            response_timestamp: now,
            blocks: fork_blocks,
            has_more: false,
            tip_height: 2,
            tip_hash: forked.hash(),
            signature: coordinator.sign(&fork_signing),
        })
        .unwrap();
        assert_eq!(
            verify_record_commitment_page(&fork_frame, &request_id, &responder, local_tip, now,)
                .unwrap_err(),
            "commitment_chain_verification_failed"
        );
    }

    #[test]
    fn response_verification_binds_signature_request_and_pagination_metadata() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let responder = coordinator.public_key_bytes();
        let request_id = [0x81; 16];
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now,
            GENESIS_PREV_HASH,
            vec![[0x82; 32]],
            &coordinator,
        );
        let valid = signed_block_page_frame(
            &coordinator,
            request_id,
            now,
            vec![block.clone()],
            false,
            1,
            block.hash(),
        );
        verify_record_commitment_page(&valid, &request_id, &responder, (0, GENESIS_PREV_HASH), now)
            .unwrap();

        let invalid_signature_bytes = record_block_range_response_signing_bytes(
            &request_id,
            &responder,
            now,
            std::slice::from_ref(&block),
            false,
            1,
            &block.hash(),
        );
        let mut invalid_signature = coordinator.sign(&invalid_signature_bytes);
        invalid_signature[0] ^= 0x01;
        let invalid_signature_frame =
            encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
                request_id,
                responder,
                response_timestamp: now,
                blocks: vec![block.clone()],
                has_more: false,
                tip_height: 1,
                tip_hash: block.hash(),
                signature: invalid_signature,
            })
            .unwrap();
        assert_eq!(
            verify_record_commitment_page(
                &invalid_signature_frame,
                &request_id,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "invalid_response_signature"
        );
        assert_eq!(
            verify_record_commitment_page(
                &valid,
                &[0x83; 16],
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "response_request_mismatch"
        );
        assert_eq!(
            verify_record_commitment_page(
                &valid,
                &request_id,
                &responder,
                (0, GENESIS_PREV_HASH),
                now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS + 1),
            )
            .unwrap_err(),
            "stale_response"
        );

        let inconsistent_tip = signed_block_page_frame(
            &coordinator,
            request_id,
            now,
            vec![block.clone()],
            false,
            1,
            [0x84; 32],
        );
        assert_eq!(
            verify_record_commitment_page(
                &inconsistent_tip,
                &request_id,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "terminal_tip_mismatch"
        );

        let inconsistent_pagination = signed_block_page_frame(
            &coordinator,
            request_id,
            now,
            vec![block],
            true,
            1,
            [0x85; 32],
        );
        assert_eq!(
            verify_record_commitment_page(
                &inconsistent_pagination,
                &request_id,
                &responder,
                (0, GENESIS_PREV_HASH),
                now,
            )
            .unwrap_err(),
            "pagination_state_mismatch"
        );
    }

    #[tokio::test]
    async fn signed_checkpoint_distinguishes_remote_lag_from_divergence() {
        let now = now_secs();
        let responder = IdentityKeyPair::generate();
        let local_writer = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x31; 32]],
            &local_writer,
        );
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x32; 32]],
            &local_writer,
        );
        storage
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();

        let request_id = [0xA2; 16];
        let responder_key = responder.public_key_bytes();
        let lagging_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder_key,
            now,
            1,
            &first.hash(),
            1,
            &first.hash(),
        );
        let lagging_frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id,
            responder: responder_key,
            response_timestamp: now,
            checkpoint_height: 1,
            checkpoint_hash: first.hash(),
            tip_height: 1,
            tip_hash: first.hash(),
            signature: responder.sign(&lagging_signing_bytes),
        })
        .unwrap();
        let lagging = verify_record_commitment_checkpoint(
            &storage,
            &lagging_frame,
            &request_id,
            &responder_key,
            (2, second.hash()),
            now,
        )
        .await
        .unwrap();
        assert_eq!(lagging.relation, CommitmentCheckpointRelation::RemoteBehind);

        let fork_hash = [0xF1; 32];
        let fork_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder_key,
            now,
            2,
            &fork_hash,
            2,
            &fork_hash,
        );
        let fork_frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id,
            responder: responder_key,
            response_timestamp: now,
            checkpoint_height: 2,
            checkpoint_hash: fork_hash,
            tip_height: 2,
            tip_hash: fork_hash,
            signature: responder.sign(&fork_signing_bytes),
        })
        .unwrap();
        let diverged = verify_record_commitment_checkpoint(
            &storage,
            &fork_frame,
            &request_id,
            &responder_key,
            (2, second.hash()),
            now,
        )
        .await
        .unwrap();
        assert_eq!(diverged.relation, CommitmentCheckpointRelation::Diverged);
    }

    #[tokio::test]
    async fn authenticated_range_sync_converges_two_commitment_ledgers() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let destination = MemoryStorage::open(":memory:", None).unwrap();

        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x11; 32], [0x22; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x33; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();
        destination.audit_record_commitment_chain().await.unwrap();

        let peer_store = Arc::new(PeerStore::new());
        let descriptor = NodeDescriptor::new(
            requester_identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now.saturating_add(600),
            "memchain-sync-test",
        );
        let descriptor = SignedNodeDescriptor::sign(descriptor, &requester_identity).unwrap();
        let import = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(import.inserted, 1);

        let request_id = [0xA7; 16];
        let requester = requester_identity.public_key_bytes();
        let signing_bytes = record_block_range_request_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            1,
            MAX_BLOCKS_PER_RESPONSE_WIRE,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: MAX_BLOCKS_PER_RESPONSE_WIRE,
            request_id,
            requester,
            request_timestamp: now,
            signature: requester_identity.sign(&signing_bytes),
        })
        .unwrap();
        let router = build_memchain_peer_router(
            Arc::clone(&source),
            peer_store,
            Arc::clone(&responder_identity),
        );
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/block-range")
                    .header(header::CONTENT_TYPE, "application/octet-stream")
                    .body(Body::from(frame))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
        let response = decode_memchain(&body[1..]).unwrap();
        let MemChainMessage::RecordBlockRangeResponseV1 {
            request_id: response_request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } = response
        else {
            panic!("expected record block range response");
        };
        assert_eq!(response_request_id, request_id);
        assert_eq!(responder, responder_identity.public_key_bytes());
        assert!(!has_more);
        assert_eq!(tip_height, 2);
        assert_eq!(tip_hash, second.hash());
        let response_signing_bytes = record_block_range_response_signing_bytes(
            &response_request_id,
            &responder,
            response_timestamp,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&response_signing_bytes, &signature)
            .unwrap();

        for block in &blocks {
            destination
                .append_record_commitment_block(block, Some(&responder))
                .await
                .unwrap();
        }
        assert_eq!(blocks, vec![first, second]);
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            source.record_commitment_chain_tip().await
        );
        let status = destination.record_commitment_chain_status().await;
        assert_eq!(status.block_count, 2);
        assert_eq!(status.commitment_count, 3);
    }

    #[tokio::test]
    async fn live_http_follower_pull_converges_with_pinned_coordinator() {
        let now = now_secs();
        let responder_identity = Arc::new(IdentityKeyPair::generate());
        let requester_identity = IdentityKeyPair::generate();
        let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

        let first = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(2),
            GENESIS_PREV_HASH,
            vec![[0x51; 32], [0x52; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let second = RecordCommitmentBlockV1::new_signed(
            2,
            now.saturating_sub(1),
            first.hash(),
            vec![[0x53; 32]],
            &responder_identity,
        );
        source
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();
        source.audit_record_commitment_chain().await.unwrap();
        destination.audit_record_commitment_chain().await.unwrap();

        let source_peers = Arc::new(PeerStore::new());
        admit_peer(&source_peers, &requester_identity, None, now);
        let router = build_memchain_peer_router(
            Arc::clone(&source),
            source_peers,
            Arc::clone(&responder_identity),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let follower_peers = PeerStore::new();
        admit_peer(
            &follower_peers,
            &responder_identity,
            Some(format!("http://{address}")),
            now,
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let before_pull = pull_record_commitment_checkpoint_with_endpoint_policy(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
            false,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(
            before_pull.relation,
            CommitmentCheckpointRelation::RemoteAhead
        );
        assert_eq!(before_pull.local_tip_height, 0);
        assert_eq!(before_pull.remote_tip_height, 2);
        let outcome = pull_record_commitment_page_with_endpoint_policy(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();

        assert_eq!(outcome.inserted, 2);
        assert_eq!(outcome.already_present, 0);
        assert!(!outcome.has_more);
        assert_eq!(outcome.remote_tip_height, 2);
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            source.record_commitment_chain_tip().await
        );
        let checkpoint = pull_record_commitment_checkpoint_with_endpoint_policy(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
            false,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(checkpoint.relation, CommitmentCheckpointRelation::Converged);
        assert_eq!(checkpoint.local_tip_height, 2);
        assert_eq!(checkpoint.remote_tip_height, 2);
        assert_eq!(checkpoint.checkpoint_height, 2);
        assert_ne!(checkpoint.evidence_digest, [0u8; 32]);
        let served = source.record_commitment_checkpoint_status();
        assert_eq!(served.requests_served_total, 2);
        assert!(served.last_served_at.is_some());
        assert_eq!(served.state, "not_checked");
        assert_eq!(served.last_checked_at, None);
        assert_eq!(served.last_divergence_at, None);
        assert_eq!(served.proofs_verified_total, 0);
        assert_eq!(served.divergences_total, 0);
        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn live_http_follower_rejects_signed_malicious_page_without_mutation() {
        let now = now_secs();
        let coordinator = Arc::new(IdentityKeyPair::generate());
        let requester = IdentityKeyPair::generate();
        let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        destination.audit_record_commitment_chain().await.unwrap();

        // The pinned coordinator signs both layers, but the block deliberately
        // forks before genesis. Envelope authenticity must never replace chain
        // continuity verification.
        let malicious_block =
            RecordCommitmentBlockV1::new_signed(1, now, [0xF1; 32], vec![[0xF2; 32]], &coordinator);
        let router = Router::new().route(
            "/api/memchain/peer/block-range",
            post({
                let coordinator = Arc::clone(&coordinator);
                move |body: Bytes| {
                    let coordinator = Arc::clone(&coordinator);
                    let malicious_block = malicious_block.clone();
                    async move {
                        assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
                        let request = decode_memchain(&body[1..]).unwrap();
                        let MemChainMessage::RecordBlockRangeRequestV1 { request_id, .. } = request
                        else {
                            panic!("expected commitment block range request");
                        };
                        let tip_hash = malicious_block.hash();
                        let frame = signed_block_page_frame(
                            &coordinator,
                            request_id,
                            now_secs(),
                            vec![malicious_block],
                            false,
                            1,
                            tip_hash,
                        );
                        (
                            StatusCode::OK,
                            [(header::CONTENT_TYPE, "application/octet-stream")],
                            frame,
                        )
                            .into_response()
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let peers = PeerStore::new();
        admit_peer(&peers, &coordinator, Some(format!("http://{address}")), now);
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let error = pull_record_commitment_page_with_endpoint_policy(
            &destination,
            &peers,
            &requester,
            &coordinator.public_key_bytes(),
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "commitment_chain_verification_failed");
        assert_eq!(
            destination.record_commitment_chain_tip().await,
            (0, GENESIS_PREV_HASH)
        );
        let status = destination.record_commitment_chain_status().await;
        assert_eq!(status.block_count, 0);
        assert_eq!(status.commitment_count, 0);

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn coordinator_witness_round_keeps_valid_proof_over_partial_failure() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let converged_witness = Arc::new(IdentityKeyPair::generate());
        let lagging_witness = Arc::new(IdentityKeyPair::generate());
        let unavailable_witness = IdentityKeyPair::generate();
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let converged = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let lagging = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x71; 32]],
            &coordinator,
        );
        local
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        converged
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        converged.audit_record_commitment_chain().await.unwrap();
        lagging.audit_record_commitment_chain().await.unwrap();

        let converged_peers = Arc::new(PeerStore::new());
        admit_peer(&converged_peers, &coordinator, None, now);
        let converged_router = build_memchain_peer_router(
            Arc::clone(&converged),
            converged_peers,
            Arc::clone(&converged_witness),
        );
        let converged_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let converged_address = converged_listener.local_addr().unwrap();
        let converged_server = tokio::spawn(async move {
            axum::serve(converged_listener, converged_router)
                .await
                .unwrap();
        });

        let lagging_peers = Arc::new(PeerStore::new());
        admit_peer(&lagging_peers, &coordinator, None, now);
        let lagging_router = build_memchain_peer_router(
            Arc::clone(&lagging),
            lagging_peers,
            Arc::clone(&lagging_witness),
        );
        let lagging_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let lagging_address = lagging_listener.local_addr().unwrap();
        let lagging_server = tokio::spawn(async move {
            axum::serve(lagging_listener, lagging_router).await.unwrap();
        });

        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &converged_witness,
            Some(format!("http://{converged_address}")),
            now,
        );
        admit_peer(
            &coordinator_peers,
            &lagging_witness,
            Some(format!("http://{lagging_address}")),
            now,
        );
        admit_peer(
            &coordinator_peers,
            &unavailable_witness,
            Some("https://[invalid".to_string()),
            now,
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            3,
            |_| true,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 3);
        assert_eq!(round.attempted, 3);
        assert_eq!(round.verified, 2);
        assert_eq!(round.converged, 1);
        assert_eq!(round.remote_behind, 1);
        assert_eq!(round.remote_ahead, 0);
        assert_eq!(round.diverged, 0);
        assert_eq!(round.failed, 1);
        let status = local.record_commitment_checkpoint_status();
        assert_eq!(status.state, "converged");
        assert_eq!(status.proofs_verified_total, 2);
        assert_eq!(status.proofs_failed_total, 1);
        assert_eq!(status.evidence_records, 2);
        assert_eq!(status.evidence_state, "verified");
        assert_eq!(status.last_round_state, "partial");
        assert_eq!(status.last_round_eligible, 3);
        assert_eq!(status.last_round_attempted, 3);
        assert_eq!(status.last_round_verified, 2);
        assert_eq!(status.last_round_failed, 1);
        assert_eq!(status.last_round_converged, 1);
        assert_eq!(status.last_round_remote_ahead, 0);
        assert_eq!(status.last_round_remote_behind, 1);
        assert_eq!(status.last_round_diverged, 0);
        assert!(status.last_round_at.is_some());

        converged_server.abort();
        lagging_server.abort();
        let _ = converged_server.await;
        let _ = lagging_server.await;
    }

    #[tokio::test]
    async fn pinned_witness_round_excludes_unpinned_permissionless_peers() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let pinned_witness = Arc::new(IdentityKeyPair::generate());
        let unpinned_peer = IdentityKeyPair::generate();
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        witness_storage
            .audit_record_commitment_chain()
            .await
            .unwrap();

        let witness_peers = Arc::new(PeerStore::new());
        admit_peer(&witness_peers, &coordinator, None, now);
        let router = build_memchain_peer_router(
            Arc::clone(&witness_storage),
            witness_peers,
            Arc::clone(&pinned_witness),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        let coordinator_peers = PeerStore::new();
        admit_peer(
            &coordinator_peers,
            &pinned_witness,
            Some(format!("http://{address}")),
            now,
        );
        admit_peer(
            &coordinator_peers,
            &unpinned_peer,
            Some("https://[invalid".to_string()),
            now,
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            &[
                pinned_witness.public_key_bytes(),
                pinned_witness.public_key_bytes(),
            ],
            2,
            |_| true,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 1);
        assert_eq!(round.attempted, 1);
        assert_eq!(round.verified, 1);
        assert_eq!(round.converged, 1);
        assert_eq!(round.failed, 0);
        assert_eq!(round.certificate_signers, 1);
        assert!(!round.certificate_persisted);
        assert_eq!(
            local.record_commitment_checkpoint_status().evidence_records,
            1
        );

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn pinned_witness_round_persists_two_signer_checkpoint_certificate() {
        let now = now_secs();
        let coordinator = IdentityKeyPair::generate();
        let witnesses = [
            Arc::new(IdentityKeyPair::generate()),
            Arc::new(IdentityKeyPair::generate()),
        ];
        let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let witness_storages = [
            Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
            Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
        ];
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            now.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0x91; 32]],
            &coordinator,
        );
        local
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        local.audit_record_commitment_chain().await.unwrap();
        local
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();

        let mut addresses = Vec::new();
        let mut servers = Vec::new();
        for (storage, witness) in witness_storages.iter().zip(witnesses.iter()) {
            storage
                .append_record_commitment_block(&block, None)
                .await
                .unwrap();
            storage.audit_record_commitment_chain().await.unwrap();
            let peers = Arc::new(PeerStore::new());
            admit_peer(&peers, &coordinator, None, now);
            let router =
                build_memchain_peer_router(Arc::clone(storage), peers, Arc::clone(witness));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            addresses.push(listener.local_addr().unwrap());
            servers.push(tokio::spawn(async move {
                axum::serve(listener, router).await.unwrap();
            }));
        }

        let coordinator_peers = PeerStore::new();
        for (witness, address) in witnesses.iter().zip(addresses.iter()) {
            admit_peer(
                &coordinator_peers,
                witness,
                Some(format!("http://{address}")),
                now,
            );
        }
        let witness_ids = [
            witnesses[0].public_key_bytes(),
            witnesses[1].public_key_bytes(),
        ];
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
            &local,
            &coordinator_peers,
            &coordinator,
            &client,
            &witness_ids,
            2,
            |_| true,
        )
        .await;

        assert_eq!(round.verified, 2);
        assert_eq!(round.converged, 2);
        assert_eq!(round.certificate_signers, 2);
        assert_eq!(round.certificate_required_signers, 2);
        assert!(round.certificate_persisted);
        assert!(!round.certificate_persistence_failed);
        let status = local.record_commitment_checkpoint_status();
        assert_eq!(status.checkpoint_certificates, 1);
        assert_eq!(status.latest_certified_height, Some(1));
        assert_eq!(status.latest_certificate_signers, 2);

        let destination_identity = IdentityKeyPair::generate();
        let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        destination.audit_record_commitment_chain().await.unwrap();
        destination
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();

        let source_identity = Arc::new(coordinator);
        let source_peers = Arc::new(PeerStore::new());
        admit_peer(&source_peers, &destination_identity, None, now);
        let source_router = build_memchain_peer_router(
            Arc::clone(&local),
            source_peers,
            Arc::clone(&source_identity),
        );
        let source_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let source_address = source_listener.local_addr().unwrap();
        let source_server = tokio::spawn(async move {
            axum::serve(source_listener, source_router).await.unwrap();
        });

        let destination_peers = PeerStore::new();
        admit_peer(
            &destination_peers,
            &source_identity,
            Some(format!("http://{source_address}")),
            now,
        );
        let imported = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &witness_ids,
            2,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
        assert_eq!(imported.checkpoint_height, 1);
        assert_eq!(imported.signer_count, 2);
        assert_eq!(imported.required_signers, 2);
        assert!(imported.persisted);
        assert_eq!(
            destination
                .record_commitment_checkpoint_status()
                .checkpoint_certificates,
            1
        );

        let third_witness = IdentityKeyPair::generate().public_key_bytes();
        let strict_witness_ids = [witness_ids[0], witness_ids[1], third_witness];
        let error = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &strict_witness_ids,
            3,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "certificate_threshold_below_policy");

        let unpinned_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        unpinned_destination
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        unpinned_destination
            .audit_record_commitment_chain()
            .await
            .unwrap();
        unpinned_destination
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        let alternative_witness = IdentityKeyPair::generate().public_key_bytes();
        let error = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &unpinned_destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &[witness_ids[0], alternative_witness],
            2,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
        assert_eq!(error, "certificate_member_not_pinned");

        for server in servers {
            server.abort();
            let _ = server.await;
        }
        source_server.abort();
        let _ = source_server.await;
    }
}
