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
//! - `POST /api/memchain/peer/checkpoint`
//! - Bincode `MemChainMessage` request/response with the existing magic byte.
//! - Signed discovery-peer admission, timestamp freshness, replay protection,
//!   per-peer rate limiting, and bounded pagination.
//! - Response signing that binds request id, block order, pagination, and tip.
//! - Default-off follower pull from one configured coordinator identity.
//! - Whole-page signature, proposer, continuity, fork, and rollback validation
//!   followed by one atomic SQLite page append.
//! - Signed tip/checkpoint comparison that distinguishes lag from a fork.
//! - Durable bounded storage of the exact verified checkpoint response before
//!   follower convergence can be reported.
//! - Bounded coordinator witness rounds that collect signed peer checkpoints
//!   as evidence without treating peer count as consensus or fork choice.
//! - Direction-isolated checkpoint telemetry: serving a requester updates only
//!   service counters and cannot manufacture local convergence or divergence.
//! - Audit-gated block pages assembled from one SQLite snapshot and
//!   canonically reverified before the node signs a response.
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
//! - Checkpoint proof establishes what a peer signed; it is not a majority,
//!   finality, leader-election, or longest-chain consensus rule.
//! - The latest bounded round summary is aggregate operational evidence only;
//!   its counts must never become voting weight or a fork-choice input.
//! - Coordinator witness failures or divergence evidence must never mutate the
//!   canonical chain; they are operator evidence until consensus is designed.
//! - Only explicit operator pins may turn signed checkpoint evidence into a
//!   startup gate. Permissionless discovery peers remain evidence-only.
//! - Never derive outbound checkpoint state from an inbound request. The peer
//!   controls its requested height/hash, so those values are not local evidence.
//! - Never sign a range assembled from separate block/tip reads or from a
//!   missing/stale process audit baseline.
//!
//! ## Last Modified
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

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use futures::StreamExt;
use rand::RngCore;
use reqwest::Url;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{
    RecordCommitmentBlockV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH,
};
use aeronyx_core::protocol::memchain::{
    decode_memchain, encode_memchain, record_block_range_request_signing_bytes,
    record_block_range_response_signing_bytes, record_chain_checkpoint_request_signing_bytes,
    record_chain_checkpoint_response_signing_bytes, MemChainMessage, MEMCHAIN_MAGIC,
};
use aeronyx_core::protocol::NodeCapability;
use sha2::{Digest, Sha256};

use crate::services::memchain::storage_ops::RecordCommitmentCheckpointEvidencePersistOutcome;
use crate::services::memchain::MemoryStorage;
use crate::services::PeerStore;

const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 512 * 1024;
const MAX_BLOCKS_PER_RESPONSE: usize = 16;
const MAX_BLOCKS_PER_RESPONSE_WIRE: u16 = 16;
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
const REPLAY_RETENTION_SECS: u64 = 120;
const MAX_PINNED_WITNESSES_PER_ROUND: usize = 3;

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
}

#[derive(Debug)]
struct VerifiedCommitmentPage {
    blocks: Vec<RecordCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
}

#[derive(Clone)]
struct MemChainPeerState {
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    guard: Arc<Mutex<PeerRequestGuard>>,
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
    fn admit(&mut self, requester: [u8; 32], request_id: [u8; 16], now: u64) -> bool {
        self.seen_requests
            .retain(|_, seen_at| now.saturating_sub(*seen_at) <= REPLAY_RETENTION_SECS);
        if self.seen_requests.contains_key(&(requester, request_id)) {
            return false;
        }

        let minute = now / 60;
        let window = self.rate_windows.entry(requester).or_default();
        if window.minute != minute {
            *window = PeerRateWindow { minute, used: 0 };
        }
        if window.used >= MAX_REQUESTS_PER_PEER_PER_MINUTE {
            return false;
        }
        window.used += 1;
        self.seen_requests.insert((requester, request_id), now);
        true
    }
}

/// Builds the signed node-to-node commitment block sync router.
#[must_use]
pub fn build_memchain_peer_router(
    storage: Arc<MemoryStorage>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
) -> Router {
    let state = MemChainPeerState {
        storage,
        peer_store,
        identity,
        guard: Arc::new(Mutex::new(PeerRequestGuard::default())),
    };
    Router::new()
        .route("/api/memchain/peer/block-range", post(block_range_handler))
        .route("/api/memchain/peer/checkpoint", post(checkpoint_handler))
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
    pull_record_commitment_checkpoint_with_witness_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        false,
    )
    .await
}

async fn pull_record_commitment_checkpoint_with_witness_policy(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    track_trusted_witness_equivocation: bool,
) -> Result<CommitmentCheckpointOutcome, String> {
    let request_timestamp = now_secs();
    let coordinator = peer_store
        .get_valid(coordinator_node_id, request_timestamp)
        .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = coordinator
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
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
            track_trusted_witness_equivocation,
        )
        .await
        .map_err(|_| "checkpoint_evidence_persist_failed".to_string())?;
    if persist_outcome == RecordCommitmentCheckpointEvidencePersistOutcome::EquivocationDetected {
        return Err("checkpoint_witness_equivocation".to_string());
    }
    Ok(outcome)
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
        checkpoint_witness_endpoint_is_public,
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
    reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        witness_node_ids,
        checkpoint_witness_endpoint_is_public,
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
    )
    .await
}

pub(crate) async fn reconcile_record_commitment_pinned_witnesses_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
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
    )
    .await
}

async fn reconcile_record_commitment_candidate_ids(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    mut candidate_ids: Vec<[u8; 32]>,
    max_witnesses: usize,
    track_trusted_witness_equivocation: bool,
) -> CommitmentReconciliationOutcome {
    let eligible_witnesses = candidate_ids.len();
    candidate_ids.truncate(max_witnesses);
    let attempted = candidate_ids.len();
    let mut outcome = CommitmentReconciliationOutcome {
        eligible_witnesses,
        attempted,
        ..CommitmentReconciliationOutcome::default()
    };
    let mut verified = Vec::with_capacity(attempted);

    for candidate_node_id in candidate_ids {
        match pull_record_commitment_checkpoint_with_witness_policy(
            storage,
            peer_store,
            identity,
            &candidate_node_id,
            client,
            track_trusted_witness_equivocation,
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
                verified.push(proof);
            }
            Err(_) => {
                outcome.failed = outcome.failed.saturating_add(1);
            }
        }
    }

    let completed_at = now_secs();
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
fn checkpoint_witness_endpoint_is_public(endpoint: &str) -> bool {
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
    let request_timestamp = now_secs();
    let coordinator = peer_store
        .get_valid(coordinator_node_id, request_timestamp)
        .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = coordinator
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
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

fn commitment_checkpoint_url(endpoint: &str) -> Result<Url, String> {
    commitment_peer_url(endpoint, "/api/memchain/peer/checkpoint")
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
        for value in 2..=MAX_REQUESTS_PER_PEER_PER_MINUTE {
            let mut request_id = [0u8; 16];
            request_id[..4].copy_from_slice(&value.to_le_bytes());
            assert!(guard.admit(peer, request_id, now));
        }
        assert!(!guard.admit(peer, [0xFF; 16], now));
        assert!(guard.admit(peer, [0xFF; 16], now + 61));
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
    }

    #[test]
    fn checkpoint_witness_endpoint_rejects_ssrf_targets() {
        assert!(checkpoint_witness_endpoint_is_public("http://8.8.8.8:8422"));
        assert!(checkpoint_witness_endpoint_is_public(
            "https://[2606:4700:4700::1111]:8422"
        ));
        for endpoint in [
            "http://127.0.0.1:8422",
            "http://10.0.0.1:8422",
            "http://100.64.0.1:8422",
            "http://169.254.1.1:8422",
            "http://172.16.0.1:8422",
            "http://192.168.1.1:8422",
            "http://198.18.0.1:8422",
            "http://203.0.113.1:8422",
            "http://node.example:8422",
            "http://[::1]:8422",
            "http://[fc00::1]:8422",
            "http://[fe80::1]:8422",
            "http://[2001:db8::1]:8422",
        ] {
            assert!(
                !checkpoint_witness_endpoint_is_public(endpoint),
                "unexpectedly accepted {endpoint}"
            );
        }
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
        let before_pull = pull_record_commitment_checkpoint(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
        )
        .await
        .unwrap();
        assert_eq!(
            before_pull.relation,
            CommitmentCheckpointRelation::RemoteAhead
        );
        assert_eq!(before_pull.local_tip_height, 0);
        assert_eq!(before_pull.remote_tip_height, 2);
        let outcome = pull_record_commitment_page(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
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
        let checkpoint = pull_record_commitment_checkpoint(
            &destination,
            &follower_peers,
            &requester_identity,
            &responder_identity.public_key_bytes(),
            &client,
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
        let error = pull_record_commitment_page(
            &destination,
            &peers,
            &requester,
            &coordinator.public_key_bytes(),
            &client,
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
            &[pinned_witness.public_key_bytes()],
            |_| true,
        )
        .await;

        assert_eq!(round.eligible_witnesses, 1);
        assert_eq!(round.attempted, 1);
        assert_eq!(round.verified, 1);
        assert_eq!(round.converged, 1);
        assert_eq!(round.failed, 0);
        assert_eq!(
            local.record_commitment_checkpoint_status().evidence_records,
            1
        );

        server.abort();
        let _ = server.await;
    }
}
