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
//!   before any block is appended to the local ledger.
//! - Signed tip/checkpoint comparison that distinguishes lag from a fork.
//!
//! ## Calling Relationships
//! - Mounted by `server.rs` on the public node peer listener and local operator
//!   listener when Local-mode `MemoryStorage` is available.
//! - Reads verified blocks through `MemoryStorage::get_record_commitment_block_range`.
//! - Uses `PeerStore::get_valid` as the node admission boundary.
//! - Uses canonical signing bytes from `aeronyx_core::protocol::memchain`.
//! - `server.rs` runs the optional low-frequency follower scheduler.
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
//!
//! ## Last Modified
//! v2.7.5-CheckpointProof - Signed cross-node checkpoint reconciliation.
//! v2.7.1-BlockFollower - Pinned coordinator pull and fail-closed page verification.
//! v2.7.0-BlockSync - Initial signed node-blind block range protocol.

use std::collections::HashMap;
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
use sha2::{Digest, Sha256};

use crate::services::memchain::{MemoryStorage, RecordCommitmentAppendOutcome};
use crate::services::PeerStore;

const MAX_REQUEST_BODY_BYTES: usize = 16 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 512 * 1024;
const MAX_BLOCKS_PER_RESPONSE: usize = 16;
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
const REPLAY_RETENTION_SECS: u64 = 120;

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

/// Obtains and verifies one signed chain-checkpoint comparison from the pinned
/// coordinator. The response proves peer attestation, not network consensus.
pub async fn pull_record_commitment_checkpoint(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
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

    let (_, known_tip_hash, known_tip_height, current_tip_hash) = storage
        .record_commitment_chain_checkpoint(u64::MAX)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if known_tip_hash != current_tip_hash {
        return Err("local_checkpoint_tip_mismatch".to_string());
    }
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
    verify_record_commitment_checkpoint(
        storage,
        &body,
        &request_id,
        coordinator_node_id,
        (known_tip_height, known_tip_hash),
        now_secs(),
    )
    .await
}

/// Pulls, verifies, and atomically appends one bounded commitment-block page.
///
/// The coordinator identity is supplied by validated operator configuration.
/// Discovery is used only to resolve that exact identity's current signed
/// endpoint; this function never selects or falls back to another peer.
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

    let local_tip = storage.record_commitment_chain_tip().await;
    let from_height = local_tip.0.saturating_add(1).max(1);
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let limit = MAX_BLOCKS_PER_RESPONSE as u16;
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

    let mut inserted = 0usize;
    let mut already_present = 0usize;
    for block in &page.blocks {
        match storage
            .append_record_commitment_block(block, Some(coordinator_node_id))
            .await
            .map_err(|_| "storage_append_rejected".to_string())?
        {
            RecordCommitmentAppendOutcome::Inserted => inserted += 1,
            RecordCommitmentAppendOutcome::AlreadyPresent => already_present += 1,
        }
    }

    Ok(CommitmentSyncPageOutcome {
        inserted,
        already_present,
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
    state
        .storage
        .record_commitment_checkpoint_served(now, relation, tip_height, known_tip_height);

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
    let blocks = match state
        .storage
        .get_record_commitment_block_range(from_height, page_limit)
        .await
    {
        Ok(blocks) => blocks,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to read verified block range");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "storage_error");
        }
    };
    let (tip_height, tip_hash) = state.storage.record_commitment_chain_tip().await;
    let page_tip = blocks
        .last()
        .map_or(from_height.saturating_sub(1), |block| block.header.height);
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
        let descriptor = SignedNodeDescriptor::sign(descriptor, identity).unwrap();
        let import = peer_store.apply_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
            now,
        );
        assert_eq!(import.inserted, 1);
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
            MAX_BLOCKS_PER_RESPONSE as u16,
            &request_id,
            &requester,
            now,
        );
        let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: MAX_BLOCKS_PER_RESPONSE as u16,
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
        assert_eq!(served.state, "converged");
        server.abort();
        let _ = server.await;
    }
}
