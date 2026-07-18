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
//! - Preserves producer-local page and request budgets on every round.
//! - Records only bounded, privacy-safe synchronization observations.
//! - Cancels an in-flight round when server shutdown is requested.
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
//! 5. Pull pages until the request budget or 45-second deadline is exhausted.
//! 6. Stop the complete round immediately when shutdown wins the select.
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
//!
//! ## Last Modified
//! v0.3.0-DirectoryReplicaBackoff - Added producer-local round deadlines,
//! exponential retry backoff, and bounded retry scheduling telemetry.
//! v0.2.0-DirectoryReplicaClient - Owns outbound Directory Sync request,
//! verification, hydration, and import in addition to scheduling.
//! v0.1.0-DirectoryReplicaCoordinator - Extracted bounded concurrent scheduling
//! from `server.rs` and added deterministic startup synchronization jitter.
// ============================================

use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_request_signing_bytes,
    directory_block_range_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_descriptor_objects_response_signing_bytes, encode_directory_sync_message,
    DirectoryCommitmentBlockV1, DirectorySyncMessage, SignedNodeDescriptor,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_SYNC_OBJECTS_V1,
};
use futures::{stream, StreamExt};
use rand::RngCore;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::api::memchain_peer::{commitment_peer_endpoint_is_public, commitment_peer_url};
use crate::api::read_bounded_http_response;
use crate::services::{
    DirectoryReplicaImportReport, DirectoryReplicaStore, DirectoryReplicaStoreError,
    DirectoryReplicaSyncRuntime, PeerStore,
};

/// Maximum pinned producers synchronized concurrently by one node.
pub(crate) const DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS: usize = 4;
/// Hard wall-clock ceiling for one producer within a synchronization round.
pub(crate) const DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS: u64 = 45;
/// Maximum producer-local retry delay after repeated consecutive failures.
pub(crate) const DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS: u64 = 30 * 60;
/// Minimum delay before the first synchronization round after startup.
const DIRECTORY_SYNC_STARTUP_DELAY_MIN_SECS: u64 = 5;
/// Inclusive startup jitter span: 5 + (identity byte modulo 11) = 5-15 seconds.
const DIRECTORY_SYNC_STARTUP_DELAY_SPAN_SECS: u64 = 11;
/// Accepted signed response clock skew in either direction.
const DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS: u64 = 60;
/// Hard response ceiling shared with the core Directory Sync decoder.
const MAX_DIRECTORY_SYNC_RESPONSE_BODY_BYTES: usize = 512 * 1024;
/// One block per outbound page bounds object hydration and inbound rate use.
const OUTBOUND_BLOCKS_PER_PAGE: u16 = 1;
/// One range request plus maximum descriptor-object chunks for one block.
#[allow(clippy::cast_possible_truncation)]
pub(crate) const DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE: u32 =
    1 + MAX_DIRECTORY_COMMITMENTS_PER_BLOCK.div_ceil(MAX_DIRECTORY_SYNC_OBJECTS_V1) as u32;
/// Hard producer-local page cap for one low-frequency synchronization round.
pub(crate) const DIRECTORY_SYNC_MAX_PAGES_PER_ROUND: u32 = 4;
/// Leaves headroom beneath the inbound 30 requests/minute identity budget.
pub(crate) const DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND: u32 = 24;

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
fn directory_sync_failure_backoff_delay_secs(
    interval_secs: u64,
    consecutive_failures: u64,
) -> u64 {
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

/// Coordinates bounded synchronization for operator-pinned Directory producers.
pub struct DirectoryReplicaSyncCoordinator {
    peers: Arc<[[u8; 32]]>,
    interval: Duration,
    store: Arc<DirectoryReplicaStore>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    client: reqwest::Client,
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
    ) -> Result<Self, &'static str> {
        if peers.is_empty() {
            return Err("directory_sync_no_pinned_producers");
        }
        if interval_secs == 0 {
            return Err("directory_sync_interval_invalid");
        }
        runtime.register_producers(&peers);
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5))
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
        })
    }

    /// Spawns the coordinator lifecycle task.
    #[must_use]
    pub fn spawn(self, mut shutdown_rx: broadcast::Receiver<()>) -> JoinHandle<()> {
        tokio::spawn(async move {
            let startup_delay = Duration::from_secs(directory_sync_startup_delay_secs(
                &self.identity.public_key_bytes(),
            ));
            let first_tick = tokio::time::Instant::now() + startup_delay;
            let mut timer = tokio::time::interval_at(first_tick, self.interval);
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            info!(
                pinned_producers = self.peers.len(),
                max_concurrent_producers = DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS,
                startup_delay_secs = startup_delay.as_secs(),
                interval_secs = self.interval.as_secs(),
                "[DIRECTORY_REPLICA] Synchronization coordinator started"
            );

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = timer.tick() => {}
                }
                let round = self.synchronize_round();
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    () = round => {}
                }
            }
            info!("[DIRECTORY_REPLICA] Synchronization coordinator stopped");
        })
    }

    async fn synchronize_round(&self) {
        stream::iter(self.peers.iter().copied())
            .for_each_concurrent(
                DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS,
                |producer| async move {
                    self.synchronize_producer(producer).await;
                },
            )
            .await;
    }

    async fn synchronize_producer(&self, producer: [u8; 32]) {
        let now = unix_now_secs();
        if let Some(retry_at) = self.runtime.deferred_retry_until(&producer, now) {
            self.runtime.record_backoff_skip(producer);
            debug!(
                retry_after_secs = retry_at.saturating_sub(now),
                "[DIRECTORY_REPLICA] Producer synchronization deferred by backoff"
            );
            return;
        }
        if tokio::time::timeout(
            Duration::from_secs(DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS),
            self.synchronize_producer_pages(producer),
        )
        .await
        .is_err()
        {
            self.record_producer_failure(producer, "directory_producer_round_timeout", None, None);
        }
    }

    async fn synchronize_producer_pages(&self, producer: [u8; 32]) {
        let mut pages_completed = 0u32;
        let mut requests_used = 0u32;
        loop {
            self.runtime.record_attempt(producer, unix_now_secs());
            match pull_directory_chain_page(
                Arc::clone(&self.store),
                &self.peer_store,
                self.identity.as_ref(),
                &producer,
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
                        break;
                    }
                }
                Err(reason) => {
                    self.record_producer_failure(
                        producer,
                        &reason,
                        Some(pages_completed),
                        Some(requests_used),
                    );
                    break;
                }
            }
        }
    }

    fn record_producer_failure(
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
            .saturating_add(1);
        let retry_delay_secs = directory_sync_failure_backoff_delay_secs(
            self.interval.as_secs(),
            consecutive_failures,
        );
        let retry_not_before =
            (retry_delay_secs > 0).then(|| failed_at.saturating_add(retry_delay_secs));
        self.runtime
            .record_failure(producer, failed_at, reason, retry_not_before);
        warn!(
            reason = %reason,
            consecutive_failures,
            retry_delay_secs,
            pages_completed = ?pages_completed,
            requests_used = ?requests_used,
            "[DIRECTORY_REPLICA] Pinned producer sync round rejected"
        );
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

async fn post_directory_frame(
    client: &reqwest::Client,
    url: reqwest::Url,
    frame: Vec<u8>,
    operation: &'static str,
) -> Result<Vec<u8>, String> {
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|_| format!("directory_{operation}_transport_failed"))?;
    if !response.status().is_success() {
        return Err(format!(
            "directory_{operation}_http_status_{}",
            response.status().as_u16()
        ));
    }
    read_bounded_http_response(response, MAX_DIRECTORY_SYNC_RESPONSE_BODY_BYTES)
        .await
        .map_err(|error| format!("directory_{operation}_{}", error.as_str()))
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
    fn catch_up_budget_allows_small_pages_but_reserves_worst_case_headroom() {
        assert_eq!(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE, 17);
        assert_eq!(directory_sync_request_count_for_objects(0), 1);
        assert_eq!(directory_sync_request_count_for_objects(1), 2);
        assert_eq!(directory_sync_request_count_for_objects(16), 2);
        assert_eq!(directory_sync_request_count_for_objects(17), 3);
        assert_eq!(directory_sync_request_count_for_objects(256), 17);
        assert!(should_continue_directory_replica_catch_up(1, 2, true));
        assert!(should_continue_directory_replica_catch_up(3, 6, true));
        assert!(!should_continue_directory_replica_catch_up(4, 8, true));
        assert!(!should_continue_directory_replica_catch_up(1, 8, true));
        assert!(!should_continue_directory_replica_catch_up(1, 2, false));
    }
}
