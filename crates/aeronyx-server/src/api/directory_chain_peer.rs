// ============================================
// File: crates/aeronyx-server/src/api/directory_chain_peer.rs
// ============================================
//! # Directory Chain Peer API
//!
//! ## Creation Reason
//! A durable local Directory Chain cannot become independently verifiable by
//! other nodes until it has a narrow authenticated transport. Discovery gossip
//! is permissionless and optimized for current descriptors, so it must not be
//! reused as an unbounded historical ledger endpoint.
//!
//! ## Main Functionality
//! - `POST /api/discovery/peer/directory/tip`
//! - `POST /api/discovery/peer/directory/block-range`
//! - `POST /api/discovery/peer/directory/descriptor-objects`
//! - `GET /api/discovery/directory/status`
//! - Dedicated operator-pinned admission in addition to live `PeerStore` proof.
//! - Ed25519 request/response authentication, timestamp freshness, replay
//!   rejection, per-peer rate limits, body limits, and audit-gated reads.
//! - Exact content-addressed descriptor batches; no partial object response.
//! - Outbound pinned-peer page pull with SSRF-safe endpoint policy, canonical
//!   response verification, exact object hydration, and atomic replica import.
//! - Privacy-tiered replica status: aggregate on the public listener and
//!   truncated producer fingerprints on the local/VPN operator listener.
//!
//! ## Calling Relationships
//! - Mounted by `server.rs` only when `DirectoryChainStore` is configured.
//! - Uses protocol contracts from `aeronyx-core/src/protocol/discovery.rs`.
//! - Reads only through `DirectoryChainStore::audited_*` methods.
//! - Uses `PeerStore::get_valid` as a second, live descriptor admission gate.
//!
//! ## Main Logical Flow
//! 1. Axum rejects an oversized body before protocol deserialization.
//! 2. The handler decodes one canonical Directory Sync V1 frame.
//! 3. Chain id, request bounds, pin, live peer descriptor, timestamp,
//!    signature, replay id, and rate budget are verified in that order.
//! 4. A blocking worker performs the complete local chain audit and bounded read.
//! 5. The local producer signs a response binding request id, ordered hashes,
//!    returned block identities, and the audited tip.
//!
//! ## Privacy Invariant
//! This API serves signed public node-directory commitments and the public
//! signed descriptors they already bind. It never serves client identities,
//! IPs, routes, selected hops, message ids, packet/chat payloads, Memory Chain
//! records, DNS contents, destinations, private keys, or wallet traffic.
//!
//! ## Important Note for Next Developer
//! - Never replace `directory_chain_sync_peer_node_ids` with permissionless
//!   discovery admission. Both a configured pin and a current signed descriptor
//!   are required.
//! - Never return a descriptor not committed by the audited local chain.
//! - Keep request/response limits synchronized with `aeronyx-core` constants.
//! - Replica import, fork quarantine, and fork choice are separate layers. A
//!   valid response proves what one producer signed; it is not consensus.
//!
//! ## Last Modified
//! v0.3.0-DirectoryReplicaStatus - Added privacy-tiered status and bounded
//! multi-page request-budget primitives.
//! v0.2.0-DirectorySyncPull - Added verified bounded replica page download.
//! v0.1.0-DirectorySyncServing - Initial authenticated bounded peer transport.
// ============================================

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rand::RngCore;
use serde::Serialize;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_request_signing_bytes,
    directory_block_range_response_signing_bytes,
    directory_descriptor_objects_request_signing_bytes,
    directory_descriptor_objects_response_signing_bytes, directory_tip_request_signing_bytes,
    directory_tip_response_signing_bytes, encode_directory_sync_message, DirectorySyncMessage,
    AERONYX_DIRECTORY_MAINNET_CHAIN_ID, MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
    MAX_DIRECTORY_SYNC_BLOCKS_V1, MAX_DIRECTORY_SYNC_OBJECTS_V1,
};

use crate::api::memchain_peer::{commitment_peer_endpoint_is_public, commitment_peer_url};
use crate::api::read_bounded_http_response;
use crate::services::{
    DirectoryChainStore, DirectoryChainStoreError, DirectoryReplicaImportReport,
    DirectoryReplicaProducerSnapshot, DirectoryReplicaStore, DirectoryReplicaStoreSnapshot,
    DirectoryReplicaSyncObservation, DirectoryReplicaSyncRuntime, PeerStore,
};

/// A request contains at most sixteen hashes plus fixed signatures and fields.
const MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES: usize = 16 * 1024;
/// Shared inbound budget for each pinned peer identity.
const MAX_REQUESTS_PER_PEER_PER_MINUTE: u32 = 30;
/// Accepted signed request clock skew in either direction.
const REQUEST_TIMESTAMP_SKEW_SECS: u64 = 60;
/// Stateful request ids remain rejected for this duration.
const REPLAY_RETENTION_SECS: u64 = 120;
/// Hard response ceiling shared with the core Directory Sync decoder.
const MAX_DIRECTORY_SYNC_RESPONSE_BODY_BYTES: usize = 512 * 1024;
/// One block per outbound round bounds object hydration and inbound rate use.
const OUTBOUND_BLOCKS_PER_PAGE: u16 = 1;
/// One range request plus the maximum descriptor-object chunks for one block.
pub(crate) const DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE: u32 = 1
    + ((MAX_DIRECTORY_COMMITMENTS_PER_BLOCK + MAX_DIRECTORY_SYNC_OBJECTS_V1 - 1)
        / MAX_DIRECTORY_SYNC_OBJECTS_V1) as u32;
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
pub(crate) fn should_continue_directory_replica_catch_up(
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
    let object_requests = object_count.saturating_add(MAX_DIRECTORY_SYNC_OBJECTS_V1 - 1)
        / MAX_DIRECTORY_SYNC_OBJECTS_V1;
    1u32.saturating_add(object_requests as u32)
}

/// Visibility tier for Directory Replica status responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryReplicaStatusScope {
    /// Public listener exposes aggregate protocol health only.
    PublicAggregate,
    /// Local/VPN operator listener may expose truncated producer fingerprints.
    LocalOperator,
}

#[derive(Clone)]
struct DirectoryReplicaStatusState {
    store: Option<Arc<DirectoryReplicaStore>>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    configured_producers: Arc<HashSet<[u8; 32]>>,
    scope: DirectoryReplicaStatusScope,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaCatchUpPolicy {
    max_pages_per_round: u32,
    request_budget_per_round: u32,
    worst_case_requests_per_page: u32,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaProducerStatus {
    producer_fingerprint: String,
    configured: bool,
    status: &'static str,
    accepted_tip_height: u64,
    signed_remote_tip_height: Option<u64>,
    lag_blocks: Option<u64>,
    accepted_tip_age_seconds: Option<u64>,
    persisted_update_age_seconds: Option<u64>,
    quarantined: bool,
    quarantine_kind: Option<String>,
    blocks: u64,
    commitments: u64,
    incidents: u64,
    last_attempt_at: Option<u64>,
    last_success_at: Option<u64>,
    last_failure_at: Option<u64>,
    last_failure_reason: Option<String>,
    consecutive_failures: u64,
    successful_pages: u64,
    failed_attempts: u64,
    requests_sent: u64,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaStatusResponse {
    contract_version: &'static str,
    generated_at: u64,
    source: &'static str,
    scope: &'static str,
    status: &'static str,
    configured_producers: u64,
    observed_producers: u64,
    synchronized_producers: u64,
    catching_up_producers: u64,
    quarantined_producers: u64,
    blocks: u64,
    commitments: u64,
    incidents: u64,
    known_lag_producers: u64,
    total_lag_blocks: u64,
    max_lag_blocks: Option<u64>,
    last_attempt_at: Option<u64>,
    last_success_at: Option<u64>,
    last_success_age_seconds: Option<u64>,
    last_failure_at: Option<u64>,
    last_failure_reason: Option<String>,
    catch_up_policy: DirectoryReplicaCatchUpPolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    producers: Option<Vec<DirectoryReplicaProducerStatus>>,
    privacy_invariant: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaStatusErrorResponse {
    contract_version: &'static str,
    generated_at: u64,
    status: &'static str,
    reason: &'static str,
    privacy_boundary: &'static str,
}

/// Builds the Directory Replica observability route.
///
/// Public and local listeners intentionally use separate scope values. Neither
/// response includes full producer identities, endpoints, signed descriptors,
/// route metadata, or user traffic.
#[must_use]
pub fn build_directory_replica_status_router(
    store: Option<Arc<DirectoryReplicaStore>>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    configured_producers: Vec<[u8; 32]>,
    scope: DirectoryReplicaStatusScope,
) -> Router {
    runtime.register_producers(&configured_producers);
    let state = DirectoryReplicaStatusState {
        store,
        runtime,
        configured_producers: Arc::new(configured_producers.into_iter().collect()),
        scope,
    };
    Router::new()
        .route(
            "/api/discovery/directory/status",
            get(directory_replica_status_handler),
        )
        .with_state(state)
}

async fn directory_replica_status_handler(
    State(state): State<DirectoryReplicaStatusState>,
) -> Response {
    let generated_at = now_secs();
    let store_enabled = state.store.is_some();
    let persisted = match state.store.clone() {
        Some(store) => match tokio::task::spawn_blocking(move || store.status_snapshot()).await {
            Ok(Ok(snapshot)) => snapshot,
            Ok(Err(error)) => {
                warn!(
                    error = %error,
                    "[DIRECTORY_REPLICA] Status snapshot rejected malformed persistence"
                );
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(DirectoryReplicaStatusErrorResponse {
                        contract_version: "directory_replica_status.v1",
                        generated_at,
                        status: "unavailable",
                        reason: "replica_status_unavailable",
                        privacy_boundary: "aggregate Directory Chain control-plane status only; no endpoints, full producer identities, client IPs, routes, selected hops, payloads, DNS contents, destinations, private keys, wallet traffic, or social graph metadata",
                    }),
                )
                    .into_response();
            }
            Err(error) => {
                warn!(
                    error = %error,
                    "[DIRECTORY_REPLICA] Status snapshot worker failed"
                );
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(DirectoryReplicaStatusErrorResponse {
                        contract_version: "directory_replica_status.v1",
                        generated_at,
                        status: "unavailable",
                        reason: "replica_status_worker_failed",
                        privacy_boundary: "aggregate Directory Chain control-plane status only; no endpoints, full producer identities, client IPs, routes, selected hops, payloads, DNS contents, destinations, private keys, wallet traffic, or social graph metadata",
                    }),
                )
                    .into_response();
            }
        },
        None => DirectoryReplicaStoreSnapshot::default(),
    };
    let runtime = state.runtime.snapshot();
    Json(build_directory_replica_status_response(
        generated_at,
        store_enabled,
        &persisted,
        &runtime,
        &state.configured_producers,
        state.scope,
    ))
    .into_response()
}

fn build_directory_replica_status_response(
    generated_at: u64,
    store_enabled: bool,
    persisted: &DirectoryReplicaStoreSnapshot,
    runtime: &[DirectoryReplicaSyncObservation],
    configured_producers: &HashSet<[u8; 32]>,
    scope: DirectoryReplicaStatusScope,
) -> DirectoryReplicaStatusResponse {
    let runtime_by_producer = runtime
        .iter()
        .map(|observation| (observation.producer, observation))
        .collect::<HashMap<_, _>>();
    let persisted_by_producer = persisted
        .producer_snapshots
        .iter()
        .map(|producer| (producer.producer, producer))
        .collect::<HashMap<_, _>>();
    let mut synchronized_producers = 0u64;
    let mut catching_up_producers = 0u64;
    let mut known_lag_producers = 0u64;
    let mut total_lag_blocks = 0u64;
    let mut max_lag_blocks = None;
    for producer in configured_producers {
        let Some(observation) = runtime_by_producer.get(producer) else {
            continue;
        };
        let lag = observation
            .remote_tip_height
            .map(|remote| remote.saturating_sub(observation.local_tip_height));
        if let Some(lag) = lag {
            known_lag_producers = known_lag_producers.saturating_add(1);
            total_lag_blocks = total_lag_blocks.saturating_add(lag);
            max_lag_blocks = Some(max_lag_blocks.map_or(lag, |current: u64| current.max(lag)));
        }
        if observation.has_more || lag.is_some_and(|value| value > 0) {
            catching_up_producers = catching_up_producers.saturating_add(1);
        } else if observation.last_success_at.is_some()
            && observation.consecutive_failures == 0
            && lag == Some(0)
        {
            synchronized_producers = synchronized_producers.saturating_add(1);
        }
    }
    let last_attempt_at = runtime
        .iter()
        .filter_map(|value| value.last_attempt_at)
        .max();
    let last_success_at = runtime
        .iter()
        .filter_map(|value| value.last_success_at)
        .max();
    let latest_failure = runtime
        .iter()
        .filter_map(|value| {
            value
                .last_failure_at
                .map(|failed_at| (failed_at, value.last_failure_reason.clone()))
        })
        .max_by_key(|value| value.0);
    let any_current_failure = runtime
        .iter()
        .any(|observation| observation.consecutive_failures > 0);
    let configured_count = configured_producers.len() as u64;
    let status = if !store_enabled {
        "disabled"
    } else if configured_producers.is_empty() {
        "local_only"
    } else if persisted.quarantined_producers > 0 {
        "quarantined"
    } else if any_current_failure {
        "degraded"
    } else if catching_up_producers > 0 {
        "catching_up"
    } else if synchronized_producers == configured_count {
        "healthy"
    } else {
        "pending"
    };
    let producers = (scope == DirectoryReplicaStatusScope::LocalOperator).then(|| {
        build_directory_replica_producer_statuses(
            generated_at,
            &persisted_by_producer,
            &runtime_by_producer,
            configured_producers,
        )
    });
    DirectoryReplicaStatusResponse {
        contract_version: "directory_replica_status.v1",
        generated_at,
        source: "rust_directory_replica_store_and_sync_runtime",
        scope: match scope {
            DirectoryReplicaStatusScope::PublicAggregate => "public_aggregate",
            DirectoryReplicaStatusScope::LocalOperator => "local_or_vpn_operator",
        },
        status,
        configured_producers: configured_count,
        observed_producers: persisted.producers,
        synchronized_producers,
        catching_up_producers,
        quarantined_producers: persisted.quarantined_producers,
        blocks: persisted.blocks,
        commitments: persisted.commitments,
        incidents: persisted.incidents,
        known_lag_producers,
        total_lag_blocks,
        max_lag_blocks,
        last_attempt_at,
        last_success_at,
        last_success_age_seconds: last_success_at
            .map(|timestamp| generated_at.saturating_sub(timestamp)),
        last_failure_at: latest_failure.as_ref().map(|value| value.0),
        last_failure_reason: latest_failure.and_then(|value| value.1),
        catch_up_policy: DirectoryReplicaCatchUpPolicy {
            max_pages_per_round: DIRECTORY_SYNC_MAX_PAGES_PER_ROUND,
            request_budget_per_round: DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND,
            worst_case_requests_per_page: DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE,
        },
        producers,
        privacy_invariant:
            "directory_replicas_store_only_public_signed_directory_evidence_in_producer_isolated_namespaces",
        privacy_boundary: "aggregate Directory Chain control-plane status only; local operator detail uses truncated producer fingerprints and never exposes endpoints, full producer identities, client IPs, routes, selected hops, payloads, DNS contents, destinations, private keys, wallet traffic, or social graph metadata",
    }
}

fn build_directory_replica_producer_statuses(
    generated_at: u64,
    persisted: &HashMap<[u8; 32], &DirectoryReplicaProducerSnapshot>,
    runtime: &HashMap<[u8; 32], &DirectoryReplicaSyncObservation>,
    configured: &HashSet<[u8; 32]>,
) -> Vec<DirectoryReplicaProducerStatus> {
    let mut producers = BTreeSet::new();
    producers.extend(persisted.keys().copied());
    producers.extend(runtime.keys().copied());
    producers.extend(configured.iter().copied());
    producers
        .into_iter()
        .map(|producer| {
            let persisted = persisted.get(&producer).copied();
            let runtime = runtime.get(&producer).copied();
            let accepted_tip_height = persisted
                .map(|value| value.tip_height)
                .or_else(|| runtime.map(|value| value.local_tip_height))
                .unwrap_or(0);
            let signed_remote_tip_height = runtime.and_then(|value| value.remote_tip_height);
            let lag_blocks =
                signed_remote_tip_height.map(|remote| remote.saturating_sub(accepted_tip_height));
            let quarantined = persisted.is_some_and(|value| value.quarantined);
            let consecutive_failures = runtime.map_or(0, |value| value.consecutive_failures);
            let has_more = runtime.is_some_and(|value| value.has_more);
            let status = if quarantined {
                "quarantined"
            } else if consecutive_failures > 0 {
                "degraded"
            } else if has_more || lag_blocks.is_some_and(|value| value > 0) {
                "catching_up"
            } else if runtime
                .is_some_and(|value| value.last_success_at.is_some() && lag_blocks == Some(0))
            {
                "synchronized"
            } else if persisted.is_some() && !configured.contains(&producer) {
                "retained"
            } else {
                "pending"
            };
            DirectoryReplicaProducerStatus {
                producer_fingerprint: hex::encode(&producer[..6]),
                configured: configured.contains(&producer),
                status,
                accepted_tip_height,
                signed_remote_tip_height,
                lag_blocks,
                accepted_tip_age_seconds: persisted
                    .map(|value| value.tip_timestamp)
                    .filter(|timestamp| *timestamp > 0)
                    .map(|timestamp| generated_at.saturating_sub(timestamp)),
                persisted_update_age_seconds: persisted
                    .map(|value| generated_at.saturating_sub(value.updated_at)),
                quarantined,
                quarantine_kind: persisted.and_then(|value| value.quarantine_kind.clone()),
                blocks: persisted.map_or(0, |value| value.blocks),
                commitments: persisted.map_or(0, |value| value.commitments),
                incidents: persisted.map_or(0, |value| value.incidents),
                last_attempt_at: runtime.and_then(|value| value.last_attempt_at),
                last_success_at: runtime.and_then(|value| value.last_success_at),
                last_failure_at: runtime.and_then(|value| value.last_failure_at),
                last_failure_reason: runtime.and_then(|value| value.last_failure_reason.clone()),
                consecutive_failures,
                successful_pages: runtime.map_or(0, |value| value.successful_pages),
                failed_attempts: runtime.map_or(0, |value| value.failed_attempts),
                requests_sent: runtime.map_or(0, |value| value.requests_sent),
            }
        })
        .collect()
}

#[derive(Clone)]
struct DirectoryChainPeerState {
    store: Arc<DirectoryChainStore>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peers: Arc<HashSet<[u8; 32]>>,
    guard: Arc<Mutex<DirectoryPeerRequestGuard>>,
}

#[derive(Debug, Default)]
struct DirectoryPeerRequestGuard {
    rate_windows: HashMap<[u8; 32], PeerRateWindow>,
    seen_requests: HashMap<([u8; 32], [u8; 16]), u64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PeerRateWindow {
    minute: u64,
    used: u32,
}

impl DirectoryPeerRequestGuard {
    fn admit(&mut self, requester: [u8; 32], request_id: [u8; 16], now: u64) -> bool {
        self.seen_requests
            .retain(|_, seen_at| now.saturating_sub(*seen_at) <= REPLAY_RETENTION_SECS);
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
        if self.seen_requests.contains_key(&(requester, request_id)) {
            return false;
        }
        self.seen_requests.insert((requester, request_id), now);
        true
    }
}

/// Builds the fail-closed Directory Chain peer router.
#[must_use]
pub fn build_directory_chain_peer_router(
    store: Arc<DirectoryChainStore>,
    peer_store: Arc<PeerStore>,
    identity: Arc<IdentityKeyPair>,
    pinned_peer_ids: Vec<[u8; 32]>,
) -> Router {
    let state = DirectoryChainPeerState {
        store,
        peer_store,
        identity,
        pinned_peers: Arc::new(pinned_peer_ids.into_iter().collect()),
        guard: Arc::new(Mutex::new(DirectoryPeerRequestGuard::default())),
    };
    Router::new()
        .route("/api/discovery/peer/directory/tip", post(tip_handler))
        .route(
            "/api/discovery/peer/directory/block-range",
            post(block_range_handler),
        )
        .route(
            "/api/discovery/peer/directory/descriptor-objects",
            post(descriptor_objects_handler),
        )
        .layer(DefaultBodyLimit::max(MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES))
        .with_state(state)
}

async fn authenticate_request(
    state: &DirectoryChainPeerState,
    requester: [u8; 32],
    request_id: [u8; 16],
    request_timestamp: u64,
    signing_bytes: &[u8],
    signature: &[u8; 64],
    now: u64,
) -> Result<(), Response> {
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err(protocol_error(StatusCode::UNAUTHORIZED, "stale_request"));
    }
    if !state.pinned_peers.contains(&requester) {
        return Err(protocol_error(StatusCode::FORBIDDEN, "peer_not_pinned"));
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return Err(protocol_error(StatusCode::FORBIDDEN, "unknown_peer"));
    }
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(signing_bytes, signature))
        .is_err()
    {
        return Err(protocol_error(
            StatusCode::UNAUTHORIZED,
            "invalid_signature",
        ));
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return Err(protocol_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_or_replay_limited",
        ));
    }
    Ok(())
}

fn decode_request(body: &[u8]) -> Result<DirectorySyncMessage, Response> {
    let message = decode_directory_sync_message(body)
        .map_err(|_| protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"))?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"))?;
    if canonical != body {
        return Err(protocol_error(
            StatusCode::BAD_REQUEST,
            "noncanonical_frame",
        ));
    }
    Ok(message)
}

async fn tip_handler(State(state): State<DirectoryChainPeerState>, body: Bytes) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::TipRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID {
        return protocol_error(StatusCode::BAD_REQUEST, "wrong_chain");
    }
    let now = now_secs();
    let signing_bytes =
        directory_tip_request_signing_bytes(&chain_id, &request_id, &requester, request_timestamp);
    if let Err(response) = authenticate_request(
        &state,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let store = Arc::clone(&state.store);
    let audit = match tokio::task::spawn_blocking(move || store.audited_tip(now)).await {
        Ok(Ok(audit)) => audit,
        Ok(Err(error)) => return store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_tip_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        audit.tip_height,
        &audit.tip_hash,
        audit.tip_timestamp,
    );
    encoded_response(DirectorySyncMessage::TipResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        tip_height: audit.tip_height,
        tip_hash: audit.tip_hash,
        tip_timestamp: audit.tip_timestamp,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn block_range_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::BlockRangeRequestV1 {
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
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || from_height == 0
        || limit == 0
        || limit > MAX_DIRECTORY_SYNC_BLOCKS_V1
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    let now = now_secs();
    let signing_bytes = directory_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let store = Arc::clone(&state.store);
    let page = match tokio::task::spawn_blocking(move || {
        store.audited_block_page(from_height, limit, now)
    })
    .await
    {
        Ok(Ok(page)) => page,
        Ok(Err(error)) => return store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let has_more = page
        .blocks
        .last()
        .is_some_and(|block| block.header.height < page.tip_height);
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &page.blocks,
        has_more,
        page.tip_height,
        &page.tip_hash,
    );
    debug!(
        blocks = page.blocks.len(),
        has_more,
        tip_height = page.tip_height,
        "[DIRECTORY_CHAIN] Served authenticated bounded block page"
    );
    encoded_response(DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks: page.blocks,
        has_more,
        tip_height: page.tip_height,
        tip_hash: page.tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

async fn descriptor_objects_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::DescriptorObjectsRequestV1 {
        chain_id,
        descriptor_hashes,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let mut unique_hashes = descriptor_hashes.clone();
    unique_hashes.sort_unstable();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || descriptor_hashes.is_empty()
        || descriptor_hashes.len() > MAX_DIRECTORY_SYNC_OBJECTS_V1
        || descriptor_hashes.iter().any(|hash| *hash == [0u8; 32])
        || unique_hashes.windows(2).any(|pair| pair[0] == pair[1])
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_object_request");
    }
    let now = now_secs();
    let signing_bytes = directory_descriptor_objects_request_signing_bytes(
        &chain_id,
        &descriptor_hashes,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }

    let store = Arc::clone(&state.store);
    let requested_hashes = descriptor_hashes.clone();
    let objects = match tokio::task::spawn_blocking(move || {
        store.audited_descriptor_objects(&requested_hashes, now)
    })
    .await
    {
        Ok(Ok(Some(objects))) => objects,
        Ok(Ok(None)) => return protocol_error(StatusCode::NOT_FOUND, "object_not_found"),
        Ok(Err(error)) => return store_error_response(&error),
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_task_failed"),
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_descriptor_objects_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &descriptor_hashes,
    );
    debug!(
        objects = objects.len(),
        "[DIRECTORY_CHAIN] Served authenticated descriptor objects"
    );
    encoded_response(DirectorySyncMessage::DescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

/// Pulls, verifies, hydrates, and atomically imports one pinned producer page.
///
/// The producer must have a current signed descriptor in `PeerStore`, and its
/// endpoint must be a public IP literal. The caller is responsible for choosing
/// only operator-pinned producer identities; this function repeats identity,
/// response, block, and object verification but does not make trust decisions.
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
    let request_timestamp = now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "replica_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("producer_quarantined".to_string());
    }
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
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "replica_height_exhausted".to_string())?;
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
    let range_frame = encode_directory_sync_message(&request)
        .map_err(|_| "directory_range_request_encode_failed".to_string())?;
    let range_response = post_directory_frame(client, range_url, range_frame, "range").await?;
    let (blocks, has_more, remote_tip_height, remote_tip_hash) = verify_block_range_response(
        &range_response,
        &request_id,
        producer,
        from_height,
        request_timestamp,
    )?;

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
        let object_timestamp = now_secs();
        let mut object_request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut object_request_id);
        let object_signing = directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            hashes,
            &object_request_id,
            &requester,
            object_timestamp,
        );
        let request = DirectorySyncMessage::DescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            descriptor_hashes: hashes.to_vec(),
            request_id: object_request_id,
            requester,
            request_timestamp: object_timestamp,
            signature: identity.sign(&object_signing),
        };
        let frame = encode_directory_sync_message(&request)
            .map_err(|_| "directory_object_request_encode_failed".to_string())?;
        let response = post_directory_frame(client, object_url.clone(), frame, "objects").await?;
        let mut verified = verify_descriptor_objects_response(
            &response,
            &object_request_id,
            producer,
            hashes,
            object_timestamp,
        )?;
        objects.append(&mut verified);
    }

    let import_store = Arc::clone(&replica_store);
    let import_blocks = blocks.clone();
    let evidence = range_response.clone();
    let producer_id = *producer;
    let observed_at = now_secs();
    let import = tokio::task::spawn_blocking(move || {
        import_store.import_verified_page(
            producer_id,
            &import_blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &evidence,
            observed_at,
        )
    })
    .await
    .map_err(|_| "directory_replica_import_task_failed".to_string())?
    .map_err(|error| match error {
        crate::services::DirectoryReplicaStoreError::Quarantined(_) => {
            "producer_quarantined".to_string()
        }
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

fn verify_block_range_response(
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
        || response_timestamp.abs_diff(now_secs()) > REQUEST_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS) < request_timestamp
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

fn verify_descriptor_objects_response(
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
        || response_timestamp.abs_diff(now_secs()) > REQUEST_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS) < request_timestamp
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
        let commitment =
            aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1::from_signed_descriptor(
                object,
            )
            .map_err(|_| "directory_object_response_invalid_descriptor".to_string())?;
        if commitment.descriptor_hash != *expected_hash {
            return Err("directory_object_response_hash_mismatch".to_string());
        }
    }
    Ok(objects)
}

fn encoded_response(message: DirectorySyncMessage) -> Response {
    match encode_directory_sync_message(&message) {
        Ok(encoded) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            encoded,
        )
            .into_response(),
        Err(error) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Failed to encode peer response");
            protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error")
        }
    }
}

fn store_error_response(error: &DirectoryChainStoreError) -> Response {
    match error {
        DirectoryChainStoreError::Request(_) => {
            protocol_error(StatusCode::BAD_REQUEST, "invalid_request")
        }
        _ => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Refused unaudited peer export");
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified")
        }
    }
}

fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, [(header::CONTENT_TYPE, "text/plain")], code).into_response()
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
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    use aeronyx_core::protocol::discovery::{
        directory_tip_response_signing_bytes, DirectoryDescriptorCommitmentV1, NodeDescriptor,
        SignedNodeDescriptor,
    };
    use tempfile::TempDir;

    fn signed_descriptor(identity: &IdentityKeyPair, now: u64) -> SignedNodeDescriptor {
        SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                identity.public_key_bytes(),
                1,
                now.saturating_sub(1),
                now + 600,
                "directory-sync-test",
            ),
            identity,
        )
        .unwrap()
    }

    fn test_router(
        pinned: bool,
    ) -> (
        Router,
        Arc<IdentityKeyPair>,
        IdentityKeyPair,
        SignedNodeDescriptor,
    ) {
        let now = now_secs();
        let producer = Arc::new(IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap());
        let requester = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
        let observed = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
        let observed_descriptor = signed_descriptor(&observed, now);
        let requester_descriptor = signed_descriptor(&requester, now);
        let peer_store = Arc::new(PeerStore::new());
        peer_store
            .upsert_verified_from_source(requester_descriptor, now, "directory_sync_test")
            .unwrap();
        let temp = TempDir::new().unwrap();
        let path = temp.keep().join("directory.db");
        let (store, _) = DirectoryChainStore::open(path, producer.public_key_bytes(), now).unwrap();
        store
            .append_descriptors(
                std::slice::from_ref(&observed_descriptor),
                now,
                producer.as_ref(),
            )
            .unwrap();
        let pins = pinned
            .then_some(requester.public_key_bytes())
            .into_iter()
            .collect();
        (
            build_directory_chain_peer_router(
                Arc::new(store),
                peer_store,
                Arc::clone(&producer),
                pins,
            ),
            producer,
            requester,
            observed_descriptor,
        )
    }

    fn tip_request(requester: &IdentityKeyPair, request_id: [u8; 16]) -> Vec<u8> {
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let signing_bytes = directory_tip_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester_id,
            timestamp,
        );
        encode_directory_sync_message(&DirectorySyncMessage::TipRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&signing_bytes),
        })
        .unwrap()
    }

    #[tokio::test]
    async fn pinned_live_peer_receives_signed_audited_tip_and_replay_is_rejected() {
        let (router, producer, requester, _) = test_router(true);
        let request_id = [0xb1; 16];
        let request = tip_request(&requester, request_id);
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(request.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let DirectorySyncMessage::TipResponseV1 {
            chain_id,
            request_id: response_request_id,
            responder,
            response_timestamp,
            tip_height,
            tip_hash,
            tip_timestamp,
            signature,
        } = decode_directory_sync_message(&body).unwrap()
        else {
            panic!("unexpected response");
        };
        assert_eq!(chain_id, AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        assert_eq!(response_request_id, request_id);
        assert_eq!(responder, producer.public_key_bytes());
        assert_eq!(tip_height, 1);
        let signing_bytes = directory_tip_response_signing_bytes(
            &chain_id,
            &response_request_id,
            &responder,
            response_timestamp,
            tip_height,
            &tip_hash,
            tip_timestamp,
        );
        IdentityPublicKey::from_bytes(&responder)
            .unwrap()
            .verify(&signing_bytes, &signature)
            .unwrap();

        let replay = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .body(Body::from(request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn unpinned_peer_is_rejected_even_with_live_descriptor_and_valid_signature() {
        let (router, _, requester, _) = test_router(false);
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/tip")
                    .body(Body::from(tip_request(&requester, [0xb2; 16])))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn block_and_descriptor_routes_return_exact_committed_objects() {
        let (router, producer, requester, expected_descriptor) = test_router(true);
        let timestamp = now_secs();
        let requester_id = requester.public_key_bytes();
        let range_id = [0xb3; 16];
        let range_signing = directory_block_range_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            1,
            1,
            &range_id,
            &requester_id,
            timestamp,
        );
        let range = encode_directory_sync_message(&DirectorySyncMessage::BlockRangeRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            from_height: 1,
            limit: 1,
            request_id: range_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&range_signing),
        })
        .unwrap();
        let response = router
            .clone()
            .oneshot(
                Request::post("/api/discovery/peer/directory/block-range")
                    .body(Body::from(range))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let verified_range = verify_block_range_response(
            &body,
            &range_id,
            &producer.public_key_bytes(),
            1,
            timestamp,
        )
        .unwrap();
        assert_eq!(verified_range.0.len(), 1);
        let mut tampered_range = body.to_vec();
        *tampered_range.last_mut().unwrap() ^= 0x01;
        assert_eq!(
            verify_block_range_response(
                &tampered_range,
                &range_id,
                &producer.public_key_bytes(),
                1,
                timestamp,
            )
            .unwrap_err(),
            "directory_range_response_invalid_signature"
        );
        let DirectorySyncMessage::BlockRangeResponseV1 {
            blocks, responder, ..
        } = decode_directory_sync_message(&body).unwrap()
        else {
            panic!("unexpected range response");
        };
        assert_eq!(responder, producer.public_key_bytes());
        assert_eq!(blocks.len(), 1);
        let descriptor_hash = blocks[0].commitments[0].descriptor_hash;
        let expected_commitment =
            DirectoryDescriptorCommitmentV1::from_signed_descriptor(&expected_descriptor).unwrap();
        assert_eq!(descriptor_hash, expected_commitment.descriptor_hash);

        let object_id = [0xb4; 16];
        let object_signing = directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &[descriptor_hash],
            &object_id,
            &requester_id,
            timestamp,
        );
        let object_request =
            encode_directory_sync_message(&DirectorySyncMessage::DescriptorObjectsRequestV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                descriptor_hashes: vec![descriptor_hash],
                request_id: object_id,
                requester: requester_id,
                request_timestamp: timestamp,
                signature: requester.sign(&object_signing),
            })
            .unwrap();
        let response = router
            .oneshot(
                Request::post("/api/discovery/peer/directory/descriptor-objects")
                    .body(Body::from(object_request))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let verified_objects = verify_descriptor_objects_response(
            &body,
            &object_id,
            &producer.public_key_bytes(),
            &[descriptor_hash],
            timestamp,
        )
        .unwrap();
        assert_eq!(verified_objects, vec![expected_descriptor.clone()]);
        let DirectorySyncMessage::DescriptorObjectsResponseV1 {
            descriptor_hashes,
            objects,
            ..
        } = decode_directory_sync_message(&body).unwrap()
        else {
            panic!("unexpected object response");
        };
        assert_eq!(descriptor_hashes, vec![descriptor_hash]);
        assert_eq!(objects, vec![expected_descriptor]);
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

    #[tokio::test]
    async fn replica_status_public_scope_redacts_producers_and_local_scope_uses_fingerprint() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            now_secs(),
        )
        .unwrap();
        let store = Arc::new(store);
        let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
        runtime.register_producers(&[producer.public_key_bytes()]);
        runtime.record_attempt(producer.public_key_bytes(), now_secs().saturating_sub(2));
        runtime.record_success(
            producer.public_key_bytes(),
            now_secs().saturating_sub(1),
            3,
            7,
            true,
            1,
            4,
            2,
        );

        let public = build_directory_replica_status_router(
            Some(Arc::clone(&store)),
            Arc::clone(&runtime),
            vec![producer.public_key_bytes()],
            DirectoryReplicaStatusScope::PublicAggregate,
        );
        let response = public
            .oneshot(
                Request::get("/api/discovery/directory/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let body_text = String::from_utf8(body.to_vec()).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&body_text).unwrap();
        assert_eq!(parsed["status"].as_str(), Some("catching_up"));
        assert_eq!(parsed["configured_producers"].as_u64(), Some(1));
        assert_eq!(parsed["max_lag_blocks"].as_u64(), Some(4));
        assert!(parsed.get("producers").is_none());
        assert!(!body_text.contains(&hex::encode(producer.public_key_bytes())));

        let local_router = build_directory_replica_status_router(
            Some(store),
            runtime,
            vec![producer.public_key_bytes()],
            DirectoryReplicaStatusScope::LocalOperator,
        );
        let response = local_router
            .oneshot(
                Request::get("/api/discovery/directory/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let expected_fingerprint = hex::encode(&producer.public_key_bytes()[..6]);
        assert_eq!(
            parsed["producers"][0]["producer_fingerprint"].as_str(),
            Some(expected_fingerprint.as_str())
        );
        assert_eq!(
            parsed["producers"][0]["status"].as_str(),
            Some("catching_up")
        );
        assert!(parsed["producers"][0].get("public_endpoint").is_none());
    }
}
