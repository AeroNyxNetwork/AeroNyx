// ============================================
// File: crates/aeronyx-server/src/api/directory_replica_status.rs
// ============================================
//! # Directory Replica Status API
//!
//! ## Creation Reason
//! Directory replica observability has a different trust boundary from the
//! authenticated peer transport. Keeping both in one module made it easy for
//! operator-only producer detail to leak into the public listener by accident.
//!
//! ## Main Functionality
//! - Serves `GET /api/discovery/directory/status`.
//! - Exposes aggregate protocol health on the public listener.
//! - Exposes only truncated producer fingerprints on local/VPN listeners.
//! - Combines audited persisted indexes with bounded runtime observations.
//! - Reports catch-up/deadline/backoff policy without endpoint or route data.
//! - Declares whether retry scheduling is audited and success-cleared atomically.
//!
//! ## Calling Relationships
//! - `server.rs` mounts this router separately for public and operator scopes.
//! - `services/directory_replica.rs` provides audited snapshots and telemetry.
//! - `directory_replica_sync.rs` owns the scheduler constants reported here.
//!
//! ## Main Logical Flow
//! 1. Read the already-audited `SQLite` indexes on a blocking worker.
//! 2. Join persisted producer snapshots with in-memory sync observations.
//! 3. Derive aggregate lag, failure, backoff, quarantine, and sync state.
//! 4. Serialize aggregate-only or fingerprint-only detail by listener scope.
//!
//! ## Privacy Invariant
//! Public output is aggregate-only. Local detail contains a 12-character
//! fingerprint and bounded control-plane counters only. Neither scope exposes
//! endpoints, full producer identities, descriptors, routes, selected hops,
//! client metadata, payloads, DNS contents, destinations, private keys, wallet
//! traffic, or social graph metadata.
//!
//! ## Important Note for Next Developer
//! - Never infer scope from request headers or caller-provided input.
//! - Scope is fixed when `server.rs` mounts the router on a listener.
//! - Keep `producers` omitted, not empty, on the public response contract.
//! - Failure reasons must remain stable internal buckets, never peer strings.
//!
//! ## Last Modified
//! `v0.3.0-DirectoryReplicaDurableBackoffStatus` - Declared audited `SQLite` retry
//! persistence and atomic successful-import cleanup as additive policy fields.
//! v0.2.0-DirectoryReplicaBackoffStatus - Added aggregate and fingerprint-only
//! producer deadline/backoff observations without widening identity exposure.
//! v0.1.0-DirectoryReplicaStatusSplit - Isolated the privacy-tiered status API
//! from authenticated Directory Chain peer transport.
// ============================================

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use tracing::warn;

use crate::api::directory_replica_sync::{
    DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS, DIRECTORY_SYNC_MAX_PAGES_PER_ROUND,
    DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE, DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS,
    DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND,
};
use crate::services::{
    DirectoryReplicaProducerSnapshot, DirectoryReplicaStore, DirectoryReplicaStoreSnapshot,
    DirectoryReplicaSyncObservation, DirectoryReplicaSyncRuntime,
};

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
    producer_round_timeout_seconds: u64,
    failure_backoff_max_seconds: u64,
    retry_state_persistence: &'static str,
    successful_import_clears_retry_atomically: bool,
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
    backoff_active: bool,
    retry_not_before: Option<u64>,
    retry_after_seconds: Option<u64>,
    backoff_skips: u64,
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
    backoff_producers: u64,
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
    next_retry_at: Option<u64>,
    next_retry_after_seconds: Option<u64>,
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

#[derive(Debug, Default)]
struct DirectoryReplicaRuntimeSummary {
    synchronized_producers: u64,
    catching_up_producers: u64,
    backoff_producers: u64,
    known_lag_producers: u64,
    total_lag_blocks: u64,
    max_lag_blocks: Option<u64>,
    last_attempt_at: Option<u64>,
    last_success_at: Option<u64>,
    last_failure_at: Option<u64>,
    last_failure_reason: Option<String>,
    next_retry_at: Option<u64>,
    any_current_failure: bool,
}

/// Builds the Directory Replica observability route.
///
/// Public and local listeners intentionally use separate scope values. Neither
/// response includes full producer identities, endpoints, signed descriptors,
/// route metadata, or user traffic.
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

fn summarize_directory_replica_runtime(
    generated_at: u64,
    runtime: &[DirectoryReplicaSyncObservation],
    runtime_by_producer: &HashMap<[u8; 32], &DirectoryReplicaSyncObservation>,
    configured_producers: &HashSet<[u8; 32]>,
) -> DirectoryReplicaRuntimeSummary {
    let mut summary = DirectoryReplicaRuntimeSummary::default();
    for producer in configured_producers {
        let Some(observation) = runtime_by_producer.get(producer) else {
            continue;
        };
        if let Some(retry_at) = observation
            .retry_not_before
            .filter(|retry_at| *retry_at > generated_at)
        {
            summary.backoff_producers = summary.backoff_producers.saturating_add(1);
            summary.next_retry_at = Some(
                summary
                    .next_retry_at
                    .map_or(retry_at, |current| current.min(retry_at)),
            );
        }
        let lag = observation
            .remote_tip_height
            .map(|remote| remote.saturating_sub(observation.local_tip_height));
        if let Some(lag) = lag {
            summary.known_lag_producers = summary.known_lag_producers.saturating_add(1);
            summary.total_lag_blocks = summary.total_lag_blocks.saturating_add(lag);
            summary.max_lag_blocks = Some(
                summary
                    .max_lag_blocks
                    .map_or(lag, |current| current.max(lag)),
            );
        }
        if observation.has_more || lag.is_some_and(|value| value > 0) {
            summary.catching_up_producers = summary.catching_up_producers.saturating_add(1);
        } else if observation.last_success_at.is_some()
            && observation.consecutive_failures == 0
            && lag == Some(0)
        {
            summary.synchronized_producers = summary.synchronized_producers.saturating_add(1);
        }
    }
    summary.last_attempt_at = runtime
        .iter()
        .filter_map(|value| value.last_attempt_at)
        .max();
    summary.last_success_at = runtime
        .iter()
        .filter_map(|value| value.last_success_at)
        .max();
    if let Some((failed_at, reason)) = runtime
        .iter()
        .filter_map(|value| {
            value
                .last_failure_at
                .map(|failed_at| (failed_at, value.last_failure_reason.clone()))
        })
        .max_by_key(|value| value.0)
    {
        summary.last_failure_at = Some(failed_at);
        summary.last_failure_reason = reason;
    }
    summary.any_current_failure = runtime
        .iter()
        .any(|observation| observation.consecutive_failures > 0);
    summary
}

const fn directory_replica_status_label(
    store_enabled: bool,
    configured_producers: u64,
    quarantined_producers: u64,
    summary: &DirectoryReplicaRuntimeSummary,
) -> &'static str {
    if !store_enabled {
        "disabled"
    } else if configured_producers == 0 {
        "local_only"
    } else if quarantined_producers > 0 {
        "quarantined"
    } else if summary.any_current_failure {
        "degraded"
    } else if summary.catching_up_producers > 0 {
        "catching_up"
    } else if summary.synchronized_producers == configured_producers {
        "healthy"
    } else {
        "pending"
    }
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
    let summary = summarize_directory_replica_runtime(
        generated_at,
        runtime,
        &runtime_by_producer,
        configured_producers,
    );
    let configured_count = configured_producers.len() as u64;
    let status = directory_replica_status_label(
        store_enabled,
        configured_count,
        persisted.quarantined_producers,
        &summary,
    );
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
        synchronized_producers: summary.synchronized_producers,
        catching_up_producers: summary.catching_up_producers,
        backoff_producers: summary.backoff_producers,
        quarantined_producers: persisted.quarantined_producers,
        blocks: persisted.blocks,
        commitments: persisted.commitments,
        incidents: persisted.incidents,
        known_lag_producers: summary.known_lag_producers,
        total_lag_blocks: summary.total_lag_blocks,
        max_lag_blocks: summary.max_lag_blocks,
        last_attempt_at: summary.last_attempt_at,
        last_success_at: summary.last_success_at,
        last_success_age_seconds: summary
            .last_success_at
            .map(|timestamp| generated_at.saturating_sub(timestamp)),
        last_failure_at: summary.last_failure_at,
        last_failure_reason: summary.last_failure_reason,
        next_retry_at: summary.next_retry_at,
        next_retry_after_seconds: summary
            .next_retry_at
            .map(|timestamp| timestamp.saturating_sub(generated_at)),
        catch_up_policy: DirectoryReplicaCatchUpPolicy {
            max_pages_per_round: DIRECTORY_SYNC_MAX_PAGES_PER_ROUND,
            request_budget_per_round: DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND,
            worst_case_requests_per_page: DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE,
            producer_round_timeout_seconds: DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS,
            failure_backoff_max_seconds: DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS,
            retry_state_persistence: "audited_sqlite",
            successful_import_clears_retry_atomically: true,
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
            let retry_not_before = runtime.and_then(|value| value.retry_not_before);
            let backoff_active = retry_not_before.is_some_and(|value| value > generated_at);
            let has_more = runtime.is_some_and(|value| value.has_more);
            let status = if quarantined {
                "quarantined"
            } else if backoff_active {
                "backoff"
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
                backoff_active,
                retry_not_before,
                retry_after_seconds: retry_not_before
                    .map(|timestamp| timestamp.saturating_sub(generated_at)),
                backoff_skips: runtime.map_or(0, |value| value.backoff_skips),
                successful_pages: runtime.map_or(0, |value| value.successful_pages),
                failed_attempts: runtime.map_or(0, |value| value.failed_attempts),
                requests_sent: runtime.map_or(0, |value| value.requests_sent),
            }
        })
        .collect()
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

    use aeronyx_core::crypto::IdentityKeyPair;
    use tempfile::TempDir;

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn status_fixture() -> TestResult<(
        Arc<DirectoryReplicaStore>,
        Arc<DirectoryReplicaSyncRuntime>,
        [u8; 32],
    )> {
        let path = TempDir::new()?.keep().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0xc1; 32])?;
        let producer = IdentityKeyPair::from_bytes(&[0xc2; 32])?;
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(path, local.public_key_bytes(), now_secs())?;
        let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
        runtime.register_producers(&[producer_id]);
        runtime.record_attempt(producer_id, now_secs().saturating_sub(2));
        runtime.record_success(
            producer_id,
            now_secs().saturating_sub(1),
            3,
            7,
            true,
            1,
            4,
            2,
        );
        Ok((Arc::new(store), runtime, producer_id))
    }

    #[tokio::test]
    async fn public_scope_redacts_all_producer_identity() -> TestResult {
        let (store, runtime, producer) = status_fixture()?;
        let router = build_directory_replica_status_router(
            Some(store),
            runtime,
            vec![producer],
            DirectoryReplicaStatusScope::PublicAggregate,
        );
        let response = router
            .oneshot(Request::get("/api/discovery/directory/status").body(Body::empty())?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await?;
        let body_text = String::from_utf8(body.to_vec())?;
        let parsed: serde_json::Value = serde_json::from_str(&body_text)?;
        assert_eq!(parsed["status"].as_str(), Some("catching_up"));
        assert_eq!(parsed["configured_producers"].as_u64(), Some(1));
        assert_eq!(parsed["max_lag_blocks"].as_u64(), Some(4));
        assert!(parsed.get("producers").is_none());
        assert!(!body_text.contains(&hex::encode(producer)));
        Ok(())
    }

    #[tokio::test]
    async fn local_scope_uses_fingerprint_without_endpoint() -> TestResult {
        let (store, runtime, producer) = status_fixture()?;
        let router = build_directory_replica_status_router(
            Some(store),
            runtime,
            vec![producer],
            DirectoryReplicaStatusScope::LocalOperator,
        );
        let response = router
            .oneshot(Request::get("/api/discovery/directory/status").body(Body::empty())?)
            .await?;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 512 * 1024).await?;
        let parsed: serde_json::Value = serde_json::from_slice(&body)?;
        let expected_fingerprint = hex::encode(&producer[..6]);
        assert_eq!(
            parsed["producers"][0]["producer_fingerprint"].as_str(),
            Some(expected_fingerprint.as_str())
        );
        assert_eq!(
            parsed["producers"][0]["status"].as_str(),
            Some("catching_up")
        );
        assert!(parsed["producers"][0].get("public_endpoint").is_none());
        Ok(())
    }

    #[tokio::test]
    async fn backoff_is_aggregate_public_and_fingerprinted_local_only() -> TestResult {
        let (store, runtime, producer) = status_fixture()?;
        let failed_at = now_secs();
        runtime.record_failure(
            producer,
            failed_at,
            "directory_range_transport_failed",
            Some(failed_at.saturating_add(300)),
        );
        runtime.record_backoff_skip(producer);

        let public_router = build_directory_replica_status_router(
            Some(Arc::clone(&store)),
            Arc::clone(&runtime),
            vec![producer],
            DirectoryReplicaStatusScope::PublicAggregate,
        );
        let public_response = public_router
            .oneshot(Request::get("/api/discovery/directory/status").body(Body::empty())?)
            .await?;
        let public_body = to_bytes(public_response.into_body(), 512 * 1024).await?;
        let public: serde_json::Value = serde_json::from_slice(&public_body)?;
        assert_eq!(public["status"].as_str(), Some("degraded"));
        assert_eq!(public["backoff_producers"].as_u64(), Some(1));
        assert!(public["next_retry_after_seconds"]
            .as_u64()
            .is_some_and(|value| value <= 300));
        assert_eq!(
            public["catch_up_policy"]["producer_round_timeout_seconds"].as_u64(),
            Some(DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS)
        );
        assert_eq!(
            public["catch_up_policy"]["retry_state_persistence"].as_str(),
            Some("audited_sqlite")
        );
        assert_eq!(
            public["catch_up_policy"]["successful_import_clears_retry_atomically"].as_bool(),
            Some(true)
        );
        assert!(public.get("producers").is_none());

        let local_router = build_directory_replica_status_router(
            Some(store),
            runtime,
            vec![producer],
            DirectoryReplicaStatusScope::LocalOperator,
        );
        let local_response = local_router
            .oneshot(Request::get("/api/discovery/directory/status").body(Body::empty())?)
            .await?;
        let local_body = to_bytes(local_response.into_body(), 512 * 1024).await?;
        let local: serde_json::Value = serde_json::from_slice(&local_body)?;
        assert_eq!(local["producers"][0]["status"].as_str(), Some("backoff"));
        assert_eq!(
            local["producers"][0]["backoff_active"].as_bool(),
            Some(true)
        );
        assert_eq!(local["producers"][0]["backoff_skips"].as_u64(), Some(1));
        assert!(local["producers"][0]["producer_fingerprint"]
            .as_str()
            .is_some());
        Ok(())
    }
}
