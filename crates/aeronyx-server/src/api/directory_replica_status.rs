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
//! - Exposes only truncated producer fingerprints in local/VPN status and
//!   incident-list responses.
//! - Combines audited persisted indexes with bounded runtime observations.
//! - Reports catch-up/deadline/backoff policy without endpoint or route data.
//! - Declares whether retry scheduling is audited and success-cleared atomically.
//! - Reports bounded multi-source commitment overlap without claiming quorum,
//!   fork choice, consensus, or global finality.
//! - Reports aggregate observer-signed checkpoint availability and freshness
//!   without exposing hashes, producer identities, or claiming finality.
//! - Reports aggregate independently recomputed witness receipts without
//!   exposing witness identities, request ids, signatures, or checkpoint hashes.
//! - Reports durable and process-lifetime witness outcome buckets so operators
//!   can distinguish unavailable evidence from transport, verification, or
//!   persistence faults without widening the privacy boundary.
//! - Reports the signed local witness-policy epoch, change count, pin count,
//!   threshold, and runtime match without exposing member identities or hashes.
//! - Declares the mature, forward-only witness target policy without exposing
//!   checkpoint hashes, witness identities, or peer scheduling details.
//! - Declares that recurring selection verification is history-bounded while
//!   complete retained-history audit remains a startup and operator boundary.
//! - Lists bounded incident summaries and exports one independently re-verified
//!   signed evidence frame on local/VPN operator listeners only.
//! - Reports signed quarantine-resolution counts while keeping mutation
//!   strictly outside every HTTP listener.
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
//! 4. Classify recent signed-observation overlap across eligible producers.
//! 5. Report observer-signed checkpoint availability without exposing its hash.
//! 6. Report external witness counts without presenting them as votes/quorum.
//! 7. Report audited aggregate witness outcomes, current-process durability,
//!    and the static mature forward-only bounded-verification target policy.
//! 8. Confirm the metadata-anchored signed witness policy matches runtime
//!    aggregate pins and threshold without returning policy membership.
//! 9. Serialize aggregate-only or fingerprint-only detail by listener scope.
//! 10. Re-verify canonical incident evidence before an operator-only export.
//!
//! ## Privacy Invariant
//! Public output is aggregate-only. Local status and incident lists contain
//! 12-character fingerprints and bounded control-plane counters. One explicitly
//! requested evidence package includes the complete producer identity required
//! to verify its Ed25519 signature. No scope exposes endpoints, descriptors,
//! routes, selected hops, client metadata, payloads, DNS contents, destinations,
//! private keys, wallet traffic, or social graph metadata.
//!
//! ## Important Note for Next Developer
//! - Never infer scope from request headers or caller-provided input.
//! - Scope is fixed when `server.rs` mounts the router on a listener.
//! - Keep `producers` omitted, not empty, on the public response contract.
//! - Failure reasons must remain stable internal buckets, never peer strings.
//! - Never rename observation overlap to quorum, consensus, or finality.
//! - Checkpoint counters prove only local signed observation continuity; never
//!   present them as votes, witness quorum, fork choice, or global finality.
//! - External witness counters prove independent recomputation by distinct
//!   pinned nodes only; they do not create quorum semantics or finality.
//! - Policy epoch status is local operator configuration history, never a
//!   validator set, vote, governance mechanism, consensus, or finality claim.
//! - Never mount incident list/export routes on the public listener.
//! - Never add quarantine mutation here; recovery needs an authenticated,
//!   audited compare-and-swap command boundary.
//!
//! ## Last Modified
//! `v0.15.0-DirectoryWitnessPolicyEpochStatus` - Added aggregate signed policy epoch, change count, pin count, threshold, and runtime-match state while keeping identities and digests host-local.
//! `v0.14.0-DirectoryWitnessFailureDrillStatus` - Added current-pin completion evidence for the latest witnessed checkpoint so retired receipts cannot be mistaken for live threshold satisfaction.
//! `v0.13.0-DirectoryMatureWitnessPipelineStatus` - Added an audited mature-checkpoint forward-floor status that does not treat the intentionally unmatured head as a witness failure.
//! `v0.12.0-DirectoryWitnessThresholdStatus` - Added additive pinned-witness corroboration target status.
//! `v0.11.0-DirectoryBoundedWitnessSelectionStatus` - Declared bounded recurring versus full startup audit semantics.
//! `v0.10.0-DirectoryMatureWitnessStatus` - Declared mature forward-only witness scheduling semantics.
//! `v0.9.0-DirectoryWitnessOutcomeStatus` - Added durable and process aggregate witness outcome diagnostics.
//! `v0.8.0-DirectoryObservationWitnessStatus` - Added aggregate external recomputation receipt status.
//! `v0.7.0-DirectoryObservationCheckpointStatus` - Added aggregate checkpoint
//! availability, sequence, and age while redacting hashes and full identities.
//! `v0.6.0-DirectoryReplicaQuarantineResolutionStatus` - Added aggregate and
//! fingerprint-scoped signed resolution counts; no mutation route was added.
//! `v0.5.0-DirectoryReplicaIncidentEvidence` - Added local-only bounded incident
//! summaries and fail-closed canonical signed-evidence export.
//! `v0.4.0-DirectoryReplicaObservationConvergence` - Added bounded aggregate
//! recent-overlap evidence and an operator-only deterministic observation root.
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

use aeronyx_core::protocol::discovery::AERONYX_DIRECTORY_MAINNET_CHAIN_ID;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::api::directory_replica_sync::{
    DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS, DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS,
    DIRECTORY_SYNC_MAX_PAGES_PER_ROUND, DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE,
    DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS, DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND,
};
use crate::services::{
    DirectoryObservationWitnessOutcomeSnapshot, DirectoryReplicaIncidentEvidence,
    DirectoryReplicaIncidentSummary, DirectoryReplicaObservationConvergenceSnapshot,
    DirectoryReplicaProducerSnapshot, DirectoryReplicaStore, DirectoryReplicaStoreSnapshot,
    DirectoryReplicaSyncObservation, DirectoryReplicaSyncRuntime,
    MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE,
};

const DEFAULT_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE: usize = 20;
const DIRECTORY_REPLICA_INCIDENT_PRIVACY_BOUNDARY: &str = "operator-only signed Directory Chain control-plane evidence; no endpoints, descriptors, client IPs, routes, selected hops, message ids, payloads, ciphertext, Memory Chain records, DNS contents, destinations, private keys, wallet traffic, or social graph metadata";

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
    witness_min_verified: usize,
    witness_maturity_delay_secs: u64,
    scope: DirectoryReplicaStatusScope,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaCatchUpPolicy {
    max_pages_per_round: u32,
    request_budget_per_round: u32,
    worst_case_requests_per_page: u32,
    catch_up_interval_seconds: u64,
    producer_round_timeout_seconds: u64,
    failure_backoff_max_seconds: u64,
    retry_state_persistence: &'static str,
    successful_import_clears_retry_atomically: bool,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaObservationConvergenceStatus {
    source_status: &'static str,
    overlap_status: &'static str,
    window_blocks: u64,
    configured_producers: u64,
    eligible_producers: u64,
    pending_producers: u64,
    excluded_quarantined_producers: u64,
    recent_commitments: u64,
    distinct_recent_commitments: u64,
    multi_source_recent_commitments: u64,
    all_eligible_source_recent_commitments: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    observation_root: Option<String>,
    evidence_basis: &'static str,
    security_model: &'static str,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaObservationCheckpointStatus {
    source_status: &'static str,
    checkpoints: u64,
    latest_sequence: u64,
    latest_age_seconds: Option<u64>,
    external_witness_receipts: u64,
    latest_witnessed_sequence: u64,
    latest_sequence_witnesses: u64,
    latest_witnessed_sequence_current_pinned_witnesses: u64,
    latest_witnessed_sequence_threshold_met: bool,
    latest_checkpoint_current_pinned_witnesses: u64,
    latest_checkpoint_externally_witnessed: bool,
    required_external_witnesses: u64,
    latest_checkpoint_witnesses_remaining: Option<u64>,
    latest_checkpoint_corroboration_status: &'static str,
    latest_checkpoint_threshold_met: bool,
    corroboration_policy: &'static str,
    evidence_basis: &'static str,
    security_model: &'static str,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaObservationWitnessPolicyStatus {
    source_status: &'static str,
    status: &'static str,
    epoch: u64,
    historical_changes: u64,
    activated_age_seconds: Option<u64>,
    configured_witnesses: u64,
    required_external_witnesses: u64,
    matches_runtime_config: bool,
    durability: &'static str,
    full_history_verification: &'static str,
    privacy_boundary: &'static str,
    security_model: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DirectoryReplicaPendingWitnessTarget {
    checkpoint_sequence: u64,
    current_pinned_witnesses: u64,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaObservationWitnessPipelineStatus {
    source_status: &'static str,
    status: &'static str,
    maturity_delay_seconds: u64,
    head_checkpoint_sequence: u64,
    head_maturity_status: &'static str,
    forward_floor_clear: bool,
    pending_mature_checkpoint_sequence: Option<u64>,
    pending_current_pinned_witnesses: Option<u64>,
    required_external_witnesses: u64,
    pending_witnesses_remaining: Option<u64>,
    target_policy: &'static str,
    health_basis: &'static str,
    privacy_boundary: &'static str,
    security_model: &'static str,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaObservationWitnessOutcomeStatus {
    source_status: &'static str,
    durability: &'static str,
    target_policy: &'static str,
    maturity_policy: &'static str,
    monotonic_floor: &'static str,
    selection_verification: &'static str,
    full_history_verification: &'static str,
    rounds: u64,
    attempts: u64,
    accepted: u64,
    evidence_unavailable: u64,
    evidence_conflict: u64,
    peer_unavailable: u64,
    transport_failures: u64,
    verification_failures: u64,
    persistence_failures: u64,
    last_checkpoint_sequence: u64,
    last_round_age_seconds: Option<u64>,
    last_success_age_seconds: Option<u64>,
    last_failure_age_seconds: Option<u64>,
    last_round_attempts: u64,
    last_round_accepted: u64,
    last_round_evidence_unavailable: u64,
    last_round_evidence_conflict: u64,
    last_round_peer_unavailable: u64,
    last_round_transport_failures: u64,
    last_round_verification_failures: u64,
    last_round_persistence_failures: u64,
    process_rounds: u64,
    process_attempts: u64,
    telemetry_persistence_failures: u64,
    evidence_basis: &'static str,
    security_model: &'static str,
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
    resolutions: u64,
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
    resolutions: u64,
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
    observation_convergence: DirectoryReplicaObservationConvergenceStatus,
    observation_checkpoint: DirectoryReplicaObservationCheckpointStatus,
    observation_witness_policy: DirectoryReplicaObservationWitnessPolicyStatus,
    observation_witness_pipeline: DirectoryReplicaObservationWitnessPipelineStatus,
    observation_witness_outcomes: DirectoryReplicaObservationWitnessOutcomeStatus,
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

#[derive(Debug, Default, Deserialize)]
struct DirectoryReplicaIncidentListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct DirectoryReplicaIncidentEvidenceQuery {
    digest: String,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaIncidentSummaryResponse {
    incident_digest: String,
    producer_fingerprint: String,
    subject_fingerprint: String,
    kind: String,
    height: u64,
    local_hash: String,
    remote_hash: String,
    observed_at: u64,
    producer_quarantined: bool,
    evidence_available: bool,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaIncidentListResponse {
    contract_version: &'static str,
    generated_at: u64,
    status: &'static str,
    incidents: Vec<DirectoryReplicaIncidentSummaryResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_cursor: Option<String>,
    ordering: &'static str,
    evidence_verification: &'static str,
    recovery_policy: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaIncidentEvidenceResponse {
    contract_version: &'static str,
    generated_at: u64,
    status: &'static str,
    chain_id: String,
    incident_digest: String,
    producer: String,
    subject_node_id: String,
    kind: String,
    height: u64,
    local_hash: String,
    remote_hash: String,
    observed_at: u64,
    producer_quarantined: bool,
    evidence_frame_b64: String,
    evidence_sha256: String,
    evidence_format: &'static str,
    verification: &'static str,
    recovery_policy: &'static str,
    privacy_boundary: &'static str,
}

#[derive(Debug, Serialize)]
struct DirectoryReplicaIncidentErrorResponse {
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

struct DirectoryReplicaRuntimeSnapshots<'a> {
    producers: &'a [DirectoryReplicaSyncObservation],
    observation_witness: &'a DirectoryObservationWitnessOutcomeSnapshot,
}

/// Builds the Directory Replica observability route.
///
/// Public and local listeners intentionally use separate scope values. Status
/// and incident-list responses omit full identities. A single local evidence
/// export includes the producer key required for signature verification, but
/// never endpoints, signed descriptors, route metadata, or user traffic.
pub fn build_directory_replica_status_router(
    store: Option<Arc<DirectoryReplicaStore>>,
    runtime: Arc<DirectoryReplicaSyncRuntime>,
    configured_producers: Vec<[u8; 32]>,
    witness_min_verified: usize,
    witness_maturity_delay_secs: u64,
    scope: DirectoryReplicaStatusScope,
) -> Router {
    runtime.register_producers(&configured_producers);
    let state = DirectoryReplicaStatusState {
        store,
        runtime,
        configured_producers: Arc::new(configured_producers.into_iter().collect()),
        witness_min_verified,
        witness_maturity_delay_secs,
        scope,
    };
    let router = Router::new().route(
        "/api/discovery/directory/status",
        get(directory_replica_status_handler),
    );
    let router = if scope == DirectoryReplicaStatusScope::LocalOperator {
        router
            .route(
                "/api/discovery/directory/incidents",
                get(directory_replica_incidents_handler),
            )
            .route(
                "/api/discovery/directory/incident",
                get(directory_replica_incident_evidence_handler),
            )
    } else {
        router
    };
    router.with_state(state)
}

async fn directory_replica_incidents_handler(
    State(state): State<DirectoryReplicaStatusState>,
    Query(query): Query<DirectoryReplicaIncidentListQuery>,
) -> Response {
    let generated_at = now_secs();
    if state.scope != DirectoryReplicaStatusScope::LocalOperator {
        return incident_error_response(
            StatusCode::NOT_FOUND,
            generated_at,
            "incident_route_not_found",
        );
    }
    let limit = query
        .limit
        .unwrap_or(DEFAULT_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE);
    if !(1..=MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE).contains(&limit) {
        return incident_error_response(
            StatusCode::BAD_REQUEST,
            generated_at,
            "invalid_incident_page_limit",
        );
    }
    let Ok(after) = query.after.as_deref().map(parse_hex32).transpose() else {
        return incident_error_response(
            StatusCode::BAD_REQUEST,
            generated_at,
            "invalid_incident_cursor",
        );
    };
    let Some(store) = state.store.clone() else {
        return incident_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            generated_at,
            "replica_store_disabled",
        );
    };
    match tokio::task::spawn_blocking(move || store.incident_summaries(after, limit)).await {
        Ok(Ok(page)) => Json(DirectoryReplicaIncidentListResponse {
            contract_version: "directory_replica_incident_list.v1",
            generated_at,
            status: "available",
            incidents: page
                .incidents
                .into_iter()
                .map(incident_summary_response)
                .collect(),
            next_cursor: page.next_cursor.map(hex::encode),
            ordering: "incident_digest_ascending_exclusive_cursor",
            evidence_verification: "startup_audited_summary_reverified_on_single_export",
            recovery_policy: "operator_review_required_automatic_recovery_disabled",
            privacy_boundary: DIRECTORY_REPLICA_INCIDENT_PRIVACY_BOUNDARY,
        })
        .into_response(),
        Ok(Err(error)) => {
            warn!(
                error = %error,
                "[DIRECTORY_REPLICA] Incident summary query rejected persistence"
            );
            incident_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                generated_at,
                "incident_list_unavailable",
            )
        }
        Err(error) => {
            warn!(
                error = %error,
                "[DIRECTORY_REPLICA] Incident summary worker failed"
            );
            incident_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                generated_at,
                "incident_list_worker_failed",
            )
        }
    }
}

async fn directory_replica_incident_evidence_handler(
    State(state): State<DirectoryReplicaStatusState>,
    Query(query): Query<DirectoryReplicaIncidentEvidenceQuery>,
) -> Response {
    let generated_at = now_secs();
    if state.scope != DirectoryReplicaStatusScope::LocalOperator {
        return incident_error_response(
            StatusCode::NOT_FOUND,
            generated_at,
            "incident_route_not_found",
        );
    }
    let Ok(digest) = parse_hex32(&query.digest) else {
        return incident_error_response(
            StatusCode::BAD_REQUEST,
            generated_at,
            "invalid_incident_digest",
        );
    };
    let Some(store) = state.store.clone() else {
        return incident_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            generated_at,
            "replica_store_disabled",
        );
    };
    match tokio::task::spawn_blocking(move || store.incident_evidence(&digest)).await {
        Ok(Ok(Some(evidence))) => {
            Json(incident_evidence_response(generated_at, evidence)).into_response()
        }
        Ok(Ok(None)) => {
            incident_error_response(StatusCode::NOT_FOUND, generated_at, "incident_not_found")
        }
        Ok(Err(error)) => {
            warn!(
                error = %error,
                "[DIRECTORY_REPLICA] Incident evidence failed verification"
            );
            incident_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                generated_at,
                "incident_evidence_verification_failed",
            )
        }
        Err(error) => {
            warn!(
                error = %error,
                "[DIRECTORY_REPLICA] Incident evidence worker failed"
            );
            incident_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                generated_at,
                "incident_evidence_worker_failed",
            )
        }
    }
}

fn incident_summary_response(
    incident: DirectoryReplicaIncidentSummary,
) -> DirectoryReplicaIncidentSummaryResponse {
    DirectoryReplicaIncidentSummaryResponse {
        incident_digest: hex::encode(incident.incident_digest),
        producer_fingerprint: hex::encode(&incident.producer[..6]),
        subject_fingerprint: hex::encode(&incident.subject_node_id[..6]),
        kind: incident.kind,
        height: incident.height,
        local_hash: hex::encode(incident.local_hash),
        remote_hash: hex::encode(incident.remote_hash),
        observed_at: incident.observed_at,
        producer_quarantined: incident.producer_quarantined,
        evidence_available: true,
    }
}

fn incident_evidence_response(
    generated_at: u64,
    evidence: DirectoryReplicaIncidentEvidence,
) -> DirectoryReplicaIncidentEvidenceResponse {
    DirectoryReplicaIncidentEvidenceResponse {
        contract_version: "directory_replica_incident_evidence.v1",
        generated_at,
        status: "verified",
        chain_id: hex::encode(AERONYX_DIRECTORY_MAINNET_CHAIN_ID),
        incident_digest: hex::encode(evidence.summary.incident_digest),
        producer: hex::encode(evidence.summary.producer),
        subject_node_id: hex::encode(evidence.summary.subject_node_id),
        kind: evidence.summary.kind,
        height: evidence.summary.height,
        local_hash: hex::encode(evidence.summary.local_hash),
        remote_hash: hex::encode(evidence.summary.remote_hash),
        observed_at: evidence.summary.observed_at,
        producer_quarantined: evidence.summary.producer_quarantined,
        evidence_frame_b64: BASE64.encode(evidence.evidence_frame),
        evidence_sha256: hex::encode(evidence.evidence_sha256),
        evidence_format: "canonical_directory_sync_block_range_response_v1_base64",
        verification: "chain_id_canonical_encoding_producer_identity_signature_incident_digest_and_evidence_sha256_verified_on_read",
        recovery_policy: "operator_review_required_automatic_recovery_disabled",
        privacy_boundary: DIRECTORY_REPLICA_INCIDENT_PRIVACY_BOUNDARY,
    }
}

fn incident_error_response(
    status_code: StatusCode,
    generated_at: u64,
    reason: &'static str,
) -> Response {
    (
        status_code,
        Json(DirectoryReplicaIncidentErrorResponse {
            contract_version: "directory_replica_incident_error.v1",
            generated_at,
            status: "unavailable",
            reason,
            privacy_boundary: DIRECTORY_REPLICA_INCIDENT_PRIVACY_BOUNDARY,
        }),
    )
        .into_response()
}

fn parse_hex32(value: &str) -> Result<[u8; 32], ()> {
    if value.len() != 64 {
        return Err(());
    }
    let bytes = hex::decode(value).map_err(|_| ())?;
    bytes.try_into().map_err(|_| ())
}

async fn directory_replica_status_handler(
    State(state): State<DirectoryReplicaStatusState>,
) -> Response {
    let generated_at = now_secs();
    let matured_before = generated_at.saturating_sub(state.witness_maturity_delay_secs);
    let store_enabled = state.store.is_some();
    let mut configured_producers = state
        .configured_producers
        .iter()
        .copied()
        .collect::<Vec<_>>();
    configured_producers.sort_unstable();
    let witness_min_verified = state.witness_min_verified;
    let (
        persisted,
        convergence,
        latest_current_pinned_witnesses,
        latest_witnessed_current_pinned_witnesses,
        pending_witness_target,
        witness_policy_matches_runtime,
    ) = if let Some(store) = state.store.clone() {
        match tokio::task::spawn_blocking(move || {
            let persisted = store.status_snapshot()?;
            let convergence = store.observation_convergence(&configured_producers)?;
            let latest_witnessed_current_pinned_witnesses = store
                .verified_observation_witness_count_for_pins(
                    persisted.observation_checkpoint_witnessed_sequence,
                    &configured_producers,
                    generated_at,
                )?;
            let latest_current_pinned_witnesses = if persisted.observation_checkpoint_sequence
                == persisted.observation_checkpoint_witnessed_sequence
            {
                latest_witnessed_current_pinned_witnesses
            } else {
                0
            };
            let pending_witness_target = if configured_producers.is_empty() || matured_before == 0 {
                None
            } else {
                store
                    .next_audited_mature_observation_checkpoint_below_witness_threshold(
                        matured_before,
                        generated_at,
                        witness_min_verified,
                        &configured_producers,
                    )?
                    .map(|target| {
                        let current_pinned_witnesses = u64::try_from(target.witnessed_by.len())
                            .map_err(|_| {
                                crate::services::DirectoryReplicaStoreError::Integrity(
                                    "pending observation witness count exceeds u64".to_string(),
                                )
                            })?;
                        Ok::<_, crate::services::DirectoryReplicaStoreError>(
                            DirectoryReplicaPendingWitnessTarget {
                                checkpoint_sequence: target.checkpoint.sequence,
                                current_pinned_witnesses,
                            },
                        )
                    })
                    .transpose()?
            };
            let witness_policy_matches_runtime = store
                .observation_witness_policy_matches(&configured_producers, witness_min_verified)?;
            Ok::<_, crate::services::DirectoryReplicaStoreError>((
                persisted,
                convergence,
                latest_current_pinned_witnesses,
                latest_witnessed_current_pinned_witnesses,
                pending_witness_target,
                witness_policy_matches_runtime,
            ))
        })
        .await
        {
            Ok(Ok(snapshots)) => snapshots,
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
        }
    } else {
        let configured_count = state.configured_producers.len() as u64;
        (
            DirectoryReplicaStoreSnapshot::default(),
            DirectoryReplicaObservationConvergenceSnapshot {
                configured_producers: configured_count,
                pending_producers: configured_count,
                ..DirectoryReplicaObservationConvergenceSnapshot::default()
            },
            0,
            0,
            None,
            false,
        )
    };
    let runtime = state.runtime.snapshot();
    let observation_witness_runtime = state.runtime.observation_witness_snapshot();
    let runtime_snapshots = DirectoryReplicaRuntimeSnapshots {
        producers: &runtime,
        observation_witness: &observation_witness_runtime,
    };
    Json(build_directory_replica_status_response(
        generated_at,
        store_enabled,
        &persisted,
        &convergence,
        &runtime_snapshots,
        &state.configured_producers,
        state.witness_min_verified,
        latest_current_pinned_witnesses,
        latest_witnessed_current_pinned_witnesses,
        state.witness_maturity_delay_secs,
        pending_witness_target,
        witness_policy_matches_runtime,
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

const fn observation_convergence_source_status(
    store_enabled: bool,
    convergence: &DirectoryReplicaObservationConvergenceSnapshot,
) -> &'static str {
    if !store_enabled {
        "disabled"
    } else if convergence.configured_producers == 0 {
        "local_only"
    } else if convergence.eligible_producers == 0 {
        "awaiting_sources"
    } else if convergence.eligible_producers == 1 {
        "single_source"
    } else {
        "multi_source"
    }
}

const fn observation_convergence_overlap_status(
    convergence: &DirectoryReplicaObservationConvergenceSnapshot,
) -> &'static str {
    if convergence.eligible_producers < 2 {
        "not_applicable"
    } else if convergence.all_eligible_source_recent_commitments > 0 {
        "all_eligible_sources_overlap"
    } else if convergence.multi_source_recent_commitments > 0 {
        "partial_multi_source_overlap"
    } else {
        "no_recent_overlap"
    }
}

fn build_observation_convergence_status(
    store_enabled: bool,
    convergence: &DirectoryReplicaObservationConvergenceSnapshot,
    scope: DirectoryReplicaStatusScope,
) -> DirectoryReplicaObservationConvergenceStatus {
    DirectoryReplicaObservationConvergenceStatus {
        source_status: observation_convergence_source_status(store_enabled, convergence),
        overlap_status: observation_convergence_overlap_status(convergence),
        window_blocks: convergence.window_blocks,
        configured_producers: convergence.configured_producers,
        eligible_producers: convergence.eligible_producers,
        pending_producers: convergence.pending_producers,
        excluded_quarantined_producers: convergence.excluded_quarantined_producers,
        recent_commitments: convergence.recent_commitments,
        distinct_recent_commitments: convergence.distinct_recent_commitments,
        multi_source_recent_commitments: convergence.multi_source_recent_commitments,
        all_eligible_source_recent_commitments: convergence
            .all_eligible_source_recent_commitments,
        observation_root: if scope == DirectoryReplicaStatusScope::LocalOperator {
            convergence.observation_root.map(hex::encode)
        } else {
            None
        },
        evidence_basis: "exact_descriptor_commitment_hash_overlap_in_recent_verified_producer_blocks",
        security_model:
            "local_recomputable_observation_evidence_not_vote_quorum_fork_choice_consensus_or_finality",
    }
}

fn build_observation_witness_policy_status(
    generated_at: u64,
    store_enabled: bool,
    persisted: &DirectoryReplicaStoreSnapshot,
    witness_policy_matches_runtime: bool,
) -> DirectoryReplicaObservationWitnessPolicyStatus {
    let matches_runtime_config = store_enabled
        && persisted.observation_witness_policy_epoch > 0
        && witness_policy_matches_runtime;
    let status = if !store_enabled {
        "disabled"
    } else if persisted.observation_witness_policy_epoch == 0 {
        "awaiting_startup_reconciliation"
    } else if matches_runtime_config {
        "active"
    } else {
        "runtime_mismatch"
    };
    DirectoryReplicaObservationWitnessPolicyStatus {
        source_status: if !store_enabled {
            "disabled"
        } else if persisted.observation_witness_policy_epoch == 0 {
            "unavailable"
        } else {
            "audited_sqlite"
        },
        status,
        epoch: persisted.observation_witness_policy_epoch,
        historical_changes: persisted
            .observation_witness_policy_epochs
            .saturating_sub(u64::from(persisted.observation_witness_policy_epochs > 0)),
        activated_age_seconds: (persisted.observation_witness_policy_activated_at > 0).then(|| {
            generated_at.saturating_sub(persisted.observation_witness_policy_activated_at)
        }),
        configured_witnesses: persisted.observation_witness_policy_members,
        required_external_witnesses: persisted.observation_witness_policy_threshold,
        matches_runtime_config,
        durability: "node_identity_signed_hash_linked_sqlite_epochs_with_metadata_head_cas",
        full_history_verification: "startup_and_explicit_operator_audit",
        privacy_boundary:
            "aggregate counts only; witness identities, endpoints, signatures, and policy digests remain host-local",
        security_model:
            "local_operator_evidence_policy_not_validator_set_vote_quorum_fork_choice_consensus_or_finality",
    }
}

fn build_observation_checkpoint_status(
    generated_at: u64,
    store_enabled: bool,
    persisted: &DirectoryReplicaStoreSnapshot,
    witness_min_verified: usize,
    latest_current_pinned_witnesses: u64,
    latest_witnessed_current_pinned_witnesses: u64,
) -> DirectoryReplicaObservationCheckpointStatus {
    let required_external_witnesses = u64::try_from(witness_min_verified).unwrap_or(u64::MAX);
    let latest_witnessed_sequence_threshold_met =
        persisted.observation_checkpoint_witnessed_sequence > 0
            && latest_witnessed_current_pinned_witnesses >= required_external_witnesses;
    let latest_checkpoint_threshold_met = persisted.observation_checkpoint_sequence > 0
        && latest_current_pinned_witnesses >= required_external_witnesses;
    let latest_checkpoint_corroboration_status = if !store_enabled {
        "disabled"
    } else if persisted.observation_checkpoint_sequence == 0 {
        "awaiting_checkpoint"
    } else if latest_checkpoint_threshold_met {
        "target_met"
    } else if latest_current_pinned_witnesses > 0 {
        "below_target"
    } else {
        "awaiting_external_receipt"
    };
    DirectoryReplicaObservationCheckpointStatus {
        source_status: if !store_enabled {
            "disabled"
        } else if persisted.observation_checkpoints == 0 {
            "awaiting_complete_synchronized_round"
        } else {
            "available"
        },
        checkpoints: persisted.observation_checkpoints,
        latest_sequence: persisted.observation_checkpoint_sequence,
        latest_age_seconds: (persisted.observation_checkpoint_observed_at > 0).then(|| {
            generated_at.saturating_sub(persisted.observation_checkpoint_observed_at)
        }),
        external_witness_receipts: persisted.observation_checkpoint_witnesses,
        latest_witnessed_sequence: persisted.observation_checkpoint_witnessed_sequence,
        latest_sequence_witnesses: persisted.observation_checkpoint_latest_witnesses,
        latest_witnessed_sequence_current_pinned_witnesses:
            latest_witnessed_current_pinned_witnesses,
        latest_witnessed_sequence_threshold_met,
        latest_checkpoint_current_pinned_witnesses: latest_current_pinned_witnesses,
        latest_checkpoint_externally_witnessed: persisted.observation_checkpoint_sequence > 0
            && persisted.observation_checkpoint_witnessed_sequence
                == persisted.observation_checkpoint_sequence,
        required_external_witnesses,
        latest_checkpoint_witnesses_remaining: (persisted.observation_checkpoint_sequence > 0)
            .then(|| {
                required_external_witnesses.saturating_sub(latest_current_pinned_witnesses)
            }),
        latest_checkpoint_corroboration_status,
        latest_checkpoint_threshold_met,
        corroboration_policy: "distinct_current_operator_pinned_external_recomputations",
        evidence_basis:
            "local_node_signed_exact_producer_tips_plus_external_nodes_independently_recomputed_exact_prefixes_and_overlap_root",
        security_model:
            "observer_and_external_recomputation_evidence_not_vote_quorum_fork_choice_consensus_or_finality",
    }
}

fn build_observation_witness_pipeline_status(
    generated_at: u64,
    store_enabled: bool,
    persisted: &DirectoryReplicaStoreSnapshot,
    witness_min_verified: usize,
    witness_maturity_delay_secs: u64,
    pending_target: Option<DirectoryReplicaPendingWitnessTarget>,
) -> DirectoryReplicaObservationWitnessPipelineStatus {
    let required_external_witnesses = u64::try_from(witness_min_verified).unwrap_or(u64::MAX);
    let head_maturity_status = if !store_enabled {
        "disabled"
    } else if persisted.observation_checkpoint_sequence == 0 {
        "awaiting_checkpoint"
    } else if persisted
        .observation_checkpoint_observed_at
        .saturating_add(witness_maturity_delay_secs)
        <= generated_at
    {
        "mature"
    } else {
        "pending_maturity"
    };
    let status = if !store_enabled {
        "disabled"
    } else if persisted.observation_checkpoint_sequence == 0 {
        "awaiting_checkpoint"
    } else if let Some(target) = pending_target {
        if target.current_pinned_witnesses == 0 {
            "awaiting_external_receipt"
        } else {
            "below_target"
        }
    } else {
        "caught_up_at_forward_floor"
    };
    DirectoryReplicaObservationWitnessPipelineStatus {
        source_status: if !store_enabled {
            "disabled"
        } else if persisted.observation_checkpoint_sequence == 0 {
            "awaiting_checkpoint"
        } else {
            "available"
        },
        status,
        maturity_delay_seconds: witness_maturity_delay_secs,
        head_checkpoint_sequence: persisted.observation_checkpoint_sequence,
        head_maturity_status,
        forward_floor_clear: store_enabled
            && persisted.observation_checkpoint_sequence > 0
            && pending_target.is_none(),
        pending_mature_checkpoint_sequence: pending_target
            .map(|target| target.checkpoint_sequence),
        pending_current_pinned_witnesses: pending_target
            .map(|target| target.current_pinned_witnesses),
        required_external_witnesses,
        pending_witnesses_remaining: pending_target.map(|target| {
            required_external_witnesses.saturating_sub(target.current_pinned_witnesses)
        }),
        target_policy: "next_forward_mature_checkpoint_below_pinned_witness_target",
        health_basis: "audited_mature_checkpoint_forward_floor_not_unmatured_head",
        privacy_boundary:
            "aggregate counts and checkpoint sequences only; no witness identities or endpoints",
        security_model:
            "external_recomputation_pipeline_health_not_vote_quorum_fork_choice_consensus_or_finality",
    }
}

fn build_observation_witness_outcome_status(
    generated_at: u64,
    store_enabled: bool,
    persisted: &DirectoryObservationWitnessOutcomeSnapshot,
    runtime: &DirectoryObservationWitnessOutcomeSnapshot,
) -> DirectoryReplicaObservationWitnessOutcomeStatus {
    DirectoryReplicaObservationWitnessOutcomeStatus {
        source_status: if !store_enabled {
            "disabled"
        } else if persisted.rounds == 0 && runtime.rounds > 0 {
            "runtime_only"
        } else if runtime.telemetry_persistence_failures > 0 {
            "degraded"
        } else if persisted.rounds == 0 {
            "awaiting_witness_round"
        } else {
            "available"
        },
        durability: "audited_sqlite_aggregate_plus_process_runtime",
        target_policy: "next_forward_mature_checkpoint_below_pinned_witness_target",
        maturity_policy: "one_configured_directory_sync_interval",
        monotonic_floor: "latest_authenticated_receipt_or_durable_outcome_sequence",
        selection_verification: "candidate_predecessor_latest_receipt_set_and_durable_outcome",
        full_history_verification: "startup_and_explicit_operator_audit",
        rounds: persisted.rounds,
        attempts: persisted.totals.attempts(),
        accepted: persisted.totals.accepted,
        evidence_unavailable: persisted.totals.evidence_unavailable,
        evidence_conflict: persisted.totals.evidence_conflict,
        peer_unavailable: persisted.totals.peer_unavailable,
        transport_failures: persisted.totals.transport_failures,
        verification_failures: persisted.totals.verification_failures,
        persistence_failures: persisted.totals.persistence_failures,
        last_checkpoint_sequence: persisted.last_checkpoint_sequence,
        last_round_age_seconds: persisted
            .last_round_at
            .map(|timestamp| generated_at.saturating_sub(timestamp)),
        last_success_age_seconds: persisted
            .last_success_at
            .map(|timestamp| generated_at.saturating_sub(timestamp)),
        last_failure_age_seconds: persisted
            .last_failure_at
            .map(|timestamp| generated_at.saturating_sub(timestamp)),
        last_round_attempts: persisted.last_round.attempts(),
        last_round_accepted: persisted.last_round.accepted,
        last_round_evidence_unavailable: persisted.last_round.evidence_unavailable,
        last_round_evidence_conflict: persisted.last_round.evidence_conflict,
        last_round_peer_unavailable: persisted.last_round.peer_unavailable,
        last_round_transport_failures: persisted.last_round.transport_failures,
        last_round_verification_failures: persisted.last_round.verification_failures,
        last_round_persistence_failures: persisted.last_round.persistence_failures,
        process_rounds: runtime.rounds,
        process_attempts: runtime.totals.attempts(),
        telemetry_persistence_failures: runtime.telemetry_persistence_failures,
        evidence_basis:
            "mutually_exclusive_privacy_safe_outcomes_from_bounded_external_recomputation_attempts",
        security_model:
            "diagnostic_aggregate_not_peer_reputation_vote_quorum_fork_choice_consensus_or_finality",
    }
}

fn build_directory_replica_status_response(
    generated_at: u64,
    store_enabled: bool,
    persisted: &DirectoryReplicaStoreSnapshot,
    convergence: &DirectoryReplicaObservationConvergenceSnapshot,
    runtime: &DirectoryReplicaRuntimeSnapshots<'_>,
    configured_producers: &HashSet<[u8; 32]>,
    witness_min_verified: usize,
    latest_current_pinned_witnesses: u64,
    latest_witnessed_current_pinned_witnesses: u64,
    witness_maturity_delay_secs: u64,
    pending_witness_target: Option<DirectoryReplicaPendingWitnessTarget>,
    witness_policy_matches_runtime: bool,
    scope: DirectoryReplicaStatusScope,
) -> DirectoryReplicaStatusResponse {
    let runtime_by_producer = runtime
        .producers
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
        runtime.producers,
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
        resolutions: persisted.resolutions,
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
            catch_up_interval_seconds: DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS,
            producer_round_timeout_seconds: DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS,
            failure_backoff_max_seconds: DIRECTORY_SYNC_FAILURE_BACKOFF_MAX_SECS,
            retry_state_persistence: "audited_sqlite",
            successful_import_clears_retry_atomically: true,
        },
        observation_convergence: build_observation_convergence_status(
            store_enabled,
            convergence,
            scope,
        ),
        observation_checkpoint: build_observation_checkpoint_status(
            generated_at,
            store_enabled,
            persisted,
            witness_min_verified,
            latest_current_pinned_witnesses,
            latest_witnessed_current_pinned_witnesses,
        ),
        observation_witness_policy: build_observation_witness_policy_status(
            generated_at,
            store_enabled,
            persisted,
            witness_policy_matches_runtime,
        ),
        observation_witness_pipeline: build_observation_witness_pipeline_status(
            generated_at,
            store_enabled,
            persisted,
            witness_min_verified,
            witness_maturity_delay_secs,
            pending_witness_target,
        ),
        observation_witness_outcomes: build_observation_witness_outcome_status(
            generated_at,
            store_enabled,
            &persisted.observation_witness_outcomes,
            runtime.observation_witness,
        ),
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
                resolutions: persisted.map_or(0, |value| value.resolutions),
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
        store.reconcile_observation_witness_policy(&local, &[producer_id], 1, now_secs())?;
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
            1,
            120,
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
        assert_eq!(
            parsed["observation_convergence"]["source_status"].as_str(),
            Some("awaiting_sources")
        );
        assert_eq!(
            parsed["observation_convergence"]["overlap_status"].as_str(),
            Some("not_applicable")
        );
        assert_eq!(
            parsed["observation_convergence"]["window_blocks"].as_u64(),
            Some(32)
        );
        assert!(parsed["observation_convergence"]
            .get("observation_root")
            .is_none());
        assert_eq!(
            parsed["observation_checkpoint"]["source_status"].as_str(),
            Some("awaiting_complete_synchronized_round")
        );
        assert_eq!(
            parsed["observation_checkpoint"]["checkpoints"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_sequence"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["external_witness_receipts"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_witnessed_sequence"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_sequence_witnesses"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_witnessed_sequence_current_pinned_witnesses"]
                .as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_witnessed_sequence_threshold_met"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_checkpoint_current_pinned_witnesses"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_checkpoint_externally_witnessed"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["required_external_witnesses"].as_u64(),
            Some(1)
        );
        assert!(
            parsed["observation_checkpoint"]["latest_checkpoint_witnesses_remaining"].is_null()
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_checkpoint_corroboration_status"].as_str(),
            Some("awaiting_checkpoint")
        );
        assert_eq!(
            parsed["observation_checkpoint"]["latest_checkpoint_threshold_met"].as_bool(),
            Some(false)
        );
        assert_eq!(
            parsed["observation_witness_policy"]["status"].as_str(),
            Some("active")
        );
        assert_eq!(
            parsed["observation_witness_policy"]["epoch"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["observation_witness_policy"]["configured_witnesses"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["observation_witness_policy"]["required_external_witnesses"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed["observation_witness_policy"]["matches_runtime_config"].as_bool(),
            Some(true)
        );
        assert_eq!(
            parsed["observation_checkpoint"]["corroboration_policy"].as_str(),
            Some("distinct_current_operator_pinned_external_recomputations")
        );
        assert_eq!(
            parsed["observation_checkpoint"]["security_model"].as_str(),
            Some(
                "observer_and_external_recomputation_evidence_not_vote_quorum_fork_choice_consensus_or_finality"
            )
        );
        assert_eq!(
            parsed["observation_witness_pipeline"]["status"].as_str(),
            Some("awaiting_checkpoint")
        );
        assert_eq!(
            parsed["observation_witness_pipeline"]["maturity_delay_seconds"].as_u64(),
            Some(120)
        );
        assert_eq!(
            parsed["observation_witness_pipeline"]["forward_floor_clear"].as_bool(),
            Some(false)
        );
        assert!(
            parsed["observation_witness_pipeline"]["pending_mature_checkpoint_sequence"].is_null()
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["source_status"].as_str(),
            Some("awaiting_witness_round")
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["attempts"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["process_attempts"].as_u64(),
            Some(0)
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["target_policy"].as_str(),
            Some("next_forward_mature_checkpoint_below_pinned_witness_target")
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["maturity_policy"].as_str(),
            Some("one_configured_directory_sync_interval")
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["monotonic_floor"].as_str(),
            Some("latest_authenticated_receipt_or_durable_outcome_sequence")
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["selection_verification"].as_str(),
            Some("candidate_predecessor_latest_receipt_set_and_durable_outcome")
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["full_history_verification"].as_str(),
            Some("startup_and_explicit_operator_audit")
        );
        assert_eq!(
            parsed["observation_witness_outcomes"]["security_model"].as_str(),
            Some(
                "diagnostic_aggregate_not_peer_reputation_vote_quorum_fork_choice_consensus_or_finality"
            )
        );
        assert!(parsed.get("producers").is_none());
        assert!(!body_text.contains(&hex::encode(producer)));
        Ok(())
    }

    #[test]
    fn checkpoint_status_distinguishes_partial_and_satisfied_witness_targets() {
        let persisted = DirectoryReplicaStoreSnapshot {
            observation_checkpoints: 9,
            observation_checkpoint_sequence: 9,
            observation_checkpoint_observed_at: 90,
            observation_checkpoint_witnessed_sequence: 9,
            observation_checkpoint_latest_witnesses: 2,
            ..DirectoryReplicaStoreSnapshot::default()
        };

        let partial = build_observation_checkpoint_status(100, true, &persisted, 2, 1, 1);
        assert_eq!(partial.latest_sequence_witnesses, 2);
        assert_eq!(
            partial.latest_witnessed_sequence_current_pinned_witnesses,
            1
        );
        assert!(!partial.latest_witnessed_sequence_threshold_met);
        assert_eq!(partial.latest_checkpoint_current_pinned_witnesses, 1);
        assert_eq!(partial.required_external_witnesses, 2);
        assert_eq!(partial.latest_checkpoint_witnesses_remaining, Some(1));
        assert_eq!(
            partial.latest_checkpoint_corroboration_status,
            "below_target"
        );
        assert!(!partial.latest_checkpoint_threshold_met);
        assert!(partial.latest_checkpoint_externally_witnessed);

        let satisfied = build_observation_checkpoint_status(100, true, &persisted, 2, 2, 2);
        assert_eq!(
            satisfied.latest_witnessed_sequence_current_pinned_witnesses,
            2
        );
        assert!(satisfied.latest_witnessed_sequence_threshold_met);
        assert_eq!(satisfied.latest_checkpoint_current_pinned_witnesses, 2);
        assert_eq!(satisfied.latest_checkpoint_witnesses_remaining, Some(0));
        assert_eq!(
            satisfied.latest_checkpoint_corroboration_status,
            "target_met"
        );
        assert!(satisfied.latest_checkpoint_threshold_met);

        let newer_head = DirectoryReplicaStoreSnapshot {
            observation_checkpoints: 10,
            observation_checkpoint_sequence: 10,
            observation_checkpoint_observed_at: 100,
            observation_checkpoint_witnessed_sequence: 9,
            observation_checkpoint_latest_witnesses: 3,
            ..DirectoryReplicaStoreSnapshot::default()
        };
        let historical_completion =
            build_observation_checkpoint_status(110, true, &newer_head, 2, 0, 2);
        assert_eq!(historical_completion.latest_sequence_witnesses, 3);
        assert_eq!(
            historical_completion.latest_witnessed_sequence_current_pinned_witnesses,
            2
        );
        assert!(historical_completion.latest_witnessed_sequence_threshold_met);
        assert_eq!(
            historical_completion.latest_checkpoint_current_pinned_witnesses,
            0
        );
        assert!(!historical_completion.latest_checkpoint_threshold_met);
    }

    #[test]
    fn witness_pipeline_uses_mature_forward_floor_instead_of_unmatured_head() {
        let persisted = DirectoryReplicaStoreSnapshot {
            observation_checkpoints: 9,
            observation_checkpoint_sequence: 9,
            observation_checkpoint_observed_at: 95,
            ..DirectoryReplicaStoreSnapshot::default()
        };
        let caught_up =
            build_observation_witness_pipeline_status(100, true, &persisted, 2, 10, None);
        assert_eq!(caught_up.status, "caught_up_at_forward_floor");
        assert_eq!(caught_up.head_maturity_status, "pending_maturity");
        assert!(caught_up.forward_floor_clear);
        assert_eq!(caught_up.required_external_witnesses, 2);
        assert_eq!(caught_up.pending_mature_checkpoint_sequence, None);

        let pending = build_observation_witness_pipeline_status(
            110,
            true,
            &persisted,
            2,
            10,
            Some(DirectoryReplicaPendingWitnessTarget {
                checkpoint_sequence: 8,
                current_pinned_witnesses: 1,
            }),
        );
        assert_eq!(pending.status, "below_target");
        assert_eq!(pending.head_maturity_status, "mature");
        assert!(!pending.forward_floor_clear);
        assert_eq!(pending.pending_mature_checkpoint_sequence, Some(8));
        assert_eq!(pending.pending_current_pinned_witnesses, Some(1));
        assert_eq!(pending.pending_witnesses_remaining, Some(1));
    }

    #[test]
    fn witness_outcome_status_separates_durable_and_process_telemetry() {
        let durable = DirectoryObservationWitnessOutcomeSnapshot {
            rounds: 4,
            totals: crate::services::DirectoryObservationWitnessOutcomeCounters {
                accepted: 3,
                evidence_unavailable: 2,
                transport_failures: 1,
                ..crate::services::DirectoryObservationWitnessOutcomeCounters::default()
            },
            last_checkpoint_sequence: 7,
            last_round_at: Some(90),
            last_success_at: Some(90),
            last_failure_at: Some(80),
            last_round: crate::services::DirectoryObservationWitnessOutcomeCounters {
                accepted: 2,
                ..crate::services::DirectoryObservationWitnessOutcomeCounters::default()
            },
            telemetry_persistence_failures: 0,
        };
        let process = DirectoryObservationWitnessOutcomeSnapshot {
            rounds: 1,
            totals: crate::services::DirectoryObservationWitnessOutcomeCounters {
                accepted: 2,
                ..crate::services::DirectoryObservationWitnessOutcomeCounters::default()
            },
            telemetry_persistence_failures: 1,
            ..DirectoryObservationWitnessOutcomeSnapshot::default()
        };

        let status = build_observation_witness_outcome_status(100, true, &durable, &process);
        assert_eq!(status.source_status, "degraded");
        assert_eq!(status.rounds, 4);
        assert_eq!(status.attempts, 6);
        assert_eq!(status.accepted, 3);
        assert_eq!(status.evidence_unavailable, 2);
        assert_eq!(status.transport_failures, 1);
        assert_eq!(status.last_round_age_seconds, Some(10));
        assert_eq!(status.last_round_attempts, 2);
        assert_eq!(status.process_rounds, 1);
        assert_eq!(status.process_attempts, 2);
        assert_eq!(status.telemetry_persistence_failures, 1);
    }

    #[tokio::test]
    async fn local_scope_uses_fingerprint_without_endpoint() -> TestResult {
        let (store, runtime, producer) = status_fixture()?;
        let router = build_directory_replica_status_router(
            Some(store),
            runtime,
            vec![producer],
            1,
            120,
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
    async fn incident_routes_are_local_only_bounded_and_fail_closed() -> TestResult {
        let (store, runtime, producer) = status_fixture()?;
        let public_router = build_directory_replica_status_router(
            Some(Arc::clone(&store)),
            Arc::clone(&runtime),
            vec![producer],
            1,
            120,
            DirectoryReplicaStatusScope::PublicAggregate,
        );
        for uri in [
            "/api/discovery/directory/incidents",
            "/api/discovery/directory/incident?digest=0000000000000000000000000000000000000000000000000000000000000000",
        ] {
            let response = public_router
                .clone()
                .oneshot(Request::get(uri).body(Body::empty())?)
                .await?;
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
        }

        let local_router = build_directory_replica_status_router(
            Some(store),
            runtime,
            vec![producer],
            1,
            120,
            DirectoryReplicaStatusScope::LocalOperator,
        );
        let list_response = local_router
            .clone()
            .oneshot(
                Request::get("/api/discovery/directory/incidents?limit=1").body(Body::empty())?,
            )
            .await?;
        assert_eq!(list_response.status(), StatusCode::OK);
        let list_body = to_bytes(list_response.into_body(), 512 * 1024).await?;
        let list: serde_json::Value = serde_json::from_slice(&list_body)?;
        assert_eq!(
            list["contract_version"].as_str(),
            Some("directory_replica_incident_list.v1")
        );
        assert_eq!(list["incidents"].as_array().map(Vec::len), Some(0));
        assert!(list.get("next_cursor").is_none());
        assert_eq!(
            list["recovery_policy"].as_str(),
            Some("operator_review_required_automatic_recovery_disabled")
        );

        for uri in [
            "/api/discovery/directory/incidents?limit=0",
            "/api/discovery/directory/incidents?limit=51",
            "/api/discovery/directory/incidents?after=not-a-digest",
            "/api/discovery/directory/incident?digest=not-a-digest",
        ] {
            let response = local_router
                .clone()
                .oneshot(Request::get(uri).body(Body::empty())?)
                .await?;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }

        let missing_response = local_router
            .oneshot(
                Request::get("/api/discovery/directory/incident?digest=0000000000000000000000000000000000000000000000000000000000000000")
                    .body(Body::empty())?,
            )
            .await?;
        assert_eq!(missing_response.status(), StatusCode::NOT_FOUND);
        Ok(())
    }

    #[test]
    fn convergence_status_is_explicitly_evidence_not_finality() {
        let convergence = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: 2,
            eligible_producers: 2,
            pending_producers: 0,
            excluded_quarantined_producers: 0,
            window_blocks: 32,
            recent_commitments: 8,
            distinct_recent_commitments: 5,
            multi_source_recent_commitments: 3,
            all_eligible_source_recent_commitments: 3,
            observation_root: Some([0x7a; 32]),
        };
        let public = build_observation_convergence_status(
            true,
            &convergence,
            DirectoryReplicaStatusScope::PublicAggregate,
        );
        assert_eq!(public.source_status, "multi_source");
        assert_eq!(public.overlap_status, "all_eligible_sources_overlap");
        assert_eq!(public.observation_root, None);
        assert_eq!(
            public.security_model,
            "local_recomputable_observation_evidence_not_vote_quorum_fork_choice_consensus_or_finality"
        );

        let local = build_observation_convergence_status(
            true,
            &convergence,
            DirectoryReplicaStatusScope::LocalOperator,
        );
        assert_eq!(local.observation_root, Some(hex::encode([0x7a; 32])));
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
            1,
            120,
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
            1,
            120,
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
