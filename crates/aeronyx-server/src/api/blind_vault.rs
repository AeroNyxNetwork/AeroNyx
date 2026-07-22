// ============================================
// File: crates/aeronyx-server/src/api/blind_vault.rs
// ============================================
//! # Blind Vault Client API
//!
//! ## Creation Reason
//! Exposes the node-blind encrypted-object service without adding account,
//! wallet, sender, receiver, conversation, namespace, or search metadata.
//!
//! ## Main Functionality
//! - `POST /api/vault/v1/lease`: atomically redeems one admission ticket.
//! - `POST /api/vault/v1/put`: stores one immutable padded ciphertext.
//! - `POST /api/vault/v1/pull`: returns a node-signed stable recovery page.
//! - `POST /api/vault/v1/delete`: administration-key deletion and receipt.
//! - Applies body, concurrency, and global identity-free rate bounds before
//!   request extraction.
//!
//! ## Dependencies
//! - `aeronyx_core::protocol::blind_vault`: stable binary frames and signatures.
//! - `services/blind_vault.rs`: synchronous transactional SQLite service.
//! - `server.rs`: mounts this router only after explicit public API enablement.
//!
//! ## Main Logical Flow
//! 1. Middleware rejects pressure before Axum buffers the body.
//! 2. The handler decodes one expected bounded frame kind.
//! 3. SQLite work runs on Tokio's blocking pool.
//! 4. Success returns a binary protocol frame; failure returns only a stable
//!    privacy-safe reason bucket.
//!
//! ## Important Note For The Next Developer
//! - Never add account auth, client IP logs, object IDs, capabilities, ticket
//!   IDs, ciphertext, commitments, cursor bytes, or raw errors to logs.
//! - Never mount these routes merely because local storage is enabled. Public
//!   API enablement and a pinned admission issuer are separate fail-closed
//!   configuration requirements.
//! - Do not replace route middleware with a permit acquired inside a `Bytes`
//!   handler; that would apply backpressure after attacker-controlled buffering.
//! - V1 admission is a signed one-time bearer credential, not blind issuance.
//!
//! Last Modified: v1.0.0-BlindVaultApi - Initial bounded binary API.
//! ============================================

use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::keys::IdentityKeyPair;
use aeronyx_core::protocol::blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultFrame, BlindVaultPullResponse,
    BlindVaultRecoveredObject, MAX_BLIND_VAULT_MUTATION_FRAME_BYTES,
};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Request, State},
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use parking_lot::Mutex;
use serde::Serialize;

use crate::api::InFlightRequestGuard;
use crate::services::{BlindVaultLeaseProvisionOutcome, BlindVaultService, BlindVaultServiceError};

const SMALL_REQUEST_BODY_MAX_BYTES: usize = 16 * 1024;
const MUTATION_REQUEST_BODY_MAX_BYTES: usize = MAX_BLIND_VAULT_MUTATION_FRAME_BYTES as usize;
const MAX_IN_FLIGHT_MUTATIONS: usize = 32;
const MAX_IN_FLIGHT_PULLS: usize = 16;
const MAX_MUTATIONS_PER_SECOND: u64 = 256;
const MAX_PULLS_PER_SECOND: u64 = 64;
const BINARY_CONTENT_TYPE: &str = "application/vnd.aeronyx.blind-vault-v1";

#[derive(Clone)]
struct BlindVaultApiState {
    service: Arc<BlindVaultService>,
    node_identity: Arc<IdentityKeyPair>,
    mutation_in_flight: Arc<AtomicUsize>,
    pull_in_flight: Arc<AtomicUsize>,
    mutation_rate: Arc<Mutex<RateWindow>>,
    pull_rate: Arc<Mutex<RateWindow>>,
}

#[derive(Debug, Default)]
struct RateWindow {
    epoch_second: u64,
    accepted: u64,
}

impl RateWindow {
    fn try_take(&mut self, epoch_second: u64, limit: u64) -> bool {
        if self.epoch_second != epoch_second {
            self.epoch_second = epoch_second;
            self.accepted = 0;
        }
        if self.accepted >= limit {
            return false;
        }
        self.accepted += 1;
        true
    }
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    success: bool,
    error: &'static str,
}

#[derive(Debug, Serialize)]
struct LeaseAdmissionBody {
    success: bool,
    existing: bool,
}

#[derive(Debug)]
struct ApiFailure {
    status: StatusCode,
    reason: &'static str,
}

impl ApiFailure {
    const fn new(status: StatusCode, reason: &'static str) -> Self {
        Self { status, reason }
    }

    const fn invalid_frame() -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_frame")
    }

    const fn backpressure() -> Self {
        Self::new(StatusCode::TOO_MANY_REQUESTS, "backpressure")
    }
}

impl IntoResponse for ApiFailure {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiErrorBody {
                success: false,
                error: self.reason,
            }),
        )
            .into_response()
    }
}

/// Builds all bounded Blind Vault client routes.
pub fn build_blind_vault_router(
    service: Arc<BlindVaultService>,
    node_identity: Arc<IdentityKeyPair>,
) -> Router {
    let state = BlindVaultApiState {
        service,
        node_identity,
        mutation_in_flight: Arc::new(AtomicUsize::new(0)),
        pull_in_flight: Arc::new(AtomicUsize::new(0)),
        mutation_rate: Arc::new(Mutex::new(RateWindow::default())),
        pull_rate: Arc::new(Mutex::new(RateWindow::default())),
    };

    let lease_router = Router::new()
        .route("/api/vault/v1/lease", post(lease_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            mutation_request_gate,
        ))
        .layer(DefaultBodyLimit::max(SMALL_REQUEST_BODY_MAX_BYTES));
    let put_router = Router::new()
        .route("/api/vault/v1/put", post(put_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            mutation_request_gate,
        ))
        .layer(DefaultBodyLimit::max(MUTATION_REQUEST_BODY_MAX_BYTES));
    let delete_router = Router::new()
        .route("/api/vault/v1/delete", post(delete_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            mutation_request_gate,
        ))
        .layer(DefaultBodyLimit::max(SMALL_REQUEST_BODY_MAX_BYTES));
    let pull_router = Router::new()
        .route("/api/vault/v1/pull", post(pull_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            pull_request_gate,
        ))
        .layer(DefaultBodyLimit::max(SMALL_REQUEST_BODY_MAX_BYTES));

    lease_router
        .merge(put_router)
        .merge(delete_router)
        .merge(pull_router)
        .with_state(state)
}

async fn mutation_request_gate(
    State(state): State<BlindVaultApiState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(_in_flight) =
        InFlightRequestGuard::try_acquire(&state.mutation_in_flight, MAX_IN_FLIGHT_MUTATIONS)
    else {
        return ApiFailure::backpressure().into_response();
    };
    if !state
        .mutation_rate
        .lock()
        .try_take(now_seconds(), MAX_MUTATIONS_PER_SECOND)
    {
        return ApiFailure::backpressure().into_response();
    }
    next.run(request).await
}

async fn pull_request_gate(
    State(state): State<BlindVaultApiState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(_in_flight) =
        InFlightRequestGuard::try_acquire(&state.pull_in_flight, MAX_IN_FLIGHT_PULLS)
    else {
        return ApiFailure::backpressure().into_response();
    };
    if !state
        .pull_rate
        .lock()
        .try_take(now_seconds(), MAX_PULLS_PER_SECOND)
    {
        return ApiFailure::backpressure().into_response();
    }
    next.run(request).await
}

async fn lease_handler(
    State(state): State<BlindVaultApiState>,
    body: Bytes,
) -> Result<Response, ApiFailure> {
    let BlindVaultFrame::LeaseAdmission(request) = decode_frame(&body)? else {
        return Err(ApiFailure::invalid_frame());
    };
    let service = Arc::clone(&state.service);
    let outcome = tokio::task::spawn_blocking(move || {
        service.provision_lease_with_admission(&request, now_millis())
    })
    .await
    .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
    .map_err(map_service_error)?;
    Ok((
        StatusCode::OK,
        Json(LeaseAdmissionBody {
            success: true,
            existing: outcome == BlindVaultLeaseProvisionOutcome::Existing,
        }),
    )
        .into_response())
}

async fn put_handler(
    State(state): State<BlindVaultApiState>,
    body: Bytes,
) -> Result<Response, ApiFailure> {
    let BlindVaultFrame::Put(request) = decode_frame(&body)? else {
        return Err(ApiFailure::invalid_frame());
    };
    let service = Arc::clone(&state.service);
    let receipt = tokio::task::spawn_blocking(move || service.put(&request, now_millis()))
        .await
        .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
        .map_err(map_service_error)?;
    binary_response(StatusCode::CREATED, BlindVaultFrame::StoredReceipt(receipt))
}

async fn pull_handler(
    State(state): State<BlindVaultApiState>,
    body: Bytes,
) -> Result<Response, ApiFailure> {
    let BlindVaultFrame::PullRequest(request) = decode_frame(&body)? else {
        return Err(ApiFailure::invalid_frame());
    };
    request
        .validate()
        .map_err(|_| ApiFailure::invalid_frame())?;
    let service = Arc::clone(&state.service);
    let lease_id = request.lease_id;
    let read_capability = request.read_capability;
    let cursor = (!request.continuation_cursor.is_empty()).then_some(request.continuation_cursor);
    let limit = usize::from(request.limit);
    let generated_at_ms = now_millis();
    let page = tokio::task::spawn_blocking(move || {
        service.pull_page(
            &lease_id,
            &read_capability,
            cursor.as_deref(),
            limit,
            generated_at_ms,
        )
    })
    .await
    .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
    .map_err(map_service_error)?;

    let objects = page
        .objects
        .into_iter()
        .map(|object| BlindVaultRecoveredObject {
            object_id: object.object_id,
            ciphertext: object.ciphertext,
            ciphertext_commitment: object.ciphertext_commitment,
            expires_at_ms: object.expires_at_ms,
        })
        .collect();
    let mut response = BlindVaultPullResponse::new(
        lease_id,
        objects,
        page.continuation_cursor.unwrap_or_default(),
        generated_at_ms,
        state.node_identity.public_key_bytes(),
    );
    response
        .sign(&state.node_identity)
        .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?;
    binary_response(StatusCode::OK, BlindVaultFrame::PullResponse(response))
}

async fn delete_handler(
    State(state): State<BlindVaultApiState>,
    body: Bytes,
) -> Result<Response, ApiFailure> {
    let BlindVaultFrame::Delete(request) = decode_frame(&body)? else {
        return Err(ApiFailure::invalid_frame());
    };
    let service = Arc::clone(&state.service);
    let receipt = tokio::task::spawn_blocking(move || service.delete(&request, now_millis()))
        .await
        .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
        .map_err(map_service_error)?;
    binary_response(StatusCode::OK, BlindVaultFrame::DeletedReceipt(receipt))
}

fn decode_frame(body: &[u8]) -> Result<BlindVaultFrame, ApiFailure> {
    decode_blind_vault_frame(body).map_err(|_| ApiFailure::invalid_frame())
}

fn binary_response(status: StatusCode, frame: BlindVaultFrame) -> Result<Response, ApiFailure> {
    let encoded = encode_blind_vault_frame(&frame)
        .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?;
    Ok((
        status,
        [(header::CONTENT_TYPE, BINARY_CONTENT_TYPE)],
        encoded,
    )
        .into_response())
}

fn map_service_error(error: BlindVaultServiceError) -> ApiFailure {
    match error {
        BlindVaultServiceError::Disabled | BlindVaultServiceError::AdmissionUnavailable => {
            ApiFailure::new(StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
        }
        BlindVaultServiceError::AdmissionIssuerRejected
        | BlindVaultServiceError::AdmissionSpent => {
            ApiFailure::new(StatusCode::FORBIDDEN, "admission_rejected")
        }
        BlindVaultServiceError::LeaseNotFound
        | BlindVaultServiceError::LeaseExpired
        | BlindVaultServiceError::ReadUnauthorized
        | BlindVaultServiceError::InvalidPullCursor => {
            ApiFailure::new(StatusCode::FORBIDDEN, "authorization_rejected")
        }
        BlindVaultServiceError::LeaseConflict
        | BlindVaultServiceError::ObjectConflict
        | BlindVaultServiceError::RequestConflict
        | BlindVaultServiceError::ObjectDeleted => {
            ApiFailure::new(StatusCode::CONFLICT, "state_conflict")
        }
        BlindVaultServiceError::ObjectNotFound => {
            ApiFailure::new(StatusCode::NOT_FOUND, "object_unavailable")
        }
        BlindVaultServiceError::QuotaExceeded => {
            ApiFailure::new(StatusCode::SERVICE_UNAVAILABLE, "capacity_exhausted")
        }
        BlindVaultServiceError::Protocol(_) => ApiFailure::invalid_frame(),
        BlindVaultServiceError::Sqlite(_)
        | BlindVaultServiceError::Filesystem
        | BlindVaultServiceError::CorruptState
        | BlindVaultServiceError::TimestampOutOfRange
        | BlindVaultServiceError::PullCursorEncryptionFailed
        | BlindVaultServiceError::AdmissionConfigurationInvalid => {
            ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
        }
    }
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_free_rate_window_resets_and_bounds() {
        let mut window = RateWindow::default();
        assert!(window.try_take(10, 2));
        assert!(window.try_take(10, 2));
        assert!(!window.try_take(10, 2));
        assert!(window.try_take(11, 2));
    }

    #[test]
    fn service_errors_collapse_to_privacy_safe_buckets() {
        let missing = map_service_error(BlindVaultServiceError::LeaseNotFound);
        let expired = map_service_error(BlindVaultServiceError::LeaseExpired);
        let capability = map_service_error(BlindVaultServiceError::ReadUnauthorized);
        assert_eq!(missing.reason, "authorization_rejected");
        assert_eq!(expired.reason, missing.reason);
        assert_eq!(capability.reason, missing.reason);

        let spent = map_service_error(BlindVaultServiceError::AdmissionSpent);
        let issuer = map_service_error(BlindVaultServiceError::AdmissionIssuerRejected);
        assert_eq!(spent.reason, "admission_rejected");
        assert_eq!(issuer.reason, spent.reason);
    }
}
