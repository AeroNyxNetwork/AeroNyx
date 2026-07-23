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
//! - `GET /api/vault/v1/issuers`: returns node-signed V2 issuer epochs.
//! - `POST /api/vault/v1/lease`: atomically redeems a V1 bearer ticket or V2
//!   RFC 9474 blind-issued credential without changing the route.
//! - `POST /api/vault/v1/put`: stores one immutable padded ciphertext.
//! - `POST /api/vault/v1/pull`: returns a node-signed stable recovery page.
//! - `POST /api/vault/v1/delete`: administration-key deletion and receipt.
//! - Applies body, concurrency, and global identity-free rate bounds before
//!   request extraction.
//!
//! ## Dependencies
//! - `aeronyx_core::protocol::blind_vault`: stable binary frames and signatures.
//! - `services/blind_vault.rs`: synchronous transactional `SQLite` service.
//! - `server.rs`: mounts this router only after explicit public API enablement.
//!
//! ## Main Logical Flow
//! 1. Middleware rejects pressure before Axum buffers the body.
//! 2. The handler decodes one expected bounded frame kind.
//! 3. `SQLite` work runs on Tokio's blocking pool.
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
//! Last Modified: v1.3.0-BlindVaultIssuerRuntime - Kept internal issuer
//! rotation failures inside one privacy-safe public availability bucket.
//! v1.2.0-BlindVaultIssuerDirectory - Added bounded,
//! node-signed public key discovery for V2 issuer rotation.
//! v1.1.0-BlindVaultBlindAdmission - Added additive V2
//! redemption on the existing bounded lease route.
//! v1.0.0-BlindVaultApi - Initial bounded binary API.
//! ============================================

use std::sync::{atomic::AtomicUsize, Arc};
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::keys::IdentityKeyPair;
use aeronyx_core::protocol::blind_vault::{
    decode_blind_vault_frame, encode_blind_vault_frame, BlindVaultBlindIssuerDirectory,
    BlindVaultFrame, BlindVaultPullResponse, BlindVaultRecoveredObject,
    MAX_BLIND_VAULT_MUTATION_FRAME_BYTES,
};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
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
const ISSUER_DIRECTORY_CACHE_CONTROL: &str = "public, max-age=60, must-revalidate";

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
    let issuer_router = Router::new()
        .route("/api/vault/v1/issuers", get(issuer_directory_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            pull_request_gate,
        ))
        .layer(DefaultBodyLimit::max(0));

    lease_router
        .merge(put_router)
        .merge(delete_router)
        .merge(pull_router)
        .merge(issuer_router)
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
    let outcome = match decode_frame(&body)? {
        BlindVaultFrame::LeaseAdmission(request) => {
            let service = Arc::clone(&state.service);
            tokio::task::spawn_blocking(move || {
                service.provision_lease_with_admission(&request, now_millis())
            })
            .await
            .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
            .map_err(map_service_error)?
        }
        BlindVaultFrame::BlindLeaseAdmission(request) => {
            let service = Arc::clone(&state.service);
            tokio::task::spawn_blocking(move || {
                service.provision_lease_with_blind_admission(&request, now_millis())
            })
            .await
            .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
            .map_err(map_service_error)?
        }
        _ => return Err(ApiFailure::invalid_frame()),
    };
    Ok((
        StatusCode::OK,
        Json(LeaseAdmissionBody {
            success: true,
            existing: outcome == BlindVaultLeaseProvisionOutcome::Existing,
        }),
    )
        .into_response())
}

async fn issuer_directory_handler(
    State(state): State<BlindVaultApiState>,
) -> Result<Response, ApiFailure> {
    let generated_at_ms = now_millis();
    let service = Arc::clone(&state.service);
    let epochs =
        tokio::task::spawn_blocking(move || service.blind_admission_issuer_epochs(generated_at_ms))
            .await
            .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?
            .map_err(map_service_error)?;
    let mut directory = BlindVaultBlindIssuerDirectory::new(
        generated_at_ms,
        state.node_identity.public_key_bytes(),
        epochs,
    );
    directory
        .sign(&state.node_identity)
        .map_err(|_| ApiFailure::new(StatusCode::INTERNAL_SERVER_ERROR, "internal_error"))?;
    let mut response = binary_response(
        StatusCode::OK,
        BlindVaultFrame::BlindIssuerDirectory(directory),
    )?;
    // [BLIND-VAULT-ISSUER-DIRECTORY 2026-07-23 by Codex] A short cache lowers
    // probe pressure while signed freshness keeps rotation fail-closed.
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(ISSUER_DIRECTORY_CACHE_CONTROL),
    );
    Ok(response)
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
        BlindVaultServiceError::Disabled
        | BlindVaultServiceError::AdmissionUnavailable
        | BlindVaultServiceError::IssuerDirectoryRollback
        | BlindVaultServiceError::IssuerDirectoryGenerationConflict
        | BlindVaultServiceError::IssuerDirectoryContinuity
        | BlindVaultServiceError::IssuerDirectoryNoActiveEpoch
        | BlindVaultServiceError::IssuerDirectoryGenerationOutOfRange => {
            // [BLIND-VAULT-ISSUER-RUNTIME 2026-07-23 by Codex] Runtime
            // directory administration errors are never a client oracle.
            ApiFailure::new(StatusCode::SERVICE_UNAVAILABLE, "service_unavailable")
        }
        BlindVaultServiceError::AdmissionIssuerRejected
        | BlindVaultServiceError::AdmissionProofRejected
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
    use aeronyx_core::protocol::blind_vault::{
        BlindVaultAdmissionTicket, BlindVaultBlindAdmissionToken,
        BlindVaultBlindLeaseAdmissionRequest, BlindVaultLeaseAdmissionRequest,
        BlindVaultLeaseCreateRequest, BlindVaultPullRequest, BlindVaultPutRequest,
        BLIND_VAULT_PROTOCOL_VERSION,
    };
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
    use blind_rsa_signatures::{DefaultRng, KeyPairSha384PSSRandomized};
    use sha2::{Digest, Sha256};
    use tempfile::TempDir;
    use tower::ServiceExt;

    use crate::config_blind_vault::{BlindVaultBlindAdmissionIssuerConfig, BlindVaultConfig};

    struct ApiFixture {
        _directory: TempDir,
        router: Router,
        issuer_key: IdentityKeyPair,
        node_key: IdentityKeyPair,
        write_key: IdentityKeyPair,
        admin_key: IdentityKeyPair,
        read_capability: [u8; 32],
        lease_id: [u8; 32],
    }

    impl ApiFixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().expect("temp directory");
            let issuer_key = IdentityKeyPair::from_bytes(&[31; 32]).expect("issuer key");
            let node_key = IdentityKeyPair::from_bytes(&[32; 32]).expect("node key");
            let write_key = IdentityKeyPair::from_bytes(&[33; 32]).expect("write key");
            let admin_key = IdentityKeyPair::from_bytes(&[34; 32]).expect("admin key");
            let config = BlindVaultConfig {
                enabled: true,
                public_api_enabled: true,
                admission_issuer_public_keys: vec![hex::encode(issuer_key.public_key_bytes())],
                db_path: directory.path().join("vault.db").display().to_string(),
                ..BlindVaultConfig::default()
            };
            let service = Arc::new(
                BlindVaultService::new(config, node_key.clone()).expect("blind vault service"),
            );
            Self {
                _directory: directory,
                router: build_blind_vault_router(service, Arc::new(node_key.clone())),
                issuer_key,
                node_key,
                write_key,
                admin_key,
                read_capability: [35; 32],
                lease_id: [36; 32],
            }
        }

        fn admission(&self, now_ms: u64) -> BlindVaultLeaseAdmissionRequest {
            let mut lease = BlindVaultLeaseCreateRequest::new(
                self.lease_id,
                [37; 16],
                self.write_key.public_key_bytes(),
                self.admin_key.public_key_bytes(),
                Sha256::digest(self.read_capability).into(),
                now_ms + 60 * 60 * 1_000,
            );
            lease.sign(&self.admin_key).expect("sign lease");
            let mut admission = BlindVaultAdmissionTicket::new(
                [38; 32],
                self.issuer_key.public_key_bytes(),
                now_ms.saturating_sub(1_000),
                now_ms + 60 * 1_000,
                2 * 60 * 60 * 1_000,
            );
            admission.sign(&self.issuer_key).expect("sign admission");
            BlindVaultLeaseAdmissionRequest { admission, lease }
        }

        async fn post_frame(&self, path: &str, frame: BlindVaultFrame) -> Response {
            let body = encode_blind_vault_frame(&frame).expect("encode request");
            self.post_body(path, body).await
        }

        async fn post_body(&self, path: &str, body: Vec<u8>) -> Response {
            self.router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header(header::CONTENT_TYPE, BINARY_CONTENT_TYPE)
                        .body(Body::from(body))
                        .expect("request"),
                )
                .await
                .expect("route response")
        }
    }

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

    // [BLIND-VAULT-HTTP-CONTRACT 2026-07-23 by Codex] Exercise the same
    // encoded frames used by external clients so router composition, body
    // extraction, service calls, and node-signed responses cannot drift apart.
    #[tokio::test]
    async fn http_contract_admits_stores_and_recovers_signed_ciphertext() {
        let fixture = ApiFixture::new();
        let now_ms = now_millis();

        let admission_response = fixture
            .post_frame(
                "/api/vault/v1/lease",
                BlindVaultFrame::LeaseAdmission(fixture.admission(now_ms)),
            )
            .await;
        assert_eq!(admission_response.status(), StatusCode::OK);
        let admission_body = to_bytes(admission_response.into_body(), 1_024)
            .await
            .expect("admission body");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&admission_body).expect("admission json"),
            serde_json::json!({"success": true, "existing": false})
        );

        let mut put = BlindVaultPutRequest::new(
            fixture.lease_id,
            [39; 32],
            [40; 16],
            vec![0xa5; 4 * 1_024],
            now_ms + 30 * 60 * 1_000,
        );
        put.sign(&fixture.write_key);
        let put_response = fixture
            .post_frame("/api/vault/v1/put", BlindVaultFrame::Put(put.clone()))
            .await;
        assert_eq!(put_response.status(), StatusCode::CREATED);
        assert_eq!(
            put_response.headers()[header::CONTENT_TYPE],
            BINARY_CONTENT_TYPE
        );
        let put_body = to_bytes(put_response.into_body(), 16 * 1_024)
            .await
            .expect("put body");
        let BlindVaultFrame::StoredReceipt(receipt) =
            decode_blind_vault_frame(&put_body).expect("stored receipt")
        else {
            panic!("put returned wrong frame kind");
        };
        assert!(receipt.matches_put(&put));
        receipt
            .validate_and_verify(&fixture.node_key.public_key())
            .expect("valid node receipt");

        let pull = BlindVaultPullRequest {
            version: BLIND_VAULT_PROTOCOL_VERSION,
            lease_id: fixture.lease_id,
            read_capability: fixture.read_capability,
            continuation_cursor: Vec::new(),
            limit: 4,
        };
        let pull_response = fixture
            .post_frame("/api/vault/v1/pull", BlindVaultFrame::PullRequest(pull))
            .await;
        assert_eq!(pull_response.status(), StatusCode::OK);
        let pull_body = to_bytes(pull_response.into_body(), 32 * 1_024)
            .await
            .expect("pull body");
        let BlindVaultFrame::PullResponse(page) =
            decode_blind_vault_frame(&pull_body).expect("pull response")
        else {
            panic!("pull returned wrong frame kind");
        };
        assert_eq!(page.objects.len(), 1);
        assert_eq!(page.objects[0].ciphertext, put.ciphertext);
        page.validate_and_verify(&fixture.node_key.public_key())
            .expect("valid signed recovery page");
    }

    // [BLIND-VAULT-RFC9474-HTTP 2026-07-23 by Codex] Cross the actual Axum
    // boundary with a finalized blind signature. This proves the public lease
    // endpoint accepts V2 without exposing the issuer-side blinding transcript.
    #[tokio::test]
    async fn http_contract_redeems_rfc9474_blind_admission() {
        let now_ms = now_millis();
        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
        let public_der = key_pair.pk.to_der().expect("public key DER");
        let issuer_key_id: [u8; 32] = Sha256::digest(&public_der).into();
        let directory = tempfile::tempdir().expect("temp directory");
        let node_key = IdentityKeyPair::from_bytes(&[61; 32]).expect("node key");
        let config = BlindVaultConfig {
            enabled: true,
            public_api_enabled: true,
            blind_admission_issuers: vec![BlindVaultBlindAdmissionIssuerConfig {
                public_key_der_base64: BASE64.encode(&public_der),
                not_before_unix_secs: now_ms / 1_000 - 60,
                expires_at_unix_secs: now_ms / 1_000 + 24 * 60 * 60,
                max_lease_ttl_secs: 14 * 24 * 60 * 60,
            }],
            db_path: directory.path().join("vault.db").display().to_string(),
            ..BlindVaultConfig::default()
        };
        let service = Arc::new(
            BlindVaultService::new(config, node_key.clone()).expect("blind vault service"),
        );
        let router = build_blind_vault_router(service, Arc::new(node_key.clone()));

        let directory_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/vault/v1/issuers")
                    .body(Body::empty())
                    .expect("issuer request"),
            )
            .await
            .expect("issuer response");
        assert_eq!(directory_response.status(), StatusCode::OK);
        assert_eq!(
            directory_response.headers()[header::CONTENT_TYPE],
            BINARY_CONTENT_TYPE
        );
        assert_eq!(
            directory_response.headers()[header::CACHE_CONTROL],
            ISSUER_DIRECTORY_CACHE_CONTROL
        );
        let directory_body = to_bytes(directory_response.into_body(), 16 * 1_024)
            .await
            .expect("issuer directory body");
        let BlindVaultFrame::BlindIssuerDirectory(issuer_directory) =
            decode_blind_vault_frame(&directory_body).expect("issuer directory frame")
        else {
            panic!("issuer route returned wrong frame kind");
        };
        issuer_directory
            .validate_and_verify(now_millis(), 60_000, 5_000, &node_key.public_key())
            .expect("valid node-signed issuer directory");
        assert_eq!(issuer_directory.epochs.len(), 1);
        assert_eq!(issuer_directory.epochs[0].issuer_key_id, issuer_key_id);
        assert_eq!(issuer_directory.epochs[0].public_key_der, public_der);

        let write_key = IdentityKeyPair::from_bytes(&[62; 32]).expect("write key");
        let admin_key = IdentityKeyPair::from_bytes(&[63; 32]).expect("admin key");
        let mut lease = BlindVaultLeaseCreateRequest::new(
            [64; 32],
            [65; 16],
            write_key.public_key_bytes(),
            admin_key.public_key_bytes(),
            Sha256::digest([66; 32]).into(),
            now_ms + 7 * 24 * 60 * 60 * 1_000,
        );
        lease.sign(&admin_key).expect("sign lease");

        let unsigned =
            BlindVaultBlindAdmissionToken::new(issuer_key_id, [67; 32], [0; 32], vec![0; 256]);
        let message = unsigned.message_bytes();
        let blinding = key_pair
            .pk
            .blind(&mut DefaultRng, &message)
            .expect("blind admission message");
        let blind_signature = key_pair
            .sk
            .blind_sign(&blinding.blind_message)
            .expect("blind sign admission");
        let signature = key_pair
            .pk
            .finalize(&blind_signature, &blinding, &message)
            .expect("finalize admission");
        let randomizer = blinding.msg_randomizer.expect("randomized-message mode").0;
        let frame = BlindVaultFrame::BlindLeaseAdmission(BlindVaultBlindLeaseAdmissionRequest {
            admission: BlindVaultBlindAdmissionToken::new(
                issuer_key_id,
                [67; 32],
                randomizer,
                signature.0,
            ),
            lease,
        });
        let body = encode_blind_vault_frame(&frame).expect("encode blind admission");

        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/vault/v1/lease")
                    .header(header::CONTENT_TYPE, BINARY_CONTENT_TYPE)
                    .body(Body::from(body.clone()))
                    .expect("request"),
            )
            .await
            .expect("route response");
        assert_eq!(response.status(), StatusCode::OK);
        let response_body = to_bytes(response.into_body(), 1_024)
            .await
            .expect("response body");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&response_body).expect("response json"),
            serde_json::json!({"success": true, "existing": false})
        );

        let retry = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/vault/v1/lease")
                    .header(header::CONTENT_TYPE, BINARY_CONTENT_TYPE)
                    .body(Body::from(body))
                    .expect("retry request"),
            )
            .await
            .expect("retry response");
        assert_eq!(retry.status(), StatusCode::OK);
        let retry_body = to_bytes(retry.into_body(), 1_024)
            .await
            .expect("retry body");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&retry_body).expect("retry json"),
            serde_json::json!({"success": true, "existing": true})
        );
    }

    #[tokio::test]
    async fn http_contract_bounds_bodies_and_collapses_private_failures() {
        let fixture = ApiFixture::new();
        let malformed = fixture
            .post_body(
                "/api/vault/v1/lease",
                b"private-capability-material".to_vec(),
            )
            .await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        let malformed_body = to_bytes(malformed.into_body(), 1_024)
            .await
            .expect("malformed body");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&malformed_body).expect("error json"),
            serde_json::json!({"success": false, "error": "invalid_frame"})
        );
        assert!(!malformed_body
            .windows(b"private-capability-material".len())
            .any(|window| window == b"private-capability-material"));

        let oversized = fixture
            .post_body(
                "/api/vault/v1/lease",
                vec![0; SMALL_REQUEST_BODY_MAX_BYTES + 1],
            )
            .await;
        assert_eq!(oversized.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
