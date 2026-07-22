// ============================================
// File: crates/aeronyx-blind-issuer/src/api.rs
// ============================================
// [BLIND-ISSUER 2026-07-23 by Codex] Identity-free, bounded loopback protocol
// with authorization and pressure controls ahead of body extraction.
//! # Internal Blind-Signing API
//!
//! ## Creation Reason
//! The entitlement backend needs a narrow local channel to request an RFC 9474
//! operation without exposing account context to the signer.
//!
//! ## Main Functionality
//! - Authenticates a backend bearer token before buffering the request body.
//! - Enforces private-operation concurrency and per-second pressure ceilings.
//! - Encodes a fixed, allocation-bounded binary request/response contract.
//! - Publishes authenticated, public-only key epochs for safe rotation.
//! - Runs private RSA operations on Tokio's blocking pool.
//!
//! ## Calling Relationships
//! `main.rs` binds this router to loopback; `signer.rs` performs the private
//! operation; the upstream backend may implement the tiny binary codec.
//!
//! ## Privacy Invariant
//! No request/response tracing middleware is installed. Frames contain only a
//! public key fingerprint and blinded RSA bytes.
//!
//! ## Next Developer Guide
//! Keep authentication and pressure middleware outside the `Bytes` extractor.
//! Never add wallet/account headers or request-body diagnostics.
//!
//! Last Modified: v0.2.0-BlindIssuerApi - Added bounded epoch snapshots and
//! separated authorization from private-operation pressure controls.
//! ============================================

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::blind_vault::{
    BlindVaultBlindIssuerEpoch, BLIND_VAULT_BLIND_ADMISSION_VERSION,
    MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES, MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS,
    MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS, MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
    MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES,
};
use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;
use zeroize::Zeroizing;

use crate::signer::{BlindSignError, BlindSignRequest, BlindSignResponse, BlindSigner};

const REQUEST_MAGIC: [u8; 4] = *b"ANBI";
const RESPONSE_MAGIC: [u8; 4] = *b"ANBS";
const EPOCH_RESPONSE_MAGIC: [u8; 4] = *b"ANBE";
const INTERNAL_WIRE_VERSION: u16 = 1;
const FRAME_HEADER_BYTES: usize = 38;
const EPOCH_RESPONSE_HEADER_BYTES: usize = 16;
const EPOCH_ENTRY_FIXED_BYTES: usize = 60;
const MAX_SIGN_REQUEST_BYTES: usize = FRAME_HEADER_BYTES + MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES;
const MAX_EPOCH_RESPONSE_BYTES: usize = EPOCH_RESPONSE_HEADER_BYTES
    + MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS
        * (EPOCH_ENTRY_FIXED_BYTES + MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES);
const AUTHORIZATION_PREFIX: &[u8] = b"Bearer ";
const CACHE_CONTROL_NO_STORE: &str = "no-store";

/// Content type for the local blind-issuer binary contract.
pub const BLIND_ISSUER_CONTENT_TYPE: &str = "application/vnd.aeronyx.blind-issuer-v1";
/// Content type for authenticated public-key epoch snapshots.
pub const BLIND_ISSUER_EPOCH_CONTENT_TYPE: &str = "application/vnd.aeronyx.blind-issuer-epochs-v1";

/// Public-only issuer key material used by the backend to coordinate rotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindIssuerEpochSnapshot {
    /// Snapshot creation time in Unix milliseconds.
    pub generated_at_ms: u64,
    /// Strictly key-ID-sorted, non-expired issuer epochs.
    pub epochs: Vec<BlindVaultBlindIssuerEpoch>,
}

/// Bounded internal wire-codec failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum BlindIssuerWireError {
    /// Frame magic, version, key ID, or length was invalid.
    #[error("blind issuer frame is invalid")]
    InvalidFrame,
    /// Frame exceeded the fixed local protocol limit.
    #[error("blind issuer frame is too large")]
    FrameTooLarge,
}

#[derive(Clone)]
struct ApiState {
    signer: Arc<BlindSigner>,
    auth_token: Arc<Zeroizing<Vec<u8>>>,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: usize,
    rate: Arc<Mutex<RateWindow>>,
    max_requests_per_second: u64,
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

struct InFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl InFlightGuard {
    fn try_acquire(counter: &Arc<AtomicUsize>, limit: usize) -> Option<Self> {
        let mut current = counter.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return None;
            }
            match counter.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(Self {
                        counter: Arc::clone(counter),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Builds the local signer router with authentication and pressure policy.
pub fn build_router(
    signer: Arc<BlindSigner>,
    auth_token: Zeroizing<Vec<u8>>,
    max_requests_per_second: u64,
    max_in_flight: usize,
) -> Router {
    let state = ApiState {
        signer,
        auth_token: Arc::new(auth_token),
        in_flight: Arc::new(AtomicUsize::new(0)),
        max_in_flight,
        rate: Arc::new(Mutex::new(RateWindow::default())),
        max_requests_per_second,
    };
    let signing = Router::new()
        .route("/internal/v1/blind-sign", post(sign_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            signing_pressure_gate,
        ))
        .layer(DefaultBodyLimit::max(MAX_SIGN_REQUEST_BYTES));
    let authenticated = signing
        .merge(Router::new().route("/internal/v1/issuer-epochs", get(epoch_handler)))
        // [BLIND-ISSUER-EPOCHS 2026-07-23 by Codex] Authorization wraps both
        // routes, while only the private RSA route consumes signing capacity.
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            authorization_gate,
        ));
    let health = Router::new().route("/internal/v1/health", get(health_handler));
    authenticated.merge(health).with_state(state)
}

async fn authorization_gate(
    State(state): State<ApiState>,
    request: Request,
    next: Next,
) -> Response {
    if !is_authorized(request.headers(), state.auth_token.as_ref().as_slice()) {
        return empty_response(StatusCode::UNAUTHORIZED);
    }
    next.run(request).await
}

async fn signing_pressure_gate(
    State(state): State<ApiState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(_guard) = InFlightGuard::try_acquire(&state.in_flight, state.max_in_flight) else {
        return empty_response(StatusCode::TOO_MANY_REQUESTS);
    };
    if !state
        .rate
        .lock()
        .try_take(now_seconds(), state.max_requests_per_second)
    {
        return empty_response(StatusCode::TOO_MANY_REQUESTS);
    }
    next.run(request).await
}

async fn sign_handler(State(state): State<ApiState>, headers: HeaderMap, body: Bytes) -> Response {
    if headers.get(header::CONTENT_TYPE).map(HeaderValue::as_bytes)
        != Some(BLIND_ISSUER_CONTENT_TYPE.as_bytes())
    {
        return empty_response(StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }
    let Ok(request) = decode_sign_request(&body) else {
        return empty_response(StatusCode::BAD_REQUEST);
    };
    let signer = Arc::clone(&state.signer);
    let result = tokio::task::spawn_blocking(move || signer.sign(&request, now_millis())).await;
    match result {
        Ok(Ok(response)) => binary_response(&response),
        Ok(Err(BlindSignError::InvalidRequest)) => empty_response(StatusCode::BAD_REQUEST),
        Ok(Err(BlindSignError::UnknownIssuer | BlindSignError::InactiveIssuer)) => {
            empty_response(StatusCode::FORBIDDEN)
        }
        Ok(Err(BlindSignError::SigningFailed)) | Err(_) => {
            empty_response(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

async fn health_handler(State(state): State<ApiState>) -> Response {
    if state.signer.has_active_key(now_millis()) {
        empty_response(StatusCode::NO_CONTENT)
    } else {
        empty_response(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn epoch_handler(State(state): State<ApiState>) -> Response {
    let generated_at_ms = now_millis();
    let snapshot = BlindIssuerEpochSnapshot {
        generated_at_ms,
        epochs: state.signer.public_epochs(generated_at_ms),
    };
    encode_epoch_snapshot(&snapshot).map_or_else(
        |_| empty_response(StatusCode::INTERNAL_SERVER_ERROR),
        |bytes| {
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, BLIND_ISSUER_EPOCH_CONTENT_TYPE),
                    (header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE),
                ],
                bytes,
            )
                .into_response()
        },
    )
}

fn is_authorized(headers: &HeaderMap, expected: &[u8]) -> bool {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return false;
    };
    let bytes = value.as_bytes();
    let Some(candidate) = bytes.strip_prefix(AUTHORIZATION_PREFIX) else {
        return false;
    };
    candidate.len() == expected.len() && bool::from(candidate.ct_eq(expected))
}

/// Encodes one backend-to-signer request.
///
/// # Errors
/// Returns a wire error for unsupported versions, zero key IDs, or bad bounds.
pub fn encode_sign_request(request: &BlindSignRequest) -> Result<Vec<u8>, BlindIssuerWireError> {
    validate_request(request)?;
    let mut bytes = Vec::with_capacity(FRAME_HEADER_BYTES + request.blinded_message.len());
    bytes.extend_from_slice(&REQUEST_MAGIC);
    bytes.extend_from_slice(&request.version.to_be_bytes());
    bytes.extend_from_slice(&request.issuer_key_id);
    bytes.extend_from_slice(&request.blinded_message);
    Ok(bytes)
}

fn decode_sign_request(bytes: &[u8]) -> Result<BlindSignRequest, BlindIssuerWireError> {
    if bytes.len() > MAX_SIGN_REQUEST_BYTES {
        return Err(BlindIssuerWireError::FrameTooLarge);
    }
    if bytes.len() < FRAME_HEADER_BYTES + MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES
        || bytes[..4] != REQUEST_MAGIC
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    let mut issuer_key_id = [0; 32];
    issuer_key_id.copy_from_slice(&bytes[6..FRAME_HEADER_BYTES]);
    let request = BlindSignRequest {
        version,
        issuer_key_id,
        blinded_message: bytes[FRAME_HEADER_BYTES..].to_vec(),
    };
    validate_request(&request)?;
    Ok(request)
}

fn validate_request(request: &BlindSignRequest) -> Result<(), BlindIssuerWireError> {
    if request.version != BLIND_VAULT_BLIND_ADMISSION_VERSION
        || request.issuer_key_id.iter().all(|byte| *byte == 0)
        || !(MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES..=MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES)
            .contains(&request.blinded_message.len())
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    Ok(())
}

fn binary_response(response: &BlindSignResponse) -> Response {
    let mut bytes = Vec::with_capacity(FRAME_HEADER_BYTES + response.blind_signature.len());
    bytes.extend_from_slice(&RESPONSE_MAGIC);
    bytes.extend_from_slice(&response.version.to_be_bytes());
    bytes.extend_from_slice(&response.issuer_key_id);
    bytes.extend_from_slice(&response.blind_signature);
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE),
            (header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE),
        ],
        bytes,
    )
        .into_response()
}

/// Decodes one signer-to-backend response.
///
/// # Errors
/// Returns a wire error for malformed, oversized, or unsupported frames.
pub fn decode_sign_response(bytes: &[u8]) -> Result<BlindSignResponse, BlindIssuerWireError> {
    if bytes.len() > MAX_SIGN_REQUEST_BYTES {
        return Err(BlindIssuerWireError::FrameTooLarge);
    }
    if bytes.len() < FRAME_HEADER_BYTES + MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES
        || bytes[..4] != RESPONSE_MAGIC
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let version = u16::from_be_bytes([bytes[4], bytes[5]]);
    let mut issuer_key_id = [0; 32];
    issuer_key_id.copy_from_slice(&bytes[6..FRAME_HEADER_BYTES]);
    let blind_signature = bytes[FRAME_HEADER_BYTES..].to_vec();
    if version != BLIND_VAULT_BLIND_ADMISSION_VERSION
        || issuer_key_id.iter().all(|byte| *byte == 0)
        || !(MIN_BLIND_VAULT_BLIND_SIGNATURE_BYTES..=MAX_BLIND_VAULT_BLIND_SIGNATURE_BYTES)
            .contains(&blind_signature.len())
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    Ok(BlindSignResponse {
        version,
        issuer_key_id,
        blind_signature,
    })
}

/// Encodes a deterministic, allocation-bounded public epoch snapshot.
///
/// # Errors
/// Returns a wire error when an epoch violates key, ordering, or time bounds.
pub fn encode_epoch_snapshot(
    snapshot: &BlindIssuerEpochSnapshot,
) -> Result<Vec<u8>, BlindIssuerWireError> {
    validate_epoch_snapshot(snapshot)?;
    let capacity = EPOCH_RESPONSE_HEADER_BYTES
        + snapshot
            .epochs
            .iter()
            .map(|epoch| EPOCH_ENTRY_FIXED_BYTES + epoch.public_key_der.len())
            .sum::<usize>();
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(&EPOCH_RESPONSE_MAGIC);
    bytes.extend_from_slice(&INTERNAL_WIRE_VERSION.to_be_bytes());
    bytes.extend_from_slice(&snapshot.generated_at_ms.to_be_bytes());
    let epoch_count =
        u16::try_from(snapshot.epochs.len()).map_err(|_| BlindIssuerWireError::InvalidFrame)?;
    bytes.extend_from_slice(&epoch_count.to_be_bytes());
    for epoch in &snapshot.epochs {
        bytes.extend_from_slice(&epoch.admission_version.to_be_bytes());
        bytes.extend_from_slice(&epoch.issuer_key_id);
        bytes.extend_from_slice(&epoch.not_before_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.expires_at_ms.to_be_bytes());
        bytes.extend_from_slice(&epoch.max_lease_ttl_ms.to_be_bytes());
        let der_length = u16::try_from(epoch.public_key_der.len())
            .map_err(|_| BlindIssuerWireError::InvalidFrame)?;
        bytes.extend_from_slice(&der_length.to_be_bytes());
        bytes.extend_from_slice(&epoch.public_key_der);
    }
    Ok(bytes)
}

/// Decodes and validates one signer public-key epoch snapshot.
///
/// # Errors
/// Returns a wire error for malformed, oversized, unsorted, or stale epochs.
pub fn decode_epoch_snapshot(
    bytes: &[u8],
) -> Result<BlindIssuerEpochSnapshot, BlindIssuerWireError> {
    if bytes.len() > MAX_EPOCH_RESPONSE_BYTES {
        return Err(BlindIssuerWireError::FrameTooLarge);
    }
    if bytes.len() < EPOCH_RESPONSE_HEADER_BYTES || bytes[..4] != EPOCH_RESPONSE_MAGIC {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let mut cursor = 4;
    let wire_version = u16::from_be_bytes(take_array(bytes, &mut cursor)?);
    if wire_version != INTERNAL_WIRE_VERSION {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let generated_at_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
    let epoch_count = usize::from(u16::from_be_bytes(take_array(bytes, &mut cursor)?));
    if epoch_count > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let mut epochs = Vec::with_capacity(epoch_count);
    for _ in 0..epoch_count {
        let admission_version = u16::from_be_bytes(take_array(bytes, &mut cursor)?);
        let issuer_key_id = take_array(bytes, &mut cursor)?;
        let not_before_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
        let expires_at_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
        let max_lease_ttl_ms = u64::from_be_bytes(take_array(bytes, &mut cursor)?);
        let der_length = usize::from(u16::from_be_bytes(take_array(bytes, &mut cursor)?));
        let der_end = cursor
            .checked_add(der_length)
            .ok_or(BlindIssuerWireError::InvalidFrame)?;
        let public_key_der = bytes
            .get(cursor..der_end)
            .ok_or(BlindIssuerWireError::InvalidFrame)?
            .to_vec();
        cursor = der_end;
        epochs.push(BlindVaultBlindIssuerEpoch {
            admission_version,
            issuer_key_id,
            public_key_der,
            not_before_ms,
            expires_at_ms,
            max_lease_ttl_ms,
        });
    }
    if cursor != bytes.len() {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    let snapshot = BlindIssuerEpochSnapshot {
        generated_at_ms,
        epochs,
    };
    validate_epoch_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn validate_epoch_snapshot(
    snapshot: &BlindIssuerEpochSnapshot,
) -> Result<(), BlindIssuerWireError> {
    if snapshot.epochs.len() > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCHS
        || snapshot
            .epochs
            .windows(2)
            .any(|pair| pair[0].issuer_key_id >= pair[1].issuer_key_id)
    {
        return Err(BlindIssuerWireError::InvalidFrame);
    }
    for epoch in &snapshot.epochs {
        let lifetime = epoch
            .expires_at_ms
            .checked_sub(epoch.not_before_ms)
            .ok_or(BlindIssuerWireError::InvalidFrame)?;
        let derived_key_id: [u8; 32] = Sha256::digest(&epoch.public_key_der).into();
        if epoch.admission_version != BLIND_VAULT_BLIND_ADMISSION_VERSION
            || epoch.issuer_key_id.iter().all(|byte| *byte == 0)
            || epoch.issuer_key_id != derived_key_id
            || epoch.public_key_der.is_empty()
            || epoch.public_key_der.len() > MAX_BLIND_VAULT_BLIND_ISSUER_DER_BYTES
            || lifetime == 0
            || lifetime > MAX_BLIND_VAULT_BLIND_ISSUER_EPOCH_MS
            || epoch.expires_at_ms <= snapshot.generated_at_ms
            || epoch.max_lease_ttl_ms == 0
        {
            return Err(BlindIssuerWireError::InvalidFrame);
        }
    }
    Ok(())
}

fn take_array<const LENGTH: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; LENGTH], BlindIssuerWireError> {
    let end = cursor
        .checked_add(LENGTH)
        .ok_or(BlindIssuerWireError::InvalidFrame)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(BlindIssuerWireError::InvalidFrame)?;
    let mut value = [0; LENGTH];
    value.copy_from_slice(slice);
    *cursor = end;
    Ok(value)
}

fn empty_response(status: StatusCode) -> Response {
    (
        status,
        [
            (header::CACHE_CONTROL, CACHE_CONTROL_NO_STORE),
            (header::CONTENT_LENGTH, "0"),
        ],
        Body::empty(),
    )
        .into_response()
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

#[cfg(all(test, unix))]
mod tests {
    // Panicking on fixture/setup errors is intentional in tests; production
    // paths remain fully fallible and return privacy-safe errors.
    #![allow(clippy::expect_used, clippy::similar_names, clippy::too_many_lines)]

    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;

    use aeronyx_core::protocol::blind_vault::BlindVaultBlindAdmissionToken;
    use axum::body::to_bytes;
    use axum::http::Request as HttpRequest;
    use blind_rsa_signatures::{BlindSignature, DefaultRng, KeyPairSha384PSSRandomized};
    use sha2::{Digest, Sha256};
    use tower::ServiceExt;

    use crate::config::{BlindIssuerConfig, BlindIssuerKeyConfig};

    const BACKEND_TOKEN: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFG";

    fn write_secret(path: &std::path::Path, bytes: &[u8]) {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
            .expect("create secret");
        file.write_all(bytes).expect("write secret");
        file.sync_all().expect("sync secret");
    }

    #[tokio::test]
    async fn authenticated_http_signing_finalizes_and_verifies() {
        let now_ms = now_millis();
        let key_pair =
            KeyPairSha384PSSRandomized::generate(&mut DefaultRng, 2048).expect("RSA key pair");
        let private_der = key_pair.sk.to_der().expect("private DER");
        let public_der = key_pair.pk.to_der().expect("public DER");
        let issuer_key_id: [u8; 32] = Sha256::digest(&public_der).into();
        let directory = tempfile::tempdir().expect("temp directory");
        let private_path = directory.path().join("issuer.der");
        write_secret(&private_path, &private_der);
        let config = BlindIssuerConfig {
            listen_addr: "127.0.0.1:9191".to_owned(),
            auth_token_file: directory.path().join("backend.token"),
            max_requests_per_second: 128,
            max_in_flight: 8,
            keys: vec![BlindIssuerKeyConfig {
                private_key_der_file: private_path,
                not_before_unix_secs: now_ms / 1_000 - 60,
                expires_at_unix_secs: now_ms / 1_000 + 24 * 60 * 60,
                max_lease_ttl_secs: 7 * 24 * 60 * 60,
            }],
        };
        let signer = Arc::new(BlindSigner::from_config(&config).expect("signer"));
        let router = build_router(
            Arc::clone(&signer),
            Zeroizing::new(BACKEND_TOKEN.to_vec()),
            2,
            8,
        );

        let unsigned =
            BlindVaultBlindAdmissionToken::new(issuer_key_id, [7; 32], [1; 32], vec![0; 256]);
        let message = unsigned.message_bytes();
        let blinding = key_pair
            .pk
            .blind(&mut DefaultRng, &message)
            .expect("blind message");
        let request = BlindSignRequest {
            version: BLIND_VAULT_BLIND_ADMISSION_VERSION,
            issuer_key_id,
            blinded_message: blinding.blind_message.0.clone(),
        };
        let body = encode_sign_request(&request).expect("encode request");

        let unauthorized = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/internal/v1/blind-sign")
                    .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
                    .body(Body::from(body.clone()))
                    .expect("unauthorized request"),
            )
            .await
            .expect("unauthorized response");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorization = format!(
            "Bearer {}",
            std::str::from_utf8(BACKEND_TOKEN).expect("ASCII backend token")
        );
        let epoch_response = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .uri("/internal/v1/issuer-epochs")
                    .header(header::AUTHORIZATION, &authorization)
                    .body(Body::empty())
                    .expect("epoch request"),
            )
            .await
            .expect("epoch response");
        assert_eq!(epoch_response.status(), StatusCode::OK);
        assert_eq!(
            epoch_response.headers()[header::CONTENT_TYPE],
            BLIND_ISSUER_EPOCH_CONTENT_TYPE
        );
        let epoch_body = to_bytes(epoch_response.into_body(), MAX_EPOCH_RESPONSE_BYTES)
            .await
            .expect("epoch body");
        let epoch_snapshot = decode_epoch_snapshot(&epoch_body).expect("epoch snapshot");
        assert_eq!(epoch_snapshot.epochs.len(), 1);
        assert_eq!(epoch_snapshot.epochs[0].issuer_key_id, issuer_key_id);
        let mut tampered_epoch_body = epoch_body.to_vec();
        *tampered_epoch_body.last_mut().expect("epoch DER byte") ^= 1;
        assert_eq!(
            decode_epoch_snapshot(&tampered_epoch_body),
            Err(BlindIssuerWireError::InvalidFrame)
        );
        let mut trailing_epoch_body = epoch_body.to_vec();
        trailing_epoch_body.push(0);
        assert_eq!(
            decode_epoch_snapshot(&trailing_epoch_body),
            Err(BlindIssuerWireError::InvalidFrame)
        );

        let response = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/internal/v1/blind-sign")
                    .header(header::AUTHORIZATION, &authorization)
                    .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
                    .body(Body::from(body.clone()))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let response_body = to_bytes(response.into_body(), 1_024)
            .await
            .expect("response body");
        let signed_response =
            decode_sign_response(&response_body).expect("blind signature response");
        assert_eq!(signed_response.issuer_key_id, issuer_key_id);
        let blind_signature = BlindSignature::new(signed_response.blind_signature.clone());
        let finalized = key_pair
            .pk
            .finalize(&blind_signature, &blinding, &message)
            .expect("finalize signature");
        key_pair
            .pk
            .verify(&finalized, blinding.msg_randomizer, &message)
            .expect("verify finalized signature");

        let retry = router
            .clone()
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/internal/v1/blind-sign")
                    .header(header::AUTHORIZATION, &authorization)
                    .header(header::CONTENT_TYPE, BLIND_ISSUER_CONTENT_TYPE)
                    .body(Body::from(body))
                    .expect("retry request"),
            )
            .await
            .expect("retry response");
        let retry_body = to_bytes(retry.into_body(), 1_024)
            .await
            .expect("retry body");
        assert_eq!(
            decode_sign_response(&retry_body)
                .expect("retry signature")
                .blind_signature,
            signed_response.blind_signature
        );

        let health = router
            .oneshot(
                HttpRequest::builder()
                    .uri("/internal/v1/health")
                    .body(Body::empty())
                    .expect("health request"),
            )
            .await
            .expect("health response");
        assert_eq!(health.status(), StatusCode::NO_CONTENT);
        assert_eq!(signer.public_epochs(now_ms).len(), 1);
        assert!(signer.has_active_key(now_ms));
        assert!(!signer.has_active_key(now_ms + 2 * 24 * 60 * 60 * 1_000));
    }
}
