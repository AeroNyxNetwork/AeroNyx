// ============================================
// File: crates/aeronyx-server/src/api/reverse_onion_source.rs
// ============================================
//! Authenticated private source API for fixed-class Blind Vault Pull delivery.
//!
//! This router is intentionally composition-only: production mounts it inside
//! the existing MPI-authenticated VPN source router. It accepts no route,
//! endpoint, recipient, payload, or deadline from the caller.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::crypto::keys::IdentityPublicKey;
use aeronyx_core::protocol::discovery::{
    SignedPrivateOnionRecipientAuthorizationV1,
    MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_BYTES,
};
use aeronyx_core::protocol::onion::reverse_delivery::{
    reverse_onion_source_pull_digest, reverse_onion_source_route_id,
    verify_reverse_onion_source_pull,
    ReverseOnionSourcePullRequestV1, MAX_REVERSE_ONION_SOURCE_PULL_REQUEST_BYTES,
    ReverseOnionSourcePullResponseV1, ReverseOnionSourcePullResponseStateV1,
};
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::post,
    Extension, Json, Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Serialize;
use crate::api::mpi::AuthenticatedOwner;
use crate::server::reverse_onion_source_runtime::{
    ReverseOnionSourceLifecycle, ReverseOnionSourceRequestAdmission,
    ReverseOnionSourceRequestPermit, SourceRuntimeError,
};

// [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] MPI uses this exact
// route for pre-auth body bounds and privacy-sensitive resource isolation.
pub(crate) const SOURCE_PULL_PATH: &str = "/api/chat/reverse-onion/source/pull";

// [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] A SaaS JWT must not
// acquire this VPN/MPI source capability if routers are miscomposed.
fn private_source_owner(owner: &AuthenticatedOwner) -> Option<[u8; 32]> {
    match owner {
        AuthenticatedOwner::Local { owner } | AuthenticatedOwner::Remote { owner, .. }
            if *owner != [0; 32] => Some(*owner),
        _ => None,
    }
}

#[derive(Clone)]
// [REVERSE-SOURCE-API-ADMISSION 2026-10-05 by Codex]
struct SourceApiState {
    lifecycle: Arc<ReverseOnionSourceLifecycle>,
    admission: Arc<ReverseOnionSourceRequestAdmission>,
}

/// [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Production mounts via
/// `build_mpi_router_with_reverse_onion_source` on the VPN listener only.
// [REVERSE-ONION-SOURCE-API 2026-10-05 by Codex]
pub(crate) fn build_reverse_onion_source_router(
    lifecycle: Arc<ReverseOnionSourceLifecycle>,
) -> Router {
    let state = SourceApiState {
        admission: lifecycle.request_admission(),
        lifecycle,
    };
    Router::new()
        .route(SOURCE_PULL_PATH, post(submit_pull))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            source_pull_admission,
        ))
        .layer(DefaultBodyLimit::max(MAX_REVERSE_ONION_SOURCE_PULL_REQUEST_BYTES))
        .layer(middleware::from_fn(private_source_response_middleware))
        .with_state(state)
}

// [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Install only outside
// MPI authentication, after composing the source and ordinary MPI routes.
// Exact-path gating leaves every other route's admission behavior unchanged.
pub(super) fn with_mpi_source_admission(
    router: Router, admission: Arc<ReverseOnionSourceRequestAdmission>,
) -> Router {
    router.layer(middleware::from_fn_with_state(admission, source_mpi_admission))
}

async fn source_mpi_admission(
    State(admission): State<Arc<ReverseOnionSourceRequestAdmission>>,
    mut request: Request, next: Next,
) -> Response {
    if request.uri().path() != SOURCE_PULL_PATH { return next.run(request).await; }
    // Unsupported methods can also enter MPI's body authenticator. Bound
    // them here while preserving its normal auth/method-fallback semantics.
    let Some(permit) = admission.try_acquire_request() else {
        return private_source_response(SourceApiFailure::busy().into_response());
    };
    request.extensions_mut().insert(permit.clone());
    let response = next.run(request).await;
    private_source_response(retain_source_response(response, &admission, permit))
}

// [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] A completed Pull
// can hold several MiB. Keep its existing HTTP slot through body delivery;
// outer MPI and inner source layers must wrap the body only once. Small
// rejection/Pending replies preserve their existing immediate-release behavior.
#[derive(Clone)]
struct SourceResponseOwner(Arc<ReverseOnionSourceRequestAdmission>);

fn retain_source_response(response: Response, admission: &Arc<ReverseOnionSourceRequestAdmission>,
    permit: ReverseOnionSourceRequestPermit) -> Response {
    if response.status() != StatusCode::OK
        || response.extensions().get::<SourceResponseOwner>()
            .is_some_and(|owner| Arc::ptr_eq(&owner.0, admission)) {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    match admission.bound_response_body(body, permit) {
        Ok(body) => {
            parts.extensions.insert(SourceResponseOwner(Arc::clone(admission)));
            Response::from_parts(parts, body)
        }
        Err(error) => map_runtime_error(error).into_response(),
    }
}

// [REVERSE-SOURCE-PRIVATE-RESPONSE-HEADERS 2026-10-06 by Codex] Keep this
// outside the body-limit and admission layers so their rejection responses
// receive the same cache policy as handler responses.
async fn private_source_response_middleware(request: Request, next: Next) -> Response {
    private_source_response(next.run(request).await)
}

// [REVERSE-ONION-SOURCE-API 2026-10-05 by Codex] Admission precedes body
// extraction, bounding concurrent JSON parsing as well as downstream work.
async fn source_pull_admission(
    State(state): State<SourceApiState>,
    request: Request,
    next: Next,
) -> Response {
    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] The existing
    // method fallback must not admit or buffer unsupported requests.
    if request.method() != axum::http::Method::POST {
        return next.run(request).await;
    }
    // [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] Validate the
    // authenticated capability before consuming parser permits or body bytes.
    if request.extensions().get::<AuthenticatedOwner>().and_then(private_source_owner).is_none() {
        return SourceApiFailure::unauthorized().into_response();
    }
    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Reuse only this
    // lifecycle's pre-auth slot. Direct authenticated compositions still get
    // their own bounded slot; a foreign local extension cannot bypass either.
    let permit = match request.extensions().get::<ReverseOnionSourceRequestPermit>() {
        Some(permit) if state.admission.recognizes_request(permit) => Some(permit.clone()),
        Some(_) => None,
        None => state.admission.try_acquire_request(),
    };
    let response = if let Some(permit) = permit {
        // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Bound only
        // the pre-effect read. Do not timeout next.run: accepted source work
        // has its own durable, cancellation-surviving lifecycle owner.
        let response = match super::buffer_reverse_onion_request(request).await {
            Ok(request) if !state.admission.is_stopped() => next.run(request).await,
            Ok(_) => SourceApiFailure::unavailable().into_response(),
            Err(response) => response,
        };
        retain_source_response(response, &state.admission, permit)
    } else {
        SourceApiFailure::busy().into_response()
    };
    response
}

// [REVERSE-SOURCE-PRIVATE-RESPONSE-HEADERS 2026-10-06 by Codex] Source pulls
// may return caller-readable results. Apply the same no-store policy to
// successful, rejected, pending, and pre-handler admission responses.
// [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] MPI decorates its own
// early auth/body failures too, before this router's middleware can run.
pub(super) fn private_source_response(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store, no-cache, must-revalidate, private"),
    );
    headers.insert(
        axum::http::header::PRAGMA,
        axum::http::HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        axum::http::HeaderValue::from_static("nosniff"),
    );
    response
}

#[derive(Serialize)]
struct SourcePullError {
    success: bool,
    error: &'static str,
}

#[derive(Debug)]
struct SourceApiFailure {
    status: StatusCode,
    reason: &'static str,
    typed_pending: bool,
    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex]
    retry_after: Option<&'static str>,
}

impl SourceApiFailure {
    fn invalid() -> Self { Self { status: StatusCode::BAD_REQUEST, reason: "source_pull_rejected", typed_pending: false, retry_after: None } }
    fn unauthorized() -> Self { Self { status: StatusCode::UNAUTHORIZED, reason: "source_pull_unauthorized", typed_pending: false, retry_after: None } }
    // [REVERSE-ONION-SOURCE-OWNED-OPERATIONS 2026-10-05 by Codex] An
    // operation may outlive its disconnected waiter; tell exact-route retries
    // to back off while the bounded owner slot drains.
    fn busy() -> Self { Self { status: StatusCode::TOO_MANY_REQUESTS, reason: "source_pull_busy", typed_pending: false, retry_after: Some("1") } }
    fn ambiguous() -> Self { Self { status: StatusCode::CONFLICT, reason: "source_pull_ambiguous", typed_pending: false, retry_after: None } }
    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex]
    fn pending() -> Self { Self { status: StatusCode::ACCEPTED, reason: "source_pull_pending", typed_pending: true, retry_after: Some("1") } }
    fn unavailable() -> Self { Self { status: StatusCode::SERVICE_UNAVAILABLE, reason: "source_pull_unavailable", typed_pending: false, retry_after: None } }
}

impl IntoResponse for SourceApiFailure {
    fn into_response(self) -> Response {
        let mut response = if self.typed_pending {
            (self.status, Json(ReverseOnionSourcePullResponseV1::pending())).into_response()
        } else {
            (self.status, Json(SourcePullError {
                success: false, error: self.reason,
            })).into_response()
        };
        if let Some(retry_after) = self.retry_after {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static(retry_after),
            );
        }
        response
    }
}

#[cfg(test)]
mod pending_response_tests {
    use super::*;
    use axum::{body::Bytes, routing::post};
    use tower::ServiceExt;

    // [PHALA-SOURCE-ADMISSION-CLOCK 2026-10-07 by Codex] Authored, not
    // run: zero-send expiry/stop is never a typed custody-Pending response.
    #[tokio::test]
    async fn expired_or_stopped_admission_never_reports_pending_custody() {
        for (error, expected) in [
            (SourceRuntimeError::Expired, StatusCode::BAD_REQUEST),
            (SourceRuntimeError::Rejected, StatusCode::BAD_REQUEST),
            (SourceRuntimeError::Stopped, StatusCode::SERVICE_UNAVAILABLE),
            (SourceRuntimeError::Unavailable, StatusCode::SERVICE_UNAVAILABLE),
        ] {
            let response = private_source_response(map_runtime_error(error).into_response());
            assert_eq!(response.status(), expected);
            assert!(response.headers().get(axum::http::header::RETRY_AFTER).is_none());
            assert_eq!(response.headers().get(axum::http::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
                Some("no-store, no-cache, must-revalidate, private"));
            let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json.get("success").and_then(|value| value.as_bool()), Some(false));
            assert!(json.get("state").is_none());
        }
    }

    // [REVERSE-ONION-SOURCE-PENDING 2026-10-05 by Codex] Pending is an
    // accepted recovery state, never a completed pull or generic ambiguity.
    #[tokio::test]
    async fn pending_result_is_retryable_without_claiming_completion() {
        let response = SourceApiFailure::pending().into_response();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1"),
        );
        let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
        let typed: ReverseOnionSourcePullResponseV1 = serde_json::from_slice(&body).unwrap();
        assert_eq!(typed.state, ReverseOnionSourcePullResponseStateV1::Pending);
        typed.validate_pending().unwrap();
    }

    // [REVERSE-ONION-SOURCE-OWNED-OPERATIONS 2026-10-05 by Codex] Authored,
    // not run: saturated source ownership returns bounded retry guidance.
    #[tokio::test]
    async fn busy_response_includes_retry_after() {
        let response = private_source_response(SourceApiFailure::busy().into_response());
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok()),
            Some("1"),
        );
        assert_eq!(response.headers().get(axum::http::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
            Some("no-store, no-cache, must-revalidate, private"));
        assert_eq!(response.headers().get(axum::http::header::PRAGMA)
            .and_then(|value| value.to_str().ok()), Some("no-cache"));
        assert_eq!(response.headers().get(axum::http::header::X_CONTENT_TYPE_OPTIONS)
            .and_then(|value| value.to_str().ok()), Some("nosniff"));
    }

    // [REVERSE-SOURCE-PRIVATE-RESPONSE-HEADERS 2026-10-06 by Codex] Authored,
    // not run: the outer middleware must decorate body-limit rejections too.
    #[tokio::test]
    async fn body_limit_rejection_is_non_cacheable() {
        let app = Router::new()
            .route("/private", post(|_: Bytes| async { StatusCode::OK }))
            .layer(DefaultBodyLimit::max(1))
            .layer(middleware::from_fn(private_source_response_middleware));
        let request = Request::builder()
            .method(axum::http::Method::POST)
            .uri("/private")
            .body(axum::body::Body::from("too large"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(response.headers().get(axum::http::header::CACHE_CONTROL)
            .and_then(|value| value.to_str().ok()),
            Some("no-store, no-cache, must-revalidate, private"));
    }
}

async fn submit_pull(
    State(state): State<SourceApiState>,
    Extension(owner): Extension<AuthenticatedOwner>,
    Json(input): Json<ReverseOnionSourcePullRequestV1>,
) -> Result<Response, SourceApiFailure> {
    // [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] Keep the handler's
    // owner contract identical to pre-body admission, not generic SaaS ownership.
    let owner = private_source_owner(&owner).ok_or_else(SourceApiFailure::unauthorized)?;
    let wallet = decode_fixed::<32>(&input.wallet_b64)?;
    let nonce = decode_fixed::<16>(&input.nonce_b64)?;
    let signature = decode_fixed::<64>(&input.signature_b64)?;
    let authorization_bytes = if input.authorization_b64.is_empty() {
        Vec::new()
    } else {
        decode_bounded_base64(
            &input.authorization_b64,
            MAX_PRIVATE_ONION_RECIPIENT_AUTHORIZATION_BYTES,
        )?
    };
    let authorization = if authorization_bytes.is_empty() {
        None
    } else {
        Some(SignedPrivateOnionRecipientAuthorizationV1::decode_canonical(
            &authorization_bytes,
        ).map_err(|_| SourceApiFailure::unauthorized())?)
    };
    if input.version != 1 || wallet != owner || nonce == [0; 16] {
        return Err(SourceApiFailure::unauthorized());
    }
    let digest = reverse_onion_source_pull_digest(
        input.version, &wallet, &nonce, input.request_timestamp,
        &input.pull, &authorization_bytes,
    ).map_err(|_| SourceApiFailure::invalid())?;
    // [REVERSE-ONION-SOURCE-ROUTE-IDENTITY 2026-10-05 by Codex] Signature
    // timestamps authenticate admission but do not alter nonce+Pull identity.
    let route = reverse_onion_source_route_id(&wallet, &nonce, &input.pull)
        .map_err(|_| SourceApiFailure::invalid())?;
    let now = unix_now_secs().map_err(|_| SourceApiFailure::unavailable())?;
    let result = if verify_reverse_onion_source_pull(
        &wallet, &nonce, input.request_timestamp, &input.pull, &authorization_bytes, &signature,
    ).is_ok() {
        match authorization {
            Some(authorization) => state.lifecycle.submit_pull(route, input.pull, authorization, now).await,
            None => state.lifecycle.submit_pull_with_live_authority(route, input.pull, now).await,
        }
    } else {
        // A stale but correctly signed request may only recover its exact
        // owner/nonce/pull route. It may replay that row's exact persisted
        // request only to the same relay and only under the original live
        // authority/deadline; it cannot create a route or choose new payload.
        let public = IdentityPublicKey::from_bytes(&wallet)
            .map_err(|_| SourceApiFailure::unauthorized())?;
        public.verify(&digest, &signature)
            .map_err(|_| SourceApiFailure::unauthorized())?;
        state.lifecycle.resume(route, now).await
    }.map_err(map_runtime_error)?;
    let response = ReverseOnionSourcePullResponseV1::completed(result.response())
        .map_err(|_| SourceApiFailure::unavailable())?;
    let body = response.encode_json().map_err(|_| SourceApiFailure::unavailable())?;
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(body))
        .map_err(|_| SourceApiFailure::unavailable())
}

fn decode_fixed<const N: usize>(encoded: &str) -> Result<[u8; N], SourceApiFailure> {
    if encoded.len() != ((N + 2) / 3) * 4 { return Err(SourceApiFailure::invalid()); }
    let decoded = STANDARD.decode(encoded).map_err(|_| SourceApiFailure::invalid())?;
    if STANDARD.encode(&decoded) != encoded {
        return Err(SourceApiFailure::invalid());
    }
    decoded.try_into().map_err(|_| SourceApiFailure::invalid())
}

// [REVERSE-ONION-SOURCE-API 2026-10-05 by Codex] Signed authority is public,
// but its wire representation is bounded and canonical before protocol decode.
fn decode_bounded_base64(encoded: &str, max_bytes: usize) -> Result<Vec<u8>, SourceApiFailure> {
    let max_encoded = (max_bytes.saturating_add(2) / 3).saturating_mul(4);
    if encoded.is_empty() || encoded.len() > max_encoded {
        return Err(SourceApiFailure::invalid());
    }
    let decoded = STANDARD.decode(encoded).map_err(|_| SourceApiFailure::invalid())?;
    if decoded.is_empty() || decoded.len() > max_bytes || STANDARD.encode(&decoded) != encoded {
        return Err(SourceApiFailure::invalid());
    }
    Ok(decoded)
}

fn map_runtime_error(error: SourceRuntimeError) -> SourceApiFailure {
    match error {
        SourceRuntimeError::Busy => SourceApiFailure::busy(),
        SourceRuntimeError::Pending => SourceApiFailure::pending(),
        SourceRuntimeError::Ambiguous | SourceRuntimeError::Conflict => SourceApiFailure::ambiguous(),
        SourceRuntimeError::Rejected | SourceRuntimeError::Expired => SourceApiFailure::invalid(),
        SourceRuntimeError::Stopped | SourceRuntimeError::Unavailable => SourceApiFailure::unavailable(),
    }
}

fn unix_now_secs() -> Result<u64, std::time::SystemTimeError> {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|value| value.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] Transport fixture bytes.
    use axum::body::Bytes;

    // [PHALA-SOURCE-OPEN-CLOCK 2026-10-07 by Codex] Authored, not run:
    // exercise the actual open/cache journal errors through the private HTTP
    // mapping. A local fault is neither completed data nor retryable Pending.
    #[cfg(unix)]
    #[tokio::test]
    async fn result_clock_fault_maps_to_private_unavailable_without_completed_data() {
        use crate::services::reverse_onion_source::{tests::Fixture, SourceJournalError};
        for cached in [false, true] {
            for unavailable in [false, true] {
                let f = Fixture::new();
                let journal = f.open(f.now());
                f.ready(&journal);
                let at = f.now() + 4;
                if cached { journal.open_result(f.route(), at).unwrap(); }
                let sample = || if unavailable { Err(SourceJournalError::Unavailable) } else { Ok(at - 1) };
                let error = if cached { journal.read_verified_at(f.route(), at, sample) }
                    else { journal.open_result_at(f.route(), at, sample) }.err().unwrap();
                let response = private_source_response(map_runtime_error(error.into()).into_response());
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert!(response.headers().get(axum::http::header::RETRY_AFTER).is_none());
                assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
                    "no-store, no-cache, must-revalidate, private");
                let bytes = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
                assert_eq!(serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
                    serde_json::json!({ "success": false, "error": "source_pull_unavailable" }));
                assert!(*journal.failure_signal().borrow());
            }
        }
    }

    // [PHALA-SOURCE-RETURNED-FAULT 2026-10-07 by Codex] Authored, not
    // run: signed API input reaches actual bounded journal admission. Full
    // quota returns private retryable backpressure, not owner shutdown.
    #[cfg(unix)]
    #[tokio::test]
    async fn full_source_journal_returns_busy_without_stopping_api_or_mutating_route() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use crate::api::PinnedPeerHttpTarget;
        use crate::services::reverse_onion_source::{tests::Fixture, SourcePhase};
        use crate::server::reverse_onion_source_runtime::{
            ReverseOnionSourceRuntime, SourceRuntimeConfig,
            SourceTransport, SourceTransportOutcome, SourceSendAdmission,
        };
        struct NoSendTransport(AtomicUsize);
        #[async_trait::async_trait]
        impl SourceTransport for NoSendTransport {
            async fn post(&self, _: PinnedPeerHttpTarget, _: Bytes, _: Bytes,
                _: SourceSendAdmission) -> SourceTransportOutcome {
                self.0.fetch_add(1, Ordering::SeqCst);
                SourceTransportOutcome::Ambiguous
            }
            async fn query(&self, _: PinnedPeerHttpTarget, _: Bytes,
                _: SourceSendAdmission) -> SourceTransportOutcome {
                self.0.fetch_add(1, Ordering::SeqCst);
                SourceTransportOutcome::Ambiguous
            }
        }
        let fixture = Fixture::new_for_runtime();
        let journal = Arc::new(fixture.open_one_slot(fixture.now()));
        fixture.prepare(&journal, fixture.now());
        let (relay, recipient, authorization) = fixture.policy_parts();
        let source = fixture.source_identity();
        // [PHALA-SOURCE-API-FIXTURE-AUTHORITY 2026-10-08 by Codex]
        // Fresh typed submission requires the actual current gossip authority
        // owner, not a recovery-only descriptor policy with no PeerStore.
        let peers = Arc::new(crate::services::peer_store::PeerStore::new());
        peers.pin_private_onion_route_identities(relay.node_id(), recipient.node_id()).unwrap();
        peers.upsert_verified_from_source(relay.clone(), fixture.now(), "test_pin").unwrap();
        peers.upsert_verified_from_source(recipient.clone(), fixture.now(), "test_pin").unwrap();
        peers.remember_issued_private_onion_authorization(authorization.clone(),
            recipient.node_id(), fixture.now()).unwrap();
        let transport = Arc::new(NoSendTransport(AtomicUsize::new(0)));
        let runtime = Arc::new(ReverseOnionSourceRuntime::new_identity_pinned_for_test(
            journal.clone(), source.clone(), relay.node_id(), recipient.node_id(),
            relay.descriptor.public_endpoint.clone().unwrap(), peers, transport.clone(),
            SourceRuntimeConfig::new(1, std::time::Duration::from_secs(1), 600,
                std::time::Duration::ZERO, std::time::Duration::from_millis(250)).unwrap(),
        ).unwrap());
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        lifecycle.install(runtime.clone()).unwrap();
        let router = build_reverse_onion_source_router(lifecycle.clone());
        let wallet = source.public_key_bytes();
        let nonce = [29; 16];
        let pull = pull();
        let route = reverse_onion_source_route_id(&wallet, &nonce, &pull).unwrap();
        assert_ne!(route, fixture.route());
        let authorization_bytes = authorization.encode_canonical().unwrap();
        for _ in 0..2 {
            let now = unix_now_secs().unwrap();
            let digest = reverse_onion_source_pull_digest(1, &wallet, &nonce, now,
                &pull, &authorization_bytes).unwrap();
            let body = serde_json::json!({
                "version": 1, "wallet_b64": STANDARD.encode(wallet),
                "nonce_b64": STANDARD.encode(nonce), "request_timestamp": now, "pull": &pull,
                "authorization_b64": STANDARD.encode(&authorization_bytes),
                "signature_b64": STANDARD.encode(source.sign(&digest)),
            });
            let request = axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .extension(AuthenticatedOwner::Local { owner: wallet })
                .body(Body::from(serde_json::to_vec(&body).unwrap())).unwrap();
            let response = router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(response.headers()[axum::http::header::RETRY_AFTER], "1");
            assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
                "no-store, no-cache, must-revalidate, private");
            let body = axum::body::to_bytes(response.into_body(), 4096).await.unwrap();
            assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap(),
                serde_json::json!({ "success": false, "error": "source_pull_busy" }));
            assert!(!lifecycle.has_failed());
            assert!(!lifecycle.request_admission().is_stopped());
            assert!(!*journal.failure_signal().borrow());
            assert_eq!(journal.lookup_phase(route, unix_now_secs().unwrap()).unwrap(), None);
            assert_eq!(journal.lookup_phase(fixture.route(), unix_now_secs().unwrap()).unwrap(), Some(SourcePhase::Prepared));
        }
        assert_eq!(transport.0.load(Ordering::SeqCst), 0);
        lifecycle.shutdown_and_drain().await.unwrap();
    }

    // [PHALA-SOURCE-FAILURE-SUPERVISION 2026-10-07 by Codex] Authored, not
    // run: panic or a failed POST-observation fence closes body intake before
    // the server supervisor is polled. Neither fault becomes typed Pending.
    #[cfg(unix)]
    #[tokio::test]
    async fn owned_source_fault_closes_source_api_before_supervisor_or_body_read() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use crate::api::PinnedPeerHttpTarget;
        use crate::server::reverse_onion_source_runtime::{
            ReverseOnionSourceRuntime, SourcePinnedRelayPolicy, SourceRuntimeConfig,
            SourceTransport, SourceTransportOutcome,
            SourceSendAdmission,
        };
        use crate::services::reverse_onion_source::tests::Fixture;
        use crate::services::reverse_onion_source::ReverseOnionSourceJournal;
        struct FaultTransport {
            posts: Arc<AtomicUsize>,
            fail_observation: Option<Arc<ReverseOnionSourceJournal>>,
            // [PHALA-TRANSPORT-OBSERVATION 2026-10-07 by Codex] A
            // carrier-local clock fault must survive an ambiguous outcome.
            fail_transport_clock: bool,
        }
        #[async_trait::async_trait]
        impl SourceTransport for FaultTransport {
            // [PHALA-HTTP-SEND-FENCE 2026-10-07 by Codex] Admit before
            // injecting an observed POST fault; a local stop is not one.
            async fn post(&self, _: PinnedPeerHttpTarget, _: axum::body::Bytes, _: axum::body::Bytes,
                admission: SourceSendAdmission) -> SourceTransportOutcome {
                if let Err(proof) = admission.check() { return SourceTransportOutcome::NotSent(proof); }
                self.posts.fetch_add(1, Ordering::SeqCst);
                if self.fail_transport_clock {
                    assert!(admission.check_at(Err(SourceRuntimeError::Unavailable)).is_err());
                    return SourceTransportOutcome::Ambiguous;
                }
                if let Some(journal) = &self.fail_observation {
                    journal.fail_next_commit_fence();
                    return SourceTransportOutcome::Ambiguous;
                }
                panic!("synthetic transport unwind after entry");
            }
            async fn query(&self, _: PinnedPeerHttpTarget, _: axum::body::Bytes,
                _: SourceSendAdmission) -> SourceTransportOutcome {
                panic!("stopped source must not query");
            }
        }
        // [PHALA-SOURCE-JOURNAL-FAULT 2026-10-07 by Codex] Also fail a
        // direct journal read: no runtime waiter/supervisor observes it first.
        // [PHALA-SOURCE-OBSERVATION-FLOOR 2026-10-07 by Codex] The
        // impossible local floor exercises a real pre-transport observer fault,
        // not a client-provided timestamp or a changed operating-system clock.
        for (fail_observation, direct_journal_fault, local_clock_fault, fail_transport_clock) in [
            (false, false, false, false), (true, false, false, false), (false, true, false, false),
            (false, false, true, false), (false, false, false, true),
        ] {
            let fixture = Fixture::new_for_runtime();
            let journal = Arc::new(fixture.open(fixture.now()));
            let (relay, recipient, authorization) = fixture.policy_parts();
            let policy = Arc::new(SourcePinnedRelayPolicy::new(
                fixture.source_identity().public_key_bytes(), relay, recipient, authorization, fixture.now(),
            ).unwrap());
            let posts = Arc::new(AtomicUsize::new(0));
            let transport = Arc::new(FaultTransport {
                posts: posts.clone(),
                fail_observation: fail_observation.then(|| journal.clone()),
                fail_transport_clock,
            });
            let runtime = Arc::new(ReverseOnionSourceRuntime::new(
                journal.clone(), fixture.source_identity(), policy.clone(), transport,
                SourceRuntimeConfig::new(1, std::time::Duration::from_secs(1), 600,
                    std::time::Duration::ZERO, std::time::Duration::from_millis(250)).unwrap(),
            ).unwrap());
            let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
            lifecycle.install(runtime.clone()).unwrap();
            let (request, expected, session, terminal) = fixture.runtime_admission_parts();
            let error = if local_clock_fault {
                lifecycle.resume(fixture.route(), u64::MAX).await.err().unwrap()
            } else if direct_journal_fault {
                journal.fail_next_commit_fence();
                assert_eq!(journal.lookup_phase(fixture.route(), fixture.now()).err(),
                    Some(crate::services::reverse_onion_source::SourceJournalError::Unavailable));
                SourceRuntimeError::Unavailable
            } else {
                lifecycle.dispatch(request, expected, session, fixture.now() + 600,
                    terminal, fixture.now(), policy).await.err().unwrap()
            };
            assert_eq!(error, SourceRuntimeError::Unavailable);
            assert_eq!(map_runtime_error(error).into_response().status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(lifecycle.request_admission().is_stopped());
            assert!(lifecycle.has_failed());
            if fail_transport_clock {
                assert_eq!(journal.lookup_phase(fixture.route(),
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()).unwrap(),
                    Some(crate::services::reverse_onion_source::SourcePhase::Armed));
            }
            assert_eq!(runtime.resume(fixture.route(), fixture.now()).await.err(), Some(SourceRuntimeError::Stopped));
            let polls = Arc::new(AtomicUsize::new(0));
            let reader_polls = polls.clone();
            let body = Body::from_stream(futures::stream::poll_fn(move |_| {
                reader_polls.fetch_add(1, Ordering::SeqCst);
                std::task::Poll::<Option<Result<axum::body::Bytes, std::io::Error>>>::Pending
            }));
            let request = axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .extension(AuthenticatedOwner::Local { owner: fixture.source_identity().public_key_bytes() })
                .body(body).unwrap();
            let response = build_reverse_onion_source_router(lifecycle.clone()).oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
            assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
                "no-store, no-cache, must-revalidate, private");
            assert_eq!(polls.load(Ordering::SeqCst), 0);
            assert_eq!(posts.load(Ordering::SeqCst), if direct_journal_fault || local_clock_fault { 0 } else { 1 });
            // A failed durability fence is a runtime fault, not a task unwind.
            let expected_drain = if fail_observation || direct_journal_fault || local_clock_fault || fail_transport_clock { Ok(()) }
                else { Err(SourceRuntimeError::Unavailable) };
            assert_eq!(lifecycle.shutdown_and_drain().await, expected_drain);
            assert_eq!(lifecycle.shutdown_and_drain().await, expected_drain);
        }
    }

    // [PHALA-SOURCE-JOURNAL-LANE 2026-10-07 by Codex] Authored, not run:
    // failure to durably record observed ambiguity, and the resulting stopped
    // owner, must never produce typed Pending or Completed at the API boundary.
    #[tokio::test]
    async fn durability_failure_and_stopped_owner_remain_private_unavailable_responses() {
        for error in [SourceRuntimeError::Unavailable, SourceRuntimeError::Stopped] {
            let response = private_source_response(map_runtime_error(error).into_response());
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
                "no-store, no-cache, must-revalidate, private");
            assert!(!response.headers().contains_key(axum::http::header::RETRY_AFTER));
            let body = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body, serde_json::json!({
                "success": false, "error": "source_pull_unavailable",
            }));
        }
    }
    use axum::body::Body;
    use aeronyx_core::protocol::blind_vault::BlindVaultPullRequest;
    use aeronyx_core::protocol::auth::signed_message_digest;
    use aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_SOURCE_PULL_DOMAIN;
    use tower::ServiceExt;

    fn pull() -> BlindVaultPullRequest {
        BlindVaultPullRequest {
            version: 1,
            lease_id: [5; 32],
            read_capability: [6; 32],
            continuation_cursor: Vec::new(),
            limit: 1,
        }
    }

    // [PHALA-SOURCE-RESPONSE-DRAIN 2026-10-07 by Codex] Authored, not
    // run: nested MPI/source layers retain only one response owner and one
    // slot. Dropping a response is delivery cancellation, not durable revocation.
    #[tokio::test]
    async fn successful_response_is_wrapped_once_and_drop_releases_its_slot() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        for consume in [false, true] {
            let permit = admission.try_acquire_request().unwrap();
            let payload = vec![46; 64 * 1024];
            let response = Response::builder().status(StatusCode::OK)
                .header("content-type", "application/json").body(Body::from(payload.clone())).unwrap();
            let response = retain_source_response(response, &admission, permit.clone());
            let response = private_source_response(retain_source_response(response, &admission, permit));
            assert_eq!(response.status(), StatusCode::OK, "double wrapping would exhaust the one-entry registry");
            assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
                "no-store, no-cache, must-revalidate, private");
            assert!(admission.try_acquire().is_none());
            if consume {
                assert_eq!(axum::body::to_bytes(response.into_body(), payload.len()).await.unwrap().as_ref(),
                    payload.as_slice());
            } else { drop(response); }
            assert!(admission.try_acquire().is_some());
        }
        lifecycle.shutdown_and_drain().await.unwrap();
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Authored, not run:
    // pre-effect buffering has a deadline, but accepted DB work does not.
    #[tokio::test(start_paused = true)]
    async fn slow_source_body_times_out_privately_and_releases_admission() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        let body = Body::from_stream(futures::stream::pending::<Result<axum::body::Bytes, std::io::Error>>());
        let request = axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] })
            .body(body).unwrap();
        let mut response = Box::pin(build_reverse_onion_source_router(lifecycle.clone()).oneshot(request));
        assert!(futures::poll!(response.as_mut()).is_pending());
        assert!(admission.try_acquire().is_none());
        lifecycle.request_stop();
        let mut drain = Box::pin(lifecycle.shutdown_and_drain());
        assert!(futures::poll!(drain.as_mut()).is_pending());
        tokio::time::advance(super::super::REVERSE_ONION_REQUEST_BODY_TIMEOUT).await;
        let response = response.await.unwrap();
        assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
        assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
            "no-store, no-cache, must-revalidate, private");
        drain.await.unwrap();
        assert!(admission.is_stopped());
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Authored, not run.
    #[tokio::test]
    async fn source_body_cancellation_releases_pre_effect_permit() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        let request = axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] })
            .body(Body::from_stream(futures::stream::pending::<Result<axum::body::Bytes, std::io::Error>>()))
            .unwrap();
        let mut response = Box::pin(build_reverse_onion_source_router(lifecycle.clone()).oneshot(request));
        assert!(futures::poll!(response.as_mut()).is_pending());
        assert!(admission.try_acquire().is_none());
        drop(response);
        assert!(admission.try_acquire().is_some());
        lifecycle.shutdown_and_drain().await.unwrap();
    }

    // [PHALA-SOURCE-HTTP-BOUNDARY 2026-10-07 by Codex] Authored, not run:
    // missing/SaaS/zero owners cannot read a body or contend for source slots.
    #[tokio::test]
    async fn non_private_owners_are_rejected_before_body_and_admission() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        for owner in [
            None,
            Some(AuthenticatedOwner::Local { owner: [0; 32] }),
            Some(AuthenticatedOwner::Remote { owner: [0; 32], owner_hex: hex::encode([0; 32]) }),
            Some(AuthenticatedOwner::Saas { owner: [1; 32], owner_hex: hex::encode([1; 32]) }),
        ] {
            for saturated in [false, true] {
                let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
                let admission = lifecycle.request_admission();
                let held = saturated.then(|| admission.try_acquire().unwrap());
                let polls = Arc::new(AtomicUsize::new(0));
                let reader_polls = polls.clone();
                let body = Body::from_stream(futures::stream::poll_fn(move |_| {
                    reader_polls.fetch_add(1, Ordering::SeqCst);
                    std::task::Poll::<Option<Result<axum::body::Bytes, std::io::Error>>>::Pending
                }));
                let mut request = axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
                    .header(axum::http::header::CONTENT_TYPE, "application/json");
                if let Some(owner) = owner.clone() { request = request.extension(owner); }
                let response = build_reverse_onion_source_router(lifecycle.clone())
                    .oneshot(request.body(body).unwrap()).await.unwrap();
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
                assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
                    "no-store, no-cache, must-revalidate, private");
                assert!(!response.headers().contains_key(axum::http::header::RETRY_AFTER));
                assert_eq!(polls.load(Ordering::SeqCst), 0);
                let bytes = axum::body::to_bytes(response.into_body(), 1024).await.unwrap();
                assert_eq!(serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
                    serde_json::json!({ "success": false, "error": "source_pull_unauthorized" }));
                drop(held);
                assert!(admission.try_acquire().is_some());
                lifecycle.shutdown_and_drain().await.unwrap();
            }
        }
    }

    // [PHALA-SOURCE-PREAUTH-DRAIN 2026-10-07 by Codex] Authored, not run:
    // local extension provenance is checked, not inferred from its presence.
    #[tokio::test]
    async fn source_reuses_only_its_own_preauth_permit() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        let foreign = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let foreign_admission = foreign.request_admission();
        let token = foreign_admission.try_acquire_request().unwrap();
        let polls = Arc::new(AtomicUsize::new(0));
        let reader_polls = polls.clone();
        let body = Body::from_stream(futures::stream::poll_fn(move |_| {
            reader_polls.fetch_add(1, Ordering::SeqCst);
            std::task::Poll::<Option<Result<axum::body::Bytes, std::io::Error>>>::Pending
        }));
        let router = build_reverse_onion_source_router(lifecycle.clone());
        let response = router.clone().oneshot(axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
            .header("content-type", "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] }).extension(token.clone())
            .body(body).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert!(admission.try_acquire().is_some());
        assert!(foreign_admission.try_acquire().is_none());
        drop(token);
        let token = admission.try_acquire_request().unwrap();
        let response = router.oneshot(axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
            .header("content-type", "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] }).extension(token.clone())
            .body(Body::from("{}")).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY,
            "the shared slot must reach JSON extraction without reacquiring capacity");
        assert!(admission.try_acquire().is_none(), "the original owner still retains its slot");
        drop(token);
        assert!(admission.try_acquire().is_some());
        lifecycle.shutdown_and_drain().await.unwrap();
        foreign.shutdown_and_drain().await.unwrap();
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Authored, not run.
    #[tokio::test]
    async fn stop_during_source_body_read_rejects_before_json_extraction() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        let (sender, receiver) = tokio::sync::mpsc::channel::<Result<axum::body::Bytes, std::io::Error>>(1);
        let body = Body::from_stream(futures::stream::unfold(receiver, |mut receiver| async {
            receiver.recv().await.map(|chunk| (chunk, receiver))
        }));
        let request = axum::http::Request::builder().method("POST").uri(SOURCE_PULL_PATH)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] })
            .body(body).unwrap();
        let mut response = Box::pin(build_reverse_onion_source_router(lifecycle.clone()).oneshot(request));
        assert!(futures::poll!(response.as_mut()).is_pending());
        assert!(admission.try_acquire().is_none());
        lifecycle.request_stop();
        sender.send(Ok(axum::body::Bytes::from_static(b"{"))).await.unwrap();
        drop(sender);
        let response = response.await.unwrap();
        // Malformed JSON would yield a 4xx if parsing ran after stop.
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[axum::http::header::CACHE_CONTROL],
            "no-store, no-cache, must-revalidate, private");
    }

    #[tokio::test]
    async fn saturated_admission_rejects_before_json_extraction() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let admission = lifecycle.request_admission();
        let _held_permit = admission.try_acquire().unwrap();
        let router = build_reverse_onion_source_router(lifecycle);
        let request = axum::http::Request::builder()
            .method("POST")
            .uri(SOURCE_PULL_PATH)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] })
            .body(Body::from("{"))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    // [REVERSE-SOURCE-API-ADMISSION 2026-10-05 by Codex] Authored, not run:
    // shutdown closes the exact gate used before request-body extraction.
    #[tokio::test]
    async fn stopped_lifecycle_rejects_before_json_extraction() {
        let lifecycle = Arc::new(ReverseOnionSourceLifecycle::with_request_limit(1));
        let router = build_reverse_onion_source_router(Arc::clone(&lifecycle));
        lifecycle.request_stop();
        let request = axum::http::Request::builder()
            .method("POST")
            .uri(SOURCE_PULL_PATH)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .extension(AuthenticatedOwner::Local { owner: [1; 32] })
            .body(Body::from("{"))
            .unwrap();

        let response = router.oneshot(request).await.unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn route_id_binds_owner_nonce_and_exact_pull_frame() {
        let owner = [1; 32];
        let nonce = [2; 16];
        let first = reverse_onion_source_route_id(&owner, &nonce, &pull()).unwrap();
        assert_ne!(first, reverse_onion_source_route_id(&[3; 32], &nonce, &pull()).unwrap());
        assert_ne!(first, reverse_onion_source_route_id(&owner, &[4; 16], &pull()).unwrap());
        let mut different = pull();
        different.read_capability = [7; 32];
        assert_ne!(first, reverse_onion_source_route_id(&owner, &nonce, &different).unwrap());
    }

    #[test]
    fn request_signature_domain_is_not_transport_signature_domain() {
        let version = [1];
        let owner = [2; 32];
        let nonce = [3; 16];
        let timestamp = 1_760_000_000u64.to_be_bytes();
        let frame = b"pull-frame";
        let digest = signed_message_digest(REVERSE_ONION_SOURCE_PULL_DOMAIN, &[
            &version[..], &owner[..], &nonce[..], &timestamp[..], frame,
        ]);
        assert_ne!(digest, signed_message_digest("AeroNyx-ChatPull-v2-http", &[&version[..], frame]));
        // [REVERSE-ONION-AUTHORITY-RENEWAL 2026-10-05 by Codex]
        let with_auth = signed_message_digest(
            REVERSE_ONION_SOURCE_PULL_DOMAIN,
            &[&version[..], &owner[..], &nonce[..], &timestamp[..], frame, b"signed-authority"],
        );
        let with_other_auth = signed_message_digest(
            REVERSE_ONION_SOURCE_PULL_DOMAIN,
            &[&version[..], &owner[..], &nonce[..], &timestamp[..], frame, b"other-authority"],
        );
        assert_ne!(with_auth, with_other_auth);
    }

    // [REVERSE-ONION-SOURCE-PULL-VECTOR 2026-10-06 by Codex] The frozen
    // fixture checks JSON decoding, canonical fields, and signature bytes.
    // Its fixed timestamp intentionally does not exercise the live freshness window.
    #[test]
    fn source_pull_v1_frozen_request_matches_api_codec_and_signature() {
        let encoded = r#"{"version":1,"wallet_b64":"iojj3XQJ8ZX9UtstPLpdcspnCb8dlBIb83SIAbQPb1w=","nonce_b64":"AgICAgICAgICAgICAgICAg==","request_timestamp":1800000000,"pull":{"version":1,"lease_id":[31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31],"read_capability":[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32],"continuation_cursor":[],"limit":1},"authorization_b64":"","signature_b64":"in+w6yBVxITkOExTz8MgXSLrzYi5LMDiPS0K1Pp7CEux7LsfrtLGAtu9DMkXIY/VakAHopteatNIJKsFxerHAg=="}"#;
        let request: ReverseOnionSourcePullRequestV1 = serde_json::from_str(encoded).unwrap();
        let wallet = decode_fixed::<32>(&request.wallet_b64).unwrap();
        let nonce = decode_fixed::<16>(&request.nonce_b64).unwrap();
        let signature = decode_fixed::<64>(&request.signature_b64).unwrap();
        assert_eq!(hex::encode(wallet), "8a88e3dd7409f195fd52db2d3cba5d72ca6709bf1d94121bf3748801b40f6f5c");
        assert_eq!(hex::encode(nonce), "02020202020202020202020202020202");
        let digest = reverse_onion_source_pull_digest(
            request.version,
            &wallet,
            &nonce,
            request.request_timestamp,
            &request.pull,
            &[],
        ).unwrap();
        IdentityPublicKey::from_bytes(&wallet).unwrap().verify(&digest, &signature).unwrap();
        // [PHALA-PULL-VECTOR-REPAIR 2026-10-08 by Codex] Independent
        // SHA-256 of domain/owner/nonce/u64 BE length/83-byte canonical Pull.
        assert_eq!(
            hex::encode(reverse_onion_source_route_id(&wallet, &nonce, &request.pull).unwrap()),
            "2735e1fb505ae0195b5efdd1298b5ac6",
        );
    }
}
