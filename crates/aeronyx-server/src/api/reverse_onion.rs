// ============================================
// File: crates/aeronyx-server/src/api/reverse_onion.rs
// ============================================
//! Default-off adjacent-hop reverse-onion Claim/Result adapter.
//!
//! [REVERSE-ONION-API-ADAPTER 2026-10-04 by Codex] This module is deliberately
//! mounted only by explicit node-peer composition. Router/auth/state ownership
//! remains with that owner. The adapter accepts only canonical core frames and uses
//! the durable queue boundary; it never exposes source completion or route
//! payload data.
//!
//! The immediate recipient is pinned by authenticated/configured state. A
//! caller cannot select a recipient, route deadline, queue key, or envelope
//! commitment. Result recovery uses the queue's read-only authenticated
//! context lookup and calls the core Result verifier before any CAS write.
//!
//! [REVERSE-ONION-CLAIM-RETRY 2026-10-04 by Codex] Historical exact Claim
//! bytes are admitted through queue idempotence after signature/R/P checks;
//! only a genuinely new Claim is subjected to fresh 30-second validation.
//!
//! [REVERSE-ONION-SOURCE-QUERY 2026-10-04 by Codex] The source evidence
//! handler is mounted only with the configured queue. It verifies a signed fixed-size query,
//! reads only the exact durable source-bound snapshot, and never mutates the
//! queue or emits unsigned absence claims.
//!
//! [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] A Claim with no queued
//! item returns HTTP 200 plus a short-lived relay-signed receipt bound to the
//! exact Claim; empty 204 is never a poll-completion signal. Rejections and
//! transient failures do not retire recipient state.
//!
//! Last Modified: v0.1.0-ReverseOnionApiAdapter - Unregistered Claim/Result
//! boundary with bounded blocking, source evidence, and deterministic coarse
//! replies.

// [REVERSE-ONION-ROUTER-ADAPTER 2026-10-04 by Codex] Keep the HTTP surface
// additive and default-unmounted: composition owns whether this router is
// merged into a node-peer listener. The handlers below accept only bounded
// canonical binary frames and never expose an admin or JSON control surface.

use std::sync::Arc;

use aeronyx_core::crypto::keys::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::chat::decode_blind_relay_envelope;
use aeronyx_core::protocol::onion::reverse_delivery::{
    ReverseOnionFrameV1, ReverseOnionKindV1, ReverseOnionNoWorkReceiptV1, ReverseOnionSourceEvidenceV1,
    ReverseOnionSourceQueryV1, MAX_REVERSE_ONION_FRAME_BYTES,
    MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES, REVERSE_ONION_SOURCE_QUERY_BYTES,
};
use aeronyx_core::protocol::onion::reverse_delivery::MAX_REVERSE_ONION_CLAIM_BYTES;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::extract::Request;
use axum::middleware::{self, Next};
use axum::Extension;
use axum::http::StatusCode;
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use rand::{rngs::OsRng, RngCore};
use tokio::sync::Semaphore;

use crate::services::reverse_onion_queue::{
    ReverseOnionQueueCompletion, ReverseOnionQueueError, ReverseOnionQueueIssue,
    ReverseOnionQueueLeaseMaterial, ReverseOnionQueueResultContext,
};
use crate::services::reverse_onion_queue_db::{
    ReverseOnionQueueDb, ReverseOnionQueueDbError,
};
use crate::api::chat_peer::PrivateBlindRelayAdmission;
use crate::services::peer_store::PeerStore;

/// The raw core Claim/Result frame bound is also the HTTP body bound. Axum
/// route registration must install the same DefaultBodyLimit before this
/// adapter is exposed; this check remains mandatory at the function boundary.
pub(crate) const MAX_REVERSE_ONION_API_BODY_BYTES: usize =
    MAX_REVERSE_ONION_FRAME_BYTES;

/// Coarse body/status response. Success bodies are exact canonical core frame
/// bytes; every failure body is empty and carries no IDs, paths, or payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReverseOnionApiReply {
    status: StatusCode,
    body: Bytes,
}

impl ReverseOnionApiReply {
    pub(crate) fn status(&self) -> StatusCode {
        self.status
    }

    pub(crate) fn body(&self) -> &[u8] {
        self.body.as_ref()
    }

    fn frame(status: StatusCode, frame: Vec<u8>) -> Self {
        Self {
            status,
            body: Bytes::from(frame),
        }
    }

    fn empty(status: StatusCode) -> Self {
        Self {
            status,
            body: Bytes::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReverseOnionApiConfigError {
    Invalid,
}

/// One configured/authenticated recipient view of the queue.
///
/// A separate instance is required for each allowlisted recipient identity;
/// no request field can replace `recipient` after construction.
pub(crate) struct ReverseOnionApi {
    queue: Arc<ReverseOnionQueueDb>,
    relay: Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    blocking: Arc<Semaphore>,
    // [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] HTTP bodies retain
    // the admitted queue slot through delivery, not merely SQLite completion.
    responses: Arc<super::ReverseOnionResponseRegistry>,
    // [PHALA-QUEUE-LIFECYCLE-OWNER 2026-10-07 by Codex] Live composition
    // replaces this with the ciphertext admission owner's exact stop signal.
    stopped: Arc<std::sync::atomic::AtomicBool>,
    max_in_flight: u32,
    allow_new_claims: bool,
    live_authority: Option<(Arc<PrivateBlindRelayAdmission>, Arc<PeerStore>)>,
}

// [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Private middleware
// capability, never decoded from a request. Its Arc retains the SAME queue
// permit during body parsing, handler work and cancellation-surviving SQL.
#[derive(Clone)]
struct ReverseOnionApiPermit(Arc<tokio::sync::OwnedSemaphorePermit>);

impl ReverseOnionApi {
    /// Defaults are intentionally absent: router/state wiring must opt in with
    /// an explicitly opened queue and an authenticated recipient allowlist.
    pub(crate) fn new(
        queue: Arc<ReverseOnionQueueDb>,
        relay: Arc<IdentityKeyPair>,
        recipient: [u8; 32],
        max_in_flight: usize,
    ) -> Result<Self, ReverseOnionApiConfigError> {
        validate_recipient(relay.public_key_bytes(), recipient, max_in_flight)?;
        Ok(Self {
            queue,
            relay,
            recipient,
            blocking: Arc::new(Semaphore::new(max_in_flight)),
            responses: Arc::new(super::ReverseOnionResponseRegistry::new(max_in_flight)),
            stopped: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            max_in_flight: max_in_flight as u32,
            allow_new_claims: false,
            live_authority: None,
        })
    }

    // [REVERSE-ONION-LIVE-CLAIM-AUTH 2026-10-05 by Codex] Live operation is
    // opt-in only when this API shares the relay admission's current authority
    // and bounded DB semaphore. Recovery APIs intentionally omit this owner.
    pub(crate) fn with_live_private_admission(
        mut self,
        admission: Arc<PrivateBlindRelayAdmission>,
        peer_store: Arc<PeerStore>,
    ) -> Result<Self, ReverseOnionApiConfigError> {
        let stop_signal = admission.queue_stop_signal();
        if admission.local_relay_node_id() != self.relay.public_key_bytes()
            || admission.recipient_node_id() != self.recipient
            || admission.queue_capacity() != self.max_in_flight as usize
            // [PHALA-QUEUE-LIFECYCLE-OWNER 2026-10-07 by Codex] Matching
            // identities/capacity cannot bind a different durable queue owner.
            || !Arc::ptr_eq(&self.queue, admission.queue())
            || self.live_authority.is_some()
            || self.stopped.load(std::sync::atomic::Ordering::Acquire)
            || stop_signal.load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(ReverseOnionApiConfigError::Invalid);
        }
        self.blocking = admission.queue_semaphore();
        self.responses = admission.queue_response_registry();
        self.stopped = stop_signal;
        self.live_authority = Some((admission, peer_store));
        self.allow_new_claims = true;
        Ok(self)
    }

    // [REVERSE-RECOVERY-BOOT 2026-10-05 by Codex] Exact durable Claim
    // replay precedes the queue's fresh-Claim verifier. Rejecting in that
    // verifier preserves historical bytes without minting any new lease.
    pub(crate) fn recovery_only(mut self) -> Self {
        self.allow_new_claims = false;
        self
    }

    // [REVERSE-QUEUE-DRAIN 2026-10-05 by Codex] Stop new admission without
    // closing the semaphore: acquiring every permit waits for detached DB work.
    pub(crate) fn request_stop(&self) {
        self.stopped.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub(crate) async fn shutdown_and_drain(&self) {
        self.request_stop();
        self.responses.drain(Arc::clone(&self.blocking), self.max_in_flight).await;
    }

    fn try_permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, ()> {
        self.responses.expire();
        if self.stopped.load(std::sync::atomic::Ordering::Acquire) { return Err(()); }
        let permit = self.blocking.clone().try_acquire_owned().map_err(|_| ())?;
        if self.stopped.load(std::sync::atomic::Ordering::Acquire) { return Err(()); }
        Ok(permit)
    }

    /// Handles one raw canonical Claim frame. The authenticated/configured P
    /// is the only accepted immediate recipient; the body cannot select P.
    pub(crate) async fn handle_claim(&self, body: Bytes, now: u64) -> ReverseOnionApiReply {
        if body.is_empty() || body.len() > MAX_REVERSE_ONION_API_BODY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let permit = match self.try_permit() {
            Ok(permit) => Arc::new(permit),
            Err(_) => return ReverseOnionApiReply::empty(StatusCode::TOO_MANY_REQUESTS),
        };
        self.handle_claim_admitted(body, now, permit).await
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] HTTP middleware
    // already owns admission before buffering; do not acquire a second permit.
    async fn handle_claim_admitted(&self, body: Bytes, now: u64,
        permit: Arc<tokio::sync::OwnedSemaphorePermit>) -> ReverseOnionApiReply {
        let received = std::time::Instant::now();
        let allow_new_claims = self.allow_new_claims;
        if body.is_empty() || body.len() > MAX_REVERSE_ONION_CLAIM_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let queue = Arc::clone(&self.queue);
        let relay = Arc::clone(&self.relay);
        let recipient = self.recipient;
        let live_authority = self.live_authority.clone();
        let body = body.to_vec();
        match tokio::task::spawn_blocking(move || {
            // [REVERSE-ONION-API-CANCELLATION 2026-10-04 by Codex] Keep the
            // owned permit inside the blocking closure. Dropping a cancelled
            // HTTP JoinHandle must not advertise capacity while SQLite work
            // continues.
            let _permit = permit;
            let Some(now) = worker_time(now, received.elapsed()) else {
                return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE);
            };
            handle_claim_with_live_authority(
                &queue, &relay, recipient, &body, now, allow_new_claims, live_authority,
            )
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
        }
    }

    /// Handles one raw canonical Result frame. The Result signer P is checked
    /// against the pinned recipient before lookup; the queue supplies the
    /// stored Claim/Lease context used by the core verifier.
    pub(crate) async fn handle_result(&self, body: Bytes, now: u64) -> ReverseOnionApiReply {
        if body.is_empty() || body.len() > MAX_REVERSE_ONION_API_BODY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let permit = match self.try_permit() {
            Ok(permit) => Arc::new(permit),
            Err(_) => return ReverseOnionApiReply::empty(StatusCode::TOO_MANY_REQUESTS),
        };
        self.handle_result_admitted(body, now, permit).await
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex]
    async fn handle_result_admitted(&self, body: Bytes, now: u64,
        permit: Arc<tokio::sync::OwnedSemaphorePermit>) -> ReverseOnionApiReply {
        let received = std::time::Instant::now();
        if body.is_empty() || body.len() > MAX_REVERSE_ONION_API_BODY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let queue = Arc::clone(&self.queue);
        let relay = Arc::clone(&self.relay);
        let recipient = self.recipient;
        let body = body.to_vec();
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            handle_result_blocking(&queue, &relay, recipient, &body, now, received)
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
        }
    }

    /// Handles one canonical source query. Explicit queue composition mounts
    /// this bounded read-only bridge for the source
    /// evidence protocol, not a new public readiness signal.
    pub(crate) async fn handle_source_query(
        &self,
        body: Bytes,
        now: u64,
    ) -> ReverseOnionApiReply {
        if body.len() != REVERSE_ONION_SOURCE_QUERY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let permit = match self.try_permit() {
            Ok(permit) => Arc::new(permit),
            Err(_) => return ReverseOnionApiReply::empty(StatusCode::TOO_MANY_REQUESTS),
        };
        self.handle_source_query_admitted(body, now, permit).await
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex]
    async fn handle_source_query_admitted(&self, body: Bytes, now: u64,
        permit: Arc<tokio::sync::OwnedSemaphorePermit>) -> ReverseOnionApiReply {
        let received = std::time::Instant::now();
        if body.len() != REVERSE_ONION_SOURCE_QUERY_BYTES {
            return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
        }
        let queue = Arc::clone(&self.queue);
        let relay = Arc::clone(&self.relay);
        let recipient = self.recipient;
        let body = body.to_vec();
        match tokio::task::spawn_blocking(move || {
            let _permit = permit;
            handle_source_query_blocking(&queue, &relay, recipient, &body, now, received)
        })
        .await
        {
            Ok(reply) => reply,
            Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
        }
    }
}

/// Stable node-peer paths used by the recipient carrier and source runtime.
/// This builder is intentionally not merged into any listener here; startup
/// owns the explicit default-off composition decision.
pub(crate) const REVERSE_ONION_CLAIM_PATH: &str = "/api/chat/peer/reverse-onion/claim";
pub(crate) const REVERSE_ONION_RESULT_PATH: &str = "/api/chat/peer/reverse-onion/result";
pub(crate) const REVERSE_ONION_SOURCE_QUERY_PATH: &str =
    "/api/chat/peer/reverse-onion/source-query";
// [REVERSE-ONION-CLAIM-REJECTION 2026-10-05 by Codex] Queue-level Rejected
// means the exact Claim was not committed; keep it distinct from malformed
// HTTP frames so the recipient can safely retire only that durable poll.
pub(crate) const REVERSE_ONION_CLAIM_REJECTED_STATUS: StatusCode =
    StatusCode::UNPROCESSABLE_ENTITY;
const REVERSE_ONION_BINARY_CONTENT_TYPE: &str = "application/octet-stream";

/// Builds the opt-in adjacent-hop queue router. The returned router is
/// unmounted by default so disabled nodes allocate no public handler state.
/// Claim uses its exact cap; Result uses the frame cap; SourceQuery has an exact 230-byte
/// request cap and never inherits the larger evidence response bound.
pub(crate) fn build_reverse_onion_router(api: Arc<ReverseOnionApi>) -> Router {
    let claim_route = Router::new()
        .route(REVERSE_ONION_CLAIM_PATH, post(handle_claim_http))
        // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Last-added
        // limit runs first, before middleware invokes the Bytes extractor.
        .route_layer(middleware::from_fn_with_state(Arc::clone(&api), reverse_onion_body_admission))
        .layer(DefaultBodyLimit::max(MAX_REVERSE_ONION_CLAIM_BYTES));
    let result_route = Router::new()
        .route(REVERSE_ONION_RESULT_PATH, post(handle_result_http))
        .route_layer(middleware::from_fn_with_state(Arc::clone(&api), reverse_onion_body_admission))
        .layer(DefaultBodyLimit::max(MAX_REVERSE_ONION_FRAME_BYTES));
    let source_routes = Router::new()
        .route(REVERSE_ONION_SOURCE_QUERY_PATH, post(handle_source_query_http))
        .route_layer(middleware::from_fn_with_state(Arc::clone(&api), reverse_onion_body_admission))
        .layer(DefaultBodyLimit::max(REVERSE_ONION_SOURCE_QUERY_BYTES));
    claim_route
        .merge(result_route)
        .merge(source_routes)
        .with_state(api)
}

// [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Bound memory and slow
// senders before any buffering/signature work. This gate shares the lifecycle
// semaphore and stop signal with ciphertext ingress; body failure has no DB
// effect, while admitted SQL retains its Arc permit if HTTP is cancelled.
async fn reverse_onion_body_admission(
    State(api): State<Arc<ReverseOnionApi>>, mut request: Request, next: Next,
) -> Response {
    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Preserve the
    // existing method rejection without inspecting or buffering its body.
    if request.method() != axum::http::Method::POST {
        return next.run(request).await;
    }
    if !has_binary_content_type(request.headers()) {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    let permit = match api.try_permit() {
        Ok(permit) => ReverseOnionApiPermit(Arc::new(permit)),
        Err(_) => return StatusCode::TOO_MANY_REQUESTS.into_response(),
    };
    // [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Keep a clone of
    // the SAME permit for handoff after the cancellation-surviving SQL owner.
    // Evidence has its own larger cap; neither path renews task authority.
    let response_bound = if request.uri().path() == REVERSE_ONION_SOURCE_QUERY_PATH {
        MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES
    } else { MAX_REVERSE_ONION_FRAME_BYTES };
    request.extensions_mut().insert(permit.clone());
    match super::buffer_reverse_onion_request(request).await {
        Ok(request) => {
            let response = next.run(request).await;
            retain_queue_http_response(response, &api.responses, permit, response_bound)
        },
        Err(response) => response,
    }
}

// [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Empty errors release
// immediately. Canonical successful binary bytes are never rewritten; timeout
// or disconnect leaves the durable Claim/Result available for exact retry.
fn retain_queue_http_response(response: Response,
    registry: &super::ReverseOnionResponseRegistry,
    permit: ReverseOnionApiPermit, max_bytes: usize,
) -> Response {
    if response.status() != StatusCode::OK || !has_binary_content_type(response.headers()) {
        return response;
    }
    let (parts, body) = response.into_parts();
    match registry.bound_body(body, permit, max_bytes) {
        Ok(body) => Response::from_parts(parts, body),
        Err(_) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

async fn handle_claim_http(
    State(api): State<Arc<ReverseOnionApi>>,
    Extension(permit): Extension<ReverseOnionApiPermit>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !has_binary_content_type(&headers) {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    into_http_reply(api.handle_claim_admitted(body, trusted_receive_time(), permit.0).await)
}

async fn handle_result_http(
    State(api): State<Arc<ReverseOnionApi>>,
    Extension(permit): Extension<ReverseOnionApiPermit>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !has_binary_content_type(&headers) {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    into_http_reply(api.handle_result_admitted(body, trusted_receive_time(), permit.0).await)
}

async fn handle_source_query_http(
    State(api): State<Arc<ReverseOnionApi>>,
    Extension(permit): Extension<ReverseOnionApiPermit>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !has_binary_content_type(&headers) {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    into_http_reply(api.handle_source_query_admitted(body, trusted_receive_time(), permit.0).await)
}

fn has_binary_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        == Some(REVERSE_ONION_BINARY_CONTENT_TYPE)
}

fn into_http_reply(reply: ReverseOnionApiReply) -> Response {
    if reply.body.is_empty() {
        return reply.status.into_response();
    }
    (
        reply.status,
        [(header::CONTENT_TYPE, REVERSE_ONION_BINARY_CONTENT_TYPE)],
        reply.body,
    )
        .into_response()
}

fn trusted_receive_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn handle_source_query_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &IdentityKeyPair,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
    received: std::time::Instant,
) -> ReverseOnionApiReply {
    // [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] One receive anchor
    // spans worker, DB and signing waits; restarting it loses fractional time.
    let Some(checked_at) = worker_time(now, received.elapsed()) else {
        return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE);
    };
    handle_source_query_blocking_at(queue, relay, recipient, body, checked_at, || {
        worker_time(now, received.elapsed()).ok_or(ReverseOnionQueueError::ClockUnavailable)
    })
}

// [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] Keep the receive-time
// preflight, then refresh after DB waits and again before evidence publication.
fn handle_source_query_blocking_at(
    queue: &ReverseOnionQueueDb,
    relay: &IdentityKeyPair,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
    mut refresh_now: impl FnMut() -> Result<u64, ReverseOnionQueueError>,
) -> ReverseOnionApiReply {
    let query = match ReverseOnionSourceQueryV1::decode(body) {
        Ok(query) => query,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if body.len() != REVERSE_ONION_SOURCE_QUERY_BYTES
        || query.relay() != relay.public_key_bytes()
        || query.route_id() == [0; 16]
        || query.original_request_commitment() == [0; 32]
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    if query
        .verify_binding(
            query.source(),
            relay.public_key_bytes(),
            query.route_id(),
            query.original_request_commitment(),
            now,
        )
        .is_err()
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let snapshot = match queue.lookup_source_at(
        query.source(),
        query.route_id(),
        query.original_request_commitment(),
        || {
            let refreshed = refresh_now()?;
            if refreshed < now {
                return Err(ReverseOnionQueueError::Rejected);
            }
            Ok(refreshed)
        },
    ) {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
        Err(error) => return ReverseOnionApiReply::empty(source_query_db_error_status(error)),
    };
    if snapshot.immediate_recipient() != recipient {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let parts = match (
        snapshot.claim_frame(),
        snapshot.lease_frame(),
        snapshot.result_frame(),
    ) {
        (Some(claim), Some(lease), Some(result)) => {
            let decoded = (
                ReverseOnionFrameV1::decode_for_recovery(claim),
                ReverseOnionFrameV1::decode_for_recovery(lease),
                ReverseOnionFrameV1::decode_for_recovery(result),
            );
            match decoded {
                (Ok(claim), Ok(lease), Ok(result))
                    if source_frames_match_recipient(
                        &claim, &lease, &result, relay.public_key_bytes(),
                        snapshot.immediate_recipient(),
                    ) => Some((claim, lease, result)),
                _ => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
            }
        }
        _ => None,
    };
    let signed_at = match refresh_now() {
        Ok(time) if time >= now && time >= snapshot.observed_at()
            && time < snapshot.available_until() => time,
        Ok(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
    };
    let authority = match query.verify_binding(
        snapshot.source_node_id(),
        relay.public_key_bytes(),
        snapshot.route_id(),
        snapshot.request_commitment(),
        signed_at,
    ) {
        Ok(authority) => authority,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    let evidence_error_status = if parts.is_none() {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let expires_at = query.expires_at().min(snapshot.available_until());
    let evidence = match parts {
        None => ReverseOnionSourceEvidenceV1::pending(
            &authority,
            signed_at,
            expires_at,
            relay,
        ),
        Some((claim, lease, result)) => ReverseOnionSourceEvidenceV1::available(
            &authority, &claim, &lease, &result, snapshot.route_deadline(),
            signed_at, expires_at, relay,
        ),
    };
    let evidence = match evidence {
        Ok(evidence) => evidence,
        Err(_) => return ReverseOnionApiReply::empty(evidence_error_status),
    };
    let encoded = evidence.encode();
    if encoded.len() > MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES {
        return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE);
    }
    let published_at = match refresh_now() {
        Ok(time) if time >= signed_at && time < snapshot.available_until() => time,
        Ok(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
    };
    if evidence.verify_for_query(&query, published_at).is_err() {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    ReverseOnionApiReply::frame(StatusCode::OK, encoded)
}

// [REVERSE-ONION-SOURCE-QUERY-AVAILABILITY 2026-10-05 by Codex] A missing
// source tuple is a request rejection; storage/runtime faults must remain
// retryable and must not masquerade as evidence that the route is invalid.
fn source_query_db_error_status(error: ReverseOnionQueueDbError) -> StatusCode {
    match error {
        ReverseOnionQueueDbError::Rejected
        | ReverseOnionQueueDbError::Conflict
        | ReverseOnionQueueDbError::NoWork => StatusCode::BAD_REQUEST,
        ReverseOnionQueueDbError::Busy => StatusCode::TOO_MANY_REQUESTS,
        ReverseOnionQueueDbError::Capacity
        | ReverseOnionQueueDbError::Corrupt
        | ReverseOnionQueueDbError::MigrationRequired
        | ReverseOnionQueueDbError::Ambiguous
        | ReverseOnionQueueDbError::Unavailable
        | ReverseOnionQueueDbError::ClockUnavailable
        | ReverseOnionQueueDbError::LeaseLost
        | ReverseOnionQueueDbError::AlreadyComplete => StatusCode::SERVICE_UNAVAILABLE,
    }
}

fn source_frames_match_recipient(
    claim: &ReverseOnionFrameV1,
    lease: &ReverseOnionFrameV1,
    result: &ReverseOnionFrameV1,
    relay: [u8; 32],
    recipient: [u8; 32],
) -> bool {
    [claim.relay(), lease.relay(), result.relay()]
        .into_iter()
        .all(|relay_id| relay_id == relay)
        && [
            claim.immediate_recipient(),
            lease.immediate_recipient(),
            result.immediate_recipient(),
        ]
        .into_iter()
        .all(|recipient_id| recipient_id == recipient)
}

#[cfg(test)]
fn handle_claim_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
) -> ReverseOnionApiReply {
    handle_claim_with_policy(queue, relay, recipient, body, now, true)
}

#[cfg(test)]
fn handle_claim_with_policy(
    queue: &ReverseOnionQueueDb,
    relay: &Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
    allow_new_claims: bool,
) -> ReverseOnionApiReply {
    handle_claim_with_live_authority(queue, relay, recipient, body, now, allow_new_claims, None)
}

fn handle_claim_with_live_authority(
    queue: &ReverseOnionQueueDb,
    relay: &Arc<IdentityKeyPair>,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
    allow_new_claims: bool,
    live_authority: Option<(Arc<PrivateBlindRelayAdmission>, Arc<PeerStore>)>,
) -> ReverseOnionApiReply {
    let claim = match ReverseOnionFrameV1::decode_for_recovery(body) {
        Ok(claim) => claim,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if claim.kind() != ReverseOnionKindV1::Claim
        || claim.relay() != relay.public_key_bytes()
        || claim.immediate_recipient() != recipient
        || relay.public_key_bytes() == recipient
        || claim.encode().as_slice() != body
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    // [REVERSE-ONION-AUTHORITY-FENCE 2026-10-05 by Codex] The DB operation
    // owns the authority read guard through the SQLite lease transaction.
    // Do not acquire it here as well: a queued writer between nested reads can
    // deadlock this request while the writer waits for this outer read guard.
    let authority_store = live_authority
        .as_ref()
        .map(|(_, peers)| peers.as_ref());
    let authority_for_claim = live_authority.clone();
    let authority_for_item = live_authority.clone();
    let authority_snapshot = Arc::new(std::sync::OnceLock::new());
    let snapshot_for_claim = Arc::clone(&authority_snapshot);
    let snapshot_for_item = Arc::clone(&authority_snapshot);
    let operation_now = Arc::new(std::sync::atomic::AtomicU64::new(now));
    let now_for_claim = Arc::clone(&operation_now);
    let now_for_item = Arc::clone(&operation_now);
    let now_for_lease = Arc::clone(&operation_now);
    let issue_started_at = std::time::Instant::now();
    let claim_id = claim.claim_id();
    let claim_commitment = claim.commitment();
    let claim_frame = body.to_vec();
    let relay_for_verify = Arc::clone(relay);
    let relay_for_builder = Arc::clone(relay);
    let issue = queue.issue_lease(
        recipient,
        claim_id,
        claim_commitment,
        claim_frame,
        now,
        Arc::clone(&operation_now),
        move || worker_time(now, issue_started_at.elapsed()),
        authority_store,
        move |frame| {
            if !allow_new_claims {
                return Err(ReverseOnionQueueError::Rejected);
            }
            let claim_now = now_for_claim.load(std::sync::atomic::Ordering::Acquire);
            let claim = ReverseOnionFrameV1::decode(frame, claim_now)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            claim
                .verify_claim(relay_for_verify.public_key_bytes(), recipient, claim_now)
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
            if let Some((admission, peers)) = &authority_for_claim {
                // [PHALA-QUEUE-AUTHORITY-HORIZON 2026-10-07 by Codex] The
                // snapshot carries only current R/P/grant expiry; row-specific
                // freshness and route caps were committed when it was queued.
                let snapshot = admission
                    .current_queue_authority_snapshot(peers, claim_now)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                snapshot_for_claim
                    .set(Some(snapshot))
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
            } else if !(cfg!(test) && allow_new_claims) {
                return Err(ReverseOnionQueueError::Rejected);
            }
            Ok(claim.commitment())
        },
        move |stored| {
            let claim_now = now_for_item.load(std::sync::atomic::Ordering::Acquire);
            if let Some((admission, peers)) = &authority_for_item {
                let Some(snapshot) = snapshot_for_item.get().copied().flatten() else {
                    return Err(ReverseOnionQueueError::Rejected);
                };
                admission
                    .validate_pending_lease_authority(stored, snapshot, claim_now)
                    .map_err(|_| ReverseOnionQueueError::Rejected)
            } else if cfg!(test) && allow_new_claims {
                // Unit-level state-machine callers use this helper directly;
                // production construction cannot enable claims without the
                // live admission capability above.
                Ok(())
            } else {
                Err(ReverseOnionQueueError::Rejected)
            }
        },
        {
            let relay = relay_for_builder;
            move |stored, claim_bytes| {
                let claim_now = now_for_lease.load(std::sync::atomic::Ordering::Acquire);
                let claim = ReverseOnionFrameV1::decode(claim_bytes, claim_now)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let envelope = decode_blind_relay_envelope(stored.envelope())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let mut lease_id = [0u8; 16];
                OsRng.fill_bytes(&mut lease_id);
                // [PHALA-CLAIM-EXECUTION-CAP 2026-10-08 by Codex] A longer
                // signed route must not produce a Lease rejected by our own
                // configured execution limit. Existing exact replay is unchanged.
                let execution_deadline = queue.maximum_execution_deadline(
                    stored.route_deadline(), claim_now,
                )?;
                let lease = ReverseOnionFrameV1::lease(
                    &claim,
                    &envelope,
                    lease_id,
                    execution_deadline,
                    claim_now,
                    &relay,
                )
                .map_err(|_| ReverseOnionQueueError::Rejected)?;
                ReverseOnionQueueLeaseMaterial::new(
                    lease_id,
                    lease.commitment(),
                    lease.encode(),
                    lease.expires_at(),
                    lease
                        .replay_evidence_deadline()
                        .map_err(|_| ReverseOnionQueueError::Rejected)?,
                )
            }
        },
    );
    match issue {
        Ok(ReverseOnionQueueIssue::Issued(lease))
        | Ok(ReverseOnionQueueIssue::Existing(lease)) => {
            ReverseOnionApiReply::frame(StatusCode::OK, lease.frame().to_vec())
        }
        Ok(ReverseOnionQueueIssue::Result(result)) => {
            ReverseOnionApiReply::frame(StatusCode::OK, result.frame().to_vec())
        }
        // [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] HTTP status/TLS
        // alone is not relay identity proof. Return an exact Claim-bound signature.
        Ok(ReverseOnionQueueIssue::NoWork) => {
            let no_work_at = operation_now.load(std::sync::atomic::Ordering::Acquire);
            match ReverseOnionNoWorkReceiptV1::issue_no_work(&claim, no_work_at, relay) {
                Ok(receipt) => ReverseOnionApiReply::frame(StatusCode::OK, receipt.encode()),
                Err(_) => ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
            }
        }
        Ok(ReverseOnionQueueIssue::Ambiguous) => {
            ReverseOnionApiReply::empty(StatusCode::CONFLICT)
        }
        // [PHALA-CLAIM-REJECTION-WIRING 2026-10-08 by Codex] Only this
        // transaction's rolled-back rejection proves no Lease was issued.
        // Storage/fence ambiguity must never acquire that no-effect status.
        Err(ReverseOnionQueueDbError::Rejected) => map_queue_error(ReverseOnionQueueError::Rejected),
        // [REVERSE-ONION-DB-ERROR-MAPPING 2026-10-06 by Codex] Preserve
        // retryable storage/clock failures rather than treating them as an
        // authenticated Claim rejection.
        Err(error) => map_queue_db_error(error),
    }
}

fn handle_result_blocking(
    queue: &ReverseOnionQueueDb,
    relay: &IdentityKeyPair,
    recipient: [u8; 32],
    body: &[u8],
    now: u64,
    received: std::time::Instant,
) -> ReverseOnionApiReply {
    // [REVERSE-ONION-RESULT-CLOCK 2026-10-05 by Codex] Queue callbacks
    // refresh trusted time only after the DB operation and connection locks.
    let result = match ReverseOnionFrameV1::decode_for_recovery(body) {
        Ok(result) => result,
        Err(_) => return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST),
    };
    if result.kind() != ReverseOnionKindV1::Result
        || result.relay() != relay.public_key_bytes()
        || result.immediate_recipient() != recipient
        || relay.public_key_bytes() == recipient
        || result.claim_id() == [0; 16]
        || result.lease_id() == [0; 16]
        || result.route_id() == [0; 16]
        || result.encode().as_slice() != body
    {
        return ReverseOnionApiReply::empty(StatusCode::BAD_REQUEST);
    }
    let context = match queue.lookup_result_context_at(
        recipient,
        result.claim_id(),
        result.lease_id(),
        result.route_id(),
        || {
            worker_time(now, received.elapsed()).ok_or(ReverseOnionQueueError::ClockUnavailable)
        },
    ) {
        Ok(context) => context,
        Err(error) => return map_queue_db_error(error),
    };
    match context {
        ReverseOnionQueueResultContext::Armed(lease) => {
            let candidate = body.to_vec();
            let completion = queue.complete_at(&lease, body, || {
                worker_time(now, received.elapsed()).ok_or(ReverseOnionQueueError::ClockUnavailable)
            }, |stored, result_bytes, verified_at| {
                if result_bytes != candidate.as_slice() {
                    return Err(ReverseOnionQueueError::Conflict);
                }
                let claim = ReverseOnionFrameV1::decode_for_recovery(stored.claim_frame())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let persisted_lease = ReverseOnionFrameV1::decode_for_recovery(stored.lease_frame())
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                let result = ReverseOnionFrameV1::decode_for_recovery(result_bytes)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                result
                    .verify_result(&claim, &persisted_lease, stored.route_deadline(), verified_at)
                    .map_err(|_| ReverseOnionQueueError::Rejected)?;
                Ok(result.commitment())
            });
            match completion {
                Ok(ReverseOnionQueueCompletion::Stored)
                | Ok(ReverseOnionQueueCompletion::AlreadyComplete) => {
                    ReverseOnionApiReply::frame(StatusCode::OK, body.to_vec())
                }
                Err(error) => map_queue_db_error(error),
            }
        }
        ReverseOnionQueueResultContext::Result(stored) => {
            let stored_frame = match ReverseOnionFrameV1::decode_for_recovery(stored.frame()) {
                Ok(frame) => frame,
                Err(_) => return ReverseOnionApiReply::empty(StatusCode::SERVICE_UNAVAILABLE),
            };
            if stored.frame() != body
                || stored_frame.kind() != ReverseOnionKindV1::Result
                || stored_frame.relay() != relay.public_key_bytes()
                || stored_frame.immediate_recipient() != recipient
                || stored_frame.claim_id() != result.claim_id()
                || stored_frame.lease_id() != result.lease_id()
                || stored_frame.route_id() != result.route_id()
                || stored_frame.commitment() != stored.commitment()
            {
                return ReverseOnionApiReply::empty(StatusCode::CONFLICT);
            }
            ReverseOnionApiReply::frame(StatusCode::OK, stored.frame().to_vec())
        }
        ReverseOnionQueueResultContext::NoWork => {
            ReverseOnionApiReply::empty(StatusCode::NO_CONTENT)
        }
        ReverseOnionQueueResultContext::Ambiguous => {
            ReverseOnionApiReply::empty(StatusCode::CONFLICT)
        }
    }
}

fn map_queue_error(error: ReverseOnionQueueError) -> ReverseOnionApiReply {
    match error {
        // [REVERSE-ONION-CLAIM-REJECTION 2026-10-05 by Codex] Only the Claim
        // transaction's explicit no-effect rejection gets its own status.
        ReverseOnionQueueError::Rejected => {
            ReverseOnionApiReply::empty(REVERSE_ONION_CLAIM_REJECTED_STATUS)
        }
        other => map_queue_db_error(other.into()),
    }
}

fn map_queue_db_error(error: ReverseOnionQueueDbError) -> ReverseOnionApiReply {
    // [REVERSE-ONION-RESULT-CLOCK 2026-10-05 by Codex] Trusted-clock failure
    // is retryable service unavailability, never a protocol rejection.
    let status = match error {
        ReverseOnionQueueDbError::NoWork => StatusCode::NO_CONTENT,
        ReverseOnionQueueDbError::Conflict | ReverseOnionQueueDbError::Ambiguous => {
            StatusCode::CONFLICT
        }
        ReverseOnionQueueDbError::Busy => StatusCode::TOO_MANY_REQUESTS,
        ReverseOnionQueueDbError::Rejected => StatusCode::BAD_REQUEST,
        ReverseOnionQueueDbError::Capacity
        | ReverseOnionQueueDbError::Corrupt
        | ReverseOnionQueueDbError::MigrationRequired
        | ReverseOnionQueueDbError::Unavailable
        | ReverseOnionQueueDbError::ClockUnavailable
        | ReverseOnionQueueDbError::LeaseLost
        | ReverseOnionQueueDbError::AlreadyComplete => StatusCode::SERVICE_UNAVAILABLE,
    };
    ReverseOnionApiReply::empty(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] Exact NoWork size.
    use aeronyx_core::protocol::onion::reverse_delivery::REVERSE_ONION_NO_WORK_RECEIPT_BYTES;

    // [PHALA-REVERSE-FIXTURE-REPAIR 2026-10-08 by Codex] Lease fixtures
    // carry an actual sealed onion layer, not an invalid arbitrary blob.
    fn test_envelope(
        relay: &IdentityKeyPair, recipient: &IdentityKeyPair,
        route_id: [u8; 16], now: u64, payload: &[u8],
    ) -> BlindRelayEnvelope {
        use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};
        let (_, kem) = recipient.to_x25519();
        build_onion_envelope(
            &[OnionHop { node_id: recipient.public_key_bytes(), kem_pub: kem.to_bytes() }],
            payload, route_id, 1, now, relay,
        ).unwrap()
    }

    // [REVERSE-ONION-SOURCE-QUERY-AVAILABILITY 2026-10-05 by Codex]
    // Authored, not run: storage failure differs from an unknown route tuple.
    #[test]
    fn source_query_storage_errors_are_retryable_http_failures() {
        assert_eq!(
            source_query_db_error_status(ReverseOnionQueueDbError::Rejected),
            StatusCode::BAD_REQUEST,
        );
        assert_eq!(
            source_query_db_error_status(ReverseOnionQueueDbError::Busy),
            StatusCode::TOO_MANY_REQUESTS,
        );
        for error in [
            ReverseOnionQueueDbError::Corrupt,
            ReverseOnionQueueDbError::Unavailable,
            ReverseOnionQueueDbError::ClockUnavailable,
            ReverseOnionQueueDbError::Ambiguous,
        ] {
            assert_eq!(source_query_db_error_status(error), StatusCode::SERVICE_UNAVAILABLE);
        }
    }

    // [REVERSE-RECOVERY-BOOT 2026-10-05 by Codex] Authored, not executed:
    // an expired Claim can recover its exact lease, but a fresh Claim cannot
    // create execution authority in recovery-only mode.
    #[cfg(unix)]
    #[test]
    fn recovery_only_replays_exact_lease_without_issuing_another() {
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let envelope = test_envelope(&relay, &recipient, [31; 16], NOW, b"opaque request");
        let item = ReverseOnionQueueItem::new(
            [13; 32], envelope.route_id, [14; 32], source.public_key_bytes(),
            [15; 32], api.recipient, [16; 32],
            encode_blind_relay_envelope(&envelope).unwrap(), NOW + 60,
        ).unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [17; 16], NOW, NOW + 30, &recipient,
        ).unwrap().encode();
        let issued = handle_claim_with_policy(&queue, &relay, api.recipient, &claim, NOW, true);
        assert_eq!(issued.status(), StatusCode::OK);
        let recovered = handle_claim_with_policy(&queue, &relay, api.recipient, &claim, NOW + 31, false);
        assert_eq!(recovered, issued);
        let new_claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [18; 16], NOW + 31, NOW + 61, &recipient,
        ).unwrap().encode();
        let refused = handle_claim_with_policy(&queue, &relay, api.recipient, &new_claim, NOW + 31, false);
        assert_ne!(refused.status(), StatusCode::OK);
        assert_ne!(refused.status(), StatusCode::NO_CONTENT);
        assert_eq!(handle_claim_with_policy(&queue, &relay, api.recipient, &claim, NOW + 31, false), issued);
    }
    // [PHALA-ROTATED-CUSTODY-RECOVERY 2026-10-08 by Codex] Authored,
    // not run: the actual API must replay/complete durable custody independently
    // of fresh grant availability. The initial row is synthetic test setup.
    #[cfg(unix)]
    #[test]
    fn renewed_descriptors_do_not_replace_armed_claim_or_result() {
        let (_directory, queue, _unused_api, _, _, _) = fixture();
        let f = crate::services::reverse_onion_source::tests::Fixture::new();
        let relay = f.relay_identity();
        let recipient = f.recipient_identity();
        let source = f.source_identity();
        let (old_relay, old_recipient, old_grant) = f.policy_parts();
        let (new_relay, new_recipient, new_grant) = f.renewed_policy_parts(NOW + 31);
        assert!(old_grant.verify_at(&new_relay, &new_recipient,
            aeronyx_core::protocol::onion::OnionRoutePurpose::BlindVaultPull.as_str(),
            NOW + 31).is_err());
        let envelope = test_envelope(&relay, &recipient, [31; 16], NOW, b"opaque request");
        queue.enqueue(&ReverseOnionQueueItem::new(
            [13; 32], envelope.route_id, [14; 32], source.public_key_bytes(),
            [15; 32], recipient.public_key_bytes(), [16; 32],
            encode_blind_relay_envelope(&envelope).unwrap(), NOW + 60,
        ).unwrap(), NOW).unwrap();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [17; 16], NOW, NOW + 30, &recipient,
        ).unwrap();
        let issued = handle_claim_with_policy(
            &queue, &relay, recipient.public_key_bytes(), &claim.encode(), NOW, true,
        );
        assert_eq!(issued.status(), StatusCode::OK);
        let lease = ReverseOnionFrameV1::decode_for_recovery(issued.body()).unwrap();
        lease.verify_recipient_lease(&claim, relay.public_key_bytes(),
            recipient.public_key_bytes(), NOW).unwrap();
        let peers = Arc::new(PeerStore::new());
        peers.pin_private_onion_route_identities(old_relay.node_id(), old_recipient.node_id()).unwrap();
        peers.upsert_verified_from_source(old_relay, NOW, "test_pin").unwrap();
        peers.upsert_verified_from_source(old_recipient, NOW, "test_pin").unwrap();
        peers.remember_issued_private_onion_authorization(old_grant,
            recipient.public_key_bytes(), NOW).unwrap();
        peers.upsert_verified_from_source(new_relay.clone(), NOW + 31, "test_pin").unwrap();
        peers.upsert_verified_from_source(new_recipient.clone(), NOW + 31, "test_pin").unwrap();
        let admission = live_admission_for_fixture(Arc::clone(&queue), &relay, &recipient, &source);
        let live = Some((Arc::clone(&admission), Arc::clone(&peers)));
        assert_eq!(handle_claim_with_live_authority(
            &queue, &relay, recipient.public_key_bytes(), &claim.encode(), NOW + 31, true, live.clone(),
        ), issued);
        let fresh = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [18; 16], NOW + 31, NOW + 61, &recipient,
        ).unwrap();
        assert_eq!(handle_claim_with_live_authority(
            &queue, &relay, recipient.public_key_bytes(), &fresh.encode(), NOW + 31, true, live.clone(),
        ).status(), REVERSE_ONION_CLAIM_REJECTED_STATUS);
        // A valid new grant permits fresh polling, but cannot lease the old
        // Armed row again. The fresh Claim gets only signed NoWork.
        peers.remember_issued_private_onion_authorization(new_grant,
            recipient.public_key_bytes(), NOW + 32).unwrap();
        let empty = handle_claim_with_live_authority(
            &queue, &relay, recipient.public_key_bytes(), &fresh.encode(), NOW + 32, true, live.clone(),
        );
        assert_eq!(empty.status(), StatusCode::OK);
        ReverseOnionNoWorkReceiptV1::decode_for_claim(
            empty.body(), &fresh, relay.public_key_bytes(), recipient.public_key_bytes(), NOW + 32,
        ).unwrap();
        let (reply_request, _) = OnionReplySession::prepare_source_sealed(
            lease.route_id(), recipient.public_key_bytes(), ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
            vec![19; 3],
        ).unwrap();
        let sealed = seal_onion_reply(lease.route_id(), &reply_request, b"opaque result", &recipient).unwrap();
        let result = ReverseOnionFrameV1::result(&claim, &lease,
            &encode_onion_sealed_response(&sealed).unwrap(), NOW + 60, NOW + 33, &recipient).unwrap();
        let exact_result = result.encode();
        let stored = handle_result_blocking(&queue, &relay, recipient.public_key_bytes(),
            &exact_result, NOW + 33, std::time::Instant::now());
        assert_eq!(stored.status(), StatusCode::OK);
        assert_eq!(stored.body(), exact_result.as_slice());
        assert_eq!(handle_claim_with_live_authority(
            &queue, &relay, recipient.public_key_bytes(), &claim.encode(), NOW + 33, true, live,
        ).body(), exact_result.as_slice());
        assert_eq!(handle_result_blocking(&queue, &relay, recipient.public_key_bytes(),
            &exact_result, NOW + 33, std::time::Instant::now()), stored);
    }

    // [REVERSE-QUEUE-DRAIN 2026-10-05 by Codex] Authored, not executed.
    #[test]
    fn worker_time_accounts_for_wait_and_rejects_overflow() {
        assert_eq!(worker_time(100, std::time::Duration::from_secs(31)), Some(131));
        assert_eq!(worker_time(u64::MAX, std::time::Duration::from_secs(1)), None);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_rejects_intake_and_drain_waits_for_owned_work() {
        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        let permit = api.try_permit().unwrap();
        api.request_stop();
        assert!(api.try_permit().is_err());
        let drain = api.shutdown_and_drain();
        tokio::pin!(drain);
        tokio::select! {
            biased;
            _ = &mut drain => panic!("drained while accepted work still owns its permit"),
            _ = std::future::ready(()) => {}
        }
        drop(permit);
        drain.await;
        assert!(api.try_permit().is_err());
    }

    // [PHALA-QUEUE-LIFECYCLE-OWNER 2026-10-07 by Codex] Authored, not run.
    #[cfg(unix)]
    fn live_admission_for_fixture(
        queue: Arc<ReverseOnionQueueDb>,
        relay: &IdentityKeyPair,
        recipient: &IdentityKeyPair,
        source: &IdentityKeyPair,
    ) -> Arc<PrivateBlindRelayAdmission> {
        Arc::new(PrivateBlindRelayAdmission::new_pinned_identity_only(
            relay.public_key_bytes(), recipient.public_key_bytes(),
            aeronyx_core::protocol::onion::OnionRoutePurpose::BlindVaultPull,
            vec![source.public_key_bytes()], queue, 1, 60,
        ).unwrap())
    }

    #[cfg(unix)]
    #[test]
    fn live_api_cannot_adopt_another_queue_or_reopen_a_stopped_owner() {
        // [PHALA-QUEUE-LIFECYCLE-OWNER 2026-10-07 by Codex] Negative
        // controls have equal identities and limits but distinct DB owners.
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let (_other_directory, other_queue, _other_api, _, _, _) = fixture();
        let mismatched = live_admission_for_fixture(other_queue, &relay, &recipient, &source);
        assert!(api.with_live_private_admission(mismatched, Arc::new(PeerStore::new())).is_err());

        let make_api = || ReverseOnionApi::new(
            Arc::clone(&queue), Arc::clone(&relay), recipient.public_key_bytes(), 1,
        ).unwrap();
        let admission = live_admission_for_fixture(Arc::clone(&queue), &relay, &recipient, &source);
        let stopped_api = make_api();
        stopped_api.request_stop();
        assert!(stopped_api.with_live_private_admission(
            Arc::clone(&admission), Arc::new(PeerStore::new()),
        ).is_err());

        let already_bound = make_api().with_live_private_admission(
            Arc::clone(&admission), Arc::new(PeerStore::new()),
        ).unwrap();
        let replacement = live_admission_for_fixture(Arc::clone(&queue), &relay, &recipient, &source);
        assert!(already_bound.with_live_private_admission(
            replacement, Arc::new(PeerStore::new()),
        ).is_err());

        let api = make_api().with_live_private_admission(
            Arc::clone(&admission), Arc::new(PeerStore::new()),
        ).unwrap();
        assert!(Arc::ptr_eq(&api.stopped, &admission.queue_stop_signal()));
        api.request_stop();
        assert!(api.try_permit().is_err());
        assert!(admission.try_queue_permit().is_err());
        assert!(make_api().with_live_private_admission(
            admission, Arc::new(PeerStore::new()),
        ).is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn live_api_stop_and_drain_cover_ciphertext_ingress_permits() {
        // [PHALA-QUEUE-LIFECYCLE-OWNER 2026-10-07 by Codex] A permit from
        // the other intake path must remain visible to this API's drain.
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let admission = live_admission_for_fixture(queue, &relay, &recipient, &source);
        let api = api.with_live_private_admission(
            Arc::clone(&admission), Arc::new(PeerStore::new()),
        ).unwrap();
        let ingress_work = admission.try_queue_permit().unwrap();
        admission.request_stop();
        assert!(api.stopped.load(std::sync::atomic::Ordering::Acquire));
        assert!(api.try_permit().is_err());
        let drain = api.shutdown_and_drain();
        tokio::pin!(drain);
        tokio::select! {
            biased;
            _ = &mut drain => panic!("API drain ignored ciphertext ingress work"),
            _ = std::future::ready(()) => {}
        }
        drop(ingress_work);
        drain.await;
        assert!(api.try_permit().is_err());
        assert!(admission.try_queue_permit().is_err());
    }
    use aeronyx_core::protocol::chat::{
        encode_blind_relay_envelope, BlindRelayEnvelope,
    };
    use aeronyx_core::protocol::onion::reverse_delivery::{
        SourceEvidencePartV1, SourceEvidenceStateV1,
    };
    use aeronyx_core::protocol::onion_reply::{
        encode_onion_sealed_response, seal_onion_reply, OnionReplySession,
        ONION_REPLY_RESPONSE_SIZE_CLASSES,
    };
    use crate::services::reverse_onion_queue::{
        ReverseOnionQueueItem, ReverseOnionQueueLeaseMaterial, ReverseOnionQueueLimits,
        ReverseOnionQueueIssue,
    };
    use crate::services::reverse_onion_queue_db::{ReverseOnionQueueDbConfig};
    use tempfile::TempDir;

    const NOW: u64 = 1_800_000_000;

    #[cfg(unix)]
    fn fixture() -> (
        TempDir,
        Arc<ReverseOnionQueueDb>,
        ReverseOnionApi,
        Arc<IdentityKeyPair>,
        Arc<IdentityKeyPair>,
        Arc<IdentityKeyPair>,
    ) {
        fixture_at(NOW)
    }

    // [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Real HTTP adapters
    // sample wall time; don't seed their queue clock with a future test epoch.
    #[cfg(unix)]
    fn fixture_at(now: u64) -> (
        TempDir, Arc<ReverseOnionQueueDb>, ReverseOnionApi,
        Arc<IdentityKeyPair>, Arc<IdentityKeyPair>, Arc<IdentityKeyPair>,
    ) {
        fixture_with_lease_cap(now, 60, 60)
    }

    // [PHALA-CLAIM-EXECUTION-CAP 2026-10-08 by Codex] Exercise the real
    // API with distinct route and execution caps, as production config allows.
    #[cfg(unix)]
    fn fixture_with_lease_cap(now: u64, lease_secs: u64, route_secs: u64) -> (
        TempDir, Arc<ReverseOnionQueueDb>, ReverseOnionApi,
        Arc<IdentityKeyPair>, Arc<IdentityKeyPair>, Arc<IdentityKeyPair>,
    ) {
        let directory = tempfile::Builder::new()
            .prefix("r6-reverse-onion-api-")
            // [PHALA-JOURNAL-FIXTURE-REPAIR 2026-10-08 by Codex]
            .permissions(<std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700))
            .tempdir_in("/Volumes/disk/aeronyx-codex-tmp")
            .expect("fixture directory");
        let limits = ReverseOnionQueueLimits::new(4, 8 * 1024 * 1024, 4, lease_secs, 120)
            .and_then(|limits| limits.with_route_max_secs(route_secs)).expect("queue limits");
        let config = ReverseOnionQueueDbConfig::new(
            directory.path().join("queue.sqlite"),
            // [PHALA-QUEUE-FIXTURE-REPAIR 2026-10-08 by Codex] Include
            // rollback-journal and metadata headroom above twice logical bytes.
            32 * 1024 * 1024,
            limits,
        )
        .expect("queue config");
        let queue = Arc::new(ReverseOnionQueueDb::open(config, now).expect("queue open"));
        let relay = Arc::new(IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap());
        let recipient = Arc::new(IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap());
        let source = Arc::new(IdentityKeyPair::from_bytes(&[0x76; 32]).unwrap());
        let api = ReverseOnionApi::new(
            Arc::clone(&queue),
            Arc::clone(&relay),
            recipient.public_key_bytes(),
            1,
        )
        .expect("api");
        (directory, queue, api, relay, recipient, source)
    }

    // [PHALA-CLAIM-EXECUTION-CAP 2026-10-08 by Codex] The HTTP producer
    // must sign the configured shorter Lease and persist its exact bytes;
    // Claim expiry must not truncate recovery or allow a second execution.
    #[cfg(unix)]
    #[test]
    fn reverse_onion_claim_uses_execution_cap_without_truncating_exact_replay() {
        let (_directory, queue, api, relay, recipient, source) = fixture_with_lease_cap(NOW, 60, 300);
        let envelope = test_envelope(&relay, &recipient, [41; 16], NOW, b"opaque request");
        let item = ReverseOnionQueueItem::new(
            [43; 32], envelope.route_id, [44; 32], source.public_key_bytes(),
            [45; 32], api.recipient, [46; 32],
            encode_blind_relay_envelope(&envelope).unwrap(), NOW + 300,
        ).unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [47; 16], NOW, NOW + 30, &recipient,
        ).unwrap();
        let issued = handle_claim_with_policy(&queue, &relay, api.recipient, &claim.encode(), NOW, true);
        assert_eq!(issued.status(), StatusCode::OK);
        let lease = ReverseOnionFrameV1::decode(&issued.body, NOW).unwrap();
        assert_eq!(lease.expires_at(), NOW + 60);
        lease.verify_lease(&claim, NOW + 300, NOW + 31).unwrap();
        assert_eq!(handle_claim_with_policy(&queue, &relay, api.recipient,
            &claim.encode(), NOW + 31, false), issued);
        assert_eq!(handle_claim_with_policy(&queue, &relay, api.recipient,
            &claim.encode(), NOW + 60, false).status(), StatusCode::CONFLICT);
        assert_eq!(queue.maximum_execution_deadline(NOW + 20, NOW).unwrap(), NOW + 20);
        assert!(queue.maximum_execution_deadline(NOW, NOW).is_err());
        assert!(queue.maximum_execution_deadline(u64::MAX, u64::MAX - 1).is_err());
        assert!(queue.maximum_execution_deadline(NOW + 300, 0).is_err());
    }

    // [REVERSE-ONION-SIGNED-NO-WORK 2026-10-06 by Codex] Authored, not run.
    #[cfg(unix)]
    #[test]
    fn empty_queue_returns_relay_signed_claim_bound_receipt() {
        let (_directory, queue, api, relay, recipient, _) = fixture();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(), [0x77; 16], NOW, NOW + 30, &recipient,
        ).unwrap();
        let reply = handle_claim_blocking(
            &queue, &relay, api.recipient, &claim.encode(), NOW,
        );
        assert_eq!(reply.status(), StatusCode::OK);
        assert_eq!(reply.body().len(), REVERSE_ONION_NO_WORK_RECEIPT_BYTES);
        assert!(ReverseOnionNoWorkReceiptV1::decode_for_claim(
            reply.body(), &claim, relay.public_key_bytes(), recipient.public_key_bytes(), NOW,
        ).is_ok());
    }

    #[cfg(unix)]
    fn query_bytes(
        source: &IdentityKeyPair,
        relay: &IdentityKeyPair,
        route: [u8; 16],
        original_request: [u8; 32],
        issued_at: u64,
        expires_at: u64,
    ) -> Vec<u8> {
        ReverseOnionSourceQueryV1::sign(
            source,
            relay.public_key_bytes(),
            route,
            original_request,
            SourceEvidencePartV1::Claim,
            [0x76; 32],
            issued_at,
            expires_at,
        )
        .unwrap()
        .encode()
    }

    #[test]
    fn constructor_rejects_self_recipient_and_zero_concurrency() {
        let relay = Arc::new(IdentityKeyPair::from_bytes(&[7; 32]).unwrap());
        assert_eq!(
            validate_recipient(relay.public_key_bytes(), relay.public_key_bytes(), 1),
            Err(ReverseOnionApiConfigError::Invalid)
        );
        assert_eq!(
            validate_recipient(relay.public_key_bytes(), [0; 32], 1),
            Err(ReverseOnionApiConfigError::Invalid)
        );
        assert_eq!(
            validate_recipient(relay.public_key_bytes(), [8; 32], 0),
            Err(ReverseOnionApiConfigError::Invalid)
        );
    }

    #[test]
    fn replies_are_empty_for_coarse_errors_and_exact_for_frames() {
        assert!(ReverseOnionApiReply::empty(StatusCode::NO_CONTENT)
            .body()
            .is_empty());
        assert_eq!(
            ReverseOnionApiReply::frame(StatusCode::OK, vec![1, 2, 3]).body(),
            &[1, 2, 3]
        );
        assert_eq!(
            map_queue_error(ReverseOnionQueueError::Conflict).status(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            map_queue_error(ReverseOnionQueueError::Rejected).status(),
            REVERSE_ONION_CLAIM_REJECTED_STATUS
        );
        assert_eq!(
            map_queue_db_error(ReverseOnionQueueDbError::Rejected).status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_rejects_bounds_wrong_relay_and_stale_query() {
        let (_directory, _queue, api, relay, _recipient, source) = fixture();
        assert_eq!(
            api.handle_source_query(Bytes::from(vec![0; REVERSE_ONION_SOURCE_QUERY_BYTES - 1]), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let canonical = query_bytes(&source, &relay, [1; 16], [2; 32], NOW, NOW + 10);
        let mut trailing = canonical.clone();
        trailing.push(0);
        assert_eq!(
            api.handle_source_query(Bytes::from(trailing), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let wrong_relay = IdentityKeyPair::from_bytes(&[0x77; 32]).unwrap();
        let wrong = query_bytes(&source, &wrong_relay, [1; 16], [2; 32], NOW, NOW + 10);
        assert_eq!(
            api.handle_source_query(Bytes::from(wrong), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
        let stale = query_bytes(&source, &relay, [1; 16], [2; 32], NOW - 20, NOW - 1);
        assert_eq!(
            api.handle_source_query(Bytes::from(stale), NOW)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_releases_no_permit_for_preflight_rejection() {
        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        for _ in 0..2 {
            assert_eq!(
                api.handle_source_query(Bytes::from_static(b"short"), NOW)
                    .await
                    .status(),
                StatusCode::BAD_REQUEST
            );
        }
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_pending_is_signed_only_for_partial_durable_snapshot() {
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let item = ReverseOnionQueueItem::new(
            [0x81; 32],
            [1; 16],
            [2; 32],
            source.public_key_bytes(),
            [3; 32],
            api.recipient,
            [4; 32],
            vec![5; 3],
            NOW + 60,
        )
        .unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let query = query_bytes(&source, &relay, [1; 16], [2; 32], NOW, NOW + 20);
        let reply = api
            .handle_source_query(Bytes::from(query.clone()), NOW + 1)
            .await;
        assert_eq!(reply.status(), StatusCode::OK);
        let evidence = ReverseOnionSourceEvidenceV1::decode(reply.body()).unwrap();
        let query = ReverseOnionSourceQueryV1::decode(&query).unwrap();
        assert_eq!(evidence.state(), SourceEvidenceStateV1::Pending);
        evidence.verify_for_query(&query, NOW + 1).unwrap();
        let other_recipient = IdentityKeyPair::from_bytes(&[0x78; 32]).unwrap();
        let other_item = ReverseOnionQueueItem::new(
            [0x89; 32],
            [7; 16],
            [8; 32],
            source.public_key_bytes(),
            [9; 32],
            other_recipient.public_key_bytes(),
            [10; 32],
            vec![11; 3],
            NOW + 60,
        )
        .unwrap();
        queue.enqueue(&other_item, NOW).unwrap();
        let wrong_target = query_bytes(&source, &relay, [7; 16], [8; 32], NOW, NOW + 20);
        assert_eq!(
            api.handle_source_query(Bytes::from(wrong_target), NOW + 1)
                .await
                .status(),
            StatusCode::BAD_REQUEST
        );
    }

    // [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] Authored, not run:
    // deterministic read/sign/publication samples, without sleeps or probes.
    #[cfg(unix)]
    #[test]
    fn source_query_rechecks_pending_expiry_at_signing_and_publication() {
        for (times, expected) in [
            ([NOW + 1, NOW + 2, NOW + 3], StatusCode::OK),
            ([NOW + 1, NOW + 20, NOW + 20], StatusCode::BAD_REQUEST),
            ([NOW + 1, NOW + 2, NOW + 20], StatusCode::BAD_REQUEST),
            ([NOW + 2, NOW + 1, NOW + 3], StatusCode::BAD_REQUEST),
            ([NOW + 1, NOW + 2, NOW + 1], StatusCode::BAD_REQUEST),
        ] {
            let (_directory, queue, api, relay, _recipient, source) = fixture();
            let item = ReverseOnionQueueItem::new(
                [0x81; 32], [1; 16], [2; 32], source.public_key_bytes(),
                [3; 32], api.recipient, [4; 32], vec![5; 3], NOW + 60,
            ).unwrap();
            queue.enqueue(&item, NOW).unwrap();
            let query = query_bytes(&source, &relay, [1; 16], [2; 32], NOW, NOW + 20);
            let mut samples = times.into_iter();
            let reply = handle_source_query_blocking_at(
                &queue, &relay, api.recipient, &query, NOW,
                || Ok(samples.next().expect("bounded clock samples")),
            );
            assert_eq!(reply.status(), expected);
            if expected == StatusCode::OK {
                let evidence = ReverseOnionSourceEvidenceV1::decode(reply.body()).unwrap();
                let query = ReverseOnionSourceQueryV1::decode(&query).unwrap();
                assert_eq!(evidence.state(), SourceEvidenceStateV1::Pending);
                evidence.verify_for_query(&query, NOW + 3).unwrap();
            } else {
                assert!(reply.body().is_empty());
            }
        }
        // A still-fresh query must not renew the queue snapshot's own bound.
        let (_directory, queue, api, relay, _recipient, source) = fixture();
        let item = ReverseOnionQueueItem::new(
            [0x81; 32], [1; 16], [2; 32], source.public_key_bytes(),
            [3; 32], api.recipient, [4; 32], vec![5; 3], NOW + 5,
        ).unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let query = query_bytes(&source, &relay, [1; 16], [2; 32], NOW, NOW + 20);
        let reply = handle_source_query_blocking_at(
            &queue, &relay, api.recipient, &query, NOW, || Ok(NOW + 1),
        );
        assert_eq!(reply.status(), StatusCode::OK);
        let bounded = ReverseOnionSourceEvidenceV1::decode(reply.body()).unwrap();
        let decoded_query = ReverseOnionSourceQueryV1::decode(&query).unwrap();
        bounded.verify_for_query(&decoded_query, NOW + 4).unwrap();
        assert!(bounded.verify_for_query(&decoded_query, NOW + 5).is_err());
        let mut times = [NOW + 1, NOW + 2, NOW + 5].into_iter();
        let reply = handle_source_query_blocking_at(
            &queue, &relay, api.recipient, &query, NOW,
            || Ok(times.next().unwrap()),
        );
        assert_eq!(reply.status(), StatusCode::BAD_REQUEST);
        assert!(reply.body().is_empty());
    }

    #[cfg(unix)]
    #[tokio::test(flavor = "current_thread")]
    async fn source_query_available_requires_complete_verified_chain() {
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let envelope = test_envelope(&relay, &recipient, [11; 16], NOW, b"opaque request");
        let envelope_bytes = encode_blind_relay_envelope(&envelope).unwrap();
        let item = ReverseOnionQueueItem::new(
            [13; 32],
            [11; 16],
            [14; 32],
            source.public_key_bytes(),
            [15; 32],
            api.recipient,
            [16; 32],
            envelope_bytes,
            NOW + 60,
        )
        .unwrap();
        queue.enqueue(&item, NOW).unwrap();
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(),
            [17; 16],
            NOW,
            NOW + 30,
            &recipient,
        )
        .unwrap();
        let lease = ReverseOnionFrameV1::lease(
            &claim,
            &envelope,
            [18; 16],
            NOW + 60,
            NOW,
            &relay,
        )
        .unwrap();
        let claim_commitment = claim.commitment();
        let lease_id = lease.lease_id();
        let lease_commitment = lease.commitment();
        let lease_frame = lease.encode();
        let lease_expiry = lease.expires_at();
        let replay_deadline = lease.replay_evidence_deadline().unwrap();
        let issue = queue
            .issue_lease(
                api.recipient,
                claim.claim_id(),
                claim_commitment,
                claim.encode(),
                NOW,
                Arc::new(std::sync::atomic::AtomicU64::new(NOW)),
                || Some(NOW),
                None,
                move |_| Ok(claim_commitment),
                |_| Ok(()),
                move |_, _| {
                    ReverseOnionQueueLeaseMaterial::new(
                        lease_id,
                        lease_commitment,
                        lease_frame,
                        lease_expiry,
                        replay_deadline,
                    )
                },
            )
            .unwrap();
        let issued = match issue {
            ReverseOnionQueueIssue::Issued(issued) => issued,
            _ => panic!("expected issued lease"),
        };
        let sealed = seal_onion_reply(
            lease.route_id(),
            &OnionReplySession::prepare_source_sealed(
                lease.route_id(),
                api.recipient,
                ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
                vec![19; 3],
            )
            .unwrap()
            .0,
            b"opaque result",
            &recipient,
        )
        .unwrap();
        let result = ReverseOnionFrameV1::result(
            &claim,
            &lease,
            &encode_onion_sealed_response(&sealed).unwrap(),
            NOW + 60,
            NOW + 1,
            &recipient,
        )
        .unwrap();
        queue
            .complete(&issued, &result.encode(), NOW + 1, |_, _| Ok(result.commitment()))
            .unwrap();
        // [REVERSE-ONION-RESULT-CUSTODY-ECHO 2026-10-05 by Codex] The relay
        // ACK is the exact durable Result bytes, including an idempotent retry.
        let exact_result = result.encode();
        let echoed = api.handle_result(Bytes::from(exact_result.clone()), NOW + 2).await;
        assert_eq!(echoed.status(), StatusCode::OK);
        assert_eq!(echoed.body(), exact_result.as_slice());
        let query_bytes = query_bytes(&source, &relay, [11; 16], [14; 32], NOW, NOW + 20);
        let reply = api
            .handle_source_query(Bytes::from(query_bytes.clone()), NOW + 2)
            .await;
        assert_eq!(reply.status(), StatusCode::OK);
        let evidence = ReverseOnionSourceEvidenceV1::decode(reply.body()).unwrap();
        let query = ReverseOnionSourceQueryV1::decode(&query_bytes).unwrap();
        assert_eq!(evidence.state(), SourceEvidenceStateV1::Available);
        evidence.verify_for_query(&query, NOW + 2).unwrap();
        // [PHALA-SOURCE-EVIDENCE-CLOCK 2026-10-07 by Codex] The complete
        // durable chain must not be published after its query expires either.
        let mut times = [NOW + 2, NOW + 3, NOW + 20].into_iter();
        let expired = handle_source_query_blocking_at(
            &queue, &relay, api.recipient, &query_bytes, NOW + 2,
            || Ok(times.next().unwrap()),
        );
        assert_eq!(expired.status(), StatusCode::BAD_REQUEST);
        assert!(expired.body().is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn source_query_rejects_internally_valid_chain_bound_to_wrong_p() {
        let relay = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
        let configured = IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap();
        let wrong = IdentityKeyPair::from_bytes(&[0x78; 32]).unwrap();
        let envelope = test_envelope(&relay, &wrong, [21; 16], NOW, b"opaque wrong recipient");
        let claim = ReverseOnionFrameV1::claim(
            relay.public_key_bytes(),
            [23; 16],
            NOW,
            NOW + 30,
            &wrong,
        )
        .unwrap();
        let lease = ReverseOnionFrameV1::lease(
            &claim,
            &envelope,
            [24; 16],
            NOW + 60,
            NOW,
            &relay,
        )
        .unwrap();
        let sealed = seal_onion_reply(
            lease.route_id(),
            &OnionReplySession::prepare_source_sealed(
                lease.route_id(),
                wrong.public_key_bytes(),
                ONION_REPLY_RESPONSE_SIZE_CLASSES[0],
                vec![25; 3],
            )
            .unwrap()
            .0,
            b"opaque wrong recipient",
            &wrong,
        )
        .unwrap();
        let result = ReverseOnionFrameV1::result(
            &claim,
            &lease,
            &encode_onion_sealed_response(&sealed).unwrap(),
            NOW + 60,
            NOW + 1,
            &wrong,
        )
        .unwrap();
        assert!(!source_frames_match_recipient(
            &claim,
            &lease,
            &result,
            relay.public_key_bytes(),
            configured.public_key_bytes(),
        ));
        assert!(source_frames_match_recipient(
            &claim,
            &lease,
            &result,
            relay.public_key_bytes(),
            wrong.public_key_bytes(),
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn router_requires_binary_content_type_and_post_only() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;

        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        let router = build_reverse_onion_router(Arc::new(api));
        let wrong_content_type = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(REVERSE_ONION_CLAIM_PATH)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(vec![0u8; 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_content_type.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);

        let wrong_method = router
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(REVERSE_ONION_CLAIM_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(wrong_method.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn router_applies_exact_source_query_request_cap() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;

        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        let router = build_reverse_onion_router(Arc::new(api));
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(REVERSE_ONION_SOURCE_QUERY_PATH)
                    .header(header::CONTENT_TYPE, REVERSE_ONION_BINARY_CONTENT_TYPE)
                    .body(Body::from(vec![0u8; REVERSE_ONION_SOURCE_QUERY_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn router_applies_exact_claim_request_cap() {
        use axum::{body::Body, http::Request};
        use tower::ServiceExt;

        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        let response = build_reverse_onion_router(Arc::new(api))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(REVERSE_ONION_CLAIM_PATH)
                    .header(header::CONTENT_TYPE, REVERSE_ONION_BINARY_CONTENT_TYPE)
                    .body(Body::from(vec![0u8; MAX_REVERSE_ONION_CLAIM_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Authored, not run.
    #[cfg(unix)]
    fn pending_body(polls: Arc<std::sync::atomic::AtomicUsize>) -> axum::body::Body {
        axum::body::Body::from_stream(futures::stream::poll_fn(move |_| {
            polls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::task::Poll::<Option<Result<Bytes, std::io::Error>>>::Pending
        }))
    }

    #[cfg(unix)]
    fn binary_request(body: axum::body::Body) -> axum::http::Request<axum::body::Body> {
        axum::http::Request::builder().method("POST")
            .uri(REVERSE_ONION_CLAIM_PATH)
            .header(header::CONTENT_TYPE, REVERSE_ONION_BINARY_CONTENT_TYPE)
            .body(body).unwrap()
    }

    // [PHALA-QUEUE-RESPONSE-DRAIN 2026-10-08 by Codex] Authored, not run.
    // Exercise actual recovery HTTP bytes, not a mock success response.
    #[cfg(unix)]
    #[tokio::test(start_paused = true)]
    async fn lease_http_body_retains_capacity_and_survives_cancelled_drain() {
        use tower::ServiceExt;
        for outcome in ["collect", "drop", "expire", "cancelled-drain"] {
            let now = trusted_receive_time();
            let (_directory, queue, api, relay, recipient, source) = fixture_at(now);
            let envelope = test_envelope(&relay, &recipient, [31; 16], now,
                &vec![12; super::super::REVERSE_ONION_RESPONSE_CHUNK_BYTES * 3]);
            let item = ReverseOnionQueueItem::new(
                [13; 32], envelope.route_id, [14; 32], source.public_key_bytes(),
                [15; 32], api.recipient, [16; 32],
                encode_blind_relay_envelope(&envelope).unwrap(), now + 60,
            ).unwrap();
            queue.enqueue(&item, now).unwrap();
            let claim = ReverseOnionFrameV1::claim(
                relay.public_key_bytes(), [17; 16], now, now + 30, &recipient,
            ).unwrap().encode();
            let issued = handle_claim_with_policy(&queue, &relay, api.recipient, &claim, now, true);
            assert_eq!(issued.status(), StatusCode::OK);
            let api = Arc::new(api.recovery_only());
            let router = build_reverse_onion_router(Arc::clone(&api));
            let response = router.clone().oneshot(binary_request(
                axum::body::Body::from(claim.clone()))).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(api.blocking.available_permits(), 0);
            let rejected = router.clone().oneshot(binary_request(
                axum::body::Body::from(vec![0u8; 1]))).await.unwrap();
            assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
            match outcome {
                "collect" => {
                    use futures::StreamExt;
                    let mut stream = response.into_body().into_data_stream();
                    let mut bytes = Vec::new();
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk.unwrap();
                        assert!(chunk.len() <= super::super::REVERSE_ONION_RESPONSE_CHUNK_BYTES);
                        bytes.extend_from_slice(&chunk);
                    }
                    assert_eq!(bytes.as_slice(), issued.body());
                },
                "drop" => drop(response),
                "expire" => {
                    tokio::time::advance(super::super::REVERSE_ONION_RESPONSE_TIMEOUT).await;
                    drop(api.try_permit().expect("expired idle body must release admission"));
                    assert!(axum::body::to_bytes(response.into_body(), MAX_REVERSE_ONION_FRAME_BYTES)
                        .await.is_err());
                },
                _ => {
                    let mut first = Box::pin(api.shutdown_and_drain());
                    assert!(futures::poll!(first.as_mut()).is_pending());
                    drop(first);
                    let elapsed = super::super::REVERSE_ONION_RESPONSE_TIMEOUT
                        - std::time::Duration::from_secs(1);
                    tokio::time::advance(elapsed).await;
                    let mut retry = Box::pin(api.shutdown_and_drain());
                    assert!(futures::poll!(retry.as_mut()).is_pending());
                    tokio::time::advance(std::time::Duration::from_secs(1)).await;
                    retry.await;
                    assert!(axum::body::to_bytes(response.into_body(), MAX_REVERSE_ONION_FRAME_BYTES)
                        .await.is_err());
                    assert!(api.try_permit().is_err());
                },
            }
            assert_eq!(api.blocking.available_permits(), 1);
            // Delivery expiry/drop cannot delete custody or mint a new Lease.
            assert_eq!(handle_claim_with_policy(&queue, &relay, api.recipient,
                &claim, now + 31, false), issued);
            api.shutdown_and_drain().await;
        }
    }

    #[cfg(unix)]
    #[tokio::test(start_paused = true)]
    async fn live_ingress_reclaims_the_same_unpolled_http_response() {
        let (_directory, queue, api, relay, recipient, source) = fixture();
        let admission = live_admission_for_fixture(
            queue, &relay, &recipient, &source);
        let api = api.with_live_private_admission(
            Arc::clone(&admission), Arc::new(PeerStore::new())).unwrap();
        assert!(Arc::ptr_eq(&api.responses, &admission.queue_response_registry()));
        let response = retain_queue_http_response(
            into_http_reply(ReverseOnionApiReply::frame(StatusCode::OK, vec![87; 48 * 1024])),
            &api.responses, ReverseOnionApiPermit(Arc::new(api.try_permit().unwrap())),
            MAX_REVERSE_ONION_FRAME_BYTES);
        assert_eq!(response.status(), StatusCode::OK);
        assert!(admission.try_queue_permit().is_err());
        tokio::time::advance(super::super::REVERSE_ONION_RESPONSE_TIMEOUT).await;
        drop(admission.try_queue_permit().expect("ingress must expire the shared body"));
        assert_eq!(api.blocking.available_permits(), 1);
        assert!(axum::body::to_bytes(response.into_body(), MAX_REVERSE_ONION_FRAME_BYTES)
            .await.is_err());
        admission.shutdown_and_drain().await;
        assert!(api.try_permit().is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn response_delivery_uses_evidence_cap_only_for_source_query() {
        use tower::ServiceExt;
        // Synthetic handler isolates body ownership; it does not model crypto.
        for path in [REVERSE_ONION_CLAIM_PATH, REVERSE_ONION_SOURCE_QUERY_PATH] {
            let (_directory, _, api, _, _, _) = fixture();
            let api = Arc::new(api);
            let payload = vec![83u8; MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES];
            let router = Router::new().route(path, post(move || async move {
                ([(header::CONTENT_TYPE, REVERSE_ONION_BINARY_CONTENT_TYPE)], payload)
                    .into_response()
            })).route_layer(middleware::from_fn_with_state(
                Arc::clone(&api), reverse_onion_body_admission));
            let request = axum::http::Request::builder().method("POST").uri(path)
                .header(header::CONTENT_TYPE, REVERSE_ONION_BINARY_CONTENT_TYPE)
                .body(axum::body::Body::empty()).unwrap();
            let response = router.oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(api.blocking.available_permits(), 0);
            let bytes = axum::body::to_bytes(response.into_body(),
                MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES).await;
            if path == REVERSE_ONION_SOURCE_QUERY_PATH {
                assert_eq!(bytes.unwrap(), Bytes::from(vec![83u8; MAX_REVERSE_ONION_SOURCE_EVIDENCE_BYTES]));
            } else {
                assert!(bytes.is_err());
            }
            assert_eq!(api.blocking.available_permits(), 1);
            api.shutdown_and_drain().await;
        }
    }

    // [PHALA-QUEUE-EFFECT-ADMISSION 2026-10-08 by Codex] Authored, not run.
    // Real middleware/permit handoff with a controlled blocking worker, not a
    // SQLite mutation, signature verifier, or evidence-acceptance substitute.
    #[cfg(unix)]
    #[tokio::test]
    async fn stopped_http_handoff_and_cancelled_worker_keep_the_original_slot() {
        use tower::ServiceExt;
        for cancel_http in [false, true] {
            let (_directory, _, api, _, _, _) = fixture();
            let api = Arc::new(api);
            let (release, released) = std::sync::mpsc::channel::<()>();
            let released = Arc::new(std::sync::Mutex::new(Some(released)));
            let (entered, started) = tokio::sync::oneshot::channel::<()>();
            let entered = Arc::new(std::sync::Mutex::new(Some(entered)));
            let router = Router::new().route(REVERSE_ONION_CLAIM_PATH,
                post(move |Extension(permit): Extension<ReverseOnionApiPermit>| {
                    let released = Arc::clone(&released);
                    let entered = Arc::clone(&entered);
                    async move {
                        let released = released.lock().unwrap().take().unwrap();
                        let entered = entered.lock().unwrap().take().unwrap();
                        tokio::task::spawn_blocking(move || {
                            let _permit = permit;
                            entered.send(()).unwrap();
                            released.recv().unwrap();
                            into_http_reply(ReverseOnionApiReply::frame(
                                StatusCode::OK, vec![89; 48 * 1024]))
                        }).await.unwrap()
                    }
                })).route_layer(middleware::from_fn_with_state(
                    Arc::clone(&api), reverse_onion_body_admission));
            let http = tokio::spawn(router.oneshot(binary_request(axum::body::Body::empty())));
            started.await.unwrap();
            assert_eq!(api.blocking.available_permits(), 0);
            let mut drain = Box::pin(api.shutdown_and_drain());
            assert!(futures::poll!(drain.as_mut()).is_pending());
            if cancel_http {
                http.abort();
                assert!(http.await.unwrap_err().is_cancelled());
                assert!(futures::poll!(drain.as_mut()).is_pending());
                release.send(()).unwrap();
            } else {
                release.send(()).unwrap();
                let response = http.await.unwrap().unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(api.blocking.available_permits(), 0);
                assert!(futures::poll!(drain.as_mut()).is_pending());
                drop(response);
            }
            drain.await;
            assert_eq!(api.blocking.available_permits(), 1);
            assert!(api.responses.expire().is_none());
            assert!(api.try_permit().is_err());
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn router_uses_one_permit_through_body_and_handler() {
        use tower::ServiceExt;
        let (_directory, _queue, api, _, _, _) = fixture();
        let api = Arc::new(api);
        let response = build_reverse_onion_router(Arc::clone(&api))
            .oneshot(binary_request(axum::body::Body::from(vec![0u8; 1])))
            .await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(api.blocking.available_permits(), 1);
    }

    #[cfg(unix)]
    #[tokio::test(start_paused = true)]
    async fn slow_body_is_bounded_and_shutdown_drains_after_timeout() {
        use tower::ServiceExt;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (_directory, _queue, api, _, _, _) = fixture();
        let api = Arc::new(api);
        let router = build_reverse_onion_router(Arc::clone(&api));
        let polls = Arc::new(AtomicUsize::new(0));
        let mut first = Box::pin(router.clone().oneshot(binary_request(pending_body(polls.clone()))));
        assert!(futures::poll!(first.as_mut()).is_pending());
        assert!(polls.load(Ordering::SeqCst) > 0);
        assert_eq!(api.blocking.available_permits(), 0);
        let rejected_polls = Arc::new(AtomicUsize::new(0));
        let rejected = router.clone().oneshot(binary_request(pending_body(rejected_polls.clone())))
            .await.unwrap();
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(rejected_polls.load(Ordering::SeqCst), 0);
        api.request_stop();
        let mut drain = Box::pin(api.shutdown_and_drain());
        assert!(futures::poll!(drain.as_mut()).is_pending());
        tokio::time::advance(super::super::REVERSE_ONION_REQUEST_BODY_TIMEOUT).await;
        assert_eq!(first.await.unwrap().status(), StatusCode::REQUEST_TIMEOUT);
        drain.await;
        let stopped_polls = Arc::new(AtomicUsize::new(0));
        let stopped = router.oneshot(binary_request(pending_body(stopped_polls.clone())))
            .await.unwrap();
        assert_eq!(stopped.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(stopped_polls.load(Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelling_pre_body_request_releases_admission() {
        use tower::ServiceExt;
        let (_directory, _queue, api, _, _, _) = fixture();
        let api = Arc::new(api);
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut pending = Box::pin(build_reverse_onion_router(Arc::clone(&api))
            .oneshot(binary_request(pending_body(polls))));
        assert!(futures::poll!(pending.as_mut()).is_pending());
        assert_eq!(api.blocking.available_permits(), 0);
        drop(pending);
        assert_eq!(api.blocking.available_permits(), 1);
        api.shutdown_and_drain().await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn router_builder_has_no_unrelated_default_route() {
        use axum::http::Request;
        use tower::ServiceExt;

        let (_directory, _queue, api, _relay, _recipient, _source) = fixture();
        // [PHALA-REVERSE-BODY-ADMISSION 2026-10-07 by Codex] Unknown
        // paths must bypass body admission and retain the original 404.
        let polls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let response = build_reverse_onion_router(Arc::new(api))
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/admin/reverse-onion")
                    .body(pending_body(polls.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(polls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}

// [REVERSE-WORKER-CLOCK 2026-10-05 by Codex] Advance the trusted receipt
// timestamp by monotonic queue wait, preserving the caller's injectable clock.
// Overflow must reject, never wrap into a fresh lease/query window.
// [REVERSE-WORKER-CLOCK-SHARED 2026-10-06 by Codex] Both source API and
// private relay enqueue use this same checked monotonic advancement.
pub(super) fn worker_time(received_at: u64, elapsed: std::time::Duration) -> Option<u64> {
    received_at.checked_add(elapsed.as_secs())
}

fn validate_recipient(
    relay: [u8; 32],
    recipient: [u8; 32],
    max_in_flight: usize,
) -> Result<(), ReverseOnionApiConfigError> {
    if !(1..=64).contains(&max_in_flight)
        || relay == recipient
        // [PHALA-PULL-IDENTITY-REPAIR 2026-10-08 by Codex] A parsable
        // all-zero encoding is still the protocol's forbidden identity sentinel.
        || relay == [0; 32] || recipient == [0; 32]
        || IdentityPublicKey::from_bytes(&recipient).is_err()
    {
        return Err(ReverseOnionApiConfigError::Invalid);
    }
    Ok(())
}
