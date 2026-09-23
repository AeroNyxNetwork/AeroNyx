// ============================================================================
// File: crates/aeronyx-server/src/api/discovery_endpoint_verification.rs
// ============================================================================
//! Unmounted authenticated HTTP adapter for discovery endpoint proofs.
//!
//! The router is intentionally unusable without a per-request
//! [`VerifiedEndpointProofPeerContext`] inserted by an upstream authenticated
//! peer middleware. Request bodies never select their target identity or
//! challenger context. This adapter has no discovery-promotion authority.

use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use aeronyx_core::protocol::{
    DiscoveryEndpointChallengeV1, DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1,
    DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1,
};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Extension, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;

use crate::services::{
    DiscoveryEndpointChallengeIssueOutcome, DiscoveryEndpointChallengeRequestV1,
    DiscoveryEndpointProofConsumeOutcome, DiscoveryEndpointVerificationService,
};

// [AUTHENTICATED-ENDPOINT-PROOF-ADAPTER 2026-09-24 by Codex] Keep a distinct,
// strict wire around the core proof frames without making it a public oracle.
const ADAPTER_MAGIC: [u8; 4] = *b"ADEA";
const ADAPTER_VERSION_V1: u8 = 1;
const HEADER_BYTES: usize = 8;
const KIND_ISSUE_REQUEST: u8 = 1;
const KIND_VERIFY_REQUEST: u8 = 2;
const KIND_ISSUE_RESPONSE: u8 = 129;
const KIND_VERIFY_RESPONSE: u8 = 130;
const MAX_ENDPOINT_BYTES: usize = 64;
const ISSUE_FIXED_BODY_BYTES: usize = 32 + 32 + 1;
const VERIFY_BODY_BYTES: usize = 32
    + 2
    + DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1
    + 2
    + DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1;
const MAX_REQUEST_FRAME_BYTES: usize = HEADER_BYTES + VERIFY_BODY_BYTES;

const ISSUE_OUTCOME_ISSUED: u8 = 1;
const ISSUE_OUTCOME_EXISTING: u8 = 2;
const ISSUE_OUTCOME_CONFLICT: u8 = 3;
const ISSUE_OUTCOME_EXPIRED: u8 = 4;
const ISSUE_OUTCOME_AT_CAPACITY: u8 = 5;
const VERIFY_OUTCOME_ACCEPTED: u8 = 1;
const VERIFY_OUTCOME_EXISTING: u8 = 2;
const VERIFY_OUTCOME_CONFLICT: u8 = 3;
const VERIFY_OUTCOME_EXPIRED: u8 = 4;
const VERIFY_OUTCOME_NOT_FOUND: u8 = 5;
const VERIFY_OUTCOME_REJECTED: u8 = 6;

/// Authenticated peer identity and route-local challenger context.
///
/// Construction is crate-private: a future mount must derive this value only
/// after transport authentication and insert it as a per-request extension.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct VerifiedEndpointProofPeerContext {
    peer_node_id: [u8; 32],
    challenger_context: [u8; 32],
}

impl fmt::Debug for VerifiedEndpointProofPeerContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VerifiedEndpointProofPeerContext")
            .finish_non_exhaustive()
    }
}

impl VerifiedEndpointProofPeerContext {
    /// Builds a context only for an already authenticated peer request.
    #[allow(dead_code, reason = "Stage C router is deliberately unmounted")]
    pub(crate) fn from_authenticated_peer(
        peer_node_id: [u8; 32],
        challenger_context: [u8; 32],
    ) -> Option<Self> {
        if is_reserved(&peer_node_id) || is_reserved(&challenger_context) {
            return None;
        }
        Some(Self {
            peer_node_id,
            challenger_context,
        })
    }
}

trait EndpointProofClock: Send + Sync {
    fn now(&self) -> u64;
}

#[allow(dead_code, reason = "Stage C router is deliberately unmounted")]
struct SystemEndpointProofClock;

impl EndpointProofClock for SystemEndpointProofClock {
    fn now(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs())
    }
}

#[derive(Clone)]
struct EndpointProofAdapterState {
    service: Arc<DiscoveryEndpointVerificationService>,
    clock: Arc<dyn EndpointProofClock>,
}

/// Builds an unmounted router that requires authenticated peer extensions.
///
/// The returned router carries no authentication middleware and therefore
/// returns `401` until a trusted parent router injects one verified context per
/// request. It must never be mounted directly on a public listener.
#[allow(dead_code, reason = "Stage C router is deliberately unmounted")]
pub(crate) fn build_discovery_endpoint_verification_router(
    service: Arc<DiscoveryEndpointVerificationService>,
) -> Router {
    build_router_with_clock(service, Arc::new(SystemEndpointProofClock))
}

fn build_router_with_clock(
    service: Arc<DiscoveryEndpointVerificationService>,
    clock: Arc<dyn EndpointProofClock>,
) -> Router {
    Router::new()
        .route(
            "/api/discovery/endpoint-proof/challenge",
            post(issue_challenge),
        )
        .route("/api/discovery/endpoint-proof/verify", post(verify_proof))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_FRAME_BYTES))
        .with_state(EndpointProofAdapterState { service, clock })
}

async fn issue_challenge(
    State(state): State<EndpointProofAdapterState>,
    auth: Option<Extension<VerifiedEndpointProofPeerContext>>,
    body: Bytes,
) -> Response {
    let Some(Extension(auth)) = auth else {
        return coarse_response(StatusCode::UNAUTHORIZED, "authentication_required");
    };
    let Ok(request) = decode_issue_request(&body) else {
        return coarse_response(StatusCode::BAD_REQUEST, "request_rejected");
    };
    let Ok(request) = DiscoveryEndpointChallengeRequestV1::new(
        request.request_id,
        auth.peer_node_id,
        request.descriptor_commitment,
        request.endpoint,
        auth.challenger_context,
    ) else {
        return coarse_response(StatusCode::BAD_REQUEST, "request_rejected");
    };
    let Ok(outcome) = state.service.issue_at(request, state.clock.now()) else {
        return coarse_response(StatusCode::SERVICE_UNAVAILABLE, "service_unavailable");
    };
    let status = match outcome {
        DiscoveryEndpointChallengeIssueOutcome::Conflict => StatusCode::CONFLICT,
        DiscoveryEndpointChallengeIssueOutcome::Expired => StatusCode::GONE,
        DiscoveryEndpointChallengeIssueOutcome::AtCapacity => StatusCode::TOO_MANY_REQUESTS,
        DiscoveryEndpointChallengeIssueOutcome::Issued { .. }
        | DiscoveryEndpointChallengeIssueOutcome::Existing { .. } => StatusCode::OK,
    };
    encode_issue_response(outcome).map_or_else(
        |()| coarse_response(StatusCode::INTERNAL_SERVER_ERROR, "response_unavailable"),
        |frame| (status, frame).into_response(),
    )
}

async fn verify_proof(
    State(state): State<EndpointProofAdapterState>,
    auth: Option<Extension<VerifiedEndpointProofPeerContext>>,
    body: Bytes,
) -> Response {
    let Some(Extension(auth)) = auth else {
        return coarse_response(StatusCode::UNAUTHORIZED, "authentication_required");
    };
    let Ok(request) = decode_verify_request(&body) else {
        return coarse_response(StatusCode::BAD_REQUEST, "request_rejected");
    };
    let Ok(challenge) = DiscoveryEndpointChallengeV1::decode(request.challenge_frame) else {
        return coarse_response(StatusCode::BAD_REQUEST, "request_rejected");
    };
    if challenge.target_node_id() != auth.peer_node_id
        || challenge.challenger_context() != auth.challenger_context
    {
        return coarse_response(StatusCode::FORBIDDEN, "authentication_mismatch");
    }
    let Ok(outcome) = state.service.verify_and_consume_at(
        request.request_id,
        request.challenge_frame,
        request.proof_frame,
        state.clock.now(),
    ) else {
        return coarse_response(StatusCode::SERVICE_UNAVAILABLE, "service_unavailable");
    };
    let status = match outcome {
        DiscoveryEndpointProofConsumeOutcome::Accepted
        | DiscoveryEndpointProofConsumeOutcome::Existing => StatusCode::OK,
        DiscoveryEndpointProofConsumeOutcome::Conflict => StatusCode::CONFLICT,
        DiscoveryEndpointProofConsumeOutcome::Expired => StatusCode::GONE,
        DiscoveryEndpointProofConsumeOutcome::NotFound => StatusCode::NOT_FOUND,
        DiscoveryEndpointProofConsumeOutcome::Rejected => StatusCode::BAD_REQUEST,
    };
    (status, encode_verify_response(outcome)).into_response()
}

struct IssueRequest<'a> {
    request_id: [u8; 32],
    descriptor_commitment: [u8; 32],
    endpoint: &'a str,
}

struct VerifyRequest<'a> {
    request_id: [u8; 32],
    challenge_frame: &'a [u8],
    proof_frame: &'a [u8],
}

fn decode_issue_request(bytes: &[u8]) -> Result<IssueRequest<'_>, ()> {
    let body = decode_frame(bytes, KIND_ISSUE_REQUEST)?;
    if body.len() < ISSUE_FIXED_BODY_BYTES {
        return Err(());
    }
    let request_id = take_array::<32>(body, 0)?;
    let descriptor_commitment = take_array::<32>(body, 32)?;
    let endpoint_len = usize::from(body[64]);
    if endpoint_len == 0
        || endpoint_len > MAX_ENDPOINT_BYTES
        || body.len() != ISSUE_FIXED_BODY_BYTES + endpoint_len
        || is_reserved(&request_id)
        || is_reserved(&descriptor_commitment)
    {
        return Err(());
    }
    let endpoint = std::str::from_utf8(&body[65..]).map_err(|_| ())?;
    Ok(IssueRequest {
        request_id,
        descriptor_commitment,
        endpoint,
    })
}

fn decode_verify_request(bytes: &[u8]) -> Result<VerifyRequest<'_>, ()> {
    let body = decode_frame(bytes, KIND_VERIFY_REQUEST)?;
    if body.len() != VERIFY_BODY_BYTES {
        return Err(());
    }
    let request_id = take_array::<32>(body, 0)?;
    if is_reserved(&request_id) {
        return Err(());
    }
    let challenge_len = usize::from(u16::from_be_bytes(take_array::<2>(body, 32)?));
    if challenge_len != DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1 {
        return Err(());
    }
    let challenge_start = 34;
    let challenge_end = challenge_start + challenge_len;
    let proof_len = usize::from(u16::from_be_bytes(take_array::<2>(body, challenge_end)?));
    if proof_len != DISCOVERY_ENDPOINT_PROOF_FRAME_BYTES_V1 {
        return Err(());
    }
    let proof_start = challenge_end + 2;
    let proof_end = proof_start + proof_len;
    if proof_end != body.len() {
        return Err(());
    }
    Ok(VerifyRequest {
        request_id,
        challenge_frame: &body[challenge_start..challenge_end],
        proof_frame: &body[proof_start..proof_end],
    })
}

fn encode_issue_response(outcome: DiscoveryEndpointChallengeIssueOutcome) -> Result<Vec<u8>, ()> {
    let (code, challenge_frame) = match outcome {
        DiscoveryEndpointChallengeIssueOutcome::Issued { challenge_frame } => {
            (ISSUE_OUTCOME_ISSUED, challenge_frame)
        }
        DiscoveryEndpointChallengeIssueOutcome::Existing { challenge_frame } => {
            (ISSUE_OUTCOME_EXISTING, challenge_frame)
        }
        DiscoveryEndpointChallengeIssueOutcome::Conflict => (ISSUE_OUTCOME_CONFLICT, Vec::new()),
        DiscoveryEndpointChallengeIssueOutcome::Expired => (ISSUE_OUTCOME_EXPIRED, Vec::new()),
        DiscoveryEndpointChallengeIssueOutcome::AtCapacity => {
            (ISSUE_OUTCOME_AT_CAPACITY, Vec::new())
        }
    };
    if !challenge_frame.is_empty()
        && challenge_frame.len() != DISCOVERY_ENDPOINT_CHALLENGE_FRAME_BYTES_V1
    {
        return Err(());
    }
    let frame_len = u16::try_from(challenge_frame.len()).map_err(|_| ())?;
    let mut body = Vec::with_capacity(3 + challenge_frame.len());
    body.push(code);
    body.extend_from_slice(&frame_len.to_be_bytes());
    body.extend_from_slice(&challenge_frame);
    encode_frame(KIND_ISSUE_RESPONSE, &body)
}

fn encode_verify_response(outcome: DiscoveryEndpointProofConsumeOutcome) -> Vec<u8> {
    let code = match outcome {
        DiscoveryEndpointProofConsumeOutcome::Accepted => VERIFY_OUTCOME_ACCEPTED,
        DiscoveryEndpointProofConsumeOutcome::Existing => VERIFY_OUTCOME_EXISTING,
        DiscoveryEndpointProofConsumeOutcome::Conflict => VERIFY_OUTCOME_CONFLICT,
        DiscoveryEndpointProofConsumeOutcome::Expired => VERIFY_OUTCOME_EXPIRED,
        DiscoveryEndpointProofConsumeOutcome::NotFound => VERIFY_OUTCOME_NOT_FOUND,
        DiscoveryEndpointProofConsumeOutcome::Rejected => VERIFY_OUTCOME_REJECTED,
    };
    let mut frame = Vec::with_capacity(HEADER_BYTES + 1);
    frame.extend_from_slice(&ADAPTER_MAGIC);
    frame.push(ADAPTER_VERSION_V1);
    frame.push(KIND_VERIFY_RESPONSE);
    frame.extend_from_slice(&1_u16.to_be_bytes());
    frame.push(code);
    frame
}

fn encode_frame(kind: u8, body: &[u8]) -> Result<Vec<u8>, ()> {
    let body_len = u16::try_from(body.len()).map_err(|_| ())?;
    let mut frame = Vec::with_capacity(HEADER_BYTES + body.len());
    frame.extend_from_slice(&ADAPTER_MAGIC);
    frame.push(ADAPTER_VERSION_V1);
    frame.push(kind);
    frame.extend_from_slice(&body_len.to_be_bytes());
    frame.extend_from_slice(body);
    Ok(frame)
}

fn decode_frame(bytes: &[u8], expected_kind: u8) -> Result<&[u8], ()> {
    if bytes.len() < HEADER_BYTES || bytes[..4] != ADAPTER_MAGIC {
        return Err(());
    }
    if bytes[4] != ADAPTER_VERSION_V1 || bytes[5] != expected_kind {
        return Err(());
    }
    let body_len = usize::from(u16::from_be_bytes([bytes[6], bytes[7]]));
    if bytes.len() != HEADER_BYTES + body_len {
        return Err(());
    }
    Ok(&bytes[HEADER_BYTES..])
}

fn take_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N], ()> {
    let end = offset.checked_add(N).ok_or(())?;
    let value = bytes.get(offset..end).ok_or(())?;
    let mut array = [0; N];
    array.copy_from_slice(value);
    Ok(array)
}

fn coarse_response(status: StatusCode, class: &'static str) -> Response {
    (status, class).into_response()
}

fn is_reserved<const N: usize>(value: &[u8; N]) -> bool {
    value.iter().all(|byte| *byte == 0) || value.iter().all(|byte| *byte == u8::MAX)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::DiscoveryEndpointProofV1;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    use crate::services::DiscoveryEndpointVerificationConfig;

    const NOW: u64 = 1_800_200_000;

    struct FixedClock(u64);

    impl EndpointProofClock for FixedClock {
        fn now(&self) -> u64 {
            self.0
        }
    }

    fn key(seed: u8) -> IdentityKeyPair {
        IdentityKeyPair::from_bytes(&[seed; 32]).expect("fixed key")
    }

    fn test_router(max_entries: usize, auth: Option<VerifiedEndpointProofPeerContext>) -> Router {
        let service = Arc::new(
            DiscoveryEndpointVerificationService::new(
                key(3),
                DiscoveryEndpointVerificationConfig {
                    max_entries,
                    challenge_ttl_secs: 60,
                },
            )
            .expect("service"),
        );
        let router = build_router_with_clock(service, Arc::new(FixedClock(NOW)));
        match auth {
            Some(auth) => router.layer(Extension(auth)),
            None => router,
        }
    }

    fn auth(target: &IdentityKeyPair, context: u8) -> VerifiedEndpointProofPeerContext {
        VerifiedEndpointProofPeerContext::from_authenticated_peer(
            target.public_key_bytes(),
            [context; 32],
        )
        .expect("auth context")
    }

    fn issue_frame(request_id: [u8; 32], descriptor: [u8; 32], endpoint: &str) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&request_id);
        body.extend_from_slice(&descriptor);
        body.push(u8::try_from(endpoint.len()).expect("bounded endpoint"));
        body.extend_from_slice(endpoint.as_bytes());
        encode_frame(KIND_ISSUE_REQUEST, &body).expect("issue frame")
    }

    fn verify_frame(request_id: [u8; 32], challenge: &[u8], proof: &[u8]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&request_id);
        body.extend_from_slice(
            &u16::try_from(challenge.len())
                .expect("challenge length")
                .to_be_bytes(),
        );
        body.extend_from_slice(challenge);
        body.extend_from_slice(
            &u16::try_from(proof.len())
                .expect("proof length")
                .to_be_bytes(),
        );
        body.extend_from_slice(proof);
        encode_frame(KIND_VERIFY_REQUEST, &body).expect("verify frame")
    }

    async fn post_frame(router: Router, path: &str, frame: Vec<u8>) -> (StatusCode, Vec<u8>) {
        let response = router
            .oneshot(
                Request::post(path)
                    .body(Body::from(frame))
                    .expect("request"),
            )
            .await
            .expect("response");
        let status = response.status();
        let body = to_bytes(response.into_body(), 2048)
            .await
            .expect("response bytes")
            .to_vec();
        (status, body)
    }

    fn decode_issue_response(bytes: &[u8]) -> (u8, Vec<u8>) {
        let body = decode_frame(bytes, KIND_ISSUE_RESPONSE).expect("issue response");
        let len = usize::from(u16::from_be_bytes([body[1], body[2]]));
        assert_eq!(body.len(), 3 + len);
        (body[0], body[3..].to_vec())
    }

    fn decode_verify_response(bytes: &[u8]) -> u8 {
        let body = decode_frame(bytes, KIND_VERIFY_RESPONSE).expect("verify response");
        assert_eq!(body.len(), 1);
        body[0]
    }

    #[tokio::test]
    async fn authenticated_issue_and_response_consume_once() {
        let target = key(9);
        let context = [0x44; 32];
        let auth = auth(&target, 0x44);
        let request_id = [1; 32];
        let service = Arc::new(
            DiscoveryEndpointVerificationService::new(
                key(3),
                DiscoveryEndpointVerificationConfig {
                    max_entries: 4,
                    challenge_ttl_secs: 60,
                },
            )
            .expect("service"),
        );
        let router =
            build_router_with_clock(service, Arc::new(FixedClock(NOW))).layer(Extension(auth));
        let (_, issue_body) = post_frame(
            router.clone(),
            "/api/discovery/endpoint-proof/challenge",
            issue_frame(request_id, [0x22; 32], "8.8.8.8:51820"),
        )
        .await;
        let (issue_code, challenge_frame) = decode_issue_response(&issue_body);
        assert_eq!(issue_code, ISSUE_OUTCOME_ISSUED);
        let challenge = DiscoveryEndpointChallengeV1::decode(&challenge_frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &context, NOW + 1, &target)
            .expect("proof")
            .encode();
        let (status, body) = post_frame(
            router.clone(),
            "/api/discovery/endpoint-proof/verify",
            verify_frame(request_id, &challenge_frame, &proof),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(decode_verify_response(&body), VERIFY_OUTCOME_ACCEPTED);
        let (_, replay_body) = post_frame(
            router,
            "/api/discovery/endpoint-proof/verify",
            verify_frame(request_id, &challenge_frame, &proof),
        )
        .await;
        assert_eq!(
            decode_verify_response(&replay_body),
            VERIFY_OUTCOME_EXISTING
        );
    }

    #[tokio::test]
    async fn missing_or_wrong_authenticated_identity_fails_closed() {
        let target = key(9);
        let request_id = [2; 32];
        let issue = issue_frame(request_id, [0x22; 32], "8.8.8.8:51820");
        let (missing_status, _) = post_frame(
            test_router(4, None),
            "/api/discovery/endpoint-proof/challenge",
            issue.clone(),
        )
        .await;
        assert_eq!(missing_status, StatusCode::UNAUTHORIZED);

        let correct_auth = auth(&target, 0x44);
        let service = Arc::new(
            DiscoveryEndpointVerificationService::new(
                key(3),
                DiscoveryEndpointVerificationConfig {
                    max_entries: 4,
                    challenge_ttl_secs: 60,
                },
            )
            .expect("service"),
        );
        let base = build_router_with_clock(service, Arc::new(FixedClock(NOW)));
        let (_, issue_body) = post_frame(
            base.clone().layer(Extension(correct_auth)),
            "/api/discovery/endpoint-proof/challenge",
            issue,
        )
        .await;
        let (_, challenge_frame) = decode_issue_response(&issue_body);
        let challenge = DiscoveryEndpointChallengeV1::decode(&challenge_frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &[0x44; 32], NOW + 1, &target)
            .expect("proof")
            .encode();

        let wrong_target = auth(&key(10), 0x44);
        let (status, _) = post_frame(
            base.layer(Extension(wrong_target)),
            "/api/discovery/endpoint-proof/verify",
            verify_frame(request_id, &challenge_frame, &proof),
        )
        .await;
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn cross_context_and_noncanonical_frames_fail_closed() {
        let target = key(9);
        let request_id = [3; 32];
        let correct_auth = auth(&target, 0x44);
        let service = Arc::new(
            DiscoveryEndpointVerificationService::new(
                key(3),
                DiscoveryEndpointVerificationConfig {
                    max_entries: 4,
                    challenge_ttl_secs: 60,
                },
            )
            .expect("service"),
        );
        let base = build_router_with_clock(service, Arc::new(FixedClock(NOW)));
        let issue = issue_frame(request_id, [0x22; 32], "8.8.8.8:51820");
        let (_, issue_body) = post_frame(
            base.clone().layer(Extension(correct_auth.clone())),
            "/api/discovery/endpoint-proof/challenge",
            issue.clone(),
        )
        .await;
        let (_, challenge_frame) = decode_issue_response(&issue_body);
        let challenge = DiscoveryEndpointChallengeV1::decode(&challenge_frame).expect("challenge");
        let proof = DiscoveryEndpointProofV1::respond(&challenge, &[0x44; 32], NOW + 1, &target)
            .expect("proof")
            .encode();

        let (cross_status, _) = post_frame(
            base.clone().layer(Extension(auth(&target, 0x45))),
            "/api/discovery/endpoint-proof/verify",
            verify_frame(request_id, &challenge_frame, &proof),
        )
        .await;
        assert_eq!(cross_status, StatusCode::FORBIDDEN);

        let mut trailing = issue;
        trailing.push(0);
        let (trailing_status, _) = post_frame(
            base.clone().layer(Extension(correct_auth.clone())),
            "/api/discovery/endpoint-proof/challenge",
            trailing,
        )
        .await;
        assert_eq!(trailing_status, StatusCode::BAD_REQUEST);

        let oversized = vec![0; MAX_REQUEST_FRAME_BYTES + 1];
        let (oversize_status, _) = post_frame(
            base.layer(Extension(correct_auth)),
            "/api/discovery/endpoint-proof/challenge",
            oversized,
        )
        .await;
        assert_eq!(oversize_status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn duplicate_conflict_and_capacity_are_typed() {
        let target = key(9);
        let auth = auth(&target, 0x44);
        let router = test_router(1, Some(auth));
        let first = issue_frame([4; 32], [0x22; 32], "8.8.8.8:51820");
        let (first_status, first_body) = post_frame(
            router.clone(),
            "/api/discovery/endpoint-proof/challenge",
            first.clone(),
        )
        .await;
        assert_eq!(first_status, StatusCode::OK);
        let (_, first_challenge) = decode_issue_response(&first_body);

        let (_, replay_body) = post_frame(
            router.clone(),
            "/api/discovery/endpoint-proof/challenge",
            first,
        )
        .await;
        let (replay_code, replay_challenge) = decode_issue_response(&replay_body);
        assert_eq!(replay_code, ISSUE_OUTCOME_EXISTING);
        assert_eq!(replay_challenge, first_challenge);

        let changed = issue_frame([4; 32], [0x23; 32], "8.8.8.8:51820");
        let (conflict_status, conflict_body) = post_frame(
            router.clone(),
            "/api/discovery/endpoint-proof/challenge",
            changed,
        )
        .await;
        assert_eq!(conflict_status, StatusCode::CONFLICT);
        assert_eq!(
            decode_issue_response(&conflict_body).0,
            ISSUE_OUTCOME_CONFLICT
        );

        let second = issue_frame([5; 32], [0x22; 32], "8.8.8.8:51820");
        let (capacity_status, capacity_body) =
            post_frame(router, "/api/discovery/endpoint-proof/challenge", second).await;
        assert_eq!(capacity_status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            decode_issue_response(&capacity_body).0,
            ISSUE_OUTCOME_AT_CAPACITY
        );
    }
}
