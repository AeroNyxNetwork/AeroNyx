// ============================================================================
// File: crates/aeronyx-server/src/api/chat_peer/outbound_transport.rs
// ============================================================================
//! Bounded outbound peer-relay carrier preparation, receipt verification,
//! and exact next-hop forwarding composition.
//!
//! [CHAT-PEER-OUTBOUND-SPLIT 2026-09-25 by Codex] Existing chat_peer exports,
//! wire bytes, retry and response contracts remain unchanged; this private
//! module owns only outbound preparation and transport orchestration.

use super::super::chat_peer_observer::{
    BlindRelayForwardObserver, PeerStoreBlindRelayForwardObserver,
};
use super::super::chat_peer_response::{
    BlindRelayInvalidResponseKind, BlindRelayResponseContext, BlindRelayResponseDecision,
    BlindRelayResponseDomain, BlindRelayResponsePolicy, BlindRelayResponseSource,
    BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS,
};
use super::super::chat_peer_retry::{
    BlindRelayRetryContext, BlindRelayRetryDomain, BlindRelayRetryPolicy,
};
use super::super::chat_peer_transport::{BlindRelayTransport, ReqwestBlindRelayTransport};
use super::*;
use crate::api::{canonical_peer_http_url, peer_endpoint_is_public_ip};
use bytes::Bytes;
use tokio::time::sleep;

/// Privacy-safe local failure while preparing an outbound direct request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectRelayRequestPreparationFailure {
    Encoding,
    BodyTooLarge,
    Backpressure,
    Unavailable,
}

/// Local bounded-worker failure or remote cryptographic rejection for one
/// direct custody receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectRelayReceiptVerificationFailure {
    Invalid(&'static str),
    Unavailable,
}

/// Privacy-safe local failure while preparing an outbound blind request.
///
/// This is deliberately distinct from a next-hop failure: no HTTP request has
/// been attempted when one of these variants is returned, so callers must not
/// penalize a selected peer or mark the route surface as exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindRelayRequestPreparationFailure {
    Encoding,
    BodyTooLarge,
    Backpressure,
    Unavailable,
}

/// Domain build failure or local carrier-preparation failure.
///
/// Keeping `Build(E)` generic lets the onion route planner preserve its typed
/// refresh/policy/construction dispositions while this module owns only CPU
/// admission and the wire carrier contract.
#[derive(Debug)]
pub(crate) enum BlindRelayRequestPreparationError<E> {
    Build(E),
    Local(BlindRelayRequestPreparationFailure),
}

/// Peer-invalid evidence or a local bounded-verifier failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlindRelayDeliveryReceiptVerificationFailure {
    Missing,
    Invalid,
    Unavailable,
}

impl BlindRelayRequestPreparationFailure {
    /// Closed aggregate label safe for relay-health telemetry.
    #[must_use]
    pub(crate) const fn reason_bucket(self) -> &'static str {
        // [OUTBOUND-BLIND-REQUEST-PREPARATION 2026-08-31 by Codex] Preserve
        // one coarse local bucket across encoding, policy, and worker faults;
        // the public health surface must not expose request-size distinctions.
        match self {
            Self::Encoding | Self::BodyTooLarge | Self::Backpressure | Self::Unavailable => {
                "onion_request_build_failed"
            }
        }
    }
}

impl DirectRelayRequestPreparationFailure {
    /// Closed aggregate label safe for route-health telemetry.
    #[must_use]
    pub(crate) const fn reason_bucket(self) -> &'static str {
        match self {
            Self::Encoding | Self::BodyTooLarge => "peer_relay_auth_encode_failed",
            // Keep the established closed telemetry vocabulary during rolling
            // upgrades. The typed variant still controls local recovery, while
            // aggregate route evidence avoids inventing a label older nodes
            // would sanitize to `unknown`.
            Self::Backpressure | Self::Unavailable => "peer_relay_auth_encode_failed",
        }
    }
}

/// Opaque, bounded HTTP carrier prepared outside the async I/O runtime.
///
/// [OUTBOUND-DIRECT-HTTP-BODY-PREPARATION 2026-08-31 by Codex] Serialize once
/// under bounded CPU admission, then reuse the exact immutable bytes for v2
/// fanout or v3 retry without blocking a Tokio I/O worker.
///
/// This type intentionally omits `Debug`: its body contains an end-to-end
/// encrypted user envelope. `Bytes` keeps exact retry and fanout clones O(1).
#[derive(Clone)]
pub(crate) struct PreparedPeerChatRelayHttpRequest {
    body: Bytes,
}

impl PreparedPeerChatRelayHttpRequest {
    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.body.clone()
    }
}

/// Prepared v2/v3 carrier whose commitment is mandatory by construction.
///
/// [AUTHENTICATED-DIRECT-CARRIER-TYPE 2026-08-31 by Codex] A signed request
/// without its exact commitment is not representable. This removes repeated
/// runtime `Option` checks from receipt verification and retry orchestration.
#[derive(Clone)]
pub(crate) struct PreparedAuthenticatedPeerChatRelayHttpRequest {
    request: PreparedPeerChatRelayHttpRequest,
    request_commitment: [u8; 32],
}

/// Opaque blind-relay HTTP carrier prepared outside the async I/O runtime.
///
/// The full request is dropped after serialization. Retaining only the exact
/// route id needed for receipt verification avoids holding both a potentially
/// large onion object graph and its JSON representation during network I/O.
/// This type intentionally omits `Debug` because its body is encrypted user
/// data even though the node cannot decrypt it.
pub(crate) struct PreparedPeerBlindRelayHttpRequest {
    body: Bytes,
    route_id: [u8; 16],
}

/// Exact hop-to-hop request retained across bounded transport retries.
///
/// [PREPARED-BLIND-FORWARD-CARRIER 2026-08-31 by Codex] Response policy needs
/// the typed request while HTTP needs only immutable bytes. Keeping both in one
/// carrier guarantees every retry uses the same serialization without asking
/// the asynchronous transport layer to encode ciphertext again.
pub(super) struct PreparedBlindRelayForwardRequest {
    request: Arc<PeerBlindRelayRequest>,
    http: PreparedPeerBlindRelayHttpRequest,
}

impl PreparedPeerBlindRelayHttpRequest {
    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.body.clone()
    }

    #[must_use]
    pub(crate) const fn route_id(&self) -> &[u8; 16] {
        &self.route_id
    }
}

impl PreparedAuthenticatedPeerChatRelayHttpRequest {
    #[must_use]
    pub(crate) fn body(&self) -> Bytes {
        self.request.body()
    }

    #[must_use]
    pub(crate) const fn request_commitment(&self) -> [u8; 32] {
        self.request_commitment
    }
}

/// Internal result of one accepted next-hop relay round.
///
/// The observation timestamp is deliberately process-local and never enters
/// the peer wire contract. Callers use it for every success-side state write,
/// keeping receipt verification and route evidence on one clock snapshot.
pub(super) struct BlindRelayForwardOutcome {
    pub(super) response: PeerBlindRelayResponse,
    pub(super) observed_at: u64,
}

/// Prepares one legacy direct request outside the asynchronous I/O runtime.
pub(crate) async fn prepare_peer_chat_relay_request_v1(
    envelope: ChatEnvelope,
) -> Result<PreparedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    prepare_direct_peer_relay_request(move || {
        encode_prepared_peer_chat_relay_request(&PeerChatRelayRequest { envelope })
    })
    .await
}

/// Prepares one authenticated v2 request outside the asynchronous I/O runtime.
pub(crate) async fn prepare_peer_chat_relay_request_v2(
    envelope: ChatEnvelope,
    node_identity: Arc<IdentityKeyPair>,
) -> Result<PreparedAuthenticatedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    prepare_direct_peer_relay_request(move || {
        let (request, request_commitment) =
            PeerChatRelayRequestV2::sign_with_commitment(envelope, node_identity.as_ref())
                .map_err(|_| DirectRelayRequestPreparationFailure::Encoding)?;
        encode_prepared_authenticated_peer_chat_relay_request(&request, request_commitment)
    })
    .await
}

/// Prepares one target-bound v3 request outside the async I/O runtime.
pub(crate) async fn prepare_peer_chat_relay_request_v3(
    envelope: ChatEnvelope,
    target_node_id: [u8; 32],
    node_identity: Arc<IdentityKeyPair>,
) -> Result<PreparedAuthenticatedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    prepare_direct_peer_relay_request(move || {
        let (request, request_commitment) = PeerChatRelayRequestV3::sign_with_commitment(
            envelope,
            target_node_id,
            node_identity.as_ref(),
        )
        .map_err(|_| DirectRelayRequestPreparationFailure::Encoding)?;
        encode_prepared_authenticated_peer_chat_relay_request(&request, request_commitment)
    })
    .await
}

/// Builds and serializes one blind request as a single bounded CPU operation.
///
/// [ATOMIC-OUTBOUND-BLIND-PREPARATION 2026-08-31 by Codex] Route planning,
/// onion KEM work, signing, and JSON encoding may be composed inside `build`
/// without returning to a Tokio I/O worker between CPU-heavy stages. The
/// context is carried beside the immutable body for receipt verification.
pub(crate) async fn prepare_peer_blind_relay_http_request_with<T, E, F>(
    build: F,
) -> Result<(PreparedPeerBlindRelayHttpRequest, T), BlindRelayRequestPreparationError<E>>
where
    T: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Result<(PeerBlindRelayRequest, T), E> + Send + 'static,
{
    let permit = blind_relay_crypto_admission()
        .try_acquire_owned()
        .map_err(|_| {
            BlindRelayRequestPreparationError::Local(
                BlindRelayRequestPreparationFailure::Backpressure,
            )
        })?;
    tokio::task::spawn_blocking(move || {
        // [OUTBOUND-BLIND-REQUEST-PREPARATION 2026-08-31 by Codex] Keep the
        // permit in the worker after caller cancellation. The composed work
        // has no I/O or effects, so abandoning the result is always retry-safe.
        let _permits = BlindRelayCryptoPermits::total_only(permit);
        let (request, context) = build().map_err(BlindRelayRequestPreparationError::Build)?;
        let request = encode_prepared_peer_blind_relay_request(&request)
            .map_err(BlindRelayRequestPreparationError::Local)?;
        Ok((request, context))
    })
    .await
    .map_err(|_| {
        warn!("[CHAT_PEER] Outbound blind relay preparation worker failed closed");
        BlindRelayRequestPreparationError::Local(BlindRelayRequestPreparationFailure::Unavailable)
    })?
}

/// Serializes an already-built blind-relay request without running I/O.
///
/// [ANONYMOUS-MAILBOX-SOURCE 2026-09-03 by Codex] The default-off anonymous
/// mailbox source journal needs the immutable JSON body before it can arm any
/// transport. Callers must keep this bounded CPU-only operation outside async
/// I/O workers; the existing async wrapper remains the normal public path.
pub(crate) fn prepare_exact_peer_blind_relay_http_request(
    request: &PeerBlindRelayRequest,
) -> Result<PreparedPeerBlindRelayHttpRequest, BlindRelayRequestPreparationFailure> {
    encode_prepared_peer_blind_relay_request(request)
}

fn encode_prepared_peer_blind_relay_request(
    request: &PeerBlindRelayRequest,
) -> Result<PreparedPeerBlindRelayHttpRequest, BlindRelayRequestPreparationFailure> {
    let route_id = request.envelope.route_id;
    let body =
        serde_json::to_vec(request).map_err(|_| BlindRelayRequestPreparationFailure::Encoding)?;
    if body.len() > PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES {
        return Err(BlindRelayRequestPreparationFailure::BodyTooLarge);
    }
    Ok(PreparedPeerBlindRelayHttpRequest {
        body: Bytes::from(body),
        route_id,
    })
}

pub(super) async fn prepare_blind_relay_forward_request(
    request: PeerBlindRelayRequest,
) -> Result<PreparedBlindRelayForwardRequest, BlindRelayError> {
    run_blind_relay_crypto(move || {
        let http =
            encode_prepared_peer_blind_relay_request(&request).map_err(|error| match error {
                BlindRelayRequestPreparationFailure::BodyTooLarge => {
                    BlindRelayError::EnvelopeTooLarge
                }
                BlindRelayRequestPreparationFailure::Backpressure
                | BlindRelayRequestPreparationFailure::Unavailable => BlindRelayError::Backpressure,
                BlindRelayRequestPreparationFailure::Encoding => BlindRelayError::ForwardFailed,
            })?;
        Ok(PreparedBlindRelayForwardRequest {
            request: Arc::new(request),
            http,
        })
    })
    .await
}

/// Verifies one terminal delivery receipt behind bounded CPU admission.
pub(crate) async fn verify_blind_relay_delivery_receipt(
    receipt: Option<BlindRelayDeliveryReceipt>,
    expected_route_id: [u8; 16],
    expected_payload_commitment: [u8; 32],
    expected_terminal_node_id: [u8; 32],
    observed_at: u64,
) -> Result<BlindRelayDeliveryReceipt, BlindRelayDeliveryReceiptVerificationFailure> {
    // Missing evidence is a peer/protocol outcome and must not be hidden by
    // unrelated local saturation.
    let receipt = receipt.ok_or(BlindRelayDeliveryReceiptVerificationFailure::Missing)?;
    // [BLIND-RECEIPT-FAIR-COMPLETION 2026-08-31 by Codex] The route has
    // already been exposed and cannot safely fall back to a new surface.
    // Await the fair semaphore instead of dropping authoritative evidence on
    // transient ingress pressure. Outbound fanout bounds these waiters.
    let permit = blind_relay_crypto_admission()
        .acquire_owned()
        .await
        .map_err(|_| BlindRelayDeliveryReceiptVerificationFailure::Unavailable)?;
    tokio::task::spawn_blocking(move || {
        // [OUTBOUND-BLIND-RECEIPT-VERIFICATION 2026-08-31 by Codex] Hold the
        // permit until signature verification really stops after cancellation.
        let _permits = BlindRelayCryptoPermits::total_only(permit);
        if blind_relay_delivery_receipt_is_valid(
            &receipt,
            &expected_route_id,
            &expected_payload_commitment,
            &expected_terminal_node_id,
            observed_at,
        ) {
            Ok(receipt)
        } else {
            Err(BlindRelayDeliveryReceiptVerificationFailure::Invalid)
        }
    })
    .await
    .map_err(|_| {
        warn!("[CHAT_PEER] Outbound blind receipt verification worker failed closed");
        BlindRelayDeliveryReceiptVerificationFailure::Unavailable
    })?
}

/// Pure receipt contract shared by the bounded worker and focused unit tests.
#[must_use]
pub(crate) fn blind_relay_delivery_receipt_is_valid(
    receipt: &BlindRelayDeliveryReceipt,
    expected_route_id: &[u8; 16],
    expected_payload_commitment: &[u8; 32],
    expected_terminal_node_id: &[u8; 32],
    observed_at: u64,
) -> bool {
    receipt.version == BLIND_RELAY_PURPOSE_BOUND_DELIVERY_RECEIPT_VERSION
        && receipt.delivered_at
            <= observed_at.saturating_add(BLIND_RELAY_DELIVERY_RECEIPT_MAX_FUTURE_SKEW_SECS)
        && observed_at.saturating_sub(receipt.delivered_at)
            <= BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS
        && receipt
            .verify_expected(
                expected_route_id,
                expected_payload_commitment,
                expected_terminal_node_id,
            )
            .is_ok()
}

fn encode_prepared_peer_chat_relay_request<T: Serialize>(
    request: &T,
) -> Result<PreparedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    let body =
        serde_json::to_vec(request).map_err(|_| DirectRelayRequestPreparationFailure::Encoding)?;
    if body.len() > PEER_CHAT_REQUEST_BODY_MAX_BYTES {
        return Err(DirectRelayRequestPreparationFailure::BodyTooLarge);
    }
    Ok(PreparedPeerChatRelayHttpRequest {
        body: Bytes::from(body),
    })
}

fn encode_prepared_authenticated_peer_chat_relay_request<T: Serialize>(
    request: &T,
    request_commitment: [u8; 32],
) -> Result<PreparedAuthenticatedPeerChatRelayHttpRequest, DirectRelayRequestPreparationFailure> {
    Ok(PreparedAuthenticatedPeerChatRelayHttpRequest {
        request: encode_prepared_peer_chat_relay_request(request)?,
        request_commitment,
    })
}

/// Verifies a signed direct-custody receipt outside the async I/O runtime.
pub(crate) async fn verify_peer_chat_relay_receipt(
    receipt: PeerChatRelayReceiptV2,
    expected_request_commitment: [u8; 32],
    expected_node_id: [u8; 32],
    observed_at: u64,
) -> Result<(), DirectRelayReceiptVerificationFailure> {
    // [DIRECT-RECEIPT-FAIR-COMPLETION 2026-08-31 by Codex] Durable custody may
    // already exist at the selected target. Await bounded fair completion
    // instead of repeating network I/O merely because local CPU is busy.
    complete_direct_relay_crypto(move || {
        receipt.verify_expected_commitment(
            &expected_request_commitment,
            &expected_node_id,
            observed_at,
        )
    })
    .await
    .map_err(|DirectRelayCryptoFailure::Unavailable| {
        DirectRelayReceiptVerificationFailure::Unavailable
    })?
    .map_err(DirectRelayReceiptVerificationFailure::Invalid)
}

async fn prepare_direct_peer_relay_request<T, F>(
    work: F,
) -> Result<T, DirectRelayRequestPreparationFailure>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, DirectRelayRequestPreparationFailure> + Send + 'static,
{
    // [OUTBOUND-DIRECT-REQUEST-PREPARATION 2026-08-31 by Codex] Outbound
    // fallback is optional and always has local durable custody behind it.
    // Fail fast instead of queueing unbounded signature work under fanout.
    let permit = direct_relay_cpu_admission()
        .try_acquire_owned()
        .map_err(|_| DirectRelayRequestPreparationFailure::Backpressure)?;
    execute_direct_relay_crypto(permit, work).await.map_err(
        |DirectRelayCryptoFailure::Unavailable| DirectRelayRequestPreparationFailure::Unavailable,
    )?
}

/// Projects a monotonic request duration onto the caller's Unix timestamp.
///
/// [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] Tests and recovery
/// probes inject a stable Unix base. `Instant` supplies elapsed time without
/// depending on wall-clock jumps, while saturation keeps failure handling
/// total even near the integer boundary.
pub(super) fn blind_relay_response_observed_at(started_at: u64, started: &Instant) -> u64 {
    started_at.saturating_add(started.elapsed().as_secs())
}

/// Replaceable capabilities composed for one forwarding operation.
struct BlindRelayForwardComponents<'a> {
    retry_policy: Arc<dyn BlindRelayRetryPolicy>,
    response_policy: Arc<dyn BlindRelayResponsePolicy>,
    transport: &'a dyn BlindRelayTransport,
    observer: &'a dyn BlindRelayForwardObserver,
}

pub(super) async fn forward_blind_relay_with_retry(
    state: &ChatPeerState,
    url: &str,
    descriptor: &SignedNodeDescriptor,
    request: PreparedBlindRelayForwardRequest,
    now: u64,
) -> Result<BlindRelayForwardOutcome, BlindRelayError> {
    let transport = ReqwestBlindRelayTransport::new(Arc::clone(&state.http_client));
    let observer = PeerStoreBlindRelayForwardObserver::new(state.peer_store.as_ref());
    forward_blind_relay_with_components(
        url,
        descriptor,
        request,
        now,
        BlindRelayForwardComponents {
            retry_policy: Arc::new(BlindRelayRetryDomain::default()),
            response_policy: Arc::new(BlindRelayResponseDomain),
            transport: &transport,
            observer: &observer,
        },
    )
    .await
}

/// [ROUTE-FAILURE-SURFACE-BINDING 2026-08-11 by Codex] Keeps the exact signed
/// descriptor that selected `url` through retries, so delayed observations can
/// update health only when the selected route surface remains current.
async fn forward_blind_relay_with_components(
    url: &str,
    descriptor: &SignedNodeDescriptor,
    request: PreparedBlindRelayForwardRequest,
    now: u64,
    components: BlindRelayForwardComponents<'_>,
) -> Result<BlindRelayForwardOutcome, BlindRelayError> {
    let PreparedBlindRelayForwardRequest { request, http } = request;
    let next_hop = descriptor.node_id();
    let failure_receipt_required = blind_relay_failure_receipt_required(descriptor);
    let success_receipt_required = blind_relay_success_receipt_required(descriptor);
    let source_sealed_terminal_proof_allowed =
        onion_source_sealed_terminal_proof_allowed(descriptor);
    let large_pull_response_allowed = onion_large_pull_response_allowed(descriptor);
    let request_started_at = Instant::now();
    for attempt in 1..=components.retry_policy.max_attempts().get() {
        let retry_context = blind_relay_retry_context(request.as_ref(), next_hop, attempt)?;
        let transport_outcome = components.transport.send(url, http.body()).await;
        let observed_at = blind_relay_response_observed_at(now, &request_started_at);
        let request_for_validation = Arc::clone(&request);
        let response_policy = Arc::clone(&components.response_policy);
        let retry_policy = Arc::clone(&components.retry_policy);
        let decision = complete_blind_relay_crypto(move || {
            // [BLIND-RESPONSE-CRYPTO-COMPLETION 2026-08-30 by Codex] Response
            // decoding is already byte-bounded by the transport. Keep policy
            // evaluation pure while moving receipt verification and payload
            // commitment hashing away from the asynchronous I/O runtime.
            Ok(response_policy.evaluate(
                transport_outcome,
                BlindRelayResponseContext {
                    request: request_for_validation.as_ref(),
                    next_hop,
                    observed_at,
                    failure_receipt_required,
                    success_receipt_required,
                    source_sealed_terminal_proof_allowed,
                    large_pull_response_allowed,
                    retry_context,
                    retry_policy: retry_policy.as_ref(),
                },
            ))
        })
        .await?;
        match decision {
            BlindRelayResponseDecision::Accepted(response) => {
                return Ok(complete_blind_relay_forward(
                    components.observer,
                    *response,
                    observed_at,
                    attempt,
                ))
            }
            BlindRelayResponseDecision::PeerDeclaredFailure {
                failure,
                status,
                receipt_authenticated,
            } => {
                // [DOWNSTREAM-FAILURE-ATTRIBUTION 2026-08-11 by Codex] A
                // bounded error ACK cannot identify which deeper hop failed.
                debug!(
                    attempt,
                    status = %status,
                    receipt_authenticated,
                    "[BLIND_RELAY] Peer-declared downstream failure left unattributed"
                );
                let error = BlindRelayError::from(failure);
                components
                    .observer
                    .rejected(observed_at, error.reason_bucket());
                return Err(error);
            }
            BlindRelayResponseDecision::RetryAfter {
                delay,
                reason,
                source,
            } => {
                components.observer.retry_attempted(observed_at, &reason);
                log_blind_relay_retry(attempt, source, &reason);
                sleep(delay).await;
            }
            BlindRelayResponseDecision::Reject(failure) => {
                let error = BlindRelayError::from(failure);
                components
                    .observer
                    .route_failed(descriptor, observed_at, error.reason_bucket());
                return Err(error);
            }
            BlindRelayResponseDecision::InvalidResponse {
                kind,
                diagnostic,
                health_reason,
                counts_as_retry_exhaustion,
            } => {
                log_invalid_blind_relay_response(attempt, kind, diagnostic);
                if counts_as_retry_exhaustion && attempt > 1 {
                    components
                        .observer
                        .retry_exhausted(observed_at, attempt, health_reason);
                }
                components
                    .observer
                    .route_failed(descriptor, observed_at, health_reason);
                return Err(BlindRelayError::ForwardFailed);
            }
            BlindRelayResponseDecision::Exhausted { reason, source } => {
                log_blind_relay_exhausted(attempt, source, &reason);
                if attempt > 1 {
                    components
                        .observer
                        .retry_exhausted(observed_at, attempt, &reason);
                }
                components
                    .observer
                    .route_failed(descriptor, observed_at, &reason);
                return Err(BlindRelayError::ForwardFailed);
            }
        }
    }

    Err(BlindRelayError::ForwardFailed)
}

fn blind_relay_retry_context(
    request: &PeerBlindRelayRequest,
    next_hop: [u8; 32],
    attempt: usize,
) -> Result<BlindRelayRetryContext, BlindRelayError> {
    BlindRelayRetryContext::new(request.envelope.route_id, next_hop, attempt)
        .ok_or(BlindRelayError::ForwardFailed)
}

fn blind_relay_failure_receipt_required(descriptor: &SignedNodeDescriptor) -> bool {
    // [FAILURE-RECEIPT-ANTI-DOWNGRADE 2026-08-11 by Codex] Negotiate from the
    // exact signed descriptor that selected this URL. An attacker cannot strip
    // this token without invalidating the descriptor signature.
    descriptor
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::BlindRelayFailureReceiptV1)
}

fn blind_relay_success_receipt_required(descriptor: &SignedNodeDescriptor) -> bool {
    descriptor
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::BlindRelaySuccessReceiptV1)
}

fn onion_source_sealed_terminal_proof_allowed(descriptor: &SignedNodeDescriptor) -> bool {
    // Both tokens are required because an opaque-only response is safe to
    // accept only when the immediate peer authenticates the exact bytes it
    // returned. Signed descriptor negotiation prevents downgrade by gossip or
    // an endpoint that does not own the advertised identity.
    blind_relay_success_receipt_required(descriptor)
        && descriptor
            .descriptor
            .advertises_protocol_feature(NodeProtocolFeature::OnionSourceSealedTerminalProofV1)
}

fn onion_large_pull_response_allowed(descriptor: &SignedNodeDescriptor) -> bool {
    // [BLIND-VAULT-LARGE-PULL-VALIDATION 2026-08-30 by Codex] A bounded large
    // ACK is accepted only from the exact signed descriptor selected for this
    // hop. The response remains opaque, so no relay learns the workload type.
    descriptor
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultLargePullV1)
}

fn complete_blind_relay_forward(
    observer: &dyn BlindRelayForwardObserver,
    response: PeerBlindRelayResponse,
    observed_at: u64,
    attempt: usize,
) -> BlindRelayForwardOutcome {
    if attempt > 1 {
        observer.retry_succeeded(observed_at, attempt);
    }
    BlindRelayForwardOutcome {
        response,
        observed_at,
    }
}

fn log_blind_relay_retry(attempt: usize, source: BlindRelayResponseSource, reason: &str) {
    match source {
        BlindRelayResponseSource::HttpStatus(status) => debug!(
            attempt,
            status = %status,
            "[BLIND_RELAY] Next-hop returned retryable status"
        ),
        BlindRelayResponseSource::Transport => debug!(
            attempt,
            reason, "[BLIND_RELAY] Next-hop forward failed; retrying"
        ),
    }
}

fn log_blind_relay_exhausted(attempt: usize, source: BlindRelayResponseSource, reason: &str) {
    match source {
        BlindRelayResponseSource::HttpStatus(status) => debug!(
            attempt,
            status = %status,
            "[BLIND_RELAY] Next-hop returned non-success"
        ),
        BlindRelayResponseSource::Transport => {
            debug!(attempt, reason, "[BLIND_RELAY] Next-hop forward failed")
        }
    }
}

fn log_invalid_blind_relay_response(
    attempt: usize,
    kind: BlindRelayInvalidResponseKind,
    diagnostic: &'static str,
) {
    match kind {
        BlindRelayInvalidResponseKind::SuccessAck => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop ACK invalid"
        ),
        BlindRelayInvalidResponseKind::SuccessReceipt => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop hop-local success receipt verification failed"
        ),
        BlindRelayInvalidResponseKind::DeliveryReceipt => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop delivery receipt verification failed"
        ),
        BlindRelayInvalidResponseKind::OpaqueTerminalResponse => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop opaque terminal response validation failed"
        ),
        BlindRelayInvalidResponseKind::FailureReceipt => debug!(
            attempt,
            reason = diagnostic,
            "[BLIND_RELAY] Next-hop failure receipt verification failed"
        ),
    }
}

pub(super) fn blind_peer_relay_url(endpoint: &str) -> Option<String> {
    // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] A next-hop descriptor is
    // permissionless input. Its signature cannot authorize localhost, private
    // networks, metadata services, DNS rebinding, or URL-controlled paths.
    if !peer_endpoint_is_public_ip(endpoint) {
        #[cfg(not(test))]
        return None;
        #[cfg(test)]
        if !crate::api::peer_endpoint_is_loopback_ip(endpoint) {
            return None;
        }
    }
    canonical_peer_http_url(endpoint, "/api/chat/peer/blind-relay")
        .ok()
        .map(|url| url.to_string())
}
