// [ARCH-SPLIT 2026-10-02]
// Opaque blind relay, onion terminal store, and middle-hop forward.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn blind_relay_authenticated_request_commitment(
    request: &PeerBlindRelayRequest,
) -> Result<[u8; 32], bincode::Error> {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] Bincode is already the
    // canonical internal encoding used by this protocol crate. Hashing the
    // complete request prevents route-id reuse from swapping optional onward
    // routing material while keeping all durable keys node-secret HMACs.
    // [STREAMING-REPLAY-COMMITMENT 2026-08-30 by Codex] `serialized_size`
    // and `serialize_into` use the same bincode 1.x canonical options as
    // `serialize`. Hashing through `Write` preserves the existing commitment
    // exactly without allocating a second request-sized byte vector.
    let encoded_len = bincode::serialized_size(request)?;
    let mut hasher = Sha256::new();
    hasher.update(BLIND_RELAY_AUTHENTICATED_REQUEST_COMMITMENT_DOMAIN);
    hasher.update(encoded_len.to_be_bytes());
    bincode::serialize_into(Sha256CommitmentWriter(&mut hasher), request)?;
    Ok(hasher.finalize().into())
}

/// Applies blind-relay backpressure before Axum reads a JSON body.
pub(super) async fn peer_blind_relay_request_gate(
    State(state): State<ChatPeerState>,
    request: Request,
    next: Next,
) -> Response {
    // [BLIND-RELAY-BODY-ADMISSION-ORDER 2026-08-24 by Codex] Reject a declared
    // or exactly-known oversized body before any service-availability signal.
    // Unknown-length streams are not read here: the existing DefaultBodyLimit
    // remains authoritative when the JSON extractor consumes an admitted body.
    let body_limit = u64::try_from(PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES).unwrap_or(u64::MAX);
    let declared_length = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let exact_length = request.body().size_hint().exact();
    if declared_length.is_some_and(|length| length > body_limit)
        || exact_length.is_some_and(|length| length > body_limit)
    {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }

    // [DURABLE-BLIND-RELAY-ADMISSION 2026-08-24 by Codex] A public relay must
    // never accept work whose at-most-once evidence disappears on restart.
    // Reject before JSON/body parsing, signature work, route mutation, or any
    // ciphertext side effect. The process-local cache remains only an internal
    // compatibility primitive for focused unit tests and non-HTTP helpers.
    let Some(relay) = state.chat_relay.as_ref() else {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "replay_protection_unavailable");
        return rejected_blind_relay_response_with_status(
            StatusCode::SERVICE_UNAVAILABLE,
            "replay_protection_unavailable",
        );
    };

    // [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] Permissionless node
    // identities are cheap to rotate, so the verified previous-hop bucket
    // cannot protect parser and process capacity by itself. Count only one
    // aggregate process window before body parsing; never create source-IP,
    // user, receiver, route, endpoint, or ciphertext-derived buckets here.
    let requests_per_minute = relay.config().peer_relay_requests_per_minute;
    let admitted = state
        .blind_relay_abuse_guard
        .admit_global(Instant::now(), requests_per_minute);
    if !admitted {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "rate_limited");
        return rejected_blind_relay_response("rate_limited");
    }

    let Some(_in_flight) = InFlightRequestGuard::try_acquire(
        &state.blind_relay_in_flight,
        MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS,
    ) else {
        state
            .peer_store
            .record_blind_relay_rejected(now_secs(), "backpressure");
        return rejected_blind_relay_response("backpressure");
    };

    next.run(request).await
}

pub(super) fn rejected_blind_relay_response(reason: &'static str) -> Response {
    rejected_blind_relay_response_with_status(StatusCode::TOO_MANY_REQUESTS, reason)
}

pub(super) fn rejected_blind_relay_response_with_status(
    status: StatusCode,
    reason: &'static str,
) -> Response {
    (
        status,
        Json(PeerBlindRelayResponse {
            accepted: false,
            terminal: false,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some(reason.to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        }),
    )
        .into_response()
}

pub(super) async fn peer_blind_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerBlindRelayRequest>,
) -> impl IntoResponse {
    let failure_route_id = request.envelope.route_id;
    let node_identity = Arc::clone(&state.node_identity);
    let authenticated = match authenticate_peer_blind_relay_request(request).await {
        Ok(authenticated) => authenticated,
        Err(error) => {
            // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] The claimed
            // previous hop has no attribution authority before verification.
            // Record only aggregate health and never sign an oracle response.
            state
                .peer_store
                .record_blind_relay_rejected(now_secs(), error.reason_bucket());
            return blind_relay_failure_response(error, failure_route_id, None, node_identity)
                .await;
        }
    };
    let failure_request_commitment = authenticated.failure_request_commitment;
    match process_authenticated_peer_blind_relay(state, authenticated).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => {
            blind_relay_failure_response(
                error,
                failure_route_id,
                Some(failure_request_commitment),
                node_identity,
            )
            .await
        }
    }
}

/// Builds the stable blind-relay failure shape and signs it only when the
/// caller supplies a commitment produced by the authenticated request type.
///
/// [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] `None` is a security
/// boundary, not legacy absence: pre-authentication failures must stay unsigned.
pub(super) async fn blind_relay_failure_response(
    error: BlindRelayError,
    route_id: [u8; 16],
    authenticated_request_commitment: Option<[u8; 32]>,
    node_identity: Arc<IdentityKeyPair>,
) -> Response {
    let status = error.status_code();
    let reason = error.reason_bucket();
    let Some(request_commitment) = authenticated_request_commitment else {
        return build_blind_relay_failure_response(status, reason, None);
    };
    let failed_at = now_secs();
    let failure_receipt = match complete_blind_relay_crypto(move || {
        // [BLIND-FAILURE-SIGNING-COMPLETION 2026-08-30 by Codex] Possession
        // of the precomputed commitment proves the request crossed the private
        // authenticated boundary. No envelope or peer-controlled bytes enter
        // this signing worker.
        Ok(BlindRelayFailureReceipt::failed(
            route_id,
            request_commitment,
            reason,
            failed_at,
            node_identity.as_ref(),
        ))
    })
    .await
    {
        Ok(receipt) => receipt,
        Err(_) => {
            // An unsigned declared protocol failure would look like a signed
            // feature downgrade. A bare retryable 429 carries no blame and
            // lets the exact durable request recover on its next attempt.
            return build_blind_relay_failure_response(
                StatusCode::TOO_MANY_REQUESTS,
                BlindRelayError::Backpressure.reason_bucket(),
                None,
            );
        }
    };
    build_blind_relay_failure_response(status, reason, Some(failure_receipt))
}

pub(super) fn build_blind_relay_failure_response(
    status: StatusCode,
    reason: &'static str,
    failure_receipt: Option<BlindRelayFailureReceipt>,
) -> Response {
    (
        status,
        Json(PeerBlindRelayResponse {
            accepted: false,
            terminal: false,
            forwarded: false,
            ttl_remaining: 0,
            reason: Some(reason.to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt,
            opaque_terminal_response_b64: None,
        }),
    )
        .into_response()
}

/// Concurrent blind-relay crypto workers for a host with `hardware_threads`:
/// half the threads rounded up, never below
/// `MIN_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT` and never above
/// `MAX_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT`.
///
/// A pure function so the policy can be tested for any core count; the
/// runtime value below feeds it the detected parallelism.
pub(super) fn blind_relay_capacity_for_threads(hardware_threads: usize) -> usize {
    (hardware_threads.saturating_add(1) / 2)
        .max(MIN_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT)
        .min(MAX_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT)
}

pub(super) fn blind_relay_crypto_capacity() -> usize {
    // [CI-DETERMINISTIC-CAPACITY 2026-10-09 by Claude] Unit tests share these
    // process-global semaphores and run in parallel, so they use the fixed
    // ceiling instead of the host's core count.
    if cfg!(test) {
        return MAX_BLIND_RELAY_CRYPTO_OPERATIONS_IN_FLIGHT;
    }
    blind_relay_capacity_for_threads(
        std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1),
    )
}

pub(super) fn blind_relay_ingress_crypto_capacity(total_capacity: usize) -> usize {
    // A one-worker host cannot reserve a permit without disabling ingress.
    // Multi-worker hosts retain one total permit outside the ingress quota.
    // [BLIND-RELAY-STABLE-CAPACITY 2026-08-31 by Codex] Keep this runtime-only
    // policy on stable Rust; `Ord::max` is not a stable const-trait operation.
    if total_capacity > 1 {
        total_capacity - 1
    } else {
        1
    }
}

pub(super) fn blind_relay_crypto_admission() -> Arc<Semaphore> {
    Arc::clone(BLIND_RELAY_CRYPTO_ADMISSION.get_or_init(|| {
        // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Reserve roughly
        // half the host for the rest of the node and reject excess work before
        // it enters Tokio's blocking queue.
        Arc::new(Semaphore::new(blind_relay_crypto_capacity()))
    }))
}

pub(super) fn blind_relay_ingress_crypto_admission() -> Arc<Semaphore> {
    Arc::clone(BLIND_RELAY_INGRESS_CRYPTO_ADMISSION.get_or_init(|| {
        Arc::new(Semaphore::new(blind_relay_ingress_crypto_capacity(
            blind_relay_crypto_capacity(),
        )))
    }))
}

pub(super) fn try_acquire_blind_relay_ingress_crypto(
) -> Result<BlindRelayCryptoPermits, BlindRelayError> {
    let ingress = blind_relay_ingress_crypto_admission()
        .try_acquire_owned()
        .map_err(|_| BlindRelayError::Backpressure)?;
    let total = blind_relay_crypto_admission()
        .try_acquire_owned()
        .map_err(|_| BlindRelayError::Backpressure)?;
    Ok(BlindRelayCryptoPermits {
        _total: total,
        _ingress: Some(ingress),
    })
}

pub(super) async fn acquire_blind_relay_ingress_crypto(
) -> Result<BlindRelayCryptoPermits, BlindRelayError> {
    let ingress = blind_relay_ingress_crypto_admission()
        .acquire_owned()
        .await
        .map_err(|_| BlindRelayError::Backpressure)?;
    let total = blind_relay_crypto_admission()
        .acquire_owned()
        .await
        .map_err(|_| BlindRelayError::Backpressure)?;
    Ok(BlindRelayCryptoPermits {
        _total: total,
        _ingress: Some(ingress),
    })
}

/// Executes one bounded blind-relay cryptographic operation off the async I/O
/// runtime. Work must remain pure with respect to network and durable storage.
pub(super) async fn run_blind_relay_crypto<T, F>(work: F) -> Result<T, BlindRelayError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BlindRelayError> + Send + 'static,
{
    let permits = try_acquire_blind_relay_ingress_crypto()?;
    execute_blind_relay_crypto(permits, work).await
}

/// Completes pure cryptographic work after an external route effect is armed.
///
/// Unlike preflight admission, this waits fairly for bounded CPU capacity so a
/// successfully returned ACK is not discarded merely because new ingress
/// verification arrived first. Completion still consumes the ingress quota,
/// preserving the outbound progress reservation. The outer HTTP in-flight
/// gate bounds waiters.
pub(super) async fn complete_blind_relay_crypto<T, F>(work: F) -> Result<T, BlindRelayError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BlindRelayError> + Send + 'static,
{
    let permits = acquire_blind_relay_ingress_crypto().await?;
    execute_blind_relay_crypto(permits, work).await
}

pub(super) async fn execute_blind_relay_crypto<T, F>(
    permits: BlindRelayCryptoPermits,
    work: F,
) -> Result<T, BlindRelayError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, BlindRelayError> + Send + 'static,
{
    match tokio::task::spawn_blocking(move || {
        // [BLIND-RELAY-CRYPTO-DOMAIN 2026-08-30 by Codex] Keep the owned
        // permit in the worker after request cancellation. This helper must
        // never contain I/O or durable effects, so retries remain safe.
        let _permits = permits;
        work()
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            warn!("[CHAT_PEER] Blind relay crypto worker failed closed");
            Err(BlindRelayError::Backpressure)
        }
    }
}

pub(super) fn blind_vault_terminal_admission() -> Arc<Semaphore> {
    Arc::clone(BLIND_VAULT_TERMINAL_ADMISSION.get_or_init(|| {
        Arc::new(Semaphore::new(
            MAX_BLIND_VAULT_TERMINAL_OPERATIONS_IN_FLIGHT,
        ))
    }))
}

pub(super) async fn authenticate_peer_blind_relay_request(
    request: PeerBlindRelayRequest,
) -> Result<AuthenticatedPeerBlindRelayRequest, BlindRelayError> {
    let permits = try_acquire_blind_relay_ingress_crypto()?;
    authenticate_peer_blind_relay_request_with_permits(permits, request).await
}

pub(super) async fn authenticate_peer_blind_relay_request_with_admission(
    admission: Arc<Semaphore>,
    request: PeerBlindRelayRequest,
) -> Result<AuthenticatedPeerBlindRelayRequest, BlindRelayError> {
    // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Acquire before
    // `spawn_blocking`, then move the owned permit into the worker. This makes
    // saturation fail immediately without queue growth and prevents a
    // cancelled HTTP future from releasing capacity before CPU work stops.
    let permit = admission
        .try_acquire_owned()
        .map_err(|_| BlindRelayError::Backpressure)?;
    authenticate_peer_blind_relay_request_with_permits(
        BlindRelayCryptoPermits::total_only(permit),
        request,
    )
    .await
}

pub(super) async fn authenticate_peer_blind_relay_request_with_permits(
    permits: BlindRelayCryptoPermits,
    request: PeerBlindRelayRequest,
) -> Result<AuthenticatedPeerBlindRelayRequest, BlindRelayError> {
    match tokio::task::spawn_blocking(move || {
        let _permits = permits;
        authenticate_blind_relay_envelope(&request.envelope, &request.previous_hop_node_id)?;
        if let Some(onward_envelope) = request.onward_envelope.as_ref() {
            // [SIGNED-ONWARD-ENVELOPE 2026-08-24 by Codex] The forwarding
            // sender already re-signs this optional legacy frame. Verify that
            // exact previous-hop signature before the complete replay
            // commitment or any route state can trust it.
            authenticate_blind_relay_envelope(onward_envelope, &request.previous_hop_node_id)?;
        }
        let failure_request_commitment =
            BlindRelayFailureReceipt::request_commitment(&request.envelope);
        let request_commitment = blind_relay_authenticated_request_commitment(&request)
            .map_err(|_| BlindRelayError::ReplayProtectionUnavailable)?;
        Ok(AuthenticatedPeerBlindRelayRequest {
            request,
            failure_request_commitment,
            request_commitment,
        })
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            // Join failures are local runtime faults. Never expose a panic or
            // scheduler detail through the privacy protocol response.
            warn!("[CHAT_PEER] Blind relay verification worker failed closed");
            Err(BlindRelayError::Backpressure)
        }
    }
}

pub(super) fn build_forwarded_onion_envelope(
    envelope: &BlindRelayEnvelope,
    next_hop: [u8; 32],
    inner: Vec<u8>,
    node_identity: &IdentityKeyPair,
) -> BlindRelayEnvelope {
    build_forwarded_onion_envelope_from_seed(
        BlindRelayForwardSeed::from(envelope),
        next_hop,
        inner,
        node_identity,
    )
}

pub(super) fn build_forwarded_onion_envelope_from_seed(
    seed: BlindRelayForwardSeed,
    next_hop: [u8; 32],
    inner: Vec<u8>,
    node_identity: &IdentityKeyPair,
) -> BlindRelayEnvelope {
    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Every field derives
    // from authenticated ingress state. Ed25519 signing is deterministic, so
    // an exact restart retry generates the same downstream request commitment.
    BlindRelayEnvelope {
        route_id: seed.route_id,
        next_hop,
        ttl: seed.ttl.saturating_sub(1),
        encrypted_blob: inner,
        timestamp: seed.timestamp,
        signature: [0u8; 64],
    }
    .sign_with(node_identity)
}

/// Attaches one immediate-hop success proof to an already accepted response.
///
/// [BLIND-RELAY-SUCCESS-RECEIPT 2026-08-29 by Codex] This is the only helper
/// allowed to sign outbound success ACKs. It binds the exact request envelope,
/// response shape, TTL, legacy delivery evidence, and opaque response while
/// ensuring a relay never propagates a deeper hop's success signature.
pub(super) async fn attach_blind_relay_success_receipt(
    envelope: Arc<BlindRelayEnvelope>,
    response: PeerBlindRelayResponse,
    accepted_at: u64,
    responder: Arc<IdentityKeyPair>,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    complete_blind_relay_crypto(move || {
        // [BLIND-SUCCESS-SIGNING-COMPLETION 2026-08-30 by Codex] The request
        // and possibly large opaque response move into the worker by ownership;
        // `Arc` avoids a ciphertext clone while durable completion waits.
        sign_blind_relay_success_receipt(
            envelope.as_ref(),
            response,
            accepted_at,
            responder.as_ref(),
        )
    })
    .await
}

pub(super) async fn attach_blind_relay_terminal_success_receipts(
    envelope: Arc<BlindRelayEnvelope>,
    mut response: PeerBlindRelayResponse,
    proof: TerminalDeliveryProofInput,
    accepted_at: u64,
    responder: Arc<IdentityKeyPair>,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    complete_blind_relay_crypto(move || {
        // [BLIND-TERMINAL-PROOF-COMPLETION 2026-08-30 by Codex] One worker
        // commits the accepted payload, signs terminal evidence, and binds that
        // exact evidence into the immediate-hop ACK before durable completion.
        if response.delivery_receipt.is_some() {
            return Err(BlindRelayError::ForwardFailed);
        }
        response.delivery_receipt = Some(BlindRelayDeliveryReceipt::accepted_for_purpose(
            envelope.route_id,
            &proof.payload,
            proof.purpose,
            accepted_at,
            responder.as_ref(),
        ));
        sign_blind_relay_success_receipt(
            envelope.as_ref(),
            response,
            accepted_at,
            responder.as_ref(),
        )
    })
    .await
}

pub(super) fn sign_blind_relay_success_receipt(
    envelope: &BlindRelayEnvelope,
    mut response: PeerBlindRelayResponse,
    accepted_at: u64,
    responder: &IdentityKeyPair,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    if !response.accepted || response.failure_receipt.is_some() {
        return Err(BlindRelayError::ForwardFailed);
    }
    let opaque_response = response
        .opaque_terminal_response_b64
        .as_deref()
        .map(str::as_bytes);
    let receipt = match (response.terminal, response.forwarded) {
        (true, false) => BlindRelaySuccessReceipt::terminal(
            envelope,
            response.ttl_remaining,
            response.reason.as_deref(),
            response.delivery_receipt.as_ref(),
            opaque_response,
            accepted_at,
            responder,
        ),
        (false, true) => BlindRelaySuccessReceipt::forwarded(
            envelope,
            response.ttl_remaining,
            response.reason.as_deref(),
            response.delivery_receipt.as_ref(),
            opaque_response,
            accepted_at,
            responder,
        ),
        _ => return Err(BlindRelayError::ForwardFailed),
    };
    response.success_receipt = Some(receipt);
    Ok(response)
}

#[cfg(test)]
pub(super) async fn process_peer_blind_relay(
    state: ChatPeerState,
    request: PeerBlindRelayRequest,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    // [BLIND-RELAY-TEST-ADMISSION-ISOLATION 2026-08-24 by Codex] Focused
    // process tests must not race one another for the production-global CPU
    // semaphore. The dedicated admission tests still exercise that runtime
    // boundary directly; this helper keeps each unrelated route test bounded
    // to one verification worker without introducing suite-order flakiness.
    let authenticated =
        authenticate_peer_blind_relay_request_with_admission(Arc::new(Semaphore::new(1)), request)
            .await
            .map_err(|error| {
                state
                    .peer_store
                    .record_blind_relay_rejected(now_secs(), error.reason_bucket());
                error
            })?;
    process_authenticated_peer_blind_relay(state, authenticated).await
}

pub(super) async fn process_authenticated_peer_blind_relay(
    state: ChatPeerState,
    authenticated: AuthenticatedPeerBlindRelayRequest,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] Reaching route state
    // requires possession of the private authenticated request capability.
    let now = now_secs();
    let route_started_at = Instant::now();
    let request_commitment = authenticated.request_commitment;
    let request = authenticated.request;
    let previous_hop_node_id = request.previous_hop_node_id;
    let onward_descriptor_hint = request.onward_descriptor_hint;
    let envelope = request.envelope;

    check_blind_relay_previous_hop_allowed(&state, previous_hop_node_id, now)?;

    validate_blind_relay_metadata(&envelope, now).map_err(|error| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, error.reason_bucket());
        error
    })?;

    let self_node_id = state.node_identity.public_key_bytes();
    if previous_hop_node_id == self_node_id {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "self_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if envelope.next_hop == self_node_id {
        // Onion routing v1: if the opaque blob is an onion layer addressed to
        // this node, peel exactly one layer and either deliver locally
        // (terminal) or forward the inner layer to the revealed next hop. Legacy
        // opaque blobs (no onion magic) fall through to the existing behavior.
        if is_onion_blob(&envelope.encrypted_blob) {
            return process_onion_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                request_commitment,
                now,
                &route_started_at,
            )
            .await;
        }
        if let Some(onward_envelope) = request.onward_envelope {
            return process_onion_middle_blind_relay(
                state,
                previous_hop_node_id,
                envelope,
                onward_envelope,
                onward_descriptor_hint,
                request_commitment,
                now,
                &route_started_at,
            )
            .await;
        }
        let envelope = Arc::new(envelope);
        let route_lease = match begin_blind_relay_route(
            &state,
            envelope.route_id,
            request_commitment,
            previous_hop_node_id,
            now,
        )? {
            BlindRelayRouteStart::Acquired(lease) => lease,
            BlindRelayRouteStart::Completed(response) => {
                return attach_blind_relay_success_receipt(
                    Arc::clone(&envelope),
                    response,
                    now,
                    Arc::clone(&state.node_identity),
                )
                .await
            }
        };
        let response = attach_blind_relay_success_receipt(
            Arc::clone(&envelope),
            PeerBlindRelayResponse {
                accepted: true,
                terminal: true,
                forwarded: false,
                ttl_remaining: envelope.ttl,
                reason: Some("terminal_next_hop".to_string()),
                delivery_receipt: None,
                success_receipt: None,
                failure_receipt: None,
                opaque_terminal_response_b64: None,
            },
            now,
            Arc::clone(&state.node_identity),
        )
        .await?;
        complete_blind_relay_route(&state, route_lease, now, response.clone())?;
        record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
        state.peer_store.record_blind_relay_terminal(
            now,
            envelope.ttl,
            envelope.encrypted_blob.len(),
        );
        return Ok(response);
    }

    if envelope.next_hop == previous_hop_node_id {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if !envelope.can_forward() {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = envelope.next_hop;
    let (descriptor, used_descriptor_hint) = resolve_blind_relay_next_hop_descriptor(
        &state,
        &next_hop,
        now,
        onward_descriptor_hint.as_ref(),
    )
    .ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        BlindRelayError::NoRoute
    })?;
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        // [ROUTE-HEALTH-REMOTE-POISONING 2026-08-11 by Codex] A remote
        // previous hop can choose this target. Preflight rejection is therefore
        // not next-hop failure evidence; only a real outbound request may
        // mutate that peer's route health. The requester is rejected below.
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let original_envelope = Arc::new(envelope);
    let mut route_lease = match begin_blind_relay_route(
        &state,
        original_envelope.route_id,
        request_commitment,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => {
            return attach_blind_relay_success_receipt(
                Arc::clone(&original_envelope),
                response,
                now,
                Arc::clone(&state.node_identity),
            )
            .await
        }
    };

    let envelope_for_forwarding = Arc::clone(&original_envelope);
    let onward_envelope_for_forwarding = request.onward_envelope;
    let forwarding_identity = Arc::clone(&state.node_identity);
    let prepared_forward = run_blind_relay_crypto(move || {
        // [AUTHENTICATED-ONWARD-DOMAIN 2026-08-30 by Codex] Re-sign the
        // complete legacy forwarding unit in one bounded worker. The
        // original outer envelope stays immutable for its success receipt.
        let envelope = envelope_for_forwarding
            .decremented_ttl()
            .ok_or(BlindRelayError::TtlExhausted)?
            .sign_with(forwarding_identity.as_ref());
        let onward_envelope = onward_envelope_for_forwarding
            .map(|envelope| envelope.sign_with(forwarding_identity.as_ref()));
        Ok(PreparedLegacyBlindRelayForward {
            envelope,
            onward_envelope,
        })
    })
    .await?;
    let forwarded_onward_descriptor_hint = onward_descriptor_hint;
    let ttl_remaining = prepared_forward.envelope.ttl;

    let prepared_forward = prepare_blind_relay_forward_request(PeerBlindRelayRequest {
        envelope: prepared_forward.envelope,
        previous_hop_node_id: self_node_id,
        onward_envelope: prepared_forward.onward_envelope,
        onward_descriptor_hint: forwarded_onward_descriptor_hint,
    })
    .await?;

    let forward_started_at = blind_relay_response_observed_at(now, &route_started_at);
    route_lease
        .arm_effect(forward_started_at)
        .map_err(|_| record_blind_relay_replay_protection_failure(&state, forward_started_at))?;
    let observed_at = match forward_blind_relay_with_retry(
        &state,
        &url,
        &descriptor,
        prepared_forward,
        forward_started_at,
    )
    .await
    {
        Ok(outcome) => outcome.observed_at,
        Err(error) => return Err(error),
    };

    let response = attach_blind_relay_success_receipt(
        original_envelope,
        PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining,
            reason: Some("forwarded".to_string()),
            delivery_receipt: None,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: None,
        },
        observed_at,
        Arc::clone(&state.node_identity),
    )
    .await?;
    complete_blind_relay_route(&state, route_lease, observed_at, response.clone())?;
    let _ = state
        .peer_store
        .record_route_forward_success_for_descriptor(&descriptor, observed_at);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
    state
        .peer_store
        .record_blind_relay_forwarded(observed_at, ttl_remaining);

    Ok(response)
}

/// Onion routing v1 — this node is the addressed hop and the opaque blob is an
/// onion layer. Peel exactly one layer with the node's rotating onion key(s),
/// then either deliver locally (terminal hop) or forward the revealed inner
/// layer to the next hop (entry/middle hop).
///
/// Privacy invariant: a relay learns only the previous hop (transport auth) and
/// the immediate next hop (from its own peeled layer). It never sees the
/// original source, the final destination, or the payload. The onion secret
/// keys (current + previous within the rotation grace window, see
/// services::onion_keys) are never logged.
pub(super) async fn process_onion_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    envelope: BlindRelayEnvelope,
    request_commitment: [u8; 32],
    now: u64,
    route_started_at: &Instant,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let self_node_id = state.node_identity.public_key_bytes();
    let envelope = Arc::new(envelope);

    // Per-route replay/dedup, identical to the opaque terminal/forward paths.
    let mut route_lease = match begin_blind_relay_route(
        &state,
        envelope.route_id,
        request_commitment,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => {
            return attach_blind_relay_success_receipt(
                Arc::clone(&envelope),
                response,
                now,
                Arc::clone(&state.node_identity),
            )
            .await
        }
    };

    // Peel exactly one onion layer with the node's rotating onion key(s): the
    // current key, plus the previous key while it is within the rotation grace
    // window (forward secrecy — see services::onion_keys). A failure yields a
    // coarse bucket only, never a payload leak.
    let onion_secrets = crate::services::onion_keys::peel_secrets(now);
    let encrypted_envelope = Arc::clone(&envelope);
    let peel = match run_blind_relay_crypto(move || {
        try_open_onion_layer(&encrypted_envelope.encrypted_blob, &onion_secrets)
            .map_err(|_| BlindRelayError::OnionPeelFailed)
    })
    .await
    {
        Ok(peel) => peel,
        Err(BlindRelayError::OnionPeelFailed) => {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "onion_peel_failed");
            return Err(BlindRelayError::OnionPeelFailed);
        }
        Err(error) => return Err(error),
    };

    match peel.next_hop {
        // Terminal hop: `inner` is a ChatEnvelope, legacy signed Blind Vault
        // Put, or reply-capable Blind Vault request. Fixed protocol magic
        // selects the parser; malformed declared frames never fall back.
        None => {
            // [PREPARED-TERMINAL-EFFECT 2026-08-30 by Codex] Parsing, response
            // negotiation, and chat sender authentication must finish before
            // mutation recovery is armed. Read-only vault observations stay
            // unarmed, so cancellation can release and safely retry the route.
            let prepared =
                match prepare_onion_terminal_payload(&state, envelope.route_id, peel.inner, now)
                    .await
                {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        // [ONION-TERMINAL-PREPARATION-HEALTH 2026-09-13 by Codex]
                        // Preparation failures happen after authenticated route
                        // admission and must reach the aggregate relay health
                        // surface just like execution failures.
                        reject_blind_relay_previous_hop(
                            &state,
                            previous_hop_node_id,
                            now,
                            error.reason_bucket(),
                        );
                        return Err(error);
                    }
                };
            let PreparedOnionTerminalWork {
                proof_payload,
                operation,
            } = prepared;
            if operation.requires_durable_guard() {
                route_lease
                    .arm_effect(now)
                    .map_err(|_| record_blind_relay_replay_protection_failure(&state, now))?;
            }
            let terminal_delivery =
                match execute_onion_terminal_payload(&state, envelope.route_id, operation, now)
                    .await
                {
                    Ok(delivery) => delivery,
                    Err(error) => {
                        let failed_at = blind_relay_response_observed_at(now, route_started_at);
                        debug!(
                            reason = error.reason_bucket(),
                            "[BLIND_RELAY] Onion terminal delivery failed"
                        );
                        reject_blind_relay_previous_hop(
                            &state,
                            previous_hop_node_id,
                            failed_at,
                            "onion_terminal_delivery_failed",
                        );
                        return Err(error);
                    }
                };
            let accepted_at = blind_relay_response_observed_at(now, route_started_at);
            // [PURPOSE-BOUND-RECEIPT 2026-08-10 by Codex] Sign v2 only after
            // the selected terminal workload has crossed its durable acceptance
            // boundary. The purpose is committed with the opaque payload hash,
            // not returned as relay-visible metadata.
            let response = PeerBlindRelayResponse {
                accepted: true,
                terminal: true,
                forwarded: false,
                ttl_remaining: envelope.ttl,
                reason: Some("onion_terminal_delivered".to_string()),
                delivery_receipt: None,
                success_receipt: None,
                failure_receipt: None,
                opaque_terminal_response_b64: terminal_delivery.opaque_response_b64,
            };
            let response = match terminal_delivery.proof_mode {
                OnionReplyProofMode::RelayVisibleTerminalReceipt => {
                    attach_blind_relay_terminal_success_receipts(
                        Arc::clone(&envelope),
                        response,
                        TerminalDeliveryProofInput {
                            payload: proof_payload,
                            purpose: terminal_delivery.purpose,
                        },
                        accepted_at,
                        Arc::clone(&state.node_identity),
                    )
                    .await?
                }
                // [SOURCE-SEALED-TERMINAL-PROOF 2026-08-29 by Codex] The
                // terminal identity and signed workload result are already
                // authenticated inside this fixed-size ciphertext. Omitting
                // the clear terminal receipt prevents every middle hop from
                // reconstructing the final route endpoint.
                OnionReplyProofMode::SourceSealedTerminalProof => {
                    attach_blind_relay_success_receipt(
                        Arc::clone(&envelope),
                        response,
                        accepted_at,
                        Arc::clone(&state.node_identity),
                    )
                    .await?
                }
            };
            complete_blind_relay_route(&state, route_lease, accepted_at, response.clone())?;
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
            state.peer_store.record_blind_relay_terminal(
                accepted_at,
                envelope.ttl,
                envelope.encrypted_blob.len(),
            );
            Ok(response)
        }
        // Entry/middle hop: forward the inner layer to the revealed next hop.
        Some(next_hop) => {
            if next_hop == self_node_id || next_hop == previous_hop_node_id {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
                return Err(BlindRelayError::RouteLoop);
            }
            if !is_onion_blob(&peel.inner) {
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "onion_inner_not_layer",
                );
                return Err(BlindRelayError::OnionPeelFailed);
            }
            if !envelope.can_forward() {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
                return Err(BlindRelayError::TtlExhausted);
            }

            let descriptor = state.peer_store.get_valid(&next_hop, now).ok_or_else(|| {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                BlindRelayError::NoRoute
            })?;
            if !descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }
            // True onion routes are the recovery/proof path for a small mesh:
            // after restart a fresh signed peer may not yet have routeability
            // evidence, and refusing the proof attempt creates a deadlock. Keep
            // the hard stop for peers under local route quarantine; forward
            // errors below will still feed route health without logging payloads.
            if state.peer_store.is_route_quarantined_now(&next_hop, now) {
                reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
                return Err(BlindRelayError::NoRoute);
            }

            let endpoint = descriptor
                .descriptor
                .public_endpoint
                .as_deref()
                .ok_or_else(|| {
                    reject_blind_relay_previous_hop(
                        &state,
                        previous_hop_node_id,
                        now,
                        "missing_endpoint",
                    );
                    BlindRelayError::InvalidEndpoint
                })?;
            let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
                reject_blind_relay_previous_hop(
                    &state,
                    previous_hop_node_id,
                    now,
                    "invalid_endpoint",
                );
                BlindRelayError::InvalidEndpoint
            })?;

            // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Preserve the
            // authenticated ingress timestamp when reconstructing this exact
            // hop. A restart retry must produce byte-identical signed onward
            // input so the downstream node can replay its durable ACK without
            // repeating terminal storage or another network effect.
            let forward_seed = BlindRelayForwardSeed::from(envelope.as_ref());
            let forwarding_identity = Arc::clone(&state.node_identity);
            let forwarded_envelope = run_blind_relay_crypto(move || {
                Ok(build_forwarded_onion_envelope_from_seed(
                    forward_seed,
                    next_hop,
                    peel.inner,
                    forwarding_identity.as_ref(),
                ))
            })
            .await?;
            let ttl_remaining = forwarded_envelope.ttl;

            let forwarded_request = prepare_blind_relay_forward_request(PeerBlindRelayRequest {
                envelope: forwarded_envelope,
                previous_hop_node_id: self_node_id,
                onward_envelope: None,
                onward_descriptor_hint: None,
            })
            .await?;

            let forward_started_at = blind_relay_response_observed_at(now, route_started_at);
            route_lease.arm_effect(forward_started_at).map_err(|_| {
                record_blind_relay_replay_protection_failure(&state, forward_started_at)
            })?;
            let next_hop_forward = match forward_blind_relay_with_retry(
                &state,
                &url,
                &descriptor,
                forwarded_request,
                forward_started_at,
            )
            .await
            {
                Ok(ack) => ack,
                Err(error) => return Err(error),
            };
            let observed_at = next_hop_forward.observed_at;
            let next_hop_ack = next_hop_forward.response;

            let response = attach_blind_relay_success_receipt(
                Arc::clone(&envelope),
                PeerBlindRelayResponse {
                    accepted: true,
                    terminal: false,
                    forwarded: true,
                    ttl_remaining,
                    reason: Some("onion_forwarded".to_string()),
                    delivery_receipt: next_hop_ack.delivery_receipt,
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: next_hop_ack.opaque_terminal_response_b64,
                },
                observed_at,
                Arc::clone(&state.node_identity),
            )
            .await?;
            complete_blind_relay_route(&state, route_lease, observed_at, response.clone())?;
            let _ = state
                .peer_store
                .record_route_forward_success_for_descriptor(&descriptor, observed_at);
            record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
            state
                .peer_store
                .record_blind_relay_forwarded(observed_at, ttl_remaining);

            Ok(response)
        }
    }
}

pub(super) fn decode_onion_terminal_payload(
    payload: &[u8],
    route_id: [u8; 16],
    local_target_node_id: [u8; 32],
    now_secs: u64,
) -> Result<DecodedOnionTerminalPayload, OnionTerminalDecodeFailure> {
    if payload.first().copied() == Some(MEMCHAIN_MAGIC) {
        return PreparedAnonymousMailboxTerminal::decode(payload, route_id, local_target_node_id)
            .map(DecodedOnionTerminalPayload::AnonymousMailbox)
            .map_err(|_| {
                OnionTerminalDecodeFailure::Protocol(BlindRelayError::OnionTerminalPayloadRejected)
            });
    }
    if is_onion_reply_request(payload) {
        return prepare_blind_vault_inline_reply(payload)
            .map(DecodedOnionTerminalPayload::BlindVaultReply)
            .map_err(|error| {
                OnionTerminalDecodeFailure::Protocol(map_terminal_reply_failure(error))
            });
    }
    if is_blind_vault_frame(payload) {
        let frame = decode_blind_vault_frame(payload).map_err(|_| {
            OnionTerminalDecodeFailure::Protocol(BlindRelayError::OnionTerminalPayloadRejected)
        })?;
        let BlindVaultFrame::Put(request) = frame else {
            // Lease admission, pull, delete, issuer, and response frames retain
            // their dedicated bounded client API. Legacy onion terminal
            // compatibility remains Put-only.
            return Err(OnionTerminalDecodeFailure::Protocol(
                BlindRelayError::OnionTerminalPayloadRejected,
            ));
        };
        return Ok(DecodedOnionTerminalPayload::LegacyBlindVaultPut(request));
    }

    let envelope = decode_envelope(payload).map_err(|_| {
        OnionTerminalDecodeFailure::Protocol(BlindRelayError::OnionTerminalPayloadRejected)
    })?;
    validate_peer_envelope(&envelope, now_secs).map_err(OnionTerminalDecodeFailure::Message)?;
    Ok(DecodedOnionTerminalPayload::Message(envelope))
}

/// Performs terminal wire parsing and sender authentication without touching
/// Blind Vault or pending-message storage.
pub(super) async fn prepare_onion_terminal_payload(
    state: &ChatPeerState,
    route_id: [u8; 16],
    payload: Vec<u8>,
    now_secs: u64,
) -> Result<PreparedOnionTerminalWork, BlindRelayError> {
    let local_target_node_id = state.node_identity.public_key_bytes();
    let (proof_payload, decoded) = run_blind_relay_crypto(move || {
        let decoded =
            decode_onion_terminal_payload(&payload, route_id, local_target_node_id, now_secs);
        Ok((payload, decoded))
    })
    .await?;

    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(OnionTerminalDecodeFailure::Protocol(error)) => return Err(error),
        Err(OnionTerminalDecodeFailure::Message(error)) => {
            record_peer_envelope_rejection(state, now_secs, &error);
            return Err(map_terminal_chat_preparation_error(error));
        }
    };

    let operation = match decoded {
        DecodedOnionTerminalPayload::AnonymousMailbox(request) => {
            state
                .anonymous_mailbox
                .as_ref()
                .ok_or(BlindRelayError::ForwardFailed)?;
            let execution_permit = blind_vault_terminal_admission()
                .try_acquire_owned()
                .map_err(|_| BlindRelayError::Backpressure)?;
            PreparedOnionTerminalPayload::AnonymousMailbox {
                request,
                execution_permit,
            }
        }
        DecodedOnionTerminalPayload::BlindVaultReply(reply) => {
            state
                .blind_vault
                .as_ref()
                .ok_or(BlindRelayError::ForwardFailed)?;
            let execution_permit = blind_vault_terminal_admission()
                .try_acquire_owned()
                .map_err(|_| BlindRelayError::Backpressure)?;
            PreparedOnionTerminalPayload::BlindVaultReply {
                reply,
                execution_permit,
            }
        }
        DecodedOnionTerminalPayload::LegacyBlindVaultPut(request) => {
            state
                .blind_vault
                .as_ref()
                .ok_or(BlindRelayError::ForwardFailed)?;
            let execution_permit = blind_vault_terminal_admission()
                .try_acquire_owned()
                .map_err(|_| BlindRelayError::Backpressure)?;
            PreparedOnionTerminalPayload::LegacyBlindVaultPut {
                request,
                execution_permit,
            }
        }
        DecodedOnionTerminalPayload::Message(envelope) => {
            let storage_permit = acquire_chat_relay_storage(state, now_secs)
                .map_err(map_terminal_chat_preparation_error)?;
            PreparedOnionTerminalPayload::Message {
                envelope,
                storage_permit,
            }
        }
    };

    Ok(PreparedOnionTerminalWork {
        proof_payload,
        operation,
    })
}

pub(super) async fn execute_onion_terminal_payload(
    state: &ChatPeerState,
    route_id: [u8; 16],
    prepared: PreparedOnionTerminalPayload,
    now_secs: u64,
) -> Result<OnionTerminalDelivery, BlindRelayError> {
    match prepared {
        PreparedOnionTerminalPayload::AnonymousMailbox {
            request,
            execution_permit,
        } => {
            let repository = Arc::clone(
                state
                    .anonymous_mailbox
                    .as_ref()
                    .ok_or(BlindRelayError::ForwardFailed)?,
            );
            let terminal_identity = Arc::clone(&state.node_identity);
            let opaque_response_b64 = tokio::task::spawn_blocking(move || {
                let _execution_permit = execution_permit;
                request.execute(repository, terminal_identity, now_secs)
            })
            .await
            .map_err(|_| BlindRelayError::ForwardFailed)?
            .map_err(map_anonymous_mailbox_terminal_failure)?;
            Ok(OnionTerminalDelivery {
                purpose: OnionRoutePurpose::AnonymousMailboxV1,
                proof_mode: OnionReplyProofMode::SourceSealedTerminalProof,
                opaque_response_b64: Some(opaque_response_b64),
            })
        }
        PreparedOnionTerminalPayload::BlindVaultReply {
            reply,
            execution_permit,
        } => {
            let vault = Arc::clone(
                state
                    .blind_vault
                    .as_ref()
                    .ok_or(BlindRelayError::ForwardFailed)?,
            );
            let terminal_identity = Arc::clone(&state.node_identity);
            let now_ms = now_secs.saturating_mul(1_000);
            let reply = tokio::task::spawn_blocking(move || {
                let _execution_permit = execution_permit;
                reply.execute(vault.as_ref(), terminal_identity.as_ref(), route_id, now_ms)
            })
            .await
            .map_err(|_| BlindRelayError::ForwardFailed)?
            .map_err(map_terminal_reply_failure)?;
            Ok(OnionTerminalDelivery {
                purpose: reply.purpose,
                proof_mode: reply.proof_mode,
                opaque_response_b64: Some(reply.opaque_response_b64),
            })
        }
        PreparedOnionTerminalPayload::LegacyBlindVaultPut {
            request,
            execution_permit,
        } => {
            let vault = Arc::clone(
                state
                    .blind_vault
                    .as_ref()
                    .ok_or(BlindRelayError::ForwardFailed)?,
            );
            let now_ms = now_secs.saturating_mul(1_000);
            tokio::task::spawn_blocking(move || {
                let _execution_permit = execution_permit;
                vault.put(&request, now_ms)
            })
            .await
            .map_err(|_| BlindRelayError::ForwardFailed)?
            .map_err(|error| map_blind_vault_put_error(&error))?;
            Ok(OnionTerminalDelivery {
                purpose: OnionRoutePurpose::BlindVaultPut,
                proof_mode: OnionReplyProofMode::RelayVisibleTerminalReceipt,
                opaque_response_b64: None,
            })
        }
        PreparedOnionTerminalPayload::Message {
            envelope,
            storage_permit,
        } => process_authenticated_peer_relay_with_storage_permit(
            state.clone(),
            envelope,
            now_secs,
            storage_permit,
        )
        .await
        .map(|_| OnionTerminalDelivery {
            purpose: OnionRoutePurpose::MessageRelay,
            proof_mode: OnionReplyProofMode::RelayVisibleTerminalReceipt,
            opaque_response_b64: None,
        })
        .map_err(|_| BlindRelayError::ForwardFailed),
    }
}

pub(super) fn map_anonymous_mailbox_terminal_failure(
    failure: AnonymousMailboxTerminalFailure,
) -> BlindRelayError {
    match failure {
        AnonymousMailboxTerminalFailure::Rejected => BlindRelayError::OnionTerminalPayloadRejected,
        AnonymousMailboxTerminalFailure::Unavailable => BlindRelayError::ForwardFailed,
    }
}

pub(super) fn map_terminal_reply_failure(failure: TerminalReplyFailure) -> BlindRelayError {
    match failure {
        TerminalReplyFailure::Rejected | TerminalReplyFailure::ResponseTooLarge => {
            BlindRelayError::OnionTerminalPayloadRejected
        }
        // [BLIND-VAULT-ENCRYPTED-FAILURE 2026-08-28 by Codex] Valid workload
        // failures are sealed inside opaque replies. This remains fail-closed
        // for any capacity failure that occurs before response sealing.
        TerminalReplyFailure::Capacity => BlindRelayError::OnionTerminalCapacityExhausted,
        TerminalReplyFailure::Unavailable => BlindRelayError::ForwardFailed,
    }
}

pub(super) fn map_terminal_chat_preparation_error(error: ChatPeerRelayError) -> BlindRelayError {
    match error {
        ChatPeerRelayError::VerificationBackpressure | ChatPeerRelayError::StorageBackpressure => {
            BlindRelayError::Backpressure
        }
        _ => BlindRelayError::ForwardFailed,
    }
}

pub(super) fn map_blind_vault_put_error(error: &BlindVaultServiceError) -> BlindRelayError {
    // [BLIND-VAULT-RETRY-CLASS 2026-08-10 by Codex] The relay must make a
    // useful retry decision without forwarding replica-local state. Every
    // authorization, signature, lease, and object conflict shares one bucket.
    match error.put_failure_class() {
        BlindVaultPutFailureClass::Rejected => BlindRelayError::OnionTerminalPayloadRejected,
        BlindVaultPutFailureClass::Capacity => BlindRelayError::OnionTerminalCapacityExhausted,
        BlindVaultPutFailureClass::Unavailable => BlindRelayError::ForwardFailed,
    }
}

pub(super) async fn process_onion_middle_blind_relay(
    state: ChatPeerState,
    previous_hop_node_id: [u8; 32],
    outer_envelope: BlindRelayEnvelope,
    onward_envelope: BlindRelayEnvelope,
    onward_descriptor_hint: Option<SignedNodeDescriptor>,
    request_commitment: [u8; 32],
    now: u64,
    route_started_at: &Instant,
) -> Result<PeerBlindRelayResponse, BlindRelayError> {
    let outer_envelope = Arc::new(outer_envelope);
    // [AUTHENTICATED-ONWARD-DOMAIN 2026-08-30 by Codex] The private
    // `AuthenticatedPeerBlindRelayRequest` constructor already verified the
    // onward signature inside bounded blocking admission. Repeating that work
    // here would let valid requests consume Ed25519 verification on Tokio.
    validate_blind_relay_metadata(&onward_envelope, now).map_err(|error| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, error.reason_bucket());
        error
    })?;

    let self_node_id = state.node_identity.public_key_bytes();
    if onward_envelope.next_hop == self_node_id || onward_envelope.next_hop == previous_hop_node_id
    {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "route_loop");
        return Err(BlindRelayError::RouteLoop);
    }

    if !onward_envelope.can_forward() {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "ttl_exhausted");
        return Err(BlindRelayError::TtlExhausted);
    }

    let next_hop = onward_envelope.next_hop;
    let (descriptor, used_descriptor_hint) = resolve_blind_relay_next_hop_descriptor(
        &state,
        &next_hop,
        now,
        onward_descriptor_hint.as_ref(),
    )
    .ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        BlindRelayError::NoRoute
    })?;
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }
    if !used_descriptor_hint && !state.peer_store.is_routeable_now(&next_hop, now) {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "no_route");
        return Err(BlindRelayError::NoRoute);
    }

    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| {
            reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "missing_endpoint");
            BlindRelayError::InvalidEndpoint
        })?;
    let url = blind_peer_relay_url(endpoint).ok_or_else(|| {
        reject_blind_relay_previous_hop(&state, previous_hop_node_id, now, "invalid_endpoint");
        BlindRelayError::InvalidEndpoint
    })?;

    let mut route_lease = match begin_blind_relay_route(
        &state,
        outer_envelope.route_id,
        request_commitment,
        previous_hop_node_id,
        now,
    )? {
        BlindRelayRouteStart::Acquired(lease) => lease,
        BlindRelayRouteStart::Completed(response) => {
            return attach_blind_relay_success_receipt(
                Arc::clone(&outer_envelope),
                response,
                now,
                Arc::clone(&state.node_identity),
            )
            .await
        }
    };

    let forwarding_identity = Arc::clone(&state.node_identity);
    let forwarded_envelope = run_blind_relay_crypto(move || {
        onward_envelope
            .decremented_ttl()
            .ok_or(BlindRelayError::TtlExhausted)
            .map(|envelope| envelope.sign_with(forwarding_identity.as_ref()))
    })
    .await?;
    let ttl_remaining = forwarded_envelope.ttl;

    let forwarded_request = prepare_blind_relay_forward_request(PeerBlindRelayRequest {
        envelope: forwarded_envelope,
        previous_hop_node_id: self_node_id,
        onward_envelope: None,
        onward_descriptor_hint: None,
    })
    .await?;

    let forward_started_at = blind_relay_response_observed_at(now, route_started_at);
    route_lease
        .arm_effect(forward_started_at)
        .map_err(|_| record_blind_relay_replay_protection_failure(&state, forward_started_at))?;
    let next_hop_forward = match forward_blind_relay_with_retry(
        &state,
        &url,
        &descriptor,
        forwarded_request,
        forward_started_at,
    )
    .await
    {
        Ok(ack) => ack,
        Err(error) => return Err(error),
    };
    let observed_at = next_hop_forward.observed_at;
    let next_hop_ack = next_hop_forward.response;

    let response = attach_blind_relay_success_receipt(
        outer_envelope,
        PeerBlindRelayResponse {
            accepted: true,
            terminal: false,
            forwarded: true,
            ttl_remaining,
            reason: Some("onion_middle_forwarded".to_string()),
            delivery_receipt: next_hop_ack.delivery_receipt,
            success_receipt: None,
            failure_receipt: None,
            opaque_terminal_response_b64: next_hop_ack.opaque_terminal_response_b64,
        },
        observed_at,
        Arc::clone(&state.node_identity),
    )
    .await?;
    complete_blind_relay_route(&state, route_lease, observed_at, response.clone())?;
    let _ = state
        .peer_store
        .record_route_forward_success_for_descriptor(&descriptor, observed_at);
    record_blind_relay_previous_hop_success(&state, previous_hop_node_id);
    state
        .peer_store
        .record_blind_relay_forwarded(observed_at, ttl_remaining);

    Ok(response)
}

pub(super) fn resolve_blind_relay_next_hop_descriptor(
    state: &ChatPeerState,
    next_hop: &[u8; 32],
    now: u64,
    descriptor_hint: Option<&SignedNodeDescriptor>,
) -> Option<(SignedNodeDescriptor, bool)> {
    if let Some(descriptor) = state.peer_store.get_valid(next_hop, now) {
        return Some((descriptor, false));
    }

    let descriptor = descriptor_hint?;
    if descriptor.node_id() != *next_hop {
        return None;
    }
    if descriptor.verify_at(now).is_err() {
        return None;
    }
    if !descriptor
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay)
    {
        return None;
    }
    Some((descriptor.clone(), true))
}

pub(super) fn begin_blind_relay_route(
    state: &ChatPeerState,
    route_id: [u8; 16],
    request_commitment: [u8; 32],
    previous_hop: [u8; 32],
    now: u64,
) -> Result<BlindRelayRouteStart, BlindRelayError> {
    if let Some(relay) = state.chat_relay.as_ref() {
        let admission = relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .map_err(|_| record_blind_relay_replay_protection_failure(state, now))?;
        return match admission {
            BlindRelayRouteAdmission::Reserved => Ok(BlindRelayRouteStart::Acquired(
                BlindRelayRouteLease::durable(
                    Arc::clone(relay),
                    route_id,
                    request_commitment,
                    false,
                ),
            )),
            BlindRelayRouteAdmission::ReservedForRecovery => Ok(BlindRelayRouteStart::Acquired(
                BlindRelayRouteLease::durable(
                    Arc::clone(relay),
                    route_id,
                    request_commitment,
                    true,
                ),
            )),
            BlindRelayRouteAdmission::Pending => {
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "route_in_flight");
                Err(BlindRelayError::RouteInFlight)
            }
            BlindRelayRouteAdmission::Conflict => {
                reject_blind_relay_previous_hop(state, previous_hop, now, "replay_conflict");
                Err(BlindRelayError::ReplayConflict)
            }
            BlindRelayRouteAdmission::CapacityExhausted => {
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "replay_capacity");
                Err(BlindRelayError::ReplayCapacity)
            }
            BlindRelayRouteAdmission::Completed {
                response,
                completed_at,
            } => {
                // Signed receipts are online evidence. Do not replay one after
                // its verifier freshness window, but retain the route row for
                // the full envelope horizon so stale retries cannot re-execute.
                if completed_at > now
                    || now.saturating_sub(completed_at) > BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS
                {
                    state
                        .peer_store
                        .record_blind_relay_rejected(now, "replay_response_expired");
                    return Err(BlindRelayError::ReplayResponseExpired);
                }
                let response = decode_durable_blind_relay_response(&response)
                    .map_err(|_| record_blind_relay_replay_protection_failure(state, now))?;
                validate_completed_blind_relay_response(&response)
                    .map_err(|_| record_blind_relay_replay_protection_failure(state, now))?;
                state
                    .peer_store
                    .record_blind_relay_rejected(now, "duplicate_route");
                record_blind_relay_previous_hop_success(state, previous_hop);
                Ok(BlindRelayRouteStart::Completed(response))
            }
        };
    }

    let decision = state
        .blind_relay_replay_registry
        .observe(route_id, request_commitment, now);

    match decision {
        BlindRelayRouteReplayDecision::New { generation } => {
            Ok(BlindRelayRouteStart::Acquired(BlindRelayRouteLease::local(
                Arc::clone(&state.blind_relay_replay_registry),
                route_id,
                request_commitment,
                generation,
            )))
        }
        BlindRelayRouteReplayDecision::InFlight => {
            // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] An unresolved first
            // attempt is not proof of acceptance. Return a retryable status and
            // leave previous-hop health unchanged; a later retry will either
            // replay the durable result or own a fresh attempt after failure.
            state
                .peer_store
                .record_blind_relay_rejected(now, "route_in_flight");
            Err(BlindRelayError::RouteInFlight)
        }
        BlindRelayRouteReplayDecision::Saturated => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "replay_capacity");
            Err(BlindRelayError::ReplayCapacity)
        }
        BlindRelayRouteReplayDecision::Conflict => {
            reject_blind_relay_previous_hop(state, previous_hop, now, "replay_conflict");
            Err(BlindRelayError::ReplayConflict)
        }
        BlindRelayRouteReplayDecision::Completed(response) => {
            // ACK-loss retries receive the exact bounded success response,
            // including any terminal-signed receipt. No payload is retained.
            state
                .peer_store
                .record_blind_relay_rejected(now, "duplicate_route");
            record_blind_relay_previous_hop_success(state, previous_hop);
            Ok(BlindRelayRouteStart::Completed(*response))
        }
    }
}

pub(super) fn complete_blind_relay_route(
    state: &ChatPeerState,
    route_lease: BlindRelayRouteLease,
    now: u64,
    response: PeerBlindRelayResponse,
) -> Result<(), BlindRelayError> {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] One completion boundary
    // keeps every terminal/forward path on the same durable failure telemetry.
    // The fixed bucket exposes no route, peer, endpoint, receipt, or payload.
    route_lease
        .complete(now, response)
        .map_err(|_| record_blind_relay_replay_protection_failure(state, now))
}

pub(super) fn record_blind_relay_replay_protection_failure(
    state: &ChatPeerState,
    now: u64,
) -> BlindRelayError {
    state
        .peer_store
        .record_blind_relay_rejected(now, "replay_protection_unavailable");
    BlindRelayError::ReplayProtectionUnavailable
}

pub(super) fn check_blind_relay_previous_hop_allowed(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
) -> Result<(), BlindRelayError> {
    let decision = state
        .blind_relay_abuse_guard
        .observe_request(previous_hop, now);

    match decision {
        BlindRelayAbuseDecision::Allowed => Ok(()),
        BlindRelayAbuseDecision::CapacityLimited => {
            // [BLIND-RELAY-BUCKET-FAIRNESS 2026-08-21 by Codex] Capacity
            // pressure is aggregate node protection, not evidence that this
            // authenticated peer misbehaved. Do not mutate peer reputation or
            // quarantine state while every retained bucket is still protected.
            state
                .peer_store
                .record_blind_relay_rejected(now, "rate_limited");
            Err(BlindRelayError::RateLimited)
        }
        BlindRelayAbuseDecision::RateLimited { quarantine_until } => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "rate_limited");
            state
                .peer_store
                .record_blind_relay_quarantine_started(now, "rate_limit");
            state
                .peer_store
                .record_peer_relay_rejection(&previous_hop, now, "rate_limited");
            state.peer_store.record_peer_relay_quarantine_started(
                &previous_hop,
                now,
                quarantine_until,
                "rate_limit",
            );
            Err(BlindRelayError::RateLimited)
        }
        BlindRelayAbuseDecision::Quarantined { quarantine_until } => {
            state
                .peer_store
                .record_blind_relay_rejected(now, "quarantined");
            state
                .peer_store
                .record_peer_relay_rejection(&previous_hop, now, "quarantined");
            state.peer_store.record_peer_relay_quarantine_started(
                &previous_hop,
                now,
                quarantine_until,
                "still_quarantined",
            );
            Err(BlindRelayError::Quarantined)
        }
    }
}

pub(super) fn reject_blind_relay_previous_hop(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
    now: u64,
    reason: &'static str,
) {
    state.peer_store.record_blind_relay_rejected(now, reason);
    state
        .peer_store
        .record_peer_relay_rejection(&previous_hop, now, reason);
    if !blind_relay_reason_counts_toward_quarantine(reason) {
        return;
    }

    let quarantine_until = state
        .blind_relay_abuse_guard
        .record_failure(previous_hop, now);
    if let Some(quarantine_until) = quarantine_until {
        state
            .peer_store
            .record_blind_relay_quarantine_started(now, "failure_threshold");
        state.peer_store.record_peer_relay_quarantine_started(
            &previous_hop,
            now,
            quarantine_until,
            "failure_threshold",
        );
    }
}

pub(super) fn record_blind_relay_previous_hop_success(
    state: &ChatPeerState,
    previous_hop: [u8; 32],
) {
    state.blind_relay_abuse_guard.record_success(previous_hop);
}

pub(super) fn blind_relay_reason_counts_toward_quarantine(reason: &str) -> bool {
    matches!(
        reason,
        "invalid_previous_hop" | "invalid_signature" | "self_loop" | "route_loop" | "ttl_exhausted"
    )
}

#[cfg(test)]
pub(super) fn validate_blind_relay_envelope(
    envelope: &BlindRelayEnvelope,
    previous_hop_node_id: &[u8; 32],
    now: u64,
) -> Result<(), BlindRelayError> {
    authenticate_blind_relay_envelope(envelope, previous_hop_node_id)?;
    validate_blind_relay_metadata(envelope, now)
}

pub(super) fn authenticate_blind_relay_envelope(
    envelope: &BlindRelayEnvelope,
    previous_hop_node_id: &[u8; 32],
) -> Result<(), BlindRelayError> {
    let previous_hop = IdentityPublicKey::from_bytes(previous_hop_node_id)
        .map_err(|_| BlindRelayError::InvalidPreviousHop)?;
    envelope
        .verify_signature_from(&previous_hop)
        .map_err(|_| BlindRelayError::InvalidSignature)?;
    Ok(())
}

pub(super) fn validate_blind_relay_metadata(
    envelope: &BlindRelayEnvelope,
    now: u64,
) -> Result<(), BlindRelayError> {
    validate_blind_relay_timestamp(envelope.timestamp, now)?;
    validate_blind_relay_envelope_size(envelope).map_err(|_| BlindRelayError::EnvelopeTooLarge)?;
    Ok(())
}

pub(super) fn validate_blind_relay_timestamp(
    timestamp: u64,
    now: u64,
) -> Result<(), BlindRelayError> {
    validate_relay_timestamp(
        timestamp,
        now,
        BLIND_RELAY_MAX_ENVELOPE_AGE_SECS,
        BLIND_RELAY_MAX_FUTURE_SKEW_SECS,
    )
    .map_err(|error| match error {
        RelayTimestampError::Expired => BlindRelayError::TimestampExpired,
        RelayTimestampError::InFuture => BlindRelayError::TimestampInFuture,
    })
}
