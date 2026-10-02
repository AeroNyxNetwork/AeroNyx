// [ARCH-SPLIT 2026-10-02]
// Signed v1/v2/v3 peer relay admission, storage, and local session delivery.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn peer_chat_relay_auth_v2_signing_data(
    previous_hop_node_id: &[u8; 32],
    envelope: &ChatEnvelope,
) -> Result<Vec<u8>, bincode::Error> {
    let encoded_envelope = encode_envelope(envelope)?;
    let mut signing_data =
        Vec::with_capacity(PEER_CHAT_RELAY_AUTH_V2_DOMAIN.len() + 32 + 8 + encoded_envelope.len());
    signing_data.extend_from_slice(PEER_CHAT_RELAY_AUTH_V2_DOMAIN);
    signing_data.extend_from_slice(previous_hop_node_id);
    signing_data.extend_from_slice(&(encoded_envelope.len() as u64).to_be_bytes());
    signing_data.extend_from_slice(&encoded_envelope);
    Ok(signing_data)
}

pub(super) fn peer_chat_relay_request_commitment(
    domain: &[u8],
    signing_data: &[u8],
    previous_hop_signature: &[u8; 64],
) -> [u8; 32] {
    // [SINGLE-PASS-DIRECT-REQUEST-COMMITMENT 2026-08-31 by Codex] This pure
    // helper is the one commitment contract for v2 and v3. Version separation
    // remains explicit in `domain`; callers cannot accidentally hash a second
    // serialization that differs from the bytes already signed.
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((signing_data.len() as u64).to_be_bytes());
    hasher.update(signing_data);
    hasher.update(previous_hop_signature);
    hasher.finalize().into()
}

pub(super) fn peer_chat_relay_auth_v3_signing_data(
    previous_hop_node_id: &[u8; 32],
    target_node_id: &[u8; 32],
    envelope: &ChatEnvelope,
) -> Result<Vec<u8>, bincode::Error> {
    let encoded_envelope = encode_envelope(envelope)?;
    let mut signing_data =
        Vec::with_capacity(PEER_CHAT_RELAY_AUTH_V3_DOMAIN.len() + 64 + 8 + encoded_envelope.len());
    signing_data.extend_from_slice(PEER_CHAT_RELAY_AUTH_V3_DOMAIN);
    signing_data.extend_from_slice(previous_hop_node_id);
    signing_data.extend_from_slice(target_node_id);
    signing_data.extend_from_slice(&(encoded_envelope.len() as u64).to_be_bytes());
    signing_data.extend_from_slice(&encoded_envelope);
    Ok(signing_data)
}

// ============================================
// Handlers
// ============================================

/// Applies ordinary peer-relay backpressure before Axum reads a JSON body.
pub(super) async fn peer_relay_request_gate(
    State(gate): State<Arc<PeerRelayRequestGate>>,
    mut request: Request,
    next: Next,
) -> Response {
    // [PEER-RELAY-ADMISSION 2026-08-15 by Codex] Count attempted direct-relay
    // work before body parsing. This intentionally uses only aggregate process
    // state: the legacy wire contract cannot authenticate a previous-hop node,
    // and sender/receiver/IP buckets would create misleading identity state.
    if !gate.admit(Instant::now()) {
        gate.record_rejected(ChatRelayInboundFailureReason::from_bucket("rate_limited"));
        return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
    }

    let Some(_in_flight) =
        InFlightRequestGuard::try_acquire(&gate.in_flight, MAX_IN_FLIGHT_PEER_CHAT_REQUESTS)
    else {
        gate.record_rejected(ChatRelayInboundFailureReason::from_bucket("backpressure"));
        return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
    };

    // [AUTHENTICATED-PEER-FAIRNESS 2026-08-15 by Codex] Pass the exact gate
    // that admitted parser work to the v2 handler. This keeps direct-relay
    // admission ownership out of the shared chat/blind relay runtime state.
    request.extensions_mut().insert(Arc::clone(&gate));
    next.run(request).await
}

pub(super) fn rejected_peer_relay_response(status: StatusCode) -> Response {
    (
        status,
        Json(PeerChatRelayResponse {
            accepted: false,
            duplicate: false,
            delivered_online: 0,
            stored_pending: false,
        }),
    )
        .into_response()
}

pub(super) fn durable_peer_acceptance_response() -> PeerChatRelayResponse {
    // [PEER-ACK-PRIVACY 2026-08-15 by Codex] Preserve the legacy JSON schema
    // while returning only the one fact another node needs: the ciphertext is
    // now in durable custody. Revealing duplicate or online-session state lets
    // arbitrary signed senders probe a receiver's presence and device count.
    PeerChatRelayResponse {
        accepted: true,
        duplicate: false,
        delivered_online: 0,
        stored_pending: true,
    }
}

pub(super) async fn peer_relay_handler(
    State(state): State<ChatPeerState>,
    Json(request): Json<PeerChatRelayRequest>,
) -> impl IntoResponse {
    peer_relay_response(state, request.envelope).await
}

pub(super) async fn peer_relay_v2_handler(
    State(state): State<ChatPeerState>,
    Extension(gate): Extension<Arc<PeerRelayRequestGate>>,
    Json(request): Json<PeerChatRelayRequestV2>,
) -> impl IntoResponse {
    // [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] Authenticate the immediate
    // node before the inner envelope reaches durable storage. Invalid claims
    // affect only aggregate health and cannot poison another node's identity.
    let authenticated =
        match authenticate_direct_peer_relay_request(DirectPeerAuthenticationRequest::V2(request))
            .await
        {
            Ok(authenticated) => authenticated,
            Err(failure) => return reject_direct_peer_authentication(&state, failure),
        };

    authenticated_peer_relay_response(
        state,
        gate,
        authenticated.envelope,
        authenticated.previous_hop_node_id,
        authenticated.request_commitment,
    )
    .await
}

pub(super) async fn peer_relay_v3_handler(
    State(state): State<ChatPeerState>,
    Extension(gate): Extension<Arc<PeerRelayRequestGate>>,
    Json(request): Json<PeerChatRelayRequestV3>,
) -> impl IntoResponse {
    // [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] Reject a request
    // signed for another node before durable storage or authenticated quota
    // attribution. Node ids are already public discovery metadata, so this
    // branch does not expose client identity, receiver state, or content.
    let local_node_id = state.node_identity.public_key_bytes();
    if request.target_node_id != local_node_id {
        if let Some(relay) = state.chat_relay.as_ref() {
            relay.record_peer_relay_inbound_rejected_typed(
                now_secs(),
                ChatRelayInboundFailureReason::from_bucket("peer_target_mismatch"),
            );
        }
        return rejected_peer_relay_response(StatusCode::UNAUTHORIZED);
    }
    let authenticated =
        match authenticate_direct_peer_relay_request(DirectPeerAuthenticationRequest::V3 {
            request,
            expected_target_node_id: local_node_id,
        })
        .await
        {
            Ok(authenticated) => authenticated,
            Err(failure) => return reject_direct_peer_authentication(&state, failure),
        };

    authenticated_peer_relay_response(
        state,
        gate,
        authenticated.envelope,
        authenticated.previous_hop_node_id,
        authenticated.request_commitment,
    )
    .await
}

pub(super) fn reject_direct_peer_authentication(
    state: &ChatPeerState,
    failure: DirectPeerAuthenticationFailure,
) -> Response {
    if let Some(relay) = state.chat_relay.as_ref() {
        relay.record_peer_relay_inbound_rejected_typed(
            now_secs(),
            ChatRelayInboundFailureReason::from_bucket(failure.reason_bucket()),
        );
    }
    rejected_peer_relay_response(failure.status_code())
}

/// Applies post-authentication fairness and emits one common custody response.
///
/// [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] v2 and v3 share the
/// exact durable-storage and signed-receipt boundary. Version-specific code is
/// limited to request authentication, avoiding divergent custody semantics.
pub(super) async fn authenticated_peer_relay_response(
    state: ChatPeerState,
    gate: Arc<PeerRelayRequestGate>,
    envelope: ChatEnvelope,
    previous_hop_node_id: [u8; 32],
    request_commitment: [u8; 32],
) -> Response {
    let replay_lease = match gate.begin_authenticated_replay(request_commitment, Instant::now()) {
        AuthenticatedPeerRelayReplayStart::Acquired(lease) => lease,
        AuthenticatedPeerRelayReplayStart::Completed(response) => {
            // [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Return the
            // exact ACK produced after durable custody. This path neither
            // consumes per-node quota nor repeats storage/live delivery.
            if let Some(relay) = state.chat_relay.as_ref() {
                relay.record_peer_relay_inbound_accepted(now_secs(), true, 0, false);
            }
            return (StatusCode::OK, Json(response)).into_response();
        }
        AuthenticatedPeerRelayReplayStart::InFlight => {
            gate.record_rejected(ChatRelayInboundFailureReason::from_bucket(
                "peer_auth_retry_in_flight",
            ));
            let status =
                StatusCode::from_u16(HTTP_TOO_EARLY_STATUS_CODE).unwrap_or(StatusCode::CONFLICT);
            return rejected_peer_relay_response(status);
        }
        AuthenticatedPeerRelayReplayStart::Saturated => {
            gate.record_rejected(ChatRelayInboundFailureReason::from_bucket(
                "peer_auth_retry_cache_saturated",
            ));
            return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
        }
    };

    if !gate.admit_authenticated(previous_hop_node_id, Instant::now()) {
        gate.record_rejected(ChatRelayInboundFailureReason::from_bucket(
            "peer_auth_rate_limited",
        ));
        return rejected_peer_relay_response(StatusCode::TOO_MANY_REQUESTS);
    }

    let node_identity = Arc::clone(&state.node_identity);
    match process_peer_relay(state, envelope).await {
        Ok(relay) => {
            // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Sign only after
            // `process_peer_relay` has established durable custody. The
            // commitment was computed from the already authenticated request.
            let accepted_at = now_secs();
            let receipt = match complete_direct_relay_crypto(move || {
                PeerChatRelayReceiptV2::accepted(
                    request_commitment,
                    accepted_at,
                    node_identity.as_ref(),
                )
            })
            .await
            {
                Ok(receipt) => receipt,
                Err(DirectRelayCryptoFailure::Unavailable) => {
                    // Durable custody is already idempotent. Leave the replay
                    // lease incomplete and ask the sender to retry; the next
                    // attempt can recover the exact custody ACK without
                    // claiming that an unsigned response is authoritative.
                    return rejected_direct_peer_relay_v2_response(StatusCode::SERVICE_UNAVAILABLE);
                }
            };
            let response = PeerChatRelayResponseV2 {
                relay,
                receipt: Some(receipt),
            };
            replay_lease.complete(response.clone());
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(error) => (
            error.status_code(),
            Json(PeerChatRelayResponseV2 {
                relay: PeerChatRelayResponse {
                    accepted: false,
                    duplicate: false,
                    delivered_online: 0,
                    stored_pending: false,
                },
                receipt: None,
            }),
        )
            .into_response(),
    }
}

pub(super) fn rejected_direct_peer_relay_v2_response(status: StatusCode) -> Response {
    (
        status,
        Json(PeerChatRelayResponseV2 {
            relay: PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            },
            receipt: None,
        }),
    )
        .into_response()
}

pub(super) async fn peer_relay_response(state: ChatPeerState, envelope: ChatEnvelope) -> Response {
    match process_peer_relay(state, envelope).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(error) => (
            error.status_code(),
            Json(PeerChatRelayResponse {
                accepted: false,
                duplicate: false,
                delivered_online: 0,
                stored_pending: false,
            }),
        )
            .into_response(),
    }
}

pub(super) async fn process_peer_relay(
    state: ChatPeerState,
    envelope: ChatEnvelope,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let now = now_secs();
    let envelope = validate_peer_envelope_for_relay(&state, envelope, now).await?;
    process_authenticated_peer_relay(state, envelope, now).await
}

/// Verifies one end-to-end sender envelope and records only a coarse rejection.
///
/// Direct HTTP relay uses its dedicated CPU partition. Onion terminal dispatch
/// validates the same envelope contract inside the blind CPU domain so one
/// public surface cannot consume another surface's reserved capacity.
pub(super) async fn validate_peer_envelope_for_relay(
    state: &ChatPeerState,
    envelope: ChatEnvelope,
    now: u64,
) -> Result<ChatEnvelope, ChatPeerRelayError> {
    let permit = match direct_relay_cpu_admission().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let error = ChatPeerRelayError::VerificationBackpressure;
            record_peer_envelope_rejection(state, now, &error);
            return Err(error);
        }
    };
    let worker = execute_direct_relay_crypto(permit, move || {
        let result = validate_peer_envelope(&envelope, now);
        (envelope, result)
    })
    .await;
    let (envelope, result) = match worker {
        Ok(result) => result,
        Err(DirectRelayCryptoFailure::Unavailable) => {
            let error = ChatPeerRelayError::VerificationUnavailable;
            record_peer_envelope_rejection(state, now, &error);
            return Err(error);
        }
    };
    if let Err(error) = result {
        record_peer_envelope_rejection(state, now, &error);
        return Err(error);
    }
    Ok(envelope)
}

pub(super) fn record_peer_envelope_rejection(
    state: &ChatPeerState,
    now: u64,
    error: &ChatPeerRelayError,
) {
    if let Some(relay) = state.chat_relay.as_ref() {
        relay.record_peer_relay_inbound_rejected_typed(
            now,
            ChatRelayInboundFailureReason::from_bucket(error.reason_bucket()),
        );
    }
}

/// Establishes durable custody for an already authenticated sender envelope.
pub(super) async fn process_authenticated_peer_relay(
    state: ChatPeerState,
    envelope: ChatEnvelope,
    now: u64,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let storage_permit = acquire_chat_relay_storage(&state, now)?;
    process_authenticated_peer_relay_with_storage_permit(state, envelope, now, storage_permit).await
}

pub(super) fn acquire_chat_relay_storage(
    state: &ChatPeerState,
    now: u64,
) -> Result<OwnedSemaphorePermit, ChatPeerRelayError> {
    if state.chat_relay.is_none() {
        return Err(ChatPeerRelayError::RelayUnavailable);
    }
    chat_relay_storage_admission()
        .try_acquire_owned()
        .map_err(|_| {
            let error = ChatPeerRelayError::StorageBackpressure;
            record_peer_envelope_rejection(state, now, &error);
            error
        })
}

pub(super) async fn process_authenticated_peer_relay_with_storage_permit(
    state: ChatPeerState,
    envelope: ChatEnvelope,
    now: u64,
    storage_permit: OwnedSemaphorePermit,
) -> Result<PeerChatRelayResponse, ChatPeerRelayError> {
    let Some(relay) = state.chat_relay.as_ref().map(Arc::clone) else {
        return Err(ChatPeerRelayError::RelayUnavailable);
    };

    // [DURABLE-RECEIPT-BOUNDARY 2026-08-15 by Codex] Persist the exact signed
    // envelope before consulting the live-delivery dedupe cache. Checking only
    // `message_id` first allowed a conflicting ciphertext to be reported as an
    // accepted retry; an onion terminal could then sign a receipt for bytes it
    // had never stored. `store_pending` is idempotent for byte-identical retries
    // and rejects same-ID/different-envelope collisions atomically.
    // [RELAY-STORAGE-ADMISSION 2026-08-30 by Codex] SQLite custody is
    // synchronous. Keep its owned permit inside the blocking worker so request
    // cancellation cannot create unbounded detached database tasks.
    let store_relay = Arc::clone(&relay);
    let worker = tokio::task::spawn_blocking(move || {
        let _storage_permit = storage_permit;
        let result = store_relay.store_pending(&envelope);
        (envelope, result)
    })
    .await;
    let (envelope, result) = match worker {
        Ok(result) => result,
        Err(_) => {
            warn!("[CHAT_PEER] Pending custody worker failed closed");
            let error = ChatPeerRelayError::StoreFailed;
            record_peer_envelope_rejection(&state, now, &error);
            return Err(error);
        }
    };
    result.map_err(|error| {
        let reason = error.reason_bucket();
        warn!(reason, "[CHAT_PEER] Failed to durably accept peer envelope");
        // [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] Preserve the
        // storage diagnostic in the local warning while exporting only a
        // validated aggregate bucket to node health.
        relay.record_peer_relay_inbound_rejected_typed(
            now,
            ChatRelayInboundFailureReason::from_bucket(reason),
        );
        map_pending_store_error(&error)
    })?;

    if relay.is_online_duplicate(&envelope.message_id) {
        debug!("[CHAT_PEER] Duplicate peer envelope ignored");
        relay.record_peer_relay_inbound_accepted(now, true, 0, true);
        return Ok(durable_peer_acceptance_response());
    }

    let target_sessions = state.sessions.get_all_by_wallet(&envelope.receiver);
    let mut delivered_online = 0usize;

    for session in target_sessions {
        if send_envelope_to_session(&envelope, &session, &state.udp).await {
            delivered_online += 1;
        }
    }

    // The authenticated receiver retires this durable copy with ChatAck only
    // after local persistence. This keeps online UDP delivery crash-safe and
    // gives terminal receipts one stable meaning: accepted into durable relay
    // custody, never merely queued to a socket.
    relay.record_peer_relay_inbound_accepted(now, false, delivered_online, true);

    Ok(durable_peer_acceptance_response())
}

pub(super) fn map_pending_store_error(error: &ChatRelayError) -> ChatPeerRelayError {
    match error {
        ChatRelayError::MessageTooLarge { size, .. } => {
            ChatPeerRelayError::EnvelopeTooLarge { size: *size }
        }
        error if error.is_capacity_exhausted() => ChatPeerRelayError::PendingCapacity,
        _ => ChatPeerRelayError::StoreFailed,
    }
}

pub(super) fn direct_relay_cpu_admission() -> Arc<Semaphore> {
    Arc::clone(DIRECT_RELAY_CPU_ADMISSION.get_or_init(|| {
        Arc::new(Semaphore::new(signature_verification_capacity(
            MAX_DIRECT_RELAY_CPU_OPERATIONS_IN_FLIGHT,
        )))
    }))
}

pub(super) fn chat_relay_storage_admission() -> Arc<Semaphore> {
    Arc::clone(
        CHAT_RELAY_STORAGE_ADMISSION
            .get_or_init(|| Arc::new(Semaphore::new(MAX_CHAT_RELAY_STORAGE_OPERATIONS_IN_FLIGHT))),
    )
}

pub(super) async fn authenticate_direct_peer_relay_request(
    request: DirectPeerAuthenticationRequest,
) -> Result<AuthenticatedDirectPeerRelayRequest, DirectPeerAuthenticationFailure> {
    let permit = direct_relay_cpu_admission()
        .try_acquire_owned()
        .map_err(|_| DirectPeerAuthenticationFailure::Backpressure)?;
    execute_direct_relay_crypto(permit, move || {
        // [DIRECT-RELAY-VERIFY-ADMISSION 2026-08-30 by Codex] The owned permit
        // remains in the shared worker after HTTP cancellation. Excess work
        // never queues, and no inner envelope or node identity reaches
        // telemetry.
        match request {
            DirectPeerAuthenticationRequest::V2(request) => {
                let request_commitment = request
                    .verified_request_commitment()
                    .ok_or(DirectPeerAuthenticationFailure::Invalid)?;
                Ok(AuthenticatedDirectPeerRelayRequest {
                    envelope: request.envelope,
                    previous_hop_node_id: request.previous_hop_node_id,
                    request_commitment,
                })
            }
            DirectPeerAuthenticationRequest::V3 {
                request,
                expected_target_node_id,
            } => {
                let request_commitment = request
                    .verified_request_commitment_for_target(&expected_target_node_id)
                    .ok_or(DirectPeerAuthenticationFailure::Invalid)?;
                Ok(AuthenticatedDirectPeerRelayRequest {
                    envelope: request.envelope,
                    previous_hop_node_id: request.previous_hop_node_id,
                    request_commitment,
                })
            }
        }
    })
    .await
    .map_err(|DirectRelayCryptoFailure::Unavailable| DirectPeerAuthenticationFailure::Unavailable)?
}

/// Waits fairly for direct-relay crypto capacity after durable custody.
///
/// Preflight verification intentionally uses `try_acquire_owned` so hostile
/// ingress cannot build a blocking-task queue. Completion is different: the
/// node already owns the ciphertext, and the parser-front in-flight gate
/// bounds these waiters, so fairness prevents fresh verification work from
/// starving authoritative receipt signing.
pub(super) async fn complete_direct_relay_crypto<T, F>(
    work: F,
) -> Result<T, DirectRelayCryptoFailure>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let permit = direct_relay_cpu_admission()
        .acquire_owned()
        .await
        .map_err(|_| DirectRelayCryptoFailure::Unavailable)?;
    execute_direct_relay_crypto(permit, work).await
}

pub(super) async fn execute_direct_relay_crypto<T, F>(
    permit: OwnedSemaphorePermit,
    work: F,
) -> Result<T, DirectRelayCryptoFailure>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        // [DIRECT-RECEIPT-SIGNING-COMPLETION 2026-08-31 by Codex] Keep the
        // permit in the worker after request cancellation. This domain owns
        // CPU-only cryptographic work and must never perform I/O or effects.
        let _permit = permit;
        work()
    })
    .await
    .map_err(|_| {
        warn!("[CHAT_PEER] Direct relay crypto worker failed closed");
        DirectRelayCryptoFailure::Unavailable
    })
}

pub(super) fn validate_peer_envelope(
    envelope: &ChatEnvelope,
    now: u64,
) -> Result<(), ChatPeerRelayError> {
    envelope
        .verify_signature()
        .map_err(|_| ChatPeerRelayError::InvalidSignature)?;

    // [PEER-RELAY-REPLAY-WINDOW 2026-08-15 by Codex] Direct compatibility
    // relay is immediate node-to-node work, not a durable replay token. Use
    // the same bounded freshness policy as blind routing so an observed signed
    // ciphertext cannot be admitted to fresh node mailboxes indefinitely.
    validate_relay_timestamp(
        envelope.timestamp,
        now,
        BLIND_RELAY_MAX_ENVELOPE_AGE_SECS,
        BLIND_RELAY_MAX_FUTURE_SKEW_SECS,
    )
    .map_err(|error| match error {
        RelayTimestampError::Expired => ChatPeerRelayError::TimestampExpired,
        RelayTimestampError::InFuture => ChatPeerRelayError::TimestampInFuture,
    })?;

    let encoded = encode_envelope(envelope).map_err(|_| ChatPeerRelayError::Serialization)?;
    if encoded.len() > MAX_PEER_CHAT_ENVELOPE_BYTES {
        return Err(ChatPeerRelayError::EnvelopeTooLarge {
            size: encoded.len(),
        });
    }

    Ok(())
}

pub(super) fn validate_relay_timestamp(
    timestamp: u64,
    now: u64,
    max_age_secs: u64,
    max_future_skew_secs: u64,
) -> Result<(), RelayTimestampError> {
    if timestamp > now.saturating_add(max_future_skew_secs) {
        return Err(RelayTimestampError::InFuture);
    }
    if now.saturating_sub(timestamp) > max_age_secs {
        return Err(RelayTimestampError::Expired);
    }
    Ok(())
}

pub(super) async fn send_envelope_to_session(
    envelope: &ChatEnvelope,
    session: &Arc<Session>,
    udp: &Arc<UdpTransport>,
) -> bool {
    let msg = MemChainMessage::ChatRelay(envelope.clone());
    let plaintext = match encode_memchain(&msg) {
        Ok(plaintext) => plaintext,
        Err(_error) => {
            warn!(
                reason = "encode_client_relay_message_failed",
                "[CHAT_PEER] Failed to encode client relay message"
            );
            return false;
        }
    };

    let crypto = DefaultTransportCrypto::new();
    let counter = session.next_tx_counter();
    let mut encrypted = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
    let len = match crypto.encrypt(
        &session.session_key,
        counter,
        session.id.as_bytes(),
        &plaintext,
        &mut encrypted,
    ) {
        Ok(len) => len,
        Err(_error) => {
            warn!(
                reason = "encrypt_client_relay_message_failed",
                "[CHAT_PEER] Failed to encrypt client relay message"
            );
            return false;
        }
    };
    encrypted.truncate(len);

    let packet = DataPacket::new(*session.id.as_bytes(), counter, encrypted);
    let bytes = encode_data_packet(&packet).to_vec();
    udp.send(&bytes, &session.endpoint()).await.is_ok()
}
