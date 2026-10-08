// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn downstream_success_ack_requires_exactly_one_delivery_disposition() {
    // [RELAY-ACK-STATE-MACHINE 2026-08-11 by Codex] Receipt-less ACKs stay
    // valid for mixed-version peers only when their state proves one real
    // terminal or forwarding action. No-op and contradictory success
    // shapes must fail before they can become route-health evidence.
    let now = 1_800_000_100;
    let route_id = [0xd5; 16];
    let immediate_next_hop = IdentityKeyPair::generate().public_key_bytes();
    let ack = |accepted, terminal, forwarded| PeerBlindRelayResponse {
        accepted,
        terminal,
        forwarded,
        ttl_remaining: 1,
        reason: None,
        delivery_receipt: None,
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };

    assert_eq!(
        validate_downstream_delivery_receipt(
            &ack(true, false, false),
            &route_id,
            &immediate_next_hop,
            now,
        ),
        Err("invalid_ack_shape")
    );
    assert_eq!(
        validate_downstream_delivery_receipt(
            &ack(true, true, true),
            &route_id,
            &immediate_next_hop,
            now,
        ),
        Err("invalid_ack_shape")
    );
    assert_eq!(
        validate_downstream_delivery_receipt(
            &ack(false, false, false),
            &route_id,
            &immediate_next_hop,
            now,
        ),
        Err("ack_not_accepted")
    );
    assert!(validate_downstream_delivery_receipt(
        &ack(true, true, false),
        &route_id,
        &immediate_next_hop,
        now,
    )
    .is_ok());
    assert!(validate_downstream_delivery_receipt(
        &ack(true, false, true),
        &route_id,
        &immediate_next_hop,
        now,
    )
    .is_ok());
}

#[test]
fn pending_store_capacity_maps_to_retryable_service_unavailable() {
    let capacity = ChatRelayError::PendingMessageQueueFull {
        current: 100,
        limit: 100,
    };
    let mapped = map_pending_store_error(&capacity);

    assert!(matches!(mapped, ChatPeerRelayError::PendingCapacity));
    assert_eq!(mapped.status_code(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(mapped.reason_bucket(), "pending_capacity_exhausted");

    let storage = ChatRelayError::Sqlite(rusqlite::Error::InvalidQuery);
    assert!(matches!(
        map_pending_store_error(&storage),
        ChatPeerRelayError::StoreFailed
    ));

    let oversized = ChatRelayError::MessageTooLarge {
        size: 65_537,
        limit: 65_536,
    };
    assert!(matches!(
        map_pending_store_error(&oversized),
        ChatPeerRelayError::EnvelopeTooLarge { size: 65_537 }
    ));
}

#[test]
fn downstream_domain_errors_preserve_public_status_mapping() {
    assert!(matches!(
        map_blind_vault_put_error(&BlindVaultServiceError::LeaseNotFound),
        BlindRelayError::OnionTerminalPayloadRejected
    ));
    assert!(matches!(
        map_blind_vault_put_error(&BlindVaultServiceError::QuotaExceeded),
        BlindRelayError::OnionTerminalCapacityExhausted
    ));
    assert!(matches!(
        map_blind_vault_put_error(&BlindVaultServiceError::Disabled),
        BlindRelayError::ForwardFailed
    ));
    assert_eq!(
        BlindRelayError::OnionTerminalCapacityExhausted.status_code(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        BlindRelayError::RouteInFlight.status_code(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        BlindRelayError::RouteInFlight.reason_bucket(),
        "route_in_flight"
    );
    assert_eq!(
        BlindRelayError::ReplayCapacity.status_code(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        BlindRelayError::ReplayCapacity.reason_bucket(),
        "replay_capacity"
    );
    assert_eq!(
        BlindRelayError::Backpressure.status_code(),
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(
        BlindRelayError::Backpressure.reason_bucket(),
        "backpressure"
    );
    assert!(matches!(
        BlindRelayError::from(BlindRelayDownstreamFailure::OnionTerminalCapacityExhausted),
        BlindRelayError::OnionTerminalCapacityExhausted
    ));
    assert!(matches!(
        BlindRelayError::from(BlindRelayDownstreamFailure::ForwardFailed),
        BlindRelayError::ForwardFailed
    ));
    assert!(matches!(
        BlindRelayError::from(BlindRelayDownstreamFailure::DownstreamRejected),
        BlindRelayError::DownstreamRejected
    ));
}

#[tokio::test]
async fn peer_routes_reject_oversized_bodies_before_json_deserialization() {
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let peer_store = Arc::new(PeerStore::new());
    let app = build_chat_peer_router(
        None,
        sessions,
        udp,
        Arc::clone(&peer_store),
        Arc::new(IdentityKeyPair::generate()),
        Arc::new(reqwest::Client::new()),
        None,
    );

    let peer_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay")
                .header("content-type", "application/json")
                .body(Body::from(vec![b' '; PEER_CHAT_REQUEST_BODY_MAX_BYTES + 1]))
                .unwrap(),
        )
        .await
        .unwrap();
    let blind_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .body(Body::from(vec![
                    b' ';
                    PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES + 1
                ]))
                .unwrap(),
        )
        .await
        .unwrap();
    let declared_blind_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .header(
                    axum::http::header::CONTENT_LENGTH,
                    (PEER_BLIND_RELAY_REQUEST_BODY_MAX_BYTES + 1).to_string(),
                )
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(peer_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(blind_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        declared_blind_response.status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let blind_stats = peer_store.status(now_secs()).runtime.blind_relay;
    assert_eq!(blind_stats.received, 0, "oversized body reached handler");
    assert_eq!(blind_stats.rejected, 0, "oversized body reached handler");
}

#[tokio::test]
async fn peer_request_in_flight_guard_enforces_backpressure_limit() {
    let (relay, path) = temp_chat_relay("blind-relay-backpressure");
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
        private_recipient_admission: None,
        chat_relay: Some(relay),
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store,
        node_identity: Arc::new(IdentityKeyPair::generate()),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };

    assert!(InFlightRequestGuard::try_acquire(
        &state.blind_relay_in_flight,
        MAX_IN_FLIGHT_BLIND_RELAY_REQUESTS,
    )
    .is_none());

    let app = Router::new()
        .route("/api/chat/peer/blind-relay", post(peer_blind_relay_handler))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            peer_blind_relay_request_gate,
        ))
        .with_state(state);
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .body(Body::from("not-json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn forged_previous_hop_signatures_cannot_poison_node_quarantine() {
    let claimed_previous_hop = IdentityKeyPair::generate();
    let attacker = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
        private_recipient_admission: None,
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&node_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let now = now_secs();
    let claimed_node_id = claimed_previous_hop.public_key_bytes();

    for attempt in 0..=BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD {
        let mut route_id = [0x91u8; 16];
        route_id[0] = u8::try_from(attempt).unwrap_or(u8::MAX);
        let forged = BlindRelayEnvelope {
            route_id,
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque forged-attribution candidate".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&attacker);

        assert!(matches!(
            process_peer_blind_relay(
                state.clone(),
                PeerBlindRelayRequest {
                    envelope: forged,
                    previous_hop_node_id: claimed_node_id,
                    onward_envelope: None,
                    onward_descriptor_hint: None,
                },
            )
            .await,
            Err(BlindRelayError::InvalidSignature)
        ));
    }

    let decision = state
        .blind_relay_abuse_guard
        .observe_request(claimed_node_id, now);
    assert_eq!(decision, BlindRelayAbuseDecision::Allowed);

    let valid = BlindRelayEnvelope {
        route_id: [0xa2u8; 16],
        next_hop: node_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque authenticated previous-hop payload".to_vec(),
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&claimed_previous_hop);
    let response = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope: valid,
            previous_hop_node_id: claimed_node_id,
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .expect("valid claimed previous hop must remain admissible");

    assert!(response.accepted);
    assert!(response.terminal);
    let blind_status = peer_store.status(now).runtime.blind_relay;
    assert_eq!(
        blind_status.rejected,
        u64::from(BLIND_RELAY_PREVIOUS_HOP_FAILURE_THRESHOLD) + 1
    );
    assert_eq!(blind_status.quarantine_started, 0);
}

#[tokio::test]
async fn peer_declared_downstream_failure_does_not_poison_next_hop_reputation() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_identity = Arc::new(IdentityKeyPair::generate());
    let next_hop_identity_for_route = Arc::clone(&next_hop_identity);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            let next_hop_identity = Arc::clone(&next_hop_identity_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                let failure_receipt = BlindRelayFailureReceipt::failed(
                    request.envelope.route_id,
                    BlindRelayFailureReceipt::request_commitment(&request.envelope),
                    "forward_failed",
                    now_secs(),
                    next_hop_identity.as_ref(),
                );
                (
                    StatusCode::BAD_GATEWAY,
                    Json(PeerBlindRelayResponse {
                        accepted: false,
                        terminal: false,
                        forwarded: false,
                        ttl_remaining: 0,
                        reason: Some("forward_failed".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: Some(failure_receipt),
                        opaque_terminal_response_b64: None,
                    }),
                )
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, next_hop_app).await.unwrap();
    });

    let now = now_secs();
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let next_hop_node_id = next_hop_identity.public_key_bytes();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(
                next_hop_identity.as_ref(),
                endpoint,
                now,
                now + 300,
            ),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_node_id, now);

    let state = ChatPeerState {
        private_recipient_admission: None,
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity,
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let envelope = BlindRelayEnvelope {
        route_id: [0x59u8; 16],
        next_hop: next_hop_node_id,
        ttl: 2,
        encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);

    let result = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await;

    server.abort();

    assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.forward_failed, 1);

    let route_status = peer_store.route_candidate_status(now + 5);
    let route_row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
        .expect("chat relay row should remain visible");
    // [DOWNSTREAM-FAILURE-ATTRIBUTION 2026-08-11 by Codex] The peer was
    // reachable and returned a valid bounded error ACK. That is an
    // end-to-end failure signal, not proof that this route surface failed.
    assert!(route_row.routeability_ready);
    assert_eq!(route_row.route_failure_count, 0);
    assert_eq!(route_row.route_consecutive_failures, 0);
    assert!(route_row.last_route_failure_reason.is_none());
    assert!(!route_row.route_quarantined);
    assert!(peer_store.recent_audit_events().iter().all(|event| {
        event.action != "blind_relay_route_health" || event.outcome != "rejected"
    }));
}
