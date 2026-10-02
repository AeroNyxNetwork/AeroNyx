// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn direct_peer_relay_receipt_binds_request_target_signature_and_freshness() {
    // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] Every receipt is
    // usable only for the exact authenticated request and selected node,
    // within the short online acknowledgement window.
    let previous_hop = IdentityKeyPair::generate();
    let target = IdentityKeyPair::generate();
    let observed_at = 1_800_000_000u64;
    let request =
        PeerChatRelayRequestV2::sign(signed_envelope_at(observed_at), &previous_hop).unwrap();
    let receipt = PeerChatRelayReceiptV2::accepted(
        request.request_commitment().unwrap(),
        observed_at,
        &target,
    );

    assert_eq!(
        receipt.verify_expected(&request, &target.public_key_bytes(), observed_at),
        Ok(())
    );

    let wrong_target = IdentityKeyPair::generate();
    assert_eq!(
        receipt.verify_expected(&request, &wrong_target.public_key_bytes(), observed_at),
        Err("receipt_binding_invalid")
    );

    let other_request = PeerChatRelayRequestV2::sign(
        signed_envelope_at(observed_at.saturating_add(1)),
        &previous_hop,
    )
    .unwrap();
    assert_eq!(
        receipt.verify_expected(&other_request, &target.public_key_bytes(), observed_at),
        Err("receipt_binding_invalid")
    );

    let mut forged = receipt.clone();
    forged.signature[0] ^= 0x01;
    assert_eq!(
        forged.verify_expected(&request, &target.public_key_bytes(), observed_at),
        Err("receipt_signature_invalid")
    );

    let expired = PeerChatRelayReceiptV2::accepted(
        request.request_commitment().unwrap(),
        observed_at.saturating_sub(PEER_CHAT_RELAY_RECEIPT_MAX_AGE_SECS + 1),
        &target,
    );
    assert_eq!(
        expired.verify_expected(&request, &target.public_key_bytes(), observed_at),
        Err("receipt_timestamp_expired")
    );

    let future = PeerChatRelayReceiptV2::accepted(
        request.request_commitment().unwrap(),
        observed_at.saturating_add(PEER_CHAT_RELAY_RECEIPT_MAX_FUTURE_SKEW_SECS + 1),
        &target,
    );
    assert_eq!(
        future.verify_expected(&request, &target.public_key_bytes(), observed_at),
        Err("receipt_timestamp_in_future")
    );

    let encoded = serde_json::to_value(PeerChatRelayResponseV2 {
        relay: durable_peer_acceptance_response(),
        receipt: Some(receipt),
    })
    .unwrap();
    let object = encoded.as_object().unwrap();
    assert_eq!(object.len(), 5);
    assert!(object.contains_key("receipt"));
    for forbidden in [
        "receiver",
        "message_id",
        "online",
        "endpoint",
        "payload_size",
    ] {
        assert!(!object.contains_key(forbidden));
    }
}

#[test]
fn delayed_receipt_validation_uses_response_observation_time() {
    // [RELAY-RESPONSE-OBSERVATION-TIME 2026-08-11 by Codex] A valid ACK
    // produced after a long request must be compared with response time,
    // not the stale ingress timestamp. Backdating only the monotonic test
    // clock keeps this regression deterministic and fast.
    let started_at = 1_800_000_000u64;
    let request_started_at = Instant::now() - Duration::from_secs(31);
    let observed_at = blind_relay_response_observed_at(started_at, &request_started_at);
    assert!(observed_at >= started_at + 31);

    let terminal = IdentityKeyPair::generate();
    let terminal_node_id = terminal.public_key_bytes();
    let route_id = [0x81u8; 16];
    let payload = b"opaque delayed message relay payload";
    let ack = PeerBlindRelayResponse {
        accepted: true,
        terminal: true,
        forwarded: false,
        ttl_remaining: 0,
        reason: Some("onion_terminal_delivered".to_string()),
        delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted_for_purpose(
            route_id,
            payload,
            OnionRoutePurpose::MessageRelay,
            started_at + 31,
            &terminal,
        )),
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };

    assert_eq!(
        validate_downstream_delivery_receipt(&ack, &route_id, &terminal_node_id, started_at,),
        Err("receipt_timestamp_in_future")
    );
    assert!(
        validate_downstream_delivery_receipt(&ack, &route_id, &terminal_node_id, observed_at,)
            .is_ok()
    );
}

#[test]
fn three_hop_forwarded_ack_accepts_downstream_terminal_receipt() {
    let now = 1_800_000_100;
    let route_id = [0xa1; 16];
    let immediate_middle = IdentityKeyPair::generate();
    let downstream_terminal = IdentityKeyPair::generate();
    assert_ne!(
        immediate_middle.public_key_bytes(),
        downstream_terminal.public_key_bytes()
    );

    let ack = PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining: 1,
        reason: Some("onion_forwarded".to_string()),
        delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
            route_id,
            [0xb2; 32],
            now,
            &downstream_terminal,
        )),
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };

    validate_downstream_delivery_receipt(
        &ack,
        &route_id,
        &immediate_middle.public_key_bytes(),
        now,
    )
    .expect("an intermediate ACK may propagate a deeper terminal receipt");
}

#[test]
fn direct_terminal_ack_requires_immediate_next_hop_receipt_signer() {
    let now = 1_800_000_100;
    let route_id = [0xc3; 16];
    let immediate_terminal = IdentityKeyPair::generate();
    let wrong_terminal = IdentityKeyPair::generate();
    let ack = PeerBlindRelayResponse {
        accepted: true,
        terminal: true,
        forwarded: false,
        ttl_remaining: 1,
        reason: Some("onion_terminal_delivered".to_string()),
        delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted(
            route_id,
            [0xd4; 32],
            now,
            &wrong_terminal,
        )),
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };

    assert_eq!(
        validate_downstream_delivery_receipt(
            &ack,
            &route_id,
            &immediate_terminal.public_key_bytes(),
            now,
        ),
        Err("terminal_receipt_signer_mismatch")
    );
}

#[test]
fn signed_failure_receipt_is_exact_fresh_and_legacy_compatible() {
    let now = 1_800_000_200;
    let previous_hop = IdentityKeyPair::generate();
    let responder = IdentityKeyPair::generate();
    let other_responder = IdentityKeyPair::generate();
    let envelope = BlindRelayEnvelope {
        route_id: [0xb1; 16],
        next_hop: responder.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque failure receipt request".to_vec(),
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);
    let request = PeerBlindRelayRequest {
        envelope: envelope.clone(),
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let signed_receipt = |failed_at, signer: &IdentityKeyPair| {
        BlindRelayFailureReceipt::failed(
            envelope.route_id,
            BlindRelayFailureReceipt::request_commitment(&envelope),
            "forward_failed",
            failed_at,
            signer,
        )
    };
    let failure_ack = |receipt| PeerBlindRelayResponse {
        accepted: false,
        terminal: false,
        forwarded: false,
        ttl_remaining: 0,
        reason: Some("forward_failed".to_string()),
        delivery_receipt: None,
        success_receipt: None,
        failure_receipt: receipt,
        opaque_terminal_response_b64: None,
    };

    let authenticated = failure_ack(Some(signed_receipt(now, &responder)));
    assert_eq!(
        validate_downstream_failure_receipt(
            &authenticated,
            &request,
            &responder.public_key_bytes(),
            now,
            false,
        ),
        Ok(true)
    );
    assert_eq!(
        validate_downstream_failure_receipt(
            &failure_ack(None),
            &request,
            &responder.public_key_bytes(),
            now,
            false,
        ),
        Ok(false),
        "missing receipt remains the explicit mixed-version path"
    );
    assert_eq!(
        validate_downstream_failure_receipt(
            &failure_ack(None),
            &request,
            &responder.public_key_bytes(),
            now,
            true,
        ),
        Err("failure_receipt_required"),
        "an advertised receipt cannot be silently downgraded"
    );
    let legacy_ack: PeerBlindRelayResponse = serde_json::from_value(serde_json::json!({
        "accepted": false,
        "terminal": false,
        "forwarded": false,
        "ttl_remaining": 0,
        "reason": "forward_failed"
    }))
    .expect("legacy ACK without receipt fields must remain decodable");
    assert!(legacy_ack.delivery_receipt.is_none());
    assert!(legacy_ack.failure_receipt.is_none());

    let mut reason_substitution = authenticated.clone();
    reason_substitution.reason = Some("no_route".to_string());
    assert_eq!(
        validate_downstream_failure_receipt(
            &reason_substitution,
            &request,
            &responder.public_key_bytes(),
            now,
            false,
        ),
        Err("failure_receipt_binding_invalid")
    );
    assert_eq!(
        validate_downstream_failure_receipt(
            &failure_ack(Some(signed_receipt(now, &other_responder))),
            &request,
            &responder.public_key_bytes(),
            now,
            false,
        ),
        Err("failure_receipt_binding_invalid")
    );
    assert_eq!(
        validate_downstream_failure_receipt(
            &failure_ack(Some(signed_receipt(
                now - BLIND_RELAY_FAILURE_RECEIPT_MAX_AGE_SECS - 1,
                &responder,
            ))),
            &request,
            &responder.public_key_bytes(),
            now,
            false,
        ),
        Err("failure_receipt_timestamp_expired")
    );
    assert_eq!(
        validate_downstream_failure_receipt(
            &failure_ack(Some(signed_receipt(
                now + BLIND_RELAY_FAILURE_RECEIPT_MAX_FUTURE_SKEW_SECS + 1,
                &responder,
            ))),
            &request,
            &responder.public_key_bytes(),
            now,
            false,
        ),
        Err("failure_receipt_timestamp_in_future")
    );

    let mut contradictory_success = authenticated;
    contradictory_success.accepted = true;
    contradictory_success.terminal = true;
    assert_eq!(
        validate_downstream_delivery_receipt(
            &contradictory_success,
            &envelope.route_id,
            &responder.public_key_bytes(),
            now,
        ),
        Err("unexpected_failure_receipt")
    );
}

#[test]
fn forwarded_ack_rejects_tampered_downstream_receipt() {
    let now = 1_800_000_100;
    let route_id = [0xe5; 16];
    let immediate_middle = IdentityKeyPair::generate();
    let downstream_terminal = IdentityKeyPair::generate();
    let mut receipt =
        BlindRelayDeliveryReceipt::accepted(route_id, [0xf6; 32], now, &downstream_terminal);
    receipt.payload_commitment[0] ^= 0xff;
    let ack = PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining: 1,
        reason: Some("onion_forwarded".to_string()),
        delivery_receipt: Some(receipt),
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };

    assert_eq!(
        validate_downstream_delivery_receipt(
            &ack,
            &route_id,
            &immediate_middle.public_key_bytes(),
            now,
        ),
        Err("receipt_signature_invalid")
    );
}

#[test]
fn validate_peer_envelope_rejects_tampered_ciphertext() {
    let mut envelope = signed_envelope();
    envelope.ciphertext.push(0x44);

    assert!(matches!(
        validate_peer_envelope(&envelope, now_secs()),
        Err(ChatPeerRelayError::InvalidSignature)
    ));
}

#[test]
fn validate_peer_envelope_enforces_bounded_replay_window() {
    let now = 1_800_000_000u64;
    assert!(validate_peer_envelope(
        &signed_envelope_at(now.saturating_sub(BLIND_RELAY_MAX_ENVELOPE_AGE_SECS)),
        now,
    )
    .is_ok());
    assert!(matches!(
        validate_peer_envelope(
            &signed_envelope_at(
                now.saturating_sub(BLIND_RELAY_MAX_ENVELOPE_AGE_SECS)
                    .saturating_sub(1)
            ),
            now,
        ),
        Err(ChatPeerRelayError::TimestampExpired)
    ));
    assert!(validate_peer_envelope(
        &signed_envelope_at(now.saturating_add(BLIND_RELAY_MAX_FUTURE_SKEW_SECS)),
        now,
    )
    .is_ok());
    assert!(matches!(
        validate_peer_envelope(
            &signed_envelope_at(
                now.saturating_add(BLIND_RELAY_MAX_FUTURE_SKEW_SECS)
                    .saturating_add(1)
            ),
            now,
        ),
        Err(ChatPeerRelayError::TimestampInFuture)
    ));
}

#[tokio::test]
async fn peer_relay_endpoint_stores_offline_receiver_message() {
    let (relay, path) = temp_chat_relay("chat-peer");
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let peer_store = Arc::new(PeerStore::new());
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let http_client = Arc::new(reqwest::Client::new());
    let envelope = signed_envelope();
    let receiver = envelope.receiver;

    let app = build_chat_peer_router(
        Some(Arc::clone(&relay)),
        sessions,
        udp,
        peer_store,
        node_identity,
        http_client,
        None,
    );
    let body = serde_json::to_vec(&PeerChatRelayRequest { envelope }).unwrap();
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay")
                .header("content-type", "application/json")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    let retry = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(retry.status(), StatusCode::OK);
    let response: PeerChatRelayResponse = serde_json::from_slice(
        &to_bytes(response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap(),
    )
    .unwrap();
    let retry: PeerChatRelayResponse = serde_json::from_slice(
        &to_bytes(retry.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap(),
    )
    .unwrap();
    let privacy_safe_acceptance = durable_peer_acceptance_response();
    assert_eq!(response, privacy_safe_acceptance);
    assert_eq!(retry, privacy_safe_acceptance);
    let (messages, has_more) = relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .expect("pending message should be readable");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    let status = relay.peer_status();
    assert_eq!(status.inbound_accepted_total, 2);
    assert_eq!(status.inbound_duplicate_total, 1);

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn peer_relay_v2_rejects_tampering_before_durable_storage() {
    // [DIRECT-RELAY-AUTH-V2 2026-08-15 by Codex] A forged previous-hop
    // claim must not reach the durable queue, while the exact signed
    // request remains accepted through the same relay processing path.
    let (relay, path) = temp_chat_relay_with_rates(
        "chat-peer-auth-v2",
        DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE,
        1,
    );
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let previous_hop = IdentityKeyPair::generate();
    let envelope = signed_envelope();
    let receiver = envelope.receiver;
    let request = PeerChatRelayRequestV2::sign(envelope, &previous_hop).unwrap();
    let mut tampered = request.clone();
    tampered.envelope.ciphertext[0] ^= 0x01;

    let target_identity = Arc::new(IdentityKeyPair::generate());
    let target_node_id = target_identity.public_key_bytes();
    let app = build_chat_peer_router(
        Some(Arc::clone(&relay)),
        sessions,
        udp,
        Arc::new(PeerStore::new()),
        Arc::clone(&target_identity),
        Arc::new(reqwest::Client::new()),
        None,
    );
    let tampered_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay-v2")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&tampered).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(tampered_response.status(), StatusCode::UNAUTHORIZED);
    assert!(relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .unwrap()
        .0
        .is_empty());
    let rejected_status = relay.peer_status();
    assert_eq!(rejected_status.inbound_rejected_total, 1);
    assert_eq!(
        rejected_status.last_inbound_failure_reason.as_deref(),
        Some("peer_auth_invalid")
    );

    let accepted_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay-v2")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(accepted_response.status(), StatusCode::OK);
    let accepted_response: PeerChatRelayResponseV2 = serde_json::from_slice(
        &to_bytes(accepted_response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(accepted_response.relay, durable_peer_acceptance_response());
    accepted_response
        .receipt
        .as_ref()
        .expect("authenticated direct relay should return signed custody evidence")
        .verify_expected(&request, &target_node_id, now_secs())
        .expect("receipt should bind the exact request to the target node");
    assert_eq!(
        relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .unwrap()
            .0
            .len(),
        1
    );
    let replayed_response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay-v2")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(replayed_response.status(), StatusCode::OK);
    let replayed_response: PeerChatRelayResponseV2 = serde_json::from_slice(
        &to_bytes(replayed_response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(replayed_response, accepted_response);
    let status = relay.peer_status();
    assert_eq!(status.inbound_rejected_total, 1);
    assert_eq!(status.inbound_accepted_total, 2);
    assert_eq!(status.inbound_duplicate_total, 1);

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn peer_relay_v3_rejects_request_signed_for_another_target() {
    // [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] A request that
    // is fully valid for target A must not become authenticated work for
    // target B. The rejected attempt also must not consume B's per-node
    // authenticated quota or reach its durable pending store.
    let (relay, path) = temp_chat_relay_with_rates(
        "chat-peer-target-binding-v3",
        DEFAULT_PEER_RELAY_REQUESTS_PER_MINUTE,
        1,
    );
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let previous_hop = IdentityKeyPair::generate();
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let target_node_id = target_identity.public_key_bytes();
    let other_target_node_id = IdentityKeyPair::generate().public_key_bytes();
    let envelope = signed_envelope();
    let receiver = envelope.receiver;
    let wrong_target_request =
        PeerChatRelayRequestV3::sign(envelope.clone(), other_target_node_id, &previous_hop)
            .unwrap();
    assert!(wrong_target_request.verify_for_target(&other_target_node_id));

    let app = build_chat_peer_router(
        Some(Arc::clone(&relay)),
        sessions,
        udp,
        Arc::new(PeerStore::new()),
        Arc::clone(&target_identity),
        Arc::new(reqwest::Client::new()),
        None,
    );
    let rejected = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay-v3")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&wrong_target_request).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::UNAUTHORIZED);
    assert!(relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .unwrap()
        .0
        .is_empty());

    let accepted_request =
        PeerChatRelayRequestV3::sign(envelope, target_node_id, &previous_hop).unwrap();
    let expected_commitment = accepted_request.request_commitment().unwrap();
    let accepted = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay-v3")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&accepted_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(accepted.status(), StatusCode::OK);
    let accepted: PeerChatRelayResponseV2 = serde_json::from_slice(
        &to_bytes(accepted.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap(),
    )
    .unwrap();
    accepted
        .receipt
        .as_ref()
        .expect("target-bound durable acceptance should be signed")
        .verify_expected_commitment(&expected_commitment, &target_node_id, now_secs())
        .expect("receipt should bind the exact v3 request to the target");
    assert_eq!(
        relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .unwrap()
            .0
            .len(),
        1
    );

    // [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex] Simulate an
    // ACK lost after durable custody. The byte-identical retry bypasses
    // the already-consumed per-node quota and returns the exact signed ACK
    // without inserting or delivering the encrypted envelope twice.
    let replayed = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay-v3")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&accepted_request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(replayed.status(), StatusCode::OK);
    let replayed: PeerChatRelayResponseV2 = serde_json::from_slice(
        &to_bytes(replayed.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(replayed, accepted);
    assert_eq!(
        relay
            .pull_pending(&receiver, 0, &[0u8; 16], 10)
            .unwrap()
            .0
            .len(),
        1
    );
    let status = relay.peer_status();
    assert_eq!(status.inbound_rejected_total, 1);
    assert_eq!(status.inbound_accepted_total, 2);
    assert_eq!(status.inbound_duplicate_total, 1);

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn peer_relay_rate_limit_rejects_before_duplicate_processing() {
    let (relay, path) = temp_chat_relay_with_peer_rate("chat-peer-rate-limit", 1);
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let envelope = signed_envelope();
    let body = serde_json::to_vec(&PeerChatRelayRequest { envelope }).unwrap();
    let app = build_chat_peer_router(
        Some(Arc::clone(&relay)),
        sessions,
        udp,
        Arc::new(PeerStore::new()),
        Arc::new(IdentityKeyPair::generate()),
        Arc::new(reqwest::Client::new()),
        None,
    );

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay")
                .header("content-type", "application/json")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    let second = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/relay")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let status = relay.peer_status();
    assert_eq!(status.inbound_accepted_total, 1);
    assert_eq!(status.inbound_duplicate_total, 0);
    assert_eq!(status.inbound_rejected_total, 1);
    assert_eq!(
        status.last_inbound_failure_reason.as_deref(),
        Some("rate_limited")
    );

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn advertised_failure_receipt_omission_penalizes_exact_next_hop_surface() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
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
                        failure_receipt: None,
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
    let current_node = Arc::new(IdentityKeyPair::generate());
    let next_hop_identity = IdentityKeyPair::generate();
    let next_hop_node_id = next_hop_identity.public_key_bytes();
    let legacy_descriptor =
        signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint.clone(), now, now + 300);
    let advertised_descriptor = SignedNodeDescriptor::sign(
        legacy_descriptor
            .descriptor
            .with_protocol_features([NodeProtocolFeature::BlindRelayFailureReceiptV1]),
        &next_hop_identity,
    )
    .unwrap();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(advertised_descriptor.clone(), now, "gossip_snapshot")
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_node_id, now);
    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&current_node),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let request = PeerBlindRelayRequest {
        envelope: BlindRelayEnvelope {
            route_id: [0x6bu8; 16],
            next_hop: next_hop_node_id,
            ttl: 1,
            encrypted_blob: b"opaque downgraded failure receipt request".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(current_node.as_ref()),
        previous_hop_node_id: current_node.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };

    let result = forward_blind_relay_with_retry(
        &state,
        &blind_peer_relay_url(&endpoint).unwrap(),
        &advertised_descriptor,
        prepare_blind_relay_forward_request(request)
            .await
            .expect("prepare downgraded failure receipt request"),
        now,
    )
    .await;
    server.abort();

    assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
    let route_status = peer_store.route_candidate_status(now + 5);
    let route_row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
        .expect("chat relay row should remain visible");
    assert_eq!(route_row.route_failure_count, 1);
    assert_eq!(route_row.route_consecutive_failures, 1);
    assert_eq!(
        route_row.last_route_failure_reason.as_deref(),
        Some("failure_receipt_downgrade")
    );
}

#[tokio::test]
async fn invalid_failure_receipt_penalizes_immediate_next_hop_protocol() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let wrong_signer = Arc::new(IdentityKeyPair::generate());
    let wrong_signer_for_route = Arc::clone(&wrong_signer);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            let wrong_signer = Arc::clone(&wrong_signer_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                let receipt = BlindRelayFailureReceipt::failed(
                    request.envelope.route_id,
                    BlindRelayFailureReceipt::request_commitment(&request.envelope),
                    "forward_failed",
                    now_secs(),
                    wrong_signer.as_ref(),
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
                        failure_receipt: Some(receipt),
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
    let current_node = Arc::new(IdentityKeyPair::generate());
    let next_hop_identity = IdentityKeyPair::generate();
    let next_hop_node_id = next_hop_identity.public_key_bytes();
    let descriptor =
        signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint.clone(), now, now + 300);
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(descriptor.clone(), now, "gossip_snapshot")
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_node_id, now);
    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&current_node),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let request = PeerBlindRelayRequest {
        envelope: BlindRelayEnvelope {
            route_id: [0x6au8; 16],
            next_hop: next_hop_node_id,
            ttl: 1,
            encrypted_blob: b"opaque invalid failure receipt request".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(current_node.as_ref()),
        previous_hop_node_id: current_node.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };

    let result = forward_blind_relay_with_retry(
        &state,
        &blind_peer_relay_url(&endpoint).unwrap(),
        &descriptor,
        prepare_blind_relay_forward_request(request)
            .await
            .expect("prepare invalid failure receipt request"),
        now,
    )
    .await;
    server.abort();

    assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 1);
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.forward_failed, 1);
    let route_status = peer_store.route_candidate_status(now + 5);
    let route_row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
        .expect("chat relay row should remain visible");
    assert_eq!(route_row.route_failure_count, 1);
    assert_eq!(route_row.route_consecutive_failures, 1);
    assert_eq!(
        route_row.last_route_failure_reason.as_deref(),
        Some("failure_receipt_invalid")
    );
}
