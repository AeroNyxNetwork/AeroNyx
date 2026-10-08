// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn configured_chat_relay_initialization_fails_closed_without_leaking_path() {
    // [CHAT-RELAY-STARTUP-INTEGRITY 2026-08-14 by Codex] A configured
    // durable relay must not disappear behind a healthy node. The returned
    // error is intentionally diagnostic-bucket-only.
    let disabled = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    assert!(disabled.init_chat_relay_service().unwrap().is_none());

    let directory = tempfile::tempdir().unwrap();
    let non_directory = directory.path().join("relay-parent-is-a-file");
    std::fs::write(&non_directory, b"not a directory").unwrap();
    let private_path = non_directory.join("private-relay.sqlite");
    let mut config = ServerConfig::default();
    config.memchain.chat_relay.enabled = true;
    config.memchain.chat_relay.db_path = private_path.display().to_string();
    let server = Server::new(config, IdentityKeyPair::generate(), None);

    let error = server
        .init_chat_relay_service()
        .err()
        .expect("configured Chat Relay storage failure must reject startup");
    let rendered = error.to_string();
    assert!(rendered.contains("Chat Relay initialization failed (sqlite_error)"));
    assert!(!rendered.contains(&private_path.display().to_string()));
    assert!(!rendered.contains("relay-parent-is-a-file"));
}

#[test]
fn signed_delivery_receipt_verification_binds_route_payload_terminal_and_freshness() {
    let now = 1_800_000_000;
    let terminal = IdentityKeyPair::generate();
    let route_id = [0x31; 16];
    let payload = b"opaque terminal payload";
    let payload_commitment = BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
        payload,
        OnionRoutePurpose::MessageRelay,
    );
    let receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
        route_id,
        payload,
        OnionRoutePurpose::MessageRelay,
        now,
        &terminal,
    );

    assert!(Server::verified_delivery_receipt(
        Some(&receipt),
        &route_id,
        &payload_commitment,
        &terminal.public_key_bytes(),
        now + 1,
    ));
    assert!(!Server::verified_delivery_receipt(
        Some(&receipt),
        &[0x32; 16],
        &payload_commitment,
        &terminal.public_key_bytes(),
        now + 1,
    ));
    assert!(!Server::verified_delivery_receipt(
        Some(&receipt),
        &route_id,
        &[0x33; 32],
        &terminal.public_key_bytes(),
        now + 1,
    ));
    assert!(!Server::verified_delivery_receipt(
        Some(&receipt),
        &route_id,
        &payload_commitment,
        &IdentityKeyPair::generate().public_key_bytes(),
        now + 1,
    ));
    assert!(!Server::verified_delivery_receipt(
        Some(&receipt),
        &route_id,
        &payload_commitment,
        &terminal.public_key_bytes(),
        now + BLIND_RELAY_DELIVERY_RECEIPT_MAX_AGE_SECS + 1,
    ));

    let legacy_receipt = BlindRelayDeliveryReceipt::accepted(
        route_id,
        BlindRelayDeliveryReceipt::payload_commitment(payload),
        now,
        &terminal,
    );
    assert!(!Server::verified_delivery_receipt(
        Some(&legacy_receipt),
        &route_id,
        &payload_commitment,
        &terminal.public_key_bytes(),
        now + 1,
    ));
}

#[tokio::test]
async fn authenticated_chat_records_privacy_safe_preflight_failure() {
    let directory = tempfile::tempdir().expect("relay status temp directory");
    let mut relay_config = ChatRelayConfig::default();
    relay_config.enabled = true;
    relay_config.db_path = directory
        .path()
        .join("relay-status.sqlite3")
        .to_string_lossy()
        .into_owned();
    let relay = crate::services::ChatRelayService::new(relay_config, [0x61; 32])
        .expect("initialize relay status service");
    let source = IdentityKeyPair::generate();
    let store = PeerStore::new();
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("build isolated relay test client");

    let outcome = Server::relay_authenticated_chat_over_onion_paths(
        Some(&client),
        Some(&relay),
        &store,
        &source,
        &source.public_key_bytes(),
        &signed_test_chat_envelope(unix_now_secs()),
        None,
    )
    .await;

    assert!(!outcome.delivered());
    // [RELAY-SELECTION-DIAGNOSTICS 2026-08-15 by Codex] A preflight miss
    // advances both backward-compatible aggregate and authenticated-onion
    // evidence with only a stable reason bucket.
    let status = relay.peer_status();
    assert_eq!(status.outbound_rounds, 1);
    assert_eq!(status.last_outbound_attempted, 0);
    assert_eq!(status.last_outbound_status.as_deref(), Some("failed"));
    assert_eq!(
        status.last_outbound_failure_reason.as_deref(),
        Some("no_receipt_capable_terminal")
    );
    assert_eq!(status.authenticated_onion_outbound.rounds, 1);
    assert_eq!(
        status.authenticated_onion_outbound.last_status.as_deref(),
        Some("failed")
    );
    assert_eq!(
        status
            .authenticated_onion_outbound
            .last_failure_reason
            .as_deref(),
        Some("no_receipt_capable_terminal")
    );
    assert_eq!(status.direct_peer_outbound.rounds, 0);
}

#[tokio::test]
async fn authenticated_chat_rejects_receipt_after_terminal_surface_rotation() {
    let now = unix_now_secs();
    let source = IdentityKeyPair::generate();
    let chat_sender = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();
    let terminal_node_id = terminal_identity.public_key_bytes();
    let middle_identity = IdentityKeyPair::generate();
    let middle_node_id = middle_identity.public_key_bytes();

    let mut envelope = ChatEnvelope {
        message_id: [0x45; 16],
        sender: chat_sender.public_key_bytes(),
        receiver: [0x46; 32],
        timestamp: now,
        ciphertext: b"opaque app ciphertext during rotation".to_vec(),
        nonce: [0x47; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    envelope.signature = chat_sender.sign(&envelope.sign_data());
    let terminal_payload = encode_envelope(&envelope).unwrap();

    let mut terminal_descriptor = NodeDescriptor::new(
        terminal_node_id,
        now,
        now,
        now + 300,
        "test-receipt-terminal-a",
    )
    .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
    terminal_descriptor.public_endpoint = Some("http://127.0.1.1:9".to_string());
    terminal_descriptor.capabilities = vec![NodeCapability::ChatRelay];
    let terminal_descriptor =
        SignedNodeDescriptor::sign(terminal_descriptor, &terminal_identity).unwrap();

    let mut rotated_terminal_body = terminal_descriptor.descriptor.clone();
    rotated_terminal_body.sequence = terminal_descriptor.sequence().saturating_add(1);
    rotated_terminal_body.public_endpoint = Some("http://127.0.2.1:9".to_string());
    let rotated_terminal =
        SignedNodeDescriptor::sign(rotated_terminal_body, &terminal_identity).unwrap();

    let store = Arc::new(PeerStore::new());
    store
        .upsert_verified(terminal_descriptor.clone(), now)
        .unwrap();
    store.record_route_forward_success(&terminal_node_id, now);
    store.record_purpose_bound_delivery_receipt_capability(&terminal_node_id, now);

    let terminal_receipt_identity = terminal_identity.clone();
    let store_for_request = Arc::clone(&store);
    let relay = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let terminal_receipt_identity = terminal_receipt_identity.clone();
            let terminal_payload = terminal_payload.clone();
            let rotated_terminal = rotated_terminal.clone();
            let store_for_request = Arc::clone(&store_for_request);
            async move {
                // [CLIENT-DELIVERY-ATOMIC-ROUTE-EVIDENCE 2026-08-11 by Codex]
                // Simulate a signed endpoint rotation after route selection
                // but before the old route's valid receipt reaches source.
                store_for_request
                    .upsert_verified(rotated_terminal, unix_now_secs())
                    .unwrap();
                Json(PeerBlindRelayResponse {
                    accepted: true,
                    terminal: false,
                    forwarded: true,
                    ttl_remaining: 1,
                    reason: Some("onion_forwarded".to_string()),
                    delivery_receipt: Some(BlindRelayDeliveryReceipt::accepted_for_purpose(
                        request.envelope.route_id,
                        &terminal_payload,
                        OnionRoutePurpose::MessageRelay,
                        unix_now_secs(),
                        &terminal_receipt_identity,
                    )),
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: None,
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let middle_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let relay_server = tokio::spawn(async move {
        axum::serve(listener, relay).await.unwrap();
    });

    let mut middle_descriptor =
        NodeDescriptor::new(middle_node_id, now, now, now + 300, "test-receipt-middle")
            .with_x25519_kem(middle_identity.x25519_public_key_bytes());
    middle_descriptor.public_endpoint = Some(middle_endpoint);
    middle_descriptor.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    let middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor, &middle_identity).unwrap();
    store
        .upsert_verified(middle_descriptor.clone(), now)
        .unwrap();
    store.record_route_forward_success(&middle_node_id, now);
    store.record_purpose_bound_delivery_receipt_capability(&middle_node_id, now);

    let outcome = Server::relay_authenticated_chat_over_onion_paths(
        Some(&reqwest::Client::new()),
        None,
        store.as_ref(),
        &source,
        &source.public_key_bytes(),
        &envelope,
        None,
    )
    .await;
    relay_server.abort();

    assert_eq!(outcome.attempted_paths, 1);
    assert_eq!(outcome.verified_receipts, 0);
    assert!(!outcome.delivered());
    let quality = store.status(unix_now_secs()).blind_relay_quality;
    assert_eq!(quality.verified_client_onion_deliveries, 0);
    assert_eq!(quality.delivery_receipt_capable_peers, 1);
    assert!(!quality.real_relay_ready);
    assert!(!store.take_client_delivery_cache_dirty());
}

#[tokio::test]
async fn authenticated_chat_rejects_receipt_capable_hops_on_same_network() {
    let now = unix_now_secs();
    let source = IdentityKeyPair::generate();
    let sender = IdentityKeyPair::generate();
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();
    let middle_node_id = middle_identity.public_key_bytes();
    let terminal_node_id = terminal_identity.public_key_bytes();

    let mut envelope = ChatEnvelope {
        message_id: [0x51; 16],
        sender: sender.public_key_bytes(),
        receiver: [0x52; 32],
        timestamp: now,
        ciphertext: b"opaque app ciphertext".to_vec(),
        nonce: [0x53; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    envelope.signature = sender.sign(&envelope.sign_data());

    let mut middle = NodeDescriptor::new(
        middle_node_id,
        now,
        now,
        now + 300,
        "test-collocated-middle",
    )
    .with_x25519_kem(middle_identity.x25519_public_key_bytes());
    middle.public_endpoint = Some("http://203.0.113.10:8422".to_string());
    middle.capabilities = vec![NodeCapability::OnionMiddle];
    let middle = SignedNodeDescriptor::sign(middle, &middle_identity).unwrap();

    let mut terminal = NodeDescriptor::new(
        terminal_node_id,
        now,
        now,
        now + 300,
        "test-collocated-terminal",
    )
    .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
    terminal.public_endpoint = Some("http://203.0.113.200:8422".to_string());
    terminal.capabilities = vec![NodeCapability::ChatRelay];
    let terminal = SignedNodeDescriptor::sign(terminal, &terminal_identity).unwrap();

    let store = PeerStore::new();
    store.upsert_verified(middle, now).unwrap();
    store.upsert_verified(terminal, now).unwrap();
    for node_id in [middle_node_id, terminal_node_id] {
        store.record_route_forward_success(&node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&node_id, now);
    }

    let outcome = Server::relay_authenticated_chat_over_onion_paths(
        Some(&reqwest::Client::new()),
        None,
        &store,
        &source,
        &source.public_key_bytes(),
        &envelope,
        None,
    )
    .await;

    assert!(!outcome.delivered());
    assert!(!outcome.fully_replicated());
    assert_eq!(
        store
            .status(unix_now_secs())
            .blind_relay_quality
            .verified_client_onion_deliveries,
        0,
    );
}

#[test]
fn local_capability_status_is_ready_when_chat_relay_is_configured_and_advertised() {
    let mut config = ServerConfig::default();
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.memchain.chat_relay.enabled = true;

    let status = Server::discovery_local_capability_status_for(&config);

    assert_eq!(status.status, "ready");
    assert!(status.chat_relay_configured);
    assert!(status.blind_relay_endpoint_ready);
    assert!(status.chat_relay_runtime_ready);
    assert!(status.safe_to_advertise_chat_relay);
    assert!(status.advertised_chat_relay_capability);
    assert!(status.capability_config_consistent);
}

#[test]
fn local_capability_status_reports_disabled_without_chat_relay_config() {
    let mut config = ServerConfig::default();
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());

    let status = Server::discovery_local_capability_status_for(&config);

    assert_eq!(status.status, "disabled");
    assert!(!status.chat_relay_configured);
    assert!(status.blind_relay_endpoint_ready);
    assert!(!status.chat_relay_runtime_ready);
    assert!(!status.safe_to_advertise_chat_relay);
    assert!(!status.advertised_chat_relay_capability);
    assert!(status.capability_config_consistent);
}

#[test]
fn local_capability_status_reports_misconfigured_when_chat_relay_lacks_peer_api() {
    let mut config = ServerConfig::default();
    config.memchain.chat_relay.enabled = true;

    let status = Server::discovery_local_capability_status_for(&config);

    assert_eq!(status.status, "misconfigured");
    assert!(status.chat_relay_configured);
    assert!(!status.blind_relay_endpoint_ready);
    assert!(status.chat_relay_runtime_ready);
    assert!(!status.safe_to_advertise_chat_relay);
    assert!(!status.advertised_chat_relay_capability);
    assert!(status.capability_config_consistent);
}

#[test]
fn local_capability_status_reports_misconfigured_when_chat_relay_runtime_is_missing() {
    let mut config = ServerConfig::default();
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.memchain.chat_relay.enabled = true;

    let status = Server::discovery_local_capability_status_for_runtime(&config, false);

    assert_eq!(status.status, "misconfigured");
    assert!(status.chat_relay_configured);
    assert!(status.blind_relay_endpoint_ready);
    assert!(!status.chat_relay_runtime_ready);
    assert!(!status.safe_to_advertise_chat_relay);
    assert!(!status.advertised_chat_relay_capability);
    assert!(status.capability_config_consistent);
    assert!(status
        .advertisement_blockers
        .contains(&"chat_relay_runtime_not_ready"));
}

#[tokio::test]
async fn discovered_chat_relay_peer_receives_encrypted_envelope_fanout() {
    let received = Arc::new(AtomicUsize::new(0));
    let received_for_handler = Arc::clone(&received);
    let app = Router::new().route(
        "/api/chat/peer/relay",
        post(move |Json(request): Json<PeerChatRelayRequest>| {
            let received_for_handler = Arc::clone(&received_for_handler);
            async move {
                assert_eq!(request.envelope.message_id, [0x55; 16]);
                received_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                Json(PeerChatRelayResponse {
                    accepted: true,
                    duplicate: false,
                    delivered_online: 0,
                    stored_pending: true,
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let peer_store = PeerStore::new();
    let peer_descriptor = signed_chat_relay_peer_descriptor(endpoint, now, now + 300);
    let peer_node_id = peer_descriptor.node_id();
    let peer_prefix = hex::encode(&peer_node_id[..4]);
    peer_store
        .upsert_verified(peer_descriptor, now)
        .expect("mock peer descriptor should verify");
    peer_store.record_route_forward_success(&peer_node_id, now);

    let client = reqwest::Client::new();
    let source_identity = IdentityKeyPair::generate();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&client),
        None,
        &peer_store,
        &source_identity,
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 1);
    assert_eq!(received.load(AtomicOrdering::SeqCst), 1);
    let route_status = peer_store.route_candidate_status(now + 1);
    let row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == peer_prefix)
        .expect("mock peer should remain in route candidate status");
    assert_eq!(row.route_health, "healthy");
    assert_eq!(row.route_consecutive_failures, 0);
    // [FLAKY-SECOND-ROLLOVER 2026-10-09 by Claude] Success is recorded at the
    // real completion time, which can fall in the second after `now` was
    // taken; 1 of 20 Linux runs failed on exactly that boundary.
    let recorded = row.last_route_success_at.expect("route success timestamp");
    assert!(
        (now..=unix_now_secs()).contains(&recorded),
        "route success recorded outside the request window"
    );
    mock_peer.abort();
}

#[tokio::test]
async fn discovered_chat_relay_uses_authenticated_v2_when_advertised() {
    let authenticated_received = Arc::new(AtomicUsize::new(0));
    let legacy_received = Arc::new(AtomicUsize::new(0));
    let authenticated_for_handler = Arc::clone(&authenticated_received);
    let legacy_for_handler = Arc::clone(&legacy_received);
    let app = Router::new()
        .route(
            "/api/chat/peer/relay-v2",
            post(move |Json(request): Json<PeerChatRelayRequestV2>| {
                let authenticated_for_handler = Arc::clone(&authenticated_for_handler);
                async move {
                    assert!(request.verify_previous_hop());
                    assert_eq!(request.envelope.message_id, [0x55; 16]);
                    authenticated_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerChatRelayResponse {
                        accepted: true,
                        duplicate: false,
                        delivered_online: 0,
                        stored_pending: true,
                    })
                }
            }),
        )
        .route(
            "/api/chat/peer/relay",
            post(move |Json(_request): Json<PeerChatRelayRequest>| {
                let legacy_for_handler = Arc::clone(&legacy_for_handler);
                async move {
                    legacy_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let peer_descriptor = signed_chat_relay_peer_descriptor_with_features(
        endpoint,
        now,
        now + 300,
        &[NodeProtocolFeature::DirectPeerRelayAuthV2],
    );
    let peer_node_id = peer_descriptor.node_id();
    peer_store
        .upsert_verified(peer_descriptor, now)
        .expect("v2 peer descriptor should verify");
    peer_store.record_route_forward_success(&peer_node_id, now);

    let source_identity = IdentityKeyPair::generate();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        None,
        &peer_store,
        &source_identity,
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 1);
    assert_eq!(authenticated_received.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(legacy_received.load(AtomicOrdering::SeqCst), 0);
    mock_peer.abort();
}

#[tokio::test]
async fn discovered_chat_relay_prefers_target_bound_v3_when_advertised() {
    // [DIRECT-RELAY-TARGET-BINDING-V3 2026-08-15 by Codex] The signed
    // rolling descriptor selects v3 without independently requiring the
    // receipt feature. The request remains target-bound, retries stay
    // exact, and the v2 endpoint must not be probed.
    let source_directory = tempfile::tempdir().expect("v3 source relay directory");
    let source_relay = test_chat_relay_service(
        &source_directory.path().join("source-relay.sqlite3"),
        [0x70; 32],
    );
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let target_node_id = target_identity.public_key_bytes();
    let identity_for_handler = Arc::clone(&target_identity);
    let v3_received = Arc::new(AtomicUsize::new(0));
    let v2_received = Arc::new(AtomicUsize::new(0));
    let v3_commitments = Arc::new(Mutex::new(Vec::new()));
    let v3_for_handler = Arc::clone(&v3_received);
    let v2_for_handler = Arc::clone(&v2_received);
    let commitments_for_handler = Arc::clone(&v3_commitments);
    let app = Router::new()
        .route(
            "/api/chat/peer/relay-v3",
            post(move |Json(request): Json<PeerChatRelayRequestV3>| {
                let identity_for_handler = Arc::clone(&identity_for_handler);
                let v3_for_handler = Arc::clone(&v3_for_handler);
                let commitments_for_handler = Arc::clone(&commitments_for_handler);
                async move {
                    let commitment = request
                        .verified_request_commitment_for_target(
                            &identity_for_handler.public_key_bytes(),
                        )
                        .expect("request should bind the selected target");
                    commitments_for_handler
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push(commitment);
                    let attempt = v3_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    if attempt == 0 {
                        // [DIRECT-RELAY-IDEMPOTENT-RETRY 2026-08-15 by Codex]
                        // Model a target that still owns the first attempt.
                        return Err(StatusCode::from_u16(HTTP_TOO_EARLY_STATUS_CODE)
                            .unwrap_or(StatusCode::CONFLICT));
                    }
                    Ok(Json(PeerChatRelayResponseV2 {
                        relay: PeerChatRelayResponse {
                            accepted: true,
                            duplicate: false,
                            delivered_online: 0,
                            stored_pending: true,
                        },
                        receipt: Some(PeerChatRelayReceiptV2::accepted(
                            commitment,
                            unix_now_secs(),
                            identity_for_handler.as_ref(),
                        )),
                    }))
                }
            }),
        )
        .route(
            "/api/chat/peer/relay-v2",
            post(move |Json(_request): Json<PeerChatRelayRequestV2>| {
                let v2_for_handler = Arc::clone(&v2_for_handler);
                async move {
                    v2_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let peer_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        endpoint,
        now.saturating_sub(1),
        now + 300,
        &[
            NodeProtocolFeature::DirectPeerRelayAuthV2,
            NodeProtocolFeature::DirectPeerRelayTargetBindingV3,
        ],
        target_identity.as_ref(),
    );
    peer_store
        .upsert_verified(peer_descriptor, now)
        .expect("target-bound peer descriptor should verify");
    peer_store.record_route_forward_success(&target_node_id, now.saturating_sub(1));

    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        Some(source_relay.as_ref()),
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 1);
    assert_eq!(v3_received.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(v2_received.load(AtomicOrdering::SeqCst), 0);
    let commitments = v3_commitments
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(commitments.len(), 2);
    assert_eq!(commitments[0], commitments[1]);
    let retry = source_relay.peer_status().direct_peer_retry;
    assert_eq!(retry.retry_triggered_total, 1);
    assert_eq!(retry.retry_recovered_total, 1);
    assert_eq!(retry.retry_exhausted_total, 0);
    assert_eq!(retry.deterministic_failure_total, 0);
    assert_eq!(retry.last_outcome.as_deref(), Some("recovered"));
    mock_peer.abort();
}

#[tokio::test]
async fn direct_relay_fanout_filters_routeability_before_limit() {
    // [ROUTEABILITY-BEFORE-FANOUT 2026-10-02 by Codex] Signed descriptors
    // without route evidence remain probe candidates, but cannot consume the
    // bounded direct-delivery budget ahead of an eligible V3 target.
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let target_for_handler = Arc::clone(&target_identity);
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_handler = Arc::clone(&calls);
    let app = Router::new().route(
        "/api/chat/peer/relay-v3",
        post(move |Json(request): Json<PeerChatRelayRequestV3>| {
            let target_for_handler = Arc::clone(&target_for_handler);
            let calls_for_handler = Arc::clone(&calls_for_handler);
            async move {
                let commitment = request
                    .verified_request_commitment_for_target(&target_for_handler.public_key_bytes())
                    .expect("routeable V3 target must receive a target-bound request");
                calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                Ok::<_, StatusCode>(Json(PeerChatRelayResponseV2 {
                    relay: PeerChatRelayResponse {
                        accepted: true,
                        duplicate: false,
                        delivered_online: 0,
                        stored_pending: true,
                    },
                    receipt: Some(PeerChatRelayReceiptV2::accepted(
                        commitment,
                        unix_now_secs(),
                        target_for_handler.as_ref(),
                    )),
                }))
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let high_capacity = NodeCapacity {
        max_sessions: 10_000,
        max_bps: Some(1_000_000_000),
        max_pps: Some(1_000_000),
    };
    let low_capacity = NodeCapacity::default();
    let mut unknown_node_ids = Vec::new();
    for _ in 0..3 {
        let identity = IdentityKeyPair::generate();
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            now.saturating_sub(1),
            now.saturating_sub(1),
            now + 300,
            "unknown-routeable-test-peer",
        );
        descriptor.public_endpoint = Some("http://127.0.0.1:9".to_string());
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        descriptor.capacity = high_capacity.clone();
        let descriptor = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
        unknown_node_ids.push(descriptor.node_id());
        peer_store.upsert_verified(descriptor, now).unwrap();
    }

    let mut target_descriptor = NodeDescriptor::new(
        target_identity.public_key_bytes(),
        now.saturating_sub(1),
        now.saturating_sub(1),
        now + 300,
        "routeable-v3-test-peer",
    )
    .with_protocol_features([NodeProtocolFeature::DirectPeerRelayTargetBindingV3]);
    target_descriptor.public_endpoint = Some(endpoint);
    target_descriptor.capabilities = vec![NodeCapability::ChatRelay];
    target_descriptor.capacity = low_capacity;
    let target_descriptor =
        SignedNodeDescriptor::sign(target_descriptor, &target_identity).unwrap();
    let target_node_id = target_descriptor.node_id();
    peer_store
        .upsert_verified(target_descriptor, now)
        .expect("target descriptor should verify");
    peer_store.record_route_forward_success(&target_node_id, now);

    let pre_fix_candidates = peer_store.route_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        now,
        3,
        &[],
    );
    assert_eq!(pre_fix_candidates.len(), 3);
    assert!(pre_fix_candidates
        .iter()
        .all(|candidate| unknown_node_ids.contains(&candidate.node_id())));
    let routeable_candidates = peer_store.routeable_route_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        now,
        3,
        &[],
    );
    assert_eq!(routeable_candidates.len(), 1);
    assert_eq!(routeable_candidates[0].node_id(), target_node_id);
    assert!(peer_store
        .routeable_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            3,
            &[target_node_id],
        )
        .is_empty());
    let expired_identity = IdentityKeyPair::generate();
    let mut expired_descriptor = NodeDescriptor::new(
        expired_identity.public_key_bytes(),
        now.saturating_sub(1),
        now.saturating_sub(1),
        now.saturating_sub(1),
        "expired-routeable-test-peer",
    );
    expired_descriptor.public_endpoint = Some("http://127.0.0.1:9".to_string());
    expired_descriptor.capabilities = vec![NodeCapability::ChatRelay];
    let expired_descriptor =
        SignedNodeDescriptor::sign(expired_descriptor, &expired_identity).unwrap();
    assert!(peer_store.upsert_verified(expired_descriptor, now).is_err());

    let client = test_peer_http_client();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(client.as_ref()),
        None,
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now),
    )
    .await;
    assert_eq!(accepted, 1);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    mock_peer.abort();
}

#[tokio::test]
async fn target_bound_v3_retry_health_distinguishes_exhaustion_and_determinism() {
    // [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] The first
    // delivery spends its exact-retry budget on two ambiguous 425 replies.
    // The second receives one deterministic 429. Metrics must distinguish
    // those outcomes without retaining target or envelope identifiers.
    let directory = tempfile::tempdir().expect("retry telemetry directory");
    let source_relay = test_chat_relay_service(
        &directory.path().join("retry-telemetry.sqlite3"),
        [0x73; 32],
    );
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_handler = Arc::clone(&calls);
    let app = Router::new().route(
        "/api/chat/peer/relay-v3",
        post(move |Json(_request): Json<PeerChatRelayRequestV3>| {
            let calls_for_handler = Arc::clone(&calls_for_handler);
            async move {
                let call = calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                let status = if call < 2 {
                    StatusCode::from_u16(HTTP_TOO_EARLY_STATUS_CODE).unwrap_or(StatusCode::CONFLICT)
                } else {
                    StatusCode::TOO_MANY_REQUESTS
                };
                Err::<Json<PeerChatRelayResponseV2>, _>(status)
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = unix_now_secs();
    let descriptor = signed_chat_relay_peer_descriptor_for_identity(
        endpoint,
        now.saturating_sub(1),
        now + 300,
        &[NodeProtocolFeature::DirectPeerRelayTargetBindingV3],
        target_identity.as_ref(),
    );
    let target_node_id = descriptor.node_id();
    let source_identity = IdentityKeyPair::generate();
    let client = test_peer_http_client();

    let exhausted_store = PeerStore::new();
    exhausted_store
        .upsert_verified(descriptor.clone(), now)
        .expect("exhaustion target descriptor should verify");
    exhausted_store.record_route_forward_success(&target_node_id, now.saturating_sub(1));
    let exhausted = Server::relay_chat_envelope_to_discovered_peers(
        Some(client.as_ref()),
        Some(source_relay.as_ref()),
        &exhausted_store,
        &source_identity,
        &signed_test_chat_envelope(now),
    )
    .await;
    assert_eq!(exhausted, 0);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    let retry = source_relay.peer_status().direct_peer_retry;
    assert_eq!(retry.retry_triggered_total, 1);
    assert_eq!(retry.retry_recovered_total, 0);
    assert_eq!(retry.retry_exhausted_total, 1);
    assert_eq!(retry.deterministic_failure_total, 0);
    assert_eq!(retry.last_outcome.as_deref(), Some("exhausted"));

    let deterministic_store = PeerStore::new();
    deterministic_store
        .upsert_verified(descriptor, now)
        .expect("deterministic target descriptor should verify");
    deterministic_store.record_route_forward_success(&target_node_id, now.saturating_sub(1));
    let deterministic = Server::relay_chat_envelope_to_discovered_peers(
        Some(client.as_ref()),
        Some(source_relay.as_ref()),
        &deterministic_store,
        &source_identity,
        &signed_test_chat_envelope(now),
    )
    .await;
    assert_eq!(deterministic, 0);
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);
    let retry = source_relay.peer_status().direct_peer_retry;
    assert_eq!(retry.retry_triggered_total, 1);
    assert_eq!(retry.retry_recovered_total, 0);
    assert_eq!(retry.retry_exhausted_total, 1);
    assert_eq!(retry.deterministic_failure_total, 1);
    assert_eq!(retry.last_outcome.as_deref(), Some("deterministic_failure"));
    mock_peer.abort();
}

#[tokio::test]
async fn target_bound_v3_circuit_blocks_legacy_protocol_downgrade() {
    // [DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Once v3 delivery health
    // opens the source-blind circuit, a compatibility-capable peer must not
    // turn an availability incident into a silent authentication downgrade.
    let directory = tempfile::tempdir().expect("direct circuit directory");
    let source_relay =
        test_chat_relay_service(&directory.path().join("direct-circuit.sqlite3"), [0x74; 32]);
    let now = unix_now_secs();
    for offset in 0..3 {
        let observed_at = now.saturating_add(offset);
        let permit = source_relay
            .begin_direct_peer_delivery(observed_at)
            .expect("closed circuit should admit failure seed");
        let allows_more =
            source_relay.complete_direct_peer_delivery(observed_at, permit, false, false, true);
        assert_eq!(allows_more, offset < 2);
    }

    let v3_calls = Arc::new(AtomicUsize::new(0));
    let v3_calls_for_handler = Arc::clone(&v3_calls);
    let v3_app = Router::new().route(
        "/api/chat/peer/relay-v3",
        post(move || {
            let v3_calls_for_handler = Arc::clone(&v3_calls_for_handler);
            async move {
                v3_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }),
    );
    let v3_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let v3_endpoint = format!("http://{}", v3_listener.local_addr().unwrap());
    let v3_node = tokio::spawn(async move {
        axum::serve(v3_listener, v3_app).await.unwrap();
    });

    let v2_calls = Arc::new(AtomicUsize::new(0));
    let v2_calls_for_handler = Arc::clone(&v2_calls);
    let v2_app = Router::new().route(
        "/api/chat/peer/relay-v2",
        post(move || {
            let v2_calls_for_handler = Arc::clone(&v2_calls_for_handler);
            async move {
                v2_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                StatusCode::OK
            }
        }),
    );
    let v2_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let v2_endpoint = format!("http://{}", v2_listener.local_addr().unwrap());
    let v2_node = tokio::spawn(async move {
        axum::serve(v2_listener, v2_app).await.unwrap();
    });

    let v3_identity = IdentityKeyPair::generate();
    let v2_identity = IdentityKeyPair::generate();
    let v3_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        v3_endpoint,
        now.saturating_sub(1),
        now.saturating_add(300),
        &[NodeProtocolFeature::DirectPeerRelayTargetBindingV3],
        &v3_identity,
    );
    let v2_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        v2_endpoint,
        now.saturating_sub(1),
        now.saturating_add(300),
        &[NodeProtocolFeature::DirectPeerRelayAuthV2],
        &v2_identity,
    );
    let v3_node_id = v3_descriptor.node_id();
    let v2_node_id = v2_descriptor.node_id();
    let peer_store = PeerStore::new();
    peer_store
        .upsert_verified(v3_descriptor, now)
        .expect("v3 circuit target should verify");
    peer_store
        .upsert_verified(v2_descriptor, now)
        .expect("v2 compatibility target should verify");
    peer_store.record_route_forward_success(&v3_node_id, now.saturating_sub(1));
    peer_store.record_route_forward_success(&v2_node_id, now.saturating_sub(1));

    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        Some(source_relay.as_ref()),
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 0);
    assert_eq!(v3_calls.load(AtomicOrdering::SeqCst), 0);
    assert_eq!(v2_calls.load(AtomicOrdering::SeqCst), 0);
    let status = source_relay.peer_status();
    assert_eq!(status.direct_peer_retry.circuit.state, "open");
    assert_eq!(status.direct_peer_retry.circuit.blocked_total, 1);
    assert_eq!(status.last_outbound_attempted, 0);
    assert_eq!(
        status.last_outbound_failure_reason.as_deref(),
        Some("peer_relay_circuit_open")
    );

    // Move the circuit to half-open-ready without sleeping. The production
    // relay call must spend exactly one probe on v3; its failure must reopen
    // the circuit before the compatibility candidate can receive traffic.
    let recovery_at = now
        .saturating_add(2)
        .saturating_add(status.direct_peer_retry.circuit.cooldown_seconds);
    let reserved_probe = source_relay
        .begin_direct_peer_delivery(recovery_at)
        .expect("cooldown expiry should reserve a half-open probe");
    assert!(reserved_probe.is_half_open());
    source_relay.cancel_direct_peer_delivery(recovery_at, reserved_probe);

    let half_open_accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        Some(source_relay.as_ref()),
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now.saturating_add(1)),
    )
    .await;
    assert_eq!(half_open_accepted, 0);
    assert_eq!(v3_calls.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(v2_calls.load(AtomicOrdering::SeqCst), 0);
    let recovered_status = source_relay.peer_status();
    assert_eq!(recovered_status.direct_peer_retry.circuit.state, "open");
    assert_eq!(
        recovered_status
            .direct_peer_retry
            .circuit
            .half_open_failed_total,
        1
    );
    assert_eq!(recovered_status.last_outbound_attempted, 1);
    v3_node.abort();
    v2_node.abort();
}

#[tokio::test]
#[ignore = "invoked only as an isolated child of the restart drill"]
async fn target_bound_v3_restart_drill_subprocess_worker() {
    let Ok(stage) = std::env::var(DIRECT_RELAY_RESTART_DRILL_STAGE_ENV) else {
        // Ordinary test-suite execution exercises orchestration through the
        // parent above. Only its explicitly scoped child has worker input.
        return;
    };
    let db_path = std::env::var_os(DIRECT_RELAY_RESTART_DRILL_DB_ENV)
        .map(std::path::PathBuf::from)
        .expect("restart drill database path");
    if stage == "verify_missing_checkpoint_table_restart" {
        verify_missing_checkpoint_table_restart(&db_path);
        return;
    }
    let relay = test_chat_relay_service(&db_path, [0x79; 32]);

    match stage.as_str() {
        "seed_crash" => seed_direct_relay_restart_drill_crash(relay.as_ref()),
        "seed_checkpoint_table_loss_crash" => {
            seed_direct_relay_checkpoint_table_loss_crash(relay.as_ref(), &db_path)
        }
        "seed_half_open_crash" => seed_direct_relay_half_open_crash(relay.as_ref()),
        "seed_half_open_progress_crash" => {
            seed_direct_relay_half_open_progress_crash(relay.as_ref())
        }
        "resume_half_open_progress" => resume_half_open_progress_after_restart(relay.as_ref()),
        "verify_closed_restart" => verify_closed_circuit_after_restart(relay.as_ref()),
        "verify_restart" => verify_direct_relay_restart_drill(relay.as_ref()).await,
        "verify_interrupted_probe_restart" => {
            verify_interrupted_half_open_restart(relay.as_ref());
        }
        other => panic!("unsupported restart drill stage: {other}"),
    }
}

#[tokio::test]
async fn discovered_chat_relay_requires_signed_receipt_when_advertised() {
    // [DIRECT-RELAY-RECEIPT-V2 2026-08-15 by Codex] The descriptor and
    // custody acknowledgement share the selected target identity; route
    // success is recorded only after that signature and request binding.
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let identity_for_handler = Arc::clone(&target_identity);
    let app = Router::new().route(
        "/api/chat/peer/relay-v2",
        post(move |Json(request): Json<PeerChatRelayRequestV2>| {
            let identity_for_handler = Arc::clone(&identity_for_handler);
            async move {
                let commitment = request
                    .verified_request_commitment()
                    .expect("source-authenticated request should verify");
                Json(PeerChatRelayResponseV2 {
                    relay: PeerChatRelayResponse {
                        accepted: true,
                        duplicate: false,
                        delivered_online: 0,
                        stored_pending: true,
                    },
                    receipt: Some(PeerChatRelayReceiptV2::accepted(
                        commitment,
                        unix_now_secs(),
                        identity_for_handler.as_ref(),
                    )),
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let peer_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        endpoint,
        now.saturating_sub(1),
        now + 300,
        &[
            NodeProtocolFeature::DirectPeerRelayAuthV2,
            NodeProtocolFeature::DirectPeerRelayReceiptV2,
        ],
        target_identity.as_ref(),
    );
    let peer_node_id = peer_descriptor.node_id();
    peer_store
        .upsert_verified(peer_descriptor, now)
        .expect("receipt-capable peer descriptor should verify");
    peer_store.record_route_forward_success(&peer_node_id, now.saturating_sub(1));

    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        None,
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 1);
    let row = peer_store
        .route_candidate_status(now + 1)
        .chat_relay
        .into_iter()
        .find(|row| row.node_id_prefix == hex::encode(&peer_node_id[..4]))
        .expect("receipt-capable peer should remain visible");
    assert_eq!(row.route_health, "healthy");
    assert_eq!(row.route_consecutive_failures, 0);
    mock_peer.abort();
}

#[tokio::test]
async fn discovered_chat_relay_rejects_forged_signed_receipt() {
    let target_identity = Arc::new(IdentityKeyPair::generate());
    let identity_for_handler = Arc::clone(&target_identity);
    let app = Router::new().route(
        "/api/chat/peer/relay-v2",
        post(move |Json(request): Json<PeerChatRelayRequestV2>| {
            let identity_for_handler = Arc::clone(&identity_for_handler);
            async move {
                let commitment = request.request_commitment().unwrap();
                let mut receipt = PeerChatRelayReceiptV2::accepted(
                    commitment,
                    unix_now_secs(),
                    identity_for_handler.as_ref(),
                );
                receipt.signature[0] ^= 0x01;
                Json(PeerChatRelayResponseV2 {
                    relay: PeerChatRelayResponse {
                        accepted: true,
                        duplicate: false,
                        delivered_online: 0,
                        stored_pending: true,
                    },
                    receipt: Some(receipt),
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let peer_descriptor = signed_chat_relay_peer_descriptor_for_identity(
        endpoint,
        now.saturating_sub(1),
        now + 300,
        &[
            NodeProtocolFeature::DirectPeerRelayAuthV2,
            NodeProtocolFeature::DirectPeerRelayReceiptV2,
        ],
        target_identity.as_ref(),
    );
    let peer_node_id = peer_descriptor.node_id();
    let peer_prefix = hex::encode(&peer_node_id[..4]);
    peer_store
        .upsert_verified(peer_descriptor, now)
        .expect("receipt-capable peer descriptor should verify");
    peer_store.record_route_forward_success(&peer_node_id, now.saturating_sub(1));

    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        None,
        &peer_store,
        &IdentityKeyPair::generate(),
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 0);
    let row = peer_store
        .route_candidate_status(now + 1)
        .chat_relay
        .into_iter()
        .find(|row| row.node_id_prefix == peer_prefix)
        .expect("forged-receipt peer should remain visible for diagnostics");
    assert_eq!(row.route_health, "degraded");
    assert_eq!(row.route_consecutive_failures, 1);
    assert_eq!(
        row.last_route_failure_reason.as_deref(),
        Some("peer_relay_receipt_signature_invalid")
    );
    mock_peer.abort();
}

#[tokio::test]
async fn discovered_chat_relay_non_durable_ack_marks_route_failure() {
    let received = Arc::new(AtomicUsize::new(0));
    let received_for_handler = Arc::clone(&received);
    let app = Router::new().route(
        "/api/chat/peer/relay",
        post(move |Json(request): Json<PeerChatRelayRequest>| {
            let received_for_handler = Arc::clone(&received_for_handler);
            async move {
                assert_eq!(request.envelope.message_id, [0x55; 16]);
                received_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                Json(PeerChatRelayResponse {
                    // [DIRECT-RELAY-DURABILITY 2026-08-15 by Codex] An
                    // HTTP acceptance without durable custody is failure.
                    accepted: true,
                    duplicate: false,
                    delivered_online: 0,
                    stored_pending: false,
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let peer_store = PeerStore::new();
    let peer_descriptor =
        signed_chat_relay_peer_descriptor(endpoint, now.saturating_sub(1), now + 300);
    let peer_node_id = peer_descriptor.node_id();
    let peer_prefix = hex::encode(&peer_node_id[..4]);
    peer_store
        .upsert_verified(peer_descriptor, now)
        .expect("mock peer descriptor should verify");
    peer_store.record_route_forward_success(&peer_node_id, now.saturating_sub(1));

    let client = reqwest::Client::new();
    let source_identity = IdentityKeyPair::generate();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&client),
        None,
        &peer_store,
        &source_identity,
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 0);
    assert_eq!(received.load(AtomicOrdering::SeqCst), 1);
    let route_status = peer_store.route_candidate_status(now + 1);
    let row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == peer_prefix)
        .expect("mock peer should remain in route candidate status");
    assert_eq!(row.route_health, "degraded");
    assert_eq!(row.route_consecutive_failures, 1);
    assert_eq!(
        row.last_route_failure_reason.as_deref(),
        Some("peer_relay_ack_rejected")
    );
    assert_eq!(row.last_route_success_at, Some(now.saturating_sub(1)));
    mock_peer.abort();
}

#[tokio::test]
async fn discovered_chat_relay_skips_peer_with_newer_failure_evidence() {
    let now = unix_now_secs();
    let peer_store = PeerStore::new();
    let descriptor = signed_chat_relay_peer_descriptor(
        "http://127.0.0.1:9".to_string(),
        now.saturating_sub(1),
        now + 300,
    );
    let node_id = descriptor.node_id();
    peer_store
        .upsert_verified(descriptor, now)
        .expect("test peer descriptor should verify");
    peer_store.record_route_forward_success(&node_id, now.saturating_sub(1));
    peer_store.record_route_forward_failure(&node_id, now, "request_failed");

    let source_identity = IdentityKeyPair::generate();
    let accepted = Server::relay_chat_envelope_to_discovered_peers(
        Some(&reqwest::Client::new()),
        None,
        &peer_store,
        &source_identity,
        &signed_test_chat_envelope(now),
    )
    .await;

    assert_eq!(accepted, 0);
    let row = peer_store
        .route_candidate_status(now + 1)
        .chat_relay
        .into_iter()
        .find(|row| row.node_id_prefix == hex::encode(&node_id[..4]))
        .expect("unreachable peer should remain visible for diagnostics");
    assert_eq!(row.routeability_state, "unreachable");
    assert_eq!(row.route_failure_count, 1);
    assert_eq!(row.route_consecutive_failures, 1);
}

#[tokio::test]
async fn signed_v1_client_delivery_cache_remains_upgrade_compatible() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_005_000;
    let middle = signed_probe_peer_descriptor(
        "https://legacy-middle.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x31; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://legacy-terminal.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x32; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle.clone(), now).unwrap();
    original_store
        .upsert_verified(terminal.clone(), now)
        .unwrap();
    original_store.record_route_forward_success(&middle.node_id(), now + 1);
    original_store.record_route_forward_success(&terminal.node_id(), now + 1);
    original_store.record_verified_client_onion_delivery(now + 2);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-client-delivery-v1-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 3)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let mut document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    document.verified_client_delivery_schema_version =
        VERIFIED_CLIENT_DELIVERY_CACHE_LEGACY_SCHEMA_VERSION;
    document.verified_client_delivery_generation = 0;
    document.verified_client_delivery_signature_ed25519 =
        Some(hex::encode(server.identity.sign(
            &document.verified_client_delivery_signing_bytes().unwrap(),
        )));
    let legacy_bytes = serde_json::to_vec_pretty(&document).unwrap();

    let restored_store = PeerStore::new();
    assert!(Server::import_bootstrap_snapshot_bytes(
        &restored_store,
        "cache",
        &path_str,
        &legacy_bytes,
        now + 4,
        Some(&server.identity),
    ));
    let status = restored_store.status(now + 4);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        1
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("legacy_unanchored")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_generation, 0);

    // Once a v2 anchor exists, replaying the same valid v1 document is a
    // downgrade attempt rather than an upgrade-compatibility case.
    tokio::fs::write(&path, &legacy_bytes).await.unwrap();
    let downgrade_store = PeerStore::new();
    server
        .load_peer_cache(&downgrade_store, &path_str, now + 5)
        .await;
    let downgrade_status = downgrade_store.status(now + 5);
    assert_eq!(
        downgrade_status
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        0
    );
    assert_eq!(
        downgrade_status
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(
        downgrade_status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}
