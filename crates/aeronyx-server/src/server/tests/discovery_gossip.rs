// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn two_hop_onion_delivery_probe_request_uses_onion_blob_and_signed_probe_envelope() {
    let source = IdentityKeyPair::generate();
    let self_node_id = source.public_key_bytes();
    let now = 1_800_000_000;
    // [PROBE-DESCRIPTOR-FIXTURE 2026-09-13 by Codex] Sequence orders
    // descriptors; it must not accidentally make issued_at future-dated.
    let middle = signed_probe_peer_descriptor(
        "http://198.51.100.10:8422".to_string(),
        1,
        now,
        now + 300,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x21; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "http://198.51.100.11:8422".to_string(),
        2,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay],
        [0x22; 32],
    );

    let (request, payload_commitment) = Server::build_two_hop_onion_delivery_probe_request(
        &source,
        &self_node_id,
        &middle,
        &terminal,
        now,
    )
    .expect("descriptors with KEM keys should build an onion delivery probe");
    let source_public = IdentityPublicKey::from_bytes(&self_node_id).unwrap();

    assert!(request.onward_envelope.is_none());
    assert!(request.onward_descriptor_hint.is_none());
    assert_eq!(request.previous_hop_node_id, self_node_id);
    assert_eq!(request.envelope.next_hop, middle.node_id());
    assert_eq!(request.envelope.ttl, 2);
    assert!(is_onion_blob(&request.envelope.encrypted_blob));
    request
        .envelope
        .verify_signature_from(&source_public)
        .expect("entry node signs the outer blind relay envelope");

    let synthetic_chat = Server::synthetic_two_hop_probe_chat_envelope(
        &source,
        &self_node_id,
        &middle.node_id(),
        &terminal.node_id(),
        request.envelope.route_id,
        now,
    );
    synthetic_chat
        .verify_signature()
        .expect("synthetic terminal ChatEnvelope must verify like user chat");
    assert_eq!(synthetic_chat.message_id, request.envelope.route_id);
    assert_eq!(synthetic_chat.sender, self_node_id);
    assert_ne!(synthetic_chat.receiver, [0u8; 32]);
    assert_eq!(synthetic_chat.content_type, ChatContentType::System);
    let encoded_chat = encode_envelope(&synthetic_chat).unwrap();
    assert_eq!(
        payload_commitment,
        BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
            &encoded_chat,
            OnionRoutePurpose::MessageRelay,
        )
    );
}

#[test]
fn two_hop_onion_delivery_probe_request_requires_published_kem_keys() {
    let source = IdentityKeyPair::generate();
    let self_node_id = source.public_key_bytes();
    let now = 1_800_000_000;
    let middle = signed_probe_peer_descriptor(
        "http://198.51.100.10:8422".to_string(),
        1,
        now,
        now + 300,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0u8; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "http://198.51.100.11:8422".to_string(),
        2,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay],
        [0x22; 32],
    );

    assert!(Server::build_two_hop_onion_delivery_probe_request(
        &source,
        &self_node_id,
        &middle,
        &terminal,
        now,
    )
    .is_none());
}

#[tokio::test]
async fn signed_purpose_receipt_feature_is_probe_eligible_without_unsigned_summary() {
    let now = 1_800_000_000;
    let identity = IdentityKeyPair::generate();
    // [SIGNED-RECEIPT-NEGOTIATION 2026-08-11 by Codex] No public endpoint
    // is intentional: consulting the legacy summary would make this peer
    // ineligible, while its signature-protected claim must be sufficient
    // to authorize only the optional probe attempt.
    let descriptor = NodeDescriptor::new(
        identity.public_key_bytes(),
        now,
        now,
        now + 300,
        "signed-purpose-receipt-peer",
    )
    .with_protocol_features([NodeProtocolFeature::PurposeBoundDeliveryReceiptV2]);
    let signed = SignedNodeDescriptor::sign(descriptor, &identity).unwrap();
    let node_id = signed.node_id();
    let store = PeerStore::new();
    store.upsert_verified(signed.clone(), now).unwrap();

    let supported = Server::purpose_bound_delivery_receipt_advertisers(
        &reqwest::Client::new(),
        &store,
        &[signed],
        now,
    )
    .await;

    assert_eq!(supported, std::collections::HashSet::from([node_id]));
    assert!(
        !store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now),
        "a descriptor claim must not be promoted to verified receipt evidence"
    );
}

#[test]
fn three_hop_onion_delivery_probe_builds_three_sealed_relay_layers() {
    let source = IdentityKeyPair::generate();
    let self_node_id = source.public_key_bytes();
    let now = 1_800_000_050;
    let first_middle = signed_probe_peer_descriptor(
        "http://198.51.100.10:8422".to_string(),
        1,
        now,
        now + 300,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x31; 32],
    );
    let second_middle = signed_probe_peer_descriptor(
        "http://203.0.113.20:8422".to_string(),
        2,
        now,
        now + 300,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x32; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "http://192.0.2.30:8422".to_string(),
        3,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay],
        [0x33; 32],
    );

    let (request, payload_commitment) = Server::build_three_hop_onion_delivery_probe_request(
        &source,
        &self_node_id,
        &first_middle,
        &second_middle,
        &terminal,
        now,
    )
    .expect("three KEM-capable descriptors should build a three-hop onion probe");

    assert_eq!(request.previous_hop_node_id, self_node_id);
    assert_eq!(request.envelope.next_hop, first_middle.node_id());
    assert_eq!(request.envelope.ttl, 3);
    assert!(request.onward_envelope.is_none());
    assert!(request.onward_descriptor_hint.is_none());
    assert!(is_onion_blob(&request.envelope.encrypted_blob));

    let synthetic_chat = Server::synthetic_three_hop_probe_chat_envelope(
        &source,
        &self_node_id,
        &first_middle.node_id(),
        &second_middle.node_id(),
        &terminal.node_id(),
        request.envelope.route_id,
        now,
    );
    synthetic_chat
        .verify_signature()
        .expect("three-hop synthetic terminal envelope must remain signed");
    let encoded_chat = encode_envelope(&synthetic_chat).unwrap();
    assert_eq!(
        payload_commitment,
        BlindRelayDeliveryReceipt::payload_commitment_for_purpose(
            &encoded_chat,
            OnionRoutePurpose::MessageRelay,
        )
    );
}

#[tokio::test]
async fn two_hop_probe_does_not_report_success_without_a_distinct_terminal() {
    let now = 1_800_000_100;
    let source = IdentityKeyPair::generate();
    let self_node_id = source.public_key_bytes();
    let middle = signed_probe_peer_descriptor(
        "http://198.51.100.10:8422".to_string(),
        1,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        [0x31; 32],
    );
    let store = PeerStore::new();
    store.upsert_verified(middle, now).unwrap();

    let outcome = Server::probe_two_hop_blind_relay_path(
        &reqwest::Client::new(),
        &store,
        &source,
        &self_node_id,
        now,
    )
    .await;

    // [TWO-HOP-PROBE-OUTCOME 2026-07-31 by Codex] Candidate discovery is
    // not network execution, and neither may be presented as terminal
    // delivery without a distinct terminal-signed receipt.
    assert!(!outcome.attempted);
    assert!(!outcome.route_accepted);
    assert!(!outcome.terminal_delivery_verified);
    let history = store.status(now).two_hop_path_proof_history;
    assert_eq!(history.attempted, 1);
    assert_eq!(history.succeeded, 0);
    assert_eq!(history.message_delivery_successes, 0);
    assert_eq!(
        history.latest_reason_bucket.as_deref(),
        Some("no_distinct_path")
    );
}

#[tokio::test]
async fn three_hop_probe_records_only_verified_terminal_delivery() {
    let now = 1_800_000_200;
    let source_identity = IdentityKeyPair::generate();
    let first_middle_identity = IdentityKeyPair::generate();
    let second_middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();
    let self_node_id = source_identity.public_key_bytes();

    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let relay_port = listener.local_addr().unwrap().port();
    let signed_descriptor = |identity: &IdentityKeyPair,
                             endpoint: String,
                             capabilities: Vec<NodeCapability>,
                             name: &str| {
        let mut descriptor =
            NodeDescriptor::new(identity.public_key_bytes(), now, now, now + 300, name)
                .with_x25519_kem(identity.x25519_public_key_bytes());
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = capabilities;
        SignedNodeDescriptor::sign(descriptor, identity).unwrap()
    };
    let first_middle_is_entry =
        first_middle_identity.public_key_bytes() < second_middle_identity.public_key_bytes();
    let first_middle_host = if first_middle_is_entry {
        "127.0.0.1"
    } else {
        "127.0.1.1"
    };
    let second_middle_host = if first_middle_is_entry {
        "127.0.1.1"
    } else {
        "127.0.0.1"
    };
    let first_middle = signed_descriptor(
        &first_middle_identity,
        format!("http://{first_middle_host}:{relay_port}"),
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        "three-hop-first",
    );
    let second_middle = signed_descriptor(
        &second_middle_identity,
        format!("http://{second_middle_host}:{relay_port}"),
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        "three-hop-second",
    );
    let terminal = signed_descriptor(
        &terminal_identity,
        format!("http://127.0.2.1:{relay_port}"),
        vec![NodeCapability::ChatRelay],
        "three-hop-terminal",
    );
    let first_middle_node_id = first_middle.node_id();
    let second_middle_node_id = second_middle.node_id();
    let terminal_node_id = terminal.node_id();
    let probe_source_identity = source_identity.clone();
    let router = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "multihop_delivery_receipt_v1": true,
                        "purpose_bound_delivery_receipt_v2": true
                    }
                }))
            }),
        )
        .route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                // [THREE-HOP-PROBE-TEST-DETERMINISM 2026-08-02 by Codex]
                // Reconstruct the synthetic terminal payload commitment
                // for the middle ordering actually selected at runtime.
                // Candidate order is randomized by design, so a fixed
                // template commitment makes this test spuriously retry.
                let selected_first_middle = request.envelope.next_hop;
                let selected_second_middle = if selected_first_middle == first_middle_node_id {
                    second_middle_node_id
                } else {
                    first_middle_node_id
                };
                let synthetic_chat = Server::synthetic_three_hop_probe_chat_envelope(
                    &probe_source_identity,
                    &self_node_id,
                    &selected_first_middle,
                    &selected_second_middle,
                    &terminal_node_id,
                    request.envelope.route_id,
                    now,
                );
                let encoded_chat = encode_envelope(&synthetic_chat).unwrap();
                let receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
                    request.envelope.route_id,
                    &encoded_chat,
                    OnionRoutePurpose::MessageRelay,
                    now,
                    &terminal_identity,
                );
                async move {
                    assert_eq!(request.envelope.ttl, 3);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: false,
                        forwarded: true,
                        ttl_remaining: 1,
                        reason: Some("onion_forwarded".to_string()),
                        delivery_receipt: Some(receipt),
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                }
            }),
        );
    let http_server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let store = PeerStore::new();
    for descriptor in [first_middle, second_middle, terminal] {
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&node_id, now);
    }
    let proven_middles = store
        .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::OnionMiddle,
            now,
            8,
            &[self_node_id],
        );
    assert_eq!(proven_middles.len(), 2);
    let terminal_exclusions = [
        self_node_id,
        proven_middles[0].node_id(),
        proven_middles[1].node_id(),
    ];
    let proven_terminals = store
        .multi_hop_delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now,
            8,
            &terminal_exclusions,
        );
    assert_eq!(proven_terminals.len(), 1);
    assert!(PeerStore::route_endpoints_are_network_diverse(
        &proven_middles[0],
        &proven_middles[1],
    ));
    assert!(proven_middles.iter().all(|middle| {
        PeerStore::route_endpoints_are_network_diverse(middle, &proven_terminals[0])
    }));
    assert!(proven_middles.iter().all(|middle| {
        middle
            .descriptor
            .public_endpoint
            .as_deref()
            .is_some_and(|endpoint| Server::blind_relay_probe_url(endpoint).is_some())
    }));
    // [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Keep three
    // distinct /24 identities while binding the deterministic score/tie
    // winner to the reachable listener. Later hops are onion-encapsulated.
    let test_client = reqwest::Client::builder().no_proxy().build().unwrap();
    let outcome = Server::probe_three_hop_blind_relay_path(
        &test_client,
        &store,
        &source_identity,
        &self_node_id,
        now,
    )
    .await;

    assert!(outcome.attempted);
    assert!(outcome.route_accepted);
    assert!(outcome.terminal_delivery_verified);
    let status = store.status(now);
    assert_eq!(status.two_hop_path_proof_history.attempted, 0);
    assert_eq!(status.three_hop_path_proof_history.attempted, 1);
    assert_eq!(status.three_hop_path_proof_history.succeeded, 1);
    assert!(status.three_hop_path_proof_history.message_delivery_ready);
    assert_eq!(
        status
            .three_hop_path_proof_history
            .path_shape_counts
            .get("entry_middle_middle_terminal"),
        Some(&1)
    );
    assert_eq!(status.blind_relay_quality.delivery_receipt_capable_peers, 3);

    http_server.abort();
    let _ = http_server.await;
}

#[tokio::test]
async fn three_hop_probe_defers_unproven_first_middle_without_route_penalty() {
    let now = 1_800_000_300;
    let source_identity = IdentityKeyPair::generate();
    let middle_identity = IdentityKeyPair::generate();
    let self_node_id = source_identity.public_key_bytes();

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let first_middle_address = listener.local_addr().unwrap();
    let blind_relay_requests = Arc::new(AtomicUsize::new(0));
    let counted_requests = Arc::clone(&blind_relay_requests);
    let router = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                // [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex]
                // v1 framing alone must not authorize a v2 three-hop path.
                Json(serde_json::json!({
                    "protocol_features": {
                        "legacy_descriptor_gossip_v1": true,
                        "multihop_delivery_receipt_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/chat/peer/blind-relay",
            post(move || {
                let requests = Arc::clone(&counted_requests);
                async move {
                    requests.fetch_add(1, AtomicOrdering::Relaxed);
                    StatusCode::BAD_GATEWAY
                }
            }),
        );
    let http_server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let mut descriptor = NodeDescriptor::new(
        middle_identity.public_key_bytes(),
        now,
        now,
        now + 300,
        "legacy-three-hop-first",
    )
    .with_x25519_kem(middle_identity.x25519_public_key_bytes());
    descriptor.public_endpoint = Some(format!("http://{first_middle_address}"));
    descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    let first_middle = SignedNodeDescriptor::sign(descriptor, &middle_identity).unwrap();
    let store = PeerStore::new();
    store.upsert_verified(first_middle, now).unwrap();

    let outcome = Server::probe_three_hop_blind_relay_path(
        &reqwest::Client::new(),
        &store,
        &source_identity,
        &self_node_id,
        now,
    )
    .await;

    assert!(!outcome.attempted);
    assert!(!outcome.route_accepted);
    assert!(!outcome.terminal_delivery_verified);
    assert_eq!(blind_relay_requests.load(AtomicOrdering::Relaxed), 0);
    assert_eq!(store.status(now).three_hop_path_proof_history.attempted, 0);

    http_server.abort();
    let _ = http_server.await;
}

#[tokio::test]
async fn authenticated_chat_uses_distinct_receipt_capable_onion_path() {
    let now = unix_now_secs();
    let source = IdentityKeyPair::generate();
    let chat_sender = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();
    let terminal_node_id = terminal_identity.public_key_bytes();
    let middle_identity = IdentityKeyPair::generate();
    let middle_node_id = middle_identity.public_key_bytes();

    let mut envelope = ChatEnvelope {
        message_id: [0x41; 16],
        sender: chat_sender.public_key_bytes(),
        receiver: [0x42; 32],
        timestamp: now,
        ciphertext: b"opaque app ciphertext".to_vec(),
        nonce: [0x43; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    envelope.signature = chat_sender.sign(&envelope.sign_data());
    let encoded_envelope = encode_envelope(&envelope).unwrap();
    let terminal_payload = encoded_envelope.clone();

    let terminal_receipt_identity = terminal_identity.clone();
    let relay_requests = Arc::new(AtomicUsize::new(0));
    let relay_requests_for_handler = Arc::clone(&relay_requests);
    let relay = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let terminal_receipt_identity = terminal_receipt_identity.clone();
            let terminal_payload = terminal_payload.clone();
            let relay_requests = Arc::clone(&relay_requests_for_handler);
            async move {
                relay_requests.fetch_add(1, AtomicOrdering::Relaxed);
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

    let mut terminal_descriptor = NodeDescriptor::new(
        terminal_node_id,
        now,
        now,
        now + 300,
        "test-receipt-terminal",
    )
    .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
    terminal_descriptor.public_endpoint = Some("http://127.0.1.1:9".to_string());
    terminal_descriptor.capabilities = vec![NodeCapability::ChatRelay];
    let terminal_descriptor =
        SignedNodeDescriptor::sign(terminal_descriptor, &terminal_identity).unwrap();

    let store = PeerStore::new();
    store.upsert_verified(middle_descriptor, now).unwrap();
    store.upsert_verified(terminal_descriptor, now).unwrap();
    store.record_route_forward_success(&middle_node_id, now);
    store.record_route_forward_success(&terminal_node_id, now);
    store.record_purpose_bound_delivery_receipt_capability(&middle_node_id, now);
    store.record_purpose_bound_delivery_receipt_capability(&terminal_node_id, now);

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

    assert!(outcome.delivered());
    assert!(outcome.fully_replicated());
    assert_eq!(outcome.attempted_paths, 1);
    assert_eq!(outcome.verified_receipts, 1);
    let terminal_receipt = outcome
        .first_terminal_receipt
        .as_ref()
        .expect("verified relay returns exact terminal receipt");
    terminal_receipt
        .verify_expected_for_purpose(
            &terminal_receipt.route_id,
            &encoded_envelope,
            OnionRoutePurpose::MessageRelay,
            &terminal_node_id,
        )
        .expect("returned receipt remains independently verifiable");
    let quality = store.status(unix_now_secs()).blind_relay_quality;
    assert!(quality.real_relay_ready);
    assert_eq!(quality.verified_client_onion_deliveries, 1);
    assert_eq!(quality.delivery_receipt_capable_peers, 2);
    assert_eq!(
        quality.evidence_mode,
        "verified_client_onion_delivery_receipt"
    );

    // [CHAT-VERIFIED-SUBMIT-LIVE-HANDLER 2026-08-23 by Codex] Reuse the
    // same real two-hop endpoint through the product-facing handler. The
    // returned response must combine terminal-verifiable delivery with
    // durable entry custody and remain independently bound to this request.
    let verified_directory = tempfile::tempdir().expect("verified submit live directory");
    let verified_relay = test_chat_relay_service(
        &verified_directory
            .path()
            .join("verified-submit-live.sqlite3"),
        [0x45; 32],
    );
    let verified_relay_option = Some(Arc::clone(&verified_relay));
    let verified_request = ChatRelayVerifiedSubmitRequestV1::signed(
        [0x44; 16],
        envelope.clone(),
        unix_now_secs(),
        &chat_sender,
    )
    .expect("sign live verified submit request");
    let verified_session = Arc::new(crate::services::Session::new(
        aeronyx_common::types::SessionId::generate(),
        chat_sender.public_key(),
        aeronyx_core::crypto::SessionKey::from_bytes([0x46; 32]),
        Ipv4Addr::new(100, 64, 0, 46),
        "127.0.0.1:1046".parse().unwrap(),
    ));
    let source_node_id = source.public_key_bytes();
    let first_client = reqwest::Client::new();
    let concurrent_client = reqwest::Client::new();
    let (verified_response, concurrent_response) = tokio::join!(
        Server::handle_verified_chat_submit(
            verified_request.clone(),
            &verified_session,
            &verified_relay_option,
            &store,
            &source_node_id,
            &source,
            Some(&first_client),
        ),
        Server::handle_verified_chat_submit(
            verified_request.clone(),
            &verified_session,
            &verified_relay_option,
            &store,
            &source_node_id,
            &source,
            Some(&concurrent_client),
        )
    );
    assert_eq!(
        verified_response.result,
        CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1
    );
    assert_eq!(concurrent_response, verified_response);
    verified_response
        .verify_terminal_receipt_for_request(&verified_request, &terminal_node_id)
        .expect("live handler response remains independently verifiable");
    assert_eq!(relay_requests.load(AtomicOrdering::Relaxed), 2);

    let replay_response = Server::handle_verified_chat_submit(
        verified_request.clone(),
        &verified_session,
        &verified_relay_option,
        &store,
        &source_node_id,
        &source,
        Some(&reqwest::Client::new()),
    )
    .await;
    assert_eq!(replay_response, verified_response);
    assert_eq!(relay_requests.load(AtomicOrdering::Relaxed), 2);

    let verified_status = verified_relay.peer_status().verified_submit;
    assert_eq!(verified_status.total, 3);
    assert_eq!(verified_status.onion_and_entry_total, 3);
    assert_eq!(verified_status.unknown_result_total, 0);
    assert_eq!(verified_status.replayed_total, 2);
    assert_eq!(verified_status.request_conflict_total, 0);

    relay_server.abort();
    let _ = relay_server.await;
}

#[test]
fn discovery_startup_self_check_reports_commercial_readiness_buckets() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.gossip_enabled = true;

    let (status, detail) = Server::discovery_startup_self_check(&config);

    assert_eq!(status, "warning");
    assert!(detail.contains("peer_cache_path"));
    assert!(detail.contains("seed_endpoints"));
    assert!(detail.contains("public_endpoint"));
    assert!(detail.contains("public_api_listener"));
    assert!(!detail.contains("https://"));
    assert!(!detail.contains("/root/"));

    config.discovery.peer_cache_path = Some("/root/private/peer-cache.json".to_string());
    config.discovery.seed_endpoints = vec!["https://seed.example.com".to_string()];
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());

    let (status, detail) = Server::discovery_startup_self_check(&config);

    assert_eq!(status, "ready");
    assert!(detail.contains("cache"));
    assert!(!detail.contains("seed.example.com"));
    assert!(!detail.contains("/root/private"));
}

#[test]
fn blind_relay_probe_cooldown_uses_recovery_interval_until_stability_window_is_ready() {
    let mut discovery = DiscoveryConfig::default();
    discovery.gossip_interval_secs = 60;

    let store = PeerStore::new();
    assert_eq!(
        Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_000,),
        super::super::BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS
    );

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_010,
        true,
        "onion_terminal_delivered",
        2,
        1,
        2,
        1,
    );
    assert_eq!(
        Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_020,),
        super::super::BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS
    );

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_030,
        true,
        "onion_terminal_delivered",
        2,
        1,
        2,
        1,
    );
    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_040,
        true,
        "onion_terminal_delivered",
        2,
        1,
        2,
        1,
    );
    assert_eq!(
        Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_050,),
        BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS
    );

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_060,
        false,
        "request_error",
        2,
        1,
        2,
        1,
    );
    assert_eq!(
        Server::blind_relay_probe_cooldown_secs_for_status(&discovery, &store, 1_700_000_070,),
        super::super::BLIND_RELAY_PROBE_RECOVERY_COOLDOWN_SECS
    );
}

// [SELF-DESCRIPTOR-TEST-REGISTRATION 2026-08-11 by Codex] This regression
// test previously lacked `#[test]`, so its privacy and signature assertions
// compiled but never executed.
#[test]
fn self_discovery_descriptor_uses_privacy_safe_public_metadata() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());
    config.discovery.region = Some("us-central".to_string());
    config.discovery.descriptor_ttl_secs = 900;
    config.discovery.public_discovery = false;

    let identity = IdentityKeyPair::generate();
    let node_id = identity.public_key_bytes();
    let server = Server::new(config, identity, None);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let signed = server.build_self_discovery_descriptor(now).unwrap();

    assert!(signed.verify_at(now + 1).is_ok());
    assert_eq!(signed.descriptor.node_id, node_id);
    assert_eq!(signed.descriptor.sequence, now);
    assert_eq!(signed.descriptor.issued_at, now);
    assert_eq!(signed.descriptor.expires_at, now + 900);
    assert_eq!(
        signed.descriptor.public_endpoint.as_deref(),
        Some("node.example.com:443")
    );
    assert_eq!(
        signed.descriptor.policy.region.as_deref(),
        Some("us-central")
    );
    assert!(!signed.descriptor.policy.allows_public_exit);
    assert!(!signed.descriptor.policy.public_discovery);
    assert!(signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::PrivacyRelay));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::BlindRelayFailureReceiptV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PurposeBoundDeliveryReceiptV2));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayAuthV2));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayReceiptV2));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::DirectPeerRelayTargetBindingV3));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionReplyV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindLeaseAdmissionV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultPutReceiptV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultLeaseRetireV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultLeaseRenewalV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultLeaseStatusV1));
    assert!(signed
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::OnionBlindVaultLeaseInventoryV1));
    assert_eq!(
        signed.descriptor.capacity.max_sessions,
        server.config.max_sessions() as u32
    );
}

#[test]
fn self_discovery_descriptor_can_fallback_to_network_endpoint() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.network.public_endpoint = Some("198.51.100.10:51820".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let signed = server
        .build_self_discovery_descriptor(1_800_000_000)
        .unwrap();

    assert_eq!(
        signed.descriptor.public_endpoint.as_deref(),
        Some("198.51.100.10:51820")
    );
}

#[test]
fn self_discovery_descriptor_requires_peer_api_endpoint_for_chat_relay_capability() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.network.public_endpoint = Some("198.51.100.10:51820".to_string());
    config.memchain.chat_relay.enabled = true;

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let signed = server
        .build_self_discovery_descriptor(1_800_000_000)
        .unwrap();

    assert_eq!(
        signed.descriptor.public_endpoint.as_deref(),
        Some("198.51.100.10:51820")
    );
    assert!(!signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay));
    assert!(!signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::OnionMiddle));
}

#[test]
fn self_discovery_descriptor_requires_peer_api_listener_for_chat_relay_capability() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.memchain.chat_relay.enabled = true;

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let signed = server
        .build_self_discovery_descriptor(1_800_000_000)
        .unwrap();

    assert!(!signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay));
    assert!(!signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::OnionMiddle));
}

#[test]
fn self_discovery_descriptor_requires_chat_relay_runtime_for_chat_relay_capability() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.memchain.chat_relay.enabled = true;

    let signed = Server::build_self_discovery_descriptor_for_runtime(
        &config,
        &IdentityKeyPair::generate(),
        1_800_000_000,
        false,
        false,
    )
    .unwrap();

    assert!(!signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay));
    assert!(signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::PrivacyRelay));
}

#[test]
fn self_discovery_work_policy_rotation_requires_a_new_exact_pin() {
    let mut config = ServerConfig::default();
    config
        .memchain
        .chat_relay
        .anonymous_mailbox
        .ticket_issue_work_bits = 12;
    let identity = IdentityKeyPair::from_bytes(&[0x72; 32]).expect("identity");
    let first = Server::build_self_discovery_descriptor_for_runtime(
        &config,
        &identity,
        1_800_000_000,
        false,
        true,
    )
    .unwrap();
    config
        .memchain
        .chat_relay
        .anonymous_mailbox
        .ticket_issue_work_bits = 13;
    let rotated = Server::build_self_discovery_descriptor_for_runtime(
        &config,
        &identity,
        1_800_000_001,
        false,
        true,
    )
    .unwrap();
    let first_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&first).unwrap();
    let rotated_pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&rotated).unwrap();

    assert_ne!(first_pin, rotated_pin);
    assert_eq!(
        rotated.anonymous_mailbox_work_policy_for_pin_at(&first_pin, 1_800_000_002,),
        Err(AnonymousMailboxWorkPolicyError::ClaimsConflict)
    );
    assert_eq!(
        rotated
            .anonymous_mailbox_work_policy_for_pin_at(&rotated_pin, 1_800_000_002,)
            .unwrap()
            .work_bits(),
        13
    );
}

#[test]
fn discovery_readiness_includes_blind_relay_runtime_quality_without_private_metadata() {
    let mut config = ServerConfig::default();
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.memchain.chat_relay.enabled = true;

    let peer_store = PeerStore::new();
    peer_store.record_blind_relay_forwarded(1_700_000_010, 1);
    let status = peer_store.status(1_700_000_020);
    let local_capabilities = Server::discovery_local_capability_status_for(&config);
    let readiness =
        crate::api::discovery::discovery_readiness_status_value(&status, &local_capabilities);
    let blind_relay_runtime = readiness
        .get("blind_relay_runtime")
        .expect("blind relay runtime readiness object");
    let route_governance = readiness
        .get("route_governance")
        .expect("route governance readiness object");
    let protocol_foundation = readiness
        .get("protocol_foundation")
        .expect("protocol foundation readiness object");

    assert_eq!(protocol_foundation["status"], "forming");
    assert_eq!(protocol_foundation["stage"], "single_hop_relay_ready");
    assert_eq!(protocol_foundation["checks_total"], 4);
    assert_eq!(protocol_foundation["checks_passed"], 2);
    assert_eq!(protocol_foundation["local_relay_ready"], true);
    assert_eq!(protocol_foundation["blind_relay_ready"], true);
    assert_eq!(
        protocol_foundation["privacy_invariant"],
        "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status"
    );
    assert_eq!(blind_relay_runtime["status"], "ready");
    assert_eq!(blind_relay_runtime["runtime_ready"], true);
    assert_eq!(blind_relay_runtime["quality_ready"], true);
    assert_eq!(blind_relay_runtime["accepted_total"], 1);
    assert_eq!(blind_relay_runtime["forward_failed"], 0);
    assert_eq!(blind_relay_runtime["last_event_age_seconds"], 10);
    assert!(blind_relay_runtime["last_probe_age_seconds"].is_null());
    assert_eq!(route_governance["contract_version"], "route_governance.v1");
    assert_eq!(route_governance["status"], "forming");
    assert_eq!(route_governance["route_pool_ready"], false);
    assert_eq!(route_governance["quality_ready"], false);
    assert_eq!(route_governance["candidates_total"], 0);
    assert!(route_governance["average_score"].is_null());

    let serialized = serde_json::to_string(&readiness).unwrap();
    assert!(!serialized.contains("https://node.example.com"));
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("payload_b64"));
    assert!(!serialized.contains("client_ip"));
}

#[test]
fn discovery_heartbeat_projection_keeps_aggregates_and_omits_heavy_local_rows() {
    let peer_store = PeerStore::new();
    peer_store.record_blind_relay_forwarded(1_700_000_010, 1);
    let status = peer_store.status(1_700_000_020);
    let projection = peer_store_heartbeat_status_value(&status);

    for required in [
        "snapshot",
        "runtime",
        "blind_relay_quality",
        "two_hop_path_proof_history",
        "three_hop_path_proof_history",
        "recent_peer_events",
        "bootstrap",
        "stability",
        "route_governance",
        "peer_quorum",
        "network_story",
    ] {
        assert!(
            projection.get(required).is_some(),
            "missing field: {required}"
        );
    }
    for local_only in [
        "recent_audit_events",
        "peer_summary",
        "route_candidates",
        "peer_health_summary",
    ] {
        assert!(
            projection.get(local_only).is_none(),
            "unexpected local diagnostic field: {local_only}"
        );
    }

    let serialized = serde_json::to_vec(&projection).unwrap();
    assert!(serialized.len() < 32 * 1024);
}

#[test]
fn discovery_heartbeat_reports_generation_bound_recovery_anchor() {
    let now = 1_700_000_020;
    let peer_store = PeerStore::new();
    peer_store.record_routeability_cache_rollback_protection(now, 2, "anchored");
    peer_store.record_two_hop_proof_cache_persisted(now, 3, true);
    peer_store.record_three_hop_proof_cache_persisted(now, 3, true);
    peer_store.record_client_delivery_cache_persisted(now, 2, 2);
    peer_store.record_client_delivery_witness_round(
        now,
        1,
        true,
        1,
        PeerStoreVerifiedDeliveryWitnessRound {
            configured: 1,
            attempted: 1,
            verified: 1,
            idempotent: 1,
            ..Default::default()
        },
    );

    let status = peer_store.status(now);
    let local_capabilities = DiscoveryLocalCapabilityStatus::default();
    let signed_peer_records = peer_store.export_signed_peer_records_for_heartbeat(now, Some(8));
    let heartbeat =
        discovery_heartbeat_status_value(now, &status, &local_capabilities, signed_peer_records);
    let recovery_anchor = &heartbeat["recovery_anchor"];

    assert_eq!(recovery_anchor["contract_version"], "recovery_anchor.v1");
    assert_eq!(recovery_anchor["status"], "blocked");
    assert_eq!(recovery_anchor["ready_for_restore"], false);
    assert_eq!(recovery_anchor["cache_generation"], 2);
    assert_eq!(recovery_anchor["local_anchor"]["ready"], true);
    assert_eq!(recovery_anchor["external_witness"]["status"], "verified");
    assert_eq!(
        recovery_anchor["external_witness"]["generation_aligned"],
        false
    );
    assert_eq!(recovery_anchor["external_witness"]["ready"], false);

    let serialized_anchor = serde_json::to_string(recovery_anchor).unwrap();
    for forbidden in [
        "https://",
        "\"anchor_digest\"",
        "\"signature\"",
        "\"endpoint\"",
        "\"peer_id\"",
        "\"route_id\"",
        "\"message_id\"",
        "\"client_ip\"",
        "\"payload_b64\"",
    ] {
        assert!(
            !serialized_anchor.contains(forbidden),
            "recovery anchor leaked forbidden field: {forbidden}"
        );
    }
    let serialized_heartbeat = serde_json::to_vec(&heartbeat).unwrap();
    assert!(serialized_heartbeat.len() < 64 * 1024);
}

#[test]
fn self_discovery_descriptor_can_advertise_onion_middle_when_explicitly_enabled() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.discovery.advertise_onion_middle = true;
    config.memchain.chat_relay.enabled = true;

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let signed = server
        .build_self_discovery_descriptor(1_800_000_000)
        .unwrap();

    assert!(signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay));
    assert!(signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::OnionMiddle));
    assert!(signed
        .descriptor
        .capabilities
        .contains(&NodeCapability::PrivacyRelay));
}

#[test]
fn discovery_gossip_url_normalizes_endpoint_forms() {
    assert_eq!(
        Server::discovery_gossip_url("198.51.100.10:51820").as_deref(),
        Some("http://198.51.100.10:51820/api/discovery/gossip")
    );
    assert_eq!(
        Server::discovery_gossip_url("https://node.example.com").as_deref(),
        Some("https://node.example.com/api/discovery/gossip")
    );
    assert_eq!(
        Server::discovery_gossip_url(
            " HTTPS://Node.Example.COM:443/untrusted/path?token=secret#fragment "
        )
        .as_deref(),
        Some("https://node.example.com/api/discovery/gossip")
    );
    assert_eq!(
        Server::discovery_gossip_url("https://user@node.example.com"),
        None
    );
    assert_eq!(Server::discovery_gossip_url("ftp://8.8.8.8"), None);
    assert_eq!(Server::discovery_gossip_url("   "), None);
    assert_eq!(
        Server::discovered_peer_gossip_url("http://8.8.8.8:8422/path").as_deref(),
        Some("http://8.8.8.8:8422/api/discovery/gossip")
    );
    for endpoint in [
        "http://127.0.0.1:8422",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.1:8422",
        "https://node.example.com",
    ] {
        assert_eq!(
            Server::discovered_peer_gossip_url(endpoint),
            None,
            "unexpectedly accepted {endpoint}"
        );
    }
    assert_eq!(
        Server::discovery_summary_url_from_gossip_url(
            "https://node.example.com/api/discovery/gossip"
        )
        .as_deref(),
        Some("https://node.example.com/api/discovery/summary")
    );
    assert_eq!(
        Server::discovery_summary_url_from_gossip_url(
            "https://node.example.com/api/discovery/status"
        ),
        None
    );
}

#[test]
fn discovery_gossip_schedule_applies_jitter_and_backpressure() {
    let mut discovery = DiscoveryConfig {
        enabled: true,
        gossip_enabled: true,
        gossip_interval_secs: 60,
        gossip_jitter_percent: 10,
        gossip_backpressure_failure_threshold: 3,
        gossip_failure_backoff_max_secs: 300,
        ..DiscoveryConfig::default()
    };
    let node_id = [0x42; 32];

    let (_delay, backpressure_active, delay_secs, jitter_secs) =
        Server::discovery_gossip_schedule(&discovery, &node_id, 1_800_000_000, 0);
    assert!(!backpressure_active);
    assert!((54..=66).contains(&delay_secs));
    assert!((-6..=6).contains(&jitter_secs));

    let (_delay, backpressure_active, delay_secs, _jitter_secs) =
        Server::discovery_gossip_schedule(&discovery, &node_id, 1_800_000_060, 5);
    assert!(backpressure_active);
    assert!((216..=264).contains(&delay_secs));

    discovery.gossip_jitter_percent = 0;
    let (_delay, backpressure_active, delay_secs, jitter_secs) =
        Server::discovery_gossip_schedule(&discovery, &node_id, 1_800_000_120, 8);
    assert!(backpressure_active);
    assert_eq!(delay_secs, 300);
    assert_eq!(jitter_secs, 0);
}

#[test]
fn blind_relay_probe_cooldown_keeps_synthetic_checks_low_frequency() {
    let mut discovery = DiscoveryConfig {
        gossip_interval_secs: 60,
        ..DiscoveryConfig::default()
    };

    assert_eq!(
        Server::blind_relay_probe_cooldown_secs(&discovery),
        BLIND_RELAY_PROBE_MIN_COOLDOWN_SECS
    );

    discovery.gossip_interval_secs = 600;
    assert_eq!(Server::blind_relay_probe_cooldown_secs(&discovery), 1_800);
}

#[test]
fn blind_relay_probe_priority_prefers_unproven_non_quarantined_peer() {
    let store = PeerStore::new();
    let now = 1_800_000_000;

    let proven = signed_probe_peer_descriptor(
        "https://proven.example".to_string(),
        1,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        [0x31; 32],
    );
    let unproven = signed_probe_peer_descriptor(
        "https://unproven.example".to_string(),
        2,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        [0x32; 32],
    );
    let quarantined = signed_probe_peer_descriptor(
        "https://quarantined.example".to_string(),
        3,
        now,
        now + 300,
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        [0x33; 32],
    );

    let proven_id = proven.node_id();
    let unproven_id = unproven.node_id();
    let quarantined_id = quarantined.node_id();
    store.upsert_verified(proven.clone(), now).unwrap();
    store.upsert_verified(unproven.clone(), now).unwrap();
    store.upsert_verified(quarantined.clone(), now).unwrap();
    store.record_route_forward_success(&proven_id, now + 1);
    store.record_route_forward_failure(&quarantined_id, now + 1, "request_failed");
    store.record_route_forward_failure(&quarantined_id, now + 2, "request_failed");
    store.record_route_forward_failure(&quarantined_id, now + 3, "request_failed");

    let mut candidates = vec![proven, quarantined, unproven];
    Server::prioritize_probe_candidates(&store, now + 4, &mut candidates);

    assert_eq!(candidates[0].node_id(), unproven_id);
    assert_eq!(candidates[1].node_id(), proven_id);
    assert_eq!(candidates[2].node_id(), quarantined_id);
}

#[test]
fn permissionless_probe_requires_fresh_exact_target_signed_terminal_receipt() {
    use aeronyx_core::protocol::chat::{BlindRelayEnvelope, BlindRelaySuccessReceipt};

    let now = 1_800_100_000;
    let source = IdentityKeyPair::generate();
    let target = IdentityKeyPair::generate();
    let other = IdentityKeyPair::generate();
    let envelope = BlindRelayEnvelope {
        route_id: [0x41; 16],
        next_hop: target.public_key_bytes(),
        ttl: 1,
        encrypted_blob: vec![0x51; 32],
        timestamp: now,
        signature: [0; 64],
    }
    .sign_with(&source);
    let mut ack = PeerBlindRelayResponse {
        accepted: true,
        terminal: true,
        forwarded: false,
        ttl_remaining: 1,
        reason: Some("terminal_next_hop".to_string()),
        delivery_receipt: None,
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };
    // [PERMISSIONLESS-ENDPOINT-PROMOTION 2026-09-24 by Codex] Unsigned
    // `accepted=true` remains legacy-compatible but cannot grant new
    // permissionless route authority.
    assert!(!Server::permissionless_promotion_probe_ack_valid(
        &envelope,
        &ack,
        &target.public_key_bytes(),
        now,
        now,
    ));
    ack.success_receipt = Some(BlindRelaySuccessReceipt::terminal(
        &envelope,
        1,
        ack.reason.as_deref(),
        None,
        None,
        now,
        &other,
    ));
    assert!(!Server::permissionless_promotion_probe_ack_valid(
        &envelope,
        &ack,
        &target.public_key_bytes(),
        now,
        now,
    ));
    ack.success_receipt = Some(BlindRelaySuccessReceipt::terminal(
        &envelope,
        1,
        ack.reason.as_deref(),
        None,
        None,
        now - 60,
        &target,
    ));
    assert!(!Server::permissionless_promotion_probe_ack_valid(
        &envelope,
        &ack,
        &target.public_key_bytes(),
        now,
        now,
    ));
    ack.success_receipt = Some(BlindRelaySuccessReceipt::terminal(
        &envelope,
        1,
        ack.reason.as_deref(),
        None,
        None,
        now,
        &target,
    ));
    let mut other_envelope = envelope.clone();
    other_envelope.route_id = [0x42; 16];
    assert!(!Server::permissionless_promotion_probe_ack_valid(
        &other_envelope,
        &ack,
        &target.public_key_bytes(),
        now,
        now,
    ));
    let mut forwarded = ack.clone();
    forwarded.terminal = false;
    forwarded.forwarded = true;
    assert!(!Server::permissionless_promotion_probe_ack_valid(
        &envelope,
        &forwarded,
        &target.public_key_bytes(),
        now,
        now,
    ));
    assert!(Server::permissionless_promotion_probe_ack_valid(
        &envelope,
        &ack,
        &target.public_key_bytes(),
        now,
        now,
    ));
}

#[tokio::test]
async fn startup_blind_relay_probe_batch_is_bounded_and_completes_coverage() {
    let now = 1_800_100_000u64;
    let requests = Arc::new(AtomicUsize::new(0));
    let handler_requests = Arc::clone(&requests);
    let router = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move || {
            let requests = Arc::clone(&handler_requests);
            async move {
                requests.fetch_add(1, AtomicOrdering::SeqCst);
                Json(PeerBlindRelayResponse {
                    accepted: true,
                    terminal: true,
                    forwarded: false,
                    ttl_remaining: 1,
                    reason: Some("terminal_next_hop".to_string()),
                    delivery_receipt: None,
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: None,
                })
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let http_server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let store = PeerStore::new();
    let endpoint = format!("http://{address}");
    let candidates = (1..=4)
        .map(|sequence| {
            signed_chat_relay_peer_descriptor(endpoint.clone(), sequence, now.saturating_add(600))
        })
        .collect::<Vec<_>>();
    let candidate_ids = candidates
        .iter()
        .map(SignedNodeDescriptor::node_id)
        .collect::<Vec<_>>();
    for candidate in candidates {
        store.upsert_verified(candidate, now).unwrap();
    }

    let identity = IdentityKeyPair::generate();
    let self_node_id = identity.public_key_bytes();
    let attempted = Server::probe_blind_relay_candidates(
        &reqwest::Client::new(),
        &store,
        &identity,
        &self_node_id,
        now,
        usize::MAX,
    )
    .await;
    assert_eq!(attempted, BLIND_RELAY_STARTUP_WARMUP_MAX_CANDIDATES);
    assert_eq!(requests.load(AtomicOrdering::SeqCst), 3);
    assert_eq!(
        candidate_ids
            .iter()
            .filter(|node_id| store.is_routeable_now(node_id, now))
            .count(),
        3
    );

    let attempted = Server::probe_blind_relay_candidates(
        &reqwest::Client::new(),
        &store,
        &identity,
        &self_node_id,
        now.saturating_add(1),
        1,
    )
    .await;
    assert_eq!(attempted, 1);
    assert_eq!(requests.load(AtomicOrdering::SeqCst), 4);
    assert!(candidate_ids
        .iter()
        .all(|node_id| store.is_routeable_now(node_id, now.saturating_add(1))));

    http_server.abort();
    let _ = http_server.await;
}

#[tokio::test]
async fn two_hop_probe_continues_past_legacy_ack_to_find_signed_receipt() {
    let now = 1_800_200_000u64;
    let source_identity = IdentityKeyPair::generate();
    let legacy_middle_identity = IdentityKeyPair::generate();
    let receipt_middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let signed_descriptor = |identity: &IdentityKeyPair,
                             endpoint: String,
                             capabilities: Vec<NodeCapability>,
                             name: &str| {
        let mut descriptor =
            NodeDescriptor::new(identity.public_key_bytes(), now, now, now + 300, name)
                .with_x25519_kem(identity.x25519_public_key_bytes());
        descriptor.public_endpoint = Some(endpoint);
        descriptor.capabilities = capabilities;
        SignedNodeDescriptor::sign(descriptor, identity).unwrap()
    };

    let terminal = signed_descriptor(
        &terminal_identity,
        "https://terminal.test".to_string(),
        vec![NodeCapability::ChatRelay],
        "receipt-terminal",
    );
    let self_node_id = source_identity.public_key_bytes();

    let legacy_requests = Arc::new(AtomicUsize::new(0));
    let legacy_requests_for_handler = Arc::clone(&legacy_requests);
    let legacy_router = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "multihop_delivery_receipt_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let requests = Arc::clone(&legacy_requests_for_handler);
                async move {
                    requests.fetch_add(1, AtomicOrdering::SeqCst);
                    let legacy_fallback = request.onward_envelope.is_some();
                    Json(PeerBlindRelayResponse {
                        accepted: legacy_fallback,
                        terminal: false,
                        forwarded: legacy_fallback,
                        ttl_remaining: 1,
                        reason: (!legacy_fallback).then(|| "legacy_only".to_string()),
                        delivery_receipt: None,
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                }
            }),
        );
    let legacy_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let legacy_address = legacy_listener.local_addr().unwrap();
    let legacy_server = tokio::spawn(async move {
        axum::serve(legacy_listener, legacy_router).await.unwrap();
    });

    let receipt_requests = Arc::new(AtomicUsize::new(0));
    let receipt_requests_for_handler = Arc::clone(&receipt_requests);
    let receipt_probe_source = source_identity.clone();
    let receipt_terminal_identity = terminal_identity.clone();
    let receipt_terminal_node_id = terminal.node_id();
    let receipt_router = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "multihop_delivery_receipt_v1": true,
                        "purpose_bound_delivery_receipt_v2": true
                    }
                }))
            }),
        )
        .route(
            "/api/chat/peer/blind-relay",
            post(move |Json(request): Json<PeerBlindRelayRequest>| {
                let requests = Arc::clone(&receipt_requests_for_handler);
                let synthetic_chat = Server::synthetic_two_hop_probe_chat_envelope(
                    &receipt_probe_source,
                    &self_node_id,
                    &request.envelope.next_hop,
                    &receipt_terminal_node_id,
                    request.envelope.route_id,
                    now,
                );
                let encoded_chat = encode_envelope(&synthetic_chat).unwrap();
                let receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
                    request.envelope.route_id,
                    &encoded_chat,
                    OnionRoutePurpose::MessageRelay,
                    now,
                    &receipt_terminal_identity,
                );
                async move {
                    requests.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(PeerBlindRelayResponse {
                        accepted: true,
                        terminal: false,
                        forwarded: true,
                        ttl_remaining: 1,
                        reason: None,
                        delivery_receipt: Some(receipt),
                        success_receipt: None,
                        failure_receipt: None,
                        opaque_terminal_response_b64: None,
                    })
                }
            }),
        );
    let receipt_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let receipt_address = receipt_listener.local_addr().unwrap();
    let receipt_server = tokio::spawn(async move {
        axum::serve(receipt_listener, receipt_router).await.unwrap();
    });

    // [PEER-ENDPOINT-SSRF 2026-07-28 by Codex] Integration listeners use
    // an IP-literal loopback seam because production permissionless
    // descriptors deliberately reject DNS names, including `localhost`.
    let legacy_middle = signed_descriptor(
        &legacy_middle_identity,
        format!("http://{legacy_address}"),
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        "legacy-middle",
    );
    let receipt_middle = signed_descriptor(
        &receipt_middle_identity,
        format!("http://{receipt_address}"),
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle],
        "receipt-middle",
    );
    let receipt_middle_id = receipt_middle.node_id();

    // [TWO-HOP-PROBE-OUTCOME 2026-07-31 by Codex] A legacy-only network
    // proves encrypted route compatibility, never terminal delivery.
    let legacy_store = PeerStore::new();
    for descriptor in [legacy_middle.clone(), terminal.clone()] {
        legacy_store.upsert_verified(descriptor, now).unwrap();
    }
    legacy_store.record_purpose_bound_delivery_receipt_capability(&receipt_terminal_node_id, now);
    let test_client = reqwest::Client::builder().no_proxy().build().unwrap();
    let legacy_outcome = Server::probe_two_hop_blind_relay_path(
        &test_client,
        &legacy_store,
        &source_identity,
        &self_node_id,
        now,
    )
    .await;
    assert!(legacy_outcome.attempted);
    assert!(legacy_outcome.route_accepted);
    assert!(!legacy_outcome.terminal_delivery_verified);
    let legacy_history = legacy_store.status(now).two_hop_path_proof_history;
    assert_eq!(legacy_history.succeeded, 1);
    assert_eq!(legacy_history.proof_scope, "control_plane");
    assert_eq!(legacy_history.message_delivery_successes, 0);

    // Reset only the test counter. Thenly route to a signed receipt.
    legacy_requests.store(0, AtomicOrdering::SeqCst);

    let store = PeerStore::new();
    for descriptor in [legacy_middle, receipt_middle, terminal] {
        store.upsert_verified(descriptor, now).unwrap();
    }
    store.record_purpose_bound_delivery_receipt_capability(&receipt_terminal_node_id, now);
    // Coverage ordering tries unknown peers first. Mark only the modern
    // middle as proven so this test deterministically exercises the legacy
    // ACK before the signed-receipt route.
    store.record_route_forward_success(&receipt_middle_id, now);

    let outcome = Server::probe_two_hop_blind_relay_path(
        &test_client,
        &store,
        &source_identity,
        &self_node_id,
        now,
    )
    .await;

    assert!(outcome.attempted);
    assert!(outcome.route_accepted);
    assert!(outcome.terminal_delivery_verified);
    assert_eq!(legacy_requests.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(receipt_requests.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(
        store
            .status(now)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        2,
    );

    legacy_server.abort();
    receipt_server.abort();
    let _ = legacy_server.await;
    let _ = receipt_server.await;
}

#[test]
fn encrypted_relay_and_discovery_logs_stay_route_safe() {
    // [ARCH-SPLIT-VERIFY 2026-10-02 by Codex] Preserve coverage of all code moved from server.rs.
    let source = [
        include_str!("../../server.rs"),
        include_str!("../startup_models.rs"),
        include_str!("../startup_chat_mailbox.rs"),
        include_str!("../startup_memchain.rs"),
        include_str!("../startup_network.rs"),
        include_str!("../startup_directory.rs"),
        include_str!("../startup_files.rs"),
        include_str!("../discovery_advertisement.rs"),
        include_str!("../bootstrap_import.rs"),
        include_str!("../verified_submit_ingress.rs"),
        include_str!("../session_ingress.rs"),
    ]
    .concat();
    let forbidden = [
        concat!("peer = %", "url"),
        concat!(
            "warn!(session = %",
            "session.id, wallet = %hex::encode(&wallet_pubkey"
        ),
        concat!(
            "info!(session_id = %",
            "session.id, wallet = %hex::encode(&wallet_pubkey"
        ),
        concat!("wallet = %hex::encode(&wallet[..4]), ", "\"[CHAT_RELAY]"),
        concat!("device_id = %hex::", "encode(device_id)"),
        concat!("device_name = %", "name_display"),
        concat!("id = %hex::", "encode(envelope.message_id)"),
        concat!("receiver = %hex::", "encode(&envelope.receiver"),
        concat!("warn!(wallet = %", "wallet_hex"),
        concat!("debug!(wallet = %&", "wallet_hex"),
        concat!("src = %", "session.virtual_ip"),
        concat!("dst = %", "target_session.virtual_ip"),
        concat!("dst_ip = %", "dst_ip"),
    ];

    for pattern in forbidden {
        assert!(
            !source.contains(pattern),
            "relay/discovery logs must not expose stable routing identifiers: {pattern}"
        );
    }
}

#[tokio::test]
async fn interrupted_half_open_probe_reopens_after_abrupt_process_restart() {
    // [DIRECT-RELAY-HALF-OPEN-CRASH 2026-08-15 by Codex] The seed process
    // uses the production admission API to commit an in-flight lease, then
    // exits without cancellation or completion. A fresh process must treat
    // the unknowable network outcome as failed and start a new cooldown.
    let directory = tempfile::tempdir().expect("half-open crash drill directory");
    let db_path = directory
        .path()
        .join("direct-relay-half-open-crash.sqlite3");

    let crashed =
        run_direct_relay_restart_drill_child("seed_half_open_crash", &db_path, None, None).await;
    assert_restart_drill_child_crashed(&crashed);

    let restarted = run_direct_relay_restart_drill_child(
        "verify_interrupted_probe_restart",
        &db_path,
        None,
        None,
    )
    .await;
    assert_restart_drill_child_succeeded("interrupted half-open restart", &restarted);
}

#[test]
fn discovery_peer_identity_hints_fail_closed_on_endpoint_collision() {
    // [DISCOVERY-IDENTITY-AMBIGUITY 2026-07-28 by Codex] A repeated signed
    // identity is stable, but conflicting identities must never recover
    // to first- or last-writer wins during one complete hint snapshot.
    let url = "https://peer.example/api/discovery/gossip".to_string();
    let first_node_id = [0x41; 32];
    let second_node_id = [0x42; 32];
    let mut hints = DiscoveryPeerIdentityHints::default();

    hints.observe_verified(url.clone(), first_node_id);
    hints.observe_verified(url.clone(), first_node_id);
    assert_eq!(hints.unique_node_id(&url), Some(first_node_id));

    hints.observe_verified(url.clone(), second_node_id);
    assert_eq!(hints.unique_node_id(&url), None);

    hints.observe_verified(url.clone(), first_node_id);
    assert_eq!(hints.unique_node_id(&url), None);
}

#[test]
fn runtime_gossip_sample_is_not_monopolized_by_low_id_clique() {
    // [PERMISSIONLESS-GOSSIP-RUNTIME 2026-09-24 by Codex] The old
    // bootstrap-export prefix would always select the first two IDs,
    // both assigned here to one /24. Runtime selection must instead
    // admit a diverse verified peer in every bounded round.
    let now = 1_800_000_000;
    let store = PeerStore::new();
    let mut identities = (1..=5u8)
        .map(|seed| {
            IdentityKeyPair::from_bytes(&[seed; 32])
                .unwrap_or_else(|_| panic!("test identity must be valid"))
        })
        .collect::<Vec<_>>();
    identities.sort_by_key(IdentityKeyPair::public_key_bytes);
    let endpoints = [
        "https://8.8.8.1",
        "https://8.8.8.2",
        "https://9.9.9.1",
        "https://11.11.11.1",
        "https://12.12.12.1",
    ];
    for (identity, endpoint) in identities.iter().zip(endpoints) {
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now - 10,
            now + 300,
            "gossip-sample-test",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.policy.public_discovery = true;
        let Ok(signed) = SignedNodeDescriptor::sign(descriptor, identity) else {
            panic!("test descriptor must sign");
        };
        assert!(store.upsert_verified(signed, now).is_ok());
    }

    for seed in 1..=16u8 {
        let mut seen_urls = std::collections::HashSet::new();
        let mut gossip_urls = Vec::new();
        Server::append_sampled_discovered_peer_gossip_urls(
            &store,
            &DiscoveryGossipSampleRequest {
                now,
                round_nonce: [seed; 32],
                round_peer_limit: 2,
                self_node_id: &[0xFF; 32],
                self_gossip_url: None,
            },
            &mut seen_urls,
            &mut gossip_urls,
        );
        assert_eq!(gossip_urls.len(), 2);
        assert_eq!(seen_urls.len(), 2);
        assert!(gossip_urls
            .iter()
            .any(|url| { !url.starts_with("https://8.8.8.") }));
    }
}

#[tokio::test]
async fn outbound_gossip_imports_snapshot_response_from_peer() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_for_handler = Arc::clone(&calls);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let remote_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:9".to_string(), now, now + 300);
    let remote_node_id = remote_descriptor.node_id();
    let snapshot_response = NodeDiscoveryMessage::SnapshotResponse {
        snapshot: NodeBootstrapSnapshot::new(now, vec![remote_descriptor.clone()]),
    };
    let app = Router::new().route(
        "/api/discovery/gossip",
        post(move |Json(message): Json<NodeDiscoveryMessage>| {
            let calls_for_handler = Arc::clone(&calls_for_handler);
            let snapshot_response = snapshot_response.clone();
            async move {
                calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex]
                // Test-only gossip mocks explicitly discard the dormant carrier.
                let response = match message {
                    NodeDiscoveryMessage::DescriptorAnnounce { .. }
                    | NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. }
                    | NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                        GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        }
                    }
                    NodeDiscoveryMessage::SnapshotRequest { .. } => GossipResponse {
                        applied: PeerStoreImportReport::empty(),
                        response: Some(snapshot_response),
                    },
                    NodeDiscoveryMessage::SnapshotResponse { .. } => GossipResponse {
                        applied: PeerStoreImportReport::empty(),
                        response: None,
                    },
                };
                Json(response)
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let peer_store = PeerStore::new();
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
    let client = reqwest::Client::new();

    let report = Server::gossip_with_peer(
        gossip_execution(&client, &peer_store, &[], now, Duration::from_secs(5)),
        &url,
        self_descriptor,
    )
    .await;

    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(report, DiscoveryPeerGossipReport::default());
    assert!(peer_store.get_valid(&remote_node_id, now + 1).is_some());
    assert_eq!(peer_store.status(now + 1).runtime.last_gossip_at, Some(now));
    mock_peer.abort();
}

#[tokio::test]
async fn outbound_gossip_preserves_proof_when_legacy_exchange_fails() {
    let calls = Arc::new(AtomicUsize::new(0));
    let proof_calls = Arc::new(AtomicUsize::new(0));
    let calls_for_handler = Arc::clone(&calls);
    let proof_calls_for_handler = Arc::clone(&proof_calls);
    let app = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "directory_descriptor_proof_gossip_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/discovery/gossip",
            post(move |Json(message): Json<NodeDiscoveryMessage>| {
                let calls_for_handler = Arc::clone(&calls_for_handler);
                let proof_calls_for_handler = Arc::clone(&proof_calls_for_handler);
                async move {
                    calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    let status = match message {
                        NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                            proof_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                            StatusCode::OK
                        }
                        NodeDiscoveryMessage::DescriptorAnnounce { .. }
                        | NodeDiscoveryMessage::SnapshotResponse { .. }
                        | NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                            StatusCode::OK
                        }
                        NodeDiscoveryMessage::SnapshotRequest { .. } => {
                            StatusCode::INTERNAL_SERVER_ERROR
                        }
                    };
                    (
                        status,
                        Json(GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        }),
                    )
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let now = unix_now_secs();
    let announcement = directory_gossip_announcement(now);
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();

    let report = Server::gossip_with_peer(
        gossip_execution(
            &client,
            &peer_store,
            std::slice::from_ref(&announcement),
            now,
            Duration::from_secs(5),
        ),
        &url,
        self_descriptor,
    )
    .await;

    assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);
    assert_eq!(proof_calls.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(
        report.directory_proof,
        DirectoryProofGossipOutcome {
            state: DirectoryProofGossipPeerState::Attempted,
            frames_attempted: 1,
            evidence_rejected: 0,
            result: DirectoryProofGossipResult::Accepted,
        }
    );
    assert_eq!(
        report
            .legacy_error
            .map(DiscoveryGossipFailure::bucket)
            .as_deref(),
        Some("snapshot_status_http_500")
    );
    assert_eq!(peer_store.status(now + 1).runtime.last_gossip_at, None);
    mock_peer.abort();
}

#[tokio::test]
async fn bounded_gossip_fanout_isolates_slow_peers() {
    let slow_calls = Arc::new(AtomicUsize::new(0));
    let fast_calls = Arc::new(AtomicUsize::new(0));
    let (slow_url, slow_peer) =
        spawn_legacy_gossip_mock(Duration::from_secs(1), Arc::clone(&slow_calls)).await;
    let (fast_url, fast_peer) =
        spawn_legacy_gossip_mock(Duration::ZERO, Arc::clone(&fast_calls)).await;
    let now = unix_now_secs();
    let started_at = tokio::time::Instant::now();
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);

    let reports = Server::gossip_with_peers_bounded(
        gossip_execution(&client, &peer_store, &[], now, Duration::from_millis(250)),
        vec![slow_url.clone(), slow_url, fast_url],
        &self_descriptor,
        2,
    )
    .await;

    assert!(started_at.elapsed() < Duration::from_millis(450));
    assert_eq!(reports.len(), 3);
    assert_eq!(
        reports[0].legacy_error,
        Some(DiscoveryGossipFailure::peer_timeout())
    );
    assert_eq!(
        reports[1].legacy_error,
        Some(DiscoveryGossipFailure::peer_timeout())
    );
    assert_eq!(reports[2].legacy_error, None);
    assert_eq!(slow_calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(fast_calls.load(AtomicOrdering::SeqCst), 2);
    slow_peer.abort();
    fast_peer.abort();
}

#[tokio::test]
async fn outbound_gossip_skips_receiver_producer_and_uses_bounded_fallback() {
    let calls = Arc::new(AtomicUsize::new(0));
    let proof_calls = Arc::new(AtomicUsize::new(0));
    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let receiver_producer = [0x91; 32];
    let calls_for_handler = Arc::clone(&calls);
    let proof_calls_for_handler = Arc::clone(&proof_calls);
    let legacy_calls_for_handler = Arc::clone(&legacy_calls);
    let app = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "directory_descriptor_proof_gossip_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/discovery/gossip",
            post(move |Json(message): Json<NodeDiscoveryMessage>| {
                let calls_for_handler = Arc::clone(&calls_for_handler);
                let proof_calls_for_handler = Arc::clone(&proof_calls_for_handler);
                let legacy_calls_for_handler = Arc::clone(&legacy_calls_for_handler);
                async move {
                    calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    let status = match message {
                        NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 {
                            producer, ..
                        } => {
                            // [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex]
                            // A verified URL identity hint must suppress the
                            // receiver's own non-replica anchor.
                            assert_ne!(producer, receiver_producer);
                            let proof_index =
                                proof_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                            if proof_index == 0 {
                                StatusCode::UNPROCESSABLE_ENTITY
                            } else {
                                StatusCode::OK
                            }
                        }
                        NodeDiscoveryMessage::DescriptorAnnounce { .. }
                        | NodeDiscoveryMessage::SnapshotRequest { .. } => {
                            legacy_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                            StatusCode::OK
                        }
                        NodeDiscoveryMessage::SnapshotResponse { .. }
                        | NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                            StatusCode::OK
                        }
                    };
                    (
                        status,
                        Json(GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        }),
                    )
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let now = unix_now_secs();
    let announcements = directory_gossip_announcements_for_receiver(now, receiver_producer);
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();
    let mut peer_identity_hints = DiscoveryPeerIdentityHints::default();
    peer_identity_hints.observe_verified(url.clone(), receiver_producer);
    let mut execution = gossip_execution(
        &client,
        &peer_store,
        &announcements,
        now,
        Duration::from_secs(5),
    );
    execution.peer_identity_hints = Some(&peer_identity_hints);

    let report = Server::gossip_with_peer(execution, &url, self_descriptor).await;

    assert_eq!(calls.load(AtomicOrdering::SeqCst), 4);
    assert_eq!(proof_calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(legacy_calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(report.legacy_error, None);
    assert_eq!(
        report.directory_proof,
        DirectoryProofGossipOutcome {
            state: DirectoryProofGossipPeerState::Attempted,
            frames_attempted: 2,
            evidence_rejected: 1,
            result: DirectoryProofGossipResult::Accepted,
        }
    );
    mock_peer.abort();
}

#[tokio::test]
async fn outbound_gossip_does_not_retry_when_replica_is_unavailable() {
    let proof_calls = Arc::new(AtomicUsize::new(0));
    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let proof_calls_for_handler = Arc::clone(&proof_calls);
    let legacy_calls_for_handler = Arc::clone(&legacy_calls);
    let app = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "directory_descriptor_proof_gossip_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/discovery/gossip",
            post(move |Json(message): Json<NodeDiscoveryMessage>| {
                let proof_calls_for_handler = Arc::clone(&proof_calls_for_handler);
                let legacy_calls_for_handler = Arc::clone(&legacy_calls_for_handler);
                async move {
                    let status = match message {
                        NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                            proof_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                            StatusCode::SERVICE_UNAVAILABLE
                        }
                        NodeDiscoveryMessage::DescriptorAnnounce { .. }
                        | NodeDiscoveryMessage::SnapshotRequest { .. } => {
                            legacy_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                            StatusCode::OK
                        }
                        NodeDiscoveryMessage::SnapshotResponse { .. }
                        | NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                            StatusCode::OK
                        }
                    };
                    (
                        status,
                        Json(GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        }),
                    )
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let now = unix_now_secs();
    let announcements = [
        directory_gossip_announcement(now),
        directory_gossip_announcement(now),
    ];
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();

    let report = Server::gossip_with_peer(
        gossip_execution(
            &client,
            &peer_store,
            &announcements,
            now,
            Duration::from_secs(5),
        ),
        &url,
        self_descriptor,
    )
    .await;

    assert_eq!(proof_calls.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(legacy_calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(report.legacy_error, None);
    assert_eq!(
        report.directory_proof,
        DirectoryProofGossipOutcome {
            state: DirectoryProofGossipPeerState::Attempted,
            frames_attempted: 1,
            evidence_rejected: 0,
            result: DirectoryProofGossipResult::ReplicaUnavailable,
        }
    );
    mock_peer.abort();
}

#[tokio::test]
async fn outbound_gossip_skips_directory_proof_without_explicit_peer_support() {
    let calls = Arc::new(AtomicUsize::new(0));
    let proof_calls = Arc::new(AtomicUsize::new(0));
    let calls_for_handler = Arc::clone(&calls);
    let proof_calls_for_handler = Arc::clone(&proof_calls);
    let app = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "contract_version": "discovery_summary.v1"
                }))
            }),
        )
        .route(
            "/api/discovery/gossip",
            post(move |Json(message): Json<NodeDiscoveryMessage>| {
                let calls_for_handler = Arc::clone(&calls_for_handler);
                let proof_calls_for_handler = Arc::clone(&proof_calls_for_handler);
                async move {
                    calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    if matches!(
                        message,
                        NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. }
                    ) {
                        proof_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    }
                    Json(GossipResponse {
                        applied: PeerStoreImportReport::empty(),
                        response: None,
                    })
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let now = unix_now_secs();
    let announcement = directory_gossip_announcement(now);
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();

    let report = Server::gossip_with_peer(
        gossip_execution(
            &client,
            &peer_store,
            std::slice::from_ref(&announcement),
            now,
            Duration::from_secs(5),
        ),
        &url,
        self_descriptor,
    )
    .await;

    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(proof_calls.load(AtomicOrdering::SeqCst), 0);
    assert_eq!(report.legacy_error, None);
    assert_eq!(
        report.directory_proof,
        DirectoryProofGossipOutcome {
            state: DirectoryProofGossipPeerState::LegacyOnly,
            frames_attempted: 0,
            evidence_rejected: 0,
            result: DirectoryProofGossipResult::NotAttempted,
        }
    );
    assert_eq!(peer_store.status(now + 1).runtime.last_gossip_at, Some(now));
    mock_peer.abort();
}

#[tokio::test]
async fn outbound_gossip_keeps_legacy_exchange_when_directory_proof_is_rejected() {
    let calls = Arc::new(AtomicUsize::new(0));
    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let calls_for_handler = Arc::clone(&calls);
    let legacy_calls_for_handler = Arc::clone(&legacy_calls);
    let app = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                Json(serde_json::json!({
                    "protocol_features": {
                        "directory_descriptor_proof_gossip_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/discovery/gossip",
            post(move |Json(message): Json<NodeDiscoveryMessage>| {
                let calls_for_handler = Arc::clone(&calls_for_handler);
                let legacy_calls_for_handler = Arc::clone(&legacy_calls_for_handler);
                async move {
                    calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                    let status = match message {
                        NodeDiscoveryMessage::DirectoryDescriptorAnnounceV1 { .. } => {
                            StatusCode::UNPROCESSABLE_ENTITY
                        }
                        NodeDiscoveryMessage::DescriptorAnnounce { .. }
                        | NodeDiscoveryMessage::SnapshotRequest { .. } => {
                            legacy_calls_for_handler.fetch_add(1, AtomicOrdering::SeqCst);
                            StatusCode::OK
                        }
                        NodeDiscoveryMessage::SnapshotResponse { .. }
                        | NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { .. } => {
                            StatusCode::OK
                        }
                    };
                    (
                        status,
                        Json(GossipResponse {
                            applied: PeerStoreImportReport::empty(),
                            response: None,
                        }),
                    )
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let now = unix_now_secs();
    let announcement = directory_gossip_announcement(now);
    let self_descriptor =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300);
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();

    let report = Server::gossip_with_peer(
        gossip_execution(
            &client,
            &peer_store,
            std::slice::from_ref(&announcement),
            now,
            Duration::from_secs(5),
        ),
        &url,
        self_descriptor,
    )
    .await;

    assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);
    assert_eq!(legacy_calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(report.legacy_error, None);
    assert_eq!(
        report.directory_proof,
        DirectoryProofGossipOutcome {
            state: DirectoryProofGossipPeerState::Attempted,
            frames_attempted: 1,
            evidence_rejected: 1,
            result: DirectoryProofGossipResult::EvidenceRejected,
        }
    );
    assert_eq!(peer_store.status(now + 1).runtime.last_gossip_at, Some(now));
    mock_peer.abort();
}
