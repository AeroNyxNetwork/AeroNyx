// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn blind_relay_signature_admission_rejects_without_queueing() {
    // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] A saturated
    // verifier must reject before spawning more blocking work. Once the
    // permit is released, the same authenticated request remains valid.
    let admission = Arc::new(Semaphore::new(1));
    let held_permit = Arc::clone(&admission)
        .try_acquire_owned()
        .expect("reserve the only verification permit");
    let previous_hop = IdentityKeyPair::generate();
    let envelope = BlindRelayEnvelope {
        route_id: [0x3du8; 16],
        next_hop: IdentityKeyPair::generate().public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque bounded verification test".to_vec(),
        timestamp: now_secs(),
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);
    let request = PeerBlindRelayRequest {
        envelope,
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };

    let rejected = authenticate_peer_blind_relay_request_with_admission(
        Arc::clone(&admission),
        request.clone(),
    )
    .await;
    assert!(matches!(rejected, Err(BlindRelayError::Backpressure)));

    drop(held_permit);
    let authenticated =
        authenticate_peer_blind_relay_request_with_admission(admission, request.clone())
            .await
            .expect("released verifier must accept valid signed work");
    assert_eq!(authenticated.request.envelope.route_id, [0x3du8; 16]);
    assert_eq!(
        authenticated.failure_request_commitment,
        BlindRelayFailureReceipt::request_commitment(&request.envelope)
    );
}

#[test]
fn blind_relay_replay_commitment_binds_optional_onward_envelope() {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] The route id alone
    // is not an idempotency key: a substituted onward frame under the same
    // authenticated outer envelope must become a conflict, never an exact
    // replay or a second relay side effect.
    let previous_hop = IdentityKeyPair::generate();
    let middle = IdentityKeyPair::generate();
    let terminal = IdentityKeyPair::generate();
    let outer = BlindRelayEnvelope {
        route_id: [0x3Eu8; 16],
        next_hop: middle.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque outer layer".to_vec(),
        timestamp: 1_800_000_000,
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);
    let base = PeerBlindRelayRequest {
        envelope: outer,
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let mut with_onward = base.clone();
    with_onward.onward_envelope = Some(
        BlindRelayEnvelope {
            route_id: [0x3Fu8; 16],
            next_hop: terminal.public_key_bytes(),
            ttl: 1,
            encrypted_blob: b"opaque inner layer".to_vec(),
            timestamp: 1_800_000_000,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop),
    );

    let base_commitment = blind_relay_authenticated_request_commitment(&base)
        .expect("commit base blind-relay request");
    assert_eq!(
        blind_relay_authenticated_request_commitment(&base)
            .expect("repeat base blind-relay commitment"),
        base_commitment
    );
    assert_ne!(
        blind_relay_authenticated_request_commitment(&with_onward)
            .expect("commit blind-relay request with onward envelope"),
        base_commitment
    );
}

#[tokio::test]
async fn blind_relay_authentication_rejects_substituted_onward_envelope() {
    // [SIGNED-ONWARD-ENVELOPE 2026-08-24 by Codex] A transport intermediary
    // cannot alter the optional legacy onward ciphertext and make this node
    // sign the substituted frame for the terminal hop.
    let previous_hop = IdentityKeyPair::generate();
    let middle = IdentityKeyPair::generate();
    let terminal = IdentityKeyPair::generate();
    let outer = BlindRelayEnvelope {
        route_id: [0x40; 16],
        next_hop: middle.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque outer carrier".to_vec(),
        timestamp: now_secs(),
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);
    let mut onward = BlindRelayEnvelope {
        route_id: [0x41; 16],
        next_hop: terminal.public_key_bytes(),
        ttl: 1,
        encrypted_blob: b"signed opaque onward frame".to_vec(),
        timestamp: now_secs(),
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);
    onward.encrypted_blob = b"substituted opaque onward frame".to_vec();

    let result = authenticate_peer_blind_relay_request_with_admission(
        Arc::new(Semaphore::new(1)),
        PeerBlindRelayRequest {
            envelope: outer,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: Some(onward),
            onward_descriptor_hint: None,
        },
    )
    .await;
    assert!(matches!(result, Err(BlindRelayError::InvalidSignature)));
}

#[tokio::test]
async fn blind_relay_global_rate_limit_rejects_before_json_deserialization() {
    let (relay, path) = temp_chat_relay_with_peer_rate("blind-relay-global-rate-limit", 1);
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let peer_store = Arc::new(PeerStore::new());
    let app = build_chat_peer_router(
        Some(relay),
        sessions,
        udp,
        Arc::clone(&peer_store),
        Arc::new(IdentityKeyPair::generate()),
        Arc::new(reqwest::Client::new()),
        None,
    );

    // [BLIND-RELAY-GLOBAL-ADMISSION 2026-08-21 by Codex] Malformed JSON
    // proves the second attempt is rejected by middleware before Axum can
    // parse an identity, route, next hop, or opaque encrypted body.
    let request = || {
        Request::builder()
            .method("POST")
            .uri("/api/chat/peer/blind-relay")
            .header("content-type", "application/json")
            .body(Body::from("not-json"))
            .unwrap()
    };
    let first = app.clone().oneshot(request()).await.unwrap();
    let second = app.oneshot(request()).await.unwrap();

    assert_ne!(first.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = axum::body::to_bytes(second.into_body(), usize::MAX)
        .await
        .unwrap();
    let response: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(response.reason.as_deref(), Some("rate_limited"));
    let stats = peer_store.status(now_secs()).runtime.blind_relay;
    assert_eq!(stats.received, 1);
    assert_eq!(stats.rejected, 1);
    assert_eq!(stats.rate_limited, 1);

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_endpoint_terminal_accepts_opaque_blob_without_parsing() {
    let (relay, path) = temp_chat_relay("blind-relay-terminal-http");
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let peer_store = Arc::new(PeerStore::new());
    let http_client = Arc::new(reqwest::Client::new());
    let opaque_blob = br#"{"looks_like":"json","must_not_be_parsed":true}"#.to_vec();
    let now = now_secs();
    let envelope = BlindRelayEnvelope {
        route_id: [0x41u8; 16],
        next_hop: node_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: opaque_blob,
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);

    let app = build_chat_peer_router(
        Some(relay),
        sessions,
        udp,
        Arc::clone(&peer_store),
        node_identity,
        http_client,
        None,
    );
    let body = serde_json::to_vec(&PeerBlindRelayRequest {
        envelope,
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    })
    .unwrap();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();

    assert!(parsed.accepted);
    assert!(parsed.terminal);
    assert!(!parsed.forwarded);
    assert_eq!(parsed.ttl_remaining, 2);
    let blind_stats = peer_store.status(now + 10).runtime.blind_relay;
    assert_eq!(blind_stats.received, 1);
    assert_eq!(blind_stats.terminal, 1);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 0);
    assert!(peer_store
        .recent_audit_events()
        .iter()
        .any(|event| event.action == "blind_relay_terminal"));
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_handler_signs_exact_failure_response() {
    let (relay, path) = temp_chat_relay("blind-relay-signed-failure-http");
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let node_id = node_identity.public_key_bytes();
    let sessions = Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60)));
    let udp = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let now = now_secs();
    let request = PeerBlindRelayRequest {
        envelope: BlindRelayEnvelope {
            route_id: [0x42u8; 16],
            next_hop: node_id,
            ttl: 2,
            encrypted_blob: b"opaque expired failure request".to_vec(),
            timestamp: now - BLIND_RELAY_MAX_ENVELOPE_AGE_SECS - 1,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop),
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let app = build_chat_peer_router(
        Some(relay),
        sessions,
        udp,
        Arc::new(PeerStore::new()),
        node_identity,
        Arc::new(reqwest::Client::new()),
        None,
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed.reason.as_deref(), Some("timestamp_expired"));
    assert!(parsed.failure_receipt.is_some());
    assert_eq!(
        validate_downstream_failure_receipt(&parsed, &request, &node_id, now_secs(), true),
        Ok(true)
    );
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_handler_never_signs_unauthenticated_failure() {
    // [BLIND-RELAY-VERIFY-ADMISSION 2026-08-21 by Codex] An attacker may
    // choose both the ciphertext and claimed node id. Invalid work gets a
    // coarse retry/error bucket, but no node-authored receipt oracle.
    let (relay, path) = temp_chat_relay("blind-relay-unsigned-failure-http");
    let claimed_previous_hop = IdentityKeyPair::generate();
    let attacker = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let request = PeerBlindRelayRequest {
        envelope: BlindRelayEnvelope {
            route_id: [0x4au8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque unauthenticated failure".to_vec(),
            timestamp: now_secs(),
            signature: [0u8; 64],
        }
        .sign_with(&attacker),
        previous_hop_node_id: claimed_previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let app = build_chat_peer_router(
        Some(relay),
        Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        Arc::new(PeerStore::new()),
        node_identity,
        Arc::new(reqwest::Client::new()),
        None,
    );
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat/peer/blind-relay")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed.reason.as_deref(), Some("invalid_signature"));
    assert!(parsed.failure_receipt.is_none());
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_rejects_stale_timestamp_without_parsing_blob() {
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
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
    let envelope = BlindRelayEnvelope {
        route_id: [0x42u8; 16],
        next_hop: node_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: br#"{"opaque":"old route frame"}"#.to_vec(),
        timestamp: now - BLIND_RELAY_MAX_ENVELOPE_AGE_SECS - 1,
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

    assert!(matches!(result, Err(BlindRelayError::TimestampExpired)));
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.received, 1);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "timestamp_expired"
    }));
}

#[tokio::test]
async fn blind_relay_rejects_future_timestamp_without_parsing_blob() {
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
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
    let envelope = BlindRelayEnvelope {
        route_id: [0x43u8; 16],
        next_hop: node_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: br#"{"opaque":"future route frame"}"#.to_vec(),
        // [TEST-CLOCK-MARGIN 2026-10-10 by Claude] process_peer_blind_relay
        // reads the live clock again; a 1 s margin failed on a slow CI runner
        // whenever a second boundary passed between `now` and the check.
        timestamp: now + BLIND_RELAY_MAX_FUTURE_SKEW_SECS + 60,
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

    assert!(matches!(result, Err(BlindRelayError::TimestampInFuture)));
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.received, 1);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "timestamp_in_future"
    }));
}

#[tokio::test]
async fn onion_terminal_layer_is_peeled_and_delivered() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    let source = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let (relay, path) = temp_chat_relay("onion-terminal");
    let state = ChatPeerState {
        chat_relay: Some(Arc::clone(&relay)),
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

    // Single-hop onion addressed to this node; inner payload is a ChatEnvelope.
    let delivered_envelope = signed_envelope();
    let receiver = delivered_envelope.receiver;
    let inner = encode_envelope(&delivered_envelope).unwrap();
    let hop = OnionHop {
        node_id: node_identity.public_key_bytes(),
        // Build to the node's CURRENT rotating onion key — what the handler
        // peels with (not the identity-derived key).
        kem_pub: crate::services::onion_keys::current_public_key(),
    };
    let envelope = build_onion_envelope(&[hop], &inner, [0x55u8; 16], 4, now, &source).unwrap();
    assert!(is_onion_blob(&envelope.encrypted_blob));

    let result = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .unwrap();

    assert!(result.terminal);
    assert!(!result.forwarded);
    assert_eq!(result.reason.as_deref(), Some("onion_terminal_delivered"));
    result
        .delivery_receipt
        .as_ref()
        .expect("terminal onion delivery must return a signed receipt")
        .verify_expected_for_purpose(
            &[0x55u8; 16],
            &inner,
            OnionRoutePurpose::MessageRelay,
            &node_identity.public_key_bytes(),
        )
        .expect("receipt must bind the exact terminal payload, purpose, and node");
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.terminal, 1);
    assert_eq!(blind_stats.rejected, 0);
    let (messages, has_more) = relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .expect("terminal onion delivery should enter pending relay queue");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_id, delivered_envelope.message_id);

    let _ = std::fs::remove_file(path);
}

#[test]
fn onion_forward_reconstruction_is_byte_stable() {
    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] A middle hop that
    // crashes after sending must reconstruct exactly the same signed frame;
    // otherwise downstream durable replay sees a conflicting request.
    let previous_hop = IdentityKeyPair::generate();
    let middle = IdentityKeyPair::generate();
    let outer = BlindRelayEnvelope {
        route_id: [0x61; 16],
        next_hop: middle.public_key_bytes(),
        ttl: 4,
        encrypted_blob: b"opaque outer onion layer".to_vec(),
        timestamp: 1_800_000_123,
        signature: [0; 64],
    }
    .sign_with(&previous_hop);
    let next_hop = IdentityKeyPair::generate().public_key_bytes();
    let inner = b"opaque inner onion layer".to_vec();

    let first = build_forwarded_onion_envelope(&outer, next_hop, inner.clone(), &middle);
    let after_restart = build_forwarded_onion_envelope(&outer, next_hop, inner, &middle);

    assert_eq!(after_restart, first);
    assert_eq!(first.timestamp, outer.timestamp);
    assert_eq!(first.ttl, outer.ttl - 1);
}

#[tokio::test]
async fn onion_terminal_armed_claim_recovers_without_duplicate_storage() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Model a crash after
    // terminal custody succeeds but before the route ACK is sealed. Exact
    // retry takes over the armed claim and reuses idempotent store_pending.
    let source = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let (old_relay, path) = temp_chat_relay("onion-terminal-armed-recovery");
    let now = now_secs();
    let delivered_envelope = signed_envelope_at(now);
    let receiver = delivered_envelope.receiver;
    let inner = encode_envelope(&delivered_envelope).expect("encode terminal payload");
    let route_id = [0x62; 16];
    let request = PeerBlindRelayRequest {
        envelope: build_onion_envelope(
            &[OnionHop {
                node_id: node_identity.public_key_bytes(),
                kem_pub: crate::services::onion_keys::current_public_key(),
            }],
            &inner,
            route_id,
            4,
            now,
            &source,
        )
        .expect("build recoverable terminal onion"),
        previous_hop_node_id: source.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let request_commitment =
        blind_relay_authenticated_request_commitment(&request).expect("commit recoverable request");
    assert_eq!(
        old_relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve pre-crash route"),
        BlindRelayRouteAdmission::Reserved
    );
    old_relay
        .arm_blind_relay_route_effect(&route_id, &request_commitment, now)
        .expect("arm pre-crash route");
    old_relay
        .store_pending(&delivered_envelope)
        .expect("complete terminal custody before crash");
    drop(old_relay);

    let aged_at =
        now.saturating_sub(crate::services::chat_relay::BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1);
    Connection::open(&path)
        .expect("open crashed relay database")
        .execute(
            "UPDATE relay_blind_route_reservations
                 SET reserved_at = ?1, owner_acquired_at = ?1",
            [i64::try_from(aged_at).expect("fit test timestamp")],
        )
        .expect("age crashed owner lease");
    let recovered_relay = Arc::new(
        ChatRelayService::new(
            test_chat_config(path.to_string_lossy().into_owned()),
            [7u8; 32],
        )
        .expect("restart relay for armed reconciliation"),
    );
    let state = ChatPeerState {
        chat_relay: Some(Arc::clone(&recovered_relay)),
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

    let recovered = process_peer_blind_relay(state, request)
        .await
        .expect("reconcile armed terminal route");
    assert!(recovered.terminal);
    assert_eq!(
        recovered.reason.as_deref(),
        Some("onion_terminal_delivered")
    );
    let (messages, has_more) = recovered_relay
        .pull_pending(&receiver, 0, &[0; 16], 10)
        .expect("pull reconciled terminal custody");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_id, delivered_envelope.message_id);
    assert_eq!(
        encode_envelope(&messages[0].envelope).expect("encode recovered message"),
        encode_envelope(&delivered_envelope).expect("encode expected message"),
    );

    drop(recovered_relay);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn onion_middle_armed_claim_recovers_through_terminal_durable_replay() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    // [MIDDLE-HOP-ARMED-RECOVERY 2026-08-25 by Codex] Exercise the real
    // HTTP boundary on both sides of a crashed middle hop. The terminal
    // has already accepted durable custody, but the middle has not sealed
    // its upstream ACK. Restart recovery must reconstruct byte-identical
    // downstream work, receive the terminal's durable replay, and avoid a
    // second pending message.
    let source = IdentityKeyPair::generate();
    let middle_identity = Arc::new(IdentityKeyPair::generate());
    let terminal_identity = Arc::new(IdentityKeyPair::generate());
    let now = now_secs();
    let delivered_envelope = signed_envelope_at(now);
    let receiver = delivered_envelope.receiver;
    let terminal_payload =
        encode_envelope(&delivered_envelope).expect("encode terminal message payload");

    let (terminal_relay, terminal_path) = temp_chat_relay("onion-terminal-replay-target");
    let terminal_peer_store = Arc::new(PeerStore::new());
    let terminal_app = build_chat_peer_router(
        Some(Arc::clone(&terminal_relay)),
        Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        Arc::clone(&terminal_peer_store),
        Arc::clone(&terminal_identity),
        Arc::new(reqwest::Client::new()),
        None,
    );
    let terminal_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let terminal_endpoint = format!("http://{}", terminal_listener.local_addr().unwrap());
    let terminal_server = tokio::spawn(async move {
        axum::serve(terminal_listener, terminal_app).await.unwrap();
    });

    let terminal_descriptor = signed_chat_relay_peer_descriptor_for(
        terminal_identity.as_ref(),
        terminal_endpoint.clone(),
        now,
        now + 300,
    );
    let middle_peer_store = Arc::new(PeerStore::new());
    middle_peer_store
        .upsert_verified_from_source(terminal_descriptor.clone(), now, "gossip_snapshot")
        .expect("install terminal descriptor at middle hop");

    let (old_middle_relay, middle_path) = temp_chat_relay("onion-middle-armed-recovery");
    let old_middle_state = ChatPeerState {
        chat_relay: Some(Arc::clone(&old_middle_relay)),
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&middle_peer_store),
        node_identity: Arc::clone(&middle_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let route_id = [0x69; 16];
    let request = PeerBlindRelayRequest {
        envelope: build_onion_envelope(
            &[
                OnionHop {
                    node_id: middle_identity.public_key_bytes(),
                    kem_pub: crate::services::onion_keys::current_public_key(),
                },
                OnionHop {
                    node_id: terminal_identity.public_key_bytes(),
                    kem_pub: crate::services::onion_keys::current_public_key(),
                },
            ],
            &terminal_payload,
            route_id,
            4,
            now,
            &source,
        )
        .expect("build recoverable two-hop onion"),
        previous_hop_node_id: source.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let request_commitment = blind_relay_authenticated_request_commitment(&request)
        .expect("commit recoverable middle-hop request");
    assert_eq!(
        old_middle_relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve pre-crash middle route"),
        BlindRelayRouteAdmission::Reserved
    );
    old_middle_relay
        .arm_blind_relay_route_effect(&route_id, &request_commitment, now)
        .expect("arm pre-crash middle route");

    let peel = try_open_onion_layer(
        &request.envelope.encrypted_blob,
        &crate::services::onion_keys::peel_secrets(now),
    )
    .expect("peel pre-crash middle layer");
    assert_eq!(peel.next_hop, Some(terminal_identity.public_key_bytes()));
    let forwarded_envelope = build_forwarded_onion_envelope(
        &request.envelope,
        terminal_identity.public_key_bytes(),
        peel.inner,
        middle_identity.as_ref(),
    );
    let terminal_url =
        blind_peer_relay_url(&terminal_endpoint).expect("construct canonical terminal relay URL");
    let first_forwarded_request = PeerBlindRelayRequest {
        envelope: forwarded_envelope,
        previous_hop_node_id: middle_identity.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let first_terminal_ack = forward_blind_relay_with_retry(
        &old_middle_state,
        &terminal_url,
        &terminal_descriptor,
        prepare_blind_relay_forward_request(first_forwarded_request.clone())
            .await
            .expect("prepare recoverable terminal request"),
        now,
    )
    .await
    .expect("terminal accepts custody before middle crash");
    assert!(first_terminal_ack.response.terminal);
    assert!(first_terminal_ack.response.delivery_receipt.is_some());
    let terminal_request_commitment =
        blind_relay_authenticated_request_commitment(&first_forwarded_request)
            .expect("commit terminal replay request");
    let sealed_terminal_response = match terminal_relay
        .reserve_blind_relay_route(&route_id, &terminal_request_commitment)
        .expect("read terminal sealed route")
    {
        BlindRelayRouteAdmission::Completed { response, .. } => response,
        admission => panic!("terminal route was not sealed: {admission:?}"),
    };
    let sealed_terminal_response = decode_durable_blind_relay_response(&sealed_terminal_response)
        .expect("decode terminal sealed response");
    validate_completed_blind_relay_response(&sealed_terminal_response)
        .expect("validate terminal sealed response");
    assert_eq!(sealed_terminal_response, first_terminal_ack.response);

    drop(old_middle_state);
    drop(old_middle_relay);
    let aged_at =
        now.saturating_sub(crate::services::chat_relay::BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1);
    Connection::open(&middle_path)
        .expect("open crashed middle database")
        .execute(
            "UPDATE relay_blind_route_reservations
                 SET reserved_at = ?1, owner_acquired_at = ?1",
            [i64::try_from(aged_at).expect("fit middle lease timestamp")],
        )
        .expect("age crashed middle owner lease");

    let recovered_middle_relay = Arc::new(
        ChatRelayService::new(
            test_chat_config(middle_path.to_string_lossy().into_owned()),
            [7u8; 32],
        )
        .expect("restart middle relay for armed reconciliation"),
    );
    let recovered_middle_state = ChatPeerState {
        chat_relay: Some(Arc::clone(&recovered_middle_relay)),
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&middle_peer_store),
        node_identity: Arc::clone(&middle_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let recovered_peel = try_open_onion_layer(
        &request.envelope.encrypted_blob,
        &crate::services::onion_keys::peel_secrets(now),
    )
    .expect("peel recovered middle layer");
    let recovered_forwarded_request = PeerBlindRelayRequest {
        envelope: build_forwarded_onion_envelope(
            &request.envelope,
            terminal_identity.public_key_bytes(),
            recovered_peel.inner,
            middle_identity.as_ref(),
        ),
        previous_hop_node_id: middle_identity.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    assert_eq!(
        serde_json::to_vec(&recovered_forwarded_request)
            .expect("encode recovered downstream request"),
        serde_json::to_vec(&first_forwarded_request).expect("encode original downstream request"),
        "middle restart changed authenticated downstream request bytes"
    );
    let recovered = process_peer_blind_relay(recovered_middle_state, request)
        .await
        .unwrap_or_else(|error| {
            panic!(
                "recover middle route through terminal durable replay: {error:?}; middle={:?}; terminal={:?}; terminal_events={:?}",
                middle_peer_store.status(now + 1).runtime.blind_relay,
                terminal_peer_store.status(now + 1).runtime.blind_relay,
                terminal_peer_store.recent_audit_events(),
            )
        });
    assert!(recovered.accepted);
    assert!(recovered.forwarded);
    assert!(!recovered.terminal);
    assert_eq!(recovered.reason.as_deref(), Some("onion_forwarded"));
    assert_eq!(
        recovered.delivery_receipt,
        first_terminal_ack.response.delivery_receipt
    );

    let (messages, has_more) = terminal_relay
        .pull_pending(&receiver, 0, &[0; 16], 10)
        .expect("pull terminal custody after middle recovery");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_id, delivered_envelope.message_id);
    let terminal_stats = terminal_peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(terminal_stats.terminal, 1);
    assert_eq!(terminal_stats.replay_dropped, 1);
    let recovery_status = recovered_middle_relay.peer_status().blind_route_recovery;
    assert_eq!(recovery_status.attempted_total, 1);
    assert_eq!(recovery_status.completed_total, 1);
    assert_eq!(recovery_status.deferred_total, 0);
    assert_eq!(recovery_status.last_outcome.as_deref(), Some("completed"));

    terminal_server.abort();
    let _ = terminal_server.await;
    drop(recovered_middle_relay);
    drop(terminal_relay);
    let _ = std::fs::remove_file(middle_path);
    let _ = std::fs::remove_file(terminal_path);
}

#[tokio::test]
async fn onion_terminal_rejects_same_message_id_with_different_ciphertext() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    let source = IdentityKeyPair::generate();
    let chat_sender = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let (relay, path) = temp_chat_relay("onion-terminal-id-conflict");
    let state = ChatPeerState {
        chat_relay: Some(Arc::clone(&relay)),
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
    let receiver = [0xA2; 32];
    let message_id = [0xA3; 16];
    let make_chat_envelope = |ciphertext: &[u8]| {
        let mut envelope = ChatEnvelope {
            message_id,
            sender: chat_sender.public_key_bytes(),
            receiver,
            timestamp: now,
            ciphertext: ciphertext.to_vec(),
            nonce: [0xA4; 24],
            content_type: ChatContentType::Text,
            signature: [0u8; 64],
        };
        envelope.signature = chat_sender.sign(&envelope.sign_data());
        envelope
    };
    let hop = OnionHop {
        node_id: node_identity.public_key_bytes(),
        kem_pub: crate::services::onion_keys::current_public_key(),
    };
    let make_request = |route_id, chat_envelope: &ChatEnvelope| {
        let payload = encode_envelope(chat_envelope).expect("encode terminal chat envelope");
        PeerBlindRelayRequest {
            envelope: build_onion_envelope(
                std::slice::from_ref(&hop),
                &payload,
                route_id,
                4,
                now,
                &source,
            )
            .expect("build terminal onion envelope"),
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        }
    };

    let original = make_chat_envelope(b"first opaque ciphertext");
    let first = process_peer_blind_relay(state.clone(), make_request([0xA5; 16], &original))
        .await
        .expect("first terminal envelope should be durably accepted");
    assert!(first.delivery_receipt.is_some());

    // [DURABLE-RECEIPT-BOUNDARY 2026-08-15 by Codex] A fresh route can
    // legitimately retry the same exact envelope, but reusing its message
    // ID for different signed bytes must never produce a terminal receipt.
    let conflict = make_chat_envelope(b"different opaque ciphertext");
    let rejected = process_peer_blind_relay(state, make_request([0xA6; 16], &conflict)).await;
    assert!(matches!(rejected, Err(BlindRelayError::ForwardFailed)));

    let (messages, has_more) = relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .expect("original durable envelope should remain readable");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(
        encode_envelope(&messages[0].envelope).expect("re-encode stored envelope"),
        encode_envelope(&original).expect("re-encode original envelope")
    );
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.terminal, 1);
    assert_eq!(blind_stats.rejected, 1);

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn onion_terminal_persists_anonymous_blind_vault_put_idempotently() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    let source = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let now = now_secs();
    let now_ms = now.saturating_mul(1_000);
    let (_directory, vault, put) = temp_blind_vault_with_put(node_identity.as_ref(), now_ms);
    let encoded_put =
        encode_blind_vault_frame(&BlindVaultFrame::Put(put)).expect("encode vault put");
    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: Some(Arc::clone(&vault)),
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

    let make_envelope = |route_id| {
        build_onion_envelope(
            &[OnionHop {
                node_id: node_identity.public_key_bytes(),
                kem_pub: crate::services::onion_keys::current_public_key(),
            }],
            &encoded_put,
            route_id,
            4,
            now,
            &source,
        )
        .expect("build vault terminal onion")
    };
    let first_route = [0x67; 16];
    let result = process_peer_blind_relay(
        state.clone(),
        PeerBlindRelayRequest {
            envelope: make_envelope(first_route),
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .expect("signed anonymous put should reach Blind Vault");

    assert!(result.accepted);
    assert!(result.terminal);
    assert!(!result.forwarded);
    assert_eq!(result.reason.as_deref(), Some("onion_terminal_delivered"));
    result
        .delivery_receipt
        .as_ref()
        .expect("vault acceptance must return a route-safe terminal receipt")
        .verify_expected_for_purpose(
            &first_route,
            &encoded_put,
            OnionRoutePurpose::BlindVaultPut,
            &node_identity.public_key_bytes(),
        )
        .expect("receipt must bind exact encoded put and purpose without exposing metadata");
    let public_ack = serde_json::to_string(&result).expect("serialize terminal ACK");
    assert!(!public_ack.contains("lease_id"));
    assert!(!public_ack.contains("object_id"));
    assert!(!public_ack.contains("ciphertext"));

    let status = vault.status(now_ms + 1).expect("vault status after put");
    assert_eq!(status.live_objects, 1);
    assert_eq!(status.live_ciphertext_bytes, 4 * 1024);

    // A source may rebuild a route after losing the first ACK. Blind Vault
    // handles the exact Put idempotently even when the relay route differs.
    process_peer_blind_relay(
        state.clone(),
        PeerBlindRelayRequest {
            envelope: make_envelope([0x68; 16]),
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .expect("same immutable put through a fresh route should be idempotent");
    assert_eq!(
        vault
            .status(now_ms + 2)
            .expect("vault status after retry")
            .live_objects,
        1
    );

    assert!(matches!(
        prepare_onion_terminal_payload(&state, [0; 16], b"ANBV".to_vec(), now).await,
        Err(BlindRelayError::OnionTerminalPayloadRejected)
    ));
}

#[tokio::test]
async fn onion_terminal_requires_chat_relay_delivery_before_ack() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    let source = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let replay_registry: Arc<dyn BlindRelayReplayRegistry> =
        Arc::new(BlindRelayReplayDomain::default());
    let abuse_guard: Arc<dyn BlindRelayAbusePolicy> = Arc::new(BlindRelayAbuseDomain::default());
    let failed_state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&node_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::clone(&replay_registry),
        blind_relay_abuse_guard: Arc::clone(&abuse_guard),
    };
    let now = now_secs();

    let delivered_envelope = signed_envelope();
    let receiver = delivered_envelope.receiver;
    let inner = encode_envelope(&delivered_envelope).unwrap();
    let hop = OnionHop {
        node_id: node_identity.public_key_bytes(),
        kem_pub: crate::services::onion_keys::current_public_key(),
    };
    let envelope = build_onion_envelope(&[hop], &inner, [0x56u8; 16], 4, now, &source).unwrap();

    let result = process_peer_blind_relay(
        failed_state,
        PeerBlindRelayRequest {
            envelope: envelope.clone(),
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await;

    assert!(matches!(result, Err(BlindRelayError::ForwardFailed)));
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.terminal, 0);
    assert_eq!(blind_stats.rejected, 1);

    let (relay, path) = temp_chat_relay("onion-terminal-retry-after-relay-failure");
    let retry_state = ChatPeerState {
        chat_relay: Some(Arc::clone(&relay)),
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&node_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::clone(&replay_registry),
        blind_relay_abuse_guard: Arc::clone(&abuse_guard),
    };

    // A terminal delivery failure must release the route id from the
    // replay cache. Otherwise a transient ChatRelay outage would make the
    // sender's retry look like a duplicate replay and permanently strand
    // the E2E-encrypted message.
    let retry = process_peer_blind_relay(
        retry_state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .expect("terminal delivery retry should not be blocked by replay cache");

    assert!(retry.terminal);
    assert_eq!(retry.reason.as_deref(), Some("onion_terminal_delivered"));
    let (messages, has_more) = relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .expect("retry should store the terminal onion payload");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_id, delivered_envelope.message_id);

    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn onion_layer_with_wrong_node_key_is_rejected() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, OnionHop};

    let source = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let wrong_target = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    let (relay, path) = temp_chat_relay("onion-wrong-node-key-retry");
    let state = ChatPeerState {
        chat_relay: Some(relay),
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

    // Layer is sealed to a different node's KEM key, but addressed (next_hop)
    // to this node — peel must fail without leaking anything.
    let inner = encode_envelope(&signed_envelope()).unwrap();
    let sealed_for_wrong = OnionHop {
        node_id: node_identity.public_key_bytes(),
        kem_pub: wrong_target.x25519_public_key_bytes(),
    };
    let envelope =
        build_onion_envelope(&[sealed_for_wrong], &inner, [0x56u8; 16], 4, now, &source).unwrap();

    // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Peeling is a
    // pure preflight step. Its failure must release the durable unarmed
    // claim so an identical retry is classified by payload validation,
    // never stranded as an in-flight side effect.
    for _ in 0..2 {
        let result = process_peer_blind_relay(
            state.clone(),
            PeerBlindRelayRequest {
                envelope: envelope.clone(),
                previous_hop_node_id: source.public_key_bytes(),
                onward_envelope: None,
                onward_descriptor_hint: None,
            },
        )
        .await;

        assert!(matches!(result, Err(BlindRelayError::OnionPeelFailed)));
    }
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.terminal, 0);
    assert_eq!(blind_stats.rejected, 2);
    drop(state);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn onion_middle_allows_fresh_signed_peer_before_routeability_probe() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, is_onion_blob, OnionHop};

    let terminal_requests: Arc<Mutex<Vec<PeerBlindRelayRequest>>> =
        Arc::new(Mutex::new(Vec::new()));
    let terminal_requests_for_route = Arc::clone(&terminal_requests);
    let terminal_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let terminal_requests_for_request = Arc::clone(&terminal_requests_for_route);
            async move {
                terminal_requests_for_request.lock().unwrap().push(request);
                Json(PeerBlindRelayResponse {
                    accepted: true,
                    terminal: true,
                    forwarded: false,
                    ttl_remaining: 2,
                    reason: Some("terminal_next_hop".to_string()),
                    delivery_receipt: None,
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: None,
                })
                .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, terminal_app).await.unwrap();
    });

    let now = now_secs();
    let source = IdentityKeyPair::generate();
    let middle_identity = Arc::new(IdentityKeyPair::generate());
    let terminal_identity = IdentityKeyPair::generate();
    let terminal_node_id = terminal_identity.public_key_bytes();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(
                &terminal_identity,
                terminal_endpoint,
                now,
                now + 300,
            ),
            now,
            "gossip_snapshot",
        )
        .unwrap();

    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&middle_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };

    // Build a true two-layer onion. The middle hop can peel only the outer
    // layer and must forward the remaining opaque onion blob without knowing
    // the final payload or the terminal's user-level receiver.
    let inner = encode_envelope(&signed_envelope()).unwrap();
    let middle_hop = OnionHop {
        node_id: middle_identity.public_key_bytes(),
        kem_pub: crate::services::onion_keys::current_public_key(),
    };
    let terminal_hop = OnionHop {
        node_id: terminal_node_id,
        kem_pub: terminal_identity.x25519_public_key_bytes(),
    };
    let envelope = build_onion_envelope(
        &[middle_hop, terminal_hop],
        &inner,
        [0x66u8; 16],
        4,
        now,
        &source,
    )
    .unwrap();

    let response = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .unwrap();

    server.abort();

    assert!(response.accepted);
    assert!(response.forwarded);
    assert!(!response.terminal);
    assert_eq!(response.reason.as_deref(), Some("onion_forwarded"));

    let terminal_requests = terminal_requests.lock().unwrap();
    assert_eq!(terminal_requests.len(), 1);
    let terminal_request = &terminal_requests[0];
    assert_eq!(
        terminal_request.previous_hop_node_id,
        middle_identity.public_key_bytes()
    );
    assert_eq!(terminal_request.envelope.next_hop, terminal_node_id);
    assert!(is_onion_blob(&terminal_request.envelope.encrypted_blob));
    assert!(terminal_request.onward_envelope.is_none());

    let route_status = peer_store.route_candidate_status(now + 5);
    let route_row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == hex::encode(&terminal_node_id[..4]))
        .expect("terminal route row should remain visible");
    assert!(route_row.routeability_ready);
    assert_eq!(route_row.routeability_state, "reachable");
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 1);
    assert_eq!(blind_stats.rejected, 0);
}

#[tokio::test]
async fn two_hop_onion_relay_delivers_real_ciphertext_payload_to_terminal_store() {
    use aeronyx_core::protocol::onion::{build_onion_envelope, open_onion_layer, OnionHop};

    let now = now_secs();
    let source = IdentityKeyPair::generate();
    let middle_identity = Arc::new(IdentityKeyPair::generate());
    let terminal_identity = IdentityKeyPair::generate();
    let terminal_receipt_identity = terminal_identity.clone();
    let terminal_node_id = terminal_identity.public_key_bytes();
    let terminal_secret = terminal_identity.to_x25519().0;

    let chat_sender = IdentityKeyPair::generate();
    let route_id = [0x7au8; 16];
    let receiver = [0x8bu8; 32];
    let mut delivered_envelope = ChatEnvelope {
        message_id: route_id,
        sender: chat_sender.public_key_bytes(),
        receiver,
        timestamp: now,
        ciphertext: b"real e2e ciphertext payload carried through two hops".to_vec(),
        nonce: [0x9cu8; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    delivered_envelope.signature = chat_sender.sign(&delivered_envelope.sign_data());
    let encoded_chat = encode_envelope(&delivered_envelope).unwrap();

    let (terminal_relay, terminal_db_path) = temp_chat_relay("two-hop-onion-terminal-store");
    let terminal_relay_for_route = Arc::clone(&terminal_relay);
    let terminal_previous_hops: Arc<Mutex<Vec<[u8; 32]>>> = Arc::new(Mutex::new(Vec::new()));
    let terminal_previous_hops_for_route = Arc::clone(&terminal_previous_hops);

    let terminal_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let terminal_relay_for_request = Arc::clone(&terminal_relay_for_route);
            let terminal_receipt_identity = terminal_receipt_identity.clone();
            let terminal_previous_hops_for_request = Arc::clone(&terminal_previous_hops_for_route);
            async move {
                if validate_blind_relay_envelope(
                    &request.envelope,
                    &request.previous_hop_node_id,
                    now_secs(),
                )
                .is_err()
                {
                    return StatusCode::BAD_REQUEST.into_response();
                }

                let peel =
                    match open_onion_layer(&request.envelope.encrypted_blob, &terminal_secret) {
                        Ok(peel) => peel,
                        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                    };
                if peel.next_hop.is_some() {
                    return StatusCode::BAD_REQUEST.into_response();
                }

                let inner = match decode_envelope(&peel.inner) {
                    Ok(envelope) => envelope,
                    Err(_) => return StatusCode::BAD_REQUEST.into_response(),
                };
                if validate_peer_envelope(&inner, now_secs()).is_err() {
                    return StatusCode::BAD_REQUEST.into_response();
                }
                if terminal_relay_for_request.store_pending(&inner).is_err() {
                    return StatusCode::INTERNAL_SERVER_ERROR.into_response();
                }
                let delivery_receipt = BlindRelayDeliveryReceipt::accepted_for_purpose(
                    request.envelope.route_id,
                    &peel.inner,
                    OnionRoutePurpose::MessageRelay,
                    now_secs(),
                    &terminal_receipt_identity,
                );
                terminal_previous_hops_for_request
                    .lock()
                    .unwrap()
                    .push(request.previous_hop_node_id);

                Json(PeerBlindRelayResponse {
                    accepted: true,
                    terminal: true,
                    forwarded: false,
                    ttl_remaining: request.envelope.ttl,
                    reason: Some("onion_terminal_delivered".to_string()),
                    delivery_receipt: Some(delivery_receipt),
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: None,
                })
                .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, terminal_app).await.unwrap();
    });

    let mut terminal_descriptor = NodeDescriptor::new(
        terminal_node_id,
        now,
        now,
        now + 300,
        "test-terminal-onion-peer",
    )
    .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
    terminal_descriptor.public_endpoint = Some(terminal_endpoint);
    terminal_descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal_descriptor.capacity = NodeCapacity {
        max_sessions: 32,
        max_bps: None,
        max_pps: None,
    };
    let terminal_descriptor =
        SignedNodeDescriptor::sign(terminal_descriptor, &terminal_identity).unwrap();

    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(terminal_descriptor, now, "gossip_snapshot")
        .unwrap();

    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&middle_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };

    let middle_hop = OnionHop {
        node_id: middle_identity.public_key_bytes(),
        kem_pub: crate::services::onion_keys::current_public_key(),
    };
    let terminal_hop = OnionHop {
        node_id: terminal_node_id,
        kem_pub: terminal_identity.x25519_public_key_bytes(),
    };
    let envelope = build_onion_envelope(
        &[middle_hop, terminal_hop],
        &encoded_chat,
        route_id,
        2,
        now,
        &source,
    )
    .unwrap();

    let response = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: source.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .expect("middle hop should forward real onion payload to terminal");

    server.abort();

    assert!(response.accepted);
    assert!(response.forwarded);
    assert!(!response.terminal);
    assert_eq!(response.ttl_remaining, 1);
    assert_eq!(response.reason.as_deref(), Some("onion_forwarded"));
    response
        .delivery_receipt
        .as_ref()
        .expect("middle hop must propagate the terminal receipt")
        .verify_expected_for_purpose(
            &route_id,
            &encoded_chat,
            OnionRoutePurpose::MessageRelay,
            &terminal_node_id,
        )
        .expect("propagated receipt must retain terminal purpose binding");

    let previous_hops = terminal_previous_hops.lock().unwrap();
    assert_eq!(
        previous_hops.as_slice(),
        &[middle_identity.public_key_bytes()]
    );
    drop(previous_hops);

    let (messages, has_more) = terminal_relay
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .expect("terminal should store the delivered E2E envelope");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_id, delivered_envelope.message_id);
    assert_eq!(
        messages[0].envelope.ciphertext,
        delivered_envelope.ciphertext
    );
    assert_eq!(messages[0].envelope.nonce, delivered_envelope.nonce);
    assert_eq!(messages[0].envelope.sender, delivered_envelope.sender);
    assert_eq!(messages[0].envelope.receiver, delivered_envelope.receiver);

    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 1);
    assert_eq!(blind_stats.rejected, 0);

    let _ = std::fs::remove_file(terminal_db_path);
}

#[tokio::test]
async fn blind_relay_http_gate_requires_durable_replay_before_body_parse() {
    // [DURABLE-BLIND-RELAY-ADMISSION 2026-08-24 by Codex] Invalid JSON is
    // deliberate: a missing replay store must stop the request before the
    // extractor can parse or allocate for an attacker-controlled envelope.
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::new(IdentityKeyPair::generate()),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
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

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = axum::body::to_bytes(response.into_body(), PEER_ACK_RESPONSE_MAX_BYTES)
        .await
        .unwrap();
    let rejection: PeerBlindRelayResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        rejection.reason.as_deref(),
        Some("replay_protection_unavailable")
    );
    let stats = peer_store.status(now_secs()).runtime.blind_relay;
    assert_eq!(stats.rejected, 1);
    assert_eq!(stats.terminal, 0);
    assert_eq!(stats.forwarded, 0);
    assert_eq!(
        rejection.delivery_receipt, None,
        "unavailable admission must not manufacture delivery evidence"
    );
    assert_eq!(
        rejection.failure_receipt, None,
        "unavailable admission must not sign failure evidence"
    );
}

#[tokio::test]
async fn blind_relay_rejects_immediate_previous_hop_loop_without_parsing_blob() {
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
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
        route_id: [0x44u8; 16],
        next_hop: previous_hop.public_key_bytes(),
        ttl: 2,
        encrypted_blob: br#"{"opaque":"must_not_be_parsed"}"#.to_vec(),
        timestamp: now_secs(),
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

    assert!(matches!(result, Err(BlindRelayError::RouteLoop)));
    let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
    assert_eq!(blind_stats.received, 1);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.loop_detected, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "route_loop"
    }));
}

#[test]
fn blind_relay_route_lease_releases_cancellation_and_commits_exact_ack() {
    // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Cancellation
    // releases an unarmed claim, but preserves an armed claim whose effect
    // may have happened. Completion still publishes the exact bounded ACK.
    let replay_registry: Arc<dyn BlindRelayReplayRegistry> =
        Arc::new(BlindRelayReplayDomain::default());
    let route_id = [0x48u8; 16];
    let request_commitment = [0xA8u8; 32];
    let started_at = 1_800_000_000;

    let first_generation = match replay_registry.observe(route_id, request_commitment, started_at) {
        BlindRelayRouteReplayDecision::New { generation } => generation,
        decision => panic!("unexpected replay decision: {decision:?}"),
    };
    drop(BlindRelayRouteLease::local(
        Arc::clone(&replay_registry),
        route_id,
        request_commitment,
        first_generation,
    ));
    let armed_generation =
        match replay_registry.observe(route_id, request_commitment, started_at + 1) {
            BlindRelayRouteReplayDecision::New { generation } => generation,
            decision => panic!("unexpected replay decision: {decision:?}"),
        };

    let mut armed_lease = BlindRelayRouteLease::local(
        Arc::clone(&replay_registry),
        route_id,
        request_commitment,
        armed_generation,
    );
    armed_lease.arm_effect(started_at + 2).unwrap();
    drop(armed_lease);
    assert_eq!(
        replay_registry.observe(route_id, request_commitment, started_at + 3),
        BlindRelayRouteReplayDecision::InFlight
    );
    assert_eq!(
        replay_registry.release(route_id, request_commitment, armed_generation),
        BlindRelayReplayMutation::Applied
    );
    let completion_generation =
        match replay_registry.observe(route_id, request_commitment, started_at + 4) {
            BlindRelayRouteReplayDecision::New { generation } => generation,
            decision => panic!("unexpected replay decision: {decision:?}"),
        };

    let response = PeerBlindRelayResponse {
        accepted: true,
        terminal: false,
        forwarded: true,
        ttl_remaining: 1,
        reason: Some("forwarded".to_string()),
        delivery_receipt: None,
        success_receipt: None,
        failure_receipt: None,
        opaque_terminal_response_b64: None,
    };
    BlindRelayRouteLease::local(
        Arc::clone(&replay_registry),
        route_id,
        request_commitment,
        completion_generation,
    )
    .complete(started_at + 5, response.clone())
    .unwrap();
    assert_eq!(
        replay_registry.observe(route_id, request_commitment, started_at + 6),
        BlindRelayRouteReplayDecision::Completed(Box::new(response))
    );
}

#[test]
fn recovered_blind_route_lease_reports_deferred_without_route_dimensions() {
    // [BLIND-ROUTE-RECOVERY-STATUS 2026-08-25 by Codex] A cancelled
    // takeover remains durably armed and emits only one aggregate deferred
    // transition. No route, request, peer, endpoint, or reason is retained.
    let (old_relay, path) = temp_chat_relay("blind-route-recovery-deferred");
    let route_id = [0x49; 16];
    let request_commitment = [0xA9; 32];
    let now = now_secs();
    assert_eq!(
        old_relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve old process route"),
        BlindRelayRouteAdmission::Reserved
    );
    old_relay
        .arm_blind_relay_route_effect(&route_id, &request_commitment, now)
        .expect("arm old process route");
    drop(old_relay);

    let aged_at =
        now.saturating_sub(crate::services::chat_relay::BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1);
    Connection::open(&path)
        .expect("open deferred recovery database")
        .execute(
            "UPDATE relay_blind_route_reservations
                 SET reserved_at = ?1, owner_acquired_at = ?1",
            [i64::try_from(aged_at).expect("fit deferred lease timestamp")],
        )
        .expect("age prior process lease");

    let recovered_relay = Arc::new(
        ChatRelayService::new(
            test_chat_config(path.to_string_lossy().into_owned()),
            [7; 32],
        )
        .expect("restart deferred recovery relay"),
    );
    assert_eq!(
        recovered_relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("take over armed route"),
        BlindRelayRouteAdmission::ReservedForRecovery
    );
    drop(BlindRelayRouteLease::durable(
        Arc::clone(&recovered_relay),
        route_id,
        request_commitment,
        true,
    ));

    let status = recovered_relay.peer_status().blind_route_recovery;
    assert_eq!(status.attempted_total, 1);
    assert_eq!(status.completed_total, 0);
    assert_eq!(status.deferred_total, 1);
    assert_eq!(status.last_outcome.as_deref(), Some("deferred"));
    assert!(status.last_event_at.is_some());

    drop(recovered_relay);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_replays_completed_response_without_forwarding_again() {
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
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
    let envelope = BlindRelayEnvelope {
        route_id: [0x45u8; 16],
        next_hop: node_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque encrypted replay candidate".to_vec(),
        timestamp: now_secs(),
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);

    let first = process_peer_blind_relay(
        state.clone(),
        PeerBlindRelayRequest {
            envelope: envelope.clone(),
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .unwrap();
    let duplicate = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .unwrap();

    assert!(first.terminal);
    assert_eq!(duplicate, first);
    let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
    assert_eq!(blind_stats.received, 2);
    assert_eq!(blind_stats.terminal, 1);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.replay_dropped, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "duplicate_route"
    }));
    assert!(!blind_relay_reason_counts_toward_quarantine(
        "duplicate_route"
    ));
}

#[tokio::test]
async fn blind_relay_in_flight_duplicate_never_returns_false_acceptance() {
    // [IDEMPOTENT-RELAY-ACK 2026-08-11 by Codex] A concurrent retry must
    // remain retryable until the owner attempt publishes a durable result.
    // Returning an accepted replay here could lose the route if that owner
    // subsequently fails.
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let route_id = [0x47u8; 16];
    let envelope = BlindRelayEnvelope {
        route_id,
        next_hop: node_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque concurrent replay candidate".to_vec(),
        timestamp: now_secs(),
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);
    let request = PeerBlindRelayRequest {
        envelope,
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let request_commitment = blind_relay_authenticated_request_commitment(&request).unwrap();
    let (relay, path) = temp_chat_relay("blind-relay-in-flight");
    assert_eq!(
        relay
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .unwrap(),
        BlindRelayRouteAdmission::Reserved
    );
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
        chat_relay: Some(relay),
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
    let error = process_peer_blind_relay(state, request).await.unwrap_err();

    assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(matches!(error, BlindRelayError::RouteInFlight));
    let blind_stats = peer_store.status(now_secs() + 1).runtime.blind_relay;
    assert_eq!(blind_stats.terminal, 0);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "route_in_flight"
    }));
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_capacity_rejects_before_terminal_or_forward_effects() {
    // [BLIND-RELAY-NO-EVICTION-ADMISSION 2026-08-24 by Codex] Exercise the
    // real authenticated handler with a full replay map. The new route must
    // fail before terminal accounting, forwarding, or receipt creation.
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let now = now_secs();
    let request = PeerBlindRelayRequest {
        envelope: BlindRelayEnvelope {
            route_id: [0x4Au8; 16],
            next_hop: node_identity.public_key_bytes(),
            ttl: 2,
            encrypted_blob: b"opaque capacity-gated relay candidate".to_vec(),
            timestamp: now,
            signature: [0u8; 64],
        }
        .sign_with(&previous_hop),
        previous_hop_node_id: previous_hop.public_key_bytes(),
        onward_envelope: None,
        onward_descriptor_hint: None,
    };
    let request_commitment = blind_relay_authenticated_request_commitment(&request).unwrap();
    let replay_registry = BlindRelayReplayDomain::default();
    for sequence in 0..MAX_BLIND_RELAY_SEEN_ROUTES {
        let mut retained_route = [0x49u8; 16];
        retained_route[..8].copy_from_slice(&(sequence as u64).to_be_bytes());
        assert!(matches!(
            replay_registry.observe(retained_route, request_commitment, now),
            BlindRelayRouteReplayDecision::New { .. }
        ));
    }
    let peer_store = Arc::new(PeerStore::new());
    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&node_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(replay_registry),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let error = process_peer_blind_relay(state, request).await.unwrap_err();

    assert!(matches!(error, BlindRelayError::ReplayCapacity));
    assert_eq!(error.status_code(), StatusCode::SERVICE_UNAVAILABLE);
    let blind_stats = peer_store.status(now + 1).runtime.blind_relay;
    assert_eq!(blind_stats.terminal, 0);
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "replay_capacity"
    }));
}

#[tokio::test]
async fn blind_relay_forward_retries_transient_next_hop_failure() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                let attempt = attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                if attempt == 0 {
                    StatusCode::SERVICE_UNAVAILABLE.into_response()
                } else {
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
                    .into_response()
                }
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
    let next_hop_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

    let state = ChatPeerState {
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
        route_id: [0x42u8; 16],
        next_hop: next_hop_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque encrypted relay bytes".to_vec(),
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&previous_hop);

    let response = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope,
            previous_hop_node_id: previous_hop.public_key_bytes(),
            onward_envelope: None,
            onward_descriptor_hint: None,
        },
    )
    .await
    .unwrap();

    server.abort();

    assert!(response.accepted);
    assert!(response.forwarded);
    assert!(!response.terminal);
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 1);
    assert_eq!(blind_stats.rejected, 0);
    assert_eq!(blind_stats.retry_attempted, 1);
    assert_eq!(blind_stats.retry_succeeded, 1);
    assert_eq!(blind_stats.retry_exhausted, 0);
    assert!(peer_store
        .recent_audit_events()
        .iter()
        .any(|event| { event.action == "blind_relay_retry" && event.outcome == "accepted" }));
}

#[tokio::test]
async fn blind_relay_middle_hop_forwards_onward_envelope_without_payload_inspection() {
    let terminal_requests: Arc<Mutex<Vec<PeerBlindRelayRequest>>> =
        Arc::new(Mutex::new(Vec::new()));
    let terminal_requests_for_route = Arc::clone(&terminal_requests);
    let terminal_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(request): Json<PeerBlindRelayRequest>| {
            let terminal_requests_for_request = Arc::clone(&terminal_requests_for_route);
            async move {
                terminal_requests_for_request.lock().unwrap().push(request);
                Json(PeerBlindRelayResponse {
                    accepted: true,
                    terminal: true,
                    forwarded: false,
                    ttl_remaining: 0,
                    reason: Some("terminal_next_hop".to_string()),
                    delivery_receipt: None,
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: None,
                })
                .into_response()
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let terminal_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, terminal_app).await.unwrap();
    });

    let now = now_secs();
    let entry_identity = IdentityKeyPair::generate();
    let middle_identity = Arc::new(IdentityKeyPair::generate());
    let terminal_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(
                &terminal_identity,
                terminal_endpoint,
                now,
                now + 300,
            ),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&terminal_identity.public_key_bytes(), now);

    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity: Arc::clone(&middle_identity),
        http_client: Arc::new(reqwest::Client::new()),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let outer_envelope = BlindRelayEnvelope {
        route_id: [0x62u8; 16],
        next_hop: middle_identity.public_key_bytes(),
        ttl: 2,
        encrypted_blob: b"opaque middle-hop carrier; do not parse".to_vec(),
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&entry_identity);
    let onward_envelope = BlindRelayEnvelope {
        route_id: [0x63u8; 16],
        next_hop: terminal_identity.public_key_bytes(),
        ttl: 1,
        encrypted_blob: b"opaque terminal relay blob; do not parse".to_vec(),
        timestamp: now,
        signature: [0u8; 64],
    }
    .sign_with(&entry_identity);

    let response = process_peer_blind_relay(
        state,
        PeerBlindRelayRequest {
            envelope: outer_envelope,
            previous_hop_node_id: entry_identity.public_key_bytes(),
            onward_envelope: Some(onward_envelope),
            onward_descriptor_hint: None,
        },
    )
    .await
    .unwrap();

    server.abort();

    assert!(response.accepted);
    assert!(response.forwarded);
    assert!(!response.terminal);
    assert_eq!(response.reason.as_deref(), Some("onion_middle_forwarded"));

    let terminal_requests = terminal_requests.lock().unwrap();
    assert_eq!(terminal_requests.len(), 1);
    let terminal_request = &terminal_requests[0];
    assert_eq!(
        terminal_request.previous_hop_node_id,
        middle_identity.public_key_bytes()
    );
    assert_eq!(
        terminal_request.envelope.next_hop,
        terminal_identity.public_key_bytes()
    );
    assert_eq!(terminal_request.envelope.ttl, 0);
    assert!(terminal_request.onward_envelope.is_none());
    let middle_public = IdentityPublicKey::from_bytes(&middle_identity.public_key_bytes()).unwrap();
    assert!(terminal_request
        .envelope
        .verify_signature_from(&middle_public)
        .is_ok());
    assert_eq!(
        terminal_request.envelope.encrypted_blob,
        b"opaque terminal relay blob; do not parse".to_vec()
    );

    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 1);
    assert_eq!(blind_stats.rejected, 0);
}

#[tokio::test]
async fn blind_relay_forward_requires_accepted_next_hop_ack() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                Json(PeerBlindRelayResponse {
                    accepted: false,
                    terminal: false,
                    forwarded: false,
                    ttl_remaining: 1,
                    reason: Some("relay_unavailable".to_string()),
                    delivery_receipt: None,
                    success_receipt: None,
                    failure_receipt: None,
                    opaque_terminal_response_b64: None,
                })
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
    let next_hop_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

    let state = ChatPeerState {
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
        route_id: [0x56u8; 16],
        next_hop: next_hop_identity.public_key_bytes(),
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
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "forward_failed"
    }));
}

#[tokio::test]
async fn blind_relay_forward_rejects_malformed_success_ack() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                (StatusCode::OK, "not-a-peer-blind-relay-ack").into_response()
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
    let next_hop_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

    let state = ChatPeerState {
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
        route_id: [0x57u8; 16],
        next_hop: next_hop_identity.public_key_bytes(),
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
    assert_eq!(blind_stats.retry_attempted, 0);
    assert_eq!(blind_stats.retry_exhausted, 0);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "forward_failed"
            && !event.detail.contains("not-a-peer-blind-relay-ack")
    }));
}

#[tokio::test]
async fn blind_relay_requires_next_hop_chat_relay_capability() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                StatusCode::OK.into_response()
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
    let next_hop_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_peer_descriptor_for(
                &next_hop_identity,
                endpoint,
                now,
                now + 300,
                vec![NodeCapability::PrivacyRelay],
            ),
            now,
            "gossip_snapshot",
        )
        .unwrap();

    let state = ChatPeerState {
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
        route_id: [0x54u8; 16],
        next_hop: next_hop_identity.public_key_bytes(),
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

    assert!(matches!(result, Err(BlindRelayError::NoRoute)));
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 0);
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.no_route, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == "no_route"
    }));
}

#[tokio::test]
async fn blind_relay_requires_routeability_evidence_before_forwarding() {
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
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
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, next_hop_app).await.unwrap();
    });

    let now = now_secs();
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let next_hop_identity = IdentityKeyPair::generate();
    let next_hop_node_id = next_hop_identity.public_key_bytes();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
            now,
            "gossip_snapshot",
        )
        .unwrap();

    let state = ChatPeerState {
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

    assert!(matches!(result, Err(BlindRelayError::NoRoute)));
    assert_eq!(attempts.load(AtomicOrdering::SeqCst), 0);
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.no_route, 1);
    let route_status = peer_store.route_candidate_status(now + 5);
    let route_row = route_status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == hex::encode(&next_hop_node_id[..4]))
        .expect("chat relay row should remain visible");
    // [ROUTE-HEALTH-REMOTE-POISONING 2026-08-11 by Codex] This request was
    // rejected before any outbound attempt. It must not let an untrusted
    // previous hop manufacture failure or quarantine evidence for a peer.
    assert_eq!(route_row.routeability_state, "unknown");
    assert!(!route_row.routeability_ready);
    assert_eq!(route_row.route_failure_count, 0);
    assert_eq!(route_row.route_consecutive_failures, 0);
    assert!(route_row.last_route_failure_reason.is_none());
    assert!(!route_row.route_quarantined);
    assert!(peer_store.recent_audit_events().iter().all(|event| {
        event.action != "blind_relay_route_health"
            || !event.detail.contains("reason=routeability_not_ready")
    }));
}

#[tokio::test]
async fn blind_relay_forward_reports_retry_exhaustion_without_payload_data() {
    let (relay, path) = temp_chat_relay("blind-relay-retry-exhaustion");
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts_for_route = Arc::clone(&attempts);
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(move |Json(_request): Json<PeerBlindRelayRequest>| {
            let attempts_for_request = Arc::clone(&attempts_for_route);
            async move {
                attempts_for_request.fetch_add(1, AtomicOrdering::SeqCst);
                StatusCode::SERVICE_UNAVAILABLE.into_response()
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
    let next_hop_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(&next_hop_identity, endpoint, now, now + 300),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

    let state = ChatPeerState {
        chat_relay: Some(relay),
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
        route_id: [0x43u8; 16],
        next_hop: next_hop_identity.public_key_bytes(),
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
    assert_eq!(
        attempts.load(AtomicOrdering::SeqCst),
        MAX_BLIND_RELAY_FORWARD_ATTEMPTS
    );
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.forward_failed, 1);
    assert_eq!(
        blind_stats.retry_attempted,
        (MAX_BLIND_RELAY_FORWARD_ATTEMPTS - 1) as u64
    );
    assert_eq!(blind_stats.retry_succeeded, 0);
    assert_eq!(blind_stats.retry_exhausted, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_retry"
            && event.outcome == "rejected"
            && !event.detail.contains("opaque encrypted relay bytes")
    }));
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn blind_relay_forward_retries_timeout_without_endpoint_leak() {
    let next_hop_app = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(
            move |Json(_request): Json<PeerBlindRelayRequest>| async move {
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
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
            },
        ),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, next_hop_app).await.unwrap();
    });

    let now = now_secs();
    let previous_hop = IdentityKeyPair::generate();
    let node_identity = Arc::new(IdentityKeyPair::generate());
    let next_hop_identity = IdentityKeyPair::generate();
    let peer_store = Arc::new(PeerStore::new());
    peer_store
        .upsert_verified_from_source(
            signed_chat_relay_peer_descriptor_for(
                &next_hop_identity,
                endpoint.clone(),
                now,
                now + 300,
            ),
            now,
            "gossip_snapshot",
        )
        .unwrap();
    peer_store.record_route_forward_success(&next_hop_identity.public_key_bytes(), now);

    let state = ChatPeerState {
        chat_relay: None,
        blind_vault: None,
        anonymous_mailbox: None,
        sessions: Arc::new(SessionManager::new(16, std::time::Duration::from_secs(60))),
        udp: Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap()),
        peer_store: Arc::clone(&peer_store),
        node_identity,
        http_client: Arc::new(
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(30))
                .build()
                .unwrap(),
        ),
        blind_relay_in_flight: Arc::new(AtomicUsize::new(0)),
        blind_relay_replay_registry: Arc::new(BlindRelayReplayDomain::default()),
        blind_relay_abuse_guard: Arc::new(BlindRelayAbuseDomain::default()),
    };
    let envelope = BlindRelayEnvelope {
        route_id: [0x58u8; 16],
        next_hop: next_hop_identity.public_key_bytes(),
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
    let blind_stats = peer_store.status(now + 5).runtime.blind_relay;
    assert_eq!(blind_stats.forwarded, 0);
    assert_eq!(blind_stats.rejected, 1);
    assert_eq!(blind_stats.forward_failed, 1);
    assert_eq!(
        blind_stats.retry_attempted,
        (MAX_BLIND_RELAY_FORWARD_ATTEMPTS - 1) as u64
    );
    assert_eq!(blind_stats.retry_succeeded, 0);
    assert_eq!(blind_stats.retry_exhausted, 1);
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_retry"
            && event.outcome == "scheduled"
            && event
                .detail
                .contains("reason_bucket=blind_relay_request_timeout")
            && !event.detail.contains(&endpoint)
    }));
    assert!(peer_store.recent_audit_events().iter().any(|event| {
        event.action == "blind_relay_retry"
            && event.outcome == "rejected"
            && event
                .detail
                .contains("reason_bucket=blind_relay_request_timeout")
            && !event.detail.contains(&endpoint)
            && !event.detail.contains("opaque encrypted relay bytes")
    }));
}

#[test]
fn blind_relay_crypto_capacity_has_a_floor_on_small_hosts() {
    // [NODE-CAPACITY 2026-10-09 by Claude] One or two vCPUs used to yield a
    // single ingress permit, so concurrent onion requests shed Backpressure.
    let table = [
        (0usize, 4usize),
        (1, 4),
        (2, 4),
        (4, 4),
        (8, 4),
        (9, 5),
        (14, 7),
        (16, 8),
        (64, 8),
        (usize::MAX, 8),
    ];
    for (threads, expected) in table {
        assert_eq!(
            blind_relay_capacity_for_threads(threads),
            expected,
            "capacity for {threads} hardware threads"
        );
    }
}

#[test]
fn blind_relay_ingress_keeps_a_reserved_progress_permit_at_every_size() {
    for threads in [1usize, 2, 3, 4, 8, 16, 64] {
        let total = blind_relay_capacity_for_threads(threads);
        let ingress = blind_relay_ingress_crypto_capacity(total);
        assert!(ingress >= 3, "{threads} threads: ingress {ingress}");
        assert!(
            ingress < total,
            "{threads} threads: ingress {ingress} must leave one of {total} for outbound/completion"
        );
    }
}

#[test]
fn a_one_core_host_admits_concurrent_ingress_instead_of_shedding_the_second() {
    // [NODE-CAPACITY 2026-10-09 by Claude] The failure this fixes: with a
    // single hardware thread the ingress semaphore had one permit, so a second
    // request arriving while the first was still in its crypto step got
    // `Backpressure` from `try_acquire`.
    let total = blind_relay_capacity_for_threads(1);
    let ingress = Arc::new(Semaphore::new(blind_relay_ingress_crypto_capacity(total)));
    let held: Vec<_> = (0..3)
        .map(|n| {
            Arc::clone(&ingress)
                .try_acquire_owned()
                .unwrap_or_else(|_| panic!("ingress request {n} was shed on a one-core host"))
        })
        .collect();
    assert!(
        Arc::clone(&ingress).try_acquire_owned().is_err(),
        "admission must still be bounded"
    );
    drop(held);
}
