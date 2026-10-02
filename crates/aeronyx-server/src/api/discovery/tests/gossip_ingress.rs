// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn open_node_admission_is_allowlist_independent_private_and_non_routeable() {
    let identity = IdentityKeyPair::generate();
    let descriptor = open_node_descriptor(&identity, 1, "https://8.8.8.8:8422");
    let descriptor_bytes = descriptor.encode_canonical().unwrap();
    let node_hex = hex::encode(descriptor.node_id());
    let store = Arc::new(PeerStore::new());
    let mut policy = DiscoveryApiPolicy::default();
    policy.allowed_peer_ids.insert("11".repeat(32));

    let response = post_open_node(
        build_discovery_router(Arc::clone(&store), policy),
        descriptor_bytes,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 4_096)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["accepted"], true);
    assert_eq!(body["status"], "candidate_admitted");
    assert_eq!(body["route_authority"], false);
    assert_eq!(
        body["economic_admission"],
        "reserved_future_eth_projection_not_enforced"
    );
    let rendered = String::from_utf8(bytes.to_vec()).unwrap();
    for forbidden in [
        node_hex.as_str(),
        "8.8.8.8",
        "8422",
        "signature",
        "descriptor",
    ] {
        assert!(!rendered.contains(forbidden), "response leaked {forbidden}");
    }
    assert_eq!(store.len(), 0, "Stage-A admission must not grant routing");
    assert_eq!(store.status(now_secs()).runtime.candidate_admitted, 1);
}

#[tokio::test]
async fn open_node_admission_is_canonical_replay_and_policy_safe() {
    let identity = IdentityKeyPair::generate();
    let descriptor = open_node_descriptor(&identity, 7, "https://8.8.8.8:8422");
    let canonical = descriptor.encode_canonical().unwrap();
    let store = Arc::new(PeerStore::new());
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

    assert_eq!(
        post_open_node(app.clone(), canonical.clone())
            .await
            .status(),
        StatusCode::OK
    );
    let exact = post_open_node(app.clone(), canonical.clone()).await;
    assert_eq!(exact.status(), StatusCode::OK);
    let exact_body = axum::body::to_bytes(exact.into_body(), 4_096)
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&exact_body).unwrap()["status"],
        "exact_replay"
    );

    let mut conflict_body = descriptor.descriptor.clone();
    conflict_body.public_endpoint = Some("https://9.9.9.9:8422".to_string());
    let conflict = SignedNodeDescriptor::sign(conflict_body, &identity)
        .unwrap()
        .encode_canonical()
        .unwrap();
    assert_eq!(
        post_open_node(app.clone(), conflict).await.status(),
        StatusCode::CONFLICT
    );

    let mut trailing = canonical;
    trailing.push(0);
    assert_eq!(
        post_open_node(app.clone(), trailing).await.status(),
        StatusCode::BAD_REQUEST
    );

    let private = open_node_descriptor(&identity, 8, "http://127.0.0.1:8422")
        .encode_canonical()
        .unwrap();
    assert_eq!(
        post_open_node(app, private).await.status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(store.len(), 0);
}

#[tokio::test]
async fn open_node_admission_honors_denylist_and_body_ceiling() {
    let identity = IdentityKeyPair::generate();
    let descriptor = open_node_descriptor(&identity, 1, "https://8.8.8.8:8422");
    let mut policy = DiscoveryApiPolicy::default();
    policy
        .denied_peer_ids
        .insert(hex::encode(descriptor.node_id()));
    let app = build_discovery_router(Arc::new(PeerStore::new()), policy);
    assert_eq!(
        post_open_node(app, descriptor.encode_canonical().unwrap())
            .await
            .status(),
        StatusCode::FORBIDDEN
    );

    let oversized = vec![0u8; MAX_SIGNED_NODE_DESCRIPTOR_BYTES + 1];
    let app = build_discovery_router(Arc::new(PeerStore::new()), DiscoveryApiPolicy::default());
    assert_eq!(
        post_open_node(app, oversized).await.status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

#[tokio::test]
async fn test_public_projection_snapshot_forces_public_only_for_default_and_false_query() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let relay_capabilities = [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];

    let private_keypair = IdentityKeyPair::from_bytes(&[211; 32]).unwrap();
    let mut private_descriptor = signed_candidate_exclusion_descriptor(
        211,
        now,
        Some("private-snapshot.invalid:443"),
        &relay_capabilities,
        true,
    );
    private_descriptor.descriptor.policy.public_discovery = false;
    let private_descriptor =
        SignedNodeDescriptor::sign(private_descriptor.descriptor, &private_keypair).unwrap();
    let private_node_id = private_descriptor.node_id();

    let public_descriptor = signed_candidate_exclusion_descriptor(
        212,
        now,
        Some("public-snapshot.invalid:443"),
        &relay_capabilities,
        true,
    );
    let public_node_id = public_descriptor.node_id();
    store.upsert_verified(private_descriptor, now).unwrap();
    store.upsert_verified(public_descriptor, now).unwrap();

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    for uri in [
        "/api/discovery/snapshot?limit=10",
        "/api/discovery/snapshot?limit=10&public_only=false",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let snapshot: NodeBootstrapSnapshot = serde_json::from_slice(&body).unwrap();
        assert_eq!(snapshot.peers.len(), 1);
        assert_eq!(snapshot.peers[0].node_id(), public_node_id);
        assert!(snapshot.peers[0].descriptor.policy.public_discovery);
        assert!(snapshot
            .peers
            .iter()
            .all(|descriptor| descriptor.node_id() != private_node_id));
    }
}

#[tokio::test]
async fn test_snapshot_endpoint_returns_snapshot() {
    let store = Arc::new(PeerStore::new());
    store
        .upsert_verified(signed_descriptor(), now_secs())
        .unwrap();
    let app = build_discovery_router(store, DiscoveryApiPolicy::default());

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/snapshot?limit=10")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_gossip_snapshot_request_returns_response() {
    let store = Arc::new(PeerStore::new());
    store
        .upsert_verified(signed_descriptor(), now_secs())
        .unwrap();
    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
        requested_at: now_secs(),
        limit: Some(1),
    })
    .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn endpoint_attestation_gossip_verifies_discards_and_replays_without_state() {
    // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex] A valid dormant
    // carrier is transport evidence only: no peer, freshness, or audit state.
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let before = store.status(now + 2);
    let body = serde_json::to_vec(&endpoint_attestation_message(now)).unwrap();
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    assert_eq!(store.status(now + 2), before);
}

#[tokio::test]
async fn endpoint_attestation_gossip_persists_exact_replay_without_peer_state() {
    // [PERMISSIONLESS-ENDPOINT-ATTESTATION-INBOX-COMPOSITION 2026-09-24 by Codex]
    // The durable quarantine records transport evidence while every
    // PeerStore projection remains byte-for-byte unchanged.
    std::fs::create_dir_all("target/test-temp").expect("external test root");
    let directory = TempDir::new_in("target/test-temp").expect("tempdir");
    let inbox_config = DiscoveryEndpointAttestationInboxConfig {
        db_path: directory.path().join("attestations.sqlite3"),
        max_entries: 8,
        max_logical_bytes: 8 * 1024,
        retention_ttl_secs: 600,
        cleanup_batch_size: 8,
    };
    let inbox = Arc::new(
        SqliteDiscoveryEndpointAttestationInbox::open(inbox_config.clone()).expect("open"),
    );
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let before = store.status(now + 2);
    let body = serde_json::to_vec(&endpoint_attestation_message(now)).unwrap();
    let app = build_discovery_router_with_local_entry_and_attestation_inbox(
        Arc::clone(&store),
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::default(),
        None,
        [0x71; 32],
        Some(Arc::clone(&inbox)),
    );

    for _ in 0..2 {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    assert_eq!(store.status(now + 2), before);
    let snapshot = inbox.eligibility_snapshot_at(now + 2).expect("snapshot");
    assert_eq!(snapshot.retained_rows, 1);
    assert_eq!(snapshot.fresh_rows, 1);
    drop(app);
    drop(inbox);
    let reopened = SqliteDiscoveryEndpointAttestationInbox::open(inbox_config).expect("reopen");
    assert_eq!(
        reopened
            .eligibility_snapshot_at(now + 2)
            .expect("restart snapshot"),
        snapshot
    );
}

#[tokio::test]
async fn endpoint_attestation_gossip_maps_conflict_and_capacity_without_peer_state() {
    std::fs::create_dir_all("target/test-temp").expect("external test root");
    let directory = TempDir::new_in("target/test-temp").expect("tempdir");
    let inbox = Arc::new(
        SqliteDiscoveryEndpointAttestationInbox::open(DiscoveryEndpointAttestationInboxConfig {
            db_path: directory.path().join("attestations.sqlite3"),
            max_entries: 1,
            max_logical_bytes: 8 * 1024,
            retention_ttl_secs: 600,
            cleanup_batch_size: 8,
        })
        .expect("open"),
    );
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let before = store.status(now + 2);
    let app = build_discovery_router_with_local_entry_and_attestation_inbox(
        Arc::clone(&store),
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::default(),
        None,
        [0x71; 32],
        Some(inbox),
    );
    let send = |message: NodeDiscoveryMessage| {
        app.clone().oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&message).unwrap()))
                .unwrap(),
        )
    };

    assert_eq!(
        send(endpoint_attestation_message_with(now, 0x72, 0x73, 9))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        send(endpoint_attestation_message_with(now, 0x72, 0x74, 9))
            .await
            .unwrap()
            .status(),
        StatusCode::CONFLICT
    );
    assert_eq!(
        send(endpoint_attestation_message_with(now, 0x75, 0x76, 10))
            .await
            .unwrap()
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(store.status(now + 2), before);
}

#[tokio::test]
async fn endpoint_attestation_gossip_rejects_signature_tamper() {
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let before = store.status(now + 2);
    let mut message = endpoint_attestation_message(now);
    let NodeDiscoveryMessage::EndpointEvidenceAttestationV1 { attestation_frame } = &mut message
    else {
        unreachable!();
    };
    let last = attestation_frame.len() - 1;
    attestation_frame[last] ^= 1;
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&message).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(store.status(now + 2), before);
}

#[tokio::test]
async fn route_domain_certificate_ingress_is_verified_and_idempotent() {
    // [ROUTE-DOMAIN-CERTIFICATE-INGRESS 2026-08-03 by Codex] The HTTP
    // sender has no authority; only the configured certificate quorum can
    // move the process-local route gate.
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let subject = IdentityKeyPair::generate();
    let attestor_a = IdentityKeyPair::generate();
    let attestor_b = IdentityKeyPair::generate();
    let route_domain = [0x61; 16];
    store
        .configure_route_domain_attestor_policy(
            &[(subject.public_key_bytes(), route_domain)],
            &[attestor_a.public_key_bytes(), attestor_b.public_key_bytes()],
            2,
            true,
        )
        .unwrap();
    let certificate = route_domain_certificate_for(
        subject.public_key_bytes(),
        route_domain,
        now,
        &[&attestor_a, &attestor_b],
    );
    let body = encode_route_domain_attestation_certificate(&certificate).unwrap();
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/route-domain-certificate")
                .header("content-type", "application/octet-stream")
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let first_body = axum::body::to_bytes(first.into_body(), usize::MAX)
        .await
        .unwrap();
    let first_json: serde_json::Value = serde_json::from_slice(&first_body).unwrap();
    assert_eq!(first_json["accepted"], true);
    assert_eq!(first_json["stored"], true);
    assert_eq!(first_json["status"], "stored");

    let duplicate = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/route-domain-certificate")
                .header("content-type", "application/octet-stream")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(duplicate.status(), StatusCode::OK);
    let duplicate_body = axum::body::to_bytes(duplicate.into_body(), usize::MAX)
        .await
        .unwrap();
    let duplicate_json: serde_json::Value = serde_json::from_slice(&duplicate_body).unwrap();
    assert_eq!(duplicate_json["accepted"], true);
    assert_eq!(duplicate_json["stored"], false);
    assert_eq!(duplicate_json["status"], "already_present");
}

#[tokio::test]
async fn route_domain_certificate_ingress_rejects_untrusted_malformed_and_oversized_frames() {
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let subject = IdentityKeyPair::generate();
    let attestor_a = IdentityKeyPair::generate();
    let attestor_b = IdentityKeyPair::generate();
    let untrusted = IdentityKeyPair::generate();
    let route_domain = [0x62; 16];
    store
        .configure_route_domain_attestor_policy(
            &[(subject.public_key_bytes(), route_domain)],
            &[attestor_a.public_key_bytes(), attestor_b.public_key_bytes()],
            2,
            true,
        )
        .unwrap();
    let untrusted_certificate = route_domain_certificate_for(
        subject.public_key_bytes(),
        route_domain,
        now,
        &[&attestor_a, &untrusted],
    );
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

    let untrusted_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/route-domain-certificate")
                .header("content-type", "application/octet-stream")
                .body(Body::from(
                    encode_route_domain_attestation_certificate(&untrusted_certificate).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        untrusted_response.status(),
        StatusCode::UNPROCESSABLE_ENTITY
    );

    let malformed_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/route-domain-certificate")
                .header("content-type", "application/octet-stream")
                .body(Body::from(vec![0u8; 8]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(malformed_response.status(), StatusCode::BAD_REQUEST);

    let oversized_response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/route-domain-certificate")
                .header("content-type", "application/octet-stream")
                .body(Body::from(vec![
                    0u8;
                    MAX_ROUTE_DOMAIN_ATTESTATION_CERTIFICATE_FRAME_BYTES
                        + 1
                ]))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(oversized_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn route_domain_certificate_ingress_has_an_isolated_rate_budget() {
    let now = now_secs();
    let store = Arc::new(PeerStore::new());
    let subject = IdentityKeyPair::generate();
    let attestor = IdentityKeyPair::generate();
    let route_domain = [0x63; 16];
    store
        .configure_route_domain_attestor_policy(
            &[(subject.public_key_bytes(), route_domain)],
            &[attestor.public_key_bytes()],
            1,
            true,
        )
        .unwrap();
    let certificate =
        route_domain_certificate_for(subject.public_key_bytes(), route_domain, now, &[&attestor]);
    let body = encode_route_domain_attestation_certificate(&certificate).unwrap();
    let mut policy = DiscoveryApiPolicy::default();
    policy.gossip_rate_limit_per_minute = 1;
    let app = build_discovery_router(Arc::clone(&store), policy);

    for expected in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/route-domain-certificate")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
    }

    let gossip_body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
        requested_at: now,
        limit: Some(1),
    })
    .unwrap();
    let gossip_response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(gossip_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        gossip_response.status(),
        StatusCode::OK,
        "certificate abuse must not exhaust the gossip recovery budget"
    );
}

#[tokio::test]
async fn gossip_rejects_oversized_body_before_deserialization() {
    let store = Arc::new(PeerStore::new());
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(vec![b' '; DISCOVERY_REQUEST_BODY_MAX_BYTES + 1]))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(store.len(), 0, "oversized gossip must not reach PeerStore");
}

#[tokio::test]
async fn gossip_descriptor_announce_stays_non_routeable_candidate() {
    let store = Arc::new(PeerStore::new());
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
    let body = serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce {
        descriptor: signed_descriptor(),
    })
    .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(store.len(), 0);
    assert_eq!(store.status(now_secs()).runtime.candidate_admitted, 1);
}

#[tokio::test]
async fn gossip_descriptor_announce_refreshes_only_a_locally_established_identity() {
    let now = now_secs();
    let identity = IdentityKeyPair::generate();
    let expired = SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1_200),
            now.saturating_sub(600),
            "locally-cached-identity",
        ),
        &identity,
    )
    .unwrap();
    let store = Arc::new(PeerStore::new());
    let restored = store.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now, vec![expired]),
        now,
        "test_local_cache",
    );
    assert_eq!(restored.inserted, 1);

    let refreshed = SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            identity.public_key_bytes(),
            2,
            now.saturating_sub(1),
            now.saturating_add(300),
            "locally-cached-identity-refresh",
        ),
        &identity,
    )
    .unwrap();
    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce {
                        descriptor: refreshed,
                    })
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let current = store
        .get_valid(&identity.public_key_bytes(), now)
        .expect("only the existing locally established identity may refresh");
    assert_eq!(current.sequence(), 2);
    assert_eq!(store.status(now).runtime.candidate_admitted, 0);
}

#[tokio::test]
async fn directory_gossip_imports_only_against_local_replica_anchor() {
    let now = now_secs();
    let (replica_store, message, descriptor) = directory_gossip_fixture(now);
    let store = Arc::new(PeerStore::new());
    let app = build_discovery_router_with_local_status_and_directory_admission(
        Arc::clone(&store),
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::default(),
        Some(replica_store),
    );
    let body = serde_json::to_vec(&message).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
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
    let parsed: GossipResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed.applied.inserted, 1);
    assert_eq!(
        store.get_valid(&descriptor.node_id(), now),
        Some(descriptor)
    );
}

#[tokio::test]
async fn directory_gossip_fails_closed_without_local_replica_or_anchor() {
    let now = now_secs();
    let (_, message, _) = directory_gossip_fixture(now);
    let store_without_replica = Arc::new(PeerStore::new());
    let app_without_replica = build_discovery_router(
        Arc::clone(&store_without_replica),
        DiscoveryApiPolicy::default(),
    );
    let body = serde_json::to_vec(&message).unwrap();
    let response = app_without_replica
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(store_without_replica.len(), 0);

    let empty_local = IdentityKeyPair::from_bytes(&[0x95; 32]).unwrap();
    let (empty_replica, _) =
        DirectoryReplicaStore::open(":memory:", empty_local.public_key_bytes(), now).unwrap();
    let store_without_anchor = Arc::new(PeerStore::new());
    let app_without_anchor = build_discovery_router_with_local_status_and_directory_admission(
        Arc::clone(&store_without_anchor),
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::default(),
        Some(Arc::new(empty_replica)),
    );
    let response = app_without_anchor
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&message).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(store_without_anchor.len(), 0);
    assert_eq!(
        store_without_anchor.status(now).runtime.rejected,
        1,
        "proof rejection must remain visible as an aggregate only"
    );
}

#[tokio::test]
async fn test_snapshot_endpoint_caps_requested_limit() {
    let store = Arc::new(PeerStore::new());
    store
        .upsert_verified(signed_descriptor(), now_secs())
        .unwrap();
    store
        .upsert_verified(signed_descriptor(), now_secs())
        .unwrap();
    let mut policy = DiscoveryApiPolicy::default();
    policy.max_snapshot_limit = 1;
    let app = build_discovery_router(store, policy);

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/snapshot?limit=50")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let snapshot: NodeBootstrapSnapshot = serde_json::from_slice(&body).unwrap();
    assert_eq!(snapshot.peers.len(), 1);
}

#[tokio::test]
async fn test_gossip_denies_blocked_descriptor() {
    let store = Arc::new(PeerStore::new());
    let descriptor = signed_descriptor();
    let mut policy = DiscoveryApiPolicy::default();
    policy
        .denied_peer_ids
        .insert(hex::encode(descriptor.node_id()));
    let app = build_discovery_router(Arc::clone(&store), policy);
    let body =
        serde_json::to_vec(&NodeDiscoveryMessage::DescriptorAnnounce { descriptor }).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/api/discovery/gossip")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(store.len(), 0);
    let status = store.status(now_secs());
    assert_eq!(status.runtime.policy_rejected, 1);
    assert_eq!(
        status
            .recent_audit_events
            .last()
            .map(|event| event.action.as_str()),
        Some("gossip_policy_rejected")
    );
}

#[tokio::test]
async fn gossip_rate_limiter_recovers_after_lock_owner_panic() {
    // [DISCOVERY-RATE-LIMIT-RECOVERY 2026-07-30 by Codex] The historical
    // std::sync::Mutex became permanently poisoned here, causing every
    // later gossip request to panic at lock().expect().
    let rate_limit = Arc::new(Mutex::new(RateLimitState::new()));
    let panic_lock = Arc::clone(&rate_limit);
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _guard = panic_lock.lock();
        panic!("test-only-discovery-rate-limit-panic");
    }));
    assert!(panic_result.is_err());

    let state = DiscoveryApiState {
        peer_store: Arc::new(PeerStore::new()),
        local_node_id: None,
        directory_replica_store: None,
        endpoint_attestation_inbox: None,
        policy: DiscoveryApiPolicy::default(),
        local_capabilities: DiscoveryLocalCapabilityStatus::default(),
        rate_limit,
        node_admission_rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
        route_domain_certificate_rate_limit: Arc::new(Mutex::new(RateLimitState::new())),
    };
    let response = gossip_handler(
        State(state),
        Json(NodeDiscoveryMessage::SnapshotRequest {
            requested_at: now_secs(),
            limit: Some(1),
        }),
    )
    .await
    .into_response();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_gossip_rate_limit_rejects_excess_requests() {
    let store = Arc::new(PeerStore::new());
    let mut policy = DiscoveryApiPolicy::default();
    policy.gossip_rate_limit_per_minute = 1;
    let app = build_discovery_router(Arc::clone(&store), policy);

    for expected_status in [StatusCode::OK, StatusCode::TOO_MANY_REQUESTS] {
        let body = serde_json::to_vec(&NodeDiscoveryMessage::SnapshotRequest {
            requested_at: now_secs(),
            limit: Some(1),
        })
        .unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/api/discovery/gossip")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected_status);
    }

    let status = store.status(now_secs());
    assert_eq!(status.runtime.rate_limited, 1);
    assert_eq!(
        status
            .recent_audit_events
            .last()
            .map(|event| event.action.as_str()),
        Some("gossip_rate_limited")
    );
}
