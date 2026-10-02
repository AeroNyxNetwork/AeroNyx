// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn coordinator_lease_endpoint_grants_renews_and_rejects_competing_instance() {
    let now = now_secs();
    let witness = Arc::new(IdentityKeyPair::generate());
    let coordinator = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.audit_record_commitment_chain().await.unwrap();
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &coordinator, None, now);
    let router = build_memchain_peer_router_with_coordinator_lease(
        Arc::clone(&storage),
        peer_store,
        Arc::clone(&witness),
        Some(coordinator.public_key_bytes()),
    );
    let first_instance = [0x71; 32];
    let first_request_id = [0x72; 16];
    let first_frame = coordinator_lease_request_frame(
        &coordinator,
        first_instance,
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        first_request_id,
        now,
    );
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(first_frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let grant = verify_record_commitment_coordinator_lease_response(
        &body,
        &first_request_id,
        &coordinator.public_key_bytes(),
        &first_instance,
        &witness.public_key_bytes(),
        (0, GENESIS_PREV_HASH),
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        now,
    )
    .unwrap();
    assert_eq!(grant.lease_epoch, 1);

    let renewal = coordinator_lease_request_frame(
        &coordinator,
        first_instance,
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x73; 16],
        now,
    );
    let renewed = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(renewal))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(renewed.status(), StatusCode::OK);

    let competing = coordinator_lease_request_frame(
        &coordinator,
        [0x74; 32],
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x75; 16],
        now,
    );
    let rejected = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(competing))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn coordinator_lease_endpoint_releases_exact_holder_and_hands_over_immediately() {
    let now = now_secs();
    let witness = Arc::new(IdentityKeyPair::generate());
    let coordinator = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.audit_record_commitment_chain().await.unwrap();
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &coordinator, None, now);
    let router = build_memchain_peer_router_with_coordinator_lease(
        storage,
        peer_store,
        Arc::clone(&witness),
        Some(coordinator.public_key_bytes()),
    );
    let first_instance = [0x76; 32];
    let second_instance = [0x77; 32];
    let acquire = coordinator_lease_request_frame(
        &coordinator,
        first_instance,
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x78; 16],
        now,
    );
    let acquired = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(acquire))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(acquired.status(), StatusCode::OK);

    let wrong_release =
        coordinator_lease_release_request_frame(&coordinator, second_instance, [0x79; 16], now);
    let wrong_release_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease/release")
                .body(Body::from(wrong_release))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wrong_release_response.status(), StatusCode::CONFLICT);

    let release_request_id = [0x7A; 16];
    let release_frame = coordinator_lease_release_request_frame(
        &coordinator,
        first_instance,
        release_request_id,
        now,
    );
    let released = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease/release")
                .body(Body::from(release_frame.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(released.status(), StatusCode::OK);
    let body = axum::body::to_bytes(released.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let release_ack = verify_record_commitment_coordinator_lease_release_response(
        &body,
        &release_request_id,
        &coordinator.public_key_bytes(),
        &first_instance,
        &witness.public_key_bytes(),
        now,
    )
    .unwrap();
    assert_eq!(release_ack.lease_epoch, 1);

    let replay = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease/release")
                .body(Body::from(release_frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(replay.status(), StatusCode::TOO_MANY_REQUESTS);

    let delayed_renewal = coordinator_lease_request_frame(
        &coordinator,
        first_instance,
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x7B; 16],
        now,
    );
    let delayed = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(delayed_renewal))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delayed.status(), StatusCode::CONFLICT);

    let takeover = coordinator_lease_request_frame(
        &coordinator,
        second_instance,
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x7C; 16],
        now,
    );
    let takeover_response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(takeover))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(takeover_response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(takeover_response.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let grant = verify_record_commitment_coordinator_lease_response(
        &body,
        &[0x7C; 16],
        &coordinator.public_key_bytes(),
        &second_instance,
        &witness.public_key_bytes(),
        (0, GENESIS_PREV_HASH),
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        now,
    )
    .unwrap();
    assert_eq!(grant.lease_epoch, 2);
}

#[test]
fn coordinator_lease_response_accepts_processing_time_remainder() {
    let coordinator = IdentityKeyPair::generate();
    let witness = IdentityKeyPair::generate();
    let chain_id = AERONYX_MEMCHAIN_MAINNET_CHAIN_ID;
    let request_id = [0x88; 16];
    let instance_id = [0x89; 32];
    let response_timestamp = 10_001;
    let lease_expires_at = 10_060;
    let signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator.public_key_bytes(),
        &instance_id,
        &witness.public_key_bytes(),
        response_timestamp,
        1,
        lease_expires_at,
        0,
        &GENESIS_PREV_HASH,
    );
    let frame = encode_memchain(&MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator: coordinator.public_key_bytes(),
        instance_id,
        witness: witness.public_key_bytes(),
        response_timestamp,
        lease_epoch: 1,
        lease_expires_at,
        witness_tip_height: 0,
        witness_tip_hash: GENESIS_PREV_HASH,
        signature: witness.sign(&signing_bytes),
    })
    .unwrap();

    let grant = verify_record_commitment_coordinator_lease_response(
        &frame,
        &request_id,
        &coordinator.public_key_bytes(),
        &instance_id,
        &witness.public_key_bytes(),
        (0, GENESIS_PREV_HASH),
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        response_timestamp,
    )
    .unwrap();
    assert_eq!(grant.valid_for_secs, 59);
}

#[test]
fn response_verification_rejects_coordinator_rollback_and_fork() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let responder = coordinator.public_key_bytes();
    let request_id = [0x61; 16];
    let local_tip = (1, [0x71; 32]);

    let rollback_signing = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        now,
        &[],
        false,
        0,
        &GENESIS_PREV_HASH,
    );
    let rollback_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp: now,
        blocks: Vec::new(),
        has_more: false,
        tip_height: 0,
        tip_hash: GENESIS_PREV_HASH,
        signature: coordinator.sign(&rollback_signing),
    })
    .unwrap();
    assert_eq!(
        verify_record_commitment_page(
            &rollback_frame,
            &request_id,
            &responder,
            &responder,
            local_tip,
            now,
        )
        .unwrap_err(),
        "coordinator_rollback_detected"
    );

    let forked =
        RecordCommitmentBlockV1::new_signed(2, now, [0x72; 32], vec![[0x73; 32]], &coordinator);
    let fork_blocks = vec![forked.clone()];
    let fork_signing = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        now,
        &fork_blocks,
        false,
        2,
        &forked.hash(),
    );
    let fork_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp: now,
        blocks: fork_blocks,
        has_more: false,
        tip_height: 2,
        tip_hash: forked.hash(),
        signature: coordinator.sign(&fork_signing),
    })
    .unwrap();
    assert_eq!(
        verify_record_commitment_page(
            &fork_frame,
            &request_id,
            &responder,
            &responder,
            local_tip,
            now,
        )
        .unwrap_err(),
        "commitment_chain_verification_failed"
    );
}

#[tokio::test]
async fn live_http_follower_pull_converges_with_pinned_coordinator() {
    let now = now_secs();
    let responder_identity = Arc::new(IdentityKeyPair::generate());
    let requester_identity = IdentityKeyPair::generate();
    let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

    let first = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0x51; 32], [0x52; 32]],
        &responder_identity,
    );
    source
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    let second = RecordCommitmentBlockV1::new_signed(
        2,
        now.saturating_sub(1),
        first.hash(),
        vec![[0x53; 32]],
        &responder_identity,
    );
    source
        .append_record_commitment_block(&second, None)
        .await
        .unwrap();
    source.audit_record_commitment_chain().await.unwrap();
    destination.audit_record_commitment_chain().await.unwrap();

    let source_peers = Arc::new(PeerStore::new());
    admit_peer(&source_peers, &requester_identity, None, now);
    let router = build_memchain_peer_router(
        Arc::clone(&source),
        source_peers,
        Arc::clone(&responder_identity),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let follower_peers = PeerStore::new();
    admit_peer(
        &follower_peers,
        &responder_identity,
        Some(format!("http://{address}")),
        now,
    );
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let before_pull = pull_record_commitment_checkpoint_with_endpoint_policy(
        &destination,
        &follower_peers,
        &requester_identity,
        &responder_identity.public_key_bytes(),
        &client,
        false,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(
        before_pull.relation,
        CommitmentCheckpointRelation::RemoteAhead
    );
    assert_eq!(before_pull.local_tip_height, 0);
    assert_eq!(before_pull.remote_tip_height, 2);
    let outcome = pull_record_commitment_page_with_endpoint_policy(
        &destination,
        &follower_peers,
        &requester_identity,
        &responder_identity.public_key_bytes(),
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();

    assert_eq!(outcome.inserted, 2);
    assert_eq!(outcome.already_present, 0);
    assert!(!outcome.has_more);
    assert_eq!(outcome.remote_tip_height, 2);
    assert_eq!(
        destination.record_commitment_chain_tip().await,
        source.record_commitment_chain_tip().await
    );
    let checkpoint = pull_record_commitment_checkpoint_with_endpoint_policy(
        &destination,
        &follower_peers,
        &requester_identity,
        &responder_identity.public_key_bytes(),
        &client,
        false,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(checkpoint.relation, CommitmentCheckpointRelation::Converged);
    assert_eq!(checkpoint.local_tip_height, 2);
    assert_eq!(checkpoint.remote_tip_height, 2);
    assert_eq!(checkpoint.checkpoint_height, 2);
    assert_ne!(checkpoint.evidence_digest, [0u8; 32]);
    let served = source.record_commitment_checkpoint_status();
    assert_eq!(served.requests_served_total, 2);
    assert!(served.last_served_at.is_some());
    assert_eq!(served.state, "not_checked");
    assert_eq!(served.last_checked_at, None);
    assert_eq!(served.last_divergence_at, None);
    assert_eq!(served.proofs_verified_total, 0);
    assert_eq!(served.divergences_total, 0);
    server.abort();
    let _ = server.await;
}
