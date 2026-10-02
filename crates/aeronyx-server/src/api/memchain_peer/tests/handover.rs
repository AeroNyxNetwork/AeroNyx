// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn handover_response_rejects_omission_and_wrong_predecessor() {
    // [AUTHORITY-HANDOVER-ADVERSARIAL 2026-08-14 by Codex] A responder
    // cannot advertise a newer history head while withholding the exact
    // next proof, nor wrap somebody else's valid transition in its own
    // authenticated transport response.
    let now = 50_000;
    let request_id = [0x41; 16];
    let active = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let carrier = IdentityKeyPair::generate();
    let omitted = signed_handover_response_frame(&active, request_id, now, None, 1);
    assert_eq!(
        control_plane::verify_record_coordinator_handover_response(
            &omitted,
            &request_id,
            &active.public_key_bytes(),
            &active.public_key_bytes(),
            0,
            1,
            now,
        )
        .unwrap_err(),
        "handover_proof_omitted"
    );

    let unrelated_previous = IdentityKeyPair::generate();
    let wrong_predecessor = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        [0x42; 32],
        [0x43; 16],
        now.saturating_sub(1),
        &unrelated_previous,
        &next,
    );
    let wrapped =
        signed_handover_response_frame(&carrier, request_id, now, Some(wrong_predecessor), 1);
    assert_eq!(
        control_plane::verify_record_coordinator_handover_response(
            &wrapped,
            &request_id,
            &carrier.public_key_bytes(),
            &active.public_key_bytes(),
            0,
            1,
            now,
        )
        .unwrap_err(),
        "handover_previous_coordinator_mismatch"
    );

    let valid_transition = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        [0x44; 32],
        [0x45; 16],
        now.saturating_sub(1),
        &active,
        &next,
    );
    let carried =
        signed_handover_response_frame(&carrier, request_id, now, Some(valid_transition), 1);
    assert!(control_plane::verify_record_coordinator_handover_response(
        &carried,
        &request_id,
        &carrier.public_key_bytes(),
        &active.public_key_bytes(),
        0,
        1,
        now,
    )
    .is_ok());
}

#[test]
fn handover_carrier_retries_only_explicit_availability_failures() {
    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Alternate pins may
    // improve availability, never mask authenticated protocol failures.
    for error in [
        "active_coordinator_unavailable",
        "active_coordinator_missing_endpoint",
        "handover_carrier_unavailable",
        "handover_carrier_missing_endpoint",
        "handover_carrier_behind",
        "handover_request_timeout",
        "handover_http_status_503",
    ] {
        assert_eq!(
            coordinator_handover_source_failure_class(error),
            CommitmentAuthoritySourceFailureClass::Availability,
            "{error}"
        );
    }
    for error in [
        "active_coordinator_unsafe_endpoint",
        "handover_carrier_unsafe_endpoint",
        "handover_http_status_401",
        "invalid_handover_response_signature",
        "handover_previous_coordinator_mismatch",
        "handover_local_authority_changed",
        "storage_append_rejected",
    ] {
        assert_eq!(
            coordinator_handover_source_failure_class(error),
            CommitmentAuthoritySourceFailureClass::Security,
            "{error}"
        );
    }
}

#[tokio::test]
async fn handover_endpoint_authenticates_before_peer_membership_admission() {
    // [AUTHORITY-HANDOVER-ADMISSION 2026-08-14 by Codex] A forged request
    // must not reveal whether its claimed requester is in PeerStore. A
    // genuinely signed but unadmitted requester remains forbidden.
    let now = now_secs();
    let responder = Arc::new(IdentityKeyPair::generate());
    let known = IdentityKeyPair::generate();
    let unknown = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &known, None, now);
    let router = build_memchain_peer_router(storage, peer_store, responder);

    for (requester, request_id) in [
        (known.public_key_bytes(), [0x44; 16]),
        (unknown.public_key_bytes(), [0x45; 16]),
    ] {
        let forged = encode_memchain(&MemChainMessage::RecordCoordinatorHandoverRequestV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            after_authority_epoch: 0,
            request_id,
            requester,
            request_timestamp: now,
            signature: [0u8; 64],
        })
        .unwrap();
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/memchain/peer/coordinator-handover")
                    .body(Body::from(forged))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    let request_id = [0x46; 16];
    let requester = unknown.public_key_bytes();
    let signing_bytes = record_coordinator_handover_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        0,
        &request_id,
        &requester,
        now,
    );
    let signed_unknown = encode_memchain(&MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        after_authority_epoch: 0,
        request_id,
        requester,
        request_timestamp: now,
        signature: unknown.sign(&signing_bytes),
    })
    .unwrap();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-handover")
                .body(Body::from(signed_unknown))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn cold_follower_interleaves_prefix_and_exact_handover() {
    // [AUTHORITY-HANDOVER-FOLLOWER 2026-08-14 by Codex] The follower first
    // learns the transition as a future boundary, pulls only block one,
    // then accepts the same dual-signed proof and switches authority for
    // height two. No responder assertion alone can rotate authority.
    let now = now_secs();
    let coordinator = Arc::new(IdentityKeyPair::generate());
    let next = IdentityKeyPair::generate();
    let follower = IdentityKeyPair::generate();
    let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    source
        .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
        .unwrap();
    source.audit_record_commitment_chain().await.unwrap();
    let first_block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0x51; 32]],
        coordinator.as_ref(),
    );
    source
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let proof = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x52; 16],
        now.saturating_sub(1),
        coordinator.as_ref(),
        &next,
    );
    source
        .persist_configured_record_coordinator_handover(&proof, now)
        .await
        .unwrap();
    let second_block = RecordCommitmentBlockV1::new_signed(
        2,
        now.saturating_sub(1),
        first_block.hash(),
        vec![[0x53; 32]],
        &next,
    );
    source
        .append_record_commitment_block(&second_block, None)
        .await
        .unwrap();

    let source_peers = Arc::new(PeerStore::new());
    admit_peer(&source_peers, &follower, None, now);
    let router =
        build_memchain_peer_router(Arc::clone(&source), source_peers, Arc::clone(&coordinator));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let destination = MemoryStorage::open(":memory:", None).unwrap();
    destination
        .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
        .unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination.configure_record_commitment_sync(false, true);
    let destination_peers = PeerStore::new();
    admit_peer(
        &destination_peers,
        coordinator.as_ref(),
        Some(format!("http://{address}")),
        now,
    );
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    let pending = sync_next_record_coordinator_handover_with_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(pending.authority_epoch, 0);
    assert_eq!(pending.active_coordinator, coordinator.public_key_bytes());
    assert_eq!(pending.next_block_height, 1);
    assert_eq!(pending.pending_activation_height, Some(2));
    assert!(!pending.handover_inserted);
    assert_eq!(pending.source, CommitmentAuthoritySyncSource::Coordinator);
    assert_eq!(pending.carrier_attempts, 0);

    let page = pull_record_commitment_page_from_source_with_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &coordinator.public_key_bytes(),
        &coordinator.public_key_bytes(),
        &client,
        &allow_test_endpoint,
        1,
    )
    .await
    .unwrap();
    assert_eq!(page.inserted, 1);
    assert!(page.has_more);

    let activated = sync_next_record_coordinator_handover_with_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(activated.authority_epoch, 1);
    assert_eq!(activated.active_coordinator, next.public_key_bytes());
    assert_eq!(activated.next_block_height, 2);
    assert_eq!(activated.pending_activation_height, None);
    assert!(activated.handover_inserted);
    assert_eq!(activated.source, CommitmentAuthoritySyncSource::Coordinator);
    assert_eq!(activated.carrier_attempts, 0);

    server.abort();
}

#[tokio::test]
async fn pinned_carrier_recovers_exact_handover_when_coordinator_is_unavailable() {
    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] The carrier signs
    // only its response envelope. The accepted authority transition must
    // still be the root coordinator's exact-next dual-signed proof bound
    // to the follower's already-audited block-one prefix.
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let stale_carrier = Arc::new(IdentityKeyPair::generate());
    let carrier = Arc::new(IdentityKeyPair::generate());
    let follower = IdentityKeyPair::generate();
    let first_block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0x54; 32]],
        &coordinator,
    );
    let proof = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x55; 16],
        now.saturating_sub(1),
        &coordinator,
        &next,
    );

    let stale_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    stale_storage
        .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
        .unwrap();
    stale_storage.audit_record_commitment_chain().await.unwrap();
    stale_storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let stale_peers = Arc::new(PeerStore::new());
    admit_peer(&stale_peers, &follower, None, now);
    let stale_router = build_memchain_peer_router(
        Arc::clone(&stale_storage),
        stale_peers,
        Arc::clone(&stale_carrier),
    );
    let stale_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let stale_address = stale_listener.local_addr().unwrap();
    let stale_server = tokio::spawn(async move {
        axum::serve(stale_listener, stale_router).await.unwrap();
    });

    let carrier_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    carrier_storage
        .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
        .unwrap();
    carrier_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();
    carrier_storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    carrier_storage
        .persist_configured_record_coordinator_handover(&proof, now)
        .await
        .unwrap();
    let carrier_peers = Arc::new(PeerStore::new());
    admit_peer(&carrier_peers, &follower, None, now);
    let router = build_memchain_peer_router(
        Arc::clone(&carrier_storage),
        carrier_peers,
        Arc::clone(&carrier),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let destination = MemoryStorage::open(":memory:", None).unwrap();
    destination
        .configure_record_commitment_authority_root(Some(coordinator.public_key_bytes()))
        .unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    destination.configure_record_commitment_sync(false, true);
    let destination_peers = PeerStore::new();
    // The active coordinator is authenticated but has no reachable
    // endpoint, forcing only the narrow availability fallback.
    admit_peer(&destination_peers, &coordinator, None, now);
    admit_peer(
        &destination_peers,
        stale_carrier.as_ref(),
        Some(format!("http://{stale_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        carrier.as_ref(),
        Some(format!("http://{address}")),
        now,
    );
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let mut cursor = CommitmentAuthorityCarrierCursor::default();
    let mut circuit_breaker = CommitmentAuthorityCarrierCircuitBreaker::default();
    let recovered = sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &[stale_carrier.public_key_bytes(), carrier.public_key_bytes()],
        &client,
        &allow_test_endpoint,
        &mut cursor,
        &mut circuit_breaker,
    )
    .await
    .unwrap();

    assert_eq!(
        recovered.source,
        CommitmentAuthoritySyncSource::PinnedCarrier
    );
    assert_eq!(recovered.carrier_attempts, 2);
    assert!(recovered.handover_inserted);
    assert_eq!(recovered.authority_epoch, 1);
    assert_eq!(recovered.active_coordinator, next.public_key_bytes());
    assert_eq!(recovered.next_block_height, 2);
    assert_eq!(recovered.pending_activation_height, None);
    let status = destination.record_commitment_sync_status();
    assert_eq!(status.authority_sync_rounds_total, 1);
    assert_eq!(status.authority_coordinator_success_total, 0);
    assert_eq!(status.authority_carrier_attempts_total, 2);
    assert_eq!(status.authority_carrier_recoveries_total, 1);
    assert_eq!(status.authority_availability_exhausted_total, 0);
    assert_eq!(status.authority_security_stops_total, 0);
    assert_eq!(
        status.last_authority_sync_result.as_deref(),
        Some("carrier_recovered")
    );
    assert!(status.last_authority_carrier_recovered_at.is_some());

    stale_server.abort();
    server.abort();
}

#[tokio::test]
async fn coordinator_lease_client_verifies_grant_release_and_immediate_handover() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let witness = Arc::new(IdentityKeyPair::generate());
    let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    witness_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();
    let witness_peers = Arc::new(PeerStore::new());
    admit_peer(&witness_peers, &coordinator, None, now);
    let router = build_memchain_peer_router_with_coordinator_lease(
        witness_storage,
        witness_peers,
        Arc::clone(&witness),
        Some(coordinator.public_key_bytes()),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let coordinator_storage = MemoryStorage::open(":memory:", None).unwrap();
    coordinator_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();
    let coordinator_peers = PeerStore::new();
    admit_peer(
        &coordinator_peers,
        &witness,
        Some(format!("http://{address}")),
        now,
    );
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();
    let grant = request_record_commitment_coordinator_lease_with_endpoint_policy(
        &coordinator_storage,
        &coordinator_peers,
        &coordinator,
        &witness.public_key_bytes(),
        &[0x91; 32],
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(grant.lease_epoch, 1);
    assert!(grant.valid_for_secs > 0);

    let error = request_record_commitment_coordinator_lease_with_endpoint_policy(
        &coordinator_storage,
        &coordinator_peers,
        &coordinator,
        &witness.public_key_bytes(),
        &[0x92; 32],
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "lease_contended");

    let release = release_record_commitment_coordinator_lease_with_endpoint_policy(
        &coordinator_peers,
        &coordinator,
        &witness.public_key_bytes(),
        &[0x91; 32],
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(release.lease_epoch, 1);

    let takeover = request_record_commitment_coordinator_lease_with_endpoint_policy(
        &coordinator_storage,
        &coordinator_peers,
        &coordinator,
        &witness.public_key_bytes(),
        &[0x92; 32],
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(takeover.lease_epoch, 2);
    server.abort();
}
