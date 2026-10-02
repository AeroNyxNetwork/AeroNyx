// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn verified_delivery_anchor_response_rejects_forged_outcome_relation() {
    let requester = IdentityKeyPair::generate();
    let witness = IdentityKeyPair::generate();
    let request_id = [0x91; 16];
    let digest = [0x92; 32];
    let now = now_secs();
    let signing_bytes = verified_delivery_anchor_witness_response_signing_bytes(
        &request_id,
        &requester.public_key_bytes(),
        10,
        &digest,
        &witness.public_key_bytes(),
        now,
        9,
        &[0x93; 32],
        VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
    );
    let frame = encode_memchain(&MemChainMessage::VerifiedDeliveryAnchorWitnessResponseV1 {
        request_id,
        requester: requester.public_key_bytes(),
        requested_generation: 10,
        requested_anchor_digest: digest,
        witness: witness.public_key_bytes(),
        response_timestamp: now,
        witness_generation: 9,
        witness_anchor_digest: [0x93; 32],
        outcome: VERIFIED_DELIVERY_WITNESS_IDEMPOTENT_V1,
        signature: witness.sign(&signing_bytes),
    })
    .unwrap();

    assert_eq!(
        control_plane::verify_delivery_anchor_witness_response(
            &frame,
            &request_id,
            &requester.public_key_bytes(),
            10,
            &digest,
            &witness.public_key_bytes(),
            now,
        )
        .unwrap_err(),
        "delivery_witness_outcome_invalid"
    );
}

#[tokio::test]
async fn pinned_block_announcement_wakes_follower_without_mutating_chain() {
    let now = now_secs();
    let follower = Arc::new(IdentityKeyPair::generate());
    let coordinator = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.audit_record_commitment_chain().await.unwrap();
    storage.configure_record_commitment_sync(false, true);
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &coordinator, None, now);
    let (notifier, mut notifications) = mpsc::channel(1);
    let router = build_memchain_peer_router_with_runtime(
        Arc::clone(&storage),
        peer_store,
        follower,
        Some(coordinator.public_key_bytes()),
        Some(notifier),
    );
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x31; 32]],
        &coordinator,
    );
    let frame = block_announce_frame(&block);

    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/block-announce")
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(frame.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let accepted = storage.record_commitment_sync_status();
    assert_eq!(
        accepted.last_announcement_result.as_deref(),
        Some("accepted")
    );
    assert_eq!(accepted.announcements_accepted_total, 1);

    let next_block = RecordCommitmentBlockV1::new_signed(
        2,
        now,
        block.header.hash(),
        vec![[0x32; 32]],
        &coordinator,
    );
    let coalesced = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/block-announce")
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(block_announce_frame(&next_block)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(coalesced.status(), StatusCode::ACCEPTED);
    let coalesced_status = storage.record_commitment_sync_status();
    assert_eq!(
        coalesced_status.last_announcement_result.as_deref(),
        Some("coalesced")
    );
    assert_eq!(coalesced_status.last_announced_height, Some(2));
    assert_eq!(coalesced_status.announcements_accepted_total, 1);
    assert_eq!(coalesced_status.announcements_coalesced_total, 1);
    assert_eq!(notifications.recv().await, Some(1));
    assert_eq!(storage.record_commitment_chain_tip().await.0, 0);

    let retry = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/block-announce")
                .body(Body::from(frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(retry.status(), StatusCode::ACCEPTED);
    assert_eq!(notifications.recv().await, Some(1));
    let retry_status = storage.record_commitment_sync_status();
    assert_eq!(
        retry_status.last_announcement_result.as_deref(),
        Some("accepted")
    );
    assert_eq!(retry_status.last_announced_height, Some(2));
    assert_eq!(retry_status.announcements_accepted_total, 2);
    assert_eq!(retry_status.announcements_coalesced_total, 1);
    assert_eq!(storage.record_commitment_chain_tip().await.0, 0);
}

#[tokio::test]
async fn block_announcement_rejects_unpinned_or_invalid_proposer() {
    let now = now_secs();
    let follower = Arc::new(IdentityKeyPair::generate());
    let coordinator = IdentityKeyPair::generate();
    let unpinned = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.audit_record_commitment_chain().await.unwrap();
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &coordinator, None, now);
    admit_peer(&peer_store, &unpinned, None, now);
    let (notifier, mut notifications) = mpsc::channel(1);
    let router = build_memchain_peer_router_with_runtime(
        storage,
        peer_store,
        follower,
        Some(coordinator.public_key_bytes()),
        Some(notifier),
    );
    let unpinned_block =
        RecordCommitmentBlockV1::new_signed(1, now, GENESIS_PREV_HASH, vec![[0x32; 32]], &unpinned);
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/block-announce")
                .body(Body::from(block_announce_frame(&unpinned_block)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert!(notifications.try_recv().is_err());

    let coordinator_block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x34; 32]],
        &coordinator,
    );
    let coordinator_block_hash = coordinator_block.hash();
    let invalid_signature = encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
        header: coordinator_block.header,
        proposer_signature: unpinned.sign(&coordinator_block_hash),
    })
    .unwrap();
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/block-announce")
                .body(Body::from(invalid_signature))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(notifications.try_recv().is_err());
}

#[test]
fn response_verification_rejects_blocks_from_an_unpinned_proposer() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let other_writer = IdentityKeyPair::generate();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x44; 32]],
        &other_writer,
    );
    let request_id = [0x33; 16];
    let responder = coordinator.public_key_bytes();
    let blocks = vec![block.clone()];
    let signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        now,
        &blocks,
        false,
        1,
        &block.hash(),
    );
    let frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp: now,
        blocks,
        has_more: false,
        tip_height: 1,
        tip_hash: block.hash(),
        signature: coordinator.sign(&signing_bytes),
    })
    .unwrap();

    let error = verify_record_commitment_page(
        &frame,
        &request_id,
        &responder,
        &responder,
        (0, GENESIS_PREV_HASH),
        now,
    )
    .unwrap_err();
    assert_eq!(error, "unexpected_block_proposer");

    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] A valid carrier
    // envelope cannot promote carrier-authored blocks into the configured
    // coordinator namespace.
    let carrier = other_writer.public_key_bytes();
    let carrier_frame = signed_block_page_frame(
        &other_writer,
        request_id,
        now,
        vec![block.clone()],
        false,
        1,
        block.hash(),
    );
    let error = verify_record_commitment_page(
        &carrier_frame,
        &request_id,
        &carrier,
        &responder,
        (0, GENESIS_PREV_HASH),
        now,
    )
    .unwrap_err();
    assert_eq!(error, "unexpected_block_proposer");
}

#[test]
fn response_verification_binds_signature_request_and_pagination_metadata() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let responder = coordinator.public_key_bytes();
    let request_id = [0x81; 16];
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x82; 32]],
        &coordinator,
    );
    let valid = signed_block_page_frame(
        &coordinator,
        request_id,
        now,
        vec![block.clone()],
        false,
        1,
        block.hash(),
    );
    verify_record_commitment_page(
        &valid,
        &request_id,
        &responder,
        &responder,
        (0, GENESIS_PREV_HASH),
        now,
    )
    .unwrap();

    let invalid_signature_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        now,
        std::slice::from_ref(&block),
        false,
        1,
        &block.hash(),
    );
    let mut invalid_signature = coordinator.sign(&invalid_signature_bytes);
    invalid_signature[0] ^= 0x01;
    let invalid_signature_frame = encode_memchain(&MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp: now,
        blocks: vec![block.clone()],
        has_more: false,
        tip_height: 1,
        tip_hash: block.hash(),
        signature: invalid_signature,
    })
    .unwrap();
    assert_eq!(
        verify_record_commitment_page(
            &invalid_signature_frame,
            &request_id,
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err(),
        "invalid_response_signature"
    );
    assert_eq!(
        verify_record_commitment_page(
            &valid,
            &[0x83; 16],
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err(),
        "response_request_mismatch"
    );
    assert_eq!(
        verify_record_commitment_page(
            &valid,
            &request_id,
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS + 1),
        )
        .unwrap_err(),
        "stale_response"
    );

    let inconsistent_tip = signed_block_page_frame(
        &coordinator,
        request_id,
        now,
        vec![block.clone()],
        false,
        1,
        [0x84; 32],
    );
    assert_eq!(
        verify_record_commitment_page(
            &inconsistent_tip,
            &request_id,
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err(),
        "terminal_tip_mismatch"
    );

    let inconsistent_pagination = signed_block_page_frame(
        &coordinator,
        request_id,
        now,
        vec![block],
        true,
        1,
        [0x85; 32],
    );
    assert_eq!(
        verify_record_commitment_page(
            &inconsistent_pagination,
            &request_id,
            &responder,
            &responder,
            (0, GENESIS_PREV_HASH),
            now,
        )
        .unwrap_err(),
        "pagination_state_mismatch"
    );
}

#[tokio::test]
async fn live_http_follower_rejects_signed_malicious_page_without_mutation() {
    let now = now_secs();
    let coordinator = Arc::new(IdentityKeyPair::generate());
    let requester = IdentityKeyPair::generate();
    let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    destination.audit_record_commitment_chain().await.unwrap();

    // The pinned coordinator signs both layers, but the block deliberately
    // forks before genesis. Envelope authenticity must never replace chain
    // continuity verification.
    let malicious_block =
        RecordCommitmentBlockV1::new_signed(1, now, [0xF1; 32], vec![[0xF2; 32]], &coordinator);
    let router = Router::new().route(
        "/api/memchain/peer/block-range",
        post({
            let coordinator = Arc::clone(&coordinator);
            move |body: Bytes| {
                let coordinator = Arc::clone(&coordinator);
                let malicious_block = malicious_block.clone();
                async move {
                    assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
                    let request = decode_memchain(&body[1..]).unwrap();
                    let MemChainMessage::RecordBlockRangeRequestV1 { request_id, .. } = request
                    else {
                        panic!("expected commitment block range request");
                    };
                    let tip_hash = malicious_block.hash();
                    let frame = signed_block_page_frame(
                        &coordinator,
                        request_id,
                        now_secs(),
                        vec![malicious_block],
                        false,
                        1,
                        tip_hash,
                    );
                    (
                        StatusCode::OK,
                        [(header::CONTENT_TYPE, "application/octet-stream")],
                        frame,
                    )
                        .into_response()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let peers = PeerStore::new();
    admit_peer(&peers, &coordinator, Some(format!("http://{address}")), now);
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let error = pull_record_commitment_page_with_endpoint_policy(
        &destination,
        &peers,
        &requester,
        &coordinator.public_key_bytes(),
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "commitment_chain_verification_failed");
    assert_eq!(
        destination.record_commitment_chain_tip().await,
        (0, GENESIS_PREV_HASH)
    );
    let status = destination.record_commitment_chain_status().await;
    assert_eq!(status.block_count, 0);
    assert_eq!(status.commitment_count, 0);

    server.abort();
    let _ = server.await;
}
