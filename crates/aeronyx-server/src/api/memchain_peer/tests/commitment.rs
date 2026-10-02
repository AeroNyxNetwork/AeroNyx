// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn verified_delivery_anchor_witness_round_is_signed_contiguous_and_bounded() {
    let now = now_secs();
    let requester = IdentityKeyPair::generate();
    let witness = Arc::new(IdentityKeyPair::generate());
    let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let witness_peers = Arc::new(PeerStore::new());
    admit_peer(&witness_peers, &requester, None, now);
    let router = build_memchain_peer_router(
        witness_storage,
        Arc::clone(&witness_peers),
        Arc::clone(&witness),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let requester_peers = PeerStore::new();
    admit_peer(
        &requester_peers,
        &witness,
        Some(format!("http://{address}")),
        now,
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let witness_id = witness.public_key_bytes();
    let digest_10 = [0x81; 32];

    let denied = witness_verified_delivery_anchor_with_endpoint_policy(
        &requester_peers,
        &requester,
        &client,
        &[witness_id],
        9,
        &[0x80; 32],
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(denied.attempted, 1);
    assert_eq!(denied.verified, 0);
    assert_eq!(denied.failed, 1);

    witness_peers.configure_verified_delivery_witness_requesters(&[requester.public_key_bytes()]);

    let advanced = witness_verified_delivery_anchor_with_endpoint_policy(
        &requester_peers,
        &requester,
        &client,
        &[witness_id, witness_id],
        10,
        &digest_10,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(
        advanced,
        VerifiedDeliveryAnchorWitnessRound {
            configured: 1,
            attempted: 1,
            verified: 1,
            advanced: 1,
            ..VerifiedDeliveryAnchorWitnessRound::default()
        }
    );

    let idempotent = witness_verified_delivery_anchor_with_endpoint_policy(
        &requester_peers,
        &requester,
        &client,
        &[witness_id],
        10,
        &digest_10,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(idempotent.verified, 1);
    assert_eq!(idempotent.idempotent, 1);

    let gap = witness_verified_delivery_anchor_with_endpoint_policy(
        &requester_peers,
        &requester,
        &client,
        &[witness_id],
        12,
        &[0x82; 32],
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(gap.verified, 1);
    assert_eq!(gap.gaps, 1);

    let advanced_next = witness_verified_delivery_anchor_with_endpoint_policy(
        &requester_peers,
        &requester,
        &client,
        &[witness_id],
        11,
        &[0x83; 32],
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(advanced_next.verified, 1);
    assert_eq!(advanced_next.advanced, 1);
    server.abort();
}

#[tokio::test]
async fn delivery_witness_authenticates_before_admission_and_rejects_padding() {
    // [WITNESS-ADMISSION-PRIVACY 2026-08-16 by Codex] The same forged
    // identity receives the same signature failure before and after local
    // admission. Alternate encodings never reach durable state.
    let now = now_secs();
    let requester = IdentityKeyPair::from_bytes(&[0x81; 32]).expect("requester identity");
    let witness = Arc::new(IdentityKeyPair::from_bytes(&[0x82; 32]).expect("witness identity"));
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let peers = Arc::new(PeerStore::new());
    admit_peer(&peers, &requester, None, now);
    let router = build_memchain_peer_router(storage, Arc::clone(&peers), witness);
    let valid_frame = delivery_witness_request_frame(&requester, 1, [0x83; 32], [0x84; 16], now);

    let mut forged_message =
        decode_memchain(&valid_frame[1..]).expect("decode delivery request for tamper");
    let MemChainMessage::VerifiedDeliveryAnchorWitnessRequestV1 {
        ref mut signature, ..
    } = forged_message
    else {
        panic!("expected delivery witness request");
    };
    signature[0] ^= 0x01;
    let forged_frame = encode_memchain(&forged_message).expect("encode forged delivery request");
    assert_eq!(
        post_delivery_witness(&router, forged_frame.clone())
            .await
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let mut padded_frame = valid_frame.clone();
    padded_frame.push(0);
    assert_eq!(
        post_delivery_witness(&router, padded_frame).await.status(),
        StatusCode::BAD_REQUEST
    );
    assert_eq!(
        post_delivery_witness(&router, valid_frame.clone())
            .await
            .status(),
        StatusCode::FORBIDDEN
    );

    peers.configure_verified_delivery_witness_requesters(&[requester.public_key_bytes()]);
    assert_eq!(
        post_delivery_witness(&router, forged_frame).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        post_delivery_witness(&router, valid_frame).await.status(),
        StatusCode::OK
    );
}

#[test]
fn custody_witness_planner_is_bounded_independent_and_non_transmitting() {
    // [CUSTODY-WITNESS-PLANNER 2026-08-16 by Codex] A pure plan is safe
    // to run in unit tests without an HTTP client or custody anchor.
    let now = now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0x91; 32]).expect("producer identity");
    let eligible = IdentityKeyPair::from_bytes(&[0x92; 32]).expect("eligible witness");
    let unavailable = IdentityKeyPair::from_bytes(&[0x93; 32]).expect("unavailable witness");
    let wrong_capability =
        IdentityKeyPair::from_bytes(&[0x94; 32]).expect("wrong-capability witness");
    let peers = PeerStore::new();
    admit_peer(
        &peers,
        &eligible,
        Some("http://127.0.0.1:8422".to_string()),
        now,
    );
    admit_peer(&peers, &unavailable, None, now);
    let mut wrong_capability_descriptor = NodeDescriptor::new(
        wrong_capability.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now.saturating_add(600),
        "memchain-sync-test",
    );
    wrong_capability_descriptor.public_endpoint = Some("http://127.0.0.1:8422".to_string());
    wrong_capability_descriptor.capabilities = vec![NodeCapability::ChatRelay];
    let wrong_capability_descriptor =
        SignedNodeDescriptor::sign(wrong_capability_descriptor, &wrong_capability)
            .expect("sign wrong-capability descriptor");
    let import = peers.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: wrong_capability_descriptor,
        },
        now,
    );
    assert_eq!(import.inserted, 1);

    let deduplicated = plan_custody_audit_witnesses_with_endpoint_policy(
        &peers,
        &producer.public_key_bytes(),
        &[
            eligible.public_key_bytes(),
            eligible.public_key_bytes(),
            producer.public_key_bytes(),
        ],
        1,
        now,
        &allow_test_endpoint,
    )
    .expect("plan duplicate and self pins");
    assert_eq!(
        deduplicated,
        CustodyAuditWitnessPlan {
            configured: 1,
            eligible: 1,
            unavailable: 0,
            duplicates_ignored: 1,
            self_excluded: 1,
            minimum_verified: 1,
            quorum_ready: true,
        }
    );

    let insufficient = plan_custody_audit_witnesses_with_endpoint_policy(
        &peers,
        &producer.public_key_bytes(),
        &[eligible.public_key_bytes(), unavailable.public_key_bytes()],
        2,
        now,
        &allow_test_endpoint,
    )
    .expect("plan insufficient current eligibility");
    assert_eq!(insufficient.configured, 2);
    assert_eq!(insufficient.eligible, 1);
    assert_eq!(insufficient.unavailable, 1);
    assert!(!insufficient.quorum_ready);

    let wrong_capability_plan = plan_custody_audit_witnesses_with_endpoint_policy(
        &peers,
        &producer.public_key_bytes(),
        &[wrong_capability.public_key_bytes()],
        1,
        now,
        &allow_test_endpoint,
    )
    .expect("plan rejects a witness without encrypted-storage capability");
    assert_eq!(wrong_capability_plan.eligible, 0);
    assert_eq!(wrong_capability_plan.unavailable, 1);
    assert!(!wrong_capability_plan.quorum_ready);

    assert!(plan_custody_audit_witnesses_with_endpoint_policy(
        &peers,
        &producer.public_key_bytes(),
        &[],
        0,
        now,
        &allow_test_endpoint,
    )
    .is_err());
    assert!(plan_custody_audit_witnesses_with_endpoint_policy(
        &peers,
        &producer.public_key_bytes(),
        &[[0x01; 32], [0x02; 32], [0x03; 32], [0x04; 32]],
        1,
        now,
        &allow_test_endpoint,
    )
    .is_err());
}

#[tokio::test]
async fn custody_witness_transport_is_concurrent_bounded_and_adverse_evidence_fails_closed() {
    // [CUSTODY-WITNESS-TRANSPORT 2026-08-16 by Codex] Exercise the exact
    // public wire path while proving duplicate/self pins cannot inflate
    // quorum and a valid adverse receipt cannot be outvoted.
    use std::sync::atomic::{AtomicBool, Ordering};

    let now = now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0x95; 32]).expect("producer identity");
    let producer_storage = MemoryStorage::open(":memory:", None).unwrap();
    let witness = Arc::new(IdentityKeyPair::from_bytes(&[0x96; 32]).expect("witness identity"));
    let second_witness =
        Arc::new(IdentityKeyPair::from_bytes(&[0x9C; 32]).expect("second witness identity"));
    let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let witness_peers = Arc::new(PeerStore::new());
    admit_peer(&witness_peers, &producer, None, now);
    witness_peers.configure_custody_audit_witness_requesters(&[producer.public_key_bytes()]);
    let concurrent_gate = Arc::new(AtomicBool::new(false));
    let concurrent_barrier = Arc::new(tokio::sync::Barrier::new(2));
    let router = build_memchain_peer_router(
        witness_storage,
        Arc::clone(&witness_peers),
        Arc::clone(&witness),
    )
    .layer(axum::middleware::from_fn({
        let gate = Arc::clone(&concurrent_gate);
        let barrier = Arc::clone(&concurrent_barrier);
        move |request: axum::extract::Request, next: axum::middleware::Next| {
            let gate = Arc::clone(&gate);
            let barrier = Arc::clone(&barrier);
            async move {
                if gate.load(Ordering::SeqCst) {
                    barrier.wait().await;
                }
                next.run(request).await
            }
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let second_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let second_peers = Arc::new(PeerStore::new());
    admit_peer(&second_peers, &producer, None, now);
    second_peers.configure_custody_audit_witness_requesters(&[producer.public_key_bytes()]);
    let second_router = build_memchain_peer_router(
        second_storage,
        Arc::clone(&second_peers),
        Arc::clone(&second_witness),
    )
    .layer(axum::middleware::from_fn({
        let gate = Arc::clone(&concurrent_gate);
        let barrier = Arc::clone(&concurrent_barrier);
        move |request: axum::extract::Request, next: axum::middleware::Next| {
            let gate = Arc::clone(&gate);
            let barrier = Arc::clone(&barrier);
            async move {
                if gate.load(Ordering::SeqCst) {
                    barrier.wait().await;
                }
                next.run(request).await
            }
        }
    }));
    let second_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let second_address = second_listener.local_addr().unwrap();
    let second_server = tokio::spawn(async move {
        axum::serve(second_listener, second_router).await.unwrap();
    });

    let producer_peers = PeerStore::new();
    admit_peer(
        &producer_peers,
        &witness,
        Some(format!("http://{address}")),
        now,
    );
    admit_peer(
        &producer_peers,
        &second_witness,
        Some(format!("http://{second_address}")),
        now,
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let witness_node_id = witness.public_key_bytes();
    let second_witness_node_id = second_witness.public_key_bytes();
    let anchor_1 = CustodyAuditAnchorV1::signed(1, 100, 10_000, [0x97; 32], &producer)
        .expect("sign generation one anchor");

    let advanced = witness_custody_audit_anchor_round_with_endpoint_policy(
        &producer_peers,
        &producer,
        &client,
        &[
            witness_node_id,
            witness_node_id,
            producer.public_key_bytes(),
        ],
        1,
        &anchor_1,
        Some(&producer_storage),
        &allow_test_endpoint,
    )
    .await
    .expect("advance exact custody anchor");
    assert_eq!(
        advanced,
        CustodyAuditWitnessRound {
            configured: 1,
            verified: 1,
            accepted: 1,
            advanced: 1,
            duplicates_ignored: 1,
            self_excluded: 1,
            minimum_verified: 1,
            quorum_satisfied: true,
            ..CustodyAuditWitnessRound::default()
        }
    );

    let idempotent = witness_custody_audit_anchor_round_with_endpoint_policy(
        &producer_peers,
        &producer,
        &client,
        &[witness_node_id],
        1,
        &anchor_1,
        Some(&producer_storage),
        &allow_test_endpoint,
    )
    .await
    .expect("retry exact custody anchor");
    assert_eq!(idempotent.verified, 1);
    assert_eq!(idempotent.accepted, 1);
    assert_eq!(idempotent.idempotent, 1);
    assert!(idempotent.quorum_satisfied);
    let anchor_1_sha256 = custody_audit_anchor_frame_sha256(&anchor_1).unwrap();
    let anchor_1_evidence = producer_storage
        .evaluate_custody_audit_witness_receipt_policy(
            &producer.public_key_bytes(),
            1,
            &anchor_1_sha256,
            &[witness_node_id],
            1,
            now_secs(),
            60,
        )
        .await
        .expect("reconstruct accepted anchor policy from durable receipts");
    assert_eq!(anchor_1_evidence.fresh_verified, 1);
    assert_eq!(anchor_1_evidence.accepted, 1);
    assert_eq!(anchor_1_evidence.adverse, 0);
    assert!(anchor_1_evidence.quorum_satisfied);

    let gap_anchor = CustodyAuditAnchorV1::signed(3, 300, 30_000, [0x98; 32], &producer)
        .expect("sign generation gap anchor");
    let mixed = witness_custody_audit_anchor_round_with_endpoint_policy(
        &producer_peers,
        &producer,
        &client,
        &[witness_node_id, second_witness_node_id],
        1,
        &gap_anchor,
        Some(&producer_storage),
        &allow_test_endpoint,
    )
    .await
    .expect("receive accepted and adverse portable evidence");
    assert_eq!(mixed.verified, 2);
    assert_eq!(mixed.accepted, 1);
    assert_eq!(mixed.advanced, 1);
    assert_eq!(mixed.gaps, 1);
    assert!(mixed.adverse_evidence);
    assert!(!mixed.quorum_satisfied);
    let gap_anchor_sha256 = custody_audit_anchor_frame_sha256(&gap_anchor).unwrap();
    let gap_evidence = producer_storage
        .evaluate_custody_audit_witness_receipt_policy(
            &producer.public_key_bytes(),
            3,
            &gap_anchor_sha256,
            &[witness_node_id, second_witness_node_id],
            1,
            now_secs(),
            60,
        )
        .await
        .expect("reconstruct mixed anchor policy from durable receipts");
    assert_eq!(gap_evidence.fresh_verified, 2);
    assert_eq!(gap_evidence.accepted, 1);
    assert_eq!(gap_evidence.adverse, 1);
    assert!(!gap_evidence.quorum_satisfied);

    // [CUSTODY-WITNESS-CONCURRENT-ROUND 2026-08-19 by Codex] Both real
    // HTTP handlers wait on the same two-party barrier. A sequential round
    // would time out before either handler could answer; the bounded
    // concurrent round reaches both and retains both signed receipts.
    concurrent_gate.store(true, Ordering::SeqCst);
    let anchor_2 = CustodyAuditAnchorV1::signed(2, 200, 20_000, [0x9D; 32], &producer)
        .expect("sign concurrent generation two anchor");
    let concurrent = witness_custody_audit_anchor_round_with_endpoint_policy(
        &producer_peers,
        &producer,
        &client,
        &[witness_node_id, second_witness_node_id],
        1,
        &anchor_2,
        Some(&producer_storage),
        &allow_test_endpoint,
    )
    .await
    .expect("complete bounded concurrent custody witness round");
    assert_eq!(concurrent.configured, 2);
    assert_eq!(concurrent.verified, 2);
    assert_eq!(concurrent.accepted, 1);
    assert_eq!(concurrent.advanced, 1);
    assert_eq!(concurrent.stale, 1);
    assert!(concurrent.adverse_evidence);
    assert!(!concurrent.quorum_satisfied);

    let unrelated = IdentityKeyPair::from_bytes(&[0x99; 32]).expect("unrelated identity");
    let unrelated_anchor = CustodyAuditAnchorV1::signed(2, 200, 20_000, [0x9A; 32], &unrelated)
        .expect("sign unrelated anchor");
    assert!(witness_custody_audit_anchor_round_with_endpoint_policy(
        &producer_peers,
        &producer,
        &client,
        &[witness_node_id],
        1,
        &unrelated_anchor,
        Some(&producer_storage),
        &allow_test_endpoint,
    )
    .await
    .is_err());

    let unpersistable_generation = CustodyAuditAnchorV1::signed(
        i64::MAX as u64 + 1,
        i64::MAX as u64 + 1,
        1,
        [0x9B; 32],
        &producer,
    )
    .expect("sign out-of-storage-range anchor");
    assert!(witness_custody_audit_anchor_round_with_endpoint_policy(
        &producer_peers,
        &producer,
        &client,
        &[witness_node_id],
        1,
        &unpersistable_generation,
        Some(&producer_storage),
        &allow_test_endpoint,
    )
    .await
    .is_err());

    server.abort();
    second_server.abort();
}

#[tokio::test]
async fn custody_audit_witness_endpoint_is_pinned_signed_and_contiguous() {
    // [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Exercise the public
    // handler in process: no external endpoint or custody metadata leaves
    // this test while admission, durable outcomes, and signatures remain
    // identical to production routing.
    let now = now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0xA1; 32]).expect("producer identity");
    let witness = Arc::new(IdentityKeyPair::from_bytes(&[0xA2; 32]).expect("witness identity"));
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let peers = Arc::new(PeerStore::new());
    admit_peer(&peers, &producer, None, now);
    let router = build_memchain_peer_router(
        Arc::clone(&storage),
        Arc::clone(&peers),
        Arc::clone(&witness),
    );
    let anchor_10 = CustodyAuditAnchorV1::signed(10, 100, 10_000, [0xA3; 32], &producer)
        .expect("sign generation 10 anchor");
    let anchor_10_sha =
        custody_audit_anchor_frame_sha256(&anchor_10).expect("hash generation 10 anchor");

    let mut forged_message = decode_memchain(
        &custody_witness_request_frame(&producer, &anchor_10, [0xC1; 16], now)[1..],
    )
    .expect("decode request for signature tamper");
    let MemChainMessage::CustodyAuditAnchorWitnessRequestV1 {
        ref mut signature, ..
    } = forged_message
    else {
        panic!("expected custody witness request");
    };
    signature[0] ^= 0x01;
    let forged_frame = encode_memchain(&forged_message).expect("encode signature-tampered request");
    let forged_unpinned = post_custody_witness(&router, forged_frame.clone()).await;
    assert_eq!(forged_unpinned.status(), StatusCode::UNAUTHORIZED);

    let denied = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &anchor_10, [0xA4; 16], now),
    )
    .await;
    assert_eq!(denied.status(), StatusCode::FORBIDDEN);

    peers.configure_custody_audit_witness_requesters(&[producer.public_key_bytes()]);
    let forged = post_custody_witness(&router, forged_frame).await;
    assert_eq!(forged.status(), StatusCode::UNAUTHORIZED);

    let stale = post_custody_witness(
        &router,
        custody_witness_request_frame(
            &producer,
            &anchor_10,
            [0xC2; 16],
            now.saturating_sub(REQUEST_TIMESTAMP_SKEW_SECS + 1),
        ),
    )
    .await;
    assert_eq!(stale.status(), StatusCode::UNAUTHORIZED);

    let advanced_request_id = [0xA4; 16];
    let advanced = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &anchor_10, advanced_request_id, now),
    )
    .await;
    assert_eq!(advanced.status(), StatusCode::OK);
    let advanced_body = axum::body::to_bytes(advanced.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let advanced_receipt = control_plane::verify_custody_audit_anchor_witness_response(
        &advanced_body,
        &advanced_request_id,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        &anchor_10,
        &anchor_10_sha,
        now,
    )
    .expect("verify advanced custody receipt");
    assert_eq!(advanced_receipt.outcome, CUSTODY_AUDIT_WITNESS_ADVANCED_V1);

    let replayed = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &anchor_10, advanced_request_id, now),
    )
    .await;
    assert_eq!(replayed.status(), StatusCode::TOO_MANY_REQUESTS);

    let idempotent_request_id = [0xA5; 16];
    let idempotent = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &anchor_10, idempotent_request_id, now),
    )
    .await;
    assert_eq!(idempotent.status(), StatusCode::OK);
    let idempotent_body = axum::body::to_bytes(idempotent.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let idempotent_receipt = control_plane::verify_custody_audit_anchor_witness_response(
        &idempotent_body,
        &idempotent_request_id,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        &anchor_10,
        &anchor_10_sha,
        now,
    )
    .expect("verify idempotent custody receipt");
    assert_eq!(
        idempotent_receipt.outcome,
        CUSTODY_AUDIT_WITNESS_IDEMPOTENT_V1
    );

    let conflicting_anchor = CustodyAuditAnchorV1::signed(10, 101, 10_001, [0xA6; 32], &producer)
        .expect("sign conflicting anchor");
    let conflict_request_id = [0xA7; 16];
    let conflict = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &conflicting_anchor, conflict_request_id, now),
    )
    .await;
    assert_eq!(conflict.status(), StatusCode::OK);
    let conflict_body = axum::body::to_bytes(conflict.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let conflict_receipt = control_plane::verify_custody_audit_anchor_witness_response(
        &conflict_body,
        &conflict_request_id,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        &conflicting_anchor,
        &custody_audit_anchor_frame_sha256(&conflicting_anchor).expect("hash conflicting anchor"),
        now,
    )
    .expect("verify conflict custody receipt");
    assert_eq!(conflict_receipt.outcome, CUSTODY_AUDIT_WITNESS_CONFLICT_V1);

    let gap_anchor = CustodyAuditAnchorV1::signed(12, 120, 12_000, [0xA8; 32], &producer)
        .expect("sign gap anchor");
    let gap_request_id = [0xA9; 16];
    let gap = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &gap_anchor, gap_request_id, now),
    )
    .await;
    assert_eq!(gap.status(), StatusCode::OK);
    let gap_body = axum::body::to_bytes(gap.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let gap_receipt = control_plane::verify_custody_audit_anchor_witness_response(
        &gap_body,
        &gap_request_id,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        &gap_anchor,
        &custody_audit_anchor_frame_sha256(&gap_anchor).expect("hash gap anchor"),
        now,
    )
    .expect("verify gap custody receipt");
    assert_eq!(gap_receipt.outcome, CUSTODY_AUDIT_WITNESS_GAP_V1);

    let anchor_11 = CustodyAuditAnchorV1::signed(11, 110, 11_000, [0xAA; 32], &producer)
        .expect("sign generation 11 anchor");
    let next_request_id = [0xAB; 16];
    let next = post_custody_witness(
        &router,
        custody_witness_request_frame(&producer, &anchor_11, next_request_id, now),
    )
    .await;
    assert_eq!(next.status(), StatusCode::OK);
    let next_body = axum::body::to_bytes(next.into_body(), MAX_RESPONSE_BODY_BYTES)
        .await
        .unwrap();
    let next_receipt = control_plane::verify_custody_audit_anchor_witness_response(
        &next_body,
        &next_request_id,
        &producer.public_key_bytes(),
        &witness.public_key_bytes(),
        &anchor_11,
        &custody_audit_anchor_frame_sha256(&anchor_11).expect("hash generation 11 anchor"),
        now,
    )
    .expect("verify generation 11 custody receipt");
    assert_eq!(next_receipt.outcome, CUSTODY_AUDIT_WITNESS_ADVANCED_V1);
}

#[tokio::test]
async fn custody_audit_witness_endpoint_rejects_self_witness_before_persistence() {
    let now = now_secs();
    let identity =
        Arc::new(IdentityKeyPair::from_bytes(&[0xB1; 32]).expect("self witness identity"));
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let peers = Arc::new(PeerStore::new());
    admit_peer(&peers, &identity, None, now);
    peers.configure_custody_audit_witness_requesters(&[identity.public_key_bytes()]);
    let router = build_memchain_peer_router(storage, peers, Arc::clone(&identity));
    let anchor = CustodyAuditAnchorV1::signed(1, 1, 1, [0xB2; 32], &identity)
        .expect("sign self custody anchor");
    let response = post_custody_witness(
        &router,
        custody_witness_request_frame(&identity, &anchor, [0xB3; 16], now),
    )
    .await;
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[test]
fn tip_announcement_status_contract_is_exact() {
    assert_eq!(
        classify_commitment_tip_announcement_status(StatusCode::ACCEPTED.as_u16()),
        CommitmentTipAnnouncementDelivery::Accepted
    );
    assert_eq!(
        classify_commitment_tip_announcement_status(StatusCode::NO_CONTENT.as_u16()),
        CommitmentTipAnnouncementDelivery::Stale
    );
    assert_eq!(
        classify_commitment_tip_announcement_status(StatusCode::OK.as_u16()),
        CommitmentTipAnnouncementDelivery::PermanentFailure
    );
    assert_eq!(
        classify_commitment_tip_announcement_status(StatusCode::SERVICE_UNAVAILABLE.as_u16()),
        CommitmentTipAnnouncementDelivery::RetryableFailure
    );
    assert_eq!(
        classify_commitment_tip_announcement_status(StatusCode::TOO_MANY_REQUESTS.as_u16()),
        CommitmentTipAnnouncementDelivery::PermanentFailure
    );
}

#[tokio::test]
async fn coordinator_tip_announcement_reaches_pinned_follower_runtime() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let follower = Arc::new(IdentityKeyPair::generate());
    let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x33; 32]],
        &coordinator,
    );
    source_storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    source_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let follower_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    follower_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();
    let follower_peers = Arc::new(PeerStore::new());
    admit_peer(&follower_peers, &coordinator, None, now);
    let (notifier, mut notifications) = mpsc::channel(1);
    let router = build_memchain_peer_router_with_runtime(
        follower_storage,
        follower_peers,
        Arc::clone(&follower),
        Some(coordinator.public_key_bytes()),
        Some(notifier),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let source_peers = PeerStore::new();
    admit_peer(
        &source_peers,
        &follower,
        Some(format!("http://{address}")),
        now,
    );
    let outcome = announce_current_record_commitment_tip_with_endpoint_policy(
        &source_storage,
        &source_peers,
        &coordinator,
        &reqwest::Client::new(),
        &[follower.public_key_bytes()],
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(outcome.announced_height, 1);
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.accepted, 1);
    assert_eq!(outcome.stale, 0);
    assert_eq!(outcome.failed, 0);
    assert_eq!(outcome.retries_attempted, 0);
    assert_eq!(outcome.retries_succeeded, 0);
    assert_eq!(outcome.retries_exhausted, 0);
    assert_eq!(notifications.recv().await, Some(1));

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn coordinator_tip_announcement_reports_current_follower_as_stale() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let follower = Arc::new(IdentityKeyPair::generate());
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x34; 32]],
        &coordinator,
    );

    let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    source_storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    source_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();
    let follower_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    follower_storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    follower_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let follower_peers = Arc::new(PeerStore::new());
    admit_peer(&follower_peers, &coordinator, None, now);
    let (notifier, mut notifications) = mpsc::channel(1);
    let router = build_memchain_peer_router_with_runtime(
        follower_storage,
        follower_peers,
        Arc::clone(&follower),
        Some(coordinator.public_key_bytes()),
        Some(notifier),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let source_peers = PeerStore::new();
    admit_peer(
        &source_peers,
        &follower,
        Some(format!("http://{address}")),
        now,
    );
    let outcome = announce_current_record_commitment_tip_with_endpoint_policy(
        &source_storage,
        &source_peers,
        &coordinator,
        &reqwest::Client::new(),
        &[follower.public_key_bytes()],
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(outcome.announced_height, 1);
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.accepted, 0);
    assert_eq!(outcome.stale, 1);
    assert_eq!(outcome.failed, 0);
    assert_eq!(outcome.retries_attempted, 0);
    assert_eq!(outcome.retries_succeeded, 0);
    assert_eq!(outcome.retries_exhausted, 0);
    assert!(notifications.try_recv().is_err());

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn coordinator_tip_announcement_retries_transient_failure_to_success() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let follower = IdentityKeyPair::generate();
    let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x35; 32]],
        &coordinator,
    );
    source_storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    source_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let attempts = Arc::new(AtomicUsize::new(0));
    let handler_attempts = Arc::clone(&attempts);
    let endpoint_checks = AtomicUsize::new(0);
    let endpoint_policy = |_endpoint: &str| {
        endpoint_checks.fetch_add(1, Ordering::SeqCst);
        true
    };
    let router = Router::new().route(
        "/api/memchain/peer/block-announce",
        post(move || {
            let attempts = Arc::clone(&handler_attempts);
            async move {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::ACCEPTED
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let source_peers = PeerStore::new();
    admit_peer(
        &source_peers,
        &follower,
        Some(format!("http://{address}")),
        now,
    );
    let outcome = announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        &source_storage,
        &source_peers,
        &coordinator,
        &reqwest::Client::new(),
        &[follower.public_key_bytes()],
        &endpoint_policy,
        CommitmentTipAnnouncementRetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
        },
    )
    .await
    .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert_eq!(endpoint_checks.load(Ordering::SeqCst), 2);
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.accepted, 1);
    assert_eq!(outcome.failed, 0);
    assert_eq!(outcome.retries_attempted, 1);
    assert_eq!(outcome.retries_succeeded, 1);
    assert_eq!(outcome.retries_exhausted, 0);

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn coordinator_tip_announcement_stops_after_retry_budget() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let follower = IdentityKeyPair::generate();
    let source_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now,
        GENESIS_PREV_HASH,
        vec![[0x36; 32]],
        &coordinator,
    );
    source_storage
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    source_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let attempts = Arc::new(AtomicUsize::new(0));
    let handler_attempts = Arc::clone(&attempts);
    let router = Router::new().route(
        "/api/memchain/peer/block-announce",
        post(move || {
            let attempts = Arc::clone(&handler_attempts);
            async move {
                attempts.fetch_add(1, Ordering::SeqCst);
                StatusCode::SERVICE_UNAVAILABLE
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let source_peers = PeerStore::new();
    admit_peer(
        &source_peers,
        &follower,
        Some(format!("http://{address}")),
        now,
    );
    let outcome = announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        &source_storage,
        &source_peers,
        &coordinator,
        &reqwest::Client::new(),
        &[follower.public_key_bytes()],
        &allow_test_endpoint,
        CommitmentTipAnnouncementRetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
        },
    )
    .await
    .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 3);
    assert_eq!(outcome.attempted, 1);
    assert_eq!(outcome.accepted, 0);
    assert_eq!(outcome.failed, 1);
    assert_eq!(outcome.retries_attempted, 2);
    assert_eq!(outcome.retries_succeeded, 0);
    assert_eq!(outcome.retries_exhausted, 1);

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn coordinator_lease_endpoint_rejects_unpinned_invalid_and_wrong_tip_requests() {
    let now = now_secs();
    let witness = Arc::new(IdentityKeyPair::generate());
    let coordinator = IdentityKeyPair::generate();
    let unpinned = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    storage.audit_record_commitment_chain().await.unwrap();
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &coordinator, None, now);
    admit_peer(&peer_store, &unpinned, None, now);
    let router = build_memchain_peer_router_with_coordinator_lease(
        storage,
        peer_store,
        witness,
        Some(coordinator.public_key_bytes()),
    );

    let unpinned_frame = coordinator_lease_request_frame(
        &unpinned,
        [0x81; 32],
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x82; 16],
        now,
    );
    let unpinned_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(unpinned_frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unpinned_response.status(), StatusCode::FORBIDDEN);

    let mut invalid_signature = coordinator_lease_request_frame(
        &coordinator,
        [0x83; 32],
        0,
        GENESIS_PREV_HASH,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x84; 16],
        now,
    );
    let last = invalid_signature.len() - 1;
    invalid_signature[last] ^= 0x01;
    let signature_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(invalid_signature))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(signature_response.status(), StatusCode::UNAUTHORIZED);

    let wrong_tip = coordinator_lease_request_frame(
        &coordinator,
        [0x85; 32],
        1,
        [0x86; 32],
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        [0x87; 16],
        now,
    );
    let tip_response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/coordinator-lease")
                .body(Body::from(wrong_tip))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(tip_response.status(), StatusCode::CONFLICT);
}

#[test]
fn commitment_range_url_is_bounded_to_the_peer_api_path() {
    let url = commitment_block_range_url("https://node.example/ignored?secret=no").unwrap();
    assert_eq!(
        url.as_str(),
        "https://node.example/api/memchain/peer/block-range"
    );
    assert!(commitment_block_range_url("ftp://node.example").is_err());
    assert!(commitment_block_range_url("https://user@node.example").is_err());
    assert_eq!(
        commitment_checkpoint_url("node.example:9281/path")
            .unwrap()
            .as_str(),
        "http://node.example:9281/api/memchain/peer/checkpoint"
    );
    assert_eq!(
        commitment_checkpoint_certificate_url("node.example:9281/path")
            .unwrap()
            .as_str(),
        "http://node.example:9281/api/memchain/peer/checkpoint-certificate"
    );
}

#[test]
fn commitment_peer_endpoint_rejects_ssrf_targets() {
    assert!(commitment_peer_endpoint_is_public("http://8.8.8.8:8422"));
    assert!(commitment_peer_endpoint_is_public(
        "https://[2606:4700:4700::1111]:8422"
    ));
    for endpoint in [
        "http://127.0.0.1:8422",
        "http://127.1:8422",
        "http://2130706433:8422",
        "http://0x7f000001:8422",
        "http://017700000001:8422",
        "http://10.0.0.1:8422",
        "http://100.64.0.1:8422",
        "http://169.254.1.1:8422",
        "http://172.16.0.1:8422",
        "http://192.168.1.1:8422",
        "http://198.18.0.1:8422",
        "http://203.0.113.1:8422",
        "http://node.example:8422",
        "http://[::1]:8422",
        "http://[::ffff:127.0.0.1]:8422",
        "http://[fc00::1]:8422",
        "http://[fe80::1]:8422",
        "http://[2001:db8::1]:8422",
    ] {
        assert!(
            !commitment_peer_endpoint_is_public(endpoint),
            "unexpectedly accepted {endpoint}"
        );
    }
}

#[tokio::test]
async fn outbound_commitment_pulls_reject_private_descriptor_targets() {
    let now = now_secs();
    let local_identity = IdentityKeyPair::generate();
    let remote_identity = IdentityKeyPair::generate();
    let peer_store = PeerStore::new();
    admit_peer(
        &peer_store,
        &remote_identity,
        Some("http://169.254.169.254/latest/meta-data".to_string()),
        now,
    );
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let remote_id = remote_identity.public_key_bytes();

    let checkpoint_error = pull_record_commitment_checkpoint(
        &storage,
        &peer_store,
        &local_identity,
        &remote_id,
        &client,
    )
    .await
    .unwrap_err();
    assert_eq!(checkpoint_error, "pinned_coordinator_unsafe_endpoint");

    let page_error =
        pull_record_commitment_page(&storage, &peer_store, &local_identity, &remote_id, &client)
            .await
            .unwrap_err();
    assert_eq!(page_error, "pinned_coordinator_unsafe_endpoint");

    let certificate_error = pull_record_commitment_checkpoint_certificate(
        &storage,
        &peer_store,
        &local_identity,
        &remote_id,
        &[remote_id, IdentityKeyPair::generate().public_key_bytes()],
        2,
        &client,
    )
    .await
    .unwrap_err();
    assert_eq!(certificate_error, "certificate_source_unsafe_endpoint");
}

#[tokio::test]
async fn witness_reconciliation_rechecks_endpoint_after_selection() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let now = now_secs();
    let local_identity = IdentityKeyPair::generate();
    let remote_identity = IdentityKeyPair::generate();
    let peer_store = PeerStore::new();
    admit_peer(
        &peer_store,
        &remote_identity,
        Some("http://8.8.8.8:8422".to_string()),
        now,
    );
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let checks = AtomicUsize::new(0);
    let round = reconcile_record_commitment_witnesses_with_endpoint_policy(
        &storage,
        &peer_store,
        &local_identity,
        &client,
        1,
        |_endpoint| checks.fetch_add(1, Ordering::SeqCst) == 0,
    )
    .await;

    assert_eq!(round.eligible_witnesses, 1);
    assert_eq!(round.attempted, 1);
    assert_eq!(round.verified, 0);
    assert_eq!(round.failed, 1);
    assert!(checks.load(Ordering::SeqCst) >= 2);
}

#[tokio::test]
async fn checkpoint_endpoint_refuses_to_sign_an_unaudited_chain() {
    let now = now_secs();
    let responder_identity = Arc::new(IdentityKeyPair::generate());
    let requester_identity = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &requester_identity, None, now);

    let request_id = [0x91; 16];
    let requester = requester_identity.public_key_bytes();
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        0,
        &GENESIS_PREV_HASH,
        &request_id,
        &requester,
        now,
    );
    let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height: 0,
        known_tip_hash: GENESIS_PREV_HASH,
        request_id,
        requester,
        request_timestamp: now,
        signature: requester_identity.sign(&signing_bytes),
    })
    .unwrap();
    let response = build_memchain_peer_router(storage, peer_store, responder_identity)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/checkpoint")
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn signed_checkpoint_distinguishes_remote_lag_from_divergence() {
    let now = now_secs();
    let responder = IdentityKeyPair::generate();
    let local_writer = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let first = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0x31; 32]],
        &local_writer,
    );
    storage
        .append_record_commitment_block(&first, None)
        .await
        .unwrap();
    let second = RecordCommitmentBlockV1::new_signed(
        2,
        now.saturating_sub(1),
        first.hash(),
        vec![[0x32; 32]],
        &local_writer,
    );
    storage
        .append_record_commitment_block(&second, None)
        .await
        .unwrap();

    let request_id = [0xA2; 16];
    let responder_key = responder.public_key_bytes();
    let lagging_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &request_id,
        &responder_key,
        now,
        1,
        &first.hash(),
        1,
        &first.hash(),
    );
    let lagging_frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        request_id,
        responder: responder_key,
        response_timestamp: now,
        checkpoint_height: 1,
        checkpoint_hash: first.hash(),
        tip_height: 1,
        tip_hash: first.hash(),
        signature: responder.sign(&lagging_signing_bytes),
    })
    .unwrap();
    let lagging = verify_record_commitment_checkpoint(
        &storage,
        &lagging_frame,
        &request_id,
        &responder_key,
        (2, second.hash()),
        now,
    )
    .await
    .unwrap();
    assert_eq!(lagging.relation, CommitmentCheckpointRelation::RemoteBehind);

    let fork_hash = [0xF1; 32];
    let fork_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &request_id,
        &responder_key,
        now,
        2,
        &fork_hash,
        2,
        &fork_hash,
    );
    let fork_frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        request_id,
        responder: responder_key,
        response_timestamp: now,
        checkpoint_height: 2,
        checkpoint_hash: fork_hash,
        tip_height: 2,
        tip_hash: fork_hash,
        signature: responder.sign(&fork_signing_bytes),
    })
    .unwrap();
    let diverged = verify_record_commitment_checkpoint(
        &storage,
        &fork_frame,
        &request_id,
        &responder_key,
        (2, second.hash()),
        now,
    )
    .await
    .unwrap();
    assert_eq!(diverged.relation, CommitmentCheckpointRelation::Diverged);
}

#[tokio::test]
async fn authenticated_range_sync_converges_two_commitment_ledgers() {
    let now = now_secs();
    let responder_identity = Arc::new(IdentityKeyPair::generate());
    let requester_identity = IdentityKeyPair::generate();
    let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let destination = MemoryStorage::open(":memory:", None).unwrap();

    let first = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0x11; 32], [0x22; 32]],
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
        vec![[0x33; 32]],
        &responder_identity,
    );
    source
        .append_record_commitment_block(&second, None)
        .await
        .unwrap();
    source.audit_record_commitment_chain().await.unwrap();
    destination.audit_record_commitment_chain().await.unwrap();

    let peer_store = Arc::new(PeerStore::new());
    let descriptor = NodeDescriptor::new(
        requester_identity.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now.saturating_add(600),
        "memchain-sync-test",
    );
    let descriptor = SignedNodeDescriptor::sign(descriptor, &requester_identity).unwrap();
    let import = peer_store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
        now,
    );
    assert_eq!(import.inserted, 1);

    let request_id = [0xA7; 16];
    let requester = requester_identity.public_key_bytes();
    let signing_bytes = record_block_range_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        1,
        MAX_BLOCKS_PER_RESPONSE_WIRE,
        &request_id,
        &requester,
        now,
    );
    let frame = encode_memchain(&MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height: 1,
        limit: MAX_BLOCKS_PER_RESPONSE_WIRE,
        request_id,
        requester,
        request_timestamp: now,
        signature: requester_identity.sign(&signing_bytes),
    })
    .unwrap();
    let router = build_memchain_peer_router(
        Arc::clone(&source),
        peer_store,
        Arc::clone(&responder_identity),
    );
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/memchain/peer/block-range")
                .header(header::CONTENT_TYPE, "application/octet-stream")
                .body(Body::from(frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    assert_eq!(body.first().copied(), Some(MEMCHAIN_MAGIC));
    let response = decode_memchain(&body[1..]).unwrap();
    let MemChainMessage::RecordBlockRangeResponseV1 {
        request_id: response_request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        panic!("expected record block range response");
    };
    assert_eq!(response_request_id, request_id);
    assert_eq!(responder, responder_identity.public_key_bytes());
    assert!(!has_more);
    assert_eq!(tip_height, 2);
    assert_eq!(tip_hash, second.hash());
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &response_request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .unwrap()
        .verify(&response_signing_bytes, &signature)
        .unwrap();

    for block in &blocks {
        destination
            .append_record_commitment_block(block, Some(&responder))
            .await
            .unwrap();
    }
    assert_eq!(blocks, vec![first, second]);
    assert_eq!(
        destination.record_commitment_chain_tip().await,
        source.record_commitment_chain_tip().await
    );
    let status = destination.record_commitment_chain_status().await;
    assert_eq!(status.block_count, 2);
    assert_eq!(status.commitment_count, 3);
}

#[tokio::test]
async fn coordinator_witness_round_keeps_valid_proof_over_partial_failure() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let converged_witness = Arc::new(IdentityKeyPair::generate());
    let lagging_witness = Arc::new(IdentityKeyPair::generate());
    let unavailable_witness = IdentityKeyPair::generate();
    let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let converged = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let lagging = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0x71; 32]],
        &coordinator,
    );
    local
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    converged
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    local.audit_record_commitment_chain().await.unwrap();
    local
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    converged.audit_record_commitment_chain().await.unwrap();
    lagging.audit_record_commitment_chain().await.unwrap();

    let converged_peers = Arc::new(PeerStore::new());
    admit_peer(&converged_peers, &coordinator, None, now);
    let converged_router = build_memchain_peer_router(
        Arc::clone(&converged),
        converged_peers,
        Arc::clone(&converged_witness),
    );
    let converged_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let converged_address = converged_listener.local_addr().unwrap();
    let converged_server = tokio::spawn(async move {
        axum::serve(converged_listener, converged_router)
            .await
            .unwrap();
    });

    let lagging_peers = Arc::new(PeerStore::new());
    admit_peer(&lagging_peers, &coordinator, None, now);
    let lagging_router = build_memchain_peer_router(
        Arc::clone(&lagging),
        lagging_peers,
        Arc::clone(&lagging_witness),
    );
    let lagging_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let lagging_address = lagging_listener.local_addr().unwrap();
    let lagging_server = tokio::spawn(async move {
        axum::serve(lagging_listener, lagging_router).await.unwrap();
    });

    let coordinator_peers = PeerStore::new();
    admit_peer(
        &coordinator_peers,
        &converged_witness,
        Some(format!("http://{converged_address}")),
        now,
    );
    admit_peer(
        &coordinator_peers,
        &lagging_witness,
        Some(format!("http://{lagging_address}")),
        now,
    );
    admit_peer(
        &coordinator_peers,
        &unavailable_witness,
        Some("https://[invalid".to_string()),
        now,
    );
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let round = reconcile_record_commitment_witnesses_with_endpoint_policy(
        &local,
        &coordinator_peers,
        &coordinator,
        &client,
        3,
        |_| true,
    )
    .await;

    assert_eq!(round.eligible_witnesses, 3);
    assert_eq!(round.attempted, 3);
    assert_eq!(round.verified, 2);
    assert_eq!(round.converged, 1);
    assert_eq!(round.remote_behind, 1);
    assert_eq!(round.remote_ahead, 0);
    assert_eq!(round.diverged, 0);
    assert_eq!(round.failed, 1);
    let status = local.record_commitment_checkpoint_status();
    assert_eq!(status.state, "converged");
    assert_eq!(status.proofs_verified_total, 2);
    assert_eq!(status.proofs_failed_total, 1);
    assert_eq!(status.evidence_records, 2);
    assert_eq!(status.evidence_state, "verified");
    assert_eq!(status.last_round_state, "partial");
    assert_eq!(status.last_round_eligible, 3);
    assert_eq!(status.last_round_attempted, 3);
    assert_eq!(status.last_round_verified, 2);
    assert_eq!(status.last_round_failed, 1);
    assert_eq!(status.last_round_converged, 1);
    assert_eq!(status.last_round_remote_ahead, 0);
    assert_eq!(status.last_round_remote_behind, 1);
    assert_eq!(status.last_round_diverged, 0);
    assert!(status.last_round_at.is_some());

    converged_server.abort();
    lagging_server.abort();
    let _ = converged_server.await;
    let _ = lagging_server.await;
}

#[tokio::test]
async fn pinned_witness_round_excludes_unpinned_permissionless_peers() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let pinned_witness = Arc::new(IdentityKeyPair::generate());
    let unpinned_peer = IdentityKeyPair::generate();
    let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    local.audit_record_commitment_chain().await.unwrap();
    local
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    witness_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let witness_peers = Arc::new(PeerStore::new());
    admit_peer(&witness_peers, &coordinator, None, now);
    let router = build_memchain_peer_router(
        Arc::clone(&witness_storage),
        witness_peers,
        Arc::clone(&pinned_witness),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let coordinator_peers = PeerStore::new();
    admit_peer(
        &coordinator_peers,
        &pinned_witness,
        Some(format!("http://{address}")),
        now,
    );
    admit_peer(
        &coordinator_peers,
        &unpinned_peer,
        Some("https://[invalid".to_string()),
        now,
    );
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        &local,
        &coordinator_peers,
        &coordinator,
        &client,
        &[
            pinned_witness.public_key_bytes(),
            pinned_witness.public_key_bytes(),
        ],
        2,
        |_| true,
    )
    .await;

    assert_eq!(round.eligible_witnesses, 1);
    assert_eq!(round.attempted, 1);
    assert_eq!(round.verified, 1);
    assert_eq!(round.converged, 1);
    assert_eq!(round.failed, 0);
    assert_eq!(round.certificate_signers, 1);
    assert!(!round.certificate_persisted);
    assert_eq!(
        local.record_commitment_checkpoint_status().evidence_records,
        1
    );

    server.abort();
    let _ = server.await;
}

// [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] Reproduces a real
// production outage: strict witness verification and all-witness lease
// acquisition run before gossip can refresh either side after a long
// reboot. Expired descriptors are transport/admission hints only; every
// request and response still verifies against the exact pinned keys.
#[tokio::test]
async fn descriptor_preflight_refreshes_legacy_witness_before_strict_gate() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let witness = IdentityKeyPair::generate();

    let witness_peers = Arc::new(PeerStore::new());
    let expired_coordinator = NodeDescriptor::new(
        coordinator.public_key_bytes(),
        1,
        now.saturating_sub(1_200),
        now.saturating_sub(600),
        "expired-coordinator-before-preflight",
    );
    let expired_coordinator =
        SignedNodeDescriptor::sign(expired_coordinator, &coordinator).unwrap();
    let imported = witness_peers.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now, vec![expired_coordinator]),
        now,
        "test_expired_coordinator_cache",
    );
    assert_eq!(imported.inserted, 1);
    assert!(witness_peers
        .get_valid(&coordinator.public_key_bytes(), now)
        .is_none());

    let router = crate::api::discovery::build_discovery_router(
        Arc::clone(&witness_peers),
        crate::api::discovery::DiscoveryApiPolicy::default(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let mut expired_witness = NodeDescriptor::new(
        witness.public_key_bytes(),
        1,
        now.saturating_sub(1_200),
        now.saturating_sub(600),
        "expired-witness-endpoint-hint",
    );
    expired_witness.public_endpoint = Some(format!("http://{address}"));
    let expired_witness = SignedNodeDescriptor::sign(expired_witness, &witness).unwrap();
    let coordinator_peers = PeerStore::new();
    let imported = coordinator_peers.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now, vec![expired_witness]),
        now,
        "test_expired_witness_cache",
    );
    assert_eq!(imported.inserted, 1);

    let mut current_coordinator = NodeDescriptor::new(
        coordinator.public_key_bytes(),
        2,
        now,
        now.saturating_add(600),
        "current-coordinator-after-endpoint-rotation",
    );
    current_coordinator.public_endpoint = Some("http://8.8.8.8:8422".to_string());
    let current_coordinator =
        SignedNodeDescriptor::sign(current_coordinator, &coordinator).unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();

    let round = publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy(
        &coordinator_peers,
        &current_coordinator,
        &client,
        &[
            witness.public_key_bytes(),
            witness.public_key_bytes(),
            coordinator.public_key_bytes(),
        ],
        &allow_test_endpoint,
    )
    .await;

    assert_eq!(
        round,
        CommitmentWitnessDescriptorPublishRound {
            configured: 1,
            attempted: 1,
            accepted: 1,
            failed: 0,
        }
    );
    let refreshed = witness_peers
        .get_valid(&coordinator.public_key_bytes(), now)
        .unwrap();
    assert_eq!(refreshed.descriptor.sequence, 2);
    assert_eq!(
        refreshed.descriptor.public_endpoint.as_deref(),
        Some("http://8.8.8.8:8422")
    );

    // [WITNESS-DESCRIPTOR-PREFLIGHT 2026-07-29 by Codex] The discovery API
    // intentionally returns a structured 200 response for stale sequence
    // input. Preflight must parse that receipt instead of misreporting the
    // transport-level success as a descriptor refresh.
    let stale_coordinator = SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            coordinator.public_key_bytes(),
            1,
            now,
            now.saturating_add(600),
            "stale-coordinator-after-endpoint-rotation",
        ),
        &coordinator,
    )
    .unwrap();
    let stale_round = publish_current_descriptor_to_commitment_witnesses_with_endpoint_policy(
        &coordinator_peers,
        &stale_coordinator,
        &client,
        &[witness.public_key_bytes()],
        &allow_test_endpoint,
    )
    .await;
    assert_eq!(stale_round.attempted, 1);
    assert_eq!(stale_round.accepted, 0);
    assert_eq!(stale_round.failed, 1);

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn descriptor_preflight_rejects_unsafe_witness_endpoint_without_request() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let witness = IdentityKeyPair::generate();
    let coordinator_peers = PeerStore::new();
    admit_peer(
        &coordinator_peers,
        &witness,
        Some("http://127.0.0.1:8422".to_string()),
        now,
    );
    let current_coordinator = SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            coordinator.public_key_bytes(),
            1,
            now,
            now.saturating_add(600),
            "unsafe-preflight-target-test",
        ),
        &coordinator,
    )
    .unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    let round = publish_current_descriptor_to_commitment_witnesses(
        &coordinator_peers,
        &current_coordinator,
        &client,
        &[witness.public_key_bytes()],
    )
    .await;

    assert_eq!(
        round,
        CommitmentWitnessDescriptorPublishRound {
            configured: 1,
            attempted: 0,
            accepted: 0,
            failed: 1,
        }
    );
}

#[tokio::test]
async fn pinned_witness_round_recovers_through_authentic_expired_cache_descriptor() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let pinned_witness = Arc::new(IdentityKeyPair::generate());
    let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let witness_storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    local.audit_record_commitment_chain().await.unwrap();
    local
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    witness_storage
        .audit_record_commitment_chain()
        .await
        .unwrap();

    let witness_peers = Arc::new(PeerStore::new());
    let expired_coordinator = NodeDescriptor::new(
        coordinator.public_key_bytes(),
        1,
        now.saturating_sub(1_200),
        now.saturating_sub(600),
        "expired-pinned-coordinator-test",
    );
    let expired_coordinator =
        SignedNodeDescriptor::sign(expired_coordinator, &coordinator).unwrap();
    let imported = witness_peers.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now, vec![expired_coordinator]),
        now,
        "test_expired_coordinator_cache",
    );
    assert_eq!(imported.inserted, 1);
    assert!(witness_peers
        .get_valid(&coordinator.public_key_bytes(), now)
        .is_none());
    let router = build_memchain_peer_router_with_coordinator_lease(
        Arc::clone(&witness_storage),
        witness_peers,
        Arc::clone(&pinned_witness),
        Some(coordinator.public_key_bytes()),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let mut expired = NodeDescriptor::new(
        pinned_witness.public_key_bytes(),
        1,
        now.saturating_sub(1_200),
        now.saturating_sub(600),
        "expired-pinned-witness-test",
    );
    expired.public_endpoint = Some(format!("http://{address}"));
    expired.capabilities = vec![NodeCapability::EncryptedStorage];
    let expired = SignedNodeDescriptor::sign(expired, &pinned_witness).unwrap();
    let coordinator_peers = PeerStore::new();
    let imported = coordinator_peers.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now, vec![expired]),
        now,
        "test_expired_cache",
    );
    assert_eq!(imported.inserted, 1);
    assert!(coordinator_peers
        .get_valid(&pinned_witness.public_key_bytes(), now)
        .is_none());
    assert!(commitment_peer_descriptor(
        &coordinator_peers,
        &pinned_witness.public_key_bytes(),
        now,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
    )
    .is_none());
    assert!(commitment_peer_descriptor(
        &coordinator_peers,
        &pinned_witness.public_key_bytes(),
        now,
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
    )
    .is_some());

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        &local,
        &coordinator_peers,
        &coordinator,
        &client,
        &[pinned_witness.public_key_bytes()],
        1,
        |_| true,
    )
    .await;

    assert_eq!(round.eligible_witnesses, 1);
    assert_eq!(round.attempted, 1);
    assert_eq!(round.verified, 1);
    assert_eq!(round.converged, 1);
    assert_eq!(round.failed, 0);
    assert_eq!(
        local.record_commitment_checkpoint_status().evidence_records,
        1
    );

    let instance_id = [0x6a; 32];
    let lease = request_record_commitment_coordinator_lease_with_endpoint_policy(
        &local,
        &coordinator_peers,
        &coordinator,
        &pinned_witness.public_key_bytes(),
        &instance_id,
        MIN_COORDINATOR_LEASE_TTL_SECS_V1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert!(lease.lease_epoch > 0);
    assert!(lease.valid_for_secs > 0);

    let released = release_record_commitment_coordinator_lease_with_endpoint_policy(
        &coordinator_peers,
        &coordinator,
        &pinned_witness.public_key_bytes(),
        &instance_id,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(released.lease_epoch, lease.lease_epoch);
    assert!(released.released_at >= now);

    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn pinned_witness_round_persists_two_signer_checkpoint_certificate() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let witnesses = [
        Arc::new(IdentityKeyPair::generate()),
        Arc::new(IdentityKeyPair::generate()),
    ];
    let local = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let witness_storages = [
        Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
        Arc::new(MemoryStorage::open(":memory:", None).unwrap()),
    ];
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0x91; 32]],
        &coordinator,
    );
    local
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    local.audit_record_commitment_chain().await.unwrap();
    local
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();

    let mut addresses = Vec::new();
    let mut servers = Vec::new();
    for (storage, witness) in witness_storages.iter().zip(witnesses.iter()) {
        storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        let peers = Arc::new(PeerStore::new());
        admit_peer(&peers, &coordinator, None, now);
        let router = build_memchain_peer_router(Arc::clone(storage), peers, Arc::clone(witness));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        addresses.push(listener.local_addr().unwrap());
        servers.push(tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        }));
    }

    let coordinator_peers = PeerStore::new();
    for (witness, address) in witnesses.iter().zip(addresses.iter()) {
        admit_peer(
            &coordinator_peers,
            witness,
            Some(format!("http://{address}")),
            now,
        );
    }
    let witness_ids = [
        witnesses[0].public_key_bytes(),
        witnesses[1].public_key_bytes(),
    ];
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let round = reconcile_record_commitment_pinned_witnesses_with_endpoint_policy(
        &local,
        &coordinator_peers,
        &coordinator,
        &client,
        &witness_ids,
        2,
        |_| true,
    )
    .await;

    assert_eq!(round.verified, 2);
    assert_eq!(round.converged, 2);
    assert_eq!(round.certificate_signers, 2);
    assert_eq!(round.certificate_required_signers, 2);
    assert!(round.certificate_persisted);
    assert!(!round.certificate_persistence_failed);
    let status = local.record_commitment_checkpoint_status();
    assert_eq!(status.checkpoint_certificates, 1);
    assert_eq!(status.latest_certified_height, Some(1));
    assert_eq!(status.latest_certificate_signers, 2);

    let destination_identity = IdentityKeyPair::generate();
    let destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    destination
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    destination.configure_record_commitment_sync(false, true);
    destination.configure_record_commitment_certificate_policy(0, 1);

    let source_identity = Arc::new(coordinator);
    let source_peers = Arc::new(PeerStore::new());
    admit_peer(&source_peers, &destination_identity, None, now);
    let source_router = build_memchain_peer_router(
        Arc::clone(&local),
        source_peers,
        Arc::clone(&source_identity),
    );
    let source_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let source_address = source_listener.local_addr().unwrap();
    let source_server = tokio::spawn(async move {
        axum::serve(source_listener, source_router).await.unwrap();
    });

    let destination_peers = PeerStore::new();
    admit_peer(
        &destination_peers,
        &source_identity,
        Some(format!("http://{source_address}")),
        now,
    );
    let disabled = sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
        &destination,
        &destination_peers,
        &destination_identity,
        &source_identity.public_key_bytes(),
        &witness_ids,
        1,
        1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(
        disabled,
        CommitmentFollowerCertificateSyncOutcome::PolicyDisabled
    );
    let disabled_status = destination.record_commitment_sync_status();
    assert_eq!(disabled_status.certificate_policy_state, "disabled");
    assert!(!disabled_status.certificate_policy_ready);
    assert!(disabled_status
        .certificate_policy_last_evaluated_at
        .is_some());

    destination.configure_record_commitment_certificate_policy(2, 2);
    let imported = sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
        &destination,
        &destination_peers,
        &destination_identity,
        &source_identity.public_key_bytes(),
        &witness_ids,
        2,
        1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    let CommitmentFollowerCertificateSyncOutcome::Refreshed(imported) = imported else {
        panic!("uncertified converged follower must import current certificate");
    };
    assert_eq!(imported.checkpoint_height, 1);
    assert_eq!(imported.signer_count, 2);
    assert_eq!(imported.required_signers, 2);
    assert!(imported.persisted);
    assert_eq!(
        destination
            .record_commitment_checkpoint_status()
            .checkpoint_certificates,
        1
    );
    let imported_status = destination.record_commitment_sync_status();
    assert_eq!(imported_status.certificate_policy_state, "ready");
    assert!(imported_status.certificate_policy_ready);
    assert_eq!(imported_status.certificate_witnesses_configured, 2);
    assert_eq!(imported_status.certificate_minimum_signers, 2);
    assert_eq!(imported_status.certificate_sync_rounds_total, 1);
    assert_eq!(imported_status.certificate_coordinator_success_total, 1);
    assert_eq!(imported_status.certificate_verified_unpersisted_total, 0);
    assert_eq!(
        imported_status.certificate_policy_evaluated_tip_height,
        Some(1)
    );
    assert!(imported_status
        .certificate_policy_last_evaluated_at
        .is_some());

    let replacement_witness = IdentityKeyPair::generate().public_key_bytes();
    destination.configure_record_commitment_certificate_policy(2, 2);
    let rotated_policy_error =
        sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &[witness_ids[0], replacement_witness],
            2,
            1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
    assert_eq!(
        rotated_policy_error, "certificate_member_not_pinned",
        "a same-height certificate under retired pins must not be current"
    );
    let rotated_status = destination.record_commitment_sync_status();
    assert_eq!(rotated_status.certificate_policy_state, "security_stopped");
    assert!(!rotated_status.certificate_policy_ready);

    let third_witness = IdentityKeyPair::generate().public_key_bytes();
    let strict_witness_ids = [witness_ids[0], witness_ids[1], third_witness];
    let error = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
        &destination,
        &destination_peers,
        &destination_identity,
        &source_identity.public_key_bytes(),
        &strict_witness_ids,
        3,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "certificate_threshold_below_policy");

    let unpinned_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    unpinned_destination
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    unpinned_destination
        .audit_record_commitment_chain()
        .await
        .unwrap();
    unpinned_destination
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    let error = pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
        &unpinned_destination,
        &destination_peers,
        &destination_identity,
        &source_identity.public_key_bytes(),
        &[witness_ids[0], replacement_witness],
        2,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "certificate_member_not_pinned");

    for server in servers {
        server.abort();
        let _ = server.await;
    }
    source_server.abort();
    let _ = source_server.await;

    // [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex] A witness serves
    // only as a read-only carrier for an already audited certificate. The
    // receiver still validates the embedded coordinator checkpoint,
    // distinct witness frames, local pins, threshold, and exact local tip.
    let carrier_destination_identity = IdentityKeyPair::generate();
    let guarded_destination_identity = IdentityKeyPair::generate();
    let carrier_identity = Arc::clone(&witnesses[0]);
    let carrier_peers = Arc::new(PeerStore::new());
    admit_peer(&carrier_peers, &carrier_destination_identity, None, now);
    admit_peer(&carrier_peers, &guarded_destination_identity, None, now);
    assert!(carrier_peers
        .get_valid(&carrier_destination_identity.public_key_bytes(), now_secs())
        .is_some());
    let carrier_router = build_memchain_peer_router(
        Arc::clone(&local),
        Arc::clone(&carrier_peers),
        Arc::clone(&carrier_identity),
    );
    let carrier_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let carrier_address = carrier_listener.local_addr().unwrap();
    let carrier_server = tokio::spawn(async move {
        axum::serve(carrier_listener, carrier_router).await.unwrap();
    });

    let carrier_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    carrier_destination
        .audit_record_commitment_chain()
        .await
        .unwrap();
    carrier_destination
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    carrier_destination.configure_record_commitment_sync(false, true);
    carrier_destination.configure_record_commitment_certificate_policy(2, 2);
    let carrier_destination_peers = PeerStore::new();
    let unavailable_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable_address = unavailable_listener.local_addr().unwrap();
    drop(unavailable_listener);
    admit_peer(
        &carrier_destination_peers,
        &source_identity,
        Some(format!("http://{unavailable_address}")),
        now,
    );
    admit_peer(
        &carrier_destination_peers,
        &carrier_identity,
        Some(format!("http://{carrier_address}")),
        now,
    );
    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] The coordinator is
    // unreachable and the destination starts at genesis. The pinned
    // witness signs only the page envelope; the imported block must retain
    // the unavailable coordinator as proposer.
    let recovered_page = pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy(
        &carrier_destination,
        &carrier_destination_peers,
        &carrier_destination_identity,
        &source_identity.public_key_bytes(),
        &witness_ids,
        2,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    assert_eq!(
        recovered_page.source,
        CommitmentSyncPageSource::PinnedCarrier
    );
    assert_eq!(recovered_page.carrier_attempts, 1);
    assert_eq!(recovered_page.page.inserted, 1);
    assert!(!recovered_page.page.has_more);
    assert_eq!(recovered_page.page.remote_tip_height, 1);
    assert_eq!(
        carrier_destination.record_commitment_chain_tip().await,
        local.record_commitment_chain_tip().await
    );
    let block_carrier_status = carrier_destination.record_commitment_sync_status();
    assert_eq!(block_carrier_status.block_page_pulls_total, 1);
    assert_eq!(block_carrier_status.block_page_coordinator_success_total, 0);
    assert_eq!(block_carrier_status.block_carrier_attempts_total, 1);
    assert_eq!(block_carrier_status.block_carrier_recoveries_total, 1);
    assert_eq!(
        block_carrier_status.block_page_availability_exhausted_total,
        0
    );
    assert_eq!(block_carrier_status.block_page_security_stops_total, 0);
    assert_eq!(
        block_carrier_status.last_block_page_pull_result.as_deref(),
        Some("carrier_recovered")
    );
    assert!(block_carrier_status
        .last_block_carrier_recovered_at
        .is_some());
    let recovered = sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
        &carrier_destination,
        &carrier_destination_peers,
        &carrier_destination_identity,
        &source_identity.public_key_bytes(),
        &witness_ids,
        2,
        1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap();
    let CommitmentFollowerCertificateSyncOutcome::Refreshed(recovered) = recovered else {
        panic!("pinned witness carrier must recover coordinator certificate availability");
    };
    assert!(recovered.persisted);
    assert_eq!(recovered.checkpoint_height, 1);
    assert_eq!(recovered.signer_count, 2);
    let carrier_status = carrier_destination.record_commitment_sync_status();
    assert_eq!(carrier_status.certificate_sync_rounds_total, 1);
    assert_eq!(carrier_status.certificate_coordinator_success_total, 0);
    assert_eq!(carrier_status.certificate_carrier_attempts_total, 1);
    assert_eq!(carrier_status.certificate_carrier_recoveries_total, 1);
    assert_eq!(carrier_status.certificate_verified_unpersisted_total, 0);
    assert_eq!(carrier_status.certificate_availability_exhausted_total, 0);
    assert_eq!(carrier_status.certificate_security_stops_total, 0);
    assert_eq!(
        carrier_status.last_certificate_sync_result.as_deref(),
        Some("carrier_recovered")
    );
    assert!(carrier_status
        .last_certificate_carrier_recovered_at
        .is_some());
    assert_eq!(carrier_status.certificate_policy_state, "ready");
    assert!(carrier_status.certificate_policy_ready);
    assert_eq!(
        carrier_status.certificate_policy_evaluated_tip_height,
        Some(1)
    );

    // A malformed primary response is a security failure, not an
    // availability event. The valid carrier must not mask it.
    let malformed_router = Router::new().route(
        "/api/memchain/peer/checkpoint-certificate",
        post(|| async { (StatusCode::OK, vec![0u8]) }),
    );
    let malformed_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let malformed_address = malformed_listener.local_addr().unwrap();
    let malformed_server = tokio::spawn(async move {
        axum::serve(malformed_listener, malformed_router)
            .await
            .unwrap();
    });
    let guarded_destination = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    guarded_destination
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    guarded_destination
        .audit_record_commitment_chain()
        .await
        .unwrap();
    guarded_destination
        .audit_record_commitment_checkpoint_evidence()
        .await
        .unwrap();
    guarded_destination.configure_record_commitment_sync(false, true);
    guarded_destination.configure_record_commitment_certificate_policy(2, 2);
    let guarded_destination_peers = PeerStore::new();
    admit_peer(
        &guarded_destination_peers,
        &source_identity,
        Some(format!("http://{malformed_address}")),
        now,
    );
    admit_peer(
        &guarded_destination_peers,
        &carrier_identity,
        Some(format!("http://{carrier_address}")),
        now,
    );
    let guarded_error =
        sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &guarded_destination,
            &guarded_destination_peers,
            &guarded_destination_identity,
            &source_identity.public_key_bytes(),
            &witness_ids,
            2,
            1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap_err();
    assert_eq!(guarded_error, "invalid_certificate_frame");
    let guarded_status = guarded_destination.record_commitment_sync_status();
    assert_eq!(guarded_status.certificate_sync_rounds_total, 1);
    assert_eq!(guarded_status.certificate_carrier_attempts_total, 0);
    assert_eq!(guarded_status.certificate_carrier_recoveries_total, 0);
    assert_eq!(guarded_status.certificate_verified_unpersisted_total, 0);
    assert_eq!(guarded_status.certificate_security_stops_total, 1);
    assert_eq!(
        guarded_status.last_certificate_sync_result.as_deref(),
        Some("security_stopped")
    );
    assert_eq!(guarded_status.certificate_policy_state, "security_stopped");
    assert!(!guarded_status.certificate_policy_ready);

    malformed_server.abort();
    let _ = malformed_server.await;
    carrier_server.abort();
    let _ = carrier_server.await;

    destination.configure_record_commitment_certificate_policy(2, 2);
    let already_current =
        sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy(
            &destination,
            &destination_peers,
            &destination_identity,
            &source_identity.public_key_bytes(),
            &witness_ids,
            2,
            1,
            &client,
            &allow_test_endpoint,
        )
        .await
        .unwrap();
    assert_eq!(
        already_current,
        CommitmentFollowerCertificateSyncOutcome::AlreadyCurrent
    );
    let already_current_status = destination.record_commitment_sync_status();
    assert_eq!(already_current_status.certificate_policy_state, "ready");
    assert!(already_current_status.certificate_policy_ready);
    assert_eq!(
        already_current_status.certificate_policy_evaluated_tip_height,
        Some(1)
    );
}
