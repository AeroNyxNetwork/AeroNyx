// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn certificate_carrier_fallback_classifies_only_availability_failures() {
    // [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex] Admission and
    // transport outages may advance to the next exact operator pin.
    // Authentication and evidence-integrity failures must remain terminal.
    for error in [
        "certificate_source_unavailable",
        "certificate_source_missing_endpoint",
        "certificate_request_timeout",
        "certificate_request_connect",
        "response_body_body",
        "certificate_http_status_403",
        "certificate_http_status_404",
        "certificate_http_status_429",
        "certificate_http_status_503",
    ] {
        assert_eq!(
            commitment_certificate_source_failure_class(error),
            CommitmentCertificateSourceFailureClass::Availability,
            "{error} must permit only bounded pinned-carrier recovery"
        );
    }
    for error in [
        "certificate_source_unsafe_endpoint",
        "certificate_http_status_400",
        "certificate_http_status_401",
        "invalid_certificate_frame",
        "invalid_certificate_response_signature",
        "certificate_local_tip_mismatch",
        "certificate_member_not_pinned",
        "certificate_digest_mismatch",
        "certificate_persist_failed",
    ] {
        assert_eq!(
            commitment_certificate_source_failure_class(error),
            CommitmentCertificateSourceFailureClass::Security,
            "{error} must stop before any fallback"
        );
    }
}

#[test]
fn block_carrier_fallback_classifies_only_availability_failures() {
    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Only source absence,
    // transient transport/status failure, or a safely ignored stale
    // carrier may advance to another exact pin. Evidence failures stop.
    for error in [
        "pinned_coordinator_unavailable",
        "pinned_coordinator_missing_endpoint",
        "request_timeout",
        "request_connect",
        "response_body_body",
        "http_status_403",
        "http_status_404",
        "http_status_429",
        "http_status_503",
        "carrier_tip_behind",
    ] {
        assert_eq!(
            commitment_block_source_failure_class(error),
            CommitmentBlockSourceFailureClass::Availability,
            "{error} must permit only bounded pinned-carrier recovery"
        );
    }
    for error in [
        "pinned_coordinator_unsafe_endpoint",
        "http_status_400",
        "http_status_401",
        "invalid_response_frame",
        "invalid_response_signature",
        "response_responder_mismatch",
        "unexpected_block_proposer",
        "commitment_chain_verification_failed",
        "storage_append_rejected",
    ] {
        assert_eq!(
            commitment_block_source_failure_class(error),
            CommitmentBlockSourceFailureClass::Security,
            "{error} must stop before another carrier"
        );
    }
}

#[tokio::test]
async fn block_carrier_policy_requires_distinct_external_pins() {
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    storage.configure_record_commitment_sync(false, true);
    let peers = PeerStore::new();
    let follower = IdentityKeyPair::generate();
    let coordinator = IdentityKeyPair::generate().public_key_bytes();
    let witness = IdentityKeyPair::generate().public_key_bytes();
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Repeating one pin or
    // including self cannot satisfy a two-witness recovery policy.
    let error = pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy(
        &storage,
        &peers,
        &follower,
        &coordinator,
        &[witness, witness, follower.public_key_bytes()],
        2,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "block_carrier_policy_invalid");

    // Backward-compatible policy keeps the historical direct-only failure
    // instead of silently enabling carrier transport.
    let error = pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy(
        &storage,
        &peers,
        &follower,
        &coordinator,
        &[witness],
        1,
        &client,
        &allow_test_endpoint,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "pinned_coordinator_unavailable");
    let status = storage.record_commitment_sync_status();
    assert_eq!(status.block_page_pulls_total, 2);
    assert_eq!(status.block_page_coordinator_success_total, 0);
    assert_eq!(status.block_carrier_attempts_total, 0);
    assert_eq!(status.block_carrier_recoveries_total, 0);
    assert_eq!(status.block_page_availability_exhausted_total, 1);
    assert_eq!(status.block_page_security_stops_total, 1);
    assert_eq!(
        status.last_block_page_pull_result.as_deref(),
        Some("availability_exhausted")
    );
}

#[tokio::test]
async fn multi_page_carrier_cursor_prefers_verified_source_and_hands_off() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let unavailable_carrier = IdentityKeyPair::generate();
    let first_live_carrier = Arc::new(IdentityKeyPair::generate());
    let second_live_carrier = Arc::new(IdentityKeyPair::generate());
    let follower = IdentityKeyPair::generate();
    let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());

    let mut previous_hash = GENESIS_PREV_HASH;
    for height in 1..=u64::try_from(MAX_BLOCKS_PER_RESPONSE + 1).unwrap() {
        let marker = u8::try_from(height).unwrap();
        let block = RecordCommitmentBlockV1::new_signed(
            height,
            now.saturating_sub(32).saturating_add(height),
            previous_hash,
            vec![[marker; 32]],
            &coordinator,
        );
        previous_hash = block.hash();
        source
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
    }
    source.audit_record_commitment_chain().await.unwrap();

    let first_peers = Arc::new(PeerStore::new());
    admit_peer(&first_peers, &follower, None, now);
    let first_router = build_memchain_peer_router(
        Arc::clone(&source),
        first_peers,
        Arc::clone(&first_live_carrier),
    );
    let first_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let first_address = first_listener.local_addr().unwrap();
    let first_server = tokio::spawn(async move {
        axum::serve(first_listener, first_router).await.unwrap();
    });

    let second_peers = Arc::new(PeerStore::new());
    admit_peer(&second_peers, &follower, None, now);
    let second_router = build_memchain_peer_router(
        Arc::clone(&source),
        second_peers,
        Arc::clone(&second_live_carrier),
    );
    let second_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let second_address = second_listener.local_addr().unwrap();
    let second_server = tokio::spawn(async move {
        axum::serve(second_listener, second_router).await.unwrap();
    });

    // Allocate dedicated closed ports after both live endpoints are fixed
    // so neither unavailable descriptor aliases a live test server.
    let coordinator_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let coordinator_address = coordinator_listener.local_addr().unwrap();
    let unavailable_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable_address = unavailable_listener.local_addr().unwrap();
    drop(coordinator_listener);
    drop(unavailable_listener);

    let destination = MemoryStorage::open(":memory:", None).unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination.configure_record_commitment_sync(false, true);
    destination.configure_record_commitment_certificate_policy(3, 2);
    let destination_peers = PeerStore::new();
    admit_peer(
        &destination_peers,
        &coordinator,
        Some(format!("http://{coordinator_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &unavailable_carrier,
        Some(format!("http://{unavailable_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &first_live_carrier,
        Some(format!("http://{first_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &second_live_carrier,
        Some(format!("http://{second_address}")),
        now,
    );
    let carrier_ids = [
        unavailable_carrier.public_key_bytes(),
        first_live_carrier.public_key_bytes(),
        second_live_carrier.public_key_bytes(),
    ];
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let mut cursor = CommitmentBlockCarrierCursor::default();

    // [MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex] Page one
    // bypasses an unavailable first pin and establishes the second pin as
    // the round-local preferred carrier.
    let first_page = pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &coordinator.public_key_bytes(),
        &carrier_ids,
        2,
        &client,
        &allow_test_endpoint,
        &mut cursor,
    )
    .await
    .unwrap();
    assert_eq!(first_page.source, CommitmentSyncPageSource::PinnedCarrier);
    assert_eq!(first_page.carrier_attempts, 2);
    assert_eq!(first_page.page.inserted, MAX_BLOCKS_PER_RESPONSE);
    assert!(first_page.page.has_more);
    assert_eq!(cursor.next_index, 1);

    first_server.abort();
    let _ = first_server.await;
    // Axum may leave an accepted keep-alive connection alive after the
    // listener task is aborted. A fresh pool models a process/network
    // outage by requiring a new connection to the now-closed endpoint.
    let failover_client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    // The preferred carrier disappears between pages. The cursor starts
    // there, then hands off directly to the next exact pin without
    // retrying the earlier unavailable pin.
    let second_page = pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &coordinator.public_key_bytes(),
        &carrier_ids,
        2,
        &failover_client,
        &allow_test_endpoint,
        &mut cursor,
    )
    .await
    .unwrap();
    assert_eq!(second_page.source, CommitmentSyncPageSource::PinnedCarrier);
    assert_eq!(second_page.carrier_attempts, 2);
    assert_eq!(second_page.page.inserted, 1);
    assert!(!second_page.page.has_more);
    assert_eq!(cursor.next_index, 2);
    assert_eq!(
        destination.record_commitment_chain_tip().await,
        source.record_commitment_chain_tip().await
    );
    destination.audit_record_commitment_chain().await.unwrap();

    let status = destination.record_commitment_sync_status();
    assert_eq!(status.block_page_pulls_total, 2);
    assert_eq!(status.block_carrier_attempts_total, 4);
    assert_eq!(status.block_carrier_recoveries_total, 2);
    assert_eq!(status.block_page_security_stops_total, 0);

    second_server.abort();
    let _ = second_server.await;
}

#[tokio::test]
async fn carrier_cursor_never_masks_a_security_failure_with_the_next_pin() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let malformed_carrier = IdentityKeyPair::generate();
    let valid_carrier = Arc::new(IdentityKeyPair::generate());
    let follower = IdentityKeyPair::generate();
    let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0x73; 32]],
        &coordinator,
    );
    source
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    source.audit_record_commitment_chain().await.unwrap();

    let malformed_router = Router::new().route(
        "/api/memchain/peer/block-range",
        post(|| async { (StatusCode::OK, vec![0u8]) }),
    );
    let malformed_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let malformed_address = malformed_listener.local_addr().unwrap();
    let malformed_server = tokio::spawn(async move {
        axum::serve(malformed_listener, malformed_router)
            .await
            .unwrap();
    });

    let valid_peers = Arc::new(PeerStore::new());
    admit_peer(&valid_peers, &follower, None, now);
    let valid_router =
        build_memchain_peer_router(Arc::clone(&source), valid_peers, Arc::clone(&valid_carrier));
    let valid_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let valid_address = valid_listener.local_addr().unwrap();
    let valid_server = tokio::spawn(async move {
        axum::serve(valid_listener, valid_router).await.unwrap();
    });

    let coordinator_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let coordinator_address = coordinator_listener.local_addr().unwrap();
    drop(coordinator_listener);

    let destination = MemoryStorage::open(":memory:", None).unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination.configure_record_commitment_sync(false, true);
    destination.configure_record_commitment_certificate_policy(2, 2);
    let destination_peers = PeerStore::new();
    admit_peer(
        &destination_peers,
        &coordinator,
        Some(format!("http://{coordinator_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &malformed_carrier,
        Some(format!("http://{malformed_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &valid_carrier,
        Some(format!("http://{valid_address}")),
        now,
    );
    let carrier_ids = [
        malformed_carrier.public_key_bytes(),
        valid_carrier.public_key_bytes(),
    ];
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let mut cursor = CommitmentBlockCarrierCursor::default();

    // [MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex] Rotation is
    // availability-only. A malformed preferred carrier must stop before a
    // valid later pin can hide the security incident.
    let error = pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &coordinator.public_key_bytes(),
        &carrier_ids,
        2,
        &client,
        &allow_test_endpoint,
        &mut cursor,
    )
    .await
    .unwrap_err();
    assert_eq!(error, "invalid_response_frame");
    assert_eq!(
        destination.record_commitment_chain_tip().await,
        (0, GENESIS_PREV_HASH)
    );
    let status = destination.record_commitment_sync_status();
    assert_eq!(status.block_page_pulls_total, 1);
    assert_eq!(status.block_carrier_attempts_total, 1);
    assert_eq!(status.block_carrier_recoveries_total, 0);
    assert_eq!(status.block_page_security_stops_total, 1);

    malformed_server.abort();
    let _ = malformed_server.await;
    valid_server.abort();
    let _ = valid_server.await;
}

#[test]
fn carrier_circuit_breaker_uses_half_open_recovery_without_identity_state() {
    let started_at = Instant::now();
    let mut circuit_breaker = CommitmentBlockCarrierCircuitBreaker::default();
    circuit_breaker.align_slots(2);

    // [BLOCK-CARRIER-CIRCUIT-BREAKER 2026-07-29 by Codex] Two consecutive
    // availability failures open the fixed slot. The first retry after the
    // monotonic cooldown is half-open; another availability failure
    // immediately reopens it, while a verified success fully resets it.
    assert_eq!(
        circuit_breaker.decision(0, started_at),
        CommitmentCarrierCircuitDecision::Closed
    );
    circuit_breaker.record_availability_failure(0, started_at);
    assert_eq!(
        circuit_breaker.decision(0, started_at),
        CommitmentCarrierCircuitDecision::Closed
    );
    circuit_breaker.record_availability_failure(0, started_at);
    assert_eq!(
        circuit_breaker.decision(
            0,
            started_at + PINNED_CARRIER_RECOVERY_COOLDOWN - Duration::from_secs(1)
        ),
        CommitmentCarrierCircuitDecision::Cooling
    );

    let half_open_at = started_at + PINNED_CARRIER_RECOVERY_COOLDOWN;
    assert_eq!(
        circuit_breaker.decision(0, half_open_at),
        CommitmentCarrierCircuitDecision::HalfOpen
    );
    circuit_breaker.record_availability_failure(0, half_open_at);
    assert_eq!(
        circuit_breaker.decision(0, half_open_at),
        CommitmentCarrierCircuitDecision::Cooling
    );
    assert_eq!(
        circuit_breaker.decision(0, half_open_at + PINNED_CARRIER_RECOVERY_COOLDOWN),
        CommitmentCarrierCircuitDecision::HalfOpen
    );

    circuit_breaker.record_success(0);
    assert_eq!(
        circuit_breaker.decision(0, half_open_at),
        CommitmentCarrierCircuitDecision::Closed
    );

    circuit_breaker.record_availability_failure(0, half_open_at);
    circuit_breaker.record_availability_failure(0, half_open_at);
    assert_eq!(
        circuit_breaker.decision(0, half_open_at),
        CommitmentCarrierCircuitDecision::Cooling
    );
    circuit_breaker.align_slots(1);
    assert_eq!(
        circuit_breaker.decision(0, half_open_at),
        CommitmentCarrierCircuitDecision::Closed
    );
}

#[tokio::test]
async fn certificate_carrier_circuit_skips_repeated_outages_across_rounds() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let first_carrier = IdentityKeyPair::generate();
    let second_carrier = IdentityKeyPair::generate();
    let follower = IdentityKeyPair::generate();
    let destination = MemoryStorage::open(":memory:", None).unwrap();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0x75; 32]],
        &coordinator,
    );
    destination
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination.configure_record_commitment_sync(false, true);
    destination.configure_record_commitment_certificate_policy(2, 2);

    let peer_store = PeerStore::new();
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("test client");
    let carrier_ids = [
        first_carrier.public_key_bytes(),
        second_carrier.public_key_bytes(),
    ];
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();

    // [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] The coordinator is
    // still attempted every round. Each missing pinned carrier is attempted
    // twice, then its anonymous slot cools and the third round avoids both
    // requests without changing certificate policy or trust.
    for _ in 0..3 {
        let error =
            sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy(
                &destination,
                &peer_store,
                &follower,
                &coordinator.public_key_bytes(),
                &carrier_ids,
                2,
                1,
                &client,
                &allow_test_endpoint,
                &mut circuit_breaker,
            )
            .await
            .unwrap_err();
        assert_eq!(error, "certificate_source_unavailable");
    }

    assert_eq!(
        circuit_breaker.decision(0, Instant::now()),
        CommitmentCarrierCircuitDecision::Cooling
    );
    assert_eq!(
        circuit_breaker.decision(1, Instant::now()),
        CommitmentCarrierCircuitDecision::Cooling
    );
    let status = destination.record_commitment_sync_status();
    assert_eq!(status.certificate_sync_rounds_total, 3);
    assert_eq!(status.certificate_carrier_attempts_total, 4);
    assert_eq!(status.certificate_availability_exhausted_total, 3);
    assert_eq!(status.certificate_security_stops_total, 0);
    assert_eq!(status.certificate_carrier_cooling_slots, 2);
    assert_eq!(status.certificate_carrier_cooldown_skips_total, 2);
    assert_eq!(status.certificate_carrier_half_open_attempts_total, 0);
}

#[tokio::test]
async fn coordinator_certificate_recovery_stops_before_later_carrier_on_security_error() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let malformed_carrier = IdentityKeyPair::generate();
    let later_carrier = IdentityKeyPair::generate();
    let destination = MemoryStorage::open(":memory:", None).unwrap();
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0x76; 32]],
        &coordinator,
    );
    destination
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination.configure_record_commitment_certificate_policy(2, 2);

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

    let later_hits = Arc::new(AtomicUsize::new(0));
    let handler_hits = Arc::clone(&later_hits);
    let later_router = Router::new().route(
        "/api/memchain/peer/checkpoint-certificate",
        post(move || {
            let hits = Arc::clone(&handler_hits);
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                StatusCode::SERVICE_UNAVAILABLE
            }
        }),
    );
    let later_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let later_address = later_listener.local_addr().unwrap();
    let later_server = tokio::spawn(async move {
        axum::serve(later_listener, later_router).await.unwrap();
    });

    let peer_store = PeerStore::new();
    admit_peer(
        &peer_store,
        &malformed_carrier,
        Some(format!("http://{malformed_address}")),
        now,
    );
    admit_peer(
        &peer_store,
        &later_carrier,
        Some(format!("http://{later_address}")),
        now,
    );
    let carrier_ids = [
        malformed_carrier.public_key_bytes(),
        later_carrier.public_key_bytes(),
    ];
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("test client");
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();

    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] A later healthy
    // or merely responsive carrier must never hide an earlier malformed
    // signed-protocol response from an exact operator pin.
    let recovery =
        recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
            &destination,
            &peer_store,
            &coordinator,
            &carrier_ids,
            2,
            2,
            &client,
            &allow_test_endpoint,
            &mut circuit_breaker,
        )
        .await;
    assert_eq!(
        recovery.disposition,
        CommitmentCertificateCarrierRecoveryDisposition::SecurityStopped
    );
    assert_eq!(recovery.carrier_attempts, 1);
    assert_eq!(recovery.cooldown_skips, 0);
    assert_eq!(recovery.half_open_attempts, 0);
    assert_eq!(recovery.cooling_slots, 0);
    assert_eq!(later_hits.load(Ordering::SeqCst), 0);

    malformed_server.abort();
    let _ = malformed_server.await;
    later_server.abort();
    let _ = later_server.await;
}

#[tokio::test]
async fn coordinator_certificate_recovery_cools_repeatedly_unavailable_carriers() {
    let coordinator = IdentityKeyPair::generate();
    let first_carrier = IdentityKeyPair::generate();
    let second_carrier = IdentityKeyPair::generate();
    let destination = MemoryStorage::open(":memory:", None).unwrap();
    let peer_store = PeerStore::new();
    let carrier_ids = [
        first_carrier.public_key_bytes(),
        second_carrier.public_key_bytes(),
    ];
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("test client");
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();

    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Coordinator
    // backfill retains only anonymous slot health. Two failed rounds open
    // both circuits; the third performs no transport attempt.
    for expected_attempts in [2, 2, 0] {
        let recovery =
            recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
                &destination,
                &peer_store,
                &coordinator,
                &carrier_ids,
                2,
                2,
                &client,
                &allow_test_endpoint,
                &mut circuit_breaker,
            )
            .await;
        assert_eq!(
            recovery.disposition,
            CommitmentCertificateCarrierRecoveryDisposition::AvailabilityExhausted
        );
        assert_eq!(recovery.carrier_attempts, expected_attempts);
    }
    assert_eq!(
        circuit_breaker.decision(0, Instant::now()),
        CommitmentCarrierCircuitDecision::Cooling
    );
    assert_eq!(
        circuit_breaker.decision(1, Instant::now()),
        CommitmentCarrierCircuitDecision::Cooling
    );
    let final_round =
        recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
            &destination,
            &peer_store,
            &coordinator,
            &carrier_ids,
            2,
            2,
            &client,
            &allow_test_endpoint,
            &mut circuit_breaker,
        )
        .await;
    assert_eq!(final_round.carrier_attempts, 0);
    assert_eq!(final_round.cooldown_skips, 2);
    assert_eq!(final_round.cooling_slots, 2);
}

#[tokio::test]
async fn carrier_circuit_breaker_skips_repeated_outage_across_sync_rounds() {
    let now = now_secs();
    let coordinator = IdentityKeyPair::generate();
    let unavailable_carrier = IdentityKeyPair::generate();
    let live_carrier = Arc::new(IdentityKeyPair::generate());
    let follower = IdentityKeyPair::generate();
    let source = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(1),
        GENESIS_PREV_HASH,
        vec![[0x74; 32]],
        &coordinator,
    );
    source
        .append_record_commitment_block(&block, None)
        .await
        .unwrap();
    source.audit_record_commitment_chain().await.unwrap();

    let live_peers = Arc::new(PeerStore::new());
    admit_peer(&live_peers, &follower, None, now);
    let live_router =
        build_memchain_peer_router(Arc::clone(&source), live_peers, Arc::clone(&live_carrier));
    let live_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let live_address = live_listener.local_addr().unwrap();
    let live_server = tokio::spawn(async move {
        axum::serve(live_listener, live_router).await.unwrap();
    });

    let coordinator_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let coordinator_address = coordinator_listener.local_addr().unwrap();
    let unavailable_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let unavailable_address = unavailable_listener.local_addr().unwrap();
    drop(coordinator_listener);
    drop(unavailable_listener);

    let destination = MemoryStorage::open(":memory:", None).unwrap();
    destination.audit_record_commitment_chain().await.unwrap();
    destination.configure_record_commitment_sync(false, true);
    destination.configure_record_commitment_certificate_policy(2, 2);
    let destination_peers = PeerStore::new();
    admit_peer(
        &destination_peers,
        &coordinator,
        Some(format!("http://{coordinator_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &unavailable_carrier,
        Some(format!("http://{unavailable_address}")),
        now,
    );
    admit_peer(
        &destination_peers,
        &live_carrier,
        Some(format!("http://{live_address}")),
        now,
    );
    let carrier_ids = [
        unavailable_carrier.public_key_bytes(),
        live_carrier.public_key_bytes(),
    ];
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let mut circuit_breaker = CommitmentBlockCarrierCircuitBreaker::default();

    // Each new cursor models a new follower round. The unavailable first
    // pin is contacted in rounds one and two, then its process-only slot
    // cools down while the exact second pin continues verified delivery.
    for expected_attempts in [2, 2, 1] {
        let mut cursor = CommitmentBlockCarrierCursor::default();
        let outcome = pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
            &destination,
            &destination_peers,
            &follower,
            &coordinator.public_key_bytes(),
            &carrier_ids,
            2,
            &client,
            &allow_test_endpoint,
            &mut cursor,
            &mut circuit_breaker,
            MAX_BLOCKS_PER_RESPONSE_WIRE,
        )
        .await
        .unwrap();
        assert_eq!(outcome.source, CommitmentSyncPageSource::PinnedCarrier);
        assert_eq!(outcome.carrier_attempts, expected_attempts);
        assert_eq!(outcome.page.remote_tip_height, 1);
        assert!(!outcome.page.has_more);
    }

    assert_eq!(
        circuit_breaker.decision(0, Instant::now()),
        CommitmentCarrierCircuitDecision::Cooling
    );

    // Force only the monotonic deadline to expire. The next real request
    // is counted as half-open, fails availability, and reopens the same
    // anonymous slot before the verified second carrier recovers the page.
    circuit_breaker.slots[0].retry_after = Some(Instant::now());
    let mut cursor = CommitmentBlockCarrierCursor::default();
    let half_open_outcome = pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
        &destination,
        &destination_peers,
        &follower,
        &coordinator.public_key_bytes(),
        &carrier_ids,
        2,
        &client,
        &allow_test_endpoint,
        &mut cursor,
        &mut circuit_breaker,
        MAX_BLOCKS_PER_RESPONSE_WIRE,
    )
    .await
    .unwrap();
    assert_eq!(
        half_open_outcome.source,
        CommitmentSyncPageSource::PinnedCarrier
    );
    assert_eq!(half_open_outcome.carrier_attempts, 2);
    assert_eq!(
        circuit_breaker.decision(0, Instant::now()),
        CommitmentCarrierCircuitDecision::Cooling
    );
    assert_eq!(destination.record_commitment_chain_tip().await.0, 1);
    destination.audit_record_commitment_chain().await.unwrap();
    let status = destination.record_commitment_sync_status();
    assert_eq!(status.block_page_pulls_total, 4);
    assert_eq!(status.block_carrier_attempts_total, 7);
    assert_eq!(status.block_carrier_recoveries_total, 4);
    assert_eq!(status.block_page_security_stops_total, 0);
    assert_eq!(status.block_carrier_cooling_slots, 1);
    assert_eq!(status.block_carrier_cooldown_skips_total, 1);
    assert_eq!(status.block_carrier_half_open_attempts_total, 1);

    live_server.abort();
    let _ = live_server.await;
}
