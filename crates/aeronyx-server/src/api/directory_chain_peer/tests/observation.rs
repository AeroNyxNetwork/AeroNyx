// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn witness_route_signs_unavailable_instead_of_trusting_observer() {
    let (router, witness, observer, _) = witness_test_router();
    let now = now_secs();
    let producer_a = IdentityKeyPair::from_bytes(&[0xc3; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0xc4; 32]).unwrap();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        1,
        now,
        [0u8; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 1,
                tip_hash: [0xc5; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 1,
                tip_hash: [0xc6; 32],
            },
        ],
        [0xc7; 32],
        &observer,
    )
    .unwrap();
    let request_id = [0xc8; 16];
    let checkpoint_hash = checkpoint.hash();
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        now,
        &checkpoint_hash,
    );
    let request = encode_directory_sync_message(
        &DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: observer.public_key_bytes(),
            request_timestamp: now,
            checkpoint,
            signature: observer.sign(&signing_bytes),
        },
    )
    .unwrap();
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-checkpoint-witness")
                .body(Body::from(request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let message = decode_directory_sync_message(&body).unwrap();
    let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        observer: response_observer,
        checkpoint_sequence,
        checkpoint_hash: response_checkpoint_hash,
        responder,
        response_timestamp,
        outcome,
        signature,
        ..
    } = message
    else {
        panic!("unexpected response");
    };
    assert_eq!(response_observer, observer.public_key_bytes());
    assert_eq!(checkpoint_sequence, 1);
    assert_eq!(response_checkpoint_hash, checkpoint_hash);
    assert_eq!(responder, witness.public_key_bytes());
    assert_eq!(
        outcome,
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1
    );
    let response_signing = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &response_observer,
        checkpoint_sequence,
        &response_checkpoint_hash,
        &responder,
        response_timestamp,
        outcome,
    );
    IdentityPublicKey::from_bytes(&responder)
        .unwrap()
        .verify(&response_signing, &signature)
        .unwrap();
}

#[tokio::test]
async fn witness_carrier_rejects_unpinned_target_before_transport() {
    let (router, observer, witness, runtime, calls) = witness_carrier_test_router(
        Err(WitnessCarrierTransportError::TargetUnavailable),
        false,
        true,
    );
    let request = witness_carrier_outer_request(
        &observer,
        &witness,
        witness_carrier_inner_request(&observer),
    );
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-checkpoint-witness-carrier")
                .body(Body::from(request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_single_witness_carrier_outcome(
        runtime.observation_witness_carrier_snapshot(),
        DirectoryObservationWitnessCarrierOutcome::PolicyRejected,
    );
}

#[tokio::test]
async fn witness_carrier_rejects_excess_outbound_work_without_queueing() {
    let observer = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xd9; 32]).unwrap();
    let inner_frame = witness_carrier_inner_request(&observer);
    let valid_response = witness_carrier_target_response(&observer, &witness, &inner_frame);
    let calls = Arc::new(AtomicU64::new(0));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Notify::new());
    let transport = Arc::new(BlockingWitnessCarrierTransport {
        response: WitnessCarrierTransportResponse {
            status: 200,
            body: valid_response,
        },
        calls: Arc::clone(&calls),
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    });
    let (router, observer, witness, runtime, _) =
        witness_carrier_test_router_with_transport(transport, true, true, 1);

    // [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Hold the sole permit
    // inside transport, then prove the next request fails immediately and
    // does not enter the transport implementation.
    let first_router = router.clone();
    let first_request =
        witness_carrier_outer_request_with_id(&observer, &witness, inner_frame.clone(), [0xe1; 16]);
    let first = tokio::spawn(async move {
        first_router
            .oneshot(
                Request::post(
                    "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                )
                .body(Body::from(first_request))
                .unwrap(),
            )
            .await
            .unwrap()
    });
    entered.wait().await;

    let second_request =
        witness_carrier_outer_request_with_id(&observer, &witness, inner_frame, [0xe2; 16]);
    let second = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-checkpoint-witness-carrier")
                .body(Body::from(second_request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    release.notify_one();
    assert_eq!(first.await.unwrap().status(), StatusCode::OK);
    let snapshot = runtime.observation_witness_carrier_snapshot();
    assert_eq!(snapshot.requests, 2);
    assert_eq!(snapshot.forwarded, 1);
    assert_eq!(snapshot.local_overloaded, 1);
    assert_eq!(
        snapshot.forwarded
            + snapshot.policy_rejected
            + snapshot.invalid_requests
            + snapshot.target_unavailable
            + snapshot.target_capability_unavailable
            + snapshot.target_rejected
            + snapshot.target_invalid_response
            + snapshot.target_cooling_down
            + snapshot.local_overloaded
            + snapshot.local_failures,
        snapshot.requests
    );
}

#[tokio::test]
async fn witness_carrier_cooldown_is_descriptor_bound_and_clears_after_recovery() {
    let calls = Arc::new(AtomicU64::new(0));
    let transport = Arc::new(RecoveringWitnessCarrierTransport {
        observer: IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap(),
        witness: IdentityKeyPair::from_bytes(&[0xd9; 32]).unwrap(),
        calls: Arc::clone(&calls),
    });
    let (router, observer, witness, runtime, peer_store) =
        witness_carrier_test_router_with_transport(transport, true, true, 1);

    let first = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-checkpoint-witness-carrier")
                .body(Body::from(witness_carrier_outer_request_with_id(
                    &observer,
                    &witness,
                    witness_carrier_inner_request_with_id(&observer, [0xf1; 16]),
                    [0xe3; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::SERVICE_UNAVAILABLE);

    let cooling = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-checkpoint-witness-carrier")
                .body(Body::from(witness_carrier_outer_request_with_id(
                    &observer,
                    &witness,
                    witness_carrier_inner_request_with_id(&observer, [0xf2; 16]),
                    [0xe4; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cooling.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    // [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] A newly signed
    // descriptor sequence is new availability evidence and must bypass the
    // old process-only cooldown immediately.
    let now = now_secs();
    let mut descriptor = NodeDescriptor::new(
        witness.public_key_bytes(),
        2,
        now,
        now + 600,
        "directory-sync-test",
    );
    descriptor.public_endpoint = Some("1.1.1.1:8422".to_string());
    peer_store
        .upsert_verified_from_source(
            SignedNodeDescriptor::sign(descriptor, &witness).unwrap(),
            now,
            "directory_witness_carrier_rotation_test",
        )
        .unwrap();

    for (request_id, inner_request_id) in [([0xe5; 16], [0xf3; 16]), ([0xe6; 16], [0xf4; 16])] {
        let recovered = router
            .clone()
            .oneshot(
                Request::post(
                    "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                )
                .body(Body::from(witness_carrier_outer_request_with_id(
                    &observer,
                    &witness,
                    witness_carrier_inner_request_with_id(&observer, inner_request_id),
                    request_id,
                )))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(recovered.status(), StatusCode::OK);
    }

    assert_eq!(calls.load(Ordering::Relaxed), 3);
    let snapshot = runtime.observation_witness_carrier_snapshot();
    assert_eq!(snapshot.requests, 4);
    assert_eq!(snapshot.forwarded, 2);
    assert_eq!(snapshot.target_unavailable, 1);
    assert_eq!(snapshot.target_cooling_down, 1);
    assert_eq!(
        snapshot.forwarded
            + snapshot.policy_rejected
            + snapshot.invalid_requests
            + snapshot.target_unavailable
            + snapshot.target_capability_unavailable
            + snapshot.target_rejected
            + snapshot.target_invalid_response
            + snapshot.target_cooling_down
            + snapshot.local_overloaded
            + snapshot.local_failures,
        snapshot.requests
    );
}

#[tokio::test]
async fn witness_carrier_handler_maps_each_transport_result_to_one_terminal_outcome() {
    let observer = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xd9; 32]).unwrap();
    let inner_frame = witness_carrier_inner_request(&observer);
    let valid_response = witness_carrier_target_response(&observer, &witness, &inner_frame);
    let cases = vec![
        (
            "forwarded",
            Ok(WitnessCarrierTransportResponse {
                status: 200,
                body: valid_response,
            }),
            StatusCode::OK,
            DirectoryObservationWitnessCarrierOutcome::Forwarded,
        ),
        (
            "route_not_found",
            Ok(WitnessCarrierTransportResponse {
                status: 404,
                body: Vec::new(),
            }),
            StatusCode::FAILED_DEPENDENCY,
            DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable,
        ),
        (
            "target_rate_limited",
            Ok(WitnessCarrierTransportResponse {
                status: 429,
                body: Vec::new(),
            }),
            StatusCode::SERVICE_UNAVAILABLE,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
        ),
        (
            "target_server_failure",
            Ok(WitnessCarrierTransportResponse {
                status: 503,
                body: Vec::new(),
            }),
            StatusCode::SERVICE_UNAVAILABLE,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
        ),
        (
            "target_rejected",
            Ok(WitnessCarrierTransportResponse {
                status: 403,
                body: Vec::new(),
            }),
            StatusCode::BAD_GATEWAY,
            DirectoryObservationWitnessCarrierOutcome::TargetRejected,
        ),
        (
            "malformed_success_body",
            Ok(WitnessCarrierTransportResponse {
                status: 200,
                body: vec![0xff],
            }),
            StatusCode::BAD_GATEWAY,
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
        ),
        (
            "response_body_too_large",
            Err(WitnessCarrierTransportError::ResponseTooLarge),
            StatusCode::BAD_GATEWAY,
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
        ),
        (
            "target_timeout_or_stream_failure",
            Err(WitnessCarrierTransportError::TargetUnavailable),
            StatusCode::SERVICE_UNAVAILABLE,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
        ),
        (
            "local_client_unavailable",
            Err(WitnessCarrierTransportError::LocalUnavailable),
            StatusCode::SERVICE_UNAVAILABLE,
            DirectoryObservationWitnessCarrierOutcome::LocalFailure,
        ),
    ];

    for (name, transport_result, expected_status, expected_outcome) in cases {
        let (router, observer, witness, runtime, calls) =
            witness_carrier_test_router(transport_result, true, true);
        let request = witness_carrier_outer_request(&observer, &witness, inner_frame.clone());
        let response = router
            .oneshot(
                Request::post(
                    "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                )
                .body(Body::from(request))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected_status, "{name}");
        assert_eq!(calls.load(Ordering::Relaxed), 1, "{name}");
        assert_single_witness_carrier_outcome(
            runtime.observation_witness_carrier_snapshot(),
            expected_outcome,
        );
    }
}

#[tokio::test]
async fn witness_carrier_handler_fails_closed_before_transport_for_inner_and_descriptor_errors() {
    let cases = [
        (
            "invalid_inner_frame",
            true,
            vec![0xff],
            StatusCode::BAD_REQUEST,
            DirectoryObservationWitnessCarrierOutcome::InvalidRequest,
        ),
        (
            "missing_target_descriptor",
            false,
            Vec::new(),
            StatusCode::SERVICE_UNAVAILABLE,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
        ),
    ];

    for (name, advertise_target, invalid_inner, expected_status, expected_outcome) in cases {
        let (router, observer, witness, runtime, calls) = witness_carrier_test_router(
            Err(WitnessCarrierTransportError::TargetUnavailable),
            true,
            advertise_target,
        );
        let inner_frame = if invalid_inner.is_empty() {
            witness_carrier_inner_request(&observer)
        } else {
            invalid_inner
        };
        let request = witness_carrier_outer_request(&observer, &witness, inner_frame);
        let response = router
            .oneshot(
                Request::post(
                    "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
                )
                .body(Body::from(request))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected_status, "{name}");
        assert_eq!(calls.load(Ordering::Relaxed), 0, "{name}");
        assert_single_witness_carrier_outcome(
            runtime.observation_witness_carrier_snapshot(),
            expected_outcome,
        );
    }
}

#[test]
fn carried_witness_response_requires_exact_target_signature() {
    let observer = IdentityKeyPair::from_bytes(&[0xdf; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xe0; 32]).unwrap();
    let other = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0xe3; 32]).unwrap();
    let now = now_secs();
    let checkpoint = DirectoryObservationCheckpointV1::new_signed(
        1,
        now,
        [0u8; 32],
        2,
        vec![
            DirectoryObservationTipV1 {
                producer: producer_a.public_key_bytes(),
                tip_height: 1,
                tip_hash: [0xe4; 32],
            },
            DirectoryObservationTipV1 {
                producer: producer_b.public_key_bytes(),
                tip_height: 1,
                tip_hash: [0xe5; 32],
            },
        ],
        [0xe6; 32],
        &observer,
    )
    .unwrap();
    let request_id = [0xe7; 16];
    let checkpoint_hash = checkpoint.hash();
    let request_signing = directory_observation_witness_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        now,
        &checkpoint_hash,
    );
    let request_frame = encode_directory_sync_message(
        &DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester: observer.public_key_bytes(),
            request_timestamp: now,
            checkpoint,
            signature: observer.sign(&request_signing),
        },
    )
    .unwrap();
    let context = verify_carried_observation_witness_request(
        &request_frame,
        &observer.public_key_bytes(),
        now,
    )
    .unwrap();
    let response_signing = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        1,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        now,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    let response_frame = encode_directory_sync_message(
        &DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: 1,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp: now,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&response_signing),
        },
    )
    .unwrap();
    assert!(verify_carried_observation_witness_response(
        &response_frame,
        &context,
        &witness.public_key_bytes(),
        now
    )
    .is_ok());
    assert_eq!(
        verify_carried_observation_witness_response(
            &response_frame,
            &context,
            &other.public_key_bytes(),
            now
        ),
        Err("carried_witness_response_contract_mismatch")
    );
}

#[tokio::test]
async fn observation_certificate_route_is_pinned_and_fails_closed_without_evidence() {
    let (router, _, observer, _) = witness_test_router();
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-certificate")
                .body(Body::from(observation_certificate_request(
                    &observer, [0xca; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let unpinned = IdentityKeyPair::from_bytes(&[0xcb; 32]).unwrap();
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-certificate")
                .body(Body::from(observation_certificate_request(
                    &unpinned, [0xcc; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn observation_certificate_route_serves_exact_current_policy_evidence() {
    let (router, observer, requester, expected_sequence) = positive_certificate_test_router();
    let request_id = [0xdd; 16];
    let request_timestamp = now_secs();
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/observation-certificate")
                .body(Body::from(observation_certificate_request(
                    &requester, request_id,
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let observed_at = now_secs();
    let authenticated = verify_observation_certificate_response(
        &body,
        &request_id,
        &requester.public_key_bytes(),
        &observer.public_key_bytes(),
        request_timestamp,
        observed_at,
    )
    .unwrap();
    let certificate = decode_directory_observation_certificate(&authenticated.frame).unwrap();
    certificate
        .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
        .unwrap();
    assert_eq!(certificate.checkpoint.observer, observer.public_key_bytes());
    assert_eq!(certificate.checkpoint.sequence, expected_sequence);
    assert_eq!(certificate.minimum_witnesses, 2);
    assert_eq!(certificate.receipts.len(), 2);
}
