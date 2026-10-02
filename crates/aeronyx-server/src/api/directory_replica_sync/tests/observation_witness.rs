// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn witness_catch_up_requires_forward_progress_and_stops_at_budget() {
    assert!(should_attempt_observation_witness_catch_up(0, None, 41));
    assert!(should_attempt_observation_witness_catch_up(1, Some(41), 42));
    assert!(!should_attempt_observation_witness_catch_up(
        1,
        Some(41),
        41
    ));
    assert!(!should_attempt_observation_witness_catch_up(
        1,
        Some(41),
        40
    ));
    assert!(!should_attempt_observation_witness_catch_up(
        DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND,
        Some(44),
        45
    ));
}

#[test]
fn witness_recovery_selects_only_operator_pinned_explicit_carriers() {
    let now = unix_now_secs();
    let observer = IdentityKeyPair::from_bytes(&[0xb1; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xb2; 32]).unwrap();
    let pinned_carrier = IdentityKeyPair::from_bytes(&[0xb3; 32]).unwrap();
    let public_outsider = IdentityKeyPair::from_bytes(&[0xb4; 32]).unwrap();
    let store = PeerStore::new();
    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    for (identity, endpoint) in [
        (&pinned_carrier, "http://8.8.8.179:8422"),
        (&public_outsider, "http://8.8.8.180:8422"),
    ] {
        let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 600,
            "witness-carrier-selection-test",
        );
        descriptor.policy.public_discovery = true;
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor
            .capabilities
            .push(NodeCapability::DirectoryMirrorCarrier);
        store
            .upsert_verified_from_source(
                SignedNodeDescriptor::sign(descriptor, identity).unwrap(),
                now,
                "witness_carrier_selection_test",
            )
            .unwrap();
        store.record_route_forward_success(&identity.public_key_bytes(), now);
    }
    let eligible = [
        witness.public_key_bytes(),
        pinned_carrier.public_key_bytes(),
    ];
    let selection = directory_observation_witness_recovery_carriers(
        &store,
        &capability_cache,
        &witness.public_key_bytes(),
        &observer.public_key_bytes(),
        &eligible,
        now,
    );
    assert_eq!(selection.candidate_count, 1);
    assert_eq!(selection.carriers.len(), 1);
    assert_eq!(
        selection.carriers[0].node_id,
        pinned_carrier.public_key_bytes()
    );
    assert_ne!(
        selection.carriers[0].node_id,
        public_outsider.public_key_bytes()
    );
}

#[test]
fn observation_certificate_response_verification_binds_source_and_exact_frame() {
    let requester = IdentityKeyPair::from_bytes(&[0xb1; 32]).unwrap();
    let source = IdentityKeyPair::from_bytes(&[0xb2; 32]).unwrap();
    let other_source = IdentityKeyPair::from_bytes(&[0xb3; 32]).unwrap();
    let request_id = [0xb4; 16];
    let certificate_frame = vec![0xb5; 128];
    let certificate_sha256: [u8; 32] = Sha256::digest(&certificate_frame).into();
    let now = unix_now_secs();
    let signing_bytes = directory_observation_certificate_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester.public_key_bytes(),
        &source.public_key_bytes(),
        now,
        &certificate_sha256,
        u64::try_from(certificate_frame.len()).unwrap(),
    );
    let response = DirectorySyncMessage::ObservationCertificateResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: requester.public_key_bytes(),
        responder: source.public_key_bytes(),
        response_timestamp: now,
        certificate_sha256,
        certificate_frame: certificate_frame.clone(),
        signature: source.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        verify_observation_certificate_response(
            &frame,
            &request_id,
            &requester.public_key_bytes(),
            &source.public_key_bytes(),
            now,
            now,
        )
        .unwrap(),
        AuthenticatedDirectoryObservationCertificate {
            frame: certificate_frame,
            certificate_sha256,
            source: source.public_key_bytes(),
            response_timestamp: now,
        }
    );
    assert_eq!(
        verify_observation_certificate_response(
            &frame,
            &request_id,
            &requester.public_key_bytes(),
            &other_source.public_key_bytes(),
            now,
            now,
        )
        .unwrap_err(),
        "observation_certificate_response_contract_mismatch"
    );

    let mut tampered = response;
    let DirectorySyncMessage::ObservationCertificateResponseV1 {
        certificate_frame, ..
    } = &mut tampered
    else {
        unreachable!();
    };
    certificate_frame[0] ^= 1;
    assert_eq!(
        verify_observation_certificate_response(
            &encode_directory_sync_message(&tampered).unwrap(),
            &request_id,
            &requester.public_key_bytes(),
            &source.public_key_bytes(),
            now,
            now,
        )
        .unwrap_err(),
        "observation_certificate_response_digest_mismatch"
    );
}

#[test]
fn observation_witness_response_verification_is_exact_and_fail_closed() {
    let observer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
    let request_id = [0xe3; 16];
    let checkpoint_hash = [0xe4; 32];
    let now = unix_now_secs();
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        7,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        now,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        observer: observer.public_key_bytes(),
        checkpoint_sequence: 7,
        checkpoint_hash,
        responder: witness.public_key_bytes(),
        response_timestamp: now,
        outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        signature: witness.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        verify_observation_witness_response(
            &frame,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            7,
            &checkpoint_hash,
        )
        .unwrap(),
        response
    );

    let unavailable_signing = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        7,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        now,
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
    );
    let unavailable = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        observer: observer.public_key_bytes(),
        checkpoint_sequence: 7,
        checkpoint_hash,
        responder: witness.public_key_bytes(),
        response_timestamp: now,
        outcome: DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
        signature: witness.sign(&unavailable_signing),
    };
    assert_eq!(
        verify_observation_witness_response(
            &encode_directory_sync_message(&unavailable).unwrap(),
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            7,
            &checkpoint_hash,
        )
        .unwrap_err(),
        "observation_witness_evidence_unavailable"
    );

    let mut tampered = frame;
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(verify_observation_witness_response(
        &tampered,
        &request_id,
        &observer.public_key_bytes(),
        &witness.public_key_bytes(),
        now,
        7,
        &checkpoint_hash,
    )
    .is_err());
}

#[test]
fn witness_carrier_response_binds_exact_transport_and_inner_frame() {
    let observer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
    let other_carrier = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
    let request_id = [0x85; 16];
    let witness_request_sha256 = [0x86; 32];
    let witness_response_frame = vec![0x87; 96];
    let witness_response_sha256: [u8; 32] = Sha256::digest(&witness_response_frame).into();
    let now = unix_now_secs();
    let signing_bytes = directory_observation_witness_carrier_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        &witness.public_key_bytes(),
        &carrier.public_key_bytes(),
        now,
        &witness_request_sha256,
        &witness_response_sha256,
        u64::try_from(witness_response_frame.len()).unwrap(),
    );
    let response = DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: observer.public_key_bytes(),
        witness: witness.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: now,
        witness_request_sha256,
        witness_response_sha256,
        witness_response_frame: witness_response_frame.clone(),
        signature: carrier.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        verify_observation_witness_carrier_response(
            &frame,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            &carrier.public_key_bytes(),
            now,
            &witness_request_sha256,
        )
        .unwrap(),
        witness_response_frame
    );
    assert_eq!(
        verify_observation_witness_carrier_response(
            &frame,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            &other_carrier.public_key_bytes(),
            now,
            &witness_request_sha256,
        )
        .unwrap_err(),
        "observation_witness_carrier_response_contract_mismatch"
    );

    let mut tampered = response;
    let DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
        witness_response_frame,
        ..
    } = &mut tampered
    else {
        unreachable!();
    };
    witness_response_frame[0] ^= 1;
    assert_eq!(
        verify_observation_witness_carrier_response(
            &encode_directory_sync_message(&tampered).unwrap(),
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            &carrier.public_key_bytes(),
            now,
            &witness_request_sha256,
        )
        .unwrap_err(),
        "observation_witness_carrier_response_contract_mismatch"
    );

    let oversized_inner =
        vec![0x88; MAX_DIRECTORY_OBSERVATION_WITNESS_CARRIER_INNER_RESPONSE_BODY_BYTES + 1];
    let oversized_sha256: [u8; 32] = Sha256::digest(&oversized_inner).into();
    let oversized_signing = directory_observation_witness_carrier_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        &witness.public_key_bytes(),
        &carrier.public_key_bytes(),
        now,
        &witness_request_sha256,
        &oversized_sha256,
        u64::try_from(oversized_inner.len()).unwrap(),
    );
    let oversized = DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester: observer.public_key_bytes(),
        witness: witness.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: now,
        witness_request_sha256,
        witness_response_sha256: oversized_sha256,
        witness_response_frame: oversized_inner,
        signature: carrier.sign(&oversized_signing),
    };
    assert_eq!(
        verify_observation_witness_carrier_response(
            &encode_directory_sync_message(&oversized).unwrap(),
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            &carrier.public_key_bytes(),
            now,
            &witness_request_sha256,
        )
        .unwrap_err(),
        "observation_witness_carrier_response_contract_mismatch"
    );
}

#[test]
fn witness_carrier_fallback_is_availability_only() {
    for error in [
        DirectoryFramePostError::Transport(DirectoryTransportFailure::RequestTimeout),
        DirectoryFramePostError::HttpStatus {
            status: 408,
            peer_code: None,
        },
        DirectoryFramePostError::HttpStatus {
            status: 503,
            peer_code: None,
        },
        DirectoryFramePostError::HttpStatus {
            status: 503,
            peer_code: Some(DirectoryPeerErrorCode::WitnessTargetUnavailable),
        },
    ] {
        assert!(observation_witness_failure_allows_carrier(error));
    }
    for error in [
        DirectoryFramePostError::HttpStatus {
            status: 403,
            peer_code: None,
        },
        DirectoryFramePostError::HttpStatus {
            status: 424,
            peer_code: Some(DirectoryPeerErrorCode::WitnessTargetCapabilityUnavailable),
        },
        DirectoryFramePostError::HttpStatus {
            status: 502,
            peer_code: Some(DirectoryPeerErrorCode::WitnessTargetRejected),
        },
        DirectoryFramePostError::Response(BoundedHttpResponseError::BodyRead),
    ] {
        assert!(!observation_witness_failure_allows_carrier(error));
    }
    assert!(observation_witness_carrier_failure_allows_next(
        DirectoryFramePostError::HttpStatus {
            status: 403,
            peer_code: None,
        }
    ));
    assert!(!observation_witness_carrier_failure_allows_next(
        DirectoryFramePostError::HttpStatus {
            status: 502,
            peer_code: Some(DirectoryPeerErrorCode::WitnessTargetRejected),
        }
    ));
    assert_eq!(DIRECTORY_OBSERVATION_WITNESS_RECOVERY_MAX_CARRIERS, 2);
}

#[test]
fn witness_recovery_without_carriers_preserves_direct_transport_failure() {
    assert_eq!(
        observation_witness_unavailable_recovery_outcome(true),
        DirectoryObservationWitnessOutcome::TransportFailure
    );
    assert_eq!(
        observation_witness_unavailable_recovery_outcome(false),
        DirectoryObservationWitnessOutcome::PeerUnavailable
    );
}

#[test]
fn witness_capability_cache_is_scoped_to_authenticated_descriptor_sequence() {
    let cache = DirectoryWitnessCapabilityCache::default();
    let witness = [0x91; 32];

    assert!(cache.should_attempt(&witness, 7));
    cache.record_unsupported(witness, 7);
    assert!(!cache.should_attempt(&witness, 7));
    assert!(cache.should_attempt(&witness, 8));

    cache.record_supported(&witness);
    assert!(cache.should_attempt(&witness, 7));
}

#[test]
fn witness_capability_http_statuses_are_narrow_and_typed() {
    for status in [404, 405, 501] {
        assert!(DirectoryFramePostError::HttpStatus {
            status,
            peer_code: None,
        }
        .witness_capability_unavailable());
    }
    for status in [400, 401, 403, 409, 429, 500, 503] {
        assert!(!DirectoryFramePostError::HttpStatus {
            status,
            peer_code: None,
        }
        .witness_capability_unavailable());
    }
    assert_eq!(
        DirectoryFramePostError::HttpStatus {
            status: 404,
            peer_code: None,
        }
        .stable_reason("range"),
        "directory_range_http_status_404"
    );
    let lagging_carrier = DirectoryFramePostError::HttpStatus {
        status: 404,
        peer_code: Some(DirectoryPeerErrorCode::ReplicaRangeNotRetained),
    };
    assert!(!lagging_carrier.witness_capability_unavailable());
    assert_eq!(
        lagging_carrier.stable_reason("replica_range"),
        "directory_replica_range_peer_replica_range_not_retained"
    );
    assert_eq!(
        DirectoryFramePostError::Response(BoundedHttpResponseError::TooLarge)
            .stable_reason("objects"),
        "directory_objects_response_too_large"
    );
}
