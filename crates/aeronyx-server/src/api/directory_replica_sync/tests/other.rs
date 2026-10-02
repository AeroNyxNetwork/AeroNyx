// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn directory_transport_telemetry_is_task_scoped_and_mutually_exclusive() -> TestResult {
    let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
    let client = build_hardened_directory_http_client().map_err(std::io::Error::other)?;

    let (unscoped_url, unscoped_server) =
        carrier_hydration_test_endpoint(StatusCode::OK, b"ok".to_vec()).await?;
    assert_eq!(
        post_directory_frame_typed_with_response_limit(&client, unscoped_url, Vec::new(), 16,)
            .await
            .map_err(|error| std::io::Error::other(format!("{error:?}")))?,
        b"ok"
    );
    unscoped_server.abort();
    assert_eq!(runtime.directory_sync_transport_snapshot().requests, 0);

    let (success_url, success_server) =
        carrier_hydration_test_endpoint(StatusCode::OK, b"ok".to_vec()).await?;
    DIRECTORY_SYNC_TRANSPORT_RUNTIME
        .scope(
            Arc::clone(&runtime),
            post_directory_frame_typed_with_response_limit(&client, success_url, Vec::new(), 16),
        )
        .await
        .map_err(|error| std::io::Error::other(format!("{error:?}")))?;
    success_server.abort();

    let (status_url, status_server) =
        carrier_hydration_test_endpoint(StatusCode::SERVICE_UNAVAILABLE, Vec::new()).await?;
    assert!(matches!(
        DIRECTORY_SYNC_TRANSPORT_RUNTIME
            .scope(
                Arc::clone(&runtime),
                post_directory_frame_typed_with_response_limit(
                    &client,
                    status_url,
                    Vec::new(),
                    16,
                ),
            )
            .await,
        Err(DirectoryFramePostError::HttpStatus { status: 503, .. })
    ));
    status_server.abort();

    let (oversized_url, oversized_server) =
        carrier_hydration_test_endpoint(StatusCode::OK, vec![0u8; 17]).await?;
    assert_eq!(
        DIRECTORY_SYNC_TRANSPORT_RUNTIME
            .scope(
                Arc::clone(&runtime),
                post_directory_frame_typed_with_response_limit(
                    &client,
                    oversized_url,
                    Vec::new(),
                    16,
                ),
            )
            .await,
        Err(DirectoryFramePostError::Response(
            BoundedHttpResponseError::TooLarge
        ))
    );
    oversized_server.abort();

    let snapshot = runtime.directory_sync_transport_snapshot();
    assert_eq!(snapshot.requests, 3);
    assert_eq!(snapshot.terminal_outcomes(), snapshot.requests);
    assert_eq!(snapshot.succeeded, 1);
    assert_eq!(snapshot.http_status_failures, 1);
    assert_eq!(snapshot.response_too_large, 1);
    assert_eq!(
        snapshot.last_outcome,
        Some(DirectoryReplicaTransportOutcome::ResponseTooLarge)
    );
    Ok(())
}

#[test]
fn startup_delay_is_stable_bounded_and_identity_spread() {
    assert_eq!(directory_sync_startup_delay_secs(&[0u8; 32]), 5);
    assert_eq!(directory_sync_startup_delay_secs(&[10u8; 32]), 15);
    assert_eq!(directory_sync_startup_delay_secs(&[11u8; 32]), 5);
    assert_eq!(directory_sync_startup_delay_secs(&[255u8; 32]), 7);
}

#[test]
fn concurrency_cap_remains_small_and_nonzero() {
    assert!((1..=4).contains(&DIRECTORY_SYNC_MAX_CONCURRENT_PRODUCERS));
    assert!((1..120).contains(&DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS));
    assert_eq!(OUTBOUND_BLOCKS_PER_PAGE, MAX_DIRECTORY_SYNC_BLOCKS_V1);
}

#[test]
fn direct_descriptor_proof_response_binds_exact_trust_anchors() {
    let (producer, _, descriptor, block, proof) = descriptor_proof_test_context();
    let request_id = [0x74; 16];
    let now = unix_now_secs();
    let block_hash = block.hash();
    let descriptor_hash = proof.commitment.descriptor_hash;
    let signing_bytes = directory_descriptor_inclusion_proof_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        now,
        &block_hash,
        &descriptor_hash,
        &proof,
    );
    let response = DirectorySyncMessage::DescriptorInclusionProofResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        responder: producer.public_key_bytes(),
        response_timestamp: now,
        block_hash,
        descriptor_hash,
        proof,
        signature: producer.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        verify_descriptor_inclusion_proof_response(
            &frame,
            &request_id,
            &producer.public_key_bytes(),
            &block_hash,
            &descriptor_hash,
            now,
            now,
        )
        .unwrap()
        .descriptor,
        descriptor
    );
    assert_eq!(
        verify_descriptor_inclusion_proof_response(
            &frame,
            &request_id,
            &producer.public_key_bytes(),
            &[0x75; 32],
            &descriptor_hash,
            now,
            now,
        )
        .unwrap_err(),
        "directory_descriptor_proof_response_contract_mismatch"
    );

    let mut invalid_signature = response;
    let DirectorySyncMessage::DescriptorInclusionProofResponseV1 { signature, .. } =
        &mut invalid_signature
    else {
        unreachable!();
    };
    signature[0] ^= 1;
    assert_eq!(
        verify_descriptor_inclusion_proof_response(
            &encode_directory_sync_message(&invalid_signature).unwrap(),
            &request_id,
            &producer.public_key_bytes(),
            &block_hash,
            &descriptor_hash,
            now,
            now,
        )
        .unwrap_err(),
        "directory_descriptor_proof_response_invalid_signature"
    );
}

#[test]
fn replica_descriptor_proof_response_requires_both_signature_layers() {
    let (producer, carrier, descriptor, block, proof) = descriptor_proof_test_context();
    let request_id = [0x76; 16];
    let now = unix_now_secs();
    let block_hash = block.hash();
    let descriptor_hash = proof.commitment.descriptor_hash;
    let signing_bytes = directory_replica_descriptor_inclusion_proof_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        now,
        &block_hash,
        &descriptor_hash,
        &proof,
    );
    let response = DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: now,
        block_hash,
        descriptor_hash,
        proof: proof.clone(),
        signature: carrier.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        verify_replica_descriptor_inclusion_proof_response(
            &frame,
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            &block_hash,
            &descriptor_hash,
            now,
            now,
        )
        .unwrap()
        .descriptor,
        descriptor
    );

    let mut invalid_carrier = response;
    let DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 { signature, .. } =
        &mut invalid_carrier
    else {
        unreachable!();
    };
    signature[0] ^= 1;
    assert_eq!(
        verify_replica_descriptor_inclusion_proof_response(
            &encode_directory_sync_message(&invalid_carrier).unwrap(),
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            &block_hash,
            &descriptor_hash,
            now,
            now,
        )
        .unwrap_err(),
        "directory_replica_proof_response_invalid_carrier_signature"
    );

    let mut invalid_producer_proof = proof;
    invalid_producer_proof.producer_signature[0] ^= 1;
    let resigning_bytes = directory_replica_descriptor_inclusion_proof_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        now,
        &block_hash,
        &descriptor_hash,
        &invalid_producer_proof,
    );
    let resigned_invalid = DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: now,
        block_hash,
        descriptor_hash,
        proof: invalid_producer_proof,
        signature: carrier.sign(&resigning_bytes),
    };
    assert_eq!(
        verify_replica_descriptor_inclusion_proof_response(
            &encode_directory_sync_message(&resigned_invalid).unwrap(),
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            &block_hash,
            &descriptor_hash,
            now,
            now,
        )
        .unwrap_err(),
        "directory_replica_proof_response_invalid_producer_proof"
    );
}

#[test]
fn directory_authenticated_admission_is_locally_anchored_and_idempotent() {
    let (producer, _, descriptor, block, proof) = descriptor_proof_test_context();
    let now = unix_now_secs();
    let replica_store = descriptor_proof_replica_store(&producer, &descriptor, &block, now);
    let peer_store = PeerStore::new();
    let authenticated = AuthenticatedDirectoryDescriptorProof {
        proof,
        transport: DirectoryDescriptorProofTransport::DirectProducer,
        direct_attempted: true,
        carrier_attempts: 0,
    };
    let descriptor_hash = authenticated.proof.commitment.descriptor_hash;
    let block_hash = block.hash();

    let inserted = admit_directory_authenticated_descriptor(
        &replica_store,
        &peer_store,
        &authenticated,
        &producer.public_key_bytes(),
        &block_hash,
        &descriptor_hash,
        now,
    )
    .unwrap();
    assert_eq!(
        inserted,
        DirectoryAuthenticatedPeerAdmission {
            inserted: true,
            transport: DirectoryDescriptorProofTransport::DirectProducer,
            direct_attempted: true,
            carrier_attempts: 0,
        }
    );
    assert_eq!(
        peer_store.get_valid(&descriptor.node_id(), now),
        Some(descriptor.clone())
    );

    let unchanged = admit_directory_authenticated_descriptor(
        &replica_store,
        &peer_store,
        &authenticated,
        &producer.public_key_bytes(),
        &block_hash,
        &descriptor_hash,
        now,
    )
    .unwrap();
    assert!(!unchanged.inserted);
    assert_eq!(peer_store.len(), 1);
}

#[test]
fn directory_gossip_admission_is_locally_anchored_and_idempotent() {
    let (producer, _, descriptor, block, proof) = descriptor_proof_test_context();
    let now = unix_now_secs();
    let replica_store = descriptor_proof_replica_store(&producer, &descriptor, &block, now);
    let peer_store = PeerStore::new();
    let descriptor_hash = proof.commitment.descriptor_hash;

    let inserted = admit_directory_gossip_descriptor(
        &replica_store,
        &peer_store,
        &proof,
        &producer.public_key_bytes(),
        &block.hash(),
        &descriptor_hash,
        now,
    )
    .unwrap();
    assert_eq!(inserted.inserted, 1);
    assert_eq!(inserted.rejected, 0);
    assert_eq!(
        peer_store.get_valid(&descriptor.node_id(), now),
        Some(descriptor.clone())
    );

    let unchanged = admit_directory_gossip_descriptor(
        &replica_store,
        &peer_store,
        &proof,
        &producer.public_key_bytes(),
        &block.hash(),
        &descriptor_hash,
        now,
    )
    .unwrap();
    assert_eq!(unchanged.unchanged, 1);
    assert_eq!(peer_store.len(), 1);
}

#[test]
fn directory_gossip_admission_rejects_unknown_anchor_and_rollback() {
    let (producer, _, descriptor, block, proof) = descriptor_proof_test_context();
    let now = unix_now_secs();
    let empty_local = IdentityKeyPair::from_bytes(&[0x7a; 32]).unwrap();
    let (empty_replica, _) =
        DirectoryReplicaStore::open(":memory:", empty_local.public_key_bytes(), now).unwrap();
    let descriptor_hash = proof.commitment.descriptor_hash;
    let peer_store = PeerStore::new();

    assert_eq!(
        admit_directory_gossip_descriptor(
            &empty_replica,
            &peer_store,
            &proof,
            &producer.public_key_bytes(),
            &block.hash(),
            &descriptor_hash,
            now,
        )
        .unwrap_err(),
        "directory_authenticated_admission_local_anchor_not_found"
    );
    assert_eq!(peer_store.len(), 0);

    let replica_store = descriptor_proof_replica_store(&producer, &descriptor, &block, now);
    let subject = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
    let newer = SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            subject.public_key_bytes(),
            descriptor.sequence() + 1,
            now.saturating_sub(1),
            now + 600,
            "directory-gossip-newer-test",
        ),
        &subject,
    )
    .unwrap();
    peer_store
        .upsert_verified_from_source(newer.clone(), now, "test_newer")
        .unwrap();

    let stale = admit_directory_gossip_descriptor(
        &replica_store,
        &peer_store,
        &proof,
        &producer.public_key_bytes(),
        &block.hash(),
        &descriptor_hash,
        now,
    )
    .unwrap();
    assert_eq!(stale.stale, 1);
    assert_eq!(stale.inserted, 0);
    assert_eq!(
        peer_store.get_valid(&subject.public_key_bytes(), now),
        Some(newer)
    );
}

#[test]
fn directory_authenticated_admission_rejects_unknown_local_anchor() {
    let (producer, _, descriptor, block, _) = descriptor_proof_test_context();
    let now = unix_now_secs();
    let replica_store = descriptor_proof_replica_store(&producer, &descriptor, &block, now);
    let alternate_block = DirectoryCommitmentBlockV1::new_signed(
        1,
        // [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] Derive the
        // alternate timestamp from the retained block. Using a second wall
        // clock read made this unknown-anchor regression flaky at a
        // one-second rollover because both blocks could become identical.
        block.header.timestamp.saturating_sub(1),
        [0u8; 32],
        vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor).unwrap()],
        &producer,
    )
    .unwrap();
    let alternate_proof =
        DirectoryDescriptorInclusionProofV1::from_block_at(&alternate_block, &descriptor, now)
            .unwrap();
    let descriptor_hash = alternate_proof.commitment.descriptor_hash;
    let authenticated = AuthenticatedDirectoryDescriptorProof {
        proof: alternate_proof,
        transport: DirectoryDescriptorProofTransport::ReplicaCarrier,
        direct_attempted: false,
        carrier_attempts: 1,
    };
    let peer_store = PeerStore::new();

    assert_eq!(
        admit_directory_authenticated_descriptor(
            &replica_store,
            &peer_store,
            &authenticated,
            &producer.public_key_bytes(),
            &alternate_block.hash(),
            &descriptor_hash,
            now,
        )
        .unwrap_err(),
        "directory_authenticated_admission_local_anchor_not_found"
    );
    assert_eq!(peer_store.len(), 0);
}

#[test]
fn directory_authenticated_admission_reverifies_public_wrapper() {
    let (producer, _, descriptor, block, mut proof) = descriptor_proof_test_context();
    let now = unix_now_secs();
    let replica_store = descriptor_proof_replica_store(&producer, &descriptor, &block, now);
    let descriptor_hash = proof.commitment.descriptor_hash;
    proof.producer_signature[0] ^= 1;
    let peer_store = PeerStore::new();
    let authenticated = AuthenticatedDirectoryDescriptorProof {
        proof,
        transport: DirectoryDescriptorProofTransport::DirectProducer,
        direct_attempted: true,
        carrier_attempts: 0,
    };

    assert_eq!(
        admit_directory_authenticated_descriptor(
            &replica_store,
            &peer_store,
            &authenticated,
            &producer.public_key_bytes(),
            &block.hash(),
            &descriptor_hash,
            now,
        )
        .unwrap_err(),
        "directory_authenticated_admission_proof_invalid"
    );
    assert_eq!(peer_store.len(), 0);

    let impossible_transport = AuthenticatedDirectoryDescriptorProof {
        proof: authenticated.proof.clone(),
        transport: DirectoryDescriptorProofTransport::DirectProducer,
        direct_attempted: false,
        carrier_attempts: 1,
    };
    assert_eq!(
        admit_directory_authenticated_descriptor(
            &replica_store,
            &peer_store,
            &impossible_transport,
            &producer.public_key_bytes(),
            &block.hash(),
            &descriptor_hash,
            now,
        )
        .unwrap_err(),
        "directory_authenticated_admission_transport_invalid"
    );
}

#[test]
fn directory_authenticated_admission_preserves_peer_store_anti_rollback() {
    let (producer, _, descriptor, block, proof) = descriptor_proof_test_context();
    let now = unix_now_secs();
    let replica_store = descriptor_proof_replica_store(&producer, &descriptor, &block, now);
    let subject = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
    let newer = SignedNodeDescriptor::sign(
        NodeDescriptor::new(
            subject.public_key_bytes(),
            descriptor.sequence() + 1,
            now.saturating_sub(1),
            now + 600,
            "directory-admission-newer-test",
        ),
        &subject,
    )
    .unwrap();
    let peer_store = PeerStore::new();
    peer_store
        .upsert_verified_from_source(newer.clone(), now, "test_newer")
        .unwrap();
    let descriptor_hash = proof.commitment.descriptor_hash;
    let authenticated = AuthenticatedDirectoryDescriptorProof {
        proof,
        transport: DirectoryDescriptorProofTransport::ReplicaCarrier,
        direct_attempted: true,
        carrier_attempts: 1,
    };

    assert_eq!(
        admit_directory_authenticated_descriptor(
            &replica_store,
            &peer_store,
            &authenticated,
            &producer.public_key_bytes(),
            &block.hash(),
            &descriptor_hash,
            now,
        )
        .unwrap_err(),
        "directory_authenticated_admission_stale_sequence"
    );
    assert_eq!(
        peer_store.get_valid(&subject.public_key_bytes(), now),
        Some(newer)
    );
}

#[test]
fn descriptor_proof_recovery_is_availability_only() {
    assert!(directory_descriptor_proof_direct_post_allows_recovery(
        DirectoryFramePostError::Transport(DirectoryTransportFailure::Connect)
    ));
    assert!(directory_descriptor_proof_direct_post_allows_recovery(
        DirectoryFramePostError::HttpStatus {
            status: 403,
            peer_code: None,
        }
    ));
    assert!(!directory_descriptor_proof_direct_post_allows_recovery(
        DirectoryFramePostError::HttpStatus {
            status: 404,
            peer_code: Some(DirectoryPeerErrorCode::ProofNotFound),
        }
    ));
    assert!(!directory_descriptor_proof_direct_post_allows_recovery(
        DirectoryFramePostError::Response(BoundedHttpResponseError::BodyRead)
    ));

    let carrier_miss = DirectoryFramePostError::HttpStatus {
        status: 404,
        peer_code: Some(DirectoryPeerErrorCode::ReplicaDescriptorProofNotFound),
    };
    assert!(directory_descriptor_proof_carrier_post_allows_next(
        carrier_miss
    ));
    assert!(directory_descriptor_proof_carrier_post_allows_next(
        DirectoryFramePostError::HttpStatus {
            status: 404,
            peer_code: Some(DirectoryPeerErrorCode::MirrorReplicaNotRetained),
        }
    ));
    assert!(!directory_descriptor_proof_carrier_post_allows_next(
        DirectoryFramePostError::Response(BoundedHttpResponseError::BodyRead)
    ));
    assert!(directory_descriptor_proof_carrier_capability_unavailable(
        DirectoryFramePostError::HttpStatus {
            status: 404,
            peer_code: None,
        }
    ));
    assert!(!directory_descriptor_proof_carrier_capability_unavailable(
        carrier_miss
    ));
}

#[test]
fn policy_anchor_response_verification_binds_the_complete_statement() {
    let observer = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
    let other_witness = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
    let request_id = [0xa4; 16];
    let policy_epoch = 7;
    let policy_digest = [0xa5; 32];
    let now = unix_now_secs();
    let response = |outcome: u8| {
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            policy_epoch,
            &policy_digest,
            &witness.public_key_bytes(),
            now,
            outcome,
        );
        DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            policy_epoch,
            policy_digest,
            responder: witness.public_key_bytes(),
            response_timestamp: now,
            outcome,
            signature: witness.sign(&signing_bytes),
        }
    };

    let accepted = response(DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1);
    let frame = encode_directory_sync_message(&accepted).unwrap();
    assert_eq!(
        verify_observation_policy_anchor_response(
            &frame,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            policy_epoch,
            &policy_digest,
        )
        .unwrap(),
        accepted
    );
    assert_eq!(
        verify_observation_policy_anchor_response(
            &frame,
            &request_id,
            &observer.public_key_bytes(),
            &other_witness.public_key_bytes(),
            now,
            policy_epoch,
            &policy_digest,
        )
        .unwrap_err(),
        "observation_policy_anchor_response_contract_mismatch"
    );
    assert_eq!(
        verify_observation_policy_anchor_response(
            &frame,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            policy_epoch,
            &[0xff; 32],
        )
        .unwrap_err(),
        "observation_policy_anchor_response_contract_mismatch"
    );

    let rollback =
        encode_directory_sync_message(&response(DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1)).unwrap();
    assert_eq!(
        verify_observation_policy_anchor_response(
            &rollback,
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            policy_epoch,
            &policy_digest,
        )
        .unwrap_err(),
        "observation_policy_anchor_rollback"
    );

    let mut tampered = response(DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1);
    let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 { signature, .. } =
        &mut tampered
    else {
        unreachable!();
    };
    signature[0] ^= 1;
    assert_eq!(
        verify_observation_policy_anchor_response(
            &encode_directory_sync_message(&tampered).unwrap(),
            &request_id,
            &observer.public_key_bytes(),
            &witness.public_key_bytes(),
            now,
            policy_epoch,
            &policy_digest,
        )
        .unwrap_err(),
        "observation_policy_anchor_response_invalid_signature"
    );
}

#[test]
fn checkpoint_requires_exact_authenticated_remote_tip() {
    let complete = DirectorySyncPullOutcome {
        import: DirectoryReplicaImportReport {
            blocks_inserted: 1,
            blocks_already_present: 0,
            commitments_inserted: 4,
            descriptor_equivocations: 0,
            tip_height: 9,
            tip_hash: [0x41; 32],
        },
        has_more: false,
        remote_tip_height: 9,
        remote_tip_hash: [0x41; 32],
        requests_made: 2,
    };
    assert!(directory_sync_outcome_is_checkpoint_complete(
        &complete,
        DirectoryMirrorPullSource::DirectProducer
    ));
    assert!(!directory_sync_outcome_is_checkpoint_complete(
        &complete,
        DirectoryMirrorPullSource::PublicCarrier
    ));

    let mut catching_up = complete;
    catching_up.has_more = true;
    assert!(!directory_sync_outcome_is_checkpoint_complete(
        &catching_up,
        DirectoryMirrorPullSource::DirectProducer
    ));
    let mut stale_height = complete;
    stale_height.remote_tip_height = 10;
    assert!(!directory_sync_outcome_is_checkpoint_complete(
        &stale_height,
        DirectoryMirrorPullSource::DirectProducer
    ));
    let mut wrong_hash = complete;
    wrong_hash.remote_tip_hash = [0x42; 32];
    assert!(!directory_sync_outcome_is_checkpoint_complete(
        &wrong_hash,
        DirectoryMirrorPullSource::DirectProducer
    ));
}
