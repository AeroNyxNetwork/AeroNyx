// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn blocking_worker_failure_preserves_privacy_safe_protocol_bucket() {
    // [DIRECTORY-BLOCKING-BOUNDARY 2026-07-30 by Codex] A panic payload
    // must never become part of the authenticated peer API response.
    let response = run_directory_chain_blocking(Arc::new(Semaphore::new(1)), "test_audit", || {
        panic!("test-only-sensitive-directory-worker-payload");
    })
    .await
    .unwrap_err();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), 1_024)
        .await
        .expect("worker failure body");
    assert_eq!(&body[..], b"audit_task_failed");
    assert!(!body
        .windows(b"sensitive-directory-worker-payload".len())
        .any(|window| window == b"sensitive-directory-worker-payload"));
}

#[tokio::test]
async fn blocking_worker_admission_rejects_concurrent_audits_without_queueing() {
    use std::sync::mpsc;

    let admission = Arc::new(Semaphore::new(1));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first_admission = Arc::clone(&admission);
    let first = tokio::spawn(async move {
        run_directory_chain_blocking_with_admission(first_admission, "test_long_audit", move || {
            entered_tx.send(()).expect("announce worker entry");
            release_rx.recv().expect("release first worker");
            7u8
        })
        .await
    });
    tokio::task::spawn_blocking(move || entered_rx.recv().expect("first worker entered"))
        .await
        .expect("entry wait task");

    let response = run_directory_chain_blocking_with_admission(
        Arc::clone(&admission),
        "test_rejected_audit",
        || 9u8,
    )
    .await
    .unwrap_err();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = to_bytes(response.into_body(), 1_024)
        .await
        .expect("admission response body");
    assert_eq!(&body[..], b"audit_busy");

    release_tx.send(()).expect("release first audit");
    assert_eq!(
        first.await.expect("first task join").expect("first audit"),
        7
    );
    assert_eq!(admission.available_permits(), 1);
}

#[tokio::test]
async fn pinned_live_peer_receives_signed_audited_tip_and_replay_is_rejected() {
    let (router, producer, requester, _) = test_router(true, true, false);
    let request_id = [0xb1; 16];
    let request = tip_request(&requester, request_id);
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/tip")
                .header("content-type", "application/octet-stream")
                .body(Body::from(request.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let DirectorySyncMessage::TipResponseV1 {
        chain_id,
        request_id: response_request_id,
        responder,
        response_timestamp,
        tip_height,
        tip_hash,
        tip_timestamp,
        signature,
    } = decode_directory_sync_message(&body).unwrap()
    else {
        panic!("unexpected response");
    };
    assert_eq!(chain_id, AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
    assert_eq!(response_request_id, request_id);
    assert_eq!(responder, producer.public_key_bytes());
    assert_eq!(tip_height, 1);
    let signing_bytes = directory_tip_response_signing_bytes(
        &chain_id,
        &response_request_id,
        &responder,
        response_timestamp,
        tip_height,
        &tip_hash,
        tip_timestamp,
    );
    IdentityPublicKey::from_bytes(&responder)
        .unwrap()
        .verify(&signing_bytes, &signature)
        .unwrap();

    let replay = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/tip")
                .body(Body::from(request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(replay.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn unpinned_public_peer_can_read_signed_local_producer_tip() {
    let (router, _, requester, _) = test_router(false, true, true);
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/tip")
                .body(Body::from(tip_request(&requester, [0xb2; 16])))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn unpinned_private_peer_cannot_read_signed_local_producer_tip() {
    let (router, _, requester, _) = test_router(false, false, true);
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/tip")
                .body(Body::from(tip_request(&requester, [0xb9; 16])))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn block_and_descriptor_routes_return_exact_committed_objects() {
    let (router, producer, requester, expected_descriptor) = test_router(true, true, false);
    let timestamp = now_secs();
    let requester_id = requester.public_key_bytes();
    let range_id = [0xb3; 16];
    let range_signing = directory_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        1,
        1,
        &range_id,
        &requester_id,
        timestamp,
    );
    let range = encode_directory_sync_message(&DirectorySyncMessage::BlockRangeRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        from_height: 1,
        limit: 1,
        request_id: range_id,
        requester: requester_id,
        request_timestamp: timestamp,
        signature: requester.sign(&range_signing),
    })
    .unwrap();
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/block-range")
                .body(Body::from(range))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let verified_range =
        verify_block_range_response(&body, &range_id, &producer.public_key_bytes(), 1, timestamp)
            .unwrap();
    assert_eq!(verified_range.0.len(), 1);
    let mut tampered_range = body.to_vec();
    *tampered_range.last_mut().unwrap() ^= 0x01;
    assert_eq!(
        verify_block_range_response(
            &tampered_range,
            &range_id,
            &producer.public_key_bytes(),
            1,
            timestamp,
        )
        .unwrap_err(),
        "directory_range_response_invalid_signature"
    );
    let DirectorySyncMessage::BlockRangeResponseV1 {
        blocks, responder, ..
    } = decode_directory_sync_message(&body).unwrap()
    else {
        panic!("unexpected range response");
    };
    assert_eq!(responder, producer.public_key_bytes());
    assert_eq!(blocks.len(), 1);
    let descriptor_hash = blocks[0].commitments[0].descriptor_hash;
    let expected_commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&expected_descriptor).unwrap();
    assert_eq!(descriptor_hash, expected_commitment.descriptor_hash);

    let object_id = [0xb4; 16];
    let object_signing = directory_descriptor_objects_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &[descriptor_hash],
        &object_id,
        &requester_id,
        timestamp,
    );
    let object_request =
        encode_directory_sync_message(&DirectorySyncMessage::DescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            descriptor_hashes: vec![descriptor_hash],
            request_id: object_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&object_signing),
        })
        .unwrap();
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/descriptor-objects")
                .body(Body::from(object_request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let verified_objects = verify_descriptor_objects_response(
        &body,
        &object_id,
        &producer.public_key_bytes(),
        &[descriptor_hash],
        timestamp,
    )
    .unwrap();
    assert_eq!(verified_objects, vec![expected_descriptor.clone()]);
    let DirectorySyncMessage::DescriptorObjectsResponseV1 {
        descriptor_hashes,
        objects,
        ..
    } = decode_directory_sync_message(&body).unwrap()
    else {
        panic!("unexpected object response");
    };
    assert_eq!(descriptor_hashes, vec![descriptor_hash]);
    assert_eq!(objects, vec![expected_descriptor]);

    let proof_id = [0xb5; 16];
    let block_hash = blocks[0].hash();
    let proof_signing = directory_descriptor_inclusion_proof_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &block_hash,
        &descriptor_hash,
        &proof_id,
        &requester_id,
        timestamp,
    );
    let proof_request =
        encode_directory_sync_message(&DirectorySyncMessage::DescriptorInclusionProofRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            block_hash,
            descriptor_hash,
            request_id: proof_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&proof_signing),
        })
        .unwrap();
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/descriptor-inclusion-proof")
                .body(Body::from(proof_request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let DirectorySyncMessage::DescriptorInclusionProofResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        block_hash: response_block_hash,
        descriptor_hash: response_descriptor_hash,
        proof,
        signature,
    } = decode_directory_sync_message(&body).unwrap()
    else {
        panic!("unexpected proof response");
    };
    assert_eq!(request_id, proof_id);
    assert_eq!(response_block_hash, block_hash);
    assert_eq!(response_descriptor_hash, descriptor_hash);
    let response_signing = directory_descriptor_inclusion_proof_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        &response_block_hash,
        &response_descriptor_hash,
        &proof,
    );
    IdentityPublicKey::from_bytes(&responder)
        .unwrap()
        .verify(&response_signing, &signature)
        .unwrap();
    proof
        .verify_at(&chain_id, &responder, &response_block_hash, now_secs())
        .unwrap();
    assert_eq!(proof.commitment.descriptor_hash, response_descriptor_hash);
}

#[tokio::test]
async fn descriptor_inclusion_proof_remains_pinned_peer_only() {
    let (router, _, requester, expected_descriptor) = test_router(false, true, true);
    let timestamp = now_secs();
    let request_id = [0xb6; 16];
    let requester_id = requester.public_key_bytes();
    let descriptor_hash =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&expected_descriptor)
            .unwrap()
            .descriptor_hash;
    let block_hash = [0x51; 32];
    let signing_bytes = directory_descriptor_inclusion_proof_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &block_hash,
        &descriptor_hash,
        &request_id,
        &requester_id,
        timestamp,
    );
    let request =
        encode_directory_sync_message(&DirectorySyncMessage::DescriptorInclusionProofRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            block_hash,
            descriptor_hash,
            request_id,
            requester: requester_id,
            request_timestamp: timestamp,
            signature: requester.sign(&signing_bytes),
        })
        .unwrap();
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/descriptor-inclusion-proof")
                .body(Body::from(request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[test]
fn transport_page_accepts_eight_unique_blocks_and_stops_before_duplicate_objects() {
    let now = now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
    let mut blocks = Vec::new();
    let mut previous_hash = [0u8; 32];
    for offset in 1u8..=8 {
        let subject = IdentityKeyPair::from_bytes(&[offset; 32]).unwrap();
        let object = signed_descriptor(&subject, now);
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&object).unwrap();
        let block = DirectoryCommitmentBlockV1::new_signed(
            u64::from(offset),
            now + u64::from(offset),
            previous_hash,
            vec![commitment],
            &producer,
        )
        .unwrap();
        previous_hash = block.hash();
        blocks.push(block);
    }
    assert_eq!(bounded_directory_transport_blocks(blocks.clone()).len(), 8);

    let repeated_commitment = blocks[0].commitments[0];
    let duplicate = DirectoryCommitmentBlockV1::new_signed(
        2,
        blocks[0].header.timestamp + 1,
        blocks[0].hash(),
        vec![repeated_commitment],
        &producer,
    )
    .unwrap();
    assert_eq!(
        bounded_directory_transport_blocks(vec![blocks[0].clone(), duplicate]).len(),
        1
    );
}
