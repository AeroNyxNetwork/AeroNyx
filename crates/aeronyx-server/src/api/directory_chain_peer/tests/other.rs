// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn runtime_owns_one_audit_gate_and_isolates_independent_nodes() {
    // [DIRECTORY-AUDIT-OWNERSHIP 2026-08-12 by Codex] Listener routers
    // sharing one node runtime must compete for one permit, while another
    // in-process node runtime must not inherit that node's admission state.
    let first_runtime = DirectoryReplicaSyncRuntime::default();
    let first_public = first_runtime.directory_audit_admission();
    let first_local = first_runtime.directory_audit_admission();
    let second_node = DirectoryReplicaSyncRuntime::default().directory_audit_admission();

    assert!(Arc::ptr_eq(&first_public, &first_local));
    assert!(!Arc::ptr_eq(&first_public, &second_node));
    assert_eq!(first_public.available_permits(), 1);
    assert_eq!(second_node.available_permits(), 1);
}

#[test]
fn request_guard_enforces_global_permissionless_budget() {
    let mut guard = DirectoryPeerRequestGuard::default();
    let now = 1_700_000_000;
    for index in 0..MAX_DIRECTORY_REQUESTS_GLOBAL_PER_MINUTE {
        let mut requester = [0u8; 32];
        requester[..4].copy_from_slice(&index.to_le_bytes());
        requester[31] = 1;
        let mut request_id = [0u8; 16];
        request_id[..4].copy_from_slice(&index.to_le_bytes());
        assert!(guard.admit(requester, request_id, now));
    }
    assert!(!guard.admit([0xff; 32], [0xff; 16], now));
    assert!(guard.admit([0xfe; 32], [0xfe; 16], now + 60));
}

#[tokio::test]
async fn mirror_mode_disabled_preserves_pinned_only_read_admission() {
    let (router, _, requester, _) = test_router(false, true, false);
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/tip")
                .body(Body::from(tip_request(&requester, [0xba; 16])))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn verified_public_peer_can_recover_only_registered_mirror_evidence() {
    let (router, carrier, requester, producer, expected_object) =
        carrier_test_router_with_access(CarrierTestPolicy::PublicMirror);
    let request_id = [0xce; 16];
    let producer_id = producer.public_key_bytes();
    let request = replica_range_request(&requester, &producer_id, request_id);
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-block-range")
                .body(Body::from(request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let (blocks, has_more, tip_height, _) = verify_replica_block_range_response(
        &body,
        &request_id,
        &producer_id,
        &carrier.public_key_bytes(),
        1,
        now_secs(),
    )
    .unwrap();
    assert_eq!(blocks.len(), 1);
    assert!(!has_more);
    assert_eq!(tip_height, 1);
    let block_hash = blocks[0].hash();

    // [MIRROR-CARRIER 2026-07-24 by Codex] A valid request beyond this
    // carrier's retained producer tip is retryable availability, not a
    // malformed-frame response that would abort the bounded carrier list.
    let unavailable_response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-block-range")
                .body(Body::from(replica_range_request_from_height(
                    &requester,
                    &producer_id,
                    3,
                    [0xcc; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unavailable_response.status(), StatusCode::NOT_FOUND);
    let unavailable_body = to_bytes(unavailable_response.into_body(), 128)
        .await
        .unwrap();
    assert_eq!(&unavailable_body[..], b"replica_range_not_retained");

    let descriptor_hash = blocks[0].commitments[0].descriptor_hash;
    let object_id = [0xcd; 16];
    let object_timestamp = now_secs();
    let requester_id = requester.public_key_bytes();
    let object_signing = directory_replica_descriptor_objects_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &producer_id,
        &[descriptor_hash],
        &object_id,
        &requester_id,
        object_timestamp,
    );
    let object_request =
        encode_directory_sync_message(&DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: producer_id,
            descriptor_hashes: vec![descriptor_hash],
            request_id: object_id,
            requester: requester_id,
            request_timestamp: object_timestamp,
            signature: requester.sign(&object_signing),
        })
        .unwrap();
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-descriptor-objects")
                .body(Body::from(object_request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    assert_eq!(
        verify_replica_descriptor_objects_response(
            &body,
            &object_id,
            &producer_id,
            &carrier.public_key_bytes(),
            &[descriptor_hash],
            object_timestamp,
        )
        .unwrap(),
        vec![expected_object.clone()]
    );

    let proof_id = [0xcb; 16];
    let response = router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-descriptor-inclusion-proof")
                .body(Body::from(replica_descriptor_proof_request(
                    &requester,
                    &producer_id,
                    &block_hash,
                    &descriptor_hash,
                    proof_id,
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let proof = verify_replica_descriptor_proof_response(
        &body,
        &proof_id,
        &producer_id,
        &carrier.public_key_bytes(),
        &block_hash,
        &descriptor_hash,
    );
    assert_eq!(proof.descriptor, expected_object);

    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-descriptor-inclusion-proof")
                .body(Body::from(replica_descriptor_proof_request(
                    &requester,
                    &producer_id,
                    &[0x35; 32],
                    &descriptor_hash,
                    [0xc8; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn verified_public_recovery_refuses_unregistered_or_disabled_namespaces() {
    let (unregistered_router, _, requester, producer, _) =
        carrier_test_router_with_access(CarrierTestPolicy::PublicWithoutMirror);
    let producer_id = producer.public_key_bytes();
    let response = unregistered_router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-block-range")
                .body(Body::from(replica_range_request(
                    &requester,
                    &producer_id,
                    [0xcf; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let response = unregistered_router
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-descriptor-inclusion-proof")
                .body(Body::from(replica_descriptor_proof_request(
                    &requester,
                    &producer_id,
                    &[0x31; 32],
                    &[0x32; 32],
                    [0xc9; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let (disabled_router, _, requester, producer, _) =
        carrier_test_router_with_access(CarrierTestPolicy::PublicMirrorDisabled);
    let response = disabled_router
        .clone()
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-block-range")
                .body(Body::from(replica_range_request(
                    &requester,
                    &producer.public_key_bytes(),
                    [0xd0; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    let response = disabled_router
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-descriptor-inclusion-proof")
                .body(Body::from(replica_descriptor_proof_request(
                    &requester,
                    &producer.public_key_bytes(),
                    &[0x33; 32],
                    &[0x34; 32],
                    [0xca; 16],
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn carrier_routes_export_only_audited_producer_bound_evidence() {
    let (router, carrier, requester, producer, expected_object) = carrier_test_router();
    let timestamp = now_secs();
    let requester_id = requester.public_key_bytes();
    let producer_id = producer.public_key_bytes();
    let range_id = [0xd6; 16];
    let range_signing = directory_replica_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &producer_id,
        1,
        1,
        &range_id,
        &requester_id,
        timestamp,
    );
    let range_request =
        encode_directory_sync_message(&DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: producer_id,
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
            Request::post("/api/discovery/peer/directory/replica-block-range")
                .body(Body::from(range_request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let (blocks, has_more, tip_height, _) = verify_replica_block_range_response(
        &body,
        &range_id,
        &producer_id,
        &carrier.public_key_bytes(),
        1,
        timestamp,
    )
    .unwrap();
    assert_eq!(blocks.len(), 1);
    assert!(!has_more);
    assert_eq!(tip_height, 1);
    let block_hash = blocks[0].hash();
    let descriptor_hash = blocks[0].commitments[0].descriptor_hash;

    let object_id = [0xd7; 16];
    let object_signing = directory_replica_descriptor_objects_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &producer_id,
        &[descriptor_hash],
        &object_id,
        &requester_id,
        timestamp,
    );
    let object_request =
        encode_directory_sync_message(&DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: producer_id,
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
            Request::post("/api/discovery/peer/directory/replica-descriptor-objects")
                .body(Body::from(object_request))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let objects = verify_replica_descriptor_objects_response(
        &body,
        &object_id,
        &producer_id,
        &carrier.public_key_bytes(),
        &[descriptor_hash],
        timestamp,
    )
    .unwrap();
    assert_eq!(objects, vec![expected_object.clone()]);

    let proof_id = [0xd8; 16];
    let response = router
        .oneshot(
            Request::post("/api/discovery/peer/directory/replica-descriptor-inclusion-proof")
                .body(Body::from(replica_descriptor_proof_request(
                    &requester,
                    &producer_id,
                    &block_hash,
                    &descriptor_hash,
                    proof_id,
                )))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 512 * 1024).await.unwrap();
    let proof = verify_replica_descriptor_proof_response(
        &body,
        &proof_id,
        &producer_id,
        &carrier.public_key_bytes(),
        &block_hash,
        &descriptor_hash,
    );
    assert_eq!(proof.descriptor, expected_object);
}
