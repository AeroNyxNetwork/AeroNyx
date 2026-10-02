// Split from crates/aeronyx-server/src/api/memchain_peer.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn audited_authority_schedule_supersedes_legacy_runtime_pin() {
    // [AUTHORITY-SCHEDULE-RUNTIME 2026-08-14 by Codex] Announcement and
    // lease handlers share this resolver. Once a handover is durable, the
    // legacy bootstrap pin cannot continue authorising later heights.
    let now = now_secs();
    let root = IdentityKeyPair::generate();
    let next = IdentityKeyPair::generate();
    let storage = MemoryStorage::open(":memory:", None).unwrap();
    storage
        .configure_record_commitment_authority_root(Some(root.public_key_bytes()))
        .unwrap();
    storage.audit_record_commitment_chain().await.unwrap();
    let first_block = RecordCommitmentBlockV1::new_signed(
        1,
        now.saturating_sub(2),
        GENESIS_PREV_HASH,
        vec![[0x61; 32]],
        &root,
    );
    storage
        .append_record_commitment_block(&first_block, None)
        .await
        .unwrap();
    let handover = RecordCoordinatorHandoverV1::new_dual_signed(
        1,
        2,
        first_block.hash(),
        [0x62; 16],
        now.saturating_sub(1),
        &root,
        &next,
    );
    storage
        .persist_configured_record_coordinator_handover(&handover, now)
        .await
        .unwrap();

    assert_eq!(
        runtime_authorized_coordinator_for_height(&storage, Some(root.public_key_bytes()), 1)
            .await
            .unwrap(),
        Some(root.public_key_bytes())
    );
    assert_eq!(
        runtime_authorized_coordinator_for_height(&storage, Some(root.public_key_bytes()), 2)
            .await
            .unwrap(),
        Some(next.public_key_bytes())
    );
    assert_eq!(
        runtime_authorized_coordinator_for_next_height(&storage, Some(root.public_key_bytes()),)
            .await
            .unwrap(),
        Some(next.public_key_bytes())
    );
}

#[test]
fn follower_certificate_telemetry_requires_durable_persistence_for_success() {
    // [CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex] Both transport
    // paths must report an authenticated-but-deferred import identically;
    // only durable persistence may count as direct or carrier recovery.
    assert_eq!(
        follower_certificate_sync_disposition(
            CommitmentFollowerCertificateSource::Coordinator,
            true,
        ),
        RecordCommitmentCertificateSyncDisposition::Coordinator
    );
    assert_eq!(
        follower_certificate_sync_disposition(
            CommitmentFollowerCertificateSource::PinnedCarrier,
            true,
        ),
        RecordCommitmentCertificateSyncDisposition::CarrierRecovered
    );
    for source in [
        CommitmentFollowerCertificateSource::Coordinator,
        CommitmentFollowerCertificateSource::PinnedCarrier,
    ] {
        assert_eq!(
            follower_certificate_sync_disposition(source, false),
            RecordCommitmentCertificateSyncDisposition::VerifiedUnpersisted
        );
    }
}

#[tokio::test]
async fn block_range_endpoint_refuses_to_sign_an_unaudited_chain() {
    let now = now_secs();
    let responder_identity = Arc::new(IdentityKeyPair::generate());
    let requester_identity = IdentityKeyPair::generate();
    let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
    let peer_store = Arc::new(PeerStore::new());
    admit_peer(&peer_store, &requester_identity, None, now);

    let request_id = [0x92; 16];
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
    let response = build_memchain_peer_router(storage, peer_store, responder_identity)
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
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
