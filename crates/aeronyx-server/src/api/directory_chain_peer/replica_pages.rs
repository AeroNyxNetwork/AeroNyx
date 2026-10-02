// [ARCH-SPLIT 2026-10-02]
// Audited non-authoritative replica pages and their HTTP handlers.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) async fn audited_replica_page_for_request(
    state: &DirectoryChainPeerState,
    producer: [u8; 32],
    from_height: u64,
    limit: u16,
    observed_at: u64,
) -> Result<DirectoryReplicaEvidencePage, Response> {
    let producer_is_pinned = state.pinned_peers.contains(&producer);
    if !producer_is_pinned && !state.allow_public_mirror_reads {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "replica_block_range_audit",
        move || {
            if producer_is_pinned {
                store.audited_evidence_page(&producer, from_height, limit, observed_at)
            } else {
                store.audited_mirror_evidence_page(&producer, from_height, limit, observed_at)
            }
        },
    )
    .await
    {
        Ok(Ok(page)) if page.tip_height > 0 => Ok(page),
        Ok(Ok(_)) => Err(protocol_error(StatusCode::NOT_FOUND, "replica_not_found")),
        Ok(Err(error)) => Err(replica_store_error_response(&error)),
        Err(response) => Err(response),
    }
}

pub(super) async fn audited_replica_objects_for_request(
    state: &DirectoryChainPeerState,
    producer: [u8; 32],
    descriptor_hashes: Vec<[u8; 32]>,
    observed_at: u64,
) -> Result<Vec<SignedNodeDescriptor>, Response> {
    let producer_is_pinned = state.pinned_peers.contains(&producer);
    if !producer_is_pinned && !state.allow_public_mirror_reads {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "replica_descriptor_objects_audit",
        move || {
            if producer_is_pinned {
                store.audited_evidence_descriptor_objects(
                    &producer,
                    &descriptor_hashes,
                    observed_at,
                )
            } else {
                store.audited_mirror_evidence_descriptor_objects(
                    &producer,
                    &descriptor_hashes,
                    observed_at,
                )
            }
        },
    )
    .await
    {
        Ok(Ok(Some(objects))) => Ok(objects),
        Ok(Ok(None)) => Err(protocol_error(
            StatusCode::NOT_FOUND,
            "replica_object_not_found",
        )),
        Ok(Err(error)) => Err(replica_store_error_response(&error)),
        Err(response) => Err(response),
    }
}

pub(super) async fn audited_replica_descriptor_proof_for_request(
    state: &DirectoryChainPeerState,
    producer: [u8; 32],
    descriptor_hash: [u8; 32],
    block_hash: [u8; 32],
    observed_at: u64,
) -> Result<DirectoryDescriptorInclusionProofV1, Response> {
    let producer_is_pinned = state.pinned_peers.contains(&producer);
    if !producer_is_pinned && !state.allow_public_mirror_reads {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "replica_descriptor_inclusion_proof_audit",
        move || {
            if producer_is_pinned {
                store.audited_evidence_descriptor_inclusion_proof(
                    &producer,
                    &descriptor_hash,
                    &block_hash,
                    observed_at,
                )
            } else {
                store.audited_mirror_evidence_descriptor_inclusion_proof(
                    &producer,
                    &descriptor_hash,
                    &block_hash,
                    observed_at,
                )
            }
        },
    )
    .await
    {
        Ok(Ok(Some(proof))) => Ok(proof),
        Ok(Ok(None)) => Err(protocol_error(
            StatusCode::NOT_FOUND,
            "replica_descriptor_proof_not_found",
        )),
        Ok(Err(error)) => Err(replica_store_error_response(&error)),
        Err(response) => Err(response),
    }
}

pub(super) async fn replica_block_range_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
        chain_id,
        producer,
        from_height,
        limit,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || producer == [0u8; 32]
        || producer == state.identity.public_key_bytes()
        || from_height == 0
        || limit == 0
        || limit > MAX_DIRECTORY_SYNC_BLOCKS_V1
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_range");
    }
    let now = now_secs();
    let signing_bytes = directory_replica_block_range_request_signing_bytes(
        &chain_id,
        &producer,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicRecovery,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    let page =
        match audited_replica_page_for_request(&state, producer, from_height, limit, now).await {
            Ok(page) => page,
            Err(response) => return response,
        };
    let blocks = bounded_directory_transport_blocks(page.blocks);
    let has_more = blocks
        .last()
        .is_some_and(|block| block.header.height < page.tip_height);
    let carrier = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_replica_block_range_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &blocks,
        has_more,
        page.tip_height,
        &page.tip_hash,
    );
    debug!(
        blocks = blocks.len(),
        has_more,
        tip_height = page.tip_height,
        "[DIRECTORY_CHAIN] Served audited replica evidence page"
    );
    encoded_response(DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        blocks,
        has_more,
        tip_height: page.tip_height,
        tip_hash: page.tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

pub(super) async fn replica_descriptor_objects_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
        chain_id,
        producer,
        descriptor_hashes,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let unique_hashes = descriptor_hashes.iter().copied().collect::<HashSet<_>>();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || producer == [0u8; 32]
        || producer == state.identity.public_key_bytes()
        || descriptor_hashes.is_empty()
        || descriptor_hashes.len() > MAX_DIRECTORY_SYNC_OBJECTS_V1
        || unique_hashes.len() != descriptor_hashes.len()
        || descriptor_hashes.iter().any(|hash| *hash == [0u8; 32])
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_object_request");
    }
    let now = now_secs();
    let signing_bytes = directory_replica_descriptor_objects_request_signing_bytes(
        &chain_id,
        &producer,
        &descriptor_hashes,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicRecovery,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    let objects =
        match audited_replica_objects_for_request(&state, producer, descriptor_hashes.clone(), now)
            .await
        {
            Ok(objects) => objects,
            Err(response) => return response,
        };
    let carrier = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_replica_descriptor_objects_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &descriptor_hashes,
    );
    debug!(
        objects = objects.len(),
        "[DIRECTORY_CHAIN] Served audited replica descriptor objects"
    );
    encoded_response(DirectorySyncMessage::ReplicaDescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

pub(super) async fn replica_descriptor_inclusion_proof_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ReplicaDescriptorInclusionProofRequestV1 {
        chain_id,
        producer,
        block_hash,
        descriptor_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || producer == [0u8; 32]
        || producer == state.identity.public_key_bytes()
        || block_hash == [0u8; 32]
        || descriptor_hash == [0u8; 32]
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_proof_request");
    }
    let now = now_secs();
    let signing_bytes = directory_replica_descriptor_inclusion_proof_request_signing_bytes(
        &chain_id,
        &producer,
        &block_hash,
        &descriptor_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicRecovery,
        requester,
        request_id,
        request_timestamp,
        &signing_bytes,
        &signature,
        now,
    )
    .await
    {
        return response;
    }
    let proof = match audited_replica_descriptor_proof_for_request(
        &state,
        producer,
        descriptor_hash,
        block_hash,
        now,
    )
    .await
    {
        Ok(proof) => proof,
        Err(response) => return response,
    };
    let carrier = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes =
        directory_replica_descriptor_inclusion_proof_response_signing_bytes(
            &chain_id,
            &request_id,
            &producer,
            &carrier,
            response_timestamp,
            &block_hash,
            &descriptor_hash,
            &proof,
        );
    debug!(
        block_height = proof.block_header.height,
        "[DIRECTORY_CHAIN] Served audited replica descriptor inclusion proof"
    );
    encoded_response(
        DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
            chain_id,
            request_id,
            producer,
            carrier,
            response_timestamp,
            block_hash,
            descriptor_hash,
            proof,
            signature: state.identity.sign(&response_signing_bytes),
        },
    )
}
