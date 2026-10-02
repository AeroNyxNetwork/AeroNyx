// [ARCH-SPLIT 2026-10-02]
// Tip, block-range, descriptor objects, and inclusion proof for this node's chain.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) async fn tip_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::TipRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID {
        return protocol_error(StatusCode::BAD_REQUEST, "wrong_chain");
    }
    let now = now_secs();
    let signing_bytes =
        directory_tip_request_signing_bytes(&chain_id, &request_id, &requester, request_timestamp);
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicMirror,
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

    let store = Arc::clone(&state.store);
    let audit = match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "producer_tip_audit",
        move || store.audited_tip(now),
    )
    .await
    {
        Ok(Ok(audit)) => audit,
        Ok(Err(error)) => return store_error_response(&error),
        Err(response) => return response,
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_tip_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        audit.tip_height,
        &audit.tip_hash,
        audit.tip_timestamp,
    );
    encoded_response(DirectorySyncMessage::TipResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        tip_height: audit.tip_height,
        tip_hash: audit.tip_hash,
        tip_timestamp: audit.tip_timestamp,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

pub(super) async fn block_range_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::BlockRangeRequestV1 {
        chain_id,
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
        || from_height == 0
        || limit == 0
        || limit > MAX_DIRECTORY_SYNC_BLOCKS_V1
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    let now = now_secs();
    let signing_bytes = directory_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicMirror,
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

    let store = Arc::clone(&state.store);
    let page = match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "producer_block_range_audit",
        move || store.audited_block_page(from_height, limit, now),
    )
    .await
    {
        Ok(Ok(page)) => page,
        Ok(Err(error)) => return store_error_response(&error),
        Err(response) => return response,
    };
    let blocks = bounded_directory_transport_blocks(page.blocks);
    let has_more = blocks
        .last()
        .is_some_and(|block| block.header.height < page.tip_height);
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
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
        "[DIRECTORY_CHAIN] Served authenticated bounded block page"
    );
    encoded_response(DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height: page.tip_height,
        tip_hash: page.tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

pub(super) async fn descriptor_objects_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::DescriptorObjectsRequestV1 {
        chain_id,
        descriptor_hashes,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let mut unique_hashes = descriptor_hashes.clone();
    unique_hashes.sort_unstable();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || descriptor_hashes.is_empty()
        || descriptor_hashes.len() > MAX_DIRECTORY_SYNC_OBJECTS_V1
        || descriptor_hashes.iter().any(|hash| *hash == [0u8; 32])
        || unique_hashes.windows(2).any(|pair| pair[0] == pair[1])
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_object_request");
    }
    let now = now_secs();
    let signing_bytes = directory_descriptor_objects_request_signing_bytes(
        &chain_id,
        &descriptor_hashes,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::VerifiedPublicMirror,
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

    let store = Arc::clone(&state.store);
    let requested_hashes = descriptor_hashes.clone();
    let objects = match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "producer_descriptor_objects_audit",
        move || store.audited_descriptor_objects(&requested_hashes, now),
    )
    .await
    {
        Ok(Ok(Some(objects))) => objects,
        Ok(Ok(None)) => return protocol_error(StatusCode::NOT_FOUND, "object_not_found"),
        Ok(Err(error)) => return store_error_response(&error),
        Err(response) => return response,
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_descriptor_objects_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &descriptor_hashes,
    );
    debug!(
        objects = objects.len(),
        "[DIRECTORY_CHAIN] Served authenticated descriptor objects"
    );
    encoded_response(DirectorySyncMessage::DescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

pub(super) async fn descriptor_inclusion_proof_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::DescriptorInclusionProofRequestV1 {
        chain_id,
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
        || block_hash == [0u8; 32]
        || descriptor_hash == [0u8; 32]
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_proof_request");
    }
    let now = now_secs();
    let signing_bytes = directory_descriptor_inclusion_proof_request_signing_bytes(
        &chain_id,
        &block_hash,
        &descriptor_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::PinnedAuthority,
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

    let store = Arc::clone(&state.store);
    let proof = match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "producer_descriptor_inclusion_proof_audit",
        move || store.audited_descriptor_inclusion_proof(&descriptor_hash, &block_hash, now),
    )
    .await
    {
        Ok(Ok(Some(proof))) => proof,
        Ok(Ok(None)) => return protocol_error(StatusCode::NOT_FOUND, "proof_not_found"),
        Ok(Err(error)) => return store_error_response(&error),
        Err(response) => return response,
    };
    let responder = state.identity.public_key_bytes();
    if proof
        .verify_at(&chain_id, &responder, &block_hash, now)
        .is_err()
        || proof.commitment.descriptor_hash != descriptor_hash
    {
        warn!("[DIRECTORY_CHAIN] Refused inconsistent descriptor inclusion proof");
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "proof_not_verified");
    }
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_descriptor_inclusion_proof_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        &block_hash,
        &descriptor_hash,
        &proof,
    );
    debug!(
        block_height = proof.block_header.height,
        "[DIRECTORY_CHAIN] Served authenticated descriptor inclusion proof"
    );
    encoded_response(DirectorySyncMessage::DescriptorInclusionProofResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        block_hash,
        descriptor_hash,
        proof,
        signature: state.identity.sign(&response_signing_bytes),
    })
}
