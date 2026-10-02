// [ARCH-SPLIT 2026-10-02]
// Checkpoint pull and the checkpoint HTTP handler.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Obtains and verifies one signed chain-checkpoint comparison from the pinned
/// coordinator. The response proves peer attestation, not network consensus.
///
/// # Errors
///
/// Returns a stable privacy-safe code when the local audited tip is
/// unavailable, the pinned peer cannot be reached, its response is invalid,
/// or durable checkpoint evidence cannot be stored.
pub async fn pull_record_commitment_checkpoint(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentCheckpointOutcome, String> {
    pull_record_commitment_checkpoint_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        false,
        CommitmentPeerDescriptorPolicy::CurrentOnly,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) async fn pull_record_commitment_checkpoint_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    track_trusted_witness_incidents: bool,
    descriptor_policy: CommitmentPeerDescriptorPolicy,
    endpoint_allowed: &F,
) -> Result<CommitmentCheckpointOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let coordinator = commitment_peer_descriptor(
        peer_store,
        coordinator_node_id,
        request_timestamp,
        descriptor_policy,
    )
    .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = coordinator
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("pinned_coordinator_unsafe_endpoint".to_string());
    }
    let url = commitment_checkpoint_url(endpoint)?;

    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("checkpoint_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "checkpoint_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let observed_at = now_secs();
    let outcome = verify_record_commitment_checkpoint(
        storage,
        &body,
        &request_id,
        coordinator_node_id,
        (known_tip_height, known_tip_hash),
        observed_at,
    )
    .await?;
    let persist_outcome = storage
        .persist_record_commitment_checkpoint_evidence_with_witness_policy(
            observed_at,
            outcome.relation.as_str(),
            outcome.local_tip_height,
            outcome.remote_tip_height,
            outcome.checkpoint_height,
            &outcome.evidence_digest,
            &body,
            track_trusted_witness_incidents,
        )
        .await
        .map_err(|_| "checkpoint_evidence_persist_failed".to_string())?;
    if persist_outcome == RecordCommitmentCheckpointEvidencePersistOutcome::EquivocationDetected {
        return Err("checkpoint_witness_equivocation".to_string());
    }
    Ok(outcome)
}

pub(super) async fn verify_record_commitment_checkpoint(
    storage: &MemoryStorage,
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    now: u64,
) -> Result<CommitmentCheckpointOutcome, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_checkpoint_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_checkpoint_frame")?;
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_checkpoint_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("checkpoint_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("checkpoint_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("checkpoint_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_checkpoint_response".to_string());
    }
    let response_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&response_signing_bytes, &signature))
        .map_err(|_| "invalid_checkpoint_signature".to_string())?;

    if tip_height == 0 && tip_hash != GENESIS_PREV_HASH {
        return Err("invalid_checkpoint_genesis".to_string());
    }
    let expected_checkpoint_height = local_tip.0.min(tip_height);
    if checkpoint_height != expected_checkpoint_height {
        return Err("checkpoint_height_mismatch".to_string());
    }
    if checkpoint_height == tip_height && checkpoint_hash != tip_hash {
        return Err("checkpoint_tip_inconsistent".to_string());
    }
    let (resolved_height, local_checkpoint_hash, _, _) = storage
        .record_commitment_chain_checkpoint(checkpoint_height)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if resolved_height != checkpoint_height {
        return Err("local_checkpoint_height_mismatch".to_string());
    }

    let relation = if local_checkpoint_hash != checkpoint_hash {
        CommitmentCheckpointRelation::Diverged
    } else if local_tip.0 == tip_height {
        CommitmentCheckpointRelation::Converged
    } else if local_tip.0 < tip_height {
        CommitmentCheckpointRelation::RemoteAhead
    } else {
        CommitmentCheckpointRelation::RemoteBehind
    };
    let evidence_digest: [u8; 32] = Sha256::digest(body).into();
    Ok(CommitmentCheckpointOutcome {
        relation,
        local_tip_height: local_tip.0,
        remote_tip_height: tip_height,
        checkpoint_height,
        evidence_digest,
    })
}

pub(super) async fn checkpoint_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordChainCheckpointRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || (known_tip_height == 0 && known_tip_hash != GENESIS_PREV_HASH)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_checkpoint_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if !coordinator_control_requester_is_admitted(&state, &requester, now) {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_chain_checkpoint_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let signature_valid = IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let (checkpoint_height, checkpoint_hash, tip_height, tip_hash) = match state
        .storage
        .record_commitment_chain_checkpoint(known_tip_height)
        .await
    {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited checkpoint proof");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified");
        }
    };
    let relation = if known_tip_height > tip_height {
        "served"
    } else if known_tip_hash != checkpoint_hash {
        "diverged"
    } else if known_tip_height == tip_height {
        "converged"
    } else {
        "remote_behind"
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_chain_checkpoint_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode checkpoint response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    // Count only a response that was successfully constructed. The inbound
    // request's relation/heights remain requester-controlled debug context and
    // cannot overwrite this node's outbound checkpoint evidence.
    state.storage.record_commitment_checkpoint_served(now);
    debug!(
        relation,
        checkpoint_height, tip_height, "[MEMCHAIN_BLOCK] Served authenticated chain checkpoint"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}
