// [ARCH-SPLIT 2026-10-02]
// Observation witness requests, carrier fallback, and certificate fetch.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

/// Whether one bounded witness catch-up batch may attempt this checkpoint.
///
/// The strict sequence comparison prevents a partially available witness set
/// from receiving repeated requests for the same checkpoint in one round.
pub(super) const fn should_attempt_observation_witness_catch_up(
    checkpoints_attempted: usize,
    previous_sequence: Option<u64>,
    candidate_sequence: u64,
) -> bool {
    checkpoints_attempted < DIRECTORY_OBSERVATION_WITNESS_CATCH_UP_CHECKPOINTS_PER_ROUND
        && match previous_sequence {
            Some(previous) => candidate_sequence > previous,
            None => true,
        }
}

pub(super) async fn request_observation_policy_anchor(
    store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness: &[u8; 32],
    client: &reqwest::Client,
    capability_cache: &DirectoryWitnessCapabilityCache,
    anchor: crate::services::directory_replica::DirectoryObservationWitnessPolicyAnchor,
) -> DirectoryObservationWitnessOutcome {
    let request_timestamp = unix_now_secs();
    let Some(descriptor) = peer_store.get_valid(witness, request_timestamp) else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    if !commitment_peer_endpoint_is_public(endpoint) {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    }
    let descriptor_sequence = descriptor.sequence();
    if !capability_cache.should_attempt(witness, descriptor_sequence) {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    }
    let Ok(url) = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/observation-policy-anchor",
    ) else {
        return DirectoryObservationWitnessOutcome::PeerUnavailable;
    };
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_policy_anchor_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester,
        request_timestamp,
        anchor.epoch,
        &anchor.previous_policy_digest,
        &anchor.policy_digest,
    );
    let request = DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester,
        request_timestamp,
        policy_epoch: anchor.epoch,
        previous_policy_digest: anchor.previous_policy_digest,
        policy_digest: anchor.policy_digest,
        signature: identity.sign(&signing_bytes),
    };
    let Ok(frame) = encode_directory_sync_message(&request) else {
        return DirectoryObservationWitnessOutcome::VerificationFailure;
    };
    let response = match post_directory_frame_typed(client, url, frame).await {
        Ok(response) => {
            capability_cache.record_supported(witness);
            response
        }
        Err(error) if error.witness_capability_unavailable() => {
            capability_cache.record_unsupported(*witness, descriptor_sequence);
            return DirectoryObservationWitnessOutcome::PeerUnavailable;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::TransportFailure,
    };
    let verified = match verify_observation_policy_anchor_response(
        &response,
        &request_id,
        &requester,
        witness,
        request_timestamp,
        anchor.epoch,
        &anchor.policy_digest,
    ) {
        Ok(response) => response,
        Err(reason) if reason == "observation_policy_anchor_rollback" => {
            return DirectoryObservationWitnessOutcome::EvidenceConflict;
        }
        Err(reason) if reason == "observation_policy_anchor_conflict" => {
            return DirectoryObservationWitnessOutcome::EvidenceConflict;
        }
        Err(reason) if reason == "observation_policy_anchor_history_gap" => {
            return DirectoryObservationWitnessOutcome::EvidenceUnavailable;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::VerificationFailure,
    };
    let durable = tokio::task::spawn_blocking(move || {
        store.persist_observation_witness_policy_anchor_receipt(&verified, unix_now_secs())
    })
    .await
    .is_ok_and(|result| result.is_ok());
    if durable {
        DirectoryObservationWitnessOutcome::Accepted
    } else {
        DirectoryObservationWitnessOutcome::PersistenceFailure
    }
}

pub(crate) fn verify_observation_policy_anchor_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_observer: &[u8; 32],
    expected_witness: &[u8; 32],
    request_timestamp: u64,
    expected_policy_epoch: u64,
    expected_policy_digest: &[u8; 32],
) -> Result<DirectorySyncMessage, String> {
    let response = decode_directory_sync_message(frame)
        .map_err(|_| "observation_policy_anchor_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&response)
        .map_err(|_| "observation_policy_anchor_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("observation_policy_anchor_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
        chain_id,
        request_id,
        observer,
        policy_epoch,
        policy_digest,
        responder,
        response_timestamp,
        outcome,
        signature,
    } = &response
    else {
        return Err("observation_policy_anchor_response_unexpected_message".to_string());
    };
    if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != expected_request_id
        || observer != expected_observer
        || responder != expected_witness
        || *policy_epoch != expected_policy_epoch
        || policy_digest != expected_policy_digest
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || ![
            DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
            DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
            DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
        ]
        .contains(outcome)
    {
        return Err("observation_policy_anchor_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_policy_anchor_response_signing_bytes(
        chain_id,
        request_id,
        observer,
        *policy_epoch,
        policy_digest,
        responder,
        *response_timestamp,
        *outcome,
    );
    IdentityPublicKey::from_bytes(responder)
        .and_then(|key| key.verify(&signing_bytes, signature))
        .map_err(|_| "observation_policy_anchor_response_invalid_signature".to_string())?;
    match *outcome {
        DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1 => Ok(response),
        DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1 => {
            Err("observation_policy_anchor_rollback".to_string())
        }
        DIRECTORY_POLICY_ANCHOR_CONFLICT_V1 => {
            Err("observation_policy_anchor_conflict".to_string())
        }
        DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1 => {
            Err("observation_policy_anchor_history_gap".to_string())
        }
        _ => Err("observation_policy_anchor_response_outcome_invalid".to_string()),
    }
}

pub(super) async fn request_observation_checkpoint_witness(
    store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness: &[u8; 32],
    eligible_carriers: &[[u8; 32]],
    client: &reqwest::Client,
    capability_cache: &DirectoryWitnessCapabilityCache,
    carrier_capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    runtime: &DirectoryReplicaSyncRuntime,
    checkpoint: DirectoryObservationCheckpointV1,
) -> DirectoryObservationWitnessOutcome {
    let requester = identity.public_key_bytes();
    let request_timestamp = unix_now_secs();
    let mut direct_transport_failed = false;
    let direct = peer_store
        .get_valid(witness, request_timestamp)
        .and_then(|descriptor| {
            let endpoint = descriptor.descriptor.public_endpoint.as_deref()?;
            if !commitment_peer_endpoint_is_public(endpoint) {
                return None;
            }
            Some((descriptor.sequence(), endpoint.to_string()))
        });
    if let Some((descriptor_sequence, endpoint)) = direct {
        if !capability_cache.should_attempt(witness, descriptor_sequence) {
            debug!(
                reason = "directory_observation_witness_capability_cached_unavailable",
                "[DIRECTORY_REPLICA] Witness request skipped for unchanged signed descriptor"
            );
            return DirectoryObservationWitnessOutcome::PeerUnavailable;
        }
        let Ok(url) = commitment_peer_url(
            &endpoint,
            "/api/discovery/peer/directory/observation-checkpoint-witness",
        ) else {
            return DirectoryObservationWitnessOutcome::PeerUnavailable;
        };
        let request = match build_observation_witness_request(identity, checkpoint.clone()) {
            Ok(request) => request,
            Err(outcome) => return outcome,
        };
        match post_directory_frame_typed(client, url, request.frame.clone()).await {
            Ok(response) => {
                capability_cache.record_supported(witness);
                return verify_and_persist_observation_witness_response(
                    store, response, &request, witness,
                )
                .await;
            }
            Err(error) if error.witness_capability_unavailable() => {
                capability_cache.record_unsupported(*witness, descriptor_sequence);
                debug!(
                    reason = "directory_observation_witness_capability_unavailable",
                    "[DIRECTORY_REPLICA] Peer descriptor does not currently expose witness service"
                );
                return DirectoryObservationWitnessOutcome::PeerUnavailable;
            }
            Err(error) if observation_witness_failure_allows_carrier(error) => {
                direct_transport_failed = true;
            }
            Err(DirectoryFramePostError::Response(_)) => {
                return DirectoryObservationWitnessOutcome::VerificationFailure;
            }
            Err(_) => return DirectoryObservationWitnessOutcome::TransportFailure,
        }
    }

    // [WITNESS-CARRIER 2026-07-26 by Codex] Direct endpoint absence and
    // availability-only failures may use at most two explicitly advertised
    // carriers. Carriers transport a fresh exact inner request; they cannot
    // sign the witness receipt or change its expected responder.
    let carrier_selection = directory_observation_witness_recovery_carriers(
        peer_store,
        carrier_capability_cache,
        witness,
        &requester,
        eligible_carriers,
        request_timestamp,
    );
    runtime.record_observation_witness_recovery_selection(
        carrier_selection.candidate_count,
        carrier_selection.routeable_candidate_count,
        carrier_selection.capability_cached_unavailable_count,
        u64::try_from(carrier_selection.carriers.len()).unwrap_or(u64::MAX),
        request_timestamp,
    );
    if carrier_selection.carriers.is_empty() {
        runtime.record_observation_witness_recovery_exhausted(request_timestamp);
        return observation_witness_unavailable_recovery_outcome(direct_transport_failed);
    }
    for carrier in carrier_selection
        .carriers
        .into_iter()
        .take(DIRECTORY_OBSERVATION_WITNESS_RECOVERY_MAX_CARRIERS)
    {
        let request = match build_observation_witness_request(identity, checkpoint.clone()) {
            Ok(request) => request,
            Err(outcome) => return outcome,
        };
        let response = match request_observation_witness_via_carrier(
            peer_store, identity, witness, carrier, client, &request,
        )
        .await
        {
            Ok(response) => {
                carrier_capability_cache.record_supported(&carrier.node_id);
                runtime.record_observation_witness_recovery_attempt(true, false, request_timestamp);
                response
            }
            Err(error) if error.witness_capability_unavailable() => {
                carrier_capability_cache
                    .record_unsupported(carrier.node_id, carrier.descriptor_sequence);
                runtime.record_observation_witness_recovery_attempt(false, true, request_timestamp);
                continue;
            }
            Err(DirectoryFramePostError::HttpStatus {
                peer_code: Some(DirectoryPeerErrorCode::WitnessTargetUnavailable),
                ..
            })
            | Err(DirectoryFramePostError::Transport(_)) => {
                runtime.record_observation_witness_recovery_attempt(
                    false,
                    false,
                    request_timestamp,
                );
                continue;
            }
            Err(DirectoryFramePostError::HttpStatus {
                peer_code: Some(DirectoryPeerErrorCode::WitnessTargetCapabilityUnavailable),
                ..
            }) => {
                if let Some(descriptor) = peer_store.get_valid(witness, unix_now_secs()) {
                    capability_cache.record_unsupported(*witness, descriptor.sequence());
                }
                runtime.record_observation_witness_recovery_attempt(false, true, request_timestamp);
                runtime.record_observation_witness_recovery_exhausted(request_timestamp);
                return DirectoryObservationWitnessOutcome::PeerUnavailable;
            }
            Err(DirectoryFramePostError::HttpStatus {
                peer_code:
                    Some(
                        DirectoryPeerErrorCode::WitnessTargetRejected
                        | DirectoryPeerErrorCode::WitnessTargetInvalidResponse,
                    ),
                ..
            })
            | Err(DirectoryFramePostError::Response(_)) => {
                runtime.record_observation_witness_recovery_failed_closed(request_timestamp);
                return DirectoryObservationWitnessOutcome::VerificationFailure;
            }
            Err(error) if observation_witness_carrier_failure_allows_next(error) => {
                runtime.record_observation_witness_recovery_attempt(
                    false,
                    false,
                    request_timestamp,
                );
                continue;
            }
            Err(_) => {
                runtime.record_observation_witness_recovery_failed_closed(request_timestamp);
                return DirectoryObservationWitnessOutcome::VerificationFailure;
            }
        };
        return verify_and_persist_observation_witness_response(store, response, &request, witness)
            .await;
    }
    runtime.record_observation_witness_recovery_exhausted(request_timestamp);
    DirectoryObservationWitnessOutcome::TransportFailure
}

pub(super) fn build_observation_witness_request(
    identity: &IdentityKeyPair,
    checkpoint: DirectoryObservationCheckpointV1,
) -> Result<ObservationWitnessRequest, DirectoryObservationWitnessOutcome> {
    let request_timestamp = unix_now_secs();
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let checkpoint_sequence = checkpoint.sequence;
    let checkpoint_hash = checkpoint.hash();
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester,
        request_timestamp,
        &checkpoint_hash,
    );
    let message = DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester,
        request_timestamp,
        checkpoint,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&message)
        .map_err(|_| DirectoryObservationWitnessOutcome::VerificationFailure)?;
    Ok(ObservationWitnessRequest {
        request_id,
        requester,
        request_timestamp,
        checkpoint_sequence,
        checkpoint_hash,
        frame,
    })
}

pub(super) const fn observation_witness_failure_allows_carrier(
    error: DirectoryFramePostError,
) -> bool {
    match error {
        DirectoryFramePostError::Transport(_) => true,
        DirectoryFramePostError::HttpStatus {
            status,
            peer_code: None,
        } => matches!(status, 408 | 429 | 500 | 502 | 503 | 504),
        DirectoryFramePostError::HttpStatus {
            peer_code: Some(DirectoryPeerErrorCode::WitnessTargetUnavailable),
            ..
        } => true,
        _ => false,
    }
}

/// [WITNESS-CARRIER 2026-07-26 by Codex] Preserves the pre-recovery outcome
/// distinction when no carrier can be attempted.
pub(super) const fn observation_witness_unavailable_recovery_outcome(
    direct_transport_failed: bool,
) -> DirectoryObservationWitnessOutcome {
    if direct_transport_failed {
        DirectoryObservationWitnessOutcome::TransportFailure
    } else {
        DirectoryObservationWitnessOutcome::PeerUnavailable
    }
}

pub(super) const fn observation_witness_carrier_failure_allows_next(
    error: DirectoryFramePostError,
) -> bool {
    match error {
        DirectoryFramePostError::Transport(_) => true,
        DirectoryFramePostError::HttpStatus {
            status,
            peer_code: None,
        } => matches!(status, 403 | 408 | 429 | 500 | 502 | 503 | 504),
        DirectoryFramePostError::HttpStatus {
            peer_code: Some(DirectoryPeerErrorCode::WitnessTargetUnavailable),
            ..
        } => true,
        _ => false,
    }
}

pub(super) async fn request_observation_witness_via_carrier(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness: &[u8; 32],
    carrier: DirectoryMirrorRecoveryCarrier,
    client: &reqwest::Client,
    request: &ObservationWitnessRequest,
) -> Result<Vec<u8>, DirectoryFramePostError> {
    let request_timestamp = unix_now_secs();
    let url = directory_observation_witness_carrier_url(peer_store, &carrier, request_timestamp)
        .map_err(|_| DirectoryFramePostError::Transport(DirectoryTransportFailure::Preflight))?;
    let mut carrier_request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut carrier_request_id);
    let request_sha256: [u8; 32] = Sha256::digest(&request.frame).into();
    let frame_bytes = u64::try_from(request.frame.len()).unwrap_or(u64::MAX);
    let signing_bytes = directory_observation_witness_carrier_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &carrier_request_id,
        &request.requester,
        request_timestamp,
        witness,
        &request_sha256,
        frame_bytes,
    );
    let message = DirectorySyncMessage::ObservationCheckpointWitnessCarrierRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id: carrier_request_id,
        requester: request.requester,
        request_timestamp,
        witness: *witness,
        witness_request_sha256: request_sha256,
        witness_request_frame: request.frame.clone(),
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&message)
        .map_err(|_| DirectoryFramePostError::Response(BoundedHttpResponseError::BodyRead))?;
    let response = post_directory_frame_typed_with_response_limit(
        client,
        url,
        frame,
        MAX_DIRECTORY_OBSERVATION_WITNESS_CARRIER_RESPONSE_BODY_BYTES,
    )
    .await?;
    verify_observation_witness_carrier_response(
        &response,
        &carrier_request_id,
        &request.requester,
        witness,
        &carrier.node_id,
        request_timestamp,
        &request_sha256,
    )
    .map_err(|_| DirectoryFramePostError::Response(BoundedHttpResponseError::BodyRead))
}

pub(super) fn verify_observation_witness_carrier_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_requester: &[u8; 32],
    expected_witness: &[u8; 32],
    expected_carrier: &[u8; 32],
    request_timestamp: u64,
    expected_request_sha256: &[u8; 32],
) -> Result<Vec<u8>, String> {
    let response = decode_directory_sync_message(frame)
        .map_err(|_| "observation_witness_carrier_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&response)
        .map_err(|_| "observation_witness_carrier_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("observation_witness_carrier_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
        chain_id,
        request_id,
        requester,
        witness,
        carrier,
        response_timestamp,
        witness_request_sha256,
        witness_response_sha256,
        witness_response_frame,
        signature,
    } = response
    else {
        return Err("observation_witness_carrier_response_unexpected_message".to_string());
    };
    let actual_response_sha256: [u8; 32] = Sha256::digest(&witness_response_frame).into();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || requester != *expected_requester
        || witness != *expected_witness
        || carrier != *expected_carrier
        || witness_request_sha256 != *expected_request_sha256
        || witness_response_sha256 == [0u8; 32]
        || witness_response_sha256 != actual_response_sha256
        || witness_response_frame.is_empty()
        || witness_response_frame.len()
            > MAX_DIRECTORY_OBSERVATION_WITNESS_CARRIER_INNER_RESPONSE_BODY_BYTES
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
    {
        return Err("observation_witness_carrier_response_contract_mismatch".to_string());
    }
    let frame_bytes = u64::try_from(witness_response_frame.len()).unwrap_or(u64::MAX);
    let signing_bytes = directory_observation_witness_carrier_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        &witness,
        &carrier,
        response_timestamp,
        &witness_request_sha256,
        &witness_response_sha256,
        frame_bytes,
    );
    IdentityPublicKey::from_bytes(&carrier)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "observation_witness_carrier_response_invalid_signature".to_string())?;
    Ok(witness_response_frame)
}

pub(super) async fn verify_and_persist_observation_witness_response(
    store: Arc<DirectoryReplicaStore>,
    response: Vec<u8>,
    request: &ObservationWitnessRequest,
    witness: &[u8; 32],
) -> DirectoryObservationWitnessOutcome {
    let verified = match verify_observation_witness_response(
        &response,
        &request.request_id,
        &request.requester,
        witness,
        request.request_timestamp,
        request.checkpoint_sequence,
        &request.checkpoint_hash,
    ) {
        Ok(verified) => verified,
        Err(reason) if reason == "observation_witness_evidence_unavailable" => {
            return DirectoryObservationWitnessOutcome::EvidenceUnavailable;
        }
        Err(reason) if reason == "observation_witness_evidence_conflict" => {
            return DirectoryObservationWitnessOutcome::EvidenceConflict;
        }
        Err(_) => return DirectoryObservationWitnessOutcome::VerificationFailure,
    };
    let durable = tokio::task::spawn_blocking(move || {
        store.persist_observation_checkpoint_witness(&verified, unix_now_secs())
    })
    .await
    .is_ok_and(|result| result.is_ok());
    if durable {
        DirectoryObservationWitnessOutcome::Accepted
    } else {
        DirectoryObservationWitnessOutcome::PersistenceFailure
    }
}

pub(super) fn witness_outcome_count(
    outcomes: &[DirectoryObservationWitnessOutcome],
    expected: DirectoryObservationWitnessOutcome,
) -> usize {
    outcomes
        .iter()
        .filter(|outcome| **outcome == expected)
        .count()
}

pub(crate) fn verify_observation_witness_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_observer: &[u8; 32],
    expected_witness: &[u8; 32],
    request_timestamp: u64,
    expected_checkpoint_sequence: u64,
    expected_checkpoint_hash: &[u8; 32],
) -> Result<DirectorySyncMessage, String> {
    let response = decode_directory_sync_message(frame)
        .map_err(|_| "observation_witness_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&response)
        .map_err(|_| "observation_witness_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("observation_witness_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id,
        request_id,
        observer,
        checkpoint_sequence,
        checkpoint_hash,
        responder,
        response_timestamp,
        outcome,
        signature,
    } = &response
    else {
        return Err("observation_witness_response_unexpected_message".to_string());
    };
    if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != expected_request_id
        || observer != expected_observer
        || responder != expected_witness
        || *checkpoint_sequence != expected_checkpoint_sequence
        || checkpoint_hash != expected_checkpoint_hash
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || ![
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
        ]
        .contains(outcome)
    {
        return Err("observation_witness_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        chain_id,
        request_id,
        observer,
        *checkpoint_sequence,
        checkpoint_hash,
        responder,
        *response_timestamp,
        *outcome,
    );
    IdentityPublicKey::from_bytes(responder)
        .and_then(|key| key.verify(&signing_bytes, signature))
        .map_err(|_| "observation_witness_response_invalid_signature".to_string())?;
    match *outcome {
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1 => Ok(response),
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1 => {
            Err("observation_witness_evidence_unavailable".to_string())
        }
        DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1 => {
            Err("observation_witness_evidence_conflict".to_string())
        }
        _ => Err("observation_witness_response_outcome_invalid".to_string()),
    }
}

pub(super) fn directory_observation_witness_recovery_carriers(
    peer_store: &PeerStore,
    capability_cache: &DirectoryMirrorCarrierCapabilityCache,
    witness: &[u8; 32],
    requester: &[u8; 32],
    eligible_carriers: &[[u8; 32]],
    now: u64,
) -> DirectoryMirrorRecoveryCarrierSelection {
    directory_mirror_recovery_carriers_with_policy(
        peer_store,
        capability_cache,
        witness,
        requester,
        now,
        true,
        Some(eligible_carriers),
    )
}

pub(super) fn directory_observation_witness_carrier_url(
    peer_store: &PeerStore,
    carrier: &DirectoryMirrorRecoveryCarrier,
    request_timestamp: u64,
) -> Result<reqwest::Url, String> {
    let descriptor = peer_store
        .get_valid(&carrier.node_id, request_timestamp)
        .ok_or_else(|| "directory_witness_carrier_unavailable".to_string())?;
    if descriptor.sequence() != carrier.descriptor_sequence {
        return Err("directory_witness_carrier_descriptor_changed".to_string());
    }
    if !descriptor.descriptor.policy.public_discovery
        || !descriptor
            .descriptor
            .capabilities
            .contains(&NodeCapability::DirectoryMirrorCarrier)
    {
        return Err("directory_witness_carrier_not_advertised".to_string());
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_witness_carrier_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_witness_carrier_unsafe_endpoint".to_string());
    }
    commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/observation-checkpoint-witness-carrier",
    )
    .map_err(|_| "directory_witness_carrier_invalid_endpoint".to_string())
}

/// Fetches one portable observation certificate from an exact pinned source.
///
/// The returned value has authenticated transport provenance and exact byte
/// integrity only. The caller must still decode and verify certificate
/// observer/witness signatures, local pins, threshold, and checkpoint age.
///
/// # Errors
/// Returns a stable privacy-safe reason for endpoint, transport, canonical
/// encoding, identity, freshness, digest, size, or signature rejection.
pub async fn fetch_authenticated_observation_certificate(
    client: &reqwest::Client,
    source_endpoint: &str,
    identity: &IdentityKeyPair,
    expected_source: &[u8; 32],
) -> Result<AuthenticatedDirectoryObservationCertificate, String> {
    if *expected_source == [0u8; 32]
        || *expected_source == identity.public_key_bytes()
        || !commitment_peer_endpoint_is_public(source_endpoint)
    {
        return Err("observation_certificate_source_invalid".to_string());
    }
    let url = commitment_peer_url(
        source_endpoint,
        "/api/discovery/peer/directory/observation-certificate",
    )
    .map_err(|_| "observation_certificate_source_invalid".to_string())?;
    let request_timestamp = unix_now_secs();
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_observation_certificate_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::ObservationCertificateRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request)
        .map_err(|_| "observation_certificate_request_encode_failed".to_string())?;
    let response =
        post_directory_frame(client, url, frame, "observation_certificate_fetch").await?;
    verify_observation_certificate_response(
        &response,
        &request_id,
        &requester,
        expected_source,
        request_timestamp,
        unix_now_secs(),
    )
}

pub(crate) fn verify_observation_certificate_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_requester: &[u8; 32],
    expected_source: &[u8; 32],
    request_timestamp: u64,
    observed_at: u64,
) -> Result<AuthenticatedDirectoryObservationCertificate, String> {
    let response = decode_directory_sync_message(frame)
        .map_err(|_| "observation_certificate_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&response)
        .map_err(|_| "observation_certificate_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("observation_certificate_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ObservationCertificateResponseV1 {
        chain_id,
        request_id,
        requester,
        responder,
        response_timestamp,
        certificate_sha256,
        certificate_frame,
        signature,
    } = response
    else {
        return Err("observation_certificate_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || requester != *expected_requester
        || responder != *expected_source
        || response_timestamp.abs_diff(observed_at) > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || certificate_frame.is_empty()
        || certificate_frame.len() > MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES
    {
        return Err("observation_certificate_response_contract_mismatch".to_string());
    }
    let computed_sha256: [u8; 32] = Sha256::digest(&certificate_frame).into();
    if certificate_sha256 != computed_sha256 {
        return Err("observation_certificate_response_digest_mismatch".to_string());
    }
    let certificate_frame_bytes = u64::try_from(certificate_frame.len())
        .map_err(|_| "observation_certificate_response_contract_mismatch".to_string())?;
    let signing_bytes = directory_observation_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        &responder,
        response_timestamp,
        &certificate_sha256,
        certificate_frame_bytes,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "observation_certificate_response_invalid_signature".to_string())?;
    Ok(AuthenticatedDirectoryObservationCertificate {
        frame: certificate_frame,
        certificate_sha256,
        source: responder,
        response_timestamp,
    })
}
