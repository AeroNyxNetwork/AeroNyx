// [ARCH-SPLIT 2026-10-02]
// Observation witness, carrier, policy anchor, and certificate exchange.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) async fn observation_checkpoint_witness_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        checkpoint,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester != checkpoint.observer
        || requester == state.identity.public_key_bytes()
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_witness_request");
    }
    let checkpoint_hash = checkpoint.hash();
    let now = now_secs();
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
        &checkpoint_hash,
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
    let checkpoint_sequence = checkpoint.sequence;
    let decision = match independently_evaluate_checkpoint(&state, &checkpoint, now).await {
        Ok(decision) => decision,
        Err(response) => return response,
    };
    let outcome = match decision {
        DirectoryObservationWitnessDecision::Accepted => DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        DirectoryObservationWitnessDecision::EvidenceUnavailable => {
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1
        }
        DirectoryObservationWitnessDecision::EvidenceConflict => {
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1
        }
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_observation_witness_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        checkpoint_sequence,
        &checkpoint_hash,
        &responder,
        response_timestamp,
        outcome,
    );
    debug!(
        accepted = outcome == DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        checkpoint_sequence,
        "[DIRECTORY_CHAIN] Evaluated authenticated observation checkpoint witness"
    );
    encoded_response(
        DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer: requester,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature: state.identity.sign(&response_signing_bytes),
        },
    )
}

pub(super) fn verify_carried_observation_witness_request(
    frame: &[u8],
    expected_requester: &[u8; 32],
    observed_at: u64,
) -> Result<CarriedObservationWitnessRequestContext, &'static str> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "carried_witness_request_decode_failed")?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "carried_witness_request_encode_failed")?;
    if canonical != frame {
        return Err("carried_witness_request_noncanonical");
    }
    let DirectorySyncMessage::ObservationCheckpointWitnessRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        checkpoint,
        signature,
    } = message
    else {
        return Err("carried_witness_request_unexpected_message");
    };
    let checkpoint_hash = checkpoint.hash();
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester != *expected_requester
        || requester != checkpoint.observer
        || observed_at.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
        || checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .is_err()
    {
        return Err("carried_witness_request_contract_mismatch");
    }
    let signing_bytes = directory_observation_witness_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
        &checkpoint_hash,
    );
    IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "carried_witness_request_invalid_signature")?;
    Ok(CarriedObservationWitnessRequestContext {
        request_id,
        requester,
        request_timestamp,
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
    })
}

pub(super) fn verify_carried_observation_witness_response(
    frame: &[u8],
    request: &CarriedObservationWitnessRequestContext,
    expected_witness: &[u8; 32],
    observed_at: u64,
) -> Result<(), &'static str> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "carried_witness_response_decode_failed")?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "carried_witness_response_encode_failed")?;
    if canonical != frame {
        return Err("carried_witness_response_noncanonical");
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
    } = message
    else {
        return Err("carried_witness_response_unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != request.request_id
        || observer != request.requester
        || checkpoint_sequence != request.checkpoint_sequence
        || checkpoint_hash != request.checkpoint_hash
        || responder != *expected_witness
        || observed_at.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS)
            < request.request_timestamp
        || ![
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_UNAVAILABLE_V1,
            DIRECTORY_OBSERVATION_WITNESS_EVIDENCE_CONFLICT_V1,
        ]
        .contains(&outcome)
    {
        return Err("carried_witness_response_contract_mismatch");
    }
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        &chain_id,
        &request_id,
        &observer,
        checkpoint_sequence,
        &checkpoint_hash,
        &responder,
        response_timestamp,
        outcome,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "carried_witness_response_invalid_signature")
}

pub(super) async fn observation_checkpoint_witness_carrier_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationCheckpointWitnessCarrierRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        witness,
        witness_request_sha256,
        witness_request_frame,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let carrier = state.identity.public_key_bytes();
    let actual_request_sha256: [u8; 32] = Sha256::digest(&witness_request_frame).into();
    let request_frame_bytes = u64::try_from(witness_request_frame.len()).unwrap_or(u64::MAX);
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester == carrier
        || requester == witness
        || witness == carrier
        || witness_request_frame.is_empty()
        || witness_request_frame.len() > MAX_DIRECTORY_SYNC_REQUEST_BODY_BYTES
        || witness_request_sha256 == [0u8; 32]
        || witness_request_sha256 != actual_request_sha256
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_witness_carrier_request");
    }
    let now = now_secs();
    let signing_bytes = directory_observation_witness_carrier_request_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        request_timestamp,
        &witness,
        &witness_request_sha256,
        request_frame_bytes,
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
    if !state.pinned_peers.contains(&witness) {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::PolicyRejected,
            protocol_error(StatusCode::FORBIDDEN, "witness_target_not_pinned"),
        );
    }
    let carried_request =
        match verify_carried_observation_witness_request(&witness_request_frame, &requester, now) {
            Ok(request) => request,
            Err(_) => {
                return witness_carrier_outcome_response(
                    &state,
                    DirectoryObservationWitnessCarrierOutcome::InvalidRequest,
                    protocol_error(StatusCode::BAD_REQUEST, "invalid_inner_witness_request"),
                );
            }
        };
    let Some(descriptor) = state.peer_store.get_valid(&witness, now) else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    };
    let descriptor_sequence = descriptor.descriptor.sequence;
    let Some(endpoint) = descriptor.descriptor.public_endpoint.as_deref() else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    };
    if !commitment_peer_endpoint_is_public(endpoint) {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    }
    let Ok(url) = commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/observation-checkpoint-witness",
    ) else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_unavailable",
            ),
        );
    };
    if state
        .witness_carrier
        .target_is_cooling_down(&witness, descriptor_sequence, now)
        .await
    {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetCoolingDown,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_target_cooling_down",
            ),
        );
    }
    // [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Fail fast when every
    // bounded slot is occupied. Queuing authenticated requests here would let
    // slow witnesses accumulate Tokio tasks and memory without increasing
    // useful throughput.
    let Ok(_in_flight_permit) = Arc::clone(&state.witness_carrier.in_flight).try_acquire_owned()
    else {
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::LocalOverloaded,
            protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "witness_carrier_overloaded",
            ),
        );
    };
    let target_response = match state
        .witness_carrier
        .transport
        .send(url, witness_request_frame)
        .await
    {
        Ok(response) => response,
        Err(WitnessCarrierTransportError::LocalUnavailable) => {
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::LocalFailure,
                protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "witness_carrier_transport_unavailable",
                ),
            );
        }
        Err(WitnessCarrierTransportError::TargetUnavailable) => {
            state
                .witness_carrier
                .cool_down_target(
                    witness,
                    descriptor_sequence,
                    now_secs(),
                    WITNESS_CARRIER_AVAILABILITY_COOLDOWN_SECS,
                )
                .await;
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
                protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "witness_target_unavailable",
                ),
            );
        }
        Err(WitnessCarrierTransportError::ResponseTooLarge) => {
            state
                .witness_carrier
                .cool_down_target(
                    witness,
                    descriptor_sequence,
                    now_secs(),
                    WITNESS_CARRIER_INVALID_RESPONSE_COOLDOWN_SECS,
                )
                .await;
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
                protocol_error(StatusCode::BAD_GATEWAY, "witness_target_invalid_response"),
            );
        }
    };
    let status = target_response.status;
    if !(200..300).contains(&status) {
        if matches!(status, 404 | 405 | 501) {
            state
                .witness_carrier
                .cool_down_target(
                    witness,
                    descriptor_sequence,
                    now_secs(),
                    WITNESS_CARRIER_CAPABILITY_COOLDOWN_SECS,
                )
                .await;
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable,
                protocol_error(
                    StatusCode::FAILED_DEPENDENCY,
                    "witness_target_capability_unavailable",
                ),
            );
        }
        if matches!(status, 408 | 429) || (500..600).contains(&status) {
            state
                .witness_carrier
                .cool_down_target(
                    witness,
                    descriptor_sequence,
                    now_secs(),
                    WITNESS_CARRIER_AVAILABILITY_COOLDOWN_SECS,
                )
                .await;
            return witness_carrier_outcome_response(
                &state,
                DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
                protocol_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "witness_target_unavailable",
                ),
            );
        }
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetRejected,
            protocol_error(StatusCode::BAD_GATEWAY, "witness_target_rejected"),
        );
    }
    let witness_response_frame = target_response.body;
    if witness_response_frame.is_empty()
        || verify_carried_observation_witness_response(
            &witness_response_frame,
            &carried_request,
            &witness,
            now_secs(),
        )
        .is_err()
    {
        state
            .witness_carrier
            .cool_down_target(
                witness,
                descriptor_sequence,
                now_secs(),
                WITNESS_CARRIER_INVALID_RESPONSE_COOLDOWN_SECS,
            )
            .await;
        return witness_carrier_outcome_response(
            &state,
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
            protocol_error(StatusCode::BAD_GATEWAY, "witness_target_invalid_response"),
        );
    }
    state.witness_carrier.clear_target_cooldown(&witness).await;
    let response_timestamp = now_secs();
    let witness_response_sha256: [u8; 32] = Sha256::digest(&witness_response_frame).into();
    let response_frame_bytes = u64::try_from(witness_response_frame.len()).unwrap_or(u64::MAX);
    let response_signing_bytes = directory_observation_witness_carrier_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        &witness,
        &carrier,
        response_timestamp,
        &witness_request_sha256,
        &witness_response_sha256,
        response_frame_bytes,
    );
    // [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] Do not emit a per-request
    // carrier success event. Even an otherwise public checkpoint sequence plus
    // log time would create a cross-node correlation handle. The aggregate,
    // process-only outcome below is the complete operational signal.
    witness_carrier_outcome_response(
        &state,
        DirectoryObservationWitnessCarrierOutcome::Forwarded,
        encoded_response(
            DirectorySyncMessage::ObservationCheckpointWitnessCarrierResponseV1 {
                chain_id,
                request_id,
                requester,
                witness,
                carrier,
                response_timestamp,
                witness_request_sha256,
                witness_response_sha256,
                witness_response_frame,
                signature: state.identity.sign(&response_signing_bytes),
            },
        ),
    )
}

/// Records one terminal carrier outcome without accepting identity-bearing data.
pub(super) fn witness_carrier_outcome_response(
    state: &DirectoryChainPeerState,
    outcome: DirectoryObservationWitnessCarrierOutcome,
    response: Response,
) -> Response {
    state
        .runtime
        .record_observation_witness_carrier_outcome(outcome, now_secs());
    response
}

pub(super) async fn observation_policy_anchor_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        policy_epoch,
        previous_policy_digest,
        policy_digest,
        signature,
    } = &message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    let position_valid = (*policy_epoch == 1 && *previous_policy_digest == [0u8; 32])
        || (*policy_epoch > 1 && *previous_policy_digest != [0u8; 32]);
    if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || *requester == state.identity.public_key_bytes()
        || *policy_digest == [0u8; 32]
        || !position_valid
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_policy_anchor_request");
    }
    let now = now_secs();
    let signing_bytes = directory_policy_anchor_request_signing_bytes(
        chain_id,
        request_id,
        requester,
        *request_timestamp,
        *policy_epoch,
        previous_policy_digest,
        policy_digest,
    );
    if let Err(response) = authenticate_request(
        &state,
        DirectoryPeerAdmission::PinnedAuthority,
        *requester,
        *request_id,
        *request_timestamp,
        &signing_bytes,
        signature,
        now,
    )
    .await
    {
        return response;
    }
    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "replica_store_disabled");
    };
    let anchor_request = message.clone();
    let decision = match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "observation_policy_anchor",
        move || store.persist_remote_observation_witness_policy_anchor(&anchor_request, now),
    )
    .await
    {
        Ok(Ok(decision)) => decision,
        Ok(Err(error)) => return replica_store_error_response(&error),
        Err(response) => return response,
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let outcome = decision.outcome();
    let response_signing_bytes = directory_policy_anchor_response_signing_bytes(
        chain_id,
        request_id,
        requester,
        *policy_epoch,
        policy_digest,
        &responder,
        response_timestamp,
        outcome,
    );
    debug!(
        accepted = decision == DirectoryObservationWitnessPolicyAnchorDecision::Accepted,
        policy_epoch, "[DIRECTORY_CHAIN] Evaluated authenticated opaque policy-head anchor"
    );
    encoded_response(
        DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id: *chain_id,
            request_id: *request_id,
            observer: *requester,
            policy_epoch: *policy_epoch,
            policy_digest: *policy_digest,
            responder,
            response_timestamp,
            outcome,
            signature: state.identity.sign(&response_signing_bytes),
        },
    )
}

pub(super) async fn observation_certificate_handler(
    State(state): State<DirectoryChainPeerState>,
    body: Bytes,
) -> Response {
    let message = match decode_request(&body) {
        Ok(message) => message,
        Err(response) => return response,
    };
    let DirectorySyncMessage::ObservationCertificateRequestV1 {
        chain_id,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || requester == state.identity.public_key_bytes()
    {
        return protocol_error(
            StatusCode::BAD_REQUEST,
            "invalid_observation_certificate_request",
        );
    }
    let now = now_secs();
    let signing_bytes = directory_observation_certificate_request_signing_bytes(
        &chain_id,
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

    let Some(store) = state.replica_store.as_ref().map(Arc::clone) else {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "replica_store_disabled");
    };
    let mut eligible_witnesses = state.pinned_peers.iter().copied().collect::<Vec<_>>();
    eligible_witnesses.sort_unstable();
    let certificate = match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "observation_certificate_audit",
        move || {
            let snapshot = store.status_snapshot()?;
            let minimum_witnesses = usize::try_from(snapshot.observation_witness_policy_threshold)
                .unwrap_or(usize::MAX);
            if minimum_witnesses == 0 || minimum_witnesses > eligible_witnesses.len() {
                return Ok(None);
            }
            store.latest_observation_certificate_for_pins(
                &eligible_witnesses,
                minimum_witnesses,
                now,
            )
        },
    )
    .await
    {
        Ok(Ok(Some(certificate))) => certificate,
        Ok(Ok(None)) => {
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "observation_certificate_unavailable",
            )
        }
        Ok(Err(error)) => return replica_store_error_response(&error),
        Err(response) => return response,
    };
    let certificate_frame = match encode_directory_observation_certificate(&certificate) {
        Ok(frame) if frame.len() <= MAX_DIRECTORY_OBSERVATION_CERTIFICATE_FRAME_BYTES => frame,
        Ok(_) => {
            return protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "observation_certificate_oversized",
            )
        }
        Err(error) => {
            warn!(
                error = %error,
                "[DIRECTORY_CHAIN] Failed to encode verified observation certificate"
            );
            return protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "observation_certificate_encode_failed",
            );
        }
    };
    let certificate_sha256: [u8; 32] = Sha256::digest(&certificate_frame).into();
    let certificate_frame_bytes = match u64::try_from(certificate_frame.len()) {
        Ok(bytes) => bytes,
        Err(_) => {
            return protocol_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "observation_certificate_oversized",
            )
        }
    };
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = directory_observation_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &requester,
        &responder,
        response_timestamp,
        &certificate_sha256,
        certificate_frame_bytes,
    );
    debug!(
        certificate_sequence = certificate.checkpoint.sequence,
        certificate_frame_bytes,
        "[DIRECTORY_CHAIN] Served authenticated portable observation certificate"
    );
    encoded_response(DirectorySyncMessage::ObservationCertificateResponseV1 {
        chain_id,
        request_id,
        requester,
        responder,
        response_timestamp,
        certificate_sha256,
        certificate_frame,
        signature: state.identity.sign(&response_signing_bytes),
    })
}

pub(super) async fn independently_evaluate_checkpoint(
    state: &DirectoryChainPeerState,
    checkpoint: &DirectoryObservationCheckpointV1,
    now: u64,
) -> Result<DirectoryObservationWitnessDecision, Response> {
    let Some(replica_store) = state.replica_store.as_ref().map(Arc::clone) else {
        return Err(protocol_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "replica_store_disabled",
        ));
    };
    let chain_store = Arc::clone(&state.store);
    match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "local_producer_witness_audit",
        move || chain_store.audit(now),
    )
    .await
    {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Local producer audit failed closed");
            return Err(protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "chain_not_verified",
            ));
        }
        Err(response) => return Err(response),
    }
    let checkpoint = checkpoint.clone();
    match run_directory_chain_blocking(
        Arc::clone(&state.audit_admission),
        "observation_witness_recomputation",
        move || replica_store.evaluate_observation_checkpoint_witness(&checkpoint, now),
    )
    .await
    {
        Ok(Ok(decision)) => Ok(decision),
        Ok(Err(DirectoryReplicaStoreError::Request(_))) => Err(protocol_error(
            StatusCode::BAD_REQUEST,
            "invalid_checkpoint",
        )),
        Ok(Err(error)) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Witness recomputation failed closed");
            Err(protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "replica_not_verified",
            ))
        }
        Err(response) => Err(response),
    }
}
