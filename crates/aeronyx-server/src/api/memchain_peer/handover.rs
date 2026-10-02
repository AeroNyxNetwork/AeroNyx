// [ARCH-SPLIT 2026-10-02]
// Coordinator handover sync from the current authority.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Pulls at most one exact-next coordinator handover proof from the currently
/// authorised coordinator and persists it only at its activation boundary.
///
/// [AUTHORITY-HANDOVER-EXCHANGE 2026-08-14 by Codex] The responder is merely a
/// transport source. Authority comes exclusively from the dual-signed proof,
/// the immutable local authority root, and the audited local block prefix.
/// A future proof is returned only as a page boundary so the caller can catch
/// up without accepting coordinator authority early.
pub async fn sync_next_record_coordinator_handover(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
) -> Result<CommitmentAuthoritySyncOutcome, String> {
    let mut cursor = CommitmentAuthorityCarrierCursor::default();
    let mut circuit_breaker = CommitmentAuthorityCarrierCircuitBreaker::default();
    sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        &[],
        client,
        &commitment_peer_endpoint_is_public,
        &mut cursor,
        &mut circuit_breaker,
    )
    .await
}

/// Synchronizes one authority proof with bounded operator-pinned recovery.
///
/// Direct coordinator transport is always attempted first. Only explicit
/// availability failures may advance to a carrier, and carriers transport but
/// never authorise the independently dual-signed transition.
pub(crate) async fn sync_next_record_coordinator_handover_with_carrier_runtime(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    carrier_node_ids: &[[u8; 32]],
    client: &reqwest::Client,
    cursor: &mut CommitmentAuthorityCarrierCursor,
    circuit_breaker: &mut CommitmentAuthorityCarrierCircuitBreaker,
) -> Result<CommitmentAuthoritySyncOutcome, String> {
    sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        carrier_node_ids,
        client,
        &commitment_peer_endpoint_is_public,
        cursor,
        circuit_breaker,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn sync_record_coordinator_handover_from_source_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    expected_authority: &RecordCommitmentAuthorityState,
    response_signer: &[u8; 32],
    source: CommitmentAuthoritySyncSource,
    carrier_attempts: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentAuthoritySyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let authority = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    if authority != *expected_authority {
        return Err("handover_local_authority_changed".to_string());
    }
    if authority.coordinator == identity.public_key_bytes() {
        return Err("active_coordinator_cannot_follow_itself".to_string());
    }

    let request_timestamp = now_secs();
    let (unavailable_error, missing_endpoint_error, unsafe_endpoint_error) = match source {
        CommitmentAuthoritySyncSource::Coordinator => (
            "active_coordinator_unavailable",
            "active_coordinator_missing_endpoint",
            "active_coordinator_unsafe_endpoint",
        ),
        CommitmentAuthoritySyncSource::PinnedCarrier => (
            "handover_carrier_unavailable",
            "handover_carrier_missing_endpoint",
            "handover_carrier_unsafe_endpoint",
        ),
    };
    let responder = peer_store
        .get_valid(response_signer, request_timestamp)
        .ok_or_else(|| unavailable_error.to_string())?;
    let endpoint = responder
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| missing_endpoint_error.to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err(unsafe_endpoint_error.to_string());
    }
    let url = commitment_coordinator_handover_url(endpoint)?;

    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_coordinator_handover_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        authority.authority_epoch,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorHandoverRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        after_authority_epoch: authority.authority_epoch,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame =
        encode_memchain(&request).map_err(|_| "handover_request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("handover_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "handover_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let verified = control_plane::verify_record_coordinator_handover_response(
        &body,
        &request_id,
        response_signer,
        &authority.coordinator,
        authority.authority_epoch,
        authority.next_block_height,
        now_secs(),
    )?;

    if source == CommitmentAuthoritySyncSource::PinnedCarrier && verified.handover.is_none() {
        // A carrier's empty local head cannot prove the active coordinator
        // made no transition; another exact operator pin may be less stale.
        return Err("handover_carrier_behind".to_string());
    }

    let mut handover_inserted = false;
    let mut pending_activation_height = None;
    if let Some(handover) = verified.handover {
        if handover.header.activation_height == authority.next_block_height {
            handover_inserted = matches!(
                storage
                    .persist_configured_record_coordinator_handover(&handover, now_secs())
                    .await?,
                RecordCoordinatorHandoverPersistOutcome::Inserted
            );
        } else {
            pending_activation_height = Some(handover.header.activation_height);
        }
    }

    let refreshed = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    Ok(CommitmentAuthoritySyncOutcome {
        authority_epoch: refreshed.authority_epoch,
        active_coordinator: refreshed.coordinator,
        next_block_height: refreshed.next_block_height,
        pending_activation_height,
        handover_inserted,
        source,
        carrier_attempts,
    })
}

#[cfg(test)]
pub(super) async fn sync_next_record_coordinator_handover_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentAuthoritySyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let authority = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    sync_record_coordinator_handover_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        &authority,
        &authority.coordinator,
        CommitmentAuthoritySyncSource::Coordinator,
        0,
        client,
        endpoint_allowed,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn sync_next_record_coordinator_handover_with_carrier_runtime_and_endpoint_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    carrier_node_ids: &[[u8; 32]],
    client: &reqwest::Client,
    endpoint_allowed: &F,
    cursor: &mut CommitmentAuthorityCarrierCursor,
    circuit_breaker: &mut CommitmentAuthorityCarrierCircuitBreaker,
) -> Result<CommitmentAuthoritySyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let authority = storage
        .record_commitment_authority_state()
        .await?
        .ok_or_else(|| "commitment_authority_not_configured".to_string())?;
    if authority.coordinator == identity.public_key_bytes() {
        return Err("active_coordinator_cannot_follow_itself".to_string());
    }

    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Recovery membership is
    // exactly the operator pin set. Discovery resolves fresh endpoints only;
    // it cannot nominate a transport source or alter proof authority.
    let carriers = eligible_pinned_commitment_carriers(
        identity.public_key_bytes(),
        &authority.coordinator,
        carrier_node_ids,
    );
    circuit_breaker.align_slots(carriers.len());
    let mut cooldown_skips = 0usize;
    let mut half_open_attempts = 0usize;
    let direct = sync_record_coordinator_handover_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        &authority,
        &authority.coordinator,
        CommitmentAuthoritySyncSource::Coordinator,
        0,
        client,
        endpoint_allowed,
    )
    .await;
    let direct_error = match direct {
        Ok(outcome) => {
            cursor.reset();
            record_commitment_authority_carrier_circuit_telemetry(
                storage,
                circuit_breaker,
                cooldown_skips,
                half_open_attempts,
            );
            storage.record_commitment_authority_sync_outcome(
                now_secs(),
                RecordCommitmentAuthoritySyncDisposition::Coordinator,
                0,
            );
            return Ok(outcome);
        }
        Err(error) => error,
    };
    if coordinator_handover_source_failure_class(&direct_error)
        == CommitmentAuthoritySourceFailureClass::Security
    {
        record_commitment_authority_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_authority_sync_outcome(
            now_secs(),
            RecordCommitmentAuthoritySyncDisposition::SecurityStopped,
            0,
        );
        return Err(direct_error);
    }

    let carrier_count = carriers.len();
    let start_index = cursor.start_index(carrier_count);
    let mut carrier_attempts = 0usize;
    for offset in 0..carrier_count {
        let carrier_index = start_index.saturating_add(offset) % carrier_count;
        match circuit_breaker.decision(carrier_index, Instant::now()) {
            CommitmentCarrierCircuitDecision::Closed => {}
            CommitmentCarrierCircuitDecision::Cooling => {
                cooldown_skips = cooldown_skips.saturating_add(1);
                continue;
            }
            CommitmentCarrierCircuitDecision::HalfOpen => {
                half_open_attempts = half_open_attempts.saturating_add(1);
            }
        }
        let carrier = carriers[carrier_index];
        carrier_attempts = carrier_attempts.saturating_add(1);
        match sync_record_coordinator_handover_from_source_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &authority,
            &carrier,
            CommitmentAuthoritySyncSource::PinnedCarrier,
            carrier_attempts,
            client,
            endpoint_allowed,
        )
        .await
        {
            Ok(outcome) => {
                circuit_breaker.record_success(carrier_index);
                cursor.prefer(carrier_index, carrier_count);
                record_commitment_authority_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_authority_sync_outcome(
                    now_secs(),
                    RecordCommitmentAuthoritySyncDisposition::CarrierRecovered,
                    carrier_attempts,
                );
                return Ok(outcome);
            }
            Err(error)
                if coordinator_handover_source_failure_class(&error)
                    == CommitmentAuthoritySourceFailureClass::Availability =>
            {
                circuit_breaker.record_availability_failure(carrier_index, Instant::now());
                cursor.advance_after_availability_failure(carrier_index, carrier_count);
            }
            Err(error) => {
                record_commitment_authority_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_authority_sync_outcome(
                    now_secs(),
                    RecordCommitmentAuthoritySyncDisposition::SecurityStopped,
                    carrier_attempts,
                );
                return Err(error);
            }
        }
    }

    // Preserve the direct source's stable availability code for existing
    // follower alerting when every bounded carrier is unavailable or behind.
    record_commitment_authority_carrier_circuit_telemetry(
        storage,
        circuit_breaker,
        cooldown_skips,
        half_open_attempts,
    );
    storage.record_commitment_authority_sync_outcome(
        now_secs(),
        RecordCommitmentAuthoritySyncDisposition::AvailabilityExhausted,
        carrier_attempts,
    );
    Err(direct_error)
}

pub(super) fn coordinator_handover_source_failure_class(
    error: &str,
) -> CommitmentAuthoritySourceFailureClass {
    let retryable_status = error
        .strip_prefix("handover_http_status_")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| matches!(status, 403 | 404 | 408 | 429 | 500 | 502 | 503 | 504));
    if retryable_status
        || matches!(
            error,
            "active_coordinator_unavailable"
                | "active_coordinator_missing_endpoint"
                | "handover_carrier_unavailable"
                | "handover_carrier_missing_endpoint"
                | "handover_carrier_behind"
                | "handover_request_timeout"
                | "handover_request_connect"
                | "response_body_timeout"
                | "response_body_connect"
                | "response_body_body"
        )
    {
        CommitmentAuthoritySourceFailureClass::Availability
    } else {
        CommitmentAuthoritySourceFailureClass::Security
    }
}
