// [ARCH-SPLIT 2026-10-02]
// Block-page pull, carrier fallback, and the block-range HTTP handler.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Pulls, verifies, and atomically appends one bounded commitment-block page.
///
/// The coordinator identity is supplied by validated operator configuration.
/// Discovery is used only to resolve that exact identity's current signed
/// endpoint; this function never selects or falls back to another peer.
///
/// # Errors
///
/// Returns a stable privacy-safe code when the local audited tip is
/// unavailable, the pinned peer cannot be reached, the signed page is invalid,
/// or the atomic local append fails closed.
pub async fn pull_record_commitment_page(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentSyncPageOutcome, String> {
    pull_record_commitment_page_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
/// Pulls one coordinator-authored page with bounded pinned-carrier recovery.
///
/// The coordinator is always attempted first. Carrier recovery is enabled only
/// when the local follower requires at least two witness signatures and has
/// enough configured witness pins to satisfy that policy. A carrier signs the
/// page envelope but every block must remain signed by `coordinator_node_id`.
/// Only classified availability failures may advance to another source.
///
/// # Errors
///
/// Returns the coordinator's stable availability code when every bounded
/// source is unavailable. Any endpoint-policy, decoding, identity, signature,
/// proposer, continuity, rollback, pagination, or storage failure stops closed
/// before another source can mask the incident.
pub async fn pull_record_commitment_page_with_carrier_recovery(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
) -> Result<CommitmentFollowerPagePullOutcome, String> {
    let mut cursor = CommitmentBlockCarrierCursor::default();
    pull_record_commitment_page_with_carrier_cursor(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        &mut cursor,
    )
    .await
}

/// Pulls one page while preserving a successful carrier preference within one
/// caller-owned multi-page synchronization round.
///
/// The cursor never weakens coordinator-first behavior or source validation.
/// It only avoids retrying earlier availability failures before a carrier that
/// already delivered a fully verified page in the same round.
pub(crate) async fn pull_record_commitment_page_with_carrier_cursor(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    cursor: &mut CommitmentBlockCarrierCursor,
) -> Result<CommitmentFollowerPagePullOutcome, String> {
    pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
        cursor,
    )
    .await
}

/// Pulls one verified page with an exact caller-selected hard limit.
///
/// The follower uses this only to stop immediately before a signed authority
/// transition. Existing callers retain the full protocol page size.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn pull_record_commitment_page_with_carrier_runtime_bounded(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    cursor: &mut CommitmentBlockCarrierCursor,
    circuit_breaker: &mut CommitmentBlockCarrierCircuitBreaker,
    max_blocks: u16,
) -> Result<CommitmentFollowerPagePullOutcome, String> {
    if !(1..=MAX_BLOCKS_PER_RESPONSE_WIRE).contains(&max_blocks) {
        return Err("invalid_block_page_limit".to_string());
    }
    pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
        cursor,
        circuit_breaker,
        max_blocks,
    )
    .await
}

pub(super) async fn pull_record_commitment_page_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentSyncPageOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    pull_record_commitment_page_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        coordinator_node_id,
        client,
        endpoint_allowed,
        MAX_BLOCKS_PER_RESPONSE_WIRE,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) async fn pull_record_commitment_page_with_carrier_recovery_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentFollowerPagePullOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let mut cursor = CommitmentBlockCarrierCursor::default();
    pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        endpoint_allowed,
        &mut cursor,
    )
    .await
}

pub(super) fn eligible_pinned_commitment_carriers(
    local_node_id: [u8; 32],
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
) -> Vec<[u8; 32]> {
    // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Block and handover
    // recovery share one immutable pin normalizer. A typed cursor may change
    // only the bounded attempt start inside this exact list; it cannot import
    // a discovery peer, include self/primary, or alter membership.
    let mut carriers = Vec::with_capacity(MAX_PINNED_WITNESSES_PER_ROUND);
    for carrier in carrier_node_ids {
        if *carrier == local_node_id || carrier == coordinator_node_id || carriers.contains(carrier)
        {
            continue;
        }
        carriers.push(*carrier);
        if carriers.len() == MAX_PINNED_WITNESSES_PER_ROUND {
            break;
        }
    }
    carriers
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn pull_record_commitment_page_with_carrier_cursor_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    cursor: &mut CommitmentBlockCarrierCursor,
) -> Result<CommitmentFollowerPagePullOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let mut circuit_breaker = CommitmentBlockCarrierCircuitBreaker::default();
    pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        carrier_node_ids,
        minimum_required_signers,
        client,
        endpoint_allowed,
        cursor,
        &mut circuit_breaker,
        MAX_BLOCKS_PER_RESPONSE_WIRE,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn pull_record_commitment_page_with_carrier_runtime_and_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    coordinator_node_id: &[u8; 32],
    carrier_node_ids: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    cursor: &mut CommitmentBlockCarrierCursor,
    circuit_breaker: &mut CommitmentBlockCarrierCircuitBreaker,
    max_blocks: u16,
) -> Result<CommitmentFollowerPagePullOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(1..=MAX_BLOCKS_PER_RESPONSE_WIRE).contains(&max_blocks) {
        return Err("invalid_block_page_limit".to_string());
    }
    // [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Align before the
    // coordinator request so an operator pin-count change clears positional
    // state even when the direct path succeeds and no carrier is contacted.
    let carriers = eligible_pinned_commitment_carriers(
        identity.public_key_bytes(),
        coordinator_node_id,
        carrier_node_ids,
    );
    circuit_breaker.align_slots(carriers.len());
    let mut cooldown_skips = 0usize;
    let mut half_open_attempts = 0usize;

    // [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex] Every terminal
    // path records one typed aggregate disposition. Recording remains inside
    // this direct-first primitive so future callers cannot omit or reinterpret
    // the source budget; the storage contract discards all source details.
    let direct = pull_record_commitment_page_from_source_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        coordinator_node_id,
        coordinator_node_id,
        client,
        endpoint_allowed,
        max_blocks,
    )
    .await;
    let direct_error = match direct {
        Ok(page) => {
            cursor.reset();
            record_commitment_block_carrier_circuit_telemetry(
                storage,
                circuit_breaker,
                cooldown_skips,
                half_open_attempts,
            );
            storage.record_commitment_block_page_pull_outcome(
                now_secs(),
                RecordCommitmentBlockPagePullDisposition::Coordinator,
                0,
            );
            return Ok(CommitmentFollowerPagePullOutcome {
                page,
                source: CommitmentSyncPageSource::Coordinator,
                carrier_attempts: 0,
            });
        }
        Err(error) => error,
    };

    if commitment_block_source_failure_class(&direct_error)
        == CommitmentBlockSourceFailureClass::Security
    {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::SecurityStopped,
            0,
        );
        return Err(direct_error);
    }
    if minimum_required_signers < 2 {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::AvailabilityExhausted,
            0,
        );
        return Err(direct_error);
    }
    if minimum_required_signers > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1 {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::SecurityStopped,
            0,
        );
        return Err("block_carrier_policy_invalid".to_string());
    }

    // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] Preserve operator order,
    // exclude self/coordinator, deduplicate, and enforce the same hard fan-out
    // cap as witness operations. Discovery never chooses a recovery source.
    if carriers.len() < minimum_required_signers {
        record_commitment_block_carrier_circuit_telemetry(
            storage,
            circuit_breaker,
            cooldown_skips,
            half_open_attempts,
        );
        storage.record_commitment_block_page_pull_outcome(
            now_secs(),
            RecordCommitmentBlockPagePullDisposition::SecurityStopped,
            0,
        );
        return Err("block_carrier_policy_invalid".to_string());
    }

    let mut carrier_attempts = 0usize;
    let carrier_count = carriers.len();
    let start_index = cursor.start_index(carrier_count);
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
        match pull_record_commitment_page_from_source_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &carrier,
            coordinator_node_id,
            client,
            endpoint_allowed,
            max_blocks,
        )
        .await
        {
            Ok(page) => {
                circuit_breaker.record_success(carrier_index);
                cursor.prefer(carrier_index, carrier_count);
                record_commitment_block_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_block_page_pull_outcome(
                    now_secs(),
                    RecordCommitmentBlockPagePullDisposition::CarrierRecovered,
                    carrier_attempts,
                );
                return Ok(CommitmentFollowerPagePullOutcome {
                    page,
                    source: CommitmentSyncPageSource::PinnedCarrier,
                    carrier_attempts,
                });
            }
            Err(error)
                if commitment_block_source_failure_class(&error)
                    == CommitmentBlockSourceFailureClass::Availability =>
            {
                circuit_breaker.record_availability_failure(carrier_index, Instant::now());
                cursor.advance_after_availability_failure(carrier_index, carrier_count);
            }
            Err(error) => {
                record_commitment_block_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    cooldown_skips,
                    half_open_attempts,
                );
                storage.record_commitment_block_page_pull_outcome(
                    now_secs(),
                    RecordCommitmentBlockPagePullDisposition::SecurityStopped,
                    carrier_attempts,
                );
                return Err(error);
            }
        }
    }

    // Preserve the coordinator's established privacy-safe code so existing
    // operations and alerting remain backward compatible.
    record_commitment_block_carrier_circuit_telemetry(
        storage,
        circuit_breaker,
        cooldown_skips,
        half_open_attempts,
    );
    storage.record_commitment_block_page_pull_outcome(
        now_secs(),
        RecordCommitmentBlockPagePullDisposition::AvailabilityExhausted,
        carrier_attempts,
    );
    Err(direct_error)
}

pub(super) fn commitment_block_source_failure_class(
    error: &str,
) -> CommitmentBlockSourceFailureClass {
    let retryable_status = error
        .strip_prefix("http_status_")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| matches!(status, 403 | 404 | 408 | 429 | 500 | 502 | 503 | 504));
    if retryable_status
        || matches!(
            error,
            "pinned_coordinator_unavailable"
                | "pinned_coordinator_missing_endpoint"
                | "request_timeout"
                | "request_connect"
                | "response_body_timeout"
                | "response_body_connect"
                | "response_body_body"
                | "carrier_tip_behind"
        )
    {
        CommitmentBlockSourceFailureClass::Availability
    } else {
        CommitmentBlockSourceFailureClass::Security
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn pull_record_commitment_page_from_source_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    response_signer_node_id: &[u8; 32],
    expected_proposer_node_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
    max_blocks: u16,
) -> Result<CommitmentSyncPageOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    let source = peer_store
        .get_valid(response_signer_node_id, request_timestamp)
        .ok_or_else(|| "pinned_coordinator_unavailable".to_string())?;
    let endpoint = source
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "pinned_coordinator_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("pinned_coordinator_unsafe_endpoint".to_string());
    }
    let url = commitment_block_range_url(endpoint)?;

    let local_tip = verified_local_commitment_tip(storage).await?;
    let from_height = local_tip.0.saturating_add(1).max(1);
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    if !(1..=MAX_BLOCKS_PER_RESPONSE_WIRE).contains(&max_blocks) {
        return Err("invalid_block_page_limit".to_string());
    }
    let limit = max_blocks;
    let signing_bytes = record_block_range_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height,
        limit,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordBlockRangeRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        from_height,
        limit,
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
        .map_err(|error| classify_http_error("request", &error))?;
    if !response.status().is_success() {
        return Err(format!("http_status_{}", response.status().as_u16()));
    }
    let body = read_bounded_response(response).await?;
    let page = verify_record_commitment_page(
        &body,
        &request_id,
        response_signer_node_id,
        expected_proposer_node_id,
        local_tip,
        now_secs(),
    )?;
    if page.blocks.len() > usize::from(max_blocks) {
        return Err("response_page_exceeds_request".to_string());
    }

    let append = storage
        .append_record_commitment_blocks_atomic(&page.blocks, Some(expected_proposer_node_id))
        .await
        .map_err(|_| "storage_append_rejected".to_string())?;

    Ok(CommitmentSyncPageOutcome {
        inserted: append.inserted,
        already_present: append.already_present,
        has_more: page.has_more,
        remote_tip_height: page.tip_height,
    })
}

pub(super) fn verify_record_commitment_page(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    expected_proposer: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    now: u64,
) -> Result<VerifiedCommitmentPage, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_response_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_response_frame")?;
    let MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_response_message".to_string());
    };

    if request_id != *expected_request_id {
        return Err("response_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("response_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_response".to_string());
    }
    if blocks.len() > MAX_BLOCKS_PER_RESPONSE {
        return Err("response_page_too_large".to_string());
    }
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&response_signing_bytes, &signature))
        .map_err(|_| "invalid_response_signature".to_string())?;

    let (local_height, local_hash) = local_tip;
    if local_height == 0 && local_hash != GENESIS_PREV_HASH {
        return Err("invalid_local_genesis".to_string());
    }
    if tip_height < local_height {
        return Err(if expected_responder == expected_proposer {
            "coordinator_rollback_detected"
        } else {
            "carrier_tip_behind"
        }
        .to_string());
    }
    if blocks.is_empty() {
        if has_more || tip_height != local_height || tip_hash != local_hash {
            return Err("empty_page_tip_mismatch".to_string());
        }
        return Ok(VerifiedCommitmentPage {
            blocks,
            has_more,
            tip_height,
        });
    }
    if tip_height <= local_height {
        return Err("unexpected_blocks_at_current_tip".to_string());
    }

    let mut expected_height = local_height.saturating_add(1);
    let mut expected_prev_hash = local_hash;
    for block in &blocks {
        if block.header.proposer != *expected_proposer {
            return Err("unexpected_block_proposer".to_string());
        }
        block
            .verify(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                expected_height,
                &expected_prev_hash,
            )
            .map_err(|_| "commitment_chain_verification_failed".to_string())?;
        expected_height = expected_height.saturating_add(1);
        expected_prev_hash = block.hash();
    }

    let page_tip_height = expected_height.saturating_sub(1);
    let expected_has_more = page_tip_height < tip_height;
    if has_more != expected_has_more {
        return Err("pagination_state_mismatch".to_string());
    }
    if !has_more && (tip_height != page_tip_height || tip_hash != expected_prev_hash) {
        return Err("terminal_tip_mismatch".to_string());
    }

    Ok(VerifiedCommitmentPage {
        blocks,
        has_more,
        tip_height,
    })
}

pub(super) async fn block_range_handler(
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
    let MemChainMessage::RecordBlockRangeRequestV1 {
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

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || from_height == 0 || limit == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_range");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_block_range_request_signing_bytes(
        &chain_id,
        from_height,
        limit,
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

    let page_limit = usize::from(limit).min(MAX_BLOCKS_PER_RESPONSE);
    let page = match state
        .storage
        .get_verified_record_commitment_block_page(from_height, page_limit)
        .await
    {
        Ok(page) => page,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unverified block range");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified");
        }
    };
    let blocks = page.blocks;
    let tip_height = page.tip_height;
    let tip_hash = page.tip_hash;
    let page_tip = blocks.last().map_or_else(
        || from_height.saturating_sub(1),
        |block| block.header.height,
    );
    let has_more = page_tip < tip_height;
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    let response = MemChainMessage::RecordBlockRangeResponseV1 {
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode block range response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        blocks = match &response {
            MemChainMessage::RecordBlockRangeResponseV1 { blocks, .. } => blocks.len(),
            _ => 0,
        },
        has_more, tip_height, "[MEMCHAIN_BLOCK] Served authenticated commitment range"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}
