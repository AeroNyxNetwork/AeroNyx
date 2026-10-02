// [ARCH-SPLIT 2026-10-02]
// Audited tip announcement and the block-announce HTTP handler.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

// Keep the receipt contract at the HTTP wire boundary. Axum and Reqwest may
// resolve different `http` crate versions, but the protocol status codes are
// stable and must not depend on either transport library's Rust type.
pub(super) fn classify_commitment_tip_announcement_status(
    status: u16,
) -> CommitmentTipAnnouncementDelivery {
    match status {
        202 => CommitmentTipAnnouncementDelivery::Accepted,
        204 => CommitmentTipAnnouncementDelivery::Stale,
        408 | 425 | 500..=599 => CommitmentTipAnnouncementDelivery::RetryableFailure,
        _ => CommitmentTipAnnouncementDelivery::PermanentFailure,
    }
}

pub(super) async fn verified_local_commitment_tip(
    storage: &MemoryStorage,
) -> Result<(u64, [u8; 32]), String> {
    let (_, checkpoint_hash, tip_height, tip_hash) = storage
        .record_commitment_chain_checkpoint(u64::MAX)
        .await
        .map_err(|_| "local_checkpoint_unavailable".to_string())?;
    if checkpoint_hash != tip_hash {
        return Err("local_checkpoint_tip_mismatch".to_string());
    }
    Ok((tip_height, tip_hash))
}

/// Announces the current audited tip to a bounded set of operator-pinned peers.
///
/// Delivery is advisory and best effort. Followers independently authenticate
/// the pinned coordinator and then run the ordinary signed page/checkpoint
/// pull, so accepting this frame never changes their canonical chain.
///
/// # Errors
///
/// Returns an error only when the local audited tip cannot be loaded or encoded.
/// Individual peer failures are represented in the privacy-safe aggregate.
pub async fn announce_current_record_commitment_tip(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
) -> Result<CommitmentTipAnnouncementOutcome, String> {
    announce_current_record_commitment_tip_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Runs the production announcement encoder, peer lookup, HTTP transport, and
/// retry queue with a localhost-capable endpoint policy for integration tests.
///
/// This seam does not exist in non-test builds. Production callers must use
/// [`announce_current_record_commitment_tip`], which enforces final-hop public
/// endpoint validation on every attempt.
#[cfg(test)]
pub(crate) async fn announce_current_record_commitment_tip_for_test(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    max_attempts: usize,
    base_delay: Duration,
) -> Result<CommitmentTipAnnouncementOutcome, String> {
    let endpoint_allowed = |_endpoint: &str| true;
    announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        &endpoint_allowed,
        CommitmentTipAnnouncementRetryPolicy {
            max_attempts,
            base_delay,
        },
    )
    .await
}

pub(super) async fn announce_current_record_commitment_tip_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    endpoint_allowed: &F,
) -> Result<CommitmentTipAnnouncementOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy(
        storage,
        peer_store,
        identity,
        client,
        pinned_peer_ids,
        endpoint_allowed,
        TIP_ANNOUNCEMENT_RETRY_POLICY,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn announce_current_record_commitment_tip_with_endpoint_policy_and_retry_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    pinned_peer_ids: &[[u8; 32]],
    endpoint_allowed: &F,
    retry_policy: CommitmentTipAnnouncementRetryPolicy,
) -> Result<CommitmentTipAnnouncementOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let (tip_height, _) = verified_local_commitment_tip(storage).await?;
    if tip_height == 0 {
        return Err("local_commitment_tip_empty".to_string());
    }
    let page = storage
        .get_verified_record_commitment_block_page(tip_height, 1)
        .await
        .map_err(|_| "local_commitment_tip_unavailable".to_string())?;
    let block = page
        .blocks
        .into_iter()
        .next()
        .filter(|block| block.header.height == tip_height)
        .ok_or_else(|| "local_commitment_tip_unavailable".to_string())?;
    if block.header.proposer != identity.public_key_bytes() {
        return Err("local_commitment_tip_not_self_proposed".to_string());
    }
    let frame = encode_memchain(&MemChainMessage::RecordBlockAnnounceV1 {
        header: block.header,
        proposer_signature: block.proposer_signature,
    })
    .map_err(|_| "tip_announcement_encode_failed".to_string())?;

    let self_node_id = identity.public_key_bytes();
    let mut distinct = HashSet::new();
    let mut outcome = CommitmentTipAnnouncementOutcome {
        announced_height: tip_height,
        ..CommitmentTipAnnouncementOutcome::default()
    };
    let mut pending = pinned_peer_ids
        .iter()
        .copied()
        .filter(|peer_id| *peer_id != self_node_id && distinct.insert(*peer_id))
        .take(MAX_PINNED_WITNESSES_PER_ROUND)
        .collect::<Vec<_>>();
    outcome.attempted = pending.len();

    // Keep the queue hard-bounded even if a future internal caller supplies a
    // malformed policy. Production uses exactly three attempts.
    let max_attempts = retry_policy.max_attempts.clamp(1, 8);
    let mut attempt_number = 1usize;
    while !pending.is_empty() {
        if attempt_number > 1 {
            let shift = u32::try_from(attempt_number.saturating_sub(2))
                .unwrap_or(u32::MAX)
                .min(7);
            let multiplier = 1u32 << shift;
            tokio::time::sleep(retry_policy.base_delay.saturating_mul(multiplier)).await;
            outcome.retries_attempted = outcome.retries_attempted.saturating_add(pending.len());
        }

        let deliveries = futures::stream::iter(pending.into_iter())
            .map(|peer_id| {
                let frame = frame.clone();
                async move {
                    let delivery = deliver_commitment_tip_announcement(
                        peer_store,
                        client,
                        peer_id,
                        frame,
                        endpoint_allowed,
                    )
                    .await;
                    (peer_id, delivery)
                }
            })
            .buffer_unordered(MAX_PINNED_WITNESSES_PER_ROUND)
            .collect::<Vec<_>>()
            .await;
        let mut retry_queue = Vec::with_capacity(deliveries.len());
        for (peer_id, delivery) in deliveries {
            match delivery {
                CommitmentTipAnnouncementDelivery::Accepted => {
                    outcome.accepted = outcome.accepted.saturating_add(1);
                    if attempt_number > 1 {
                        outcome.retries_succeeded = outcome.retries_succeeded.saturating_add(1);
                    }
                }
                CommitmentTipAnnouncementDelivery::Stale => {
                    outcome.stale = outcome.stale.saturating_add(1);
                    if attempt_number > 1 {
                        outcome.retries_succeeded = outcome.retries_succeeded.saturating_add(1);
                    }
                }
                CommitmentTipAnnouncementDelivery::RetryableFailure
                    if attempt_number < max_attempts =>
                {
                    retry_queue.push(peer_id);
                }
                CommitmentTipAnnouncementDelivery::RetryableFailure => {
                    outcome.failed = outcome.failed.saturating_add(1);
                    outcome.retries_exhausted = outcome.retries_exhausted.saturating_add(1);
                }
                CommitmentTipAnnouncementDelivery::PermanentFailure => {
                    outcome.failed = outcome.failed.saturating_add(1);
                }
            }
        }
        pending = retry_queue;
        attempt_number = attempt_number.saturating_add(1);
    }
    Ok(outcome)
}

pub(super) async fn deliver_commitment_tip_announcement<F>(
    peer_store: &PeerStore,
    client: &reqwest::Client,
    peer_id: [u8; 32],
    frame: Vec<u8>,
    endpoint_allowed: &F,
) -> CommitmentTipAnnouncementDelivery
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let Some(peer) = peer_store.get_valid(&peer_id, now_secs()) else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    let Some(endpoint) = peer.descriptor.public_endpoint.as_deref() else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    if !endpoint_allowed(endpoint) {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    }
    let Ok(url) = commitment_block_announce_url(endpoint) else {
        return CommitmentTipAnnouncementDelivery::PermanentFailure;
    };
    match client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
    {
        Ok(response) => classify_commitment_tip_announcement_status(response.status().as_u16()),
        Err(_) => CommitmentTipAnnouncementDelivery::RetryableFailure,
    }
}

pub(super) async fn block_announce_handler(
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
    let MemChainMessage::RecordBlockAnnounceV1 {
        header,
        proposer_signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if header.protocol_version != RECORD_COMMITMENT_BLOCK_VERSION_V1
        || header.chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID
        || header.height == 0
        || header.timestamp == 0
        || header.timestamp > now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS)
        || header.record_count == 0
        || header.record_count as usize > MAX_RECORD_COMMITMENTS_PER_BLOCK
        || (header.height == 1 && header.prev_block_hash != GENESIS_PREV_HASH)
    {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_block_announcement");
    }
    // [COORDINATOR-CONTROL-ADMISSION 2026-08-14 by Codex] Authenticate before
    // consulting authority or PeerStore state. This avoids membership and
    // authority probes while keeping unauthenticated traffic away from the
    // storage-backed authority audit.
    let header_hash = header.hash();
    let signature_valid = IdentityPublicKey::from_bytes(&header.proposer)
        .and_then(|key| key.verify(&header_hash, &proposer_signature))
        .is_ok();
    if !signature_valid {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    let authorized_coordinator = match runtime_authorized_coordinator_for_height(
        &state.storage,
        state.lease_authorized_coordinator,
        header.height,
    )
    .await
    {
        Ok(Some(coordinator)) => coordinator,
        Ok(None) => return protocol_error(StatusCode::FORBIDDEN, "follower_sync_disabled"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited announcement authority");
            return protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "coordinator_authority_unavailable",
            );
        }
    };
    if header.proposer != authorized_coordinator {
        return protocol_error(StatusCode::FORBIDDEN, "coordinator_not_authorized");
    }
    if state.peer_store.get_valid(&header.proposer, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    if !state
        .guard
        .lock()
        .await
        .admit_idempotent_hint(header.proposer, now)
    {
        // Keep the established wire error for older coordinators even though
        // authenticated tip hints are now rejected only by the shared rate cap.
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }
    let (local_tip_height, _) = match verified_local_commitment_tip(&state.storage).await {
        Ok(tip) => tip,
        Err(_) => return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified"),
    };
    if header.height <= local_tip_height {
        state.storage.record_commitment_sync_announcement(
            now,
            header.height,
            RecordCommitmentAnnouncementDisposition::Stale,
        );
        return StatusCode::NO_CONTENT.into_response();
    }
    let Some(notifier) = state.block_announce_notifier.as_ref() else {
        state.storage.record_commitment_sync_announcement(
            now,
            header.height,
            RecordCommitmentAnnouncementDisposition::Unavailable,
        );
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "sync_notifier_unavailable");
    };
    match notifier.try_send(header.height) {
        Ok(()) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Accepted,
            );
            debug!(
                announced_height = header.height,
                local_tip_height, "[MEMCHAIN_BLOCK] Authenticated follower wake-up accepted"
            );
            StatusCode::ACCEPTED.into_response()
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Coalesced,
            );
            debug!(
                announced_height = header.height,
                local_tip_height, "[MEMCHAIN_BLOCK] Authenticated follower wake-up coalesced"
            );
            StatusCode::ACCEPTED.into_response()
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            state.storage.record_commitment_sync_announcement(
                now,
                header.height,
                RecordCommitmentAnnouncementDisposition::Unavailable,
            );
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "sync_notifier_unavailable")
        }
    }
}

pub(super) async fn runtime_authorized_coordinator_for_height(
    storage: &MemoryStorage,
    legacy_coordinator: Option<[u8; 32]>,
    height: u64,
) -> Result<Option<[u8; 32]>, String> {
    // [AUTHORITY-SCHEDULE-RUNTIME 2026-08-14 by Codex] Preserve the legacy
    // static pin when no authority root is configured. Once enabled, only the
    // fully audited append-only schedule may authorise a proposer.
    if storage.record_commitment_authority_enforced() {
        storage.record_commitment_authority_for_height(height).await
    } else {
        Ok(legacy_coordinator)
    }
}

pub(super) async fn runtime_authorized_coordinator_for_next_height(
    storage: &MemoryStorage,
    legacy_coordinator: Option<[u8; 32]>,
) -> Result<Option<[u8; 32]>, String> {
    if storage.record_commitment_authority_enforced() {
        Ok(storage
            .record_commitment_authority_state()
            .await?
            .map(|authority| authority.coordinator))
    } else {
        Ok(legacy_coordinator)
    }
}
