// [ARCH-SPLIT 2026-10-02]
// Authenticate a directory peer request and run the blocking store call.
// Bodies are unchanged. Private inherent items are pub(super) so the parent flow can call them.
use super::*;

pub(super) async fn authenticate_request(
    state: &DirectoryChainPeerState,
    admission: DirectoryPeerAdmission,
    requester: [u8; 32],
    request_id: [u8; 16],
    request_timestamp: u64,
    signing_bytes: &[u8],
    signature: &[u8; 64],
    now: u64,
) -> Result<(), Response> {
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err(protocol_error(StatusCode::UNAUTHORIZED, "stale_request"));
    }
    if admission == DirectoryPeerAdmission::PinnedAuthority
        && !state.pinned_peers.contains(&requester)
    {
        return Err(protocol_error(StatusCode::FORBIDDEN, "peer_not_pinned"));
    }
    let requester_is_pinned = state.pinned_peers.contains(&requester);
    let permissionless_read = matches!(
        admission,
        DirectoryPeerAdmission::VerifiedPublicMirror
            | DirectoryPeerAdmission::VerifiedPublicRecovery
    );
    if permissionless_read && !state.allow_public_mirror_reads && !requester_is_pinned {
        return Err(protocol_error(
            StatusCode::FORBIDDEN,
            "public_mirror_disabled",
        ));
    }
    let Some(descriptor) = state.peer_store.get_valid(&requester, now) else {
        return Err(protocol_error(StatusCode::FORBIDDEN, "unknown_peer"));
    };
    let public_descriptor_required = admission == DirectoryPeerAdmission::VerifiedPublicMirror
        || (admission == DirectoryPeerAdmission::VerifiedPublicRecovery && !requester_is_pinned);
    if public_descriptor_required && !descriptor.descriptor.policy.public_discovery {
        return Err(protocol_error(StatusCode::FORBIDDEN, "peer_not_public"));
    }
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(signing_bytes, signature))
        .is_err()
    {
        return Err(protocol_error(
            StatusCode::UNAUTHORIZED,
            "invalid_signature",
        ));
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return Err(protocol_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_or_replay_limited",
        ));
    }
    Ok(())
}

pub(super) fn decode_request(body: &[u8]) -> Result<DirectorySyncMessage, Response> {
    let message = decode_directory_sync_message(body)
        .map_err(|_| protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"))?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"))?;
    if canonical != body {
        return Err(protocol_error(
            StatusCode::BAD_REQUEST,
            "noncanonical_frame",
        ));
    }
    Ok(message)
}

pub(super) fn bounded_directory_transport_blocks(
    blocks: Vec<DirectoryCommitmentBlockV1>,
) -> Vec<DirectoryCommitmentBlockV1> {
    let mut commitment_count = 0usize;
    let mut descriptor_hashes = HashSet::new();
    let mut accepted = 0usize;
    for block in &blocks {
        let Some(next_count) = commitment_count.checked_add(block.commitments.len()) else {
            break;
        };
        if next_count > MAX_DIRECTORY_COMMITMENTS_PER_BLOCK
            || block
                .commitments
                .iter()
                .any(|commitment| descriptor_hashes.contains(&commitment.descriptor_hash))
        {
            break;
        }
        descriptor_hashes.extend(
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash),
        );
        commitment_count = next_count;
        accepted = accepted.saturating_add(1);
    }
    blocks.into_iter().take(accepted).collect()
}

pub(super) fn encoded_response(message: DirectorySyncMessage) -> Response {
    match encode_directory_sync_message(&message) {
        Ok(encoded) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream")],
            encoded,
        )
            .into_response(),
        Err(error) => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Failed to encode peer response");
            protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error")
        }
    }
}

pub(super) fn store_error_response(error: &DirectoryChainStoreError) -> Response {
    match error {
        DirectoryChainStoreError::Request(_) => {
            protocol_error(StatusCode::BAD_REQUEST, "invalid_request")
        }
        _ => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Refused unaudited peer export");
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "chain_not_verified")
        }
    }
}

pub(super) fn replica_store_error_response(error: &DirectoryReplicaStoreError) -> Response {
    match error {
        DirectoryReplicaStoreError::Request(_) => {
            protocol_error(StatusCode::BAD_REQUEST, "invalid_replica_request")
        }
        // [MIRROR-CARRIER 2026-07-24 by Codex] A lagging carrier is an
        // availability miss, not evidence that the signed request is invalid.
        // A distinct 404 keeps true 400 contract failures fail-closed while the
        // requester advances to another bounded verified carrier.
        DirectoryReplicaStoreError::RangeNotRetained { .. } => {
            protocol_error(StatusCode::NOT_FOUND, "replica_range_not_retained")
        }
        DirectoryReplicaStoreError::Quarantined(_) => {
            protocol_error(StatusCode::CONFLICT, "producer_quarantined")
        }
        DirectoryReplicaStoreError::MirrorNotRetained => {
            protocol_error(StatusCode::NOT_FOUND, "mirror_replica_not_retained")
        }
        _ => {
            warn!(error = %error, "[DIRECTORY_CHAIN] Refused unaudited replica export");
            protocol_error(StatusCode::SERVICE_UNAVAILABLE, "replica_not_verified")
        }
    }
}

pub(super) fn protocol_error(status: StatusCode, code: &'static str) -> Response {
    (status, [(header::CONTENT_TYPE, "text/plain")], code).into_response()
}

/// Runs one synchronous Directory peer operation on Tokio's blocking pool.
///
/// [DIRECTORY-BLOCKING-BOUNDARY 2026-07-30 by Codex] `operation` must be a
/// static code-defined role. A failed join always keeps the established
/// `audit_task_failed` protocol bucket while logs receive only the shared fixed
/// category, never the potentially payload-bearing raw `JoinError`.
pub(super) async fn run_directory_chain_blocking<T, F>(
    admission: Arc<Semaphore>,
    operation: &'static str,
    worker: F,
) -> Result<T, Response>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    run_directory_chain_blocking_with_admission(admission, operation, worker).await
}

pub(super) async fn run_directory_chain_blocking_with_admission<T, F>(
    admission: Arc<Semaphore>,
    operation: &'static str,
    worker: F,
) -> Result<T, Response>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    // [DIRECTORY-AUDIT-ADMISSION 2026-08-12 by Codex] Acquire before
    // `spawn_blocking`: rejected work never occupies the blocking queue. The
    // owned permit moves into the worker and is released on return or unwind.
    let permit = admission.try_acquire_owned().map_err(|_| {
        debug!(operation, "[DIRECTORY_CHAIN] Audit admission busy");
        protocol_error(StatusCode::SERVICE_UNAVAILABLE, "audit_busy")
    })?;
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        worker()
    })
    .await
    {
        Ok(result) => Ok(result),
        Err(error) => {
            warn!(
                operation,
                failure = ?RuntimeTaskJoinFailureKind::classify(&error),
                "[DIRECTORY_CHAIN] Blocking worker failed closed"
            );
            Err(protocol_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "audit_task_failed",
            ))
        }
    }
}

pub(super) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
