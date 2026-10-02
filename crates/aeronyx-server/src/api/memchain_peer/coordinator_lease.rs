// [ARCH-SPLIT 2026-10-02]
// Coordinator lease request, release, and response verification.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Requests and verifies one short-lived lease from an operator-pinned witness.
///
/// The response authorizes only `instance_id` and the exact audited local tip.
/// It does not expose the current holder when contended and does not establish
/// permissionless consensus, fork choice, or finality.
///
/// # Errors
///
/// Returns a stable privacy-safe code for invalid policy, peer admission,
/// unsafe endpoint, contention, transport failure, stale response, tip
/// mismatch, or signature failure.
pub async fn request_record_commitment_coordinator_lease(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
    client: &reqwest::Client,
) -> Result<CommitmentCoordinatorLeaseGrant, String> {
    request_record_commitment_coordinator_lease_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        witness_node_id,
        instance_id,
        requested_ttl_secs,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

/// Releases one previously acquired witness lease during graceful shutdown.
///
/// A failed or partial release is safe: the unreleased witnesses retain their
/// short expiry and the next process remains fail-closed until it can acquire
/// every configured grant.
pub async fn release_record_commitment_coordinator_lease(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    client: &reqwest::Client,
) -> Result<CommitmentCoordinatorLeaseRelease, String> {
    release_record_commitment_coordinator_lease_with_endpoint_policy(
        peer_store,
        identity,
        witness_node_id,
        instance_id,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) async fn release_record_commitment_coordinator_lease_with_endpoint_policy<F>(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCoordinatorLeaseRelease, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let request_timestamp = now_secs();
    // [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] The caller supplies an
    // operator-pinned witness identity and verifies the signed response against
    // that exact key, so an authentic expired descriptor is only an endpoint
    // recovery hint during cold start.
    let witness = commitment_peer_descriptor(
        peer_store,
        witness_node_id,
        request_timestamp,
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
    )
    .ok_or_else(|| "lease_release_witness_unavailable".to_string())?;
    let endpoint = witness
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "lease_release_witness_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("lease_release_witness_unsafe_endpoint".to_string());
    }
    let url = commitment_coordinator_lease_release_url(endpoint)?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let coordinator = identity.public_key_bytes();
    let signing_bytes = record_coordinator_lease_release_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &coordinator,
        instance_id,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseReleaseRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        coordinator,
        instance_id: *instance_id,
        request_id,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "lease_release_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("lease_release", &error))?;
    if response.status().as_u16() == StatusCode::CONFLICT.as_u16() {
        return Err("lease_release_not_holder".to_string());
    }
    if !response.status().is_success() {
        return Err(format!(
            "lease_release_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    verify_record_commitment_coordinator_lease_release_response(
        &body,
        &request_id,
        &coordinator,
        instance_id,
        witness_node_id,
        now_secs(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn request_record_commitment_coordinator_lease_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    witness_node_id: &[u8; 32],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCoordinatorLeaseGrant, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(MIN_COORDINATOR_LEASE_TTL_SECS_V1..=MAX_COORDINATOR_LEASE_TTL_SECS_V1)
        .contains(&requested_ttl_secs)
    {
        return Err("lease_policy_invalid".to_string());
    }
    let request_timestamp = now_secs();
    // [PINNED-WITNESS-BOOTSTRAP 2026-07-26 by Codex] Lease acquisition has the
    // same explicit identity pin and signed-response boundary as startup
    // checkpoint reconciliation. Descriptor expiry cannot grant authority.
    let witness = commitment_peer_descriptor(
        peer_store,
        witness_node_id,
        request_timestamp,
        CommitmentPeerDescriptorPolicy::AllowExpiredForPinnedWitness,
    )
    .ok_or_else(|| "lease_witness_unavailable".to_string())?;
    let endpoint = witness
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "lease_witness_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("lease_witness_unsafe_endpoint".to_string());
    }
    let url = commitment_coordinator_lease_url(endpoint)?;
    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let coordinator = identity.public_key_bytes();
    let signing_bytes = record_coordinator_lease_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        &coordinator,
        instance_id,
        known_tip_height,
        &known_tip_hash,
        requested_ttl_secs,
        &request_id,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCoordinatorLeaseRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        coordinator,
        instance_id: *instance_id,
        known_tip_height,
        known_tip_hash,
        requested_ttl_secs,
        request_id,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "lease_request_encode_failed".to_string())?;
    let request_started = Instant::now();
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("lease_request", &error))?;
    if response.status().as_u16() == StatusCode::CONFLICT.as_u16() {
        return Err("lease_contended".to_string());
    }
    if !response.status().is_success() {
        return Err(format!("lease_http_status_{}", response.status().as_u16()));
    }
    let body = read_bounded_response(response).await?;
    let mut grant = verify_record_commitment_coordinator_lease_response(
        &body,
        &request_id,
        &coordinator,
        instance_id,
        witness_node_id,
        (known_tip_height, known_tip_hash),
        requested_ttl_secs,
        now_secs(),
    )?;
    grant.valid_for_secs = grant
        .valid_for_secs
        .saturating_sub(request_started.elapsed().as_secs());
    if grant.valid_for_secs == 0 {
        return Err("lease_expired_in_transit".to_string());
    }
    Ok(grant)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_record_commitment_coordinator_lease_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_coordinator: &[u8; 32],
    expected_instance_id: &[u8; 32],
    expected_witness: &[u8; 32],
    expected_tip: (u64, [u8; 32]),
    requested_ttl_secs: u32,
    now: u64,
) -> Result<CommitmentCoordinatorLeaseGrant, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_lease_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_lease_frame")?;
    let canonical = encode_memchain(&response).map_err(|_| "invalid_lease_frame")?;
    if canonical != body {
        return Err("noncanonical_lease_frame".to_string());
    }
    let MemChainMessage::RecordCoordinatorLeaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height,
        witness_tip_hash,
        signature,
    } = response
    else {
        return Err("unexpected_lease_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("lease_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("lease_request_mismatch".to_string());
    }
    if coordinator != *expected_coordinator || instance_id != *expected_instance_id {
        return Err("lease_instance_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("lease_witness_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_lease_response".to_string());
    }
    if (witness_tip_height, witness_tip_hash) != expected_tip {
        return Err("lease_tip_mismatch".to_string());
    }
    if lease_epoch == 0 || lease_expires_at <= now {
        return Err("lease_expiry_invalid".to_string());
    }
    let valid_for_secs = lease_expires_at
        .checked_sub(response_timestamp)
        .ok_or_else(|| "lease_expiry_invalid".to_string())?;
    // The signed remainder can be slightly shorter than the minimum request
    // TTL when persistence and response signing cross a second boundary.
    if valid_for_secs == 0 || valid_for_secs > u64::from(requested_ttl_secs) {
        return Err("lease_duration_invalid".to_string());
    }
    let signing_bytes = record_coordinator_lease_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        response_timestamp,
        lease_epoch,
        lease_expires_at,
        witness_tip_height,
        &witness_tip_hash,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_lease_signature".to_string())?;
    Ok(CommitmentCoordinatorLeaseGrant {
        lease_epoch,
        lease_expires_at,
        valid_for_secs,
    })
}

pub(super) fn verify_record_commitment_coordinator_lease_release_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_coordinator: &[u8; 32],
    expected_instance_id: &[u8; 32],
    expected_witness: &[u8; 32],
    now: u64,
) -> Result<CommitmentCoordinatorLeaseRelease, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_lease_release_frame".to_string());
    }
    let response =
        decode_memchain(&body[1..]).map_err(|_| "invalid_lease_release_frame".to_string())?;
    let canonical =
        encode_memchain(&response).map_err(|_| "invalid_lease_release_frame".to_string())?;
    if canonical != body {
        return Err("noncanonical_lease_release_frame".to_string());
    }
    let MemChainMessage::RecordCoordinatorLeaseReleaseResponseV1 {
        chain_id,
        request_id,
        coordinator,
        instance_id,
        witness,
        released_at,
        lease_epoch,
        signature,
    } = response
    else {
        return Err("unexpected_lease_release_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("lease_release_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("lease_release_request_mismatch".to_string());
    }
    if coordinator != *expected_coordinator || instance_id != *expected_instance_id {
        return Err("lease_release_instance_mismatch".to_string());
    }
    if witness != *expected_witness {
        return Err("lease_release_witness_mismatch".to_string());
    }
    if lease_epoch == 0 || now.abs_diff(released_at) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("lease_release_timestamp_invalid".to_string());
    }
    let signing_bytes = record_coordinator_lease_release_response_signing_bytes(
        &chain_id,
        &request_id,
        &coordinator,
        &instance_id,
        &witness,
        released_at,
        lease_epoch,
    );
    IdentityPublicKey::from_bytes(&witness)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_lease_release_signature".to_string())?;
    Ok(CommitmentCoordinatorLeaseRelease {
        lease_epoch,
        released_at,
    })
}
