// [ARCH-SPLIT 2026-10-02]
// Authenticated descriptor proof fetch and peer-store admission.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

/// Fetches one exact descriptor inclusion proof with bounded availability
/// recovery.
///
/// [REPLICA-PROOF-RECOVERY 2026-07-27 by Codex] The caller supplies all trust
/// anchors: original producer, selected producer-signed block hash, and exact
/// descriptor hash. The producer is contacted first. At most two current,
/// explicitly advertised `DirectoryMirrorCarrier` nodes are considered only
/// after typed transport, route, or admission unavailability.
///
/// A carrier response is accepted only after independently verifying its
/// transport signature and the original producer-signed proof. Noncanonical
/// frames, contract mismatches, bad signatures, invalid Merkle paths, semantic
/// producer absence, and wrong trust anchors stop closed without failover.
///
/// # Errors
/// Returns a stable privacy-safe reason without endpoint, carrier identity,
/// request id, descriptor hash, or response material.
pub async fn fetch_directory_descriptor_inclusion_proof_with_recovery(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    client: &reqwest::Client,
) -> Result<AuthenticatedDirectoryDescriptorProof, String> {
    let requester = identity.public_key_bytes();
    if *producer == [0u8; 32]
        || *producer == requester
        || *block_hash == [0u8; 32]
        || *descriptor_hash == [0u8; 32]
    {
        return Err("directory_descriptor_proof_request_invalid".to_string());
    }

    let direct_timestamp = unix_now_secs();
    let mut direct_attempted = false;
    match directory_descriptor_inclusion_proof_url(peer_store, producer, direct_timestamp) {
        Ok(url) => {
            direct_attempted = true;
            match request_directory_descriptor_inclusion_proof(
                identity,
                producer,
                block_hash,
                descriptor_hash,
                client,
                url,
                direct_timestamp,
            )
            .await
            {
                Ok(proof) => {
                    return Ok(AuthenticatedDirectoryDescriptorProof {
                        proof,
                        transport: DirectoryDescriptorProofTransport::DirectProducer,
                        direct_attempted,
                        carrier_attempts: 0,
                    });
                }
                Err(DirectoryDescriptorProofRequestError::Post(error))
                    if directory_descriptor_proof_direct_post_allows_recovery(error) => {}
                Err(error) => return Err(error.stable_reason("descriptor_proof")),
            }
        }
        Err(reason) if directory_descriptor_proof_direct_url_allows_recovery(&reason) => {}
        Err(reason) => return Err(reason),
    }

    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let selection = directory_mirror_recovery_carriers_with_requirement(
        peer_store,
        &capability_cache,
        producer,
        &requester,
        unix_now_secs(),
        true,
    );
    let mut carrier_attempts = 0u8;
    for carrier in selection
        .carriers
        .into_iter()
        .take(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE)
    {
        let request_timestamp = unix_now_secs();
        let Ok(url) = directory_replica_descriptor_inclusion_proof_url(
            peer_store,
            &carrier,
            request_timestamp,
        ) else {
            continue;
        };
        carrier_attempts = carrier_attempts.saturating_add(1);
        match request_directory_replica_descriptor_inclusion_proof(
            identity,
            producer,
            &carrier.node_id,
            block_hash,
            descriptor_hash,
            client,
            url,
            request_timestamp,
        )
        .await
        {
            Ok(proof) => {
                capability_cache.record_supported(&carrier.node_id);
                return Ok(AuthenticatedDirectoryDescriptorProof {
                    proof,
                    transport: DirectoryDescriptorProofTransport::ReplicaCarrier,
                    direct_attempted,
                    carrier_attempts,
                });
            }
            Err(DirectoryDescriptorProofRequestError::Post(error))
                if directory_descriptor_proof_carrier_post_allows_next(error) =>
            {
                if directory_descriptor_proof_carrier_capability_unavailable(error) {
                    capability_cache
                        .record_unsupported(carrier.node_id, carrier.descriptor_sequence);
                }
            }
            Err(error) => return Err(error.stable_reason("replica_descriptor_proof")),
        }
    }
    Err("directory_descriptor_proof_recovery_exhausted".to_string())
}

/// Fetches and admits one directory-authenticated node descriptor.
///
/// [DIRECTORY-PEER-ADMISSION 2026-07-27 by Codex] The local replica is audited
/// before any network request, preventing a caller from probing an unknown
/// network-selected anchor. After recovery, [`admit_directory_authenticated_descriptor`]
/// repeats the audit and requires exact deterministic proof equality before the
/// existing `PeerStore` signature, validity-window, capacity, and anti-rollback
/// checks run.
///
/// # Errors
/// Returns only stable privacy-safe buckets. It never returns producer,
/// descriptor, block, carrier, endpoint, request, proof, or database material.
pub async fn fetch_and_admit_directory_authenticated_descriptor(
    replica_store: &DirectoryReplicaStore,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    client: &reqwest::Client,
) -> Result<DirectoryAuthenticatedPeerAdmission, String> {
    locally_audited_directory_descriptor_proof(
        replica_store,
        producer,
        block_hash,
        descriptor_hash,
        unix_now_secs(),
    )?;
    let authenticated = fetch_directory_descriptor_inclusion_proof_with_recovery(
        peer_store,
        identity,
        producer,
        block_hash,
        descriptor_hash,
        client,
    )
    .await?;
    admit_directory_authenticated_descriptor(
        replica_store,
        peer_store,
        &authenticated,
        producer,
        block_hash,
        descriptor_hash,
        unix_now_secs(),
    )
}

/// Admits one recovered descriptor only when it exactly matches local replica
/// evidence under caller-supplied trust anchors.
///
/// This function deliberately re-verifies the producer proof even though the
/// network helper already did so: the authenticated wrapper is a public data
/// type and must never become an authority token by construction alone.
///
/// # Errors
/// Returns a stable privacy-safe reason when transport metadata is impossible,
/// proof verification fails, the local anchor is absent/quarantined/corrupt,
/// deterministic evidence differs, or `PeerStore` rejects the descriptor.
pub fn admit_directory_authenticated_descriptor(
    replica_store: &DirectoryReplicaStore,
    peer_store: &PeerStore,
    authenticated: &AuthenticatedDirectoryDescriptorProof,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    observed_at: u64,
) -> Result<DirectoryAuthenticatedPeerAdmission, String> {
    if *producer == [0u8; 32]
        || *block_hash == [0u8; 32]
        || *descriptor_hash == [0u8; 32]
        || observed_at == 0
    {
        return Err("directory_authenticated_admission_request_invalid".to_string());
    }
    if !directory_authenticated_transport_summary_valid(authenticated) {
        return Err("directory_authenticated_admission_transport_invalid".to_string());
    }
    verify_locally_anchored_directory_descriptor_proof(
        replica_store,
        &authenticated.proof,
        producer,
        block_hash,
        descriptor_hash,
        observed_at,
    )?;

    let inserted = peer_store
        .upsert_verified_from_source(
            authenticated.proof.descriptor.clone(),
            observed_at,
            "directory_proof",
        )
        .map_err(|error| directory_authenticated_peer_store_error(&error).to_string())?;
    Ok(DirectoryAuthenticatedPeerAdmission {
        inserted,
        transport: authenticated.transport,
        direct_attempted: authenticated.direct_attempted,
        carrier_attempts: authenticated.carrier_attempts,
    })
}

/// Admits one proof-carrying discovery announcement against exact local
/// Directory replica evidence.
///
/// [DIRECTORY-GOSSIP-ADMISSION 2026-07-27 by Codex] The gossip sender is not
/// an input because it receives no authority. The original producer signature,
/// caller-supplied exact block/hash contract, deterministic local proof, and
/// normal PeerStore anti-rollback checks are the complete admission boundary.
///
/// # Errors
/// Returns only stable privacy-safe buckets when the request contract, producer
/// proof, local anchor, or deterministic proof equality check fails.
pub fn admit_directory_gossip_descriptor(
    replica_store: &DirectoryReplicaStore,
    peer_store: &PeerStore,
    proof: &DirectoryDescriptorInclusionProofV1,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    observed_at: u64,
) -> Result<PeerStoreImportReport, String> {
    verify_locally_anchored_directory_descriptor_proof(
        replica_store,
        proof,
        producer,
        block_hash,
        descriptor_hash,
        observed_at,
    )?;
    Ok(peer_store.apply_verified_descriptor_from_source(
        proof.descriptor.clone(),
        observed_at,
        "directory_gossip_proof",
    ))
}

pub(super) fn verify_locally_anchored_directory_descriptor_proof(
    replica_store: &DirectoryReplicaStore,
    proof: &DirectoryDescriptorInclusionProofV1,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    observed_at: u64,
) -> Result<(), String> {
    if *producer == [0u8; 32]
        || *block_hash == [0u8; 32]
        || *descriptor_hash == [0u8; 32]
        || observed_at == 0
    {
        return Err("directory_authenticated_admission_request_invalid".to_string());
    }
    proof
        .verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer,
            block_hash,
            observed_at,
        )
        .map_err(|_| "directory_authenticated_admission_proof_invalid".to_string())?;
    if proof.commitment.descriptor_hash != *descriptor_hash {
        return Err("directory_authenticated_admission_descriptor_mismatch".to_string());
    }

    let local_proof = locally_audited_directory_descriptor_proof(
        replica_store,
        producer,
        block_hash,
        descriptor_hash,
        observed_at,
    )?;
    if local_proof != *proof {
        return Err("directory_authenticated_admission_local_evidence_mismatch".to_string());
    }
    Ok(())
}

pub(super) fn locally_audited_directory_descriptor_proof(
    replica_store: &DirectoryReplicaStore,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    observed_at: u64,
) -> Result<DirectoryDescriptorInclusionProofV1, String> {
    match replica_store.audited_evidence_descriptor_inclusion_proof(
        producer,
        descriptor_hash,
        block_hash,
        observed_at,
    ) {
        Ok(Some(proof)) => Ok(proof),
        Ok(None) => Err("directory_authenticated_admission_local_anchor_not_found".to_string()),
        Err(DirectoryReplicaStoreError::Quarantined(_)) => {
            Err("directory_authenticated_admission_local_anchor_quarantined".to_string())
        }
        Err(_) => Err("directory_authenticated_admission_local_anchor_audit_failed".to_string()),
    }
}

#[must_use]
pub(super) fn directory_authenticated_transport_summary_valid(
    authenticated: &AuthenticatedDirectoryDescriptorProof,
) -> bool {
    match authenticated.transport {
        DirectoryDescriptorProofTransport::DirectProducer => {
            authenticated.direct_attempted && authenticated.carrier_attempts == 0
        }
        DirectoryDescriptorProofTransport::ReplicaCarrier => {
            authenticated.carrier_attempts > 0
                && usize::from(authenticated.carrier_attempts)
                    <= DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE
        }
    }
}

pub(super) const fn directory_authenticated_peer_store_error(
    error: &PeerStoreError,
) -> &'static str {
    match error {
        PeerStoreError::VerificationFailed => {
            "directory_authenticated_admission_descriptor_invalid"
        }
        PeerStoreError::StaleSequence { .. } => "directory_authenticated_admission_stale_sequence",
        PeerStoreError::CapacityExceeded { .. } => {
            "directory_authenticated_admission_peer_capacity_exceeded"
        }
    }
}

pub(super) const fn directory_descriptor_proof_direct_post_allows_recovery(
    error: DirectoryFramePostError,
) -> bool {
    match error {
        DirectoryFramePostError::Transport(_) => true,
        DirectoryFramePostError::HttpStatus {
            peer_code: Some(_), ..
        }
        | DirectoryFramePostError::Response(_) => false,
        DirectoryFramePostError::HttpStatus {
            status,
            peer_code: None,
        } => matches!(status, 403 | 404 | 405 | 408 | 429) || status >= 500,
    }
}

pub(super) const fn directory_descriptor_proof_carrier_post_allows_next(
    error: DirectoryFramePostError,
) -> bool {
    match error {
        DirectoryFramePostError::Transport(_)
        | DirectoryFramePostError::HttpStatus {
            peer_code:
                Some(
                    DirectoryPeerErrorCode::ReplicaDescriptorProofNotFound
                    | DirectoryPeerErrorCode::MirrorReplicaNotRetained,
                ),
            ..
        } => true,
        DirectoryFramePostError::HttpStatus {
            peer_code: Some(_), ..
        }
        | DirectoryFramePostError::Response(_) => false,
        DirectoryFramePostError::HttpStatus {
            status,
            peer_code: None,
        } => matches!(status, 403 | 404 | 405 | 408 | 429) || status >= 500,
    }
}

pub(super) const fn directory_descriptor_proof_carrier_capability_unavailable(
    error: DirectoryFramePostError,
) -> bool {
    matches!(
        error,
        DirectoryFramePostError::HttpStatus {
            status: 404 | 405 | 501,
            peer_code: None
        }
    )
}

pub(super) fn directory_descriptor_proof_direct_url_allows_recovery(reason: &str) -> bool {
    matches!(
        reason,
        "directory_descriptor_proof_peer_unavailable"
            | "directory_descriptor_proof_peer_missing_endpoint"
            | "directory_descriptor_proof_peer_unsafe_endpoint"
            | "directory_descriptor_proof_peer_invalid_endpoint"
    )
}

pub(super) fn directory_descriptor_inclusion_proof_url(
    peer_store: &PeerStore,
    producer: &[u8; 32],
    request_timestamp: u64,
) -> Result<reqwest::Url, String> {
    let descriptor = peer_store
        .get_valid(producer, request_timestamp)
        .ok_or_else(|| "directory_descriptor_proof_peer_unavailable".to_string())?;
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_descriptor_proof_peer_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_descriptor_proof_peer_unsafe_endpoint".to_string());
    }
    commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/descriptor-inclusion-proof",
    )
    .map_err(|_| "directory_descriptor_proof_peer_invalid_endpoint".to_string())
}

pub(super) fn directory_replica_descriptor_inclusion_proof_url(
    peer_store: &PeerStore,
    carrier: &DirectoryMirrorRecoveryCarrier,
    request_timestamp: u64,
) -> Result<reqwest::Url, String> {
    let descriptor = peer_store
        .get_valid(&carrier.node_id, request_timestamp)
        .ok_or_else(|| "directory_replica_proof_carrier_unavailable".to_string())?;
    if descriptor.sequence() != carrier.descriptor_sequence {
        return Err("directory_replica_proof_carrier_descriptor_changed".to_string());
    }
    if !descriptor.descriptor.policy.public_discovery
        || !descriptor
            .descriptor
            .capabilities
            .contains(&NodeCapability::DirectoryMirrorCarrier)
    {
        return Err("directory_replica_proof_carrier_not_advertised".to_string());
    }
    let endpoint = descriptor
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "directory_replica_proof_carrier_missing_endpoint".to_string())?;
    if !commitment_peer_endpoint_is_public(endpoint) {
        return Err("directory_replica_proof_carrier_unsafe_endpoint".to_string());
    }
    commitment_peer_url(
        endpoint,
        "/api/discovery/peer/directory/replica-descriptor-inclusion-proof",
    )
    .map_err(|_| "directory_replica_proof_carrier_invalid_endpoint".to_string())
}

pub(super) async fn request_directory_descriptor_inclusion_proof(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    client: &reqwest::Client,
    proof_url: reqwest::Url,
    request_timestamp: u64,
) -> Result<DirectoryDescriptorInclusionProofV1, DirectoryDescriptorProofRequestError> {
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_descriptor_inclusion_proof_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        block_hash,
        descriptor_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::DescriptorInclusionProofRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        block_hash: *block_hash,
        descriptor_hash: *descriptor_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request).map_err(|_| {
        DirectoryDescriptorProofRequestError::FailClosed(
            "directory_descriptor_proof_request_encode_failed".to_string(),
        )
    })?;
    let response = post_directory_frame_typed(client, proof_url, frame)
        .await
        .map_err(DirectoryDescriptorProofRequestError::Post)?;
    verify_descriptor_inclusion_proof_response(
        &response,
        &request_id,
        producer,
        block_hash,
        descriptor_hash,
        request_timestamp,
        unix_now_secs(),
    )
    .map_err(DirectoryDescriptorProofRequestError::FailClosed)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn request_directory_replica_descriptor_inclusion_proof(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    block_hash: &[u8; 32],
    descriptor_hash: &[u8; 32],
    client: &reqwest::Client,
    proof_url: reqwest::Url,
    request_timestamp: u64,
) -> Result<DirectoryDescriptorInclusionProofV1, DirectoryDescriptorProofRequestError> {
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_replica_descriptor_inclusion_proof_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer,
        block_hash,
        descriptor_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::ReplicaDescriptorInclusionProofRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer: *producer,
        block_hash: *block_hash,
        descriptor_hash: *descriptor_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request).map_err(|_| {
        DirectoryDescriptorProofRequestError::FailClosed(
            "directory_replica_proof_request_encode_failed".to_string(),
        )
    })?;
    let response = post_directory_frame_typed(client, proof_url, frame)
        .await
        .map_err(DirectoryDescriptorProofRequestError::Post)?;
    verify_replica_descriptor_inclusion_proof_response(
        &response,
        &request_id,
        producer,
        carrier,
        block_hash,
        descriptor_hash,
        request_timestamp,
        unix_now_secs(),
    )
    .map_err(DirectoryDescriptorProofRequestError::FailClosed)
}

pub(crate) fn verify_descriptor_inclusion_proof_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_block_hash: &[u8; 32],
    expected_descriptor_hash: &[u8; 32],
    request_timestamp: u64,
    observed_at: u64,
) -> Result<DirectoryDescriptorInclusionProofV1, String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_descriptor_proof_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_descriptor_proof_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_descriptor_proof_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::DescriptorInclusionProofResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        block_hash,
        descriptor_hash,
        proof,
        signature,
    } = message
    else {
        return Err("directory_descriptor_proof_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || responder != *expected_producer
        || response_timestamp.abs_diff(observed_at) > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || block_hash != *expected_block_hash
        || descriptor_hash != *expected_descriptor_hash
    {
        return Err("directory_descriptor_proof_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_descriptor_inclusion_proof_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        &block_hash,
        &descriptor_hash,
        &proof,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_descriptor_proof_response_invalid_signature".to_string())?;
    proof
        .verify_at(
            &chain_id,
            expected_producer,
            expected_block_hash,
            observed_at,
        )
        .map_err(|_| "directory_descriptor_proof_response_invalid_proof".to_string())?;
    if proof.commitment.descriptor_hash != *expected_descriptor_hash {
        return Err("directory_descriptor_proof_response_descriptor_mismatch".to_string());
    }
    Ok(proof)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_replica_descriptor_inclusion_proof_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_carrier: &[u8; 32],
    expected_block_hash: &[u8; 32],
    expected_descriptor_hash: &[u8; 32],
    request_timestamp: u64,
    observed_at: u64,
) -> Result<DirectoryDescriptorInclusionProofV1, String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_replica_proof_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_replica_proof_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_replica_proof_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ReplicaDescriptorInclusionProofResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        block_hash,
        descriptor_hash,
        proof,
        signature,
    } = message
    else {
        return Err("directory_replica_proof_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || producer != *expected_producer
        || carrier != *expected_carrier
        || carrier == producer
        || response_timestamp.abs_diff(observed_at) > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || block_hash != *expected_block_hash
        || descriptor_hash != *expected_descriptor_hash
    {
        return Err("directory_replica_proof_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_replica_descriptor_inclusion_proof_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &block_hash,
        &descriptor_hash,
        &proof,
    );
    IdentityPublicKey::from_bytes(&carrier)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_replica_proof_response_invalid_carrier_signature".to_string())?;
    proof
        .verify_at(
            &chain_id,
            expected_producer,
            expected_block_hash,
            observed_at,
        )
        .map_err(|_| "directory_replica_proof_response_invalid_producer_proof".to_string())?;
    if proof.commitment.descriptor_hash != *expected_descriptor_hash {
        return Err("directory_replica_proof_response_descriptor_mismatch".to_string());
    }
    Ok(proof)
}
