// [ARCH-SPLIT 2026-10-02]
// Directory page pull, descriptor hydration, and range import.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

/// Pulls, verifies, hydrates, and atomically imports one pinned producer page.
///
/// The producer must have a current signed descriptor in `PeerStore`, and its
/// endpoint must be a public IP literal. Every response is canonicalized and
/// signature-verified before the blocking atomic import begins.
///
/// # Errors
/// Returns a stable privacy-safe reason code for unavailable descriptors,
/// unsafe endpoints, transport/status/body failures, invalid signed responses,
/// missing objects, replica integrity failures, or durable quarantine.
pub async fn pull_directory_chain_page(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "replica_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("producer_quarantined".to_string());
    }
    let (range_url, object_url) =
        directory_sync_peer_urls(peer_store, producer, request_timestamp)?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "replica_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_block_page(
        identity,
        producer,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_descriptor_objects(
        identity,
        producer,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_range_page(replica_store, *producer, page, objects, requests_made).await
}

/// Pulls one signed page into the bounded non-authoritative mirror set.
///
/// The producer is always tried first. Only availability/admission failures may
/// enter the bounded carrier path; canonical, signature, producer-binding,
/// descriptor, hash-chain, and durable integrity failures stop immediately.
pub(super) async fn pull_directory_chain_mirror_page_with_recovery(
    replica_store: Arc<DirectoryReplicaStore>,
    runtime: &DirectoryReplicaSyncRuntime,
    peer_store: &PeerStore,
    carrier_capabilities: &DirectoryMirrorCarrierCapabilityCache,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    client: &reqwest::Client,
) -> Result<(DirectorySyncPullOutcome, DirectoryMirrorPullSource), DirectoryMirrorPullFailure> {
    match pull_directory_chain_mirror_page(
        Arc::clone(&replica_store),
        peer_store,
        identity,
        producer,
        descriptor_sequence,
        max_mirror_producers,
        client,
    )
    .await
    {
        Ok(outcome) => Ok((outcome, DirectoryMirrorPullSource::DirectProducer)),
        Err(reason) if directory_mirror_failure_allows_recovery(&reason) => {
            // [MIRROR-CATCHUP 2026-07-24 by Codex] Conservatively reserve one
            // request for the direct attempt even when endpoint validation may
            // have failed before transport. Each retryable carrier failure can
            // consume at most its range request before another carrier is used.
            let mut prior_requests = 1u32;
            let carrier_selection = directory_mirror_recovery_carriers(
                peer_store,
                carrier_capabilities,
                producer,
                &identity.public_key_bytes(),
                unix_now_secs(),
            );
            // [MIRROR-DIVERSITY 2026-07-24 by Codex] Persist only aggregate
            // selection properties. Carrier identities, endpoints, regions,
            // producer identities, and route order never enter telemetry.
            runtime.record_full_node_mirror_carrier_selection(
                carrier_selection.candidate_count,
                carrier_selection.routeable_candidate_count,
                carrier_selection.explicitly_advertised_candidate_count,
                carrier_selection.unadvertised_compatibility_candidate_count,
                carrier_selection.capability_cached_unavailable_count,
                u64::try_from(carrier_selection.carriers.len()).unwrap_or(u64::MAX),
                carrier_selection.selected_routeable_count,
                carrier_selection.selected_explicitly_advertised_count,
                carrier_selection.selected_unadvertised_compatibility_count,
                carrier_selection.selected_region_hint_count,
                carrier_selection.distinct_selected_region_hint_count,
            );
            for carrier in carrier_selection.carriers {
                match pull_directory_chain_mirror_page_via_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    descriptor_sequence,
                    max_mirror_producers,
                    &carrier.node_id,
                    carrier.descriptor_sequence,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        carrier_capabilities.record_supported(&carrier.node_id);
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        return Ok((outcome, DirectoryMirrorPullSource::PublicCarrier));
                    }
                    Err(carrier_reason)
                        if directory_mirror_failure_allows_recovery(&carrier_reason) =>
                    {
                        if directory_mirror_carrier_capability_unavailable(&carrier_reason) {
                            carrier_capabilities
                                .record_unsupported(carrier.node_id, carrier.descriptor_sequence);
                        }
                        prior_requests = prior_requests.saturating_add(1);
                        debug!(
                            reason = carrier_reason,
                            "[DIRECTORY_REPLICA] Full-node Mirror recovery carrier unavailable"
                        );
                    }
                    Err(carrier_reason) => {
                        return Err(DirectoryMirrorPullFailure {
                            reason: carrier_reason,
                            recovery_attempted: true,
                        });
                    }
                }
            }
            Err(DirectoryMirrorPullFailure {
                reason: "directory_mirror_recovery_exhausted".to_string(),
                recovery_attempted: true,
            })
        }
        Err(reason) => Err(DirectoryMirrorPullFailure {
            reason,
            recovery_attempted: false,
        }),
    }
}

/// Pulls one direct signed page into the bounded non-authoritative mirror set.
///
/// The exact discovery descriptor sequence selected for this attempt must still
/// be current and public when URLs are derived. This function performs no
/// fallback and never alters configured checkpoint/witness authority membership.
pub(super) async fn pull_directory_chain_mirror_page(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "directory_mirror_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("directory_mirror_producer_quarantined".to_string());
    }
    let (range_url, object_url) =
        directory_mirror_peer_urls(peer_store, producer, descriptor_sequence, request_timestamp)?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "directory_mirror_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_block_page(
        identity,
        producer,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_descriptor_objects(
        identity,
        producer,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_mirror_range_page(
        replica_store,
        *producer,
        descriptor_sequence,
        max_mirror_producers,
        page,
        objects,
        requests_made,
    )
    .await
}

// Each argument is a distinct authenticated protocol boundary. Grouping them
// into an opaque context would make producer/carrier confusion easier during
// security review, so keep the identities and bounded retention policy explicit.
#[allow(clippy::too_many_arguments)]
pub(super) async fn pull_directory_chain_mirror_page_via_carrier(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    carrier: &[u8; 32],
    carrier_descriptor_sequence: u64,
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "directory_mirror_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("directory_mirror_producer_quarantined".to_string());
    }
    let (range_url, object_url) = directory_mirror_recovery_carrier_urls(
        peer_store,
        carrier,
        carrier_descriptor_sequence,
        request_timestamp,
    )?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "directory_mirror_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_replica_block_page(
        identity,
        producer,
        carrier,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_replica_descriptor_objects(
        identity,
        producer,
        carrier,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_mirror_range_page(
        replica_store,
        *producer,
        descriptor_sequence,
        max_mirror_producers,
        page,
        objects,
        requests_made,
    )
    .await
}

pub(super) async fn pull_directory_chain_page_with_carriers(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carriers: &[[u8; 32]],
    carrier_capabilities: &DirectoryMirrorCarrierCapabilityCache,
    client: &reqwest::Client,
) -> Result<(DirectorySyncPullOutcome, DirectoryMirrorPullSource), String> {
    match pull_directory_chain_page(
        Arc::clone(&replica_store),
        peer_store,
        identity,
        producer,
        client,
    )
    .await
    {
        Ok(outcome) => Ok((outcome, DirectoryMirrorPullSource::DirectProducer)),
        Err(reason) if directory_sync_failure_allows_carrier_fallback(&reason) => {
            // [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] Bound every
            // availability fallback and account conservatively for the failed
            // direct range even when endpoint validation consumed no request.
            let requester = identity.public_key_bytes();
            let mut prior_requests = 1u32;
            let mut attempted = HashSet::new();
            for carrier in carriers
                .iter()
                .copied()
                .filter(|candidate| candidate != producer && *candidate != requester)
                .take(DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE)
            {
                attempted.insert(carrier);
                match pull_directory_chain_page_via_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    &carrier,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        debug!(
                            requests_made = outcome.requests_made,
                            "[DIRECTORY_REPLICA] Pinned carrier recovered producer evidence"
                        );
                        return Ok((outcome, DirectoryMirrorPullSource::PublicCarrier));
                    }
                    Err(carrier_reason)
                        if directory_sync_failure_allows_carrier_fallback(&carrier_reason) =>
                    {
                        prior_requests = prior_requests.saturating_add(1);
                    }
                    Err(carrier_reason) => return Err(carrier_reason),
                }
            }

            let selection = directory_mirror_recovery_carriers_with_requirement(
                peer_store,
                carrier_capabilities,
                producer,
                &requester,
                unix_now_secs(),
                true,
            );
            for carrier in selection.carriers {
                if !attempted.insert(carrier.node_id) {
                    continue;
                }
                match pull_directory_chain_pinned_page_via_discovered_carrier(
                    Arc::clone(&replica_store),
                    peer_store,
                    identity,
                    producer,
                    carrier,
                    client,
                )
                .await
                {
                    Ok(mut outcome) => {
                        carrier_capabilities.record_supported(&carrier.node_id);
                        outcome.requests_made =
                            outcome.requests_made.saturating_add(prior_requests);
                        debug!(
                            requests_made = outcome.requests_made,
                            "[DIRECTORY_REPLICA] Explicit public carrier cold-recovered pinned producer evidence"
                        );
                        return Ok((outcome, DirectoryMirrorPullSource::PublicCarrier));
                    }
                    Err(carrier_failure)
                        if directory_mirror_failure_allows_recovery(&carrier_failure.reason) =>
                    {
                        if directory_mirror_carrier_capability_unavailable(&carrier_failure.reason)
                        {
                            carrier_capabilities
                                .record_unsupported(carrier.node_id, carrier.descriptor_sequence);
                        }
                        prior_requests =
                            prior_requests.saturating_add(carrier_failure.requests_made);
                    }
                    Err(carrier_failure) => return Err(carrier_failure.reason),
                }
            }
            Err("directory_carrier_fallback_exhausted".to_string())
        }
        Err(reason) => Err(reason),
    }
}

pub(super) async fn pull_directory_chain_page_via_carrier(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, String> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| "replica_tip_unavailable".to_string())?;
    if local_tip.quarantined {
        return Err("producer_quarantined".to_string());
    }
    let (range_url, object_url) =
        directory_replica_carrier_urls(peer_store, carrier, request_timestamp)?;
    let from_height = local_tip
        .tip_height
        .checked_add(1)
        .ok_or_else(|| "replica_height_exhausted".to_string())?;
    let requester = identity.public_key_bytes();
    let page = request_directory_replica_block_page(
        identity,
        producer,
        carrier,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await?;
    let (objects, requests_made) = hydrate_directory_replica_descriptor_objects(
        identity,
        producer,
        carrier,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_range_page(replica_store, *producer, page, objects, requests_made).await
}

/// Pull one pinned-producer page through a permissionless carrier whose exact
/// signed descriptor sequence advertised `DirectoryMirrorCarrier`.
///
/// [CARRIER-COLD-BOOTSTRAP 2026-07-26 by Codex] The carrier signs only the
/// transport envelope. `import_directory_range_page` independently verifies
/// the pinned producer's block signatures, genesis/hash chain, advertised tip,
/// and exact descriptor commitments before an atomic import.
pub(super) async fn pull_directory_chain_pinned_page_via_discovered_carrier(
    replica_store: Arc<DirectoryReplicaStore>,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: DirectoryMirrorRecoveryCarrier,
    client: &reqwest::Client,
) -> Result<DirectorySyncPullOutcome, DirectoryCarrierPullFailure> {
    let request_timestamp = unix_now_secs();
    let local_tip = replica_store
        .producer_tip(producer)
        .map_err(|_| DirectoryCarrierPullFailure::new("replica_tip_unavailable".to_string(), 0))?;
    if local_tip.quarantined {
        return Err(DirectoryCarrierPullFailure::new(
            "producer_quarantined".to_string(),
            0,
        ));
    }
    let (range_url, object_url) = directory_mirror_recovery_carrier_urls(
        peer_store,
        &carrier.node_id,
        carrier.descriptor_sequence,
        request_timestamp,
    )
    .map_err(|reason| DirectoryCarrierPullFailure::new(reason, 0))?;
    let from_height = local_tip.tip_height.checked_add(1).ok_or_else(|| {
        DirectoryCarrierPullFailure::new("replica_height_exhausted".to_string(), 0)
    })?;
    let requester = identity.public_key_bytes();
    let page = request_directory_replica_block_page(
        identity,
        producer,
        &carrier.node_id,
        client,
        range_url,
        from_height,
        request_timestamp,
    )
    .await
    .map_err(|reason| DirectoryCarrierPullFailure::new(reason, 1))?;
    let (objects, requests_made) = hydrate_directory_replica_descriptor_objects_tracked(
        identity,
        producer,
        &carrier.node_id,
        client,
        object_url,
        &requester,
        &page.blocks,
    )
    .await?;
    import_directory_range_page(replica_store, *producer, page, objects, requests_made)
        .await
        .map_err(|reason| DirectoryCarrierPullFailure::new(reason, requests_made))
}

pub(super) async fn request_directory_block_page(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    client: &reqwest::Client,
    range_url: reqwest::Url,
    from_height: u64,
    request_timestamp: u64,
) -> Result<DirectoryRangePage, String> {
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        from_height,
        OUTBOUND_BLOCKS_PER_PAGE,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::BlockRangeRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        from_height,
        limit: OUTBOUND_BLOCKS_PER_PAGE,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request)
        .map_err(|_| "directory_range_request_encode_failed".to_string())?;
    let signed_response = post_directory_frame(client, range_url, frame, "range").await?;
    let (blocks, has_more, remote_tip_height, remote_tip_hash) = verify_block_range_response(
        &signed_response,
        &request_id,
        producer,
        from_height,
        request_timestamp,
    )?;
    Ok(DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    })
}

pub(super) async fn request_directory_replica_block_page(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
    range_url: reqwest::Url,
    from_height: u64,
    request_timestamp: u64,
) -> Result<DirectoryRangePage, String> {
    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = directory_replica_block_range_request_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer,
        from_height,
        OUTBOUND_BLOCKS_PER_PAGE,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = DirectorySyncMessage::ReplicaBlockRangeRequestV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        producer: *producer,
        from_height,
        limit: OUTBOUND_BLOCKS_PER_PAGE,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&request)
        .map_err(|_| "directory_replica_range_request_encode_failed".to_string())?;
    let signed_response = post_directory_frame(client, range_url, frame, "replica_range").await?;
    let (blocks, has_more, remote_tip_height, remote_tip_hash) =
        verify_replica_block_range_response(
            &signed_response,
            &request_id,
            producer,
            carrier,
            from_height,
            request_timestamp,
        )?;
    Ok(DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    })
}

pub(super) async fn hydrate_directory_descriptor_objects(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    client: &reqwest::Client,
    object_url: reqwest::Url,
    requester: &[u8; 32],
    blocks: &[DirectoryCommitmentBlockV1],
) -> Result<(Vec<SignedNodeDescriptor>, u32), String> {
    let descriptor_hashes = blocks
        .iter()
        .flat_map(|block| {
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash)
        })
        .collect::<Vec<_>>();
    let requests_made = directory_sync_request_count_for_objects(descriptor_hashes.len());
    let mut objects = Vec::with_capacity(descriptor_hashes.len());
    for hashes in descriptor_hashes.chunks(MAX_DIRECTORY_SYNC_OBJECTS_V1) {
        let request_timestamp = unix_now_secs();
        let mut request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut request_id);
        let signing_bytes = directory_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            hashes,
            &request_id,
            requester,
            request_timestamp,
        );
        let request = DirectorySyncMessage::DescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            descriptor_hashes: hashes.to_vec(),
            request_id,
            requester: *requester,
            request_timestamp,
            signature: identity.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&request)
            .map_err(|_| "directory_object_request_encode_failed".to_string())?;
        let response = post_directory_frame(client, object_url.clone(), frame, "objects").await?;
        let mut verified = verify_descriptor_objects_response(
            &response,
            &request_id,
            producer,
            hashes,
            request_timestamp,
        )?;
        objects.append(&mut verified);
    }
    Ok((objects, requests_made))
}

pub(super) async fn hydrate_directory_replica_descriptor_objects(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
    object_url: reqwest::Url,
    requester: &[u8; 32],
    blocks: &[DirectoryCommitmentBlockV1],
) -> Result<(Vec<SignedNodeDescriptor>, u32), String> {
    hydrate_directory_replica_descriptor_objects_tracked(
        identity, producer, carrier, client, object_url, requester, blocks,
    )
    .await
    .map_err(|failure| failure.reason)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn hydrate_directory_replica_descriptor_objects_tracked(
    identity: &IdentityKeyPair,
    producer: &[u8; 32],
    carrier: &[u8; 32],
    client: &reqwest::Client,
    object_url: reqwest::Url,
    requester: &[u8; 32],
    blocks: &[DirectoryCommitmentBlockV1],
) -> Result<(Vec<SignedNodeDescriptor>, u32), DirectoryCarrierPullFailure> {
    // [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Count the successful
    // range plus each object request at its dispatch boundary so carrier
    // failover cannot reset or understate the operator smoke budget.
    let descriptor_hashes = blocks
        .iter()
        .flat_map(|block| {
            block
                .commitments
                .iter()
                .map(|commitment| commitment.descriptor_hash)
        })
        .collect::<Vec<_>>();
    // The successful range request has already been consumed before hydration.
    let mut requests_made = 1u32;
    let mut objects = Vec::with_capacity(descriptor_hashes.len());
    for hashes in descriptor_hashes.chunks(MAX_DIRECTORY_SYNC_OBJECTS_V1) {
        let request_timestamp = unix_now_secs();
        let mut request_id = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut request_id);
        let signing_bytes = directory_replica_descriptor_objects_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer,
            hashes,
            &request_id,
            requester,
            request_timestamp,
        );
        let request = DirectorySyncMessage::ReplicaDescriptorObjectsRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            producer: *producer,
            descriptor_hashes: hashes.to_vec(),
            request_id,
            requester: *requester,
            request_timestamp,
            signature: identity.sign(&signing_bytes),
        };
        let frame = encode_directory_sync_message(&request).map_err(|_| {
            DirectoryCarrierPullFailure::new(
                "directory_replica_object_request_encode_failed".to_string(),
                requests_made,
            )
        })?;
        requests_made = requests_made.saturating_add(1);
        let response = post_directory_frame(client, object_url.clone(), frame, "replica_objects")
            .await
            .map_err(|reason| DirectoryCarrierPullFailure::new(reason, requests_made))?;
        let mut verified = verify_replica_descriptor_objects_response(
            &response,
            &request_id,
            producer,
            carrier,
            hashes,
            request_timestamp,
        )
        .map_err(|reason| DirectoryCarrierPullFailure::new(reason, requests_made))?;
        objects.append(&mut verified);
    }
    Ok((objects, requests_made))
}

pub(super) async fn import_directory_range_page(
    replica_store: Arc<DirectoryReplicaStore>,
    producer: [u8; 32],
    page: DirectoryRangePage,
    objects: Vec<SignedNodeDescriptor>,
    requests_made: u32,
) -> Result<DirectorySyncPullOutcome, String> {
    let DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    } = page;
    let import = tokio::task::spawn_blocking(move || {
        replica_store.import_verified_page(
            producer,
            &blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &signed_response,
            unix_now_secs(),
        )
    })
    .await
    .map_err(|_| "directory_replica_import_task_failed".to_string())?
    .map_err(|error| match error {
        DirectoryReplicaStoreError::Quarantined(_) => "producer_quarantined".to_string(),
        _ => "directory_replica_import_rejected".to_string(),
    })?;
    Ok(DirectorySyncPullOutcome {
        import,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        requests_made,
    })
}

pub(super) async fn import_directory_mirror_range_page(
    replica_store: Arc<DirectoryReplicaStore>,
    producer: [u8; 32],
    descriptor_sequence: u64,
    max_mirror_producers: usize,
    page: DirectoryRangePage,
    objects: Vec<SignedNodeDescriptor>,
    requests_made: u32,
) -> Result<DirectorySyncPullOutcome, String> {
    let DirectoryRangePage {
        blocks,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        signed_response,
    } = page;
    let import = tokio::task::spawn_blocking(move || {
        replica_store.import_verified_mirror_page(
            producer,
            descriptor_sequence,
            max_mirror_producers,
            &blocks,
            &objects,
            remote_tip_height,
            remote_tip_hash,
            &signed_response,
            unix_now_secs(),
        )
    })
    .await
    .map_err(|_| "directory_mirror_import_task_failed".to_string())?
    .map_err(|error| match error {
        DirectoryReplicaStoreError::MirrorCapacity => "directory_mirror_capacity_full".to_string(),
        DirectoryReplicaStoreError::Quarantined(_) => {
            "directory_mirror_producer_quarantined".to_string()
        }
        _ => "directory_mirror_import_rejected".to_string(),
    })?;
    Ok(DirectorySyncPullOutcome {
        import,
        has_more,
        remote_tip_height,
        remote_tip_hash,
        requests_made,
    })
}

pub(crate) fn verify_block_range_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_from_height: u64,
    request_timestamp: u64,
) -> Result<
    (
        Vec<aeronyx_core::protocol::discovery::DirectoryCommitmentBlockV1>,
        bool,
        u64,
        [u8; 32],
    ),
    String,
> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_range_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_range_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_range_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::BlockRangeResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("directory_range_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || responder != *expected_producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || blocks.len() > usize::from(OUTBOUND_BLOCKS_PER_PAGE)
        || blocks
            .first()
            .is_some_and(|block| block.header.height != expected_from_height)
        || blocks
            .iter()
            .any(|block| block.header.producer != *expected_producer)
    {
        return Err("directory_range_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_block_range_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_range_response_invalid_signature".to_string())?;
    Ok((blocks, has_more, tip_height, tip_hash))
}

pub(crate) fn verify_replica_block_range_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_carrier: &[u8; 32],
    expected_from_height: u64,
    request_timestamp: u64,
) -> Result<(Vec<DirectoryCommitmentBlockV1>, bool, u64, [u8; 32]), String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_replica_range_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_replica_range_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_replica_range_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        blocks,
        has_more,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("directory_replica_range_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || producer != *expected_producer
        || carrier != *expected_carrier
        || carrier == producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || blocks.len() > usize::from(OUTBOUND_BLOCKS_PER_PAGE)
        || blocks
            .first()
            .is_some_and(|block| block.header.height != expected_from_height)
        || blocks
            .iter()
            .any(|block| block.header.producer != *expected_producer)
    {
        return Err("directory_replica_range_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_replica_block_range_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &blocks,
        has_more,
        tip_height,
        &tip_hash,
    );
    IdentityPublicKey::from_bytes(&carrier)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_replica_range_response_invalid_signature".to_string())?;
    Ok((blocks, has_more, tip_height, tip_hash))
}

pub(crate) fn verify_descriptor_objects_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_hashes: &[[u8; 32]],
    request_timestamp: u64,
) -> Result<Vec<aeronyx_core::protocol::discovery::SignedNodeDescriptor>, String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_object_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_object_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_object_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::DescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature,
    } = message
    else {
        return Err("directory_object_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || responder != *expected_producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || descriptor_hashes != expected_hashes
        || objects.len() != expected_hashes.len()
    {
        return Err("directory_object_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_descriptor_objects_response_signing_bytes(
        &request_id,
        &responder,
        response_timestamp,
        &descriptor_hashes,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_object_response_invalid_signature".to_string())?;
    for (expected_hash, object) in expected_hashes.iter().zip(&objects) {
        let commitment = aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            object,
        )
        .map_err(|_| "directory_object_response_invalid_descriptor".to_string())?;
        if commitment.descriptor_hash != *expected_hash {
            return Err("directory_object_response_hash_mismatch".to_string());
        }
    }
    Ok(objects)
}

pub(crate) fn verify_replica_descriptor_objects_response(
    frame: &[u8],
    expected_request_id: &[u8; 16],
    expected_producer: &[u8; 32],
    expected_carrier: &[u8; 32],
    expected_hashes: &[[u8; 32]],
    request_timestamp: u64,
) -> Result<Vec<SignedNodeDescriptor>, String> {
    let message = decode_directory_sync_message(frame)
        .map_err(|_| "directory_replica_object_response_decode_failed".to_string())?;
    let canonical = encode_directory_sync_message(&message)
        .map_err(|_| "directory_replica_object_response_encode_failed".to_string())?;
    if canonical != frame {
        return Err("directory_replica_object_response_noncanonical".to_string());
    }
    let DirectorySyncMessage::ReplicaDescriptorObjectsResponseV1 {
        chain_id,
        request_id,
        producer,
        carrier,
        response_timestamp,
        descriptor_hashes,
        objects,
        signature,
    } = message
    else {
        return Err("directory_replica_object_response_unexpected_message".to_string());
    };
    if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
        || request_id != *expected_request_id
        || producer != *expected_producer
        || carrier != *expected_carrier
        || carrier == producer
        || response_timestamp.abs_diff(unix_now_secs())
            > DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS
        || response_timestamp.saturating_add(DIRECTORY_SYNC_RESPONSE_TIMESTAMP_SKEW_SECS)
            < request_timestamp
        || descriptor_hashes != expected_hashes
        || objects.len() != expected_hashes.len()
    {
        return Err("directory_replica_object_response_contract_mismatch".to_string());
    }
    let signing_bytes = directory_replica_descriptor_objects_response_signing_bytes(
        &chain_id,
        &request_id,
        &producer,
        &carrier,
        response_timestamp,
        &descriptor_hashes,
    );
    IdentityPublicKey::from_bytes(&carrier)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "directory_replica_object_response_invalid_signature".to_string())?;
    for (expected_hash, object) in expected_hashes.iter().zip(&objects) {
        let commitment = aeronyx_core::protocol::discovery::DirectoryDescriptorCommitmentV1::from_signed_descriptor(
            object,
        )
        .map_err(|_| "directory_replica_object_response_invalid_descriptor".to_string())?;
        if commitment.descriptor_hash != *expected_hash {
            return Err("directory_replica_object_response_hash_mismatch".to_string());
        }
    }
    Ok(objects)
}
