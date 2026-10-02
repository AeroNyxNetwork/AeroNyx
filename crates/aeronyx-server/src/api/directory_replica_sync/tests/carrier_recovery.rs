// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn carrier_fallback_is_limited_to_availability_and_admission_failures() {
    for reason in [
        "pinned_directory_peer_unavailable",
        "pinned_directory_peer_missing_endpoint",
        "directory_range_transport_failed",
        "directory_range_http_status_403",
        "directory_range_http_status_404",
        "directory_range_http_status_408",
        "directory_range_http_status_429",
        "directory_range_http_status_500",
        "directory_replica_range_http_status_503",
        "directory_replica_objects_transport_failed",
        "directory_replica_range_peer_replica_range_not_retained",
        "directory_replica_objects_peer_replica_object_not_found",
        "directory_replica_objects_http_status_503",
    ] {
        assert!(directory_sync_failure_allows_carrier_fallback(reason));
    }
    for reason in [
        "directory_range_response_noncanonical",
        "directory_range_response_invalid_signature",
        "directory_range_response_contract_mismatch",
        "directory_object_response_hash_mismatch",
        "directory_range_http_status_400",
        "directory_range_http_status_401",
        "directory_range_http_status_409",
    ] {
        assert!(!directory_sync_failure_allows_carrier_fallback(reason));
    }
}

#[test]
fn pinned_and_explicit_carrier_recovery_stays_inside_request_budget() {
    let worst_case_requests = 1usize
        .saturating_add(DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE)
        .saturating_add(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE)
        .saturating_add(
            usize::try_from(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE)
                .expect("request bound fits usize"),
        );
    assert!(
        worst_case_requests
            <= usize::try_from(DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND)
                .expect("round request budget fits usize")
    );
    assert_eq!(DIRECTORY_PINNED_RECOVERY_MAX_CARRIERS_PER_PAGE, 2);
    assert_eq!(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE, 2);
}

#[test]
fn carrier_cold_bootstrap_multi_page_budget_is_bounded() {
    // [CARRIER-MULTIPAGE-RECOVERY 2026-07-26 by Codex] Every attempt
    // reserves a full worst-case page while accounting the exact requests
    // already consumed. Sparse pages can prove multi-page continuation;
    // dense pages stop before crossing the normal producer-round ceiling.
    assert_eq!(DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES, 3);
    assert_eq!(
        DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET,
        DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND
    );
    assert!(should_continue_directory_carrier_cold_bootstrap(1, 1, true));
    assert!(should_continue_directory_carrier_cold_bootstrap(2, 2, true));
    assert!(!should_continue_directory_carrier_cold_bootstrap(
        DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_MAX_PAGES,
        0,
        true,
    ));
    assert!(!should_continue_directory_carrier_cold_bootstrap(
        1, 0, false,
    ));
    assert!(!should_continue_directory_carrier_cold_bootstrap(
        1,
        DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET,
        true,
    ));
    assert!(!directory_carrier_cold_bootstrap_prefix_ready(1));
    assert!(directory_carrier_cold_bootstrap_prefix_ready(2));
    assert!(directory_carrier_cold_bootstrap_prefix_ready(3));
}

#[test]
fn carrier_cold_bootstrap_retries_only_availability_failures() {
    for reason in [
        "directory_replica_range_transport_failed",
        "directory_replica_objects_transport_failed",
        "directory_replica_range_http_status_503",
        "directory_replica_range_peer_replica_range_not_retained",
        "directory_mirror_recovery_carrier_descriptor_changed",
    ] {
        assert_eq!(
            directory_carrier_recovery_disposition(reason),
            DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure
        );
    }
    for reason in [
        "directory_replica_range_response_noncanonical",
        "directory_replica_range_response_invalid_signature",
        "directory_replica_object_response_hash_mismatch",
        "directory_replica_import_rejected",
        "producer_quarantined",
    ] {
        assert_eq!(
            directory_carrier_recovery_disposition(reason),
            DirectoryCarrierRecoveryDisposition::StopClosed
        );
    }
}

#[tokio::test]
async fn carrier_hydration_availability_failure_preserves_request_count() -> TestResult {
    let (requester, producer, carrier, block) = carrier_hydration_test_context();
    let (object_url, server) =
        carrier_hydration_test_endpoint(StatusCode::SERVICE_UNAVAILABLE, Vec::new()).await?;
    let client = reqwest::Client::builder().no_proxy().build()?;

    let Err(failure) = hydrate_directory_replica_descriptor_objects_tracked(
        &requester,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        &client,
        object_url,
        &requester.public_key_bytes(),
        &[block],
    )
    .await
    else {
        return Err(std::io::Error::other("expected carrier availability failure").into());
    };
    server.abort();

    // One already-successful range plus one dispatched object request.
    assert_eq!(failure.requests_made, 2);
    assert_eq!(
        directory_carrier_recovery_disposition(&failure.reason),
        DirectoryCarrierRecoveryDisposition::RetryAvailabilityFailure
    );
    Ok(())
}

#[tokio::test]
async fn carrier_hydration_corruption_stops_closed_without_losing_request_count() -> TestResult {
    let (requester, producer, carrier, block) = carrier_hydration_test_context();
    let (object_url, server) =
        carrier_hydration_test_endpoint(StatusCode::OK, b"corrupt-frame".to_vec()).await?;
    let client = reqwest::Client::builder().no_proxy().build()?;

    let Err(failure) = hydrate_directory_replica_descriptor_objects_tracked(
        &requester,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        &client,
        object_url,
        &requester.public_key_bytes(),
        &[block],
    )
    .await
    else {
        return Err(std::io::Error::other("expected carrier corruption failure").into());
    };
    server.abort();

    assert_eq!(failure.requests_made, 2);
    assert_eq!(
        directory_carrier_recovery_disposition(&failure.reason),
        DirectoryCarrierRecoveryDisposition::StopClosed
    );
    Ok(())
}

#[tokio::test]
async fn cold_bootstrap_smoke_fails_closed_without_transport() {
    let local = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0xc2; 32])
        .unwrap()
        .public_key_bytes();
    let report =
        run_directory_carrier_cold_bootstrap_smoke(&[producer], &PeerStore::new(), &local, None)
            .await;

    assert!(!report.success);
    assert_eq!(
        report.failure_reason,
        Some("smoke_http_client_initialization_failed")
    );
    assert_eq!(report.live_store_effect, "none_isolated_memory_store_only");
    assert_eq!(report.pages_imported, 0);
    assert_eq!(
        report.request_budget,
        u64::from(DIRECTORY_CARRIER_COLD_BOOTSTRAP_SMOKE_REQUEST_BUDGET)
    );
    assert!(!report.multi_page_prefix_verified);
}

#[test]
fn mirror_recovery_carrier_selection_excludes_participants_and_is_deterministic() {
    let now = unix_now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
    let requester = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
    let store = PeerStore::new();
    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let mut expected_excluded = HashSet::new();
    expected_excluded.insert(producer.public_key_bytes());
    expected_excluded.insert(requester.public_key_bytes());

    for seed in [0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7] {
        let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now.saturating_sub(1),
            now + 600,
            "mirror-recovery-test",
        );
        descriptor.policy.public_discovery = true;
        descriptor.public_endpoint = Some(format!("http://8.8.8.{seed}:8422"));
        store
            .upsert_verified_from_source(
                SignedNodeDescriptor::sign(descriptor, &identity).unwrap(),
                now,
                "directory_mirror_recovery_test",
            )
            .unwrap();
    }

    let selection = directory_mirror_recovery_carriers(
        &store,
        &capability_cache,
        &producer.public_key_bytes(),
        &requester.public_key_bytes(),
        now,
    );
    assert_eq!(
        selection.carriers.len(),
        DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE
    );
    assert_eq!(selection.explicitly_advertised_candidate_count, 0);
    assert_eq!(selection.unadvertised_compatibility_candidate_count, 5);
    assert_eq!(selection.selected_explicitly_advertised_count, 0);
    assert_eq!(selection.selected_unadvertised_compatibility_count, 2);
    assert!(selection
        .carriers
        .iter()
        .all(|candidate| !expected_excluded.contains(&candidate.node_id)));
    assert_eq!(
        selection,
        directory_mirror_recovery_carriers(
            &store,
            &capability_cache,
            &producer.public_key_bytes(),
            &requester.public_key_bytes(),
            now,
        )
    );
}

#[test]
fn mirror_recovery_carrier_selection_prefers_live_diverse_fresh_peers() {
    let now = unix_now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap();
    let requester = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
    let store = PeerStore::new();
    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let mut fresh_routeable = HashSet::new();

    // [MIRROR-DIVERSITY 2026-07-24 by Codex] Three fresh routeable peers
    // include two identical signed region hints. The second selected peer
    // must use the different hint without sacrificing routeability or
    // descriptor freshness. These hints are not operator/ASN proof.
    for (seed, issued_at, region, routeable, advertised) in [
        (0xd3, now - 1, Some("region-a"), true, true),
        (0xd4, now - 2, Some("REGION-A"), true, false),
        (0xd5, now - 3, Some("region-b"), true, true),
        (
            0xd6,
            now - DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS - 1,
            Some("region-c"),
            true,
            true,
        ),
        (0xd7, now - 4, Some("region-d"), false, true),
    ] {
        let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            issued_at,
            now + 600,
            "mirror-diversity-test",
        );
        descriptor.policy.public_discovery = true;
        descriptor.policy.region = region.map(str::to_string);
        descriptor.public_endpoint = Some(format!("http://8.8.8.{seed}:8422"));
        if advertised {
            descriptor
                .capabilities
                .push(NodeCapability::DirectoryMirrorCarrier);
        }
        store
            .upsert_verified_from_source(
                SignedNodeDescriptor::sign(descriptor, &identity).unwrap(),
                now,
                "directory_mirror_diversity_test",
            )
            .unwrap();
        if routeable {
            store.record_route_forward_success(&identity.public_key_bytes(), now);
        }
        if routeable
            && issued_at >= now.saturating_sub(DIRECTORY_MIRROR_RECOVERY_FRESH_DESCRIPTOR_SECS)
        {
            fresh_routeable.insert(identity.public_key_bytes());
        }
    }

    let quarantined = IdentityKeyPair::from_bytes(&[0xd8; 32]).unwrap();
    let mut quarantined_descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
        quarantined.public_key_bytes(),
        1,
        now - 1,
        now + 600,
        "mirror-diversity-test",
    );
    quarantined_descriptor.policy.public_discovery = true;
    quarantined_descriptor.policy.region = Some("region-e".to_string());
    quarantined_descriptor.public_endpoint = Some("http://8.8.8.216:8422".to_string());
    store
        .upsert_verified_from_source(
            SignedNodeDescriptor::sign(quarantined_descriptor, &quarantined).unwrap(),
            now,
            "directory_mirror_diversity_test",
        )
        .unwrap();
    for _ in 0..3 {
        store.record_route_forward_failure(&quarantined.public_key_bytes(), now, "request_failed");
    }

    let selection = directory_mirror_recovery_carriers(
        &store,
        &capability_cache,
        &producer.public_key_bytes(),
        &requester.public_key_bytes(),
        now,
    );
    assert_eq!(selection.candidate_count, 5);
    assert_eq!(selection.routeable_candidate_count, 4);
    assert_eq!(selection.explicitly_advertised_candidate_count, 4);
    assert_eq!(selection.unadvertised_compatibility_candidate_count, 1);
    assert_eq!(selection.carriers.len(), 2);
    assert_eq!(selection.selected_routeable_count, 2);
    assert_eq!(selection.selected_explicitly_advertised_count, 2);
    assert_eq!(selection.selected_unadvertised_compatibility_count, 0);
    assert_eq!(selection.selected_region_hint_count, 2);
    assert_eq!(selection.distinct_selected_region_hint_count, 2);
    assert!(selection
        .carriers
        .iter()
        .all(|candidate| fresh_routeable.contains(&candidate.node_id)));
    assert!(!selection
        .carriers
        .iter()
        .any(|candidate| candidate.node_id == quarantined.public_key_bytes()));

    // [MIRROR-CARRIER-SMOKE 2026-07-25 by Codex] Manual verification must
    // never silently fall back to an unadvertised compatibility carrier.
    let smoke_selection = directory_mirror_recovery_carriers_with_requirement(
        &store,
        &capability_cache,
        &producer.public_key_bytes(),
        &requester.public_key_bytes(),
        now,
        true,
    );
    assert_eq!(smoke_selection.candidate_count, 4);
    assert_eq!(smoke_selection.explicitly_advertised_candidate_count, 4);
    assert_eq!(
        smoke_selection.unadvertised_compatibility_candidate_count,
        0
    );
    assert_eq!(smoke_selection.selected_explicitly_advertised_count, 2);
    assert_eq!(smoke_selection.selected_unadvertised_compatibility_count, 0);
}

#[test]
fn carrier_smoke_failures_collapse_to_privacy_safe_buckets() {
    assert_eq!(
        directory_mirror_carrier_smoke_failure_bucket("directory_replica_range_http_status_503"),
        "carrier_unavailable"
    );
    assert_eq!(
        directory_mirror_carrier_smoke_failure_bucket(
            "directory_replica_object_response_invalid_signature"
        ),
        "carrier_evidence_rejected"
    );
    assert_eq!(
        directory_mirror_carrier_smoke_failure_bucket(
            "directory_replica_range_request_encode_failed"
        ),
        "carrier_request_failed"
    );
}

#[test]
fn carrier_range_response_verification_binds_producer_carrier_and_signature() {
    let producer = IdentityKeyPair::from_bytes(&[0xf1; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0xf2; 32]).unwrap();
    let other = IdentityKeyPair::from_bytes(&[0xf3; 32]).unwrap();
    let request_id = [0xf4; 16];
    let now = unix_now_secs();
    let blocks = Vec::new();
    let signing_bytes = directory_replica_block_range_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &producer.public_key_bytes(),
        &carrier.public_key_bytes(),
        now,
        &blocks,
        false,
        0,
        &[0u8; 32],
    );
    let response = DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        producer: producer.public_key_bytes(),
        carrier: carrier.public_key_bytes(),
        response_timestamp: now,
        blocks,
        has_more: false,
        tip_height: 0,
        tip_hash: [0u8; 32],
        signature: carrier.sign(&signing_bytes),
    };
    let frame = encode_directory_sync_message(&response).unwrap();
    assert_eq!(
        verify_replica_block_range_response(
            &frame,
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            1,
            now,
        )
        .unwrap(),
        (Vec::new(), false, 0, [0u8; 32])
    );
    assert_eq!(
        verify_replica_block_range_response(
            &frame,
            &request_id,
            &other.public_key_bytes(),
            &carrier.public_key_bytes(),
            1,
            now,
        )
        .unwrap_err(),
        "directory_replica_range_response_contract_mismatch"
    );
    assert_eq!(
        verify_replica_block_range_response(
            &frame,
            &request_id,
            &producer.public_key_bytes(),
            &other.public_key_bytes(),
            1,
            now,
        )
        .unwrap_err(),
        "directory_replica_range_response_contract_mismatch"
    );

    let mut tampered = response;
    let DirectorySyncMessage::ReplicaBlockRangeResponseV1 { signature, .. } = &mut tampered else {
        unreachable!();
    };
    signature[0] ^= 1;
    assert_eq!(
        verify_replica_block_range_response(
            &encode_directory_sync_message(&tampered).unwrap(),
            &request_id,
            &producer.public_key_bytes(),
            &carrier.public_key_bytes(),
            1,
            now,
        )
        .unwrap_err(),
        "directory_replica_range_response_invalid_signature"
    );
}

#[test]
fn mirror_carrier_capability_cache_is_bounded_and_failure_specific() {
    let cache = DirectoryMirrorCarrierCapabilityCache::default();
    for index in 0..=DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES {
        let mut carrier = [0u8; 32];
        carrier[..8].copy_from_slice(
            &u64::try_from(index)
                .expect("test cache index fits u64")
                .to_be_bytes(),
        );
        cache.record_unsupported(carrier, 1);
    }
    let mut oldest = [0u8; 32];
    oldest[..8].copy_from_slice(&0u64.to_be_bytes());
    let mut newest = [0u8; 32];
    newest[..8].copy_from_slice(
        &u64::try_from(DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES)
            .expect("test cache capacity fits u64")
            .to_be_bytes(),
    );
    assert_eq!(
        cache.len(),
        DIRECTORY_MIRROR_CARRIER_CAPABILITY_CACHE_MAX_ENTRIES
    );
    assert!(cache.should_attempt(&oldest, 1));
    assert!(!cache.should_attempt(&newest, 1));
    assert!(cache.should_attempt(&newest, 2));

    for reason in [
        "directory_replica_range_http_status_404",
        "directory_replica_range_http_status_405",
        "directory_replica_objects_http_status_501",
    ] {
        assert!(directory_mirror_carrier_capability_unavailable(reason));
    }
    for reason in [
        "directory_range_http_status_404",
        "directory_replica_range_transport_failed",
        "directory_replica_range_http_status_403",
        "directory_replica_range_http_status_408",
        "directory_replica_range_http_status_429",
        "directory_replica_range_http_status_500",
        "directory_replica_range_response_invalid_signature",
        "directory_replica_range_peer_replica_not_found",
        "directory_replica_range_peer_replica_range_not_retained",
        "directory_replica_objects_peer_replica_object_not_found",
    ] {
        assert!(!directory_mirror_carrier_capability_unavailable(reason));
    }
}

#[test]
fn mirror_carrier_endpoint_is_bound_to_selected_descriptor_sequence() {
    let now = unix_now_secs();
    let store = PeerStore::new();
    let carrier = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
    let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
        carrier.public_key_bytes(),
        7,
        now.saturating_sub(1),
        now + 600,
        "mirror-capability-sequence-test",
    );
    descriptor.policy.public_discovery = true;
    descriptor.public_endpoint = Some("http://8.8.8.146:8422".to_string());
    store
        .upsert_verified_from_source(
            SignedNodeDescriptor::sign(descriptor, &carrier).unwrap(),
            now,
            "directory_mirror_capability_sequence_test",
        )
        .unwrap();

    assert!(
        directory_mirror_recovery_carrier_urls(&store, &carrier.public_key_bytes(), 7, now,)
            .is_ok()
    );
    assert_eq!(
        directory_mirror_recovery_carrier_urls(&store, &carrier.public_key_bytes(), 8, now,)
            .unwrap_err(),
        "directory_mirror_recovery_carrier_descriptor_changed"
    );
}

#[test]
fn replica_proof_carrier_endpoint_requires_current_explicit_capability() {
    let now = unix_now_secs();
    let store = PeerStore::new();
    let carrier = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
    let mut descriptor = NodeDescriptor::new(
        carrier.public_key_bytes(),
        7,
        now.saturating_sub(1),
        now + 600,
        "replica-proof-capability-test",
    );
    descriptor.policy.public_discovery = true;
    descriptor.public_endpoint = Some("http://8.8.8.147:8422".to_string());
    descriptor
        .capabilities
        .push(NodeCapability::DirectoryMirrorCarrier);
    store
        .upsert_verified_from_source(
            SignedNodeDescriptor::sign(descriptor, &carrier).unwrap(),
            now,
            "directory_replica_proof_capability_test",
        )
        .unwrap();
    let selected = DirectoryMirrorRecoveryCarrier {
        node_id: carrier.public_key_bytes(),
        descriptor_sequence: 7,
    };
    assert!(directory_replica_descriptor_inclusion_proof_url(&store, &selected, now).is_ok());

    let stale = DirectoryMirrorRecoveryCarrier {
        descriptor_sequence: 8,
        ..selected
    };
    assert_eq!(
        directory_replica_descriptor_inclusion_proof_url(&store, &stale, now).unwrap_err(),
        "directory_replica_proof_carrier_descriptor_changed"
    );
}
