// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn catch_up_budget_allows_small_pages_but_reserves_worst_case_headroom() {
    assert_eq!(DIRECTORY_SYNC_MAX_REQUESTS_PER_PAGE, 18);
    assert_eq!(directory_sync_request_count_for_objects(0), 1);
    assert_eq!(directory_sync_request_count_for_objects(1), 2);
    assert_eq!(directory_sync_request_count_for_objects(16), 2);
    assert_eq!(directory_sync_request_count_for_objects(17), 3);
    assert_eq!(directory_sync_request_count_for_objects(256), 17);
    assert!(should_continue_directory_replica_catch_up(1, 2, true));
    assert!(should_continue_directory_replica_catch_up(4, 8, true));
    assert!(should_continue_directory_replica_catch_up(6, 12, true));
    assert!(!should_continue_directory_replica_catch_up(7, 14, true));
    assert!(!should_continue_directory_replica_catch_up(8, 16, true));
    assert!(should_continue_directory_replica_catch_up(1, 12, true));
    assert!(!should_continue_directory_replica_catch_up(1, 13, true));
    assert!(!should_continue_directory_replica_catch_up(1, 2, false));
}

#[test]
fn mirror_recovery_is_bounded_and_rejects_security_failures() {
    assert_eq!(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE, 2);
    let recovery_carrier_count = u64::try_from(DIRECTORY_MIRROR_RECOVERY_MAX_CARRIERS_PER_PAGE)
        .expect("bounded recovery carrier count fits u64");
    assert!(
        DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS * (1 + recovery_carrier_count)
            < DIRECTORY_SYNC_PRODUCER_ROUND_TIMEOUT_SECS
    );
    for reason in [
        "directory_mirror_peer_unavailable",
        "directory_range_transport_failed",
        "directory_objects_transport_failed",
        "directory_range_http_status_404",
        "directory_replica_range_http_status_405",
        "directory_replica_range_http_status_429",
        "directory_replica_objects_http_status_503",
        "directory_replica_range_peer_replica_range_not_retained",
        "directory_replica_objects_peer_replica_object_not_found",
        "directory_mirror_recovery_carrier_unavailable",
        "directory_mirror_recovery_carrier_descriptor_changed",
    ] {
        assert!(directory_mirror_failure_allows_recovery(reason));
    }
    for reason in [
        "directory_mirror_descriptor_changed",
        "directory_range_response_noncanonical",
        "directory_range_response_invalid_signature",
        "directory_replica_range_response_contract_mismatch",
        "directory_replica_range_response_invalid_signature",
        "directory_replica_object_response_hash_mismatch",
        "directory_mirror_import_rejected",
        "directory_range_http_status_400",
        "directory_range_http_status_401",
        "directory_range_http_status_409",
    ] {
        assert!(!directory_mirror_failure_allows_recovery(reason));
    }
}

#[test]
fn mirror_catch_up_stops_at_page_request_or_convergence_boundaries() {
    // [MIRROR-CATCHUP 2026-07-24 by Codex] Permissionless work must remain
    // strictly below the pinned producer budget.
    assert!(DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND < DIRECTORY_SYNC_MAX_PAGES_PER_ROUND);
    assert!(
        DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND
            < DIRECTORY_SYNC_REQUEST_BUDGET_PER_ROUND
    );
    assert!(
        DIRECTORY_MIRROR_MAX_REQUESTS_PER_PAGE
            <= DIRECTORY_MIRROR_REQUEST_BUDGET_PER_PRODUCER_ROUND
    );
    assert!(should_continue_directory_mirror_catch_up(1, 1, true));
    assert!(should_continue_directory_mirror_catch_up(1, 5, true));
    assert!(!should_continue_directory_mirror_catch_up(1, 1, false));
    assert!(!should_continue_directory_mirror_catch_up(
        DIRECTORY_MIRROR_MAX_PAGES_PER_PRODUCER_ROUND,
        1,
        true
    ));
    assert!(!should_continue_directory_mirror_catch_up(1, 6, true));
}

#[test]
fn mirror_recovery_selection_skips_only_the_cached_descriptor_sequence() {
    let now = unix_now_secs();
    let producer = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
    let requester = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
    let store = PeerStore::new();
    let capability_cache = DirectoryMirrorCarrierCapabilityCache::default();
    let mut carriers = Vec::new();

    for seed in [0xc3, 0xc4, 0xc5] {
        let identity = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let mut descriptor = aeronyx_core::protocol::discovery::NodeDescriptor::new(
            identity.public_key_bytes(),
            7,
            now.saturating_sub(1),
            now + 600,
            "mirror-capability-test",
        );
        descriptor.policy.public_discovery = true;
        descriptor.public_endpoint = Some(format!("http://8.8.8.{seed}:8422"));
        store
            .upsert_verified_from_source(
                SignedNodeDescriptor::sign(descriptor, &identity).unwrap(),
                now,
                "directory_mirror_capability_test",
            )
            .unwrap();
        store.record_route_forward_success(&identity.public_key_bytes(), now);
        carriers.push(identity.public_key_bytes());
    }

    capability_cache.record_unsupported(carriers[0], 7);
    let selection = directory_mirror_recovery_carriers(
        &store,
        &capability_cache,
        &producer.public_key_bytes(),
        &requester.public_key_bytes(),
        now,
    );
    assert_eq!(selection.candidate_count, 3);
    assert_eq!(selection.explicitly_advertised_candidate_count, 0);
    assert_eq!(selection.unadvertised_compatibility_candidate_count, 3);
    assert_eq!(selection.capability_cached_unavailable_count, 1);
    assert_eq!(selection.carriers.len(), 2);
    assert_eq!(selection.selected_unadvertised_compatibility_count, 2);
    assert!(!selection
        .carriers
        .iter()
        .any(|candidate| candidate.node_id == carriers[0]));
    assert!(capability_cache.should_attempt(&carriers[0], 8));
}

#[test]
fn expired_retained_mirror_remains_a_resume_candidate_without_live_descriptor() {
    // [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] An empty live
    // candidate set models a restarted node after the producer descriptor
    // expired. The durable cursor remains schedulable so the direct lookup
    // can fail normally and current admitted carriers can resume tip + 1.
    let producer = [0x51; 32];
    let newcomer = [0x52; 32];
    let retained = [DirectoryRetainedMirrorCursor {
        producer,
        descriptor_sequence: 7,
    }];
    assert_eq!(
        directory_full_node_mirror_candidates(&retained, Vec::new(), 1),
        vec![(producer, 7)]
    );
    assert_eq!(
        directory_full_node_mirror_candidates(&retained, vec![(newcomer, 1)], 1),
        vec![(producer, 7)]
    );
    assert_eq!(
        directory_full_node_mirror_candidates(&retained, vec![(producer, 9)], 1),
        vec![(producer, 9)]
    );
}
