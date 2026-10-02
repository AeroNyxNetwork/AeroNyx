// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn live_gossip_proof_selection_rotates_and_excludes_expired_descriptors() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap();
    let descriptor_a = descriptor(&subject_a, 1);
    let descriptor_b = descriptor(&subject_b, 1);
    let descriptor_a_second = descriptor(&local, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &descriptor_a);
    let block_a_second = block(&producer_a, 2, block_a.hash(), &descriptor_a_second);
    let block_b = block(&producer_b, 1, [0u8; 32], &descriptor_b);
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    import_replica_block(&store, &producer_a, &descriptor_a, &block_a, [0x77; 16]);
    import_replica_block(
        &store,
        &producer_a,
        &descriptor_a_second,
        &block_a_second,
        [0x78; 16],
    );
    import_replica_block(&store, &producer_b, &descriptor_b, &block_b, [0x79; 16]);

    // [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] Adjacent seeds must
    // rotate producers even when one producer contributes several live
    // descriptors to the bounded candidate window.
    let first = store
        .audited_live_descriptor_gossip_announcement(NOW + 22, 20, 0)
        .unwrap()
        .unwrap();
    let second = store
        .audited_live_descriptor_gossip_announcement(NOW + 22, 20, 1)
        .unwrap()
        .unwrap();
    assert_ne!(first.producer, second.producer);
    for selected in [&first, &second] {
        selected
            .proof
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &selected.producer,
                &selected.block_hash,
                NOW + 22,
            )
            .unwrap();
        assert_eq!(
            selected.proof.commitment.descriptor_hash,
            selected.descriptor_hash
        );
    }

    // [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] The exact same
    // audited blocks are withheld when the configured maturity boundary
    // has not elapsed. Admission remains exact-anchor based.
    let immature_result = store.audited_live_descriptor_gossip_announcement(NOW + 22, 22, 0);
    assert!(matches!(immature_result, Ok(None)));

    // Authentic historical descriptors remain stored but cannot be
    // re-announced as current routeable state after expiry.
    assert!(store
        .audited_live_descriptor_gossip_announcement(NOW + 3_601, 20, 0)
        .unwrap()
        .is_none());
}

#[test]
fn sync_runtime_tracks_bounded_success_and_stable_failure_without_endpoint_data() {
    let producer = [0x44; 32];
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.register_producers(&[producer]);
    runtime.record_attempt(producer, NOW);
    runtime.record_success(producer, NOW + 1, 3, 7, true, 1, 4, 2);
    runtime.record_attempt(producer, NOW + 2);
    runtime.record_failure(
        producer,
        NOW + 3,
        "pinned_directory_peer_unavailable_and_reason_is_bounded",
        Some(NOW + 123),
    );
    runtime.record_backoff_skip(producer);

    let observations = runtime.snapshot();
    assert_eq!(observations.len(), 1);
    let observation = &observations[0];
    assert_eq!(observation.producer, producer);
    assert_eq!(observation.last_success_at, Some(NOW + 1));
    assert_eq!(observation.last_failure_at, Some(NOW + 3));
    assert_eq!(observation.remote_tip_height, Some(7));
    assert_eq!(observation.local_tip_height, 3);
    assert!(observation.has_more);
    assert_eq!(observation.consecutive_failures, 1);
    assert_eq!(observation.total_attempts, 2);
    assert_eq!(observation.successful_pages, 1);
    assert_eq!(observation.failed_attempts, 1);
    assert_eq!(observation.retry_not_before, Some(NOW + 123));
    assert_eq!(observation.backoff_skips, 1);
    assert_eq!(observation.blocks_inserted, 1);
    assert_eq!(observation.commitments_inserted, 4);
    assert_eq!(observation.requests_sent, 2);
    assert_eq!(
        observation.last_failure_reason.as_deref(),
        Some("pinned_directory_peer_unavailable_and_reason_is_bounded")
    );
    assert_eq!(
        runtime.deferred_retry_until(&producer, NOW + 20),
        Some(NOW + 123)
    );
    assert_eq!(runtime.deferred_retry_until(&producer, NOW + 123), None);
    assert_eq!(runtime.consecutive_failures(&producer), 1);
}

#[test]
fn sync_runtime_success_clears_backoff_without_erasing_history() {
    let producer = [0x45; 32];
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.record_failure(
        producer,
        NOW,
        "directory_range_transport_failed",
        Some(NOW + 60),
    );
    runtime.record_backoff_skip(producer);
    runtime.record_success(producer, NOW + 61, 4, 4, false, 1, 2, 1);

    let observation = &runtime.snapshot()[0];
    assert_eq!(observation.consecutive_failures, 0);
    assert_eq!(observation.retry_not_before, None);
    assert_eq!(observation.failed_attempts, 1);
    assert_eq!(observation.backoff_skips, 1);
    assert_eq!(observation.successful_pages, 1);
}

#[test]
fn directory_sync_transport_outcomes_are_mutually_exclusive_and_bounded() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime
        .record_directory_sync_transport_outcome(DirectoryReplicaTransportOutcome::Succeeded, NOW);
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::ConnectTimeout,
        NOW + 1,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::RequestTimeout,
        NOW + 2,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::ConnectFailure,
        NOW + 3,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::RequestFailure,
        NOW + 4,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::HttpStatusFailure,
        NOW + 5,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::ResponseTooLarge,
        NOW + 6,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::ResponseBodyReadFailure,
        NOW + 7,
    );
    runtime.record_directory_sync_transport_outcome(DirectoryReplicaTransportOutcome::Succeeded, 0);

    let snapshot = runtime.directory_sync_transport_snapshot();
    assert_eq!(snapshot.requests, 8);
    assert_eq!(snapshot.terminal_outcomes(), snapshot.requests);
    assert_eq!(snapshot.succeeded, 1);
    assert_eq!(snapshot.connect_timeouts, 1);
    assert_eq!(snapshot.request_timeouts, 1);
    assert_eq!(snapshot.connect_failures, 1);
    assert_eq!(snapshot.request_failures, 1);
    assert_eq!(snapshot.http_status_failures, 1);
    assert_eq!(snapshot.response_too_large, 1);
    assert_eq!(snapshot.response_body_read_failures, 1);
    assert_eq!(snapshot.recent_window_capacity, 32);
    assert_eq!(snapshot.recent_requests, 8);
    assert_eq!(snapshot.recent_succeeded, 1);
    assert_eq!(snapshot.recent_failures, 7);
    assert_eq!(snapshot.consecutive_failures, 7);
    assert_eq!(snapshot.degraded_transitions, 1);
    assert_eq!(snapshot.recovery_transitions, 0);
    assert_eq!(snapshot.degraded_since_at, Some(NOW + 1));
    assert_eq!(snapshot.last_degraded_at, Some(NOW + 1));
    assert_eq!(snapshot.last_recovered_at, None);
    assert!(snapshot.terminal_outcomes_consistent());
    assert!(snapshot.recent_outcomes_consistent());
    assert!(snapshot.lifecycle_consistent());
    assert_eq!(snapshot.health(), DirectoryReplicaTransportHealth::Degraded);
    assert_eq!(
        snapshot.last_outcome,
        Some(DirectoryReplicaTransportOutcome::ResponseBodyReadFailure)
    );
    assert_eq!(snapshot.last_request_at, Some(NOW + 7));
    assert_eq!(snapshot.last_success_at, Some(NOW));
    assert_eq!(snapshot.last_failure_at, Some(NOW + 7));
}

#[test]
fn directory_sync_transport_recent_window_evicts_old_failures() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    for _ in 0..DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY {
        runtime.record_directory_sync_transport_outcome(
            DirectoryReplicaTransportOutcome::RequestFailure,
            NOW,
        );
    }
    let failed = runtime.directory_sync_transport_snapshot();
    assert_eq!(failed.requests, 32);
    assert_eq!(failed.recent_requests, 32);
    assert_eq!(failed.recent_succeeded, 0);
    assert_eq!(failed.recent_failures, 32);
    assert_eq!(failed.consecutive_failures, 32);
    assert_eq!(failed.degraded_transitions, 1);
    assert_eq!(failed.recovery_transitions, 0);
    assert_eq!(failed.degraded_since_at, Some(NOW));
    assert_eq!(failed.health(), DirectoryReplicaTransportHealth::Degraded);

    for _ in 0..DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY {
        runtime.record_directory_sync_transport_outcome(
            DirectoryReplicaTransportOutcome::Succeeded,
            NOW + 1,
        );
    }
    let recovered = runtime.directory_sync_transport_snapshot();
    assert_eq!(recovered.requests, 64);
    assert_eq!(recovered.terminal_outcomes(), recovered.requests);
    assert_eq!(recovered.succeeded, 32);
    assert_eq!(recovered.request_failures, 32);
    assert_eq!(recovered.recent_window_capacity, 32);
    assert_eq!(recovered.recent_requests, 32);
    assert_eq!(recovered.recent_succeeded, 32);
    assert_eq!(recovered.recent_failures, 0);
    assert_eq!(recovered.consecutive_failures, 0);
    assert_eq!(recovered.degraded_transitions, 1);
    assert_eq!(recovered.recovery_transitions, 1);
    assert_eq!(recovered.degraded_since_at, None);
    assert_eq!(recovered.last_degraded_at, Some(NOW));
    assert_eq!(recovered.last_recovered_at, Some(NOW + 1));
    assert!(recovered.lifecycle_consistent());
    assert_eq!(recovered.health(), DirectoryReplicaTransportHealth::Healthy);
}

#[test]
fn directory_sync_transport_lifecycle_survives_wall_clock_rollback() {
    // [DIRECTORY-TRANSPORT-LIFECYCLE 2026-07-29 by Codex] NTP or operator
    // clock rollback must not regress public ages or duplicate lifecycle
    // transitions while the bounded health window changes state.
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::Succeeded,
        NOW + 10,
    );
    runtime.record_directory_sync_transport_outcome(
        DirectoryReplicaTransportOutcome::RequestFailure,
        NOW + 5,
    );

    let degraded = runtime.directory_sync_transport_snapshot();
    assert_eq!(degraded.last_request_at, Some(NOW + 10));
    assert_eq!(degraded.last_success_at, Some(NOW + 10));
    assert_eq!(degraded.last_failure_at, Some(NOW + 5));
    assert_eq!(degraded.degraded_transitions, 1);
    assert_eq!(degraded.degraded_since_at, Some(NOW + 10));
    assert_eq!(degraded.health(), DirectoryReplicaTransportHealth::Degraded);

    for offset in 11..=15 {
        runtime.record_directory_sync_transport_outcome(
            DirectoryReplicaTransportOutcome::Succeeded,
            NOW + offset,
        );
    }
    let recovered = runtime.directory_sync_transport_snapshot();
    assert_eq!(recovered.recovery_transitions, 1);
    assert_eq!(recovered.last_recovered_at, Some(NOW + 14));
    assert_eq!(recovered.health(), DirectoryReplicaTransportHealth::Healthy);

    for _ in 0..3 {
        runtime.record_directory_sync_transport_outcome(
            DirectoryReplicaTransportOutcome::ConnectTimeout,
            NOW + 2,
        );
    }
    let degraded_again = runtime.directory_sync_transport_snapshot();
    assert_eq!(degraded_again.last_request_at, Some(NOW + 15));
    assert_eq!(degraded_again.last_failure_at, Some(NOW + 5));
    assert_eq!(degraded_again.degraded_transitions, 2);
    assert_eq!(degraded_again.recovery_transitions, 1);
    assert_eq!(degraded_again.degraded_since_at, Some(NOW + 15));
    assert_eq!(degraded_again.last_degraded_at, Some(NOW + 15));
    assert!(degraded_again.lifecycle_consistent());
    assert_eq!(
        degraded_again.health(),
        DirectoryReplicaTransportHealth::Degraded
    );
}

#[test]
fn sync_runtime_restores_only_bounded_scheduler_state() {
    let producer = [0x46; 32];
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.register_producers(&[producer]);
    runtime.restore_retry_states(&[DirectoryReplicaRetryState {
        producer,
        consecutive_failures: DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES,
        retry_not_before: Some(NOW + 600),
        last_failure_at: NOW,
        last_failure_reason: "directory_range_transport_failed".to_string(),
        backoff_skips: 7,
    }]);

    let observation = &runtime.snapshot()[0];
    assert_eq!(
        observation.consecutive_failures,
        DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
    );
    assert_eq!(observation.retry_not_before, Some(NOW + 600));
    assert_eq!(observation.last_attempt_at, Some(NOW));
    assert_eq!(observation.last_failure_at, Some(NOW));
    assert_eq!(observation.backoff_skips, 7);
    assert_eq!(observation.total_attempts, 0);
    assert_eq!(observation.failed_attempts, 0);
    assert_eq!(observation.successful_pages, 0);

    runtime.record_failure(
        producer,
        NOW + 601,
        "directory_range_transport_failed",
        Some(NOW + 1_200),
    );
    assert_eq!(
        runtime.snapshot()[0].consecutive_failures,
        DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
    );
}
