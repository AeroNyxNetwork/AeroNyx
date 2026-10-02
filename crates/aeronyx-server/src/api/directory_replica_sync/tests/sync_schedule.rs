// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn repeated_failures_use_bounded_exponential_backoff() {
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 0), 0);
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 1), 0);
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 2), 120);
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 3), 360);
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 4), 840);
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 5), 1_800);
    assert_eq!(directory_sync_failure_backoff_delay_secs(120, 99), 1_800);
}

#[test]
fn coordinator_restores_retry_state_for_configured_producers_only() {
    let temp = TempDir::new().unwrap();
    let local = Arc::new(IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap());
    let configured = IdentityKeyPair::from_bytes(&[0xd2; 32])
        .unwrap()
        .public_key_bytes();
    let retired = IdentityKeyPair::from_bytes(&[0xd3; 32])
        .unwrap()
        .public_key_bytes();
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        TEST_NOW,
    )
    .unwrap();
    for producer in [configured, retired] {
        store
            .persist_retry_failure(
                producer,
                2,
                Some(TEST_NOW + 300),
                TEST_NOW,
                "directory_range_transport_failed",
            )
            .unwrap();
    }
    let store = Arc::new(store);
    let runtime = Arc::new(DirectoryReplicaSyncRuntime::default());
    assert_eq!(
        DirectoryReplicaSyncCoordinator::new_with_policy(
            vec![configured],
            120,
            Arc::clone(&store),
            Arc::clone(&runtime),
            Arc::new(PeerStore::new()),
            Arc::clone(&local),
            DirectoryReplicaSyncPolicy {
                witness_min_verified: 0,
                full_node_mirror_enabled: false,
                full_node_mirror_max_producers: 32,
            },
        )
        .err(),
        Some("directory_observation_witness_threshold_invalid")
    );
    assert_eq!(
        DirectoryReplicaSyncCoordinator::new_with_policy(
            vec![configured],
            120,
            Arc::clone(&store),
            Arc::clone(&runtime),
            Arc::new(PeerStore::new()),
            Arc::clone(&local),
            DirectoryReplicaSyncPolicy {
                witness_min_verified: 2,
                full_node_mirror_enabled: false,
                full_node_mirror_max_producers: 32,
            },
        )
        .err(),
        Some("directory_observation_witness_threshold_invalid")
    );
    let coordinator = DirectoryReplicaSyncCoordinator::new_with_policy(
        vec![configured],
        120,
        store,
        Arc::clone(&runtime),
        Arc::new(PeerStore::new()),
        local,
        DirectoryReplicaSyncPolicy {
            witness_min_verified: 1,
            full_node_mirror_enabled: false,
            full_node_mirror_max_producers: 32,
        },
    )
    .unwrap();

    assert_eq!(coordinator.restored_retry_states, 1);
    let restored = runtime.snapshot();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].producer, configured);
    assert_eq!(restored[0].consecutive_failures, 2);
    assert_eq!(restored[0].retry_not_before, Some(TEST_NOW + 300));
}

#[test]
fn incomplete_rounds_use_bounded_catch_up_cadence() {
    let configured = Duration::from_secs(120);
    assert_eq!(
        directory_sync_next_round_delay(configured, true),
        configured
    );
    assert_eq!(
        directory_sync_next_round_delay(configured, false),
        Duration::from_secs(DIRECTORY_SYNC_CATCH_UP_INTERVAL_SECS)
    );
    let already_fast = Duration::from_secs(30);
    assert_eq!(
        directory_sync_next_round_delay(already_fast, false),
        already_fast
    );
}
