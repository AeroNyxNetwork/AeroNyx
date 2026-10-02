// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn retry_state_survives_reopen_and_is_fully_audited() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
    let producer_id = producer.public_key_bytes();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    store
        .persist_retry_failure(
            producer_id,
            2,
            Some(NOW + 120),
            NOW,
            "directory_range_transport_failed",
        )
        .unwrap();
    store.persist_retry_skip(producer_id, NOW + 1).unwrap();
    let states = store.retry_states().unwrap();
    assert_eq!(states.len(), 1);
    assert_eq!(states[0].consecutive_failures, 2);
    assert_eq!(states[0].retry_not_before, Some(NOW + 120));
    assert_eq!(states[0].backoff_skips, 1);
    drop(store);

    let (reopened, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 2).unwrap();
    assert_eq!(audit.producers, 1);
    assert_eq!(audit.retry_states, 1);
    assert_eq!(reopened.retry_states().unwrap(), states);
}

#[test]
fn older_failure_timestamp_cannot_shorten_retry_boundary() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x3a; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x3b; 32]).unwrap();
    let producer_id = producer.public_key_bytes();
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW,
    )
    .unwrap();
    store
        .persist_retry_failure(
            producer_id,
            2,
            Some(NOW + 300),
            NOW,
            "directory_range_transport_failed",
        )
        .unwrap();
    store
        .persist_retry_failure(
            producer_id,
            3,
            Some(NOW + 100),
            NOW - 100,
            "directory_object_transport_failed",
        )
        .unwrap();

    let state = &store.retry_states().unwrap()[0];
    assert_eq!(state.consecutive_failures, 3);
    assert_eq!(state.last_failure_at, NOW);
    assert_eq!(state.retry_not_before, Some(NOW + 300));
    assert_eq!(
        state.last_failure_reason,
        "directory_range_transport_failed"
    );
}

#[test]
fn authenticated_import_atomically_clears_durable_retry_state() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x34; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x35; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x36; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let first = block(&producer, 1, [0u8; 32], &object);
    let frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x37; 16],
    );
    let producer_id = producer.public_key_bytes();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    store
        .persist_retry_failure(
            producer_id,
            3,
            Some(NOW + 300),
            NOW,
            "directory_object_transport_failed",
        )
        .unwrap();
    assert_eq!(store.retry_states().unwrap().len(), 1);

    store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&first),
            std::slice::from_ref(&object),
            1,
            first.hash(),
            &frame,
            NOW + 20,
        )
        .unwrap();
    assert!(store.retry_states().unwrap().is_empty());
    drop(store);

    let (_, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(audit.retry_states, 0);
    assert_eq!(audit.blocks, 1);
}

#[test]
fn retry_state_rejects_peer_controlled_or_unbounded_fields() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x38; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x39; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW,
    )
    .unwrap();
    assert!(store
        .persist_retry_failure(
            producer.public_key_bytes(),
            1,
            None,
            NOW,
            "https://peer.example/private",
        )
        .is_err());
    assert!(store
        .persist_retry_failure(
            producer.public_key_bytes(),
            DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES + 1,
            None,
            NOW,
            "directory_range_transport_failed",
        )
        .is_err());
    assert!(store.retry_states().unwrap().is_empty());
}
