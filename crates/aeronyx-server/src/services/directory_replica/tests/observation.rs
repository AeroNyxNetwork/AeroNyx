// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn recent_observation_convergence_is_multi_source_and_order_independent() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x21; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x23; 32]).unwrap();
    let pending = IdentityKeyPair::from_bytes(&[0x24; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x25; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let frame_a = response_frame(
        &producer_a,
        vec![block_a.clone()],
        false,
        1,
        block_a.hash(),
        [0x26; 16],
    );
    let frame_b = response_frame(
        &producer_b,
        vec![block_b.clone()],
        false,
        1,
        block_b.hash(),
        [0x27; 16],
    );
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    for (producer, replica_block, frame) in [
        (&producer_a, &block_a, &frame_a),
        (&producer_b, &block_b, &frame_b),
    ] {
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(replica_block),
                std::slice::from_ref(&object),
                1,
                replica_block.hash(),
                frame,
                NOW + 20,
            )
            .unwrap();
    }

    let configured = [
        producer_a.public_key_bytes(),
        producer_b.public_key_bytes(),
        pending.public_key_bytes(),
    ];
    let snapshot = store.observation_convergence(&configured).unwrap();
    assert_eq!(snapshot.configured_producers, 3);
    assert_eq!(snapshot.eligible_producers, 2);
    assert_eq!(snapshot.pending_producers, 1);
    assert_eq!(snapshot.excluded_quarantined_producers, 0);
    assert_eq!(snapshot.window_blocks, 32);
    assert_eq!(snapshot.recent_commitments, 2);
    assert_eq!(snapshot.distinct_recent_commitments, 1);
    assert_eq!(snapshot.multi_source_recent_commitments, 1);
    assert_eq!(snapshot.all_eligible_source_recent_commitments, 1);
    assert!(snapshot.observation_root.is_some());

    let reversed = [configured[2], configured[1], configured[0]];
    assert_eq!(store.observation_convergence(&reversed).unwrap(), snapshot);
    assert!(store
        .observation_convergence(&[configured[0], configured[0]])
        .is_err());
    assert!(store
        .append_observation_checkpoint(&configured, &local, NOW + 21)
        .is_err());
}

#[test]
fn observation_checkpoints_are_signed_linked_recomputed_and_idempotent() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x54; 32]).unwrap();
    let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let first_object = descriptor(&subject, 1);
    let first_a = block(&producer_a, 1, [0u8; 32], &first_object);
    let first_b = block(&producer_b, 1, [0u8; 32], &first_object);
    let first_frame_a = response_frame(
        &producer_a,
        vec![first_a.clone()],
        false,
        1,
        first_a.hash(),
        [0x55; 16],
    );
    let first_frame_b = response_frame(
        &producer_b,
        vec![first_b.clone()],
        false,
        1,
        first_b.hash(),
        [0x56; 16],
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    for (producer, replica_block, frame) in [
        (&producer_a, &first_a, &first_frame_a),
        (&producer_b, &first_b, &first_frame_b),
    ] {
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(replica_block),
                std::slice::from_ref(&first_object),
                1,
                replica_block.hash(),
                frame,
                NOW + 20,
            )
            .unwrap();
    }

    let first = store
        .append_observation_checkpoint(&configured, &local, NOW + 21)
        .unwrap();
    assert!(first.appended);
    assert_eq!(first.sequence, 1);
    assert_eq!(first.producer_count, 2);
    let unchanged = store
        .append_observation_checkpoint(&configured, &local, NOW + 22)
        .unwrap();
    assert!(!unchanged.appended);
    assert_eq!(
        unchanged,
        DirectoryObservationCheckpointAppendReport {
            appended: false,
            ..first
        }
    );
    assert!(store
        .append_observation_checkpoint(
            &configured,
            &IdentityKeyPair::from_bytes(&[0x57; 32]).unwrap(),
            NOW + 22,
        )
        .is_err());

    let second_object = descriptor(&subject, 2);
    let second_a = block(&producer_a, 2, first_a.hash(), &second_object);
    let second_b = block(&producer_b, 2, first_b.hash(), &second_object);
    let second_frame_a = response_frame(
        &producer_a,
        vec![second_a.clone()],
        false,
        2,
        second_a.hash(),
        [0x58; 16],
    );
    let second_frame_b = response_frame(
        &producer_b,
        vec![second_b.clone()],
        false,
        2,
        second_b.hash(),
        [0x59; 16],
    );
    for (producer, replica_block, frame) in [
        (&producer_a, &second_a, &second_frame_a),
        (&producer_b, &second_b, &second_frame_b),
    ] {
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(replica_block),
                std::slice::from_ref(&second_object),
                2,
                replica_block.hash(),
                frame,
                NOW + 23,
            )
            .unwrap();
    }
    let second = store
        .append_observation_checkpoint(&configured, &local, NOW + 24)
        .unwrap();
    assert!(second.appended);
    assert_eq!(second.sequence, 2);
    assert_ne!(second.checkpoint_hash, first.checkpoint_hash);
    assert_eq!(
        store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 23, NOW + 25)
            .unwrap()
            .unwrap()
            .sequence,
        1
    );
    assert_eq!(
        store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 24, NOW + 25)
            .unwrap()
            .unwrap()
            .sequence,
        2
    );
    assert!(store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 26, NOW + 25)
        .is_err());
    store
        .persist_observation_witness_outcome_round(
            2,
            NOW + 25,
            &[DirectoryObservationWitnessOutcome::EvidenceUnavailable],
        )
        .unwrap();
    assert!(store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 23, NOW + 26)
        .unwrap()
        .is_none());
    let snapshot = store.status_snapshot().unwrap();
    assert_eq!(snapshot.observation_checkpoints, 2);
    assert_eq!(snapshot.observation_checkpoint_sequence, 2);
    assert_eq!(snapshot.observation_checkpoint_hash, second.checkpoint_hash);
    let audit = store.audit(NOW + 26).unwrap();
    assert_eq!(audit.observation_checkpoints, 2);
    assert_eq!(audit.observation_checkpoint_sequence, 2);
    assert_eq!(audit.observation_checkpoint_hash, second.checkpoint_hash);
    drop(store);

    let (reopened_store, reopened) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 27).unwrap();
    assert_eq!(reopened.observation_checkpoints, 2);
    assert_eq!(reopened.observation_checkpoint_sequence, 2);
    {
        let connection = reopened_store.connection.lock();
        connection
            .execute(
                "UPDATE directory_observation_checkpoints
                 SET observed_at = observed_at + 1 WHERE sequence = 1",
                [],
            )
            .unwrap();
    }
    assert!(reopened_store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 24, NOW + 28)
        .is_err());
}

#[test]
fn tampered_observation_checkpoint_fails_startup_audit() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let frame_a = response_frame(
        &producer_a,
        vec![block_a.clone()],
        false,
        1,
        block_a.hash(),
        [0x65; 16],
    );
    let frame_b = response_frame(
        &producer_b,
        vec![block_b.clone()],
        false,
        1,
        block_b.hash(),
        [0x66; 16],
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    for (producer, replica_block, frame) in [
        (&producer_a, &block_a, &frame_a),
        (&producer_b, &block_b, &frame_b),
    ] {
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(replica_block),
                std::slice::from_ref(&object),
                1,
                replica_block.hash(),
                frame,
                NOW + 20,
            )
            .unwrap();
    }
    store
        .append_observation_checkpoint(
            &[producer_a.public_key_bytes(), producer_b.public_key_bytes()],
            &local,
            NOW + 21,
        )
        .unwrap();
    let connection = store.connection.lock();
    let mut blob: Vec<u8> = connection
        .query_row(
            "SELECT checkpoint_blob FROM directory_observation_checkpoints
             WHERE sequence = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    *blob.last_mut().unwrap() ^= 1;
    connection
        .execute(
            "UPDATE directory_observation_checkpoints
             SET checkpoint_blob = ?1 WHERE sequence = 1",
            params![blob],
        )
        .unwrap();
    drop(connection);
    assert!(store.audit(NOW + 22).is_err());
    assert!(store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 22)
        .is_err());
    drop(store);
    assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 22).is_err());
}

#[test]
fn observation_convergence_reads_only_the_bounded_recent_block_window() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x70; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    let mut previous = [0u8; 32];
    for height in 1..=DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS + 1 {
        let object = descriptor(&subject, height);
        let replica_block = block(&producer, height, previous, &object);
        let frame = response_frame(
            &producer,
            vec![replica_block.clone()],
            false,
            height,
            replica_block.hash(),
            [u8::try_from(height).unwrap(); 16],
        );
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&replica_block),
                std::slice::from_ref(&object),
                height,
                replica_block.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        previous = replica_block.hash();
    }

    let snapshot = store
        .observation_convergence(&[producer.public_key_bytes()])
        .unwrap();
    assert_eq!(snapshot.eligible_producers, 1);
    assert_eq!(snapshot.window_blocks, 32);
    assert_eq!(snapshot.recent_commitments, 32);
    assert_eq!(snapshot.distinct_recent_commitments, 32);
    assert_eq!(snapshot.multi_source_recent_commitments, 0);
    assert_eq!(snapshot.observation_root, None);
}

#[test]
fn quarantined_producer_is_excluded_from_observation_convergence() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x28; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x29; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x2a; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x2b; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x2c; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let first_a = block(&producer_a, 1, [0u8; 32], &object_a);
    let first_b = block(&producer_b, 1, [0u8; 32], &object_a);
    let fork_b = block(&producer_b, 1, [0u8; 32], &object_b);
    let frame_a = response_frame(
        &producer_a,
        vec![first_a.clone()],
        false,
        1,
        first_a.hash(),
        [0x2d; 16],
    );
    let frame_b = response_frame(
        &producer_b,
        vec![first_b.clone()],
        false,
        1,
        first_b.hash(),
        [0x2e; 16],
    );
    let fork_frame = response_frame(
        &producer_b,
        vec![fork_b.clone()],
        false,
        1,
        fork_b.hash(),
        [0x2f; 16],
    );
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    store
        .import_verified_page(
            producer_a.public_key_bytes(),
            std::slice::from_ref(&first_a),
            std::slice::from_ref(&object_a),
            1,
            first_a.hash(),
            &frame_a,
            NOW + 20,
        )
        .unwrap();
    store
        .import_verified_page(
            producer_b.public_key_bytes(),
            std::slice::from_ref(&first_b),
            std::slice::from_ref(&object_a),
            1,
            first_b.hash(),
            &frame_b,
            NOW + 20,
        )
        .unwrap();
    assert!(matches!(
        store.import_verified_page(
            producer_b.public_key_bytes(),
            std::slice::from_ref(&fork_b),
            std::slice::from_ref(&object_b),
            1,
            fork_b.hash(),
            &fork_frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Quarantined(_))
    ));

    let snapshot = store
        .observation_convergence(&[producer_a.public_key_bytes(), producer_b.public_key_bytes()])
        .unwrap();
    assert_eq!(snapshot.configured_producers, 2);
    assert_eq!(snapshot.eligible_producers, 1);
    assert_eq!(snapshot.pending_producers, 0);
    assert_eq!(snapshot.excluded_quarantined_producers, 1);
    assert_eq!(snapshot.recent_commitments, 1);
    assert_eq!(snapshot.multi_source_recent_commitments, 0);
    assert_eq!(snapshot.all_eligible_source_recent_commitments, 0);
    assert_eq!(snapshot.observation_root, None);
}
