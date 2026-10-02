// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn observation_witness_requires_independent_evidence_and_survives_restart() {
    let observer_temp = TempDir::new().unwrap();
    let witness_temp = TempDir::new().unwrap();
    let empty_witness_temp = TempDir::new().unwrap();
    let observer_path = observer_temp.path().join("directory.db");
    let observer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
    let empty_witness = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x85; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x86; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let (observer_store, _) =
        DirectoryReplicaStore::open(&observer_path, observer.public_key_bytes(), NOW + 20).unwrap();
    let (witness_store, _) = DirectoryReplicaStore::open(
        witness_temp.path().join("directory.db"),
        witness.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    for store in [&observer_store, &witness_store] {
        import_replica_block(store, &producer_a, &object, &block_a, [0x87; 16]);
        import_replica_block(store, &producer_b, &object, &block_b, [0x88; 16]);
    }
    observer_store
        .append_observation_checkpoint(&configured, &observer, NOW + 21)
        .unwrap();
    let checkpoint = observer_store
        .latest_audited_observation_checkpoint(NOW + 22)
        .unwrap()
        .unwrap();
    assert_eq!(
        witness_store
            .evaluate_observation_checkpoint_witness(&checkpoint, NOW + 22)
            .unwrap(),
        DirectoryObservationWitnessDecision::Accepted
    );

    let (empty_store, _) = DirectoryReplicaStore::open(
        empty_witness_temp.path().join("directory.db"),
        empty_witness.public_key_bytes(),
        NOW + 22,
    )
    .unwrap();
    assert_eq!(
        empty_store
            .evaluate_observation_checkpoint_witness(&checkpoint, NOW + 22)
            .unwrap(),
        DirectoryObservationWitnessDecision::EvidenceUnavailable
    );
    let conflicting = DirectoryObservationCheckpointV1::new_signed(
        checkpoint.sequence,
        checkpoint.observed_at,
        checkpoint.previous_checkpoint_hash,
        checkpoint.configured_producer_count,
        checkpoint.producer_tips.clone(),
        [0xf1; 32],
        &observer,
    )
    .unwrap();
    assert_eq!(
        witness_store
            .evaluate_observation_checkpoint_witness(&conflicting, NOW + 22)
            .unwrap(),
        DirectoryObservationWitnessDecision::EvidenceConflict
    );

    let request_id = [0x89; 16];
    let checkpoint_hash = checkpoint.hash();
    let response_timestamp = NOW + 22;
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        checkpoint.sequence,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        response_timestamp,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        observer: observer.public_key_bytes(),
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
        responder: witness.public_key_bytes(),
        response_timestamp,
        outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        signature: witness.sign(&signing_bytes),
    };
    assert!(observer_store
        .persist_observation_checkpoint_witness(&response, NOW + 22)
        .unwrap());
    assert!(!observer_store
        .persist_observation_checkpoint_witness(&response, NOW + 22)
        .unwrap());
    assert!(observer_store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 22)
        .unwrap()
        .is_none());
    let first_outcomes = [
        DirectoryObservationWitnessOutcome::Accepted,
        DirectoryObservationWitnessOutcome::EvidenceUnavailable,
    ];
    let first_outcome_snapshot = observer_store
        .persist_observation_witness_outcome_round(1, NOW + 22, &first_outcomes)
        .unwrap();
    assert_eq!(first_outcome_snapshot.rounds, 1);
    assert_eq!(first_outcome_snapshot.totals.attempts(), 2);
    assert_eq!(first_outcome_snapshot.totals.accepted, 1);
    assert_eq!(first_outcome_snapshot.totals.evidence_unavailable, 1);
    let second_outcomes = [
        DirectoryObservationWitnessOutcome::EvidenceConflict,
        DirectoryObservationWitnessOutcome::TransportFailure,
    ];
    let second_outcome_snapshot = observer_store
        .persist_observation_witness_outcome_round(1, NOW + 23, &second_outcomes)
        .unwrap();
    assert_eq!(second_outcome_snapshot.rounds, 2);
    assert_eq!(second_outcome_snapshot.totals.attempts(), 4);
    assert_eq!(second_outcome_snapshot.totals.evidence_conflict, 1);
    assert_eq!(second_outcome_snapshot.totals.transport_failures, 1);
    assert_eq!(second_outcome_snapshot.last_round, {
        let mut expected = DirectoryObservationWitnessOutcomeCounters::default();
        expected.record(DirectoryObservationWitnessOutcome::EvidenceConflict);
        expected.record(DirectoryObservationWitnessOutcome::TransportFailure);
        expected
    });
    let snapshot = observer_store.status_snapshot().unwrap();
    assert_eq!(snapshot.observation_checkpoint_witnesses, 1);
    assert_eq!(snapshot.observation_checkpoint_witnessed_sequence, 1);
    assert_eq!(snapshot.observation_checkpoint_latest_witnesses, 1);
    assert_eq!(
        snapshot.observation_witness_outcomes,
        second_outcome_snapshot
    );
    let audit = observer_store.audit(NOW + 24).unwrap();
    assert_eq!(audit.observation_checkpoint_witnesses, 1);
    assert_eq!(audit.observation_checkpoint_witnessed_sequence, 1);
    assert_eq!(audit.observation_checkpoint_latest_witnesses, 1);
    assert_eq!(audit.observation_witness_outcomes, second_outcome_snapshot);
    drop(observer_store);

    let (_, reopened) =
        DirectoryReplicaStore::open(&observer_path, observer.public_key_bytes(), NOW + 25).unwrap();
    assert_eq!(reopened.observation_checkpoint_witnesses, 1);
    assert_eq!(reopened.observation_checkpoint_witnessed_sequence, 1);
    assert_eq!(
        reopened.observation_witness_outcomes,
        second_outcome_snapshot
    );
}

#[test]
fn observation_witness_set_is_bounded_at_write_and_audit() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let observer = IdentityKeyPair::from_bytes(&[0xa8; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0xa9; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0xaa; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0xab; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let (store, _) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer_a, &object, &block_a, [0xac; 16]);
    import_replica_block(&store, &producer_b, &object, &block_b, [0xad; 16]);
    store
        .append_observation_checkpoint(&configured, &observer, NOW + 21)
        .unwrap();
    let checkpoint = store
        .latest_audited_observation_checkpoint(NOW + 22)
        .unwrap()
        .unwrap();
    let checkpoint_hash = checkpoint.hash();

    for index in 0..=MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
        let seed = 0xb0u8.saturating_add(u8::try_from(index).unwrap());
        let witness = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
        let request_id = [seed; 16];
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            NOW + 22,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp: NOW + 22,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        };
        let result = store.persist_observation_checkpoint_witness(&response, NOW + 22);
        if index < MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            assert!(result.unwrap());
        } else {
            assert!(result.is_err());
        }
    }

    let snapshot = store.status_snapshot().unwrap();
    assert_eq!(
        snapshot.observation_checkpoint_witnesses,
        u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).unwrap()
    );
    assert_eq!(
        snapshot.observation_checkpoint_latest_witnesses,
        u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).unwrap()
    );
    assert!(store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 23)
        .unwrap()
        .is_none());
    assert_eq!(
        store
            .audit(NOW + 23)
            .unwrap()
            .observation_checkpoint_witnesses,
        u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).unwrap()
    );
}

#[test]
fn observation_witness_target_counts_only_current_pins_and_survives_restart() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let observer = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0xc3; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0xc4; 32]).unwrap();
    let witness_a = IdentityKeyPair::from_bytes(&[0xc5; 32]).unwrap();
    let witness_b = IdentityKeyPair::from_bytes(&[0xc6; 32]).unwrap();
    let retired_witness = IdentityKeyPair::from_bytes(&[0xc7; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let configured_producers = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let eligible_witnesses = [witness_a.public_key_bytes(), witness_b.public_key_bytes()];
    let (store, _) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer_a, &object, &block_a, [0xc8; 16]);
    import_replica_block(&store, &producer_b, &object, &block_b, [0xc9; 16]);
    store
        .append_observation_checkpoint(&configured_producers, &observer, NOW + 21)
        .unwrap();
    let checkpoint = store
        .latest_audited_observation_checkpoint(NOW + 22)
        .unwrap()
        .unwrap();

    let initial = store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 21,
            NOW + 22,
            2,
            &eligible_witnesses,
        )
        .unwrap()
        .unwrap();
    assert_eq!(initial.checkpoint, checkpoint);
    assert!(initial.witnessed_by.is_empty());
    assert_eq!(initial.minimum_witnesses, 2);

    let newer_object = descriptor(&subject, 2);
    let newer_block_a = block(&producer_a, 2, block_a.hash(), &newer_object);
    let newer_block_b = block(&producer_b, 2, block_b.hash(), &newer_object);
    import_replica_block(
        &store,
        &producer_a,
        &newer_object,
        &newer_block_a,
        [0xcd; 16],
    );
    import_replica_block(
        &store,
        &producer_b,
        &newer_object,
        &newer_block_b,
        [0xce; 16],
    );
    let newer_checkpoint = store
        .append_observation_checkpoint(&configured_producers, &observer, NOW + 23)
        .unwrap();
    assert!(newer_checkpoint.appended);
    assert_eq!(newer_checkpoint.sequence, checkpoint.sequence + 1);

    let retired_response =
        accepted_observation_witness_response(&observer, &retired_witness, &checkpoint, 0xca);
    assert!(store
        .persist_observation_checkpoint_witness(&retired_response, NOW + 22)
        .unwrap());
    let after_retired = store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 24,
            2,
            &eligible_witnesses,
        )
        .unwrap()
        .unwrap();
    assert_eq!(after_retired.checkpoint.sequence, checkpoint.sequence);
    assert!(after_retired.witnessed_by.is_empty());
    assert_eq!(
        store
            .verified_observation_witness_count_for_pins(
                checkpoint.sequence,
                &eligible_witnesses,
                NOW + 24,
            )
            .unwrap(),
        0
    );
    assert!(store
        .latest_observation_certificate_for_pins(&eligible_witnesses, 1, NOW + 24)
        .unwrap()
        .is_none());

    let witness_a_response =
        accepted_observation_witness_response(&observer, &witness_a, &checkpoint, 0xcb);
    assert!(store
        .persist_observation_checkpoint_witness(&witness_a_response, NOW + 22)
        .unwrap());
    let partial = store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 24,
            2,
            &eligible_witnesses,
        )
        .unwrap()
        .unwrap();
    assert_eq!(partial.checkpoint.sequence, checkpoint.sequence);
    assert_eq!(partial.witnessed_by, vec![witness_a.public_key_bytes()]);
    assert_eq!(
        store
            .verified_observation_witness_count_for_pins(
                checkpoint.sequence,
                &eligible_witnesses,
                NOW + 24,
            )
            .unwrap(),
        1
    );
    assert!(store
        .latest_observation_certificate_for_pins(&eligible_witnesses, 2, NOW + 24)
        .unwrap()
        .is_none());
    assert!(store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 21,
            NOW + 23,
            1,
            &[witness_a.public_key_bytes()],
        )
        .unwrap()
        .is_none());
    let witness_b_response =
        accepted_observation_witness_response(&observer, &witness_b, &checkpoint, 0xcc);
    assert!(store
        .persist_observation_checkpoint_witness(&witness_b_response, NOW + 22)
        .unwrap());
    assert!(store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 21,
            NOW + 24,
            2,
            &eligible_witnesses,
        )
        .unwrap()
        .is_none());
    assert_eq!(
        store
            .verified_observation_witness_count_for_pins(
                checkpoint.sequence,
                &eligible_witnesses,
                NOW + 24,
            )
            .unwrap(),
        2
    );
    let certificate = store
        .latest_observation_certificate_for_pins(&eligible_witnesses, 2, NOW + 24)
        .unwrap()
        .unwrap();
    assert_eq!(certificate.checkpoint, checkpoint);
    assert_eq!(certificate.minimum_witnesses, 2);
    assert_eq!(certificate.receipts.len(), 2);
    assert!(certificate
        .receipts
        .iter()
        .all(|receipt| eligible_witnesses.contains(&receipt.responder)));
    assert!(certificate
        .receipts
        .iter()
        .all(|receipt| receipt.responder != retired_witness.public_key_bytes()));
    certificate
        .verify_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, NOW + 24)
        .unwrap();
    assert!(store
        .verified_observation_witness_count_for_pins(
            checkpoint.sequence,
            &[witness_a.public_key_bytes(), witness_a.public_key_bytes()],
            NOW + 24,
        )
        .is_err());
    let next = store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 24,
            2,
            &eligible_witnesses,
        )
        .unwrap()
        .unwrap();
    assert_eq!(next.checkpoint.sequence, newer_checkpoint.sequence);
    assert!(next.witnessed_by.is_empty());
    assert!(store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 21,
            NOW + 23,
            0,
            &eligible_witnesses,
        )
        .is_err());
    assert!(store
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 21,
            NOW + 23,
            2,
            &[witness_a.public_key_bytes(), witness_a.public_key_bytes()],
        )
        .is_err());
    drop(store);

    let (reopened, audit) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 24).unwrap();
    assert_eq!(audit.observation_checkpoint_witnesses, 3);
    assert!(reopened
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 21,
            NOW + 24,
            2,
            &eligible_witnesses,
        )
        .unwrap()
        .is_none());
    assert!(reopened
        .latest_observation_certificate_for_pins(&eligible_witnesses, 2, NOW + 24)
        .unwrap()
        .is_some());
}

#[test]
fn witness_failure_drill_keeps_unsatisfied_floor_across_restart_and_pin_rotation() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let observer = IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0xd3; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0xd4; 32]).unwrap();
    let witness_a = IdentityKeyPair::from_bytes(&[0xd5; 32]).unwrap();
    let witness_b = IdentityKeyPair::from_bytes(&[0xd6; 32]).unwrap();
    let witness_c = IdentityKeyPair::from_bytes(&[0xd7; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let configured_producers = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let original_pins = [witness_a.public_key_bytes(), witness_b.public_key_bytes()];
    let rotated_pins = [witness_b.public_key_bytes(), witness_c.public_key_bytes()];
    let (store, _) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer_a, &object, &block_a, [0xd8; 16]);
    import_replica_block(&store, &producer_b, &object, &block_b, [0xd9; 16]);
    store
        .append_observation_checkpoint(&configured_producers, &observer, NOW + 21)
        .unwrap();
    let first_checkpoint = store
        .latest_audited_observation_checkpoint(NOW + 22)
        .unwrap()
        .unwrap();

    let newer_object = descriptor(&subject, 2);
    let newer_block_a = block(&producer_a, 2, block_a.hash(), &newer_object);
    let newer_block_b = block(&producer_b, 2, block_b.hash(), &newer_object);
    import_replica_block(
        &store,
        &producer_a,
        &newer_object,
        &newer_block_a,
        [0xda; 16],
    );
    import_replica_block(
        &store,
        &producer_b,
        &newer_object,
        &newer_block_b,
        [0xdb; 16],
    );
    let second_checkpoint = store
        .append_observation_checkpoint(&configured_producers, &observer, NOW + 23)
        .unwrap();
    assert!(second_checkpoint.appended);

    let witness_a_response =
        accepted_observation_witness_response(&observer, &witness_a, &first_checkpoint, 0xdc);
    assert!(store
        .persist_observation_checkpoint_witness(&witness_a_response, NOW + 24)
        .unwrap());
    store
        .persist_observation_witness_outcome_round(
            first_checkpoint.sequence,
            NOW + 24,
            &[
                DirectoryObservationWitnessOutcome::Accepted,
                DirectoryObservationWitnessOutcome::PeerUnavailable,
            ],
        )
        .unwrap();
    drop(store);

    let (reopened, audit) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 25).unwrap();
    assert_eq!(audit.observation_checkpoint_witnesses, 1);
    let after_restart = reopened
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 25,
            2,
            &original_pins,
        )
        .unwrap()
        .unwrap();
    assert_eq!(after_restart.checkpoint.sequence, first_checkpoint.sequence);
    assert_eq!(
        after_restart.witnessed_by,
        vec![witness_a.public_key_bytes()]
    );

    reopened
        .persist_observation_witness_outcome_round(
            first_checkpoint.sequence,
            NOW + 25,
            &[DirectoryObservationWitnessOutcome::PeerUnavailable],
        )
        .unwrap();
    let still_blocked = reopened
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 25,
            2,
            &original_pins,
        )
        .unwrap()
        .unwrap();
    assert_eq!(still_blocked.checkpoint.sequence, first_checkpoint.sequence);
    assert_eq!(
        still_blocked.witnessed_by,
        vec![witness_a.public_key_bytes()]
    );

    let witness_b_response =
        accepted_observation_witness_response(&observer, &witness_b, &first_checkpoint, 0xdd);
    assert!(reopened
        .persist_observation_checkpoint_witness(&witness_b_response, NOW + 26)
        .unwrap());
    let completed_snapshot = reopened
        .persist_observation_witness_outcome_round(
            first_checkpoint.sequence,
            NOW + 26,
            &[DirectoryObservationWitnessOutcome::Accepted],
        )
        .unwrap();
    assert_eq!(completed_snapshot.rounds, 3);
    assert_eq!(completed_snapshot.totals.accepted, 2);
    assert_eq!(completed_snapshot.totals.peer_unavailable, 2);
    let next_original_target = reopened
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 26,
            2,
            &original_pins,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        next_original_target.checkpoint.sequence,
        second_checkpoint.sequence
    );

    let reopened_by_rotation = reopened
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 26,
            2,
            &rotated_pins,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        reopened_by_rotation.checkpoint.sequence,
        first_checkpoint.sequence
    );
    assert_eq!(
        reopened_by_rotation.witnessed_by,
        vec![witness_b.public_key_bytes()]
    );
    let witness_c_response =
        accepted_observation_witness_response(&observer, &witness_c, &first_checkpoint, 0xde);
    assert!(reopened
        .persist_observation_checkpoint_witness(&witness_c_response, NOW + 26)
        .unwrap());
    assert_eq!(
        reopened
            .verified_observation_witness_count_for_pins(
                first_checkpoint.sequence,
                &rotated_pins,
                NOW + 26,
            )
            .unwrap(),
        2
    );
    let next_rotated_target = reopened
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 26,
            2,
            &rotated_pins,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        next_rotated_target.checkpoint.sequence,
        second_checkpoint.sequence
    );
    drop(reopened);

    let (reopened_again, audit) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 27).unwrap();
    assert_eq!(audit.observation_checkpoint_witnesses, 3);
    let restart_target = reopened_again
        .next_audited_mature_observation_checkpoint_below_witness_threshold(
            NOW + 23,
            NOW + 27,
            2,
            &rotated_pins,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        restart_target.checkpoint.sequence,
        second_checkpoint.sequence
    );
    assert!(restart_target.witnessed_by.is_empty());
}

#[test]
fn witness_runtime_uses_bounded_mutually_exclusive_buckets() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    let outcomes = [
        DirectoryObservationWitnessOutcome::Accepted,
        DirectoryObservationWitnessOutcome::EvidenceUnavailable,
        DirectoryObservationWitnessOutcome::EvidenceConflict,
        DirectoryObservationWitnessOutcome::PeerUnavailable,
        DirectoryObservationWitnessOutcome::TransportFailure,
        DirectoryObservationWitnessOutcome::VerificationFailure,
        DirectoryObservationWitnessOutcome::PersistenceFailure,
    ];
    runtime.record_observation_witness_round(3, NOW, &outcomes, false);
    let snapshot = runtime.observation_witness_snapshot();
    assert_eq!(snapshot.rounds, 1);
    assert_eq!(snapshot.totals.attempts(), 7);
    assert_eq!(snapshot.totals.accepted, 1);
    assert_eq!(snapshot.totals.evidence_unavailable, 1);
    assert_eq!(snapshot.totals.evidence_conflict, 1);
    assert_eq!(snapshot.totals.peer_unavailable, 1);
    assert_eq!(snapshot.totals.transport_failures, 1);
    assert_eq!(snapshot.totals.verification_failures, 1);
    assert_eq!(snapshot.totals.persistence_failures, 1);
    assert_eq!(snapshot.last_checkpoint_sequence, 3);
    assert_eq!(snapshot.last_round_at, Some(NOW));
    assert_eq!(snapshot.last_success_at, Some(NOW));
    assert_eq!(snapshot.last_failure_at, Some(NOW));
    assert_eq!(snapshot.telemetry_persistence_failures, 1);
}

#[test]
fn witness_recovery_runtime_keeps_only_aggregate_transport_outcomes() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.record_observation_witness_recovery_selection(5, 3, 1, 2, NOW);
    runtime.record_observation_witness_recovery_attempt(false, true, NOW + 1);
    runtime.record_observation_witness_recovery_attempt(false, false, NOW + 2);
    runtime.record_observation_witness_recovery_attempt(true, false, NOW + 3);
    runtime.record_observation_witness_recovery_exhausted(NOW + 4);
    runtime.record_observation_witness_recovery_failed_closed(NOW + 5);

    let snapshot = runtime.observation_witness_recovery_snapshot();
    assert_eq!(snapshot.selections, 1);
    assert_eq!(snapshot.latest_candidates, 5);
    assert_eq!(snapshot.latest_routeable_candidates, 3);
    assert_eq!(snapshot.latest_capability_cached_unavailable, 1);
    assert_eq!(snapshot.latest_selected, 2);
    assert_eq!(snapshot.attempts, 4);
    assert_eq!(snapshot.succeeded, 1);
    assert_eq!(snapshot.capability_unavailable, 1);
    assert_eq!(snapshot.transport_failures, 1);
    assert_eq!(snapshot.exhausted, 1);
    assert_eq!(snapshot.failed_closed, 1);
    assert_eq!(snapshot.last_attempt_at, Some(NOW + 5));
    assert_eq!(snapshot.last_success_at, Some(NOW + 3));
    assert_eq!(snapshot.last_failure_at, Some(NOW + 5));
    assert_eq!(
        snapshot.last_outcome,
        Some(DirectoryObservationWitnessRecoveryOutcome::FailedClosed)
    );
    assert!(snapshot.attempt_outcomes_consistent());
    assert_eq!(
        snapshot.health(),
        DirectoryObservationWitnessRecoveryHealth::FailedClosed
    );
}

#[test]
fn witness_carrier_runtime_uses_mutually_exclusive_privacy_safe_outcomes() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    let outcomes = [
        DirectoryObservationWitnessCarrierOutcome::Forwarded,
        DirectoryObservationWitnessCarrierOutcome::PolicyRejected,
        DirectoryObservationWitnessCarrierOutcome::InvalidRequest,
        DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
        DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable,
        DirectoryObservationWitnessCarrierOutcome::TargetRejected,
        DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse,
        DirectoryObservationWitnessCarrierOutcome::TargetCoolingDown,
        DirectoryObservationWitnessCarrierOutcome::LocalOverloaded,
        DirectoryObservationWitnessCarrierOutcome::LocalFailure,
    ];
    for (offset, outcome) in outcomes.into_iter().enumerate() {
        runtime.record_observation_witness_carrier_outcome(
            outcome,
            NOW + u64::try_from(offset).unwrap(),
        );
    }

    let snapshot = runtime.observation_witness_carrier_snapshot();
    assert_eq!(snapshot.requests, 10);
    assert_eq!(snapshot.forwarded, 1);
    assert_eq!(snapshot.policy_rejected, 1);
    assert_eq!(snapshot.invalid_requests, 1);
    assert_eq!(snapshot.target_unavailable, 1);
    assert_eq!(snapshot.target_capability_unavailable, 1);
    assert_eq!(snapshot.target_rejected, 1);
    assert_eq!(snapshot.target_invalid_response, 1);
    assert_eq!(snapshot.target_cooling_down, 1);
    assert_eq!(snapshot.local_overloaded, 1);
    assert_eq!(snapshot.local_failures, 1);
    assert_eq!(snapshot.last_request_at, Some(NOW + 9));
    assert_eq!(snapshot.last_forwarded_at, Some(NOW));
    assert_eq!(snapshot.last_failure_at, Some(NOW + 9));
    assert_eq!(
        snapshot.last_outcome,
        Some(DirectoryObservationWitnessCarrierOutcome::LocalFailure)
    );
    assert!(snapshot.terminal_outcomes_consistent());
    assert_eq!(
        snapshot.health(),
        DirectoryObservationWitnessCarrierHealth::Degraded
    );
}

#[test]
fn witness_availability_health_preserves_same_second_terminal_order() {
    // [WITNESS-TERMINAL-STATE 2026-07-29 by Codex] Unix-second equality
    // cannot reveal call order. The service-owned terminal enum must.
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.record_observation_witness_recovery_attempt(true, false, NOW);
    runtime.record_observation_witness_recovery_exhausted(NOW);
    let exhausted = runtime.observation_witness_recovery_snapshot();
    assert_eq!(exhausted.last_success_at, Some(NOW));
    assert_eq!(exhausted.last_failure_at, Some(NOW));
    assert_eq!(
        exhausted.health(),
        DirectoryObservationWitnessRecoveryHealth::Exhausted
    );

    runtime.record_observation_witness_recovery_attempt(true, false, NOW);
    assert_eq!(
        runtime.observation_witness_recovery_snapshot().health(),
        DirectoryObservationWitnessRecoveryHealth::Recovered
    );

    runtime.record_observation_witness_carrier_outcome(
        DirectoryObservationWitnessCarrierOutcome::Forwarded,
        NOW,
    );
    runtime.record_observation_witness_carrier_outcome(
        DirectoryObservationWitnessCarrierOutcome::TargetUnavailable,
        NOW,
    );
    let degraded = runtime.observation_witness_carrier_snapshot();
    assert_eq!(degraded.last_forwarded_at, Some(NOW));
    assert_eq!(degraded.last_failure_at, Some(NOW));
    assert_eq!(
        degraded.health(),
        DirectoryObservationWitnessCarrierHealth::Degraded
    );

    runtime.record_observation_witness_carrier_outcome(
        DirectoryObservationWitnessCarrierOutcome::Forwarded,
        NOW,
    );
    assert_eq!(
        runtime.observation_witness_carrier_snapshot().health(),
        DirectoryObservationWitnessCarrierHealth::Active
    );
}

#[test]
fn witness_availability_health_fails_closed_on_counter_drift() {
    // [WITNESS-TERMINAL-STATE 2026-07-29 by Codex] Snapshot integrity is
    // checked independently from presentation and last-result ordering.
    let recovery = DirectoryObservationWitnessRecoverySnapshot {
        attempts: 1,
        ..DirectoryObservationWitnessRecoverySnapshot::default()
    };
    assert!(!recovery.attempt_outcomes_consistent());
    assert_eq!(
        recovery.health(),
        DirectoryObservationWitnessRecoveryHealth::Inconsistent
    );

    let carrier = DirectoryObservationWitnessCarrierSnapshot {
        requests: 1,
        ..DirectoryObservationWitnessCarrierSnapshot::default()
    };
    assert!(!carrier.terminal_outcomes_consistent());
    assert_eq!(
        carrier.health(),
        DirectoryObservationWitnessCarrierHealth::Inconsistent
    );
}

#[test]
fn tampered_witness_outcome_aggregate_fails_startup_audit() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x8a; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x8b; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x8c; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x8d; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer_a, &object, &block_a, [0x8e; 16]);
    import_replica_block(&store, &producer_b, &object, &block_b, [0x8f; 16]);
    store
        .append_observation_checkpoint(&configured, &local, NOW + 21)
        .unwrap();
    store
        .persist_observation_witness_outcome_round(
            1,
            NOW + 22,
            &[DirectoryObservationWitnessOutcome::EvidenceUnavailable],
        )
        .unwrap();
    {
        let connection = store.connection.lock();
        connection
            .pragma_update(None, "ignore_check_constraints", true)
            .unwrap();
        connection
            .execute(
                "UPDATE directory_observation_witness_outcomes
                 SET attempts_total = attempts_total + 1 WHERE singleton = 1",
                [],
            )
            .unwrap();
    }
    assert!(store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 23)
        .is_err());
    drop(store);

    assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 23).is_err());
}

#[test]
fn observation_witness_combines_local_producer_and_remote_replica_evidence() {
    let observer_temp = TempDir::new().unwrap();
    let witness_temp = TempDir::new().unwrap();
    let observer = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
    let remote = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0xa4; 32]).unwrap();
    let object = descriptor(&subject, 1);

    let witness_path = witness_temp.path().join("directory.db");
    let (local_chain, _) =
        DirectoryChainStore::open(&witness_path, witness.public_key_bytes(), NOW + 1).unwrap();
    local_chain
        .append_descriptors(std::slice::from_ref(&object), NOW + 1, &witness)
        .unwrap();
    let local_block = local_chain.block(1).unwrap().unwrap();
    let remote_block = block(&remote, 1, [0u8; 32], &object);
    let (witness_store, _) =
        DirectoryReplicaStore::open(&witness_path, witness.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&witness_store, &remote, &object, &remote_block, [0xa5; 16]);

    let observer_path = observer_temp.path().join("directory.db");
    let (observer_store, _) =
        DirectoryReplicaStore::open(&observer_path, observer.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&observer_store, &witness, &object, &local_block, [0xa6; 16]);
    import_replica_block(&observer_store, &remote, &object, &remote_block, [0xa7; 16]);
    observer_store
        .append_observation_checkpoint(
            &[witness.public_key_bytes(), remote.public_key_bytes()],
            &observer,
            NOW + 21,
        )
        .unwrap();
    let checkpoint = observer_store
        .latest_audited_observation_checkpoint(NOW + 22)
        .unwrap()
        .unwrap();
    assert!(checkpoint
        .producer_tips
        .iter()
        .any(|tip| tip.producer == witness.public_key_bytes()));
    assert_eq!(
        witness_store
            .evaluate_observation_checkpoint_witness(&checkpoint, NOW + 22)
            .unwrap(),
        DirectoryObservationWitnessDecision::Accepted
    );
}

#[test]
fn tampered_observation_witness_fails_startup_audit() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let observer = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
    let witness = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
    let producer_a = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
    let producer_b = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x95; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&producer_a, 1, [0u8; 32], &object);
    let block_b = block(&producer_b, 1, [0u8; 32], &object);
    let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
    let (store, _) =
        DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer_a, &object, &block_a, [0x96; 16]);
    import_replica_block(&store, &producer_b, &object, &block_b, [0x97; 16]);
    store
        .append_observation_checkpoint(&configured, &observer, NOW + 21)
        .unwrap();
    let checkpoint = store
        .latest_audited_observation_checkpoint(NOW + 22)
        .unwrap()
        .unwrap();
    let request_id = [0x98; 16];
    let checkpoint_hash = checkpoint.hash();
    let signing_bytes = directory_observation_witness_response_signing_bytes(
        &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        &request_id,
        &observer.public_key_bytes(),
        checkpoint.sequence,
        &checkpoint_hash,
        &witness.public_key_bytes(),
        NOW + 22,
        DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
    );
    let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
        chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
        request_id,
        observer: observer.public_key_bytes(),
        checkpoint_sequence: checkpoint.sequence,
        checkpoint_hash,
        responder: witness.public_key_bytes(),
        response_timestamp: NOW + 22,
        outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        signature: witness.sign(&signing_bytes),
    };
    store
        .persist_observation_checkpoint_witness(&response, NOW + 22)
        .unwrap();
    {
        let connection = store.connection.lock();
        let mut response_blob: Vec<u8> = connection
            .query_row(
                "SELECT response_blob FROM directory_observation_checkpoint_witnesses",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let last = response_blob.len() - 1;
        response_blob[last] ^= 1;
        connection
            .execute(
                "UPDATE directory_observation_checkpoint_witnesses SET response_blob = ?1",
                params![response_blob],
            )
            .unwrap();
    }
    assert!(store
        .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 23)
        .is_err());
    assert!(store
        .verified_observation_witness_count_for_pins(
            checkpoint.sequence,
            &[witness.public_key_bytes()],
            NOW + 23,
        )
        .is_err());
    // [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] Export is a
    // fresh audit boundary, so a post-startup database mutation cannot be
    // wrapped into apparently valid portable evidence.
    assert!(store
        .latest_observation_certificate_for_pins(&[witness.public_key_bytes()], 1, NOW + 23,)
        .is_err());
    drop(store);
    assert!(DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 23).is_err());
}

#[test]
fn schema_v6_adds_witness_policy_metadata_and_table_atomically() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x2c; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "ALTER TABLE directory_replica_meta RENAME TO directory_replica_meta_v7;
             CREATE TABLE directory_replica_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32)
             );
             INSERT INTO directory_replica_meta
                 (singleton, schema_version, chain_id, local_node_id)
             SELECT singleton, 6, chain_id, local_node_id
             FROM directory_replica_meta_v7;
             DROP TABLE directory_replica_meta_v7;
             DROP TABLE directory_observation_witness_policies;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(audit.observation_witness_policy_epochs, 0);
    let connection = store.connection.lock();
    let version: i64 = connection
        .query_row(
            "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let policy_columns: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('directory_replica_meta')
             WHERE name IN ('witness_policy_epoch', 'witness_policy_head')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let policy_table: String = connection
        .query_row(
            "SELECT name FROM sqlite_master
             WHERE type = 'table'
               AND name = 'directory_observation_witness_policies'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(policy_columns, 2);
    assert_eq!(policy_table, "directory_observation_witness_policies");
}
