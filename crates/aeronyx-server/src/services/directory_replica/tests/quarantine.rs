// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn carrier_block_fork_requires_valid_producer_signature_before_quarantine() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x65; 32]).unwrap();
    let other_producer = IdentityKeyPair::from_bytes(&[0x78; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let fork = block(&producer, 1, [0u8; 32], &object_b);
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    import_replica_block(&store, &producer, &object_a, &first, [0x66; 16]);

    let mut tampered_fork = fork.clone();
    tampered_fork.producer_signature[0] ^= 1;
    let tampered_frame = carrier_response_frame(
        &producer,
        &carrier,
        vec![tampered_fork.clone()],
        false,
        1,
        tampered_fork.hash(),
        [0x67; 16],
    );
    assert!(matches!(
        store.import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&tampered_fork),
            std::slice::from_ref(&object_b),
            1,
            tampered_fork.hash(),
            &tampered_frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Block(_))
    ));
    assert!(
        !store
            .producer_tip(&producer.public_key_bytes())
            .unwrap()
            .quarantined
    );
    assert!(store
        .incident_summaries(None, 1)
        .unwrap()
        .incidents
        .is_empty());

    let wrong_producer_fork = block(&other_producer, 1, [0u8; 32], &object_b);
    let wrong_producer_frame = carrier_response_frame(
        &producer,
        &carrier,
        vec![wrong_producer_fork.clone()],
        false,
        1,
        wrong_producer_fork.hash(),
        [0x79; 16],
    );
    assert!(matches!(
        store.import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&wrong_producer_fork),
            std::slice::from_ref(&object_b),
            1,
            wrong_producer_fork.hash(),
            &wrong_producer_frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Integrity(_))
    ));
    assert!(
        !store
            .producer_tip(&producer.public_key_bytes())
            .unwrap()
            .quarantined
    );
    assert!(store
        .incident_summaries(None, 1)
        .unwrap()
        .incidents
        .is_empty());

    let fork_frame = carrier_response_frame(
        &producer,
        &carrier,
        vec![fork.clone()],
        false,
        1,
        fork.hash(),
        [0x68; 16],
    );
    assert!(matches!(
        store.import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&fork),
            std::slice::from_ref(&object_b),
            1,
            fork.hash(),
            &fork_frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Quarantined(_))
    ));
    let tip = store.producer_tip(&producer.public_key_bytes()).unwrap();
    assert!(tip.quarantined);
    assert_eq!(tip.tip_hash, first.hash());
    let incidents = store.incident_summaries(None, 1).unwrap().incidents;
    assert_eq!(incidents.len(), 1);
    assert_eq!(incidents[0].kind, "signed_block_fork");
}

#[test]
fn producer_signed_duplicate_height_fork_still_quarantines() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
    let subject_c = IdentityKeyPair::from_bytes(&[0x75; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let object_c = descriptor(&subject_c, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let second = block(&producer, 2, first.hash(), &object_b);
    let signed_fork = block(&producer, 2, first.hash(), &object_c);
    let frame = response_frame(
        &producer,
        vec![second.clone(), signed_fork.clone()],
        false,
        2,
        signed_fork.hash(),
        [0x76; 16],
    );
    let path = temp.path().join("directory.db");
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer, &object_a, &first, [0x77; 16]);

    assert!(matches!(
        store.import_verified_page(
            producer.public_key_bytes(),
            &[second, signed_fork.clone()],
            &[object_b, object_c],
            2,
            signed_fork.hash(),
            &frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Quarantined(ref kind))
            if kind == "signed_block_fork"
    ));
    let tip = store.producer_tip(&producer.public_key_bytes()).unwrap();
    assert!(tip.quarantined);
    assert_eq!(tip.quarantine_kind.as_deref(), Some("signed_block_fork"));
    assert_eq!(tip.tip_height, 1);
    assert_eq!(tip.tip_hash, first.hash());
    let incidents = store.incident_summaries(None, 1).unwrap().incidents;
    assert_eq!(incidents.len(), 1);
    assert_eq!(incidents[0].kind, "signed_block_fork");
    let audit = store.audit(NOW + 21).unwrap();
    assert_eq!(audit.blocks, 1);
    assert_eq!(audit.incidents, 1);
    assert_eq!(audit.quarantined_producers, 1);
    drop(store);
    let (_, reopened_audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 22).unwrap();
    assert_eq!(reopened_audit.blocks, 1);
    assert_eq!(reopened_audit.incidents, 1);
    assert_eq!(reopened_audit.quarantined_producers, 1);
}

#[test]
fn signed_block_fork_is_durably_quarantined_without_rewriting_prefix() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x54; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let fork = block(&producer, 1, [0u8; 32], &object_b);
    let first_frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x55; 16],
    );
    let fork_frame = response_frame(
        &producer,
        vec![fork.clone()],
        false,
        1,
        fork.hash(),
        [0x56; 16],
    );
    let mut invalid_evidence = fork_frame.clone();
    *invalid_evidence.last_mut().unwrap() ^= 0x01;
    assert!(
        verify_incident_response_evidence(&invalid_evidence, &producer.public_key_bytes()).is_err()
    );
    let path = temp.path().join("directory.db");
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    store
        .import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&first),
            std::slice::from_ref(&object_a),
            1,
            first.hash(),
            &first_frame,
            NOW + 20,
        )
        .unwrap();
    let error = store
        .import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&fork),
            std::slice::from_ref(&object_b),
            1,
            fork.hash(),
            &fork_frame,
            NOW + 20,
        )
        .unwrap_err();
    assert!(matches!(error, DirectoryReplicaStoreError::Quarantined(_)));
    let tip = store.producer_tip(&producer.public_key_bytes()).unwrap();
    assert!(tip.quarantined);
    assert_eq!(tip.tip_hash, first.hash());
    assert!(store.incident_summaries(None, 0).is_err());
    assert!(store
        .incident_summaries(None, MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE + 1)
        .is_err());
    let page = store.incident_summaries(None, 1).unwrap();
    assert_eq!(page.incidents.len(), 1);
    assert_eq!(page.next_cursor, None);
    let summary = &page.incidents[0];
    assert_eq!(summary.producer, producer.public_key_bytes());
    assert_eq!(summary.subject_node_id, producer.public_key_bytes());
    assert_eq!(summary.kind, "signed_tip_fork");
    assert_eq!(summary.height, 1);
    assert_eq!(summary.local_hash, first.hash());
    assert_eq!(summary.remote_hash, fork.hash());
    assert!(summary.producer_quarantined);
    assert!(store
        .incident_summaries(Some(summary.incident_digest), 1)
        .unwrap()
        .incidents
        .is_empty());
    let evidence = store
        .incident_evidence(&summary.incident_digest)
        .unwrap()
        .unwrap();
    assert_eq!(evidence.summary, *summary);
    assert_eq!(evidence.evidence_frame, fork_frame);
    let expected_evidence_sha256: [u8; 32] = Sha256::digest(&fork_frame).into();
    assert_eq!(evidence.evidence_sha256, expected_evidence_sha256);
    let incident_digest = summary.incident_digest;
    drop(store);
    let (reopened, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(audit.quarantined_producers, 1);
    assert_eq!(audit.incidents, 1);
    assert!(reopened
        .incident_evidence(&incident_digest)
        .unwrap()
        .is_some());

    let mut corrupted_frame = fork_frame;
    *corrupted_frame.last_mut().unwrap() ^= 0x01;
    let connection = reopened.connection.lock();
    connection
        .execute(
            "UPDATE directory_replica_incidents SET evidence_frame = ?2
             WHERE incident_digest = ?1",
            params![incident_digest.as_slice(), corrupted_frame],
        )
        .unwrap();
    drop(connection);
    assert!(reopened.incident_evidence(&incident_digest).is_err());
}

#[test]
fn signed_resolution_resumes_only_exact_prefix_and_links_repeated_incidents() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let fork = block(&producer, 1, [0u8; 32], &object_b);
    let first_frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x85; 16],
    );
    let fork_frame = response_frame(
        &producer,
        vec![fork.clone()],
        false,
        1,
        fork.hash(),
        [0x86; 16],
    );
    let producer_id = producer.public_key_bytes();
    let path = temp.path().join("directory.db");
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&first),
            std::slice::from_ref(&object_a),
            1,
            first.hash(),
            &first_frame,
            NOW + 20,
        )
        .unwrap();
    assert!(store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&fork),
            std::slice::from_ref(&object_b),
            1,
            fork.hash(),
            &fork_frame,
            NOW + 20,
        )
        .is_err());
    store
        .persist_retry_failure(
            producer_id,
            2,
            Some(NOW + 120),
            NOW + 21,
            "directory_range_transport_failed",
        )
        .unwrap();
    let incident_digest = store.incident_summaries(None, 1).unwrap().incidents[0].incident_digest;
    let tip = store.producer_tip(&producer_id).unwrap();
    assert_eq!(tip.active_incident_digest, Some(incident_digest));

    let predates_incident = resolution_command(&local, incident_digest, &tip, [0x87; 16], NOW + 19);
    assert!(store
        .resolve_quarantine(&predates_incident, NOW + 19)
        .is_err());

    let mut stale_tip = tip.clone();
    stale_tip.tip_hash = [0x99; 32];
    let stale = resolution_command(&local, incident_digest, &stale_tip, [0x88; 16], NOW + 22);
    assert!(store.resolve_quarantine(&stale, NOW + 22).is_err());
    assert_eq!(store.status_snapshot().unwrap().resolutions, 0);

    let first_resolution = resolution_command(&local, incident_digest, &tip, [0x89; 16], NOW + 22);
    let first_report = store
        .resolve_quarantine(&first_resolution, NOW + 22)
        .unwrap();
    let resumed = store.producer_tip(&producer_id).unwrap();
    assert!(!resumed.quarantined);
    assert_eq!(resumed.active_incident_digest, None);
    assert_eq!(
        resumed.last_resolution_digest,
        Some(first_report.resolution_digest)
    );
    assert_eq!(resumed.tip_hash, first.hash());
    assert!(store.retry_states().unwrap().is_empty());
    assert!(store
        .resolve_quarantine(&first_resolution, NOW + 22)
        .is_err());

    assert!(store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&fork),
            std::slice::from_ref(&object_b),
            1,
            fork.hash(),
            &fork_frame,
            NOW + 24,
        )
        .is_err());
    let requarantined = store.producer_tip(&producer_id).unwrap();
    assert_eq!(requarantined.active_incident_digest, Some(incident_digest));
    assert_eq!(
        requarantined.last_resolution_digest,
        Some(first_report.resolution_digest)
    );
    let predates_predecessor = resolution_command(
        &local,
        incident_digest,
        &requarantined,
        [0x8a; 16],
        NOW + 21,
    );
    assert!(store
        .resolve_quarantine(&predates_predecessor, NOW + 21)
        .is_err());
    let second_resolution = resolution_command(
        &local,
        incident_digest,
        &requarantined,
        [0x8b; 16],
        NOW + 25,
    );
    let second_report = store
        .resolve_quarantine(&second_resolution, NOW + 25)
        .unwrap();
    assert_ne!(
        first_report.resolution_digest,
        second_report.resolution_digest
    );
    let snapshot = store.status_snapshot().unwrap();
    assert_eq!(snapshot.incidents, 1);
    assert_eq!(snapshot.resolutions, 2);
    assert_eq!(snapshot.producer_snapshots[0].resolutions, 2);
    let audit = store.audit(NOW + 26).unwrap();
    assert_eq!(audit.incidents, 1);
    assert_eq!(audit.resolutions, 2);
    drop(store);
    let (_, reopened_audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 27).unwrap();
    assert_eq!(reopened_audit.resolutions, 2);
}

#[test]
fn forged_quarantine_clear_without_signed_resolution_fails_audit() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let fork = block(&producer, 1, [0u8; 32], &object_b);
    let first_frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x95; 16],
    );
    let fork_frame = response_frame(
        &producer,
        vec![fork.clone()],
        false,
        1,
        fork.hash(),
        [0x96; 16],
    );
    let producer_id = producer.public_key_bytes();
    let path = temp.path().join("directory.db");
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&first),
            std::slice::from_ref(&object_a),
            1,
            first.hash(),
            &first_frame,
            NOW + 20,
        )
        .unwrap();
    assert!(store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&fork),
            std::slice::from_ref(&object_b),
            1,
            fork.hash(),
            &fork_frame,
            NOW + 20,
        )
        .is_err());
    let connection = store.connection.lock();
    connection
        .execute(
            "UPDATE directory_replica_chains
             SET quarantined = 0, quarantine_kind = NULL,
                 active_incident_digest = NULL
             WHERE producer = ?1",
            params![producer_id.as_slice()],
        )
        .unwrap();
    drop(connection);
    assert!(store.audit(NOW + 21).is_err());
    drop(store);
    assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 22).is_err());
}

#[test]
fn tampered_resolution_signature_fails_startup_audit() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0xa4; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let fork = block(&producer, 1, [0u8; 32], &object_b);
    let first_frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0xa5; 16],
    );
    let fork_frame = response_frame(
        &producer,
        vec![fork.clone()],
        false,
        1,
        fork.hash(),
        [0xa6; 16],
    );
    let producer_id = producer.public_key_bytes();
    let path = temp.path().join("directory.db");
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&first),
            std::slice::from_ref(&object_a),
            1,
            first.hash(),
            &first_frame,
            NOW + 20,
        )
        .unwrap();
    assert!(store
        .import_verified_page(
            producer_id,
            std::slice::from_ref(&fork),
            std::slice::from_ref(&object_b),
            1,
            fork.hash(),
            &fork_frame,
            NOW + 20,
        )
        .is_err());
    let incident_digest = store.incident_summaries(None, 1).unwrap().incidents[0].incident_digest;
    let tip = store.producer_tip(&producer_id).unwrap();
    let command = resolution_command(&local, incident_digest, &tip, [0xa7; 16], NOW + 21);
    let report = store.resolve_quarantine(&command, NOW + 21).unwrap();
    let connection = store.connection.lock();
    connection
        .execute(
            "UPDATE directory_replica_resolutions SET signature = ?2
             WHERE resolution_digest = ?1",
            params![report.resolution_digest.as_slice(), [0u8; 64].as_slice()],
        )
        .unwrap();
    drop(connection);
    assert!(store.audit(NOW + 22).is_err());
    drop(store);
    assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 23).is_err());
}
