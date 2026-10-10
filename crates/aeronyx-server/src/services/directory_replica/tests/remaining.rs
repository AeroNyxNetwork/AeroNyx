// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn producer_scoped_export_fails_closed_without_cross_producer_coupling() {
    // [DIRECTORY-PRODUCER-AUDIT 2026-07-22 by Codex] Prove that public
    // export re-verifies every target object while an unrelated namespace
    // cannot turn one producer's availability into a global failure.
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x21; 32]).unwrap();
    let target = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
    let unrelated = IdentityKeyPair::from_bytes(&[0x23; 32]).unwrap();
    let target_subject = IdentityKeyPair::from_bytes(&[0x24; 32]).unwrap();
    let unrelated_subject = IdentityKeyPair::from_bytes(&[0x25; 32]).unwrap();
    let target_object = descriptor(&target_subject, 1);
    let unrelated_object = descriptor(&unrelated_subject, 1);
    let target_block = block(&target, 1, [0u8; 32], &target_object);
    let unrelated_block = block(&unrelated, 1, [0u8; 32], &unrelated_object);
    let target_frame = response_frame(
        &target,
        vec![target_block.clone()],
        false,
        1,
        target_block.hash(),
        [0x26; 16],
    );
    let unrelated_frame = response_frame(
        &unrelated,
        vec![unrelated_block.clone()],
        false,
        1,
        unrelated_block.hash(),
        [0x27; 16],
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    store
        .import_verified_mirror_page(
            target.public_key_bytes(),
            11,
            4,
            std::slice::from_ref(&target_block),
            std::slice::from_ref(&target_object),
            1,
            target_block.hash(),
            &target_frame,
            NOW + 20,
        )
        .unwrap();
    store
        .import_verified_mirror_page(
            unrelated.public_key_bytes(),
            12,
            4,
            std::slice::from_ref(&unrelated_block),
            std::slice::from_ref(&unrelated_object),
            1,
            unrelated_block.hash(),
            &unrelated_frame,
            NOW + 20,
        )
        .unwrap();

    let external = Connection::open(&path).unwrap();
    external
        .execute(
            "UPDATE directory_replica_blocks SET block_hash = ?2
             WHERE producer = ?1 AND height = 1",
            params![
                unrelated.public_key_bytes().as_slice(),
                [0xff_u8; 32].as_slice()
            ],
        )
        .unwrap();
    let page = store
        .audited_mirror_evidence_page(&target.public_key_bytes(), 1, 1, NOW + 21)
        .unwrap();
    assert_eq!(page.blocks, vec![target_block.clone()]);
    assert!(matches!(
        store.audit(NOW + 21),
        Err(DirectoryReplicaStoreError::Integrity(_))
    ));

    external
        .execute(
            "UPDATE directory_replica_blocks SET block_hash = ?2
             WHERE producer = ?1 AND height = 1",
            params![
                unrelated.public_key_bytes().as_slice(),
                unrelated_block.hash().as_slice()
            ],
        )
        .unwrap();
    assert!(store.audit(NOW + 21).is_ok());
    external
        .execute(
            "UPDATE directory_replica_blocks SET block_hash = ?2
             WHERE producer = ?1 AND height = 1",
            params![
                target.public_key_bytes().as_slice(),
                [0xee_u8; 32].as_slice()
            ],
        )
        .unwrap();
    assert!(matches!(
        store.audited_mirror_evidence_page(&target.public_key_bytes(), 1, 1, NOW + 21),
        Err(DirectoryReplicaStoreError::Integrity(_))
    ));
    assert!(matches!(
        store.audited_mirror_evidence_descriptor_objects(
            &target.public_key_bytes(),
            &[target_block.commitments[0].descriptor_hash],
            NOW + 21,
        ),
        Err(DirectoryReplicaStoreError::Integrity(_))
    ));
}

#[test]
fn concurrent_valid_append_does_not_false_fail_streaming_audit() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
    let first_subject = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
    let second_subject = IdentityKeyPair::from_bytes(&[0x34; 32]).unwrap();
    let first_object = descriptor(&first_subject, 1);
    let first = block(&producer, 1, [0u8; 32], &first_object);
    let second_object = descriptor(&second_subject, 1);
    let second = block(&producer, 2, first.hash(), &second_object);
    let (reader, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&reader, &producer, &first_object, &first, [0x35; 16]);
    // A separately opened host-local store has an independent mutex and
    // models a live sync writer racing a CLI/startup audit of the same WAL.
    let (writer, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    let audit_reached = Arc::new(Barrier::new(2));
    let writer_finished = Arc::new(Barrier::new(2));
    let observer_reached = Arc::clone(&audit_reached);
    let observer_finished = Arc::clone(&writer_finished);
    let observer = observe_directory_replica_audit(move |event| {
        if event == DirectoryReplicaAuditTestEvent::BlockVerified(1) {
            observer_reached.wait();
            observer_finished.wait();
        }
    });

    let audit = std::thread::scope(|scope| {
        let writer = scope.spawn(|| {
            audit_reached.wait();
            import_replica_block(&writer, &producer, &second_object, &second, [0x36; 16]);
            writer_finished.wait();
        });
        let audit = reader.audit(NOW + 21);
        writer.join().unwrap();
        audit
    })
    .unwrap();
    drop(observer);

    // [DIRECTORY-AUDIT-SNAPSHOT 2026-08-31 by Codex] The in-flight audit
    // is a truthful snapshot of the one-block prefix, while a later audit
    // observes the atomically appended second block without a false
    // orphan-index failure.
    assert_eq!(audit.blocks, 1);
    assert_eq!(audit.commitments, 1);
    let current = reader.audit(NOW + 21).unwrap();
    assert_eq!(current.blocks, 2);
    assert_eq!(current.commitments, 2);
}

#[test]
fn cross_snapshot_index_swap_cannot_false_pass_streaming_audit() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x42; 32]).unwrap();
    let original_subject = IdentityKeyPair::from_bytes(&[0x43; 32]).unwrap();
    let extra_subject = IdentityKeyPair::from_bytes(&[0x44; 32]).unwrap();
    let replacement_subject = IdentityKeyPair::from_bytes(&[0x45; 32]).unwrap();
    let original_object = descriptor(&original_subject, 1);
    let extra_object = descriptor(&extra_subject, 1);
    let replacement_object = descriptor(&replacement_subject, 1);
    let original_commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&original_object).unwrap();
    let extra_commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&extra_object).unwrap();
    let replacement_commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&replacement_object).unwrap();
    let first = block(&producer, 1, [0u8; 32], &original_object);
    let producer_id = producer.public_key_bytes();
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer, &original_object, &first, [0x46; 16]);
    {
        let connection = store.connection.lock();
        connection
            .execute(
                "INSERT INTO directory_replica_descriptor_objects
                    (producer, descriptor_hash, node_id, sequence_le, descriptor_blob)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    producer_id.as_slice(),
                    extra_commitment.descriptor_hash.as_slice(),
                    extra_commitment.node_id.as_slice(),
                    extra_commitment.sequence.to_le_bytes().as_slice(),
                    encode_descriptor_object(&extra_object).unwrap()
                ],
            )
            .unwrap();
    }

    let audit_reached = Arc::new(Barrier::new(2));
    let writer_finished = Arc::new(Barrier::new(2));
    let observer_reached = Arc::clone(&audit_reached);
    let observer_finished = Arc::clone(&writer_finished);
    let observer = observe_directory_replica_audit(move |event| {
        if event == DirectoryReplicaAuditTestEvent::BlockVerified(1) {
            observer_reached.wait();
            observer_finished.wait();
        }
    });

    let audit = std::thread::scope(|scope| {
        let writer_path = path.clone();
        let writer = scope.spawn(move || {
            audit_reached.wait();
            let mut connection = Connection::open(writer_path).unwrap();
            connection
                .pragma_update(None, "journal_mode", "WAL")
                .unwrap();
            connection
                .pragma_update(None, "foreign_keys", true)
                .unwrap();
            let transaction = connection
                .transaction_with_behavior(TransactionBehavior::Immediate)
                .unwrap();
            transaction
                .execute(
                    "DELETE FROM directory_replica_commitments
                     WHERE producer = ?1 AND commitment_hash = ?2",
                    params![
                        producer_id.as_slice(),
                        original_commitment.hash().as_slice()
                    ],
                )
                .unwrap();
            transaction
                .execute(
                    "DELETE FROM directory_replica_descriptor_objects
                     WHERE producer = ?1 AND descriptor_hash IN (?2, ?3)",
                    params![
                        producer_id.as_slice(),
                        original_commitment.descriptor_hash.as_slice(),
                        extra_commitment.descriptor_hash.as_slice()
                    ],
                )
                .unwrap();
            transaction
                .execute(
                    "INSERT INTO directory_replica_descriptor_objects
                        (producer, descriptor_hash, node_id, sequence_le, descriptor_blob)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![
                        producer_id.as_slice(),
                        replacement_commitment.descriptor_hash.as_slice(),
                        replacement_commitment.node_id.as_slice(),
                        replacement_commitment.sequence.to_le_bytes().as_slice(),
                        encode_descriptor_object(&replacement_object).unwrap()
                    ],
                )
                .unwrap();
            transaction
                .execute(
                    "INSERT INTO directory_replica_commitments
                        (producer, commitment_hash, node_id, sequence_le,
                         descriptor_hash, block_height)
                     VALUES (?1, ?2, ?3, ?4, ?5, 1)",
                    params![
                        producer_id.as_slice(),
                        replacement_commitment.hash().as_slice(),
                        replacement_commitment.node_id.as_slice(),
                        replacement_commitment.sequence.to_le_bytes().as_slice(),
                        replacement_commitment.descriptor_hash.as_slice()
                    ],
                )
                .unwrap();
            transaction.commit().unwrap();
            writer_finished.wait();
        });
        let audit = store.audit(NOW + 21);
        writer.join().unwrap();
        audit
    });
    drop(observer);

    assert!(matches!(
        audit,
        Err(DirectoryReplicaStoreError::Integrity(message))
            if message == "replica contains orphaned commitment or descriptor indexes"
    ));
    assert!(matches!(
        store.audit(NOW + 21),
        Err(DirectoryReplicaStoreError::Integrity(message))
            if message == "replica block 1 commitment index mismatch"
    ));
}

#[test]
fn oversized_persisted_blobs_are_rejected_before_materialization() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&object).unwrap();
    let first = block(&producer, 1, [0u8; 32], &object);
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer, &object, &first, [0x54; 16]);

    let block_materializations = Arc::new(AtomicUsize::new(0));
    let descriptor_materializations = Arc::new(AtomicUsize::new(0));
    let observed_blocks = Arc::clone(&block_materializations);
    let observed_descriptors = Arc::clone(&descriptor_materializations);
    let observer = observe_directory_replica_audit(move |event| match event {
        DirectoryReplicaAuditTestEvent::BlobMaterialized(PersistedReplicaBlobKind::Block) => {
            observed_blocks.fetch_add(1, Ordering::Relaxed);
        }
        DirectoryReplicaAuditTestEvent::BlobMaterialized(PersistedReplicaBlobKind::Descriptor) => {
            observed_descriptors.fetch_add(1, Ordering::Relaxed);
        }
        DirectoryReplicaAuditTestEvent::BlockVerified(_) => {}
    });

    {
        let connection = store.connection.lock();
        connection
            .execute(
                "UPDATE directory_replica_blocks
                 SET block_blob = zeroblob(?3)
                 WHERE producer = ?1 AND height = ?2",
                params![
                    producer.public_key_bytes().as_slice(),
                    1i64,
                    u64_to_i64(MAX_DIRECTORY_BLOCK_BYTES + 1, "test block blob size").unwrap()
                ],
            )
            .unwrap();
    }
    assert!(matches!(
        store.audit(NOW + 21),
        Err(DirectoryReplicaStoreError::Codec(message))
            if message == "replica block exceeds its byte limit"
    ));
    assert_eq!(block_materializations.load(Ordering::Relaxed), 0);
    assert_eq!(descriptor_materializations.load(Ordering::Relaxed), 0);

    {
        let connection = store.connection.lock();
        connection
            .execute(
                "UPDATE directory_replica_blocks SET block_blob = ?3
                 WHERE producer = ?1 AND height = ?2",
                params![
                    producer.public_key_bytes().as_slice(),
                    1i64,
                    encode_block(&first).unwrap()
                ],
            )
            .unwrap();
        connection
            .execute(
                "UPDATE directory_replica_descriptor_objects
                 SET descriptor_blob = zeroblob(?3)
                 WHERE producer = ?1 AND descriptor_hash = ?2",
                params![
                    producer.public_key_bytes().as_slice(),
                    commitment.descriptor_hash.as_slice(),
                    u64_to_i64(
                        MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES + 1,
                        "test descriptor blob size"
                    )
                    .unwrap()
                ],
            )
            .unwrap();
    }
    assert!(matches!(
        store.audit(NOW + 21),
        Err(DirectoryReplicaStoreError::Codec(message))
            if message == "replica descriptor object exceeds its byte limit"
    ));
    assert_eq!(block_materializations.load(Ordering::Relaxed), 1);
    assert_eq!(descriptor_materializations.load(Ordering::Relaxed), 0);
    drop(observer);
}

#[test]
fn audited_carrier_export_is_bounded_exact_and_importable() {
    let source_temp = TempDir::new().unwrap();
    let receiver_temp = TempDir::new().unwrap();
    let mirror_receiver_temp = TempDir::new().unwrap();
    let invalid_receiver_temp = TempDir::new().unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x12; 32]).unwrap();
    let receiver = IdentityKeyPair::from_bytes(&[0x13; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x14; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x15; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let replica_block = block(&producer, 1, [0u8; 32], &object);
    let (source, _) = DirectoryReplicaStore::open(
        source_temp.path().join("directory.db"),
        carrier.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    import_replica_block(&source, &producer, &object, &replica_block, [0x16; 16]);

    let page = source
        .audited_evidence_page(&producer.public_key_bytes(), 1, 1, NOW + 21)
        .unwrap();
    assert_eq!(page.blocks, vec![replica_block.clone()]);
    assert_eq!(page.tip_height, 1);
    assert_eq!(page.tip_hash, replica_block.hash());
    assert!(matches!(
        source.audited_evidence_page(&producer.public_key_bytes(), 3, 1, NOW + 21),
        Err(DirectoryReplicaStoreError::RangeNotRetained {
            from_height: 3,
            tip_height: 1,
        })
    ));
    let descriptor_hash = replica_block.commitments[0].descriptor_hash;
    assert_eq!(
        source
            .audited_evidence_descriptor_objects(
                &producer.public_key_bytes(),
                &[descriptor_hash],
                NOW + 21,
            )
            .unwrap(),
        Some(vec![object.clone()])
    );
    assert!(source
        .audited_evidence_descriptor_objects(&producer.public_key_bytes(), &[[0x17; 32]], NOW + 21,)
        .unwrap()
        .is_none());
    let proof = source
        .audited_evidence_descriptor_inclusion_proof(
            &producer.public_key_bytes(),
            &descriptor_hash,
            &replica_block.hash(),
            NOW + 21,
        )
        .unwrap()
        .unwrap();
    proof
        .verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &producer.public_key_bytes(),
            &replica_block.hash(),
            NOW + 21,
        )
        .unwrap();
    assert_eq!(proof.commitment.descriptor_hash, descriptor_hash);
    assert!(source
        .audited_evidence_descriptor_inclusion_proof(
            &producer.public_key_bytes(),
            &descriptor_hash,
            &[0x17; 32],
            NOW + 21,
        )
        .unwrap()
        .is_none());
    assert!(source
        .audited_evidence_descriptor_inclusion_proof(
            &producer.public_key_bytes(),
            &[0x17; 32],
            &replica_block.hash(),
            NOW + 21,
        )
        .unwrap()
        .is_none());

    let frame = carrier_response_frame(
        &producer,
        &carrier,
        page.blocks.clone(),
        false,
        page.tip_height,
        page.tip_hash,
        [0x18; 16],
    );
    let (destination, _) = DirectoryReplicaStore::open(
        receiver_temp.path().join("directory.db"),
        receiver.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    let imported = destination
        .import_verified_page(
            producer.public_key_bytes(),
            &page.blocks,
            std::slice::from_ref(&object),
            page.tip_height,
            page.tip_hash,
            &frame,
            NOW + 20,
        )
        .unwrap();
    assert_eq!(imported.blocks_inserted, 1);

    let (mirror_destination, _) = DirectoryReplicaStore::open(
        mirror_receiver_temp.path().join("directory.db"),
        receiver.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    let mirror_imported = mirror_destination
        .import_verified_mirror_page(
            producer.public_key_bytes(),
            1,
            4,
            &page.blocks,
            std::slice::from_ref(&object),
            page.tip_height,
            page.tip_hash,
            &frame,
            NOW + 20,
        )
        .unwrap();
    assert_eq!(mirror_imported.blocks_inserted, 1);
    assert_eq!(
        mirror_destination.mirror_producer_ids().unwrap(),
        vec![producer.public_key_bytes()]
    );
    let mirror_proof = mirror_destination
        .audited_mirror_evidence_descriptor_inclusion_proof(
            &producer.public_key_bytes(),
            &descriptor_hash,
            &replica_block.hash(),
            NOW + 21,
        )
        .unwrap()
        .unwrap();
    assert_eq!(mirror_proof, proof);
    assert_eq!(
        mirror_destination
            .verify_retained_carrier_page(
                producer.public_key_bytes(),
                1,
                &page.blocks,
                std::slice::from_ref(&object),
                page.tip_height,
                page.tip_hash,
                &frame,
                NOW + 20,
            )
            .unwrap(),
        (1, 1)
    );
    assert_eq!(
        mirror_destination
            .producer_tip(&producer.public_key_bytes())
            .unwrap()
            .tip_height,
        1
    );

    let mut producer_tampered_block = replica_block.clone();
    producer_tampered_block.header.timestamp =
        producer_tampered_block.header.timestamp.saturating_add(1);
    let producer_tampered_hash = producer_tampered_block.hash();
    let producer_tampered_frame = carrier_response_frame(
        &producer,
        &carrier,
        vec![producer_tampered_block.clone()],
        false,
        1,
        producer_tampered_hash,
        [0x19; 16],
    );
    let (invalid_destination, _) = DirectoryReplicaStore::open(
        invalid_receiver_temp.path().join("directory.db"),
        receiver.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    assert!(matches!(
        invalid_destination.import_verified_mirror_page(
            producer.public_key_bytes(),
            1,
            4,
            &[producer_tampered_block],
            std::slice::from_ref(&object),
            1,
            producer_tampered_hash,
            &producer_tampered_frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Block(_))
    ));

    let mut tampered = frame;
    let last = tampered.len() - 1;
    tampered[last] ^= 1;
    assert!(verify_incident_response_evidence(&tampered, &producer.public_key_bytes()).is_err());
}

#[test]
fn carrier_duplicate_height_with_invalid_signature_is_atomic_validation_failure() {
    // [DIRECTORY-CONFLICT-VERIFICATION 2026-09-01 by Codex] A carrier must
    // not turn an unverified duplicate-height payload into durable producer
    // blame after an earlier valid block in the same page was staged.
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x69; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x6a; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x6b; 32]).unwrap();
    let subject_a = IdentityKeyPair::from_bytes(&[0x6c; 32]).unwrap();
    let subject_b = IdentityKeyPair::from_bytes(&[0x6d; 32]).unwrap();
    let subject_c = IdentityKeyPair::from_bytes(&[0x6e; 32]).unwrap();
    let object_a = descriptor(&subject_a, 1);
    let object_b = descriptor(&subject_b, 1);
    let object_c = descriptor(&subject_c, 1);
    let first = block(&producer, 1, [0u8; 32], &object_a);
    let second = block(&producer, 2, first.hash(), &object_b);
    let mut invalid_fork = block(&producer, 2, first.hash(), &object_c);
    invalid_fork.producer_signature[0] ^= 1;
    let frame = carrier_response_frame(
        &producer,
        &carrier,
        vec![second.clone(), invalid_fork.clone()],
        false,
        2,
        invalid_fork.hash(),
        [0x6f; 16],
    );
    let path = temp.path().join("directory.db");
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    import_replica_block(&store, &producer, &object_a, &first, [0x70; 16]);
    let retained_before = store.producer_tip(&producer.public_key_bytes()).unwrap();

    assert!(matches!(
        store.import_verified_page(
            producer.public_key_bytes(),
            &[second, invalid_fork.clone()],
            &[object_b, object_c],
            2,
            invalid_fork.hash(),
            &frame,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Block(_))
    ));
    assert_eq!(
        store.producer_tip(&producer.public_key_bytes()).unwrap(),
        retained_before
    );
    assert!(store
        .incident_summaries(None, 1)
        .unwrap()
        .incidents
        .is_empty());
    let audit = store.audit(NOW + 21).unwrap();
    assert_eq!(audit.blocks, 1);
    assert_eq!(audit.commitments, 1);
    assert_eq!(audit.incidents, 0);
    assert_eq!(audit.quarantined_producers, 0);
}

#[test]
fn malformed_or_unrelated_objects_are_rejected_before_sqlite_changes() {
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
    let unrelated = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let wrong = descriptor(&unrelated, 1);
    let first = block(&producer, 1, [0u8; 32], &object);
    let frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x65; 16],
    );
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    assert!(store
        .import_verified_page(
            producer.public_key_bytes(),
            &[first],
            &[wrong],
            1,
            frame_tip_hash(&frame),
            &frame,
            NOW + 20,
        )
        .is_err());
    assert_eq!(
        store
            .producer_tip(&producer.public_key_bytes())
            .unwrap()
            .tip_height,
        0
    );
}

// [PARALLEL-DIRECTORY-AUDIT 2026-10-10 by Claude] Long enough that the replica
// audit verifies batches on several threads.
const LONG_REPLICA_BLOCKS: u64 = 64;

fn long_replica_chain(path: &std::path::Path, local: &IdentityKeyPair, producer: &IdentityKeyPair) {
    let subject = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(path, local.public_key_bytes(), NOW + 20).unwrap();
    let mut previous = [0u8; 32];
    for height in 1..=LONG_REPLICA_BLOCKS {
        let object = descriptor(&subject, height);
        let replica_block = block(producer, height, previous, &object);
        let request_id = [u8::try_from(height).unwrap(); 16];
        import_replica_block(&store, producer, &object, &replica_block, request_id);
        previous = replica_block.hash();
    }
}

fn flip_last_replica_byte(path: &std::path::Path, table: &str, blob: &str, filter: &str) {
    let connection = Connection::open(path).unwrap();
    let mut bytes: Vec<u8> = connection
        .query_row(
            &format!("SELECT {blob} FROM {table} WHERE {filter}"),
            [],
            |row| row.get(0),
        )
        .unwrap();
    *bytes.last_mut().unwrap() ^= 0x01;
    let changed = connection
        .execute(
            &format!("UPDATE {table} SET {blob} = ?1 WHERE {filter}"),
            [bytes],
        )
        .unwrap();
    assert_eq!(changed, 1);
}

#[test]
fn long_replica_audit_runs_in_parallel_and_matches_the_chain() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
    long_replica_chain(&path, &local, &producer);
    let (store, opened) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(opened.blocks, LONG_REPLICA_BLOCKS);
    assert_eq!(opened.commitments, LONG_REPLICA_BLOCKS);
    assert_eq!(store.audit(NOW + 21).unwrap(), opened);
}

#[test]
fn long_replica_block_signature_tampering_deep_in_the_chain_fails_closed() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x65; 32]).unwrap();
    long_replica_chain(&path, &local, &producer);
    flip_last_replica_byte(
        &path,
        "directory_replica_blocks",
        "block_blob",
        "height = 41",
    );
    assert!(matches!(
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21),
        Err(DirectoryReplicaStoreError::Block(
            DirectoryCommitmentValidationError::InvalidSignature
        ))
    ));
}

#[test]
fn long_replica_descriptor_signature_tampering_deep_in_the_chain_fails_closed() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x66; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x67; 32]).unwrap();
    long_replica_chain(&path, &local, &producer);
    flip_last_replica_byte(
        &path,
        "directory_replica_descriptor_objects",
        "descriptor_blob",
        "descriptor_hash = (SELECT descriptor_hash FROM directory_replica_commitments
                            WHERE block_height = 40)",
    );
    assert!(matches!(
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21),
        Err(DirectoryReplicaStoreError::Descriptor(_))
    ));
}
