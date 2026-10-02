// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn producer_replicas_are_isolated_idempotent_and_reopen_cleanly() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let first = block(&producer, 1, [0u8; 32], &object);
    let frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x41; 16],
    );
    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    assert_eq!(audit, DirectoryReplicaAudit::default());
    let imported = store
        .import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&first),
            std::slice::from_ref(&object),
            1,
            first.hash(),
            &frame,
            NOW + 20,
        )
        .unwrap();
    assert_eq!(imported.blocks_inserted, 1);
    let repeated = store
        .import_verified_page(
            producer.public_key_bytes(),
            std::slice::from_ref(&first),
            std::slice::from_ref(&object),
            1,
            first.hash(),
            &frame,
            NOW + 20,
        )
        .unwrap();
    assert_eq!(repeated.blocks_already_present, 1);
    let snapshot = store.status_snapshot().unwrap();
    assert_eq!(snapshot.producers, 1);
    assert_eq!(snapshot.quarantined_producers, 0);
    assert_eq!(snapshot.blocks, 1);
    assert_eq!(snapshot.commitments, 1);
    assert_eq!(snapshot.incidents, 0);
    assert_eq!(snapshot.producer_snapshots.len(), 1);
    assert_eq!(
        snapshot.producer_snapshots[0].producer,
        producer.public_key_bytes()
    );
    assert_eq!(snapshot.producer_snapshots[0].tip_height, 1);
    drop(store);
    let (_, reopened) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(reopened.producers, 1);
    assert_eq!(reopened.blocks, 1);
    assert_eq!(reopened.commitments, 1);
}

#[test]
fn carrier_reported_tip_cannot_quarantine_or_mutate_retained_prefix() {
    // [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] A carrier may lie
    // about terminal metadata, but it cannot turn that report into durable
    // producer rollback, fork, or empty-gap evidence.
    let temp = TempDir::new().unwrap();
    let local = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x42; 32]).unwrap();
    let carrier = IdentityKeyPair::from_bytes(&[0x43; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x44; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let first = block(&producer, 1, [0u8; 32], &object);
    let (store, _) = DirectoryReplicaStore::open(
        temp.path().join("directory.db"),
        local.public_key_bytes(),
        NOW + 20,
    )
    .unwrap();
    import_replica_block(&store, &producer, &object, &first, [0x45; 16]);

    let attacks = [
        (0, [0u8; 32], [0x46; 16]),
        (1, [0x47; 32], [0x48; 16]),
        (2, [0x49; 32], [0x4a; 16]),
    ];
    for (tip_height, tip_hash, request_id) in attacks {
        let frame = carrier_response_frame(
            &producer,
            &carrier,
            Vec::new(),
            false,
            tip_height,
            tip_hash,
            request_id,
        );
        assert!(matches!(
            store.import_verified_page(
                producer.public_key_bytes(),
                &[],
                &[],
                tip_height,
                tip_hash,
                &frame,
                NOW + 20,
            ),
            Err(DirectoryReplicaStoreError::Integrity(_))
        ));
        let tip = store.producer_tip(&producer.public_key_bytes()).unwrap();
        assert_eq!(tip.tip_height, 1);
        assert_eq!(tip.tip_hash, first.hash());
        assert!(!tip.quarantined);
    }
    assert!(store
        .incident_summaries(None, 1)
        .unwrap()
        .incidents
        .is_empty());
    let audit = store.audit(NOW + 21).unwrap();
    assert_eq!(audit.incidents, 0);
    assert_eq!(audit.quarantined_producers, 0);
}
