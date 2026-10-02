// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn schema_v9_is_atomically_migrated_to_v10_certificate_import_history() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x08; 32]).unwrap();
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    drop(store);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_observation_certificate_imports;
             ALTER TABLE directory_replica_meta RENAME TO directory_replica_meta_v10;
             CREATE TABLE directory_replica_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32),
                 witness_policy_epoch INTEGER NOT NULL DEFAULT 0
                     CHECK (witness_policy_epoch >= 0),
                 witness_policy_head BLOB
                     CHECK (witness_policy_head IS NULL
                         OR length(witness_policy_head) = 32)
             );
             INSERT INTO directory_replica_meta
                 (singleton, schema_version, chain_id, local_node_id,
                  witness_policy_epoch, witness_policy_head)
             SELECT singleton, 9, chain_id, local_node_id,
                    witness_policy_epoch, witness_policy_head
             FROM directory_replica_meta_v10;
             DROP TABLE directory_replica_meta_v10;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(audit.imported_observation_certificates, 0);
    let connection = store.connection.lock();
    let (version, import_columns, import_table): (i64, i64, String) = connection
        .query_row(
            "SELECT m.schema_version,
                    (SELECT COUNT(*) FROM pragma_table_info('directory_replica_meta')
                     WHERE name IN (
                         'certificate_import_sequence',
                         'certificate_import_head'
                     )),
                    t.name
             FROM directory_replica_meta m
             JOIN sqlite_master t
               ON t.type = 'table'
              AND t.name = 'directory_observation_certificate_imports'
             WHERE m.singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(import_columns, 2);
    assert_eq!(import_table, "directory_observation_certificate_imports");
}

#[test]
fn portable_certificate_import_is_idempotent_and_restart_audited() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x09; 32]).unwrap();
    let observer = IdentityKeyPair::from_bytes(&[0x0a; 32]).unwrap();
    let witness_a = IdentityKeyPair::from_bytes(&[0x0b; 32]).unwrap();
    let witness_b = IdentityKeyPair::from_bytes(&[0x0c; 32]).unwrap();
    let verified_at = NOW + 50;
    let (frame, frame_sha256, trust_policy) = portable_observation_certificate_fixture(
        &observer,
        &[&witness_a, &witness_b],
        7,
        0x0d,
        verified_at,
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), verified_at).unwrap();
    let inserted = store
        .import_observation_certificate(&frame, &frame_sha256, &trust_policy, &local, verified_at)
        .unwrap();
    assert!(inserted.inserted);
    assert_eq!(inserted.import_sequence, 1);
    assert_eq!(inserted.retained_certificates, 1);
    assert_eq!(inserted.checkpoint_sequence, 7);

    let unchanged = store
        .import_observation_certificate(
            &frame,
            &frame_sha256,
            &trust_policy,
            &local,
            verified_at + 1,
        )
        .unwrap();
    assert!(!unchanged.inserted);
    assert_eq!(unchanged.import_sequence, inserted.import_sequence);
    assert_eq!(unchanged.import_digest, inserted.import_digest);
    assert_eq!(unchanged.verified_at, inserted.verified_at);
    let snapshot = store.status_snapshot().unwrap();
    assert_eq!(snapshot.imported_observation_certificates, 1);
    assert_eq!(snapshot.imported_observation_certificate_sequence, 1);
    assert_eq!(
        snapshot.imported_observation_certificate_head,
        inserted.import_digest
    );
    drop(store);

    let (reopened, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), verified_at + 2).unwrap();
    assert_eq!(audit.imported_observation_certificates, 1);
    assert_eq!(audit.imported_observation_certificate_sequence, 1);
    assert_eq!(
        audit.imported_observation_certificate_head,
        inserted.import_digest
    );
    assert_eq!(
        reopened
            .status_snapshot()
            .unwrap()
            .imported_observation_certificates,
        1
    );
}

#[test]
fn portable_certificate_import_rejects_rollback_conflict_and_policy_change() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x0e; 32]).unwrap();
    let observer = IdentityKeyPair::from_bytes(&[0x0f; 32]).unwrap();
    let witness_a = IdentityKeyPair::from_bytes(&[0x10; 32]).unwrap();
    let witness_b = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
    let verified_at = NOW + 50;
    let (frame, frame_sha256, trust_policy) = portable_observation_certificate_fixture(
        &observer,
        &[&witness_a, &witness_b],
        7,
        0x12,
        verified_at,
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), verified_at).unwrap();
    store
        .import_observation_certificate(&frame, &frame_sha256, &trust_policy, &local, verified_at)
        .unwrap();

    let weaker_policy = DirectoryObservationCertificateTrustPolicy::new(
        observer.public_key_bytes(),
        vec![witness_a.public_key_bytes(), witness_b.public_key_bytes()],
        1,
    )
    .unwrap();
    assert!(matches!(
        store.import_observation_certificate(
            &frame,
            &frame_sha256,
            &weaker_policy,
            &local,
            verified_at + 1,
        ),
        Err(DirectoryReplicaStoreError::Request(_))
    ));

    let (rollback_frame, rollback_sha256, rollback_policy) =
        portable_observation_certificate_fixture(
            &observer,
            &[&witness_a, &witness_b],
            6,
            0x13,
            verified_at + 1,
        );
    assert!(matches!(
        store.import_observation_certificate(
            &rollback_frame,
            &rollback_sha256,
            &rollback_policy,
            &local,
            verified_at + 1,
        ),
        Err(DirectoryReplicaStoreError::Request(_))
    ));

    let (conflict_frame, conflict_sha256, conflict_policy) =
        portable_observation_certificate_fixture(
            &observer,
            &[&witness_a, &witness_b],
            7,
            0x14,
            verified_at + 1,
        );
    assert!(matches!(
        store.import_observation_certificate(
            &conflict_frame,
            &conflict_sha256,
            &conflict_policy,
            &local,
            verified_at + 1,
        ),
        Err(DirectoryReplicaStoreError::Request(_))
    ));
    assert_eq!(
        store
            .status_snapshot()
            .unwrap()
            .imported_observation_certificates,
        1
    );
}

#[test]
fn tampered_or_deleted_certificate_import_fails_restart_audit() {
    let run_case = |delete_row: bool| {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x15; 32]).unwrap();
        let observer = IdentityKeyPair::from_bytes(&[0x16; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x17; 32]).unwrap();
        let verified_at = NOW + 50;
        let (frame, frame_sha256, trust_policy) =
            portable_observation_certificate_fixture(&observer, &[&witness], 8, 0x18, verified_at);
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), verified_at).unwrap();
        store
            .import_observation_certificate(
                &frame,
                &frame_sha256,
                &trust_policy,
                &local,
                verified_at,
            )
            .unwrap();
        {
            let connection = store.connection.lock();
            if delete_row {
                connection
                    .execute(
                        "DELETE FROM directory_observation_certificate_imports
                         WHERE import_sequence = 1",
                        [],
                    )
                    .unwrap();
            } else {
                let mut stored_frame: Vec<u8> = connection
                    .query_row(
                        "SELECT certificate_frame
                         FROM directory_observation_certificate_imports
                         WHERE import_sequence = 1",
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                let last = stored_frame.len() - 1;
                stored_frame[last] ^= 1;
                connection
                    .execute(
                        "UPDATE directory_observation_certificate_imports
                         SET certificate_frame = ?1 WHERE import_sequence = 1",
                        params![stored_frame],
                    )
                    .unwrap();
            }
        }
        assert!(store.audit(verified_at + 1).is_err());
        drop(store);
        assert!(
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), verified_at + 1).is_err()
        );
    };
    run_case(false);
    run_case(true);
}

#[test]
fn schema_v7_adds_policy_anchor_evidence_tables_atomically() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x2b; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V7],
        )
        .unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_observation_policy_anchor_receipts;
             DROP TABLE directory_observation_remote_policy_anchors;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(audit.observation_witness_policy_anchor_receipts, 0);
    assert_eq!(audit.observation_witness_remote_policy_anchors, 0);
    let connection = store.connection.lock();
    let version: i64 = connection
        .query_row(
            "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let anchor_tables: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND name IN (
                   'directory_observation_remote_policy_anchors',
                   'directory_observation_policy_anchor_receipts'
               )",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let receipt_index: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'index'
               AND name = 'directory_policy_anchor_receipts_by_epoch'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(anchor_tables, 2);
    assert_eq!(receipt_index, 1);
}
