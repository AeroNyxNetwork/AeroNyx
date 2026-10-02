// Split from crates/aeronyx-server/src/services/directory_replica.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn full_node_mirror_runtime_tracks_only_aggregate_recovery_outcomes() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    // [MIRROR-CATCHUP 2026-07-24 by Codex] One producer converges, one
    // advances under the bounded budget, and one fails after prior pages.
    runtime.record_full_node_mirror_catch_up_round(5, 3, 1, 1, 1, 7, 11, NOW + 1);
    runtime.record_full_node_mirror_carrier_selection(5, 4, 3, 2, 1, 2, 2, 2, 0, 2, 2);
    runtime.record_full_node_mirror_recovery(true, NOW + 2);
    runtime.record_full_node_mirror_recovery(false, NOW + 3);

    let snapshot = runtime.full_node_mirror_snapshot();
    assert_eq!(snapshot.rounds, 1);
    assert_eq!(snapshot.last_round_succeeded, 2);
    assert_eq!(snapshot.last_round_failed, 1);
    assert_eq!(snapshot.last_round_converged, 1);
    assert_eq!(snapshot.last_round_catching_up, 1);
    assert_eq!(snapshot.last_round_pages_succeeded, 7);
    assert_eq!(snapshot.last_round_requests_sent, 11);
    assert_eq!(snapshot.pages_succeeded, 7);
    assert_eq!(snapshot.requests_sent, 11);
    assert_eq!(snapshot.attempts_failed, 1);
    assert_eq!(snapshot.recovery_attempts, 2);
    assert_eq!(snapshot.recovery_succeeded, 1);
    assert_eq!(snapshot.recovery_failed, 1);
    assert_eq!(snapshot.last_recovery_carrier_candidates, 5);
    assert_eq!(snapshot.last_recovery_routeable_carrier_candidates, 4);
    assert_eq!(snapshot.last_recovery_explicit_capability_candidates, 3);
    assert_eq!(
        snapshot.last_recovery_unadvertised_compatibility_candidates,
        2
    );
    assert_eq!(snapshot.last_recovery_capability_cached_unavailable, 1);
    assert_eq!(snapshot.last_recovery_carriers_selected, 2);
    assert_eq!(snapshot.last_recovery_routeable_carriers_selected, 2);
    assert_eq!(snapshot.last_recovery_explicit_capability_selected, 2);
    assert_eq!(
        snapshot.last_recovery_unadvertised_compatibility_selected,
        0
    );
    assert_eq!(snapshot.last_recovery_selected_region_hints, 2);
    assert_eq!(snapshot.last_recovery_distinct_region_hints, 2);
    assert_eq!(snapshot.last_recovery_attempt_at, Some(NOW + 3));
    assert_eq!(snapshot.last_recovery_success_at, Some(NOW + 2));
    assert_eq!(snapshot.last_recovery_failure_at, Some(NOW + 3));
}

#[test]
fn legacy_full_node_mirror_round_recording_remains_compatible() {
    let runtime = DirectoryReplicaSyncRuntime::default();
    runtime.record_full_node_mirror_round(4, 2, 1, NOW + 1);

    let snapshot = runtime.full_node_mirror_snapshot();
    assert_eq!(snapshot.last_round_succeeded, 1);
    assert_eq!(snapshot.last_round_failed, 1);
    assert_eq!(snapshot.last_round_converged, 1);
    assert_eq!(snapshot.last_round_catching_up, 0);
    assert_eq!(snapshot.pages_succeeded, 1);
}

#[test]
fn full_node_mirror_registry_is_bounded_and_promotes_only_by_operator_pin() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x01; 32]).unwrap();
    let mirror_a = IdentityKeyPair::from_bytes(&[0x02; 32]).unwrap();
    let mirror_b = IdentityKeyPair::from_bytes(&[0x03; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x04; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let block_a = block(&mirror_a, 1, [0u8; 32], &object);
    let block_b = block(&mirror_b, 1, [0u8; 32], &object);
    let frame_a = response_frame(
        &mirror_a,
        vec![block_a.clone()],
        false,
        1,
        block_a.hash(),
        [0x05; 16],
    );
    let frame_b = response_frame(
        &mirror_b,
        vec![block_b.clone()],
        false,
        1,
        block_b.hash(),
        [0x06; 16],
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();

    store
        .import_verified_mirror_page(
            mirror_a.public_key_bytes(),
            7,
            1,
            std::slice::from_ref(&block_a),
            std::slice::from_ref(&object),
            1,
            block_a.hash(),
            &frame_a,
            NOW + 20,
        )
        .unwrap();
    assert_eq!(
        store.mirror_producer_ids().unwrap(),
        vec![mirror_a.public_key_bytes()]
    );
    assert_eq!(store.status_snapshot().unwrap().mirror_producers, 1);
    assert!(matches!(
        store.import_verified_mirror_page(
            mirror_b.public_key_bytes(),
            8,
            1,
            std::slice::from_ref(&block_b),
            std::slice::from_ref(&object),
            1,
            block_b.hash(),
            &frame_b,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::MirrorCapacity)
    ));
    assert_eq!(
        store
            .producer_tip(&mirror_b.public_key_bytes())
            .unwrap()
            .tip_height,
        0
    );
    store
        .import_verified_mirror_page(
            mirror_b.public_key_bytes(),
            8,
            2,
            std::slice::from_ref(&block_b),
            std::slice::from_ref(&object),
            1,
            block_b.hash(),
            &frame_b,
            NOW + 20,
        )
        .unwrap();
    assert!(matches!(
        store.ensure_mirror_capacity(1),
        Err(DirectoryReplicaStoreError::MirrorCapacity)
    ));
    store.ensure_mirror_capacity(2).unwrap();

    let public_page = store
        .audited_mirror_evidence_page(&mirror_a.public_key_bytes(), 1, 1, NOW + 21)
        .unwrap();
    assert_eq!(public_page.blocks, vec![block_a.clone()]);
    let object_hash = block_a.commitments[0].descriptor_hash;
    assert_eq!(
        store
            .audited_mirror_evidence_descriptor_objects(
                &mirror_a.public_key_bytes(),
                &[object_hash],
                NOW + 21,
            )
            .unwrap(),
        Some(vec![object.clone()])
    );

    assert_eq!(
        store
            .promote_pinned_producers(&[mirror_a.public_key_bytes(), mirror_b.public_key_bytes(),])
            .unwrap(),
        2
    );
    assert!(store.mirror_producer_ids().unwrap().is_empty());
    assert_eq!(store.audit(NOW + 21).unwrap().mirror_producers, 0);
    assert!(matches!(
        store.audited_mirror_evidence_page(&mirror_a.public_key_bytes(), 1, 1, NOW + 21,),
        Err(DirectoryReplicaStoreError::MirrorNotRetained)
    ));
    assert!(matches!(
        store.import_verified_mirror_page(
            mirror_a.public_key_bytes(),
            9,
            1,
            std::slice::from_ref(&block_a),
            std::slice::from_ref(&object),
            1,
            block_a.hash(),
            &frame_a,
            NOW + 20,
        ),
        Err(DirectoryReplicaStoreError::Request(_))
    ));
}

#[test]
fn retained_mirror_cursor_survives_restart_with_latest_descriptor_sequence() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
    let producer = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
    let subject = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
    let object = descriptor(&subject, 1);
    let first = block(&producer, 1, [0u8; 32], &object);
    let frame = response_frame(
        &producer,
        vec![first.clone()],
        false,
        1,
        first.hash(),
        [0x34; 16],
    );
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    for descriptor_sequence in [7, 11] {
        store
            .import_verified_mirror_page(
                producer.public_key_bytes(),
                descriptor_sequence,
                4,
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
    }
    drop(store);

    let (reopened, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 3_700).unwrap();
    assert_eq!(audit.mirror_producers, 1);
    assert_eq!(
        reopened.retained_mirror_cursors().unwrap(),
        vec![DirectoryRetainedMirrorCursor {
            producer: producer.public_key_bytes(),
            descriptor_sequence: 11,
        }]
    );
}

#[test]
fn schema_v8_is_atomically_migrated_to_v9_mirror_registry() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x07; 32]).unwrap();
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    drop(store);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch("DROP TABLE directory_replica_mirror_producers;")
        .unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V8],
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(audit.mirror_producers, 0);
    let connection = store.connection.lock();
    let version: i64 = connection
        .query_row(
            "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
}

#[test]
fn schema_v10_is_atomically_migrated_to_v11_route_domain_history() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x10; 32]).unwrap();
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    drop(store);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_route_domain_policies;
             ALTER TABLE directory_replica_meta RENAME TO directory_replica_meta_v11;
             CREATE TABLE directory_replica_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32),
                 witness_policy_epoch INTEGER NOT NULL DEFAULT 0
                     CHECK (witness_policy_epoch >= 0),
                 witness_policy_head BLOB
                     CHECK (witness_policy_head IS NULL
                         OR length(witness_policy_head) = 32),
                 certificate_import_sequence INTEGER NOT NULL DEFAULT 0
                     CHECK (certificate_import_sequence >= 0),
                 certificate_import_head BLOB
                     CHECK (certificate_import_head IS NULL
                         OR length(certificate_import_head) = 32)
             );
             INSERT INTO directory_replica_meta
                 (singleton, schema_version, chain_id, local_node_id,
                  witness_policy_epoch, witness_policy_head,
                  certificate_import_sequence, certificate_import_head)
             SELECT singleton, 10, chain_id, local_node_id,
                    witness_policy_epoch, witness_policy_head,
                    certificate_import_sequence, certificate_import_head
             FROM directory_replica_meta_v11;
             DROP TABLE directory_replica_meta_v11;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(audit.route_domain_policy_epochs, 0);
    let connection = store.connection.lock();
    let (version, policy_columns, policy_table): (i64, i64, String) = connection
        .query_row(
            "SELECT m.schema_version,
                    (SELECT COUNT(*) FROM pragma_table_info('directory_replica_meta')
                     WHERE name IN (
                         'route_domain_policy_epoch',
                         'route_domain_policy_head'
                     )),
                    t.name
             FROM directory_replica_meta m
             JOIN sqlite_master t
               ON t.type = 'table'
              AND t.name = 'directory_route_domain_policies'
             WHERE m.singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(policy_columns, 2);
    assert_eq!(policy_table, "directory_route_domain_policies");
}

#[test]
fn schema_v11_is_atomically_migrated_to_v12_route_domain_attestor_history() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x12; 32]).unwrap();
    let (store, _) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
    drop(store);
    let connection = Connection::open(&path).unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_route_domain_attestor_policies;
             ALTER TABLE directory_replica_meta RENAME TO directory_replica_meta_v12;
             CREATE TABLE directory_replica_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32),
                 witness_policy_epoch INTEGER NOT NULL DEFAULT 0,
                 witness_policy_head BLOB,
                 route_domain_policy_epoch INTEGER NOT NULL DEFAULT 0,
                 route_domain_policy_head BLOB,
                 certificate_import_sequence INTEGER NOT NULL DEFAULT 0,
                 certificate_import_head BLOB
             );
             INSERT INTO directory_replica_meta
                 (singleton, schema_version, chain_id, local_node_id,
                  witness_policy_epoch, witness_policy_head,
                  route_domain_policy_epoch, route_domain_policy_head,
                  certificate_import_sequence, certificate_import_head)
             SELECT singleton, 11, chain_id, local_node_id,
                    witness_policy_epoch, witness_policy_head,
                    route_domain_policy_epoch, route_domain_policy_head,
                    certificate_import_sequence, certificate_import_head
             FROM directory_replica_meta_v12;
             DROP TABLE directory_replica_meta_v12;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
    assert_eq!(audit.route_domain_attestor_policy_epochs, 0);
    let connection = store.connection.lock();
    let (version, policy_columns, policy_table): (i64, i64, String) = connection
        .query_row(
            "SELECT m.schema_version,
                    (SELECT COUNT(*) FROM pragma_table_info('directory_replica_meta')
                     WHERE name IN (
                         'route_domain_attestor_policy_epoch',
                         'route_domain_attestor_policy_head'
                     )),
                    t.name
             FROM directory_replica_meta m
             JOIN sqlite_master t
               ON t.type = 'table'
              AND t.name = 'directory_route_domain_attestor_policies'
             WHERE m.singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(policy_columns, 2);
    assert_eq!(policy_table, "directory_route_domain_attestor_policies");
}

#[test]
fn schema_v1_is_atomically_migrated_to_v7() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V1],
        )
        .unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_observation_witness_outcomes;
             DROP TABLE directory_observation_checkpoint_witnesses;
             DROP TABLE directory_observation_checkpoints;
             DROP TABLE directory_replica_resolutions;
             DROP TABLE directory_replica_retry_state;
             ALTER TABLE directory_replica_chains DROP COLUMN active_incident_digest;
             ALTER TABLE directory_replica_chains DROP COLUMN last_resolution_digest;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(audit.retry_states, 0);
    let connection = store.connection.lock();
    let version: i64 = connection
        .query_row(
            "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let retry_table: String = connection
        .query_row(
            "SELECT name FROM sqlite_master
             WHERE type = 'table' AND name = 'directory_replica_retry_state'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(retry_table, "directory_replica_retry_state");
    let resolution_columns: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('directory_replica_chains')
             WHERE name IN ('active_incident_digest', 'last_resolution_digest')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(resolution_columns, 2);
}

#[test]
fn schema_v2_is_atomically_migrated_to_v7() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x30; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V2],
        )
        .unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_observation_witness_outcomes;
             DROP TABLE directory_observation_checkpoint_witnesses;
             DROP TABLE directory_observation_checkpoints;
             DROP TABLE directory_replica_resolutions;
             ALTER TABLE directory_replica_chains DROP COLUMN active_incident_digest;
             ALTER TABLE directory_replica_chains DROP COLUMN last_resolution_digest;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(audit.resolutions, 0);
    let connection = store.connection.lock();
    let version: i64 = connection
        .query_row(
            "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let resolution_table: String = connection
        .query_row(
            "SELECT name FROM sqlite_master
             WHERE type = 'table' AND name = 'directory_replica_resolutions'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(resolution_table, "directory_replica_resolutions");
}

#[test]
fn schema_v3_is_atomically_migrated_to_v7() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x2f; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V3],
        )
        .unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_observation_witness_outcomes;
             DROP TABLE directory_observation_checkpoint_witnesses;
             DROP TABLE directory_observation_checkpoints;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(audit.observation_checkpoints, 0);
    let connection = store.connection.lock();
    let (version, checkpoint_table): (i64, String) = connection
        .query_row(
            "SELECT m.schema_version, t.name
             FROM directory_replica_meta m
             JOIN sqlite_master t
               ON t.type = 'table' AND t.name = 'directory_observation_checkpoints'
             WHERE m.singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(checkpoint_table, "directory_observation_checkpoints");
}

#[test]
fn schema_v4_is_atomically_migrated_to_v7() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x2e; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V4],
        )
        .unwrap();
    connection
        .execute_batch(
            "DROP TABLE directory_observation_witness_outcomes;
             DROP TABLE directory_observation_checkpoint_witnesses;",
        )
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(audit.observation_checkpoint_witnesses, 0);
    let connection = store.connection.lock();
    let (version, witness_table): (i64, String) = connection
        .query_row(
            "SELECT m.schema_version, t.name
             FROM directory_replica_meta m
             JOIN sqlite_master t
               ON t.type = 'table'
              AND t.name = 'directory_observation_checkpoint_witnesses'
             WHERE m.singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(witness_table, "directory_observation_checkpoint_witnesses");
}

#[test]
fn schema_v5_is_atomically_migrated_to_v7() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("directory.db");
    let local = IdentityKeyPair::from_bytes(&[0x2d; 32]).unwrap();
    let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
    drop(store);

    let connection = Connection::open(&path).unwrap();
    connection
        .execute(
            "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION_V5],
        )
        .unwrap();
    connection
        .execute_batch("DROP TABLE directory_observation_witness_outcomes;")
        .unwrap();
    drop(connection);

    let (store, audit) =
        DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
    assert_eq!(
        audit.observation_witness_outcomes,
        DirectoryObservationWitnessOutcomeSnapshot::default()
    );
    let connection = store.connection.lock();
    let (version, outcome_table): (i64, String) = connection
        .query_row(
            "SELECT m.schema_version, t.name
             FROM directory_replica_meta m
             JOIN sqlite_master t
               ON t.type = 'table'
              AND t.name = 'directory_observation_witness_outcomes'
             WHERE m.singleton = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    assert_eq!(outcome_table, "directory_observation_witness_outcomes");
}
