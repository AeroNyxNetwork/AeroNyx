// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn disabled_open_has_no_filesystem_side_effect() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("absent").join("mailbox.sqlite");
    let config = AnonymousMailboxStoreConfig {
        db_path: path.display().to_string(),
        ..Default::default()
    };
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(config, [1; 32], CURSOR_SECRET),
        Err(AnonymousMailboxStoreError::Disabled)
    ));
    assert!(!path.exists());
}

#[cfg(unix)]
#[test]
fn new_database_dirent_is_synchronized_before_open_returns() {
    let context = TestContext::new();
    let before = PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get);
    let store = context.open();
    let after = PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get);
    assert_eq!(after, before + 1);
    drop(store);
}

#[cfg(unix)]
#[test]
fn parent_sync_failure_is_fail_closed_and_retry_resynchronizes() {
    let context = TestContext::new();
    FORCE_PARENT_SYNC_FAILURE.with(|forced| forced.set(true));
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            CURSOR_SECRET,
        ),
        Err(AnonymousMailboxStoreError::Unavailable)
    ));
    FORCE_PARENT_SYNC_FAILURE.with(|forced| forced.set(false));
    let before = PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get);
    let store = context.open();
    assert_eq!(
        PARENT_DURABILITY_SYNCS.with(std::cell::Cell::get),
        before + 1
    );
    drop(store);
}

#[test]
fn v2_store_migrates_additively_without_rewriting_custody_rows() {
    let context = TestContext::new();
    let mailbox = [0xCF; 32];
    let lease = context.lease(mailbox, [0xD0; 16], 1, 32, NOW + 1_000);
    let put = context.put(mailbox, [0xD1; 16], b"opaque-v2", NOW + 900);
    let store = context.open();
    store.create(&lease, NOW).unwrap();
    store.put(&put, NOW).unwrap();
    drop(store);

    let connection = Connection::open(&context.config.db_path).unwrap();
    connection
        .execute_batch(
            "DROP INDEX anonymous_mailbox_pull_replay_expiry;
                 DROP TABLE anonymous_mailbox_pull_replays;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_rows;
                 ALTER TABLE anonymous_mailbox_meta DROP COLUMN pull_replay_bytes;
                 UPDATE anonymous_mailbox_meta SET schema_version = 2 WHERE singleton = 1;
                 PRAGMA user_version = 2;",
        )
        .unwrap();
    drop(connection);

    let reopened = context.open();
    let connection = reopened.connection.lock();
    assert_eq!(schema_user_version(&connection), 4);
    assert_eq!(
        connection
            .query_row(
                "SELECT sealed_envelope FROM anonymous_mailbox_items
                     WHERE mailbox_id = ?1 AND item_id = ?2",
                params![&mailbox[..], &put.item_id[..]],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap(),
        b"opaque-v2"
    );
}

#[test]
fn sealed_item_tampering_is_rejected_by_hot_read_and_startup_audit() {
    let context = TestContext::new();
    let mailbox = [0x59; 32];
    let store = context.open();
    store
        .create(&context.lease(mailbox, [0x5A; 16], 2, 64, NOW + 1_000), NOW)
        .unwrap();
    let put = context.put(mailbox, [0x5B; 16], b"opaque", NOW + 100);
    store.put(&put, NOW).unwrap();
    store
        .connection
        .lock()
        .execute(
            "UPDATE anonymous_mailbox_items SET sealed_envelope = ?1",
            params![&b"change"[..]],
        )
        .unwrap();
    assert_eq!(
        store.pull_one(&context.pull(mailbox, Vec::new(), NOW + 1), NOW + 1),
        Err(AnonymousMailboxStoreError::Corrupt)
    );
    assert_eq!(
        store.put(&put, NOW + 1),
        Err(AnonymousMailboxStoreError::Corrupt)
    );
    store
        .connection
        .lock()
        .execute(
            "UPDATE anonymous_mailbox_items
                 SET sealed_envelope = ?1, sealed_commitment = zeroblob(32)",
            params![&b"opaque"[..]],
        )
        .unwrap();
    assert_eq!(
        store.pull_one(&context.pull(mailbox, Vec::new(), NOW + 1), NOW + 1),
        Err(AnonymousMailboxStoreError::Corrupt)
    );
    drop(store);
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            CURSOR_SECRET,
        ),
        Err(AnonymousMailboxStoreError::Corrupt)
    ));
}

#[test]
fn global_capacity_is_atomic_across_connections() {
    let mut context = TestContext::new();
    context.config.max_leases_total = 1;
    let first = Arc::new(context.open());
    let second = Arc::new(context.open());
    let request_a = context.lease([0x61; 32], [0x62; 16], 1, 8, NOW + 1_000);
    let request_b = context.lease([0x63; 32], [0x64; 16], 1, 8, NOW + 1_000);
    let barrier = Arc::new(Barrier::new(3));
    let run = |store: Arc<SqliteAnonymousMailboxStore>,
               request: AnonymousMailboxLeaseCreateV1,
               barrier: Arc<Barrier>| {
        std::thread::spawn(move || {
            barrier.wait();
            store.create(&request, NOW).unwrap()
        })
    };
    let a = run(Arc::clone(&first), request_a, Arc::clone(&barrier));
    let b = run(Arc::clone(&second), request_b, Arc::clone(&barrier));
    barrier.wait();
    let outcomes = [a.join().unwrap(), b.join().unwrap()];
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, AnonymousMailboxCreateOutcome::Created(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, AnonymousMailboxCreateOutcome::AtCapacity))
            .count(),
        1
    );
}

#[test]
fn in_flight_gate_is_shared_by_all_repository_methods() {
    let mut context = TestContext::new();
    context.config.max_in_flight = 1;
    let store = context.open();
    let _held = store.acquire().unwrap();
    assert_eq!(store.cleanup(NOW), Err(AnonymousMailboxStoreError::Busy));
}

#[test]
fn mutation_hot_paths_do_not_invoke_full_database_audit() {
    let mut context = TestContext::new();
    context.config.max_leases_total = 64;
    let store = context.open();
    let audits_after_startup = FULL_AUDIT_CALLS.with(std::cell::Cell::get);
    for discriminator in 1_u8..=48 {
        let lease = context.lease([discriminator; 32], [discriminator; 16], 2, 64, NOW + 1_000);
        assert!(matches!(
            store.create(&lease, NOW).unwrap(),
            AnonymousMailboxCreateOutcome::Created(_)
        ));
    }
    let mailbox = [48; 32];
    let put = context.put(mailbox, [0xF0; 16], b"opaque", NOW + 100);
    store.put(&put, NOW).unwrap();
    let ack = AnonymousMailboxAckV1::new(
        mailbox,
        [0xF1; 16],
        put.item_id,
        put.sealed_commitment(),
        NOW + 1,
        &context.reader,
    )
    .unwrap();
    store.ack(&ack, NOW + 1).unwrap();
    store.cleanup(NOW + 2).unwrap();
    assert_eq!(
        FULL_AUDIT_CALLS.with(std::cell::Cell::get),
        audits_after_startup
    );
}

#[test]
fn schema_has_no_identity_or_routing_columns_and_unknown_version_fails_closed() {
    let context = TestContext::new();
    let store = context.open();
    let connection = store.connection.lock();
    let mut statement = connection
        .prepare(
            "SELECT m.name, p.name FROM sqlite_master m, pragma_table_info(m.name) p
                 WHERE m.type = 'table' AND m.name LIKE 'anonymous_mailbox_%'",
        )
        .unwrap();
    let columns: Vec<String> = statement
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    for forbidden in [
        "sender", "receiver", "wallet", "route", "endpoint", "identity",
    ] {
        assert!(columns.iter().all(|column| !column.contains(forbidden)));
    }
    drop(statement);
    drop(connection);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for path in [
            std::path::PathBuf::from(&context.config.db_path),
            std::path::PathBuf::from(format!("{}-wal", context.config.db_path)),
            std::path::PathBuf::from(format!("{}-shm", context.config.db_path)),
        ] {
            if path.exists() {
                let mode = std::fs::symlink_metadata(path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777;
                assert_eq!(mode, 0o600);
            }
        }
    }
    drop(store);
    let connection = Connection::open(&context.config.db_path).unwrap();
    connection.pragma_update(None, "user_version", 99).unwrap();
    drop(connection);
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            CURSOR_SECRET,
        ),
        Err(AnonymousMailboxStoreError::UnsupportedSchema)
    ));
}

#[test]
fn foreign_table_on_user_version_zero_fails_without_mailbox_mutation() {
    let context = TestContext::new();
    let connection = Connection::open(&context.config.db_path).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE pending_messages (
                    id INTEGER PRIMARY KEY,
                    body TEXT NOT NULL
                 );
                 INSERT INTO pending_messages (body) VALUES ('still-here');",
        )
        .unwrap();
    drop(connection);

    assert!(matches!(
        SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            CURSOR_SECRET,
        ),
        Err(AnonymousMailboxStoreError::UnsupportedSchema)
    ));

    let connection = Connection::open(&context.config.db_path).unwrap();
    assert_eq!(schema_user_version(&connection), 0);
    assert_eq!(anonymous_mailbox_object_count(&connection), 0);
    assert_eq!(
        connection
            .query_row(
                "SELECT body FROM pending_messages WHERE id = 1",
                [],
                |row| { row.get::<_, String>(0) }
            )
            .unwrap(),
        "still-here"
    );
}

#[test]
fn foreign_view_and_trigger_on_user_version_zero_fail_without_mailbox_mutation() {
    let context = TestContext::new();
    let connection = Connection::open(&context.config.db_path).unwrap();
    connection
        .execute_batch(
            "CREATE VIEW pending_view AS SELECT name FROM sqlite_master;
                 CREATE TRIGGER pending_view_block
                 INSTEAD OF INSERT ON pending_view
                 BEGIN
                    SELECT RAISE(ABORT, 'blocked');
                 END;",
        )
        .unwrap();
    drop(connection);

    assert!(matches!(
        SqliteAnonymousMailboxStore::open(
            context.config.clone(),
            context.target.public_key_bytes(),
            CURSOR_SECRET,
        ),
        Err(AnonymousMailboxStoreError::UnsupportedSchema)
    ));

    let connection = Connection::open(&context.config.db_path).unwrap();
    assert_eq!(schema_user_version(&connection), 0);
    assert_eq!(anonymous_mailbox_object_count(&connection), 0);
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                     WHERE name IN ('pending_view', 'pending_view_block')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        2
    );
}

#[test]
fn empty_reserved_database_initializes_and_reopens() {
    let context = TestContext::new();
    std::fs::write(&context.config.db_path, []).unwrap();

    let store = context.open();
    drop(store);

    let connection = Connection::open(&context.config.db_path).unwrap();
    assert_eq!(schema_user_version(&connection), SCHEMA_VERSION);
    assert!(anonymous_mailbox_object_count(&connection) > 0);
    drop(connection);

    let reopened = context.open();
    drop(reopened);
}

#[cfg(unix)]
#[test]
fn symlink_database_path_fails_closed() {
    use std::os::unix::fs::symlink;

    let context = TestContext::new();
    let target_path = context._directory.path().join("target.sqlite");
    Connection::open(&target_path).unwrap();
    let link_path = context._directory.path().join("alias.sqlite");
    symlink(&target_path, &link_path).unwrap();
    let mut config = context.config.clone();
    config.db_path = link_path.display().to_string();
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(config, context.target.public_key_bytes(), CURSOR_SECRET,),
        Err(AnonymousMailboxStoreError::Rejected)
    ));
}

#[cfg(unix)]
#[test]
fn hardlinked_database_is_rejected_without_mode_side_effect() {
    use std::os::unix::fs::PermissionsExt;

    let context = TestContext::new();
    let parent = std::fs::canonicalize(context._directory.path()).unwrap();
    let external = parent.join("external.sqlite");
    std::fs::write(&external, b"external").unwrap();
    std::fs::set_permissions(&external, std::fs::Permissions::from_mode(0o640)).unwrap();
    let alias = parent.join("mailbox-hardlink.sqlite");
    std::fs::hard_link(&external, &alias).unwrap();
    let mode_before = std::fs::metadata(&external).unwrap().permissions().mode() & 0o777;
    let mut config = context.config.clone();
    config.db_path = alias.display().to_string();
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(config, context.target.public_key_bytes(), CURSOR_SECRET,),
        Err(AnonymousMailboxStoreError::Rejected)
    ));
    let mode_after = std::fs::metadata(&external).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode_after, mode_before);
}

#[cfg(unix)]
#[test]
fn existing_owned_database_mode_is_normalized_before_sqlite_activation() {
    use std::os::unix::fs::PermissionsExt;

    let context = TestContext::new();
    let database = PathBuf::from(&context.config.db_path);
    std::fs::write(&database, []).unwrap();
    std::fs::set_permissions(&database, std::fs::Permissions::from_mode(0o640)).unwrap();
    let store = context.open();
    let mode = std::fs::metadata(&database).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    drop(store);
}

#[cfg(unix)]
#[test]
fn symlink_parent_component_is_rejected_without_database_creation() {
    use std::os::unix::fs::{symlink, PermissionsExt};

    let context = TestContext::new();
    let parent = std::fs::canonicalize(context._directory.path()).unwrap();
    let actual = parent.join("actual-private");
    std::fs::create_dir(&actual).unwrap();
    std::fs::set_permissions(&actual, std::fs::Permissions::from_mode(0o700)).unwrap();
    let alias = parent.join("parent-alias");
    symlink(&actual, &alias).unwrap();
    let mut config = context.config.clone();
    config.db_path = alias.join("mailbox.sqlite").display().to_string();
    assert!(matches!(
        SqliteAnonymousMailboxStore::open(config, context.target.public_key_bytes(), CURSOR_SECRET,),
        Err(AnonymousMailboxStoreError::Rejected)
    ));
    assert!(!actual.join("mailbox.sqlite").exists());
}
