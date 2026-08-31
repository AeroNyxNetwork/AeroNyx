// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_tests.rs
// ============================================
// Version: 1.6.0-BlindRouteResourceSchemaV5
//
// Creation Reason:
//   [CHAT-RELAY-TEST-MODULE-SPLIT 2026-08-27 by Codex] Move the complete
//   `chat_relay` in-crate test module out of the production implementation.
//
// Modification Reason:
//   [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] Pins logical-byte,
//   concurrent, restart-safe v5, TTL, and pinned-reader WAL recovery behavior.
//   [BLIND-ROUTE-RESPONSE-SCHEMA-V4 2026-08-31 by Codex] Pins shared-ceiling
//   DDL, byte-preserving v3 migration, rollback, retry, and writer locking.
//   [CHAT-RELAY-TEST-IMPORTS 2026-08-31 by Codex] Imports the cursor value and
//   dedup capability from their extracted owner modules explicitly.
//   [CHAT-RELAY-CLEANUP-EXECUTION-DOMAIN 2026-08-28 by Codex] Declared the
//   test module's transaction behavior dependency explicitly after cleanup
//   transaction ownership moved out of the parent relay implementation.
//   [CHAT-RELAY-PENDING-SCHEMA-DOMAIN 2026-08-27 by Codex] Declared the test
//   module's `HashSet` dependency explicitly after schema migration ownership
//   moved out of the parent relay implementation.
//   [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] Pin public-plan
//   validation before any private path or maintenance-lock side effect.
//
// Main Functionality:
//   - Preserves all unit, integration, restart, concurrency, and fault tests.
//   - Retains private parent-module access through `use super::*`.
//   - Verifies relay custody, backup, recovery, routing, and cleanup behavior.
//
// Dependencies:
//   - Loaded only by `chat_relay.rs` through its explicit `#[path]` module.
//   - Uses private parent items and existing crate test dependencies.
//
// Main Logical Flow:
//   1. Construct bounded test configurations and isolated relay services.
//   2. Exercise public and private state transitions under controlled inputs.
//   3. Assert success, failure, replay, restart, and concurrency invariants.
//   4. Leave production compilation free of test-only implementation volume.
//
// Important Note for Next Developer:
//   - Keep this as the `chat_relay::tests` module; test paths are stable.
//   - Relative `include_str!` paths assume this file remains in `services/`.
//   - Do not expose private test fixtures through production APIs.
//   - New relay tests belong here or in a focused extracted domain module.
//
// Last Modified:
//   v1.6.0-BlindRouteResourceSchemaV5 - Covered byte and WAL admission
//   v1.5.0-BlindRouteResponseSchemaV4 - Covered bounded atomic CHECK migration
//   v1.4.0-ExplicitExtractedDomainDependencies - Restored test compilation
//   v1.3.0-ExplicitTransactionDependency - Removed parent-import coupling
//   v1.2.0-ExplicitCollectionDependency - Removed parent-import coupling
//   v1.1.0-RestoreValidationSideEffectInvariant - Pinned pre-path rejection
//   v1.0.0-TestModuleSplit - Mechanical extraction from `chat_relay.rs`
// ============================================

use super::*;
use crate::services::chat_relay_blind_route::MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES;
use crate::services::chat_relay_message_dedup::OnlineMessageDeduplication;
use crate::services::chat_relay_pull_cursor::PullCursorV2;
use crate::services::chat_relay_replay_schema::{
    ChatRelayReplaySchemaMigration, ReplaySchemaContract, ReplaySchemaVersion,
    SqliteChatRelayReplaySchemaMigrator,
};
use aeronyx_common::types::SessionId;
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::chat::ChatContentType;
use rusqlite::TransactionBehavior;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Barrier;
use std::time::Duration;

fn test_config() -> ChatRelayConfig {
    ChatRelayConfig {
        enabled: true,
        db_path: ":memory:".to_string(),
        offline_ttl_secs: 259_200,
        max_pending_per_wallet: 5,
        max_pending_messages_total: 100,
        max_pending_message_bytes_total: 1024 * 1024,
        max_message_size: 65_536,
        max_blob_size: 1_024,
        max_blobs_per_receiver: 3,
        max_pending_blobs_total: 10,
        max_pending_blob_bytes_total: 10 * 1024,
        cleanup_interval_secs: 60,
        dedup_lru_capacity: 10,
        expired_notification_ttl_secs: 604_800,
        peer_relay_requests_per_minute: 1_200,
        peer_relay_authenticated_requests_per_minute: 240,
        // [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Service tests
        // inherit recovery-planning defaults unless a case overrides them.
        ..ChatRelayConfig::default()
    }
}

fn make_service() -> ChatRelayService {
    make_service_with_config(test_config())
}

fn make_service_with_config(config: ChatRelayConfig) -> ChatRelayService {
    let secret = derive_node_secret(&[0x42u8; 32]);
    ChatRelayService::new(config, secret).expect("init")
}

fn test_replay_schema_migrator(blind_route_capacity: usize) -> SqliteChatRelayReplaySchemaMigrator {
    test_replay_schema_migrator_with_budget(
        blind_route_capacity,
        ChatRelayConfig::default().blind_route_replay_max_bytes_total,
    )
}

fn test_replay_schema_migrator_with_budget(
    blind_route_capacity: usize,
    blind_route_live_opaque_bytes: u64,
) -> SqliteChatRelayReplaySchemaMigrator {
    SqliteChatRelayReplaySchemaMigrator::new(ReplaySchemaContract::new(
        ReplaySchemaVersion::new(
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
        ),
        ReplaySchemaVersion::new(
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V4_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION,
        ),
        VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
        BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
        REPLAY_PROCESS_EPOCH_BYTES,
        blind_route_capacity,
        blind_route_live_opaque_bytes,
    ))
}

fn replace_blind_route_response_table_with_v3(connection: &Connection) {
    // Test-only fixture for the exact production v3 CHECK that P4 replaces.
    connection
        .execute_batch(
            "DROP TABLE relay_blind_route_responses;
             CREATE TABLE relay_blind_route_responses (
                cache_key           BLOB    PRIMARY KEY CHECK(LENGTH(cache_key) = 32),
                request_fingerprint BLOB    NOT NULL CHECK(LENGTH(request_fingerprint) = 32),
                response_nonce      BLOB    NOT NULL CHECK(LENGTH(response_nonce) = 24),
                response_ciphertext BLOB    NOT NULL CHECK(
                    LENGTH(response_ciphertext) > 16
                    AND LENGTH(response_ciphertext) <= 2064
                ),
                completed_at        INTEGER NOT NULL CHECK(completed_at >= 0)
             );
             CREATE INDEX idx_blind_route_response_retention
                ON relay_blind_route_responses(completed_at);",
        )
        .expect("replace response table with v3 fixture");
    connection
        .execute(
            "UPDATE relay_schema_features SET schema_version = ?1
             WHERE feature = ?2",
            params![
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
            ],
        )
        .expect("mark blind-route response fixture as v3");
}

fn complete_direct_peer_test_delivery(
    svc: &ChatRelayService,
    now: u64,
    retry_triggered: bool,
    delivery_succeeded: bool,
    final_failure_deterministic: bool,
) {
    let permit = svc
        .begin_direct_peer_delivery(now)
        .expect("test delivery should be admitted");
    svc.complete_direct_peer_delivery(
        now,
        permit,
        retry_triggered,
        delivery_succeeded,
        final_failure_deterministic,
    );
}

fn unique_test_db_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "aeronyx-chat-relay-{}-{}-{}.sqlite",
        label,
        std::process::id(),
        rand::random::<u64>()
    ))
}

fn remove_test_database(path: &Path) {
    let _ = std::fs::remove_file(path);
    for suffix in ["-wal", "-shm"] {
        let _ = std::fs::remove_file(PathBuf::from(format!("{}{suffix}", path.display())));
    }
    #[cfg(unix)]
    if let Ok(path) = ChatRelayRuntimeFence::control_path(path) {
        let _ = std::fs::remove_file(path);
    }
}

fn remove_test_db(path: &Path) {
    let _ = std::fs::remove_file(path);
    remove_test_db_sidecars(path);
}

fn remove_test_db_sidecars(path: &Path) {
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
    let _ = std::fs::remove_file(format!("{}-journal", path.display()));
    #[cfg(unix)]
    if let Ok(path) = ChatRelayRuntimeFence::control_path(path) {
        let _ = std::fs::remove_file(path);
    }
}

fn backup_directory_snapshot(path: &Path) -> Vec<(String, Vec<u8>)> {
    let mut snapshot = std::fs::read_dir(path)
        .expect("read backup directory snapshot")
        .map(|entry| {
            let entry = entry.expect("read backup directory entry");
            let name = entry
                .file_name()
                .into_string()
                .expect("test backup name is UTF-8");
            let bytes = std::fs::read(entry.path()).expect("read backup artifact");
            (name, bytes)
        })
        .collect::<Vec<_>>();
    snapshot.sort_by(|left, right| left.0.cmp(&right.0));
    snapshot
}

fn insert_expired_pending_rows(svc: &ChatRelayService, count: usize, prefix: u8) {
    let identity = IdentityKeyPair::generate();
    let mut conn = svc.conn.lock();
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .expect("start bulk pending insert");
    {
        let mut stmt = tx
            .prepare(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status,
                  queue_sequence)
                 VALUES (?1, ?2, ?3, 0, ?4, 0, 0, ?5)",
            )
            .expect("prepare bulk pending insert");
        for sequence in 0..count {
            let mut message_id = [0u8; 16];
            message_id[0] = prefix;
            message_id[8..]
                .copy_from_slice(&u64::try_from(sequence).unwrap_or(u64::MAX).to_be_bytes());
            let mut envelope = ChatEnvelope {
                message_id,
                sender: identity.public_key_bytes(),
                receiver: [0xA3u8; 32],
                timestamp: 0,
                ciphertext: vec![0xA4],
                nonce: [0u8; 24],
                content_type: ChatContentType::System,
                signature: [0u8; 64],
            };
            envelope.signature = identity.sign(&envelope.sign_data());
            let encoded_envelope = encode_envelope(&envelope).expect("encode expired envelope");
            let queue_sequence =
                allocate_queue_sequence(&tx).expect("allocate test queue sequence");
            stmt.execute(params![
                message_id.as_slice(),
                envelope.sender.as_slice(),
                envelope.receiver.as_slice(),
                encoded_envelope,
                queue_sequence,
            ])
            .expect("insert expired pending row");
        }
    }
    tx.commit().expect("commit bulk pending insert");
}

#[test]
fn chat_relay_custody_uses_full_sqlite_durability() {
    // [CHAT-RELAY-FULL-DURABILITY 2026-08-16 by Codex] A successful
    // service construction is also the activation gate for signed custody
    // receipts, so the effective connection mode must be FULL or EXTRA.
    let svc = make_service();
    let synchronous_level = svc
        .conn
        .lock()
        .query_row("PRAGMA synchronous", [], |row| row.get::<_, i64>(0))
        .expect("read effective relay durability");
    assert!(synchronous_level >= CHAT_RELAY_SQLITE_MINIMUM_SYNCHRONOUS_LEVEL);
    let durability = svc.peer_status().custody_durability;
    assert_eq!(durability.state, "full");
    assert!(durability.full_durability_verified);
    assert_eq!(
        durability.synchronous_level,
        Some(u8::try_from(synchronous_level).expect("SQLite level fits u8"))
    );
}

#[cfg(unix)]
#[test]
fn chat_relay_custody_files_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] Cover both an existing
    // permissive database and the WAL/SHM files SQLite creates after the
    // primary mode is tightened. This test never changes process umask.
    let db_path = unique_test_db_path("private-custody-file");
    std::fs::write(&db_path, []).expect("create permissive relay database");
    std::fs::set_permissions(&db_path, std::fs::Permissions::from_mode(0o666))
        .expect("make relay database permissive");

    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let service = make_service_with_config(config);

    for path in [
        db_path.clone(),
        PathBuf::from(format!("{}-wal", db_path.display())),
        PathBuf::from(format!("{}-shm", db_path.display())),
        ChatRelayRuntimeFence::control_path(&db_path).expect("derive private runtime fence path"),
    ] {
        let mode = std::fs::metadata(&path)
            .unwrap_or_else(|error| panic!("inspect {}: {error}", path.display()))
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o600, "{} must be owner-only", path.display());
    }

    drop(service);
    remove_test_db(&db_path);
}

#[cfg(unix)]
#[test]
fn chat_relay_runtime_fence_rejects_concurrent_database_owner() {
    // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] Reservation takeover
    // is safe only after the predecessor has exited. Prove that a second
    // live service cannot open, migrate, or mutate the same custody store,
    // while RAII release still permits a normal restart.
    let db_path = unique_test_db_path("runtime-fence-owner");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xA7; 32];

    let owner =
        ChatRelayService::new(config.clone(), secret).expect("acquire first relay runtime fence");
    let error = match ChatRelayService::new(config.clone(), secret) {
        Ok(_) => panic!("concurrent relay owner must fail closed"),
        Err(error) => error,
    };
    assert!(matches!(
        &error,
        ChatRelayError::RuntimeFenceUnavailable {
            reason: "already_owned",
            ..
        }
    ));
    assert_eq!(error.reason_bucket(), "runtime_fence_unavailable");
    assert!(!error
        .to_string()
        .contains(db_path.to_string_lossy().as_ref()));

    drop(owner);
    let restarted = ChatRelayService::new(config, secret)
        .expect("kernel releases relay runtime fence after owner drop");
    drop(restarted);
    remove_test_db(&db_path);
}

#[cfg(unix)]
#[test]
fn chat_relay_runtime_fence_rejects_hard_linked_control_file() {
    use std::os::unix::fs::PermissionsExt;

    // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] The sidecar must be a
    // unique inode. Reject a hard link before permission tightening so an
    // unrelated owner file cannot be modified through the control path.
    let db_path = unique_test_db_path("runtime-fence-hard-link");
    let control_path =
        ChatRelayRuntimeFence::control_path(&db_path).expect("derive hard-link runtime fence path");
    let target_path = db_path.with_extension("owner-state");
    std::fs::write(&target_path, b"owner-state").expect("write hard-link target");
    std::fs::set_permissions(&target_path, std::fs::Permissions::from_mode(0o640))
        .expect("set hard-link target mode");
    std::fs::hard_link(&target_path, &control_path).expect("install hard-linked control file");

    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let error = match ChatRelayService::new(config, [0xA8; 32]) {
        Ok(_) => panic!("hard-linked runtime fence must fail closed"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        ChatRelayError::RuntimeFenceUnavailable {
            reason: "unsafe_control_file",
            ..
        }
    ));
    assert_eq!(
        std::fs::read(&target_path).expect("read unchanged hard-link target"),
        b"owner-state"
    );
    assert_eq!(
        std::fs::metadata(&target_path)
            .expect("inspect unchanged hard-link target")
            .permissions()
            .mode()
            & 0o777,
        0o640
    );

    let _ = std::fs::remove_file(control_path);
    let _ = std::fs::remove_file(target_path);
    remove_test_db(&db_path);
}

#[cfg(unix)]
#[test]
fn chat_relay_permission_failure_is_path_private_and_fail_closed() {
    // [CHAT-RELAY-PRIVATE-FILE 2026-08-16 by Codex] A missing target
    // deterministically exercises the activation error without relying on
    // process-global umask, root privileges, or platform ACL behavior.
    let db_path = unique_test_db_path("missing-private-custody-file");
    remove_test_db(&db_path);

    let error = ChatRelayService::restrict_sqlite_file_permissions(&db_path)
        .expect_err("unrestrictable relay database must fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");
    let rendered = error.to_string();
    assert!(rendered.contains("unable to restrict relay database permissions"));
    assert!(!rendered.contains(db_path.to_string_lossy().as_ref()));
}

#[test]
fn chat_relay_startup_integrity_rejects_corrupt_schema_before_activation() {
    // [CHAT-RELAY-STARTUP-QUICK-CHECK 2026-08-16 by Codex] Build a valid
    // production-shaped store, corrupt one schema root page through
    // SQLite's test-only writable schema, then exercise a real restart.
    let db_path = unique_test_db_path("startup-integrity-corrupt-schema");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x42u8; 32]);

    let service = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    drop(service);

    let corruptor = Connection::open(&db_path).expect("open relay store for corruption drill");
    corruptor
        .execute_batch(
            "PRAGMA writable_schema=ON;
             UPDATE sqlite_schema
             SET rootpage=2147483647
             WHERE type='table' AND name='pending_messages';
             PRAGMA writable_schema=OFF;",
        )
        .expect("install malformed root page fixture");
    drop(corruptor);

    let error = match ChatRelayService::new(config, secret) {
        Ok(_) => panic!("corrupt custody database must not activate"),
        Err(error) => error,
    };
    assert_eq!(error.reason_bucket(), "corrupt_stored_data");
    let rendered = error.to_string();
    assert!(rendered.contains("sqlite_startup_integrity"));
    assert!(!rendered.contains(db_path.to_string_lossy().as_ref()));

    remove_test_db(&db_path);
}

#[test]
fn verified_backup_restores_committed_custody_and_circuit_state() {
    // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] Disable automatic
    // checkpoints so this drill proves the backup API reads committed WAL
    // pages rather than merely copying the primary database file.
    let directory = tempfile::tempdir().expect("verified backup directory");
    let source_path = directory.path().join("source.sqlite");
    let mut source_config = test_config();
    source_config.db_path = source_path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x8Au8; 32]);
    let identity = IdentityKeyPair::generate();
    let receiver = [0x8Bu8; 32];
    let first = make_envelope(&identity, receiver);
    let second = make_envelope(&identity, receiver);
    let blob_data = b"opaque-encrypted-backup-blob";
    let blob_hash = Sha256::digest(blob_data);
    let mut blob_hash_array = [0u8; 32];
    blob_hash_array.copy_from_slice(&blob_hash);
    let circuit_started_at = now_secs();
    let pending_submit = ChatRelayVerifiedSubmitRequestV1::signed(
        [0x8C; 16],
        make_envelope(&identity, [0x8D; 32]),
        circuit_started_at,
        &identity,
    )
    .expect("sign backup pending verified submit request");
    let pending_blind_route_id = [0x8E; 16];
    let pending_blind_request_commitment = [0x8F; 32];

    let source =
        ChatRelayService::new(source_config.clone(), secret).expect("create source relay store");
    source
        .conn
        .lock()
        .execute_batch("PRAGMA wal_autocheckpoint=0")
        .expect("keep committed custody in WAL");
    source.store_pending(&first).expect("store pending message");
    assert_eq!(
        source
            .reserve_verified_submit(&pending_submit)
            .expect("reserve pending verified submit before backup"),
        VerifiedSubmitAdmission::Reserved
    );
    assert_eq!(
        source
            .reserve_blind_relay_route(&pending_blind_route_id, &pending_blind_request_commitment,)
            .expect("reserve pending blind route before backup"),
        BlindRelayRouteAdmission::Reserved
    );
    let blob_id = source
        .put_blob(
            &identity.public_key_bytes(),
            &receiver,
            blob_data,
            &blob_hash_array,
        )
        .expect("store encrypted blob");
    for offset in 0..3 {
        complete_direct_peer_test_delivery(
            &source,
            circuit_started_at.saturating_add(offset),
            false,
            false,
            true,
        );
    }
    assert_eq!(source.peer_status().direct_peer_retry.circuit.state, "open");

    let backup_path = source
        .create_verified_backup()
        .expect("create verified backup");
    source
        .store_pending(&second)
        .expect("store post-snapshot message");
    drop(source);

    let mut restored_config = source_config.clone();
    restored_config.db_path = backup_path.to_string_lossy().into_owned();
    let restored =
        ChatRelayService::new(restored_config, secret).expect("activate verified recovery image");
    let (messages, has_more) = restored
        .pull_pending(&receiver, 0, &[0u8; 16], 10)
        .expect("read restored custody");
    assert!(!has_more);
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].message_id, first.message_id);
    assert_eq!(messages[0].envelope.ciphertext, first.ciphertext);
    assert_eq!(restored.get_blob(&blob_id).unwrap(), blob_data);
    assert_eq!(
        restored.storage_usage().unwrap(),
        ChatRelayStorageUsage {
            pending_messages: 1,
            pending_message_bytes: encode_envelope(&first).unwrap().len() as u64,
            pending_blobs: 1,
            pending_blob_bytes: blob_data.len() as u64,
        }
    );
    let circuit = restored.peer_status().direct_peer_retry.circuit;
    assert_eq!(circuit.state, "open");
    assert!(circuit.restart_protected);
    assert_eq!(circuit.opened_total, 1);
    assert!(matches!(
        restored
            .verified_submit_cache_lookup(&pending_submit)
            .expect("restore pending verified submit reservation"),
        VerifiedSubmitCacheLookup::Pending
    ));
    assert_eq!(
        restored
            .reserve_blind_relay_route(&pending_blind_route_id, &pending_blind_request_commitment,)
            .expect("restore pending blind-route reservation"),
        BlindRelayRouteAdmission::Pending
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let backup_mode = std::fs::metadata(&backup_path)
            .expect("inspect backup file")
            .permissions()
            .mode()
            & 0o777;
        let backup_directory_mode = std::fs::metadata(backup_path.parent().unwrap())
            .expect("inspect backup directory")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(backup_mode, 0o600);
        assert_eq!(backup_directory_mode, 0o700);
    }

    drop(restored);
    remove_test_db(&backup_path);
    remove_test_db(&source_path);
}

#[test]
fn restore_readiness_selects_verified_backup_without_mutating_artifacts() {
    // [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] A positive
    // preflight proves the newest image is fully usable while preserving
    // both active custody and every recovery artifact byte-for-byte.
    let directory = tempfile::tempdir().expect("restore readiness directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config.clone());
    source
        .create_verified_backup_for_operation("restore-readiness")
        .expect("create verified readiness image");
    drop(source);
    remove_test_db_sidecars(&source_path);

    let backup_directory = directory.path().join(".aeronyx-relay-backups");
    let before = backup_directory_snapshot(&backup_directory);
    let active_before = std::fs::read(&source_path).expect("read active custody before audit");
    let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(&config)
        .expect("audit restore readiness");
    let after = backup_directory_snapshot(&backup_directory);

    assert!(receipt.ready);
    assert_eq!(receipt.verified_backup_count, 1);
    assert!(receipt.selected_backup_bytes > 0);
    assert!(receipt.active_database_present);
    assert_eq!(receipt.active_database_bytes, active_before.len() as u64);
    assert!(!receipt.active_sidecars_present);
    assert_eq!(receipt.blocker, None);
    assert_eq!(before, after);
    assert_eq!(
        std::fs::read(&source_path).expect("read active custody after audit"),
        active_before
    );

    // [CHAT-RELAY-RESTORE-READINESS 2026-08-16 by Codex] Pin the public
    // aggregate contract: operators may automate against these fields,
    // while custody and recovery paths must remain private.
    let json = serde_json::to_value(&receipt).expect("serialize restore readiness receipt");
    let object = json.as_object().expect("readiness JSON is an object");
    assert_eq!(object.len(), 7);
    for field in [
        "ready",
        "verified_backup_count",
        "selected_backup_bytes",
        "active_database_present",
        "active_database_bytes",
        "active_sidecars_present",
        "blocker",
    ] {
        assert!(object.contains_key(field), "missing JSON field: {field}");
    }
    let encoded = serde_json::to_string(&receipt).expect("encode readiness JSON");
    assert!(!encoded.contains(source_path.to_string_lossy().as_ref()));
    assert!(!encoded.contains(".aeronyx-relay-backups"));
}

#[test]
fn restore_readiness_reports_missing_verified_backup_without_execution() {
    let directory = tempfile::tempdir().expect("missing restore image directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config.clone());
    drop(source);
    remove_test_db_sidecars(&source_path);

    let active_before = std::fs::read(&source_path).expect("read active custody");
    let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(&config)
        .expect("report missing verified backup");

    assert!(!receipt.ready);
    assert_eq!(receipt.verified_backup_count, 0);
    assert_eq!(receipt.selected_backup_bytes, 0);
    assert_eq!(receipt.blocker, Some("no_verified_backup"));
    assert_eq!(
        std::fs::read(&source_path).expect("read unchanged active custody"),
        active_before
    );
}

#[test]
fn restore_readiness_fails_closed_while_active_sidecar_exists() {
    let directory = tempfile::tempdir().expect("restore sidecar directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config.clone());
    source
        .create_verified_backup_for_operation("restore-sidecar")
        .expect("create verified sidecar image");
    drop(source);
    remove_test_db_sidecars(&source_path);
    let wal_path = PathBuf::from(format!("{}-wal", source_path.display()));
    std::fs::write(&wal_path, b"stopped-state-sidecar-marker").expect("create sidecar marker");

    let receipt = ChatRelayService::audit_latest_restore_readiness_for_config(&config)
        .expect("report active sidecar blocker");
    assert!(!receipt.ready);
    assert!(receipt.active_sidecars_present);
    assert_eq!(receipt.blocker, Some("active_sqlite_sidecars_present"));
    assert!(
        wal_path.exists(),
        "readiness must not remove active sidecars"
    );
}

#[test]
fn restore_plan_is_path_free_unique_and_verifiable() {
    // [CHAT-RELAY-RESTORE-PLAN 2026-08-16 by Codex] The public plan binds
    // private artifact identity without exposing a filename or path.
    let directory = tempfile::tempdir().expect("restore plan directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config.clone());
    source
        .create_verified_backup_for_operation("restore-plan")
        .expect("create verified restore-plan image");
    drop(source);
    remove_test_db_sidecars(&source_path);

    let secret = derive_node_secret(&[0x42u8; 32]);
    let issued_at = now_secs();
    let first = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
        .expect("create first authenticated plan");
    let second = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
        .expect("create second authenticated plan");

    assert_eq!(first.version, CHAT_RELAY_RESTORE_PLAN_VERSION);
    assert_eq!(
        first.expires_at - first.issued_at,
        CHAT_RELAY_RESTORE_PLAN_VALIDITY_SECS
    );
    assert_ne!(first.nonce, second.nonce);
    assert_ne!(first.commitment, second.commitment);
    assert!(ChatRelayService::is_lower_hex(&first.nonce, 32));
    assert!(ChatRelayService::is_lower_hex(&first.commitment, 64));
    ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &first, issued_at)
        .expect("verify fresh restore plan");

    let encoded = serde_json::to_string(&first).expect("encode restore plan");
    assert!(!encoded.contains(source_path.to_string_lossy().as_ref()));
    assert!(!encoded.contains(".aeronyx-relay-backups"));
    assert!(!encoded.contains("relay-custody-operation"));
    let json = serde_json::to_value(&first).expect("serialize restore-plan contract");
    let object = json.as_object().expect("restore plan JSON object");
    assert_eq!(object.len(), 9);
    for field in [
        "version",
        "issued_at",
        "expires_at",
        "verified_backup_count",
        "selected_backup_bytes",
        "active_database_present",
        "active_database_bytes",
        "nonce",
        "commitment",
    ] {
        assert!(object.contains_key(field), "missing plan field: {field}");
    }
}

#[test]
fn restore_plan_rejects_tampering_wrong_key_and_expiry() {
    let directory = tempfile::tempdir().expect("restore plan auth directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config.clone());
    source
        .create_verified_backup_for_operation("restore-plan-auth")
        .expect("create verified restore-plan auth image");
    drop(source);
    remove_test_db_sidecars(&source_path);

    let secret = derive_node_secret(&[0x42u8; 32]);
    let issued_at = now_secs();
    let plan = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
        .expect("create authenticated plan");

    let mut tampered = plan.clone();
    tampered.selected_backup_bytes = tampered.selected_backup_bytes.saturating_add(1);
    ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &tampered, issued_at)
        .expect_err("aggregate tampering must fail closed");
    ChatRelayService::verify_latest_restore_plan_at(&config, &[0xA5u8; 32], &plan, issued_at)
        .expect_err("wrong node secret must fail closed");
    ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &plan, plan.expires_at)
        .expect_err("expired plan must fail closed");
}

#[test]
fn invalid_restore_plan_is_rejected_before_private_path_resolution() {
    // [CHAT-RELAY-RESTORE-COMMAND-DOMAIN 2026-08-27 by Codex] The public
    // contract is untrusted input. Reject it before resolving the backup
    // directory so malformed traffic cannot create host-local artifacts.
    let directory = tempfile::tempdir().expect("restore validation directory");
    let unresolved_parent = directory.path().join("must-remain-absent");
    let mut config = test_config();
    config.db_path = unresolved_parent
        .join("relay.sqlite")
        .to_string_lossy()
        .into_owned();
    let invalid = ChatRelayRestorePlanReceipt {
        version: 0,
        issued_at: 1_000,
        expires_at: 1_600,
        verified_backup_count: 0,
        selected_backup_bytes: 0,
        active_database_present: false,
        active_database_bytes: 0,
        nonce: "00".repeat(CHAT_RELAY_RESTORE_PLAN_NONCE_BYTES),
        commitment: "00".repeat(32),
    };

    ChatRelayService::verify_latest_restore_plan_at(&config, &[0x42u8; 32], &invalid, 1_000)
        .expect_err("invalid public plan must fail before path resolution");
    assert!(
        !unresolved_parent.exists(),
        "invalid plan must not create the configured custody parent"
    );
}

#[test]
fn restore_plan_rejects_private_state_drift_even_when_size_is_unchanged() {
    let directory = tempfile::tempdir().expect("restore plan drift directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config.clone());
    source
        .create_verified_backup_for_operation("restore-plan-drift")
        .expect("create verified restore-plan drift image");
    drop(source);
    remove_test_db_sidecars(&source_path);

    let secret = derive_node_secret(&[0x42u8; 32]);
    let issued_at = now_secs();
    let plan = ChatRelayService::create_latest_restore_plan_at(&config, &secret, issued_at)
        .expect("create state-bound plan");
    let active_bytes = std::fs::read(&source_path).expect("read active custody bytes");
    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&source_path, &active_bytes).expect("rewrite same active custody bytes");
    assert_eq!(
        std::fs::metadata(&source_path)
            .expect("inspect rewritten custody")
            .len(),
        plan.active_database_bytes
    );

    ChatRelayService::verify_latest_restore_plan_at(&config, &secret, &plan, issued_at)
        .expect_err("private metadata drift must invalidate the plan");
}

#[test]
fn verified_backup_rejects_inconsistent_usage_without_partial_artifact() {
    let directory = tempfile::tempdir().expect("counter mismatch backup directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config);
    let identity = IdentityKeyPair::generate();
    source
        .store_pending(&make_envelope(&identity, [0x8Cu8; 32]))
        .expect("store canonical custody row");
    source
        .conn
        .lock()
        .execute(
            "UPDATE relay_storage_usage
             SET pending_message_count = 0, pending_message_bytes = 0
             WHERE singleton = 1",
            [],
        )
        .expect("tamper derived usage counters");

    let error = source
        .create_verified_backup()
        .expect_err("inconsistent backup must fail closed");
    assert!(matches!(
        error,
        ChatRelayError::CorruptStoredData {
            field: "sqlite_backup_logical_integrity"
        }
    ));
    let backup_directory = source_path.parent().unwrap().join(".aeronyx-relay-backups");
    assert_eq!(
        std::fs::read_dir(&backup_directory)
            .expect("inspect private backup directory")
            .count(),
        0,
        "failed certification must remove all partial artifacts"
    );

    drop(source);
    remove_test_db(&source_path);
}

#[test]
fn verified_backup_rejects_invalid_blind_route_claim_metadata() {
    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Backup
    // certification enforces the same process-fencing and lease shape as
    // startup; malformed ownership must not enter a trusted recovery image.
    let directory = tempfile::tempdir().expect("claim metadata backup directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config);
    source
        .reserve_blind_relay_route(&[0x91; 16], &[0x92; 32])
        .expect("reserve blind-route claim before backup corruption");
    source
        .conn
        .lock()
        .execute(
            "UPDATE relay_blind_route_reservations
             SET owner_epoch = '0000000000000000',
                 owner_acquired_at = 'not-an-integer'",
            [],
        )
        .expect("install malformed claim metadata fixture");

    let error = source
        .create_verified_backup()
        .expect_err("malformed claim backup must fail closed");
    assert!(
        matches!(
            error,
            ChatRelayError::CorruptStoredData {
                field: "sqlite_backup_logical_integrity"
            }
        ),
        "unexpected backup rejection: {error:?}"
    );
    let backup_directory = source_path.parent().unwrap().join(".aeronyx-relay-backups");
    assert_eq!(
        std::fs::read_dir(&backup_directory)
            .expect("inspect private backup directory")
            .count(),
        0,
        "failed claim certification must remove all partial artifacts"
    );

    drop(source);
    remove_test_db(&source_path);
}

#[test]
fn verified_backup_rejects_in_memory_storage() {
    let error = make_service()
        .create_verified_backup()
        .expect_err("in-memory relay must not escape its storage boundary");
    assert_eq!(error.reason_bucket(), "sqlite_error");
    assert!(error
        .to_string()
        .contains("in-memory relay storage has no private backup boundary"));
}

#[test]
fn audited_backup_replay_reuses_one_verified_artifact_across_restart() {
    // [CHAT-RELAY-BACKUP-IDEMPOTENCY 2026-08-16 by Codex] The durable
    // artifact key depends only on the stable node secret and opaque
    // operation ID. Process-local command deduplication is not involved.
    let directory = tempfile::tempdir().expect("idempotent backup directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let secret = [0x91u8; 32];

    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    let first = source
        .create_verified_backup_for_operation("cms-command-42")
        .expect("create audited backup");
    assert!(first.created);
    assert!(first.size_bytes > 0);
    drop(source);

    let restarted = ChatRelayService::new(config, secret).expect("restart relay store");
    let replay = restarted
        .create_verified_backup_for_operation("cms-command-42")
        .expect("reuse verified backup after restart");
    assert!(!replay.created);
    assert_eq!(replay.size_bytes, first.size_bytes);

    let second = restarted
        .create_verified_backup_for_operation("cms-command-43")
        .expect("different operation creates a distinct backup");
    assert!(second.created);
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect operation artifacts")
            .count(),
        2
    );
}

#[test]
fn concurrent_audited_backup_replay_publishes_exactly_once() {
    let directory = tempfile::tempdir().expect("concurrent backup directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = Arc::new(
        ChatRelayService::new(config, [0x93u8; 32]).expect("create concurrent relay store"),
    );
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let source = Arc::clone(&source);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                source
                    .create_verified_backup_for_operation("cms-command-concurrent")
                    .expect("complete concurrent audited backup")
            })
        })
        .collect();
    barrier.wait();
    let receipts: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().expect("join backup worker"))
        .collect();

    assert_eq!(receipts.iter().filter(|receipt| receipt.created).count(), 1);
    assert_eq!(receipts[0].size_bytes, receipts[1].size_bytes);
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect concurrent operation artifacts")
            .count(),
        1
    );
}

#[test]
fn audited_backup_replay_rejects_corrupt_existing_artifact_without_overwrite() {
    let directory = tempfile::tempdir().expect("corrupt replay directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = ChatRelayService::new(config, [0x92u8; 32]).expect("create relay store");
    source
        .create_verified_backup_for_operation("cms-command-corrupt")
        .expect("create audited backup");
    let backup_directory = directory.path().join(".aeronyx-relay-backups");
    let artifact = std::fs::read_dir(&backup_directory)
        .expect("read audited backup directory")
        .next()
        .expect("one audited backup")
        .expect("valid directory entry")
        .path();
    std::fs::write(&artifact, b"corrupt-replay-fixture").expect("corrupt backup fixture");

    let error = source
        .create_verified_backup_for_operation("cms-command-corrupt")
        .expect_err("corrupt replay artifact must fail closed");
    assert_eq!(error.reason_bucket(), "corrupt_stored_data");
    assert_eq!(
        std::fs::read(&artifact).expect("read preserved corrupt artifact"),
        b"corrupt-replay-fixture"
    );
    assert_eq!(
        std::fs::read_dir(&backup_directory)
            .expect("inspect preserved backup directory")
            .count(),
        1,
        "replay must not replace or duplicate a corrupt artifact"
    );
}

#[test]
fn audited_backup_replay_rejects_mutable_sidecar_state() {
    let directory = tempfile::tempdir().expect("sidecar replay directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = ChatRelayService::new(config, [0x94u8; 32]).expect("create relay store");
    source
        .create_verified_backup_for_operation("cms-command-sidecar")
        .expect("create audited backup");
    let backup_directory = directory.path().join(".aeronyx-relay-backups");
    let artifact = std::fs::read_dir(&backup_directory)
        .expect("read audited backup directory")
        .next()
        .expect("one audited backup")
        .expect("valid directory entry")
        .path();
    let mut wal_path = artifact.as_os_str().to_os_string();
    wal_path.push("-wal");
    std::fs::write(PathBuf::from(wal_path), b"mutable-sidecar-fixture")
        .expect("install mutable sidecar fixture");

    let error = source
        .create_verified_backup_for_operation("cms-command-sidecar")
        .expect_err("mutable sidecar state must fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");
    assert!(error.to_string().contains("mutable sidecar state"));
}

#[test]
fn audited_backup_rejects_unbounded_operation_ids_before_storage_access() {
    let source = make_service();
    for operation_id in ["".to_string(), "x".repeat(129)] {
        let error = source
            .create_verified_backup_for_operation(&operation_id)
            .expect_err("invalid operation ID must fail closed");
        assert_eq!(error.reason_bucket(), "sqlite_error");
        if !operation_id.is_empty() {
            assert!(!error.to_string().contains(&operation_id));
        }
    }
}

#[test]
fn backup_retention_audit_reports_excess_without_deleting_artifacts() {
    // [CHAT-RELAY-BACKUP-RETENTION 2026-08-16 by Codex] Retention is a
    // local read-only decision aid until an explicitly-authorized deletion
    // command exists. The audit must never make policy irreversible.
    let directory = tempfile::tempdir().expect("retention audit directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 2;
    config.custody_backup_retention_target_bytes = u64::MAX;
    let source = ChatRelayService::new(config, [0x95u8; 32]).expect("create relay store");
    for operation in ["retention-1", "retention-2", "retention-3"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create audited recovery image");
    }

    let receipt = source
        .audit_verified_backup_retention()
        .expect("audit verified backup retention");
    assert_eq!(receipt.retained_count, 2);
    assert_eq!(receipt.excess_count, 1);
    assert!(receipt.retained_bytes > 0);
    assert!(receipt.excess_bytes > 0);
    assert!(receipt.budget_exceeded);
    assert_eq!(receipt.partial_count, 0);
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect untouched recovery images")
            .count(),
        3,
        "read-only retention audit must not remove excess artifacts"
    );
}

#[test]
fn backup_retention_audit_keeps_one_recovery_point_over_byte_budget() {
    let directory = tempfile::tempdir().expect("retention byte budget directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 8;
    config.custody_backup_retention_target_bytes = 1;
    let source = ChatRelayService::new(config, [0x96u8; 32]).expect("create relay store");
    source
        .create_verified_backup_for_operation("retention-byte-1")
        .expect("create first recovery image");
    source
        .create_verified_backup_for_operation("retention-byte-2")
        .expect("create second recovery image");

    let receipt = source
        .audit_verified_backup_retention()
        .expect("audit byte-limited retention");
    assert_eq!(receipt.retained_count, 1);
    assert_eq!(receipt.excess_count, 1);
    assert!(receipt.retained_bytes > 1);
    assert!(receipt.budget_exceeded);
}

#[test]
fn backup_retention_audit_reports_private_partial_without_removing_it() {
    let directory = tempfile::tempdir().expect("retention partial directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = ChatRelayService::new(config, [0x97u8; 32]).expect("create relay store");
    source
        .create_verified_backup_for_operation("retention-partial")
        .expect("create recovery image");
    let partial = directory
        .path()
        .join(".aeronyx-relay-backups")
        .join(".relay-custody-1800000000-0123456789abcdef.tmp");
    std::fs::write(&partial, b"interrupted-private-snapshot")
        .expect("install interrupted snapshot fixture");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600))
            .expect("restrict partial fixture");
    }

    let receipt = source
        .audit_verified_backup_retention()
        .expect("audit interrupted private snapshot");
    assert_eq!(receipt.partial_count, 1);
    assert_eq!(receipt.partial_bytes, 28);
    assert!(partial.exists(), "read-only audit must preserve partials");
}

#[test]
fn backup_retention_audit_rejects_unmanaged_entries_without_side_effects() {
    let directory = tempfile::tempdir().expect("retention unmanaged directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = ChatRelayService::new(config, [0x98u8; 32]).expect("create relay store");
    source
        .create_verified_backup_for_operation("retention-unmanaged")
        .expect("create recovery image");
    let unmanaged = directory
        .path()
        .join(".aeronyx-relay-backups")
        .join("operator-note.txt");
    std::fs::write(&unmanaged, b"do not interpret").expect("install unmanaged fixture");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&unmanaged, std::fs::Permissions::from_mode(0o600))
            .expect("restrict unmanaged fixture");
    }

    let error = source
        .audit_verified_backup_retention()
        .expect_err("unmanaged entry must make audit fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");
    assert!(unmanaged.exists());
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect preserved directory")
            .count(),
        2
    );
}

#[test]
fn backup_retention_audit_rejects_corrupt_managed_artifact_without_mutation() {
    let directory = tempfile::tempdir().expect("retention corrupt directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = ChatRelayService::new(config, [0x99u8; 32]).expect("create relay store");
    source
        .create_verified_backup_for_operation("retention-corrupt")
        .expect("create recovery image");
    let backup_directory = directory.path().join(".aeronyx-relay-backups");
    let artifact = std::fs::read_dir(&backup_directory)
        .expect("read recovery directory")
        .next()
        .expect("one recovery image")
        .expect("valid recovery entry")
        .path();
    std::fs::write(&artifact, b"corrupt-retention-fixture")
        .expect("corrupt managed recovery fixture");

    let error = source
        .audit_verified_backup_retention()
        .expect_err("corrupt managed recovery image must fail audit");
    assert_eq!(error.reason_bucket(), "corrupt_stored_data");
    assert_eq!(
        std::fs::read(&artifact).expect("read preserved corrupt fixture"),
        b"corrupt-retention-fixture"
    );
    assert_eq!(
        std::fs::read_dir(&backup_directory)
            .expect("inspect preserved corrupt directory")
            .count(),
        1
    );
}

#[test]
fn backup_prune_dry_run_plans_without_deleting_and_writes_private_audit() {
    // [CHAT-RELAY-BACKUP-PRUNE 2026-08-16 by Codex] Dry-run is the default
    // operator experience and must leave every recovery artifact intact.
    let directory = tempfile::tempdir().expect("backup prune dry-run directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 2;
    config.custody_backup_retention_target_bytes = u64::MAX;
    let secret = [0xa1u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["prune-dry-1", "prune-dry-2", "prune-dry-3"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create recovery image");
    }
    let partial = directory
        .path()
        .join(".aeronyx-relay-backups")
        .join(".relay-custody-1800000000-0123456789abcdef.tmp");
    std::fs::write(&partial, b"interrupted-private-snapshot")
        .expect("install interrupted snapshot fixture");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&partial, std::fs::Permissions::from_mode(0o600))
            .expect("restrict partial fixture");
    }

    let receipt = ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs() + config.custody_backup_partial_grace_secs + 1,
    )
    .expect("complete retention dry-run");
    assert!(!receipt.executed);
    assert_eq!(receipt.planned_backup_count, 1);
    assert_eq!(receipt.planned_partial_count, 1);
    assert_eq!(receipt.deleted_backup_count, 0);
    assert_eq!(receipt.deleted_partial_count, 0);
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect dry-run artifacts")
            .count(),
        4
    );
    let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
    let mut audit = ChatRelayService::open_private_backup_control_file(&audit_path, true)
        .expect("open private maintenance audit");
    assert_eq!(
        ChatRelayService::verify_backup_audit_log(&mut audit, &secret)
            .expect("verify maintenance audit")
            .receipt()
            .record_count,
        1
    );
    drop(audit);

    // [CHAT-RELAY-AUDIT-VERIFY 2026-08-16 by Codex] Pin the public
    // aggregate contract and prove that verification uses this node's
    // secret without exposing the chain MAC or private filesystem state.
    let verification =
        ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
            .expect("verify public maintenance audit contract");
    assert!(verification.verified);
    assert_eq!(verification.record_count, 1);
    assert!(verification.last_recorded_at.is_some());
    assert_eq!(verification.dry_run_count, 1);
    assert_eq!(verification.planned_count, 0);
    assert_eq!(verification.completed_count, 0);
    assert_eq!(verification.failed_count, 0);
    assert!(verification.verified_bytes > 0);

    let json = serde_json::to_value(verification).expect("serialize audit verification");
    let object = json.as_object().expect("verification JSON is an object");
    assert_eq!(object.len(), 13);
    for field in [
        "verified",
        "record_count",
        "last_recorded_at",
        "dry_run_count",
        "planned_count",
        "completed_count",
        "failed_count",
        "verified_bytes",
        "checkpoint_count",
        "archived_record_count",
        "active_record_count",
        "archived_bytes",
        "rotation_pending",
    ] {
        assert!(object.contains_key(field), "missing JSON field: {field}");
    }
    let encoded = serde_json::to_string(&json).expect("encode verification JSON");
    assert!(!encoded.contains(directory.path().to_string_lossy().as_ref()));
    assert!(!encoded.contains("previous_mac"));
    assert!(!encoded.contains("operation"));

    let error =
        ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &[0xffu8; 32])
            .expect_err("wrong node secret must not authenticate maintenance history");
    assert_eq!(error.reason_bucket(), "sqlite_error");

    let audit_record = std::fs::read_to_string(&audit_path).expect("read audit schema fixture");
    let extended = audit_record.replacen(
        "\"version\":1",
        "\"version\":1,\"uncommitted_extension\":true",
        1,
    );
    assert_ne!(extended, audit_record, "fixture must gain an unknown field");
    std::fs::write(&audit_path, extended).expect("write extended audit schema fixture");
    let error = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect_err("unknown audit fields must not bypass canonical authentication");
    assert_eq!(error.reason_bucket(), "sqlite_error");
}

#[test]
fn backup_audit_verification_accepts_absent_history_without_creating_it() {
    let directory = tempfile::tempdir().expect("empty backup audit directory");
    let mut config = test_config();
    config.db_path = directory
        .path()
        .join("source.sqlite")
        .to_string_lossy()
        .into_owned();
    let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);

    let receipt =
        ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &[0xb1u8; 32])
            .expect("verify absent maintenance history");

    assert_eq!(
        receipt,
        ChatRelayBackupAuditVerificationReceipt {
            verified: true,
            ..Default::default()
        }
    );
    assert!(
        !audit_path.exists(),
        "read-only verification must not initialize an audit history"
    );
}

#[test]
fn backup_audit_verification_rejects_oversized_and_truncated_records() {
    let directory = tempfile::tempdir().expect("bounded backup audit directory");
    let mut config = test_config();
    config.db_path = directory
        .path()
        .join("source.sqlite")
        .to_string_lossy()
        .into_owned();
    let secret = [0xb2u8; 32];
    ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect("initialize private maintenance boundary");
    let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);

    let mut oversized = vec![b'x'; CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES + 1];
    oversized.push(b'\n');
    std::fs::write(&audit_path, oversized).expect("write oversized audit record");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&audit_path, std::fs::Permissions::from_mode(0o600))
            .expect("restrict oversized audit fixture");
    }
    let error = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect_err("oversized audit record must fail before JSON parsing");
    assert_eq!(error.reason_bucket(), "sqlite_error");

    std::fs::write(&audit_path, b"{}").expect("replace fixture with unterminated audit record");
    let error = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect_err("truncated audit record must fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");
}

#[test]
fn backup_audit_rotation_preserves_global_chain_and_checkpoint_aggregates() {
    // [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] Rotation must not
    // reset the v1 record sequence or MAC chain. The checkpoint is an
    // authenticated boundary around immutable bytes, not a new genesis.
    let directory = tempfile::tempdir().expect("rotating backup audit directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 1;
    let secret = [0xb3u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["rotate-audit-1", "rotate-audit-2"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create rotation recovery image");
    }
    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs(),
    )
    .expect("write first active audit record");

    let backup_directory = ChatRelayService::private_backup_directory_for_config(&config)
        .expect("rotation backup boundary");
    let parent = backup_directory.parent().expect("rotation audit parent");
    {
        let _lock = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
            .expect("hold rotation lock");
        let chain = ChatRelayService::verify_backup_audit_chain(parent, &secret)
            .expect("verify pre-rotation chain");
        ChatRelayService::rotate_backup_audit_segment(parent, &secret, &chain.state)
            .expect("rotate active audit segment");
    }

    let rotated = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect("verify rotated maintenance chain");
    assert_eq!(rotated.record_count, 1);
    assert_eq!(rotated.checkpoint_count, 1);
    assert_eq!(rotated.archived_record_count, 1);
    assert_eq!(rotated.active_record_count, 0);
    assert_eq!(rotated.archived_bytes, rotated.verified_bytes);
    assert!(!rotated.rotation_pending);

    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs() + 1,
    )
    .expect("append after audit rotation");
    let continued = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect("verify continued maintenance chain");
    assert_eq!(continued.record_count, 2);
    assert_eq!(continued.dry_run_count, 2);
    assert_eq!(continued.checkpoint_count, 1);
    assert_eq!(continued.archived_record_count, 1);
    assert_eq!(continued.active_record_count, 1);
    assert!(continued.verified_bytes > continued.archived_bytes);
    assert!(!continued.rotation_pending);
}

#[test]
fn backup_audit_anchor_covers_only_complete_authenticated_checkpoints() {
    // [CUSTODY-AUDIT-ANCHOR 2026-08-16 by Codex] Portable evidence must
    // advance only at an immutable checkpoint. Active-tail writes cannot
    // silently change an already exportable generation, and either the
    // wrong node identity or a crash-window publication state fails closed.
    let directory = tempfile::tempdir().expect("anchored backup audit directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 1;
    let identity = IdentityKeyPair::from_bytes(&[0xb6; 32]).expect("anchor identity");
    let secret = derive_node_secret(&identity.to_bytes());
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["anchor-audit-1", "anchor-audit-2"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create anchor recovery fixture");
    }

    let no_checkpoint =
        ChatRelayService::create_backup_maintenance_audit_anchor_for_config(&config, &identity)
            .expect_err("active or absent audit cannot produce a portable anchor");
    assert_eq!(no_checkpoint.reason_bucket(), "sqlite_error");

    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        1_787_100_001,
    )
    .expect("write anchor audit record");
    let backup_directory = ChatRelayService::private_backup_directory_for_config(&config)
        .expect("anchor backup boundary");
    let parent = backup_directory.parent().expect("anchor audit parent");
    let range = ChatRelayBackupAuditSegmentRange {
        first_sequence: 1,
        last_sequence: 1,
    };
    let segment_path = parent.join(ChatRelayService::backup_audit_segment_file_name(range));
    let active_path = parent.join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
    {
        let _lock = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
            .expect("hold anchor rotation lock");
        let chain = ChatRelayService::verify_backup_audit_chain(parent, &secret)
            .expect("verify anchor precondition");
        ChatRelayService::rotate_backup_audit_segment(parent, &secret, &chain.state)
            .expect("publish anchor checkpoint");
    }

    let first =
        ChatRelayService::create_backup_maintenance_audit_anchor_for_config(&config, &identity)
            .expect("create portable checkpoint anchor");
    first
        .verify_expected(&identity.public_key_bytes(), 1)
        .expect("verify producer and rollback floor");
    assert_eq!(first.checkpoint_generation, 1);
    assert_eq!(first.archived_record_count, 1);
    assert!(first.archived_bytes > 0);

    // [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] The producer
    // import guard must bind the current anchor to the complete persistence
    // phase by excluding every concurrent checkpoint-producing operation.
    let guarded =
        ChatRelayService::hold_backup_maintenance_audit_anchor_for_config(&config, &identity)
            .expect("hold current custody anchor");
    assert_eq!(guarded.anchor(), &first);
    let concurrent = ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        1_787_100_002,
    )
    .expect_err("held current-anchor guard must exclude checkpoint maintenance");
    assert_eq!(concurrent.reason_bucket(), "sqlite_error");
    drop(guarded);

    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        1_787_100_003,
    )
    .expect("append uncheckpointed audit tail");
    let with_active_tail =
        ChatRelayService::create_backup_maintenance_audit_anchor_for_config(&config, &identity)
            .expect("anchor latest complete checkpoint");
    assert_eq!(with_active_tail.checkpoint_generation, 1);
    assert_eq!(with_active_tail.archived_record_count, 1);
    assert_eq!(with_active_tail.archived_bytes, first.archived_bytes);
    assert_eq!(with_active_tail.anchor_digest, first.anchor_digest);
    assert_eq!(
        with_active_tail, first,
        "one complete checkpoint must always export the same signed frame"
    );

    let wrong_identity = IdentityKeyPair::from_bytes(&[0xb7; 32]).expect("wrong identity");
    let wrong_key = ChatRelayService::create_backup_maintenance_audit_anchor_for_config(
        &config,
        &wrong_identity,
    )
    .expect_err("wrong node key cannot authenticate the private checkpoint");
    assert_eq!(wrong_key.reason_bucket(), "sqlite_error");

    std::fs::remove_file(&active_path).expect("remove active tail fixture");
    std::fs::rename(&segment_path, &active_path)
        .expect("reproduce interrupted segment publication");
    let pending =
        ChatRelayService::create_backup_maintenance_audit_anchor_for_config(&config, &identity)
            .expect_err("pending rotation cannot produce external evidence");
    assert_eq!(pending.reason_bucket(), "sqlite_error");
}

#[test]
fn backup_audit_rotation_capacity_boundaries_are_exact() {
    // [CHAT-RELAY-AUDIT-ROTATION 2026-08-16 by Codex] A segment may use
    // its final byte and final record slot; only the following append
    // rotates. This prevents both premature churn and an over-limit write.
    let record_bytes = CHAT_RELAY_BACKUP_AUDIT_MAX_RECORD_BYTES;
    assert!(!ChatRelayService::backup_audit_segment_needs_rotation(
        (CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS - 1) as u64,
        CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES - record_bytes as u64,
        record_bytes,
    )
    .expect("exact segment boundary is valid"));
    assert!(ChatRelayService::backup_audit_segment_needs_rotation(
        CHAT_RELAY_BACKUP_AUDIT_MAX_RECORDS as u64,
        0,
        1,
    )
    .expect("record boundary requires rotation"));
    assert!(ChatRelayService::backup_audit_segment_needs_rotation(
        1,
        CHAT_RELAY_BACKUP_AUDIT_MAX_BYTES - record_bytes as u64 + 1,
        record_bytes,
    )
    .expect("byte boundary requires rotation"));
}

#[test]
fn backup_audit_rotation_recovers_both_crash_publication_windows() {
    let directory = tempfile::tempdir().expect("recoverable backup audit directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 1;
    let secret = [0xb4u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["recover-audit-1", "recover-audit-2"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create recovery fixture");
    }
    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs(),
    )
    .expect("write initial audit record");
    let backup_directory = ChatRelayService::private_backup_directory_for_config(&config)
        .expect("recovery backup boundary");
    let parent = backup_directory.parent().expect("recovery audit parent");
    let first_range = ChatRelayBackupAuditSegmentRange {
        first_sequence: 1,
        last_sequence: 1,
    };
    let active_path = parent.join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
    let first_segment = parent.join(ChatRelayService::backup_audit_segment_file_name(
        first_range,
    ));
    {
        let _lock = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
            .expect("hold first rotation lock");
        let chain = ChatRelayService::verify_backup_audit_chain(parent, &secret)
            .expect("verify first active chain");
        ChatRelayService::rotate_backup_audit_segment(parent, &secret, &chain.state)
            .expect("publish first segment");
    }

    // Simulate power loss after checkpoint publication but before the
    // active name is linked to its immutable segment name.
    std::fs::rename(&first_segment, &active_path).expect("restore pending-publication active name");
    let pending_publish =
        ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
            .expect("verify pending segment publication");
    assert!(pending_publish.rotation_pending);
    assert_eq!(pending_publish.archived_record_count, 1);
    assert_eq!(pending_publish.active_record_count, 0);

    let abandoned_checkpoint = parent.join(format!(
        "{CHAT_RELAY_BACKUP_AUDIT_CHECKPOINT_TEMP_PREFIX}0123456789abcdef"
    ));
    ChatRelayService::reserve_private_backup_file(&abandoned_checkpoint)
        .expect("install abandoned checkpoint temporary");

    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs() + 1,
    )
    .expect("recover pending publication before append");
    assert!(first_segment.exists());
    assert!(
        !abandoned_checkpoint.exists(),
        "locked maintenance must remove abandoned checkpoint temporaries"
    );

    // Rotate the second active record, then reproduce the hard-link window
    // where both active and immutable names durably identify the segment.
    let second_range = ChatRelayBackupAuditSegmentRange {
        first_sequence: 2,
        last_sequence: 2,
    };
    let second_segment = parent.join(ChatRelayService::backup_audit_segment_file_name(
        second_range,
    ));
    {
        let _lock = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
            .expect("hold second rotation lock");
        let chain = ChatRelayService::verify_backup_audit_chain(parent, &secret)
            .expect("verify second active chain");
        ChatRelayService::rotate_backup_audit_segment(parent, &secret, &chain.state)
            .expect("publish second segment");
    }
    std::fs::hard_link(&second_segment, &active_path)
        .expect("reproduce duplicate active publication name");
    let pending_remove =
        ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
            .expect("verify duplicate active publication");
    assert!(pending_remove.rotation_pending);
    assert_eq!(pending_remove.checkpoint_count, 2);
    assert_eq!(pending_remove.archived_record_count, 2);

    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs() + 2,
    )
    .expect("remove duplicate active name before append");
    let recovered = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect("verify chain after both recovery windows");
    assert!(!recovered.rotation_pending);
    assert_eq!(recovered.record_count, 3);
    assert_eq!(recovered.archived_record_count, 2);
    assert_eq!(recovered.active_record_count, 1);
}

#[test]
fn backup_audit_rotation_rejects_checkpoint_and_segment_tampering() {
    let directory = tempfile::tempdir().expect("tampered rotated audit directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 1;
    let secret = [0xb5u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["tamper-rotation-1", "tamper-rotation-2"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create tamper recovery fixture");
    }
    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs(),
    )
    .expect("write tamper audit record");
    let backup_directory = ChatRelayService::private_backup_directory_for_config(&config)
        .expect("tamper backup boundary");
    let parent = backup_directory.parent().expect("tamper audit parent");
    let range = ChatRelayBackupAuditSegmentRange {
        first_sequence: 1,
        last_sequence: 1,
    };
    {
        let _lock = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
            .expect("hold tamper rotation lock");
        let chain = ChatRelayService::verify_backup_audit_chain(parent, &secret)
            .expect("verify tamper precondition");
        ChatRelayService::rotate_backup_audit_segment(parent, &secret, &chain.state)
            .expect("rotate tamper fixture");
    }
    let checkpoint_path = parent.join(ChatRelayService::backup_audit_checkpoint_file_name(range));
    let original_checkpoint = std::fs::read(&checkpoint_path).expect("read checkpoint fixture");
    let mut checkpoint: serde_json::Value =
        serde_json::from_slice(&original_checkpoint).expect("decode checkpoint fixture");
    checkpoint["segment_bytes"] = serde_json::json!(1);
    std::fs::write(
        &checkpoint_path,
        serde_json::to_vec(&checkpoint).expect("encode tampered checkpoint"),
    )
    .expect("tamper checkpoint fixture");
    let error = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect_err("tampered checkpoint must fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");

    std::fs::write(&checkpoint_path, original_checkpoint).expect("restore checkpoint fixture");
    let segment_path = parent.join(ChatRelayService::backup_audit_segment_file_name(range));
    let mut segment = std::fs::read(&segment_path).expect("read immutable segment fixture");
    let byte = segment.first_mut().expect("segment fixture is non-empty");
    *byte ^= 1;
    std::fs::write(&segment_path, segment).expect("tamper immutable segment fixture");
    let error = ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
        .expect_err("tampered immutable segment must fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");
}

#[test]
fn backup_prune_execution_requires_both_confirmations_without_side_effects() {
    let directory = tempfile::tempdir().expect("backup prune confirmation directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 1;
    let secret = [0xa2u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["prune-confirm-1", "prune-confirm-2"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create recovery image");
    }

    for request in [
        ChatRelayBackupPruneRequest {
            execute: true,
            confirmation: None,
            node_stopped_confirmed: true,
        },
        ChatRelayBackupPruneRequest {
            execute: true,
            confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
            node_stopped_confirmed: false,
        },
    ] {
        let error = ChatRelayService::prune_verified_backup_retention_at(
            &config,
            &secret,
            &request,
            now_secs(),
        )
        .expect_err("incomplete confirmation must fail closed");
        assert_eq!(error.reason_bucket(), "sqlite_error");
    }
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect confirmed artifacts")
            .count(),
        2
    );
    assert!(!directory
        .path()
        .join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME)
        .exists());
}

#[test]
fn backup_prune_deletes_only_excess_and_grace_expired_artifacts() {
    let directory = tempfile::tempdir().expect("backup prune execution directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 2;
    config.custody_backup_retention_target_bytes = u64::MAX;
    let secret = [0xa3u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["prune-execute-1", "prune-execute-2", "prune-execute-3"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create recovery image");
    }
    let backup_directory = directory.path().join(".aeronyx-relay-backups");
    let stale_partial = backup_directory.join(".relay-custody-1-0123456789abcdef.tmp");
    let fresh_partial = backup_directory.join(".relay-custody-2-fedcba9876543210.tmp");
    for partial in [&stale_partial, &fresh_partial] {
        std::fs::write(partial, b"private-partial").expect("install partial fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(partial, std::fs::Permissions::from_mode(0o600))
                .expect("restrict partial fixture");
        }
    }
    let now = now_secs();
    let stale_now = now + config.custody_backup_partial_grace_secs + 1;
    let request = ChatRelayBackupPruneRequest {
        execute: true,
        confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
        node_stopped_confirmed: true,
    };

    // Both partial fixtures have the same current mtime, so first prove a
    // normal execution treats neither as stale.
    let fresh_receipt =
        ChatRelayService::prune_verified_backup_retention_at(&config, &secret, &request, now)
            .expect("prune excess complete backup only");
    assert_eq!(fresh_receipt.deleted_backup_count, 1);
    assert_eq!(fresh_receipt.deleted_partial_count, 0);
    assert!(stale_partial.exists());
    assert!(fresh_partial.exists());

    // Advance the internal test clock beyond the mandatory grace period.
    let stale_receipt =
        ChatRelayService::prune_verified_backup_retention_at(&config, &secret, &request, stale_now)
            .expect("prune grace-expired partials");
    assert_eq!(stale_receipt.deleted_backup_count, 0);
    assert_eq!(stale_receipt.deleted_partial_count, 2);
    assert_eq!(stale_receipt.remaining.retained_count, 2);
    assert_eq!(stale_receipt.remaining.excess_count, 0);
    assert!(!stale_partial.exists());
    assert!(!fresh_partial.exists());

    let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
    let mut audit = ChatRelayService::open_private_backup_control_file(&audit_path, true)
        .expect("open private maintenance audit");
    assert_eq!(
        ChatRelayService::verify_backup_audit_log(&mut audit, &secret)
            .expect("verify maintenance audit")
            .receipt()
            .record_count,
        4,
        "two successful executions write planned and completed records"
    );
    drop(audit);
    let verification =
        ChatRelayService::verify_backup_maintenance_audit_for_config(&config, &secret)
            .expect("verify completed maintenance history");
    assert_eq!(verification.planned_count, 2);
    assert_eq!(verification.completed_count, 2);
    assert_eq!(verification.dry_run_count, 0);
    assert_eq!(verification.failed_count, 0);
}

#[test]
fn backup_prune_rejects_tampered_audit_before_deletion() {
    let directory = tempfile::tempdir().expect("backup prune tamper directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    config.custody_backup_retention_target_artifacts = 1;
    let secret = [0xa4u8; 32];
    let source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    for operation in ["prune-tamper-1", "prune-tamper-2"] {
        source
            .create_verified_backup_for_operation(operation)
            .expect("create recovery image");
    }
    ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs(),
    )
    .expect("write dry-run audit");
    let audit_path = directory.path().join(CHAT_RELAY_BACKUP_AUDIT_FILE_NAME);
    let encoded = std::fs::read_to_string(&audit_path).expect("read audit fixture");
    std::fs::write(&audit_path, encoded.replace("dry_run", "completed"))
        .expect("tamper audit fixture");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&audit_path, std::fs::Permissions::from_mode(0o600))
            .expect("restore private audit permissions");
    }
    let request = ChatRelayBackupPruneRequest {
        execute: true,
        confirmation: Some(CHAT_RELAY_BACKUP_PRUNE_CONFIRMATION.to_string()),
        node_stopped_confirmed: true,
    };
    let error = ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &request,
        now_secs(),
    )
    .expect_err("tampered audit must block deletion");
    assert_eq!(error.reason_bucket(), "sqlite_error");
    assert_eq!(
        std::fs::read_dir(directory.path().join(".aeronyx-relay-backups"))
            .expect("inspect preserved recovery images")
            .count(),
        2
    );
}

#[test]
fn backup_prune_fails_closed_while_cross_process_lock_is_held() {
    let directory = tempfile::tempdir().expect("backup prune lock directory");
    let source_path = directory.path().join("source.sqlite");
    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let secret = [0xa5u8; 32];
    let _source = ChatRelayService::new(config.clone(), secret).expect("create relay store");
    let backup_directory =
        ChatRelayService::private_backup_directory_for_config(&config).expect("backup boundary");
    let _held = ChatRelayService::acquire_backup_filesystem_lock(&backup_directory)
        .expect("hold maintenance lock");

    let error = ChatRelayService::prune_verified_backup_retention_at(
        &config,
        &secret,
        &ChatRelayBackupPruneRequest::default(),
        now_secs(),
    )
    .expect_err("held lock must block concurrent maintenance");
    assert_eq!(error.reason_bucket(), "sqlite_error");
}

#[cfg(unix)]
#[test]
fn verified_backup_rejects_symlinked_storage_boundary() {
    use std::os::unix::fs::symlink;

    // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] A process with
    // access to the database parent must not redirect ciphertext custody
    // through a pre-created symlink before an operator backup runs.
    let directory = tempfile::tempdir().expect("symlink backup boundary directory");
    let source_path = directory.path().join("source.sqlite");
    let outside = directory.path().join("redirect-target");
    std::fs::create_dir(&outside).expect("create redirect target");
    symlink(&outside, directory.path().join(".aeronyx-relay-backups"))
        .expect("install backup boundary symlink");

    let mut config = test_config();
    config.db_path = source_path.to_string_lossy().into_owned();
    let source = make_service_with_config(config);
    let error = source
        .create_verified_backup()
        .expect_err("symlinked backup boundary must fail closed");
    assert_eq!(error.reason_bucket(), "sqlite_error");
    assert_eq!(
        std::fs::read_dir(&outside)
            .expect("inspect redirect target")
            .count(),
        0
    );
}

#[test]
fn test_peer_relay_outbound_health_tracks_failure_and_recovery() {
    let svc = make_service();

    svc.record_peer_relay_outbound(
        1_800_000_010,
        2,
        1,
        Some("peer_relay_request_timeout".to_string()),
    );
    let status = svc.peer_status();
    assert_eq!(status.last_outbound_status.as_deref(), Some("degraded"));
    assert_eq!(status.last_outbound_attempted, 2);
    assert_eq!(status.last_outbound_accepted, 1);
    assert_eq!(status.last_outbound_failed, 1);
    assert_eq!(status.consecutive_outbound_failures, 0);
    assert_eq!(status.last_outbound_success_at, Some(1_800_000_010));

    svc.record_peer_relay_outbound(1_800_000_020, 1, 0, Some("peer_relay_http_503".to_string()));
    let status = svc.peer_status();
    assert_eq!(status.last_outbound_status.as_deref(), Some("failed"));
    assert_eq!(
        status.last_outbound_failure_reason.as_deref(),
        Some("peer_relay_http_503")
    );
    assert_eq!(status.consecutive_outbound_failures, 1);

    svc.record_peer_relay_outbound(1_800_000_030, 1, 1, None);
    let status = svc.peer_status();
    assert_eq!(status.last_outbound_status.as_deref(), Some("healthy"));
    assert_eq!(status.last_outbound_failure_reason, None);
    assert_eq!(status.consecutive_outbound_failures, 0);
    assert_eq!(status.last_outbound_success_at, Some(1_800_000_030));
    assert_eq!(status.direct_peer_outbound.rounds, 3);
    assert_eq!(
        status.direct_peer_outbound.last_status.as_deref(),
        Some("healthy")
    );
    assert_eq!(status.authenticated_onion_outbound.rounds, 0);
}

#[test]
fn relay_health_reason_boundary_preserves_buckets_and_redacts_raw_input() {
    let svc = make_service();

    // [RELAY-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] The legacy API
    // remains callable during rolling upgrades, but neither a URL nor an
    // invalid status suffix may become heartbeat-visible text.
    svc.record_peer_relay_outbound(
        1_800_000_031,
        1,
        0,
        Some("https://peer.example/secret?message_id=42".to_string()),
    );
    assert_eq!(
        svc.peer_status().last_outbound_failure_reason.as_deref(),
        Some("unknown")
    );

    svc.record_peer_relay_outbound(1_800_000_032, 1, 0, Some("peer_relay_http_503".to_string()));
    assert_eq!(
        svc.peer_status().last_outbound_failure_reason.as_deref(),
        Some("peer_relay_http_503")
    );

    svc.record_peer_relay_outbound(1_800_000_033, 1, 0, Some("peer_relay_http_999".to_string()));
    assert_eq!(
        svc.peer_status().last_outbound_failure_reason.as_deref(),
        Some("unknown")
    );

    svc.record_peer_relay_inbound_rejected(
        1_800_000_034,
        "invalid_signature receiver=private-key-material",
    );
    assert_eq!(
        svc.peer_status().last_inbound_failure_reason.as_deref(),
        Some("unknown")
    );

    svc.record_peer_relay_inbound_rejected(1_800_000_035, "invalid_signature");
    assert_eq!(
        svc.peer_status().last_inbound_failure_reason.as_deref(),
        Some("invalid_signature")
    );
}

#[test]
fn authenticated_onion_health_survives_direct_fallback_result() {
    let svc = make_service();

    // [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] This reproduces the
    // production order: receipt-verified onion fails, then compatibility
    // direct relay succeeds for availability. Aggregate remains backward
    // compatible while the authenticated proof keeps its true result.
    svc.record_authenticated_onion_outbound(
        1_800_000_040,
        1,
        0,
        Some("onion_delivery_receipt_rejected".to_string()),
    );
    svc.record_peer_relay_outbound(1_800_000_041, 1, 1, None);

    let status = svc.peer_status();
    assert_eq!(status.outbound_rounds, 2);
    assert_eq!(status.last_outbound_status.as_deref(), Some("healthy"));
    assert_eq!(status.authenticated_onion_outbound.rounds, 1);
    assert_eq!(
        status.authenticated_onion_outbound.last_status.as_deref(),
        Some("failed")
    );
    assert_eq!(
        status
            .authenticated_onion_outbound
            .last_failure_reason
            .as_deref(),
        Some("onion_delivery_receipt_rejected")
    );
    assert_eq!(status.direct_peer_outbound.rounds, 1);
    assert_eq!(
        status.direct_peer_outbound.last_status.as_deref(),
        Some("healthy")
    );
}

#[test]
fn verified_submit_health_tracks_closed_result_vocabulary() {
    let svc = make_service();

    svc.record_verified_submit_result(1_800_000_060, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    svc.record_verified_submit_result(1_800_000_061, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1);
    svc.record_verified_submit_result(1_800_000_062, CHAT_VERIFIED_SUBMIT_ONION_ONLY_V1);
    svc.record_verified_submit_result(1_800_000_063, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1);
    svc.record_verified_submit_result(1_800_000_064, u8::MAX);
    svc.record_verified_submit_replay(1_800_000_065, CHAT_VERIFIED_SUBMIT_ONION_AND_ENTRY_V1);
    svc.record_verified_submit_conflict(1_800_000_066, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    svc.record_verified_submit_pending_rejection(1_800_000_067, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    svc.record_verified_submit_capacity_rejection(1_800_000_068, CHAT_VERIFIED_SUBMIT_REJECTED_V1);
    svc.record_verified_submit_recovery_attempted(1_800_000_069);
    svc.record_verified_submit_recovery_outcome(
        1_800_000_070,
        VerifiedSubmitRecoveryOutcome::Completed,
    );
    svc.record_verified_submit_recovery_attempted(1_800_000_071);
    svc.record_verified_submit_recovery_outcome(
        1_800_000_072,
        VerifiedSubmitRecoveryOutcome::Failed,
    );
    svc.record_verified_submit_recovery_attempted(1_800_000_073);
    svc.record_verified_submit_recovery_outcome(
        1_800_000_074,
        VerifiedSubmitRecoveryOutcome::Deferred,
    );

    let status = svc.peer_status().verified_submit;
    assert_eq!(status.total, 9);
    assert_eq!(status.rejected_total, 4);
    assert_eq!(status.entry_retry_total, 1);
    assert_eq!(status.onion_only_total, 1);
    assert_eq!(status.onion_and_entry_total, 2);
    assert_eq!(status.unknown_result_total, 1);
    assert_eq!(status.replayed_total, 1);
    assert_eq!(status.request_conflict_total, 1);
    assert_eq!(status.pending_rejected_total, 1);
    assert_eq!(status.capacity_rejected_total, 1);
    assert_eq!(status.entry_recovery.attempted_total, 3);
    assert_eq!(status.entry_recovery.completed_total, 1);
    assert_eq!(status.entry_recovery.failed_total, 1);
    assert_eq!(status.entry_recovery.deferred_total, 1);
    assert_eq!(
        status.entry_recovery.last_outcome.as_deref(),
        Some("deferred")
    );
    assert_eq!(status.entry_recovery.last_event_at, Some(1_800_000_074));
    assert_eq!(status.last_result.as_deref(), Some("rejected"));
    assert_eq!(status.last_at, Some(1_800_000_068));
}

#[test]
fn verified_submit_recovery_outcome_truth_table_is_closed() {
    assert_eq!(
        VerifiedSubmitRecoveryOutcome::from_results(true, true),
        VerifiedSubmitRecoveryOutcome::Completed
    );
    assert_eq!(
        VerifiedSubmitRecoveryOutcome::from_results(false, true),
        VerifiedSubmitRecoveryOutcome::Failed
    );
    assert_eq!(
        VerifiedSubmitRecoveryOutcome::from_results(true, false),
        VerifiedSubmitRecoveryOutcome::Deferred
    );
    assert_eq!(
        VerifiedSubmitRecoveryOutcome::from_results(false, false),
        VerifiedSubmitRecoveryOutcome::Deferred
    );
}

#[test]
fn verified_submit_response_replays_privately_across_restart() {
    // [DURABLE-VERIFIED-SUBMIT-IDEMPOTENCY 2026-08-24 by Codex] Exercise
    // the real SQLite boundary. The durable row exposes only fixed-size
    // HMACs and sealed response bytes, while exact replay and envelope
    // conflict survive process-local cache loss.
    let db_path = unique_test_db_path("verified-submit-restart");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xC1; 32];
    let sender = IdentityKeyPair::generate();
    let envelope = make_envelope(&sender, [0xC2; 32]);
    let request =
        ChatRelayVerifiedSubmitRequestV1::signed([0xC3; 16], envelope.clone(), now_secs(), &sender)
            .expect("sign verified submit request");
    let response = ChatRelayVerifiedSubmitResponseV1::rejected(
        request.request_id,
        request.envelope.message_id,
    );

    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("create durable verified submit relay");
        assert_eq!(
            service
                .reserve_verified_submit(&request)
                .expect("reserve verified submit response"),
            VerifiedSubmitAdmission::Reserved
        );
        service
            .remember_verified_submit_response(&request, &response)
            .expect("persist verified submit response");
        let conn = service.conn.lock();
        let (cache_key, envelope_fingerprint, nonce, ciphertext) = conn
            .query_row(
                "SELECT cache_key, envelope_fingerprint, response_nonce,
                        response_ciphertext
                 FROM relay_verified_submit_responses",
                [],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                },
            )
            .expect("read private durable replay row");
        assert_eq!(cache_key.len(), 32);
        assert_eq!(envelope_fingerprint.len(), 32);
        assert_eq!(
            nonce.len(),
            crate::services::chat_relay_verified_submit::RESPONSE_NONCE_BYTES
        );
        for sensitive in [
            request.request_id.as_slice(),
            request.envelope.message_id.as_slice(),
            request.envelope.sender.as_slice(),
        ] {
            assert!(
                !ciphertext
                    .windows(sensitive.len())
                    .any(|window| window == sensitive),
                "sealed durable response must not expose request metadata"
            );
        }
    }

    let restarted =
        ChatRelayService::new(config, secret).expect("restart durable verified submit relay");
    let replayed = restarted
        .verified_submit_cache_lookup(&request)
        .expect("recover durable verified submit response");
    let VerifiedSubmitCacheLookup::Exact(replayed) = replayed else {
        panic!("exact verified submit must replay after restart");
    };
    assert_eq!(replayed, response);

    let conflicting_envelope = make_envelope(&sender, [0xC4; 32]);
    let conflicting_request = ChatRelayVerifiedSubmitRequestV1::signed(
        request.request_id,
        conflicting_envelope,
        now_secs(),
        &sender,
    )
    .expect("sign conflicting verified submit request");
    assert!(matches!(
        restarted
            .verified_submit_cache_lookup(&conflicting_request)
            .expect("classify durable request conflict"),
        VerifiedSubmitCacheLookup::Conflict
    ));

    drop(restarted);
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_response_capacity_rejects_without_eviction_and_ttl_cleans() {
    let db_path = unique_test_db_path("verified-submit-retention");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    config.dedup_lru_capacity = 1;
    let secret = [0xD1; 32];
    let sender = IdentityKeyPair::generate();
    let service = ChatRelayService::new(config.clone(), secret)
        .expect("create bounded verified submit relay");

    let first_request = ChatRelayVerifiedSubmitRequestV1::signed(
        [0xD2; 16],
        make_envelope(&sender, [0xD4; 32]),
        now_secs(),
        &sender,
    )
    .expect("sign first bounded verified submit request");
    let first_response = ChatRelayVerifiedSubmitResponseV1::rejected(
        first_request.request_id,
        first_request.envelope.message_id,
    );
    assert_eq!(
        service
            .reserve_verified_submit(&first_request)
            .expect("reserve first bounded verified submit request"),
        VerifiedSubmitAdmission::Reserved
    );
    service
        .remember_verified_submit_response(&first_request, &first_response)
        .expect("persist first bounded verified submit response");

    let second_request = ChatRelayVerifiedSubmitRequestV1::signed(
        [0xD3; 16],
        make_envelope(&sender, [0xD5; 32]),
        now_secs(),
        &sender,
    )
    .expect("sign second bounded verified submit request");
    assert_eq!(
        service
            .reserve_verified_submit(&second_request)
            .expect("classify saturated verified submit request"),
        VerifiedSubmitAdmission::CapacityExhausted
    );
    assert!(matches!(
        service
            .verified_submit_cache_lookup(&first_request)
            .expect("replay retained first response"),
        VerifiedSubmitCacheLookup::Exact(response) if response == first_response
    ));
    assert_eq!(
        service
            .conn
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM relay_verified_submit_responses",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("count bounded durable replay rows"),
        1
    );
    service
        .conn
        .lock()
        .execute(
            "UPDATE relay_verified_submit_responses SET completed_at = 0",
            [],
        )
        .expect("age durable replay row");
    let (summary, failure) = service.run_cleanup_at(
        i64::try_from(VERIFIED_SUBMIT_RESPONSE_TTL_SECS + 2).unwrap(),
        1,
    );
    assert!(failure.is_none());
    assert_eq!(summary.removed_verified_submit_responses, 1);
    assert_eq!(
        service
            .conn
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM relay_verified_submit_responses",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("count cleaned durable replay rows"),
        0
    );
    assert_eq!(
        service
            .reserve_verified_submit(&second_request)
            .expect("reuse capacity after expiry"),
        VerifiedSubmitAdmission::Reserved
    );

    drop(service);
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_pending_reservation_recovers_with_owner_fencing() {
    // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] A replacement
    // process may recover exact entry custody after a short grace period,
    // while owner-CAS completion fences the predecessor and private rows
    // continue to expose no raw request metadata.
    let db_path = unique_test_db_path("verified-submit-pending-restart");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xE1; 32];
    let sender = IdentityKeyPair::generate();
    let request = ChatRelayVerifiedSubmitRequestV1::signed(
        [0xE2; 16],
        make_envelope(&sender, [0xE3; 32]),
        now_secs(),
        &sender,
    )
    .expect("sign crash-window verified submit request");

    let predecessor = ChatRelayService::new(config.clone(), secret)
        .expect("create crash-window verified submit relay");
    assert_eq!(
        predecessor
            .reserve_verified_submit(&request)
            .expect("persist crash-window reservation"),
        VerifiedSubmitAdmission::Reserved
    );
    let (cache_key, fingerprint, owner_epoch) = predecessor
        .conn
        .lock()
        .query_row(
            "SELECT cache_key, envelope_fingerprint, owner_epoch
             FROM relay_verified_submit_reservations",
            [],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            },
        )
        .expect("read private reservation row");
    assert_eq!(cache_key.len(), 32);
    assert_eq!(fingerprint.len(), 32);
    assert_eq!(owner_epoch.len(), REPLAY_PROCESS_EPOCH_BYTES);
    for private_value in [
        request.request_id.as_slice(),
        request.envelope.message_id.as_slice(),
        request.envelope.sender.as_slice(),
    ] {
        assert!(!cache_key
            .windows(private_value.len())
            .any(|window| window == private_value));
        assert!(!fingerprint
            .windows(private_value.len())
            .any(|window| window == private_value));
        assert!(!owner_epoch
            .windows(private_value.len())
            .any(|window| window == private_value));
    }

    let aged_owner = sqlite_integer(
        now_secs().saturating_sub(VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS + 1),
        "verified_submit_test_aged_owner",
    )
    .expect("convert aged verified-submit owner timestamp");
    predecessor
        .conn
        .lock()
        .execute(
            "UPDATE relay_verified_submit_reservations
             SET reserved_at = ?1, owner_acquired_at = ?1",
            params![aged_owner],
        )
        .expect("age crash-window owner lease");
    drop(predecessor);

    let restarted = ChatRelayService::new(config.clone(), secret)
        .expect("restart crash-window verified submit relay");
    assert!(matches!(
        restarted
            .verified_submit_cache_lookup(&request)
            .expect("classify crash-window reservation"),
        VerifiedSubmitCacheLookup::Pending
    ));
    assert_eq!(
        restarted
            .reserve_verified_submit(&request)
            .expect("take over abandoned entry custody"),
        VerifiedSubmitAdmission::ReservedForEntryRecovery
    );
    assert_eq!(
        restarted
            .peer_status()
            .verified_submit
            .entry_recovery
            .attempted_total,
        1
    );
    let recovered_response = ChatRelayVerifiedSubmitResponseV1::from_evidence(
        request.request_id,
        request.envelope.message_id,
        false,
        true,
        None,
    );
    restarted
        .remember_verified_submit_response(&request, &recovered_response)
        .expect("complete recovered entry custody as current owner");
    assert!(matches!(
        restarted
            .verified_submit_cache_lookup(&request)
            .expect("replay recovered entry response"),
        VerifiedSubmitCacheLookup::Exact(response) if response == recovered_response
    ));

    let replacement_request = ChatRelayVerifiedSubmitRequestV1::signed(
        [0xE6; 16],
        make_envelope(&sender, [0xE7; 32]),
        now_secs(),
        &sender,
    )
    .expect("sign replacement verified submit request");
    assert_eq!(
        restarted
            .reserve_verified_submit(&replacement_request)
            .expect("reserve unrelated replacement request"),
        VerifiedSubmitAdmission::Reserved
    );
    restarted
        .conn
        .lock()
        .execute(
            "UPDATE relay_verified_submit_reservations
             SET reserved_at = 0, owner_acquired_at = 0",
            [],
        )
        .expect("age replacement reservation");
    let (summary, failure) = restarted.run_cleanup_at(
        i64::try_from(VERIFIED_SUBMIT_RESPONSE_TTL_SECS + 2).unwrap(),
        1,
    );
    assert!(failure.is_none());
    assert_eq!(summary.removed_verified_submit_reservations, 1);
    assert!(matches!(
        restarted
            .verified_submit_cache_lookup(&replacement_request)
            .expect("release expired replacement reservation"),
        VerifiedSubmitCacheLookup::Miss
    ));

    drop(restarted);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_response_replays_privately_across_restart() {
    // [DURABLE-BLIND-RELAY-REPLAY 2026-08-24 by Codex] The restart path
    // must recover the exact sealed ACK, reject route-id substitution, and
    // persist no raw route, request commitment, or response bytes.
    let db_path = unique_test_db_path("blind-route-response-restart");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xF1; 32];
    let route_id = [0xF2; 16];
    let request_commitment = [0xF3; 32];
    let conflicting_commitment = [0xF4; 32];
    let response = b"bounded opaque blind relay acknowledgement".to_vec();
    let completed_at = now_secs();

    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("create durable blind-route relay");
        assert_eq!(
            service
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("reserve durable blind route"),
            BlindRelayRouteAdmission::Reserved
        );
        service
            .remember_blind_relay_route_response(
                &route_id,
                &request_commitment,
                &response,
                completed_at,
            )
            .expect("persist sealed blind-route response");

        let (cache_key, fingerprint, nonce, ciphertext) = service
            .conn
            .lock()
            .query_row(
                "SELECT cache_key, request_fingerprint, response_nonce,
                        response_ciphertext
                 FROM relay_blind_route_responses",
                [],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                    ))
                },
            )
            .expect("read private blind-route replay row");
        assert_eq!(cache_key.len(), 32);
        assert_eq!(fingerprint.len(), 32);
        assert_eq!(nonce.len(), BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES);
        for private_value in [
            route_id.as_slice(),
            request_commitment.as_slice(),
            response.as_slice(),
        ] {
            assert!(!cache_key
                .windows(private_value.len())
                .any(|window| window == private_value));
            assert!(!fingerprint
                .windows(private_value.len())
                .any(|window| window == private_value));
            assert!(!ciphertext
                .windows(private_value.len())
                .any(|window| window == private_value));
        }
    }

    let restarted =
        ChatRelayService::new(config, secret).expect("restart durable blind-route relay");
    assert_eq!(
        restarted
            .reserve_blind_relay_route(&route_id, &conflicting_commitment)
            .expect("classify durable blind-route conflict"),
        BlindRelayRouteAdmission::Conflict
    );
    assert_eq!(
        restarted
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("recover durable blind-route response"),
        BlindRelayRouteAdmission::Completed {
            response: response.clone(),
            completed_at,
        }
    );

    restarted
        .conn
        .lock()
        .execute(
            "UPDATE relay_blind_route_responses SET completed_at = 0",
            [],
        )
        .expect("age durable blind-route response");
    let cleanup_now = i64::try_from(BLIND_RELAY_ROUTE_REPLAY_TTL_SECS + 2).unwrap_or(i64::MAX);
    let (summary, failure) = restarted.run_cleanup_at(cleanup_now, 1);
    assert!(failure.is_none());
    assert_eq!(summary.removed_blind_route_responses, 1);
    assert_eq!(
        restarted
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reuse expired blind route"),
        BlindRelayRouteAdmission::Reserved
    );

    drop(restarted);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_pending_reservation_survives_restart_and_fails_closed() {
    // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] A crash after
    // arming has an ambiguous external effect. Preserve that claim across
    // restart instead of forwarding or storing the ciphertext twice.
    let db_path = unique_test_db_path("blind-route-pending-restart");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xE8; 32];
    let route_id = [0xE9; 16];
    let request_commitment = [0xEA; 32];

    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("create pending blind-route relay");
        assert_eq!(
            service
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("persist blind-route reservation"),
            BlindRelayRouteAdmission::Reserved
        );
        service
            .arm_blind_relay_route_effect(&route_id, &request_commitment, now_secs())
            .expect("arm ambiguous blind-route effect");
        let (cache_key, fingerprint, owner_epoch, effect_started_at) = service
            .conn
            .lock()
            .query_row(
                "SELECT cache_key, request_fingerprint, owner_epoch,
                        effect_started_at
                 FROM relay_blind_route_reservations",
                [],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Option<i64>>(3)?,
                    ))
                },
            )
            .expect("read private blind-route reservation");
        assert_eq!(cache_key.len(), 32);
        assert_eq!(fingerprint.len(), 32);
        assert_eq!(owner_epoch.len(), REPLAY_PROCESS_EPOCH_BYTES);
        assert!(effect_started_at.is_some());
        assert!(!cache_key
            .windows(route_id.len())
            .any(|window| window == route_id));
        assert!(!fingerprint
            .windows(request_commitment.len())
            .any(|window| window == request_commitment));
    }

    let restarted =
        ChatRelayService::new(config, secret).expect("restart pending blind-route relay");
    assert_eq!(
        restarted
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("classify pending blind-route reservation"),
        BlindRelayRouteAdmission::Pending
    );
    assert_eq!(
        restarted
            .reserve_blind_relay_route(&route_id, &[0xEB; 32])
            .expect("classify pending blind-route conflict"),
        BlindRelayRouteAdmission::Conflict
    );

    restarted
        .conn
        .lock()
        .execute(
            "UPDATE relay_blind_route_reservations SET reserved_at = 0",
            [],
        )
        .expect("age pending blind-route reservation");
    let cleanup_now = i64::try_from(BLIND_RELAY_ROUTE_REPLAY_TTL_SECS + 2).unwrap_or(i64::MAX);
    let (summary, failure) = restarted.run_cleanup_at(cleanup_now, 1);
    assert!(failure.is_none());
    assert_eq!(summary.removed_blind_route_reservations, 1);
    assert_eq!(
        restarted
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reuse expired blind-route reservation"),
        BlindRelayRouteAdmission::Reserved
    );

    drop(restarted);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_unarmed_claim_is_reclaimed_after_predecessor_exit() {
    // [CHAT-RELAY-RUNTIME-FENCE 2026-08-25 by Codex] A replacement cannot
    // coexist with the predecessor. After kernel-confirmed predecessor
    // exit, it may take only an aged, unarmed durable claim; owner CAS
    // remains the persisted recovery boundary.
    let db_path = unique_test_db_path("blind-route-unarmed-takeover");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xD1; 32];
    let route_id = [0xD2; 16];
    let request_commitment = [0xD3; 32];
    let response = b"owner-fenced response";

    let old_owner =
        ChatRelayService::new(config.clone(), secret).expect("create old blind-route owner");
    assert_eq!(
        old_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve old unarmed claim"),
        BlindRelayRouteAdmission::Reserved
    );
    old_owner
        .conn
        .lock()
        .execute(
            "UPDATE relay_blind_route_reservations
             SET reserved_at = ?1, owner_acquired_at = ?1",
            params![sqlite_integer(
                now_secs().saturating_sub(BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1),
                "blind_relay_test_aged_unarmed_claim",
            )
            .expect("convert aged unarmed claim timestamp")],
        )
        .expect("age unarmed claim beyond takeover grace");
    let takeover_evidence_at = old_owner
        .conn
        .lock()
        .query_row(
            "SELECT reserved_at FROM relay_blind_route_reservations",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("read immutable reservation timestamp");
    drop(old_owner);

    let new_owner =
        ChatRelayService::new(config.clone(), secret).expect("create new blind-route owner");
    assert_eq!(
        new_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("take over aged unarmed claim"),
        BlindRelayRouteAdmission::Reserved
    );
    let retained_reserved_at = new_owner
        .conn
        .lock()
        .query_row(
            "SELECT reserved_at FROM relay_blind_route_reservations",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("read retained reservation timestamp");
    assert_eq!(retained_reserved_at, takeover_evidence_at);
    new_owner
        .arm_blind_relay_route_effect(&route_id, &request_commitment, now_secs())
        .expect("arm claim under new owner epoch");
    new_owner
        .remember_blind_relay_route_response(&route_id, &request_commitment, response, now_secs())
        .expect("complete claim under new owner epoch");
    assert!(matches!(
        new_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("recover new owner's exact response"),
        BlindRelayRouteAdmission::Completed { response: stored, .. }
            if stored == response
    ));

    drop(new_owner);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_armed_claim_is_recovered_after_predecessor_exit() {
    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] An armed claim is
    // recoverable only by a later process presenting the exact request.
    // Takeover preserves original evidence age after the runtime fence has
    // proved that the old process no longer owns the custody database.
    let db_path = unique_test_db_path("blind-route-armed-takeover");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xA4; 32];
    let route_id = [0xA5; 16];
    let request_commitment = [0xA6; 32];
    let response = b"reconciled exact response";

    let old_owner = ChatRelayService::new(config.clone(), secret).expect("create armed old owner");
    assert_eq!(
        old_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve armed claim"),
        BlindRelayRouteAdmission::Reserved
    );
    old_owner
        .arm_blind_relay_route_effect(&route_id, &request_commitment, now_secs())
        .expect("arm recoverable claim");
    old_owner
        .conn
        .lock()
        .execute(
            "UPDATE relay_blind_route_reservations
             SET reserved_at = ?1, owner_acquired_at = ?1",
            params![sqlite_integer(
                now_secs().saturating_sub(BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1),
                "blind_relay_test_aged_armed_owner",
            )
            .expect("convert aged owner timestamp")],
        )
        .expect("age armed owner lease");
    let takeover_evidence_at = old_owner
        .conn
        .lock()
        .query_row(
            "SELECT reserved_at FROM relay_blind_route_reservations",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("read armed evidence age");
    drop(old_owner);

    let new_owner =
        ChatRelayService::new(config.clone(), secret).expect("create armed recovery owner");
    assert_eq!(
        new_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("take over exact armed claim"),
        BlindRelayRouteAdmission::ReservedForRecovery
    );
    assert_eq!(
        new_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("keep recovered claim single-flight"),
        BlindRelayRouteAdmission::Pending
    );
    let retained_reserved_at = new_owner
        .conn
        .lock()
        .query_row(
            "SELECT reserved_at FROM relay_blind_route_reservations",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("read recovered evidence age");
    assert_eq!(retained_reserved_at, takeover_evidence_at);
    new_owner
        .remember_blind_relay_route_response(&route_id, &request_commitment, response, now_secs())
        .expect("persist reconciled response under new owner");
    assert!(matches!(
        new_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("replay reconciled response"),
        BlindRelayRouteAdmission::Completed { response: stored, .. }
            if stored == response
    ));

    drop(new_owner);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_release_only_removes_owned_unarmed_claim() {
    let db_path = unique_test_db_path("blind-route-unarmed-release");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let service =
        ChatRelayService::new(config, [0xC1; 32]).expect("create blind-route release relay");
    let route_id = [0xC2; 16];
    let request_commitment = [0xC3; 32];

    assert_eq!(
        service
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve releasable claim"),
        BlindRelayRouteAdmission::Reserved
    );
    assert!(service
        .release_unarmed_blind_relay_route(&route_id, &request_commitment)
        .expect("release owned unarmed claim"));
    assert_eq!(
        service
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("reserve released route again"),
        BlindRelayRouteAdmission::Reserved
    );
    service
        .arm_blind_relay_route_effect(&route_id, &request_commitment, now_secs())
        .expect("arm retained claim");
    assert!(!service
        .release_unarmed_blind_relay_route(&route_id, &request_commitment)
        .expect("preserve armed claim"));
    assert_eq!(
        service
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("classify retained armed claim"),
        BlindRelayRouteAdmission::Pending
    );

    drop(service);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v5_fresh_check_uses_shared_crypto_ceiling() {
    // [BLIND-ROUTE-RESPONSE-SCHEMA-V4 2026-08-31 by Codex] Prove the active
    // SQLite CHECK accepts the shared maximum and rejects the next byte and
    // non-BLOB storage without duplicating the ceiling in this fixture.
    let service = make_service();
    let connection = service.conn.lock();
    let schema_sql = connection
        .query_row(
            "SELECT sql FROM sqlite_master
             WHERE type = 'table' AND name = 'relay_blind_route_responses'",
            [],
            |row| row.get::<_, String>(0),
        )
        .expect("read fresh blind-route response schema");
    assert!(schema_sql.contains(&format!(
        "LENGTH(response_ciphertext) <= {MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES}"
    )));
    assert!(schema_sql.contains("TYPEOF(response_ciphertext) = 'blob'"));

    connection
        .execute(
            "INSERT INTO relay_blind_route_responses (
                cache_key, request_fingerprint, response_nonce,
                response_ciphertext, completed_at
             ) VALUES (zeroblob(32), zeroblob(32), zeroblob(24), zeroblob(?1), 1)",
            params![i64::try_from(MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES)
                .expect("shared response ceiling fits SQLite")],
        )
        .expect("accept exact shared ciphertext ceiling");
    connection
        .execute("DELETE FROM relay_blind_route_responses", [])
        .expect("remove boundary fixture");
    assert!(connection
        .execute(
            "INSERT INTO relay_blind_route_responses (
                cache_key, request_fingerprint, response_nonce,
                response_ciphertext, completed_at
             ) VALUES (zeroblob(32), zeroblob(32), zeroblob(24), zeroblob(?1), 1)",
            params![i64::try_from(MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES + 1)
                .expect("oversized response fixture fits SQLite")],
        )
        .is_err());
    assert!(connection
        .execute(
            "INSERT INTO relay_blind_route_responses (
                cache_key, request_fingerprint, response_nonce,
                response_ciphertext, completed_at
             ) VALUES (zeroblob(32), zeroblob(32), ?1, zeroblob(17), 1)",
            params!["N".repeat(BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES)],
        )
        .is_err());
}

#[test]
fn blind_route_schema_v3_migrates_bytes_and_owner_state_idempotently() {
    let db_path = unique_test_db_path("blind-route-v3-response-migration");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0x91; 32];
    let route_id = [0x92; 16];
    let request_commitment = [0x93; 32];
    let cache_key = [0x94; 32];
    let request_fingerprint = [0x95; 32];
    let nonce = vec![0x96; BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES];
    let ciphertext = vec![0x97; 2064];
    let completed_at = now_secs();
    let reservation_before;

    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install blind-route response schema v4 fixture");
        assert_eq!(
            service
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("seed v3 owner-fenced reservation"),
            BlindRelayRouteAdmission::Reserved
        );
        let connection = service.conn.lock();
        reservation_before = connection
            .query_row(
                "SELECT owner_epoch, owner_acquired_at, effect_started_at
                 FROM relay_blind_route_reservations",
                [],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                    ))
                },
            )
            .expect("snapshot v3 reservation state");
        replace_blind_route_response_table_with_v3(&connection);
        connection
            .execute(
                "INSERT INTO relay_blind_route_responses (
                    cache_key, request_fingerprint, response_nonce,
                    response_ciphertext, completed_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    cache_key.as_slice(),
                    request_fingerprint.as_slice(),
                    nonce.as_slice(),
                    ciphertext.as_slice(),
                    i64::try_from(completed_at).expect("completion time fits SQLite"),
                ],
            )
            .expect("seed maximum v3 response");
    }

    let migrated = ChatRelayService::new(config.clone(), secret)
        .expect("atomically migrate blind-route response v3 to v4");
    let connection = migrated.conn.lock();
    let migrated_row = connection
        .query_row(
            "SELECT cache_key, request_fingerprint, response_nonce,
                    response_ciphertext, completed_at
             FROM relay_blind_route_responses",
            [],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )
        .expect("read migrated response bytes");
    assert_eq!(migrated_row.0, cache_key);
    assert_eq!(migrated_row.1, request_fingerprint);
    assert_eq!(migrated_row.2, nonce);
    assert_eq!(migrated_row.3, ciphertext);
    assert_eq!(migrated_row.4, completed_at as i64);
    let reservation_after = connection
        .query_row(
            "SELECT owner_epoch, owner_acquired_at, effect_started_at
             FROM relay_blind_route_reservations",
            [],
            |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            },
        )
        .expect("read preserved v3 reservation state");
    assert_eq!(reservation_after, reservation_before);
    assert_eq!(
        connection
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .expect("read migrated marker"),
        BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'idx_blind_route_response_retention'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("verify migrated retention index"),
        1
    );
    drop(connection);
    drop(migrated);

    let reopened = ChatRelayService::new(config, secret).expect("repeat v4 migration idempotently");
    assert_eq!(
        reopened
            .conn
            .lock()
            .query_row(
                "SELECT response_ciphertext FROM relay_blind_route_responses",
                [],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .expect("read response after idempotent reopen"),
        ciphertext
    );
    drop(reopened);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v4_marker_failure_rolls_back_and_retries() {
    let db_path = unique_test_db_path("blind-route-v4-marker-rollback");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xA1; 32];
    {
        let service =
            ChatRelayService::new(config.clone(), secret).expect("install rollback fixture schema");
        let connection = service.conn.lock();
        replace_blind_route_response_table_with_v3(&connection);
        connection
            .execute(
                "INSERT INTO relay_blind_route_responses VALUES (
                    zeroblob(32), zeroblob(32), zeroblob(24), zeroblob(17), ?1
                 )",
                params![i64::try_from(now_secs()).expect("test time fits SQLite")],
            )
            .expect("seed rollback response");
        connection
            .execute_batch(
                "CREATE TRIGGER fail_blind_route_v4_marker
                 BEFORE UPDATE OF schema_version ON relay_schema_features
                 WHEN OLD.feature = 'blind_relay_route_replay'
                 BEGIN
                    SELECT RAISE(ABORT, 'test marker failure');
                 END;",
            )
            .expect("install deterministic marker failure");
    }

    assert!(matches!(
        ChatRelayService::new(config.clone(), secret),
        Err(ChatRelayError::Sqlite(_))
    ));
    let connection = Connection::open(&db_path).expect("open rolled-back v3 database");
    let (version, response_count, candidate_exists, schema_sql) = connection
        .query_row(
            "SELECT
                (SELECT schema_version FROM relay_schema_features WHERE feature = ?1),
                (SELECT COUNT(*) FROM relay_blind_route_responses),
                EXISTS(SELECT 1 FROM sqlite_master
                       WHERE type = 'table' AND name = 'relay_blind_route_responses_v4'),
                (SELECT sql FROM sqlite_master
                 WHERE type = 'table' AND name = 'relay_blind_route_responses')",
            params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .expect("inspect rolled-back migration");
    assert_eq!(version, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION);
    assert_eq!(response_count, 1);
    assert_eq!(candidate_exists, 0);
    assert!(schema_sql.contains("LENGTH(response_ciphertext) <= 2064"));
    connection
        .execute("DROP TRIGGER fail_blind_route_v4_marker", [])
        .expect("remove deterministic marker failure");
    drop(connection);

    let retried = ChatRelayService::new(config, secret).expect("retry rolled-back v4 migration");
    assert_eq!(
        retried
            .conn
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM relay_blind_route_responses",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .expect("count response after retry"),
        1
    );
    drop(retried);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v4_unknown_version_and_candidate_fail_closed() {
    for (label, prepare, expected_field) in [
        (
            "unknown-version",
            "UPDATE relay_schema_features SET schema_version = 99
             WHERE feature = 'blind_relay_route_replay'",
            "blind_relay_route_replay_installation_version",
        ),
        (
            "candidate-residue",
            "CREATE TABLE relay_blind_route_responses_v4 (unexpected INTEGER)",
            "blind_relay_route_response_migration_candidate",
        ),
    ] {
        let db_path = unique_test_db_path(label);
        let mut config = test_config();
        config.db_path = db_path.to_string_lossy().into_owned();
        let secret = [0xB1; 32];
        drop(
            ChatRelayService::new(config.clone(), secret)
                .expect("install fail-closed fixture schema"),
        );
        Connection::open(&db_path)
            .expect("open fail-closed fixture")
            .execute_batch(prepare)
            .expect("prepare fail-closed schema state");

        assert!(matches!(
            ChatRelayService::new(config, secret),
            Err(ChatRelayError::CorruptStoredData { field }) if field == expected_field
        ));
        remove_test_database(&db_path);
    }
}

#[test]
fn blind_route_schema_v4_binary_contract_rejects_committed_v5() {
    // [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] A rolled-back v4 binary
    // fails closed on committed v5 instead of running without byte admission.
    let service = make_service();
    let mut connection = service.conn.lock();
    let v4_migrator = SqliteChatRelayReplaySchemaMigrator::new(ReplaySchemaContract::new(
        ReplaySchemaVersion::new(
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
            VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION,
        ),
        ReplaySchemaVersion::new(
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V4_VERSION,
            BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V4_VERSION,
        ),
        VERIFIED_SUBMIT_RESPONSE_TTL_SECS,
        BLIND_RELAY_ROUTE_REPLAY_TTL_SECS,
        REPLAY_PROCESS_EPOCH_BYTES,
        BLIND_RELAY_ROUTE_REPLAY_CAPACITY,
        ChatRelayConfig::default().blind_route_replay_max_bytes_total,
    ));
    assert!(matches!(
        v4_migrator.migrate_blind_route(&mut connection, now_secs()),
        Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_replay_installation_version"
        })
    ));
}

#[test]
fn blind_route_schema_v3_polluted_row_rolls_back_without_candidate() {
    let db_path = unique_test_db_path("blind-route-v3-polluted-row");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xC1; 32];
    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install polluted-row fixture schema");
        let connection = service.conn.lock();
        replace_blind_route_response_table_with_v3(&connection);
        connection
            .execute(
                "INSERT INTO relay_blind_route_responses VALUES (
                    zeroblob(32), zeroblob(32), ?1, zeroblob(17), ?2
                 )",
                params![
                    "N".repeat(BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES),
                    i64::try_from(now_secs()).expect("test time fits SQLite"),
                ],
            )
            .expect("seed non-BLOB v3 nonce");
    }

    assert!(matches!(
        ChatRelayService::new(config.clone(), secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_replay_row_shape"
        })
    ));
    let connection = Connection::open(&db_path).expect("inspect polluted v3 database");
    assert_eq!(
        connection
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .expect("read unchanged v3 marker"),
        BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master
                 WHERE type = 'table' AND name = 'relay_blind_route_responses_v4')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("inspect absent migration candidate"),
        0
    );
    drop(connection);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v5_capacity_is_checked_after_ttl_cleanup() {
    let service = make_service();
    let mut connection = service.conn.lock();
    let active_at = i64::try_from(now_secs()).expect("test time fits SQLite");
    connection
        .execute(
            "INSERT INTO relay_blind_route_responses VALUES (
                zeroblob(32), zeroblob(32), zeroblob(24), zeroblob(17), ?1
             )",
            params![active_at],
        )
        .expect("seed first response above test capacity");
    connection
        .execute(
            "INSERT INTO relay_blind_route_responses VALUES (
                CAST('11111111111111111111111111111111' AS BLOB),
                zeroblob(32), zeroblob(24), zeroblob(17), ?1
             )",
            params![active_at],
        )
        .expect("seed second response above test capacity");
    assert!(matches!(
        test_replay_schema_migrator(1).migrate_blind_route(&mut connection, now_secs()),
        Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_replay_capacity"
        })
    ));
}

#[test]
fn blind_route_schema_v4_to_v5_is_byte_preserving_idempotent_and_keeps_lease() {
    // [CHAT-RELAY-RESOURCE-BOUND 2026-08-31 by Codex] V4 already owns the
    // shared crypto CHECK, so v5 advances only the marker after resource
    // validation and must not rewrite BLOBs or lease timestamps.
    let db_path = unique_test_db_path("blind-route-v4-resource-migration");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xD4; 32];
    let response_key = [0x41_u8; 32];
    let response_fingerprint = [0x42_u8; 32];
    let response_nonce = vec![0x43_u8; BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES];
    let response_ciphertext = vec![0x44_u8; 4_096];
    let reservation_key = [0x45_u8; 32];
    let reservation_fingerprint = [0x46_u8; 32];
    let owner_epoch = vec![0x47_u8; REPLAY_PROCESS_EPOCH_BYTES];
    let reserved_at = i64::try_from(now_secs().saturating_sub(10)).expect("test time fits SQLite");
    let owner_acquired_at = reserved_at + 1;
    let effect_started_at = reserved_at + 2;
    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install v5 resource migration fixture");
        let connection = service.conn.lock();
        connection
            .execute(
                "INSERT INTO relay_blind_route_responses VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    response_key.as_slice(),
                    response_fingerprint.as_slice(),
                    response_nonce.as_slice(),
                    response_ciphertext.as_slice(),
                    reserved_at,
                ],
            )
            .expect("seed v4 response bytes");
        connection
            .execute(
                "INSERT INTO relay_blind_route_reservations VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    reservation_key.as_slice(),
                    reservation_fingerprint.as_slice(),
                    reserved_at,
                    owner_epoch.as_slice(),
                    owner_acquired_at,
                    effect_started_at,
                ],
            )
            .expect("seed v4 owner lease");
        connection
            .execute(
                "UPDATE relay_schema_features SET schema_version = ?1 WHERE feature = ?2",
                params![
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V4_VERSION,
                    BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                ],
            )
            .expect("mark fixture as committed v4");
    }

    for attempt in 0..2 {
        let service = ChatRelayService::new(config.clone(), secret)
            .unwrap_or_else(|error| panic!("v5 migration attempt {attempt} failed: {error}"));
        let (version, stored_nonce, stored_ciphertext, stored_reserved, stored_acquired, stored_effect) =
            service
                .conn
                .lock()
                .query_row(
                    "SELECT
                        (SELECT schema_version FROM relay_schema_features WHERE feature = ?1),
                        (SELECT response_nonce FROM relay_blind_route_responses WHERE cache_key = ?2),
                        (SELECT response_ciphertext FROM relay_blind_route_responses WHERE cache_key = ?2),
                        reserved_at, owner_acquired_at, effect_started_at
                     FROM relay_blind_route_reservations WHERE cache_key = ?3",
                    params![
                        BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
                        response_key.as_slice(),
                        reservation_key.as_slice(),
                    ],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, Vec<u8>>(1)?,
                            row.get::<_, Vec<u8>>(2)?,
                            row.get::<_, i64>(3)?,
                            row.get::<_, i64>(4)?,
                            row.get::<_, Option<i64>>(5)?,
                        ))
                    },
                )
                .expect("inspect v5 byte-preserving migration");
        assert_eq!(version, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION);
        assert_eq!(stored_nonce, response_nonce);
        assert_eq!(stored_ciphertext, response_ciphertext);
        assert_eq!(stored_reserved, reserved_at);
        assert_eq!(stored_acquired, owner_acquired_at);
        assert_eq!(stored_effect, Some(effect_started_at));
        drop(service);
    }
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v5_rejects_live_bytes_but_prunes_expired_bytes() {
    let service = make_service();
    let mut connection = service.conn.lock();
    let now = now_secs();
    let active_at = i64::try_from(now).expect("active test time fits SQLite");
    let expired_at = i64::try_from(now.saturating_sub(BLIND_RELAY_ROUTE_REPLAY_TTL_SECS + 1))
        .expect("expired test time fits SQLite");
    let max_bytes = i64::try_from(MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES)
        .expect("max protected bytes fit SQLite");
    connection
        .execute(
            "INSERT INTO relay_blind_route_responses VALUES (
                zeroblob(32), zeroblob(32), zeroblob(24), zeroblob(?1), ?2
             )",
            params![max_bytes, active_at],
        )
        .expect("seed maximum active response");
    connection
        .execute(
            "INSERT INTO relay_blind_route_responses VALUES (
                CAST('22222222222222222222222222222222' AS BLOB),
                zeroblob(32), zeroblob(24), zeroblob(17), ?1
             )",
            params![active_at],
        )
        .expect("seed active byte overflow");
    let one_max_response = 32_u64
        + 32
        + u64::try_from(BLIND_RELAY_ROUTE_RESPONSE_NONCE_BYTES).expect("nonce size fits")
        + u64::try_from(MAX_PROTECTED_BLIND_ROUTE_RESPONSE_BYTES).expect("ciphertext size fits");
    assert!(matches!(
        test_replay_schema_migrator_with_budget(
            BLIND_RELAY_ROUTE_REPLAY_CAPACITY,
            one_max_response
        )
        .migrate_blind_route(&mut connection, now),
        Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_replay_byte_capacity"
        })
    ));
    connection
        .execute(
            "UPDATE relay_blind_route_responses SET completed_at = ?1",
            params![expired_at],
        )
        .expect("expire over-budget responses");
    test_replay_schema_migrator_with_budget(BLIND_RELAY_ROUTE_REPLAY_CAPACITY, one_max_response)
        .migrate_blind_route(&mut connection, now)
        .expect("TTL cleanup restores byte capacity");
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM relay_blind_route_responses",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("count pruned response bytes"),
        0
    );
}

#[test]
fn blind_route_schema_v3_writer_contention_fails_then_retries() {
    let db_path = unique_test_db_path("blind-route-v3-writer-contention");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xD1; 32];
    {
        let service = ChatRelayService::new(config, secret)
            .expect("install writer-contention fixture schema");
        replace_blind_route_response_table_with_v3(&service.conn.lock());
    }
    let mut blocker = Connection::open(&db_path).expect("open blocking writer");
    let blocking_tx = blocker
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .expect("acquire blocking writer transaction");
    let mut contender = Connection::open(&db_path).expect("open migration contender");
    contender
        .busy_timeout(Duration::ZERO)
        .expect("disable contender wait");
    assert!(matches!(
        test_replay_schema_migrator(BLIND_RELAY_ROUTE_REPLAY_CAPACITY)
            .migrate_blind_route(&mut contender, now_secs()),
        Err(ChatRelayError::Sqlite(_))
    ));
    blocking_tx.rollback().expect("release blocking writer");
    test_replay_schema_migrator(BLIND_RELAY_ROUTE_REPLAY_CAPACITY)
        .migrate_blind_route(&mut contender, now_secs())
        .expect("retry migration after writer release");
    assert_eq!(
        contender
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .expect("read retried v4 marker"),
        BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION
    );
    drop(blocker);
    drop(contender);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v1_migrates_claims_as_armed() {
    // [RECOVERABLE-BLIND-RELAY-CLAIM 2026-08-24 by Codex] Version 1 did
    // not persist an effect boundary. Migration must classify every legacy
    // claim as ambiguous rather than making it eligible for takeover.
    let db_path = unique_test_db_path("blind-route-v1-migration");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xB1; 32];
    let route_id = [0xB2; 16];
    let request_commitment = [0xB3; 32];

    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install current blind-route schema");
        assert_eq!(
            service
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("seed legacy reservation"),
            BlindRelayRouteAdmission::Reserved
        );
        let conn = service.conn.lock();
        conn.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
             WHERE feature = ?2",
            params![
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_LEGACY_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
            ],
        )
        .expect("downgrade blind-route marker to v1");
        conn.execute(
            "ALTER TABLE relay_blind_route_reservations
             DROP COLUMN effect_started_at",
            [],
        )
        .expect("remove v2 effect marker");
        conn.execute(
            "ALTER TABLE relay_blind_route_reservations
             DROP COLUMN owner_acquired_at",
            [],
        )
        .expect("remove v3 owner lease marker");
        conn.execute(
            "ALTER TABLE relay_blind_route_reservations DROP COLUMN owner_epoch",
            [],
        )
        .expect("remove v2 process epoch");
    }

    let migrated =
        ChatRelayService::new(config, secret).expect("migrate blind-route replay schema v1 to v4");
    let (version, owner_epoch, reserved_at, owner_acquired_at, effect_started_at) = migrated
        .conn
        .lock()
        .query_row(
            "SELECT
                (SELECT schema_version FROM relay_schema_features
                 WHERE feature = ?1),
                owner_epoch, reserved_at, owner_acquired_at,
                effect_started_at
             FROM relay_blind_route_reservations",
            params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                ))
            },
        )
        .expect("inspect migrated blind-route claim");
    assert_eq!(version, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION);
    assert_eq!(owner_epoch, vec![0; REPLAY_PROCESS_EPOCH_BYTES]);
    assert_eq!(owner_acquired_at, reserved_at);
    assert_eq!(effect_started_at, Some(reserved_at));
    assert_eq!(
        migrated
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("preserve legacy ambiguous claim"),
        BlindRelayRouteAdmission::Pending
    );

    drop(migrated);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v3_missing_claim_column_fails_closed() {
    let db_path = unique_test_db_path("blind-route-v3-missing-column");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xA1; 32];
    drop(ChatRelayService::new(config.clone(), secret).expect("install blind-route schema v4"));
    let connection = Connection::open(&db_path).expect("open blind-route replay database");
    connection
        .execute(
            "UPDATE relay_schema_features SET schema_version = ?1 WHERE feature = ?2",
            params![
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V3_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
            ],
        )
        .expect("mark fixture as v3");
    connection
        .execute(
            "ALTER TABLE relay_blind_route_reservations
             DROP COLUMN effect_started_at",
            [],
        )
        .expect("remove required v3 claim column");
    drop(connection);

    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_replay_reservation_columns"
        })
    ));
    remove_test_database(&db_path);
}

#[test]
fn blind_route_schema_v2_preserves_explicit_unarmed_state() {
    // [ARMED-BLIND-RELAY-RECOVERY 2026-08-25 by Codex] Unlike ambiguous
    // v1 rows, v2 persisted the effect boundary. Migration must retain an
    // explicit unarmed claim and initialize only its independent lease age.
    let db_path = unique_test_db_path("blind-route-v2-migration");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xB4; 32];
    let route_id = [0xB5; 16];
    let request_commitment = [0xB6; 32];

    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install current blind-route schema");
        assert_eq!(
            service
                .reserve_blind_relay_route(&route_id, &request_commitment)
                .expect("seed v2 unarmed reservation"),
            BlindRelayRouteAdmission::Reserved
        );
        let conn = service.conn.lock();
        conn.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
             WHERE feature = ?2",
            params![
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_V2_VERSION,
                BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE,
            ],
        )
        .expect("downgrade marker to v2");
        conn.execute(
            "ALTER TABLE relay_blind_route_reservations
             DROP COLUMN owner_acquired_at",
            [],
        )
        .expect("remove v3 owner lease column");
    }

    let migrated = ChatRelayService::new(config.clone(), secret)
        .expect("migrate blind-route replay schema v2 to v4");
    let (version, reserved_at, owner_acquired_at, effect_started_at) = migrated
        .conn
        .lock()
        .query_row(
            "SELECT
                (SELECT schema_version FROM relay_schema_features
                 WHERE feature = ?1),
                reserved_at, owner_acquired_at, effect_started_at
             FROM relay_blind_route_reservations",
            params![BLIND_RELAY_ROUTE_REPLAY_SCHEMA_FEATURE],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            },
        )
        .expect("inspect migrated v2 claim");
    assert_eq!(version, BLIND_RELAY_ROUTE_REPLAY_SCHEMA_VERSION);
    assert_eq!(owner_acquired_at, reserved_at);
    assert_eq!(effect_started_at, None);

    migrated
        .conn
        .lock()
        .execute(
            "UPDATE relay_blind_route_reservations
             SET reserved_at = ?1, owner_acquired_at = ?1",
            params![sqlite_integer(
                now_secs().saturating_sub(BLIND_RELAY_OWNER_TAKEOVER_GRACE_SECS + 1),
                "blind_relay_test_aged_migrated_owner",
            )
            .expect("convert migrated owner timestamp")],
        )
        .expect("age migrated owner lease");
    drop(migrated);
    let next_owner =
        ChatRelayService::new(config, secret).expect("open migrated claim from next process");
    assert_eq!(
        next_owner
            .reserve_blind_relay_route(&route_id, &request_commitment)
            .expect("take over migrated unarmed claim"),
        BlindRelayRouteAdmission::Reserved
    );

    drop(next_owner);
    remove_test_database(&db_path);
}

#[test]
fn blind_route_installed_schema_table_loss_fails_closed() {
    let db_path = unique_test_db_path("blind-route-missing-table");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xEC; 32];
    drop(ChatRelayService::new(config.clone(), secret).expect("install blind-route replay schema"));
    Connection::open(&db_path)
        .expect("open blind-route replay database")
        .execute("DROP TABLE relay_blind_route_reservations", [])
        .expect("remove installed blind-route reservation table");

    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "blind_relay_route_replay_table"
        })
    ));
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_schema_v1_migrates_reservations_atomically() {
    let db_path = unique_test_db_path("verified-submit-v1-migration");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xE4; 32];
    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install current verified submit schema");
        let conn = service.conn.lock();
        conn.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
             WHERE feature = ?2",
            params![
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_LEGACY_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
            ],
        )
        .expect("downgrade marker to legacy version");
        conn.execute("DROP TABLE relay_verified_submit_reservations", [])
            .expect("simulate legacy schema without reservations");
    }

    let migrated =
        ChatRelayService::new(config, secret).expect("migrate verified submit schema v1 to v3");
    let (version, reservations_table) = migrated
        .conn
        .lock()
        .query_row(
            "SELECT
                (SELECT schema_version FROM relay_schema_features WHERE feature = ?1),
                EXISTS(SELECT 1 FROM sqlite_master
                       WHERE type = 'table'
                         AND name = 'relay_verified_submit_reservations')",
            params![VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .expect("inspect migrated verified submit schema");
    assert_eq!(version, VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION);
    assert_eq!(reservations_table, 1);

    drop(migrated);
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_schema_v2_migrates_owner_fencing_atomically() {
    // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] Historical v2
    // reservations become recoverable without exposing raw identifiers or
    // temporarily installing a marker newer than the table shape.
    let db_path = unique_test_db_path("verified-submit-v2-migration");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xE8; 32];
    let sender = IdentityKeyPair::generate();
    let request = ChatRelayVerifiedSubmitRequestV1::signed(
        [0xE9; 16],
        make_envelope(&sender, [0xEA; 32]),
        now_secs(),
        &sender,
    )
    .expect("sign migrated verified-submit request");
    {
        let service = ChatRelayService::new(config.clone(), secret)
            .expect("install current verified-submit schema");
        assert_eq!(
            service
                .reserve_verified_submit(&request)
                .expect("reserve row before v2 simulation"),
            VerifiedSubmitAdmission::Reserved
        );
        let conn = service.conn.lock();
        conn.execute(
            "UPDATE relay_schema_features SET schema_version = ?1
             WHERE feature = ?2",
            params![
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_V2_VERSION,
                VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE,
            ],
        )
        .expect("set historical verified-submit v2 marker");
        conn.execute(
            "ALTER TABLE relay_verified_submit_reservations
             RENAME TO relay_verified_submit_reservations_v3",
            [],
        )
        .expect("preserve current reservation rows");
        conn.execute_batch(
            "CREATE TABLE relay_verified_submit_reservations (
                cache_key BLOB PRIMARY KEY NOT NULL CHECK(length(cache_key) = 32),
                envelope_fingerprint BLOB NOT NULL
                    CHECK(length(envelope_fingerprint) = 32),
                reserved_at INTEGER NOT NULL CHECK(reserved_at >= 0)
             ) WITHOUT ROWID;
             INSERT INTO relay_verified_submit_reservations (
                cache_key, envelope_fingerprint, reserved_at
             )
             SELECT cache_key, envelope_fingerprint, reserved_at
             FROM relay_verified_submit_reservations_v3;
             DROP TABLE relay_verified_submit_reservations_v3;",
        )
        .expect("install historical verified-submit v2 table");
    }

    let migrated = ChatRelayService::new(config.clone(), secret)
        .expect("migrate verified-submit schema v2 to v3");
    let (version, reserved_at, owner_epoch, owner_acquired_at) = migrated
        .conn
        .lock()
        .query_row(
            "SELECT
                (SELECT schema_version FROM relay_schema_features WHERE feature = ?1),
                reserved_at, owner_epoch, owner_acquired_at
             FROM relay_verified_submit_reservations",
            params![VERIFIED_SUBMIT_RESPONSE_SCHEMA_FEATURE],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .expect("inspect migrated verified-submit owner fencing");
    assert_eq!(version, VERIFIED_SUBMIT_RESPONSE_SCHEMA_VERSION);
    assert_eq!(owner_epoch, vec![0_u8; REPLAY_PROCESS_EPOCH_BYTES]);
    assert_eq!(owner_acquired_at, reserved_at);

    let aged_owner = sqlite_integer(
        now_secs().saturating_sub(VERIFIED_SUBMIT_OWNER_TAKEOVER_GRACE_SECS + 1),
        "verified_submit_test_aged_migrated_owner",
    )
    .expect("convert migrated verified-submit owner timestamp");
    migrated
        .conn
        .lock()
        .execute(
            "UPDATE relay_verified_submit_reservations
             SET reserved_at = ?1, owner_acquired_at = ?1",
            params![aged_owner],
        )
        .expect("age migrated verified-submit owner lease");
    assert_eq!(
        migrated
            .reserve_verified_submit(&request)
            .expect("recover migrated verified-submit reservation"),
        VerifiedSubmitAdmission::ReservedForEntryRecovery
    );

    drop(migrated);
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_installed_owner_column_loss_fails_closed() {
    // [VERIFIED-SUBMIT-ENTRY-RECOVERY 2026-08-25 by Codex] A current v3
    // marker with a downgraded table must never silently disable fencing.
    let db_path = unique_test_db_path("verified-submit-owner-column-loss");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xEB; 32];
    drop(
        ChatRelayService::new(config.clone(), secret)
            .expect("install verified-submit owner-fencing schema"),
    );
    Connection::open(&db_path)
        .expect("open verified-submit owner-fencing database")
        .execute_batch(
            "DROP TABLE relay_verified_submit_reservations;
             CREATE TABLE relay_verified_submit_reservations (
                cache_key BLOB PRIMARY KEY NOT NULL CHECK(length(cache_key) = 32),
                envelope_fingerprint BLOB NOT NULL
                    CHECK(length(envelope_fingerprint) = 32),
                reserved_at INTEGER NOT NULL CHECK(reserved_at >= 0)
             ) WITHOUT ROWID;",
        )
        .expect("remove current verified-submit owner columns");

    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "verified_submit_reservation_columns"
        })
    ));
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_missing_installed_reservation_table_fails_closed() {
    let db_path = unique_test_db_path("verified-submit-missing-reservation-table");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xE5; 32];
    drop(
        ChatRelayService::new(config.clone(), secret)
            .expect("install verified submit reservation schema"),
    );
    Connection::open(&db_path)
        .expect("open verified submit reservation database")
        .execute("DROP TABLE relay_verified_submit_reservations", [])
        .expect("remove installed reservation table");

    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "verified_submit_reservation_table"
        })
    ));
    remove_test_database(&db_path);
}

#[test]
fn verified_submit_response_missing_installed_table_fails_closed() {
    let db_path = unique_test_db_path("verified-submit-missing-table");
    let mut config = test_config();
    config.db_path = db_path.to_string_lossy().into_owned();
    let secret = [0xE1; 32];
    drop(
        ChatRelayService::new(config.clone(), secret)
            .expect("install durable verified submit schema"),
    );
    Connection::open(&db_path)
        .expect("open durable verified submit database")
        .execute("DROP TABLE relay_verified_submit_responses", [])
        .expect("remove installed durable verified submit table");

    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "verified_submit_response_table"
        })
    ));
    remove_test_database(&db_path);
}

#[test]
fn peer_health_deserializes_pre_route_class_snapshot() {
    let svc = make_service();
    svc.record_peer_relay_outbound(1_800_000_050, 1, 1, None);
    let mut encoded = serde_json::to_value(svc.peer_status()).expect("serialize peer status");
    let object = encoded.as_object_mut().expect("peer status JSON object");
    object.remove("custody_durability");
    object.remove("authenticated_onion_outbound");
    object.remove("direct_peer_outbound");
    object.remove("direct_peer_retry");
    object.remove("verified_submit");
    object.remove("blind_route_recovery");

    // [RELAY-ROUTE-CLASS-HEALTH 2026-08-15 by Codex] Additive health
    // fields must not invalidate status cached or forwarded by an older
    // node during a rolling upgrade.
    let decoded: ChatRelayPeerStatus =
        serde_json::from_value(encoded).expect("deserialize legacy peer status");
    assert_eq!(decoded.outbound_rounds, 1);
    assert_eq!(decoded.last_outbound_status.as_deref(), Some("healthy"));
    assert_eq!(
        decoded.custody_durability,
        ChatRelayCustodyDurabilityStatus::default()
    );
    assert_eq!(decoded.authenticated_onion_outbound.rounds, 0);
    assert_eq!(decoded.direct_peer_outbound.rounds, 0);
    assert_eq!(
        decoded.direct_peer_retry,
        ChatRelayDirectPeerRetryStatus::default()
    );
    assert_eq!(
        decoded.verified_submit,
        ChatRelayVerifiedSubmitStatus::default()
    );
    assert_eq!(
        decoded.blind_route_recovery,
        ChatRelayBlindRouteRecoveryStatus::default()
    );
}

#[test]
fn verified_submit_health_preserves_nested_rolling_compatibility() {
    // [VERIFIED-SUBMIT-RECOVERY-STATUS 2026-08-25 by Codex] An older
    // heartbeat snapshot has normal verified-submit counters but no nested
    // recovery evidence. New readers must preserve the former and default
    // the additive recovery object without guessing historical outcomes.
    let svc = make_service();
    svc.record_verified_submit_result(1_800_000_075, CHAT_VERIFIED_SUBMIT_ENTRY_RETRY_V1);
    svc.record_verified_submit_recovery_attempted(1_800_000_076);
    svc.record_verified_submit_recovery_outcome(
        1_800_000_077,
        VerifiedSubmitRecoveryOutcome::Completed,
    );
    let mut encoded = serde_json::to_value(svc.peer_status()).expect("serialize peer status");
    encoded
        .get_mut("verified_submit")
        .and_then(serde_json::Value::as_object_mut)
        .expect("verified-submit status object")
        .remove("entry_recovery");

    let decoded: ChatRelayPeerStatus =
        serde_json::from_value(encoded).expect("deserialize pre-recovery verified-submit status");
    assert_eq!(decoded.verified_submit.total, 1);
    assert_eq!(decoded.verified_submit.entry_retry_total, 1);
    assert_eq!(
        decoded.verified_submit.entry_recovery,
        ChatRelayVerifiedSubmitRecoveryStatus::default()
    );
}

#[test]
fn direct_peer_retry_health_preserves_nested_rolling_compatibility() {
    let svc = make_service();
    complete_direct_peer_test_delivery(&svc, 1_800_000_051, true, true, false);
    let mut encoded = serde_json::to_value(svc.peer_status()).expect("serialize peer status");
    let retry = encoded
        .get_mut("direct_peer_retry")
        .and_then(serde_json::Value::as_object_mut)
        .expect("direct peer retry JSON object");
    let recent = retry
        .get("recent_window")
        .and_then(serde_json::Value::as_object)
        .expect("recent delivery SLO JSON object");
    assert_eq!(recent.get("window_seconds"), Some(&serde_json::json!(300)));
    assert_eq!(recent.get("deliveries_total"), Some(&serde_json::json!(1)));
    assert_eq!(recent.get("status"), Some(&serde_json::json!("healthy")));
    let circuit = retry
        .get("circuit")
        .and_then(serde_json::Value::as_object)
        .expect("direct relay circuit JSON object");
    assert_eq!(circuit.get("state"), Some(&serde_json::json!("closed")));
    assert_eq!(
        circuit.get("restart_protected"),
        Some(&serde_json::json!(true))
    );
    retry.remove("recent_window");
    retry.remove("circuit");

    // [DIRECT-RELAY-RETRY-SLO 2026-08-15 by Codex] Nodes may first learn
    // the lifetime retry counters and only later learn the rolling SLO.
    // Missing additive nested health must therefore deserialize as idle.
    let decoded: ChatRelayPeerStatus =
        serde_json::from_value(encoded).expect("deserialize pre-SLO peer status");
    assert_eq!(decoded.direct_peer_retry.retry_triggered_total, 1);
    assert_eq!(
        decoded.direct_peer_retry.recent_window,
        ChatRelayDirectPeerSloStatus::default()
    );
    assert_eq!(
        decoded.direct_peer_retry.circuit,
        ChatRelayDirectPeerCircuitStatus::default()
    );
}

#[test]
fn direct_peer_retry_health_tracks_recovery_exhaustion_and_determinism() {
    let svc = make_service();

    // [DIRECT-RELAY-RETRY-TELEMETRY 2026-08-15 by Codex] A normal first
    // attempt is invisible to lifetime exception counters but remains in
    // the recent delivery SLO denominator.
    complete_direct_peer_test_delivery(&svc, 1_800_000_001, false, true, false);
    let retry = svc.peer_status().direct_peer_retry;
    assert_eq!(retry.retry_triggered_total, 0);
    assert_eq!(retry.retry_recovered_total, 0);
    assert_eq!(retry.retry_exhausted_total, 0);
    assert_eq!(retry.deterministic_failure_total, 0);
    assert_eq!(retry.recent_window.deliveries_total, 1);
    assert_eq!(retry.recent_window.delivered_total, 1);
    assert_eq!(retry.recent_window.delivery_success_bps, Some(10_000));
    assert_eq!(retry.recent_window.status, "healthy");

    complete_direct_peer_test_delivery(&svc, 1_800_000_010, true, true, false);
    complete_direct_peer_test_delivery(&svc, 1_800_000_020, true, false, true);
    complete_direct_peer_test_delivery(&svc, 1_800_000_030, false, false, true);

    let retry = svc.peer_status().direct_peer_retry;
    assert_eq!(retry.retry_triggered_total, 2);
    assert_eq!(retry.retry_recovered_total, 1);
    assert_eq!(retry.retry_exhausted_total, 1);
    assert_eq!(retry.deterministic_failure_total, 2);
    assert_eq!(
        retry.retry_recovered_total + retry.retry_exhausted_total,
        retry.retry_triggered_total
    );
    assert_eq!(retry.last_outcome.as_deref(), Some("deterministic_failure"));
    assert_eq!(retry.last_at, Some(1_800_000_030));
    assert_eq!(retry.recent_window.deliveries_total, 4);
    assert_eq!(retry.recent_window.delivered_total, 2);
    assert_eq!(retry.recent_window.failed_total, 2);
    assert_eq!(retry.recent_window.delivery_success_bps, Some(5_000));
    assert_eq!(retry.recent_window.retry_recovery_bps, Some(5_000));
    assert_eq!(retry.recent_window.meets_slo, Some(false));
    assert_eq!(retry.recent_window.status, "degraded");
}

#[test]
fn direct_peer_circuit_opens_half_opens_and_requires_two_successes() {
    let svc = make_service();
    let base = 1_800_000_100;

    for offset in 0..3 {
        complete_direct_peer_test_delivery(&svc, base + offset, false, false, true);
    }
    let opened = svc.direct_peer_relay_circuit.lock().snapshot(base + 2);
    assert_eq!(opened.state, "open");
    assert_eq!(opened.opened_total, 1);
    assert_eq!(
        opened.open_remaining_seconds,
        Some(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
    );
    assert!(svc.begin_direct_peer_delivery(base + 3).is_none());

    let first_probe = svc
        .begin_direct_peer_delivery(base + 2 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
        .expect("cooldown expiry should admit one half-open probe");
    assert!(first_probe.is_half_open());
    assert!(svc
        .begin_direct_peer_delivery(base + 2 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
        .is_none());
    svc.complete_direct_peer_delivery(base + 33, first_probe, false, true, false);
    let recovering = svc.direct_peer_relay_circuit.lock().snapshot(base + 33);
    assert_eq!(recovering.state, "half_open");
    assert_eq!(recovering.half_open_consecutive_successes, 1);

    // A duplicate late outcome for the consumed permit cannot overwrite
    // the newer generation or count as a failed recovery probe.
    svc.complete_direct_peer_delivery(base + 34, first_probe, false, false, true);
    let after_stale = svc.direct_peer_relay_circuit.lock().snapshot(base + 34);
    assert_eq!(after_stale.state, "half_open");
    assert_eq!(after_stale.half_open_failed_total, 0);
    assert_eq!(
        svc.peer_telemetry
            .direct_peer_slo_snapshot(base + 34)
            .deliveries_total,
        4
    );

    let second_probe = svc
        .begin_direct_peer_delivery(base + 34)
        .expect("second recovery proof should be admitted serially");
    assert!(second_probe.is_half_open());
    svc.complete_direct_peer_delivery(base + 34, second_probe, false, true, false);
    let recovered = svc.direct_peer_relay_circuit.lock().snapshot(base + 34);
    assert_eq!(recovered.state, "closed");
    assert_eq!(recovered.half_open_attempted_total, 2);
    assert_eq!(recovered.half_open_succeeded_total, 2);
    assert_eq!(recovered.recovered_total, 1);
    assert_eq!(recovered.blocked_total, 2);

    // The previous failed SLO is not enough to reopen by itself, but one
    // new failed delivery while that window remains failed is.
    complete_direct_peer_test_delivery(&svc, base + 35, false, false, true);
    let reopened = svc.direct_peer_relay_circuit.lock().snapshot(base + 35);
    assert_eq!(reopened.state, "open");
    assert_eq!(reopened.opened_total, 2);
}

#[test]
fn direct_peer_circuit_recovers_abandoned_half_open_permit_fail_closed() {
    let svc = make_service();
    let base = 1_800_000_200;
    for offset in 0..3 {
        complete_direct_peer_test_delivery(&svc, base + offset, false, false, true);
    }

    let first_probe_at = base + 2 + DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS;
    let abandoned = svc
        .begin_direct_peer_delivery(first_probe_at)
        .expect("expired open circuit should admit a half-open probe");
    assert!(abandoned.is_half_open());
    assert!(svc
        .begin_direct_peer_delivery(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS)
        .is_none());
    let timed_out = svc
        .direct_peer_relay_circuit
        .lock()
        .snapshot(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS);
    assert_eq!(timed_out.state, "open");
    assert_eq!(timed_out.opened_total, 2);
    assert_eq!(timed_out.half_open_failed_total, 1);

    // A later completion from the expired lease is stale and cannot close
    // the newly opened generation.
    svc.complete_direct_peer_delivery(
        first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS + 1,
        abandoned,
        false,
        true,
        false,
    );
    assert_eq!(
        svc.direct_peer_relay_circuit
            .lock()
            .snapshot(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS + 1)
            .state,
        "open"
    );
    assert_eq!(
        svc.peer_telemetry
            .direct_peer_slo_snapshot(first_probe_at + DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS + 1,)
            .deliveries_total,
        3
    );
}

#[test]
fn direct_peer_circuit_open_state_survives_restart() {
    // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Restarting a node
    // during an outage must not reset target-bound admission to closed.
    let path = unique_test_db_path("direct-circuit-restart");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x73; 32]);
    let base = now_secs();
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        for offset in 0..3 {
            complete_direct_peer_test_delivery(
                &svc,
                base.saturating_add(offset),
                false,
                false,
                true,
            );
        }
        let status = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(status.state, "open");
        assert!(status.restart_protected);
        assert_eq!(status.opened_total, 1);
    }
    {
        let svc = ChatRelayService::new(config, secret).expect("restart relay");
        let status = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(status.state, "open");
        assert!(status.restart_protected);
        assert_eq!(status.opened_total, 1);
        assert!(status.checkpoint_loaded_at.is_some());
        assert!(svc
            .begin_direct_peer_delivery(base.saturating_add(3))
            .is_none());
    }
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_interrupted_half_open_probe_reopens_on_restart() {
    // A persisted in-flight probe has an unknowable outcome after process
    // loss, so startup classifies it failed and starts a fresh cooldown.
    let path = unique_test_db_path("direct-circuit-interrupted-probe");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x74; 32]);
    let now = now_secs();
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute(
                "UPDATE relay_direct_peer_circuit_checkpoint
                 SET state = 'half_open_in_flight',
                     successful_probes = 1,
                     deadline_at = ?1,
                     opened_total = 1,
                     half_open_attempted_total = 1,
                     half_open_succeeded_total = 1,
                     last_transition_at = ?2,
                     updated_at = ?2
                 WHERE singleton = 1",
                params![
                    sqlite_integer(
                        now.saturating_add(DIRECT_PEER_RELAY_HALF_OPEN_LEASE_SECS),
                        "test_probe_deadline"
                    )
                    .unwrap(),
                    sqlite_integer(now, "test_probe_updated_at").unwrap()
                ],
            )
            .expect("seed interrupted probe checkpoint");
    }
    {
        let svc = ChatRelayService::new(config, secret).expect("restart relay");
        let status = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(status.state, "open");
        assert!(status.restart_protected);
        assert_eq!(status.half_open_failed_total, 1);
        assert_eq!(status.opened_total, 2);
        assert_eq!(
            status.open_remaining_seconds,
            Some(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
        );
    }
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_half_open_progress_completes_across_restart() {
    // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] A completed probe
    // is safe to retain; restart may resume with one new serial probe but
    // may never treat an in-flight probe as completed.
    let path = unique_test_db_path("direct-circuit-half-open-progress");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x78; 32]);
    let now = now_secs();
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute(
                "UPDATE relay_direct_peer_circuit_checkpoint
                 SET state = 'half_open_ready',
                     successful_probes = 1,
                     opened_total = 1,
                     half_open_attempted_total = 1,
                     half_open_succeeded_total = 1,
                     last_transition_at = ?1,
                     updated_at = ?1
                 WHERE singleton = 1",
                params![sqlite_integer(now, "test_half_open_ready_time").unwrap()],
            )
            .expect("seed completed first probe");
    }
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("restart relay");
        let restored = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(restored.state, "half_open");
        assert_eq!(restored.half_open_consecutive_successes, 1);
        let permit = svc
            .begin_direct_peer_delivery(now)
            .expect("restored circuit should admit the second probe");
        assert!(permit.is_half_open());
        assert!(svc.complete_direct_peer_delivery(now, permit, false, true, false));
        let closed = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(closed.state, "closed");
        assert_eq!(closed.recovered_total, 1);
        assert!(closed.restart_protected);
    }
    {
        let svc = ChatRelayService::new(config, secret).expect("verify closed restart");
        let closed = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(closed.state, "closed");
        assert_eq!(closed.recovered_total, 1);
        assert!(closed.restart_protected);
    }
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_clock_rollback_recovers_fail_closed() {
    let path = unique_test_db_path("direct-circuit-clock-rollback");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x75; 32]);
    let now = now_secs();
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute(
                "UPDATE relay_direct_peer_circuit_checkpoint
                 SET updated_at = ?1
                 WHERE singleton = 1",
                params![sqlite_integer(now.saturating_add(120), "test_future_time").unwrap()],
            )
            .expect("seed future checkpoint timestamp");
    }
    {
        let svc = ChatRelayService::new(config, secret).expect("restart relay");
        let status = svc.peer_status().direct_peer_retry.circuit;
        assert_eq!(status.state, "open");
        assert!(status.restart_protected);
        assert_eq!(status.opened_total, 1);
        assert!(status
            .checkpoint_persisted_at
            .is_some_and(|value| value <= now_secs()));
    }
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_corrupt_checkpoint_rejects_relay_restart() {
    let path = unique_test_db_path("direct-circuit-corrupt");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x76; 32]);
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute(
                "UPDATE relay_direct_peer_circuit_checkpoint
                 SET state = 'closed', successful_probes = 1,
                     opened_total = 1,
                     half_open_attempted_total = 1,
                     half_open_succeeded_total = 1
                 WHERE singleton = 1",
                [],
            )
            .expect("seed semantically corrupt checkpoint");
    }
    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "direct_peer_circuit_checkpoint_state"
        })
    ));
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_missing_checkpoint_rejects_existing_schema() {
    // [DURABLE-DIRECT-RELAY-CIRCUIT 2026-08-15 by Codex] Only a first-time
    // schema upgrade may create the singleton. Its later disappearance is
    // corruption, not permission to reset an unknown circuit to closed.
    let path = unique_test_db_path("direct-circuit-missing");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x77; 32]);
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute(
                "DELETE FROM relay_direct_peer_circuit_checkpoint WHERE singleton = 1",
                [],
            )
            .expect("delete checkpoint singleton");
    }
    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "direct_peer_circuit_checkpoint_singleton"
        })
    ));
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_missing_checkpoint_table_rejects_installed_schema() {
    // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] The installation
    // marker distinguishes destructive table loss from a first upgrade.
    // Restart must not manufacture a closed checkpoint after that loss.
    let path = unique_test_db_path("direct-circuit-missing-table");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x78; 32]);
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute("DROP TABLE relay_direct_peer_circuit_checkpoint", [])
            .expect("remove installed checkpoint table");
    }
    assert!(matches!(
        ChatRelayService::new(config, secret),
        Err(ChatRelayError::CorruptStoredData {
            field: "direct_peer_circuit_checkpoint_table"
        })
    ));
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_existing_checkpoint_installs_missing_schema_sentinel() {
    // [DIRECT-RELAY-SCHEMA-SENTINEL 2026-08-16 by Codex] Deployed v2.3
    // databases already have a validated checkpoint but no feature marker.
    // Their first v2.4 startup installs the marker in the same transaction.
    let path = unique_test_db_path("direct-circuit-marker-upgrade");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x79; 32]);
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        svc.conn
            .lock()
            .execute(
                "DELETE FROM relay_schema_features WHERE feature = ?1",
                params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
            )
            .expect("simulate pre-sentinel database");
    }
    {
        let svc = ChatRelayService::new(config, secret).expect("upgrade existing relay");
        let installed_version = svc
            .conn
            .lock()
            .query_row(
                "SELECT schema_version FROM relay_schema_features WHERE feature = ?1",
                params![DIRECT_PEER_RELAY_CIRCUIT_SCHEMA_FEATURE],
                |row| row.get::<_, i64>(0),
            )
            .expect("read installed schema sentinel");
        assert_eq!(
            installed_version,
            DIRECT_PEER_RELAY_CIRCUIT_CHECKPOINT_VERSION
        );
    }
    remove_test_db(&path);
}

#[test]
fn direct_peer_circuit_runtime_checkpoint_failure_denies_delivery() {
    let svc = make_service();
    let base = now_secs();
    svc.conn
        .lock()
        .execute("DROP TABLE relay_direct_peer_circuit_checkpoint", [])
        .expect("remove checkpoint table");

    for offset in 0..3 {
        complete_direct_peer_test_delivery(&svc, base.saturating_add(offset), false, false, true);
    }
    let status = svc.peer_status().direct_peer_retry.circuit;
    assert_eq!(status.state, "open");
    assert!(!status.restart_protected);
    assert_eq!(status.checkpoint_failures_total, 1);
    assert_eq!(status.last_checkpoint_failure_at, Some(base + 2));
    assert!(svc
        .begin_direct_peer_delivery(
            base.saturating_add(2)
                .saturating_add(DIRECT_PEER_RELAY_CIRCUIT_COOLDOWN_SECS)
        )
        .is_none());
    let blocked = svc.peer_status().direct_peer_retry.circuit;
    assert_eq!(blocked.state, "open");
    assert!(!blocked.restart_protected);
    assert_eq!(blocked.checkpoint_failures_total, 2);
}

#[test]
fn test_peer_relay_inbound_health_tracks_accept_and_reject() {
    let svc = make_service();

    svc.record_peer_relay_inbound_accepted(1_800_000_010, false, 0, true);
    let status = svc.peer_status();
    assert_eq!(status.inbound_accepted_total, 1);
    assert_eq!(status.inbound_stored_pending_total, 1);
    assert_eq!(status.last_inbound_status.as_deref(), Some("accepted"));
    assert_eq!(status.last_inbound_failure_reason, None);

    svc.record_peer_relay_inbound_accepted(1_800_000_020, true, 0, false);
    let status = svc.peer_status();
    assert_eq!(status.inbound_accepted_total, 2);
    assert_eq!(status.inbound_duplicate_total, 1);
    assert_eq!(status.last_inbound_status.as_deref(), Some("duplicate"));

    svc.record_peer_relay_inbound_rejected(1_800_000_030, "invalid_signature");
    let status = svc.peer_status();
    assert_eq!(status.inbound_rejected_total, 1);
    assert_eq!(status.last_inbound_status.as_deref(), Some("rejected"));
    assert_eq!(
        status.last_inbound_failure_reason.as_deref(),
        Some("invalid_signature")
    );
}

fn make_envelope(kp: &IdentityKeyPair, receiver: [u8; 32]) -> ChatEnvelope {
    let mut env = ChatEnvelope {
        message_id: rand::random(),
        sender: kp.public_key_bytes(),
        receiver,
        timestamp: now_secs(),
        ciphertext: b"encrypted".to_vec(),
        nonce: [0x02; 24],
        content_type: ChatContentType::Text,
        signature: [0u8; 64],
    };
    let data = env.sign_data();
    env.signature = kp.sign(&data);
    env
}

fn make_session() -> SessionId {
    SessionId::from_bytes(&rand::random::<[u8; 16]>()).expect("random bytes form valid SessionId")
}

fn make_addr(port: u16) -> SocketAddr {
    format!("127.0.0.1:{}", port).parse().unwrap()
}

// ── Schema init ──────────────────────────────────────────────────────

#[test]
fn test_service_init() {
    let svc = make_service();
    let (m, b) = svc.run_cleanup().expect("cleanup");
    assert_eq!(m, 0);
    assert_eq!(b, 0);
}

#[test]
fn test_legacy_queue_sequence_migration_is_atomic_and_restart_stable() {
    let path = unique_test_db_path("sequence-migration");
    {
        let conn = Connection::open(&path).expect("open legacy database");
        conn.execute_batch(
            "CREATE TABLE pending_messages (
                message_id  BLOB(16) PRIMARY KEY,
                sender      BLOB(32) NOT NULL,
                receiver    BLOB(32) NOT NULL,
                timestamp   INTEGER NOT NULL,
                envelope    BLOB NOT NULL,
                received_at INTEGER NOT NULL,
                status      INTEGER NOT NULL DEFAULT 0
            );",
        )
        .expect("create legacy pending schema");
        for marker in [0x11_u8, 0x22] {
            conn.execute(
                "INSERT INTO pending_messages
                 (message_id, sender, receiver, timestamp, envelope, received_at, status)
                 VALUES (?1, ?2, ?3, 1, ?4, 1, 0)",
                params![
                    [marker; 16].as_slice(),
                    [0x31_u8; 32].as_slice(),
                    [0x41_u8; 32].as_slice(),
                    [0x51_u8].as_slice(),
                ],
            )
            .expect("insert legacy row");
        }
    }

    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x42; 32]);
    {
        let svc = ChatRelayService::new(config.clone(), secret).expect("migrate legacy queue");
        let conn = svc.conn.lock();
        let mut stmt = conn
            .prepare("SELECT queue_sequence FROM pending_messages ORDER BY rowid")
            .expect("prepare migrated sequence query");
        let sequences: Vec<i64> = stmt
            .query_map([], |row| row.get(0))
            .expect("query migrated sequences")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect migrated sequences");
        assert_eq!(sequences, vec![1, 2]);
        let last: i64 = conn
            .query_row(
                "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .expect("read migrated high-water mark");
        assert_eq!(last, 2);
    }
    {
        let svc = ChatRelayService::new(config, secret).expect("reopen migrated queue");
        let conn = svc.conn.lock();
        let mut stmt = conn
            .prepare("SELECT queue_sequence FROM pending_messages ORDER BY rowid")
            .expect("prepare restart sequence query");
        let sequences: Vec<i64> = stmt
            .query_map([], |row| row.get(0))
            .expect("query restart sequences")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect restart sequences");
        assert_eq!(sequences, vec![1, 2]);
    }
    remove_test_db(&path);
}

#[test]
fn chat_relay_logs_stay_free_of_routing_identifiers() {
    let source = include_str!("chat_relay.rs");
    let message_log = concat!("id = %hex::", "encode(envelope.message_id)");
    let receiver_log = concat!("receiver = %hex::", "encode");
    let sender_log = concat!("sender = %hex::", "encode");
    let blob_log = concat!("blob_id", " = %");

    for forbidden in [message_log, receiver_log, sender_log, blob_log] {
        assert!(
            !source.contains(forbidden),
            "relay logs must not expose stable routing identifiers"
        );
    }
}

// ── v1.3.0: wallet_routes field accessible ───────────────────────────

#[test]
fn test_wallet_routes_field_accessible() {
    let svc = make_service();
    let wallet = [0xAAu8; 32];
    let sid = make_session();
    let addr = make_addr(9000);

    // announce via the public field
    svc.wallet_routes.announce(&wallet, sid.clone(), addr);

    let results = svc.wallet_routes.lookup(&wallet);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, sid);
}

#[test]
fn test_wallet_routes_arc_clone_shares_state() {
    let svc = make_service();
    let routes_clone = Arc::clone(&svc.wallet_routes);

    let wallet = [0xBBu8; 32];
    let sid = make_session();

    // announce via original
    svc.wallet_routes
        .announce(&wallet, sid.clone(), make_addr(9001));

    // lookup via clone — must see the same entry
    let results = routes_clone.lookup(&wallet);
    assert_eq!(
        results.len(),
        1,
        "Arc clone must share the same underlying cache"
    );
}

// ── store → pull → ack (preserved) ───────────────────────────────────

#[test]
fn test_store_pull_ack_roundtrip() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBBu8; 32];
    let env = make_envelope(&kp, receiver);
    let mid = env.message_id;

    svc.store_pending(&env).expect("store");
    let usage = svc.storage_usage().expect("usage after store");
    assert_eq!(usage.pending_messages, 1);
    assert!(usage.pending_message_bytes > 0);

    let (msgs, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull");
    assert_eq!(msgs.len(), 1);
    assert!(!has_more);
    assert_eq!(msgs[0].message_id, mid);

    let deleted = svc.ack_messages(&[mid], &receiver).expect("ack");
    assert_eq!(deleted, 1);
    let usage = svc.storage_usage().expect("usage after ack");
    assert_eq!(usage.pending_messages, 0);
    assert_eq!(usage.pending_message_bytes, 0);

    let (msgs2, _) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull2");
    assert!(msgs2.is_empty());
}

#[test]
fn test_pull_isolates_malformed_row_and_delivers_valid_message() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBEu8; 32];
    let envelope = make_envelope(&kp, receiver);
    let expected_message_id = envelope.message_id;
    svc.store_pending(&envelope).expect("store valid message");
    svc.conn
        .lock()
        .execute(
            "INSERT INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status)
             VALUES (?1, ?2, ?3, 1, ?4, 1, 0)",
            params![
                [0x01u8; 15].as_slice(),
                kp.public_key_bytes().as_slice(),
                receiver.as_slice(),
                [0xFFu8].as_slice(),
            ],
        )
        .expect("insert malformed pending row");

    let (messages, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull with poison-row isolation");
    assert_eq!(messages.len(), 1);
    assert!(!has_more);
    assert_eq!(messages[0].message_id, expected_message_id);
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);

    let status = svc.maintenance_status();
    assert_eq!(status.quarantined_pending_messages_total, 1);
    assert_eq!(status.quarantine_events_retained, 1);
    let event: (String, String, i64) = svc
        .conn
        .lock()
        .query_row(
            "SELECT source_kind, reason, row_count FROM relay_quarantine_events",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("read pending quarantine event");
    assert_eq!(event.0, QUARANTINE_SOURCE_PENDING_MESSAGE);
    assert_eq!(event.1, "pending_message_id");
    assert_eq!(event.2, 1);
}

#[test]
fn test_pull_quarantines_message_id_envelope_mismatch() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBFu8; 32];
    let envelope = make_envelope(&kp, receiver);
    svc.store_pending(&envelope).expect("store valid message");
    svc.conn
        .lock()
        .execute(
            "UPDATE pending_messages SET message_id = ?1 WHERE message_id = ?2",
            params![[0xFEu8; 16].as_slice(), envelope.message_id.as_slice()],
        )
        .expect("tamper durable message id");

    let (messages, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull mismatched durable row");
    assert!(messages.is_empty());
    assert!(!has_more);
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

    let reason: String = svc
        .conn
        .lock()
        .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
            row.get(0)
        })
        .expect("read mismatch reason");
    assert_eq!(reason, "pending_message_id_mismatch");
}

#[test]
fn test_pull_quarantines_stored_sender_mismatch_before_delivery() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xC2u8; 32];
    let envelope = make_envelope(&kp, receiver);
    svc.store_pending(&envelope).expect("store valid message");
    svc.conn
        .lock()
        .execute(
            "UPDATE pending_messages SET sender = ?1 WHERE message_id = ?2",
            params![[0xF1u8; 32].as_slice(), envelope.message_id.as_slice()],
        )
        .expect("tamper durable sender");

    let (messages, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull mismatched durable sender");
    assert!(messages.is_empty());
    assert!(!has_more);
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

    let reason: String = svc
        .conn
        .lock()
        .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
            row.get(0)
        })
        .expect("read mismatch reason");
    assert_eq!(reason, "pending_message_sender_mismatch");
}

#[test]
fn test_pull_quarantines_invalid_durable_signature() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xC3u8; 32];
    let envelope = make_envelope(&kp, receiver);
    svc.store_pending(&envelope).expect("store valid message");
    let mut tampered = envelope.clone();
    tampered.signature[0] ^= 0xFF;
    let tampered_bytes = encode_envelope(&tampered).expect("encode tampered envelope");
    svc.conn
        .lock()
        .execute(
            "UPDATE pending_messages SET envelope = ?1 WHERE message_id = ?2",
            params![tampered_bytes, envelope.message_id.as_slice()],
        )
        .expect("tamper durable signature");

    let (messages, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull invalid durable signature");
    assert!(messages.is_empty());
    assert!(!has_more);
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

    let reason: String = svc
        .conn
        .lock()
        .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
            row.get(0)
        })
        .expect("read signature reason");
    assert_eq!(reason, "pending_message_signature");
}

#[test]
fn test_store_rejects_timestamp_outside_sqlite_domain() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBCu8; 32];
    let mut envelope = make_envelope(&kp, receiver);
    envelope.timestamp = u64::MAX;
    envelope.signature = kp.sign(&envelope.sign_data());

    let error = svc
        .store_pending(&envelope)
        .expect_err("out-of-range timestamp must be rejected");
    assert!(matches!(error, ChatRelayError::TimestampOutOfRange));
    assert_eq!(error.reason_bucket(), "timestamp_out_of_range");
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);
}

#[test]
fn test_pull_out_of_range_timestamp_fails_closed() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBDu8; 32];
    let mut envelope = make_envelope(&kp, receiver);
    envelope.timestamp = 1;
    envelope.signature = kp.sign(&envelope.sign_data());
    svc.store_pending(&envelope).expect("store pending message");

    let (messages, has_more) = svc
        .pull_pending(&receiver, u64::MAX, &[0u8; 16], 50)
        .expect("bounded pull");
    assert!(messages.is_empty());
    assert!(!has_more);
}

#[test]
fn test_store_duplicate_ignored() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBBu8; 32];
    let env = make_envelope(&kp, receiver);

    svc.store_pending(&env).expect("first store");
    svc.store_pending(&env)
        .expect("duplicate store — should not error");

    let (msgs, _) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull");
    assert_eq!(msgs.len(), 1);
}

#[test]
fn test_store_rejects_message_id_conflict_without_replacing_original() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xC0u8; 32];
    let original = make_envelope(&kp, receiver);
    let mut conflict = make_envelope(&kp, receiver);
    conflict.message_id = original.message_id;
    conflict.ciphertext = b"different ciphertext".to_vec();
    conflict.signature = kp.sign(&conflict.sign_data());

    svc.store_pending(&original).expect("store original");
    let error = svc
        .store_pending(&conflict)
        .expect_err("conflicting message id must fail");
    assert!(matches!(error, ChatRelayError::MessageIdConflict));
    assert_eq!(error.reason_bucket(), "message_id_conflict");
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);

    let (messages, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull original");
    assert_eq!(messages.len(), 1);
    assert!(!has_more);
    assert_eq!(messages[0].envelope.ciphertext, original.ciphertext);
}

#[test]
fn test_pull_zero_limit_makes_progress_with_minimum_page() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xC1u8; 32];
    let envelope = make_envelope(&kp, receiver);
    svc.store_pending(&envelope).expect("store message");

    let (messages, has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 0)
        .expect("zero limit pull");
    assert_eq!(messages.len(), 1);
    assert!(!has_more);
}

#[test]
fn test_store_enforces_configured_ciphertext_size_limit() {
    let mut config = test_config();
    config.max_message_size = 4;
    let svc = make_service_with_config(config);
    let kp = IdentityKeyPair::generate();
    let envelope = make_envelope(&kp, [0x10; 32]);

    assert!(matches!(
        svc.store_pending(&envelope),
        Err(ChatRelayError::MessageTooLarge { size: 9, limit: 4 })
    ));
    assert_eq!(
        svc.storage_usage().unwrap(),
        ChatRelayStorageUsage::default()
    );
}

#[test]
fn test_global_message_count_quota_preserves_duplicate_idempotence() {
    let mut config = test_config();
    config.max_pending_messages_total = 1;
    let svc = make_service_with_config(config);
    let kp = IdentityKeyPair::generate();
    let first = make_envelope(&kp, [0x11; 32]);

    svc.store_pending(&first).expect("first store");
    svc.store_pending(&first)
        .expect("duplicate remains successful at global capacity");

    let second = make_envelope(&kp, [0x22; 32]);
    assert!(matches!(
        svc.store_pending(&second),
        Err(ChatRelayError::PendingMessageQueueFull { .. })
    ));
    assert_eq!(svc.storage_usage().unwrap().pending_messages, 1);
}

#[test]
fn test_global_message_byte_quota_spans_distinct_receivers() {
    let kp = IdentityKeyPair::generate();
    let first = make_envelope(&kp, [0x31; 32]);
    let encoded_bytes = encode_envelope(&first).unwrap().len() as u64;
    let mut config = test_config();
    config.max_pending_message_bytes_total = encoded_bytes;
    let svc = make_service_with_config(config);

    svc.store_pending(&first).expect("first store");
    let second = make_envelope(&kp, [0x32; 32]);
    assert!(matches!(
        svc.store_pending(&second),
        Err(ChatRelayError::PendingMessageBytesExceeded { .. })
    ));
}

#[test]
fn test_storage_usage_reconciles_from_canonical_rows_on_restart() {
    let path = std::env::temp_dir().join(format!(
        "aeronyx-chat-relay-usage-{}.db",
        rand::random::<u64>()
    ));
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let kp = IdentityKeyPair::generate();
    let envelope = make_envelope(&kp, [0x61; 32]);

    {
        let svc = make_service_with_config(config.clone());
        svc.store_pending(&envelope).expect("store before restart");
    }
    {
        let conn = Connection::open(&path).expect("open usage database");
        conn.execute(
            "UPDATE relay_storage_usage
             SET pending_message_count = 0, pending_message_bytes = 0
             WHERE singleton = 1",
            [],
        )
        .expect("tamper derived usage row");
    }

    let restarted = make_service_with_config(config);
    let usage = restarted.storage_usage().expect("reconciled usage");
    assert_eq!(usage.pending_messages, 1);
    assert_eq!(
        usage.pending_message_bytes,
        encode_envelope(&envelope).unwrap().len() as u64
    );
    drop(restarted);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}-wal", path.display()));
    let _ = std::fs::remove_file(format!("{}-shm", path.display()));
}

#[test]
fn storage_usage_rejects_negative_durable_counter() {
    // [CHAT-RELAY-VERIFIED-BACKUP 2026-08-16 by Codex] The previous
    // conversion silently mapped a negative tampered counter to zero,
    // which could disguise unavailable quota state as spare capacity.
    let svc = make_service();
    svc.conn
        .lock()
        .execute_batch(
            "PRAGMA ignore_check_constraints=ON;
             UPDATE relay_storage_usage
             SET pending_blob_bytes = -1
             WHERE singleton = 1;",
        )
        .expect("install negative usage fixture");
    assert!(matches!(
        svc.storage_usage(),
        Err(ChatRelayError::CorruptStoredData {
            field: "pending_blob_bytes"
        })
    ));
}

#[test]
fn test_mailbox_full_rejected() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBBu8; 32];

    for _ in 0..5 {
        let env = make_envelope(&kp, receiver);
        svc.store_pending(&env).expect("store");
    }

    let env6 = make_envelope(&kp, receiver);
    let result = svc.store_pending(&env6);
    assert!(matches!(result, Err(ChatRelayError::MailboxFull { .. })));
}

#[test]
fn test_ack_wrong_receiver_cannot_delete() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBBu8; 32];
    let env = make_envelope(&kp, receiver);
    let mid = env.message_id;

    svc.store_pending(&env).expect("store");

    let wrong_receiver = [0xCCu8; 32];
    let deleted = svc.ack_messages(&[mid], &wrong_receiver).expect("ack");
    assert_eq!(deleted, 0, "Wrong receiver must not delete messages");

    let (msgs, _) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull");
    assert_eq!(msgs.len(), 1);
}

#[test]
fn test_ack_batch_is_atomic_and_deduplicated() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBD; 32];
    let first = make_envelope(&kp, receiver);
    let second = make_envelope(&kp, receiver);

    svc.store_pending(&first).expect("store first");
    svc.store_pending(&second).expect("store second");
    let deleted = svc
        .ack_messages(
            &[first.message_id, first.message_id, second.message_id],
            &receiver,
        )
        .expect("deduplicated ACK");

    assert_eq!(deleted, 2);
    assert_eq!(
        svc.storage_usage().expect("usage after ACK"),
        ChatRelayStorageUsage::default()
    );
}

#[test]
fn test_ack_batch_above_protocol_ceiling_is_rejected() {
    let svc = make_service();
    let ids = vec![[0x11; 16]; MAX_CHAT_ACK_MESSAGE_IDS + 1];

    assert!(matches!(
        svc.ack_messages(&ids, &[0xBE; 32]),
        Err(ChatRelayError::AckBatchTooLarge {
            size,
            limit: MAX_CHAT_ACK_MESSAGE_IDS,
        }) if size == MAX_CHAT_ACK_MESSAGE_IDS + 1
    ));
}

// ── Pagination (preserved) ───────────────────────────────────────────

#[test]
fn test_pull_pagination() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBBu8; 32];

    for _ in 0..5 {
        let env = make_envelope(&kp, receiver);
        svc.store_pending(&env).expect("store");
    }

    let (page1, has_more1) = svc.pull_pending(&receiver, 0, &[0u8; 16], 3).expect("p1");
    assert_eq!(page1.len(), 3);
    assert!(has_more1);

    let cursor = page1.last().unwrap().message_id;
    let (page2, has_more2) = svc.pull_pending(&receiver, 0, &cursor, 3).expect("p2");
    assert_eq!(page2.len(), 2);
    assert!(!has_more2);
}

#[test]
fn test_pull_cursor_does_not_skip_rows_across_timestamps() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBC; 32];
    let fixtures = [
        (100, [0xF0; 16]),
        (200, [0xE0; 16]),
        (300, [0xD0; 16]),
        (400, [0xC0; 16]),
    ];

    for (timestamp, message_id) in fixtures {
        let mut envelope = make_envelope(&kp, receiver);
        envelope.timestamp = timestamp;
        envelope.message_id = message_id;
        envelope.signature = kp.sign(&envelope.sign_data());
        svc.store_pending(&envelope).expect("store ordered fixture");
    }

    let (first_page, first_has_more) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 2)
        .expect("first cursor page");
    assert_eq!(first_page.len(), 2);
    assert!(first_has_more);

    let cursor = first_page.last().expect("first page cursor").message_id;
    let (second_page, second_has_more) = svc
        .pull_pending(&receiver, 0, &cursor, 2)
        .expect("second cursor page");
    assert_eq!(second_page.len(), 2);
    assert!(!second_has_more);

    let actual: HashSet<[u8; 16]> = first_page
        .iter()
        .chain(&second_page)
        .map(|message| message.message_id)
        .collect();
    let expected: HashSet<[u8; 16]> = fixtures
        .into_iter()
        .map(|(_, message_id)| message_id)
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn test_queue_sequence_is_monotonic_and_idempotent_retries_do_not_consume_it() {
    let svc = make_service();
    let identity = IdentityKeyPair::generate();
    let receiver = [0xC4; 32];
    let first = make_envelope(&identity, receiver);
    let second = make_envelope(&identity, receiver);
    let third = make_envelope(&identity, receiver);

    svc.store_pending(&first).expect("store first");
    svc.store_pending(&second).expect("store second");
    svc.store_pending(&first).expect("retry first idempotently");
    svc.store_pending(&third).expect("store third");

    let conn = svc.conn.lock();
    let sequences: Vec<i64> = conn
        .prepare("SELECT queue_sequence FROM pending_messages ORDER BY queue_sequence")
        .expect("prepare sequence query")
        .query_map([], |row| row.get(0))
        .expect("query sequences")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect sequences");
    assert_eq!(sequences, vec![1, 2, 3]);
    let last: i64 = conn
        .query_row(
            "SELECT last_sequence FROM relay_queue_sequence WHERE singleton = 1",
            [],
            |row| row.get(0),
        )
        .expect("read sequence high-water mark");
    assert_eq!(last, 3);
}

#[test]
fn test_pull_v2_snapshot_excludes_concurrent_inserts_without_skipping() {
    let svc = make_service();
    let identity = IdentityKeyPair::generate();
    let receiver = [0xC5; 32];
    let initial: Vec<ChatEnvelope> = (0..3).map(|_| make_envelope(&identity, receiver)).collect();
    for envelope in &initial {
        svc.store_pending(envelope).expect("store snapshot fixture");
    }

    let first_page = svc
        .pull_pending_v2(&receiver, 0, &[], 2)
        .expect("first v2 snapshot page");
    assert_eq!(first_page.messages.len(), 2);
    assert!(first_page.has_more);
    assert_eq!(first_page.next_cursor.len(), CHAT_PULL_CURSOR_V2_BYTES);

    let concurrent = make_envelope(&identity, receiver);
    svc.store_pending(&concurrent)
        .expect("store concurrent post-snapshot message");
    let second_page = svc
        .pull_pending_v2(&receiver, 0, &first_page.next_cursor, 2)
        .expect("second v2 snapshot page");
    assert_eq!(second_page.messages.len(), 1);
    assert!(!second_page.has_more);

    let delivered: HashSet<[u8; 16]> = first_page
        .messages
        .iter()
        .chain(&second_page.messages)
        .map(|message| message.message_id)
        .collect();
    let expected: HashSet<[u8; 16]> = initial.iter().map(|envelope| envelope.message_id).collect();
    assert_eq!(delivered, expected);
    assert!(!delivered.contains(&concurrent.message_id));

    let delivered_ids: Vec<[u8; 16]> = delivered.into_iter().collect();
    svc.ack_messages(&delivered_ids, &receiver)
        .expect("ack completed snapshot");
    let fresh_snapshot = svc
        .pull_pending_v2(&receiver, 0, &[], 10)
        .expect("start fresh snapshot");
    assert_eq!(fresh_snapshot.messages.len(), 1);
    assert_eq!(fresh_snapshot.messages[0].message_id, concurrent.message_id);
}

#[test]
fn test_pull_v2_cursor_rejects_tampering_and_binding_replay() {
    let svc = make_service();
    let identity = IdentityKeyPair::generate();
    let receiver = [0xC6; 32];
    for _ in 0..2 {
        svc.store_pending(&make_envelope(&identity, receiver))
            .expect("store cursor fixture");
    }
    let page = svc
        .pull_pending_v2(&receiver, 0, &[], 1)
        .expect("issue cursor");
    let decoded = svc
        .decode_pull_cursor_v2(&receiver, 0, &page.next_cursor)
        .expect("decode server-owned cursor in test");
    assert_eq!(
        decoded,
        PullCursorV2 {
            position: 1,
            ceiling: 2
        }
    );
    assert!(!page
        .next_cursor
        .windows(8)
        .any(|window| window == decoded.position.to_le_bytes()));
    assert!(!page
        .next_cursor
        .windows(8)
        .any(|window| window == decoded.ceiling.to_le_bytes()));

    let mut tampered = page.next_cursor.clone();
    let last = tampered.last_mut().expect("non-empty cursor");
    *last ^= 0x01;
    assert!(matches!(
        svc.pull_pending_v2(&receiver, 0, &tampered, 1),
        Err(ChatRelayError::InvalidPullCursor)
    ));
    assert!(matches!(
        svc.pull_pending_v2(&[0xC7; 32], 0, &page.next_cursor, 1),
        Err(ChatRelayError::InvalidPullCursor)
    ));
    assert!(matches!(
        svc.pull_pending_v2(&receiver, 1, &page.next_cursor, 1),
        Err(ChatRelayError::InvalidPullCursor)
    ));

    let foreign =
        ChatRelayService::new(test_config(), [0x91; 32]).expect("create foreign-key relay");
    assert!(matches!(
        foreign.pull_pending_v2(&receiver, 0, &page.next_cursor, 1),
        Err(ChatRelayError::InvalidPullCursor)
    ));
}

#[test]
fn test_pull_v2_cursor_survives_restart_with_same_node_secret() {
    let path = unique_test_db_path("cursor-restart");
    let mut config = test_config();
    config.db_path = path.to_string_lossy().into_owned();
    let secret = derive_node_secret(&[0x62; 32]);
    let identity = IdentityKeyPair::generate();
    let receiver = [0xC8; 32];
    let cursor = {
        let svc = ChatRelayService::new(config.clone(), secret).expect("create relay");
        for _ in 0..3 {
            svc.store_pending(&make_envelope(&identity, receiver))
                .expect("store restart fixture");
        }
        let page = svc
            .pull_pending_v2(&receiver, 0, &[], 2)
            .expect("issue pre-restart cursor");
        assert!(page.has_more);
        page.next_cursor
    };
    {
        let svc = ChatRelayService::new(config, secret).expect("restart relay");
        let page = svc
            .pull_pending_v2(&receiver, 0, &cursor, 2)
            .expect("resume cursor after restart");
        assert_eq!(page.messages.len(), 1);
        assert!(!page.has_more);
    }
    remove_test_db(&path);
}

#[test]
fn test_pull_v2_quarantines_poison_row_and_advances_snapshot() {
    let svc = make_service();
    let identity = IdentityKeyPair::generate();
    let receiver = [0xC9; 32];
    let poison = make_envelope(&identity, receiver);
    let valid = make_envelope(&identity, receiver);
    svc.store_pending(&poison).expect("store poison fixture");
    svc.store_pending(&valid).expect("store valid fixture");
    svc.conn
        .lock()
        .execute(
            "UPDATE pending_messages SET envelope = ?1 WHERE message_id = ?2",
            params![[0xFF_u8].as_slice(), poison.message_id.as_slice()],
        )
        .expect("corrupt first sequence row");

    let first = svc
        .pull_pending_v2(&receiver, 0, &[], 1)
        .expect("pull through poison row");
    assert_eq!(first.messages.len(), 1);
    assert_eq!(first.messages[0].message_id, valid.message_id);
    assert!(first.has_more);
    let second = svc
        .pull_pending_v2(&receiver, 0, &first.next_cursor, 1)
        .expect("complete snapshot after poison quarantine");
    assert!(second.messages.is_empty());
    assert!(!second.has_more);
    assert_eq!(
        svc.maintenance_status().quarantined_pending_messages_total,
        1
    );
}

#[test]
fn test_queue_sequence_exhaustion_rolls_back_message_insert() {
    let svc = make_service();
    svc.conn
        .lock()
        .execute(
            "UPDATE relay_queue_sequence SET last_sequence = ?1 WHERE singleton = 1",
            params![i64::MAX],
        )
        .expect("force sequence exhaustion");
    let identity = IdentityKeyPair::generate();
    let envelope = make_envelope(&identity, [0xCA; 32]);
    assert!(matches!(
        svc.store_pending(&envelope),
        Err(ChatRelayError::QueueSequenceExhausted)
    ));
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);
}

// ── Blob (preserved) ─────────────────────────────────────────────────

#[test]
fn test_blob_put_get_delete() {
    let svc = make_service();
    let sender = [0xAAu8; 32];
    let receiver = [0xBBu8; 32];
    let data = b"encrypted_image_bytes";
    let file_hash: [u8; 32] = Sha256::digest(data).into();

    let blob_id = svc
        .put_blob(&sender, &receiver, data, &file_hash)
        .expect("put");
    assert_eq!(blob_id.len(), 32);
    let usage = svc.storage_usage().expect("usage after blob put");
    assert_eq!(usage.pending_blobs, 1);
    assert_eq!(usage.pending_blob_bytes, data.len() as u64);

    let fetched = svc.get_blob(&blob_id).expect("get");
    assert_eq!(fetched, data);

    svc.delete_blob(&blob_id, &sender).expect("delete");
    let usage = svc.storage_usage().expect("usage after blob delete");
    assert_eq!(usage.pending_blobs, 0);
    assert_eq!(usage.pending_blob_bytes, 0);
    assert!(matches!(
        svc.get_blob(&blob_id),
        Err(ChatRelayError::BlobNotFound { .. })
    ));
}

#[test]
fn test_blob_too_large_rejected() {
    let svc = make_service();
    let sender = [0xAAu8; 32];
    let receiver = [0xBBu8; 32];
    let data = vec![0u8; 2048];
    let file_hash: [u8; 32] = Sha256::digest(&data).into();

    let result = svc.put_blob(&sender, &receiver, &data, &file_hash);
    assert!(matches!(result, Err(ChatRelayError::BlobTooLarge { .. })));
}

#[test]
fn test_global_blob_count_quota_preserves_duplicate_idempotence() {
    let mut config = test_config();
    config.max_pending_blobs_total = 1;
    let svc = make_service_with_config(config);
    let sender = [0x41; 32];
    let first_receiver = [0x42; 32];
    let first_data = b"first encrypted blob";
    let first_hash: [u8; 32] = Sha256::digest(first_data).into();

    let first_id = svc
        .put_blob(&sender, &first_receiver, first_data, &first_hash)
        .expect("first put");
    let duplicate_id = svc
        .put_blob(&sender, &first_receiver, first_data, &first_hash)
        .expect("duplicate remains successful at global capacity");
    assert_eq!(duplicate_id, first_id);

    let second_data = b"second encrypted blob";
    let second_hash: [u8; 32] = Sha256::digest(second_data).into();
    assert!(matches!(
        svc.put_blob(&sender, &[0x43; 32], second_data, &second_hash),
        Err(ChatRelayError::PendingBlobStoreFull { .. })
    ));
    assert_eq!(svc.storage_usage().unwrap().pending_blobs, 1);
}

#[test]
fn test_global_blob_byte_quota_spans_distinct_receivers() {
    let data = b"bounded encrypted blob";
    let mut config = test_config();
    config.max_pending_blob_bytes_total = data.len() as u64;
    let svc = make_service_with_config(config);
    let sender = [0x51; 32];
    let first_hash: [u8; 32] = Sha256::digest(data).into();

    svc.put_blob(&sender, &[0x52; 32], data, &first_hash)
        .expect("first put");
    let second_hash: [u8; 32] = Sha256::digest(b"different hash").into();
    assert!(matches!(
        svc.put_blob(&sender, &[0x53; 32], data, &second_hash),
        Err(ChatRelayError::PendingBlobBytesExceeded { .. })
    ));
}

#[test]
fn test_blob_delete_wrong_sender_rejected() {
    let svc = make_service();
    let sender = [0xAAu8; 32];
    let receiver = [0xBBu8; 32];
    let data = b"file";
    let file_hash: [u8; 32] = Sha256::digest(data).into();

    let blob_id = svc
        .put_blob(&sender, &receiver, data, &file_hash)
        .expect("put");
    let wrong = [0xCCu8; 32];
    assert!(matches!(
        svc.delete_blob(&blob_id, &wrong),
        Err(ChatRelayError::Unauthorized)
    ));
}

// ── Online dedup (preserved) ─────────────────────────────────────────

#[test]
fn test_online_dedup() {
    let svc = make_service();
    let id = [0x01u8; 16];
    assert!(!svc.is_online_duplicate(&id));
    assert!(svc.is_online_duplicate(&id));
}

#[test]
fn test_online_dedup_is_atomic_under_concurrency() {
    const WORKERS: usize = 16;
    let dedup = Arc::new(MessageDedup::new(32));
    let barrier = Arc::new(Barrier::new(WORKERS));
    let id = [0x02u8; 16];
    let handles: Vec<_> = (0..WORKERS)
        .map(|_| {
            let dedup = Arc::clone(&dedup);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                dedup.check_and_insert(&id)
            })
        })
        .collect();

    let duplicate_results = handles
        .into_iter()
        .map(|handle| handle.join().expect("dedup worker must not panic"))
        .collect::<Vec<_>>();
    assert_eq!(
        duplicate_results
            .iter()
            .filter(|is_duplicate| !**is_duplicate)
            .count(),
        1,
        "exactly one concurrent caller must win first delivery"
    );
}

// ── TTL cleanup (preserved) ──────────────────────────────────────────

#[test]
fn test_cleanup_does_not_touch_fresh_messages() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xBBu8; 32];
    let env = make_envelope(&kp, receiver);

    svc.store_pending(&env).expect("store");
    let (expired, blobs) = svc.run_cleanup().expect("cleanup");
    assert_eq!(expired, 0);
    assert_eq!(blobs, 0);

    let (msgs, _) = svc
        .pull_pending(&receiver, 0, &[0u8; 16], 50)
        .expect("pull");
    assert_eq!(msgs.len(), 1);
}

#[test]
fn test_cleanup_chunks_expiry_notifications_and_reconciles_usage() {
    let mut config = test_config();
    config.max_pending_per_wallet = 100;
    let svc = make_service_with_config(config);
    let kp = IdentityKeyPair::generate();
    let sender = kp.public_key_bytes();
    let receiver = [0xC1; 32];
    let mut expected_ids = HashSet::new();

    for _ in 0..70 {
        let envelope = make_envelope(&kp, receiver);
        expected_ids.insert(envelope.message_id);
        svc.store_pending(&envelope)
            .expect("store expiring message");
    }
    svc.conn
        .lock()
        .execute("UPDATE pending_messages SET received_at = 0", [])
        .expect("age pending messages");

    let (expired, blobs) = svc.run_cleanup().expect("cleanup expired messages");
    assert_eq!(expired, 70);
    assert_eq!(blobs, 0);
    assert_eq!(
        svc.storage_usage().expect("usage after cleanup"),
        ChatRelayStorageUsage::default()
    );

    let (notifications, has_more) = svc
        .pull_pending_notifications(&sender)
        .expect("pull expiry notifications");
    assert!(!has_more);
    assert_eq!(notifications.len(), 3);

    let mut chunk_lengths = Vec::new();
    let mut actual_ids = HashSet::new();
    for notification in &notifications {
        assert_eq!(notification.sender, sender);
        assert_eq!(notification.receiver, receiver);
        let ids = notification.message_ids().expect("decode notification");
        chunk_lengths.push(ids.len());
        actual_ids.extend(ids);
    }
    chunk_lengths.sort_unstable();
    assert_eq!(chunk_lengths, vec![6, 32, 32]);
    assert_eq!(actual_ids, expected_ids);

    let pending_rows: i64 = svc
        .conn
        .lock()
        .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
            row.get(0)
        })
        .expect("count pending rows");
    assert_eq!(pending_rows, 0);
}

#[test]
fn test_cleanup_quarantines_malformed_row_without_blocking() {
    let svc = make_service();
    svc.conn
        .lock()
        .execute(
            "INSERT INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status)
             VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
            params![
                [0xA1u8; 15].as_slice(),
                [0xA2u8; 32].as_slice(),
                [0xA3u8; 32].as_slice(),
                [0xA4u8].as_slice(),
            ],
        )
        .expect("insert malformed durable row");

    assert_eq!(svc.run_cleanup().expect("quarantine cleanup"), (0, 0));
    assert_eq!(
        svc.storage_usage().expect("usage after quarantine"),
        ChatRelayStorageUsage::default()
    );

    let status = svc.maintenance_status();
    assert_eq!(status.cleanup_runs_total, 1);
    assert_eq!(status.cleanup_failures_total, 0);
    assert_eq!(status.cleanup_batches_total, 1);
    assert_eq!(status.quarantined_pending_messages_total, 1);
    assert_eq!(status.quarantine_events_retained, 1);
    assert_eq!(status.last_cleanup_quarantined_pending_messages, 1);
    assert!(status.last_quarantine_at.is_some());
    assert_eq!(status.last_cleanup_status.as_deref(), Some("succeeded"));

    let conn = svc.conn.lock();
    let pending_rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
            row.get(0)
        })
        .expect("count pending rows");
    assert_eq!(pending_rows, 0);
    let event: (String, String, i64, i64) = conn
        .query_row(
            "SELECT source_kind, reason, row_count, encoded_bytes
             FROM relay_quarantine_events",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .expect("read de-identified quarantine event");
    assert_eq!(event.0, QUARANTINE_SOURCE_PENDING_MESSAGE);
    assert_eq!(event.1, "expired_message_id");
    assert_eq!(event.2, 1);
    assert!(event.3 > 0);

    let mut schema_stmt = conn
        .prepare("PRAGMA table_info(relay_quarantine_events)")
        .expect("prepare quarantine schema query");
    let columns: Vec<String> = schema_stmt
        .query_map([], |row| row.get(1))
        .expect("query quarantine columns")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect quarantine columns");
    for forbidden in ["message_id", "sender", "receiver", "envelope", "ciphertext"] {
        assert!(!columns.iter().any(|column| column == forbidden));
    }
}

#[test]
fn test_cleanup_does_not_notify_tampered_stored_sender() {
    let svc = make_service();
    let kp = IdentityKeyPair::generate();
    let receiver = [0xD4u8; 32];
    let envelope = make_envelope(&kp, receiver);
    let tampered_sender = [0xD5u8; 32];
    svc.store_pending(&envelope).expect("store valid message");
    svc.conn
        .lock()
        .execute(
            "UPDATE pending_messages
             SET sender = ?1, received_at = 0
             WHERE message_id = ?2",
            params![tampered_sender.as_slice(), envelope.message_id.as_slice()],
        )
        .expect("tamper expired message sender");

    assert_eq!(svc.run_cleanup().expect("cleanup tampered sender"), (0, 0));
    let (notifications, has_more) = svc
        .pull_pending_notifications(&tampered_sender)
        .expect("pull attacker notifications");
    assert!(notifications.is_empty());
    assert!(!has_more);
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 0);

    let reason: String = svc
        .conn
        .lock()
        .query_row("SELECT reason FROM relay_quarantine_events", [], |row| {
            row.get(0)
        })
        .expect("read cleanup mismatch reason");
    assert_eq!(reason, "expired_message_sender_mismatch");
}

#[test]
fn test_cleanup_defers_backlog_at_batch_budget_and_recovers_next_run() {
    let svc = make_service();
    insert_expired_pending_rows(&svc, CLEANUP_MESSAGE_BATCH_SIZE + 1, 0x10);

    let (first_expired, first_blobs) = svc
        .run_cleanup_with_batch_budget(1)
        .expect("first bounded cleanup");
    assert_eq!(first_expired, CLEANUP_MESSAGE_BATCH_SIZE);
    assert_eq!(first_blobs, 0);
    assert_eq!(
        svc.storage_usage().expect("first usage").pending_messages,
        1
    );

    let deferred = svc.maintenance_status();
    assert_eq!(deferred.cleanup_runs_total, 1);
    assert_eq!(deferred.cleanup_batches_total, 1);
    assert_eq!(deferred.cleanup_backlog_deferred_total, 1);
    assert_eq!(
        deferred.expired_messages_total,
        u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(u64::MAX)
    );
    assert_eq!(deferred.last_cleanup_batches, 1);
    assert!(deferred.last_cleanup_backlog_deferred);

    let (second_expired, second_blobs) = svc
        .run_cleanup_with_batch_budget(1)
        .expect("second bounded cleanup");
    assert_eq!(second_expired, 1);
    assert_eq!(second_blobs, 0);
    assert_eq!(
        svc.storage_usage().expect("second usage").pending_messages,
        0
    );

    let recovered = svc.maintenance_status();
    assert_eq!(recovered.cleanup_runs_total, 2);
    assert_eq!(recovered.cleanup_batches_total, 2);
    assert_eq!(recovered.cleanup_backlog_deferred_total, 1);
    assert_eq!(
        recovered.expired_messages_total,
        u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE + 1).unwrap_or(u64::MAX)
    );
    assert!(!recovered.last_cleanup_backlog_deferred);
}

#[test]
fn test_cleanup_isolates_trailing_poison_row_after_committed_batch() {
    let svc = make_service();
    insert_expired_pending_rows(&svc, CLEANUP_MESSAGE_BATCH_SIZE, 0x10);
    svc.conn
        .lock()
        .execute(
            "INSERT INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status)
             VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
            params![
                [0xF0u8; 15].as_slice(),
                [0xA2u8; 32].as_slice(),
                [0xA3u8; 32].as_slice(),
                [0xA4u8].as_slice(),
            ],
        )
        .expect("insert malformed trailing row");

    let (expired, blobs) = svc
        .run_cleanup_with_batch_budget(2)
        .expect("bounded cleanup with poison-row isolation");
    assert_eq!(expired, CLEANUP_MESSAGE_BATCH_SIZE);
    assert_eq!(blobs, 0);

    let status = svc.maintenance_status();
    assert_eq!(status.cleanup_runs_total, 1);
    assert_eq!(status.cleanup_failures_total, 0);
    assert_eq!(status.cleanup_batches_total, 2);
    assert_eq!(
        status.expired_messages_total,
        u64::try_from(CLEANUP_MESSAGE_BATCH_SIZE).unwrap_or(u64::MAX)
    );
    assert_eq!(status.quarantined_pending_messages_total, 1);
    assert_eq!(status.last_cleanup_batches, 2);
    assert_eq!(status.last_cleanup_status.as_deref(), Some("succeeded"));
    assert_eq!(
        svc.storage_usage()
            .expect("remaining usage")
            .pending_messages,
        0
    );
}

#[test]
fn test_quarantine_persistence_failure_rolls_back_source_deletion() {
    let svc = make_service();
    svc.conn
        .lock()
        .execute(
            "INSERT INTO pending_messages
             (message_id, sender, receiver, timestamp, envelope, received_at, status)
             VALUES (?1, ?2, ?3, 0, ?4, 0, 0)",
            params![
                [0xA1u8; 15].as_slice(),
                [0xA2u8; 32].as_slice(),
                [0xA3u8; 32].as_slice(),
                [0xA4u8].as_slice(),
            ],
        )
        .expect("insert malformed durable row");
    svc.conn
        .lock()
        .execute("DROP TABLE relay_quarantine_events", [])
        .expect("simulate quarantine persistence failure");

    assert!(matches!(svc.run_cleanup(), Err(ChatRelayError::Sqlite(_))));
    assert_eq!(svc.storage_usage().expect("usage").pending_messages, 1);
    let pending_rows: i64 = svc
        .conn
        .lock()
        .query_row("SELECT COUNT(*) FROM pending_messages", [], |row| {
            row.get(0)
        })
        .expect("count retained source rows");
    assert_eq!(pending_rows, 1);
    let status = svc.maintenance_status();
    assert_eq!(status.cleanup_failures_total, 1);
    assert_eq!(status.quarantined_pending_messages_total, 0);
}

#[test]
fn test_quarantine_event_store_enforces_hard_retention_cap() {
    let svc = make_service();
    {
        let mut conn = svc.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .expect("start quarantine event insert");
        let mut stmt = tx
            .prepare(
                "INSERT INTO relay_quarantine_events
                 (source_kind, reason, row_count, encoded_bytes, quarantined_at)
                 VALUES (?1, 'test_reason', 1, 1, ?2)",
            )
            .expect("prepare quarantine event insert");
        for _ in 0..=MAX_QUARANTINE_EVENTS {
            stmt.execute(params![QUARANTINE_SOURCE_PENDING_MESSAGE, i64::MAX])
                .expect("insert quarantine event");
        }
        drop(stmt);
        tx.commit().expect("commit quarantine events");
    }

    svc.run_cleanup_with_batch_budget(1)
        .expect("trim quarantine event overflow");
    let retained: i64 = svc
        .conn
        .lock()
        .query_row("SELECT COUNT(*) FROM relay_quarantine_events", [], |row| {
            row.get(0)
        })
        .expect("count bounded quarantine events");
    assert_eq!(
        retained,
        i64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(i64::MAX)
    );
    let status = svc.maintenance_status();
    assert_eq!(status.quarantine_events_removed_total, 1);
    assert_eq!(
        status.quarantine_events_retained,
        u64::try_from(MAX_QUARANTINE_EVENTS).unwrap_or(u64::MAX)
    );
    assert!(!status.last_cleanup_backlog_deferred);
}

#[test]
fn test_cleanup_out_of_range_ttl_fails_closed() {
    let mut config = test_config();
    config.offline_ttl_secs = u64::MAX;
    let svc = ChatRelayService::new(config, [0x42; 32]).expect("service");
    let kp = IdentityKeyPair::generate();
    let receiver = [0xB4; 32];
    let envelope = make_envelope(&kp, receiver);

    svc.store_pending(&envelope).expect("store pending message");
    let (expired, _) = svc.run_cleanup().expect("cleanup");

    assert_eq!(expired, 0);
    assert_eq!(
        svc.storage_usage().expect("storage usage").pending_messages,
        1
    );
}

#[test]
fn test_maintenance_status_deserializes_older_snapshot() {
    let status: ChatRelayMaintenanceStatus = serde_json::from_value(serde_json::json!({
        "cleanup_runs_total": 7,
        "last_cleanup_status": "succeeded"
    }))
    .expect("deserialize backward-compatible maintenance snapshot");

    assert_eq!(status.cleanup_runs_total, 7);
    assert_eq!(status.cleanup_batches_total, 0);
    assert_eq!(status.quarantined_pending_messages_total, 0);
    assert_eq!(status.quarantine_events_retained, 0);
    assert!(!status.last_cleanup_backlog_deferred);
}

#[test]
fn test_expiry_notification_pull_is_bounded_and_pageable() {
    let svc = make_service();
    let sender = [0xD1; 32];
    let receiver = [0xD2u8; 32];
    {
        let mut conn = svc.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .expect("start notification insert");
        for created_at in 0..17i64 {
            let ids =
                bincode::serialize(&vec![[created_at as u8; 16]]).expect("serialize notification");
            tx.execute(
                "INSERT INTO expired_notifications
                 (sender, receiver, message_ids, created_at, pushed)
                 VALUES (?1, ?2, ?3, ?4, 0)",
                params![sender.as_slice(), receiver.as_slice(), ids, created_at],
            )
            .expect("insert notification");
        }
        tx.commit().expect("commit notifications");
    }

    let (first_page, first_has_more) = svc
        .pull_pending_notifications(&sender)
        .expect("first notification page");
    assert_eq!(first_page.len(), MAX_EXPIRED_NOTIFICATIONS_PER_PULL);
    assert!(first_has_more);
    let first_ids: Vec<i64> = first_page
        .iter()
        .map(|notification| notification.id)
        .collect();
    svc.mark_notifications_pushed(&first_ids)
        .expect("mark first page");

    let (second_page, second_has_more) = svc
        .pull_pending_notifications(&sender)
        .expect("second notification page");
    assert_eq!(second_page.len(), 1);
    assert!(!second_has_more);
}

#[test]
fn test_malformed_expiry_notification_isolated_without_blocking_valid_rows() {
    let svc = make_service();
    let sender = [0xE1; 32];
    let valid_receiver = [0xE4; 32];
    let ids = bincode::serialize(&vec![[0xE2; 16]]).expect("serialize notification");
    {
        let mut conn = svc.conn.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .expect("start mixed notification transaction");
        tx.execute(
            "INSERT INTO expired_notifications
             (sender, receiver, message_ids, created_at, pushed)
             VALUES (?1, ?2, ?3, 0, 0)",
            params![sender.as_slice(), [0xE3u8; 31].as_slice(), &ids],
        )
        .expect("insert malformed notification");
        tx.execute(
            "INSERT INTO expired_notifications
             (sender, receiver, message_ids, created_at, pushed)
             VALUES (?1, ?2, ?3, 1, 0)",
            params![sender.as_slice(), valid_receiver.as_slice(), ids],
        )
        .expect("insert valid notification");
        tx.commit().expect("commit mixed notifications");
    }

    let (notifications, has_more) = svc
        .pull_pending_notifications(&sender)
        .expect("pull must isolate poison row");
    assert_eq!(notifications.len(), 1);
    assert!(!has_more);
    assert_eq!(notifications[0].receiver, valid_receiver);

    let status = svc.maintenance_status();
    assert_eq!(status.quarantined_expired_notifications_total, 1);
    assert_eq!(status.quarantine_events_retained, 1);
    assert!(status.last_quarantine_at.is_some());

    let conn = svc.conn.lock();
    let remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM expired_notifications WHERE pushed = 0",
            [],
            |row| row.get(0),
        )
        .expect("count valid notification");
    assert_eq!(remaining, 1);
    let event: (String, String, i64) = conn
        .query_row(
            "SELECT source_kind, reason, row_count FROM relay_quarantine_events",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("read notification quarantine event");
    assert_eq!(event.0, QUARANTINE_SOURCE_EXPIRED_NOTIFICATION);
    assert_eq!(event.1, "expired_notification_receiver");
    assert_eq!(event.2, 1);
}

#[test]
fn expired_notifications_are_wired_to_authenticated_chat_pull() {
    let source = include_str!("../server.rs");
    assert!(source.contains("relay.pull_pending_notifications(&wallet)"));
    assert!(source.contains("Self::push_expired_notifications("));
    assert!(source.contains("has_more |= notification_has_more || !delivery_complete"));
    assert!(source.contains("self.spawn_chat_relay_cleanup_task(Arc::clone(relay))"));
    assert!(source.contains("tokio::task::spawn_blocking(move || cleanup_relay.run_cleanup())"));
    assert!(source.contains("tokio::time::MissedTickBehavior::Skip"));
    assert!(source.contains("relay.record_maintenance_worker_failure(reason)"));
    assert!(source.contains("\"maintenance\": relay.maintenance_status()"));
}

// ── node_secret derivation (preserved) ───────────────────────────────

#[test]
fn test_derive_node_secret_deterministic() {
    let sk = [0x42u8; 32];
    let s1 = derive_node_secret(&sk);
    let s2 = derive_node_secret(&sk);
    assert_eq!(s1, s2);
}

#[test]
fn test_derive_node_secret_different_keys() {
    let s1 = derive_node_secret(&[0x01u8; 32]);
    let s2 = derive_node_secret(&[0x02u8; 32]);
    assert_ne!(s1, s2);
}
