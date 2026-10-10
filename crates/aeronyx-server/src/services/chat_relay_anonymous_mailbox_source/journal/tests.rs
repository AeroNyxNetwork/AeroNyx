// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/journal/tests.rs
// ============================================
//! # Tests: encrypted `SQLite` source journal
//!
//! Unit tests for exact replay, phase CAS, terminal retention, cleanup, startup
//! audit, bounded state and private-inode open, moved from the former inline
//! `tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use super::*;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    encode_anonymous_mailbox_terminal_frame, AnonymousMailboxOperationV1,
    AnonymousMailboxOutcomeV1, AnonymousMailboxPullOneV1, AnonymousMailboxPullResultV1,
    AnonymousMailboxTerminalFrameV1, AnonymousMailboxTerminalResponseV1,
    MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES, MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES,
};

use super::super::storage::SOURCE_JOURNAL_SCHEMA_VERSION;
use super::super::test_support::{
    journal, journal_record, journal_with_limits, journal_with_max_bytes, retain_terminal_record,
    target, ticket_request, NOW,
};

fn create_legacy_source_schema(connection: &Connection) {
    connection
        .execute_batch(
            "CREATE TABLE anonymous_mailbox_source_journal (
                    route_id BLOB PRIMARY KEY NOT NULL CHECK (length(route_id) = 16),
                    request_commitment BLOB NOT NULL CHECK (length(request_commitment) = 32),
                    target_node_id BLOB NOT NULL CHECK (length(target_node_id) = 32),
                    descriptor_commitment BLOB NOT NULL CHECK (length(descriptor_commitment) > 0),
                    body BLOB NOT NULL CHECK (length(body) > 0),
                    phase INTEGER NOT NULL CHECK (phase BETWEEN 1 AND 5),
                    state_nonce BLOB NOT NULL CHECK (length(state_nonce) = 24),
                    protected_state BLOB NOT NULL CHECK (length(protected_state) >= 16)
                 );
                 PRAGMA user_version = 1;",
        )
        .expect("legacy source schema");
}

fn insert_legacy_source_record(
    connection: &Connection,
    sealer: &SqliteAnonymousMailboxSourceJournal,
    record: &SourceJournalRecord,
) {
    let descriptor = bincode::serialize(&record.descriptor_commitment).expect("legacy descriptor");
    let (nonce, protected) = sealer
        .seal_state(
            &record.route_id,
            &record.request_commitment,
            &record.target_node_id,
            &record.state,
        )
        .expect("legacy protected state");
    connection
        .execute(
            "INSERT INTO anonymous_mailbox_source_journal
                   (route_id, request_commitment, target_node_id, descriptor_commitment, body,
                    phase, state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                record.route_id.as_slice(),
                record.request_commitment.as_slice(),
                record.target_node_id.as_slice(),
                descriptor,
                &record.body,
                record.phase.code(),
                nonce,
                protected
            ],
        )
        .expect("legacy source row");
}

#[test]
fn journal_exact_replay_is_stable_and_conflict_does_not_overwrite() {
    let target = target();
    let terminal = ticket_request(&target);
    let body = vec![0x34; 73];
    let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
        node_id: target.public_key_bytes(),
        sequence: 7,
        descriptor_hash: [0x33; 32],
    };
    let record = SourceJournalRecord {
        route_id: [0x31; 16],
        request_commitment: source_request_commitment(
            &[0x31; 16],
            &target.public_key_bytes(),
            &descriptor_commitment,
            &terminal,
        ),
        target_node_id: target.public_key_bytes(),
        descriptor_commitment,
        body: body.clone(),
        phase: AnonymousMailboxSourcePhase::Prepared,
        retain_until: None,
        state: encode_state(&body, &terminal, None, None).expect("state"),
    };
    let journal = journal();
    let first = journal.insert_or_exact(&record).expect("insert");
    let replay = journal.insert_or_exact(&record).expect("exact replay");
    assert_eq!(first.body, replay.body);
    let mut conflict = record;
    conflict.request_commitment = [0x35; 32];
    conflict.body = vec![0x36; 73];
    assert!(matches!(
        journal.insert_or_exact(&conflict),
        Err(AnonymousMailboxSourceError::Conflict)
    ));
    assert_eq!(
        journal
            .load(&[0x31; 16])
            .expect("load")
            .expect("record")
            .body,
        vec![0x34; 73]
    );
}

#[test]
fn journal_phase_cas_keeps_terminal_outcomes_irreversible() {
    let target = target();
    let terminal = ticket_request(&target);
    let body = vec![0x3a; 64];
    let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
        node_id: target.public_key_bytes(),
        sequence: 8,
        descriptor_hash: [0x39; 32],
    };
    let record = SourceJournalRecord {
        route_id: [0x37; 16],
        request_commitment: source_request_commitment(
            &[0x37; 16],
            &target.public_key_bytes(),
            &descriptor_commitment,
            &terminal,
        ),
        target_node_id: target.public_key_bytes(),
        descriptor_commitment,
        body: body.clone(),
        phase: AnonymousMailboxSourcePhase::Prepared,
        retain_until: None,
        state: encode_state(&body, &terminal, None, None).expect("state"),
    };
    let journal = journal();
    journal.insert_or_exact(&record).expect("insert");
    journal
        .transition(
            &record,
            AnonymousMailboxSourcePhase::Prepared,
            AnonymousMailboxSourcePhase::Armed,
            record.state.clone(),
        )
        .expect("arm");
    journal
        .transition(
            &record,
            AnonymousMailboxSourcePhase::Armed,
            AnonymousMailboxSourcePhase::Completed,
            encode_state(&body, &terminal, None, Some(&terminal)).expect("completed state"),
        )
        .expect("complete");
    assert!(matches!(
        journal.transition(
            &record,
            AnonymousMailboxSourcePhase::Armed,
            AnonymousMailboxSourcePhase::Ambiguous,
            record.state.clone(),
        ),
        Err(AnonymousMailboxSourceError::Ambiguous)
    ));
    assert_eq!(
        journal
            .load(&record.route_id)
            .expect("load")
            .expect("record")
            .phase,
        AnonymousMailboxSourcePhase::Completed
    );
}

#[test]
fn terminal_retention_preserves_exact_replay_until_strict_deadline() {
    // [ANONYMOUS-MAILBOX-SOURCE-RETENTION 2026-09-13 by Codex] A
    // terminal row is the durable idempotency result until strictly after
    // its deadline. Equality remains retained to avoid a clock-boundary
    // retry racing cleanup.
    let journal = journal_with_limits(2, 4096, 10, 2);
    let record = journal_record(0x81, 0x82, AnonymousMailboxSourcePhase::Prepared);
    let completed = vec![0x83; 37];
    let retained = retain_terminal_record(
        &journal,
        &record,
        AnonymousMailboxSourcePhase::Completed,
        NOW,
        Some(&completed),
    );
    assert_eq!(retained.retain_until, Some(NOW + 10));

    let replay = journal.insert_or_exact(&record).expect("exact retry");
    assert_eq!(replay.phase, AnonymousMailboxSourcePhase::Completed);
    assert_eq!(
        decode_state(&replay.state)
            .expect("decode replay")
            .completed,
        Some(completed)
    );
    let mut conflict = journal_record(0x81, 0x84, AnonymousMailboxSourcePhase::Prepared);
    conflict.request_commitment = [0x85; 32];
    assert!(matches!(
        journal.insert_or_exact(&conflict),
        Err(AnonymousMailboxSourceError::Conflict)
    ));

    assert_eq!(
        journal
            .cleanup_terminal_records(NOW + 10)
            .expect("deadline cleanup"),
        AnonymousMailboxSourceCleanupReport::default()
    );
    assert!(journal
        .load(&record.route_id)
        .expect("load at deadline")
        .is_some());
    let reclaimed = journal
        .cleanup_terminal_records(NOW + 11)
        .expect("expired cleanup");
    assert_eq!(reclaimed.rows_removed, 1);
    assert!(reclaimed.bytes_removed > 0);
    assert!(journal
        .load(&record.route_id)
        .expect("load reclaimed")
        .is_none());
}

#[test]
fn cleanup_never_reclaims_unresolved_rows_and_releases_quota() {
    let journal = journal_with_limits(3, 8192, 10, 3);
    let prepared = journal_record(0x86, 0x87, AnonymousMailboxSourcePhase::Prepared);
    let armed = journal_record(0x88, 0x89, AnonymousMailboxSourcePhase::Prepared);
    let ambiguous = journal_record(0x8a, 0x8b, AnonymousMailboxSourcePhase::Prepared);
    journal.insert_or_exact(&prepared).expect("prepared");
    journal.insert_or_exact(&armed).expect("armed insert");
    journal
        .transition_at(
            &armed,
            AnonymousMailboxSourcePhase::Prepared,
            AnonymousMailboxSourcePhase::Armed,
            armed.state.clone(),
            NOW,
        )
        .expect("arm");
    journal
        .insert_or_exact(&ambiguous)
        .expect("ambiguous insert");
    journal
        .transition_at(
            &ambiguous,
            AnonymousMailboxSourcePhase::Prepared,
            AnonymousMailboxSourcePhase::Ambiguous,
            ambiguous.state.clone(),
            NOW,
        )
        .expect("mark ambiguous");
    assert_eq!(
        journal
            .cleanup_terminal_records(i64::MAX as u64)
            .expect("unresolved cleanup"),
        AnonymousMailboxSourceCleanupReport::default()
    );
    for route_id in [prepared.route_id, armed.route_id, ambiguous.route_id] {
        let loaded = journal.load(&route_id).expect("load unresolved");
        assert!(loaded.is_some());
        assert_eq!(loaded.expect("unresolved row").retain_until, None);
    }

    let quota = journal_with_limits(1, 4096, 10, 1);
    let expired = journal_record(0x8c, 0x8d, AnonymousMailboxSourcePhase::Prepared);
    retain_terminal_record(
        &quota,
        &expired,
        AnonymousMailboxSourcePhase::Rejected,
        NOW,
        None,
    );
    let replacement = journal_record(0x8e, 0x8f, AnonymousMailboxSourcePhase::Prepared);
    assert!(matches!(
        quota.insert_or_exact(&replacement),
        Err(AnonymousMailboxSourceError::Rejected)
    ));
    assert_eq!(
        quota
            .cleanup_terminal_records(NOW + 11)
            .expect("quota cleanup")
            .rows_removed,
        1
    );
    quota
        .insert_or_exact(&replacement)
        .expect("quota released after atomic cleanup");
}

#[test]
fn cleanup_is_bounded_and_rolls_back_row_and_meta_together() {
    let journal = journal_with_limits(3, 8192, 10, 1);
    let first = journal_record(0x90, 0x91, AnonymousMailboxSourcePhase::Prepared);
    let second = journal_record(0x92, 0x93, AnonymousMailboxSourcePhase::Prepared);
    retain_terminal_record(
        &journal,
        &first,
        AnonymousMailboxSourcePhase::Completed,
        NOW,
        Some(&[0x94; 8]),
    );
    retain_terminal_record(
        &journal,
        &second,
        AnonymousMailboxSourcePhase::Rejected,
        NOW,
        None,
    );
    let before = load_source_meta(&journal.connection.lock()).expect("meta before cleanup");
    journal
        .connection
        .lock()
        .execute_batch(
            "CREATE TRIGGER abort_source_cleanup BEFORE DELETE
                 ON anonymous_mailbox_source_journal
                 BEGIN SELECT RAISE(ABORT, 'bounded rollback fixture'); END;",
        )
        .expect("install rollback trigger");
    assert!(matches!(
        journal.cleanup_terminal_records(NOW + 11),
        Err(AnonymousMailboxSourceError::Unavailable)
    ));
    assert_eq!(
        load_source_meta(&journal.connection.lock()).expect("meta after rollback"),
        before
    );
    assert!(journal
        .load(&first.route_id)
        .expect("first after rollback")
        .is_some());
    assert!(journal
        .load(&second.route_id)
        .expect("second after rollback")
        .is_some());
    journal
        .connection
        .lock()
        .execute_batch("DROP TRIGGER abort_source_cleanup;")
        .expect("drop rollback trigger");

    assert_eq!(
        journal
            .cleanup_terminal_records(NOW + 11)
            .expect("first bounded batch")
            .rows_removed,
        1
    );
    assert_eq!(
        journal
            .cleanup_terminal_records(NOW + 11)
            .expect("batch plus one")
            .rows_removed,
        1
    );
    assert_eq!(
        journal
            .cleanup_terminal_records(NOW + 11)
            .expect("empty cleanup")
            .rows_removed,
        0
    );
}

#[test]
fn legacy_v1_migration_grants_full_window_and_preserves_unresolved_rows() {
    let sealer = journal();
    let mut completed = journal_record(0x95, 0x96, AnonymousMailboxSourcePhase::Completed);
    completed.state = encode_state(
        &completed.body,
        &decode_state(&completed.state)
            .expect("decode completed fixture")
            .terminal_frame,
        None,
        Some(&[0x97; 19]),
    )
    .expect("completed fixture state");
    let armed = journal_record(0x98, 0x99, AnonymousMailboxSourcePhase::Armed);
    let mut connection = Connection::open_in_memory().expect("legacy sqlite");
    create_legacy_source_schema(&connection);
    insert_legacy_source_record(&connection, &sealer, &completed);
    insert_legacy_source_record(&connection, &sealer, &armed);

    initialize_or_verify_source_schema(&mut connection, NOW, 10).expect("migrate v1");
    let version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("schema version");
    assert_eq!(version, SOURCE_JOURNAL_SCHEMA_VERSION);
    let terminal_deadline: Option<i64> = connection
        .query_row(
            "SELECT retain_until FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
            params![completed.route_id.as_slice()],
            |row| row.get(0),
        )
        .expect("terminal deadline");
    assert_eq!(
        terminal_deadline,
        Some(source_i64(NOW + 10).expect("deadline"))
    );
    let unresolved_deadline: Option<i64> = connection
        .query_row(
            "SELECT retain_until FROM anonymous_mailbox_source_journal WHERE route_id = ?1",
            params![armed.route_id.as_slice()],
            |row| row.get(0),
        )
        .expect("unresolved deadline");
    assert_eq!(unresolved_deadline, None);

    let migrated = SqliteAnonymousMailboxSourceJournal {
        connection: Mutex::new(connection),
        journal_key: [0x22; 32],
        max_entries: 4,
        max_bytes: 4096,
        terminal_retention_secs: 10,
        cleanup_batch_size: 2,
        #[cfg(unix)]
        _database_parent: None,
    };
    migrated.audit_startup().expect("migrated startup audit");
    let replay = migrated
        .insert_or_exact(&journal_record(
            0x95,
            0x96,
            AnonymousMailboxSourcePhase::Prepared,
        ))
        .expect("legacy exact retry");
    assert_eq!(replay.phase, AnonymousMailboxSourcePhase::Completed);
    assert_eq!(replay.retain_until, Some(NOW + 10));
    assert_eq!(
        decode_state(&replay.state)
            .expect("legacy replay state")
            .completed,
        Some(vec![0x97; 19])
    );
    assert_eq!(
        migrated
            .cleanup_terminal_records(NOW + 10)
            .expect("migration boundary")
            .rows_removed,
        0
    );
    assert_eq!(
        migrated
            .cleanup_terminal_records(NOW + 11)
            .expect("migration expiry")
            .rows_removed,
        1
    );
    assert_eq!(
        migrated
            .load(&armed.route_id)
            .expect("legacy unresolved load")
            .expect("legacy unresolved row")
            .phase,
        AnonymousMailboxSourcePhase::Armed
    );
}

#[test]
fn startup_audit_rejects_retention_and_meta_corruption() {
    let journal = journal_with_limits(2, 4096, 10, 2);
    let terminal = journal_record(0x9a, 0x9b, AnonymousMailboxSourcePhase::Prepared);
    retain_terminal_record(
        &journal,
        &terminal,
        AnonymousMailboxSourcePhase::Completed,
        NOW,
        Some(&[0x9c; 7]),
    );
    journal
        .connection
        .lock()
        .execute(
            "UPDATE anonymous_mailbox_source_journal SET retain_until = NULL
                 WHERE route_id = ?1",
            params![terminal.route_id.as_slice()],
        )
        .expect("corrupt terminal retention");
    assert!(matches!(
        journal.audit_startup(),
        Err(AnonymousMailboxSourceError::Corrupt)
    ));
    journal
        .connection
        .lock()
        .execute(
            "UPDATE anonymous_mailbox_source_journal SET retain_until = ?1
                 WHERE route_id = ?2",
            params![
                source_i64(NOW + 10).expect("deadline"),
                terminal.route_id.as_slice()
            ],
        )
        .expect("restore terminal retention");
    journal
        .connection
        .lock()
        .execute(
            "UPDATE anonymous_mailbox_source_meta SET total_entries = total_entries + 1
                 WHERE singleton = 1",
            [],
        )
        .expect("corrupt source meta");
    assert!(matches!(
        journal.audit_startup(),
        Err(AnonymousMailboxSourceError::Corrupt)
    ));
}

#[test]
fn completed_pull_over_budget_becomes_restart_safe_ambiguous() {
    let target = target();
    let reader = IdentityKeyPair::from_bytes(&[0x3b; 32]).expect("reader");
    let request = AnonymousMailboxPullOneV1::new([0x3c; 32], [0x3d; 16], Vec::new(), NOW, &reader)
        .expect("pull request");
    let terminal = encode_anonymous_mailbox_terminal_frame(
        &AnonymousMailboxTerminalFrameV1::PullOne(request.clone()),
    )
    .expect("pull frame");
    let pulled = AnonymousMailboxPullResultV1::new(
        [0x3e; 16],
        Vec::new(),
        vec![0x3f; MAX_ANONYMOUS_MAILBOX_SEALED_ITEM_BYTES],
    )
    .and_then(|value| value.encode())
    .expect("maximum pull result");
    let response = AnonymousMailboxTerminalResponseV1::signed(
        AnonymousMailboxOperationV1::PullOne,
        request.request_id,
        request.request_commitment().expect("request commitment"),
        AnonymousMailboxOutcomeV1::Accepted,
        pulled,
        NOW,
        &target,
    )
    .expect("pull response");
    let completed = encode_anonymous_mailbox_terminal_frame(
        &AnonymousMailboxTerminalFrameV1::PullOneResponse(response),
    )
    .expect("response frame");
    assert!(completed.len() <= MAX_ANONYMOUS_MAILBOX_TERMINAL_FRAME_BYTES);

    let body = vec![0x40; 64];
    let prepared_state = encode_state(&body, &terminal, None, None).expect("prepared state");
    let exact_prepared_budget =
        u64::try_from(body.len() + prepared_state.len() + JOURNAL_AEAD_TAG_BYTES).expect("budget");
    let journal = journal_with_max_bytes(exact_prepared_budget);
    let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
        node_id: target.public_key_bytes(),
        sequence: 9,
        descriptor_hash: [0x43; 32],
    };
    let record = SourceJournalRecord {
        route_id: [0x41; 16],
        request_commitment: source_request_commitment(
            &[0x41; 16],
            &target.public_key_bytes(),
            &descriptor_commitment,
            &terminal,
        ),
        target_node_id: target.public_key_bytes(),
        descriptor_commitment,
        body: body.clone(),
        phase: AnonymousMailboxSourcePhase::Prepared,
        retain_until: None,
        state: prepared_state.clone(),
    };
    journal.insert_or_exact(&record).expect("insert");
    journal
        .transition(
            &record,
            AnonymousMailboxSourcePhase::Prepared,
            AnonymousMailboxSourcePhase::Armed,
            prepared_state,
        )
        .expect("arm");
    let completed_state =
        encode_state(&body, &terminal, None, Some(&completed)).expect("completed state");
    assert!(matches!(
        journal.transition(
            &record,
            AnonymousMailboxSourcePhase::Armed,
            AnonymousMailboxSourcePhase::Completed,
            completed_state,
        ),
        Err(AnonymousMailboxSourceError::Ambiguous)
    ));

    let loaded = journal
        .load(&record.route_id)
        .expect("restart load")
        .expect("retained record");
    assert_eq!(loaded.phase, AnonymousMailboxSourcePhase::Ambiguous);
    let compact = decode_state(&loaded.state).expect("compact state");
    assert_eq!(compact.terminal_frame, terminal);
    assert!(compact.restart.is_none());
    assert!(compact.completed.is_none());
    let used: i64 = journal
        .connection
        .lock()
        .query_row(
            "SELECT SUM(length(body) + length(protected_state))
                 FROM anonymous_mailbox_source_journal",
            [],
            |row| row.get(0),
        )
        .expect("aggregate bytes");
    assert!(u64::try_from(used).expect("non-negative") <= exact_prepared_budget);
}

#[test]
fn fixed_state_bounds_reject_oversize_before_state_materialization() {
    let journal = journal_with_max_bytes(
        u64::try_from(MAX_JOURNAL_PROTECTED_STATE_BYTES * 2).expect("large config"),
    );
    assert!(matches!(
        journal.seal_state(
            &[0x44; 16],
            &[0x45; 32],
            &[0x46; 32],
            &vec![0; MAX_JOURNAL_CLEAR_STATE_BYTES + 1],
        ),
        Err(AnonymousMailboxSourceError::Rejected)
    ));

    let descriptor = bincode::serialize(&DirectoryDescriptorCommitmentV1 {
        node_id: [0x46; 32],
        sequence: 10,
        descriptor_hash: [0x47; 32],
    })
    .expect("descriptor");
    journal
        .connection
        .lock()
        .execute(
            "INSERT INTO anonymous_mailbox_source_journal
                   (route_id, request_commitment, target_node_id, descriptor_commitment, body,
                    phase, state_nonce, protected_state)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                [0x44_u8; 16].as_slice(),
                [0x45_u8; 32].as_slice(),
                [0x46_u8; 32].as_slice(),
                descriptor,
                [0x48_u8].as_slice(),
                AnonymousMailboxSourcePhase::Prepared.code(),
                [0x49_u8; JOURNAL_NONCE_BYTES].as_slice(),
                vec![0x4a_u8; MAX_JOURNAL_PROTECTED_STATE_BYTES + 1],
            ],
        )
        .expect("inject oversized protected state");
    assert!(matches!(
        journal.load(&[0x44; 16]),
        Err(AnonymousMailboxSourceError::Corrupt)
    ));
}

#[test]
fn source_journal_private_open_restarts_only_from_one_owner_private_inode() {
    let directory = tempfile::tempdir().expect("tempdir");
    let path = std::fs::canonicalize(directory.path())
        .expect("canonical private directory")
        .join("source.db");
    let config = AnonymousMailboxSourceConfig {
        enabled: true,
        db_path: path.to_string_lossy().into_owned(),
        ..AnonymousMailboxSourceConfig::default()
    };
    let target = target();
    let terminal = ticket_request(&target);
    let body = vec![0x67; 64];
    let descriptor_commitment = DirectoryDescriptorCommitmentV1 {
        node_id: target.public_key_bytes(),
        sequence: 12,
        descriptor_hash: [0x68; 32],
    };
    let record = SourceJournalRecord {
        route_id: [0x69; 16],
        request_commitment: source_request_commitment(
            &[0x69; 16],
            &target.public_key_bytes(),
            &descriptor_commitment,
            &terminal,
        ),
        target_node_id: target.public_key_bytes(),
        descriptor_commitment,
        body: body.clone(),
        phase: AnonymousMailboxSourcePhase::Prepared,
        retain_until: None,
        state: encode_state(&body, &terminal, None, None).expect("state"),
    };
    let journal =
        SqliteAnonymousMailboxSourceJournal::open(config.clone(), [0x67; 32]).expect("first open");
    journal.insert_or_exact(&record).expect("insert");
    journal
        .transition(
            &record,
            AnonymousMailboxSourcePhase::Prepared,
            AnonymousMailboxSourcePhase::Armed,
            record.state.clone(),
        )
        .expect("arm");
    drop(journal);
    let reopened =
        SqliteAnonymousMailboxSourceJournal::open(config, [0x67; 32]).expect("restart open");
    let resumed = reopened
        .load(&record.route_id)
        .expect("restart load")
        .expect("record");
    assert_eq!(resumed.phase, AnonymousMailboxSourcePhase::Armed);
    assert_eq!(resumed.body, body);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            std::fs::metadata(path).expect("metadata").mode() & 0o777,
            0o600
        );
    }
}

#[cfg(unix)]
#[test]
fn source_journal_rejects_symlink_and_hardlink_before_sqlite_mutation() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("tempdir");
    let private_directory =
        std::fs::canonicalize(directory.path()).expect("canonical private directory");
    let target = private_directory.join("target.db");
    std::fs::File::create(&target).expect("target");
    let symlink_path = private_directory.join("symlink.db");
    symlink(&target, &symlink_path).expect("symlink");
    let symlink_config = AnonymousMailboxSourceConfig {
        enabled: true,
        db_path: symlink_path.to_string_lossy().into_owned(),
        ..AnonymousMailboxSourceConfig::default()
    };
    assert!(matches!(
        SqliteAnonymousMailboxSourceJournal::open(symlink_config, [0x68; 32]),
        Err(AnonymousMailboxSourceError::Rejected | AnonymousMailboxSourceError::Unavailable)
    ));

    let hardlink_path = private_directory.join("hardlink.db");
    std::fs::hard_link(&target, &hardlink_path).expect("hardlink");
    let hardlink_config = AnonymousMailboxSourceConfig {
        enabled: true,
        db_path: hardlink_path.to_string_lossy().into_owned(),
        ..AnonymousMailboxSourceConfig::default()
    };
    assert!(matches!(
        SqliteAnonymousMailboxSourceJournal::open(hardlink_config, [0x69; 32]),
        Err(AnonymousMailboxSourceError::Rejected)
    ));
}
