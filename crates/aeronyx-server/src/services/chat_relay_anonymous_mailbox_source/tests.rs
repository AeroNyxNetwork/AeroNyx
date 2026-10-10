// ============================================
// File: crates/aeronyx-server/src/services/chat_relay_anonymous_mailbox_source/tests.rs
// ============================================
//! # Tests: source journal crash drills
//!
//! Process-level crash drills for the source journal, moved from the former
//! inline `tests` module. They stay in the module root on purpose: the
//! subprocess worker is selected by its libtest path
//! `services::chat_relay_anonymous_mailbox_source::tests::...`, which is part of
//! the drill and must not change.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/chat_relay_anonymous_mailbox_source.rs`; bodies unchanged.

use super::*;

use std::path::{Path, PathBuf};
use std::process::Output;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::anonymous_mailbox::{
    decode_anonymous_mailbox_terminal_frame, encode_anonymous_mailbox_terminal_frame,
    AnonymousMailboxOutcomeV1, AnonymousMailboxTerminalFrameV1,
    AnonymousMailboxTicketIssueResponseV1,
};
use aeronyx_core::protocol::discovery::{
    DirectoryDescriptorCommitmentV1, NodeCapability, NodeDescriptor, NodeProtocolFeature,
    SignedNodeDescriptor,
};
use rusqlite::{params, Connection};

use crate::config_chat_relay::AnonymousMailboxSourceConfig;

use super::crash_drill::{
    crash_after_source_journal_commit, SOURCE_JOURNAL_CRASH_BARRIER_ENV,
    SOURCE_JOURNAL_CRASH_EXIT_CODE, SOURCE_JOURNAL_CRASH_PHASE_ENV,
};
use super::state_codec::{decode_state, encode_state};
use super::test_support::{target, ticket_request, ExactOnlyResolver, NOW};

const SOURCE_CRASH_STAGE_ENV: &str = "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_STAGE";
const SOURCE_CRASH_DB_ENV: &str = "AERONYX_TEST_ANONYMOUS_MAILBOX_SOURCE_DB";
const SOURCE_CRASH_WORKER: &str = concat!(
    "services::chat_relay_anonymous_mailbox_source::tests::",
    "source_journal_crash_drill_subprocess_worker"
);

fn source_crash_descriptor(target: &IdentityKeyPair) -> SignedNodeDescriptor {
    let mut descriptor =
        NodeDescriptor::new(target.public_key_bytes(), 9, NOW - 1, NOW + 60, "test")
            .with_x25519_kem(target.x25519_public_key_bytes())
            .with_protocol_features([
                NodeProtocolFeature::AnonymousMailboxV1,
                NodeProtocolFeature::OnionReplyV1,
                NodeProtocolFeature::BlindRelaySuccessReceiptV1,
                NodeProtocolFeature::OnionSourceSealedTerminalProofV1,
            ]);
    descriptor.public_endpoint = Some("https://1.1.1.1:443".into());
    descriptor.capabilities = vec![NodeCapability::ChatRelay];
    SignedNodeDescriptor::sign(descriptor, target).expect("signed crash-drill descriptor")
}

fn source_crash_config(path: &Path) -> AnonymousMailboxSourceConfig {
    AnonymousMailboxSourceConfig {
        enabled: true,
        db_path: path.to_string_lossy().into_owned(),
        ..AnonymousMailboxSourceConfig::default()
    }
}

fn source_crash_coordinator(
    journal: Arc<SqliteAnonymousMailboxSourceJournal>,
) -> (AnonymousMailboxSourceCoordinator, Arc<ExactOnlyResolver>) {
    let target = target();
    let resolver = Arc::new(ExactOnlyResolver {
        descriptor: source_crash_descriptor(&target),
        calls: AtomicUsize::new(0),
    });
    let coordinator = AnonymousMailboxSourceCoordinator::new(
        Arc::new(IdentityKeyPair::from_bytes(&[0x72; 32]).expect("source identity")),
        resolver.clone(),
        journal,
    );
    (coordinator, resolver)
}

async fn run_source_crash_child(
    stage: &str,
    crash_phase: &str,
    db_path: &Path,
    barrier_path: &Path,
) -> Output {
    let executable = std::env::current_exe().expect("resolve source crash test binary");
    let mut child = crate::isolated_child_command(executable);
    child
        .arg(SOURCE_CRASH_WORKER)
        .arg("--exact")
        .arg("--ignored")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(SOURCE_CRASH_STAGE_ENV, stage)
        .env(SOURCE_CRASH_DB_ENV, db_path)
        .env(SOURCE_JOURNAL_CRASH_PHASE_ENV, crash_phase)
        .env(SOURCE_JOURNAL_CRASH_BARRIER_ENV, barrier_path)
        .kill_on_drop(true);
    tokio::time::timeout(Duration::from_secs(20), child.output())
        .await
        .expect("source crash child exceeded bounded deadline")
        .expect("start source crash child")
}

fn assert_source_crash_child(output: &Output, barrier_path: &Path) {
    assert_eq!(
        output.status.code(),
        Some(SOURCE_JOURNAL_CRASH_EXIT_CODE),
        "source crash worker missed the commit boundary; status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        std::fs::read(barrier_path).expect("read source crash barrier"),
        b"phase-commit-observed"
    );
}

#[test]
#[ignore = "spawned only by the bounded source-journal crash drill"]
fn source_journal_crash_drill_subprocess_worker() {
    let stage = std::env::var(SOURCE_CRASH_STAGE_ENV).expect("source crash drill stage");
    let db_path =
        PathBuf::from(std::env::var_os(SOURCE_CRASH_DB_ENV).expect("source crash drill database"));
    let config = source_crash_config(&db_path);
    let route_id = [0x7a; 16];
    let journal = Arc::new(
        SqliteAnonymousMailboxSourceJournal::open(config, [0x7b; 32])
            .expect("open source crash journal"),
    );

    match stage.as_str() {
        "prepared" => {
            let target = target();
            let descriptor = source_crash_descriptor(&target);
            let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                .expect("source crash descriptor commitment");
            let resolver = Arc::new(ExactOnlyResolver {
                descriptor,
                calls: AtomicUsize::new(0),
            });
            let coordinator = AnonymousMailboxSourceCoordinator::new(
                Arc::new(IdentityKeyPair::from_bytes(&[0x72; 32]).expect("source identity")),
                resolver,
                journal,
            );
            let _ = coordinator
                .prepare(
                    ExactAnonymousMailboxTargetPin::new(target.public_key_bytes(), commitment),
                    route_id,
                    ticket_request(&target),
                    NOW,
                )
                .expect("prepare should reach crash hook");
        }
        "armed" => {
            let (coordinator, _) = source_crash_coordinator(journal);
            let _ = coordinator
                .begin_dispatch(route_id, NOW)
                .expect("arming should reach crash hook");
        }
        "completed" => {
            let record = journal
                .load(&route_id)
                .expect("load armed source record")
                .expect("armed source record");
            assert_eq!(record.phase, AnonymousMailboxSourcePhase::Armed);
            let decoded = decode_state(&record.state).expect("decode armed source state");
            let request = match decode_anonymous_mailbox_terminal_frame(&decoded.terminal_frame)
                .expect("decode source request")
            {
                AnonymousMailboxTerminalFrameV1::TicketIssue(request) => request,
                _ => panic!("source crash fixture must retain ticket request"),
            };
            let target = target();
            let response = AnonymousMailboxTicketIssueResponseV1::signed(
                &request,
                AnonymousMailboxOutcomeV1::Rejected,
                None,
                NOW,
                &target,
            )
            .expect("sign source crash response");
            let completed = encode_anonymous_mailbox_terminal_frame(
                &AnonymousMailboxTerminalFrameV1::TicketIssueResponse(response),
            )
            .expect("encode source crash response");
            journal
                .transition(
                    &record,
                    AnonymousMailboxSourcePhase::Armed,
                    AnonymousMailboxSourcePhase::Completed,
                    encode_state(
                        &record.body,
                        &decoded.terminal_frame,
                        None,
                        Some(&completed),
                    )
                    .expect("encode completed source state"),
                )
                .expect("completion should reach crash hook");
        }
        "tampered" => {
            drop(journal);
            let connection = Connection::open(&db_path).expect("open source row for tamper");
            connection
                .execute_batch("PRAGMA synchronous=FULL; BEGIN IMMEDIATE;")
                .expect("begin durable source tamper");
            let updated = connection
                .execute(
                    "UPDATE anonymous_mailbox_source_journal
                         SET body = zeroblob(length(body)) WHERE route_id = ?1",
                    params![route_id.as_slice()],
                )
                .expect("tamper source body");
            assert_eq!(updated, 1);
            connection
                .execute_batch("COMMIT;")
                .expect("commit source tamper");
            crash_after_source_journal_commit(AnonymousMailboxSourcePhase::Prepared);
        }
        _ => panic!("unknown source crash drill stage"),
    }
    panic!("source crash drill missed its post-commit exit hook");
}

#[tokio::test]
async fn source_journal_phases_survive_abrupt_exit_and_completed_never_redispatches() {
    // [ANONYMOUS-MAILBOX-SOURCE-CRASH-DRILL 2026-09-05 by Codex] Each
    // child exits immediately after a FULL-durability phase commit and
    // before the journal method returns. Reopening in the parent therefore
    // cannot inherit any process-local SQLite or source-session state.
    let directory = tempfile::tempdir().expect("source crash directory");
    let private_directory =
        std::fs::canonicalize(directory.path()).expect("canonical source crash directory");
    let db_path = private_directory.join("source-crash.sqlite3");

    let prepared_barrier = private_directory.join("prepared.barrier");
    let prepared =
        run_source_crash_child("prepared", "prepared", &db_path, &prepared_barrier).await;
    assert_source_crash_child(&prepared, &prepared_barrier);

    let prepared_journal =
        SqliteAnonymousMailboxSourceJournal::open(source_crash_config(&db_path), [0x7b; 32])
            .expect("reopen prepared source journal");
    let prepared_record = prepared_journal
        .load(&[0x7a; 16])
        .expect("load prepared after crash")
        .expect("durable prepared record");
    assert_eq!(prepared_record.phase, AnonymousMailboxSourcePhase::Prepared);
    let prepared_state = decode_state(&prepared_record.state).expect("decode prepared state");
    let prepared_restart = prepared_state
        .restart
        .as_ref()
        .expect("prepared restart session")
        .encode_restart_state()
        .expect("encode prepared restart session")
        .as_bytes()
        .to_vec();
    let exact_body = prepared_record.body.clone();
    drop(prepared_journal);

    let armed_barrier = private_directory.join("armed.barrier");
    let armed = run_source_crash_child("armed", "armed", &db_path, &armed_barrier).await;
    assert_source_crash_child(&armed, &armed_barrier);

    let armed_journal =
        SqliteAnonymousMailboxSourceJournal::open(source_crash_config(&db_path), [0x7b; 32])
            .expect("reopen armed source journal");
    let armed_record = armed_journal
        .load(&[0x7a; 16])
        .expect("load armed after crash")
        .expect("durable armed record");
    assert_eq!(armed_record.phase, AnonymousMailboxSourcePhase::Armed);
    assert_eq!(armed_record.body, exact_body, "armed body must be exact");
    let armed_state = decode_state(&armed_record.state).expect("decode armed state");
    assert_eq!(
        armed_state
            .restart
            .as_ref()
            .expect("armed restart session")
            .encode_restart_state()
            .expect("encode armed restart session")
            .as_bytes(),
        prepared_restart,
        "arming must retain the exact one-shot source session"
    );
    drop(armed_journal);

    let completed_barrier = private_directory.join("completed.barrier");
    let completed =
        run_source_crash_child("completed", "completed", &db_path, &completed_barrier).await;
    assert_source_crash_child(&completed, &completed_barrier);

    let completed_journal = Arc::new(
        SqliteAnonymousMailboxSourceJournal::open(source_crash_config(&db_path), [0x7b; 32])
            .expect("reopen completed source journal"),
    );
    let completed_record = completed_journal
        .load(&[0x7a; 16])
        .expect("load completed after crash")
        .expect("durable completed record");
    assert_eq!(
        completed_record.phase,
        AnonymousMailboxSourcePhase::Completed
    );
    assert_eq!(completed_record.body, exact_body);
    let completed_state = decode_state(&completed_record.state).expect("decode completed state");
    assert!(completed_state.restart.is_none());
    let exact_completed = completed_state.completed.expect("completed response");
    let (coordinator, resolver) = source_crash_coordinator(completed_journal);
    assert!(matches!(
        coordinator.result([0x7a; 16]).expect("completed result"),
        AnonymousMailboxSourceResult::Completed(bytes) if bytes == exact_completed
    ));
    assert!(matches!(
        coordinator.begin_dispatch([0x7a; 16], NOW),
        Err(AnonymousMailboxSourceError::Rejected)
    ));
    assert_eq!(
        resolver.calls.load(Ordering::Relaxed),
        0,
        "completed restart must stop before target resolution or outbound release"
    );
}

#[tokio::test]
async fn source_journal_crash_reopen_rejects_tampered_projection_before_outbound() {
    let directory = tempfile::tempdir().expect("source tamper crash directory");
    let private_directory =
        std::fs::canonicalize(directory.path()).expect("canonical source tamper crash directory");
    let db_path = private_directory.join("source-tamper-crash.sqlite3");
    let prepared_barrier = private_directory.join("tamper-prepared.barrier");
    let prepared =
        run_source_crash_child("prepared", "prepared", &db_path, &prepared_barrier).await;
    assert_source_crash_child(&prepared, &prepared_barrier);

    let tampered_barrier = private_directory.join("tampered.barrier");
    let tampered =
        run_source_crash_child("tampered", "prepared", &db_path, &tampered_barrier).await;
    assert_source_crash_child(&tampered, &tampered_barrier);

    let journal = match SqliteAnonymousMailboxSourceJournal::open(
        source_crash_config(&db_path),
        [0x7b; 32],
    ) {
        Err(AnonymousMailboxSourceError::Corrupt) => return,
        Err(error) => panic!("tampered startup returned unexpected error: {error}"),
        Ok(journal) => Arc::new(journal),
    };
    let (coordinator, resolver) = source_crash_coordinator(journal);
    assert!(matches!(
        coordinator.begin_dispatch([0x7a; 16], NOW),
        Err(AnonymousMailboxSourceError::Corrupt)
    ));
    assert_eq!(
        resolver.calls.load(Ordering::Relaxed),
        0,
        "tampered durable projections must fail before outbound resolution"
    );
}
