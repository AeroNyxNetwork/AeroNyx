// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn anonymous_mailbox_custody_startup_cleanup_precedes_readiness() {
    // [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex] Seed one
    // valid expired ticket through the real repository, then prove the
    // async production initializer reclaims it before returning Some.
    let directory = tempfile::tempdir().expect("custody cleanup directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical custody directory");
    let db_path = private_parent.join("custody.sqlite");
    let server = test_anonymous_mailbox_custody_server(true, &db_path);
    let store = server
        .init_anonymous_mailbox_store_at(ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW)
        .await
        .expect("initial custody open")
        .expect("enabled custody");
    let request =
        test_anonymous_mailbox_ticket_issue(&server, ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW + 10);
    assert!(matches!(
        store
            .issue_ticket(&request, ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW)
            .expect("issue cleanup fixture"),
        AnonymousMailboxTicketIssueOutcome::Issued(_)
    ));
    drop(store);
    let before: i64 = rusqlite::Connection::open(&db_path)
        .expect("inspect custody before restart")
        .query_row(
            "SELECT COUNT(*) FROM anonymous_mailbox_issued_tickets",
            [],
            |row| row.get(0),
        )
        .expect("issued ticket count before cleanup");
    assert_eq!(before, 1);

    let reopened = server
        .init_anonymous_mailbox_store_at(ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW + 11)
        .await
        .expect("custody restart cleanup")
        .expect("custody ready after cleanup");
    drop(reopened);
    let after: i64 = rusqlite::Connection::open(&db_path)
        .expect("inspect custody after restart")
        .query_row(
            "SELECT COUNT(*) FROM anonymous_mailbox_issued_tickets",
            [],
            |row| row.get(0),
        )
        .expect("issued ticket count after cleanup");
    assert_eq!(after, 0);
}

#[tokio::test]
async fn anonymous_mailbox_cleanup_disabled_has_no_path_or_task() {
    let directory = tempfile::tempdir().expect("disabled cleanup directory");
    let missing_parent = directory.path().join("not-created");
    let db_path = missing_parent.join("custody.sqlite");
    let server = test_anonymous_mailbox_custody_server(false, &db_path);
    assert!(server
        .init_anonymous_mailbox_store_at(ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW)
        .await
        .expect("disabled custody")
        .is_none());
    assert!(server
        .spawn_anonymous_mailbox_cleanup_task(None, None)
        .is_none());
    assert!(!missing_parent.exists());
}

#[test]
fn anonymous_mailbox_cleanup_errors_are_retryable_or_fatal_by_contract() {
    use super::super::{
        AnonymousMailboxCleanupCycleOutcome, AnonymousMailboxCleanupLoopDirective,
        AnonymousMailboxSourceError,
    };

    assert_eq!(
        Server::observe_anonymous_mailbox_cleanup_cycle(AnonymousMailboxCleanupCycleOutcome {
            custody: Some(Err(AnonymousMailboxStoreError::Busy)),
            source: Some(Err(AnonymousMailboxSourceError::Unavailable)),
        }),
        AnonymousMailboxCleanupLoopDirective::Continue
    );
    assert_eq!(
        Server::observe_anonymous_mailbox_cleanup_cycle(AnonymousMailboxCleanupCycleOutcome {
            custody: Some(Err(AnonymousMailboxStoreError::UnsupportedSchema)),
            source: None,
        }),
        AnonymousMailboxCleanupLoopDirective::StopFatal
    );
    assert_eq!(
        Server::observe_anonymous_mailbox_cleanup_cycle(AnonymousMailboxCleanupCycleOutcome {
            custody: None,
            source: Some(Err(AnonymousMailboxSourceError::Corrupt)),
        }),
        AnonymousMailboxCleanupLoopDirective::StopFatal
    );
}

#[tokio::test(start_paused = true)]
async fn anonymous_mailbox_cleanup_loop_skips_overlap_and_finishes_shutdown() {
    use super::super::{
        run_anonymous_mailbox_cleanup_loop, AnonymousMailboxCleanupLoopDirective,
        AnonymousMailboxCleanupLoopExit,
    };
    use tokio::sync::Notify;

    let calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let maximum_active = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);
    let loop_calls = Arc::clone(&calls);
    let loop_active = Arc::clone(&active);
    let loop_maximum = Arc::clone(&maximum_active);
    let loop_started = Arc::clone(&started);
    let loop_release = Arc::clone(&release);
    let task = tokio::spawn(run_anonymous_mailbox_cleanup_loop(
        Duration::from_secs(10),
        shutdown_rx,
        move || {
            let calls = Arc::clone(&loop_calls);
            let active = Arc::clone(&loop_active);
            let maximum = Arc::clone(&loop_maximum);
            let started = Arc::clone(&loop_started);
            let release = Arc::clone(&loop_release);
            async move {
                let call = calls.fetch_add(1, AtomicOrdering::SeqCst);
                let concurrent = active.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                maximum.fetch_max(concurrent, AtomicOrdering::SeqCst);
                if call == 0 {
                    started.notify_one();
                    release.notified().await;
                }
                active.fetch_sub(1, AtomicOrdering::SeqCst);
                AnonymousMailboxCleanupLoopDirective::Continue
            }
        },
    ));
    started.notified().await;
    tokio::time::advance(Duration::from_secs(100)).await;
    tokio::task::yield_now().await;
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    assert_eq!(maximum_active.load(AtomicOrdering::SeqCst), 1);

    release.notify_one();
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(maximum_active.load(AtomicOrdering::SeqCst), 1);
    shutdown_tx.send(()).expect("request cleanup shutdown");
    assert_eq!(
        task.await.expect("cleanup loop task"),
        AnonymousMailboxCleanupLoopExit::Shutdown
    );
}

#[tokio::test(start_paused = true)]
async fn anonymous_mailbox_cleanup_loop_retries_then_stops_on_fatal() {
    use super::super::{
        run_anonymous_mailbox_cleanup_loop, AnonymousMailboxCleanupLoopDirective,
        AnonymousMailboxCleanupLoopExit,
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let loop_calls = Arc::clone(&calls);
    let (_shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);
    let task = tokio::spawn(run_anonymous_mailbox_cleanup_loop(
        Duration::from_secs(10),
        shutdown_rx,
        move || {
            let call = loop_calls.fetch_add(1, AtomicOrdering::SeqCst);
            async move {
                if call == 0 {
                    AnonymousMailboxCleanupLoopDirective::Continue
                } else {
                    AnonymousMailboxCleanupLoopDirective::StopFatal
                }
            }
        },
    ));
    tokio::task::yield_now().await;
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    assert!(
        !task.is_finished(),
        "retryable cycle must keep supervision alive"
    );
    tokio::time::advance(Duration::from_secs(10)).await;
    tokio::task::yield_now().await;
    assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(
        task.await.expect("fatal cleanup loop"),
        AnonymousMailboxCleanupLoopExit::Fatal
    );
}

#[tokio::test]
async fn anonymous_mailbox_source_disabled_startup_creates_no_private_path() {
    let directory = tempfile::tempdir().expect("test directory");
    let missing_parent = directory.path().join("not-created");
    let db_path = missing_parent.join("source.sqlite");
    let server = test_anonymous_mailbox_source_server(false, &db_path);

    let source = server
        .init_anonymous_mailbox_source_coordinator(Arc::new(PeerStore::new()))
        .await
        .expect("disabled source must remain optional");

    assert!(source.is_none());
    assert!(
        !missing_parent.exists(),
        "disabled startup must not create the configured private parent"
    );
}

#[tokio::test]
async fn anonymous_mailbox_source_enabled_startup_opens_and_reopens_private_journal() {
    let directory = tempfile::tempdir().expect("test directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical test private parent");
    let db_path = private_parent.join("source.sqlite");
    let server = test_anonymous_mailbox_source_server(true, &db_path);

    let first = server
        .init_anonymous_mailbox_source_coordinator(Arc::new(PeerStore::new()))
        .await
        .expect("enabled source must open a private journal")
        .expect("enabled source runtime");
    let cleanup = Server::run_anonymous_mailbox_cleanup_cycle(
        None,
        Some(Arc::clone(&first.journal)),
        ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
    );
    let report = cleanup
        .source
        .expect("source cleanup")
        .expect("source report");
    assert_eq!(report.rows_removed, 0);
    assert_eq!(report.bytes_removed, 0);
    assert!(db_path.is_file());
    drop(first);

    let reopened = server
        .init_anonymous_mailbox_source_coordinator(Arc::new(PeerStore::new()))
        .await
        .expect("enabled source must reopen its private journal");
    assert!(reopened.is_some());
}

#[tokio::test]
async fn anonymous_mailbox_source_startup_cleanup_precedes_runtime_composition() {
    // [ANONYMOUS-MAILBOX-CLEANUP-RUNTIME 2026-09-13 by Codex] Prepare two
    // canonical encrypted journal rows through the real coordinator, then
    // project crash-persisted terminal phase/deadline columns. Reopen must
    // reclaim both before returning a coordinator/runtime handle.
    let directory = tempfile::tempdir().expect("source cleanup directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical source directory");
    let db_path = private_parent.join("source-cleanup.sqlite");
    let server = test_anonymous_mailbox_source_server(true, &db_path);
    let target = IdentityKeyPair::from_bytes(&[0xd2; 32]).expect("source cleanup target");
    let descriptor =
        test_anonymous_mailbox_source_descriptor(&target, ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW);
    let descriptor_commitment =
        DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .expect("source cleanup descriptor commitment");
    let peer_store = Arc::new(PeerStore::new());
    assert!(peer_store
        .upsert_verified(descriptor, ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW)
        .expect("admit source cleanup target"));
    let runtime = server
        .init_anonymous_mailbox_source_coordinator_at(
            Arc::clone(&peer_store),
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
        )
        .await
        .expect("initial source open")
        .expect("source runtime");
    for (route_byte, request_byte) in [(0xd3_u8, 0xd4_u8), (0xd5_u8, 0xd6_u8)] {
        runtime
            .coordinator
            .prepare(
                ExactAnonymousMailboxTargetPin::new(
                    target.public_key_bytes(),
                    descriptor_commitment,
                ),
                [route_byte; 16],
                test_anonymous_mailbox_source_terminal_frame(&target, request_byte),
                ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
            )
            .expect("prepare source cleanup row");
    }
    drop(runtime);

    let connection = rusqlite::Connection::open(&db_path).expect("open source fixture");
    for (route_byte, phase) in [(0xd3_u8, 3_i64), (0xd5_u8, 5_i64)] {
        assert_eq!(
            connection
                .execute(
                    "UPDATE anonymous_mailbox_source_journal
                     SET phase = ?1, retain_until = ?2 WHERE route_id = ?3",
                    rusqlite::params![
                        phase,
                        ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW - 1,
                        [route_byte; 16].as_slice()
                    ],
                )
                .expect("project terminal source row"),
            1
        );
    }
    let before: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM anonymous_mailbox_source_journal
             WHERE phase IN (3, 5)",
            [],
            |row| row.get(0),
        )
        .expect("terminal source rows before restart");
    assert_eq!(before, 2);
    drop(connection);

    let reopened = server
        .init_anonymous_mailbox_source_coordinator_at(
            peer_store,
            ANONYMOUS_MAILBOX_CLEANUP_TEST_NOW,
        )
        .await
        .expect("source restart cleanup")
        .expect("source runtime after cleanup");
    let after: i64 = rusqlite::Connection::open(&db_path)
        .expect("inspect source cleanup result")
        .query_row(
            "SELECT COUNT(*) FROM anonymous_mailbox_source_journal",
            [],
            |row| row.get(0),
        )
        .expect("source rows after restart cleanup");
    assert_eq!(after, 0);
    drop(reopened);
}

#[tokio::test]
async fn anonymous_mailbox_source_startup_cleanup_failure_builds_no_runtime_or_task() {
    let directory = tempfile::tempdir().expect("source cleanup failure directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical source failure directory");
    let db_path = private_parent.join("source-cleanup-failure.sqlite");
    let server = test_anonymous_mailbox_source_server(true, &db_path);
    let result = server
        .init_anonymous_mailbox_source_coordinator_at(Arc::new(PeerStore::new()), u64::MAX)
        .await;

    assert!(result.is_err(), "cleanup rejection must stop startup");
    assert!(
        db_path.is_file(),
        "journal open must precede cleanup failure"
    );
    assert!(server
        .spawn_anonymous_mailbox_cleanup_task(None, None)
        .is_none());
}

#[tokio::test]
async fn anonymous_mailbox_source_startup_rejects_foreign_schema_without_readiness() {
    let directory = tempfile::tempdir().expect("test directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical test private parent");
    let db_path = private_parent.join("foreign.sqlite");
    let connection = rusqlite::Connection::open(&db_path).expect("foreign sqlite");
    connection
        .execute_batch("CREATE TABLE foreign_state (value INTEGER NOT NULL);")
        .expect("foreign schema");
    drop(connection);
    let server = test_anonymous_mailbox_source_server(true, &db_path);

    let error = match server
        .init_anonymous_mailbox_source_coordinator(Arc::new(PeerStore::new()))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("foreign schema must not produce source readiness"),
    };

    assert_eq!(
        error.to_string(),
        "Server failed to start: Anonymous mailbox source initialization failed"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn anonymous_mailbox_source_startup_rejects_unsafe_symlink_without_readiness() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().expect("test directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical test private parent");
    let target = private_parent.join("target.sqlite");
    std::fs::File::create(&target).expect("test-owned target");
    let unsafe_path = private_parent.join("source-link.sqlite");
    symlink(&target, &unsafe_path).expect("test-owned symlink");
    let server = test_anonymous_mailbox_source_server(true, &unsafe_path);

    let error = match server
        .init_anonymous_mailbox_source_coordinator(Arc::new(PeerStore::new()))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("symlink target must not produce source readiness"),
    };

    assert_eq!(
        error.to_string(),
        "Server failed to start: Anonymous mailbox source initialization failed"
    );
    assert_eq!(
        std::fs::metadata(target)
            .expect("test target metadata")
            .len(),
        0,
        "unsafe path rejection must not initialize the linked target"
    );
}

#[tokio::test]
async fn anonymous_mailbox_source_without_mpi_is_rejected_before_api_routes_start() {
    let directory = tempfile::tempdir().expect("test directory");
    let private_parent =
        std::fs::canonicalize(directory.path()).expect("canonical test private parent");
    let db_path = private_parent.join("source.sqlite");
    let server = test_anonymous_mailbox_source_server(true, &db_path);
    let peer_store = Arc::new(PeerStore::new());
    let source = server
        .init_anonymous_mailbox_source_coordinator(Arc::clone(&peer_store))
        .await
        .expect("source journal")
        .expect("enabled source");
    let (ip_pool, sessions, routing) = server.init_services().expect("test services");
    let node_policy = Arc::new(crate::services::NodePolicyRuntime::default());
    let encrypted_message_counter = Arc::new(AtomicU64::new(0));
    let traffic_tracker = Arc::new(crate::services::traffic_tracker::TrafficTracker::new());
    let deny_list = Arc::new(crate::services::deny_list::DenyList::new());
    let packet_handler = Arc::new(crate::handlers::PacketHandler::new(
        Arc::clone(&sessions),
        Arc::clone(&routing),
        Arc::clone(&traffic_tracker),
        Arc::clone(&encrypted_message_counter),
        Arc::clone(&node_policy),
    ));
    let handshake_service = Arc::new(crate::services::HandshakeService::new(
        server.identity.clone(),
        Arc::clone(&ip_pool),
        Arc::clone(&sessions),
        Arc::clone(&routing),
        Arc::clone(&deny_list),
        Arc::clone(&node_policy),
    ));
    // The existing API constructor takes ownership of the UDP transport
    // type even though this fail-closed branch returns before any API bind,
    // request handling, route construction, or transport traffic.
    let udp = Arc::new(
        UdpTransport::bind("127.0.0.1:0")
            .await
            .expect("test-only loopback UDP transport"),
    );
    let peer_http_clients =
        PeerHttpClients::build(&server.config).expect("local proxy-free HTTP client profiles");
    let (critical_failure_tx, _critical_failure_rx) = tokio::sync::mpsc::channel(1);

    // [NODE-ROLES 2026-10-09 by Claude] Same values, grouped by role.
    let plane = super::super::DataPlane {
        udp,
        #[cfg(target_os = "linux")]
        tun: None,
        ip_pool,
        sessions,
        routing,
        traffic_tracker,
        encrypted_message_counter,
        deny_list,
        node_policy,
        packet_handler,
        handshake_service,
        voucher_verifier: Arc::new(crate::voucher_verifier::VoucherVerifier::new()),
    };
    let directory = super::super::Directory {
        chain_store: None,
        replica_store: None,
        replica_sync_runtime: Arc::new(DirectoryReplicaSyncRuntime::default()),
    };
    let messaging = super::super::Messaging {
        chat_relay: None,
        anonymous_mailbox: None,
        anonymous_mailbox_source: Some(source.coordinator),
        anonymous_mailbox_cleanup_supervised: false,
        anonymous_mailbox_readiness: super::super::AnonymousMailboxReadinessProjection::default(),
    };

    let error = match server
        .start_combined_api(
            "127.0.0.1:0".parse().expect("loopback API address"),
            None,
            &plane,
            peer_store,
            &directory,
            &messaging,
            None,
            &peer_http_clients,
            None,
            critical_failure_tx,
        )
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("missing MPI must reject before API startup"),
    };

    assert_eq!(
        error.to_string(),
        "Server failed to start: Anonymous mailbox source requires authenticated VPN MPI runtime"
    );
}

#[test]
fn self_discovery_descriptor_publishes_work_policy_only_when_mailbox_runtime_is_ready() {
    // [ANONYMOUS-MAILBOX-WORK-POLICY 2026-09-07 by Codex] Configured work
    // bits become public only with the live target runtime; disabled and
    // failed-open callers retain the old descriptor representation.
    let mut config = ServerConfig::default();
    config
        .memchain
        .chat_relay
        .anonymous_mailbox
        .ticket_issue_work_bits = 17;
    let identity = IdentityKeyPair::from_bytes(&[0x71; 32]).expect("identity");

    let disabled = Server::build_self_discovery_descriptor_for_runtime(
        &config,
        &identity,
        1_800_000_000,
        false,
        false,
    )
    .expect("disabled descriptor");
    assert!(!disabled
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::AnonymousMailboxV1));
    assert_eq!(
        disabled.anonymous_mailbox_work_policy_at(1_800_000_001),
        Err(AnonymousMailboxWorkPolicyError::MissingPolicy)
    );

    let ready = Server::build_self_discovery_descriptor_for_runtime(
        &config,
        &identity,
        1_800_000_000,
        false,
        true,
    )
    .expect("ready descriptor");
    let pin = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&ready).expect("pin");
    let policy = ready
        .anonymous_mailbox_work_policy_for_pin_at(&pin, 1_800_000_001)
        .expect("work policy");
    assert_eq!(policy.target_node_id(), identity.public_key_bytes());
    assert_eq!(policy.descriptor_sequence(), ready.descriptor.sequence);
    assert_eq!(policy.issued_at(), ready.descriptor.issued_at);
    assert_eq!(policy.expires_at(), ready.descriptor.expires_at);
    assert_eq!(policy.work_bits(), 17);
}
