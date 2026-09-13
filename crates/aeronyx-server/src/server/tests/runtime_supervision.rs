// ============================================
// File: crates/aeronyx-server/src/server/tests/runtime_supervision.rs
// ============================================
// [SERVER-DECOMPOSITION-PHASE0 2026-09-14 by Codex] Required-listener tests
// live in a focused child module without changing production visibility or
// runtime behavior.

use super::*;

#[tokio::test]
async fn required_api_listener_bind_fails_closed_on_address_conflict() {
    // [STARTUP-READINESS 2026-07-29 by Codex] A detached bind failure used
    // to leave the process active without an API. The startup barrier must
    // now reject the same conflict and release cleanly afterward.
    let occupied = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = occupied.local_addr().unwrap();
    let error = Server::bind_required_api_listener("test_api", address)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("test_api"));
    assert!(error.to_string().contains(&address.to_string()));

    drop(occupied);
    let rebound = Server::bind_required_api_listener("test_api", address)
        .await
        .unwrap();
    assert_eq!(rebound.local_addr().unwrap(), address);
}

#[test]
fn vpn_client_api_listener_uses_configured_gateway_and_node_api_port() {
    // [FAIL-CLOSED-SHUTDOWN-SIGNALS 2026-08-12 by Codex] A node using a
    // non-default privacy subnet must expose the client API on that
    // configured gateway while retaining the validated node API port.
    let gateway = Ipv4Addr::new(10, 77, 0, 1);
    let node_api = std::net::SocketAddr::from(([127, 0, 0, 1], 9443));

    assert_eq!(
        Server::vpn_client_api_listen_addr(gateway, node_api),
        std::net::SocketAddr::from(([10, 77, 0, 1], 9443)),
    );
}

#[tokio::test]
async fn required_api_listener_supervisor_reports_unexpected_failure() {
    // [RUNTIME-SUPERVISION 2026-07-29 by Codex] A required surface that
    // disappears without global shutdown must reach the main task.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let mut listeners = tokio::task::JoinSet::new();
    let address = "127.0.0.1:8421".parse().unwrap();
    listeners.spawn(async move {
        RequiredApiListenerExit {
            role: "test_api",
            address,
            result: Err(std::io::Error::other("forced listener failure")),
        }
    });

    let supervisor = tokio::spawn(Server::supervise_required_api_listeners(
        listeners,
        Arc::clone(&shutdown_requested),
        shutdown_tx.subscribe(),
        failure_tx,
    ));
    let failure = tokio::time::timeout(Duration::from_secs(1), failure_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        failure,
        CriticalRuntimeFailure {
            task: "test_api",
            reason: "required listener 127.0.0.1:8421 failed".to_string(),
        }
    );

    shutdown_requested.store(true, std::sync::atomic::Ordering::Release);
    shutdown_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn required_api_listener_supervisor_sanitizes_panic_failure() {
    // [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] A panic payload can be
    // request-derived. It must not cross the process-health channel even
    // though Tokio reports it through JoinError.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let mut listeners = tokio::task::JoinSet::new();
    listeners.spawn(async move {
        panic!("test-only-sensitive-listener-payload");
        #[allow(unreachable_code)]
        RequiredApiListenerExit {
            role: "test_api",
            address: "127.0.0.1:8421".parse().unwrap(),
            result: Ok(()),
        }
    });

    let supervisor = tokio::spawn(Server::supervise_required_api_listeners(
        listeners,
        Arc::clone(&shutdown_requested),
        shutdown_tx.subscribe(),
        failure_tx,
    ));
    let failure = tokio::time::timeout(Duration::from_secs(1), failure_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        failure,
        CriticalRuntimeFailure {
            task: "required_api_listener_group",
            reason: "required API listener task panicked".to_string(),
        }
    );
    assert!(!failure.reason.contains("sensitive"));

    shutdown_requested.store(true, std::sync::atomic::Ordering::Release);
    shutdown_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn required_api_listener_supervisor_accepts_global_shutdown() {
    // [RUNTIME-SUPERVISION 2026-07-29 by Codex] Expected listener exits
    // after the global stop marker must not cause a restart loop.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (shutdown_tx, _) = tokio::sync::broadcast::channel(2);
    let mut listener_shutdown_rx = shutdown_tx.subscribe();
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let mut listeners = tokio::task::JoinSet::new();
    listeners.spawn(async move {
        let _ = listener_shutdown_rx.recv().await;
        RequiredApiListenerExit {
            role: "test_api",
            address: "127.0.0.1:8421".parse().unwrap(),
            result: Ok(()),
        }
    });

    let supervisor = tokio::spawn(Server::supervise_required_api_listeners(
        listeners,
        Arc::clone(&shutdown_requested),
        shutdown_tx.subscribe(),
        failure_tx,
    ));
    shutdown_requested.store(true, std::sync::atomic::Ordering::Release);
    shutdown_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), failure_rx.recv())
            .await
            .unwrap()
            .is_none()
    );
}

// [SERVER-DECOMPOSITION-PHASE05 2026-09-14 by Codex] Keep the remaining
// runtime-supervision tests cohesive while production visibility stays fixed.
#[test]
fn data_plane_receive_failure_policy_backs_off_then_stops() {
    // [DATA-PLANE-FAILURE-POLICY 2026-07-30 by Codex] Transient errors
    // receive bounded exponential retry, while a persistent broken socket
    // or TUN descriptor reaches the required-task supervisor.
    assert_eq!(
        data_plane_receive_failure_action(1),
        DataPlaneReceiveFailureAction::RetryAfter(Duration::from_millis(25))
    );
    assert_eq!(
        data_plane_receive_failure_action(2),
        DataPlaneReceiveFailureAction::RetryAfter(Duration::from_millis(50))
    );
    assert_eq!(
        data_plane_receive_failure_action(6),
        DataPlaneReceiveFailureAction::RetryAfter(Duration::from_millis(800))
    );
    assert_eq!(
        data_plane_receive_failure_action(7),
        DataPlaneReceiveFailureAction::RetryAfter(Duration::from_millis(1_000))
    );
    assert_eq!(
        data_plane_receive_failure_action(DATA_PLANE_RECV_FAILURE_LIMIT),
        DataPlaneReceiveFailureAction::Stop
    );
    assert_eq!(
        data_plane_receive_failure_action(u32::MAX),
        DataPlaneReceiveFailureAction::Stop
    );
}

#[tokio::test]
async fn data_plane_receive_failure_policy_stops_at_limit_and_shutdown() {
    // [DATA-PLANE-FAILURE-POLICY 2026-07-30 by Codex] The async boundary
    // must stop deterministically without sleeping once recovery is
    // required or global shutdown has already started.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);
    let mut shutdown_rx = shutdown_tx.subscribe();
    let shutdown_requested = AtomicBool::new(false);
    let mut failures = DATA_PLANE_RECV_FAILURE_LIMIT - 1;
    assert!(
        !retry_required_data_plane_receive(
            "test-data-plane",
            &"forced receive failure",
            &mut failures,
            &shutdown_requested,
            &mut shutdown_rx,
        )
        .await
    );
    assert_eq!(failures, DATA_PLANE_RECV_FAILURE_LIMIT);

    shutdown_requested.store(true, std::sync::atomic::Ordering::Release);
    let mut shutdown_failures = 0;
    assert!(
        !retry_required_data_plane_receive(
            "test-data-plane",
            &"ignored during shutdown",
            &mut shutdown_failures,
            &shutdown_requested,
            &mut shutdown_rx,
        )
        .await
    );
    assert_eq!(shutdown_failures, 0);
}

#[test]
fn pre_ready_runtime_gate_distinguishes_empty_failure_and_disconnect() {
    // [PRE-READY-RUNTIME-GATE 2026-07-30 by Codex] Readiness may proceed
    // only while the supervisor channel is live and has no queued failure.
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    assert_eq!(take_pre_ready_runtime_failure(&mut failure_rx), None);

    let expected = CriticalRuntimeFailure {
        task: "management-heartbeat",
        reason: "required runtime task exited unexpectedly".to_string(),
    };
    failure_tx.try_send(expected.clone()).unwrap();
    assert_eq!(
        take_pre_ready_runtime_failure(&mut failure_rx),
        Some(expected)
    );

    drop(failure_tx);
    assert_eq!(
        take_pre_ready_runtime_failure(&mut failure_rx),
        Some(required_runtime_supervisor_channel_closed())
    );
}

#[tokio::test]
async fn runtime_task_registry_aborts_tasks_when_startup_unwinds() {
    // [STARTUP-TASK-REGISTRY 2026-07-30 by Codex] Dropping the registry
    // models any `?` after a process task has started.
    let task = tokio::spawn(std::future::pending::<()>());
    let abort_handle = task.abort_handle();
    let mut registry = RuntimeTaskRegistry::default();
    registry.push(("test-startup-task", task));

    drop(registry);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !abort_handle.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(abort_handle.is_finished());
}

#[tokio::test]
async fn runtime_task_registry_hands_tasks_to_bounded_shutdown() {
    // [STARTUP-TASK-REGISTRY 2026-07-30 by Codex] Successful startup
    // disarms Drop ownership and transfers the same handles into the
    // established concurrent shutdown policy.
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let _ = release_rx.await;
    });
    let abort_handle = task.abort_handle();
    let mut registry = RuntimeTaskRegistry::default();
    registry.push(("test-runtime-task", task));

    let tasks = registry.take_for_shutdown();
    assert!(!abort_handle.is_finished());
    release_tx.send(()).unwrap();
    let reports = Server::shutdown_runtime_tasks(tasks).await;
    assert_eq!(
        reports,
        vec![RuntimeTaskShutdownReport {
            name: "test-runtime-task",
            outcome: RuntimeTaskShutdownOutcome::Completed,
        }]
    );
    assert!(abort_handle.is_finished());
}

#[test]
fn directory_replica_runtime_is_optional_only_when_unconfigured() {
    // [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Preserve the
    // default-off compatibility boundary: no pins plus mirror disabled
    // needs neither a replica store nor a background task.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let task = server
        .spawn_directory_replica_sync_task(
            Arc::new(PeerStore::new()),
            None,
            Arc::new(DirectoryReplicaSyncRuntime::default()),
            test_peer_http_client(),
        )
        .unwrap();
    assert!(task.is_none());
}

#[test]
fn configured_directory_replica_runtime_requires_store() {
    // [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Programmatic
    // configuration must not bypass the validated file-config invariant
    // and advertise readiness without the configured mirror runtime.
    let mut config = ServerConfig::default();
    config.discovery.directory_full_node_mirror_enabled = true;
    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let error = server
        .spawn_directory_replica_sync_task(
            Arc::new(PeerStore::new()),
            None,
            Arc::new(DirectoryReplicaSyncRuntime::default()),
            test_peer_http_client(),
        )
        .err()
        .expect("configured mirror without a store must fail startup");
    assert_eq!(
            error.to_string(),
            "Server failed to start: Directory replica synchronization requires an initialized replica store"
        );
}

#[test]
fn configured_directory_replica_runtime_propagates_coordinator_error() {
    // [DIRECTORY-SYNC-RUNTIME-GATE 2026-07-30 by Codex] Do not reduce a
    // coordinator construction failure to a log line and `None`.
    let identity = IdentityKeyPair::generate();
    let directory = tempfile::tempdir().unwrap();
    let (store, _) = DirectoryReplicaStore::open(
        directory.path().join("directory.sqlite"),
        identity.public_key_bytes(),
        1_800_000_000,
    )
    .unwrap();
    let mut config = ServerConfig::default();
    config.discovery.directory_chain_sync_peer_node_ids = vec![hex::encode([0x51_u8; 32])];
    config.discovery.directory_chain_sync_interval_secs = 0;
    config.discovery.directory_observation_witness_min_verified = 1;
    let server = Server::new(config, identity, None);

    let error = server
        .spawn_directory_replica_sync_task(
            Arc::new(PeerStore::new()),
            Some(Arc::new(store)),
            Arc::new(DirectoryReplicaSyncRuntime::default()),
            test_peer_http_client(),
        )
        .err()
        .expect("invalid coordinator state must fail startup");
    assert_eq!(
            error.to_string(),
            "Server failed to start: Directory replica synchronization initialization failed: directory_sync_interval_invalid"
        );
}

#[tokio::test]
async fn required_runtime_task_supervisor_reports_unexpected_exit() {
    // [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] A configured
    // follower that returns without global shutdown is a process-health
    // failure, even when its inner future returned `Ok(())`.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let supervisor = Server::supervise_required_runtime_task(
        "memchain-block-sync",
        tokio::spawn(async {}),
        shutdown_requested,
        failure_tx,
    );

    let failure = tokio::time::timeout(Duration::from_secs(1), failure_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        failure,
        CriticalRuntimeFailure {
            task: "memchain-block-sync",
            reason: "required runtime task exited unexpectedly".to_string(),
        }
    );
    tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn required_runtime_task_supervisor_sanitizes_panic_failure() {
    // [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] JoinError details
    // are deliberately discarded so panic payloads cannot cross the
    // process-health privacy boundary.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let failed_task = tokio::spawn(async {
        panic!("test-only panic payload");
    });
    let supervisor = Server::supervise_required_runtime_task(
        "memchain-block-sync",
        failed_task,
        shutdown_requested,
        failure_tx,
    );

    let failure = tokio::time::timeout(Duration::from_secs(1), failure_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        failure,
        CriticalRuntimeFailure {
            task: "memchain-block-sync",
            reason: "required runtime task panicked".to_string(),
        }
    );
    assert!(!failure.reason.contains("test-only"));
    tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn required_runtime_task_supervisor_accepts_global_shutdown() {
    // [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] The global marker
    // is written before broadcasting shutdown. A cooperative inner return
    // after that marker must not create a systemd restart loop.
    let shutdown_requested = Arc::new(AtomicBool::new(false));
    let (release_tx, release_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let _ = release_rx.await;
    });
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let supervisor = Server::supervise_required_runtime_task(
        "memchain-block-sync",
        task,
        Arc::clone(&shutdown_requested),
        failure_tx,
    );

    shutdown_requested.store(true, std::sync::atomic::Ordering::Release);
    release_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap();
    assert!(
        tokio::time::timeout(Duration::from_secs(1), failure_rx.recv())
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn required_runtime_task_supervisor_abort_cancels_inner_task() {
    // [REQUIRED-TASK-OWNERSHIP 2026-07-30 by Codex] Tokio detaches a task
    // when its JoinHandle is merely dropped. Cancelling the supervisor must
    // instead propagate cancellation to the owned inner task.
    let inner_task = tokio::spawn(std::future::pending::<()>());
    let inner_abort_handle = inner_task.abort_handle();
    let (failure_tx, _failure_rx) = tokio::sync::mpsc::channel(1);
    let supervisor = Server::supervise_required_runtime_task(
        "memchain-block-sync",
        inner_task,
        Arc::new(AtomicBool::new(false)),
        failure_tx,
    );

    tokio::task::yield_now().await;
    supervisor.abort();
    assert!(tokio::time::timeout(Duration::from_secs(1), supervisor)
        .await
        .unwrap()
        .unwrap_err()
        .is_cancelled());
    tokio::time::timeout(Duration::from_secs(1), async {
        while !inner_abort_handle.is_finished() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(inner_abort_handle.is_finished());
}

#[test]
fn shutdown_signal_setup_failure_enters_required_runtime_failure_path() {
    // [FAIL-CLOSED-SHUTDOWN-SIGNALS 2026-08-12 by Codex] Signal setup
    // failures must be handled by the normal supervised shutdown path;
    // they must never panic past cache persistence and task ownership.
    assert_eq!(
        Server::shutdown_signal_failure("SIGTERM", "registration denied"),
        CriticalRuntimeFailure {
            task: "shutdown_signal_listener",
            reason: "SIGTERM shutdown signal listener failed: registration denied".to_string(),
        }
    );
}

#[tokio::test]
async fn wait_for_shutdown_accepts_programmatic_shutdown() {
    // [RUNTIME-SUPERVISION 2026-07-29 by Codex] An operator-initiated
    // in-process stop is graceful even if listener shutdown closes the
    // critical-failure channel at the same time.
    let server = Arc::new(Server::new(
        ServerConfig::default(),
        IdentityKeyPair::generate(),
        None,
    ));
    let (_failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    let waiting_server = Arc::clone(&server);
    let waiter =
        tokio::spawn(async move { waiting_server.wait_for_shutdown(&mut failure_rx).await });

    tokio::task::yield_now().await;
    server.shutdown();

    let outcome = tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .unwrap()
        .unwrap();
    assert!(outcome.is_none());
}

#[tokio::test]
async fn wait_for_shutdown_rejects_silent_required_task_group_loss() {
    // [REQUIRED-TASK-SUPERVISION 2026-07-30 by Codex] Losing every
    // required-task sender without an explicit failure must still trigger
    // process recovery; otherwise the node could remain half healthy.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let (failure_tx, mut failure_rx) = tokio::sync::mpsc::channel(1);
    drop(failure_tx);

    let failure = tokio::time::timeout(
        Duration::from_secs(1),
        server.wait_for_shutdown(&mut failure_rx),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        failure,
        CriticalRuntimeFailure {
            task: "required_runtime_task_group",
            reason: "critical runtime supervisor channel closed unexpectedly".to_string(),
        }
    );
}

#[tokio::test]
async fn runtime_task_shutdown_reports_cooperative_completion() {
    // [TASK-SHUTDOWN 2026-07-29 by Codex] A task that observes graceful
    // shutdown remains distinguishable from timeout cancellation.
    let task = tokio::spawn(async {});
    let report = Server::join_runtime_task(
        "cooperative-test",
        task,
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .await;

    assert_eq!(
        report,
        RuntimeTaskShutdownReport {
            name: "cooperative-test",
            outcome: RuntimeTaskShutdownOutcome::Completed,
        }
    );
    assert_eq!(
        Server::runtime_task_shutdown_grace("memchain-coordinator-lease"),
        Duration::from_secs(12)
    );
    assert_eq!(
        Server::runtime_task_shutdown_grace("directory-chain-persistence"),
        Duration::from_secs(20)
    );
    assert_eq!(
        Server::runtime_task_shutdown_grace("discovery-gossip"),
        Duration::from_secs(24)
    );
    assert_eq!(
        Server::runtime_task_shutdown_grace("anonymous-mailbox-cleanup"),
        Duration::from_secs(15)
    );
    assert_eq!(
        Server::runtime_task_shutdown_grace("udp"),
        Duration::from_secs(5)
    );
}

#[tokio::test]
async fn runtime_delay_observes_shutdown_before_deadline() {
    // [BACKGROUND-SHUTDOWN-COOPERATION 2026-08-12 by Codex] Initial
    // background-task delays must not force the shutdown supervisor to
    // abort an otherwise idle task.
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel(1);
    let waiter = tokio::spawn(async move {
        Server::runtime_delay_interrupted_by_shutdown(&mut shutdown_rx, Duration::from_secs(60))
            .await
    });

    tokio::task::yield_now().await;
    shutdown_tx.send(()).expect("deliver shutdown signal");
    let interrupted = tokio::time::timeout(Duration::from_secs(1), waiter)
        .await
        .expect("interruptible delay timed out")
        .expect("interruptible delay task failed");
    assert!(interrupted);
}

#[tokio::test]
async fn runtime_task_shutdown_aborts_after_grace_timeout() {
    // [TASK-SHUTDOWN 2026-07-29 by Codex] Reproduce the historical leak:
    // a non-terminating task must reach a cancelled JoinError instead of
    // continuing after its timeout future is dropped.
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let _ = started_tx.send(());
        std::future::pending::<()>().await;
    });
    assert!(
        started_rx.await.is_ok(),
        "stuck test task must start before its shutdown deadline"
    );

    let report = Server::join_runtime_task(
        "stuck-test",
        task,
        Duration::from_millis(10),
        Duration::from_secs(1),
    )
    .await;

    assert_eq!(
        report,
        RuntimeTaskShutdownReport {
            name: "stuck-test",
            outcome: RuntimeTaskShutdownOutcome::CancelledAfterTimeout,
        }
    );
}

#[tokio::test]
async fn runtime_task_shutdown_classifies_panic_without_retaining_payload() {
    // [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] Shutdown diagnostics
    // retain only a typed category, never JoinError's panic payload.
    let task = tokio::spawn(async {
        panic!("test-only-sensitive-shutdown-payload");
    });
    let report = Server::join_runtime_task(
        "panic-test",
        task,
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .await;

    assert_eq!(
        report,
        RuntimeTaskShutdownReport {
            name: "panic-test",
            outcome: RuntimeTaskShutdownOutcome::JoinFailed(RuntimeTaskJoinFailureKind::Panicked,),
        }
    );
    assert!(!format!("{report:?}").contains("sensitive"));
}

#[tokio::test]
async fn runtime_task_shutdown_distinguishes_external_cancellation() {
    // [JOIN-FAILURE-PRIVACY 2026-07-30 by Codex] Cancellation remains
    // operationally distinct from panic while carrying no JoinError text.
    let task = tokio::spawn(std::future::pending::<()>());
    task.abort();
    let report = Server::join_runtime_task(
        "cancelled-test",
        task,
        Duration::from_secs(1),
        Duration::from_secs(1),
    )
    .await;

    assert_eq!(
        report,
        RuntimeTaskShutdownReport {
            name: "cancelled-test",
            outcome: RuntimeTaskShutdownOutcome::JoinFailed(RuntimeTaskJoinFailureKind::Cancelled,),
        }
    );
}
