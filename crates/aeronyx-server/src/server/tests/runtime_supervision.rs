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
