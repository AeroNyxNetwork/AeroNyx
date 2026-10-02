// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn client_hello_version_preflight_precedes_stateful_admission() {
    // [HANDSHAKE-PRE-ADMISSION-VERSION 2026-09-21 by Codex] Mirror the
    // production gate order with deterministic clocks and counters. An
    // unknown version must not consume either limiter or voucher state.
    let datagram = |version| {
        let mut bytes = vec![0u8; CLIENT_HELLO_SIZE];
        bytes[0] = MessageType::ClientHello.as_byte();
        bytes[1] = version;
        bytes
    };
    let limiter = HandshakeLimiter::new(1.0, 1.0, 1.0, 2.0);
    let verifier = VoucherVerifier::new();
    let now = Instant::now();

    let logs = capture_server_info_logs(|| {
        for version in [0, 3, u8::MAX] {
            assert!(!client_hello_wire_version_is_supported(&datagram(version)));
        }
    });
    assert!(
        logs.is_empty(),
        "unknown-version preflight must be silent: {logs}"
    );

    for version in [0, 3, u8::MAX] {
        let bytes = datagram(version);
        if client_hello_wire_version_is_supported(&bytes) {
            assert!(limiter.allow_at(Ipv4Addr::new(198, 51, 100, 1).into(), now));
            assert!(verifier.accept_client_hello_extension(Vec::new()).await);
        }
    }
    assert_eq!(verifier.metrics_snapshot().total, 0);

    for (version, host) in [(PROTOCOL_VERSION_V1, 11), (PROTOCOL_VERSION_V2, 12)] {
        let bytes = datagram(version);
        assert!(client_hello_wire_version_is_supported(&bytes));
        assert!(limiter.allow_at(Ipv4Addr::new(198, 51, 100, host).into(), now));
        assert!(verifier.accept_client_hello_extension(Vec::new()).await);
    }
    let voucher_metrics = verifier.metrics_snapshot();
    assert_eq!(voucher_metrics.total, 2);
    assert_eq!(voucher_metrics.missing, 2);
    assert!(
        !limiter.allow_at(Ipv4Addr::new(198, 51, 100, 13).into(), now),
        "two supported hellos must retain the historical global cap"
    );

    let per_ip = HandshakeLimiter::new(1.0, 1.0, 100.0, 10.0);
    let source = Ipv4Addr::new(203, 0, 113, 21).into();
    assert!(per_ip.allow_at(source, now));
    assert!(
        !per_ip.allow_at(source, now),
        "supported hellos must retain the historical per-IP cap"
    );
}

#[test]
fn v1_handshake_process_and_verify_logs_redact_assigned_ip() {
    // [V1-HANDSHAKE-IP-LOG-PRIVACY 2026-09-21 by Codex] Exercise both the
    // production server-side V1 derivation and client-side verifier with a
    // conspicuous address while preserving the actual wire value.
    let server = DefaultHandshakeCrypto::new(IdentityKeyPair::generate());
    let client = IdentityKeyPair::generate();
    let ephemeral = EphemeralKeyPair::generate();
    let hello = create_client_hello(&client, ephemeral.public_key_bytes(), PROTOCOL_VERSION_V1);
    let sentinel = [198, 51, 100, 247];
    let logs = capture_core_handshake_debug_logs(|| {
        server.verify_client_hello(&hello).expect("V1 ClientHello");
        let (server_hello, _) = server
            .process_handshake(&hello, sentinel, [0xA5; 16])
            .expect("V1 server handshake");
        assert_eq!(server_hello.assigned_ip, sentinel);
        verify_server_hello(&server_hello, &client.public_key_bytes()).expect("V1 ServerHello");
    });
    assert!(
        logs.contains("assigned_ip"),
        "coarse flow event missing: {logs}"
    );
    assert!(
        logs.matches("<redacted:4 bytes>").count() >= 3,
        "all server/client assigned-IP diagnostics must be redacted: {logs}"
    );
    for forbidden in ["198.51.100.247", "[198, 51, 100, 247]"] {
        assert!(
            !logs.contains(forbidden),
            "assigned IP leaked as {forbidden}: {logs}"
        );
    }
}

#[test]
fn handshake_rejection_logging_uses_closed_privacy_safe_classes() {
    // [V2-HANDSHAKE-LOG-PRIVACY 2026-09-20 by Codex] Exercise every
    // allowlisted bucket plus the fail-closed unknown bucket without a
    // socket fixture; the logger never receives the peer endpoint.
    let policy = ServerError::WalletDenied {
        reason: "SENSITIVE_POLICY_SENTINEL".to_string(),
    };
    let capacity = ServerError::IpAlreadyAssigned(Ipv4Addr::new(198, 51, 100, 247));
    let authentication = ServerError::Core(aeronyx_core::CoreError::SignatureVerification);
    let admission = ServerError::session_creation_failed("SENSITIVE_SESSION_SENTINEL");
    let unknown = ServerError::InvalidPacket {
        from_addr: "203.0.113.248:42424".to_string(),
        reason: "SENSITIVE_INTERNAL_SENTINEL".to_string(),
    };

    assert_eq!(
        handshake_rejection_class(&policy),
        HandshakeRejectionClass::Policy
    );
    assert_eq!(
        handshake_rejection_class(&capacity),
        HandshakeRejectionClass::Capacity
    );
    assert_eq!(
        handshake_rejection_class(&authentication),
        HandshakeRejectionClass::Authentication
    );
    assert_eq!(
        handshake_rejection_class(&admission),
        HandshakeRejectionClass::SessionAdmission
    );
    assert_eq!(
        handshake_rejection_class(&unknown),
        HandshakeRejectionClass::Internal
    );

    let logs = capture_server_info_logs(|| {
        for error in [&policy, &capacity, &authentication, &admission, &unknown] {
            log_handshake_rejection(error, 7);
        }
    });

    for reason in [
        "policy",
        "capacity",
        "authentication",
        "session_admission",
        "internal",
    ] {
        assert!(logs.contains(&format!("reason=\"{reason}\"")), "{logs}");
    }
    assert!(logs.contains("event=\"handshake_rejected\""), "{logs}");
    assert!(logs.contains("active_sessions=7"), "{logs}");
    for forbidden in [
        "198.51.100.247",
        "203.0.113.248:42424",
        "SENSITIVE_POLICY_SENTINEL",
        "SENSITIVE_SESSION_SENTINEL",
        "SENSITIVE_INTERNAL_SENTINEL",
        "client=",
        "endpoint=",
        "session_id=",
        "virtual_ip=",
        "error=",
    ] {
        assert!(!logs.contains(forbidden), "leaked {forbidden}: {logs}");
    }
}

#[test]
fn systemd_notifier_is_backward_compatible_without_notify_socket() {
    // [STARTUP-READINESS 2026-07-29 by Codex] CLI and container starts
    // without systemd must not acquire a new runtime requirement.
    let notifier = SystemdNotifier::from_socket(None);
    assert!(!notifier.status("initializing").unwrap());
    assert!(!notifier.ready("ready").unwrap());
    assert!(!notifier.stopping("stopping").unwrap());
}

#[cfg(target_os = "linux")]
#[test]
fn systemd_notifier_sends_sanitized_ready_datagram() {
    // [STARTUP-READINESS 2026-07-29 by Codex] Exercise the filesystem
    // namespace used by tests; production systemd commonly uses the
    // abstract namespace, which shares the same validated send path.
    let directory = tempfile::tempdir().unwrap();
    let socket_path = directory.path().join("notify.sock");
    let receiver = UnixDatagram::bind(&socket_path).unwrap();
    receiver
        .set_read_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    let notifier = SystemdNotifier::from_socket(Some(socket_path.as_os_str().to_os_string()));

    assert!(notifier.ready("ready\nwithout injection").unwrap());

    let mut buffer = [0_u8; 128];
    let received = receiver.recv(&mut buffer).unwrap();
    assert_eq!(
        &buffer[..received],
        b"READY=1\nSTATUS=ready without injection"
    );
}

#[tokio::test]
async fn peer_http_profiles_share_redirect_free_transport_policy() -> anyhow::Result<()> {
    let followed = Arc::new(AtomicUsize::new(0));
    let followed_handler = Arc::clone(&followed);
    let router = Router::new()
        .route(
            "/redirect",
            get(|| async {
                (
                    StatusCode::TEMPORARY_REDIRECT,
                    [(axum::http::header::LOCATION, "/target")],
                )
            }),
        )
        .route(
            "/target",
            get(move || {
                let followed = Arc::clone(&followed_handler);
                async move {
                    followed.fetch_add(1, AtomicOrdering::SeqCst);
                    StatusCode::OK
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let clients = PeerHttpClients::build(&ServerConfig::default())?;
    let url = format!("http://{address}/redirect");

    for (profile, client) in [
        ("control", clients.control.as_ref()),
        ("directory_sync", clients.directory_sync.as_ref()),
        ("directory_operator", clients.directory_operator.as_ref()),
        ("sync", clients.sync.as_ref()),
        ("gossip", clients.gossip.as_ref()),
    ] {
        let response = client.get(&url).send().await?;
        assert_eq!(
            response.status().as_u16(),
            StatusCode::TEMPORARY_REDIRECT.as_u16(),
            "{profile} profile followed an untrusted redirect"
        );
    }
    assert_eq!(followed.load(AtomicOrdering::SeqCst), 0);

    server.abort();
    let _ = server.await;
    Ok(())
}

#[test]
fn peer_http_profiles_preserve_role_specific_deadlines() {
    assert_eq!(
        DIRECTORY_SYNC_HTTP_PROFILE.connect_timeout_secs,
        Some(DIRECTORY_SYNC_CONNECT_TIMEOUT_SECS)
    );
    assert_eq!(
        DIRECTORY_SYNC_HTTP_PROFILE.request_timeout_secs,
        DIRECTORY_SYNC_HTTP_REQUEST_TIMEOUT_SECS
    );
    assert_eq!(DIRECTORY_OPERATOR_HTTP_PROFILE.request_timeout_secs, 12);
    assert_eq!(MEMCHAIN_SYNC_HTTP_PROFILE.request_timeout_secs, 15);
}
