// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn directory_mirror_carrier_capability_is_rollout_gated_and_runtime_honest() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.discovery.public_discovery = true;
    config.discovery.directory_chain_path = Some("/var/lib/aeronyx/directory-chain.db".to_string());
    config.discovery.directory_full_node_mirror_enabled = true;

    // [MIRROR-CAPABILITY 2026-07-24 by Codex] A compatible binary must not
    // change its wire advertisement until the operator completes the
    // mixed-version decoder rollout.
    let identity = IdentityKeyPair::generate();
    let staged =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_000).unwrap();
    assert!(!staged
        .descriptor
        .capabilities
        .contains(&NodeCapability::DirectoryMirrorCarrier));

    config.discovery.advertise_directory_mirror_carrier = true;
    let advertised =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_001).unwrap();
    assert!(advertised
        .descriptor
        .capabilities
        .contains(&NodeCapability::DirectoryMirrorCarrier));
    assert!(advertised.verify_at(1_800_000_002).is_ok());

    config.discovery.public_discovery = false;
    let private =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_002).unwrap();
    assert!(!private
        .descriptor
        .capabilities
        .contains(&NodeCapability::DirectoryMirrorCarrier));
}

#[test]
fn blind_vault_replica_capability_is_rollout_gated_and_runtime_honest() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("https://node.example.com".to_string());
    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.memchain.chat_relay.enabled = true;
    config.blind_vault.enabled = true;
    config.blind_vault.public_api_enabled = true;
    let issuer = IdentityKeyPair::from_bytes(&[31; 32]).expect("issuer key");
    config.blind_vault.admission_issuer_public_keys = vec![hex::encode(issuer.public_key_bytes())];

    let capabilities = Server::discovery_capabilities_for_runtime(&config, true);
    assert!(!capabilities.contains(&NodeCapability::BlindVaultReplica));

    config.blind_vault.advertise_replica = true;
    let advertised = Server::discovery_capabilities_for_runtime(&config, true);
    assert!(advertised.contains(&NodeCapability::BlindVaultReplica));
    assert!(advertised.contains(&NodeCapability::ChatRelay));

    let relay_unavailable = Server::discovery_capabilities_for_runtime(&config, false);
    assert!(!relay_unavailable.contains(&NodeCapability::BlindVaultReplica));

    config.discovery.public_api_listen_addr = None;
    let private = Server::discovery_capabilities_for_runtime(&config, true);
    assert!(!private.contains(&NodeCapability::BlindVaultReplica));
}

#[tokio::test]
async fn bounded_peer_response_accepts_small_json_and_rejects_unsafe_bodies() {
    let app = Router::new()
        .route(
            "/small",
            post(|| async { Json(serde_json::json!({ "accepted": true })) }),
        )
        .route("/oversized", post(|| async { "x".repeat(257) }))
        .route("/malformed", post(|| async { "{not-json" }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let client = reqwest::Client::new();

    let small = client
        .post(format!("{endpoint}/small"))
        .send()
        .await
        .unwrap();
    let decoded = decode_bounded_json_response::<serde_json::Value>(small, 256)
        .await
        .unwrap();
    assert_eq!(decoded["accepted"], true);

    let oversized = client
        .post(format!("{endpoint}/oversized"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        read_bounded_http_response(oversized, 256)
            .await
            .unwrap_err(),
        BoundedHttpResponseError::TooLarge
    );

    let malformed = client
        .post(format!("{endpoint}/malformed"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        decode_bounded_json_response::<serde_json::Value>(malformed, 256)
            .await
            .unwrap_err(),
        BoundedHttpResponseError::JsonDecode
    );
    mock_peer.abort();
}

#[tokio::test]
async fn bounded_recovery_file_accepts_exact_limit_and_rejects_oversize() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-bounded-read-{unique}.json"));

    tokio::fs::write(&path, [0x5a; 32]).await.unwrap();
    assert_eq!(
        Server::read_bounded_file(&path, 32).await.unwrap().len(),
        32
    );

    tokio::fs::write(&path, [0x5a; 33]).await.unwrap();
    let error = Server::read_bounded_file(&path, 32).await.unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(Server::bounded_file_error_reason(&error), "file_too_large");

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn proof_timeout_preserves_mandatory_legacy_budget() {
    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let legacy_calls_for_handler = Arc::clone(&legacy_calls);
    let app = Router::new()
        .route(
            "/api/discovery/summary",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(300)).await;
                Json(serde_json::json!({
                    "protocol_features": {
                        "directory_descriptor_proof_gossip_v1": true
                    }
                }))
            }),
        )
        .route(
            "/api/discovery/gossip",
            post(move |Json(_message): Json<NodeDiscoveryMessage>| {
                let legacy_calls = Arc::clone(&legacy_calls_for_handler);
                async move {
                    legacy_calls.fetch_add(1, AtomicOrdering::SeqCst);
                    Json(GossipResponse {
                        applied: PeerStoreImportReport::empty(),
                        response: None,
                    })
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "http://{}/api/discovery/gossip",
        listener.local_addr().unwrap()
    );
    let mock_peer = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let now = unix_now_secs();
    let client = reqwest::Client::new();
    let peer_store = PeerStore::new();
    let announcements = [directory_gossip_announcement(now)];
    let report = Server::gossip_with_peer(
        gossip_execution(
            &client,
            &peer_store,
            &announcements,
            now,
            Duration::from_millis(240),
        ),
        &url,
        signed_chat_relay_peer_descriptor("http://127.0.0.1:1".to_string(), now, now + 300),
    )
    .await;

    assert_eq!(legacy_calls.load(AtomicOrdering::SeqCst), 2);
    assert_eq!(report.legacy_error, None);
    assert_eq!(
        report.directory_proof,
        DirectoryProofGossipOutcome::negotiation_failed(
            DirectoryProofGossipResult::TransportFailed
        )
    );
    mock_peer.abort();
}

#[tokio::test]
async fn endpoint_evidence_disabled_has_zero_filesystem_side_effect() {
    std::fs::create_dir_all("target/test-temp").expect("external-disk test temp root");
    let directory = tempfile::TempDir::new_in("target/test-temp").expect("private directory");
    let absent = directory.path().join("disabled/evidence.sqlite3");
    let mut discovery = DiscoveryConfig::default();
    discovery.permissionless_endpoint_evidence_db_path = absent.to_string_lossy().into_owned();

    let opened = open_endpoint_evidence_store(&discovery, &IdentityKeyPair::generate())
        .await
        .expect("disabled evidence");
    assert!(opened.is_none());
    assert!(!absent.exists());
    assert!(!absent.parent().expect("parent").exists());
}

#[tokio::test]
async fn endpoint_attestation_inbox_disabled_has_zero_filesystem_side_effect() {
    std::fs::create_dir_all("target/test-temp").expect("external-disk test temp root");
    let directory = tempfile::TempDir::new_in("target/test-temp").expect("private directory");
    let absent = directory.path().join("disabled/attestations.sqlite3");
    let mut discovery = DiscoveryConfig::default();
    discovery.permissionless_endpoint_attestation_inbox_db_path =
        absent.to_string_lossy().into_owned();

    let opened = open_endpoint_attestation_inbox(&discovery)
        .await
        .expect("disabled inbox");
    assert!(opened.is_none());
    assert!(!absent.exists());
    assert!(!absent.parent().expect("parent").exists());
}
