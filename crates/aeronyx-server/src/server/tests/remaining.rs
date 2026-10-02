// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn invalid_terminal_receipt_does_not_poison_first_hop_reputation() {
    let now = unix_now_secs();
    let source = IdentityKeyPair::generate();
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();
    let middle_node_id = middle_identity.public_key_bytes();
    let terminal_node_id = terminal_identity.public_key_bytes();

    let relay = Router::new().route(
        "/api/chat/peer/blind-relay",
        post(|| async {
            Json(PeerBlindRelayResponse {
                accepted: true,
                terminal: false,
                forwarded: true,
                ttl_remaining: 1,
                reason: Some("onion_forwarded".to_string()),
                delivery_receipt: None,
                success_receipt: None,
                failure_receipt: None,
                opaque_terminal_response_b64: None,
            })
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let middle_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let relay_server = tokio::spawn(async move {
        axum::serve(listener, relay).await.unwrap();
    });

    let mut middle = NodeDescriptor::new(
        middle_node_id,
        now,
        now,
        now + 300,
        "unattributed-receipt-middle",
    )
    .with_x25519_kem(middle_identity.x25519_public_key_bytes());
    middle.public_endpoint = Some(middle_endpoint);
    middle.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    let middle = SignedNodeDescriptor::sign(middle, &middle_identity).unwrap();

    let mut terminal = NodeDescriptor::new(
        terminal_node_id,
        now,
        now,
        now + 300,
        "unattributed-receipt-terminal",
    )
    .with_x25519_kem(terminal_identity.x25519_public_key_bytes());
    terminal.public_endpoint = Some("http://127.0.1.1:9".to_string());
    terminal.capabilities = vec![NodeCapability::ChatRelay];
    let terminal = SignedNodeDescriptor::sign(terminal, &terminal_identity).unwrap();

    let store = PeerStore::new();
    for descriptor in [middle, terminal] {
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&node_id, now);
    }

    let outcome = Server::relay_authenticated_chat_over_onion_paths(
        Some(&reqwest::Client::new()),
        None,
        &store,
        &source,
        &source.public_key_bytes(),
        &signed_test_chat_envelope(now),
        None,
    )
    .await;
    relay_server.abort();
    let _ = relay_server.await;

    assert_eq!(outcome.attempted_paths, 1);
    assert_eq!(outcome.verified_receipts, 0);
    let row = store
        .route_candidate_status(unix_now_secs())
        .onion_middle
        .into_iter()
        .find(|row| row.node_id_prefix == hex::encode(&middle_node_id[..4]))
        .expect("middle must remain visible in route diagnostics");
    assert_eq!(row.route_health, "healthy");
    assert_eq!(row.route_consecutive_failures, 0);
    assert_eq!(row.last_route_failure_reason, None);
}

#[tokio::test]
async fn verified_client_delivery_triggers_debounced_cache_flush() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.peer_cache_write_interval_secs = 3600;
    config.discovery.peer_cache_path = Some(
        std::env::temp_dir()
            .join(format!(
                "aeronyx-peer-cache-client-delivery-flush-{}.json",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
            .to_string_lossy()
            .to_string(),
    );
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config.clone(), IdentityKeyPair::generate(), None);
    let now = unix_now_secs();
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let peer_store = Arc::new(PeerStore::new());
    assert!(peer_store.upsert_verified(signed, now).unwrap());
    let path = config.discovery.peer_cache_path.unwrap();
    let peer_http_clients = PeerHttpClients::build(&server.config).unwrap();
    let handle = server
        .spawn_peer_store_persistence_task(
            Arc::clone(&peer_store),
            Arc::clone(&peer_http_clients.control),
        )
        .expect("peer cache persistence should be enabled");

    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if tokio::fs::metadata(&path).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("initial interval tick should persist peer cache");

    peer_store.record_verified_client_onion_delivery(now);
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let cache_contains_delivery = if let Ok(bytes) = tokio::fs::read(&path).await {
                PeerStoreCacheDocument::from_json_bytes(&bytes)
                    .ok()
                    .and_then(|document| document.verified_client_delivery_evidence)
                    .is_some_and(|evidence| evidence.verified_deliveries == 1)
            } else {
                false
            };
            let persisted_status = peer_store.status(now + 2).bootstrap;
            if cache_contains_delivery
                && persisted_status.last_client_delivery_cache_persisted == 1
                && persisted_status.last_client_delivery_cache_generation >= 2
                && persisted_status
                    .last_client_delivery_cache_rollback_protection
                    .as_deref()
                    == Some("anchored")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("client-delivery notification should flush signed aggregate evidence");

    assert_eq!(
        peer_store
            .status(now + 2)
            .bootstrap
            .last_client_delivery_cache_persisted,
        1
    );
    server
        .shutdown_tx
        .send(())
        .expect("shutdown receiver should be subscribed");
    handle.await.expect("peer cache task should stop cleanly");

    let _ = tokio::fs::remove_file(&path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path)).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path)).await;
}
