// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

// [PHALA-NODE-ATTESTATION-API 2026-10-06 by Codex] Endpoint support is a
// signed additive feature only when the dstack socket is explicitly enabled.
#[test]
fn phala_node_attestation_feature_is_opt_in_and_signature_bound() {
    let identity = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.advertise_self = true;
    config.discovery.public_discovery = true;
    let disabled = Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_000)
        .unwrap();
    assert!(!disabled
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));

    config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    config.discovery.phala_attestation_socket_path =
        Some("/var/run/aeronyx/phala-agent.sock".into());
    // [PHALA-ENDPOINT-BOOTSTRAP 2026-10-06 by Codex] During first Phala
    // boot the hostname is not known yet, so the API works without advertising
    // a false endpoint. Supplying the actual origin activates the signed hint.
    let endpoint_pending =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_001)
            .unwrap();
    assert!(!endpoint_pending
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));
    // [PHALA-EFFECTIVE-DISCOVERY-ENDPOINT 2026-10-06 by Codex] The network
    // endpoint fallback must advertise the same contract that config validates.
    config.network.public_endpoint = Some("https://network-node.example.com".into());
    let network_endpoint =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_002)
            .unwrap();
    assert!(network_endpoint
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));
    assert_eq!(
        network_endpoint.descriptor.public_endpoint.as_deref(),
        Some("https://network-node.example.com")
    );
    config.discovery.public_endpoint = Some("https://node.example.com".into());
    let enabled = Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_003)
        .unwrap();
    assert!(enabled
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));

    // [PHALA-GATEWAY-DESCRIPTOR-REGRESSION 2026-10-06 by Codex] Bind the
    // deployment's assigned app-port hostname into the same signed capability
    // contract used by ordinary public HTTPS origins.
    config.discovery.public_endpoint = Some(
        "https://1e598a2f983dd80c413627e0b50d91905f3f48be-8422.dstack-prod5.phala.network"
            .into(),
    );
    let phala_gateway =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_003)
            .unwrap();
    assert!(phala_gateway
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));
    assert_eq!(
        phala_gateway.descriptor.public_endpoint.as_deref(),
        Some("https://1e598a2f983dd80c413627e0b50d91905f3f48be-8422.dstack-prod5.phala.network")
    );

    // [PHALA-ATTESTATION-PUBLIC-HTTPS 2026-10-06 by Codex] Descriptor
    // builders may be called without config validation; never sign an
    // attestation capability for an insecure or private transport origin.
    config.discovery.public_endpoint = Some("http://node.example.com".into());
    let insecure_endpoint =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_004)
            .unwrap();
    assert!(!insecure_endpoint
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));
    config.discovery.public_endpoint = Some("https://127.0.0.1:8422".into());
    let private_endpoint =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_005)
            .unwrap();
    assert!(!private_endpoint
        .descriptor
        .advertises_protocol_feature(NodeProtocolFeature::PhalaNodeAttestationV1));

    config.discovery.public_endpoint = Some("https://node.example.com".into());
    assert!(!enabled.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PhalaPrivateRecipientAttestationV1,
    ));

    // [PHALA-RECIPIENT-ATTESTATION-BINDING 2026-10-06 by Codex] Advertise
    // the additive binding only for one fixed recipient identity.
    let recipient = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
    config.reverse_onion.queue.enabled = true;
    config.reverse_onion.queue.recipient_node_ids =
        vec![hex::encode(recipient.public_key_bytes())];
    let recipient_bound =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_006)
            .unwrap();
    assert!(recipient_bound.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PhalaPrivateRecipientAttestationV1,
    ));
    // [PHALA-QUEUE-RECOVERY-GATE 2026-10-06 by Codex] Recovery mode keeps
    // node-level quotes but neither promises fresh recipient-bound authority
    // nor exposes that binding as a discovery capability.
    config.reverse_onion.queue.recovery_only = true;
    let recovery_only =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_007)
            .unwrap();
    assert!(recovery_only.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PhalaNodeAttestationV1,
    ));
    assert!(!recovery_only.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PhalaPrivateRecipientAttestationV1,
    ));
    config.reverse_onion.queue.recovery_only = false;
    config.reverse_onion.queue.recipient_node_ids.push(hex::encode(
        IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap().public_key_bytes(),
    ));
    let ambiguous_targets =
        Server::build_self_discovery_descriptor_for(&config, &identity, 1_800_000_008)
            .unwrap();
    assert!(!ambiguous_targets.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PhalaPrivateRecipientAttestationV1,
    ));
    assert!(enabled.verify_at(1_800_000_004).is_ok());
}

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

// [PRIVATE-ONION-PULL-ROLE 2026-10-05 by Codex] Authored, not executed.
#[test]
fn private_pull_feature_requires_live_terminal_without_advertising_public_replica() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.gossip_enabled = true;
    config.blind_vault.enabled = true;
    config.blind_vault.public_api_enabled = false;
    config.reverse_onion.recipient.enabled = true;
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] Keep the signed
    // recipient policy private, not merely endpoint-free.
    config.discovery.public_discovery = false;
    config.memchain.chat_relay.enabled = true;
    let identity = IdentityKeyPair::from_bytes(&[39; 32]).unwrap();

    let unavailable = Server::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
        &config, &identity, 1_800_000_000, true, false, false, false,
    ).unwrap();
    assert!(!unavailable.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionBlindVaultPullTerminalV1,
    ));

    let ready = Server::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
        &config, &identity, 1_800_000_001, true, false, false, true,
    ).unwrap();
    assert!(ready.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionBlindVaultPullTerminalV1,
    ));
    assert!(ready.descriptor.public_endpoint.is_none());
    assert!(!ready.descriptor.policy.public_discovery);
    assert!(!ready.descriptor.capabilities.contains(&NodeCapability::BlindVaultReplica));
    assert!(!ready.descriptor.capabilities.contains(&NodeCapability::ChatRelay));
    assert!(!config.blind_vault.public_api_enabled);

    let mut relay_config = ServerConfig::default();
    relay_config.discovery.enabled = true;
    relay_config.discovery.public_endpoint = Some("https://8.8.8.8:8422".to_owned());
    relay_config.discovery.public_api_listen_addr = Some("0.0.0.0:8422".parse().unwrap());
    relay_config.discovery.advertise_onion_middle = true;
    relay_config.memchain.chat_relay.enabled = true;
    // [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] Do not advertise a
    // middle hop when the durable relay service failed to start.
    let relay_without_chat_store =
        Server::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
            &relay_config,
            &IdentityKeyPair::from_bytes(&[42; 32]).unwrap(),
            1_800_000_000,
            false,
            false,
            false,
            false,
        )
        .unwrap();
    assert!(!relay_without_chat_store
        .descriptor
        .capabilities
        .contains(&NodeCapability::OnionMiddle));
    // [PHALA-ONION-RELAY-OPT-IN 2026-10-06 by Codex] A caller that bypasses
    // config-file validation must not advertise routeability when the durable
    // ChatRelay service itself is disabled, even if its readiness flag is true.
    let mut relay_without_chat_config = relay_config.clone();
    relay_without_chat_config.memchain.chat_relay.enabled = false;
    let relay_without_chat_config =
        Server::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
            &relay_without_chat_config,
            &IdentityKeyPair::from_bytes(&[43; 32]).unwrap(),
            1_800_000_001,
            true,
            false,
            false,
            false,
        )
        .unwrap();
    assert!(!relay_without_chat_config
        .descriptor
        .capabilities
        .contains(&NodeCapability::ChatRelay));
    assert!(!relay_without_chat_config
        .descriptor
        .capabilities
        .contains(&NodeCapability::OnionMiddle));
    let relay = Server::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
        &relay_config,
        &IdentityKeyPair::from_bytes(&[40; 32]).unwrap(),
        1_800_000_001,
        true,
        false,
        false,
        false,
    )
    .unwrap();
    assert!(relay.descriptor.capabilities.contains(&NodeCapability::ChatRelay));
    assert!(relay.descriptor.capabilities.contains(&NodeCapability::OnionMiddle));
    assert!(reverse_onion_current_pull_roles_valid(&relay, &ready));
    // [PHALA-PRIVATE-DISCOVERY-ISOLATION 2026-10-06 by Codex] Endpoint-free
    // alone must not pass the signed private-recipient role gate.
    // [PHALA-NODE-COMPILE-REPAIR 2026-10-08 by Codex] ready is already
    // the signed descriptor, so its single descriptor field is the body.
    let mut public_policy = ready.descriptor.clone();
    public_policy.policy.public_discovery = true;
    let public_policy = aeronyx_core::protocol::discovery::SignedNodeDescriptor::sign(
        public_policy, &identity,
    )
    .unwrap();
    assert!(!reverse_onion_current_pull_roles_valid(&relay, &public_policy));

    config.reverse_onion.recipient.recovery_only = true;
    let recovery = Server::build_self_discovery_descriptor_for_runtime_state_with_private_pull(
        &config, &identity, 1_800_000_002, true, false, false, true,
    ).unwrap();
    assert!(!recovery.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionBlindVaultPullTerminalV1,
    ));
}

// [REVERSE-ONION-AUTHORITY-GOSSIP 2026-10-05 by Codex]
#[test]
fn private_authorization_gossip_feature_tracks_live_queue_mode() {
    let mut config = ServerConfig::default();
    let identity = IdentityKeyPair::from_bytes(&[91; 32]).unwrap();
    let now = 1_800_000_000;

    let ordinary = Server::build_self_discovery_descriptor_for(&config, &identity, now).unwrap();
    assert!(!ordinary.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionAuthorizationGossipV1,
    ));

    config.reverse_onion.queue.enabled = true;
    let live = Server::build_self_discovery_descriptor_for(&config, &identity, now + 1).unwrap();
    assert!(live.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionAuthorizationGossipV1,
    ));

    config.reverse_onion.queue.recovery_only = true;
    let recovery = Server::build_self_discovery_descriptor_for(&config, &identity, now + 2).unwrap();
    assert!(!recovery.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionAuthorizationGossipV1,
    ));

    config.reverse_onion.queue.enabled = false;
    config.reverse_onion.source.enabled = true;
    let live_source = Server::build_self_discovery_descriptor_for(&config, &identity, now + 3).unwrap();
    assert!(live_source.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionAuthorizationGossipV1,
    ));
    config.reverse_onion.source.recovery_only = true;
    let source_recovery = Server::build_self_discovery_descriptor_for(&config, &identity, now + 4).unwrap();
    assert!(!source_recovery.descriptor.advertises_protocol_feature(
        NodeProtocolFeature::PrivateOnionAuthorizationGossipV1,
    ));
}

// [REVERSE-ONION-IDENTITY-SEED 2026-10-05 by Codex] Authored, not executed.
#[test]
fn expired_private_route_seed_pins_identity_without_reusing_stale_kem() {
    use aeronyx_core::protocol::discovery::SignedPrivateOnionRecipientAuthorizationV1;
    use aeronyx_core::protocol::onion::ONION_FORWARD_HOP_REQUIRED_CAPABILITIES;

    let now = 1_800_000_100;
    let relay_key = IdentityKeyPair::from_bytes(&[101; 32]).unwrap();
    let recipient_key = IdentityKeyPair::from_bytes(&[102; 32]).unwrap();
    let mut relay_body = NodeDescriptor::new(
        relay_key.public_key_bytes(), 1, now - 300, now - 100, "relay",
    )
    .with_x25519_kem(relay_key.x25519_public_key_bytes())
    .with_protocol_features(
        OnionRoutePurpose::BlindVaultPull.required_path_protocol_features().iter().copied(),
    );
    relay_body.public_endpoint = Some("https://8.8.8.8:8422".to_owned());
    relay_body.capabilities = ONION_FORWARD_HOP_REQUIRED_CAPABILITIES.to_vec();
    let relay = SignedNodeDescriptor::sign(relay_body, &relay_key).unwrap();
    let recipient_body = NodeDescriptor::new(
        recipient_key.public_key_bytes(), 1, now - 300, now - 100, "private-recipient",
    )
    .with_x25519_kem(recipient_key.x25519_public_key_bytes())
    .with_protocol_features(
        OnionRoutePurpose::BlindVaultPull.required_terminal_protocol_features().iter().copied(),
    );
    let recipient = SignedNodeDescriptor::sign(recipient_body, &recipient_key).unwrap();
    let authorization = SignedPrivateOnionRecipientAuthorizationV1::new_signed(
        &relay,
        &recipient,
        OnionRoutePurpose::BlindVaultPull.as_str(),
        now - 250,
        now - 150,
        &recipient_key,
    )
    .unwrap();

    assert!(!reverse_onion_recipient_seed_is_current(
        relay_key.public_key_bytes(), &relay, &recipient, &authorization, now,
    ).unwrap());
    assert!(reverse_onion_recipient_seed_is_current(
        recipient_key.public_key_bytes(), &relay, &recipient, &authorization, now,
    ).is_err());
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
