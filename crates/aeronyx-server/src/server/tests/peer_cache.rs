// Split from crates/aeronyx-server/src/server.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn startup_peer_store_keeps_pre_router_snapshot_candidate_only() {
    // [PERMISSIONLESS-DISCOVERY-STARTUP-GATE 2026-09-24 by Codex]
    // Simulate the response that an immediate gossip task can import
    // before any API router is constructed. Only an independently
    // established peer may remain live or enter the sampled URL set.
    let now = 1_800_000_000;
    let store = Server::new_discovery_peer_store();
    let signed_public_peer = |seed: u8, endpoint: &str| {
        let identity = IdentityKeyPair::from_bytes(&[seed; 32])
            .unwrap_or_else(|_| panic!("test identity must be valid"));
        let mut descriptor = NodeDescriptor::new(
            identity.public_key_bytes(),
            1,
            now - 10,
            now + 300,
            "startup-candidate-test",
        );
        descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.policy.public_discovery = true;
        descriptor.capabilities = vec![NodeCapability::ChatRelay];
        SignedNodeDescriptor::sign(descriptor, &identity)
            .unwrap_or_else(|_| panic!("test descriptor must sign"))
    };
    let established = signed_public_peer(0xA1, "https://9.9.9.9");
    assert!(store.upsert_verified(established.clone(), now).is_ok());
    let untrusted = signed_public_peer(0xA2, "https://8.8.8.8");
    let report = store.apply_discovery_message(
        &NodeDiscoveryMessage::SnapshotResponse {
            snapshot: NodeBootstrapSnapshot::new(now, vec![untrusted.clone()]),
        },
        now,
    );
    assert_eq!(report.inserted, 0);
    assert_eq!(report.candidates, 1);
    assert_eq!(store.snapshot(now).valid_peers, 1);
    assert!(store.get_valid(&untrusted.node_id(), now).is_none());
    assert!(store.get_valid(&established.node_id(), now).is_some());

    let seed_url = "https://1.1.1.1/api/discovery/gossip".to_string();
    let mut seen_urls = std::collections::HashSet::from([seed_url.clone()]);
    let mut gossip_urls = vec![seed_url.clone()];
    Server::append_sampled_discovered_peer_gossip_urls(
        &store,
        &DiscoveryGossipSampleRequest {
            now,
            round_nonce: [0xA3; 32],
            round_peer_limit: 3,
            self_node_id: &[0xA4; 32],
            self_gossip_url: None,
        },
        &mut seen_urls,
        &mut gossip_urls,
    );
    assert_eq!(gossip_urls[0], seed_url);
    assert_eq!(gossip_urls.len(), 2);
    assert!(gossip_urls[1].starts_with("https://9.9.9.9/"));
    assert!(!gossip_urls.iter().any(|url| url.contains("8.8.8.8")));
}

#[test]
fn peer_cache_recovery_anchor_golden_bytes_remain_stable() {
    // [SERVER-DECOMPOSITION-PHASE12A 2026-09-21 by Codex] Freeze the exact
    // legacy/current signing transcripts and pretty-JSON projection across
    // the pure-domain extraction.
    let template = PeerStoreVerifiedClientDeliveryAnchor {
        contract_version: VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT.to_string(),
        cache_generation: 7,
        cache_generated_at: 9,
        evidence: None,
        route_state_digest: None,
        two_hop_path_proof_digest: None,
        three_hop_path_proof_digest: None,
        signer_node_id: "22".repeat(32),
        signature_ed25519: "33".repeat(64),
    };
    let v1 = template.clone();
    let mut v2 = template.clone();
    v2.contract_version = VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT.to_string();
    v2.two_hop_path_proof_digest = Some("44".repeat(32));
    v2.three_hop_path_proof_digest = Some("55".repeat(32));
    let mut v3 = v2.clone();
    v3.contract_version = "peer_store_recovery_anchor.v3".to_string();
    v3.route_state_digest = Some("66".repeat(32));

    assert_eq!(
        hex::encode(v1.signing_bytes().unwrap()),
        "35000000000000006165726f6e79782d706565722d63616368652d76657269666965642d636c69656e742d64656c69766572792d616e63686f722d76312d00000000000000706565725f73746f72655f76657269666965645f636c69656e745f64656c69766572795f616e63686f722e76310700000000000000090000000000000000"
    );
    assert_eq!(
        hex::encode(v2.signing_bytes().unwrap()),
        "25000000000000006165726f6e79782d706565722d63616368652d7265636f766572792d616e63686f722d76321d00000000000000706565725f73746f72655f7265636f766572795f616e63686f722e763207000000000000000900000000000000000140000000000000003434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343401400000000000000035353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535"
    );
    assert_eq!(
        hex::encode(v3.signing_bytes().unwrap()),
        "25000000000000006165726f6e79782d706565722d63616368652d7265636f766572792d616e63686f722d76331d00000000000000706565725f73746f72655f7265636f766572795f616e63686f722e76330700000000000000090000000000000000014000000000000000363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636360140000000000000003434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343434343401400000000000000035353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535353535"
    );
    assert_eq!(
        String::from_utf8(v3.to_json_pretty().unwrap()).unwrap(),
        concat!(
            "{\n",
            "  \"contract_version\": \"peer_store_recovery_anchor.v3\",\n",
            "  \"cache_generation\": 7,\n",
            "  \"cache_generated_at\": 9,\n",
            "  \"evidence\": null,\n",
            "  \"route_state_digest\": \"6666666666666666666666666666666666666666666666666666666666666666\",\n",
            "  \"two_hop_path_proof_digest\": \"4444444444444444444444444444444444444444444444444444444444444444\",\n",
            "  \"three_hop_path_proof_digest\": \"5555555555555555555555555555555555555555555555555555555555555555\",\n",
            "  \"signer_node_id\": \"2222222222222222222222222222222222222222222222222222222222222222\",\n",
            "  \"signature_ed25519\": \"33333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333\"\n",
            "}"
        )
    );
}

#[tokio::test]
async fn peer_store_cache_persists_verified_snapshot_json() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now = unix_now_secs();
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let peer_store = Arc::new(PeerStore::new());
    assert!(peer_store.upsert_verified(signed, now).unwrap());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();

    Server::save_peer_store_cache_snapshot(&server.identity, &peer_store, &path_str, now)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
    let document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();

    assert_eq!(snapshot.peers.len(), 1);
    assert_eq!(snapshot.verified_count_at(now + 1), 1);
    assert_eq!(
        document.routeability_evidence_schema_version,
        ROUTEABILITY_CACHE_EVIDENCE_SCHEMA_VERSION
    );
    assert!(document.routeability_evidence.is_empty());
    assert_eq!(
        document.route_quarantine_schema_version,
        ROUTE_QUARANTINE_CACHE_SCHEMA_VERSION
    );
    assert!(document.route_quarantine_evidence.is_empty());
    assert_eq!(
        document.two_hop_path_proof_schema_version,
        TWO_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION
    );
    assert!(document.two_hop_path_proof_events.is_empty());
    assert_eq!(
        document.three_hop_path_proof_schema_version,
        THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION
    );
    assert!(document.three_hop_path_proof_events.is_empty());
    assert_eq!(
        document.route_domain_certificate_schema_version,
        ROUTE_DOMAIN_CERTIFICATE_CACHE_SCHEMA_VERSION
    );
    assert!(document.route_domain_certificates.is_empty());
    assert_eq!(
        document.verified_client_delivery_schema_version,
        VERIFIED_CLIENT_DELIVERY_CACHE_SCHEMA_VERSION
    );
    assert_eq!(document.verified_client_delivery_generation, 1);
    assert!(document.verified_client_delivery_evidence.is_none());
    assert!(document
        .verify_routeability_evidence_signature(&server.identity)
        .is_ok());
    assert!(document
        .verify_two_hop_path_proof_signature(&server.identity)
        .is_ok());
    assert!(document
        .verify_three_hop_path_proof_signature(&server.identity)
        .is_ok());
    assert!(document
        .verify_verified_client_delivery_signature(&server.identity)
        .is_ok());

    // [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] A rolling-upgrade
    // node must continue validating the exact v1 signing domain. v1 has no
    // quarantine section and therefore cannot claim restored quarantine.
    let mut legacy_document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    legacy_document.routeability_evidence_schema_version =
        ROUTEABILITY_CACHE_EVIDENCE_LEGACY_SCHEMA_VERSION;
    legacy_document.route_quarantine_schema_version = 0;
    legacy_document.route_quarantine_evidence.clear();
    legacy_document.routeability_evidence_signature_ed25519 = Some(hex::encode(
        server.identity.sign(
            &legacy_document
                .routeability_evidence_signing_bytes()
                .unwrap(),
        ),
    ));
    let legacy_bytes = serde_json::to_vec_pretty(&legacy_document).unwrap();
    let parsed_legacy = PeerStoreCacheDocument::from_json_bytes(&legacy_bytes).unwrap();
    assert!(parsed_legacy
        .verify_routeability_evidence_signature(&server.identity)
        .is_ok());

    let anchor_path = Server::peer_cache_client_delivery_anchor_path(&path_str);
    let anchor_bytes = tokio::fs::read(&anchor_path).await.unwrap();
    let anchor = PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(&anchor_bytes).unwrap();
    assert!(anchor.verify(&server.identity).is_ok());
    assert!(anchor.matches_document(&document));
    assert!(anchor.matches_route_state_section(&document));
    assert!(anchor.matches_two_hop_path_proof_section(&document));
    assert!(anchor.matches_three_hop_path_proof_section(&document));
    let witness_digest = anchor.witness_digest().unwrap();
    assert_eq!(witness_digest, anchor.witness_digest().unwrap());
    let mut next_anchor = anchor.clone();
    next_anchor.cache_generation += 1;
    next_anchor.signature_ed25519 =
        hex::encode(server.identity.sign(&next_anchor.signing_bytes().unwrap()));
    assert!(next_anchor.verify(&server.identity).is_ok());
    assert_ne!(witness_digest, next_anchor.witness_digest().unwrap());
    let mut legacy_anchor = anchor.clone();
    legacy_anchor.contract_version = VERIFIED_CLIENT_DELIVERY_ANCHOR_LEGACY_CONTRACT.to_string();
    legacy_anchor.route_state_digest = None;
    legacy_anchor.two_hop_path_proof_digest = None;
    legacy_anchor.three_hop_path_proof_digest = None;
    legacy_anchor.signature_ed25519 = hex::encode(
        server
            .identity
            .sign(&legacy_anchor.signing_bytes().unwrap()),
    );
    let parsed_legacy = PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(
        &legacy_anchor.to_json_pretty().unwrap(),
    )
    .unwrap();
    assert!(parsed_legacy.verify(&server.identity).is_ok());
    assert!(!parsed_legacy.matches_route_state_section(&document));
    assert!(!parsed_legacy.matches_two_hop_path_proof_section(&document));
    assert!(!parsed_legacy.matches_three_hop_path_proof_section(&document));

    // [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] v2 anchors keep
    // their exact proof-signing contract during rolling upgrades, but they
    // predate route-state commitment and therefore cannot authorize route
    // or quarantine recovery.
    let mut previous_anchor = anchor.clone();
    previous_anchor.contract_version =
        VERIFIED_CLIENT_DELIVERY_ANCHOR_PREVIOUS_CONTRACT.to_string();
    previous_anchor.route_state_digest = None;
    previous_anchor.signature_ed25519 = hex::encode(
        server
            .identity
            .sign(&previous_anchor.signing_bytes().unwrap()),
    );
    let parsed_previous = PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(
        &previous_anchor.to_json_pretty().unwrap(),
    )
    .unwrap();
    assert!(parsed_previous.verify(&server.identity).is_ok());
    assert!(!parsed_previous.matches_route_state_section(&document));
    assert!(parsed_previous.matches_two_hop_path_proof_section(&document));
    assert!(parsed_previous.matches_three_hop_path_proof_section(&document));
    assert_eq!(
        PeerStoreVerifiedClientDeliveryAnchorState::Verified(parsed_previous)
            .route_state_protection_for(&document),
        "legacy_unanchored"
    );
    let serialized_anchor = String::from_utf8(anchor_bytes).unwrap();
    for forbidden in [
        "route_id",
        "peer_id",
        "sender",
        "receiver",
        "message_id",
        "payload_commitment",
        "ciphertext",
        "public_endpoint",
    ] {
        assert!(!serialized_anchor.contains(forbidden));
    }

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(anchor_path).await;
}

#[tokio::test]
async fn peer_store_cache_keeps_active_route_quarantine_across_restart() {
    // [ROUTE-QUARANTINE-RECOVERY 2026-08-21 by Codex] Exercise the real
    // signed cache document and startup importer. A peer that was healthy
    // before three consecutive failures must remain excluded after restart.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_040_000;
    let peer_identity = IdentityKeyPair::generate();
    let descriptor = signed_chat_relay_peer_descriptor_for_identity(
        "https://quarantined-peer.example".to_string(),
        7,
        now + 4_000,
        &[],
        &peer_identity,
    );
    let node_id = descriptor.node_id();
    let original = Arc::new(PeerStore::new());
    assert!(original.upsert_verified(descriptor.clone(), now).unwrap());
    assert!(original.record_route_forward_success_for_descriptor(&descriptor, now + 1));
    for observed_at in [now + 2, now + 3, now + 4] {
        assert!(original.record_route_forward_failure_for_descriptor(
            &descriptor,
            observed_at,
            "request_failed",
        ));
    }
    assert!(original.is_route_quarantined_now(&node_id, now + 4));
    assert!(!original.is_routeable_now(&node_id, now + 4));

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("aeronyx-peer-cache-route-quarantine-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original, &path_str, now + 5)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    assert_eq!(document.routeability_evidence_schema_version, 2);
    assert!(document.routeability_evidence.is_empty());
    assert_eq!(document.route_quarantine_evidence.len(), 1);
    assert!(document
        .verify_routeability_evidence_signature(&server.identity)
        .is_ok());
    let quarantine_json = serde_json::to_string(&document.route_quarantine_evidence).unwrap();
    for forbidden in [
        "quarantined-peer.example",
        "request_failed",
        "last_failure_reason",
        "payload",
        "route_id",
    ] {
        assert!(!quarantine_json.contains(forbidden));
    }

    let mut tampered = document;
    tampered.route_quarantine_evidence.clear();
    assert!(tampered
        .verify_routeability_evidence_signature(&server.identity)
        .is_err());

    let restored = PeerStore::new();
    server.load_peer_cache(&restored, &path_str, now + 6).await;
    assert!(restored.is_route_quarantined_now(&node_id, now + 6));
    assert!(!restored.is_routeable_now(&node_id, now + 6));
    let status = restored.status(now + 6);
    assert_eq!(
        status
            .bootstrap
            .last_routeability_cache_rollback_protection
            .as_deref(),
        Some("anchored")
    );
    assert!(status
        .bootstrap
        .last_cache_load_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("route_quarantine_restored=1")));

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_older_signed_route_state_generation() {
    // [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Keep the newest
    // monotonic anchor while replaying an older, otherwise valid cache.
    // Descriptors remain recoverable, but stale route health must not.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_041_000;
    let peer_identity = IdentityKeyPair::generate();
    let descriptor = signed_chat_relay_peer_descriptor_for_identity(
        "https://rollback-peer.example".to_string(),
        11,
        now + 4_000,
        &[],
        &peer_identity,
    );
    let node_id = descriptor.node_id();
    let original = Arc::new(PeerStore::new());
    assert!(original.upsert_verified(descriptor.clone(), now).unwrap());
    assert!(original.record_route_forward_success_for_descriptor(&descriptor, now + 1));

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-route-state-rollback-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original, &path_str, now + 2)
        .await
        .unwrap();
    let older_cache = tokio::fs::read(&path).await.unwrap();
    let older_document = PeerStoreCacheDocument::from_json_bytes(&older_cache).unwrap();
    assert_eq!(older_document.verified_client_delivery_generation, 1);
    assert_eq!(older_document.routeability_evidence.len(), 1);

    for observed_at in [now + 3, now + 4, now + 5] {
        assert!(original.record_route_forward_failure_for_descriptor(
            &descriptor,
            observed_at,
            "request_failed",
        ));
    }
    assert!(original.is_route_quarantined_now(&node_id, now + 5));
    Server::save_peer_store_cache_snapshot(&server.identity, &original, &path_str, now + 6)
        .await
        .unwrap();
    let anchor_path = Server::peer_cache_client_delivery_anchor_path(&path_str);
    let newest_anchor_bytes = tokio::fs::read(&anchor_path).await.unwrap();
    let newest_anchor =
        PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(&newest_anchor_bytes).unwrap();
    assert_eq!(newest_anchor.cache_generation, 2);
    assert!(newest_anchor.verify(&server.identity).is_ok());

    tokio::fs::write(&path, &older_cache).await.unwrap();
    let restored = PeerStore::new();
    server.load_peer_cache(&restored, &path_str, now + 7).await;

    assert!(restored.get_valid(&node_id, now + 7).is_some());
    assert!(!restored.is_routeable_now(&node_id, now + 7));
    assert!(!restored.is_route_quarantined_now(&node_id, now + 7));
    let status = restored.status(now + 7);
    assert_eq!(
        status
            .bootstrap
            .last_routeability_cache_rollback_protection
            .as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(
        status.bootstrap.last_routeability_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(status.bootstrap.last_routeability_cache_restored, 0);
    assert_eq!(status.bootstrap.last_routeability_cache_rejected, 1);

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(anchor_path).await;
}

#[tokio::test]
async fn peer_store_cache_revalidates_route_domain_certificates_after_restart() {
    // [ROUTE-DOMAIN-CERTIFICATE-RECOVERY 2026-08-03 by Codex] Exercise the
    // real atomic cache document rather than only the in-memory helpers.
    // The restored node installs its own policy before accepting evidence.
    let identity = IdentityKeyPair::generate();
    let server = Server::new(ServerConfig::default(), identity, None);
    let now = 1_700_030_000;
    let subject = IdentityKeyPair::generate();
    let attestor_a = IdentityKeyPair::generate();
    let attestor_b = IdentityKeyPair::generate();
    let route_domain = [0x71; 16];
    let allowed = [attestor_a.public_key_bytes(), attestor_b.public_key_bytes()];
    let attestations = [&attestor_a, &attestor_b]
        .into_iter()
        .enumerate()
        .map(|(index, attestor)| {
            RouteDomainAttestationV1::new_signed(
                subject.public_key_bytes(),
                route_domain,
                now - 2 + u64::try_from(index).unwrap(),
                now + 600,
                attestor,
            )
            .unwrap()
        })
        .collect();
    let certificate = RouteDomainAttestationCertificateV1::new_verified(
        subject.public_key_bytes(),
        route_domain,
        attestations,
        now,
    )
    .unwrap();

    let original = PeerStore::new();
    original
        .configure_route_domain_attestor_policy(
            &[(subject.public_key_bytes(), route_domain)],
            &allowed,
            2,
            true,
        )
        .unwrap();
    assert!(original
        .import_route_domain_attestation_certificate(certificate, now)
        .unwrap());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-route-domain-certificate-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original, &path_str, now)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    assert_eq!(document.route_domain_certificates.len(), 1);

    let restored = PeerStore::new();
    restored
        .configure_route_domain_attestor_policy(
            &[(subject.public_key_bytes(), route_domain)],
            &allowed,
            2,
            true,
        )
        .unwrap();
    let _ = Server::import_bootstrap_snapshot_bytes(
        &restored,
        "cache",
        &path_str,
        &bytes,
        now + 1,
        Some(&server.identity),
    );
    assert_eq!(
        restored
            .export_route_domain_attestation_certificates(now + 1)
            .len(),
        1
    );
    assert!(restored
        .status(now + 1)
        .bootstrap
        .last_cache_load_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("route_domain_certificates_restored=1")));

    let _ = tokio::fs::remove_file(&path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}

#[tokio::test]
async fn peer_store_cache_can_restore_verified_peers_after_restart() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_000_000;
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let original_store = Arc::new(PeerStore::new());
    assert!(original_store.upsert_verified(signed.clone(), now).unwrap());
    original_store.record_route_forward_success(&signed.node_id(), now + 1);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-restore-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();

    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 2)
        .await
        .unwrap();

    let restored_store = PeerStore::new();
    let bytes = tokio::fs::read(&path).await.unwrap();
    Server::import_bootstrap_snapshot_bytes(
        &restored_store,
        "cache",
        &path_str,
        &bytes,
        now + 3,
        Some(&server.identity),
    );

    assert_eq!(restored_store.len(), 1);
    assert!(restored_store
        .get_valid(&signed.node_id(), now + 3)
        .is_some());
    assert!(restored_store.is_routeable_now(&signed.node_id(), now + 3));

    let status = restored_store.status(now + 3);
    assert_eq!(status.snapshot.valid_peers, 1);
    assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("success")
    );
    assert_eq!(
        status.bootstrap.last_cache_load_source.as_deref(),
        Some("cache")
    );
    assert_eq!(
        status.bootstrap.last_cache_load_status.as_deref(),
        Some("success")
    );
    assert_eq!(status.bootstrap.last_cache_load_at, Some(now + 3));
    assert_eq!(
        status.bootstrap.last_routeability_cache_status.as_deref(),
        Some("restored")
    );
    assert_eq!(status.bootstrap.last_routeability_cache_restored, 1);
    assert_eq!(status.bootstrap.last_routeability_cache_rejected, 0);
    assert_eq!(status.bootstrap.last_routeability_cache_at, Some(now + 3));
    assert!(status
        .recent_audit_events
        .iter()
        .any(|event| event.action == "bootstrap_source"));

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn legacy_peer_cache_without_routeability_fields_remains_accepted() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_010_000;
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let legacy = NodeBootstrapSnapshot::new(now, vec![signed.clone()]);
    let bytes = legacy.to_json_pretty().unwrap();
    let restored_store = PeerStore::new();

    assert!(Server::import_bootstrap_snapshot_bytes(
        &restored_store,
        "cache",
        "legacy-test-cache",
        &bytes,
        now + 1,
        Some(&server.identity),
    ));
    assert!(restored_store
        .get_valid(&signed.node_id(), now + 1)
        .is_some());
    assert!(!restored_store.is_routeable_now(&signed.node_id(), now + 1));
    assert_eq!(
        restored_store
            .status(now + 1)
            .bootstrap
            .last_routeability_cache_status
            .as_deref(),
        Some("empty")
    );
    assert_eq!(
        restored_store
            .status(now + 1)
            .bootstrap
            .last_two_hop_proof_cache_status
            .as_deref(),
        Some("empty")
    );
    assert_eq!(
        restored_store
            .status(now + 1)
            .bootstrap
            .last_three_hop_proof_cache_status
            .as_deref(),
        Some("empty")
    );
    assert_eq!(
        restored_store
            .status(now + 1)
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("empty")
    );
}

#[tokio::test]
async fn peer_store_cache_restores_independently_signed_path_proof_windows() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_015_000;
    let middle = signed_probe_peer_descriptor(
        "https://middle-cache.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x41; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://terminal-cache.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x42; 32],
    );
    let second_middle = signed_probe_peer_descriptor(
        "https://second-middle-cache.example".to_string(),
        3,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x43; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle.clone(), now).unwrap();
    original_store
        .upsert_verified(terminal.clone(), now)
        .unwrap();
    original_store
        .upsert_verified(second_middle.clone(), now)
        .unwrap();
    original_store.record_route_forward_success(&middle.node_id(), now + 1);
    original_store.record_route_forward_success(&terminal.node_id(), now + 1);
    original_store.record_route_forward_success(&second_middle.node_id(), now + 1);
    for offset in 2..=4 {
        original_store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            2,
            2,
            1,
        );
    }
    for offset in 2..=4 {
        original_store.record_blind_relay_three_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            3,
            3,
            2,
        );
    }
    original_store.record_verified_client_onion_delivery(now + 4);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-two-hop-proof-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();
    Server::persist_peer_store_cache_once(&server.identity, &original_store, &path_str, now + 5)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    assert_eq!(document.two_hop_path_proof_events.len(), 3);
    assert!(document
        .verify_two_hop_path_proof_signature(&server.identity)
        .is_ok());
    assert_eq!(
        document.three_hop_path_proof_schema_version,
        THREE_HOP_PATH_PROOF_CACHE_SCHEMA_VERSION
    );
    assert_eq!(document.three_hop_path_proof_events.len(), 3);
    assert!(document
        .verify_three_hop_path_proof_signature(&server.identity)
        .is_ok());
    assert_eq!(
        document.verified_client_delivery_evidence,
        Some(PeerStoreVerifiedClientDeliveryCacheEvidence {
            verified_deliveries: 1,
            last_verified_at: now + 4,
        })
    );
    assert!(document
        .verify_verified_client_delivery_signature(&server.identity)
        .is_ok());
    let persisted_status = original_store.status(now + 5);
    assert_eq!(
        persisted_status
            .bootstrap
            .last_two_hop_proof_cache_persisted,
        3
    );
    assert!(
        persisted_status
            .bootstrap
            .last_two_hop_proof_cache_persisted_stability_ready
    );
    assert_eq!(
        persisted_status
            .bootstrap
            .last_three_hop_proof_cache_persisted,
        3
    );
    assert!(
        persisted_status
            .bootstrap
            .last_three_hop_proof_cache_persisted_stability_ready
    );
    assert_eq!(
        persisted_status
            .bootstrap
            .last_client_delivery_cache_persisted,
        1
    );
    let restored_store = PeerStore::new();
    server
        .load_peer_cache(&restored_store, &path_str, now + 6)
        .await;

    assert!(restored_store.is_routeable_now(&middle.node_id(), now + 6));
    assert!(restored_store.is_routeable_now(&terminal.node_id(), now + 6));
    assert!(restored_store.is_routeable_now(&second_middle.node_id(), now + 6));
    let status = restored_store.status(now + 6);
    assert_eq!(
        status.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("restored")
    );
    assert_eq!(
        status
            .bootstrap
            .last_two_hop_proof_cache_authentication
            .as_deref(),
        Some("verified")
    );
    assert_eq!(status.bootstrap.last_two_hop_proof_cache_restored, 3);
    assert_eq!(
        status
            .bootstrap
            .last_two_hop_proof_cache_rollback_protection
            .as_deref(),
        Some("anchored")
    );
    assert!(
        status
            .bootstrap
            .last_two_hop_proof_cache_restored_stability_ready
    );
    assert!(status.two_hop_path_proof_history.stability_ready);
    assert!(
        status
            .two_hop_path_proof_history
            .recent_message_delivery_ready
    );
    assert_eq!(
        status
            .bootstrap
            .last_three_hop_proof_cache_status
            .as_deref(),
        Some("restored")
    );
    assert_eq!(
        status
            .bootstrap
            .last_three_hop_proof_cache_authentication
            .as_deref(),
        Some("verified")
    );
    assert_eq!(status.bootstrap.last_three_hop_proof_cache_restored, 3);
    assert_eq!(
        status
            .bootstrap
            .last_three_hop_proof_cache_rollback_protection
            .as_deref(),
        Some("anchored")
    );
    assert!(
        status
            .bootstrap
            .last_three_hop_proof_cache_restored_stability_ready
    );
    assert!(status.three_hop_path_proof_history.stability_ready);
    assert!(
        status
            .three_hop_path_proof_history
            .recent_message_delivery_ready
    );
    assert_eq!(status.runtime.blind_relay.received, 0);
    assert_eq!(status.runtime.blind_relay.terminal, 0);
    assert_eq!(status.runtime.blind_relay.forwarded, 0);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        1
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("restored")
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_authentication
            .as_deref(),
        Some("verified")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_restored, 1);
    assert!(!status.blind_relay_quality.real_relay_ready);

    for node_id in [middle.node_id(), terminal.node_id()] {
        restored_store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 6);
    }
    assert!(
        restored_store
            .status(now + 7)
            .blind_relay_quality
            .real_relay_ready
    );

    let anchor_path = Server::peer_cache_client_delivery_anchor_path(&path_str);
    let anchor_bytes = tokio::fs::read(&anchor_path).await.unwrap();
    tokio::fs::remove_file(&anchor_path).await.unwrap();
    let missing_anchor_store = PeerStore::new();
    server
        .load_peer_cache(&missing_anchor_store, &path_str, now + 8)
        .await;
    let missing_anchor_status = missing_anchor_store.status(now + 8);
    assert_eq!(missing_anchor_status.snapshot.valid_peers, 3);
    assert!(!missing_anchor_store.is_routeable_now(&middle.node_id(), now + 8));
    assert!(!missing_anchor_store.is_routeable_now(&terminal.node_id(), now + 8));
    assert_eq!(
        missing_anchor_status
            .bootstrap
            .last_routeability_cache_rollback_protection
            .as_deref(),
        Some("anchor_missing")
    );
    assert_eq!(
        missing_anchor_status
            .bootstrap
            .last_routeability_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(
        missing_anchor_status
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        0
    );
    assert_eq!(
        missing_anchor_status
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("anchor_missing")
    );
    assert_eq!(
        missing_anchor_status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(
        missing_anchor_status
            .bootstrap
            .last_three_hop_proof_cache_rollback_protection
            .as_deref(),
        Some("anchor_missing")
    );
    assert_eq!(
        missing_anchor_status
            .bootstrap
            .last_three_hop_proof_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(
        missing_anchor_status.three_hop_path_proof_history.attempted,
        0
    );

    // [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] A present but
    // unauthentic anchor is not equivalent to the valid crash window.
    // Preserve signed peer descriptors, but fail route state, proofs, and
    // aggregate delivery evidence closed until probes rebuild readiness.
    let mut invalid_anchor =
        PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(&anchor_bytes).unwrap();
    invalid_anchor.signature_ed25519 = "00".repeat(64);
    tokio::fs::write(&anchor_path, invalid_anchor.to_json_pretty().unwrap())
        .await
        .unwrap();
    let invalid_anchor_store = PeerStore::new();
    server
        .load_peer_cache(&invalid_anchor_store, &path_str, now + 9)
        .await;
    let invalid_anchor_status = invalid_anchor_store.status(now + 9);
    assert_eq!(invalid_anchor_status.snapshot.valid_peers, 3);
    assert!(!invalid_anchor_store.is_routeable_now(&middle.node_id(), now + 9));
    assert!(!invalid_anchor_store.is_routeable_now(&terminal.node_id(), now + 9));
    assert_eq!(
        invalid_anchor_status
            .bootstrap
            .last_routeability_cache_rollback_protection
            .as_deref(),
        Some("anchor_invalid")
    );
    assert_eq!(
        invalid_anchor_status
            .bootstrap
            .last_routeability_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(
        invalid_anchor_status
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        0
    );
    assert_eq!(
        invalid_anchor_status
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("anchor_invalid")
    );
    assert_eq!(
        invalid_anchor_status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(
        invalid_anchor_status
            .bootstrap
            .last_three_hop_proof_cache_rollback_protection
            .as_deref(),
        Some("anchor_invalid")
    );
    assert_eq!(
        invalid_anchor_status
            .bootstrap
            .last_three_hop_proof_cache_status
            .as_deref(),
        Some("rejected")
    );

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(anchor_path).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_older_signed_path_proof_generation() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_016_000;
    let middle = signed_probe_peer_descriptor(
        "https://proof-rollback-middle-a.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x71; 32],
    );
    let second_middle = signed_probe_peer_descriptor(
        "https://proof-rollback-middle-b.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x72; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://proof-rollback-terminal.example".to_string(),
        3,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x73; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    for descriptor in [&middle, &second_middle, &terminal] {
        original_store
            .upsert_verified(descriptor.clone(), now)
            .unwrap();
        original_store.record_route_forward_success(&descriptor.node_id(), now + 1);
    }
    original_store.record_blind_relay_two_hop_probe_result_with_context(
        now + 2,
        true,
        "onion_terminal_delivered",
        2,
        2,
        2,
        1,
    );
    original_store.record_blind_relay_three_hop_probe_result_with_context(
        now + 2,
        true,
        "onion_terminal_delivered",
        3,
        3,
        3,
        2,
    );

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-path-proof-rollback-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 3)
        .await
        .unwrap();
    let older_signed_cache = tokio::fs::read(&path).await.unwrap();

    original_store.record_blind_relay_two_hop_probe_result_with_context(
        now + 4,
        true,
        "onion_terminal_delivered",
        2,
        2,
        2,
        1,
    );
    original_store.record_blind_relay_three_hop_probe_result_with_context(
        now + 4,
        true,
        "onion_terminal_delivered",
        3,
        3,
        3,
        2,
    );
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 5)
        .await
        .unwrap();
    tokio::fs::write(&path, older_signed_cache).await.unwrap();

    let restored_store = PeerStore::new();
    server
        .load_peer_cache(&restored_store, &path_str, now + 6)
        .await;
    let status = restored_store.status(now + 6);
    assert_eq!(status.snapshot.valid_peers, 3);
    assert_eq!(
        status
            .bootstrap
            .last_two_hop_proof_cache_rollback_protection
            .as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(
        status.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("rejected")
    );
    assert!(status.two_hop_path_proof_history.events.is_empty());
    assert_eq!(
        status
            .bootstrap
            .last_three_hop_proof_cache_rollback_protection
            .as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(
        status
            .bootstrap
            .last_three_hop_proof_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert!(status.three_hop_path_proof_history.events.is_empty());

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_older_signed_client_delivery_generation() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_017_000;
    let middle = signed_probe_peer_descriptor(
        "https://rollback-middle.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x43; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://rollback-terminal.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x44; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle.clone(), now).unwrap();
    original_store
        .upsert_verified(terminal.clone(), now)
        .unwrap();
    original_store.record_route_forward_success(&middle.node_id(), now + 1);
    original_store.record_route_forward_success(&terminal.node_id(), now + 1);
    original_store.record_verified_client_onion_delivery(now + 2);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-client-delivery-rollback-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 3)
        .await
        .unwrap();
    let older_signed_cache = tokio::fs::read(&path).await.unwrap();

    original_store.record_verified_client_onion_delivery(now + 4);
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 5)
        .await
        .unwrap();
    let current =
        PeerStoreCacheDocument::from_json_bytes(&tokio::fs::read(&path).await.unwrap()).unwrap();
    assert_eq!(current.verified_client_delivery_generation, 2);
    assert_eq!(
        current
            .verified_client_delivery_evidence
            .unwrap()
            .verified_deliveries,
        2
    );

    // [ROUTE-STATE-ROLLBACK-ANCHOR 2026-08-21 by Codex] Replacing only the
    // cache with an older, still-valid signed copy must not revive route
    // health or aggregate delivery evidence. Signed descriptors remain
    // usable and fresh probes can rebuild route readiness.
    tokio::fs::write(&path, &older_signed_cache).await.unwrap();
    let restored_store = PeerStore::new();
    server
        .load_peer_cache(&restored_store, &path_str, now + 6)
        .await;
    let status = restored_store.status(now + 6);
    assert_eq!(status.snapshot.valid_peers, 2);
    assert!(!restored_store.is_routeable_now(&middle.node_id(), now + 6));
    assert!(!restored_store.is_routeable_now(&terminal.node_id(), now + 6));
    assert_eq!(
        status
            .bootstrap
            .last_routeability_cache_rollback_protection
            .as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(
        status.bootstrap.last_routeability_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_authentication
            .as_deref(),
        Some("verified")
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_generation, 1);
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}

#[tokio::test]
async fn peer_store_cache_accepts_cache_ahead_crash_window_and_repairs_anchor() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_017_500;
    let middle = signed_probe_peer_descriptor(
        "https://cache-ahead-middle.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x45; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://cache-ahead-terminal.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x46; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle.clone(), now).unwrap();
    original_store
        .upsert_verified(terminal.clone(), now)
        .unwrap();
    original_store.record_route_forward_success(&middle.node_id(), now + 1);
    original_store.record_route_forward_success(&terminal.node_id(), now + 1);
    original_store.record_verified_client_onion_delivery(now + 2);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-client-delivery-cache-ahead-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    let anchor_path = Server::peer_cache_client_delivery_anchor_path(&path_str);
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 3)
        .await
        .unwrap();
    let older_anchor = tokio::fs::read(&anchor_path).await.unwrap();

    original_store.record_verified_client_onion_delivery(now + 4);
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 5)
        .await
        .unwrap();
    tokio::fs::write(&anchor_path, older_anchor).await.unwrap();

    let restored_store = PeerStore::new();
    server
        .load_peer_cache(&restored_store, &path_str, now + 6)
        .await;
    let status = restored_store.status(now + 6);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        2
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("cache_ahead")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_generation, 2);

    Server::persist_peer_store_cache_once(&server.identity, &restored_store, &path_str, now + 7)
        .await
        .unwrap();
    let repaired_document =
        PeerStoreCacheDocument::from_json_bytes(&tokio::fs::read(&path).await.unwrap()).unwrap();
    let repaired_anchor = PeerStoreVerifiedClientDeliveryAnchor::from_json_bytes(
        &tokio::fs::read(&anchor_path).await.unwrap(),
    )
    .unwrap();
    assert_eq!(repaired_document.verified_client_delivery_generation, 3);
    assert!(repaired_anchor.verify(&server.identity).is_ok());
    assert!(repaired_anchor.matches_document(&repaired_document));
    assert_eq!(
        restored_store
            .status(now + 7)
            .bootstrap
            .last_client_delivery_cache_rollback_protection
            .as_deref(),
        Some("anchored")
    );

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
    let _ = tokio::fs::remove_file(anchor_path).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_tampered_two_hop_proof_independently() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_018_000;
    let middle = signed_probe_peer_descriptor(
        "https://middle-tamper.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x51; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://terminal-tamper.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x52; 32],
    );
    let second_middle = signed_probe_peer_descriptor(
        "https://second-middle-tamper.example".to_string(),
        3,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x53; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle.clone(), now).unwrap();
    original_store
        .upsert_verified(terminal.clone(), now)
        .unwrap();
    original_store
        .upsert_verified(second_middle.clone(), now)
        .unwrap();
    original_store.record_route_forward_success(&middle.node_id(), now + 1);
    original_store.record_route_forward_success(&terminal.node_id(), now + 1);
    original_store.record_route_forward_success(&second_middle.node_id(), now + 1);
    original_store.record_blind_relay_two_hop_probe_result_with_context(
        now + 2,
        true,
        "onion_terminal_delivered",
        2,
        2,
        2,
        1,
    );
    original_store.record_blind_relay_three_hop_probe_result_with_context(
        now + 2,
        true,
        "onion_terminal_delivered",
        3,
        3,
        3,
        2,
    );
    original_store.record_verified_client_onion_delivery(now + 2);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("aeronyx-peer-cache-two-hop-tamper-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 3)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let mut document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    document.two_hop_path_proof_events[0].outcome = "rejected".to_string();
    let tampered = serde_json::to_vec_pretty(&document).unwrap();
    let restored_store = PeerStore::new();
    let anchor_state =
        Server::read_peer_cache_client_delivery_anchor(&path_str, &server.identity).await;
    assert!(Server::import_bootstrap_snapshot_bytes_with_anchor(
        &restored_store,
        "cache",
        &path_str,
        &tampered,
        now + 4,
        Some(&server.identity),
        &anchor_state,
    ));

    assert!(restored_store.is_routeable_now(&middle.node_id(), now + 4));
    assert!(restored_store.is_routeable_now(&terminal.node_id(), now + 4));
    let status = restored_store.status(now + 4);
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("warning")
    );
    assert_eq!(
        status.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(
        status
            .bootstrap
            .last_two_hop_proof_cache_authentication
            .as_deref(),
        Some("signature_invalid")
    );
    assert_eq!(status.bootstrap.last_two_hop_proof_cache_restored, 0);
    assert!(
        !status
            .bootstrap
            .last_two_hop_proof_cache_restored_stability_ready
    );
    assert_eq!(status.bootstrap.last_two_hop_proof_cache_rejected, 1);
    assert!(status.two_hop_path_proof_history.events.is_empty());
    assert_eq!(
        status
            .bootstrap
            .last_three_hop_proof_cache_authentication
            .as_deref(),
        Some("verified")
    );
    assert_eq!(status.bootstrap.last_three_hop_proof_cache_restored, 1);
    assert_eq!(status.three_hop_path_proof_history.events.len(), 1);
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("restored")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_restored, 1);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        1
    );

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_tampered_client_delivery_independently() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_019_000;
    let middle = signed_probe_peer_descriptor(
        "https://middle-client-tamper.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x61; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://terminal-client-tamper.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x62; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle.clone(), now).unwrap();
    original_store
        .upsert_verified(terminal.clone(), now)
        .unwrap();
    original_store.record_route_forward_success(&middle.node_id(), now + 1);
    original_store.record_route_forward_success(&terminal.node_id(), now + 1);
    for offset in 2..=4 {
        original_store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            2,
            2,
            1,
        );
    }
    original_store.record_verified_client_onion_delivery(now + 4);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-client-delivery-tamper-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 5)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let mut document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    document
        .verified_client_delivery_evidence
        .as_mut()
        .unwrap()
        .verified_deliveries = 99;
    let tampered = serde_json::to_vec_pretty(&document).unwrap();
    let restored_store = PeerStore::new();
    let anchor_state =
        Server::read_peer_cache_client_delivery_anchor(&path_str, &server.identity).await;
    assert!(Server::import_bootstrap_snapshot_bytes_with_anchor(
        &restored_store,
        "cache",
        &path_str,
        &tampered,
        now + 6,
        Some(&server.identity),
        &anchor_state,
    ));

    let status = restored_store.status(now + 6);
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("warning")
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_authentication
            .as_deref(),
        Some("signature_invalid")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_restored, 0);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(
        status.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("restored")
    );
    assert!(status.two_hop_path_proof_history.stability_ready);
    assert!(restored_store.is_routeable_now(&middle.node_id(), now + 6));
    assert!(restored_store.is_routeable_now(&terminal.node_id(), now + 6));

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_expired_signed_client_delivery_evidence() {
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_019_500;
    let middle = signed_probe_peer_descriptor(
        "https://middle-client-expired.example".to_string(),
        1,
        now,
        now + 4_000,
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
        [0x63; 32],
    );
    let terminal = signed_probe_peer_descriptor(
        "https://terminal-client-expired.example".to_string(),
        2,
        now,
        now + 4_000,
        vec![NodeCapability::ChatRelay],
        [0x64; 32],
    );
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(middle, now).unwrap();
    original_store.upsert_verified(terminal, now).unwrap();
    original_store.record_verified_client_onion_delivery(now + 1);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "aeronyx-peer-cache-client-delivery-expired-{unique}.json"
    ));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 2)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    assert!(document
        .verify_verified_client_delivery_signature(&server.identity)
        .is_ok());

    // Client-delivery evidence is intentionally valid for only the same
    // 30-minute window as routeability. A valid signature cannot revive
    // historical delivery readiness after that freshness window closes.
    let restore_at = now + 1_902;
    let restored_store = PeerStore::new();
    assert!(Server::import_bootstrap_snapshot_bytes(
        &restored_store,
        "cache",
        &path_str,
        &bytes,
        restore_at,
        Some(&server.identity),
    ));

    let status = restored_store.status(restore_at);
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_authentication
            .as_deref(),
        Some("verified")
    );
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert_eq!(status.bootstrap.last_client_delivery_cache_restored, 0);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert!(!status.blind_relay_quality.real_relay_ready);

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn peer_store_cache_rejects_tampered_routeability_signature() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_020_000;
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let original_store = Arc::new(PeerStore::new());
    original_store.upsert_verified(signed.clone(), now).unwrap();
    original_store.record_route_forward_success(&signed.node_id(), now + 1);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("aeronyx-peer-cache-signature-tamper-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 2)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let mut document = PeerStoreCacheDocument::from_json_bytes(&bytes).unwrap();
    document.routeability_evidence[0].last_success_at = now + 2;
    let tampered = serde_json::to_vec_pretty(&document).unwrap();
    let restored_store = PeerStore::new();

    assert!(Server::import_bootstrap_snapshot_bytes(
        &restored_store,
        "cache",
        &path_str,
        &tampered,
        now + 3,
        Some(&server.identity),
    ));
    assert!(restored_store
        .get_valid(&signed.node_id(), now + 3)
        .is_some());
    assert!(!restored_store.is_routeable_now(&signed.node_id(), now + 3));
    assert_eq!(
        restored_store
            .status(now + 3)
            .bootstrap
            .last_source_status
            .as_deref(),
        Some("warning")
    );
    let status = restored_store.status(now + 3);
    assert_eq!(
        status.bootstrap.last_routeability_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(status.bootstrap.last_routeability_cache_restored, 0);
    assert_eq!(status.bootstrap.last_routeability_cache_rejected, 1);

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn peer_store_cache_retains_expired_signed_peers_after_restart() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now: u64 = 1_800_000_000;
    let peer_key = IdentityKeyPair::generate();
    let expired_descriptor = NodeDescriptor::new(
        peer_key.public_key_bytes(),
        7,
        now.saturating_sub(2_000),
        now.saturating_sub(1_000),
        "aeronyx-test-expired",
    );
    let expired = SignedNodeDescriptor::sign(expired_descriptor, &peer_key).unwrap();
    let node_id = expired.node_id();
    let original_store = Arc::new(PeerStore::new());
    let cache_snapshot = NodeBootstrapSnapshot::new(now, vec![expired]);
    let report = original_store.load_peer_cache_snapshot_from_source(&cache_snapshot, now, "cache");
    assert_eq!(report.inserted, 1);
    assert_eq!(original_store.status(now).peer_summary.expired_peers, 1);

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("aeronyx-peer-cache-expired-restore-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();

    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now)
        .await
        .unwrap();

    let restored_store = PeerStore::new();
    let bytes = tokio::fs::read(&path).await.unwrap();
    Server::import_bootstrap_snapshot_bytes(
        &restored_store,
        "cache",
        &path_str,
        &bytes,
        now,
        Some(&server.identity),
    );

    assert_eq!(restored_store.len(), 1);
    assert!(restored_store.get_valid(&node_id, now).is_none());
    let status = restored_store.status(now);
    assert_eq!(status.snapshot.valid_peers, 0);
    assert_eq!(status.peer_summary.expired_peers, 1);
    assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("success")
    );

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn peer_store_cache_falls_back_to_backup_when_primary_is_corrupt() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now = unix_now_secs();
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let original_store = Arc::new(PeerStore::new());
    assert!(original_store.upsert_verified(signed.clone(), now).unwrap());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("aeronyx-peer-cache-backup-restore-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();
    let backup_path = Server::peer_cache_backup_path(&path_str);

    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now)
        .await
        .unwrap();
    let first_generation = tokio::fs::read(&path).await.unwrap();
    // [PEER-CACHE-BACKUP-DURABILITY 2026-09-21 by Codex] Exercise the
    // production rotation rather than manufacturing a backup in the test.
    // The second publish must preserve the first complete signed primary.
    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now + 1)
        .await
        .unwrap();
    assert_eq!(
        tokio::fs::read(&backup_path).await.unwrap(),
        first_generation
    );
    tokio::fs::write(&path, b"{not-json").await.unwrap();

    let restored_store = PeerStore::new();
    server
        .load_peer_cache(&restored_store, &path_str, now + 2)
        .await;

    assert!(restored_store
        .get_valid(&signed.node_id(), now + 2)
        .is_some());

    let status = restored_store.status(now + 2);
    assert_eq!(status.snapshot.valid_peers, 1);
    assert_eq!(
        status.bootstrap.last_source_kind.as_deref(),
        Some("cache_backup")
    );
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        // The complete prior generation restores descriptors, while the
        // newer anchor correctly keeps rollback-sensitive readiness in a
        // warning state rather than blessing older evidence.
        Some("warning")
    );
    assert_eq!(
        status.bootstrap.last_cache_load_source.as_deref(),
        Some("cache_backup")
    );
    assert_eq!(
        status.bootstrap.last_cache_load_status.as_deref(),
        Some("warning")
    );
    assert_eq!(status.bootstrap.last_cache_load_at, Some(now + 2));

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(backup_path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_client_delivery_anchor_path(&path_str)).await;
}

#[tokio::test]
async fn peer_store_cache_falls_back_to_backup_when_primary_has_no_usable_peers() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now = unix_now_secs();
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let original_store = Arc::new(PeerStore::new());
    assert!(original_store.upsert_verified(signed.clone(), now).unwrap());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-empty-restore-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();
    let backup_path = Server::peer_cache_backup_path(&path_str);

    Server::save_peer_store_cache_snapshot(&server.identity, &original_store, &path_str, now)
        .await
        .unwrap();
    tokio::fs::copy(&path, &backup_path).await.unwrap();

    let empty_snapshot = NodeBootstrapSnapshot::new(now, Vec::new());
    tokio::fs::write(&path, empty_snapshot.to_json_pretty().unwrap())
        .await
        .unwrap();

    let restored_store = PeerStore::new();
    server
        .load_peer_cache(&restored_store, &path_str, now + 1)
        .await;

    assert!(restored_store
        .get_valid(&signed.node_id(), now + 1)
        .is_some());

    let status = restored_store.status(now + 1);
    assert_eq!(status.snapshot.valid_peers, 1);
    assert_eq!(
        status.bootstrap.last_source_kind.as_deref(),
        Some("cache_backup")
    );
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("success")
    );

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(backup_path).await;
}

#[tokio::test]
async fn peer_store_immediate_cache_save_records_restart_recovery_evidence() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.peer_cache_path = Some(
        std::env::temp_dir()
            .join(format!(
                "aeronyx-peer-cache-immediate-{}.json",
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
    let peer_http_clients = PeerHttpClients::build(&server.config).unwrap();
    let peer_store = server
        .init_peer_store(false, peer_http_clients.control.as_ref())
        .await
        .unwrap();
    let path = config.discovery.peer_cache_path.unwrap();

    let bytes = tokio::fs::read(&path)
        .await
        .expect("initial PeerStore cache should be saved during bootstrap");
    let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
    let now = unix_now_secs();
    assert!(snapshot.verified_count_at(now) >= 1);

    let status = peer_store.status(now);
    assert_eq!(
        status.bootstrap.last_cache_save_status.as_deref(),
        Some("success")
    );
    assert_eq!(
        status.bootstrap.last_cache_save_detail.as_deref(),
        Some("snapshot_persisted")
    );
    assert!(status.stability.restart_recovery_configured);

    let _ = tokio::fs::remove_file(path).await;
}

#[tokio::test]
async fn peer_store_cache_write_failure_rearms_delivery_evidence() {
    // [PEER-CACHE-DIRTY-RECOVERY 2026-08-12 by Codex] Reproduce the
    // persistence-task order: claim the pending evidence first, then make
    // the atomic write fail because its parent path is not a directory.
    let server = Server::new(ServerConfig::default(), IdentityKeyPair::generate(), None);
    let peer_store = PeerStore::new();
    peer_store.mark_client_delivery_cache_dirty();
    assert!(peer_store.take_client_delivery_cache_dirty());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let non_directory_parent =
        std::env::temp_dir().join(format!("aeronyx-peer-cache-write-failure-{unique}"));
    tokio::fs::write(&non_directory_parent, b"not a directory")
        .await
        .unwrap();
    let path = non_directory_parent.join("peer-cache.json");
    // [PEER-CACHE-TEST-HTTP-CLIENT 2026-08-23 by Codex] Keep isolated
    // cache tests off macOS system proxy discovery; DynamicStore is not
    // available in every CI/sandbox session.
    let http_client = test_peer_http_client();
    let result = Server::persist_peer_store_cache_with_delivery_witnesses(
        &server.identity,
        &peer_store,
        &server.config.discovery,
        http_client.as_ref(),
        &path.to_string_lossy(),
        1_800_040_000,
        false,
    )
    .await;

    assert!(result.is_err());
    assert!(peer_store.take_client_delivery_cache_dirty());
    tokio::fs::remove_file(non_directory_parent).await.unwrap();
}

#[tokio::test]
async fn peer_store_cache_save_preserves_existing_recovery_snapshot_when_current_view_empty() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, IdentityKeyPair::generate(), None);
    let now = 1_800_000_000;
    let signed = server.build_self_discovery_descriptor(now).unwrap();
    let recovery_store = Arc::new(PeerStore::new());
    assert!(recovery_store.upsert_verified(signed.clone(), now).unwrap());

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("aeronyx-peer-cache-preserve-{unique}.json"));
    let path_str = path.to_string_lossy().to_string();

    Server::save_peer_store_cache_snapshot(&server.identity, &recovery_store, &path_str, now)
        .await
        .unwrap();

    let empty_store = Arc::new(PeerStore::new());
    Server::persist_peer_store_cache_once(&server.identity, &empty_store, &path_str, now + 1)
        .await
        .unwrap();

    let bytes = tokio::fs::read(&path).await.unwrap();
    let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
    assert_eq!(snapshot.verified_count_at(now + 2), 1);
    assert_eq!(snapshot.peers[0].node_id(), signed.node_id());

    let status = empty_store.status(now + 2);
    assert_eq!(
        status.bootstrap.last_cache_save_status.as_deref(),
        Some("skipped")
    );
    assert_eq!(
        status.bootstrap.last_cache_save_detail.as_deref(),
        Some("preserved_existing_snapshot")
    );

    let _ = tokio::fs::remove_file(path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&path_str)).await;
}

#[tokio::test]
async fn peer_store_cache_source_supersedes_expired_static_bootstrap_warning() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let bootstrap_path =
        std::env::temp_dir().join(format!("aeronyx-expired-bootstrap-{unique}.json"));
    let cache_path = std::env::temp_dir().join(format!("aeronyx-fresh-cache-{unique}.json"));
    let bootstrap_path_str = bootstrap_path.to_string_lossy().to_string();
    let cache_path_str = cache_path.to_string_lossy().to_string();

    let now = unix_now_secs();
    let expired =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:9".to_string(), now - 600, now - 1);
    let expired_snapshot = NodeBootstrapSnapshot::new(now - 600, vec![expired]);
    tokio::fs::write(&bootstrap_path, expired_snapshot.to_json_pretty().unwrap())
        .await
        .unwrap();

    let fresh =
        signed_chat_relay_peer_descriptor("http://127.0.0.1:10".to_string(), now, now + 300);
    let cache_store = Arc::new(PeerStore::new());
    assert!(cache_store.upsert_verified(fresh.clone(), now).unwrap());
    let identity = IdentityKeyPair::generate();
    Server::save_peer_store_cache_snapshot(&identity, &cache_store, &cache_path_str, now)
        .await
        .unwrap();

    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.advertise_self = false;
    config.discovery.bootstrap_snapshot_path = Some(bootstrap_path_str.clone());
    config.discovery.peer_cache_path = Some(cache_path_str.clone());
    config.discovery.public_endpoint = Some("node.example.com:443".to_string());

    let server = Server::new(config, identity, None);
    let peer_http_clients = PeerHttpClients::build(&server.config).unwrap();
    let restored_store = server
        .init_peer_store(false, peer_http_clients.control.as_ref())
        .await
        .unwrap();
    let status = restored_store.status(now + 1);

    assert!(restored_store
        .get_valid(&fresh.node_id(), now + 1)
        .is_some());
    assert_eq!(status.snapshot.valid_peers, 1);
    assert_eq!(status.runtime.rejected, 1);
    assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("success")
    );
    assert_eq!(status.bootstrap.recovery_status.as_deref(), Some("success"));

    let _ = tokio::fs::remove_file(bootstrap_path).await;
    let _ = tokio::fs::remove_file(cache_path).await;
    let _ = tokio::fs::remove_file(Server::peer_cache_backup_path(&cache_path_str)).await;
}

#[tokio::test]
async fn peer_store_persistence_task_flushes_cache_on_shutdown() {
    let mut config = ServerConfig::default();
    config.discovery.enabled = true;
    config.discovery.peer_cache_write_interval_secs = 3600;
    config.discovery.peer_cache_path = Some(
        std::env::temp_dir()
            .join(format!(
                "aeronyx-peer-cache-shutdown-{}.json",
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
    server
        .shutdown_tx
        .send(())
        .expect("shutdown receiver should be subscribed");
    handle.await.expect("peer cache task should stop cleanly");

    let bytes = tokio::fs::read(&path)
        .await
        .expect("shutdown should flush PeerStore cache");
    let snapshot = NodeBootstrapSnapshot::from_json_bytes(&bytes).unwrap();
    assert_eq!(snapshot.verified_count_at(now + 1), 1);

    let status = peer_store.status(now + 1);
    assert_eq!(
        status.bootstrap.last_cache_save_status.as_deref(),
        Some("success")
    );
    assert_eq!(
        status.bootstrap.last_cache_save_detail.as_deref(),
        Some("snapshot_persisted")
    );

    let _ = tokio::fs::remove_file(path).await;
}
