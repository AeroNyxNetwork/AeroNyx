// Split from crates/aeronyx-server/src/services/peer_store.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn test_route_candidates_rank_healthy_endpoint_peers_without_payload_data() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let preferred_kp = IdentityKeyPair::generate();
    let stale_kp = IdentityKeyPair::generate();
    let no_endpoint_kp = IdentityKeyPair::generate();

    let mut preferred = signed_descriptor_for(&preferred_kp, 1, now + 2_000);
    preferred.descriptor.public_endpoint = Some("https://preferred.example".to_string());
    preferred.descriptor.capacity.max_sessions = 512;
    preferred = SignedNodeDescriptor::sign(preferred.descriptor, &preferred_kp).unwrap();

    let mut stale = signed_descriptor_for(&stale_kp, 1, now + 90);
    stale.descriptor.public_endpoint = Some("https://stale.example".to_string());
    stale.descriptor.capacity.max_sessions = 64;
    stale = SignedNodeDescriptor::sign(stale.descriptor, &stale_kp).unwrap();

    let no_endpoint = signed_descriptor_for(&no_endpoint_kp, 1, now + 2_000);

    store
        .upsert_verified_from_source(preferred.clone(), now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(stale.clone(), now, "gossip_snapshot")
        .unwrap();
    store
        .upsert_verified_from_source(no_endpoint, now, "gossip_announce")
        .unwrap();

    let candidates = store.route_candidates_with_capability(NodeCapability::ChatRelay, now, 8);
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].node_id(), preferred.node_id());
    assert_eq!(candidates[1].node_id(), stale.node_id());

    let status = store.route_candidate_status(now);
    assert_eq!(status.chat_relay.len(), 2);
    assert_eq!(status.chat_relay[0].health, "healthy");
    assert_eq!(status.chat_relay[1].health, "stale");
    assert!(status.chat_relay[0].endpoint_advertised);
    assert_eq!(status.chat_relay[0].routeability_state, "unknown");
    assert!(!status.chat_relay[0].routeability_ready);
    assert!(status.chat_relay[0].score > status.chat_relay[1].score);
}

#[test]
fn test_is_routeable_now_requires_fresh_successful_route_evidence() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
    descriptor.descriptor.public_endpoint = Some("https://routeable.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();

    store
        .upsert_verified_from_source(descriptor, now, "gossip_announce")
        .unwrap();

    assert!(!store.is_routeable_now(&node_id, now));

    store.record_route_forward_failure(&node_id, now + 1, "request_failed");
    assert!(!store.is_routeable_now(&node_id, now + 2));

    store.record_route_forward_success(&node_id, now + 3);
    assert!(store.is_routeable_now(&node_id, now + 4));

    assert!(!store.is_routeable_now(&node_id, now + 3 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1));
}

#[test]
fn test_routeability_cache_restores_across_sequence_only_descriptor_refresh() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 7, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://warm-restart.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();

    let original = PeerStore::new();
    original
        .upsert_verified_from_source(descriptor.clone(), now, "gossip_announce")
        .unwrap();
    original.record_route_forward_success(&node_id, now + 10);

    let mut refreshed_body = descriptor.descriptor.clone();
    refreshed_body.sequence = 8;
    refreshed_body.issued_at = now + 15;
    refreshed_body.expires_at = now + 4_015;
    let refreshed = SignedNodeDescriptor::sign(refreshed_body, &peer_kp).unwrap();
    original
        .upsert_verified_from_source(refreshed.clone(), now + 15, "gossip_announce")
        .unwrap();
    assert!(original.is_routeable_now(&node_id, now + 16));

    let evidence = original.export_routeability_cache_evidence(now + 20);
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].descriptor_sequence, 8);
    assert_eq!(
        evidence[0].evidence_kind,
        ROUTEABILITY_EVIDENCE_KIND_ROUTE_SURFACE
    );
    assert!(!serde_json::to_string(&evidence)
        .unwrap()
        .contains("warm-restart.example"));

    let mut newer_body = refreshed.descriptor;
    newer_body.sequence = 9;
    newer_body.issued_at = now + 21;
    newer_body.expires_at = now + 4_021;
    let newer = SignedNodeDescriptor::sign(newer_body, &peer_kp).unwrap();
    let restored = PeerStore::new();
    restored
        .upsert_verified_from_source(newer, now + 21, "cache")
        .unwrap();
    let report = restored.restore_routeability_cache_evidence(&evidence, now + 21);

    assert_eq!(
        report,
        PeerStoreRouteabilityCacheRestoreReport {
            total: 1,
            restored: 1,
            rejected: 0,
        }
    );
    assert!(restored.is_routeable_now(&node_id, now + 21));
    let status = restored.status(now + 21);
    assert_eq!(
        status.bootstrap.last_routeability_cache_status.as_deref(),
        Some("restored")
    );
    assert_eq!(status.bootstrap.last_routeability_cache_restored, 1);
    assert_eq!(status.bootstrap.last_routeability_cache_rejected, 0);
    assert_eq!(status.bootstrap.last_routeability_cache_at, Some(now + 21));
}

#[test]
fn routeability_restore_cannot_outlive_descriptor_surface_rotation() {
    // [ROUTEABILITY-RESTORE-RACE 2026-09-25 by Codex] Hold the health
    // lock so restore must retain the descriptor read lock while waiting.
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut body = signed_descriptor_for(&peer_kp, 7, now + 4_000).descriptor;
    body.public_endpoint = Some("https://route-a.example".to_string());
    let original = SignedNodeDescriptor::sign(body.clone(), &peer_kp).unwrap();
    let node_id = original.node_id();
    let source = PeerStore::new();
    source
        .upsert_verified_from_source(original.clone(), now, "gossip_announce")
        .unwrap();
    source.record_route_forward_success(&node_id, now + 1);
    let evidence = source.export_routeability_cache_evidence(now + 2);

    let restored = Arc::new(PeerStore::new());
    restored
        .upsert_verified_from_source(original, now + 2, "cache")
        .unwrap();
    let route_health_guard = restored.route_health.write();
    let restore_store = Arc::clone(&restored);
    let restore = std::thread::spawn(move || {
        restore_store.restore_routeability_cache_evidence(&evidence, now + 3)
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while restored.peers.try_write().is_some() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert!(restored.peers.try_write().is_none());
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(restored.peers.try_write().is_none());

    body.sequence = 8;
    body.issued_at = now + 4;
    body.public_endpoint = Some("https://route-b.example".to_string());
    let rotated = SignedNodeDescriptor::sign(body, &peer_kp).unwrap();
    let rotate_store = Arc::clone(&restored);
    let rotate = std::thread::spawn(move || {
        rotate_store.upsert_verified_from_source(rotated, now + 4, "gossip_announce")
    });
    drop(route_health_guard);
    assert_eq!(restore.join().unwrap().restored, 1);
    assert!(rotate.join().unwrap().unwrap());
    assert!(!restored.is_routeable_now(&node_id, now + 5));
}

#[test]
fn test_route_quarantine_cache_survives_restart_without_failure_details() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 7, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://quarantine.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();

    let original = PeerStore::new();
    original.upsert_verified(descriptor.clone(), now).unwrap();
    original.record_route_forward_success(&node_id, now + 1);
    for observed_at in [now + 2, now + 3, now + 4] {
        assert!(original.record_route_forward_failure_for_descriptor(
            &descriptor,
            observed_at,
            "request_failed",
        ));
    }
    assert!(original.is_route_quarantined_now(&node_id, now + 4));
    assert!(original.take_peer_cache_dirty());

    let evidence = original.export_route_quarantine_cache_evidence(now + 5);
    assert_eq!(evidence.len(), 1);
    let rendered = serde_json::to_string(&evidence).unwrap();
    assert!(!rendered.contains("quarantine.example"));
    assert!(!rendered.contains("request_failed"));

    let mut refreshed_body = descriptor.descriptor.clone();
    refreshed_body.sequence = 8;
    refreshed_body.issued_at = now + 6;
    refreshed_body.expires_at = now + 4_006;
    let refreshed = SignedNodeDescriptor::sign(refreshed_body, &peer_kp).unwrap();
    let restored = PeerStore::new();
    restored.upsert_verified(refreshed, now + 6).unwrap();
    let report = restored.restore_route_quarantine_cache_evidence(&evidence, now + 6);

    assert_eq!(
        report,
        PeerStoreRouteQuarantineCacheRestoreReport {
            total: 1,
            restored: 1,
            rejected: 0,
        }
    );
    assert!(restored.is_route_quarantined_now(&node_id, now + 6));
    assert!(!restored.is_routeable_now(&node_id, now + 6));
    let row = restored
        .status(now + 6)
        .peer_health_summary
        .peers
        .into_iter()
        .find(|row| row.node_id_prefix == hex::encode(&node_id[..4]))
        .expect("restored peer health row");
    assert_eq!(row.route_health, "quarantined");
    assert_eq!(
        row.last_route_failure_reason.as_deref(),
        Some("restart_restored_quarantine")
    );
}

#[test]
fn test_route_quarantine_cache_rejects_rotation_expiry_and_oversized_window() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();

    let source = PeerStore::new();
    source.upsert_verified(descriptor.clone(), now).unwrap();
    for observed_at in [now + 1, now + 2, now + 3] {
        source.record_route_forward_failure(&node_id, observed_at, "request_failed");
    }
    let valid = source
        .export_route_quarantine_cache_evidence(now + 4)
        .remove(0);

    let mut rotated_body = descriptor.descriptor;
    rotated_body.sequence = 2;
    rotated_body.issued_at = now + 5;
    rotated_body.expires_at = now + 4_005;
    rotated_body.public_endpoint = Some("https://route-b.example".to_string());
    let rotated = SignedNodeDescriptor::sign(rotated_body, &peer_kp).unwrap();
    let restored = PeerStore::new();
    restored.upsert_verified(rotated, now + 5).unwrap();

    let mut expired = valid.clone();
    expired.quarantine_until = now + 5;
    let mut oversized = valid.clone();
    oversized.quarantine_until = oversized.quarantined_at + PEER_ROUTE_FAILURE_QUARANTINE_SECS + 1;
    let report =
        restored.restore_route_quarantine_cache_evidence(&[valid, expired, oversized], now + 5);

    assert_eq!(report.total, 3);
    assert_eq!(report.restored, 0);
    assert_eq!(report.rejected, 3);
    assert!(!restored.is_route_quarantined_now(&node_id, now + 5));
}

#[test]
fn test_route_success_clears_quarantine_and_requests_cache_refresh() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://recovery.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();
    let store = PeerStore::new();
    store.upsert_verified(descriptor, now).unwrap();

    for observed_at in [now + 1, now + 2, now + 3] {
        store.record_route_forward_failure(&node_id, observed_at, "request_failed");
    }
    assert!(store.take_peer_cache_dirty());
    store.record_route_forward_success(&node_id, now + 4);

    assert!(store.take_peer_cache_dirty());
    assert!(!store.is_route_quarantined_now(&node_id, now + 4));
    assert!(store
        .export_route_quarantine_cache_evidence(now + 4)
        .is_empty());
}

#[test]
fn test_routeability_cache_accepts_legacy_exact_descriptor_evidence() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 4, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://legacy-cache.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();
    let evidence = PeerStoreRouteabilityCacheEvidence {
        node_id_hex: hex::encode(node_id),
        descriptor_sequence: descriptor.sequence(),
        descriptor_fingerprint_sha256: PeerStore::descriptor_routeability_fingerprint(&descriptor)
            .unwrap(),
        last_success_at: now + 10,
        evidence_kind: ROUTEABILITY_EVIDENCE_KIND_EXACT_DESCRIPTOR.to_string(),
    };

    let restored = PeerStore::new();
    restored.upsert_verified(descriptor, now + 11).unwrap();
    let report = restored.restore_routeability_cache_evidence(&[evidence], now + 11);

    assert_eq!(report.restored, 1);
    assert_eq!(report.rejected, 0);
    assert!(restored.is_routeable_now(&node_id, now + 11));
}

#[test]
fn test_route_surface_changes_invalidate_success_until_reprobed() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 7, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();

    let store = PeerStore::new();
    store.upsert_verified(descriptor.clone(), now).unwrap();
    store.record_route_forward_success(&node_id, now + 10);
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 10);
    assert!(store.is_routeable_now(&node_id, now + 11));
    assert!(store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 11));

    let mut endpoint_changed_body = descriptor.descriptor.clone();
    endpoint_changed_body.sequence = 8;
    endpoint_changed_body.issued_at = now + 20;
    endpoint_changed_body.expires_at = now + 4_020;
    endpoint_changed_body.public_endpoint = Some("https://route-b.example".to_string());
    let endpoint_changed = SignedNodeDescriptor::sign(endpoint_changed_body, &peer_kp).unwrap();
    store
        .upsert_verified(endpoint_changed.clone(), now + 20)
        .unwrap();
    assert!(!store.is_routeable_now(&node_id, now + 21));
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 21));
    assert_eq!(
        store
            .status(now + 21)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        0
    );
    assert!(store
        .export_routeability_cache_evidence(now + 21)
        .is_empty());

    store.record_route_forward_success(&node_id, now + 22);
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 22);
    assert!(store.is_routeable_now(&node_id, now + 23));
    assert!(store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 23));

    let mut kem_changed_body = endpoint_changed.descriptor;
    kem_changed_body.sequence = 9;
    kem_changed_body.issued_at = now + 30;
    kem_changed_body.expires_at = now + 4_030;
    kem_changed_body.kem_alg = 1;
    kem_changed_body.kem_public = [7u8; 32];
    let kem_changed = SignedNodeDescriptor::sign(kem_changed_body, &peer_kp).unwrap();
    store
        .upsert_verified(kem_changed.clone(), now + 30)
        .unwrap();
    assert!(!store.is_routeable_now(&node_id, now + 31));
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 31));

    store.record_route_forward_success(&node_id, now + 32);
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 32);
    let mut capability_changed_body = kem_changed.descriptor;
    capability_changed_body.sequence = 10;
    capability_changed_body.issued_at = now + 40;
    capability_changed_body.expires_at = now + 4_040;
    capability_changed_body
        .capabilities
        .push(NodeCapability::DirectoryMirrorCarrier);
    let capability_changed = SignedNodeDescriptor::sign(capability_changed_body, &peer_kp).unwrap();
    store
        .upsert_verified(capability_changed.clone(), now + 40)
        .unwrap();
    // [MIRROR-CAPABILITY 2026-07-24 by Codex] A newly advertised carrier
    // surface must be probed; a prior role's success cannot be inherited.
    assert!(!store.is_routeable_now(&node_id, now + 41));
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 41));

    store.record_route_forward_success(&node_id, now + 42);
    let mut feature_changed_body = capability_changed.descriptor;
    feature_changed_body.sequence = 11;
    feature_changed_body.issued_at = now + 50;
    feature_changed_body.expires_at = now + 4_050;
    feature_changed_body = feature_changed_body
        .with_protocol_features([NodeProtocolFeature::BlindRelayFailureReceiptV1]);
    let feature_changed = SignedNodeDescriptor::sign(feature_changed_body, &peer_kp).unwrap();
    store.upsert_verified(feature_changed, now + 50).unwrap();
    assert!(
        !store.is_routeable_now(&node_id, now + 51),
        "a negotiated response contract requires a fresh route probe"
    );
}

#[test]
fn test_route_success_rejects_non_current_route_surface() {
    let now = 1_700_000_100;
    let identity = IdentityKeyPair::generate();
    let mut original = signed_descriptor_for(&identity, 7, now + 4_000);
    original.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    original = SignedNodeDescriptor::sign(original.descriptor, &identity).unwrap();
    let node_id = original.node_id();

    let store = PeerStore::new();
    store.upsert_verified(original.clone(), now).unwrap();

    let mut rotated_body = original.descriptor.clone();
    rotated_body.sequence = 8;
    rotated_body.issued_at = now + 20;
    rotated_body.expires_at = now + 4_020;
    rotated_body.public_endpoint = Some("https://route-b.example".to_string());
    let rotated = SignedNodeDescriptor::sign(rotated_body, &identity).unwrap();
    store.upsert_verified(rotated.clone(), now + 20).unwrap();

    assert!(!store.record_route_forward_success_for_descriptor(&original, now + 21));
    assert!(!store.is_routeable_now(&node_id, now + 21));
    assert!(store.record_route_forward_success_for_descriptor(&rotated, now + 22));
    assert!(store.is_routeable_now(&node_id, now + 22));

    let mut refreshed_body = rotated.descriptor.clone();
    refreshed_body.sequence = 9;
    refreshed_body.issued_at = now + 30;
    refreshed_body.expires_at = now + 4_030;
    let refreshed = SignedNodeDescriptor::sign(refreshed_body, &identity).unwrap();
    store.upsert_verified(refreshed, now + 30).unwrap();

    // Sequence/validity refreshes preserve the same signed routing surface.
    assert!(store.record_route_forward_success_for_descriptor(&rotated, now + 31));
    assert!(store.is_routeable_now(&node_id, now + 31));
}

#[test]
fn test_verified_client_delivery_route_evidence_is_all_or_nothing() {
    let now = 1_700_000_100;
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut middle = signed_descriptor_for(&middle_identity, 7, now + 4_000);
    middle.descriptor.public_endpoint = Some("https://middle-a.example".to_string());
    middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_identity).unwrap();
    let middle_node_id = middle.node_id();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://terminal-a.example".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();
    let terminal_node_id = terminal.node_id();

    let store = PeerStore::new();
    store.upsert_verified(middle.clone(), now).unwrap();
    store.upsert_verified(terminal.clone(), now).unwrap();

    let mut rotated_terminal_body = terminal.descriptor.clone();
    rotated_terminal_body.sequence = 8;
    rotated_terminal_body.issued_at = now + 20;
    rotated_terminal_body.expires_at = now + 4_020;
    rotated_terminal_body.public_endpoint = Some("https://terminal-b.example".to_string());
    let rotated_terminal =
        SignedNodeDescriptor::sign(rotated_terminal_body, &terminal_identity).unwrap();
    store
        .upsert_verified(rotated_terminal.clone(), now + 20)
        .unwrap();

    // [CLIENT-DELIVERY-ATOMIC-ROUTE-EVIDENCE 2026-08-11 by Codex] One
    // rotated hop rejects the entire receipt transition. The unchanged
    // middle must not receive partial capability or route-success credit.
    assert!(!store.record_verified_client_onion_route_delivery(&middle, &terminal, now + 21,));
    let rejected = store.status(now + 21);
    assert_eq!(
        rejected
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        0
    );
    assert_eq!(
        rejected.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    assert!(!store.is_routeable_now(&middle_node_id, now + 21));
    assert!(!store.is_routeable_now(&terminal_node_id, now + 21));
    assert!(!store.take_client_delivery_cache_dirty());

    assert!(store.record_verified_client_onion_route_delivery(
        &middle,
        &rotated_terminal,
        now + 22,
    ));
    let accepted = store.status(now + 22);
    assert_eq!(
        accepted
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        1
    );
    assert_eq!(
        accepted.blind_relay_quality.delivery_receipt_capable_peers,
        2
    );
    assert!(
        accepted
            .blind_relay_quality
            .authenticated_delivery_path_ready
    );
    assert_eq!(
        accepted
            .blind_relay_quality
            .authenticated_delivery_path_reason,
        "authenticated_receipt_path_ready"
    );
    assert!(accepted.blind_relay_quality.real_relay_ready);
    assert!(store.is_routeable_now(&middle_node_id, now + 22));
    assert!(store.is_routeable_now(&terminal_node_id, now + 22));
    assert!(store.take_client_delivery_cache_dirty());
}

#[test]
fn test_route_failure_rejects_non_current_route_surface() {
    let now = 1_700_000_100;
    let identity = IdentityKeyPair::generate();
    let mut original = signed_descriptor_for(&identity, 7, now + 4_000);
    original.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    original = SignedNodeDescriptor::sign(original.descriptor, &identity).unwrap();
    let node_id = original.node_id();

    let store = PeerStore::new();
    store.upsert_verified(original.clone(), now).unwrap();

    let mut rotated_body = original.descriptor.clone();
    rotated_body.sequence = 8;
    rotated_body.issued_at = now + 20;
    rotated_body.expires_at = now + 4_020;
    rotated_body.public_endpoint = Some("https://route-b.example".to_string());
    let rotated = SignedNodeDescriptor::sign(rotated_body, &identity).unwrap();
    store.upsert_verified(rotated.clone(), now + 20).unwrap();
    assert!(store.record_route_forward_success_for_descriptor(&rotated, now + 21));

    // [ROUTE-FAILURE-SURFACE-BINDING 2026-08-11 by Codex] A delayed
    // failure from route A must not poison route B or start its quarantine.
    for observed_at in [now + 22, now + 23, now + 24] {
        assert!(!store.record_route_forward_failure_for_descriptor(
            &original,
            observed_at,
            "request_failed",
        ));
    }
    let health = store.route_health.read();
    let route_health = health.get(&node_id).unwrap();
    assert_eq!(route_health.failure_count, 0);
    assert_eq!(route_health.consecutive_failures, 0);
    assert!(route_health.quarantine_until.is_none());
    drop(health);
    assert!(store.is_routeable_now(&node_id, now + 24));

    let mut refreshed_body = rotated.descriptor.clone();
    refreshed_body.sequence = 9;
    refreshed_body.issued_at = now + 30;
    refreshed_body.expires_at = now + 4_030;
    let refreshed = SignedNodeDescriptor::sign(refreshed_body, &identity).unwrap();
    store.upsert_verified(refreshed, now + 30).unwrap();

    // A sequence/validity-only refresh preserves the same route surface.
    for observed_at in [now + 31, now + 32, now + 33] {
        assert!(store.record_route_forward_failure_for_descriptor(
            &rotated,
            observed_at,
            "request_failed",
        ));
    }
    assert!(store.is_route_quarantined_now(&node_id, now + 33));
}

#[test]
fn test_route_failure_reason_admission_preserves_known_buckets_and_redacts_open_text() {
    let now = 1_700_000_100;
    let identity = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&identity, 1, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://route.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
    let node_id = descriptor.node_id();

    let store = PeerStore::new();
    store.upsert_verified(descriptor.clone(), now).unwrap();

    assert!(store.record_route_forward_failure_for_descriptor(
        &descriptor,
        now + 1,
        "peer_relay_http_502",
    ));
    assert_eq!(
        store
            .route_health
            .read()
            .get(&node_id)
            .and_then(|health| health.last_failure_reason.as_deref()),
        Some("peer_relay_http_502")
    );

    // [PEER-HEALTH-REASON-BOUNDARY 2026-08-21 by Codex] A malformed
    // status or appended peer-controlled detail must never enter route
    // reputation diagnostics, even through the legacy string API.
    assert!(store.record_route_forward_failure_for_descriptor(
        &descriptor,
        now + 2,
        "peer_relay_http_600 receiver=private",
    ));
    assert_eq!(
        store
            .route_health
            .read()
            .get(&node_id)
            .and_then(|health| health.last_failure_reason.as_deref()),
        Some(PrivacySafePeerHealthReason::UNKNOWN)
    );
    assert!(store.recent_audit_events().iter().all(|event| {
        !event.detail.contains("receiver=private") && !event.detail.contains("peer_relay_http_600")
    }));
}

#[test]
fn test_relay_protection_reason_admission_redacts_unknown_details() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let node_id = [23u8; 32];

    store.record_blind_relay_rejected(now, "invalid_signature route=private");
    store.record_blind_relay_quarantine_started(now + 1, "operator=private");
    store.record_peer_relay_rejection(&node_id, now + 2, "receiver=private");
    store.record_peer_relay_quarantine_started(&node_id, now + 3, now + 303, "endpoint=private");

    let protection = store.relay_protection_health.read();
    let health = protection.get(&node_id).expect("peer protection health");
    assert_eq!(
        health.last_rejection_reason.as_deref(),
        Some(PrivacySafePeerHealthReason::UNKNOWN)
    );
    assert_eq!(
        health.last_quarantine_reason.as_deref(),
        Some(PrivacySafePeerHealthReason::UNKNOWN)
    );
    drop(protection);

    let audit = store.recent_audit_events();
    assert!(audit.iter().all(|event| {
        !event.detail.contains("route=private")
            && !event.detail.contains("operator=private")
            && !event.detail.contains("receiver=private")
            && !event.detail.contains("endpoint=private")
    }));
    assert!(audit.iter().any(|event| {
        event.action == "blind_relay_forward"
            && event.outcome == "rejected"
            && event.detail == PrivacySafePeerHealthReason::UNKNOWN
    }));
}

#[test]
fn test_route_success_and_surface_rotation_are_atomic() {
    let now = 1_700_000_100;
    let identity = IdentityKeyPair::generate();
    let mut original = signed_descriptor_for(&identity, 7, now + 4_000);
    original.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    original = SignedNodeDescriptor::sign(original.descriptor, &identity).unwrap();
    let node_id = original.node_id();

    let mut rotated_body = original.descriptor.clone();
    rotated_body.sequence = 8;
    rotated_body.issued_at = now + 20;
    rotated_body.expires_at = now + 4_020;
    rotated_body.public_endpoint = Some("https://route-b.example".to_string());
    let rotated = SignedNodeDescriptor::sign(rotated_body, &identity).unwrap();

    let store = Arc::new(PeerStore::new());
    store.upsert_verified(original.clone(), now).unwrap();
    let start = Arc::new(Barrier::new(3));

    // [ROUTE-SUCCESS-SURFACE-BINDING 2026-08-10 by Codex] Race the old
    // route observation against the signed route rotation. Either the old
    // write is rejected, or the later rotation invalidates it.
    let observer_store = Arc::clone(&store);
    let observer_start = Arc::clone(&start);
    let observer = std::thread::spawn(move || {
        observer_start.wait();
        observer_store.record_route_forward_success_for_descriptor(&original, now + 21)
    });

    let writer_store = Arc::clone(&store);
    let writer_start = Arc::clone(&start);
    let writer = std::thread::spawn(move || {
        writer_start.wait();
        writer_store.upsert_verified(rotated, now + 20).unwrap();
    });

    start.wait();
    let _ = observer.join().unwrap();
    writer.join().unwrap();

    assert_eq!(
        store
            .get_valid(&node_id, now + 21)
            .unwrap()
            .descriptor
            .public_endpoint
            .as_deref(),
        Some("https://route-b.example")
    );
    assert!(!store.is_routeable_now(&node_id, now + 21));
}

#[test]
fn test_receipt_evidence_rejects_non_current_route_surface() {
    let now = 1_700_000_100;
    let identity = IdentityKeyPair::generate();
    let mut original = signed_descriptor_for(&identity, 7, now + 4_000);
    original.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    original.descriptor.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    original = SignedNodeDescriptor::sign(original.descriptor, &identity).unwrap();
    let node_id = original.node_id();

    let store = PeerStore::new();
    store.upsert_verified(original.clone(), now).unwrap();

    let mut rotated_body = original.descriptor.clone();
    rotated_body.sequence = 8;
    rotated_body.issued_at = now + 10;
    rotated_body.expires_at = now + 4_010;
    rotated_body.public_endpoint = Some("https://route-b.example".to_string());
    let rotated = SignedNodeDescriptor::sign(rotated_body, &identity).unwrap();
    store.upsert_verified(rotated.clone(), now + 10).unwrap();

    // [RECEIPT-EVIDENCE-SURFACE-BINDING 2026-08-10 by Codex] A receipt
    // verified against the old route cannot authorize the rotated route,
    // even though both descriptors have the same stable node identity.
    assert!(!store
        .record_purpose_bound_delivery_receipt_capability_for_descriptor(&original, now + 11,));
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 11));

    assert!(
        store.record_purpose_bound_delivery_receipt_capability_for_descriptor(&rotated, now + 12,)
    );
    assert!(store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 13));

    let mut refreshed_body = rotated.descriptor.clone();
    refreshed_body.sequence = 9;
    refreshed_body.issued_at = now + 20;
    refreshed_body.expires_at = now + 4_020;
    let refreshed = SignedNodeDescriptor::sign(refreshed_body, &identity).unwrap();
    store.upsert_verified(refreshed, now + 20).unwrap();

    // Sequence/TTL-only refreshes retain the same signed route surface, so
    // an otherwise current observation remains valid across normal leases.
    assert!(
        store.record_purpose_bound_delivery_receipt_capability_for_descriptor(&rotated, now + 21,)
    );
    assert!(store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 22));
}

#[test]
fn test_routeability_cache_rejects_tampered_future_and_stale_evidence() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 3, now + 10_000);
    descriptor.descriptor.public_endpoint = Some("https://strict-cache.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();

    let source = PeerStore::new();
    source.upsert_verified(descriptor.clone(), now).unwrap();
    source.record_route_forward_success(&node_id, now + 5);
    let valid = source.export_routeability_cache_evidence(now + 6).remove(0);
    let mut tampered = valid.clone();
    let replacement = if tampered.descriptor_fingerprint_sha256.starts_with('0') {
        "1"
    } else {
        "0"
    };
    tampered
        .descriptor_fingerprint_sha256
        .replace_range(..1, replacement);
    let mut future = valid.clone();
    future.last_success_at = now + 100;
    let mut stale = valid;
    stale.last_success_at = now.saturating_sub(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1);

    let restored = PeerStore::new();
    restored.upsert_verified(descriptor, now + 7).unwrap();
    let report = restored.restore_routeability_cache_evidence(&[tampered, future, stale], now + 7);

    assert_eq!(report.total, 3);
    assert_eq!(report.restored, 0);
    assert_eq!(report.rejected, 3);
    assert!(!restored.is_routeable_now(&node_id, now + 7));
    assert_eq!(
        restored
            .status(now + 7)
            .bootstrap
            .last_routeability_cache_status
            .as_deref(),
        Some("rejected")
    );
}

#[test]
fn test_two_hop_proof_cache_restores_only_with_current_route_pool() {
    let now = 1_700_200_000;
    let middle_kp = IdentityKeyPair::generate();
    let terminal_kp = IdentityKeyPair::generate();
    let mut middle = signed_descriptor_for(&middle_kp, 1, now + 4_000);
    middle.descriptor.public_endpoint = Some("https://middle.example".to_string());
    middle
        .descriptor
        .capabilities
        .push(NodeCapability::OnionMiddle);
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_kp).unwrap();
    let mut terminal = signed_descriptor_for(&terminal_kp, 1, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://terminal.example".to_string());
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_kp).unwrap();

    let source = PeerStore::new();
    source.upsert_verified(middle.clone(), now).unwrap();
    source.upsert_verified(terminal.clone(), now).unwrap();
    source.record_route_forward_success(&middle.node_id(), now + 1);
    source.record_route_forward_success(&terminal.node_id(), now + 1);
    for offset in 2..=4 {
        source.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            2,
            2,
            1,
        );
    }
    let route_evidence = source.export_routeability_cache_evidence(now + 5);
    let proof_events = source.export_two_hop_path_proof_cache_events(now + 5);

    let without_routes = PeerStore::new();
    assert_eq!(
        without_routes.restore_two_hop_path_proof_cache_events(&proof_events, now + 6),
        PeerStoreTwoHopProofCacheRestoreReport {
            total: 3,
            restored: 0,
            rejected: 3,
        }
    );

    let restored = PeerStore::new();
    restored.upsert_verified(middle, now + 6).unwrap();
    restored.upsert_verified(terminal, now + 6).unwrap();
    assert_eq!(
        restored
            .restore_routeability_cache_evidence(&route_evidence, now + 6)
            .restored,
        2
    );
    let report = restored.restore_two_hop_path_proof_cache_events(&proof_events, now + 6);

    assert_eq!(report.total, 3);
    assert_eq!(report.restored, 3);
    assert_eq!(report.rejected, 0);
    let status = restored.status(now + 6);
    assert_eq!(
        status.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("restored")
    );
    assert_eq!(status.bootstrap.last_two_hop_proof_cache_restored, 3);
    assert_eq!(status.bootstrap.last_two_hop_proof_cache_rejected, 0);
    assert!(status.two_hop_path_proof_history.stability_ready);
    assert!(
        status
            .two_hop_path_proof_history
            .recent_message_delivery_ready
    );
    assert_eq!(status.runtime.blind_relay.two_hop_probe_attempted, 3);
    assert_eq!(status.runtime.blind_relay.two_hop_probe_succeeded, 3);
    assert_eq!(status.runtime.blind_relay.received, 0);
    assert_eq!(status.runtime.blind_relay.terminal, 0);
    assert_eq!(status.runtime.blind_relay.forwarded, 0);
}

#[test]
fn test_route_candidates_deprioritize_recent_forward_failures_without_payload_data() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let healthy_kp = IdentityKeyPair::generate();
    let failing_kp = IdentityKeyPair::generate();

    let mut healthy = signed_descriptor_for(&healthy_kp, 1, now + 2_000);
    healthy.descriptor.public_endpoint = Some("https://healthy.example".to_string());
    healthy = SignedNodeDescriptor::sign(healthy.descriptor, &healthy_kp).unwrap();

    let mut failing = signed_descriptor_for(&failing_kp, 1, now + 2_000);
    failing.descriptor.public_endpoint = Some("https://failing.example".to_string());
    failing = SignedNodeDescriptor::sign(failing.descriptor, &failing_kp).unwrap();
    let failing_node_id = failing.node_id();
    let failing_prefix = hex::encode(&failing_node_id[..4]);

    store
        .upsert_verified_from_source(healthy.clone(), now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(failing.clone(), now, "gossip_announce")
        .unwrap();

    store.record_route_forward_failure(&failing_node_id, now + 1, "request_failed");
    store.record_route_forward_failure(&failing_node_id, now + 2, "request_failed");
    store.record_route_forward_failure(&failing_node_id, now + 3, "http_502");

    let candidates = store.route_candidates_with_capability(NodeCapability::ChatRelay, now + 4, 8);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].node_id(), healthy.node_id());

    let status = store.route_candidate_status(now + 4);
    let failing_row = status
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == failing_prefix)
        .expect("failing peer should remain visible as a quarantined candidate");
    assert_eq!(failing_row.route_health, "quarantined");
    assert_eq!(failing_row.routeability_state, "quarantined");
    assert!(!failing_row.routeability_ready);
    assert!(failing_row.route_quarantined);
    assert_eq!(failing_row.route_quarantine_count, 1);
    assert_eq!(
        failing_row.route_quarantine_remaining_seconds,
        Some(PEER_ROUTE_FAILURE_QUARANTINE_SECS - 1)
    );
    assert_eq!(failing_row.route_failure_count, 3);
    assert_eq!(failing_row.route_consecutive_failures, 3);
    assert_eq!(
        failing_row.last_route_failure_reason.as_deref(),
        Some("http_502")
    );

    let status_json = serde_json::to_string(&status).unwrap();
    assert!(!status_json.contains("failing.example"));
    assert!(!status_json.contains(&hex::encode(failing_node_id)));
    assert!(!status_json.contains("encrypted_blob"));

    store.record_route_forward_success(&failing_node_id, now + 5);
    let recovered = store.route_candidate_status(now + 6);
    let recovered_row = recovered
        .chat_relay
        .iter()
        .find(|row| row.node_id_prefix == failing_prefix)
        .expect("recovered peer should still be reported");
    assert_eq!(recovered_row.route_health, "healthy");
    assert_eq!(recovered_row.routeability_state, "reachable");
    assert!(recovered_row.routeability_ready);
    assert_eq!(recovered_row.last_routeability_probe_at, Some(now + 5));
    assert!(!recovered_row.route_quarantined);
    assert_eq!(recovered_row.route_quarantine_remaining_seconds, None);
    assert_eq!(recovered_row.route_consecutive_failures, 0);
    assert_eq!(recovered_row.last_route_success_at, Some(now + 5));
}

#[test]
fn test_route_governance_summarizes_quality_without_route_metadata() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let relay_kp = IdentityKeyPair::generate();
    let middle_kp = IdentityKeyPair::generate();
    let failing_kp = IdentityKeyPair::generate();

    let mut relay = signed_descriptor_for(&relay_kp, 1, now + 2_000);
    relay.descriptor.public_endpoint = Some("https://relay.example".to_string());
    relay = SignedNodeDescriptor::sign(relay.descriptor, &relay_kp).unwrap();
    let relay_node_id = relay.node_id();

    let mut middle = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle.descriptor.public_endpoint = Some("https://middle.example".to_string());
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle.node_id();

    let mut failing = signed_descriptor_for(&failing_kp, 1, now + 2_000);
    failing.descriptor.public_endpoint = Some("https://failing.example".to_string());
    failing = SignedNodeDescriptor::sign(failing.descriptor, &failing_kp).unwrap();
    let failing_node_id = failing.node_id();

    store
        .upsert_verified_from_source(relay, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(middle, now, "gossip_snapshot")
        .unwrap();
    store
        .upsert_verified_from_source(failing, now, "gossip_announce")
        .unwrap();

    store.record_route_forward_success(&relay_node_id, now + 1);
    store.record_route_forward_success(&middle_node_id, now + 2);
    store.record_route_forward_failure(&failing_node_id, now + 3, "request_failed");
    store.record_route_forward_failure(&failing_node_id, now + 4, "request_failed");
    store.record_route_forward_failure(&failing_node_id, now + 5, "http_502");

    let governance = store.status(now + 6).route_governance;
    assert_eq!(governance.contract_version, "route_governance.v1");
    assert_eq!(governance.source, "peer_store_route_candidates");
    assert_eq!(governance.status, "attention");
    assert!(governance.route_pool_ready);
    assert!(!governance.quality_ready);
    assert!(governance.chat_single_hop_ready);
    assert!(governance.chat_two_hop_onion_ready);
    assert_eq!(governance.candidates_total, 5);
    assert_eq!(governance.routeable_total, 3);
    assert_eq!(governance.routeable_privacy_relays, 1);
    assert_eq!(governance.routeable_chat_relays, 1);
    assert_eq!(governance.routeable_onion_middle_hops, 1);
    assert_eq!(governance.quarantined_total, 2);
    assert_eq!(governance.failing_total, 0);
    assert_eq!(governance.degraded_total, 0);
    assert_eq!(governance.unreachable_total, 0);
    assert_eq!(
        governance.quarantine_threshold,
        PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD
    );
    assert_eq!(
        governance.quarantine_seconds,
        PEER_ROUTE_FAILURE_QUARANTINE_SECS
    );
    assert_eq!(
        governance.routeability_stale_after_seconds,
        PEER_ROUTEABILITY_STALE_AFTER_SECS
    );
    assert!(governance.best_score.is_some());
    assert!(governance.worst_score.is_some());
    assert!(governance.average_score.is_some());

    let governance_json = serde_json::to_string(&governance).unwrap();
    assert!(!governance_json.contains("relay.example"));
    assert!(!governance_json.contains("middle.example"));
    assert!(!governance_json.contains("failing.example"));
    assert!(!governance_json.contains(&hex::encode(relay_node_id)));
    assert!(!governance_json.contains(&hex::encode(middle_node_id)));
    assert!(!governance_json.contains(&hex::encode(failing_node_id)));
    assert!(!governance_json.contains("route_id"));
    assert!(!governance_json.contains("encrypted_blob"));
    assert!(!governance_json.contains("receiver_pubkey"));
    assert!(!governance_json.contains("payload_b64"));
}

#[test]
fn test_peer_health_summary_reports_gossip_route_and_quarantine_without_payload_data() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
    descriptor.descriptor.public_endpoint = Some("https://peer-health.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();
    let node_prefix = hex::encode(&node_id[..4]);

    store
        .upsert_verified_from_source(descriptor, now, "gossip_announce")
        .unwrap();
    store.record_route_forward_failure(&node_id, now + 1, "request_failed");
    store.record_route_forward_failure(&node_id, now + 2, "http_502");
    store.record_peer_relay_rejection(&node_id, now + 3, "duplicate_route");
    store.record_peer_relay_quarantine_started(&node_id, now + 4, now + 304, "failure_threshold");

    let status = store.status(now + 10).peer_health_summary;
    assert_eq!(status.total_peers, 1);
    assert_eq!(status.quarantined_peers, 1);
    let row = status.peers.first().expect("peer health row");
    assert_eq!(row.node_id_prefix, node_prefix);
    assert_eq!(row.health, "quarantined");
    assert_eq!(row.source, "gossip_announce");
    assert_eq!(row.last_successful_gossip_at, Some(now));
    assert_eq!(row.last_successful_gossip_age_seconds, Some(10));
    assert_eq!(row.route_failure_count, 2);
    assert_eq!(row.route_consecutive_failures, 2);
    assert_eq!(row.routeability_state, "unreachable");
    assert!(!row.routeability_ready);
    assert_eq!(row.last_routeability_probe_at, Some(now + 2));
    assert_eq!(row.last_routeability_probe_age_seconds, Some(8));
    assert_eq!(row.last_route_failure_reason.as_deref(), Some("http_502"));
    assert!(row.relay_quarantined);
    assert_eq!(row.relay_rejection_count, 1);
    assert_eq!(row.relay_quarantine_count, 1);
    assert_eq!(row.relay_quarantine_remaining_seconds, Some(294));
    assert_eq!(
        row.last_relay_rejection_reason.as_deref(),
        Some("duplicate_route")
    );
    assert_eq!(
        row.last_relay_quarantine_reason.as_deref(),
        Some("failure_threshold")
    );

    let status_json = serde_json::to_string(&status).unwrap();
    assert!(!status_json.contains("peer-health.example"));
    assert!(!status_json.contains(&hex::encode(node_id)));
    assert!(!status_json.contains("encrypted_blob"));
    assert!(!status_json.contains("route_id"));
}

#[test]
fn test_route_candidates_apply_exclusion_before_limit_for_self_filtering() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let self_kp = IdentityKeyPair::generate();
    let peer_kp = IdentityKeyPair::generate();

    let mut self_descriptor = signed_descriptor_for(&self_kp, 1, now + 2_000);
    self_descriptor.descriptor.public_endpoint = Some("https://self.example".to_string());
    self_descriptor.descriptor.capacity.max_sessions = 1024;
    self_descriptor = SignedNodeDescriptor::sign(self_descriptor.descriptor, &self_kp).unwrap();
    let self_node_id = self_descriptor.node_id();

    let mut peer_descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
    peer_descriptor.descriptor.public_endpoint = Some("https://peer.example".to_string());
    peer_descriptor.descriptor.capacity.max_sessions = 32;
    peer_descriptor = SignedNodeDescriptor::sign(peer_descriptor.descriptor, &peer_kp).unwrap();
    let peer_node_id = peer_descriptor.node_id();

    store
        .upsert_verified_from_source(self_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(peer_descriptor, now, "gossip_announce")
        .unwrap();

    let candidates = store.route_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        now,
        1,
        &[self_node_id],
    );

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].node_id(), peer_node_id);
}

#[test]
fn test_route_endpoint_network_anti_affinity_is_fail_closed_and_prefix_aware() {
    let now = 1_700_000_100;
    let descriptor = |endpoint: &str| {
        let identity = IdentityKeyPair::generate();
        let mut descriptor = signed_descriptor_for(&identity, 1, now + 2_000);
        descriptor.descriptor.public_endpoint = Some(endpoint.to_string());
        SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap()
    };

    let ipv4_a = descriptor("http://203.0.113.10:8422");
    let ipv4_same_24 = descriptor("https://203.0.113.240:8422");
    let ipv4_other_24 = descriptor("http://203.0.114.10:8422");
    assert!(!PeerStore::route_endpoints_are_network_diverse(
        &ipv4_a,
        &ipv4_same_24,
    ));
    assert!(PeerStore::route_endpoints_are_network_diverse(
        &ipv4_a,
        &ipv4_other_24,
    ));

    let ipv6_a = descriptor("http://[2001:db8:1::10]:8422");
    let ipv6_same_48 = descriptor("https://[2001:db8:1:ffff::20]:8422");
    let ipv6_other_48 = descriptor("http://[2001:db8:2::10]:8422");
    assert!(!PeerStore::route_endpoints_are_network_diverse(
        &ipv6_a,
        &ipv6_same_48,
    ));
    assert!(PeerStore::route_endpoints_are_network_diverse(
        &ipv6_a,
        &ipv6_other_48,
    ));

    let dns_a = descriptor("https://Relay.Example.com.:8422");
    let dns_same = descriptor("https://relay.example.com:9443");
    let dns_other = descriptor("https://relay.other.example:8422");
    let malformed = descriptor("http:///");
    assert!(!PeerStore::route_endpoints_are_network_diverse(
        &dns_a, &dns_same,
    ));
    assert!(PeerStore::route_endpoints_are_network_diverse(
        &dns_a, &dns_other,
    ));
    assert!(!PeerStore::route_endpoints_are_network_diverse(
        &dns_a, &malformed,
    ));
}

#[test]
fn test_route_path_planner_selects_unique_hops_without_payload_data() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let self_kp = IdentityKeyPair::generate();
    let shared_kp = IdentityKeyPair::generate();
    let middle_kp = IdentityKeyPair::generate();
    let relay_kp = IdentityKeyPair::generate();

    let mut self_descriptor = signed_descriptor_for(&self_kp, 1, now + 2_000);
    self_descriptor.descriptor.capabilities =
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay];
    self_descriptor.descriptor.public_endpoint = Some("https://self.example".to_string());
    self_descriptor.descriptor.capacity.max_sessions = 4096;
    self_descriptor = SignedNodeDescriptor::sign(self_descriptor.descriptor, &self_kp).unwrap();
    let self_node_id = self_descriptor.node_id();

    let mut shared_descriptor = signed_descriptor_for(&shared_kp, 1, now + 2_000);
    shared_descriptor.descriptor.capabilities =
        vec![NodeCapability::OnionMiddle, NodeCapability::ChatRelay];
    shared_descriptor.descriptor.public_endpoint = Some("https://shared.example".to_string());
    shared_descriptor.descriptor.capacity.max_sessions = 2048;
    shared_descriptor =
        SignedNodeDescriptor::sign(shared_descriptor.descriptor, &shared_kp).unwrap();
    let shared_node_id = shared_descriptor.node_id();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint = Some("https://middle.example".to_string());
    middle_descriptor.descriptor.capacity.max_sessions = 512;
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle_descriptor.node_id();

    let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
    relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    relay_descriptor.descriptor.public_endpoint = Some("https://relay.example".to_string());
    relay_descriptor.descriptor.capacity.max_sessions = 256;
    relay_descriptor = SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
    let relay_node_id = relay_descriptor.node_id();

    store
        .upsert_verified_from_source(self_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(shared_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
        .unwrap();
    store.record_route_forward_success(&shared_node_id, now);
    store.record_route_forward_success(&middle_node_id, now);
    store.record_route_forward_success(&relay_node_id, now);

    let path = store
        .route_path_with_capabilities_excluding(
            &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
            now,
            &[self_node_id],
        )
        .expect("two-hop path should be available");

    assert_eq!(path.len(), 2);
    assert_eq!(path[0].node_id(), shared_node_id);
    assert_eq!(path[1].node_id(), relay_node_id);
    assert_ne!(path[0].node_id(), path[1].node_id());
    assert!(!path.iter().any(|hop| hop.node_id() == self_node_id));
}

#[test]
fn test_route_path_planner_backtracks_around_collocated_high_score_hop() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let preferred_middle_kp = IdentityKeyPair::generate();
    let alternate_middle_kp = IdentityKeyPair::generate();
    let terminal_kp = IdentityKeyPair::generate();

    let mut preferred_middle = signed_descriptor_for(&preferred_middle_kp, 1, now + 2_000);
    preferred_middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    preferred_middle.descriptor.public_endpoint = Some("http://203.0.113.10:8422".to_string());
    preferred_middle.descriptor.capacity.max_sessions = 4096;
    preferred_middle =
        SignedNodeDescriptor::sign(preferred_middle.descriptor, &preferred_middle_kp).unwrap();
    let preferred_middle_id = preferred_middle.node_id();

    let mut alternate_middle = signed_descriptor_for(&alternate_middle_kp, 1, now + 2_000);
    alternate_middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    alternate_middle.descriptor.public_endpoint = Some("http://198.51.100.10:8422".to_string());
    alternate_middle.descriptor.capacity.max_sessions = 64;
    alternate_middle =
        SignedNodeDescriptor::sign(alternate_middle.descriptor, &alternate_middle_kp).unwrap();
    let alternate_middle_id = alternate_middle.node_id();

    let mut terminal = signed_descriptor_for(&terminal_kp, 1, now + 2_000);
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal.descriptor.public_endpoint = Some("http://203.0.113.90:8422".to_string());
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_kp).unwrap();
    let terminal_id = terminal.node_id();

    for descriptor in [preferred_middle, alternate_middle, terminal] {
        store
            .upsert_verified_from_source(descriptor, now, "gossip_announce")
            .unwrap();
    }
    for node_id in [preferred_middle_id, alternate_middle_id, terminal_id] {
        store.record_route_forward_success(&node_id, now);
    }

    let path = store
        .route_path_with_capabilities_excluding(
            &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
            now,
            &[],
        )
        .expect("a lower-scored network-diverse path should be selected");

    assert_eq!(path[0].node_id(), alternate_middle_id);
    assert_eq!(path[1].node_id(), terminal_id);
    assert_ne!(path[0].node_id(), preferred_middle_id);
    assert!(PeerStore::route_endpoints_are_network_diverse(
        &path[0], &path[1],
    ));
}

#[test]
fn test_route_path_status_is_privacy_safe_and_marks_incomplete_paths() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let middle_kp = IdentityKeyPair::generate();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint =
        Some("https://private-middle.example".to_string());
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle_descriptor.node_id();

    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();

    let status = store.route_candidate_status(now);
    assert!(!status.planned_paths.chat_single_hop.complete);
    assert_eq!(status.planned_paths.chat_single_hop.hop_count, 0);
    assert!(!status.planned_paths.chat_two_hop_onion_ready.complete);
    assert_eq!(status.planned_paths.chat_two_hop_onion_ready.hop_count, 0);
    assert_eq!(status.onion_middle.len(), 1);
    assert_eq!(status.onion_middle[0].routeability_state, "unknown");
    assert!(!status.onion_middle[0].routeability_ready);

    let status_json = serde_json::to_string(&status).unwrap();
    assert!(!status_json.contains("private-middle.example"));
    assert!(!status_json.contains(&hex::encode(middle_node_id)));
    assert!(!status_json.contains("encrypted_blob"));
    assert!(!status_json.contains("receiver_pubkey"));
}

#[test]
fn test_route_path_status_does_not_count_quarantined_peers_as_ready() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let middle_kp = IdentityKeyPair::generate();
    let relay_kp = IdentityKeyPair::generate();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint =
        Some("https://quarantined-middle.example".to_string());
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle_descriptor.node_id();

    let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
    relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    relay_descriptor.descriptor.public_endpoint =
        Some("https://quarantined-relay.example".to_string());
    relay_descriptor = SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
    let relay_node_id = relay_descriptor.node_id();

    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
        .unwrap();

    for node_id in [&middle_node_id, &relay_node_id] {
        store.record_route_forward_success(node_id, now + 1);
        store.record_route_forward_failure(node_id, now + 2, "request_failed");
        store.record_route_forward_failure(node_id, now + 3, "request_failed");
        store.record_route_forward_failure(node_id, now + 4, "http_502");
    }

    let status = store.route_candidate_status(now + 5);
    assert!(!status.planned_paths.chat_single_hop.complete);
    assert_eq!(status.planned_paths.chat_single_hop.hop_count, 0);
    assert!(!status.planned_paths.chat_two_hop_onion_ready.complete);
    assert_eq!(status.planned_paths.chat_two_hop_onion_ready.hop_count, 0);
    assert_eq!(status.chat_relay[0].route_health, "quarantined");
    assert!(status.chat_relay[0].route_quarantined);
    assert_eq!(status.onion_middle[0].route_health, "quarantined");
    assert!(status.onion_middle[0].route_quarantined);

    let status_json = serde_json::to_string(&status).unwrap();
    assert!(!status_json.contains("quarantined-middle.example"));
    assert!(!status_json.contains("quarantined-relay.example"));
    assert!(!status_json.contains(&hex::encode(middle_node_id)));
    assert!(!status_json.contains(&hex::encode(relay_node_id)));
    assert!(!status_json.contains("encrypted_blob"));
    assert!(!status_json.contains("receiver_pubkey"));
}

#[test]
fn test_route_quarantine_expiry_allows_reprobe_without_marking_ready() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
    descriptor.descriptor.public_endpoint =
        Some("https://reprobe-after-quarantine.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();
    let node_prefix = hex::encode(&node_id[..4]);

    store
        .upsert_verified_from_source(descriptor.clone(), now, "gossip_announce")
        .unwrap();

    let cold_start_probe_candidates = store.route_probe_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        now + 1,
        8,
        &[],
    );
    assert_eq!(cold_start_probe_candidates.len(), 1);
    assert_eq!(cold_start_probe_candidates[0].node_id(), node_id);

    store.record_route_forward_success(&node_id, now + 1);
    store.record_route_forward_failure(&node_id, now + 2, "request_failed");
    store.record_route_forward_failure(&node_id, now + 3, "request_failed");
    store.record_route_forward_failure(&node_id, now + 4, "http_502");

    let quarantined_candidates =
        store.route_candidates_with_capability(NodeCapability::ChatRelay, now + 5, 8);
    assert!(quarantined_candidates.is_empty());

    let after_quarantine = now + 4 + PEER_ROUTE_FAILURE_QUARANTINE_SECS + 1;
    let reprobe_candidates =
        store.route_candidates_with_capability(NodeCapability::ChatRelay, after_quarantine, 8);
    assert_eq!(reprobe_candidates.len(), 1);
    assert_eq!(reprobe_candidates[0].node_id(), descriptor.node_id());

    let status = store.route_candidate_status(after_quarantine);
    let row = status
        .chat_relay
        .iter()
        .find(|candidate| candidate.node_id_prefix == node_prefix)
        .expect("expired quarantine peer should be visible for reprobe");
    assert_eq!(row.route_health, "failing");
    assert_eq!(row.routeability_state, "unreachable");
    assert!(!row.routeability_ready);
    assert!(!row.route_quarantined);
    assert_eq!(row.route_quarantine_remaining_seconds, None);
    assert_eq!(row.route_quarantine_count, 1);

    store.record_route_forward_success(&node_id, after_quarantine + 1);
    let recovered = store.route_candidate_status(after_quarantine + 2);
    let recovered_row = recovered
        .chat_relay
        .iter()
        .find(|candidate| candidate.node_id_prefix == node_prefix)
        .expect("successful reprobe should keep peer visible");
    assert_eq!(recovered_row.route_health, "healthy");
    assert_eq!(recovered_row.routeability_state, "reachable");
    assert!(recovered_row.routeability_ready);
    assert_eq!(
        recovered_row.last_routeability_probe_at,
        Some(after_quarantine + 1)
    );

    let status_json = serde_json::to_string(&recovered).unwrap();
    assert!(!status_json.contains("reprobe-after-quarantine.example"));
    assert!(!status_json.contains(&hex::encode(node_id)));
    assert!(!status_json.contains("encrypted_blob"));
    assert!(!status_json.contains("receiver_pubkey"));
}

#[test]
fn test_route_probe_candidates_allow_cooled_down_quarantine_without_user_routeability() {
    let store = PeerStore::new();
    let now = 1_700_000_200;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 1, now + 2_000);
    descriptor.descriptor.public_endpoint = Some("https://probe-recovery-only.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();
    let node_id = descriptor.node_id();
    let node_prefix = hex::encode(&node_id[..4]);

    store
        .upsert_verified_from_source(descriptor.clone(), now, "gossip_announce")
        .unwrap();

    store.record_route_forward_success(&node_id, now + 1);
    store.record_route_forward_failure(&node_id, now + 2, "request_failed");
    store.record_route_forward_failure(&node_id, now + 3, "request_failed");
    store.record_route_forward_failure(&node_id, now + 4, "http_502");

    let still_cooling_down_at = now + 4 + PEER_ROUTE_RECOVERY_PROBE_AFTER_SECS - 1;
    let strict_during_cooldown =
        store.route_candidates_with_capability(NodeCapability::ChatRelay, still_cooling_down_at, 8);
    let probe_during_cooldown = store.route_probe_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        still_cooling_down_at,
        8,
        &[],
    );
    assert!(strict_during_cooldown.is_empty());
    assert!(probe_during_cooldown.is_empty());

    let recovery_probe_at = now + 4 + PEER_ROUTE_RECOVERY_PROBE_AFTER_SECS;
    let strict_recovery_candidates =
        store.route_candidates_with_capability(NodeCapability::ChatRelay, recovery_probe_at, 8);
    let probe_recovery_candidates = store.route_probe_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        recovery_probe_at,
        8,
        &[],
    );
    assert!(strict_recovery_candidates.is_empty());
    assert_eq!(probe_recovery_candidates.len(), 1);
    assert_eq!(probe_recovery_candidates[0].node_id(), node_id);

    let status = store.route_candidate_status(recovery_probe_at);
    let row = status
        .chat_relay
        .iter()
        .find(|candidate| candidate.node_id_prefix == node_prefix)
        .expect("cooled-down quarantined peer remains visible in status");
    assert_eq!(row.route_health, "quarantined");
    assert_eq!(row.routeability_state, "quarantined");
    assert!(!row.routeability_ready);
    assert!(row.route_quarantined);

    store.record_route_forward_success(&node_id, recovery_probe_at + 1);
    let recovered =
        store.route_candidates_with_capability(NodeCapability::ChatRelay, recovery_probe_at + 2, 8);
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].node_id(), node_id);

    let status_json = serde_json::to_string(&status).unwrap();
    assert!(!status_json.contains("probe-recovery-only.example"));
    assert!(!status_json.contains(&hex::encode(node_id)));
    assert!(!status_json.contains("encrypted_blob"));
    assert!(!status_json.contains("receiver_pubkey"));
}

#[test]
fn permissionless_admission_is_exact_replay_safe_and_live_sequence_fenced() {
    let now = 1_780_000_000;
    let identity = IdentityKeyPair::generate();
    let descriptor = permissionless_descriptor_for(&identity, 7, now, "https://8.8.8.8:8422");
    let store = PeerStore::with_max_peers(1);

    assert_eq!(
        store.admit_permissionless_descriptor(descriptor.clone(), now),
        PermissionlessNodeAdmissionOutcome::Admitted
    );
    assert_eq!(
        store.admit_permissionless_descriptor(descriptor.clone(), now),
        PermissionlessNodeAdmissionOutcome::ExactReplay
    );

    let mut conflict_body = descriptor.descriptor.clone();
    conflict_body.capabilities.push(NodeCapability::AgentRelay);
    let conflict = SignedNodeDescriptor::sign(conflict_body, &identity).unwrap();
    assert_eq!(
        store.admit_permissionless_descriptor(conflict, now),
        PermissionlessNodeAdmissionOutcome::Conflict
    );

    let live = permissionless_descriptor_for(&identity, 9, now, "https://8.8.8.8:8422");
    assert!(store.upsert_verified(live, now).unwrap());
    assert_eq!(
        store.admit_permissionless_descriptor(descriptor, now),
        PermissionlessNodeAdmissionOutcome::Stale
    );
    assert_eq!(store.len(), 1);
}

#[test]
fn permissionless_admission_rejects_unsafe_or_ambiguous_descriptor_shapes() {
    let now = 1_780_000_000;
    let identity = IdentityKeyPair::generate();
    let store = PeerStore::new();

    let mut bad_signature =
        permissionless_descriptor_for(&identity, 1, now, "https://8.8.8.8:8422");
    bad_signature.signature[0] ^= 0x01;
    assert_eq!(
        store.admit_permissionless_descriptor(bad_signature, now),
        PermissionlessNodeAdmissionOutcome::Rejected
    );

    assert_eq!(
        store.admit_permissionless_descriptor(
            permissionless_descriptor_for(&identity, 0, now, "https://8.8.8.8:8422"),
            now,
        ),
        PermissionlessNodeAdmissionOutcome::Rejected
    );

    let mut overlong =
        permissionless_descriptor_for(&identity, 1, now, "https://8.8.8.8:8422").descriptor;
    overlong.expires_at = now + UNTRUSTED_DISCOVERY_MAX_LIFETIME_SECS + 1;
    assert_eq!(
        store.admit_permissionless_descriptor(
            SignedNodeDescriptor::sign(overlong, &identity).unwrap(),
            now,
        ),
        PermissionlessNodeAdmissionOutcome::Rejected
    );

    for endpoint in [
        "http://127.0.0.1:8422",
        "http://10.0.0.1:8422",
        "https://node.example:8422",
        " https://8.8.8.8:8422",
    ] {
        assert_eq!(
            store.admit_permissionless_descriptor(
                permissionless_descriptor_for(&identity, 1, now, endpoint),
                now,
            ),
            PermissionlessNodeAdmissionOutcome::Rejected
        );
    }

    let mut duplicate =
        permissionless_descriptor_for(&identity, 2, now, "https://8.8.8.8:8422").descriptor;
    duplicate.capabilities.push(NodeCapability::ChatRelay);
    assert_eq!(
        store.admit_permissionless_descriptor(
            SignedNodeDescriptor::sign(duplicate, &identity).unwrap(),
            now,
        ),
        PermissionlessNodeAdmissionOutcome::Rejected
    );

    let mut invalid_kem =
        permissionless_descriptor_for(&identity, 3, now, "https://8.8.8.8:8422").descriptor;
    invalid_kem.kem_alg = 1;
    assert_eq!(
        store.admit_permissionless_descriptor(
            SignedNodeDescriptor::sign(invalid_kem, &identity).unwrap(),
            now,
        ),
        PermissionlessNodeAdmissionOutcome::Rejected
    );
    assert_eq!(store.len(), 0);
}

#[test]
fn permissionless_admission_capacity_is_hard_and_independent_from_live_peers() {
    let now = 1_780_000_000;
    let store = PeerStore::with_max_peers(1);
    let mut first = None;
    for index in 0..UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY {
        let identity = IdentityKeyPair::generate();
        let descriptor = permissionless_descriptor_for(&identity, 1, now, "https://8.8.8.8:8422");
        if index == 0 {
            first = Some(descriptor.clone());
        }
        assert_eq!(
            store.admit_permissionless_descriptor(descriptor, now),
            PermissionlessNodeAdmissionOutcome::Admitted
        );
    }
    let overflow = IdentityKeyPair::generate();
    assert_eq!(
        store.admit_permissionless_descriptor(
            permissionless_descriptor_for(&overflow, 1, now, "https://8.8.8.8:8422"),
            now,
        ),
        PermissionlessNodeAdmissionOutcome::Saturated
    );
    assert_eq!(
        store.admit_permissionless_descriptor(first.unwrap(), now),
        PermissionlessNodeAdmissionOutcome::ExactReplay,
        "capacity must not reject an exact retry already retained"
    );
    assert_eq!(store.len(), 0);

    let live =
        permissionless_descriptor_for(&IdentityKeyPair::generate(), 1, now, "https://9.9.9.9:8422");
    assert!(store.upsert_verified(live, now).unwrap());
    assert_eq!(store.len(), 1);
}

#[test]
fn test_delivery_witness_requester_admission_is_explicit_and_replaceable() {
    let store = PeerStore::new();
    let first = [0x11; 32];
    let second = [0x22; 32];
    assert!(!store.verified_delivery_witness_requester_allowed(&first));

    store.configure_verified_delivery_witness_requesters(&[first]);
    assert!(store.verified_delivery_witness_requester_allowed(&first));
    assert!(!store.verified_delivery_witness_requester_allowed(&second));

    store.configure_verified_delivery_witness_requesters(&[second]);
    assert!(!store.verified_delivery_witness_requester_allowed(&first));
    assert!(store.verified_delivery_witness_requester_allowed(&second));
}

#[test]
fn test_custody_witness_requester_admission_is_independent_and_replaceable() {
    let store = PeerStore::new();
    let delivery = [0x31; 32];
    let custody = [0x32; 32];
    let replacement = [0x33; 32];

    store.configure_verified_delivery_witness_requesters(&[delivery]);
    store.configure_custody_audit_witness_requesters(&[custody]);
    assert!(store.verified_delivery_witness_requester_allowed(&delivery));
    assert!(!store.custody_audit_witness_requester_allowed(&delivery));
    assert!(store.custody_audit_witness_requester_allowed(&custody));
    assert!(!store.verified_delivery_witness_requester_allowed(&custody));

    store.configure_custody_audit_witness_requesters(&[replacement]);
    assert!(!store.custody_audit_witness_requester_allowed(&custody));
    assert!(store.custody_audit_witness_requester_allowed(&replacement));
    assert!(store.verified_delivery_witness_requester_allowed(&delivery));
}

#[test]
fn test_external_witness_route_gate_clears_anchored_readiness_not_descriptors() {
    // [EXTERNAL-WITNESS-ROUTE-GATE 2026-08-21 by Codex] Model the state
    // present immediately after cache import and before public listeners.
    // A whole-host rollback decision must revoke every readiness section
    // committed by anchor v3 while retaining signed discovery descriptors.
    let store = PeerStore::new();
    let now = 1_700_100_000;
    let routeable_identity = IdentityKeyPair::generate();
    let quarantined_identity = IdentityKeyPair::generate();
    let mut routeable = signed_descriptor_for(&routeable_identity, 1, now + 4_000);
    routeable.descriptor.public_endpoint = Some("https://routeable.example".to_string());
    routeable = SignedNodeDescriptor::sign(routeable.descriptor, &routeable_identity).unwrap();
    let mut quarantined = signed_descriptor_for(&quarantined_identity, 1, now + 4_000);
    quarantined.descriptor.public_endpoint = Some("https://quarantined.example".to_string());
    quarantined =
        SignedNodeDescriptor::sign(quarantined.descriptor, &quarantined_identity).unwrap();
    let routeable_node_id = routeable.node_id();
    let quarantined_node_id = quarantined.node_id();
    store.upsert_verified(routeable, now).unwrap();
    store.upsert_verified(quarantined, now).unwrap();
    store.record_route_forward_success(&routeable_node_id, now + 1);
    for observed_at in [now + 2, now + 3, now + 4] {
        store.record_route_forward_failure(&quarantined_node_id, observed_at, "request_failed");
    }
    for offset in 5..=7 {
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            2,
            2,
            1,
        );
        store.record_blind_relay_three_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            3,
            3,
            2,
        );
    }
    store.record_verified_client_onion_delivery(now + 7);
    store.record_blind_relay_terminal(now + 7, 0, 32);

    assert!(store.is_routeable_now(&routeable_node_id, now + 8));
    assert!(store.is_route_quarantined_now(&quarantined_node_id, now + 8));
    let before = store.status(now + 8);
    assert_eq!(before.snapshot.valid_peers, 2);
    assert_eq!(before.two_hop_path_proof_history.attempted, 3);
    assert_eq!(before.three_hop_path_proof_history.attempted, 3);
    assert_eq!(
        before.runtime.blind_relay.verified_client_onion_deliveries,
        1
    );
    assert!(store.take_peer_cache_dirty());

    store.clear_restored_peer_cache_readiness_evidence(now + 9, "external_witness_rollback");
    let after = store.status(now + 10);
    assert_eq!(after.snapshot.valid_peers, 2);
    assert!(store.get_valid(&routeable_node_id, now + 10).is_some());
    assert!(store.get_valid(&quarantined_node_id, now + 10).is_some());
    assert!(!store.is_routeable_now(&routeable_node_id, now + 10));
    assert!(!store.is_route_quarantined_now(&quarantined_node_id, now + 10));
    assert_eq!(after.two_hop_path_proof_history.attempted, 0);
    assert_eq!(after.three_hop_path_proof_history.attempted, 0);
    assert_eq!(
        after.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(after.runtime.blind_relay.terminal, 1);
    assert_eq!(
        after.bootstrap.last_routeability_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(
        after.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(
        after.bootstrap.last_three_hop_proof_cache_status.as_deref(),
        Some("rejected")
    );
    assert_eq!(
        after.bootstrap.last_client_delivery_cache_status.as_deref(),
        Some("rejected")
    );
    assert!(after.recent_audit_events.iter().any(|event| {
        event.action == "peer_cache_external_witness_gate"
            && event.outcome == "rejected"
            && event.detail.contains("reason=external_witness_rollback")
            && !event.detail.contains("routeable.example")
            && !event.detail.contains("quarantined.example")
    }));
    assert!(store.take_peer_cache_dirty());
}

#[test]
fn test_peer_quorum_reports_peer_view_ready_without_routeable_endpoint() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, true, true, 2);
    store
        .upsert_verified_from_source(signed_descriptor(1, now + 1_000), now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(signed_descriptor(1, now + 1_000), now, "gossip_snapshot")
        .unwrap();
    store.record_gossip_round(now + 20, 2, 2, 1, None);

    let status = store.status(now + 60);

    assert_eq!(status.peer_quorum.status, "peer_view_ready");
    assert!(!status.peer_quorum.quorum_ready);
    assert_eq!(status.peer_quorum.valid_peers, 2);
    assert_eq!(status.peer_quorum.routeable_chat_relays, 0);
    assert_eq!(status.peer_quorum.healthy_ratio_percent, 100);
    assert!(status
        .peer_quorum
        .next_action
        .contains("public chat relay endpoint"));
}

#[test]
fn test_peer_quorum_ready_requires_fresh_routeable_restart_recoverable_peers() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let first_kp = IdentityKeyPair::generate();
    let second_kp = IdentityKeyPair::generate();

    let mut first = signed_descriptor_for(&first_kp, 1, now + 1_000);
    first.descriptor.public_endpoint = Some("https://peer-one.example".to_string());
    first = SignedNodeDescriptor::sign(first.descriptor, &first_kp).unwrap();
    let first_node_id = first.node_id();

    let mut second = signed_descriptor_for(&second_kp, 1, now + 1_000);
    second.descriptor.public_endpoint = Some("https://peer-two.example".to_string());
    second = SignedNodeDescriptor::sign(second.descriptor, &second_kp).unwrap();
    let second_node_id = second.node_id();

    store.configure_bootstrap_status(true, true, true, 2);
    store
        .upsert_verified_from_source(first, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(second, now, "gossip_snapshot")
        .unwrap();
    store.record_gossip_round(now + 20, 2, 2, 1, None);
    store.record_route_forward_success(&first_node_id, now + 30);
    store.record_route_forward_success(&second_node_id, now + 31);

    let status = store.status(now + 60);

    assert_eq!(status.peer_quorum.status, "route_ready");
    assert!(status.peer_quorum.quorum_ready);
    assert_eq!(status.peer_quorum.min_valid_peers, 2);
    assert_eq!(status.peer_quorum.valid_peers, 2);
    assert_eq!(status.peer_quorum.healthy_peers, 2);
    assert_eq!(status.peer_quorum.routeable_chat_relays, 2);
    assert!(status.peer_quorum.restart_recovery_configured);
    assert!(status.peer_quorum.relay_foundation_ready);
    assert!(status
        .peer_quorum
        .privacy_boundary
        .contains("not public-chain consensus"));
}
