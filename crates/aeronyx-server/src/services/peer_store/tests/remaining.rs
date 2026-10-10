// Split from crates/aeronyx-server/src/services/peer_store.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn endpoint_attestation_direct_peer_store_entries_reject_without_mutation() {
    // [ENDPOINT-ATTESTATION-TRANSPORT 2026-09-24 by Codex] Only the API
    // adapter owns verification. Trusted and untrusted PeerStore entry
    // points must reject this carrier without counters, audit, or peers.
    let now = 1_780_000_000;
    let store = PeerStore::new();
    let message = NodeDiscoveryMessage::EndpointEvidenceAttestationV1 {
        attestation_frame: vec![0x55; 289],
    };
    let before = store.status(now);
    let expected = PeerStoreImportReport {
        total: 1,
        inserted: 0,
        candidates: 0,
        unchanged: 0,
        stale: 0,
        rejected: 1,
    };

    assert_eq!(store.apply_discovery_message(&message, now), expected);
    assert_eq!(store.status(now), before);
    assert_eq!(
        store.apply_untrusted_discovery_message(&message, now),
        expected
    );
    assert_eq!(store.status(now), before);
}

#[test]
fn test_upsert_verified_stores_descriptor() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);

    assert_eq!(
        store.upsert_verified(descriptor, 1_700_000_100).unwrap(),
        true
    );
    assert_eq!(store.len(), 1);
    assert_eq!(store.snapshot(1_700_000_100).valid_peers, 1);
}

#[test]
fn test_same_sequence_is_idempotent() {
    let store = PeerStore::new();
    let kp = IdentityKeyPair::generate();
    let descriptor = signed_descriptor_for(&kp, 1, 1_700_001_000);
    let same = signed_descriptor_for(&kp, 1, 1_700_001_000);

    assert_eq!(
        store
            .upsert_verified(descriptor, 1_700_000_100)
            .expect("first insert"),
        true
    );
    assert_eq!(
        store
            .upsert_verified(same, 1_700_000_100)
            .expect("same sequence"),
        false
    );
}

#[test]
fn test_stale_sequence_rejected() {
    let store = PeerStore::new();
    let kp = IdentityKeyPair::generate();
    let newer = signed_descriptor_for(&kp, 2, 1_700_001_000);
    let older = signed_descriptor_for(&kp, 1, 1_700_001_000);

    store.upsert_verified(newer, 1_700_000_100).unwrap();
    let err = store.upsert_verified(older, 1_700_000_100).unwrap_err();

    assert!(matches!(err, PeerStoreError::StaleSequence { .. }));
}

#[test]
fn test_expired_descriptor_rejected() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_000_050);

    let err = store
        .upsert_verified(descriptor, 1_700_000_100)
        .unwrap_err();
    assert!(matches!(err, PeerStoreError::VerificationFailed));
    assert!(store.is_empty());
}

#[test]
fn test_max_peers_rejects_new_descriptor_but_allows_existing_update() {
    let store = PeerStore::with_max_peers(1);
    let kp = IdentityKeyPair::generate();
    let first = signed_descriptor_for(&kp, 1, 1_700_001_000);
    let first_update = signed_descriptor_for(&kp, 2, 1_700_001_000);
    let second = signed_descriptor(1, 1_700_001_000);

    assert!(store.upsert_verified(first, 1_700_000_100).unwrap());
    assert!(store.upsert_verified(first_update, 1_700_000_100).unwrap());

    let err = store.upsert_verified(second, 1_700_000_100).unwrap_err();
    assert!(matches!(err, PeerStoreError::CapacityExceeded { .. }));
    assert_eq!(store.len(), 1);
    assert_eq!(store.status(1_700_000_100).runtime.capacity_rejected, 1);
}

#[test]
fn test_capability_query_returns_only_valid_matching_peers() {
    let store = PeerStore::new();
    let matching = signed_descriptor(1, 1_700_001_000);
    let mut non_matching = signed_descriptor(1, 1_700_001_000);
    let kp = IdentityKeyPair::generate();
    non_matching.descriptor.node_id = kp.public_key_bytes();
    non_matching.descriptor.capabilities = vec![NodeCapability::EncryptedStorage];
    non_matching = SignedNodeDescriptor::sign(non_matching.descriptor, &kp).unwrap();

    store.upsert_verified(matching, 1_700_000_100).unwrap();
    store.upsert_verified(non_matching, 1_700_000_100).unwrap();

    let peers = store.peers_with_capability(NodeCapability::ChatRelay, 1_700_000_100);
    assert_eq!(peers.len(), 1);
}

#[test]
fn test_verified_receipt_peers_do_not_imply_network_diverse_path() {
    // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Preserve
    // valid terminal receipt evidence while refusing to advertise a live
    // App path when both eligible hops share one endpoint network identity.
    let now = 1_700_000_100;
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut middle = signed_descriptor_for(&middle_identity, 7, now + 4_000);
    middle.descriptor.public_endpoint = Some("https://collocated.example:8422".to_string());
    middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_identity).unwrap();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://collocated.example:9422".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();

    let store = PeerStore::new();
    store.upsert_verified(middle.clone(), now).unwrap();
    store.upsert_verified(terminal.clone(), now).unwrap();
    assert!(store.record_verified_client_onion_route_delivery(&middle, &terminal, now + 1,));

    let quality = store.status(now + 2).blind_relay_quality;
    assert_eq!(quality.delivery_receipt_capable_peers, 2);
    assert!(!quality.authenticated_delivery_path_ready);
    assert_eq!(
        quality.authenticated_delivery_path_reason,
        "no_network_diverse_receipt_path"
    );
    assert!(!quality.real_relay_ready);
}

#[test]
fn test_verified_synthetic_probe_path_evidence_is_all_or_nothing() {
    let now = 1_700_000_100;
    let first_identity = IdentityKeyPair::generate();
    let second_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut first = signed_descriptor_for(&first_identity, 7, now + 4_000);
    first.descriptor.public_endpoint = Some("https://first.example".to_string());
    first.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    first = SignedNodeDescriptor::sign(first.descriptor, &first_identity).unwrap();

    let mut second = signed_descriptor_for(&second_identity, 7, now + 4_000);
    second.descriptor.public_endpoint = Some("https://second-a.example".to_string());
    second.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    second = SignedNodeDescriptor::sign(second.descriptor, &second_identity).unwrap();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://terminal.example".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();

    let first_node_id = first.node_id();
    let second_node_id = second.node_id();
    let terminal_node_id = terminal.node_id();
    let store = PeerStore::new();
    for descriptor in [first.clone(), second.clone(), terminal.clone()] {
        store.upsert_verified(descriptor, now).unwrap();
    }

    let mut rotated_second_body = second.descriptor.clone();
    rotated_second_body.sequence = 8;
    rotated_second_body.issued_at = now + 20;
    rotated_second_body.expires_at = now + 4_020;
    rotated_second_body.public_endpoint = Some("https://second-b.example".to_string());
    let rotated_second = SignedNodeDescriptor::sign(rotated_second_body, &second_identity).unwrap();
    store
        .upsert_verified(rotated_second.clone(), now + 20)
        .unwrap();

    // [ATOMIC-MULTIHOP-PROOF-EVIDENCE 2026-08-11 by Codex] A rotation of
    // any hop rejects the whole proof. No unchanged hop receives partial
    // health/capability credit and no success enters admission history.
    assert!(!store.record_verified_three_hop_probe_delivery(
        &first,
        &second,
        &terminal,
        now + 21,
        2,
        1,
    ));
    let rejected = store.status(now + 21);
    assert_eq!(rejected.three_hop_path_proof_history.attempted, 0);
    assert_eq!(
        rejected.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    for node_id in [first_node_id, second_node_id, terminal_node_id] {
        assert!(!store.is_routeable_now(&node_id, now + 21));
    }
    assert!(!store.take_peer_cache_dirty());

    assert!(store.record_verified_three_hop_probe_delivery(
        &first,
        &rotated_second,
        &terminal,
        now + 22,
        2,
        1,
    ));
    let accepted = store.status(now + 22);
    assert_eq!(accepted.three_hop_path_proof_history.attempted, 1);
    assert_eq!(accepted.three_hop_path_proof_history.succeeded, 1);
    assert_eq!(
        accepted
            .three_hop_path_proof_history
            .message_delivery_successes,
        1
    );
    assert_eq!(
        accepted.blind_relay_quality.delivery_receipt_capable_peers,
        3
    );
    for node_id in [first_node_id, second_node_id, terminal_node_id] {
        assert!(store.is_routeable_now(&node_id, now + 22));
    }
    assert!(store.take_peer_cache_dirty());
}

#[test]
fn test_legacy_control_probe_binds_both_surfaces_without_receipt_upgrade() {
    let now = 1_700_000_100;
    let middle_identity = IdentityKeyPair::generate();
    let terminal_identity = IdentityKeyPair::generate();

    let mut middle = signed_descriptor_for(&middle_identity, 7, now + 4_000);
    middle.descriptor.public_endpoint = Some("https://middle.example".to_string());
    middle.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle = SignedNodeDescriptor::sign(middle.descriptor, &middle_identity).unwrap();

    let mut terminal = signed_descriptor_for(&terminal_identity, 7, now + 4_000);
    terminal.descriptor.public_endpoint = Some("https://terminal-a.example".to_string());
    terminal.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    terminal = SignedNodeDescriptor::sign(terminal.descriptor, &terminal_identity).unwrap();

    let middle_node_id = middle.node_id();
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

    // [LEGACY-CONTROL-PROOF-SURFACE-BINDING 2026-08-11 by Codex] A stale
    // terminal invalidates the full compatibility proof before any middle
    // health or proof-history success is published.
    assert!(!store.record_verified_two_hop_control_probe(&middle, &terminal, now + 21, 2, 1,));
    let rejected = store.status(now + 21);
    assert_eq!(rejected.two_hop_path_proof_history.attempted, 0);
    assert_eq!(
        rejected.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    assert!(!store.is_routeable_now(&middle_node_id, now + 21));
    assert!(!store.take_peer_cache_dirty());

    assert!(store.record_verified_two_hop_control_probe(
        &middle,
        &rotated_terminal,
        now + 22,
        2,
        1,
    ));
    let accepted = store.status(now + 22);
    assert_eq!(accepted.two_hop_path_proof_history.succeeded, 1);
    assert_eq!(
        accepted
            .two_hop_path_proof_history
            .latest_reason_bucket
            .as_deref(),
        Some("legacy_control_forwarded")
    );
    assert_eq!(
        accepted.blind_relay_quality.delivery_receipt_capable_peers,
        0
    );
    assert!(store.is_routeable_now(&middle_node_id, now + 22));
    assert!(!store.is_routeable_now(&terminal_node_id, now + 22));
    assert!(store.take_peer_cache_dirty());
}

#[test]
fn test_peer_health_reason_vocabularies_are_closed_and_protocol_complete() {
    for reason in [
        "http_100",
        "http_599",
        "onion_delivery_http_425",
        "peer_relay_http_502",
        "ack_response_too_large",
        "onion_ack_response_json_decode_failed",
        "peer_relay_ack_response_body_read_failed",
        "blind_relay_probe_timeout",
        "two_hop_onion_delivery_probe_connect",
        "three_hop_onion_delivery_probe_http_503",
        "onion_delivery_request_decode",
        "peer_relay_request_unknown",
        "peer_relay_receipt_signature_invalid",
    ] {
        assert!(is_route_failure_reason(reason), "rejected {reason}");
    }

    for reason in [
        "http_099",
        "http_600",
        "http_502_private",
        "peer_relay_http_502 endpoint=private",
        "ack_peer_body",
        "unknown_phase_timeout",
        "peer_relay_request_private_detail",
        "PEER_RELAY_HTTP_502",
        "",
    ] {
        assert!(!is_route_failure_reason(reason), "admitted {reason}");
    }

    assert!(is_blind_relay_rejection_reason("backpressure"));
    assert!(is_blind_relay_rejection_reason(
        "onion_terminal_delivery_failed"
    ));
    assert!(is_peer_relay_rejection_reason("duplicate_route"));
    assert!(is_quarantine_reason("failure_threshold"));
    assert!(!is_quarantine_reason("failure_threshold peer=private"));
}

#[test]
fn test_same_sequence_descriptor_conflict_is_rejected() {
    let now = 1_700_000_100;
    let peer_kp = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&peer_kp, 7, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://route-a.example".to_string());
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &peer_kp).unwrap();

    let store = PeerStore::new();
    store.upsert_verified(descriptor.clone(), now).unwrap();
    let mut conflicting_body = descriptor.descriptor;
    conflicting_body.public_endpoint = Some("https://route-b.example".to_string());
    let conflicting = SignedNodeDescriptor::sign(conflicting_body, &peer_kp).unwrap();

    assert!(matches!(
        store.upsert_verified(conflicting, now + 1),
        Err(PeerStoreError::VerificationFailed)
    ));
    assert_eq!(store.len(), 1);
}

#[test]
fn test_network_story_reports_onion_ready_without_endpoint_or_full_node_id() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, true, true, 2);
    store.record_gossip_round(now + 20, 2, 2, 1, None);

    let middle_kp = IdentityKeyPair::generate();
    let relay_kp = IdentityKeyPair::generate();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint = Some("https://story-middle.example".to_string());
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle_descriptor.node_id();

    let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
    relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    relay_descriptor.descriptor.public_endpoint = Some("https://story-relay.example".to_string());
    relay_descriptor = SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
    let relay_node_id = relay_descriptor.node_id();

    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
        .unwrap();
    store.record_route_forward_success(&middle_node_id, now + 25);
    store.record_route_forward_success(&relay_node_id, now + 26);

    let story = store.status(now + 30).network_story;

    assert_eq!(story.status, "onion_ready");
    assert!(story.chat_single_hop_ready);
    assert!(story.chat_two_hop_onion_ready);
    assert_eq!(story.valid_nodes, 2);
    assert_eq!(story.routeable_chat_relays, 1);
    assert_eq!(story.routeable_onion_middle_hops, 1);
    assert!(story.relay_foundation_ready);
    assert!(story.restart_recovery_configured);

    let story_json = serde_json::to_string(&story).unwrap();
    assert!(!story_json.contains("story-middle.example"));
    assert!(!story_json.contains("story-relay.example"));
    assert!(!story_json.contains(&hex::encode(middle_node_id)));
    assert!(!story_json.contains(&hex::encode(relay_node_id)));
    assert!(!story_json.contains("encrypted_blob"));
    assert!(!story_json.contains("receiver_pubkey"));
}

#[test]
fn test_network_story_attention_overrides_peer_view_when_recovery_is_missing() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, false, false, 0);

    let first_kp = IdentityKeyPair::generate();
    let second_kp = IdentityKeyPair::generate();

    let mut first_descriptor = signed_descriptor_for(&first_kp, 1, now + 2_000);
    first_descriptor.descriptor.public_endpoint =
        Some("https://recovery-missing-a.example".to_string());
    first_descriptor = SignedNodeDescriptor::sign(first_descriptor.descriptor, &first_kp)
        .expect("descriptor should sign");
    let first_node_id = first_descriptor.node_id();

    let mut second_descriptor = signed_descriptor_for(&second_kp, 1, now + 2_000);
    second_descriptor.descriptor.public_endpoint =
        Some("https://recovery-missing-b.example".to_string());
    second_descriptor = SignedNodeDescriptor::sign(second_descriptor.descriptor, &second_kp)
        .expect("descriptor should sign");
    let second_node_id = second_descriptor.node_id();

    store
        .upsert_verified_from_source(first_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(second_descriptor, now, "gossip_announce")
        .unwrap();

    let story = store.status(now + 30).network_story;

    assert_eq!(story.status, "attention");
    assert_eq!(story.valid_nodes, 2);
    assert!(!story.relay_foundation_ready);
    assert!(!story.restart_recovery_configured);

    let story_json = serde_json::to_string(&story).unwrap();
    assert!(!story_json.contains("recovery-missing-a.example"));
    assert!(!story_json.contains("recovery-missing-b.example"));
    assert!(!story_json.contains(&hex::encode(first_node_id)));
    assert!(!story_json.contains(&hex::encode(second_node_id)));
    assert!(!story_json.contains("encrypted_blob"));
    assert!(!story_json.contains("receiver_pubkey"));
}

#[test]
fn test_cleanup_expired_degrades_and_retains_old_peers() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);

    store.upsert_verified(descriptor, 1_700_000_100).unwrap();
    assert_eq!(store.cleanup_expired(1_700_002_000), 1);
    assert_eq!(store.len(), 1);
    assert_eq!(store.snapshot(1_700_002_001).valid_peers, 0);
    assert_eq!(store.cleanup_expired(1_700_002_100), 0);

    let status = store.status(1_700_002_001);
    assert_eq!(status.runtime.expired_removed, 0);
    assert_eq!(status.runtime.expired_degraded, 1);
    assert_eq!(status.runtime.last_cleanup_at, Some(1_700_002_000));
    assert_eq!(status.peer_summary.expired_peers, 1);
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_expired"
            && event.outcome == "degraded"
            && event.source == "cleanup"
            && event.reason.as_deref() == Some("descriptor_expired_retained")
    }));
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "expired_peer_cleanup"
            && event.outcome == "accepted"
            && event.detail.contains("degraded=1")
            && event.detail.contains("removed=0")
    }));

    let public_snapshot =
        store.export_bootstrap_snapshot(1_700_002_002, 1_700_002_002, false, None);
    assert_eq!(public_snapshot.peers.len(), 0);
    let cache_snapshot = store.export_peer_cache_snapshot(1_700_002_002);
    assert_eq!(cache_snapshot.peers.len(), 1);
}

#[test]
fn untrusted_discovery_candidates_do_not_consume_live_routing_capacity() {
    let now = 1_700_000_100;
    let store = PeerStore::with_max_peers(1);
    store.enable_untrusted_discovery_candidate_mode();
    let candidate = signed_descriptor(1, now + 600);
    let candidate_id = candidate.node_id();

    let admitted = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: candidate,
        },
        now,
    );
    assert_eq!(admitted.candidates, 1);
    assert_eq!(admitted.inserted, 0);
    assert_eq!(store.len(), 0);
    assert!(store.get_valid(&candidate_id, now).is_none());
    assert_eq!(store.status(now).runtime.candidate_admitted, 1);

    assert!(store
        .upsert_verified(signed_descriptor(1, now + 600), now)
        .expect("independently anchored live import"));
    assert_eq!(store.len(), 1);
}

#[test]
fn untrusted_discovery_candidate_limits_conflicts_and_expiry_release_slots() {
    let now = 1_700_000_100;
    let store = PeerStore::new();
    store.enable_untrusted_discovery_candidate_mode();
    let identity = IdentityKeyPair::generate();
    let descriptor = signed_descriptor_for(&identity, 1, now + 1);

    let first = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: descriptor.clone(),
        },
        now,
    );
    assert_eq!(first.candidates, 1);
    let exact = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: descriptor.clone(),
        },
        now,
    );
    assert_eq!(exact.unchanged, 1);

    let mut conflicting_body = descriptor.descriptor.clone();
    conflicting_body.public_endpoint = Some("https://conflict.example".to_string());
    let conflict = SignedNodeDescriptor::sign(conflicting_body, &identity).unwrap();
    let conflicting = store.apply_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: conflict,
        },
        now,
    );
    assert_eq!(conflicting.rejected, 1);

    assert_eq!(store.cleanup_expired(now + 2), 1);
    let rolled_back_exact = store.apply_untrusted_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce { descriptor },
        now,
    );
    assert_eq!(rolled_back_exact.unchanged, 1);

    let overlong = signed_descriptor(1, now + UNTRUSTED_DISCOVERY_MAX_LIFETIME_SECS + 2);
    let rejected = store.apply_untrusted_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: overlong,
        },
        now,
    );
    assert_eq!(rejected.rejected, 1);
}

#[test]
fn untrusted_candidate_exhaustion_cannot_starve_verified_live_capacity() {
    let now = 1_700_000_100;
    let store = PeerStore::with_max_peers(1);
    store.enable_untrusted_discovery_candidate_mode();
    for _ in 0..UNTRUSTED_DISCOVERY_CANDIDATE_CAPACITY {
        let report = store.apply_untrusted_discovery_message(
            &NodeDiscoveryMessage::DescriptorAnnounce {
                descriptor: signed_descriptor(1, now + 600),
            },
            now,
        );
        assert_eq!(report.candidates, 1);
    }
    let saturated = store.apply_untrusted_discovery_message(
        &NodeDiscoveryMessage::DescriptorAnnounce {
            descriptor: signed_descriptor(1, now + 600),
        },
        now,
    );
    assert_eq!(saturated.rejected, 1);
    assert_eq!(store.len(), 0);
    assert!(store
        .upsert_verified(signed_descriptor(1, now + 600), now)
        .expect("candidate exhaustion cannot consume live capacity"));
    assert_eq!(store.len(), 1);
}

#[test]
fn test_valid_public_descriptors_is_bounded_and_filters_private_peers() {
    let store = PeerStore::new();
    let public_a = signed_descriptor(1, 1_700_001_000);
    let public_b = signed_descriptor(2, 1_700_001_000);
    let private_key = IdentityKeyPair::generate();
    let mut private = signed_descriptor_for(&private_key, 1, 1_700_001_000);
    private.descriptor.policy.public_discovery = false;
    private = SignedNodeDescriptor::sign(private.descriptor, &private_key).unwrap();
    for descriptor in [public_a, public_b, private] {
        store.upsert_verified(descriptor, 1_700_000_100).unwrap();
    }

    assert!(store.valid_public_descriptors(1_700_000_100, 0).is_empty());
    let selected = store.valid_public_descriptors(1_700_000_100, 1);
    assert_eq!(selected.len(), 1);
    assert!(selected[0].descriptor.policy.public_discovery);
}

#[test]
fn test_valid_public_endpoint_identities_is_complete_and_side_effect_free(
) -> Result<(), Box<dyn std::error::Error>> {
    let store = PeerStore::new();
    let now = 1_700_000_100;

    let public_key = IdentityKeyPair::generate();
    let mut public = signed_descriptor_for(&public_key, 1, 1_700_001_000);
    public.descriptor.public_endpoint = Some("https://public.example".to_string());
    public = SignedNodeDescriptor::sign(public.descriptor, &public_key)?;
    let public_node_id = public.node_id();

    let private_key = IdentityKeyPair::generate();
    let mut private = signed_descriptor_for(&private_key, 1, 1_700_001_000);
    private.descriptor.public_endpoint = Some("https://private.example".to_string());
    private.descriptor.policy.public_discovery = false;
    private = SignedNodeDescriptor::sign(private.descriptor, &private_key)?;

    let no_endpoint = signed_descriptor(1, 1_700_001_000);
    for descriptor in [public, private, no_endpoint] {
        store.upsert_verified(descriptor, now)?;
    }

    let identities = store.valid_public_endpoint_identities(now);

    assert_eq!(
        identities,
        vec![(public_node_id, "https://public.example".to_string())]
    );
    Ok(())
}

#[test]
fn test_heartbeat_signed_peer_records_export_only_verifiable_live_records() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let valid = signed_descriptor(1, 1_700_001_000);
    let valid_node_id = valid.node_id();
    let expired = signed_descriptor(1, 1_699_999_999);
    let mut tampered = signed_descriptor(1, 1_700_001_000);
    tampered.signature[0] ^= 0x01;

    store.upsert_verified(valid, now).unwrap();
    store.peers.write().insert(expired.node_id(), expired);
    store.peers.write().insert(tampered.node_id(), tampered);

    let signed_records = store.export_signed_peer_records_for_heartbeat(now, Some(8));

    assert_eq!(signed_records.total_retained_records, 3);
    assert_eq!(signed_records.valid_signed_records, 1);
    assert_eq!(signed_records.exported_signed_records, 1);
    assert_eq!(signed_records.records.generated_at, now);
    assert_eq!(signed_records.records.peers.len(), 1);
    assert_eq!(signed_records.records.peers[0].node_id(), valid_node_id);
    assert!(signed_records.records.peers[0].verify_at(now).is_ok());
    assert!(signed_records
        .verification_rule
        .contains("SignedNodeDescriptor::verify_at"));
    assert!(signed_records
        .privacy_boundary
        .contains("signed node discovery descriptors only"));
    assert!(store.recent_audit_events().iter().any(|event| {
        event.action == "heartbeat_signed_peer_records_export"
            && event.outcome == "accepted"
            && event.detail.contains("retained=3")
            && event.detail.contains("valid=1")
            && event.detail.contains("exported=1")
    }));
}

#[test]
fn test_apply_descriptor_announce_imports_peer() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);
    let message = NodeDiscoveryMessage::DescriptorAnnounce { descriptor };

    let report = store.apply_discovery_message(&message, 1_700_000_100);

    assert_eq!(report.inserted, 1);
    assert_eq!(store.len(), 1);
    let status = store.status(1_700_000_100);
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_inserted"
            && event.outcome == "accepted"
            && event.source == "gossip_announce"
            && event.sequence == Some(1)
            && event.reason.is_none()
    }));
}

#[test]
fn test_runtime_rejection_counters_are_recorded() {
    let store = PeerStore::new();

    store.record_policy_rejected(1_700_000_300, "allow_list_enabled=true");
    store.record_rate_limited(1_700_000_301, "global_limit_per_minute=1");
    store.mark_gossip_at(1_700_000_333);

    let status = store.status(1_700_000_400);
    assert_eq!(status.runtime.policy_rejected, 1);
    assert_eq!(status.runtime.rate_limited, 1);
    assert_eq!(status.runtime.last_gossip_at, Some(1_700_000_333));
    assert_eq!(status.recent_audit_events.len(), 2);
    assert_eq!(
        status.recent_audit_events[0].action,
        "gossip_policy_rejected"
    );
    assert_eq!(status.recent_audit_events[1].action, "gossip_rate_limited");
}

#[test]
fn startup_self_check_status_is_recorded_without_config_values() {
    let store = PeerStore::new();

    store.record_startup_self_check(
        1_700_000_350,
        "warning",
        "missing=peer_cache_path,seed_endpoints,public_endpoint",
    );

    let status = store.status(1_700_000_400);
    assert_eq!(
        status.bootstrap.startup_self_check_status.as_deref(),
        Some("warning")
    );
    assert_eq!(
        status.bootstrap.startup_self_check_detail.as_deref(),
        Some("missing=peer_cache_path,seed_endpoints,public_endpoint")
    );
    assert_eq!(status.bootstrap.startup_self_check_at, Some(1_700_000_350));
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "startup_self_check"
            && event.outcome == "warning"
            && !event.detail.contains("https://")
            && !event.detail.contains("/root/")
    }));
}

#[test]
fn test_audit_log_is_bounded() {
    let store = PeerStore::new();

    for i in 0..70 {
        store.record_audit_event(1_700_000_000 + i, "snapshot_export", "accepted", "test");
    }

    let events = store.recent_audit_events();
    assert_eq!(events.len(), MAX_AUDIT_EVENTS);
    assert_eq!(events[0].at, 1_700_000_006);
    assert_eq!(events[MAX_AUDIT_EVENTS - 1].at, 1_700_000_069);
}

#[test]
fn test_delivery_receipt_capability_is_verified_peer_and_freshness_bounded() {
    let now = 1_700_000_000;
    let identity = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&identity, 1, now + 4_000);
    descriptor.descriptor.public_endpoint = Some("https://relay.example".to_string());
    descriptor.descriptor.capabilities =
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
    let node_id = descriptor.node_id();
    let store = PeerStore::new();
    store.upsert_verified(descriptor, now).unwrap();
    store.record_route_forward_success(&node_id, now + 1);
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 2);
    assert!(store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 3));

    let candidates = store.delivery_receipt_route_candidates_with_capability_excluding(
        NodeCapability::ChatRelay,
        now + 3,
        4,
        &[],
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        store
            .status(now + 3)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        1
    );

    store.record_route_forward_failure(&node_id, now + 4, "request_failed");
    assert!(store
        .delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            now + 5,
            4,
            &[],
        )
        .is_empty());

    let stale_at = now + 2 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1;
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, stale_at));
    assert!(store
        .delivery_receipt_route_candidates_with_capability_excluding(
            NodeCapability::ChatRelay,
            stale_at,
            4,
            &[],
        )
        .is_empty());
    assert_eq!(
        store
            .status(stale_at)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        0
    );
}

#[test]
fn test_delivery_receipt_capability_status_excludes_expired_peer() {
    let now = 1_700_000_000;
    let identity = IdentityKeyPair::generate();
    let mut descriptor = signed_descriptor_for(&identity, 1, now + 10);
    descriptor.descriptor.public_endpoint = Some("https://relay.example".to_string());
    descriptor.descriptor.capabilities =
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
    let node_id = descriptor.node_id();
    let store = PeerStore::new();
    store.upsert_verified(descriptor, now).unwrap();
    store.record_purpose_bound_delivery_receipt_capability(&node_id, now + 1);

    assert_eq!(
        store
            .status(now + 2)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        1
    );
    assert!(!store.has_fresh_purpose_bound_delivery_receipt_capability(&node_id, now + 11));
    assert_eq!(
        store
            .status(now + 11)
            .blind_relay_quality
            .delivery_receipt_capable_peers,
        0
    );
}

#[test]
fn test_delivery_witness_status_counts_only_accepted_signed_outcomes() {
    let store = PeerStore::new();
    let now = 1_700_000_000;
    let status = store.record_client_delivery_witness_round(
        now,
        7,
        true,
        2,
        PeerStoreVerifiedDeliveryWitnessRound {
            configured: 3,
            attempted: 3,
            verified: 3,
            advanced: 1,
            idempotent: 1,
            stale: 1,
            ..PeerStoreVerifiedDeliveryWitnessRound::default()
        },
    );
    assert_eq!(status, "rollback_detected");
    let bootstrap = store.status(now).bootstrap;
    assert_eq!(
        bootstrap.last_client_delivery_witness_status.as_deref(),
        Some("rollback_detected")
    );
    assert_eq!(bootstrap.last_client_delivery_witness_generation, 7);
    assert!(bootstrap.last_client_delivery_witness_required);
    assert_eq!(bootstrap.last_client_delivery_witness_minimum_verified, 2);
    assert_eq!(bootstrap.last_client_delivery_witness_configured, 3);
    assert_eq!(bootstrap.last_client_delivery_witness_verified, 3);
    assert_eq!(bootstrap.last_client_delivery_witness_stale, 1);
}

#[test]
fn test_external_witness_gate_clears_only_client_delivery_evidence() {
    let store = PeerStore::new();
    let now = 1_700_000_000;
    store.record_verified_client_onion_delivery(now);
    store.record_blind_relay_terminal(now + 1, 0, 32);
    assert_eq!(
        store
            .status(now + 2)
            .runtime
            .blind_relay
            .verified_client_onion_deliveries,
        1
    );

    store.clear_restored_verified_client_delivery_evidence(now + 3, "external_witness_rollback");
    let status = store.status(now + 4);
    assert_eq!(
        status.runtime.blind_relay.verified_client_onion_deliveries,
        0
    );
    assert_eq!(status.runtime.blind_relay.terminal, 1);
    assert_eq!(
        status
            .bootstrap
            .last_client_delivery_cache_status
            .as_deref(),
        Some("rejected")
    );
    assert!(store.take_client_delivery_cache_dirty());
}

#[test]
fn test_verified_client_delivery_receipt_drives_real_relay_readiness_only_while_fresh() {
    let store = PeerStore::new();
    let now = 1_700_000_000;
    store.record_verified_client_onion_delivery(now);

    let unproven_mesh = store.status(now + 1).blind_relay_quality;
    assert!(!unproven_mesh.real_relay_ready);
    assert_eq!(unproven_mesh.delivery_receipt_capable_peers, 0);
    assert_eq!(unproven_mesh.status, "observing");
    assert_eq!(
        unproven_mesh.readiness_reason,
        "verified_client_onion_delivery_peer_revalidation_required"
    );

    for (offset, endpoint, capability) in [
        (
            1,
            "https://middle-receipt.example",
            NodeCapability::OnionMiddle,
        ),
        (
            2,
            "https://terminal-receipt.example",
            NodeCapability::ChatRelay,
        ),
    ] {
        let identity = IdentityKeyPair::generate();
        let mut descriptor = signed_descriptor_for(&identity, offset, now + 4_000);
        descriptor.descriptor.public_endpoint = Some(endpoint.to_string());
        descriptor.descriptor.capabilities = vec![capability];
        descriptor = SignedNodeDescriptor::sign(descriptor.descriptor, &identity).unwrap();
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
        store.record_purpose_bound_delivery_receipt_capability(&node_id, now);
    }

    let fresh = store.status(now + 10).blind_relay_quality;
    assert!(fresh.real_relay_ready);
    assert_eq!(fresh.verified_client_onion_deliveries, 1);
    assert_eq!(
        fresh.last_verified_client_onion_delivery_age_seconds,
        Some(10)
    );
    assert_eq!(
        fresh.evidence_mode,
        "verified_client_onion_delivery_receipt"
    );
    assert_eq!(fresh.proof_scope, "client_message_delivery");
    assert_eq!(
        fresh.readiness_reason,
        "verified_client_onion_delivery_receipt_ready"
    );

    let stale = store
        .status(now + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        .blind_relay_quality;
    assert!(!stale.real_relay_ready);
    assert_eq!(stale.status, "stale");
    assert_eq!(
        stale.readiness_reason,
        "verified_client_onion_delivery_receipt_stale"
    );
}

#[test]
fn test_peer_summary_tracks_source_ttl_health_and_capabilities() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    let healthy = signed_descriptor(1, now + 1_000);
    let stale = signed_descriptor(1, now + 120);

    store
        .upsert_verified_from_source(healthy.clone(), now, "cache")
        .unwrap();
    store
        .upsert_verified_from_source(stale.clone(), now, "gossip_snapshot")
        .unwrap();
    store
        .upsert_verified_from_source(healthy, now + 20, "gossip_announce")
        .unwrap();

    let status = store.status(now + 30);

    assert_eq!(status.peer_summary.total_peers, 2);
    assert_eq!(status.peer_summary.valid_peers, 2);
    assert_eq!(status.peer_summary.healthy_peers, 1);
    assert_eq!(status.peer_summary.stale_peers, 1);
    assert_eq!(status.peer_summary.expired_peers, 0);
    assert_eq!(status.peer_summary.chat_relay_peers, 2);
    assert_eq!(status.peer_summary.privacy_relay_peers, 2);
    assert_eq!(
        status.peer_summary.source_counts.get("gossip_announce"),
        Some(&1)
    );
    assert_eq!(
        status.peer_summary.source_counts.get("gossip_snapshot"),
        Some(&1)
    );
    assert!(status
        .peer_summary
        .peers
        .iter()
        .any(|peer| peer.health == "stale" && peer.ttl_remaining_seconds == Some(90)));
    assert!(status
        .peer_summary
        .peers
        .iter()
        .all(|peer| peer.capabilities.contains(&"chat_relay".to_string())));
}

// [NODE-TLS-BINDING 2026-10-10 by Claude] The production store publishes the
// identity-bound TLS directory as soon as a TLS-advertising peer is admitted,
// so the very next outbound URL to it is pinned HTTPS, and a store that is not
// enabled publishes nothing. The endpoint is unique to this test.
#[test]
fn test_identity_tls_directory_follows_admission() {
    use aeronyx_core::protocol::discovery::NodeProtocolFeature;

    const ENDPOINT: &str = "http://34.117.201.77:8422";
    let kp = IdentityKeyPair::generate();
    let mut descriptor = NodeDescriptor::new(
        kp.public_key_bytes(),
        1,
        1_700_000_000,
        1_700_001_000,
        "test",
    )
    .with_protocol_features([NodeProtocolFeature::IdentityBoundTlsV1]);
    descriptor.public_endpoint = Some(ENDPOINT.to_string());
    descriptor.capabilities = vec![NodeCapability::PrivacyRelay, NodeCapability::ChatRelay];
    let signed = SignedNodeDescriptor::sign(descriptor, &kp).unwrap();

    let silent = PeerStore::new();
    silent
        .upsert_verified(signed.clone(), 1_700_000_100)
        .unwrap();
    assert_eq!(
        crate::api::peer_transport_url(ENDPOINT, "/x")
            .unwrap()
            .scheme(),
        "http",
        "a store that is not enabled must not publish"
    );

    let store = PeerStore::new();
    store.enable_identity_tls_directory();
    store.upsert_verified(signed, 1_700_000_100).unwrap();
    let url = crate::api::peer_transport_url(ENDPOINT, "/api/discovery/gossip").unwrap();
    assert_eq!(url.scheme(), "https");
    assert_eq!(
        crate::api::peer_tls::decode_peer_tls_host(url.host_str().unwrap()),
        Some((kp.public_key_bytes(), "34.117.201.77".parse().unwrap()))
    );

    crate::api::peer_tls::install_directory(Default::default());
}
