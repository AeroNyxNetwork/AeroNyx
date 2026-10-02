// Split from crates/aeronyx-server/src/services/peer_store.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn test_two_hop_proof_cache_exports_only_newest_strict_fresh_window() {
    let now = 1_700_100_000;
    let store = PeerStore::new();
    for offset in 1..=10 {
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            3,
            2,
            1,
        );
    }
    store.record_blind_relay_two_hop_probe_result_with_context(
        now + 11,
        false,
        "not_allowlisted",
        3,
        3,
        2,
        1,
    );
    store.record_blind_relay_two_hop_probe_result_with_context(
        now + 100,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );
    store.record_blind_relay_two_hop_probe_result_with_context(
        now.saturating_sub(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1),
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );

    let events = store.export_two_hop_path_proof_cache_events(now + 20);

    assert_eq!(events.len(), TWO_HOP_PATH_PROOF_CACHE_MAX_ENTRIES);
    assert_eq!(events.first().map(|event| event.at), Some(now + 3));
    assert_eq!(events.last().map(|event| event.at), Some(now + 10));
    assert!(events.iter().all(|event| {
        event.reason_bucket == "onion_terminal_delivered"
            && event.evidence_mode == "synthetic_onion_message_delivery_probe"
            && event.proof_scope == "message_delivery"
    }));
    let json = serde_json::to_string(&events).unwrap();
    for forbidden in ["node_id", "endpoint", "route_id", "payload", "receiver"] {
        assert!(!json.contains(forbidden));
    }
}

#[test]
fn test_two_hop_proof_cache_rejects_inconsistent_semantics() {
    let now = 1_700_300_000;
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
    let store = PeerStore::new();
    store.upsert_verified(middle.clone(), now).unwrap();
    store.upsert_verified(terminal.clone(), now).unwrap();
    store.record_route_forward_success(&middle.node_id(), now + 1);
    store.record_route_forward_success(&terminal.node_id(), now + 1);

    let invalid = PeerStoreTwoHopPathProofEvent {
        at: now + 2,
        outcome: "accepted".to_string(),
        reason_bucket: "onion_terminal_delivered".to_string(),
        evidence_mode: "synthetic_two_hop_control_probe".to_string(),
        proof_scope: "control_plane".to_string(),
        path_shape: "entry_middle_terminal".to_string(),
        hop_count: 2,
        path_policy: TWO_HOP_PATH_POLICY_NETWORK_DIVERSE.to_string(),
        middle_candidate_bucket: "few".to_string(),
        terminal_candidate_bucket: "few".to_string(),
        ttl_shape: "entry_ttl_2_onward_ttl_1".to_string(),
    };

    let report = store.restore_two_hop_path_proof_cache_events(&[invalid], now + 3);
    assert_eq!(report.restored, 0);
    assert_eq!(report.rejected, 1);
    assert!(store
        .status(now + 3)
        .two_hop_path_proof_history
        .events
        .is_empty());
}

#[test]
fn test_two_hop_proof_cache_marks_empty_authentication_failure_rejected() {
    let store = PeerStore::new();
    let now = 1_700_400_000;

    let report = store.reject_two_hop_path_proof_cache_events(0, now, "signature_invalid");

    assert_eq!(report.total, 0);
    assert_eq!(report.restored, 0);
    assert_eq!(report.rejected, 0);
    let status = store.status(now);
    assert_eq!(
        status.bootstrap.last_two_hop_proof_cache_status.as_deref(),
        Some("rejected")
    );
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "two_hop_proof_cache_restore"
            && event.outcome == "rejected"
            && event.detail.contains("reason=signature_invalid")
    }));
}

#[test]
fn test_network_story_uses_fresh_two_hop_path_proof_as_onion_ready_evidence() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, true, true, 2);
    store.record_gossip_round(now + 20, 2, 2, 1, None);

    let middle_kp = IdentityKeyPair::generate();
    let relay_kp = IdentityKeyPair::generate();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 2_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint = Some("https://proof-middle.example".to_string());
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();

    let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 2_000);
    relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    relay_descriptor.descriptor.public_endpoint = Some("https://proof-relay.example".to_string());
    relay_descriptor = SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();

    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
        .unwrap();
    store.record_blind_relay_two_hop_probe_result_with_context(
        now + 25,
        true,
        "onion_terminal_delivered",
        1,
        1,
        2,
        1,
    );

    let story = store.status(now + 30).network_story;

    assert_eq!(story.status, "onion_ready");
    assert!(story.chat_two_hop_onion_ready);
    assert_eq!(story.routeable_chat_relays, 0);
    assert_eq!(story.routeable_onion_middle_hops, 0);
    assert!(story.detail.contains("two_hop_path_proof_recent=true"));
    let story_json = serde_json::to_string(&story).unwrap();
    assert!(!story_json.contains("proof-middle.example"));
    assert!(!story_json.contains("proof-relay.example"));
}

#[test]
fn test_network_story_does_not_use_stale_two_hop_path_proof_as_onion_ready_evidence() {
    let store = PeerStore::new();
    let now = 1_700_000_100;
    store.configure_bootstrap_status(true, true, true, 2);
    store.record_gossip_round(now + 20, 2, 2, 1, None);

    let middle_kp = IdentityKeyPair::generate();
    let relay_kp = IdentityKeyPair::generate();

    let mut middle_descriptor = signed_descriptor_for(&middle_kp, 1, now + 4_000);
    middle_descriptor.descriptor.capabilities = vec![NodeCapability::OnionMiddle];
    middle_descriptor.descriptor.public_endpoint =
        Some("https://stale-proof-middle.example".to_string());
    middle_descriptor =
        SignedNodeDescriptor::sign(middle_descriptor.descriptor, &middle_kp).unwrap();
    let middle_node_id = middle_descriptor.node_id();

    let mut relay_descriptor = signed_descriptor_for(&relay_kp, 1, now + 4_000);
    relay_descriptor.descriptor.capabilities = vec![NodeCapability::ChatRelay];
    relay_descriptor.descriptor.public_endpoint =
        Some("https://stale-proof-relay.example".to_string());
    relay_descriptor = SignedNodeDescriptor::sign(relay_descriptor.descriptor, &relay_kp).unwrap();
    let relay_node_id = relay_descriptor.node_id();

    store
        .upsert_verified_from_source(middle_descriptor, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(relay_descriptor, now, "gossip_announce")
        .unwrap();
    store.record_blind_relay_two_hop_probe_result_with_context(
        now + 25,
        true,
        "onion_terminal_delivered",
        1,
        1,
        2,
        1,
    );

    let stale_now = now + 25 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1;
    store.record_gossip_round(stale_now - 10, 2, 2, 1, None);
    let status = store.status(stale_now);
    let history = status.two_hop_path_proof_history;
    let story = status.network_story;

    assert_eq!(history.status, "stale");
    assert_eq!(history.freshness_bucket, "stale_success");
    assert!(!history.recent_success_ready);
    assert_eq!(story.status, "peer_view_ready");
    assert!(!story.chat_two_hop_onion_ready);
    assert!(story.detail.contains("two_hop_path_proof_recent=false"));

    let story_json = serde_json::to_string(&story).unwrap();
    assert!(!story_json.contains("stale-proof-middle.example"));
    assert!(!story_json.contains("stale-proof-relay.example"));
    assert!(!story_json.contains(&hex::encode(middle_node_id)));
    assert!(!story_json.contains(&hex::encode(relay_node_id)));
    assert!(!story_json.contains("encrypted_blob"));
    assert!(!story_json.contains("receiver_pubkey"));
}

#[test]
fn test_two_hop_blind_relay_probe_reports_path_proof_without_private_metadata() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_010,
        true,
        "accepted",
        4,
        3,
        2,
        1,
    );

    let status = store.status(1_700_000_025);
    let stats = status.runtime.blind_relay;
    let quality = status.blind_relay_quality;

    assert_eq!(stats.two_hop_probe_attempted, 1);
    assert_eq!(stats.two_hop_probe_succeeded, 1);
    assert_eq!(stats.two_hop_probe_failed, 0);
    assert_eq!(stats.last_two_hop_probe_at, Some(1_700_000_010));
    assert_eq!(quality.status, "ready");
    assert!(quality.runtime_ready);
    assert!(quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(quality.synthetic_probe_ready);
    assert!(quality.two_hop_probe_ready);
    assert_eq!(quality.evidence_mode, "synthetic_two_hop_control_probe");
    assert_eq!(quality.proof_scope, "control_plane");
    assert_eq!(
        quality.readiness_reason,
        "synthetic_two_hop_control_probe_ready"
    );
    assert!(quality.detail.contains("proof_scope=control_plane"));
    assert!(quality
        .next_action
        .contains("do not present it as App/user traffic"));
    assert_eq!(quality.last_two_hop_probe_age_seconds, Some(15));
    assert_eq!(status.two_hop_path_proof_history.attempted, 1);
    assert_eq!(status.two_hop_path_proof_history.succeeded, 1);
    assert_eq!(status.two_hop_path_proof_history.failed, 0);
    assert_eq!(status.two_hop_path_proof_history.status, "ready");
    assert_eq!(
        status.two_hop_path_proof_history.freshness_bucket,
        "fresh_success"
    );
    assert!(status.two_hop_path_proof_history.proof_ready);
    assert!(status.two_hop_path_proof_history.recent_success_ready);
    assert!(!status.two_hop_path_proof_history.message_delivery_ready);
    assert!(
        !status
            .two_hop_path_proof_history
            .recent_message_delivery_ready
    );
    assert_eq!(
        status.two_hop_path_proof_history.message_delivery_successes,
        0
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .latest_message_delivery_age_seconds,
        None
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .consecutive_message_delivery_successes,
        0
    );
    assert!(!status.two_hop_path_proof_history.failure_streak_active);
    assert_eq!(
        status.two_hop_path_proof_history.stale_after_seconds,
        PEER_ROUTEABILITY_STALE_AFTER_SECS
    );
    assert_eq!(status.two_hop_path_proof_history.success_percent, 100);
    assert_eq!(
        status.two_hop_path_proof_history.stability_window_size,
        TWO_HOP_PATH_PROOF_STABILITY_WINDOW_EVENTS
    );
    assert_eq!(
        status.two_hop_path_proof_history.stability_window_attempted,
        1
    );
    assert_eq!(
        status.two_hop_path_proof_history.stability_window_succeeded,
        1
    );
    assert_eq!(status.two_hop_path_proof_history.stability_window_failed, 0);
    assert_eq!(
        status.two_hop_path_proof_history.stability_success_percent,
        100
    );
    assert_eq!(
        status.two_hop_path_proof_history.stability_status,
        "warming_up"
    );
    assert!(!status.two_hop_path_proof_history.stability_ready);
    assert_eq!(
        status
            .two_hop_path_proof_history
            .failure_circuit_breaker_threshold,
        TWO_HOP_PATH_PROOF_FAILURE_CIRCUIT_BREAKER_THRESHOLD
    );
    assert!(
        !status
            .two_hop_path_proof_history
            .failure_circuit_breaker_active
    );
    assert_eq!(status.two_hop_path_proof_history.latest_age_bucket, "fresh");
    assert_eq!(
        status.two_hop_path_proof_history.latest_outcome.as_deref(),
        Some("accepted")
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .latest_reason_bucket
            .as_deref(),
        Some("accepted")
    );
    assert_eq!(
        status.two_hop_path_proof_history.latest_age_seconds,
        Some(15)
    );
    assert_eq!(
        status.two_hop_path_proof_history.latest_success_age_seconds,
        Some(15)
    );
    assert_eq!(
        status.two_hop_path_proof_history.latest_failure_age_seconds,
        None
    );
    assert_eq!(
        status.two_hop_path_proof_history.proof_scope,
        "control_plane"
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .proof_scope_counts
            .get("control_plane"),
        Some(&1)
    );
    assert_eq!(status.two_hop_path_proof_history.consecutive_successes, 1);
    assert_eq!(status.two_hop_path_proof_history.consecutive_failures, 0);
    assert_eq!(status.two_hop_path_proof_history.events.len(), 1);
    assert_eq!(
        status.two_hop_path_proof_history.events[0].path_shape,
        "entry_middle_terminal"
    );
    assert_eq!(status.two_hop_path_proof_history.events[0].hop_count, 2);
    assert_eq!(
        status.two_hop_path_proof_history.events[0].path_policy,
        TWO_HOP_PATH_POLICY_NETWORK_DIVERSE
    );
    assert_eq!(
        status.two_hop_path_proof_history.events[0].middle_candidate_bucket,
        "healthy"
    );
    assert_eq!(
        status.two_hop_path_proof_history.events[0].terminal_candidate_bucket,
        "few"
    );
    assert_eq!(
        status.two_hop_path_proof_history.events[0].ttl_shape,
        "entry_ttl_2_onward_ttl_1"
    );
    assert_eq!(
        status.two_hop_path_proof_history.events[0].evidence_mode,
        "synthetic_two_hop_control_probe"
    );
    assert_eq!(
        status.two_hop_path_proof_history.events[0].proof_scope,
        "control_plane"
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .path_shape_counts
            .get("entry_middle_terminal"),
        Some(&1)
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .candidate_pool_counts
            .get("forming"),
        Some(&1)
    );
    assert_eq!(
        status
            .two_hop_path_proof_history
            .ttl_shape_counts
            .get("entry_ttl_2_onward_ttl_1"),
        Some(&1)
    );
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "blind_relay_two_hop_probe"
            && event.outcome == "accepted"
            && event.detail.contains("reason_bucket=accepted")
            && event.detail.contains("middle_candidates=healthy")
            && event.detail.contains("terminal_candidates=few")
            && event.detail.contains("ttl_shape=entry_ttl_2_onward_ttl_1")
    }));
    let audit_detail = status
        .recent_audit_events
        .iter()
        .find(|event| event.action == "blind_relay_two_hop_probe")
        .map(|event| event.detail.as_str())
        .unwrap_or_default();
    assert!(!audit_detail.contains("endpoint"));
    assert!(!audit_detail.contains("node_id"));
    assert!(!audit_detail.contains("route_id"));
    assert!(!audit_detail.contains("payload"));

    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("endpoint"));
    assert!(!quality.detail.contains("node_id"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
    assert!(!quality.detail.contains("client_ip"));
    assert!(!status
        .two_hop_path_proof_history
        .privacy_boundary
        .contains("route_id"));
    assert!(status
        .two_hop_path_proof_history
        .privacy_boundary
        .contains("no node IDs"));
}

#[test]
fn test_blind_relay_quality_does_not_keep_ready_from_stale_two_hop_probe() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result(1_700_000_010, true, "accepted");

    let quality = store
        .status(1_700_000_010 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        .blind_relay_quality;

    assert_eq!(quality.status, "stale");
    assert!(!quality.runtime_ready);
    assert!(!quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(!quality.synthetic_probe_ready);
    assert!(!quality.two_hop_probe_ready);
    assert_eq!(quality.evidence_mode, "synthetic_two_hop_control_probe");
    assert_eq!(quality.proof_scope, "control_plane");
    assert_eq!(
        quality.readiness_reason,
        "synthetic_two_hop_control_probe_stale"
    );
    assert_eq!(
        quality.last_two_hop_probe_age_seconds,
        Some(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
    );
    assert_eq!(quality.last_accepted_age_seconds, None);
    assert!(quality.next_action.contains("synthetic path proof"));
    assert!(!quality.detail.contains("https://"));
    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
}

#[test]
fn test_two_hop_onion_delivery_probe_reports_message_delivery_scope() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_010,
        true,
        "onion_terminal_delivered",
        3,
        2,
        2,
        1,
    );

    let status = store.status(1_700_000_015);
    assert_eq!(
        status.blind_relay_quality.evidence_mode,
        "synthetic_onion_message_delivery_probe"
    );
    assert_eq!(status.blind_relay_quality.proof_scope, "message_delivery");
    assert_eq!(
        status.blind_relay_quality.readiness_reason,
        "synthetic_onion_message_delivery_probe_ready"
    );
    assert!(!status.blind_relay_quality.real_relay_ready);
    assert!(!status.blind_relay_quality.accepted_relay_ready);
    let history = status.two_hop_path_proof_history;

    assert_eq!(
        history.latest_reason_bucket.as_deref(),
        Some("onion_terminal_delivered")
    );
    assert_eq!(history.proof_scope, "message_delivery");
    assert_eq!(history.proof_scope_counts.get("message_delivery"), Some(&1));
    assert!(history.proof_ready);
    assert!(history.recent_success_ready);
    assert!(history.message_delivery_ready);
    assert!(history.recent_message_delivery_ready);
    assert_eq!(history.message_delivery_successes, 1);
    assert_eq!(
        history.message_delivery_evidence_mode,
        "synthetic_onion_message_delivery_probe"
    );
    assert_eq!(history.latest_message_delivery_age_seconds, Some(5));
    assert_eq!(history.consecutive_message_delivery_successes, 1);
    assert_eq!(history.stability_window_attempted, 1);
    assert_eq!(history.stability_window_succeeded, 1);
    assert_eq!(history.stability_success_percent, 100);
    assert_eq!(history.stability_status, "warming_up");
    assert!(!history.stability_ready);
    assert!(!history.failure_circuit_breaker_active);
    assert_eq!(history.latest_age_bucket, "fresh");
    assert_eq!(history.events.len(), 1);
    assert_eq!(
        history.events[0].evidence_mode,
        "synthetic_onion_message_delivery_probe"
    );
    assert_eq!(history.events[0].proof_scope, "message_delivery");
    assert!(!history.privacy_boundary.contains("route_id="));
    assert!(!history.privacy_boundary.contains("receiver="));
    assert!(!history.privacy_boundary.contains("encrypted_blob="));
}

#[test]
fn test_legacy_two_hop_ack_remains_control_plane_evidence() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_010,
        true,
        "legacy_control_forwarded",
        3,
        2,
        2,
        1,
    );

    // [TWO-HOP-PROBE-OUTCOME 2026-07-31 by Codex] A rolling-upgrade ACK
    // keeps route compatibility observable, but only a terminal-signed
    // receipt may produce message-delivery evidence.
    let history = store.status(1_700_000_015).two_hop_path_proof_history;
    assert_eq!(history.succeeded, 1);
    assert_eq!(history.proof_scope, "control_plane");
    assert_eq!(history.message_delivery_successes, 0);
    assert_eq!(history.message_delivery_evidence_mode, "none");
    assert_eq!(
        history.latest_reason_bucket.as_deref(),
        Some("legacy_control_forwarded")
    );
    assert_eq!(
        history.events[0].evidence_mode,
        "synthetic_two_hop_control_probe"
    );
    assert_eq!(history.events[0].proof_scope, "control_plane");
}

#[test]
fn test_two_hop_message_delivery_readiness_distinguishes_latest_from_recent() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_010,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );
    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_020,
        true,
        "accepted",
        3,
        3,
        2,
        1,
    );

    let history = store.status(1_700_000_030).two_hop_path_proof_history;

    assert!(history.proof_ready);
    assert!(history.recent_success_ready);
    assert!(!history.message_delivery_ready);
    assert!(history.recent_message_delivery_ready);
    assert_eq!(history.message_delivery_successes, 1);
    assert_eq!(history.latest_message_delivery_age_seconds, Some(20));
    assert_eq!(history.consecutive_successes, 2);
    assert_eq!(history.consecutive_message_delivery_successes, 0);
    assert_eq!(history.proof_scope, "control_plane");
    assert_eq!(history.proof_scope_counts.get("message_delivery"), Some(&1));
    assert_eq!(history.proof_scope_counts.get("control_plane"), Some(&1));

    let serialized = serde_json::to_string(&history).expect("history serializes");
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("endpoint="));
    assert!(!serialized.contains("https://"));
    assert!(!serialized.contains("http://"));
    assert!(!serialized.contains("receiver="));
    assert!(!serialized.contains("payload="));
    assert!(!serialized.contains("node_id"));
}

#[test]
fn test_two_hop_path_proof_history_buckets_failures_without_private_metadata() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result(1_700_000_010, false, "http_502");
    store.record_blind_relay_two_hop_probe_result(1_700_000_020, false, "endpoint://leak");
    store.record_blind_relay_two_hop_probe_result(
        1_700_000_030,
        false,
        "two_hop_blind_relay_probe_timeout",
    );
    store.record_blind_relay_two_hop_probe_result(1_700_000_040, true, "accepted");

    let status = store.status(1_700_000_055);
    let history = status.two_hop_path_proof_history;

    assert_eq!(history.window_size, MAX_TWO_HOP_PATH_PROOF_EVENTS);
    assert_eq!(history.retained_events, 4);
    assert_eq!(history.attempted, 4);
    assert_eq!(history.succeeded, 1);
    assert_eq!(history.message_delivery_successes, 0);
    assert_eq!(history.failed, 3);
    assert_eq!(history.status, "ready");
    assert_eq!(history.freshness_bucket, "fresh_success");
    assert!(history.proof_ready);
    assert!(history.recent_success_ready);
    assert!(!history.message_delivery_ready);
    assert!(!history.recent_message_delivery_ready);
    assert!(!history.failure_streak_active);
    assert_eq!(history.success_percent, 25);
    assert_eq!(history.stability_window_attempted, 4);
    assert_eq!(history.stability_window_succeeded, 1);
    assert_eq!(history.stability_window_failed, 3);
    assert_eq!(history.stability_success_percent, 25);
    assert_eq!(history.stability_status, "degraded");
    assert!(!history.stability_ready);
    assert!(!history.failure_circuit_breaker_active);
    assert_eq!(history.latest_age_bucket, "fresh");
    assert_eq!(history.latest_outcome.as_deref(), Some("accepted"));
    assert_eq!(history.latest_reason_bucket.as_deref(), Some("accepted"));
    assert_eq!(history.latest_age_seconds, Some(15));
    assert_eq!(history.latest_success_age_seconds, Some(15));
    assert_eq!(history.latest_failure_age_seconds, Some(25));
    assert_eq!(history.latest_message_delivery_age_seconds, None);
    assert_eq!(history.consecutive_successes, 1);
    assert_eq!(history.consecutive_failures, 0);
    assert_eq!(history.consecutive_message_delivery_successes, 0);
    assert_eq!(history.events[0].reason_bucket, "http_error");
    assert_eq!(history.events[1].reason_bucket, "unknown");
    assert_eq!(history.events[2].reason_bucket, "request_error");
    assert_eq!(history.events[3].reason_bucket, "accepted");
    assert_eq!(history.reason_bucket_counts.get("http_error"), Some(&1));
    assert_eq!(history.reason_bucket_counts.get("unknown"), Some(&1));
    assert_eq!(history.reason_bucket_counts.get("request_error"), Some(&1));
    assert_eq!(history.reason_bucket_counts.get("accepted"), Some(&1));
    assert_eq!(
        history.failure_reason_bucket_counts.get("http_error"),
        Some(&1)
    );
    assert_eq!(
        history.failure_reason_bucket_counts.get("unknown"),
        Some(&1)
    );
    assert_eq!(
        history.failure_reason_bucket_counts.get("request_error"),
        Some(&1)
    );
    assert_eq!(history.failure_reason_bucket_counts.get("accepted"), None);
    assert_eq!(
        history.path_shape_counts.get("entry_middle_terminal"),
        Some(&4)
    );
    assert_eq!(history.candidate_pool_counts.get("incomplete"), Some(&4));
    assert_eq!(
        history.ttl_shape_counts.get("entry_ttl_2_onward_ttl_1"),
        Some(&4)
    );

    let serialized = serde_json::to_string(&history).expect("history serializes");
    assert!(!serialized.contains("endpoint://leak"));
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("node_id"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("payload="));
    assert!(!serialized.contains("client_ip"));
}

#[test]
fn test_two_hop_path_proof_history_marks_attention_and_stale_states() {
    let failing_store = PeerStore::new();
    failing_store.record_blind_relay_two_hop_probe_result(1_700_000_010, false, "ack_rejected");

    let failing_history = failing_store
        .status(1_700_000_020)
        .two_hop_path_proof_history;
    assert_eq!(failing_history.status, "attention");
    assert_eq!(failing_history.freshness_bucket, "recent_failure");
    assert!(!failing_history.proof_ready);
    assert!(!failing_history.recent_success_ready);
    assert!(failing_history.failure_streak_active);
    assert_eq!(failing_history.latest_success_age_seconds, None);
    assert_eq!(failing_history.latest_failure_age_seconds, Some(10));
    assert_eq!(failing_history.consecutive_failures, 1);
    assert_eq!(failing_history.stability_window_attempted, 1);
    assert_eq!(failing_history.stability_window_succeeded, 0);
    assert_eq!(failing_history.stability_success_percent, 0);
    assert_eq!(failing_history.stability_status, "warming_up");
    assert!(!failing_history.stability_ready);
    assert!(!failing_history.failure_circuit_breaker_active);
    assert_eq!(failing_history.latest_age_bucket, "fresh");
    assert!(failing_history.next_action.contains("routeability"));

    let stale_store = PeerStore::new();
    stale_store.record_blind_relay_two_hop_probe_result(1_700_000_010, true, "accepted");
    let stale_history = stale_store
        .status(1_700_000_010 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        .two_hop_path_proof_history;
    assert_eq!(stale_history.status, "stale");
    assert_eq!(stale_history.freshness_bucket, "stale_success");
    assert!(!stale_history.proof_ready);
    assert!(!stale_history.recent_success_ready);
    assert!(!stale_history.failure_streak_active);
    assert_eq!(
        stale_history.latest_success_age_seconds,
        Some(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
    );
    assert_eq!(stale_history.latest_failure_age_seconds, None);
    assert_eq!(stale_history.consecutive_successes, 1);
    assert_eq!(stale_history.stability_status, "stale");
    assert!(!stale_history.stability_ready);
    assert_eq!(stale_history.latest_age_bucket, "stale");
    assert!(stale_history
        .next_action
        .contains("fresh two-hop path proof"));
}

#[test]
fn test_path_proof_history_fails_closed_until_future_evidence_is_observable() {
    let store = PeerStore::new();
    let now = 1_700_100_000;

    for offset in [10, 20, 30] {
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            3,
            2,
            1,
        );
        store.record_blind_relay_three_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            3,
            1,
            3,
            2,
        );
    }

    // [PATH-PROOF-CLOCK-GUARD 2026-08-03 by Codex] A wall-clock rollback
    // must not make future proof events appear zero seconds old. Both hop
    // histories share the same fail-closed summarizer.
    let guarded = store.status(now);
    for history in [
        &guarded.two_hop_path_proof_history,
        &guarded.three_hop_path_proof_history,
    ] {
        assert_eq!(history.retained_events, 3);
        assert_eq!(history.future_events_ignored, 3);
        assert_eq!(history.attempted, 0);
        assert_eq!(history.status, "attention");
        assert_eq!(history.freshness_bucket, "future_ignored");
        assert_eq!(history.stability_status, "clock_attention");
        assert!(!history.proof_ready);
        assert!(!history.recent_success_ready);
        assert!(!history.message_delivery_ready);
        assert!(!history.recent_message_delivery_ready);
        assert!(!history.stability_ready);
        assert_eq!(history.latest_age_seconds, None);
        assert_eq!(history.latest_message_delivery_age_seconds, None);
        assert!(history.next_action.contains("local clock"));
    }

    // Once wall time has safely passed every retained event, the same
    // signed aggregate evidence can mature naturally without manual reset.
    let recovered = store.status(now + 40);
    for history in [
        &recovered.two_hop_path_proof_history,
        &recovered.three_hop_path_proof_history,
    ] {
        assert_eq!(history.future_events_ignored, 0);
        assert_eq!(history.attempted, 3);
        assert_eq!(history.status, "ready");
        assert_eq!(history.stability_status, "stable");
        assert!(history.proof_ready);
        assert!(history.recent_message_delivery_ready);
        assert!(history.stability_ready);
        assert_eq!(history.latest_message_delivery_age_seconds, Some(10));
    }
}

#[test]
fn test_two_hop_path_proof_stability_window_marks_stable_and_circuit_breaker() {
    let stable_store = PeerStore::new();
    stable_store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_010,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );
    stable_store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_020,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );
    stable_store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_030,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );

    let stable_history = stable_store
        .status(1_700_000_040)
        .two_hop_path_proof_history;
    assert_eq!(stable_history.stability_window_attempted, 3);
    assert_eq!(stable_history.stability_window_succeeded, 3);
    assert_eq!(stable_history.stability_window_failed, 0);
    assert_eq!(stable_history.stability_success_percent, 100);
    assert_eq!(stable_history.stability_status, "stable");
    assert!(stable_history.stability_ready);
    assert!(!stable_history.failure_circuit_breaker_active);
    assert_eq!(stable_history.latest_age_bucket, "fresh");
    assert_eq!(stable_history.consecutive_message_delivery_successes, 3);

    let circuit_store = PeerStore::new();
    circuit_store.record_blind_relay_two_hop_probe_result(1_700_000_010, false, "http_502");
    circuit_store.record_blind_relay_two_hop_probe_result(1_700_000_020, false, "ack_rejected");
    circuit_store.record_blind_relay_two_hop_probe_result(1_700_000_030, false, "request_error");

    let circuit_history = circuit_store
        .status(1_700_000_040)
        .two_hop_path_proof_history;
    assert_eq!(circuit_history.status, "attention");
    assert_eq!(circuit_history.stability_window_attempted, 3);
    assert_eq!(circuit_history.stability_window_succeeded, 0);
    assert_eq!(circuit_history.stability_window_failed, 3);
    assert_eq!(circuit_history.stability_success_percent, 0);
    assert_eq!(circuit_history.consecutive_failures, 3);
    assert_eq!(circuit_history.stability_status, "circuit_breaker");
    assert!(!circuit_history.stability_ready);
    assert!(circuit_history.failure_circuit_breaker_active);
    assert_eq!(circuit_history.latest_age_bucket, "fresh");

    let expired_circuit_history = circuit_store
        .status(1_700_000_030 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        .two_hop_path_proof_history;
    assert_eq!(expired_circuit_history.stability_window_attempted, 0);
    assert!(!expired_circuit_history.failure_streak_active);
    assert!(!expired_circuit_history.failure_circuit_breaker_active);
    assert!(!expired_circuit_history.stability_ready);

    let serialized = serde_json::to_string(&circuit_history).expect("history serializes");
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("endpoint="));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("receiver="));
    assert!(!serialized.contains("client_ip"));
}

#[test]
fn test_two_hop_stability_prefers_message_delivery_over_control_plane_fallback() {
    let store = PeerStore::new();

    store.record_blind_relay_two_hop_probe_result(1_700_000_010, false, "onion_ack_rejected");
    store.record_blind_relay_two_hop_probe_result(1_700_000_020, true, "accepted");
    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_030,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );
    store.record_blind_relay_two_hop_probe_result(1_700_000_040, false, "onion_ack_rejected");
    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_050,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );
    store.record_blind_relay_two_hop_probe_result_with_context(
        1_700_000_060,
        true,
        "onion_terminal_delivered",
        3,
        3,
        2,
        1,
    );

    let history = store.status(1_700_000_070).two_hop_path_proof_history;

    assert_eq!(history.proof_scope, "message_delivery");
    assert_eq!(history.message_delivery_successes, 3);
    assert_eq!(history.stability_window_attempted, 3);
    assert_eq!(history.stability_window_succeeded, 3);
    assert_eq!(history.stability_window_failed, 0);
    assert_eq!(history.stability_success_percent, 100);
    assert_eq!(history.stability_status, "stable");
    assert!(history.stability_ready);
    assert!(history.recent_message_delivery_ready);
    assert!(!history.failure_circuit_breaker_active);

    let serialized = serde_json::to_string(&history).expect("history serializes");
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("endpoint="));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("receiver="));
    assert!(!serialized.contains("client_ip"));
}

#[test]
fn test_three_hop_runtime_proof_isolated_from_two_hop_readiness() {
    let store = PeerStore::new();
    let now = 1_800_300_000;
    store.record_blind_relay_two_hop_probe_result_with_context(
        now,
        true,
        "onion_terminal_delivered",
        3,
        2,
        2,
        1,
    );
    store.record_blind_relay_three_hop_probe_result_with_context(
        now + 1,
        false,
        "onion_receipt_unverified",
        3,
        1,
        3,
        2,
    );

    let status = store.status(now + 1);
    assert_eq!(status.two_hop_path_proof_history.attempted, 1);
    assert_eq!(status.two_hop_path_proof_history.succeeded, 1);
    assert!(status.two_hop_path_proof_history.proof_ready);
    assert_eq!(status.three_hop_path_proof_history.attempted, 1);
    assert_eq!(status.three_hop_path_proof_history.failed, 1);
    assert!(!status.three_hop_path_proof_history.proof_ready);
    assert_eq!(
        status
            .three_hop_path_proof_history
            .latest_reason_bucket
            .as_deref(),
        Some("onion_receipt_unverified")
    );
    assert_eq!(
        status
            .three_hop_path_proof_history
            .path_shape_counts
            .get("entry_middle_middle_terminal"),
        Some(&1)
    );
}
