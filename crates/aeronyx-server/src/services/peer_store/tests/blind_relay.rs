// Split from crates/aeronyx-server/src/services/peer_store.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn test_blind_relay_runtime_stats_track_drop_reasons_without_payload_data() {
    let store = PeerStore::new();

    store.record_blind_relay_terminal(1_700_000_010, 2, 128);
    store.record_blind_relay_forwarded(1_700_000_011, 1);
    store.record_blind_relay_rejected(1_700_000_012, "backpressure");
    store.record_blind_relay_rejected(1_700_000_013, "invalid_signature");
    store.record_blind_relay_rejected(1_700_000_014, "ttl_exhausted");
    store.record_blind_relay_rejected(1_700_000_015, "no_route");
    store.record_blind_relay_rejected(1_700_000_016, "missing_endpoint");
    store.record_blind_relay_rejected(1_700_000_017, "http_502");
    store.record_blind_relay_rejected(1_700_000_018, "route_loop");
    store.record_blind_relay_rejected(1_700_000_019, "duplicate_route");
    store.record_blind_relay_rejected(1_700_000_020, "rate_limited");
    store.record_blind_relay_rejected(1_700_000_021, "quarantined");
    store.record_blind_relay_rejected(1_700_000_022, "blind_relay_request_timeout");
    store.record_blind_relay_rejected(1_700_000_023, "timestamp_expired");
    store.record_blind_relay_rejected(1_700_000_024, "timestamp_in_future");
    store.record_blind_relay_quarantine_started(1_700_000_025, "failure_threshold");

    let status = store.status(1_700_000_026);
    let stats = status.runtime.blind_relay;

    assert_eq!(stats.received, 15);
    assert_eq!(stats.terminal, 1);
    assert_eq!(stats.forwarded, 1);
    assert_eq!(stats.rejected, 13);
    assert_eq!(stats.backpressure_dropped, 1);
    assert_eq!(stats.invalid_signature, 1);
    assert_eq!(stats.ttl_exhausted, 1);
    assert_eq!(stats.no_route, 1);
    assert_eq!(stats.invalid_endpoint, 1);
    assert_eq!(stats.forward_failed, 2);
    assert_eq!(stats.loop_detected, 1);
    assert_eq!(stats.replay_dropped, 1);
    assert_eq!(stats.timestamp_rejected, 2);
    assert_eq!(stats.rate_limited, 1);
    assert_eq!(stats.quarantined, 1);
    assert_eq!(stats.quarantine_started, 1);
    assert_eq!(stats.last_event_at, Some(1_700_000_025));
    assert!(status
        .recent_audit_events
        .iter()
        .all(|event| !event.detail.contains("route_id")));
    assert!(status
        .recent_audit_events
        .iter()
        .all(|event| !event.detail.contains("encrypted_blob=")));
    assert!(status
        .recent_audit_events
        .iter()
        .all(|event| !event.detail.contains("encrypted_blob_bytes")));
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "blind_relay_terminal"
            && event.detail.contains("encrypted_blob_size_bucket=lte_4kb")
    }));
}

#[test]
fn test_blind_relay_quality_reports_ready_without_private_metadata() {
    let store = PeerStore::new();

    store.record_blind_relay_terminal(1_700_000_010, 2, 128);
    store.record_blind_relay_forwarded(1_700_000_011, 1);

    let quality = store.status(1_700_000_021).blind_relay_quality;

    assert_eq!(quality.status, "ready");
    assert!(quality.runtime_ready);
    assert!(quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(quality.accepted_relay_ready);
    assert!(!quality.synthetic_probe_ready);
    assert_eq!(quality.evidence_mode, "opaque_relay_acceptance");
    assert_eq!(quality.proof_scope, "relay_acceptance");
    assert_eq!(quality.readiness_reason, "opaque_relay_acceptance_observed");
    assert_eq!(quality.accepted_total, 2);
    assert_eq!(quality.accepted_percent, 100);
    assert_eq!(quality.last_accepted_at, Some(1_700_000_011));
    assert_eq!(quality.last_event_age_seconds, Some(10));
    assert_eq!(quality.last_accepted_age_seconds, Some(10));
    assert_eq!(quality.last_probe_age_seconds, None);
    assert!(!quality.detail.contains("https://"));
    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
    assert!(quality
        .privacy_boundary
        .contains("aggregate blind relay runtime counters only"));
}

#[test]
fn test_blind_relay_quality_does_not_stay_ready_after_stale_opaque_relay_evidence() {
    let store = PeerStore::new();

    store.record_blind_relay_terminal(1_700_000_010, 2, 128);
    store.record_blind_relay_forwarded(1_700_000_011, 1);

    let quality = store
        .status(1_700_000_011 + PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
        .blind_relay_quality;

    assert_eq!(quality.status, "stale");
    assert!(!quality.runtime_ready);
    assert!(!quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(!quality.accepted_relay_ready);
    assert!(!quality.synthetic_probe_ready);
    assert_eq!(quality.evidence_mode, "opaque_relay_acceptance");
    assert_eq!(quality.proof_scope, "relay_acceptance");
    assert_eq!(quality.readiness_reason, "opaque_relay_acceptance_stale");
    assert_eq!(quality.accepted_total, 2);
    assert_eq!(quality.last_accepted_at, Some(1_700_000_011));
    assert_eq!(
        quality.last_accepted_age_seconds,
        Some(PEER_ROUTEABILITY_STALE_AFTER_SECS + 1)
    );
    assert!(quality.next_action.contains("refresh relay readiness"));
    assert!(quality.detail.contains("stale_after_seconds=1800"));
    assert!(!quality.detail.contains("https://"));
    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
}

#[test]
fn test_blind_relay_quality_marks_timestamp_replay_protection_active() {
    let store = PeerStore::new();

    store.record_blind_relay_rejected(1_700_000_010, "timestamp_expired");

    let status = store.status(1_700_000_020);
    let stats = status.runtime.blind_relay;
    let quality = status.blind_relay_quality;

    assert_eq!(stats.timestamp_rejected, 1);
    assert_eq!(quality.status, "protecting");
    assert!(!quality.runtime_ready);
    assert!(!quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(!quality.accepted_relay_ready);
    assert_eq!(quality.evidence_mode, "opaque_relay_attempted");
    assert_eq!(quality.readiness_reason, "protection_active");
    assert_eq!(quality.timestamp_rejected, 1);
    assert!(quality.protection_active);
    assert_eq!(quality.last_event_age_seconds, Some(10));
    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("endpoint"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
}

#[test]
fn test_blind_relay_quality_surfaces_transport_attention_without_endpoint_data() {
    let store = PeerStore::new();

    store.record_blind_relay_forwarded(1_700_000_010, 1);
    store.record_blind_relay_retry_attempt(1_700_000_011, "blind_relay_request_timeout");
    store.record_blind_relay_retry_exhausted(1_700_000_012, 2, "blind_relay_request_timeout");
    store.record_blind_relay_rejected(1_700_000_013, "blind_relay_request_timeout");

    let quality = store.status(1_700_000_018).blind_relay_quality;

    assert_eq!(quality.status, "attention");
    assert!(quality.runtime_ready);
    assert!(!quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(quality.accepted_relay_ready);
    assert_eq!(quality.evidence_mode, "opaque_relay_acceptance");
    assert_eq!(quality.proof_scope, "relay_acceptance");
    assert_eq!(quality.readiness_reason, "opaque_relay_transport_attention");
    assert_eq!(quality.forward_failed, 1);
    assert_eq!(quality.retry_exhausted, 1);
    assert_eq!(quality.last_event_age_seconds, Some(5));
    assert!(quality.next_action.contains("next-hop reachability"));
    assert!(!quality.detail.contains("https://"));
    assert!(!quality.detail.contains("endpoint"));
    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
}

#[test]
fn test_blind_relay_quality_recovers_after_stable_message_delivery_window() {
    let store = PeerStore::new();

    store.record_blind_relay_forwarded(1_700_000_010, 1);
    store.record_blind_relay_retry_attempt(1_700_000_011, "blind_relay_request_timeout");
    store.record_blind_relay_retry_exhausted(1_700_000_012, 2, "blind_relay_request_timeout");
    store.record_blind_relay_rejected(1_700_000_013, "blind_relay_request_timeout");

    for at in [1_700_000_020, 1_700_000_030, 1_700_000_040] {
        store.record_blind_relay_two_hop_probe_result_with_context(
            at,
            true,
            "onion_terminal_delivered",
            3,
            3,
            2,
            1,
        );
    }

    let status = store.status(1_700_000_050);
    let quality = status.blind_relay_quality;
    let history = status.two_hop_path_proof_history;

    assert_eq!(history.stability_status, "stable");
    assert!(history.stability_ready);
    assert!(history.recent_message_delivery_ready);
    assert!(!history.failure_streak_active);
    assert!(!history.failure_circuit_breaker_active);

    assert_eq!(quality.status, "ready");
    assert!(quality.runtime_ready);
    assert!(quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(quality.accepted_relay_ready);
    assert_eq!(quality.evidence_mode, "opaque_relay_acceptance");
    assert_eq!(quality.proof_scope, "relay_acceptance");
    assert_eq!(quality.readiness_reason, "opaque_relay_acceptance_observed");
    assert_eq!(quality.forward_failed, 1);
    assert_eq!(quality.retry_exhausted, 1);
    assert!(quality
        .detail
        .contains("transport_attention_recovered=true"));
    assert!(quality.detail.contains("proof_stability_status=stable"));
    assert!(quality
        .next_action
        .contains("accepted encrypted relay work"));
    assert!(!quality.detail.contains("https://"));
    assert!(!quality.detail.contains("endpoint"));
    assert!(!quality.detail.contains("route_id"));
    assert!(!quality.detail.contains("encrypted_blob"));
    assert!(!quality.detail.contains("payload"));
}

#[test]
fn test_blind_relay_probe_quality_does_not_inflate_real_traffic_counters() {
    let store = PeerStore::new();

    store.record_blind_relay_probe_result(1_700_000_010, true, "accepted");

    let status = store.status(1_700_000_020);
    let stats = status.runtime.blind_relay;
    let quality = status.blind_relay_quality;

    assert_eq!(stats.received, 0);
    assert_eq!(stats.terminal, 0);
    assert_eq!(stats.forwarded, 0);
    assert_eq!(stats.probe_attempted, 1);
    assert_eq!(stats.probe_succeeded, 1);
    assert_eq!(stats.probe_failed, 0);
    assert_eq!(stats.last_probe_at, Some(1_700_000_010));
    assert_eq!(quality.accepted_total, 0);
    assert!(quality.runtime_ready);
    assert!(quality.quality_ready);
    assert!(!quality.real_relay_ready);
    assert!(quality.synthetic_probe_ready);
    assert_eq!(quality.evidence_mode, "synthetic_probe");
    assert_eq!(quality.proof_scope, "single_hop_control_plane");
    assert_eq!(quality.readiness_reason, "synthetic_probe_ready");
    assert_eq!(quality.probe_attempted, 1);
    assert_eq!(quality.probe_succeeded, 1);
    assert_eq!(quality.probe_failed, 0);
    assert_eq!(quality.timestamp_rejected, 0);
    assert_eq!(quality.last_probe_age_seconds, Some(10));
    assert!(quality.detail.contains("last_probe_age_seconds=10"));
    assert!(quality.detail.contains("evidence_mode=synthetic_probe"));
    assert!(quality
        .detail
        .contains("readiness_reason=synthetic_probe_ready"));
    assert!(quality
        .next_action
        .contains("do not present it as App/user traffic"));

    let serialized = serde_json::to_string(&quality).unwrap();
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("payload_b64"));
    assert!(!serialized.contains("client_ip"));
    assert!(!serialized.contains("http://"));
    assert!(!serialized.contains("https://"));
}

#[test]
fn test_stability_requires_restart_recovery_for_relay_foundation() {
    let store = PeerStore::new();
    store.configure_bootstrap_status(true, false, true, 0);
    let now = 1_700_000_180;
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
        .unwrap();
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
        .unwrap();
    store.record_gossip_round(now - 60, 2, 2, 0, None);

    let status = store.status(now);

    assert_eq!(status.stability.health, "degraded");
    assert!(!status.stability.relay_foundation_ready);
    assert!(!status.stability.seed_recovery_configured);
    assert!(!status.stability.restart_recovery_configured);
    assert!(status.stability.restart_recovery_sources.is_empty());
    assert!(status.stability.next_action.contains("peer_cache_path"));
}
