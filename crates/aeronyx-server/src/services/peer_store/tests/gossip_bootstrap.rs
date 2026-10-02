// Split from crates/aeronyx-server/src/services/peer_store.rs `mod tests` for navigation.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn untrusted_snapshot_is_message_bounded_before_candidate_capacity() {
    let now = 1_700_000_100;
    let store = PeerStore::new();
    store.enable_untrusted_discovery_candidate_mode();
    let descriptors = (0..=UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE)
        .map(|_| signed_descriptor(1, now + 600))
        .collect::<Vec<_>>();
    let report = store.apply_discovery_message(
        &NodeDiscoveryMessage::SnapshotResponse {
            snapshot: NodeBootstrapSnapshot::new(now, descriptors),
        },
        now,
    );
    assert_eq!(
        report.candidates,
        UNTRUSTED_DISCOVERY_CANDIDATES_PER_MESSAGE
    );
    assert_eq!(report.rejected, 1);
    assert_eq!(store.len(), 0);
}

#[test]
fn test_peer_cache_snapshot_retains_expired_signed_records_without_making_them_live() {
    let store = PeerStore::new();
    let expired = signed_descriptor(1, 1_700_001_000);
    let node_id = expired.node_id();
    let snapshot = NodeBootstrapSnapshot::new(1_700_002_000, vec![expired.clone()]);

    let report = store.load_peer_cache_snapshot_from_source(&snapshot, 1_700_002_000, "cache");

    assert_eq!(
        report,
        PeerStoreImportReport {
            total: 1,
            inserted: 1,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 0,
        }
    );
    assert_eq!(store.len(), 1);
    assert!(store.get_valid(&node_id, 1_700_002_000).is_none());
    let status = store.status(1_700_002_000);
    assert_eq!(status.snapshot.valid_peers, 0);
    assert_eq!(status.peer_summary.expired_peers, 1);
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_expired"
            && event.outcome == "retained"
            && event.source == "cache"
            && event.reason.as_deref() == Some("signature_valid_descriptor_expired")
    }));

    let mut tampered = expired;
    tampered.signature[0] ^= 0x01;
    let rejected = store.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(1_700_002_010, vec![tampered]),
        1_700_002_010,
        "cache",
    );
    assert_eq!(rejected.rejected, 1);
}

#[test]
fn test_peer_cache_snapshot_rejects_expired_same_sequence_conflict() {
    let now = 1_700_002_000;
    let peer_kp = IdentityKeyPair::generate();
    let mut original_body = signed_descriptor_for(&peer_kp, 7, now - 1).descriptor;
    original_body.public_endpoint = Some("https://cache-a.example".to_string());
    let original = SignedNodeDescriptor::sign(original_body, &peer_kp).unwrap();
    let node_id = original.node_id();
    let store = PeerStore::new();

    let inserted = store.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now, vec![original.clone()]),
        now,
        "cache",
    );
    assert_eq!(inserted.inserted, 1);

    let exact_retry = store.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now + 1, vec![original.clone()]),
        now + 1,
        "cache",
    );
    assert_eq!(exact_retry.unchanged, 1);

    let mut conflicting_body = original.descriptor.clone();
    conflicting_body.public_endpoint = Some("https://cache-b.example".to_string());
    let conflicting = SignedNodeDescriptor::sign(conflicting_body, &peer_kp).unwrap();
    let rejected = store.load_peer_cache_snapshot_from_source(
        &NodeBootstrapSnapshot::new(now + 2, vec![conflicting]),
        now + 2,
        "cache",
    );

    assert_eq!(rejected.rejected, 1);
    assert_eq!(rejected.unchanged, 0);
    assert_eq!(store.len(), 1);
    assert!(store.get_valid(&node_id, now + 2).is_none());
    assert_eq!(
        store.export_peer_cache_snapshot(now + 2).peers,
        vec![original]
    );
    assert!(store
        .status(now + 2)
        .recent_peer_events
        .iter()
        .any(|event| {
            event.event == "peer_rejected"
                && event.outcome == "rejected"
                && event.reason.as_deref() == Some("sequence_conflict")
        }));
}

#[test]
fn test_load_bootstrap_snapshot_reports_inserted_and_rejected() {
    let store = PeerStore::new();
    let valid = signed_descriptor(1, 1_700_001_000);
    let expired = signed_descriptor(1, 1_700_000_050);
    let snapshot = NodeBootstrapSnapshot::new(1_700_000_010, vec![valid, expired]);

    let report = store.load_bootstrap_snapshot(&snapshot, 1_700_000_100);

    assert_eq!(
        report,
        PeerStoreImportReport {
            total: 2,
            inserted: 1,
            candidates: 0,
            unchanged: 0,
            stale: 0,
            rejected: 1,
        }
    );
    assert!(report.changed());
    assert_eq!(store.len(), 1);
    let status = store.status(1_700_000_100);
    assert_eq!(status.runtime.total_imported, 2);
    assert_eq!(status.runtime.inserted, 1);
    assert_eq!(status.runtime.rejected, 1);
    assert_eq!(status.runtime.last_import_at, Some(1_700_000_100));
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_inserted"
            && event.outcome == "accepted"
            && event.source == "unknown"
            && event.sequence == Some(1)
    }));
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_rejected"
            && event.outcome == "rejected"
            && event.source == "unknown"
            && event.reason.as_deref() == Some("verification_failed")
    }));
}

#[test]
fn test_load_bootstrap_snapshot_reports_unchanged_and_stale() {
    let store = PeerStore::new();
    let kp = IdentityKeyPair::generate();
    let newer = signed_descriptor_for(&kp, 2, 1_700_001_000);
    let same = signed_descriptor_for(&kp, 2, 1_700_001_000);
    let older = signed_descriptor_for(&kp, 1, 1_700_001_000);

    store.load_bootstrap_snapshot(
        &NodeBootstrapSnapshot::new(1_700_000_010, vec![newer]),
        1_700_000_100,
    );
    let report = store.load_bootstrap_snapshot(
        &NodeBootstrapSnapshot::new(1_700_000_020, vec![same, older]),
        1_700_000_100,
    );

    assert_eq!(
        report,
        PeerStoreImportReport {
            total: 2,
            inserted: 0,
            candidates: 0,
            unchanged: 1,
            stale: 1,
            rejected: 0,
        }
    );
    assert!(!report.changed());
    let status = store.status(1_700_000_100);
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_refreshed"
            && event.outcome == "ignored"
            && event.source == "unknown"
            && event.reason.as_deref() == Some("same_sequence")
    }));
    assert!(status.recent_peer_events.iter().any(|event| {
        event.event == "peer_rejected"
            && event.outcome == "rejected"
            && event.source == "unknown"
            && event.reason.as_deref() == Some("stale_sequence")
    }));
}

#[test]
fn test_export_bootstrap_snapshot_filters_private_peers() {
    let store = PeerStore::new();
    let public = signed_descriptor(1, 1_700_001_000);
    let private_key = IdentityKeyPair::generate();
    let mut private = signed_descriptor_for(&private_key, 1, 1_700_001_000);
    private.descriptor.policy.public_discovery = false;
    private = SignedNodeDescriptor::sign(private.descriptor, &private_key).unwrap();

    store.upsert_verified(public, 1_700_000_100).unwrap();
    store.upsert_verified(private, 1_700_000_100).unwrap();

    let snapshot = store.export_bootstrap_snapshot(1_700_000_200, 1_700_000_100, true, None);
    assert_eq!(snapshot.peers.len(), 1);
    assert!(snapshot.peers[0].descriptor.policy.public_discovery);
}

#[test]
fn test_build_snapshot_response_honors_limit() {
    let store = PeerStore::new();
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
        .unwrap();
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
        .unwrap();

    let message = store.build_snapshot_response(1_700_000_200, 1_700_000_100, true, Some(1));

    match message {
        NodeDiscoveryMessage::SnapshotResponse { snapshot } => {
            assert_eq!(snapshot.peers.len(), 1);
        }
        _ => panic!("expected snapshot response"),
    }
    assert_eq!(
        store.status(1_700_000_100).runtime.last_snapshot_at,
        Some(1_700_000_200)
    );
}

#[test]
fn test_apply_snapshot_response_imports_peers() {
    let store = PeerStore::new();
    let descriptor = signed_descriptor(1, 1_700_001_000);
    let message = NodeDiscoveryMessage::SnapshotResponse {
        snapshot: NodeBootstrapSnapshot::new(1_700_000_200, vec![descriptor]),
    };

    let report = store.apply_discovery_message(&message, 1_700_000_100);

    assert_eq!(report.inserted, 1);
    assert_eq!(store.len(), 1);
    assert_eq!(store.status(1_700_000_100).runtime.inserted, 1);
}

#[test]
fn test_bootstrap_status_is_recorded() {
    let store = PeerStore::new();

    store.configure_bootstrap_status(true, true, true, 2);
    store.record_bootstrap_source(
        1_700_000_010,
        "cache",
        "success",
        "total=1 inserted=1 unchanged=0 stale=0 rejected=0",
    );
    store.record_self_descriptor_status(1_700_000_011, "success", "registered");
    store.record_cache_save_status(1_700_000_012, "success", "exported=1");
    store.record_gossip_schedule(1_700_000_012, true, 180, -7);
    store.record_gossip_round(
        1_700_000_013,
        2,
        1,
        1,
        Some("snapshot_request_timeout".to_string()),
    );

    let status = store.status(1_700_000_100);
    assert!(status.bootstrap.enabled);
    assert!(status.bootstrap.peer_cache_configured);
    assert!(status.bootstrap.gossip_enabled);
    assert_eq!(status.bootstrap.seed_endpoints_configured, 2);
    assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("cache"));
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("success")
    );
    assert_eq!(
        status.bootstrap.self_descriptor_status.as_deref(),
        Some("success")
    );
    assert_eq!(
        status.bootstrap.last_cache_save_status.as_deref(),
        Some("success")
    );
    assert_eq!(status.bootstrap.last_gossip_attempted, 2);
    assert_eq!(status.bootstrap.last_gossip_seed_attempted, 1);
    assert_eq!(status.bootstrap.last_gossip_succeeded, 1);
    assert_eq!(status.bootstrap.last_gossip_failed, 1);
    assert_eq!(
        status.bootstrap.last_gossip_status.as_deref(),
        Some("degraded")
    );
    assert_eq!(
        status.bootstrap.last_gossip_failure_reason.as_deref(),
        Some("snapshot_request_timeout")
    );
    assert_eq!(status.bootstrap.consecutive_gossip_failures, 0);
    assert_eq!(status.bootstrap.last_gossip_success_at, Some(1_700_000_013));
    assert!(status.bootstrap.gossip_backpressure_active);
    assert_eq!(status.bootstrap.next_gossip_delay_seconds, Some(180));
    assert_eq!(status.bootstrap.next_gossip_jitter_seconds, -7);
    assert_eq!(
        status.bootstrap.last_gossip_schedule_at,
        Some(1_700_000_012)
    );
    assert!(status
        .recent_audit_events
        .iter()
        .any(|event| event.action == "outbound_gossip_round"));
    assert!(status
        .recent_audit_events
        .iter()
        .any(|event| event.action == "outbound_gossip_backpressure"));
}

#[test]
fn test_gossip_round_tracks_consecutive_failures() {
    let store = PeerStore::new();

    store.record_gossip_round(
        1_700_000_020,
        1,
        0,
        1,
        Some("announce_request_connect".to_string()),
    );
    store.record_gossip_round(
        1_700_000_030,
        1,
        0,
        1,
        Some("snapshot_request_timeout".to_string()),
    );

    let status = store.status(1_700_000_040);
    assert_eq!(
        status.bootstrap.last_gossip_status.as_deref(),
        Some("failed")
    );
    assert_eq!(
        status.bootstrap.last_gossip_failure_reason.as_deref(),
        Some("snapshot_request_timeout")
    );
    assert_eq!(status.bootstrap.consecutive_gossip_failures, 2);
    assert_eq!(status.bootstrap.last_gossip_success_at, None);

    store.record_gossip_round(1_700_000_050, 1, 1, 1, None);
    let status = store.status(1_700_000_060);
    assert_eq!(
        status.bootstrap.last_gossip_status.as_deref(),
        Some("healthy")
    );
    assert_eq!(status.bootstrap.last_gossip_failure_reason, None);
    assert_eq!(status.bootstrap.consecutive_gossip_failures, 0);
    assert_eq!(status.bootstrap.last_gossip_success_at, Some(1_700_000_050));
}

#[test]
fn test_directory_proof_gossip_round_tracks_bounded_convergence() {
    let store = PeerStore::new();

    // [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] One peer accepts
    // the primary proof while another accepts only the bounded fallback.
    store.record_directory_proof_gossip_round(
        1_700_000_070,
        PeerStoreDirectoryProofGossipRound {
            capability_checked: 3,
            capable: 2,
            peers_attempted: 2,
            frames_attempted: 3,
            accepted: 2,
            evidence_rejected: 1,
            replica_unavailable: 0,
            rate_limited: 0,
            protocol_rejected: 0,
            transport_failed: 0,
        },
    );

    let status = store.status(1_700_000_080);
    let bootstrap = &status.bootstrap;
    assert_eq!(
        bootstrap.last_directory_proof_gossip_status.as_deref(),
        Some("converged")
    );
    assert_eq!(bootstrap.last_directory_proof_gossip_capability_checked, 3);
    assert_eq!(bootstrap.last_directory_proof_gossip_capable, 2);
    assert_eq!(bootstrap.last_directory_proof_gossip_frames_attempted, 3);
    assert_eq!(
        bootstrap.last_directory_proof_gossip_fallback_frames_attempted,
        1
    );
    assert_eq!(bootstrap.last_directory_proof_gossip_accepted, 2);
    assert_eq!(
        bootstrap.last_directory_proof_gossip_acceptance_percent,
        100
    );
    assert_eq!(bootstrap.last_directory_proof_gossip_evidence_rejected, 1);
    assert_eq!(
        bootstrap.consecutive_directory_proof_gossip_zero_acceptance_rounds,
        0
    );
    assert_eq!(
        bootstrap.last_directory_proof_gossip_success_at,
        Some(1_700_000_070)
    );
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "directory_proof_gossip_round"
            && event.outcome == "accepted"
            && !event.detail.contains("http://")
    }));

    store.record_directory_proof_gossip_round(
        1_700_000_090,
        PeerStoreDirectoryProofGossipRound {
            capability_checked: 3,
            capable: 2,
            peers_attempted: 2,
            frames_attempted: 4,
            accepted: 0,
            evidence_rejected: 4,
            replica_unavailable: 0,
            rate_limited: 0,
            protocol_rejected: 0,
            transport_failed: 0,
        },
    );
    let status = store.status(1_700_000_100);
    let bootstrap = &status.bootstrap;
    assert_eq!(
        bootstrap.last_directory_proof_gossip_status.as_deref(),
        Some("evidence_diverged")
    );
    assert_eq!(
        bootstrap.consecutive_directory_proof_gossip_zero_acceptance_rounds,
        1
    );
    assert_eq!(
        bootstrap.last_directory_proof_gossip_success_at,
        Some(1_700_000_070)
    );
}

#[test]
fn test_directory_proof_gossip_negotiation_failure_is_not_legacy_only() {
    let store = PeerStore::new();

    store.record_directory_proof_gossip_round(
        1_700_000_110,
        PeerStoreDirectoryProofGossipRound {
            capability_checked: 2,
            capable: 0,
            peers_attempted: 0,
            frames_attempted: 0,
            accepted: 0,
            evidence_rejected: 0,
            replica_unavailable: 0,
            rate_limited: 0,
            protocol_rejected: 1,
            transport_failed: 1,
        },
    );

    let status = store.status(1_700_000_120);
    assert_eq!(
        status
            .bootstrap
            .last_directory_proof_gossip_status
            .as_deref(),
        Some("degraded")
    );
    assert_eq!(
        status
            .bootstrap
            .last_directory_proof_gossip_transport_failed,
        1
    );
}

#[test]
fn test_gossip_schedule_status_tracks_backpressure_without_peer_details() {
    let store = PeerStore::new();

    store.record_gossip_schedule(1_700_000_020, true, 240, 12);
    assert_eq!(store.consecutive_gossip_failures(), 0);

    let status = store.status(1_700_000_030);
    assert!(status.bootstrap.gossip_backpressure_active);
    assert_eq!(status.bootstrap.next_gossip_delay_seconds, Some(240));
    assert_eq!(status.bootstrap.next_gossip_jitter_seconds, 12);
    assert_eq!(
        status.bootstrap.last_gossip_schedule_at,
        Some(1_700_000_020)
    );
    assert!(status.recent_audit_events.iter().any(|event| {
        event.action == "outbound_gossip_backpressure"
            && event.outcome == "limited"
            && !event.detail.contains("http")
    }));
}

#[test]
fn test_gossip_success_sets_effective_recovery_status_without_hiding_source_warning() {
    let store = PeerStore::new();

    store.record_bootstrap_source(
        1_700_000_010,
        "file",
        "warning",
        "total=1 inserted=0 unchanged=0 stale=0 rejected=1",
    );
    store.record_gossip_round(1_700_000_050, 2, 2, 1, None);

    let status = store.status(1_700_000_060);
    assert_eq!(
        status.bootstrap.last_source_status.as_deref(),
        Some("warning")
    );
    assert_eq!(status.bootstrap.last_source_kind.as_deref(), Some("file"));
    assert_eq!(status.bootstrap.recovery_status.as_deref(), Some("success"));
    assert_eq!(
        status.bootstrap.recovery_detail.as_deref(),
        Some("gossip_recovered attempted=2 succeeded=2 seed_attempted=1")
    );
    assert_eq!(status.bootstrap.recovery_at, Some(1_700_000_050));
}

#[test]
fn test_peer_cache_load_evidence_is_separate_from_generic_recovery_status() {
    let store = PeerStore::new();

    store.record_bootstrap_source(1_700_000_010, "cache", "failed", "json_rejected");
    store.record_bootstrap_source(
        1_700_000_011,
        "cache_backup",
        "success",
        "total=2 inserted=2 unchanged=0 stale=0 rejected=0",
    );
    store.record_gossip_round(1_700_000_050, 2, 2, 1, None);

    let status = store.status(1_700_000_060);
    assert_eq!(
        status.bootstrap.last_cache_load_source.as_deref(),
        Some("cache_backup")
    );
    assert_eq!(
        status.bootstrap.last_cache_load_status.as_deref(),
        Some("success")
    );
    assert_eq!(
        status.bootstrap.last_cache_load_detail.as_deref(),
        Some("total=2 inserted=2 unchanged=0 stale=0 rejected=0")
    );
    assert_eq!(status.bootstrap.last_cache_load_at, Some(1_700_000_011));
    assert_eq!(status.bootstrap.recovery_status.as_deref(), Some("success"));
    assert_eq!(
        status.bootstrap.recovery_detail.as_deref(),
        Some("gossip_recovered attempted=2 succeeded=2 seed_attempted=1")
    );
}

#[test]
fn test_stability_marks_ready_when_peer_view_and_gossip_are_fresh() {
    let store = PeerStore::new();
    store.configure_bootstrap_status(true, true, true, 2);
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
        .unwrap();
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
        .unwrap();
    store.record_gossip_round(1_700_000_120, 2, 2, 1, None);

    let status = store.status(1_700_000_180);

    assert_eq!(status.stability.health, "healthy");
    assert!(status.stability.relay_foundation_ready);
    assert_eq!(status.stability.last_gossip_success_age_seconds, Some(60));
    assert_eq!(status.stability.last_gossip_round_age_seconds, Some(60));
    assert!(status.stability.seed_recovery_configured);
    assert!(status.stability.restart_recovery_configured);
    assert_eq!(
        status.stability.restart_recovery_sources,
        vec!["seed_endpoints".to_string(), "peer_cache".to_string()]
    );
}

#[test]
fn test_stability_blocks_after_repeated_gossip_failures() {
    let store = PeerStore::new();
    store.configure_bootstrap_status(true, true, true, 1);
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
        .unwrap();
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), 1_700_000_100)
        .unwrap();

    for now in [1_700_000_120, 1_700_000_180, 1_700_000_240] {
        store.record_gossip_round(now, 2, 0, 1, Some("snapshot_request_timeout".to_string()));
    }

    let status = store.status(1_700_000_300);

    assert_eq!(status.stability.health, "failed");
    assert!(!status.stability.relay_foundation_ready);
    assert_eq!(status.bootstrap.consecutive_gossip_failures, 3);
    assert_eq!(status.stability.last_gossip_round_age_seconds, Some(60));
    assert_eq!(status.stability.last_gossip_success_age_seconds, None);
}

#[test]
fn test_stability_marks_gossip_success_as_stale() {
    let store = PeerStore::new();
    store.configure_bootstrap_status(true, true, true, 1);
    store
        .upsert_verified(signed_descriptor(1, 1_700_010_000), 1_700_000_100)
        .unwrap();
    store
        .upsert_verified(signed_descriptor(1, 1_700_010_000), 1_700_000_100)
        .unwrap();
    store.record_gossip_round(1_700_000_120, 2, 2, 1, None);

    let status = store.status(1_700_001_100);

    assert_eq!(status.stability.health, "stale");
    assert!(!status.stability.relay_foundation_ready);
    assert_eq!(status.stability.last_gossip_success_age_seconds, Some(980));
    assert_eq!(
        status.stability.stale_after_seconds,
        DISCOVERY_GOSSIP_STALE_AFTER_SECS
    );
}

#[test]
fn test_stability_accepts_peer_cache_as_restart_recovery() {
    let store = PeerStore::new();
    store.configure_bootstrap_status(true, true, true, 0);
    let now = 1_700_000_180;
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
        .unwrap();
    store
        .upsert_verified(signed_descriptor(1, 1_700_001_000), now)
        .unwrap();
    store.record_gossip_round(now - 60, 2, 2, 0, None);

    let status = store.status(now);

    assert_eq!(status.stability.health, "healthy");
    assert!(status.stability.relay_foundation_ready);
    assert!(!status.stability.seed_recovery_configured);
    assert!(status.stability.restart_recovery_configured);
    assert_eq!(
        status.stability.restart_recovery_sources,
        vec!["peer_cache".to_string()]
    );
}
