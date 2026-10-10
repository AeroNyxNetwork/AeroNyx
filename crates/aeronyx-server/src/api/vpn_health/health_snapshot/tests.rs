// ============================================
// File: crates/aeronyx-server/src/api/vpn_health/health_snapshot/tests.rs
// ============================================
//! # Tests: VPN health snapshot assembly
//!
//! Unit tests for VPN health snapshot assembly, moved from the former
//! `api::vpn_health::tests` module.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `api/vpn_health.rs`; bodies unchanged.

use super::*;

use crate::api::vpn_health::AnonymousMailboxReadinessProjection;

#[test]
fn chat_relay_health_distinguishes_missing_enabled_runtime() {
    let status = collect_chat_relay_health_status(true, None);

    assert!(status.configured_enabled);
    assert!(!status.runtime_ready);
    assert_eq!(
        status.peer_relay.last_outbound_status.as_deref(),
        Some("failed")
    );
    assert_eq!(
        status.peer_relay.last_outbound_failure_reason.as_deref(),
        Some("chat_relay_runtime_unavailable")
    );
    assert_eq!(status.peer_relay.custody_durability.state, "unknown");
    assert!(
        !status
            .peer_relay
            .custody_durability
            .full_durability_verified
    );
    assert_eq!(status.peer_relay.custody_durability.synchronous_level, None);
    assert_eq!(status.source, "rust_chat_relay_runtime_unavailable");
    let encoded = serde_json::to_value(status).expect("serialize legacy health");
    assert!(encoded["peer_relay"]
        .get("anonymous_mailbox_readiness")
        .is_none());
}

#[test]
fn anonymous_mailbox_readiness_is_applied_only_after_explicit_publish() {
    let projection = AnonymousMailboxReadinessProjection::default();
    let mut before = collect_chat_relay_health_status(false, None);
    projection.apply_to(&mut before.peer_relay);
    let before = serde_json::to_value(before).expect("serialize unpublished readiness");
    assert!(before["peer_relay"]
        .get("anonymous_mailbox_readiness")
        .is_none());

    projection.publish_local_composition(true, true, true, true, true, true);
    let mut after = collect_chat_relay_health_status(false, None);
    projection.apply_to(&mut after.peer_relay);
    let after = serde_json::to_value(after).expect("serialize published readiness");
    let readiness = &after["peer_relay"]["anonymous_mailbox_readiness"];
    assert_eq!(readiness["version"], 1);
    assert_eq!(readiness["configured"], true);
    assert_eq!(readiness["custody_store_opened"], true);
    assert_eq!(readiness["ticket_terminal_wired"], true);
    assert_eq!(readiness["source_coordinator_enabled"], true);
    assert_eq!(readiness["dispatcher_admitted"], true);
    assert_eq!(readiness["cleanup_runtime_supervised"], true);
    assert_eq!(readiness["scope"], "local_node_runtime_only");
    assert!(readiness.get("e2e_ready").is_none());
    assert!(readiness.get("live").is_none());
}

#[test]
fn chat_relay_health_preserves_service_owned_snapshot() {
    let mut peer_status = ChatRelayPeerStatus::new(true);
    peer_status.outbound_attempted_total = 4;
    peer_status.outbound_accepted_total = 3;
    peer_status.last_outbound_status = Some("degraded".to_string());
    peer_status.last_outbound_failure_reason = Some("peer_relay_request_timeout".to_string());
    peer_status.direct_peer_retry.retry_triggered_total = 2;
    peer_status.direct_peer_retry.retry_recovered_total = 1;
    peer_status.direct_peer_retry.retry_exhausted_total = 1;
    peer_status.direct_peer_retry.last_outcome = Some("exhausted".to_string());
    peer_status.custody_durability.state = "full".to_string();
    peer_status.custody_durability.full_durability_verified = true;
    peer_status.custody_durability.synchronous_level = Some(2);

    // [RELAY-HEALTH-DIAGNOSTICS 2026-08-15 by Codex] Health consumes the
    // service snapshot verbatim instead of maintaining parallel counters.
    let status = collect_chat_relay_health_status(true, Some(peer_status.clone()));
    assert!(status.runtime_ready);
    assert_eq!(status.peer_relay, peer_status);
    assert_eq!(status.source, "rust_chat_relay_service");
    let encoded = serde_json::to_value(&status).expect("serialize relay health");
    assert_eq!(encoded["runtime_ready"], true);
    assert_eq!(encoded["peer_relay"]["outbound_attempted_total"], 4);
    assert_eq!(encoded["peer_relay"]["custody_durability"]["state"], "full");
    assert_eq!(
        encoded["peer_relay"]["custody_durability"]["full_durability_verified"],
        true
    );
    assert_eq!(
        encoded["peer_relay"]["direct_peer_retry"]["retry_triggered_total"],
        2
    );
    assert_eq!(
        encoded["peer_relay"]["direct_peer_retry"]["last_outcome"],
        "exhausted"
    );
    assert_eq!(
        encoded["peer_relay"]["last_outbound_failure_reason"],
        "peer_relay_request_timeout"
    );
}

#[test]
fn local_discovery_health_includes_shared_recovery_anchor_contract() {
    let status = collect_discovery_status_value(&PeerStore::new());

    assert_eq!(
        status["recovery_anchor"]["contract_version"],
        "recovery_anchor.v1"
    );
    assert_eq!(status["recovery_anchor"]["status"], "idle");
    assert_eq!(status["recovery_anchor"]["ready_for_restore"], false);
    let rendered = serde_json::to_string(&status).expect("serialize discovery health");
    for forbidden in ["anchor_digest", "signature_hex", "witness_endpoint"] {
        assert!(!rendered.contains(forbidden));
    }
}
