// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

// [PHALA-QUOTE-RESPONSE-OWNERSHIP 2026-10-08 by Codex] Authored only.
// Local registries/permits keep paused-clock fixtures independent of public
// process admission and unrelated tests; the production body code is reused.
#[tokio::test(start_paused = true)]
async fn phala_quote_response_retains_capacity_and_expires_unpolled_buffers() {
    let registry = Arc::new(PhalaQuoteResponseRegistry::default());
    let owner = Arc::new(PhalaAttestationDeliveryOwner {
        closed: std::sync::atomic::AtomicBool::new(false), registry: Arc::clone(&registry),
    });
    let permits = Arc::new(tokio::sync::Semaphore::new(1));
    let body = owner.response_body(vec![17; PHALA_QUOTE_RESPONSE_CHUNK_BYTES * 3],
        Arc::clone(&permits).try_acquire_owned().unwrap()).unwrap();
    assert_eq!(permits.available_permits(), 0);
    let mut stream = body.into_data_stream();
    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.len(), PHALA_QUOTE_RESPONSE_CHUNK_BYTES);
    assert_eq!(permits.available_permits(), 0);
    tokio::time::advance(PHALA_QUOTE_RESPONSE_TIMEOUT).await;
    owner.expire_buffers();
    assert_eq!(permits.available_permits(), 1, "no HTTP poll is needed to reclaim the remaining large allocation");
    assert!(registry.responses.lock().is_empty());
    assert_eq!(first.as_ref(), vec![17u8; PHALA_QUOTE_RESPONSE_CHUNK_BYTES].as_slice(), "detached socket chunk remains bounded");
    assert!(stream.next().await.unwrap().is_err(), "expiry cannot turn a truncated response into successful EOF");
    assert!(stream.next().await.is_none());
    let body = owner.response_body(vec![19; 3], Arc::clone(&permits).try_acquire_owned().unwrap()).unwrap();
    drop(body);
    assert_eq!(permits.available_permits(), 1);
    let body = owner.response_body(vec![21; 5], Arc::clone(&permits).try_acquire_owned().unwrap()).unwrap();
    let mut stream = body.into_data_stream();
    assert_eq!(stream.next().await.unwrap().unwrap().as_ref(), &[21; 5]);
    assert!(stream.next().await.is_none());
    assert_eq!(permits.available_permits(), 1);
}

#[tokio::test]
async fn phala_quote_response_stop_fences_handoff_without_stopping_other_owner() {
    let registry = Arc::new(PhalaQuoteResponseRegistry::default());
    let owner = Arc::new(PhalaAttestationDeliveryOwner {
        closed: std::sync::atomic::AtomicBool::new(false), registry: Arc::clone(&registry),
    });
    let other = Arc::new(PhalaAttestationDeliveryOwner {
        closed: std::sync::atomic::AtomicBool::new(false), registry: Arc::clone(&registry),
    });
    let permits = Arc::new(tokio::sync::Semaphore::new(2));
    let body = owner.response_body(vec![11; 10], Arc::clone(&permits).try_acquire_owned().unwrap()).unwrap();
    let other_body = other.response_body(vec![13; 10], Arc::clone(&permits).try_acquire_owned().unwrap()).unwrap();
    owner.stop();
    assert!(owner.is_stopped());
    assert!(!other.is_stopped());
    assert_eq!(permits.available_permits(), 1);
    assert!(owner.response_body(vec![15; 10], Arc::clone(&permits).try_acquire_owned().unwrap()).is_err());
    assert_eq!(permits.available_permits(), 1, "rejected handoff returns its permit");
    assert!(body.into_data_stream().next().await.unwrap().is_err());
    assert_eq!(other_body.into_data_stream().next().await.unwrap().unwrap().as_ref(), &[13; 10]);
    assert_eq!(permits.available_permits(), 2);
    assert!(other.response_body(vec![0; PHALA_QUOTE_RESPONSE_MAX_BYTES + 1],
        Arc::clone(&permits).try_acquire_owned().unwrap()).is_err());
    assert_eq!(permits.available_permits(), 2);
    let policy = DiscoveryApiPolicy::default();
    assert!(Arc::ptr_eq(&policy.phala_attestation_delivery_owner(), &policy.clone().phala_attestation_delivery_owner()));
}

// [PUBLIC-DISCOVERY-PROJECTION 2026-09-01 by Codex] These fixtures bind
// public handler output to full verified descriptor membership and exercise
// both the direct snapshot leak and every prefix-only status projection.
#[test]
fn test_public_runtime_event_projection_skips_private_and_unknown_audit_actions() {
    let store = PeerStore::new();
    let generated_at = 1_700_000_100;
    let private_prefix = "node_prefix=deadbeef result=success";

    store.record_blind_relay_forwarded(generated_at - 4, 1);
    store.record_audit_event(
        generated_at - 3,
        "blind_relay_route_health",
        "accepted",
        private_prefix,
    );
    store.record_audit_event(
        generated_at - 2,
        "blind_relay_future_internal_action",
        "accepted",
        "node_prefix=cafebabe internal=unknown",
    );
    store.record_blind_relay_quarantine_started(generated_at - 4, "failure_threshold");
    store.record_audit_event(
        generated_at - 3,
        "blind_relay_peer_quarantine",
        "limited",
        "node_prefix=deadbeef reason=failure_threshold",
    );
    store.record_audit_event(
        generated_at - 2,
        "blind_relay_future_internal_action",
        "rejected",
        "node_prefix=cafebabe internal=unknown",
    );

    let status = store.status(generated_at);
    let successful = latest_blind_relay_event_value(&status, generated_at, true);
    let failed = latest_blind_relay_event_value(&status, generated_at, false);

    assert_eq!(successful["action"], "blind_relay_forward");
    assert_eq!(successful["reason_bucket"], "opaque_forward_accepted");
    assert_eq!(failed["action"], "blind_relay_quarantine");
    assert_eq!(failed["reason_bucket"], "relay_quarantine_started");
    let serialized = serde_json::to_string(&(successful, failed)).unwrap();
    assert!(!serialized.contains("deadbeef"));
    assert!(!serialized.contains("cafebabe"));
    assert!(!serialized.contains("future_internal_action"));

    let unknown_only = PeerStore::new();
    unknown_only.record_audit_event(
        generated_at,
        "blind_relay_future_internal_action",
        "accepted",
        "node_prefix=deadbeef",
    );
    assert!(
        latest_blind_relay_event_value(&unknown_only.status(generated_at), generated_at, true,)
            .is_null()
    );
}

#[tokio::test]
async fn public_runtime_projection_suppresses_private_and_unknown_events_across_surfaces() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let relay_capabilities = [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    let private_keypair = IdentityKeyPair::from_bytes(&[218; 32]).unwrap();
    let mut private_descriptor = signed_candidate_exclusion_descriptor(
        218,
        now,
        Some("private-runtime.invalid:443"),
        &relay_capabilities,
        true,
    );
    private_descriptor.descriptor.policy.public_discovery = false;
    let private_descriptor =
        SignedNodeDescriptor::sign(private_descriptor.descriptor, &private_keypair).unwrap();
    let private_node_id = private_descriptor.node_id();
    let private_prefix = hex::encode(&private_node_id[..4]);
    store.upsert_verified(private_descriptor, now).unwrap();

    // Safe aggregate events remain visible even when newer per-node,
    // unknown, or incorrectly identity-bearing events are present.
    store.record_blind_relay_forwarded(now, 1);
    store.record_blind_relay_rejected(now, "rate_limited");
    store.record_route_forward_success(&private_node_id, now);
    store.record_route_forward_failure(&private_node_id, now, "request_failed");
    store.record_audit_event(
        now,
        "blind_relay_future_runtime_event",
        "accepted",
        "future_detail=opaque",
    );
    store.record_audit_event(
        now,
        "blind_relay_future_runtime_event",
        "rejected",
        "future_detail=opaque",
    );
    store.record_audit_event(
        now,
        "blind_relay_forward",
        "accepted",
        format!("node_prefix={private_prefix} result=success"),
    );
    store.record_audit_event(
        now,
        "blind_relay_probe",
        "rejected",
        format!("node_prefix={private_prefix} result=failure"),
    );

    let local_capabilities = DiscoveryLocalCapabilityStatus::default();
    let status = store.status(now);
    let runtime = blind_relay_runtime_status_value(now, &status, &local_capabilities);
    assert_eq!(
        runtime["last_successful_blind_relay"]["action"].as_str(),
        Some("blind_relay_forward")
    );
    assert_eq!(
        runtime["last_successful_blind_relay"]["reason_bucket"].as_str(),
        Some("opaque_forward_accepted")
    );
    assert_eq!(
        runtime["last_failed_blind_relay"]["action"].as_str(),
        Some("blind_relay_probe")
    );
    assert_eq!(
        runtime["last_failed_blind_relay"]["reason_bucket"].as_str(),
        Some("synthetic_probe_rejected")
    );
    let serialized_runtime = serde_json::to_string(&runtime).unwrap();
    assert!(!serialized_runtime.contains(&private_prefix));
    assert!(!serialized_runtime.contains("blind_relay_future_runtime_event"));

    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
    for uri in [
        "/api/discovery/status",
        "/api/discovery/summary",
        "/api/discovery/public-card",
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let serialized = String::from_utf8(body.to_vec()).unwrap();
        assert!(!serialized.contains(&private_prefix), "uri={uri}");
        if uri != "/api/discovery/public-card" {
            let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
            assert_eq!(
                parsed["blind_relay_runtime"]["last_successful_blind_relay"]["action"].as_str(),
                Some("blind_relay_forward")
            );
            assert_eq!(
                parsed["blind_relay_runtime"]["last_failed_blind_relay"]["reason_bucket"].as_str(),
                Some("synthetic_probe_rejected")
            );
        }
    }

    let unsafe_only = PeerStore::new();
    unsafe_only.record_audit_event(
        now,
        "blind_relay_route_health",
        "accepted",
        format!("node_prefix={private_prefix} result=success"),
    );
    unsafe_only.record_audit_event(
        now,
        "blind_relay_future_runtime_event",
        "rejected",
        "future_detail=opaque",
    );
    let unsafe_runtime =
        blind_relay_runtime_status_value(now, &unsafe_only.status(now), &local_capabilities);
    assert!(unsafe_runtime["last_successful_blind_relay"].is_null());
    assert!(unsafe_runtime["last_failed_blind_relay"].is_null());
}

#[tokio::test]
async fn test_two_hop_candidates_wait_for_signed_restart_continuity() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
    let first_node_id = first.node_id();
    let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
    let second_node_id = second.node_id();

    store.upsert_verified(first, now).unwrap();
    store.upsert_verified(second, now).unwrap();
    store.record_route_forward_success(&first_node_id, now);
    store.record_route_forward_success(&second_node_id, now);
    record_stable_runtime_path_proof(store.as_ref(), now, 2);

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

    assert!(parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_runtime_proof_ready);
    assert!(!parsed.requested_restart_continuity_ready);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 1);
    assert!(parsed.fallback_required);
    assert_eq!(parsed.pool_status, "continuity_warming");
    assert_eq!(parsed.route_plan, "standard_relay_fallback");
    assert_eq!(
        parsed.fallback_reason,
        "requested_path_restart_continuity_not_ready"
    );
    assert_eq!(
        parsed.readiness_reason,
        "waiting_for_requested_path_restart_continuity"
    );
}

#[tokio::test]
async fn test_high_privacy_candidates_require_pairwise_network_diversity() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let first = signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422");
    let first_node_id = first.node_id();
    let second = signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422");
    let second_node_id = second.node_id();
    let collocated_third =
        signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.20:8422");
    let third_node_id = collocated_third.node_id();

    store.upsert_verified(first, now).unwrap();
    store.upsert_verified(second, now).unwrap();
    store.upsert_verified(collocated_third, now).unwrap();
    store.record_route_forward_success(&first_node_id, now);
    store.record_route_forward_success(&second_node_id, now);
    store.record_route_forward_success(&third_node_id, now);
    record_stable_path_proof(store.as_ref(), now, 2);
    record_stable_path_proof(store.as_ref(), now, 3);

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

    // [ONION-NETWORK-DIVERSITY 2026-08-03 by Codex] Candidate count and
    // mature proof are insufficient when two of three hops share an IPv4
    // /24. The already-diverse two-hop subset remains a safe fallback.
    assert_eq!(parsed.count, 3);
    assert!(parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_network_diversity_required);
    assert!(!parsed.requested_network_diversity_ready);
    assert!(parsed.requested_runtime_proof_ready);
    assert!(parsed.requested_restart_continuity_ready);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 2);
    assert!(parsed.fallback_required);
    assert_eq!(parsed.pool_status, "diversity_limited");
    assert_eq!(parsed.route_plan, "two_hop_onion_path");
    assert_eq!(
        parsed.fallback_reason,
        "requested_path_network_diversity_not_ready"
    );
    assert_eq!(
        parsed.readiness_reason,
        "waiting_for_network_diverse_onion_relays"
    );
    assert_eq!(
        parsed.next_action,
        "use the network-diverse two-hop fallback until a diverse third hop is available"
    );
    assert!(parsed.network_diversity_policy.contains("ipv4_24"));
    assert!(parsed
        .network_diversity_policy
        .contains("not_operator_or_as_proof"));
}

#[tokio::test]
async fn test_strict_pinned_route_domains_preserve_distinct_high_privacy_pool() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let local = signed_routeable_chat_descriptor(1, now + 300, "https://10.0.0.5:8422");
    let local_node_id = local.node_id();
    store.upsert_verified(local, now).unwrap();

    let remotes = [
        signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://203.0.113.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://198.18.0.10:8422"),
    ];
    let remote_node_ids = remotes
        .iter()
        .map(SignedNodeDescriptor::node_id)
        .collect::<Vec<_>>();
    for remote in remotes {
        let node_id = remote.node_id();
        store.upsert_verified(remote, now).unwrap();
        store.record_route_forward_success(&node_id, now);
    }
    record_stable_path_proof(store.as_ref(), now, 3);

    let mut config = DiscoveryConfig::default();
    config.require_pinned_route_domains_for_multi_hop = true;
    config.pinned_route_domains.insert(
        hex::encode(local_node_id),
        "11111111111111111111111111111111".to_string(),
    );
    for (node_id, domain) in remote_node_ids.iter().copied().zip([
        "22222222222222222222222222222222",
        "22222222222222222222222222222222",
        "33333333333333333333333333333333",
        "44444444444444444444444444444444",
    ]) {
        config
            .pinned_route_domains
            .insert(hex::encode(node_id), domain.to_string());
    }
    let policy = DiscoveryApiPolicy::from_config(&config);

    let app = build_discovery_router_with_local_entry(
        store,
        policy,
        DiscoveryLocalCapabilityStatus::default(),
        None,
        local_node_id,
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

    // [PINNED-ROUTE-DOMAINS 2026-08-03 by Codex] Four routeable remote
    // nodes exist, but two share one audited failure domain. Strict mode
    // emits only a pairwise-distinct three-hop pool, keeps the path ready,
    // and never discloses the opaque local assignment tokens.
    assert_eq!(parsed.count, 3);
    assert!(parsed.requested_pinned_route_domain_required);
    assert!(parsed.requested_pinned_route_domain_ready);
    assert!(parsed.local_entry_pinned_route_domain_enforced);
    assert!(parsed.requested_network_diversity_ready);
    assert!(parsed.requested_path_ready);
    assert_eq!(parsed.route_plan, "three_hop_onion_path");
    assert!(parsed
        .pinned_route_domain_policy
        .contains("operator_audited_local_opaque_assignments"));
    assert_eq!(
        parsed
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.node_id == hex::encode(remote_node_ids[0])
                    || candidate.node_id == hex::encode(remote_node_ids[1])
            })
            .count(),
        1
    );
    let encoded = String::from_utf8(body.to_vec()).unwrap();
    assert!(!encoded.contains("11111111111111111111111111111111"));
    assert!(!encoded.contains("22222222222222222222222222222222"));
}

#[tokio::test]
async fn test_strict_pinned_route_domains_fail_closed_without_local_entry_assignment() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let local = signed_routeable_chat_descriptor(1, now + 300, "https://10.0.1.5:8422");
    let local_node_id = local.node_id();
    store.upsert_verified(local, now).unwrap();

    let remotes = [
        signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://203.0.113.10:8422"),
    ];
    let mut config = DiscoveryConfig::default();
    config.require_pinned_route_domains_for_multi_hop = true;
    for (remote, domain) in remotes.into_iter().zip([
        "55555555555555555555555555555555",
        "66666666666666666666666666666666",
        "77777777777777777777777777777777",
    ]) {
        let node_id = remote.node_id();
        config
            .pinned_route_domains
            .insert(hex::encode(node_id), domain.to_string());
        store.upsert_verified(remote, now).unwrap();
        store.record_route_forward_success(&node_id, now);
    }
    record_stable_path_proof(store.as_ref(), now, 3);

    let app = build_discovery_router_with_local_entry(
        store,
        DiscoveryApiPolicy::from_config(&config),
        DiscoveryLocalCapabilityStatus::default(),
        None,
        local_node_id,
    );
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(parsed.count, 3);
    assert!(parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_pinned_route_domain_required);
    assert!(!parsed.local_entry_pinned_route_domain_enforced);
    assert!(!parsed.requested_pinned_route_domain_ready);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 1);
    assert_eq!(parsed.pool_status, "routing_domain_limited");
    assert_eq!(
        parsed.fallback_reason,
        "requested_path_pinned_route_domain_not_ready"
    );
    assert_eq!(
        parsed.readiness_reason,
        "waiting_for_operator_audited_route_domain_coverage"
    );
    assert_eq!(parsed.route_plan, "standard_relay_fallback");
}

#[tokio::test]
async fn test_high_privacy_candidates_fall_back_until_three_hop_proof_is_mature() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let first = signed_routeable_chat_descriptor(1, now + 300, "https://relay-one.example");
    let first_node_id = first.node_id();
    let second = signed_routeable_chat_descriptor(1, now + 300, "https://relay-two.example");
    let second_node_id = second.node_id();
    let third = signed_routeable_chat_descriptor(1, now + 300, "https://relay-three.example");
    let third_node_id = third.node_id();

    store.upsert_verified(first, now).unwrap();
    store.upsert_verified(second, now).unwrap();
    store.upsert_verified(third, now).unwrap();
    store.record_route_forward_success(&first_node_id, now);
    store.record_route_forward_success(&second_node_id, now);
    store.record_route_forward_success(&third_node_id, now);
    record_stable_path_proof(store.as_ref(), now, 2);

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?privacy_mode=high&limit=3")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();

    assert_eq!(parsed.count, 3);
    assert_eq!(parsed.requested_hops, 3);
    assert!(parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_runtime_proof_required);
    assert!(!parsed.requested_runtime_proof_ready);
    assert!(parsed.requested_restart_continuity_required);
    assert!(!parsed.requested_restart_continuity_ready);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 2);
    assert!(parsed.fallback_required);
    assert_eq!(parsed.pool_status, "proof_warming");
    assert_eq!(parsed.route_plan, "two_hop_onion_path");
    assert_eq!(
        parsed.fallback_reason,
        "requested_path_runtime_proof_not_ready"
    );
    assert_eq!(
        parsed.readiness_reason,
        "waiting_for_stable_requested_path_runtime_proof"
    );
    assert_eq!(
        parsed.next_action,
        "use the mature two-hop onion fallback while requested path evidence warms"
    );
}
