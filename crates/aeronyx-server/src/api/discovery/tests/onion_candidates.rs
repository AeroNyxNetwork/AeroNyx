// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[test]
fn test_onion_candidate_exclusion_telemetry_suppresses_small_buckets() {
    let telemetry = OnionCandidateExclusionCounts {
        capability_or_feature: 3,
        routeability_unknown_or_stale: 2,
        routeability_failed_or_quarantined: 0,
        missing_kem_or_endpoint: 1,
        anti_affinity_or_policy: 4,
        unclassified: 0,
    }
    .into_telemetry(10);

    assert_eq!(
        telemetry.contract_version,
        ONION_CANDIDATE_EXCLUSION_TELEMETRY_CONTRACT_VERSION
    );
    assert_eq!(
        telemetry.status,
        OnionCandidateExclusionTelemetryStatus::Partial
    );
    assert_eq!(telemetry.buckets.capability_or_feature, Some(3));
    assert_eq!(telemetry.buckets.routeability_unknown_or_stale, None);
    assert_eq!(
        telemetry.buckets.routeability_failed_or_quarantined,
        Some(0)
    );
    assert_eq!(telemetry.buckets.missing_kem_or_endpoint, None);
    assert_eq!(telemetry.buckets.anti_affinity_or_policy, Some(4));

    let encoded = serde_json::to_value(&telemetry).unwrap();
    assert!(encoded.get("observed_descriptors").is_none());
    assert!(encoded.get("unclassified").is_none());
    let buckets = encoded["buckets"].as_object().unwrap();
    assert!(!buckets.contains_key("routeability_unknown_or_stale"));
    assert!(!buckets.contains_key("missing_kem_or_endpoint"));

    let small_sample = OnionCandidateExclusionCounts {
        capability_or_feature: 2,
        ..OnionCandidateExclusionCounts::default()
    }
    .into_telemetry(2);
    assert_eq!(
        small_sample.status,
        OnionCandidateExclusionTelemetryStatus::SuppressedSmallSample
    );
    assert_eq!(
        serde_json::to_value(&small_sample).unwrap()["buckets"],
        serde_json::json!({})
    );
}

#[test]
fn test_onion_candidate_exclusion_observer_classifies_all_coarse_gates() {
    let store = PeerStore::new();
    let now = now_secs();
    let relay_capabilities = [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    let capability_limited = [NodeCapability::ChatRelay];

    for seed in 1..=3 {
        let endpoint = format!("https://198.18.2.{seed}:8422");
        let descriptor = signed_candidate_exclusion_descriptor(
            seed,
            now,
            Some(&endpoint),
            &capability_limited,
            true,
        );
        store.upsert_verified(descriptor, now).unwrap();
    }
    for seed in 4..=6 {
        let endpoint = format!("https://198.51.100.{seed}:8422");
        let descriptor = signed_candidate_exclusion_descriptor(
            seed,
            now,
            Some(&endpoint),
            &relay_capabilities,
            true,
        );
        store.upsert_verified(descriptor, now).unwrap();
    }
    for seed in 7..=9 {
        let endpoint = format!("https://203.0.113.{seed}:8422");
        let descriptor = signed_candidate_exclusion_descriptor(
            seed,
            now,
            Some(&endpoint),
            &relay_capabilities,
            true,
        );
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        for _ in 0..3 {
            store.record_route_forward_failure(&node_id, now, "request_failed");
        }
    }
    for seed in 10..=12 {
        let endpoint = format!("https://198.18.1.{seed}:8422");
        let descriptor = signed_candidate_exclusion_descriptor(
            seed,
            now,
            Some(&endpoint),
            &relay_capabilities,
            false,
        );
        store.upsert_verified(descriptor, now).unwrap();
    }
    for seed in 13..=15 {
        let endpoint = format!("https://192.0.2.{seed}:8422");
        let descriptor = signed_candidate_exclusion_descriptor(
            seed,
            now,
            Some(&endpoint),
            &relay_capabilities,
            true,
        );
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
    }

    let local_descriptor = signed_candidate_exclusion_descriptor(
        200,
        now,
        Some("https://192.0.2.5:8422"),
        &relay_capabilities,
        true,
    );
    let descriptors = store.valid_public_descriptors(now, 64);
    let counts = onion_candidate_exclusion_counts(
        &store,
        &descriptors,
        now,
        Some(local_descriptor.node_id()),
        Some(&local_descriptor),
        &[],
        &DiscoveryApiPolicy::default(),
        false,
    );

    assert_eq!(
        counts,
        OnionCandidateExclusionCounts {
            capability_or_feature: 3,
            routeability_unknown_or_stale: 3,
            routeability_failed_or_quarantined: 3,
            missing_kem_or_endpoint: 3,
            anti_affinity_or_policy: 3,
            unclassified: 0,
        }
    );
    let telemetry = counts.into_telemetry(descriptors.len());
    assert_eq!(
        telemetry.status,
        OnionCandidateExclusionTelemetryStatus::Ready
    );
    assert_eq!(telemetry.buckets.capability_or_feature, Some(3));
    assert_eq!(telemetry.buckets.routeability_unknown_or_stale, Some(3));
    assert_eq!(
        telemetry.buckets.routeability_failed_or_quarantined,
        Some(3)
    );
    assert_eq!(telemetry.buckets.missing_kem_or_endpoint, Some(3));
    assert_eq!(telemetry.buckets.anti_affinity_or_policy, Some(3));
}

#[tokio::test]
async fn test_onion_candidate_exclusion_telemetry_is_additive_for_legacy_clients() {
    let app = build_discovery_router(Arc::new(PeerStore::new()), DiscoveryApiPolicy::default());
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
    let mut current: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        current["candidate_exclusion_telemetry"]["contract_version"],
        ONION_CANDIDATE_EXCLUSION_TELEMETRY_CONTRACT_VERSION
    );
    assert_eq!(
        current["candidate_exclusion_telemetry"]["status"],
        "suppressed_small_sample"
    );
    assert_eq!(
        current["candidate_exclusion_telemetry"]["buckets"],
        serde_json::json!({})
    );

    current
        .as_object_mut()
        .unwrap()
        .remove("candidate_exclusion_telemetry");
    let legacy: OnionCandidatesResponse = serde_json::from_value(current).unwrap();
    assert!(legacy.candidate_exclusion_telemetry.is_none());
    assert!(serde_json::to_value(legacy)
        .unwrap()
        .get("candidate_exclusion_telemetry")
        .is_none());
}

#[tokio::test]
async fn test_onion_candidates_endpoint_excludes_private_descriptor_and_preserves_public_control() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let relay_capabilities = [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];

    // [PUBLIC-ONION-CANDIDATE-BOUNDARY 2026-09-01 by Codex] Give the
    // private descriptor the stronger route score and request one result,
    // making the pre-fix public projection deterministically expose it.
    let private_keypair = IdentityKeyPair::from_bytes(&[201; 32]).unwrap();
    let mut private_descriptor = signed_candidate_exclusion_descriptor(
        201,
        now,
        Some("private-candidate.invalid:443"),
        &relay_capabilities,
        true,
    );
    private_descriptor.descriptor.policy.public_discovery = false;
    private_descriptor.descriptor.capacity = NodeCapacity {
        max_sessions: 10_000,
        max_bps: Some(10_000_000_000),
        max_pps: Some(1_000_000),
    };
    let private_descriptor =
        SignedNodeDescriptor::sign(private_descriptor.descriptor, &private_keypair).unwrap();
    let private_node_id = private_descriptor.node_id();
    store.upsert_verified(private_descriptor, now).unwrap();
    store.record_route_forward_success(&private_node_id, now);

    let public_descriptor = signed_candidate_exclusion_descriptor(
        202,
        now,
        Some("public-candidate.invalid:443"),
        &relay_capabilities,
        true,
    );
    let public_node_id = public_descriptor.node_id();
    store.upsert_verified(public_descriptor, now).unwrap();
    store.record_route_forward_success(&public_node_id, now);

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?limit=1")
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

    assert_eq!(parsed.count, 1);
    assert_eq!(parsed.candidates.len(), 1);
    assert_eq!(parsed.candidates[0].node_id, hex::encode(public_node_id));
    assert!(
        parsed.candidates[0]
            .signed_descriptor
            .descriptor
            .policy
            .public_discovery
    );
    assert!(parsed
        .candidates
        .iter()
        .all(|candidate| candidate.node_id != hex::encode(private_node_id)));

    let telemetry = parsed.candidate_exclusion_telemetry.unwrap();
    assert_eq!(
        telemetry.status,
        OnionCandidateExclusionTelemetryStatus::SuppressedSmallSample
    );
    assert_eq!(telemetry.buckets, OnionCandidateExclusionBuckets::default());
}

#[tokio::test]
async fn test_onion_candidates_endpoint_exposes_routeable_kem_relays() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();

    // (a) Routeable ChatRelay + OnionMiddle advertising a KEM key and
    // endpoint -> included.
    let kp = IdentityKeyPair::generate();
    let kem = kp.x25519_public_key_bytes();
    let mut included = NodeDescriptor::new(
        kp.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now + 300,
        "test",
    );
    included.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    included.public_endpoint = Some("relay.example:443".to_string());
    let included = included.with_x25519_kem(kem);
    let included = aeronyx_core::protocol::SignedNodeDescriptor::sign(included, &kp).unwrap();
    let want_node_id = hex::encode(included.node_id());
    let included_node_id = included.node_id();
    store.upsert_verified(included, now).unwrap();
    store.record_route_forward_success(&included_node_id, now);

    // (b) ChatRelay + OnionMiddle WITHOUT a KEM key -> filtered out.
    let kp2 = IdentityKeyPair::generate();
    let mut no_kem = NodeDescriptor::new(
        kp2.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now + 300,
        "test",
    );
    no_kem.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    no_kem.public_endpoint = Some("nokem.example:443".to_string());
    let no_kem = aeronyx_core::protocol::SignedNodeDescriptor::sign(no_kem, &kp2).unwrap();
    store.upsert_verified(no_kem, now).unwrap();

    // (c) KEM-bearing ChatRelay + OnionMiddle without routeability
    // evidence -> filtered out. This keeps clients from building paths
    // through unknown peers while allowing probes to keep learning.
    let kp3 = IdentityKeyPair::generate();
    let mut unknown = NodeDescriptor::new(
        kp3.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now + 300,
        "test",
    );
    unknown.capabilities = vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle];
    unknown.public_endpoint = Some("unknown.example:443".to_string());
    let unknown = unknown.with_x25519_kem(kp3.x25519_public_key_bytes());
    let unknown = aeronyx_core::protocol::SignedNodeDescriptor::sign(unknown, &kp3).unwrap();
    store.upsert_verified(unknown, now).unwrap();

    // (d) Routeable KEM-bearing ChatRelay without OnionMiddle -> filtered
    // out. It can serve a standard encrypted relay path, but must never be
    // counted as a blind multi-hop onion relay.
    let kp4 = IdentityKeyPair::generate();
    let mut single_hop_only = NodeDescriptor::new(
        kp4.public_key_bytes(),
        1,
        now.saturating_sub(1),
        now + 300,
        "test",
    );
    single_hop_only.capabilities = vec![NodeCapability::ChatRelay];
    single_hop_only.public_endpoint = Some("single-hop.example:443".to_string());
    // [ONION-CAPABILITY-GATE 2026-08-02 by Codex] Give the ineligible
    // relay a higher route-capacity score. With `limit=1`, this proves the
    // limit is applied after capability filtering rather than before it.
    single_hop_only.capacity = NodeCapacity {
        max_sessions: 10_000,
        max_bps: Some(10_000_000_000),
        max_pps: Some(1_000_000),
    };
    let single_hop_only = single_hop_only.with_x25519_kem(kp4.x25519_public_key_bytes());
    let single_hop_only =
        aeronyx_core::protocol::SignedNodeDescriptor::sign(single_hop_only, &kp4).unwrap();
    let single_hop_node_id = single_hop_only.node_id();
    store.upsert_verified(single_hop_only, now).unwrap();
    store.record_route_forward_success(&single_hop_node_id, now);

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?limit=1")
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

    // Only the routeable KEM-bearing relay is exposed, with its KEM key for the client.
    assert_eq!(parsed.contract_version, ONION_CANDIDATES_CONTRACT_VERSION);
    assert_eq!(parsed.source, ONION_CANDIDATES_SOURCE);
    assert_eq!(
        parsed.required_capabilities,
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle]
    );
    assert_eq!(parsed.requested_purpose, "message_relay");
    assert!(parsed.requested_purpose_supported);
    assert_eq!(
        parsed.terminal_required_capabilities,
        vec![NodeCapability::ChatRelay, NodeCapability::OnionMiddle]
    );
    assert_eq!(parsed.terminal_candidate_count, 1);
    assert!(parsed.requested_terminal_capability_ready);
    assert_eq!(parsed.selection_policy, ONION_CANDIDATES_SELECTION_POLICY);
    assert_eq!(
        parsed.candidate_verification,
        "signed_node_descriptor_ed25519_v2"
    );
    assert_eq!(
        parsed.refresh_after_seconds,
        ONION_CANDIDATES_REFRESH_AFTER_SECONDS
    );
    assert_eq!(
        parsed.routeability_stale_after_seconds,
        ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS
    );
    assert_eq!(parsed.count, 1);
    assert_eq!(
        parsed.min_candidates_for_two_hop,
        ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
    );
    assert_eq!(parsed.requested_privacy_mode, "enhanced");
    assert_eq!(parsed.requested_hops, 2);
    assert_eq!(parsed.min_candidates_for_requested_hops, 2);
    assert!(!parsed.requested_path_ready);
    assert!(!parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_runtime_proof_required);
    assert!(!parsed.requested_runtime_proof_ready);
    assert!(parsed.requested_restart_continuity_required);
    assert!(!parsed.requested_restart_continuity_ready);
    assert_eq!(parsed.recommended_hops, 1);
    assert!(!parsed.two_hop_ready);
    assert!(parsed.fallback_required);
    assert_eq!(parsed.pool_status, "client_limited");
    assert_eq!(parsed.route_plan, "standard_relay_fallback");
    assert_eq!(parsed.fallback_reason, "client_limit_below_two_hop_minimum");
    assert_eq!(parsed.readiness_reason, "client_limit_blocks_two_hop_pool");
    assert_eq!(
        parsed.next_action,
        "increase candidate limit or use standard encrypted relay fallback"
    );
    assert_eq!(
        parsed.path_selection_strategy,
        "weighted_random_health_ranked_distinct_hops"
    );
    assert_eq!(
        parsed.region_diversity_policy,
        "prefer_distinct_regions_when_available_without_exposing_selected_route"
    );
    assert!(parsed.user_choice_policy.contains("privacy_mode"));
    assert_eq!(parsed.candidates.len(), 1);
    let candidate = &parsed.candidates[0];
    assert_eq!(candidate.node_id, want_node_id);
    assert_eq!(candidate.kem_alg, 1);
    assert_eq!(candidate.kem_public, hex::encode(kem));
    assert_eq!(candidate.public_endpoint, "relay.example:443");
    assert!(candidate.capabilities.contains(&NodeCapability::ChatRelay));
    assert!(candidate
        .capabilities
        .contains(&NodeCapability::OnionMiddle));
    assert_eq!(candidate.selection_weight, 1_000);
    assert_eq!(candidate.region, None);
    assert_eq!(candidate.max_sessions, 0);
    assert!(candidate
        .signed_descriptor
        .verify_at(parsed.generated_at)
        .is_ok());
    assert_eq!(
        candidate.node_id,
        hex::encode(candidate.signed_descriptor.node_id())
    );
    assert_eq!(
        candidate.kem_alg,
        candidate.signed_descriptor.descriptor.kem_alg
    );
    assert_eq!(
        candidate.kem_public,
        hex::encode(
            candidate
                .signed_descriptor
                .descriptor
                .x25519_kem_public()
                .expect("candidate proof must carry its projected X25519 KEM key")
        )
    );
    assert_eq!(
        Some(candidate.public_endpoint.as_str()),
        candidate
            .signed_descriptor
            .descriptor
            .public_endpoint
            .as_deref()
    );
    assert_eq!(
        candidate.capabilities,
        candidate.signed_descriptor.descriptor.capabilities
    );
    assert_eq!(
        candidate.max_sessions,
        candidate.signed_descriptor.descriptor.capacity.max_sessions
    );
    assert_eq!(
        candidate.max_bps,
        candidate.signed_descriptor.descriptor.capacity.max_bps
    );
    assert_eq!(
        candidate.max_pps,
        candidate.signed_descriptor.descriptor.capacity.max_pps
    );
    assert_eq!(
        candidate.region,
        candidate.signed_descriptor.descriptor.policy.region
    );
    let encoded = serde_json::to_value(&parsed).unwrap();
    assert!(encoded["candidates"][0]["signed_descriptor"].is_object());
    assert_eq!(
        encoded["candidate_verification"],
        "signed_node_descriptor_ed25519_v2"
    );
    assert!(parsed.privacy_boundary.contains("fresh routeable"));
    assert!(parsed.privacy_boundary.contains("descriptor proof"));
}

#[tokio::test]
async fn test_onion_candidates_endpoint_marks_two_hop_ready_when_pool_is_sufficient() {
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
    record_stable_path_proof(store.as_ref(), now, 2);

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

    assert_eq!(parsed.count, 2);
    assert_eq!(
        parsed.min_candidates_for_two_hop,
        ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
    );
    assert_eq!(parsed.requested_privacy_mode, "enhanced");
    assert_eq!(parsed.requested_hops, 2);
    assert_eq!(parsed.min_candidates_for_requested_hops, 2);
    assert!(parsed.requested_path_ready);
    assert!(parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_runtime_proof_required);
    assert!(parsed.requested_runtime_proof_ready);
    assert!(parsed.requested_restart_continuity_required);
    assert!(parsed.requested_restart_continuity_ready);
    assert_eq!(parsed.recommended_hops, 2);
    assert!(parsed.two_hop_ready);
    assert!(!parsed.fallback_required);
    assert_eq!(parsed.pool_status, "ready");
    assert_eq!(parsed.route_plan, "two_hop_onion_path");
    assert_eq!(parsed.fallback_reason, "ready");
    assert_eq!(parsed.readiness_reason, "two_hop_candidate_pool_ready");
    assert_eq!(
        parsed.next_action,
        "build a weighted-random onion path with fresh distinct candidates"
    );
    assert_eq!(parsed.candidates[0].selection_weight, 1_000);
    assert_eq!(parsed.candidates[1].selection_weight, 900);
}

#[tokio::test]
async fn test_onion_candidates_endpoint_marks_client_limit_fallback() {
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

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?limit=1")
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

    assert_eq!(parsed.count, 1);
    assert_eq!(parsed.requested_privacy_mode, "enhanced");
    assert_eq!(parsed.requested_hops, 2);
    assert_eq!(parsed.min_candidates_for_requested_hops, 2);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 1);
    assert!(!parsed.two_hop_ready);
    assert!(parsed.fallback_required);
    assert_eq!(parsed.pool_status, "client_limited");
    assert_eq!(parsed.route_plan, "standard_relay_fallback");
    assert_eq!(parsed.fallback_reason, "client_limit_below_two_hop_minimum");
    assert_eq!(parsed.readiness_reason, "client_limit_blocks_two_hop_pool");
    assert_eq!(
        parsed.next_action,
        "increase candidate limit or use standard encrypted relay fallback"
    );
}

#[tokio::test]
async fn test_onion_candidates_endpoint_supports_high_privacy_three_hop_policy() {
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

    assert_eq!(parsed.requested_privacy_mode, "high");
    assert_eq!(parsed.requested_hops, 3);
    assert_eq!(parsed.min_candidates_for_requested_hops, 3);
    assert!(parsed.requested_path_ready);
    assert!(parsed.requested_candidate_pool_ready);
    assert!(parsed.requested_runtime_proof_required);
    assert!(parsed.requested_runtime_proof_ready);
    assert!(parsed.requested_restart_continuity_required);
    assert!(parsed.requested_restart_continuity_ready);
    assert_eq!(parsed.recommended_hops, 3);
    assert!(parsed.two_hop_ready);
    assert!(!parsed.fallback_required);
    assert_eq!(parsed.pool_status, "ready");
    assert_eq!(parsed.route_plan, "three_hop_onion_path");
    assert_eq!(parsed.fallback_reason, "ready");
    assert_eq!(
        parsed.readiness_reason,
        "requested_onion_candidate_pool_ready"
    );
    assert_eq!(
        parsed.next_action,
        "build a weighted-random onion path with fresh distinct candidates"
    );
    assert_eq!(parsed.candidates.len(), 3);
    assert_eq!(parsed.candidates[0].selection_weight, 1_000);
    assert_eq!(parsed.candidates[1].selection_weight, 900);
    assert_eq!(parsed.candidates[2].selection_weight, 800);
}

#[test]
fn test_bounded_onion_pool_preserves_lower_rank_network_diverse_candidate() {
    let candidates = vec![
        onion_candidate_for_test("https://192.0.2.10:8422", 0),
        onion_candidate_for_test("https://192.0.2.20:8422", 1),
        onion_candidate_for_test("https://198.51.100.10:8422", 2),
        onion_candidate_for_test("https://203.0.113.10:8422", 3),
    ];

    let selected = select_onion_candidate_response_pool(candidates, 3, 3);

    // [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] The first three
    // ranked entries cannot form a diverse path because two share an IPv4
    // /24. The bounded pool must retain the fourth candidate, preserve its
    // original health weight, and exclude one collocated relay.
    assert_eq!(selected.len(), 3);
    assert!(onion_candidate_network_diversity_ready(&selected, 3));
    assert!(selected
        .iter()
        .any(|candidate| candidate.public_endpoint == "https://203.0.113.10:8422"));
    assert!(selected
        .iter()
        .any(|candidate| candidate.selection_weight == 700));
    assert_eq!(
        selected
            .iter()
            .filter(|candidate| candidate.public_endpoint.starts_with("https://192.0.2."))
            .count(),
        1
    );
}

#[test]
fn test_bounded_onion_pool_preserves_signed_specialized_terminal() {
    let candidates = vec![
        onion_candidate_for_test("https://192.0.2.10:8422", 0),
        onion_candidate_for_test("https://198.51.100.10:8422", 1),
        onion_candidate_for_test_with_capabilities(
            "https://203.0.113.10:8422",
            2,
            &[NodeCapability::BlindVaultReplica],
        ),
    ];

    let selected = select_onion_candidate_response_pool_with_policy_and_terminal(
        candidates,
        2,
        2,
        &DiscoveryApiPolicy::default(),
        false,
        OnionTerminalRequirement {
            capability: Some(NodeCapability::BlindVaultReplica),
            protocol_features: &[],
        },
    );

    // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] A healthier generic pair
    // must not hide the only signed storage terminal under a small limit.
    assert_eq!(selected.len(), 2);
    assert!(selected.iter().any(|candidate| {
        candidate.public_endpoint == "https://203.0.113.10:8422"
            && candidate
                .signed_descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::BlindVaultReplica)
    }));
    assert!(onion_candidate_route_diversity_ready_for_terminal(
        &selected,
        2,
        &DiscoveryApiPolicy::default(),
        false,
        // [ONION-TERMINAL-TEST-CONTRACT 2026-08-31 by Codex] Readiness
        // consumes the same typed capability/feature contract as selection.
        OnionTerminalRequirement {
            capability: Some(NodeCapability::BlindVaultReplica),
            protocol_features: &[],
        },
    ));
}

#[tokio::test]
async fn test_onion_candidates_route_purpose_fails_closed_without_terminal() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let descriptor = signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422");
    let node_id = descriptor.node_id();
    store.upsert_verified(descriptor, now).unwrap();
    store.record_route_forward_success(&node_id, now);
    let app = build_discovery_router(store, DiscoveryApiPolicy::default());

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?purpose=blind_vault_put&hops=1")
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
    assert_eq!(parsed.requested_purpose, "blind_vault_put");
    assert!(parsed.requested_purpose_supported);
    assert!(parsed
        .terminal_required_capabilities
        .contains(&NodeCapability::BlindVaultReplica));
    assert_eq!(parsed.terminal_candidate_count, 0);
    assert!(!parsed.requested_terminal_capability_ready);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 0);
    assert_eq!(parsed.pool_status, "terminal_limited");
    assert_eq!(parsed.route_plan, "defer_specialized_delivery");
    assert_eq!(
        parsed.fallback_reason,
        "requested_terminal_capability_not_ready"
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates?purpose=unknown_storage_mode")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed.requested_purpose, "unsupported");
    assert!(!parsed.requested_purpose_supported);
    assert!(parsed.terminal_required_capabilities.is_empty());
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.recommended_hops, 0);
    assert_eq!(parsed.pool_status, "unsupported_purpose");
    assert_eq!(parsed.route_plan, "reject_unsupported_purpose");
    assert_eq!(parsed.fallback_reason, "unsupported_route_purpose");

    let empty_app =
        build_discovery_router(Arc::new(PeerStore::new()), DiscoveryApiPolicy::default());
    let response = empty_app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/onion-candidates")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: OnionCandidatesResponse = serde_json::from_slice(&body).unwrap();
    // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] The additive purpose gate
    // must preserve the legacy message-relay diagnosis for an empty pool.
    assert_eq!(parsed.requested_purpose, "message_relay");
    assert!(!parsed.requested_terminal_capability_ready);
    assert_eq!(parsed.pool_status, "empty");
    assert_eq!(parsed.fallback_reason, "no_routeable_candidates");
}

#[tokio::test]
async fn test_production_onion_pool_excludes_candidate_collocated_with_entry() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let local = signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.5:8422");
    let local_node_id = local.node_id();
    store.upsert_verified(local, now).unwrap();

    let remotes = [
        signed_routeable_chat_descriptor(1, now + 300, "https://192.0.2.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://198.51.100.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://203.0.113.10:8422"),
        signed_routeable_chat_descriptor(1, now + 300, "https://198.18.0.10:8422"),
    ];
    for remote in remotes {
        let node_id = remote.node_id();
        store.upsert_verified(remote, now).unwrap();
        store.record_route_forward_success(&node_id, now);
    }
    record_stable_path_proof(store.as_ref(), now, 3);

    let app = build_discovery_router_with_local_entry(
        store,
        DiscoveryApiPolicy::default(),
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

    // [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] The collocated
    // remote is valid and routeable, but a production pool must remove it
    // before count, diversity, and requested-path readiness are computed.
    assert_eq!(parsed.count, 3);
    assert!(parsed.local_entry_network_diversity_enforced);
    assert!(parsed.requested_network_diversity_ready);
    assert!(parsed.requested_path_ready);
    assert!(parsed
        .network_diversity_policy
        .contains("against_local_entry"));
    assert!(parsed
        .candidates
        .iter()
        .all(|candidate| !candidate.public_endpoint.starts_with("https://192.0.2.")));
}

#[tokio::test]
async fn test_production_onion_pool_fails_closed_without_entry_descriptor() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    for endpoint in [
        "https://192.0.2.10:8422",
        "https://198.51.100.10:8422",
        "https://203.0.113.10:8422",
    ] {
        let remote = signed_routeable_chat_descriptor(1, now + 300, endpoint);
        let node_id = remote.node_id();
        store.upsert_verified(remote, now).unwrap();
        store.record_route_forward_success(&node_id, now);
    }
    record_stable_path_proof(store.as_ref(), now, 3);

    let missing_local_node_id = IdentityKeyPair::generate().public_key_bytes();
    let app = build_discovery_router_with_local_entry(
        store,
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::default(),
        None,
        missing_local_node_id,
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
    assert!(!parsed.local_entry_network_diversity_enforced);
    assert!(!parsed.requested_network_diversity_ready);
    assert!(!parsed.requested_path_ready);
    assert_eq!(parsed.pool_status, "diversity_limited");
    assert_eq!(
        parsed.fallback_reason,
        "requested_path_network_diversity_not_ready"
    );
    assert!(parsed
        .network_diversity_policy
        .contains("local_entry_descriptor_unavailable_fail_closed"));
}

#[test]
fn test_onion_admission_requires_and_accepts_signed_proof_persistence() {
    let store = PeerStore::new();
    let now = now_secs();
    let first = signed_routeable_chat_descriptor(1, now + 1_000, "https://continuity-one.example");
    let first_node_id = first.node_id();
    let second = signed_routeable_chat_descriptor(1, now + 1_000, "https://continuity-two.example");
    let second_node_id = second.node_id();

    store.configure_bootstrap_status(true, true, true, 2);
    store
        .upsert_verified_from_source(first, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(second, now, "gossip_snapshot")
        .unwrap();
    store.record_gossip_round(now + 1, 2, 2, 2, None);
    store.record_route_forward_success(&first_node_id, now + 2);
    store.record_route_forward_success(&second_node_id, now + 3);
    for offset in 4..=6 {
        store.record_blind_relay_two_hop_probe_result_with_context(
            now + offset,
            true,
            "onion_terminal_delivered",
            2,
            1,
            2,
            1,
        );
    }

    let local_capabilities = DiscoveryLocalCapabilityStatus::new(true, true, true, true);
    store.record_two_hop_proof_cache_persisted(now + 60, 3, true);
    let before_persist = store.status(now + 7);
    let before_admission = onion_relay_admission_status_value(&before_persist, &local_capabilities);
    assert!(before_persist.two_hop_path_proof_history.stability_ready);
    assert_eq!(before_admission["status"].as_str(), Some("warming"));
    assert_eq!(
        before_admission["warmup_stage"].as_str(),
        Some("proof_restart_continuity")
    );
    assert_eq!(
        before_admission["admission_blockers"][0].as_str(),
        Some("proof_restart_continuity_not_ready")
    );
    assert_eq!(
        before_admission["proof_cache_signed_persistence_ready"].as_bool(),
        Some(false)
    );

    store.record_cache_save_status(now + 8, "success", "snapshot_persisted");
    store.record_two_hop_proof_cache_persisted(now + 8, 3, true);
    // [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] Production persistence
    // records the aggregate generation in the same successful cache round.
    store.record_client_delivery_cache_persisted(now + 8, 0, 1);
    let after_persist = store.status(now + 8);
    let after_admission = onion_relay_admission_status_value(&after_persist, &local_capabilities);

    assert_eq!(after_admission["status"].as_str(), Some("eligible"));
    assert_eq!(after_admission["eligible"].as_bool(), Some(true));
    assert_eq!(
        after_admission["restart_recovery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        after_admission["proof_restart_continuity_source"].as_str(),
        Some("signed_persistence")
    );
    assert_eq!(
        after_admission["proof_cache_signed_persistence_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        after_admission["proof_cache_rollback_protection"].as_str(),
        Some("anchored")
    );

    let summary = discovery_summary_response(now + 8, &after_persist, &local_capabilities);
    assert_eq!(
        summary.two_hop_path_proof["restart_survivable_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        summary.two_hop_path_proof["restart_recovery_basis"].as_str(),
        Some("message_delivery_proof_with_verified_restart_continuity")
    );

    store.record_client_delivery_witness_round(
        now + 9,
        1,
        true,
        1,
        crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
            configured: 1,
            attempted: 1,
            failed: 1,
            ..Default::default()
        },
    );
    let witness_blocked =
        onion_relay_admission_status_value(&store.status(now + 9), &local_capabilities);
    assert_eq!(witness_blocked["status"].as_str(), Some("warming"));
    assert_eq!(
        witness_blocked["proof_restart_continuity_source"].as_str(),
        Some("external_witness_not_ready")
    );
    assert_eq!(
        witness_blocked["proof_cache_external_witness"].as_str(),
        Some("unavailable")
    );

    store.record_client_delivery_witness_round(
        now + 10,
        1,
        true,
        1,
        crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
            configured: 1,
            attempted: 1,
            verified: 1,
            idempotent: 1,
            ..Default::default()
        },
    );
    let witness_verified =
        onion_relay_admission_status_value(&store.status(now + 10), &local_capabilities);
    assert_eq!(witness_verified["status"].as_str(), Some("eligible"));
    assert_eq!(
        witness_verified["proof_cache_external_witness"].as_str(),
        Some("verified")
    );
}
