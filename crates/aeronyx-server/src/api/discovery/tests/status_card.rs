// [ARCH-SPLIT 2026-10-02] Tests moved out of the parent `mod tests`.
// Behavior is unchanged. Names resolve through `use super::*;`.
use super::*;

#[tokio::test]
async fn test_public_status_and_summary_never_export_private_runtime_audit_detail() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let private_prefix = "deadbeef";

    store.record_blind_relay_forwarded(now.saturating_sub(4), 1);
    store.record_audit_event(
        now.saturating_sub(3),
        "blind_relay_route_health",
        "accepted",
        format!("node_prefix={private_prefix} result=success"),
    );
    store.record_blind_relay_quarantine_started(now.saturating_sub(4), "failure_threshold");
    store.record_audit_event(
        now.saturating_sub(3),
        "blind_relay_route_health",
        "rejected",
        format!("node_prefix={private_prefix} result=failure"),
    );

    let app = build_discovery_router(Arc::clone(&store), DiscoveryApiPolicy::default());
    for uri in ["/api/discovery/status", "/api/discovery/summary"] {
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
        assert!(!serialized.contains(private_prefix));
        assert!(!serialized.contains("blind_relay_route_health"));

        if uri.ends_with("/status") {
            let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
            assert_eq!(
                parsed["blind_relay_runtime"]["last_successful_blind_relay"]["action"],
                "blind_relay_forward"
            );
            assert_eq!(
                parsed["blind_relay_runtime"]["last_failed_blind_relay"]["action"],
                "blind_relay_quarantine"
            );
        }
    }
}

#[tokio::test]
async fn test_public_projection_status_filters_private_rows_and_preserves_public_order() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let relay_capabilities = [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];

    let private_keypair = IdentityKeyPair::from_bytes(&[213; 32]).unwrap();
    let mut private_descriptor = signed_candidate_exclusion_descriptor(
        213,
        now,
        Some("private-status.invalid:443"),
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

    let public_high_keypair = IdentityKeyPair::from_bytes(&[214; 32]).unwrap();
    let mut public_high = signed_candidate_exclusion_descriptor(
        214,
        now,
        Some("public-high.invalid:443"),
        &relay_capabilities,
        true,
    );
    public_high.descriptor.capacity = NodeCapacity {
        max_sessions: 512,
        max_bps: Some(1_000_000_000),
        max_pps: Some(100_000),
    };
    let public_high =
        SignedNodeDescriptor::sign(public_high.descriptor, &public_high_keypair).unwrap();
    let public_high_node_id = public_high.node_id();

    let public_low = signed_candidate_exclusion_descriptor(
        215,
        now,
        Some("public-low.invalid:443"),
        &relay_capabilities,
        true,
    );
    let public_low_node_id = public_low.node_id();

    for descriptor in [private_descriptor, public_high, public_low] {
        let node_id = descriptor.node_id();
        store
            .upsert_verified_from_source(descriptor, now, "gossip_announce")
            .unwrap();
        store.record_route_forward_success(&node_id, now);
    }

    let private_prefix = hex::encode(&private_node_id[..4]);
    let public_high_prefix = hex::encode(&public_high_node_id[..4]);
    let public_low_prefix = hex::encode(&public_low_node_id[..4]);
    assert_ne!(private_prefix, public_high_prefix);
    assert_ne!(private_prefix, public_low_prefix);
    assert_ne!(public_high_prefix, public_low_prefix);

    let app = build_discovery_router(store, DiscoveryApiPolicy::default());
    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let peer_store = &parsed["peer_store"];

    assert_eq!(peer_store["snapshot"]["valid_peers"], 3);
    assert_eq!(peer_store["snapshot"]["public_peers"], 2);
    for candidate_group in ["chat_relay", "onion_middle"] {
        let prefixes = peer_store["route_candidates"][candidate_group]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["node_id_prefix"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            prefixes,
            vec![public_high_prefix.as_str(), public_low_prefix.as_str()]
        );
    }

    for rows in [
        peer_store["peer_summary"]["peers"].as_array().unwrap(),
        peer_store["peer_health_summary"]["peers"]
            .as_array()
            .unwrap(),
        peer_store["recent_peer_events"].as_array().unwrap(),
    ] {
        assert!(rows.iter().all(|row| {
            matches!(
                row["node_id_prefix"].as_str(),
                Some(prefix) if prefix == public_high_prefix || prefix == public_low_prefix
            )
        }));
        assert!(rows
            .iter()
            .any(|row| { row["node_id_prefix"].as_str() == Some(public_high_prefix.as_str()) }));
        assert!(rows
            .iter()
            .any(|row| { row["node_id_prefix"].as_str() == Some(public_low_prefix.as_str()) }));
    }

    let planned_paths = &peer_store["route_candidates"]["planned_paths"];
    let path_hops = planned_paths["chat_single_hop"]["hops"]
        .as_array()
        .unwrap()
        .iter()
        .chain(
            planned_paths["chat_two_hop_onion_ready"]["hops"]
                .as_array()
                .unwrap(),
        )
        .collect::<Vec<_>>();
    assert!(path_hops.iter().all(|hop| {
        matches!(
            hop["node_id_prefix"].as_str(),
            Some(prefix) if prefix == public_high_prefix || prefix == public_low_prefix
        )
    }));
    assert!(path_hops
        .iter()
        .any(|hop| { hop["node_id_prefix"].as_str() == Some(public_high_prefix.as_str()) }));

    let serialized = serde_json::to_string(peer_store).unwrap();
    assert!(!serialized.contains(&private_prefix));
    assert!(!serialized.contains("private-status.invalid"));
}

#[test]
fn test_public_projection_status_rejects_prefix_collision_and_unknown_rows() {
    let store = PeerStore::new();
    let now = now_secs();
    let relay_capabilities = [NodeCapability::ChatRelay, NodeCapability::OnionMiddle];

    let public_descriptor = signed_candidate_exclusion_descriptor(
        216,
        now,
        Some("public-collision.invalid:443"),
        &relay_capabilities,
        true,
    );
    let public_node_id = public_descriptor.node_id();
    let private_keypair = IdentityKeyPair::from_bytes(&[217; 32]).unwrap();
    let mut private_descriptor = signed_candidate_exclusion_descriptor(
        217,
        now,
        Some("private-collision.invalid:443"),
        &relay_capabilities,
        true,
    );
    private_descriptor.descriptor.policy.public_discovery = false;
    let private_descriptor =
        SignedNodeDescriptor::sign(private_descriptor.descriptor, &private_keypair).unwrap();
    let private_node_id = private_descriptor.node_id();

    for descriptor in [public_descriptor, private_descriptor] {
        let node_id = descriptor.node_id();
        store.upsert_verified(descriptor, now).unwrap();
        store.record_route_forward_success(&node_id, now);
    }

    let public_prefix = hex::encode(&public_node_id[..4]);
    let private_prefix = hex::encode(&private_node_id[..4]);
    assert_ne!(public_prefix, private_prefix);
    let unknown_prefix = "ffffffff".to_string();
    assert_ne!(public_prefix, unknown_prefix);

    let mut status = store.status(now);
    let public_descriptors = store.valid_public_descriptors(now, usize::MAX);
    for row in &mut status.peer_summary.peers {
        if row.node_id_prefix == private_prefix {
            row.node_id_prefix = public_prefix.clone();
        }
    }
    if let Some(mut unknown) = status.peer_summary.peers.first().cloned() {
        unknown.node_id_prefix = unknown_prefix.clone();
        status.peer_summary.peers.push(unknown);
    }
    for row in &mut status.peer_health_summary.peers {
        if row.node_id_prefix == private_prefix {
            row.node_id_prefix = public_prefix.clone();
        }
    }
    if let Some(mut unknown) = status.peer_health_summary.peers.first().cloned() {
        unknown.node_id_prefix = unknown_prefix.clone();
        status.peer_health_summary.peers.push(unknown);
    }
    for row in &mut status.recent_peer_events {
        if row.node_id_prefix == private_prefix {
            row.node_id_prefix = public_prefix.clone();
        }
    }
    if let Some(mut unknown) = status.recent_peer_events.first().cloned() {
        unknown.node_id_prefix = unknown_prefix.clone();
        status.recent_peer_events.push(unknown);
    }
    for event in &mut status.recent_audit_events {
        event.detail = event.detail.replace(&private_prefix, &public_prefix);
    }
    if let Some(mut unknown) = status
        .recent_audit_events
        .iter()
        .find(|event| event.detail.contains("node_prefix="))
        .cloned()
    {
        unknown.detail = unknown.detail.replace(&public_prefix, &unknown_prefix);
        status.recent_audit_events.push(unknown);
    }
    for candidates in [
        &mut status.route_candidates.privacy_relay,
        &mut status.route_candidates.chat_relay,
        &mut status.route_candidates.onion_middle,
    ] {
        for candidate in candidates.iter_mut() {
            if candidate.node_id_prefix == private_prefix {
                candidate.node_id_prefix = public_prefix.clone();
            }
        }
        if let Some(mut unknown) = candidates.first().cloned() {
            unknown.node_id_prefix = unknown_prefix.clone();
            candidates.push(unknown);
        }
    }
    for path in [
        &mut status.route_candidates.planned_paths.chat_single_hop,
        &mut status
            .route_candidates
            .planned_paths
            .chat_two_hop_onion_ready,
    ] {
        for hop in &mut path.hops {
            if hop.node_id_prefix == private_prefix {
                hop.node_id_prefix = public_prefix.clone();
            }
        }
        if let Some(mut unknown) = path.hops.first().cloned() {
            unknown.node_id_prefix = unknown_prefix.clone();
            path.hops.push(unknown);
        }
    }

    let chat_single_hop_count = status
        .route_candidates
        .planned_paths
        .chat_single_hop
        .hop_count;
    let chat_single_complete = status
        .route_candidates
        .planned_paths
        .chat_single_hop
        .complete;
    let chat_two_hop_count = status
        .route_candidates
        .planned_paths
        .chat_two_hop_onion_ready
        .hop_count;
    let chat_two_hop_complete = status
        .route_candidates
        .planned_paths
        .chat_two_hop_onion_ready
        .complete;

    let sanitized = sanitize_public_peer_store_status(status, &public_descriptors);
    assert_eq!(sanitized.snapshot.valid_peers, 2);
    assert_eq!(sanitized.snapshot.public_peers, 1);
    assert!(sanitized.peer_summary.peers.is_empty());
    assert!(sanitized.peer_health_summary.peers.is_empty());
    assert!(sanitized.recent_peer_events.is_empty());
    assert!(sanitized
        .recent_audit_events
        .iter()
        .all(|event| !event.detail.contains("node_prefix=")));
    assert!(sanitized.route_candidates.privacy_relay.is_empty());
    assert!(sanitized.route_candidates.chat_relay.is_empty());
    assert!(sanitized.route_candidates.onion_middle.is_empty());
    assert!(sanitized
        .route_candidates
        .planned_paths
        .chat_single_hop
        .hops
        .is_empty());
    assert_eq!(
        sanitized
            .route_candidates
            .planned_paths
            .chat_single_hop
            .hop_count,
        chat_single_hop_count
    );
    assert_eq!(
        sanitized
            .route_candidates
            .planned_paths
            .chat_single_hop
            .complete,
        chat_single_complete
    );
    assert!(sanitized
        .route_candidates
        .planned_paths
        .chat_two_hop_onion_ready
        .hops
        .is_empty());
    assert_eq!(
        sanitized
            .route_candidates
            .planned_paths
            .chat_two_hop_onion_ready
            .hop_count,
        chat_two_hop_count
    );
    assert_eq!(
        sanitized
            .route_candidates
            .planned_paths
            .chat_two_hop_onion_ready
            .complete,
        chat_two_hop_complete
    );
}

#[tokio::test]
async fn test_status_endpoint_returns_peer_store_status() {
    let store = Arc::new(PeerStore::new());
    store
        .upsert_verified(signed_descriptor(), now_secs())
        .unwrap();
    let app = build_discovery_router(store, DiscoveryApiPolicy::default());

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_status_endpoint_returns_local_capability_status() {
    let store = Arc::new(PeerStore::new());
    let app = build_discovery_router_with_local_status(
        store,
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::new(true, true, true, true),
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        parsed["local_capabilities"]["status"].as_str(),
        Some("ready")
    );
    assert_eq!(
        parsed["local_capabilities"]["chat_relay_configured"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["local_capabilities"]["blind_relay_endpoint_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["local_capabilities"]["chat_relay_runtime_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["local_capabilities"]["safe_to_advertise_chat_relay"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["local_capabilities"]["advertised_chat_relay_capability"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["local_capabilities"]["capability_config_consistent"].as_bool(),
        Some(true)
    );
}

#[tokio::test]
async fn test_status_endpoint_returns_compact_discovery_readiness_without_private_metadata() {
    let store = Arc::new(PeerStore::new());
    store.record_blind_relay_forwarded(now_secs(), 1);
    let app = build_discovery_router_with_local_status(
        store,
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::new(true, true, true, true),
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(
        parsed["discovery_readiness"]["chat_relay_capability"]["status"].as_str(),
        Some("ready")
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["status"].as_str(),
        Some("forming")
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["stage"].as_str(),
        Some("single_hop_relay_ready")
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["checks_total"].as_u64(),
        Some(4)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["checks_passed"].as_u64(),
        Some(2)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["blind_relay_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["relay_evidence_mode"].as_str(),
        Some("opaque_relay_acceptance")
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["relay_readiness_reason"].as_str(),
        Some("opaque_relay_acceptance_observed")
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["timestamp_rejected"].as_u64(),
        Some(0)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["real_relay_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["accepted_relay_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["synthetic_probe_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["two_hop_path_proof_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["two_hop_probe_succeeded"].as_u64(),
        Some(0)
    );
    assert_eq!(
        parsed["discovery_readiness"]["protocol_foundation"]["privacy_invariant"].as_str(),
        Some("blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status")
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["status"].as_str(),
        Some("ready")
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["runtime_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["evidence_mode"].as_str(),
        Some("opaque_relay_acceptance")
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["readiness_reason"].as_str(),
        Some("opaque_relay_acceptance_observed")
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["real_relay_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["accepted_relay_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["synthetic_probe_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["two_hop_probe_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["accepted_total"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["discovery_readiness"]["blind_relay_runtime"]["timestamp_rejected"].as_u64(),
        Some(0)
    );

    let serialized = serde_json::to_string(&parsed["discovery_readiness"]).unwrap();
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("payload_b64"));
    assert!(!serialized.contains("client_ip"));
    assert_eq!(
        parsed["recovery_anchor"]["contract_version"].as_str(),
        Some("recovery_anchor.v1")
    );
    assert_eq!(parsed["recovery_anchor"]["status"].as_str(), Some("idle"));
}

#[tokio::test]
async fn test_summary_endpoint_returns_public_safe_protocol_summary() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    store.record_blind_relay_forwarded(now, 1);
    store.record_blind_relay_two_hop_probe_result_with_context(
        now,
        true,
        "onion_terminal_delivered",
        4,
        3,
        2,
        1,
    );
    store.record_blind_relay_three_hop_probe_result_with_context(
        now,
        true,
        "onion_terminal_delivered",
        3,
        2,
        3,
        2,
    );
    let app = build_discovery_router_with_local_status(
        store,
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::new(true, true, true, true),
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/summary")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(parsed["source"].as_str(), Some("rust_discovery_summary"));
    assert_eq!(
        parsed["contract_version"].as_str(),
        Some("discovery_summary.v1")
    );
    assert_eq!(
        parsed["protocol_features"]["legacy_descriptor_gossip_v1"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["protocol_features"]["directory_descriptor_proof_gossip_v1"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["protocol_features"]["multihop_delivery_receipt_v1"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["protocol_features"]["purpose_bound_delivery_receipt_v2"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["protocol_features"]["onion_route_purpose_v1"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["protocol_features"]["onion_route_purposes"],
        serde_json::json!([
            "message_relay",
            "blind_vault_put",
            "blind_vault_pull",
            "blind_vault_delete",
            "blind_vault_lease_admission",
            "blind_vault_put_receipt",
            "blind_vault_lease_retire",
            "blind_vault_lease_renewal",
            "blind_vault_lease_status",
            "blind_vault_lease_inventory",
            "anonymous_mailbox_v1"
        ])
    );
    assert_eq!(
        parsed["recovery_anchor"]["contract_version"].as_str(),
        Some("recovery_anchor.v1")
    );
    assert_eq!(parsed["local_capability"]["status"].as_str(), Some("ready"));
    assert_eq!(
        parsed["onion_relay_admission"]["status"].as_str(),
        Some("warming")
    );
    assert_eq!(
        parsed["onion_relay_admission"]["admission_score_percent"].as_u64(),
        Some(40)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["warmup_stage"].as_str(),
        Some("route_pool")
    );
    assert_eq!(
        parsed["onion_relay_admission"]["local_relay_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["recent_path_proof_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["route_pool_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(parsed["blind_relay"]["runtime_ready"].as_bool(), Some(true));
    assert_eq!(
        parsed["blind_relay"]["evidence_mode"].as_str(),
        Some("opaque_relay_acceptance")
    );
    assert_eq!(
        parsed["blind_relay"]["readiness_reason"].as_str(),
        Some("opaque_relay_acceptance_observed")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["proof_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["message_delivery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["recent_message_delivery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["message_delivery_evidence_mode"].as_str(),
        Some("synthetic_onion_message_delivery_probe")
    );
    assert_eq!(parsed["two_hop_path_proof"]["succeeded"].as_u64(), Some(1));
    assert_eq!(
        parsed["three_hop_path_proof"]["proof_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["three_hop_path_proof"]["message_delivery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["three_hop_path_proof"]["succeeded"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["three_hop_path_proof"]["path_shape_counts"]["entry_middle_middle_terminal"]
            .as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["three_hop_path_proof"]["persistence"].as_str(),
        Some("signed_local_cache_with_runtime_revalidation")
    );
    assert_eq!(
        parsed["three_hop_path_proof"]["proof_cache_rollback_protection"].as_str(),
        Some("not_observed")
    );
    assert_eq!(
        parsed["three_hop_path_proof"]["proof_cache_external_witness_required"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["message_delivery_successes"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_window_attempted"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_window_succeeded"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_window_failed"].as_u64(),
        Some(0)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_success_percent"].as_u64(),
        Some(100)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_status"].as_str(),
        Some("warming_up")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["failure_circuit_breaker_active"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["latest_age_bucket"].as_str(),
        Some("fresh")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["latest_reason_bucket"].as_str(),
        Some("onion_terminal_delivered")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["proof_scope"].as_str(),
        Some("message_delivery")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["proof_scope_counts"]["message_delivery"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["consecutive_message_delivery_successes"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["latest_message_delivery_age_seconds"].as_u64(),
        Some(0)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["restart_recovery_configured"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["peer_quorum_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["restart_survivable_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["restart_recovery_basis"].as_str(),
        Some("waiting_for_peer_quorum")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["path_shape_counts"]["entry_middle_terminal"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["candidate_pool_counts"]["forming"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["ttl_shape_counts"]["entry_ttl_2_onward_ttl_1"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["route_governance"]["contract_version"].as_str(),
        Some("route_governance.v1")
    );
    assert_eq!(
        parsed["route_governance"]["status"].as_str(),
        Some("forming")
    );
    assert_eq!(
        parsed["route_governance"]["route_pool_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["route_governance"]["quality_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["route_governance"]["candidates_total"].as_u64(),
        Some(0)
    );
    assert_eq!(parsed["stage"].as_str(), Some("two_hop_path_ready"));
    assert_eq!(
        parsed["privacy_invariant"].as_str(),
        Some("blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status")
    );

    let serialized = serde_json::to_string(&parsed).unwrap();
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("payload_b64"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("client_ip"));
    assert!(!serialized.contains("receiver_pubkey"));
    assert!(!serialized.contains("public_endpoint"));
    assert!(!serialized.contains("selected_hop"));
}

#[test]
fn test_recovery_anchor_status_requires_exact_witness_generation() {
    let store = PeerStore::new();
    let now = now_secs();
    store.record_routeability_cache_rollback_protection(now, 2, "anchored");
    store.record_two_hop_proof_cache_persisted(now, 3, true);
    store.record_three_hop_proof_cache_persisted(now, 3, true);
    store.record_client_delivery_cache_persisted(now, 2, 2);
    store.record_client_delivery_witness_round(
        now,
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

    let mismatched_status = store.status(now + 1);
    let mismatched_anchor = recovery_anchor_status_value(&mismatched_status);
    assert_eq!(mismatched_anchor["status"].as_str(), Some("blocked"));
    assert_eq!(
        mismatched_anchor["local_anchor"]["ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        mismatched_anchor["external_witness"]["status"].as_str(),
        Some("verified")
    );
    assert_eq!(
        mismatched_anchor["external_witness"]["generation_aligned"].as_bool(),
        Some(false)
    );
    assert_eq!(
        mismatched_anchor["external_witness"]["ready"].as_bool(),
        Some(false)
    );
    assert!(!two_hop_proof_restart_continuity(&mismatched_status).ready);

    store.record_client_delivery_witness_round(
        now + 2,
        2,
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
    let aligned_status = store.status(now + 2);
    let aligned_anchor = recovery_anchor_status_value(&aligned_status);
    assert_eq!(aligned_anchor["status"].as_str(), Some("ready"));
    assert_eq!(
        aligned_anchor["external_witness"]["generation_aligned"].as_bool(),
        Some(true)
    );
    assert_eq!(
        aligned_anchor["external_witness"]["ready"].as_bool(),
        Some(true)
    );
    assert!(two_hop_proof_restart_continuity(&aligned_status).ready);
}

#[test]
fn test_optional_external_witness_adverse_evidence_blocks_recovery() {
    let store = PeerStore::new();
    let now = now_secs();
    store.record_routeability_cache_rollback_protection(now, 1, "anchored");
    store.record_two_hop_proof_cache_persisted(now, 3, true);
    store.record_three_hop_proof_cache_persisted(now, 3, true);
    store.record_client_delivery_cache_persisted(now, 0, 1);

    // [EXTERNAL-WITNESS-ADVERSE-GATE 2026-08-21 by Codex] An optional
    // witness may be unavailable without becoming an availability
    // dependency. Once a valid witness reports rollback, conflict, or a
    // generation gap, however, both recovery and relay admission must
    // reject the anchored state even though strict quorum was not enabled.
    store.record_client_delivery_witness_round(
        now + 1,
        1,
        false,
        1,
        crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
            configured: 1,
            attempted: 1,
            failed: 1,
            ..Default::default()
        },
    );
    let unavailable_status = store.status(now + 1);
    let unavailable_anchor = recovery_anchor_status_value(&unavailable_status);
    assert_eq!(unavailable_anchor["status"].as_str(), Some("ready"));
    assert_eq!(
        unavailable_anchor["external_witness"]["required"].as_bool(),
        Some(false)
    );
    assert_eq!(
        unavailable_anchor["external_witness"]["ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        unavailable_anchor["external_witness"]["adverse_evidence"].as_bool(),
        Some(false)
    );
    assert!(two_hop_proof_restart_continuity(&unavailable_status).ready);

    let adverse_rounds = [
        (
            "rollback_detected",
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                stale: 1,
                ..Default::default()
            },
        ),
        (
            "conflict",
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                conflicts: 1,
                ..Default::default()
            },
        ),
        (
            "gap",
            crate::services::peer_store::PeerStoreVerifiedDeliveryWitnessRound {
                configured: 1,
                attempted: 1,
                verified: 1,
                gaps: 1,
                ..Default::default()
            },
        ),
    ];

    for (offset, (expected_status, round)) in adverse_rounds.into_iter().enumerate() {
        let observed_at = now + offset as u64 + 2;
        store.record_client_delivery_witness_round(observed_at, 1, false, 1, round);
        let status = store.status(observed_at);
        let anchor = recovery_anchor_status_value(&status);

        assert_eq!(
            anchor["external_witness"]["status"].as_str(),
            Some(expected_status)
        );
        assert_eq!(anchor["status"].as_str(), Some("blocked"));
        assert_eq!(anchor["ready_for_restore"].as_bool(), Some(false));
        assert_eq!(
            anchor["external_witness"]["adverse_evidence"].as_bool(),
            Some(true)
        );
        assert_eq!(anchor["external_witness"]["ready"].as_bool(), Some(false));
        let continuity = two_hop_proof_restart_continuity(&status);
        assert!(!continuity.ready);
        assert_eq!(continuity.source, "external_witness_not_ready");
    }
}

#[tokio::test]
async fn test_public_card_endpoint_returns_minimal_product_protocol_card() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let middle = signed_routeable_chat_descriptor(1, now + 300, "https://middle.example");
    let middle_node_id = middle.node_id();
    let terminal = signed_routeable_chat_descriptor(1, now + 300, "https://terminal.example");
    let terminal_node_id = terminal.node_id();

    store.upsert_verified(middle, now).unwrap();
    store.upsert_verified(terminal, now).unwrap();
    // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Public
    // `real_relay_ready` now requires current routeability in addition to
    // purpose-bound receipt evidence and network-diverse endpoints.
    store.record_route_forward_success(&middle_node_id, now);
    store.record_route_forward_success(&terminal_node_id, now);
    store.record_purpose_bound_delivery_receipt_capability(&middle_node_id, now);
    store.record_purpose_bound_delivery_receipt_capability(&terminal_node_id, now);
    store.record_blind_relay_terminal(now, 2, 128);
    store.record_blind_relay_forwarded(now, 1);
    store.record_blind_relay_two_hop_probe_result_with_context(
        now,
        true,
        "onion_terminal_delivered",
        4,
        3,
        2,
        1,
    );
    store.record_verified_client_onion_delivery(now);
    let app = build_discovery_router_with_local_status(
        store,
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::new(true, true, true, true),
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/public-card")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(
        parsed["source"].as_str(),
        Some(DISCOVERY_PUBLIC_CARD_SOURCE)
    );
    assert_eq!(
        parsed["contract_version"].as_str(),
        Some(DISCOVERY_PUBLIC_CARD_CONTRACT_VERSION)
    );
    assert_eq!(
        parsed["cards"]["protocol_health"]["label"].as_str(),
        Some("AeroNyx Privacy Protocol")
    );
    assert_eq!(
        parsed["cards"]["verified_mesh"]["label"].as_str(),
        Some("Verified Node Mesh")
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["label"].as_str(),
        Some("Blind Relay")
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["terminal_delivered_count"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["middle_forwarded_count"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["real_relay_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["verified_client_onion_deliveries"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["proof_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["message_delivery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["cards"]["blind_relay"]["message_delivery_evidence_mode"].as_str(),
        Some("verified_client_onion_delivery_receipt")
    );
    assert_eq!(
        parsed["signals"]["permissionless_node_admission"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["signals"]["latest_path_proof_reason_bucket"].as_str(),
        Some("onion_terminal_delivered")
    );
    assert_eq!(
        parsed["display_policy"]["primary_surface"].as_str(),
        Some("show_protocol_health_verified_mesh_and_blind_relay")
    );
    assert_eq!(
        parsed["privacy_invariant"].as_str(),
        Some("blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status")
    );

    let serialized = serde_json::to_string(&parsed).unwrap();
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("payload_b64"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("client_ip"));
    assert!(!serialized.contains("receiver_pubkey"));
    assert!(!serialized.contains("public_endpoint"));
    assert!(!serialized.contains("selected_hop"));
}

#[tokio::test]
async fn test_summary_status_recovers_when_latest_two_hop_message_delivery_is_ready() {
    let store = Arc::new(PeerStore::new());
    let now = now_secs();
    let first = signed_routeable_chat_descriptor(1, now + 1_000, "https://peer-one.example");
    let first_node_id = first.node_id();
    let second = signed_routeable_chat_descriptor(1, now + 1_000, "https://peer-two.example");
    let second_node_id = second.node_id();

    store.configure_bootstrap_status(true, true, true, 2);
    store
        .upsert_verified_from_source(first, now, "gossip_announce")
        .unwrap();
    store
        .upsert_verified_from_source(second, now, "gossip_snapshot")
        .unwrap();
    store.record_gossip_round(now, 2, 2, 2, None);
    store.record_route_forward_success(&first_node_id, now);
    store.record_route_forward_success(&second_node_id, now);

    for _ in 0..6 {
        store.record_blind_relay_two_hop_probe_result_with_context(
            now,
            false,
            "request_error",
            2,
            1,
            2,
            1,
        );
    }
    store.record_blind_relay_two_hop_probe_result_with_context(
        now,
        true,
        "onion_terminal_delivered",
        2,
        1,
        2,
        1,
    );

    let app = build_discovery_router_with_local_status(
        store,
        DiscoveryApiPolicy::default(),
        DiscoveryLocalCapabilityStatus::new(true, true, true, true),
    );

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::GET)
                .uri("/api/discovery/summary")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(parsed["status"].as_str(), Some("ready"));
    assert_eq!(parsed["stage"].as_str(), Some("two_hop_path_ready"));
    assert_eq!(
        parsed["blind_relay"]["evidence_mode"].as_str(),
        Some("synthetic_onion_message_delivery_probe")
    );
    assert_eq!(
        parsed["blind_relay"]["readiness_reason"].as_str(),
        Some("synthetic_onion_message_delivery_probe_ready")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["latest_reason_bucket"].as_str(),
        Some("onion_terminal_delivered")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["recent_message_delivery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["message_delivery_evidence_mode"].as_str(),
        Some("synthetic_onion_message_delivery_probe")
    );
    assert_eq!(
        parsed["peer_mesh"]["chat_two_hop_onion_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["status"].as_str(),
        Some("warming")
    );
    assert_eq!(
        parsed["onion_relay_admission"]["admission_score_percent"].as_u64(),
        Some(60)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["warmup_stage"].as_str(),
        Some("stability_window")
    );
    assert_eq!(
        parsed["onion_relay_admission"]["admission_blockers"][0].as_str(),
        Some("stable_path_proof_not_ready")
    );
    assert_eq!(
        parsed["onion_relay_admission"]["route_pool_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["restart_recovery_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["peer_restart_recovery_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["proof_restart_continuity_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["stable_path_proof_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["two_hop_stability_window_attempted"].as_u64(),
        Some(7)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["two_hop_stability_window_succeeded"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["two_hop_stability_min_attempts"].as_u64(),
        Some(3)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["two_hop_stability_remaining_attempts"].as_u64(),
        Some(0)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["two_hop_stability_success_threshold_percent"].as_u64(),
        Some(80)
    );
    assert_eq!(
        parsed["onion_relay_admission"]["probe_cadence_policy"].as_str(),
        Some("recovery_cadence_until_stability_window_ready_then_low_frequency")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["restart_recovery_configured"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["peer_quorum_ready"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["restart_survivable_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["restart_recovery_basis"].as_str(),
        Some("proof_restart_continuity_not_ready")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_window_attempted"].as_u64(),
        Some(7)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_window_succeeded"].as_u64(),
        Some(1)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_window_failed"].as_u64(),
        Some(6)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_status"].as_str(),
        Some("degraded")
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["stability_ready"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["failure_circuit_breaker_active"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed["two_hop_path_proof"]["latest_age_bucket"].as_str(),
        Some("fresh")
    );

    let serialized = serde_json::to_string(&parsed).unwrap();
    assert!(!serialized.contains("route_id"));
    assert!(!serialized.contains("payload_b64"));
    assert!(!serialized.contains("encrypted_blob"));
    assert!(!serialized.contains("client_ip"));
    assert!(!serialized.contains("receiver_pubkey"));
    assert!(!serialized.contains("public_endpoint"));
    assert!(!serialized.contains("selected_hop"));
}
