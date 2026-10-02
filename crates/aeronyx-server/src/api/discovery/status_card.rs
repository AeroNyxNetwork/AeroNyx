// [ARCH-SPLIT 2026-10-02]
// Public discovery readiness, summary, and card projections.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn recovery_anchor_protection_ready(protection: Option<&str>) -> bool {
    matches!(protection, Some("anchored" | "cache_ahead"))
}

pub(super) fn external_witness_recovery_admission(
    status: &str,
    required: bool,
    cache_generation: u64,
    witness_generation: u64,
) -> ExternalWitnessRecoveryAdmission {
    // [EXTERNAL-WITNESS-ADVERSE-GATE 2026-08-21 by Codex] Optional witnessing
    // makes transport availability advisory; it never makes authenticated
    // rollback, conflict, or generation-gap evidence advisory. Keep this one
    // decision shared by recovery health and path-proof admission so an
    // operator surface cannot fail closed while the data plane stays eligible.
    let adverse_evidence = matches!(status, "rollback_detected" | "conflict" | "gap");
    let generation_aligned = cache_generation != 0 && witness_generation == cache_generation;
    let ready = !adverse_evidence && (!required || (status == "verified" && generation_aligned));

    ExternalWitnessRecoveryAdmission {
        ready,
        adverse_evidence,
        generation_aligned,
    }
}

/// Builds an aggregate view of the local recovery anchor and its witnesses.
///
/// [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] This contract deliberately
/// excludes anchor digests, signatures, file paths, witness identities and
/// endpoints, peer identifiers, routes, messages, clients, and payload data.
/// A witness is ready only when it verified the exact cache generation now
/// represented by the restored or newly persisted local state.
#[must_use]
pub fn recovery_anchor_status_value(status: &PeerStoreStatus) -> serde_json::Value {
    let bootstrap = &status.bootstrap;
    let cache_generation = bootstrap.last_client_delivery_cache_generation;
    let witness_generation = bootstrap.last_client_delivery_witness_generation;
    let witness_status = bootstrap
        .last_client_delivery_witness_status
        .as_deref()
        .unwrap_or("not_observed");
    let witness_required = bootstrap.last_client_delivery_witness_required;
    let witness_admission = external_witness_recovery_admission(
        witness_status,
        witness_required,
        cache_generation,
        witness_generation,
    );
    let routeability_protection = bootstrap
        .last_routeability_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let two_hop_protection = bootstrap
        .last_two_hop_proof_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let three_hop_protection = bootstrap
        .last_three_hop_proof_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let delivery_protection = bootstrap
        .last_client_delivery_cache_rollback_protection
        .as_deref()
        .unwrap_or("not_observed");
    let local_anchor_ready = cache_generation != 0
        && [
            Some(routeability_protection),
            Some(two_hop_protection),
            Some(three_hop_protection),
            Some(delivery_protection),
        ]
        .into_iter()
        .all(recovery_anchor_protection_ready);
    let ready_for_restore = local_anchor_ready && witness_admission.ready;
    let adverse_local_evidence = [
        routeability_protection,
        two_hop_protection,
        three_hop_protection,
        delivery_protection,
    ]
    .into_iter()
    .any(|protection| {
        matches!(
            protection,
            "anchor_invalid" | "anchor_conflict" | "rollback_detected"
        )
    });
    let status_bucket = if cache_generation == 0 {
        "idle"
    } else if ready_for_restore {
        "ready"
    } else if adverse_local_evidence || witness_admission.adverse_evidence || witness_required {
        "blocked"
    } else {
        "attention"
    };
    let next_action = match status_bucket {
        "idle" => "persist the first signed peer-cache recovery generation",
        "ready" => "continue bounded persistence and exact-generation witnessing",
        "blocked" if witness_admission.adverse_evidence => {
            "reject restored readiness and inspect the authenticated witness failure bucket"
        }
        "blocked" if witness_required && !witness_admission.generation_aligned => {
            "obtain the required witness quorum for the current cache generation"
        }
        "blocked" => "inspect aggregate rollback or witness failure buckets before restore",
        _ => "complete local recovery-anchor protection before relying on restored readiness",
    };

    serde_json::json!({
        "contract_version": "recovery_anchor.v1",
        "status": status_bucket,
        "ready_for_restore": ready_for_restore,
        "cache_generation": cache_generation,
        "local_anchor": {
            "ready": local_anchor_ready,
            "routeability": routeability_protection,
            "two_hop_proof": two_hop_protection,
            "three_hop_proof": three_hop_protection,
            "aggregate_delivery": delivery_protection,
        },
        "external_witness": {
            "status": witness_status,
            "required": witness_required,
            "ready": witness_admission.ready,
            "adverse_evidence": witness_admission.adverse_evidence,
            "generation": witness_generation,
            "generation_aligned": witness_admission.generation_aligned,
            "minimum_verified": bootstrap.last_client_delivery_witness_minimum_verified,
            "configured": bootstrap.last_client_delivery_witness_configured,
            "attempted": bootstrap.last_client_delivery_witness_attempted,
            "verified": bootstrap.last_client_delivery_witness_verified,
            "accepted": bootstrap.last_client_delivery_witness_advanced
                .saturating_add(bootstrap.last_client_delivery_witness_idempotent),
            "adverse": bootstrap.last_client_delivery_witness_stale
                .saturating_add(bootstrap.last_client_delivery_witness_conflicts)
                .saturating_add(bootstrap.last_client_delivery_witness_gaps),
            "failed": bootstrap.last_client_delivery_witness_failed,
        },
        "rollback_boundary": "signed_sections_plus_monotonic_local_anchor_with_optional_exact_generation_external_witness",
        "next_action": next_action,
        "privacy_boundary": "aggregate recovery control state only; no anchor digests, signatures, file paths, witness identities or endpoints, peer ids, routes, messages, clients, or payload metadata",
    })
}

pub(super) fn path_proof_restart_continuity(
    status: &PeerStoreStatus,
    generated_at: u64,
    stale_after_seconds: u64,
    authentication: Option<&str>,
    rollback_protection: Option<&str>,
    restored_stability_ready: bool,
    restored_at: Option<u64>,
    restored: u64,
    persisted_stability_ready: bool,
    persisted_at: Option<u64>,
    persisted: u64,
) -> PathProofRestartContinuity {
    let authentication = authentication.unwrap_or("not_observed").to_string();
    let rollback_protection = rollback_protection.unwrap_or("not_observed").to_string();
    let external_witness = status
        .bootstrap
        .last_client_delivery_witness_status
        .as_deref()
        .unwrap_or("not_observed")
        .to_string();
    let external_witness_required = status.bootstrap.last_client_delivery_witness_required;
    let rollback_protection_ready =
        matches!(rollback_protection.as_str(), "anchored" | "cache_ahead");
    // [RECOVERY-ANCHOR-STATUS 2026-08-21 by Codex] Persistence updates the
    // local cache generation before the post-write witness round completes.
    // Never let the prior generation's `verified` bucket authorize this short
    // interval or any later state restored from a mismatched generation.
    let external_witness_admission = external_witness_recovery_admission(
        &external_witness,
        external_witness_required,
        status.bootstrap.last_client_delivery_cache_generation,
        status.bootstrap.last_client_delivery_witness_generation,
    );
    let restore_evidence_fresh = restored_at
        .map(|at| at <= generated_at && generated_at.saturating_sub(at) <= stale_after_seconds)
        .unwrap_or(false);
    let persistence_evidence_fresh = persisted_at
        .map(|at| at <= generated_at && generated_at.saturating_sub(at) <= stale_after_seconds)
        .unwrap_or(false);
    // [PATH-PROOF-ROLLBACK-ANCHOR 2026-08-02 by Codex] A valid section
    // signature proves authorship, not freshness. Restart continuity therefore
    // also requires the monotonic local anchor and, when configured as a
    // startup gate, the existing opaque external witness quorum.
    let authenticated_restore_ready = authentication == "verified"
        && rollback_protection_ready
        && external_witness_admission.ready
        && restored_stability_ready
        && restore_evidence_fresh;
    let signed_persistence_ready = rollback_protection_ready
        && external_witness_admission.ready
        && persisted_stability_ready
        && persistence_evidence_fresh;
    let peer_recovery_configured = status.peer_quorum.restart_recovery_configured;
    let ready = authenticated_restore_ready || signed_persistence_ready;
    let source = match (authenticated_restore_ready, signed_persistence_ready) {
        (true, true) => "verified_restore_and_signed_persistence",
        (true, false) => "verified_restore",
        (false, true) => "signed_persistence",
        (false, false) if authentication == "signature_invalid" => "restore_signature_invalid",
        (false, false) if authentication == "identity_unavailable" => {
            "restore_identity_unavailable"
        }
        (false, false) if authentication == "legacy_descriptor_only" => "legacy_cache",
        (false, false) if !rollback_protection_ready => "rollback_protection_not_ready",
        (false, false) if !external_witness_admission.ready => "external_witness_not_ready",
        _ => "not_ready",
    };

    PathProofRestartContinuity {
        peer_recovery_configured,
        authenticated_restore_ready,
        signed_persistence_ready,
        ready,
        source,
        authentication,
        rollback_protection,
        external_witness,
        external_witness_required,
        restored,
        persisted,
    }
}

pub(super) fn two_hop_proof_restart_continuity(
    status: &PeerStoreStatus,
) -> PathProofRestartContinuity {
    let proof = &status.two_hop_path_proof_history;
    let bootstrap = &status.bootstrap;
    path_proof_restart_continuity(
        status,
        proof.generated_at,
        proof.stale_after_seconds,
        bootstrap.last_two_hop_proof_cache_authentication.as_deref(),
        bootstrap
            .last_two_hop_proof_cache_rollback_protection
            .as_deref(),
        bootstrap.last_two_hop_proof_cache_restored_stability_ready,
        bootstrap.last_two_hop_proof_cache_at,
        bootstrap.last_two_hop_proof_cache_restored,
        bootstrap.last_two_hop_proof_cache_persisted_stability_ready,
        bootstrap.last_two_hop_proof_cache_persisted_at,
        bootstrap.last_two_hop_proof_cache_persisted,
    )
}

pub(super) fn three_hop_proof_restart_continuity(
    status: &PeerStoreStatus,
) -> PathProofRestartContinuity {
    let proof = &status.three_hop_path_proof_history;
    let bootstrap = &status.bootstrap;
    path_proof_restart_continuity(
        status,
        proof.generated_at,
        proof.stale_after_seconds,
        bootstrap
            .last_three_hop_proof_cache_authentication
            .as_deref(),
        bootstrap
            .last_three_hop_proof_cache_rollback_protection
            .as_deref(),
        bootstrap.last_three_hop_proof_cache_restored_stability_ready,
        bootstrap.last_three_hop_proof_cache_at,
        bootstrap.last_three_hop_proof_cache_restored,
        bootstrap.last_three_hop_proof_cache_persisted_stability_ready,
        bootstrap.last_three_hop_proof_cache_persisted_at,
        bootstrap.last_three_hop_proof_cache_persisted,
    )
}

/// Builds the aggregate relay-pool admission contract.
///
/// This is the Rust-side source of truth for nodeboard, backend aggregation,
/// website counters, and AI runbooks that need to know whether this node is
/// mature enough to participate in the permissionless onion relay pool. It is
/// deliberately aggregate-only: it exposes gate booleans, counts, score, and
/// stable reason buckets, but never endpoints, route IDs, selected hops,
/// receiver keys, encrypted payloads, client IPs, DNS, Memory Chain plaintext,
/// private keys, wallet-level traffic, or social graph metadata.
#[must_use]
pub fn onion_relay_admission_status_value(
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> serde_json::Value {
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let proof = &status.two_hop_path_proof_history;
    let local_relay_ready = local_capabilities.safe_to_advertise_chat_relay;
    let route_pool_ready = peer_quorum.routeable_chat_relays
        >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        && peer_quorum.routeable_onion_middle_hops >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;
    let recent_path_proof_ready = proof.proof_ready && !proof.failure_streak_active;
    let stable_path_proof_ready = proof.stability_ready && !proof.failure_circuit_breaker_active;
    let proof_restart_continuity = two_hop_proof_restart_continuity(status);
    let peer_restart_recovery_ready = proof_restart_continuity.peer_recovery_configured;
    let proof_restart_continuity_ready = proof_restart_continuity.ready;
    let restart_recovery_ready = peer_restart_recovery_ready && proof_restart_continuity_ready;
    let stability_remaining_attempts =
        ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS.saturating_sub(proof.stability_window_attempted);
    let checks_total = 5u8;
    let checks_passed = [
        local_relay_ready,
        route_pool_ready,
        recent_path_proof_ready,
        stable_path_proof_ready,
        restart_recovery_ready,
    ]
    .into_iter()
    .filter(|ready| *ready)
    .count() as u8;
    let admission_score_percent =
        ((u16::from(checks_passed) * 100) / u16::from(checks_total)).min(100) as u8;
    let admission_ready = checks_passed == checks_total;
    let attention = proof.failure_circuit_breaker_active
        || proof.failure_streak_active
        || local_capabilities.status == "misconfigured";
    let admission_status = if !local_capabilities.chat_relay_configured {
        "disabled"
    } else if admission_ready {
        "eligible"
    } else if attention {
        "attention"
    } else {
        "warming"
    };
    let warmup_stage = if !local_relay_ready {
        "local_relay"
    } else if !route_pool_ready {
        "route_pool"
    } else if !recent_path_proof_ready {
        "path_proof"
    } else if !stable_path_proof_ready {
        "stability_window"
    } else if !peer_restart_recovery_ready {
        "restart_recovery"
    } else if !proof_restart_continuity_ready {
        "proof_restart_continuity"
    } else {
        "eligible"
    };
    let mut admission_blockers = Vec::new();
    if !local_relay_ready {
        admission_blockers.push("local_relay_not_ready");
    }
    if !route_pool_ready {
        admission_blockers.push("route_pool_not_ready");
    }
    if !recent_path_proof_ready {
        admission_blockers.push("recent_path_proof_not_ready");
    }
    if !stable_path_proof_ready {
        admission_blockers.push("stable_path_proof_not_ready");
    }
    if !peer_restart_recovery_ready {
        admission_blockers.push("restart_recovery_not_ready");
    }
    if !proof_restart_continuity_ready {
        admission_blockers.push("proof_restart_continuity_not_ready");
    }
    let warmup_hint = match warmup_stage {
        "eligible" => "node is eligible for client-selected two-hop onion relay paths".to_string(),
        "local_relay" => {
            "align ChatRelay config, runtime, public peer API, and advertised capability"
                .to_string()
        }
        "route_pool" => {
            "wait for at least two fresh routeable ChatRelay and OnionMiddle peers".to_string()
        }
        "path_proof" => "wait for a fresh accepted entry-middle-terminal proof".to_string(),
        "stability_window" => format!(
            "collect {stability_remaining_attempts} more recent two-hop proof sample(s) and keep success rate at or above {ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT}%"
        ),
        "restart_recovery" => {
            "configure peer cache or seed endpoints before treating admission as restart-resilient"
                .to_string()
        }
        "proof_restart_continuity" => {
            "persist or restore a fresh independently signed stable proof window"
                .to_string()
        }
        _ => "continue warming relay admission gates".to_string(),
    };

    let mut admission = serde_json::json!({
        "status": admission_status,
        "eligible": admission_ready,
        "permissionless": true,
        "admission_score_percent": admission_score_percent,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "admission_blockers": admission_blockers,
        "warmup_stage": warmup_stage,
        "warmup_hint": warmup_hint,
        "local_relay_ready": local_relay_ready,
        "route_pool_ready": route_pool_ready,
        "recent_path_proof_ready": recent_path_proof_ready,
        "stable_path_proof_ready": stable_path_proof_ready,
        "restart_recovery_ready": restart_recovery_ready,
        "routeable_chat_relays": peer_quorum.routeable_chat_relays,
        "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
        "min_routeable_chat_relays": ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        "min_routeable_onion_middle_hops": ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        "two_hop_stability_status": &proof.stability_status,
        "two_hop_stability_ready": proof.stability_ready,
        "two_hop_stability_window_size": proof.stability_window_size,
        "two_hop_stability_window_attempted": proof.stability_window_attempted,
        "two_hop_stability_window_succeeded": proof.stability_window_succeeded,
        "two_hop_stability_window_failed": proof.stability_window_failed,
        "two_hop_stability_min_attempts": ONION_RELAY_ADMISSION_STABILITY_MIN_PROOFS,
        "two_hop_stability_remaining_attempts": stability_remaining_attempts,
        "two_hop_stability_success_percent": proof.stability_success_percent,
        "two_hop_stability_success_threshold_percent": ONION_RELAY_ADMISSION_STABILITY_SUCCESS_PERCENT,
        "latest_path_proof_age_seconds": proof.latest_age_seconds,
        "latest_success_age_seconds": proof.latest_success_age_seconds,
        "latest_message_delivery_age_seconds": proof.latest_message_delivery_age_seconds,
        "failure_circuit_breaker_active": proof.failure_circuit_breaker_active,
        "failure_streak_active": proof.failure_streak_active,
        "routeability_stale_after_seconds": ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        "refresh_after_seconds": ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
        "probe_cadence_policy": "recovery_cadence_until_stability_window_ready_then_low_frequency",
        "client_route_policy": "client_selected_two_hop_onion_when_eligible",
        "network_story_status": &network_story.status,
        "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        "privacy_boundary": "aggregate onion relay admission gates only; no node endpoints, route ids, selected hops, receiver keys, encrypted payloads, client IPs, DNS contents, destinations, Memory Chain plaintext, private keys, wallet-level traffic, or social graph metadata",
    });
    let continuity_fields = serde_json::json!({
        "peer_restart_recovery_ready": peer_restart_recovery_ready,
        "proof_restart_continuity_ready": proof_restart_continuity_ready,
        "proof_restart_continuity_source": proof_restart_continuity.source,
        "proof_cache_authentication": proof_restart_continuity.authentication,
        "proof_cache_rollback_protection": proof_restart_continuity.rollback_protection,
        "proof_cache_external_witness": proof_restart_continuity.external_witness,
        "proof_cache_external_witness_required": proof_restart_continuity.external_witness_required,
        "proof_cache_authenticated_restore_ready": proof_restart_continuity.authenticated_restore_ready,
        "proof_cache_signed_persistence_ready": proof_restart_continuity.signed_persistence_ready,
        "proof_cache_restored_events": proof_restart_continuity.restored,
        "proof_cache_persisted_events": proof_restart_continuity.persisted,
    });

    // [DISCOVERY-PANIC-CONTAINMENT 2026-08-12 by Codex] Keep the established
    // flat response contract without assuming either untyped JSON value has an
    // object shape. A future schema refactor now emits a diagnostic and returns
    // the intact base contract instead of panicking in the request path.
    match (&mut admission, continuity_fields) {
        (serde_json::Value::Object(admission_fields), serde_json::Value::Object(fields)) => {
            admission_fields.extend(fields);
        }
        _ => {
            tracing::error!(
                "onion relay admission continuity fields could not be merged into JSON object"
            );
        }
    }
    admission
}

/// Builds the compact aggregate discovery readiness contract.
///
/// This helper intentionally mirrors only privacy-safe, operator-facing fields
/// from `PeerStoreStatus` and `DiscoveryLocalCapabilityStatus`. It is used by
/// both `/api/discovery/status` and backend heartbeat payloads so nodeboard,
/// public website surfaces, and AI runbooks can depend on one stable JSON shape
/// without parsing the full internal peer store object.
#[must_use]
pub fn discovery_readiness_status_value(
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> serde_json::Value {
    let onion_relay_admission = onion_relay_admission_status_value(status, local_capabilities);
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let route_governance = &status.route_governance;
    let recent_message_delivery_ready = status
        .two_hop_path_proof_history
        .recent_message_delivery_ready
        && !status.two_hop_path_proof_history.failure_streak_active;
    let local_relay_ready = local_capabilities.safe_to_advertise_chat_relay;
    let peer_mesh_ready = peer_quorum.quorum_ready;
    let blind_relay_ready = blind_relay_quality.runtime_ready
        && (blind_relay_quality.quality_ready || recent_message_delivery_ready);
    let two_hop_path_ready = network_story.chat_two_hop_onion_ready
        || blind_relay_quality.two_hop_probe_ready
        || recent_message_delivery_ready;
    let restart_recovery_ready = peer_quorum.restart_recovery_configured;
    let checks_total = 4u8;
    let checks_passed = [
        local_relay_ready,
        peer_mesh_ready,
        blind_relay_ready,
        restart_recovery_ready,
    ]
    .into_iter()
    .filter(|ready| *ready)
    .count() as u8;
    let foundation_status = if checks_passed == checks_total {
        "ready"
    } else if blind_relay_ready && peer_quorum.valid_peers >= peer_quorum.min_valid_peers {
        "live"
    } else if peer_quorum.valid_peers > 0 || blind_relay_quality.runtime_ready {
        "forming"
    } else if !local_capabilities.chat_relay_configured {
        "disabled"
    } else {
        "pending"
    };
    let foundation_stage = if two_hop_path_ready {
        "two_hop_path_ready"
    } else if network_story.chat_single_hop_ready || blind_relay_ready {
        "single_hop_relay_ready"
    } else if peer_quorum.valid_peers > 0 {
        "verified_peer_view"
    } else {
        "bootstrap"
    };
    let foundation_headline = match foundation_status {
        "ready" => "AeroNyx privacy protocol foundation is live",
        "live" => "AeroNyx privacy protocol has live relay evidence",
        "forming" => "AeroNyx nodes are forming a verified relay mesh",
        "disabled" => "AeroNyx privacy protocol discovery is not enabled",
        _ => "AeroNyx privacy protocol is waiting for live peer evidence",
    };
    let foundation_next_action = match foundation_status {
        "ready" => "monitor peer freshness, blind relay probe age, and restart recovery",
        "live" if !restart_recovery_ready => {
            "configure peer cache or seed endpoints before treating relay state as restart-resilient"
        }
        "live" => "wait for peer quorum to become fully ready",
        "forming" => "add or recover verified peers and routeable relay candidates",
        "disabled" => {
            "enable discovery and chat relay capability before advertising protocol readiness"
        }
        _ => "wait for verified peer discovery and the first blind relay runtime check",
    };

    serde_json::json!({
        "protocol_foundation": {
            "status": foundation_status,
            "stage": foundation_stage,
            "headline": foundation_headline,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "local_relay_ready": local_relay_ready,
            "peer_mesh_ready": peer_mesh_ready,
            "blind_relay_ready": blind_relay_ready,
            "restart_recovery_ready": restart_recovery_ready,
            "single_hop_relay_ready": network_story.chat_single_hop_ready,
            "two_hop_onion_ready": two_hop_path_ready,
            "two_hop_path_proof_ready": blind_relay_quality.two_hop_probe_ready,
            "two_hop_message_delivery_ready": status
                .two_hop_path_proof_history
                .message_delivery_ready,
            "two_hop_recent_message_delivery_ready": status
                .two_hop_path_proof_history
                .recent_message_delivery_ready,
            "two_hop_message_delivery_evidence_mode": &status
                .two_hop_path_proof_history
                .message_delivery_evidence_mode,
            "two_hop_probe_attempted": blind_relay_quality.two_hop_probe_attempted,
            "two_hop_probe_succeeded": blind_relay_quality.two_hop_probe_succeeded,
            "two_hop_probe_failed": blind_relay_quality.two_hop_probe_failed,
            "last_two_hop_probe_age_seconds": blind_relay_quality.last_two_hop_probe_age_seconds,
            "last_two_hop_message_delivery_age_seconds": status
                .two_hop_path_proof_history
                .latest_message_delivery_age_seconds,
            "verified_peer_count": peer_quorum.valid_peers,
            "routeable_relay_count": peer_quorum.routeable_chat_relays,
            "last_probe_age_seconds": blind_relay_quality.last_probe_age_seconds,
            "relay_evidence_mode": &blind_relay_quality.evidence_mode,
            "relay_readiness_reason": &blind_relay_quality.readiness_reason,
            "timestamp_rejected": blind_relay_quality.timestamp_rejected,
            "real_relay_ready": blind_relay_quality.real_relay_ready,
            "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
            "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
            "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
            "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
            "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
            "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
            "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
            "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
            "next_action": foundation_next_action,
        },
        "chat_relay_capability": {
            "status": local_capabilities.status,
            "chat_relay_configured": local_capabilities.chat_relay_configured,
            "blind_relay_endpoint_ready": local_capabilities.blind_relay_endpoint_ready,
            "chat_relay_runtime_ready": local_capabilities.chat_relay_runtime_ready,
            "advertised_chat_relay_capability": local_capabilities.advertised_chat_relay_capability,
            "safe_to_advertise_chat_relay": local_capabilities.safe_to_advertise_chat_relay,
            "capability_config_consistent": local_capabilities.capability_config_consistent,
            "advertisement_blockers": &local_capabilities.advertisement_blockers,
            "detail": local_capabilities.detail,
        },
        // [BLIND-VAULT-RUNTIME-ADVERTISEMENT 2026-08-28 by Codex] Keep
        // anonymous storage readiness independent from the required chat relay
        // foundation so an optional full replica does not distort relay SLOs.
        "blind_vault_replica_capability": {
            "configured": local_capabilities.blind_vault_replica_configured,
            "runtime_ready": local_capabilities.blind_vault_runtime_ready,
            "advertised": local_capabilities.advertised_blind_vault_replica_capability,
            "safe_to_advertise": local_capabilities.safe_to_advertise_blind_vault_replica,
            "capability_consistent": local_capabilities.blind_vault_capability_consistent,
            "advertisement_blockers": &local_capabilities.blind_vault_advertisement_blockers,
        },
        "peer_quorum": {
            "status": &peer_quorum.status,
            "quorum_ready": peer_quorum.quorum_ready,
            "valid_peers": peer_quorum.valid_peers,
            "healthy_peers": peer_quorum.healthy_peers,
            "stale_peers": peer_quorum.stale_peers,
            "routeable_chat_relays": peer_quorum.routeable_chat_relays,
            "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
            "restart_recovery_configured": peer_quorum.restart_recovery_configured,
            "relay_foundation_ready": peer_quorum.relay_foundation_ready,
            "next_action": &peer_quorum.next_action,
        },
        "network_story": {
            "status": &network_story.status,
            "headline": &network_story.headline,
            "chat_single_hop_ready": network_story.chat_single_hop_ready,
            "chat_two_hop_onion_ready": network_story.chat_two_hop_onion_ready,
            "routeable_chat_relays": network_story.routeable_chat_relays,
            "routeable_onion_middle_hops": network_story.routeable_onion_middle_hops,
        },
        "route_governance": {
            "contract_version": &route_governance.contract_version,
            "status": &route_governance.status,
            "route_pool_ready": route_governance.route_pool_ready,
            "quality_ready": route_governance.quality_ready,
            "candidates_total": route_governance.candidates_total,
            "routeable_total": route_governance.routeable_total,
            "routeable_chat_relays": route_governance.routeable_chat_relays,
            "routeable_onion_middle_hops": route_governance.routeable_onion_middle_hops,
            "routeable_privacy_relays": route_governance.routeable_privacy_relays,
            "quarantined_total": route_governance.quarantined_total,
            "failing_total": route_governance.failing_total,
            "degraded_total": route_governance.degraded_total,
            "unknown_routeability_total": route_governance.unknown_routeability_total,
            "stale_routeability_total": route_governance.stale_routeability_total,
            "unreachable_total": route_governance.unreachable_total,
            "best_score": route_governance.best_score,
            "worst_score": route_governance.worst_score,
            "average_score": route_governance.average_score,
            "chat_single_hop_ready": route_governance.chat_single_hop_ready,
            "chat_two_hop_onion_ready": route_governance.chat_two_hop_onion_ready,
            "quarantine_threshold": route_governance.quarantine_threshold,
            "quarantine_seconds": route_governance.quarantine_seconds,
            "routeability_stale_after_seconds": route_governance.routeability_stale_after_seconds,
            "next_action": &route_governance.next_action,
        },
        "onion_relay_admission": onion_relay_admission,
        "blind_relay_runtime": {
            "status": &blind_relay_quality.status,
            "runtime_ready": blind_relay_quality.runtime_ready,
            "quality_ready": blind_relay_quality.quality_ready,
            "real_relay_ready": blind_relay_quality.real_relay_ready,
            "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
            "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
            "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
            "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
            "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
            "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
            "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
            "evidence_mode": &blind_relay_quality.evidence_mode,
            "readiness_reason": &blind_relay_quality.readiness_reason,
            "accepted_total": blind_relay_quality.accepted_total,
            "forward_failed": blind_relay_quality.forward_failed,
            "retry_exhausted": blind_relay_quality.retry_exhausted,
            "backpressure_dropped": blind_relay_quality.backpressure_dropped,
            "probe_attempted": blind_relay_quality.probe_attempted,
            "probe_succeeded": blind_relay_quality.probe_succeeded,
            "probe_failed": blind_relay_quality.probe_failed,
            "two_hop_probe_ready": blind_relay_quality.two_hop_probe_ready,
            "two_hop_probe_attempted": blind_relay_quality.two_hop_probe_attempted,
            "two_hop_probe_succeeded": blind_relay_quality.two_hop_probe_succeeded,
            "two_hop_probe_failed": blind_relay_quality.two_hop_probe_failed,
            "timestamp_rejected": blind_relay_quality.timestamp_rejected,
            "protection_active": blind_relay_quality.protection_active,
            "accepted_percent": blind_relay_quality.accepted_percent,
            "last_event_age_seconds": blind_relay_quality.last_event_age_seconds,
            "last_probe_age_seconds": blind_relay_quality.last_probe_age_seconds,
            "last_two_hop_probe_age_seconds": blind_relay_quality.last_two_hop_probe_age_seconds,
            "next_action": &blind_relay_quality.next_action,
        },
        "source": "rust_discovery_readiness",
        "privacy_boundary": "aggregate discovery readiness only; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, or wallet-level traffic",
    })
}

/// Builds the product-facing blind relay runtime observability contract.
///
/// This view intentionally mirrors only aggregate counters and stable event
/// buckets from `PeerStoreStatus`. It exists so nodeboard, backend aggregation,
/// public website status, and AI runbooks can show whether a node is actually
/// participating in the encrypted relay network without reconstructing routes.
/// Never add endpoints, full node IDs, route IDs, encrypted blobs, receiver
/// identities, client IPs, DNS contents, destinations, Memory Chain plaintext,
/// private keys, wallet-level traffic, or social graph metadata here.
#[must_use]
pub fn blind_relay_runtime_status_value(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> serde_json::Value {
    let stats = &status.runtime.blind_relay;
    let quality = &status.blind_relay_quality;
    let proof = &status.two_hop_path_proof_history;
    let peer_quorum = &status.peer_quorum;
    let route_pool_ready = peer_quorum.routeable_chat_relays
        >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        && peer_quorum.routeable_onion_middle_hops >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;

    let mut value = serde_json::json!({
        "generated_at": generated_at,
        "contract_version": "blind_relay_runtime.v1",
        "source": "rust_blind_relay_runtime",
        "status": &quality.status,
        "runtime_ready": quality.runtime_ready,
        "quality_ready": quality.quality_ready,
        "real_relay_ready": quality.real_relay_ready,
        "verified_client_onion_deliveries": quality.verified_client_onion_deliveries,
        "last_verified_client_onion_delivery_age_seconds": quality.last_verified_client_onion_delivery_age_seconds,
        "delivery_receipt_capable_peers": quality.delivery_receipt_capable_peers,
        "accepted_relay_ready": quality.accepted_relay_ready,
        "synthetic_probe_ready": quality.synthetic_probe_ready,
        "evidence_mode": &quality.evidence_mode,
        "readiness_reason": &quality.readiness_reason,
        "onion_candidates": {
            "two_hop_ready": route_pool_ready,
            "routeable_chat_relays": peer_quorum.routeable_chat_relays,
            "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
            "min_candidates_for_two_hop": ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
            "selection_policy": ONION_CANDIDATES_SELECTION_POLICY,
            "refresh_after_seconds": ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
            "routeability_stale_after_seconds": ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        },
        "relay_counters": {
            "received": stats.received,
            "accepted_total": quality.accepted_total,
            "verified_client_onion_deliveries": stats.verified_client_onion_deliveries,
            "terminal_delivered_count": stats.terminal,
            "middle_forwarded_count": stats.forwarded,
            "rejected": stats.rejected,
            "route_ttl_exhausted": stats.ttl_exhausted,
            "forward_failed": stats.forward_failed,
            "retry_attempted": stats.retry_attempted,
            "retry_succeeded": stats.retry_succeeded,
            "retry_exhausted": stats.retry_exhausted,
            "backpressure_dropped": stats.backpressure_dropped,
            "timestamp_rejected": stats.timestamp_rejected,
            "replay_dropped": stats.replay_dropped,
            "loop_detected": stats.loop_detected,
            "rate_limited": stats.rate_limited,
            "quarantined": stats.quarantined,
        },
        "proof_counters": {
            "proof_ready": proof.proof_ready,
            "message_delivery_ready": proof.message_delivery_ready,
            "recent_message_delivery_ready": proof.recent_message_delivery_ready,
            "message_delivery_evidence_mode": &proof.message_delivery_evidence_mode,
            "proof_accepted": proof.succeeded,
            "proof_rejected": proof.failed,
            "proof_attempted": proof.attempted,
            "message_delivery_successes": proof.message_delivery_successes,
            "success_percent": proof.success_percent,
            "stability_ready": proof.stability_ready,
            "stability_status": &proof.stability_status,
            "stability_window_attempted": proof.stability_window_attempted,
            "stability_window_succeeded": proof.stability_window_succeeded,
            "stability_window_failed": proof.stability_window_failed,
            "failure_streak_active": proof.failure_streak_active,
            "failure_circuit_breaker_active": proof.failure_circuit_breaker_active,
            "latest_outcome": &proof.latest_outcome,
            "latest_reason_bucket": &proof.latest_reason_bucket,
            "latest_age_seconds": proof.latest_age_seconds,
            "latest_success_age_seconds": proof.latest_success_age_seconds,
            "latest_failure_age_seconds": proof.latest_failure_age_seconds,
            "latest_message_delivery_age_seconds": proof.latest_message_delivery_age_seconds,
            "proof_scope": &proof.proof_scope,
        },
        "probe_counters": {
            "single_hop_attempted": stats.probe_attempted,
            "single_hop_succeeded": stats.probe_succeeded,
            "single_hop_failed": stats.probe_failed,
            "two_hop_attempted": stats.two_hop_probe_attempted,
            "two_hop_succeeded": stats.two_hop_probe_succeeded,
            "two_hop_failed": stats.two_hop_probe_failed,
            "last_probe_age_seconds": quality.last_probe_age_seconds,
            "last_two_hop_probe_age_seconds": quality.last_two_hop_probe_age_seconds,
        },
        "last_successful_blind_relay": latest_blind_relay_event_value(status, generated_at, true),
        "last_failed_blind_relay": latest_blind_relay_event_value(status, generated_at, false),
        "local_capability": {
            "status": local_capabilities.status,
            "chat_relay_configured": local_capabilities.chat_relay_configured,
            "blind_relay_endpoint_ready": local_capabilities.blind_relay_endpoint_ready,
            "chat_relay_runtime_ready": local_capabilities.chat_relay_runtime_ready,
            "safe_to_advertise_chat_relay": local_capabilities.safe_to_advertise_chat_relay,
        },
        "last_event_age_seconds": quality.last_event_age_seconds,
        "last_accepted_age_seconds": quality.last_accepted_age_seconds,
        "accepted_percent": quality.accepted_percent,
        "next_action": &quality.next_action,
        "privacy_invariant": "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        "privacy_boundary": "aggregate blind relay runtime counters only; no node endpoints, route ids, selected hops, receiver keys, encrypted payloads, client IPs, DNS contents, destinations, Memory Chain plaintext, private keys, wallet-level traffic, or social graph metadata",
    });
    // [AUTHENTICATED-RELAY-PATH-READINESS 2026-08-15 by Codex] Insert these
    // fields after the legacy macro expansion so this already-large stable
    // contract does not require a crate-wide recursion-limit increase.
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "authenticated_delivery_path_ready".to_string(),
            quality.authenticated_delivery_path_ready.into(),
        );
        object.insert(
            "authenticated_delivery_path_reason".to_string(),
            quality.authenticated_delivery_path_reason.clone().into(),
        );
    }
    value
}

pub(super) fn latest_blind_relay_event_value(
    status: &PeerStoreStatus,
    generated_at: u64,
    successful: bool,
) -> serde_json::Value {
    // [PUBLIC-DISCOVERY-RUNTIME-PRIVACY 2026-09-01 by Codex] Runtime status is
    // assembled before the operator-oriented PeerStore payload is projected.
    // Never copy its free-form audit detail into a public response. A closed
    // action/outcome projection preserves aggregate evidence while making
    // route-health prefixes, peer quarantine details, and future unknown
    // audit actions unrepresentable on this surface.
    let event = status.recent_audit_events.iter().rev().find_map(|event| {
        let reason_bucket =
            public_blind_relay_event_reason_bucket(event.action.as_str(), event.outcome.as_str())?;
        let outcome_matches = if successful {
            event.outcome == "accepted"
        } else {
            event.outcome == "rejected" || event.outcome == "limited"
        };
        outcome_matches.then_some((event, reason_bucket))
    });

    match event {
        Some((event, reason_bucket)) => serde_json::json!({
            "at": event.at,
            "age_seconds": generated_at.saturating_sub(event.at),
            "action": &event.action,
            "outcome": &event.outcome,
            "reason_bucket": reason_bucket,
        }),
        None => serde_json::Value::Null,
    }
}

pub(super) fn public_blind_relay_event_reason_bucket(
    action: &str,
    outcome: &str,
) -> Option<&'static str> {
    match (action, outcome) {
        ("blind_relay_terminal", "accepted") => Some("opaque_terminal_delivery_accepted"),
        ("blind_relay_forward", "accepted") => Some("opaque_forward_accepted"),
        ("blind_relay_forward", "rejected") => Some("opaque_forward_rejected"),
        ("blind_relay_retry", "accepted") => Some("relay_retry_succeeded"),
        ("blind_relay_retry", "rejected") => Some("relay_retry_exhausted"),
        ("blind_relay_probe", "accepted") => Some("synthetic_probe_accepted"),
        ("blind_relay_probe", "rejected") => Some("synthetic_probe_rejected"),
        ("blind_relay_two_hop_probe", "accepted") => Some("two_hop_probe_accepted"),
        ("blind_relay_two_hop_probe", "rejected") => Some("two_hop_probe_rejected"),
        ("blind_relay_three_hop_probe", "accepted") => Some("three_hop_probe_accepted"),
        ("blind_relay_three_hop_probe", "rejected") => Some("three_hop_probe_rejected"),
        ("blind_relay_quarantine", "limited") => Some("relay_quarantine_started"),
        ("blind_relay_purpose_bound_receipt_capability", "accepted") => {
            Some("purpose_bound_receipt_capability_accepted")
        }
        ("blind_relay_purpose_bound_receipt_capability", "rejected") => {
            Some("purpose_bound_receipt_capability_rejected")
        }
        ("blind_relay_control_path_proof_evidence", "rejected") => {
            Some("control_path_proof_rejected")
        }
        ("blind_relay_path_proof_evidence", "rejected") => Some("path_proof_rejected"),
        ("blind_relay_client_delivery_receipt", "accepted") => {
            Some("client_delivery_receipt_accepted")
        }
        ("blind_relay_client_delivery_receipt", "rejected") => {
            Some("client_delivery_receipt_rejected")
        }
        _ => None,
    }
}

/// Builds the compact public-safe discovery summary response.
///
/// Keep this helper intentionally narrow. `/api/discovery/status` remains the
/// operator/debug payload, while `/api/discovery/summary` is the small contract
/// for product surfaces that should not receive descriptors, endpoints, full
/// peer ids, route ids, or encrypted payload metadata.
#[must_use]
pub fn discovery_summary_response(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> DiscoverySummaryResponse {
    let readiness = discovery_readiness_status_value(status, local_capabilities);
    let protocol_foundation = &readiness["protocol_foundation"];
    let onion_relay_admission = onion_relay_admission_status_value(status, local_capabilities);
    let blind_relay_runtime =
        blind_relay_runtime_status_value(generated_at, status, local_capabilities);
    let recovery_anchor = recovery_anchor_status_value(status);
    let peer_quorum = &status.peer_quorum;
    let network_story = &status.network_story;
    let blind_relay_quality = &status.blind_relay_quality;
    let two_hop_history = &status.two_hop_path_proof_history;
    let three_hop_history = &status.three_hop_path_proof_history;
    let proof_restart_continuity = two_hop_proof_restart_continuity(status);
    let three_hop_restart_continuity = three_hop_proof_restart_continuity(status);
    let two_hop_restart_survivable_ready = two_hop_history.recent_message_delivery_ready
        && peer_quorum.quorum_ready
        && proof_restart_continuity.peer_recovery_configured
        && proof_restart_continuity.ready;
    let two_hop_restart_recovery_basis = if two_hop_restart_survivable_ready {
        "message_delivery_proof_with_verified_restart_continuity"
    } else if !two_hop_history.recent_message_delivery_ready {
        "waiting_for_fresh_message_delivery_proof"
    } else if !peer_quorum.quorum_ready {
        "waiting_for_peer_quorum"
    } else if !proof_restart_continuity.peer_recovery_configured {
        "restart_recovery_not_configured"
    } else {
        "proof_restart_continuity_not_ready"
    };

    let status_bucket = protocol_foundation["status"]
        .as_str()
        .unwrap_or("forming")
        .to_string();
    let stage_bucket = protocol_foundation["stage"]
        .as_str()
        .unwrap_or("bootstrap")
        .to_string();
    let headline = protocol_foundation["headline"]
        .as_str()
        .unwrap_or("AeroNyx nodes are forming a verified relay mesh")
        .to_string();
    let next_action = protocol_foundation["next_action"]
        .as_str()
        .unwrap_or("monitor verified peer discovery and relay path proof freshness")
        .to_string();
    let mut two_hop_path_proof = serde_json::Map::new();
    two_hop_path_proof.insert(
        "status".to_string(),
        serde_json::json!(&two_hop_history.status),
    );
    two_hop_path_proof.insert(
        "freshness_bucket".to_string(),
        serde_json::json!(&two_hop_history.freshness_bucket),
    );
    two_hop_path_proof.insert(
        "proof_ready".to_string(),
        serde_json::json!(two_hop_history.proof_ready),
    );
    two_hop_path_proof.insert(
        "recent_success_ready".to_string(),
        serde_json::json!(two_hop_history.recent_success_ready),
    );
    two_hop_path_proof.insert(
        "message_delivery_ready".to_string(),
        serde_json::json!(two_hop_history.message_delivery_ready),
    );
    two_hop_path_proof.insert(
        "recent_message_delivery_ready".to_string(),
        serde_json::json!(two_hop_history.recent_message_delivery_ready),
    );
    two_hop_path_proof.insert(
        "message_delivery_evidence_mode".to_string(),
        serde_json::json!(&two_hop_history.message_delivery_evidence_mode),
    );
    two_hop_path_proof.insert(
        "failure_streak_active".to_string(),
        serde_json::json!(two_hop_history.failure_streak_active),
    );
    two_hop_path_proof.insert(
        "retained_events".to_string(),
        serde_json::json!(two_hop_history.retained_events),
    );
    two_hop_path_proof.insert(
        "attempted".to_string(),
        serde_json::json!(two_hop_history.attempted),
    );
    two_hop_path_proof.insert(
        "succeeded".to_string(),
        serde_json::json!(two_hop_history.succeeded),
    );
    two_hop_path_proof.insert(
        "message_delivery_successes".to_string(),
        serde_json::json!(two_hop_history.message_delivery_successes),
    );
    two_hop_path_proof.insert(
        "failed".to_string(),
        serde_json::json!(two_hop_history.failed),
    );
    two_hop_path_proof.insert(
        "success_percent".to_string(),
        serde_json::json!(two_hop_history.success_percent),
    );
    two_hop_path_proof.insert(
        "stability_window_size".to_string(),
        serde_json::json!(two_hop_history.stability_window_size),
    );
    two_hop_path_proof.insert(
        "stability_window_attempted".to_string(),
        serde_json::json!(two_hop_history.stability_window_attempted),
    );
    two_hop_path_proof.insert(
        "stability_window_succeeded".to_string(),
        serde_json::json!(two_hop_history.stability_window_succeeded),
    );
    two_hop_path_proof.insert(
        "stability_window_failed".to_string(),
        serde_json::json!(two_hop_history.stability_window_failed),
    );
    two_hop_path_proof.insert(
        "stability_success_percent".to_string(),
        serde_json::json!(two_hop_history.stability_success_percent),
    );
    two_hop_path_proof.insert(
        "stability_status".to_string(),
        serde_json::json!(&two_hop_history.stability_status),
    );
    two_hop_path_proof.insert(
        "stability_ready".to_string(),
        serde_json::json!(two_hop_history.stability_ready),
    );
    two_hop_path_proof.insert(
        "failure_circuit_breaker_threshold".to_string(),
        serde_json::json!(two_hop_history.failure_circuit_breaker_threshold),
    );
    two_hop_path_proof.insert(
        "failure_circuit_breaker_active".to_string(),
        serde_json::json!(two_hop_history.failure_circuit_breaker_active),
    );
    two_hop_path_proof.insert(
        "latest_age_bucket".to_string(),
        serde_json::json!(&two_hop_history.latest_age_bucket),
    );
    two_hop_path_proof.insert(
        "latest_outcome".to_string(),
        serde_json::json!(&two_hop_history.latest_outcome),
    );
    two_hop_path_proof.insert(
        "latest_reason_bucket".to_string(),
        serde_json::json!(&two_hop_history.latest_reason_bucket),
    );
    two_hop_path_proof.insert(
        "latest_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_age_seconds),
    );
    two_hop_path_proof.insert(
        "latest_success_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_success_age_seconds),
    );
    two_hop_path_proof.insert(
        "latest_failure_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_failure_age_seconds),
    );
    two_hop_path_proof.insert(
        "latest_message_delivery_age_seconds".to_string(),
        serde_json::json!(two_hop_history.latest_message_delivery_age_seconds),
    );
    two_hop_path_proof.insert(
        "consecutive_successes".to_string(),
        serde_json::json!(two_hop_history.consecutive_successes),
    );
    two_hop_path_proof.insert(
        "consecutive_failures".to_string(),
        serde_json::json!(two_hop_history.consecutive_failures),
    );
    two_hop_path_proof.insert(
        "consecutive_message_delivery_successes".to_string(),
        serde_json::json!(two_hop_history.consecutive_message_delivery_successes),
    );
    two_hop_path_proof.insert(
        "path_shape_counts".to_string(),
        serde_json::json!(&two_hop_history.path_shape_counts),
    );
    two_hop_path_proof.insert(
        "candidate_pool_counts".to_string(),
        serde_json::json!(&two_hop_history.candidate_pool_counts),
    );
    two_hop_path_proof.insert(
        "ttl_shape_counts".to_string(),
        serde_json::json!(&two_hop_history.ttl_shape_counts),
    );
    two_hop_path_proof.insert(
        "proof_scope".to_string(),
        serde_json::json!(&two_hop_history.proof_scope),
    );
    two_hop_path_proof.insert(
        "proof_scope_counts".to_string(),
        serde_json::json!(&two_hop_history.proof_scope_counts),
    );
    two_hop_path_proof.insert(
        "restart_recovery_configured".to_string(),
        serde_json::json!(peer_quorum.restart_recovery_configured),
    );
    two_hop_path_proof.insert(
        "peer_quorum_ready".to_string(),
        serde_json::json!(peer_quorum.quorum_ready),
    );
    two_hop_path_proof.insert(
        "restart_survivable_ready".to_string(),
        serde_json::json!(two_hop_restart_survivable_ready),
    );
    two_hop_path_proof.insert(
        "proof_restart_continuity_ready".to_string(),
        serde_json::json!(proof_restart_continuity.ready),
    );
    two_hop_path_proof.insert(
        "proof_restart_continuity_source".to_string(),
        serde_json::json!(proof_restart_continuity.source),
    );
    two_hop_path_proof.insert(
        "proof_cache_authentication".to_string(),
        serde_json::json!(proof_restart_continuity.authentication),
    );
    two_hop_path_proof.insert(
        "proof_cache_rollback_protection".to_string(),
        serde_json::json!(proof_restart_continuity.rollback_protection),
    );
    two_hop_path_proof.insert(
        "proof_cache_external_witness".to_string(),
        serde_json::json!(proof_restart_continuity.external_witness),
    );
    two_hop_path_proof.insert(
        "proof_cache_external_witness_required".to_string(),
        serde_json::json!(proof_restart_continuity.external_witness_required),
    );
    two_hop_path_proof.insert(
        "proof_cache_restored_events".to_string(),
        serde_json::json!(proof_restart_continuity.restored),
    );
    two_hop_path_proof.insert(
        "proof_cache_persisted_events".to_string(),
        serde_json::json!(proof_restart_continuity.persisted),
    );
    two_hop_path_proof.insert(
        "restart_recovery_basis".to_string(),
        serde_json::json!(two_hop_restart_recovery_basis),
    );
    two_hop_path_proof.insert(
        "stale_after_seconds".to_string(),
        serde_json::json!(two_hop_history.stale_after_seconds),
    );
    two_hop_path_proof.insert(
        "next_action".to_string(),
        serde_json::json!(&two_hop_history.next_action),
    );

    DiscoverySummaryResponse {
        generated_at,
        contract_version: "discovery_summary.v1",
        source: "rust_discovery_summary",
        protocol_features: serde_json::json!({
            "legacy_descriptor_gossip_v1": true,
            "directory_descriptor_proof_gossip_v1": true,
            // [THREE-HOP-FEATURE-NEGOTIATION 2026-08-02 by Codex] This is an
            // unsigned transport hint only. A successful path still requires
            // the terminal's signed, route-bound delivery receipt.
            "multihop_delivery_receipt_v1": true,
            // [PURPOSE-BOUND-RECEIPT-NEGOTIATION 2026-08-10 by Codex] Keep v2
            // separate from v1 so a legacy relay cannot be selected for a
            // workload-bound proof merely because it understands ACK framing.
            "purpose_bound_delivery_receipt_v2": true,
            // Hop-local signatures replace deeper success evidence before an
            // ACK travels upstream, keeping terminal topology private.
            "blind_relay_success_receipt_v1": true,
            "source_sealed_terminal_proof_v1": true,
            // [BLIND-VAULT-LARGE-PULL-NEGOTIATION 2026-08-30 by Codex]
            // Unsigned visibility hint only. Route authority comes from the
            // matching feature token in every hop's signed descriptor.
            "blind_vault_large_pull_v1": true,
            // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Canonical values come
            // from aeronyx-core so all implementations negotiate one contract.
            "onion_route_purpose_v1": true,
            "onion_route_purposes": ONION_ROUTE_PURPOSE_VALUES,
            // Operational hint only; route selection trusts the matching
            // token in each terminal's signed descriptor.
            "blind_vault_encrypted_terminal_failure_v1": true,
        }),
        status: status_bucket,
        stage: stage_bucket,
        headline,
        local_capability: serde_json::json!({
            "status": local_capabilities.status,
            "chat_relay_configured": local_capabilities.chat_relay_configured,
            "blind_relay_endpoint_ready": local_capabilities.blind_relay_endpoint_ready,
            "chat_relay_runtime_ready": local_capabilities.chat_relay_runtime_ready,
            "safe_to_advertise_chat_relay": local_capabilities.safe_to_advertise_chat_relay,
            "capability_config_consistent": local_capabilities.capability_config_consistent,
            "advertisement_blockers": &local_capabilities.advertisement_blockers,
            "blind_vault_replica_configured": local_capabilities.blind_vault_replica_configured,
            "blind_vault_runtime_ready": local_capabilities.blind_vault_runtime_ready,
            "advertised_blind_vault_replica_capability": local_capabilities.advertised_blind_vault_replica_capability,
            "safe_to_advertise_blind_vault_replica": local_capabilities.safe_to_advertise_blind_vault_replica,
            "blind_vault_capability_consistent": local_capabilities.blind_vault_capability_consistent,
            "blind_vault_advertisement_blockers": &local_capabilities.blind_vault_advertisement_blockers,
        }),
        peer_mesh: serde_json::json!({
            "status": &peer_quorum.status,
            "quorum_ready": peer_quorum.quorum_ready,
            "valid_peers": peer_quorum.valid_peers,
            "healthy_peers": peer_quorum.healthy_peers,
            "stale_peers": peer_quorum.stale_peers,
            "min_valid_peers": peer_quorum.min_valid_peers,
            "routeable_chat_relays": peer_quorum.routeable_chat_relays,
            "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
            "restart_recovery_configured": peer_quorum.restart_recovery_configured,
            "relay_foundation_ready": peer_quorum.relay_foundation_ready,
            "network_story_status": &network_story.status,
            "chat_single_hop_ready": network_story.chat_single_hop_ready,
            "chat_two_hop_onion_ready": network_story.chat_two_hop_onion_ready,
        }),
        route_governance: serde_json::json!(&status.route_governance),
        blind_relay: serde_json::json!({
            "status": &blind_relay_quality.status,
            "runtime_ready": blind_relay_quality.runtime_ready,
            "quality_ready": blind_relay_quality.quality_ready,
            "real_relay_ready": blind_relay_quality.real_relay_ready,
            "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
            "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
            "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
            "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
            "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
            "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
            "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
            "evidence_mode": &blind_relay_quality.evidence_mode,
            "readiness_reason": &blind_relay_quality.readiness_reason,
            "accepted_total": blind_relay_quality.accepted_total,
            "forward_failed": blind_relay_quality.forward_failed,
            "timestamp_rejected": blind_relay_quality.timestamp_rejected,
            "last_event_age_seconds": blind_relay_quality.last_event_age_seconds,
            "last_probe_age_seconds": blind_relay_quality.last_probe_age_seconds,
            "next_action": &blind_relay_quality.next_action,
        }),
        blind_relay_runtime,
        two_hop_path_proof: serde_json::Value::Object(two_hop_path_proof),
        three_hop_path_proof: serde_json::json!({
            "status": &three_hop_history.status,
            "freshness_bucket": &three_hop_history.freshness_bucket,
            "proof_ready": three_hop_history.proof_ready,
            "recent_success_ready": three_hop_history.recent_success_ready,
            "message_delivery_ready": three_hop_history.message_delivery_ready,
            "recent_message_delivery_ready": three_hop_history.recent_message_delivery_ready,
            "attempted": three_hop_history.attempted,
            "succeeded": three_hop_history.succeeded,
            "failed": three_hop_history.failed,
            "success_percent": three_hop_history.success_percent,
            "latest_age_bucket": &three_hop_history.latest_age_bucket,
            "latest_reason_bucket": &three_hop_history.latest_reason_bucket,
            "latest_message_delivery_age_seconds": three_hop_history.latest_message_delivery_age_seconds,
            "path_shape_counts": &three_hop_history.path_shape_counts,
            "ttl_shape_counts": &three_hop_history.ttl_shape_counts,
            "proof_scope": &three_hop_history.proof_scope,
            "persistence": "signed_local_cache_with_runtime_revalidation",
            "proof_restart_continuity_ready": three_hop_restart_continuity.ready,
            "proof_restart_continuity_source": three_hop_restart_continuity.source,
            "proof_cache_authentication": three_hop_restart_continuity.authentication,
            "proof_cache_rollback_protection": three_hop_restart_continuity.rollback_protection,
            "proof_cache_external_witness": three_hop_restart_continuity.external_witness,
            "proof_cache_external_witness_required": three_hop_restart_continuity.external_witness_required,
            "proof_cache_restored_events": three_hop_restart_continuity.restored,
            "proof_cache_persisted_events": three_hop_restart_continuity.persisted,
            "rollback_boundary": "signed_section_plus_monotonic_local_anchor; whole_host_rollback_is_fail_closed_when_external_witness_is_required",
            "privacy_boundary": &three_hop_history.privacy_boundary,
        }),
        onion_relay_admission,
        recovery_anchor,
        next_action,
        privacy_invariant: "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        privacy_boundary: "aggregate discovery summary only; no signed descriptors, full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, wallet-level traffic, or social graph metadata",
    }
}

/// Builds the smallest product-facing discovery card response.
///
/// `/api/discovery/public-card` is for first-level UX surfaces: website home
/// cards, Nodeboard overview, App status modules, and AI-agent quick health
/// checks. It intentionally compresses detailed route governance into a few
/// stable readiness signals so product surfaces can feel trustworthy without
/// leaking route metadata or overwhelming users with raw diagnostics.
#[must_use]
pub fn discovery_public_card_response(
    generated_at: u64,
    status: &PeerStoreStatus,
    local_capabilities: &DiscoveryLocalCapabilityStatus,
) -> DiscoveryPublicCardResponse {
    let readiness = discovery_readiness_status_value(status, local_capabilities);
    let protocol_foundation = &readiness["protocol_foundation"];
    let onion_relay_admission = onion_relay_admission_status_value(status, local_capabilities);
    let peer_quorum = &status.peer_quorum;
    let blind_relay_quality = &status.blind_relay_quality;
    let relay_stats = &status.runtime.blind_relay;
    let two_hop_history = &status.two_hop_path_proof_history;
    let real_delivery_evidence_seen = blind_relay_quality.verified_client_onion_deliveries > 0;
    let message_delivery_proof_ready =
        blind_relay_quality.real_relay_ready || two_hop_history.proof_ready;
    let message_delivery_ready =
        blind_relay_quality.real_relay_ready || two_hop_history.message_delivery_ready;
    let latest_delivery_proof_age_seconds = if real_delivery_evidence_seen {
        blind_relay_quality.last_verified_client_onion_delivery_age_seconds
    } else {
        two_hop_history.latest_age_seconds
    };

    let status_bucket = protocol_foundation["status"]
        .as_str()
        .unwrap_or("forming")
        .to_string();
    let stage_bucket = protocol_foundation["stage"]
        .as_str()
        .unwrap_or("bootstrap")
        .to_string();
    let headline = protocol_foundation["headline"]
        .as_str()
        .unwrap_or("AeroNyx nodes are forming a verified relay mesh")
        .to_string();
    let next_action = protocol_foundation["next_action"]
        .as_str()
        .unwrap_or("monitor verified peer discovery and relay path proof freshness")
        .to_string();
    let checks_passed = protocol_foundation["checks_passed"].as_u64().unwrap_or(0);
    let checks_total = protocol_foundation["checks_total"]
        .as_u64()
        .unwrap_or(4)
        .max(1);
    let foundation_confidence = ((checks_passed * 100) / checks_total).min(100) as u8;
    let admission_score = onion_relay_admission["admission_score_percent"]
        .as_u64()
        .unwrap_or(0)
        .min(100) as u8;
    let confidence_percent =
        ((u16::from(foundation_confidence) * 2 + u16::from(admission_score)) / 3).min(100) as u8;
    let health_label = match status_bucket.as_str() {
        "ready" => "Live protocol",
        "live" => "Relay evidence live",
        "forming" => "Mesh forming",
        "disabled" => "Not advertising",
        "pending" => "Waiting for peers",
        _ => "Protocol warming",
    };
    let two_hop_ready = protocol_foundation["two_hop_onion_ready"]
        .as_bool()
        .unwrap_or(false);

    DiscoveryPublicCardResponse {
        generated_at,
        contract_version: DISCOVERY_PUBLIC_CARD_CONTRACT_VERSION,
        source: DISCOVERY_PUBLIC_CARD_SOURCE,
        status: status_bucket,
        stage: stage_bucket,
        headline,
        health_label,
        confidence_percent,
        cards: serde_json::json!({
            "protocol_health": {
                "label": "AeroNyx Privacy Protocol",
                "status": protocol_foundation["status"],
                "stage": protocol_foundation["stage"],
                "confidence_percent": confidence_percent,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
                "two_hop_onion_ready": two_hop_ready,
                "restart_recovery_ready": protocol_foundation["restart_recovery_ready"],
            },
            "verified_mesh": {
                "label": "Verified Node Mesh",
                "status": &peer_quorum.status,
                "healthy_peers": peer_quorum.healthy_peers,
                "valid_peers": peer_quorum.valid_peers,
                "routeable_relays": peer_quorum.routeable_chat_relays,
                "routeable_onion_middle_hops": peer_quorum.routeable_onion_middle_hops,
                "restart_recovery_configured": peer_quorum.restart_recovery_configured,
            },
            "blind_relay": {
                "label": "Blind Relay",
                "status": &blind_relay_quality.status,
                "runtime_ready": blind_relay_quality.runtime_ready,
                "real_relay_ready": blind_relay_quality.real_relay_ready,
                "verified_client_onion_deliveries": blind_relay_quality.verified_client_onion_deliveries,
                "last_verified_client_onion_delivery_age_seconds": blind_relay_quality.last_verified_client_onion_delivery_age_seconds,
                "delivery_receipt_capable_peers": blind_relay_quality.delivery_receipt_capable_peers,
                "authenticated_delivery_path_ready": blind_relay_quality.authenticated_delivery_path_ready,
                "authenticated_delivery_path_reason": &blind_relay_quality.authenticated_delivery_path_reason,
                "accepted_relay_ready": blind_relay_quality.accepted_relay_ready,
                "synthetic_probe_ready": blind_relay_quality.synthetic_probe_ready,
                "proof_ready": message_delivery_proof_ready,
                "message_delivery_ready": message_delivery_ready,
                "message_delivery_evidence_mode": &blind_relay_quality.evidence_mode,
                "latest_proof_age_seconds": latest_delivery_proof_age_seconds,
                "terminal_delivered_count": relay_stats.terminal,
                "middle_forwarded_count": relay_stats.forwarded,
            }
        }),
        signals: serde_json::json!({
            "local_relay_ready": protocol_foundation["local_relay_ready"],
            "peer_mesh_ready": protocol_foundation["peer_mesh_ready"],
            "blind_relay_ready": protocol_foundation["blind_relay_ready"],
            "two_hop_onion_ready": two_hop_ready,
            "onion_admission_status": onion_relay_admission["status"],
            "onion_admission_eligible": onion_relay_admission["eligible"],
            "onion_admission_score_percent": admission_score,
            "onion_warmup_stage": onion_relay_admission["warmup_stage"],
            "stable_path_proof_ready": onion_relay_admission["stable_path_proof_ready"],
            "failure_circuit_breaker_active": two_hop_history.failure_circuit_breaker_active,
            "failure_streak_active": two_hop_history.failure_streak_active,
            "latest_path_proof_outcome": &two_hop_history.latest_outcome,
            "latest_path_proof_reason_bucket": &two_hop_history.latest_reason_bucket,
            "latest_path_proof_age_bucket": &two_hop_history.latest_age_bucket,
            "permissionless_node_admission": true,
        }),
        display_policy: serde_json::json!({
            "primary_surface": "show_protocol_health_verified_mesh_and_blind_relay",
            "detail_surface": "link_to_discovery_summary_or_nodeboard_detail_for_diagnostics",
            "recommended_cards": ["protocol_health", "verified_mesh", "blind_relay"],
            "avoid_first_level_fields": [
                "signed_descriptors",
                "raw_route_metadata",
                "raw_audit_events",
                "raw_peer_diagnostics",
                "path_selection_details"
            ],
        }),
        next_action,
        privacy_invariant: "blind_nodes_route_only_opaque_ciphertext_and_aggregate_control_status",
        privacy_boundary: "public protocol card aggregates only; no signed descriptors, full node ids, endpoint URLs, route ids, selected hops, encrypted payloads, receiver identities, client public IPs, DNS contents, destinations, Memory Chain plaintext, voucher secrets, private keys, wallet-level traffic, or social graph metadata",
    }
}

pub(super) fn public_status_audit_event_is_publishable(
    detail: &str,
    membership: &PublicStatusProjectionMembership,
) -> bool {
    let mut remaining = detail;
    while let Some(marker_offset) = remaining.find("node_prefix=") {
        let value = &remaining[marker_offset + "node_prefix=".len()..];
        let Some(prefix) = value.get(..8) else {
            return false;
        };
        if !prefix.bytes().all(|byte| byte.is_ascii_hexdigit())
            || value.as_bytes().get(8).is_some_and(u8::is_ascii_hexdigit)
            || !membership.permits_prefix(prefix)
        {
            return false;
        }
        remaining = &value[8..];
    }
    true
}

/// Removes identity-bearing private or ambiguous rows from the public status.
pub(super) fn sanitize_public_peer_store_status(
    mut status: PeerStoreStatus,
    public_descriptors: &[SignedNodeDescriptor],
) -> PeerStoreStatus {
    let membership = PublicStatusProjectionMembership::from_status(&status, public_descriptors);

    status
        .peer_summary
        .peers
        .retain(|peer| peer.public_discovery && membership.permits_prefix(&peer.node_id_prefix));
    status
        .peer_health_summary
        .peers
        .retain(|peer| membership.permits_prefix(&peer.node_id_prefix));
    status
        .recent_peer_events
        .retain(|event| membership.permits_prefix(&event.node_id_prefix));
    status
        .recent_audit_events
        .retain(|event| public_status_audit_event_is_publishable(&event.detail, &membership));

    for candidates in [
        &mut status.route_candidates.privacy_relay,
        &mut status.route_candidates.chat_relay,
        &mut status.route_candidates.onion_middle,
    ] {
        candidates.retain(|candidate| {
            candidate.public_discovery && membership.permits_prefix(&candidate.node_id_prefix)
        });
    }

    for path in [
        &mut status.route_candidates.planned_paths.chat_single_hop,
        &mut status
            .route_candidates
            .planned_paths
            .chat_two_hop_onion_ready,
    ] {
        path.hops
            .retain(|hop| membership.permits_prefix(&hop.node_id_prefix));
    }

    status
}

pub(super) async fn status_handler(
    State(state): State<DiscoveryApiState>,
) -> Json<DiscoveryStatusResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    let discovery_readiness = discovery_readiness_status_value(&peer_store, &local_capabilities);
    let blind_relay_runtime =
        blind_relay_runtime_status_value(now, &peer_store, &local_capabilities);
    let recovery_anchor = recovery_anchor_status_value(&peer_store);
    let public_descriptors = state.peer_store.valid_public_descriptors(now, usize::MAX);
    let peer_store = sanitize_public_peer_store_status(peer_store, &public_descriptors);
    Json(DiscoveryStatusResponse {
        generated_at: now,
        peer_store,
        policy: DiscoveryPolicyStatus {
            max_snapshot_limit: state.policy.max_snapshot_limit,
            gossip_rate_limit_per_minute: state.policy.gossip_rate_limit_per_minute,
            allow_list_enabled: !state.policy.allowed_peer_ids.is_empty(),
            allowed_peer_count: state.policy.allowed_peer_ids.len(),
            denied_peer_count: state.policy.denied_peer_ids.len(),
            pinned_route_domain_count: state.policy.pinned_route_domains.len(),
            require_pinned_route_domains_for_multi_hop: state
                .policy
                .require_pinned_route_domains_for_multi_hop,
            snapshot_default_public_only: true,
            private_descriptors_hidden_by_default: true,
        },
        local_capabilities,
        discovery_readiness,
        blind_relay_runtime,
        recovery_anchor,
    })
}

pub(super) async fn summary_handler(
    State(state): State<DiscoveryApiState>,
) -> Json<DiscoverySummaryResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    Json(discovery_summary_response(
        now,
        &peer_store,
        &local_capabilities,
    ))
}

pub(super) async fn public_card_handler(
    State(state): State<DiscoveryApiState>,
) -> Json<DiscoveryPublicCardResponse> {
    let now = now_secs();
    let peer_store = state.peer_store.status(now);
    let local_capabilities = state.local_capabilities;
    Json(discovery_public_card_response(
        now,
        &peer_store,
        &local_capabilities,
    ))
}

pub(super) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
