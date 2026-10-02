// [ARCH-SPLIT 2026-10-02]
// Onion candidate selection and the onion-candidates HTTP handler.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

pub(super) fn onion_required_capabilities() -> Vec<NodeCapability> {
    ONION_REQUIRED_CAPABILITIES.to_vec()
}

/// Resolves an optional query value without turning an unknown explicit value
/// into the default message workload.
pub(super) fn onion_route_purpose_from_query(value: Option<&str>) -> Option<OnionRoutePurpose> {
    match value {
        Some(value) => OnionRoutePurpose::from_wire_value(value),
        None => Some(OnionRoutePurpose::MessageRelay),
    }
}

pub(super) fn onion_route_purpose_name(purpose: Option<OnionRoutePurpose>) -> &'static str {
    purpose.map_or("unsupported", OnionRoutePurpose::as_str)
}

pub(super) fn onion_terminal_required_capabilities(
    purpose: Option<OnionRoutePurpose>,
) -> Vec<NodeCapability> {
    let Some(purpose) = purpose else {
        return Vec::new();
    };
    let mut capabilities = onion_required_capabilities();
    if let Some(capability) = purpose.specialized_terminal_capability() {
        capabilities.push(capability);
    }
    capabilities
}

/// Signed SemVer build-metadata tokens required from the selected terminal.
///
/// [ONION-TERMINAL-FEATURE-CONTRACT 2026-08-28 by Codex] Non-Rust Apps, SDKs,
/// and agents receive the same fail-closed protocol contract used by this
/// server instead of reconstructing purpose-specific feature requirements.
pub(super) fn onion_terminal_required_protocol_features(
    purpose: Option<OnionRoutePurpose>,
) -> Vec<String> {
    purpose.map_or_else(Vec::new, |purpose| {
        purpose
            .required_terminal_protocol_features()
            .iter()
            .map(|feature| feature.semver_build_token().to_string())
            .collect()
    })
}

/// Signed feature tokens required from every hop in the selected path.
pub(super) fn onion_path_required_protocol_features(
    purpose: Option<OnionRoutePurpose>,
) -> Vec<String> {
    purpose.map_or_else(Vec::new, |purpose| {
        purpose
            .required_path_protocol_features()
            .iter()
            .map(|feature| feature.semver_build_token().to_string())
            .collect()
    })
}

pub(super) fn onion_terminal_candidate_matches(
    purpose: Option<OnionRoutePurpose>,
    candidate: &OnionRelayCandidate,
) -> bool {
    purpose.is_some()
        && ONION_REQUIRED_CAPABILITIES.iter().all(|required| {
            candidate
                .signed_descriptor
                .descriptor
                .capabilities
                .contains(required)
        })
        && OnionTerminalRequirement::for_purpose(purpose).matches(candidate)
}

/// Observes first-matching exclusion reasons without changing the candidate
/// vector, route ranking, or admission decision.
pub(super) fn onion_candidate_exclusion_counts(
    peer_store: &PeerStore,
    descriptors: &[SignedNodeDescriptor],
    now: u64,
    local_node_id: Option<[u8; 32]>,
    local_descriptor: Option<&SignedNodeDescriptor>,
    path_protocol_features: &[NodeProtocolFeature],
    policy: &DiscoveryApiPolicy,
    pinned_route_domain_required: bool,
) -> OnionCandidateExclusionCounts {
    // [ONION-CANDIDATE-EXCLUSION-TELEMETRY 2026-08-31 by Codex] Match the
    // existing privacy-safe health rows only inside this process. Prefix
    // collisions or bounded health-summary omissions remain unclassified and
    // force `partial`; they are never guessed or serialized.
    let mut routeability_by_prefix: HashMap<String, Option<String>> = HashMap::new();
    for peer in peer_store.peer_health_summary(now).peers {
        match routeability_by_prefix.entry(peer.node_id_prefix) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(Some(peer.routeability_state));
            }
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.insert(None);
            }
        }
    }

    let local_pinned_route_domain =
        local_node_id.and_then(|node_id| policy.pinned_route_domain(&node_id));
    let mut counts = OnionCandidateExclusionCounts::default();
    for descriptor in descriptors {
        let node_id = descriptor.node_id();
        if local_node_id == Some(node_id) {
            counts.anti_affinity_or_policy = counts.anti_affinity_or_policy.saturating_add(1);
            continue;
        }
        if !ONION_REQUIRED_CAPABILITIES
            .iter()
            .all(|required| descriptor.descriptor.capabilities.contains(required))
            || !path_protocol_features
                .iter()
                .all(|required| descriptor.descriptor.advertises_protocol_feature(*required))
        {
            counts.capability_or_feature = counts.capability_or_feature.saturating_add(1);
            continue;
        }
        if descriptor.descriptor.x25519_kem_public().is_none()
            || descriptor.descriptor.public_endpoint.is_none()
        {
            counts.missing_kem_or_endpoint = counts.missing_kem_or_endpoint.saturating_add(1);
            continue;
        }

        let node_prefix = hex::encode(&node_id[..4]);
        match routeability_by_prefix
            .get(&node_prefix)
            .and_then(Option::as_deref)
        {
            Some("unknown" | "stale") => {
                counts.routeability_unknown_or_stale =
                    counts.routeability_unknown_or_stale.saturating_add(1);
                continue;
            }
            Some("unreachable" | "quarantined") => {
                counts.routeability_failed_or_quarantined =
                    counts.routeability_failed_or_quarantined.saturating_add(1);
                continue;
            }
            Some("reachable") => {}
            _ => {
                counts.unclassified = counts.unclassified.saturating_add(1);
                continue;
            }
        }

        if local_descriptor.is_some_and(|local_descriptor| {
            !PeerStore::route_endpoints_are_network_diverse(local_descriptor, descriptor)
        }) {
            counts.anti_affinity_or_policy = counts.anti_affinity_or_policy.saturating_add(1);
            continue;
        }
        let candidate_route_domain = policy.pinned_route_domain(&node_id);
        if local_pinned_route_domain
            .zip(candidate_route_domain)
            .is_some_and(|(local, candidate)| local == candidate)
            || (pinned_route_domain_required && candidate_route_domain.is_none())
        {
            counts.anti_affinity_or_policy = counts.anti_affinity_or_policy.saturating_add(1);
        }
    }
    counts
}

/// `GET /api/discovery/onion-candidates` — health-ranked onion relay candidates
/// for client-side path selection.
///
/// Each candidate advertises a KEM public key (so the client can build an onion
/// layer addressed to it) and a reachable public endpoint. Only signed, public
/// node discovery metadata is exposed — never client traffic, route ids, or
/// payloads. Candidates without a KEM key or a public endpoint are filtered out
/// (they cannot serve as an onion hop). Candidates also need fresh routeability
/// evidence from local probes or successful forwards; signed descriptors prove
/// identity/capability, but they do not prove the endpoint is currently usable.
/// Because the KEM key rotates on the relay's onion-key schedule, clients should
/// fetch fresh candidates rather than caching keys for long periods.
pub(super) async fn onion_candidates_handler(
    State(state): State<DiscoveryApiState>,
    Query(query): Query<OnionCandidatesQuery>,
) -> Json<OnionCandidatesResponse> {
    let now = now_secs();
    let limit = state.policy.snapshot_limit(query.limit);
    let requested_purpose = onion_route_purpose_from_query(query.purpose.as_deref());
    let terminal_requirement = OnionTerminalRequirement::for_purpose(requested_purpose);
    let path_protocol_features =
        requested_purpose.map_or(&[][..], OnionRoutePurpose::required_path_protocol_features);
    let requested_privacy_mode = OnionPrivacyMode::from_query(query.privacy_mode.as_deref());
    let requested_hops = normalize_requested_hops(requested_privacy_mode, query.hops);
    let pinned_route_domain_required =
        requested_hops >= 2 && state.policy.require_pinned_route_domains_for_multi_hop;
    let local_pinned_route_domain = state
        .local_node_id
        .and_then(|node_id| state.policy.pinned_route_domain(&node_id));
    let local_descriptor = state
        .local_node_id
        .and_then(|node_id| state.peer_store.get_valid(&node_id, now));
    let diagnostic_descriptors = state
        .peer_store
        .valid_public_descriptors(now, state.policy.max_snapshot_limit);
    let candidate_exclusion_telemetry = onion_candidate_exclusion_counts(
        state.peer_store.as_ref(),
        &diagnostic_descriptors,
        now,
        state.local_node_id,
        local_descriptor.as_ref(),
        path_protocol_features,
        &state.policy,
        pinned_route_domain_required,
    )
    .into_telemetry(diagnostic_descriptors.len());
    // [ONION-CAPABILITY-GATE 2026-08-02 by Codex] Query the bounded policy
    // pool first, then apply every onion-hop eligibility rule before the
    // client limit and ranking. Limiting earlier can let ineligible high-rank
    // ChatRelay peers hide valid OnionMiddle relays below them.
    let eligible_candidates: Vec<OnionRelayCandidate> = state
        .peer_store
        .route_candidates_with_capability(
            NodeCapability::ChatRelay,
            now,
            state.policy.max_snapshot_limit,
        )
        .into_iter()
        // [PUBLIC-ONION-CANDIDATE-BOUNDARY 2026-09-01 by Codex] The internal
        // route selector also serves non-public callers, so enforce descriptor
        // visibility at this public projection boundary before exposing proof,
        // endpoint, KEM, or identity fields. Filtering preserves relative rank.
        .filter(|descriptor| descriptor.descriptor.policy.public_discovery)
        .filter_map(|descriptor| {
            let node_id = descriptor.node_id();
            if state.local_node_id == Some(node_id) {
                return None;
            }
            if !descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::OnionMiddle)
            {
                return None;
            }
            if !path_protocol_features
                .iter()
                .all(|required| descriptor.descriptor.advertises_protocol_feature(*required))
            {
                return None;
            }
            if !state.peer_store.is_routeable_now(&node_id, now) {
                return None;
            }
            let kem_public = descriptor.descriptor.x25519_kem_public()?;
            let public_endpoint = descriptor.descriptor.public_endpoint.clone()?;
            // [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] A first remote
            // hop collocated with the entry weakens the route before pairwise
            // candidate diversity is considered. Missing/malformed entry or
            // candidate endpoints fail this production gate closed.
            if local_descriptor.as_ref().is_some_and(|local_descriptor| {
                !PeerStore::route_endpoints_are_network_diverse(local_descriptor, &descriptor)
            }) {
                return None;
            }
            // [PINNED-ROUTE-DOMAINS 2026-08-03 by Codex] Known same-domain
            // entry/candidate pairs are excluded even before strict rollout.
            // Strict mode additionally requires complete remote coverage; the
            // local-entry coverage check remains a separate fail-closed gate.
            let candidate_route_domain = state.policy.pinned_route_domain(&node_id);
            if local_pinned_route_domain
                .zip(candidate_route_domain)
                .is_some_and(|(local, candidate)| local == candidate)
            {
                return None;
            }
            if pinned_route_domain_required && candidate_route_domain.is_none() {
                return None;
            }
            Some((descriptor, kem_public, public_endpoint))
        })
        .enumerate()
        .map(|(rank, (descriptor, kem_public, public_endpoint))| {
            let capacity = descriptor.descriptor.capacity.clone();
            OnionRelayCandidate {
                node_id: hex::encode(descriptor.node_id()),
                kem_alg: descriptor.descriptor.kem_alg,
                kem_public: hex::encode(kem_public),
                public_endpoint,
                capabilities: descriptor.descriptor.capabilities.clone(),
                selection_weight: onion_candidate_selection_weight(rank),
                region: descriptor.descriptor.policy.region.clone(),
                max_sessions: capacity.max_sessions,
                max_bps: capacity.max_bps,
                max_pps: capacity.max_pps,
                signed_descriptor: descriptor,
            }
        })
        .collect();
    // [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] The health-ranked
    // weights above belong to the full eligible pool and remain unchanged.
    // Preserve a diverse requested path before applying a small response
    // limit, then let the client independently choose the actual route.
    let eligible_candidate_count = eligible_candidates.len();
    let candidates = select_onion_candidate_response_pool_with_policy_and_terminal(
        eligible_candidates,
        limit,
        requested_hops as usize,
        &state.policy,
        pinned_route_domain_required,
        terminal_requirement,
    );
    let two_hop_ready = candidates.len() >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES;
    let min_candidates_for_requested_hops = requested_hops as usize;
    let terminal_candidate_count = candidates
        .iter()
        .filter(|candidate| onion_terminal_candidate_matches(requested_purpose, candidate))
        .count();
    let requested_terminal_capability_ready = terminal_candidate_count > 0;
    // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] The legacy message purpose has
    // no terminal role beyond the base candidate contract. Its empty-pool
    // state must therefore remain `no_routeable_candidates`; only a purpose
    // with an additional signed terminal capability uses the new terminal
    // readiness gate.
    let terminal_capability_gate_ready =
        !terminal_requirement.is_specialized() || requested_terminal_capability_ready;
    let purpose_admission = OnionPurposeAdmission {
        supported: requested_purpose.is_some(),
        terminal_capability_ready: terminal_capability_gate_ready,
    };
    // [ONION-ENTRY-ANTI-AFFINITY 2026-08-03 by Codex] Legacy builders have no
    // entry context and retain their historical pairwise behavior. Production
    // builders inject an id and fail multi-hop readiness closed until its
    // signed descriptor can be resolved and used for entry anti-affinity.
    let local_entry_context_ready = state.local_node_id.is_none() || local_descriptor.is_some();
    // [ONION-PURPOSE-COMPATIBILITY 2026-09-13 by Codex] Generic purposes
    // with no path-feature contract do not require a local descriptor. New
    // specialized purposes still fail closed on their signed feature set.
    let local_path_protocol_ready = path_protocol_features.is_empty()
        || state.local_node_id.is_none()
        || local_descriptor.as_ref().is_some_and(|descriptor| {
            path_protocol_features
                .iter()
                .all(|required| descriptor.descriptor.advertises_protocol_feature(*required))
        });
    let requested_network_diversity_ready = local_entry_context_ready
        && onion_candidate_route_diversity_ready_for_terminal(
            &candidates,
            min_candidates_for_requested_hops,
            &DiscoveryApiPolicy::default(),
            false,
            terminal_requirement,
        );
    let local_entry_pinned_route_domain_enforced =
        state.local_node_id.is_some() && local_pinned_route_domain.is_some();
    let requested_pinned_route_domain_ready = !pinned_route_domain_required
        || (local_entry_pinned_route_domain_enforced
            && onion_candidate_route_diversity_ready_for_terminal(
                &candidates,
                min_candidates_for_requested_hops,
                &state.policy,
                true,
                terminal_requirement,
            ));
    let peer_status = state.peer_store.status(now);
    let requested_path_gates = onion_requested_path_gates(
        &peer_status,
        requested_hops,
        OnionCandidateAdmissionInput {
            purpose: purpose_admission,
            candidate_pool_ready: local_path_protocol_ready
                && limit >= min_candidates_for_requested_hops
                && eligible_candidate_count >= min_candidates_for_requested_hops,
            network_diversity_ready: requested_network_diversity_ready,
            pinned_route_domain: OnionRequirementGate {
                required: pinned_route_domain_required,
                ready: requested_pinned_route_domain_ready,
            },
        },
    );
    let requested_path_ready = requested_path_gates.ready();
    let two_hop_network_diversity_ready = local_entry_context_ready
        && onion_candidate_route_diversity_ready_for_terminal(
            &candidates,
            ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
            &DiscoveryApiPolicy::default(),
            false,
            terminal_requirement,
        );
    let two_hop_fallback_ready = requested_hops > 2
        && onion_requested_path_gates(
            &peer_status,
            2,
            OnionCandidateAdmissionInput {
                purpose: purpose_admission,
                candidate_pool_ready: local_path_protocol_ready
                    && limit >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
                    && eligible_candidate_count >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
                network_diversity_ready: two_hop_network_diversity_ready,
                pinned_route_domain: OnionRequirementGate {
                    required: state.policy.require_pinned_route_domains_for_multi_hop,
                    ready: !state.policy.require_pinned_route_domains_for_multi_hop
                        || (local_entry_pinned_route_domain_enforced
                            && onion_candidate_route_diversity_ready_for_terminal(
                                &candidates,
                                ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
                                &state.policy,
                                true,
                                terminal_requirement,
                            )),
                },
            },
        )
        .ready();
    let recommended_hops = if requested_purpose.is_some() && terminal_capability_gate_ready {
        recommended_onion_hops(
            candidates.len(),
            requested_hops,
            requested_path_ready,
            two_hop_fallback_ready,
        )
    } else {
        0
    };
    let fallback_reason = onion_candidate_fallback_reason(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
        requested_path_gates,
    );
    let pool_status = onion_candidate_pool_status(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
        requested_path_gates,
    );
    let route_plan = onion_candidate_route_plan(
        requested_path_ready,
        requested_hops,
        recommended_hops,
        fallback_reason,
    );
    let readiness_reason = onion_candidate_readiness_reason(
        candidates.len(),
        limit,
        min_candidates_for_requested_hops,
        requested_path_gates,
    );
    let next_action =
        onion_candidate_next_action(requested_path_ready, recommended_hops, fallback_reason);

    Json(OnionCandidatesResponse {
        generated_at: now,
        contract_version: ONION_CANDIDATES_CONTRACT_VERSION.to_string(),
        source: ONION_CANDIDATES_SOURCE.to_string(),
        required_capabilities: onion_required_capabilities(),
        requested_purpose: onion_route_purpose_name(requested_purpose).to_string(),
        requested_purpose_supported: requested_purpose.is_some(),
        terminal_required_capabilities: onion_terminal_required_capabilities(requested_purpose),
        terminal_required_protocol_features: onion_terminal_required_protocol_features(
            requested_purpose,
        ),
        path_required_protocol_features: onion_path_required_protocol_features(requested_purpose),
        terminal_candidate_count,
        requested_terminal_capability_ready,
        count: candidates.len(),
        candidate_exclusion_telemetry: Some(candidate_exclusion_telemetry),
        min_candidates_for_two_hop: ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
        two_hop_ready,
        requested_privacy_mode: requested_privacy_mode.as_str().to_string(),
        requested_hops,
        min_candidates_for_requested_hops,
        requested_path_ready,
        requested_candidate_pool_ready: requested_path_gates.candidate_pool_ready,
        requested_network_diversity_required: requested_path_gates.network_diversity_required,
        requested_network_diversity_ready: requested_path_gates.network_diversity_ready,
        local_entry_network_diversity_enforced: local_descriptor.is_some(),
        requested_pinned_route_domain_required: requested_path_gates
            .pinned_route_domain_required,
        requested_pinned_route_domain_ready: requested_path_gates.pinned_route_domain_ready,
        local_entry_pinned_route_domain_enforced,
        requested_runtime_proof_required: requested_path_gates.runtime_proof_required,
        requested_runtime_proof_ready: requested_path_gates.runtime_proof_ready,
        requested_restart_continuity_required: requested_path_gates
            .restart_continuity_required,
        requested_restart_continuity_ready: requested_path_gates.restart_continuity_ready,
        recommended_hops,
        fallback_required: !requested_path_ready,
        pool_status: pool_status.to_string(),
        route_plan: route_plan.to_string(),
        fallback_reason: fallback_reason.to_string(),
        readiness_reason: readiness_reason.to_string(),
        next_action: next_action.to_string(),
        selection_policy: ONION_CANDIDATES_SELECTION_POLICY.to_string(),
        candidate_verification: "signed_node_descriptor_ed25519_v2".to_string(),
        path_selection_strategy: "weighted_random_health_ranked_distinct_hops".to_string(),
        network_diversity_policy: match (state.local_node_id.is_some(), local_descriptor.is_some()) {
            (_, true) => "required_against_local_entry_and_pairwise_ipv4_24_ipv6_48_or_distinct_dns_hostnames; not_operator_or_as_proof",
            (true, false) => "local_entry_descriptor_unavailable_fail_closed; required_pairwise_ipv4_24_ipv6_48_or_distinct_dns_hostnames; not_operator_or_as_proof",
            (false, false) => "required_pairwise_ipv4_24_ipv6_48_or_distinct_dns_hostnames; legacy_local_entry_context_unavailable; not_operator_or_as_proof",
        }
        .to_string(),
        pinned_route_domain_policy: if state
            .policy
            .require_pinned_route_domains_for_multi_hop
        {
            "required_for_multi_hop; operator_audited_local_opaque_assignments; distinct_entry_and_remote_domains; not_permissionless_consensus_as_proof_or_sybil_resistance"
        } else if state.policy.pinned_route_domains.is_empty() {
            "disabled_backward_compatible; coarse_endpoint_anti_affinity_only"
        } else {
            "best_effort_known_same_domain_exclusion; incomplete_assignments_allowed; not_permissionless_consensus_as_proof_or_sybil_resistance"
        }
        .to_string(),
        region_diversity_policy:
            "prefer_distinct_regions_when_available_without_exposing_selected_route".to_string(),
        user_choice_policy:
            "users_choose_privacy_mode; clients select distinct routeable relays automatically"
                .to_string(),
        refresh_after_seconds: ONION_CANDIDATES_REFRESH_AFTER_SECONDS,
        routeability_stale_after_seconds: ONION_CANDIDATES_ROUTEABILITY_STALE_AFTER_SECONDS,
        candidates,
        privacy_boundary: "fresh routeable signed node discovery metadata with the original public descriptor proof (node id, KEM public key, public endpoint, capabilities, capacity, and region); no client IPs, route ids, encrypted payloads, receiver identities, DNS contents, destinations, voucher secrets, private keys, wallet-level traffic, or social graph metadata".to_string(),
    })
}

pub(super) fn normalize_requested_hops(mode: OnionPrivacyMode, requested: Option<u8>) -> u8 {
    requested
        .unwrap_or_else(|| mode.default_hops())
        .clamp(1, ONION_CANDIDATES_MAX_CLIENT_HOPS)
}

pub(super) fn onion_requested_path_gates(
    status: &PeerStoreStatus,
    requested_hops: u8,
    admission: OnionCandidateAdmissionInput,
) -> OnionRequestedPathGates {
    let network_diversity_required = requested_hops >= 2;
    let runtime_proof_required = requested_hops >= 2;
    let restart_continuity_required = requested_hops >= 2;
    let (runtime_proof_ready, restart_continuity_ready) = match requested_hops {
        3.. => {
            let proof = &status.three_hop_path_proof_history;
            let continuity = three_hop_proof_restart_continuity(status);
            (
                proof.recent_message_delivery_ready
                    && proof.stability_ready
                    && !proof.failure_streak_active
                    && !proof.failure_circuit_breaker_active,
                continuity.peer_recovery_configured && continuity.ready,
            )
        }
        2 => {
            let proof = &status.two_hop_path_proof_history;
            let continuity = two_hop_proof_restart_continuity(status);
            (
                proof.recent_message_delivery_ready
                    && proof.stability_ready
                    && !proof.failure_streak_active
                    && !proof.failure_circuit_breaker_active,
                continuity.peer_recovery_configured && continuity.ready,
            )
        }
        _ => (true, true),
    };

    OnionRequestedPathGates {
        purpose_supported: admission.purpose.supported,
        terminal_capability_ready: admission.purpose.terminal_capability_ready,
        candidate_pool_ready: admission.candidate_pool_ready,
        network_diversity_required,
        network_diversity_ready: admission.network_diversity_ready,
        pinned_route_domain_required: admission.pinned_route_domain.required,
        pinned_route_domain_ready: admission.pinned_route_domain.ready,
        runtime_proof_required,
        runtime_proof_ready,
        restart_continuity_required,
        restart_continuity_ready,
    }
}

/// Returns whether the public candidate set contains a pairwise-diverse path.
///
/// [ONION-NETWORK-DIVERSITY 2026-08-03 by Codex] This bounded backtracking
/// search shares the internal path planner's endpoint anti-affinity rule. It
/// returns only one aggregate decision and never exposes the selected subset.
pub(super) fn onion_candidates_are_route_diverse(
    left: &OnionRelayCandidate,
    right: &OnionRelayCandidate,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
) -> bool {
    if !PeerStore::route_endpoints_are_network_diverse(
        &left.signed_descriptor,
        &right.signed_descriptor,
    ) {
        return false;
    }

    match (
        policy.pinned_route_domain(&left.signed_descriptor.node_id()),
        policy.pinned_route_domain(&right.signed_descriptor.node_id()),
    ) {
        (Some(left), Some(right)) => left != right,
        _ => !require_pinned_route_domains,
    }
}

#[cfg(test)]
pub(super) fn onion_candidate_route_diverse_subset_indices(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
) -> Option<Vec<usize>> {
    onion_candidate_route_diverse_subset_indices_for_terminal(
        candidates,
        required_hops,
        policy,
        require_pinned_route_domains,
        OnionTerminalRequirement::default(),
    )
}

pub(super) fn onion_candidate_supports_specialized_terminal(
    candidate: &OnionRelayCandidate,
    terminal_requirement: OnionTerminalRequirement,
) -> bool {
    terminal_requirement.matches(candidate)
}

/// Finds a diverse candidate subset that includes the required terminal role.
///
/// [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] A generic diverse subset is not
/// sufficient for encrypted storage: the terminal-capable node must itself be
/// inside that subset. This remains bounded by the existing candidate limit
/// and returns indices only to the in-process pool constructor.
pub(super) fn onion_candidate_route_diverse_subset_indices_for_terminal(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
) -> Option<Vec<usize>> {
    fn search(
        candidates: &[OnionRelayCandidate],
        start: usize,
        remaining: usize,
        selected: &mut Vec<usize>,
        subset_policy: OnionTerminalSubsetPolicy<'_>,
        terminal_satisfied: bool,
    ) -> bool {
        if remaining == 0 {
            return terminal_satisfied;
        }
        if candidates.len().saturating_sub(start) < remaining {
            return false;
        }
        if !terminal_satisfied
            && !candidates[start..].iter().any(|candidate| {
                onion_candidate_supports_specialized_terminal(
                    candidate,
                    subset_policy.terminal_requirement,
                )
            })
        {
            return false;
        }

        for index in start..candidates.len() {
            let candidate = &candidates[index];
            if subset_policy.require_pinned_route_domains
                && subset_policy
                    .route_policy
                    .pinned_route_domain(&candidate.signed_descriptor.node_id())
                    .is_none()
            {
                continue;
            }
            if selected.iter().any(|selected_index| {
                !onion_candidates_are_route_diverse(
                    candidate,
                    &candidates[*selected_index],
                    subset_policy.route_policy,
                    subset_policy.require_pinned_route_domains,
                )
            }) {
                continue;
            }
            let candidate_satisfies_terminal = onion_candidate_supports_specialized_terminal(
                candidate,
                subset_policy.terminal_requirement,
            );
            selected.push(index);
            if search(
                candidates,
                index + 1,
                remaining - 1,
                selected,
                subset_policy,
                terminal_satisfied || candidate_satisfies_terminal,
            ) {
                return true;
            }
            selected.pop();
        }
        false
    }

    let subset_policy = OnionTerminalSubsetPolicy {
        route_policy: policy,
        require_pinned_route_domains,
        terminal_requirement,
    };
    if required_hops == 0 {
        return (!terminal_requirement.is_specialized()).then(Vec::new);
    }
    if candidates.len() < required_hops {
        return None;
    }
    let mut selected = Vec::with_capacity(required_hops);
    if search(
        candidates,
        0,
        required_hops,
        &mut selected,
        subset_policy,
        !terminal_requirement.is_specialized(),
    ) {
        Some(selected)
    } else {
        None
    }
}

#[cfg(test)]
pub(super) fn onion_candidate_network_diverse_subset_indices(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
) -> Option<Vec<usize>> {
    onion_candidate_route_diverse_subset_indices(
        candidates,
        required_hops,
        &DiscoveryApiPolicy::default(),
        false,
    )
}

#[cfg(test)]
pub(super) fn onion_candidate_network_diversity_ready(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
) -> bool {
    onion_candidate_network_diverse_subset_indices(candidates, required_hops).is_some()
}

pub(super) fn onion_candidate_route_diversity_ready_for_terminal(
    candidates: &[OnionRelayCandidate],
    required_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
) -> bool {
    onion_candidate_route_diverse_subset_indices_for_terminal(
        candidates,
        required_hops,
        policy,
        require_pinned_route_domains,
        terminal_requirement,
    )
    .is_some()
}

/// Produces the bounded public pool without hiding a valid diverse path.
///
/// [ONION-DIVERSITY-AWARE-POOL 2026-08-03 by Codex] Ranking remains the
/// health-derived ordering/weight from the full eligible pool. When the first
/// `limit` entries are collocated, one lower-ranked candidate may be promoted
/// into the response only to preserve a pairwise-diverse requested path. For a
/// three-hop request that is not diverse-ready, a diverse two-hop subset is
/// preserved as the safe fallback. This is pool construction, not server-side
/// route selection; no chosen path, route id, or client metadata is exposed.
#[cfg(test)]
pub(super) fn select_onion_candidate_response_pool(
    candidates: Vec<OnionRelayCandidate>,
    limit: usize,
    requested_hops: usize,
) -> Vec<OnionRelayCandidate> {
    select_onion_candidate_response_pool_with_policy(
        candidates,
        limit,
        requested_hops,
        &DiscoveryApiPolicy::default(),
        false,
    )
}

#[cfg(test)]
pub(super) fn select_onion_candidate_response_pool_with_policy(
    candidates: Vec<OnionRelayCandidate>,
    limit: usize,
    requested_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
) -> Vec<OnionRelayCandidate> {
    select_onion_candidate_response_pool_with_policy_and_terminal(
        candidates,
        limit,
        requested_hops,
        policy,
        require_pinned_route_domains,
        OnionTerminalRequirement::default(),
    )
}

pub(super) fn select_onion_candidate_response_pool_with_policy_and_terminal(
    candidates: Vec<OnionRelayCandidate>,
    limit: usize,
    requested_hops: usize,
    policy: &DiscoveryApiPolicy,
    require_pinned_route_domains: bool,
    terminal_requirement: OnionTerminalRequirement,
) -> Vec<OnionRelayCandidate> {
    if limit == 0 {
        return Vec::new();
    }
    if candidates.len() <= limit && !require_pinned_route_domains {
        return candidates;
    }

    let requested_subset = ((requested_hops >= 2 || terminal_requirement.is_specialized())
        && limit >= requested_hops)
        .then(|| {
            onion_candidate_route_diverse_subset_indices_for_terminal(
                &candidates,
                requested_hops,
                policy,
                require_pinned_route_domains,
                terminal_requirement,
            )
        })
        .flatten();
    let fallback_subset = (requested_hops > ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
        && limit >= ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES)
        .then(|| {
            onion_candidate_route_diverse_subset_indices_for_terminal(
                &candidates,
                ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES,
                policy,
                require_pinned_route_domains,
                terminal_requirement,
            )
        })
        .flatten();
    // [ONION-ROUTE-PURPOSE 2026-08-10 by Codex] Preserve one healthy
    // specialized terminal even when the requested multi-hop subset is not
    // yet mature. This keeps purpose readiness observable under a small client
    // limit; route gates still defer delivery until the requested path is safe.
    let terminal_fallback = terminal_requirement
        .is_specialized()
        .then(|| {
            candidates
                .iter()
                .position(|candidate| {
                    onion_candidate_supports_specialized_terminal(candidate, terminal_requirement)
                        && (!require_pinned_route_domains
                            || policy
                                .pinned_route_domain(&candidate.signed_descriptor.node_id())
                                .is_some())
                })
                .map(|index| vec![index])
        })
        .flatten();
    let preferred_indices = requested_subset
        .or(fallback_subset)
        .or(terminal_fallback)
        .unwrap_or_default();

    let mut selected = Vec::with_capacity(limit.min(candidates.len()));
    let mut included = vec![false; candidates.len()];
    for index in preferred_indices {
        if selected.len() >= limit {
            break;
        }
        if let Some(candidate) = candidates.get(index) {
            included[index] = true;
            selected.push(candidate.clone());
        }
    }
    for (index, candidate) in candidates.into_iter().enumerate() {
        if selected.len() >= limit {
            break;
        }
        if !included[index]
            && (!require_pinned_route_domains
                || (policy
                    .pinned_route_domain(&candidate.signed_descriptor.node_id())
                    .is_some()
                    && selected.iter().all(|selected_candidate| {
                        onion_candidates_are_route_diverse(
                            &candidate,
                            selected_candidate,
                            policy,
                            true,
                        )
                    })))
        {
            selected.push(candidate);
        }
    }
    selected
}

pub(super) fn recommended_onion_hops(
    candidate_count: usize,
    requested_hops: u8,
    requested_path_ready: bool,
    two_hop_fallback_ready: bool,
) -> u8 {
    let candidate_recommendation =
        (candidate_count.min(requested_hops as usize) as u8).min(ONION_CANDIDATES_MAX_CLIENT_HOPS);
    if requested_path_ready {
        candidate_recommendation
    } else if requested_hops > 2 && two_hop_fallback_ready {
        2
    } else {
        candidate_recommendation.min(1)
    }
}

pub(super) fn onion_candidate_selection_weight(rank: usize) -> u16 {
    1_000u16
        .saturating_sub((rank as u16).saturating_mul(100))
        .max(100)
}

pub(super) fn onion_candidate_pool_status(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
    gates: OnionRequestedPathGates,
) -> &'static str {
    if !gates.purpose_supported {
        "unsupported_purpose"
    } else if !gates.terminal_capability_ready {
        "terminal_limited"
    } else if limit < required_candidates {
        "client_limited"
    } else if gates.pinned_route_domain_required && !gates.pinned_route_domain_ready {
        "routing_domain_limited"
    } else if !gates.candidate_pool_ready {
        if candidate_count == 0 {
            "empty"
        } else {
            "warming"
        }
    } else if gates.network_diversity_required && !gates.network_diversity_ready {
        "diversity_limited"
    } else if gates.runtime_proof_required && !gates.runtime_proof_ready {
        "proof_warming"
    } else if gates.restart_continuity_required && !gates.restart_continuity_ready {
        "continuity_warming"
    } else {
        "ready"
    }
}

pub(super) fn onion_candidate_route_plan(
    requested_path_ready: bool,
    requested_hops: u8,
    recommended_hops: u8,
    fallback_reason: &str,
) -> &'static str {
    if fallback_reason == "unsupported_route_purpose" {
        "reject_unsupported_purpose"
    } else if fallback_reason == "requested_terminal_capability_not_ready" {
        "defer_specialized_delivery"
    } else if !requested_path_ready && requested_hops > 2 && recommended_hops == 2 {
        "two_hop_onion_path"
    } else if !requested_path_ready {
        "standard_relay_fallback"
    } else if recommended_hops >= 3 {
        "three_hop_onion_path"
    } else if recommended_hops == 2 {
        "two_hop_onion_path"
    } else if recommended_hops == 1 {
        "single_hop_encrypted_relay"
    } else {
        "standard_relay_fallback"
    }
}

pub(super) fn onion_candidate_fallback_reason(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
    gates: OnionRequestedPathGates,
) -> &'static str {
    if !gates.purpose_supported {
        "unsupported_route_purpose"
    } else if !gates.terminal_capability_ready {
        "requested_terminal_capability_not_ready"
    } else if limit < required_candidates {
        if required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES {
            "client_limit_below_two_hop_minimum"
        } else {
            "client_limit_below_requested_hops"
        }
    } else if gates.pinned_route_domain_required && !gates.pinned_route_domain_ready {
        "requested_path_pinned_route_domain_not_ready"
    } else if !gates.candidate_pool_ready && candidate_count == 0 {
        "no_routeable_candidates"
    } else if !gates.candidate_pool_ready
        && candidate_count == 1
        && required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES
    {
        "single_routeable_candidate"
    } else if !gates.candidate_pool_ready {
        "insufficient_routeable_candidates"
    } else if gates.network_diversity_required && !gates.network_diversity_ready {
        "requested_path_network_diversity_not_ready"
    } else if gates.runtime_proof_required && !gates.runtime_proof_ready {
        "requested_path_runtime_proof_not_ready"
    } else if gates.restart_continuity_required && !gates.restart_continuity_ready {
        "requested_path_restart_continuity_not_ready"
    } else {
        "ready"
    }
}

pub(super) fn onion_candidate_readiness_reason(
    candidate_count: usize,
    limit: usize,
    required_candidates: usize,
    gates: OnionRequestedPathGates,
) -> &'static str {
    match onion_candidate_fallback_reason(candidate_count, limit, required_candidates, gates) {
        "ready" => {
            if required_candidates == ONION_CANDIDATES_MIN_TWO_HOP_CANDIDATES {
                "two_hop_candidate_pool_ready"
            } else {
                "requested_onion_candidate_pool_ready"
            }
        }
        "unsupported_route_purpose" => "requested_route_purpose_is_not_supported",
        "requested_terminal_capability_not_ready" => {
            "waiting_for_routeable_signed_terminal_capability"
        }
        "client_limit_below_two_hop_minimum" => "client_limit_blocks_two_hop_pool",
        "client_limit_below_requested_hops" => "client_limit_blocks_requested_hops",
        "no_routeable_candidates" => "waiting_for_routeable_kem_relays",
        "single_routeable_candidate" => "waiting_for_second_routeable_kem_relay",
        "requested_path_pinned_route_domain_not_ready" => {
            "waiting_for_operator_audited_route_domain_coverage"
        }
        "requested_path_network_diversity_not_ready" => "waiting_for_network_diverse_onion_relays",
        "requested_path_runtime_proof_not_ready" => {
            "waiting_for_stable_requested_path_runtime_proof"
        }
        "requested_path_restart_continuity_not_ready" => {
            "waiting_for_requested_path_restart_continuity"
        }
        _ => "waiting_for_more_routeable_kem_relays",
    }
}

pub(super) fn onion_candidate_next_action(
    requested_path_ready: bool,
    recommended_hops: u8,
    fallback_reason: &str,
) -> &'static str {
    if fallback_reason == "unsupported_route_purpose" {
        "reject the unsupported route purpose without sending the payload"
    } else if fallback_reason == "requested_terminal_capability_not_ready" {
        "keep the ciphertext queued locally and refresh until an admitted terminal is routeable"
    } else if requested_path_ready {
        "build a weighted-random onion path with fresh distinct candidates"
    } else if recommended_hops == 2
        && fallback_reason == "requested_path_pinned_route_domain_not_ready"
    {
        "use the audited two-hop fallback until a distinct pinned third route domain is available"
    } else if fallback_reason == "requested_path_pinned_route_domain_not_ready" {
        "use standard encrypted relay fallback until pinned route-domain coverage is complete"
    } else if recommended_hops == 2
        && fallback_reason == "requested_path_network_diversity_not_ready"
    {
        "use the network-diverse two-hop fallback until a diverse third hop is available"
    } else if recommended_hops == 2 {
        "use the mature two-hop onion fallback while requested path evidence warms"
    } else if fallback_reason == "client_limit_below_requested_hops"
        || fallback_reason == "client_limit_below_two_hop_minimum"
    {
        "increase candidate limit or use standard encrypted relay fallback"
    } else {
        "use standard encrypted relay fallback and refresh candidate pool later"
    }
}
