// ============================================================================
// File: crates/aeronyx-server/src/services/peer_store/status.rs
// ============================================================================
//! Privacy-safe, read-only discovery and relay status projections.
//!
//! [PEER-STORE-STATUS-SPLIT 2026-09-25 by Codex] These inherent projections
//! preserve their public signatures, lock acquisition order, and bounded
//! aggregate-only diagnostics while live routing decisions remain in parent.

use super::*;

impl PeerStore {
    /// Returns a monitoring snapshot.
    #[must_use]
    pub fn snapshot(&self, now: u64) -> PeerStoreSnapshot {
        let peers = self.peers.read();
        let mut valid_peers = 0usize;
        let mut public_peers = 0usize;
        let mut public_exit_peers = 0usize;

        for descriptor in peers.values() {
            if descriptor.verify_at(now).is_ok() {
                valid_peers += 1;
                if descriptor.descriptor.policy.public_discovery {
                    public_peers += 1;
                }
                if descriptor.descriptor.policy.allows_public_exit {
                    public_exit_peers += 1;
                }
            }
        }

        PeerStoreSnapshot {
            total_peers: peers.len(),
            valid_peers,
            public_peers,
            public_exit_peers,
        }
    }

    fn optional_age(now: u64, timestamp: Option<u64>) -> Option<u64> {
        timestamp.map(|value| now.saturating_sub(value))
    }

    fn stability(
        snapshot: &PeerStoreSnapshot,
        bootstrap: &PeerStoreBootstrapStatus,
        now: u64,
    ) -> PeerStoreStabilityStatus {
        let last_gossip_success_age_seconds =
            Self::optional_age(now, bootstrap.last_gossip_success_at);
        let last_gossip_round_age_seconds = Self::optional_age(now, bootstrap.last_gossip_round_at);
        let seed_recovery_configured = bootstrap.seed_endpoints_configured > 0;
        let mut restart_recovery_sources = Vec::new();
        if seed_recovery_configured {
            restart_recovery_sources.push("seed_endpoints".to_string());
        }
        if bootstrap.peer_cache_configured {
            restart_recovery_sources.push("peer_cache".to_string());
        }
        let restart_recovery_configured = !restart_recovery_sources.is_empty();
        let has_minimum_peer_view = snapshot.valid_peers >= 2;
        let gossip_success_is_fresh = !bootstrap.gossip_enabled
            || last_gossip_success_age_seconds
                .map(|age| age <= DISCOVERY_GOSSIP_STALE_AFTER_SECS)
                .unwrap_or(false);
        let repeated_gossip_failure =
            bootstrap.consecutive_gossip_failures >= DISCOVERY_GOSSIP_FAILURE_ATTENTION_THRESHOLD;
        let last_gossip_failed = bootstrap.last_gossip_status.as_deref() == Some("failed");
        let last_gossip_stale = bootstrap.gossip_enabled
            && last_gossip_success_age_seconds
                .map(|age| age > DISCOVERY_GOSSIP_STALE_AFTER_SECS)
                .unwrap_or(false);

        let (health, relay_foundation_ready, detail, next_action) = if !bootstrap.enabled {
            (
                "disabled",
                false,
                "Discovery is disabled in local configuration.",
                "Enable discovery and configure signed seed or peer bootstrap before relying on peer discovery.",
            )
        } else if snapshot.valid_peers == 0 {
            (
                "pending",
                false,
                "No valid signed AeroNyx peers are currently in PeerStore.",
                "Confirm local descriptor registration, seed endpoints, peer cache, and inbound discovery reachability.",
            )
        } else if bootstrap.gossip_enabled && repeated_gossip_failure {
            (
                "failed",
                false,
                "Outbound discovery gossip has failed for multiple consecutive rounds.",
                "Check peer reachability, seed recovery, firewall rules, and discovery endpoint health.",
            )
        } else if last_gossip_stale {
            (
                "stale",
                false,
                "PeerStore has valid peers, but the last successful outbound gossip is stale.",
                "Wait for a fresh gossip success or inspect seed recovery and peer endpoint connectivity.",
            )
        } else if bootstrap.gossip_enabled && last_gossip_failed {
            (
                "degraded",
                false,
                "The latest outbound gossip round failed, but the consecutive failure threshold has not been reached.",
                "Monitor the next gossip round and inspect the privacy-safe failure bucket if it repeats.",
            )
        } else if !has_minimum_peer_view {
            (
                "degraded",
                false,
                "PeerStore has only one valid signed peer, so the multi-node relay foundation is incomplete.",
                "Add or recover at least one additional signed AeroNyx peer before testing relay paths.",
            )
        } else if !restart_recovery_configured {
            (
                "degraded",
                false,
                "PeerStore has fresh peers, but no seed recovery or peer cache is configured for restart resilience.",
                "Configure discovery seed endpoints or peer_cache_path before treating this node as a stable relay foundation.",
            )
        } else if gossip_success_is_fresh {
            (
                "healthy",
                true,
                "PeerStore has multiple valid signed peers and discovery gossip is fresh enough for relay foundation checks.",
                "Continue monitoring gossip freshness, rejected descriptors, and peer-cache persistence.",
            )
        } else {
            (
                "pending",
                false,
                "Discovery is enabled and peers exist, but no successful outbound gossip has been observed yet.",
                "Wait for the first successful gossip round or verify configured seed endpoints.",
            )
        };

        PeerStoreStabilityStatus {
            health: health.to_string(),
            relay_foundation_ready,
            detail: detail.to_string(),
            next_action: next_action.to_string(),
            last_gossip_success_age_seconds,
            last_gossip_round_age_seconds,
            seed_recovery_configured,
            stale_after_seconds: DISCOVERY_GOSSIP_STALE_AFTER_SECS,
            restart_recovery_configured,
            restart_recovery_sources,
        }
    }

    /// Returns nodeboard-friendly peer store status.
    #[must_use]
    pub fn status(&self, now: u64) -> PeerStoreStatus {
        let snapshot = self.snapshot(now);
        let bootstrap = self.bootstrap_status.read().clone();
        let two_hop_path_proof_history = self.two_hop_path_proof_history(now);
        let three_hop_path_proof_history = self.three_hop_path_proof_history(now);
        let runtime = self.counters.snapshot();
        let delivery_receipt_capable_peers =
            self.fresh_purpose_bound_delivery_receipt_peer_count(now);
        let authenticated_delivery_path =
            self.authenticated_delivery_path_readiness_excluding(now, &[]);
        let blind_relay_quality = Self::blind_relay_quality_status(
            now,
            &runtime.blind_relay,
            &two_hop_path_proof_history,
            delivery_receipt_capable_peers,
            authenticated_delivery_path,
        );
        let stability = Self::stability(&snapshot, &bootstrap, now);
        let peer_summary = self.peer_summary(now);
        let route_candidates = self.route_candidate_status(now);
        let route_governance = Self::route_governance_status(now, &route_candidates);
        let peer_health_summary = self.peer_health_summary(now);
        let peer_quorum =
            Self::peer_quorum_status(now, &stability, &peer_summary, &route_candidates);
        let network_story = Self::network_story_status(
            now,
            &stability,
            &peer_summary,
            &route_candidates,
            &two_hop_path_proof_history,
        );

        PeerStoreStatus {
            snapshot,
            runtime,
            blind_relay_quality,
            two_hop_path_proof_history,
            three_hop_path_proof_history,
            max_peers: self.max_peers(),
            recent_audit_events: self.recent_audit_events(),
            recent_peer_events: self.recent_peer_events(),
            bootstrap,
            stability,
            peer_summary,
            route_candidates,
            route_governance,
            peer_health_summary,
            peer_quorum,
            network_story,
        }
    }

    fn blind_relay_quality_status(
        now: u64,
        stats: &PeerStoreBlindRelayStats,
        two_hop_path_proof_history: &PeerStoreTwoHopPathProofHistory,
        delivery_receipt_capable_peers: usize,
        authenticated_delivery_path: AuthenticatedDeliveryPathReadiness,
    ) -> PeerStoreBlindRelayQualityStatus {
        let accepted_total = stats.terminal.saturating_add(stats.forwarded);
        let last_event_age_seconds = stats
            .last_event_at
            .map(|last_event_at| now.saturating_sub(last_event_at));
        let last_accepted_age_seconds = stats
            .last_accepted_at
            .map(|last_accepted_at| now.saturating_sub(last_accepted_at));
        let last_probe_age_seconds = stats
            .last_probe_at
            .map(|last_probe_at| now.saturating_sub(last_probe_at));
        let last_two_hop_probe_age_seconds = stats
            .last_two_hop_probe_at
            .map(|last_probe_at| now.saturating_sub(last_probe_at));
        let last_verified_client_onion_delivery_age_seconds = stats
            .last_verified_client_onion_delivery_at
            .map(|at| now.saturating_sub(at));
        let accepted_evidence_seen = accepted_total > 0;
        let verified_client_onion_evidence_seen = stats.verified_client_onion_deliveries > 0;
        let verified_client_onion_evidence_fresh = verified_client_onion_evidence_seen
            && last_verified_client_onion_delivery_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let probe_evidence_seen = stats.probe_succeeded > 0;
        let two_hop_probe_evidence_seen = stats.two_hop_probe_succeeded > 0;
        let synthetic_onion_delivery_evidence_seen = two_hop_probe_evidence_seen
            && two_hop_path_proof_history.message_delivery_successes > 0
            && two_hop_path_proof_history.message_delivery_evidence_mode
                == "synthetic_onion_message_delivery_probe";
        let accepted_relay_ready = accepted_evidence_seen
            && last_accepted_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let real_relay_ready = verified_client_onion_evidence_fresh
            && delivery_receipt_capable_peers >= TWO_HOP_DELIVERY_RECEIPT_MIN_CAPABLE_PEERS
            && authenticated_delivery_path.ready;
        let probe_ready = probe_evidence_seen
            && last_probe_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let two_hop_probe_ready = two_hop_probe_evidence_seen
            && last_two_hop_probe_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let synthetic_probe_ready = probe_ready || two_hop_probe_ready;
        let runtime_ready = real_relay_ready || accepted_relay_ready || synthetic_probe_ready;
        let stale_success_evidence = (verified_client_onion_evidence_seen
            || accepted_evidence_seen
            || probe_evidence_seen
            || two_hop_probe_evidence_seen)
            && !runtime_ready;
        let evidence_mode = if verified_client_onion_evidence_seen {
            "verified_client_onion_delivery_receipt"
        } else if accepted_evidence_seen {
            "opaque_relay_acceptance"
        } else if synthetic_onion_delivery_evidence_seen {
            "synthetic_onion_message_delivery_probe"
        } else if two_hop_probe_evidence_seen {
            "synthetic_two_hop_control_probe"
        } else if probe_evidence_seen {
            "synthetic_probe"
        } else if stats.two_hop_probe_attempted > 0 {
            "two_hop_probe_failed"
        } else if stats.probe_attempted > 0 {
            "probe_failed"
        } else if stats.received > 0 {
            "opaque_relay_attempted"
        } else {
            "idle"
        };
        let proof_scope = if verified_client_onion_evidence_seen {
            "client_message_delivery"
        } else if accepted_evidence_seen {
            "relay_acceptance"
        } else if synthetic_onion_delivery_evidence_seen {
            "message_delivery"
        } else if two_hop_probe_evidence_seen || stats.two_hop_probe_attempted > 0 {
            "control_plane"
        } else if probe_evidence_seen || stats.probe_attempted > 0 {
            "single_hop_control_plane"
        } else if stats.received > 0 {
            "attempted"
        } else {
            "none"
        };
        let protection_active = stats.rate_limited > 0
            || stats.quarantined > 0
            || stats.quarantine_started > 0
            || stats.replay_dropped > 0
            || stats.loop_detected > 0
            || stats.timestamp_rejected > 0
            || stats.invalid_signature > 0;
        let transport_attention = stats.forward_failed > 0
            || stats.retry_exhausted > 0
            || stats.backpressure_dropped > 0
            || stats.probe_failed > 0;
        let transport_attention_recovered_by_stable_message_delivery = transport_attention
            && accepted_relay_ready
            && two_hop_path_proof_history.stability_ready
            && two_hop_path_proof_history.recent_message_delivery_ready
            && !two_hop_path_proof_history.failure_streak_active
            && !two_hop_path_proof_history.failure_circuit_breaker_active;
        let active_transport_attention =
            transport_attention && !transport_attention_recovered_by_stable_message_delivery;
        let quality_ready = runtime_ready && !active_transport_attention;

        let status = if stats.received == 0
            && stats.probe_attempted == 0
            && stats.two_hop_probe_attempted == 0
            && !verified_client_onion_evidence_seen
        {
            "idle"
        } else if active_transport_attention
            && (stats.retry_exhausted > 0 || stats.backpressure_dropped > 0)
        {
            "attention"
        } else if active_transport_attention && (stats.forward_failed > 0 || stats.probe_failed > 0)
        {
            "degraded"
        } else if protection_active {
            "protecting"
        } else if runtime_ready {
            "ready"
        } else if verified_client_onion_evidence_fresh {
            "observing"
        } else if stale_success_evidence {
            "stale"
        } else {
            "observing"
        };

        let readiness_reason =
            if real_relay_ready && !active_transport_attention && !protection_active {
                "verified_client_onion_delivery_receipt_ready"
            } else if real_relay_ready && active_transport_attention {
                "verified_client_onion_delivery_transport_attention"
            } else if real_relay_ready && protection_active {
                "verified_client_onion_delivery_protection_active"
            } else if accepted_relay_ready && !active_transport_attention && !protection_active {
                "opaque_relay_acceptance_observed"
            } else if accepted_relay_ready && active_transport_attention {
                "opaque_relay_transport_attention"
            } else if accepted_relay_ready && protection_active {
                "opaque_relay_protection_active"
            } else if two_hop_probe_ready
                && synthetic_onion_delivery_evidence_seen
                && !active_transport_attention
                && !protection_active
            {
                "synthetic_onion_message_delivery_probe_ready"
            } else if two_hop_probe_ready && !active_transport_attention && !protection_active {
                "synthetic_two_hop_control_probe_ready"
            } else if synthetic_probe_ready && !active_transport_attention && !protection_active {
                "synthetic_probe_ready"
            } else if stats.two_hop_probe_attempted > 0 && stats.two_hop_probe_succeeded == 0 {
                "synthetic_two_hop_control_probe_failed"
            } else if stats.probe_attempted > 0 && stats.probe_succeeded == 0 {
                "synthetic_probe_failed"
            } else if active_transport_attention {
                "transport_attention"
            } else if protection_active {
                "protection_active"
            } else if verified_client_onion_evidence_fresh && !real_relay_ready {
                "verified_client_onion_delivery_peer_revalidation_required"
            } else if verified_client_onion_evidence_seen && !real_relay_ready {
                "verified_client_onion_delivery_receipt_stale"
            } else if accepted_evidence_seen && !accepted_relay_ready {
                "opaque_relay_acceptance_stale"
            } else if synthetic_onion_delivery_evidence_seen && !two_hop_probe_ready {
                "synthetic_onion_message_delivery_probe_stale"
            } else if two_hop_probe_evidence_seen && !two_hop_probe_ready {
                "synthetic_two_hop_control_probe_stale"
            } else if probe_evidence_seen && !probe_ready {
                "synthetic_probe_stale"
            } else if stats.received > 0 {
                "opaque_relay_attempted"
            } else {
                "idle_waiting_for_relay"
            };

        let accepted_percent = if stats.received == 0 {
            0
        } else {
            ((accepted_total.saturating_mul(100)) / stats.received).min(100) as u8
        };
        let next_action = match status {
            "idle" => "wait for encrypted blind relay traffic or run a synthetic relay probe",
            "attention" => {
                "inspect next-hop reachability, retry exhaustion, and local relay backpressure"
            }
            "degraded" => "inspect route candidates and next-hop transport health",
            "protecting" => {
                "review aggregate abuse-guard buckets while preserving blind relay metadata"
            }
            "ready" if real_relay_ready => {
                "fresh terminal-signed receipt proves authenticated client onion delivery"
            }
            "ready" if accepted_relay_ready => {
                "blind relay runtime has accepted encrypted relay work"
            }
            "ready" if two_hop_probe_ready && synthetic_onion_delivery_evidence_seen => {
                "synthetic onion terminal delivery probe succeeded; do not present it as App/user traffic"
            }
            "ready" if two_hop_probe_ready => {
                "two-hop synthetic path proof succeeded; do not present it as App/user traffic"
            }
            "ready" => "synthetic relay probe succeeded; do not present it as App/user traffic",
            "stale" => "refresh relay readiness with a new accepted encrypted relay event or synthetic path proof",
            _ => "observe additional relay traffic before declaring runtime quality ready",
        };

        let detail = format!(
            "received={} accepted_total={} terminal={} forwarded={} rejected={} forward_failed={} retry_attempted={} retry_succeeded={} retry_exhausted={} backpressure_dropped={} probe_attempted={} probe_succeeded={} probe_failed={} two_hop_probe_attempted={} two_hop_probe_succeeded={} two_hop_probe_failed={} timestamp_rejected={} real_relay_ready={} verified_client_onion_deliveries={} delivery_receipt_capable_peers={} authenticated_delivery_path_ready={} authenticated_delivery_path_reason={} accepted_relay_ready={} synthetic_probe_ready={} two_hop_probe_ready={} evidence_mode={} proof_scope={} readiness_reason={} protection_active={} accepted_percent={} transport_attention_recovered={} proof_stability_status={} stale_after_seconds={} last_event_age_seconds={} last_accepted_age_seconds={} last_probe_age_seconds={} last_two_hop_probe_age_seconds={} last_verified_client_onion_delivery_age_seconds={}",
            stats.received,
            accepted_total,
            stats.terminal,
            stats.forwarded,
            stats.rejected,
            stats.forward_failed,
            stats.retry_attempted,
            stats.retry_succeeded,
            stats.retry_exhausted,
            stats.backpressure_dropped,
            stats.probe_attempted,
            stats.probe_succeeded,
            stats.probe_failed,
            stats.two_hop_probe_attempted,
            stats.two_hop_probe_succeeded,
            stats.two_hop_probe_failed,
            stats.timestamp_rejected,
            real_relay_ready,
            stats.verified_client_onion_deliveries,
            delivery_receipt_capable_peers,
            authenticated_delivery_path.ready,
            authenticated_delivery_path.reason,
            accepted_relay_ready,
            synthetic_probe_ready,
            two_hop_probe_ready,
            evidence_mode,
            proof_scope,
            readiness_reason,
            protection_active,
            accepted_percent,
            transport_attention_recovered_by_stable_message_delivery,
            two_hop_path_proof_history.stability_status,
            PEER_ROUTEABILITY_STALE_AFTER_SECS,
            last_event_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_accepted_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_probe_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_two_hop_probe_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            last_verified_client_onion_delivery_age_seconds
                .map(|age| age.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );

        PeerStoreBlindRelayQualityStatus {
            generated_at: now,
            status: status.to_string(),
            runtime_ready,
            quality_ready,
            real_relay_ready,
            verified_client_onion_deliveries: stats.verified_client_onion_deliveries,
            last_verified_client_onion_delivery_age_seconds,
            delivery_receipt_capable_peers,
            authenticated_delivery_path_ready: authenticated_delivery_path.ready,
            authenticated_delivery_path_reason: authenticated_delivery_path.reason.to_string(),
            accepted_relay_ready,
            synthetic_probe_ready,
            evidence_mode: evidence_mode.to_string(),
            proof_scope: proof_scope.to_string(),
            readiness_reason: readiness_reason.to_string(),
            accepted_total,
            last_accepted_at: stats.last_accepted_at,
            forward_failed: stats.forward_failed,
            retry_exhausted: stats.retry_exhausted,
            backpressure_dropped: stats.backpressure_dropped,
            probe_attempted: stats.probe_attempted,
            probe_succeeded: stats.probe_succeeded,
            probe_failed: stats.probe_failed,
            two_hop_probe_ready,
            two_hop_probe_attempted: stats.two_hop_probe_attempted,
            two_hop_probe_succeeded: stats.two_hop_probe_succeeded,
            two_hop_probe_failed: stats.two_hop_probe_failed,
            last_two_hop_probe_age_seconds,
            timestamp_rejected: stats.timestamp_rejected,
            protection_active,
            accepted_percent,
            last_event_age_seconds,
            last_accepted_age_seconds,
            last_probe_age_seconds,
            detail,
            next_action: next_action.to_string(),
            privacy_boundary: "aggregate blind relay runtime counters only; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client IPs, DNS contents, destinations, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    fn route_governance_status(
        now: u64,
        route_candidates: &PeerStoreRouteCandidateStatus,
    ) -> PeerStoreRouteGovernanceStatus {
        let mut candidates_total = 0usize;
        let mut routeable_total = 0usize;
        let mut routeable_privacy_relays = 0usize;
        let mut routeable_chat_relays = 0usize;
        let mut routeable_onion_middle_hops = 0usize;
        let mut quarantined_total = 0usize;
        let mut failing_total = 0usize;
        let mut degraded_total = 0usize;
        let mut unknown_routeability_total = 0usize;
        let mut stale_routeability_total = 0usize;
        let mut unreachable_total = 0usize;
        let mut best_score: Option<i64> = None;
        let mut worst_score: Option<i64> = None;
        let mut score_sum = 0i64;

        for candidates in [
            &route_candidates.privacy_relay,
            &route_candidates.chat_relay,
            &route_candidates.onion_middle,
        ] {
            for candidate in candidates.iter() {
                candidates_total = candidates_total.saturating_add(1);
                score_sum = score_sum.saturating_add(candidate.score);
                best_score = Some(
                    best_score
                        .map(|score| score.max(candidate.score))
                        .unwrap_or(candidate.score),
                );
                worst_score = Some(
                    worst_score
                        .map(|score| score.min(candidate.score))
                        .unwrap_or(candidate.score),
                );

                if candidate.routeability_ready && !candidate.route_quarantined {
                    routeable_total = routeable_total.saturating_add(1);
                    match candidate.capability.as_str() {
                        "privacy_relay" => {
                            routeable_privacy_relays = routeable_privacy_relays.saturating_add(1);
                        }
                        "chat_relay" => {
                            routeable_chat_relays = routeable_chat_relays.saturating_add(1);
                        }
                        "onion_middle" => {
                            routeable_onion_middle_hops =
                                routeable_onion_middle_hops.saturating_add(1);
                        }
                        _ => {}
                    }
                }

                match candidate.route_health.as_str() {
                    "quarantined" => {
                        quarantined_total = quarantined_total.saturating_add(1);
                    }
                    "failing" => {
                        failing_total = failing_total.saturating_add(1);
                    }
                    "degraded" => {
                        degraded_total = degraded_total.saturating_add(1);
                    }
                    _ => {}
                }

                match candidate.routeability_state.as_str() {
                    "unknown" => {
                        unknown_routeability_total = unknown_routeability_total.saturating_add(1);
                    }
                    "stale" => {
                        stale_routeability_total = stale_routeability_total.saturating_add(1);
                    }
                    "unreachable" => {
                        unreachable_total = unreachable_total.saturating_add(1);
                    }
                    _ => {}
                }
            }
        }

        let chat_single_hop_ready = route_candidates.planned_paths.chat_single_hop.complete;
        let chat_two_hop_onion_ready = route_candidates
            .planned_paths
            .chat_two_hop_onion_ready
            .complete;
        let route_pool_ready = chat_two_hop_onion_ready;
        let attention_active = quarantined_total > 0
            || failing_total > 0
            || degraded_total > 0
            || unreachable_total > 0;
        let quality_ready = route_pool_ready && !attention_active;
        let status = if attention_active {
            "attention"
        } else if route_pool_ready {
            "healthy"
        } else {
            "forming"
        };
        let average_score = if candidates_total == 0 {
            None
        } else {
            Some(score_sum / (candidates_total as i64))
        };
        let next_action = match status {
            "attention" if quarantined_total > 0 => {
                "wait for quarantined peers to recover or add more routeable nodes before increasing relay traffic"
            }
            "attention" => {
                "watch degraded routeability evidence and prefer peers with fresh successful opaque forwards"
            }
            "healthy" => "route governance is healthy for controlled encrypted relay experiments",
            _ if candidates_total == 0 => {
                "publish signed descriptors and seed peers before planning multi-hop relay paths"
            }
            _ => "wait for fresh routeability probes to complete the two-hop privacy path pool",
        };
        let detail = format!(
            "candidates_total={candidates_total} routeable_total={routeable_total} routeable_chat_relays={routeable_chat_relays} routeable_onion_middle_hops={routeable_onion_middle_hops} quarantined_total={quarantined_total} failing_total={failing_total} degraded_total={degraded_total} unreachable_total={unreachable_total}"
        );

        PeerStoreRouteGovernanceStatus {
            generated_at: now,
            contract_version: "route_governance.v1".to_string(),
            source: "peer_store_route_candidates".to_string(),
            status: status.to_string(),
            route_pool_ready,
            quality_ready,
            candidates_total,
            routeable_total,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            routeable_privacy_relays,
            quarantined_total,
            failing_total,
            degraded_total,
            unknown_routeability_total,
            stale_routeability_total,
            unreachable_total,
            best_score,
            worst_score,
            average_score,
            chat_single_hop_ready,
            chat_two_hop_onion_ready,
            quarantine_threshold: PEER_ROUTE_FAILURE_QUARANTINE_THRESHOLD,
            quarantine_seconds: PEER_ROUTE_FAILURE_QUARANTINE_SECS,
            routeability_stale_after_seconds: PEER_ROUTEABILITY_STALE_AFTER_SECS,
            detail,
            next_action: next_action.to_string(),
            privacy_boundary: "aggregate route governance only; no full node ids, endpoint URLs, selected paths, route ids, encrypted payloads, receiver identities, client IPs, DNS contents, destinations, voucher secrets, private keys, wallet-level traffic, plaintext, or social graph metadata".to_string(),
        }
    }

    fn peer_quorum_status(
        now: u64,
        stability: &PeerStoreStabilityStatus,
        peer_summary: &PeerStorePeerSummaryStatus,
        route_candidates: &PeerStoreRouteCandidateStatus,
    ) -> PeerStorePeerQuorumStatus {
        let routeable_chat_relays = route_candidates
            .chat_relay
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();
        let routeable_onion_middle_hops = route_candidates
            .onion_middle
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();
        let healthy_ratio_percent = if peer_summary.valid_peers == 0 {
            0
        } else {
            ((peer_summary.healthy_peers * 100) / peer_summary.valid_peers).min(100) as u8
        };
        let enough_valid_peers = peer_summary.valid_peers >= PEER_QUORUM_MIN_VALID_PEERS;
        let enough_routeable_chat_relays =
            routeable_chat_relays >= PEER_QUORUM_MIN_ROUTEABLE_CHAT_RELAYS;
        let stability_needs_attention =
            matches!(stability.health.as_str(), "failed" | "degraded" | "stale");
        let quorum_ready = !stability_needs_attention
            && enough_valid_peers
            && enough_routeable_chat_relays
            && stability.restart_recovery_configured
            && stability.relay_foundation_ready;

        let status = if stability.health == "disabled" {
            "disabled"
        } else if stability_needs_attention {
            "attention"
        } else if !enough_valid_peers {
            "forming"
        } else if enough_routeable_chat_relays {
            "route_ready"
        } else {
            "peer_view_ready"
        };

        let next_action = match status {
            "disabled" => "enable discovery and configure peer recovery before testing multi-hop routing",
            "attention" => "restore discovery stability before relying on the peer view",
            "forming" => "add seed endpoints, peer cache, or live gossip until the verified peer view reaches quorum",
            "peer_view_ready" if route_candidates.chat_relay.iter().any(|peer| peer.endpoint_advertised) => {
                "wait for a fresh successful routeability probe before declaring encrypted relay readiness"
            }
            "peer_view_ready" => "ensure at least one verified peer advertises a public chat relay endpoint",
            _ if !stability.restart_recovery_configured => {
                "configure peer cache or seed endpoints so peer quorum survives restart"
            }
            _ => "peer quorum is ready for controlled encrypted relay experiments",
        };

        let detail = format!(
            "valid_peers={} healthy_peers={} stale_peers={} routeable_chat_relays={} routeable_onion_middle_hops={} restart_recovery_configured={} relay_foundation_ready={}",
            peer_summary.valid_peers,
            peer_summary.healthy_peers,
            peer_summary.stale_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            stability.restart_recovery_configured,
            stability.relay_foundation_ready
        );

        PeerStorePeerQuorumStatus {
            generated_at: now,
            status: status.to_string(),
            quorum_ready,
            min_valid_peers: PEER_QUORUM_MIN_VALID_PEERS,
            min_routeable_chat_relays: PEER_QUORUM_MIN_ROUTEABLE_CHAT_RELAYS,
            valid_peers: peer_summary.valid_peers,
            healthy_peers: peer_summary.healthy_peers,
            stale_peers: peer_summary.stale_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            healthy_ratio_percent,
            restart_recovery_configured: stability.restart_recovery_configured,
            relay_foundation_ready: stability.relay_foundation_ready,
            detail,
            next_action: next_action.to_string(),
            privacy_boundary: "aggregate local peer-view readiness only; not public-chain consensus; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client IPs, destinations, DNS contents, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    fn network_story_status(
        now: u64,
        stability: &PeerStoreStabilityStatus,
        peer_summary: &PeerStorePeerSummaryStatus,
        route_candidates: &PeerStoreRouteCandidateStatus,
        two_hop_path_proof_history: &PeerStoreTwoHopPathProofHistory,
    ) -> PeerStoreNetworkStoryStatus {
        let chat_single_hop_ready = route_candidates.planned_paths.chat_single_hop.complete;
        let planned_two_hop_onion_ready = route_candidates
            .planned_paths
            .chat_two_hop_onion_ready
            .complete;
        let proof_backed_two_hop_onion_ready = two_hop_path_proof_history.recent_success_ready
            && !two_hop_path_proof_history.failure_streak_active;
        let chat_two_hop_onion_ready =
            planned_two_hop_onion_ready || proof_backed_two_hop_onion_ready;
        let routeable_chat_relays = route_candidates
            .chat_relay
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();
        let routeable_onion_middle_hops = route_candidates
            .onion_middle
            .iter()
            .filter(|peer| peer.routeability_ready && !peer.route_quarantined)
            .count();

        let stability_needs_attention =
            matches!(stability.health.as_str(), "failed" | "degraded" | "stale");

        let status = if stability.health == "disabled" {
            "disabled"
        } else if stability_needs_attention {
            "attention"
        } else if chat_two_hop_onion_ready {
            "onion_ready"
        } else if chat_single_hop_ready {
            "relay_ready"
        } else if peer_summary.valid_peers > 0 {
            "peer_view_ready"
        } else {
            "discovering"
        };

        let headline = match status {
            "onion_ready" => "Discovery can plan a two-hop privacy path",
            "relay_ready" => "Discovery can route encrypted relay traffic",
            "peer_view_ready" => "Discovery has a verified peer view",
            "disabled" => "Discovery is disabled",
            "attention" => "Discovery needs operator attention",
            _ => "Discovery is learning the network",
        };

        let detail = format!(
            "valid_nodes={} chat_relay_nodes={} onion_middle_nodes={} routeable_chat_relays={} routeable_onion_middle_hops={} two_hop_path_proof_recent={} restart_recovery_configured={} relay_foundation_ready={}",
            peer_summary.valid_peers,
            peer_summary.chat_relay_peers,
            peer_summary.onion_middle_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            proof_backed_two_hop_onion_ready,
            stability.restart_recovery_configured,
            stability.relay_foundation_ready
        );

        PeerStoreNetworkStoryStatus {
            generated_at: now,
            status: status.to_string(),
            headline: headline.to_string(),
            detail,
            discovered_nodes: peer_summary.total_peers,
            valid_nodes: peer_summary.valid_peers,
            chat_relay_nodes: peer_summary.chat_relay_peers,
            onion_middle_nodes: peer_summary.onion_middle_peers,
            routeable_chat_relays,
            routeable_onion_middle_hops,
            chat_single_hop_ready,
            chat_two_hop_onion_ready,
            restart_recovery_configured: stability.restart_recovery_configured,
            relay_foundation_ready: stability.relay_foundation_ready,
            privacy_boundary: "aggregate node discovery status only; no full node ids, endpoint URLs, route ids, encrypted payloads, receiver identities, client IPs, destinations, DNS contents, voucher secrets, private keys, wallet-level traffic, or plaintext".to_string(),
        }
    }

    fn route_candidate_summaries(
        &self,
        capability: NodeCapability,
        now: u64,
    ) -> Vec<PeerStoreRouteCandidate> {
        self.scored_route_candidates(capability, now, Some(PEER_ROUTE_STATUS_LIMIT), true)
            .into_iter()
            .map(|candidate| candidate.summary)
            .collect()
    }

    fn route_path_plan_preview(
        &self,
        label: &str,
        capabilities: &[NodeCapability],
        now: u64,
    ) -> PeerStoreRoutePathPlan {
        let planned = self.scored_route_path_with_capabilities_excluding(capabilities, now, &[]);
        let hops = planned
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(hop_index, candidate)| PeerStoreRoutePathHop {
                hop_index,
                capability: Self::capability_label(capabilities[hop_index]).to_string(),
                node_id_prefix: candidate.summary.node_id_prefix,
                score: candidate.summary.score,
                health: candidate.summary.health,
                route_health: candidate.summary.route_health,
                last_seen_age_seconds: candidate.summary.last_seen_age_seconds,
                ttl_remaining_seconds: candidate.summary.ttl_remaining_seconds,
                region: candidate.summary.region,
            })
            .collect::<Vec<_>>();

        PeerStoreRoutePathPlan {
            label: label.to_string(),
            required_capabilities: capabilities
                .iter()
                .map(|capability| Self::capability_label(*capability).to_string())
                .collect(),
            complete: hops.len() == capabilities.len(),
            hop_count: hops.len(),
            hops,
        }
    }

    pub(super) fn route_path_status(&self, now: u64) -> PeerStoreRoutePathStatus {
        PeerStoreRoutePathStatus {
            chat_single_hop: self.route_path_plan_preview(
                "chat_single_hop",
                &[NodeCapability::ChatRelay],
                now,
            ),
            chat_two_hop_onion_ready: self.route_path_plan_preview(
                "chat_two_hop_onion_ready",
                &[NodeCapability::OnionMiddle, NodeCapability::ChatRelay],
                now,
            ),
        }
    }

    /// Builds privacy-safe route candidate lists for nodeboard.
    #[must_use]
    pub fn route_candidate_status(&self, now: u64) -> PeerStoreRouteCandidateStatus {
        PeerStoreRouteCandidateStatus {
            generated_at: now,
            privacy_relay: self.route_candidate_summaries(NodeCapability::PrivacyRelay, now),
            chat_relay: self.route_candidate_summaries(NodeCapability::ChatRelay, now),
            onion_middle: self.route_candidate_summaries(NodeCapability::OnionMiddle, now),
            planned_paths: self.route_path_status(now),
        }
    }

    /// Builds a commercial peer summary for heartbeat/nodeboard.
    #[must_use]
    pub fn peer_summary(&self, now: u64) -> PeerStorePeerSummaryStatus {
        let peers = self.peers.read();
        let metadata = self.peer_runtime.read();
        let mut source_counts = BTreeMap::new();
        let mut rows = Vec::with_capacity(peers.len().min(64));
        let mut healthy_peers = 0usize;
        let mut stale_peers = 0usize;
        let mut expired_peers = 0usize;
        let mut valid_peers = 0usize;
        let mut privacy_relay_peers = 0usize;
        let mut chat_relay_peers = 0usize;
        let mut encrypted_storage_peers = 0usize;
        let mut agent_relay_peers = 0usize;
        let mut onion_middle_peers = 0usize;

        for (node_id, descriptor) in peers.iter() {
            let meta = metadata.get(node_id);
            let source = meta
                .map(|value| value.source.clone())
                .unwrap_or_else(|| "unknown".to_string());
            *source_counts.entry(source.clone()).or_insert(0) += 1;

            let (health, ttl_remaining_seconds) = Self::descriptor_health(descriptor, now);
            match health {
                "healthy" => {
                    healthy_peers += 1;
                    valid_peers += 1;
                }
                "stale" => {
                    stale_peers += 1;
                    valid_peers += 1;
                }
                _ => expired_peers += 1,
            }

            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::PrivacyRelay)
            {
                privacy_relay_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::ChatRelay)
            {
                chat_relay_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::EncryptedStorage)
            {
                encrypted_storage_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::AgentRelay)
            {
                agent_relay_peers += 1;
            }
            if descriptor
                .descriptor
                .capabilities
                .contains(&NodeCapability::OnionMiddle)
            {
                onion_middle_peers += 1;
            }

            rows.push(PeerStorePeerSummary {
                node_id_prefix: hex::encode(&node_id[..4]),
                source,
                sequence: descriptor.sequence(),
                imported_count: meta.map(|value| value.imported_count).unwrap_or(0),
                first_seen_at: meta.map(|value| value.first_seen_at).unwrap_or(now),
                last_seen_at: meta.map(|value| value.last_seen_at).unwrap_or(now),
                last_seen_age_seconds: meta
                    .map(|value| now.saturating_sub(value.last_seen_at))
                    .unwrap_or(0),
                expires_at: descriptor.descriptor.expires_at,
                ttl_remaining_seconds,
                health: health.to_string(),
                capabilities: descriptor
                    .descriptor
                    .capabilities
                    .iter()
                    .copied()
                    .map(Self::capability_label)
                    .map(str::to_string)
                    .collect(),
                endpoint_advertised: descriptor.descriptor.public_endpoint.is_some(),
                public_discovery: descriptor.descriptor.policy.public_discovery,
                region: descriptor.descriptor.policy.region.clone(),
            });
        }

        rows.sort_by(|a, b| {
            a.health
                .cmp(&b.health)
                .then_with(|| a.source.cmp(&b.source))
                .then_with(|| a.node_id_prefix.cmp(&b.node_id_prefix))
        });

        PeerStorePeerSummaryStatus {
            total_peers: peers.len(),
            valid_peers,
            healthy_peers,
            stale_peers,
            expired_peers,
            privacy_relay_peers,
            chat_relay_peers,
            encrypted_storage_peers,
            agent_relay_peers,
            onion_middle_peers,
            source_counts,
            peers: rows,
        }
    }

    fn source_is_live_gossip(source: &str) -> bool {
        matches!(source, "gossip_snapshot" | "gossip_announce")
    }

    fn peer_health_bucket(
        descriptor_health: &str,
        route_health: &str,
        relay_quarantined: bool,
        relay_rejection_count: u64,
    ) -> &'static str {
        if relay_quarantined || route_health == "quarantined" {
            return "quarantined";
        }
        if descriptor_health == "expired" {
            return "expired";
        }
        if route_health == "failing" {
            return "failing";
        }
        if descriptor_health == "stale" || route_health == "degraded" || relay_rejection_count > 0 {
            return "degraded";
        }
        "healthy"
    }

    /// Builds a privacy-safe per-peer health summary for nodeboard.
    ///
    /// The summary intentionally combines only node-level control-plane state:
    /// signed descriptor freshness, import/gossip source buckets, local opaque
    /// route-health counters, and relay-protection buckets. It never includes
    /// endpoint URLs, route ids, encrypted blobs, receiver identities, client
    /// IPs, destinations, DNS contents, voucher secrets, private keys,
    /// wallet-level traffic, or plaintext content.
    #[must_use]
    pub fn peer_health_summary(&self, now: u64) -> PeerStorePeerHealthStatus {
        let peers = self.peers.read();
        let metadata = self.peer_runtime.read();
        let route_health = self.route_health.read();
        let relay_health = self.relay_protection_health.read();

        let mut rows = Vec::with_capacity(peers.len().min(PEER_HEALTH_STATUS_LIMIT));
        let mut healthy_peers = 0usize;
        let mut degraded_peers = 0usize;
        let mut failing_peers = 0usize;
        let mut quarantined_peers = 0usize;

        for (node_id, descriptor) in peers.iter() {
            let meta = metadata.get(node_id);
            let source = meta
                .map(|value| value.source.clone())
                .unwrap_or_else(|| "unknown".to_string());
            let last_seen_at = meta.map(|value| value.last_seen_at).unwrap_or(now);
            let last_seen_age_seconds = now.saturating_sub(last_seen_at);
            let last_successful_gossip_at = meta.and_then(|value| {
                Self::source_is_live_gossip(&value.source).then_some(value.last_seen_at)
            });
            let last_successful_gossip_age_seconds =
                last_successful_gossip_at.map(|value| now.saturating_sub(value));

            let (descriptor_health, _) = Self::descriptor_health(descriptor, now);
            let route_health_entry = route_health.get(node_id);
            let (route_health_bucket, _) =
                Self::route_health_bucket_and_score(route_health_entry, now);
            let (routeability_state, routeability_ready) =
                Self::routeability_state_and_ready(route_health_entry, now);
            let last_routeability_probe_at = Self::routeability_probe_at(route_health_entry);
            let last_routeability_probe_age_seconds =
                last_routeability_probe_at.map(|probe_at| now.saturating_sub(probe_at));
            let route_quarantine_remaining_seconds = route_health_entry
                .and_then(|value| Self::route_quarantine_remaining_seconds(value, now));
            let relay_health_entry = relay_health.get(node_id);
            let relay_quarantine_remaining_seconds = relay_health_entry
                .and_then(|value| value.quarantine_until)
                .and_then(|quarantine_until| {
                    (now < quarantine_until).then_some(quarantine_until.saturating_sub(now))
                });
            let relay_quarantined = relay_quarantine_remaining_seconds.is_some();
            let relay_rejection_count = relay_health_entry
                .map(|value| value.rejection_count)
                .unwrap_or(0);
            let health = Self::peer_health_bucket(
                descriptor_health,
                route_health_bucket,
                relay_quarantined,
                relay_rejection_count,
            );

            match health {
                "quarantined" => quarantined_peers += 1,
                "failing" => failing_peers += 1,
                "degraded" | "expired" => degraded_peers += 1,
                _ => healthy_peers += 1,
            }

            rows.push(PeerStorePeerHealth {
                node_id_prefix: hex::encode(&node_id[..4]),
                health: health.to_string(),
                descriptor_health: descriptor_health.to_string(),
                source,
                last_successful_gossip_at,
                last_successful_gossip_age_seconds,
                last_seen_at,
                last_seen_age_seconds,
                route_health: route_health_bucket.to_string(),
                routeability_state: routeability_state.to_string(),
                routeability_ready,
                last_routeability_probe_at,
                last_routeability_probe_age_seconds,
                route_success_count: route_health_entry
                    .map(|value| value.success_count)
                    .unwrap_or(0),
                route_failure_count: route_health_entry
                    .map(|value| value.failure_count)
                    .unwrap_or(0),
                route_consecutive_failures: route_health_entry
                    .map(|value| value.consecutive_failures)
                    .unwrap_or(0),
                last_route_success_at: route_health_entry.and_then(|value| value.last_success_at),
                last_route_failure_at: route_health_entry.and_then(|value| value.last_failure_at),
                last_route_failure_reason: route_health_entry
                    .and_then(|value| value.last_failure_reason.clone()),
                route_quarantined: route_quarantine_remaining_seconds.is_some(),
                route_quarantine_remaining_seconds,
                route_quarantine_count: route_health_entry
                    .map(|value| value.quarantine_count)
                    .unwrap_or(0),
                last_route_quarantine_at: route_health_entry
                    .and_then(|value| value.last_quarantine_at),
                last_route_quarantine_reason: route_health_entry
                    .and_then(|value| value.last_quarantine_reason.clone()),
                relay_rejection_count,
                relay_quarantine_count: relay_health_entry
                    .map(|value| value.quarantine_count)
                    .unwrap_or(0),
                relay_quarantined,
                relay_quarantine_remaining_seconds,
                last_relay_rejection_at: relay_health_entry
                    .and_then(|value| value.last_rejection_at),
                last_relay_rejection_reason: relay_health_entry
                    .and_then(|value| value.last_rejection_reason.clone()),
                last_relay_quarantine_at: relay_health_entry
                    .and_then(|value| value.last_quarantine_at),
                last_relay_quarantine_reason: relay_health_entry
                    .and_then(|value| value.last_quarantine_reason.clone()),
            });
        }

        rows.sort_by(|a, b| {
            peer_health_rank(&a.health)
                .cmp(&peer_health_rank(&b.health))
                .then_with(|| {
                    b.route_consecutive_failures
                        .cmp(&a.route_consecutive_failures)
                })
                .then_with(|| b.relay_rejection_count.cmp(&a.relay_rejection_count))
                .then_with(|| a.last_seen_age_seconds.cmp(&b.last_seen_age_seconds))
                .then_with(|| a.node_id_prefix.cmp(&b.node_id_prefix))
        });
        rows.truncate(PEER_HEALTH_STATUS_LIMIT);

        PeerStorePeerHealthStatus {
            generated_at: now,
            total_peers: peers.len(),
            healthy_peers,
            degraded_peers,
            failing_peers,
            quarantined_peers,
            peers: rows,
        }
    }
}

fn peer_health_rank(health: &str) -> u8 {
    match health {
        "quarantined" => 0,
        "failing" => 1,
        "degraded" => 2,
        "expired" => 3,
        "stale" => 4,
        _ => 5,
    }
}
