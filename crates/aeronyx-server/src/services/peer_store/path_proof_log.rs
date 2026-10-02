// [ARCH-SPLIT 2026-10-02]
// Path-proof history and privacy-safe audit events.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Records a privacy-safe discovery control-plane audit event.
    ///
    /// Events are bounded and newest-last. Full peer public keys, client
    /// identifiers, traffic metadata, and payload-derived information must not
    /// be passed into this method.
    pub fn record_audit_event(
        &self,
        now: u64,
        action: impl Into<String>,
        outcome: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let mut events = self.audit_events.write();
        if events.len() >= MAX_AUDIT_EVENTS {
            events.pop_front();
        }
        events.push_back(PeerStoreAuditEvent {
            at: now,
            action: action.into(),
            outcome: outcome.into(),
            detail: detail.into(),
        });
    }

    pub(super) fn record_path_proof_event(
        &self,
        now: u64,
        accepted: bool,
        reason: &str,
        middle_candidate_count: usize,
        terminal_candidate_count: usize,
        entry_ttl: u8,
        onward_ttl: u8,
        hop_count: u8,
        path_shape: &'static str,
    ) {
        let kind = if hop_count == 3 {
            PathProofCacheKind::ThreeHop
        } else {
            PathProofCacheKind::TwoHop
        };
        let mut events = self.path_proof_events(kind).write();
        if events.len() >= MAX_TWO_HOP_PATH_PROOF_EVENTS {
            events.pop_front();
        }
        events.push_back(PeerStoreTwoHopPathProofEvent {
            at: now,
            outcome: if accepted { "accepted" } else { "rejected" }.to_string(),
            reason_bucket: Self::two_hop_path_proof_reason_bucket(reason),
            evidence_mode: Self::path_proof_evidence_mode(reason, hop_count).to_string(),
            proof_scope: Self::two_hop_path_proof_scope(reason).to_string(),
            path_shape: path_shape.to_string(),
            hop_count,
            path_policy: TWO_HOP_PATH_POLICY_NETWORK_DIVERSE.to_string(),
            middle_candidate_bucket: Self::two_hop_candidate_count_bucket(middle_candidate_count)
                .to_string(),
            terminal_candidate_bucket: Self::two_hop_candidate_count_bucket(
                terminal_candidate_count,
            )
            .to_string(),
            ttl_shape: Self::two_hop_ttl_shape(entry_ttl, onward_ttl),
        });
    }

    pub(super) fn path_proof_events(
        &self,
        kind: PathProofCacheKind,
    ) -> &RwLock<VecDeque<PeerStoreTwoHopPathProofEvent>> {
        match kind {
            PathProofCacheKind::TwoHop => &self.two_hop_path_proof_events,
            PathProofCacheKind::ThreeHop => &self.three_hop_path_proof_events,
        }
    }

    pub(super) fn path_proof_evidence_mode(reason: &str, hop_count: u8) -> &'static str {
        let bucket = Self::two_hop_path_proof_reason_bucket(reason);
        match bucket.as_str() {
            "onion_terminal_delivered" => "synthetic_onion_message_delivery_probe",
            _ if hop_count == 3 => "synthetic_three_hop_control_probe",
            _ => "synthetic_two_hop_control_probe",
        }
    }

    pub(super) fn two_hop_path_proof_scope(reason: &str) -> &'static str {
        let bucket = Self::two_hop_path_proof_reason_bucket(reason);
        match bucket.as_str() {
            "onion_terminal_delivered" => "message_delivery",
            _ => "control_plane",
        }
    }

    pub(super) fn two_hop_candidate_count_bucket(count: usize) -> &'static str {
        match count {
            0 => "none",
            1 => "one",
            2..=3 => "few",
            4..=8 => "healthy",
            _ => "deep",
        }
    }

    pub(super) fn two_hop_ttl_shape(entry_ttl: u8, onward_ttl: u8) -> String {
        match (entry_ttl, onward_ttl) {
            (2, 1) => "entry_ttl_2_onward_ttl_1".to_string(),
            (entry, onward) => format!("entry_ttl_{entry}_onward_ttl_{onward}"),
        }
    }

    pub(super) fn two_hop_path_proof_reason_bucket(reason: &str) -> String {
        match reason {
            "accepted" => "accepted".to_string(),
            "onion_terminal_delivered" => "onion_terminal_delivered".to_string(),
            // [TWO-HOP-PROBE-OUTCOME 2026-07-31 by Codex] A mixed-version ACK
            // proves only that the encrypted control route forwarded. Without
            // a terminal-signed receipt it must never enter message-delivery
            // history or unlock authenticated App routing.
            "legacy_control_forwarded" => "legacy_control_forwarded".to_string(),
            "onion_ack_rejected" => "onion_ack_rejected".to_string(),
            "onion_ack_decode" => "onion_ack_decode".to_string(),
            "onion_receipt_unverified" => "onion_receipt_unverified".to_string(),
            "onion_kem_unavailable" => "onion_kem_unavailable".to_string(),
            "ack_rejected" => "ack_rejected".to_string(),
            "ack_decode" => "ack_decode".to_string(),
            "no_distinct_path" => "no_distinct_path".to_string(),
            "no_network_diverse_path" => "no_network_diverse_path".to_string(),
            "middle_missing_endpoint" => "middle_missing_endpoint".to_string(),
            "middle_invalid_endpoint" => "middle_invalid_endpoint".to_string(),
            value if value.starts_with("onion_http_") => "onion_http_error".to_string(),
            value if value.starts_with("http_") => "http_error".to_string(),
            value if value.starts_with("two_hop_onion_delivery_probe_") => {
                "onion_request_error".to_string()
            }
            value if value.starts_with("two_hop_blind_relay_probe_") => "request_error".to_string(),
            value if value.starts_with("two_hop_blind_relay_probe_request_") => {
                "request_error".to_string()
            }
            value if value.starts_with("three_hop_onion_delivery_probe_") => {
                "onion_request_error".to_string()
            }
            _ => "unknown".to_string(),
        }
    }

    pub(super) fn two_hop_path_proof_history(&self, now: u64) -> PeerStoreTwoHopPathProofHistory {
        let events = self
            .two_hop_path_proof_events
            .read()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        Self::path_proof_history(events, now, "two-hop")
    }

    pub(super) fn three_hop_path_proof_history(&self, now: u64) -> PeerStoreTwoHopPathProofHistory {
        let events = self
            .three_hop_path_proof_events
            .read()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        Self::path_proof_history(events, now, "three-hop")
    }

    /// Builds one privacy-safe summary for an isolated relay-hop history.
    ///
    /// [THREE-HOP-RUNTIME-PROOF 2026-08-01 by Codex] The summary algorithm is
    /// shared so freshness, stability, failure streaks, and privacy bucketing
    /// cannot drift between two-hop and three-hop runtime evidence.
    pub(super) fn path_proof_history(
        events: Vec<PeerStoreTwoHopPathProofEvent>,
        now: u64,
        hop_label: &'static str,
    ) -> PeerStoreTwoHopPathProofHistory {
        // [PATH-PROOF-CLOCK-GUARD 2026-08-03 by Codex] A monotonic runtime can
        // still observe wall-clock rollback after NTP or operator correction.
        // Never let `saturating_sub` turn future proof timestamps into age 0.
        // Retain them for aggregate diagnostics, but exclude them from every
        // readiness, quality, and reason calculation until the clock catches up.
        let future_events_ignored = events.iter().filter(|event| event.at > now).count() as u64;
        let observed_events = events
            .iter()
            .filter(|event| event.at <= now)
            .cloned()
            .collect::<Vec<_>>();
        let attempted = observed_events.len() as u64;
        let succeeded = observed_events
            .iter()
            .filter(|event| event.outcome == "accepted")
            .count() as u64;
        let message_delivery_successes = observed_events
            .iter()
            .filter(|event| event.outcome == "accepted" && event.proof_scope == "message_delivery")
            .count() as u64;
        let failed = attempted.saturating_sub(succeeded);
        let success_percent = if attempted == 0 {
            0
        } else {
            ((succeeded.saturating_mul(100)) / attempted).min(100) as u8
        };
        // Stability is a freshness claim, so old retained diagnostics must not
        // satisfy its sample threshold or keep a failure circuit open forever.
        let fresh_events = observed_events
            .iter()
            .filter(|event| now.saturating_sub(event.at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
            .collect::<Vec<_>>();
        let fresh_message_delivery_events = fresh_events
            .iter()
            .copied()
            .filter(|event| event.proof_scope == "message_delivery")
            .collect::<Vec<_>>();
        let stability_source_events = if (fresh_message_delivery_events.len() as u64)
            >= TWO_HOP_PATH_PROOF_STABILITY_MIN_ATTEMPTS
        {
            fresh_message_delivery_events
        } else {
            fresh_events
        };
        let stability_events = stability_source_events
            .iter()
            .rev()
            .take(TWO_HOP_PATH_PROOF_STABILITY_WINDOW_EVENTS)
            .copied()
            .collect::<Vec<_>>();
        let stability_window_attempted = stability_events.len() as u64;
        let stability_window_succeeded = stability_events
            .iter()
            .filter(|event| event.outcome == "accepted")
            .count() as u64;
        let stability_window_failed =
            stability_window_attempted.saturating_sub(stability_window_succeeded);
        let stability_success_percent = if stability_window_attempted == 0 {
            0
        } else {
            ((stability_window_succeeded.saturating_mul(100)) / stability_window_attempted).min(100)
                as u8
        };
        let latest = observed_events.last();
        let latest_age_seconds = latest.map(|event| now.saturating_sub(event.at));
        let latest_success_age_seconds = observed_events
            .iter()
            .rev()
            .find(|event| event.outcome == "accepted")
            .map(|event| now.saturating_sub(event.at));
        let latest_failure_age_seconds = observed_events
            .iter()
            .rev()
            .find(|event| event.outcome == "rejected")
            .map(|event| now.saturating_sub(event.at));
        let latest_message_delivery_age_seconds = observed_events
            .iter()
            .rev()
            .find(|event| event.outcome == "accepted" && event.proof_scope == "message_delivery")
            .map(|event| now.saturating_sub(event.at));
        let message_delivery_evidence_mode = observed_events
            .iter()
            .rev()
            .find(|event| event.outcome == "accepted" && event.proof_scope == "message_delivery")
            .map(|event| event.evidence_mode.clone())
            .unwrap_or_else(|| "none".to_string());
        let consecutive_successes =
            Self::count_trailing_two_hop_outcomes(&observed_events, "accepted");
        let consecutive_failures =
            Self::count_trailing_two_hop_outcomes(&observed_events, "rejected");
        let consecutive_message_delivery_successes =
            Self::count_trailing_two_hop_message_delivery_successes(&observed_events);
        let reason_bucket_counts = Self::count_two_hop_event_buckets(&observed_events, |event| {
            event.reason_bucket.as_str()
        });
        let failure_events = observed_events
            .iter()
            .filter(|event| event.outcome == "rejected")
            .cloned()
            .collect::<Vec<_>>();
        let failure_reason_bucket_counts =
            Self::count_two_hop_event_buckets(&failure_events, |event| {
                event.reason_bucket.as_str()
            });
        let path_shape_counts =
            Self::count_two_hop_event_buckets(&observed_events, |event| event.path_shape.as_str());
        let candidate_pool_counts = Self::count_two_hop_event_buckets(&observed_events, |event| {
            if event.middle_candidate_bucket.is_empty()
                || event.terminal_candidate_bucket.is_empty()
                || event.middle_candidate_bucket == "none"
                || event.terminal_candidate_bucket == "none"
            {
                "incomplete"
            } else if event.middle_candidate_bucket == "one"
                || event.terminal_candidate_bucket == "one"
            {
                "thin"
            } else if event.middle_candidate_bucket == "few"
                || event.terminal_candidate_bucket == "few"
            {
                "forming"
            } else {
                "healthy"
            }
        });
        let ttl_shape_counts =
            Self::count_two_hop_event_buckets(&observed_events, |event| event.ttl_shape.as_str());
        let proof_scope_counts = Self::count_two_hop_event_buckets(&observed_events, |event| {
            if event.proof_scope.is_empty() {
                "unknown"
            } else {
                event.proof_scope.as_str()
            }
        });
        let clock_sanity_ready = future_events_ignored == 0;
        let recent_success_ready = clock_sanity_ready
            && latest
                .map(|event| {
                    event.outcome == "accepted"
                        && now.saturating_sub(event.at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS
                })
                .unwrap_or(false);
        let message_delivery_ready = clock_sanity_ready
            && latest
                .map(|event| {
                    event.outcome == "accepted"
                        && event.proof_scope == "message_delivery"
                        && now.saturating_sub(event.at) <= PEER_ROUTEABILITY_STALE_AFTER_SECS
                })
                .unwrap_or(false);
        let recent_message_delivery_ready = clock_sanity_ready
            && latest_message_delivery_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let latest_is_fresh = clock_sanity_ready
            && latest_age_seconds
                .map(|age| age <= PEER_ROUTEABILITY_STALE_AFTER_SECS)
                .unwrap_or(false);
        let failure_streak_active = latest_is_fresh && consecutive_failures > 0;
        let failure_circuit_breaker_active = latest_is_fresh
            && consecutive_failures >= TWO_HOP_PATH_PROOF_FAILURE_CIRCUIT_BREAKER_THRESHOLD;
        let latest_age_bucket = Self::two_hop_path_proof_age_bucket(latest_age_seconds);
        let stability_ready = stability_window_attempted
            >= TWO_HOP_PATH_PROOF_STABILITY_MIN_ATTEMPTS
            && stability_success_percent >= TWO_HOP_PATH_PROOF_STABILITY_SUCCESS_PERCENT
            && recent_message_delivery_ready
            && !failure_circuit_breaker_active;
        let stability_status = if !clock_sanity_ready {
            "clock_attention"
        } else if attempted == 0 {
            "forming"
        } else if failure_circuit_breaker_active {
            "circuit_breaker"
        } else if latest_age_bucket == "stale" {
            "stale"
        } else if stability_window_attempted < TWO_HOP_PATH_PROOF_STABILITY_MIN_ATTEMPTS {
            "warming_up"
        } else if stability_ready {
            "stable"
        } else if stability_success_percent >= TWO_HOP_PATH_PROOF_STABILITY_SUCCESS_PERCENT {
            "route_stable_waiting_message_delivery"
        } else if stability_window_succeeded > 0 {
            "degraded"
        } else {
            "failing"
        };
        let (status, proof_ready, next_action) = if !clock_sanity_ready {
            (
                "attention",
                false,
                format!(
                    "verify the local clock before accepting future-dated {hop_label} path proof evidence"
                ),
            )
        } else if attempted == 0 {
            (
                "forming",
                false,
                format!("wait for the first synthetic {hop_label} path proof"),
            )
        } else if recent_success_ready {
            (
                "ready",
                true,
                format!("continue monitoring repeated {hop_label} path proof freshness"),
            )
        } else if failure_streak_active {
            (
                "attention",
                false,
                format!(
                    "inspect recent routeability, middle-hop endpoint, and terminal-hop {hop_label} proof buckets"
                ),
            )
        } else if succeeded > 0 {
            (
                "stale",
                false,
                format!(
                    "wait for a fresh {hop_label} path proof or verify peer routeability gossip"
                ),
            )
        } else {
            (
                "idle",
                false,
                format!(
                    "wait for route candidates before advertising {hop_label} path proof readiness"
                ),
            )
        };
        let freshness_bucket = if !clock_sanity_ready {
            "future_ignored"
        } else if attempted == 0 {
            "forming"
        } else if recent_success_ready {
            "fresh_success"
        } else if failure_streak_active {
            "recent_failure"
        } else if latest_success_age_seconds.is_some() {
            "stale_success"
        } else {
            "no_success"
        };

        PeerStoreTwoHopPathProofHistory {
            generated_at: now,
            status: status.to_string(),
            freshness_bucket: freshness_bucket.to_string(),
            proof_ready,
            recent_success_ready,
            message_delivery_ready,
            recent_message_delivery_ready,
            message_delivery_evidence_mode,
            failure_streak_active,
            window_size: MAX_TWO_HOP_PATH_PROOF_EVENTS,
            retained_events: events.len(),
            future_events_ignored,
            attempted,
            succeeded,
            message_delivery_successes,
            failed,
            success_percent,
            stability_window_size: TWO_HOP_PATH_PROOF_STABILITY_WINDOW_EVENTS,
            stability_window_attempted,
            stability_window_succeeded,
            stability_window_failed,
            stability_success_percent,
            stability_status: stability_status.to_string(),
            stability_ready,
            failure_circuit_breaker_threshold:
                TWO_HOP_PATH_PROOF_FAILURE_CIRCUIT_BREAKER_THRESHOLD,
            failure_circuit_breaker_active,
            latest_age_bucket: latest_age_bucket.to_string(),
            latest_outcome: latest.map(|event| event.outcome.clone()),
            latest_reason_bucket: latest.map(|event| event.reason_bucket.clone()),
            latest_age_seconds,
            latest_success_age_seconds,
            latest_failure_age_seconds,
            latest_message_delivery_age_seconds,
            consecutive_successes,
            consecutive_failures,
            consecutive_message_delivery_successes,
            reason_bucket_counts,
            failure_reason_bucket_counts,
            path_shape_counts,
            candidate_pool_counts,
            ttl_shape_counts,
            proof_scope_counts,
            stale_after_seconds: PEER_ROUTEABILITY_STALE_AFTER_SECS,
            proof_scope: latest
                .map(|event| {
                    if event.proof_scope.is_empty() {
                        "unknown".to_string()
                    } else {
                        event.proof_scope.clone()
                    }
                })
                .unwrap_or_else(|| "none".to_string()),
            next_action,
            events,
            privacy_invariant: "blind_nodes_route_only_opaque_ciphertext".to_string(),
            privacy_boundary: format!(
                "bounded synthetic {hop_label} proof history only; no node IDs, endpoints, route IDs, encrypted payloads, receiver identities, client IPs, DNS contents, domains, URLs, Memory Chain plaintext, voucher secrets, wallet-level traffic, or social graph edges"
            ),
        }
    }

    pub(super) fn two_hop_path_proof_age_bucket(latest_age_seconds: Option<u64>) -> &'static str {
        match latest_age_seconds {
            None => "none",
            Some(age) if age <= PEER_ROUTE_LAST_SEEN_FRESH_SECS => "fresh",
            Some(age) if age <= PEER_ROUTE_LAST_SEEN_ACCEPTABLE_SECS => "acceptable",
            Some(age) if age <= PEER_ROUTEABILITY_STALE_AFTER_SECS => "aging",
            Some(_) => "stale",
        }
    }

    pub(super) fn count_trailing_two_hop_outcomes(
        events: &[PeerStoreTwoHopPathProofEvent],
        outcome: &str,
    ) -> u64 {
        events
            .iter()
            .rev()
            .take_while(|event| event.outcome == outcome)
            .count() as u64
    }

    pub(super) fn count_trailing_two_hop_message_delivery_successes(
        events: &[PeerStoreTwoHopPathProofEvent],
    ) -> u64 {
        events
            .iter()
            .rev()
            .take_while(|event| {
                event.outcome == "accepted" && event.proof_scope == "message_delivery"
            })
            .count() as u64
    }

    pub(super) fn count_two_hop_event_buckets<'a>(
        events: &'a [PeerStoreTwoHopPathProofEvent],
        bucket: impl Fn(&'a PeerStoreTwoHopPathProofEvent) -> &'a str,
    ) -> BTreeMap<String, u64> {
        let mut counts = BTreeMap::new();
        for event in events {
            let key = bucket(event);
            if key.is_empty() {
                continue;
            }
            *counts.entry(key.to_string()).or_insert(0) += 1;
        }
        counts
    }

    pub(super) fn record_peer_event(
        &self,
        now: u64,
        event: impl Into<String>,
        outcome: impl Into<String>,
        source: impl Into<String>,
        node_id: &[u8; 32],
        sequence: Option<u64>,
        reason: Option<&str>,
    ) {
        let mut events = self.peer_events.write();
        if events.len() >= MAX_PEER_EVENTS {
            events.pop_front();
        }
        events.push_back(PeerStorePeerEvent {
            at: now,
            event: event.into(),
            outcome: outcome.into(),
            source: source.into(),
            node_id_prefix: Self::node_id_prefix(node_id),
            sequence,
            reason: reason.map(str::to_string),
        });
    }

    pub(super) fn node_id_prefix(node_id: &[u8; 32]) -> String {
        hex::encode(&node_id[..4])
    }

    /// Returns newest discovery audit events in chronological order.
    #[must_use]
    pub fn recent_audit_events(&self) -> Vec<PeerStoreAuditEvent> {
        self.audit_events.read().iter().cloned().collect()
    }

    /// Returns recent peer lifecycle events in chronological order.
    #[must_use]
    pub fn recent_peer_events(&self) -> Vec<PeerStorePeerEvent> {
        self.peer_events.read().iter().cloned().collect()
    }
}
