// [ARCH-SPLIT 2026-10-02]
// Gossip, bootstrap, and client-delivery witness counters.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl PeerStore {
    /// Records discovery bootstrap feature flags from local config.
    pub fn configure_bootstrap_status(
        &self,
        enabled: bool,
        peer_cache_configured: bool,
        gossip_enabled: bool,
        seed_endpoints_configured: usize,
    ) {
        let mut status = self.bootstrap_status.write();
        status.enabled = enabled;
        status.peer_cache_configured = peer_cache_configured;
        status.gossip_enabled = gossip_enabled;
        status.seed_endpoints_configured = seed_endpoints_configured as u64;
    }

    /// Records startup discovery readiness without exposing endpoint values.
    ///
    /// `detail` must be a stable bucket list such as
    /// `missing=peer_cache_path,seed_endpoints`; callers must not include raw
    /// file paths, public endpoints, seed URLs, peer IDs, client IPs,
    /// destinations, DNS contents, packet payloads, chat plaintext, voucher
    /// secrets, private keys, or wallet-level traffic.
    pub fn record_startup_self_check(
        &self,
        now: u64,
        status_bucket: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let status_bucket = status_bucket.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.startup_self_check_status = Some(status_bucket.clone());
            status.startup_self_check_detail = Some(detail.clone());
            status.startup_self_check_at = Some(now);
        }
        let outcome = match status_bucket.as_str() {
            "ready" => "accepted",
            "skipped" => "ignored",
            _ => "warning",
        };
        self.record_audit_event(now, "startup_self_check", outcome, detail);
    }

    /// Records self descriptor registration status.
    pub fn record_self_descriptor_status(
        &self,
        now: u64,
        source_status: impl Into<String>,
        detail: impl Into<String>,
    ) {
        let source_status = source_status.into();
        let detail = detail.into();
        {
            let mut status = self.bootstrap_status.write();
            status.self_descriptor_status = Some(source_status.clone());
            status.self_descriptor_at = Some(now);
        }
        self.record_audit_event(now, "self_descriptor", source_status, detail);
    }

    /// Records one bounded external delivery-cache witness round.
    ///
    /// Valid adverse responses take precedence over availability. A threshold
    /// is satisfied only by `advanced + idempotent`; signed stale, conflict,
    /// and gap responses are verified evidence but never count as protection.
    /// The method returns the stable status bucket used by startup policy.
    pub fn record_client_delivery_witness_round(
        &self,
        now: u64,
        generation: u64,
        required_for_restore: bool,
        minimum_verified: usize,
        round: PeerStoreVerifiedDeliveryWitnessRound,
    ) -> &'static str {
        let accepted = round.advanced.saturating_add(round.idempotent);
        let status_bucket = if round.configured == 0 {
            "disabled"
        } else if round.stale > 0 {
            "rollback_detected"
        } else if round.conflicts > 0 {
            "conflict"
        } else if round.gaps > 0 {
            "gap"
        } else if accepted >= minimum_verified as u64 {
            "verified"
        } else if accepted > 0 {
            "partial"
        } else {
            "unavailable"
        };
        {
            let mut status = self.bootstrap_status.write();
            status.last_client_delivery_witness_status = Some(status_bucket.to_string());
            status.last_client_delivery_witness_checked_at = Some(now);
            status.last_client_delivery_witness_generation = generation;
            status.last_client_delivery_witness_required = required_for_restore;
            status.last_client_delivery_witness_minimum_verified = minimum_verified as u64;
            status.last_client_delivery_witness_configured = round.configured;
            status.last_client_delivery_witness_attempted = round.attempted;
            status.last_client_delivery_witness_verified = round.verified;
            status.last_client_delivery_witness_advanced = round.advanced;
            status.last_client_delivery_witness_idempotent = round.idempotent;
            status.last_client_delivery_witness_stale = round.stale;
            status.last_client_delivery_witness_conflicts = round.conflicts;
            status.last_client_delivery_witness_gaps = round.gaps;
            status.last_client_delivery_witness_failed = round.failed;
        }
        let audit_outcome = match status_bucket {
            "verified" | "disabled" => "accepted",
            "partial" | "unavailable" => "warning",
            _ => "rejected",
        };
        self.record_audit_event(
            now,
            "client_delivery_cache_external_witness",
            audit_outcome,
            format!(
                "generation={generation} status={status_bucket} required={} minimum_verified={} configured={} attempted={} verified={} advanced={} idempotent={} stale={} conflicts={} gaps={} failed={}",
                required_for_restore,
                minimum_verified,
                round.configured,
                round.attempted,
                round.verified,
                round.advanced,
                round.idempotent,
                round.stale,
                round.conflicts,
                round.gaps,
                round.failed,
            ),
        );
        status_bucket
    }

    /// Records outbound gossip round result.
    pub fn record_gossip_round(
        &self,
        now: u64,
        attempted: usize,
        succeeded: usize,
        seed_attempted: usize,
        failure_reason: Option<String>,
    ) {
        let failed = attempted.saturating_sub(succeeded);
        let status_bucket = if attempted == 0 && failure_reason.is_some() {
            "failed"
        } else if attempted == 0 {
            "idle"
        } else if succeeded == attempted {
            "healthy"
        } else if succeeded > 0 {
            "degraded"
        } else {
            "failed"
        };
        let failure_reason = if failed > 0 || (attempted == 0 && status_bucket == "failed") {
            Some(failure_reason.unwrap_or_else(|| "unknown".to_string()))
        } else {
            None
        };
        let consecutive_gossip_failures;
        {
            let mut status = self.bootstrap_status.write();
            if succeeded > 0 {
                status.consecutive_gossip_failures = 0;
                status.last_gossip_success_at = Some(now);
                status.recovery_status = Some("success".to_string());
                status.recovery_detail = Some(format!(
                    "gossip_recovered attempted={attempted} succeeded={succeeded} seed_attempted={seed_attempted}"
                ));
                status.recovery_at = Some(now);
            } else if attempted > 0 || failure_reason.is_some() {
                status.consecutive_gossip_failures =
                    status.consecutive_gossip_failures.saturating_add(1);
            }
            consecutive_gossip_failures = status.consecutive_gossip_failures;
            status.last_gossip_attempted = attempted as u64;
            status.last_gossip_seed_attempted = seed_attempted as u64;
            status.last_gossip_succeeded = succeeded as u64;
            status.last_gossip_failed = failed as u64;
            status.last_gossip_status = Some(status_bucket.to_string());
            status.last_gossip_failure_reason = failure_reason.clone();
            status.last_gossip_round_at = Some(now);
        }
        let outcome = match status_bucket {
            "healthy" => "accepted",
            "idle" => "ignored",
            _ => "warning",
        };
        let reason_detail = failure_reason
            .as_deref()
            .map(|reason| format!(" failure_reason={reason}"))
            .unwrap_or_default();
        self.record_audit_event(
            now,
            "outbound_gossip_round",
            outcome,
            format!(
                "attempted={attempted} succeeded={succeeded} failed={failed} seed_attempted={seed_attempted} status={status_bucket} consecutive_failures={consecutive_gossip_failures}{reason_detail}"
            ),
        );
    }

    /// Records one privacy-safe Directory proof-gossip convergence round.
    ///
    /// [DIRECTORY-GOSSIP-RELIABILITY 2026-07-28 by Codex] Result buckets are
    /// aggregate-only. They intentionally distinguish replica convergence
    /// misses from service, rate-limit, protocol, and transport failures while
    /// retaining no peer, endpoint, producer, descriptor, block, proof, route,
    /// message, payload, client, or traffic dimensions.
    pub fn record_directory_proof_gossip_round(
        &self,
        now: u64,
        round: PeerStoreDirectoryProofGossipRound,
    ) {
        let fallback_frames_attempted =
            round.frames_attempted.saturating_sub(round.peers_attempted);
        let acceptance_percent = if round.capable == 0 {
            0
        } else {
            round.accepted.saturating_mul(100) / round.capable
        };
        let negotiation_failed = round.capable == 0
            && (round.replica_unavailable > 0
                || round.rate_limited > 0
                || round.protocol_rejected > 0
                || round.transport_failed > 0);
        let status_bucket = if round.capability_checked == 0 {
            "idle"
        } else if negotiation_failed {
            // [DISCOVERY-GOSSIP-ISOLATION 2026-07-28 by Codex] An unavailable
            // or malformed negotiation surface is not a valid legacy-only
            // response. Keep this aggregate-only and fail visibly degraded.
            "degraded"
        } else if round.capable == 0 {
            "legacy_only"
        } else if round.accepted >= round.capable {
            "converged"
        } else if round.accepted > 0 {
            "partial"
        } else if round.evidence_rejected > 0
            && round.replica_unavailable == 0
            && round.rate_limited == 0
            && round.protocol_rejected == 0
            && round.transport_failed == 0
        {
            "evidence_diverged"
        } else {
            "degraded"
        };

        let consecutive_zero_acceptance_rounds;
        {
            let mut status = self.bootstrap_status.write();
            if round.accepted > 0 {
                status.consecutive_directory_proof_gossip_zero_acceptance_rounds = 0;
                status.last_directory_proof_gossip_success_at = Some(now);
            } else if round.capable > 0 {
                status.consecutive_directory_proof_gossip_zero_acceptance_rounds = status
                    .consecutive_directory_proof_gossip_zero_acceptance_rounds
                    .saturating_add(1);
            }
            consecutive_zero_acceptance_rounds =
                status.consecutive_directory_proof_gossip_zero_acceptance_rounds;
            status.last_directory_proof_gossip_status = Some(status_bucket.to_string());
            status.last_directory_proof_gossip_capability_checked = round.capability_checked as u64;
            status.last_directory_proof_gossip_capable = round.capable as u64;
            status.last_directory_proof_gossip_peers_attempted = round.peers_attempted as u64;
            status.last_directory_proof_gossip_frames_attempted = round.frames_attempted as u64;
            status.last_directory_proof_gossip_fallback_frames_attempted =
                fallback_frames_attempted as u64;
            status.last_directory_proof_gossip_accepted = round.accepted as u64;
            status.last_directory_proof_gossip_acceptance_percent = acceptance_percent as u64;
            status.last_directory_proof_gossip_evidence_rejected = round.evidence_rejected as u64;
            status.last_directory_proof_gossip_replica_unavailable =
                round.replica_unavailable as u64;
            status.last_directory_proof_gossip_rate_limited = round.rate_limited as u64;
            status.last_directory_proof_gossip_protocol_rejected = round.protocol_rejected as u64;
            status.last_directory_proof_gossip_transport_failed = round.transport_failed as u64;
            status.last_directory_proof_gossip_round_at = Some(now);
        }

        let outcome = match status_bucket {
            "converged" => "accepted",
            "idle" | "legacy_only" => "ignored",
            _ => "warning",
        };
        self.record_audit_event(
            now,
            "directory_proof_gossip_round",
            outcome,
            format!(
                "status={status_bucket} capability_checked={} capable={} peers_attempted={} frames_attempted={} fallback_frames_attempted={} accepted={} acceptance_percent={} evidence_rejected={} replica_unavailable={} rate_limited={} protocol_rejected={} transport_failed={} consecutive_zero_acceptance_rounds={consecutive_zero_acceptance_rounds}",
                round.capability_checked,
                round.capable,
                round.peers_attempted,
                round.frames_attempted,
                fallback_frames_attempted,
                round.accepted,
                acceptance_percent,
                round.evidence_rejected,
                round.replica_unavailable,
                round.rate_limited,
                round.protocol_rejected,
                round.transport_failed,
            ),
        );
    }

    /// Records the next outbound gossip schedule.
    ///
    /// This is intentionally aggregate scheduler state only. It never stores
    /// peer URLs, seed URLs, client identifiers, ciphertext, plaintext, packet
    /// payloads, voucher secrets, private keys, or wallet-level traffic.
    pub fn record_gossip_schedule(
        &self,
        now: u64,
        backpressure_active: bool,
        next_delay_seconds: u64,
        jitter_seconds: i64,
    ) {
        {
            let mut status = self.bootstrap_status.write();
            status.gossip_backpressure_active = backpressure_active;
            status.next_gossip_delay_seconds = Some(next_delay_seconds);
            status.next_gossip_jitter_seconds = jitter_seconds;
            status.last_gossip_schedule_at = Some(now);
        }

        if backpressure_active {
            self.record_audit_event(
                now,
                "outbound_gossip_backpressure",
                "limited",
                format!("next_delay_seconds={next_delay_seconds} jitter_seconds={jitter_seconds}"),
            );
        }
    }

    /// Returns the current consecutive outbound gossip failure count.
    ///
    /// The scheduler uses this aggregate counter to decide whether to reduce
    /// fanout. It is exposed as a method so task code does not need to inspect
    /// the full status snapshot or any peer-level state.
    pub fn consecutive_gossip_failures(&self) -> u64 {
        self.bootstrap_status.read().consecutive_gossip_failures
    }

    /// Records a discovery gossip exchange timestamp.
    pub fn mark_gossip_at(&self, now: u64) {
        self.counters.last_gossip_at.store(now, Ordering::Relaxed);
    }

    /// Records an allow/deny policy rejection.
    pub fn record_policy_rejected(&self, now: u64, detail: impl Into<String>) {
        self.counters
            .policy_rejected
            .fetch_add(1, Ordering::Relaxed);
        self.record_audit_event(now, "gossip_policy_rejected", "rejected", detail);
    }

    /// Records a rate-limited inbound request.
    pub fn record_rate_limited(&self, now: u64, detail: impl Into<String>) {
        self.counters.rate_limited.fetch_add(1, Ordering::Relaxed);
        self.record_audit_event(now, "gossip_rate_limited", "limited", detail);
    }
}
