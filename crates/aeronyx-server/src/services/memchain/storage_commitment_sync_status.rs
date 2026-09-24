// File: crates/aeronyx-server/src/services/memchain/storage_commitment_sync_status.rs
//! Commitment follower/coordinator sync telemetry and readiness.
//!
//! [MEMCHAIN-COMMITMENT-SYNC-STATUS-SPLIT 2026-09-25 by Codex] Keep the
//! aggregate sync-status domain isolated while preserving the parent module's
//! inherent-method and helper paths. This module does not own schema, chain
//! persistence, lease, authority, or network policy.

use super::*;

/// Advances an event timestamp without allowing a backwards wall-clock step to
/// make process-local operational evidence regress.
///
/// [STICKY-SECURITY-EVIDENCE 2026-07-29 by Codex] Source-recovery domains use
/// one helper so latest-result and sticky security timestamps share exactly the
/// same monotonic policy. This helper carries no identity or protocol material.
pub(super) fn record_monotonic_observation(last_observed_at: &mut Option<u64>, now: u64) -> u64 {
    let observed_at = last_observed_at.map_or(now, |previous| previous.max(now));
    *last_observed_at = Some(observed_at);
    observed_at
}

/// Derives fail-closed follower readiness without changing the legacy state.
///
/// [FOLLOWER-EFFECTIVE-READINESS 2026-07-30 by Codex] `current` proves block
/// convergence only. A required exact-tip certificate must also be ready before
/// the additive readiness contract reports `ready`. Certified carrier recovery
/// remains degraded because it cannot prove that the producer has no later tip.
pub(super) fn record_commitment_follower_readiness(
    role: &str,
    enabled: bool,
    sync_state: &str,
    certificate_policy_state: &str,
    certificate_policy_ready: bool,
    convergence_observation_is_stale: bool,
) -> RecordCommitmentFollowerReadiness {
    if role != "follower" || !enabled {
        return RecordCommitmentFollowerReadiness::NotApplicable;
    }
    if sync_state == "stopped" {
        return RecordCommitmentFollowerReadiness::Stopped;
    }
    if certificate_policy_state == "security_stopped" {
        return RecordCommitmentFollowerReadiness::SecurityStopped;
    }
    if certificate_policy_state == "configuration_error" {
        return RecordCommitmentFollowerReadiness::ConfigurationError;
    }
    if sync_state == "current" && convergence_observation_is_stale {
        return RecordCommitmentFollowerReadiness::Stale;
    }

    match sync_state {
        "starting" => RecordCommitmentFollowerReadiness::Starting,
        "catching_up" | "checkpointing" => RecordCommitmentFollowerReadiness::Synchronizing,
        "backoff" => RecordCommitmentFollowerReadiness::Backoff,
        "certified_recovered" => RecordCommitmentFollowerReadiness::CertifiedRecovered,
        "current" => match certificate_policy_state {
            "disabled" => RecordCommitmentFollowerReadiness::Ready,
            "ready" if certificate_policy_ready => RecordCommitmentFollowerReadiness::Ready,
            "source_unavailable" => RecordCommitmentFollowerReadiness::SourceUnavailable,
            "waiting_for_convergence" | "waiting_for_certificate" | "ready" => {
                RecordCommitmentFollowerReadiness::WaitingForCertificate
            }
            _ => RecordCommitmentFollowerReadiness::WaitingForCertificate,
        },
        _ => RecordCommitmentFollowerReadiness::Synchronizing,
    }
}

pub(super) fn privacy_safe_sync_error_code(reason: &str) -> String {
    match reason {
        "invalid_pinned_coordinator"
        | "invalid_authority_carrier_policy"
        | "coordinator_self_reference"
        | "http_client_init_failed"
        | "pinned_coordinator_unavailable"
        | "pinned_coordinator_missing_endpoint"
        | "pinned_coordinator_invalid_endpoint"
        | "request_encode_failed"
        | "request_timeout"
        | "request_connect"
        | "request_body"
        | "request_decode"
        | "request_request"
        | "request_unknown"
        | "response_body_timeout"
        | "response_body_connect"
        | "response_body_body"
        | "response_body_decode"
        | "response_body_request"
        | "response_body_unknown"
        | "response_too_large"
        | "invalid_response_frame"
        | "unexpected_response_message"
        | "response_request_mismatch"
        | "response_responder_mismatch"
        | "stale_response"
        | "response_page_too_large"
        | "invalid_response_signature"
        | "invalid_local_genesis"
        | "coordinator_rollback_detected"
        | "empty_page_tip_mismatch"
        | "unexpected_blocks_at_current_tip"
        | "unexpected_block_proposer"
        | "commitment_chain_verification_failed"
        | "pagination_state_mismatch"
        | "terminal_tip_mismatch"
        | "storage_append_rejected"
        | "checkpoint_request_timeout"
        | "checkpoint_request_connect"
        | "checkpoint_request_body"
        | "checkpoint_request_decode"
        | "checkpoint_request_request"
        | "checkpoint_request_unknown"
        | "local_checkpoint_unavailable"
        | "local_checkpoint_tip_mismatch"
        | "invalid_checkpoint_frame"
        | "unexpected_checkpoint_message"
        | "checkpoint_chain_mismatch"
        | "checkpoint_request_mismatch"
        | "checkpoint_responder_mismatch"
        | "stale_checkpoint_response"
        | "invalid_checkpoint_signature"
        | "invalid_checkpoint_genesis"
        | "checkpoint_height_mismatch"
        | "checkpoint_tip_inconsistent"
        | "local_checkpoint_height_mismatch"
        | "checkpoint_evidence_persist_failed"
        | "signed_checkpoint_remote_behind"
        | "signed_checkpoint_divergence" => reason.to_string(),
        // Exact status codes are useful in process logs but add needless
        // cardinality to public health data, so all 3-digit responses collapse
        // to one stable evidence code.
        _ if reason.strip_prefix("http_status_").is_some_and(|status| {
            status.len() == 3 && status.bytes().all(|byte| byte.is_ascii_digit())
        }) =>
        {
            "http_status_error".to_string()
        }
        _ if reason
            .strip_prefix("checkpoint_http_status_")
            .is_some_and(|status| {
                status.len() == 3 && status.bytes().all(|byte| byte.is_ascii_digit())
            }) =>
        {
            "checkpoint_http_status_error".to_string()
        }
        _ => "internal_sync_error".to_string(),
    }
}

fn push_commitment_sync_event(
    runtime: &mut RecordCommitmentSyncRuntime,
    timestamp: u64,
    kind: &'static str,
    error_code: Option<String>,
    next_poll_at: Option<u64>,
) {
    let event = RecordCommitmentSyncEvent {
        sequence: runtime.next_event_sequence,
        timestamp,
        kind: kind.to_string(),
        error_code,
        consecutive_failures: runtime.consecutive_failures,
        next_poll_at,
    };
    runtime.next_event_sequence = runtime.next_event_sequence.saturating_add(1);
    if runtime.recent_events.len() == COMMITMENT_SYNC_EVENT_CAPACITY {
        runtime.recent_events.pop_front();
    }
    runtime.recent_events.push_back(event);
}

impl MemoryStorage {
    /// Marks the start of one scheduled bounded follower pull round.
    ///
    /// This entry point remains stable for existing callers. Event-driven
    /// rounds use `record_commitment_sync_announcement_attempt` so operators
    /// can distinguish fallback polling from authenticated wake-ups.
    pub fn record_commitment_sync_attempt(&self, now: u64) {
        self.record_commitment_sync_attempt_with_trigger(now, "scheduled");
    }

    /// Marks the start of one bounded pull round triggered by an authenticated
    /// block announcement from the pinned coordinator.
    pub fn record_commitment_sync_announcement_attempt(&self, now: u64) {
        self.record_commitment_sync_attempt_with_trigger(now, "block_announce");
    }

    fn record_commitment_sync_attempt_with_trigger(&self, now: u64, trigger: &'static str) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        runtime.state = "syncing";
        runtime.last_trigger = trigger;
        runtime.last_attempt_at = Some(now);
        runtime.next_poll_at = None;
    }

    /// Records how one authenticated coordinator block announcement was
    /// handled by the follower scheduler.
    ///
    /// This method does not claim that the announced block was imported or
    /// canonical. The signed pull and checkpoint path remains the only path
    /// that can advance replicated commitment state.
    pub fn record_commitment_sync_announcement(
        &self,
        now: u64,
        announced_height: u64,
        disposition: RecordCommitmentAnnouncementDisposition,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        runtime.last_announcement_at = Some(now);
        runtime.last_announced_height = Some(
            runtime
                .last_announced_height
                .map_or(announced_height, |height| height.max(announced_height)),
        );
        runtime.last_announcement_result = Some(disposition.as_str());
        match disposition {
            RecordCommitmentAnnouncementDisposition::Accepted => {
                runtime.announcements_accepted_total =
                    runtime.announcements_accepted_total.saturating_add(1);
            }
            RecordCommitmentAnnouncementDisposition::Coalesced => {
                runtime.announcements_coalesced_total =
                    runtime.announcements_coalesced_total.saturating_add(1);
            }
            RecordCommitmentAnnouncementDisposition::Stale => {
                runtime.announcements_stale_total =
                    runtime.announcements_stale_total.saturating_add(1);
            }
            RecordCommitmentAnnouncementDisposition::Unavailable => {
                runtime.announcements_unavailable_total =
                    runtime.announcements_unavailable_total.saturating_add(1);
            }
        }
    }

    /// Records one coordinator-side best-effort tip announcement round.
    ///
    /// `accepted` means the peer returned exactly `202 Accepted`; `stale`
    /// means exactly `204 No Content`. Other HTTP statuses, transport errors,
    /// and preflight rejections belong in `failed`. Retry counters contain
    /// only aggregate attempts and terminal outcomes; they never expose peers,
    /// endpoints, response bodies, or timing. These counters are operational
    /// evidence only and never establish replication, consensus, fork choice,
    /// or finality.
    pub fn record_commitment_outbound_announcement(
        &self,
        now: u64,
        announced_height: u64,
        attempted: usize,
        accepted: usize,
        stale: usize,
        failed: usize,
        retries_attempted: usize,
        retries_succeeded: usize,
        retries_exhausted: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if runtime.role != "coordinator" {
            return;
        }

        let attempted = u64::try_from(attempted).unwrap_or(u64::MAX);
        let accepted = u64::try_from(accepted).unwrap_or(u64::MAX);
        let stale = u64::try_from(stale).unwrap_or(u64::MAX);
        let failed = u64::try_from(failed).unwrap_or(u64::MAX);
        let retries_attempted = u64::try_from(retries_attempted).unwrap_or(u64::MAX);
        let retries_succeeded = u64::try_from(retries_succeeded).unwrap_or(u64::MAX);
        let retries_exhausted = u64::try_from(retries_exhausted).unwrap_or(u64::MAX);
        let classified = accepted.saturating_add(stale).saturating_add(failed);
        let retry_evidence_valid = retries_succeeded <= retries_attempted
            && retries_exhausted <= retries_attempted
            && retries_succeeded.saturating_add(retries_exhausted) <= attempted;
        let result = if classified != attempted || !retry_evidence_valid {
            "failed"
        } else if attempted == 0 {
            "no_targets"
        } else if accepted == attempted {
            "all_woken"
        } else if failed == 0 {
            "delivered"
        } else if accepted.saturating_add(stale) > 0 {
            "partial"
        } else {
            "failed"
        };

        runtime.last_outbound_announcement_at = Some(now);
        runtime.last_outbound_announced_height = Some(announced_height);
        runtime.last_outbound_announcement_result = Some(result);
        runtime.outbound_announcement_rounds_total =
            runtime.outbound_announcement_rounds_total.saturating_add(1);
        runtime.outbound_announcements_attempted_total = runtime
            .outbound_announcements_attempted_total
            .saturating_add(attempted);
        runtime.outbound_announcements_accepted_total = runtime
            .outbound_announcements_accepted_total
            .saturating_add(accepted);
        runtime.outbound_announcements_stale_total = runtime
            .outbound_announcements_stale_total
            .saturating_add(stale);
        runtime.outbound_announcements_failed_total = runtime
            .outbound_announcements_failed_total
            .saturating_add(failed);
        runtime.outbound_announcement_retries_attempted_total = runtime
            .outbound_announcement_retries_attempted_total
            .saturating_add(retries_attempted);
        runtime.outbound_announcement_retries_succeeded_total = runtime
            .outbound_announcement_retries_succeeded_total
            .saturating_add(retries_succeeded);
        runtime.outbound_announcement_retries_exhausted_total = runtime
            .outbound_announcement_retries_exhausted_total
            .saturating_add(retries_exhausted);
    }

    /// Records a coordinator round that could not encode or load an audited
    /// tip and therefore made no peer delivery claim.
    pub fn record_commitment_outbound_announcement_skipped(&self, now: u64) {
        let mut runtime = self.commitment_sync.write();
        if runtime.role != "coordinator" {
            return;
        }
        runtime.last_outbound_announcement_at = Some(now);
        runtime.last_outbound_announced_height = None;
        runtime.last_outbound_announcement_result = Some("skipped");
        runtime.outbound_announcement_rounds_total =
            runtime.outbound_announcement_rounds_total.saturating_add(1);
        runtime.outbound_announcement_rounds_skipped_total = runtime
            .outbound_announcement_rounds_skipped_total
            .saturating_add(1);
    }

    /// Records an in-flight coordinator round canceled for a strictly newer
    /// audited tip.
    ///
    /// Supersession makes no per-peer delivery claim because a canceled HTTP
    /// request may or may not have reached its follower. It is process-local
    /// scheduling evidence only and cannot affect chain, witness, lease,
    /// consensus, fork-choice, or finality state.
    pub fn record_commitment_outbound_announcement_superseded(&self, now: u64) {
        let mut runtime = self.commitment_sync.write();
        if runtime.role != "coordinator" {
            return;
        }
        runtime.last_outbound_announcement_at = Some(now);
        runtime.last_outbound_announced_height = None;
        runtime.last_outbound_announcement_result = Some("superseded");
        runtime.outbound_announcement_rounds_total =
            runtime.outbound_announcement_rounds_total.saturating_add(1);
        runtime.outbound_announcement_rounds_superseded_total = runtime
            .outbound_announcement_rounds_superseded_total
            .saturating_add(1);
    }

    /// Records one fully verified response page after all included blocks have
    /// been accepted by the local transactional chain store.
    pub fn record_commitment_sync_page_success(
        &self,
        now: u64,
        verified_blocks: u64,
        remote_tip_height: u64,
        has_more: bool,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        runtime.state = if has_more {
            "catching_up"
        } else {
            "checkpointing"
        };
        runtime.last_success_at = Some(now);
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.pages_received_total = runtime.pages_received_total.saturating_add(1);
        runtime.blocks_received_total = runtime
            .blocks_received_total
            .saturating_add(verified_blocks);
    }

    /// Marks a follower current only after a signed equal-tip checkpoint.
    ///
    /// A terminal block-range page alone is insufficient because the remote
    /// tip may move between page construction and local append. This explicit
    /// gate keeps `current` equivalent to a signature-verified convergence
    /// observation.
    pub fn record_commitment_sync_checkpoint_success(&self, now: u64, remote_tip_height: u64) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        let recovered = runtime.consecutive_failures > 0;
        let confirmed_at = runtime
            .follower_convergence_confirmed_at
            .map_or(now, |previous| previous.max(now));
        runtime.state = "current";
        runtime.last_success_at = Some(now);
        runtime.follower_convergence_confirmed_at = Some(confirmed_at);
        runtime.follower_readiness_stale_after = runtime
            .follower_readiness_max_age_secs
            .map(|max_age_secs| confirmed_at.saturating_add(max_age_secs));
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.consecutive_failures = 0;
        runtime.last_error_code = None;
        if recovered {
            runtime.last_recovered_at = Some(now);
            runtime.recovery_events_total = runtime.recovery_events_total.saturating_add(1);
            push_commitment_sync_event(&mut runtime, now, "recovered", None, None);
        }
    }

    /// Marks a follower restored to an exact threshold-certified local tip.
    ///
    /// [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex] This is intentionally
    /// distinct from `current`: while the producer is unavailable, a carrier
    /// and immutable witness certificate can prove the recovered prefix but
    /// cannot prove that no later producer block exists. The state contains no
    /// carrier identity, endpoint, route, hash, signature, or user metadata.
    pub fn record_commitment_sync_certified_recovery_success(
        &self,
        now: u64,
        certified_tip_height: u64,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        runtime.state = "certified_recovered";
        runtime.last_success_at = Some(now);
        runtime.follower_convergence_confirmed_at = None;
        runtime.follower_readiness_stale_after = None;
        runtime.remote_tip_height = Some(certified_tip_height);
        runtime.consecutive_failures = 0;
        runtime.last_error_code = None;
    }

    /// Records one terminal follower authority-proof retrieval outcome.
    ///
    /// [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] The storage contract
    /// accepts only one source-blind disposition and the number of actual
    /// bounded carrier requests. It cannot retain source identities,
    /// endpoints, authority proofs, epochs, heights, hashes, signatures,
    /// errors, or routes, and this evidence never participates in authority.
    pub(crate) fn record_commitment_authority_sync_outcome(
        &self,
        now: u64,
        disposition: RecordCommitmentAuthoritySyncDisposition,
        carrier_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let observed_at = record_monotonic_observation(&mut runtime.last_authority_sync_at, now);
        let carrier_attempts = u64::try_from(carrier_attempts).unwrap_or(u64::MAX);

        runtime.last_authority_sync_result = Some(disposition.as_str());
        runtime.authority_sync_rounds_total = runtime.authority_sync_rounds_total.saturating_add(1);
        runtime.authority_carrier_attempts_total = runtime
            .authority_carrier_attempts_total
            .saturating_add(carrier_attempts);

        match disposition {
            RecordCommitmentAuthoritySyncDisposition::Coordinator => {
                runtime.authority_coordinator_success_total = runtime
                    .authority_coordinator_success_total
                    .saturating_add(1);
            }
            RecordCommitmentAuthoritySyncDisposition::CarrierRecovered => {
                runtime.last_authority_carrier_recovered_at = Some(observed_at);
                runtime.authority_carrier_recoveries_total =
                    runtime.authority_carrier_recoveries_total.saturating_add(1);
            }
            RecordCommitmentAuthoritySyncDisposition::AvailabilityExhausted => {
                runtime.authority_availability_exhausted_total = runtime
                    .authority_availability_exhausted_total
                    .saturating_add(1);
            }
            RecordCommitmentAuthoritySyncDisposition::SecurityStopped => {
                runtime.last_authority_security_stop_at = Some(observed_at);
                runtime.authority_security_stops_total =
                    runtime.authority_security_stops_total.saturating_add(1);
            }
        }
    }

    /// Records one identity-blind authority-carrier circuit observation.
    ///
    /// [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Authority handover,
    /// block-page, and certificate recovery have independent typed circuits.
    /// This follower-only projection retains only a current anonymous cooling
    /// gauge and saturating scheduler counters; it cannot affect selection.
    pub(crate) fn record_commitment_authority_carrier_circuit_observation(
        &self,
        cooling_slots: usize,
        cooldown_skips: usize,
        half_open_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let cooldown_skips = u64::try_from(cooldown_skips).unwrap_or(u64::MAX);
        let half_open_attempts = u64::try_from(half_open_attempts).unwrap_or(u64::MAX);

        runtime.authority_carrier_cooling_slots = cooling_slots;
        runtime.authority_carrier_cooldown_skips_total = runtime
            .authority_carrier_cooldown_skips_total
            .saturating_add(cooldown_skips);
        runtime.authority_carrier_half_open_attempts_total = runtime
            .authority_carrier_half_open_attempts_total
            .saturating_add(half_open_attempts);
    }

    /// Records one terminal commitment-block page retrieval outcome.
    ///
    /// [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex] The counters are
    /// mutually exclusive per page retrieval, process-local, and follower-only.
    /// `carrier_attempts` counts actual bounded carrier requests, excluding the
    /// direct coordinator request. No source identity, endpoint, block, route,
    /// certificate, hash, signature, or raw failure can enter this state.
    pub(crate) fn record_commitment_block_page_pull_outcome(
        &self,
        now: u64,
        disposition: RecordCommitmentBlockPagePullDisposition,
        carrier_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let observed_at = record_monotonic_observation(&mut runtime.last_block_page_pull_at, now);
        let carrier_attempts = u64::try_from(carrier_attempts).unwrap_or(u64::MAX);

        runtime.last_block_page_pull_result = Some(disposition.as_str());
        runtime.block_page_pulls_total = runtime.block_page_pulls_total.saturating_add(1);
        runtime.block_carrier_attempts_total = runtime
            .block_carrier_attempts_total
            .saturating_add(carrier_attempts);

        match disposition {
            RecordCommitmentBlockPagePullDisposition::Coordinator => {
                runtime.block_page_coordinator_success_total = runtime
                    .block_page_coordinator_success_total
                    .saturating_add(1);
            }
            RecordCommitmentBlockPagePullDisposition::CarrierRecovered => {
                runtime.last_block_carrier_recovered_at = Some(observed_at);
                runtime.block_carrier_recoveries_total =
                    runtime.block_carrier_recoveries_total.saturating_add(1);
            }
            RecordCommitmentBlockPagePullDisposition::AvailabilityExhausted => {
                runtime.block_page_availability_exhausted_total = runtime
                    .block_page_availability_exhausted_total
                    .saturating_add(1);
            }
            RecordCommitmentBlockPagePullDisposition::SecurityStopped => {
                // [STICKY-SECURITY-EVIDENCE 2026-07-29 by Codex] A later
                // successful source may replace the latest result, but must
                // not erase when this fail-closed event was last observed.
                runtime.last_block_page_security_stop_at = Some(observed_at);
                runtime.block_page_security_stops_total =
                    runtime.block_page_security_stops_total.saturating_add(1);
            }
        }
    }

    /// Records one identity-blind snapshot of block-carrier circuit activity.
    ///
    /// [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] The current gauge
    /// and cumulative scheduling counters are follower-only and process-local.
    /// They cannot retain slot order, source identities, endpoints, status
    /// codes, errors, timing, blocks, certificates, payloads, or routes, and
    /// they never participate in source selection or chain decisions.
    pub(crate) fn record_commitment_block_carrier_circuit_observation(
        &self,
        cooling_slots: usize,
        cooldown_skips: usize,
        half_open_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let cooldown_skips = u64::try_from(cooldown_skips).unwrap_or(u64::MAX);
        let half_open_attempts = u64::try_from(half_open_attempts).unwrap_or(u64::MAX);

        runtime.block_carrier_cooling_slots = cooling_slots;
        runtime.block_carrier_cooldown_skips_total = runtime
            .block_carrier_cooldown_skips_total
            .saturating_add(cooldown_skips);
        runtime.block_carrier_half_open_attempts_total = runtime
            .block_carrier_half_open_attempts_total
            .saturating_add(half_open_attempts);
    }

    /// Records one identity-blind certificate-carrier circuit observation.
    ///
    /// [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] This state is
    /// intentionally separate from block-page carrier health. It stores only a
    /// current anonymous cooling-slot gauge and saturating process counters;
    /// no source identity, slot order, endpoint, timing, error, certificate, or
    /// chain material can enter the runtime contract.
    pub(crate) fn record_commitment_certificate_carrier_circuit_observation(
        &self,
        cooling_slots: usize,
        cooldown_skips: usize,
        half_open_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let cooldown_skips = u64::try_from(cooldown_skips).unwrap_or(u64::MAX);
        let half_open_attempts = u64::try_from(half_open_attempts).unwrap_or(u64::MAX);

        runtime.certificate_carrier_cooling_slots = cooling_slots;
        runtime.certificate_carrier_cooldown_skips_total = runtime
            .certificate_carrier_cooldown_skips_total
            .saturating_add(cooldown_skips);
        runtime.certificate_carrier_half_open_attempts_total = runtime
            .certificate_carrier_half_open_attempts_total
            .saturating_add(half_open_attempts);
    }

    /// Records one terminal checkpoint-certificate retrieval outcome.
    ///
    /// [FOLLOWER-CERTIFICATE-TELEMETRY 2026-07-29 by Codex] This method keeps
    /// only mutually exclusive aggregate outcomes and the number of bounded
    /// carrier requests. It deliberately cannot retain a source identity,
    /// endpoint, certificate frame, hash, signature, or raw error. The
    /// counters are process-local observability and never affect chain state.
    pub(crate) fn record_commitment_certificate_sync_outcome(
        &self,
        now: u64,
        disposition: RecordCommitmentCertificateSyncDisposition,
        carrier_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let observed_at = record_monotonic_observation(&mut runtime.last_certificate_sync_at, now);
        let carrier_attempts = u64::try_from(carrier_attempts).unwrap_or(u64::MAX);

        runtime.last_certificate_sync_result = Some(disposition.as_str());
        runtime.certificate_sync_rounds_total =
            runtime.certificate_sync_rounds_total.saturating_add(1);
        runtime.certificate_carrier_attempts_total = runtime
            .certificate_carrier_attempts_total
            .saturating_add(carrier_attempts);

        match disposition {
            RecordCommitmentCertificateSyncDisposition::Coordinator => {
                runtime.certificate_coordinator_success_total = runtime
                    .certificate_coordinator_success_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateSyncDisposition::CarrierRecovered => {
                runtime.last_certificate_carrier_recovered_at = Some(observed_at);
                runtime.certificate_carrier_recoveries_total = runtime
                    .certificate_carrier_recoveries_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateSyncDisposition::VerifiedUnpersisted => {
                // [CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex] A
                // cryptographically valid response is not a recovery until the
                // exact current-policy certificate is durable locally.
                runtime.certificate_verified_unpersisted_total = runtime
                    .certificate_verified_unpersisted_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateSyncDisposition::AvailabilityExhausted => {
                runtime.certificate_availability_exhausted_total = runtime
                    .certificate_availability_exhausted_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateSyncDisposition::SecurityStopped => {
                runtime.last_certificate_security_stop_at = Some(observed_at);
                runtime.certificate_security_stops_total =
                    runtime.certificate_security_stops_total.saturating_add(1);
            }
        }
    }

    /// Atomically records one coordinator certificate-backfill outcome.
    ///
    /// [CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex] Outcome and
    /// circuit aggregates share one coordinator-only write so cancellation
    /// cannot publish a terminal result without its matching scheduler
    /// evidence. The method accepts no identity-bearing or cryptographic
    /// material, and these process-local counters never influence source
    /// selection, certificate policy, chain state, or production authority.
    pub(crate) fn record_commitment_certificate_backfill_outcome(
        &self,
        now: u64,
        disposition: RecordCommitmentCertificateBackfillDisposition,
        carrier_attempts: usize,
        cooling_slots: usize,
        cooldown_skips: usize,
        half_open_attempts: usize,
    ) {
        let mut runtime = self.commitment_sync.write();
        if runtime.role != "coordinator" {
            return;
        }
        let observed_at = record_monotonic_observation(
            &mut runtime.last_coordinator_certificate_backfill_at,
            now,
        );
        let carrier_attempts = u64::try_from(carrier_attempts).unwrap_or(u64::MAX);
        let cooldown_skips = u64::try_from(cooldown_skips).unwrap_or(u64::MAX);
        let half_open_attempts = u64::try_from(half_open_attempts).unwrap_or(u64::MAX);

        runtime.last_coordinator_certificate_backfill_result = Some(disposition.as_str());
        runtime.coordinator_certificate_backfill_rounds_total = runtime
            .coordinator_certificate_backfill_rounds_total
            .saturating_add(1);
        runtime.coordinator_certificate_backfill_carrier_attempts_total = runtime
            .coordinator_certificate_backfill_carrier_attempts_total
            .saturating_add(carrier_attempts);
        runtime.coordinator_certificate_backfill_carrier_cooling_slots = cooling_slots;
        runtime.coordinator_certificate_backfill_carrier_cooldown_skips_total = runtime
            .coordinator_certificate_backfill_carrier_cooldown_skips_total
            .saturating_add(cooldown_skips);
        runtime.coordinator_certificate_backfill_carrier_half_open_attempts_total = runtime
            .coordinator_certificate_backfill_carrier_half_open_attempts_total
            .saturating_add(half_open_attempts);

        match disposition {
            RecordCommitmentCertificateBackfillDisposition::Persisted => {
                runtime.coordinator_certificate_backfill_persisted_total = runtime
                    .coordinator_certificate_backfill_persisted_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateBackfillDisposition::VerifiedUnpersisted => {
                runtime.coordinator_certificate_backfill_verified_unpersisted_total = runtime
                    .coordinator_certificate_backfill_verified_unpersisted_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateBackfillDisposition::AvailabilityExhausted => {
                runtime.coordinator_certificate_backfill_availability_exhausted_total = runtime
                    .coordinator_certificate_backfill_availability_exhausted_total
                    .saturating_add(1);
            }
            RecordCommitmentCertificateBackfillDisposition::SecurityStopped => {
                runtime.last_coordinator_certificate_backfill_security_stop_at = Some(observed_at);
                runtime.coordinator_certificate_backfill_security_stops_total = runtime
                    .coordinator_certificate_backfill_security_stops_total
                    .saturating_add(1);
            }
        }
    }

    /// Records one exact local follower certificate-policy evaluation.
    ///
    /// This is separate from transport outcome counters: an already-current
    /// durable certificate can become `ready` without any network request,
    /// while a successful response is ready only after durable local policy
    /// validation succeeds.
    ///
    /// [FOLLOWER-CERTIFICATE-READINESS 2026-07-29 by Codex]
    pub(crate) fn record_commitment_certificate_policy_evaluation(
        &self,
        now: u64,
        readiness: RecordCommitmentCertificatePolicyReadiness,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled || runtime.role != "follower" {
            return;
        }
        let observed_at = runtime
            .certificate_policy_last_evaluated_at
            .map_or(now, |previous| previous.max(now));
        runtime.certificate_policy_last_evaluated_at = Some(observed_at);
        runtime.certificate_policy_state = readiness.as_str();
        runtime.certificate_policy_ready = readiness.is_ready();
        runtime.certificate_policy_evaluated_tip_height = readiness.evaluated_tip_height();
    }

    /// Records a fail-closed follower error using only a stable allow-listed
    /// code. Free text, URLs, identities, and endpoints are never retained.
    pub fn record_commitment_sync_failure(
        &self,
        now: u64,
        reason: &str,
        consecutive_failures: u32,
        next_poll_at: u64,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        let error_code = privacy_safe_sync_error_code(reason);
        runtime.state = "backoff";
        runtime.last_failure_at = Some(now);
        runtime.next_poll_at = Some(next_poll_at);
        runtime.consecutive_failures = consecutive_failures;
        runtime.last_error_code = Some(error_code.clone());
        runtime.failure_events_total = runtime.failure_events_total.saturating_add(1);
        push_commitment_sync_event(
            &mut runtime,
            now,
            "failure",
            Some(error_code),
            Some(next_poll_at),
        );
    }

    /// Records the next normal poll time after a successful bounded round.
    pub fn schedule_next_commitment_sync_poll(&self, next_poll_at: u64) {
        let mut runtime = self.commitment_sync.write();
        if runtime.enabled {
            runtime.next_poll_at = Some(next_poll_at);
        }
    }

    /// Marks follower shutdown without fabricating a failure event.
    pub fn stop_record_commitment_sync(&self) {
        let mut runtime = self.commitment_sync.write();
        if runtime.enabled {
            runtime.state = "stopped";
            runtime.next_poll_at = None;
        }
    }

    /// Returns a privacy-safe snapshot for local APIs and heartbeat reporting.
    pub fn record_commitment_sync_status(&self) -> RecordCommitmentSyncStatus {
        self.record_commitment_sync_status_at(unix_now_secs())
    }

    /// Builds a snapshot at an explicit time for deterministic boundary tests.
    pub(super) fn record_commitment_sync_status_at(&self, now: u64) -> RecordCommitmentSyncStatus {
        // [FOLLOWER-CERTIFICATE-TIP-BINDING 2026-07-29 by Codex] A certificate
        // decision is valid only for the audited tip it evaluated. Derive the
        // externally visible state against the latest complete integrity
        // baseline so scheduler delay cannot leave an old tip reported ready.
        let audited_tip_height = self
            .commitment_integrity
            .read()
            .as_ref()
            .map(|integrity| integrity.verified_tip_height);
        let runtime = self.commitment_sync.read();
        let certificate_policy_evaluation_is_stale =
            runtime.certificate_policy_evaluated_tip_height.is_some()
                && runtime.certificate_policy_evaluated_tip_height != audited_tip_height;
        let certificate_policy_state = if certificate_policy_evaluation_is_stale {
            if audited_tip_height.is_some() {
                "waiting_for_certificate"
            } else {
                "waiting_for_convergence"
            }
        } else {
            runtime.certificate_policy_state
        };
        let certificate_policy_ready =
            runtime.certificate_policy_ready && !certificate_policy_evaluation_is_stale;
        // [FOLLOWER-READINESS-FRESHNESS 2026-07-30 by Codex] `current` is a
        // recent signed producer observation, not an eternal process claim.
        // The task marks `syncing` before network I/O, while this deadline also
        // catches a follower that vanished between scheduled rounds.
        let convergence_observation_is_stale = runtime.role == "follower"
            && runtime.enabled
            && runtime.state == "current"
            && runtime
                .follower_readiness_stale_after
                .is_some_and(|deadline| now >= deadline);
        let follower_readiness = record_commitment_follower_readiness(
            runtime.role,
            runtime.enabled,
            runtime.state,
            certificate_policy_state,
            certificate_policy_ready,
            convergence_observation_is_stale,
        );
        RecordCommitmentSyncStatus {
            contract_version: "record_commitment_sync.v1",
            role: runtime.role.to_string(),
            state: runtime.state.to_string(),
            follower_readiness_state: follower_readiness.as_str().to_string(),
            follower_fully_ready: follower_readiness.is_fully_ready(),
            follower_convergence_confirmed_at: runtime.follower_convergence_confirmed_at,
            follower_readiness_stale_after: runtime.follower_readiness_stale_after,
            enabled: runtime.enabled,
            last_trigger: runtime.last_trigger.to_string(),
            last_announcement_at: runtime.last_announcement_at,
            last_announced_height: runtime.last_announced_height,
            last_announcement_result: runtime.last_announcement_result.map(str::to_string),
            announcements_accepted_total: runtime.announcements_accepted_total,
            announcements_coalesced_total: runtime.announcements_coalesced_total,
            announcements_stale_total: runtime.announcements_stale_total,
            announcements_unavailable_total: runtime.announcements_unavailable_total,
            last_outbound_announcement_at: runtime.last_outbound_announcement_at,
            last_outbound_announced_height: runtime.last_outbound_announced_height,
            last_outbound_announcement_result: runtime
                .last_outbound_announcement_result
                .map(str::to_string),
            outbound_announcement_rounds_total: runtime.outbound_announcement_rounds_total,
            outbound_announcement_rounds_skipped_total: runtime
                .outbound_announcement_rounds_skipped_total,
            outbound_announcement_rounds_superseded_total: runtime
                .outbound_announcement_rounds_superseded_total,
            outbound_announcements_attempted_total: runtime
                .outbound_announcements_attempted_total,
            outbound_announcements_accepted_total: runtime
                .outbound_announcements_accepted_total,
            outbound_announcements_stale_total: runtime.outbound_announcements_stale_total,
            outbound_announcements_failed_total: runtime.outbound_announcements_failed_total,
            outbound_announcement_retries_attempted_total: runtime
                .outbound_announcement_retries_attempted_total,
            outbound_announcement_retries_succeeded_total: runtime
                .outbound_announcement_retries_succeeded_total,
            outbound_announcement_retries_exhausted_total: runtime
                .outbound_announcement_retries_exhausted_total,
            last_authority_sync_at: runtime.last_authority_sync_at,
            last_authority_sync_result: runtime
                .last_authority_sync_result
                .map(str::to_string),
            last_authority_carrier_recovered_at: runtime
                .last_authority_carrier_recovered_at,
            authority_sync_rounds_total: runtime.authority_sync_rounds_total,
            authority_coordinator_success_total: runtime.authority_coordinator_success_total,
            authority_carrier_attempts_total: runtime.authority_carrier_attempts_total,
            authority_carrier_recoveries_total: runtime.authority_carrier_recoveries_total,
            authority_availability_exhausted_total: runtime
                .authority_availability_exhausted_total,
            authority_security_stops_total: runtime.authority_security_stops_total,
            last_authority_security_stop_at: runtime.last_authority_security_stop_at,
            authority_carrier_cooling_slots: runtime.authority_carrier_cooling_slots,
            authority_carrier_cooldown_skips_total: runtime
                .authority_carrier_cooldown_skips_total,
            authority_carrier_half_open_attempts_total: runtime
                .authority_carrier_half_open_attempts_total,
            last_block_page_pull_at: runtime.last_block_page_pull_at,
            last_block_page_pull_result: runtime
                .last_block_page_pull_result
                .map(str::to_string),
            last_block_carrier_recovered_at: runtime.last_block_carrier_recovered_at,
            block_page_pulls_total: runtime.block_page_pulls_total,
            block_page_coordinator_success_total: runtime
                .block_page_coordinator_success_total,
            block_carrier_attempts_total: runtime.block_carrier_attempts_total,
            block_carrier_recoveries_total: runtime.block_carrier_recoveries_total,
            block_page_availability_exhausted_total: runtime
                .block_page_availability_exhausted_total,
            block_page_security_stops_total: runtime.block_page_security_stops_total,
            last_block_page_security_stop_at: runtime.last_block_page_security_stop_at,
            block_carrier_cooling_slots: runtime.block_carrier_cooling_slots,
            block_carrier_cooldown_skips_total: runtime.block_carrier_cooldown_skips_total,
            block_carrier_half_open_attempts_total: runtime
                .block_carrier_half_open_attempts_total,
            certificate_policy_state: certificate_policy_state.to_string(),
            certificate_policy_ready,
            certificate_policy_last_evaluated_at: runtime
                .certificate_policy_last_evaluated_at,
            certificate_policy_evaluated_tip_height: runtime
                .certificate_policy_evaluated_tip_height,
            certificate_witnesses_configured: runtime.certificate_witnesses_configured,
            certificate_minimum_signers: runtime.certificate_minimum_signers,
            last_certificate_sync_at: runtime.last_certificate_sync_at,
            last_certificate_sync_result: runtime
                .last_certificate_sync_result
                .map(str::to_string),
            last_certificate_carrier_recovered_at: runtime
                .last_certificate_carrier_recovered_at,
            certificate_sync_rounds_total: runtime.certificate_sync_rounds_total,
            certificate_coordinator_success_total: runtime.certificate_coordinator_success_total,
            certificate_carrier_attempts_total: runtime.certificate_carrier_attempts_total,
            certificate_carrier_recoveries_total: runtime.certificate_carrier_recoveries_total,
            certificate_verified_unpersisted_total: runtime
                .certificate_verified_unpersisted_total,
            certificate_availability_exhausted_total: runtime
                .certificate_availability_exhausted_total,
            certificate_security_stops_total: runtime.certificate_security_stops_total,
            last_certificate_security_stop_at: runtime.last_certificate_security_stop_at,
            certificate_carrier_cooling_slots: runtime.certificate_carrier_cooling_slots,
            certificate_carrier_cooldown_skips_total: runtime
                .certificate_carrier_cooldown_skips_total,
            certificate_carrier_half_open_attempts_total: runtime
                .certificate_carrier_half_open_attempts_total,
            last_coordinator_certificate_backfill_at: runtime
                .last_coordinator_certificate_backfill_at,
            last_coordinator_certificate_backfill_result: runtime
                .last_coordinator_certificate_backfill_result
                .map(str::to_string),
            coordinator_certificate_backfill_rounds_total: runtime
                .coordinator_certificate_backfill_rounds_total,
            coordinator_certificate_backfill_persisted_total: runtime
                .coordinator_certificate_backfill_persisted_total,
            coordinator_certificate_backfill_verified_unpersisted_total: runtime
                .coordinator_certificate_backfill_verified_unpersisted_total,
            coordinator_certificate_backfill_availability_exhausted_total: runtime
                .coordinator_certificate_backfill_availability_exhausted_total,
            coordinator_certificate_backfill_security_stops_total: runtime
                .coordinator_certificate_backfill_security_stops_total,
            last_coordinator_certificate_backfill_security_stop_at: runtime
                .last_coordinator_certificate_backfill_security_stop_at,
            coordinator_certificate_backfill_carrier_attempts_total: runtime
                .coordinator_certificate_backfill_carrier_attempts_total,
            coordinator_certificate_backfill_carrier_cooling_slots: runtime
                .coordinator_certificate_backfill_carrier_cooling_slots,
            coordinator_certificate_backfill_carrier_cooldown_skips_total: runtime
                .coordinator_certificate_backfill_carrier_cooldown_skips_total,
            coordinator_certificate_backfill_carrier_half_open_attempts_total: runtime
                .coordinator_certificate_backfill_carrier_half_open_attempts_total,
            last_attempt_at: runtime.last_attempt_at,
            last_success_at: runtime.last_success_at,
            last_failure_at: runtime.last_failure_at,
            last_recovered_at: runtime.last_recovered_at,
            next_poll_at: runtime.next_poll_at,
            consecutive_failures: runtime.consecutive_failures,
            last_error_code: runtime.last_error_code.clone(),
            remote_tip_height: runtime.remote_tip_height,
            pages_received_total: runtime.pages_received_total,
            blocks_received_total: runtime.blocks_received_total,
            failure_events_total: runtime.failure_events_total,
            recovery_events_total: runtime.recovery_events_total,
            recent_events: runtime.recent_events.iter().cloned().collect(),
            privacy_policy:
                "aggregate runtime only; effective follower readiness combines task liveness, bounded signed convergence freshness, and exact-tip local certificate-policy readiness while follower authority-proof/block-page/certificate recovery plus coordinator certificate backfill expose role-isolated, source-blind terminal results, bounded attempt counts, independently scoped anonymous circuit cooling/skip/half-open aggregates, monotonic latest-observation timestamps, and sticky security-stop timestamps but no coordinator or carrier identity, circuit slot order, witness set, endpoint, authority proof, epoch, block, certificate frame, hash, signature, security reason, raw error, record commitment, owner, payload, route, or client metadata; readiness and counters are operations evidence, not authority, reputation, consensus, finality, or fork choice",
        }
    }
}
