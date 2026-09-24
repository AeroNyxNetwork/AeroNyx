// ============================================
// File: crates/aeronyx-server/src/server/chat_custody_witness_runtime.rs
// ============================================
// [CHAT-CUSTODY-WITNESS-RUNTIME 2026-09-25 by Codex] Keep signed
// custody-anchor readiness, bounded renewal, and fail-closed runtime
// supervision together; startup ordering remains owned by Server::run.
use super::*;

/// Stable fail-closed buckets for strict local custody-receipt readiness.
///
/// [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] Startup and runtime use
/// this same typed decision. These values are safe
/// for process-health output: they contain no identity, path, anchor, digest,
/// signature, endpoint, message, user, route, or payload information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CustodyWitnessReadinessBlockReason {
    CurrentAnchorUnavailable,
    ReceiptVaultInvalid,
    ReceiptPolicyInvalid,
    AdverseEvidence,
    EvidenceUnavailable,
    ThresholdUnmet,
}

impl CustodyWitnessReadinessBlockReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::CurrentAnchorUnavailable => "current_anchor_unavailable",
            Self::ReceiptVaultInvalid => "receipt_vault_invalid",
            Self::ReceiptPolicyInvalid => "receipt_policy_invalid",
            Self::AdverseEvidence => "signed_adverse_evidence",
            Self::EvidenceUnavailable => "fresh_receipt_unavailable",
            Self::ThresholdUnmet => "fresh_receipt_threshold_unmet",
        }
    }
}

impl std::fmt::Display for CustodyWitnessReadinessBlockReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub(super) const fn custody_witness_readiness_decision(
    readiness: CustodyAuditWitnessPolicyReadiness,
) -> std::result::Result<(), CustodyWitnessReadinessBlockReason> {
    match readiness {
        CustodyAuditWitnessPolicyReadiness::Ready => Ok(()),
        CustodyAuditWitnessPolicyReadiness::EvidenceUnavailable => {
            Err(CustodyWitnessReadinessBlockReason::EvidenceUnavailable)
        }
        CustodyAuditWitnessPolicyReadiness::ThresholdUnmet => {
            Err(CustodyWitnessReadinessBlockReason::ThresholdUnmet)
        }
        CustodyAuditWitnessPolicyReadiness::AdverseEvidence => {
            Err(CustodyWitnessReadinessBlockReason::AdverseEvidence)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CustodyWitnessAuditEvidence {
    pub(super) checkpoint_generation: u64,
    pub(super) evaluated_at: u64,
    pub(super) snapshot: CustodyAuditWitnessReceiptReadinessSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CustodyWitnessRenewalAttempt {
    pub(super) round: CustodyAuditWitnessRound,
    pub(super) audit: CustodyWitnessAuditEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CustodyWitnessRenewalAttemptError {
    CollectionFailed(CustodyWitnessAuditEvidence),
    Readiness(CustodyWitnessReadinessBlockReason),
}

pub(super) const CUSTODY_WITNESS_RUNTIME_AUDIT_MIN_INTERVAL_SECS: u64 = 30;
pub(super) const CUSTODY_WITNESS_RUNTIME_AUDIT_MAX_INTERVAL_SECS: u64 = 300;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CustodyWitnessRenewalStatus {
    pub(super) valid_through: u64,
    pub(super) valid_for_secs: u64,
    pub(super) warning_window_secs: u64,
    pub(super) renewal_recommended: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CustodyWitnessRenewalRetryAction {
    Attempt,
    BackingOff {
        retry_in_secs: u64,
        consecutive_failures: u32,
    },
    Exhausted {
        consecutive_failures: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CustodyWitnessRenewalRetrySchedule {
    pub(super) delay_secs: u64,
    pub(super) consecutive_failures: u32,
    pub(super) retry_before_expiry: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CustodyWitnessRenewalRetryState {
    pub(super) valid_through: Option<u64>,
    pub(super) consecutive_failures: u32,
    pub(super) retry_not_before: Option<Instant>,
    pub(super) exhausted_for_horizon: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum CustodyWitnessRuntimePhase {
    Disabled,
    #[default]
    Monitoring,
    Healthy,
    RenewalDue,
    BackingOff,
    Exhausted,
    FailedClosed,
}

impl CustodyWitnessRuntimePhase {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Monitoring => "monitoring",
            Self::Healthy => "healthy",
            Self::RenewalDue => "renewal_due",
            Self::BackingOff => "backing_off",
            Self::Exhausted => "exhausted",
            Self::FailedClosed => "failed_closed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub(super) struct CustodyWitnessRuntimeSnapshot {
    pub(super) runtime_required: bool,
    pub(super) auto_renewal_enabled: bool,
    pub(super) status: &'static str,
    pub(super) audit_interval_seconds: u64,
    pub(super) freshness_window_seconds: u64,
    pub(super) audits_total: u64,
    pub(super) renewal_attempts_total: u64,
    pub(super) renewal_successes_total: u64,
    pub(super) renewal_failures_total: u64,
    pub(super) partial_collection_successes_total: u64,
    pub(super) backoff_skips_total: u64,
    pub(super) exhausted_skips_total: u64,
    pub(super) fail_closed_total: u64,
    pub(super) consecutive_failures: u32,
    pub(super) retry_after_seconds: Option<u64>,
    pub(super) retry_before_expiry: Option<bool>,
    pub(super) quorum_valid_through: Option<u64>,
    pub(super) quorum_valid_for_seconds: Option<u64>,
    pub(super) last_checkpoint_generation: Option<u64>,
    pub(super) last_audit_at: Option<u64>,
    pub(super) last_attempt_at: Option<u64>,
    pub(super) last_success_at: Option<u64>,
    pub(super) last_failure_at: Option<u64>,
    pub(super) last_failure_reason: Option<&'static str>,
    pub(super) last_fail_closed_at: Option<u64>,
    pub(super) last_fail_closed_reason: Option<&'static str>,
}

#[derive(Debug, Default)]
pub(super) struct CustodyWitnessRuntimeState {
    pub(super) phase: CustodyWitnessRuntimePhase,
    pub(super) audits_total: u64,
    pub(super) renewal_attempts_total: u64,
    pub(super) renewal_successes_total: u64,
    pub(super) renewal_failures_total: u64,
    pub(super) partial_collection_successes_total: u64,
    pub(super) backoff_skips_total: u64,
    pub(super) exhausted_skips_total: u64,
    pub(super) fail_closed_total: u64,
    pub(super) consecutive_failures: u32,
    pub(super) retry_after_seconds: Option<u64>,
    pub(super) retry_before_expiry: Option<bool>,
    pub(super) quorum_valid_through: Option<u64>,
    pub(super) quorum_valid_for_seconds: Option<u64>,
    pub(super) last_checkpoint_generation: Option<u64>,
    pub(super) last_audit_at: Option<u64>,
    pub(super) last_attempt_at: Option<u64>,
    pub(super) last_success_at: Option<u64>,
    pub(super) last_failure_at: Option<u64>,
    pub(super) last_failure_reason: Option<&'static str>,
    pub(super) last_fail_closed_at: Option<u64>,
    pub(super) last_fail_closed_reason: Option<&'static str>,
}

#[derive(Debug)]
pub(super) struct CustodyWitnessRuntimeTelemetry {
    pub(super) runtime_required: bool,
    pub(super) auto_renewal_enabled: bool,
    pub(super) audit_interval_seconds: u64,
    pub(super) freshness_window_seconds: u64,
    pub(super) state: Mutex<CustodyWitnessRuntimeState>,
}

impl CustodyWitnessRuntimeTelemetry {
    pub(super) fn new(
        runtime_required: bool,
        auto_renewal_enabled: bool,
        max_age_secs: u64,
    ) -> Self {
        let mut state = CustodyWitnessRuntimeState::default();
        if !runtime_required {
            state.phase = CustodyWitnessRuntimePhase::Disabled;
        }
        Self {
            runtime_required,
            auto_renewal_enabled,
            audit_interval_seconds: custody_witness_runtime_audit_interval_secs(max_age_secs),
            freshness_window_seconds: max_age_secs,
            state: Mutex::new(state),
        }
    }

    pub(super) fn with_state(&self, update: impl FnOnce(&mut CustodyWitnessRuntimeState)) {
        // [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex] Telemetry must never
        // become a second availability dependency. Recover a poisoned lock and
        // retain the last aggregate state; no protocol decision reads it back.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        update(&mut state);
    }

    pub(super) fn snapshot(&self) -> CustodyWitnessRuntimeSnapshot {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        CustodyWitnessRuntimeSnapshot {
            runtime_required: self.runtime_required,
            auto_renewal_enabled: self.auto_renewal_enabled,
            status: state.phase.as_str(),
            audit_interval_seconds: self.audit_interval_seconds,
            freshness_window_seconds: self.freshness_window_seconds,
            audits_total: state.audits_total,
            renewal_attempts_total: state.renewal_attempts_total,
            renewal_successes_total: state.renewal_successes_total,
            renewal_failures_total: state.renewal_failures_total,
            partial_collection_successes_total: state.partial_collection_successes_total,
            backoff_skips_total: state.backoff_skips_total,
            exhausted_skips_total: state.exhausted_skips_total,
            fail_closed_total: state.fail_closed_total,
            consecutive_failures: state.consecutive_failures,
            retry_after_seconds: state.retry_after_seconds,
            retry_before_expiry: state.retry_before_expiry,
            quorum_valid_through: state.quorum_valid_through,
            quorum_valid_for_seconds: state.quorum_valid_for_seconds,
            last_checkpoint_generation: state.last_checkpoint_generation,
            last_audit_at: state.last_audit_at,
            last_attempt_at: state.last_attempt_at,
            last_success_at: state.last_success_at,
            last_failure_at: state.last_failure_at,
            last_failure_reason: state.last_failure_reason,
            last_fail_closed_at: state.last_fail_closed_at,
            last_fail_closed_reason: state.last_fail_closed_reason,
        }
    }

    pub(super) fn record_audit(
        &self,
        checkpoint_generation: u64,
        evaluated_at: u64,
        renewal: CustodyWitnessRenewalStatus,
    ) {
        self.with_state(|state| {
            state.audits_total = state.audits_total.saturating_add(1);
            state.last_checkpoint_generation = Some(checkpoint_generation);
            state.last_audit_at = Some(evaluated_at);
            state.quorum_valid_through = Some(renewal.valid_through);
            state.quorum_valid_for_seconds = Some(renewal.valid_for_secs);
            if renewal.renewal_recommended {
                state.phase = CustodyWitnessRuntimePhase::RenewalDue;
            } else {
                state.phase = CustodyWitnessRuntimePhase::Healthy;
                state.consecutive_failures = 0;
                state.retry_after_seconds = None;
                state.retry_before_expiry = None;
            }
        });
    }

    pub(super) fn record_attempt(&self, attempted_at: u64) {
        self.with_state(|state| {
            state.phase = CustodyWitnessRuntimePhase::RenewalDue;
            state.renewal_attempts_total = state.renewal_attempts_total.saturating_add(1);
            state.last_attempt_at = Some(attempted_at);
        });
    }

    pub(super) fn record_backoff_skip(&self, retry_in_secs: u64, consecutive_failures: u32) {
        self.with_state(|state| {
            state.phase = CustodyWitnessRuntimePhase::BackingOff;
            state.backoff_skips_total = state.backoff_skips_total.saturating_add(1);
            state.consecutive_failures = consecutive_failures;
            state.retry_after_seconds = Some(retry_in_secs);
            state.retry_before_expiry = Some(true);
        });
    }

    pub(super) fn record_exhausted_skip(&self, consecutive_failures: u32) {
        self.with_state(|state| {
            state.phase = CustodyWitnessRuntimePhase::Exhausted;
            state.exhausted_skips_total = state.exhausted_skips_total.saturating_add(1);
            state.consecutive_failures = consecutive_failures;
            state.retry_after_seconds = None;
            state.retry_before_expiry = Some(false);
        });
    }

    pub(super) fn record_failure(
        &self,
        failed_at: u64,
        reason: &'static str,
        schedule: CustodyWitnessRenewalRetrySchedule,
    ) {
        self.with_state(|state| {
            state.phase = if schedule.retry_before_expiry {
                CustodyWitnessRuntimePhase::BackingOff
            } else {
                CustodyWitnessRuntimePhase::Exhausted
            };
            state.renewal_failures_total = state.renewal_failures_total.saturating_add(1);
            state.consecutive_failures = schedule.consecutive_failures;
            state.retry_after_seconds = schedule.retry_before_expiry.then_some(schedule.delay_secs);
            state.retry_before_expiry = Some(schedule.retry_before_expiry);
            state.last_failure_at = Some(failed_at);
            state.last_failure_reason = Some(reason);
        });
    }

    pub(super) fn record_success(
        &self,
        succeeded_at: u64,
        renewal: CustodyWitnessRenewalStatus,
        partial_collection: bool,
    ) {
        self.with_state(|state| {
            state.phase = CustodyWitnessRuntimePhase::Healthy;
            state.renewal_successes_total = state.renewal_successes_total.saturating_add(1);
            if partial_collection {
                state.partial_collection_successes_total =
                    state.partial_collection_successes_total.saturating_add(1);
            }
            state.consecutive_failures = 0;
            state.retry_after_seconds = None;
            state.retry_before_expiry = None;
            state.quorum_valid_through = Some(renewal.valid_through);
            state.quorum_valid_for_seconds = Some(renewal.valid_for_secs);
            state.last_success_at = Some(succeeded_at);
        });
    }

    pub(super) fn record_fail_closed(
        &self,
        failed_at: u64,
        reason: CustodyWitnessReadinessBlockReason,
    ) {
        self.with_state(|state| {
            state.phase = CustodyWitnessRuntimePhase::FailedClosed;
            state.fail_closed_total = state.fail_closed_total.saturating_add(1);
            state.last_fail_closed_at = Some(failed_at);
            state.last_fail_closed_reason = Some(reason.as_str());
        });
    }
}

impl CustodyWitnessRenewalRetryState {
    pub(super) fn action(
        &mut self,
        renewal: CustodyWitnessRenewalStatus,
        now: Instant,
    ) -> CustodyWitnessRenewalRetryAction {
        // [CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex] A new aggregate
        // quorum horizon is a new incident. It must not inherit cooldown from
        // an older receipt set, while repeated observations of one horizon do.
        if self.valid_through != Some(renewal.valid_through) {
            self.valid_through = Some(renewal.valid_through);
            self.consecutive_failures = 0;
            self.retry_not_before = None;
            self.exhausted_for_horizon = false;
        }
        if self.exhausted_for_horizon {
            return CustodyWitnessRenewalRetryAction::Exhausted {
                consecutive_failures: self.consecutive_failures,
            };
        }
        let Some(retry_not_before) = self.retry_not_before else {
            return CustodyWitnessRenewalRetryAction::Attempt;
        };
        if now >= retry_not_before {
            self.retry_not_before = None;
            return CustodyWitnessRenewalRetryAction::Attempt;
        }
        let remaining = retry_not_before.duration_since(now);
        let retry_in_secs = remaining
            .as_secs()
            .saturating_add(u64::from(remaining.subsec_nanos() != 0));
        CustodyWitnessRenewalRetryAction::BackingOff {
            retry_in_secs,
            consecutive_failures: self.consecutive_failures,
        }
    }

    pub(super) fn record_failure(
        &mut self,
        renewal: CustodyWitnessRenewalStatus,
        audit_interval_secs: u64,
        self_node_id: &[u8],
        now: Instant,
    ) -> CustodyWitnessRenewalRetrySchedule {
        if self.valid_through != Some(renewal.valid_through) {
            self.consecutive_failures = 0;
        }
        self.valid_through = Some(renewal.valid_through);
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        let (delay_secs, retry_before_expiry) = custody_witness_renewal_retry_delay_secs(
            renewal,
            audit_interval_secs,
            self_node_id,
            self.consecutive_failures,
        );
        self.exhausted_for_horizon = !retry_before_expiry;
        self.retry_not_before =
            retry_before_expiry.then_some(now + Duration::from_secs(delay_secs));
        CustodyWitnessRenewalRetrySchedule {
            delay_secs,
            consecutive_failures: self.consecutive_failures,
            retry_before_expiry,
        }
    }

    pub(super) fn record_success(&mut self, renewal: CustodyWitnessRenewalStatus) {
        self.valid_through = Some(renewal.valid_through);
        self.consecutive_failures = 0;
        self.retry_not_before = None;
        self.exhausted_for_horizon = false;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CustodyWitnessRenewalLogAction {
    Healthy,
    WarningEntered,
    WarningSuppressed,
    Recovered,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct CustodyWitnessRenewalLogState {
    pub(super) warned_valid_through: Option<u64>,
}

impl CustodyWitnessRenewalLogState {
    pub(super) fn observe(
        &mut self,
        renewal: CustodyWitnessRenewalStatus,
    ) -> CustodyWitnessRenewalLogAction {
        // [CUSTODY-RENEWAL-LIFECYCLE 2026-08-18 by Codex] The aggregate expiry
        // horizon is a stable incident key. Repeated timer observations for
        // that horizon stay debug-only, while a refreshed horizon may open a
        // new warning or close the prior one explicitly.
        if renewal.renewal_recommended {
            if self.warned_valid_through == Some(renewal.valid_through) {
                CustodyWitnessRenewalLogAction::WarningSuppressed
            } else {
                self.warned_valid_through = Some(renewal.valid_through);
                CustodyWitnessRenewalLogAction::WarningEntered
            }
        } else if self.warned_valid_through.take().is_some() {
            CustodyWitnessRenewalLogAction::Recovered
        } else {
            CustodyWitnessRenewalLogAction::Healthy
        }
    }
}

/// Derives a bounded cadence from the operator's receipt freshness policy.
///
/// [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] A quarter-window keeps
/// normal audits well ahead of expiry while fixed bounds prevent a hot loop or
/// an excessively slow security reaction. Config validation guarantees that
/// `max_age_secs` is at least 60 seconds.
pub(super) const fn custody_witness_runtime_audit_interval_secs(max_age_secs: u64) -> u64 {
    let quarter_window = max_age_secs / 4;
    if quarter_window < CUSTODY_WITNESS_RUNTIME_AUDIT_MIN_INTERVAL_SECS {
        CUSTODY_WITNESS_RUNTIME_AUDIT_MIN_INTERVAL_SECS
    } else if quarter_window > CUSTODY_WITNESS_RUNTIME_AUDIT_MAX_INTERVAL_SECS {
        CUSTODY_WITNESS_RUNTIME_AUDIT_MAX_INTERVAL_SECS
    } else {
        quarter_window
    }
}

pub(super) fn custody_witness_renewal_status(
    audit: &CustodyWitnessAuditEvidence,
    max_age_secs: u64,
) -> Option<CustodyWitnessRenewalStatus> {
    // [CUSTODY-QUORUM-EXPIRY 2026-08-18 by Codex] The storage snapshot derives
    // this from the threshold-th newest accepted receipt, never a witness id.
    let valid_through = audit.snapshot.policy.quorum_valid_through?;
    let valid_for_secs = audit
        .snapshot
        .policy
        .quorum_valid_for_secs(audit.evaluated_at)?;
    let warning_window_secs = custody_witness_renewal_warning_window_secs(max_age_secs);
    Some(CustodyWitnessRenewalStatus {
        valid_through,
        valid_for_secs,
        warning_window_secs,
        renewal_recommended: valid_for_secs <= warning_window_secs,
    })
}

pub(super) const fn custody_witness_auto_renewal_due(
    enabled: bool,
    renewal: CustodyWitnessRenewalStatus,
) -> bool {
    // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Keep the network
    // transition explicit and independently testable. A healthy quorum never
    // creates witness traffic, even when the operator enabled renewal.
    enabled && renewal.renewal_recommended
}

pub(super) fn custody_witness_renewal_retry_delay_secs(
    renewal: CustodyWitnessRenewalStatus,
    audit_interval_secs: u64,
    self_node_id: &[u8],
    consecutive_failures: u32,
) -> (u64, bool) {
    // [CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex] Retry only on audit-tick
    // boundaries: sub-period jitter would still wake every synchronized node
    // on the same next tick. The identity-derived +/- one-tick spread contains
    // fleet retry bursts without logging identity or weakening local audits.
    let audit_interval_secs = audit_interval_secs.max(1);
    let last_safe_retry_tick = renewal
        .valid_for_secs
        .saturating_sub(1)
        .checked_div(audit_interval_secs)
        .unwrap_or(0);
    let retry_before_expiry = last_safe_retry_tick > 0;
    let maximum_ticks = last_safe_retry_tick.max(1);
    let backoff_steps = consecutive_failures.clamp(1, 4);
    let nominal_ticks = 1u64 << backoff_steps;

    let mut mixed = renewal.valid_through
        ^ u64::from(consecutive_failures).rotate_left(17)
        ^ audit_interval_secs.rotate_left(7);
    for byte in self_node_id.iter().take(16) {
        mixed = mixed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(u64::from(*byte));
    }
    let jitter_ticks = match mixed % 3 {
        0 => -1i64,
        1 => 0i64,
        _ => 1i64,
    };
    let delayed_ticks = (i128::from(nominal_ticks) + i128::from(jitter_ticks))
        .clamp(1, i128::from(maximum_ticks)) as u64;
    (
        audit_interval_secs.saturating_mul(delayed_ticks),
        retry_before_expiry,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn audit_chat_relay_custody_witness_anchor(
    config: &ServerConfig,
    identity: &IdentityKeyPair,
    storage: &MemoryStorage,
    checkpoint_generation: u64,
    anchor_sha256: &[u8; 32],
    evaluated_at: u64,
) -> std::result::Result<CustodyWitnessAuditEvidence, CustodyWitnessReadinessBlockReason> {
    let witness_node_ids = config.discovery.custody_audit_witness_node_id_bytes();
    let snapshot = storage
        .audit_custody_audit_witness_receipt_readiness(
            &identity.public_key_bytes(),
            checkpoint_generation,
            anchor_sha256,
            &witness_node_ids,
            config.discovery.custody_audit_witness_min_verified,
            evaluated_at,
            config.discovery.custody_audit_witness_max_age_secs,
        )
        .await
        .map_err(|error| match error {
            CustodyAuditWitnessReadinessError::VaultInvalid => {
                CustodyWitnessReadinessBlockReason::ReceiptVaultInvalid
            }
            CustodyAuditWitnessReadinessError::PolicyInvalid => {
                CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid
            }
        })?;
    custody_witness_readiness_decision(snapshot.readiness)?;
    Ok(CustodyWitnessAuditEvidence {
        checkpoint_generation,
        evaluated_at,
        snapshot,
    })
}

pub(super) async fn audit_chat_relay_custody_witness_state(
    config: &ServerConfig,
    identity: &IdentityKeyPair,
    storage: &MemoryStorage,
) -> std::result::Result<CustodyWitnessAuditEvidence, CustodyWitnessReadinessBlockReason> {
    // [CUSTODY-WITNESS-RUNTIME-GUARD 2026-08-18 by Codex] The maintenance
    // guard binds the receipt policy to one immutable current custody anchor.
    // This helper performs no network I/O and exposes no anchor material.
    let anchor_guard = ChatRelayService::hold_backup_maintenance_audit_anchor_for_config(
        &config.memchain.chat_relay,
        identity,
    )
    .map_err(|_| CustodyWitnessReadinessBlockReason::CurrentAnchorUnavailable)?;
    let checkpoint_generation = anchor_guard.anchor().checkpoint_generation;
    let anchor_sha256 = custody_audit_anchor_frame_sha256(anchor_guard.anchor())
        .map_err(|_| CustodyWitnessReadinessBlockReason::CurrentAnchorUnavailable)?;
    let evaluated_at = unix_now_secs();
    audit_chat_relay_custody_witness_anchor(
        config,
        identity,
        storage,
        checkpoint_generation,
        &anchor_sha256,
        evaluated_at,
    )
    .await
}

pub(super) async fn renew_chat_relay_custody_witness_state(
    config: &ServerConfig,
    identity: &IdentityKeyPair,
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    client: &reqwest::Client,
) -> std::result::Result<CustodyWitnessRenewalAttempt, CustodyWitnessRenewalAttemptError> {
    // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Hold the same
    // cross-process maintenance guard used by the operator command across
    // network collection, durable receipt writes, and the final atomic audit.
    // A backup rotation therefore cannot move the anchor mid-renewal.
    let anchor_guard = ChatRelayService::hold_backup_maintenance_audit_anchor_for_config(
        &config.memchain.chat_relay,
        identity,
    )
    .map_err(|_| {
        CustodyWitnessRenewalAttemptError::Readiness(
            CustodyWitnessReadinessBlockReason::CurrentAnchorUnavailable,
        )
    })?;
    let checkpoint_generation = anchor_guard.anchor().checkpoint_generation;
    let anchor_sha256 = custody_audit_anchor_frame_sha256(anchor_guard.anchor()).map_err(|_| {
        CustodyWitnessRenewalAttemptError::Readiness(
            CustodyWitnessReadinessBlockReason::CurrentAnchorUnavailable,
        )
    })?;
    let witness_node_ids = config.discovery.custody_audit_witness_node_id_bytes();
    let round = witness_custody_audit_anchor_round_durable(
        storage,
        peer_store,
        identity,
        client,
        &witness_node_ids,
        config.discovery.custody_audit_witness_min_verified,
        anchor_guard.anchor(),
    )
    .await;
    // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Always audit after
    // the concurrent round, including a persistence error. Other completed
    // futures may already have durably retained adverse signed evidence; that
    // security result must outrank and immediately surface past transport.
    let audit = audit_chat_relay_custody_witness_anchor(
        config,
        identity,
        storage,
        checkpoint_generation,
        &anchor_sha256,
        unix_now_secs(),
    )
    .await
    .map_err(CustodyWitnessRenewalAttemptError::Readiness)?;
    let round = round.map_err(|_| CustodyWitnessRenewalAttemptError::CollectionFailed(audit))?;
    Ok(CustodyWitnessRenewalAttempt { round, audit })
}

impl Server {
    /// Enforces the current custody anchor against durable signed receipts.
    ///
    /// [CUSTODY-WITNESS-STARTUP-GATE 2026-08-18 by Codex] This runs before
    /// `PeerStore` bootstrap, listeners, self-advertisement, gossip, and runtime
    /// tasks. It performs no network I/O and keeps the `ChatRelay` maintenance
    /// guard held while the independent `MemChain` receipt vault is fully audited.
    pub(super) async fn verify_chat_relay_custody_witness_startup(
        &self,
        storage: &MemoryStorage,
    ) -> Result<()> {
        let fail = |reason: CustodyWitnessReadinessBlockReason| {
            warn!(
                reason = reason.as_str(),
                "[CHAT_RELAY] Custody witness startup guard rejected local state"
            );
            ServerError::startup_failed(format!(
                "Chat Relay custody witness startup guard: {reason}"
            ))
        };
        let audit = audit_chat_relay_custody_witness_state(&self.config, &self.identity, storage)
            .await
            .map_err(fail)?;
        let vault = audit.snapshot.vault;
        let evidence = audit.snapshot.policy;
        let renewal = custody_witness_renewal_status(
            &audit,
            self.config.discovery.custody_audit_witness_max_age_secs,
        )
        .ok_or_else(|| fail(CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid))?;
        if self.config.discovery.custody_audit_witness_runtime_required {
            // [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex] Strict runtime
            // mode has already completed a valid audit before listeners start.
            // Seed the process snapshot so heartbeat does not report a false
            // monitoring gap until the first periodic tick.
            self.custody_witness_runtime.record_audit(
                audit.checkpoint_generation,
                audit.evaluated_at,
                renewal,
            );
        }
        info!(
            checkpoint_generation = audit.checkpoint_generation,
            vault_records = vault.records,
            vault_accepted_records = vault.accepted_records,
            vault_adverse_records = vault.adverse_records,
            configured = evidence.configured,
            fresh_verified = evidence.fresh_verified,
            accepted = evidence.accepted,
            minimum_verified = evidence.minimum_verified,
            freshness_window_secs = self.config.discovery.custody_audit_witness_max_age_secs,
            quorum_valid_through = renewal.valid_through,
            quorum_valid_for_secs = renewal.valid_for_secs,
            renewal_recommended = renewal.renewal_recommended,
            "[CHAT_RELAY] Custody witness startup guard passed"
        );
        Ok(())
    }

    /// Reports one strict custody failure and preserves its typed reason.
    pub(super) async fn stop_for_custody_witness_runtime_failure(
        critical_failure_tx: &mpsc::Sender<CriticalRuntimeFailure>,
        shutdown_rx: &mut broadcast::Receiver<()>,
        telemetry: &CustodyWitnessRuntimeTelemetry,
        reason: CustodyWitnessReadinessBlockReason,
    ) {
        // [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] Runtime audit
        // and post-renewal audit share one failure edge so neither can race a
        // specific policy reason with the supervisor's generic task-exit path.
        telemetry.record_fail_closed(unix_now_secs(), reason);
        let failure = custody_witness_runtime_failure(reason);
        error!(
            reason = reason.as_str(),
            "[CHAT_RELAY] Runtime custody witness guard rejected local state"
        );
        if critical_failure_tx.send(failure).await.is_err() {
            error!(
                task = "custody-witness-runtime",
                "[RUNTIME] Main task dropped the critical failure receiver"
            );
            return;
        }
        let _ = shutdown_rx.recv().await;
    }

    /// Revalidates strict custody evidence throughout the process lifetime.
    ///
    /// [CUSTODY-WITNESS-AUTO-RENEWAL 2026-08-21 by Codex] When the independent
    /// opt-in flag is set, an approaching expiry runs one bounded exact-pin
    /// durable round before the final local audit. Existing configurations
    /// retain the v1.05 local-only behavior and create no witness traffic.
    pub(super) fn spawn_chat_relay_custody_witness_runtime_guard(
        &self,
        storage: Arc<MemoryStorage>,
        peer_store: Arc<PeerStore>,
        client: Arc<reqwest::Client>,
        critical_failure_tx: mpsc::Sender<CriticalRuntimeFailure>,
    ) -> JoinHandle<()> {
        let config = self.config.clone();
        let identity = self.identity.clone();
        let telemetry = Arc::clone(&self.custody_witness_runtime);
        let interval_secs = custody_witness_runtime_audit_interval_secs(
            config.discovery.custody_audit_witness_max_age_secs,
        );
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            let period = Duration::from_secs(interval_secs);
            let start = tokio::time::Instant::now() + period;
            let mut timer = tokio::time::interval_at(start, period);
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            info!(
                interval_secs,
                freshness_window_secs = config.discovery.custody_audit_witness_max_age_secs,
                auto_renewal_enabled = config.discovery.custody_audit_witness_auto_renewal_enabled,
                "[CHAT_RELAY] Runtime custody witness guard started"
            );
            let mut renewal_log_state = CustodyWitnessRenewalLogState::default();
            let mut renewal_retry_state = CustodyWitnessRenewalRetryState::default();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => return,
                    _ = timer.tick() => {
                        let mut audit = match audit_chat_relay_custody_witness_state(
                            &config,
                            &identity,
                            &storage,
                        )
                        .await
                        {
                            Ok(audit) => audit,
                            Err(reason) => {
                                Self::stop_for_custody_witness_runtime_failure(
                                    &critical_failure_tx,
                                    &mut shutdown_rx,
                                    &telemetry,
                                    reason,
                                )
                                .await;
                                return;
                            }
                        };
                        let Some(mut renewal) = custody_witness_renewal_status(
                            &audit,
                            config.discovery.custody_audit_witness_max_age_secs,
                        ) else {
                            Self::stop_for_custody_witness_runtime_failure(
                                &critical_failure_tx,
                                &mut shutdown_rx,
                                &telemetry,
                                CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid,
                            )
                            .await;
                            return;
                        };
                        telemetry.record_audit(
                            audit.checkpoint_generation,
                            audit.evaluated_at,
                            renewal,
                        );

                        if custody_witness_auto_renewal_due(
                            config.discovery.custody_audit_witness_auto_renewal_enabled,
                            renewal,
                        ) {
                            match renewal_retry_state.action(renewal, Instant::now()) {
                                CustodyWitnessRenewalRetryAction::BackingOff {
                                    retry_in_secs,
                                    consecutive_failures,
                                } => {
                                    telemetry.record_backoff_skip(
                                        retry_in_secs,
                                        consecutive_failures,
                                    );
                                    debug!(
                                        reason = "receipt_renewal_backoff",
                                        checkpoint_generation = audit.checkpoint_generation,
                                        quorum_valid_through = renewal.valid_through,
                                        quorum_valid_for_secs = renewal.valid_for_secs,
                                        retry_in_secs,
                                        consecutive_failures,
                                        "[CHAT_RELAY] Custody witness automatic renewal is cooling down"
                                    );
                                }
                                CustodyWitnessRenewalRetryAction::Exhausted {
                                    consecutive_failures,
                                } => {
                                    telemetry.record_exhausted_skip(consecutive_failures);
                                    debug!(
                                        reason = "receipt_renewal_retry_exhausted",
                                        checkpoint_generation = audit.checkpoint_generation,
                                        quorum_valid_through = renewal.valid_through,
                                        quorum_valid_for_secs = renewal.valid_for_secs,
                                        consecutive_failures,
                                        "[CHAT_RELAY] No custody witness retry tick remains before expiry"
                                    );
                                }
                                CustodyWitnessRenewalRetryAction::Attempt => {
                                    telemetry.record_attempt(unix_now_secs());
                                    match renew_chat_relay_custody_witness_state(
                                        &config,
                                        &identity,
                                        &storage,
                                        &peer_store,
                                        &client,
                                    )
                                    .await
                                    {
                                        Ok(attempt) => {
                                            audit = attempt.audit;
                                            let Some(refreshed) = custody_witness_renewal_status(
                                                &audit,
                                                config.discovery.custody_audit_witness_max_age_secs,
                                            ) else {
                                                Self::stop_for_custody_witness_runtime_failure(
                                                    &critical_failure_tx,
                                                    &mut shutdown_rx,
                                                    &telemetry,
                                                    CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid,
                                                )
                                                .await;
                                                return;
                                            };
                                            renewal = refreshed;
                                            // [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex]
                                            // The post-round durable audit is authoritative for
                                            // both the checkpoint generation and quorum horizon.
                                            telemetry.record_audit(
                                                audit.checkpoint_generation,
                                                audit.evaluated_at,
                                                renewal,
                                            );
                                            if renewal.renewal_recommended {
                                                let schedule = renewal_retry_state.record_failure(
                                                    renewal,
                                                    interval_secs,
                                                    &identity.public_key_bytes(),
                                                    Instant::now(),
                                                );
                                                telemetry.record_failure(
                                                    unix_now_secs(),
                                                    "quorum_not_refreshed",
                                                    schedule,
                                                );
                                                warn!(
                                                    reason = "receipt_renewal_quorum_not_refreshed",
                                                    checkpoint_generation = audit.checkpoint_generation,
                                                    configured = attempt.round.configured,
                                                    verified = attempt.round.verified,
                                                    accepted = attempt.round.accepted,
                                                    failed = attempt.round.failed,
                                                    adverse = attempt.round.adverse_evidence,
                                                    round_quorum_satisfied = attempt.round.quorum_satisfied,
                                                    quorum_valid_through = renewal.valid_through,
                                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                                    retry_after_secs = schedule.delay_secs,
                                                    retry_before_expiry = schedule.retry_before_expiry,
                                                    consecutive_failures = schedule.consecutive_failures,
                                                    "[CHAT_RELAY] Custody witness renewal round did not refresh the quorum"
                                                );
                                            } else {
                                                renewal_retry_state.record_success(renewal);
                                                telemetry.record_success(
                                                    unix_now_secs(),
                                                    renewal,
                                                    false,
                                                );
                                                info!(
                                                    checkpoint_generation = audit.checkpoint_generation,
                                                    configured = attempt.round.configured,
                                                    verified = attempt.round.verified,
                                                    accepted = attempt.round.accepted,
                                                    failed = attempt.round.failed,
                                                    adverse = attempt.round.adverse_evidence,
                                                    round_quorum_satisfied = attempt.round.quorum_satisfied,
                                                    quorum_valid_through = renewal.valid_through,
                                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                                    "[CHAT_RELAY] Custody witness automatic renewal refreshed the quorum"
                                                );
                                            }
                                        }
                                        Err(CustodyWitnessRenewalAttemptError::CollectionFailed(
                                            refreshed_audit,
                                        )) => {
                                            // [CUSTODY-RENEWAL-BACKOFF 2026-08-21 by Codex]
                                            // The durable audit is authoritative even when one
                                            // collection future failed after peers persisted data.
                                            audit = refreshed_audit;
                                            let Some(refreshed) = custody_witness_renewal_status(
                                                &audit,
                                                config.discovery.custody_audit_witness_max_age_secs,
                                            ) else {
                                                Self::stop_for_custody_witness_runtime_failure(
                                                    &critical_failure_tx,
                                                    &mut shutdown_rx,
                                                    &telemetry,
                                                    CustodyWitnessReadinessBlockReason::ReceiptPolicyInvalid,
                                                )
                                                .await;
                                                return;
                                            };
                                            renewal = refreshed;
                                            // [CUSTODY-RENEWAL-TELEMETRY 2026-08-21 by Codex]
                                            // A partially failed collection may still have
                                            // persisted enough receipts to advance the quorum.
                                            telemetry.record_audit(
                                                audit.checkpoint_generation,
                                                audit.evaluated_at,
                                                renewal,
                                            );
                                            if renewal.renewal_recommended {
                                                let schedule = renewal_retry_state.record_failure(
                                                    renewal,
                                                    interval_secs,
                                                    &identity.public_key_bytes(),
                                                    Instant::now(),
                                                );
                                                telemetry.record_failure(
                                                    unix_now_secs(),
                                                    "collection_failed",
                                                    schedule,
                                                );
                                                warn!(
                                                    reason = "receipt_renewal_collection_failed",
                                                    checkpoint_generation = audit.checkpoint_generation,
                                                    quorum_valid_through = renewal.valid_through,
                                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                                    retry_after_secs = schedule.delay_secs,
                                                    retry_before_expiry = schedule.retry_before_expiry,
                                                    consecutive_failures = schedule.consecutive_failures,
                                                    "[CHAT_RELAY] Custody witness automatic renewal entered bounded backoff"
                                                );
                                            } else {
                                                renewal_retry_state.record_success(renewal);
                                                telemetry.record_success(
                                                    unix_now_secs(),
                                                    renewal,
                                                    true,
                                                );
                                                warn!(
                                                    reason = "receipt_renewal_collection_partial",
                                                    checkpoint_generation = audit.checkpoint_generation,
                                                    quorum_valid_through = renewal.valid_through,
                                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                                    "[CHAT_RELAY] Custody witness quorum refreshed despite a partial collection failure"
                                                );
                                            }
                                        }
                                        Err(CustodyWitnessRenewalAttemptError::Readiness(reason)) => {
                                            Self::stop_for_custody_witness_runtime_failure(
                                                &critical_failure_tx,
                                                &mut shutdown_rx,
                                                &telemetry,
                                                reason,
                                            )
                                            .await;
                                            return;
                                        }
                                    }
                                }
                            }
                        }

                        let vault = audit.snapshot.vault;
                        let evidence = audit.snapshot.policy;
                        match renewal_log_state.observe(renewal) {
                            CustodyWitnessRenewalLogAction::WarningEntered => {
                                warn!(
                                    reason = "receipt_renewal_required",
                                    checkpoint_generation = audit.checkpoint_generation,
                                    quorum_valid_through = renewal.valid_through,
                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                    warning_window_secs = renewal.warning_window_secs,
                                    fresh_verified = evidence.fresh_verified,
                                    accepted = evidence.accepted,
                                    minimum_verified = evidence.minimum_verified,
                                    "[CHAT_RELAY] Custody witness evidence is approaching expiry"
                                );
                            }
                            CustodyWitnessRenewalLogAction::Recovered => {
                                info!(
                                    reason = "receipt_renewal_recovered",
                                    checkpoint_generation = audit.checkpoint_generation,
                                    quorum_valid_through = renewal.valid_through,
                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                    fresh_verified = evidence.fresh_verified,
                                    accepted = evidence.accepted,
                                    minimum_verified = evidence.minimum_verified,
                                    "[CHAT_RELAY] Custody witness evidence freshness recovered"
                                );
                            }
                            CustodyWitnessRenewalLogAction::WarningSuppressed => {
                                debug!(
                                    reason = "receipt_renewal_pending",
                                    checkpoint_generation = audit.checkpoint_generation,
                                    quorum_valid_through = renewal.valid_through,
                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                    "[CHAT_RELAY] Custody witness renewal warning already active"
                                );
                            }
                            CustodyWitnessRenewalLogAction::Healthy => {
                                debug!(
                                    checkpoint_generation = audit.checkpoint_generation,
                                    vault_records = vault.records,
                                    fresh_verified = evidence.fresh_verified,
                                    accepted = evidence.accepted,
                                    minimum_verified = evidence.minimum_verified,
                                    quorum_valid_through = renewal.valid_through,
                                    quorum_valid_for_secs = renewal.valid_for_secs,
                                    "[CHAT_RELAY] Runtime custody witness guard passed"
                                );
                            }
                        }
                    }
                }
            }
        })
    }
}
