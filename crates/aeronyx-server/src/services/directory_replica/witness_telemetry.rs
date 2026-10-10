// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/witness_telemetry.rs
// ============================================
//! # Witness recovery and carrier telemetry
//!
//! Owns the process-only, aggregate witness availability-recovery and
//! carrier-service outcome, health, and snapshot values.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

/// Terminal result of one bounded witness availability-recovery operation.
///
/// [WITNESS-TERMINAL-STATE 2026-07-29 by Codex] Event order is represented by
/// the last assigned enum, not inferred from second-granularity timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessRecoveryOutcome {
    /// An exact carrier envelope was verified before inner witness validation.
    Recovered,
    /// Every selected availability route was exhausted.
    Exhausted,
    /// A canonical, signature, admission, or target contract check stopped closed.
    FailedClosed,
}

impl DirectoryObservationWitnessRecoveryOutcome {
    /// Stable public outcome label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recovered => "recovered",
            Self::Exhausted => "exhausted",
            Self::FailedClosed => "failed_closed",
        }
    }
}

/// Current process-local witness availability-recovery health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessRecoveryHealth {
    /// No recovery operation has reached a terminal result.
    Standby,
    /// The latest terminal recovery result verified a carrier envelope.
    Recovered,
    /// The latest terminal recovery result exhausted every bounded route.
    Exhausted,
    /// The latest terminal recovery result stopped closed.
    FailedClosed,
    /// Mutually exclusive attempt counters no longer sum to attempts.
    Inconsistent,
}

impl DirectoryObservationWitnessRecoveryHealth {
    /// Stable public status label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standby => "standby",
            Self::Recovered => "recovered",
            Self::Exhausted => "exhausted",
            Self::FailedClosed => "failed_closed",
            Self::Inconsistent => "inconsistent",
        }
    }
}

/// Process-lifetime aggregate checkpoint-witness carrier telemetry.
///
/// [WITNESS-CARRIER 2026-07-26 by Codex] These counters describe bounded
/// availability recovery only. They intentionally omit observer, witness,
/// carrier, endpoint, route, descriptor-sequence, checkpoint-hash, and frame
/// data and are never interpreted as authority or reputation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessRecoverySnapshot {
    /// Bounded carrier selections after a direct-path availability failure.
    pub selections: u64,
    /// Candidates visible to the latest selection.
    pub latest_candidates: u64,
    /// Latest candidates with fresh local routeability evidence.
    pub latest_routeable_candidates: u64,
    /// Latest candidates skipped by descriptor-sequence capability memory.
    pub latest_capability_cached_unavailable: u64,
    /// Capacity-bounded carriers selected by the latest recovery.
    pub latest_selected: u64,
    /// Total carrier HTTP attempts during this process lifetime.
    pub attempts: u64,
    /// Carrier envelopes successfully verified before inner witness validation.
    pub succeeded: u64,
    /// Explicit optional carrier- or target-route absence outcomes.
    pub capability_unavailable: u64,
    /// Availability-only carrier or target transport failures.
    pub transport_failures: u64,
    /// Recoveries exhausted without a verified carrier envelope.
    pub exhausted: u64,
    /// Canonical, signature, admission, or target-contract failures that stopped closed.
    pub failed_closed: u64,
    /// Latest selection or carrier attempt timestamp.
    pub last_attempt_at: Option<u64>,
    /// Latest verified carrier-envelope timestamp.
    pub last_success_at: Option<u64>,
    /// Latest exhausted or fail-closed recovery timestamp.
    pub last_failure_at: Option<u64>,
    /// Latest terminal recovery result in exact process event order.
    pub last_outcome: Option<DirectoryObservationWitnessRecoveryOutcome>,
}

impl DirectoryObservationWitnessRecoverySnapshot {
    /// Returns the sum of all mutually exclusive carrier-attempt buckets.
    #[must_use]
    pub const fn attempt_outcomes(&self) -> u64 {
        self.succeeded
            .saturating_add(self.capability_unavailable)
            .saturating_add(self.transport_failures)
            .saturating_add(self.failed_closed)
    }

    /// Verifies that each recorded carrier attempt entered exactly one bucket.
    #[must_use]
    pub const fn attempt_outcomes_consistent(&self) -> bool {
        self.attempt_outcomes() == self.attempts
    }

    /// Classifies current recovery health from exact terminal event order.
    #[must_use]
    pub const fn health(&self) -> DirectoryObservationWitnessRecoveryHealth {
        if !self.attempt_outcomes_consistent() {
            DirectoryObservationWitnessRecoveryHealth::Inconsistent
        } else {
            match self.last_outcome {
                None => DirectoryObservationWitnessRecoveryHealth::Standby,
                Some(DirectoryObservationWitnessRecoveryOutcome::Recovered) => {
                    DirectoryObservationWitnessRecoveryHealth::Recovered
                }
                Some(DirectoryObservationWitnessRecoveryOutcome::Exhausted) => {
                    DirectoryObservationWitnessRecoveryHealth::Exhausted
                }
                Some(DirectoryObservationWitnessRecoveryOutcome::FailedClosed) => {
                    DirectoryObservationWitnessRecoveryHealth::FailedClosed
                }
            }
        }
    }
}

/// Process-lifetime aggregate telemetry for this node acting as a witness carrier.
///
/// [WITNESS-CARRIER-SERVICE 2026-07-27 by Codex] One authenticated request is
/// reduced to exactly one terminal outcome before entering this snapshot.
/// Requester, witness, endpoint, route, descriptor, checkpoint, frame, digest,
/// signature, and user-plane data are deliberately not accepted by this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessCarrierSnapshot {
    /// Authenticated pinned-requester requests completed by this process.
    pub requests: u64,
    /// Requests that returned an independently verified target-witness frame.
    pub forwarded: u64,
    /// Authenticated requests rejected because the target was not operator-pinned.
    pub policy_rejected: u64,
    /// Authenticated requests whose inner frame failed canonical or signature checks.
    pub invalid_requests: u64,
    /// Requests whose pinned target descriptor, endpoint, or transport was unavailable.
    pub target_unavailable: u64,
    /// Requests whose target explicitly lacked the witness route.
    pub target_capability_unavailable: u64,
    /// Requests explicitly rejected by the target witness.
    pub target_rejected: u64,
    /// Requests whose successful target response failed bounds or verification.
    pub target_invalid_response: u64,
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Requests skipped while
    /// the current target descriptor remained in process-only cooldown.
    pub target_cooling_down: u64,
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Requests rejected
    /// immediately because all bounded local carrier slots were busy.
    pub local_overloaded: u64,
    /// Requests that could not create the bounded local transport client.
    pub local_failures: u64,
    /// Latest authenticated carrier request completion timestamp.
    pub last_request_at: Option<u64>,
    /// Latest independently verified target response timestamp.
    pub last_forwarded_at: Option<u64>,
    /// Latest non-success carrier request timestamp.
    pub last_failure_at: Option<u64>,
    /// Latest request result in exact process event order.
    pub last_outcome: Option<DirectoryObservationWitnessCarrierOutcome>,
}

/// Stable mutually exclusive carrier-side request outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessCarrierOutcome {
    /// Exact target-witness response verified and returned.
    Forwarded,
    /// Target identity was outside the carrier's operator pins.
    PolicyRejected,
    /// Inner observer request failed canonical or signature verification.
    InvalidRequest,
    /// Target descriptor, endpoint, body stream, or transport was unavailable.
    TargetUnavailable,
    /// Target explicitly did not implement the witness route.
    TargetCapabilityUnavailable,
    /// Target rejected the otherwise valid forwarded request.
    TargetRejected,
    /// Target returned an oversized, malformed, or wrongly signed success body.
    TargetInvalidResponse,
    /// [WITNESS-CARRIER-ADMISSION 2026-07-27 by Codex] Current target
    /// descriptor remained inside a process-only failure cooldown.
    TargetCoolingDown,
    /// Every bounded local outbound carrier slot was already occupied.
    LocalOverloaded,
    /// Carrier could not initialize its bounded no-proxy transport client.
    LocalFailure,
}

impl DirectoryObservationWitnessCarrierOutcome {
    /// Stable public outcome label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Forwarded => "forwarded",
            Self::PolicyRejected => "policy_rejected",
            Self::InvalidRequest => "invalid_request",
            Self::TargetUnavailable => "target_unavailable",
            Self::TargetCapabilityUnavailable => "target_capability_unavailable",
            Self::TargetRejected => "target_rejected",
            Self::TargetInvalidResponse => "target_invalid_response",
            Self::TargetCoolingDown => "target_cooling_down",
            Self::LocalOverloaded => "local_overloaded",
            Self::LocalFailure => "local_failure",
        }
    }
}

/// Current process-local health of this node acting as a witness carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessCarrierHealth {
    /// No authenticated carrier request has completed.
    Standby,
    /// The latest authenticated request forwarded an exact verified frame.
    Active,
    /// The latest authenticated request did not forward a verified frame.
    Degraded,
    /// Mutually exclusive request counters no longer sum to requests.
    Inconsistent,
}

impl DirectoryObservationWitnessCarrierHealth {
    /// Stable public status label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standby => "standby",
            Self::Active => "active",
            Self::Degraded => "degraded",
            Self::Inconsistent => "inconsistent",
        }
    }
}

impl DirectoryObservationWitnessCarrierSnapshot {
    /// Returns the sum of all mutually exclusive request outcome buckets.
    #[must_use]
    pub const fn terminal_outcomes(&self) -> u64 {
        self.forwarded
            .saturating_add(self.policy_rejected)
            .saturating_add(self.invalid_requests)
            .saturating_add(self.target_unavailable)
            .saturating_add(self.target_capability_unavailable)
            .saturating_add(self.target_rejected)
            .saturating_add(self.target_invalid_response)
            .saturating_add(self.target_cooling_down)
            .saturating_add(self.local_overloaded)
            .saturating_add(self.local_failures)
    }

    /// Verifies that each completed request entered exactly one outcome bucket.
    #[must_use]
    pub const fn terminal_outcomes_consistent(&self) -> bool {
        self.terminal_outcomes() == self.requests
    }

    /// Classifies carrier health from exact terminal request order.
    #[must_use]
    pub const fn health(&self) -> DirectoryObservationWitnessCarrierHealth {
        if !self.terminal_outcomes_consistent() {
            DirectoryObservationWitnessCarrierHealth::Inconsistent
        } else {
            match self.last_outcome {
                None => DirectoryObservationWitnessCarrierHealth::Standby,
                Some(DirectoryObservationWitnessCarrierOutcome::Forwarded) => {
                    DirectoryObservationWitnessCarrierHealth::Active
                }
                Some(_) => DirectoryObservationWitnessCarrierHealth::Degraded,
            }
        }
    }
}
