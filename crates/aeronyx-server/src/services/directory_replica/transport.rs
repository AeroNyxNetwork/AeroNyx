// ============================================================================
// File: crates/aeronyx-server/src/services/directory_replica/transport.rs
// ============================================================================
//! Process-only Directory synchronization transport health.
//!
//! [DIRECTORY-REPLICA-TRANSPORT-MODULE 2026-09-23 by Codex] This module owns
//! only bounded aggregate outcome classes and lifecycle state. It must never
//! accept peer, endpoint, request, payload, signature, or durable chain data.

use std::collections::VecDeque;

/// Stable mutually exclusive outcomes for one completed Directory coordinator
/// HTTP exchange.
///
/// [DIRECTORY-TRANSPORT-TELEMETRY 2026-07-28 by Codex] This enum deliberately
/// omits operation, peer, endpoint, producer, carrier, URL, status code,
/// request id, frame, and response data. It is diagnostic evidence only and
/// must never influence authority, reputation, routing, or chain selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryReplicaTransportOutcome {
    /// A bounded success body was read completely.
    Succeeded,
    /// Connection establishment exceeded its configured deadline.
    ConnectTimeout,
    /// The request exceeded its deadline after connection classification.
    RequestTimeout,
    /// Connection establishment failed without a timeout.
    ConnectFailure,
    /// Another request-layer failure occurred before a response was available.
    RequestFailure,
    /// The peer returned a non-success HTTP status.
    HttpStatusFailure,
    /// The declared or streamed success body exceeded its protocol ceiling.
    ResponseTooLarge,
    /// The bounded success body stream failed before completion.
    ResponseBodyReadFailure,
}

/// Number of terminal Directory transport outcomes retained for recent health.
///
/// [DIRECTORY-TRANSPORT-WINDOW 2026-07-28 by Codex] This bounds both memory
/// and the amount of process history represented by the health classification.
/// Entries are outcome classes only; no request metadata is retained.
pub const DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY: usize = 32;
const DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY_U64: u64 = 32;
/// Recent failure percentage that marks Directory synchronization degraded.
pub const DIRECTORY_REPLICA_TRANSPORT_DEGRADED_FAILURE_PERCENT: u64 = 20;
/// Consecutive recent failures that mark Directory synchronization degraded.
pub const DIRECTORY_REPLICA_TRANSPORT_DEGRADED_CONSECUTIVE_FAILURES: u64 = 3;

/// Current process-local health of the Directory synchronization transport.
///
/// [DIRECTORY-TRANSPORT-LIFECYCLE 2026-07-29 by Codex] This classification is
/// diagnostic only. It cannot select peers, change authority, mutate chain
/// evidence, or create/resolve durable security incidents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryReplicaTransportHealth {
    /// No completed coordinator-owned request has been observed.
    Idle,
    /// Recent aggregate evidence remains below both degradation thresholds.
    Healthy,
    /// Recent failure ratio or trailing failure count reached its threshold.
    Degraded,
    /// Lifetime, recent-window, or transition invariants do not agree.
    Inconsistent,
}

impl DirectoryReplicaTransportHealth {
    /// Stable public status label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Inconsistent => "inconsistent",
        }
    }
}

/// Process-lifetime aggregate Directory synchronization transport telemetry.
///
/// Every completed coordinator-owned request contributes to `requests` and
/// exactly one terminal bucket. The snapshot never accepts identity-bearing or
/// request-bearing values and is intentionally not persisted across restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaTransportSnapshot {
    /// Completed coordinator-owned HTTP requests.
    pub requests: u64,
    /// Requests with a completely read bounded success body.
    pub succeeded: u64,
    /// Connection-establishment timeouts.
    pub connect_timeouts: u64,
    /// Non-connect request timeouts.
    pub request_timeouts: u64,
    /// Non-timeout connection failures.
    pub connect_failures: u64,
    /// Other request-layer failures.
    pub request_failures: u64,
    /// Non-success HTTP responses.
    pub http_status_failures: u64,
    /// Oversized declared or streamed success bodies.
    pub response_too_large: u64,
    /// Interrupted bounded success-body streams.
    pub response_body_read_failures: u64,
    /// Maximum number of outcome classes represented by the recent window.
    pub recent_window_capacity: u64,
    /// Terminal outcomes currently represented by the recent window.
    pub recent_requests: u64,
    /// Successful outcomes currently represented by the recent window.
    pub recent_succeeded: u64,
    /// Non-success outcomes currently represented by the recent window.
    pub recent_failures: u64,
    /// Trailing non-success outcomes, capped by the recent window capacity.
    pub consecutive_failures: u64,
    /// Process-lifetime transitions from non-degraded into degraded health.
    pub degraded_transitions: u64,
    /// Process-lifetime transitions from degraded back into healthy health.
    pub recovery_transitions: u64,
    /// Monotonic wall-clock timestamp at which the current degradation began.
    pub degraded_since_at: Option<u64>,
    /// Monotonic wall-clock timestamp of the latest degraded transition.
    pub last_degraded_at: Option<u64>,
    /// Monotonic wall-clock timestamp of the latest recovery transition.
    pub last_recovered_at: Option<u64>,
    /// Latest completed terminal outcome.
    pub last_outcome: Option<DirectoryReplicaTransportOutcome>,
    /// Latest completed request timestamp.
    pub last_request_at: Option<u64>,
    /// Latest successful request timestamp.
    pub last_success_at: Option<u64>,
    /// Latest non-success request timestamp.
    pub last_failure_at: Option<u64>,
}

impl Default for DirectoryReplicaTransportSnapshot {
    fn default() -> Self {
        Self {
            requests: 0,
            succeeded: 0,
            connect_timeouts: 0,
            request_timeouts: 0,
            connect_failures: 0,
            request_failures: 0,
            http_status_failures: 0,
            response_too_large: 0,
            response_body_read_failures: 0,
            recent_window_capacity: DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY_U64,
            recent_requests: 0,
            recent_succeeded: 0,
            recent_failures: 0,
            consecutive_failures: 0,
            degraded_transitions: 0,
            recovery_transitions: 0,
            degraded_since_at: None,
            last_degraded_at: None,
            last_recovered_at: None,
            last_outcome: None,
            last_request_at: None,
            last_success_at: None,
            last_failure_at: None,
        }
    }
}

impl DirectoryReplicaTransportSnapshot {
    /// Returns the sum of all mutually exclusive terminal buckets.
    #[must_use]
    pub const fn terminal_outcomes(self) -> u64 {
        self.succeeded
            .saturating_add(self.connect_timeouts)
            .saturating_add(self.request_timeouts)
            .saturating_add(self.connect_failures)
            .saturating_add(self.request_failures)
            .saturating_add(self.http_status_failures)
            .saturating_add(self.response_too_large)
            .saturating_add(self.response_body_read_failures)
    }

    /// Returns the failure percentage represented by the bounded recent window.
    #[must_use]
    pub const fn recent_failure_percent(&self) -> u64 {
        if self.recent_requests == 0 {
            0
        } else {
            self.recent_failures.saturating_mul(100) / self.recent_requests
        }
    }

    /// Verifies the mutually exclusive process-lifetime outcome counters.
    #[must_use]
    pub const fn terminal_outcomes_consistent(&self) -> bool {
        self.terminal_outcomes() == self.requests
    }

    /// Verifies bounded recent-window counters independently from lifetime data.
    #[must_use]
    pub const fn recent_outcomes_consistent(&self) -> bool {
        self.recent_succeeded.saturating_add(self.recent_failures) == self.recent_requests
            && self.recent_window_capacity > 0
            && self.recent_requests <= self.recent_window_capacity
            && self.recent_requests <= self.requests
            && self.consecutive_failures <= self.recent_failures
            && self.consecutive_failures <= self.recent_window_capacity
            && (self.requests == 0) == (self.recent_requests == 0)
    }

    /// Verifies aggregate degradation/recovery transition invariants.
    #[must_use]
    pub const fn lifecycle_consistent(&self) -> bool {
        let currently_degraded = self.recent_window_degraded();
        let open_degradations = self
            .degraded_transitions
            .saturating_sub(self.recovery_transitions);
        let transition_counts_consistent = self.degraded_transitions >= self.recovery_transitions
            && open_degradations <= 1
            && (open_degradations == 1) == currently_degraded;
        let transition_presence_consistent = self.degraded_since_at.is_some() == currently_degraded
            && (self.degraded_transitions == 0) == self.last_degraded_at.is_none()
            && (self.recovery_transitions == 0) == self.last_recovered_at.is_none();
        let transition_order_consistent = match (self.last_degraded_at, self.last_recovered_at) {
            (Some(degraded), Some(recovered)) if currently_degraded => degraded >= recovered,
            (Some(degraded), Some(recovered)) => recovered >= degraded,
            (Some(_), None) => currently_degraded,
            (None, None) => !currently_degraded,
            (None, Some(_)) => false,
        };
        transition_counts_consistent
            && transition_presence_consistent
            && transition_order_consistent
    }

    /// Classifies current transport health from one canonical policy.
    #[must_use]
    pub const fn health(&self) -> DirectoryReplicaTransportHealth {
        if !self.terminal_outcomes_consistent()
            || !self.recent_outcomes_consistent()
            || !self.lifecycle_consistent()
        {
            DirectoryReplicaTransportHealth::Inconsistent
        } else if self.requests == 0 {
            DirectoryReplicaTransportHealth::Idle
        } else if self.recent_window_degraded() {
            DirectoryReplicaTransportHealth::Degraded
        } else {
            DirectoryReplicaTransportHealth::Healthy
        }
    }

    const fn recent_window_degraded(&self) -> bool {
        self.recent_failure_percent() >= DIRECTORY_REPLICA_TRANSPORT_DEGRADED_FAILURE_PERCENT
            || self.consecutive_failures
                >= DIRECTORY_REPLICA_TRANSPORT_DEGRADED_CONSECUTIVE_FAILURES
    }
}

/// Fixed-memory transport runtime that owns lifetime and recent-window state.
///
/// The queue stores only terminal outcome classes. Keeping it private prevents
/// future status code from accidentally serializing request-level history.
#[derive(Debug)]
pub(super) struct DirectoryReplicaTransportRuntime {
    snapshot: DirectoryReplicaTransportSnapshot,
    recent_outcomes: VecDeque<DirectoryReplicaTransportOutcome>,
}

impl Default for DirectoryReplicaTransportRuntime {
    fn default() -> Self {
        Self {
            snapshot: DirectoryReplicaTransportSnapshot::default(),
            recent_outcomes: VecDeque::with_capacity(DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY),
        }
    }
}

impl DirectoryReplicaTransportRuntime {
    pub(super) fn record(&mut self, outcome: DirectoryReplicaTransportOutcome, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        if self.recent_outcomes.len() == DIRECTORY_REPLICA_TRANSPORT_WINDOW_CAPACITY {
            self.recent_outcomes.pop_front();
        }
        self.recent_outcomes.push_back(outcome);
        self.snapshot.requests = self.snapshot.requests.saturating_add(1);
        self.snapshot.last_outcome = Some(outcome);
        let transition_at = latest_runtime_timestamp(self.snapshot.last_request_at, completed_at);
        self.snapshot.last_request_at = Some(transition_at);
        match outcome {
            DirectoryReplicaTransportOutcome::Succeeded => {
                self.snapshot.succeeded = self.snapshot.succeeded.saturating_add(1);
                self.snapshot.last_success_at = Some(latest_runtime_timestamp(
                    self.snapshot.last_success_at,
                    completed_at,
                ));
            }
            DirectoryReplicaTransportOutcome::ConnectTimeout => {
                self.snapshot.connect_timeouts = self.snapshot.connect_timeouts.saturating_add(1);
            }
            DirectoryReplicaTransportOutcome::RequestTimeout => {
                self.snapshot.request_timeouts = self.snapshot.request_timeouts.saturating_add(1);
            }
            DirectoryReplicaTransportOutcome::ConnectFailure => {
                self.snapshot.connect_failures = self.snapshot.connect_failures.saturating_add(1);
            }
            DirectoryReplicaTransportOutcome::RequestFailure => {
                self.snapshot.request_failures = self.snapshot.request_failures.saturating_add(1);
            }
            DirectoryReplicaTransportOutcome::HttpStatusFailure => {
                self.snapshot.http_status_failures =
                    self.snapshot.http_status_failures.saturating_add(1);
            }
            DirectoryReplicaTransportOutcome::ResponseTooLarge => {
                self.snapshot.response_too_large =
                    self.snapshot.response_too_large.saturating_add(1);
            }
            DirectoryReplicaTransportOutcome::ResponseBodyReadFailure => {
                self.snapshot.response_body_read_failures =
                    self.snapshot.response_body_read_failures.saturating_add(1);
            }
        }
        if outcome != DirectoryReplicaTransportOutcome::Succeeded {
            self.snapshot.last_failure_at = Some(latest_runtime_timestamp(
                self.snapshot.last_failure_at,
                completed_at,
            ));
        }
        self.refresh_recent_window();
        self.update_lifecycle(transition_at);
    }

    fn refresh_recent_window(&mut self) {
        self.snapshot.recent_requests = u64::try_from(self.recent_outcomes.len())
            .unwrap_or(self.snapshot.recent_window_capacity);
        self.snapshot.recent_succeeded = u64::try_from(
            self.recent_outcomes
                .iter()
                .filter(|outcome| **outcome == DirectoryReplicaTransportOutcome::Succeeded)
                .count(),
        )
        .unwrap_or(self.snapshot.recent_window_capacity);
        self.snapshot.recent_failures = self
            .snapshot
            .recent_requests
            .saturating_sub(self.snapshot.recent_succeeded);
        self.snapshot.consecutive_failures = u64::try_from(
            self.recent_outcomes
                .iter()
                .rev()
                .take_while(|outcome| **outcome != DirectoryReplicaTransportOutcome::Succeeded)
                .count(),
        )
        .unwrap_or(self.snapshot.recent_window_capacity);
    }

    fn update_lifecycle(&mut self, transition_at: u64) {
        let currently_degraded = self.snapshot.degraded_since_at.is_some();
        let next_degraded = self.snapshot.recent_window_degraded();
        match (currently_degraded, next_degraded) {
            (false, true) => {
                self.snapshot.degraded_transitions =
                    self.snapshot.degraded_transitions.saturating_add(1);
                self.snapshot.degraded_since_at = Some(transition_at);
                self.snapshot.last_degraded_at = Some(latest_runtime_timestamp(
                    self.snapshot.last_degraded_at,
                    transition_at,
                ));
            }
            (true, false) => {
                self.snapshot.recovery_transitions =
                    self.snapshot.recovery_transitions.saturating_add(1);
                self.snapshot.degraded_since_at = None;
                self.snapshot.last_recovered_at = Some(latest_runtime_timestamp(
                    self.snapshot.last_recovered_at,
                    transition_at,
                ));
            }
            (false, false) | (true, true) => {}
        }
    }

    pub(super) const fn snapshot(&self) -> DirectoryReplicaTransportSnapshot {
        self.snapshot
    }
}

pub(super) fn latest_runtime_timestamp(current: Option<u64>, candidate: u64) -> u64 {
    current.map_or(candidate, |timestamp| timestamp.max(candidate))
}
