// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/sync_runtime.rs
// ============================================
//! # Synchronization runtime telemetry
//!
//! Owns producer sync observations, Full-node Mirror runtime snapshots, and the
//! shared process-lifetime `DirectoryReplicaSyncRuntime`.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{
    latest_runtime_timestamp, Arc, DirectoryObservationWitnessCarrierOutcome,
    DirectoryObservationWitnessCarrierSnapshot, DirectoryObservationWitnessOutcome,
    DirectoryObservationWitnessOutcomeCounters, DirectoryObservationWitnessOutcomeSnapshot,
    DirectoryObservationWitnessRecoveryOutcome, DirectoryObservationWitnessRecoverySnapshot,
    DirectoryReplicaRetryState, DirectoryReplicaTransportOutcome, DirectoryReplicaTransportRuntime,
    DirectoryReplicaTransportSnapshot, HashMap, Mutex, OnceLock, Semaphore,
    DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES, MAX_DIRECTORY_AUDITS_IN_FLIGHT,
};

/// Runtime-only synchronization observation for one pinned producer.
///
/// These fields intentionally contain no endpoint, full response, descriptor,
/// route, payload, client, or wallet metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaSyncObservation {
    /// Pinned producer identity used only for internal status correlation.
    pub producer: [u8; 32],
    /// Most recent bounded pull attempt.
    pub last_attempt_at: Option<u64>,
    /// Most recent authenticated successful page.
    pub last_success_at: Option<u64>,
    /// Most recent rejected or failed page.
    pub last_failure_at: Option<u64>,
    /// Stable privacy-safe reason code from the most recent failure.
    pub last_failure_reason: Option<String>,
    /// Earliest Unix timestamp at which this producer may be attempted again.
    pub retry_not_before: Option<u64>,
    /// Signed remote tip height most recently observed.
    pub remote_tip_height: Option<u64>,
    /// Accepted local replica height after the most recent success.
    pub local_tip_height: u64,
    /// Whether the most recent signed response indicated additional pages.
    pub has_more: bool,
    /// Consecutive failed attempts since the last successful page.
    pub consecutive_failures: u64,
    /// Total bounded attempts during this process lifetime.
    pub total_attempts: u64,
    /// Total authenticated pages accepted during this process lifetime.
    pub successful_pages: u64,
    /// Total failed attempts during this process lifetime.
    pub failed_attempts: u64,
    /// Total scheduled rounds skipped while this producer was in backoff.
    pub backoff_skips: u64,
    /// Total new blocks committed during this process lifetime.
    pub blocks_inserted: u64,
    /// Total new commitments committed during this process lifetime.
    pub commitments_inserted: u64,
    /// Total HTTP requests consumed by authenticated successful pages.
    pub requests_sent: u64,
}

/// Aggregate process-lifetime Full-node Mirror scheduling telemetry.
///
/// This intentionally omits identities, endpoints, descriptor hashes, routes,
/// and response details. Mirror observations are diagnostic transport evidence,
/// never authority, consensus, fork choice, voting, or finality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryFullNodeMirrorRuntimeSnapshot {
    /// Completed bounded mirror-selection rounds.
    pub rounds: u64,
    /// Valid public candidates considered by the latest round.
    pub last_round_candidates: u64,
    /// Capacity-bounded candidates selected by the latest round.
    pub last_round_selected: u64,
    /// Selected producers that completed without a terminal failure.
    pub last_round_succeeded: u64,
    /// Selected candidates that failed or were rejected in the latest round.
    pub last_round_failed: u64,
    /// Selected producers that reached their authenticated signed tip.
    pub last_round_converged: u64,
    /// Selected producers that advanced but still have authenticated lag.
    pub last_round_catching_up: u64,
    /// Authenticated pages accepted in the latest round.
    pub last_round_pages_succeeded: u64,
    /// Successful HTTP requests consumed by the latest round.
    pub last_round_requests_sent: u64,
    /// Authenticated mirror pages accepted during this process lifetime.
    pub pages_succeeded: u64,
    /// Successful mirror HTTP requests during this process lifetime.
    pub requests_sent: u64,
    /// Failed bounded mirror attempts during this process lifetime.
    pub attempts_failed: u64,
    /// Direct-page failures that entered bounded carrier recovery.
    pub recovery_attempts: u64,
    /// Mirror pages recovered through an independently authenticated carrier.
    pub recovery_succeeded: u64,
    /// Bounded carrier recovery attempts that exhausted or failed closed.
    pub recovery_failed: u64,
    /// Valid non-quarantined public carriers considered by the latest recovery.
    pub last_recovery_carrier_candidates: u64,
    /// Latest candidates with fresh local routeability evidence.
    pub last_recovery_routeable_carrier_candidates: u64,
    /// Latest candidates with a signed Directory Mirror carrier capability.
    pub last_recovery_explicit_capability_candidates: u64,
    /// Latest candidates retained only as unadvertised compatibility fallback.
    ///
    /// These peers may be old or may have deliberately opted out. They remain
    /// discoverable but are never classified as capability supporting until a
    /// signed descriptor explicitly says so.
    pub last_recovery_unadvertised_compatibility_candidates: u64,
    /// Latest valid carriers skipped for the exact signed descriptor sequence
    /// after an explicit optional replica-endpoint absence response.
    pub last_recovery_capability_cached_unavailable: u64,
    /// Capacity-bounded carriers selected by the latest recovery.
    pub last_recovery_carriers_selected: u64,
    /// Latest selected carriers with fresh local routeability evidence.
    pub last_recovery_routeable_carriers_selected: u64,
    /// Latest selected carriers with explicit signed capability.
    pub last_recovery_explicit_capability_selected: u64,
    /// Latest selected unadvertised compatibility fallbacks.
    pub last_recovery_unadvertised_compatibility_selected: u64,
    /// Latest selected carriers that supplied a non-empty signed region hint.
    pub last_recovery_selected_region_hints: u64,
    /// Distinct signed region hints in the latest bounded selection.
    ///
    /// This is an availability hint only, not operator, ASN, jurisdiction,
    /// identity, Sybil-resistance, consensus, or finality evidence.
    pub last_recovery_distinct_region_hints: u64,
    /// Timestamp of the latest completed mirror round.
    pub last_round_at: Option<u64>,
    /// Timestamp of the latest authenticated mirror import.
    pub last_success_at: Option<u64>,
    /// Timestamp of the latest failed mirror attempt.
    pub last_failure_at: Option<u64>,
    /// Timestamp of the latest bounded carrier recovery attempt.
    pub last_recovery_attempt_at: Option<u64>,
    /// Timestamp of the latest successful carrier recovery.
    pub last_recovery_success_at: Option<u64>,
    /// Timestamp of the latest failed or exhausted carrier recovery.
    pub last_recovery_failure_at: Option<u64>,
}

impl DirectoryReplicaSyncObservation {
    fn new(producer: [u8; 32]) -> Self {
        Self {
            producer,
            last_attempt_at: None,
            last_success_at: None,
            last_failure_at: None,
            last_failure_reason: None,
            retry_not_before: None,
            remote_tip_height: None,
            local_tip_height: 0,
            has_more: false,
            consecutive_failures: 0,
            total_attempts: 0,
            successful_pages: 0,
            failed_attempts: 0,
            backoff_skips: 0,
            blocks_inserted: 0,
            commitments_inserted: 0,
            requests_sent: 0,
        }
    }
}

/// Shared process-lifetime synchronization and witness telemetry.
#[derive(Debug, Default)]
pub struct DirectoryReplicaSyncRuntime {
    observations: Mutex<HashMap<[u8; 32], DirectoryReplicaSyncObservation>>,
    directory_sync_transport: Mutex<DirectoryReplicaTransportRuntime>,
    observation_witness: Mutex<DirectoryObservationWitnessOutcomeSnapshot>,
    observation_witness_recovery: Mutex<DirectoryObservationWitnessRecoverySnapshot>,
    observation_witness_carrier: Mutex<DirectoryObservationWitnessCarrierSnapshot>,
    full_node_mirror: Mutex<DirectoryFullNodeMirrorRuntimeSnapshot>,
    directory_audit_admission: OnceLock<Arc<Semaphore>>,
}

impl DirectoryReplicaSyncRuntime {
    /// Returns the process-shared fail-fast admission gate for history audits.
    ///
    /// [DIRECTORY-AUDIT-OWNERSHIP 2026-08-12 by Codex] Both public and local
    /// routers are built from this runtime. Lazy initialization keeps
    /// compatibility constructors cheap while ensuring their production
    /// clones converge on exactly one permit.
    pub(crate) fn directory_audit_admission(&self) -> Arc<Semaphore> {
        Arc::clone(
            self.directory_audit_admission
                .get_or_init(|| Arc::new(Semaphore::new(MAX_DIRECTORY_AUDITS_IN_FLIGHT))),
        )
    }

    /// Records exactly one completed coordinator-owned transport outcome.
    ///
    /// The caller has already reduced the exchange to a privacy-safe class;
    /// this boundary cannot receive peer, endpoint, request, or payload data.
    pub fn record_directory_sync_transport_outcome(
        &self,
        outcome: DirectoryReplicaTransportOutcome,
        completed_at: u64,
    ) {
        self.directory_sync_transport
            .lock()
            .record(outcome, completed_at);
    }

    /// Returns process-lifetime aggregate Directory sync transport telemetry.
    #[must_use]
    pub fn directory_sync_transport_snapshot(&self) -> DirectoryReplicaTransportSnapshot {
        self.directory_sync_transport.lock().snapshot()
    }

    /// Registers configured pins so status reports can distinguish pending from
    /// disabled before the first low-frequency synchronization round.
    pub fn register_producers(&self, producers: &[[u8; 32]]) {
        let mut observations = self.observations.lock();
        for producer in producers {
            if *producer != [0u8; 32] {
                observations
                    .entry(*producer)
                    .or_insert_with(|| DirectoryReplicaSyncObservation::new(*producer));
            }
        }
    }

    /// Restores audited retry boundaries before the coordinator starts.
    ///
    /// Process-lifetime attempt/page counters remain zero after restart; only
    /// the active failure streak, retry boundary, reason, and skip count are
    /// restored because those fields control request pressure.
    pub fn restore_retry_states(&self, states: &[DirectoryReplicaRetryState]) {
        let mut observations = self.observations.lock();
        for state in states {
            let observation = observations
                .entry(state.producer)
                .or_insert_with(|| DirectoryReplicaSyncObservation::new(state.producer));
            observation.last_attempt_at = Some(state.last_failure_at);
            observation.last_failure_at = Some(state.last_failure_at);
            observation.last_failure_reason = Some(state.last_failure_reason.clone());
            observation.retry_not_before = state.retry_not_before;
            observation.consecutive_failures = state.consecutive_failures;
            observation.backoff_skips = state.backoff_skips;
        }
        drop(observations);
    }

    /// Records the beginning of one bounded page request.
    pub fn record_attempt(&self, producer: [u8; 32], attempted_at: u64) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(attempted_at);
        observation.total_attempts = observation.total_attempts.saturating_add(1);
    }

    /// Records one authenticated page after its atomic import completes.
    #[allow(clippy::too_many_arguments)]
    pub fn record_success(
        &self,
        producer: [u8; 32],
        succeeded_at: u64,
        local_tip_height: u64,
        remote_tip_height: u64,
        has_more: bool,
        blocks_inserted: u64,
        commitments_inserted: u64,
        requests_sent: u32,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(succeeded_at);
        observation.last_success_at = Some(succeeded_at);
        observation.remote_tip_height = Some(remote_tip_height);
        observation.local_tip_height = local_tip_height;
        observation.has_more = has_more;
        observation.consecutive_failures = 0;
        observation.retry_not_before = None;
        observation.successful_pages = observation.successful_pages.saturating_add(1);
        observation.blocks_inserted = observation.blocks_inserted.saturating_add(blocks_inserted);
        observation.commitments_inserted = observation
            .commitments_inserted
            .saturating_add(commitments_inserted);
        observation.requests_sent = observation
            .requests_sent
            .saturating_add(u64::from(requests_sent));
    }

    /// Records one stable failure code without retaining peer endpoints,
    /// response bodies, or underlying transport error strings.
    pub fn record_failure(
        &self,
        producer: [u8; 32],
        failed_at: u64,
        reason: &str,
        retry_not_before: Option<u64>,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(failed_at);
        observation.last_failure_at = Some(failed_at);
        observation.last_failure_reason = Some(reason.chars().take(96).collect());
        observation.retry_not_before = retry_not_before.map(|value| value.max(failed_at));
        observation.consecutive_failures = observation
            .consecutive_failures
            .saturating_add(1)
            .min(DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES);
        observation.failed_attempts = observation.failed_attempts.saturating_add(1);
        drop(observations);
    }

    /// Returns the future retry boundary for one producer, if backoff is active.
    #[must_use]
    pub fn deferred_retry_until(&self, producer: &[u8; 32], now: u64) -> Option<u64> {
        self.observations
            .lock()
            .get(producer)
            .and_then(|observation| observation.retry_not_before)
            .filter(|retry_at| *retry_at > now)
    }

    /// Returns the current consecutive failure count for backoff calculation.
    #[must_use]
    pub fn consecutive_failures(&self, producer: &[u8; 32]) -> u64 {
        self.observations
            .lock()
            .get(producer)
            .map_or(0, |observation| observation.consecutive_failures)
    }

    /// Records one timer tick intentionally skipped by producer-local backoff.
    pub fn record_backoff_skip(&self, producer: [u8; 32]) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.backoff_skips = observation.backoff_skips.saturating_add(1);
        drop(observations);
    }

    /// Records one bounded outbound witness round using mutually exclusive,
    /// privacy-safe outcome buckets.
    ///
    /// `telemetry_durable` describes only the aggregate telemetry write. An
    /// accepted attempt already means its signed receipt was persisted.
    pub fn record_observation_witness_round(
        &self,
        checkpoint_sequence: u64,
        observed_at: u64,
        outcomes: &[DirectoryObservationWitnessOutcome],
        telemetry_durable: bool,
    ) {
        if checkpoint_sequence == 0 || observed_at == 0 || outcomes.is_empty() {
            return;
        }
        let round = DirectoryObservationWitnessOutcomeCounters::from_outcomes(outcomes);
        let mut snapshot = self.observation_witness.lock();
        snapshot.rounds = snapshot.rounds.saturating_add(1);
        snapshot.totals = snapshot.totals.saturating_add(round);
        snapshot.last_checkpoint_sequence = checkpoint_sequence;
        snapshot.last_round_at = Some(observed_at);
        if round.accepted > 0 {
            snapshot.last_success_at = Some(observed_at);
        }
        if round.failures() > 0 {
            snapshot.last_failure_at = Some(observed_at);
        }
        snapshot.last_round = round;
        if !telemetry_durable {
            snapshot.telemetry_persistence_failures =
                snapshot.telemetry_persistence_failures.saturating_add(1);
        }
    }

    /// Returns process-lifetime aggregate witness telemetry.
    #[must_use]
    pub fn observation_witness_snapshot(&self) -> DirectoryObservationWitnessOutcomeSnapshot {
        *self.observation_witness.lock()
    }

    /// Records aggregate properties of one bounded witness-carrier selection.
    ///
    /// Identity-bearing candidate data is reduced to counts before this
    /// boundary and is never retained by runtime status.
    pub fn record_observation_witness_recovery_selection(
        &self,
        candidates: u64,
        routeable_candidates: u64,
        capability_cached_unavailable: u64,
        selected: u64,
        attempted_at: u64,
    ) {
        if attempted_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.selections = snapshot.selections.saturating_add(1);
        snapshot.latest_candidates = candidates;
        snapshot.latest_routeable_candidates = routeable_candidates.min(candidates);
        snapshot.latest_capability_cached_unavailable =
            capability_cached_unavailable.min(candidates);
        snapshot.latest_selected = selected.min(candidates);
        snapshot.last_attempt_at = Some(latest_runtime_timestamp(
            snapshot.last_attempt_at,
            attempted_at,
        ));
    }

    /// Records one carrier transport attempt using mutually exclusive buckets.
    pub fn record_observation_witness_recovery_attempt(
        &self,
        succeeded: bool,
        capability_unavailable: bool,
        completed_at: u64,
    ) {
        if completed_at == 0 || (succeeded && capability_unavailable) {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.attempts = snapshot.attempts.saturating_add(1);
        snapshot.last_attempt_at = Some(latest_runtime_timestamp(
            snapshot.last_attempt_at,
            completed_at,
        ));
        if succeeded {
            snapshot.succeeded = snapshot.succeeded.saturating_add(1);
            snapshot.last_success_at = Some(latest_runtime_timestamp(
                snapshot.last_success_at,
                completed_at,
            ));
            snapshot.last_outcome = Some(DirectoryObservationWitnessRecoveryOutcome::Recovered);
        } else if capability_unavailable {
            snapshot.capability_unavailable = snapshot.capability_unavailable.saturating_add(1);
        } else {
            snapshot.transport_failures = snapshot.transport_failures.saturating_add(1);
        }
    }

    /// Records one bounded recovery that exhausted every selected carrier.
    pub fn record_observation_witness_recovery_exhausted(&self, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.exhausted = snapshot.exhausted.saturating_add(1);
        snapshot.last_failure_at = Some(latest_runtime_timestamp(
            snapshot.last_failure_at,
            completed_at,
        ));
        snapshot.last_outcome = Some(DirectoryObservationWitnessRecoveryOutcome::Exhausted);
    }

    /// Records a carrier or target contract failure that stopped closed.
    pub fn record_observation_witness_recovery_failed_closed(&self, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_recovery.lock();
        snapshot.attempts = snapshot.attempts.saturating_add(1);
        snapshot.failed_closed = snapshot.failed_closed.saturating_add(1);
        snapshot.last_attempt_at = Some(latest_runtime_timestamp(
            snapshot.last_attempt_at,
            completed_at,
        ));
        snapshot.last_failure_at = Some(latest_runtime_timestamp(
            snapshot.last_failure_at,
            completed_at,
        ));
        snapshot.last_outcome = Some(DirectoryObservationWitnessRecoveryOutcome::FailedClosed);
    }

    /// Returns process-lifetime aggregate witness-carrier telemetry.
    #[must_use]
    pub fn observation_witness_recovery_snapshot(
        &self,
    ) -> DirectoryObservationWitnessRecoverySnapshot {
        *self.observation_witness_recovery.lock()
    }

    /// Records one authenticated carrier request as exactly one aggregate outcome.
    ///
    /// The caller must perform pinned requester authentication first. No
    /// identity-bearing or frame-bearing value crosses this runtime boundary.
    pub fn record_observation_witness_carrier_outcome(
        &self,
        outcome: DirectoryObservationWitnessCarrierOutcome,
        completed_at: u64,
    ) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.observation_witness_carrier.lock();
        snapshot.requests = snapshot.requests.saturating_add(1);
        snapshot.last_request_at = Some(latest_runtime_timestamp(
            snapshot.last_request_at,
            completed_at,
        ));
        snapshot.last_outcome = Some(outcome);
        match outcome {
            DirectoryObservationWitnessCarrierOutcome::Forwarded => {
                snapshot.forwarded = snapshot.forwarded.saturating_add(1);
                snapshot.last_forwarded_at = Some(latest_runtime_timestamp(
                    snapshot.last_forwarded_at,
                    completed_at,
                ));
            }
            DirectoryObservationWitnessCarrierOutcome::PolicyRejected => {
                snapshot.policy_rejected = snapshot.policy_rejected.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::InvalidRequest => {
                snapshot.invalid_requests = snapshot.invalid_requests.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetUnavailable => {
                snapshot.target_unavailable = snapshot.target_unavailable.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetCapabilityUnavailable => {
                snapshot.target_capability_unavailable =
                    snapshot.target_capability_unavailable.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetRejected => {
                snapshot.target_rejected = snapshot.target_rejected.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetInvalidResponse => {
                snapshot.target_invalid_response =
                    snapshot.target_invalid_response.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::TargetCoolingDown => {
                snapshot.target_cooling_down = snapshot.target_cooling_down.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::LocalOverloaded => {
                snapshot.local_overloaded = snapshot.local_overloaded.saturating_add(1);
            }
            DirectoryObservationWitnessCarrierOutcome::LocalFailure => {
                snapshot.local_failures = snapshot.local_failures.saturating_add(1);
            }
        }
        if outcome != DirectoryObservationWitnessCarrierOutcome::Forwarded {
            snapshot.last_failure_at = Some(latest_runtime_timestamp(
                snapshot.last_failure_at,
                completed_at,
            ));
        }
    }

    /// Returns process-lifetime aggregate telemetry for this node as a carrier.
    #[must_use]
    pub fn observation_witness_carrier_snapshot(
        &self,
    ) -> DirectoryObservationWitnessCarrierSnapshot {
        *self.observation_witness_carrier.lock()
    }

    /// Records one aggregate bounded non-authoritative mirror round.
    pub fn record_full_node_mirror_round(
        &self,
        candidates: usize,
        selected: usize,
        succeeded: usize,
        completed_at: u64,
    ) {
        self.record_full_node_mirror_catch_up_round(
            candidates,
            selected,
            succeeded,
            0,
            selected.saturating_sub(succeeded),
            u64::try_from(succeeded).unwrap_or(u64::MAX),
            0,
            completed_at,
        );
    }

    /// Records one aggregate bounded multi-page mirror catch-up round.
    ///
    /// [MIRROR-CATCHUP 2026-07-24 by Codex] Outcome buckets are mutually
    /// exclusive at producer level. Page/request totals remain aggregate-only
    /// and cannot reveal which public producer or carrier served the data.
    #[allow(clippy::too_many_arguments)]
    pub fn record_full_node_mirror_catch_up_round(
        &self,
        candidates: usize,
        selected: usize,
        converged: usize,
        catching_up: usize,
        failed: usize,
        pages_succeeded: u64,
        requests_sent: u64,
        completed_at: u64,
    ) {
        if completed_at == 0
            || converged.saturating_add(catching_up).saturating_add(failed) != selected
        {
            return;
        }
        let succeeded = converged.saturating_add(catching_up);
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.rounds = snapshot.rounds.saturating_add(1);
        snapshot.last_round_candidates = u64::try_from(candidates).unwrap_or(u64::MAX);
        snapshot.last_round_selected = u64::try_from(selected).unwrap_or(u64::MAX);
        snapshot.last_round_succeeded = u64::try_from(succeeded).unwrap_or(u64::MAX);
        snapshot.last_round_failed = u64::try_from(failed).unwrap_or(u64::MAX);
        snapshot.last_round_converged = u64::try_from(converged).unwrap_or(u64::MAX);
        snapshot.last_round_catching_up = u64::try_from(catching_up).unwrap_or(u64::MAX);
        snapshot.last_round_pages_succeeded = pages_succeeded;
        snapshot.last_round_requests_sent = requests_sent;
        snapshot.pages_succeeded = snapshot.pages_succeeded.saturating_add(pages_succeeded);
        snapshot.requests_sent = snapshot.requests_sent.saturating_add(requests_sent);
        snapshot.attempts_failed = snapshot
            .attempts_failed
            .saturating_add(snapshot.last_round_failed);
        snapshot.last_round_at = Some(completed_at);
        if pages_succeeded > 0 {
            snapshot.last_success_at = Some(completed_at);
        }
        if failed > 0 {
            snapshot.last_failure_at = Some(completed_at);
        }
    }

    /// Records one privacy-safe carrier recovery outcome.
    ///
    /// Carrier identities, endpoints, routes, producer identities, response
    /// bodies, and failure strings are intentionally excluded.
    pub fn record_full_node_mirror_recovery(&self, succeeded: bool, completed_at: u64) {
        if completed_at == 0 {
            return;
        }
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.recovery_attempts = snapshot.recovery_attempts.saturating_add(1);
        snapshot.last_recovery_attempt_at = Some(completed_at);
        if succeeded {
            snapshot.recovery_succeeded = snapshot.recovery_succeeded.saturating_add(1);
            snapshot.last_recovery_success_at = Some(completed_at);
        } else {
            snapshot.recovery_failed = snapshot.recovery_failed.saturating_add(1);
            snapshot.last_recovery_failure_at = Some(completed_at);
        }
    }

    /// Records aggregate properties of the latest bounded carrier selection.
    ///
    /// [MIRROR-CAPABILITY 2026-07-24 by Codex] Capability and signed-region
    /// values are reduced to counts before this boundary. This method never
    /// receives identities, endpoint URLs, region strings, producer
    /// identities, descriptor sequences, or selected order.
    pub fn record_full_node_mirror_carrier_selection(
        &self,
        candidates: u64,
        routeable_candidates: u64,
        explicit_capability_candidates: u64,
        unadvertised_compatibility_candidates: u64,
        capability_cached_unavailable: u64,
        selected: u64,
        selected_routeable: u64,
        selected_explicit_capability: u64,
        selected_unadvertised_compatibility: u64,
        selected_region_hints: u64,
        distinct_region_hints: u64,
    ) {
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.last_recovery_carrier_candidates = candidates;
        snapshot.last_recovery_routeable_carrier_candidates = routeable_candidates.min(candidates);
        snapshot.last_recovery_explicit_capability_candidates =
            explicit_capability_candidates.min(candidates);
        snapshot.last_recovery_unadvertised_compatibility_candidates =
            unadvertised_compatibility_candidates.min(
                candidates.saturating_sub(snapshot.last_recovery_explicit_capability_candidates),
            );
        snapshot.last_recovery_capability_cached_unavailable =
            capability_cached_unavailable.min(candidates);
        snapshot.last_recovery_carriers_selected = selected.min(candidates);
        snapshot.last_recovery_routeable_carriers_selected = selected_routeable
            .min(snapshot.last_recovery_carriers_selected)
            .min(snapshot.last_recovery_routeable_carrier_candidates);
        snapshot.last_recovery_explicit_capability_selected =
            selected_explicit_capability.min(snapshot.last_recovery_carriers_selected);
        snapshot.last_recovery_unadvertised_compatibility_selected =
            selected_unadvertised_compatibility.min(
                snapshot
                    .last_recovery_carriers_selected
                    .saturating_sub(snapshot.last_recovery_explicit_capability_selected),
            );
        snapshot.last_recovery_selected_region_hints =
            selected_region_hints.min(snapshot.last_recovery_carriers_selected);
        snapshot.last_recovery_distinct_region_hints =
            distinct_region_hints.min(snapshot.last_recovery_selected_region_hints);
    }

    /// Returns aggregate permissionless mirror telemetry without peer metadata.
    #[must_use]
    pub fn full_node_mirror_snapshot(&self) -> DirectoryFullNodeMirrorRuntimeSnapshot {
        *self.full_node_mirror.lock()
    }

    /// Returns producer observations in deterministic identity order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DirectoryReplicaSyncObservation> {
        let mut observations = self
            .observations
            .lock()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        observations.sort_by_key(|observation| observation.producer);
        observations
    }
}
