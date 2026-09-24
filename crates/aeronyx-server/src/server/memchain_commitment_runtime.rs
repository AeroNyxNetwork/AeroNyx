// ============================================
// File: crates/aeronyx-server/src/server/memchain_commitment_runtime.rs
// ============================================
// [MEMCHAIN-COMMITMENT-RUNTIME 2026-09-25 by Codex] Keep operator-pinned
// signed witness gates, coordinator leases, reconciliation, and follower
// liveness together; this is safety supervision, not consensus.
use super::*;

/// Typed startup outcomes keep security policy branches exhaustive.
///
/// These variants describe an operator-pinned evidence policy. They are not
/// network votes, consensus, quorum, finality, leader election, or fork choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CommitmentWitnessStartupDecision {
    Verified,
    DegradedUnverified,
    DegradedBelowThreshold,
}

impl CommitmentWitnessStartupDecision {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::DegradedUnverified => "degraded_unverified",
            Self::DegradedBelowThreshold => "degraded_below_threshold",
        }
    }
}

/// Fail-closed reasons produced by authenticated witness evidence or policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CommitmentWitnessStartupBlockReason {
    Equivocation,
    TrustedDivergence,
    Divergence,
    RemoteAhead,
    Unavailable,
    ThresholdUnmet,
    CertificateUnavailable,
}

impl CommitmentWitnessStartupBlockReason {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Equivocation => "signed_checkpoint_equivocation",
            Self::TrustedDivergence => "trusted_checkpoint_divergence_incident",
            Self::Divergence => "signed_checkpoint_divergence",
            Self::RemoteAhead => "signed_checkpoint_remote_ahead",
            Self::Unavailable => "signed_checkpoint_unavailable",
            Self::ThresholdUnmet => "signed_checkpoint_threshold_unmet",
            Self::CertificateUnavailable => "signed_checkpoint_certificate_unavailable",
        }
    }
}

impl std::fmt::Display for CommitmentWitnessStartupBlockReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub(super) fn commitment_witness_startup_decision(
    round: &CommitmentReconciliationOutcome,
    signed_evidence_required: bool,
    minimum_verified_witnesses: usize,
) -> std::result::Result<CommitmentWitnessStartupDecision, CommitmentWitnessStartupBlockReason> {
    if round.diverged > 0 {
        return Err(CommitmentWitnessStartupBlockReason::Divergence);
    }
    if round.remote_ahead > 0 {
        return Err(CommitmentWitnessStartupBlockReason::RemoteAhead);
    }
    let minimum_verified_witnesses = minimum_verified_witnesses.max(1);
    if round.verified < minimum_verified_witnesses {
        if signed_evidence_required {
            return if round.verified == 0 {
                Err(CommitmentWitnessStartupBlockReason::Unavailable)
            } else {
                Err(CommitmentWitnessStartupBlockReason::ThresholdUnmet)
            };
        }
        return if round.verified == 0 {
            Ok(CommitmentWitnessStartupDecision::DegradedUnverified)
        } else {
            Ok(CommitmentWitnessStartupDecision::DegradedBelowThreshold)
        };
    }
    Ok(CommitmentWitnessStartupDecision::Verified)
}

pub(super) const COORDINATOR_LEASE_PRODUCTION_SAFETY_SECS: u64 = 15;
pub(super) const COORDINATOR_LEASE_DEGRADED_RETRY_SECS: u64 = 10;

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CommitmentCoordinatorLeaseRound {
    pub(super) attempted: usize,
    pub(super) granted: usize,
    pub(super) contended: usize,
    pub(super) failed: usize,
    pub(super) minimum_valid_for_secs: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CommitmentCoordinatorLeaseReleaseRound {
    pub(super) attempted: usize,
    pub(super) released: usize,
    pub(super) not_holder: usize,
    pub(super) failed: usize,
}

pub(super) async fn collect_commitment_coordinator_lease_round(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    instance_id: &[u8; 32],
    requested_ttl_secs: u32,
) -> CommitmentCoordinatorLeaseRound {
    let requests = witness_node_ids.iter().map(|witness_node_id| {
        request_record_commitment_coordinator_lease(
            storage,
            peer_store,
            identity,
            witness_node_id,
            instance_id,
            requested_ttl_secs,
            client,
        )
    });
    let results = futures::future::join_all(requests).await;
    let mut round = CommitmentCoordinatorLeaseRound {
        attempted: witness_node_ids.len(),
        minimum_valid_for_secs: u64::MAX,
        ..Default::default()
    };
    for result in results {
        match result {
            Ok(grant) => {
                round.granted = round.granted.saturating_add(1);
                round.minimum_valid_for_secs =
                    round.minimum_valid_for_secs.min(grant.valid_for_secs);
            }
            Err(error) if error == "lease_contended" => {
                round.contended = round.contended.saturating_add(1);
            }
            Err(_) => round.failed = round.failed.saturating_add(1),
        }
    }
    if round.granted == 0 {
        round.minimum_valid_for_secs = 0;
    }
    round
}

/// Releases one process instance from every configured witness concurrently.
///
/// The aggregate deliberately omits witness identities and endpoints. A
/// partial release remains safe because any unreleased grant expires at its
/// original signed deadline and cannot authorize the next random instance.
pub(super) async fn collect_commitment_coordinator_lease_release_round(
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    client: &reqwest::Client,
    witness_node_ids: &[[u8; 32]],
    instance_id: &[u8; 32],
) -> CommitmentCoordinatorLeaseReleaseRound {
    let requests = witness_node_ids.iter().map(|witness_node_id| {
        release_record_commitment_coordinator_lease(
            peer_store,
            identity,
            witness_node_id,
            instance_id,
            client,
        )
    });
    let results = futures::future::join_all(requests).await;
    let mut round = CommitmentCoordinatorLeaseReleaseRound {
        attempted: witness_node_ids.len(),
        ..Default::default()
    };
    for result in results {
        match result {
            Ok(_) => round.released = round.released.saturating_add(1),
            Err(error) if error == "lease_release_not_holder" => {
                round.not_holder = round.not_holder.saturating_add(1);
            }
            Err(_) => round.failed = round.failed.saturating_add(1),
        }
    }
    round
}

/// Returns the conservative local production window for a complete lease round.
///
/// The coordinator intentionally requires every configured witness rather than
/// reusing the checkpoint evidence threshold. A partial or ambiguous round must
/// not refresh production authority; the safety margin also ensures local
/// production stops before the earliest remote grant expires.
pub(super) fn commitment_coordinator_lease_production_valid_for(
    round: &CommitmentCoordinatorLeaseRound,
    required_witnesses: usize,
) -> Option<u64> {
    if required_witnesses == 0
        || round.attempted != required_witnesses
        || round.granted != required_witnesses
        || round.contended != 0
        || round.failed != 0
    {
        return None;
    }
    let valid_for_secs = round
        .minimum_valid_for_secs
        .saturating_sub(COORDINATOR_LEASE_PRODUCTION_SAFETY_SECS);
    (valid_for_secs > 0).then_some(valid_for_secs)
}

/// Selects a bounded retry delay after an incomplete lease round.
///
/// While existing authority remains valid, retries converge toward its
/// monotonic deadline so a recovered witness can refresh the lease before
/// production stops. Once authority is unavailable, a fixed low-frequency
/// recovery probe avoids both a 40-second blind spot and a hot retry loop.
pub(super) fn commitment_coordinator_lease_degraded_retry_delay(
    normal_interval_secs: u64,
    production_permitted: bool,
    seconds_remaining: Option<u64>,
) -> u64 {
    let normal_interval_secs = normal_interval_secs.max(1);
    if !production_permitted {
        return COORDINATOR_LEASE_DEGRADED_RETRY_SECS.min(normal_interval_secs);
    }
    seconds_remaining.map_or(
        COORDINATOR_LEASE_DEGRADED_RETRY_SECS.min(normal_interval_secs),
        |remaining| (remaining / 2).max(1).min(normal_interval_secs),
    )
}

/// Typed result of one bounded follower block/certificate synchronization round.
///
/// [FOLLOWER-CERTIFICATE-RETRY 2026-07-30 by Codex] Block backlog, certified
/// carrier recovery, and a verified-but-unpersisted certificate have different
/// scheduling and trust meanings. Keeping them as named fields prevents tuple
/// position mistakes and avoids treating deferred durability as either a block
/// transport failure or a completed certificate recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CommitmentFollowerRoundOutcome {
    pub(super) inserted: usize,
    pub(super) remote_tip_height: u64,
    pub(super) block_backlog_remaining: bool,
    pub(super) certificate_retry_pending: bool,
    pub(super) certified_recovered: bool,
}

/// Selects the next successful follower-round delay without creating a hot loop.
///
/// Block backlog remains the highest-priority one-second continuation. A
/// certificate that verified but lost a local persistence race retries with
/// bounded exponential delay until it reaches the configured normal interval.
/// This state is process-local and cannot affect chain choice or peer authority.
pub(super) fn commitment_follower_success_retry_delay(
    base_interval_secs: u64,
    outcome: &CommitmentFollowerRoundOutcome,
    consecutive_certificate_deferrals: u32,
) -> u64 {
    const MAX_CERTIFICATE_RETRY_SHIFT: u32 = 12;

    let base_interval_secs = base_interval_secs.max(1);
    if outcome.block_backlog_remaining {
        return 1;
    }
    if !outcome.certificate_retry_pending {
        return base_interval_secs;
    }
    let shift = consecutive_certificate_deferrals
        .max(1)
        .saturating_sub(1)
        .min(MAX_CERTIFICATE_RETRY_SHIFT);
    (1u64 << shift).min(base_interval_secs)
}

/// Result of waiting for one outbound tip announcement while continuing to
/// observe the bounded miner notification channel.
///
/// A superseded future is deliberately dropped. The peer protocol is
/// idempotent, so a canceled HTTP request that already reached a follower can
/// only produce an extra wake-up; it cannot change either node's chain.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum CommitmentTipAnnouncementWaitOutcome<T> {
    Completed(T),
    Superseded(u64),
}

/// Waits for an outbound announcement unless a strictly newer audited tip is
/// observed first.
///
/// Equal or older notifications are coalesced as stale process-local work.
/// The caller owns shutdown handling so dropping this helper also cancels the
/// in-flight HTTP/retry future without coupling delivery to block production.
pub(super) async fn await_commitment_tip_announcement_or_newer<F>(
    current_tip_height: u64,
    commitment_tip_rx: &mut mpsc::Receiver<u64>,
    announcement: F,
) -> CommitmentTipAnnouncementWaitOutcome<F::Output>
where
    F: std::future::Future,
{
    tokio::pin!(announcement);
    let mut notification_channel_open = true;
    loop {
        tokio::select! {
            outcome = &mut announcement => {
                return CommitmentTipAnnouncementWaitOutcome::Completed(outcome);
            }
            next_tip = commitment_tip_rx.recv(), if notification_channel_open => {
                match next_tip {
                    Some(next_tip_height) if next_tip_height > current_tip_height => {
                        return CommitmentTipAnnouncementWaitOutcome::Superseded(next_tip_height);
                    }
                    Some(_) => {}
                    None => notification_channel_open = false,
                }
            }
        }
    }
}

/// Ensures a configured follower cannot retain a stale ready state after its
/// Tokio task exits, panics, or is aborted.
///
/// [FOLLOWER-TASK-LIVENESS 2026-07-30 by Codex] This guard changes only
/// process-local operational status. It never writes chain data, alters trust
/// policy, or turns task liveness into consensus evidence.
pub(super) struct CommitmentSyncTaskLivenessGuard {
    pub(super) storage: Arc<MemoryStorage>,
}

impl CommitmentSyncTaskLivenessGuard {
    pub(super) fn new(storage: Arc<MemoryStorage>) -> Self {
        Self { storage }
    }
}

impl Drop for CommitmentSyncTaskLivenessGuard {
    fn drop(&mut self) {
        self.storage.stop_record_commitment_sync();
    }
}

impl Server {
    /// Publishes the current coordinator descriptor before strict witness gates.
    ///
    /// [WITNESS-DESCRIPTOR-PREFLIGHT 2026-07-29 by Codex] Background gossip is
    /// intentionally started only after startup integrity checks. This bounded
    /// preflight closes the endpoint-rotation deadlock without moving authority:
    /// it sends only the already-public signed descriptor, and checkpoint plus
    /// all-witness lease verification still decide whether production starts.
    pub(super) async fn publish_memchain_commitment_descriptor_preflight(
        &self,
        peer_store: &PeerStore,
        control_http_client: &reqwest::Client,
    ) {
        let memchain = &self.config.memchain;
        let strict_gate_enabled = memchain.commitment_witness_startup_required
            || memchain.commitment_coordinator_lease_required;
        if !memchain.commitment_coordinator_enabled
            || !strict_gate_enabled
            || !self.config.discovery.enabled
            || !self.config.discovery.advertise_self
        {
            return;
        }

        let witness_node_ids = memchain.commitment_witness_node_id_bytes();
        if witness_node_ids.is_empty() {
            return;
        }
        let now = unix_now_secs();
        let self_node_id = self.identity.public_key_bytes();
        let Some(self_descriptor) = peer_store.get_valid(&self_node_id, now) else {
            warn!(
                configured = witness_node_ids.len(),
                "[MEMCHAIN_BLOCK] Coordinator descriptor preflight unavailable; strict witness gates remain authoritative"
            );
            return;
        };

        let round = publish_current_descriptor_to_commitment_witnesses(
            peer_store,
            &self_descriptor,
            control_http_client,
            &witness_node_ids,
        )
        .await;
        if round.accepted == round.configured && round.configured > 0 {
            info!(
                configured = round.configured,
                attempted = round.attempted,
                accepted = round.accepted,
                "[MEMCHAIN_BLOCK] Coordinator descriptor preflight completed"
            );
        } else {
            warn!(
                configured = round.configured,
                attempted = round.attempted,
                accepted = round.accepted,
                failed = round.failed,
                "[MEMCHAIN_BLOCK] Coordinator descriptor preflight incomplete; strict witness gates remain authoritative"
            );
        }
    }

    /// Verifies operator-pinned external checkpoint evidence before listeners.
    ///
    /// A pinned audited follower may attest that its canonical chain copy is
    /// beyond, or inconsistent with, the local audited tip. Such positive
    /// evidence fails closed. Network absence fails open unless the operator
    /// explicitly enables `commitment_witness_startup_required`, preserving
    /// backward-compatible availability while allowing strict deployments.
    pub(super) async fn verify_memchain_commitment_startup_witnesses(
        &self,
        storage: &MemoryStorage,
        peer_store: &PeerStore,
        control_http_client: &reqwest::Client,
    ) -> Result<()> {
        if !self.config.memchain.commitment_coordinator_enabled {
            return Ok(());
        }

        let witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        if witness_node_ids.is_empty() {
            info!(
                "[MEMCHAIN_BLOCK] External startup witness guard not configured; local signed tip guard remains active"
            );
            return Ok(());
        }

        let existing_equivocations = storage
            .count_record_commitment_checkpoint_equivocations_for_witnesses(&witness_node_ids)
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "MemChain external witness rollback guard: durable incident query failed: {error}"
                ))
            })?;
        if existing_equivocations > 0 {
            return Err(ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {}",
                CommitmentWitnessStartupBlockReason::Equivocation
            )));
        }
        let existing_trusted_divergences = storage
            .count_record_commitment_checkpoint_trusted_divergences_for_witnesses(
                &witness_node_ids,
            )
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "MemChain external witness rollback guard: durable divergence query failed: {error}"
                ))
            })?;
        if existing_trusted_divergences > 0 {
            return Err(ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {}",
                CommitmentWitnessStartupBlockReason::TrustedDivergence
            )));
        }

        let round = reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
            storage,
            peer_store,
            &self.identity,
            control_http_client,
            &witness_node_ids,
            self.config.memchain.commitment_witness_min_verified,
        )
        .await;
        let observed_equivocations = storage
            .count_record_commitment_checkpoint_equivocations_for_witnesses(&witness_node_ids)
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "MemChain external witness rollback guard: post-round incident query failed: {error}"
                ))
            })?;
        if observed_equivocations > 0 {
            return Err(ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {}",
                CommitmentWitnessStartupBlockReason::Equivocation
            )));
        }
        let observed_trusted_divergences = storage
            .count_record_commitment_checkpoint_trusted_divergences_for_witnesses(
                &witness_node_ids,
            )
            .await
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "MemChain external witness rollback guard: post-round divergence query failed: {error}"
                ))
            })?;
        if observed_trusted_divergences > 0 {
            return Err(ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {}",
                CommitmentWitnessStartupBlockReason::TrustedDivergence
            )));
        }
        let certificate_required = self.config.memchain.commitment_witness_startup_required
            && self.config.memchain.commitment_witness_min_verified >= 2;
        if certificate_required
            && round.verified >= self.config.memchain.commitment_witness_min_verified
            && (!round.certificate_persisted || round.certificate_persistence_failed)
        {
            return Err(ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {}",
                CommitmentWitnessStartupBlockReason::CertificateUnavailable
            )));
        }
        let decision = commitment_witness_startup_decision(
            &round,
            self.config.memchain.commitment_witness_startup_required,
            self.config.memchain.commitment_witness_min_verified,
        )
        .map_err(|reason| {
            ServerError::startup_failed(format!(
                "MemChain external witness rollback guard: {reason}"
            ))
        })?;

        match decision {
            CommitmentWitnessStartupDecision::DegradedUnverified
            | CommitmentWitnessStartupDecision::DegradedBelowThreshold => {
                warn!(
                    configured = witness_node_ids.len(),
                    eligible = round.eligible_witnesses,
                    attempted = round.attempted,
                    verified = round.verified,
                    failed = round.failed,
                    minimum_verified = self.config.memchain.commitment_witness_min_verified,
                    strict = self.config.memchain.commitment_witness_startup_required,
                    decision = decision.as_str(),
                    "[MEMCHAIN_BLOCK] External startup witness guard is below the configured evidence threshold; continuing in availability mode"
                );
            }
            CommitmentWitnessStartupDecision::Verified => {
                info!(
                    configured = witness_node_ids.len(),
                    eligible = round.eligible_witnesses,
                    attempted = round.attempted,
                    verified = round.verified,
                    converged = round.converged,
                    remote_behind = round.remote_behind,
                    certificate_signers = round.certificate_signers,
                    certificate_required_signers = round.certificate_required_signers,
                    certificate_persisted = round.certificate_persisted,
                    minimum_verified = self.config.memchain.commitment_witness_min_verified,
                    strict = self.config.memchain.commitment_witness_startup_required,
                    "[MEMCHAIN_BLOCK] External startup witness rollback guard passed"
                );
            }
        }
        Ok(())
    }

    /// Obtains the initial all-witness coordinator lease before listeners.
    ///
    /// Enforcement remains default-off. When enabled, every operator-pinned
    /// witness must return a fresh signature for the same random process
    /// instance and exact audited tip. A partial round never authorizes block
    /// production and exposes no witness identity in logs or status.
    pub(super) async fn acquire_memchain_commitment_coordinator_lease(
        &self,
        storage: &MemoryStorage,
        peer_store: &PeerStore,
        control_http_client: &reqwest::Client,
    ) -> Result<Option<[u8; 32]>> {
        if !self.config.memchain.commitment_coordinator_lease_required {
            return Ok(None);
        }
        let witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        if witness_node_ids.is_empty() {
            return Err(ServerError::startup_failed(
                "MemChain coordinator lease: no validated witness policy",
            ));
        }
        let mut instance_id = [0u8; 32];
        while instance_id.iter().all(|byte| *byte == 0) {
            rand::rngs::OsRng.fill_bytes(&mut instance_id);
        }
        let round = collect_commitment_coordinator_lease_round(
            storage,
            peer_store,
            &self.identity,
            control_http_client,
            &witness_node_ids,
            &instance_id,
            self.config.memchain.commitment_coordinator_lease_ttl_secs,
        )
        .await;
        let Some(valid_for_secs) =
            commitment_coordinator_lease_production_valid_for(&round, witness_node_ids.len())
        else {
            storage.record_commitment_coordinator_lease_failure(round.granted);
            return Err(ServerError::startup_failed(format!(
                "MemChain coordinator lease: all-witness grant unavailable (configured={}, attempted={}, granted={}, contended={}, failed={})",
                witness_node_ids.len(),
                round.attempted,
                round.granted,
                round.contended,
                round.failed,
            )));
        };
        let now = unix_now_secs();
        storage
            .apply_record_commitment_coordinator_lease(round.granted, valid_for_secs, now)
            .map_err(|error| {
                ServerError::startup_failed(format!(
                    "MemChain coordinator lease: runtime gate rejected grant: {error}"
                ))
            })?;
        info!(
            configured = witness_node_ids.len(),
            granted = round.granted,
            production_valid_for_secs = valid_for_secs,
            "[MEMCHAIN_BLOCK] All-witness coordinator lease acquired"
        );
        Ok(Some(instance_id))
    }

    /// Renews strict coordinator authority before its monotonic deadline.
    pub(super) fn spawn_memchain_commitment_coordinator_lease_task(
        &self,
        storage: Arc<MemoryStorage>,
        peer_store: Arc<PeerStore>,
        instance_id: Option<[u8; 32]>,
        control_http_client: Arc<reqwest::Client>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.memchain.commitment_coordinator_lease_required {
            return None;
        }
        let instance_id = instance_id?;
        let witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        let requested_ttl_secs = self.config.memchain.commitment_coordinator_lease_ttl_secs;
        let renewal_interval_secs = (u64::from(requested_ttl_secs) / 3).max(10);
        let identity = self.identity.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        Some(tokio::spawn(async move {
            info!(
                interval_secs = renewal_interval_secs,
                required_witnesses = witness_node_ids.len(),
                "[MEMCHAIN_BLOCK] Coordinator lease renewal started"
            );
            let mut next_round_delay_secs = renewal_interval_secs;
            loop {
                let shutdown_requested = tokio::select! {
                    _ = shutdown_rx.recv() => true,
                    _ = tokio::time::sleep(Duration::from_secs(next_round_delay_secs)) => false,
                };
                if shutdown_requested {
                    break;
                }

                // Once started, a renewal round must finish before release.
                // Cancelling HTTP futures here could let a witness persist a
                // late grant after the shutdown path has already released it.
                let round = collect_commitment_coordinator_lease_round(
                    &storage,
                    &peer_store,
                    &identity,
                    control_http_client.as_ref(),
                    &witness_node_ids,
                    &instance_id,
                    requested_ttl_secs,
                )
                .await;
                if let Some(valid_for_secs) = commitment_coordinator_lease_production_valid_for(
                    &round,
                    witness_node_ids.len(),
                ) {
                    let previous_lease_status = storage.record_commitment_chain_integrity_status();
                    if let Err(error) = storage.apply_record_commitment_coordinator_lease(
                        round.granted,
                        valid_for_secs,
                        unix_now_secs(),
                    ) {
                        storage.record_commitment_coordinator_lease_failure(round.granted);
                        let degraded = storage.record_commitment_chain_integrity_status();
                        next_round_delay_secs = commitment_coordinator_lease_degraded_retry_delay(
                            renewal_interval_secs,
                            degraded.coordinator_lease_production_permitted,
                            degraded.coordinator_lease_seconds_remaining,
                        );
                        error!(
                            error = %error,
                            lease_state = degraded.coordinator_lease_state,
                            next_retry_secs = next_round_delay_secs,
                            "[MEMCHAIN_BLOCK] Coordinator lease runtime update failed"
                        );
                    } else if previous_lease_status.coordinator_lease_consecutive_failures > 0 {
                        next_round_delay_secs = renewal_interval_secs;
                        let recovered = storage.record_commitment_chain_integrity_status();
                        info!(
                            granted = round.granted,
                            production_valid_for_secs = valid_for_secs,
                            previous_consecutive_failures =
                                previous_lease_status.coordinator_lease_consecutive_failures,
                            recoveries_total = recovered.coordinator_lease_recoveries_total,
                            "[MEMCHAIN_BLOCK] Coordinator lease renewal recovered"
                        );
                    } else {
                        next_round_delay_secs = renewal_interval_secs;
                        debug!(
                            granted = round.granted,
                            production_valid_for_secs = valid_for_secs,
                            "[MEMCHAIN_BLOCK] Coordinator lease renewed"
                        );
                    }
                } else {
                    storage.record_commitment_coordinator_lease_failure(round.granted);
                    let degraded = storage.record_commitment_chain_integrity_status();
                    next_round_delay_secs = commitment_coordinator_lease_degraded_retry_delay(
                        renewal_interval_secs,
                        degraded.coordinator_lease_production_permitted,
                        degraded.coordinator_lease_seconds_remaining,
                    );
                    warn!(
                        attempted = round.attempted,
                        granted = round.granted,
                        contended = round.contended,
                        failed = round.failed,
                        lease_state = degraded.coordinator_lease_state,
                        production_permitted = degraded.coordinator_lease_production_permitted,
                        seconds_remaining =
                            degraded.coordinator_lease_seconds_remaining.unwrap_or(0),
                        consecutive_failures = degraded.coordinator_lease_consecutive_failures,
                        next_retry_secs = next_round_delay_secs,
                        "[MEMCHAIN_BLOCK] Coordinator lease renewal incomplete"
                    );
                }
            }

            let release_round = collect_commitment_coordinator_lease_release_round(
                &peer_store,
                &identity,
                control_http_client.as_ref(),
                &witness_node_ids,
                &instance_id,
            )
            .await;
            if release_round.released == release_round.attempted {
                info!(
                    attempted = release_round.attempted,
                    released = release_round.released,
                    "[MEMCHAIN_BLOCK] Coordinator leases released for graceful shutdown"
                );
            } else {
                warn!(
                    attempted = release_round.attempted,
                    released = release_round.released,
                    not_holder = release_round.not_holder,
                    failed = release_round.failed,
                    "[MEMCHAIN_BLOCK] Coordinator lease release incomplete; remaining grants retain bounded expiry"
                );
            }
            info!("[MEMCHAIN_BLOCK] Coordinator lease renewal stopped");
        }))
    }

    /// Starts low-frequency signed checkpoint evidence collection on the
    /// configured Block Sync v1 coordinator.
    ///
    /// Discovered encrypted-storage peers act only as witnesses. Their signed
    /// observations are independently verified and stored, but peer count is
    /// never interpreted as votes, quorum, finality, or fork choice. A remote
    /// ahead/diverged result is operator evidence and cannot mutate the local
    /// canonical chain.
    pub(super) fn spawn_memchain_commitment_reconciliation_task(
        &self,
        storage: Arc<MemoryStorage>,
        peer_store: Arc<PeerStore>,
        mut commitment_tip_rx: mpsc::Receiver<u64>,
        sync_http_client: Arc<reqwest::Client>,
    ) -> Option<JoinHandle<()>> {
        if !self.config.memchain.commitment_coordinator_enabled {
            return None;
        }

        const INITIAL_DELAY_SECS: u64 = 15;
        const MIN_INTERVAL_SECS: u64 = 300;
        const MAX_WITNESSES_PER_ROUND: usize = 3;
        let identity = self.identity.clone();
        let interval_secs = self
            .config
            .memchain
            .commitment_sync_interval_secs
            .max(MIN_INTERVAL_SECS);
        let pinned_witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        let certificate_minimum_signers = self.config.memchain.commitment_witness_min_verified;
        let witness_scope = if pinned_witness_node_ids.is_empty() {
            "permissionless_evidence"
        } else {
            "operator_pinned"
        };
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        Some(tokio::spawn(async move {
            let mut next_delay = Duration::from_secs(INITIAL_DELAY_SECS);
            let mut consecutive_unverified_rounds = 0u32;
            // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] This circuit
            // belongs only to coordinator post-startup certificate backfill.
            // It never shares follower state and retains no source identity.
            let mut certificate_carrier_circuit_breaker =
                CommitmentCertificateCarrierCircuitBreaker::default();
            info!(
                interval_secs,
                max_witnesses = MAX_WITNESSES_PER_ROUND,
                witness_scope,
                event_driven = true,
                "[MEMCHAIN_BLOCK] Coordinator witness reconciliation started"
            );

            'witness_loop: loop {
                let (trigger, announced_tip_height) = tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    _ = tokio::time::sleep(next_delay) => ("scheduled", None),
                    Some(tip_height) = commitment_tip_rx.recv() => {
                        ("new_commitment_tip", Some(tip_height))
                    }
                };
                if let Some(tip_height) = announced_tip_height {
                    info!(
                        tip_height,
                        "[MEMCHAIN_BLOCK] New commitment tip triggered witness reconciliation"
                    );
                    if !pinned_witness_node_ids.is_empty() {
                        let mut pending_tip_height = tip_height;
                        loop {
                            // Re-read the authoritative audited tip before each
                            // attempt. This prevents a coalesced older channel
                            // value from canceling delivery of a newer local tip.
                            let announcement_tip_height = storage
                                .record_commitment_chain_tip()
                                .await
                                .0
                                .max(pending_tip_height);
                            let wait_outcome = tokio::select! {
                                _ = shutdown_rx.recv() => break 'witness_loop,
                                outcome = await_commitment_tip_announcement_or_newer(
                                    announcement_tip_height,
                                    &mut commitment_tip_rx,
                                    announce_current_record_commitment_tip(
                                        &storage,
                                        &peer_store,
                                        &identity,
                                        sync_http_client.as_ref(),
                                        &pinned_witness_node_ids,
                                    ),
                                ) => outcome,
                            };
                            let completed_height = match wait_outcome {
                                CommitmentTipAnnouncementWaitOutcome::Superseded(
                                    newer_tip_height,
                                ) => {
                                    storage.record_commitment_outbound_announcement_superseded(
                                        unix_now_secs(),
                                    );
                                    pending_tip_height = pending_tip_height.max(newer_tip_height);
                                    debug!(
                                        superseded_height = announcement_tip_height,
                                        latest_height = pending_tip_height,
                                        "[MEMCHAIN_BLOCK] Newer commitment tip superseded an in-flight announcement"
                                    );
                                    continue;
                                }
                                CommitmentTipAnnouncementWaitOutcome::Completed(Ok(delivery)) => {
                                    storage.record_commitment_outbound_announcement(
                                        unix_now_secs(),
                                        delivery.announced_height,
                                        delivery.attempted,
                                        delivery.accepted,
                                        delivery.stale,
                                        delivery.failed,
                                        delivery.retries_attempted,
                                        delivery.retries_succeeded,
                                        delivery.retries_exhausted,
                                    );
                                    debug!(
                                        announced_height = delivery.announced_height,
                                        attempted = delivery.attempted,
                                        accepted = delivery.accepted,
                                        stale = delivery.stale,
                                        failed = delivery.failed,
                                        retries_attempted = delivery.retries_attempted,
                                        retries_succeeded = delivery.retries_succeeded,
                                        retries_exhausted = delivery.retries_exhausted,
                                        "[MEMCHAIN_BLOCK] Sent bounded follower tip announcements"
                                    );
                                    delivery.announced_height
                                }
                                CommitmentTipAnnouncementWaitOutcome::Completed(Err(reason)) => {
                                    storage.record_commitment_outbound_announcement_skipped(
                                        unix_now_secs(),
                                    );
                                    warn!(
                                        reason = %reason,
                                        "[MEMCHAIN_BLOCK] Tip announcement skipped; witness reconciliation retained"
                                    );
                                    announcement_tip_height
                                }
                            };

                            // The notification channel has capacity one, but
                            // drain defensively so a completion/new-tip race
                            // cannot send witness work ahead of the latest tip.
                            while let Ok(queued_tip_height) = commitment_tip_rx.try_recv() {
                                pending_tip_height = pending_tip_height.max(queued_tip_height);
                            }
                            if pending_tip_height > completed_height {
                                debug!(
                                    completed_height,
                                    latest_height = pending_tip_height,
                                    "[MEMCHAIN_BLOCK] Delivering a tip queued during announcement completion"
                                );
                                continue;
                            }
                            break;
                        }
                    }
                }

                let round = tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    outcome = async {
                        if pinned_witness_node_ids.is_empty() {
                            reconcile_record_commitment_witnesses(
                                &storage,
                                &peer_store,
                                &identity,
                                sync_http_client.as_ref(),
                                MAX_WITNESSES_PER_ROUND,
                            )
                            .await
                        } else {
                            reconcile_record_commitment_pinned_witnesses_with_certificate_threshold(
                                &storage,
                                &peer_store,
                                &identity,
                                sync_http_client.as_ref(),
                                &pinned_witness_node_ids,
                                certificate_minimum_signers,
                            )
                            .await
                        }
                    } => outcome,
                };
                next_delay = Duration::from_secs(interval_secs);

                if storage.record_commitment_production_halted() {
                    error!(
                        attempted = round.attempted,
                        verified = round.verified,
                        failed = round.failed,
                        trigger,
                        "[MEMCHAIN_BLOCK] Trusted witness security incident halted local commitment production"
                    );
                    break;
                }
                if round.certificate_persistence_failed {
                    error!(
                        certificate_signers = round.certificate_signers,
                        certificate_required_signers = round.certificate_required_signers,
                        trigger,
                        "[MEMCHAIN_BLOCK] Checkpoint certificate persistence failed"
                    );
                }
                if !pinned_witness_node_ids.is_empty()
                    && certificate_minimum_signers >= 2
                    && !round.certificate_persisted
                    && !round.certificate_persistence_failed
                {
                    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex]
                    // Certificate exchange is post-startup evidence only. One
                    // fail-closed primitive now owns source ordering, circuit
                    // state, and error classification: only availability may
                    // advance, and a verified non-persisted result stops this
                    // round so a local tip/policy race cannot fan out.
                    let recovery = tokio::select! {
                        _ = shutdown_rx.recv() => break 'witness_loop,
                        recovery =
                            recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime(
                                &storage,
                                &peer_store,
                                &identity,
                                &pinned_witness_node_ids,
                                certificate_minimum_signers,
                                MAX_WITNESSES_PER_ROUND,
                                sync_http_client.as_ref(),
                                &mut certificate_carrier_circuit_breaker,
                            ) => recovery,
                    };
                    let telemetry_disposition = match recovery.disposition {
                        CommitmentCertificateCarrierRecoveryDisposition::Persisted => {
                            RecordCommitmentCertificateBackfillDisposition::Persisted
                        }
                        CommitmentCertificateCarrierRecoveryDisposition::VerifiedUnpersisted => {
                            RecordCommitmentCertificateBackfillDisposition::VerifiedUnpersisted
                        }
                        CommitmentCertificateCarrierRecoveryDisposition::AvailabilityExhausted => {
                            RecordCommitmentCertificateBackfillDisposition::AvailabilityExhausted
                        }
                        CommitmentCertificateCarrierRecoveryDisposition::SecurityStopped => {
                            RecordCommitmentCertificateBackfillDisposition::SecurityStopped
                        }
                    };
                    // [CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex]
                    // Publish the terminal result and its anonymous circuit
                    // observation atomically before logging or later control
                    // flow can return from this round.
                    storage.record_commitment_certificate_backfill_outcome(
                        unix_now_secs(),
                        telemetry_disposition,
                        recovery.carrier_attempts,
                        recovery.cooling_slots,
                        recovery.cooldown_skips,
                        recovery.half_open_attempts,
                    );
                    match recovery.disposition {
                        CommitmentCertificateCarrierRecoveryDisposition::Persisted => {
                            debug!(
                                checkpoint_height = recovery.checkpoint_height,
                                certificate_signers = recovery.signer_count,
                                certificate_required_signers = recovery.required_signers,
                                carrier_attempts = recovery.carrier_attempts,
                                carrier_cooldown_skips = recovery.cooldown_skips,
                                carrier_half_open_attempts = recovery.half_open_attempts,
                                carrier_cooling_slots = recovery.cooling_slots,
                                "[MEMCHAIN_BLOCK] Imported audited checkpoint certificate evidence"
                            );
                        }
                        CommitmentCertificateCarrierRecoveryDisposition::VerifiedUnpersisted => {
                            debug!(
                                checkpoint_height = recovery.checkpoint_height,
                                certificate_signers = recovery.signer_count,
                                certificate_required_signers = recovery.required_signers,
                                carrier_attempts = recovery.carrier_attempts,
                                carrier_cooldown_skips = recovery.cooldown_skips,
                                carrier_half_open_attempts = recovery.half_open_attempts,
                                carrier_cooling_slots = recovery.cooling_slots,
                                "[MEMCHAIN_BLOCK] Verified checkpoint certificate deferred after local state changed"
                            );
                        }
                        CommitmentCertificateCarrierRecoveryDisposition::AvailabilityExhausted => {
                            debug!(
                                carrier_attempts = recovery.carrier_attempts,
                                carrier_cooldown_skips = recovery.cooldown_skips,
                                carrier_half_open_attempts = recovery.half_open_attempts,
                                carrier_cooling_slots = recovery.cooling_slots,
                                "[MEMCHAIN_BLOCK] Checkpoint certificate carriers unavailable"
                            );
                        }
                        CommitmentCertificateCarrierRecoveryDisposition::SecurityStopped => {
                            warn!(
                                carrier_attempts = recovery.carrier_attempts,
                                carrier_cooldown_skips = recovery.cooldown_skips,
                                carrier_half_open_attempts = recovery.half_open_attempts,
                                carrier_cooling_slots = recovery.cooling_slots,
                                "[MEMCHAIN_BLOCK] Checkpoint certificate carrier recovery stopped on a security failure"
                            );
                        }
                    }
                    if storage.record_commitment_production_halted() {
                        error!(
                            "[MEMCHAIN_BLOCK] Imported checkpoint evidence triggered a trusted security incident; commitment production remains halted"
                        );
                        break;
                    }
                }

                if round.attempted == 0 {
                    consecutive_unverified_rounds = 0;
                    debug!(
                        eligible_witnesses = round.eligible_witnesses,
                        "[MEMCHAIN_BLOCK] Coordinator witness round waiting for eligible peers"
                    );
                    continue;
                }
                if round.verified == 0 {
                    consecutive_unverified_rounds = consecutive_unverified_rounds.saturating_add(1);
                    if consecutive_unverified_rounds == 1
                        || consecutive_unverified_rounds.is_power_of_two()
                    {
                        warn!(
                            attempted = round.attempted,
                            failed = round.failed,
                            consecutive_unverified_rounds,
                            trigger,
                            "[MEMCHAIN_BLOCK] Coordinator witness round established no signed evidence"
                        );
                    }
                    continue;
                }

                consecutive_unverified_rounds = 0;
                if round.diverged > 0 || round.remote_ahead > 0 {
                    warn!(
                        attempted = round.attempted,
                        verified = round.verified,
                        converged = round.converged,
                        remote_ahead = round.remote_ahead,
                        remote_behind = round.remote_behind,
                        diverged = round.diverged,
                        failed = round.failed,
                        trigger,
                        "[MEMCHAIN_BLOCK] Coordinator witness round found signed chain attention evidence"
                    );
                } else {
                    debug!(
                        attempted = round.attempted,
                        verified = round.verified,
                        converged = round.converged,
                        remote_behind = round.remote_behind,
                        failed = round.failed,
                        trigger,
                        "[MEMCHAIN_BLOCK] Coordinator witness round complete"
                    );
                }
            }

            info!("[MEMCHAIN_BLOCK] Coordinator witness reconciliation stopped");
        }))
    }

    /// Starts the default-off Block Sync v1 follower.
    ///
    /// The configured coordinator identity is the initial trust root. When an
    /// immutable authority root is enabled, exact-next dual-signed handovers
    /// select subsequent coordinators at audited block boundaries. Discovery
    /// resolves only the active identity's signed endpoint, while pull helpers
    /// verify the response signer, every block proposer, and full chain
    /// continuity before SQLite is changed. [CERTIFIED-BLOCK-CARRIER
    /// 2026-07-29 by Codex] Availability-only failure may use a bounded
    /// operator-pinned witness as a read-only carrier, but terminal recovery
    /// requires the configured threshold certificate. There is no arbitrary
    /// discovery fallback, longest-chain selection, or carrier authority.
    pub(super) fn spawn_memchain_commitment_sync_task(
        &self,
        storage: Arc<MemoryStorage>,
        peer_store: Arc<PeerStore>,
        mut block_announce_rx: mpsc::Receiver<u64>,
        sync_http_client: Arc<reqwest::Client>,
    ) -> Result<Option<JoinHandle<()>>> {
        if !self.config.memchain.commitment_sync_enabled {
            return Ok(None);
        }
        let Some(coordinator_node_id) = self.config.memchain.commitment_sync_coordinator_node_id()
        else {
            let now = unix_now_secs();
            storage.record_commitment_sync_failure(
                now,
                "invalid_pinned_coordinator",
                1,
                now.saturating_add(600),
            );
            error!(
                "[MEMCHAIN_BLOCK] Follower sync disabled at runtime: invalid pinned coordinator"
            );
            return Err(ServerError::startup_failed(
                "MemChain commitment follower initialization failed: invalid pinned coordinator",
            ));
        };
        if self.identity.public_key_bytes() == coordinator_node_id {
            let now = unix_now_secs();
            storage.record_commitment_sync_failure(
                now,
                "coordinator_self_reference",
                1,
                now.saturating_add(600),
            );
            error!(
                "[MEMCHAIN_BLOCK] Follower sync disabled at runtime: coordinator cannot follow itself"
            );
            return Err(ServerError::startup_failed(
                "MemChain commitment follower initialization failed: coordinator cannot follow itself",
            ));
        }

        // [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Build one
        // fallible policy before spawning. Runtime must not independently
        // derive a label and pins or silently discard malformed identities.
        let authority_carrier_policy = self
            .config
            .memchain
            .effective_commitment_authority_carrier_policy()
            .map_err(|error| {
                let now = unix_now_secs();
                storage.record_commitment_sync_failure(
                    now,
                    "invalid_authority_carrier_policy",
                    1,
                    now.saturating_add(600),
                );
                error!(
                    "[MEMCHAIN_BLOCK] Follower sync startup rejected invalid authority carrier policy"
                );
                error
            })?;

        let identity = self.identity.clone();
        let base_interval_secs = self.config.memchain.commitment_sync_interval_secs;
        let max_pages_per_round = self.config.memchain.commitment_sync_max_pages_per_round;
        let certificate_witness_node_ids = self.config.memchain.commitment_witness_node_id_bytes();
        let authority_carrier_node_ids = authority_carrier_policy.node_ids().to_vec();
        let authority_carrier_policy_label = authority_carrier_policy.source_label();
        let certificate_minimum_signers = self.config.memchain.commitment_witness_min_verified;
        let authority_handover_enabled = storage.record_commitment_authority_enforced();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        Ok(Some(tokio::spawn(async move {
            const MAX_BACKOFF_SECS: u64 = 600;

            // [FOLLOWER-TASK-LIVENESS 2026-07-30 by Codex] Construct before
            // any await so panic, cancellation, or an unexpected loop exit
            // cannot leave the last successful readiness snapshot active.
            let liveness_guard = CommitmentSyncTaskLivenessGuard::new(Arc::clone(&storage));
            let mut consecutive_failures = 0u32;
            let mut consecutive_certificate_deferrals = 0u32;
            let mut next_delay = Duration::from_secs(0);
            let mut announcement_channel_open = true;
            // [BLOCK-CARRIER-CIRCUIT-BREAKER 2026-07-29 by Codex] This
            // process-only fixed-slot state survives follower rounds but is
            // discarded on restart. It contains no identity, endpoint, error,
            // payload, route, or wall-clock data.
            let mut block_carrier_circuit = CommitmentBlockCarrierCircuitBreaker::default();
            // [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Handover
            // evidence uses independent process-only circuit state. A stale
            // proof carrier cannot cool block or certificate transport.
            let mut authority_carrier_circuit = CommitmentAuthorityCarrierCircuitBreaker::default();
            // [CERTIFICATE-CARRIER-CIRCUIT 2026-07-29 by Codex] Certificate
            // transport has an independent typed circuit. A block-page outage
            // cannot suppress certificate evidence recovery, or vice versa.
            let mut certificate_carrier_circuit =
                CommitmentCertificateCarrierCircuitBreaker::default();
            info!(
                interval_secs = base_interval_secs,
                max_pages_per_round,
                event_driven = true,
                authority_handover_enabled,
                authority_carrier_policy = authority_carrier_policy_label,
                authority_carrier_pins = authority_carrier_node_ids.len(),
                "[MEMCHAIN_BLOCK] Authority-scheduled coordinator follower started"
            );

            'sync_loop: loop {
                let mut trigger = "scheduled";
                if !next_delay.is_zero() {
                    let deadline = tokio::time::Instant::now() + next_delay;
                    loop {
                        tokio::select! {
                            _ = shutdown_rx.recv() => break 'sync_loop,
                            _ = tokio::time::sleep_until(deadline) => break,
                            announcement = block_announce_rx.recv(), if announcement_channel_open => {
                                match announcement {
                                    Some(announced_height) if consecutive_failures == 0 => {
                                        trigger = "block_announce";
                                        debug!(
                                            announced_height,
                                            "[MEMCHAIN_BLOCK] Pinned coordinator announcement woke follower"
                                        );
                                        break;
                                    }
                                    Some(announced_height) => {
                                        debug!(
                                            announced_height,
                                            consecutive_failures,
                                            "[MEMCHAIN_BLOCK] Announcement retained failure backoff"
                                        );
                                    }
                                    None => announcement_channel_open = false,
                                }
                            }
                        }
                        if tokio::time::Instant::now() >= deadline || trigger == "block_announce" {
                            break;
                        }
                    }
                }

                let attempt_at = unix_now_secs();
                if trigger == "block_announce" {
                    storage.record_commitment_sync_announcement_attempt(attempt_at);
                } else {
                    storage.record_commitment_sync_attempt(attempt_at);
                }
                let round_future = async {
                    let mut inserted = 0usize;
                    let mut remote_tip_height = 0u64;
                    // [MULTIPAGE-BLOCK-CARRIER-HANDOFF 2026-07-29 by Codex]
                    // Preference is scoped to this bounded page round. It is
                    // discarded before backoff/retry and never enters status,
                    // persistence, routing policy, or trust decisions.
                    let mut block_carrier_cursor = CommitmentBlockCarrierCursor::default();
                    let mut authority_carrier_cursor = CommitmentAuthorityCarrierCursor::default();
                    for _ in 0..max_pages_per_round {
                        // [AUTHORITY-HANDOVER-FOLLOWER 2026-08-14 by Codex]
                        // Pull exactly one proof before each block page. A
                        // future proof only caps this page at activation - 1;
                        // after that prefix is audited, the next iteration
                        // persists the proof and resolves the new coordinator.
                        let (active_coordinator, max_blocks) = if authority_handover_enabled {
                            let authority =
                                sync_next_record_coordinator_handover_with_carrier_runtime(
                                    &storage,
                                    &peer_store,
                                    &identity,
                                    &authority_carrier_node_ids,
                                    sync_http_client.as_ref(),
                                    &mut authority_carrier_cursor,
                                    &mut authority_carrier_circuit,
                                )
                                .await?;
                            if authority.source == CommitmentAuthoritySyncSource::PinnedCarrier {
                                debug!(
                                    carrier_attempts = authority.carrier_attempts,
                                    "[MEMCHAIN_BLOCK] Recovered dual-signed authority proof through pinned transport"
                                );
                            }
                            let max_blocks = authority
                                .pending_activation_height
                                .map(|activation_height| {
                                    activation_height
                                        .saturating_sub(authority.next_block_height)
                                        .min(u64::from(MAX_BLOCKS_PER_RESPONSE_WIRE))
                                        as u16
                                })
                                .unwrap_or(MAX_BLOCKS_PER_RESPONSE_WIRE);
                            if max_blocks == 0 {
                                return Err("handover_activation_boundary_invalid".to_string());
                            }
                            (authority.active_coordinator, max_blocks)
                        } else {
                            (coordinator_node_id, MAX_BLOCKS_PER_RESPONSE_WIRE)
                        };
                        let page_pull = pull_record_commitment_page_with_carrier_runtime_bounded(
                            &storage,
                            &peer_store,
                            &identity,
                            &active_coordinator,
                            &certificate_witness_node_ids,
                            certificate_minimum_signers,
                            sync_http_client.as_ref(),
                            &mut block_carrier_cursor,
                            &mut block_carrier_circuit,
                            max_blocks,
                        )
                        .await?;
                        let page_source = page_pull.source;
                        let outcome = page_pull.page;
                        let verified_blocks = outcome
                            .inserted
                            .saturating_add(outcome.already_present)
                            .try_into()
                            .unwrap_or(u64::MAX);
                        storage.record_commitment_sync_page_success(
                            unix_now_secs(),
                            verified_blocks,
                            outcome.remote_tip_height,
                            outcome.has_more,
                        );
                        inserted = inserted.saturating_add(outcome.inserted);
                        remote_tip_height = outcome.remote_tip_height;
                        if !outcome.has_more {
                            if page_source == CommitmentSyncPageSource::PinnedCarrier {
                                // [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex]
                                // A terminal carrier page proves only an
                                // authenticated producer-signed prefix. It may
                                // become operationally recovered only when the
                                // exact local tip satisfies the configured
                                // immutable witness certificate threshold.
                                match sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime(
                                    &storage,
                                    &peer_store,
                                    &identity,
                                    &active_coordinator,
                                    &certificate_witness_node_ids,
                                    certificate_minimum_signers,
                                    outcome.remote_tip_height,
                                    sync_http_client.as_ref(),
                                    &mut certificate_carrier_circuit,
                                )
                                .await
                                {
                                    Ok(
                                        CommitmentFollowerCertificateSyncOutcome::AlreadyCurrent,
                                    ) => {}
                                    Ok(
                                        CommitmentFollowerCertificateSyncOutcome::Refreshed(
                                            certificate,
                                        ),
                                    ) if certificate.persisted => {
                                        info!(
                                            checkpoint_height = certificate.checkpoint_height,
                                            signer_count = certificate.signer_count,
                                            required_signers = certificate.required_signers,
                                            carrier_attempts = page_pull.carrier_attempts,
                                            "[MEMCHAIN_BLOCK] Follower completed certified carrier recovery"
                                        );
                                    }
                                    Ok(
                                        CommitmentFollowerCertificateSyncOutcome::PolicyDisabled,
                                    ) => {
                                        return Err(
                                            "block_carrier_certificate_required".to_string()
                                        );
                                    }
                                    Ok(
                                        CommitmentFollowerCertificateSyncOutcome::Refreshed(_),
                                    ) => {
                                        return Err(
                                            "block_carrier_certificate_policy_unsatisfied"
                                                .to_string(),
                                        );
                                    }
                                    Err(reason) => return Err(reason),
                                }
                                storage.record_commitment_sync_certified_recovery_success(
                                    unix_now_secs(),
                                    outcome.remote_tip_height,
                                );
                                return Ok(CommitmentFollowerRoundOutcome {
                                    inserted,
                                    remote_tip_height: outcome.remote_tip_height,
                                    block_backlog_remaining: false,
                                    certificate_retry_pending: false,
                                    certified_recovered: true,
                                });
                            }
                            let checkpoint = match pull_record_commitment_checkpoint(
                                &storage,
                                &peer_store,
                                &identity,
                                &active_coordinator,
                                sync_http_client.as_ref(),
                            )
                            .await
                            {
                                Ok(checkpoint) => checkpoint,
                                Err(reason) => {
                                    storage.record_commitment_checkpoint_failure(unix_now_secs());
                                    return Err(reason);
                                }
                            };
                            let checked_at = unix_now_secs();
                            storage.record_commitment_checkpoint_verified(
                                checked_at,
                                checkpoint.relation.as_str(),
                                checkpoint.local_tip_height,
                                checkpoint.remote_tip_height,
                            );
                            match checkpoint.relation {
                                CommitmentCheckpointRelation::Converged => {
                                    storage.record_commitment_sync_checkpoint_success(
                                        checked_at,
                                        checkpoint.remote_tip_height,
                                    );
                                    let mut certificate_retry_pending = false;
                                    // [FOLLOWER-CERTIFICATE-CARRIER 2026-07-29 by Codex]
                                    // Certificate availability is additive.
                                    // The coordinator is tried first; only a
                                    // transport/compatibility outage permits
                                    // fallback to an exact configured witness.
                                    // Verification failures stop immediately,
                                    // and no result can erase convergence.
                                    match sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime(
                                        &storage,
                                        &peer_store,
                                        &identity,
                                        &active_coordinator,
                                        &certificate_witness_node_ids,
                                        certificate_minimum_signers,
                                        checkpoint.remote_tip_height,
                                        sync_http_client.as_ref(),
                                        &mut certificate_carrier_circuit,
                                    )
                                    .await
                                    {
                                        Ok(
                                            CommitmentFollowerCertificateSyncOutcome::Refreshed(
                                                outcome,
                                            ),
                                        ) if outcome.persisted => {
                                            info!(
                                                checkpoint_height = outcome.checkpoint_height,
                                                signer_count = outcome.signer_count,
                                                required_signers = outcome.required_signers,
                                                "[MEMCHAIN_BLOCK] Follower imported audited checkpoint certificate"
                                            );
                                        }
                                        Ok(
                                            CommitmentFollowerCertificateSyncOutcome::Refreshed(
                                                outcome,
                                            ),
                                        ) => {
                                            certificate_retry_pending = true;
                                            warn!(
                                                checkpoint_height = outcome.checkpoint_height,
                                                signer_count = outcome.signer_count,
                                                required_signers = outcome.required_signers,
                                                "[MEMCHAIN_BLOCK] Follower certificate did not satisfy durable local policy"
                                            );
                                        }
                                        Ok(
                                            CommitmentFollowerCertificateSyncOutcome::PolicyDisabled
                                            | CommitmentFollowerCertificateSyncOutcome::AlreadyCurrent,
                                        ) => {}
                                        Err(reason) => {
                                            debug!(
                                                reason = %reason,
                                                "[MEMCHAIN_BLOCK] Follower certificate refresh deferred"
                                            );
                                        }
                                    }
                                    return Ok(CommitmentFollowerRoundOutcome {
                                        inserted,
                                        remote_tip_height: checkpoint.remote_tip_height,
                                        block_backlog_remaining: false,
                                        certificate_retry_pending,
                                        certified_recovered: false,
                                    });
                                }
                                CommitmentCheckpointRelation::RemoteAhead => {
                                    remote_tip_height = checkpoint.remote_tip_height;
                                    continue;
                                }
                                CommitmentCheckpointRelation::RemoteBehind => {
                                    return Err("signed_checkpoint_remote_behind".to_string());
                                }
                                CommitmentCheckpointRelation::Diverged => {
                                    return Err("signed_checkpoint_divergence".to_string());
                                }
                            }
                        }
                    }
                    Ok(CommitmentFollowerRoundOutcome {
                        inserted,
                        remote_tip_height,
                        block_backlog_remaining: true,
                        certificate_retry_pending: false,
                        certified_recovered: false,
                    })
                };
                let round: std::result::Result<CommitmentFollowerRoundOutcome, String> = tokio::select! {
                    _ = shutdown_rx.recv() => break 'sync_loop,
                    result = round_future => result,
                };

                match round {
                    Ok(outcome) => {
                        consecutive_failures = 0;
                        if outcome.certificate_retry_pending {
                            consecutive_certificate_deferrals =
                                consecutive_certificate_deferrals.saturating_add(1);
                        } else {
                            consecutive_certificate_deferrals = 0;
                        }
                        if outcome.inserted > 0 {
                            info!(
                                blocks = outcome.inserted,
                                tip_height = outcome.remote_tip_height,
                                block_backlog_remaining = outcome.block_backlog_remaining,
                                certificate_retry_pending = outcome.certificate_retry_pending,
                                certified_recovered = outcome.certified_recovered,
                                "[MEMCHAIN_BLOCK] Follower catch-up advanced"
                            );
                        } else {
                            debug!(
                                tip_height = outcome.remote_tip_height,
                                trigger,
                                certificate_retry_pending = outcome.certificate_retry_pending,
                                certified_recovered = outcome.certified_recovered,
                                "[MEMCHAIN_BLOCK] Follower completed verified sync round"
                            );
                        }
                        // Keep each round bounded but drain a verified backlog
                        // promptly. Deferred certificate durability uses an
                        // independent bounded retry schedule and never changes
                        // the block transport failure streak.
                        let retry_secs = commitment_follower_success_retry_delay(
                            base_interval_secs,
                            &outcome,
                            consecutive_certificate_deferrals,
                        );
                        next_delay = Duration::from_secs(retry_secs);
                        storage.schedule_next_commitment_sync_poll(
                            unix_now_secs().saturating_add(next_delay.as_secs()),
                        );
                    }
                    Err(reason) => {
                        consecutive_failures = consecutive_failures.saturating_add(1);
                        let shift = consecutive_failures.saturating_sub(1).min(5);
                        let multiplier = 1u64 << shift;
                        let retry_secs = base_interval_secs
                            .saturating_mul(multiplier)
                            .min(MAX_BACKOFF_SECS);
                        next_delay = Duration::from_secs(retry_secs);
                        let failed_at = unix_now_secs();
                        storage.record_commitment_sync_failure(
                            failed_at,
                            &reason,
                            consecutive_failures,
                            failed_at.saturating_add(retry_secs),
                        );
                        if consecutive_failures == 1 || consecutive_failures.is_power_of_two() {
                            warn!(
                                consecutive_failures,
                                retry_secs,
                                reason = %reason,
                                "[MEMCHAIN_BLOCK] Follower catch-up failed closed"
                            );
                        }
                    }
                }
            }
            drop(liveness_guard);
            info!("[MEMCHAIN_BLOCK] Pinned coordinator follower stopped");
        })))
    }
}
