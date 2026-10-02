// [ARCH-SPLIT 2026-10-02]
// Checkpoint certificate pull, recovery, follower sync, and HTTP handler.
// Bodies are unchanged. The parent re-exports every name at its original visibility.
use super::*;

/// Pulls and imports one current-tip certificate from an admitted peer.
///
/// The serving peer is transport only. Every historical member must verify as
/// an exact signed checkpoint frame from a distinct identity in
/// `allowed_witnesses`. `minimum_required_signers` is the receiver's current
/// operator policy and cannot be downgraded by the serving peer. Callers must
/// use this after startup; a replayed bundle never replaces the live startup
/// witness round.
///
/// # Errors
///
/// Returns a stable privacy-safe code when peer admission, transport, outer
/// freshness, member signatures, operator pinning, digest, or durable storage
/// verification fails.
pub async fn pull_record_commitment_checkpoint_certificate(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
) -> Result<CommitmentCertificateImportOutcome, String> {
    pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        client,
        &commitment_peer_endpoint_is_public,
    )
    .await
}

pub(super) fn normalized_commitment_certificate_carriers(
    local_node_id: &[u8; 32],
    excluded_primary: Option<&[u8; 32]>,
    allowed_witnesses: &[[u8; 32]],
    max_carriers: usize,
) -> Vec<[u8; 32]> {
    let carrier_limit = max_carriers.min(MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1);
    let mut carriers = Vec::with_capacity(carrier_limit);
    for witness in allowed_witnesses
        .iter()
        .take(MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1)
    {
        if carriers.len() >= carrier_limit {
            break;
        }
        if witness == local_node_id
            || excluded_primary.is_some_and(|excluded| witness == excluded)
            || carriers.contains(witness)
        {
            continue;
        }
        carriers.push(*witness);
    }
    carriers
}

/// Runs one bounded, fail-closed certificate carrier sequence.
///
/// Availability is the only failure class that may advance to another exact
/// operator pin. Any verified response, including one made non-durable by a
/// concurrent local state change, stops the round. Security failures retain
/// their private stable code only long enough for the follower adapter to
/// preserve its existing API; the coordinator adapter collapses them before
/// returning to runtime logging.
///
/// [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex]
#[allow(clippy::too_many_arguments)]
pub(super) async fn pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    carriers: &[[u8; 32]],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> CommitmentCertificateCarrierPullRound
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    circuit_breaker.align_slots(carriers.len());
    let mut carrier_attempts = 0usize;
    let mut cooldown_skips = 0usize;
    let mut half_open_attempts = 0usize;

    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&minimum_required_signers)
        || allowed_witnesses.len() < minimum_required_signers
    {
        return CommitmentCertificateCarrierPullRound {
            terminal: CommitmentCertificateCarrierPullTerminal::SecurityStopped(
                "certificate_policy_invalid".to_string(),
            ),
            carrier_attempts,
            cooldown_skips,
            half_open_attempts,
        };
    }

    for (carrier_index, candidate) in carriers.iter().enumerate() {
        match circuit_breaker.decision(carrier_index, Instant::now()) {
            CommitmentCarrierCircuitDecision::Closed => {}
            CommitmentCarrierCircuitDecision::Cooling => {
                cooldown_skips = cooldown_skips.saturating_add(1);
                continue;
            }
            CommitmentCarrierCircuitDecision::HalfOpen => {
                half_open_attempts = half_open_attempts.saturating_add(1);
            }
        }
        carrier_attempts = carrier_attempts.saturating_add(1);

        match pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            candidate,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
        )
        .await
        {
            Ok(imported) => {
                circuit_breaker.record_success(carrier_index);
                return CommitmentCertificateCarrierPullRound {
                    terminal: CommitmentCertificateCarrierPullTerminal::Imported(imported),
                    carrier_attempts,
                    cooldown_skips,
                    half_open_attempts,
                };
            }
            Err(error)
                if commitment_certificate_source_failure_class(&error)
                    == CommitmentCertificateSourceFailureClass::Availability =>
            {
                circuit_breaker.record_availability_failure(carrier_index, Instant::now());
            }
            Err(error) => {
                return CommitmentCertificateCarrierPullRound {
                    terminal: CommitmentCertificateCarrierPullTerminal::SecurityStopped(error),
                    carrier_attempts,
                    cooldown_skips,
                    half_open_attempts,
                };
            }
        }
    }

    CommitmentCertificateCarrierPullRound {
        terminal: CommitmentCertificateCarrierPullTerminal::AvailabilityExhausted,
        carrier_attempts,
        cooldown_skips,
        half_open_attempts,
    }
}

/// Recovers post-startup certificate evidence from exact operator pins.
///
/// This coordinator-side path never grants startup authority, selects a chain,
/// or treats peer count as consensus. It returns only source-blind aggregates
/// suitable for runtime logs.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    max_carriers: usize,
    client: &reqwest::Client,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> CommitmentCertificateCarrierRecoveryRound {
    recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        allowed_witnesses,
        minimum_required_signers,
        max_carriers,
        client,
        &commitment_peer_endpoint_is_public,
        circuit_breaker,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn recover_record_commitment_checkpoint_certificate_from_pinned_carriers_with_runtime_and_endpoint_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    max_carriers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> CommitmentCertificateCarrierRecoveryRound
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let carriers = normalized_commitment_certificate_carriers(
        &identity.public_key_bytes(),
        None,
        allowed_witnesses,
        max_carriers,
    );
    let pull_round =
        pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &carriers,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
            circuit_breaker,
        )
        .await;
    let (disposition, checkpoint_height, signer_count, required_signers) = match pull_round.terminal
    {
        CommitmentCertificateCarrierPullTerminal::Imported(imported) if imported.persisted => (
            CommitmentCertificateCarrierRecoveryDisposition::Persisted,
            imported.checkpoint_height,
            imported.signer_count,
            imported.required_signers,
        ),
        CommitmentCertificateCarrierPullTerminal::Imported(imported) => (
            CommitmentCertificateCarrierRecoveryDisposition::VerifiedUnpersisted,
            imported.checkpoint_height,
            imported.signer_count,
            imported.required_signers,
        ),
        CommitmentCertificateCarrierPullTerminal::AvailabilityExhausted => (
            CommitmentCertificateCarrierRecoveryDisposition::AvailabilityExhausted,
            0,
            0,
            0,
        ),
        CommitmentCertificateCarrierPullTerminal::SecurityStopped(_) => (
            CommitmentCertificateCarrierRecoveryDisposition::SecurityStopped,
            0,
            0,
            0,
        ),
    };

    CommitmentCertificateCarrierRecoveryRound {
        disposition,
        checkpoint_height,
        signer_count,
        required_signers,
        carrier_attempts: pull_round.carrier_attempts,
        cooldown_skips: pull_round.cooldown_skips,
        half_open_attempts: pull_round.half_open_attempts,
        cooling_slots: circuit_breaker.cooling_slots(Instant::now()),
    }
}

/// Refreshes follower certificate evidence after signed tip convergence.
///
/// The follower's configured coordinator is transport only. The response must
/// still satisfy the receiver's witness allowlist and minimum threshold. A
/// threshold below two disables certificate replication for backward
/// compatibility; malformed enabled policy fails closed. If the coordinator is
/// unavailable, exact operator-pinned witnesses may carry the same immutable
/// certificate. They gain no authority over chain state or certificate policy.
///
/// # Errors
///
/// Returns a stable privacy-safe code when local policy, peer transport,
/// certificate verification, or durable storage validation fails.
pub async fn sync_follower_record_commitment_checkpoint_certificate(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String> {
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        converged_tip_height,
        client,
        &mut circuit_breaker,
    )
    .await
}

/// Refreshes follower certificate evidence with process-lifetime carrier state.
///
/// The caller must retain this circuit only for this exact follower policy
/// domain. The typed marker prevents block-page circuit state from being
/// passed here accidentally.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String> {
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        converged_tip_height,
        client,
        &commitment_peer_endpoint_is_public,
        circuit_breaker,
    )
    .await
}

#[cfg(test)]
pub(super) async fn sync_follower_record_commitment_checkpoint_certificate_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    let mut circuit_breaker = CommitmentCertificateCarrierCircuitBreaker::default();
    sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy(
        storage,
        peer_store,
        identity,
        source_node_id,
        allowed_witnesses,
        minimum_required_signers,
        converged_tip_height,
        client,
        endpoint_allowed,
        &mut circuit_breaker,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn sync_follower_record_commitment_checkpoint_certificate_with_carrier_runtime_and_endpoint_policy<
    F,
>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    converged_tip_height: u64,
    client: &reqwest::Client,
    endpoint_allowed: &F,
    circuit_breaker: &mut CommitmentCertificateCarrierCircuitBreaker,
) -> Result<CommitmentFollowerCertificateSyncOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if minimum_required_signers < 2 {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::Disabled,
        );
        return Ok(CommitmentFollowerCertificateSyncOutcome::PolicyDisabled);
    }
    if minimum_required_signers > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
        || allowed_witnesses.len() < minimum_required_signers
    {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::ConfigurationError,
        );
        return Err("certificate_policy_invalid".to_string());
    }
    if converged_tip_height == 0 {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::WaitingForConvergence,
        );
        return Err("certificate_local_tip_unavailable".to_string());
    }
    let (_, _, local_tip_height, _) = match storage
        .record_commitment_chain_checkpoint(converged_tip_height)
        .await
    {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
            );
            return Err(error);
        }
    };
    if local_tip_height != converged_tip_height {
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::WaitingForConvergence,
        );
        return Err("certificate_converged_tip_changed".to_string());
    }

    let local_node_id = identity.public_key_bytes();
    let carriers = normalized_commitment_certificate_carriers(
        &local_node_id,
        Some(source_node_id),
        allowed_witnesses,
        MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1,
    );
    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Align before the
    // already-current check so a pin-count change cannot retain positional
    // circuit state even when no transport request is needed.
    circuit_breaker.align_slots(carriers.len());

    let certificate_already_current = match storage
        .record_commitment_checkpoint_certificate_satisfies_policy(
            converged_tip_height,
            allowed_witnesses,
            minimum_required_signers,
        )
        .await
    {
        Ok(current) => current,
        Err(error) => {
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
            );
            return Err(error);
        }
    };
    if certificate_already_current {
        record_commitment_certificate_carrier_circuit_telemetry(storage, circuit_breaker, 0, 0);
        storage.record_commitment_certificate_policy_evaluation(
            now_secs(),
            RecordCommitmentCertificatePolicyReadiness::Ready {
                tip_height: converged_tip_height,
            },
        );
        return Ok(CommitmentFollowerCertificateSyncOutcome::AlreadyCurrent);
    }

    // [CERTIFICATE-CARRIER-RECOVERY 2026-07-29 by Codex] Coordinator transport
    // remains first. Only its narrow availability class may enter the shared
    // carrier primitive; every other coordinator error stops immediately.
    let coordinator_availability_failure =
        match pull_record_commitment_checkpoint_certificate_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            source_node_id,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
        )
        .await
        {
            Ok(imported) => {
                if imported.checkpoint_height != converged_tip_height {
                    record_commitment_certificate_carrier_circuit_telemetry(
                        storage,
                        circuit_breaker,
                        0,
                        0,
                    );
                    storage.record_commitment_certificate_sync_outcome(
                        now_secs(),
                        RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                        0,
                    );
                    storage.record_commitment_certificate_policy_evaluation(
                        now_secs(),
                        RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
                    );
                    return Err("certificate_converged_tip_changed".to_string());
                }
                record_commitment_certificate_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    0,
                    0,
                );
                storage.record_commitment_certificate_sync_outcome(
                    now_secs(),
                    follower_certificate_sync_disposition(
                        CommitmentFollowerCertificateSource::Coordinator,
                        imported.persisted,
                    ),
                    0,
                );
                let readiness = if imported.persisted {
                    RecordCommitmentCertificatePolicyReadiness::Ready {
                        tip_height: converged_tip_height,
                    }
                } else {
                    RecordCommitmentCertificatePolicyReadiness::WaitingForCertificate {
                        tip_height: converged_tip_height,
                    }
                };
                storage.record_commitment_certificate_policy_evaluation(now_secs(), readiness);
                return Ok(CommitmentFollowerCertificateSyncOutcome::Refreshed(
                    imported,
                ));
            }
            Err(error)
                if commitment_certificate_source_failure_class(&error)
                    == CommitmentCertificateSourceFailureClass::Availability =>
            {
                error
            }
            Err(error) => {
                record_commitment_certificate_carrier_circuit_telemetry(
                    storage,
                    circuit_breaker,
                    0,
                    0,
                );
                storage.record_commitment_certificate_sync_outcome(
                    now_secs(),
                    RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                    0,
                );
                storage.record_commitment_certificate_policy_evaluation(
                    now_secs(),
                    RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
                );
                return Err(error);
            }
        };

    let carrier_round =
        pull_record_commitment_checkpoint_certificate_from_carriers_with_endpoint_policy(
            storage,
            peer_store,
            identity,
            &carriers,
            allowed_witnesses,
            minimum_required_signers,
            client,
            endpoint_allowed,
            circuit_breaker,
        )
        .await;
    record_commitment_certificate_carrier_circuit_telemetry(
        storage,
        circuit_breaker,
        carrier_round.cooldown_skips,
        carrier_round.half_open_attempts,
    );
    match carrier_round.terminal {
        CommitmentCertificateCarrierPullTerminal::Imported(imported) => {
            if imported.checkpoint_height != converged_tip_height {
                storage.record_commitment_certificate_sync_outcome(
                    now_secs(),
                    RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                    carrier_round.carrier_attempts,
                );
                storage.record_commitment_certificate_policy_evaluation(
                    now_secs(),
                    RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
                );
                return Err("certificate_converged_tip_changed".to_string());
            }
            storage.record_commitment_certificate_sync_outcome(
                now_secs(),
                follower_certificate_sync_disposition(
                    CommitmentFollowerCertificateSource::PinnedCarrier,
                    imported.persisted,
                ),
                carrier_round.carrier_attempts,
            );
            let readiness = if imported.persisted {
                RecordCommitmentCertificatePolicyReadiness::Ready {
                    tip_height: converged_tip_height,
                }
            } else {
                RecordCommitmentCertificatePolicyReadiness::WaitingForCertificate {
                    tip_height: converged_tip_height,
                }
            };
            storage.record_commitment_certificate_policy_evaluation(now_secs(), readiness);
            Ok(CommitmentFollowerCertificateSyncOutcome::Refreshed(
                imported,
            ))
        }
        CommitmentCertificateCarrierPullTerminal::AvailabilityExhausted => {
            storage.record_commitment_certificate_sync_outcome(
                now_secs(),
                RecordCommitmentCertificateSyncDisposition::AvailabilityExhausted,
                carrier_round.carrier_attempts,
            );
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SourceUnavailable {
                    tip_height: converged_tip_height,
                },
            );
            Err(coordinator_availability_failure)
        }
        CommitmentCertificateCarrierPullTerminal::SecurityStopped(error) => {
            storage.record_commitment_certificate_sync_outcome(
                now_secs(),
                RecordCommitmentCertificateSyncDisposition::SecurityStopped,
                carrier_round.carrier_attempts,
            );
            storage.record_commitment_certificate_policy_evaluation(
                now_secs(),
                RecordCommitmentCertificatePolicyReadiness::SecurityStopped,
            );
            Err(error)
        }
    }
}

pub(super) fn commitment_certificate_source_failure_class(
    error: &str,
) -> CommitmentCertificateSourceFailureClass {
    let retryable_status = error
        .strip_prefix("certificate_http_status_")
        .and_then(|status| status.parse::<u16>().ok())
        .is_some_and(|status| matches!(status, 403 | 404 | 408 | 429 | 500 | 502 | 503 | 504));
    if retryable_status
        || matches!(
            error,
            "certificate_source_unavailable"
                | "certificate_source_missing_endpoint"
                | "certificate_request_timeout"
                | "certificate_request_connect"
                | "response_body_timeout"
                | "response_body_connect"
                | "response_body_body"
        )
    {
        CommitmentCertificateSourceFailureClass::Availability
    } else {
        CommitmentCertificateSourceFailureClass::Security
    }
}

pub(super) async fn pull_record_commitment_checkpoint_certificate_with_endpoint_policy<F>(
    storage: &MemoryStorage,
    peer_store: &PeerStore,
    identity: &IdentityKeyPair,
    source_node_id: &[u8; 32],
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    client: &reqwest::Client,
    endpoint_allowed: &F,
) -> Result<CommitmentCertificateImportOutcome, String>
where
    F: Fn(&str) -> bool + Send + Sync + ?Sized,
{
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&minimum_required_signers)
        || allowed_witnesses.len() < minimum_required_signers
    {
        return Err("certificate_policy_invalid".to_string());
    }
    let request_timestamp = now_secs();
    let source = peer_store
        .get_valid(source_node_id, request_timestamp)
        .ok_or_else(|| "certificate_source_unavailable".to_string())?;
    let endpoint = source
        .descriptor
        .public_endpoint
        .as_deref()
        .ok_or_else(|| "certificate_source_missing_endpoint".to_string())?;
    if !endpoint_allowed(endpoint) {
        return Err("certificate_source_unsafe_endpoint".to_string());
    }
    let url = commitment_checkpoint_certificate_url(endpoint)?;
    let (known_tip_height, known_tip_hash) = verified_local_commitment_tip(storage).await?;
    if known_tip_height == 0 {
        return Err("certificate_local_tip_unavailable".to_string());
    }

    let mut request_id = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut request_id);
    let requester = identity.public_key_bytes();
    let signing_bytes = record_checkpoint_certificate_request_signing_bytes(
        &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    let request = MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature: identity.sign(&signing_bytes),
    };
    let frame = encode_memchain(&request).map_err(|_| "request_encode_failed".to_string())?;
    let response = client
        .post(url)
        .header("content-type", "application/octet-stream")
        .body(frame)
        .send()
        .await
        .map_err(|error| classify_http_error("certificate_request", &error))?;
    if !response.status().is_success() {
        return Err(format!(
            "certificate_http_status_{}",
            response.status().as_u16()
        ));
    }
    let body = read_bounded_response(response).await?;
    let verified = verify_checkpoint_certificate_response(
        &body,
        &request_id,
        source_node_id,
        (known_tip_height, known_tip_hash),
        allowed_witnesses,
        minimum_required_signers,
        now_secs(),
    )?;

    let mut evidence_digests = Vec::with_capacity(verified.members.len());
    for member in &verified.members {
        let relation = if member.remote_tip_height == verified.checkpoint_height {
            "converged"
        } else {
            "remote_ahead"
        };
        let persist_outcome = storage
            .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                member.observed_at,
                relation,
                verified.checkpoint_height,
                member.remote_tip_height,
                verified.checkpoint_height,
                &member.evidence_digest,
                &member.frame,
                true,
            )
            .await
            .map_err(|_| "certificate_member_persist_failed".to_string())?;
        if persist_outcome != RecordCommitmentCheckpointEvidencePersistOutcome::Stored {
            return Err("certificate_member_security_incident".to_string());
        }
        evidence_digests.push(member.evidence_digest);
    }
    let persisted = storage
        .persist_record_commitment_checkpoint_certificate(
            now_secs(),
            verified.required_signers,
            allowed_witnesses,
            &evidence_digests,
        )
        .await
        .map_err(|_| "certificate_persist_failed".to_string())?;
    Ok(CommitmentCertificateImportOutcome {
        checkpoint_height: verified.checkpoint_height,
        signer_count: verified.members.len(),
        required_signers: verified.required_signers,
        persisted,
    })
}

#[allow(clippy::too_many_arguments)]
/// Verifies one request-bound portable custody witness receipt.
///
/// [CUSTODY-WITNESS-NETWORK 2026-08-16 by Codex] Negative receipts are valid
/// evidence, so this verifier authenticates their exact relation rather than
/// collapsing them into transport errors. The outer signature binds the
/// portable receipt to one request; the nested signature remains independently
/// verifiable by later auditors.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_checkpoint_certificate_response(
    body: &[u8],
    expected_request_id: &[u8; 16],
    expected_responder: &[u8; 32],
    local_tip: (u64, [u8; 32]),
    allowed_witnesses: &[[u8; 32]],
    minimum_required_signers: usize,
    now: u64,
) -> Result<VerifiedCheckpointCertificate, String> {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("invalid_certificate_frame".to_string());
    }
    let response = decode_memchain(&body[1..]).map_err(|_| "invalid_certificate_frame")?;
    let canonical = encode_memchain(&response).map_err(|_| "invalid_certificate_frame")?;
    if canonical != body {
        return Err("noncanonical_certificate_frame".to_string());
    }
    let MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        certificate_digest,
        required_signers,
        members,
        signature,
    } = response
    else {
        return Err("unexpected_certificate_message".to_string());
    };
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
        return Err("certificate_chain_mismatch".to_string());
    }
    if request_id != *expected_request_id {
        return Err("certificate_request_mismatch".to_string());
    }
    if responder != *expected_responder {
        return Err("certificate_responder_mismatch".to_string());
    }
    if now.abs_diff(response_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return Err("stale_certificate_response".to_string());
    }
    if checkpoint_height == 0 || checkpoint_height != local_tip.0 || checkpoint_hash != local_tip.1
    {
        return Err("certificate_local_tip_mismatch".to_string());
    }
    let signer_count = members.iter().flatten().count();
    let required_signers = usize::from(required_signers);
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&required_signers)
        || signer_count < required_signers
        || signer_count > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
    {
        return Err("certificate_threshold_invalid".to_string());
    }
    if required_signers < minimum_required_signers {
        return Err("certificate_threshold_below_policy".to_string());
    }
    let signing_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        checkpoint_height,
        &checkpoint_hash,
        &certificate_digest,
        required_signers as u8,
        signer_count as u8,
    );
    IdentityPublicKey::from_bytes(&responder)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .map_err(|_| "invalid_certificate_response_signature".to_string())?;

    let mut saw_empty_slot = false;
    let mut previous_responder = None;
    let mut digest_members = Vec::with_capacity(signer_count);
    let mut verified_members = Vec::with_capacity(signer_count);
    for slot in members {
        let Some(member) = slot else {
            saw_empty_slot = true;
            continue;
        };
        if saw_empty_slot {
            return Err("certificate_members_not_packed".to_string());
        }
        if previous_responder.is_some_and(|previous| previous >= member.responder) {
            return Err("certificate_members_not_distinct_sorted".to_string());
        }
        previous_responder = Some(member.responder);
        if !allowed_witnesses.contains(&member.responder) {
            return Err("certificate_member_not_pinned".to_string());
        }
        if member.response_timestamp > now.saturating_add(REQUEST_TIMESTAMP_SKEW_SECS) {
            return Err("certificate_member_timestamp_invalid".to_string());
        }
        if member.checkpoint_height != checkpoint_height
            || member.checkpoint_hash != checkpoint_hash
            || member.tip_height < checkpoint_height
            || (member.tip_height == checkpoint_height && member.tip_hash != checkpoint_hash)
        {
            return Err("certificate_member_claim_invalid".to_string());
        }
        let member_signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &chain_id,
            &member.request_id,
            &member.responder,
            member.response_timestamp,
            member.checkpoint_height,
            &member.checkpoint_hash,
            member.tip_height,
            &member.tip_hash,
        );
        IdentityPublicKey::from_bytes(&member.responder)
            .and_then(|key| key.verify(&member_signing_bytes, &member.signature))
            .map_err(|_| "invalid_certificate_member_signature".to_string())?;
        let frame = encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id,
            request_id: member.request_id,
            responder: member.responder,
            response_timestamp: member.response_timestamp,
            checkpoint_height: member.checkpoint_height,
            checkpoint_hash: member.checkpoint_hash,
            tip_height: member.tip_height,
            tip_hash: member.tip_hash,
            signature: member.signature,
        })
        .map_err(|_| "certificate_member_encode_failed".to_string())?;
        let evidence_digest: [u8; 32] = Sha256::digest(&frame).into();
        digest_members.push((member.responder, evidence_digest));
        verified_members.push(VerifiedCertificateMember {
            observed_at: member.response_timestamp,
            remote_tip_height: member.tip_height,
            evidence_digest,
            frame,
        });
    }
    let computed_digest = record_checkpoint_certificate_digest_v1(
        &chain_id,
        checkpoint_height,
        &checkpoint_hash,
        required_signers,
        &digest_members,
    );
    if computed_digest != certificate_digest {
        return Err("certificate_digest_mismatch".to_string());
    }
    Ok(VerifiedCheckpointCertificate {
        checkpoint_height,
        required_signers,
        members: verified_members,
    })
}

pub(super) fn checkpoint_certificate_member_from_frame(
    frame: &[u8],
    expected_chain_id: &[u8; 32],
) -> Result<RecordCheckpointCertificateMemberV1, String> {
    if frame.first().copied() != Some(MEMCHAIN_MAGIC) {
        return Err("certificate_member_frame_invalid".to_string());
    }
    let message =
        decode_memchain(&frame[1..]).map_err(|_| "certificate_member_frame_invalid".to_string())?;
    let canonical =
        encode_memchain(&message).map_err(|_| "certificate_member_frame_invalid".to_string())?;
    if canonical != frame {
        return Err("certificate_member_frame_noncanonical".to_string());
    }
    let MemChainMessage::RecordChainCheckpointResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    } = message
    else {
        return Err("certificate_member_frame_unexpected".to_string());
    };
    if chain_id != *expected_chain_id {
        return Err("certificate_member_chain_mismatch".to_string());
    }
    Ok(RecordCheckpointCertificateMemberV1 {
        request_id,
        responder,
        response_timestamp,
        checkpoint_height,
        checkpoint_hash,
        tip_height,
        tip_hash,
        signature,
    })
}

pub(super) async fn checkpoint_certificate_handler(
    State(state): State<MemChainPeerState>,
    body: Bytes,
) -> Response {
    if body.first().copied() != Some(MEMCHAIN_MAGIC) {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame");
    }
    let message = match decode_memchain(&body[1..]) {
        Ok(message) => message,
        Err(_) => return protocol_error(StatusCode::BAD_REQUEST, "invalid_frame"),
    };
    let MemChainMessage::RecordCheckpointCertificateRequestV1 {
        chain_id,
        known_tip_height,
        known_tip_hash,
        request_id,
        requester,
        request_timestamp,
        signature,
    } = message
    else {
        return protocol_error(StatusCode::BAD_REQUEST, "unexpected_message");
    };

    let now = now_secs();
    if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID || known_tip_height == 0 {
        return protocol_error(StatusCode::BAD_REQUEST, "invalid_certificate_request");
    }
    if now.abs_diff(request_timestamp) > REQUEST_TIMESTAMP_SKEW_SECS {
        return protocol_error(StatusCode::UNAUTHORIZED, "stale_request");
    }
    if state.peer_store.get_valid(&requester, now).is_none() {
        return protocol_error(StatusCode::FORBIDDEN, "unknown_peer");
    }
    let signing_bytes = record_checkpoint_certificate_request_signing_bytes(
        &chain_id,
        known_tip_height,
        &known_tip_hash,
        &request_id,
        &requester,
        request_timestamp,
    );
    if IdentityPublicKey::from_bytes(&requester)
        .and_then(|key| key.verify(&signing_bytes, &signature))
        .is_err()
    {
        return protocol_error(StatusCode::UNAUTHORIZED, "invalid_signature");
    }
    if !state.guard.lock().await.admit(requester, request_id, now) {
        return protocol_error(StatusCode::TOO_MANY_REQUESTS, "rate_or_replay_limited");
    }

    let bundle = match state
        .storage
        .record_commitment_checkpoint_certificate_bundle(known_tip_height, &known_tip_hash)
        .await
    {
        Ok(Some(bundle)) => bundle,
        Ok(None) => return protocol_error(StatusCode::NOT_FOUND, "certificate_unavailable"),
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Refused unaudited certificate export");
            return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
        }
    };
    if !(2..=MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1).contains(&bundle.required_signers)
        || bundle.member_frames.len() < bundle.required_signers
        || bundle.member_frames.len() > MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1
    {
        return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
    }
    let mut members = [None; MAX_CHECKPOINT_CERTIFICATE_MEMBERS_V1];
    for (slot, frame) in members.iter_mut().zip(bundle.member_frames.iter()) {
        *slot = match checkpoint_certificate_member_from_frame(frame, &chain_id) {
            Ok(member) => Some(member),
            Err(error) => {
                warn!(error = %error, "[MEMCHAIN_BLOCK] Refused invalid certificate member");
                return protocol_error(StatusCode::SERVICE_UNAVAILABLE, "certificate_not_verified");
            }
        };
    }
    let responder = state.identity.public_key_bytes();
    let response_timestamp = now_secs();
    let response_signing_bytes = record_checkpoint_certificate_response_signing_bytes(
        &chain_id,
        &request_id,
        &responder,
        response_timestamp,
        bundle.checkpoint_height,
        &bundle.checkpoint_hash,
        &bundle.certificate_digest,
        bundle.required_signers as u8,
        bundle.member_frames.len() as u8,
    );
    let response = MemChainMessage::RecordCheckpointCertificateResponseV1 {
        chain_id,
        request_id,
        responder,
        response_timestamp,
        checkpoint_height: bundle.checkpoint_height,
        checkpoint_hash: bundle.checkpoint_hash,
        certificate_digest: bundle.certificate_digest,
        required_signers: bundle.required_signers as u8,
        members,
        signature: state.identity.sign(&response_signing_bytes),
    };
    let encoded = match encode_memchain(&response) {
        Ok(encoded) => encoded,
        Err(error) => {
            warn!(error = %error, "[MEMCHAIN_BLOCK] Failed to encode certificate response");
            return protocol_error(StatusCode::INTERNAL_SERVER_ERROR, "encode_error");
        }
    };
    debug!(
        checkpoint_height = bundle.checkpoint_height,
        signer_count = bundle.member_frames.len(),
        "[MEMCHAIN_BLOCK] Served authenticated checkpoint certificate"
    );
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/octet-stream")],
        encoded,
    )
        .into_response()
}

/// Maps certificate source and durable outcome into one telemetry disposition.
///
/// [CERTIFICATE-PERSISTENCE-TRUTH 2026-07-29 by Codex] Verification proves that
/// a response is authentic; it does not prove recovery until the exact current
/// policy certificate is durable locally. Keeping this mapping in one pure
/// function prevents direct and carrier paths from drifting apart.
pub(super) const fn follower_certificate_sync_disposition(
    source: CommitmentFollowerCertificateSource,
    persisted: bool,
) -> RecordCommitmentCertificateSyncDisposition {
    match (source, persisted) {
        (CommitmentFollowerCertificateSource::Coordinator, true) => {
            RecordCommitmentCertificateSyncDisposition::Coordinator
        }
        (CommitmentFollowerCertificateSource::PinnedCarrier, true) => {
            RecordCommitmentCertificateSyncDisposition::CarrierRecovered
        }
        (_, false) => RecordCommitmentCertificateSyncDisposition::VerifiedUnpersisted,
    }
}
