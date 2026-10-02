// [ARCH-SPLIT 2026-10-02]
// Checkpoint status, evidence persistence, audit, and witness divergence counts.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

// ============================================
// impl MemoryStorage — Chain State
// ============================================

pub(super) fn checkpoint_observation_freshness(
    last_evidence_at: Option<u64>,
    now: u64,
) -> (&'static str, Option<u64>) {
    let Some(observed_at) = last_evidence_at else {
        return ("unavailable", None);
    };
    let Some(age) = now.checked_sub(observed_at) else {
        // A wall-clock rollback must not make future-dated evidence appear
        // fresh. The signed frame remains in the audited local vault.
        return ("unavailable", None);
    };
    if age <= CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS {
        ("fresh", Some(age))
    } else {
        ("stale", Some(age))
    }
}

/// Classifies one bounded coordinator witness round without implying consensus.
///
/// `attention` is reserved for signed evidence that the coordinator may be
/// behind or that a shared prefix diverged. `shared_prefix` means every
/// attempted witness produced valid evidence compatible with the local chain;
/// it is deliberately not named quorum, finality, or consensus.
pub(super) fn checkpoint_witness_round_state(
    attempted: usize,
    verified: usize,
    failed: usize,
    remote_ahead: usize,
    diverged: usize,
) -> &'static str {
    if attempted == 0 {
        "unavailable"
    } else if verified == 0 {
        "unverified"
    } else if remote_ahead > 0 || diverged > 0 {
        "attention"
    } else if failed > 0 || verified < attempted {
        "partial"
    } else {
        "shared_prefix"
    }
}

/// Derives privacy-safe certificate coverage for the fully audited local tip.
///
/// A certificate proves only that the configured operator-pinned witnesses
/// signed one checkpoint. The state deliberately avoids `finalized`, `quorum`,
/// and `consensus`: those terms require a separate network protocol.
pub(super) fn commitment_block_confirmation_state(
    integrity_verified: bool,
    verified_tip_height: u64,
    latest_certified_height: Option<u64>,
    certificate_signers: usize,
    certificate_required_signers: usize,
) -> (&'static str, u64) {
    if !integrity_verified {
        return ("not_verified", 0);
    }
    if verified_tip_height == 0 {
        return ("empty", 0);
    }
    let Some(certified_height) = latest_certified_height else {
        return ("uncertified", verified_tip_height);
    };
    if certified_height > verified_tip_height {
        return ("certificate_ahead", 0);
    }
    if certificate_required_signers < 2 || certificate_signers < certificate_required_signers {
        return (
            "certificate_invalid",
            verified_tip_height.saturating_sub(certified_height),
        );
    }
    let lag = verified_tip_height.saturating_sub(certified_height);
    if lag == 0 {
        ("witness_certified", 0)
    } else {
        ("certificate_lagging", lag)
    }
}

impl MemoryStorage {
    /// Returns aggregate signed checkpoint evidence without peer, hash,
    /// signature, endpoint, or user metadata.
    pub fn record_commitment_checkpoint_status(&self) -> RecordCommitmentCheckpointStatus {
        let integrity = self.record_commitment_chain_integrity_status();
        let runtime = self.commitment_checkpoint.read();
        let certificate_guard = self.commitment_checkpoint_certificate_anchor.read();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let (observation_freshness, observation_age_seconds) =
            if runtime.evidence_state == "verified" {
                checkpoint_observation_freshness(runtime.last_evidence_at, now)
            } else {
                ("unavailable", None)
            };
        let (block_confirmation_state, uncertified_block_count) =
            commitment_block_confirmation_state(
                integrity.state == "verified",
                integrity.verified_tip_height,
                runtime.latest_certified_height,
                runtime.latest_certificate_signers,
                runtime.latest_certificate_required_signers,
            );
        RecordCommitmentCheckpointStatus {
            contract_version: "record_commitment_checkpoint.v1",
            state: runtime.state.to_string(),
            last_checked_at: runtime.last_checked_at,
            last_converged_at: runtime.last_converged_at,
            last_divergence_at: runtime.last_divergence_at,
            last_failure_at: runtime.last_failure_at,
            last_served_at: runtime.last_served_at,
            local_tip_height: runtime.local_tip_height,
            remote_tip_height: runtime.remote_tip_height,
            proofs_verified_total: runtime.proofs_verified_total,
            proofs_failed_total: runtime.proofs_failed_total,
            divergences_total: runtime.divergences_total,
            requests_served_total: runtime.requests_served_total,
            evidence_state: runtime.evidence_state.to_string(),
            evidence_records: runtime.evidence_records,
            applicable_evidence_records: runtime.applicable_evidence_records,
            deferred_evidence_records: runtime.deferred_evidence_records,
            divergence_evidence_records: runtime.divergence_evidence_records,
            equivocation_incidents: runtime.equivocation_incidents,
            trusted_divergence_incidents: runtime.trusted_divergence_incidents,
            checkpoint_certificates: runtime.checkpoint_certificates,
            latest_certified_height: runtime.latest_certified_height,
            latest_certificate_signers: runtime.latest_certificate_signers,
            latest_certificate_required_signers: runtime
                .latest_certificate_required_signers,
            block_confirmation_state: block_confirmation_state.to_string(),
            uncertified_block_count,
            block_confirmation_policy: "operator-pinned witness certificate coverage of the audited local commitment tip; not permissionless consensus, quorum finality, fork choice, or a public-chain confirmation",
            certificate_rollback_guard_state: certificate_guard.state.to_string(),
            certificate_rollback_guard_height: certificate_guard.anchored_height,
            certificate_rollback_guard_last_verified_at: certificate_guard.last_verified_at,
            certificate_rollback_guard_last_persisted_at: certificate_guard.last_persisted_at,
            certificate_rollback_guard_write_failures_total: certificate_guard
                .write_failures_total,
            certificate_rollback_guard_scope: "detects checkpoint-certificate SQLite rollback or replacement while the separate host-side signed sidecar remains; does not detect whole-host or whole-disk snapshot rollback and is not consensus, quorum, fork choice, or finality",
            production_halted: self.record_commitment_production_halted(),
            last_evidence_at: runtime.last_evidence_at,
            observation_freshness: observation_freshness.to_string(),
            observation_age_seconds,
            freshness_window_seconds: CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
            last_round_state: runtime.last_round_state.to_string(),
            last_round_at: runtime.last_round_at,
            last_round_eligible: runtime.last_round_eligible,
            last_round_attempted: runtime.last_round_attempted,
            last_round_verified: runtime.last_round_verified,
            last_round_failed: runtime.last_round_failed,
            last_round_converged: runtime.last_round_converged,
            last_round_remote_ahead: runtime.last_round_remote_ahead,
            last_round_remote_behind: runtime.last_round_remote_behind,
            last_round_diverged: runtime.last_round_diverged,
            evidence_persistence_failures_total: runtime
                .evidence_persistence_failures_total,
            privacy_policy: "aggregate signed checkpoint outcomes, bounded witness-round coverage, immutable certificate counts, local signed rollback-guard state, applicable/deferred evidence counts, durable observation freshness, and local evidence-vault health only; certificates prove configured pinned-witness observations, not global consensus or fork choice; raw frames, peer identities, block hashes, signatures, certificate digests, sidecar paths, request ids, commitments, owners, payloads, endpoints, routes, and client metadata never leave the node",
        }
    }

    /// Records the aggregate result of one completed bounded witness round.
    ///
    /// This method intentionally stores no peer identity, endpoint, hash,
    /// signature, or request id. It cannot alter the canonical commitment
    /// chain and its counts must never be interpreted as consensus.
    #[allow(clippy::too_many_arguments)]
    pub fn record_commitment_checkpoint_witness_round(
        &self,
        now: u64,
        eligible: usize,
        attempted: usize,
        verified: usize,
        failed: usize,
        converged: usize,
        remote_ahead: usize,
        remote_behind: usize,
        diverged: usize,
    ) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.last_round_state =
            checkpoint_witness_round_state(attempted, verified, failed, remote_ahead, diverged);
        runtime.last_round_at = Some(now);
        runtime.last_round_eligible = eligible;
        runtime.last_round_attempted = attempted;
        runtime.last_round_verified = verified;
        runtime.last_round_failed = failed;
        runtime.last_round_converged = converged;
        runtime.last_round_remote_ahead = remote_ahead;
        runtime.last_round_remote_behind = remote_behind;
        runtime.last_round_diverged = diverged;
    }

    /// Records one outbound, signature-verified checkpoint comparison.
    pub fn record_commitment_checkpoint_verified(
        &self,
        now: u64,
        relation: &'static str,
        local_tip_height: u64,
        remote_tip_height: u64,
    ) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.state = relation;
        runtime.last_checked_at = Some(now);
        runtime.local_tip_height = Some(local_tip_height);
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.proofs_verified_total = runtime.proofs_verified_total.saturating_add(1);
        if relation == "converged" {
            runtime.last_converged_at = Some(now);
        } else if relation == "diverged" {
            runtime.last_divergence_at = Some(now);
            runtime.divergences_total = runtime.divergences_total.saturating_add(1);
        }
    }

    /// Records a checkpoint attempt that did not establish signed evidence.
    pub fn record_commitment_checkpoint_failure(&self, now: u64) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.state = "proof_failed";
        runtime.last_checked_at = Some(now);
        runtime.last_failure_at = Some(now);
        runtime.proofs_failed_total = runtime.proofs_failed_total.saturating_add(1);
    }

    /// Records an authenticated checkpoint response served to another node.
    ///
    /// The requester's claimed height and hash are untrusted inbound context.
    /// They may influence the signed response but must never overwrite this
    /// node's outbound convergence/divergence evidence or observed heights.
    pub fn record_commitment_checkpoint_served(&self, now: u64) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.last_served_at = Some(now);
        runtime.requests_served_total = runtime.requests_served_total.saturating_add(1);
    }

    /// Persists the exact response frame after the node-peer verifier has
    /// completed chain, freshness, identity, signature, and relation checks.
    ///
    /// This method independently rechecks the digest and storage bounds, writes
    /// in one immediate transaction, prunes the oldest non-divergence proof
    /// first, and then re-audits the complete bounded vault. A failure is
    /// returned to the follower so convergence cannot be declared without
    /// durable evidence applicable to the current verified chain.
    ///
    /// # Errors
    ///
    /// Returns an error when inputs violate bounds or relation invariants,
    /// when any retained frame fails cryptographic/canonical verification, or
    /// when the atomic SQLite transaction cannot be completed.
    pub async fn persist_record_commitment_checkpoint_evidence(
        &self,
        observed_at: u64,
        relation: &str,
        local_tip_height: u64,
        remote_tip_height: u64,
        checkpoint_height: u64,
        evidence_digest: &[u8; 32],
        signed_response: &[u8],
    ) -> Result<(), String> {
        self.persist_record_commitment_checkpoint_evidence_with_witness_policy(
            observed_at,
            relation,
            local_tip_height,
            remote_tip_height,
            checkpoint_height,
            evidence_digest,
            signed_response,
            false,
        )
        .await
        .map(|_| ())
    }

    /// Persists evidence while optionally binding it to trusted-witness
    /// security ledgers. Only a caller that has matched the responder to an
    /// explicit operator pin may set `track_trusted_witness_incidents`.
    pub(crate) async fn persist_record_commitment_checkpoint_evidence_with_witness_policy(
        &self,
        observed_at: u64,
        relation: &str,
        local_tip_height: u64,
        remote_tip_height: u64,
        checkpoint_height: u64,
        evidence_digest: &[u8; 32],
        signed_response: &[u8],
        track_trusted_witness_incidents: bool,
    ) -> Result<RecordCommitmentCheckpointEvidencePersistOutcome, String> {
        let failure = |message: String| {
            let mut runtime = self.commitment_checkpoint.write();
            runtime.evidence_persistence_failures_total = runtime
                .evidence_persistence_failures_total
                .saturating_add(1);
            Err(message)
        };
        if !matches!(
            relation,
            "converged" | "remote_ahead" | "remote_behind" | "diverged"
        ) {
            return failure("checkpoint evidence relation is invalid".to_string());
        }
        if signed_response.is_empty() || signed_response.len() > MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES
        {
            return failure("checkpoint evidence frame violates bounds".to_string());
        }
        let computed_digest: [u8; 32] = Sha256::digest(signed_response).into();
        if &computed_digest != evidence_digest {
            return failure("checkpoint evidence digest mismatch".to_string());
        }
        let observed_at_u64 = observed_at;
        let checkpoint_height_u64 = checkpoint_height;
        let observed_at = match i64::try_from(observed_at) {
            Ok(value) => value,
            Err(_) => return failure("checkpoint evidence time exceeds SQLite range".to_string()),
        };
        let local_tip_height = match i64::try_from(local_tip_height) {
            Ok(value) => value,
            Err(_) => {
                return failure("checkpoint evidence local height exceeds SQLite range".to_string())
            }
        };
        let remote_tip_height = match i64::try_from(remote_tip_height) {
            Ok(value) => value,
            Err(_) => {
                return failure(
                    "checkpoint evidence remote height exceeds SQLite range".to_string(),
                )
            }
        };
        let checkpoint_height = match i64::try_from(checkpoint_height) {
            Ok(value) => value,
            Err(_) => {
                return failure("checkpoint evidence height exceeds SQLite range".to_string())
            }
        };

        let result = {
            let mut conn = self.conn.lock().await;
            let operation =
                (|| -> Result<(RecordCommitmentCheckpointEvidenceAudit, usize, usize), String> {
                    let transaction = conn
                        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                        .map_err(|error| {
                            format!("begin checkpoint evidence transaction: {error}")
                        })?;
                    transaction
                        .execute(
                            "INSERT OR IGNORE INTO record_checkpoint_evidence
                         (evidence_digest,observed_at,relation,local_tip_height,
                          remote_tip_height,checkpoint_height,signed_response,created_at)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?2)",
                            params![
                                evidence_digest.as_slice(),
                                observed_at,
                                relation,
                                local_tip_height,
                                remote_tip_height,
                                checkpoint_height,
                                signed_response,
                            ],
                        )
                        .map_err(|error| format!("insert checkpoint evidence: {error}"))?;
                    let stored = transaction
                        .query_row(
                            "SELECT observed_at,relation,local_tip_height,remote_tip_height,
                                checkpoint_height,signed_response
                         FROM record_checkpoint_evidence WHERE evidence_digest=?1",
                            params![evidence_digest.as_slice()],
                            |row| {
                                Ok((
                                    row.get::<_, i64>(0)?,
                                    row.get::<_, String>(1)?,
                                    row.get::<_, i64>(2)?,
                                    row.get::<_, i64>(3)?,
                                    row.get::<_, i64>(4)?,
                                    row.get::<_, Vec<u8>>(5)?,
                                ))
                            },
                        )
                        .map_err(|error| format!("read inserted checkpoint evidence: {error}"))?;
                    if stored.0 != observed_at
                        || stored.1 != relation
                        || stored.2 != local_tip_height
                        || stored.3 != remote_tip_height
                        || stored.4 != checkpoint_height
                        || stored.5.as_slice() != signed_response
                    {
                        return Err("existing checkpoint evidence conflicts with verified frame"
                            .to_string());
                    }
                    let new_equivocation_incidents = if track_trusted_witness_incidents {
                        detect_trusted_checkpoint_equivocations(
                            &transaction,
                            evidence_digest,
                            signed_response,
                            observed_at_u64,
                        )?
                    } else {
                        0
                    };
                    let new_trusted_divergence_incidents =
                        if track_trusted_witness_incidents && relation == "diverged" {
                            insert_trusted_checkpoint_divergence_incident(
                                &transaction,
                                evidence_digest,
                                signed_response,
                                checkpoint_height_u64,
                                observed_at_u64,
                            )?
                        } else {
                            0
                        };
                    let count_i64 = transaction
                        .query_row(
                            "SELECT COUNT(*) FROM record_checkpoint_evidence",
                            [],
                            |row| row.get::<_, i64>(0),
                        )
                        .map_err(|error| format!("count checkpoint evidence: {error}"))?;
                    let excess = count_i64.saturating_sub(CHECKPOINT_EVIDENCE_CAPACITY as i64);
                    if excess > 0 {
                        transaction
                            .execute(
                                "DELETE FROM record_checkpoint_evidence WHERE rowid IN (
                                SELECT rowid FROM record_checkpoint_evidence
                                WHERE evidence_digest NOT IN (
                                    SELECT first_evidence_digest
                                    FROM record_checkpoint_equivocations
                                    UNION
                                    SELECT second_evidence_digest
                                    FROM record_checkpoint_equivocations
                                    UNION
                                    SELECT evidence_digest
                                    FROM record_checkpoint_trusted_divergences
                                    UNION
                                    SELECT evidence_digest
                                    FROM record_checkpoint_certificate_members
                                )
                                ORDER BY CASE WHEN relation='diverged' THEN 1 ELSE 0 END ASC,
                                         observed_at ASC,rowid ASC
                                LIMIT ?1
                            )",
                                params![excess],
                            )
                            .map_err(|error| format!("prune checkpoint evidence: {error}"))?;
                        let retained_count = transaction
                            .query_row(
                                "SELECT COUNT(*) FROM record_checkpoint_evidence",
                                [],
                                |row| row.get::<_, i64>(0),
                            )
                            .map_err(|error| format!("recount checkpoint evidence: {error}"))?;
                        if retained_count > CHECKPOINT_EVIDENCE_CAPACITY as i64 {
                            return Err(
                                "checkpoint evidence capacity is reserved by security incidents"
                                    .to_string(),
                            );
                        }
                    }
                    // The vault is deliberately small. Re-audit every retained
                    // frame inside the same write transaction. Invalid new data
                    // therefore rolls back atomically, and evidence deferred by a
                    // follower rollback becomes applicable only after its local
                    // chain prefix has actually been restored and reverified.
                    let report = audit_checkpoint_evidence_snapshot(&transaction)?;
                    transaction
                        .commit()
                        .map_err(|error| format!("commit checkpoint evidence: {error}"))?;
                    Ok((
                        report,
                        new_equivocation_incidents,
                        new_trusted_divergence_incidents,
                    ))
                })();
            if matches!(
                &operation,
                Ok((_, equivocations, divergences)) if *equivocations > 0 || *divergences > 0
            ) {
                // Set while this task still owns the SQLite connection. Any
                // racing append must wait, then observe the one-way latch.
                self.commitment_production_halted
                    .store(true, Ordering::Release);
            }
            operation
        };

        match result {
            Ok((report, new_equivocation_incidents, new_trusted_divergence_incidents)) => {
                let mut runtime = self.commitment_checkpoint.write();
                runtime.evidence_records = report.evidence_records;
                runtime.applicable_evidence_records = report.applicable_evidence_records;
                runtime.deferred_evidence_records = report.deferred_evidence_records;
                runtime.divergence_evidence_records = report.divergence_evidence_records;
                runtime.equivocation_incidents = report.equivocation_incidents;
                runtime.trusted_divergence_incidents = report.trusted_divergence_incidents;
                runtime.checkpoint_certificates = report.checkpoint_certificates;
                runtime.latest_certified_height = report.latest_certified_height;
                runtime.latest_certificate_signers = report.latest_certificate_signers;
                runtime.latest_certificate_required_signers =
                    report.latest_certificate_required_signers;
                runtime.last_evidence_at = report.last_evidence_at;
                if new_equivocation_incidents > 0 {
                    Ok(RecordCommitmentCheckpointEvidencePersistOutcome::EquivocationDetected)
                } else if new_trusted_divergence_incidents > 0 {
                    Ok(RecordCommitmentCheckpointEvidencePersistOutcome::TrustedDivergenceDetected)
                } else {
                    Ok(RecordCommitmentCheckpointEvidencePersistOutcome::Stored)
                }
            }
            Err(error) => failure(error),
        }
    }

    pub(super) fn apply_record_commitment_checkpoint_audit(
        &self,
        report: &RecordCommitmentCheckpointEvidenceAudit,
    ) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.evidence_records = report.evidence_records;
        runtime.applicable_evidence_records = report.applicable_evidence_records;
        runtime.deferred_evidence_records = report.deferred_evidence_records;
        runtime.divergence_evidence_records = report.divergence_evidence_records;
        runtime.equivocation_incidents = report.equivocation_incidents;
        runtime.trusted_divergence_incidents = report.trusted_divergence_incidents;
        runtime.checkpoint_certificates = report.checkpoint_certificates;
        runtime.latest_certified_height = report.latest_certified_height;
        runtime.latest_certificate_signers = report.latest_certificate_signers;
        runtime.latest_certificate_required_signers = report.latest_certificate_required_signers;
        runtime.last_evidence_at = report.last_evidence_at;
    }

    /// Re-verifies every durable checkpoint frame before networking starts.
    /// Any malformed digest, non-canonical frame, invalid signature, impossible
    /// height relation, or available local historical-hash mismatch fails
    /// startup closed. Valid frames above the currently audited local tip are
    /// retained as deferred recovery evidence and cannot affect live state.
    ///
    /// # Errors
    ///
    /// Returns an error when the bounded vault or any currently applicable
    /// relation cannot be verified from one consistent SQLite snapshot.
    pub async fn audit_record_commitment_checkpoint_evidence(
        &self,
    ) -> Result<RecordCommitmentCheckpointEvidenceAudit, String> {
        self.commitment_checkpoint.write().evidence_state = "not_audited";
        let result = {
            let mut conn = self.conn.lock().await;
            audit_checkpoint_evidence_connection(&mut conn)
        };
        match result {
            Ok(report) => {
                let mut runtime = self.commitment_checkpoint.write();
                runtime.evidence_state = "verified";
                runtime.evidence_records = report.evidence_records;
                runtime.applicable_evidence_records = report.applicable_evidence_records;
                runtime.deferred_evidence_records = report.deferred_evidence_records;
                runtime.divergence_evidence_records = report.divergence_evidence_records;
                runtime.equivocation_incidents = report.equivocation_incidents;
                runtime.trusted_divergence_incidents = report.trusted_divergence_incidents;
                runtime.checkpoint_certificates = report.checkpoint_certificates;
                runtime.latest_certified_height = report.latest_certified_height;
                runtime.latest_certificate_signers = report.latest_certificate_signers;
                runtime.latest_certificate_required_signers =
                    report.latest_certificate_required_signers;
                runtime.last_evidence_at = report.last_evidence_at;
                Ok(report)
            }
            Err(error) => {
                self.commitment_checkpoint.write().evidence_state = "invalid";
                Err(error)
            }
        }
    }

    /// Counts durable equivocation incidents for the currently configured
    /// operator-pinned witnesses. This method returns only an aggregate and
    /// must be called after the full evidence-vault audit has succeeded.
    pub async fn count_record_commitment_checkpoint_equivocations_for_witnesses(
        &self,
        witness_node_ids: &[[u8; 32]],
    ) -> Result<u64, String> {
        let conn = self.conn.lock().await;
        let mut total = 0u64;
        let mut seen = Vec::with_capacity(witness_node_ids.len());
        for node_id in witness_node_ids {
            if seen.contains(node_id) {
                continue;
            }
            seen.push(*node_id);
            let count = conn
                .query_row(
                    "SELECT COUNT(*) FROM record_checkpoint_equivocations WHERE responder=?1",
                    params![node_id.as_slice()],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(|error| format!("count trusted checkpoint equivocations: {error}"))?;
            let count = u64::try_from(count)
                .map_err(|_| "trusted checkpoint equivocation count is invalid".to_string())?;
            total = total.saturating_add(count);
        }
        Ok(total)
    }

    /// Counts sticky divergent-prefix incidents for currently configured
    /// operator-pinned witnesses. Witness identities never leave this method.
    pub async fn count_record_commitment_checkpoint_trusted_divergences_for_witnesses(
        &self,
        witness_node_ids: &[[u8; 32]],
    ) -> Result<u64, String> {
        let conn = self.conn.lock().await;
        let mut total = 0u64;
        let mut seen = Vec::with_capacity(witness_node_ids.len());
        for node_id in witness_node_ids {
            if seen.contains(node_id) {
                continue;
            }
            seen.push(*node_id);
            let count = conn
                .query_row(
                    "SELECT COUNT(*) FROM record_checkpoint_trusted_divergences
                     WHERE responder=?1",
                    params![node_id.as_slice()],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(|error| format!("count trusted checkpoint divergences: {error}"))?;
            let count = u64::try_from(count)
                .map_err(|_| "trusted checkpoint divergence count is invalid".to_string())?;
            total = total.saturating_add(count);
        }
        Ok(total)
    }
}
