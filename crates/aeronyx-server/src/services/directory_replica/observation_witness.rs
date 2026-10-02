// [ARCH-SPLIT 2026-10-02]
// Witness eligibility, persistence, and response checks.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Returns the number of current operator-pinned witnesses whose canonical
    /// accepted receipts cover the specified latest checkpoint.
    ///
    /// Historical receipts from removed pins remain durable but are excluded.
    /// The complete latest receipt set and its checkpoint are cryptographically
    /// re-verified before the aggregate count is returned.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when pins are malformed or
    /// duplicated, latest retained receipt evidence fails verification, or
    /// `SQLite` cannot complete the bounded read.
    pub fn verified_observation_witness_count_for_pins(
        &self,
        checkpoint_sequence: u64,
        eligible_witnesses: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        if checkpoint_sequence == 0 || eligible_witnesses.is_empty() {
            return Ok(0);
        }
        let eligible_witnesses =
            Self::validate_observation_witness_eligibility(eligible_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let latest = Self::latest_verified_observation_witness_set(
            &connection,
            &self.local_node_id,
            observed_at,
        )?;
        if latest.sequence != checkpoint_sequence {
            return Ok(0);
        }
        u64::try_from(
            latest
                .witness_node_ids
                .iter()
                .filter(|witness| eligible_witnesses.contains(*witness))
                .count(),
        )
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "current pinned observation witness count exceeds u64".to_string(),
            )
        })
    }

    /// Independently evaluates an external observer checkpoint against this
    /// node's own producer-isolated replicas.
    ///
    /// Signature validity alone is never acceptance. Every exact producer tip
    /// must exist locally and the overlap root must recompute identically.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for malformed signatures, wrong
    /// chain/time, self-witness attempts, or local store integrity failures.
    pub fn evaluate_observation_checkpoint_witness(
        &self,
        checkpoint: &DirectoryObservationCheckpointV1,
        observed_at: u64,
    ) -> Result<DirectoryObservationWitnessDecision, DirectoryReplicaStoreError> {
        checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if checkpoint.observer == self.local_node_id {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint cannot witness itself".to_string(),
            ));
        }
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        for tip in &checkpoint.producer_tips {
            match Self::observation_block_hash_at(
                &connection,
                &self.local_node_id,
                &tip.producer,
                tip.tip_height,
            )? {
                None => return Ok(DirectoryObservationWitnessDecision::EvidenceUnavailable),
                Some(local_hash) if local_hash != tip.tip_hash => {
                    return Ok(DirectoryObservationWitnessDecision::EvidenceConflict)
                }
                Some(_) => {}
            }
        }
        let recomputed = Self::recompute_observation_checkpoint_root(
            &connection,
            checkpoint,
            &self.local_node_id,
        )?;
        drop(connection);
        if recomputed != checkpoint.observation_root {
            return Ok(DirectoryObservationWitnessDecision::EvidenceConflict);
        }
        Ok(DirectoryObservationWitnessDecision::Accepted)
    }

    /// Persists one accepted, canonical external witness receipt for a local
    /// checkpoint. Exact retries are idempotent.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the response is malformed,
    /// noncanonical, not accepted, self-signed, does not bind a retained local
    /// checkpoint, conflicts at the same witness/sequence, or has an invalid
    /// timestamp or signature.
    #[allow(clippy::too_many_lines)]
    pub fn persist_observation_checkpoint_witness(
        &self,
        response: &DirectorySyncMessage,
        observed_at: u64,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let response_blob = encode_directory_sync_message(response)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if response_blob.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_BYTES {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness response exceeds size bound".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Request(
                "unexpected observation witness response".to_string(),
            ));
        };
        if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || *observer != self.local_node_id
            || *responder == self.local_node_id
            || *responder == [0u8; 32]
            || *checkpoint_sequence == 0
            || *checkpoint_hash == [0u8; 32]
            || *outcome != DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1
            || *response_timestamp == 0
            || *response_timestamp > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness response contract mismatch".to_string(),
            ));
        }
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            chain_id,
            request_id,
            observer,
            *checkpoint_sequence,
            checkpoint_hash,
            responder,
            *response_timestamp,
            *outcome,
        );
        IdentityPublicKey::from_bytes(responder)
            .and_then(|key| key.verify(&signing_bytes, signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Request(
                    "observation witness response signature is invalid".to_string(),
                )
            })?;

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let checkpoint_blob: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT checkpoint_blob FROM directory_observation_checkpoints
                 WHERE sequence = ?1 AND checkpoint_hash = ?2",
                params![
                    u64_to_i64(
                        *checkpoint_sequence,
                        "observation witness checkpoint sequence"
                    )?,
                    checkpoint_hash.as_slice()
                ],
                |row| row.get(0),
            )
            .optional()?;
        let Some(checkpoint_blob) = checkpoint_blob else {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness references an unknown local checkpoint".to_string(),
            ));
        };
        let checkpoint = decode_observation_checkpoint(&checkpoint_blob)?;
        if checkpoint.observer != *observer
            || checkpoint.sequence != *checkpoint_sequence
            || checkpoint.hash() != *checkpoint_hash
            || *response_timestamp < checkpoint.observed_at
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness does not match its local checkpoint".to_string(),
            ));
        }
        let existing_hash: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT checkpoint_hash FROM directory_observation_checkpoint_witnesses
                 WHERE checkpoint_sequence = ?1 AND witness_node_id = ?2",
                params![
                    u64_to_i64(
                        *checkpoint_sequence,
                        "observation witness checkpoint sequence"
                    )?,
                    responder.as_slice()
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(existing_hash) = existing_hash {
            if bytes32(
                &existing_hash,
                "observation witness existing checkpoint hash",
            )? != *checkpoint_hash
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation witness signed conflicting hashes at one sequence".to_string(),
                ));
            }
            return Ok(false);
        }
        let existing_witnesses = nonnegative_i64_to_u64(
            transaction.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoint_witnesses
                 WHERE checkpoint_sequence = ?1",
                params![u64_to_i64(
                    *checkpoint_sequence,
                    "observation witness checkpoint sequence"
                )?],
                |row| row.get(0),
            )?,
            "observation checkpoint witness count",
        )?;
        let maximum_witnesses =
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness producer bound exceeds u64".to_string(),
                )
            })?;
        if existing_witnesses >= maximum_witnesses {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint witness set exceeds producer bound".to_string(),
            ));
        }
        transaction.execute(
            "INSERT INTO directory_observation_checkpoint_witnesses
                (checkpoint_hash, checkpoint_sequence, observer, witness_node_id,
                 witnessed_at, response_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                checkpoint_hash.as_slice(),
                u64_to_i64(
                    *checkpoint_sequence,
                    "observation witness checkpoint sequence"
                )?,
                observer.as_slice(),
                responder.as_slice(),
                u64_to_i64(*response_timestamp, "observation witness timestamp")?,
                response_blob,
            ],
        )?;
        transaction.commit()?;
        drop(connection);
        Ok(true)
    }

    /// Atomically persists one bounded round of privacy-safe witness outcomes.
    ///
    /// Only aggregate mutually exclusive counters and timestamps are retained.
    /// The row contains no witness identity, endpoint, request id, signature,
    /// checkpoint hash, response body, route, or user-plane metadata.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the round is empty or over
    /// the protocol producer bound, time/sequence regresses, counters overflow,
    /// the existing aggregate is malformed, or `SQLite` rejects the write.
    pub fn persist_observation_witness_outcome_round(
        &self,
        checkpoint_sequence: u64,
        observed_at: u64,
        outcomes: &[DirectoryObservationWitnessOutcome],
    ) -> Result<DirectoryObservationWitnessOutcomeSnapshot, DirectoryReplicaStoreError> {
        if checkpoint_sequence == 0
            || observed_at == 0
            || outcomes.is_empty()
            || outcomes.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness outcome round fields are invalid".to_string(),
            ));
        }
        let round = DirectoryObservationWitnessOutcomeCounters::from_outcomes(outcomes);
        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let checkpoint_exists = transaction.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM directory_observation_checkpoints WHERE sequence = ?1
             )",
            params![u64_to_i64(
                checkpoint_sequence,
                "observation witness checkpoint sequence"
            )?],
            |row| row.get::<_, i64>(0),
        )?;
        if checkpoint_exists != 1 {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness outcome references an unknown checkpoint".to_string(),
            ));
        }
        let snapshot = Self::load_observation_witness_outcome_snapshot(&transaction)?
            .next_durable_round(checkpoint_sequence, observed_at, round)?;
        Self::upsert_observation_witness_outcome_snapshot(&transaction, &snapshot)?;
        transaction.commit()?;
        drop(connection);
        Ok(snapshot)
    }

    // The long SQL statement is intentionally isolated from policy and state
    // transition logic so its column/parameter ordering can be audited as one unit.
    #[allow(clippy::too_many_lines)]
    pub(super) fn upsert_observation_witness_outcome_snapshot(
        transaction: &Transaction<'_>,
        snapshot: &DirectoryObservationWitnessOutcomeSnapshot,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let totals = snapshot.totals;
        let round = snapshot.last_round;
        let observed_at = snapshot.last_round_at.ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness outcome round timestamp is missing".to_string(),
            )
        })?;
        transaction.execute(
            "INSERT INTO directory_observation_witness_outcomes
                (singleton, rounds_total, attempts_total, accepted_total,
                 evidence_unavailable_total, evidence_conflict_total,
                 peer_unavailable_total, transport_failures_total,
                 verification_failures_total, persistence_failures_total,
                 last_checkpoint_sequence, last_round_at, last_success_at,
                 last_failure_at, last_round_attempts, last_round_accepted,
                 last_round_evidence_unavailable, last_round_evidence_conflict,
                 last_round_peer_unavailable, last_round_transport_failures,
                 last_round_verification_failures,
                 last_round_persistence_failures, updated_at)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                     ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?11)
             ON CONFLICT(singleton) DO UPDATE SET
                 rounds_total = excluded.rounds_total,
                 attempts_total = excluded.attempts_total,
                 accepted_total = excluded.accepted_total,
                 evidence_unavailable_total = excluded.evidence_unavailable_total,
                 evidence_conflict_total = excluded.evidence_conflict_total,
                 peer_unavailable_total = excluded.peer_unavailable_total,
                 transport_failures_total = excluded.transport_failures_total,
                 verification_failures_total = excluded.verification_failures_total,
                 persistence_failures_total = excluded.persistence_failures_total,
                 last_checkpoint_sequence = excluded.last_checkpoint_sequence,
                 last_round_at = excluded.last_round_at,
                 last_success_at = excluded.last_success_at,
                 last_failure_at = excluded.last_failure_at,
                 last_round_attempts = excluded.last_round_attempts,
                 last_round_accepted = excluded.last_round_accepted,
                 last_round_evidence_unavailable = excluded.last_round_evidence_unavailable,
                 last_round_evidence_conflict = excluded.last_round_evidence_conflict,
                 last_round_peer_unavailable = excluded.last_round_peer_unavailable,
                 last_round_transport_failures = excluded.last_round_transport_failures,
                 last_round_verification_failures = excluded.last_round_verification_failures,
                 last_round_persistence_failures = excluded.last_round_persistence_failures,
                 updated_at = excluded.updated_at",
            params![
                u64_to_i64(snapshot.rounds, "observation witness outcome rounds")?,
                u64_to_i64(totals.attempts(), "observation witness outcome attempts")?,
                u64_to_i64(totals.accepted, "observation witness accepted total")?,
                u64_to_i64(
                    totals.evidence_unavailable,
                    "observation witness unavailable total"
                )?,
                u64_to_i64(
                    totals.evidence_conflict,
                    "observation witness conflict total"
                )?,
                u64_to_i64(totals.peer_unavailable, "observation witness peer total")?,
                u64_to_i64(
                    totals.transport_failures,
                    "observation witness transport total"
                )?,
                u64_to_i64(
                    totals.verification_failures,
                    "observation witness verification total"
                )?,
                u64_to_i64(
                    totals.persistence_failures,
                    "observation witness persistence total"
                )?,
                u64_to_i64(
                    snapshot.last_checkpoint_sequence,
                    "observation witness checkpoint sequence"
                )?,
                u64_to_i64(observed_at, "observation witness round timestamp")?,
                snapshot
                    .last_success_at
                    .map(|value| u64_to_i64(value, "observation witness success timestamp"))
                    .transpose()?,
                snapshot
                    .last_failure_at
                    .map(|value| u64_to_i64(value, "observation witness failure timestamp"))
                    .transpose()?,
                u64_to_i64(round.attempts(), "observation witness last round attempts")?,
                u64_to_i64(round.accepted, "observation witness last round accepted")?,
                u64_to_i64(
                    round.evidence_unavailable,
                    "observation witness last round unavailable"
                )?,
                u64_to_i64(
                    round.evidence_conflict,
                    "observation witness last round conflict"
                )?,
                u64_to_i64(
                    round.peer_unavailable,
                    "observation witness last round peer unavailable"
                )?,
                u64_to_i64(
                    round.transport_failures,
                    "observation witness last round transport"
                )?,
                u64_to_i64(
                    round.verification_failures,
                    "observation witness last round verification"
                )?,
                u64_to_i64(
                    round.persistence_failures,
                    "observation witness last round persistence"
                )?,
            ],
        )?;
        Ok(())
    }

    pub(super) fn load_observation_witness_summary(
        connection: &Connection,
    ) -> Result<ObservationWitnessAudit, DirectoryReplicaStoreError> {
        let witnesses = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoint_witnesses",
                [],
                |row| row.get(0),
            )?,
            "observation witness count",
        )?;
        if witnesses == 0 {
            return Ok(ObservationWitnessAudit::default());
        }
        let latest_sequence = positive_i64_to_u64(
            connection.query_row(
                "SELECT MAX(checkpoint_sequence)
                 FROM directory_observation_checkpoint_witnesses",
                [],
                |row| row.get(0),
            )?,
            "latest witnessed checkpoint sequence",
        )?;
        let latest_witnesses = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoint_witnesses
                 WHERE checkpoint_sequence = ?1",
                params![u64_to_i64(
                    latest_sequence,
                    "latest witnessed checkpoint sequence"
                )?],
                |row| row.get(0),
            )?,
            "latest checkpoint witness count",
        )?;
        Ok(ObservationWitnessAudit {
            witnesses,
            latest_sequence,
            latest_witnesses,
        })
    }

    pub(super) fn query_observation_witness_outcome_row(
        connection: &Connection,
    ) -> Result<Option<StoredObservationWitnessOutcomeRow>, DirectoryReplicaStoreError> {
        Ok(connection
            .query_row(
                "SELECT rounds_total, attempts_total, accepted_total,
                        evidence_unavailable_total, evidence_conflict_total,
                        peer_unavailable_total, transport_failures_total,
                        verification_failures_total, persistence_failures_total,
                        last_checkpoint_sequence, last_round_at, last_success_at,
                        last_failure_at, last_round_attempts, last_round_accepted,
                        last_round_evidence_unavailable,
                        last_round_evidence_conflict, last_round_peer_unavailable,
                        last_round_transport_failures,
                        last_round_verification_failures,
                        last_round_persistence_failures, updated_at
                 FROM directory_observation_witness_outcomes WHERE singleton = 1",
                [],
                |row| {
                    Ok(StoredObservationWitnessOutcomeRow {
                        rounds: row.get(0)?,
                        attempts: row.get(1)?,
                        totals: [
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                            row.get(6)?,
                            row.get(7)?,
                            row.get(8)?,
                        ],
                        last_checkpoint_sequence: row.get(9)?,
                        last_round_at: row.get(10)?,
                        last_success_at: row.get(11)?,
                        last_failure_at: row.get(12)?,
                        last_round_attempts: row.get(13)?,
                        last_round: [
                            row.get(14)?,
                            row.get(15)?,
                            row.get(16)?,
                            row.get(17)?,
                            row.get(18)?,
                            row.get(19)?,
                            row.get(20)?,
                        ],
                        updated_at: row.get(21)?,
                    })
                },
            )
            .optional()?)
    }

    pub(super) fn decode_observation_witness_outcome_counters(
        values: [i64; 7],
        prefix: &str,
    ) -> Result<DirectoryObservationWitnessOutcomeCounters, DirectoryReplicaStoreError> {
        Ok(DirectoryObservationWitnessOutcomeCounters {
            accepted: nonnegative_i64_to_u64(values[0], &format!("{prefix} accepted"))?,
            evidence_unavailable: nonnegative_i64_to_u64(
                values[1],
                &format!("{prefix} evidence unavailable"),
            )?,
            evidence_conflict: nonnegative_i64_to_u64(
                values[2],
                &format!("{prefix} evidence conflict"),
            )?,
            peer_unavailable: nonnegative_i64_to_u64(
                values[3],
                &format!("{prefix} peer unavailable"),
            )?,
            transport_failures: nonnegative_i64_to_u64(
                values[4],
                &format!("{prefix} transport failures"),
            )?,
            verification_failures: nonnegative_i64_to_u64(
                values[5],
                &format!("{prefix} verification failures"),
            )?,
            persistence_failures: nonnegative_i64_to_u64(
                values[6],
                &format!("{prefix} persistence failures"),
            )?,
        })
    }

    pub(super) fn load_observation_witness_outcome_snapshot(
        connection: &Connection,
    ) -> Result<DirectoryObservationWitnessOutcomeSnapshot, DirectoryReplicaStoreError> {
        let row = Self::query_observation_witness_outcome_row(connection)?;
        let Some(row) = row else {
            return Ok(DirectoryObservationWitnessOutcomeSnapshot::default());
        };
        let totals = Self::decode_observation_witness_outcome_counters(
            row.totals,
            "observation witness total",
        )?;
        let last_round = Self::decode_observation_witness_outcome_counters(
            row.last_round,
            "observation witness last round",
        )?;
        let rounds = positive_i64_to_u64(row.rounds, "observation witness rounds")?;
        let attempts = positive_i64_to_u64(row.attempts, "observation witness attempts")?;
        let last_round_attempts = positive_i64_to_u64(
            row.last_round_attempts,
            "observation witness last round attempts",
        )?;
        let optional_timestamp = |value: Option<i64>, field: &str| {
            value
                .map(|value| positive_i64_to_u64(value, field))
                .transpose()
        };
        let last_checkpoint_sequence = positive_i64_to_u64(
            row.last_checkpoint_sequence,
            "observation witness checkpoint sequence",
        )?;
        let last_round_at = positive_i64_to_u64(
            row.last_round_at,
            "observation witness last round timestamp",
        )?;
        let last_success_at =
            optional_timestamp(row.last_success_at, "observation witness success timestamp")?;
        let last_failure_at =
            optional_timestamp(row.last_failure_at, "observation witness failure timestamp")?;
        let updated_at = positive_i64_to_u64(
            row.updated_at,
            "observation witness outcome update timestamp",
        )?;
        let maximum_round_attempts = u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness producer bound exceeds u64".to_string(),
                )
            })?;
        if totals.attempts() != attempts
            || last_round.attempts() != last_round_attempts
            || last_round_attempts > maximum_round_attempts
            || rounds > attempts
            || attempts < last_round_attempts
            || updated_at != last_round_at
            || last_success_at.is_some_and(|timestamp| timestamp > last_round_at)
            || last_failure_at.is_some_and(|timestamp| timestamp > last_round_at)
            || (totals.accepted > 0) != last_success_at.is_some()
            || (totals.failures() > 0) != last_failure_at.is_some()
            || (last_round.accepted > 0 && last_success_at != Some(last_round_at))
            || (last_round.failures() > 0 && last_failure_at != Some(last_round_at))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome aggregate is inconsistent".to_string(),
            ));
        }
        Ok(DirectoryObservationWitnessOutcomeSnapshot {
            rounds,
            totals,
            last_checkpoint_sequence,
            last_round_at: Some(last_round_at),
            last_success_at,
            last_failure_at,
            last_round,
            telemetry_persistence_failures: 0,
        })
    }

    pub(super) fn validate_observation_witness_eligibility(
        witnesses: &[[u8; 32]],
    ) -> Result<HashSet<[u8; 32]>, DirectoryReplicaStoreError> {
        if witnesses.is_empty()
            || witnesses.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
            || witnesses.iter().any(|witness| *witness == [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness eligibility set is invalid".to_string(),
            ));
        }
        let unique = witnesses.iter().copied().collect::<HashSet<_>>();
        if unique.len() != witnesses.len() {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness eligibility set contains duplicates".to_string(),
            ));
        }
        Ok(unique)
    }

    pub(super) fn latest_verified_observation_witness_set(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<VerifiedObservationWitnessSet, DirectoryReplicaStoreError> {
        let latest_sequence: Option<i64> = connection.query_row(
            "SELECT MAX(checkpoint_sequence)
             FROM directory_observation_checkpoint_witnesses",
            [],
            |row| row.get(0),
        )?;
        let Some(latest_sequence) = latest_sequence else {
            return Ok(VerifiedObservationWitnessSet::default());
        };
        let latest_sequence = positive_i64_to_u64(
            latest_sequence,
            "latest observation witness checkpoint sequence",
        )?;
        let row_limit = i64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1.saturating_add(1))
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness verification bound exceeds i64".to_string(),
                )
            })?;
        let mut statement = connection.prepare(
            "SELECT checkpoint_hash, checkpoint_sequence, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_checkpoint_witnesses
             WHERE checkpoint_sequence = ?1
             ORDER BY witness_node_id ASC
             LIMIT ?2",
        )?;
        let rows = statement.query_map(
            params![
                u64_to_i64(
                    latest_sequence,
                    "latest observation witness checkpoint sequence"
                )?,
                row_limit
            ],
            |row| {
                Ok(StoredObservationWitnessRow {
                    checkpoint_hash: row.get(0)?,
                    checkpoint_sequence: row.get(1)?,
                    observer: row.get(2)?,
                    witness_node_id: row.get(3)?,
                    witnessed_at: row.get(4)?,
                    response_blob: row.get(5)?,
                })
            },
        )?;
        let mut verified_rows = 0usize;
        let mut witness_node_ids = Vec::with_capacity(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1);
        let mut receipts = Vec::with_capacity(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1);
        for row in rows {
            let row = row?;
            let verified =
                Self::verify_observation_witness_response(&row, local_node_id, observed_at)?;
            Self::verify_observation_witness_checkpoint(connection, &row, &verified)?;
            witness_node_ids.push(bytes32(
                &row.witness_node_id,
                "latest observation witness node id",
            )?);
            receipts.push(verified.receipt);
            verified_rows = verified_rows.saturating_add(1);
        }
        drop(statement);
        if verified_rows == 0 || verified_rows > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "latest observation witness set violates its verification bound".to_string(),
            ));
        }
        Self::verify_observation_checkpoint_at_sequence(
            connection,
            local_node_id,
            observed_at,
            latest_sequence,
        )?;
        Ok(VerifiedObservationWitnessSet {
            sequence: latest_sequence,
            witness_node_ids,
            receipts,
        })
    }

    pub(super) fn audit_observation_witnesses(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<ObservationWitnessAudit, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT checkpoint_hash, checkpoint_sequence, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_checkpoint_witnesses
             ORDER BY checkpoint_sequence ASC, witness_node_id ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredObservationWitnessRow {
                checkpoint_hash: row.get(0)?,
                checkpoint_sequence: row.get(1)?,
                observer: row.get(2)?,
                witness_node_id: row.get(3)?,
                witnessed_at: row.get(4)?,
                response_blob: row.get(5)?,
            })
        })?;
        let mut audit = ObservationWitnessAudit::default();
        let maximum_witnesses =
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness producer bound exceeds u64".to_string(),
                )
            })?;
        for row in rows {
            let row = row?;
            let verified =
                Self::verify_observation_witness_response(&row, local_node_id, observed_at)?;
            Self::verify_observation_witness_checkpoint(connection, &row, &verified)?;
            audit.witnesses = audit.witnesses.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness count exceeds u64".to_string(),
                )
            })?;
            if verified.sequence > audit.latest_sequence {
                audit.latest_sequence = verified.sequence;
                audit.latest_witnesses = 1;
            } else if verified.sequence == audit.latest_sequence {
                audit.latest_witnesses =
                    audit.latest_witnesses.checked_add(1).ok_or_else(|| {
                        DirectoryReplicaStoreError::Integrity(
                            "latest observation witness count exceeds u64".to_string(),
                        )
                    })?;
            }
            if audit.latest_witnesses > maximum_witnesses {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint witness set exceeds producer bound".to_string(),
                ));
            }
        }
        drop(statement);
        Ok(audit)
    }

    pub(super) fn verify_observation_witness_response(
        row: &StoredObservationWitnessRow,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<VerifiedObservationWitness, DirectoryReplicaStoreError> {
        if row.response_blob.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_BYTES {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness response exceeds size bound".to_string(),
            ));
        }
        let response = decode_directory_sync_message(&row.response_blob).map_err(|error| {
            DirectoryReplicaStoreError::Integrity(format!(
                "observation witness response decode failed: {error}"
            ))
        })?;
        if encode_directory_sync_message(&response).map_err(|error| {
            DirectoryReplicaStoreError::Integrity(format!(
                "observation witness response encode failed: {error}"
            ))
        })? != row.response_blob
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness response encoding is not canonical".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness row contains an unexpected frame".to_string(),
            ));
        };
        let row_sequence = positive_i64_to_u64(
            row.checkpoint_sequence,
            "observation witness checkpoint sequence",
        )?;
        let row_timestamp = positive_i64_to_u64(row.witnessed_at, "observation witness timestamp")?;
        if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || observer != *local_node_id
            || responder == *local_node_id
            || responder == [0u8; 32]
            || outcome != DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1
            || checkpoint_sequence != row_sequence
            || checkpoint_hash
                != bytes32(&row.checkpoint_hash, "observation witness checkpoint hash")?
            || observer != bytes32(&row.observer, "observation witness observer")?
            || responder != bytes32(&row.witness_node_id, "observation witness identity")?
            || response_timestamp != row_timestamp
            || response_timestamp > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness row does not match its signed response".to_string(),
            ));
        }
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &chain_id,
            &request_id,
            &observer,
            checkpoint_sequence,
            &checkpoint_hash,
            &responder,
            response_timestamp,
            outcome,
        );
        IdentityPublicKey::from_bytes(&responder)
            .and_then(|key| key.verify(&signing_bytes, &signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness response signature is invalid".to_string(),
                )
            })?;
        let receipt = DirectoryObservationWitnessReceiptV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        };
        Ok(VerifiedObservationWitness {
            sequence: checkpoint_sequence,
            checkpoint_hash,
            observer,
            response_timestamp,
            receipt,
        })
    }

    pub(super) fn verify_observation_witness_checkpoint(
        connection: &Connection,
        row: &StoredObservationWitnessRow,
        witness: &VerifiedObservationWitness,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let checkpoint_blob: Option<Vec<u8>> = connection
            .query_row(
                "SELECT checkpoint_blob FROM directory_observation_checkpoints
                 WHERE sequence = ?1 AND checkpoint_hash = ?2",
                params![row.checkpoint_sequence, witness.checkpoint_hash.as_slice()],
                |checkpoint_row| checkpoint_row.get(0),
            )
            .optional()?;
        let checkpoint = checkpoint_blob
            .as_deref()
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness references a missing checkpoint".to_string(),
                )
            })
            .and_then(decode_observation_checkpoint)?;
        if checkpoint.observer != witness.observer
            || checkpoint.sequence != witness.sequence
            || checkpoint.hash() != witness.checkpoint_hash
            || witness.response_timestamp < checkpoint.observed_at
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness does not bind its retained checkpoint".to_string(),
            ));
        }
        Ok(())
    }
}
