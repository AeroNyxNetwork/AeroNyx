// [ARCH-SPLIT 2026-10-02]
// Checkpoint append, load, and row verification.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Signs and appends one complete configured-producer observation.
    ///
    /// The transaction refuses partial, empty, or quarantined producer sets.
    /// If the exact overlap root is already the checkpoint tip, the operation
    /// is idempotent and returns the existing tip without another write.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the configured producer set
    /// is invalid or incomplete, the signing identity differs from replica
    /// metadata, the local clock regresses, recomputation fails, or `SQLite`
    /// cannot atomically append the checkpoint.
    pub fn append_observation_checkpoint(
        &self,
        configured_producers: &[[u8; 32]],
        identity: &IdentityKeyPair,
        observed_at: u64,
    ) -> Result<DirectoryObservationCheckpointAppendReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id || observed_at == 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint identity or timestamp is invalid".to_string(),
            ));
        }
        let configured = self.validate_convergence_producers(configured_producers)?;
        if configured.len() < 2 || configured.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint requires two to sixteen configured producers".to_string(),
            ));
        }

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        let mut convergence = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: u64::try_from(configured.len()).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint producer count exceeds u64".to_string(),
                )
            })?,
            window_blocks: DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS,
            ..DirectoryReplicaObservationConvergenceSnapshot::default()
        };
        let eligible_tips =
            Self::eligible_convergence_tips(&transaction, &configured, &mut convergence)?;
        if eligible_tips.len() != configured.len() {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint requires every configured producer to be eligible"
                    .to_string(),
            ));
        }
        let occurrences =
            Self::recent_commitment_occurrences(&transaction, &eligible_tips, &mut convergence)?;
        Self::complete_observation_convergence(&mut convergence, &eligible_tips, &occurrences)?;
        let observation_root = convergence.observation_root.ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint overlap root is unavailable".to_string(),
            )
        })?;
        let previous = Self::load_observation_checkpoint_tip(&transaction)?;
        if previous.sequence > 0 && previous.observation_root == observation_root {
            transaction.commit()?;
            drop(connection);
            return Ok(Self::observation_checkpoint_report(false, previous));
        }
        let report = Self::insert_observation_checkpoint(
            &transaction,
            previous,
            &eligible_tips,
            observation_root,
            identity,
            observed_at,
        )?;
        transaction.commit()?;
        drop(connection);
        Ok(report)
    }

    /// Returns the latest checkpoint after re-auditing the complete local
    /// checkpoint chain and every referenced producer prefix.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when metadata, any historical
    /// checkpoint, its signature/link, or a retained producer prefix is invalid.
    pub fn latest_audited_observation_checkpoint(
        &self,
        observed_at: u64,
    ) -> Result<Option<DirectoryObservationCheckpointV1>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let (count, tip) =
            Self::audit_observation_checkpoints(&connection, &self.local_node_id, observed_at)?;
        if count == 0 {
            return Ok(None);
        }
        let checkpoint_blob: Vec<u8> = connection.query_row(
            "SELECT checkpoint_blob FROM directory_observation_checkpoints
             WHERE sequence = ?1 AND checkpoint_hash = ?2",
            params![
                u64_to_i64(tip.sequence, "observation checkpoint sequence")?,
                tip.checkpoint_hash.as_slice()
            ],
            |row| row.get(0),
        )?;
        drop(connection);
        Ok(Some(decode_observation_checkpoint(&checkpoint_blob)?))
    }

    /// Returns the newest mature checkpoint that still lacks a witness receipt.
    ///
    /// Selection is forward-only. The lower sequence bound is the newest
    /// authenticated witness receipt or durable outcome round, whichever is
    /// greater. This prevents a restart or already-witnessed head from causing
    /// the coordinator to work backwards through historical gaps.
    ///
    /// The complete retained history is audited at startup and by [`Self::audit`].
    /// Recurring selection keeps work independent of history length: it verifies
    /// the signed candidate and predecessor rows, recomputes the candidate root,
    /// verifies the bounded latest receipt set, and validates the durable outcome
    /// checkpoint. Every row that can move or satisfy the selection floor is
    /// therefore authenticated before the indexed candidate query is trusted.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the maturity cutoff is zero
    /// or in the future, metadata or bounded selection evidence fails audit, the
    /// selected row is malformed, or `SQLite` cannot complete the indexed query.
    pub fn latest_audited_mature_unwitnessed_observation_checkpoint(
        &self,
        matured_before: u64,
        observed_at: u64,
    ) -> Result<Option<DirectoryObservationCheckpointV1>, DirectoryReplicaStoreError> {
        self.audited_mature_observation_checkpoint_below_witness_threshold_internal(
            matured_before,
            observed_at,
            1,
            None,
        )
        .map(|target| target.map(|target| target.checkpoint))
    }

    /// Returns the next forward mature checkpoint below the configured
    /// independent receipt target among current operator-pinned witnesses.
    ///
    /// Existing receipts from removed pins remain fully audited historical
    /// evidence but do not satisfy the current operational threshold. The
    /// returned witness identities let the coordinator skip duplicate requests.
    /// This is corroboration evidence only, never consensus or finality.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the witness set is empty,
    /// duplicated, larger than the protocol bound, the threshold is outside the
    /// witness set, or any bounded selection evidence fails verification.
    pub fn next_audited_mature_observation_checkpoint_below_witness_threshold(
        &self,
        matured_before: u64,
        observed_at: u64,
        minimum_witnesses: usize,
        eligible_witnesses: &[[u8; 32]],
    ) -> Result<Option<DirectoryObservationWitnessTarget>, DirectoryReplicaStoreError> {
        self.audited_mature_observation_checkpoint_below_witness_threshold_internal(
            matured_before,
            observed_at,
            minimum_witnesses,
            Some(eligible_witnesses),
        )
    }

    pub(super) fn audited_mature_observation_checkpoint_below_witness_threshold_internal(
        &self,
        matured_before: u64,
        observed_at: u64,
        minimum_witnesses: usize,
        eligible_witnesses: Option<&[[u8; 32]]>,
    ) -> Result<Option<DirectoryObservationWitnessTarget>, DirectoryReplicaStoreError> {
        if matured_before == 0 || matured_before > observed_at {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness maturity cutoff is invalid".to_string(),
            ));
        }
        if minimum_witnesses == 0 || minimum_witnesses > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness threshold is outside the protocol bound".to_string(),
            ));
        }
        let eligible_witness_set = eligible_witnesses
            .map(|witnesses| {
                if minimum_witnesses > witnesses.len() {
                    return Err(DirectoryReplicaStoreError::Request(
                        "observation witness threshold exceeds its eligibility set".to_string(),
                    ));
                }
                Self::validate_observation_witness_eligibility(witnesses)
            })
            .transpose()?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let checkpoint_tip = Self::load_observation_checkpoint_tip(&connection)?;
        if checkpoint_tip.sequence == 0 {
            return Ok(None);
        }
        let latest_witness_set = Self::latest_verified_observation_witness_set(
            &connection,
            &self.local_node_id,
            observed_at,
        )?;
        let outcome_snapshot = Self::load_observation_witness_outcome_snapshot(&connection)?;
        if outcome_snapshot.last_checkpoint_sequence > checkpoint_tip.sequence {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome references an unknown checkpoint".to_string(),
            ));
        }
        if outcome_snapshot.last_checkpoint_sequence > 0 {
            Self::verify_observation_checkpoint_at_sequence(
                &connection,
                &self.local_node_id,
                observed_at,
                outcome_snapshot.last_checkpoint_sequence,
            )?;
        }
        let minimum_sequence = latest_witness_set
            .sequence
            .max(outcome_snapshot.last_checkpoint_sequence)
            .max(1);
        let mut query_parameters = vec![
            Value::Integer(u64_to_i64(
                matured_before,
                "observation witness maturity cutoff",
            )?),
            Value::Integer(u64_to_i64(
                minimum_sequence,
                "observation witness sequence floor",
            )?),
        ];
        let threshold_mode = eligible_witnesses.is_some();
        let receipt_scope = if let Some(eligible_witnesses) = eligible_witnesses {
            let placeholders = std::iter::repeat_n("?", eligible_witnesses.len())
                .collect::<Vec<_>>()
                .join(", ");
            query_parameters.extend(
                eligible_witnesses
                    .iter()
                    .map(|witness| Value::Blob(witness.to_vec())),
            );
            format!(" AND witness.witness_node_id IN ({placeholders})")
        } else {
            String::new()
        };
        query_parameters.push(Value::Integer(i64::try_from(minimum_witnesses).map_err(
            |_| {
                DirectoryReplicaStoreError::Request(
                    "observation witness threshold exceeds i64".to_string(),
                )
            },
        )?));
        // Threshold collection finishes the current forward-floor checkpoint
        // before advancing. The compatibility wrapper keeps the historical
        // newest-unwitnessed behavior when no eligible pin set is supplied.
        let candidate_order = if threshold_mode { "ASC" } else { "DESC" };
        let checkpoint_query = format!(
            "SELECT checkpoint.sequence
             FROM directory_observation_checkpoints AS checkpoint
             WHERE checkpoint.observed_at <= ?
               AND checkpoint.sequence >= ?
               AND (
                   SELECT COUNT(*)
                   FROM directory_observation_checkpoint_witnesses AS witness
                   WHERE witness.checkpoint_sequence = checkpoint.sequence{receipt_scope}
               ) < ?
             ORDER BY checkpoint.sequence {candidate_order}
             LIMIT 1"
        );
        let checkpoint_sequence = connection
            .query_row(
                &checkpoint_query,
                params_from_iter(query_parameters.iter()),
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        let checkpoint = checkpoint_sequence
            .map(|sequence| {
                let sequence =
                    positive_i64_to_u64(sequence, "observation witness candidate sequence")?;
                Self::verify_observation_checkpoint_at_sequence(
                    &connection,
                    &self.local_node_id,
                    observed_at,
                    sequence,
                )
            })
            .transpose()?;
        let target = checkpoint.map(|checkpoint| {
            let witnessed_by = if checkpoint.sequence == latest_witness_set.sequence {
                latest_witness_set
                    .witness_node_ids
                    .iter()
                    .copied()
                    .filter(|witness| {
                        eligible_witness_set
                            .as_ref()
                            .is_none_or(|eligible| eligible.contains(witness))
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            DirectoryObservationWitnessTarget {
                checkpoint,
                witnessed_by,
                minimum_witnesses,
            }
        });
        drop(connection);
        if target
            .as_ref()
            .is_some_and(|target| target.checkpoint.observed_at > matured_before)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness candidate violates maturity cutoff".to_string(),
            ));
        }
        if target
            .as_ref()
            .is_some_and(|target| target.witnessed_by.len() >= minimum_witnesses)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness candidate already satisfies its threshold".to_string(),
            ));
        }
        Ok(target)
    }

    pub(super) fn insert_observation_checkpoint(
        transaction: &Transaction<'_>,
        previous: ObservationCheckpointTip,
        eligible_tips: &[DirectoryReplicaTip],
        observation_root: [u8; 32],
        identity: &IdentityKeyPair,
        observed_at: u64,
    ) -> Result<DirectoryObservationCheckpointAppendReport, DirectoryReplicaStoreError> {
        if observed_at < previous.observed_at {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint timestamp regressed".to_string(),
            ));
        }
        let sequence = previous.sequence.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint sequence exhausted".to_string(),
            )
        })?;
        let producer_count = u16::try_from(eligible_tips.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint producer count exceeds u16".to_string(),
            )
        })?;
        let producer_tips = eligible_tips
            .iter()
            .map(|tip| DirectoryObservationTipV1 {
                producer: tip.producer,
                tip_height: tip.tip_height,
                tip_hash: tip.tip_hash,
            })
            .collect();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            sequence,
            observed_at,
            previous.checkpoint_hash,
            producer_count,
            producer_tips,
            observation_root,
            identity,
        )
        .map_err(|error| DirectoryReplicaStoreError::Integrity(error.to_string()))?;
        let checkpoint_hash = checkpoint.hash();
        let checkpoint_blob = encode_observation_checkpoint(&checkpoint)?;
        transaction.execute(
            "INSERT INTO directory_observation_checkpoints
                (sequence, checkpoint_hash, previous_checkpoint_hash, observed_at,
                 observation_root, producer_count, checkpoint_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                u64_to_i64(sequence, "observation checkpoint sequence")?,
                checkpoint_hash.as_slice(),
                checkpoint.previous_checkpoint_hash.as_slice(),
                u64_to_i64(observed_at, "observation checkpoint timestamp")?,
                observation_root.as_slice(),
                i64::from(producer_count),
                checkpoint_blob,
            ],
        )?;
        Ok(DirectoryObservationCheckpointAppendReport {
            appended: true,
            sequence,
            checkpoint_hash,
            observed_at,
            producer_count,
            observation_root,
        })
    }

    pub(super) const fn observation_checkpoint_report(
        appended: bool,
        tip: ObservationCheckpointTip,
    ) -> DirectoryObservationCheckpointAppendReport {
        DirectoryObservationCheckpointAppendReport {
            appended,
            sequence: tip.sequence,
            checkpoint_hash: tip.checkpoint_hash,
            observed_at: tip.observed_at,
            producer_count: tip.producer_count,
            observation_root: tip.observation_root,
        }
    }

    pub(super) fn load_observation_checkpoint_tip(
        connection: &Connection,
    ) -> Result<ObservationCheckpointTip, DirectoryReplicaStoreError> {
        let row = connection
            .query_row(
                "SELECT sequence, checkpoint_hash, observed_at, producer_count,
                        observation_root
                 FROM directory_observation_checkpoints
                 ORDER BY sequence DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((sequence, checkpoint_hash, observed_at, producer_count, observation_root)) = row
        else {
            return Ok(ObservationCheckpointTip::default());
        };
        let producer_count = u16::try_from(producer_count).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint producer count exceeds u16".to_string(),
            )
        })?;
        if usize::from(producer_count) < 2
            || usize::from(producer_count) > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint producer count is invalid".to_string(),
            ));
        }
        Ok(ObservationCheckpointTip {
            sequence: positive_i64_to_u64(sequence, "observation checkpoint sequence")?,
            checkpoint_hash: bytes32(&checkpoint_hash, "observation checkpoint hash")?,
            observed_at: positive_i64_to_u64(observed_at, "observation checkpoint timestamp")?,
            producer_count,
            observation_root: bytes32(&observation_root, "observation checkpoint root")?,
        })
    }

    pub(super) fn load_observation_checkpoint_row(
        connection: &Connection,
        sequence: u64,
    ) -> Result<Option<StoredObservationCheckpointRow>, DirectoryReplicaStoreError> {
        connection
            .query_row(
                "SELECT sequence, checkpoint_hash, previous_checkpoint_hash, observed_at,
                        observation_root, producer_count, checkpoint_blob
                 FROM directory_observation_checkpoints WHERE sequence = ?1",
                params![u64_to_i64(sequence, "observation checkpoint sequence")?],
                |row| {
                    Ok(StoredObservationCheckpointRow {
                        sequence: row.get(0)?,
                        checkpoint_hash: row.get(1)?,
                        previous_checkpoint_hash: row.get(2)?,
                        observed_at: row.get(3)?,
                        observation_root: row.get(4)?,
                        producer_count: row.get(5)?,
                        checkpoint_blob: row.get(6)?,
                    })
                },
            )
            .optional()
            .map_err(DirectoryReplicaStoreError::from)
    }

    pub(super) fn decode_canonical_observation_checkpoint_row(
        row: &StoredObservationCheckpointRow,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let checkpoint = decode_observation_checkpoint(&row.checkpoint_blob)?;
        if encode_observation_checkpoint(&checkpoint)? != row.checkpoint_blob {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint encoding is not canonical".to_string(),
            ));
        }
        Ok(checkpoint)
    }

    pub(super) fn verify_observation_checkpoint_row_metadata(
        row: &StoredObservationCheckpointRow,
        checkpoint: &DirectoryObservationCheckpointV1,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        if checkpoint.observer != *local_node_id
            || positive_i64_to_u64(row.sequence, "observation checkpoint sequence")?
                != checkpoint.sequence
            || bytes32(&row.checkpoint_hash, "observation checkpoint hash")? != checkpoint.hash()
            || bytes32(
                &row.previous_checkpoint_hash,
                "observation checkpoint previous hash",
            )? != checkpoint.previous_checkpoint_hash
            || positive_i64_to_u64(row.observed_at, "observation checkpoint timestamp")?
                != checkpoint.observed_at
            || bytes32(&row.observation_root, "observation checkpoint root")?
                != checkpoint.observation_root
            || u16::try_from(row.producer_count).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint producer count exceeds u16".to_string(),
                )
            })? != checkpoint.configured_producer_count
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint row does not match its signed object".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn verify_observation_checkpoint_anchor_row(
        row: &StoredObservationCheckpointRow,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let checkpoint = Self::decode_canonical_observation_checkpoint_row(row)?;
        checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .map_err(|error| DirectoryReplicaStoreError::Integrity(error.to_string()))?;
        Self::verify_observation_checkpoint_row_metadata(row, &checkpoint, local_node_id)?;
        Ok(checkpoint)
    }

    pub(super) fn verify_observation_checkpoint_at_sequence(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
        sequence: u64,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let row =
            Self::load_observation_checkpoint_row(connection, sequence)?.ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint selection references a missing row".to_string(),
                )
            })?;
        let (previous_hash, previous_observed_at) = if sequence == 1 {
            ([0u8; 32], 0)
        } else {
            let previous_sequence = sequence.checked_sub(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint predecessor sequence underflow".to_string(),
                )
            })?;
            let previous_row =
                Self::load_observation_checkpoint_row(connection, previous_sequence)?.ok_or_else(
                    || {
                        DirectoryReplicaStoreError::Integrity(
                            "observation checkpoint predecessor is missing".to_string(),
                        )
                    },
                )?;
            let previous = Self::verify_observation_checkpoint_anchor_row(
                &previous_row,
                local_node_id,
                observed_at,
            )?;
            (previous.hash(), previous.observed_at)
        };
        Self::verify_observation_checkpoint_row(
            connection,
            local_node_id,
            observed_at,
            sequence,
            &previous_hash,
            previous_observed_at,
            &row,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_observation_checkpoint_row(
        connection: &Connection,
        local_node_id: &[u8; 32],
        verifier_observed_at: u64,
        expected_sequence: u64,
        expected_previous_hash: &[u8; 32],
        previous_observed_at: u64,
        row: &StoredObservationCheckpointRow,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let checkpoint = Self::decode_canonical_observation_checkpoint_row(row)?;
        checkpoint
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_sequence,
                expected_previous_hash,
                previous_observed_at,
                verifier_observed_at,
            )
            .map_err(|error| DirectoryReplicaStoreError::Integrity(error.to_string()))?;
        Self::verify_observation_checkpoint_row_metadata(row, &checkpoint, local_node_id)?;
        if Self::recompute_observation_checkpoint_root(connection, &checkpoint, local_node_id)?
            != checkpoint.observation_root
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint root does not match retained producer prefixes".to_string(),
            ));
        }
        Ok(checkpoint)
    }

    pub(super) fn recompute_observation_checkpoint_root(
        connection: &Connection,
        checkpoint: &DirectoryObservationCheckpointV1,
        local_node_id: &[u8; 32],
    ) -> Result<[u8; 32], DirectoryReplicaStoreError> {
        let mut tips = Vec::with_capacity(checkpoint.producer_tips.len());
        for observed_tip in &checkpoint.producer_tips {
            if Self::observation_block_hash_at(
                connection,
                local_node_id,
                &observed_tip.producer,
                observed_tip.tip_height,
            )? != Some(observed_tip.tip_hash)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint references a missing producer prefix".to_string(),
                ));
            }
            tips.push(DirectoryReplicaTip {
                producer: observed_tip.producer,
                tip_height: observed_tip.tip_height,
                tip_hash: observed_tip.tip_hash,
                tip_timestamp: 0,
                quarantined: false,
                quarantine_kind: None,
                active_incident_digest: None,
                last_resolution_digest: None,
            });
        }
        let mut snapshot = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: u64::from(checkpoint.configured_producer_count),
            eligible_producers: u64::from(checkpoint.configured_producer_count),
            window_blocks: DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS,
            ..DirectoryReplicaObservationConvergenceSnapshot::default()
        };
        let occurrences = Self::recent_commitment_occurrences_with_local_producer(
            connection,
            local_node_id,
            &tips,
            &mut snapshot,
        )?;
        Ok(observation_convergence_root(&tips, &occurrences))
    }

    pub(super) fn observation_block_hash_at(
        connection: &Connection,
        local_node_id: &[u8; 32],
        producer: &[u8; 32],
        height: u64,
    ) -> Result<Option<[u8; 32]>, DirectoryReplicaStoreError> {
        if producer != local_node_id {
            return Self::block_hash_at(connection, producer, height);
        }
        let block_hash: Option<Vec<u8>> = connection
            .query_row(
                "SELECT block_hash FROM directory_chain_blocks WHERE height = ?1",
                params![u64_to_i64(height, "local observation block height")?],
                |row| row.get(0),
            )
            .optional()?;
        block_hash
            .as_deref()
            .map(|hash| bytes32(hash, "local observation block hash"))
            .transpose()
    }

    pub(super) fn audit_observation_checkpoints(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<(u64, ObservationCheckpointTip), DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT sequence, checkpoint_hash, previous_checkpoint_hash, observed_at,
                    observation_root, producer_count, checkpoint_blob
             FROM directory_observation_checkpoints ORDER BY sequence ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredObservationCheckpointRow {
                sequence: row.get(0)?,
                checkpoint_hash: row.get(1)?,
                previous_checkpoint_hash: row.get(2)?,
                observed_at: row.get(3)?,
                observation_root: row.get(4)?,
                producer_count: row.get(5)?,
                checkpoint_blob: row.get(6)?,
            })
        })?;
        let mut count = 0u64;
        let mut expected_sequence = 1u64;
        let mut previous_hash = [0u8; 32];
        let mut previous_observed_at = 0u64;
        let mut tip = ObservationCheckpointTip::default();
        for row in rows {
            let row = row?;
            let checkpoint = Self::verify_observation_checkpoint_row(
                connection,
                local_node_id,
                observed_at,
                expected_sequence,
                &previous_hash,
                previous_observed_at,
                &row,
            )?;
            let checkpoint_hash = checkpoint.hash();
            tip = ObservationCheckpointTip {
                sequence: checkpoint.sequence,
                checkpoint_hash,
                observed_at: checkpoint.observed_at,
                producer_count: checkpoint.configured_producer_count,
                observation_root: checkpoint.observation_root,
            };
            previous_hash = checkpoint_hash;
            previous_observed_at = checkpoint.observed_at;
            count = count.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint count exceeds u64".to_string(),
                )
            })?;
            expected_sequence = expected_sequence.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint sequence exhausted".to_string(),
                )
            })?;
        }
        drop(statement);
        Ok((count, tip))
    }
}
