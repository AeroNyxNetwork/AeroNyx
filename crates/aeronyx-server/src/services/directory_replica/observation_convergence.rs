// [ARCH-SPLIT 2026-10-02]
// Recent-window commitment overlap and observation root.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Computes bounded recent commitment overlap for configured producers.
    ///
    /// Only non-empty, non-quarantined producer prefixes are eligible. The
    /// returned root binds the exact eligible producer tips and commitment
    /// hashes observed by every eligible source inside the recent block
    /// window. It is a local evidence digest, not a signature, vote, quorum,
    /// fork choice, consensus result, or finalized checkpoint.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the producer set exceeds
    /// the configured protocol bound, contains invalid identities, persisted
    /// rows are malformed, or `SQLite` cannot complete a bounded query.
    pub fn observation_convergence(
        &self,
        configured_producers: &[[u8; 32]],
    ) -> Result<DirectoryReplicaObservationConvergenceSnapshot, DirectoryReplicaStoreError> {
        let configured = self.validate_convergence_producers(configured_producers)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let snapshot = Self::observation_convergence_from_connection(&connection, &configured);
        drop(connection);
        snapshot
    }

    pub(super) fn validate_convergence_producers(
        &self,
        configured_producers: &[[u8; 32]],
    ) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
        if configured_producers.len() > MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS {
            return Err(DirectoryReplicaStoreError::Request(format!(
                "observation convergence supports at most {MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS} producers"
            )));
        }
        let mut configured = configured_producers.to_vec();
        configured.sort_unstable();
        if configured.windows(2).any(|values| values[0] == values[1]) {
            return Err(DirectoryReplicaStoreError::Request(
                "observation convergence producer identities must be unique".to_string(),
            ));
        }
        if configured
            .iter()
            .any(|producer| *producer == [0u8; 32] || producer == &self.local_node_id)
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation convergence producer identity is invalid".to_string(),
            ));
        }
        Ok(configured)
    }

    pub(super) fn observation_convergence_from_connection(
        connection: &Connection,
        configured: &[[u8; 32]],
    ) -> Result<DirectoryReplicaObservationConvergenceSnapshot, DirectoryReplicaStoreError> {
        let mut snapshot = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: u64::try_from(configured.len()).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation convergence producer count exceeds u64".to_string(),
                )
            })?,
            window_blocks: DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS,
            ..DirectoryReplicaObservationConvergenceSnapshot::default()
        };
        let eligible_tips = Self::eligible_convergence_tips(connection, configured, &mut snapshot)?;
        let occurrences =
            Self::recent_commitment_occurrences(connection, &eligible_tips, &mut snapshot)?;
        Self::complete_observation_convergence(&mut snapshot, &eligible_tips, &occurrences)?;
        Ok(snapshot)
    }

    pub(super) fn eligible_convergence_tips(
        connection: &Connection,
        configured: &[[u8; 32]],
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
    ) -> Result<Vec<DirectoryReplicaTip>, DirectoryReplicaStoreError> {
        let mut eligible_tips = Vec::with_capacity(configured.len());
        for producer in configured {
            let tip = Self::load_tip(connection, producer)?;
            if tip.quarantined {
                snapshot.excluded_quarantined_producers =
                    snapshot.excluded_quarantined_producers.saturating_add(1);
            } else if tip.tip_height == 0 {
                snapshot.pending_producers = snapshot.pending_producers.saturating_add(1);
            } else {
                eligible_tips.push(tip);
            }
        }
        snapshot.eligible_producers = u64::try_from(eligible_tips.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation convergence eligible count exceeds u64".to_string(),
            )
        })?;
        Ok(eligible_tips)
    }

    pub(super) fn recent_commitment_occurrences(
        connection: &Connection,
        eligible_tips: &[DirectoryReplicaTip],
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
    ) -> Result<BTreeMap<[u8; 32], u64>, DirectoryReplicaStoreError> {
        let mut occurrence_by_commitment = BTreeMap::<[u8; 32], u64>::new();
        for tip in eligible_tips {
            let first_height = tip
                .tip_height
                .saturating_sub(DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS.saturating_sub(1))
                .max(1);
            let first_height =
                u64_to_i64(first_height, "observation convergence first block height")?;
            let tip_height =
                u64_to_i64(tip.tip_height, "observation convergence tip block height")?;
            let mut statement = connection.prepare(
                "SELECT commitment_hash FROM directory_replica_commitments
                 WHERE producer = ?1 AND block_height BETWEEN ?2 AND ?3
                 ORDER BY commitment_hash ASC",
            )?;
            let hashes = statement
                .query_map(
                    params![tip.producer.as_slice(), first_height, tip_height],
                    |row| row.get::<_, Vec<u8>>(0),
                )?
                .collect::<Result<Vec<_>, _>>()?;
            let mut seen_for_producer = HashSet::with_capacity(hashes.len());
            for hash in hashes {
                let hash = bytes32(&hash, "observation convergence commitment hash")?;
                if !seen_for_producer.insert(hash) {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "observation convergence found a duplicate producer commitment".to_string(),
                    ));
                }
                snapshot.recent_commitments = snapshot.recent_commitments.saturating_add(1);
                let occurrence = occurrence_by_commitment.entry(hash).or_default();
                *occurrence = occurrence.saturating_add(1);
            }
        }
        Ok(occurrence_by_commitment)
    }

    pub(super) fn recent_commitment_occurrences_with_local_producer(
        connection: &Connection,
        local_node_id: &[u8; 32],
        eligible_tips: &[DirectoryReplicaTip],
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
    ) -> Result<BTreeMap<[u8; 32], u64>, DirectoryReplicaStoreError> {
        let mut occurrence_by_commitment = BTreeMap::<[u8; 32], u64>::new();
        for tip in eligible_tips {
            let first_height = tip
                .tip_height
                .saturating_sub(DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS.saturating_sub(1))
                .max(1);
            let first_height = u64_to_i64(first_height, "observation witness first block height")?;
            let tip_height = u64_to_i64(tip.tip_height, "observation witness tip block height")?;
            let hashes = if tip.producer == *local_node_id {
                let mut statement = connection.prepare(
                    "SELECT commitment_hash FROM directory_chain_commitments
                     WHERE block_height BETWEEN ?1 AND ?2
                     ORDER BY commitment_hash ASC",
                )?;
                let rows = statement.query_map(params![first_height, tip_height], |row| {
                    row.get::<_, Vec<u8>>(0)
                })?;
                rows.collect::<Result<Vec<_>, _>>()?
            } else {
                let mut statement = connection.prepare(
                    "SELECT commitment_hash FROM directory_replica_commitments
                     WHERE producer = ?1 AND block_height BETWEEN ?2 AND ?3
                     ORDER BY commitment_hash ASC",
                )?;
                let rows = statement.query_map(
                    params![tip.producer.as_slice(), first_height, tip_height],
                    |row| row.get::<_, Vec<u8>>(0),
                )?;
                rows.collect::<Result<Vec<_>, _>>()?
            };
            let mut seen_for_producer = HashSet::with_capacity(hashes.len());
            for hash in hashes {
                let hash = bytes32(&hash, "observation witness commitment hash")?;
                if !seen_for_producer.insert(hash) {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "observation witness found a duplicate producer commitment".to_string(),
                    ));
                }
                snapshot.recent_commitments = snapshot.recent_commitments.saturating_add(1);
                let occurrence = occurrence_by_commitment.entry(hash).or_default();
                *occurrence = occurrence.saturating_add(1);
            }
        }
        Ok(occurrence_by_commitment)
    }

    pub(super) fn complete_observation_convergence(
        snapshot: &mut DirectoryReplicaObservationConvergenceSnapshot,
        eligible_tips: &[DirectoryReplicaTip],
        occurrence_by_commitment: &BTreeMap<[u8; 32], u64>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        snapshot.distinct_recent_commitments = u64::try_from(occurrence_by_commitment.len())
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation convergence commitment count exceeds u64".to_string(),
                )
            })?;
        if snapshot.eligible_producers >= 2 {
            snapshot.multi_source_recent_commitments = occurrence_by_commitment
                .values()
                .filter(|occurrence| **occurrence >= 2)
                .count()
                .try_into()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation convergence multi-source count exceeds u64".to_string(),
                    )
                })?;
            snapshot.all_eligible_source_recent_commitments = occurrence_by_commitment
                .values()
                .filter(|occurrence| **occurrence == snapshot.eligible_producers)
                .count()
                .try_into()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation convergence all-source count exceeds u64".to_string(),
                    )
                })?;
            snapshot.observation_root = Some(observation_convergence_root(
                eligible_tips,
                occurrence_by_commitment,
            ));
        }
        Ok(())
    }
}
