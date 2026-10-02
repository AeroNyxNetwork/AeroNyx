// [ARCH-SPLIT 2026-10-02]
// Producer-local retry boundaries.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Returns all audited restart-durable producer retry states.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when metadata or any bounded
    /// retry-state row is malformed.
    pub fn retry_states(
        &self,
    ) -> Result<Vec<DirectoryReplicaRetryState>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        Self::load_retry_states(&connection, &self.local_node_id)
    }

    /// Persists one producer failure before exposing its retry boundary.
    ///
    /// The failure reason must be a stable ASCII bucket. Peer-controlled error
    /// text, endpoints, response bodies, and payloads are rejected.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid bounded state or a
    /// failed atomic `SQLite` transaction.
    pub fn persist_retry_failure(
        &self,
        producer: [u8; 32],
        consecutive_failures: u64,
        retry_not_before: Option<u64>,
        last_failure_at: u64,
        last_failure_reason: &str,
    ) -> Result<(), DirectoryReplicaStoreError> {
        validate_retry_state_fields(
            &producer,
            &self.local_node_id,
            consecutive_failures,
            retry_not_before,
            last_failure_at,
            last_failure_reason,
        )
        .map_err(|reason| DirectoryReplicaStoreError::Request(reason.to_string()))?;
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::ensure_producer_row(&transaction, &producer, last_failure_at)?;
        transaction.execute(
            "INSERT INTO directory_replica_retry_state
                (producer, consecutive_failures, retry_not_before,
                 last_failure_at, last_failure_reason, backoff_skips, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 0, ?4)
             ON CONFLICT(producer) DO UPDATE SET
                 consecutive_failures = excluded.consecutive_failures,
                 retry_not_before = CASE
                     WHEN excluded.last_failure_at >= directory_replica_retry_state.last_failure_at
                     THEN excluded.retry_not_before
                     ELSE directory_replica_retry_state.retry_not_before
                 END,
                 last_failure_at = MAX(
                     directory_replica_retry_state.last_failure_at,
                     excluded.last_failure_at
                 ),
                 last_failure_reason = CASE
                     WHEN excluded.last_failure_at >= directory_replica_retry_state.last_failure_at
                     THEN excluded.last_failure_reason
                     ELSE directory_replica_retry_state.last_failure_reason
                 END,
                 updated_at = MAX(
                     directory_replica_retry_state.updated_at,
                     excluded.updated_at
                 )",
            params![
                producer.as_slice(),
                u64_to_i64(consecutive_failures, "replica retry consecutive failures")?,
                retry_not_before
                    .map(|value| u64_to_i64(value, "replica retry boundary"))
                    .transpose()?,
                u64_to_i64(last_failure_at, "replica retry failure timestamp")?,
                last_failure_reason,
            ],
        )?;
        transaction.commit()?;
        drop(connection);
        Ok(())
    }

    /// Persists one scheduled round skipped by an active retry boundary.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the expected active durable
    /// retry row is missing or `SQLite` cannot commit the update.
    pub fn persist_retry_skip(
        &self,
        producer: [u8; 32],
        skipped_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if producer == [0u8; 32] || producer == self.local_node_id || skipped_at == 0 {
            return Err(DirectoryReplicaStoreError::Request(
                "replica retry skip fields are invalid".to_string(),
            ));
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let changed = transaction.execute(
            "UPDATE directory_replica_retry_state
             SET backoff_skips = CASE
                     WHEN backoff_skips < 9223372036854775807
                     THEN backoff_skips + 1
                     ELSE backoff_skips
                 END,
                 updated_at = MAX(updated_at, ?2)
             WHERE producer = ?1
               AND retry_not_before IS NOT NULL
               AND retry_not_before > ?2",
            params![
                producer.as_slice(),
                u64_to_i64(skipped_at, "replica retry skip timestamp")?
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "active replica retry state is missing during skip".to_string(),
            ));
        }
        transaction.commit()?;
        drop(connection);
        Ok(())
    }

    pub(super) fn clear_retry_state(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute(
            "DELETE FROM directory_replica_retry_state WHERE producer = ?1",
            params![producer.as_slice()],
        )?;
        Ok(())
    }

    pub(super) fn load_retry_states(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<Vec<DirectoryReplicaRetryState>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT producer, consecutive_failures, retry_not_before,
                    last_failure_at, last_failure_reason, backoff_skips, updated_at
             FROM directory_replica_retry_state
             ORDER BY producer ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        rows.into_iter()
            .map(
                |(
                    producer,
                    consecutive_failures,
                    retry_not_before,
                    last_failure_at,
                    last_failure_reason,
                    backoff_skips,
                    updated_at,
                )| {
                    let producer = bytes32(&producer, "replica retry producer")?;
                    let consecutive_failures = nonnegative_i64_to_u64(
                        consecutive_failures,
                        "replica retry consecutive failures",
                    )?;
                    let retry_not_before = retry_not_before
                        .map(|value| nonnegative_i64_to_u64(value, "replica retry boundary"))
                        .transpose()?;
                    let last_failure_at =
                        nonnegative_i64_to_u64(last_failure_at, "replica retry failure timestamp")?;
                    validate_retry_state_fields(
                        &producer,
                        local_node_id,
                        consecutive_failures,
                        retry_not_before,
                        last_failure_at,
                        &last_failure_reason,
                    )
                    .map_err(|reason| DirectoryReplicaStoreError::Integrity(reason.to_string()))?;
                    let updated_at =
                        nonnegative_i64_to_u64(updated_at, "replica retry update timestamp")?;
                    if updated_at < last_failure_at {
                        return Err(DirectoryReplicaStoreError::Integrity(
                            "replica retry update timestamp predates failure".to_string(),
                        ));
                    }
                    Ok(DirectoryReplicaRetryState {
                        producer,
                        consecutive_failures,
                        retry_not_before,
                        last_failure_at,
                        last_failure_reason,
                        backoff_skips: nonnegative_i64_to_u64(
                            backoff_skips,
                            "replica retry skip count",
                        )?,
                    })
                },
            )
            .collect()
    }
}
