// [ARCH-SPLIT 2026-10-02]
// Open the replica, then expose tip and mirror-registry reads.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Opens or creates replica tables and audits every accepted producer prefix.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for filesystem/SQLite failures,
    /// incompatible metadata, invalid signed blocks, malformed indexes, or
    /// invalid durable incident evidence.
    pub fn open(
        path: impl AsRef<Path>,
        local_node_id: [u8; 32],
        observed_at: u64,
    ) -> Result<(Self, DirectoryReplicaAudit), DirectoryReplicaStoreError> {
        if local_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "local node identity must not be the zero sentinel".to_string(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let mut connection = Connection::open(&path)?;
        connection.busy_timeout(DIRECTORY_REPLICA_BUSY_TIMEOUT)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        Self::initialize_schema(&mut connection, &local_node_id)?;
        let store = Self {
            connection: Mutex::new(connection),
            path,
            local_node_id,
        };
        let audit = store.audit(observed_at)?;
        Ok((store, audit))
    }

    /// Returns the shared Directory Chain `SQLite` path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns one producer's accepted prefix and quarantine state.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when a persisted row is malformed.
    pub fn producer_tip(
        &self,
        producer: &[u8; 32],
    ) -> Result<DirectoryReplicaTip, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::load_tip(&connection, producer)
    }

    /// Returns durable non-authoritative mirror resume cursors in stable order.
    ///
    /// [DIRECTORY-MIRROR-PROVENANCE 2026-09-01 by Codex] The discovery
    /// descriptor that originally admitted a mirror may expire while its
    /// producer-signed prefix remains useful. Preserve the last authenticated
    /// descriptor sequence so the scheduler can retry direct discovery and
    /// then current admitted carriers without inventing a new namespace.
    pub(crate) fn retained_mirror_cursors(
        &self,
    ) -> Result<Vec<DirectoryRetainedMirrorCursor>, DirectoryReplicaStoreError> {
        let rows = {
            let connection = self.connection.lock();
            Self::validate_metadata(&connection, &self.local_node_id)?;
            let mut statement = connection.prepare(
                "SELECT producer, descriptor_sequence
                 FROM directory_replica_mirror_producers
                 ORDER BY last_selected_at DESC, producer ASC",
            )?;
            let rows = statement
                .query_map([], |row| {
                    Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(statement);
            drop(connection);
            rows
        };
        rows.into_iter()
            .map(|(producer, descriptor_sequence)| {
                Ok(DirectoryRetainedMirrorCursor {
                    producer: bytes32(&producer, "directory mirror producer")?,
                    descriptor_sequence: positive_i64_to_u64(
                        descriptor_sequence,
                        "directory mirror descriptor sequence",
                    )?,
                })
            })
            .collect()
    }

    /// Returns durable non-authoritative mirror producer ids in stable order.
    ///
    /// Identities are for internal scheduling only and must not be exposed by
    /// public status. Registry membership grants no checkpoint or witness role.
    pub(crate) fn mirror_producer_ids(&self) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
        self.retained_mirror_cursors()
            .map(|cursors| cursors.into_iter().map(|cursor| cursor.producer).collect())
    }

    /// Verifies that the durable mirror registry fits an operator capacity.
    ///
    /// Capacity changes never delete signed history implicitly. Lowering the
    /// configured ceiling below the retained count therefore fails startup and
    /// requires an explicit operator decision to raise the ceiling or migrate
    /// the store.
    pub(crate) fn ensure_mirror_capacity(
        &self,
        max_producers: usize,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !(1..=MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS).contains(&max_producers) {
            return Err(DirectoryReplicaStoreError::Request(
                "directory mirror capacity is outside protocol bounds".to_string(),
            ));
        }
        let retained: i64 = self.connection.lock().query_row(
            "SELECT COUNT(*) FROM directory_replica_mirror_producers",
            [],
            |row| row.get(0),
        )?;
        let retained = usize::try_from(nonnegative_i64_to_u64(
            retained,
            "directory mirror capacity count",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "directory mirror capacity count exceeds usize".to_string(),
            )
        })?;
        if retained > max_producers {
            return Err(DirectoryReplicaStoreError::MirrorCapacity);
        }
        Ok(())
    }

    /// Atomically promotes configured producers out of non-authoritative mirror
    /// classification before authority synchronization starts.
    pub(crate) fn promote_pinned_producers(
        &self,
        producers: &[[u8; 32]],
    ) -> Result<u64, DirectoryReplicaStoreError> {
        let mut unique = producers
            .iter()
            .copied()
            .filter(|producer| *producer != [0u8; 32] && *producer != self.local_node_id)
            .collect::<Vec<_>>();
        unique.sort_unstable();
        unique.dedup();
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut promoted = 0u64;
        for producer in unique {
            promoted = promoted.saturating_add(
                u64::try_from(transaction.execute(
                    "DELETE FROM directory_replica_mirror_producers WHERE producer = ?1",
                    params![producer.as_slice()],
                )?)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "directory mirror promotion count exceeds u64".to_string(),
                    )
                })?,
            );
        }
        transaction.commit()?;
        drop(connection);
        Ok(promoted)
    }
}
