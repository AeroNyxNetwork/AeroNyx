// [ARCH-SPLIT 2026-10-02]
// Quarantine, signed resolution, and resolution audit.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Applies one signed, host-local compare-and-swap quarantine resolution.
    ///
    /// This operation only resumes synchronization from the already accepted
    /// prefix. It never deletes an incident, rewinds a block, selects a remote
    /// fork, or accepts unaudited content. The signed resolution is inserted
    /// atomically before the active incident flag is cleared.
    ///
    /// # Security
    /// Callers must keep this method behind the host-local CLI boundary. It is
    /// intentionally not wired to any Axum router or peer protocol.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the signature, timestamp,
    /// incident, tip, quarantine kind, or linked prior resolution differs from
    /// the operator's signed compare-and-swap view.
    pub fn resolve_quarantine(
        &self,
        command: &DirectoryReplicaResolutionCommand,
        observed_at: u64,
    ) -> Result<DirectoryReplicaResolutionReport, DirectoryReplicaStoreError> {
        self.verify_resolution_command(command, observed_at)?;
        let resolution_digest = command.digest();
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        Self::validate_resolution_cas(&transaction, command)?;
        Self::persist_resolution(&transaction, command, &resolution_digest)?;
        transaction.commit()?;
        drop(connection);
        Ok(DirectoryReplicaResolutionReport {
            resolution_digest,
            command_id: command.command_id,
            producer: command.producer,
            retained_tip_height: command.expected_tip_height,
            retained_tip_hash: command.expected_tip_hash,
            resolved_at: command.resolved_at,
        })
    }

    pub(super) fn verify_resolution_command(
        &self,
        command: &DirectoryReplicaResolutionCommand,
        observed_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        command.validate_unsigned_fields()?;
        if command.resolver_node_id != self.local_node_id
            || command.producer == self.local_node_id
            || command.resolved_at.abs_diff(observed_at)
                > DIRECTORY_REPLICA_RESOLUTION_TIMESTAMP_SKEW_SECS
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution identity or timestamp is invalid".to_string(),
            ));
        }
        IdentityPublicKey::from_bytes(&command.resolver_node_id)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution identity is invalid".to_string(),
                )
            })?
            .verify(&command.signing_bytes(), &command.signature)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution signature is invalid".to_string(),
                )
            })
    }

    pub(super) fn validate_resolution_cas(
        transaction: &Transaction<'_>,
        command: &DirectoryReplicaResolutionCommand,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let tip = Self::load_tip(transaction, &command.producer)?;
        if !tip.quarantined
            || tip.active_incident_digest != Some(command.incident_digest)
            || tip.tip_height != command.expected_tip_height
            || tip.tip_hash != command.expected_tip_hash
            || tip.quarantine_kind.as_deref() != Some(command.expected_quarantine_kind.as_str())
            || tip.last_resolution_digest != command.previous_resolution_digest
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution compare-and-swap state is stale".to_string(),
            ));
        }

        let incident_observed_at = Self::resolution_incident_observed_at(transaction, command)?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution incident does not match quarantine".to_string(),
                )
            })?;
        if command.resolved_at < incident_observed_at {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution predates its incident".to_string(),
            ));
        }
        if let Some(previous_digest) = command.previous_resolution_digest {
            let previous_resolved_at = transaction
                .query_row(
                    "SELECT resolved_at FROM directory_replica_resolutions
                     WHERE resolution_digest = ?1 AND producer = ?2",
                    params![previous_digest.as_slice(), command.producer.as_slice()],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .ok_or_else(|| {
                    DirectoryReplicaStoreError::Integrity(
                        "directory replica resolution predecessor is unavailable".to_string(),
                    )
                })?;
            if positive_i64_to_u64(previous_resolved_at, "previous resolution timestamp")?
                > command.resolved_at
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution predates its predecessor".to_string(),
                ));
            }
        }
        let retained_hash = if command.expected_tip_height == 0 {
            Some([0u8; 32])
        } else {
            Self::block_hash_at(transaction, &command.producer, command.expected_tip_height)?
        };
        if retained_hash != Some(command.expected_tip_hash) {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution tip is not a retained block".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn resolution_incident_observed_at(
        connection: &Connection,
        command: &DirectoryReplicaResolutionCommand,
    ) -> Result<Option<u64>, DirectoryReplicaStoreError> {
        connection
            .query_row(
                "SELECT observed_at FROM directory_replica_incidents
                 WHERE incident_digest = ?1 AND producer = ?2
                   AND subject_node_id = ?2 AND kind = ?3",
                params![
                    command.incident_digest.as_slice(),
                    command.producer.as_slice(),
                    command.expected_quarantine_kind
                ],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .map(|value| positive_i64_to_u64(value, "resolution incident timestamp"))
            .transpose()
    }

    pub(super) fn persist_resolution(
        transaction: &Transaction<'_>,
        command: &DirectoryReplicaResolutionCommand,
        resolution_digest: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute(
            "INSERT INTO directory_replica_resolutions
                (resolution_digest, command_id, incident_digest, producer, action,
                 expected_tip_height, expected_tip_hash, expected_quarantine_kind,
                 previous_resolution_digest, resolved_at, resolver_node_id, signature)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                resolution_digest.as_slice(),
                command.command_id.as_slice(),
                command.incident_digest.as_slice(),
                command.producer.as_slice(),
                DIRECTORY_REPLICA_RESOLUTION_ACTION,
                u64_to_i64(command.expected_tip_height, "resolution tip height")?,
                command.expected_tip_hash.as_slice(),
                command.expected_quarantine_kind,
                command
                    .previous_resolution_digest
                    .as_ref()
                    .map(<[u8; 32]>::as_slice),
                u64_to_i64(command.resolved_at, "resolution timestamp")?,
                command.resolver_node_id.as_slice(),
                command.signature.as_slice(),
            ],
        )?;
        let changed = transaction.execute(
            "UPDATE directory_replica_chains
             SET quarantined = 0, quarantine_kind = NULL,
                 active_incident_digest = NULL, last_resolution_digest = ?3,
                 updated_at = ?4
             WHERE producer = ?1 AND quarantined = 1
               AND active_incident_digest = ?2
               AND tip_height = ?5 AND tip_hash = ?6 AND quarantine_kind = ?7
               AND ((?8 IS NULL AND last_resolution_digest IS NULL)
                    OR last_resolution_digest = ?8)",
            params![
                command.producer.as_slice(),
                command.incident_digest.as_slice(),
                resolution_digest.as_slice(),
                u64_to_i64(command.resolved_at, "resolution timestamp")?,
                u64_to_i64(command.expected_tip_height, "resolution tip height")?,
                command.expected_tip_hash.as_slice(),
                command.expected_quarantine_kind,
                command
                    .previous_resolution_digest
                    .as_ref()
                    .map(<[u8; 32]>::as_slice),
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution compare-and-swap update failed".to_string(),
            ));
        }
        Self::clear_retry_state(transaction, &command.producer)?;
        Ok(())
    }

    pub(super) fn persist_quarantine(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        incident: &QuarantineIncident<'_>,
        observed_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let digest = incident_digest(producer, producer, incident);
        Self::insert_incident(transaction, producer, producer, incident, observed_at)?;
        transaction.execute(
            "UPDATE directory_replica_chains
             SET quarantined = 1, quarantine_kind = ?2,
                 active_incident_digest = ?3, updated_at = ?4
             WHERE producer = ?1",
            params![
                producer.as_slice(),
                incident.kind,
                digest.as_slice(),
                u64_to_i64(observed_at, "quarantine timestamp")?
            ],
        )?;
        Ok(())
    }

    pub(super) fn insert_incident(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        subject_node_id: &[u8; 32],
        incident: &QuarantineIncident<'_>,
        observed_at: u64,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let digest = incident_digest(producer, subject_node_id, incident);
        let changed = transaction.execute(
            "INSERT OR IGNORE INTO directory_replica_incidents
                (incident_digest, producer, subject_node_id, kind, height,
                 local_hash, remote_hash, evidence_frame, observed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                digest.as_slice(),
                producer.as_slice(),
                subject_node_id.as_slice(),
                incident.kind,
                u64_to_i64(incident.height, "incident height")?,
                incident.local_hash.as_slice(),
                incident.remote_hash.as_slice(),
                incident.evidence_frame,
                u64_to_i64(observed_at, "incident timestamp")?
            ],
        )?;
        Ok(changed == 1)
    }

    pub(super) fn audit_incidents(
        connection: &Connection,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT incident_digest, producer, subject_node_id, kind, height,
                    local_hash, remote_hash, evidence_frame
             FROM directory_replica_incidents ORDER BY incident_digest ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Vec<u8>>(5)?,
                row.get::<_, Vec<u8>>(6)?,
                row.get::<_, Vec<u8>>(7)?,
            ))
        })?;
        let mut count = 0u64;
        for row in rows {
            let (digest, producer, subject, kind, height, local, remote, evidence) = row?;
            validate_incident_kind(&kind)?;
            if evidence.is_empty() || evidence.len() > MAX_DIRECTORY_SYNC_EVIDENCE_BYTES {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica incident metadata or evidence is invalid".to_string(),
                ));
            }
            let incident = QuarantineIncident {
                kind: &kind,
                height: nonnegative_i64_to_u64(height, "incident height")?,
                local_hash: bytes32(&local, "incident local hash")?,
                remote_hash: bytes32(&remote, "incident remote hash")?,
                evidence_frame: &evidence,
            };
            let producer = bytes32(&producer, "incident producer")?;
            let subject = bytes32(&subject, "incident subject")?;
            if bytes32(&digest, "incident digest")?
                != incident_digest(&producer, &subject, &incident)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica incident digest mismatch".to_string(),
                ));
            }
            verify_incident_response_evidence(&evidence, &producer)?;
            count = count.saturating_add(1);
        }
        Ok(count)
    }

    pub(super) fn audit_resolutions(
        connection: &Connection,
        local_node_id: &[u8; 32],
        tips: &[DirectoryReplicaTip],
    ) -> Result<u64, DirectoryReplicaStoreError> {
        let mut index = Self::load_verified_resolution_index(connection, local_node_id)?;
        let count = u64::try_from(index.commands.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "directory replica resolution count exceeds u64".to_string(),
            )
        })?;
        Self::audit_resolution_histories(connection, tips, &mut index)?;
        Ok(count)
    }

    pub(super) fn load_verified_resolution_index(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<AuditedResolutionIndex, DirectoryReplicaStoreError> {
        let rows = Self::load_resolution_rows(connection)?;
        let mut index = AuditedResolutionIndex {
            commands: HashMap::with_capacity(rows.len()),
            ..AuditedResolutionIndex::default()
        };
        for row in rows {
            let (digest, command) = Self::verify_resolution_row(connection, local_node_id, row)?;
            index
                .by_producer
                .entry(command.producer)
                .or_default()
                .insert(digest);
            index
                .resolved_incidents
                .entry(command.producer)
                .or_default()
                .insert(command.incident_digest);
            if index.commands.insert(digest, command).is_some() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "duplicate directory replica resolution digest".to_string(),
                ));
            }
        }
        Ok(index)
    }

    pub(super) fn load_resolution_rows(
        connection: &Connection,
    ) -> Result<Vec<StoredResolutionRow>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT resolution_digest, command_id, incident_digest, producer, action,
                    expected_tip_height, expected_tip_hash, expected_quarantine_kind,
                    previous_resolution_digest, resolved_at, resolver_node_id, signature
             FROM directory_replica_resolutions ORDER BY resolution_digest ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok(StoredResolutionRow {
                    digest: row.get(0)?,
                    command_id: row.get(1)?,
                    incident_digest: row.get(2)?,
                    producer: row.get(3)?,
                    action: row.get(4)?,
                    expected_tip_height: row.get(5)?,
                    expected_tip_hash: row.get(6)?,
                    expected_quarantine_kind: row.get(7)?,
                    previous_resolution_digest: row.get(8)?,
                    resolved_at: row.get(9)?,
                    resolver_node_id: row.get(10)?,
                    signature: row.get(11)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        Ok(rows)
    }

    pub(super) fn verify_resolution_row(
        connection: &Connection,
        local_node_id: &[u8; 32],
        row: StoredResolutionRow,
    ) -> Result<([u8; 32], DirectoryReplicaResolutionCommand), DirectoryReplicaStoreError> {
        if row.action != DIRECTORY_REPLICA_RESOLUTION_ACTION {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution action is invalid".to_string(),
            ));
        }
        let command = DirectoryReplicaResolutionCommand {
            command_id: bytes16(&row.command_id, "resolution command id")?,
            incident_digest: bytes32(&row.incident_digest, "resolution incident digest")?,
            producer: bytes32(&row.producer, "resolution producer")?,
            expected_tip_height: nonnegative_i64_to_u64(
                row.expected_tip_height,
                "resolution expected tip height",
            )?,
            expected_tip_hash: bytes32(&row.expected_tip_hash, "resolution expected tip hash")?,
            expected_quarantine_kind: row.expected_quarantine_kind,
            previous_resolution_digest: row
                .previous_resolution_digest
                .map(|value| bytes32(&value, "previous resolution digest"))
                .transpose()?,
            resolved_at: positive_i64_to_u64(row.resolved_at, "resolution timestamp")?,
            resolver_node_id: bytes32(&row.resolver_node_id, "resolution node identity")?,
            signature: bytes64(&row.signature, "resolution signature")?,
        };
        command.validate_unsigned_fields()?;
        if command.resolver_node_id != *local_node_id {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution belongs to another local node".to_string(),
            ));
        }
        IdentityPublicKey::from_bytes(&command.resolver_node_id)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution identity is invalid".to_string(),
                )
            })?
            .verify(&command.signing_bytes(), &command.signature)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution signature is invalid".to_string(),
                )
            })?;
        let digest = bytes32(&row.digest, "resolution digest")?;
        if digest != command.digest() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution digest mismatch".to_string(),
            ));
        }
        let incident_observed_at = Self::resolution_incident_observed_at(connection, &command)?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution references a mismatched incident".to_string(),
                )
            })?;
        if command.resolved_at < incident_observed_at {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution predates its incident".to_string(),
            ));
        }
        let retained_hash = if command.expected_tip_height == 0 {
            Some([0u8; 32])
        } else {
            Self::block_hash_at(connection, &command.producer, command.expected_tip_height)?
        };
        if retained_hash != Some(command.expected_tip_hash) {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution references a missing retained prefix".to_string(),
            ));
        }
        Ok((digest, command))
    }

    pub(super) fn audit_resolution_histories(
        connection: &Connection,
        tips: &[DirectoryReplicaTip],
        index: &mut AuditedResolutionIndex,
    ) -> Result<(), DirectoryReplicaStoreError> {
        for tip in tips {
            let mut pending = index.by_producer.remove(&tip.producer).unwrap_or_default();
            let mut cursor = tip.last_resolution_digest;
            if pending.is_empty() != cursor.is_none() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution head does not match its history".to_string(),
                ));
            }
            while let Some(digest) = cursor {
                if !pending.remove(&digest) {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "directory replica resolution history is missing, cyclic, or branched"
                            .to_string(),
                    ));
                }
                let command = index.commands.get(&digest).ok_or_else(|| {
                    DirectoryReplicaStoreError::Integrity(
                        "directory replica resolution head references a missing record".to_string(),
                    )
                })?;
                if command.producer != tip.producer {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "directory replica resolution history crosses producer namespaces"
                            .to_string(),
                    ));
                }
                if let Some(previous_digest) = command.previous_resolution_digest {
                    let previous = index.commands.get(&previous_digest).ok_or_else(|| {
                        DirectoryReplicaStoreError::Integrity(
                            "directory replica resolution predecessor is missing".to_string(),
                        )
                    })?;
                    if previous.producer != tip.producer
                        || previous.resolved_at > command.resolved_at
                    {
                        return Err(DirectoryReplicaStoreError::Integrity(
                            "directory replica resolution predecessor is incompatible".to_string(),
                        ));
                    }
                }
                cursor = command.previous_resolution_digest;
            }
            if !pending.is_empty() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "directory replica resolution history contains an orphaned branch".to_string(),
                ));
            }

            let mut incident_statement = connection.prepare(
                "SELECT incident_digest FROM directory_replica_incidents
                 WHERE producer = ?1 AND subject_node_id = ?1",
            )?;
            let incident_digests = incident_statement
                .query_map(params![tip.producer.as_slice()], |row| {
                    row.get::<_, Vec<u8>>(0)
                })?
                .collect::<Result<Vec<_>, _>>()?;
            drop(incident_statement);
            for incident_digest in incident_digests {
                let incident_digest = bytes32(&incident_digest, "producer incident digest")?;
                let has_resolution = index
                    .resolved_incidents
                    .get(&tip.producer)
                    .is_some_and(|digests| digests.contains(&incident_digest));
                if tip.active_incident_digest != Some(incident_digest) && !has_resolution {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "producer incident is neither active nor covered by signed resolution"
                            .to_string(),
                    ));
                }
            }
        }
        if !index.by_producer.is_empty() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution references a missing producer".to_string(),
            ));
        }
        Ok(())
    }
}
