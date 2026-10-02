// [ARCH-SPLIT 2026-10-02]
// Observation certificate import and its startup audit.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Exports the latest witnessed checkpoint as a portable certificate when
    /// the current operator-pinned witness set still satisfies the requested
    /// threshold.
    ///
    /// [PORTABLE-OBSERVATION-CERTIFICATE 2026-07-26 by Codex] Every read
    /// re-verifies the retained checkpoint, canonical response frames, witness
    /// signatures, checkpoint bindings, current pin membership, and threshold.
    /// Historical receipts from removed pins remain durable but are excluded.
    ///
    /// This evidence package is deliberately not a vote, validator set, quorum
    /// certificate, fork choice, consensus statement, or finality proof.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when pins or threshold are
    /// invalid, persisted evidence fails audit, certificate construction fails,
    /// or `SQLite` cannot complete the bounded read.
    pub fn latest_observation_certificate_for_pins(
        &self,
        eligible_witnesses: &[[u8; 32]],
        minimum_witnesses: usize,
        observed_at: u64,
    ) -> Result<Option<DirectoryObservationCertificateV1>, DirectoryReplicaStoreError> {
        if minimum_witnesses == 0
            || minimum_witnesses > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
            || minimum_witnesses > eligible_witnesses.len()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "portable observation certificate threshold is invalid".to_string(),
            ));
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
        if latest.sequence == 0 {
            return Ok(None);
        }
        let receipts = latest
            .receipts
            .into_iter()
            .filter(|receipt| eligible_witnesses.contains(&receipt.responder))
            .collect::<Vec<_>>();
        if receipts.len() < minimum_witnesses {
            return Ok(None);
        }
        let checkpoint = Self::verify_observation_checkpoint_at_sequence(
            &connection,
            &self.local_node_id,
            observed_at,
            latest.sequence,
        )?;
        drop(connection);
        let minimum_witnesses = u16::try_from(minimum_witnesses).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "portable observation certificate threshold exceeds u16".to_string(),
            )
        })?;
        DirectoryObservationCertificateV1::new_verified(
            checkpoint,
            minimum_witnesses,
            receipts,
            observed_at,
        )
        .map(Some)
        .map_err(|error| {
            DirectoryReplicaStoreError::Integrity(format!(
                "portable observation certificate failed verification: {error}"
            ))
        })
    }

    /// Imports one externally observed certificate into the local signed log.
    ///
    /// The operation is host-local, bounded, append-only, and idempotent for
    /// the exact same certificate and policy. A rollback or conflicting
    /// checkpoint for the same observer is rejected before any row changes.
    /// Imported evidence never affects producer selection, fork choice,
    /// routing, authorization, consensus, or finality.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when verification or pins fail,
    /// the observer attempts a rollback/equivocation, capacity is exhausted,
    /// existing history fails audit, or the metadata compare-and-swap fails.
    pub fn import_observation_certificate(
        &self,
        frame: &[u8],
        expected_sha256: &[u8; 32],
        trust_policy: &DirectoryObservationCertificateTrustPolicy,
        identity: &IdentityKeyPair,
        verified_at: u64,
    ) -> Result<DirectoryObservationCertificateImportReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id {
            return Err(DirectoryReplicaStoreError::Request(
                "certificate importer identity does not match replica metadata".to_string(),
            ));
        }
        let verified = verify_directory_observation_certificate_frame(
            frame,
            expected_sha256,
            trust_policy,
            verified_at,
        )?;
        let checkpoint = &verified.certificate.checkpoint;
        if checkpoint.observer == self.local_node_id {
            return Err(DirectoryReplicaStoreError::Request(
                "local observation certificate cannot be imported as third-party evidence"
                    .to_string(),
            ));
        }

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let audited = Self::audit_observation_certificate_imports(
            &transaction,
            &self.local_node_id,
            verified_at,
        )?;

        let existing = transaction
            .query_row(
                "SELECT import_sequence, import_digest, certificate_sha256,
                        policy_digest, observer, checkpoint_sequence,
                        checkpoint_hash, verified_at
                 FROM directory_observation_certificate_imports
                 WHERE certificate_id = ?1",
                params![verified.certificate_id.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, Vec<u8>>(6)?,
                        row.get::<_, i64>(7)?,
                    ))
                },
            )
            .optional()?;
        if let Some((
            import_sequence,
            import_digest,
            certificate_sha256,
            policy_digest,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            existing_verified_at,
        )) = existing
        {
            if bytes32(&certificate_sha256, "imported certificate SHA-256")?
                != verified.certificate_sha256
                || bytes32(&observer, "imported certificate observer")? != checkpoint.observer
                || positive_i64_to_u64(
                    checkpoint_sequence,
                    "imported certificate checkpoint sequence",
                )? != checkpoint.sequence
                || bytes32(&checkpoint_hash, "imported certificate checkpoint hash")?
                    != checkpoint.hash()
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "existing observation certificate import conflicts with exact evidence"
                        .to_string(),
                ));
            }
            if bytes32(&policy_digest, "imported certificate policy digest")?
                != verified.policy_digest
            {
                return Err(DirectoryReplicaStoreError::Request(
                    "observation certificate was already imported under a different local trust policy"
                        .to_string(),
                ));
            }
            return Ok(DirectoryObservationCertificateImportReport {
                inserted: false,
                import_sequence: positive_i64_to_u64(
                    import_sequence,
                    "certificate import sequence",
                )?,
                import_digest: bytes32(&import_digest, "certificate import digest")?,
                certificate_id: verified.certificate_id,
                certificate_sha256: verified.certificate_sha256,
                observer: checkpoint.observer,
                checkpoint_sequence: checkpoint.sequence,
                checkpoint_hash: checkpoint.hash(),
                retained_certificates: audited.imports,
                verified_at: positive_i64_to_u64(
                    existing_verified_at,
                    "certificate import verification timestamp",
                )?,
            });
        }

        let latest_for_observer = transaction
            .query_row(
                "SELECT checkpoint_sequence, certificate_id
                 FROM directory_observation_certificate_imports
                 WHERE observer = ?1
                 ORDER BY checkpoint_sequence DESC LIMIT 1",
                params![checkpoint.observer.as_slice()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?;
        if let Some((stored_sequence, stored_certificate_id)) = latest_for_observer {
            let stored_sequence =
                positive_i64_to_u64(stored_sequence, "latest imported checkpoint sequence")?;
            if checkpoint.sequence < stored_sequence {
                return Err(DirectoryReplicaStoreError::Request(
                    "observation certificate checkpoint would roll back this observer".to_string(),
                ));
            }
            if checkpoint.sequence == stored_sequence {
                let stored_certificate_id =
                    bytes32(&stored_certificate_id, "latest imported certificate id")?;
                return Err(DirectoryReplicaStoreError::Request(format!(
                    "observation certificate conflicts at checkpoint sequence {} ({})",
                    checkpoint.sequence,
                    hex::encode(stored_certificate_id)
                )));
            }
        }
        let retained_imports = usize::try_from(audited.imports).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation certificate import count exceeds platform capacity".to_string(),
            )
        })?;
        if retained_imports >= MAX_DIRECTORY_OBSERVATION_CERTIFICATE_IMPORTS {
            return Err(DirectoryReplicaStoreError::Request(
                "observation certificate import capacity is exhausted".to_string(),
            ));
        }

        let import_sequence = audited.imports.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation certificate import sequence exhausted".to_string(),
            )
        })?;
        let entry = DirectoryObservationCertificateImportEntry::sign(
            identity,
            import_sequence,
            audited.head,
            &verified,
        )?;
        let import_digest = entry.digest();
        let witness_node_ids = trust_policy
            .allowed_witnesses()
            .iter()
            .flat_map(|node_id| node_id.iter().copied())
            .collect::<Vec<_>>();
        transaction.execute(
            "INSERT INTO directory_observation_certificate_imports
                (import_sequence, import_digest, previous_import_digest,
                 certificate_id, observer, checkpoint_sequence, checkpoint_hash,
                 checkpoint_observed_at, certificate_sha256, certificate_frame,
                 policy_digest, policy_minimum_witnesses, policy_witness_count,
                 policy_witness_node_ids, verified_at, importer_node_id, signature)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                     ?13, ?14, ?15, ?16, ?17)",
            params![
                u64_to_i64(import_sequence, "certificate import sequence")?,
                import_digest.as_slice(),
                audited.head.as_slice(),
                entry.certificate_id.as_slice(),
                entry.observer.as_slice(),
                u64_to_i64(
                    entry.checkpoint_sequence,
                    "imported certificate checkpoint sequence"
                )?,
                entry.checkpoint_hash.as_slice(),
                u64_to_i64(
                    entry.checkpoint_observed_at,
                    "imported certificate checkpoint timestamp"
                )?,
                entry.certificate_sha256.as_slice(),
                frame,
                entry.policy_digest.as_slice(),
                i64::from(trust_policy.minimum_witnesses()),
                i64::try_from(trust_policy.allowed_witnesses().len()).map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "certificate policy witness count exceeds i64".to_string(),
                    )
                })?,
                witness_node_ids,
                u64_to_i64(
                    entry.verified_at,
                    "certificate import verification timestamp"
                )?,
                entry.importer_node_id.as_slice(),
                entry.signature.as_slice(),
            ],
        )?;
        let metadata_changed = transaction.execute(
            "UPDATE directory_replica_meta
             SET certificate_import_sequence = ?1, certificate_import_head = ?2
             WHERE singleton = 1
               AND certificate_import_sequence = ?3
               AND ((?3 = 0 AND certificate_import_head IS NULL)
                    OR (?3 > 0 AND certificate_import_head = ?4))",
            params![
                u64_to_i64(import_sequence, "certificate import sequence")?,
                import_digest.as_slice(),
                u64_to_i64(audited.imports, "previous certificate import sequence")?,
                audited.head.as_slice(),
            ],
        )?;
        if metadata_changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation certificate import metadata compare-and-swap failed".to_string(),
            ));
        }
        transaction.commit()?;

        Ok(DirectoryObservationCertificateImportReport {
            inserted: true,
            import_sequence,
            import_digest,
            certificate_id: entry.certificate_id,
            certificate_sha256: entry.certificate_sha256,
            observer: entry.observer,
            checkpoint_sequence: entry.checkpoint_sequence,
            checkpoint_hash: entry.checkpoint_hash,
            retained_certificates: import_sequence,
            verified_at: entry.verified_at,
        })
    }

    pub(super) fn audit_observation_certificate_imports(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<ObservationCertificateImportAudit, DirectoryReplicaStoreError> {
        let (metadata_sequence, metadata_head) = connection.query_row(
            "SELECT certificate_import_sequence, certificate_import_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_sequence = nonnegative_i64_to_u64(
            metadata_sequence,
            "replica metadata certificate import sequence",
        )?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata certificate import head"))
            .transpose()?;

        let mut statement = connection.prepare(
            "SELECT import_sequence, import_digest, previous_import_digest,
                    certificate_id, observer, checkpoint_sequence, checkpoint_hash,
                    checkpoint_observed_at, certificate_sha256, certificate_frame,
                    policy_digest, policy_minimum_witnesses, policy_witness_count,
                    policy_witness_node_ids, verified_at, importer_node_id, signature
             FROM directory_observation_certificate_imports
             ORDER BY import_sequence ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok(StoredObservationCertificateImportRow {
                    import_sequence: row.get(0)?,
                    import_digest: row.get(1)?,
                    previous_import_digest: row.get(2)?,
                    certificate_id: row.get(3)?,
                    observer: row.get(4)?,
                    checkpoint_sequence: row.get(5)?,
                    checkpoint_hash: row.get(6)?,
                    checkpoint_observed_at: row.get(7)?,
                    certificate_sha256: row.get(8)?,
                    certificate_frame: row.get(9)?,
                    policy_digest: row.get(10)?,
                    policy_minimum_witnesses: row.get(11)?,
                    policy_witness_count: row.get(12)?,
                    policy_witness_node_ids: row.get(13)?,
                    verified_at: row.get(14)?,
                    importer_node_id: row.get(15)?,
                    signature: row.get(16)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        if rows.len() > MAX_DIRECTORY_OBSERVATION_CERTIFICATE_IMPORTS {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation certificate import history exceeds capacity".to_string(),
            ));
        }

        let mut audit = ObservationCertificateImportAudit::default();
        let mut latest_by_observer = BTreeMap::<[u8; 32], u64>::new();
        for row in rows {
            let import_sequence =
                positive_i64_to_u64(row.import_sequence, "certificate import sequence")?;
            let expected_sequence = audit.imports.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "certificate import audit sequence exhausted".to_string(),
                )
            })?;
            let stored_import_digest = bytes32(&row.import_digest, "certificate import digest")?;
            let previous_import_digest = bytes32(
                &row.previous_import_digest,
                "previous certificate import digest",
            )?;
            let observer = bytes32(&row.observer, "imported certificate observer")?;
            let checkpoint_sequence = positive_i64_to_u64(
                row.checkpoint_sequence,
                "imported certificate checkpoint sequence",
            )?;
            let checkpoint_hash =
                bytes32(&row.checkpoint_hash, "imported certificate checkpoint hash")?;
            let checkpoint_observed_at = positive_i64_to_u64(
                row.checkpoint_observed_at,
                "imported certificate checkpoint timestamp",
            )?;
            let certificate_sha256 =
                bytes32(&row.certificate_sha256, "imported certificate SHA-256")?;
            let policy_digest = bytes32(&row.policy_digest, "imported certificate policy digest")?;
            let minimum_witnesses = u16::try_from(positive_i64_to_u64(
                row.policy_minimum_witnesses,
                "imported certificate policy threshold",
            )?)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "imported certificate policy threshold exceeds u16".to_string(),
                )
            })?;
            let witness_count = usize::try_from(positive_i64_to_u64(
                row.policy_witness_count,
                "imported certificate policy witness count",
            )?)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "imported certificate witness count exceeds usize".to_string(),
                )
            })?;
            if witness_count > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
                || row.policy_witness_node_ids.len() != witness_count.saturating_mul(32)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "imported certificate witness policy blob is malformed".to_string(),
                ));
            }
            let allowed_witnesses = row
                .policy_witness_node_ids
                .chunks_exact(32)
                .map(|node_id| bytes32(node_id, "imported certificate policy witness"))
                .collect::<Result<Vec<_>, _>>()?;
            let verified_at =
                positive_i64_to_u64(row.verified_at, "certificate import verification timestamp")?;
            if verified_at
                > observed_at
                    .saturating_add(DIRECTORY_OBSERVATION_CERTIFICATE_IMPORT_TIMESTAMP_SKEW_SECS)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "certificate import verification timestamp is in the future".to_string(),
                ));
            }
            let importer_node_id =
                bytes32(&row.importer_node_id, "certificate importer node identity")?;
            let signature = bytes64(&row.signature, "certificate import signature")?;
            let certificate_id = bytes32(&row.certificate_id, "imported certificate identity")?;
            if import_sequence != expected_sequence
                || previous_import_digest != audit.head
                || observer == *local_node_id
                || importer_node_id != *local_node_id
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation certificate import history is not canonical".to_string(),
                ));
            }
            if latest_by_observer
                .get(&observer)
                .is_some_and(|previous_sequence| checkpoint_sequence <= *previous_sequence)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation certificate import history rolls back or conflicts".to_string(),
                ));
            }

            let trust_policy = DirectoryObservationCertificateTrustPolicy::new(
                observer,
                allowed_witnesses,
                minimum_witnesses,
            )
            .map_err(|error| {
                DirectoryReplicaStoreError::Integrity(format!(
                    "imported certificate trust policy is invalid: {error}"
                ))
            })?;
            let verified = verify_directory_observation_certificate_frame(
                &row.certificate_frame,
                &certificate_sha256,
                &trust_policy,
                verified_at,
            )
            .map_err(|error| {
                DirectoryReplicaStoreError::Integrity(format!(
                    "imported observation certificate failed audit: {error}"
                ))
            })?;
            if verified.certificate_id != certificate_id
                || verified.policy_digest != policy_digest
                || verified.certificate.checkpoint.observer != observer
                || verified.certificate.checkpoint.sequence != checkpoint_sequence
                || verified.certificate.checkpoint.hash() != checkpoint_hash
                || verified.certificate.checkpoint.observed_at != checkpoint_observed_at
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "imported observation certificate row does not match its frame".to_string(),
                ));
            }
            let entry = DirectoryObservationCertificateImportEntry {
                import_sequence,
                previous_import_digest,
                certificate_id,
                observer,
                checkpoint_sequence,
                checkpoint_hash,
                checkpoint_observed_at,
                certificate_sha256,
                policy_digest,
                verified_at,
                importer_node_id,
                signature,
            };
            entry.validate_unsigned_fields()?;
            IdentityPublicKey::from_bytes(&entry.importer_node_id)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "certificate importer identity is invalid".to_string(),
                    )
                })?
                .verify(&entry.signing_bytes(), &entry.signature)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "certificate import signature is invalid".to_string(),
                    )
                })?;
            if entry.digest() != stored_import_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "certificate import digest is invalid".to_string(),
                ));
            }
            audit.imports = expected_sequence;
            audit.head = stored_import_digest;
            latest_by_observer.insert(observer, checkpoint_sequence);
        }

        if audit.imports != metadata_sequence
            || (audit.imports == 0 && metadata_head.is_some())
            || (audit.imports > 0 && metadata_head != Some(audit.head))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "certificate import metadata head does not match audited history".to_string(),
            ));
        }
        Ok(audit)
    }
}
