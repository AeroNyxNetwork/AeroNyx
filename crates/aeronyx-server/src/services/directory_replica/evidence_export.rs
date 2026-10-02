// [ARCH-SPLIT 2026-10-02]
// Operator evidence pages, inclusion proofs, and status snapshots.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Audits the complete requested producer namespace, then exports one
    /// bounded page from the same read transaction.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid bounds, a missing or
    /// quarantined producer, any audit failure, or malformed persisted bytes.
    pub fn audited_evidence_page(
        &self,
        producer: &[u8; 32],
        from_height: u64,
        limit: u16,
        observed_at: u64,
    ) -> Result<DirectoryReplicaEvidencePage, DirectoryReplicaStoreError> {
        self.audited_evidence_page_with_scope(
            producer,
            from_height,
            limit,
            observed_at,
            DirectoryReplicaEvidenceScope::AnyAudited,
        )
    }

    pub(super) fn audited_evidence_page_with_scope(
        &self,
        producer: &[u8; 32],
        from_height: u64,
        limit: u16,
        observed_at: u64,
        scope: DirectoryReplicaEvidenceScope,
    ) -> Result<DirectoryReplicaEvidencePage, DirectoryReplicaStoreError> {
        if *producer == [0u8; 32]
            || *producer == self.local_node_id
            || from_height == 0
            || limit == 0
            || limit > MAX_DIRECTORY_SYNC_BLOCKS_V1
            || observed_at == 0
        {
            return Err(DirectoryReplicaStoreError::Request(
                "replica evidence page fields are invalid".to_string(),
            ));
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        // [DIRECTORY-PRODUCER-AUDIT 2026-07-22 by Codex] Keep export
        // verification fail-closed for the target without coupling availability
        // to unrelated producer, checkpoint, witness, or retry histories.
        let tip = Self::audit_evidence_producer(
            &transaction,
            &self.local_node_id,
            producer,
            observed_at,
            scope,
        )?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }
        if from_height > tip.tip_height.saturating_add(1) {
            return Err(DirectoryReplicaStoreError::RangeNotRetained {
                from_height,
                tip_height: tip.tip_height,
            });
        }
        let mut statement = transaction.prepare(
            "SELECT length(block_blob),
                    CASE WHEN length(block_blob) <= ?4 THEN block_blob END
             FROM directory_replica_blocks
             WHERE producer = ?1 AND height >= ?2
             ORDER BY height ASC LIMIT ?3",
        )?;
        let blobs = statement
            .query_map(
                params![
                    producer.as_slice(),
                    u64_to_i64(from_height, "replica evidence from height")?,
                    i64::from(limit),
                    u64_to_i64(MAX_DIRECTORY_BLOCK_BYTES, "replica block byte limit")?
                ],
                |row| {
                    Ok((
                        row.get::<_, Option<i64>>(0)?,
                        row.get::<_, Option<Vec<u8>>>(1)?,
                    ))
                },
            )?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        let blocks = blobs
            .into_iter()
            .map(|(length, blob)| {
                let blob = materialize_admitted_replica_blob(
                    length,
                    blob,
                    PersistedReplicaBlobKind::Block,
                )?;
                decode_block(&blob)
            })
            .collect::<Result<Vec<_>, _>>()?;
        transaction.commit()?;
        Ok(DirectoryReplicaEvidencePage {
            blocks,
            tip_height: tip.tip_height,
            tip_hash: tip.tip_hash,
        })
    }

    /// Exports one audited page only when the producer remains in the durable
    /// non-authoritative mirror registry.
    ///
    /// Public recovery admission must use this method instead of the general
    /// authority evidence reader. Registry membership grants read eligibility
    /// only; it never grants checkpoint, witness, policy, voting, fork-choice,
    /// consensus, or finality authority.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError::MirrorNotRetained`] when this node
    /// does not retain the requested producer as a permissionless mirror, plus
    /// the ordinary audited evidence-page errors.
    pub(crate) fn audited_mirror_evidence_page(
        &self,
        producer: &[u8; 32],
        from_height: u64,
        limit: u16,
        observed_at: u64,
    ) -> Result<DirectoryReplicaEvidencePage, DirectoryReplicaStoreError> {
        self.audited_evidence_page_with_scope(
            producer,
            from_height,
            limit,
            observed_at,
            DirectoryReplicaEvidenceScope::RetainedMirror,
        )
    }

    /// Audits the complete requested producer namespace, then loads exact
    /// producer-bound descriptor objects from the same read transaction and in
    /// request order.
    ///
    /// `Ok(None)` means at least one requested object is not retained for this
    /// producer. Partial responses are never returned.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid bounds, quarantine,
    /// any audit failure, or malformed persisted bytes/indexes.
    pub fn audited_evidence_descriptor_objects(
        &self,
        producer: &[u8; 32],
        descriptor_hashes: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<Option<Vec<SignedNodeDescriptor>>, DirectoryReplicaStoreError> {
        self.audited_evidence_descriptor_objects_with_scope(
            producer,
            descriptor_hashes,
            observed_at,
            DirectoryReplicaEvidenceScope::AnyAudited,
        )
    }

    pub(super) fn audited_evidence_descriptor_objects_with_scope(
        &self,
        producer: &[u8; 32],
        descriptor_hashes: &[[u8; 32]],
        observed_at: u64,
        scope: DirectoryReplicaEvidenceScope,
    ) -> Result<Option<Vec<SignedNodeDescriptor>>, DirectoryReplicaStoreError> {
        let unique = descriptor_hashes.iter().copied().collect::<HashSet<_>>();
        if *producer == [0u8; 32]
            || *producer == self.local_node_id
            || descriptor_hashes.is_empty()
            || descriptor_hashes.len()
                > aeronyx_core::protocol::discovery::MAX_DIRECTORY_SYNC_OBJECTS_V1
            || unique.len() != descriptor_hashes.len()
            || descriptor_hashes.iter().any(|hash| *hash == [0u8; 32])
            || observed_at == 0
        {
            return Err(DirectoryReplicaStoreError::Request(
                "replica evidence object fields are invalid".to_string(),
            ));
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        // [DIRECTORY-PRODUCER-AUDIT 2026-07-22 by Codex] Object hydration uses
        // the identical producer-scoped audit boundary as block export.
        let tip = Self::audit_evidence_producer(
            &transaction,
            &self.local_node_id,
            producer,
            observed_at,
            scope,
        )?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }
        let mut statement = transaction.prepare(
            "SELECT length(descriptor_blob),
                    CASE WHEN length(descriptor_blob) <= ?3 THEN descriptor_blob END
             FROM directory_replica_descriptor_objects
             WHERE producer = ?1 AND descriptor_hash = ?2",
        )?;
        let mut objects = Vec::with_capacity(descriptor_hashes.len());
        for descriptor_hash in descriptor_hashes {
            let blob = statement
                .query_row(
                    params![
                        producer.as_slice(),
                        descriptor_hash.as_slice(),
                        u64_to_i64(
                            MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES,
                            "replica descriptor byte limit"
                        )?
                    ],
                    |row| {
                        Ok((
                            row.get::<_, Option<i64>>(0)?,
                            row.get::<_, Option<Vec<u8>>>(1)?,
                        ))
                    },
                )
                .optional()?;
            let Some((length, blob)) = blob else {
                drop(statement);
                transaction.commit()?;
                return Ok(None);
            };
            let blob = materialize_admitted_replica_blob(
                length,
                blob,
                PersistedReplicaBlobKind::Descriptor,
            )?;
            let object = decode_descriptor_object(&blob)?;
            let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&object)
                .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            if commitment.descriptor_hash != *descriptor_hash {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica evidence descriptor hash mismatch".to_string(),
                ));
            }
            objects.push(object);
        }
        drop(statement);
        transaction.commit()?;
        Ok(Some(objects))
    }

    /// Loads exact audited descriptor objects only for a producer retained in
    /// the durable non-authoritative mirror registry.
    ///
    /// This is the object-hydration companion to
    /// [`Self::audited_mirror_evidence_page`]. It deliberately preserves exact
    /// producer binding and refuses partial responses.
    pub(crate) fn audited_mirror_evidence_descriptor_objects(
        &self,
        producer: &[u8; 32],
        descriptor_hashes: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<Option<Vec<SignedNodeDescriptor>>, DirectoryReplicaStoreError> {
        self.audited_evidence_descriptor_objects_with_scope(
            producer,
            descriptor_hashes,
            observed_at,
            DirectoryReplicaEvidenceScope::RetainedMirror,
        )
    }

    /// Builds one exact producer-signed descriptor proof from an audited
    /// replica namespace.
    ///
    /// The complete target producer namespace, commitment indexes, and
    /// descriptor objects are audited before the exact block and object are
    /// loaded in the same read transaction. `Ok(None)` means the descriptor is
    /// absent or belongs to a different selected block; another block is never
    /// substituted silently.
    ///
    /// [REPLICA-INCLUSION-PROOF 2026-07-27 by Codex] This returns original
    /// producer evidence only. A later carrier signature may authenticate
    /// transport, but cannot rewrite the proof or gain authority.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid sentinels, quarantine,
    /// audit failure, malformed persisted data, or inconsistent proof creation.
    pub fn audited_evidence_descriptor_inclusion_proof(
        &self,
        producer: &[u8; 32],
        descriptor_hash: &[u8; 32],
        expected_block_hash: &[u8; 32],
        observed_at: u64,
    ) -> Result<Option<DirectoryDescriptorInclusionProofV1>, DirectoryReplicaStoreError> {
        self.audited_evidence_descriptor_inclusion_proof_with_scope(
            producer,
            descriptor_hash,
            expected_block_hash,
            observed_at,
            DirectoryReplicaEvidenceScope::AnyAudited,
        )
    }

    pub(super) fn audited_evidence_descriptor_inclusion_proof_with_scope(
        &self,
        producer: &[u8; 32],
        descriptor_hash: &[u8; 32],
        expected_block_hash: &[u8; 32],
        observed_at: u64,
        scope: DirectoryReplicaEvidenceScope,
    ) -> Result<Option<DirectoryDescriptorInclusionProofV1>, DirectoryReplicaStoreError> {
        if *producer == [0u8; 32]
            || *producer == self.local_node_id
            || *descriptor_hash == [0u8; 32]
            || *expected_block_hash == [0u8; 32]
            || observed_at == 0
        {
            return Err(DirectoryReplicaStoreError::Request(
                "replica descriptor proof fields are invalid".to_string(),
            ));
        }
        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let tip = Self::audit_evidence_producer(
            &transaction,
            &self.local_node_id,
            producer,
            observed_at,
            scope,
        )?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }
        let persisted = transaction
            .query_row(
                "SELECT length(blocks.block_blob),
                        CASE WHEN length(blocks.block_blob) <= ?3
                             THEN blocks.block_blob END,
                        length(objects.descriptor_blob),
                        CASE WHEN length(objects.descriptor_blob) <= ?4
                             THEN objects.descriptor_blob END
                 FROM directory_replica_commitments AS commitments
                 INNER JOIN directory_replica_blocks AS blocks
                    ON blocks.producer = commitments.producer
                   AND blocks.height = commitments.block_height
                 INNER JOIN directory_replica_descriptor_objects AS objects
                    ON objects.producer = commitments.producer
                   AND objects.descriptor_hash = commitments.descriptor_hash
                 WHERE commitments.producer = ?1
                   AND commitments.descriptor_hash = ?2",
                params![
                    producer.as_slice(),
                    descriptor_hash.as_slice(),
                    u64_to_i64(MAX_DIRECTORY_BLOCK_BYTES, "replica block byte limit")?,
                    u64_to_i64(
                        MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES,
                        "replica descriptor byte limit"
                    )?
                ],
                |row| {
                    Ok((
                        row.get::<_, Option<i64>>(0)?,
                        row.get::<_, Option<Vec<u8>>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                        row.get::<_, Option<Vec<u8>>>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((block_length, block_blob, descriptor_length, descriptor_blob)) = persisted else {
            transaction.commit()?;
            return Ok(None);
        };
        let block_blob = materialize_admitted_replica_blob(
            block_length,
            block_blob,
            PersistedReplicaBlobKind::Block,
        )?;
        let descriptor_blob = materialize_admitted_replica_blob(
            descriptor_length,
            descriptor_blob,
            PersistedReplicaBlobKind::Descriptor,
        )?;
        let block = decode_block(&block_blob)?;
        if block.hash() != *expected_block_hash {
            transaction.commit()?;
            return Ok(None);
        }
        let descriptor = decode_descriptor_object(&descriptor_blob)?;
        let proof =
            DirectoryDescriptorInclusionProofV1::from_block_at(&block, &descriptor, observed_at)
                .map_err(|error| {
                    DirectoryReplicaStoreError::Integrity(format!(
                        "failed to build audited replica descriptor proof: {error}"
                    ))
                })?;
        proof
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                producer,
                expected_block_hash,
                observed_at,
            )
            .map_err(|error| {
                DirectoryReplicaStoreError::Integrity(format!(
                    "failed to re-verify audited replica descriptor proof: {error}"
                ))
            })?;
        if proof.commitment.descriptor_hash != *descriptor_hash {
            return Err(DirectoryReplicaStoreError::Integrity(
                "audited replica proof descriptor hash mismatch".to_string(),
            ));
        }
        transaction.commit()?;
        Ok(Some(proof))
    }

    /// Builds an exact descriptor proof only for a producer retained in the
    /// durable non-authoritative mirror registry.
    pub(crate) fn audited_mirror_evidence_descriptor_inclusion_proof(
        &self,
        producer: &[u8; 32],
        descriptor_hash: &[u8; 32],
        expected_block_hash: &[u8; 32],
        observed_at: u64,
    ) -> Result<Option<DirectoryDescriptorInclusionProofV1>, DirectoryReplicaStoreError> {
        self.audited_evidence_descriptor_inclusion_proof_with_scope(
            producer,
            descriptor_hash,
            expected_block_hash,
            observed_at,
            DirectoryReplicaEvidenceScope::RetainedMirror,
        )
    }

    /// Selects one mature live public descriptor and rebuilds its exact proof.
    ///
    /// Selection reads at most [`DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT`]
    /// recent mature descriptor rows from non-quarantined, non-local producer
    /// namespaces. A block is mature only when its signed `produced_at` is no
    /// newer than `observed_at - minimum_block_age_secs`. `selection_seed`
    /// rotates the chosen live row so periodic gossip does not continuously
    /// amplify one descriptor. The lightweight candidate read never
    /// establishes trust: after selection, the complete producer namespace and
    /// exact block/object indexes are audited by
    /// [`Self::audited_evidence_descriptor_inclusion_proof`].
    ///
    /// `Ok(None)` means this replica currently has no live descriptor suitable
    /// for proof gossip. Expired authentic descriptors are retained as history
    /// but are never announced as routeable state.
    ///
    /// [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] The returned proof is
    /// additive rollout evidence. It does not make the carrier a producer,
    /// authority, validator, voter, or consensus participant.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid time, malformed
    /// persisted candidates, SQLite failure, quarantine races, or any complete
    /// producer-audit/proof mismatch.
    pub(crate) fn audited_live_descriptor_gossip_announcement(
        &self,
        observed_at: u64,
        minimum_block_age_secs: u64,
        selection_seed: u64,
    ) -> Result<Option<DirectoryReplicaGossipAnnouncement>, DirectoryReplicaStoreError> {
        if observed_at == 0 || minimum_block_age_secs == 0 {
            return Err(DirectoryReplicaStoreError::Request(
                "gossip proof maturity fields are invalid".to_string(),
            ));
        }
        // [DIRECTORY-PROOF-MATURITY 2026-07-28 by Codex] Proof validity alone
        // does not imply peer availability of the exact block anchor. Select
        // only evidence old enough for independent replica pull rounds.
        let matured_before = observed_at.saturating_sub(minimum_block_age_secs);

        let persisted_candidates = {
            let connection = self.connection.lock();
            let mut statement = connection.prepare(
                "SELECT commitments.producer,
                        blocks.block_hash,
                        commitments.descriptor_hash,
                        length(objects.descriptor_blob),
                        CASE WHEN length(objects.descriptor_blob) <= ?4
                             THEN objects.descriptor_blob END
                 FROM directory_replica_commitments AS commitments
                 INNER JOIN directory_replica_chains AS chains
                    ON chains.producer = commitments.producer
                 INNER JOIN directory_replica_blocks AS blocks
                    ON blocks.producer = commitments.producer
                   AND blocks.height = commitments.block_height
                 INNER JOIN directory_replica_descriptor_objects AS objects
                    ON objects.producer = commitments.producer
                   AND objects.descriptor_hash = commitments.descriptor_hash
                 WHERE chains.quarantined = 0
                   AND commitments.producer != ?1
                   AND blocks.produced_at <= ?2
                 ORDER BY blocks.produced_at DESC,
                          commitments.producer ASC,
                          commitments.descriptor_hash ASC
                 LIMIT ?3",
            )?;
            let candidates = statement
                .query_map(
                    params![
                        self.local_node_id.as_slice(),
                        i64::try_from(matured_before).map_err(|_| {
                            DirectoryReplicaStoreError::Request(
                                "gossip proof maturity boundary exceeds SQLite range".to_string(),
                            )
                        })?,
                        i64::try_from(DIRECTORY_GOSSIP_PROOF_CANDIDATE_LIMIT).map_err(|_| {
                            DirectoryReplicaStoreError::Integrity(
                                "gossip proof candidate limit exceeds SQLite range".to_string(),
                            )
                        })?,
                        u64_to_i64(
                            MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES,
                            "replica descriptor byte limit"
                        )?
                    ],
                    |row| {
                        Ok((
                            row.get::<_, Vec<u8>>(0)?,
                            row.get::<_, Vec<u8>>(1)?,
                            row.get::<_, Vec<u8>>(2)?,
                            row.get::<_, Option<i64>>(3)?,
                            row.get::<_, Option<Vec<u8>>>(4)?,
                        ))
                    },
                )?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            candidates
        };

        let mut latest_live_candidates = BTreeMap::new();
        for (
            producer_bytes,
            block_hash_bytes,
            descriptor_hash_bytes,
            descriptor_blob_length,
            descriptor_blob,
        ) in persisted_candidates
        {
            let producer = bytes32(&producer_bytes, "gossip proof producer")?;
            let block_hash = bytes32(&block_hash_bytes, "gossip proof block hash")?;
            let descriptor_hash = bytes32(&descriptor_hash_bytes, "gossip proof descriptor hash")?;
            let descriptor_blob = materialize_admitted_replica_blob(
                descriptor_blob_length,
                descriptor_blob,
                PersistedReplicaBlobKind::Descriptor,
            )?;
            let descriptor = decode_descriptor_object(&descriptor_blob)?;
            descriptor.verify_signature().map_err(|error| {
                DirectoryReplicaStoreError::Descriptor(format!(
                    "gossip proof candidate signature is invalid: {error}"
                ))
            })?;
            let derived = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            if derived.descriptor_hash != descriptor_hash {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "gossip proof candidate descriptor hash mismatch".to_string(),
                ));
            }
            if descriptor.descriptor.is_valid_at(observed_at) {
                // [DIRECTORY-GOSSIP-PUBLISH 2026-07-27 by Codex] One producer
                // may retain several still-live revisions for the same public
                // node. Announce only its highest sequence from this bounded
                // window so old-but-valid history cannot crowd out diversity.
                let candidate_key = (producer, descriptor.node_id());
                match latest_live_candidates.entry(candidate_key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(DirectoryReplicaGossipCandidate {
                            producer,
                            block_hash,
                            descriptor_hash,
                            descriptor,
                        });
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        let current = entry.get();
                        if descriptor.sequence() > current.descriptor.sequence() {
                            entry.insert(DirectoryReplicaGossipCandidate {
                                producer,
                                block_hash,
                                descriptor_hash,
                                descriptor,
                            });
                        } else if descriptor.sequence() == current.descriptor.sequence()
                            && descriptor_hash != current.descriptor_hash
                        {
                            return Err(DirectoryReplicaStoreError::Integrity(
                                "gossip proof candidates contain descriptor equivocation"
                                    .to_string(),
                            ));
                        }
                    }
                }
            }
        }

        // [DIRECTORY-PROOF-DIVERSITY 2026-07-28 by Codex] A producer can place
        // several public descriptors in one block. Rotate the outer dimension
        // by producer first; otherwise adjacent fallback seeds can repeatedly
        // select evidence anchored by the same namespace.
        let mut candidates_by_producer =
            BTreeMap::<[u8; 32], Vec<DirectoryReplicaGossipCandidate>>::new();
        for candidate in latest_live_candidates.into_values() {
            candidates_by_producer
                .entry(candidate.producer)
                .or_default()
                .push(candidate);
        }
        if candidates_by_producer.is_empty() {
            return Ok(None);
        }
        let producer_count = u64::try_from(candidates_by_producer.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "gossip proof producer count exceeds u64".to_string(),
            )
        })?;
        let selected_producer_index =
            usize::try_from(selection_seed % producer_count).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "gossip proof producer index exceeds usize".to_string(),
                )
            })?;
        let producer_candidates = candidates_by_producer
            .values_mut()
            .nth(selected_producer_index)
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "selected gossip proof producer disappeared".to_string(),
                )
            })?;
        let candidate_count = u64::try_from(producer_candidates.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "gossip proof descriptor count exceeds u64".to_string(),
            )
        })?;
        let descriptor_seed = selection_seed / producer_count;
        let selected_index = usize::try_from(descriptor_seed % candidate_count).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "gossip proof descriptor index exceeds usize".to_string(),
            )
        })?;
        let selected = producer_candidates.swap_remove(selected_index);
        let proof = self
            .audited_evidence_descriptor_inclusion_proof(
                &selected.producer,
                &selected.descriptor_hash,
                &selected.block_hash,
                observed_at,
            )?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "selected gossip descriptor proof disappeared during audit".to_string(),
                )
            })?;
        if proof.descriptor != selected.descriptor
            || proof.commitment.descriptor_hash != selected.descriptor_hash
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "selected gossip descriptor proof does not match candidate".to_string(),
            ));
        }

        Ok(Some(DirectoryReplicaGossipAnnouncement {
            producer: selected.producer,
            block_hash: selected.block_hash,
            descriptor_hash: selected.descriptor_hash,
            proof,
        }))
    }

    /// Re-verifies one complete producer namespace for a transactional export.
    ///
    /// [DIRECTORY-PRODUCER-AUDIT 2026-07-22 by Codex] A carrier must remain a
    /// blind transport for independently signed public evidence. This helper
    /// deliberately avoids a cache or database-supplied watermark: metadata and
    /// public mirror admission are checked first, then every target block,
    /// commitment, descriptor object, signature, hash link, index, and tip is
    /// verified before any evidence is read from the same transaction.
    pub(super) fn audit_evidence_producer(
        connection: &Connection,
        local_node_id: &[u8; 32],
        producer: &[u8; 32],
        observed_at: u64,
        scope: DirectoryReplicaEvidenceScope,
    ) -> Result<DirectoryReplicaTip, DirectoryReplicaStoreError> {
        Self::validate_metadata(connection, local_node_id)?;
        if scope == DirectoryReplicaEvidenceScope::RetainedMirror {
            Self::audit_mirror_registry(connection, local_node_id)?;
        }
        Self::require_evidence_scope(connection, producer, scope)?;
        let tip = Self::load_tip(connection, producer)?;
        let mut report = DirectoryReplicaAudit::default();
        Self::audit_producer(connection, &tip, observed_at, &mut report)?;
        Ok(tip)
    }

    pub(super) fn require_evidence_scope(
        connection: &Connection,
        producer: &[u8; 32],
        scope: DirectoryReplicaEvidenceScope,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if scope == DirectoryReplicaEvidenceScope::AnyAudited {
            return Ok(());
        }
        let retained = connection
            .query_row(
                "SELECT 1 FROM directory_replica_mirror_producers WHERE producer = ?1",
                params![producer.as_slice()],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !retained {
            return Err(DirectoryReplicaStoreError::MirrorNotRetained);
        }
        Ok(())
    }

    /// Returns a low-cost aggregate snapshot of persisted, already-audited
    /// replica indexes.
    ///
    /// This is an observability read, not a replacement for [`Self::audit`].
    /// Startup still performs the full signature, linkage, object, index, and
    /// incident audit before synchronization or API serving begins.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when a persisted status row is
    /// malformed or `SQLite` cannot complete the bounded aggregate query.
    pub fn status_snapshot(
        &self,
    ) -> Result<DirectoryReplicaStoreSnapshot, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let mut statement = connection.prepare(
            "SELECT c.producer, c.tip_height, c.tip_timestamp, c.quarantined,
                    c.quarantine_kind, c.updated_at,
                    (SELECT COUNT(*) FROM directory_replica_blocks b
                     WHERE b.producer = c.producer),
                    (SELECT COUNT(*) FROM directory_replica_commitments m
                     WHERE m.producer = c.producer),
                    (SELECT COUNT(*) FROM directory_replica_incidents i
                     WHERE i.producer = c.producer),
                    (SELECT COUNT(*) FROM directory_replica_resolutions r
                     WHERE r.producer = c.producer)
             FROM directory_replica_chains c
             ORDER BY c.producer ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        let mut snapshot = DirectoryReplicaStoreSnapshot::default();
        snapshot.mirror_producers = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_replica_mirror_producers",
                [],
                |row| row.get(0),
            )?,
            "directory mirror producer count",
        )?;
        for (
            producer,
            tip_height,
            tip_timestamp,
            quarantined,
            quarantine_kind,
            updated_at,
            blocks,
            commitments,
            incidents,
            resolutions,
        ) in rows
        {
            if quarantined != 0 && quarantined != 1 {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica status quarantine flag is invalid".to_string(),
                ));
            }
            let producer_snapshot = DirectoryReplicaProducerSnapshot {
                producer: bytes32(&producer, "replica status producer")?,
                tip_height: nonnegative_i64_to_u64(tip_height, "replica status tip height")?,
                tip_timestamp: nonnegative_i64_to_u64(
                    tip_timestamp,
                    "replica status tip timestamp",
                )?,
                quarantined: quarantined == 1,
                quarantine_kind,
                updated_at: nonnegative_i64_to_u64(updated_at, "replica status updated at")?,
                blocks: nonnegative_i64_to_u64(blocks, "replica status blocks")?,
                commitments: nonnegative_i64_to_u64(commitments, "replica status commitments")?,
                incidents: nonnegative_i64_to_u64(incidents, "replica status incidents")?,
                resolutions: nonnegative_i64_to_u64(resolutions, "replica status resolutions")?,
            };
            snapshot.producers = snapshot.producers.saturating_add(1);
            snapshot.quarantined_producers = snapshot
                .quarantined_producers
                .saturating_add(u64::from(producer_snapshot.quarantined));
            snapshot.blocks = snapshot.blocks.saturating_add(producer_snapshot.blocks);
            snapshot.commitments = snapshot
                .commitments
                .saturating_add(producer_snapshot.commitments);
            snapshot.incidents = snapshot
                .incidents
                .saturating_add(producer_snapshot.incidents);
            snapshot.resolutions = snapshot
                .resolutions
                .saturating_add(producer_snapshot.resolutions);
            snapshot.producer_snapshots.push(producer_snapshot);
        }
        Self::populate_observation_status_snapshot(
            &connection,
            &self.local_node_id,
            &mut snapshot,
        )?;
        Ok(snapshot)
    }

    pub(super) fn populate_observation_status_snapshot(
        connection: &Connection,
        local_node_id: &[u8; 32],
        snapshot: &mut DirectoryReplicaStoreSnapshot,
    ) -> Result<(), DirectoryReplicaStoreError> {
        snapshot.observation_checkpoints = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoints",
                [],
                |row| row.get(0),
            )?,
            "observation checkpoint count",
        )?;
        let checkpoint_tip = Self::load_observation_checkpoint_tip(connection)?;
        snapshot.observation_checkpoint_sequence = checkpoint_tip.sequence;
        snapshot.observation_checkpoint_hash = checkpoint_tip.checkpoint_hash;
        snapshot.observation_checkpoint_observed_at = checkpoint_tip.observed_at;
        let witness_summary = Self::load_observation_witness_summary(connection)?;
        snapshot.observation_checkpoint_witnesses = witness_summary.witnesses;
        snapshot.observation_checkpoint_witnessed_sequence = witness_summary.latest_sequence;
        snapshot.observation_checkpoint_latest_witnesses = witness_summary.latest_witnesses;
        snapshot.observation_witness_outcomes =
            Self::load_observation_witness_outcome_snapshot(connection)?;
        let witness_policy =
            Self::load_current_observation_witness_policy(connection, local_node_id)?;
        snapshot.observation_witness_policy_epochs = witness_policy.epochs;
        snapshot.observation_witness_policy_epoch = witness_policy.epochs;
        if let Some(policy) = witness_policy.current {
            snapshot.observation_witness_policy_activated_at = policy.activated_at;
            snapshot.observation_witness_policy_members =
                u64::try_from(policy.witness_node_ids.len()).map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy member count exceeds u64".to_string(),
                    )
                })?;
            snapshot.observation_witness_policy_threshold = u64::try_from(policy.minimum_witnesses)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy threshold exceeds u64".to_string(),
                    )
                })?;
        }
        snapshot.observation_witness_policy_anchor_receipts = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_policy_anchor_receipts",
                [],
                |row| row.get(0),
            )?,
            "observation policy anchor receipt count",
        )?;
        snapshot.observation_witness_remote_policy_anchors = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_remote_policy_anchors",
                [],
                |row| row.get(0),
            )?,
            "remote observation policy anchor count",
        )?;
        let (import_sequence, import_head, import_count) = connection.query_row(
            "SELECT m.certificate_import_sequence, m.certificate_import_head,
                    (SELECT COUNT(*) FROM directory_observation_certificate_imports)
             FROM directory_replica_meta m WHERE m.singleton = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<Vec<u8>>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )?;
        snapshot.imported_observation_certificate_sequence =
            nonnegative_i64_to_u64(import_sequence, "certificate import status sequence")?;
        snapshot.imported_observation_certificates =
            nonnegative_i64_to_u64(import_count, "certificate import status count")?;
        snapshot.imported_observation_certificate_head = import_head
            .as_deref()
            .map(|value| bytes32(value, "certificate import status head"))
            .transpose()?
            .unwrap_or([0u8; 32]);
        if snapshot.imported_observation_certificates
            != snapshot.imported_observation_certificate_sequence
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "certificate import status count does not match metadata".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns one bounded, deterministic page of incident summaries.
    ///
    /// Summaries are ordered by content digest and use an exclusive cursor.
    /// The exact evidence frame is deliberately omitted from this low-cost
    /// listing operation. Every returned row was cryptographically audited at
    /// startup; callers must use [`Self::incident_evidence`] to re-verify the
    /// complete proof immediately before export.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the limit is outside
    /// `1..=50`, metadata is malformed, or `SQLite` cannot complete the bounded
    /// query.
    pub fn incident_summaries(
        &self,
        after: Option<[u8; 32]>,
        limit: usize,
    ) -> Result<DirectoryReplicaIncidentPage, DirectoryReplicaStoreError> {
        if !(1..=MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE).contains(&limit) {
            return Err(DirectoryReplicaStoreError::Request(
                "incident page limit must be between 1 and 50".to_string(),
            ));
        }
        let fetch_limit = limit.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Request("incident page limit overflow".to_string())
        })?;
        let fetch_limit = i64::try_from(fetch_limit).map_err(|_| {
            DirectoryReplicaStoreError::Request("incident page limit overflow".to_string())
        })?;
        let cursor = after.map(|value| value.to_vec());
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let mut statement = connection.prepare(
            "SELECT i.incident_digest, i.producer, i.subject_node_id, i.kind,
                    i.height, i.local_hash, i.remote_hash, i.observed_at,
                    c.quarantined
             FROM directory_replica_incidents i
             JOIN directory_replica_chains c ON c.producer = i.producer
             WHERE (?1 IS NULL OR i.incident_digest > ?1)
             ORDER BY i.incident_digest ASC LIMIT ?2",
        )?;
        let rows = statement
            .query_map(params![cursor.as_deref(), fetch_limit], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Vec<u8>>(5)?,
                    row.get::<_, Vec<u8>>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        drop(connection);
        let mut incidents = rows
            .into_iter()
            .map(
                |(
                    digest,
                    producer,
                    subject,
                    kind,
                    height,
                    local_hash,
                    remote_hash,
                    observed_at,
                    quarantined,
                )| {
                    validate_incident_kind(&kind)?;
                    if quarantined != 0 && quarantined != 1 {
                        return Err(DirectoryReplicaStoreError::Integrity(
                            "incident producer quarantine flag is invalid".to_string(),
                        ));
                    }
                    Ok(DirectoryReplicaIncidentSummary {
                        incident_digest: bytes32(&digest, "incident digest")?,
                        producer: bytes32(&producer, "incident producer")?,
                        subject_node_id: bytes32(&subject, "incident subject")?,
                        kind,
                        height: nonnegative_i64_to_u64(height, "incident height")?,
                        local_hash: bytes32(&local_hash, "incident local hash")?,
                        remote_hash: bytes32(&remote_hash, "incident remote hash")?,
                        observed_at: positive_i64_to_u64(observed_at, "incident observed at")?,
                        producer_quarantined: quarantined == 1,
                    })
                },
            )
            .collect::<Result<Vec<_>, DirectoryReplicaStoreError>>()?;
        let has_more = incidents.len() > limit;
        incidents.truncate(limit);
        let next_cursor = has_more
            .then(|| incidents.last().map(|incident| incident.incident_digest))
            .flatten();
        Ok(DirectoryReplicaIncidentPage {
            incidents,
            next_cursor,
        })
    }

    /// Loads and independently re-verifies one complete incident proof.
    ///
    /// Canonical encoding, chain id, producer identity, producer signature,
    /// incident digest, evidence size, and all persisted metadata are checked
    /// on every read. No evidence is returned after any mismatch.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when persistence is malformed,
    /// evidence verification fails, or `SQLite` cannot complete the lookup.
    pub fn incident_evidence(
        &self,
        digest: &[u8; 32],
    ) -> Result<Option<DirectoryReplicaIncidentEvidence>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let row = connection
            .query_row(
                "SELECT i.producer, i.subject_node_id, i.kind, i.height,
                        i.local_hash, i.remote_hash, i.evidence_frame,
                        i.observed_at, c.quarantined
                 FROM directory_replica_incidents i
                 JOIN directory_replica_chains c ON c.producer = i.producer
                 WHERE i.incident_digest = ?1",
                params![digest.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, Vec<u8>>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                        row.get::<_, Vec<u8>>(5)?,
                        row.get::<_, Vec<u8>>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, i64>(8)?,
                    ))
                },
            )
            .optional()?;
        drop(connection);
        let Some((
            producer,
            subject,
            kind,
            height,
            local_hash,
            remote_hash,
            evidence_frame,
            observed_at,
            quarantined,
        )) = row
        else {
            return Ok(None);
        };
        validate_incident_kind(&kind)?;
        if evidence_frame.is_empty() || evidence_frame.len() > MAX_DIRECTORY_SYNC_EVIDENCE_BYTES {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica incident evidence size is invalid".to_string(),
            ));
        }
        if quarantined != 0 && quarantined != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "incident producer quarantine flag is invalid".to_string(),
            ));
        }
        let producer = bytes32(&producer, "incident producer")?;
        let subject = bytes32(&subject, "incident subject")?;
        let incident = QuarantineIncident {
            kind: &kind,
            height: nonnegative_i64_to_u64(height, "incident height")?,
            local_hash: bytes32(&local_hash, "incident local hash")?,
            remote_hash: bytes32(&remote_hash, "incident remote hash")?,
            evidence_frame: &evidence_frame,
        };
        if incident_digest(&producer, &subject, &incident) != *digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica incident digest mismatch".to_string(),
            ));
        }
        verify_incident_response_evidence(&evidence_frame, &producer)?;
        let height = incident.height;
        let local_hash = incident.local_hash;
        let remote_hash = incident.remote_hash;
        let summary = DirectoryReplicaIncidentSummary {
            incident_digest: *digest,
            producer,
            subject_node_id: subject,
            kind,
            height,
            local_hash,
            remote_hash,
            observed_at: positive_i64_to_u64(observed_at, "incident observed at")?,
            producer_quarantined: quarantined == 1,
        };
        let evidence_sha256 = Sha256::digest(&evidence_frame).into();
        Ok(Some(DirectoryReplicaIncidentEvidence {
            summary,
            evidence_frame,
            evidence_sha256,
        }))
    }
}
