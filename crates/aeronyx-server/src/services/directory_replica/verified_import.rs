// [ARCH-SPLIT 2026-10-02]
// Verified block-page import, fork preflight, and producer audit.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Re-verifies and atomically imports one signed bounded producer page.
    ///
    /// The exact encoded `BlockRangeResponseV1` is required as durable fork
    /// evidence. Descriptor objects must exactly cover every commitment in the
    /// supplied page, without extras or duplicates.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid evidence, blocks,
    /// descriptor objects, chain gaps, storage errors, or durable quarantine.
    #[allow(clippy::too_many_arguments)]
    pub fn import_verified_page(
        &self,
        producer: [u8; 32],
        blocks: &[DirectoryCommitmentBlockV1],
        objects: &[SignedNodeDescriptor],
        advertised_tip_height: u64,
        advertised_tip_hash: [u8; 32],
        signed_response_frame: &[u8],
        observed_at: u64,
    ) -> Result<DirectoryReplicaImportReport, DirectoryReplicaStoreError> {
        self.import_verified_page_with_mode(
            producer,
            blocks,
            objects,
            advertised_tip_height,
            advertised_tip_hash,
            signed_response_frame,
            observed_at,
            DirectoryReplicaImportMode::PinnedAuthority,
        )
    }

    /// Re-verifies and atomically imports one permissionless mirror page.
    ///
    /// A first accepted page reserves one durable bounded mirror slot. Mirror
    /// membership never changes the configured producer set used by checkpoint,
    /// witness, policy-anchor, consensus, or finality code paths.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn import_verified_mirror_page(
        &self,
        producer: [u8; 32],
        descriptor_sequence: u64,
        max_producers: usize,
        blocks: &[DirectoryCommitmentBlockV1],
        objects: &[SignedNodeDescriptor],
        advertised_tip_height: u64,
        advertised_tip_hash: [u8; 32],
        signed_response_frame: &[u8],
        observed_at: u64,
    ) -> Result<DirectoryReplicaImportReport, DirectoryReplicaStoreError> {
        self.import_verified_page_with_mode(
            producer,
            blocks,
            objects,
            advertised_tip_height,
            advertised_tip_hash,
            signed_response_frame,
            observed_at,
            DirectoryReplicaImportMode::FullNodeMirror {
                descriptor_sequence,
                max_producers,
            },
        )
    }

    /// Re-verifies one carrier-returned page against a retained mirror anchor.
    ///
    /// [MIRROR-CARRIER-SMOKE 2026-07-25 by Codex] This is deliberately a
    /// read-only release-gate primitive. It audits the complete local producer
    /// namespace in a deferred transaction, re-verifies the carrier response,
    /// checks exact descriptor coverage, and validates any producer-signed
    /// continuation after the retained anchor. It never imports blocks,
    /// reserves mirror capacity, changes authority, clears retry state, or
    /// writes quarantine evidence.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the producer is not a
    /// retained mirror, the local anchor is missing, or any carrier, producer,
    /// pagination, descriptor, hash-chain, or signature evidence is invalid.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn verify_retained_carrier_page(
        &self,
        producer: [u8; 32],
        from_height: u64,
        blocks: &[DirectoryCommitmentBlockV1],
        objects: &[SignedNodeDescriptor],
        advertised_tip_height: u64,
        advertised_tip_hash: [u8; 32],
        signed_response_frame: &[u8],
        observed_at: u64,
    ) -> Result<(u64, u64), DirectoryReplicaStoreError> {
        if producer == [0u8; 32]
            || producer == self.local_node_id
            || from_height == 0
            || blocks.is_empty()
            || blocks.len() > usize::from(MAX_DIRECTORY_SYNC_BLOCKS_V1)
            || blocks
                .first()
                .is_some_and(|block| block.header.height != from_height)
            || signed_response_frame.is_empty()
            || signed_response_frame.len() > MAX_DIRECTORY_SYNC_EVIDENCE_BYTES
            || observed_at == 0
        {
            return Err(DirectoryReplicaStoreError::Request(
                "read-only carrier verification fields are invalid".to_string(),
            ));
        }
        let response_admission = verify_range_response_evidence(
            signed_response_frame,
            &producer,
            blocks,
            advertised_tip_height,
            &advertised_tip_hash,
            observed_at,
        )?;
        if response_admission.tip_provenance != DirectoryRangeTipProvenance::CarrierReported {
            return Err(DirectoryReplicaStoreError::Integrity(
                "read-only carrier verification requires carrier-reported evidence".to_string(),
            ));
        }
        validate_page_tip_contract(
            blocks,
            response_admission.has_more,
            advertised_tip_height,
            &advertised_tip_hash,
        )?;
        let descriptors = validate_exact_descriptor_objects(blocks, objects)?;
        let local_anchor =
            self.audited_mirror_evidence_page(&producer, from_height, 1, observed_at)?;
        let Some(anchor) = local_anchor.blocks.first() else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "retained mirror anchor is missing".to_string(),
            ));
        };
        if blocks.first() != Some(anchor) {
            return Err(DirectoryReplicaStoreError::Integrity(
                "carrier page does not match the retained mirror anchor".to_string(),
            ));
        }

        let mut previous_hash = anchor.hash();
        let mut previous_timestamp = anchor.header.timestamp;
        let mut expected_height = from_height.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "read-only carrier verification height exhausted".to_string(),
            )
        })?;
        for block in blocks.iter().skip(1) {
            block.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_height,
                &previous_hash,
                previous_timestamp,
                observed_at,
            )?;
            previous_hash = block.hash();
            previous_timestamp = block.header.timestamp;
            expected_height = expected_height.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "read-only carrier verification height exhausted".to_string(),
                )
            })?;
        }

        for block in blocks {
            for commitment in &block.commitments {
                let descriptor = descriptors
                    .get(&commitment.descriptor_hash)
                    .ok_or_else(|| {
                        DirectoryReplicaStoreError::Integrity(
                            "carrier page is missing an exact descriptor object".to_string(),
                        )
                    })?;
                let derived = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                    .map_err(|error| {
                    DirectoryReplicaStoreError::Descriptor(error.to_string())
                })?;
                if derived != *commitment {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "carrier descriptor object does not match its block commitment".to_string(),
                    ));
                }
            }
        }

        Ok((
            u64::try_from(blocks.len()).unwrap_or(u64::MAX),
            u64::try_from(objects.len()).unwrap_or(u64::MAX),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn import_verified_page_with_mode(
        &self,
        producer: [u8; 32],
        blocks: &[DirectoryCommitmentBlockV1],
        objects: &[SignedNodeDescriptor],
        advertised_tip_height: u64,
        advertised_tip_hash: [u8; 32],
        signed_response_frame: &[u8],
        observed_at: u64,
        mode: DirectoryReplicaImportMode,
    ) -> Result<DirectoryReplicaImportReport, DirectoryReplicaStoreError> {
        if producer == [0u8; 32] || producer == self.local_node_id {
            return Err(DirectoryReplicaStoreError::Request(
                "remote producer must be non-zero and differ from the local node".to_string(),
            ));
        }
        if blocks.len() > usize::from(MAX_DIRECTORY_SYNC_BLOCKS_V1) {
            return Err(DirectoryReplicaStoreError::Request(
                "block page exceeds the Directory Sync V1 bound".to_string(),
            ));
        }
        if signed_response_frame.is_empty()
            || signed_response_frame.len() > MAX_DIRECTORY_SYNC_EVIDENCE_BYTES
        {
            return Err(DirectoryReplicaStoreError::Request(
                "signed response evidence is empty or oversized".to_string(),
            ));
        }
        let response_admission = verify_range_response_evidence(
            signed_response_frame,
            &producer,
            blocks,
            advertised_tip_height,
            &advertised_tip_hash,
            observed_at,
        )?;
        validate_page_tip_contract(
            blocks,
            response_admission.has_more,
            advertised_tip_height,
            &advertised_tip_hash,
        )?;
        let descriptors = validate_exact_descriptor_objects(blocks, objects)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let producer_existed = Self::producer_row_exists(&transaction, &producer)?;
        Self::ensure_producer_row(&transaction, &producer, observed_at)?;
        Self::reconcile_import_mode(&transaction, &producer, observed_at, producer_existed, mode)?;
        let mut tip = Self::load_tip(&transaction, &producer)?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }

        let verified_page_fork = match response_admission.tip_provenance {
            DirectoryRangeTipProvenance::ProducerSigned => {
                if advertised_tip_height < tip.tip_height {
                    let incident = QuarantineIncident {
                        kind: "signed_tip_rollback",
                        height: advertised_tip_height,
                        local_hash: tip.tip_hash,
                        remote_hash: advertised_tip_hash,
                        evidence_frame: signed_response_frame,
                    };
                    Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
                    transaction.commit()?;
                    return Err(DirectoryReplicaStoreError::Quarantined(
                        incident.kind.to_string(),
                    ));
                }
                if advertised_tip_height == tip.tip_height && advertised_tip_hash != tip.tip_hash {
                    let incident = QuarantineIncident {
                        kind: "signed_tip_fork",
                        height: advertised_tip_height,
                        local_hash: tip.tip_hash,
                        remote_hash: advertised_tip_hash,
                        evidence_frame: signed_response_frame,
                    };
                    Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
                    transaction.commit()?;
                    return Err(DirectoryReplicaStoreError::Quarantined(
                        incident.kind.to_string(),
                    ));
                }
                if blocks.is_empty() && advertised_tip_height > tip.tip_height {
                    let incident = QuarantineIncident {
                        kind: "signed_empty_range_gap",
                        height: tip.tip_height.saturating_add(1),
                        local_hash: tip.tip_hash,
                        remote_hash: advertised_tip_hash,
                        evidence_frame: signed_response_frame,
                    };
                    Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
                    transaction.commit()?;
                    return Err(DirectoryReplicaStoreError::Quarantined(
                        incident.kind.to_string(),
                    ));
                }
                Self::preflight_page_block_fork(&transaction, &producer, blocks, observed_at)?
            }
            DirectoryRangeTipProvenance::CarrierReported => {
                // A carrier cannot authenticate producer-tip metadata. Check
                // any retained-prefix conflict first, then preflight same-page
                // conflicts so genuine producer-signed evidence remains
                // durable before rejecting unauthenticated tip-only claims.
                for block in blocks {
                    let existing_hash =
                        Self::block_hash_at(&transaction, &producer, block.header.height)?;
                    if let Some(existing_hash) = existing_hash {
                        if existing_hash != block.hash() {
                            Self::verify_conflicting_block_at_retained_position(
                                &transaction,
                                &producer,
                                block,
                                observed_at,
                            )?;
                            let incident = QuarantineIncident {
                                kind: "signed_block_fork",
                                height: block.header.height,
                                local_hash: existing_hash,
                                remote_hash: block.hash(),
                                evidence_frame: signed_response_frame,
                            };
                            Self::persist_quarantine(
                                &transaction,
                                &producer,
                                &incident,
                                observed_at,
                            )?;
                            transaction.commit()?;
                            return Err(DirectoryReplicaStoreError::Quarantined(
                                incident.kind.to_string(),
                            ));
                        }
                    }
                }
                let verified_page_fork =
                    Self::preflight_page_block_fork(&transaction, &producer, blocks, observed_at)?;
                if verified_page_fork.is_none()
                    && (advertised_tip_height < tip.tip_height
                        || (advertised_tip_height == tip.tip_height
                            && advertised_tip_hash != tip.tip_hash)
                        || (blocks.is_empty() && advertised_tip_height > tip.tip_height))
                {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "carrier-reported tip contradicts the retained producer prefix".to_string(),
                    ));
                }
                verified_page_fork
            }
        };
        if let Some(fork) = verified_page_fork {
            // No page block, object, or commitment has been inserted yet. Keep
            // the call-entry prefix unchanged while retaining both verified
            // producer claims for operator review.
            let incident = QuarantineIncident {
                kind: "signed_block_fork",
                height: fork.height,
                local_hash: fork.first_hash,
                remote_hash: fork.conflicting_hash,
                evidence_frame: signed_response_frame,
            };
            Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
            transaction.commit()?;
            return Err(DirectoryReplicaStoreError::Quarantined(
                incident.kind.to_string(),
            ));
        }
        if blocks.is_empty() {
            Self::clear_retry_state(&transaction, &producer)?;
            transaction.commit()?;
            return Ok(DirectoryReplicaImportReport {
                blocks_inserted: 0,
                blocks_already_present: 0,
                commitments_inserted: 0,
                descriptor_equivocations: 0,
                tip_height: tip.tip_height,
                tip_hash: tip.tip_hash,
            });
        }

        let mut blocks_inserted = 0u64;
        let mut blocks_already_present = 0u64;
        let mut commitments_inserted = 0u64;
        let mut descriptor_equivocations = 0u64;
        for block in blocks {
            if block.header.producer != producer {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "range contains a block signed for another producer".to_string(),
                ));
            }
            let existing_hash = Self::block_hash_at(&transaction, &producer, block.header.height)?;
            if let Some(existing_hash) = existing_hash {
                if existing_hash != block.hash() {
                    // [DIRECTORY-CONFLICT-VERIFICATION 2026-09-01 by Codex]
                    // A prior block in this page may now occupy this height.
                    // Authenticate the conflicting producer claim before any
                    // page mutation or carrier envelope can become blame.
                    Self::verify_conflicting_block_at_retained_position(
                        &transaction,
                        &producer,
                        block,
                        observed_at,
                    )?;
                    let incident = QuarantineIncident {
                        kind: "signed_block_fork",
                        height: block.header.height,
                        local_hash: existing_hash,
                        remote_hash: block.hash(),
                        evidence_frame: signed_response_frame,
                    };
                    Self::persist_quarantine(&transaction, &producer, &incident, observed_at)?;
                    transaction.commit()?;
                    return Err(DirectoryReplicaStoreError::Quarantined(
                        incident.kind.to_string(),
                    ));
                }
                blocks_already_present = blocks_already_present.saturating_add(1);
                continue;
            }
            let expected_height = tip.tip_height.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity("replica chain height exhausted".to_string())
            })?;
            block.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_height,
                &tip.tip_hash,
                tip.tip_timestamp,
                observed_at,
            )?;
            let block_objects = block
                .commitments
                .iter()
                .map(|commitment| {
                    descriptors
                        .get(&commitment.descriptor_hash)
                        .copied()
                        .ok_or_else(|| {
                            DirectoryReplicaStoreError::Integrity(
                                "validated descriptor object map became incomplete".to_string(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (inserted, equivocations) = Self::insert_block(
                &transaction,
                &producer,
                block,
                &block_objects,
                signed_response_frame,
                observed_at,
            )?;
            commitments_inserted = commitments_inserted.saturating_add(inserted);
            descriptor_equivocations = descriptor_equivocations.saturating_add(equivocations);
            blocks_inserted = blocks_inserted.saturating_add(1);
            tip.tip_height = block.header.height;
            tip.tip_hash = block.hash();
            tip.tip_timestamp = block.header.timestamp;
        }
        transaction.execute(
            "UPDATE directory_replica_chains
             SET tip_height = ?2, tip_hash = ?3, tip_timestamp = ?4, updated_at = ?5
             WHERE producer = ?1",
            params![
                producer.as_slice(),
                u64_to_i64(tip.tip_height, "replica tip height")?,
                tip.tip_hash.as_slice(),
                u64_to_i64(tip.tip_timestamp, "replica tip timestamp")?,
                u64_to_i64(observed_at, "replica update timestamp")?
            ],
        )?;
        Self::clear_retry_state(&transaction, &producer)?;
        transaction.commit()?;
        Ok(DirectoryReplicaImportReport {
            blocks_inserted,
            blocks_already_present,
            commitments_inserted,
            descriptor_equivocations,
            tip_height: tip.tip_height,
            tip_hash: tip.tip_hash,
        })
    }

    pub(super) fn reconcile_import_mode(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        observed_at: u64,
        producer_existed: bool,
        mode: DirectoryReplicaImportMode,
    ) -> Result<(), DirectoryReplicaStoreError> {
        match mode {
            DirectoryReplicaImportMode::PinnedAuthority => {
                transaction.execute(
                    "DELETE FROM directory_replica_mirror_producers WHERE producer = ?1",
                    params![producer.as_slice()],
                )?;
            }
            DirectoryReplicaImportMode::FullNodeMirror {
                descriptor_sequence,
                max_producers,
            } => {
                if descriptor_sequence == 0
                    || !(1..=MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS).contains(&max_producers)
                {
                    return Err(DirectoryReplicaStoreError::Request(
                        "directory mirror admission fields are invalid".to_string(),
                    ));
                }
                let registered: bool = transaction.query_row(
                    "SELECT EXISTS(
                         SELECT 1 FROM directory_replica_mirror_producers WHERE producer = ?1
                     )",
                    params![producer.as_slice()],
                    |row| row.get(0),
                )?;
                if producer_existed && !registered {
                    return Err(DirectoryReplicaStoreError::Request(
                        "authority producer cannot be reclassified as a permissionless mirror"
                            .to_string(),
                    ));
                }
                if registered {
                    transaction.execute(
                        "UPDATE directory_replica_mirror_producers
                         SET last_selected_at = MAX(last_selected_at, ?2),
                             descriptor_sequence = MAX(descriptor_sequence, ?3)
                         WHERE producer = ?1",
                        params![
                            producer.as_slice(),
                            u64_to_i64(observed_at, "directory mirror selection timestamp")?,
                            u64_to_i64(
                                descriptor_sequence,
                                "directory mirror descriptor sequence"
                            )?
                        ],
                    )?;
                } else {
                    let mirror_count: i64 = transaction.query_row(
                        "SELECT COUNT(*) FROM directory_replica_mirror_producers",
                        [],
                        |row| row.get(0),
                    )?;
                    let mirror_count = usize::try_from(nonnegative_i64_to_u64(
                        mirror_count,
                        "directory mirror capacity count",
                    )?)
                    .map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "directory mirror capacity count exceeds usize".to_string(),
                        )
                    })?;
                    if mirror_count >= max_producers {
                        return Err(DirectoryReplicaStoreError::MirrorCapacity);
                    }
                    transaction.execute(
                        "INSERT INTO directory_replica_mirror_producers
                            (producer, admitted_at, last_selected_at, descriptor_sequence)
                         VALUES (?1, ?2, ?2, ?3)",
                        params![
                            producer.as_slice(),
                            u64_to_i64(observed_at, "directory mirror admission timestamp")?,
                            u64_to_i64(
                                descriptor_sequence,
                                "directory mirror descriptor sequence"
                            )?
                        ],
                    )?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn preflight_page_block_fork(
        connection: &Connection,
        producer: &[u8; 32],
        blocks: &[DirectoryCommitmentBlockV1],
        observed_at: u64,
    ) -> Result<Option<VerifiedPageBlockFork>, DirectoryReplicaStoreError> {
        // [DIRECTORY-CONFLICT-VERIFICATION 2026-09-01 by Codex] Detect a page
        // fork before insert_block can make response order an implicit choice.
        // Both claims must independently authenticate at the same retained
        // predecessor; an unverifiable claim aborts the untouched transaction.
        let mut first_by_height: HashMap<u64, &DirectoryCommitmentBlockV1> =
            HashMap::with_capacity(blocks.len());
        for block in blocks {
            if block.header.producer != *producer {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "range contains a block signed for another producer".to_string(),
                ));
            }
            let block_hash = block.hash();
            if let Some(first) = first_by_height.get(&block.header.height) {
                let first = *first;
                let first_hash = first.hash();
                if first_hash == block_hash {
                    continue;
                }
                Self::verify_conflicting_block_at_retained_position(
                    connection,
                    producer,
                    first,
                    observed_at,
                )?;
                Self::verify_conflicting_block_at_retained_position(
                    connection,
                    producer,
                    block,
                    observed_at,
                )?;
                return Ok(Some(VerifiedPageBlockFork {
                    height: block.header.height,
                    first_hash,
                    conflicting_hash: block_hash,
                }));
            }
            first_by_height.insert(block.header.height, block);
        }
        Ok(None)
    }

    pub(super) fn verify_conflicting_block_at_retained_position(
        connection: &Connection,
        producer: &[u8; 32],
        block: &DirectoryCommitmentBlockV1,
        observed_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        // [DIRECTORY-CONFLICT-VERIFICATION 2026-09-01 by Codex] A response
        // envelope cannot substitute for the embedded producer signature.
        // Verify every conflicting block against the exact transaction-visible
        // predecessor before either import path may persist producer blame.
        if block.header.producer != *producer {
            return Err(DirectoryReplicaStoreError::Integrity(
                "conflicting block is signed for another producer".to_string(),
            ));
        }
        let (previous_hash, previous_timestamp) = if block.header.height == 1 {
            ([0u8; 32], 0)
        } else {
            let previous_height = block.header.height.checked_sub(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "conflicting replica block height is invalid".to_string(),
                )
            })?;
            let previous = connection
                .query_row(
                    "SELECT block_hash, produced_at
                     FROM directory_replica_blocks
                     WHERE producer = ?1 AND height = ?2",
                    params![
                        producer.as_slice(),
                        u64_to_i64(previous_height, "replica predecessor height")?
                    ],
                    |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?)),
                )
                .optional()?;
            let Some((previous_hash, previous_timestamp)) = previous else {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "conflicting replica block predecessor is missing".to_string(),
                ));
            };
            (
                bytes32(&previous_hash, "replica predecessor hash")?,
                positive_i64_to_u64(previous_timestamp, "replica predecessor timestamp")?,
            )
        };
        block.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            block.header.height,
            &previous_hash,
            previous_timestamp,
            observed_at,
        )?;
        Ok(())
    }

    pub(super) fn insert_block(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        block: &DirectoryCommitmentBlockV1,
        descriptors: &[&SignedNodeDescriptor],
        evidence_frame: &[u8],
        observed_at: u64,
    ) -> Result<(u64, u64), DirectoryReplicaStoreError> {
        if descriptors.len() != block.commitments.len() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "descriptor count does not match block commitments".to_string(),
            ));
        }
        let height = u64_to_i64(block.header.height, "replica block height")?;
        transaction.execute(
            "INSERT INTO directory_replica_blocks
                (producer, height, block_hash, prev_block_hash, produced_at,
                 commitment_count, block_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                producer.as_slice(),
                height,
                block.hash().as_slice(),
                block.header.prev_block_hash.as_slice(),
                u64_to_i64(block.header.timestamp, "replica block timestamp")?,
                i64::from(block.header.commitment_count),
                encode_block(block)?
            ],
        )?;
        let mut inserted = 0u64;
        let mut equivocations = 0u64;
        for (commitment, descriptor) in block.commitments.iter().zip(descriptors) {
            let derived = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            if derived != *commitment {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "descriptor object does not match its block commitment".to_string(),
                ));
            }
            let conflicting = transaction
                .query_row(
                    "SELECT descriptor_hash FROM directory_replica_commitments
                     WHERE producer = ?1 AND node_id = ?2 AND sequence_le = ?3
                       AND descriptor_hash != ?4 LIMIT 1",
                    params![
                        producer.as_slice(),
                        commitment.node_id.as_slice(),
                        commitment.sequence.to_le_bytes().as_slice(),
                        commitment.descriptor_hash.as_slice()
                    ],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .optional()?;
            if let Some(conflicting) = conflicting {
                let conflicting = bytes32(&conflicting, "equivocation descriptor hash")?;
                let incident = QuarantineIncident {
                    kind: "descriptor_sequence_equivocation",
                    height: block.header.height,
                    local_hash: conflicting,
                    remote_hash: commitment.descriptor_hash,
                    evidence_frame,
                };
                if Self::insert_incident(
                    transaction,
                    producer,
                    &commitment.node_id,
                    &incident,
                    observed_at,
                )? {
                    equivocations = equivocations.saturating_add(1);
                }
            }
            transaction.execute(
                "INSERT INTO directory_replica_descriptor_objects
                    (producer, descriptor_hash, node_id, sequence_le, descriptor_blob)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    producer.as_slice(),
                    commitment.descriptor_hash.as_slice(),
                    commitment.node_id.as_slice(),
                    commitment.sequence.to_le_bytes().as_slice(),
                    encode_descriptor_object(descriptor)?
                ],
            )?;
            transaction.execute(
                "INSERT INTO directory_replica_commitments
                    (producer, commitment_hash, node_id, sequence_le,
                     descriptor_hash, block_height)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    producer.as_slice(),
                    commitment.hash().as_slice(),
                    commitment.node_id.as_slice(),
                    commitment.sequence.to_le_bytes().as_slice(),
                    commitment.descriptor_hash.as_slice(),
                    height
                ],
            )?;
            inserted = inserted.saturating_add(1);
        }
        Ok((inserted, equivocations))
    }

    pub(super) fn block_hash_at(
        connection: &Connection,
        producer: &[u8; 32],
        height: u64,
    ) -> Result<Option<[u8; 32]>, DirectoryReplicaStoreError> {
        let value = connection
            .query_row(
                "SELECT block_hash FROM directory_replica_blocks
                 WHERE producer = ?1 AND height = ?2",
                params![
                    producer.as_slice(),
                    u64_to_i64(height, "replica block height")?
                ],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        value
            .as_deref()
            .map(|bytes| bytes32(bytes, "replica block hash"))
            .transpose()
    }

    pub(super) fn load_tip(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<DirectoryReplicaTip, DirectoryReplicaStoreError> {
        let row = connection
            .query_row(
                "SELECT tip_height, tip_hash, tip_timestamp, quarantined, quarantine_kind,
                        active_incident_digest, last_resolution_digest
                 FROM directory_replica_chains WHERE producer = ?1",
                params![producer.as_slice()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<Vec<u8>>>(5)?,
                        row.get::<_, Option<Vec<u8>>>(6)?,
                    ))
                },
            )
            .optional()?;
        let Some((
            height,
            hash,
            timestamp,
            quarantined,
            quarantine_kind,
            active_incident_digest,
            last_resolution_digest,
        )) = row
        else {
            return Ok(DirectoryReplicaTip::empty(*producer));
        };
        if quarantined != 0 && quarantined != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica quarantine flag is invalid".to_string(),
            ));
        }
        if quarantined == 1
            && (quarantine_kind.as_deref().unwrap_or_default().is_empty()
                || active_incident_digest.is_none())
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "quarantined producer is missing its active incident".to_string(),
            ));
        }
        if quarantined == 0 && (quarantine_kind.is_some() || active_incident_digest.is_some()) {
            return Err(DirectoryReplicaStoreError::Integrity(
                "non-quarantined producer retains active incident state".to_string(),
            ));
        }
        Ok(DirectoryReplicaTip {
            producer: *producer,
            tip_height: nonnegative_i64_to_u64(height, "replica tip height")?,
            tip_hash: bytes32(&hash, "replica tip hash")?,
            tip_timestamp: nonnegative_i64_to_u64(timestamp, "replica tip timestamp")?,
            quarantined: quarantined == 1,
            quarantine_kind,
            active_incident_digest: active_incident_digest
                .map(|value| bytes32(&value, "replica active incident digest"))
                .transpose()?,
            last_resolution_digest: last_resolution_digest
                .map(|value| bytes32(&value, "replica last resolution digest"))
                .transpose()?,
        })
    }

    pub(super) fn load_all_tips(
        connection: &Connection,
    ) -> Result<Vec<DirectoryReplicaTip>, DirectoryReplicaStoreError> {
        let mut statement = connection
            .prepare("SELECT producer FROM directory_replica_chains ORDER BY producer ASC")?;
        let producers = statement
            .query_map([], |row| row.get::<_, Vec<u8>>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        producers
            .into_iter()
            .map(|producer| {
                let producer = bytes32(&producer, "replica producer")?;
                Self::load_tip(connection, &producer)
            })
            .collect()
    }

    pub(super) fn load_verified_commitments_for_block(
        connection: &Connection,
        producer: &[u8; 32],
        block_height: i64,
    ) -> Result<Vec<DirectoryDescriptorCommitmentV1>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT c.commitment_hash, c.node_id, c.sequence_le, c.descriptor_hash,
                    o.node_id, o.sequence_le, length(o.descriptor_blob),
                    CASE WHEN length(o.descriptor_blob) <= ?3
                         THEN o.descriptor_blob END
             FROM directory_replica_commitments c
             LEFT JOIN directory_replica_descriptor_objects o
               ON o.producer = c.producer
              AND o.descriptor_hash = c.descriptor_hash
             WHERE c.producer = ?1 AND c.block_height = ?2
             ORDER BY c.commitment_hash ASC",
        )?;
        let rows = statement.query_map(
            params![
                producer.as_slice(),
                block_height,
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
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, Option<Vec<u8>>>(4)?,
                    row.get::<_, Option<Vec<u8>>>(5)?,
                    row.get::<_, Option<i64>>(6)?,
                    row.get::<_, Option<Vec<u8>>>(7)?,
                ))
            },
        )?;
        let mut commitments = Vec::with_capacity(MAX_DIRECTORY_COMMITMENTS_PER_BLOCK);
        for row in rows {
            let (
                hash,
                node_id,
                sequence,
                descriptor_hash,
                object_node_id,
                object_sequence,
                object_blob_length,
                object_blob,
            ) = row?;
            if commitments.len() >= MAX_DIRECTORY_COMMITMENTS_PER_BLOCK {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica block commitment index exceeds the protocol limit".to_string(),
                ));
            }
            let sequence: [u8; 8] = sequence.try_into().map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "replica commitment sequence must contain 8 bytes".to_string(),
                )
            })?;
            let commitment = DirectoryDescriptorCommitmentV1 {
                node_id: bytes32(&node_id, "replica commitment node id")?,
                sequence: u64::from_le_bytes(sequence),
                descriptor_hash: bytes32(&descriptor_hash, "replica commitment descriptor hash")?,
            };
            if bytes32(&hash, "replica commitment hash")? != commitment.hash() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica commitment content hash mismatch".to_string(),
                ));
            }
            let object_node_id = object_node_id.ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "replica commitment is missing its descriptor object".to_string(),
                )
            })?;
            let object_sequence = object_sequence.ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "replica commitment is missing its descriptor sequence".to_string(),
                )
            })?;
            let object_blob = materialize_admitted_replica_blob(
                object_blob_length,
                object_blob,
                PersistedReplicaBlobKind::Descriptor,
            )?;
            let descriptor = decode_descriptor_object(&object_blob)?;
            let object_commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                    .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            let object_sequence: [u8; 8] = object_sequence.try_into().map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "replica descriptor sequence must contain 8 bytes".to_string(),
                )
            })?;
            if object_commitment != commitment
                || bytes32(&object_node_id, "replica descriptor node id")? != commitment.node_id
                || u64::from_le_bytes(object_sequence) != commitment.sequence
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica descriptor object index mismatch".to_string(),
                ));
            }
            commitments.push(commitment);
        }
        Ok(commitments)
    }

    pub(super) fn ensure_producer_row(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        observed_at: u64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute(
            "INSERT OR IGNORE INTO directory_replica_chains
                (producer, tip_height, tip_hash, tip_timestamp,
                 quarantined, quarantine_kind, active_incident_digest,
                 last_resolution_digest, updated_at)
             VALUES (?1, 0, ?2, 0, 0, NULL, NULL, NULL, ?3)",
            params![
                producer.as_slice(),
                [0u8; 32].as_slice(),
                u64_to_i64(observed_at, "replica observed timestamp")?
            ],
        )?;
        Ok(())
    }

    pub(super) fn producer_row_exists(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let count: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM directory_replica_chains WHERE producer = ?1",
            params![producer.as_slice()],
            |row| row.get(0),
        )?;
        match count {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DirectoryReplicaStoreError::Integrity(
                "directory replica producer primary key is inconsistent".to_string(),
            )),
        }
    }

    pub(super) fn audit_producer(
        connection: &Connection,
        tip: &DirectoryReplicaTip,
        observed_at: u64,
        report: &mut DirectoryReplicaAudit,
    ) -> Result<(), DirectoryReplicaStoreError> {
        report.producers = report.producers.saturating_add(1);
        if tip.quarantined {
            report.quarantined_producers = report.quarantined_producers.saturating_add(1);
            let incident_exists = connection
                .query_row(
                    "SELECT 1 FROM directory_replica_incidents
                     WHERE incident_digest = ?2 AND producer = ?1
                       AND subject_node_id = ?1 AND kind = ?3 LIMIT 1",
                    params![
                        tip.producer.as_slice(),
                        tip.active_incident_digest
                            .as_ref()
                            .map(<[u8; 32]>::as_slice),
                        tip.quarantine_kind.as_deref().unwrap_or_default()
                    ],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();
            if !incident_exists {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "quarantined producer is missing matching signed incident evidence".to_string(),
                ));
            }
        }
        // [DIRECTORY-STREAMING-AUDIT 2026-08-31 by Codex] A full-node mirror
        // retains an append-only history. Keep restart/export audit memory
        // bounded by one protocol-limited block instead of collecting three
        // complete producer indexes before verification.
        let mut statement = connection.prepare(
            "SELECT height, block_hash, prev_block_hash, produced_at,
                    commitment_count, length(block_blob),
                    CASE WHEN length(block_blob) <= ?2 THEN block_blob END
             FROM directory_replica_blocks WHERE producer = ?1 ORDER BY height ASC",
        )?;
        let mut rows = statement.query(params![
            tip.producer.as_slice(),
            u64_to_i64(MAX_DIRECTORY_BLOCK_BYTES, "replica block byte limit")?
        ])?;
        let mut expected_height = 1u64;
        let mut previous_hash = [0u8; 32];
        let mut previous_timestamp = 0u64;
        let mut audited_commitments = 0u64;
        while let Some(row) = rows.next()? {
            let block_blob = materialize_admitted_replica_blob(
                row.get::<_, Option<i64>>(5)?,
                row.get::<_, Option<Vec<u8>>>(6)?,
                PersistedReplicaBlobKind::Block,
            )?;
            let row = StoredReplicaBlockRow {
                height: row.get(0)?,
                block_hash: row.get(1)?,
                prev_block_hash: row.get(2)?,
                produced_at: row.get(3)?,
                commitment_count: row.get(4)?,
                block_blob,
            };
            let block = decode_block(&row.block_blob)?;
            let height = positive_i64_to_u64(row.height, "replica block height")?;
            if block.header.producer != tip.producer
                || height != expected_height
                || height != block.header.height
                || bytes32(&row.block_hash, "stored replica block hash")? != block.hash()
                || bytes32(&row.prev_block_hash, "stored replica previous hash")?
                    != block.header.prev_block_hash
                || positive_i64_to_u64(row.produced_at, "replica produced timestamp")?
                    != block.header.timestamp
                || nonnegative_i64_to_u64(row.commitment_count, "replica commitment count")?
                    != u64::from(block.header.commitment_count)
            {
                return Err(DirectoryReplicaStoreError::Integrity(format!(
                    "replica block {height} columns do not match its signed object"
                )));
            }
            block.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_height,
                &previous_hash,
                previous_timestamp,
                observed_at,
            )?;
            let mut actual =
                Self::load_verified_commitments_for_block(connection, &tip.producer, row.height)?;
            actual.sort_unstable();
            if actual != block.commitments {
                return Err(DirectoryReplicaStoreError::Integrity(format!(
                    "replica block {height} commitment index mismatch"
                )));
            }
            report.blocks = report.blocks.saturating_add(1);
            report.commitments = report
                .commitments
                .saturating_add(u64::from(block.header.commitment_count));
            audited_commitments = audited_commitments
                .checked_add(u64::from(block.header.commitment_count))
                .ok_or_else(|| {
                    DirectoryReplicaStoreError::Integrity(
                        "replica producer commitment count exhausted".to_string(),
                    )
                })?;
            #[cfg(test)]
            notify_directory_replica_audit_test_observer(
                DirectoryReplicaAuditTestEvent::BlockVerified(height),
            );
            previous_hash = block.hash();
            previous_timestamp = block.header.timestamp;
            expected_height = expected_height.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity("replica height exhausted".to_string())
            })?;
        }
        drop(rows);
        drop(statement);

        let (stored_commitments, stored_objects): (i64, i64) = connection.query_row(
            "SELECT
                (SELECT COUNT(*) FROM directory_replica_commitments
                 WHERE producer = ?1),
                (SELECT COUNT(*) FROM directory_replica_descriptor_objects
                 WHERE producer = ?1)",
            params![tip.producer.as_slice()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let stored_commitments =
            nonnegative_i64_to_u64(stored_commitments, "replica stored commitment count")?;
        let stored_objects =
            nonnegative_i64_to_u64(stored_objects, "replica stored descriptor object count")?;
        if stored_commitments != audited_commitments || stored_objects != audited_commitments {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica contains orphaned commitment or descriptor indexes".to_string(),
            ));
        }
        let audited_height = expected_height.saturating_sub(1);
        if tip.tip_height != audited_height
            || tip.tip_hash != previous_hash
            || tip.tip_timestamp != previous_timestamp
            || (tip.tip_height == 0 && tip.tip_hash != [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "replica producer tip does not match its accepted block prefix".to_string(),
            ));
        }
        Ok(())
    }
}
