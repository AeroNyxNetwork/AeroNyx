// [ARCH-SPLIT 2026-10-02]
// Schema create, migrate, and connection audit. open() calls this before serving reads.
// Bodies are unchanged. Private items are pub(super) so the parent flow can call them.
use super::*;

impl DirectoryReplicaStore {
    /// Audits metadata, evidence, resolutions, prefixes, indexes, and retries.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] on the first malformed row,
    /// invalid signature/link/root, missing object/index, or invalid incident.
    pub fn audit(
        &self,
        observed_at: u64,
    ) -> Result<DirectoryReplicaAudit, DirectoryReplicaStoreError> {
        let mut connection = self.connection.lock();
        // [DIRECTORY-AUDIT-SNAPSHOT 2026-08-31 by Codex] The streaming block
        // cursor, its per-block index lookups, final orphan counts, and tip
        // comparison must describe one SQLite state. A deferred read
        // transaction keeps WAL writers concurrent while preventing a valid
        // append or hostile index swap from splicing multiple snapshots into
        // one audit result.
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Deferred)
            .map_err(DirectoryReplicaStoreError::Sqlite)?;
        match Self::audit_connection(&transaction, &self.local_node_id, observed_at) {
            Ok(report) => {
                transaction
                    .commit()
                    .map_err(DirectoryReplicaStoreError::Sqlite)?;
                Ok(report)
            }
            Err(audit_error) => match transaction.rollback() {
                Ok(()) => Err(audit_error),
                Err(rollback_error) => Err(DirectoryReplicaStoreError::Sqlite(rollback_error)),
            },
        }
    }

    pub(super) fn audit_connection(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<DirectoryReplicaAudit, DirectoryReplicaStoreError> {
        Self::validate_metadata(connection, local_node_id)?;
        let producers = Self::load_all_tips(connection)?;
        let mirror_producers = Self::audit_mirror_registry(connection, local_node_id)?;
        let observation_witness_policies =
            Self::audit_observation_witness_policies(connection, local_node_id)?;
        let route_domain_policies = Self::audit_route_domain_policies(connection, local_node_id)?;
        let route_domain_attestor_policies =
            Self::audit_route_domain_attestor_policies(connection, local_node_id)?;
        let observation_witness_remote_policy_anchors =
            Self::audit_remote_observation_policy_anchors(connection, local_node_id, observed_at)?;
        let observation_witness_policy_anchor_receipts = u64::try_from(
            Self::audit_observation_policy_anchor_receipts(connection, local_node_id, observed_at)?
                .len(),
        )
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt count exceeds u64".to_string(),
            )
        })?;
        let observation_certificate_imports =
            Self::audit_observation_certificate_imports(connection, local_node_id, observed_at)?;
        let (observation_checkpoints, observation_tip) =
            Self::audit_observation_checkpoints(connection, local_node_id, observed_at)?;
        let observation_witnesses =
            Self::audit_observation_witnesses(connection, local_node_id, observed_at)?;
        let observation_witness_outcomes =
            Self::load_observation_witness_outcome_snapshot(connection)?;
        if observation_witness_outcomes.last_checkpoint_sequence > observation_tip.sequence {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome references an unknown checkpoint".to_string(),
            ));
        }
        let mut report = DirectoryReplicaAudit {
            mirror_producers,
            incidents: Self::audit_incidents(connection)?,
            resolutions: Self::audit_resolutions(connection, local_node_id, &producers)?,
            observation_checkpoints,
            observation_checkpoint_sequence: observation_tip.sequence,
            observation_checkpoint_hash: observation_tip.checkpoint_hash,
            observation_checkpoint_observed_at: observation_tip.observed_at,
            observation_checkpoint_witnesses: observation_witnesses.witnesses,
            observation_checkpoint_witnessed_sequence: observation_witnesses.latest_sequence,
            observation_checkpoint_latest_witnesses: observation_witnesses.latest_witnesses,
            observation_witness_outcomes,
            observation_witness_policy_epochs: observation_witness_policies.epochs,
            observation_witness_policy_epoch: observation_witness_policies.epochs,
            observation_witness_policy_activated_at: observation_witness_policies
                .current
                .as_ref()
                .map_or(0, |policy| policy.activated_at),
            observation_witness_policy_members: observation_witness_policies
                .current
                .as_ref()
                .map(|policy| u64::try_from(policy.witness_node_ids.len()))
                .transpose()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy member count exceeds u64".to_string(),
                    )
                })?
                .unwrap_or(0),
            observation_witness_policy_threshold: observation_witness_policies
                .current
                .as_ref()
                .map(|policy| u64::try_from(policy.minimum_witnesses))
                .transpose()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy threshold exceeds u64".to_string(),
                    )
                })?
                .unwrap_or(0),
            observation_witness_policy_anchor_receipts,
            observation_witness_remote_policy_anchors,
            route_domain_policy_epochs: route_domain_policies.epochs,
            route_domain_policy_epoch: route_domain_policies.epochs,
            route_domain_policy_activated_at: route_domain_policies
                .current
                .as_ref()
                .map_or(0, |policy| policy.activated_at),
            route_domain_policy_assignments: route_domain_policies
                .current
                .as_ref()
                .map(|policy| u64::try_from(policy.assignments.len()))
                .transpose()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain policy assignment count exceeds u64".to_string(),
                    )
                })?
                .unwrap_or(0),
            route_domain_policy_strict: route_domain_policies
                .current
                .is_some_and(|policy| policy.strict_required),
            route_domain_attestor_policy_epochs: route_domain_attestor_policies.epochs,
            route_domain_attestor_policy_epoch: route_domain_attestor_policies.epochs,
            route_domain_attestor_policy_activated_at: route_domain_attestor_policies
                .current
                .as_ref()
                .map_or(0, |policy| policy.activated_at),
            route_domain_attestor_policy_members: route_domain_attestor_policies
                .current
                .as_ref()
                .map(|policy| u64::try_from(policy.attestor_node_ids.len()))
                .transpose()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain attestor policy member count exceeds u64".to_string(),
                    )
                })?
                .unwrap_or(0),
            route_domain_attestor_policy_threshold: route_domain_attestor_policies
                .current
                .as_ref()
                .map(|policy| u64::try_from(policy.minimum_attestors))
                .transpose()
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "route-domain attestor policy threshold exceeds u64".to_string(),
                    )
                })?
                .unwrap_or(0),
            route_domain_attestor_policy_strict: route_domain_attestor_policies
                .current
                .is_some_and(|policy| policy.strict_required),
            imported_observation_certificates: observation_certificate_imports.imports,
            imported_observation_certificate_sequence: observation_certificate_imports.imports,
            imported_observation_certificate_head: observation_certificate_imports.head,
            ..DirectoryReplicaAudit::default()
        };
        for tip in producers {
            Self::audit_producer(connection, &tip, observed_at, &mut report)?;
        }
        report.retry_states = u64::try_from(
            Self::load_retry_states(connection, local_node_id)?.len(),
        )
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "replica retry state count exceeds platform bounds".to_string(),
            )
        })?;
        Ok(report)
    }

    pub(super) fn audit_mirror_registry(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<u64, DirectoryReplicaStoreError> {
        let orphaned: i64 = connection.query_row(
            "SELECT COUNT(*)
             FROM directory_replica_mirror_producers m
             LEFT JOIN directory_replica_chains c ON c.producer = m.producer
             WHERE c.producer IS NULL",
            [],
            |row| row.get(0),
        )?;
        if orphaned != 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory mirror registry contains an orphaned producer".to_string(),
            ));
        }
        let mut statement = connection.prepare(
            "SELECT producer, admitted_at, last_selected_at, descriptor_sequence
             FROM directory_replica_mirror_producers ORDER BY producer ASC",
        )?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        if rows.len() > MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory mirror registry exceeds the protocol capacity".to_string(),
            ));
        }
        for (producer, admitted_at, last_selected_at, descriptor_sequence) in &rows {
            let producer = bytes32(producer, "directory mirror producer")?;
            let admitted_at = positive_i64_to_u64(*admitted_at, "directory mirror admission")?;
            let last_selected_at =
                positive_i64_to_u64(*last_selected_at, "directory mirror selection")?;
            let descriptor_sequence =
                positive_i64_to_u64(*descriptor_sequence, "directory mirror descriptor sequence")?;
            if producer == [0u8; 32]
                || producer == *local_node_id
                || last_selected_at < admitted_at
                || descriptor_sequence == 0
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "directory mirror registry row is invalid".to_string(),
                ));
            }
        }
        u64::try_from(rows.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "directory mirror registry count exceeds u64".to_string(),
            )
        })
    }

    pub(super) fn initialize_schema(
        connection: &mut Connection,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::create_schema_tables(&transaction)?;
        Self::migrate_schema_metadata(&transaction, local_node_id)?;
        transaction.commit()?;
        Self::validate_metadata(connection, local_node_id)
    }

    // Keeping the versioned DDL in one batch makes partial table creation
    // impossible and lets reviewers compare the complete schema in one place.
    #[allow(clippy::too_many_lines)]
    pub(super) fn create_schema_tables(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute_batch(
            "CREATE TABLE IF NOT EXISTS directory_replica_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32),
                 witness_policy_epoch INTEGER NOT NULL DEFAULT 0
                     CHECK (witness_policy_epoch >= 0),
                 witness_policy_head BLOB
                     CHECK (witness_policy_head IS NULL OR length(witness_policy_head) = 32),
                 route_domain_policy_epoch INTEGER NOT NULL DEFAULT 0
                     CHECK (route_domain_policy_epoch >= 0),
                 route_domain_policy_head BLOB
                     CHECK (route_domain_policy_head IS NULL
                         OR length(route_domain_policy_head) = 32),
                 route_domain_attestor_policy_epoch INTEGER NOT NULL DEFAULT 0
                     CHECK (route_domain_attestor_policy_epoch >= 0),
                 route_domain_attestor_policy_head BLOB
                     CHECK (route_domain_attestor_policy_head IS NULL
                         OR length(route_domain_attestor_policy_head) = 32),
                 certificate_import_sequence INTEGER NOT NULL DEFAULT 0
                     CHECK (certificate_import_sequence >= 0),
                 certificate_import_head BLOB
                     CHECK (certificate_import_head IS NULL
                         OR length(certificate_import_head) = 32),
                 CHECK ((witness_policy_epoch = 0 AND witness_policy_head IS NULL)
                     OR (witness_policy_epoch > 0 AND witness_policy_head IS NOT NULL)),
                 CHECK ((route_domain_policy_epoch = 0
                         AND route_domain_policy_head IS NULL)
                     OR (route_domain_policy_epoch > 0
                         AND route_domain_policy_head IS NOT NULL)),
                 CHECK ((route_domain_attestor_policy_epoch = 0
                         AND route_domain_attestor_policy_head IS NULL)
                     OR (route_domain_attestor_policy_epoch > 0
                         AND route_domain_attestor_policy_head IS NOT NULL)),
                 CHECK ((certificate_import_sequence = 0
                         AND certificate_import_head IS NULL)
                     OR (certificate_import_sequence > 0
                         AND certificate_import_head IS NOT NULL))
             );
             CREATE TABLE IF NOT EXISTS directory_replica_chains (
                 producer BLOB PRIMARY KEY CHECK (length(producer) = 32),
                 tip_height INTEGER NOT NULL CHECK (tip_height >= 0),
                 tip_hash BLOB NOT NULL CHECK (length(tip_hash) = 32),
                 tip_timestamp INTEGER NOT NULL CHECK (tip_timestamp >= 0),
                 quarantined INTEGER NOT NULL CHECK (quarantined IN (0, 1)),
                 quarantine_kind TEXT,
                 active_incident_digest BLOB
                     CHECK (active_incident_digest IS NULL OR length(active_incident_digest) = 32),
                 last_resolution_digest BLOB
                     CHECK (last_resolution_digest IS NULL OR length(last_resolution_digest) = 32),
                 updated_at INTEGER NOT NULL CHECK (updated_at > 0)
             );
             CREATE TABLE IF NOT EXISTS directory_replica_mirror_producers (
                 producer BLOB PRIMARY KEY CHECK (length(producer) = 32),
                 admitted_at INTEGER NOT NULL CHECK (admitted_at > 0),
                 last_selected_at INTEGER NOT NULL CHECK (last_selected_at >= admitted_at),
                 descriptor_sequence INTEGER NOT NULL CHECK (descriptor_sequence > 0),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_replica_mirrors_by_selection
                 ON directory_replica_mirror_producers(last_selected_at, producer);
             CREATE TABLE IF NOT EXISTS directory_replica_blocks (
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 height INTEGER NOT NULL CHECK (height > 0),
                 block_hash BLOB NOT NULL CHECK (length(block_hash) = 32),
                 prev_block_hash BLOB NOT NULL CHECK (length(prev_block_hash) = 32),
                 produced_at INTEGER NOT NULL CHECK (produced_at > 0),
                 commitment_count INTEGER NOT NULL CHECK (commitment_count > 0),
                 block_blob BLOB NOT NULL,
                 PRIMARY KEY (producer, height),
                 UNIQUE (producer, block_hash),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_replica_descriptor_objects (
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 descriptor_hash BLOB NOT NULL CHECK (length(descriptor_hash) = 32),
                 node_id BLOB NOT NULL CHECK (length(node_id) = 32),
                 sequence_le BLOB NOT NULL CHECK (length(sequence_le) = 8),
                 descriptor_blob BLOB NOT NULL,
                 PRIMARY KEY (producer, descriptor_hash),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_replica_commitments (
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 commitment_hash BLOB NOT NULL CHECK (length(commitment_hash) = 32),
                 node_id BLOB NOT NULL CHECK (length(node_id) = 32),
                 sequence_le BLOB NOT NULL CHECK (length(sequence_le) = 8),
                 descriptor_hash BLOB NOT NULL CHECK (length(descriptor_hash) = 32),
                 block_height INTEGER NOT NULL CHECK (block_height > 0),
                 PRIMARY KEY (producer, commitment_hash),
                 UNIQUE (producer, descriptor_hash),
                 FOREIGN KEY (producer, block_height)
                     REFERENCES directory_replica_blocks(producer, height)
                     ON UPDATE RESTRICT ON DELETE RESTRICT,
                 FOREIGN KEY (producer, descriptor_hash)
                     REFERENCES directory_replica_descriptor_objects(producer, descriptor_hash)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_replica_commitments_by_block
                 ON directory_replica_commitments(producer, block_height, commitment_hash);
             CREATE INDEX IF NOT EXISTS directory_replica_commitments_by_subject
                 ON directory_replica_commitments(producer, node_id, sequence_le);
             CREATE TABLE IF NOT EXISTS directory_replica_incidents (
                 incident_digest BLOB PRIMARY KEY CHECK (length(incident_digest) = 32),
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 subject_node_id BLOB NOT NULL CHECK (length(subject_node_id) = 32),
                 kind TEXT NOT NULL,
                 height INTEGER NOT NULL CHECK (height >= 0),
                 local_hash BLOB NOT NULL CHECK (length(local_hash) = 32),
                 remote_hash BLOB NOT NULL CHECK (length(remote_hash) = 32),
                 evidence_frame BLOB NOT NULL,
                 observed_at INTEGER NOT NULL CHECK (observed_at > 0),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_replica_resolutions (
                 resolution_digest BLOB PRIMARY KEY CHECK (length(resolution_digest) = 32),
                 command_id BLOB NOT NULL UNIQUE CHECK (length(command_id) = 16),
                 incident_digest BLOB NOT NULL CHECK (length(incident_digest) = 32),
                 producer BLOB NOT NULL CHECK (length(producer) = 32),
                 action TEXT NOT NULL CHECK (action = 'resume_existing_prefix'),
                 expected_tip_height INTEGER NOT NULL CHECK (expected_tip_height >= 0),
                 expected_tip_hash BLOB NOT NULL CHECK (length(expected_tip_hash) = 32),
                 expected_quarantine_kind TEXT NOT NULL,
                 previous_resolution_digest BLOB
                     CHECK (previous_resolution_digest IS NULL OR length(previous_resolution_digest) = 32),
                 resolved_at INTEGER NOT NULL CHECK (resolved_at > 0),
                 resolver_node_id BLOB NOT NULL CHECK (length(resolver_node_id) = 32),
                 signature BLOB NOT NULL CHECK (length(signature) = 64),
                 FOREIGN KEY (incident_digest)
                     REFERENCES directory_replica_incidents(incident_digest)
                     ON UPDATE RESTRICT ON DELETE RESTRICT,
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT,
                 FOREIGN KEY (previous_resolution_digest)
                     REFERENCES directory_replica_resolutions(resolution_digest)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_replica_resolutions_by_producer
                 ON directory_replica_resolutions(producer, resolved_at, resolution_digest);
             CREATE TABLE IF NOT EXISTS directory_observation_checkpoints (
                 sequence INTEGER PRIMARY KEY CHECK (sequence > 0),
                 checkpoint_hash BLOB NOT NULL UNIQUE CHECK (length(checkpoint_hash) = 32),
                 previous_checkpoint_hash BLOB NOT NULL UNIQUE
                     CHECK (length(previous_checkpoint_hash) = 32),
                 observed_at INTEGER NOT NULL CHECK (observed_at > 0),
                 observation_root BLOB NOT NULL CHECK (length(observation_root) = 32),
                 producer_count INTEGER NOT NULL CHECK (producer_count BETWEEN 2 AND 16),
                 checkpoint_blob BLOB NOT NULL
                     CHECK (length(checkpoint_blob) BETWEEN 1 AND 4096)
             );
             CREATE TABLE IF NOT EXISTS directory_observation_checkpoint_witnesses (
                 checkpoint_hash BLOB NOT NULL CHECK (length(checkpoint_hash) = 32),
                 checkpoint_sequence INTEGER NOT NULL CHECK (checkpoint_sequence > 0),
                 observer BLOB NOT NULL CHECK (length(observer) = 32),
                 witness_node_id BLOB NOT NULL CHECK (length(witness_node_id) = 32),
                 witnessed_at INTEGER NOT NULL CHECK (witnessed_at > 0),
                 response_blob BLOB NOT NULL
                     CHECK (length(response_blob) BETWEEN 1 AND 2048),
                 PRIMARY KEY (checkpoint_hash, witness_node_id),
                 UNIQUE (checkpoint_sequence, witness_node_id),
                 FOREIGN KEY (checkpoint_hash)
                     REFERENCES directory_observation_checkpoints(checkpoint_hash)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_observation_witnesses_by_sequence
                 ON directory_observation_checkpoint_witnesses(
                     checkpoint_sequence, witness_node_id
                 );
             CREATE TABLE IF NOT EXISTS directory_observation_witness_outcomes (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 rounds_total INTEGER NOT NULL CHECK (rounds_total > 0),
                 attempts_total INTEGER NOT NULL CHECK (attempts_total > 0),
                 accepted_total INTEGER NOT NULL CHECK (accepted_total >= 0),
                 evidence_unavailable_total INTEGER NOT NULL
                     CHECK (evidence_unavailable_total >= 0),
                 evidence_conflict_total INTEGER NOT NULL
                     CHECK (evidence_conflict_total >= 0),
                 peer_unavailable_total INTEGER NOT NULL
                     CHECK (peer_unavailable_total >= 0),
                 transport_failures_total INTEGER NOT NULL
                     CHECK (transport_failures_total >= 0),
                 verification_failures_total INTEGER NOT NULL
                     CHECK (verification_failures_total >= 0),
                 persistence_failures_total INTEGER NOT NULL
                     CHECK (persistence_failures_total >= 0),
                 last_checkpoint_sequence INTEGER NOT NULL
                     CHECK (last_checkpoint_sequence > 0),
                 last_round_at INTEGER NOT NULL CHECK (last_round_at > 0),
                 last_success_at INTEGER CHECK (last_success_at > 0),
                 last_failure_at INTEGER CHECK (last_failure_at > 0),
                 last_round_attempts INTEGER NOT NULL
                     CHECK (last_round_attempts BETWEEN 1 AND 16),
                 last_round_accepted INTEGER NOT NULL CHECK (last_round_accepted >= 0),
                 last_round_evidence_unavailable INTEGER NOT NULL
                     CHECK (last_round_evidence_unavailable >= 0),
                 last_round_evidence_conflict INTEGER NOT NULL
                     CHECK (last_round_evidence_conflict >= 0),
                 last_round_peer_unavailable INTEGER NOT NULL
                     CHECK (last_round_peer_unavailable >= 0),
                 last_round_transport_failures INTEGER NOT NULL
                     CHECK (last_round_transport_failures >= 0),
                 last_round_verification_failures INTEGER NOT NULL
                     CHECK (last_round_verification_failures >= 0),
                 last_round_persistence_failures INTEGER NOT NULL
                     CHECK (last_round_persistence_failures >= 0),
                 updated_at INTEGER NOT NULL CHECK (updated_at > 0),
                 CHECK (attempts_total = accepted_total
                     + evidence_unavailable_total + evidence_conflict_total
                     + peer_unavailable_total + transport_failures_total
                     + verification_failures_total + persistence_failures_total),
                 CHECK (last_round_attempts = last_round_accepted
                     + last_round_evidence_unavailable + last_round_evidence_conflict
                     + last_round_peer_unavailable + last_round_transport_failures
                     + last_round_verification_failures
                     + last_round_persistence_failures),
                 CHECK (attempts_total >= last_round_attempts),
                 CHECK (last_success_at IS NOT NULL OR accepted_total = 0),
                 CHECK (last_failure_at IS NOT NULL OR attempts_total = accepted_total),
                 FOREIGN KEY (last_checkpoint_sequence)
                     REFERENCES directory_observation_checkpoints(sequence)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE TABLE IF NOT EXISTS directory_observation_witness_policies (
                 epoch INTEGER PRIMARY KEY CHECK (epoch > 0),
                 policy_digest BLOB NOT NULL UNIQUE CHECK (length(policy_digest) = 32),
                 previous_policy_digest BLOB NOT NULL UNIQUE
                     CHECK (length(previous_policy_digest) = 32),
                 activated_at INTEGER NOT NULL CHECK (activated_at > 0),
                 witness_threshold INTEGER NOT NULL
                     CHECK (witness_threshold BETWEEN 1 AND 16),
                 witness_count INTEGER NOT NULL CHECK (witness_count BETWEEN 0 AND 16),
                 witness_node_ids BLOB NOT NULL CHECK (length(witness_node_ids) <= 512),
                 signer_node_id BLOB NOT NULL CHECK (length(signer_node_id) = 32),
                 signature BLOB NOT NULL CHECK (length(signature) = 64),
                 CHECK (length(witness_node_ids) = witness_count * 32),
                 CHECK ((witness_count = 0 AND witness_threshold = 1)
                     OR (witness_count > 0 AND witness_threshold <= witness_count))
             );
             CREATE TABLE IF NOT EXISTS directory_observation_remote_policy_anchors (
                 observer BLOB NOT NULL CHECK (length(observer) = 32),
                 policy_epoch INTEGER NOT NULL CHECK (policy_epoch > 0),
                 previous_policy_digest BLOB NOT NULL
                     CHECK (length(previous_policy_digest) = 32),
                 policy_digest BLOB NOT NULL CHECK (length(policy_digest) = 32),
                 request_timestamp INTEGER NOT NULL CHECK (request_timestamp > 0),
                 request_blob BLOB NOT NULL
                     CHECK (length(request_blob) BETWEEN 1 AND 2048),
                 PRIMARY KEY (observer, policy_epoch),
                 UNIQUE (observer, policy_digest)
             );
             CREATE TABLE IF NOT EXISTS directory_observation_policy_anchor_receipts (
                 policy_epoch INTEGER NOT NULL CHECK (policy_epoch > 0),
                 policy_digest BLOB NOT NULL CHECK (length(policy_digest) = 32),
                 observer BLOB NOT NULL CHECK (length(observer) = 32),
                 witness_node_id BLOB NOT NULL CHECK (length(witness_node_id) = 32),
                 witnessed_at INTEGER NOT NULL CHECK (witnessed_at > 0),
                 response_blob BLOB NOT NULL
                     CHECK (length(response_blob) BETWEEN 1 AND 2048),
                 PRIMARY KEY (policy_digest, witness_node_id),
                 UNIQUE (policy_epoch, witness_node_id),
                 FOREIGN KEY (policy_digest)
                     REFERENCES directory_observation_witness_policies(policy_digest)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_policy_anchor_receipts_by_epoch
                 ON directory_observation_policy_anchor_receipts(
                     policy_epoch, witness_node_id
                 );
             CREATE TABLE IF NOT EXISTS directory_observation_certificate_imports (
                 import_sequence INTEGER PRIMARY KEY CHECK (import_sequence > 0),
                 import_digest BLOB NOT NULL UNIQUE CHECK (length(import_digest) = 32),
                 previous_import_digest BLOB NOT NULL UNIQUE
                     CHECK (length(previous_import_digest) = 32),
                 certificate_id BLOB NOT NULL UNIQUE CHECK (length(certificate_id) = 32),
                 observer BLOB NOT NULL CHECK (length(observer) = 32),
                 checkpoint_sequence INTEGER NOT NULL CHECK (checkpoint_sequence > 0),
                 checkpoint_hash BLOB NOT NULL CHECK (length(checkpoint_hash) = 32),
                 checkpoint_observed_at INTEGER NOT NULL CHECK (checkpoint_observed_at > 0),
                 certificate_sha256 BLOB NOT NULL UNIQUE
                     CHECK (length(certificate_sha256) = 32),
                 certificate_frame BLOB NOT NULL
                     CHECK (length(certificate_frame) BETWEEN 1 AND 65537),
                 policy_digest BLOB NOT NULL CHECK (length(policy_digest) = 32),
                 policy_minimum_witnesses INTEGER NOT NULL
                     CHECK (policy_minimum_witnesses BETWEEN 1 AND 16),
                 policy_witness_count INTEGER NOT NULL
                     CHECK (policy_witness_count BETWEEN 1 AND 16),
                 policy_witness_node_ids BLOB NOT NULL
                     CHECK (length(policy_witness_node_ids) BETWEEN 32 AND 512),
                 verified_at INTEGER NOT NULL CHECK (verified_at > 0),
                 importer_node_id BLOB NOT NULL CHECK (length(importer_node_id) = 32),
                 signature BLOB NOT NULL CHECK (length(signature) = 64),
                 UNIQUE (observer, checkpoint_sequence),
                 CHECK (policy_minimum_witnesses <= policy_witness_count),
                 CHECK (length(policy_witness_node_ids) = policy_witness_count * 32)
             );
             CREATE INDEX IF NOT EXISTS directory_observation_certificate_imports_by_observer
                 ON directory_observation_certificate_imports(
                     observer, checkpoint_sequence DESC
                 );
             CREATE TABLE IF NOT EXISTS directory_route_domain_policies (
                 epoch INTEGER PRIMARY KEY CHECK (epoch > 0),
                 policy_digest BLOB NOT NULL UNIQUE CHECK (length(policy_digest) = 32),
                 previous_policy_digest BLOB NOT NULL UNIQUE
                     CHECK (length(previous_policy_digest) = 32),
                 activated_at INTEGER NOT NULL CHECK (activated_at > 0),
                 strict_required INTEGER NOT NULL CHECK (strict_required IN (0, 1)),
                 assignment_count INTEGER NOT NULL
                     CHECK (assignment_count BETWEEN 0 AND 256),
                 assignments BLOB NOT NULL CHECK (length(assignments) <= 12288),
                 signer_node_id BLOB NOT NULL CHECK (length(signer_node_id) = 32),
                 signature BLOB NOT NULL CHECK (length(signature) = 64),
                 CHECK (length(assignments) = assignment_count * 48),
                 CHECK (strict_required = 0 OR assignment_count > 0)
             );
             CREATE TABLE IF NOT EXISTS directory_route_domain_attestor_policies (
                 epoch INTEGER PRIMARY KEY CHECK (epoch > 0),
                 policy_digest BLOB NOT NULL UNIQUE CHECK (length(policy_digest) = 32),
                 previous_policy_digest BLOB NOT NULL UNIQUE
                     CHECK (length(previous_policy_digest) = 32),
                 activated_at INTEGER NOT NULL CHECK (activated_at > 0),
                 strict_required INTEGER NOT NULL CHECK (strict_required IN (0, 1)),
                 attestor_threshold INTEGER NOT NULL
                     CHECK (attestor_threshold BETWEEN 1 AND 16),
                 attestor_count INTEGER NOT NULL
                     CHECK (attestor_count BETWEEN 0 AND 16),
                 attestor_node_ids BLOB NOT NULL CHECK (length(attestor_node_ids) <= 512),
                 signer_node_id BLOB NOT NULL CHECK (length(signer_node_id) = 32),
                 signature BLOB NOT NULL CHECK (length(signature) = 64),
                 CHECK (length(attestor_node_ids) = attestor_count * 32),
                 CHECK ((attestor_count = 0 AND attestor_threshold = 1
                         AND strict_required = 0)
                     OR (attestor_count > 0 AND attestor_threshold <= attestor_count))
             );
             CREATE TABLE IF NOT EXISTS directory_replica_retry_state (
                 producer BLOB PRIMARY KEY CHECK (length(producer) = 32),
                 consecutive_failures INTEGER NOT NULL
                     CHECK (consecutive_failures > 0 AND consecutive_failures <= 64),
                 retry_not_before INTEGER
                     CHECK (retry_not_before IS NULL OR retry_not_before > 0),
                 last_failure_at INTEGER NOT NULL CHECK (last_failure_at > 0),
                 last_failure_reason TEXT NOT NULL
                     CHECK (length(last_failure_reason) BETWEEN 1 AND 96),
                 backoff_skips INTEGER NOT NULL CHECK (backoff_skips >= 0),
                 updated_at INTEGER NOT NULL CHECK (updated_at > 0),
                 FOREIGN KEY (producer) REFERENCES directory_replica_chains(producer)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );",
        )?;
        Ok(())
    }

    pub(super) fn migrate_schema_metadata(
        transaction: &Transaction<'_>,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let existing: Option<(i64, Vec<u8>, Vec<u8>)> = transaction
            .query_row(
                "SELECT schema_version, chain_id, local_node_id
                 FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        match existing {
            None => {
                transaction.execute(
                    "INSERT INTO directory_replica_meta
                        (singleton, schema_version, chain_id, local_node_id)
                     VALUES (1, ?1, ?2, ?3)",
                    params![
                        DIRECTORY_REPLICA_SCHEMA_VERSION,
                        AERONYX_DIRECTORY_MAINNET_CHAIN_ID.as_slice(),
                        local_node_id.as_slice()
                    ],
                )?;
            }
            Some((version, chain_id, stored_local_node_id)) => {
                if bytes32(&chain_id, "replica metadata chain id")?
                    != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                    || bytes32(&stored_local_node_id, "replica metadata local node id")?
                        != *local_node_id
                {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "directory replica metadata identity is incompatible".to_string(),
                    ));
                }
                match version {
                    DIRECTORY_REPLICA_SCHEMA_VERSION => {
                        Self::require_resolution_columns(transaction)?;
                        Self::require_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
                        Self::require_certificate_import_metadata_columns(transaction)?;
                        Self::require_certificate_import_table(transaction)?;
                        Self::require_route_domain_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_policy_table(transaction)?;
                        Self::require_route_domain_attestor_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_attestor_policy_table(transaction)?;
                    }
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V11 => {
                        Self::require_resolution_columns(transaction)?;
                        Self::require_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
                        Self::require_certificate_import_metadata_columns(transaction)?;
                        Self::require_certificate_import_table(transaction)?;
                        Self::require_route_domain_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_policy_table(transaction)?;
                        Self::add_route_domain_attestor_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_attestor_policy_table(transaction)?;
                        Self::set_schema_version(transaction, version)?;
                    }
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V10 => {
                        Self::require_resolution_columns(transaction)?;
                        Self::require_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
                        Self::require_certificate_import_metadata_columns(transaction)?;
                        Self::require_certificate_import_table(transaction)?;
                        Self::add_route_domain_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_policy_table(transaction)?;
                        Self::add_route_domain_attestor_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_attestor_policy_table(transaction)?;
                        Self::set_schema_version(transaction, version)?;
                    }
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V9 => {
                        Self::require_resolution_columns(transaction)?;
                        Self::require_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
                        Self::add_certificate_import_metadata_columns(transaction)?;
                        Self::require_certificate_import_table(transaction)?;
                        Self::add_route_domain_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_policy_table(transaction)?;
                        Self::add_route_domain_attestor_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_attestor_policy_table(transaction)?;
                        Self::set_schema_version(transaction, version)?;
                    }
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V8
                    | DIRECTORY_REPLICA_SCHEMA_VERSION_V7
                    | DIRECTORY_REPLICA_SCHEMA_VERSION_V6
                    | DIRECTORY_REPLICA_SCHEMA_VERSION_V5
                    | DIRECTORY_REPLICA_SCHEMA_VERSION_V4
                    | DIRECTORY_REPLICA_SCHEMA_VERSION_V3 => {
                        Self::require_resolution_columns(transaction)?;
                        Self::add_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
                        Self::add_certificate_import_metadata_columns(transaction)?;
                        Self::require_certificate_import_table(transaction)?;
                        Self::add_route_domain_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_policy_table(transaction)?;
                        Self::add_route_domain_attestor_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_attestor_policy_table(transaction)?;
                        Self::set_schema_version(transaction, version)?;
                    }
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V1 | DIRECTORY_REPLICA_SCHEMA_VERSION_V2 => {
                        Self::add_resolution_columns(transaction)?;
                        Self::add_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
                        Self::add_certificate_import_metadata_columns(transaction)?;
                        Self::require_certificate_import_table(transaction)?;
                        Self::add_route_domain_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_policy_table(transaction)?;
                        Self::add_route_domain_attestor_policy_metadata_columns(transaction)?;
                        Self::require_route_domain_attestor_policy_table(transaction)?;
                        transaction.execute(
                            "UPDATE directory_replica_chains AS c
                             SET active_incident_digest = (
                                 SELECT i.incident_digest
                                 FROM directory_replica_incidents i
                                 WHERE i.producer = c.producer
                                   AND i.subject_node_id = c.producer
                                   AND i.kind = c.quarantine_kind
                                 ORDER BY i.observed_at DESC, i.incident_digest DESC
                                 LIMIT 1
                             )
                             WHERE c.quarantined = 1
                               AND c.active_incident_digest IS NULL",
                            [],
                        )?;
                        let missing_active: i64 = transaction.query_row(
                            "SELECT COUNT(*) FROM directory_replica_chains
                             WHERE quarantined = 1 AND active_incident_digest IS NULL",
                            [],
                            |row| row.get(0),
                        )?;
                        if missing_active != 0 {
                            return Err(DirectoryReplicaStoreError::Integrity(
                                "cannot migrate quarantined producer without matching incident"
                                    .to_string(),
                            ));
                        }
                        Self::set_schema_version(transaction, version)?;
                    }
                    _ => {
                        return Err(DirectoryReplicaStoreError::Integrity(
                            "directory replica schema version is unsupported".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn set_schema_version(
        transaction: &Transaction<'_>,
        previous_version: i64,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let changed = transaction.execute(
            "UPDATE directory_replica_meta
             SET schema_version = ?1
             WHERE singleton = 1 AND schema_version = ?2",
            params![DIRECTORY_REPLICA_SCHEMA_VERSION, previous_version],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema migration compare-and-swap failed".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn add_resolution_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::table_has_column(transaction, "active_incident_digest")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_chains
                 ADD COLUMN active_incident_digest BLOB
                 CHECK (active_incident_digest IS NULL OR length(active_incident_digest) = 32);",
            )?;
        }
        if !Self::table_has_column(transaction, "last_resolution_digest")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_chains
                 ADD COLUMN last_resolution_digest BLOB
                 CHECK (last_resolution_digest IS NULL OR length(last_resolution_digest) = 32);",
            )?;
        }
        Self::require_resolution_columns(transaction)
    }

    pub(super) fn require_resolution_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::table_has_column(transaction, "active_incident_digest")?
            || !Self::table_has_column(transaction, "last_resolution_digest")?
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v3 resolution columns are missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn add_witness_policy_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "witness_policy_epoch")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN witness_policy_epoch INTEGER NOT NULL DEFAULT 0
                 CHECK (witness_policy_epoch >= 0);",
            )?;
        }
        if !Self::metadata_has_column(transaction, "witness_policy_head")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN witness_policy_head BLOB
                 CHECK (witness_policy_head IS NULL OR length(witness_policy_head) = 32);",
            )?;
        }
        Self::require_witness_policy_metadata_columns(transaction)
    }

    pub(super) fn require_witness_policy_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "witness_policy_epoch")?
            || !Self::metadata_has_column(transaction, "witness_policy_head")?
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v7 witness policy metadata is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn require_mirror_registry_table(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let present: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'directory_replica_mirror_producers'",
            [],
            |row| row.get(0),
        )?;
        if present != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v9 mirror registry is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn add_certificate_import_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "certificate_import_sequence")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN certificate_import_sequence INTEGER NOT NULL DEFAULT 0
                 CHECK (certificate_import_sequence >= 0);",
            )?;
        }
        if !Self::metadata_has_column(transaction, "certificate_import_head")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN certificate_import_head BLOB
                 CHECK (certificate_import_head IS NULL
                     OR length(certificate_import_head) = 32);",
            )?;
        }
        Self::require_certificate_import_metadata_columns(transaction)
    }

    pub(super) fn require_certificate_import_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "certificate_import_sequence")?
            || !Self::metadata_has_column(transaction, "certificate_import_head")?
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v10 certificate import metadata is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn require_certificate_import_table(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let present: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table'
               AND name = 'directory_observation_certificate_imports'",
            [],
            |row| row.get(0),
        )?;
        if present != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v10 certificate import table is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn add_route_domain_policy_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "route_domain_policy_epoch")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN route_domain_policy_epoch INTEGER NOT NULL DEFAULT 0
                 CHECK (route_domain_policy_epoch >= 0);",
            )?;
        }
        if !Self::metadata_has_column(transaction, "route_domain_policy_head")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN route_domain_policy_head BLOB
                 CHECK (route_domain_policy_head IS NULL
                     OR length(route_domain_policy_head) = 32);",
            )?;
        }
        Self::require_route_domain_policy_metadata_columns(transaction)
    }

    pub(super) fn require_route_domain_policy_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "route_domain_policy_epoch")?
            || !Self::metadata_has_column(transaction, "route_domain_policy_head")?
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v11 route-domain policy metadata is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn require_route_domain_policy_table(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let present: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'directory_route_domain_policies'",
            [],
            |row| row.get(0),
        )?;
        if present != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v11 route-domain policy table is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn add_route_domain_attestor_policy_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "route_domain_attestor_policy_epoch")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN route_domain_attestor_policy_epoch INTEGER NOT NULL DEFAULT 0
                 CHECK (route_domain_attestor_policy_epoch >= 0);",
            )?;
        }
        if !Self::metadata_has_column(transaction, "route_domain_attestor_policy_head")? {
            transaction.execute_batch(
                "ALTER TABLE directory_replica_meta
                 ADD COLUMN route_domain_attestor_policy_head BLOB
                 CHECK (route_domain_attestor_policy_head IS NULL
                     OR length(route_domain_attestor_policy_head) = 32);",
            )?;
        }
        Self::require_route_domain_attestor_policy_metadata_columns(transaction)
    }

    pub(super) fn require_route_domain_attestor_policy_metadata_columns(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        if !Self::metadata_has_column(transaction, "route_domain_attestor_policy_epoch")?
            || !Self::metadata_has_column(transaction, "route_domain_attestor_policy_head")?
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v12 route-domain attestor metadata is missing"
                    .to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn require_route_domain_attestor_policy_table(
        transaction: &Transaction<'_>,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let present: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM sqlite_master
             WHERE type = 'table' AND name = 'directory_route_domain_attestor_policies'",
            [],
            |row| row.get(0),
        )?;
        if present != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica schema v12 route-domain attestor table is missing".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn metadata_has_column(
        transaction: &Transaction<'_>,
        expected: &str,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let mut statement = transaction.prepare("PRAGMA table_info(directory_replica_meta)")?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(columns.iter().any(|column| column == expected))
    }

    pub(super) fn table_has_column(
        transaction: &Transaction<'_>,
        expected: &str,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let mut statement = transaction.prepare("PRAGMA table_info(directory_replica_chains)")?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(columns.iter().any(|column| column == expected))
    }

    pub(super) fn validate_metadata(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let metadata = connection
            .query_row(
                "SELECT schema_version, chain_id, local_node_id,
                        witness_policy_epoch, witness_policy_head,
                        route_domain_policy_epoch, route_domain_policy_head,
                        route_domain_attestor_policy_epoch,
                        route_domain_attestor_policy_head,
                        certificate_import_sequence, certificate_import_head
                 FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, Option<Vec<u8>>>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, Option<Vec<u8>>>(8)?,
                        row.get::<_, i64>(9)?,
                        row.get::<_, Option<Vec<u8>>>(10)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "directory replica metadata row is missing".to_string(),
                )
            })?;
        if metadata.0 != DIRECTORY_REPLICA_SCHEMA_VERSION
            || bytes32(&metadata.1, "replica metadata chain id")?
                != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || bytes32(&metadata.2, "replica metadata local node id")? != *local_node_id
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica metadata does not match this node and Directory Sync V1"
                    .to_string(),
            ));
        }
        let policy_epoch =
            nonnegative_i64_to_u64(metadata.3, "replica metadata witness policy epoch")?;
        let policy_head = metadata
            .4
            .as_deref()
            .map(|value| bytes32(value, "replica metadata witness policy head"))
            .transpose()?;
        if (policy_epoch == 0) != policy_head.is_none() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica witness policy metadata is inconsistent".to_string(),
            ));
        }
        let route_domain_policy_epoch =
            nonnegative_i64_to_u64(metadata.5, "replica metadata route-domain policy epoch")?;
        let route_domain_policy_head = metadata
            .6
            .as_deref()
            .map(|value| bytes32(value, "replica metadata route-domain policy head"))
            .transpose()?;
        if (route_domain_policy_epoch == 0) != route_domain_policy_head.is_none() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica route-domain policy metadata is inconsistent".to_string(),
            ));
        }
        let route_domain_attestor_policy_epoch = nonnegative_i64_to_u64(
            metadata.7,
            "replica metadata route-domain attestor policy epoch",
        )?;
        let route_domain_attestor_policy_head = metadata
            .8
            .as_deref()
            .map(|value| bytes32(value, "replica metadata route-domain attestor policy head"))
            .transpose()?;
        if (route_domain_attestor_policy_epoch == 0) != route_domain_attestor_policy_head.is_none()
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica route-domain attestor metadata is inconsistent".to_string(),
            ));
        }
        let certificate_import_sequence =
            nonnegative_i64_to_u64(metadata.9, "replica metadata certificate import sequence")?;
        let certificate_import_head = metadata
            .10
            .as_deref()
            .map(|value| bytes32(value, "replica metadata certificate import head"))
            .transpose()?;
        if (certificate_import_sequence == 0) != certificate_import_head.is_none() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica certificate import metadata is inconsistent".to_string(),
            ));
        }
        Ok(())
    }
}
