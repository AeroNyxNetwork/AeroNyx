// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/storage_rows.rs
// ============================================
//! # Replica storage row shapes
//!
//! Owns the raw `SQLite` row structs read during audits and imports, the audit
//! block batch size, and the pending-commitment and quarantine-incident carriers.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::DirectoryDescriptorCommitmentV1;

/// One commitment row read during an audit, checked for everything except the
/// descriptor signature, which is verified later in a parallel batch.
///
/// [PARALLEL-DIRECTORY-AUDIT 2026-10-10 by Claude]
#[derive(Debug)]
pub struct PendingReplicaCommitment {
    pub(super) commitment: DirectoryDescriptorCommitmentV1,
    pub(super) object_node_id: [u8; 32],
    pub(super) object_sequence: u64,
    pub(super) descriptor_blob: Vec<u8>,
}

/// Replica blocks read and verified per batch during an audit.
pub(super) const AUDIT_REPLICA_BLOCK_BATCH: usize = 512;

#[derive(Debug)]
pub(super) struct StoredReplicaBlockRow {
    pub(super) height: i64,
    pub(super) block_hash: Vec<u8>,
    pub(super) prev_block_hash: Vec<u8>,
    pub(super) produced_at: i64,
    pub(super) commitment_count: i64,
    pub(super) block_blob: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct QuarantineIncident<'a> {
    pub(super) kind: &'a str,
    pub(super) height: u64,
    pub(super) local_hash: [u8; 32],
    pub(super) remote_hash: [u8; 32],
    pub(super) evidence_frame: &'a [u8],
}

#[derive(Debug)]
pub(super) struct StoredResolutionRow {
    pub(super) digest: Vec<u8>,
    pub(super) command_id: Vec<u8>,
    pub(super) incident_digest: Vec<u8>,
    pub(super) producer: Vec<u8>,
    pub(super) action: String,
    pub(super) expected_tip_height: i64,
    pub(super) expected_tip_hash: Vec<u8>,
    pub(super) expected_quarantine_kind: String,
    pub(super) previous_resolution_digest: Option<Vec<u8>>,
    pub(super) resolved_at: i64,
    pub(super) resolver_node_id: Vec<u8>,
    pub(super) signature: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredObservationCheckpointRow {
    pub(super) sequence: i64,
    pub(super) checkpoint_hash: Vec<u8>,
    pub(super) previous_checkpoint_hash: Vec<u8>,
    pub(super) observed_at: i64,
    pub(super) observation_root: Vec<u8>,
    pub(super) producer_count: i64,
    pub(super) checkpoint_blob: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredObservationWitnessRow {
    pub(super) checkpoint_hash: Vec<u8>,
    pub(super) checkpoint_sequence: i64,
    pub(super) observer: Vec<u8>,
    pub(super) witness_node_id: Vec<u8>,
    pub(super) witnessed_at: i64,
    pub(super) response_blob: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredObservationWitnessOutcomeRow {
    pub(super) rounds: i64,
    pub(super) attempts: i64,
    pub(super) totals: [i64; 7],
    pub(super) last_checkpoint_sequence: i64,
    pub(super) last_round_at: i64,
    pub(super) last_success_at: Option<i64>,
    pub(super) last_failure_at: Option<i64>,
    pub(super) last_round_attempts: i64,
    pub(super) last_round: [i64; 7],
    pub(super) updated_at: i64,
}

#[derive(Debug)]
pub(super) struct StoredObservationWitnessPolicyRow {
    pub(super) epoch: i64,
    pub(super) policy_digest: Vec<u8>,
    pub(super) previous_policy_digest: Vec<u8>,
    pub(super) activated_at: i64,
    pub(super) witness_threshold: i64,
    pub(super) witness_count: i64,
    pub(super) witness_node_ids: Vec<u8>,
    pub(super) signer_node_id: Vec<u8>,
    pub(super) signature: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredRouteDomainPolicyRow {
    pub(super) epoch: i64,
    pub(super) policy_digest: Vec<u8>,
    pub(super) previous_policy_digest: Vec<u8>,
    pub(super) activated_at: i64,
    pub(super) strict_required: i64,
    pub(super) assignment_count: i64,
    pub(super) assignments: Vec<u8>,
    pub(super) signer_node_id: Vec<u8>,
    pub(super) signature: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredRouteDomainAttestorPolicyRow {
    pub(super) epoch: i64,
    pub(super) policy_digest: Vec<u8>,
    pub(super) previous_policy_digest: Vec<u8>,
    pub(super) activated_at: i64,
    pub(super) strict_required: i64,
    pub(super) attestor_threshold: i64,
    pub(super) attestor_count: i64,
    pub(super) attestor_node_ids: Vec<u8>,
    pub(super) signer_node_id: Vec<u8>,
    pub(super) signature: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredObservationWitnessPolicyAnchorReceiptRow {
    pub(super) policy_epoch: i64,
    pub(super) policy_digest: Vec<u8>,
    pub(super) observer: Vec<u8>,
    pub(super) witness_node_id: Vec<u8>,
    pub(super) witnessed_at: i64,
    pub(super) response_blob: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct StoredObservationCertificateImportRow {
    pub(super) import_sequence: i64,
    pub(super) import_digest: Vec<u8>,
    pub(super) previous_import_digest: Vec<u8>,
    pub(super) certificate_id: Vec<u8>,
    pub(super) observer: Vec<u8>,
    pub(super) checkpoint_sequence: i64,
    pub(super) checkpoint_hash: Vec<u8>,
    pub(super) checkpoint_observed_at: i64,
    pub(super) certificate_sha256: Vec<u8>,
    pub(super) certificate_frame: Vec<u8>,
    pub(super) policy_digest: Vec<u8>,
    pub(super) policy_minimum_witnesses: i64,
    pub(super) policy_witness_count: i64,
    pub(super) policy_witness_node_ids: Vec<u8>,
    pub(super) verified_at: i64,
    pub(super) importer_node_id: Vec<u8>,
    pub(super) signature: Vec<u8>,
}
