// ============================================
// File: crates/aeronyx-server/src/services/directory_replica/store_reports.rs
// ============================================
//! # Replica audit, snapshot and report values
//!
//! Owns the aggregate startup-audit result, the low-cost store snapshot, the
//! observation convergence snapshot, checkpoint-append and page-import reports,
//! and the durable producer retry-state value.
//!
//! [ARCH-SPLIT 2026-10-10 by Claude] Split from `services/directory_replica.rs`; bodies unchanged.

use super::{DirectoryObservationWitnessOutcomeSnapshot, DirectoryReplicaProducerSnapshot};

/// Aggregate result of a complete replica startup audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryReplicaAudit {
    /// Number of producer namespaces.
    pub producers: u64,
    /// Producer namespaces admitted only as non-authoritative mirrors.
    pub mirror_producers: u64,
    /// Number of producer namespaces currently quarantined.
    pub quarantined_producers: u64,
    /// Number of verified remote blocks.
    pub blocks: u64,
    /// Number of commitments exactly matched to block payloads.
    pub commitments: u64,
    /// Number of durable authenticated incidents.
    pub incidents: u64,
    /// Number of node-identity-signed operator resolutions.
    pub resolutions: u64,
    /// Number of audited observer-signed convergence checkpoints.
    pub observation_checkpoints: u64,
    /// Latest audited checkpoint sequence, or zero when none exists.
    pub observation_checkpoint_sequence: u64,
    /// Latest audited checkpoint hash, or zero when none exists.
    pub observation_checkpoint_hash: [u8; 32],
    /// Latest audited checkpoint timestamp, or zero when none exists.
    pub observation_checkpoint_observed_at: u64,
    /// Number of independently signed accepted witness receipts.
    pub observation_checkpoint_witnesses: u64,
    /// Latest local checkpoint sequence with at least one accepted witness.
    pub observation_checkpoint_witnessed_sequence: u64,
    /// Distinct witnesses retained for the latest witnessed sequence.
    pub observation_checkpoint_latest_witnesses: u64,
    /// Audited privacy-safe witness attempt aggregates.
    pub observation_witness_outcomes: DirectoryObservationWitnessOutcomeSnapshot,
    /// Number of audited local witness-policy epochs.
    pub observation_witness_policy_epochs: u64,
    /// Current local witness-policy epoch, or zero before reconciliation.
    pub observation_witness_policy_epoch: u64,
    /// Timestamp bound into the current local witness policy.
    pub observation_witness_policy_activated_at: u64,
    /// Number of operator-pinned witnesses in the current policy.
    pub observation_witness_policy_members: u64,
    /// External receipt threshold in the current local policy.
    pub observation_witness_policy_threshold: u64,
    /// Signed external anchor receipts retained for local policy epochs.
    pub observation_witness_policy_anchor_receipts: u64,
    /// Opaque foreign policy heads this node retains for independent observers.
    pub observation_witness_remote_policy_anchors: u64,
    /// Number of audited local route-domain policy epochs.
    pub route_domain_policy_epochs: u64,
    /// Current local route-domain policy epoch, or zero before first use.
    pub route_domain_policy_epoch: u64,
    /// Timestamp bound into the current route-domain policy.
    pub route_domain_policy_activated_at: u64,
    /// Number of opaque node-to-domain assignments in the current policy.
    pub route_domain_policy_assignments: u64,
    /// Whether current multi-hop selection requires complete pinned coverage.
    pub route_domain_policy_strict: bool,
    /// Number of audited local route-domain attestor-policy epochs.
    pub route_domain_attestor_policy_epochs: u64,
    /// Current local route-domain attestor-policy epoch, or zero before use.
    pub route_domain_attestor_policy_epoch: u64,
    /// Timestamp bound into the current route-domain attestor policy.
    pub route_domain_attestor_policy_activated_at: u64,
    /// Number of locally pinned route-domain attestors.
    pub route_domain_attestor_policy_members: u64,
    /// Locally required distinct valid route-domain attestations.
    pub route_domain_attestor_policy_threshold: u64,
    /// Whether current multi-hop selection requires attested route domains.
    pub route_domain_attestor_policy_strict: bool,
    /// Third-party portable observation certificates in the audited import log.
    pub imported_observation_certificates: u64,
    /// Latest node-signed certificate-import sequence, or zero when empty.
    pub imported_observation_certificate_sequence: u64,
    /// Latest node-signed certificate-import digest, or zero when empty.
    pub imported_observation_certificate_head: [u8; 32],
    /// Number of audited producer-local retry rows.
    pub retry_states: u64,
}

/// Low-cost aggregate view of the already audited replica namespace.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectoryReplicaStoreSnapshot {
    /// Number of producer namespaces currently persisted.
    pub producers: u64,
    /// Producer namespaces retained only as non-authoritative mirrors.
    pub mirror_producers: u64,
    /// Number of producer namespaces blocked by durable quarantine.
    pub quarantined_producers: u64,
    /// Number of verified remote blocks retained across all producers.
    pub blocks: u64,
    /// Number of verified descriptor commitments retained across all producers.
    pub commitments: u64,
    /// Number of durable authenticated incidents.
    pub incidents: u64,
    /// Number of durable signed quarantine resolutions.
    pub resolutions: u64,
    /// Number of durable observer-signed convergence checkpoints.
    pub observation_checkpoints: u64,
    /// Latest checkpoint sequence, or zero when none exists.
    pub observation_checkpoint_sequence: u64,
    /// Latest checkpoint hash, or zero when none exists.
    pub observation_checkpoint_hash: [u8; 32],
    /// Latest checkpoint timestamp, or zero when none exists.
    pub observation_checkpoint_observed_at: u64,
    /// Number of independently signed accepted witness receipts.
    pub observation_checkpoint_witnesses: u64,
    /// Latest local checkpoint sequence with at least one accepted witness.
    pub observation_checkpoint_witnessed_sequence: u64,
    /// Distinct witnesses retained for the latest witnessed sequence.
    pub observation_checkpoint_latest_witnesses: u64,
    /// Audited privacy-safe witness attempt aggregates.
    pub observation_witness_outcomes: DirectoryObservationWitnessOutcomeSnapshot,
    /// Number of durable, signed local witness-policy epochs.
    pub observation_witness_policy_epochs: u64,
    /// Current local witness-policy epoch, or zero before reconciliation.
    pub observation_witness_policy_epoch: u64,
    /// Timestamp bound into the current local witness policy.
    pub observation_witness_policy_activated_at: u64,
    /// Number of operator-pinned witnesses in the current policy.
    pub observation_witness_policy_members: u64,
    /// External receipt threshold in the current local policy.
    pub observation_witness_policy_threshold: u64,
    /// Signed external anchor receipts retained for local policy epochs.
    pub observation_witness_policy_anchor_receipts: u64,
    /// Opaque foreign policy heads this node retains for independent observers.
    pub observation_witness_remote_policy_anchors: u64,
    /// Third-party portable observation certificates retained after audit.
    pub imported_observation_certificates: u64,
    /// Latest local certificate-import sequence, or zero when empty.
    pub imported_observation_certificate_sequence: u64,
    /// Latest local certificate-import digest, or zero when empty.
    pub imported_observation_certificate_head: [u8; 32],
    /// Per-producer accepted-prefix summaries for local operator presentation.
    pub producer_snapshots: Vec<DirectoryReplicaProducerSnapshot>,
}

/// Bounded, locally recomputable overlap across verified producer replicas.
///
/// This snapshot compares exact commitment hashes from each eligible
/// producer's most recent block window. It does not assign voting weight,
/// choose a chain, or create a globally finalized checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DirectoryReplicaObservationConvergenceSnapshot {
    /// Unique producer pins supplied by the validated node configuration.
    pub configured_producers: u64,
    /// Configured producers with a non-empty, non-quarantined accepted prefix.
    pub eligible_producers: u64,
    /// Configured producers that have not supplied an accepted block yet.
    pub pending_producers: u64,
    /// Configured producers excluded because signed evidence quarantined them.
    pub excluded_quarantined_producers: u64,
    /// Maximum number of recent blocks inspected per eligible producer.
    pub window_blocks: u64,
    /// Commitment observations across all eligible producer windows.
    pub recent_commitments: u64,
    /// Unique commitment hashes across all eligible producer windows.
    pub distinct_recent_commitments: u64,
    /// Commitments observed by at least two eligible producer chains.
    pub multi_source_recent_commitments: u64,
    /// Commitments observed by every eligible producer when at least two exist.
    pub all_eligible_source_recent_commitments: u64,
    /// Deterministic digest of eligible tips and their exact common commitments.
    pub observation_root: Option<[u8; 32]>,
}

/// Result of attempting to append one complete observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryObservationCheckpointAppendReport {
    /// Whether a new checkpoint was written. An unchanged root is idempotent.
    pub appended: bool,
    /// Latest checkpoint sequence after the transaction.
    pub sequence: u64,
    /// Latest checkpoint hash after the transaction.
    pub checkpoint_hash: [u8; 32],
    /// Timestamp bound into the latest checkpoint.
    pub observed_at: u64,
    /// Number of configured producer tips bound into the checkpoint.
    pub producer_count: u16,
    /// Recomputable multi-source overlap root.
    pub observation_root: [u8; 32],
}

/// Result of one verified, atomic bounded-page import.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaImportReport {
    /// New blocks committed by this transaction.
    pub blocks_inserted: u64,
    /// Exact existing blocks accepted idempotently.
    pub blocks_already_present: u64,
    /// New descriptor commitments committed by this transaction.
    pub commitments_inserted: u64,
    /// Newly recorded same-node/same-sequence descriptor conflicts.
    pub descriptor_equivocations: u64,
    /// Accepted producer prefix height after import.
    pub tip_height: u64,
    /// Accepted producer prefix hash after import.
    pub tip_hash: [u8; 32],
}

/// Restart-durable producer-local synchronization failure state.
///
/// The state contains bounded control-plane scheduling metadata only. It never
/// contains endpoints, response bodies, descriptors, routes, payloads, client
/// identifiers, private keys, wallet traffic, or social graph data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaRetryState {
    /// Remote producer identity used as the local scheduling key.
    pub producer: [u8; 32],
    /// Consecutive failures since the last authenticated successful page.
    pub consecutive_failures: u64,
    /// Earliest Unix timestamp at which another pull may begin.
    pub retry_not_before: Option<u64>,
    /// Timestamp of the most recent failed pull.
    pub last_failure_at: u64,
    /// Stable bounded internal failure bucket.
    pub last_failure_reason: String,
    /// Number of timer rounds skipped while durable backoff was active.
    pub backoff_skips: u64,
}
