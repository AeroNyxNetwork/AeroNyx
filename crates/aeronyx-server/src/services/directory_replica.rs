// ============================================================================
// File: crates/aeronyx-server/src/services/directory_replica.rs
// ============================================================================
//! # Directory Chain Replica Store
//!
//! ## Creation Reason
//! Directory Sync responses are producer attestations, not global consensus.
//! A node therefore needs a durable producer-scoped replica namespace instead
//! of inserting remote blocks into its own locally produced Directory Chain.
//!
//! ## Main Functionality
//! - Stores each remote producer chain under an independent `SQLite` namespace.
//! - Re-verifies signed response evidence, blocks, commitments, and descriptor
//!   objects before one atomic import transaction.
//! - Makes exact repeated pages idempotent.
//! - Persists signed fork/rollback evidence and permanently quarantines only
//!   the producer that authored conflicting chain claims.
//! - Records authenticated descriptor equivocation without blaming an honest
//!   chain producer that merely observed the conflicting public descriptors.
//! - Audits every replica chain and index before the node may synchronize.
//! - Persists bounded producer retry state across process restarts and clears
//!   it atomically with the next authenticated successful page.
//! - Computes a bounded recent-window intersection of exact descriptor
//!   commitments across non-quarantined configured producer replicas.
//! - Exposes low-cost aggregate snapshots and privacy-safe synchronization
//!   observations, including bounded retry state, without re-running a full
//!   cryptographic audit per API read.
//! - Exports bounded incident summaries and re-verified signed evidence for
//!   authenticated operator review without adding an automatic recovery path.
//! - Resolves quarantine only through a node-identity-signed, host-local,
//!   compare-and-swap command while retaining the accepted prefix and every
//!   incident and resolution as an append-only audit trail.
//! - Persists local-identity-signed, hash-linked observation checkpoints only
//!   after a complete configured producer set yields a recomputable overlap.
//! - Independently recomputes checkpoints received from pinned observers and
//!   persists only canonical, accepted, externally signed witness receipts.
//! - Persists privacy-safe aggregate witness outcomes separately from signed
//!   receipts so operators can distinguish unavailable evidence from faults.
//! - Persists every local witness pin/threshold change as a node-identity-
//!   signed, hash-linked policy epoch with a metadata-anchored durable head.
//! - Retains opaque policy heads observed by independent nodes and accepted
//!   signed external anchor receipts without exposing policy member identities.
//! - Selects only forward-moving, mature, unwitnessed checkpoints for external
//!   recomputation so asymmetric sync schedules cannot chase the moving tip.
//! - Keeps recurring witness selection history-bounded by verifying only the
//!   candidate, its predecessor, the latest receipt set, and durable outcome;
//!   startup and explicit audits still verify the complete retained history.
//! - Exports bounded producer blocks and descriptors only after a complete
//!   audit inside the same `SQLite` read transaction for signed carrier use.
//! - Retains a bounded, durable registry for permissionless full-node mirrors
//!   without granting those producers checkpoint, witness, or policy authority.
//!
//! ## Calling Relationships
//! - `server.rs` opens this store beside `DirectoryChainStore` at startup.
//! - `api/directory_replica_sync.rs` verifies and downloads bounded peer pages,
//!   then calls `import_verified_page` from a blocking worker.
//! - `api/directory_replica_status.rs` reads only low-cost audited snapshots.
//! - The local producer store remains the only source served by peer routes.
//!
//! ## Main Logical Flow
//! 1. Open the existing Directory Chain `SQLite` file and initialize only the
//!    `directory_replica_*` tables.
//! 2. Pin schema, chain id, and the local node identity in replica metadata.
//! 3. Audit every accepted producer prefix and all durable incident and
//!    operator-resolution evidence.
//! 4. Re-verify the signed range-response frame and exact descriptor objects.
//! 5. Atomically append a contiguous producer prefix, clear its retry state,
//!    or persist quarantine without mutating another producer namespace.
//! 6. Derive recent multi-source observation evidence without choosing a fork,
//!    producer, quorum, or globally finalized height.
//! 7. Sign and append a checkpoint only when every configured producer has an
//!    eligible prefix; re-derive every historical root during startup audit.
//! 8. Select only mature forward-moving checkpoints for external witnessing,
//!    recompute them from locally retained exact producer prefixes, and audit
//!    every accepted receipt again on restart.
//! 9. Audit bounded aggregate witness outcome counters without retaining peer
//!    identity, endpoint, request id, signature, or checkpoint hash metadata.
//! 10. Keep periodic selection cost independent of retained history while
//!     retaining complete fail-closed audits at startup and operator request.
//! 11. Canonicalize the configured witness set and append a policy epoch only
//!     when pins or threshold change; verify the complete policy chain before
//!     synchronization or any listener starts.
//! 12. Exchange only opaque epoch/digest policy heads with pinned witnesses;
//!     reject rollback, same-epoch conflict, and non-contiguous progression.
//! 13. Admit dynamically discovered mirrors into a capacity-bounded registry;
//!     promote them out of mirror status atomically if later operator-pinned.
//!
//! ## Privacy Invariant
//! Replica tables contain only public signed node descriptors, public
//! descriptor commitments, signed Directory Chain blocks, and signed incident
//! evidence. They must never contain client identities, IPs, routes, selected
//! hops, message ids, payloads, ciphertext, Memory Chain records, DNS contents,
//! destinations, private keys, or wallet traffic.
//!
//! ## Important Note for Next Developer
//! - Never merge remote blocks into `directory_chain_blocks`.
//! - A Full-node Mirror producer is untrusted replicated evidence. Never include
//!   mirror registry membership in observation checkpoints or authority policy.
//! - Never auto-delete, auto-rewind, or auto-select through a quarantined fork.
//! - A producer-signed block fork/rollback quarantines that producer. A signed
//!   descriptor equivocation is evidence about the descriptor owner and does
//!   not automatically quarantine the observing producer.
//! - Keep all limits synchronized with the core Directory Sync V1 contract.
//! - Observation convergence is a local digest over independently verified
//!   producer evidence. It is never consensus, voting, fork choice, or finality.
//! - Observation checkpoints preserve that exact boundary. They are one
//!   observer's signed evidence and must never be presented as global blocks.
//! - A witness receipt proves one external recomputation of one exact local
//!   checkpoint. It is not a vote, quorum, fork choice, consensus, or finality.
//! - A policy anchor receipt proves only external retention of an opaque local
//!   policy head. It neither reveals nor approves policy membership.
//! - Witness outcome telemetry is aggregate diagnostic evidence only. Never add
//!   witness identities, endpoints, request ids, signatures, or hashes to it.
//! - Witness policy epochs describe only this operator's local evidence target.
//!   They are not a validator set, vote, quorum, fork choice, consensus, or
//!   finality, and public status must never expose their full member identities.
//! - A carrier may export only non-quarantined producer evidence retained in
//!   this store. The receiver must still verify producer and carrier signatures.
//! - Incident evidence export is read-only. Quarantine resolution requires a
//!   separately authenticated, audited compare-and-swap command boundary.
//! - Never expose [`DirectoryReplicaStore::resolve_quarantine`] through the
//!   peer or public HTTP routers. It belongs only to the host-local CLI, whose
//!   caller must also possess the node identity key and database permissions.
//!
//! ## Last Modified
//! v0.18.0-FullNodeMirror - Added schema v9 bounded non-authoritative mirror registry
//! v0.17.0-DirectoryPolicyHeadAnchor - Added schema v8 opaque external policy-head anchors and signed receipts
//! v0.16.0-DirectoryWitnessPolicyEpoch - Added schema v7 signed hash-linked local witness policy history, metadata-head partial-deletion protection, startup reconciliation, and tamper tests
//! v0.15.0-DirectoryWitnessFailureDrills - Locked partial-receipt restart recovery and current-pin rotation fail-closed behavior with deterministic state-machine coverage
//! v0.14.0-DirectoryWitnessThreshold - Added configurable pinned-witness corroboration targets
//! v0.13.0-DirectoryBoundedWitnessSelectionAudit - Bounded recurring selection verification without weakening startup audit
//! v0.12.0-DirectoryMatureWitnessScheduling - Added audited mature unwitnessed checkpoint selection
//! v0.11.0-DirectoryWitnessCapabilityNegotiation - Clarified peer-unavailable witness semantics for rolling upgrades
//! v0.10.0-DirectoryWitnessOutcomeTelemetry - Added schema v6 privacy-safe durable and runtime witness outcome buckets
//! v0.9.0-DirectoryEvidenceCarrier - Added transactional audited producer evidence export and carrier-frame audit
//! v0.8.0-DirectoryObservationWitness - Added schema v5, independent checkpoint recomputation, and receipt audit
//! v0.7.0-DirectoryObservationCheckpoints - Added schema v4, append-only signed
//! checkpoints, exact-prefix root recomputation, and startup tamper detection.
//! v0.6.0-DirectoryReplicaQuarantineResolution - Added schema v3, signed local
//! operator resolution commands, exact incident/tip CAS, and linked immutable
//! resolution auditing without deleting or rewinding accepted evidence.
//! v0.5.0-DirectoryReplicaIncidentEvidence - Added bounded incident pagination
//! and fail-closed, signature-reverified evidence export for local operators.
//! v0.4.0-DirectoryReplicaObservationConvergence - Added bounded recent-window
//! multi-source commitment overlap and a deterministic local observation root.
//! v0.3.0-DirectoryReplicaDurableRetry - Added an atomic schema v1-to-v2
//! migration and audited restart-durable producer retry state.
//! v0.2.2-DirectoryReplicaRetryRuntime - Added producer-local retry boundaries
//! and backoff skip counters to process-lifetime synchronization telemetry.
//! v0.2.1-DirectoryReplicaModuleSplit - Updated transport and status ownership.
//! v0.2.0-DirectoryReplicaStatus - Added aggregate status snapshots and shared
//! synchronization observations for bounded catch-up visibility.
//! v0.1.0-DirectoryReplicaStore - Initial producer-isolated replica persistence.
// ============================================================================

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::protocol::discovery::{
    decode_directory_sync_message, directory_block_range_response_signing_bytes,
    directory_observation_witness_response_signing_bytes,
    directory_policy_anchor_request_signing_bytes, directory_policy_anchor_response_signing_bytes,
    directory_replica_block_range_response_signing_bytes, encode_directory_sync_message,
    DirectoryCommitmentBlockV1, DirectoryCommitmentValidationError,
    DirectoryDescriptorCommitmentV1, DirectoryObservationCheckpointV1, DirectoryObservationTipV1,
    DirectorySyncMessage, SignedNodeDescriptor, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1, DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
    DIRECTORY_POLICY_ANCHOR_CONFLICT_V1, DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
    DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1, MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1,
    MAX_DIRECTORY_SYNC_BLOCKS_V1,
};
use bincode::Options;
use parking_lot::Mutex;
use rusqlite::{
    params, params_from_iter, types::Value, Connection, OptionalExtension, Transaction,
    TransactionBehavior,
};
use sha2::{Digest, Sha256};

const DIRECTORY_REPLICA_SCHEMA_VERSION: i64 = 9;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V8: i64 = 8;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V7: i64 = 7;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V6: i64 = 6;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V5: i64 = 5;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V4: i64 = 4;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V3: i64 = 3;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V2: i64 = 2;
const DIRECTORY_REPLICA_SCHEMA_VERSION_V1: i64 = 1;
const MAX_DIRECTORY_BLOCK_BYTES: u64 = 64 * 1024;
const MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES: u64 = 32 * 1024;
const MAX_DIRECTORY_SYNC_EVIDENCE_BYTES: usize = 512 * 1024;
const DIRECTORY_REPLICA_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const RESPONSE_TIMESTAMP_SKEW_SECS: u64 = 60;
const MAX_DIRECTORY_REPLICA_FAILURE_REASON_BYTES: usize = 96;
const DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS: u64 = 32;
const MAX_DIRECTORY_REPLICA_CONVERGENCE_PRODUCERS: usize = 16;
const MAX_DIRECTORY_REPLICA_INCIDENT_KIND_BYTES: usize = 64;
const MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES: u64 = 4 * 1024;
const MAX_DIRECTORY_OBSERVATION_WITNESS_BYTES: usize = 2 * 1024;
const MAX_DIRECTORY_POLICY_ANCHOR_BYTES: usize = 2 * 1024;
const MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS: usize = 16;
const DIRECTORY_REPLICA_RESOLUTION_ACTION: &str = "resume_existing_prefix";
const DIRECTORY_REPLICA_RESOLUTION_TIMESTAMP_SKEW_SECS: u64 = 60;
/// Maximum incident summaries returned by one operator API read.
pub(crate) const MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE: usize = 50;
/// Maximum producer failure streak retained in memory and audited `SQLite`.
pub(crate) const DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES: u64 = 64;
/// Maximum durable retry delay accepted by the replica store and scheduler.
pub(crate) const DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS: u64 = 30 * 60;
/// Hard implementation ceiling for durable permissionless mirror namespaces.
pub(crate) const MAX_DIRECTORY_FULL_NODE_MIRROR_PRODUCERS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectoryReplicaImportMode {
    PinnedAuthority,
    FullNodeMirror {
        descriptor_sequence: u64,
        max_producers: usize,
    },
}

/// Failures returned by the producer-isolated replica store.
#[derive(Debug, thiserror::Error)]
pub enum DirectoryReplicaStoreError {
    /// Filesystem setup failed.
    #[error("directory replica filesystem error: {0}")]
    Io(#[from] std::io::Error),
    /// `SQLite` rejected a schema, query, or transaction operation.
    #[error("directory replica sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A protocol object could not be encoded or decoded safely.
    #[error("directory replica codec error: {0}")]
    Codec(String),
    /// A descriptor object did not reproduce its signed commitment.
    #[error("directory replica descriptor error: {0}")]
    Descriptor(String),
    /// A block failed the canonical Directory Chain V1 contract.
    #[error("directory replica block validation error: {0}")]
    Block(#[from] DirectoryCommitmentValidationError),
    /// Durable metadata, chain, index, or evidence is inconsistent.
    #[error("directory replica integrity error: {0}")]
    Integrity(String),
    /// A bounded import request violates the V1 transport contract.
    #[error("directory replica request error: {0}")]
    Request(String),
    /// The producer is durably isolated pending operator review.
    #[error("directory producer is quarantined: {0}")]
    Quarantined(String),
    /// The configured durable permissionless mirror namespace ceiling is full.
    #[error("directory full-node mirror capacity reached")]
    MirrorCapacity,
}

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

/// Audited mature checkpoint that has not reached its configured corroboration
/// target among the current operator-pinned witnesses.
///
/// The retained witness identities are public node signing keys required only
/// to avoid duplicate outbound requests. They must never be exposed by public
/// status or interpreted as voting weight, consensus membership, or finality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryObservationWitnessTarget {
    /// Canonical observer-signed checkpoint requiring more external evidence.
    pub checkpoint: DirectoryObservationCheckpointV1,
    /// Current pinned witnesses with an audited accepted receipt for this row.
    pub witnessed_by: Vec<[u8; 32]>,
    /// Required number of distinct pinned witness receipts.
    pub minimum_witnesses: usize,
}

/// Result of independently evaluating an external observation checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessDecision {
    /// Every exact producer prefix exists locally and the root recomputes.
    Accepted,
    /// At least one exact referenced producer prefix is not retained locally.
    EvidenceUnavailable,
    /// Retained producer evidence conflicts or recomputes a different root.
    EvidenceConflict,
}

/// Stable privacy-safe result bucket for one outbound witness attempt.
///
/// The enum deliberately excludes peer identity, endpoint, request id,
/// signature, checkpoint hash, transport text, and response body data. New
/// variants require a schema migration and additive status-contract review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectoryObservationWitnessOutcome {
    /// A canonical accepted receipt was verified and durably retained.
    Accepted,
    /// The witness does not yet retain every exact referenced producer prefix.
    EvidenceUnavailable,
    /// Locally retained evidence conflicts with the observed checkpoint.
    EvidenceConflict,
    /// The witness is not admitted, reachable, or serving the optional route.
    PeerUnavailable,
    /// The bounded outbound request failed before a verifiable frame arrived.
    TransportFailure,
    /// A received frame failed canonical contract or signature verification.
    VerificationFailure,
    /// A verified accepted receipt could not be durably retained.
    PersistenceFailure,
}

/// Aggregate counters for a bounded set of witness attempts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessOutcomeCounters {
    /// Canonical accepted receipts durably retained.
    pub accepted: u64,
    /// Witnesses missing at least one exact producer prefix.
    pub evidence_unavailable: u64,
    /// Witnesses whose retained evidence conflicts with the checkpoint.
    pub evidence_conflict: u64,
    /// Witnesses unavailable at admission, endpoint, or capability validation.
    pub peer_unavailable: u64,
    /// Bounded outbound transport failures.
    pub transport_failures: u64,
    /// Canonical contract or signature verification failures.
    pub verification_failures: u64,
    /// Verified receipts rejected by durable persistence.
    pub persistence_failures: u64,
}

impl DirectoryObservationWitnessOutcomeCounters {
    fn from_outcomes(outcomes: &[DirectoryObservationWitnessOutcome]) -> Self {
        let mut counters = Self::default();
        for outcome in outcomes {
            counters.record(*outcome);
        }
        counters
    }

    fn record(&mut self, outcome: DirectoryObservationWitnessOutcome) {
        let counter = match outcome {
            DirectoryObservationWitnessOutcome::Accepted => &mut self.accepted,
            DirectoryObservationWitnessOutcome::EvidenceUnavailable => {
                &mut self.evidence_unavailable
            }
            DirectoryObservationWitnessOutcome::EvidenceConflict => &mut self.evidence_conflict,
            DirectoryObservationWitnessOutcome::PeerUnavailable => &mut self.peer_unavailable,
            DirectoryObservationWitnessOutcome::TransportFailure => &mut self.transport_failures,
            DirectoryObservationWitnessOutcome::VerificationFailure => {
                &mut self.verification_failures
            }
            DirectoryObservationWitnessOutcome::PersistenceFailure => {
                &mut self.persistence_failures
            }
        };
        *counter = counter.saturating_add(1);
    }

    fn checked_add(self, other: Self) -> Result<Self, DirectoryReplicaStoreError> {
        let add = |left: u64, right: u64| {
            left.checked_add(right).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness outcome counter exhausted".to_string(),
                )
            })
        };
        Ok(Self {
            accepted: add(self.accepted, other.accepted)?,
            evidence_unavailable: add(self.evidence_unavailable, other.evidence_unavailable)?,
            evidence_conflict: add(self.evidence_conflict, other.evidence_conflict)?,
            peer_unavailable: add(self.peer_unavailable, other.peer_unavailable)?,
            transport_failures: add(self.transport_failures, other.transport_failures)?,
            verification_failures: add(self.verification_failures, other.verification_failures)?,
            persistence_failures: add(self.persistence_failures, other.persistence_failures)?,
        })
    }

    const fn saturating_add(self, other: Self) -> Self {
        Self {
            accepted: self.accepted.saturating_add(other.accepted),
            evidence_unavailable: self
                .evidence_unavailable
                .saturating_add(other.evidence_unavailable),
            evidence_conflict: self
                .evidence_conflict
                .saturating_add(other.evidence_conflict),
            peer_unavailable: self.peer_unavailable.saturating_add(other.peer_unavailable),
            transport_failures: self
                .transport_failures
                .saturating_add(other.transport_failures),
            verification_failures: self
                .verification_failures
                .saturating_add(other.verification_failures),
            persistence_failures: self
                .persistence_failures
                .saturating_add(other.persistence_failures),
        }
    }

    /// Total attempts represented by these mutually exclusive buckets.
    #[must_use]
    pub const fn attempts(self) -> u64 {
        self.accepted
            .saturating_add(self.evidence_unavailable)
            .saturating_add(self.evidence_conflict)
            .saturating_add(self.peer_unavailable)
            .saturating_add(self.transport_failures)
            .saturating_add(self.verification_failures)
            .saturating_add(self.persistence_failures)
    }

    /// Non-accepted attempts represented by these buckets.
    #[must_use]
    pub const fn failures(self) -> u64 {
        self.attempts().saturating_sub(self.accepted)
    }
}

/// Audited aggregate witness telemetry retained across restarts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryObservationWitnessOutcomeSnapshot {
    /// Completed bounded witness rounds.
    pub rounds: u64,
    /// Cumulative mutually exclusive attempt outcomes.
    pub totals: DirectoryObservationWitnessOutcomeCounters,
    /// Latest local checkpoint sequence evaluated by a witness round.
    pub last_checkpoint_sequence: u64,
    /// Timestamp of the latest completed witness round.
    pub last_round_at: Option<u64>,
    /// Latest round containing at least one accepted receipt.
    pub last_success_at: Option<u64>,
    /// Latest round containing at least one non-accepted attempt.
    pub last_failure_at: Option<u64>,
    /// Mutually exclusive outcomes from only the latest completed round.
    pub last_round: DirectoryObservationWitnessOutcomeCounters,
    /// Process-only failures while persisting this telemetry itself.
    /// Durable snapshots always keep this field at zero.
    pub telemetry_persistence_failures: u64,
}

impl DirectoryObservationWitnessOutcomeSnapshot {
    fn next_durable_round(
        self,
        checkpoint_sequence: u64,
        observed_at: u64,
        round: DirectoryObservationWitnessOutcomeCounters,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        if checkpoint_sequence < self.last_checkpoint_sequence
            || self
                .last_round_at
                .is_some_and(|last_round_at| observed_at < last_round_at)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome round regressed".to_string(),
            ));
        }
        Ok(Self {
            rounds: self.rounds.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness outcome round counter exhausted".to_string(),
                )
            })?,
            totals: self.totals.checked_add(round)?,
            last_checkpoint_sequence: checkpoint_sequence,
            last_round_at: Some(observed_at),
            last_success_at: if round.accepted > 0 {
                Some(observed_at)
            } else {
                self.last_success_at
            },
            last_failure_at: if round.failures() > 0 {
                Some(observed_at)
            } else {
                self.last_failure_at
            },
            last_round: round,
            telemetry_persistence_failures: 0,
        })
    }
}

/// Persisted aggregate state for one producer namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaProducerSnapshot {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Timestamp signed into the accepted tip block.
    pub tip_timestamp: u64,
    /// Whether imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable authenticated incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Last time this namespace metadata changed locally.
    pub updated_at: u64,
    /// Verified blocks retained for this producer.
    pub blocks: u64,
    /// Verified commitments retained for this producer.
    pub commitments: u64,
    /// Durable incidents attributed to this producer response stream.
    pub incidents: u64,
    /// Signed operator resolutions retained for this producer.
    pub resolutions: u64,
}

/// Bounded metadata for one startup-audited Directory Replica incident.
///
/// The summary intentionally excludes the potentially large signed response
/// frame. Call [`DirectoryReplicaStore::incident_evidence`] for an independent,
/// fail-closed verification immediately before exporting that frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentSummary {
    /// Content-addressed incident identifier used as the pagination cursor.
    pub incident_digest: [u8; 32],
    /// Producer that signed the conflicting Directory Sync response.
    pub producer: [u8; 32],
    /// Identity whose chain or descriptor assertion conflicts.
    pub subject_node_id: [u8; 32],
    /// Stable internal incident classification.
    pub kind: String,
    /// Conflicting block or advertised tip height.
    pub height: u64,
    /// Previously accepted local claim.
    pub local_hash: [u8; 32],
    /// Conflicting producer-signed remote claim.
    pub remote_hash: [u8; 32],
    /// Local Unix timestamp at which the signed evidence was persisted.
    pub observed_at: u64,
    /// Whether this producer remains quarantined at read time.
    pub producer_quarantined: bool,
}

/// Deterministic cursor page of incident metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentPage {
    /// Incident summaries ordered by ascending content digest.
    pub incidents: Vec<DirectoryReplicaIncidentSummary>,
    /// Last returned digest when another page exists.
    pub next_cursor: Option<[u8; 32]>,
}

/// Complete independently verifiable evidence for one durable incident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaIncidentEvidence {
    /// Validated incident metadata and current quarantine state.
    pub summary: DirectoryReplicaIncidentSummary,
    /// Exact canonical producer-signed `BlockRangeResponseV1` bytes.
    pub evidence_frame: Vec<u8>,
    /// SHA-256 digest of `evidence_frame` for transport/file verification.
    pub evidence_sha256: [u8; 32],
}

/// Current accepted prefix and isolation state for one producer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaTip {
    /// Remote producer identity.
    pub producer: [u8; 32],
    /// Accepted contiguous prefix height.
    pub tip_height: u64,
    /// Accepted tip hash, or zero for an empty prefix.
    pub tip_hash: [u8; 32],
    /// Accepted tip timestamp, or zero for an empty prefix.
    pub tip_timestamp: u64,
    /// Whether further imports are blocked pending operator review.
    pub quarantined: bool,
    /// Stable incident kind when quarantined.
    pub quarantine_kind: Option<String>,
    /// Exact unresolved incident when quarantined.
    pub active_incident_digest: Option<[u8; 32]>,
    /// Latest signed resolution in this producer's linked audit history.
    pub last_resolution_digest: Option<[u8; 32]>,
}

/// One bounded page exported from a fully audited producer replica.
///
/// The carrier signs transport metadata separately. Every block in this page
/// remains signed by the original producer and is re-verified by the receiver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaEvidencePage {
    /// Contiguous producer-signed blocks in ascending height order.
    pub blocks: Vec<DirectoryCommitmentBlockV1>,
    /// Audited accepted producer tip height at export time.
    pub tip_height: u64,
    /// Audited accepted producer tip hash at export time.
    pub tip_hash: [u8; 32],
}

impl DirectoryReplicaTip {
    const fn empty(producer: [u8; 32]) -> Self {
        Self {
            producer,
            tip_height: 0,
            tip_hash: [0u8; 32],
            tip_timestamp: 0,
            quarantined: false,
            quarantine_kind: None,
            active_incident_digest: None,
            last_resolution_digest: None,
        }
    }
}

/// Node-identity-signed command that resumes one exact quarantined prefix.
///
/// The command cannot select a fork, delete evidence, or rewind a chain. Its
/// compare-and-swap fields bind one immutable incident to the exact prefix and
/// previous resolution history inspected by the host-local operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaResolutionCommand {
    /// Random operator command identifier; unique across this replica store.
    pub command_id: [u8; 16],
    /// Immutable incident explicitly approved by the operator.
    pub incident_digest: [u8; 32],
    /// Producer namespace whose existing prefix may resume synchronization.
    pub producer: [u8; 32],
    /// Accepted prefix height observed before signing.
    pub expected_tip_height: u64,
    /// Accepted prefix hash observed before signing.
    pub expected_tip_hash: [u8; 32],
    /// Quarantine classification observed before signing.
    pub expected_quarantine_kind: String,
    /// Previous linked resolution, or `None` for this producer's first one.
    pub previous_resolution_digest: Option<[u8; 32]>,
    /// Host timestamp at which the operator approved the command.
    pub resolved_at: u64,
    /// Local node identity that must match replica metadata.
    pub resolver_node_id: [u8; 32],
    /// Ed25519 signature over every command field and the fixed action.
    pub signature: [u8; 64],
}

impl DirectoryReplicaResolutionCommand {
    /// Constructs and signs one exact `resume_existing_prefix` command.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when any bounded command field is
    /// invalid. Signing never reads or modifies the replica database.
    #[allow(clippy::too_many_arguments)]
    pub fn sign(
        identity: &IdentityKeyPair,
        command_id: [u8; 16],
        incident_digest: [u8; 32],
        producer: [u8; 32],
        expected_tip_height: u64,
        expected_tip_hash: [u8; 32],
        expected_quarantine_kind: String,
        previous_resolution_digest: Option<[u8; 32]>,
        resolved_at: u64,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut command = Self {
            command_id,
            incident_digest,
            producer,
            expected_tip_height,
            expected_tip_hash,
            expected_quarantine_kind,
            previous_resolution_digest,
            resolved_at,
            resolver_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        command.validate_unsigned_fields()?;
        command.signature = identity.sign(&command.signing_bytes());
        Ok(command)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.command_id == [0u8; 16]
            || self.incident_digest == [0u8; 32]
            || self.producer == [0u8; 32]
            || self.resolver_node_id == [0u8; 32]
            || self.resolved_at == 0
            || (self.expected_tip_height == 0 && self.expected_tip_hash != [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "directory replica resolution command contains an invalid sentinel".to_string(),
            ));
        }
        validate_incident_kind(&self.expected_quarantine_kind)
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(320);
        bytes.extend_from_slice(b"AeroNyx-DirectoryReplicaResolution-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.command_id);
        bytes.extend_from_slice(&self.incident_digest);
        bytes.extend_from_slice(&self.producer);
        bytes.extend_from_slice(&self.expected_tip_height.to_le_bytes());
        bytes.extend_from_slice(&self.expected_tip_hash);
        bytes.extend_from_slice(&(self.expected_quarantine_kind.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self.expected_quarantine_kind.as_bytes());
        match self.previous_resolution_digest {
            Some(digest) => {
                bytes.push(1);
                bytes.extend_from_slice(&digest);
            }
            None => bytes.push(0),
        }
        bytes.extend_from_slice(&self.resolved_at.to_le_bytes());
        bytes.extend_from_slice(&self.resolver_node_id);
        bytes.extend_from_slice(&(DIRECTORY_REPLICA_RESOLUTION_ACTION.len() as u64).to_le_bytes());
        bytes.extend_from_slice(DIRECTORY_REPLICA_RESOLUTION_ACTION.as_bytes());
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Durable result of one successful compare-and-swap resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryReplicaResolutionReport {
    /// Content address of the signed resolution audit record.
    pub resolution_digest: [u8; 32],
    /// Unique command identifier supplied by the operator CLI.
    pub command_id: [u8; 16],
    /// Producer namespace that resumed its already accepted prefix.
    pub producer: [u8; 32],
    /// Prefix height retained without rewind or fork selection.
    pub retained_tip_height: u64,
    /// Prefix hash retained without modification.
    pub retained_tip_hash: [u8; 32],
    /// Signed operator approval timestamp.
    pub resolved_at: u64,
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

/// Runtime-only synchronization observation for one pinned producer.
///
/// These fields intentionally contain no endpoint, full response, descriptor,
/// route, payload, client, or wallet metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectoryReplicaSyncObservation {
    /// Pinned producer identity used only for internal status correlation.
    pub producer: [u8; 32],
    /// Most recent bounded pull attempt.
    pub last_attempt_at: Option<u64>,
    /// Most recent authenticated successful page.
    pub last_success_at: Option<u64>,
    /// Most recent rejected or failed page.
    pub last_failure_at: Option<u64>,
    /// Stable privacy-safe reason code from the most recent failure.
    pub last_failure_reason: Option<String>,
    /// Earliest Unix timestamp at which this producer may be attempted again.
    pub retry_not_before: Option<u64>,
    /// Signed remote tip height most recently observed.
    pub remote_tip_height: Option<u64>,
    /// Accepted local replica height after the most recent success.
    pub local_tip_height: u64,
    /// Whether the most recent signed response indicated additional pages.
    pub has_more: bool,
    /// Consecutive failed attempts since the last successful page.
    pub consecutive_failures: u64,
    /// Total bounded attempts during this process lifetime.
    pub total_attempts: u64,
    /// Total authenticated pages accepted during this process lifetime.
    pub successful_pages: u64,
    /// Total failed attempts during this process lifetime.
    pub failed_attempts: u64,
    /// Total scheduled rounds skipped while this producer was in backoff.
    pub backoff_skips: u64,
    /// Total new blocks committed during this process lifetime.
    pub blocks_inserted: u64,
    /// Total new commitments committed during this process lifetime.
    pub commitments_inserted: u64,
    /// Total HTTP requests consumed by authenticated successful pages.
    pub requests_sent: u64,
}

/// Aggregate process-lifetime Full-node Mirror scheduling telemetry.
///
/// This intentionally omits identities, endpoints, descriptor hashes, routes,
/// and response details. Mirror observations are diagnostic transport evidence,
/// never authority, consensus, fork choice, voting, or finality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirectoryFullNodeMirrorRuntimeSnapshot {
    /// Completed bounded mirror-selection rounds.
    pub rounds: u64,
    /// Valid public candidates considered by the latest round.
    pub last_round_candidates: u64,
    /// Capacity-bounded candidates selected by the latest round.
    pub last_round_selected: u64,
    /// Authenticated pages accepted in the latest round.
    pub last_round_succeeded: u64,
    /// Selected candidates that failed or were rejected in the latest round.
    pub last_round_failed: u64,
    /// Authenticated mirror pages accepted during this process lifetime.
    pub pages_succeeded: u64,
    /// Failed bounded mirror attempts during this process lifetime.
    pub attempts_failed: u64,
    /// Timestamp of the latest completed mirror round.
    pub last_round_at: Option<u64>,
    /// Timestamp of the latest authenticated mirror import.
    pub last_success_at: Option<u64>,
    /// Timestamp of the latest failed mirror attempt.
    pub last_failure_at: Option<u64>,
}

impl DirectoryReplicaSyncObservation {
    fn new(producer: [u8; 32]) -> Self {
        Self {
            producer,
            last_attempt_at: None,
            last_success_at: None,
            last_failure_at: None,
            last_failure_reason: None,
            retry_not_before: None,
            remote_tip_height: None,
            local_tip_height: 0,
            has_more: false,
            consecutive_failures: 0,
            total_attempts: 0,
            successful_pages: 0,
            failed_attempts: 0,
            backoff_skips: 0,
            blocks_inserted: 0,
            commitments_inserted: 0,
            requests_sent: 0,
        }
    }
}

/// Shared process-lifetime synchronization and witness telemetry.
#[derive(Debug, Default)]
pub struct DirectoryReplicaSyncRuntime {
    observations: Mutex<HashMap<[u8; 32], DirectoryReplicaSyncObservation>>,
    observation_witness: Mutex<DirectoryObservationWitnessOutcomeSnapshot>,
    full_node_mirror: Mutex<DirectoryFullNodeMirrorRuntimeSnapshot>,
}

impl DirectoryReplicaSyncRuntime {
    /// Registers configured pins so status reports can distinguish pending from
    /// disabled before the first low-frequency synchronization round.
    pub fn register_producers(&self, producers: &[[u8; 32]]) {
        let mut observations = self.observations.lock();
        for producer in producers {
            if *producer != [0u8; 32] {
                observations
                    .entry(*producer)
                    .or_insert_with(|| DirectoryReplicaSyncObservation::new(*producer));
            }
        }
    }

    /// Restores audited retry boundaries before the coordinator starts.
    ///
    /// Process-lifetime attempt/page counters remain zero after restart; only
    /// the active failure streak, retry boundary, reason, and skip count are
    /// restored because those fields control request pressure.
    pub fn restore_retry_states(&self, states: &[DirectoryReplicaRetryState]) {
        let mut observations = self.observations.lock();
        for state in states {
            let observation = observations
                .entry(state.producer)
                .or_insert_with(|| DirectoryReplicaSyncObservation::new(state.producer));
            observation.last_attempt_at = Some(state.last_failure_at);
            observation.last_failure_at = Some(state.last_failure_at);
            observation.last_failure_reason = Some(state.last_failure_reason.clone());
            observation.retry_not_before = state.retry_not_before;
            observation.consecutive_failures = state.consecutive_failures;
            observation.backoff_skips = state.backoff_skips;
        }
        drop(observations);
    }

    /// Records the beginning of one bounded page request.
    pub fn record_attempt(&self, producer: [u8; 32], attempted_at: u64) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(attempted_at);
        observation.total_attempts = observation.total_attempts.saturating_add(1);
    }

    /// Records one authenticated page after its atomic import completes.
    #[allow(clippy::too_many_arguments)]
    pub fn record_success(
        &self,
        producer: [u8; 32],
        succeeded_at: u64,
        local_tip_height: u64,
        remote_tip_height: u64,
        has_more: bool,
        blocks_inserted: u64,
        commitments_inserted: u64,
        requests_sent: u32,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(succeeded_at);
        observation.last_success_at = Some(succeeded_at);
        observation.remote_tip_height = Some(remote_tip_height);
        observation.local_tip_height = local_tip_height;
        observation.has_more = has_more;
        observation.consecutive_failures = 0;
        observation.retry_not_before = None;
        observation.successful_pages = observation.successful_pages.saturating_add(1);
        observation.blocks_inserted = observation.blocks_inserted.saturating_add(blocks_inserted);
        observation.commitments_inserted = observation
            .commitments_inserted
            .saturating_add(commitments_inserted);
        observation.requests_sent = observation
            .requests_sent
            .saturating_add(u64::from(requests_sent));
    }

    /// Records one stable failure code without retaining peer endpoints,
    /// response bodies, or underlying transport error strings.
    pub fn record_failure(
        &self,
        producer: [u8; 32],
        failed_at: u64,
        reason: &str,
        retry_not_before: Option<u64>,
    ) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.last_attempt_at = Some(failed_at);
        observation.last_failure_at = Some(failed_at);
        observation.last_failure_reason = Some(reason.chars().take(96).collect());
        observation.retry_not_before = retry_not_before.map(|value| value.max(failed_at));
        observation.consecutive_failures = observation
            .consecutive_failures
            .saturating_add(1)
            .min(DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES);
        observation.failed_attempts = observation.failed_attempts.saturating_add(1);
        drop(observations);
    }

    /// Returns the future retry boundary for one producer, if backoff is active.
    #[must_use]
    pub fn deferred_retry_until(&self, producer: &[u8; 32], now: u64) -> Option<u64> {
        self.observations
            .lock()
            .get(producer)
            .and_then(|observation| observation.retry_not_before)
            .filter(|retry_at| *retry_at > now)
    }

    /// Returns the current consecutive failure count for backoff calculation.
    #[must_use]
    pub fn consecutive_failures(&self, producer: &[u8; 32]) -> u64 {
        self.observations
            .lock()
            .get(producer)
            .map_or(0, |observation| observation.consecutive_failures)
    }

    /// Records one timer tick intentionally skipped by producer-local backoff.
    pub fn record_backoff_skip(&self, producer: [u8; 32]) {
        let mut observations = self.observations.lock();
        let observation = observations
            .entry(producer)
            .or_insert_with(|| DirectoryReplicaSyncObservation::new(producer));
        observation.backoff_skips = observation.backoff_skips.saturating_add(1);
        drop(observations);
    }

    /// Records one bounded outbound witness round using mutually exclusive,
    /// privacy-safe outcome buckets.
    ///
    /// `telemetry_durable` describes only the aggregate telemetry write. An
    /// accepted attempt already means its signed receipt was persisted.
    pub fn record_observation_witness_round(
        &self,
        checkpoint_sequence: u64,
        observed_at: u64,
        outcomes: &[DirectoryObservationWitnessOutcome],
        telemetry_durable: bool,
    ) {
        if checkpoint_sequence == 0 || observed_at == 0 || outcomes.is_empty() {
            return;
        }
        let round = DirectoryObservationWitnessOutcomeCounters::from_outcomes(outcomes);
        let mut snapshot = self.observation_witness.lock();
        snapshot.rounds = snapshot.rounds.saturating_add(1);
        snapshot.totals = snapshot.totals.saturating_add(round);
        snapshot.last_checkpoint_sequence = checkpoint_sequence;
        snapshot.last_round_at = Some(observed_at);
        if round.accepted > 0 {
            snapshot.last_success_at = Some(observed_at);
        }
        if round.failures() > 0 {
            snapshot.last_failure_at = Some(observed_at);
        }
        snapshot.last_round = round;
        if !telemetry_durable {
            snapshot.telemetry_persistence_failures =
                snapshot.telemetry_persistence_failures.saturating_add(1);
        }
    }

    /// Returns process-lifetime aggregate witness telemetry.
    #[must_use]
    pub fn observation_witness_snapshot(&self) -> DirectoryObservationWitnessOutcomeSnapshot {
        *self.observation_witness.lock()
    }

    /// Records one aggregate bounded non-authoritative mirror round.
    pub fn record_full_node_mirror_round(
        &self,
        candidates: usize,
        selected: usize,
        succeeded: usize,
        completed_at: u64,
    ) {
        if completed_at == 0 || succeeded > selected {
            return;
        }
        let failed = selected.saturating_sub(succeeded);
        let mut snapshot = self.full_node_mirror.lock();
        snapshot.rounds = snapshot.rounds.saturating_add(1);
        snapshot.last_round_candidates = u64::try_from(candidates).unwrap_or(u64::MAX);
        snapshot.last_round_selected = u64::try_from(selected).unwrap_or(u64::MAX);
        snapshot.last_round_succeeded = u64::try_from(succeeded).unwrap_or(u64::MAX);
        snapshot.last_round_failed = u64::try_from(failed).unwrap_or(u64::MAX);
        snapshot.pages_succeeded = snapshot
            .pages_succeeded
            .saturating_add(snapshot.last_round_succeeded);
        snapshot.attempts_failed = snapshot
            .attempts_failed
            .saturating_add(snapshot.last_round_failed);
        snapshot.last_round_at = Some(completed_at);
        if succeeded > 0 {
            snapshot.last_success_at = Some(completed_at);
        }
        if failed > 0 {
            snapshot.last_failure_at = Some(completed_at);
        }
    }

    /// Returns aggregate permissionless mirror telemetry without peer metadata.
    #[must_use]
    pub fn full_node_mirror_snapshot(&self) -> DirectoryFullNodeMirrorRuntimeSnapshot {
        *self.full_node_mirror.lock()
    }

    /// Returns producer observations in deterministic identity order.
    #[must_use]
    pub fn snapshot(&self) -> Vec<DirectoryReplicaSyncObservation> {
        let mut observations = self
            .observations
            .lock()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        observations.sort_by_key(|observation| observation.producer);
        observations
    }
}

#[derive(Debug)]
struct StoredReplicaBlockRow {
    height: i64,
    block_hash: Vec<u8>,
    prev_block_hash: Vec<u8>,
    produced_at: i64,
    commitment_count: i64,
    block_blob: Vec<u8>,
}

#[derive(Debug)]
struct QuarantineIncident<'a> {
    kind: &'a str,
    height: u64,
    local_hash: [u8; 32],
    remote_hash: [u8; 32],
    evidence_frame: &'a [u8],
}

#[derive(Debug)]
struct StoredResolutionRow {
    digest: Vec<u8>,
    command_id: Vec<u8>,
    incident_digest: Vec<u8>,
    producer: Vec<u8>,
    action: String,
    expected_tip_height: i64,
    expected_tip_hash: Vec<u8>,
    expected_quarantine_kind: String,
    previous_resolution_digest: Option<Vec<u8>>,
    resolved_at: i64,
    resolver_node_id: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationCheckpointRow {
    sequence: i64,
    checkpoint_hash: Vec<u8>,
    previous_checkpoint_hash: Vec<u8>,
    observed_at: i64,
    observation_root: Vec<u8>,
    producer_count: i64,
    checkpoint_blob: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationWitnessRow {
    checkpoint_hash: Vec<u8>,
    checkpoint_sequence: i64,
    observer: Vec<u8>,
    witness_node_id: Vec<u8>,
    witnessed_at: i64,
    response_blob: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationWitnessOutcomeRow {
    rounds: i64,
    attempts: i64,
    totals: [i64; 7],
    last_checkpoint_sequence: i64,
    last_round_at: i64,
    last_success_at: Option<i64>,
    last_failure_at: Option<i64>,
    last_round_attempts: i64,
    last_round: [i64; 7],
    updated_at: i64,
}

#[derive(Debug)]
struct StoredObservationWitnessPolicyRow {
    epoch: i64,
    policy_digest: Vec<u8>,
    previous_policy_digest: Vec<u8>,
    activated_at: i64,
    witness_threshold: i64,
    witness_count: i64,
    witness_node_ids: Vec<u8>,
    signer_node_id: Vec<u8>,
    signature: Vec<u8>,
}

#[derive(Debug)]
struct StoredObservationWitnessPolicyAnchorReceiptRow {
    policy_epoch: i64,
    policy_digest: Vec<u8>,
    observer: Vec<u8>,
    witness_node_id: Vec<u8>,
    witnessed_at: i64,
    response_blob: Vec<u8>,
}

/// One node-identity-signed, hash-linked local witness admission policy.
///
/// This is operator configuration history, not a network vote, validator set,
/// fork-choice rule, consensus object, or finality certificate. Full member
/// identities remain in the host-local database and are never returned by the
/// public aggregate status endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectoryObservationWitnessPolicyEpoch {
    epoch: u64,
    previous_policy_digest: [u8; 32],
    activated_at: u64,
    witness_node_ids: Vec<[u8; 32]>,
    minimum_witnesses: usize,
    signer_node_id: [u8; 32],
    signature: [u8; 64],
}

impl DirectoryObservationWitnessPolicyEpoch {
    fn sign(
        identity: &IdentityKeyPair,
        epoch: u64,
        previous_policy_digest: [u8; 32],
        activated_at: u64,
        witness_node_ids: Vec<[u8; 32]>,
        minimum_witnesses: usize,
    ) -> Result<Self, DirectoryReplicaStoreError> {
        let mut policy = Self {
            epoch,
            previous_policy_digest,
            activated_at,
            witness_node_ids,
            minimum_witnesses,
            signer_node_id: identity.public_key_bytes(),
            signature: [0u8; 64],
        };
        policy.validate_unsigned_fields()?;
        policy.signature = identity.sign(&policy.signing_bytes());
        Ok(policy)
    }

    fn validate_unsigned_fields(&self) -> Result<(), DirectoryReplicaStoreError> {
        if self.epoch == 0 || self.activated_at == 0 || self.signer_node_id == [0u8; 32] {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy contains an invalid sentinel".to_string(),
            ));
        }
        validate_observation_witness_policy_members(&self.witness_node_ids, self.minimum_witnesses)
    }

    fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(192 + self.witness_node_ids.len() * 32);
        bytes.extend_from_slice(b"AeroNyx-DirectoryObservationWitnessPolicy-v1");
        bytes.extend_from_slice(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
        bytes.extend_from_slice(&self.epoch.to_le_bytes());
        bytes.extend_from_slice(&self.previous_policy_digest);
        bytes.extend_from_slice(&self.activated_at.to_le_bytes());
        bytes.extend_from_slice(&(self.minimum_witnesses as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.witness_node_ids.len() as u64).to_le_bytes());
        for witness_node_id in &self.witness_node_ids {
            bytes.extend_from_slice(witness_node_id);
        }
        bytes.extend_from_slice(&self.signer_node_id);
        bytes
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.signing_bytes());
        hasher.update(self.signature);
        hasher.finalize().into()
    }
}

/// Result of reconciling validated runtime pins into the signed local history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectoryObservationWitnessPolicyReconcileReport {
    /// True only when pins or threshold created a new durable epoch.
    pub(crate) appended: bool,
    /// Current local policy epoch after reconciliation.
    pub(crate) epoch: u64,
    /// Content digest of the current signed policy.
    pub(crate) policy_digest: [u8; 32],
    /// Timestamp bound into the current policy.
    pub(crate) activated_at: u64,
    /// Number of canonical current witness pins.
    pub(crate) witness_members: u64,
    /// Required distinct external receipts.
    pub(crate) minimum_witnesses: u64,
}

/// Privacy-bounded current local policy head exported to pinned witnesses.
///
/// The digest commits to the complete node-signed local policy, while member
/// identities remain host-local and never enter the anchor protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DirectoryObservationWitnessPolicyAnchor {
    pub(crate) epoch: u64,
    pub(crate) previous_policy_digest: [u8; 32],
    pub(crate) policy_digest: [u8; 32],
}

/// Result of evaluating one authenticated foreign policy-head anchor request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectoryObservationWitnessPolicyAnchorDecision {
    /// Exact head was already retained or appended durably.
    Accepted,
    /// Request regressed below the latest retained observer epoch.
    Rollback,
    /// The same epoch was previously retained with another digest.
    Conflict,
    /// A forward request did not link to the immediately retained head.
    HistoryGap,
}

impl DirectoryObservationWitnessPolicyAnchorDecision {
    #[must_use]
    pub(crate) const fn outcome(self) -> u8 {
        match self {
            Self::Accepted => DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            Self::Rollback => DIRECTORY_POLICY_ANCHOR_ROLLBACK_V1,
            Self::Conflict => DIRECTORY_POLICY_ANCHOR_CONFLICT_V1,
            Self::HistoryGap => DIRECTORY_POLICY_ANCHOR_HISTORY_GAP_V1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifiedObservationWitness {
    sequence: u64,
    checkpoint_hash: [u8; 32],
    observer: [u8; 32],
    response_timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ObservationCheckpointTip {
    sequence: u64,
    checkpoint_hash: [u8; 32],
    observed_at: u64,
    producer_count: u16,
    observation_root: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ObservationWitnessAudit {
    witnesses: u64,
    latest_sequence: u64,
    latest_witnesses: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ObservationWitnessPolicyAudit {
    epochs: u64,
    current: Option<DirectoryObservationWitnessPolicyEpoch>,
    current_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct VerifiedObservationWitnessSet {
    sequence: u64,
    witness_node_ids: Vec<[u8; 32]>,
}

#[derive(Debug, Default)]
struct AuditedResolutionIndex {
    commands: HashMap<[u8; 32], DirectoryReplicaResolutionCommand>,
    by_producer: HashMap<[u8; 32], HashSet<[u8; 32]>>,
    resolved_incidents: HashMap<[u8; 32], HashSet<[u8; 32]>>,
}

/// Durable producer-scoped replica namespace.
pub struct DirectoryReplicaStore {
    connection: Mutex<Connection>,
    path: PathBuf,
    local_node_id: [u8; 32],
}

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

    /// Reconciles validated runtime witness pins into a signed local policy
    /// epoch without rewriting historical policy or witness receipts.
    ///
    /// Reordered pins are canonicalized and therefore idempotent. A pin-set or
    /// threshold change appends exactly one node-identity-signed, hash-linked
    /// epoch and advances the metadata head in the same immediate transaction.
    /// This history describes only this node operator's corroboration policy;
    /// it does not define network membership, voting weight, consensus, fork
    /// choice, or finality.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for invalid pins/thresholds,
    /// identity mismatch, malformed prior history, counter exhaustion, or an
    /// atomic SQLite compare-and-swap failure.
    pub(crate) fn reconcile_observation_witness_policy(
        &self,
        identity: &IdentityKeyPair,
        witness_node_ids: &[[u8; 32]],
        minimum_witnesses: usize,
        activated_at: u64,
    ) -> Result<DirectoryObservationWitnessPolicyReconcileReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id || activated_at == 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy identity or timestamp is invalid".to_string(),
            ));
        }
        let canonical_witnesses =
            canonical_observation_witness_policy_members(witness_node_ids, minimum_witnesses)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        let previous = Self::audit_observation_witness_policies(&transaction, &self.local_node_id)?;
        if let Some(current) = previous.current.as_ref() {
            if current.witness_node_ids == canonical_witnesses
                && current.minimum_witnesses == minimum_witnesses
            {
                transaction.commit()?;
                return Ok(DirectoryObservationWitnessPolicyReconcileReport {
                    appended: false,
                    epoch: current.epoch,
                    policy_digest: previous.current_digest,
                    activated_at: current.activated_at,
                    witness_members: u64::try_from(current.witness_node_ids.len()).map_err(
                        |_| {
                            DirectoryReplicaStoreError::Integrity(
                                "observation witness policy member count exceeds u64".to_string(),
                            )
                        },
                    )?,
                    minimum_witnesses: u64::try_from(current.minimum_witnesses).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "observation witness policy threshold exceeds u64".to_string(),
                        )
                    })?,
                });
            }
        }

        let epoch = previous.epochs.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness policy epoch exhausted".to_string(),
            )
        })?;
        let previous_policy_digest = previous.current_digest;
        let activated_at = previous
            .current
            .as_ref()
            .map_or(activated_at, |policy| activated_at.max(policy.activated_at));
        let policy = DirectoryObservationWitnessPolicyEpoch::sign(
            identity,
            epoch,
            previous_policy_digest,
            activated_at,
            canonical_witnesses,
            minimum_witnesses,
        )?;
        let policy_digest = policy.digest();
        let witness_node_ids = policy
            .witness_node_ids
            .iter()
            .flat_map(|node_id| node_id.iter().copied())
            .collect::<Vec<_>>();
        transaction.execute(
            "INSERT INTO directory_observation_witness_policies
                (epoch, policy_digest, previous_policy_digest, activated_at,
                 witness_threshold, witness_count, witness_node_ids,
                 signer_node_id, signature)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                u64_to_i64(policy.epoch, "observation witness policy epoch")?,
                policy_digest.as_slice(),
                policy.previous_policy_digest.as_slice(),
                u64_to_i64(
                    policy.activated_at,
                    "observation witness policy activation timestamp"
                )?,
                u64_to_i64(
                    u64::try_from(policy.minimum_witnesses).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "observation witness policy threshold exceeds u64".to_string(),
                        )
                    })?,
                    "observation witness policy threshold"
                )?,
                u64_to_i64(
                    u64::try_from(policy.witness_node_ids.len()).map_err(|_| {
                        DirectoryReplicaStoreError::Integrity(
                            "observation witness policy member count exceeds u64".to_string(),
                        )
                    })?,
                    "observation witness policy member count"
                )?,
                witness_node_ids,
                policy.signer_node_id.as_slice(),
                policy.signature.as_slice(),
            ],
        )?;
        let previous_head = previous
            .current
            .as_ref()
            .map(|_| previous.current_digest.to_vec());
        let changed = transaction.execute(
            "UPDATE directory_replica_meta
             SET witness_policy_epoch = ?1, witness_policy_head = ?2
             WHERE singleton = 1 AND witness_policy_epoch = ?3
               AND ((?4 IS NULL AND witness_policy_head IS NULL)
                    OR witness_policy_head = ?4)",
            params![
                u64_to_i64(epoch, "observation witness policy epoch")?,
                policy_digest.as_slice(),
                u64_to_i64(previous.epochs, "previous observation witness policy epoch")?,
                previous_head,
            ],
        )?;
        if changed != 1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy head compare-and-swap failed".to_string(),
            ));
        }
        let audited = Self::audit_observation_witness_policies(&transaction, &self.local_node_id)?;
        if audited.epochs != epoch || audited.current_digest != policy_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy post-append audit diverged".to_string(),
            ));
        }
        transaction.commit()?;
        Ok(DirectoryObservationWitnessPolicyReconcileReport {
            appended: true,
            epoch,
            policy_digest,
            activated_at: policy.activated_at,
            witness_members: u64::try_from(policy.witness_node_ids.len()).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy member count exceeds u64".to_string(),
                )
            })?,
            minimum_witnesses: u64::try_from(policy.minimum_witnesses).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy threshold exceeds u64".to_string(),
                )
            })?,
        })
    }

    /// Verifies that the metadata-anchored signed policy head contains the
    /// exact canonical runtime pin set and threshold.
    ///
    /// This comparison returns only a boolean to callers. It never widens the
    /// public status boundary to include policy member identities or digests.
    pub(crate) fn observation_witness_policy_matches(
        &self,
        witness_node_ids: &[[u8; 32]],
        minimum_witnesses: usize,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let canonical_witnesses =
            canonical_observation_witness_policy_members(witness_node_ids, minimum_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let current =
            Self::load_current_observation_witness_policy(&connection, &self.local_node_id)?;
        Ok(current.current.is_some_and(|policy| {
            policy.witness_node_ids == canonical_witnesses
                && policy.minimum_witnesses == minimum_witnesses
        }))
    }

    /// Returns the current opaque policy head after verifying its local chain.
    ///
    /// Policy member identities remain in the local signed policy row and are
    /// deliberately not included in this export object.
    pub(crate) fn current_observation_witness_policy_anchor(
        &self,
    ) -> Result<Option<DirectoryObservationWitnessPolicyAnchor>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let current =
            Self::load_current_observation_witness_policy(&connection, &self.local_node_id)?;
        Ok(current
            .current
            .map(|policy| DirectoryObservationWitnessPolicyAnchor {
                epoch: policy.epoch,
                previous_policy_digest: policy.previous_policy_digest,
                policy_digest: current.current_digest,
            }))
    }

    /// Evaluates and durably retains one authenticated foreign policy head.
    ///
    /// The first head for an observer is a signed trust-on-first-observation
    /// anchor. Later heads must be exact retries or the immediately linked next
    /// epoch. Rollback, same-epoch conflict, and gaps never mutate persistence.
    /// Member identities are never transmitted or stored by this operation.
    pub(crate) fn persist_remote_observation_witness_policy_anchor(
        &self,
        request: &DirectorySyncMessage,
        observed_at: u64,
    ) -> Result<DirectoryObservationWitnessPolicyAnchorDecision, DirectoryReplicaStoreError> {
        let request_blob = encode_directory_sync_message(request)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if request_blob.len() > MAX_DIRECTORY_POLICY_ANCHOR_BYTES {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor request exceeds size bound".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
            chain_id,
            request_id,
            requester,
            request_timestamp,
            policy_epoch,
            previous_policy_digest,
            policy_digest,
            signature,
        } = request
        else {
            return Err(DirectoryReplicaStoreError::Request(
                "unexpected observation policy anchor request".to_string(),
            ));
        };
        let position_valid = (*policy_epoch == 1 && *previous_policy_digest == [0u8; 32])
            || (*policy_epoch > 1 && *previous_policy_digest != [0u8; 32]);
        if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || *requester == [0u8; 32]
            || *requester == self.local_node_id
            || *policy_digest == [0u8; 32]
            || !position_valid
            || *request_timestamp == 0
            || observed_at.abs_diff(*request_timestamp) > RESPONSE_TIMESTAMP_SKEW_SECS
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor request contract mismatch".to_string(),
            ));
        }
        let signing_bytes = directory_policy_anchor_request_signing_bytes(
            chain_id,
            request_id,
            requester,
            *request_timestamp,
            *policy_epoch,
            previous_policy_digest,
            policy_digest,
        );
        IdentityPublicKey::from_bytes(requester)
            .and_then(|key| key.verify(&signing_bytes, signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Request(
                    "observation policy anchor request signature is invalid".to_string(),
                )
            })?;

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let latest = transaction
            .query_row(
                "SELECT policy_epoch, policy_digest
                 FROM directory_observation_remote_policy_anchors
                 WHERE observer = ?1 ORDER BY policy_epoch DESC LIMIT 1",
                params![requester.as_slice()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?
            .map(|(epoch, digest)| {
                Ok::<_, DirectoryReplicaStoreError>((
                    positive_i64_to_u64(epoch, "remote policy anchor epoch")?,
                    bytes32(&digest, "remote policy anchor digest")?,
                ))
            })
            .transpose()?;
        if let Some((latest_epoch, latest_digest)) = latest {
            if *policy_epoch < latest_epoch {
                return Ok(DirectoryObservationWitnessPolicyAnchorDecision::Rollback);
            }
            if *policy_epoch == latest_epoch {
                return Ok(if *policy_digest == latest_digest {
                    DirectoryObservationWitnessPolicyAnchorDecision::Accepted
                } else {
                    DirectoryObservationWitnessPolicyAnchorDecision::Conflict
                });
            }
            if *policy_epoch != latest_epoch.saturating_add(1)
                || *previous_policy_digest != latest_digest
            {
                return Ok(DirectoryObservationWitnessPolicyAnchorDecision::HistoryGap);
            }
        }
        transaction.execute(
            "INSERT INTO directory_observation_remote_policy_anchors
                (observer, policy_epoch, previous_policy_digest, policy_digest,
                 request_timestamp, request_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                requester.as_slice(),
                u64_to_i64(*policy_epoch, "remote policy anchor epoch")?,
                previous_policy_digest.as_slice(),
                policy_digest.as_slice(),
                u64_to_i64(*request_timestamp, "remote policy anchor timestamp")?,
                request_blob,
            ],
        )?;
        transaction.commit()?;
        Ok(DirectoryObservationWitnessPolicyAnchorDecision::Accepted)
    }

    /// Persists one accepted external receipt for an exact local policy head.
    /// Exact retries are idempotent; a witness cannot replace its receipt for
    /// the same epoch with another digest.
    pub(crate) fn persist_observation_witness_policy_anchor_receipt(
        &self,
        response: &DirectorySyncMessage,
        observed_at: u64,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let response_blob = encode_directory_sync_message(response)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if response_blob.len() > MAX_DIRECTORY_POLICY_ANCHOR_BYTES {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor response exceeds size bound".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id,
            request_id,
            observer,
            policy_epoch,
            policy_digest,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Request(
                "unexpected observation policy anchor response".to_string(),
            ));
        };
        if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || *observer != self.local_node_id
            || *responder == [0u8; 32]
            || *responder == self.local_node_id
            || *policy_epoch == 0
            || *policy_digest == [0u8; 32]
            || *outcome != DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1
            || *response_timestamp == 0
            || observed_at.abs_diff(*response_timestamp) > RESPONSE_TIMESTAMP_SKEW_SECS
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor response contract mismatch".to_string(),
            ));
        }
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            chain_id,
            request_id,
            observer,
            *policy_epoch,
            policy_digest,
            responder,
            *response_timestamp,
            *outcome,
        );
        IdentityPublicKey::from_bytes(responder)
            .and_then(|key| key.verify(&signing_bytes, signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Request(
                    "observation policy anchor response signature is invalid".to_string(),
                )
            })?;

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let current =
            Self::load_current_observation_witness_policy(&transaction, &self.local_node_id)?;
        let Some(current_policy) = current.current else {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor receipt references an unknown policy".to_string(),
            ));
        };
        if current_policy.epoch != *policy_epoch
            || current.current_digest != *policy_digest
            || !current_policy.witness_node_ids.contains(responder)
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation policy anchor receipt is outside the current local policy".to_string(),
            ));
        }
        let existing: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT policy_digest FROM directory_observation_policy_anchor_receipts
                 WHERE policy_epoch = ?1 AND witness_node_id = ?2",
                params![
                    u64_to_i64(*policy_epoch, "policy anchor receipt epoch")?,
                    responder.as_slice()
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(existing) = existing {
            if bytes32(&existing, "existing policy anchor receipt digest")? != *policy_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "policy anchor witness signed conflicting digests at one epoch".to_string(),
                ));
            }
            return Ok(false);
        }
        transaction.execute(
            "INSERT INTO directory_observation_policy_anchor_receipts
                (policy_epoch, policy_digest, observer, witness_node_id,
                 witnessed_at, response_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                u64_to_i64(*policy_epoch, "policy anchor receipt epoch")?,
                policy_digest.as_slice(),
                observer.as_slice(),
                responder.as_slice(),
                u64_to_i64(*response_timestamp, "policy anchor receipt timestamp")?,
                response_blob,
            ],
        )?;
        transaction.commit()?;
        Ok(true)
    }

    /// Counts verified current-pin receipts for one exact local policy head.
    pub(crate) fn verified_observation_witness_policy_anchor_count_for_pins(
        &self,
        policy_epoch: u64,
        policy_digest: &[u8; 32],
        eligible_witnesses: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        if policy_epoch == 0 || *policy_digest == [0u8; 32] || eligible_witnesses.is_empty() {
            return Ok(0);
        }
        let eligible = Self::validate_observation_witness_eligibility(eligible_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let receipts = Self::audit_current_observation_policy_anchor_receipts(
            &connection,
            &self.local_node_id,
            policy_epoch,
            policy_digest,
            observed_at,
        )?;
        u64::try_from(
            receipts
                .into_iter()
                .filter(|(epoch, digest, witness)| {
                    *epoch == policy_epoch && digest == policy_digest && eligible.contains(witness)
                })
                .count(),
        )
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt count exceeds u64".to_string(),
            )
        })
    }

    /// Returns current pins with verified receipts for one exact policy head.
    pub(crate) fn verified_observation_witness_policy_anchor_witnesses_for_pins(
        &self,
        policy_epoch: u64,
        policy_digest: &[u8; 32],
        eligible_witnesses: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
        if policy_epoch == 0 || *policy_digest == [0u8; 32] || eligible_witnesses.is_empty() {
            return Ok(Vec::new());
        }
        let eligible = Self::validate_observation_witness_eligibility(eligible_witnesses)?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let mut witnesses = Self::audit_current_observation_policy_anchor_receipts(
            &connection,
            &self.local_node_id,
            policy_epoch,
            policy_digest,
            observed_at,
        )?
        .into_iter()
        .filter_map(|(epoch, digest, witness)| {
            (epoch == policy_epoch && digest == *policy_digest && eligible.contains(&witness))
                .then_some(witness)
        })
        .collect::<Vec<_>>();
        witnesses.sort_unstable();
        Ok(witnesses)
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

    /// Returns durable non-authoritative mirror producer ids in stable order.
    ///
    /// Identities are for internal scheduling only and must not be exposed by
    /// public status. Registry membership grants no checkpoint or witness role.
    pub(crate) fn mirror_producer_ids(
        &self,
    ) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
        let rows = {
            let connection = self.connection.lock();
            Self::validate_metadata(&connection, &self.local_node_id)?;
            let mut statement = connection.prepare(
                "SELECT producer FROM directory_replica_mirror_producers
                 ORDER BY last_selected_at DESC, producer ASC",
            )?;
            let rows = statement
                .query_map([], |row| row.get::<_, Vec<u8>>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            drop(statement);
            drop(connection);
            rows
        };
        rows.into_iter()
            .map(|value| bytes32(&value, "directory mirror producer"))
            .collect()
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
            promoted = promoted.saturating_add(u64::try_from(transaction.execute(
                "DELETE FROM directory_replica_mirror_producers WHERE producer = ?1",
                params![producer.as_slice()],
            )?)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "directory mirror promotion count exceeds u64".to_string(),
                )
            })?);
        }
        transaction.commit()?;
        drop(connection);
        Ok(promoted)
    }

    /// Audits the complete replica namespace, then exports one bounded producer
    /// page from the same read transaction.
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
        Self::audit_connection(&transaction, &self.local_node_id, observed_at)?;
        let tip = Self::load_tip(&transaction, producer)?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }
        if from_height > tip.tip_height.saturating_add(1) {
            return Err(DirectoryReplicaStoreError::Request(
                "replica evidence range starts beyond the audited tip".to_string(),
            ));
        }
        let mut statement = transaction.prepare(
            "SELECT block_blob FROM directory_replica_blocks
             WHERE producer = ?1 AND height >= ?2
             ORDER BY height ASC LIMIT ?3",
        )?;
        let blobs = statement
            .query_map(
                params![
                    producer.as_slice(),
                    u64_to_i64(from_height, "replica evidence from height")?,
                    i64::from(limit)
                ],
                |row| row.get::<_, Vec<u8>>(0),
            )?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        let blocks = blobs
            .iter()
            .map(|blob| decode_block(blob))
            .collect::<Result<Vec<_>, _>>()?;
        transaction.commit()?;
        Ok(DirectoryReplicaEvidencePage {
            blocks,
            tip_height: tip.tip_height,
            tip_hash: tip.tip_hash,
        })
    }

    /// Audits the complete replica namespace, then loads exact producer-bound
    /// descriptor objects from the same read transaction and in request order.
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
        Self::audit_connection(&transaction, &self.local_node_id, observed_at)?;
        let tip = Self::load_tip(&transaction, producer)?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }
        let mut statement = transaction.prepare(
            "SELECT descriptor_blob FROM directory_replica_descriptor_objects
             WHERE producer = ?1 AND descriptor_hash = ?2",
        )?;
        let mut objects = Vec::with_capacity(descriptor_hashes.len());
        for descriptor_hash in descriptor_hashes {
            let blob = statement
                .query_row(
                    params![producer.as_slice(), descriptor_hash.as_slice()],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .optional()?;
            let Some(blob) = blob else {
                drop(statement);
                transaction.commit()?;
                return Ok(None);
            };
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

    fn populate_observation_status_snapshot(
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

    fn verify_resolution_command(
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

    fn validate_resolution_cas(
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

    fn resolution_incident_observed_at(
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

    fn persist_resolution(
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

    /// Signs and appends one complete configured-producer observation.
    ///
    /// The transaction refuses partial, empty, or quarantined producer sets.
    /// If the exact overlap root is already the checkpoint tip, the operation
    /// is idempotent and returns the existing tip without another write.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the configured producer set
    /// is invalid or incomplete, the signing identity differs from replica
    /// metadata, the local clock regresses, recomputation fails, or `SQLite`
    /// cannot atomically append the checkpoint.
    pub fn append_observation_checkpoint(
        &self,
        configured_producers: &[[u8; 32]],
        identity: &IdentityKeyPair,
        observed_at: u64,
    ) -> Result<DirectoryObservationCheckpointAppendReport, DirectoryReplicaStoreError> {
        if identity.public_key_bytes() != self.local_node_id || observed_at == 0 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint identity or timestamp is invalid".to_string(),
            ));
        }
        let configured = self.validate_convergence_producers(configured_producers)?;
        if configured.len() < 2 || configured.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint requires two to sixteen configured producers".to_string(),
            ));
        }

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        Self::validate_metadata(&transaction, &self.local_node_id)?;
        let mut convergence = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: u64::try_from(configured.len()).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint producer count exceeds u64".to_string(),
                )
            })?,
            window_blocks: DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS,
            ..DirectoryReplicaObservationConvergenceSnapshot::default()
        };
        let eligible_tips =
            Self::eligible_convergence_tips(&transaction, &configured, &mut convergence)?;
        if eligible_tips.len() != configured.len() {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint requires every configured producer to be eligible"
                    .to_string(),
            ));
        }
        let occurrences =
            Self::recent_commitment_occurrences(&transaction, &eligible_tips, &mut convergence)?;
        Self::complete_observation_convergence(&mut convergence, &eligible_tips, &occurrences)?;
        let observation_root = convergence.observation_root.ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint overlap root is unavailable".to_string(),
            )
        })?;
        let previous = Self::load_observation_checkpoint_tip(&transaction)?;
        if previous.sequence > 0 && previous.observation_root == observation_root {
            transaction.commit()?;
            drop(connection);
            return Ok(Self::observation_checkpoint_report(false, previous));
        }
        let report = Self::insert_observation_checkpoint(
            &transaction,
            previous,
            &eligible_tips,
            observation_root,
            identity,
            observed_at,
        )?;
        transaction.commit()?;
        drop(connection);
        Ok(report)
    }

    /// Returns the latest checkpoint after re-auditing the complete local
    /// checkpoint chain and every referenced producer prefix.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when metadata, any historical
    /// checkpoint, its signature/link, or a retained producer prefix is invalid.
    pub fn latest_audited_observation_checkpoint(
        &self,
        observed_at: u64,
    ) -> Result<Option<DirectoryObservationCheckpointV1>, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let (count, tip) =
            Self::audit_observation_checkpoints(&connection, &self.local_node_id, observed_at)?;
        if count == 0 {
            return Ok(None);
        }
        let checkpoint_blob: Vec<u8> = connection.query_row(
            "SELECT checkpoint_blob FROM directory_observation_checkpoints
             WHERE sequence = ?1 AND checkpoint_hash = ?2",
            params![
                u64_to_i64(tip.sequence, "observation checkpoint sequence")?,
                tip.checkpoint_hash.as_slice()
            ],
            |row| row.get(0),
        )?;
        drop(connection);
        Ok(Some(decode_observation_checkpoint(&checkpoint_blob)?))
    }

    /// Returns the newest mature checkpoint that still lacks a witness receipt.
    ///
    /// Selection is forward-only. The lower sequence bound is the newest
    /// authenticated witness receipt or durable outcome round, whichever is
    /// greater. This prevents a restart or already-witnessed head from causing
    /// the coordinator to work backwards through historical gaps.
    ///
    /// The complete retained history is audited at startup and by [`Self::audit`].
    /// Recurring selection keeps work independent of history length: it verifies
    /// the signed candidate and predecessor rows, recomputes the candidate root,
    /// verifies the bounded latest receipt set, and validates the durable outcome
    /// checkpoint. Every row that can move or satisfy the selection floor is
    /// therefore authenticated before the indexed candidate query is trusted.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the maturity cutoff is zero
    /// or in the future, metadata or bounded selection evidence fails audit, the
    /// selected row is malformed, or `SQLite` cannot complete the indexed query.
    pub fn latest_audited_mature_unwitnessed_observation_checkpoint(
        &self,
        matured_before: u64,
        observed_at: u64,
    ) -> Result<Option<DirectoryObservationCheckpointV1>, DirectoryReplicaStoreError> {
        self.audited_mature_observation_checkpoint_below_witness_threshold_internal(
            matured_before,
            observed_at,
            1,
            None,
        )
        .map(|target| target.map(|target| target.checkpoint))
    }

    /// Returns the next forward mature checkpoint below the configured
    /// independent receipt target among current operator-pinned witnesses.
    ///
    /// Existing receipts from removed pins remain fully audited historical
    /// evidence but do not satisfy the current operational threshold. The
    /// returned witness identities let the coordinator skip duplicate requests.
    /// This is corroboration evidence only, never consensus or finality.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the witness set is empty,
    /// duplicated, larger than the protocol bound, the threshold is outside the
    /// witness set, or any bounded selection evidence fails verification.
    pub fn next_audited_mature_observation_checkpoint_below_witness_threshold(
        &self,
        matured_before: u64,
        observed_at: u64,
        minimum_witnesses: usize,
        eligible_witnesses: &[[u8; 32]],
    ) -> Result<Option<DirectoryObservationWitnessTarget>, DirectoryReplicaStoreError> {
        self.audited_mature_observation_checkpoint_below_witness_threshold_internal(
            matured_before,
            observed_at,
            minimum_witnesses,
            Some(eligible_witnesses),
        )
    }

    fn audited_mature_observation_checkpoint_below_witness_threshold_internal(
        &self,
        matured_before: u64,
        observed_at: u64,
        minimum_witnesses: usize,
        eligible_witnesses: Option<&[[u8; 32]]>,
    ) -> Result<Option<DirectoryObservationWitnessTarget>, DirectoryReplicaStoreError> {
        if matured_before == 0 || matured_before > observed_at {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness maturity cutoff is invalid".to_string(),
            ));
        }
        if minimum_witnesses == 0 || minimum_witnesses > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness threshold is outside the protocol bound".to_string(),
            ));
        }
        let eligible_witness_set = eligible_witnesses
            .map(|witnesses| {
                if minimum_witnesses > witnesses.len() {
                    return Err(DirectoryReplicaStoreError::Request(
                        "observation witness threshold exceeds its eligibility set".to_string(),
                    ));
                }
                Self::validate_observation_witness_eligibility(witnesses)
            })
            .transpose()?;
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let checkpoint_tip = Self::load_observation_checkpoint_tip(&connection)?;
        if checkpoint_tip.sequence == 0 {
            return Ok(None);
        }
        let latest_witness_set = Self::latest_verified_observation_witness_set(
            &connection,
            &self.local_node_id,
            observed_at,
        )?;
        let outcome_snapshot = Self::load_observation_witness_outcome_snapshot(&connection)?;
        if outcome_snapshot.last_checkpoint_sequence > checkpoint_tip.sequence {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome references an unknown checkpoint".to_string(),
            ));
        }
        if outcome_snapshot.last_checkpoint_sequence > 0 {
            Self::verify_observation_checkpoint_at_sequence(
                &connection,
                &self.local_node_id,
                observed_at,
                outcome_snapshot.last_checkpoint_sequence,
            )?;
        }
        let minimum_sequence = latest_witness_set
            .sequence
            .max(outcome_snapshot.last_checkpoint_sequence)
            .max(1);
        let mut query_parameters = vec![
            Value::Integer(u64_to_i64(
                matured_before,
                "observation witness maturity cutoff",
            )?),
            Value::Integer(u64_to_i64(
                minimum_sequence,
                "observation witness sequence floor",
            )?),
        ];
        let threshold_mode = eligible_witnesses.is_some();
        let receipt_scope = if let Some(eligible_witnesses) = eligible_witnesses {
            let placeholders = std::iter::repeat_n("?", eligible_witnesses.len())
                .collect::<Vec<_>>()
                .join(", ");
            query_parameters.extend(
                eligible_witnesses
                    .iter()
                    .map(|witness| Value::Blob(witness.to_vec())),
            );
            format!(" AND witness.witness_node_id IN ({placeholders})")
        } else {
            String::new()
        };
        query_parameters.push(Value::Integer(i64::try_from(minimum_witnesses).map_err(
            |_| {
                DirectoryReplicaStoreError::Request(
                    "observation witness threshold exceeds i64".to_string(),
                )
            },
        )?));
        // Threshold collection finishes the current forward-floor checkpoint
        // before advancing. The compatibility wrapper keeps the historical
        // newest-unwitnessed behavior when no eligible pin set is supplied.
        let candidate_order = if threshold_mode { "ASC" } else { "DESC" };
        let checkpoint_query = format!(
            "SELECT checkpoint.sequence
             FROM directory_observation_checkpoints AS checkpoint
             WHERE checkpoint.observed_at <= ?
               AND checkpoint.sequence >= ?
               AND (
                   SELECT COUNT(*)
                   FROM directory_observation_checkpoint_witnesses AS witness
                   WHERE witness.checkpoint_sequence = checkpoint.sequence{receipt_scope}
               ) < ?
             ORDER BY checkpoint.sequence {candidate_order}
             LIMIT 1"
        );
        let checkpoint_sequence = connection
            .query_row(
                &checkpoint_query,
                params_from_iter(query_parameters.iter()),
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        let checkpoint = checkpoint_sequence
            .map(|sequence| {
                let sequence =
                    positive_i64_to_u64(sequence, "observation witness candidate sequence")?;
                Self::verify_observation_checkpoint_at_sequence(
                    &connection,
                    &self.local_node_id,
                    observed_at,
                    sequence,
                )
            })
            .transpose()?;
        let target = checkpoint.map(|checkpoint| {
            let witnessed_by = if checkpoint.sequence == latest_witness_set.sequence {
                latest_witness_set
                    .witness_node_ids
                    .iter()
                    .copied()
                    .filter(|witness| {
                        eligible_witness_set
                            .as_ref()
                            .is_none_or(|eligible| eligible.contains(witness))
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            DirectoryObservationWitnessTarget {
                checkpoint,
                witnessed_by,
                minimum_witnesses,
            }
        });
        drop(connection);
        if target
            .as_ref()
            .is_some_and(|target| target.checkpoint.observed_at > matured_before)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness candidate violates maturity cutoff".to_string(),
            ));
        }
        if target
            .as_ref()
            .is_some_and(|target| target.witnessed_by.len() >= minimum_witnesses)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness candidate already satisfies its threshold".to_string(),
            ));
        }
        Ok(target)
    }

    /// Returns the number of current operator-pinned witnesses whose canonical
    /// accepted receipts cover the specified latest checkpoint.
    ///
    /// Historical receipts from removed pins remain durable but are excluded.
    /// The complete latest receipt set and its checkpoint are cryptographically
    /// re-verified before the aggregate count is returned.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when pins are malformed or
    /// duplicated, latest retained receipt evidence fails verification, or
    /// `SQLite` cannot complete the bounded read.
    pub fn verified_observation_witness_count_for_pins(
        &self,
        checkpoint_sequence: u64,
        eligible_witnesses: &[[u8; 32]],
        observed_at: u64,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        if checkpoint_sequence == 0 || eligible_witnesses.is_empty() {
            return Ok(0);
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
        if latest.sequence != checkpoint_sequence {
            return Ok(0);
        }
        u64::try_from(
            latest
                .witness_node_ids
                .iter()
                .filter(|witness| eligible_witnesses.contains(*witness))
                .count(),
        )
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "current pinned observation witness count exceeds u64".to_string(),
            )
        })
    }

    /// Independently evaluates an external observer checkpoint against this
    /// node's own producer-isolated replicas.
    ///
    /// Signature validity alone is never acceptance. Every exact producer tip
    /// must exist locally and the overlap root must recompute identically.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] for malformed signatures, wrong
    /// chain/time, self-witness attempts, or local store integrity failures.
    pub fn evaluate_observation_checkpoint_witness(
        &self,
        checkpoint: &DirectoryObservationCheckpointV1,
        observed_at: u64,
    ) -> Result<DirectoryObservationWitnessDecision, DirectoryReplicaStoreError> {
        checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if checkpoint.observer == self.local_node_id {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint cannot witness itself".to_string(),
            ));
        }
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        for tip in &checkpoint.producer_tips {
            match Self::observation_block_hash_at(
                &connection,
                &self.local_node_id,
                &tip.producer,
                tip.tip_height,
            )? {
                None => return Ok(DirectoryObservationWitnessDecision::EvidenceUnavailable),
                Some(local_hash) if local_hash != tip.tip_hash => {
                    return Ok(DirectoryObservationWitnessDecision::EvidenceConflict)
                }
                Some(_) => {}
            }
        }
        let recomputed = Self::recompute_observation_checkpoint_root(
            &connection,
            checkpoint,
            &self.local_node_id,
        )?;
        drop(connection);
        if recomputed != checkpoint.observation_root {
            return Ok(DirectoryObservationWitnessDecision::EvidenceConflict);
        }
        Ok(DirectoryObservationWitnessDecision::Accepted)
    }

    /// Persists one accepted, canonical external witness receipt for a local
    /// checkpoint. Exact retries are idempotent.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the response is malformed,
    /// noncanonical, not accepted, self-signed, does not bind a retained local
    /// checkpoint, conflicts at the same witness/sequence, or has an invalid
    /// timestamp or signature.
    #[allow(clippy::too_many_lines)]
    pub fn persist_observation_checkpoint_witness(
        &self,
        response: &DirectorySyncMessage,
        observed_at: u64,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let response_blob = encode_directory_sync_message(response)
            .map_err(|error| DirectoryReplicaStoreError::Request(error.to_string()))?;
        if response_blob.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_BYTES {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness response exceeds size bound".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Request(
                "unexpected observation witness response".to_string(),
            ));
        };
        if *chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || *observer != self.local_node_id
            || *responder == self.local_node_id
            || *responder == [0u8; 32]
            || *checkpoint_sequence == 0
            || *checkpoint_hash == [0u8; 32]
            || *outcome != DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1
            || *response_timestamp == 0
            || *response_timestamp > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness response contract mismatch".to_string(),
            ));
        }
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            chain_id,
            request_id,
            observer,
            *checkpoint_sequence,
            checkpoint_hash,
            responder,
            *response_timestamp,
            *outcome,
        );
        IdentityPublicKey::from_bytes(responder)
            .and_then(|key| key.verify(&signing_bytes, signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Request(
                    "observation witness response signature is invalid".to_string(),
                )
            })?;

        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let checkpoint_blob: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT checkpoint_blob FROM directory_observation_checkpoints
                 WHERE sequence = ?1 AND checkpoint_hash = ?2",
                params![
                    u64_to_i64(
                        *checkpoint_sequence,
                        "observation witness checkpoint sequence"
                    )?,
                    checkpoint_hash.as_slice()
                ],
                |row| row.get(0),
            )
            .optional()?;
        let Some(checkpoint_blob) = checkpoint_blob else {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness references an unknown local checkpoint".to_string(),
            ));
        };
        let checkpoint = decode_observation_checkpoint(&checkpoint_blob)?;
        if checkpoint.observer != *observer
            || checkpoint.sequence != *checkpoint_sequence
            || checkpoint.hash() != *checkpoint_hash
            || *response_timestamp < checkpoint.observed_at
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness does not match its local checkpoint".to_string(),
            ));
        }
        let existing_hash: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT checkpoint_hash FROM directory_observation_checkpoint_witnesses
                 WHERE checkpoint_sequence = ?1 AND witness_node_id = ?2",
                params![
                    u64_to_i64(
                        *checkpoint_sequence,
                        "observation witness checkpoint sequence"
                    )?,
                    responder.as_slice()
                ],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(existing_hash) = existing_hash {
            if bytes32(
                &existing_hash,
                "observation witness existing checkpoint hash",
            )? != *checkpoint_hash
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation witness signed conflicting hashes at one sequence".to_string(),
                ));
            }
            return Ok(false);
        }
        let existing_witnesses = nonnegative_i64_to_u64(
            transaction.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoint_witnesses
                 WHERE checkpoint_sequence = ?1",
                params![u64_to_i64(
                    *checkpoint_sequence,
                    "observation witness checkpoint sequence"
                )?],
                |row| row.get(0),
            )?,
            "observation checkpoint witness count",
        )?;
        let maximum_witnesses =
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness producer bound exceeds u64".to_string(),
                )
            })?;
        if existing_witnesses >= maximum_witnesses {
            return Err(DirectoryReplicaStoreError::Request(
                "observation checkpoint witness set exceeds producer bound".to_string(),
            ));
        }
        transaction.execute(
            "INSERT INTO directory_observation_checkpoint_witnesses
                (checkpoint_hash, checkpoint_sequence, observer, witness_node_id,
                 witnessed_at, response_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                checkpoint_hash.as_slice(),
                u64_to_i64(
                    *checkpoint_sequence,
                    "observation witness checkpoint sequence"
                )?,
                observer.as_slice(),
                responder.as_slice(),
                u64_to_i64(*response_timestamp, "observation witness timestamp")?,
                response_blob,
            ],
        )?;
        transaction.commit()?;
        drop(connection);
        Ok(true)
    }

    /// Atomically persists one bounded round of privacy-safe witness outcomes.
    ///
    /// Only aggregate mutually exclusive counters and timestamps are retained.
    /// The row contains no witness identity, endpoint, request id, signature,
    /// checkpoint hash, response body, route, or user-plane metadata.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] when the round is empty or over
    /// the protocol producer bound, time/sequence regresses, counters overflow,
    /// the existing aggregate is malformed, or `SQLite` rejects the write.
    pub fn persist_observation_witness_outcome_round(
        &self,
        checkpoint_sequence: u64,
        observed_at: u64,
        outcomes: &[DirectoryObservationWitnessOutcome],
    ) -> Result<DirectoryObservationWitnessOutcomeSnapshot, DirectoryReplicaStoreError> {
        if checkpoint_sequence == 0
            || observed_at == 0
            || outcomes.is_empty()
            || outcomes.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness outcome round fields are invalid".to_string(),
            ));
        }
        let round = DirectoryObservationWitnessOutcomeCounters::from_outcomes(outcomes);
        let mut connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.local_node_id)?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let checkpoint_exists = transaction.query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM directory_observation_checkpoints WHERE sequence = ?1
             )",
            params![u64_to_i64(
                checkpoint_sequence,
                "observation witness checkpoint sequence"
            )?],
            |row| row.get::<_, i64>(0),
        )?;
        if checkpoint_exists != 1 {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness outcome references an unknown checkpoint".to_string(),
            ));
        }
        let snapshot = Self::load_observation_witness_outcome_snapshot(&transaction)?
            .next_durable_round(checkpoint_sequence, observed_at, round)?;
        Self::upsert_observation_witness_outcome_snapshot(&transaction, &snapshot)?;
        transaction.commit()?;
        drop(connection);
        Ok(snapshot)
    }

    // The long SQL statement is intentionally isolated from policy and state
    // transition logic so its column/parameter ordering can be audited as one unit.
    #[allow(clippy::too_many_lines)]
    fn upsert_observation_witness_outcome_snapshot(
        transaction: &Transaction<'_>,
        snapshot: &DirectoryObservationWitnessOutcomeSnapshot,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let totals = snapshot.totals;
        let round = snapshot.last_round;
        let observed_at = snapshot.last_round_at.ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness outcome round timestamp is missing".to_string(),
            )
        })?;
        transaction.execute(
            "INSERT INTO directory_observation_witness_outcomes
                (singleton, rounds_total, attempts_total, accepted_total,
                 evidence_unavailable_total, evidence_conflict_total,
                 peer_unavailable_total, transport_failures_total,
                 verification_failures_total, persistence_failures_total,
                 last_checkpoint_sequence, last_round_at, last_success_at,
                 last_failure_at, last_round_attempts, last_round_accepted,
                 last_round_evidence_unavailable, last_round_evidence_conflict,
                 last_round_peer_unavailable, last_round_transport_failures,
                 last_round_verification_failures,
                 last_round_persistence_failures, updated_at)
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                     ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?11)
             ON CONFLICT(singleton) DO UPDATE SET
                 rounds_total = excluded.rounds_total,
                 attempts_total = excluded.attempts_total,
                 accepted_total = excluded.accepted_total,
                 evidence_unavailable_total = excluded.evidence_unavailable_total,
                 evidence_conflict_total = excluded.evidence_conflict_total,
                 peer_unavailable_total = excluded.peer_unavailable_total,
                 transport_failures_total = excluded.transport_failures_total,
                 verification_failures_total = excluded.verification_failures_total,
                 persistence_failures_total = excluded.persistence_failures_total,
                 last_checkpoint_sequence = excluded.last_checkpoint_sequence,
                 last_round_at = excluded.last_round_at,
                 last_success_at = excluded.last_success_at,
                 last_failure_at = excluded.last_failure_at,
                 last_round_attempts = excluded.last_round_attempts,
                 last_round_accepted = excluded.last_round_accepted,
                 last_round_evidence_unavailable = excluded.last_round_evidence_unavailable,
                 last_round_evidence_conflict = excluded.last_round_evidence_conflict,
                 last_round_peer_unavailable = excluded.last_round_peer_unavailable,
                 last_round_transport_failures = excluded.last_round_transport_failures,
                 last_round_verification_failures = excluded.last_round_verification_failures,
                 last_round_persistence_failures = excluded.last_round_persistence_failures,
                 updated_at = excluded.updated_at",
            params![
                u64_to_i64(snapshot.rounds, "observation witness outcome rounds")?,
                u64_to_i64(totals.attempts(), "observation witness outcome attempts")?,
                u64_to_i64(totals.accepted, "observation witness accepted total")?,
                u64_to_i64(
                    totals.evidence_unavailable,
                    "observation witness unavailable total"
                )?,
                u64_to_i64(
                    totals.evidence_conflict,
                    "observation witness conflict total"
                )?,
                u64_to_i64(totals.peer_unavailable, "observation witness peer total")?,
                u64_to_i64(
                    totals.transport_failures,
                    "observation witness transport total"
                )?,
                u64_to_i64(
                    totals.verification_failures,
                    "observation witness verification total"
                )?,
                u64_to_i64(
                    totals.persistence_failures,
                    "observation witness persistence total"
                )?,
                u64_to_i64(
                    snapshot.last_checkpoint_sequence,
                    "observation witness checkpoint sequence"
                )?,
                u64_to_i64(observed_at, "observation witness round timestamp")?,
                snapshot
                    .last_success_at
                    .map(|value| u64_to_i64(value, "observation witness success timestamp"))
                    .transpose()?,
                snapshot
                    .last_failure_at
                    .map(|value| u64_to_i64(value, "observation witness failure timestamp"))
                    .transpose()?,
                u64_to_i64(round.attempts(), "observation witness last round attempts")?,
                u64_to_i64(round.accepted, "observation witness last round accepted")?,
                u64_to_i64(
                    round.evidence_unavailable,
                    "observation witness last round unavailable"
                )?,
                u64_to_i64(
                    round.evidence_conflict,
                    "observation witness last round conflict"
                )?,
                u64_to_i64(
                    round.peer_unavailable,
                    "observation witness last round peer unavailable"
                )?,
                u64_to_i64(
                    round.transport_failures,
                    "observation witness last round transport"
                )?,
                u64_to_i64(
                    round.verification_failures,
                    "observation witness last round verification"
                )?,
                u64_to_i64(
                    round.persistence_failures,
                    "observation witness last round persistence"
                )?,
            ],
        )?;
        Ok(())
    }

    fn insert_observation_checkpoint(
        transaction: &Transaction<'_>,
        previous: ObservationCheckpointTip,
        eligible_tips: &[DirectoryReplicaTip],
        observation_root: [u8; 32],
        identity: &IdentityKeyPair,
        observed_at: u64,
    ) -> Result<DirectoryObservationCheckpointAppendReport, DirectoryReplicaStoreError> {
        if observed_at < previous.observed_at {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint timestamp regressed".to_string(),
            ));
        }
        let sequence = previous.sequence.checked_add(1).ok_or_else(|| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint sequence exhausted".to_string(),
            )
        })?;
        let producer_count = u16::try_from(eligible_tips.len()).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint producer count exceeds u16".to_string(),
            )
        })?;
        let producer_tips = eligible_tips
            .iter()
            .map(|tip| DirectoryObservationTipV1 {
                producer: tip.producer,
                tip_height: tip.tip_height,
                tip_hash: tip.tip_hash,
            })
            .collect();
        let checkpoint = DirectoryObservationCheckpointV1::new_signed(
            sequence,
            observed_at,
            previous.checkpoint_hash,
            producer_count,
            producer_tips,
            observation_root,
            identity,
        )
        .map_err(|error| DirectoryReplicaStoreError::Integrity(error.to_string()))?;
        let checkpoint_hash = checkpoint.hash();
        let checkpoint_blob = encode_observation_checkpoint(&checkpoint)?;
        transaction.execute(
            "INSERT INTO directory_observation_checkpoints
                (sequence, checkpoint_hash, previous_checkpoint_hash, observed_at,
                 observation_root, producer_count, checkpoint_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                u64_to_i64(sequence, "observation checkpoint sequence")?,
                checkpoint_hash.as_slice(),
                checkpoint.previous_checkpoint_hash.as_slice(),
                u64_to_i64(observed_at, "observation checkpoint timestamp")?,
                observation_root.as_slice(),
                i64::from(producer_count),
                checkpoint_blob,
            ],
        )?;
        Ok(DirectoryObservationCheckpointAppendReport {
            appended: true,
            sequence,
            checkpoint_hash,
            observed_at,
            producer_count,
            observation_root,
        })
    }

    const fn observation_checkpoint_report(
        appended: bool,
        tip: ObservationCheckpointTip,
    ) -> DirectoryObservationCheckpointAppendReport {
        DirectoryObservationCheckpointAppendReport {
            appended,
            sequence: tip.sequence,
            checkpoint_hash: tip.checkpoint_hash,
            observed_at: tip.observed_at,
            producer_count: tip.producer_count,
            observation_root: tip.observation_root,
        }
    }

    fn validate_convergence_producers(
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

    fn observation_convergence_from_connection(
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

    fn eligible_convergence_tips(
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

    fn recent_commitment_occurrences(
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

    fn complete_observation_convergence(
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

    #[allow(clippy::too_many_arguments)]
    fn import_verified_page_with_mode(
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
        let has_more = verify_range_response_evidence(
            signed_response_frame,
            &producer,
            blocks,
            advertised_tip_height,
            &advertised_tip_hash,
            observed_at,
        )?;
        validate_page_tip_contract(
            blocks,
            has_more,
            advertised_tip_height,
            &advertised_tip_hash,
        )?;
        let descriptors = validate_exact_descriptor_objects(blocks, objects)?;

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let producer_existed = Self::producer_row_exists(&transaction, &producer)?;
        Self::ensure_producer_row(&transaction, &producer, observed_at)?;
        Self::reconcile_import_mode(
            &transaction,
            &producer,
            observed_at,
            producer_existed,
            mode,
        )?;
        let mut tip = Self::load_tip(&transaction, &producer)?;
        if tip.quarantined {
            return Err(DirectoryReplicaStoreError::Quarantined(
                tip.quarantine_kind
                    .unwrap_or_else(|| "producer_fork".to_string()),
            ));
        }

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
        if blocks.is_empty() {
            if advertised_tip_height > tip.tip_height {
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

    /// Audits metadata, evidence, resolutions, prefixes, indexes, and retries.
    ///
    /// # Errors
    /// Returns [`DirectoryReplicaStoreError`] on the first malformed row,
    /// invalid signature/link/root, missing object/index, or invalid incident.
    pub fn audit(
        &self,
        observed_at: u64,
    ) -> Result<DirectoryReplicaAudit, DirectoryReplicaStoreError> {
        let connection = self.connection.lock();
        Self::audit_connection(&connection, &self.local_node_id, observed_at)
    }

    fn decode_observation_witness_policy(
        row: StoredObservationWitnessPolicyRow,
    ) -> Result<([u8; 32], DirectoryObservationWitnessPolicyEpoch), DirectoryReplicaStoreError>
    {
        let epoch = positive_i64_to_u64(row.epoch, "observation witness policy epoch")?;
        let activated_at = positive_i64_to_u64(
            row.activated_at,
            "observation witness policy activation timestamp",
        )?;
        let minimum_witnesses = usize::try_from(positive_i64_to_u64(
            row.witness_threshold,
            "observation witness policy threshold",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness policy threshold exceeds usize".to_string(),
            )
        })?;
        let witness_count = usize::try_from(nonnegative_i64_to_u64(
            row.witness_count,
            "observation witness policy member count",
        )?)
        .map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation witness policy member count exceeds usize".to_string(),
            )
        })?;
        if witness_count > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS
            || row.witness_node_ids.len() != witness_count.saturating_mul(32)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy member blob is malformed".to_string(),
            ));
        }
        let witness_node_ids = row
            .witness_node_ids
            .chunks_exact(32)
            .map(|node_id| bytes32(node_id, "observation witness policy member"))
            .collect::<Result<Vec<_>, _>>()?;
        let policy = DirectoryObservationWitnessPolicyEpoch {
            epoch,
            previous_policy_digest: bytes32(
                &row.previous_policy_digest,
                "observation witness previous policy digest",
            )?,
            activated_at,
            witness_node_ids,
            minimum_witnesses,
            signer_node_id: bytes32(&row.signer_node_id, "observation witness policy signer")?,
            signature: bytes64(&row.signature, "observation witness policy signature")?,
        };
        policy.validate_unsigned_fields()?;
        let stored_digest = bytes32(&row.policy_digest, "observation witness policy digest")?;
        Ok((stored_digest, policy))
    }

    fn audit_observation_witness_policies(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<ObservationWitnessPolicyAudit, DirectoryReplicaStoreError> {
        let (metadata_epoch, metadata_head) = connection.query_row(
            "SELECT witness_policy_epoch, witness_policy_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_epoch =
            nonnegative_i64_to_u64(metadata_epoch, "replica metadata witness policy epoch")?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata witness policy head"))
            .transpose()?;

        let mut statement = connection.prepare(
            "SELECT epoch, policy_digest, previous_policy_digest, activated_at,
                    witness_threshold, witness_count, witness_node_ids,
                    signer_node_id, signature
             FROM directory_observation_witness_policies ORDER BY epoch ASC",
        )?;
        let mut rows = statement.query([])?;
        let mut audit = ObservationWitnessPolicyAudit::default();
        let mut expected_previous_digest = [0u8; 32];
        let mut previous_activated_at = 0u64;
        while let Some(row) = rows.next()? {
            let stored = StoredObservationWitnessPolicyRow {
                epoch: row.get(0)?,
                policy_digest: row.get(1)?,
                previous_policy_digest: row.get(2)?,
                activated_at: row.get(3)?,
                witness_threshold: row.get(4)?,
                witness_count: row.get(5)?,
                witness_node_ids: row.get(6)?,
                signer_node_id: row.get(7)?,
                signature: row.get(8)?,
            };
            let (stored_digest, policy) = Self::decode_observation_witness_policy(stored)?;
            let expected_epoch = audit.epochs.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy epoch exhausted".to_string(),
                )
            })?;
            if policy.epoch != expected_epoch
                || policy.previous_policy_digest != expected_previous_digest
                || policy.activated_at < previous_activated_at
                || policy.signer_node_id != *local_node_id
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation witness policy history is not canonical".to_string(),
                ));
            }
            IdentityPublicKey::from_bytes(&policy.signer_node_id)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy signer is invalid".to_string(),
                    )
                })?
                .verify(&policy.signing_bytes(), &policy.signature)
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "observation witness policy signature is invalid".to_string(),
                    )
                })?;
            if policy.digest() != stored_digest {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation witness policy digest is invalid".to_string(),
                ));
            }
            audit.epochs = expected_epoch;
            expected_previous_digest = stored_digest;
            previous_activated_at = policy.activated_at;
            audit.current_digest = stored_digest;
            audit.current = Some(policy);
        }
        drop(rows);
        drop(statement);
        if audit.epochs != metadata_epoch
            || (audit.epochs == 0 && metadata_head.is_some())
            || (audit.epochs > 0 && metadata_head != Some(audit.current_digest))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy head does not match audited history".to_string(),
            ));
        }
        Ok(audit)
    }

    /// Loads and verifies only the metadata-anchored policy head for bounded
    /// runtime status. Complete history verification remains a startup and
    /// explicit operator-audit responsibility.
    fn load_current_observation_witness_policy(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<ObservationWitnessPolicyAudit, DirectoryReplicaStoreError> {
        let (metadata_epoch, metadata_head) = connection.query_row(
            "SELECT witness_policy_epoch, witness_policy_head
             FROM directory_replica_meta WHERE singleton = 1",
            [],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )?;
        let metadata_epoch =
            nonnegative_i64_to_u64(metadata_epoch, "replica metadata witness policy epoch")?;
        let metadata_head = metadata_head
            .as_deref()
            .map(|value| bytes32(value, "replica metadata witness policy head"))
            .transpose()?;
        if metadata_epoch == 0 {
            if metadata_head.is_some() {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "empty observation witness policy has a non-empty head".to_string(),
                ));
            }
            return Ok(ObservationWitnessPolicyAudit::default());
        }
        let stored = connection
            .query_row(
                "SELECT epoch, policy_digest, previous_policy_digest, activated_at,
                        witness_threshold, witness_count, witness_node_ids,
                        signer_node_id, signature
                 FROM directory_observation_witness_policies WHERE epoch = ?1",
                params![u64_to_i64(
                    metadata_epoch,
                    "replica metadata witness policy epoch"
                )?],
                |row| {
                    Ok(StoredObservationWitnessPolicyRow {
                        epoch: row.get(0)?,
                        policy_digest: row.get(1)?,
                        previous_policy_digest: row.get(2)?,
                        activated_at: row.get(3)?,
                        witness_threshold: row.get(4)?,
                        witness_count: row.get(5)?,
                        witness_node_ids: row.get(6)?,
                        signer_node_id: row.get(7)?,
                        signature: row.get(8)?,
                    })
                },
            )
            .optional()?
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy head row is missing".to_string(),
                )
            })?;
        let (stored_digest, policy) = Self::decode_observation_witness_policy(stored)?;
        if policy.epoch != metadata_epoch
            || policy.signer_node_id != *local_node_id
            || metadata_head != Some(stored_digest)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy head is inconsistent".to_string(),
            ));
        }
        IdentityPublicKey::from_bytes(&policy.signer_node_id)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy signer is invalid".to_string(),
                )
            })?
            .verify(&policy.signing_bytes(), &policy.signature)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness policy signature is invalid".to_string(),
                )
            })?;
        if policy.digest() != stored_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness policy digest is invalid".to_string(),
            ));
        }
        Ok(ObservationWitnessPolicyAudit {
            epochs: metadata_epoch,
            current: Some(policy),
            current_digest: stored_digest,
        })
    }

    fn audit_remote_observation_policy_anchors(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<u64, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT observer, policy_epoch, previous_policy_digest, policy_digest,
                    request_timestamp, request_blob
             FROM directory_observation_remote_policy_anchors
             ORDER BY observer ASC, policy_epoch ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, Vec<u8>>(5)?,
            ))
        })?;
        let mut heads = HashMap::<[u8; 32], (u64, [u8; 32])>::new();
        let mut count = 0u64;
        for row in rows {
            let (observer, epoch, previous_digest, digest, request_timestamp, request_blob) = row?;
            let observer = bytes32(&observer, "remote policy anchor observer")?;
            let epoch = positive_i64_to_u64(epoch, "remote policy anchor epoch")?;
            let previous_digest =
                bytes32(&previous_digest, "remote policy anchor previous digest")?;
            let digest = bytes32(&digest, "remote policy anchor digest")?;
            let request_timestamp =
                positive_i64_to_u64(request_timestamp, "remote policy anchor timestamp")?;
            let request = decode_directory_sync_message(&request_blob)
                .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
            if encode_directory_sync_message(&request)
                .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
                != request_blob
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor request is noncanonical".to_string(),
                ));
            }
            let DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
                chain_id,
                request_id,
                requester,
                request_timestamp: signed_at,
                policy_epoch,
                previous_policy_digest,
                policy_digest,
                signature,
            } = request
            else {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor row contains an unexpected frame".to_string(),
                ));
            };
            let position_valid = (epoch == 1 && previous_digest == [0u8; 32])
                || (epoch > 1 && previous_digest != [0u8; 32]);
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                || requester != observer
                || requester == *local_node_id
                || policy_epoch != epoch
                || previous_policy_digest != previous_digest
                || policy_digest != digest
                || signed_at != request_timestamp
                || digest == [0u8; 32]
                || !position_valid
                || request_timestamp > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor row violates its signed contract".to_string(),
                ));
            }
            let signing_bytes = directory_policy_anchor_request_signing_bytes(
                &chain_id,
                &request_id,
                &requester,
                signed_at,
                policy_epoch,
                &previous_policy_digest,
                &policy_digest,
            );
            IdentityPublicKey::from_bytes(&requester)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "remote policy anchor signature is invalid".to_string(),
                    )
                })?;
            if let Some((previous_epoch, previous_head)) = heads.get(&observer) {
                if epoch != previous_epoch.saturating_add(1) || previous_digest != *previous_head {
                    return Err(DirectoryReplicaStoreError::Integrity(
                        "remote policy anchor history is not contiguous".to_string(),
                    ));
                }
            }
            heads.insert(observer, (epoch, digest));
            count = count.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "remote policy anchor count overflow".to_string(),
                )
            })?;
        }
        Ok(count)
    }

    fn verify_observation_policy_anchor_receipt_row(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
        row: StoredObservationWitnessPolicyAnchorReceiptRow,
    ) -> Result<(u64, [u8; 32], [u8; 32]), DirectoryReplicaStoreError> {
        let epoch = positive_i64_to_u64(row.policy_epoch, "policy anchor receipt epoch")?;
        let digest = bytes32(&row.policy_digest, "policy anchor receipt digest")?;
        let observer = bytes32(&row.observer, "policy anchor receipt observer")?;
        let witness = bytes32(&row.witness_node_id, "policy anchor receipt witness")?;
        let witnessed_at =
            positive_i64_to_u64(row.witnessed_at, "policy anchor receipt timestamp")?;
        let response = decode_directory_sync_message(&row.response_blob)
            .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
        if encode_directory_sync_message(&response)
            .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
            != row.response_blob
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt is noncanonical".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id,
            request_id,
            observer: signed_observer,
            policy_epoch,
            policy_digest,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt contains an unexpected frame".to_string(),
            ));
        };
        if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || observer != *local_node_id
            || signed_observer != observer
            || policy_epoch != epoch
            || policy_digest != digest
            || responder != witness
            || response_timestamp != witnessed_at
            || outcome != DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1
            || witnessed_at > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt violates its signed contract".to_string(),
            ));
        }
        let policy_members: (Vec<u8>, Vec<u8>) = connection.query_row(
            "SELECT policy_digest, witness_node_ids
             FROM directory_observation_witness_policies WHERE epoch = ?1",
            params![u64_to_i64(epoch, "policy anchor receipt epoch")?],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if policy_members.1.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS * 32
            || policy_members.1.len() % 32 != 0
            || bytes32(&policy_members.0, "policy anchor local policy digest")? != digest
            || !policy_members
                .1
                .chunks_exact(32)
                .any(|member| member == witness.as_slice())
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt is not admitted by its policy epoch".to_string(),
            ));
        }
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            &chain_id,
            &request_id,
            &signed_observer,
            policy_epoch,
            &policy_digest,
            &responder,
            response_timestamp,
            outcome,
        );
        IdentityPublicKey::from_bytes(&responder)
            .and_then(|key| key.verify(&signing_bytes, &signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "policy anchor receipt signature is invalid".to_string(),
                )
            })?;
        Ok((epoch, digest, witness))
    }

    fn audit_observation_policy_anchor_receipts(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<Vec<(u64, [u8; 32], [u8; 32])>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT policy_epoch, policy_digest, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_policy_anchor_receipts
             ORDER BY policy_epoch ASC, witness_node_id ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredObservationWitnessPolicyAnchorReceiptRow {
                policy_epoch: row.get(0)?,
                policy_digest: row.get(1)?,
                observer: row.get(2)?,
                witness_node_id: row.get(3)?,
                witnessed_at: row.get(4)?,
                response_blob: row.get(5)?,
            })
        })?;
        let mut verified = Vec::new();
        for row in rows {
            verified.push(Self::verify_observation_policy_anchor_receipt_row(
                connection,
                local_node_id,
                observed_at,
                row?,
            )?);
        }
        Ok(verified)
    }

    fn audit_current_observation_policy_anchor_receipts(
        connection: &Connection,
        local_node_id: &[u8; 32],
        policy_epoch: u64,
        policy_digest: &[u8; 32],
        observed_at: u64,
    ) -> Result<Vec<(u64, [u8; 32], [u8; 32])>, DirectoryReplicaStoreError> {
        let current = Self::load_current_observation_witness_policy(connection, local_node_id)?;
        let Some(current_policy) = current.current else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt query has no current local policy".to_string(),
            ));
        };
        if current_policy.epoch != policy_epoch || current.current_digest != *policy_digest {
            return Err(DirectoryReplicaStoreError::Integrity(
                "policy anchor receipt query does not match the current local policy".to_string(),
            ));
        }
        let row_limit = MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS + 1;
        let mut statement = connection.prepare(
            "SELECT policy_epoch, policy_digest, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_policy_anchor_receipts
             WHERE policy_epoch = ?1 AND policy_digest = ?2
             ORDER BY witness_node_id ASC LIMIT ?3",
        )?;
        let rows = statement.query_map(
            params![
                u64_to_i64(policy_epoch, "policy anchor receipt query epoch")?,
                policy_digest.as_slice(),
                i64::try_from(row_limit).map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "policy anchor receipt query limit exceeds i64".to_string(),
                    )
                })?
            ],
            |row| {
                Ok(StoredObservationWitnessPolicyAnchorReceiptRow {
                    policy_epoch: row.get(0)?,
                    policy_digest: row.get(1)?,
                    observer: row.get(2)?,
                    witness_node_id: row.get(3)?,
                    witnessed_at: row.get(4)?,
                    response_blob: row.get(5)?,
                })
            },
        )?;
        let mut verified = Vec::new();
        for row in rows {
            let verified_row = Self::verify_observation_policy_anchor_receipt_row(
                connection,
                local_node_id,
                observed_at,
                row?,
            )?;
            if !current_policy.witness_node_ids.contains(&verified_row.2) {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "current policy anchor receipt witness is outside the signed policy"
                        .to_string(),
                ));
            }
            verified.push(verified_row);
        }
        if verified.len() > current_policy.witness_node_ids.len() {
            return Err(DirectoryReplicaStoreError::Integrity(
                "current policy anchor receipts exceed signed policy membership".to_string(),
            ));
        }
        Ok(verified)
    }

    fn audit_connection(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<DirectoryReplicaAudit, DirectoryReplicaStoreError> {
        Self::validate_metadata(connection, local_node_id)?;
        let producers = Self::load_all_tips(connection)?;
        let mirror_producers = Self::audit_mirror_registry(connection, local_node_id)?;
        let observation_witness_policies =
            Self::audit_observation_witness_policies(connection, local_node_id)?;
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

    fn audit_mirror_registry(
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
            let descriptor_sequence = positive_i64_to_u64(
                *descriptor_sequence,
                "directory mirror descriptor sequence",
            )?;
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

    fn initialize_schema(
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
    fn create_schema_tables(
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
                 CHECK ((witness_policy_epoch = 0 AND witness_policy_head IS NULL)
                     OR (witness_policy_epoch > 0 AND witness_policy_head IS NOT NULL))
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

    fn migrate_schema_metadata(
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
                        Self::set_schema_version(transaction, version)?;
                    }
                    DIRECTORY_REPLICA_SCHEMA_VERSION_V1 | DIRECTORY_REPLICA_SCHEMA_VERSION_V2 => {
                        Self::add_resolution_columns(transaction)?;
                        Self::add_witness_policy_metadata_columns(transaction)?;
                        Self::require_mirror_registry_table(transaction)?;
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

    fn set_schema_version(
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

    fn add_resolution_columns(
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

    fn require_resolution_columns(
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

    fn add_witness_policy_metadata_columns(
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

    fn require_witness_policy_metadata_columns(
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

    fn require_mirror_registry_table(
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

    fn metadata_has_column(
        transaction: &Transaction<'_>,
        expected: &str,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let mut statement = transaction.prepare("PRAGMA table_info(directory_replica_meta)")?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(columns.iter().any(|column| column == expected))
    }

    fn table_has_column(
        transaction: &Transaction<'_>,
        expected: &str,
    ) -> Result<bool, DirectoryReplicaStoreError> {
        let mut statement = transaction.prepare("PRAGMA table_info(directory_replica_chains)")?;
        let columns = statement
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(columns.iter().any(|column| column == expected))
    }

    fn validate_metadata(
        connection: &Connection,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        let metadata = connection
            .query_row(
                "SELECT schema_version, chain_id, local_node_id,
                        witness_policy_epoch, witness_policy_head
                 FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<Vec<u8>>>(4)?,
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
        Ok(())
    }

    fn ensure_producer_row(
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

    fn producer_row_exists(
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

    fn reconcile_import_mode(
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

    fn clear_retry_state(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        transaction.execute(
            "DELETE FROM directory_replica_retry_state WHERE producer = ?1",
            params![producer.as_slice()],
        )?;
        Ok(())
    }

    fn load_observation_checkpoint_tip(
        connection: &Connection,
    ) -> Result<ObservationCheckpointTip, DirectoryReplicaStoreError> {
        let row = connection
            .query_row(
                "SELECT sequence, checkpoint_hash, observed_at, producer_count,
                        observation_root
                 FROM directory_observation_checkpoints
                 ORDER BY sequence DESC LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Vec<u8>>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((sequence, checkpoint_hash, observed_at, producer_count, observation_root)) = row
        else {
            return Ok(ObservationCheckpointTip::default());
        };
        let producer_count = u16::try_from(producer_count).map_err(|_| {
            DirectoryReplicaStoreError::Integrity(
                "observation checkpoint producer count exceeds u16".to_string(),
            )
        })?;
        if usize::from(producer_count) < 2
            || usize::from(producer_count) > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint producer count is invalid".to_string(),
            ));
        }
        Ok(ObservationCheckpointTip {
            sequence: positive_i64_to_u64(sequence, "observation checkpoint sequence")?,
            checkpoint_hash: bytes32(&checkpoint_hash, "observation checkpoint hash")?,
            observed_at: positive_i64_to_u64(observed_at, "observation checkpoint timestamp")?,
            producer_count,
            observation_root: bytes32(&observation_root, "observation checkpoint root")?,
        })
    }

    fn load_observation_witness_summary(
        connection: &Connection,
    ) -> Result<ObservationWitnessAudit, DirectoryReplicaStoreError> {
        let witnesses = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoint_witnesses",
                [],
                |row| row.get(0),
            )?,
            "observation witness count",
        )?;
        if witnesses == 0 {
            return Ok(ObservationWitnessAudit::default());
        }
        let latest_sequence = positive_i64_to_u64(
            connection.query_row(
                "SELECT MAX(checkpoint_sequence)
                 FROM directory_observation_checkpoint_witnesses",
                [],
                |row| row.get(0),
            )?,
            "latest witnessed checkpoint sequence",
        )?;
        let latest_witnesses = nonnegative_i64_to_u64(
            connection.query_row(
                "SELECT COUNT(*) FROM directory_observation_checkpoint_witnesses
                 WHERE checkpoint_sequence = ?1",
                params![u64_to_i64(
                    latest_sequence,
                    "latest witnessed checkpoint sequence"
                )?],
                |row| row.get(0),
            )?,
            "latest checkpoint witness count",
        )?;
        Ok(ObservationWitnessAudit {
            witnesses,
            latest_sequence,
            latest_witnesses,
        })
    }

    fn query_observation_witness_outcome_row(
        connection: &Connection,
    ) -> Result<Option<StoredObservationWitnessOutcomeRow>, DirectoryReplicaStoreError> {
        Ok(connection
            .query_row(
                "SELECT rounds_total, attempts_total, accepted_total,
                        evidence_unavailable_total, evidence_conflict_total,
                        peer_unavailable_total, transport_failures_total,
                        verification_failures_total, persistence_failures_total,
                        last_checkpoint_sequence, last_round_at, last_success_at,
                        last_failure_at, last_round_attempts, last_round_accepted,
                        last_round_evidence_unavailable,
                        last_round_evidence_conflict, last_round_peer_unavailable,
                        last_round_transport_failures,
                        last_round_verification_failures,
                        last_round_persistence_failures, updated_at
                 FROM directory_observation_witness_outcomes WHERE singleton = 1",
                [],
                |row| {
                    Ok(StoredObservationWitnessOutcomeRow {
                        rounds: row.get(0)?,
                        attempts: row.get(1)?,
                        totals: [
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                            row.get(6)?,
                            row.get(7)?,
                            row.get(8)?,
                        ],
                        last_checkpoint_sequence: row.get(9)?,
                        last_round_at: row.get(10)?,
                        last_success_at: row.get(11)?,
                        last_failure_at: row.get(12)?,
                        last_round_attempts: row.get(13)?,
                        last_round: [
                            row.get(14)?,
                            row.get(15)?,
                            row.get(16)?,
                            row.get(17)?,
                            row.get(18)?,
                            row.get(19)?,
                            row.get(20)?,
                        ],
                        updated_at: row.get(21)?,
                    })
                },
            )
            .optional()?)
    }

    fn decode_observation_witness_outcome_counters(
        values: [i64; 7],
        prefix: &str,
    ) -> Result<DirectoryObservationWitnessOutcomeCounters, DirectoryReplicaStoreError> {
        Ok(DirectoryObservationWitnessOutcomeCounters {
            accepted: nonnegative_i64_to_u64(values[0], &format!("{prefix} accepted"))?,
            evidence_unavailable: nonnegative_i64_to_u64(
                values[1],
                &format!("{prefix} evidence unavailable"),
            )?,
            evidence_conflict: nonnegative_i64_to_u64(
                values[2],
                &format!("{prefix} evidence conflict"),
            )?,
            peer_unavailable: nonnegative_i64_to_u64(
                values[3],
                &format!("{prefix} peer unavailable"),
            )?,
            transport_failures: nonnegative_i64_to_u64(
                values[4],
                &format!("{prefix} transport failures"),
            )?,
            verification_failures: nonnegative_i64_to_u64(
                values[5],
                &format!("{prefix} verification failures"),
            )?,
            persistence_failures: nonnegative_i64_to_u64(
                values[6],
                &format!("{prefix} persistence failures"),
            )?,
        })
    }

    fn load_observation_witness_outcome_snapshot(
        connection: &Connection,
    ) -> Result<DirectoryObservationWitnessOutcomeSnapshot, DirectoryReplicaStoreError> {
        let row = Self::query_observation_witness_outcome_row(connection)?;
        let Some(row) = row else {
            return Ok(DirectoryObservationWitnessOutcomeSnapshot::default());
        };
        let totals = Self::decode_observation_witness_outcome_counters(
            row.totals,
            "observation witness total",
        )?;
        let last_round = Self::decode_observation_witness_outcome_counters(
            row.last_round,
            "observation witness last round",
        )?;
        let rounds = positive_i64_to_u64(row.rounds, "observation witness rounds")?;
        let attempts = positive_i64_to_u64(row.attempts, "observation witness attempts")?;
        let last_round_attempts = positive_i64_to_u64(
            row.last_round_attempts,
            "observation witness last round attempts",
        )?;
        let optional_timestamp = |value: Option<i64>, field: &str| {
            value
                .map(|value| positive_i64_to_u64(value, field))
                .transpose()
        };
        let last_checkpoint_sequence = positive_i64_to_u64(
            row.last_checkpoint_sequence,
            "observation witness checkpoint sequence",
        )?;
        let last_round_at = positive_i64_to_u64(
            row.last_round_at,
            "observation witness last round timestamp",
        )?;
        let last_success_at =
            optional_timestamp(row.last_success_at, "observation witness success timestamp")?;
        let last_failure_at =
            optional_timestamp(row.last_failure_at, "observation witness failure timestamp")?;
        let updated_at = positive_i64_to_u64(
            row.updated_at,
            "observation witness outcome update timestamp",
        )?;
        let maximum_round_attempts = u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1)
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness producer bound exceeds u64".to_string(),
                )
            })?;
        if totals.attempts() != attempts
            || last_round.attempts() != last_round_attempts
            || last_round_attempts > maximum_round_attempts
            || rounds > attempts
            || attempts < last_round_attempts
            || updated_at != last_round_at
            || last_success_at.is_some_and(|timestamp| timestamp > last_round_at)
            || last_failure_at.is_some_and(|timestamp| timestamp > last_round_at)
            || (totals.accepted > 0) != last_success_at.is_some()
            || (totals.failures() > 0) != last_failure_at.is_some()
            || (last_round.accepted > 0 && last_success_at != Some(last_round_at))
            || (last_round.failures() > 0 && last_failure_at != Some(last_round_at))
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness outcome aggregate is inconsistent".to_string(),
            ));
        }
        Ok(DirectoryObservationWitnessOutcomeSnapshot {
            rounds,
            totals,
            last_checkpoint_sequence,
            last_round_at: Some(last_round_at),
            last_success_at,
            last_failure_at,
            last_round,
            telemetry_persistence_failures: 0,
        })
    }

    fn load_tip(
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

    fn load_all_tips(
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

    fn load_retry_states(
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

    fn block_hash_at(
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

    fn insert_block(
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

    fn persist_quarantine(
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

    fn insert_incident(
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

    fn audit_producer(
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
        let rows = Self::load_block_rows(connection, &tip.producer)?;
        let mut commitments = Self::load_commitment_index(connection, &tip.producer)?;
        let mut objects = Self::load_descriptor_objects(connection, &tip.producer)?;
        let mut expected_height = 1u64;
        let mut previous_hash = [0u8; 32];
        let mut previous_timestamp = 0u64;
        for row in rows {
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
            let mut actual = commitments.remove(&row.height).unwrap_or_default();
            actual.sort_unstable();
            if actual != block.commitments {
                return Err(DirectoryReplicaStoreError::Integrity(format!(
                    "replica block {height} commitment index mismatch"
                )));
            }
            for commitment in &block.commitments {
                let actual = objects.remove(&commitment.descriptor_hash).ok_or_else(|| {
                    DirectoryReplicaStoreError::Integrity(format!(
                        "replica block {height} is missing a descriptor object"
                    ))
                })?;
                if actual != *commitment {
                    return Err(DirectoryReplicaStoreError::Integrity(format!(
                        "replica block {height} descriptor object mismatch"
                    )));
                }
            }
            report.blocks = report.blocks.saturating_add(1);
            report.commitments = report
                .commitments
                .saturating_add(u64::from(block.header.commitment_count));
            previous_hash = block.hash();
            previous_timestamp = block.header.timestamp;
            expected_height = expected_height.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity("replica height exhausted".to_string())
            })?;
        }
        if !commitments.is_empty() || !objects.is_empty() {
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

    fn load_block_rows(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<Vec<StoredReplicaBlockRow>, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT height, block_hash, prev_block_hash, produced_at,
                    commitment_count, block_blob
             FROM directory_replica_blocks WHERE producer = ?1 ORDER BY height ASC",
        )?;
        let rows = statement
            .query_map(params![producer.as_slice()], |row| {
                Ok(StoredReplicaBlockRow {
                    height: row.get(0)?,
                    block_hash: row.get(1)?,
                    prev_block_hash: row.get(2)?,
                    produced_at: row.get(3)?,
                    commitment_count: row.get(4)?,
                    block_blob: row.get(5)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(DirectoryReplicaStoreError::from)?;
        Ok(rows)
    }

    fn load_commitment_index(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<BTreeMap<i64, Vec<DirectoryDescriptorCommitmentV1>>, DirectoryReplicaStoreError>
    {
        let mut statement = connection.prepare(
            "SELECT block_height, commitment_hash, node_id, sequence_le, descriptor_hash
             FROM directory_replica_commitments WHERE producer = ?1
             ORDER BY block_height ASC, commitment_hash ASC",
        )?;
        let rows = statement.query_map(params![producer.as_slice()], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?;
        let mut index = BTreeMap::new();
        for row in rows {
            let (height, hash, node_id, sequence, descriptor_hash) = row?;
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
            index
                .entry(height)
                .or_insert_with(Vec::new)
                .push(commitment);
        }
        Ok(index)
    }

    fn load_descriptor_objects(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<HashMap<[u8; 32], DirectoryDescriptorCommitmentV1>, DirectoryReplicaStoreError>
    {
        let mut statement = connection.prepare(
            "SELECT descriptor_hash, node_id, sequence_le, descriptor_blob
             FROM directory_replica_descriptor_objects WHERE producer = ?1",
        )?;
        let rows = statement.query_map(params![producer.as_slice()], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })?;
        let mut objects = HashMap::new();
        for row in rows {
            let (hash, node_id, sequence, blob) = row?;
            let descriptor = decode_descriptor_object(&blob)?;
            let commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                    .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
            let sequence: [u8; 8] = sequence.try_into().map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "replica descriptor sequence must contain 8 bytes".to_string(),
                )
            })?;
            let hash = bytes32(&hash, "replica descriptor hash")?;
            if hash != commitment.descriptor_hash
                || bytes32(&node_id, "replica descriptor node id")? != commitment.node_id
                || u64::from_le_bytes(sequence) != commitment.sequence
                || objects.insert(hash, commitment).is_some()
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "replica descriptor object index mismatch".to_string(),
                ));
            }
        }
        Ok(objects)
    }

    fn load_observation_checkpoint_row(
        connection: &Connection,
        sequence: u64,
    ) -> Result<Option<StoredObservationCheckpointRow>, DirectoryReplicaStoreError> {
        connection
            .query_row(
                "SELECT sequence, checkpoint_hash, previous_checkpoint_hash, observed_at,
                        observation_root, producer_count, checkpoint_blob
                 FROM directory_observation_checkpoints WHERE sequence = ?1",
                params![u64_to_i64(sequence, "observation checkpoint sequence")?],
                |row| {
                    Ok(StoredObservationCheckpointRow {
                        sequence: row.get(0)?,
                        checkpoint_hash: row.get(1)?,
                        previous_checkpoint_hash: row.get(2)?,
                        observed_at: row.get(3)?,
                        observation_root: row.get(4)?,
                        producer_count: row.get(5)?,
                        checkpoint_blob: row.get(6)?,
                    })
                },
            )
            .optional()
            .map_err(DirectoryReplicaStoreError::from)
    }

    fn validate_observation_witness_eligibility(
        witnesses: &[[u8; 32]],
    ) -> Result<HashSet<[u8; 32]>, DirectoryReplicaStoreError> {
        if witnesses.is_empty()
            || witnesses.len() > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1
            || witnesses.iter().any(|witness| *witness == [0u8; 32])
        {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness eligibility set is invalid".to_string(),
            ));
        }
        let unique = witnesses.iter().copied().collect::<HashSet<_>>();
        if unique.len() != witnesses.len() {
            return Err(DirectoryReplicaStoreError::Request(
                "observation witness eligibility set contains duplicates".to_string(),
            ));
        }
        Ok(unique)
    }

    fn decode_canonical_observation_checkpoint_row(
        row: &StoredObservationCheckpointRow,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let checkpoint = decode_observation_checkpoint(&row.checkpoint_blob)?;
        if encode_observation_checkpoint(&checkpoint)? != row.checkpoint_blob {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint encoding is not canonical".to_string(),
            ));
        }
        Ok(checkpoint)
    }

    fn verify_observation_checkpoint_row_metadata(
        row: &StoredObservationCheckpointRow,
        checkpoint: &DirectoryObservationCheckpointV1,
        local_node_id: &[u8; 32],
    ) -> Result<(), DirectoryReplicaStoreError> {
        if checkpoint.observer != *local_node_id
            || positive_i64_to_u64(row.sequence, "observation checkpoint sequence")?
                != checkpoint.sequence
            || bytes32(&row.checkpoint_hash, "observation checkpoint hash")? != checkpoint.hash()
            || bytes32(
                &row.previous_checkpoint_hash,
                "observation checkpoint previous hash",
            )? != checkpoint.previous_checkpoint_hash
            || positive_i64_to_u64(row.observed_at, "observation checkpoint timestamp")?
                != checkpoint.observed_at
            || bytes32(&row.observation_root, "observation checkpoint root")?
                != checkpoint.observation_root
            || u16::try_from(row.producer_count).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint producer count exceeds u16".to_string(),
                )
            })? != checkpoint.configured_producer_count
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint row does not match its signed object".to_string(),
            ));
        }
        Ok(())
    }

    fn verify_observation_checkpoint_anchor_row(
        row: &StoredObservationCheckpointRow,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let checkpoint = Self::decode_canonical_observation_checkpoint_row(row)?;
        checkpoint
            .verify_standalone_at(&AERONYX_DIRECTORY_MAINNET_CHAIN_ID, observed_at)
            .map_err(|error| DirectoryReplicaStoreError::Integrity(error.to_string()))?;
        Self::verify_observation_checkpoint_row_metadata(row, &checkpoint, local_node_id)?;
        Ok(checkpoint)
    }

    fn verify_observation_checkpoint_at_sequence(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
        sequence: u64,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let row =
            Self::load_observation_checkpoint_row(connection, sequence)?.ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint selection references a missing row".to_string(),
                )
            })?;
        let (previous_hash, previous_observed_at) = if sequence == 1 {
            ([0u8; 32], 0)
        } else {
            let previous_sequence = sequence.checked_sub(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint predecessor sequence underflow".to_string(),
                )
            })?;
            let previous_row =
                Self::load_observation_checkpoint_row(connection, previous_sequence)?.ok_or_else(
                    || {
                        DirectoryReplicaStoreError::Integrity(
                            "observation checkpoint predecessor is missing".to_string(),
                        )
                    },
                )?;
            let previous = Self::verify_observation_checkpoint_anchor_row(
                &previous_row,
                local_node_id,
                observed_at,
            )?;
            (previous.hash(), previous.observed_at)
        };
        Self::verify_observation_checkpoint_row(
            connection,
            local_node_id,
            observed_at,
            sequence,
            &previous_hash,
            previous_observed_at,
            &row,
        )
    }

    fn latest_verified_observation_witness_set(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<VerifiedObservationWitnessSet, DirectoryReplicaStoreError> {
        let latest_sequence: Option<i64> = connection.query_row(
            "SELECT MAX(checkpoint_sequence)
             FROM directory_observation_checkpoint_witnesses",
            [],
            |row| row.get(0),
        )?;
        let Some(latest_sequence) = latest_sequence else {
            return Ok(VerifiedObservationWitnessSet::default());
        };
        let latest_sequence = positive_i64_to_u64(
            latest_sequence,
            "latest observation witness checkpoint sequence",
        )?;
        let row_limit = i64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1.saturating_add(1))
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness verification bound exceeds i64".to_string(),
                )
            })?;
        let mut statement = connection.prepare(
            "SELECT checkpoint_hash, checkpoint_sequence, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_checkpoint_witnesses
             WHERE checkpoint_sequence = ?1
             ORDER BY witness_node_id ASC
             LIMIT ?2",
        )?;
        let rows = statement.query_map(
            params![
                u64_to_i64(
                    latest_sequence,
                    "latest observation witness checkpoint sequence"
                )?,
                row_limit
            ],
            |row| {
                Ok(StoredObservationWitnessRow {
                    checkpoint_hash: row.get(0)?,
                    checkpoint_sequence: row.get(1)?,
                    observer: row.get(2)?,
                    witness_node_id: row.get(3)?,
                    witnessed_at: row.get(4)?,
                    response_blob: row.get(5)?,
                })
            },
        )?;
        let mut verified_rows = 0usize;
        let mut witness_node_ids = Vec::with_capacity(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1);
        for row in rows {
            let row = row?;
            let verified =
                Self::verify_observation_witness_response(&row, local_node_id, observed_at)?;
            Self::verify_observation_witness_checkpoint(connection, &row, &verified)?;
            witness_node_ids.push(bytes32(
                &row.witness_node_id,
                "latest observation witness node id",
            )?);
            verified_rows = verified_rows.saturating_add(1);
        }
        drop(statement);
        if verified_rows == 0 || verified_rows > MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            return Err(DirectoryReplicaStoreError::Integrity(
                "latest observation witness set violates its verification bound".to_string(),
            ));
        }
        Self::verify_observation_checkpoint_at_sequence(
            connection,
            local_node_id,
            observed_at,
            latest_sequence,
        )?;
        Ok(VerifiedObservationWitnessSet {
            sequence: latest_sequence,
            witness_node_ids,
        })
    }

    fn audit_observation_checkpoints(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<(u64, ObservationCheckpointTip), DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT sequence, checkpoint_hash, previous_checkpoint_hash, observed_at,
                    observation_root, producer_count, checkpoint_blob
             FROM directory_observation_checkpoints ORDER BY sequence ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredObservationCheckpointRow {
                sequence: row.get(0)?,
                checkpoint_hash: row.get(1)?,
                previous_checkpoint_hash: row.get(2)?,
                observed_at: row.get(3)?,
                observation_root: row.get(4)?,
                producer_count: row.get(5)?,
                checkpoint_blob: row.get(6)?,
            })
        })?;
        let mut count = 0u64;
        let mut expected_sequence = 1u64;
        let mut previous_hash = [0u8; 32];
        let mut previous_observed_at = 0u64;
        let mut tip = ObservationCheckpointTip::default();
        for row in rows {
            let row = row?;
            let checkpoint = Self::verify_observation_checkpoint_row(
                connection,
                local_node_id,
                observed_at,
                expected_sequence,
                &previous_hash,
                previous_observed_at,
                &row,
            )?;
            let checkpoint_hash = checkpoint.hash();
            tip = ObservationCheckpointTip {
                sequence: checkpoint.sequence,
                checkpoint_hash,
                observed_at: checkpoint.observed_at,
                producer_count: checkpoint.configured_producer_count,
                observation_root: checkpoint.observation_root,
            };
            previous_hash = checkpoint_hash;
            previous_observed_at = checkpoint.observed_at;
            count = count.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint count exceeds u64".to_string(),
                )
            })?;
            expected_sequence = expected_sequence.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint sequence exhausted".to_string(),
                )
            })?;
        }
        drop(statement);
        Ok((count, tip))
    }

    fn audit_observation_witnesses(
        connection: &Connection,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<ObservationWitnessAudit, DirectoryReplicaStoreError> {
        let mut statement = connection.prepare(
            "SELECT checkpoint_hash, checkpoint_sequence, observer, witness_node_id,
                    witnessed_at, response_blob
             FROM directory_observation_checkpoint_witnesses
             ORDER BY checkpoint_sequence ASC, witness_node_id ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredObservationWitnessRow {
                checkpoint_hash: row.get(0)?,
                checkpoint_sequence: row.get(1)?,
                observer: row.get(2)?,
                witness_node_id: row.get(3)?,
                witnessed_at: row.get(4)?,
                response_blob: row.get(5)?,
            })
        })?;
        let mut audit = ObservationWitnessAudit::default();
        let maximum_witnesses =
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness producer bound exceeds u64".to_string(),
                )
            })?;
        for row in rows {
            let row = row?;
            let verified =
                Self::verify_observation_witness_response(&row, local_node_id, observed_at)?;
            Self::verify_observation_witness_checkpoint(connection, &row, &verified)?;
            audit.witnesses = audit.witnesses.checked_add(1).ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness count exceeds u64".to_string(),
                )
            })?;
            if verified.sequence > audit.latest_sequence {
                audit.latest_sequence = verified.sequence;
                audit.latest_witnesses = 1;
            } else if verified.sequence == audit.latest_sequence {
                audit.latest_witnesses =
                    audit.latest_witnesses.checked_add(1).ok_or_else(|| {
                        DirectoryReplicaStoreError::Integrity(
                            "latest observation witness count exceeds u64".to_string(),
                        )
                    })?;
            }
            if audit.latest_witnesses > maximum_witnesses {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint witness set exceeds producer bound".to_string(),
                ));
            }
        }
        drop(statement);
        Ok(audit)
    }

    fn verify_observation_witness_response(
        row: &StoredObservationWitnessRow,
        local_node_id: &[u8; 32],
        observed_at: u64,
    ) -> Result<VerifiedObservationWitness, DirectoryReplicaStoreError> {
        if row.response_blob.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_BYTES {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness response exceeds size bound".to_string(),
            ));
        }
        let response = decode_directory_sync_message(&row.response_blob).map_err(|error| {
            DirectoryReplicaStoreError::Integrity(format!(
                "observation witness response decode failed: {error}"
            ))
        })?;
        if encode_directory_sync_message(&response).map_err(|error| {
            DirectoryReplicaStoreError::Integrity(format!(
                "observation witness response encode failed: {error}"
            ))
        })? != row.response_blob
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness response encoding is not canonical".to_string(),
            ));
        }
        let DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id,
            request_id,
            observer,
            checkpoint_sequence,
            checkpoint_hash,
            responder,
            response_timestamp,
            outcome,
            signature,
        } = response
        else {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness row contains an unexpected frame".to_string(),
            ));
        };
        let row_sequence = positive_i64_to_u64(
            row.checkpoint_sequence,
            "observation witness checkpoint sequence",
        )?;
        let row_timestamp = positive_i64_to_u64(row.witnessed_at, "observation witness timestamp")?;
        if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
            || observer != *local_node_id
            || responder == *local_node_id
            || responder == [0u8; 32]
            || outcome != DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1
            || checkpoint_sequence != row_sequence
            || checkpoint_hash
                != bytes32(&row.checkpoint_hash, "observation witness checkpoint hash")?
            || observer != bytes32(&row.observer, "observation witness observer")?
            || responder != bytes32(&row.witness_node_id, "observation witness identity")?
            || response_timestamp != row_timestamp
            || response_timestamp > observed_at.saturating_add(RESPONSE_TIMESTAMP_SKEW_SECS)
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness row does not match its signed response".to_string(),
            ));
        }
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &chain_id,
            &request_id,
            &observer,
            checkpoint_sequence,
            &checkpoint_hash,
            &responder,
            response_timestamp,
            outcome,
        );
        IdentityPublicKey::from_bytes(&responder)
            .and_then(|key| key.verify(&signing_bytes, &signature))
            .map_err(|_| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness response signature is invalid".to_string(),
                )
            })?;
        Ok(VerifiedObservationWitness {
            sequence: checkpoint_sequence,
            checkpoint_hash,
            observer,
            response_timestamp,
        })
    }

    fn verify_observation_witness_checkpoint(
        connection: &Connection,
        row: &StoredObservationWitnessRow,
        witness: &VerifiedObservationWitness,
    ) -> Result<(), DirectoryReplicaStoreError> {
        let checkpoint_blob: Option<Vec<u8>> = connection
            .query_row(
                "SELECT checkpoint_blob FROM directory_observation_checkpoints
                 WHERE sequence = ?1 AND checkpoint_hash = ?2",
                params![row.checkpoint_sequence, witness.checkpoint_hash.as_slice()],
                |checkpoint_row| checkpoint_row.get(0),
            )
            .optional()?;
        let checkpoint = checkpoint_blob
            .as_deref()
            .ok_or_else(|| {
                DirectoryReplicaStoreError::Integrity(
                    "observation witness references a missing checkpoint".to_string(),
                )
            })
            .and_then(decode_observation_checkpoint)?;
        if checkpoint.observer != witness.observer
            || checkpoint.sequence != witness.sequence
            || checkpoint.hash() != witness.checkpoint_hash
            || witness.response_timestamp < checkpoint.observed_at
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation witness does not bind its retained checkpoint".to_string(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_observation_checkpoint_row(
        connection: &Connection,
        local_node_id: &[u8; 32],
        verifier_observed_at: u64,
        expected_sequence: u64,
        expected_previous_hash: &[u8; 32],
        previous_observed_at: u64,
        row: &StoredObservationCheckpointRow,
    ) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
        let checkpoint = Self::decode_canonical_observation_checkpoint_row(row)?;
        checkpoint
            .verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_sequence,
                expected_previous_hash,
                previous_observed_at,
                verifier_observed_at,
            )
            .map_err(|error| DirectoryReplicaStoreError::Integrity(error.to_string()))?;
        Self::verify_observation_checkpoint_row_metadata(row, &checkpoint, local_node_id)?;
        if Self::recompute_observation_checkpoint_root(connection, &checkpoint, local_node_id)?
            != checkpoint.observation_root
        {
            return Err(DirectoryReplicaStoreError::Integrity(
                "observation checkpoint root does not match retained producer prefixes".to_string(),
            ));
        }
        Ok(checkpoint)
    }

    fn recompute_observation_checkpoint_root(
        connection: &Connection,
        checkpoint: &DirectoryObservationCheckpointV1,
        local_node_id: &[u8; 32],
    ) -> Result<[u8; 32], DirectoryReplicaStoreError> {
        let mut tips = Vec::with_capacity(checkpoint.producer_tips.len());
        for observed_tip in &checkpoint.producer_tips {
            if Self::observation_block_hash_at(
                connection,
                local_node_id,
                &observed_tip.producer,
                observed_tip.tip_height,
            )? != Some(observed_tip.tip_hash)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "observation checkpoint references a missing producer prefix".to_string(),
                ));
            }
            tips.push(DirectoryReplicaTip {
                producer: observed_tip.producer,
                tip_height: observed_tip.tip_height,
                tip_hash: observed_tip.tip_hash,
                tip_timestamp: 0,
                quarantined: false,
                quarantine_kind: None,
                active_incident_digest: None,
                last_resolution_digest: None,
            });
        }
        let mut snapshot = DirectoryReplicaObservationConvergenceSnapshot {
            configured_producers: u64::from(checkpoint.configured_producer_count),
            eligible_producers: u64::from(checkpoint.configured_producer_count),
            window_blocks: DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS,
            ..DirectoryReplicaObservationConvergenceSnapshot::default()
        };
        let occurrences = Self::recent_commitment_occurrences_with_local_producer(
            connection,
            local_node_id,
            &tips,
            &mut snapshot,
        )?;
        Ok(observation_convergence_root(&tips, &occurrences))
    }

    fn observation_block_hash_at(
        connection: &Connection,
        local_node_id: &[u8; 32],
        producer: &[u8; 32],
        height: u64,
    ) -> Result<Option<[u8; 32]>, DirectoryReplicaStoreError> {
        if producer != local_node_id {
            return Self::block_hash_at(connection, producer, height);
        }
        let block_hash: Option<Vec<u8>> = connection
            .query_row(
                "SELECT block_hash FROM directory_chain_blocks WHERE height = ?1",
                params![u64_to_i64(height, "local observation block height")?],
                |row| row.get(0),
            )
            .optional()?;
        block_hash
            .as_deref()
            .map(|hash| bytes32(hash, "local observation block hash"))
            .transpose()
    }

    fn recent_commitment_occurrences_with_local_producer(
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

    fn audit_incidents(connection: &Connection) -> Result<u64, DirectoryReplicaStoreError> {
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

    fn audit_resolutions(
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

    fn load_verified_resolution_index(
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

    fn load_resolution_rows(
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

    fn verify_resolution_row(
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

    fn audit_resolution_histories(
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

fn canonical_observation_witness_policy_members(
    witness_node_ids: &[[u8; 32]],
    minimum_witnesses: usize,
) -> Result<Vec<[u8; 32]>, DirectoryReplicaStoreError> {
    let mut canonical = witness_node_ids.to_vec();
    canonical.sort_unstable();
    let original_len = canonical.len();
    canonical.dedup();
    if canonical.len() != original_len {
        return Err(DirectoryReplicaStoreError::Request(
            "observation witness policy contains duplicate node identities".to_string(),
        ));
    }
    validate_observation_witness_policy_members(&canonical, minimum_witnesses)?;
    Ok(canonical)
}

fn validate_observation_witness_policy_members(
    witness_node_ids: &[[u8; 32]],
    minimum_witnesses: usize,
) -> Result<(), DirectoryReplicaStoreError> {
    if witness_node_ids.len() > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS
        || minimum_witnesses == 0
        || minimum_witnesses > MAX_DIRECTORY_OBSERVATION_WITNESS_POLICY_MEMBERS
        || (witness_node_ids.is_empty() && minimum_witnesses != 1)
        || (!witness_node_ids.is_empty() && minimum_witnesses > witness_node_ids.len())
        || witness_node_ids.iter().any(|node_id| *node_id == [0u8; 32])
        || witness_node_ids.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(DirectoryReplicaStoreError::Request(
            "observation witness policy pins or threshold are invalid".to_string(),
        ));
    }
    Ok(())
}

fn validate_incident_kind(kind: &str) -> Result<(), DirectoryReplicaStoreError> {
    if kind.is_empty()
        || kind.len() > MAX_DIRECTORY_REPLICA_INCIDENT_KIND_BYTES
        || !matches!(
            kind,
            "signed_tip_rollback"
                | "signed_tip_fork"
                | "signed_empty_range_gap"
                | "signed_block_fork"
                | "descriptor_sequence_equivocation"
        )
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "replica incident kind is invalid".to_string(),
        ));
    }
    Ok(())
}

fn observation_convergence_root(
    eligible_tips: &[DirectoryReplicaTip],
    occurrence_by_commitment: &BTreeMap<[u8; 32], u64>,
) -> [u8; 32] {
    debug_assert!(eligible_tips.len() >= 2);
    let eligible_count = eligible_tips.len() as u64;
    let common_count = occurrence_by_commitment
        .values()
        .filter(|occurrence| **occurrence == eligible_count)
        .count() as u64;
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaObservationConvergence-v1");
    hasher.update(AERONYX_DIRECTORY_MAINNET_CHAIN_ID);
    hasher.update(DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS.to_le_bytes());
    hasher.update(eligible_count.to_le_bytes());
    for tip in eligible_tips {
        hasher.update(tip.producer);
        hasher.update(tip.tip_height.to_le_bytes());
        hasher.update(tip.tip_hash);
    }
    hasher.update(common_count.to_le_bytes());
    for (commitment, occurrence) in occurrence_by_commitment {
        if *occurrence == eligible_count {
            hasher.update(commitment);
        }
    }
    hasher.finalize().into()
}

fn verify_incident_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    verify_signed_range_response_evidence(frame, expected_producer).map(|_| ())
}

struct VerifiedRangeResponseEvidence {
    response_timestamp: u64,
    blocks: Vec<DirectoryCommitmentBlockV1>,
    has_more: bool,
    tip_height: u64,
    tip_hash: [u8; 32],
}

fn verify_signed_range_response_evidence(
    frame: &[u8],
    expected_producer: &[u8; 32],
) -> Result<VerifiedRangeResponseEvidence, DirectoryReplicaStoreError> {
    let message = decode_directory_sync_message(frame)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?;
    if encode_directory_sync_message(&message)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))?
        != frame
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence frame is not canonical".to_string(),
        ));
    }
    match message {
        DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id,
            request_id,
            responder,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID || responder != *expected_producer {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "range evidence belongs to another chain or producer".to_string(),
                ));
            }
            let signing_bytes = directory_block_range_response_signing_bytes(
                &request_id,
                &responder,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&responder)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "range evidence producer signature is invalid".to_string(),
                    )
                })?;
            Ok(VerifiedRangeResponseEvidence {
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
            })
        }
        DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id,
            request_id,
            producer,
            carrier,
            response_timestamp,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature,
        } => {
            if chain_id != AERONYX_DIRECTORY_MAINNET_CHAIN_ID
                || producer != *expected_producer
                || carrier == [0u8; 32]
                || carrier == producer
                || blocks.iter().any(|block| block.header.producer != producer)
            {
                return Err(DirectoryReplicaStoreError::Integrity(
                    "carrier range evidence belongs to another chain or producer".to_string(),
                ));
            }
            let signing_bytes = directory_replica_block_range_response_signing_bytes(
                &chain_id,
                &request_id,
                &producer,
                &carrier,
                response_timestamp,
                &blocks,
                has_more,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&carrier)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| {
                    DirectoryReplicaStoreError::Integrity(
                        "carrier range evidence signature is invalid".to_string(),
                    )
                })?;
            Ok(VerifiedRangeResponseEvidence {
                response_timestamp,
                blocks,
                has_more,
                tip_height,
                tip_hash,
            })
        }
        _ => Err(DirectoryReplicaStoreError::Integrity(
            "incident evidence is not a supported block-range response".to_string(),
        )),
    }
}

fn verify_range_response_evidence(
    frame: &[u8],
    producer: &[u8; 32],
    expected_blocks: &[DirectoryCommitmentBlockV1],
    expected_tip_height: u64,
    expected_tip_hash: &[u8; 32],
    observed_at: u64,
) -> Result<bool, DirectoryReplicaStoreError> {
    let verified = verify_signed_range_response_evidence(frame, producer)?;
    if verified.blocks != expected_blocks
        || verified.tip_height != expected_tip_height
        || verified.tip_hash != *expected_tip_hash
        || verified.response_timestamp.abs_diff(observed_at) > RESPONSE_TIMESTAMP_SKEW_SECS
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "signed range evidence does not match the import".to_string(),
        ));
    }
    Ok(verified.has_more)
}

fn validate_page_tip_contract(
    blocks: &[DirectoryCommitmentBlockV1],
    has_more: bool,
    tip_height: u64,
    tip_hash: &[u8; 32],
) -> Result<(), DirectoryReplicaStoreError> {
    if tip_height == 0 && *tip_hash != [0u8; 32] {
        return Err(DirectoryReplicaStoreError::Integrity(
            "empty advertised tip must use the zero hash".to_string(),
        ));
    }
    let Some(last) = blocks.last() else {
        if has_more {
            return Err(DirectoryReplicaStoreError::Integrity(
                "an empty response cannot advertise more pages".to_string(),
            ));
        }
        return Ok(());
    };
    if last.header.height > tip_height
        || (has_more && last.header.height >= tip_height)
        || (!has_more && (last.header.height != tip_height || last.hash() != *tip_hash))
    {
        return Err(DirectoryReplicaStoreError::Integrity(
            "range pagination fields contradict the signed tip".to_string(),
        ));
    }
    Ok(())
}

fn validate_exact_descriptor_objects<'a>(
    blocks: &[DirectoryCommitmentBlockV1],
    objects: &'a [SignedNodeDescriptor],
) -> Result<HashMap<[u8; 32], &'a SignedNodeDescriptor>, DirectoryReplicaStoreError> {
    let required = blocks
        .iter()
        .flat_map(|block| block.commitments.iter().map(|entry| entry.descriptor_hash))
        .collect::<Vec<_>>();
    let required_set = required.iter().copied().collect::<HashSet<_>>();
    if required_set.len() != required.len() || objects.len() != required.len() {
        return Err(DirectoryReplicaStoreError::Request(
            "descriptor objects must exactly cover unique page commitments".to_string(),
        ));
    }
    let mut mapped = HashMap::with_capacity(objects.len());
    for descriptor in objects {
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
            .map_err(|error| DirectoryReplicaStoreError::Descriptor(error.to_string()))?;
        if !required_set.contains(&commitment.descriptor_hash)
            || mapped
                .insert(commitment.descriptor_hash, descriptor)
                .is_some()
        {
            return Err(DirectoryReplicaStoreError::Request(
                "descriptor response contains an extra or duplicate object".to_string(),
            ));
        }
    }
    Ok(mapped)
}

fn incident_digest(
    producer: &[u8; 32],
    subject_node_id: &[u8; 32],
    incident: &QuarantineIncident<'_>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"AeroNyx-DirectoryReplicaIncident-v1");
    hasher.update(producer);
    hasher.update(subject_node_id);
    hasher.update((incident.kind.len() as u64).to_le_bytes());
    hasher.update(incident.kind.as_bytes());
    hasher.update(incident.height.to_le_bytes());
    hasher.update(incident.local_hash);
    hasher.update(incident.remote_hash);
    hasher.update((incident.evidence_frame.len() as u64).to_le_bytes());
    hasher.update(incident.evidence_frame);
    hasher.finalize().into()
}

fn encode_observation_checkpoint(
    checkpoint: &DirectoryObservationCheckpointV1,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES)
        .serialize(checkpoint)
        .map_err(|error| {
            DirectoryReplicaStoreError::Codec(format!(
                "encode directory observation checkpoint: {error}"
            ))
        })
}

fn decode_observation_checkpoint(
    bytes: &[u8],
) -> Result<DirectoryObservationCheckpointV1, DirectoryReplicaStoreError> {
    if bytes.is_empty()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX)
            > MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES
    {
        return Err(DirectoryReplicaStoreError::Codec(
            "directory observation checkpoint size is invalid".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .with_limit(MAX_DIRECTORY_OBSERVATION_CHECKPOINT_BYTES)
        .deserialize(bytes)
        .map_err(|error| {
            DirectoryReplicaStoreError::Codec(format!(
                "decode directory observation checkpoint: {error}"
            ))
        })
}

fn encode_block(block: &DirectoryCommitmentBlockV1) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .serialize(block)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn decode_block(bytes: &[u8]) -> Result<DirectoryCommitmentBlockV1, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_DIRECTORY_BLOCK_BYTES) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica block exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn encode_descriptor_object(
    descriptor: &SignedNodeDescriptor,
) -> Result<Vec<u8>, DirectoryReplicaStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .serialize(descriptor)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn decode_descriptor_object(
    bytes: &[u8],
) -> Result<SignedNodeDescriptor, DirectoryReplicaStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| {
        length > MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES
    }) {
        return Err(DirectoryReplicaStoreError::Codec(
            "replica descriptor object exceeds its byte limit".to_string(),
        ));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryReplicaStoreError::Codec(error.to_string()))
}

fn validate_retry_state_fields(
    producer: &[u8; 32],
    local_node_id: &[u8; 32],
    consecutive_failures: u64,
    retry_not_before: Option<u64>,
    last_failure_at: u64,
    last_failure_reason: &str,
) -> Result<(), &'static str> {
    if *producer == [0u8; 32] || producer == local_node_id {
        return Err("replica retry producer is invalid");
    }
    if consecutive_failures == 0
        || consecutive_failures > DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
    {
        return Err("replica retry failure count is invalid");
    }
    if last_failure_at == 0 {
        return Err("replica retry failure timestamp is invalid");
    }
    if retry_not_before.is_some_and(|retry_at| {
        retry_at < last_failure_at
            || retry_at.saturating_sub(last_failure_at) > DIRECTORY_REPLICA_FAILURE_BACKOFF_MAX_SECS
    }) {
        return Err("replica retry boundary is invalid");
    }
    if last_failure_reason.is_empty()
        || last_failure_reason.len() > MAX_DIRECTORY_REPLICA_FAILURE_REASON_BYTES
        || !last_failure_reason
            .bytes()
            .all(|value| value.is_ascii_lowercase() || value.is_ascii_digit() || value == b'_')
    {
        return Err("replica retry failure reason is invalid");
    }
    Ok(())
}

fn bytes32(bytes: &[u8], field: &str) -> Result<[u8; 32], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 32 bytes"))
    })
}

fn bytes16(bytes: &[u8], field: &str) -> Result<[u8; 16], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 16 bytes"))
    })
}

fn bytes64(bytes: &[u8], field: &str) -> Result<[u8; 64], DirectoryReplicaStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} must contain exactly 64 bytes"))
    })
}

fn u64_to_i64(value: u64, field: &str) -> Result<i64, DirectoryReplicaStoreError> {
    i64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} exceeds SQLite integer range"))
    })
}

fn positive_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryReplicaStoreError> {
    if value <= 0 {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{field} must be positive"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

fn nonnegative_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryReplicaStoreError> {
    if value < 0 {
        return Err(DirectoryReplicaStoreError::Integrity(format!(
            "{field} must not be negative"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryReplicaStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::DirectoryChainStore;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::protocol::discovery::{
        directory_block_range_response_signing_bytes, encode_directory_sync_message, NodeDescriptor,
    };
    use tempfile::TempDir;

    const NOW: u64 = 1_700_000_100;

    fn descriptor(identity: &IdentityKeyPair, sequence: u64) -> SignedNodeDescriptor {
        SignedNodeDescriptor::sign(
            NodeDescriptor::new(
                identity.public_key_bytes(),
                sequence,
                NOW - 10,
                NOW + 3_600,
                "replica-test",
            ),
            identity,
        )
        .unwrap()
    }

    fn block(
        producer: &IdentityKeyPair,
        height: u64,
        previous: [u8; 32],
        object: &SignedNodeDescriptor,
    ) -> DirectoryCommitmentBlockV1 {
        DirectoryCommitmentBlockV1::new_signed(
            height,
            NOW + height,
            previous,
            vec![DirectoryDescriptorCommitmentV1::from_signed_descriptor(object).unwrap()],
            producer,
        )
        .unwrap()
    }

    fn response_frame(
        producer: &IdentityKeyPair,
        blocks: Vec<DirectoryCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let responder = producer.public_key_bytes();
        let signing = directory_block_range_response_signing_bytes(
            &request_id,
            &responder,
            NOW + 20,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_directory_sync_message(&DirectorySyncMessage::BlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            responder,
            response_timestamp: NOW + 20,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: producer.sign(&signing),
        })
        .unwrap()
    }

    fn carrier_response_frame(
        producer: &IdentityKeyPair,
        carrier: &IdentityKeyPair,
        blocks: Vec<DirectoryCommitmentBlockV1>,
        has_more: bool,
        tip_height: u64,
        tip_hash: [u8; 32],
        request_id: [u8; 16],
    ) -> Vec<u8> {
        let producer_id = producer.public_key_bytes();
        let carrier_id = carrier.public_key_bytes();
        let signing = directory_replica_block_range_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &producer_id,
            &carrier_id,
            NOW + 20,
            &blocks,
            has_more,
            tip_height,
            &tip_hash,
        );
        encode_directory_sync_message(&DirectorySyncMessage::ReplicaBlockRangeResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            producer: producer_id,
            carrier: carrier_id,
            response_timestamp: NOW + 20,
            blocks,
            has_more,
            tip_height,
            tip_hash,
            signature: carrier.sign(&signing),
        })
        .unwrap()
    }

    fn accepted_observation_witness_response(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        checkpoint: &DirectoryObservationCheckpointV1,
        request_seed: u8,
    ) -> DirectorySyncMessage {
        let request_id = [request_seed; 16];
        let checkpoint_hash = checkpoint.hash();
        let responder = witness.public_key_bytes();
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &responder,
            NOW + 22,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder,
            response_timestamp: NOW + 22,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        }
    }

    fn policy_anchor_request(
        observer: &IdentityKeyPair,
        anchor: DirectoryObservationWitnessPolicyAnchor,
        request_seed: u8,
    ) -> DirectorySyncMessage {
        let request_id = [request_seed; 16];
        let requester = observer.public_key_bytes();
        let signing_bytes = directory_policy_anchor_request_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &requester,
            NOW + 30,
            anchor.epoch,
            &anchor.previous_policy_digest,
            &anchor.policy_digest,
        );
        DirectorySyncMessage::ObservationWitnessPolicyAnchorRequestV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            requester,
            request_timestamp: NOW + 30,
            policy_epoch: anchor.epoch,
            previous_policy_digest: anchor.previous_policy_digest,
            policy_digest: anchor.policy_digest,
            signature: observer.sign(&signing_bytes),
        }
    }

    fn accepted_policy_anchor_response(
        observer: &IdentityKeyPair,
        witness: &IdentityKeyPair,
        anchor: DirectoryObservationWitnessPolicyAnchor,
        request_seed: u8,
    ) -> DirectorySyncMessage {
        let request_id = [request_seed; 16];
        let responder = witness.public_key_bytes();
        let signing_bytes = directory_policy_anchor_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            anchor.epoch,
            &anchor.policy_digest,
            &responder,
            NOW + 31,
            DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
        );
        DirectorySyncMessage::ObservationWitnessPolicyAnchorResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            policy_epoch: anchor.epoch,
            policy_digest: anchor.policy_digest,
            responder,
            response_timestamp: NOW + 31,
            outcome: DIRECTORY_POLICY_ANCHOR_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        }
    }

    fn import_replica_block(
        store: &DirectoryReplicaStore,
        producer: &IdentityKeyPair,
        object: &SignedNodeDescriptor,
        replica_block: &DirectoryCommitmentBlockV1,
        request_id: [u8; 16],
    ) {
        let frame = response_frame(
            producer,
            vec![replica_block.clone()],
            false,
            replica_block.header.height,
            replica_block.hash(),
            request_id,
        );
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(replica_block),
                std::slice::from_ref(object),
                replica_block.header.height,
                replica_block.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
    }

    fn resolution_command(
        resolver: &IdentityKeyPair,
        incident_digest: [u8; 32],
        tip: &DirectoryReplicaTip,
        command_id: [u8; 16],
        resolved_at: u64,
    ) -> DirectoryReplicaResolutionCommand {
        DirectoryReplicaResolutionCommand::sign(
            resolver,
            command_id,
            incident_digest,
            tip.producer,
            tip.tip_height,
            tip.tip_hash,
            tip.quarantine_kind.clone().unwrap(),
            tip.last_resolution_digest,
            resolved_at,
        )
        .unwrap()
    }

    #[test]
    fn full_node_mirror_registry_is_bounded_and_promotes_only_by_operator_pin() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x01; 32]).unwrap();
        let mirror_a = IdentityKeyPair::from_bytes(&[0x02; 32]).unwrap();
        let mirror_b = IdentityKeyPair::from_bytes(&[0x03; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x04; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&mirror_a, 1, [0u8; 32], &object);
        let block_b = block(&mirror_b, 1, [0u8; 32], &object);
        let frame_a = response_frame(
            &mirror_a,
            vec![block_a.clone()],
            false,
            1,
            block_a.hash(),
            [0x05; 16],
        );
        let frame_b = response_frame(
            &mirror_b,
            vec![block_b.clone()],
            false,
            1,
            block_b.hash(),
            [0x06; 16],
        );
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();

        store
            .import_verified_mirror_page(
                mirror_a.public_key_bytes(),
                7,
                1,
                std::slice::from_ref(&block_a),
                std::slice::from_ref(&object),
                1,
                block_a.hash(),
                &frame_a,
                NOW + 20,
            )
            .unwrap();
        assert_eq!(store.mirror_producer_ids().unwrap(), vec![mirror_a.public_key_bytes()]);
        assert_eq!(store.status_snapshot().unwrap().mirror_producers, 1);
        assert!(matches!(
            store.import_verified_mirror_page(
                mirror_b.public_key_bytes(),
                8,
                1,
                std::slice::from_ref(&block_b),
                std::slice::from_ref(&object),
                1,
                block_b.hash(),
                &frame_b,
                NOW + 20,
            ),
            Err(DirectoryReplicaStoreError::MirrorCapacity)
        ));
        assert_eq!(store.producer_tip(&mirror_b.public_key_bytes()).unwrap().tip_height, 0);
        store
            .import_verified_mirror_page(
                mirror_b.public_key_bytes(),
                8,
                2,
                std::slice::from_ref(&block_b),
                std::slice::from_ref(&object),
                1,
                block_b.hash(),
                &frame_b,
                NOW + 20,
            )
            .unwrap();
        assert!(matches!(
            store.ensure_mirror_capacity(1),
            Err(DirectoryReplicaStoreError::MirrorCapacity)
        ));
        store.ensure_mirror_capacity(2).unwrap();

        assert_eq!(
            store
                .promote_pinned_producers(&[
                    mirror_a.public_key_bytes(),
                    mirror_b.public_key_bytes(),
                ])
                .unwrap(),
            2
        );
        assert!(store.mirror_producer_ids().unwrap().is_empty());
        assert_eq!(store.audit(NOW + 21).unwrap().mirror_producers, 0);
        assert!(matches!(
            store.import_verified_mirror_page(
                mirror_a.public_key_bytes(),
                9,
                1,
                std::slice::from_ref(&block_a),
                std::slice::from_ref(&object),
                1,
                block_a.hash(),
                &frame_a,
                NOW + 20,
            ),
            Err(DirectoryReplicaStoreError::Request(_))
        ));
    }

    #[test]
    fn schema_v8_is_atomically_migrated_to_v9_mirror_registry() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x07; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        drop(store);
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("DROP TABLE directory_replica_mirror_producers;")
            .unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V8],
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(audit.mirror_producers, 0);
        let connection = store.connection.lock();
        let version: i64 = connection
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
    }

    #[test]
    fn producer_replicas_are_isolated_idempotent_and_reopen_cleanly() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let first = block(&producer, 1, [0u8; 32], &object);
        let frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x41; 16],
        );
        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        assert_eq!(audit, DirectoryReplicaAudit::default());
        let imported = store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert_eq!(imported.blocks_inserted, 1);
        let repeated = store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert_eq!(repeated.blocks_already_present, 1);
        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(snapshot.producers, 1);
        assert_eq!(snapshot.quarantined_producers, 0);
        assert_eq!(snapshot.blocks, 1);
        assert_eq!(snapshot.commitments, 1);
        assert_eq!(snapshot.incidents, 0);
        assert_eq!(snapshot.producer_snapshots.len(), 1);
        assert_eq!(
            snapshot.producer_snapshots[0].producer,
            producer.public_key_bytes()
        );
        assert_eq!(snapshot.producer_snapshots[0].tip_height, 1);
        drop(store);
        let (_, reopened) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(reopened.producers, 1);
        assert_eq!(reopened.blocks, 1);
        assert_eq!(reopened.commitments, 1);
    }

    #[test]
    fn audited_carrier_export_is_bounded_exact_and_importable() {
        let source_temp = TempDir::new().unwrap();
        let receiver_temp = TempDir::new().unwrap();
        let carrier = IdentityKeyPair::from_bytes(&[0x12; 32]).unwrap();
        let receiver = IdentityKeyPair::from_bytes(&[0x13; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x14; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x15; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let replica_block = block(&producer, 1, [0u8; 32], &object);
        let (source, _) = DirectoryReplicaStore::open(
            source_temp.path().join("directory.db"),
            carrier.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        import_replica_block(&source, &producer, &object, &replica_block, [0x16; 16]);

        let page = source
            .audited_evidence_page(&producer.public_key_bytes(), 1, 1, NOW + 21)
            .unwrap();
        assert_eq!(page.blocks, vec![replica_block.clone()]);
        assert_eq!(page.tip_height, 1);
        assert_eq!(page.tip_hash, replica_block.hash());
        let descriptor_hash = replica_block.commitments[0].descriptor_hash;
        assert_eq!(
            source
                .audited_evidence_descriptor_objects(
                    &producer.public_key_bytes(),
                    &[descriptor_hash],
                    NOW + 21,
                )
                .unwrap(),
            Some(vec![object.clone()])
        );
        assert!(source
            .audited_evidence_descriptor_objects(
                &producer.public_key_bytes(),
                &[[0x17; 32]],
                NOW + 21,
            )
            .unwrap()
            .is_none());

        let frame = carrier_response_frame(
            &producer,
            &carrier,
            page.blocks.clone(),
            false,
            page.tip_height,
            page.tip_hash,
            [0x18; 16],
        );
        let (destination, _) = DirectoryReplicaStore::open(
            receiver_temp.path().join("directory.db"),
            receiver.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        let imported = destination
            .import_verified_page(
                producer.public_key_bytes(),
                &page.blocks,
                &[object],
                page.tip_height,
                page.tip_hash,
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert_eq!(imported.blocks_inserted, 1);

        let mut tampered = frame;
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        assert!(
            verify_incident_response_evidence(&tampered, &producer.public_key_bytes()).is_err()
        );
    }

    #[test]
    fn recent_observation_convergence_is_multi_source_and_order_independent() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x21; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x23; 32]).unwrap();
        let pending = IdentityKeyPair::from_bytes(&[0x24; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x25; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let frame_a = response_frame(
            &producer_a,
            vec![block_a.clone()],
            false,
            1,
            block_a.hash(),
            [0x26; 16],
        );
        let frame_b = response_frame(
            &producer_b,
            vec![block_b.clone()],
            false,
            1,
            block_b.hash(),
            [0x27; 16],
        );
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        for (producer, replica_block, frame) in [
            (&producer_a, &block_a, &frame_a),
            (&producer_b, &block_b, &frame_b),
        ] {
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(replica_block),
                    std::slice::from_ref(&object),
                    1,
                    replica_block.hash(),
                    frame,
                    NOW + 20,
                )
                .unwrap();
        }

        let configured = [
            producer_a.public_key_bytes(),
            producer_b.public_key_bytes(),
            pending.public_key_bytes(),
        ];
        let snapshot = store.observation_convergence(&configured).unwrap();
        assert_eq!(snapshot.configured_producers, 3);
        assert_eq!(snapshot.eligible_producers, 2);
        assert_eq!(snapshot.pending_producers, 1);
        assert_eq!(snapshot.excluded_quarantined_producers, 0);
        assert_eq!(snapshot.window_blocks, 32);
        assert_eq!(snapshot.recent_commitments, 2);
        assert_eq!(snapshot.distinct_recent_commitments, 1);
        assert_eq!(snapshot.multi_source_recent_commitments, 1);
        assert_eq!(snapshot.all_eligible_source_recent_commitments, 1);
        assert!(snapshot.observation_root.is_some());

        let reversed = [configured[2], configured[1], configured[0]];
        assert_eq!(store.observation_convergence(&reversed).unwrap(), snapshot);
        assert!(store
            .observation_convergence(&[configured[0], configured[0]])
            .is_err());
        assert!(store
            .append_observation_checkpoint(&configured, &local, NOW + 21)
            .is_err());
    }

    #[test]
    fn observation_checkpoints_are_signed_linked_recomputed_and_idempotent() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x54; 32]).unwrap();
        let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let first_object = descriptor(&subject, 1);
        let first_a = block(&producer_a, 1, [0u8; 32], &first_object);
        let first_b = block(&producer_b, 1, [0u8; 32], &first_object);
        let first_frame_a = response_frame(
            &producer_a,
            vec![first_a.clone()],
            false,
            1,
            first_a.hash(),
            [0x55; 16],
        );
        let first_frame_b = response_frame(
            &producer_b,
            vec![first_b.clone()],
            false,
            1,
            first_b.hash(),
            [0x56; 16],
        );
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        for (producer, replica_block, frame) in [
            (&producer_a, &first_a, &first_frame_a),
            (&producer_b, &first_b, &first_frame_b),
        ] {
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(replica_block),
                    std::slice::from_ref(&first_object),
                    1,
                    replica_block.hash(),
                    frame,
                    NOW + 20,
                )
                .unwrap();
        }

        let first = store
            .append_observation_checkpoint(&configured, &local, NOW + 21)
            .unwrap();
        assert!(first.appended);
        assert_eq!(first.sequence, 1);
        assert_eq!(first.producer_count, 2);
        let unchanged = store
            .append_observation_checkpoint(&configured, &local, NOW + 22)
            .unwrap();
        assert!(!unchanged.appended);
        assert_eq!(
            unchanged,
            DirectoryObservationCheckpointAppendReport {
                appended: false,
                ..first
            }
        );
        assert!(store
            .append_observation_checkpoint(
                &configured,
                &IdentityKeyPair::from_bytes(&[0x57; 32]).unwrap(),
                NOW + 22,
            )
            .is_err());

        let second_object = descriptor(&subject, 2);
        let second_a = block(&producer_a, 2, first_a.hash(), &second_object);
        let second_b = block(&producer_b, 2, first_b.hash(), &second_object);
        let second_frame_a = response_frame(
            &producer_a,
            vec![second_a.clone()],
            false,
            2,
            second_a.hash(),
            [0x58; 16],
        );
        let second_frame_b = response_frame(
            &producer_b,
            vec![second_b.clone()],
            false,
            2,
            second_b.hash(),
            [0x59; 16],
        );
        for (producer, replica_block, frame) in [
            (&producer_a, &second_a, &second_frame_a),
            (&producer_b, &second_b, &second_frame_b),
        ] {
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(replica_block),
                    std::slice::from_ref(&second_object),
                    2,
                    replica_block.hash(),
                    frame,
                    NOW + 23,
                )
                .unwrap();
        }
        let second = store
            .append_observation_checkpoint(&configured, &local, NOW + 24)
            .unwrap();
        assert!(second.appended);
        assert_eq!(second.sequence, 2);
        assert_ne!(second.checkpoint_hash, first.checkpoint_hash);
        assert_eq!(
            store
                .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 23, NOW + 25)
                .unwrap()
                .unwrap()
                .sequence,
            1
        );
        assert_eq!(
            store
                .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 24, NOW + 25)
                .unwrap()
                .unwrap()
                .sequence,
            2
        );
        assert!(store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 26, NOW + 25)
            .is_err());
        store
            .persist_observation_witness_outcome_round(
                2,
                NOW + 25,
                &[DirectoryObservationWitnessOutcome::EvidenceUnavailable],
            )
            .unwrap();
        assert!(store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 23, NOW + 26)
            .unwrap()
            .is_none());
        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(snapshot.observation_checkpoints, 2);
        assert_eq!(snapshot.observation_checkpoint_sequence, 2);
        assert_eq!(snapshot.observation_checkpoint_hash, second.checkpoint_hash);
        let audit = store.audit(NOW + 26).unwrap();
        assert_eq!(audit.observation_checkpoints, 2);
        assert_eq!(audit.observation_checkpoint_sequence, 2);
        assert_eq!(audit.observation_checkpoint_hash, second.checkpoint_hash);
        drop(store);

        let (reopened_store, reopened) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 27).unwrap();
        assert_eq!(reopened.observation_checkpoints, 2);
        assert_eq!(reopened.observation_checkpoint_sequence, 2);
        {
            let connection = reopened_store.connection.lock();
            connection
                .execute(
                    "UPDATE directory_observation_checkpoints
                     SET observed_at = observed_at + 1 WHERE sequence = 1",
                    [],
                )
                .unwrap();
        }
        assert!(reopened_store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 24, NOW + 28)
            .is_err());
    }

    #[test]
    fn observation_witness_requires_independent_evidence_and_survives_restart() {
        let observer_temp = TempDir::new().unwrap();
        let witness_temp = TempDir::new().unwrap();
        let empty_witness_temp = TempDir::new().unwrap();
        let observer_path = observer_temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let empty_witness = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x85; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x86; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let (observer_store, _) =
            DirectoryReplicaStore::open(&observer_path, observer.public_key_bytes(), NOW + 20)
                .unwrap();
        let (witness_store, _) = DirectoryReplicaStore::open(
            witness_temp.path().join("directory.db"),
            witness.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        for store in [&observer_store, &witness_store] {
            import_replica_block(store, &producer_a, &object, &block_a, [0x87; 16]);
            import_replica_block(store, &producer_b, &object, &block_b, [0x88; 16]);
        }
        observer_store
            .append_observation_checkpoint(&configured, &observer, NOW + 21)
            .unwrap();
        let checkpoint = observer_store
            .latest_audited_observation_checkpoint(NOW + 22)
            .unwrap()
            .unwrap();
        assert_eq!(
            witness_store
                .evaluate_observation_checkpoint_witness(&checkpoint, NOW + 22)
                .unwrap(),
            DirectoryObservationWitnessDecision::Accepted
        );

        let (empty_store, _) = DirectoryReplicaStore::open(
            empty_witness_temp.path().join("directory.db"),
            empty_witness.public_key_bytes(),
            NOW + 22,
        )
        .unwrap();
        assert_eq!(
            empty_store
                .evaluate_observation_checkpoint_witness(&checkpoint, NOW + 22)
                .unwrap(),
            DirectoryObservationWitnessDecision::EvidenceUnavailable
        );
        let conflicting = DirectoryObservationCheckpointV1::new_signed(
            checkpoint.sequence,
            checkpoint.observed_at,
            checkpoint.previous_checkpoint_hash,
            checkpoint.configured_producer_count,
            checkpoint.producer_tips.clone(),
            [0xf1; 32],
            &observer,
        )
        .unwrap();
        assert_eq!(
            witness_store
                .evaluate_observation_checkpoint_witness(&conflicting, NOW + 22)
                .unwrap(),
            DirectoryObservationWitnessDecision::EvidenceConflict
        );

        let request_id = [0x89; 16];
        let checkpoint_hash = checkpoint.hash();
        let response_timestamp = NOW + 22;
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            response_timestamp,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        };
        assert!(observer_store
            .persist_observation_checkpoint_witness(&response, NOW + 22)
            .unwrap());
        assert!(!observer_store
            .persist_observation_checkpoint_witness(&response, NOW + 22)
            .unwrap());
        assert!(observer_store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 22)
            .unwrap()
            .is_none());
        let first_outcomes = [
            DirectoryObservationWitnessOutcome::Accepted,
            DirectoryObservationWitnessOutcome::EvidenceUnavailable,
        ];
        let first_outcome_snapshot = observer_store
            .persist_observation_witness_outcome_round(1, NOW + 22, &first_outcomes)
            .unwrap();
        assert_eq!(first_outcome_snapshot.rounds, 1);
        assert_eq!(first_outcome_snapshot.totals.attempts(), 2);
        assert_eq!(first_outcome_snapshot.totals.accepted, 1);
        assert_eq!(first_outcome_snapshot.totals.evidence_unavailable, 1);
        let second_outcomes = [
            DirectoryObservationWitnessOutcome::EvidenceConflict,
            DirectoryObservationWitnessOutcome::TransportFailure,
        ];
        let second_outcome_snapshot = observer_store
            .persist_observation_witness_outcome_round(1, NOW + 23, &second_outcomes)
            .unwrap();
        assert_eq!(second_outcome_snapshot.rounds, 2);
        assert_eq!(second_outcome_snapshot.totals.attempts(), 4);
        assert_eq!(second_outcome_snapshot.totals.evidence_conflict, 1);
        assert_eq!(second_outcome_snapshot.totals.transport_failures, 1);
        assert_eq!(second_outcome_snapshot.last_round, {
            let mut expected = DirectoryObservationWitnessOutcomeCounters::default();
            expected.record(DirectoryObservationWitnessOutcome::EvidenceConflict);
            expected.record(DirectoryObservationWitnessOutcome::TransportFailure);
            expected
        });
        let snapshot = observer_store.status_snapshot().unwrap();
        assert_eq!(snapshot.observation_checkpoint_witnesses, 1);
        assert_eq!(snapshot.observation_checkpoint_witnessed_sequence, 1);
        assert_eq!(snapshot.observation_checkpoint_latest_witnesses, 1);
        assert_eq!(
            snapshot.observation_witness_outcomes,
            second_outcome_snapshot
        );
        let audit = observer_store.audit(NOW + 24).unwrap();
        assert_eq!(audit.observation_checkpoint_witnesses, 1);
        assert_eq!(audit.observation_checkpoint_witnessed_sequence, 1);
        assert_eq!(audit.observation_checkpoint_latest_witnesses, 1);
        assert_eq!(audit.observation_witness_outcomes, second_outcome_snapshot);
        drop(observer_store);

        let (_, reopened) =
            DirectoryReplicaStore::open(&observer_path, observer.public_key_bytes(), NOW + 25)
                .unwrap();
        assert_eq!(reopened.observation_checkpoint_witnesses, 1);
        assert_eq!(reopened.observation_checkpoint_witnessed_sequence, 1);
        assert_eq!(
            reopened.observation_witness_outcomes,
            second_outcome_snapshot
        );
    }

    #[test]
    fn observation_witness_set_is_bounded_at_write_and_audit() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xa8; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0xa9; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xaa; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0xab; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        import_replica_block(&store, &producer_a, &object, &block_a, [0xac; 16]);
        import_replica_block(&store, &producer_b, &object, &block_b, [0xad; 16]);
        store
            .append_observation_checkpoint(&configured, &observer, NOW + 21)
            .unwrap();
        let checkpoint = store
            .latest_audited_observation_checkpoint(NOW + 22)
            .unwrap()
            .unwrap();
        let checkpoint_hash = checkpoint.hash();

        for index in 0..=MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
            let seed = 0xb0u8.saturating_add(u8::try_from(index).unwrap());
            let witness = IdentityKeyPair::from_bytes(&[seed; 32]).unwrap();
            let request_id = [seed; 16];
            let signing_bytes = directory_observation_witness_response_signing_bytes(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                &request_id,
                &observer.public_key_bytes(),
                checkpoint.sequence,
                &checkpoint_hash,
                &witness.public_key_bytes(),
                NOW + 22,
                DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            );
            let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
                chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                request_id,
                observer: observer.public_key_bytes(),
                checkpoint_sequence: checkpoint.sequence,
                checkpoint_hash,
                responder: witness.public_key_bytes(),
                response_timestamp: NOW + 22,
                outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
                signature: witness.sign(&signing_bytes),
            };
            let result = store.persist_observation_checkpoint_witness(&response, NOW + 22);
            if index < MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1 {
                assert!(result.unwrap());
            } else {
                assert!(result.is_err());
            }
        }

        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(
            snapshot.observation_checkpoint_witnesses,
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).unwrap()
        );
        assert_eq!(
            snapshot.observation_checkpoint_latest_witnesses,
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).unwrap()
        );
        assert!(store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 23)
            .unwrap()
            .is_none());
        assert_eq!(
            store
                .audit(NOW + 23)
                .unwrap()
                .observation_checkpoint_witnesses,
            u64::try_from(MAX_DIRECTORY_OBSERVATION_PRODUCERS_V1).unwrap()
        );
    }

    #[test]
    fn observation_witness_target_counts_only_current_pins_and_survives_restart() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xc1; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0xc2; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xc3; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0xc4; 32]).unwrap();
        let witness_a = IdentityKeyPair::from_bytes(&[0xc5; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0xc6; 32]).unwrap();
        let retired_witness = IdentityKeyPair::from_bytes(&[0xc7; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let configured_producers = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let eligible_witnesses = [witness_a.public_key_bytes(), witness_b.public_key_bytes()];
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        import_replica_block(&store, &producer_a, &object, &block_a, [0xc8; 16]);
        import_replica_block(&store, &producer_b, &object, &block_b, [0xc9; 16]);
        store
            .append_observation_checkpoint(&configured_producers, &observer, NOW + 21)
            .unwrap();
        let checkpoint = store
            .latest_audited_observation_checkpoint(NOW + 22)
            .unwrap()
            .unwrap();

        let initial = store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 21,
                NOW + 22,
                2,
                &eligible_witnesses,
            )
            .unwrap()
            .unwrap();
        assert_eq!(initial.checkpoint, checkpoint);
        assert!(initial.witnessed_by.is_empty());
        assert_eq!(initial.minimum_witnesses, 2);

        let newer_object = descriptor(&subject, 2);
        let newer_block_a = block(&producer_a, 2, block_a.hash(), &newer_object);
        let newer_block_b = block(&producer_b, 2, block_b.hash(), &newer_object);
        import_replica_block(
            &store,
            &producer_a,
            &newer_object,
            &newer_block_a,
            [0xcd; 16],
        );
        import_replica_block(
            &store,
            &producer_b,
            &newer_object,
            &newer_block_b,
            [0xce; 16],
        );
        let newer_checkpoint = store
            .append_observation_checkpoint(&configured_producers, &observer, NOW + 23)
            .unwrap();
        assert!(newer_checkpoint.appended);
        assert_eq!(newer_checkpoint.sequence, checkpoint.sequence + 1);

        let retired_response =
            accepted_observation_witness_response(&observer, &retired_witness, &checkpoint, 0xca);
        assert!(store
            .persist_observation_checkpoint_witness(&retired_response, NOW + 22)
            .unwrap());
        let after_retired = store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 24,
                2,
                &eligible_witnesses,
            )
            .unwrap()
            .unwrap();
        assert_eq!(after_retired.checkpoint.sequence, checkpoint.sequence);
        assert!(after_retired.witnessed_by.is_empty());
        assert_eq!(
            store
                .verified_observation_witness_count_for_pins(
                    checkpoint.sequence,
                    &eligible_witnesses,
                    NOW + 24,
                )
                .unwrap(),
            0
        );

        let witness_a_response =
            accepted_observation_witness_response(&observer, &witness_a, &checkpoint, 0xcb);
        assert!(store
            .persist_observation_checkpoint_witness(&witness_a_response, NOW + 22)
            .unwrap());
        let partial = store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 24,
                2,
                &eligible_witnesses,
            )
            .unwrap()
            .unwrap();
        assert_eq!(partial.checkpoint.sequence, checkpoint.sequence);
        assert_eq!(partial.witnessed_by, vec![witness_a.public_key_bytes()]);
        assert_eq!(
            store
                .verified_observation_witness_count_for_pins(
                    checkpoint.sequence,
                    &eligible_witnesses,
                    NOW + 24,
                )
                .unwrap(),
            1
        );
        assert!(store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 21,
                NOW + 23,
                1,
                &[witness_a.public_key_bytes()],
            )
            .unwrap()
            .is_none());
        let witness_b_response =
            accepted_observation_witness_response(&observer, &witness_b, &checkpoint, 0xcc);
        assert!(store
            .persist_observation_checkpoint_witness(&witness_b_response, NOW + 22)
            .unwrap());
        assert!(store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 21,
                NOW + 24,
                2,
                &eligible_witnesses,
            )
            .unwrap()
            .is_none());
        assert_eq!(
            store
                .verified_observation_witness_count_for_pins(
                    checkpoint.sequence,
                    &eligible_witnesses,
                    NOW + 24,
                )
                .unwrap(),
            2
        );
        assert!(store
            .verified_observation_witness_count_for_pins(
                checkpoint.sequence,
                &[witness_a.public_key_bytes(), witness_a.public_key_bytes()],
                NOW + 24,
            )
            .is_err());
        let next = store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 24,
                2,
                &eligible_witnesses,
            )
            .unwrap()
            .unwrap();
        assert_eq!(next.checkpoint.sequence, newer_checkpoint.sequence);
        assert!(next.witnessed_by.is_empty());
        assert!(store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 21,
                NOW + 23,
                0,
                &eligible_witnesses,
            )
            .is_err());
        assert!(store
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 21,
                NOW + 23,
                2,
                &[witness_a.public_key_bytes(), witness_a.public_key_bytes()],
            )
            .is_err());
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 24).unwrap();
        assert_eq!(audit.observation_checkpoint_witnesses, 3);
        assert!(reopened
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 21,
                NOW + 24,
                2,
                &eligible_witnesses,
            )
            .unwrap()
            .is_none());
    }

    #[test]
    fn witness_failure_drill_keeps_unsatisfied_floor_across_restart_and_pin_rotation() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xd1; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0xd2; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0xd3; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0xd4; 32]).unwrap();
        let witness_a = IdentityKeyPair::from_bytes(&[0xd5; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0xd6; 32]).unwrap();
        let witness_c = IdentityKeyPair::from_bytes(&[0xd7; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let configured_producers = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let original_pins = [witness_a.public_key_bytes(), witness_b.public_key_bytes()];
        let rotated_pins = [witness_b.public_key_bytes(), witness_c.public_key_bytes()];
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        import_replica_block(&store, &producer_a, &object, &block_a, [0xd8; 16]);
        import_replica_block(&store, &producer_b, &object, &block_b, [0xd9; 16]);
        store
            .append_observation_checkpoint(&configured_producers, &observer, NOW + 21)
            .unwrap();
        let first_checkpoint = store
            .latest_audited_observation_checkpoint(NOW + 22)
            .unwrap()
            .unwrap();

        let newer_object = descriptor(&subject, 2);
        let newer_block_a = block(&producer_a, 2, block_a.hash(), &newer_object);
        let newer_block_b = block(&producer_b, 2, block_b.hash(), &newer_object);
        import_replica_block(
            &store,
            &producer_a,
            &newer_object,
            &newer_block_a,
            [0xda; 16],
        );
        import_replica_block(
            &store,
            &producer_b,
            &newer_object,
            &newer_block_b,
            [0xdb; 16],
        );
        let second_checkpoint = store
            .append_observation_checkpoint(&configured_producers, &observer, NOW + 23)
            .unwrap();
        assert!(second_checkpoint.appended);

        let witness_a_response =
            accepted_observation_witness_response(&observer, &witness_a, &first_checkpoint, 0xdc);
        assert!(store
            .persist_observation_checkpoint_witness(&witness_a_response, NOW + 24)
            .unwrap());
        store
            .persist_observation_witness_outcome_round(
                first_checkpoint.sequence,
                NOW + 24,
                &[
                    DirectoryObservationWitnessOutcome::Accepted,
                    DirectoryObservationWitnessOutcome::PeerUnavailable,
                ],
            )
            .unwrap();
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 25).unwrap();
        assert_eq!(audit.observation_checkpoint_witnesses, 1);
        let after_restart = reopened
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 25,
                2,
                &original_pins,
            )
            .unwrap()
            .unwrap();
        assert_eq!(after_restart.checkpoint.sequence, first_checkpoint.sequence);
        assert_eq!(
            after_restart.witnessed_by,
            vec![witness_a.public_key_bytes()]
        );

        reopened
            .persist_observation_witness_outcome_round(
                first_checkpoint.sequence,
                NOW + 25,
                &[DirectoryObservationWitnessOutcome::PeerUnavailable],
            )
            .unwrap();
        let still_blocked = reopened
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 25,
                2,
                &original_pins,
            )
            .unwrap()
            .unwrap();
        assert_eq!(still_blocked.checkpoint.sequence, first_checkpoint.sequence);
        assert_eq!(
            still_blocked.witnessed_by,
            vec![witness_a.public_key_bytes()]
        );

        let witness_b_response =
            accepted_observation_witness_response(&observer, &witness_b, &first_checkpoint, 0xdd);
        assert!(reopened
            .persist_observation_checkpoint_witness(&witness_b_response, NOW + 26)
            .unwrap());
        let completed_snapshot = reopened
            .persist_observation_witness_outcome_round(
                first_checkpoint.sequence,
                NOW + 26,
                &[DirectoryObservationWitnessOutcome::Accepted],
            )
            .unwrap();
        assert_eq!(completed_snapshot.rounds, 3);
        assert_eq!(completed_snapshot.totals.accepted, 2);
        assert_eq!(completed_snapshot.totals.peer_unavailable, 2);
        let next_original_target = reopened
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 26,
                2,
                &original_pins,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            next_original_target.checkpoint.sequence,
            second_checkpoint.sequence
        );

        let reopened_by_rotation = reopened
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 26,
                2,
                &rotated_pins,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            reopened_by_rotation.checkpoint.sequence,
            first_checkpoint.sequence
        );
        assert_eq!(
            reopened_by_rotation.witnessed_by,
            vec![witness_b.public_key_bytes()]
        );
        let witness_c_response =
            accepted_observation_witness_response(&observer, &witness_c, &first_checkpoint, 0xde);
        assert!(reopened
            .persist_observation_checkpoint_witness(&witness_c_response, NOW + 26)
            .unwrap());
        assert_eq!(
            reopened
                .verified_observation_witness_count_for_pins(
                    first_checkpoint.sequence,
                    &rotated_pins,
                    NOW + 26,
                )
                .unwrap(),
            2
        );
        let next_rotated_target = reopened
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 26,
                2,
                &rotated_pins,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            next_rotated_target.checkpoint.sequence,
            second_checkpoint.sequence
        );
        drop(reopened);

        let (reopened_again, audit) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 27).unwrap();
        assert_eq!(audit.observation_checkpoint_witnesses, 3);
        let restart_target = reopened_again
            .next_audited_mature_observation_checkpoint_below_witness_threshold(
                NOW + 23,
                NOW + 27,
                2,
                &rotated_pins,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            restart_target.checkpoint.sequence,
            second_checkpoint.sequence
        );
        assert!(restart_target.witnessed_by.is_empty());
    }

    #[test]
    fn witness_policy_epochs_are_canonical_idempotent_and_restart_durable() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xe1; 32]).unwrap();
        let witness_a = IdentityKeyPair::from_bytes(&[0xe2; 32]).unwrap();
        let witness_b = IdentityKeyPair::from_bytes(&[0xe3; 32]).unwrap();
        let witness_c = IdentityKeyPair::from_bytes(&[0xe4; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();

        let first = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_b.public_key_bytes(), witness_a.public_key_bytes()],
                2,
                NOW + 20,
            )
            .unwrap();
        assert!(first.appended);
        assert_eq!(first.epoch, 1);
        assert_eq!(first.witness_members, 2);
        assert_eq!(first.minimum_witnesses, 2);

        let reordered = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_a.public_key_bytes(), witness_b.public_key_bytes()],
                2,
                NOW + 21,
            )
            .unwrap();
        assert!(!reordered.appended);
        assert_eq!(reordered.epoch, 1);
        assert_eq!(reordered.policy_digest, first.policy_digest);
        assert_eq!(reordered.activated_at, first.activated_at);

        let threshold_change = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_a.public_key_bytes(), witness_b.public_key_bytes()],
                1,
                NOW + 22,
            )
            .unwrap();
        assert!(threshold_change.appended);
        assert_eq!(threshold_change.epoch, 2);
        assert_ne!(threshold_change.policy_digest, first.policy_digest);

        let rotation = store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_b.public_key_bytes(), witness_c.public_key_bytes()],
                2,
                NOW + 23,
            )
            .unwrap();
        assert!(rotation.appended);
        assert_eq!(rotation.epoch, 3);
        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(snapshot.observation_witness_policy_epochs, 3);
        assert_eq!(snapshot.observation_witness_policy_epoch, 3);
        assert_eq!(snapshot.observation_witness_policy_members, 2);
        assert_eq!(snapshot.observation_witness_policy_threshold, 2);
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 24).unwrap();
        assert_eq!(audit.observation_witness_policy_epochs, 3);
        assert_eq!(audit.observation_witness_policy_epoch, 3);
        assert_eq!(audit.observation_witness_policy_activated_at, NOW + 23);
        assert_eq!(audit.observation_witness_policy_members, 2);
        assert_eq!(audit.observation_witness_policy_threshold, 2);
        assert!(reopened
            .observation_witness_policy_matches(
                &[witness_b.public_key_bytes(), witness_c.public_key_bytes()],
                2,
            )
            .unwrap());
        assert!(!reopened
            .observation_witness_policy_matches(
                &[witness_a.public_key_bytes(), witness_c.public_key_bytes()],
                2,
            )
            .unwrap());
        let idempotent_after_restart = reopened
            .reconcile_observation_witness_policy(
                &observer,
                &[witness_c.public_key_bytes(), witness_b.public_key_bytes()],
                2,
                NOW + 25,
            )
            .unwrap();
        assert!(!idempotent_after_restart.appended);
        assert_eq!(idempotent_after_restart.epoch, 3);
        assert_eq!(
            idempotent_after_restart.policy_digest,
            rotation.policy_digest
        );
    }

    #[test]
    fn tampered_witness_policy_signature_or_metadata_head_fails_startup_audit() {
        for tamper_head in [false, true] {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("directory.db");
            let observer = IdentityKeyPair::from_bytes(&[0xe5; 32]).unwrap();
            let witness = IdentityKeyPair::from_bytes(&[0xe6; 32]).unwrap();
            let (store, _) =
                DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
            store
                .reconcile_observation_witness_policy(
                    &observer,
                    &[witness.public_key_bytes()],
                    1,
                    NOW + 20,
                )
                .unwrap();
            drop(store);

            let connection = Connection::open(&path).unwrap();
            if tamper_head {
                connection
                    .execute(
                        "UPDATE directory_replica_meta SET witness_policy_head = ?1
                         WHERE singleton = 1",
                        params![[0x99u8; 32].as_slice()],
                    )
                    .unwrap();
            } else {
                connection
                    .execute(
                        "UPDATE directory_observation_witness_policies SET signature = ?1
                         WHERE epoch = 1",
                        params![[0u8; 64].as_slice()],
                    )
                    .unwrap();
            }
            drop(connection);
            assert!(
                DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 21).is_err()
            );
        }
    }

    #[test]
    fn deleted_witness_policy_table_cannot_reset_anchored_history() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xe7; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xe8; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("DROP TABLE directory_observation_witness_policies;")
            .unwrap();
        drop(connection);
        assert!(DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 21).is_err());
    }

    #[test]
    fn remote_policy_anchor_is_monotonic_idempotent_and_restart_durable() {
        let observer_temp = TempDir::new().unwrap();
        let witness_temp = TempDir::new().unwrap();
        let observer = IdentityKeyPair::from_bytes(&[0xf1; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xf2; 32]).unwrap();
        let replacement = IdentityKeyPair::from_bytes(&[0xf3; 32]).unwrap();
        let (observer_store, _) = DirectoryReplicaStore::open(
            observer_temp.path().join("observer.db"),
            observer.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        observer_store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        let first = observer_store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        let witness_path = witness_temp.path().join("witness.db");
        let (witness_store, _) =
            DirectoryReplicaStore::open(&witness_path, witness.public_key_bytes(), NOW + 20)
                .unwrap();
        let first_request = policy_anchor_request(&observer, first, 0xa1);
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(&first_request, NOW + 30)
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Accepted
        );
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(&first_request, NOW + 30)
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Accepted
        );

        let conflicting = DirectoryObservationWitnessPolicyAnchor {
            policy_digest: [0x44; 32],
            ..first
        };
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(
                    &policy_anchor_request(&observer, conflicting, 0xa2),
                    NOW + 30,
                )
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Conflict
        );
        let gap = DirectoryObservationWitnessPolicyAnchor {
            epoch: 3,
            previous_policy_digest: [0x45; 32],
            policy_digest: [0x46; 32],
        };
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(
                    &policy_anchor_request(&observer, gap, 0xa3),
                    NOW + 30,
                )
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::HistoryGap
        );

        observer_store
            .reconcile_observation_witness_policy(
                &observer,
                &[replacement.public_key_bytes()],
                1,
                NOW + 21,
            )
            .unwrap();
        let second = observer_store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        assert_eq!(second.epoch, 2);
        assert_eq!(second.previous_policy_digest, first.policy_digest);
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(
                    &policy_anchor_request(&observer, second, 0xa4),
                    NOW + 30,
                )
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Accepted
        );
        assert_eq!(
            witness_store
                .persist_remote_observation_witness_policy_anchor(&first_request, NOW + 30)
                .unwrap(),
            DirectoryObservationWitnessPolicyAnchorDecision::Rollback
        );
        drop(witness_store);
        let (_, audit) =
            DirectoryReplicaStore::open(&witness_path, witness.public_key_bytes(), NOW + 32)
                .unwrap();
        assert_eq!(audit.observation_witness_remote_policy_anchors, 2);
    }

    #[test]
    fn policy_anchor_receipts_are_pinned_signed_and_tamper_evident() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xf4; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xf5; 32]).unwrap();
        let outsider = IdentityKeyPair::from_bytes(&[0xf6; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_observation_witness_policy(
                &observer,
                &[witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        let anchor = store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        let response = accepted_policy_anchor_response(&observer, &witness, anchor, 0xb1);
        assert!(store
            .persist_observation_witness_policy_anchor_receipt(&response, NOW + 31)
            .unwrap());
        assert!(!store
            .persist_observation_witness_policy_anchor_receipt(&response, NOW + 31)
            .unwrap());
        assert_eq!(
            store
                .verified_observation_witness_policy_anchor_count_for_pins(
                    anchor.epoch,
                    &anchor.policy_digest,
                    &[witness.public_key_bytes()],
                    NOW + 31,
                )
                .unwrap(),
            1
        );
        assert!(store
            .persist_observation_witness_policy_anchor_receipt(
                &accepted_policy_anchor_response(&observer, &outsider, anchor, 0xb2),
                NOW + 31,
            )
            .is_err());
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_observation_policy_anchor_receipts
                 SET response_blob = zeroblob(length(response_blob))",
                [],
            )
            .unwrap();
        drop(connection);
        assert!(DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 32).is_err());
    }

    #[test]
    fn policy_anchor_receipt_rejects_a_stale_policy_after_rotation() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0xe4; 32]).unwrap();
        let retired_witness = IdentityKeyPair::from_bytes(&[0xe5; 32]).unwrap();
        let replacement_witness = IdentityKeyPair::from_bytes(&[0xe6; 32]).unwrap();
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        store
            .reconcile_observation_witness_policy(
                &observer,
                &[retired_witness.public_key_bytes()],
                1,
                NOW + 20,
            )
            .unwrap();
        let retired_anchor = store
            .current_observation_witness_policy_anchor()
            .unwrap()
            .unwrap();
        let stale_response =
            accepted_policy_anchor_response(&observer, &retired_witness, retired_anchor, 0xc1);

        store
            .reconcile_observation_witness_policy(
                &observer,
                &[replacement_witness.public_key_bytes()],
                1,
                NOW + 30,
            )
            .unwrap();
        assert!(matches!(
            store
                .persist_observation_witness_policy_anchor_receipt(&stale_response, NOW + 31)
                .unwrap_err(),
            DirectoryReplicaStoreError::Request(message)
                if message == "observation policy anchor receipt is outside the current local policy"
        ));
        assert_eq!(
            store
                .status_snapshot()
                .unwrap()
                .observation_witness_policy_anchor_receipts,
            0
        );
    }

    #[test]
    fn witness_runtime_uses_bounded_mutually_exclusive_buckets() {
        let runtime = DirectoryReplicaSyncRuntime::default();
        let outcomes = [
            DirectoryObservationWitnessOutcome::Accepted,
            DirectoryObservationWitnessOutcome::EvidenceUnavailable,
            DirectoryObservationWitnessOutcome::EvidenceConflict,
            DirectoryObservationWitnessOutcome::PeerUnavailable,
            DirectoryObservationWitnessOutcome::TransportFailure,
            DirectoryObservationWitnessOutcome::VerificationFailure,
            DirectoryObservationWitnessOutcome::PersistenceFailure,
        ];
        runtime.record_observation_witness_round(3, NOW, &outcomes, false);
        let snapshot = runtime.observation_witness_snapshot();
        assert_eq!(snapshot.rounds, 1);
        assert_eq!(snapshot.totals.attempts(), 7);
        assert_eq!(snapshot.totals.accepted, 1);
        assert_eq!(snapshot.totals.evidence_unavailable, 1);
        assert_eq!(snapshot.totals.evidence_conflict, 1);
        assert_eq!(snapshot.totals.peer_unavailable, 1);
        assert_eq!(snapshot.totals.transport_failures, 1);
        assert_eq!(snapshot.totals.verification_failures, 1);
        assert_eq!(snapshot.totals.persistence_failures, 1);
        assert_eq!(snapshot.last_checkpoint_sequence, 3);
        assert_eq!(snapshot.last_round_at, Some(NOW));
        assert_eq!(snapshot.last_success_at, Some(NOW));
        assert_eq!(snapshot.last_failure_at, Some(NOW));
        assert_eq!(snapshot.telemetry_persistence_failures, 1);
    }

    #[test]
    fn tampered_witness_outcome_aggregate_fails_startup_audit() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x8a; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x8b; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x8c; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x8d; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        import_replica_block(&store, &producer_a, &object, &block_a, [0x8e; 16]);
        import_replica_block(&store, &producer_b, &object, &block_b, [0x8f; 16]);
        store
            .append_observation_checkpoint(&configured, &local, NOW + 21)
            .unwrap();
        store
            .persist_observation_witness_outcome_round(
                1,
                NOW + 22,
                &[DirectoryObservationWitnessOutcome::EvidenceUnavailable],
            )
            .unwrap();
        {
            let connection = store.connection.lock();
            connection
                .pragma_update(None, "ignore_check_constraints", true)
                .unwrap();
            connection
                .execute(
                    "UPDATE directory_observation_witness_outcomes
                     SET attempts_total = attempts_total + 1 WHERE singleton = 1",
                    [],
                )
                .unwrap();
        }
        assert!(store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 23)
            .is_err());
        drop(store);

        assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 23).is_err());
    }

    #[test]
    fn observation_witness_combines_local_producer_and_remote_replica_evidence() {
        let observer_temp = TempDir::new().unwrap();
        let witness_temp = TempDir::new().unwrap();
        let observer = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
        let remote = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0xa4; 32]).unwrap();
        let object = descriptor(&subject, 1);

        let witness_path = witness_temp.path().join("directory.db");
        let (local_chain, _) =
            DirectoryChainStore::open(&witness_path, witness.public_key_bytes(), NOW + 1).unwrap();
        local_chain
            .append_descriptors(std::slice::from_ref(&object), NOW + 1, &witness)
            .unwrap();
        let local_block = local_chain.block(1).unwrap().unwrap();
        let remote_block = block(&remote, 1, [0u8; 32], &object);
        let (witness_store, _) =
            DirectoryReplicaStore::open(&witness_path, witness.public_key_bytes(), NOW + 20)
                .unwrap();
        import_replica_block(&witness_store, &remote, &object, &remote_block, [0xa5; 16]);

        let observer_path = observer_temp.path().join("directory.db");
        let (observer_store, _) =
            DirectoryReplicaStore::open(&observer_path, observer.public_key_bytes(), NOW + 20)
                .unwrap();
        import_replica_block(&observer_store, &witness, &object, &local_block, [0xa6; 16]);
        import_replica_block(&observer_store, &remote, &object, &remote_block, [0xa7; 16]);
        observer_store
            .append_observation_checkpoint(
                &[witness.public_key_bytes(), remote.public_key_bytes()],
                &observer,
                NOW + 21,
            )
            .unwrap();
        let checkpoint = observer_store
            .latest_audited_observation_checkpoint(NOW + 22)
            .unwrap()
            .unwrap();
        assert!(checkpoint
            .producer_tips
            .iter()
            .any(|tip| tip.producer == witness.public_key_bytes()));
        assert_eq!(
            witness_store
                .evaluate_observation_checkpoint_witness(&checkpoint, NOW + 22)
                .unwrap(),
            DirectoryObservationWitnessDecision::Accepted
        );
    }

    #[test]
    fn tampered_observation_witness_fails_startup_audit() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let observer = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let witness = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x95; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let configured = [producer_a.public_key_bytes(), producer_b.public_key_bytes()];
        let (store, _) =
            DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 20).unwrap();
        import_replica_block(&store, &producer_a, &object, &block_a, [0x96; 16]);
        import_replica_block(&store, &producer_b, &object, &block_b, [0x97; 16]);
        store
            .append_observation_checkpoint(&configured, &observer, NOW + 21)
            .unwrap();
        let checkpoint = store
            .latest_audited_observation_checkpoint(NOW + 22)
            .unwrap()
            .unwrap();
        let request_id = [0x98; 16];
        let checkpoint_hash = checkpoint.hash();
        let signing_bytes = directory_observation_witness_response_signing_bytes(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            &request_id,
            &observer.public_key_bytes(),
            checkpoint.sequence,
            &checkpoint_hash,
            &witness.public_key_bytes(),
            NOW + 22,
            DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
        );
        let response = DirectorySyncMessage::ObservationCheckpointWitnessResponseV1 {
            chain_id: AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            request_id,
            observer: observer.public_key_bytes(),
            checkpoint_sequence: checkpoint.sequence,
            checkpoint_hash,
            responder: witness.public_key_bytes(),
            response_timestamp: NOW + 22,
            outcome: DIRECTORY_OBSERVATION_WITNESS_ACCEPTED_V1,
            signature: witness.sign(&signing_bytes),
        };
        store
            .persist_observation_checkpoint_witness(&response, NOW + 22)
            .unwrap();
        {
            let connection = store.connection.lock();
            let mut response_blob: Vec<u8> = connection
                .query_row(
                    "SELECT response_blob FROM directory_observation_checkpoint_witnesses",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            let last = response_blob.len() - 1;
            response_blob[last] ^= 1;
            connection
                .execute(
                    "UPDATE directory_observation_checkpoint_witnesses SET response_blob = ?1",
                    params![response_blob],
                )
                .unwrap();
        }
        assert!(store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 23)
            .is_err());
        assert!(store
            .verified_observation_witness_count_for_pins(
                checkpoint.sequence,
                &[witness.public_key_bytes()],
                NOW + 23,
            )
            .is_err());
        drop(store);
        assert!(DirectoryReplicaStore::open(&path, observer.public_key_bytes(), NOW + 23).is_err());
    }

    #[test]
    fn tampered_observation_checkpoint_fails_startup_audit() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let block_a = block(&producer_a, 1, [0u8; 32], &object);
        let block_b = block(&producer_b, 1, [0u8; 32], &object);
        let frame_a = response_frame(
            &producer_a,
            vec![block_a.clone()],
            false,
            1,
            block_a.hash(),
            [0x65; 16],
        );
        let frame_b = response_frame(
            &producer_b,
            vec![block_b.clone()],
            false,
            1,
            block_b.hash(),
            [0x66; 16],
        );
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        for (producer, replica_block, frame) in [
            (&producer_a, &block_a, &frame_a),
            (&producer_b, &block_b, &frame_b),
        ] {
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(replica_block),
                    std::slice::from_ref(&object),
                    1,
                    replica_block.hash(),
                    frame,
                    NOW + 20,
                )
                .unwrap();
        }
        store
            .append_observation_checkpoint(
                &[producer_a.public_key_bytes(), producer_b.public_key_bytes()],
                &local,
                NOW + 21,
            )
            .unwrap();
        let connection = store.connection.lock();
        let mut blob: Vec<u8> = connection
            .query_row(
                "SELECT checkpoint_blob FROM directory_observation_checkpoints
                 WHERE sequence = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        *blob.last_mut().unwrap() ^= 1;
        connection
            .execute(
                "UPDATE directory_observation_checkpoints
                 SET checkpoint_blob = ?1 WHERE sequence = 1",
                params![blob],
            )
            .unwrap();
        drop(connection);
        assert!(store.audit(NOW + 22).is_err());
        assert!(store
            .latest_audited_mature_unwitnessed_observation_checkpoint(NOW + 21, NOW + 22)
            .is_err());
        drop(store);
        assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 22).is_err());
    }

    #[test]
    fn observation_convergence_reads_only_the_bounded_recent_block_window() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x70; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        let mut previous = [0u8; 32];
        for height in 1..=DIRECTORY_REPLICA_CONVERGENCE_WINDOW_BLOCKS + 1 {
            let object = descriptor(&subject, height);
            let replica_block = block(&producer, height, previous, &object);
            let frame = response_frame(
                &producer,
                vec![replica_block.clone()],
                false,
                height,
                replica_block.hash(),
                [u8::try_from(height).unwrap(); 16],
            );
            store
                .import_verified_page(
                    producer.public_key_bytes(),
                    std::slice::from_ref(&replica_block),
                    std::slice::from_ref(&object),
                    height,
                    replica_block.hash(),
                    &frame,
                    NOW + 20,
                )
                .unwrap();
            previous = replica_block.hash();
        }

        let snapshot = store
            .observation_convergence(&[producer.public_key_bytes()])
            .unwrap();
        assert_eq!(snapshot.eligible_producers, 1);
        assert_eq!(snapshot.window_blocks, 32);
        assert_eq!(snapshot.recent_commitments, 32);
        assert_eq!(snapshot.distinct_recent_commitments, 32);
        assert_eq!(snapshot.multi_source_recent_commitments, 0);
        assert_eq!(snapshot.observation_root, None);
    }

    #[test]
    fn quarantined_producer_is_excluded_from_observation_convergence() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x28; 32]).unwrap();
        let producer_a = IdentityKeyPair::from_bytes(&[0x29; 32]).unwrap();
        let producer_b = IdentityKeyPair::from_bytes(&[0x2a; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0x2b; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0x2c; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first_a = block(&producer_a, 1, [0u8; 32], &object_a);
        let first_b = block(&producer_b, 1, [0u8; 32], &object_a);
        let fork_b = block(&producer_b, 1, [0u8; 32], &object_b);
        let frame_a = response_frame(
            &producer_a,
            vec![first_a.clone()],
            false,
            1,
            first_a.hash(),
            [0x2d; 16],
        );
        let frame_b = response_frame(
            &producer_b,
            vec![first_b.clone()],
            false,
            1,
            first_b.hash(),
            [0x2e; 16],
        );
        let fork_frame = response_frame(
            &producer_b,
            vec![fork_b.clone()],
            false,
            1,
            fork_b.hash(),
            [0x2f; 16],
        );
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        store
            .import_verified_page(
                producer_a.public_key_bytes(),
                std::slice::from_ref(&first_a),
                std::slice::from_ref(&object_a),
                1,
                first_a.hash(),
                &frame_a,
                NOW + 20,
            )
            .unwrap();
        store
            .import_verified_page(
                producer_b.public_key_bytes(),
                std::slice::from_ref(&first_b),
                std::slice::from_ref(&object_a),
                1,
                first_b.hash(),
                &frame_b,
                NOW + 20,
            )
            .unwrap();
        assert!(matches!(
            store.import_verified_page(
                producer_b.public_key_bytes(),
                std::slice::from_ref(&fork_b),
                std::slice::from_ref(&object_b),
                1,
                fork_b.hash(),
                &fork_frame,
                NOW + 20,
            ),
            Err(DirectoryReplicaStoreError::Quarantined(_))
        ));

        let snapshot = store
            .observation_convergence(&[
                producer_a.public_key_bytes(),
                producer_b.public_key_bytes(),
            ])
            .unwrap();
        assert_eq!(snapshot.configured_producers, 2);
        assert_eq!(snapshot.eligible_producers, 1);
        assert_eq!(snapshot.pending_producers, 0);
        assert_eq!(snapshot.excluded_quarantined_producers, 1);
        assert_eq!(snapshot.recent_commitments, 1);
        assert_eq!(snapshot.multi_source_recent_commitments, 0);
        assert_eq!(snapshot.all_eligible_source_recent_commitments, 0);
        assert_eq!(snapshot.observation_root, None);
    }

    #[test]
    fn schema_v1_is_atomically_migrated_to_v7() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V1],
            )
            .unwrap();
        connection
            .execute_batch(
                "DROP TABLE directory_observation_witness_outcomes;
                 DROP TABLE directory_observation_checkpoint_witnesses;
                 DROP TABLE directory_observation_checkpoints;
                 DROP TABLE directory_replica_resolutions;
                 DROP TABLE directory_replica_retry_state;
                 ALTER TABLE directory_replica_chains DROP COLUMN active_incident_digest;
                 ALTER TABLE directory_replica_chains DROP COLUMN last_resolution_digest;",
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.retry_states, 0);
        let connection = store.connection.lock();
        let version: i64 = connection
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let retry_table: String = connection
            .query_row(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name = 'directory_replica_retry_state'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(retry_table, "directory_replica_retry_state");
        let resolution_columns: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('directory_replica_chains')
                 WHERE name IN ('active_incident_digest', 'last_resolution_digest')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(resolution_columns, 2);
    }

    #[test]
    fn schema_v2_is_atomically_migrated_to_v7() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x30; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V2],
            )
            .unwrap();
        connection
            .execute_batch(
                "DROP TABLE directory_observation_witness_outcomes;
                 DROP TABLE directory_observation_checkpoint_witnesses;
                 DROP TABLE directory_observation_checkpoints;
                 DROP TABLE directory_replica_resolutions;
                 ALTER TABLE directory_replica_chains DROP COLUMN active_incident_digest;
                 ALTER TABLE directory_replica_chains DROP COLUMN last_resolution_digest;",
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.resolutions, 0);
        let connection = store.connection.lock();
        let version: i64 = connection
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let resolution_table: String = connection
            .query_row(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table' AND name = 'directory_replica_resolutions'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(resolution_table, "directory_replica_resolutions");
    }

    #[test]
    fn schema_v3_is_atomically_migrated_to_v7() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x2f; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V3],
            )
            .unwrap();
        connection
            .execute_batch(
                "DROP TABLE directory_observation_witness_outcomes;
                 DROP TABLE directory_observation_checkpoint_witnesses;
                 DROP TABLE directory_observation_checkpoints;",
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.observation_checkpoints, 0);
        let connection = store.connection.lock();
        let (version, checkpoint_table): (i64, String) = connection
            .query_row(
                "SELECT m.schema_version, t.name
                 FROM directory_replica_meta m
                 JOIN sqlite_master t
                   ON t.type = 'table' AND t.name = 'directory_observation_checkpoints'
                 WHERE m.singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(checkpoint_table, "directory_observation_checkpoints");
    }

    #[test]
    fn schema_v4_is_atomically_migrated_to_v7() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x2e; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V4],
            )
            .unwrap();
        connection
            .execute_batch(
                "DROP TABLE directory_observation_witness_outcomes;
                 DROP TABLE directory_observation_checkpoint_witnesses;",
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.observation_checkpoint_witnesses, 0);
        let connection = store.connection.lock();
        let (version, witness_table): (i64, String) = connection
            .query_row(
                "SELECT m.schema_version, t.name
                 FROM directory_replica_meta m
                 JOIN sqlite_master t
                   ON t.type = 'table'
                  AND t.name = 'directory_observation_checkpoint_witnesses'
                 WHERE m.singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(witness_table, "directory_observation_checkpoint_witnesses");
    }

    #[test]
    fn schema_v5_is_atomically_migrated_to_v7() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x2d; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V5],
            )
            .unwrap();
        connection
            .execute_batch("DROP TABLE directory_observation_witness_outcomes;")
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(
            audit.observation_witness_outcomes,
            DirectoryObservationWitnessOutcomeSnapshot::default()
        );
        let connection = store.connection.lock();
        let (version, outcome_table): (i64, String) = connection
            .query_row(
                "SELECT m.schema_version, t.name
                 FROM directory_replica_meta m
                 JOIN sqlite_master t
                   ON t.type = 'table'
                  AND t.name = 'directory_observation_witness_outcomes'
                 WHERE m.singleton = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(outcome_table, "directory_observation_witness_outcomes");
    }

    #[test]
    fn schema_v6_adds_witness_policy_metadata_and_table_atomically() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x2c; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "ALTER TABLE directory_replica_meta RENAME TO directory_replica_meta_v7;
                 CREATE TABLE directory_replica_meta (
                     singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                     schema_version INTEGER NOT NULL,
                     chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                     local_node_id BLOB NOT NULL CHECK (length(local_node_id) = 32)
                 );
                 INSERT INTO directory_replica_meta
                     (singleton, schema_version, chain_id, local_node_id)
                 SELECT singleton, 6, chain_id, local_node_id
                 FROM directory_replica_meta_v7;
                 DROP TABLE directory_replica_meta_v7;
                 DROP TABLE directory_observation_witness_policies;",
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.observation_witness_policy_epochs, 0);
        let connection = store.connection.lock();
        let version: i64 = connection
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let policy_columns: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('directory_replica_meta')
                 WHERE name IN ('witness_policy_epoch', 'witness_policy_head')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let policy_table: String = connection
            .query_row(
                "SELECT name FROM sqlite_master
                 WHERE type = 'table'
                   AND name = 'directory_observation_witness_policies'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(policy_columns, 2);
        assert_eq!(policy_table, "directory_observation_witness_policies");
    }

    #[test]
    fn schema_v7_adds_policy_anchor_evidence_tables_atomically() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x2b; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        drop(store);

        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_replica_meta SET schema_version = ?1 WHERE singleton = 1",
                params![DIRECTORY_REPLICA_SCHEMA_VERSION_V7],
            )
            .unwrap();
        connection
            .execute_batch(
                "DROP TABLE directory_observation_policy_anchor_receipts;
                 DROP TABLE directory_observation_remote_policy_anchors;",
            )
            .unwrap();
        drop(connection);

        let (store, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 1).unwrap();
        assert_eq!(audit.observation_witness_policy_anchor_receipts, 0);
        assert_eq!(audit.observation_witness_remote_policy_anchors, 0);
        let connection = store.connection.lock();
        let version: i64 = connection
            .query_row(
                "SELECT schema_version FROM directory_replica_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let anchor_tables: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'table'
                   AND name IN (
                       'directory_observation_remote_policy_anchors',
                       'directory_observation_policy_anchor_receipts'
                   )",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let receipt_index: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index'
                   AND name = 'directory_policy_anchor_receipts_by_epoch'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(version, DIRECTORY_REPLICA_SCHEMA_VERSION);
        assert_eq!(anchor_tables, 2);
        assert_eq!(receipt_index, 1);
    }

    #[test]
    fn retry_state_survives_reopen_and_is_fully_audited() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x33; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        store
            .persist_retry_failure(
                producer_id,
                2,
                Some(NOW + 120),
                NOW,
                "directory_range_transport_failed",
            )
            .unwrap();
        store.persist_retry_skip(producer_id, NOW + 1).unwrap();
        let states = store.retry_states().unwrap();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].consecutive_failures, 2);
        assert_eq!(states[0].retry_not_before, Some(NOW + 120));
        assert_eq!(states[0].backoff_skips, 1);
        drop(store);

        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 2).unwrap();
        assert_eq!(audit.producers, 1);
        assert_eq!(audit.retry_states, 1);
        assert_eq!(reopened.retry_states().unwrap(), states);
    }

    #[test]
    fn older_failure_timestamp_cannot_shorten_retry_boundary() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x3a; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x3b; 32]).unwrap();
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW,
        )
        .unwrap();
        store
            .persist_retry_failure(
                producer_id,
                2,
                Some(NOW + 300),
                NOW,
                "directory_range_transport_failed",
            )
            .unwrap();
        store
            .persist_retry_failure(
                producer_id,
                3,
                Some(NOW + 100),
                NOW - 100,
                "directory_object_transport_failed",
            )
            .unwrap();

        let state = &store.retry_states().unwrap()[0];
        assert_eq!(state.consecutive_failures, 3);
        assert_eq!(state.last_failure_at, NOW);
        assert_eq!(state.retry_not_before, Some(NOW + 300));
        assert_eq!(
            state.last_failure_reason,
            "directory_range_transport_failed"
        );
    }

    #[test]
    fn authenticated_import_atomically_clears_durable_retry_state() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let local = IdentityKeyPair::from_bytes(&[0x34; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x35; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x36; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let first = block(&producer, 1, [0u8; 32], &object);
        let frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x37; 16],
        );
        let producer_id = producer.public_key_bytes();
        let (store, _) = DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW).unwrap();
        store
            .persist_retry_failure(
                producer_id,
                3,
                Some(NOW + 300),
                NOW,
                "directory_object_transport_failed",
            )
            .unwrap();
        assert_eq!(store.retry_states().unwrap().len(), 1);

        store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&first),
                std::slice::from_ref(&object),
                1,
                first.hash(),
                &frame,
                NOW + 20,
            )
            .unwrap();
        assert!(store.retry_states().unwrap().is_empty());
        drop(store);

        let (_, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(audit.retry_states, 0);
        assert_eq!(audit.blocks, 1);
    }

    #[test]
    fn retry_state_rejects_peer_controlled_or_unbounded_fields() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x38; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x39; 32]).unwrap();
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW,
        )
        .unwrap();
        assert!(store
            .persist_retry_failure(
                producer.public_key_bytes(),
                1,
                None,
                NOW,
                "https://peer.example/private",
            )
            .is_err());
        assert!(store
            .persist_retry_failure(
                producer.public_key_bytes(),
                DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES + 1,
                None,
                NOW,
                "directory_range_transport_failed",
            )
            .is_err());
        assert!(store.retry_states().unwrap().is_empty());
    }

    #[test]
    fn sync_runtime_tracks_bounded_success_and_stable_failure_without_endpoint_data() {
        let producer = [0x44; 32];
        let runtime = DirectoryReplicaSyncRuntime::default();
        runtime.register_producers(&[producer]);
        runtime.record_attempt(producer, NOW);
        runtime.record_success(producer, NOW + 1, 3, 7, true, 1, 4, 2);
        runtime.record_attempt(producer, NOW + 2);
        runtime.record_failure(
            producer,
            NOW + 3,
            "pinned_directory_peer_unavailable_and_reason_is_bounded",
            Some(NOW + 123),
        );
        runtime.record_backoff_skip(producer);

        let observations = runtime.snapshot();
        assert_eq!(observations.len(), 1);
        let observation = &observations[0];
        assert_eq!(observation.producer, producer);
        assert_eq!(observation.last_success_at, Some(NOW + 1));
        assert_eq!(observation.last_failure_at, Some(NOW + 3));
        assert_eq!(observation.remote_tip_height, Some(7));
        assert_eq!(observation.local_tip_height, 3);
        assert!(observation.has_more);
        assert_eq!(observation.consecutive_failures, 1);
        assert_eq!(observation.total_attempts, 2);
        assert_eq!(observation.successful_pages, 1);
        assert_eq!(observation.failed_attempts, 1);
        assert_eq!(observation.retry_not_before, Some(NOW + 123));
        assert_eq!(observation.backoff_skips, 1);
        assert_eq!(observation.blocks_inserted, 1);
        assert_eq!(observation.commitments_inserted, 4);
        assert_eq!(observation.requests_sent, 2);
        assert_eq!(
            observation.last_failure_reason.as_deref(),
            Some("pinned_directory_peer_unavailable_and_reason_is_bounded")
        );
        assert_eq!(
            runtime.deferred_retry_until(&producer, NOW + 20),
            Some(NOW + 123)
        );
        assert_eq!(runtime.deferred_retry_until(&producer, NOW + 123), None);
        assert_eq!(runtime.consecutive_failures(&producer), 1);
    }

    #[test]
    fn sync_runtime_success_clears_backoff_without_erasing_history() {
        let producer = [0x45; 32];
        let runtime = DirectoryReplicaSyncRuntime::default();
        runtime.record_failure(
            producer,
            NOW,
            "directory_range_transport_failed",
            Some(NOW + 60),
        );
        runtime.record_backoff_skip(producer);
        runtime.record_success(producer, NOW + 61, 4, 4, false, 1, 2, 1);

        let observation = &runtime.snapshot()[0];
        assert_eq!(observation.consecutive_failures, 0);
        assert_eq!(observation.retry_not_before, None);
        assert_eq!(observation.failed_attempts, 1);
        assert_eq!(observation.backoff_skips, 1);
        assert_eq!(observation.successful_pages, 1);
    }

    #[test]
    fn sync_runtime_restores_only_bounded_scheduler_state() {
        let producer = [0x46; 32];
        let runtime = DirectoryReplicaSyncRuntime::default();
        runtime.register_producers(&[producer]);
        runtime.restore_retry_states(&[DirectoryReplicaRetryState {
            producer,
            consecutive_failures: DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES,
            retry_not_before: Some(NOW + 600),
            last_failure_at: NOW,
            last_failure_reason: "directory_range_transport_failed".to_string(),
            backoff_skips: 7,
        }]);

        let observation = &runtime.snapshot()[0];
        assert_eq!(
            observation.consecutive_failures,
            DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
        );
        assert_eq!(observation.retry_not_before, Some(NOW + 600));
        assert_eq!(observation.last_attempt_at, Some(NOW));
        assert_eq!(observation.last_failure_at, Some(NOW));
        assert_eq!(observation.backoff_skips, 7);
        assert_eq!(observation.total_attempts, 0);
        assert_eq!(observation.failed_attempts, 0);
        assert_eq!(observation.successful_pages, 0);

        runtime.record_failure(
            producer,
            NOW + 601,
            "directory_range_transport_failed",
            Some(NOW + 1_200),
        );
        assert_eq!(
            runtime.snapshot()[0].consecutive_failures,
            DIRECTORY_REPLICA_MAX_CONSECUTIVE_FAILURES
        );
    }

    #[test]
    fn signed_block_fork_is_durably_quarantined_without_rewriting_prefix() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0x53; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0x54; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first = block(&producer, 1, [0u8; 32], &object_a);
        let fork = block(&producer, 1, [0u8; 32], &object_b);
        let first_frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x55; 16],
        );
        let fork_frame = response_frame(
            &producer,
            vec![fork.clone()],
            false,
            1,
            fork.hash(),
            [0x56; 16],
        );
        let mut invalid_evidence = fork_frame.clone();
        *invalid_evidence.last_mut().unwrap() ^= 0x01;
        assert!(
            verify_incident_response_evidence(&invalid_evidence, &producer.public_key_bytes())
                .is_err()
        );
        let path = temp.path().join("directory.db");
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&first),
                std::slice::from_ref(&object_a),
                1,
                first.hash(),
                &first_frame,
                NOW + 20,
            )
            .unwrap();
        let error = store
            .import_verified_page(
                producer.public_key_bytes(),
                std::slice::from_ref(&fork),
                std::slice::from_ref(&object_b),
                1,
                fork.hash(),
                &fork_frame,
                NOW + 20,
            )
            .unwrap_err();
        assert!(matches!(error, DirectoryReplicaStoreError::Quarantined(_)));
        let tip = store.producer_tip(&producer.public_key_bytes()).unwrap();
        assert!(tip.quarantined);
        assert_eq!(tip.tip_hash, first.hash());
        assert!(store.incident_summaries(None, 0).is_err());
        assert!(store
            .incident_summaries(None, MAX_DIRECTORY_REPLICA_INCIDENT_PAGE_SIZE + 1)
            .is_err());
        let page = store.incident_summaries(None, 1).unwrap();
        assert_eq!(page.incidents.len(), 1);
        assert_eq!(page.next_cursor, None);
        let summary = &page.incidents[0];
        assert_eq!(summary.producer, producer.public_key_bytes());
        assert_eq!(summary.subject_node_id, producer.public_key_bytes());
        assert_eq!(summary.kind, "signed_tip_fork");
        assert_eq!(summary.height, 1);
        assert_eq!(summary.local_hash, first.hash());
        assert_eq!(summary.remote_hash, fork.hash());
        assert!(summary.producer_quarantined);
        assert!(store
            .incident_summaries(Some(summary.incident_digest), 1)
            .unwrap()
            .incidents
            .is_empty());
        let evidence = store
            .incident_evidence(&summary.incident_digest)
            .unwrap()
            .unwrap();
        assert_eq!(evidence.summary, *summary);
        assert_eq!(evidence.evidence_frame, fork_frame);
        let expected_evidence_sha256: [u8; 32] = Sha256::digest(&fork_frame).into();
        assert_eq!(evidence.evidence_sha256, expected_evidence_sha256);
        let incident_digest = summary.incident_digest;
        drop(store);
        let (reopened, audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 21).unwrap();
        assert_eq!(audit.quarantined_producers, 1);
        assert_eq!(audit.incidents, 1);
        assert!(reopened
            .incident_evidence(&incident_digest)
            .unwrap()
            .is_some());

        let mut corrupted_frame = fork_frame;
        *corrupted_frame.last_mut().unwrap() ^= 0x01;
        let connection = reopened.connection.lock();
        connection
            .execute(
                "UPDATE directory_replica_incidents SET evidence_frame = ?2
                 WHERE incident_digest = ?1",
                params![incident_digest.as_slice(), corrupted_frame],
            )
            .unwrap();
        drop(connection);
        assert!(reopened.incident_evidence(&incident_digest).is_err());
    }

    #[test]
    fn signed_resolution_resumes_only_exact_prefix_and_links_repeated_incidents() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x81; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x82; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0x83; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0x84; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first = block(&producer, 1, [0u8; 32], &object_a);
        let fork = block(&producer, 1, [0u8; 32], &object_b);
        let first_frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x85; 16],
        );
        let fork_frame = response_frame(
            &producer,
            vec![fork.clone()],
            false,
            1,
            fork.hash(),
            [0x86; 16],
        );
        let producer_id = producer.public_key_bytes();
        let path = temp.path().join("directory.db");
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&first),
                std::slice::from_ref(&object_a),
                1,
                first.hash(),
                &first_frame,
                NOW + 20,
            )
            .unwrap();
        assert!(store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&fork),
                std::slice::from_ref(&object_b),
                1,
                fork.hash(),
                &fork_frame,
                NOW + 20,
            )
            .is_err());
        store
            .persist_retry_failure(
                producer_id,
                2,
                Some(NOW + 120),
                NOW + 21,
                "directory_range_transport_failed",
            )
            .unwrap();
        let incident_digest =
            store.incident_summaries(None, 1).unwrap().incidents[0].incident_digest;
        let tip = store.producer_tip(&producer_id).unwrap();
        assert_eq!(tip.active_incident_digest, Some(incident_digest));

        let predates_incident =
            resolution_command(&local, incident_digest, &tip, [0x87; 16], NOW + 19);
        assert!(store
            .resolve_quarantine(&predates_incident, NOW + 19)
            .is_err());

        let mut stale_tip = tip.clone();
        stale_tip.tip_hash = [0x99; 32];
        let stale = resolution_command(&local, incident_digest, &stale_tip, [0x88; 16], NOW + 22);
        assert!(store.resolve_quarantine(&stale, NOW + 22).is_err());
        assert_eq!(store.status_snapshot().unwrap().resolutions, 0);

        let first_resolution =
            resolution_command(&local, incident_digest, &tip, [0x89; 16], NOW + 22);
        let first_report = store
            .resolve_quarantine(&first_resolution, NOW + 22)
            .unwrap();
        let resumed = store.producer_tip(&producer_id).unwrap();
        assert!(!resumed.quarantined);
        assert_eq!(resumed.active_incident_digest, None);
        assert_eq!(
            resumed.last_resolution_digest,
            Some(first_report.resolution_digest)
        );
        assert_eq!(resumed.tip_hash, first.hash());
        assert!(store.retry_states().unwrap().is_empty());
        assert!(store
            .resolve_quarantine(&first_resolution, NOW + 22)
            .is_err());

        assert!(store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&fork),
                std::slice::from_ref(&object_b),
                1,
                fork.hash(),
                &fork_frame,
                NOW + 24,
            )
            .is_err());
        let requarantined = store.producer_tip(&producer_id).unwrap();
        assert_eq!(requarantined.active_incident_digest, Some(incident_digest));
        assert_eq!(
            requarantined.last_resolution_digest,
            Some(first_report.resolution_digest)
        );
        let predates_predecessor = resolution_command(
            &local,
            incident_digest,
            &requarantined,
            [0x8a; 16],
            NOW + 21,
        );
        assert!(store
            .resolve_quarantine(&predates_predecessor, NOW + 21)
            .is_err());
        let second_resolution = resolution_command(
            &local,
            incident_digest,
            &requarantined,
            [0x8b; 16],
            NOW + 25,
        );
        let second_report = store
            .resolve_quarantine(&second_resolution, NOW + 25)
            .unwrap();
        assert_ne!(
            first_report.resolution_digest,
            second_report.resolution_digest
        );
        let snapshot = store.status_snapshot().unwrap();
        assert_eq!(snapshot.incidents, 1);
        assert_eq!(snapshot.resolutions, 2);
        assert_eq!(snapshot.producer_snapshots[0].resolutions, 2);
        let audit = store.audit(NOW + 26).unwrap();
        assert_eq!(audit.incidents, 1);
        assert_eq!(audit.resolutions, 2);
        drop(store);
        let (_, reopened_audit) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 27).unwrap();
        assert_eq!(reopened_audit.resolutions, 2);
    }

    #[test]
    fn forged_quarantine_clear_without_signed_resolution_fails_audit() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x91; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x92; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0x93; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0x94; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first = block(&producer, 1, [0u8; 32], &object_a);
        let fork = block(&producer, 1, [0u8; 32], &object_b);
        let first_frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x95; 16],
        );
        let fork_frame = response_frame(
            &producer,
            vec![fork.clone()],
            false,
            1,
            fork.hash(),
            [0x96; 16],
        );
        let producer_id = producer.public_key_bytes();
        let path = temp.path().join("directory.db");
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&first),
                std::slice::from_ref(&object_a),
                1,
                first.hash(),
                &first_frame,
                NOW + 20,
            )
            .unwrap();
        assert!(store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&fork),
                std::slice::from_ref(&object_b),
                1,
                fork.hash(),
                &fork_frame,
                NOW + 20,
            )
            .is_err());
        let connection = store.connection.lock();
        connection
            .execute(
                "UPDATE directory_replica_chains
                 SET quarantined = 0, quarantine_kind = NULL,
                     active_incident_digest = NULL
                 WHERE producer = ?1",
                params![producer_id.as_slice()],
            )
            .unwrap();
        drop(connection);
        assert!(store.audit(NOW + 21).is_err());
        drop(store);
        assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 22).is_err());
    }

    #[test]
    fn tampered_resolution_signature_fails_startup_audit() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0xa1; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0xa2; 32]).unwrap();
        let subject_a = IdentityKeyPair::from_bytes(&[0xa3; 32]).unwrap();
        let subject_b = IdentityKeyPair::from_bytes(&[0xa4; 32]).unwrap();
        let object_a = descriptor(&subject_a, 1);
        let object_b = descriptor(&subject_b, 1);
        let first = block(&producer, 1, [0u8; 32], &object_a);
        let fork = block(&producer, 1, [0u8; 32], &object_b);
        let first_frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0xa5; 16],
        );
        let fork_frame = response_frame(
            &producer,
            vec![fork.clone()],
            false,
            1,
            fork.hash(),
            [0xa6; 16],
        );
        let producer_id = producer.public_key_bytes();
        let path = temp.path().join("directory.db");
        let (store, _) =
            DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 20).unwrap();
        store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&first),
                std::slice::from_ref(&object_a),
                1,
                first.hash(),
                &first_frame,
                NOW + 20,
            )
            .unwrap();
        assert!(store
            .import_verified_page(
                producer_id,
                std::slice::from_ref(&fork),
                std::slice::from_ref(&object_b),
                1,
                fork.hash(),
                &fork_frame,
                NOW + 20,
            )
            .is_err());
        let incident_digest =
            store.incident_summaries(None, 1).unwrap().incidents[0].incident_digest;
        let tip = store.producer_tip(&producer_id).unwrap();
        let command = resolution_command(&local, incident_digest, &tip, [0xa7; 16], NOW + 21);
        let report = store.resolve_quarantine(&command, NOW + 21).unwrap();
        let connection = store.connection.lock();
        connection
            .execute(
                "UPDATE directory_replica_resolutions SET signature = ?2
                 WHERE resolution_digest = ?1",
                params![report.resolution_digest.as_slice(), [0u8; 64].as_slice()],
            )
            .unwrap();
        drop(connection);
        assert!(store.audit(NOW + 22).is_err());
        drop(store);
        assert!(DirectoryReplicaStore::open(&path, local.public_key_bytes(), NOW + 23).is_err());
    }

    #[test]
    fn malformed_or_unrelated_objects_are_rejected_before_sqlite_changes() {
        let temp = TempDir::new().unwrap();
        let local = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
        let subject = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
        let unrelated = IdentityKeyPair::from_bytes(&[0x64; 32]).unwrap();
        let object = descriptor(&subject, 1);
        let wrong = descriptor(&unrelated, 1);
        let first = block(&producer, 1, [0u8; 32], &object);
        let frame = response_frame(
            &producer,
            vec![first.clone()],
            false,
            1,
            first.hash(),
            [0x65; 16],
        );
        let (store, _) = DirectoryReplicaStore::open(
            temp.path().join("directory.db"),
            local.public_key_bytes(),
            NOW + 20,
        )
        .unwrap();
        assert!(store
            .import_verified_page(
                producer.public_key_bytes(),
                &[first],
                &[wrong],
                1,
                frame_tip_hash(&frame),
                &frame,
                NOW + 20,
            )
            .is_err());
        assert_eq!(
            store
                .producer_tip(&producer.public_key_bytes())
                .unwrap()
                .tip_height,
            0
        );
    }

    fn frame_tip_hash(frame: &[u8]) -> [u8; 32] {
        let DirectorySyncMessage::BlockRangeResponseV1 { tip_hash, .. } =
            decode_directory_sync_message(frame).unwrap()
        else {
            unreachable!()
        };
        tip_hash
    }
}
