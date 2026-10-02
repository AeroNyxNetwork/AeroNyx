// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_ops.rs
// ============================================
//! # Storage Operations — Core MemoryStorage Methods
//!
//! ## Creation Reason
//! Extracted from storage.rs to reduce file size. Contains all extended
//! operations that are NOT core CRUD: rawlog ops, feedback ops, chain state,
//! statistics, miner support, and overview queries.
//!
//! ## Main Functionality
//! - RawLog and feedback operations live in `storage_feedback.rs`.
//! - Chain State: set_chain_state, last_block_hash, last_block_height
//! - Stats: stats(), count(), total_inserted/rejected
//! - Miner: count_by_layer, compact_episodes_to_archive, get_records_needing_embedding,
//!   get_correction_records, update_topic_tags, supersede_record
//! - MVF Weights: load_user_weights, save_user_weights
//! - Content Dedup: has_active_content
//! - v2.2.0: get_embedding_model, get_overview
//! - v2.3.0: count_distinct_owners, owner_exists (Phase 1 remote storage capacity check)
//! - v2.5.3+Isolation: get_active_records_by_context (project_id context filter for /recall)
//! - v2.7.0-BlockSync: atomic commitment append, bounded range reads, authoritative
//!   tip recovery, aggregate status, and uncommitted blind-record selection
//! - v2.7.3-BlockAudit: full startup verification of persisted commitment blocks,
//!   denormalized rows, signatures, continuity, and membership indexes
//! - v2.7.4-BlockIntegrityStatus: snapshot-consistent audits plus runtime/API
//!   evidence that advances only with transactionally verified appends
//! - v2.7.5-CheckpointProof: atomic verified-tip checkpoint reads and
//!   privacy-safe signed reconciliation runtime evidence
//! - v2.7.6-EvidenceVault: bounded durable proof frames, fail-closed persistence,
//!   and complete restart-time cryptographic evidence audit
//! - v2.7.7-EvidenceRestartRecovery: file-backed WAL visibility, clean reopen,
//!   v8-to-v9 preservation, and tampered-disk restart regression coverage
//! - v2.7.10-CheckpointDirectionIsolation: inbound proof serving counters no
//!   longer overwrite outbound convergence, divergence, or height evidence
//! - v2.7.11-CheckpointFreshness: age-bounded durable observation state that
//!   remains independent from vault integrity and transport attempts
//! - v2.7.12-WitnessRoundEvidence: explicit bounded-round coverage and result
//!   that remains evidence only and never becomes a consensus rule
//! - v2.7.13-CommitmentDurability: coordinator-only SQLite FULL durability,
//!   startup fail-closed verification, and aggregate durability evidence
//! - v2.7.17-AtomicBlockPage: one SQLite transaction per verified peer page,
//!   with single-block mining delegated to the same authoritative path
//! - v2.7.18-VerifiedRangeSnapshot: audit-gated range serving from one SQLite
//!   snapshot with canonical payload and signature re-verification
//! - v2.7.19-FollowerEvidenceRecovery: retains cryptographically valid
//!   historical checkpoint frames across follower-local rollback while
//!   excluding deferred evidence from current freshness and fork telemetry
//! - v2.7.20-WitnessEquivocation: atomically retains conflicting signed claims
//!   from explicitly trusted witnesses and blocks later evidence rotation from
//!   erasing the incident
//! - v2.7.21-TrustedDivergenceHalt: promotes an operator-pinned divergent
//!   prefix to a sticky incident and closes local production at the append layer
//! - v2.7.22-CheckpointCertificate: immutable bounded multi-witness bundles
//! - v2.7.23-CertificateExchange: audited exact-frame bundle export for the
//!   admitted fixed-size peer exchange protocol
//! - v2.7.24-CertificateRollbackGuard: signed local certificate high-water
//!   sidecar with serialized DB/sidecar commits and fail-closed recovery
//! - v2.7.25-CoordinatorProductionFence: process-lifetime OS lock that rejects
//!   duplicate local coordinators before audit, listeners, or block production
//! - v2.8.10-CoordinatorLease: durable short-lived witness grants that fence
//!   duplicate cross-host coordinator process instances
//! - v2.8.11-CoordinatorLeaseRelease: exact-instance graceful release that
//!   preserves monotonic epochs and immediate planned-restart handover
//! - v2.8.12-LeaseFailClosedTelemetry: monotonic remaining authority,
//!   consecutive renewal failures, last attempt/failure, and recovery counts
//! - v2.8.13-BlockConfirmation: derives privacy-safe witness-certificate
//!   coverage and lag from the audited local tip without claiming finality
//! - v2.8.28-VerifiedDeliveryAnchorWitness: atomic, bounded, generation-
//!   contiguous external high-water decisions for signed delivery-cache anchors
//! - [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Generalizes the durable
//!   opaque monotonic witness transition while keeping delivery and custody
//!   generations in separate schema-v16 tables.
//! - v2.8.31-FollowerCertificatePolicy: re-audits current-tip certificate
//!   membership against the follower's current local witness pins
//! - v2.8.59-FollowerEffectiveReadiness: derives one fail-closed follower
//!   readiness state from block convergence and exact-tip certificate policy
//! - [MEMCHAIN-CHAIN-LEASE 2026-08-12 by Codex] Scopes witness-side
//!   coordinator exclusivity and monotonic epochs to the chain across key
//!   rotation while safely normalizing legacy per-identity rows
//! - [COORDINATOR-HANDOVER 2026-08-12 by Codex] Persists and re-audits the
//!   complete dual-signed coordinator authority history against exact block
//!   prefixes, contiguous epochs, and non-overlapping witness leases
//! - [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Enforces the immutable
//!   root and exact-height active coordinator during startup audit, explicit
//!   handover audit, and atomic live block append
//! - [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Retains one bounded,
//!   identity-blind diagnostic when typed follower carrier policy construction
//!   fails before its required runtime task is spawned
//! - [CUSTODY-WITNESS-TIME-HARDENING 2026-08-18 by Codex] Uses one-sided
//!   receipt age with a bounded future-clock tolerance so imported future
//!   timestamps cannot extend startup evidence freshness.
//! - [CUSTODY-WITNESS-ATOMIC-READINESS 2026-08-18 by Codex] Derives typed
//!   exact-anchor readiness and aggregate vault audit from one SQLite snapshot.
//! - [CUSTODY-WITNESS-TWO-PHASE-AUDIT 2026-08-18 by Codex] Copies the bounded
//!   receipt rows under one SQLite snapshot, then verifies them after releasing
//!   the connection lock on read-only audit and readiness paths.
//! - [CUSTODY-WITNESS-FULL-DURABILITY 2026-09-24 by Codex] Requires verified
//!   SQLite FULL-or-stronger durability before acknowledging a new receipt.
//! - [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] Receipt admission,
//!   bounded retention, policy evaluation, and tests live together in the
//!   private storage_witness_receipts module; stable re-exports remain here.
//! - [ANCHOR-WORKER-PRIVACY 2026-07-30 by Codex] Runs signed local-anchor
//!   writes through one privacy-safe blocking worker boundary.
//!
//! ## Split Architecture (v2.4.0+Search)
//! This file was split into focused extension modules to reduce size:
//!   - storage_ops.rs   (this file) — commitment, stats, and legacy memory operations
//!   - storage_feedback.rs          — raw-log ingestion and memory feedback
//!   - storage_graph.rs             — cognitive graph CRUD (Episodes, Entities, Edges, etc.)
//!   - storage_miner.rs             — Miner step support (get_rawlogs_for_session, merge_entities, etc.)
//!
//! All public types and methods remain accessible via their respective modules.
//! Callers use `use super::storage_ops::{OverviewData, OverviewRecord}` etc. — unchanged.
//!
//! ## Dependencies
//! - storage.rs — MemoryStorage struct, LruCache, schema, core CRUD
//! - storage_crypto.rs — encrypt/decrypt functions for rawlog and content
//!
//! ⚠️ Important Note for Next Developer:
//! - All methods here access self.conn (TokioMutex<Connection>) and self.cache (RwLock<LruCache>)
//! - Custody startup and operator surfaces must consume the atomic readiness
//!   snapshot; do not independently reinterpret its aggregate counters.
//! - When adding new query methods, use self.query_rows() for SELECT queries that
//!   return MemoryRecord (it handles record_key decryption transparently)
//! - For raw SQL that reads encrypted_content directly (like get_overview), you MUST
//!   manually decrypt using self.record_key
//! - count_distinct_owners() and owner_exists() are used by the MPI auth middleware
//!   for max_remote_owners capacity checks. They must be fast (indexed queries).
//! - Cognitive graph methods have been moved to storage_graph.rs.
//! - Miner Step support methods have been moved to storage_miner.rs.
//! - get_active_records_by_context() uses LEFT JOIN records→sessions to check
//!   project_id on EITHER the record directly OR via its session. Records inserted
//!   via /remember directly (no session) are matched by records.project_id only.
//! - `append_record_commitment_block` is the only authoritative tip advance for
//!   the new chain. It delegates to `append_record_commitment_blocks_atomic`;
//!   never update the tip independently of that transaction.
//! - Peer range reads return commitments only; full records remain owner-scoped.
//!   Public serving must use `get_verified_record_commitment_block_page`, not
//!   the compatibility range reader, so no unaudited/stale view is signed.
//! - An inbound checkpoint request describes the requester's state, not this
//!   node's observation of the network. Serving it must never mutate outbound
//!   checkpoint relation, heights, or divergence counters.
//! - Freshness derives only from an audited, currently applicable durable
//!   `last_evidence_at`; a deferred historical frame, failed attempt, or served
//!   response must never make stale or unavailable evidence appear fresh.
//! - Witness round state is process-local aggregate telemetry. Do not use its
//!   counts as votes, quorum, finality, leader election, or fork choice.
//! - A coordinator must call `configure_record_commitment_durability(true)`
//!   before startup audit. Failure to confirm FULL-or-stronger is fatal; this
//!   protects acknowledged commitment tips from ordinary host power loss.
//! - A coordinator must configure the signed tip anchor after the full startup
//!   audit and before mining. It detects an older/replaced SQLite chain while
//!   the host-side anchor remains, but not rollback of the entire host/disk.
//! - A released coordinator lease row must be retained. Its expiry/update
//!   sentinel prevents delayed renewal by the old process while allowing only
//!   a different random instance to advance the durable epoch immediately.
//! - Coordinator leases are duplicate-writer fencing, not consensus, leader
//!   election, finality, quorum, or fork choice.
//! - Verified-delivery anchor witnesses retain only one generation and opaque
//!   digest per requester. They never store delivery counts, timestamps,
//!   routes, message identifiers, payloads, endpoints, or user identities.
//!
//! ## Modification History
//! [MEMCHAIN-STORAGE-FEEDBACK-SPLIT 2026-09-23 by Codex] Extracted raw-log
//! ingestion and feedback persistence without changing their SQL or API.
//! [MEMCHAIN-WITNESS-CLOCK 2026-09-05 by Codex] Serializes witness lease
//! measurement and mutation behind a monotonic hold plus an OS advisory lock;
//! restart and ambiguous commits remain conservatively fail-closed.
//! [MEMCHAIN-WITNESS-CLOCK-PROCESS 2026-09-05 by Codex] Adds deterministic
//! child-process coverage for OS-lock exclusion, clean release, and
//! unreleased crash recovery without changing production semantics.
//! [MEMCHAIN-WITNESS-DB-IDENTITY 2026-09-05 by Codex] Derives the host-local
//! witness lock from verified SQLite device/inode identity, closing symlink,
//! hardlink, rename, and replacement path-spelling ambiguity.
//! v2.8.65-CustodyWitnessReceiptImport - Added bounded host-local receipt
//! import without weakening the live network persistence freshness policy.
//! v2.8.64-CustodyWitnessReceiptVault - Added bounded producer receipt
//! persistence, restart audit, exact-anchor policy reconstruction, and
//! adverse-evidence-preserving retention.
//! v2.8.59-FollowerEffectiveReadiness - Added composite follower readiness without changing legacy state.
//! v2.8.53-TypedCarrierCircuit - Added isolated certificate-carrier circuit scheduling counters.
//! v2.8.52-BlockCarrierCircuitTelemetry - Added anonymous carrier circuit scheduling counters.
//! v2.8.45-FollowerCertificateTelemetry - Added source-blind recovery outcome counters.
//! v2.8.18-TipSupersession - Added aggregate superseded announcement rounds.
//! v2.8.17-TipRetryQueue - Added bounded announcement retry outcome counters.
//! v2.2.0               - 🌟 Extracted from storage.rs; added get_embedding_model, get_overview
//! v2.3.0+RemoteStorage - 🌟 Added count_distinct_owners(), owner_exists()
//! v2.4.0-GraphCognition - 🌟 Added full CRUD for cognitive graph tables
//!   (now in storage_graph.rs) + Miner Step support (now in storage_miner.rs)
//! v2.4.0+Search        - 🌟 Split into storage_ops.rs / storage_graph.rs / storage_miner.rs
//! v2.5.3+Isolation     - 🌟 Added get_active_records_by_context() for /recall context filter
//! v2.7.0-BlockSync     - Added transactional signed commitment chain storage.
//! v2.7.1-BlockSyncStatus - Added bounded runtime follower lifecycle evidence.
//! v2.7.3-BlockAudit    - Added fail-closed startup audit for the complete chain.
//! v2.7.4-BlockIntegrityStatus - Added privacy-safe verified-chain evidence.
//! v2.7.5-CheckpointProof - Added signed checkpoint reconciliation evidence.
//! v2.7.6-EvidenceVault - Added durable bounded proof storage and startup audit.
//! v2.7.7-EvidenceRestartRecovery - Added real SQLite restart/migration tests.
//! v2.7.10-CheckpointDirectionIsolation - Isolated inbound service telemetry.
//! v2.7.11-CheckpointFreshness - Added durable proof age classification.
//! v2.7.12-WitnessRoundEvidence - Added bounded round evidence classification.
//! v2.7.13-CommitmentDurability - Enforced coordinator FULL SQLite commits.
//! v2.7.14-CommitmentTipAnchor - Added a signed local high-water rollback guard.
//! v2.7.17-AtomicBlockPage - Made verified peer pages one atomic SQLite append.
//! v2.7.18-VerifiedRangeSnapshot - Added fail-closed snapshot range serving.
//! v2.7.19-FollowerEvidenceRecovery - Deferred valid evidence above a rolled-back tip.
//! v2.7.20-WitnessEquivocation - Added durable trusted-witness conflict incidents.
//! v2.7.21-TrustedDivergenceHalt - Added sticky trusted-prefix incidents and
//!   an atomic local commitment-production halt.
//! v2.7.22-CheckpointCertificate - Added immutable threshold certificates.
//! v2.7.23-CertificateExchange - Added snapshot-audited bundle export.
//! v2.7.24-CertificateRollbackGuard - Detect local certificate-vault rollback.
//! v2.8.10-CoordinatorLease - Added durable exclusive witness lease grants.
//! v2.8.11-CoordinatorLeaseRelease - Added exact-instance graceful handover.
//! v2.8.12-LeaseFailClosedTelemetry - Added partition/recovery state evidence.
//!
//! ## Last Modified
//! [MEMCHAIN-COMMITMENT-CHAIN-SPLIT 2026-09-25 by Codex] Moved live
//! commitment-chain persistence, audit, atomic append, and range serving into
//! a focused extension module while retaining storage_ops compatibility paths.
//! [MEMCHAIN-COORDINATOR-AUTHORITY-SPLIT 2026-09-25 by Codex] Moved durable
//! coordinator authority, handover, lease fencing, and production gates into
//! a focused private extension module while retaining storage_ops APIs.
//! [MEMCHAIN-CHECKPOINT-EVIDENCE-SPLIT 2026-09-25 by Codex] Moved
//! canonical checkpoint-evidence validation, bounded incident retention, and
//! certificate reconstruction into a focused private extension module while
//! retaining all storage_ops-private call paths and public contracts.
//! [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] Isolated the complete
//! receipt-vault state machine behind a private transaction/snapshot owner.
//! [CUSTODY-WITNESS-FULL-DURABILITY 2026-09-24 by Codex] Refuse receipt
//! acknowledgement until the shared SQLite connection verifies FULL commits.
//! v2.8.65-CustodyWitnessReceiptImport - Reused one atomic vault transaction
//! for live receipts and explicitly time-bounded operator imports.
//! v2.8.64-CustodyWitnessReceiptVault - Persisted and re-audited exact signed
//! custody witness receipts before they can contribute to durable policy.
//! [FOLLOWER-POLICY-STARTUP-GATE 2026-08-14 by Codex] Added a fixed startup carrier-policy failure code.
//! [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Added source-blind,
//! follower-only authority-proof recovery and circuit runtime evidence.
//! [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Connected persisted
//! coordinator handovers to complete-chain and live proposer authorization.
//! v2.8.60-AnchorWorkerPrivacy - Redact Tokio panic payloads from signed
//! local-anchor persistence failures.
//! v2.8.57-CertificatePersistenceTruth - Distinguish verified-unpersisted follower outcomes.
//! v2.8.56-StickySecurityEvidence - Retain source-blind security-stop times across later success.
//! v2.8.55-CertificateBackfillTelemetry - Record atomic coordinator-only recovery aggregates.
//! v2.8.53-TypedCarrierCircuit - Kept block and certificate circuit telemetry independent.
//! v2.8.52-BlockCarrierCircuitTelemetry - Report anonymous cooling, skip, and half-open state.
//! v2.8.49-FollowerCertificateTipBinding - Prevent stale readiness after the audited tip advances.
//! v2.8.48-FollowerCertificateReadiness - Track exact follower certificate-policy readiness.
//! v2.8.31-FollowerCertificatePolicy - Validate retained certificates against current follower pins.
//! v2.8.18-TipSupersession - Prioritize fresh tips without delivery claims.
//! v2.8.17-TipRetryQueue - Track retry attempts, recoveries, and exhaustion.
//! v2.8.15-AnnouncementReceipts - Track exact coordinator tip-delivery outcomes.
//! v2.8.14-SyncObservability - Distinguish scheduled and authenticated follower wake-ups.
//! v2.8.13-BlockConfirmation - Expose audited witness-certificate block coverage.
//! v2.8.12-LeaseFailClosedTelemetry - Expose bounded fail-closed lease state.
//! v2.8.16-CustodyQuorumExpiry - Derive the exact aggregate validity horizon
//! for the newest accepted receipts sufficient to satisfy local custody policy.
//! v2.8.11-CoordinatorLeaseRelease - Retain epochs while releasing planned restarts.
//! v2.8.10-CoordinatorLease - Fence duplicate coordinators across witness hosts.
//! v2.7.23-CertificateExchange - Export only fully re-audited certificate frames.
//! v2.7.21-TrustedDivergenceHalt - Freeze production after trusted fork evidence.
//! v2.7.20-WitnessEquivocation - Retain trusted signed conflicts across restart and rotation.
//! v2.7.19-FollowerEvidenceRecovery - Allow audited followers to resync after local rollback.
//! v2.7.18-VerifiedRangeSnapshot - Serve only canonically reverified audit-backed pages.
//! v2.7.17-AtomicBlockPage - Share one atomic append path across mining and sync.
//! v2.7.11-CheckpointFreshness - Distinguish vault integrity from proof recency.
//! v2.7.12-WitnessRoundEvidence - Report bounded witness round coverage.
//! v2.7.10-CheckpointDirectionIsolation - Prevent requester-driven status pollution.
//! v2.7.4-BlockIntegrityStatus - Report and maintain the verified-chain baseline.
//! v2.7.5-CheckpointProof - Serve only audit-backed checkpoints and report outcomes.
//! v2.7.3-BlockAudit - Verify persisted blocks and indexes before networking starts.
//! v2.7.1-BlockSyncStatus - Privacy-safe follower status, failures, and recovery.
//! v2.7.0-BlockSync - Transactional commitment chain, ranges, and safe status.

use std::collections::HashMap;
use std::fs::{File, OpenOptions, Permissions};
use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nix::errno::Errno;
use nix::fcntl::{openat, Flock, FlockArg, OFlag};
use nix::sys::stat::Mode;
use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};
use tracing::{error, info, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{
    MemoryLayer, MemoryRecord, RecordCommitmentBlockV1, RecordCoordinatorHandoverV1,
    AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, GENESIS_PREV_HASH,
};
use aeronyx_core::protocol::memchain::{
    record_checkpoint_certificate_digest_v1, MAX_COORDINATOR_LEASE_TTL_SECS_V1,
    MIN_COORDINATOR_LEASE_TTL_SECS_V1,
};

use crate::error::RuntimeTaskJoinFailureKind;

// [MEMCHAIN-CHECKPOINT-EVIDENCE-SPLIT 2026-09-25 by Codex] Preserve the
// existing storage_ops-private helper paths while the vault owns validation.
pub use super::storage_checkpoint_evidence::RecordCommitmentCheckpointEvidenceAudit;
pub(crate) use super::storage_checkpoint_evidence::RecordCommitmentCheckpointEvidencePersistOutcome;
pub(super) use super::storage_checkpoint_evidence::{
    audit_checkpoint_evidence_connection, audit_checkpoint_evidence_snapshot,
    decode_checkpoint_evidence_claims, detect_trusted_checkpoint_equivocations,
    insert_trusted_checkpoint_divergence_incident,
};

// [MEMCHAIN-COORDINATOR-AUTHORITY-SPLIT 2026-09-25 by Codex] Keep stable
// storage_ops paths while the coordinator authority/lease boundary owns the
// implementation and its transaction-scoped validation helpers.
pub(super) use super::storage_coordinator_authority::{
    acquire_commitment_coordinator_fence, commitment_coordinator_fence_path,
    commitment_witness_lease_clock_path, read_record_coordinator_handover_history_transaction,
    record_commitment_authority_at_height, verify_record_commitment_proposer_history_transaction,
    WitnessLeaseCommitObservation,
};
pub use super::storage_coordinator_authority::{
    RecordCommitmentAuthorityState, RecordCoordinatorHandoverPage,
    RecordCoordinatorHandoverPersistOutcome, RecordCoordinatorLeaseGrantOutcome,
    RecordCoordinatorLeaseReleaseOutcome,
};

// [MEMCHAIN-COMMITMENT-CHAIN-SPLIT 2026-09-25 by Codex] Keep all existing
// storage_ops type/constant paths stable while the live chain owns the code.
pub use super::storage_commitment_chain::{
    RecordCommitmentAppendOutcome, RecordCommitmentBatchAppendOutcome, RecordCommitmentChainAudit,
    VerifiedRecordCommitmentBlockPage,
};
pub(super) use super::storage_commitment_chain::{
    MAX_ATOMIC_COMMITMENT_BLOCK_BATCH, MAX_STORED_COMMITMENT_BLOCK_BYTES,
};

// [CUSTODY-WITNESS-VAULT-MODULE 2026-09-24 by Codex] Preserve existing
// storage_ops type/function import paths while the private module owns them.
pub use super::storage_witness_receipts::{
    custody_witness_renewal_warning_window_secs, CustodyAuditWitnessPolicyReadiness,
    CustodyAuditWitnessReadinessError, CustodyAuditWitnessReceiptPersistOutcome,
    CustodyAuditWitnessReceiptPolicyEvidence, CustodyAuditWitnessReceiptReadinessSnapshot,
    CustodyAuditWitnessReceiptVaultAudit,
};

use super::storage::{
    probe_storage_database_file_identity, LayerCounts, MemoryStorage,
    RecordCommitmentAnnouncementDisposition, RecordCommitmentAuthoritySyncDisposition,
    RecordCommitmentBlockPagePullDisposition, RecordCommitmentCertificateBackfillDisposition,
    RecordCommitmentCertificatePolicyReadiness, RecordCommitmentCertificateSyncDisposition,
    RecordCommitmentCheckpointCertificateAnchorConfig,
    RecordCommitmentCheckpointCertificateAnchorRuntime,
    RecordCommitmentCheckpointCertificateBundle, RecordCommitmentCheckpointStatus,
    RecordCommitmentFollowerReadiness, RecordCommitmentIntegrityRuntime, RecordCommitmentSyncEvent,
    RecordCommitmentSyncRuntime, RecordCommitmentSyncStatus, RecordCommitmentTipAnchorConfig,
    RecordCommitmentWitnessLeaseClockRuntime, RecordCommitmentWitnessLeaseHold,
    StorageDatabaseFileIdentity, StorageStats, CHECKPOINT_CERTIFICATE_CAPACITY,
    CHECKPOINT_EVIDENCE_CAPACITY, CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
    COMMITMENT_SYNC_EVENT_CAPACITY, MAX_CHECKPOINT_CERTIFICATE_SIGNERS,
    MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES,
};
use super::storage_crypto::{decrypt_record_content, encrypt_record_content};

// [MEMCHAIN-COMMITMENT-SYNC-STATUS-SPLIT 2026-09-25 by Codex] Preserve
// existing helper paths while the child module owns aggregate sync status.
#[path = "storage_commitment_sync_status.rs"]
mod storage_commitment_sync_status;
use storage_commitment_sync_status::{
    privacy_safe_sync_error_code, record_commitment_follower_readiness,
    record_monotonic_observation,
};

// ============================================

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod checkpoint_certificate;
mod checkpoint_evidence;
mod commitment_runtime;
mod extended_queries;
mod tip_anchor;
mod witness_anchor;

use checkpoint_certificate::checkpoint_certificate_anchor_path;
use checkpoint_certificate::checkpoint_certificate_anchor_signing_bytes;
use checkpoint_certificate::decode_checkpoint_certificate_anchor_hex;
use checkpoint_certificate::persist_checkpoint_certificate_anchor;
use checkpoint_certificate::read_checkpoint_certificate_anchor;
use checkpoint_certificate::read_latest_checkpoint_certificate_anchor_state;
use checkpoint_certificate::write_checkpoint_certificate_anchor_atomic;
use checkpoint_evidence::checkpoint_observation_freshness;
use checkpoint_evidence::checkpoint_witness_round_state;
use checkpoint_evidence::commitment_block_confirmation_state;
pub(super) use commitment_runtime::ensure_full_sqlite_durability;
pub(super) use commitment_runtime::read_sqlite_durability;
use tip_anchor::decode_fixed_hex;
pub(super) use tip_anchor::persist_record_commitment_tip_anchor;
use tip_anchor::read_record_commitment_tip_anchor;
pub(super) use tip_anchor::read_record_commitment_tip_transaction;
use tip_anchor::read_signed_local_anchor_bytes;
use tip_anchor::record_commitment_tip_anchor_signing_bytes;
use tip_anchor::run_blocking_local_anchor_write;
pub(super) use tip_anchor::unix_now_secs;
use tip_anchor::write_record_commitment_tip_anchor_atomic;
use tip_anchor::write_signed_local_anchor_atomic;

// Overview Types (v2.2.0)
// ============================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct OverviewRecord {
    pub record_id: String,
    pub content: String,
    pub topic_tags: Vec<String>,
    pub timestamp: u64,
    pub access_count: u32,
    pub positive_feedback: u32,
    pub negative_feedback: u32,
    pub source_ai: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct OverviewData {
    pub by_layer: HashMap<String, u64>,
    pub recent_by_layer: HashMap<String, Vec<OverviewRecord>>,
    pub last_memory_at: u64,
}

/// Result of one serialized opaque monotonic anchor witness decision.
///
/// Every variant returns the witness's durable high-water state. Callers must
/// treat `Stale`, `Conflict`, and `Gap` as evidence that the requester cannot
/// safely claim continuity; none of those outcomes mutates the high-water row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicAnchorWitnessOutcome {
    /// First observation, or an exact one-generation advance, was committed.
    Advanced {
        generation: u64,
        anchor_digest: [u8; 32],
    },
    /// The same generation and digest were already durable.
    Idempotent {
        generation: u64,
        anchor_digest: [u8; 32],
    },
    /// The request is older than the durable high-water mark.
    Stale {
        generation: u64,
        anchor_digest: [u8; 32],
    },
    /// The request reused the durable generation with another digest.
    Conflict {
        generation: u64,
        anchor_digest: [u8; 32],
    },
    /// The request skipped at least one generation after witness bootstrap.
    Gap {
        generation: u64,
        anchor_digest: [u8; 32],
    },
}

/// Backward-compatible name for delivery-cache witness decisions.
pub type VerifiedDeliveryAnchorWitnessOutcome = MonotonicAnchorWitnessOutcome;

/// Custody-checkpoint witness decision in its independent durable namespace.
pub type CustodyAuditAnchorWitnessOutcome = MonotonicAnchorWitnessOutcome;

#[derive(Debug, Clone, Copy)]
enum MonotonicAnchorWitnessNamespace {
    VerifiedDelivery,
    CustodyAudit,
}

impl MonotonicAnchorWitnessNamespace {
    const fn label(self) -> &'static str {
        match self {
            Self::VerifiedDelivery => "verified-delivery",
            Self::CustodyAudit => "custody-audit",
        }
    }

    const fn select_sql(self) -> &'static str {
        match self {
            Self::VerifiedDelivery => {
                "SELECT generation, anchor_digest
                 FROM verified_delivery_anchor_witnesses WHERE requester=?1"
            }
            Self::CustodyAudit => {
                "SELECT generation, frame_sha256
                 FROM custody_audit_anchor_witnesses WHERE producer=?1"
            }
        }
    }

    const fn refresh_sql(self) -> &'static str {
        match self {
            Self::VerifiedDelivery => {
                "UPDATE verified_delivery_anchor_witnesses
                 SET observed_at=MAX(observed_at, ?2) WHERE requester=?1"
            }
            Self::CustodyAudit => {
                "UPDATE custody_audit_anchor_witnesses
                 SET observed_at=MAX(observed_at, ?2) WHERE producer=?1"
            }
        }
    }

    const fn advance_sql(self) -> &'static str {
        match self {
            Self::VerifiedDelivery => {
                "UPDATE verified_delivery_anchor_witnesses
                 SET generation=?2, anchor_digest=?3, observed_at=?4
                 WHERE requester=?1"
            }
            Self::CustodyAudit => {
                "UPDATE custody_audit_anchor_witnesses
                 SET generation=?2, frame_sha256=?3, observed_at=?4
                 WHERE producer=?1"
            }
        }
    }

    const fn insert_sql(self) -> &'static str {
        match self {
            Self::VerifiedDelivery => {
                "INSERT INTO verified_delivery_anchor_witnesses
                 (requester, generation, anchor_digest, observed_at)
                 VALUES (?1, ?2, ?3, ?4)"
            }
            Self::CustodyAudit => {
                "INSERT INTO custody_audit_anchor_witnesses
                 (producer, generation, frame_sha256, observed_at)
                 VALUES (?1, ?2, ?3, ?4)"
            }
        }
    }
}

/// Aggregate local state for the node-blind commitment chain.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecordCommitmentChainStatus {
    /// Stable wire contract name.
    pub contract_version: &'static str,
    /// Production chain identifier as lowercase hexadecimal.
    pub chain_id: String,
    /// Number of verified blocks stored locally.
    pub block_count: u64,
    /// Number of opaque record commitments represented by those blocks.
    pub commitment_count: u64,
    /// Current one-based tip height, or zero when empty.
    pub tip_height: u64,
    /// Current tip hash as lowercase hexadecimal, or `None` when empty.
    pub tip_hash: Option<String>,
    /// Privacy contract exposed to operators and API consumers.
    pub payload_policy: &'static str,
    /// Runtime evidence for the last complete chain audit and verified appends.
    pub integrity: RecordCommitmentChainIntegrityStatus,
    /// Aggregate signed checkpoint reconciliation evidence.
    pub checkpoint: RecordCommitmentCheckpointStatus,
}

/// Privacy-safe runtime integrity evidence for the commitment chain.
///
/// `verified` means this process completed a full snapshot-consistent audit
/// and every later tip advance used the same atomic validation path. A process
/// restart, failed re-audit, or unexpected tip transition resets the state to
/// `not_verified`; persisted data is never silently repaired.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RecordCommitmentChainIntegrityStatus {
    /// Stable API/heartbeat contract name.
    pub contract_version: &'static str,
    /// `verified` or `not_verified`.
    pub state: &'static str,
    /// Time the complete persisted-chain baseline was established.
    pub baseline_verified_at: Option<u64>,
    /// Time of the most recent full audit or verified atomic append.
    pub last_verified_at: Option<u64>,
    /// Wall-clock duration of the baseline audit.
    pub verification_duration_ms: Option<u64>,
    /// Number of blocks covered by the current verified baseline.
    pub verified_block_count: u64,
    /// Number of opaque commitments covered by the current verified baseline.
    pub verified_commitment_count: u64,
    /// Last verified one-based height, or zero for an empty verified chain.
    pub verified_tip_height: u64,
    /// Effective SQLite durability mode: `off`, `normal`, `full`, or `extra`.
    pub durability_mode: &'static str,
    /// Local coordinator production fence: `unconfigured`, `not_required`,
    /// `isolated_in_memory`, `held`, `contended`, or `failed`.
    pub coordinator_fence_state: &'static str,
    /// Most recent successful OS lock acquisition in this process.
    pub coordinator_fence_acquired_at: Option<u64>,
    /// Failed or contended acquisition attempts in this process.
    pub coordinator_fence_acquisition_failures_total: u64,
    /// Exact local-only protection boundary; never implies distributed lease.
    pub coordinator_fence_scope: &'static str,
    /// Witness-backed cross-host lease state.
    pub coordinator_lease_state: &'static str,
    /// Grants returned by the most recent lease round.
    pub coordinator_lease_granted_witnesses: usize,
    /// Number of operator-pinned witnesses required for production.
    pub coordinator_lease_required_witnesses: usize,
    /// Conservative local production-authority deadline.
    pub coordinator_lease_expires_at: Option<u64>,
    /// Monotonic seconds of production authority remaining, zero when expired.
    pub coordinator_lease_seconds_remaining: Option<u64>,
    /// Whether the lease gate alone currently permits coordinator production.
    pub coordinator_lease_production_permitted: bool,
    /// Most recent all-witness lease round attempt.
    pub coordinator_lease_last_attempted_at: Option<u64>,
    /// Most recent successful all-witness lease round.
    pub coordinator_lease_last_renewed_at: Option<u64>,
    /// Most recent incomplete or failed all-witness lease round.
    pub coordinator_lease_last_failure_at: Option<u64>,
    /// Failed lease rounds in this process lifetime.
    pub coordinator_lease_renewal_failures_total: u64,
    /// Consecutive failed rounds since the latest complete grant.
    pub coordinator_lease_consecutive_failures: u64,
    /// Degraded or expired periods recovered by a later complete grant.
    pub coordinator_lease_recoveries_total: u64,
    /// Exact anti-overclaim boundary for the witness lease mechanism.
    pub coordinator_lease_scope: &'static str,
    /// Local signed high-water guard state. Never contains the anchor path,
    /// signer, signature, or block hash.
    pub rollback_guard_state: &'static str,
    /// Highest commitment height covered by the local signed guard.
    pub rollback_guard_height: u64,
    /// Last time the sidecar signature and ancestry were verified.
    pub rollback_guard_last_verified_at: Option<u64>,
    /// Last time a new sidecar value was durably persisted.
    pub rollback_guard_last_persisted_at: Option<u64>,
    /// Process-lifetime count of failed atomic sidecar writes.
    pub rollback_guard_write_failures_total: u64,
    /// Explicit anti-overclaim boundary for this local mechanism.
    pub rollback_guard_scope: &'static str,
    /// Explicit scope and privacy boundary for operators.
    pub verification_policy: &'static str,
}

/// A V1 handover proof is fixed-size and currently below 400 bytes. The
/// storage bound rejects corrupted or future incompatible payloads before
/// deserialization while leaving room for bincode representation details.
pub(super) const MAX_STORED_COORDINATOR_HANDOVER_BYTES: usize = 1024;
/// Authority history is never pruned because cold followers need every epoch.
/// Bound one in-memory audit to prevent a replaced SQLite file from forcing an
/// unbounded allocation; a protocol-version migration is required beyond it.
pub(super) const MAX_COORDINATOR_HANDOVER_HISTORY: usize = 4096;
/// A v1 tip anchor is under 1 KiB. Keep disk reads bounded before JSON decode.
const MAX_COMMITMENT_TIP_ANCHOR_BYTES: u64 = 4 * 1024;
const COMMITMENT_TIP_ANCHOR_CONTRACT: &str = "record_commitment_tip_anchor.v1";
const COMMITMENT_TIP_ANCHOR_DOMAIN: &[u8] = b"aeronyx.record_commitment_tip_anchor.v1\0";
const MAX_CHECKPOINT_CERTIFICATE_ANCHOR_BYTES: u64 = 4 * 1024;
const CHECKPOINT_CERTIFICATE_ANCHOR_CONTRACT: &str = "record_checkpoint_certificate_anchor.v1";
const CHECKPOINT_CERTIFICATE_ANCHOR_DOMAIN: &[u8] =
    b"aeronyx.record_checkpoint_certificate_anchor.v1\0";
/// Prevent immediate lease takeover at the exact wall-clock expiry boundary.
pub(super) const COORDINATOR_LEASE_HANDOVER_GRACE_SECS: u64 = 15;
pub(super) const WITNESS_LEASE_RESTART_HOLD_SECS: u64 =
    MAX_COORDINATOR_LEASE_TTL_SECS_V1 as u64 + COORDINATOR_LEASE_HANDOVER_GRACE_SECS;
static SIGNED_LOCAL_ANCHOR_TEMP_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordCommitmentTipAnchorV1 {
    contract_version: String,
    chain_id: String,
    tip_height: u64,
    tip_hash: String,
    signer: String,
    updated_at: u64,
    signature: String,
}

impl RecordCommitmentTipAnchorV1 {
    fn new_signed(
        tip_height: u64,
        tip_hash: [u8; 32],
        identity: &IdentityKeyPair,
        updated_at: u64,
    ) -> Self {
        let signer = identity.public_key_bytes();
        let signing_bytes = record_commitment_tip_anchor_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            tip_height,
            &tip_hash,
            &signer,
            updated_at,
        );
        Self {
            contract_version: COMMITMENT_TIP_ANCHOR_CONTRACT.to_string(),
            chain_id: hex::encode(AERONYX_MEMCHAIN_MAINNET_CHAIN_ID),
            tip_height,
            tip_hash: hex::encode(tip_hash),
            signer: hex::encode(signer),
            updated_at,
            signature: hex::encode(identity.sign(&signing_bytes)),
        }
    }

    fn verify(&self, expected_signer: &[u8; 32]) -> Result<VerifiedCommitmentTipAnchor, String> {
        if self.contract_version != COMMITMENT_TIP_ANCHOR_CONTRACT {
            return Err("commitment tip anchor contract is unsupported".to_string());
        }
        let chain_id = decode_fixed_hex::<32>(&self.chain_id, "chain id")?;
        if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
            return Err("commitment tip anchor chain id is invalid".to_string());
        }
        let tip_hash = decode_fixed_hex::<32>(&self.tip_hash, "tip hash")?;
        if self.tip_height == 0 && tip_hash != GENESIS_PREV_HASH {
            return Err("commitment tip anchor genesis hash is invalid".to_string());
        }
        let signer = decode_fixed_hex::<32>(&self.signer, "signer")?;
        if &signer != expected_signer {
            return Err("commitment tip anchor signer does not match this node".to_string());
        }
        let signature = decode_fixed_hex::<64>(&self.signature, "signature")?;
        let signing_bytes = record_commitment_tip_anchor_signing_bytes(
            &chain_id,
            self.tip_height,
            &tip_hash,
            &signer,
            self.updated_at,
        );
        IdentityPublicKey::from_bytes(&signer)
            .and_then(|key| key.verify(&signing_bytes, &signature))
            .map_err(|_| "commitment tip anchor signature is invalid".to_string())?;
        Ok(VerifiedCommitmentTipAnchor {
            tip_height: self.tip_height,
            tip_hash,
            updated_at: self.updated_at,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct VerifiedCommitmentTipAnchor {
    tip_height: u64,
    tip_hash: [u8; 32],
    updated_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CheckpointCertificateAnchorState {
    certificate_height: u64,
    checkpoint_hash: [u8; 32],
    certificate_digest: [u8; 32],
    required_signers: u64,
    signer_count: u64,
}

impl CheckpointCertificateAnchorState {
    const EMPTY: Self = Self {
        certificate_height: 0,
        checkpoint_hash: GENESIS_PREV_HASH,
        certificate_digest: [0u8; 32],
        required_signers: 0,
        signer_count: 0,
    };
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordCheckpointCertificateAnchorV1 {
    contract_version: String,
    chain_id: String,
    certificate_height: u64,
    checkpoint_hash: String,
    certificate_digest: String,
    required_signers: u64,
    signer_count: u64,
    signer: String,
    updated_at: u64,
    signature: String,
}

impl RecordCheckpointCertificateAnchorV1 {
    fn new_signed(
        state: CheckpointCertificateAnchorState,
        identity: &IdentityKeyPair,
        updated_at: u64,
    ) -> Self {
        let signer = identity.public_key_bytes();
        let signing_bytes = checkpoint_certificate_anchor_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            state,
            &signer,
            updated_at,
        );
        Self {
            contract_version: CHECKPOINT_CERTIFICATE_ANCHOR_CONTRACT.to_string(),
            chain_id: hex::encode(AERONYX_MEMCHAIN_MAINNET_CHAIN_ID),
            certificate_height: state.certificate_height,
            checkpoint_hash: hex::encode(state.checkpoint_hash),
            certificate_digest: hex::encode(state.certificate_digest),
            required_signers: state.required_signers,
            signer_count: state.signer_count,
            signer: hex::encode(signer),
            updated_at,
            signature: hex::encode(identity.sign(&signing_bytes)),
        }
    }

    fn verify(
        &self,
        expected_signer: &[u8; 32],
    ) -> Result<VerifiedCheckpointCertificateAnchor, String> {
        if self.contract_version != CHECKPOINT_CERTIFICATE_ANCHOR_CONTRACT {
            return Err("checkpoint certificate anchor contract is unsupported".to_string());
        }
        let chain_id = decode_checkpoint_certificate_anchor_hex::<32>(&self.chain_id, "chain id")?;
        if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
            return Err("checkpoint certificate anchor chain id is invalid".to_string());
        }
        let checkpoint_hash = decode_checkpoint_certificate_anchor_hex::<32>(
            &self.checkpoint_hash,
            "checkpoint hash",
        )?;
        let certificate_digest = decode_checkpoint_certificate_anchor_hex::<32>(
            &self.certificate_digest,
            "certificate digest",
        )?;
        let state = CheckpointCertificateAnchorState {
            certificate_height: self.certificate_height,
            checkpoint_hash,
            certificate_digest,
            required_signers: self.required_signers,
            signer_count: self.signer_count,
        };
        if state.certificate_height == 0 {
            if state != CheckpointCertificateAnchorState::EMPTY {
                return Err("checkpoint certificate anchor empty state is invalid".to_string());
            }
        } else if !(2..=MAX_CHECKPOINT_CERTIFICATE_SIGNERS as u64).contains(&state.required_signers)
            || state.signer_count < state.required_signers
            || state.signer_count > MAX_CHECKPOINT_CERTIFICATE_SIGNERS as u64
        {
            return Err("checkpoint certificate anchor signer metadata is invalid".to_string());
        }
        let signer = decode_checkpoint_certificate_anchor_hex::<32>(&self.signer, "signer")?;
        if &signer != expected_signer {
            return Err(
                "checkpoint certificate anchor signer does not match this node".to_string(),
            );
        }
        let signature =
            decode_checkpoint_certificate_anchor_hex::<64>(&self.signature, "signature")?;
        let signing_bytes =
            checkpoint_certificate_anchor_signing_bytes(&chain_id, state, &signer, self.updated_at);
        IdentityPublicKey::from_bytes(&signer)
            .and_then(|key| key.verify(&signing_bytes, &signature))
            .map_err(|_| "checkpoint certificate anchor signature is invalid".to_string())?;
        Ok(VerifiedCheckpointCertificateAnchor {
            state,
            updated_at: self.updated_at,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct VerifiedCheckpointCertificateAnchor {
    state: CheckpointCertificateAnchorState,
    updated_at: u64,
}

/// Reads only the latest certificate metadata after a complete vault audit.
///
/// The returned digest and hash are private coordinator material used to bind
/// the signed local sidecar. Callers must never log or serialize this value.
type LatestCheckpointCertificateAnchorRow = (i64, Vec<u8>, Vec<u8>, Vec<u8>, i64, i64);

// ============================================
// impl MemoryStorage — Statistics
// ============================================

// ============================================
// impl MemoryStorage — Miner Support
// ============================================

// ============================================
// impl MemoryStorage — Content Dedup
// ============================================

// ============================================
// impl MemoryStorage — v2.2.0 MemExplorer
// ============================================

// ============================================
// impl MemoryStorage — v2.3.0 Remote Storage
// ============================================

// ============================================
// impl MemoryStorage — v2.5.3+Isolation: Context Filter
// ============================================

// ============================================
// Tests
// ============================================
// v2.4.0+Search: Cognitive graph tests moved to storage_graph.rs.
//   Miner step support tests moved to storage_miner.rs.

#[cfg(test)]
mod tests {
    mod commitment_anchor;
    mod extended_queries;
    mod other;

    use super::*;
    // [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] File-migration fixtures
    // must follow the authoritative latest schema while each migration step
    // continues to advance through hardcoded intermediate versions.
    use super::super::storage::SCHEMA_VERSION;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::ledger::MemoryRecord;
    use aeronyx_core::protocol::memchain::{
        encode_memchain, record_chain_checkpoint_response_signing_bytes, MemChainMessage,
        MEMCHAIN_MAGIC,
    };
    use std::process::{ExitStatus, Stdio};
    use tempfile::TempDir;
    use tokio::io::AsyncReadExt;

    fn signed_commitment_block(
        height: u64,
        previous_hash: [u8; 32],
        record_byte: u8,
        identity: &IdentityKeyPair,
    ) -> RecordCommitmentBlockV1 {
        RecordCommitmentBlockV1::new_signed(
            height,
            1_700_600_000u64.saturating_add(height),
            previous_hash,
            vec![[record_byte; 32]],
            identity,
        )
    }

    fn make_rec_owner(ts: u64, owner: [u8; 32], layer: MemoryLayer) -> MemoryRecord {
        MemoryRecord::new(
            owner,
            ts,
            layer,
            vec!["test".into()],
            "ai".into(),
            format!("content_{}", ts).into_bytes(),
            vec![0.5; 4],
        )
    }

    fn signed_checkpoint_response_frame(
        identity: &IdentityKeyPair,
        request_id: [u8; 16],
        timestamp: u64,
        checkpoint_height: u64,
        checkpoint_hash: [u8; 32],
        tip_height: u64,
        tip_hash: [u8; 32],
    ) -> Vec<u8> {
        let responder = identity.public_key_bytes();
        let signing_bytes = record_chain_checkpoint_response_signing_bytes(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            &request_id,
            &responder,
            timestamp,
            checkpoint_height,
            &checkpoint_hash,
            tip_height,
            &tip_hash,
        );
        encode_memchain(&MemChainMessage::RecordChainCheckpointResponseV1 {
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            request_id,
            responder,
            response_timestamp: timestamp,
            checkpoint_height,
            checkpoint_hash,
            tip_height,
            tip_hash,
            signature: identity.sign(&signing_bytes),
        })
        .unwrap()
    }

    async fn seed_checkpoint_evidence(
        storage: &MemoryStorage,
        observed_at: u64,
    ) -> (RecordCommitmentBlockV1, Vec<u8>, [u8; 32]) {
        let proposer = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            observed_at.saturating_sub(1),
            GENESIS_PREV_HASH,
            vec![[0xA7; 32]],
            &proposer,
        );
        storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();

        let responder = IdentityKeyPair::generate();
        let frame = signed_checkpoint_response_frame(
            &responder,
            [0xA8; 16],
            observed_at,
            1,
            block.hash(),
            1,
            block.hash(),
        );
        let digest: [u8; 32] = Sha256::digest(&frame).into();
        storage
            .persist_record_commitment_checkpoint_evidence(
                observed_at,
                "converged",
                1,
                1,
                1,
                &digest,
                &frame,
            )
            .await
            .unwrap();
        storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        (block, frame, digest)
    }

    async fn seed_checkpoint_certificate(
        storage: &MemoryStorage,
        observed_at: u64,
    ) -> ([[u8; 32]; 2], [[u8; 32]; 2]) {
        let proposer = IdentityKeyPair::generate();
        let block = signed_commitment_block(1, GENESIS_PREV_HASH, 0xB1, &proposer);
        storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();

        let witnesses = [IdentityKeyPair::generate(), IdentityKeyPair::generate()];
        let mut digests = [[0u8; 32]; 2];
        for (index, witness) in witnesses.iter().enumerate() {
            let frame = signed_checkpoint_response_frame(
                witness,
                [0xC0u8.saturating_add(index as u8); 16],
                observed_at.saturating_add(index as u64),
                1,
                block.hash(),
                1,
                block.hash(),
            );
            digests[index] = Sha256::digest(&frame).into();
            storage
                .persist_record_commitment_checkpoint_evidence_with_witness_policy(
                    observed_at.saturating_add(index as u64),
                    "converged",
                    1,
                    1,
                    1,
                    &digests[index],
                    &frame,
                    true,
                )
                .await
                .unwrap();
        }
        (
            [
                witnesses[0].public_key_bytes(),
                witnesses[1].public_key_bytes(),
            ],
            digests,
        )
    }

    const WITNESS_LEASE_PROCESS_STAGE_ENV: &str = "AERONYX_TEST_WITNESS_LEASE_PROCESS_STAGE";
    const WITNESS_LEASE_PROCESS_DB_ENV: &str = "AERONYX_TEST_WITNESS_LEASE_PROCESS_DB";
    // [ARCH-SPLIT-VERIFY 2026-10-02 by Codex] --exact needs the moved worker's full module path.
    const WITNESS_LEASE_PROCESS_WORKER: &str = concat!(
        "services::memchain::storage_ops::tests::commitment_anchor::",
        "test_witness_lease_cross_process_worker"
    );
    const WITNESS_LEASE_PROCESS_CRASH_EXIT_CODE: i32 = 74;
    const WITNESS_LEASE_PROCESS_COMMITTED_SIGNAL: &str = "witness-lease-committed";

    struct WitnessLeaseChildOutput {
        status: ExitStatus,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    }

    async fn run_witness_lease_child(stage: &str, db_path: &Path) -> WitnessLeaseChildOutput {
        // [MEMCHAIN-WITNESS-CLOCK-PROCESS 2026-09-05 by Codex] Re-execute the
        // current test binary so no process-local mutex, Instant, or flock
        // handle is shared with the parent. The deadline is supervision only;
        // lease correctness never depends on elapsed sleep time.
        let executable = std::env::current_exe().expect("resolve witness lease test binary");
        let mut command = crate::isolated_child_command(executable);
        command
            .arg(WITNESS_LEASE_PROCESS_WORKER)
            .arg("--exact")
            .arg("--ignored")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(WITNESS_LEASE_PROCESS_STAGE_ENV, stage)
            .env(WITNESS_LEASE_PROCESS_DB_ENV, db_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = command.spawn().expect("spawn witness lease child");
        let mut stdout = child.stdout.take().expect("capture witness child stdout");
        let mut stderr = child.stderr.take().expect("capture witness child stderr");
        let stdout_reader = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stdout
                .read_to_end(&mut bytes)
                .await
                .expect("read witness child stdout");
            bytes
        });
        let stderr_reader = tokio::spawn(async move {
            let mut bytes = Vec::new();
            stderr
                .read_to_end(&mut bytes)
                .await
                .expect("read witness child stderr");
            bytes
        });

        let status = match tokio::time::timeout(Duration::from_secs(20), child.wait()).await {
            Ok(result) => result.expect("wait for witness lease child"),
            Err(_) => {
                child.kill().await.expect("kill timed-out witness child");
                let status = child.wait().await.expect("reap timed-out witness child");
                let stdout = stdout_reader
                    .await
                    .expect("join timed-out witness stdout reader");
                let stderr = stderr_reader
                    .await
                    .expect("join timed-out witness stderr reader");
                panic!(
                    "witness child {stage} exceeded deadline with status {:?}\nstdout:\n{}\nstderr:\n{}",
                    status.code(),
                    String::from_utf8_lossy(&stdout),
                    String::from_utf8_lossy(&stderr)
                );
            }
        };
        WitnessLeaseChildOutput {
            status,
            stdout: stdout_reader.await.expect("join witness stdout reader"),
            stderr: stderr_reader.await.expect("join witness stderr reader"),
        }
    }

    fn assert_witness_lease_child_succeeded(stage: &str, output: &WitnessLeaseChildOutput) {
        assert!(
            output.status.success(),
            "witness child {stage} failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    async fn commitment_audit_fixture() -> (MemoryStorage, RecordCommitmentBlockV1) {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let identity = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_250_001,
            GENESIS_PREV_HASH,
            vec![[0x41; 32], [0x42; 32]],
            &identity,
        );
        storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        (storage, block)
    }
}
