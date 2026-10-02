// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage.rs
// ============================================
//! # MemoryStorage — SQLite Core (Schema, CRUD, LRU Cache)
//!
//! ## Creation Reason
//! Core persistent storage layer for MemChain. Provides SQLite-backed
//! memory record storage with optional ChaCha20 encryption, LRU caching,
//! and schema migration support.
//!
//! ## Split Structure (v2.2.0)
//! storage.rs was split into 3 files for maintainability:
//! - `storage.rs` (THIS FILE) — struct, open, schema, migration, core CRUD, LRU
//! - `storage_crypto.rs` — all encryption/decryption functions
//! - `storage_ops.rs` — rawlog, feedback, chain state, stats, miner, overview
//!
//! All 3 files impl on the same `MemoryStorage` struct.
//! External API is unchanged — mod.rs re-exports everything.
//!
//! ## Schema History
//! - v1: Initial schema (records only)
//! - v2: Added embedding column to records
//! - v4: Added positive_feedback, negative_feedback, conflict_with to records
//!        Added memory_edges, user_weights, memory_feedback, chain_state tables
//! - v5 (v2.4.0-GraphCognition): Three-layer cognitive graph schema:
//!   - episodes: Episode layer (complete original conversations, non-lossy)
//!   - entities: Semantic Entity layer nodes (GLiNER-extracted)
//!   - knowledge_edges: Semantic Entity layer edges (with temporal validity)
//!   - episode_edges: Bridge layer (Episode ↔ Entity bidirectional links)
//!   - communities: Community layer (label propagation auto-clustering)
//!   - projects: Project table (Community specialization for code projects)
//!   - sessions: Conversation session metadata + summaries
//!   - artifacts: Code/document artifacts with version chains
//!   - records ALTER: added project_id, session_id, episode_id columns
//!   - memory_edges data migrated to knowledge_edges (relation_type = 'RELATED_TO')
//! - v6 (v2.5.0-SuperNode): Async cognitive task queue + LLM usage tracking:
//!   - cognitive_tasks: SuperNode async LLM task queue
//!   - llm_usage_log: Per-call token counts and latency log
//!   - sessions ALTER: added title column (SuperNode-generated session title)
//! - v7: Added the node-blind marker to client-sealed records.
//! - v8 (v2.7.0-BlockSync): Authoritative commitment chain tables:
//!   - record_commitment_blocks: signed block payload and verified chain data
//!   - record_block_commitments: unique record ID to block-height membership
//! - v9 (v2.7.6-EvidenceVault): Bounded local checkpoint proof evidence:
//!   - record_checkpoint_evidence: exact signature-verified peer response frames
//!     retained locally for restart-safe operator audit; never exposed raw
//! - v10 (v2.7.20-WitnessEquivocation): Durable same-height double-sign proof.
//! - v11 (v2.7.21-TrustedDivergenceHalt): Durable operator-pinned witness
//!   divergence incidents and a process-local fail-closed production latch.
//! - v12 (v2.7.22-CheckpointCertificate): Immutable bounded certificates that
//!   bind distinct operator-pinned witness frames to one locally audited
//!   checkpoint without claiming distributed consensus or fork choice.
//! - v13 (v2.7.26-CoordinatorLease): Durable exclusive coordinator process
//!   lease grants retained by audited follower witnesses.
//! - v14 (v2.8.28-VerifiedDeliveryAnchorWitness): One bounded durable
//!   generation-and-digest high-water row per admitted node identity.
//! - v15 (v2.8.14-CoordinatorHandover): Replayable dual-signed coordinator
//!   authority history.
//! - v16 (v2.8.29-CustodyAuditWitness): Separate producer-scoped custody
//!   checkpoint high-water rows containing exact opaque frame digests only.
//! - v17 (v2.8.64-CustodyWitnessReceiptVault): Bounded immutable producer-side
//!   receipt frames for restart-safe signature and policy revalidation.
//! - v18 (v2.8.65-CustodyWitnessReceiptImport): Explicit per-row admission
//!   evidence keeps live transport at 60 seconds while permitting bounded,
//!   operator-approved air-gapped receipt import without weakening re-audit.
//! - v2.7.23-CertificateExchange: Snapshot-audited internal bundle export for
//!   the admitted fixed-size peer protocol; the schema remains v12.
//! - v2.7.4-BlockIntegrityStatus: Runtime-only evidence for the most recent
//!   complete persisted-chain audit and subsequently verified appends.
//! - v2.7.5-CheckpointProof: Runtime-only signed checkpoint reconciliation
//!   evidence without peer identities, hashes, signatures, or user metadata.
//! - v2.7.10-CheckpointDirectionIsolation: Inbound checkpoint serving updates
//!   service counters only and cannot overwrite outbound verification evidence.
//! - v2.7.11-CheckpointFreshness: Separates durable vault integrity from the
//!   age of the most recent signature-verified outbound observation.
//! - v2.7.12-WitnessRoundEvidence: Reports the latest bounded witness-round
//!   coverage and result without treating peer count as consensus.
//! - v2.7.13-CommitmentDurability: Tracks the effective SQLite synchronous
//!   level so a commitment coordinator can fail closed unless WAL commits use
//!   FULL-or-stronger durability before startup audit and block production.
//! - v2.7.19-FollowerEvidenceRecovery: Separates cryptographically valid
//!   checkpoint frames from evidence that is applicable to the currently
//!   audited local tip, allowing a follower to recover from a local rollback
//!   without presenting historical evidence as current convergence.
//!
//! ## Thread Safety
//! `rusqlite::Connection` behind `tokio::sync::Mutex`. Phase 2+ can use r2d2 pooling.
//!
//! ## Dependencies
//! - storage_crypto.rs: encrypt_record_content / decrypt_record_content
//! - storage_ops.rs: rawlog, feedback, chain_state, stats, miner, overview
//! - storage_graph.rs: cognitive graph CRUD (entities, edges, communities, sessions)
//! - storage_supernode.rs: cognitive_tasks CRUD + llm_usage_log (v2.5.0)
//!
//! ⚠️ Important Notes for Next Developer:
//! - Schema migrations: use `ALTER TABLE` in `maybe_migrate()`, NEVER drop tables.
//! - `record_id` is PRIMARY KEY. Duplicate inserts use `INSERT OR IGNORE`.
//! - `embedding` stored as raw f32 LE bytes. 384-dim = 1536 bytes.
//! - `query_rows()` is `&self` — it needs `self.record_key` for decryption.
//! - LRU cache stores PLAINTEXT records (decrypted) for fast reads.
//! - v4 migration is additive-only (ALTER TABLE ADD COLUMN).
//! - v5 migration creates 8 new tables, ALTERs records, migrates memory_edges data.
//!   All additive — no data loss on upgrade.
//! - v6 migration creates cognitive_tasks + llm_usage_log tables, adds sessions.title.
//!   All additive — no data loss on upgrade.
//! - CRITICAL: Each migrate block MUST update schema_version to its own target
//!   version number (hardcoded integer), NOT the SCHEMA_VERSION constant.
//!   Using SCHEMA_VERSION would skip intermediate migrations when upgrading
//!   across multiple versions (e.g., v4→v5 would set version=6, skipping v6 block).
//! - Rawlog key migration clears old raw_logs on first run after key fix.
//! - episodes.encrypted_content uses same ChaCha20 encryption as records.encrypted_content.
//! - knowledge_edges.valid_until = NULL means "currently valid".
//!   Query pattern: WHERE valid_until IS NULL (current state).
//! - update_record_content: content change clears embedding (NULL) so Miner re-embeds.
//!   Do NOT persist old embeddings after a content change.
//!   SECURITY: ownership check + UPDATE now run in a single lock (no TOCTOU).
//! - find_records_by_content: O(n) scan, prefer FTS5 for large datasets.
//!   pub(crate) only, hard-capped at 100 results to prevent DoS.
//! - get_record_provenance: turn_index lookup uses a LENGTH heuristic (best-effort).
//!   It does not guarantee the correct turn — callers must treat it as advisory.
//! - set_record_session_id / set_record_episode_id: require owner param (prevents
//!   provenance chain tampering by callers that only know the record_id).
//! - conn_lock(): pub(crate) — never expose raw Connection to external crates.
//! - record_key: wrapped in Zeroizing<[u8;32]> — wiped from memory on drop.
//! - Encryption failure in insert/update returns Err (no silent plaintext fallback).
//! - FTS5 backfill skips encrypted databases — indexing ciphertext is meaningless
//!   and leaks encrypted token patterns into an unencrypted FTS table.
//! - row_to_record returns rusqlite::Error on malformed BLOB length — no silent
//!   zero-ID records that corrupt cache / search results.
//! - complete_task enforces AND status='processing' guard (storage_supernode.rs).
//! - v8 tables are integrity commitments, not replicated memory storage. Never
//!   add owner, tags, embeddings, or decrypted content columns to them.
//! - commitment_integrity is runtime-only. It may contain aggregate counts and
//!   heights, but never hashes, proposer identity, commitments, or user data.
//! - commitment_checkpoint is aggregate reconciliation telemetry only. Signed
//!   evidence stays in the bounded local evidence vault and must not enter APIs,
//!   heartbeat, or logs.
//! - Inbound checkpoint requests are requester-controlled observations. Serving
//!   one may update only `last_served_at` and `requests_served_total`; it must
//!   never set convergence, divergence, failure, or observed peer heights.
//! - Checkpoint freshness is derived only from `last_evidence_at` after a valid
//!   vault audit. Evidence above the currently audited local tip is deferred
//!   and cannot refresh freshness or divergence state. Attempts, failures, and
//!   inbound serving never refresh it.
//! - Witness-round fields are process-local aggregate observations. They must
//!   never be interpreted as votes, quorum, finality, or fork choice.
//! - Outbound announcement retry fields are process-local aggregate counters.
//!   Never add peer identities, endpoints, response bodies, or retry timing.
//! - [BLOCK-CARRIER-CIRCUIT-TELEMETRY 2026-07-29 by Codex] Carrier circuit
//!   status may contain only current cooling-slot and cumulative scheduling
//!   counts. Never retain slot order, identities, endpoints, errors, or timing.
//! - Checkpoint certificates are immutable local evidence bundles over
//!   independently verified operator-pinned witness frames. They prove only
//!   that the configured threshold signed one local checkpoint; they are not
//!   BFT finality, global consensus, leader election, or a fork-choice rule.
//! - Block confirmation is derived only from the fully audited local tip and
//!   latest immutable checkpoint certificate. It reports certificate coverage
//!   and lag; it must never be renamed or interpreted as network finality.
//! - commitment_durability is process-local SQLite configuration evidence.
//!   A coordinator must configure FULL-or-stronger before startup audit; never
//!   silently downgrade it while the process is serving commitment traffic.
//! - commitment_tip_anchor keeps its path and signing identity process-local.
//!   Public status may expose only aggregate state, height, timestamps, and
//!   failure counts; never expose its path, signature, signer, or tip hash.
//! - commitment_checkpoint_certificate_anchor applies the same boundary to
//!   the latest fully audited certificate. It uses a separate signed sidecar
//!   so certificate writes cannot race or regress the canonical tip anchor.
//! - commitment_production_halted is one-way for the process lifetime. Only a
//!   trusted-witness security incident may set it; no network input or later
//!   converged frame may clear it. Recovery requires operator review/restart.
//! - [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] The commitment
//!   authority root is process-local and immutable after installation. It must
//!   never be inferred from SQLite, serialized, logged, or exposed in status.
//! - [CUSTODY-WITNESS-RECEIPT-VAULT 2026-08-16 by Codex] Producer-side
//!   receipt evidence stores only canonical signed witness decisions. Normal
//!   receipts may rotate at the hard capacity; adverse evidence is never
//!   silently pruned to make a later round appear healthy.
//! - [CUSTODY-WITNESS-RECEIPT-IMPORT 2026-08-17 by Codex] Every receipt row
//!   records whether it entered through strict live transport or an explicit
//!   operator import, plus that admission's bounded delay. Never infer or
//!   widen this policy during restart audit.
//! - [VOLUME-GROWTH-ADMISSION 2026-08-31 by Codex] SaaS-managed instances may
//!   carry an optional byte-growth admission policy. Local/global instances
//!   retain their original behavior; reads, deletion, and recovery never need
//!   a permit.
//!
//! ## Last Modified
//! [MEMCHAIN-WITNESS-DB-IDENTITY 2026-09-05 by Codex] Bound witness lease
//! authority to a verified SQLite device/inode identity rather than a
//! caller-controlled database path spelling.
//! [MEMCHAIN-WITNESS-CLOCK 2026-09-05 by Codex] Added one process-local,
//! monotonic witness-lease authority per SQLite repository while preserving
//! the durable wall-clock lease schema and public API.
//! v2.8.66-ManagedVolumeGrowth - Added optional managed-volume admission.
//! v2.8.65-CustodyWitnessReceiptImport - Added schema-v18 typed receipt
//! admission evidence for restart-safe live and air-gapped workflows.
//! [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] Added follower-only,
//! source-blind authority-proof recovery and circuit telemetry.
//! [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] Added the process-local
//! immutable proposer-authority trust anchor used by startup and live audits.
//! v2.8.57-CertificatePersistenceTruth - Separated verified-unpersisted follower outcomes.
//! v2.8.56-StickySecurityEvidence - Retained role-isolated security-stop times across later success.
//! v2.8.55-CertificateBackfillTelemetry - Added coordinator-only, source-blind recovery evidence.
//! v2.8.53-TypedCarrierCircuit - Added isolated certificate-carrier circuit aggregates.
//! v2.8.52-BlockCarrierCircuitTelemetry - Added source-blind circuit health aggregates.
//! v2.8.49-FollowerCertificateTipBinding - Bound readiness to the exact audited tip height.
//! v2.8.48-FollowerCertificateReadiness - Added identity-blind current-policy readiness evidence.
//! v2.8.18-TipSupersession - Added aggregate superseded announcement evidence.
//! v2.8.17-TipRetryQueue - Added aggregate bounded announcement retry evidence.
//! v2.8.15-AnnouncementReceipts - Added coordinator-side aggregate delivery evidence.
//! v2.8.14-SyncObservability - Added authenticated announcement dispositions and trigger evidence.
//! v2.8.13-BlockConfirmation - Added privacy-safe witness-certificate coverage.
//! v1.0.0 - Initial SQLite storage engine
//! v2.1.0 - 4-layer, plaintext embedding BLOB, compaction via layer change
//! v2.1.0+MVF - Schema v4, feedback columns, content dedup
//! v2.1.0+MVF+Encryption - Record encryption, rawlog key fix
//! v2.2.0 - Split into storage.rs + storage_crypto.rs + storage_ops.rs
//! v2.4.0-GraphCognition - Schema v5: Three-layer cognitive graph (8 new tables,
//!   records ALTER, memory_edges migration)
//! v2.5.0-SuperNode - Schema v6: cognitive_tasks + llm_usage_log + sessions.title
//!   BUG FIX: v5 migrate block hardcoded version to 5 (was incorrectly using
//!   SCHEMA_VERSION constant which would skip v6 migration on v4→v6 upgrades)
//! v2.5.2+Provenance  - Added update_record_content, set_record_session_id,
//!   set_record_episode_id, get_records_for_session, find_records_by_content,
//!   get_record_provenance, RecordProvenance struct.
//!   BUG FIX: memory_edges migration used source_id as owner (wrong bytes).
//!   BUG FIX: turn_index LENGTH heuristic unit mismatch — documented.
//! v2.5.2+SecAudit    - P0: set_record_session/episode_id now require owner param.
//!   P0: update_record_content ownership check + UPDATE merged into single lock
//!   (eliminates TOCTOU race). UPDATE WHERE adds owner+status guard.
//!   P1: insert/update_record_content encryption failure now returns Err (no
//!   silent plaintext fallback). P1: FTS5 backfill skips encrypted DB.
//!   P1: complete_task adds AND status='processing' guard (storage_supernode.rs).
//!   P2: find_records_by_content → pub(crate), hard-cap limit at 100.
//! v2.7.4-BlockIntegrityStatus - Added runtime-only verified-chain baseline.
//! v2.7.5-CheckpointProof - Added privacy-safe reconciliation runtime evidence.
//! v2.7.6-EvidenceVault - Schema v9: bounded durable signed checkpoint evidence
//!   with aggregate-only runtime/API reporting.
//!   P2: record_key wrapped in Zeroizing<[u8;32]>.
//!   P3: conn_lock() → pub(crate). row_to_record returns Err on bad BLOB length.
//! v2.7.10-CheckpointDirectionIsolation - Separated inbound service counters
//!   from outbound signed checkpoint verification state.
//! v2.7.11-CheckpointFreshness - Added age-bounded verified observation status.
//! v2.7.12-WitnessRoundEvidence - Added privacy-safe bounded round coverage.
//! v2.7.13-CommitmentDurability - Added coordinator fail-closed durability state.
//! v2.7.14-CommitmentTipAnchor - Added a signed local high-water rollback guard.
//! v2.6.1+BlindVectorRecovery - Added all-owner active embedding enumeration so
//!   node-blind/remote Local-mode nodes can rebuild every isolated vector
//!   partition after restart without weakening owner-scoped recall.
//! v2.7.3-BlockAudit - Made v8 commitment tables self-creating for direct
//!   legacy migration callers before startup integrity verification.
//! v2.7.1-BlockSyncStatus - Runtime-only bounded follower status and fault evidence.
//! v2.7.0-BlockSync - Schema v8 commitment blocks and unique membership index.
//! v2.7.20-WitnessEquivocation - Schema v10 durable trusted-witness double-sign evidence.
//! v2.7.21-TrustedDivergenceHalt - Schema v11 sticky trusted divergence incidents
//!   and an atomic local commitment-production safety latch.
//! v2.7.22-CheckpointCertificate - Schema v12 immutable bounded multi-witness
//!   checkpoint certificates with restart-time cryptographic re-audit.
//! v2.7.23-CertificateExchange - Added audited internal certificate export;
//! v2.7.24-CertificateRollbackGuard - Added a signed local high-water sidecar
//!   for the latest fully audited checkpoint certificate.
//! v2.7.25-CoordinatorProductionFence - Added an OS-owned exclusive lock that
//!   prevents duplicate local coordinator processes from producing against
//!   the same SQLite chain and sidecars.
//!   No SQLite schema change; aggregate public fence status was added.
//! v2.7.26-CoordinatorLease - Added durable witness-side coordinator lease
//!   grants and process-local lease validity state for cross-host fencing.
//! v2.8.29-CustodyAuditWitness - Schema v16 separate producer-scoped custody
//!   checkpoint witness high-water decisions containing exact frame digests.
//! v2.8.28-VerifiedDeliveryAnchorWitness - Schema v14 durable aggregate-only
//!   delivery-anchor high-water decisions for admitted node peers.
//! v2.8.12-LeaseFailClosedTelemetry - Added process-local lease attempt,
//!   consecutive-failure, recovery, and monotonic remaining-window evidence.
// ============================================

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nix::fcntl::Flock;
use parking_lot::RwLock;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::ledger::{
    MemoryLayer, MemoryRecord, RecordStatus, MAX_RECORD_COMMITMENTS_PER_BLOCK,
};
use zeroize::Zeroizing;

use super::storage_crypto::{decrypt_record_content, encrypt_record_content};

// ============================================
// Constants
// ============================================

/// Current schema version.
/// v4 → v5: cognitive graph tables
/// v5 → v6: SuperNode cognitive_tasks + llm_usage_log + sessions.title
/// v7 → v8: signed node-blind commitment chain + membership index
/// v8 → v9: bounded local signed checkpoint evidence vault
/// v9 → v10: durable signed-witness equivocation incidents
/// v10 → v11: durable trusted-witness divergent-prefix incidents
/// v11 → v12: immutable pinned-witness checkpoint certificates
/// v12 → v13: durable exclusive coordinator lease grants on follower witnesses
/// v13 → v14: durable verified-delivery anchor high-water decisions
/// v14 → v15: append-only dual-signed coordinator authority history
/// v15 → v16: independent custody audit anchor witness high-water decisions
/// v16 → v17: bounded producer-side custody witness receipt evidence
/// v17 → v18: per-receipt custody witness admission policy evidence
///
/// ⚠️ CRITICAL: When bumping this, you MUST also add a new migrate block
/// in `maybe_migrate()`. The migrate block MUST use a hardcoded integer
/// (not this constant) for `UPDATE schema_version`, to prevent skipping
/// intermediate migrations on multi-version upgrades.
// [CUSTODY-AUDIT-WITNESS 2026-08-16 by Codex] Keep this visible only inside
// the memchain module so cross-file migration tests assert the authoritative
// current version without coupling production migration steps to this value.
pub(super) const SCHEMA_VERSION: u32 = 18;

const LRU_CACHE_CAPACITY: usize = 1000;
const DEFAULT_PAGE_SIZE: usize = 100;

// ============================================
// Managed-volume growth admission
// ============================================

/// Privacy-safe failure to admit a byte-growing managed-volume mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StorageGrowthError {
    #[error("Managed storage has reached its configured byte capacity")]
    AtCapacity,
    #[error("Managed storage capacity measurement is temporarily unavailable")]
    ProbeUnavailable,
    #[error("Managed storage byte accounting overflow")]
    AccountingOverflow,
    #[error("Managed storage volume is unavailable")]
    VolumeUnavailable,
}

// [MEMORY-V2-OWNER-SLOT 2026-10-02 by Codex] Remote-owner admission is a
// durable row invariant, distinct from byte-growth admission.  The error is
// deliberately coarse so callers cannot turn quota or SQLite state into an
// owner/database oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum OwnerSlotAdmissionError {
    #[error("remote owner capacity reached")]
    AtCapacity,
    #[error("remote owner admission unavailable")]
    StorageUnavailable,
}

impl OwnerSlotAdmissionError {
    pub(crate) const fn is_at_capacity(self) -> bool {
        matches!(self, Self::AtCapacity)
    }
}

// [MEMORY-V2-OWNER-SLOT 2026-10-02 by Codex] This policy is copied into one
// SQLite transaction and contains only the local public owner and configured
// aggregate ceiling; no request body or endpoint data crosses the storage API.
#[derive(Clone, Copy)]
pub(crate) struct OwnerSlotPolicy {
    pub(crate) local_owner: [u8; 32],
    pub(crate) max_remote_owners: usize,
}

impl StorageGrowthError {
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::AtCapacity => "at_capacity",
            Self::ProbeUnavailable => "probe_unavailable",
            Self::AccountingOverflow => "accounting_overflow",
            Self::VolumeUnavailable => "volume_unavailable",
        }
    }

    pub(crate) const fn is_at_capacity(self) -> bool {
        matches!(self, Self::AtCapacity)
    }
}

/// Opaque RAII permit held through one complete logical mutation.
pub(crate) struct StorageGrowthPermit {
    _inner: Box<dyn Send + Sync>,
}

impl StorageGrowthPermit {
    pub(crate) fn new(inner: impl Send + Sync + 'static) -> Self {
        Self {
            _inner: Box::new(inner),
        }
    }

    fn unmanaged() -> Self {
        Self::new(())
    }
}

impl std::fmt::Debug for StorageGrowthPermit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("StorageGrowthPermit")
    }
}

/// Replaceable policy boundary used only by SaaS managed-volume instances.
#[async_trait::async_trait]
pub(crate) trait StorageGrowthAdmission: Send + Sync {
    async fn acquire(
        &self,
        minimum_growth_bytes: u64,
    ) -> Result<StorageGrowthPermit, StorageGrowthError>;
}

// ============================================
// StorageStats / LayerCounts
// ============================================

#[derive(Debug, Clone, serde::Serialize)]
pub struct StorageStats {
    pub total_records: u64,
    pub active_records: u64,
    pub by_layer: LayerCounts,
    pub content_bytes: u64,
    pub records_with_embedding: u64,
    pub session_inserts: u64,
    pub session_rejects: u64,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct LayerCounts {
    pub identity: u64,
    pub knowledge: u64,
    pub episode: u64,
    pub archive: u64,
}

// ============================================
// RawLogRow
// ============================================

#[derive(Debug, Clone)]
pub struct RawLogRow {
    pub log_id: i64,
    pub session_id: String,
    pub turn_index: i64,
    pub role: String,
    pub content: Vec<u8>,
    pub encrypted: i64,
    pub recall_context: Option<String>,
    pub extractable: Option<i64>,
    pub feedback_signal: Option<i64>,
}

// ============================================
// RecordProvenance (v2.5.2+Provenance)
// ============================================

/// Full provenance chain for a memory record.
///
/// Returned by `get_record_provenance()`.
///
/// ⚠️ `turn_index` is best-effort only — derived from a LENGTH heuristic
/// against raw_logs. It is advisory and may point to the wrong turn when
/// multiple turns have similar content lengths. Do not rely on it for
/// exact replay without verification.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecordProvenance {
    pub record_id: String,
    pub session_id: Option<String>,
    pub session_title: Option<String>,
    pub session_started_at: Option<i64>,
    /// Best-effort turn index (LENGTH heuristic, may be inaccurate). Advisory only.
    pub turn_index: Option<i64>,
    pub layer: String,
    pub topic_tags: Vec<String>,
    pub extracted_at: u64,
    pub source_ai: String,
}

// ============================================
// Commitment Sync Runtime Evidence (v2.7.1)
// ============================================

/// Authenticated block-announcement disposition recorded by a follower.
///
/// This enum deliberately describes only scheduler handling. It does not
/// claim that the announced block was imported, valid, or canonical; those
/// properties are established later by the signed pull and checkpoint path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCommitmentAnnouncementDisposition {
    /// The bounded follower wake-up channel accepted a new event.
    Accepted,
    /// An equivalent wake-up was already pending and the event was coalesced.
    Coalesced,
    /// The authenticated announcement did not exceed the audited local tip.
    Stale,
    /// The follower task was unavailable while the HTTP runtime remained up.
    Unavailable,
}

impl RecordCommitmentAnnouncementDisposition {
    /// Returns the stable privacy-safe API value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Coalesced => "coalesced",
            Self::Stale => "stale",
            Self::Unavailable => "unavailable",
        }
    }
}

/// One privacy-safe follower lifecycle event.
///
/// Events contain no node identity, endpoint, block hash, record commitment,
/// owner, payload, route, or client metadata. The in-memory ring is capped by
/// `COMMITMENT_SYNC_EVENT_CAPACITY` and is never written to SQLite.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RecordCommitmentSyncEvent {
    /// Monotonic process-local event sequence.
    pub sequence: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Stable lifecycle kind: `failure` or `recovered`.
    pub kind: String,
    /// Stable allow-listed failure code; absent for recovery events.
    pub error_code: Option<String>,
    /// Failure streak after this event.
    pub consecutive_failures: u32,
    /// Scheduled retry time for a failure event.
    pub next_poll_at: Option<u64>,
}

/// One terminal outcome for a follower checkpoint-certificate retrieval round.
///
/// [FOLLOWER-CERTIFICATE-TELEMETRY 2026-07-29 by Codex] This process-local
/// classification deliberately excludes source identities, endpoints,
/// certificate material, hashes, signatures, and raw errors. It is operational
/// evidence only and must never become authority, reputation, or fork choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentCertificateSyncDisposition {
    Coordinator,
    CarrierRecovered,
    VerifiedUnpersisted,
    AvailabilityExhausted,
    SecurityStopped,
}

impl RecordCommitmentCertificateSyncDisposition {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::CarrierRecovered => "carrier_recovered",
            Self::VerifiedUnpersisted => "verified_unpersisted",
            Self::AvailabilityExhausted => "availability_exhausted",
            Self::SecurityStopped => "security_stopped",
        }
    }
}

/// One terminal outcome for a coordinator certificate-backfill round.
///
/// [CERTIFICATE-BACKFILL-TELEMETRY 2026-07-29 by Codex] Coordinator backfill
/// is operationally distinct from follower certificate synchronization. This
/// classification therefore has a separate contract and deliberately cannot
/// retain carrier identities, endpoints, witness sets, certificate material,
/// hashes, signatures, raw errors, or per-source timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentCertificateBackfillDisposition {
    Persisted,
    VerifiedUnpersisted,
    AvailabilityExhausted,
    SecurityStopped,
}

impl RecordCommitmentCertificateBackfillDisposition {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Persisted => "persisted",
            Self::VerifiedUnpersisted => "verified_unpersisted",
            Self::AvailabilityExhausted => "availability_exhausted",
            Self::SecurityStopped => "security_stopped",
        }
    }
}

/// One terminal outcome for a follower commitment-block page retrieval.
///
/// [FOLLOWER-BLOCK-CARRIER-TELEMETRY 2026-07-29 by Codex] This process-local
/// classification is source-blind: it can distinguish direct success,
/// pinned-carrier page recovery, exhausted availability, and fail-closed
/// security stops without retaining any node identity, endpoint, block
/// material, route, or raw failure. It never participates in chain decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentBlockPagePullDisposition {
    Coordinator,
    CarrierRecovered,
    AvailabilityExhausted,
    SecurityStopped,
}

impl RecordCommitmentBlockPagePullDisposition {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::CarrierRecovered => "carrier_recovered",
            Self::AvailabilityExhausted => "availability_exhausted",
            Self::SecurityStopped => "security_stopped",
        }
    }
}

/// One terminal outcome for a follower coordinator-handover retrieval round.
///
/// [AUTHORITY-HANDOVER-CARRIER 2026-08-14 by Codex] This process-local
/// classification reports whether an exact dual-signed authority proof came
/// directly from the active coordinator or through an already-pinned carrier.
/// It cannot retain identities, endpoints, proof material, epochs, heights,
/// hashes, signatures, raw errors, or routes, and never grants authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentAuthoritySyncDisposition {
    Coordinator,
    CarrierRecovered,
    AvailabilityExhausted,
    SecurityStopped,
}

impl RecordCommitmentAuthoritySyncDisposition {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Coordinator => "coordinator",
            Self::CarrierRecovered => "carrier_recovered",
            Self::AvailabilityExhausted => "availability_exhausted",
            Self::SecurityStopped => "security_stopped",
        }
    }
}

/// Current follower certificate-policy readiness after exact local validation.
///
/// [FOLLOWER-CERTIFICATE-READINESS 2026-07-29 by Codex] These states are
/// process-local and identity-blind. They describe whether the current audited
/// tip satisfies the follower's current pins and threshold; they never expose
/// certificate members, hashes, signatures, endpoints, or raw failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentCertificatePolicyReadiness {
    Disabled,
    Ready { tip_height: u64 },
    WaitingForConvergence,
    WaitingForCertificate { tip_height: u64 },
    SourceUnavailable { tip_height: u64 },
    SecurityStopped,
    ConfigurationError,
}

impl RecordCommitmentCertificatePolicyReadiness {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Ready { .. } => "ready",
            Self::WaitingForConvergence => "waiting_for_convergence",
            Self::WaitingForCertificate { .. } => "waiting_for_certificate",
            Self::SourceUnavailable { .. } => "source_unavailable",
            Self::SecurityStopped => "security_stopped",
            Self::ConfigurationError => "configuration_error",
        }
    }

    pub(crate) const fn is_ready(self) -> bool {
        matches!(self, Self::Ready { .. })
    }

    pub(crate) const fn evaluated_tip_height(self) -> Option<u64> {
        match self {
            Self::Ready { tip_height }
            | Self::WaitingForCertificate { tip_height }
            | Self::SourceUnavailable { tip_height } => Some(tip_height),
            Self::Disabled
            | Self::WaitingForConvergence
            | Self::SecurityStopped
            | Self::ConfigurationError => None,
        }
    }
}

/// Effective follower readiness after combining block convergence and policy.
///
/// [FOLLOWER-EFFECTIVE-READINESS 2026-07-30 by Codex] This process-local,
/// identity-blind classification prevents a signed equal-tip checkpoint from
/// being mistaken for complete follower readiness while a required certificate
/// is pending, unavailable, or security-stopped. It is status only and never
/// participates in chain choice, certificate policy, or peer authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordCommitmentFollowerReadiness {
    NotApplicable,
    Starting,
    Synchronizing,
    Backoff,
    Stopped,
    Stale,
    CertifiedRecovered,
    WaitingForCertificate,
    SourceUnavailable,
    SecurityStopped,
    ConfigurationError,
    Ready,
}

impl RecordCommitmentFollowerReadiness {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Starting => "starting",
            Self::Synchronizing => "synchronizing",
            Self::Backoff => "backoff",
            Self::Stopped => "stopped",
            Self::Stale => "stale",
            Self::CertifiedRecovered => "certified_recovered",
            Self::WaitingForCertificate => "waiting_for_certificate",
            Self::SourceUnavailable => "source_unavailable",
            Self::SecurityStopped => "security_stopped",
            Self::ConfigurationError => "configuration_error",
            Self::Ready => "ready",
        }
    }

    pub(crate) const fn is_fully_ready(self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// Privacy-safe runtime status for commitment block replication.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RecordCommitmentSyncStatus {
    /// Stable API contract name.
    pub contract_version: &'static str,
    /// Node role: `coordinator`, `follower`, or `verifier`.
    pub role: String,
    /// Runtime state such as `current`, `certified_recovered`, `catching_up`,
    /// or `backoff`. [CERTIFIED-BLOCK-CARRIER 2026-07-29 by Codex]
    pub state: String,
    /// Effective follower state after block and certificate-policy evaluation.
    pub follower_readiness_state: String,
    /// Whether a follower is producer-current and satisfies local policy.
    pub follower_fully_ready: bool,
    /// Most recent signed equal-tip producer convergence confirmation.
    pub follower_convergence_confirmed_at: Option<u64>,
    /// Time after which the current convergence observation is no longer ready.
    pub follower_readiness_stale_after: Option<u64>,
    /// Whether active follower polling is configured.
    pub enabled: bool,
    /// Trigger used by the most recent pull attempt: `scheduled` or
    /// `block_announce`.
    pub last_trigger: String,
    /// Most recent authenticated announcement handling time.
    pub last_announcement_at: Option<u64>,
    /// Highest authenticated announced height observed this process lifetime.
    pub last_announced_height: Option<u64>,
    /// Last scheduler disposition: `accepted`, `coalesced`, `stale`, or
    /// `unavailable`.
    pub last_announcement_result: Option<String>,
    /// Announcements inserted into the bounded follower wake-up channel.
    pub announcements_accepted_total: u64,
    /// Announcements merged because one wake-up was already pending.
    pub announcements_coalesced_total: u64,
    /// Authenticated announcements at or below the audited local tip.
    pub announcements_stale_total: u64,
    /// Authenticated announcements received after the follower task closed.
    pub announcements_unavailable_total: u64,
    /// Most recent coordinator-side tip announcement round time.
    pub last_outbound_announcement_at: Option<u64>,
    /// Audited local tip height encoded by the most recent outbound round.
    pub last_outbound_announced_height: Option<u64>,
    /// Last aggregate delivery result: `all_woken`, `delivered`, `partial`,
    /// `failed`, `no_targets`, `superseded`, or `skipped`.
    pub last_outbound_announcement_result: Option<String>,
    /// Outbound announcement rounds observed since process start.
    pub outbound_announcement_rounds_total: u64,
    /// Rounds skipped before a peer delivery outcome could be produced.
    pub outbound_announcement_rounds_skipped_total: u64,
    /// In-flight rounds canceled in favor of a strictly newer audited tip.
    pub outbound_announcement_rounds_superseded_total: u64,
    /// Distinct pinned peers considered by outbound announcement rounds.
    pub outbound_announcements_attempted_total: u64,
    /// Peers returning exactly `202 Accepted`.
    pub outbound_announcements_accepted_total: u64,
    /// Peers returning exactly `204 No Content` because their tip was current.
    pub outbound_announcements_stale_total: u64,
    /// Missing, unsafe, unreachable, or protocol-incompatible peers.
    pub outbound_announcements_failed_total: u64,
    /// Additional delivery attempts after transient failures.
    pub outbound_announcement_retries_attempted_total: u64,
    /// Peers recovered by a bounded retry.
    pub outbound_announcement_retries_succeeded_total: u64,
    /// Peers still transiently failing after the retry budget.
    pub outbound_announcement_retries_exhausted_total: u64,
    /// Most recent terminal coordinator-handover retrieval round.
    pub last_authority_sync_at: Option<u64>,
    /// Latest source-blind authority-proof retrieval result.
    pub last_authority_sync_result: Option<String>,
    /// Most recent proof recovered through an already-pinned carrier.
    pub last_authority_carrier_recovered_at: Option<u64>,
    /// Terminal authority-proof retrieval rounds observed since process start.
    pub authority_sync_rounds_total: u64,
    /// Rounds completed directly through the active coordinator.
    pub authority_coordinator_success_total: u64,
    /// Pinned carrier requests attempted after coordinator availability faults.
    pub authority_carrier_attempts_total: u64,
    /// Rounds recovered through an already-pinned carrier.
    pub authority_carrier_recoveries_total: u64,
    /// Rounds where the coordinator and every bounded carrier were unavailable.
    pub authority_availability_exhausted_total: u64,
    /// Rounds stopped by a security or protocol-integrity failure.
    pub authority_security_stops_total: u64,
    /// Most recent fail-closed authority-proof security stop.
    pub last_authority_security_stop_at: Option<u64>,
    /// Fixed authority-carrier slots cooling at the latest observation.
    pub authority_carrier_cooling_slots: usize,
    /// Authority-carrier selections skipped during anonymous cooldown.
    pub authority_carrier_cooldown_skips_total: u64,
    /// Authority-carrier requests started after anonymous cooldown expiry.
    pub authority_carrier_half_open_attempts_total: u64,
    /// Most recent terminal commitment-block page retrieval.
    pub last_block_page_pull_at: Option<u64>,
    /// Latest source-blind page retrieval result.
    pub last_block_page_pull_result: Option<String>,
    /// Most recent page recovered through an already-pinned carrier.
    pub last_block_carrier_recovered_at: Option<u64>,
    /// Terminal block-page retrievals observed since process start.
    pub block_page_pulls_total: u64,
    /// Pages retrieved directly from the configured coordinator.
    pub block_page_coordinator_success_total: u64,
    /// Pinned carrier requests attempted after coordinator availability faults.
    pub block_carrier_attempts_total: u64,
    /// Pages recovered through an already-pinned carrier.
    pub block_carrier_recoveries_total: u64,
    /// Page retrievals where every eligible bounded source was unavailable.
    pub block_page_availability_exhausted_total: u64,
    /// Page retrievals stopped by security or protocol-integrity failures.
    pub block_page_security_stops_total: u64,
    /// Most recent fail-closed block-page security stop in this process.
    pub last_block_page_security_stop_at: Option<u64>,
    /// Fixed operator-pin slots cooling at the latest follower observation.
    pub block_carrier_cooling_slots: usize,
    /// Carrier selections skipped while their anonymous slot was cooling.
    pub block_carrier_cooldown_skips_total: u64,
    /// Outbound carrier requests started after an anonymous cooldown expired.
    pub block_carrier_half_open_attempts_total: u64,
    /// Current certificate policy state without witness identities.
    pub certificate_policy_state: String,
    /// Whether the exact current audited tip satisfies current local policy.
    pub certificate_policy_ready: bool,
    /// Most recent exact local policy evaluation time.
    pub certificate_policy_last_evaluated_at: Option<u64>,
    /// Audited local tip height covered by the latest applicable evaluation.
    pub certificate_policy_evaluated_tip_height: Option<u64>,
    /// Count of configured external witnesses; identities are never exposed.
    pub certificate_witnesses_configured: usize,
    /// Minimum distinct signatures required by local follower policy.
    pub certificate_minimum_signers: usize,
    /// Most recent completed checkpoint-certificate retrieval round.
    pub last_certificate_sync_at: Option<u64>,
    /// Latest aggregate terminal result. No source identity is retained.
    pub last_certificate_sync_result: Option<String>,
    /// Most recent round recovered through an already-pinned carrier.
    pub last_certificate_carrier_recovered_at: Option<u64>,
    /// Completed certificate retrieval rounds observed since process start.
    pub certificate_sync_rounds_total: u64,
    /// Rounds completed directly through the configured coordinator.
    pub certificate_coordinator_success_total: u64,
    /// Pinned carrier requests attempted after coordinator availability faults.
    pub certificate_carrier_attempts_total: u64,
    /// Rounds recovered through one already-pinned certificate carrier.
    pub certificate_carrier_recoveries_total: u64,
    /// Verified rounds deferred because local state changed before persistence.
    pub certificate_verified_unpersisted_total: u64,
    /// Rounds where the coordinator and every bounded carrier were unavailable.
    pub certificate_availability_exhausted_total: u64,
    /// Rounds stopped by a security or protocol-integrity failure.
    pub certificate_security_stops_total: u64,
    /// Most recent fail-closed follower certificate security stop.
    pub last_certificate_security_stop_at: Option<u64>,
    /// Fixed certificate-carrier slots cooling at the latest observation.
    pub certificate_carrier_cooling_slots: usize,
    /// Certificate-carrier selections skipped during anonymous cooldown.
    pub certificate_carrier_cooldown_skips_total: u64,
    /// Certificate-carrier requests started after anonymous cooldown expiry.
    pub certificate_carrier_half_open_attempts_total: u64,
    /// Most recent coordinator post-startup certificate-backfill round.
    pub last_coordinator_certificate_backfill_at: Option<u64>,
    /// Latest source-blind coordinator backfill terminal result.
    pub last_coordinator_certificate_backfill_result: Option<String>,
    /// Coordinator certificate-backfill rounds observed since process start.
    pub coordinator_certificate_backfill_rounds_total: u64,
    /// Backfill rounds that durably persisted an audited certificate.
    pub coordinator_certificate_backfill_persisted_total: u64,
    /// Verified rounds deferred because local state changed before persistence.
    pub coordinator_certificate_backfill_verified_unpersisted_total: u64,
    /// Backfill rounds where every bounded carrier was unavailable.
    pub coordinator_certificate_backfill_availability_exhausted_total: u64,
    /// Backfill rounds stopped by security or protocol-integrity failures.
    pub coordinator_certificate_backfill_security_stops_total: u64,
    /// Most recent fail-closed coordinator certificate-backfill security stop.
    pub last_coordinator_certificate_backfill_security_stop_at: Option<u64>,
    /// Bounded carrier requests attempted by coordinator backfill.
    pub coordinator_certificate_backfill_carrier_attempts_total: u64,
    /// Anonymous carrier slots cooling at the latest coordinator observation.
    pub coordinator_certificate_backfill_carrier_cooling_slots: usize,
    /// Coordinator backfill selections skipped during anonymous cooldown.
    pub coordinator_certificate_backfill_carrier_cooldown_skips_total: u64,
    /// Coordinator backfill requests started after anonymous cooldown expiry.
    pub coordinator_certificate_backfill_carrier_half_open_attempts_total: u64,
    /// Most recent pull attempt time.
    pub last_attempt_at: Option<u64>,
    /// Most recent successfully verified page time.
    pub last_success_at: Option<u64>,
    /// Most recent failed pull time.
    pub last_failure_at: Option<u64>,
    /// Most recent transition from a failure streak back to success.
    pub last_recovered_at: Option<u64>,
    /// Next scheduled poll or retry time.
    pub next_poll_at: Option<u64>,
    /// Current consecutive failure count.
    pub consecutive_failures: u32,
    /// Last stable allow-listed failure code.
    pub last_error_code: Option<String>,
    /// Last coordinator tip height observed in a verified response.
    pub remote_tip_height: Option<u64>,
    /// Number of successfully verified response pages this process received.
    pub pages_received_total: u64,
    /// Number of verified blocks represented by those pages.
    pub blocks_received_total: u64,
    /// Number of failure events observed by this process.
    pub failure_events_total: u64,
    /// Number of failure-to-success recoveries observed by this process.
    pub recovery_events_total: u64,
    /// Most recent bounded lifecycle events, oldest first.
    pub recent_events: Vec<RecordCommitmentSyncEvent>,
    /// Explicit privacy boundary for operators and API consumers.
    pub privacy_policy: &'static str,
}

/// Privacy-safe runtime status for signed chain-checkpoint reconciliation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RecordCommitmentCheckpointStatus {
    /// Stable API contract name.
    pub contract_version: &'static str,
    /// Last outbound observation: `not_checked`, `converged`, `remote_ahead`,
    /// `remote_behind`, `diverged`, or `proof_failed`. Inbound requests cannot
    /// change this field.
    pub state: String,
    /// Most recent outbound checkpoint verification attempt.
    pub last_checked_at: Option<u64>,
    /// Most recent cryptographically verified equal tip.
    pub last_converged_at: Option<u64>,
    /// Most recent signed shared-prefix mismatch.
    pub last_divergence_at: Option<u64>,
    /// Most recent invalid or unavailable checkpoint proof.
    pub last_failure_at: Option<u64>,
    /// Most recent authenticated checkpoint response served to a peer.
    pub last_served_at: Option<u64>,
    /// Local height used by the most recent outbound verified observation.
    pub local_tip_height: Option<u64>,
    /// Remote height used by the most recent outbound verified observation.
    pub remote_tip_height: Option<u64>,
    /// Signed checkpoint responses verified since process start.
    pub proofs_verified_total: u64,
    /// Checkpoint attempts rejected before a valid proof was established.
    pub proofs_failed_total: u64,
    /// Signed shared-prefix mismatches observed since process start.
    pub divergences_total: u64,
    /// Authenticated checkpoint responses served since process start.
    pub requests_served_total: u64,
    /// Startup cryptographic audit state for durable proof frames:
    /// `not_audited`, `verified`, or `invalid`. A verified frame can still be
    /// deferred when its historical local tip is not yet available.
    pub evidence_state: String,
    /// Signature-verified proof frames currently retained in the bounded vault.
    pub evidence_records: u64,
    /// Retained proofs applicable to the currently audited local chain.
    pub applicable_evidence_records: u64,
    /// Cryptographically valid historical proofs whose recorded local tip is
    /// above the currently audited tip. They remain retained for recovery but
    /// cannot affect freshness, convergence, or divergence telemetry.
    pub deferred_evidence_records: u64,
    /// Applicable retained proofs that established a shared-prefix mismatch.
    pub divergence_evidence_records: u64,
    /// Durable incidents where one trusted witness signed incompatible hashes
    /// for the same checkpoint or tip height. Witness identities stay local.
    pub equivocation_incidents: u64,
    /// Durable incidents where an operator-pinned witness signed a checkpoint
    /// that diverges from the locally audited prefix. Identities stay local.
    pub trusted_divergence_incidents: u64,
    /// Immutable multi-witness certificates retained after full re-audit.
    pub checkpoint_certificates: u64,
    /// Highest local checkpoint covered by a retained certificate.
    pub latest_certified_height: Option<u64>,
    /// Distinct signed witness frames in the latest certificate.
    pub latest_certificate_signers: usize,
    /// Threshold recorded by the latest immutable certificate.
    pub latest_certificate_required_signers: usize,
    /// `not_verified`, `empty`, `uncertified`, `witness_certified`,
    /// `certificate_lagging`, `certificate_invalid`, or `certificate_ahead`.
    /// This is certificate coverage, not global finality.
    pub block_confirmation_state: String,
    /// Number of locally verified tip blocks above the latest certificate.
    /// Zero does not imply consensus; consult `block_confirmation_state`.
    pub uncertified_block_count: u64,
    /// Exact anti-overclaim boundary for the derived confirmation state.
    pub block_confirmation_policy: &'static str,
    /// Signed local certificate high-water state: `disabled`, `checking`,
    /// `initialized`, `verified`, `repaired`, `invalid`,
    /// `rollback_detected`, or `write_failed`.
    pub certificate_rollback_guard_state: String,
    /// Highest certificate height retained by the signed local high-water mark.
    pub certificate_rollback_guard_height: u64,
    /// Most recent successful comparison with the audited certificate vault.
    pub certificate_rollback_guard_last_verified_at: Option<u64>,
    /// Most recent durable signed sidecar replacement.
    pub certificate_rollback_guard_last_persisted_at: Option<u64>,
    /// Sidecar persistence failures observed since process start.
    pub certificate_rollback_guard_write_failures_total: u64,
    /// Precise protection boundary; this guard is not network finality.
    pub certificate_rollback_guard_scope: &'static str,
    /// Whether local canonical commitment production is fail-closed for this
    /// process after a trusted witness security incident.
    pub production_halted: bool,
    /// Most recent applicable durable evidence observation time.
    pub last_evidence_at: Option<u64>,
    /// `unavailable`, `fresh`, or `stale` for the most recent durable signed
    /// outbound observation. Vault integrity is reported separately.
    pub observation_freshness: String,
    /// Age of the latest durable signed observation; absent when unavailable.
    pub observation_age_seconds: Option<u64>,
    /// Maximum observation age still classified as `fresh`.
    pub freshness_window_seconds: u64,
    /// Latest coordinator witness-round result: `not_checked`, `unavailable`,
    /// `unverified`, `partial`, `shared_prefix`, or `attention`.
    pub last_round_state: String,
    /// Completion time of the latest bounded witness round.
    pub last_round_at: Option<u64>,
    /// Valid discovered witnesses eligible before the per-round cap.
    pub last_round_eligible: usize,
    /// Witnesses contacted after the per-round cap.
    pub last_round_attempted: usize,
    /// Responses that established durable signed evidence.
    pub last_round_verified: usize,
    /// Attempts that established no durable signed evidence.
    pub last_round_failed: usize,
    /// Verified witnesses at the same signed tip.
    pub last_round_converged: usize,
    /// Verified witnesses extending the local signed prefix.
    pub last_round_remote_ahead: usize,
    /// Verified witnesses behind on the same signed prefix.
    pub last_round_remote_behind: usize,
    /// Verified witnesses signing a different shared-height hash.
    pub last_round_diverged: usize,
    /// Local SQLite persistence failures observed since process start.
    pub evidence_persistence_failures_total: u64,
    /// Explicit privacy boundary for operators and API consumers.
    pub privacy_policy: &'static str,
}

/// Fully audited internal certificate material for one peer response.
///
/// This type must never be serialized into public status or heartbeat. Member
/// frames expose witness identities and signatures, so only the authenticated
/// MemChain peer handler may consume it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecordCommitmentCheckpointCertificateBundle {
    pub(crate) checkpoint_height: u64,
    pub(crate) checkpoint_hash: [u8; 32],
    pub(crate) certificate_digest: [u8; 32],
    pub(crate) required_signers: usize,
    pub(crate) member_frames: Vec<Vec<u8>>,
}

pub(crate) const COMMITMENT_SYNC_EVENT_CAPACITY: usize = 16;
/// Maximum signed checkpoint frames retained locally. Divergence evidence is
/// pruned last so normal convergence checks cannot erase the most useful proof.
pub(crate) const CHECKPOINT_EVIDENCE_CAPACITY: usize = 256;
/// A detected trusted-witness conflict is retained rather than rotated away.
/// Bound the incident ledger so unbounded corruption cannot grow startup work.
pub(crate) const CHECKPOINT_EQUIVOCATION_CAPACITY: usize = 64;
/// A trusted divergent-prefix proof is sticky and cannot be displaced by later
/// convergence. Only explicitly pinned witnesses can consume this capacity.
pub(crate) const CHECKPOINT_TRUSTED_DIVERGENCE_CAPACITY: usize = 64;
/// Certificates retain their exact signed member frames. Sixty-four
/// certificates with at most three members stay within the 256-frame evidence
/// bound while preserving a useful rolling audit history.
pub(crate) const CHECKPOINT_CERTIFICATE_CAPACITY: usize = 64;
pub(crate) const MAX_CHECKPOINT_CERTIFICATE_SIGNERS: usize = 3;
/// Defensive bound for one stored proof frame. Increasing this requires an
/// explicit wire and startup-memory review.
pub(crate) const MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES: usize = 4 * 1024;
/// A coordinator witness round has a minimum five-minute interval. Three
/// missed rounds make the last durable signed observation stale without
/// overreacting to one transient transport failure.
pub(crate) const CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS: u64 = 15 * 60;
/// Maximum canonical custody witness receipt frames retained by one producer.
/// Normal accepted evidence rotates oldest-first; adverse evidence reserves
/// capacity and forces operator review instead of being silently erased.
pub(crate) const CUSTODY_WITNESS_RECEIPT_EVIDENCE_CAPACITY: usize = 256;

#[derive(Debug, Clone)]
pub(crate) struct RecordCommitmentSyncRuntime {
    pub(crate) role: &'static str,
    pub(crate) state: &'static str,
    pub(crate) enabled: bool,
    pub(crate) last_trigger: &'static str,
    pub(crate) last_announcement_at: Option<u64>,
    pub(crate) last_announced_height: Option<u64>,
    pub(crate) last_announcement_result: Option<&'static str>,
    pub(crate) announcements_accepted_total: u64,
    pub(crate) announcements_coalesced_total: u64,
    pub(crate) announcements_stale_total: u64,
    pub(crate) announcements_unavailable_total: u64,
    pub(crate) last_outbound_announcement_at: Option<u64>,
    pub(crate) last_outbound_announced_height: Option<u64>,
    pub(crate) last_outbound_announcement_result: Option<&'static str>,
    pub(crate) outbound_announcement_rounds_total: u64,
    pub(crate) outbound_announcement_rounds_skipped_total: u64,
    pub(crate) outbound_announcement_rounds_superseded_total: u64,
    pub(crate) outbound_announcements_attempted_total: u64,
    pub(crate) outbound_announcements_accepted_total: u64,
    pub(crate) outbound_announcements_stale_total: u64,
    pub(crate) outbound_announcements_failed_total: u64,
    pub(crate) outbound_announcement_retries_attempted_total: u64,
    pub(crate) outbound_announcement_retries_succeeded_total: u64,
    pub(crate) outbound_announcement_retries_exhausted_total: u64,
    pub(crate) last_authority_sync_at: Option<u64>,
    pub(crate) last_authority_sync_result: Option<&'static str>,
    pub(crate) last_authority_carrier_recovered_at: Option<u64>,
    pub(crate) authority_sync_rounds_total: u64,
    pub(crate) authority_coordinator_success_total: u64,
    pub(crate) authority_carrier_attempts_total: u64,
    pub(crate) authority_carrier_recoveries_total: u64,
    pub(crate) authority_availability_exhausted_total: u64,
    pub(crate) authority_security_stops_total: u64,
    pub(crate) last_authority_security_stop_at: Option<u64>,
    pub(crate) authority_carrier_cooling_slots: usize,
    pub(crate) authority_carrier_cooldown_skips_total: u64,
    pub(crate) authority_carrier_half_open_attempts_total: u64,
    pub(crate) last_block_page_pull_at: Option<u64>,
    pub(crate) last_block_page_pull_result: Option<&'static str>,
    pub(crate) last_block_carrier_recovered_at: Option<u64>,
    pub(crate) block_page_pulls_total: u64,
    pub(crate) block_page_coordinator_success_total: u64,
    pub(crate) block_carrier_attempts_total: u64,
    pub(crate) block_carrier_recoveries_total: u64,
    pub(crate) block_page_availability_exhausted_total: u64,
    pub(crate) block_page_security_stops_total: u64,
    pub(crate) last_block_page_security_stop_at: Option<u64>,
    pub(crate) block_carrier_cooling_slots: usize,
    pub(crate) block_carrier_cooldown_skips_total: u64,
    pub(crate) block_carrier_half_open_attempts_total: u64,
    pub(crate) certificate_policy_state: &'static str,
    pub(crate) certificate_policy_ready: bool,
    pub(crate) certificate_policy_last_evaluated_at: Option<u64>,
    pub(crate) certificate_policy_evaluated_tip_height: Option<u64>,
    pub(crate) certificate_witnesses_configured: usize,
    pub(crate) certificate_minimum_signers: usize,
    pub(crate) last_certificate_sync_at: Option<u64>,
    pub(crate) last_certificate_sync_result: Option<&'static str>,
    pub(crate) last_certificate_carrier_recovered_at: Option<u64>,
    pub(crate) certificate_sync_rounds_total: u64,
    pub(crate) certificate_coordinator_success_total: u64,
    pub(crate) certificate_carrier_attempts_total: u64,
    pub(crate) certificate_carrier_recoveries_total: u64,
    pub(crate) certificate_verified_unpersisted_total: u64,
    pub(crate) certificate_availability_exhausted_total: u64,
    pub(crate) certificate_security_stops_total: u64,
    pub(crate) last_certificate_security_stop_at: Option<u64>,
    pub(crate) certificate_carrier_cooling_slots: usize,
    pub(crate) certificate_carrier_cooldown_skips_total: u64,
    pub(crate) certificate_carrier_half_open_attempts_total: u64,
    pub(crate) last_coordinator_certificate_backfill_at: Option<u64>,
    pub(crate) last_coordinator_certificate_backfill_result: Option<&'static str>,
    pub(crate) coordinator_certificate_backfill_rounds_total: u64,
    pub(crate) coordinator_certificate_backfill_persisted_total: u64,
    pub(crate) coordinator_certificate_backfill_verified_unpersisted_total: u64,
    pub(crate) coordinator_certificate_backfill_availability_exhausted_total: u64,
    pub(crate) coordinator_certificate_backfill_security_stops_total: u64,
    pub(crate) last_coordinator_certificate_backfill_security_stop_at: Option<u64>,
    pub(crate) coordinator_certificate_backfill_carrier_attempts_total: u64,
    pub(crate) coordinator_certificate_backfill_carrier_cooling_slots: usize,
    pub(crate) coordinator_certificate_backfill_carrier_cooldown_skips_total: u64,
    pub(crate) coordinator_certificate_backfill_carrier_half_open_attempts_total: u64,
    pub(crate) last_attempt_at: Option<u64>,
    pub(crate) last_success_at: Option<u64>,
    /// Configured maximum age of one producer convergence observation.
    ///
    /// [FOLLOWER-READINESS-FRESHNESS 2026-07-30 by Codex] This is derived
    /// solely from the operator's polling interval and never from peer input.
    pub(crate) follower_readiness_max_age_secs: Option<u64>,
    pub(crate) follower_convergence_confirmed_at: Option<u64>,
    pub(crate) follower_readiness_stale_after: Option<u64>,
    pub(crate) last_failure_at: Option<u64>,
    pub(crate) last_recovered_at: Option<u64>,
    pub(crate) next_poll_at: Option<u64>,
    pub(crate) consecutive_failures: u32,
    pub(crate) last_error_code: Option<String>,
    pub(crate) remote_tip_height: Option<u64>,
    pub(crate) pages_received_total: u64,
    pub(crate) blocks_received_total: u64,
    pub(crate) failure_events_total: u64,
    pub(crate) recovery_events_total: u64,
    pub(crate) next_event_sequence: u64,
    pub(crate) recent_events: VecDeque<RecordCommitmentSyncEvent>,
}

impl Default for RecordCommitmentSyncRuntime {
    fn default() -> Self {
        Self {
            role: "verifier",
            state: "disabled",
            enabled: false,
            last_trigger: "none",
            last_announcement_at: None,
            last_announced_height: None,
            last_announcement_result: None,
            announcements_accepted_total: 0,
            announcements_coalesced_total: 0,
            announcements_stale_total: 0,
            announcements_unavailable_total: 0,
            last_outbound_announcement_at: None,
            last_outbound_announced_height: None,
            last_outbound_announcement_result: None,
            outbound_announcement_rounds_total: 0,
            outbound_announcement_rounds_skipped_total: 0,
            outbound_announcement_rounds_superseded_total: 0,
            outbound_announcements_attempted_total: 0,
            outbound_announcements_accepted_total: 0,
            outbound_announcements_stale_total: 0,
            outbound_announcements_failed_total: 0,
            outbound_announcement_retries_attempted_total: 0,
            outbound_announcement_retries_succeeded_total: 0,
            outbound_announcement_retries_exhausted_total: 0,
            last_authority_sync_at: None,
            last_authority_sync_result: None,
            last_authority_carrier_recovered_at: None,
            authority_sync_rounds_total: 0,
            authority_coordinator_success_total: 0,
            authority_carrier_attempts_total: 0,
            authority_carrier_recoveries_total: 0,
            authority_availability_exhausted_total: 0,
            authority_security_stops_total: 0,
            last_authority_security_stop_at: None,
            authority_carrier_cooling_slots: 0,
            authority_carrier_cooldown_skips_total: 0,
            authority_carrier_half_open_attempts_total: 0,
            last_block_page_pull_at: None,
            last_block_page_pull_result: None,
            last_block_carrier_recovered_at: None,
            block_page_pulls_total: 0,
            block_page_coordinator_success_total: 0,
            block_carrier_attempts_total: 0,
            block_carrier_recoveries_total: 0,
            block_page_availability_exhausted_total: 0,
            block_page_security_stops_total: 0,
            last_block_page_security_stop_at: None,
            block_carrier_cooling_slots: 0,
            block_carrier_cooldown_skips_total: 0,
            block_carrier_half_open_attempts_total: 0,
            certificate_policy_state: "not_applicable",
            certificate_policy_ready: false,
            certificate_policy_last_evaluated_at: None,
            certificate_policy_evaluated_tip_height: None,
            certificate_witnesses_configured: 0,
            certificate_minimum_signers: 0,
            last_certificate_sync_at: None,
            last_certificate_sync_result: None,
            last_certificate_carrier_recovered_at: None,
            certificate_sync_rounds_total: 0,
            certificate_coordinator_success_total: 0,
            certificate_carrier_attempts_total: 0,
            certificate_carrier_recoveries_total: 0,
            certificate_verified_unpersisted_total: 0,
            certificate_availability_exhausted_total: 0,
            certificate_security_stops_total: 0,
            last_certificate_security_stop_at: None,
            certificate_carrier_cooling_slots: 0,
            certificate_carrier_cooldown_skips_total: 0,
            certificate_carrier_half_open_attempts_total: 0,
            last_coordinator_certificate_backfill_at: None,
            last_coordinator_certificate_backfill_result: None,
            coordinator_certificate_backfill_rounds_total: 0,
            coordinator_certificate_backfill_persisted_total: 0,
            coordinator_certificate_backfill_verified_unpersisted_total: 0,
            coordinator_certificate_backfill_availability_exhausted_total: 0,
            coordinator_certificate_backfill_security_stops_total: 0,
            last_coordinator_certificate_backfill_security_stop_at: None,
            coordinator_certificate_backfill_carrier_attempts_total: 0,
            coordinator_certificate_backfill_carrier_cooling_slots: 0,
            coordinator_certificate_backfill_carrier_cooldown_skips_total: 0,
            coordinator_certificate_backfill_carrier_half_open_attempts_total: 0,
            last_attempt_at: None,
            last_success_at: None,
            follower_readiness_max_age_secs: None,
            follower_convergence_confirmed_at: None,
            follower_readiness_stale_after: None,
            last_failure_at: None,
            last_recovered_at: None,
            next_poll_at: None,
            consecutive_failures: 0,
            last_error_code: None,
            remote_tip_height: None,
            pages_received_total: 0,
            blocks_received_total: 0,
            failure_events_total: 0,
            recovery_events_total: 0,
            next_event_sequence: 1,
            recent_events: VecDeque::with_capacity(COMMITMENT_SYNC_EVENT_CAPACITY),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RecordCommitmentCheckpointRuntime {
    pub(crate) state: &'static str,
    pub(crate) last_checked_at: Option<u64>,
    pub(crate) last_converged_at: Option<u64>,
    pub(crate) last_divergence_at: Option<u64>,
    pub(crate) last_failure_at: Option<u64>,
    pub(crate) last_served_at: Option<u64>,
    pub(crate) local_tip_height: Option<u64>,
    pub(crate) remote_tip_height: Option<u64>,
    pub(crate) proofs_verified_total: u64,
    pub(crate) proofs_failed_total: u64,
    pub(crate) divergences_total: u64,
    pub(crate) requests_served_total: u64,
    pub(crate) evidence_state: &'static str,
    pub(crate) evidence_records: u64,
    pub(crate) applicable_evidence_records: u64,
    pub(crate) deferred_evidence_records: u64,
    pub(crate) divergence_evidence_records: u64,
    pub(crate) equivocation_incidents: u64,
    pub(crate) trusted_divergence_incidents: u64,
    pub(crate) checkpoint_certificates: u64,
    pub(crate) latest_certified_height: Option<u64>,
    pub(crate) latest_certificate_signers: usize,
    pub(crate) latest_certificate_required_signers: usize,
    pub(crate) last_evidence_at: Option<u64>,
    pub(crate) last_round_state: &'static str,
    pub(crate) last_round_at: Option<u64>,
    pub(crate) last_round_eligible: usize,
    pub(crate) last_round_attempted: usize,
    pub(crate) last_round_verified: usize,
    pub(crate) last_round_failed: usize,
    pub(crate) last_round_converged: usize,
    pub(crate) last_round_remote_ahead: usize,
    pub(crate) last_round_remote_behind: usize,
    pub(crate) last_round_diverged: usize,
    pub(crate) evidence_persistence_failures_total: u64,
}

impl Default for RecordCommitmentCheckpointRuntime {
    fn default() -> Self {
        Self {
            state: "not_checked",
            last_checked_at: None,
            last_converged_at: None,
            last_divergence_at: None,
            last_failure_at: None,
            last_served_at: None,
            local_tip_height: None,
            remote_tip_height: None,
            proofs_verified_total: 0,
            proofs_failed_total: 0,
            divergences_total: 0,
            requests_served_total: 0,
            evidence_state: "not_audited",
            evidence_records: 0,
            applicable_evidence_records: 0,
            deferred_evidence_records: 0,
            divergence_evidence_records: 0,
            equivocation_incidents: 0,
            trusted_divergence_incidents: 0,
            checkpoint_certificates: 0,
            latest_certified_height: None,
            latest_certificate_signers: 0,
            latest_certificate_required_signers: 0,
            last_evidence_at: None,
            last_round_state: "not_checked",
            last_round_at: None,
            last_round_eligible: 0,
            last_round_attempted: 0,
            last_round_verified: 0,
            last_round_failed: 0,
            last_round_converged: 0,
            last_round_remote_ahead: 0,
            last_round_remote_behind: 0,
            last_round_diverged: 0,
            evidence_persistence_failures_total: 0,
        }
    }
}

/// Last complete commitment-chain verification known to this process.
///
/// This state is deliberately runtime-only: a restart must independently
/// re-audit SQLite before it can claim a verified baseline. The tip hash is
/// retained only inside this process so proof serving can detect same-height
/// SQLite tampering; it is never serialized, logged, or reported. The
/// structure must never gain identities, commitment IDs, owner information,
/// payloads, peers, or endpoints.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecordCommitmentIntegrityRuntime {
    pub(crate) baseline_verified_at: u64,
    pub(crate) last_verified_at: u64,
    pub(crate) verification_duration_ms: u64,
    pub(crate) verified_block_count: u64,
    pub(crate) verified_commitment_count: u64,
    pub(crate) verified_tip_height: u64,
    pub(crate) verified_tip_hash: [u8; 32],
}

/// Runtime-only ownership of the local coordinator production fence.
///
/// `handle` owns the kernel advisory lock and must remain alive for the whole
/// storage lifetime. The file path, process id, host identity, and lock errno
/// are deliberately not retained, serialized, logged, or reported.
pub(crate) struct RecordCommitmentCoordinatorFenceRuntime {
    pub(crate) handle: Option<Flock<File>>,
    pub(crate) state: &'static str,
    pub(crate) acquired_at: Option<u64>,
    pub(crate) acquisition_failures_total: u64,
}

impl Default for RecordCommitmentCoordinatorFenceRuntime {
    fn default() -> Self {
        Self {
            handle: None,
            state: "unconfigured",
            acquired_at: None,
            acquisition_failures_total: 0,
        }
    }
}

/// Runtime-only coordinator lease validity and aggregate telemetry.
///
/// `valid_until` uses the local monotonic clock so wall-clock changes cannot
/// extend this process's production authority. The random instance id and
/// witness identities remain in the lease task and are never reported.
pub(crate) struct RecordCommitmentCoordinatorLeaseRuntime {
    pub(crate) required: bool,
    pub(crate) state: &'static str,
    pub(crate) granted_witnesses: usize,
    pub(crate) required_witnesses: usize,
    pub(crate) expires_at: Option<u64>,
    pub(crate) valid_until: Option<Instant>,
    pub(crate) last_attempted_at: Option<u64>,
    pub(crate) last_renewed_at: Option<u64>,
    pub(crate) last_failure_at: Option<u64>,
    pub(crate) renewal_failures_total: u64,
    pub(crate) consecutive_failures: u64,
    pub(crate) recoveries_total: u64,
}

/// One process-local serialized clock authority for witness lease decisions.
///
/// [MEMCHAIN-WITNESS-CLOCK 2026-09-05 by Codex] Durable wall timestamps remain
/// wire/data compatible, but an active holder is fenced by a monotonic deadline.
/// On restart every unreleased row receives a fresh conservative hold before a
/// competing grant can be evaluated. The advisory lock prevents two
/// `MemoryStorage` handles from independently interpreting the same database.
pub(crate) struct RecordCommitmentWitnessLeaseClockRuntime {
    pub(crate) handle: Option<Flock<File>>,
    pub(crate) initialized_chains: HashSet<[u8; 32]>,
    pub(crate) holds: HashMap<[u8; 32], RecordCommitmentWitnessLeaseHold>,
}

impl Default for RecordCommitmentWitnessLeaseClockRuntime {
    fn default() -> Self {
        Self {
            handle: None,
            initialized_chains: HashSet::new(),
            holds: HashMap::new(),
        }
    }
}

/// Monotonic refusal window for one durable chain lease.
pub(crate) struct RecordCommitmentWitnessLeaseHold {
    pub(crate) coordinator: [u8; 32],
    pub(crate) instance_id: [u8; 32],
    pub(crate) lease_epoch: u64,
    pub(crate) valid_until: Instant,
}

/// Stable host-local identity of the on-disk SQLite main database.
///
/// [MEMCHAIN-WITNESS-DB-IDENTITY 2026-09-05 by Codex] Witness authority uses
/// device/inode identity rather than a caller-controlled path spelling. The
/// values remain process-local and are never serialized, logged, or exposed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageDatabaseFileIdentity {
    pub(crate) device: u64,
    pub(crate) inode: u64,
}

/// Opens the final database component without following a symlink and returns
/// a single-link, current-user regular-file identity. `Ok(None)` means the
/// candidate does not exist yet; all other uncertainty is fail-closed.
pub(crate) fn probe_storage_database_file_identity(
    path: &Path,
) -> Result<Option<StorageDatabaseFileIdentity>, ()> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_CLOEXEC | nix::libc::O_NOFOLLOW)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(()),
    };
    let metadata = file.metadata().map_err(|_| ())?;
    if !metadata.file_type().is_file()
        || metadata.nlink() != 1
        || metadata.uid() != unsafe { nix::libc::geteuid() }
    {
        return Err(());
    }
    Ok(Some(StorageDatabaseFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }))
}

impl Default for RecordCommitmentCoordinatorLeaseRuntime {
    fn default() -> Self {
        Self {
            required: false,
            state: "disabled",
            granted_witnesses: 0,
            required_witnesses: 0,
            expires_at: None,
            valid_until: None,
            last_attempted_at: None,
            last_renewed_at: None,
            last_failure_at: None,
            renewal_failures_total: 0,
            consecutive_failures: 0,
            recoveries_total: 0,
        }
    }
}

/// Private coordinator material required to advance the local signed anchor.
///
/// The identity is cloned only while writing a new anchor, then dropped. This
/// object must never derive `Debug`, serialize, enter logs, or cross an API.
pub(crate) struct RecordCommitmentTipAnchorConfig {
    pub(crate) path: PathBuf,
    pub(crate) identity: IdentityKeyPair,
}

/// Runtime-only state for the cross-restart commitment-tip rollback guard.
///
/// The persisted sidecar contains a signed tip, but public runtime reporting is
/// intentionally aggregate. `config` is private key material and must remain
/// excluded from every API/heartbeat status structure.
pub(crate) struct RecordCommitmentTipAnchorRuntime {
    pub(crate) config: Option<RecordCommitmentTipAnchorConfig>,
    pub(crate) state: &'static str,
    pub(crate) anchored_height: u64,
    pub(crate) last_verified_at: Option<u64>,
    pub(crate) last_persisted_at: Option<u64>,
    pub(crate) write_failures_total: u64,
}

impl Default for RecordCommitmentTipAnchorRuntime {
    fn default() -> Self {
        Self {
            config: None,
            state: "disabled",
            anchored_height: 0,
            last_verified_at: None,
            last_persisted_at: None,
            write_failures_total: 0,
        }
    }
}

/// Private coordinator material required to advance the certificate anchor.
///
/// The path and identity must never enter status, heartbeat, logs, or errors.
/// This is deliberately separate from the block-tip anchor configuration so
/// independently committed certificate writes cannot overwrite tip state.
pub(crate) struct RecordCommitmentCheckpointCertificateAnchorConfig {
    pub(crate) path: PathBuf,
    pub(crate) identity: IdentityKeyPair,
}

/// Runtime-only state for cross-restart certificate-vault rollback detection.
pub(crate) struct RecordCommitmentCheckpointCertificateAnchorRuntime {
    pub(crate) config: Option<RecordCommitmentCheckpointCertificateAnchorConfig>,
    pub(crate) state: &'static str,
    pub(crate) anchored_height: u64,
    pub(crate) last_verified_at: Option<u64>,
    pub(crate) last_persisted_at: Option<u64>,
    pub(crate) write_failures_total: u64,
}

impl Default for RecordCommitmentCheckpointCertificateAnchorRuntime {
    fn default() -> Self {
        Self {
            config: None,
            state: "disabled",
            anchored_height: 0,
            last_verified_at: None,
            last_persisted_at: None,
            write_failures_total: 0,
        }
    }
}

// ============================================
// LRU Cache
// ============================================

pub(crate) struct LruCache {
    map: HashMap<[u8; 32], (usize, MemoryRecord)>,
    order_counter: usize,
    capacity: usize,
}

impl LruCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            order_counter: 0,
            capacity,
        }
    }

    pub fn get(&mut self, id: &[u8; 32]) -> Option<&MemoryRecord> {
        if let Some(entry) = self.map.get_mut(id) {
            self.order_counter += 1;
            entry.0 = self.order_counter;
            Some(&entry.1)
        } else {
            None
        }
    }

    pub fn put(&mut self, record: MemoryRecord) {
        let id = record.record_id;
        self.order_counter += 1;
        self.map.insert(id, (self.order_counter, record));
        if self.map.len() > self.capacity {
            if let Some((&evict_id, _)) = self.map.iter().min_by_key(|(_, (ord, _))| *ord) {
                self.map.remove(&evict_id);
            }
        }
    }

    pub fn invalidate(&mut self, id: &[u8; 32]) {
        self.map.remove(id);
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order_counter = 0;
    }
}

// ============================================
// MemoryStorage
// ============================================

pub struct MemoryStorage {
    pub(crate) conn: TokioMutex<Connection>,
    /// On-disk `SQLite` path used only to derive the private coordinator fence.
    /// `None` denotes an isolated in-memory database used by tests/tools.
    pub(crate) database_path: Option<PathBuf>,
    /// Verified main-database identity captured across SQLite open. `None`
    /// for in-memory or unsafe/ambiguous file candidates; ordinary storage
    /// remains compatible, while witness authority refuses such candidates.
    pub(crate) database_identity: Option<StorageDatabaseFileIdentity>,
    pub(crate) total_inserted: AtomicU64,
    pub(crate) total_rejected: AtomicU64,
    pub(crate) cache: RwLock<LruCache>,
    /// Wrapped in Zeroizing so the key bytes are wiped from memory on drop.
    /// (P2 SecAudit: prevents key exposure in core dumps / process memory scans)
    pub(crate) record_key: Option<Zeroizing<[u8; 32]>>,
    /// Runtime-only Block Sync status. Never persisted and never stores peer
    /// identity, endpoint, block hash, commitment, owner, or payload metadata.
    pub(crate) commitment_sync: RwLock<RecordCommitmentSyncRuntime>,
    /// Runtime-only complete-chain audit baseline. Cleared before every audit
    /// and advanced only after an atomic, fully validated block append.
    pub(crate) commitment_integrity: RwLock<Option<RecordCommitmentIntegrityRuntime>>,
    /// Immutable process-local trust anchor for commitment proposer authority.
    ///
    /// [COMMITMENT-AUTHORITY-RUNTIME 2026-08-14 by Codex] The root comes from
    /// validated operator configuration or the role's backward-compatible
    /// identity fallback. It is never inferred from mutable SQLite state and
    /// is never serialized, logged, or exposed through management telemetry.
    pub(crate) commitment_authority_root: RwLock<Option<[u8; 32]>>,
    /// Effective SQLite `PRAGMA synchronous` level for this process. This is
    /// aggregate configuration evidence only and never contains chain data.
    pub(crate) commitment_durability: AtomicU64,
    /// Kernel-owned exclusive coordinator lock. It prevents a second local
    /// process from producing blocks or replacing sidecars for this database.
    pub(crate) commitment_coordinator_fence: RwLock<RecordCommitmentCoordinatorFenceRuntime>,
    /// Witness-backed cross-host production authority. This remains disabled
    /// unless the operator explicitly enables strict coordinator leasing.
    pub(crate) commitment_coordinator_lease: RwLock<RecordCommitmentCoordinatorLeaseRuntime>,
    /// Witness-side clock/lock authority. It is deliberately independent from
    /// the coordinator's production-side lease telemetry above.
    pub(crate) commitment_witness_lease_clock: TokioMutex<RecordCommitmentWitnessLeaseClockRuntime>,
    /// Signed local high-water mark outside SQLite. The private config never
    /// leaves this process; only aggregate status is reportable.
    pub(crate) commitment_tip_anchor: RwLock<RecordCommitmentTipAnchorRuntime>,
    /// Runtime-only aggregate signed-checkpoint reconciliation evidence.
    pub(crate) commitment_checkpoint: RwLock<RecordCommitmentCheckpointRuntime>,
    /// Signed high-water mark for the latest audited checkpoint certificate.
    /// Private key material remains process-local; APIs expose aggregates only.
    pub(crate) commitment_checkpoint_certificate_anchor:
        RwLock<RecordCommitmentCheckpointCertificateAnchorRuntime>,
    /// Serializes certificate DB commits with sidecar replacement so two
    /// concurrent witness/import paths cannot persist anchors out of order.
    pub(crate) commitment_checkpoint_certificate_anchor_write: TokioMutex<()>,
    /// One-way process-local safety latch. It is set while the incident write
    /// still owns the SQLite connection, closing the append race window.
    pub(crate) commitment_production_halted: AtomicBool,
    /// Optional SaaS managed-volume byte-growth policy. It contains no owner
    /// key and its implementation must never expose the configured path.
    growth_admission: Option<Arc<dyn StorageGrowthAdmission>>,
}

// [ARCH-SPLIT 2026-10-02] Child modules keep the same call paths.
mod record_read;
mod record_write;
mod schema;

pub(crate) use record_read::bytes_to_embedding;
pub(crate) use record_read::embedding_to_bytes;

impl MemoryStorage {
    pub fn open(path: impl AsRef<Path>, record_key: Option<[u8; 32]>) -> Result<Self, String> {
        let path = path.as_ref();

        let (conn, database_identity_before) = if path.to_str() == Some(":memory:") {
            (Connection::open_in_memory(), None)
        } else {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        format!(
                            "Failed to create DB directory '{}': {}",
                            parent.display(),
                            e
                        )
                    })?;
                }
            }
            let identity = Some(probe_storage_database_file_identity(path));
            (Connection::open(path), identity)
        };
        let conn = conn.map_err(|e| format!("Failed to open SQLite: {}", e))?;

        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -8000;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;",
        )
        .map_err(|e| format!("Failed to set pragmas: {}", e))?;

        Self::create_schema(&conn)?;
        Self::maybe_migrate(&conn)?;

        let database_identity = database_identity_before.and_then(|before| {
            let after = probe_storage_database_file_identity(path);
            match (before, after) {
                (Ok(None), Ok(Some(after))) => Some(after),
                (Ok(Some(before)), Ok(Some(after))) if before == after => Some(after),
                _ => None,
            }
        });

        let mode = if record_key.is_some() {
            "encrypted"
        } else {
            "plaintext"
        };
        // [VOLUME-GROWTH-ADMISSION 2026-08-31 by Codex] Managed DB filenames
        // are derived from owner keys. Log only mode/schema, never the path.
        info!(
            mode = mode,
            "[STORAGE] ✅ SQLite opened (schema v{})", SCHEMA_VERSION
        );

        Ok(Self {
            conn: TokioMutex::new(conn),
            database_path: (path.to_str() != Some(":memory:")).then(|| path.to_path_buf()),
            database_identity,
            total_inserted: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            cache: RwLock::new(LruCache::new(LRU_CACHE_CAPACITY)),
            record_key: record_key.map(Zeroizing::new),
            commitment_sync: RwLock::new(RecordCommitmentSyncRuntime::default()),
            commitment_integrity: RwLock::new(None),
            commitment_authority_root: RwLock::new(None),
            // `open` explicitly configures NORMAL. A coordinator upgrades this
            // to FULL and verifies the effective value before startup audit.
            commitment_durability: AtomicU64::new(1),
            commitment_coordinator_fence: RwLock::new(
                RecordCommitmentCoordinatorFenceRuntime::default(),
            ),
            commitment_coordinator_lease: RwLock::new(
                RecordCommitmentCoordinatorLeaseRuntime::default(),
            ),
            commitment_witness_lease_clock: TokioMutex::new(
                RecordCommitmentWitnessLeaseClockRuntime::default(),
            ),
            commitment_tip_anchor: RwLock::new(RecordCommitmentTipAnchorRuntime::default()),
            commitment_checkpoint: RwLock::new(RecordCommitmentCheckpointRuntime::default()),
            commitment_checkpoint_certificate_anchor: RwLock::new(
                RecordCommitmentCheckpointCertificateAnchorRuntime::default(),
            ),
            commitment_checkpoint_certificate_anchor_write: TokioMutex::new(()),
            commitment_production_halted: AtomicBool::new(false),
            growth_admission: None,
        })
    }

    /// Attach the internal managed-volume growth policy without changing the
    /// public `open` constructor or Local/global storage semantics.
    pub(crate) fn with_growth_admission(
        mut self,
        admission: Arc<dyn StorageGrowthAdmission>,
    ) -> Self {
        self.growth_admission = Some(admission);
        self
    }

    /// Acquire a permit for a logical operation that may add managed bytes.
    /// Reads, deletion, and recovery deliberately do not call this method.
    pub(crate) async fn acquire_growth_permit(
        &self,
        minimum_growth_bytes: u64,
    ) -> Result<StorageGrowthPermit, StorageGrowthError> {
        match &self.growth_admission {
            Some(admission) => admission.acquire(minimum_growth_bytes).await,
            None => Ok(StorageGrowthPermit::unmanaged()),
        }
    }

    /// Acquire the inner SQLite connection lock.
    ///
    /// ## P3 SecAudit: pub(crate) only
    /// Exposing raw Connection externally lets callers bypass owner checks,
    /// encryption, and cache invalidation. Restricted to crate-internal use.
    pub(crate) async fn conn_lock(&self) -> tokio::sync::MutexGuard<'_, Connection> {
        self.conn.lock().await
    }

    pub fn total_inserted(&self) -> u64 {
        self.total_inserted.load(Ordering::Relaxed)
    }
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected.load(Ordering::Relaxed)
    }

    // ========================================
    // Private helpers
    // ========================================

    const SELECT_RECORD_COLS: &'static str =
        "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                status,supersedes,encrypted_content,embedding,signature,access_count,
                positive_feedback,negative_feedback,conflict_with,blind
         FROM records WHERE record_id = ?1";
}

impl std::fmt::Debug for MemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStorage")
            .field("inserted", &self.total_inserted())
            .field("rejected", &self.total_rejected())
            .field("encrypted", &self.record_key.is_some())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    mod migration;
    mod other;
    mod records;

    use super::*;

    fn make_rec(ts: u64, layer: MemoryLayer, src: &str) -> MemoryRecord {
        MemoryRecord::new(
            [0xAA; 32],
            ts,
            layer,
            vec!["test".into()],
            src.into(),
            b"encrypted_data".to_vec(),
            vec![0.1, 0.2, 0.3],
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
}
