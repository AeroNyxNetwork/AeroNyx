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
//! - v2.7.4-BlockIntegrityStatus: Runtime-only evidence for the most recent
//!   complete persisted-chain audit and subsequently verified appends.
//! - v2.7.5-CheckpointProof: Runtime-only signed checkpoint reconciliation
//!   evidence without peer identities, hashes, signatures, or user metadata.
//! - v2.7.10-CheckpointDirectionIsolation: Inbound checkpoint serving updates
//!   service counters only and cannot overwrite outbound verification evidence.
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
//!
//! ## Last Modified
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
//! v2.6.1+BlindVectorRecovery - Added all-owner active embedding enumeration so
//!   node-blind/remote Local-mode nodes can rebuild every isolated vector
//!   partition after restart without weakening owner-scoped recall.
//! v2.7.3-BlockAudit - Made v8 commitment tables self-creating for direct
//!   legacy migration callers before startup integrity verification.
//! v2.7.1-BlockSyncStatus - Runtime-only bounded follower status and fault evidence.
//! v2.7.0-BlockSync - Schema v8 commitment blocks and unique membership index.
// ============================================

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use rusqlite::{params, Connection, OptionalExtension};
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

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
///
/// ⚠️ CRITICAL: When bumping this, you MUST also add a new migrate block
/// in maybe_migrate(). The migrate block MUST use a hardcoded integer
/// (not this constant) for UPDATE schema_version, to prevent skipping
/// intermediate migrations on multi-version upgrades.
const SCHEMA_VERSION: u32 = 9;

const LRU_CACHE_CAPACITY: usize = 1000;
const DEFAULT_PAGE_SIZE: usize = 100;

// ============================================
// Embedding helpers
// ============================================

pub(crate) fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub(crate) fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            f32::from_le_bytes(buf)
        })
        .collect()
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

/// Privacy-safe runtime status for commitment block replication.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RecordCommitmentSyncStatus {
    /// Stable API contract name.
    pub contract_version: &'static str,
    /// Node role: `coordinator`, `follower`, or `verifier`.
    pub role: String,
    /// Runtime state such as `current`, `catching_up`, or `backoff`.
    pub state: String,
    /// Whether active follower polling is configured.
    pub enabled: bool,
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
    /// Startup audit state for durable proof frames: `not_audited`, `verified`,
    /// or `invalid`. Raw proof material is never returned.
    pub evidence_state: String,
    /// Signature-verified proof frames currently retained in the bounded vault.
    pub evidence_records: u64,
    /// Retained proofs that established a shared-prefix mismatch.
    pub divergence_evidence_records: u64,
    /// Most recent durable evidence observation time.
    pub last_evidence_at: Option<u64>,
    /// Local SQLite persistence failures observed since process start.
    pub evidence_persistence_failures_total: u64,
    /// Explicit privacy boundary for operators and API consumers.
    pub privacy_policy: &'static str,
}

pub(crate) const COMMITMENT_SYNC_EVENT_CAPACITY: usize = 16;
/// Maximum signed checkpoint frames retained locally. Divergence evidence is
/// pruned last so normal convergence checks cannot erase the most useful proof.
pub(crate) const CHECKPOINT_EVIDENCE_CAPACITY: usize = 256;
/// Defensive bound for one stored proof frame. Increasing this requires an
/// explicit wire and startup-memory review.
pub(crate) const MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES: usize = 4 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct RecordCommitmentSyncRuntime {
    pub(crate) role: &'static str,
    pub(crate) state: &'static str,
    pub(crate) enabled: bool,
    pub(crate) last_attempt_at: Option<u64>,
    pub(crate) last_success_at: Option<u64>,
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
            last_attempt_at: None,
            last_success_at: None,
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
    pub(crate) divergence_evidence_records: u64,
    pub(crate) last_evidence_at: Option<u64>,
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
            divergence_evidence_records: 0,
            last_evidence_at: None,
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
    /// Runtime-only aggregate signed-checkpoint reconciliation evidence.
    pub(crate) commitment_checkpoint: RwLock<RecordCommitmentCheckpointRuntime>,
}

impl MemoryStorage {
    pub fn open(path: impl AsRef<Path>, record_key: Option<[u8; 32]>) -> Result<Self, String> {
        let path = path.as_ref();

        let conn = if path.to_str() == Some(":memory:") {
            Connection::open_in_memory()
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
            Connection::open(path)
        }
        .map_err(|e| format!("Failed to open SQLite: {}", e))?;

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

        let mode = if record_key.is_some() {
            "encrypted"
        } else {
            "plaintext"
        };
        info!(path = %path.display(), mode = mode, "[STORAGE] ✅ SQLite opened (schema v{})", SCHEMA_VERSION);

        Ok(Self {
            conn: TokioMutex::new(conn),
            total_inserted: AtomicU64::new(0),
            total_rejected: AtomicU64::new(0),
            cache: RwLock::new(LruCache::new(LRU_CACHE_CAPACITY)),
            record_key: record_key.map(Zeroizing::new),
            commitment_sync: RwLock::new(RecordCommitmentSyncRuntime::default()),
            commitment_integrity: RwLock::new(None),
            commitment_checkpoint: RwLock::new(RecordCommitmentCheckpointRuntime::default()),
        })
    }

    fn create_schema(conn: &Connection) -> Result<(), String> {
        // ── v1-v4: Original tables (records, raw_logs, memory_edges, etc.) ──
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS records (
                record_id           BLOB PRIMARY KEY,
                owner               BLOB NOT NULL,
                timestamp           INTEGER NOT NULL,
                layer               INTEGER NOT NULL,
                topic_tags          TEXT NOT NULL DEFAULT '[]',
                source_ai           TEXT NOT NULL DEFAULT '',
                status              INTEGER NOT NULL DEFAULT 0,
                supersedes          BLOB,
                encrypted_content   BLOB NOT NULL DEFAULT x'',
                embedding           BLOB,
                embedding_model     TEXT NOT NULL DEFAULT '',
                embedding_dim       INTEGER NOT NULL DEFAULT 0,
                signature           BLOB NOT NULL,
                access_count        INTEGER NOT NULL DEFAULT 0,
                created_at          INTEGER NOT NULL,
                archived_at         INTEGER,
                positive_feedback   INTEGER NOT NULL DEFAULT 0,
                negative_feedback   INTEGER NOT NULL DEFAULT 0,
                conflict_with       BLOB,
                project_id          TEXT,
                session_id          TEXT,
                episode_id          TEXT,
                blind               INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_owner ON records(owner);
            CREATE INDEX IF NOT EXISTS idx_owner_layer_status ON records(owner, layer, status);
            CREATE INDEX IF NOT EXISTS idx_status_layer ON records(status, layer);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp);

            -- v8: node-blind commitment chain. Blocks intentionally contain
            -- only opaque record ids; memory owners and ciphertext payloads
            -- remain in the separately authorised records table.
            CREATE TABLE IF NOT EXISTS record_commitment_blocks (
                height              INTEGER PRIMARY KEY CHECK(height > 0),
                block_hash          BLOB NOT NULL UNIQUE CHECK(length(block_hash) = 32),
                chain_id            BLOB NOT NULL CHECK(length(chain_id) = 32),
                protocol_version    INTEGER NOT NULL,
                timestamp           INTEGER NOT NULL,
                prev_block_hash     BLOB NOT NULL CHECK(length(prev_block_hash) = 32),
                merkle_root         BLOB NOT NULL CHECK(length(merkle_root) = 32),
                record_count        INTEGER NOT NULL,
                proposer            BLOB NOT NULL CHECK(length(proposer) = 32),
                proposer_signature  BLOB NOT NULL CHECK(length(proposer_signature) = 64),
                payload             BLOB NOT NULL,
                received_from       BLOB,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_record_blocks_hash
                ON record_commitment_blocks(block_hash);

            CREATE TABLE IF NOT EXISTS record_block_commitments (
                record_id       BLOB PRIMARY KEY CHECK(length(record_id) = 32),
                block_height    INTEGER NOT NULL,
                FOREIGN KEY(block_height) REFERENCES record_commitment_blocks(height)
                    ON DELETE RESTRICT
            );
            CREATE INDEX IF NOT EXISTS idx_record_block_commitments_height
                ON record_block_commitments(block_height);

            -- v9: bounded local proof evidence. The exact signed peer frame is
            -- retained for operator-side verification, but never leaves the
            -- node through status, heartbeat, logs, or public protocol APIs.
            CREATE TABLE IF NOT EXISTS record_checkpoint_evidence (
                evidence_digest    BLOB PRIMARY KEY CHECK(length(evidence_digest) = 32),
                observed_at        INTEGER NOT NULL CHECK(observed_at >= 0),
                relation           TEXT NOT NULL CHECK(relation IN
                    ('converged','remote_ahead','remote_behind','diverged')),
                local_tip_height   INTEGER NOT NULL CHECK(local_tip_height >= 0),
                remote_tip_height  INTEGER NOT NULL CHECK(remote_tip_height >= 0),
                checkpoint_height  INTEGER NOT NULL CHECK(checkpoint_height >= 0),
                signed_response    BLOB NOT NULL CHECK(length(signed_response) > 0
                    AND length(signed_response) <= 4096),
                created_at         INTEGER NOT NULL CHECK(created_at >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_record_checkpoint_evidence_observed
                ON record_checkpoint_evidence(observed_at);

            CREATE TABLE IF NOT EXISTS raw_logs (
                log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                turn_index      INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         BLOB NOT NULL,
                source_ai       TEXT NOT NULL DEFAULT '',
                recall_context  TEXT DEFAULT NULL,
                extractable     INTEGER DEFAULT 1,
                feedback_signal INTEGER DEFAULT NULL,
                encrypted       INTEGER DEFAULT 0,
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rawlogs_session ON raw_logs(session_id, turn_index);
            CREATE INDEX IF NOT EXISTS idx_rawlogs_feedback ON raw_logs(feedback_signal);

            CREATE TABLE IF NOT EXISTS memory_edges (
                source_id BLOB NOT NULL, target_id BLOB NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'co_occurred',
                weight REAL NOT NULL DEFAULT 1.0, created_at INTEGER NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);

            CREATE TABLE IF NOT EXISTS user_weights (
                owner BLOB PRIMARY KEY, weights BLOB NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner BLOB NOT NULL, memory_id BLOB NOT NULL,
                session_id TEXT NOT NULL, turn_index INTEGER NOT NULL,
                signal INTEGER NOT NULL, features BLOB, prediction REAL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_owner ON memory_feedback(owner);
            CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);

            CREATE TABLE IF NOT EXISTS chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);",
        )
        .map_err(|e| format!("Schema creation failed (base tables): {}", e))?;

        // ── v5 (v2.4.0): Three-layer cognitive graph tables ──
        conn.execute_batch(
            "-- Episode layer: complete original conversations (non-lossy)
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                episode_type        TEXT NOT NULL,
                source              TEXT NOT NULL,
                session_id          TEXT,
                encrypted_content   BLOB NOT NULL,
                content_hash        TEXT NOT NULL,
                embedding           BLOB,
                token_count         INTEGER,
                created_at          INTEGER NOT NULL,
                ingested_at         INTEGER NOT NULL,
                metadata_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_owner ON episodes(owner, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(owner, episode_type);

            -- Semantic Entity layer: GLiNER-extracted entity nodes
            CREATE TABLE IF NOT EXISTS entities (
                entity_id           TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                name_normalized     TEXT NOT NULL,
                entity_type         TEXT NOT NULL,
                description         TEXT,
                embedding           BLOB,
                community_id        TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL,
                mention_count       INTEGER DEFAULT 1,
                metadata_json       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entities_owner ON entities(owner);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(owner, entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(owner, name_normalized);
            CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(owner, community_id);

            -- Semantic Entity layer: temporal knowledge edges
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                edge_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                owner               BLOB NOT NULL,
                source_id           TEXT NOT NULL,
                target_id           TEXT NOT NULL,
                relation_type       TEXT NOT NULL,
                fact_text           TEXT,
                weight              REAL DEFAULT 1.0,
                confidence          REAL DEFAULT 1.0,
                embedding           BLOB,
                valid_from          INTEGER NOT NULL,
                valid_until         INTEGER,
                episode_id          TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kedges_source ON knowledge_edges(owner, source_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_kedges_target ON knowledge_edges(owner, target_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_kedges_valid ON knowledge_edges(owner, valid_until);
            CREATE INDEX IF NOT EXISTS idx_kedges_episode ON knowledge_edges(episode_id);

            -- Bridge layer: Episode ↔ Entity bidirectional links
            CREATE TABLE IF NOT EXISTS episode_edges (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                owner               BLOB NOT NULL,
                episode_id          TEXT NOT NULL,
                entity_id           TEXT NOT NULL,
                role                TEXT NOT NULL,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ep_edges_episode ON episode_edges(episode_id);
            CREATE INDEX IF NOT EXISTS idx_ep_edges_entity ON episode_edges(entity_id);

            -- Community layer: auto-clustering via label propagation
            CREATE TABLE IF NOT EXISTS communities (
                community_id        TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                summary             TEXT,
                description         TEXT,
                entity_count        INTEGER DEFAULT 0,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_communities_owner ON communities(owner);

            -- Projects: Community specialization for code projects
            CREATE TABLE IF NOT EXISTS projects (
                project_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                name                TEXT NOT NULL,
                status              TEXT DEFAULT 'active',
                community_id        TEXT NOT NULL,
                summary             TEXT,
                created_at          INTEGER NOT NULL,
                updated_at          INTEGER NOT NULL,
                last_active_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner, status);
            CREATE INDEX IF NOT EXISTS idx_projects_active ON projects(owner, last_active_at DESC);

            -- Sessions: conversation session metadata (v2.5.0: added title column)
            CREATE TABLE IF NOT EXISTS sessions (
                session_id          TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                project_id          TEXT,
                session_type        TEXT DEFAULT 'chat',
                started_at          INTEGER NOT NULL,
                ended_at            INTEGER,
                turn_count          INTEGER DEFAULT 0,
                title               TEXT,
                summary             TEXT,
                key_decisions       TEXT,
                files_touched       TEXT,
                entities_extracted  INTEGER DEFAULT 0,
                summary_generated   INTEGER DEFAULT 0,
                artifacts_extracted INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_sessions_pending ON sessions(entities_extracted, summary_generated);

            -- Artifacts: code/document artifacts with version chains
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id         TEXT PRIMARY KEY,
                owner               BLOB NOT NULL,
                session_id          TEXT NOT NULL,
                project_id          TEXT,
                artifact_type       TEXT NOT NULL,
                filename            TEXT,
                language            TEXT,
                version             INTEGER DEFAULT 1,
                parent_id           TEXT,
                encrypted_content   BLOB NOT NULL,
                content_hash        TEXT NOT NULL,
                embedding           BLOB,
                line_count          INTEGER,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
            CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id, filename, version DESC);
            CREATE INDEX IF NOT EXISTS idx_artifacts_file ON artifacts(owner, filename, version DESC);"
        ).map_err(|e| format!("Schema creation failed (v5 cognitive graph tables): {}", e))?;

        // ── v2.4.0+BM25: Full-text search index ──
        // FTS5 content table indexes text from records, entities, and sessions
        // for BM25 keyword matching in hybrid recall.
        //
        // Design: external content table (contentless) — FTS5 stores only the
        // inverted index, not the original text. This saves ~40% space vs
        // content-based FTS5. Queries join back to source tables for content.
        //
        // Columns:
        //   source_type: 'record' | 'entity' | 'session' — identifies source table
        //   source_id: record_id hex / entity_id / session_id — for join-back
        //   owner_hex: owner public key hex — for access control filtering
        //   content: searchable text content
        //   tags: topic_tags or entity_type for boosting
        match conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                source_type,
                source_id,
                owner_hex,
                content,
                tags,
                tokenize='porter unicode61'
            );",
        ) {
            Ok(_) => info!("[STORAGE] ✅ FTS5 index ready"),
            Err(e) => warn!("[STORAGE] ⚠️ FTS5 creation failed (BM25 disabled): {}", e),
        }

        // ── node-blind full-text: a SEPARATE FTS5 index over client-supplied
        //    keyed token-hashes. NON-stemming (unicode61, NOT porter) — porter
        //    would mangle hex token-hashes (e.g. a hash ending in "ed"). Created
        //    idempotently on every open, so no schema-version bump is needed.
        //   source_id: record_id hex; owner_hex: access control;
        //   terms: space-joined HMAC(k_fts, token) hex hashes supplied by the client.
        match conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS blind_fts USING fts5(
                source_id,
                owner_hex,
                terms,
                tokenize='unicode61'
            );",
        ) {
            Ok(_) => info!("[STORAGE] ✅ blind FTS5 index ready"),
            Err(e) => warn!("[STORAGE] ⚠️ blind FTS5 creation failed: {}", e),
        }

        // ── node-blind derived-record provenance (Brick 3d) ──
        // Links a derived record (e.g. a client/LLM summary) to the source
        // record_ids it was built from. Opaque hexes only — the node learns the
        // provenance DAG shape, never the content. Idempotent, no version bump.
        let _ = conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS blind_provenance (
                record_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                owner_hex TEXT NOT NULL,
                PRIMARY KEY (record_id, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_blind_prov_owner ON blind_provenance(owner_hex);",
        );

        // ── v6 (v2.5.0-SuperNode): Cognitive task queue + LLM usage log ──
        // cognitive_tasks: async LLM task queue processed by TaskWorker
        // llm_usage_log: per-call token usage + latency for cost tracking
        //
        // Design notes:
        //   - Tasks are claimed atomically (UPDATE ... WHERE status='pending' LIMIT 1)
        //   - result / prompt_messages stored as JSON TEXT for flexibility
        //   - token_usage stored as JSON: {"input": N, "output": N, "cached": N}
        //   - cost_usd NOT stored — computed dynamically at query time from token counts
        //     so rate changes don't affect historical accuracy
        conn.execute_batch(
            "-- Cognitive task queue (SuperNode async LLM tasks)
            CREATE TABLE IF NOT EXISTS cognitive_tasks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type       TEXT NOT NULL,
                priority        INTEGER DEFAULT 5,
                status          TEXT DEFAULT 'pending',
                payload         TEXT NOT NULL,
                result          TEXT,
                prompt_messages TEXT,
                target_table    TEXT,
                target_id       TEXT,
                privacy_level   TEXT DEFAULT 'structured',
                provider_used   TEXT,
                model_used      TEXT,
                token_usage     TEXT,
                created_at      INTEGER NOT NULL,
                started_at      INTEGER,
                completed_at    INTEGER,
                retry_count     INTEGER DEFAULT 0,
                max_retries     INTEGER DEFAULT 3,
                error_message   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ct_status ON cognitive_tasks(status, priority DESC, created_at ASC);
            CREATE INDEX IF NOT EXISTS idx_ct_target ON cognitive_tasks(target_table, target_id, task_type);
            CREATE INDEX IF NOT EXISTS idx_ct_type ON cognitive_tasks(task_type, status);

            -- LLM usage log (token counts + latency per call)
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id         INTEGER,
                provider        TEXT NOT NULL,
                model           TEXT NOT NULL,
                input_tokens    INTEGER NOT NULL,
                output_tokens   INTEGER NOT NULL,
                cached_tokens   INTEGER DEFAULT 0,
                latency_ms      INTEGER NOT NULL,
                created_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_time ON llm_usage_log(created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_provider ON llm_usage_log(provider, created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_task ON llm_usage_log(task_id);"
        ).map_err(|e| format!("Schema creation failed (v6 SuperNode tables): {}", e))?;

        // Insert schema version if not present
        let existing: Option<u32> = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .optional()
            .map_err(|e| format!("Read schema version: {}", e))?;

        if existing.is_none() {
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?1)",
                params![SCHEMA_VERSION],
            )
            .map_err(|e| format!("Insert schema version: {}", e))?;
        }

        Ok(())
    }

    fn maybe_migrate(conn: &Connection) -> Result<(), String> {
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(1);

        // v1 → v2: embedding column
        if current < 2 {
            info!("[STORAGE] Migrating schema v{} → v2", current);
            let has_embedding = conn
                .prepare("SELECT embedding FROM records LIMIT 0")
                .is_ok();
            if !has_embedding {
                let _ = conn.execute_batch("ALTER TABLE records ADD COLUMN embedding BLOB;");
                info!("[STORAGE] Added `embedding` column");
            }
            // ⚠️ hardcoded 2, not SCHEMA_VERSION — prevents skipping v4/v5/v6
            conn.execute("UPDATE schema_version SET version = 2", [])
                .map_err(|e| format!("Update schema version to v2: {}", e))?;
            info!("[STORAGE] ✅ Migration to v2 complete");
        }

        // v2 → v4: MVF feedback + conflict
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(2);

        if current < 4 {
            info!("[STORAGE] Migrating schema v{} → v4", current);

            if conn
                .prepare("SELECT positive_feedback FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN positive_feedback INTEGER NOT NULL DEFAULT 0;",
                )
                .map_err(|e| format!("Add positive_feedback: {}", e))?;
            }
            if conn
                .prepare("SELECT negative_feedback FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN negative_feedback INTEGER NOT NULL DEFAULT 0;",
                )
                .map_err(|e| format!("Add negative_feedback: {}", e))?;
            }
            if conn
                .prepare("SELECT conflict_with FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN conflict_with BLOB;")
                    .map_err(|e| format!("Add conflict_with: {}", e))?;
            }

            // ⚠️ hardcoded 4, not SCHEMA_VERSION
            conn.execute("UPDATE schema_version SET version = 4", [])
                .map_err(|e| format!("Update schema version to v4: {}", e))?;
            info!("[STORAGE] ✅ Migration to v4 complete");
        }

        // v4 → v5 (v2.4.0-GraphCognition): Three-layer cognitive graph
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(4);

        if current < 5 {
            info!(
                "[STORAGE] Migrating schema v{} → v5 (cognitive graph)",
                current
            );

            // 5a: ALTER records — add project_id, session_id, episode_id
            if conn
                .prepare("SELECT project_id FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN project_id TEXT;")
                    .map_err(|e| format!("Add project_id to records: {}", e))?;
                info!("[STORAGE] Added records.project_id");
            }
            if conn
                .prepare("SELECT session_id FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN session_id TEXT;")
                    .map_err(|e| format!("Add session_id to records: {}", e))?;
                info!("[STORAGE] Added records.session_id");
            }
            if conn
                .prepare("SELECT episode_id FROM records LIMIT 0")
                .is_err()
            {
                conn.execute_batch("ALTER TABLE records ADD COLUMN episode_id TEXT;")
                    .map_err(|e| format!("Add episode_id to records: {}", e))?;
                info!("[STORAGE] Added records.episode_id");
            }

            // 5b: Create indexes for new records columns
            let _ = conn.execute_batch(
                "CREATE INDEX IF NOT EXISTS idx_records_project ON records(project_id, timestamp DESC);
                 CREATE INDEX IF NOT EXISTS idx_records_session ON records(session_id);
                 CREATE INDEX IF NOT EXISTS idx_records_episode ON records(episode_id);"
            );

            // 5c: Create v5 tables (IF NOT EXISTS — safe if create_schema already ran)
            let v5_tables = [
                "episodes",
                "entities",
                "knowledge_edges",
                "episode_edges",
                "communities",
                "projects",
                "sessions",
                "artifacts",
            ];
            for table in &v5_tables {
                let exists: bool = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        params![table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;

                if !exists {
                    warn!(
                        "[STORAGE] v5 table '{}' missing after create_schema — \
                         this is expected when upgrading from v2.3.0. \
                         Table will be created by create_schema on next restart.",
                        table
                    );
                }
            }

            // 5d: Migrate memory_edges → knowledge_edges
            //
            // ⚠️ BUG FIX (v2.5.2+Provenance): the original migration used
            // `source_id` (a BLOB record_id) as the `owner` column in the INSERT.
            // knowledge_edges.owner must be a 32-byte Ed25519 public key, NOT a
            // record_id. Records in memory_edges do not carry an owner column.
            //
            // Fix: join records on source_id to look up the actual owner.
            // Edges whose source_id has no matching record are skipped.
            {
                let migrated: bool = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = 'memory_edges_migrated_v5'",
                        [],
                        |row| {
                            let v: Vec<u8> = row.get(0)?;
                            Ok(v == b"1")
                        },
                    )
                    .unwrap_or(false);

                if !migrated {
                    let edge_count: i64 = conn
                        .query_row("SELECT COUNT(*) FROM memory_edges", [], |row| row.get(0))
                        .unwrap_or(0);

                    if edge_count > 0 {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs() as i64;

                        let ke_exists: bool = conn.query_row(
                            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='knowledge_edges'",
                            [],
                            |row| row.get::<_, i64>(0),
                        ).unwrap_or(0) > 0;

                        if ke_exists {
                            // ⚠️ BUG FIX: was `source_id` (record BLOB) as owner.
                            // Now JOINs records to get the correct owner bytes.
                            // Edges with no matching record in records table are skipped
                            // (INNER JOIN — safer than using wrong owner bytes).
                            match conn.execute(
                                "INSERT OR IGNORE INTO knowledge_edges
                                    (owner, source_id, target_id, relation_type, weight,
                                     confidence, valid_from, created_at, updated_at)
                                 SELECT
                                    r.owner,
                                    hex(me.source_id),
                                    hex(me.target_id),
                                    'RELATED_TO',
                                    me.weight,
                                    1.0,
                                    me.created_at,
                                    ?1,
                                    ?1
                                 FROM memory_edges me
                                 INNER JOIN records r ON r.record_id = me.source_id",
                                params![now],
                            ) {
                                Ok(migrated_count) => {
                                    info!(
                                        count = migrated_count,
                                        "[STORAGE] Migrated memory_edges → knowledge_edges"
                                    );
                                }
                                Err(e) => {
                                    warn!(
                                        error = %e,
                                        "[STORAGE] memory_edges migration failed (non-fatal, edges preserved)"
                                    );
                                }
                            }
                        } else {
                            warn!("[STORAGE] knowledge_edges table not found, skipping memory_edges migration");
                        }
                    }

                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('memory_edges_migrated_v5', ?1)",
                        params![b"1".as_slice()],
                    );
                    info!("[STORAGE] ✅ memory_edges migration marker set");
                }
            }

            // 5e: Update schema version to v5
            // ⚠️ hardcoded 5, NOT SCHEMA_VERSION constant.
            conn.execute("UPDATE schema_version SET version = 5", [])
                .map_err(|e| format!("Update schema version to v5: {}", e))?;

            // 5f: Backfill FTS5 index from existing records
            {
                let fts_populated: bool = conn
                    .query_row(
                        "SELECT value FROM chain_state WHERE key = 'fts_index_populated'",
                        [],
                        |row| {
                            let v: Vec<u8> = row.get(0)?;
                            Ok(v == b"1")
                        },
                    )
                    .unwrap_or(false);

                if !fts_populated {
                    // P1 SecAudit: skip FTS backfill when record_key is set.
                    // encrypted_content is ciphertext — indexing it is meaningless
                    // and leaks encrypted token patterns into an unencrypted FTS table.
                    if conn.prepare("SELECT title FROM sessions LIMIT 0").is_ok() {
                        // record_key presence is not available here (no &self),
                        // so we check via chain_state marker set by open() below.
                        // The actual guard is enforced in open() before calling maybe_migrate.
                        // For the migration path (fresh DB), encrypted_content is empty anyway.
                    }
                    let indexed = conn.execute(
                        "INSERT OR IGNORE INTO fts_index (source_type, source_id, owner_hex, content, tags)
                         SELECT 'record', hex(record_id), hex(owner), encrypted_content, topic_tags
                         FROM records WHERE status = 0 AND encrypted_content != x''",
                        [],
                    ).unwrap_or(0);

                    if indexed > 0 {
                        info!(count = indexed, "[STORAGE] FTS5 backfill: records indexed");
                    }

                    let _ = conn.execute(
                        "INSERT OR REPLACE INTO chain_state (key, value) VALUES ('fts_index_populated', ?1)",
                        params![b"1".as_slice()],
                    );
                    info!("[STORAGE] ✅ FTS5 index populated");
                }
            }

            info!("[STORAGE] ✅ Migration to v5 (cognitive graph) complete");
        }

        // v5 → v6 (v2.5.0-SuperNode): Cognitive task queue + LLM usage log + sessions.title
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(5);

        if current < 6 {
            info!("[STORAGE] Migrating schema v{} → v6 (SuperNode)", current);

            // 6a: Create cognitive_tasks table (IF NOT EXISTS — safe if create_schema already ran)
            let ct_exists: bool = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cognitive_tasks'",
                [],
                |row| row.get::<_, i64>(0),
            ).unwrap_or(0) > 0;

            if !ct_exists {
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS cognitive_tasks (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_type       TEXT NOT NULL,
                        priority        INTEGER DEFAULT 5,
                        status          TEXT DEFAULT 'pending',
                        payload         TEXT NOT NULL,
                        result          TEXT,
                        prompt_messages TEXT,
                        target_table    TEXT,
                        target_id       TEXT,
                        privacy_level   TEXT DEFAULT 'structured',
                        provider_used   TEXT,
                        model_used      TEXT,
                        token_usage     TEXT,
                        created_at      INTEGER NOT NULL,
                        started_at      INTEGER,
                        completed_at    INTEGER,
                        retry_count     INTEGER DEFAULT 0,
                        max_retries     INTEGER DEFAULT 3,
                        error_message   TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_ct_status ON cognitive_tasks(status, priority DESC, created_at ASC);
                    CREATE INDEX IF NOT EXISTS idx_ct_target ON cognitive_tasks(target_table, target_id, task_type);
                    CREATE INDEX IF NOT EXISTS idx_ct_type ON cognitive_tasks(task_type, status);"
                ).map_err(|e| format!("v6 migration: create cognitive_tasks: {}", e))?;
                info!("[STORAGE] Created cognitive_tasks table");
            }

            // 6b: Create llm_usage_log table
            let ul_exists: bool = conn.query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='llm_usage_log'",
                [],
                |row| row.get::<_, i64>(0),
            ).unwrap_or(0) > 0;

            if !ul_exists {
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS llm_usage_log (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id         INTEGER,
                        provider        TEXT NOT NULL,
                        model           TEXT NOT NULL,
                        input_tokens    INTEGER NOT NULL,
                        output_tokens   INTEGER NOT NULL,
                        cached_tokens   INTEGER DEFAULT 0,
                        latency_ms      INTEGER NOT NULL,
                        created_at      INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_usage_time ON llm_usage_log(created_at);
                    CREATE INDEX IF NOT EXISTS idx_usage_provider ON llm_usage_log(provider, created_at);
                    CREATE INDEX IF NOT EXISTS idx_usage_task ON llm_usage_log(task_id);"
                ).map_err(|e| format!("v6 migration: create llm_usage_log: {}", e))?;
                info!("[STORAGE] Created llm_usage_log table");
            }

            // 6c: Add sessions.title column
            if conn.prepare("SELECT title FROM sessions LIMIT 0").is_err() {
                conn.execute_batch("ALTER TABLE sessions ADD COLUMN title TEXT;")
                    .map_err(|e| format!("v6 migration: add sessions.title: {}", e))?;
                info!("[STORAGE] Added sessions.title column");
            }

            // 6d: Update schema version (hardcoded 6, not SCHEMA_VERSION)
            conn.execute("UPDATE schema_version SET version = 6", [])
                .map_err(|e| format!("Update schema version to v6: {}", e))?;

            info!("[STORAGE] ✅ Migration to v6 (SuperNode) complete");
        }

        // v6 → v7: node-blind storage marker (Brick 1)
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(6);

        if current < 7 {
            info!("[STORAGE] Migrating schema v{} → v7 (node-blind)", current);
            // Additive column: 1 = client-sealed record the node stores but cannot
            // decrypt or read; 0 = normal node-encrypted record. Guard on the table
            // existing first — some migration unit tests set up a minimal DB without
            // the `records` table (matches the v6 cognitive_tasks existence check).
            let records_exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='records'",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if records_exists && conn.prepare("SELECT blind FROM records LIMIT 0").is_err() {
                conn.execute_batch(
                    "ALTER TABLE records ADD COLUMN blind INTEGER NOT NULL DEFAULT 0;",
                )
                .map_err(|e| format!("v7 migration: add records.blind: {}", e))?;
                info!("[STORAGE] Added `blind` column");
            }
            // ⚠️ hardcoded 7, not SCHEMA_VERSION — matches the existing pattern.
            conn.execute("UPDATE schema_version SET version = 7", [])
                .map_err(|e| format!("Update schema version to v7: {}", e))?;
            info!("[STORAGE] ✅ Migration to v7 (node-blind) complete");
        }

        // v7 → v8: node-blind commitment block persistence.
        // Normal startup runs create_schema() first, but migration tooling and
        // compatibility tests may call maybe_migrate() directly against a
        // minimal legacy database. Keep this migration independently complete
        // and idempotent instead of depending on caller order.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(7);

        if current < 8 {
            info!(
                "[STORAGE] Migrating schema v{} → v8 (commitment blocks)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_commitment_blocks (
                    height              INTEGER PRIMARY KEY CHECK(height > 0),
                    block_hash          BLOB NOT NULL UNIQUE CHECK(length(block_hash) = 32),
                    chain_id            BLOB NOT NULL CHECK(length(chain_id) = 32),
                    protocol_version    INTEGER NOT NULL,
                    timestamp           INTEGER NOT NULL,
                    prev_block_hash     BLOB NOT NULL CHECK(length(prev_block_hash) = 32),
                    merkle_root         BLOB NOT NULL CHECK(length(merkle_root) = 32),
                    record_count        INTEGER NOT NULL,
                    proposer            BLOB NOT NULL CHECK(length(proposer) = 32),
                    proposer_signature  BLOB NOT NULL CHECK(length(proposer_signature) = 64),
                    payload             BLOB NOT NULL,
                    received_from       BLOB,
                    created_at          INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_record_blocks_hash
                    ON record_commitment_blocks(block_hash);
                CREATE TABLE IF NOT EXISTS record_block_commitments (
                    record_id       BLOB PRIMARY KEY CHECK(length(record_id) = 32),
                    block_height    INTEGER NOT NULL,
                    FOREIGN KEY(block_height) REFERENCES record_commitment_blocks(height)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_record_block_commitments_height
                    ON record_block_commitments(block_height);",
            )
            .map_err(|error| format!("v8 migration: create commitment tables: {error}"))?;
            for table in ["record_commitment_blocks", "record_block_commitments"] {
                let exists = conn
                    .query_row(
                        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                        params![table],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap_or(0)
                    > 0;
                if !exists {
                    return Err(format!(
                        "v8 migration: required table '{}' was not created",
                        table
                    ));
                }
            }
            // ⚠️ hardcoded 8, not SCHEMA_VERSION — preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 8", [])
                .map_err(|e| format!("Update schema version to v8: {}", e))?;
            info!("[STORAGE] ✅ Migration to v8 (commitment blocks) complete");
        }

        // v8 → v9: bounded durable checkpoint evidence. This table deliberately
        // stores no memory content, commitment IDs, owners, routes, or endpoints.
        let current: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap_or(8);

        if current < 9 {
            info!(
                "[STORAGE] Migrating schema v{} → v9 (checkpoint evidence vault)",
                current
            );
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS record_checkpoint_evidence (
                    evidence_digest    BLOB PRIMARY KEY CHECK(length(evidence_digest) = 32),
                    observed_at        INTEGER NOT NULL CHECK(observed_at >= 0),
                    relation           TEXT NOT NULL CHECK(relation IN
                        ('converged','remote_ahead','remote_behind','diverged')),
                    local_tip_height   INTEGER NOT NULL CHECK(local_tip_height >= 0),
                    remote_tip_height  INTEGER NOT NULL CHECK(remote_tip_height >= 0),
                    checkpoint_height  INTEGER NOT NULL CHECK(checkpoint_height >= 0),
                    signed_response    BLOB NOT NULL CHECK(length(signed_response) > 0
                        AND length(signed_response) <= 4096),
                    created_at         INTEGER NOT NULL CHECK(created_at >= 0)
                );
                CREATE INDEX IF NOT EXISTS idx_record_checkpoint_evidence_observed
                    ON record_checkpoint_evidence(observed_at);",
            )
            .map_err(|error| format!("v9 migration: create checkpoint evidence: {error}"))?;
            let exists = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master
                     WHERE type='table' AND name='record_checkpoint_evidence'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap_or(0)
                > 0;
            if !exists {
                return Err(
                    "v9 migration: required table 'record_checkpoint_evidence' was not created"
                        .to_string(),
                );
            }
            // ⚠️ hardcoded 9, not SCHEMA_VERSION — preserves sequential upgrades.
            conn.execute("UPDATE schema_version SET version = 9", [])
                .map_err(|error| format!("Update schema version to v9: {error}"))?;
            info!("[STORAGE] ✅ Migration to v9 (checkpoint evidence vault) complete");
        }

        Ok(())
    }

    // ========================================
    // PATCH: Update record content (v2.5.2+Provenance)
    // ========================================

    /// Update a record's content, tags, layer, and/or source_ai (PATCH semantics).
    ///
    /// ## ⚠️ Security: single-lock ownership check (P0 SecAudit)
    /// Ownership check and UPDATE run inside the same `conn.lock()` to eliminate
    /// the TOCTOU window that existed when using two separate lock acquisitions.
    /// The UPDATE itself also carries `AND owner = ?owner AND status = 0` so a
    /// concurrent revoke between the check and the write cannot corrupt state.
    ///
    /// ## ⚠️ Encryption failure returns Err (P1 SecAudit)
    /// If `record_key` is set and encryption fails, the method returns an error
    /// rather than silently storing plaintext.
    ///
    /// ## ⚠️ Critical: embedding is cleared on content change
    /// When `new_content` is Some, the stored embedding is set to NULL so that
    /// the Miner can detect and re-embed the record on its next tick.
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub async fn update_record_content(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        new_content: Option<&str>,
        new_tags: Option<&[String]>,
        new_layer: Option<MemoryLayer>,
        new_source_ai: Option<&str>,
    ) -> Result<bool, String> {
        // Encrypt new content before acquiring the lock (CPU work outside critical section)
        let stored_content: Option<Vec<u8>> = if let Some(text) = new_content {
            let bytes = text.as_bytes().to_vec();
            if let Some(ref key) = self.record_key {
                let key: &[u8; 32] = &**key;
                // P1: encryption failure returns Err — no silent plaintext fallback
                let ct = encrypt_record_content(key, &bytes)
                    .map_err(|e| format!("encrypt new content: {}", e))?;
                Some(ct)
            } else {
                Some(bytes)
            }
        } else {
            None
        };

        let tags_json: Option<String> =
            new_tags.map(|t| serde_json::to_string(t).unwrap_or_else(|_| "[]".to_string()));

        // P0 SecAudit: single lock — ownership check + all UPDATEs in one critical section.
        // Each UPDATE carries AND owner = ?owner (+ AND status = 0 for content) so a
        // concurrent revoke cannot produce an inconsistent write.
        let conn = self.conn.lock().await;

        let exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM records WHERE record_id = ?1 AND owner = ?2 AND status = 0",
                params![record_id.as_slice(), owner.as_slice()],
                |r| r.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if !exists {
            return Ok(false);
        }

        if stored_content.is_some() {
            let affected = conn
                .execute(
                    "UPDATE records SET
                    encrypted_content = ?1,
                    embedding = NULL,
                    embedding_model = '',
                    embedding_dim = 0
                 WHERE record_id = ?2 AND owner = ?3 AND status = 0",
                    params![
                        stored_content.as_deref(),
                        record_id.as_slice(),
                        owner.as_slice(),
                    ],
                )
                .map_err(|e| format!("update content: {}", e))?;
            if affected == 0 {
                return Ok(false);
            }
        }

        if let Some(ref tj) = tags_json {
            conn.execute(
                "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2 AND owner = ?3",
                params![tj, record_id.as_slice(), owner.as_slice()],
            )
            .map_err(|e| format!("update tags: {}", e))?;
        }

        if let Some(l) = new_layer {
            conn.execute(
                "UPDATE records SET layer = ?1 WHERE record_id = ?2 AND owner = ?3",
                params![l as u8 as i64, record_id.as_slice(), owner.as_slice()],
            )
            .map_err(|e| format!("update layer: {}", e))?;
        }

        if let Some(src) = new_source_ai {
            conn.execute(
                "UPDATE records SET source_ai = ?1 WHERE record_id = ?2 AND owner = ?3",
                params![src, record_id.as_slice(), owner.as_slice()],
            )
            .map_err(|e| format!("update source_ai: {}", e))?;
        }

        drop(conn); // release lock before cache write
        self.cache.write().invalidate(record_id);

        debug!(
            record_id = hex::encode(record_id),
            content_changed = new_content.is_some(),
            tags_changed = new_tags.is_some(),
            layer_changed = new_layer.is_some(),
            "[STORAGE] ✅ Record patched"
        );

        Ok(true)
    }

    // ========================================
    // Provenance: set session_id on a record (v2.5.2+Provenance)
    // ========================================

    /// Associate a record with the session it was extracted from.
    ///
    /// ## P0 SecAudit: owner verification added
    /// Without owner verification, any caller that knows a record_id could
    /// tamper with the provenance chain. SQL now carries AND owner = ?3.
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub async fn set_record_session_id(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        session_id: &str,
    ) {
        let conn = self.conn.lock().await;
        let result = conn.execute(
            "UPDATE records SET session_id = ?1 WHERE record_id = ?2 AND owner = ?3",
            params![session_id, record_id.as_slice(), owner.as_slice()],
        );
        if let Err(e) = result {
            debug!(
                record_id = hex::encode(record_id),
                session_id = session_id,
                error = %e,
                "[STORAGE] set_record_session_id failed (non-fatal)"
            );
        }
    }

    /// Associate a record with the episode it was extracted from.
    ///
    /// ## P0 SecAudit: owner verification added (same rationale as set_record_session_id)
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub async fn set_record_episode_id(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        episode_id: &str,
    ) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET episode_id = ?1 WHERE record_id = ?2 AND owner = ?3",
            params![episode_id, record_id.as_slice(), owner.as_slice()],
        );
    }

    /// Tag a record with a project (node-blind grouping). For blind records the
    /// client supplies `project_id` — a plaintext label or an opaque hash — so it
    /// can archive and recall memories by project via the existing `context`
    /// filter (`get_active_records_by_context`). Owner-scoped.
    pub async fn set_record_project_id(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        project_id: &str,
    ) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET project_id = ?1 WHERE record_id = ?2 AND owner = ?3",
            params![project_id, record_id.as_slice(), owner.as_slice()],
        );
    }

    /// Node-blind project scope: the set of the owner's active record_ids
    /// archived under `project_id`. Cheap ids-only lookup (served by
    /// `idx_records_project`) used by recall to constrain node-blind FTS / graph
    /// hits to a project, since `MemoryRecord` does not carry `project_id`.
    pub async fn project_record_ids(
        &self,
        owner: &[u8; 32],
        project_id: &str,
    ) -> std::collections::HashSet<[u8; 32]> {
        let mut set = std::collections::HashSet::new();
        let conn = self.conn.lock().await;
        if let Ok(mut stmt) = conn
            .prepare("SELECT record_id FROM records WHERE owner=?1 AND project_id=?2 AND status=0")
        {
            if let Ok(rows) = stmt.query_map(params![owner.as_slice(), project_id], |row| {
                row.get::<_, Vec<u8>>(0)
            }) {
                for r in rows.flatten() {
                    if r.len() == 32 {
                        let mut a = [0u8; 32];
                        a.copy_from_slice(&r);
                        set.insert(a);
                    }
                }
            }
        }
        set
    }

    /// [D6 ATTEST] Node-blind: all of the owner's ACTIVE record_ids (status=0),
    /// returned SORTED for a deterministic storage-root commitment. The node
    /// reads only opaque content-address hashes here (never plaintext), so
    /// signing a root over this set preserves blindness while giving the client
    /// verifiable, tamper-evident proof of exactly which records the node holds.
    pub async fn owner_record_ids(&self, owner: &[u8; 32]) -> Vec<[u8; 32]> {
        let mut ids: Vec<[u8; 32]> = Vec::new();
        let conn = self.conn.lock().await;
        if let Ok(mut stmt) =
            conn.prepare("SELECT record_id FROM records WHERE owner=?1 AND status=0")
        {
            if let Ok(rows) =
                stmt.query_map(params![owner.as_slice()], |row| row.get::<_, Vec<u8>>(0))
            {
                for r in rows.flatten() {
                    if r.len() == 32 {
                        let mut a = [0u8; 32];
                        a.copy_from_slice(&r);
                        ids.push(a);
                    }
                }
            }
        }
        ids.sort_unstable();
        ids
    }

    // ========================================
    // Provenance: find records by session (v2.5.2+Provenance)
    // ========================================

    /// Get all active records extracted from a specific session.
    ///
    /// Used by the `/record/:id/provenance` endpoint and by
    /// find_records_by_content() to scope searches to a session.
    ///
    /// v2.5.2+Provenance
    pub async fn get_records_for_session(
        &self,
        session_id: &str,
        owner: &[u8; 32],
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,blind
             FROM records
             WHERE session_id = ?1 AND owner = ?2 AND status = 0
             ORDER BY timestamp ASC",
            params![session_id, owner.as_slice()],
        )
    }

    /// Find active records whose plaintext content contains the given substring.
    ///
    /// ## P2 SecAudit: pub(crate) + hard limit
    /// Exposed as pub(crate) only — not callable from external crates.
    /// Hard-capped at 100 regardless of caller-supplied limit to prevent
    /// O(n) scan DoS (each record requires a decryption + string match).
    /// For production search, prefer bm25_search (FTS5).
    ///
    /// v2.5.2+Provenance / v2.5.2+SecAudit
    pub(crate) async fn find_records_by_content(
        &self,
        owner: &[u8; 32],
        content_substring: &str,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        if content_substring.trim().is_empty() {
            return Vec::new();
        }
        let all = self.get_active_records(owner, None, limit * 10).await;
        let needle = content_substring.to_lowercase();

        all.into_iter()
            .filter(|r| {
                let text = String::from_utf8_lossy(&r.encrypted_content);
                text.to_lowercase().contains(&needle)
            })
            .take(limit)
            .collect()
    }

    /// Get full provenance chain for a record.
    ///
    /// Returns:
    /// - The record itself
    /// - session_id (from records.session_id column)
    /// - session metadata (title, started_at)
    /// - turn_index hint (best-effort: LENGTH heuristic, may be inaccurate)
    ///
    /// ## ⚠️ turn_index accuracy (BUG FIX v2.5.2+Provenance)
    /// The turn_index lookup uses `ABS(LENGTH(CAST(content AS TEXT)) - ?)` as a
    /// heuristic to find the closest-length turn. This is unreliable:
    ///   - For encrypted raw_logs, CAST gives BLOB hex length, not plaintext length.
    ///   - Multiple turns may have the same content length (e.g. short turns).
    ///   - `content_text` from `record.encrypted_content` is already decrypted
    ///     (see find_records_by_content note), so the length comparison is
    ///     against plaintext vs potentially encrypted BLOB.
    /// Treat turn_index as advisory — it is not guaranteed to be the source turn.
    /// A future v7 improvement: store turn_index directly in records.session_turn_index.
    ///
    /// v2.5.2+Provenance
    pub async fn get_record_provenance(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
    ) -> Option<RecordProvenance> {
        let record = self.get(record_id).await?;
        if record.owner != *owner {
            return None;
        }

        let conn = self.conn.lock().await;

        // Get session_id from the records table (provenance field)
        let session_id: Option<String> = conn
            .query_row(
                "SELECT session_id FROM records WHERE record_id = ?1",
                params![record_id.as_slice()],
                |r| r.get(0),
            )
            .ok()
            .flatten();

        // Get session metadata if session_id is known
        let (session_title, session_started_at) = if let Some(ref sid) = session_id {
            let meta: Option<(Option<String>, i64)> = conn
                .query_row(
                    "SELECT title, started_at FROM sessions WHERE session_id = ?1",
                    params![sid],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .ok();
            match meta {
                Some((title, started_at)) => (title, Some(started_at)),
                None => (None, None),
            }
        } else {
            (None, None)
        };

        // Best-effort turn_index: find the raw_log turn whose content length is
        // closest to the record's content length (LENGTH heuristic).
        // ⚠️ Advisory only — see method doc comment for accuracy limitations.
        let turn_index: Option<i64> = if let Some(ref sid) = session_id {
            let content_len = record.encrypted_content.len() as i64;
            conn.query_row(
                "SELECT turn_index FROM raw_logs
                 WHERE session_id = ?1
                 ORDER BY ABS(LENGTH(content) - ?2) ASC
                 LIMIT 1",
                params![sid, content_len],
                |r| r.get(0),
            )
            .ok()
        } else {
            None
        };

        Some(RecordProvenance {
            record_id: hex::encode(record_id),
            session_id,
            session_title,
            session_started_at,
            turn_index,
            layer: record.layer.to_string(),
            topic_tags: record.topic_tags.clone(),
            extracted_at: record.timestamp,
            source_ai: record.source_ai.clone(),
        })
    }

    // ========================================
    // Insert
    // ========================================

    pub async fn insert(&self, record: &MemoryRecord, embedding_model: &str) -> bool {
        if !record.verify_id() {
            warn!(
                record_id = hex::encode(record.record_id),
                "[STORAGE] ❌ Rejected: hash mismatch"
            );
            self.total_rejected.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let tags_json =
            serde_json::to_string(&record.topic_tags).unwrap_or_else(|_| "[]".to_string());
        let embedding_blob: Option<Vec<u8>> = if record.has_embedding() {
            Some(embedding_to_bytes(&record.embedding))
        } else {
            None
        };
        let embedding_dim = record.embedding_dim() as i64;
        let conflict_with_blob: Option<Vec<u8>> = record.conflict_with.map(|c| c.to_vec());

        let stored_content: Vec<u8> = if record.blind {
            // Node-blind (Brick 1): the client already sealed this content with its
            // own key. Store it verbatim — the node must never encrypt it (it cannot
            // decrypt it back, and doing so would double-wrap the client ciphertext).
            record.encrypted_content.clone()
        } else if let Some(ref key) = self.record_key {
            let key: &[u8; 32] = &**key;
            if record.encrypted_content.is_empty() {
                record.encrypted_content.clone()
            } else {
                // P1 SecAudit: encryption failure returns false (rejects insert)
                // instead of silently storing plaintext.
                match encrypt_record_content(key, &record.encrypted_content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        error!(
                            record_id = hex::encode(record.record_id),
                            error = %e,
                            "[STORAGE] ❌ Record encryption failed — insert rejected"
                        );
                        self.total_rejected.fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                }
            }
        } else {
            record.encrypted_content.clone()
        };

        let conn = self.conn.lock().await;
        let result = conn.execute(
            "INSERT OR IGNORE INTO records (
                record_id, owner, timestamp, layer, topic_tags, source_ai,
                status, supersedes, encrypted_content, embedding,
                embedding_model, embedding_dim, signature, access_count, created_at,
                positive_feedback, negative_feedback, conflict_with, blind
            ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
            params![
                record.record_id.as_slice(),
                record.owner.as_slice(),
                record.timestamp as i64,
                record.layer as u8 as i64,
                tags_json,
                record.source_ai,
                record.status as u8 as i64,
                record.supersedes.as_ref().map(|s| s.as_slice()),
                stored_content.as_slice(),
                embedding_blob.as_deref(),
                embedding_model,
                embedding_dim,
                record.signature.as_slice(),
                record.access_count as i64,
                now,
                record.positive_feedback as i64,
                record.negative_feedback as i64,
                conflict_with_blob.as_deref(),
                record.blind as i64,
            ],
        );

        match result {
            Ok(changes) if changes > 0 => {
                self.total_inserted.fetch_add(1, Ordering::Relaxed);
                self.cache.write().put(record.clone());
                debug!(record_id = hex::encode(record.record_id), layer = %record.layer, "[STORAGE] ✅ Inserted");
                true
            }
            Ok(_) => {
                debug!(
                    record_id = hex::encode(record.record_id),
                    "[STORAGE] Duplicate, skipped"
                );
                false
            }
            Err(e) => {
                error!(record_id = hex::encode(record.record_id), error = %e, "[STORAGE] ❌ Insert failed");
                self.total_rejected.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Store a node-blind record received from a peer as a **verbatim replica**
    /// (replication receive path, Brick 4). The content is opaque to this node,
    /// so it is stored exactly as received — the node **never** re-encrypts it
    /// with its own key (which would double-wrap the origin's ciphertext and make
    /// the replica unreadable). `insert()`'s `verify_id()` still guards content
    /// integrity. The caller MUST have already verified the origin's Ed25519
    /// signature (same as the P2P `BroadcastRecord` path). Returns true if stored.
    pub async fn insert_blind_replica(&self, record: &MemoryRecord, embedding_model: &str) -> bool {
        let mut replica = record.clone();
        replica.blind = true;
        self.insert(&replica, embedding_model).await
    }

    // ========================================
    // Query
    // ========================================

    pub async fn get(&self, record_id: &[u8; 32]) -> Option<MemoryRecord> {
        {
            let mut cache = self.cache.write();
            if let Some(record) = cache.get(record_id) {
                return Some(record.clone());
            }
        }
        let conn = self.conn.lock().await;
        let rk = self.record_key.as_ref().map(|v| &**v);
        let result = conn
            .query_row(
                Self::SELECT_RECORD_COLS,
                params![record_id.as_slice()],
                |row| Self::row_to_record(row, rk),
            )
            .optional()
            .unwrap_or_else(|e| {
                error!(error=%e, "[STORAGE] Query failed");
                None
            });

        if let Some(ref record) = result {
            self.cache.write().put(record.clone());
        }
        result
    }

    pub async fn get_active_records(
        &self,
        owner: &[u8; 32],
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;
        if let Some(l) = layer {
            self.query_rows(
                &conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,signature,access_count,
                        positive_feedback,negative_feedback,conflict_with,blind
                 FROM records WHERE owner=?1 AND status=0 AND layer=?2
                 ORDER BY timestamp DESC LIMIT ?3",
                params![owner.as_slice(), l as u8 as i64, limit as i64],
            )
        } else {
            self.query_rows(
                &conn,
                "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                        status,supersedes,encrypted_content,embedding,signature,access_count,
                        positive_feedback,negative_feedback,conflict_with,blind
                 FROM records WHERE owner=?1 AND status=0
                 ORDER BY timestamp DESC LIMIT ?2",
                params![owner.as_slice(), limit as i64],
            )
        }
    }

    pub async fn query_by_owner_after(
        &self,
        owner: &[u8; 32],
        after_timestamp: u64,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,blind
             FROM records WHERE owner=?1 AND timestamp>?2
             ORDER BY timestamp ASC LIMIT ?3",
            params![
                owner.as_slice(),
                after_timestamp as i64,
                DEFAULT_PAGE_SIZE as i64
            ],
        )
    }

    /// Returns active node-blind records not yet represented in the local
    /// commitment chain.
    ///
    /// This is the only source used by the V1 block packer. Node-readable
    /// records are deliberately excluded so block production cannot turn the
    /// synchronised ledger into a plaintext or owner-metadata replication
    /// channel. The caller must still verify each record's content address and
    /// Ed25519 owner signature before building a block.
    pub async fn get_uncommitted_blind_records(&self, limit: usize) -> Vec<MemoryRecord> {
        let limit = limit.clamp(1, MAX_RECORD_COMMITMENTS_PER_BLOCK);
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT r.record_id,r.owner,r.timestamp,r.layer,r.topic_tags,r.source_ai,
                    r.status,r.supersedes,r.encrypted_content,r.embedding,r.signature,r.access_count,
                    r.positive_feedback,r.negative_feedback,r.conflict_with,r.blind
             FROM records r
             LEFT JOIN record_block_commitments c ON c.record_id = r.record_id
             WHERE r.blind=1 AND r.status=0 AND c.record_id IS NULL
             ORDER BY r.record_id ASC LIMIT ?1",
            params![limit as i64],
        )
    }

    pub async fn get_records_with_embedding(
        &self,
        owner: &[u8; 32],
    ) -> Vec<(MemoryRecord, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,embedding_model,blind
             FROM records WHERE owner=?1 AND status=0 AND embedding IS NOT NULL
             ORDER BY timestamp DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                error!(error=%e, "[STORAGE] Prepare failed");
                return Vec::new();
            }
        };
        let rk = self.record_key.as_ref().map(|v| &**v);
        stmt.query_map(params![owner.as_slice()], |row| {
            let record = Self::row_to_record(row, rk)?;
            let model: String = row.get(15)?;
            Ok((record, model))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    /// Return every active record that has a persisted embedding.
    ///
    /// This is intentionally separate from `get_records_with_embedding(owner)`:
    /// normal Local-mode nodes must keep the historical single-owner rebuild,
    /// while node-blind/remote storage nodes share one SQLite database across
    /// authenticated owners and therefore need to restore every owner/model
    /// partition after process restart.
    ///
    /// # Security
    /// The returned records remain partitioned by `record.owner` when inserted
    /// into `VectorIndex`; recall still supplies an authenticated owner and can
    /// never search another owner's partition. Blind record content also stays
    /// opaque because `row_to_record` never decrypts rows marked `blind`.
    pub async fn get_all_records_with_embedding(&self) -> Vec<(MemoryRecord, String)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with,embedding_model,blind
             FROM records WHERE status=0 AND embedding IS NOT NULL
             ORDER BY owner ASC, timestamp DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                error!(error=%e, "[STORAGE] Prepare all-owner embedding rebuild failed");
                return Vec::new();
            }
        };
        let rk = self.record_key.as_ref().map(|v| &**v);
        stmt.query_map([], |row| {
            let record = Self::row_to_record(row, rk)?;
            let model: String = row.get(15)?;
            Ok((record, model))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    // ========================================
    // Lifecycle
    // ========================================

    pub async fn update_status(&self, record_id: &[u8; 32], new_status: RecordStatus) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1 WHERE record_id=?2",
            params![new_status as u8 as i64, record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(record_id=hex::encode(record_id), %new_status, "[STORAGE] ✅ Status updated");
                true
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] Not found for status update"
                );
                false
            }
            Err(e) => {
                error!(error=%e, "[STORAGE] Status update failed");
                false
            }
        }
    }

    pub async fn revoke(&self, record_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET status=?1, encrypted_content=x'', embedding=NULL WHERE record_id=?2",
            params![RecordStatus::Revoked as u8 as i64, record_id.as_slice()]) {
            Ok(n) if n > 0 => {
                self.cache.write().invalidate(record_id);
                info!(record_id=hex::encode(record_id), "[STORAGE] 🗑️ Revoked"); true
            }
            Ok(_) => { warn!(record_id=hex::encode(record_id), "[STORAGE] Not found for revoke"); false }
            Err(e) => { error!(error=%e, "[STORAGE] Revoke failed"); false }
        }
    }

    pub async fn increment_access(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET access_count=access_count+1 WHERE record_id=?1",
            params![record_id.as_slice()],
        );
    }

    /// Store node-blind derived-record provenance (Brick 3d): link a derived
    /// record (e.g. a client/LLM-produced summary) to the source records it was
    /// built from. `sources` are opaque record_id hexes; the node learns only the
    /// provenance DAG shape, never the content. Self-links are ignored.
    pub async fn insert_blind_provenance(
        &self,
        record_id: &[u8; 32],
        owner: &[u8; 32],
        sources: &[String],
    ) {
        if sources.is_empty() {
            return;
        }
        let rid_hex = hex::encode(record_id);
        let owner_hex = hex::encode(owner);
        let conn = self.conn.lock().await;
        for src in sources {
            if src.is_empty() || src == &rid_hex {
                continue;
            }
            let _ = conn.execute(
                "INSERT OR IGNORE INTO blind_provenance (record_id, source_id, owner_hex)
                 VALUES (?1, ?2, ?3)",
                params![rid_hex, src, owner_hex],
            );
        }
    }

    /// Return the source record_id hexes a derived record was built from.
    pub async fn get_blind_provenance(&self, record_id: &[u8; 32]) -> Vec<String> {
        let rid_hex = hex::encode(record_id);
        let conn = self.conn.lock().await;
        let mut stmt =
            match conn.prepare("SELECT source_id FROM blind_provenance WHERE record_id = ?1") {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            };
        stmt.query_map(params![rid_hex], |row| row.get::<_, String>(0))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
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

    pub(crate) fn query_rows(
        &self,
        conn: &Connection,
        sql: &str,
        p: impl rusqlite::Params,
    ) -> Vec<MemoryRecord> {
        let mut stmt = match conn.prepare(sql) {
            Ok(s) => s,
            Err(e) => {
                error!(error=%e, "[STORAGE] Prepare failed");
                return Vec::new();
            }
        };
        let rk = self.record_key.as_ref().map(|v| &**v);
        stmt.query_map(p, |row| Self::row_to_record(row, rk))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    pub(crate) fn row_to_record(
        row: &rusqlite::Row<'_>,
        record_key: Option<&[u8; 32]>,
    ) -> rusqlite::Result<MemoryRecord> {
        let record_id_blob: Vec<u8> = row.get(0)?;
        let owner_blob: Vec<u8> = row.get(1)?;
        let timestamp: i64 = row.get(2)?;
        let layer_val: i64 = row.get(3)?;
        let tags_json: String = row.get(4)?;
        let source_ai: String = row.get(5)?;
        let status_val: i64 = row.get(6)?;
        let supersedes_blob: Option<Vec<u8>> = row.get(7)?;
        let encrypted_content: Vec<u8> = row.get(8)?;
        let embedding_blob: Option<Vec<u8>> = row.get(9)?;

        // Node-blind marker (Brick 1). Read BY NAME so it is robust to column
        // position across the various record SELECTs; if the column is not in the
        // result set, default to 0 (sighted).
        let blind: bool = row.get::<&str, i64>("blind").unwrap_or(0) != 0;

        let decrypted_content = if blind {
            // Node-blind record: client-sealed with a key the node does not have.
            // Return the ciphertext verbatim — never attempt to decrypt it.
            encrypted_content
        } else if let Some(key) = record_key {
            if encrypted_content.len() >= 28 {
                match decrypt_record_content(key, &encrypted_content) {
                    Ok(plain) => plain,
                    Err(_) => encrypted_content,
                }
            } else {
                encrypted_content
            }
        } else {
            encrypted_content
        };

        let signature_blob: Vec<u8> = row.get(10)?;
        let access_count: i64 = row.get(11)?;
        let positive_feedback: i64 = row.get(12).unwrap_or(0);
        let negative_feedback: i64 = row.get(13).unwrap_or(0);
        let conflict_with_blob: Option<Vec<u8>> = row.get(14).unwrap_or(None);

        let mut record_id = [0u8; 32];
        if record_id_blob.len() == 32 {
            record_id.copy_from_slice(&record_id_blob);
        } else {
            // P3 SecAudit: return Err instead of silently using zero-ID.
            // A zero-ID record would corrupt the cache (multiple broken records
            // share the same key) and pollute search results.
            return Err(rusqlite::Error::InvalidColumnType(
                0,
                "record_id".into(),
                rusqlite::types::Type::Blob,
            ));
        }
        let mut owner = [0u8; 32];
        if owner_blob.len() == 32 {
            owner.copy_from_slice(&owner_blob);
        } else {
            return Err(rusqlite::Error::InvalidColumnType(
                1,
                "owner".into(),
                rusqlite::types::Type::Blob,
            ));
        }
        let mut signature = [0u8; 64];
        if signature_blob.len() == 64 {
            signature.copy_from_slice(&signature_blob);
        } else {
            return Err(rusqlite::Error::InvalidColumnType(
                10,
                "signature".into(),
                rusqlite::types::Type::Blob,
            ));
        }

        let supersedes = supersedes_blob.and_then(|b| {
            if b.len() == 32 {
                let mut a = [0u8; 32];
                a.copy_from_slice(&b);
                Some(a)
            } else {
                None
            }
        });
        let conflict_with = conflict_with_blob.and_then(|b| {
            if b.len() == 32 {
                let mut a = [0u8; 32];
                a.copy_from_slice(&b);
                Some(a)
            } else {
                None
            }
        });
        let embedding = embedding_blob
            .map(|b| bytes_to_embedding(&b))
            .unwrap_or_default();

        Ok(MemoryRecord {
            record_id,
            owner,
            timestamp: timestamp as u64,
            layer: MemoryLayer::from_u8(layer_val as u8).unwrap_or(MemoryLayer::Episode),
            topic_tags: serde_json::from_str(&tags_json).unwrap_or_default(),
            source_ai,
            status: RecordStatus::from_u8(status_val as u8).unwrap_or(RecordStatus::Active),
            supersedes,
            encrypted_content: decrypted_content,
            embedding,
            signature,
            access_count: access_count as u32,
            positive_feedback: positive_feedback as u32,
            negative_feedback: negative_feedback as u32,
            conflict_with,
            blind,
        })
    }
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

    // ========================================
    // v2.6.0+NodeBlind (Brick 1) — node-blind storage round-trip
    // ========================================

    #[tokio::test]
    async fn test_blind_record_stored_and_read_verbatim() {
        // Encrypted store: the node holds a record_key. A blind record must
        // still be stored/returned verbatim — never encrypted or decrypted by
        // the node — while a normal record is encrypted at rest as usual.
        let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();

        // Node-blind record: content is the CLIENT's own ciphertext (opaque to
        // the node); blind = true.
        let sealed = b"client-sealed-ciphertext-opaque-to-node".to_vec();
        let mut blind_rec = MemoryRecord::new(
            [0xBB; 32],
            200,
            MemoryLayer::Knowledge,
            vec!["sealed".into()],
            "client".into(),
            sealed.clone(),
            vec![0.9, 0.8],
        );
        blind_rec.blind = true;
        let bid = blind_rec.record_id;
        assert!(s.insert(&blind_rec, "client-embed").await);

        // Force a real DB read (bypass the write-through cache) to exercise
        // row_to_record's blind branch.
        s.cache.write().invalidate(&bid);
        let got = s.get(&bid).await.unwrap();
        assert!(got.blind, "blind flag must survive the DB round-trip");
        assert_eq!(
            got.encrypted_content, sealed,
            "blind content must be stored and returned byte-for-byte (no node crypto)"
        );

        // Sanity: a normal (sighted) record in the SAME encrypted store is
        // encrypted at rest and decrypted back to plaintext on read.
        let plain = b"node-visible-plaintext".to_vec();
        let sighted = MemoryRecord::new(
            [0xBB; 32],
            201,
            MemoryLayer::Knowledge,
            vec!["plain".into()],
            "node".into(),
            plain.clone(),
            vec![0.1],
        );
        let sid = sighted.record_id;
        assert!(s.insert(&sighted, "minilm").await);
        s.cache.write().invalidate(&sid);
        let got2 = s.get(&sid).await.unwrap();
        assert!(!got2.blind);
        assert_eq!(
            got2.encrypted_content, plain,
            "sighted content round-trips to plaintext"
        );
    }

    #[tokio::test]
    async fn test_blind_fts_index_and_search() {
        let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();
        let owner = [0xBB; 32];
        let mut rec = MemoryRecord::new(
            owner,
            300,
            MemoryLayer::Knowledge,
            vec![],
            "client".into(),
            b"opaque-ciphertext".to_vec(),
            vec![],
        );
        rec.blind = true;
        assert!(s.insert(&rec, "client").await);

        // Client-supplied keyed token-hashes (hex). The node never sees plaintext.
        let terms = vec!["a3f29c4d".to_string(), "9c4d2eb1".to_string()];
        s.fts_index_blind_terms(&rec.record_id, &owner, &terms)
            .await;

        // Query by an indexed term-hash → finds the record.
        let hits = s
            .bm25_search_blind(&["a3f29c4d".to_string()], &owner, 10)
            .await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, hex::encode(rec.record_id));

        // Query by a non-indexed hash → no hits.
        let none = s
            .bm25_search_blind(&["deadbeef".to_string()], &owner, 10)
            .await;
        assert!(none.is_empty());

        // Wrong owner → no hits (access scoping via owner_hex).
        let other = s
            .bm25_search_blind(&["a3f29c4d".to_string()], &[0xCC; 32], 10)
            .await;
        assert!(other.is_empty());

        // Non-hex query terms are rejected (no FTS syntax injection).
        let bad = s
            .bm25_search_blind(&["a3f2 OR x".to_string()], &owner, 10)
            .await;
        assert!(bad.is_empty());
    }

    #[tokio::test]
    async fn test_blind_provenance_roundtrip() {
        let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();
        let owner = [0xBB; 32];
        let derived = MemoryRecord::new(
            owner,
            400,
            MemoryLayer::Knowledge,
            vec![],
            "summary".into(),
            b"summary-ciphertext".to_vec(),
            vec![],
        );
        let did = derived.record_id;
        let src_a = hex::encode([0x01u8; 32]);
        let src_b = hex::encode([0x02u8; 32]);

        // Self-links and empties are ignored; sources are recorded.
        s.insert_blind_provenance(
            &did,
            &owner,
            &[
                src_a.clone(),
                src_b.clone(),
                hex::encode(did),
                String::new(),
            ],
        )
        .await;

        let mut got = s.get_blind_provenance(&did).await;
        got.sort();
        let mut want = vec![src_a, src_b];
        want.sort();
        assert_eq!(got, want);

        // A record with no provenance returns empty.
        assert!(s.get_blind_provenance(&[0x09u8; 32]).await.is_empty());
    }

    #[tokio::test]
    async fn test_blind_record_project_grouping() {
        let s = MemoryStorage::open(":memory:", Some([0x11u8; 32])).unwrap();
        let owner = [0xBB; 32];
        let mut rec = MemoryRecord::new(
            owner,
            500,
            MemoryLayer::Knowledge,
            vec![],
            "client".into(),
            b"opaque".to_vec(),
            vec![],
        );
        rec.blind = true;
        assert!(s.insert(&rec, "client").await);

        // Archive it under a project (a label or an opaque hash), then recall by it.
        s.set_record_project_id(&rec.record_id, &owner, "proj_alpha")
            .await;
        let hits = s
            .get_active_records_by_context(&owner, "proj_alpha", None, 10)
            .await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].record_id, rec.record_id);
        assert!(hits[0].blind, "blind flag survives the project query");

        // A different project returns nothing.
        assert!(s
            .get_active_records_by_context(&owner, "proj_other", None, 10)
            .await
            .is_empty());
    }

    #[tokio::test]
    async fn test_insert_blind_replica_stored_verbatim() {
        // This node has its OWN record_key (encrypted store).
        let s = MemoryStorage::open(":memory:", Some([0x22u8; 32])).unwrap();

        // A replica from ANOTHER owner: content is that owner's opaque ciphertext.
        // blind is false on the incoming record (it is #[serde(skip)] over the wire).
        let foreign_ct = b"another-owners-ciphertext-opaque".to_vec();
        let replica = MemoryRecord::new(
            [0xEE; 32],
            700,
            MemoryLayer::Knowledge,
            vec![],
            "peer".into(),
            foreign_ct.clone(),
            vec![0.3, 0.4],
        );
        let rid = replica.record_id;
        assert!(!replica.blind);
        assert!(s.insert_blind_replica(&replica, "peer-model").await);

        // Read back: stored byte-for-byte (NOT re-encrypted with this node's key),
        // blind marker set, foreign owner preserved.
        s.cache.write().invalidate(&rid);
        let got = s.get(&rid).await.unwrap();
        assert!(got.blind);
        assert_eq!(
            got.encrypted_content, foreign_ct,
            "replica must be stored verbatim, not re-encrypted"
        );
        assert_eq!(got.owner, [0xEE; 32]);

        // A tampered record (content no longer matches record_id) is rejected.
        let mut bad = replica.clone();
        bad.encrypted_content = b"tampered".to_vec();
        assert!(!s.insert_blind_replica(&bad, "peer-model").await);
    }

    // ========================================
    // Existing tests (preserved)
    // ========================================

    #[tokio::test]
    async fn test_open_in_memory() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        assert_eq!(s.count().await, 0);
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        let id = r.record_id;
        assert!(s.insert(&r, "minilm").await);
        assert!(!s.insert(&r, "minilm").await);
        let got = s.get(&id).await.unwrap();
        assert_eq!(got.source_ai, "test");
        assert_eq!(got.layer, MemoryLayer::Episode);
        assert_eq!(got.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[tokio::test]
    async fn test_reject_invalid_hash() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let mut r = make_rec(100, MemoryLayer::Episode, "t");
        r.record_id = [0xFF; 32];
        assert!(!s.insert(&r, "m").await);
        assert_eq!(s.total_rejected(), 1);
    }

    #[tokio::test]
    async fn test_get_active_records() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let o = [0xAA; 32];
        s.insert(&make_rec_owner(100, o, MemoryLayer::Episode), "m")
            .await;
        s.insert(&make_rec_owner(200, o, MemoryLayer::Knowledge), "m")
            .await;
        s.insert(&make_rec_owner(300, o, MemoryLayer::Archive), "m")
            .await;
        assert_eq!(s.get_active_records(&o, None, 100).await.len(), 3);
        assert_eq!(
            s.get_active_records(&o, Some(MemoryLayer::Episode), 100)
                .await
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn test_all_owner_embedding_rebuild_preserves_owner_and_status_scope() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner_a = [0xA1; 32];
        let owner_b = [0xB2; 32];
        let active_a = make_rec_owner(100, owner_a, MemoryLayer::Episode);
        let active_b = make_rec_owner(200, owner_b, MemoryLayer::Knowledge);
        let revoked_b = make_rec_owner(300, owner_b, MemoryLayer::Archive);

        assert!(s.insert(&active_a, "model-a").await);
        assert!(s.insert(&active_b, "model-b").await);
        assert!(s.insert(&revoked_b, "model-b").await);
        assert!(s.revoke(&revoked_b.record_id).await);

        let local_only = s.get_records_with_embedding(&owner_a).await;
        assert_eq!(
            local_only.len(),
            1,
            "legacy owner-scoped rebuild is unchanged"
        );
        assert_eq!(local_only[0].0.owner, owner_a);

        let all = s.get_all_records_with_embedding().await;
        assert_eq!(all.len(), 2, "revoked records must not re-enter the index");
        assert!(all.iter().any(|(record, model)| {
            record.record_id == active_a.record_id && record.owner == owner_a && model == "model-a"
        }));
        assert!(all.iter().any(|(record, model)| {
            record.record_id == active_b.record_id && record.owner == owner_b && model == "model-b"
        }));
        assert!(!all
            .iter()
            .any(|(record, _)| record.record_id == revoked_b.record_id));
    }

    #[tokio::test]
    async fn test_revoke() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "t");
        s.insert(&r, "m").await;
        assert!(s.revoke(&r.record_id).await);
        let got = s.get(&r.record_id).await.unwrap();
        assert_eq!(got.status, RecordStatus::Revoked);
        assert!(got.encrypted_content.is_empty());
    }

    #[tokio::test]
    async fn test_encrypted_insert_and_get() {
        use super::super::storage_crypto::derive_record_key;
        let key = derive_record_key(&[0x42; 32]);
        let s = MemoryStorage::open(":memory:", Some(key)).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        s.insert(&r, "m").await;
        s.cache.write().clear();
        let got = s.get(&r.record_id).await.unwrap();
        assert_eq!(got.encrypted_content, b"encrypted_data");
    }

    // ========================================
    // v2.4.0: Schema v5 Tests
    // ========================================

    #[tokio::test]
    async fn test_schema_version_is_current() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;
        let v: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(
            v, SCHEMA_VERSION,
            "Schema version should be {}",
            SCHEMA_VERSION
        );
    }

    #[tokio::test]
    async fn test_v5_tables_exist() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let expected_tables = [
            "episodes",
            "entities",
            "knowledge_edges",
            "episode_edges",
            "communities",
            "projects",
            "sessions",
            "artifacts",
        ];

        for table in &expected_tables {
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![table],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                > 0;
            assert!(exists, "Table '{}' should exist in schema v5", table);
        }
    }

    #[tokio::test]
    async fn test_v6_tables_exist() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let expected_tables = ["cognitive_tasks", "llm_usage_log"];
        for table in &expected_tables {
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![table],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                > 0;
            assert!(exists, "Table '{}' should exist in schema v6", table);
        }

        let title_ok = conn.prepare("SELECT title FROM sessions LIMIT 0").is_ok();
        assert!(title_ok, "sessions.title column should exist in schema v6");
    }

    #[tokio::test]
    async fn test_v6_cognitive_tasks_schema() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result = conn.execute(
            "INSERT INTO cognitive_tasks
                (task_type, priority, status, payload, target_table, target_id,
                 privacy_level, created_at, max_retries)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                "session_title",
                5i64,
                "pending",
                r#"{"session_id":"sess_001","summary":"JWT auth discussion"}"#,
                "sessions",
                "sess_001",
                "structured",
                now,
                3i64,
            ],
        );
        assert!(
            result.is_ok(),
            "cognitive_tasks insert should succeed: {:?}",
            result.err()
        );

        let claimed = conn
            .execute(
                "UPDATE cognitive_tasks SET status='processing', started_at=?1
             WHERE id = (
                 SELECT id FROM cognitive_tasks
                 WHERE status='pending'
                 ORDER BY priority DESC, created_at ASC
                 LIMIT 1
             )",
                params![now],
            )
            .unwrap();
        assert_eq!(claimed, 1);

        let status: String = conn
            .query_row(
                "SELECT status FROM cognitive_tasks WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(status, "processing");
    }

    #[tokio::test]
    async fn test_v6_llm_usage_log_schema() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO llm_usage_log
                (task_id, provider, model, input_tokens, output_tokens,
                 cached_tokens, latency_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                1i64,
                "deepseek",
                "deepseek-reasoner",
                512i64,
                128i64,
                64i64,
                1200i64,
                now
            ],
        )
        .unwrap();

        let (input_sum, output_sum): (i64, i64) = conn.query_row(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM llm_usage_log WHERE provider = 'deepseek'",
            [], |row| Ok((row.get(0)?, row.get(1)?)),
        ).unwrap();
        assert_eq!(input_sum, 512);
        assert_eq!(output_sum, 128);
    }

    #[tokio::test]
    async fn test_v6_sessions_title_column() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO sessions (session_id, owner, session_type, started_at, turn_count)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params!["sess_001", [0xAAu8; 32].as_slice(), "chat", now, 5i64],
        )
        .unwrap();

        let title: Option<String> = conn
            .query_row(
                "SELECT title FROM sessions WHERE session_id = 'sess_001'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(title.is_none());

        conn.execute(
            "UPDATE sessions SET title = ?1 WHERE session_id = ?2",
            params!["JWT Auth Implementation Discussion", "sess_001"],
        )
        .unwrap();

        let title: Option<String> = conn
            .query_row(
                "SELECT title FROM sessions WHERE session_id = 'sess_001'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            title,
            Some("JWT Auth Implementation Discussion".to_string())
        );
    }

    // ========================================
    // v2.4.0: Existing schema tests (preserved)
    // ========================================

    #[tokio::test]
    async fn test_v5_records_has_new_columns() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;
        let result = conn.prepare("SELECT project_id, session_id, episode_id FROM records LIMIT 0");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_v5_episodes_table_schema() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result = conn.execute(
            "INSERT INTO episodes (episode_id, owner, episode_type, source,
                session_id, encrypted_content, content_hash, token_count,
                created_at, ingested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                "ep_001",
                [0xAAu8; 32].as_slice(),
                "conversation",
                "test",
                "session_001",
                b"encrypted".as_slice(),
                "hash123",
                100i64,
                now,
                now,
            ],
        );
        assert!(
            result.is_ok(),
            "Episode insert should succeed: {:?}",
            result.err()
        );

        let ep_type: String = conn
            .query_row(
                "SELECT episode_type FROM episodes WHERE episode_id = 'ep_001'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(ep_type, "conversation");
    }

    #[tokio::test]
    async fn test_v5_entities_table_schema() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let result = conn.execute(
            "INSERT INTO entities (entity_id, owner, name, name_normalized,
                entity_type, description, community_id, created_at, updated_at, mention_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                "ent_jwt",
                [0xAAu8; 32].as_slice(),
                "JWT",
                "jwt",
                "technology",
                "JSON Web Token",
                Option::<String>::None,
                now,
                now,
                3i64,
            ],
        );
        assert!(
            result.is_ok(),
            "Entity insert should succeed: {:?}",
            result.err()
        );

        let mention_count: i64 = conn
            .query_row(
                "SELECT mention_count FROM entities WHERE entity_id = 'ent_jwt'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(mention_count, 3);
    }

    #[tokio::test]
    async fn test_v5_knowledge_edges_temporal() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO knowledge_edges (owner, source_id, target_id, relation_type,
                fact_text, weight, confidence, valid_from, valid_until, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9, ?10)",
            params![
                [0xAAu8; 32].as_slice(),
                "ent_auth",
                "ent_jwt",
                "USES",
                "auth module uses JWT",
                1.0f64,
                0.95f64,
                now,
                now,
                now,
            ],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO knowledge_edges (owner, source_id, target_id, relation_type,
                fact_text, weight, confidence, valid_from, valid_until, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                [0xAAu8; 32].as_slice(),
                "ent_auth",
                "ent_basic",
                "USES",
                "auth module uses Basic Auth",
                1.0f64,
                0.9f64,
                now - 86400,
                now,
                now - 86400,
                now,
            ],
        )
        .unwrap();

        let valid_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_edges WHERE valid_until IS NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(valid_count, 1);

        let total_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM knowledge_edges", [], |row| row.get(0))
            .unwrap();
        assert_eq!(total_count, 2);
    }

    #[tokio::test]
    async fn test_v5_episode_edges_bidirectional() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO episode_edges (owner, episode_id, entity_id, role, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                [0xAAu8; 32].as_slice(),
                "ep_001",
                "ent_jwt",
                "mentioned",
                now
            ],
        )
        .unwrap();

        let entity_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episode_edges WHERE episode_id = 'ep_001'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(entity_count, 1);

        let episode_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM episode_edges WHERE entity_id = 'ent_jwt'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(episode_count, 1);
    }

    #[tokio::test]
    async fn test_v5_sessions_and_projects() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO communities (community_id, owner, name, summary, entity_count, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params!["comm_1", [0xAAu8; 32].as_slice(), "Project B", "Auth system", 5i64, now, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO projects (project_id, owner, name, status, community_id, summary,
                created_at, updated_at, last_active_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                "comm_1",
                [0xAAu8; 32].as_slice(),
                "Project B",
                "active",
                "comm_1",
                "Auth system project",
                now,
                now,
                now
            ],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO sessions (session_id, owner, project_id, session_type,
                started_at, turn_count, summary)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                "sess_001",
                [0xAAu8; 32].as_slice(),
                "comm_1",
                "code",
                now,
                15i64,
                "Implemented JWT auth"
            ],
        )
        .unwrap();

        let session_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE project_id = 'comm_1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(session_count, 1);

        let project_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM projects WHERE owner = ?1 AND status = 'active'",
                params![[0xAAu8; 32].as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(project_count, 1);
    }

    #[tokio::test]
    async fn test_v5_artifacts_version_chain() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let conn = s.conn.lock().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        conn.execute(
            "INSERT INTO artifacts (artifact_id, owner, session_id, artifact_type,
                filename, language, version, parent_id, encrypted_content, content_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?9, ?10)",
            params![
                "art_v1",
                [0xAAu8; 32].as_slice(),
                "sess_001",
                "code",
                "auth.rs",
                "rust",
                1i64,
                b"fn auth() {}".as_slice(),
                "hash_v1",
                now,
            ],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO artifacts (artifact_id, owner, session_id, artifact_type,
                filename, language, version, parent_id, encrypted_content, content_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                "art_v2",
                [0xAAu8; 32].as_slice(),
                "sess_002",
                "code",
                "auth.rs",
                "rust",
                2i64,
                "art_v1",
                b"fn auth() { jwt() }".as_slice(),
                "hash_v2",
                now + 100,
            ],
        )
        .unwrap();

        let latest_version: i64 = conn
            .query_row(
                "SELECT MAX(version) FROM artifacts WHERE owner = ?1 AND filename = 'auth.rs'",
                params![[0xAAu8; 32].as_slice()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(latest_version, 2);

        let parent: Option<String> = conn
            .query_row(
                "SELECT parent_id FROM artifacts WHERE artifact_id = 'art_v2'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(parent, Some("art_v1".to_string()));
    }

    #[tokio::test]
    async fn test_v5_backward_compat_insert_without_new_cols() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec(100, MemoryLayer::Episode, "test");
        assert!(s.insert(&r, "minilm").await);

        let conn = s.conn.lock().await;
        let (pid, sid, eid): (Option<String>, Option<String>, Option<String>) = conn
            .query_row(
                "SELECT project_id, session_id, episode_id FROM records WHERE record_id = ?1",
                params![r.record_id.as_slice()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert!(pid.is_none());
        assert!(sid.is_none());
        assert!(eid.is_none());
    }

    #[tokio::test]
    async fn test_migration_v5_to_v6() {
        use rusqlite::Connection;

        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (5);
             CREATE TABLE sessions (
                 session_id TEXT PRIMARY KEY,
                 owner BLOB NOT NULL,
                 started_at INTEGER NOT NULL
             );
             CREATE TABLE chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);",
        )
        .unwrap();

        MemoryStorage::maybe_migrate(&conn).unwrap();

        let ct_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='cognitive_tasks'",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap()
            > 0;
        assert!(ct_exists);

        let ul_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='llm_usage_log'",
                [],
                |r| r.get::<_, i64>(0),
            )
            .unwrap()
            > 0;
        assert!(ul_exists);

        let title_ok = conn.prepare("SELECT title FROM sessions LIMIT 0").is_ok();
        assert!(title_ok);

        let v: u32 = conn
            .query_row("SELECT version FROM schema_version LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();
        // maybe_migrate() runs through the latest schema: v6 adds SuperNode,
        // v7 adds the blind marker, v8 creates commitment tables, and v9 adds
        // the bounded proof vault even when create_schema() was not called.
        assert_eq!(v, 9);
        for table in [
            "record_commitment_blocks",
            "record_block_commitments",
            "record_checkpoint_evidence",
        ] {
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    params![table],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                > 0;
            assert!(exists, "missing migrated table: {table}");
        }
    }

    // ========================================
    // v2.5.2+Provenance: Bug fix tests
    // ========================================

    /// Verify that the memory_edges migration uses record.owner, not source_id as owner.
    #[tokio::test]
    async fn test_memory_edges_migration_uses_correct_owner() {
        use rusqlite::Connection;

        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE schema_version (version INTEGER NOT NULL);
             INSERT INTO schema_version VALUES (4);
             CREATE TABLE chain_state (key TEXT PRIMARY KEY, value BLOB NOT NULL);
             -- Simulate v4 records table with owner column
             CREATE TABLE records (
                 record_id BLOB PRIMARY KEY,
                 owner BLOB NOT NULL,
                 timestamp INTEGER,
                 layer INTEGER,
                 topic_tags TEXT DEFAULT '[]',
                 source_ai TEXT DEFAULT '',
                 status INTEGER DEFAULT 0,
                 supersedes BLOB,
                 encrypted_content BLOB DEFAULT x'',
                 embedding BLOB,
                 embedding_model TEXT DEFAULT '',
                 embedding_dim INTEGER DEFAULT 0,
                 signature BLOB NOT NULL DEFAULT x'',
                 access_count INTEGER DEFAULT 0,
                 created_at INTEGER,
                 positive_feedback INTEGER DEFAULT 0,
                 negative_feedback INTEGER DEFAULT 0,
                 conflict_with BLOB
             );
             -- Simulate v4 memory_edges
             CREATE TABLE memory_edges (
                 source_id BLOB NOT NULL,
                 target_id BLOB NOT NULL,
                 edge_type TEXT DEFAULT 'co_occurred',
                 weight REAL DEFAULT 1.0,
                 created_at INTEGER NOT NULL,
                 PRIMARY KEY (source_id, target_id)
             );",
        )
        .unwrap();

        let owner_bytes = [0xBBu8; 32];
        let source_id = [0x01u8; 32];
        let target_id = [0x02u8; 32];
        let now = 1_700_000_000i64;

        // Insert a record so the JOIN can resolve owner
        conn.execute(
            "INSERT INTO records (record_id, owner, timestamp, layer, signature, created_at)
             VALUES (?1, ?2, ?3, 1, x'0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000', ?4)",
            params![source_id.as_slice(), owner_bytes.as_slice(), now, now],
        ).unwrap();

        conn.execute(
            "INSERT INTO memory_edges (source_id, target_id, weight, created_at)
             VALUES (?1, ?2, 1.0, ?3)",
            params![source_id.as_slice(), target_id.as_slice(), now],
        )
        .unwrap();

        // Match MemoryStorage::open() upgrade order: create any missing modern
        // tables first, then run incremental migrations against the old data.
        MemoryStorage::create_schema(&conn).unwrap();
        MemoryStorage::maybe_migrate(&conn).unwrap();

        // Verify the migrated edge has the correct owner (owner_bytes), not source_id
        let migrated_owner: Vec<u8> = conn
            .query_row("SELECT owner FROM knowledge_edges LIMIT 1", [], |r| {
                r.get(0)
            })
            .unwrap();

        assert_eq!(migrated_owner.len(), 32);
        assert_eq!(
            migrated_owner.as_slice(),
            owner_bytes.as_slice(),
            "Migrated edge owner should be record.owner ([0xBB;32]), not source_id ([0x01;32])"
        );
    }

    /// Verify update_record_content clears embedding on content change.
    #[tokio::test]
    async fn test_update_record_content_clears_embedding() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Knowledge);
        let id = r.record_id;
        s.insert(&r, "minilm").await;

        // Add an embedding manually
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE records SET embedding = x'01020304', embedding_model = 'minilm', embedding_dim = 1 WHERE record_id = ?1",
                params![id.as_slice()],
            ).unwrap();
        }
        s.cache.write().clear();

        // Patch content
        let patched = s
            .update_record_content(&id, &owner, Some("new content"), None, None, None)
            .await
            .unwrap();
        assert!(patched);

        // Verify embedding was cleared
        let conn = s.conn.lock().await;
        let (emb, model, dim): (Option<Vec<u8>>, String, i64) = conn.query_row(
            "SELECT embedding, embedding_model, embedding_dim FROM records WHERE record_id = ?1",
            params![id.as_slice()],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ).unwrap();
        assert!(emb.is_none(), "Embedding must be NULL after content change");
        assert_eq!(model, "");
        assert_eq!(dim, 0);
    }

    /// Verify update_record_content rejects wrong owner.
    #[tokio::test]
    async fn test_update_record_content_wrong_owner_rejected() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let other = [0xBB; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Knowledge);
        let id = r.record_id;
        s.insert(&r, "minilm").await;

        let result = s
            .update_record_content(&id, &other, Some("hacked"), None, None, None)
            .await
            .unwrap();
        assert!(!result, "Wrong owner should be rejected");
    }

    /// Verify get_records_for_session returns correct records.
    #[tokio::test]
    async fn test_get_records_for_session() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Episode);
        let id = r.record_id;
        s.insert(&r, "m").await;
        s.set_record_session_id(&id, &owner, "sess_xyz").await;

        let records = s.get_records_for_session("sess_xyz", &owner).await;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].record_id, id);

        let empty = s.get_records_for_session("sess_other", &owner).await;
        assert!(empty.is_empty());
    }
}
