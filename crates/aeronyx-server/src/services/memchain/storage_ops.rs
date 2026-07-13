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
//! - RawLog: insert_raw_log, read_rawlog_content, update_rawlog_feedback,
//!   get_unprocessed_rawlogs, get_rawlogs_for_session (→ storage_miner.rs)
//! - Feedback: increment_positive/negative_feedback, set_conflict_with,
//!   insert_feedback, get_recent_feedback
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
//!
//! ## Split Architecture (v2.4.0+Search)
//! This file was split into three files to reduce size:
//!   - storage_ops.rs   (this file) — core/legacy ops (RawLog, Feedback, Stats, Miner, etc.)
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
//!   the new chain. Never update its tip independently of the block transaction.
//! - Peer range reads return commitments only; full records remain owner-scoped.
//!
//! ## Modification History
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
//!
//! ## Last Modified
//! v2.7.4-BlockIntegrityStatus - Report and maintain the verified-chain baseline.
//! v2.7.3-BlockAudit - Verify persisted blocks and indexes before networking starts.
//! v2.7.1-BlockSyncStatus - Privacy-safe follower status, failures, and recovery.
//! v2.7.0-BlockSync - Transactional commitment chain, ranges, and safe status.

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::{debug, error, info, warn};

use aeronyx_core::ledger::{
    MemoryLayer, MemoryRecord, RecordCommitmentBlockV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
    GENESIS_PREV_HASH,
};

use super::storage::{
    embedding_to_bytes, LayerCounts, MemoryStorage, RawLogRow, RecordCommitmentIntegrityRuntime,
    RecordCommitmentSyncEvent, RecordCommitmentSyncRuntime, RecordCommitmentSyncStatus,
    StorageStats, COMMITMENT_SYNC_EVENT_CAPACITY,
};
use super::storage_crypto::{
    decrypt_rawlog_content, decrypt_record_content, encrypt_rawlog_content, encrypt_record_content,
};

// ============================================
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

/// Result of atomically appending a V1 commitment block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCommitmentAppendOutcome {
    /// The block extended the local verified tip.
    Inserted,
    /// The exact same block was already stored at that height.
    AlreadyPresent,
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
    /// Explicit scope and privacy boundary for operators.
    pub verification_policy: &'static str,
}

/// Result of a complete persisted commitment-chain integrity audit.
///
/// The report is aggregate and contains no commitment, owner, peer, endpoint,
/// payload, or routing metadata. Audit failures identify only the block height
/// and violated invariant so startup logs cannot become a privacy side channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordCommitmentChainAudit {
    /// Number of fully verified persisted blocks.
    pub block_count: u64,
    /// Number of commitments verified against the membership index.
    pub commitment_count: u64,
    /// Last verified height, or zero for an empty chain.
    pub tip_height: u64,
}

struct StoredRecordCommitmentBlockRow {
    height: i64,
    block_hash: Vec<u8>,
    chain_id: Vec<u8>,
    protocol_version: i64,
    timestamp: i64,
    prev_block_hash: Vec<u8>,
    merkle_root: Vec<u8>,
    record_count: i64,
    proposer: Vec<u8>,
    proposer_signature: Vec<u8>,
    payload: Vec<u8>,
}

// A valid V1 block contains at most 256 fixed-size commitments and serializes
// below 9 KiB. Keep startup decoding bounded even if the SQLite file was
// replaced or modified outside the process.
const MAX_STORED_COMMITMENT_BLOCK_BYTES: usize = 16 * 1024;

// ============================================
// impl MemoryStorage — RawLog Operations
// ============================================

impl MemoryStorage {
    pub async fn insert_raw_log(
        &self,
        session_id: &str,
        turn_index: i64,
        role: &str,
        content: &str,
        source_ai: &str,
        recall_context: Option<&str>,
        extractable: i64,
        feedback_signal: Option<i64>,
        rawlog_key: Option<&[u8; 32]>,
    ) -> Result<i64, String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let (stored_content, encrypted_flag): (Vec<u8>, i64) = match rawlog_key {
            Some(key) => match encrypt_rawlog_content(key, content.as_bytes()) {
                Ok(ciphertext) => (ciphertext, 1),
                Err(e) => {
                    warn!(
                        "[STORAGE] RawLog encryption failed, storing plaintext: {}",
                        e
                    );
                    (content.as_bytes().to_vec(), 0)
                }
            },
            None => (content.as_bytes().to_vec(), 0),
        };

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO raw_logs (session_id, turn_index, role, content, source_ai,
                recall_context, extractable, feedback_signal, encrypted, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                session_id,
                turn_index,
                role,
                stored_content,
                source_ai,
                recall_context,
                extractable,
                feedback_signal,
                encrypted_flag,
                now,
            ],
        )
        .map_err(|e| format!("RawLog insert: {}", e))?;

        let log_id = conn.last_insert_rowid();
        Ok(log_id)
    }

    pub async fn read_rawlog_content(
        &self,
        log_id: i64,
        rawlog_key: Option<&[u8; 32]>,
    ) -> Option<String> {
        let conn = self.conn.lock().await;
        let row: Option<(Vec<u8>, i64)> = conn
            .query_row(
                "SELECT content, encrypted FROM raw_logs WHERE log_id = ?1",
                params![log_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .ok()?;

        let (content_bytes, encrypted) = row?;

        if encrypted == 0 {
            String::from_utf8(content_bytes).ok()
        } else if let Some(key) = rawlog_key {
            decrypt_rawlog_content(key, &content_bytes)
                .ok()
                .and_then(|bytes| String::from_utf8(bytes).ok())
        } else {
            warn!("[STORAGE] Encrypted rawlog but no key provided");
            None
        }
    }

    pub async fn update_rawlog_feedback(&self, log_id: i64, signal: i64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE raw_logs SET feedback_signal = ?1 WHERE log_id = ?2",
            params![signal, log_id],
        );
    }

    pub async fn get_unprocessed_rawlogs(&self, limit: usize) -> Vec<RawLogRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT log_id, session_id, turn_index, role, content, encrypted,
                    recall_context, extractable, feedback_signal
             FROM raw_logs
             WHERE feedback_signal IS NULL
             ORDER BY session_id, turn_index
             LIMIT ?1",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![limit as i64], |row| {
            Ok(RawLogRow {
                log_id: row.get(0)?,
                session_id: row.get(1)?,
                turn_index: row.get(2)?,
                role: row.get(3)?,
                content: row.get(4)?,
                encrypted: row.get(5)?,
                recall_context: row.get(6)?,
                extractable: row.get(7)?,
                feedback_signal: row.get(8)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — Feedback Operations
// ============================================

impl MemoryStorage {
    pub async fn increment_positive_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET positive_feedback = positive_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] positive_feedback incremented"
                );
                self.cache.write().invalidate(record_id);
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] positive_feedback: record not found"
                );
            }
            Err(e) => {
                error!(
                    record_id = hex::encode(record_id), error = %e,
                    "[STORAGE] ❌ positive_feedback increment failed (schema migration needed?)"
                );
            }
        }
    }

    pub async fn increment_negative_feedback(&self, record_id: &[u8; 32]) {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET negative_feedback = negative_feedback + 1 WHERE record_id = ?1",
            params![record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] negative_feedback incremented"
                );
                self.cache.write().invalidate(record_id);
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] negative_feedback: record not found"
                );
            }
            Err(e) => {
                error!(
                    record_id = hex::encode(record_id), error = %e,
                    "[STORAGE] ❌ negative_feedback increment failed (schema migration needed?)"
                );
            }
        }
    }

    pub async fn set_conflict_with(&self, record_id: &[u8; 32], conflict_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE records SET conflict_with = ?1 WHERE record_id = ?2",
            params![conflict_id.as_slice(), record_id.as_slice()],
        ) {
            Ok(n) if n > 0 => {
                debug!(
                    record_id = hex::encode(record_id),
                    conflict = hex::encode(conflict_id),
                    "[STORAGE] conflict_with set"
                );
                self.cache.write().invalidate(record_id);
                true
            }
            Ok(_) => {
                warn!(
                    record_id = hex::encode(record_id),
                    "[STORAGE] conflict_with: record not found"
                );
                false
            }
            Err(e) => {
                error!(error = %e, "[STORAGE] ❌ set_conflict_with failed");
                false
            }
        }
    }

    pub async fn insert_feedback(
        &self,
        owner: &[u8; 32],
        memory_id: &[u8; 32],
        session_id: &str,
        turn_index: i64,
        signal: i64,
        features: Option<&[f32; 9]>,
        prediction: Option<f32>,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let features_blob: Option<Vec<u8>> =
            features.map(|f| f.iter().flat_map(|v| v.to_le_bytes()).collect());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO memory_feedback (owner, memory_id, session_id, turn_index,
                signal, features, prediction, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                owner.as_slice(),
                memory_id.as_slice(),
                session_id,
                turn_index,
                signal,
                features_blob.as_deref(),
                prediction,
                now,
            ],
        );
    }

    pub async fn get_recent_feedback(&self, limit: usize) -> Vec<(i64, f32)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT signal, prediction FROM memory_feedback ORDER BY created_at DESC LIMIT ?1",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        stmt.query_map(params![limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1).unwrap_or(0.0)))
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

// ============================================
// impl MemoryStorage — Chain State
// ============================================

fn privacy_safe_sync_error_code(reason: &str) -> String {
    match reason {
        "invalid_pinned_coordinator"
        | "coordinator_self_reference"
        | "http_client_init_failed"
        | "pinned_coordinator_unavailable"
        | "pinned_coordinator_missing_endpoint"
        | "pinned_coordinator_invalid_endpoint"
        | "request_encode_failed"
        | "request_timeout"
        | "request_connect"
        | "request_body"
        | "request_decode"
        | "request_request"
        | "request_unknown"
        | "response_body_timeout"
        | "response_body_connect"
        | "response_body_body"
        | "response_body_decode"
        | "response_body_request"
        | "response_body_unknown"
        | "response_too_large"
        | "invalid_response_frame"
        | "unexpected_response_message"
        | "response_request_mismatch"
        | "response_responder_mismatch"
        | "stale_response"
        | "response_page_too_large"
        | "invalid_response_signature"
        | "invalid_local_genesis"
        | "coordinator_rollback_detected"
        | "empty_page_tip_mismatch"
        | "unexpected_blocks_at_current_tip"
        | "unexpected_block_proposer"
        | "commitment_chain_verification_failed"
        | "pagination_state_mismatch"
        | "terminal_tip_mismatch"
        | "storage_append_rejected" => reason.to_string(),
        // Exact status codes are useful in process logs but add needless
        // cardinality to public health data, so all 3-digit responses collapse
        // to one stable evidence code.
        _ if reason.strip_prefix("http_status_").is_some_and(|status| {
            status.len() == 3 && status.bytes().all(|byte| byte.is_ascii_digit())
        }) =>
        {
            "http_status_error".to_string()
        }
        _ => "internal_sync_error".to_string(),
    }
}

fn push_commitment_sync_event(
    runtime: &mut RecordCommitmentSyncRuntime,
    timestamp: u64,
    kind: &'static str,
    error_code: Option<String>,
    next_poll_at: Option<u64>,
) {
    let event = RecordCommitmentSyncEvent {
        sequence: runtime.next_event_sequence,
        timestamp,
        kind: kind.to_string(),
        error_code,
        consecutive_failures: runtime.consecutive_failures,
        next_poll_at,
    };
    runtime.next_event_sequence = runtime.next_event_sequence.saturating_add(1);
    if runtime.recent_events.len() == COMMITMENT_SYNC_EVENT_CAPACITY {
        runtime.recent_events.pop_front();
    }
    runtime.recent_events.push_back(event);
}

impl MemoryStorage {
    /// Configures the process-local Block Sync role once startup validation has
    /// completed. This does not alter SQLite or the canonical block chain.
    pub fn configure_record_commitment_sync(&self, coordinator: bool, follower: bool) {
        let mut runtime = self.commitment_sync.write();
        *runtime = RecordCommitmentSyncRuntime::default();
        match (coordinator, follower) {
            (true, false) => {
                runtime.role = "coordinator";
                runtime.state = "producing";
            }
            (false, true) => {
                runtime.role = "follower";
                runtime.state = "starting";
                runtime.enabled = true;
            }
            (false, false) => {}
            (true, true) => {
                runtime.state = "configuration_error";
                runtime.last_error_code = Some("role_conflict".to_string());
            }
        }
    }

    /// Marks the start of one bounded follower pull round.
    pub fn record_commitment_sync_attempt(&self, now: u64) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        runtime.state = "syncing";
        runtime.last_attempt_at = Some(now);
        runtime.next_poll_at = None;
    }

    /// Records one fully verified response page after all included blocks have
    /// been accepted by the local transactional chain store.
    pub fn record_commitment_sync_page_success(
        &self,
        now: u64,
        verified_blocks: u64,
        remote_tip_height: u64,
        has_more: bool,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        let recovered = runtime.consecutive_failures > 0;
        runtime.state = if has_more { "catching_up" } else { "current" };
        runtime.last_success_at = Some(now);
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.pages_received_total = runtime.pages_received_total.saturating_add(1);
        runtime.blocks_received_total = runtime
            .blocks_received_total
            .saturating_add(verified_blocks);
        runtime.consecutive_failures = 0;
        if recovered {
            runtime.last_recovered_at = Some(now);
            runtime.recovery_events_total = runtime.recovery_events_total.saturating_add(1);
            push_commitment_sync_event(&mut runtime, now, "recovered", None, None);
        }
    }

    /// Records a fail-closed follower error using only a stable allow-listed
    /// code. Free text, URLs, identities, and endpoints are never retained.
    pub fn record_commitment_sync_failure(
        &self,
        now: u64,
        reason: &str,
        consecutive_failures: u32,
        next_poll_at: u64,
    ) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        let error_code = privacy_safe_sync_error_code(reason);
        runtime.state = "backoff";
        runtime.last_failure_at = Some(now);
        runtime.next_poll_at = Some(next_poll_at);
        runtime.consecutive_failures = consecutive_failures;
        runtime.last_error_code = Some(error_code.clone());
        runtime.failure_events_total = runtime.failure_events_total.saturating_add(1);
        push_commitment_sync_event(
            &mut runtime,
            now,
            "failure",
            Some(error_code),
            Some(next_poll_at),
        );
    }

    /// Records the next normal poll time after a successful bounded round.
    pub fn schedule_next_commitment_sync_poll(&self, next_poll_at: u64) {
        let mut runtime = self.commitment_sync.write();
        if runtime.enabled {
            runtime.next_poll_at = Some(next_poll_at);
        }
    }

    /// Marks follower shutdown without fabricating a failure event.
    pub fn stop_record_commitment_sync(&self) {
        let mut runtime = self.commitment_sync.write();
        if runtime.enabled {
            runtime.state = "stopped";
            runtime.next_poll_at = None;
        }
    }

    /// Returns a privacy-safe snapshot for local APIs and heartbeat reporting.
    pub fn record_commitment_sync_status(&self) -> RecordCommitmentSyncStatus {
        let runtime = self.commitment_sync.read();
        RecordCommitmentSyncStatus {
            contract_version: "record_commitment_sync.v1",
            role: runtime.role.to_string(),
            state: runtime.state.to_string(),
            enabled: runtime.enabled,
            last_attempt_at: runtime.last_attempt_at,
            last_success_at: runtime.last_success_at,
            last_failure_at: runtime.last_failure_at,
            last_recovered_at: runtime.last_recovered_at,
            next_poll_at: runtime.next_poll_at,
            consecutive_failures: runtime.consecutive_failures,
            last_error_code: runtime.last_error_code.clone(),
            remote_tip_height: runtime.remote_tip_height,
            pages_received_total: runtime.pages_received_total,
            blocks_received_total: runtime.blocks_received_total,
            failure_events_total: runtime.failure_events_total,
            recovery_events_total: runtime.recovery_events_total,
            recent_events: runtime.recent_events.iter().cloned().collect(),
            privacy_policy:
                "aggregate runtime only; no coordinator identity, endpoint, block hash, record commitment, owner, payload, route, or client metadata",
        }
    }

    /// Returns the privacy-safe chain-integrity baseline for this process.
    ///
    /// A persisted flag is intentionally not used: every process lifetime must
    /// re-establish the baseline from the complete SQLite chain before it may
    /// report `verified`.
    pub fn record_commitment_chain_integrity_status(&self) -> RecordCommitmentChainIntegrityStatus {
        const POLICY: &str = "full snapshot-consistent startup audit plus transactionally verified appends; no block hashes, proposer identities, commitment ids, owners, payloads, peers, endpoints, routes, or client metadata";
        match *self.commitment_integrity.read() {
            Some(runtime) => RecordCommitmentChainIntegrityStatus {
                contract_version: "record_commitment_integrity.v1",
                state: "verified",
                baseline_verified_at: Some(runtime.baseline_verified_at),
                last_verified_at: Some(runtime.last_verified_at),
                verification_duration_ms: Some(runtime.verification_duration_ms),
                verified_block_count: runtime.verified_block_count,
                verified_commitment_count: runtime.verified_commitment_count,
                verified_tip_height: runtime.verified_tip_height,
                verification_policy: POLICY,
            },
            None => RecordCommitmentChainIntegrityStatus {
                contract_version: "record_commitment_integrity.v1",
                state: "not_verified",
                baseline_verified_at: None,
                last_verified_at: None,
                verification_duration_ms: None,
                verified_block_count: 0,
                verified_commitment_count: 0,
                verified_tip_height: 0,
                verification_policy: POLICY,
            },
        }
    }

    /// Re-verifies the complete persisted commitment chain before networking.
    ///
    /// The audit treats the signed payload as canonical, verifies every block
    /// from genesis, and requires the denormalized SQL row plus membership
    /// index to describe that exact payload. Any mismatch fails closed; this
    /// method never repairs, truncates, or rewrites evidence automatically.
    pub async fn audit_record_commitment_chain(
        &self,
    ) -> Result<RecordCommitmentChainAudit, String> {
        // Clear first so a failed re-audit can never leave stale `verified`
        // evidence visible to operators or the central health plane.
        *self.commitment_integrity.write() = None;
        let started = Instant::now();
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
            .map_err(|error| format!("begin commitment audit snapshot: {error}"))?;
        let mut block_statement = transaction
            .prepare(
                "SELECT height,block_hash,chain_id,protocol_version,timestamp,
                        prev_block_hash,merkle_root,record_count,proposer,
                        proposer_signature,payload
                 FROM record_commitment_blocks ORDER BY height ASC",
            )
            .map_err(|error| format!("prepare commitment audit blocks: {error}"))?;
        let mut membership_statement = transaction
            .prepare(
                "SELECT record_id FROM record_block_commitments
                 WHERE block_height=?1 ORDER BY record_id ASC",
            )
            .map_err(|error| format!("prepare commitment audit memberships: {error}"))?;
        let mut rows = block_statement
            .query([])
            .map_err(|error| format!("query commitment audit blocks: {error}"))?;

        let mut expected_height = 1u64;
        let mut expected_prev_hash = GENESIS_PREV_HASH;
        let mut block_count = 0u64;
        let mut commitment_count = 0u64;

        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read commitment audit block: {error}"))?
        {
            let stored = StoredRecordCommitmentBlockRow {
                height: row
                    .get(0)
                    .map_err(|error| format!("read commitment audit height: {error}"))?,
                block_hash: row
                    .get(1)
                    .map_err(|error| format!("read commitment audit hash: {error}"))?,
                chain_id: row
                    .get(2)
                    .map_err(|error| format!("read commitment audit chain: {error}"))?,
                protocol_version: row
                    .get(3)
                    .map_err(|error| format!("read commitment audit version: {error}"))?,
                timestamp: row
                    .get(4)
                    .map_err(|error| format!("read commitment audit timestamp: {error}"))?,
                prev_block_hash: row
                    .get(5)
                    .map_err(|error| format!("read commitment audit previous hash: {error}"))?,
                merkle_root: row
                    .get(6)
                    .map_err(|error| format!("read commitment audit merkle root: {error}"))?,
                record_count: row
                    .get(7)
                    .map_err(|error| format!("read commitment audit count: {error}"))?,
                proposer: row
                    .get(8)
                    .map_err(|error| format!("read commitment audit proposer: {error}"))?,
                proposer_signature: row
                    .get(9)
                    .map_err(|error| format!("read commitment audit signature: {error}"))?,
                payload: row
                    .get(10)
                    .map_err(|error| format!("read commitment audit payload: {error}"))?,
            };

            let stored_height = u64::try_from(stored.height)
                .map_err(|_| "commitment audit found an invalid stored height".to_string())?;
            if stored.payload.len() > MAX_STORED_COMMITMENT_BLOCK_BYTES {
                return Err(format!(
                    "commitment audit payload exceeds bound at height {stored_height}"
                ));
            }
            let block =
                bincode::deserialize::<RecordCommitmentBlockV1>(&stored.payload).map_err(|_| {
                    format!("commitment audit payload decode failed at height {stored_height}")
                })?;
            let canonical_payload = bincode::serialize(&block).map_err(|_| {
                format!("commitment audit payload encode failed at height {stored_height}")
            })?;
            if canonical_payload != stored.payload {
                return Err(format!(
                    "commitment audit payload is non-canonical at height {stored_height}"
                ));
            }

            block
                .verify(
                    &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                    expected_height,
                    &expected_prev_hash,
                )
                .map_err(|error| {
                    format!(
                        "commitment audit block validation failed at height {stored_height}: {error}"
                    )
                })?;

            let block_height = i64::try_from(block.header.height).map_err(|_| {
                format!("commitment audit height overflow at height {stored_height}")
            })?;
            let block_timestamp = i64::try_from(block.header.timestamp).map_err(|_| {
                format!("commitment audit timestamp overflow at height {stored_height}")
            })?;
            let block_hash = block.hash();
            if stored.height != block_height
                || stored.block_hash.as_slice() != block_hash.as_slice()
                || stored.chain_id.as_slice() != block.header.chain_id.as_slice()
                || stored.protocol_version != i64::from(block.header.protocol_version)
                || stored.timestamp != block_timestamp
                || stored.prev_block_hash.as_slice() != block.header.prev_block_hash.as_slice()
                || stored.merkle_root.as_slice() != block.header.merkle_root.as_slice()
                || stored.record_count != i64::from(block.header.record_count)
                || stored.proposer.as_slice() != block.header.proposer.as_slice()
                || stored.proposer_signature.as_slice() != block.proposer_signature.as_slice()
            {
                return Err(format!(
                    "commitment audit stored row mismatch at height {stored_height}"
                ));
            }

            let mut membership_rows =
                membership_statement
                    .query(params![stored.height])
                    .map_err(|error| {
                        format!(
                            "query commitment audit memberships at height {stored_height}: {error}"
                        )
                    })?;
            let mut indexed_record_ids = Vec::with_capacity(block.record_ids.len());
            while let Some(membership_row) = membership_rows.next().map_err(|error| {
                format!("read commitment audit membership at height {stored_height}: {error}")
            })? {
                let bytes: Vec<u8> = membership_row.get(0).map_err(|error| {
                    format!("decode commitment audit membership at height {stored_height}: {error}")
                })?;
                let record_id: [u8; 32] = bytes.try_into().map_err(|bytes: Vec<u8>| {
                    format!(
                        "commitment audit membership length {} at height {stored_height}",
                        bytes.len()
                    )
                })?;
                indexed_record_ids.push(record_id);
            }
            if indexed_record_ids != block.record_ids {
                return Err(format!(
                    "commitment audit membership index mismatch at height {stored_height}"
                ));
            }

            block_count = block_count.saturating_add(1);
            commitment_count = commitment_count.saturating_add(block.record_ids.len() as u64);
            expected_height = expected_height.saturating_add(1);
            expected_prev_hash = block_hash;
        }

        drop(rows);
        drop(block_statement);
        drop(membership_statement);
        let indexed_total: i64 = transaction
            .query_row("SELECT COUNT(*) FROM record_block_commitments", [], |row| {
                row.get(0)
            })
            .map_err(|error| format!("count commitment audit memberships: {error}"))?;
        let indexed_total = u64::try_from(indexed_total)
            .map_err(|_| "commitment audit found an invalid membership count".to_string())?;
        if indexed_total != commitment_count {
            return Err("commitment audit contains orphaned membership rows".to_string());
        }
        transaction
            .commit()
            .map_err(|error| format!("commit commitment audit snapshot: {error}"))?;

        let report = RecordCommitmentChainAudit {
            block_count,
            commitment_count,
            tip_height: expected_height.saturating_sub(1),
        };
        let verified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let verification_duration_ms =
            u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        *self.commitment_integrity.write() = Some(RecordCommitmentIntegrityRuntime {
            baseline_verified_at: verified_at,
            last_verified_at: verified_at,
            verification_duration_ms,
            verified_block_count: report.block_count,
            verified_commitment_count: report.commitment_count,
            verified_tip_height: report.tip_height,
        });
        drop(conn);
        Ok(report)
    }

    /// Atomically verifies and appends one node-blind commitment block.
    ///
    /// Height uniqueness, previous-hash continuity, proposer signature,
    /// Merkle integrity, commitment uniqueness, block persistence, membership
    /// indexing, and tip updates share one SQLite transaction. A crash or
    /// validation failure therefore cannot leave a partially advanced chain.
    pub async fn append_record_commitment_block(
        &self,
        block: &RecordCommitmentBlockV1,
        received_from: Option<&[u8; 32]>,
    ) -> Result<RecordCommitmentAppendOutcome, String> {
        let block_height = i64::try_from(block.header.height)
            .map_err(|_| "commitment block height exceeds SQLite range".to_string())?;
        let block_timestamp = i64::try_from(block.header.timestamp)
            .map_err(|_| "commitment block timestamp exceeds SQLite range".to_string())?;
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin commitment block transaction: {error}"))?;

        let existing_hash: Option<Vec<u8>> = transaction
            .query_row(
                "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                params![block_height],
                |row| row.get(0),
            )
            .optional()
            .map_err(|error| format!("read existing commitment block: {error}"))?;
        if let Some(existing_hash) = existing_hash {
            if existing_hash.as_slice() == block.hash().as_slice() {
                return Ok(RecordCommitmentAppendOutcome::AlreadyPresent);
            }
            return Err(format!(
                "commitment chain fork at height {}",
                block.header.height
            ));
        }

        let tip: Option<(i64, Vec<u8>)> = transaction
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| format!("read commitment chain tip: {error}"))?;
        let (expected_height, expected_prev_hash) = match tip {
            Some((height, hash)) => {
                let height = u64::try_from(height)
                    .map_err(|_| "stored commitment tip has invalid height".to_string())?;
                let hash: [u8; 32] = hash.try_into().map_err(|value: Vec<u8>| {
                    format!("stored tip hash has invalid length {}", value.len())
                })?;
                (height.saturating_add(1), hash)
            }
            None => (1, GENESIS_PREV_HASH),
        };

        block
            .verify(
                &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
                expected_height,
                &expected_prev_hash,
            )
            .map_err(|error| format!("commitment block validation failed: {error}"))?;

        let payload = bincode::serialize(block)
            .map_err(|error| format!("serialize commitment block: {error}"))?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        transaction
            .execute(
                "INSERT INTO record_commitment_blocks
                 (height,block_hash,chain_id,protocol_version,timestamp,
                  prev_block_hash,merkle_root,record_count,proposer,
                  proposer_signature,payload,received_from,created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
                params![
                    block_height,
                    block.hash().as_slice(),
                    block.header.chain_id.as_slice(),
                    block.header.protocol_version as i64,
                    block_timestamp,
                    block.header.prev_block_hash.as_slice(),
                    block.header.merkle_root.as_slice(),
                    block.header.record_count as i64,
                    block.header.proposer.as_slice(),
                    block.proposer_signature.as_slice(),
                    payload,
                    received_from.map(|peer| peer.as_slice()),
                    now,
                ],
            )
            .map_err(|error| format!("persist commitment block: {error}"))?;

        for record_id in &block.record_ids {
            transaction
                .execute(
                    "INSERT INTO record_block_commitments (record_id,block_height)
                     VALUES (?1,?2)",
                    params![record_id.as_slice(), block_height],
                )
                .map_err(|error| {
                    format!(
                        "persist commitment membership at height {}: {error}",
                        block.header.height
                    )
                })?;
        }

        for (key, value) in [
            ("record_block_tip_hash", block.hash().to_vec()),
            (
                "record_block_tip_height",
                block.header.height.to_le_bytes().to_vec(),
            ),
            (
                "record_block_chain_id",
                AERONYX_MEMCHAIN_MAINNET_CHAIN_ID.to_vec(),
            ),
        ] {
            transaction
                .execute(
                    "INSERT OR REPLACE INTO chain_state (key,value) VALUES (?1,?2)",
                    params![key, value],
                )
                .map_err(|error| format!("update commitment chain state: {error}"))?;
        }

        transaction
            .commit()
            .map_err(|error| format!("commit commitment block transaction: {error}"))?;

        // Preserve the full-audit guarantee across normal operation. If the
        // runtime baseline and committed append ever disagree on continuity,
        // discard the claim and require another complete audit.
        let appended_at = u64::try_from(now).unwrap_or_default();
        let mut integrity = self.commitment_integrity.write();
        let can_advance = integrity
            .as_ref()
            .map(|runtime| {
                runtime.verified_tip_height.saturating_add(1) == block.header.height
                    && runtime.verified_block_count.saturating_add(1) == block.header.height
            })
            .unwrap_or(false);
        if can_advance {
            if let Some(runtime) = integrity.as_mut() {
                runtime.last_verified_at = appended_at;
                runtime.verified_block_count = runtime.verified_block_count.saturating_add(1);
                runtime.verified_commitment_count = runtime
                    .verified_commitment_count
                    .saturating_add(block.record_ids.len() as u64);
                runtime.verified_tip_height = block.header.height;
            }
        } else if integrity.is_some() {
            *integrity = None;
        }
        drop(integrity);
        drop(conn);
        info!(
            height = block.header.height,
            commitments = block.record_ids.len(),
            hash = %block.header.hash_hex(),
            source = if received_from.is_some() { "peer" } else { "local" },
            "[MEMCHAIN_BLOCK] Verified commitment block appended"
        );
        Ok(RecordCommitmentAppendOutcome::Inserted)
    }

    /// Reads a bounded, height-ordered range for peer catch-up.
    pub async fn get_record_commitment_block_range(
        &self,
        from_height: u64,
        limit: usize,
    ) -> Result<Vec<RecordCommitmentBlockV1>, String> {
        let from_height = from_height.max(1);
        let from_height = i64::try_from(from_height).unwrap_or(i64::MAX);
        let limit = limit.clamp(1, 32);
        let conn = self.conn.lock().await;
        let mut statement = conn
            .prepare(
                "SELECT payload FROM record_commitment_blocks
                 WHERE height>=?1 ORDER BY height ASC LIMIT ?2",
            )
            .map_err(|error| format!("prepare commitment range query: {error}"))?;
        let rows = statement
            .query_map(params![from_height, limit as i64], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .map_err(|error| format!("query commitment block range: {error}"))?;

        let mut blocks = Vec::new();
        for row in rows {
            let payload = row.map_err(|error| format!("read commitment block payload: {error}"))?;
            let block = bincode::deserialize::<RecordCommitmentBlockV1>(&payload)
                .map_err(|error| format!("decode stored commitment block: {error}"))?;
            blocks.push(block);
        }
        Ok(blocks)
    }

    /// Returns the verified commitment chain tip from the authoritative block
    /// table rather than trusting mutable key/value state.
    pub async fn record_commitment_chain_tip(&self) -> (u64, [u8; 32]) {
        let conn = self.conn.lock().await;
        let tip: Option<(u64, Vec<u8>)> = conn
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, i64>(0)? as u64, row.get(1)?)),
            )
            .optional()
            .unwrap_or(None);
        match tip {
            Some((height, hash)) if hash.len() == 32 => {
                let mut value = [0u8; 32];
                value.copy_from_slice(&hash);
                (height, value)
            }
            _ => (0, GENESIS_PREV_HASH),
        }
    }

    /// Returns aggregate chain health without exposing record commitments,
    /// proposer identities, or peer metadata.
    pub async fn record_commitment_chain_status(&self) -> RecordCommitmentChainStatus {
        let conn = self.conn.lock().await;
        let (block_count, commitment_count): (u64, u64) = conn
            .query_row(
                "SELECT COUNT(*),COALESCE(SUM(record_count),0)
                 FROM record_commitment_blocks",
                [],
                |row| Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as u64)),
            )
            .unwrap_or((0, 0));
        let tip: Option<(u64, Vec<u8>)> = conn
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, i64>(0)? as u64, row.get(1)?)),
            )
            .optional()
            .unwrap_or(None);
        let (tip_height, tip_hash) = match tip {
            Some((height, hash)) if hash.len() == 32 => (height, Some(hex::encode(hash))),
            _ => (0, None),
        };
        let integrity = self.record_commitment_chain_integrity_status();
        drop(conn);
        RecordCommitmentChainStatus {
            contract_version: "record_commitment_chain.v1",
            chain_id: hex::encode(AERONYX_MEMCHAIN_MAINNET_CHAIN_ID),
            block_count,
            commitment_count,
            tip_height,
            tip_hash,
            payload_policy: "opaque_record_commitments_only_no_memory_payload_or_owner_metadata",
            integrity,
        }
    }

    /// Legacy mutable chain-state setter retained for old Fact blocks only.
    pub async fn set_chain_state(&self, block_hash: &[u8; 32], height: u64) {
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT OR REPLACE INTO chain_state (key,value) VALUES ('last_block_hash',?1)",
            params![block_hash.as_slice()],
        );
        let _ = conn.execute(
            "INSERT OR REPLACE INTO chain_state (key,value) VALUES ('last_block_height',?1)",
            params![height.to_le_bytes().as_slice()],
        );
    }

    pub async fn last_block_hash(&self) -> [u8; 32] {
        let (commitment_height, commitment_hash) = self.record_commitment_chain_tip().await;
        if commitment_height > 0 {
            return commitment_hash;
        }
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_hash'",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                let mut h = [0u8; 32];
                if blob.len() == 32 {
                    h.copy_from_slice(&blob);
                }
                Ok(h)
            },
        )
        .unwrap_or([0u8; 32])
    }

    pub async fn last_block_height(&self) -> u64 {
        let (commitment_height, _) = self.record_commitment_chain_tip().await;
        if commitment_height > 0 {
            return commitment_height;
        }
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT value FROM chain_state WHERE key='last_block_height'",
            [],
            |row| {
                let blob: Vec<u8> = row.get(0)?;
                if blob.len() == 8 {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&blob);
                    Ok(u64::from_le_bytes(b))
                } else {
                    Ok(0u64)
                }
            },
        )
        .unwrap_or(0)
    }
}

// ============================================
// impl MemoryStorage — Statistics
// ============================================

impl MemoryStorage {
    pub async fn stats(&self) -> StorageStats {
        let conn = self.conn.lock().await;
        let q = |sql: &str, p: &[&dyn rusqlite::ToSql]| -> u64 {
            conn.query_row(sql, p, |r| r.get::<_, i64>(0)).unwrap_or(0) as u64
        };
        let total = q("SELECT COUNT(*) FROM records", &[]);
        let active = q("SELECT COUNT(*) FROM records WHERE status=0", &[]);
        StorageStats {
            total_records: total,
            active_records: active,
            by_layer: LayerCounts {
                identity: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=0",
                    &[],
                ),
                knowledge: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=1",
                    &[],
                ),
                episode: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=2",
                    &[],
                ),
                archive: q(
                    "SELECT COUNT(*) FROM records WHERE status=0 AND layer=3",
                    &[],
                ),
            },
            content_bytes: q(
                "SELECT COALESCE(SUM(LENGTH(encrypted_content)),0) FROM records WHERE status=0",
                &[],
            ),
            records_with_embedding: q(
                "SELECT COUNT(*) FROM records WHERE status=0 AND embedding IS NOT NULL",
                &[],
            ),
            session_inserts: self.total_inserted(),
            session_rejects: self.total_rejected(),
        }
    }

    pub async fn count(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(*) FROM records", [], |r| r.get::<_, i64>(0))
            .unwrap_or(0) as usize
    }
}

// ============================================
// impl MemoryStorage — Miner Support
// ============================================

impl MemoryStorage {
    pub async fn count_by_layer(&self, layer: MemoryLayer) -> u64 {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT COUNT(*) FROM records WHERE status=0 AND layer=?1",
            params![layer as u8 as i64],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(0) as u64
    }

    pub async fn compact_episodes_to_archive(
        &self,
        owner: &[u8; 32],
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        let records = self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE owner=?1 AND status=0 AND layer=?2 ORDER BY timestamp ASC LIMIT ?3",
            params![
                owner.as_slice(),
                MemoryLayer::Episode as u8 as i64,
                limit as i64
            ],
        );
        if records.is_empty() {
            return records;
        }
        if conn.execute_batch("BEGIN TRANSACTION").is_err() {
            return Vec::new();
        }
        for r in &records {
            let now_ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            if let Err(e) = conn.execute(
                "UPDATE records SET layer=?1, archived_at=?2 WHERE record_id=?3",
                params![
                    MemoryLayer::Archive as u8 as i64,
                    now_ts,
                    r.record_id.as_slice()
                ],
            ) {
                error!(error=%e, "[STORAGE] ❌ Compact update failed, rolling back");
                let _ = conn.execute_batch("ROLLBACK");
                return Vec::new();
            }
        }
        if conn.execute_batch("COMMIT").is_err() {
            return Vec::new();
        }
        info!(
            count = records.len(),
            "[STORAGE] ⛏️ Episodes compacted to Archive layer"
        );
        records
    }

    pub async fn get_records_needing_embedding(&self, limit: usize) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE embedding IS NULL AND status = 0 LIMIT ?1",
            params![limit as i64],
        )
    }

    pub async fn get_correction_records(&self) -> Vec<MemoryRecord> {
        let conn = self.conn.lock().await;
        self.query_rows(
            &conn,
            "SELECT record_id,owner,timestamp,layer,topic_tags,source_ai,
                    status,supersedes,encrypted_content,embedding,signature,access_count,
                    positive_feedback,negative_feedback,conflict_with
             FROM records WHERE topic_tags LIKE '%_correction%' AND status = 0",
            [],
        )
    }

    pub async fn update_topic_tags(&self, record_id: &[u8; 32], tags: &[String]) {
        let json = serde_json::to_string(tags).unwrap_or_else(|_| "[]".into());
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "UPDATE records SET topic_tags = ?1 WHERE record_id = ?2",
            params![json, record_id.as_slice()],
        );
    }

    pub async fn supersede_record(&self, old_id: &[u8; 32], new_id: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        let r1 = conn.execute(
            "UPDATE records SET status = 1 WHERE record_id = ?1",
            params![old_id.as_slice()],
        );
        let r2 = conn.execute(
            "UPDATE records SET supersedes = ?1 WHERE record_id = ?2",
            params![old_id.as_slice(), new_id.as_slice()],
        );
        r1.is_ok() && r2.is_ok()
    }

    pub async fn load_user_weights(&self, owner: &[u8; 32]) -> Option<Vec<u8>> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT weights FROM user_weights WHERE owner = ?1",
            params![owner.as_slice()],
            |row| row.get::<_, Vec<u8>>(0),
        )
        .optional()
        .ok()?
    }

    pub async fn save_user_weights(&self, owner: &[u8; 32], weights_blob: &[u8], version: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let conn = self.conn.lock().await;
        let _ = conn.execute(
            "INSERT INTO user_weights (owner, weights, version, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(owner) DO UPDATE SET weights=?2, version=?3, updated_at=?4",
            params![owner.as_slice(), weights_blob, version as i64, now],
        );
    }
}

// ============================================
// impl MemoryStorage — Content Dedup
// ============================================

impl MemoryStorage {
    pub async fn has_active_content(&self, owner: &[u8; 32], content: &[u8]) -> bool {
        let compare_content: Vec<u8> = if let Some(ref key) = self.record_key {
            if content.is_empty() {
                content.to_vec()
            } else {
                match encrypt_record_content(key, content) {
                    Ok(ct) => ct,
                    Err(e) => {
                        warn!("[STORAGE] has_active_content encryption failed: {}", e);
                        content.to_vec()
                    }
                }
            }
        } else {
            content.to_vec()
        };
        let conn = self.conn.lock().await;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM records WHERE owner = ?1 AND encrypted_content = ?2 AND status = 0 LIMIT 1",
            params![owner.as_slice(), compare_content.as_slice()],
            |row| row.get(0),
        ).unwrap_or(0);
        count > 0
    }
}

// ============================================
// impl MemoryStorage — v2.2.0 MemExplorer
// ============================================

impl MemoryStorage {
    pub async fn get_embedding_model(&self, record_id: &[u8; 32]) -> Option<String> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT embedding_model FROM records WHERE record_id = ?1",
            params![record_id.as_slice()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .unwrap_or(None)
    }

    pub async fn get_overview(&self, owner: &[u8; 32], per_layer_limit: usize) -> OverviewData {
        let conn = self.conn.lock().await;
        let limit = per_layer_limit.min(50).max(1);
        let mut by_layer = HashMap::new();
        let layer_names = [
            (0i64, "identity"),
            (1, "knowledge"),
            (2, "episode"),
            (3, "archive"),
        ];
        for (layer_val, layer_name) in &layer_names {
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM records WHERE owner = ?1 AND status = 0 AND layer = ?2",
                    params![owner.as_slice(), layer_val],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            by_layer.insert(layer_name.to_string(), count as u64);
        }
        let mut recent_by_layer = HashMap::new();
        let rk = self.record_key.as_ref().map(|v| &**v);
        for (layer_val, layer_name) in &layer_names {
            let mut stmt = match conn.prepare(
                "SELECT record_id, encrypted_content, topic_tags, timestamp,
                        access_count, positive_feedback, negative_feedback, source_ai
                 FROM records WHERE owner = ?1 AND layer = ?2 AND status = 0
                 ORDER BY timestamp DESC LIMIT ?3",
            ) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let records: Vec<OverviewRecord> = stmt
                .query_map(params![owner.as_slice(), layer_val, limit as i64], |row| {
                    let rid_blob: Vec<u8> = row.get(0)?;
                    let raw_content: Vec<u8> = row.get(1)?;
                    let tags_json: String = row.get(2)?;
                    let timestamp: i64 = row.get(3)?;
                    let access_count: i64 = row.get(4)?;
                    let pos_fb: i64 = row.get(5)?;
                    let neg_fb: i64 = row.get(6)?;
                    let source_ai: String = row.get(7)?;
                    let content = if let Some(key) = rk {
                        if raw_content.len() >= 28 {
                            match decrypt_record_content(key, &raw_content) {
                                Ok(plain) => String::from_utf8_lossy(&plain).to_string(),
                                Err(_) => String::from_utf8_lossy(&raw_content).to_string(),
                            }
                        } else {
                            String::from_utf8_lossy(&raw_content).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&raw_content).to_string()
                    };
                    let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
                    Ok(OverviewRecord {
                        record_id: hex::encode(&rid_blob),
                        content,
                        topic_tags: tags,
                        timestamp: timestamp as u64,
                        access_count: access_count as u32,
                        positive_feedback: pos_fb as u32,
                        negative_feedback: neg_fb as u32,
                        source_ai,
                    })
                })
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();
            recent_by_layer.insert(layer_name.to_string(), records);
        }
        let last_memory_at: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(timestamp), 0) FROM records WHERE owner = ?1 AND status = 0",
                params![owner.as_slice()],
                |row| row.get(0),
            )
            .unwrap_or(0);
        OverviewData {
            by_layer,
            recent_by_layer,
            last_memory_at: last_memory_at as u64,
        }
    }
}

// ============================================
// impl MemoryStorage — v2.3.0 Remote Storage
// ============================================

impl MemoryStorage {
    pub async fn count_distinct_owners(&self) -> usize {
        let conn = self.conn.lock().await;
        conn.query_row("SELECT COUNT(DISTINCT owner) FROM records", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap_or(0) as usize
    }

    pub async fn owner_exists(&self, owner: &[u8; 32]) -> bool {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM records WHERE owner = ?1 LIMIT 1)",
            params![owner.as_slice()],
            |row| row.get::<_, bool>(0),
        )
        .unwrap_or(false)
    }
}

// ============================================
// impl MemoryStorage — v2.5.3+Isolation: Context Filter
// ============================================

impl MemoryStorage {
    /// Get active records filtered by project_id (context isolation).
    ///
    /// Matches records via TWO paths:
    /// 1. `records.project_id = project_id` — direct tag (set by future /remember context param)
    /// 2. `records.session_id → sessions.project_id = project_id` — via session association
    ///    (set by /log handler when `context` field is provided, stored by upsert_session)
    ///
    /// Records inserted via `/remember` directly without a session use path 1 only.
    /// Records inserted via `/log` with `context` field use path 2.
    ///
    /// ## Usage
    /// Called by `recall_handler.rs` Step 4.1 when `RecallRequest.context` is
    /// a non-"all" value. The caller builds a HashSet of returned record_ids
    /// and filters `scored` with `retain()`.
    ///
    /// ## Performance
    /// LEFT JOIN on session_id. For best performance, ensure an index exists on
    /// `records.session_id` and `sessions.project_id`. With typical MemChain
    /// data sizes (< 100k records per user) this is fast without extra indexes.
    ///
    /// v2.5.3+Isolation
    pub async fn get_active_records_by_context(
        &self,
        owner: &[u8; 32],
        project_id: &str,
        layer: Option<MemoryLayer>,
        limit: usize,
    ) -> Vec<MemoryRecord> {
        let limit = limit.min(1000).max(1);
        let conn = self.conn.lock().await;

        if let Some(l) = layer {
            self.query_rows(
                &conn,
                "SELECT r.record_id, r.owner, r.timestamp, r.layer, r.topic_tags, r.source_ai,
                        r.status, r.supersedes, r.encrypted_content, r.embedding, r.signature,
                        r.access_count, r.positive_feedback, r.negative_feedback, r.conflict_with,
                        r.blind
                 FROM records r
                 LEFT JOIN sessions s ON r.session_id = s.session_id
                 WHERE r.owner = ?1
                   AND r.status = 0
                   AND r.layer = ?2
                   AND (r.project_id = ?3 OR s.project_id = ?3)
                 ORDER BY r.timestamp DESC
                 LIMIT ?4",
                params![owner.as_slice(), l as u8 as i64, project_id, limit as i64],
            )
        } else {
            self.query_rows(
                &conn,
                "SELECT r.record_id, r.owner, r.timestamp, r.layer, r.topic_tags, r.source_ai,
                        r.status, r.supersedes, r.encrypted_content, r.embedding, r.signature,
                        r.access_count, r.positive_feedback, r.negative_feedback, r.conflict_with,
                        r.blind
                 FROM records r
                 LEFT JOIN sessions s ON r.session_id = s.session_id
                 WHERE r.owner = ?1
                   AND r.status = 0
                   AND (r.project_id = ?2 OR s.project_id = ?2)
                 ORDER BY r.timestamp DESC
                 LIMIT ?3",
                params![owner.as_slice(), project_id, limit as i64],
            )
        }
    }
}

// ============================================
// Tests
// ============================================
// v2.4.0+Search: Cognitive graph tests moved to storage_graph.rs.
//   Miner step support tests moved to storage_miner.rs.

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::crypto::IdentityKeyPair;
    use aeronyx_core::ledger::MemoryRecord;

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

    #[tokio::test]
    async fn test_has_active_content() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xBB; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Episode);
        s.insert(&r, "m").await;
        assert!(s.has_active_content(&owner, &r.encrypted_content).await);
        assert!(!s.has_active_content(&owner, b"nonexistent").await);
        assert!(
            !s.has_active_content(&[0xCC; 32], &r.encrypted_content)
                .await
        );
    }

    #[tokio::test]
    async fn test_insert_raw_log_plaintext() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let log_id = s
            .insert_raw_log("s1", 0, "user", "hello", "test", None, 1, None, None)
            .await
            .unwrap();
        assert!(log_id > 0);
        let content = s.read_rawlog_content(log_id, None).await.unwrap();
        assert_eq!(content, "hello");
    }

    #[tokio::test]
    async fn test_insert_raw_log_encrypted() {
        use super::super::storage_crypto::derive_rawlog_key;
        let rlk = derive_rawlog_key(&[0x42; 32]);
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let log_id = s
            .insert_raw_log("s1", 0, "user", "secret", "test", None, 1, None, Some(&rlk))
            .await
            .unwrap();
        let content = s.read_rawlog_content(log_id, Some(&rlk)).await.unwrap();
        assert_eq!(content, "secret");
        assert!(s.read_rawlog_content(log_id, None).await.is_none());
    }

    #[tokio::test]
    async fn test_feedback_operations() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let r = make_rec_owner(100, owner, MemoryLayer::Episode);
        let rid = r.record_id;
        s.insert(&r, "m").await;
        s.increment_positive_feedback(&rid).await;
        s.increment_negative_feedback(&rid).await;
        s.increment_negative_feedback(&rid).await;
        let got = s.get(&rid).await.unwrap();
        assert_eq!(got.positive_feedback, 1);
        assert_eq!(got.negative_feedback, 2);
    }

    #[tokio::test]
    async fn test_stats() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m")
            .await;
        s.insert(&make_rec_owner(200, owner, MemoryLayer::Identity), "m")
            .await;
        let stats = s.stats().await;
        assert_eq!(stats.total_records, 2);
        assert_eq!(stats.active_records, 2);
        assert_eq!(stats.by_layer.episode, 1);
        assert_eq!(stats.by_layer.identity, 1);
    }

    #[tokio::test]
    async fn test_get_embedding_model() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let r = make_rec_owner(100, [0xAA; 32], MemoryLayer::Episode);
        let rid = r.record_id;
        s.insert(&r, "minilm-l6-v2").await;
        assert_eq!(
            s.get_embedding_model(&rid).await,
            Some("minilm-l6-v2".into())
        );
    }

    #[tokio::test]
    async fn test_get_overview() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m")
            .await;
        s.insert(&make_rec_owner(200, owner, MemoryLayer::Identity), "m")
            .await;
        s.insert(&make_rec_owner(300, owner, MemoryLayer::Knowledge), "m")
            .await;
        let ov = s.get_overview(&owner, 20).await;
        assert_eq!(*ov.by_layer.get("episode").unwrap(), 1);
        assert_eq!(*ov.by_layer.get("identity").unwrap(), 1);
        assert_eq!(*ov.by_layer.get("knowledge").unwrap(), 1);
        assert_eq!(ov.last_memory_at, 300);
    }

    #[tokio::test]
    async fn test_count_distinct_owners_empty() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        assert_eq!(s.count_distinct_owners().await, 0);
    }

    #[tokio::test]
    async fn test_count_distinct_owners_multiple() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert(&make_rec_owner(100, [0xAA; 32], MemoryLayer::Episode), "m")
            .await;
        s.insert(&make_rec_owner(200, [0xBB; 32], MemoryLayer::Episode), "m")
            .await;
        s.insert(&make_rec_owner(300, [0xCC; 32], MemoryLayer::Episode), "m")
            .await;
        s.insert(&make_rec_owner(400, [0xAA; 32], MemoryLayer::Identity), "m")
            .await;
        assert_eq!(s.count_distinct_owners().await, 3);
    }

    #[tokio::test]
    async fn test_owner_exists_after_insert() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        s.insert(&make_rec_owner(100, owner, MemoryLayer::Episode), "m")
            .await;
        assert!(s.owner_exists(&owner).await);
        assert!(!s.owner_exists(&[0xBB; 32]).await);
    }

    // ── v2.5.3+Isolation: get_active_records_by_context tests ──

    #[tokio::test]
    async fn test_get_active_records_by_context_via_session() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner = [0xAA; 32];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Register a session with project_id = "work"
        s.upsert_session("sess_work", &owner, Some("work"), "chat", now, 2)
            .await
            .unwrap();

        // Insert two records linked to the work session
        let mut r1 = make_rec_owner(100, owner, MemoryLayer::Episode);
        // Manually set session_id on the record row after insert
        s.insert(&r1, "m").await;
        {
            let conn = s.conn_lock().await;
            let _ = conn.execute(
                "UPDATE records SET session_id = 'sess_work' WHERE record_id = ?1",
                params![r1.record_id.as_slice()],
            );
        }

        // Insert a record with no session (personal — untagged)
        let r2 = make_rec_owner(200, owner, MemoryLayer::Episode);
        s.insert(&r2, "m").await;

        // Query context = "work" should return only r1
        let results = s
            .get_active_records_by_context(&owner, "work", None, 100)
            .await;
        assert_eq!(
            results.len(),
            1,
            "Only the work-tagged record should be returned"
        );
        assert_eq!(results[0].record_id, r1.record_id);

        // Query context = "personal" should return nothing (no records tagged personal)
        let personal = s
            .get_active_records_by_context(&owner, "personal", None, 100)
            .await;
        assert!(personal.is_empty(), "No personal records exist");
    }

    #[tokio::test]
    async fn test_get_active_records_by_context_no_cross_owner() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let owner_a = [0xAA; 32];
        let owner_b = [0xBB; 32];
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        s.upsert_session("sess_a", &owner_a, Some("work"), "chat", now, 1)
            .await
            .unwrap();

        let r = make_rec_owner(100, owner_a, MemoryLayer::Episode);
        s.insert(&r, "m").await;
        {
            let conn = s.conn_lock().await;
            let _ = conn.execute(
                "UPDATE records SET session_id = 'sess_a' WHERE record_id = ?1",
                params![r.record_id.as_slice()],
            );
        }

        // owner_b querying "work" must see nothing (different owner)
        let results = s
            .get_active_records_by_context(&owner_b, "work", None, 100)
            .await;
        assert!(results.is_empty(), "Cross-owner isolation must be enforced");
    }

    #[tokio::test]
    async fn test_commitment_chain_append_range_and_status() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        assert_eq!(
            storage.audit_record_commitment_chain().await.unwrap(),
            RecordCommitmentChainAudit {
                block_count: 0,
                commitment_count: 0,
                tip_height: 0,
            }
        );
        let empty_integrity = storage.record_commitment_chain_integrity_status();
        assert_eq!(empty_integrity.state, "verified");
        assert_eq!(empty_integrity.verified_block_count, 0);
        assert_eq!(empty_integrity.verified_commitment_count, 0);
        assert_eq!(empty_integrity.verified_tip_height, 0);
        assert!(empty_integrity.baseline_verified_at.is_some());
        assert!(empty_integrity.verification_duration_ms.is_some());
        let identity = IdentityKeyPair::generate();
        let first = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_200_001,
            GENESIS_PREV_HASH,
            vec![[0x01; 32], [0x02; 32]],
            &identity,
        );
        assert_eq!(
            storage
                .append_record_commitment_block(&first, None)
                .await
                .unwrap(),
            RecordCommitmentAppendOutcome::Inserted
        );
        assert_eq!(
            storage
                .append_record_commitment_block(&first, None)
                .await
                .unwrap(),
            RecordCommitmentAppendOutcome::AlreadyPresent
        );

        let second = RecordCommitmentBlockV1::new_signed(
            2,
            1_700_200_002,
            first.hash(),
            vec![[0x03; 32]],
            &identity,
        );
        storage
            .append_record_commitment_block(&second, Some(&identity.public_key_bytes()))
            .await
            .unwrap();

        let range = storage
            .get_record_commitment_block_range(1, 10)
            .await
            .unwrap();
        assert_eq!(range, vec![first.clone(), second.clone()]);
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (2, second.hash())
        );
        assert_eq!(storage.last_block_height().await, 2);
        assert_eq!(storage.last_block_hash().await, second.hash());

        let status = storage.record_commitment_chain_status().await;
        assert_eq!(status.block_count, 2);
        assert_eq!(status.commitment_count, 3);
        assert_eq!(status.tip_height, 2);
        assert_eq!(status.tip_hash, Some(hex::encode(second.hash())));
        assert_eq!(status.integrity.state, "verified");
        assert_eq!(status.integrity.verified_block_count, 2);
        assert_eq!(status.integrity.verified_commitment_count, 3);
        assert_eq!(status.integrity.verified_tip_height, 2);
        assert!(status.integrity.last_verified_at >= status.integrity.baseline_verified_at);
        assert_eq!(
            storage.audit_record_commitment_chain().await.unwrap(),
            RecordCommitmentChainAudit {
                block_count: 2,
                commitment_count: 3,
                tip_height: 2,
            }
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

    #[tokio::test]
    async fn test_commitment_chain_audit_rejects_payload_corruption() {
        let (storage, _) = commitment_audit_fixture().await;
        assert_eq!(
            storage.record_commitment_chain_integrity_status().state,
            "verified"
        );
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_commitment_blocks SET payload=?1 WHERE height=1",
                params![vec![0xFFu8]],
            )
            .unwrap();
        }
        assert!(storage
            .audit_record_commitment_chain()
            .await
            .unwrap_err()
            .contains("payload decode failed"));
        let integrity = storage.record_commitment_chain_integrity_status();
        assert_eq!(integrity.state, "not_verified");
        assert_eq!(integrity.verified_block_count, 0);
        assert_eq!(integrity.verified_commitment_count, 0);
        assert_eq!(integrity.verified_tip_height, 0);
    }

    #[tokio::test]
    async fn test_commitment_chain_audit_bounds_persisted_payload_before_decode() {
        let (storage, _) = commitment_audit_fixture().await;
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_commitment_blocks SET payload=?1 WHERE height=1",
                params![vec![0u8; MAX_STORED_COMMITMENT_BLOCK_BYTES + 1]],
            )
            .unwrap();
        }
        assert!(storage
            .audit_record_commitment_chain()
            .await
            .unwrap_err()
            .contains("payload exceeds bound"));
    }

    #[tokio::test]
    async fn test_commitment_chain_audit_rejects_denormalized_row_tampering() {
        let (storage, _) = commitment_audit_fixture().await;
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_commitment_blocks SET merkle_root=?1 WHERE height=1",
                params![[0xEEu8; 32].as_slice()],
            )
            .unwrap();
        }
        assert!(storage
            .audit_record_commitment_chain()
            .await
            .unwrap_err()
            .contains("stored row mismatch"));
    }

    #[tokio::test]
    async fn test_commitment_chain_audit_rejects_membership_index_tampering() {
        let (storage, block) = commitment_audit_fixture().await;
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "DELETE FROM record_block_commitments WHERE record_id=?1",
                params![block.record_ids[0].as_slice()],
            )
            .unwrap();
        }
        assert!(storage
            .audit_record_commitment_chain()
            .await
            .unwrap_err()
            .contains("membership index mismatch"));
    }

    #[tokio::test]
    async fn test_commitment_chain_audit_reverifies_persisted_signature() {
        let (storage, mut block) = commitment_audit_fixture().await;
        block.proposer_signature[0] ^= 0x01;
        let payload = bincode::serialize(&block).unwrap();
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_commitment_blocks
                 SET proposer_signature=?1,payload=?2 WHERE height=1",
                params![block.proposer_signature.as_slice(), payload],
            )
            .unwrap();
        }
        assert!(storage
            .audit_record_commitment_chain()
            .await
            .unwrap_err()
            .contains("proposer signature is invalid"));
    }

    #[tokio::test]
    async fn test_commitment_chain_append_rejects_sqlite_integer_overflow() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let identity = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            u64::MAX,
            GENESIS_PREV_HASH,
            vec![[0x55; 32]],
            &identity,
        );
        assert!(storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap_err()
            .contains("timestamp exceeds SQLite range"));
    }

    #[tokio::test]
    async fn test_commitment_chain_rejects_fork_and_rolls_back_duplicate() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let identity = IdentityKeyPair::generate();
        let first = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_300_001,
            GENESIS_PREV_HASH,
            vec![[0x11; 32]],
            &identity,
        );
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();

        let fork = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_300_002,
            GENESIS_PREV_HASH,
            vec![[0x22; 32]],
            &identity,
        );
        assert!(storage
            .append_record_commitment_block(&fork, None)
            .await
            .unwrap_err()
            .contains("fork"));

        let duplicate = RecordCommitmentBlockV1::new_signed(
            2,
            1_700_300_003,
            first.hash(),
            vec![[0x11; 32]],
            &identity,
        );
        assert!(storage
            .append_record_commitment_block(&duplicate, None)
            .await
            .is_err());
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (1, first.hash())
        );
        assert_eq!(
            storage
                .get_record_commitment_block_range(1, 10)
                .await
                .unwrap(),
            vec![first]
        );
    }

    #[test]
    fn test_commitment_sync_runtime_tracks_failure_recovery_and_bounds_events() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let initial = storage.record_commitment_sync_status();
        assert_eq!(initial.role, "verifier");
        assert_eq!(initial.state, "disabled");
        assert!(!initial.enabled);

        storage.configure_record_commitment_sync(false, true);
        storage.record_commitment_sync_attempt(100);
        storage.record_commitment_sync_failure(
            101,
            "request timeout https://private.example",
            1,
            131,
        );
        let failed = storage.record_commitment_sync_status();
        assert_eq!(failed.role, "follower");
        assert_eq!(failed.state, "backoff");
        assert_eq!(failed.last_attempt_at, Some(100));
        assert_eq!(failed.last_failure_at, Some(101));
        assert_eq!(failed.next_poll_at, Some(131));
        assert_eq!(
            failed.last_error_code.as_deref(),
            Some("internal_sync_error")
        );
        assert_eq!(failed.recent_events.len(), 1);
        assert_eq!(failed.recent_events[0].kind, "failure");

        storage.record_commitment_sync_attempt(132);
        storage.record_commitment_sync_page_success(133, 3, 9, false);
        storage.schedule_next_commitment_sync_poll(163);
        let recovered = storage.record_commitment_sync_status();
        assert_eq!(recovered.state, "current");
        assert_eq!(recovered.last_success_at, Some(133));
        assert_eq!(recovered.last_recovered_at, Some(133));
        assert_eq!(recovered.remote_tip_height, Some(9));
        assert_eq!(recovered.pages_received_total, 1);
        assert_eq!(recovered.blocks_received_total, 3);
        assert_eq!(recovered.consecutive_failures, 0);
        assert_eq!(recovered.next_poll_at, Some(163));
        assert_eq!(recovered.failure_events_total, 1);
        assert_eq!(recovered.recovery_events_total, 1);
        assert_eq!(recovered.recent_events.len(), 2);
        assert_eq!(recovered.recent_events[1].kind, "recovered");

        for failure in 1..=20u32 {
            storage.record_commitment_sync_failure(
                200 + u64::from(failure),
                "request_timeout",
                failure,
                500 + u64::from(failure),
            );
        }
        let bounded = storage.record_commitment_sync_status();
        assert_eq!(bounded.recent_events.len(), COMMITMENT_SYNC_EVENT_CAPACITY);
        assert_eq!(bounded.recent_events.first().unwrap().sequence, 7);
        assert_eq!(bounded.recent_events.last().unwrap().sequence, 22);
        assert_eq!(bounded.failure_events_total, 21);

        storage.stop_record_commitment_sync();
        assert_eq!(storage.record_commitment_sync_status().state, "stopped");
    }

    #[test]
    fn test_commitment_sync_runtime_reports_coordinator_without_polling() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        storage.configure_record_commitment_sync(true, false);
        let status = storage.record_commitment_sync_status();
        assert_eq!(status.role, "coordinator");
        assert_eq!(status.state, "producing");
        assert!(!status.enabled);
        assert!(status.recent_events.is_empty());
    }

    #[test]
    fn test_commitment_sync_error_codes_are_explicitly_allow_listed() {
        assert_eq!(
            privacy_safe_sync_error_code("request_timeout"),
            "request_timeout"
        );
        assert_eq!(
            privacy_safe_sync_error_code("http_status_503"),
            "http_status_error"
        );
        assert_eq!(
            privacy_safe_sync_error_code("unknown_but_well_formed"),
            "internal_sync_error"
        );
        assert_eq!(
            privacy_safe_sync_error_code("http_status_50x"),
            "internal_sync_error"
        );
    }

    #[tokio::test]
    async fn test_block_packer_source_is_active_blind_and_uncommitted_only() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let owner = IdentityKeyPair::generate();
        let mut blind = make_rec_owner(
            1_700_400_001,
            owner.public_key_bytes(),
            MemoryLayer::Episode,
        );
        blind.blind = true;
        blind.signature = owner.sign(&blind.record_id);
        assert!(storage.insert_blind_replica(&blind, "sealed").await);

        let local = make_rec_owner(1_700_400_002, [0x55; 32], MemoryLayer::Episode);
        assert!(storage.insert(&local, "local").await);
        let pending = storage.get_uncommitted_blind_records(32).await;
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].record_id, blind.record_id);

        let proposer = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_400_003,
            GENESIS_PREV_HASH,
            vec![blind.record_id],
            &proposer,
        );
        storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap();
        assert!(storage.get_uncommitted_blind_records(32).await.is_empty());
    }
}
