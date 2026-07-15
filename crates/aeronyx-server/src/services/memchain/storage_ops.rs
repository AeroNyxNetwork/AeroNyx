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
//!   the new chain. It delegates to `append_record_commitment_blocks_atomic`;
//!   never update the tip independently of that transaction.
//! - Peer range reads return commitments only; full records remain owner-scoped.
//! - An inbound checkpoint request describes the requester's state, not this
//!   node's observation of the network. Serving it must never mutate outbound
//!   checkpoint relation, heights, or divergence counters.
//! - Freshness derives only from an audited, durable `last_evidence_at`; a
//!   failed attempt or served response must never make stale evidence fresh.
//! - Witness round state is process-local aggregate telemetry. Do not use its
//!   counts as votes, quorum, finality, leader election, or fork choice.
//! - A coordinator must call `configure_record_commitment_durability(true)`
//!   before startup audit. Failure to confirm FULL-or-stronger is fatal; this
//!   protects acknowledged commitment tips from ordinary host power loss.
//! - A coordinator must configure the signed tip anchor after the full startup
//!   audit and before mining. It detects an older/replaced SQLite chain while
//!   the host-side anchor remains, but not rollback of the entire host/disk.
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
//! v2.7.5-CheckpointProof - Added signed checkpoint reconciliation evidence.
//! v2.7.6-EvidenceVault - Added durable bounded proof storage and startup audit.
//! v2.7.7-EvidenceRestartRecovery - Added real SQLite restart/migration tests.
//! v2.7.10-CheckpointDirectionIsolation - Isolated inbound service telemetry.
//! v2.7.11-CheckpointFreshness - Added durable proof age classification.
//! v2.7.12-WitnessRoundEvidence - Added bounded round evidence classification.
//! v2.7.13-CommitmentDurability - Enforced coordinator FULL SQLite commits.
//! v2.7.14-CommitmentTipAnchor - Added a signed local high-water rollback guard.
//! v2.7.17-AtomicBlockPage - Made verified peer pages one atomic SQLite append.
//!
//! ## Last Modified
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
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use sha2::{Digest, Sha256};
use tracing::{debug, error, info, warn};

use aeronyx_core::crypto::{IdentityKeyPair, IdentityPublicKey};
use aeronyx_core::ledger::{
    MemoryLayer, MemoryRecord, RecordCommitmentBlockV1, AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
    GENESIS_PREV_HASH,
};
use aeronyx_core::protocol::memchain::{
    decode_memchain, encode_memchain, record_chain_checkpoint_response_signing_bytes,
    MemChainMessage, MEMCHAIN_MAGIC,
};

use super::storage::{
    LayerCounts, MemoryStorage, RawLogRow, RecordCommitmentCheckpointStatus,
    RecordCommitmentIntegrityRuntime, RecordCommitmentSyncEvent, RecordCommitmentSyncRuntime,
    RecordCommitmentSyncStatus, RecordCommitmentTipAnchorConfig, StorageStats,
    CHECKPOINT_EVIDENCE_CAPACITY, CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
    COMMITMENT_SYNC_EVENT_CAPACITY, MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES,
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

/// Aggregate result of one atomic bounded commitment-block append.
///
/// Counts contain no block hashes, record commitments, owners, or peer
/// identities, so the result is safe for aggregate sync telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecordCommitmentBatchAppendOutcome {
    /// Blocks committed by this transaction.
    pub inserted: usize,
    /// Exact blocks that were already durable when the transaction began.
    pub already_present: usize,
}

#[derive(Debug)]
struct CommittedRecordCommitmentBatch {
    outcome: RecordCommitmentBatchAppendOutcome,
    inserted_indices: Vec<usize>,
    appended_at: u64,
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

/// Aggregate result of a complete local checkpoint-evidence vault audit.
///
/// Raw frames, peer identities, hashes, signatures, request IDs, and endpoints
/// are intentionally absent so this report is safe for startup logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordCommitmentCheckpointEvidenceAudit {
    pub evidence_records: u64,
    pub divergence_evidence_records: u64,
    pub last_evidence_at: Option<u64>,
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
/// Bound lock time, rollback journal growth, and caller-controlled allocation.
/// Peer protocol pages currently use 16 blocks; 32 leaves room for internal
/// maintenance without turning this storage primitive into an unbounded API.
const MAX_ATOMIC_COMMITMENT_BLOCK_BATCH: usize = 32;
/// A v1 tip anchor is under 1 KiB. Keep disk reads bounded before JSON decode.
const MAX_COMMITMENT_TIP_ANCHOR_BYTES: u64 = 4 * 1024;
const COMMITMENT_TIP_ANCHOR_CONTRACT: &str = "record_commitment_tip_anchor.v1";
const COMMITMENT_TIP_ANCHOR_DOMAIN: &[u8] = b"aeronyx.record_commitment_tip_anchor.v1\0";
static COMMITMENT_TIP_ANCHOR_TEMP_NONCE: AtomicU64 = AtomicU64::new(1);

fn persist_record_commitment_block_transaction(
    transaction: &rusqlite::Transaction<'_>,
    block: &RecordCommitmentBlockV1,
    received_from: Option<&[u8; 32]>,
    expected_height: u64,
    expected_prev_hash: &[u8; 32],
    created_at: i64,
) -> Result<RecordCommitmentAppendOutcome, String> {
    let block_height = i64::try_from(block.header.height)
        .map_err(|_| "commitment block height exceeds SQLite range".to_string())?;
    let block_timestamp = i64::try_from(block.header.timestamp)
        .map_err(|_| "commitment block timestamp exceeds SQLite range".to_string())?;
    let block_hash = block.hash();
    let existing_hash: Option<Vec<u8>> = transaction
        .query_row(
            "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
            params![block_height],
            |row| row.get(0),
        )
        .optional()
        .map_err(|error| format!("read existing commitment block: {error}"))?;
    if let Some(existing_hash) = existing_hash {
        if existing_hash.as_slice() == block_hash.as_slice() {
            return Ok(RecordCommitmentAppendOutcome::AlreadyPresent);
        }
        return Err(format!(
            "commitment chain fork at height {}",
            block.header.height
        ));
    }

    block
        .verify(
            &AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            expected_height,
            expected_prev_hash,
        )
        .map_err(|error| format!("commitment block validation failed: {error}"))?;
    let payload = bincode::serialize(block)
        .map_err(|error| format!("serialize commitment block: {error}"))?;
    if payload.len() > MAX_STORED_COMMITMENT_BLOCK_BYTES {
        return Err("serialized commitment block exceeds storage limit".to_string());
    }

    transaction
        .execute(
            "INSERT INTO record_commitment_blocks
             (height,block_hash,chain_id,protocol_version,timestamp,
              prev_block_hash,merkle_root,record_count,proposer,
              proposer_signature,payload,received_from,created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13)",
            params![
                block_height,
                block_hash.as_slice(),
                block.header.chain_id.as_slice(),
                i64::from(block.header.protocol_version),
                block_timestamp,
                block.header.prev_block_hash.as_slice(),
                block.header.merkle_root.as_slice(),
                i64::from(block.header.record_count),
                block.header.proposer.as_slice(),
                block.proposer_signature.as_slice(),
                payload,
                received_from.map(<[u8; 32]>::as_slice),
                created_at,
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
    Ok(RecordCommitmentAppendOutcome::Inserted)
}

fn persist_record_commitment_tip_transaction(
    transaction: &rusqlite::Transaction<'_>,
    height: u64,
    hash: &[u8; 32],
) -> Result<(), String> {
    for (key, value) in [
        ("record_block_tip_hash", hash.to_vec()),
        ("record_block_tip_height", height.to_le_bytes().to_vec()),
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
    Ok(())
}

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

fn record_commitment_tip_anchor_signing_bytes(
    chain_id: &[u8; 32],
    tip_height: u64,
    tip_hash: &[u8; 32],
    signer: &[u8; 32],
    updated_at: u64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(COMMITMENT_TIP_ANCHOR_DOMAIN.len() + 112);
    bytes.extend_from_slice(COMMITMENT_TIP_ANCHOR_DOMAIN);
    bytes.extend_from_slice(chain_id);
    bytes.extend_from_slice(&tip_height.to_le_bytes());
    bytes.extend_from_slice(tip_hash);
    bytes.extend_from_slice(signer);
    bytes.extend_from_slice(&updated_at.to_le_bytes());
    bytes
}

fn decode_fixed_hex<const N: usize>(value: &str, label: &str) -> Result<[u8; N], String> {
    let decoded = hex::decode(value)
        .map_err(|_| format!("commitment tip anchor {label} is not hexadecimal"))?;
    decoded.try_into().map_err(|decoded: Vec<u8>| {
        format!(
            "commitment tip anchor {label} has invalid length {}",
            decoded.len()
        )
    })
}

fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

async fn read_record_commitment_tip_anchor(
    path: &Path,
) -> Result<Option<RecordCommitmentTipAnchorV1>, String> {
    let metadata = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("inspect commitment tip anchor: {error}")),
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("commitment tip anchor must be a regular file".to_string());
    }
    if metadata.len() > MAX_COMMITMENT_TIP_ANCHOR_BYTES {
        return Err("commitment tip anchor exceeds the defensive size bound".to_string());
    }
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|error| format!("read commitment tip anchor: {error}"))?;
    if bytes.len() as u64 > MAX_COMMITMENT_TIP_ANCHOR_BYTES {
        return Err("commitment tip anchor exceeds the defensive size bound".to_string());
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|error| format!("decode commitment tip anchor: {error}"))
}

fn write_record_commitment_tip_anchor_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create commitment tip anchor directory: {error}"))?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err("commitment tip anchor target must be a regular file".to_string());
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect commitment tip anchor target: {error}")),
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "commitment tip anchor path has no valid file name".to_string())?;
    let nonce = COMMITMENT_TIP_ANCHOR_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let time_nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp_path = parent.join(format!(
        ".{file_name}.{}.{}.{}.tmp",
        std::process::id(),
        nonce,
        time_nonce
    ));

    let result = (|| -> Result<(), String> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(&temp_path)
            .map_err(|error| format!("create commitment tip anchor temp file: {error}"))?;
        file.write_all(bytes)
            .map_err(|error| format!("write commitment tip anchor: {error}"))?;
        file.flush()
            .map_err(|error| format!("flush commitment tip anchor: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync commitment tip anchor: {error}"))?;
        drop(file);
        std::fs::rename(&temp_path, path)
            .map_err(|error| format!("replace commitment tip anchor: {error}"))?;
        #[cfg(unix)]
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("sync commitment tip anchor directory: {error}"))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

async fn persist_record_commitment_tip_anchor(
    path: PathBuf,
    tip_height: u64,
    tip_hash: [u8; 32],
    identity: &IdentityKeyPair,
) -> Result<u64, String> {
    let updated_at = unix_now_secs();
    let anchor =
        RecordCommitmentTipAnchorV1::new_signed(tip_height, tip_hash, identity, updated_at);
    let bytes = serde_json::to_vec(&anchor)
        .map_err(|error| format!("encode commitment tip anchor: {error}"))?;
    tokio::task::spawn_blocking(move || write_record_commitment_tip_anchor_atomic(&path, &bytes))
        .await
        .map_err(|error| format!("join commitment tip anchor write: {error}"))??;
    Ok(updated_at)
}

fn audit_checkpoint_evidence_connection(
    conn: &mut rusqlite::Connection,
) -> Result<RecordCommitmentCheckpointEvidenceAudit, String> {
    let transaction = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)
        .map_err(|error| format!("begin checkpoint evidence audit snapshot: {error}"))?;
    let current_tip_height_i64 = transaction
        .query_row(
            "SELECT COALESCE(MAX(height), 0) FROM record_commitment_blocks",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|error| format!("read checkpoint evidence local tip: {error}"))?;
    let current_tip_height = u64::try_from(current_tip_height_i64)
        .map_err(|_| "checkpoint evidence local tip is invalid".to_string())?;

    let mut evidence_records = 0u64;
    let mut divergence_evidence_records = 0u64;
    let mut last_evidence_at = None;
    {
        let mut statement = transaction
            .prepare(
                "SELECT evidence_digest,observed_at,relation,local_tip_height,
                        remote_tip_height,checkpoint_height,signed_response
                 FROM record_checkpoint_evidence
                 ORDER BY observed_at ASC,evidence_digest ASC",
            )
            .map_err(|error| format!("prepare checkpoint evidence audit: {error}"))?;
        let mut rows = statement
            .query([])
            .map_err(|error| format!("query checkpoint evidence audit: {error}"))?;
        while let Some(row) = rows
            .next()
            .map_err(|error| format!("read checkpoint evidence row: {error}"))?
        {
            let digest: Vec<u8> = row
                .get(0)
                .map_err(|error| format!("read checkpoint evidence digest: {error}"))?;
            let observed_at_i64: i64 = row
                .get(1)
                .map_err(|error| format!("read checkpoint evidence time: {error}"))?;
            let relation: String = row
                .get(2)
                .map_err(|error| format!("read checkpoint evidence relation: {error}"))?;
            let local_tip_i64: i64 = row
                .get(3)
                .map_err(|error| format!("read checkpoint evidence local height: {error}"))?;
            let remote_tip_i64: i64 = row
                .get(4)
                .map_err(|error| format!("read checkpoint evidence remote height: {error}"))?;
            let stored_checkpoint_i64: i64 = row
                .get(5)
                .map_err(|error| format!("read checkpoint evidence height: {error}"))?;
            let frame: Vec<u8> = row
                .get(6)
                .map_err(|error| format!("read checkpoint evidence frame: {error}"))?;

            let observed_at = u64::try_from(observed_at_i64)
                .map_err(|_| "checkpoint evidence time is invalid".to_string())?;
            let local_tip_height = u64::try_from(local_tip_i64)
                .map_err(|_| "checkpoint evidence local height is invalid".to_string())?;
            let stored_remote_tip = u64::try_from(remote_tip_i64)
                .map_err(|_| "checkpoint evidence remote height is invalid".to_string())?;
            let stored_checkpoint_height = u64::try_from(stored_checkpoint_i64)
                .map_err(|_| "checkpoint evidence height is invalid".to_string())?;
            if local_tip_height > current_tip_height {
                return Err("checkpoint evidence references an unavailable local tip".to_string());
            }
            if digest.len() != 32 || Sha256::digest(&frame).as_slice() != digest.as_slice() {
                return Err("checkpoint evidence digest mismatch".to_string());
            }
            if frame.is_empty()
                || frame.len() > MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES
                || frame.first().copied() != Some(MEMCHAIN_MAGIC)
            {
                return Err("checkpoint evidence frame violates bounds".to_string());
            }
            let message = decode_memchain(&frame[1..])
                .map_err(|_| "checkpoint evidence frame decode failed".to_string())?;
            let canonical = encode_memchain(&message)
                .map_err(|_| "checkpoint evidence canonical encode failed".to_string())?;
            if canonical != frame {
                return Err("checkpoint evidence frame is non-canonical".to_string());
            }
            let MemChainMessage::RecordChainCheckpointResponseV1 {
                chain_id,
                request_id,
                responder,
                response_timestamp,
                checkpoint_height,
                checkpoint_hash,
                tip_height,
                tip_hash,
                signature,
            } = message
            else {
                return Err("checkpoint evidence contains an unexpected frame".to_string());
            };
            if chain_id != AERONYX_MEMCHAIN_MAINNET_CHAIN_ID {
                return Err("checkpoint evidence chain mismatch".to_string());
            }
            if observed_at.abs_diff(response_timestamp) > 60 {
                return Err("checkpoint evidence observation time mismatch".to_string());
            }
            let signing_bytes = record_chain_checkpoint_response_signing_bytes(
                &chain_id,
                &request_id,
                &responder,
                response_timestamp,
                checkpoint_height,
                &checkpoint_hash,
                tip_height,
                &tip_hash,
            );
            IdentityPublicKey::from_bytes(&responder)
                .and_then(|key| key.verify(&signing_bytes, &signature))
                .map_err(|_| "checkpoint evidence signature is invalid".to_string())?;
            if tip_height == 0 && tip_hash != GENESIS_PREV_HASH {
                return Err("checkpoint evidence genesis tip is invalid".to_string());
            }
            if tip_height != stored_remote_tip
                || checkpoint_height != stored_checkpoint_height
                || checkpoint_height != local_tip_height.min(tip_height)
            {
                return Err("checkpoint evidence height relation mismatch".to_string());
            }
            if checkpoint_height == tip_height && checkpoint_hash != tip_hash {
                return Err("checkpoint evidence tip hash is inconsistent".to_string());
            }
            let local_checkpoint_hash: Vec<u8> = if checkpoint_height == 0 {
                GENESIS_PREV_HASH.to_vec()
            } else {
                transaction
                    .query_row(
                        "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                        params![i64::try_from(checkpoint_height).map_err(|_| {
                            "checkpoint evidence height exceeds SQLite range".to_string()
                        })?],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|error| format!("read checkpoint evidence local block: {error}"))?
                    .ok_or_else(|| "checkpoint evidence local block is unavailable".to_string())?
            };
            let expected_relation = if local_checkpoint_hash.as_slice() != checkpoint_hash {
                "diverged"
            } else if local_tip_height == tip_height {
                "converged"
            } else if local_tip_height < tip_height {
                "remote_ahead"
            } else {
                "remote_behind"
            };
            if relation != expected_relation {
                return Err("checkpoint evidence classification mismatch".to_string());
            }

            evidence_records = evidence_records.saturating_add(1);
            if relation == "diverged" {
                divergence_evidence_records = divergence_evidence_records.saturating_add(1);
            }
            last_evidence_at =
                Some(last_evidence_at.map_or(observed_at, |last: u64| last.max(observed_at)));
        }
    }
    if evidence_records > CHECKPOINT_EVIDENCE_CAPACITY as u64 {
        return Err("checkpoint evidence vault exceeds its configured capacity".to_string());
    }
    transaction
        .commit()
        .map_err(|error| format!("finish checkpoint evidence audit snapshot: {error}"))?;
    Ok(RecordCommitmentCheckpointEvidenceAudit {
        evidence_records,
        divergence_evidence_records,
        last_evidence_at,
    })
}

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
        | "storage_append_rejected"
        | "checkpoint_request_timeout"
        | "checkpoint_request_connect"
        | "checkpoint_request_body"
        | "checkpoint_request_decode"
        | "checkpoint_request_request"
        | "checkpoint_request_unknown"
        | "local_checkpoint_unavailable"
        | "local_checkpoint_tip_mismatch"
        | "invalid_checkpoint_frame"
        | "unexpected_checkpoint_message"
        | "checkpoint_chain_mismatch"
        | "checkpoint_request_mismatch"
        | "checkpoint_responder_mismatch"
        | "stale_checkpoint_response"
        | "invalid_checkpoint_signature"
        | "invalid_checkpoint_genesis"
        | "checkpoint_height_mismatch"
        | "checkpoint_tip_inconsistent"
        | "local_checkpoint_height_mismatch"
        | "checkpoint_evidence_persist_failed"
        | "signed_checkpoint_remote_behind"
        | "signed_checkpoint_divergence" => reason.to_string(),
        // Exact status codes are useful in process logs but add needless
        // cardinality to public health data, so all 3-digit responses collapse
        // to one stable evidence code.
        _ if reason.strip_prefix("http_status_").is_some_and(|status| {
            status.len() == 3 && status.bytes().all(|byte| byte.is_ascii_digit())
        }) =>
        {
            "http_status_error".to_string()
        }
        _ if reason
            .strip_prefix("checkpoint_http_status_")
            .is_some_and(|status| {
                status.len() == 3 && status.bytes().all(|byte| byte.is_ascii_digit())
            }) =>
        {
            "checkpoint_http_status_error".to_string()
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

fn checkpoint_observation_freshness(
    last_evidence_at: Option<u64>,
    now: u64,
) -> (&'static str, Option<u64>) {
    let Some(observed_at) = last_evidence_at else {
        return ("unavailable", None);
    };
    let Some(age) = now.checked_sub(observed_at) else {
        // A wall-clock rollback must not make future-dated evidence appear
        // fresh. The signed frame remains in the audited local vault.
        return ("unavailable", None);
    };
    if age <= CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS {
        ("fresh", Some(age))
    } else {
        ("stale", Some(age))
    }
}

/// Classifies one bounded coordinator witness round without implying consensus.
///
/// `attention` is reserved for signed evidence that the coordinator may be
/// behind or that a shared prefix diverged. `shared_prefix` means every
/// attempted witness produced valid evidence compatible with the local chain;
/// it is deliberately not named quorum, finality, or consensus.
fn checkpoint_witness_round_state(
    attempted: usize,
    verified: usize,
    failed: usize,
    remote_ahead: usize,
    diverged: usize,
) -> &'static str {
    if attempted == 0 {
        "unavailable"
    } else if verified == 0 {
        "unverified"
    } else if remote_ahead > 0 || diverged > 0 {
        "attention"
    } else if failed > 0 || verified < attempted {
        "partial"
    } else {
        "shared_prefix"
    }
}

impl MemoryStorage {
    /// Returns aggregate signed checkpoint evidence without peer, hash,
    /// signature, endpoint, or user metadata.
    pub fn record_commitment_checkpoint_status(&self) -> RecordCommitmentCheckpointStatus {
        let runtime = self.commitment_checkpoint.read();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        let (observation_freshness, observation_age_seconds) =
            if runtime.evidence_state == "verified" {
                checkpoint_observation_freshness(runtime.last_evidence_at, now)
            } else {
                ("unavailable", None)
            };
        RecordCommitmentCheckpointStatus {
            contract_version: "record_commitment_checkpoint.v1",
            state: runtime.state.to_string(),
            last_checked_at: runtime.last_checked_at,
            last_converged_at: runtime.last_converged_at,
            last_divergence_at: runtime.last_divergence_at,
            last_failure_at: runtime.last_failure_at,
            last_served_at: runtime.last_served_at,
            local_tip_height: runtime.local_tip_height,
            remote_tip_height: runtime.remote_tip_height,
            proofs_verified_total: runtime.proofs_verified_total,
            proofs_failed_total: runtime.proofs_failed_total,
            divergences_total: runtime.divergences_total,
            requests_served_total: runtime.requests_served_total,
            evidence_state: runtime.evidence_state.to_string(),
            evidence_records: runtime.evidence_records,
            divergence_evidence_records: runtime.divergence_evidence_records,
            last_evidence_at: runtime.last_evidence_at,
            observation_freshness: observation_freshness.to_string(),
            observation_age_seconds,
            freshness_window_seconds: CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
            last_round_state: runtime.last_round_state.to_string(),
            last_round_at: runtime.last_round_at,
            last_round_eligible: runtime.last_round_eligible,
            last_round_attempted: runtime.last_round_attempted,
            last_round_verified: runtime.last_round_verified,
            last_round_failed: runtime.last_round_failed,
            last_round_converged: runtime.last_round_converged,
            last_round_remote_ahead: runtime.last_round_remote_ahead,
            last_round_remote_behind: runtime.last_round_remote_behind,
            last_round_diverged: runtime.last_round_diverged,
            evidence_persistence_failures_total: runtime
                .evidence_persistence_failures_total,
            privacy_policy: "aggregate signed checkpoint outcomes, bounded witness-round coverage, durable observation freshness, and local evidence-vault health only; counts are not consensus; raw frames, peer identities, block hashes, signatures, request ids, commitments, owners, payloads, endpoints, routes, and client metadata never leave the node",
        }
    }

    /// Records the aggregate result of one completed bounded witness round.
    ///
    /// This method intentionally stores no peer identity, endpoint, hash,
    /// signature, or request id. It cannot alter the canonical commitment
    /// chain and its counts must never be interpreted as consensus.
    #[allow(clippy::too_many_arguments)]
    pub fn record_commitment_checkpoint_witness_round(
        &self,
        now: u64,
        eligible: usize,
        attempted: usize,
        verified: usize,
        failed: usize,
        converged: usize,
        remote_ahead: usize,
        remote_behind: usize,
        diverged: usize,
    ) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.last_round_state =
            checkpoint_witness_round_state(attempted, verified, failed, remote_ahead, diverged);
        runtime.last_round_at = Some(now);
        runtime.last_round_eligible = eligible;
        runtime.last_round_attempted = attempted;
        runtime.last_round_verified = verified;
        runtime.last_round_failed = failed;
        runtime.last_round_converged = converged;
        runtime.last_round_remote_ahead = remote_ahead;
        runtime.last_round_remote_behind = remote_behind;
        runtime.last_round_diverged = diverged;
    }

    /// Records one outbound, signature-verified checkpoint comparison.
    pub fn record_commitment_checkpoint_verified(
        &self,
        now: u64,
        relation: &'static str,
        local_tip_height: u64,
        remote_tip_height: u64,
    ) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.state = relation;
        runtime.last_checked_at = Some(now);
        runtime.local_tip_height = Some(local_tip_height);
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.proofs_verified_total = runtime.proofs_verified_total.saturating_add(1);
        if relation == "converged" {
            runtime.last_converged_at = Some(now);
        } else if relation == "diverged" {
            runtime.last_divergence_at = Some(now);
            runtime.divergences_total = runtime.divergences_total.saturating_add(1);
        }
    }

    /// Records a checkpoint attempt that did not establish signed evidence.
    pub fn record_commitment_checkpoint_failure(&self, now: u64) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.state = "proof_failed";
        runtime.last_checked_at = Some(now);
        runtime.last_failure_at = Some(now);
        runtime.proofs_failed_total = runtime.proofs_failed_total.saturating_add(1);
    }

    /// Records an authenticated checkpoint response served to another node.
    ///
    /// The requester's claimed height and hash are untrusted inbound context.
    /// They may influence the signed response but must never overwrite this
    /// node's outbound convergence/divergence evidence or observed heights.
    pub fn record_commitment_checkpoint_served(&self, now: u64) {
        let mut runtime = self.commitment_checkpoint.write();
        runtime.last_served_at = Some(now);
        runtime.requests_served_total = runtime.requests_served_total.saturating_add(1);
    }

    /// Persists the exact response frame after the node-peer verifier has
    /// completed chain, freshness, identity, signature, and relation checks.
    ///
    /// This method independently rechecks the digest and storage bounds, writes
    /// in one immediate transaction, and prunes the oldest non-divergence proof
    /// first. A failure is returned to the follower so convergence cannot be
    /// declared without durable evidence.
    pub async fn persist_record_commitment_checkpoint_evidence(
        &self,
        observed_at: u64,
        relation: &str,
        local_tip_height: u64,
        remote_tip_height: u64,
        checkpoint_height: u64,
        evidence_digest: &[u8; 32],
        signed_response: &[u8],
    ) -> Result<(), String> {
        let failure = |message: String| {
            let mut runtime = self.commitment_checkpoint.write();
            runtime.evidence_persistence_failures_total = runtime
                .evidence_persistence_failures_total
                .saturating_add(1);
            Err(message)
        };
        if !matches!(
            relation,
            "converged" | "remote_ahead" | "remote_behind" | "diverged"
        ) {
            return failure("checkpoint evidence relation is invalid".to_string());
        }
        if signed_response.is_empty() || signed_response.len() > MAX_CHECKPOINT_EVIDENCE_FRAME_BYTES
        {
            return failure("checkpoint evidence frame violates bounds".to_string());
        }
        let computed_digest: [u8; 32] = Sha256::digest(signed_response).into();
        if &computed_digest != evidence_digest {
            return failure("checkpoint evidence digest mismatch".to_string());
        }
        let observed_at = match i64::try_from(observed_at) {
            Ok(value) => value,
            Err(_) => return failure("checkpoint evidence time exceeds SQLite range".to_string()),
        };
        let local_tip_height = match i64::try_from(local_tip_height) {
            Ok(value) => value,
            Err(_) => {
                return failure("checkpoint evidence local height exceeds SQLite range".to_string())
            }
        };
        let remote_tip_height = match i64::try_from(remote_tip_height) {
            Ok(value) => value,
            Err(_) => {
                return failure(
                    "checkpoint evidence remote height exceeds SQLite range".to_string(),
                )
            }
        };
        let checkpoint_height = match i64::try_from(checkpoint_height) {
            Ok(value) => value,
            Err(_) => {
                return failure("checkpoint evidence height exceeds SQLite range".to_string())
            }
        };

        let result = {
            let mut conn = self.conn.lock().await;
            (|| -> Result<(u64, u64, Option<u64>), String> {
                let transaction = conn
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                    .map_err(|error| format!("begin checkpoint evidence transaction: {error}"))?;
                transaction
                    .execute(
                        "INSERT OR IGNORE INTO record_checkpoint_evidence
                         (evidence_digest,observed_at,relation,local_tip_height,
                          remote_tip_height,checkpoint_height,signed_response,created_at)
                         VALUES (?1,?2,?3,?4,?5,?6,?7,?2)",
                        params![
                            evidence_digest.as_slice(),
                            observed_at,
                            relation,
                            local_tip_height,
                            remote_tip_height,
                            checkpoint_height,
                            signed_response,
                        ],
                    )
                    .map_err(|error| format!("insert checkpoint evidence: {error}"))?;
                let stored = transaction
                    .query_row(
                        "SELECT observed_at,relation,local_tip_height,remote_tip_height,
                                checkpoint_height,signed_response
                         FROM record_checkpoint_evidence WHERE evidence_digest=?1",
                        params![evidence_digest.as_slice()],
                        |row| {
                            Ok((
                                row.get::<_, i64>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, i64>(2)?,
                                row.get::<_, i64>(3)?,
                                row.get::<_, i64>(4)?,
                                row.get::<_, Vec<u8>>(5)?,
                            ))
                        },
                    )
                    .map_err(|error| format!("read inserted checkpoint evidence: {error}"))?;
                if stored.0 != observed_at
                    || stored.1 != relation
                    || stored.2 != local_tip_height
                    || stored.3 != remote_tip_height
                    || stored.4 != checkpoint_height
                    || stored.5.as_slice() != signed_response
                {
                    return Err(
                        "existing checkpoint evidence conflicts with verified frame".to_string()
                    );
                }
                let count_i64 = transaction
                    .query_row(
                        "SELECT COUNT(*) FROM record_checkpoint_evidence",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .map_err(|error| format!("count checkpoint evidence: {error}"))?;
                let excess = count_i64.saturating_sub(CHECKPOINT_EVIDENCE_CAPACITY as i64);
                if excess > 0 {
                    transaction
                        .execute(
                            "DELETE FROM record_checkpoint_evidence WHERE rowid IN (
                                SELECT rowid FROM record_checkpoint_evidence
                                ORDER BY CASE WHEN relation='diverged' THEN 1 ELSE 0 END ASC,
                                         observed_at ASC,rowid ASC
                                LIMIT ?1
                            )",
                            params![excess],
                        )
                        .map_err(|error| format!("prune checkpoint evidence: {error}"))?;
                }
                let summary = transaction
                    .query_row(
                        "SELECT COUNT(*),
                                COALESCE(SUM(CASE WHEN relation='diverged' THEN 1 ELSE 0 END),0),
                                MAX(observed_at)
                         FROM record_checkpoint_evidence",
                        [],
                        |row| {
                            Ok((
                                row.get::<_, i64>(0)?,
                                row.get::<_, i64>(1)?,
                                row.get::<_, Option<i64>>(2)?,
                            ))
                        },
                    )
                    .map_err(|error| format!("summarize checkpoint evidence: {error}"))?;
                transaction
                    .commit()
                    .map_err(|error| format!("commit checkpoint evidence: {error}"))?;
                Ok((
                    u64::try_from(summary.0)
                        .map_err(|_| "checkpoint evidence count is invalid".to_string())?,
                    u64::try_from(summary.1).map_err(|_| {
                        "checkpoint divergence evidence count is invalid".to_string()
                    })?,
                    summary
                        .2
                        .map(u64::try_from)
                        .transpose()
                        .map_err(|_| "checkpoint evidence time is invalid".to_string())?,
                ))
            })()
        };

        match result {
            Ok((evidence_records, divergence_records, last_evidence_at)) => {
                let mut runtime = self.commitment_checkpoint.write();
                runtime.evidence_records = evidence_records;
                runtime.divergence_evidence_records = divergence_records;
                runtime.last_evidence_at = last_evidence_at;
                Ok(())
            }
            Err(error) => failure(error),
        }
    }

    /// Re-verifies every durable checkpoint frame before networking starts.
    /// Any malformed digest, non-canonical frame, invalid signature, impossible
    /// height relation, or local historical-hash mismatch fails startup closed.
    pub async fn audit_record_commitment_checkpoint_evidence(
        &self,
    ) -> Result<RecordCommitmentCheckpointEvidenceAudit, String> {
        self.commitment_checkpoint.write().evidence_state = "not_audited";
        let result = {
            let mut conn = self.conn.lock().await;
            audit_checkpoint_evidence_connection(&mut conn)
        };
        match result {
            Ok(report) => {
                let mut runtime = self.commitment_checkpoint.write();
                runtime.evidence_state = "verified";
                runtime.evidence_records = report.evidence_records;
                runtime.divergence_evidence_records = report.divergence_evidence_records;
                runtime.last_evidence_at = report.last_evidence_at;
                Ok(report)
            }
            Err(error) => {
                self.commitment_checkpoint.write().evidence_state = "invalid";
                Err(error)
            }
        }
    }

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

    /// Configures and verifies SQLite commit durability before chain audit.
    ///
    /// WAL + `NORMAL` preserves database consistency but may lose a recently
    /// acknowledged transaction after host power failure. The single-writer
    /// coordinator therefore upgrades the shared connection to `FULL` and
    /// refuses startup unless SQLite reports FULL-or-stronger. Followers keep
    /// the existing mode to avoid imposing coordinator write latency on every
    /// verifier. This setting contains no chain or user data.
    pub async fn configure_record_commitment_durability(
        &self,
        coordinator: bool,
    ) -> Result<&'static str, String> {
        let conn = self.conn.lock().await;
        if coordinator {
            conn.pragma_update(None, "synchronous", "FULL")
                .map_err(|error| format!("set coordinator SQLite durability: {error}"))?;
        }
        let level: i64 = conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .map_err(|error| format!("read SQLite durability: {error}"))?;
        let mode = match level {
            0 => "off",
            1 => "normal",
            2 => "full",
            3 => "extra",
            _ => return Err(format!("unsupported SQLite synchronous level {level}")),
        };
        if coordinator && level < 2 {
            return Err(format!(
                "commitment coordinator requires SQLite FULL durability; effective mode is {mode}"
            ));
        }
        self.commitment_durability
            .store(level as u64, Ordering::Release);
        info!(
            role = if coordinator {
                "coordinator"
            } else {
                "non_coordinator"
            },
            durability_mode = mode,
            "[MEMCHAIN_BLOCK] SQLite commitment durability configured"
        );
        Ok(mode)
    }

    /// Verifies or initializes the coordinator's signed tip high-water mark.
    ///
    /// This must run after the complete SQLite chain audit. A stored anchor may
    /// be behind the audited tip only when its exact height/hash remains an
    /// ancestor of that chain; in that case it is atomically advanced. An
    /// anchor ahead of SQLite, a same-height mismatch, an invalid signature,
    /// or an ancestry mismatch clears runtime integrity and fails startup.
    ///
    /// Scope: this detects SQLite rollback/replacement while the host-side
    /// anchor remains. Replaying a whole disk/VM snapshot can roll back both
    /// files and requires an external peer witness in a later protocol layer.
    pub async fn configure_record_commitment_tip_anchor(
        &self,
        path: impl AsRef<Path>,
        identity: &IdentityKeyPair,
    ) -> Result<&'static str, String> {
        let path = path.as_ref().to_path_buf();
        let (current_height, current_hash) = self
            .commitment_integrity
            .read()
            .as_ref()
            .map(|runtime| (runtime.verified_tip_height, runtime.verified_tip_hash))
            .ok_or_else(|| {
                "commitment tip anchor requires a successful full chain audit".to_string()
            })?;
        {
            let mut runtime = self.commitment_tip_anchor.write();
            *runtime = Default::default();
            runtime.config = Some(RecordCommitmentTipAnchorConfig {
                path: path.clone(),
                identity: identity.clone(),
            });
            runtime.state = "checking";
        }

        let stored = match read_record_commitment_tip_anchor(&path).await {
            Ok(stored) => stored,
            Err(error) => {
                self.fail_record_commitment_tip_anchor("invalid", 0, false);
                return Err(error);
            }
        };
        let now = unix_now_secs();

        match stored {
            None => {
                let persisted_at = persist_record_commitment_tip_anchor(
                    path,
                    current_height,
                    current_hash,
                    identity,
                )
                .await
                .map_err(|error| {
                    self.fail_record_commitment_tip_anchor("write_failed", current_height, true);
                    error
                })?;
                let mut runtime = self.commitment_tip_anchor.write();
                runtime.state = "initialized";
                runtime.anchored_height = current_height;
                runtime.last_verified_at = Some(now);
                runtime.last_persisted_at = Some(persisted_at);
                info!(
                    tip_height = current_height,
                    "[MEMCHAIN_BLOCK] Signed commitment tip anchor initialized"
                );
                Ok("initialized")
            }
            Some(anchor) => {
                let verified = match anchor.verify(&identity.public_key_bytes()) {
                    Ok(verified) => verified,
                    Err(error) => {
                        self.fail_record_commitment_tip_anchor("invalid", 0, false);
                        return Err(error);
                    }
                };
                if verified.tip_height > current_height {
                    self.fail_record_commitment_tip_anchor(
                        "rollback_detected",
                        verified.tip_height,
                        false,
                    );
                    return Err(format!(
                        "commitment SQLite tip height {current_height} is behind signed local anchor height {}",
                        verified.tip_height
                    ));
                }

                let ancestor_hash = if verified.tip_height == 0 {
                    GENESIS_PREV_HASH
                } else {
                    let height = i64::try_from(verified.tip_height).map_err(|_| {
                        self.fail_record_commitment_tip_anchor(
                            "rollback_detected",
                            verified.tip_height,
                            false,
                        );
                        "commitment tip anchor height exceeds SQLite range".to_string()
                    })?;
                    let conn = self.conn.lock().await;
                    let hash: Option<Vec<u8>> = conn
                        .query_row(
                            "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                            params![height],
                            |row| row.get(0),
                        )
                        .optional()
                        .map_err(|error| {
                            self.fail_record_commitment_tip_anchor(
                                "invalid",
                                verified.tip_height,
                                false,
                            );
                            format!("read anchored commitment ancestor: {error}")
                        })?;
                    hash.ok_or_else(|| {
                        self.fail_record_commitment_tip_anchor(
                            "rollback_detected",
                            verified.tip_height,
                            false,
                        );
                        "signed commitment tip anchor is not present in the audited chain"
                            .to_string()
                    })?
                    .try_into()
                    .map_err(|hash: Vec<u8>| {
                        self.fail_record_commitment_tip_anchor(
                            "rollback_detected",
                            verified.tip_height,
                            false,
                        );
                        format!(
                            "anchored commitment ancestor hash has invalid length {}",
                            hash.len()
                        )
                    })?
                };
                if ancestor_hash != verified.tip_hash {
                    self.fail_record_commitment_tip_anchor(
                        "rollback_detected",
                        verified.tip_height,
                        false,
                    );
                    return Err(format!(
                        "signed commitment tip anchor ancestry mismatch at height {}",
                        verified.tip_height
                    ));
                }

                if verified.tip_height == current_height {
                    let mut runtime = self.commitment_tip_anchor.write();
                    runtime.state = "verified";
                    runtime.anchored_height = current_height;
                    runtime.last_verified_at = Some(now);
                    runtime.last_persisted_at = Some(verified.updated_at);
                    info!(
                        tip_height = current_height,
                        "[MEMCHAIN_BLOCK] Signed commitment tip anchor verified"
                    );
                    return Ok("verified");
                }

                let persisted_at = persist_record_commitment_tip_anchor(
                    path,
                    current_height,
                    current_hash,
                    identity,
                )
                .await
                .map_err(|error| {
                    self.fail_record_commitment_tip_anchor(
                        "write_failed",
                        verified.tip_height,
                        true,
                    );
                    error
                })?;
                let mut runtime = self.commitment_tip_anchor.write();
                runtime.state = "repaired";
                runtime.anchored_height = current_height;
                runtime.last_verified_at = Some(now);
                runtime.last_persisted_at = Some(persisted_at);
                info!(
                    previous_height = verified.tip_height,
                    tip_height = current_height,
                    "[MEMCHAIN_BLOCK] Signed commitment tip anchor advanced after audited DB-ahead recovery"
                );
                Ok("repaired")
            }
        }
    }

    fn fail_record_commitment_tip_anchor(
        &self,
        state: &'static str,
        anchored_height: u64,
        write_failure: bool,
    ) {
        let mut runtime = self.commitment_tip_anchor.write();
        runtime.state = state;
        runtime.anchored_height = anchored_height;
        runtime.last_verified_at = None;
        if write_failure {
            runtime.write_failures_total = runtime.write_failures_total.saturating_add(1);
        }
        drop(runtime);
        *self.commitment_integrity.write() = None;
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
        runtime.state = if has_more {
            "catching_up"
        } else {
            "checkpointing"
        };
        runtime.last_success_at = Some(now);
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.pages_received_total = runtime.pages_received_total.saturating_add(1);
        runtime.blocks_received_total = runtime
            .blocks_received_total
            .saturating_add(verified_blocks);
    }

    /// Marks a follower current only after a signed equal-tip checkpoint.
    ///
    /// A terminal block-range page alone is insufficient because the remote
    /// tip may move between page construction and local append. This explicit
    /// gate keeps `current` equivalent to a signature-verified convergence
    /// observation.
    pub fn record_commitment_sync_checkpoint_success(&self, now: u64, remote_tip_height: u64) {
        let mut runtime = self.commitment_sync.write();
        if !runtime.enabled {
            return;
        }
        let recovered = runtime.consecutive_failures > 0;
        runtime.state = "current";
        runtime.last_success_at = Some(now);
        runtime.remote_tip_height = Some(remote_tip_height);
        runtime.consecutive_failures = 0;
        runtime.last_error_code = None;
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
        const POLICY: &str = "full snapshot-consistent startup audit plus transactionally verified appends; coordinator requires SQLite FULL-or-stronger commit durability and a signed local tip anchor; no block hashes, proposer identities, commitment ids, owners, payloads, peers, endpoints, routes, or client metadata";
        const ROLLBACK_GUARD_SCOPE: &str = "detects commitment SQLite rollback or replacement while the host-side signed anchor remains; does not detect whole-host or whole-disk snapshot rollback and is not consensus, quorum, or finality";
        let durability_mode = match self.commitment_durability.load(Ordering::Acquire) {
            0 => "off",
            1 => "normal",
            2 => "full",
            3 => "extra",
            _ => "unknown",
        };
        let rollback_guard = self.commitment_tip_anchor.read();
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
                durability_mode,
                rollback_guard_state: rollback_guard.state,
                rollback_guard_height: rollback_guard.anchored_height,
                rollback_guard_last_verified_at: rollback_guard.last_verified_at,
                rollback_guard_last_persisted_at: rollback_guard.last_persisted_at,
                rollback_guard_write_failures_total: rollback_guard.write_failures_total,
                rollback_guard_scope: ROLLBACK_GUARD_SCOPE,
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
                durability_mode,
                rollback_guard_state: rollback_guard.state,
                rollback_guard_height: rollback_guard.anchored_height,
                rollback_guard_last_verified_at: rollback_guard.last_verified_at,
                rollback_guard_last_persisted_at: rollback_guard.last_persisted_at,
                rollback_guard_write_failures_total: rollback_guard.write_failures_total,
                rollback_guard_scope: ROLLBACK_GUARD_SCOPE,
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
            verified_tip_hash: expected_prev_hash,
        });
        drop(conn);
        Ok(report)
    }

    /// Verifies and appends one node-blind commitment block.
    ///
    /// This compatibility API delegates to the bounded batch primitive so
    /// local mining and peer catch-up cannot drift into different validation,
    /// durability, integrity-baseline, or rollback-anchor behavior.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, persistence, integrity advancement,
    /// or configured signed-tip anchor persistence fails.
    pub async fn append_record_commitment_block(
        &self,
        block: &RecordCommitmentBlockV1,
        received_from: Option<&[u8; 32]>,
    ) -> Result<RecordCommitmentAppendOutcome, String> {
        let outcome = self
            .append_record_commitment_blocks_atomic(std::slice::from_ref(block), received_from)
            .await?;
        match (outcome.inserted, outcome.already_present) {
            (1, 0) => Ok(RecordCommitmentAppendOutcome::Inserted),
            (0, 1) => Ok(RecordCommitmentAppendOutcome::AlreadyPresent),
            _ => Err("single commitment append produced an invalid aggregate outcome".to_string()),
        }
    }

    /// Atomically verifies and appends one bounded node-blind block page.
    ///
    /// Height order, previous-hash continuity, proposer signatures, Merkle
    /// integrity, commitment uniqueness, block rows, membership indexes, and
    /// the final tip share one `SQLite` `IMMEDIATE` transaction. Exact durable
    /// prefixes are idempotent, which permits safe retry after a lost response;
    /// any fork, gap, invalid block, or storage failure rolls back every newly
    /// inserted block in the page.
    ///
    /// The signed sidecar remains intentionally outside `SQLite`. After a page
    /// commit, it advances once to the final verified tip. A sidecar failure
    /// fails closed exactly like the legacy single-block path: `SQLite` remains
    /// durable, verified runtime state is cleared, and restart audit must
    /// safely repair the high-water mark before another append.
    ///
    /// # Errors
    ///
    /// Returns an error for oversized or non-contiguous input, a fork, an
    /// invalid block, a storage failure, a stale integrity baseline, or a
    /// configured signed-tip anchor failure.
    pub async fn append_record_commitment_blocks_atomic(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        received_from: Option<&[u8; 32]>,
    ) -> Result<RecordCommitmentBatchAppendOutcome, String> {
        if blocks.is_empty() {
            return Ok(RecordCommitmentBatchAppendOutcome::default());
        }
        if blocks.len() > MAX_ATOMIC_COMMITMENT_BLOCK_BATCH {
            return Err(format!(
                "commitment block batch exceeds maximum of {MAX_ATOMIC_COMMITMENT_BLOCK_BATCH}"
            ));
        }

        let (anchor_enabled, anchor_ready) = {
            let runtime = self.commitment_tip_anchor.read();
            (
                runtime.config.is_some(),
                matches!(runtime.state, "initialized" | "verified" | "repaired"),
            )
        };
        if anchor_enabled && !anchor_ready {
            return Err(
                "commitment tip anchor is not ready; restart and complete startup audit"
                    .to_string(),
            );
        }

        let committed = self
            .append_record_commitment_block_page_transaction(blocks, received_from)
            .await?;
        if committed.inserted_indices.is_empty() {
            return Ok(committed.outcome);
        }

        let can_advance = self.advance_record_commitment_integrity_for_batch(
            blocks,
            &committed.inserted_indices,
            committed.appended_at,
        );
        self.persist_record_commitment_batch_anchor(
            blocks,
            &committed.inserted_indices,
            anchor_enabled,
            can_advance,
        )
        .await?;

        for index in committed.inserted_indices {
            let block = &blocks[index];
            info!(
                height = block.header.height,
                commitments = block.record_ids.len(),
                hash = %block.header.hash_hex(),
                source = if received_from.is_some() { "peer" } else { "local" },
                "[MEMCHAIN_BLOCK] Verified commitment block appended"
            );
        }
        Ok(committed.outcome)
    }

    async fn append_record_commitment_block_page_transaction(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        received_from: Option<&[u8; 32]>,
    ) -> Result<CommittedRecordCommitmentBatch, String> {
        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|error| format!("begin commitment block batch transaction: {error}"))?;
        let tip: Option<(i64, Vec<u8>)> = transaction
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| format!("read commitment chain tip: {error}"))?;
        let (mut current_height, mut current_hash) = match tip {
            Some((height, hash)) => {
                let height = u64::try_from(height)
                    .map_err(|_| "stored commitment tip has invalid height".to_string())?;
                let hash: [u8; 32] = hash.try_into().map_err(|value: Vec<u8>| {
                    format!("stored tip hash has invalid length {}", value.len())
                })?;
                (height, hash)
            }
            None => (0, GENESIS_PREV_HASH),
        };
        let appended_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let created_at = i64::try_from(appended_at)
            .map_err(|_| "commitment append time exceeds SQLite range".to_string())?;
        let mut outcome = RecordCommitmentBatchAppendOutcome::default();
        let mut inserted_indices = Vec::with_capacity(blocks.len());
        let mut previous_input_height: Option<u64> = None;

        for (index, block) in blocks.iter().enumerate() {
            if previous_input_height
                .is_some_and(|height| height.checked_add(1) != Some(block.header.height))
            {
                return Err("commitment block batch is not height-contiguous".to_string());
            }
            previous_input_height = Some(block.header.height);
            let expected_height = current_height
                .checked_add(1)
                .ok_or_else(|| "commitment chain height exhausted".to_string())?;
            match persist_record_commitment_block_transaction(
                &transaction,
                block,
                received_from,
                expected_height,
                &current_hash,
                created_at,
            )? {
                RecordCommitmentAppendOutcome::Inserted => {
                    current_height = block.header.height;
                    current_hash = block.hash();
                    inserted_indices.push(index);
                    outcome.inserted = outcome.inserted.saturating_add(1);
                }
                RecordCommitmentAppendOutcome::AlreadyPresent => {
                    outcome.already_present = outcome.already_present.saturating_add(1);
                }
            }
        }

        if !inserted_indices.is_empty() {
            persist_record_commitment_tip_transaction(&transaction, current_height, &current_hash)?;
        }
        transaction
            .commit()
            .map_err(|error| format!("commit commitment block batch transaction: {error}"))?;
        drop(conn);
        Ok(CommittedRecordCommitmentBatch {
            outcome,
            inserted_indices,
            appended_at,
        })
    }

    fn advance_record_commitment_integrity_for_batch(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        inserted_indices: &[usize],
        appended_at: u64,
    ) -> bool {
        let mut integrity = self.commitment_integrity.write();
        let mut can_advance = integrity.is_some();
        for index in inserted_indices {
            let Some(block) = blocks.get(*index) else {
                can_advance = false;
                break;
            };
            let step_valid = integrity.as_ref().is_some_and(|runtime| {
                runtime.verified_tip_height.checked_add(1) == Some(block.header.height)
                    && runtime.verified_block_count.checked_add(1) == Some(block.header.height)
                    && runtime.verified_tip_hash == block.header.prev_block_hash
            });
            if !step_valid {
                can_advance = false;
                break;
            }
            if let Some(runtime) = integrity.as_mut() {
                runtime.last_verified_at = appended_at;
                runtime.verified_block_count = runtime.verified_block_count.saturating_add(1);
                runtime.verified_commitment_count = runtime
                    .verified_commitment_count
                    .saturating_add(block.record_ids.len() as u64);
                runtime.verified_tip_height = block.header.height;
                runtime.verified_tip_hash = block.hash();
            }
        }
        if !can_advance && integrity.is_some() {
            *integrity = None;
        }
        can_advance
    }

    async fn persist_record_commitment_batch_anchor(
        &self,
        blocks: &[RecordCommitmentBlockV1],
        inserted_indices: &[usize],
        anchor_enabled: bool,
        can_advance: bool,
    ) -> Result<(), String> {
        let final_block = inserted_indices
            .last()
            .and_then(|index| blocks.get(*index))
            .ok_or_else(|| "commitment batch lost its inserted outcome".to_string())?;
        if anchor_enabled && !can_advance {
            self.fail_record_commitment_tip_anchor("invalid", final_block.header.height, false);
            return Err(
                "commitment block was committed in an atomic batch but the verified runtime baseline could not advance; restart and re-audit before producing another block"
                    .to_string(),
            );
        }

        let anchor_config = self
            .commitment_tip_anchor
            .read()
            .config
            .as_ref()
            .map(|config| (config.path.clone(), config.identity.clone()));
        let Some((path, identity)) = anchor_config else {
            return Ok(());
        };
        let persisted_at = match persist_record_commitment_tip_anchor(
            path,
            final_block.header.height,
            final_block.hash(),
            &identity,
        )
        .await
        {
            Ok(persisted_at) => persisted_at,
            Err(error) => {
                let previous_height = inserted_indices
                    .first()
                    .and_then(|index| blocks.get(*index))
                    .map_or(final_block.header.height, |block| block.header.height)
                    .saturating_sub(1);
                self.fail_record_commitment_tip_anchor("write_failed", previous_height, true);
                return Err(format!(
                    "commitment block was committed in an atomic batch but signed tip anchor persistence failed; restart and re-audit: {error}"
                ));
            }
        };
        let mut runtime = self.commitment_tip_anchor.write();
        runtime.state = "verified";
        runtime.anchored_height = final_block.header.height;
        runtime.last_verified_at = Some(persisted_at);
        runtime.last_persisted_at = Some(persisted_at);
        drop(runtime);
        Ok(())
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

    /// Returns one audit-backed comparison checkpoint and the current tip from
    /// a single SQLite view. Height zero uses the protocol genesis hash.
    pub async fn record_commitment_chain_checkpoint(
        &self,
        requested_height: u64,
    ) -> Result<(u64, [u8; 32], u64, [u8; 32]), String> {
        let conn = self.conn.lock().await;
        let integrity = (*self.commitment_integrity.read())
            .ok_or_else(|| "commitment chain is not fully audited".to_string())?;
        let tip: Option<(i64, Vec<u8>)> = conn
            .query_row(
                "SELECT height,block_hash FROM record_commitment_blocks
                 ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|error| format!("read commitment checkpoint tip: {error}"))?;
        let (tip_height, tip_hash) = match tip {
            Some((height, hash)) => {
                let height = u64::try_from(height)
                    .map_err(|_| "commitment checkpoint tip height is invalid".to_string())?;
                let hash: [u8; 32] = hash.try_into().map_err(|hash: Vec<u8>| {
                    format!("commitment checkpoint tip hash length {}", hash.len())
                })?;
                (height, hash)
            }
            None => (0, GENESIS_PREV_HASH),
        };
        if integrity.verified_tip_height != tip_height || integrity.verified_tip_hash != tip_hash {
            return Err("commitment chain audit baseline is stale".to_string());
        }

        let checkpoint_height = requested_height.min(tip_height);
        let checkpoint_hash = if checkpoint_height == 0 {
            GENESIS_PREV_HASH
        } else {
            let height = i64::try_from(checkpoint_height)
                .map_err(|_| "commitment checkpoint height exceeds SQLite range".to_string())?;
            let hash: Vec<u8> = conn
                .query_row(
                    "SELECT block_hash FROM record_commitment_blocks WHERE height=?1",
                    params![height],
                    |row| row.get(0),
                )
                .map_err(|error| format!("read commitment checkpoint hash: {error}"))?;
            hash.try_into().map_err(|hash: Vec<u8>| {
                format!("commitment checkpoint hash length {}", hash.len())
            })?
        };
        Ok((checkpoint_height, checkpoint_hash, tip_height, tip_hash))
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
        let checkpoint = self.record_commitment_checkpoint_status();
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
            checkpoint,
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
    use tempfile::TempDir;

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
    async fn test_commitment_coordinator_upgrades_sqlite_durability() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        assert_eq!(
            storage
                .record_commitment_chain_integrity_status()
                .durability_mode,
            "normal"
        );
        assert_eq!(
            storage
                .configure_record_commitment_durability(false)
                .await
                .unwrap(),
            "normal"
        );
        assert_eq!(
            storage
                .configure_record_commitment_durability(true)
                .await
                .unwrap(),
            "full"
        );
        let effective_level: i64 = {
            let conn = storage.conn_lock().await;
            conn.query_row("PRAGMA synchronous", [], |row| row.get(0))
                .unwrap()
        };
        assert!(effective_level >= 2);
        assert_eq!(
            storage
                .record_commitment_chain_integrity_status()
                .durability_mode,
            "full"
        );
    }

    #[tokio::test]
    async fn test_commitment_tip_anchor_initializes_advances_and_verifies_after_restart() {
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("anchor-restart.db");
        let anchor_path = directory.path().join("commitment-tip.json");
        let identity = IdentityKeyPair::generate();

        let storage = MemoryStorage::open(&db_path, None).unwrap();
        storage
            .configure_record_commitment_durability(true)
            .await
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        assert_eq!(
            storage
                .configure_record_commitment_tip_anchor(&anchor_path, &identity)
                .await
                .unwrap(),
            "initialized"
        );
        let initialized = storage.record_commitment_chain_integrity_status();
        assert_eq!(initialized.rollback_guard_state, "initialized");
        assert_eq!(initialized.rollback_guard_height, 0);
        assert!(anchor_path.is_file());

        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x61, &identity);
        assert_eq!(
            storage
                .append_record_commitment_block(&first, None)
                .await
                .unwrap(),
            RecordCommitmentAppendOutcome::Inserted
        );
        let advanced = storage.record_commitment_chain_integrity_status();
        assert_eq!(advanced.state, "verified");
        assert_eq!(advanced.rollback_guard_state, "verified");
        assert_eq!(advanced.rollback_guard_height, 1);
        assert!(advanced.rollback_guard_last_verified_at.is_some());
        assert!(advanced.rollback_guard_last_persisted_at.is_some());
        drop(storage);

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        reopened
            .configure_record_commitment_durability(true)
            .await
            .unwrap();
        reopened.audit_record_commitment_chain().await.unwrap();
        assert_eq!(
            reopened
                .configure_record_commitment_tip_anchor(&anchor_path, &identity)
                .await
                .unwrap(),
            "verified"
        );
        let verified = reopened.record_commitment_chain_integrity_status();
        assert_eq!(verified.rollback_guard_state, "verified");
        assert_eq!(verified.rollback_guard_height, 1);
        assert_eq!(verified.verified_tip_height, 1);
    }

    #[tokio::test]
    async fn test_commitment_tip_anchor_rejects_valid_older_database_snapshot() {
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("anchor-rollback.db");
        let anchor_path = directory.path().join("commitment-tip.json");
        let identity = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        storage
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap();
        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x62, &identity);
        let second = signed_commitment_block(2, first.hash(), 0x63, &identity);
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        storage
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();
        drop(storage);

        // Simulate restoring a self-consistent SQLite snapshot from height 1
        // while leaving the separately persisted signed high-water mark at 2.
        let mut rollback = rusqlite::Connection::open(&db_path).unwrap();
        rollback.execute_batch("PRAGMA foreign_keys=ON;").unwrap();
        let transaction = rollback.transaction().unwrap();
        transaction
            .execute(
                "DELETE FROM record_block_commitments WHERE block_height=2",
                [],
            )
            .unwrap();
        transaction
            .execute("DELETE FROM record_commitment_blocks WHERE height=2", [])
            .unwrap();
        for (key, value) in [
            ("record_block_tip_hash", first.hash().to_vec()),
            ("record_block_tip_height", 1u64.to_le_bytes().to_vec()),
        ] {
            transaction
                .execute(
                    "INSERT OR REPLACE INTO chain_state (key,value) VALUES (?1,?2)",
                    params![key, value],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
        drop(rollback);

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        let audit = reopened.audit_record_commitment_chain().await.unwrap();
        assert_eq!(audit.tip_height, 1);
        let error = reopened
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap_err();
        assert!(error.contains("behind signed local anchor height 2"));
        let rejected = reopened.record_commitment_chain_integrity_status();
        assert_eq!(rejected.state, "not_verified");
        assert_eq!(rejected.rollback_guard_state, "rollback_detected");
        assert_eq!(rejected.rollback_guard_height, 2);
    }

    #[tokio::test]
    async fn test_commitment_tip_anchor_rejects_same_height_hash_conflict() {
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("anchor-conflict.db");
        let anchor_path = directory.path().join("commitment-tip.json");
        let identity = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        storage
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap();
        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x64, &identity);
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        drop(storage);

        // The conflict is signed by the correct node identity. Rejection must
        // therefore come from ancestry comparison, not signature validation.
        let conflicting =
            RecordCommitmentTipAnchorV1::new_signed(1, [0xA7; 32], &identity, unix_now_secs());
        let bytes = serde_json::to_vec(&conflicting).unwrap();
        write_record_commitment_tip_anchor_atomic(&anchor_path, &bytes).unwrap();

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        reopened.audit_record_commitment_chain().await.unwrap();
        let error = reopened
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap_err();
        assert!(error.contains("ancestry mismatch at height 1"));
        let rejected = reopened.record_commitment_chain_integrity_status();
        assert_eq!(rejected.state, "not_verified");
        assert_eq!(rejected.rollback_guard_state, "rollback_detected");
    }

    #[tokio::test]
    async fn test_commitment_tip_anchor_repairs_audited_database_ahead_state() {
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("anchor-db-ahead.db");
        let anchor_path = directory.path().join("commitment-tip.json");
        let identity = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        storage
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap();
        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x65, &identity);
        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        drop(storage);

        // Model the only expected mismatch window: SQLite committed a later
        // verified block, then the process stopped before sidecar replacement.
        let db_ahead = MemoryStorage::open(&db_path, None).unwrap();
        db_ahead.audit_record_commitment_chain().await.unwrap();
        let second = signed_commitment_block(2, first.hash(), 0x66, &identity);
        db_ahead
            .append_record_commitment_block(&second, None)
            .await
            .unwrap();
        drop(db_ahead);

        let reopened = MemoryStorage::open(&db_path, None).unwrap();
        reopened.audit_record_commitment_chain().await.unwrap();
        assert_eq!(
            reopened
                .configure_record_commitment_tip_anchor(&anchor_path, &identity)
                .await
                .unwrap(),
            "repaired"
        );
        let repaired = reopened.record_commitment_chain_integrity_status();
        assert_eq!(repaired.state, "verified");
        assert_eq!(repaired.rollback_guard_state, "repaired");
        assert_eq!(repaired.rollback_guard_height, 2);
    }

    #[tokio::test]
    async fn test_commitment_tip_anchor_write_failure_commits_once_then_fails_closed() {
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("anchor-write-failure.db");
        let anchor_parent = directory.path().join("anchor-parent");
        let anchor_path = anchor_parent.join("commitment-tip.json");
        let identity = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        storage
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap();

        std::fs::remove_file(&anchor_path).unwrap();
        std::fs::remove_dir(&anchor_parent).unwrap();
        std::fs::write(&anchor_parent, b"blocks child creation").unwrap();

        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0x67, &identity);
        let error = storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap_err();
        assert!(error.contains("block was committed"));
        assert!(error.contains("anchor persistence failed"));
        assert_eq!(storage.record_commitment_chain_tip().await.0, 1);
        let failed = storage.record_commitment_chain_integrity_status();
        assert_eq!(failed.state, "not_verified");
        assert_eq!(failed.rollback_guard_state, "write_failed");
        assert_eq!(failed.rollback_guard_height, 0);
        assert_eq!(failed.rollback_guard_write_failures_total, 1);

        let second = signed_commitment_block(2, first.hash(), 0x68, &identity);
        let blocked = storage
            .append_record_commitment_block(&second, None)
            .await
            .unwrap_err();
        assert!(blocked.contains("anchor is not ready"));
        assert_eq!(storage.record_commitment_chain_tip().await.0, 1);
    }

    #[tokio::test]
    async fn test_commitment_append_failure_rolls_back_entire_block_after_reopen() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("commitment-atomicity.db");
        let storage = MemoryStorage::open(&path, None).unwrap();
        storage
            .configure_record_commitment_durability(true)
            .await
            .unwrap();
        storage.audit_record_commitment_chain().await.unwrap();

        // Inject a deterministic storage failure after the first membership
        // row. The block row, both memberships, and chain_state tip must remain
        // one atomic unit even when the transaction aborts mid-append.
        {
            let conn = storage.conn_lock().await;
            conn.execute_batch(
                "CREATE TEMP TRIGGER fail_second_commitment_membership
                 BEFORE INSERT ON record_block_commitments
                 WHEN (SELECT COUNT(*) FROM record_block_commitments) = 1
                 BEGIN
                     SELECT RAISE(ABORT, 'injected membership failure');
                 END;",
            )
            .unwrap();
        }
        let identity = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_190_001,
            GENESIS_PREV_HASH,
            vec![[0x91; 32], [0x92; 32]],
            &identity,
        );
        assert!(storage
            .append_record_commitment_block(&block, None)
            .await
            .unwrap_err()
            .contains("injected membership failure"));
        {
            let conn = storage.conn_lock().await;
            conn.execute_batch("DROP TRIGGER fail_second_commitment_membership;")
                .unwrap();
            let (blocks, memberships, tip_keys): (i64, i64, i64) = conn
                .query_row(
                    "SELECT
                         (SELECT COUNT(*) FROM record_commitment_blocks),
                         (SELECT COUNT(*) FROM record_block_commitments),
                         (SELECT COUNT(*) FROM chain_state WHERE key IN
                            ('record_block_tip_hash','record_block_tip_height',
                             'record_block_chain_id'))",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            assert_eq!((blocks, memberships, tip_keys), (0, 0, 0));
        }
        drop(storage);

        // A fresh process must see the same empty, valid chain. This catches
        // accidental reliance on connection-local rollback visibility.
        let reopened = MemoryStorage::open(&path, None).unwrap();
        reopened
            .configure_record_commitment_durability(true)
            .await
            .unwrap();
        assert_eq!(
            reopened.audit_record_commitment_chain().await.unwrap(),
            RecordCommitmentChainAudit {
                block_count: 0,
                commitment_count: 0,
                tip_height: 0,
            }
        );
    }

    #[tokio::test]
    async fn test_commitment_page_failure_rolls_back_every_block() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("commitment-page-atomicity.db");
        let storage = MemoryStorage::open(&path, None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();

        // Fail while inserting the second block. A per-block transaction would
        // leave height 1 durable; the page transaction must leave no trace.
        {
            let conn = storage.conn_lock().await;
            conn.execute_batch(
                "CREATE TEMP TRIGGER fail_second_page_block
                 BEFORE INSERT ON record_commitment_blocks
                 WHEN NEW.height = 2
                 BEGIN
                     SELECT RAISE(ABORT, 'injected second block failure');
                 END;",
            )
            .unwrap();
        }
        let identity = IdentityKeyPair::generate();
        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xA1, &identity);
        let second = signed_commitment_block(2, first.hash(), 0xA2, &identity);
        let error = storage
            .append_record_commitment_blocks_atomic(&[first.clone(), second.clone()], None)
            .await
            .unwrap_err();
        assert!(error.contains("injected second block failure"));
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (0, GENESIS_PREV_HASH)
        );
        {
            let conn = storage.conn_lock().await;
            let (blocks, memberships, tip_keys): (i64, i64, i64) = conn
                .query_row(
                    "SELECT
                         (SELECT COUNT(*) FROM record_commitment_blocks),
                         (SELECT COUNT(*) FROM record_block_commitments),
                         (SELECT COUNT(*) FROM chain_state WHERE key IN
                            ('record_block_tip_hash','record_block_tip_height',
                             'record_block_chain_id'))",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
            assert_eq!((blocks, memberships, tip_keys), (0, 0, 0));
            conn.execute_batch("DROP TRIGGER fail_second_page_block;")
                .unwrap();
        }

        // Cross-block commitment reuse is a consensus-invalid page, not a
        // reason to retain the valid prefix that happened to be processed first.
        let duplicate_second = signed_commitment_block(2, first.hash(), 0xA1, &identity);
        let duplicate_error = storage
            .append_record_commitment_blocks_atomic(&[first.clone(), duplicate_second], None)
            .await
            .unwrap_err();
        assert!(duplicate_error.contains("persist commitment membership"));
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (0, GENESIS_PREV_HASH)
        );

        let outcome = storage
            .append_record_commitment_blocks_atomic(&[first, second.clone()], None)
            .await
            .unwrap();
        assert_eq!(
            outcome,
            RecordCommitmentBatchAppendOutcome {
                inserted: 2,
                already_present: 0,
            }
        );
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (2, second.hash())
        );
        assert_eq!(
            storage.record_commitment_chain_integrity_status().state,
            "verified"
        );
    }

    #[tokio::test]
    async fn test_commitment_page_retry_is_idempotent_and_rejects_forks() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        let identity = IdentityKeyPair::generate();
        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xB1, &identity);
        let second = signed_commitment_block(2, first.hash(), 0xB2, &identity);
        let third = signed_commitment_block(3, second.hash(), 0xB3, &identity);

        storage
            .append_record_commitment_block(&first, None)
            .await
            .unwrap();
        let mixed = storage
            .append_record_commitment_blocks_atomic(
                &[first.clone(), second.clone(), third.clone()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(mixed.inserted, 2);
        assert_eq!(mixed.already_present, 1);

        let replay = storage
            .append_record_commitment_blocks_atomic(
                &[first.clone(), second.clone(), third.clone()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(replay.inserted, 0);
        assert_eq!(replay.already_present, 3);

        let fork = signed_commitment_block(2, first.hash(), 0xBF, &identity);
        let error = storage
            .append_record_commitment_blocks_atomic(&[first.clone(), fork], None)
            .await
            .unwrap_err();
        assert!(error.contains("fork at height 2"));
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (3, third.hash())
        );

        let oversized = vec![first.clone(); MAX_ATOMIC_COMMITMENT_BLOCK_BATCH + 1];
        let oversized_error = storage
            .append_record_commitment_blocks_atomic(&oversized, None)
            .await
            .unwrap_err();
        assert!(oversized_error.contains("exceeds maximum"));

        let non_contiguous = storage
            .append_record_commitment_blocks_atomic(&[first, third], None)
            .await
            .unwrap_err();
        assert!(non_contiguous.contains("not height-contiguous"));
        assert_eq!(
            storage
                .append_record_commitment_blocks_atomic(&[], None)
                .await
                .unwrap(),
            RecordCommitmentBatchAppendOutcome::default()
        );
    }

    #[tokio::test]
    async fn test_commitment_page_advances_integrity_and_anchor_to_final_tip() {
        let directory = TempDir::new().unwrap();
        let db_path = directory.path().join("commitment-page-anchor.db");
        let anchor_path = directory.path().join("commitment-page-tip.json");
        let identity = IdentityKeyPair::generate();
        let storage = MemoryStorage::open(&db_path, None).unwrap();
        storage.audit_record_commitment_chain().await.unwrap();
        storage
            .configure_record_commitment_tip_anchor(&anchor_path, &identity)
            .await
            .unwrap();
        let first = signed_commitment_block(1, GENESIS_PREV_HASH, 0xC1, &identity);
        let second = signed_commitment_block(2, first.hash(), 0xC2, &identity);

        let outcome = storage
            .append_record_commitment_blocks_atomic(&[first, second.clone()], None)
            .await
            .unwrap();
        assert_eq!(outcome.inserted, 2);
        let status = storage.record_commitment_chain_integrity_status();
        assert_eq!(status.state, "verified");
        assert_eq!(status.verified_block_count, 2);
        assert_eq!(status.verified_commitment_count, 2);
        assert_eq!(status.verified_tip_height, 2);
        assert_eq!(status.rollback_guard_state, "verified");
        assert_eq!(status.rollback_guard_height, 2);
        assert_eq!(
            storage.record_commitment_chain_tip().await,
            (2, second.hash())
        );
    }

    #[tokio::test]
    async fn test_commitment_chain_append_range_and_status() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        assert!(storage
            .record_commitment_chain_checkpoint(0)
            .await
            .unwrap_err()
            .contains("not fully audited"));
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
        assert_eq!(
            storage.record_commitment_chain_checkpoint(0).await.unwrap(),
            (0, GENESIS_PREV_HASH, 0, GENESIS_PREV_HASH)
        );
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
        assert_eq!(
            storage.record_commitment_chain_checkpoint(1).await.unwrap(),
            (1, first.hash(), 2, second.hash())
        );
        assert_eq!(
            storage
                .record_commitment_chain_checkpoint(u64::MAX)
                .await
                .unwrap(),
            (2, second.hash(), 2, second.hash())
        );

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
    async fn test_checkpoint_refuses_same_height_tip_tampering_after_audit() {
        let (storage, _) = commitment_audit_fixture().await;
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_commitment_blocks SET block_hash=?1 WHERE height=1",
                params![[0xE7u8; 32].as_slice()],
            )
            .unwrap();
        }
        assert!(storage
            .record_commitment_chain_checkpoint(1)
            .await
            .unwrap_err()
            .contains("audit baseline is stale"));
        // Public integrity evidence remains hash-free even though the private
        // runtime baseline retains the value needed for this fail-closed gate.
        let value =
            serde_json::to_value(storage.record_commitment_chain_integrity_status()).unwrap();
        assert!(value.get("verified_tip_hash").is_none());
        assert!(value.get("tip_hash").is_none());
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
        let awaiting_proof = storage.record_commitment_sync_status();
        assert_eq!(awaiting_proof.state, "checkpointing");
        assert_eq!(awaiting_proof.consecutive_failures, 1);
        storage.record_commitment_sync_checkpoint_success(134, 9);
        storage.schedule_next_commitment_sync_poll(163);
        let recovered = storage.record_commitment_sync_status();
        assert_eq!(recovered.state, "current");
        assert_eq!(recovered.last_success_at, Some(134));
        assert_eq!(recovered.last_recovered_at, Some(134));
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
            privacy_safe_sync_error_code("checkpoint_http_status_503"),
            "checkpoint_http_status_error"
        );
        assert_eq!(
            privacy_safe_sync_error_code("signed_checkpoint_divergence"),
            "signed_checkpoint_divergence"
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

    #[test]
    fn test_checkpoint_runtime_is_aggregate_and_tracks_proof_lifecycle() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let initial = storage.record_commitment_checkpoint_status();
        assert_eq!(initial.state, "not_checked");
        assert_eq!(initial.proofs_verified_total, 0);
        assert_eq!(initial.last_round_state, "not_checked");
        assert_eq!(initial.last_round_at, None);

        storage.record_commitment_checkpoint_failure(100);
        storage.record_commitment_checkpoint_verified(110, "remote_ahead", 3, 4);
        storage.record_commitment_checkpoint_verified(120, "converged", 4, 4);
        storage.record_commitment_checkpoint_served(130);
        storage.record_commitment_checkpoint_witness_round(140, 4, 3, 2, 1, 1, 0, 1, 0);
        let status = storage.record_commitment_checkpoint_status();
        assert_eq!(status.state, "converged");
        assert_eq!(status.last_checked_at, Some(120));
        assert_eq!(status.last_converged_at, Some(120));
        assert_eq!(status.last_divergence_at, None);
        assert_eq!(status.last_failure_at, Some(100));
        assert_eq!(status.last_served_at, Some(130));
        assert_eq!(status.local_tip_height, Some(4));
        assert_eq!(status.remote_tip_height, Some(4));
        assert_eq!(status.proofs_verified_total, 2);
        assert_eq!(status.proofs_failed_total, 1);
        assert_eq!(status.divergences_total, 0);
        assert_eq!(status.requests_served_total, 1);
        assert_eq!(status.last_round_state, "partial");
        assert_eq!(status.last_round_at, Some(140));
        assert_eq!(status.last_round_eligible, 4);
        assert_eq!(status.last_round_attempted, 3);
        assert_eq!(status.last_round_verified, 2);
        assert_eq!(status.last_round_failed, 1);
        assert_eq!(status.last_round_converged, 1);
        assert_eq!(status.last_round_remote_ahead, 0);
        assert_eq!(status.last_round_remote_behind, 1);
        assert_eq!(status.last_round_diverged, 0);
        let value = serde_json::to_value(status).unwrap();
        for forbidden in [
            "peer",
            "block_hash",
            "tip_hash",
            "signature",
            "request_id",
            "evidence_digest",
            "endpoint",
            "owner",
            "payload",
        ] {
            assert!(value.get(forbidden).is_none(), "unexpected {forbidden}");
        }
    }

    #[test]
    fn test_checkpoint_serving_cannot_create_outbound_evidence() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();

        storage.record_commitment_checkpoint_served(50);

        let status = storage.record_commitment_checkpoint_status();
        assert_eq!(status.state, "not_checked");
        assert_eq!(status.last_checked_at, None);
        assert_eq!(status.last_converged_at, None);
        assert_eq!(status.last_divergence_at, None);
        assert_eq!(status.last_failure_at, None);
        assert_eq!(status.last_served_at, Some(50));
        assert_eq!(status.local_tip_height, None);
        assert_eq!(status.remote_tip_height, None);
        assert_eq!(status.proofs_verified_total, 0);
        assert_eq!(status.proofs_failed_total, 0);
        assert_eq!(status.divergences_total, 0);
        assert_eq!(status.requests_served_total, 1);
        assert_eq!(status.observation_freshness, "unavailable");
        assert_eq!(status.observation_age_seconds, None);
        assert_eq!(
            status.freshness_window_seconds,
            CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS
        );
        assert_eq!(status.last_round_state, "not_checked");
        assert_eq!(status.last_round_at, None);
    }

    #[test]
    fn test_checkpoint_witness_round_state_never_overclaims_consensus() {
        assert_eq!(checkpoint_witness_round_state(0, 0, 0, 0, 0), "unavailable");
        assert_eq!(checkpoint_witness_round_state(3, 0, 3, 0, 0), "unverified");
        assert_eq!(checkpoint_witness_round_state(3, 2, 1, 0, 0), "partial");
        assert_eq!(
            checkpoint_witness_round_state(3, 3, 0, 0, 0),
            "shared_prefix"
        );
        assert_eq!(checkpoint_witness_round_state(3, 3, 0, 1, 0), "attention");
        assert_eq!(checkpoint_witness_round_state(3, 3, 0, 0, 1), "attention");
    }

    #[test]
    fn test_checkpoint_observation_freshness_is_age_bounded() {
        assert_eq!(
            checkpoint_observation_freshness(None, 1_000),
            ("unavailable", None)
        );
        assert_eq!(
            checkpoint_observation_freshness(Some(1_001), 1_000),
            ("unavailable", None)
        );
        assert_eq!(
            checkpoint_observation_freshness(Some(1_000), 1_000),
            ("fresh", Some(0))
        );
        assert_eq!(
            checkpoint_observation_freshness(
                Some(1_000),
                1_000 + CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
            ),
            ("fresh", Some(CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS))
        );
        assert_eq!(
            checkpoint_observation_freshness(
                Some(1_000),
                1_001 + CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS,
            ),
            ("stale", Some(CHECKPOINT_OBSERVATION_FRESHNESS_SECONDS + 1))
        );
    }

    #[tokio::test]
    async fn test_checkpoint_evidence_persists_and_reaudits_signed_frame() {
        let (storage, block) = commitment_audit_fixture().await;
        let responder = IdentityKeyPair::generate();
        let observed_at = 1_700_500_001;
        let frame = signed_checkpoint_response_frame(
            &responder,
            [0x31; 16],
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

        let audit = storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        assert_eq!(audit.evidence_records, 1);
        assert_eq!(audit.divergence_evidence_records, 0);
        assert_eq!(audit.last_evidence_at, Some(observed_at));
        let status = storage.record_commitment_checkpoint_status();
        assert_eq!(status.evidence_state, "verified");
        assert_eq!(status.evidence_records, 1);
        assert_eq!(status.last_evidence_at, Some(observed_at));
    }

    #[tokio::test]
    async fn test_checkpoint_evidence_audit_fails_closed_after_sqlite_tampering() {
        let (storage, block) = commitment_audit_fixture().await;
        let responder = IdentityKeyPair::generate();
        let observed_at = 1_700_500_010;
        let frame = signed_checkpoint_response_frame(
            &responder,
            [0x32; 16],
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
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_checkpoint_evidence SET signed_response=?1",
                params![vec![MEMCHAIN_MAGIC, 0x00]],
            )
            .unwrap();
        }
        assert!(storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap_err()
            .contains("digest mismatch"));
        assert_eq!(
            storage.record_commitment_checkpoint_status().evidence_state,
            "invalid"
        );
    }

    #[tokio::test]
    async fn test_checkpoint_evidence_duplicate_conflict_fails_closed() {
        let (storage, block) = commitment_audit_fixture().await;
        let responder = IdentityKeyPair::generate();
        let observed_at = 1_700_500_020;
        let frame = signed_checkpoint_response_frame(
            &responder,
            [0x33; 16],
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
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "UPDATE record_checkpoint_evidence SET relation='remote_ahead'",
                [],
            )
            .unwrap();
        }
        assert!(storage
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
            .unwrap_err()
            .contains("conflicts with verified frame"));
        assert_eq!(
            storage
                .record_commitment_checkpoint_status()
                .evidence_persistence_failures_total,
            1
        );
    }

    #[tokio::test]
    async fn test_checkpoint_evidence_is_visible_through_wal_and_survives_restart() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("checkpoint-restart.db");
        let writer = MemoryStorage::open(&path, None).unwrap();
        let observed_at = 1_700_500_100;
        let (block, _, _) = seed_checkpoint_evidence(&writer, observed_at).await;

        // Open a second storage while the writer connection remains alive. This
        // proves the committed proof is visible through SQLite WAL, not merely
        // flushed as a side effect of closing the first process.
        let wal_reader = MemoryStorage::open(&path, None).unwrap();
        let chain = wal_reader.audit_record_commitment_chain().await.unwrap();
        assert_eq!(chain.block_count, 1);
        assert_eq!(chain.tip_height, 1);
        assert_eq!(
            wal_reader.record_commitment_chain_tip().await.1,
            block.hash()
        );
        let wal_audit = wal_reader
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        assert_eq!(wal_audit.evidence_records, 1);
        assert_eq!(wal_audit.last_evidence_at, Some(observed_at));
        drop(wal_reader);
        drop(writer);

        // A later clean process must independently re-establish both audit
        // baselines instead of trusting runtime state from the prior process.
        let reopened = MemoryStorage::open(&path, None).unwrap();
        assert_eq!(
            reopened.record_commitment_chain_integrity_status().state,
            "not_verified"
        );
        assert_eq!(
            reopened
                .record_commitment_checkpoint_status()
                .evidence_state,
            "not_audited"
        );
        reopened.audit_record_commitment_chain().await.unwrap();
        let reopened_audit = reopened
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        assert_eq!(reopened_audit.evidence_records, 1);
        assert_eq!(
            reopened
                .record_commitment_checkpoint_status()
                .evidence_state,
            "verified"
        );
    }

    #[tokio::test]
    async fn test_v8_to_v9_file_migration_preserves_commitment_chain() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("checkpoint-migration.db");
        let original = MemoryStorage::open(&path, None).unwrap();
        let (block, _, _) = seed_checkpoint_evidence(&original, 1_700_500_200).await;
        drop(original);

        // Reconstruct the exact relevant v8 boundary: the commitment chain is
        // present, but the v9 evidence table does not yet exist.
        let legacy = rusqlite::Connection::open(&path).unwrap();
        legacy
            .execute_batch(
                "DROP TABLE record_checkpoint_evidence;
                 UPDATE schema_version SET version=8;",
            )
            .unwrap();
        drop(legacy);

        let migrated = MemoryStorage::open(&path, None).unwrap();
        let version: u32 = {
            let conn = migrated.conn_lock().await;
            conn.query_row("SELECT version FROM schema_version", [], |row| row.get(0))
                .unwrap()
        };
        assert_eq!(version, 9);
        let chain = migrated.audit_record_commitment_chain().await.unwrap();
        assert_eq!(chain.block_count, 1);
        assert_eq!(migrated.record_commitment_chain_tip().await.1, block.hash());
        let evidence = migrated
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        assert_eq!(evidence.evidence_records, 0);
    }

    #[tokio::test]
    async fn test_reopened_checkpoint_evidence_rejects_disk_tampering() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("checkpoint-tamper.db");
        let original = MemoryStorage::open(&path, None).unwrap();
        seed_checkpoint_evidence(&original, 1_700_500_300).await;
        drop(original);

        let tamper = rusqlite::Connection::open(&path).unwrap();
        tamper
            .execute(
                "UPDATE record_checkpoint_evidence SET signed_response=?1",
                params![vec![MEMCHAIN_MAGIC, 0x00]],
            )
            .unwrap();
        drop(tamper);

        let reopened = MemoryStorage::open(&path, None).unwrap();
        reopened.audit_record_commitment_chain().await.unwrap();
        assert!(reopened
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap_err()
            .contains("digest mismatch"));
        assert_eq!(
            reopened
                .record_commitment_checkpoint_status()
                .evidence_state,
            "invalid"
        );
    }

    #[tokio::test]
    async fn test_checkpoint_evidence_rejects_invalid_input_and_tracks_failure() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let frame = vec![MEMCHAIN_MAGIC, 0x01];
        let wrong_digest = [0u8; 32];
        assert!(storage
            .persist_record_commitment_checkpoint_evidence(
                1,
                "converged",
                0,
                0,
                0,
                &wrong_digest,
                &frame,
            )
            .await
            .unwrap_err()
            .contains("digest mismatch"));
        let digest: [u8; 32] = Sha256::digest(&frame).into();
        assert!(storage
            .persist_record_commitment_checkpoint_evidence(1, "unknown", 0, 0, 0, &digest, &frame,)
            .await
            .unwrap_err()
            .contains("relation is invalid"));
        assert_eq!(
            storage
                .record_commitment_checkpoint_status()
                .evidence_persistence_failures_total,
            2
        );
    }

    #[tokio::test]
    async fn test_checkpoint_evidence_capacity_preserves_divergence() {
        let (storage, block) = commitment_audit_fixture().await;
        let responder = IdentityKeyPair::generate();
        let diverged_at = 1_700_510_000;
        let fork_hash = [0xE7; 32];
        let divergent = signed_checkpoint_response_frame(
            &responder,
            [0x40; 16],
            diverged_at,
            1,
            fork_hash,
            1,
            fork_hash,
        );
        let divergent_digest: [u8; 32] = Sha256::digest(&divergent).into();
        storage
            .persist_record_commitment_checkpoint_evidence(
                diverged_at,
                "diverged",
                1,
                1,
                1,
                &divergent_digest,
                &divergent,
            )
            .await
            .unwrap();

        for index in 0..CHECKPOINT_EVIDENCE_CAPACITY + 8 {
            let observed_at = diverged_at + 1 + index as u64;
            let mut request_id = [0u8; 16];
            request_id[..8].copy_from_slice(&(index as u64).to_le_bytes());
            let frame = signed_checkpoint_response_frame(
                &responder,
                request_id,
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
        }
        let audit = storage
            .audit_record_commitment_checkpoint_evidence()
            .await
            .unwrap();
        assert_eq!(audit.evidence_records, CHECKPOINT_EVIDENCE_CAPACITY as u64);
        assert_eq!(audit.divergence_evidence_records, 1);
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
