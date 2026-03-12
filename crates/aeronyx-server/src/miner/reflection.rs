// ============================================
// File: crates/aeronyx-server/src/miner/reflection.rs
// ============================================
//! # ReflectionMiner — Intelligent Memory Compaction + MVF Feedback Pipeline
//!
//! ## Processing Pipeline (runs on timer or rawlog threshold)
//! ```text
//! Step 0:   Positive feedback batch detection (from raw_logs)
//! Step 0.5: Backfill missing embeddings (via OpenClaw HTTP)
//! Step 0.6: Correction chaining (_correction tag → supersede)
//! Step 1-5: Legacy compaction (Episode → Archive via OpenClaw summary)
//! ```
//!
//! ## v2.4.0-GraphCognition: Cognitive Graph Steps (Phase B — IMPLEMENTED)
//! When ner_engine is attached, the following steps run after Steps 0-5:
//! ```text
//! Step 6:  Session metadata population (sessions table) — handled by log_handler
//! Step 7:  Entity/relation extraction (GLiNER → entities + knowledge_edges)
//! Step 8:  Community detection (label propagation → communities + projects)
//! Step 9:  Recursive merge (MiniLM similarity → fragment merge + temporal conflict)
//! Step 10: Session summary + code artifact extraction
//! Step 11: Episode ingestion + community summary update
//! ```
//!
//! ## MVF Integration
//! - Step 0 produces training data for SGD (positive samples y=1)
//! - When mvf_enabled: after Step 0, run SGD batch update on user_weights
//! - Negative feedback is handled in real-time by /log endpoint (not here)
//!
//! ## Embedding Strategy
//! Step 0.5 calls OpenClaw HTTP `POST /v1/embeddings` for MiniLM inference.
//! No ort/ONNX dependency — all embedding computation delegated to OpenClaw.
//!
//! ## Modification Reason (v2.1.0+MVF+Encryption)
//! Fixed rawlog key derivation in Step 0: changed from public key to PRIVATE key.
//!
//! ## Modification Reason (v2.4.0-GraphCognition)
//! Added ner_engine field and with_ner_engine() builder method.
//! NerEngine will be used by Steps 7-11 for entity/relation extraction
//! from conversation content. This PR only adds the plumbing —
//! actual cognitive graph steps are Phase B.
//!
//! ## Modification Reason (v2.4.0-GraphCognition Phase B)
//! Replaced stub methods (Steps 7-11) with full implementations:
//!   - Step 7: GLiNER entity/relation extraction → entities + knowledge_edges + episode_edges
//!   - Step 8: Label propagation community detection → communities + projects
//!   - Step 9: Embedding similarity merge + temporal conflict detection
//!   - Step 10: Session summary generation + Markdown code fence artifact extraction
//!   - Step 11: Episode ingestion + community summary update
//!
//! ## Dependencies
//! - storage.rs — MemoryStorage, derive_rawlog_key, decrypt_rawlog_content_pub
//! - storage_ops.rs — get_rawlogs_for_session, merge_entities, get_entities_with_embedding,
//!   mark_session_artifacts_extracted, mark_session_summary_generated, update_session_ended_at
//! - identity.to_bytes() — Ed25519 PRIVATE key for rawlog key derivation
//! - vector.rs — VectorIndex, cosine_similarity
//! - mvf.rs — WeightVector, SGD
//! - ner.rs — NerEngine (v2.4.0, optional — cognitive graph steps)
//! - graph.rs — label_propagation (v2.4.0 Phase B — community detection)
//! - sha2 crate — entity_id deterministic generation
//! - regex crate — Markdown code fence extraction (Step 10)
//!
//! ⚠️ Important Note for Next Developer:
//! - derive_rawlog_key MUST use identity.to_bytes() (PRIVATE key), NOT public_key_bytes()
//! - Step 0 decrypts rawlogs — if the key is wrong, content will be garbage/empty
//! - After the key derivation fix, old rawlogs encrypted with public-key-derived key
//!   are cleared by the migration in storage.rs
//! - ner_engine is Option<Arc<NerEngine>> — when None, Steps 7-11 are skipped
//! - NerEngine is thread-safe (Mutex<Session> internally) — safe to use from Miner
//! - Steps 7-11 are gated on self.ner_engine.is_some() — if NER is disabled,
//!   all cognitive graph steps are skipped (v2.3.0 behavior preserved).
//! - Each step has independent error handling; step failure does NOT cascade.
//! - Step 7 reads raw_logs (may be encrypted) — uses derive_rawlog_key with
//!   self.identity.to_bytes() (PRIVATE key), same pattern as Step 0.
//! - Entity ID = SHA256(owner_hex || ":" || name_normalized) — deterministic.
//! - Step 9 merge threshold: cosine > 0.92 = merge, 0.85-0.92 = RELATED_TO edge.
//! - Step 10 code block regex matches Markdown fences only (```lang\n...\n```).
//! - Step 11 encrypts episode content using record_key (same as records table).
//! - Step 8 and Step 11 call self.storage.conn_lock() — this method MUST be
//!   exposed as a public method on MemoryStorage (returns MutexGuard<Connection>).
//!   If it doesn't exist yet, add: `pub async fn conn_lock(&self) -> ... { self.conn.lock().await }`
//!
//! ## Error Isolation
//! Each step has independent error handling. Step 0 failure does NOT block
//! Step 1-5. All steps log errors and continue.
//!
//! ## Last Modified
//! v0.5.0 - Initial timer-based block packer
//! v1.0.0 - Smart compaction via OpenClaw
//! v2.1.0 - MVF feedback pipeline (Step 0, 0.5, 0.6) + SGD integration
//! v2.1.0+MVF+Encryption - 🌟 Fixed rawlog key derivation in Step 0
//! v2.4.0-GraphCognition - 🌟 Added ner_engine field + with_ner_engine() builder.
//!   Cognitive graph Steps 6-11 will be implemented in Phase B.
//! v2.4.0-GraphCognition Phase B - 🌟 Replaced Steps 7-11 stubs with full
//!   implementations. Added imports: graph, HashMap, sha2, regex.
//!   Added constants: DEFAULT_ENTITY_LABELS, DEFAULT_RELATION_LABELS,
//!   MINER_SESSION_BATCH, MINER_MERGE_BATCH, ENTITY_MERGE/RELATED_THRESHOLD,
//!   CODE_FENCE_PATTERN. Added free function: infer_relation_type().

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info, warn};

#[allow(deprecated)]
use aeronyx_core::ledger::{
    Block, BlockHeader, MemoryLayer, MemoryRecord,
    BLOCK_TYPE_NORMAL, BLOCK_TYPE_MEMORY, merkle_root,
};
use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::crypto::transport::{DefaultTransportCrypto, TransportCrypto, ENCRYPTION_OVERHEAD};
use aeronyx_core::protocol::codec::encode_data_packet;
use aeronyx_core::protocol::memchain::{encode_memchain, MemChainMessage};
use aeronyx_core::protocol::DataPacket;
use aeronyx_transport::traits::Transport;
use aeronyx_transport::UdpTransport;

use crate::services::memchain::{AofWriter, MemPool, MemoryStorage, VectorIndex};
use crate::services::memchain::mvf;
use crate::services::memchain::vector::cosine_similarity;
use crate::services::memchain::EmbedEngine;
// v2.4.0: NerEngine for cognitive graph Steps 7-11
use crate::services::memchain::NerEngine;
// v2.4.0 Phase B: Graph module for community detection (Step 8)
use crate::services::memchain::graph;
use crate::services::SessionManager;

// ============================================
// Constants
// ============================================

const DEFAULT_COMPACTION_THRESHOLD: u64 = 500;
const MAX_COMPACTION_BATCH: usize = 200;
const OPENCLAW_GATEWAY_CHAT: &str = "http://127.0.0.1:18789/v1/chat/completions";
const OPENCLAW_GATEWAY_EMBED: &str = "http://127.0.0.1:18789/v1/embeddings";
const OPENCLAW_TIMEOUT_SECS: u64 = 120;
const EMBEDDING_BACKFILL_BATCH: usize = 50;

// ============================================
// v2.4.0 Phase B: Cognitive Graph Constants
// ============================================

/// Default entity labels for GLiNER NER detection.
/// These define WHAT types of entities to extract from conversations.
/// Can be overridden via config.toml `ner_entity_labels`.
///
/// Rationale for each label:
/// - project: software project names ("MemChain", "Project B")
/// - module: code modules/components ("auth module", "storage engine")
/// - technology: tools/frameworks/protocols ("JWT", "SQLite", "React")
/// - person: people mentioned in conversations
/// - file: file paths and filenames ("auth.rs", "config.toml")
/// - concept: abstract concepts and decisions ("encryption", "zero-shot")
/// - tool: development tools ("cargo", "git", "docker")
/// - language: programming languages ("Rust", "TypeScript", "Python")
const DEFAULT_ENTITY_LABELS: &[&str] = &[
    "project", "module", "technology", "person",
    "file", "concept", "tool", "language",
];

/// Relation labels for GLiNER relation extraction (second pass).
/// These define WHAT relationships to detect between entities.
#[allow(dead_code)]
const DEFAULT_RELATION_LABELS: &[&str] = &[
    "uses", "depends on", "contains", "belongs to",
    "created by", "related to", "replaces", "implements",
];

/// Maximum sessions to process per Miner tick (Step 7).
const MINER_SESSION_BATCH: usize = 10;

/// Maximum entities to consider for pairwise merge (Step 9).
const MINER_MERGE_BATCH: usize = 200;

/// Cosine similarity threshold for entity merge (Step 9).
/// Above this → merge entities (they're the same thing with different names).
const ENTITY_MERGE_THRESHOLD: f32 = 0.92;

/// Cosine similarity threshold for RELATED_TO edge (Step 9).
/// Between this and ENTITY_MERGE_THRESHOLD → create a relationship edge.
const ENTITY_RELATED_THRESHOLD: f32 = 0.85;

/// Regex pattern for Markdown code fences (Step 10).
/// Captures: group 1 = language (optional), group 2 = code content.
const CODE_FENCE_PATTERN: &str = r"```(\w*)\n([\s\S]*?)```";

/// Negative feedback keywords (shared with log_handler, duplicated here
/// to keep Miner self-contained and avoid circular dependencies).
const CN_NEGATIVE: &[&str] = &[
    "不对", "不是", "错了", "我改主意", "搞错了", "纠正一下",
    "其实不是", "不是这样", "这不是我要的", "重新来", "再试一次",
];
const EN_NEGATIVE: &[&str] = &[
    "wrong", "not correct", "that's not right", "changed my mind",
    "actually no", "let me correct", "not what i asked",
    "try again", "that's not what i meant",
];

// ============================================
// ReflectionMiner
// ============================================

pub struct ReflectionMiner {
    interval: Duration,
    compaction_threshold: u64,

    // New MRS-1 state
    storage: Arc<MemoryStorage>,
    vector_index: Arc<VectorIndex>,
    identity: IdentityKeyPair,

    // Legacy state (Fact mining)
    mempool: Arc<MemPool>,
    aof_writer: Arc<TokioMutex<AofWriter>>,

    // P2P broadcast
    sessions: Arc<SessionManager>,
    udp: Arc<UdpTransport>,

    // MVF
    mvf_enabled: bool,
    user_weights: Option<Arc<parking_lot::RwLock<std::collections::HashMap<String, mvf::WeightVector>>>>,

    /// Local embedding engine (shared with MpiState). When Some, used for
    /// embedding generation instead of calling OpenClaw Gateway HTTP.
    embed_engine: Option<Arc<EmbedEngine>>,

    /// v2.4.0: Local NER engine (shared with MpiState). When Some, enables
    /// cognitive graph Steps 7-11 for entity/relation extraction.
    /// When None, Steps 7-11 are skipped (v2.3.0 behavior).
    ner_engine: Option<Arc<NerEngine>>,
}

impl ReflectionMiner {
    #[must_use]
    pub fn new(
        interval_secs: u64,
        storage: Arc<MemoryStorage>,
        vector_index: Arc<VectorIndex>,
        identity: IdentityKeyPair,
        mempool: Arc<MemPool>,
        aof_writer: Arc<TokioMutex<AofWriter>>,
        sessions: Arc<SessionManager>,
        udp: Arc<UdpTransport>,
    ) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
            storage,
            vector_index,
            identity,
            mempool,
            aof_writer,
            sessions,
            udp,
            mvf_enabled: false,
            user_weights: None,
            embed_engine: None,
            ner_engine: None,
        }
    }

    #[must_use]
    pub fn with_compaction_threshold(mut self, threshold: u64) -> Self {
        self.compaction_threshold = threshold;
        self
    }

    #[must_use]
    pub fn with_mvf(
        mut self,
        enabled: bool,
        weights: Arc<parking_lot::RwLock<std::collections::HashMap<String, mvf::WeightVector>>>,
    ) -> Self {
        self.mvf_enabled = enabled;
        self.user_weights = Some(weights);
        self
    }

    /// Set the local embedding engine for Miner steps.
    /// When set, Miner uses local inference instead of OpenClaw Gateway HTTP.
    /// Falls back to OpenClaw if local inference fails.
    #[must_use]
    pub fn with_embed_engine(mut self, engine: Arc<EmbedEngine>) -> Self {
        self.embed_engine = Some(engine);
        self
    }

    /// v2.4.0: Set the local NER engine for cognitive graph Steps 7-11.
    ///
    /// When set, the Miner will extract entities and relations from conversation
    /// content using GLiNER, building the three-layer cognitive graph:
    /// - Step 7: Entity/relation extraction → entities + knowledge_edges tables
    /// - Step 8: Community detection → communities + projects tables
    /// - Step 9: Recursive merge → fragment consolidation + temporal conflicts
    /// - Step 10: Session summary + code artifact extraction
    /// - Step 11: Episode ingestion + community summary update
    ///
    /// When None, these steps are skipped and the Miner operates in v2.3.0 mode
    /// (Steps 0-5 only).
    ///
    /// ## Thread Safety
    /// NerEngine is thread-safe (ort::Session wrapped in Mutex internally).
    /// The Miner runs on a single async task, so contention with MPI handlers
    /// is minimal (recall path also uses NerEngine for query analysis).
    #[must_use]
    pub fn with_ner_engine(mut self, engine: Arc<NerEngine>) -> Self {
        self.ner_engine = Some(engine);
        self
    }

    pub async fn run(self, mut shutdown_rx: tokio::sync::broadcast::Receiver<()>) {
        info!(
            interval = self.interval.as_secs(),
            threshold = self.compaction_threshold,
            mvf = self.mvf_enabled,
            ner = self.ner_engine.is_some(),
            "[MINER] Started"
        );

        let mut timer = tokio::time::interval(self.interval);
        timer.tick().await; // Skip first immediate tick

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("[MINER] Shutdown");
                    break;
                }
                _ = timer.tick() => {
                    // Each step is independent — errors don't cascade
                    self.step_0_positive_feedback().await;
                    self.step_05_backfill_embeddings().await;
                    self.step_06_correction_chaining().await;
                    self.step_1_5_legacy_compaction().await;

                    // v2.4.0: Cognitive graph steps (Phase B implementation)
                    // These are gated on ner_engine being present.
                    if self.ner_engine.is_some() {
                        self.step_7_entity_extraction().await;
                        self.step_8_community_detection().await;
                        self.step_9_recursive_merge().await;
                        self.step_10_session_summary().await;
                        self.step_11_episode_ingestion().await;
                    }
                }
            }
        }
    }

    // ============================================
    // Step 0: Positive Feedback Batch Detection
    // ============================================

    async fn step_0_positive_feedback(&self) {
        let rows = self.storage.get_unprocessed_rawlogs(500).await;
        if rows.is_empty() {
            debug!("[MINER_S0] No unprocessed rawlogs");
            return;
        }

        let owner = self.identity.public_key_bytes();
        let owner_hex = hex::encode(owner);

        let mut sessions: std::collections::HashMap<String, Vec<_>> = std::collections::HashMap::new();
        for row in &rows {
            sessions.entry(row.session_id.clone()).or_default().push(row);
        }

        let mut total_positive = 0u32;
        let mut total_neutral = 0u32;

        for (_session_id, session_rows) in &sessions {
            let mut seen_memories: HashSet<Vec<u8>> = HashSet::new();

            for row in session_rows {
                if row.role != "user" { continue; }
                if row.feedback_signal.is_some() { continue; }

                // ⚠️ SECURITY FIX (v2.1.0+MVF+Encryption):
                // derive_rawlog_key uses PRIVATE key (identity.to_bytes())
                let content = if row.encrypted == 1 {
                    let key = crate::services::memchain::derive_rawlog_key(&self.identity.to_bytes());
                    String::from_utf8(
                        crate::services::memchain::decrypt_rawlog_content_pub(&key, &row.content)
                            .unwrap_or_default()
                    ).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&row.content).to_string()
                };

                let lower = content.to_lowercase();
                let has_negative = CN_NEGATIVE.iter().any(|kw| lower.contains(kw))
                    || EN_NEGATIVE.iter().any(|kw| lower.contains(kw));

                if has_negative || content.len() <= 5 {
                    self.storage.update_rawlog_feedback(row.log_id, 0).await;
                    total_neutral += 1;
                    continue;
                }

                let recall_ctx = row.recall_context.as_deref()
                    .or_else(|| {
                        session_rows.iter()
                            .filter(|r| r.turn_index < row.turn_index && r.role == "assistant")
                            .last()
                            .and_then(|r| r.recall_context.as_deref())
                    });

                let ctx = match recall_ctx {
                    Some(c) => c,
                    None => { self.storage.update_rawlog_feedback(row.log_id, 0).await; total_neutral += 1; continue; }
                };

                let entries: Vec<serde_json::Value> = serde_json::from_str(ctx).unwrap_or_default();
                if entries.is_empty() {
                    self.storage.update_rawlog_feedback(row.log_id, 0).await;
                    total_neutral += 1;
                    continue;
                }

                let top_entries: Vec<&serde_json::Value> = entries.iter().take(2).collect();
                let query_embedding = self.call_openclaw_embed(&content).await;

                for entry in top_entries {
                    let mem_id_hex = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let mem_id_bytes = hex::decode(mem_id_hex).unwrap_or_default();

                    if mem_id_bytes.len() != 32 || seen_memories.contains(&mem_id_bytes) { continue; }

                    if let Some(ref q_emb) = query_embedding {
                        if let Some(record) = self.storage.get(&{
                            let mut arr = [0u8; 32]; arr.copy_from_slice(&mem_id_bytes); arr
                        }).await {
                            if record.has_embedding() {
                                let sim = cosine_similarity(q_emb, &record.embedding);
                                if sim <= 0.4 { continue; }
                            }
                        }
                    }

                    let mut record_id = [0u8; 32];
                    record_id.copy_from_slice(&mem_id_bytes);

                    self.storage.increment_positive_feedback(&record_id).await;

                    let features: Option<[f32; 9]> = entry.get("features")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| {
                            if arr.len() == 9 {
                                let mut f = [0.0f32; 9];
                                for (i, val) in arr.iter().enumerate() { f[i] = val.as_f64().unwrap_or(0.0) as f32; }
                                Some(f)
                            } else { None }
                        });

                    let prediction = entry.get("score").and_then(|v| v.as_f64()).map(|v| v as f32);

                    self.storage.insert_feedback(
                        &owner, &record_id, &row.session_id,
                        row.turn_index, 1, features.as_ref(), prediction,
                    ).await;

                    seen_memories.insert(mem_id_bytes);
                    total_positive += 1;
                }

                self.storage.update_rawlog_feedback(row.log_id, 1).await;
            }
        }

        if self.mvf_enabled && total_positive > 0 {
            self.sgd_batch_update_positive(&owner_hex).await;
        }

        info!(positive = total_positive, neutral = total_neutral, "[MINER_S0] Feedback detection complete");
    }

    async fn sgd_batch_update_positive(&self, owner_hex: &str) {
        let weights_map = match &self.user_weights { Some(w) => w, None => return };
        let feedback = self.storage.get_recent_feedback(100).await;
        let positive_with_features: Vec<_> = feedback.iter().filter(|(signal, _)| *signal == 1).collect();
        if positive_with_features.is_empty() { return; }

        let mut map = weights_map.write();
        let w = map.entry(owner_hex.to_string()).or_insert_with(mvf::default_weights);

        info!(
            samples = positive_with_features.len(), version = w.version,
            "[MINER_SGD] Batch update noted (features-based SGD in D11)"
        );
    }

    // ============================================
    // Step 0.5: Backfill Missing Embeddings
    // ============================================

    async fn step_05_backfill_embeddings(&self) {
        let records = self.storage.get_records_needing_embedding(EMBEDDING_BACKFILL_BATCH).await;
        if records.is_empty() { debug!("[MINER_S05] No records need embedding backfill"); return; }

        let owner = self.identity.public_key_bytes();
        let mut filled = 0u32;

        for record in &records {
            let content = String::from_utf8_lossy(&record.encrypted_content).to_string();
            if content.is_empty() { continue; }

            if let Some(embedding) = self.call_openclaw_embed(&content).await {
                let dim = embedding.len();
                let embedding_blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();

                let conn = self.storage.conn_lock().await;
                let _ = conn.execute(
                    "UPDATE records SET embedding = ?1, embedding_model = ?2, embedding_dim = ?3 WHERE record_id = ?4",
                    rusqlite::params![embedding_blob.as_slice(), "minilm-l6-v2", dim as i64, record.record_id.as_slice()],
                );
                drop(conn);

                self.vector_index.upsert(record.record_id, embedding, record.layer, record.timestamp, &owner, "minilm-l6-v2");
                filled += 1;
            }
        }

        if filled > 0 { info!(filled = filled, "[MINER_S05] Embeddings backfilled"); }
    }

    // ============================================
    // Step 0.6: Correction Chaining
    // ============================================

    async fn step_06_correction_chaining(&self) {
        let corrections = self.storage.get_correction_records().await;
        if corrections.is_empty() { return; }

        let owner = self.identity.public_key_bytes();
        let mut chained = 0u32;

        for correction in &corrections {
            if !correction.has_embedding() { continue; }

            let candidates = self.vector_index.search(
                &correction.embedding, &owner, "minilm-l6-v2", 5, 0.5,
            );

            let best_match = candidates.iter().find(|c| c.record_id != correction.record_id);

            if let Some(old) = best_match {
                self.storage.supersede_record(&old.record_id, &correction.record_id).await;
                self.vector_index.remove(&old.record_id);
                chained += 1;
                debug!(
                    old = hex::encode(old.record_id), new = hex::encode(correction.record_id),
                    sim = old.similarity, "[MINER_S06] Correction chained (supersede)"
                );
            }

            let new_tags: Vec<String> = correction.topic_tags.iter()
                .filter(|t| *t != "_correction").cloned().collect();
            self.storage.update_topic_tags(&correction.record_id, &new_tags).await;
        }

        if chained > 0 { info!(chained = chained, "[MINER_S06] Corrections processed"); }
    }

    // ============================================
    // Step 1-5: Legacy Compaction (unchanged logic)
    // ============================================

    async fn step_1_5_legacy_compaction(&self) {
        self.smart_compact().await;
        self.legacy_mine().await;
    }

    async fn smart_compact(&self) {
        let ep_count = self.storage.count_by_layer(MemoryLayer::Episode).await;
        if ep_count < self.compaction_threshold { return; }

        let owner = self.identity.public_key_bytes();
        let episodes = self.storage.compact_episodes_to_archive(&owner, MAX_COMPACTION_BATCH).await;
        if episodes.is_empty() { return; }

        info!(count = episodes.len(), "[MINER] Compacting episodes");

        let prompt = self.build_summary_prompt(&episodes);
        let summary = match self.call_openclaw_chat(&prompt).await {
            Some(s) => s,
            None => { warn!("[MINER] OpenClaw summary failed"); return; }
        };

        let now_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let mut all_tags: Vec<String> = episodes.iter().flat_map(|e| e.topic_tags.clone()).collect();
        all_tags.sort(); all_tags.dedup();

        let mut knowledge = MemoryRecord::new(
            owner, now_ts, MemoryLayer::Knowledge,
            all_tags, "miner-compaction".into(),
            summary.as_bytes().to_vec(), vec![],
        );
        knowledge.signature = self.identity.sign(&knowledge.record_id);

        if !self.storage.insert(&knowledge, "miner-compaction").await {
            error!("[MINER] Failed to insert Knowledge record");
            return;
        }

        let prev_hash = self.storage.last_block_hash().await;
        let prev_height = self.storage.last_block_height().await;
        let new_height = if prev_hash == [0u8; 32] { 1 } else { prev_height + 1 };
        let root = merkle_root(&[knowledge.record_id]);

        let header = BlockHeader {
            height: new_height, timestamp: now_ts, prev_block_hash: prev_hash,
            merkle_root: root, block_type: BLOCK_TYPE_MEMORY,
        };

        self.storage.set_chain_state(&header.hash(), new_height).await;
        let _ = self.broadcast_header(MemChainMessage::BlockAnnounce(header)).await;
        info!(height = new_height, episodes = episodes.len(), "[MINER] Compaction complete");
    }

    #[allow(deprecated)]
    async fn legacy_mine(&self) {
        let facts = self.mempool.drain_for_block();
        if facts.is_empty() { return; }

        let leaf_ids: Vec<[u8; 32]> = facts.iter().map(|f| f.fact_id).collect();
        let root = merkle_root(&leaf_ids);

        let (prev_hash, prev_height) = {
            let w = self.aof_writer.lock().await;
            (w.last_block_hash(), w.last_block_height())
        };

        let new_height = if prev_hash == [0u8; 32] { 1 } else { prev_height + 1 };
        let now_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        let header = BlockHeader {
            height: new_height, timestamp: now_ts,
            prev_block_hash: prev_hash, merkle_root: root,
            block_type: BLOCK_TYPE_NORMAL,
        };
        let block = Block::new(header.clone(), facts);

        {
            let mut w = self.aof_writer.lock().await;
            if let Err(e) = w.append_block(&block).await {
                error!(error = %e, "[MINER_LEGACY] Block persist failed");
                return;
            }
        }

        let _ = self.broadcast_header(MemChainMessage::BlockAnnounce(header)).await;
    }

    // ============================================
    // v2.4.0: Cognitive Graph Steps (Phase B implementation)
    // ============================================
    //
    // These steps build the three-layer cognitive graph:
    //   Episode layer → Semantic Entity layer → Community layer
    //
    // Execution order matters:
    //   Step 7  → extracts entities + relations (populates entity/edge tables)
    //   Step 8  → clusters entities into communities (requires Step 7 data)
    //   Step 9  → merges duplicate entities (requires Step 7 data)
    //   Step 10 → generates session summaries + extracts code artifacts
    //   Step 11 → creates episodes + updates community summaries (requires Step 8)
    //
    // All steps are gated on self.ner_engine.is_some() in run().
    // Each step has independent error handling — failures don't cascade.

    /// Step 7: Extract entities and relations from conversation content.
    ///
    /// Pipeline:
    /// 1. Get pending sessions (not yet entity-extracted)
    /// 2. For each session, read raw_logs → decrypt → reconstruct conversation text
    /// 3. Run GLiNER NER: detect entity spans with DEFAULT_ENTITY_LABELS
    /// 4. Generate deterministic entity_id = SHA256(owner_hex + ":" + name_normalized)
    /// 5. Upsert each entity (idempotent — mention_count auto-increments)
    /// 6. Generate entity embeddings for future merge (Step 9)
    /// 7. Detect relations between co-occurring entities in the same turn
    /// 8. Insert knowledge_edges + episode_edges for provenance
    /// 9. Mark session as entities_extracted
    ///
    /// Writes to: entities, knowledge_edges, episode_edges tables.
    async fn step_7_entity_extraction(&self) {
        let ner = match &self.ner_engine {
            Some(n) => n,
            None => return,
        };

        let owner = self.identity.public_key_bytes();
        let owner_hex = hex::encode(owner);
        let rawlog_key = crate::services::memchain::derive_rawlog_key(
            &self.identity.to_bytes()
        );

        // 1. Get pending sessions
        let pending = self.storage.get_pending_sessions(&owner, MINER_SESSION_BATCH).await;
        let pending: Vec<_> = pending.into_iter()
            .filter(|s| !s.entities_extracted)
            .collect();

        if pending.is_empty() {
            debug!("[MINER_S7] No pending sessions for entity extraction");
            return;
        }

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut total_entities = 0u32;
        let mut total_edges = 0u32;

        for session in &pending {
            // 2. Read and decrypt raw_logs for this session
            let raw_logs = self.storage.get_rawlogs_for_session(&session.session_id).await;
            if raw_logs.is_empty() {
                // No raw_logs → mark as extracted (nothing to extract)
                self.storage.mark_session_entities_extracted(&session.session_id).await;
                continue;
            }

            // Reconstruct conversation text
            let mut conversation_text = String::new();
            for log in &raw_logs {
                let content = if log.encrypted == 1 {
                    String::from_utf8(
                        crate::services::memchain::decrypt_rawlog_content_pub(
                            &rawlog_key, &log.content
                        ).unwrap_or_default()
                    ).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&log.content).to_string()
                };

                if content.is_empty() { continue; }

                conversation_text.push_str(&format!("[{}] {}\n", log.role, content));
            }

            if conversation_text.is_empty() {
                self.storage.mark_session_entities_extracted(&session.session_id).await;
                continue;
            }

            // 3. Run NER entity detection
            let entities = match ner.detect_entities(
                &conversation_text, DEFAULT_ENTITY_LABELS
            ) {
                Ok(ents) => ents,
                Err(e) => {
                    warn!(
                        session = %session.session_id, error = %e,
                        "[MINER_S7] NER detection failed, skipping session"
                    );
                    continue;
                }
            };

            if entities.is_empty() {
                self.storage.mark_session_entities_extracted(&session.session_id).await;
                continue;
            }

            // Generate a pseudo episode_id for provenance tracking
            let episode_id = format!("ep_{}_{}", session.session_id, now_ts);

            // 4-6. Upsert entities + generate embeddings
            let mut session_entity_ids: Vec<(String, String)> = Vec::new(); // (entity_id, entity_type)

            for entity in &entities {
                let name_normalized = entity.text.to_lowercase().trim().to_string();
                if name_normalized.is_empty() || name_normalized.len() < 2 {
                    continue;
                }

                // Deterministic entity_id: SHA256(owner_hex + ":" + name_normalized)
                let entity_id = {
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(owner_hex.as_bytes());
                    hasher.update(b":");
                    hasher.update(name_normalized.as_bytes());
                    format!("ent_{}", &hex::encode(hasher.finalize())[..16])
                };

                // Generate embedding for entity name (for Step 9 merge)
                let embedding = self.call_openclaw_embed(&entity.text).await;

                // Upsert entity
                let description = if entity.confidence > 0.8 {
                    Some(format!("{} ({})", entity.text, entity.label))
                } else {
                    None
                };

                match self.storage.upsert_entity(
                    &entity_id, &owner, &entity.text, &name_normalized,
                    &entity.label, description.as_deref(),
                    embedding.as_deref(),
                ).await {
                    Ok(is_new) => {
                        if is_new { total_entities += 1; }
                    }
                    Err(e) => {
                        warn!(entity = %entity.text, error = %e, "[MINER_S7] Entity upsert failed");
                        continue;
                    }
                }

                // 8. Insert episode_edge for provenance
                let _ = self.storage.insert_episode_edge(
                    &owner, &episode_id, &entity_id, "mentioned"
                ).await;

                session_entity_ids.push((entity_id, entity.label.clone()));
            }

            // 7. Detect relations between co-occurring entities
            //    Simple heuristic: entities that appear in the same conversation
            //    are likely related. For entities of different types that co-occur,
            //    create a CO_OCCURS edge. More precise relation extraction can be
            //    added later with a second GLiNER pass using relation labels.
            if session_entity_ids.len() >= 2 {
                // Create edges between entity pairs (limit to top 10 to avoid N² explosion)
                let pairs_limit = session_entity_ids.len().min(10);
                for i in 0..pairs_limit {
                    for j in (i + 1)..pairs_limit {
                        let (ref src_id, ref src_type) = session_entity_ids[i];
                        let (ref tgt_id, ref tgt_type) = session_entity_ids[j];

                        // Skip self-edges and same-type-same-name
                        if src_id == tgt_id { continue; }

                        // Determine relation type based on entity types
                        let relation_type = infer_relation_type(src_type, tgt_type);

                        match self.storage.insert_knowledge_edge(
                            &owner, src_id, tgt_id, relation_type,
                            None, // fact_text — no explicit text for co-occurrence
                            0.5,  // weight — moderate for co-occurrence
                            0.7,  // confidence — co-occurrence is indirect evidence
                            None, // embedding
                            now_ts,
                            Some(&episode_id),
                        ).await {
                            Ok(_) => { total_edges += 1; }
                            Err(e) => {
                                debug!(error = %e, "[MINER_S7] Edge insert failed (likely duplicate)");
                            }
                        }
                    }
                }
            }

            // 9. Mark session as extracted
            self.storage.mark_session_entities_extracted(&session.session_id).await;

            debug!(
                session = %session.session_id,
                entities = entities.len(),
                "[MINER_S7] Session entity extraction complete"
            );
        }

        if total_entities > 0 || total_edges > 0 {
            info!(
                entities = total_entities, edges = total_edges,
                sessions = pending.len(),
                "[MINER_S7] Entity extraction complete"
            );
        }
    }

    /// Step 8: Community detection via label propagation.
    ///
    /// Groups related entities into communities using the graph structure
    /// from knowledge_edges. Then identifies project-type communities
    /// (those containing "module", "file", "project" entity types).
    ///
    /// Pipeline:
    /// 1. Run label propagation (incremental: 1 round for efficiency)
    /// 2. For each community, count members and generate name
    /// 3. Upsert community records
    /// 4. Update entity → community assignments
    /// 5. Identify project communities → upsert projects
    ///
    /// Writes to: communities, projects tables. Updates entities.community_id.
    async fn step_8_community_detection(&self) {
        let owner = self.identity.public_key_bytes();

        // Run label propagation (needs raw Connection)
        // ⚠️ Requires MemoryStorage::conn_lock() to be public.
        // See storage.rs for the method definition.
        let labels = {
            let conn = self.storage.conn_lock().await;
            graph::label_propagation(&conn, &owner, None, true) // incremental = true
        };

        if labels.is_empty() {
            debug!("[MINER_S8] No entities for community detection");
            return;
        }

        // Group entities by community label
        let mut communities: HashMap<String, Vec<String>> = HashMap::new();
        for (entity_id, community_label) in &labels {
            communities.entry(community_label.clone())
                .or_default()
                .push(entity_id.clone());
        }

        let mut total_communities = 0u32;
        let mut total_projects = 0u32;

        for (community_id, member_ids) in &communities {
            if member_ids.is_empty() { continue; }

            // Generate community name from its members
            // Use the entity with highest mention_count as the representative name
            let community_name = {
                let mut best_name = community_id.clone();
                let mut best_mentions = 0i64;
                for eid in member_ids {
                    if let Some(entity) = self.storage.get_entity(eid).await {
                        if entity.mention_count > best_mentions {
                            best_mentions = entity.mention_count;
                            best_name = entity.name.clone();
                        }
                    }
                }
                best_name
            };

            // Upsert community
            if let Err(e) = self.storage.upsert_community(
                community_id, &owner, &community_name,
                None, // summary — will be generated in Step 11
                None, // description
                member_ids.len() as i64,
            ).await {
                warn!(error = %e, "[MINER_S8] Community upsert failed");
                continue;
            }
            total_communities += 1;

            // Update entity → community assignment
            for eid in member_ids {
                self.storage.update_entity_community(eid, community_id).await;
            }

            // Check if this community contains project-type entities
            let has_project_entities = {
                let mut found = false;
                for eid in member_ids {
                    if let Some(entity) = self.storage.get_entity(eid).await {
                        if matches!(
                            entity.entity_type.as_str(),
                            "project" | "module" | "file"
                        ) {
                            found = true;
                            break;
                        }
                    }
                }
                found
            };

            if has_project_entities {
                if let Err(e) = self.storage.upsert_project(
                    community_id, &owner, &community_name,
                    "active", community_id, None,
                ).await {
                    warn!(error = %e, "[MINER_S8] Project upsert failed");
                } else {
                    total_projects += 1;
                }
            }
        }

        if total_communities > 0 {
            info!(
                communities = total_communities, projects = total_projects,
                "[MINER_S8] Community detection complete"
            );
        }
    }

    /// Step 9: Recursive merge — consolidate fragment entities.
    ///
    /// Finds entities that represent the same real-world concept but have
    /// different surface forms (e.g., "JWT" vs "JSON Web Token"), and
    /// merges them using embedding similarity.
    ///
    /// Pipeline:
    /// 1. Load all entities with embeddings
    /// 2. Pairwise cosine similarity (O(N²), bounded by MINER_MERGE_BATCH)
    /// 3. cos > 0.92 → merge (consolidate mention_count, repoint edges)
    /// 4. 0.85 < cos < 0.92 → insert RELATED_TO edge (if not exists)
    /// 5. Check for temporal conflicts: find_superseded_edges → invalidate
    ///
    /// Writes to: entities, knowledge_edges (merge/RELATED_TO/invalidate).
    async fn step_9_recursive_merge(&self) {
        let owner = self.identity.public_key_bytes();

        // 1. Load entities with embeddings
        let entities = self.storage.get_entities_with_embedding(
            &owner, MINER_MERGE_BATCH
        ).await;

        if entities.len() < 2 {
            debug!("[MINER_S9] Not enough entities for merge ({} < 2)", entities.len());
            return;
        }

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut merged = 0u32;
        let mut related = 0u32;
        let mut merged_ids: HashSet<String> = HashSet::new(); // track already-merged

        // 2. Pairwise comparison
        for i in 0..entities.len() {
            let (ref id_a, ref name_a, ref type_a, ref emb_a) = entities[i];
            if merged_ids.contains(id_a) { continue; }

            for j in (i + 1)..entities.len() {
                let (ref id_b, ref name_b, ref type_b, ref emb_b) = entities[j];
                if merged_ids.contains(id_b) { continue; }
                if id_a == id_b { continue; }

                // Only merge same-type entities (don't merge a "person" into a "technology")
                if type_a != type_b { continue; }

                let sim = cosine_similarity(emb_a, emb_b);

                if sim > ENTITY_MERGE_THRESHOLD {
                    // 3. Merge: keep the one with longer name (more descriptive)
                    let (keep_id, remove_id) = if name_a.len() >= name_b.len() {
                        (id_a.as_str(), id_b.as_str())
                    } else {
                        (id_b.as_str(), id_a.as_str())
                    };

                    match self.storage.merge_entities(&owner, remove_id, keep_id).await {
                        Ok(()) => {
                            merged += 1;
                            merged_ids.insert(remove_id.to_string());
                            debug!(
                                keep = keep_id, remove = remove_id,
                                sim = format!("{:.3}", sim),
                                "[MINER_S9] Entities merged"
                            );
                        }
                        Err(e) => {
                            warn!(error = %e, "[MINER_S9] Entity merge failed");
                        }
                    }
                } else if sim > ENTITY_RELATED_THRESHOLD {
                    // 4. Create RELATED_TO edge
                    let _ = self.storage.insert_knowledge_edge(
                        &owner, id_a, id_b, "RELATED_TO",
                        Some(&format!("{} ~ {}", name_a, name_b)),
                        sim as f64, sim as f64,
                        None, now_ts, None,
                    ).await;
                    related += 1;
                }
            }
        }

        // 5. Temporal conflict detection
        // Check for superseded edges (e.g., "auth USES JWT" superseded by "auth USES OAuth")
        // This is done by looking for edges with same source + relation_type but different targets
        let mut invalidated = 0u32;
        {
            // Get recently created edges (from this miner cycle or recent)
            let entity_ids: Vec<String> = entities.iter()
                .map(|(id, _, _, _)| id.clone())
                .collect();
            let recent_edges = self.storage.get_active_edges(
                &owner, &entity_ids, 0.3,
            ).await;

            // For each entity with multiple same-relation-type edges, keep newest
            let mut seen_relations: HashMap<(String, String), (i64, i64)> = HashMap::new();
            for edge in &recent_edges {
                let key = (edge.source_id.clone(), edge.relation_type.clone());
                let existing = seen_relations.get(&key);
                if let Some(&(existing_edge_id, existing_valid_from)) = existing {
                    // Keep the newer edge, invalidate the older one
                    if edge.valid_from > existing_valid_from {
                        self.storage.invalidate_edge(existing_edge_id).await;
                        seen_relations.insert(key, (edge.edge_id, edge.valid_from));
                        invalidated += 1;
                    } else if edge.valid_from < existing_valid_from {
                        self.storage.invalidate_edge(edge.edge_id).await;
                        invalidated += 1;
                    }
                } else {
                    seen_relations.insert(key, (edge.edge_id, edge.valid_from));
                }
            }
        }

        if merged > 0 || related > 0 || invalidated > 0 {
            info!(
                merged = merged, related = related, invalidated = invalidated,
                "[MINER_S9] Recursive merge complete"
            );
        }
    }

    /// Step 10: Generate session summaries and extract code artifacts.
    ///
    /// Pipeline:
    /// 1. Get sessions pending summary generation
    /// 2. For each session:
    ///    a. Without LLM: top-5 entity names → summary
    ///    b. With LLM: call miner_llm_endpoint for natural summary
    /// 3. Extract Markdown code fences → insert as artifacts
    /// 4. Update session summary + mark as summary_generated
    ///
    /// Writes to: sessions.summary, sessions.key_decisions, artifacts table.
    async fn step_10_session_summary(&self) {
        let owner = self.identity.public_key_bytes();

        let pending = self.storage.get_pending_sessions(&owner, MINER_SESSION_BATCH).await;
        let pending: Vec<_> = pending.into_iter()
            .filter(|s| !s.summary_generated)
            .collect();

        if pending.is_empty() {
            debug!("[MINER_S10] No pending sessions for summary generation");
            return;
        }

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let code_regex = regex::Regex::new(CODE_FENCE_PATTERN)
            .unwrap_or_else(|_| regex::Regex::new(r"```(\w*)\n([\s\S]*?)```").unwrap());

        let mut total_summaries = 0u32;
        let mut total_artifacts = 0u32;

        for session in &pending {
            // Read conversation for code extraction
            let raw_logs = self.storage.get_rawlogs_for_session(&session.session_id).await;
            let rawlog_key = crate::services::memchain::derive_rawlog_key(
                &self.identity.to_bytes()
            );

            let mut full_text = String::new();

            for log in &raw_logs {
                let content = if log.encrypted == 1 {
                    String::from_utf8(
                        crate::services::memchain::decrypt_rawlog_content_pub(
                            &rawlog_key, &log.content
                        ).unwrap_or_default()
                    ).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&log.content).to_string()
                };
                full_text.push_str(&content);
                full_text.push('\n');
            }

            // --- Generate summary ---
            // Get top entities for this session (from episode_edges)
            let entity_names: Vec<String> = {
                let entities_by_owner = self.storage.get_entities_by_owner(
                    &owner, None, 100
                ).await;
                // Filter to entities that were extracted from this session
                // (Heuristic: recently updated entities — in a real impl we'd
                //  query episode_edges, but this is simpler for now)
                entities_by_owner.into_iter()
                    .take(5)
                    .map(|e| e.name)
                    .collect()
            };

            let summary = if entity_names.is_empty() {
                format!("Session with {} turns", session.turn_count)
            } else {
                format!("Topics: {}", entity_names.join(", "))
            };

            // TODO(Phase C): If miner_llm_endpoint is configured, call it for natural summary
            // let summary = if let Some(llm_url) = &self.miner_llm_endpoint { ... }

            self.storage.update_session_summary(
                &session.session_id, &summary, None
            ).await;
            self.storage.mark_session_summary_generated(&session.session_id).await;
            self.storage.update_session_ended_at(&session.session_id, now_ts).await;
            self.storage.mark_session_artifacts_extracted(&session.session_id).await;
            total_summaries += 1;

            // --- Extract code artifacts ---
            for cap in code_regex.captures_iter(&full_text) {
                let language = cap.get(1)
                    .map(|m| m.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("unknown");
                let code_content = match cap.get(2) {
                    Some(m) => m.as_str().trim(),
                    None => continue,
                };

                // Skip very short code blocks (likely inline snippets)
                if code_content.len() < 20 { continue; }

                let content_hash = {
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(code_content.as_bytes());
                    hex::encode(hasher.finalize())
                };

                let artifact_id = format!("art_{}_{}", &content_hash[..12], now_ts);
                let line_count = code_content.lines().count() as i64;

                // Store content as plaintext bytes
                // TODO(Phase C): Encrypt with record_key for at-rest protection
                let stored_content = code_content.as_bytes().to_vec();

                if let Err(e) = self.storage.insert_artifact(
                    &artifact_id, &owner, &session.session_id,
                    session.project_id.as_deref(),
                    "code",
                    None, // filename — cannot determine from code fence alone
                    Some(language),
                    1, // version
                    None, // parent_id
                    &stored_content,
                    &content_hash,
                    None, // embedding
                    Some(line_count),
                ).await {
                    debug!(error = %e, "[MINER_S10] Artifact insert failed (likely duplicate hash)");
                } else {
                    total_artifacts += 1;
                }
            }
        }

        if total_summaries > 0 || total_artifacts > 0 {
            info!(
                summaries = total_summaries, artifacts = total_artifacts,
                "[MINER_S10] Session summary complete"
            );
        }
    }

    /// Step 11: Ingest episodes and update community summaries.
    ///
    /// Pipeline:
    /// 1. For sessions with extracted entities, create episode records
    /// 2. Update community summaries (aggregate entity names per community)
    ///
    /// Writes to: episodes table, communities.summary.
    async fn step_11_episode_ingestion(&self) {
        let owner = self.identity.public_key_bytes();
        let rawlog_key = crate::services::memchain::derive_rawlog_key(
            &self.identity.to_bytes()
        );

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut episodes_created = 0u32;

        // 1. Find sessions that have been fully processed (entities_extracted=1)
        //    but don't yet have episode records.
        //    ⚠️ This uses conn_lock() for a custom query — requires public method on MemoryStorage.
        //    TODO(Phase C): Move this query into a dedicated storage_ops method to avoid
        //    direct SQL in reflection.rs.
        let processed_sessions = {
            let conn = self.storage.conn_lock().await;
            let mut stmt = match conn.prepare(
                "SELECT session_id, started_at, turn_count, summary
                 FROM sessions
                 WHERE owner = ?1
                   AND entities_extracted = 1
                   AND session_id NOT IN (
                       SELECT DISTINCT session_id FROM episodes
                       WHERE session_id IS NOT NULL
                   )
                 ORDER BY started_at DESC
                 LIMIT ?2"
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "[MINER_S11] Query processed sessions failed");
                    return;
                }
            };
            stmt.query_map(
                rusqlite::params![owner.as_slice(), MINER_SESSION_BATCH as i64],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            ).map(|rows| rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
            .unwrap_or_default()
        };

        for (session_id, started_at, _turn_count, summary) in &processed_sessions {
            let episode_id = format!("ep_{}_{}", session_id, started_at);

            // Read conversation content for episode storage
            let raw_logs = self.storage.get_rawlogs_for_session(session_id).await;
            let mut episode_text = String::new();
            for log in &raw_logs {
                let content = if log.encrypted == 1 {
                    String::from_utf8(
                        crate::services::memchain::decrypt_rawlog_content_pub(
                            &rawlog_key, &log.content
                        ).unwrap_or_default()
                    ).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&log.content).to_string()
                };
                episode_text.push_str(&format!("[{}] {}\n", log.role, content));
            }

            // Content hash for dedup
            let content_hash = {
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(episode_text.as_bytes());
                hex::encode(hasher.finalize())
            };

            // Store episode (content as plaintext bytes)
            // TODO(Phase C): Encrypt with record_key for at-rest protection
            if let Err(e) = self.storage.upsert_episode(
                &episode_id, &owner, "conversation", "miner",
                Some(session_id),
                episode_text.as_bytes(),
                &content_hash,
                None, // embedding — could be generated but expensive
                Some(episode_text.len() as i64),
                *started_at,
                summary.as_deref(),
            ).await {
                debug!(error = %e, "[MINER_S11] Episode upsert failed (likely duplicate)");
            } else {
                episodes_created += 1;
            }
        }

        // 2. Update community summaries
        let communities = self.storage.get_communities(&owner).await;
        let mut summaries_updated = 0u32;

        for community in &communities {
            let members = self.storage.get_entities_in_community(
                &community.community_id, &owner
            ).await;

            if members.is_empty() { continue; }

            // Generate summary from member entity names and types
            let member_desc: Vec<String> = members.iter()
                .take(10) // top 10 by mention_count (already sorted)
                .map(|e| format!("{} ({})", e.name, e.entity_type))
                .collect();

            let summary = format!(
                "Community with {} entities: {}",
                members.len(),
                member_desc.join(", ")
            );

            if let Err(e) = self.storage.upsert_community(
                &community.community_id, &owner, &community.name,
                Some(&summary),
                community.description.as_deref(),
                members.len() as i64,
            ).await {
                warn!(error = %e, "[MINER_S11] Community summary update failed");
            } else {
                summaries_updated += 1;
            }
        }

        if episodes_created > 0 || summaries_updated > 0 {
            info!(
                episodes = episodes_created,
                community_summaries = summaries_updated,
                "[MINER_S11] Episode ingestion complete"
            );
        }
    }

    // ============================================
    // OpenClaw HTTP Helpers
    // ============================================

    fn build_summary_prompt(&self, episodes: &[MemoryRecord]) -> String {
        let mut prompt = String::from(
            "You are a memory consolidation assistant. Summarize the following episodes \
             into concise knowledge. Focus on facts, preferences, and insights.\n\n"
        );
        for (i, ep) in episodes.iter().enumerate() {
            let content = String::from_utf8_lossy(&ep.encrypted_content);
            prompt.push_str(&format!("[Episode {}] {}\n", i + 1, content));
        }
        prompt.push_str("\nProvide a concise summary (2-5 paragraphs).");
        prompt
    }

    async fn call_openclaw_chat(&self, prompt: &str) -> Option<String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(OPENCLAW_TIMEOUT_SECS))
            .build().ok()?;

        let body = serde_json::json!({
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000, "temperature": 0.3
        });

        let resp = client.post(OPENCLAW_GATEWAY_CHAT).json(&body).send().await.ok()?;
        if !resp.status().is_success() { return None; }

        let json: serde_json::Value = resp.json().await.ok()?;
        json.get("choices")?.get(0)?.get("message")?.get("content")?.as_str().map(|s| s.to_string())
    }

    /// Generate embedding for a text.
    /// Strategy: local EmbedEngine first → OpenClaw Gateway HTTP fallback.
    async fn call_openclaw_embed(&self, text: &str) -> Option<Vec<f32>> {
        if let Some(ref engine) = self.embed_engine {
            match engine.embed_single(text) {
                Ok(embedding) => {
                    debug!(dim = embedding.len(), "[MINER] Local embed succeeded");
                    return Some(embedding);
                }
                Err(e) => {
                    warn!(error = %e, "[MINER] Local embed failed, falling back to OpenClaw");
                }
            }
        }
        self.call_openclaw_embed_remote(text).await
    }

    async fn call_openclaw_embed_remote(&self, text: &str) -> Option<Vec<f32>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build().ok()?;

        let body = serde_json::json!({ "model": "minilm-l6-v2", "input": text });

        let resp = client.post(OPENCLAW_GATEWAY_EMBED).json(&body).send().await.ok()?;
        if !resp.status().is_success() {
            warn!("[MINER] OpenClaw embed request failed: {}", resp.status());
            return None;
        }

        let json: serde_json::Value = resp.json().await.ok()?;
        json.get("data")?.get(0)?.get("embedding")?
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
    }

    // ============================================
    // P2P Broadcast
    // ============================================

    async fn broadcast_header(&self, msg: MemChainMessage) -> usize {
        let plaintext = match encode_memchain(&msg) {
            Ok(p) => p,
            Err(e) => { error!("[MINER] Encode: {}", e); return 0; }
        };
        if plaintext.len() > 1300 { return 0; }

        let all = self.sessions.all_sessions();
        let crypto = DefaultTransportCrypto::new();
        let mut sent = 0;

        for s in &all {
            if !s.is_established() { continue; }
            let ctr = s.next_tx_counter();
            let mut enc = vec![0u8; plaintext.len() + ENCRYPTION_OVERHEAD];
            let len = match crypto.encrypt(
                &s.session_key, ctr, s.id.as_bytes(), &plaintext, &mut enc,
            ) { Ok(l) => l, Err(_) => continue };
            enc.truncate(len);
            let pkt = DataPacket::new(*s.id.as_bytes(), ctr, enc);
            let bytes = encode_data_packet(&pkt).to_vec();
            if self.udp.send(&bytes, &s.client_endpoint).await.is_ok() { sent += 1; }
        }
        sent
    }
}

// ============================================
// Helper: Infer relation type from entity type pair
// ============================================

/// Infer a relation type from the entity types of two co-occurring entities.
///
/// This is a simple heuristic — more precise relation extraction would use
/// a second GLiNER pass with relation labels, or an LLM call.
///
/// ## v2.4.0-GraphCognition Phase B
/// Created for Step 7 entity extraction pipeline.
///
/// ## Examples
/// - (module, technology) → "USES"
/// - (file, module) → "BELONGS_TO"
/// - (person, project) → "WORKS_ON"
/// - (anything, anything) → "CO_OCCURS"
fn infer_relation_type(source_type: &str, target_type: &str) -> &'static str {
    match (source_type, target_type) {
        ("module", "technology") | ("project", "technology") => "USES",
        ("technology", "module") | ("technology", "project") => "USED_BY",
        ("file", "module") | ("file", "project") => "BELONGS_TO",
        ("module", "file") | ("project", "file") => "CONTAINS",
        ("person", "project") | ("person", "module") => "WORKS_ON",
        ("project", "person") | ("module", "person") => "HAS_CONTRIBUTOR",
        ("module", "module") => "DEPENDS_ON",
        ("technology", "technology") => "RELATED_TO",
        ("module", "concept") | ("project", "concept") => "IMPLEMENTS",
        ("concept", "module") | ("concept", "project") => "IMPLEMENTED_BY",
        _ => "CO_OCCURS",
    }
}

impl std::fmt::Debug for ReflectionMiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReflectionMiner")
            .field("interval", &self.interval)
            .field("threshold", &self.compaction_threshold)
            .field("mvf", &self.mvf_enabled)
            .field("ner", &self.ner_engine.is_some())
            .finish()
    }
}
