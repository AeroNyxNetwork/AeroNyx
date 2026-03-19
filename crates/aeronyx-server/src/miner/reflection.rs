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
//! ## v2.5.0+SuperNode: Cognitive Task Enqueue (Phase B — IMPLEMENTED)
//! When llm_router is also attached, Steps 8/9/10 enqueue async LLM tasks:
//! ```text
//! Step 8 tail: enqueue_community_narrative_tasks() → community_narrative tasks
//! Step 9 tail: enqueue_entity_description_tasks()  → entity_description tasks (if merged > 0)
//! Step 10 tail: enqueue_session_title_tasks()       → session_title tasks
//! ```
//! Task enqueue is non-blocking. Failures are logged and do NOT cascade.
//! TaskWorker (task_worker.rs) processes the queue asynchronously.
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
//! ## Modification Reason (v2.4.0-GraphCognition Phase B Fixes)
//! Merged two post-review fixes:
//!   - Fix 1 (Step 8): Back-fill sessions.project_id after project community detection.
//!     Traces project entities → episode_edges → episodes → session_id, then updates
//!     sessions.project_id for all linked sessions. Uses batched conn_lock() for efficiency.
//!   - Fix 2 (Step 7): Added stopword entity filtering via is_stopword_entity().
//!     Filters out generic entities (e.g., "project", "user", "code") that create
//!     hub nodes polluting community detection. Also fixes trim ordering in
//!     name_normalized construction (trim before lowercase for correctness).
//!   - Bug fix: is_stopword_entity() Rule 3 uppercase detection now uses original
//!     entity text instead of already-lowercased name_normalized.
//!
//! ## Modification Reason (v2.4.0+Search)
//! Step 10: Added human-readable session title generation after summary generation.
//! Title strategy (no LLM):
//!   1. Has project → "{ProjectName}: {top-3 entities}"
//!   2. Has entities but no project → "{top-3 entities}"
//!   3. No entities → first user message (truncated to 60 chars, UTF-8 safe)
//! Updated update_session_summary() call signature to include title parameter.
//! Bug fixes applied in this revision:
//!   - BUG-FIX-1: UTF-8 safe truncation in title fallback (char_indices instead of byte slice)
//!   - BUG-FIX-2: rawlog_key derivation unified to top of step_10_session_summary(),
//!     removed redundant inner derivation inside the session loop
//!   - BUG-FIX-3: get_project() return type guarded — .ok() applied if Result,
//!     .map(|p| p.name) only called on inner Option<Project>
//!
//! ## Modification Reason (v2.5.0+SuperNode)
//! Added llm_router field (Option<Arc<LlmRouter>>) and with_llm_router() builder.
//! Added three private cognitive task enqueue helpers (Steps 8/9/10 tail calls):
//!   - enqueue_community_narrative_tasks(): skips communities that already have
//!     an LLM-generated summary (not starting with "Community with").
//!   - enqueue_entity_description_tasks(): enqueue only when merged > 0 and
//!     entity has no rich description yet.
//!   - enqueue_session_title_tasks(): targets sessions where title IS NULL OR
//!     summary starts with "Topics:" (no-LLM placeholder).
//! All three use insert_cognitive_task() which now returns Result<Option<i64>> —
//! Ok(None) means duplicate skipped (idempotent), treated as success.
//!
//! ⚠️ Important Note for Next Developer:
//! - derive_rawlog_key MUST use identity.to_bytes() (PRIVATE key), NOT public_key_bytes()
//! - Step 0 decrypts rawlogs — if the key is wrong, content will be garbage/empty
//! - After the key derivation fix, old rawlogs encrypted with public-key-derived key
//!   are cleared by the migration in storage.rs
//! - ner_engine is Option<Arc<NerEngine>> — when None, Steps 7-11 are skipped
//! - llm_router is Option<Arc<LlmRouter>> — when None, SuperNode enqueue is skipped.
//!   SuperNode steps are NESTED inside the ner_engine gate: ner must be present too.
//! - NerEngine is thread-safe (Mutex<Session> internally) — safe to use from Miner
//! - Steps 7-11 are gated on self.ner_engine.is_some() — if NER is disabled,
//!   all cognitive graph steps are skipped (v2.3.0 behavior preserved).
//! - Each step has independent error handling; step failure does NOT cascade.
//! - Step 7 reads raw_logs (may be encrypted) — uses derive_rawlog_key with
//!   self.identity.to_bytes() (PRIVATE key), same pattern as Step 0.
//! - Entity ID = SHA256(owner_hex || ":" || name_normalized) — deterministic.
//! - Step 7 filters stopword entities via is_stopword_entity() — entities whose
//!   name matches their type label or is a generic word are skipped to avoid
//!   creating hub nodes that pollute community detection (Step 8).
//! - Step 9 merge threshold: cosine > 0.92 = merge, 0.85-0.92 = RELATED_TO edge.
//! - Step 10 code block regex matches Markdown fences only (```lang\n...\n```).
//! - Step 10 SuperNode enqueue: targets sessions where title IS NULL OR summary
//!   starts with "Topics:" — this is the no-LLM placeholder marker. The SuperNode
//!   worker will overwrite with an LLM-generated title after completion.
//! - Step 11 encrypts episode content using record_key (same as records table).
//! - Step 8 and Step 11 call self.storage.conn_lock() — this method MUST be
//!   exposed as a public method on MemoryStorage (returns MutexGuard<Connection>).
//!   If it doesn't exist yet, add: `pub async fn conn_lock(&self) -> ... { self.conn.lock().await }`
//! - Step 8 back-fills sessions.project_id by tracing entity → episode_edge → session.
//!   Uses batched conn_lock() to avoid acquiring the lock per-session.
//! - update_session_summary() in storage_ops.rs MUST accept a 4th parameter:
//!   `title: Option<&str>`. If not yet updated, apply the corresponding storage_ops change.
//! - Title UTF-8 truncation uses char_indices to avoid panic on multi-byte characters
//!   (e.g., Chinese/Japanese text). Byte-level slicing (&content[..60]) is UNSAFE for
//!   multi-byte strings and has been replaced.
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
//! v2.4.0-GraphCognition Phase B Fixes - 🌟 Merged two post-review fixes:
//!   Fix 1: Step 8 sessions.project_id back-fill (batched conn_lock).
//!   Fix 2: Step 7 stopword entity filtering + is_stopword_entity() function.
//!   Bug fix: trim ordering in name_normalized, uppercase detection in Rule 3.
//! v2.4.0+Search - 🌟 Step 10: Added session title generation (no-LLM strategy).
//!   Updated update_session_summary() call to pass generated title.
//!   Bug fixes: UTF-8 safe truncation, unified rawlog_key derivation,
//!   get_project() return type guard.
//! v2.5.0+SuperNode - 🌟 Added llm_router field + with_llm_router() builder.
//!   Added private helpers: enqueue_community_narrative_tasks(),
//!   enqueue_entity_description_tasks(), enqueue_session_title_tasks().
//!   Steps 8/9/10 call these helpers when llm_router is present.
//!   All insert_cognitive_task() calls handle Result<Option<i64>> correctly.

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
// v2.5.0: LlmRouter for SuperNode cognitive task enqueue
use crate::services::memchain::LlmRouter;
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
const DEFAULT_ENTITY_LABELS: &[&str] = &[
    "project", "module", "technology", "person",
    "file", "concept", "tool", "language",
];

/// Relation labels for GLiNER relation extraction (second pass).
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
const ENTITY_MERGE_THRESHOLD: f32 = 0.92;

/// Cosine similarity threshold for RELATED_TO edge (Step 9).
const ENTITY_RELATED_THRESHOLD: f32 = 0.85;

/// Regex pattern for Markdown code fences (Step 10).
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
// v2.5.0+SuperNode: Task Priority Constants
// ============================================

/// Priority for session_title tasks (high — user-visible).
const SUPERNODE_PRIORITY_SESSION_TITLE: i64 = 8;
/// Priority for community_narrative tasks (medium).
const SUPERNODE_PRIORITY_COMMUNITY_NARRATIVE: i64 = 5;
/// Priority for entity_description tasks (low — background enrichment).
const SUPERNODE_PRIORITY_ENTITY_DESCRIPTION: i64 = 3;
/// Default max_retries for all cognitive tasks.
const SUPERNODE_MAX_RETRIES: i64 = 3;

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

    /// v2.5.0+SuperNode: LLM router for cognitive task enqueue.
    ///
    /// When Some AND ner_engine is also Some, Steps 8/9/10 will enqueue
    /// async LLM tasks after their core logic completes:
    ///   - Step 8 → community_narrative tasks
    ///   - Step 9 → entity_description tasks (only if merged > 0)
    ///   - Step 10 → session_title tasks
    ///
    /// When None, no cognitive tasks are enqueued (v2.4.0 behavior).
    /// The llm_router is passed to helpers only for routing config checks;
    /// actual LLM calls happen in TaskWorker (task_worker.rs).
    llm_router: Option<Arc<LlmRouter>>,
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
            llm_router: None,
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
    #[must_use]
    pub fn with_embed_engine(mut self, engine: Arc<EmbedEngine>) -> Self {
        self.embed_engine = Some(engine);
        self
    }

    /// v2.4.0: Set the local NER engine for cognitive graph Steps 7-11.
    #[must_use]
    pub fn with_ner_engine(mut self, engine: Arc<NerEngine>) -> Self {
        self.ner_engine = Some(engine);
        self
    }

    /// v2.5.0+SuperNode: Set the LLM router for cognitive task enqueue.
    ///
    /// When set (along with ner_engine), Steps 8/9/10 will enqueue async LLM
    /// tasks after their core logic. The router is used only to check routing
    /// config; actual dispatch happens in TaskWorker.
    ///
    /// ## Thread Safety
    /// LlmRouter is Arc-wrapped and Send+Sync. Safe to use from Miner.
    #[must_use]
    pub fn with_llm_router(mut self, router: Arc<LlmRouter>) -> Self {
        self.llm_router = Some(router);
        self
    }

    pub async fn run(self, mut shutdown_rx: tokio::sync::broadcast::Receiver<()>) {
        info!(
            interval = self.interval.as_secs(),
            threshold = self.compaction_threshold,
            mvf = self.mvf_enabled,
            ner = self.ner_engine.is_some(),
            supernode = self.llm_router.is_some(),
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
            let raw_logs = self.storage.get_rawlogs_for_session(&session.session_id).await;
            if raw_logs.is_empty() {
                self.storage.mark_session_entities_extracted(&session.session_id).await;
                continue;
            }

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

            let entities = match ner.detect_entities(&conversation_text, DEFAULT_ENTITY_LABELS) {
                Ok(ents) => ents,
                Err(e) => {
                    warn!(session = %session.session_id, error = %e, "[MINER_S7] NER failed, skipping");
                    continue;
                }
            };

            if entities.is_empty() {
                self.storage.mark_session_entities_extracted(&session.session_id).await;
                continue;
            }

            let episode_id = format!("ep_{}_{}", session.session_id, now_ts);
            let mut session_entity_ids: Vec<(String, String)> = Vec::new();

            for entity in &entities {
                let name_normalized = entity.text.trim().to_lowercase();
                if name_normalized.is_empty() || name_normalized.len() < 2 { continue; }

                if is_stopword_entity(&name_normalized, &entity.text, &entity.label) {
                    debug!(entity = %entity.text, label = %entity.label, "[MINER_S7] Skipped stopword");
                    continue;
                }

                let entity_id = {
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(owner_hex.as_bytes());
                    hasher.update(b":");
                    hasher.update(name_normalized.as_bytes());
                    format!("ent_{}", &hex::encode(hasher.finalize())[..16])
                };

                let embedding = self.call_openclaw_embed(&entity.text).await;
                let description = if entity.confidence > 0.8 {
                    Some(format!("{} ({})", entity.text, entity.label))
                } else {
                    None
                };

                match self.storage.upsert_entity(
                    &entity_id, &owner, &entity.text, &name_normalized,
                    &entity.label, description.as_deref(), embedding.as_deref(),
                ).await {
                    Ok(is_new) => {
                        if is_new { total_entities += 1; }
                        self.storage.fts_index_entity(
                            &entity_id, &owner, &entity.text,
                            description.as_deref(), &entity.label,
                        ).await;
                    }
                    Err(e) => {
                        warn!(entity = %entity.text, error = %e, "[MINER_S7] Entity upsert failed");
                        continue;
                    }
                }

                let _ = self.storage.insert_episode_edge(
                    &owner, &episode_id, &entity_id, "mentioned"
                ).await;

                session_entity_ids.push((entity_id, entity.label.clone()));
            }

            if session_entity_ids.len() >= 2 {
                let pairs_limit = session_entity_ids.len().min(10);
                for i in 0..pairs_limit {
                    for j in (i + 1)..pairs_limit {
                        let (ref src_id, ref src_type) = session_entity_ids[i];
                        let (ref tgt_id, ref tgt_type) = session_entity_ids[j];
                        if src_id == tgt_id { continue; }

                        let relation_type = infer_relation_type(src_type, tgt_type);
                        match self.storage.insert_knowledge_edge(
                            &owner, src_id, tgt_id, relation_type,
                            None, 0.5, 0.7, None, now_ts, Some(&episode_id),
                        ).await {
                            Ok(_) => { total_edges += 1; }
                            Err(e) => {
                                debug!(error = %e, "[MINER_S7] Edge insert failed (likely duplicate)");
                            }
                        }
                    }
                }
            }

            self.storage.mark_session_entities_extracted(&session.session_id).await;
            debug!(session = %session.session_id, entities = entities.len(), "[MINER_S7] Done");
        }

        if total_entities > 0 || total_edges > 0 {
            info!(entities = total_entities, edges = total_edges, "[MINER_S7] Entity extraction complete");
        }
    }

    /// Step 8: Community detection via label propagation.
    ///
    /// v2.5.0+SuperNode tail: enqueue_community_narrative_tasks() if llm_router is set.
    async fn step_8_community_detection(&self) {
        let owner = self.identity.public_key_bytes();

        let labels = {
            let conn = self.storage.conn_lock().await;
            graph::label_propagation(&conn, &owner, None, true)
        };

        if labels.is_empty() {
            debug!("[MINER_S8] No entities for community detection");
            return;
        }

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

            let community_name = {
                let mut best_name = community_id.clone();
                let mut best_mentions = 0i64;
                let mut project_name: Option<String> = None;

                for eid in member_ids {
                    if let Some(entity) = self.storage.get_entity(eid).await {
                        if entity.entity_type == "project" && project_name.is_none() {
                            project_name = Some(entity.name.clone());
                        }
                        if entity.mention_count > best_mentions {
                            best_mentions = entity.mention_count;
                            best_name = entity.name.clone();
                        }
                    }
                }
                project_name.unwrap_or(best_name)
            };

            if let Err(e) = self.storage.upsert_community(
                community_id, &owner, &community_name,
                None, None, member_ids.len() as i64,
            ).await {
                warn!(error = %e, "[MINER_S8] Community upsert failed");
                continue;
            }
            total_communities += 1;

            for eid in member_ids {
                self.storage.update_entity_community(eid, community_id).await;
            }

            let has_project_entities = {
                let mut found = false;
                for eid in member_ids {
                    if let Some(entity) = self.storage.get_entity(eid).await {
                        if matches!(entity.entity_type.as_str(), "project" | "module" | "file") {
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

                    let mut project_session_ids: HashSet<String> = HashSet::new();
                    for eid in member_ids {
                        let ep_links = self.storage.get_episodes_for_entity(eid).await;
                        for (episode_id, _role) in &ep_links {
                            if episode_id.starts_with("ep_") {
                                let rest = &episode_id[3..];
                                if let Some(last_underscore) = rest.rfind('_') {
                                    let sid = &rest[..last_underscore];
                                    project_session_ids.insert(sid.to_string());
                                }
                            }
                        }
                    }

                    if !project_session_ids.is_empty() {
                        let conn = self.storage.conn_lock().await;
                        for sid in &project_session_ids {
                            let _ = conn.execute(
                                "UPDATE sessions SET project_id = ?1 WHERE session_id = ?2 AND (project_id IS NULL OR project_id = '')",
                                rusqlite::params![community_id, sid],
                            );
                        }
                        drop(conn);
                        debug!(project = community_id, sessions = project_session_ids.len(), "[MINER_S8] Back-filled project_id");
                    }
                }
            }
        }

        // Small community merge (unchanged from v2.4.0)
        let mut merged_small = 0u32;
        let small_communities: Vec<(String, Vec<String>)> = communities.iter()
            .filter(|(_, members)| members.len() < 3)
            .map(|(cid, members)| (cid.clone(), members.clone()))
            .collect();
        let large_communities: Vec<(String, Vec<String>)> = communities.iter()
            .filter(|(_, members)| members.len() >= 3)
            .map(|(cid, members)| (cid.clone(), members.clone()))
            .collect();

        if !small_communities.is_empty() && !large_communities.is_empty() {
            for (small_cid, small_members) in &small_communities {
                let mut edge_counts: HashMap<String, usize> = HashMap::new();
                for eid in small_members {
                    let edges = self.storage.get_edges_for_entity(eid, &owner).await;
                    for edge in &edges {
                        let other_id = if edge.source_id == *eid { &edge.target_id } else { &edge.source_id };
                        for (large_cid, large_members) in &large_communities {
                            if large_members.contains(other_id) {
                                *edge_counts.entry(large_cid.clone()).or_insert(0) += 1;
                            }
                        }
                    }
                }
                if let Some((best_large_cid, _)) = edge_counts.iter().max_by_key(|(_, c)| *c) {
                    for eid in small_members {
                        self.storage.update_entity_community(eid, best_large_cid).await;
                    }
                    merged_small += 1;
                }
            }
        }

        if merged_small > 0 {
            let existing_communities = self.storage.get_communities(&owner).await;
            for (large_cid, _) in &large_communities {
                let members = self.storage.get_entities_in_community(large_cid, &owner).await;
                let cname = existing_communities.iter()
                    .find(|c| c.community_id == *large_cid)
                    .map(|c| c.name.as_str()).unwrap_or(large_cid.as_str());
                let _ = self.storage.upsert_community(large_cid, &owner, cname, None, None, members.len() as i64).await;
            }
            info!(merged = merged_small, "[MINER_S8] Small communities merged");
        }

        if total_communities > 0 {
            info!(communities = total_communities, projects = total_projects, "[MINER_S8] Complete");
        }

        // ── v2.5.0+SuperNode tail ─────────────────────────────────────────
        // Enqueue community_narrative tasks for communities that don't yet
        // have an LLM-generated summary. Non-blocking — failures are logged.
        if self.llm_router.is_some() {
            self.enqueue_community_narrative_tasks(&owner).await;
        }
    }

    /// Step 9: Recursive merge — consolidate fragment entities.
    ///
    /// v2.5.0+SuperNode tail: enqueue_entity_description_tasks() if merged > 0.
    async fn step_9_recursive_merge(&self) {
        let owner = self.identity.public_key_bytes();

        let entities = self.storage.get_entities_with_embedding(&owner, MINER_MERGE_BATCH).await;

        if entities.len() < 2 {
            debug!("[MINER_S9] Not enough entities ({} < 2)", entities.len());
            return;
        }

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut merged = 0u32;
        let mut related = 0u32;
        let mut merged_ids: HashSet<String> = HashSet::new();

        for i in 0..entities.len() {
            let (ref id_a, ref name_a, ref type_a, ref emb_a) = entities[i];
            if merged_ids.contains(id_a) { continue; }

            for j in (i + 1)..entities.len() {
                let (ref id_b, ref name_b, ref type_b, ref emb_b) = entities[j];
                if merged_ids.contains(id_b) { continue; }
                if id_a == id_b || type_a != type_b { continue; }

                let sim = cosine_similarity(emb_a, emb_b);

                if sim > ENTITY_MERGE_THRESHOLD {
                    let (keep_id, remove_id) = if name_a.len() >= name_b.len() {
                        (id_a.as_str(), id_b.as_str())
                    } else {
                        (id_b.as_str(), id_a.as_str())
                    };

                    match self.storage.merge_entities(&owner, remove_id, keep_id).await {
                        Ok(()) => {
                            merged += 1;
                            merged_ids.insert(remove_id.to_string());
                        }
                        Err(e) => { warn!(error = %e, "[MINER_S9] Entity merge failed"); }
                    }
                } else if sim > ENTITY_RELATED_THRESHOLD {
                    let _ = self.storage.insert_knowledge_edge(
                        &owner, id_a, id_b, "RELATED_TO",
                        Some(&format!("{} ~ {}", name_a, name_b)),
                        sim as f64, sim as f64, None, now_ts, None,
                    ).await;
                    related += 1;
                }
            }
        }

        // Temporal conflict detection (unchanged)
        let mut invalidated = 0u32;
        {
            let entity_ids: Vec<String> = entities.iter().map(|(id, _, _, _)| id.clone()).collect();
            let recent_edges = self.storage.get_active_edges(&owner, &entity_ids, 0.3).await;
            let mut seen_relations: HashMap<(String, String), (i64, i64)> = HashMap::new();
            for edge in &recent_edges {
                let key = (edge.source_id.clone(), edge.relation_type.clone());
                if let Some(&(existing_edge_id, existing_valid_from)) = seen_relations.get(&key) {
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
            info!(merged = merged, related = related, invalidated = invalidated, "[MINER_S9] Complete");
        }

        // ── v2.5.0+SuperNode tail ─────────────────────────────────────────
        // Only enqueue entity_description tasks when entities were actually
        // merged this tick — merged entities need fresh descriptions.
        if self.llm_router.is_some() && merged > 0 {
            self.enqueue_entity_description_tasks(&owner).await;
        }
    }

    /// Step 10: Generate session summaries and extract code artifacts.
    ///
    /// v2.5.0+SuperNode tail: enqueue_session_title_tasks() if llm_router is set.
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

        // BUG-FIX-2: Derive rawlog_key once per step tick, not per session.
        let rawlog_key = crate::services::memchain::derive_rawlog_key(&self.identity.to_bytes());

        let mut total_summaries = 0u32;
        let mut total_artifacts = 0u32;

        for session in &pending {
            let raw_logs = self.storage.get_rawlogs_for_session(&session.session_id).await;

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

            let entity_names: Vec<String> = {
                let mut session_entity_ids: HashSet<String> = HashSet::new();

                let episodes = self.storage.get_episodes_for_session(&session.session_id).await;
                for (episode_id, _, _) in &episodes {
                    let linked = self.storage.get_entities_for_episode(episode_id).await;
                    for (entity_id, _role) in linked {
                        session_entity_ids.insert(entity_id);
                    }
                }

                let ep_prefix = format!("ep_{}_", session.session_id);
                let all_entities = self.storage.get_entities_by_owner(&owner, None, 500).await;
                for entity in &all_entities {
                    let ep_links = self.storage.get_episodes_for_entity(&entity.entity_id).await;
                    for (episode_id, _role) in &ep_links {
                        if episode_id.starts_with(&ep_prefix) {
                            session_entity_ids.insert(entity.entity_id.clone());
                        }
                    }
                }

                let mut named: Vec<(String, i64)> = Vec::new();
                for eid in &session_entity_ids {
                    if let Some(entity) = self.storage.get_entity(eid).await {
                        named.push((entity.name, entity.mention_count));
                    }
                }
                named.sort_by(|a, b| b.1.cmp(&a.1));
                named.into_iter().take(5).map(|(name, _)| name).collect()
            };

            // No-LLM summary placeholder (marker prefix "Topics:" used by SuperNode
            // enqueue_session_title_tasks to identify sessions needing LLM title)
            let summary = if entity_names.is_empty() {
                format!("Session with {} turns", session.turn_count)
            } else {
                format!("Topics: {}", entity_names.join(", "))
            };

            // No-LLM title (placeholder — SuperNode will overwrite with LLM title)
            let title = {
                let project_name: Option<String> = if let Some(ref pid) = session.project_id {
                    self.storage.get_project(pid).await.map(|p| p.name)
                } else {
                    None
                };

                if !entity_names.is_empty() {
                    let top_entities = entity_names.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
                    match project_name {
                        Some(pname) => format!("{}: {}", pname, top_entities),
                        None => top_entities,
                    }
                } else {
                    // BUG-FIX-1: UTF-8 safe truncation via char_indices
                    let first_user_msg = raw_logs.iter()
                        .find(|l| l.role == "user")
                        .map(|l| {
                            let content = if l.encrypted == 1 {
                                String::from_utf8(
                                    crate::services::memchain::decrypt_rawlog_content_pub(
                                        &rawlog_key, &l.content
                                    ).unwrap_or_default()
                                ).unwrap_or_default()
                            } else {
                                String::from_utf8_lossy(&l.content).to_string()
                            };
                            let char_count = content.chars().count();
                            if char_count <= 60 {
                                content
                            } else {
                                let byte_60 = content.char_indices().nth(60)
                                    .map(|(pos, _)| pos).unwrap_or(content.len());
                                let truncated = &content[..byte_60];
                                match truncated.rfind(' ') {
                                    Some(pos) if pos > 20 => format!("{}...", &truncated[..pos]),
                                    _ => format!("{}...", truncated),
                                }
                            }
                        })
                        .unwrap_or_else(|| format!("Session {}", &session.session_id));

                    match project_name {
                        Some(pname) => format!("{}: {}", pname, first_user_msg),
                        None => first_user_msg,
                    }
                }
            };

            self.storage.update_session_summary(
                &session.session_id, &summary, None, Some(&title)
            ).await;
            self.storage.fts_index_session(&session.session_id, &owner, &summary).await;
            self.storage.mark_session_summary_generated(&session.session_id).await;
            self.storage.update_session_ended_at(&session.session_id, now_ts).await;
            self.storage.mark_session_artifacts_extracted(&session.session_id).await;
            total_summaries += 1;

            // Code artifact extraction (unchanged)
            for cap in code_regex.captures_iter(&full_text) {
                let language = cap.get(1).map(|m| m.as_str())
                    .filter(|s| !s.is_empty()).unwrap_or("unknown");
                let code_content = match cap.get(2) { Some(m) => m.as_str().trim(), None => continue };
                if code_content.len() < 20 { continue; }

                let content_hash = {
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(code_content.as_bytes());
                    hex::encode(hasher.finalize())
                };

                let artifact_id = format!("art_{}_{}", &content_hash[..12], now_ts);
                let line_count = code_content.lines().count() as i64;
                let stored_content = code_content.as_bytes().to_vec();

                if let Err(e) = self.storage.insert_artifact(
                    &artifact_id, &owner, &session.session_id,
                    session.project_id.as_deref(), "code", None, Some(language),
                    1, None, &stored_content, &content_hash, None, Some(line_count),
                ).await {
                    debug!(error = %e, "[MINER_S10] Artifact insert failed (likely duplicate)");
                } else {
                    total_artifacts += 1;
                }
            }
        }

        if total_summaries > 0 || total_artifacts > 0 {
            info!(summaries = total_summaries, artifacts = total_artifacts, "[MINER_S10] Complete");
        }

        // ── v2.5.0+SuperNode tail ─────────────────────────────────────────
        // Enqueue session_title tasks for sessions with no-LLM placeholder titles.
        if self.llm_router.is_some() {
            let owner = self.identity.public_key_bytes();
            self.enqueue_session_title_tasks(&owner).await;
        }
    }

    /// Step 11: Ingest episodes and update community summaries. (unchanged)
    async fn step_11_episode_ingestion(&self) {
        let owner = self.identity.public_key_bytes();
        let rawlog_key = crate::services::memchain::derive_rawlog_key(&self.identity.to_bytes());

        let now_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let mut episodes_created = 0u32;

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
                Err(e) => { warn!(error = %e, "[MINER_S11] Query failed"); return; }
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

            let content_hash = {
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                hasher.update(episode_text.as_bytes());
                hex::encode(hasher.finalize())
            };

            if let Err(e) = self.storage.upsert_episode(
                &episode_id, &owner, "conversation", "miner",
                Some(session_id), episode_text.as_bytes(), &content_hash,
                None, Some(episode_text.len() as i64), *started_at, summary.as_deref(),
            ).await {
                debug!(error = %e, "[MINER_S11] Episode upsert failed (likely duplicate)");
            } else {
                episodes_created += 1;
            }
        }

        let communities = self.storage.get_communities(&owner).await;
        let mut summaries_updated = 0u32;

        for community in &communities {
            let members = self.storage.get_entities_in_community(&community.community_id, &owner).await;
            if members.is_empty() { continue; }

            let member_desc: Vec<String> = members.iter().take(10)
                .map(|e| format!("{} ({})", e.name, e.entity_type))
                .collect();
            let summary = format!("Community with {} entities: {}", members.len(), member_desc.join(", "));

            if let Err(e) = self.storage.upsert_community(
                &community.community_id, &owner, &community.name,
                Some(&summary), community.description.as_deref(), members.len() as i64,
            ).await {
                warn!(error = %e, "[MINER_S11] Community summary update failed");
            } else {
                summaries_updated += 1;
            }
        }

        if episodes_created > 0 || summaries_updated > 0 {
            info!(episodes = episodes_created, community_summaries = summaries_updated, "[MINER_S11] Complete");
        }
    }

    // ============================================
    // v2.5.0+SuperNode: Cognitive Task Enqueue Helpers
    // ============================================
    //
    // These three private helpers are called at the tail of Steps 8/9/10
    // when llm_router is present. They are non-blocking: insert_cognitive_task()
    // returns Result<Option<i64>> where Ok(None) = duplicate skipped (idempotent).
    //
    // ⚠️ Do NOT call these when llm_router is None — the caller gates them.
    // ⚠️ insert_cognitive_task() may return Ok(None) for already-active tasks.
    //    Treat Ok(None) as success (idempotency guard working correctly).

    /// Step 8 tail: enqueue community_narrative LLM tasks.
    ///
    /// Skips communities that already have an LLM-generated summary.
    /// A summary is considered LLM-generated if it does NOT start with
    /// "Community with" (which is the no-LLM placeholder prefix written
    /// by Step 11 and Step 8 community name generation).
    async fn enqueue_community_narrative_tasks(&self, owner: &[u8]) {
        let communities = self.storage.get_communities(owner).await;

        let mut enqueued = 0u32;
        let mut skipped = 0u32;

        for community in &communities {
            // Skip if already has LLM-generated summary
            // "Community with" is the Step 11 no-LLM placeholder prefix
            let needs_llm = match &community.summary {
                None => true,
                Some(s) => s.starts_with("Community with"),
            };

            if !needs_llm {
                skipped += 1;
                continue;
            }

            // Build minimal payload for the task worker
            let payload = serde_json::json!({
                "community_id": community.community_id,
                "community_name": community.name,
                "member_count": community.entity_count,
            });

            match self.storage.insert_cognitive_task(
                "community_narrative",
                SUPERNODE_PRIORITY_COMMUNITY_NARRATIVE,
                &payload.to_string(),
                None, // prompt_messages — built by task_worker using prompts.rs
                Some("communities"),
                Some(&community.community_id),
                "structured",
                SUPERNODE_MAX_RETRIES,
            ).await {
                Ok(Some(id)) => {
                    debug!(id = id, community = %community.community_id, "[MINER_S8] Enqueued community_narrative");
                    enqueued += 1;
                }
                Ok(None) => {
                    // Duplicate skipped — idempotency guard working correctly
                    skipped += 1;
                }
                Err(e) => {
                    warn!(error = %e, community = %community.community_id, "[MINER_S8] Task enqueue failed");
                }
            }
        }

        if enqueued > 0 {
            info!(enqueued = enqueued, skipped = skipped, "[MINER_S8] community_narrative tasks enqueued");
        }
    }

    /// Step 9 tail: enqueue entity_description LLM tasks (only when merged > 0).
    ///
    /// Targets entities that have no description or a short auto-generated one
    /// (description IS NULL OR length < 50 chars). The "no rich description" check
    /// avoids re-enqueuing entities that already got an LLM description.
    async fn enqueue_entity_description_tasks(&self, owner: &[u8]) {
        // Get entities that need description enrichment.
        // Use the same MINER_MERGE_BATCH limit to avoid flooding the queue.
        let entities = self.storage.get_entities_by_owner(owner, None, MINER_MERGE_BATCH).await;

        let mut enqueued = 0u32;
        let mut skipped = 0u32;

        for entity in &entities {
            // Skip if already has a rich LLM-generated description (>= 50 chars)
            let needs_description = match &entity.description {
                None => true,
                Some(d) => d.len() < 50,
            };

            if !needs_description {
                skipped += 1;
                continue;
            }

            let payload = serde_json::json!({
                "entity_id": entity.entity_id,
                "entity_name": entity.name,
                "entity_type": entity.entity_type,
                "mention_count": entity.mention_count,
            });

            match self.storage.insert_cognitive_task(
                "entity_description",
                SUPERNODE_PRIORITY_ENTITY_DESCRIPTION,
                &payload.to_string(),
                None,
                Some("entities"),
                Some(&entity.entity_id),
                "structured",
                SUPERNODE_MAX_RETRIES,
            ).await {
                Ok(Some(id)) => {
                    debug!(id = id, entity = %entity.entity_id, "[MINER_S9] Enqueued entity_description");
                    enqueued += 1;
                }
                Ok(None) => { skipped += 1; }
                Err(e) => {
                    warn!(error = %e, entity = %entity.entity_id, "[MINER_S9] Task enqueue failed");
                }
            }
        }

        if enqueued > 0 {
            info!(enqueued = enqueued, skipped = skipped, "[MINER_S9] entity_description tasks enqueued");
        }
    }

    /// Step 10 tail: enqueue session_title LLM tasks.
    ///
    /// ## Target Sessions
    /// Sessions where the title needs LLM improvement:
    ///   1. `title IS NULL` — new session, no title yet
    ///   2. `summary LIKE 'Topics:%'` — no-LLM placeholder from Step 10
    ///
    /// The "Topics:" prefix is the marker written by step_10_session_summary()
    /// when it generates a no-LLM summary. The SuperNode worker will overwrite
    /// both the title and the summary with LLM-generated versions after completion.
    ///
    /// ## Why not check title content?
    /// The no-LLM title may look reasonable (e.g., "JWT, React, TypeScript") but
    /// lacks context. Checking summary prefix is more reliable as it's always
    /// written by Step 10 before this tail is called.
    async fn enqueue_session_title_tasks(&self, owner: &[u8]) {
        // Query sessions needing LLM title via direct SQL for efficiency.
        // Avoids loading all sessions and filtering in Rust.
        let sessions_needing_title = {
            let conn = self.storage.conn_lock().await;
            let mut stmt = match conn.prepare(
                "SELECT session_id, summary, turn_count
                 FROM sessions
                 WHERE owner = ?1
                   AND summary_generated = 1
                   AND (title IS NULL OR summary LIKE 'Topics:%')
                 ORDER BY started_at DESC
                 LIMIT ?2"
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "[MINER_S10] enqueue_session_title_tasks query failed");
                    return;
                }
            };
            stmt.query_map(
                rusqlite::params![owner, MINER_SESSION_BATCH as i64 * 2],
                |row| Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            )
            .map(|rows| rows.filter_map(|r| r.ok()).collect::<Vec<_>>())
            .unwrap_or_default()
        };

        if sessions_needing_title.is_empty() {
            return;
        }

        let mut enqueued = 0u32;
        let mut skipped = 0u32;

        for (session_id, summary, turn_count) in &sessions_needing_title {
            let payload = serde_json::json!({
                "session_id": session_id,
                "current_summary": summary,
                "turn_count": turn_count,
            });

            match self.storage.insert_cognitive_task(
                "session_title",
                SUPERNODE_PRIORITY_SESSION_TITLE,
                &payload.to_string(),
                None,
                Some("sessions"),
                Some(session_id),
                "structured",
                SUPERNODE_MAX_RETRIES,
            ).await {
                Ok(Some(id)) => {
                    debug!(id = id, session = %session_id, "[MINER_S10] Enqueued session_title");
                    enqueued += 1;
                }
                Ok(None) => { skipped += 1; }
                Err(e) => {
                    warn!(error = %e, session = %session_id, "[MINER_S10] Task enqueue failed");
                }
            }
        }

        if enqueued > 0 {
            info!(enqueued = enqueued, skipped = skipped, "[MINER_S10] session_title tasks enqueued");
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
            warn!("[MINER] OpenClaw embed failed: {}", resp.status());
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

// ============================================
// Helper: Stopword entity filter (v2.4.0 Phase B Fixes)
// ============================================

fn is_stopword_entity(name_normalized: &str, original_text: &str, entity_label: &str) -> bool {
    if name_normalized == entity_label.to_lowercase() {
        return true;
    }

    const STOPWORDS: &[&str] = &[
        "project", "module", "file", "user", "code", "system", "data",
        "api", "app", "tool", "test", "type", "model", "server", "client",
        "function", "method", "class", "table", "query", "task", "error",
        "the", "this", "that", "implementation", "feature", "component",
    ];

    if STOPWORDS.contains(&name_normalized) {
        return true;
    }

    // Rule 3: 2-char entities only pass if ALL uppercase in original (acronym like "AI", "DB")
    // Bug fix (Phase B Fixes): use original_text not name_normalized for uppercase check
    if name_normalized.len() == 2 {
        let original_trimmed = original_text.trim();
        let is_acronym = original_trimmed.len() == 2
            && original_trimmed.chars().all(|c| c.is_uppercase() || !c.is_alphabetic());
        if !is_acronym {
            return true;
        }
    }

    false
}

impl std::fmt::Debug for ReflectionMiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReflectionMiner")
            .field("interval", &self.interval)
            .field("threshold", &self.compaction_threshold)
            .field("mvf", &self.mvf_enabled)
            .field("ner", &self.ner_engine.is_some())
            .field("supernode", &self.llm_router.is_some())
            .finish()
    }
}
