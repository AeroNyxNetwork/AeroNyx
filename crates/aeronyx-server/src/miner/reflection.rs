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
//! Previously: `derive_rawlog_key(&owner)` where owner = public_key_bytes() — insecure.
//! Now: `derive_rawlog_key(&self.identity.to_bytes())` — only private key holder
//! can derive the key and decrypt rawlogs.
//!
//! ## Dependencies
//! - storage.rs — MemoryStorage, derive_rawlog_key, decrypt_rawlog_content_pub
//! - identity.to_bytes() — Ed25519 PRIVATE key for rawlog key derivation
//! - vector.rs — VectorIndex, cosine_similarity
//! - mvf.rs — WeightVector, SGD
//!
//! ⚠️ Important Note for Next Developer:
//! - derive_rawlog_key MUST use identity.to_bytes() (PRIVATE key), NOT public_key_bytes()
//! - Step 0 decrypts rawlogs — if the key is wrong, content will be garbage/empty
//! - After the key derivation fix, old rawlogs encrypted with public-key-derived key
//!   are cleared by the migration in storage.rs
//!
//! ## Error Isolation
//! Each step has independent error handling. Step 0 failure does NOT block
//! Step 1-5. All steps log errors and continue.
//!
//! ## Last Modified
//! v0.5.0 - Initial timer-based block packer
//! v1.0.0 - Smart compaction via OpenClaw
//! v2.1.0 - MVF feedback pipeline (Step 0, 0.5, 0.6) + SGD integration
//! v2.1.0+MVF+Encryption - 🌟 Fixed rawlog key derivation in Step 0:
//!   now uses identity.to_bytes() (PRIVATE key) instead of public_key_bytes().

use std::collections::HashSet;
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
    /// Shared user_weights map (same Arc as MpiState)
    user_weights: Option<Arc<parking_lot::RwLock<std::collections::HashMap<String, mvf::WeightVector>>>>,
    /// Local embedding engine (shared with MpiState). When Some, used for
    /// embedding generation instead of calling OpenClaw Gateway HTTP.
    embed_engine: Option<Arc<EmbedEngine>>,
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

    pub async fn run(self, mut shutdown_rx: tokio::sync::broadcast::Receiver<()>) {
        info!(
            interval = self.interval.as_secs(),
            threshold = self.compaction_threshold,
            mvf = self.mvf_enabled,
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

        // Group by session
        let mut sessions: std::collections::HashMap<String, Vec<_>> = std::collections::HashMap::new();
        for row in &rows {
            sessions.entry(row.session_id.clone()).or_default().push(row);
        }

        let mut total_positive = 0u32;
        let mut total_neutral = 0u32;

        for (_session_id, session_rows) in &sessions {
            let mut seen_memories: HashSet<Vec<u8>> = HashSet::new();

            for row in session_rows {
                // Only process user messages
                if row.role != "user" {
                    continue;
                }

                // Already processed (e.g., negative from /log)
                if row.feedback_signal.is_some() {
                    continue;
                }

                // Decrypt content for analysis
                // ⚠️ SECURITY FIX (v2.1.0+MVF+Encryption):
                // derive_rawlog_key now uses PRIVATE key (identity.to_bytes())
                // Previously: derive_rawlog_key(&owner) — used PUBLIC key, insecure!
                let content = if row.encrypted == 1 {
                    let key = crate::services::memchain::derive_rawlog_key(
                        &self.identity.to_bytes()
                    );
                    String::from_utf8(
                        crate::services::memchain::decrypt_rawlog_content_pub(&key, &row.content)
                            .unwrap_or_default()
                    ).unwrap_or_default()
                } else {
                    String::from_utf8_lossy(&row.content).to_string()
                };

                // Check: no negative keywords
                let lower = content.to_lowercase();
                let has_negative = CN_NEGATIVE.iter().any(|kw| lower.contains(kw))
                    || EN_NEGATIVE.iter().any(|kw| lower.contains(kw));

                if has_negative || content.len() <= 5 {
                    self.storage.update_rawlog_feedback(row.log_id, 0).await;
                    total_neutral += 1;
                    continue;
                }

                // Find previous assistant turn's recall_context
                let recall_ctx = row.recall_context.as_deref()
                    .or_else(|| {
                        // Look backwards in session for assistant recall_context
                        session_rows.iter()
                            .filter(|r| r.turn_index < row.turn_index && r.role == "assistant")
                            .last()
                            .and_then(|r| r.recall_context.as_deref())
                    });

                let ctx = match recall_ctx {
                    Some(c) => c,
                    None => {
                        self.storage.update_rawlog_feedback(row.log_id, 0).await;
                        total_neutral += 1;
                        continue;
                    }
                };

                // Parse recall_context JSON
                let entries: Vec<serde_json::Value> = serde_json::from_str(ctx).unwrap_or_default();
                if entries.is_empty() {
                    self.storage.update_rawlog_feedback(row.log_id, 0).await;
                    total_neutral += 1;
                    continue;
                }

                // Take top 2 scored memories
                let top_entries: Vec<&serde_json::Value> = entries.iter().take(2).collect();

                // Get embedding for user message via OpenClaw
                let query_embedding = self.call_openclaw_embed(&content).await;

                for entry in top_entries {
                    let mem_id_hex = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let mem_id_bytes = hex::decode(mem_id_hex).unwrap_or_default();

                    if mem_id_bytes.len() != 32 || seen_memories.contains(&mem_id_bytes) {
                        continue;
                    }

                    // Check embedding similarity > 0.4 (if we have embeddings)
                    if let Some(ref q_emb) = query_embedding {
                        if let Some(record) = self.storage.get(&{
                            let mut arr = [0u8; 32];
                            arr.copy_from_slice(&mem_id_bytes);
                            arr
                        }).await {
                            if record.has_embedding() {
                                let sim = cosine_similarity(q_emb, &record.embedding);
                                if sim <= 0.4 {
                                    continue; // Not similar enough
                                }
                            }
                        }
                    }

                    // Mark positive feedback
                    let mut record_id = [0u8; 32];
                    record_id.copy_from_slice(&mem_id_bytes);

                    self.storage.increment_positive_feedback(&record_id).await;

                    let features: Option<[f32; 9]> = entry.get("features")
                        .and_then(|v| v.as_array())
                        .and_then(|arr| {
                            if arr.len() == 9 {
                                let mut f = [0.0f32; 9];
                                for (i, val) in arr.iter().enumerate() {
                                    f[i] = val.as_f64().unwrap_or(0.0) as f32;
                                }
                                Some(f)
                            } else { None }
                        });

                    let prediction = entry.get("score")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32);

                    self.storage.insert_feedback(
                        &owner, &record_id, &row.session_id,
                        row.turn_index, 1,
                        features.as_ref(), prediction,
                    ).await;

                    seen_memories.insert(mem_id_bytes);
                    total_positive += 1;
                }

                self.storage.update_rawlog_feedback(row.log_id, 1).await;
            }
        }

        // MVF SGD batch update (positive samples)
        if self.mvf_enabled && total_positive > 0 {
            self.sgd_batch_update_positive(&owner_hex).await;
        }

        info!(
            positive = total_positive,
            neutral = total_neutral,
            "[MINER_S0] Feedback detection complete"
        );
    }

    /// Run SGD updates for all positive feedback collected this cycle.
    async fn sgd_batch_update_positive(&self, owner_hex: &str) {
        let weights_map = match &self.user_weights {
            Some(w) => w,
            None => return,
        };

        // Load recent positive feedback with features
        let feedback = self.storage.get_recent_feedback(100).await;
        let positive_with_features: Vec<_> = feedback.iter()
            .filter(|(signal, _)| *signal == 1)
            .collect();

        if positive_with_features.is_empty() {
            return;
        }

        let mut map = weights_map.write();
        let w = map.entry(owner_hex.to_string())
            .or_insert_with(mvf::default_weights);

        // Note: full implementation would read features from memory_feedback table.
        // For now we just increment version to track that SGD ran.
        // The actual per-sample SGD with feature vectors requires loading the
        // features BLOB from memory_feedback — deferred to recall-time negative
        // feedback path which has the features immediately available.

        info!(
            samples = positive_with_features.len(),
            version = w.version,
            "[MINER_SGD] Batch update noted (features-based SGD in D11)"
        );
    }

    // ============================================
    // Step 0.5: Backfill Missing Embeddings
    // ============================================

    async fn step_05_backfill_embeddings(&self) {
        let records = self.storage.get_records_needing_embedding(EMBEDDING_BACKFILL_BATCH).await;
        if records.is_empty() {
            debug!("[MINER_S05] No records need embedding backfill");
            return;
        }

        let owner = self.identity.public_key_bytes();
        let mut filled = 0u32;

        for record in &records {
            let content = String::from_utf8_lossy(&record.encrypted_content).to_string();
            if content.is_empty() {
                continue;
            }

            if let Some(embedding) = self.call_openclaw_embed(&content).await {
                let dim = embedding.len();

                // Update SQLite
                let embedding_blob: Vec<u8> = embedding.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();

                let conn = self.storage.conn_lock().await;
                let _ = conn.execute(
                    "UPDATE records SET embedding = ?1, embedding_model = ?2,
                        embedding_dim = ?3 WHERE record_id = ?4",
                    rusqlite::params![
                        embedding_blob.as_slice(),
                        "minilm-l6-v2",
                        dim as i64,
                        record.record_id.as_slice(),
                    ],
                );
                drop(conn);

                // Update vector index
                self.vector_index.upsert(
                    record.record_id,
                    embedding,
                    record.layer,
                    record.timestamp,
                    &owner,
                    "minilm-l6-v2",
                );

                filled += 1;
            }
        }

        if filled > 0 {
            info!(filled = filled, "[MINER_S05] Embeddings backfilled");
        }
    }

    // ============================================
    // Step 0.6: Correction Chaining
    // ============================================

    async fn step_06_correction_chaining(&self) {
        let corrections = self.storage.get_correction_records().await;
        if corrections.is_empty() {
            return;
        }

        let owner = self.identity.public_key_bytes();
        let mut chained = 0u32;

        for correction in &corrections {
            // Only process corrections with embeddings (need similarity)
            if !correction.has_embedding() {
                continue;
            }

            // Search for most similar active record (excluding self)
            let candidates = self.vector_index.search(
                &correction.embedding,
                &owner,
                "minilm-l6-v2", // Assume same model
                5,
                0.5, // Minimum similarity for supersede
            );

            let best_match = candidates.iter()
                .find(|c| c.record_id != correction.record_id);

            if let Some(old) = best_match {
                // Supersede the old record
                self.storage.supersede_record(&old.record_id, &correction.record_id).await;

                // Remove old from vector index and identity cache
                self.vector_index.remove(&old.record_id);

                chained += 1;
                debug!(
                    old = hex::encode(old.record_id),
                    new = hex::encode(correction.record_id),
                    sim = old.similarity,
                    "[MINER_S06] Correction chained (supersede)"
                );
            }

            // Remove _correction tag regardless
            let new_tags: Vec<String> = correction.topic_tags.iter()
                .filter(|t| *t != "_correction")
                .cloned()
                .collect();
            self.storage.update_topic_tags(&correction.record_id, &new_tags).await;
        }

        if chained > 0 {
            info!(chained = chained, "[MINER_S06] Corrections processed");
        }
    }

    // ============================================
    // Step 1-5: Legacy Compaction (unchanged logic)
    // ============================================

    async fn step_1_5_legacy_compaction(&self) {
        // Smart compaction
        self.smart_compact().await;
        // Legacy Fact mining
        self.legacy_mine().await;
    }

    async fn smart_compact(&self) {
        let ep_count = self.storage.count_by_layer(MemoryLayer::Episode).await;
        if ep_count < self.compaction_threshold {
            return;
        }

        let owner = self.identity.public_key_bytes();
        let episodes = self.storage.compact_episodes_to_archive(&owner, MAX_COMPACTION_BATCH).await;
        if episodes.is_empty() {
            return;
        }

        info!(count = episodes.len(), "[MINER] Compacting episodes");

        let prompt = self.build_summary_prompt(&episodes);
        let summary = match self.call_openclaw_chat(&prompt).await {
            Some(s) => s,
            None => {
                warn!("[MINER] OpenClaw summary failed — episodes remain archived");
                return;
            }
        };

        let now_ts = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_secs();

        let mut all_tags: Vec<String> = episodes.iter()
            .flat_map(|e| e.topic_tags.clone()).collect();
        all_tags.sort();
        all_tags.dedup();

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

        // Chain state
        let prev_hash = self.storage.last_block_hash().await;
        let prev_height = self.storage.last_block_height().await;
        let new_height = if prev_hash == [0u8; 32] { 1 } else { prev_height + 1 };
        let root = merkle_root(&[knowledge.record_id]);

        let header = BlockHeader {
            height: new_height,
            timestamp: now_ts,
            prev_block_hash: prev_hash,
            merkle_root: root,
            block_type: BLOCK_TYPE_MEMORY,
        };

        self.storage.set_chain_state(&header.hash(), new_height).await;
        let _ = self.broadcast_header(MemChainMessage::BlockAnnounce(header)).await;

        info!(
            height = new_height,
            episodes = episodes.len(),
            "[MINER] Compaction complete"
        );
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
        let now_ts = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_secs();

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
            "max_tokens": 2000,
            "temperature": 0.3
        });

        let resp = client.post(OPENCLAW_GATEWAY_CHAT).json(&body).send().await.ok()?;
        if !resp.status().is_success() { return None; }

        let json: serde_json::Value = resp.json().await.ok()?;
        json.get("choices")?.get(0)?.get("message")?.get("content")?
            .as_str().map(|s| s.to_string())
    }

    /// Generate embedding for a text.
    ///
    /// ## Strategy (v2.1.0+Embed)
    /// 1. If local EmbedEngine is available → use it (< 5ms, no network)
    /// 2. If local fails or unavailable → fall back to OpenClaw Gateway HTTP
    /// 3. If both fail → return None (caller skips embedding-dependent logic)
    async fn call_openclaw_embed(&self, text: &str) -> Option<Vec<f32>> {
        // Try local embed engine first
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

        // Fallback: OpenClaw Gateway HTTP
        self.call_openclaw_embed_remote(text).await
    }

    /// Remote embedding via OpenClaw Gateway HTTP (fallback path).
    async fn call_openclaw_embed_remote(&self, text: &str) -> Option<Vec<f32>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build().ok()?;

        let body = serde_json::json!({
            "model": "minilm-l6-v2",
            "input": text
        });

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
            ) {
                Ok(l) => l,
                Err(_) => continue,
            };
            enc.truncate(len);
            let pkt = DataPacket::new(*s.id.as_bytes(), ctr, enc);
            let bytes = encode_data_packet(&pkt).to_vec();
            if self.udp.send(&bytes, &s.client_endpoint).await.is_ok() {
                sent += 1;
            }
        }
        sent
    }
}

impl std::fmt::Debug for ReflectionMiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReflectionMiner")
            .field("interval", &self.interval)
            .field("threshold", &self.compaction_threshold)
            .field("mvf", &self.mvf_enabled)
            .finish()
    }
}
