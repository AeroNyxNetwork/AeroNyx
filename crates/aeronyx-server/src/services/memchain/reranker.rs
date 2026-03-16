// ============================================
// File: crates/aeronyx-server/src/services/memchain/reranker.rs
// ============================================
//! # RerankerEngine — Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
//!
//! ## Creation Reason (v2.4.0+Reranker)
//! Provides cross-encoder reranking for the recall pipeline. Unlike bi-encoders
//! (EmbedEngine) that encode query and document separately, cross-encoders
//! concatenate query+document and produce a single relevance score. This
//! captures fine-grained query-document interaction that bi-encoders miss.
//!
//! ## Architecture Position
//! ```text
//! recall_handler.rs pipeline:
//!   Step 2a: Vector search → candidates
//!   Step 2a-bis: BM25 search → candidates
//!   Step 2b-2c: Graph traversal → candidates
//!   Step 3: RRF fusion → merged candidates (top 30)
//!   Step 3.5: ★ Cross-encoder rerank (THIS MODULE) → reordered top_k
//!   Step 4: Token budget trimming → final response
//! ```
//!
//! ## Model
//! `cross-encoder/ms-marco-MiniLM-L-6-v2` — 22MB ONNX
//! - Input: [CLS] query [SEP] document [SEP] → single relevance logit
//! - Output: float32 relevance score (higher = more relevant)
//! - Latency: ~1ms per (query, document) pair on CPU
//! - 30 pairs → ~30ms total (acceptable for recall endpoint)
//!
//! ## ONNX Input Format
//! Standard BERT-style 3-tensor input:
//! - input_ids: [batch, seq_len] — tokenized [CLS] query [SEP] doc [SEP]
//! - attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
//! - token_type_ids: [batch, seq_len] — 0 for query tokens, 1 for doc tokens
//!
//! ## Performance
//! - 30 pairs × ~1ms = ~30ms rerank latency
//! - Memory: ~22MB model weights + tokenizer
//! - Output: single float per pair (no pooling needed)
//!
//! ## Fallback
//! If model files are missing or reranker disabled:
//! - `RerankerEngine::load()` returns Err → server starts without reranker
//! - recall pipeline skips Step 3.5, returns RRF-fused results directly
//! - `/api/mpi/status` reports `reranker_ready: false`
//!
//! ## Dependencies
//! - `ort` 2.0.0-rc.11 — shared ORT runtime with embed.rs and ner.rs
//! - `tokenizers` 0.21 — shared with embed.rs and ner.rs
//!
//! ⚠️ Important Note for Next Developer:
//! - Cross-encoder output is a RAW LOGIT, not a probability. Higher = more relevant.
//!   Do NOT apply sigmoid — the raw score is used directly for ranking.
//! - The tokenizer MUST be the cross-encoder's own tokenizer (BERT-base uncased),
//!   NOT the MiniLM embedding tokenizer or GLiNER DeBERTa tokenizer.
//! - max_seq_length for cross-encoder is typically 512 (query+doc combined).
//!   Truncation happens on the DOCUMENT side, not the query.
//! - ORT runtime is shared via Once — safe to load alongside embed + ner.
//! - Session::run() requires &mut self → Mutex wrapper (same pattern).
//! - Batch inference (multiple pairs at once) is much more efficient than
//!   one-by-one. Always use `rerank_batch()`.
//!
//! ## Last Modified
//! v2.4.0+Reranker - 🌟 Initial implementation

use std::path::Path;
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::{debug, info, warn};

// ============================================
// Constants
// ============================================

const MODEL_FILENAME: &str = "model.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Default max sequence length for cross-encoder (query + document combined).
/// ms-marco-MiniLM-L-6-v2 supports up to 512 tokens.
const DEFAULT_MAX_SEQ_LENGTH: usize = 512;

/// Maximum number of (query, document) pairs per batch.
/// 50 is generous — typical usage is 30.
const MAX_BATCH_SIZE: usize = 50;

/// Default number of candidates to rerank.
/// The recall pipeline sends the top 30 merged candidates.
pub const DEFAULT_RERANK_TOP_N: usize = 30;

// ============================================
// RerankerEngine
// ============================================

/// Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 ONNX model.
///
/// Thread-safe: `ort::Session` wrapped in `Mutex` (same pattern as EmbedEngine/NerEngine).
///
/// ## Usage
/// ```rust,ignore
/// let engine = RerankerEngine::load("models/reranker", 512)?;
/// let scores = engine.rerank_batch("what is RS256?", &[
///     "RS256 uses RSA public-key cryptography for JWT signing",
///     "User likes spicy food",
/// ])?;
/// // scores[0] >> scores[1] (first doc much more relevant)
/// ```
pub struct RerankerEngine {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_seq_length: usize,
}

/// A reranked candidate with its cross-encoder score.
#[derive(Debug, Clone)]
pub struct RerankedCandidate {
    /// Index into the original candidates array.
    pub original_index: usize,
    /// Cross-encoder relevance score (raw logit, higher = better).
    pub ce_score: f32,
}

impl RerankerEngine {
    /// Load cross-encoder ONNX model and tokenizer.
    ///
    /// ## Prerequisites
    /// ORT runtime must be initialized (by EmbedEngine::load or init_ort_runtime).
    ///
    /// ## Arguments
    /// * `model_dir` - Directory containing model.onnx and tokenizer.json
    /// * `max_seq_length` - Max combined query+document length (default 512)
    pub fn load(
        model_dir: impl AsRef<Path>,
        max_seq_length: usize,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let max_seq_length = if max_seq_length == 0 { DEFAULT_MAX_SEQ_LENGTH } else { max_seq_length };

        let model_path = model_dir.join(MODEL_FILENAME);
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);

        if !model_path.exists() {
            return Err(format!(
                "Reranker ONNX model not found: {} — run scripts/download_models.sh --reranker-only",
                model_path.display()
            ));
        }
        if !tokenizer_path.exists() {
            return Err(format!(
                "Reranker tokenizer not found: {} — run scripts/download_models.sh --reranker-only",
                tokenizer_path.display()
            ));
        }

        let session = Session::builder()
            .map_err(|e| format!("Reranker session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Reranker optimization: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("Reranker threads: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("Reranker model load ({}): {}", model_path.display(), e))?;

        info!(model = %model_path.display(), "[RERANKER] ONNX model loaded");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Reranker tokenizer ({}): {}", tokenizer_path.display(), e))?;

        info!(
            tokenizer = %tokenizer_path.display(),
            max_seq_length = max_seq_length,
            "[RERANKER] Tokenizer loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_seq_length,
        })
    }

    /// Rerank a batch of documents against a query.
    ///
    /// ## Arguments
    /// * `query` - The search query
    /// * `documents` - Candidate document texts to rerank
    ///
    /// ## Returns
    /// `Vec<RerankedCandidate>` sorted by ce_score descending (best first).
    /// Each entry contains the original index and the cross-encoder score.
    ///
    /// ## Pipeline
    /// 1. For each document, create "[CLS] query [SEP] document [SEP]" input
    /// 2. Batch tokenize with padding + truncation
    /// 3. ONNX inference → one logit per pair
    /// 4. Sort by logit descending
    pub fn rerank_batch(
        &self,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<RerankedCandidate>, String> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        if documents.len() > MAX_BATCH_SIZE {
            return Err(format!(
                "Rerank batch size {} exceeds max {}",
                documents.len(), MAX_BATCH_SIZE
            ));
        }

        let batch_size = documents.len();

        // Step 1: Tokenize each (query, document) pair
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_seq_length,
                ..Default::default()
            }))
            .map_err(|e| format!("Reranker truncation: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // Cross-encoder input: encode pairs (query, document)
        let pairs: Vec<(String, String)> = documents.iter()
            .map(|doc| (query.to_string(), doc.to_string()))
            .collect();

        let encodings = tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| format!("Reranker tokenization: {}", e))?;

        if encodings.is_empty() {
            return Ok(Vec::new());
        }

        let seq_len = encodings[0].get_ids().len();
        let total = batch_size * seq_len;

        // Step 2: Build tensors
        let mut input_ids = Vec::with_capacity(total);
        let mut attention_mask = Vec::with_capacity(total);
        let mut token_type_ids = Vec::with_capacity(total);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();

            for i in 0..seq_len {
                input_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                attention_mask.push(mask.get(i).copied().unwrap_or(0) as i64);
                token_type_ids.push(types.get(i).copied().unwrap_or(0) as i64);
            }
        }

        let shape = [batch_size, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids.into_boxed_slice()))
            .map_err(|e| format!("Reranker input_ids tensor: {}", e))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask.into_boxed_slice()))
            .map_err(|e| format!("Reranker attention_mask tensor: {}", e))?;
        let types_tensor = Tensor::from_array((shape, token_type_ids.into_boxed_slice()))
            .map_err(|e| format!("Reranker token_type_ids tensor: {}", e))?;

        // Step 3: ONNX inference
        let mut session = self.session.lock()
            .map_err(|e| format!("Reranker session lock: {}", e))?;

        let outputs = session.run(
            ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => types_tensor,
            ]
        ).map_err(|e| format!("Reranker inference: {}", e))?;

        // Output: logits [batch_size, 1] or [batch_size]
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Reranker output: {}", e))?;

        let logits_shape = logits.shape();

        // Step 4: Extract scores and sort
        let mut candidates: Vec<RerankedCandidate> = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let score = if logits_shape.len() == 2 {
                logits[[i, 0]] // [batch, 1]
            } else if logits_shape.len() == 1 {
                logits[[i]] // [batch]
            } else {
                // Unexpected shape — try flat index
                let flat = logits.as_slice().unwrap_or(&[]);
                flat.get(i).copied().unwrap_or(0.0)
            };

            candidates.push(RerankedCandidate {
                original_index: i,
                ce_score: score,
            });
        }

        // Sort by score descending
        candidates.sort_by(|a, b| {
            b.ce_score.partial_cmp(&a.ce_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!(
            batch = batch_size,
            seq_len = seq_len,
            top_score = format!("{:.3}", candidates.first().map(|c| c.ce_score).unwrap_or(0.0)),
            "[RERANKER] Batch rerank complete"
        );

        Ok(candidates)
    }

    /// Rerank a single (query, document) pair. Returns the relevance score.
    pub fn rerank_single(&self, query: &str, document: &str) -> Result<f32, String> {
        let results = self.rerank_batch(query, &[document])?;
        Ok(results.first().map(|c| c.ce_score).unwrap_or(0.0))
    }

    /// Returns the configured max sequence length.
    #[must_use]
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }
}

impl std::fmt::Debug for RerankerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RerankerEngine")
            .field("max_seq_length", &self.max_seq_length)
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_model_returns_error() {
        let result = RerankerEngine::load("/nonexistent/path", 512);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    fn reranker_model_dir() -> String {
        std::env::var("MEMCHAIN_RERANKER_MODEL_PATH")
            .unwrap_or_else(|_| "models/reranker".to_string())
    }

    fn try_load_reranker() -> Option<RerankerEngine> {
        match RerankerEngine::load(&reranker_model_dir(), 512) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("⏭️ Skipping reranker test (model not available): {}", e);
                None
            }
        }
    }

    #[test]
    fn test_rerank_basic() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        let scores = engine.rerank_batch(
            "What signing algorithm does the auth module use?",
            &[
                "The auth module uses RS256 signing with the ring crate for RSA operations",
                "User likes spicy food and is allergic to shellfish",
                "Project Alpha was started last month by the engineering team",
            ],
        ).unwrap();

        assert_eq!(scores.len(), 3);
        // The first document should score highest (most relevant)
        assert_eq!(scores[0].original_index, 0,
            "RS256 document should be ranked first, got index {}", scores[0].original_index);
    }

    #[test]
    fn test_rerank_empty() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        let scores = engine.rerank_batch("test query", &[]).unwrap();
        assert!(scores.is_empty());
    }

    #[test]
    fn test_rerank_single() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        let score = engine.rerank_single(
            "rate limiting",
            "Token bucket rate limiting at 100 requests per minute",
        ).unwrap();

        // Should be a positive relevance score
        assert!(score > 0.0, "Relevant pair should have positive score, got {}", score);
    }
}
