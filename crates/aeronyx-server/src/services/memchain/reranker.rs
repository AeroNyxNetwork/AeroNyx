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
//! - Output: float32 relevance score (raw logit, higher = more relevant)
//! - Latency: ~1ms per (query, document) pair on CPU
//! - 30 pairs → ~30ms total (acceptable for recall endpoint)
//!
//! ## ONNX Input Format
//! Standard BERT-style 3-tensor input:
//! - input_ids: [batch, seq_len] — tokenized [CLS] query [SEP] doc [SEP]
//! - attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding
//! - token_type_ids: [batch, seq_len] — 0 for query tokens, 1 for doc tokens
//!
//! ## Score Normalization
//! Raw CE logits are in an unbounded range (empirically ~[-8, +8] for ms-marco).
//! We use dynamic min-max normalization across the batch to scale scores to [0, 1]:
//!   normalized = (score - min) / (max - min + epsilon)
//! This is then blended with the original RRF score:
//!   final = CE_BLEND_WEIGHT * normalized_ce + (1 - CE_BLEND_WEIGHT) * rrf_score
//! Fallback: if all CE scores are identical (degenerate batch), sigmoid is used instead.
//!
//! ## ORT Runtime Sharing
//! init_ort_runtime() in embed.rs uses a serialized success-only gate.
//! RerankerEngine::load() calls it with the reranker model directory — if ORT
//! is already initialized (by EmbedEngine), the existing runtime is reused. If
//! an earlier engine failed to find a runtime, reranker may retry from its own
//! independently configured model directory.
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
//! - Cross-encoder output is a RAW LOGIT, not a probability. Do NOT apply sigmoid
//!   for ranking — use raw scores for sort order. Sigmoid is only used as a
//!   normalization fallback when the entire batch has identical scores.
//! - The tokenizer MUST be the cross-encoder's own tokenizer (BERT-base uncased).
//!   NOT the MiniLM embedding tokenizer or GLiNER DeBERTa tokenizer.
//! - max_seq_length for cross-encoder is 512 (query+doc combined). Truncation
//!   happens on the DOCUMENT side (TruncationParams::max_length applies to the pair).
//! - ORT runtime is shared via the retryable gate in embed.rs — safe to load
//!   alongside embed + ner.
//! - Session::run() requires &mut self → shared `InferenceSession` wrapper
//!   (same recovery contract as embed.rs and ner.rs).
//! - Always use rerank_batch() — it processes all pairs in one ONNX session.run() call,
//!   which is significantly faster than calling one-by-one due to batching overhead.
//! - tokenizer is cloned and configured per-call (same pattern as embed.rs embed_minilm).
//!   This is the project-standard pattern for tokenizers 0.21 + Rust Send + Sync.
//! - CE_BLEND_WEIGHT = 0.7 is the initial value. If retrieval quality tests show
//!   the reranker is too aggressive (degrading good BM25/graph results), lower to 0.5.
//!
//! ## Last Modified
//! v2.4.6-ModelSequenceBounds -
//!   [MEMCHAIN-MODEL-SEQUENCE-BOUNDS 2026-07-30 by Codex] Rejects sequence
//!   limits above the cross-encoder's positional capacity during engine load.
//! v2.4.5-TokenizerBatchBoundary -
//!   [MEMCHAIN-TOKENIZER-BATCH 2026-07-30 by Codex] Reused the shared padded
//!   batch contract and stopped silently fabricating missing tensor fields.
//! v2.4.4-OnnxOutputBoundary -
//!   [MEMCHAIN-ONNX-OUTPUT-BOUNDARY 2026-07-30 by Codex] Replaced panicking
//!   positional output indexing with the shared bounds-checked inference
//!   boundary.
//! v2.4.3-RerankerOutputValidation -
//!   [MEMCHAIN-RERANKER-OUTPUT 2026-07-30 by Codex] Added strict ONNX output
//!   shape/cardinality validation, rejected non-finite logits, and moved score
//!   normalization to f64 so malformed model output cannot silently mis-rank recall.
//! v2.4.2-RetryableOrtInitialization -
//!   [MEMCHAIN-ORT-RECOVERY 2026-07-30 by Codex] Documented the shared
//!   success-only initialization contract used by all inference engines.
//! v2.4.1-InferenceSessionRecovery -
//!   [MEMCHAIN-INFERENCE-SESSION 2026-07-30 by Codex] Reused the shared
//!   non-poisoning ONNX session boundary.
//! v2.4.0+Reranker - 🌟 Initial implementation
//!   Bug fixes vs original spec:
//!   - RERANK_TOP_N: Option<usize> → usize (the Option wrapper was meaningless)
//!   - CE score normalization: hardcoded +10/6 range → dynamic min-max normalization
//!     with sigmoid fallback for degenerate batches (all-identical scores)
//!   - tokenizer clone pattern aligned with embed.rs (project standard)

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::{debug, info, warn};

// Re-use the ORT initialization and session boundary from embed.rs.
use super::embed::{
    init_ort_runtime, require_onnx_output, resolve_model_sequence_length, validate_tokenized_batch,
    InferenceSession,
};

// ============================================
// Constants
// ============================================

const MODEL_FILENAME: &str = "model.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Default max sequence length for cross-encoder (query + document combined).
/// ms-marco-MiniLM-L-6-v2 supports up to 512 tokens.
const DEFAULT_MAX_SEQ_LENGTH: usize = 512;

/// Maximum positional sequence length supported by ms-marco-MiniLM-L-6-v2.
const MODEL_MAX_SEQ_LENGTH: usize = 512;

/// Maximum number of (query, document) pairs per batch.
const MAX_BATCH_SIZE: usize = 50;

/// Number of RRF+graph candidates to send to cross-encoder reranker.
/// 30 pairs × ~1ms/pair ≈ 30ms added latency — acceptable for recall endpoint.
/// Used by recall_handler.rs to slice the candidate list before reranking.
pub const RERANK_TOP_N: usize = 30;

/// Weight of cross-encoder score in the final blended score.
/// Final = CE_BLEND_WEIGHT * normalized_ce + (1 - CE_BLEND_WEIGHT) * rrf_score
/// 0.7 means cross-encoder has strong influence but doesn't completely override RRF.
const CE_BLEND_WEIGHT: f64 = 0.7;

/// Minimum denominator for min-max normalization to avoid division by zero.
const NORM_EPSILON: f64 = 1e-6;

// ============================================
// Output contract and score normalization
// ============================================

/// Validate the ONNX cross-encoder output before it reaches recall ranking.
///
/// [MEMCHAIN-RERANKER-OUTPUT 2026-07-30 by Codex] The supported model exports
/// produce exactly one relevance logit per input pair, represented as either
/// `[batch]` or `[batch, 1]`. Treat every other shape, cardinality mismatch, and
/// non-finite value as an inference failure so recall can use its RRF fallback.
fn validate_reranker_logits(
    shape: &[usize],
    logits: Vec<f32>,
    expected_batch: usize,
) -> Result<Vec<f32>, String> {
    let shape_matches = match shape {
        [batch] => *batch == expected_batch,
        [batch, width] => *batch == expected_batch && *width == 1,
        _ => false,
    };

    if !shape_matches {
        return Err(format!(
            "Reranker output shape {:?} does not match expected [{batch}] or [{batch}, 1]",
            shape,
            batch = expected_batch
        ));
    }

    if logits.len() != expected_batch {
        return Err(format!(
            "Reranker output cardinality {} does not match batch {}",
            logits.len(),
            expected_batch
        ));
    }

    if let Some((index, score)) = logits
        .iter()
        .enumerate()
        .find(|(_, score)| !score.is_finite())
    {
        return Err(format!(
            "Reranker output contains non-finite logit at index {}: {}",
            index, score
        ));
    }

    Ok(logits)
}

/// Normalize validated finite logits to [0, 1] without overflowing f32.
///
/// The ONNX tensor remains f32, but range arithmetic is performed in f64 so
/// opposite finite f32 extremes cannot produce an infinite denominator and NaN.
fn normalize_reranker_scores(raw_scores: &[f32]) -> Vec<f64> {
    let score_min = raw_scores
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::INFINITY, f64::min);
    let score_max = raw_scores
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let score_range = score_max - score_min;

    if score_range > NORM_EPSILON {
        raw_scores
            .iter()
            .map(|&score| (f64::from(score) - score_min) / score_range)
            .collect()
    } else {
        raw_scores
            .iter()
            .map(|&score| 1.0 / (1.0 + (-f64::from(score)).exp()))
            .collect()
    }
}

// ============================================
// RerankerEngine
// ============================================

/// Cross-encoder reranker using ms-marco-MiniLM-L-6-v2 ONNX model.
///
/// Thread-safe: `ort::Session` is serialized through `InferenceSession`.
///
/// ## Usage
/// ```rust,ignore
/// let engine = RerankerEngine::load("models/reranker", 512)?;
/// let candidates = engine.rerank_batch("what is RS256?", &[
///     "RS256 uses RSA public-key cryptography for JWT signing",
///     "User likes spicy food",
/// ])?;
/// // candidates[0].original_index == 0 (first doc ranked highest)
/// // candidates[0].ce_score >> candidates[1].ce_score
/// ```
pub struct RerankerEngine {
    session: InferenceSession<Session>,
    tokenizer: Tokenizer,
    max_seq_length: usize,
}

/// A reranked candidate with its cross-encoder score and blended final score.
#[derive(Debug, Clone)]
pub struct RerankedCandidate {
    /// Index into the original candidates array (before reranking).
    pub original_index: usize,
    /// Raw cross-encoder logit (higher = more relevant). Use for debugging/logging.
    pub ce_score: f32,
    /// Normalized CE score in [0, 1] after min-max normalization.
    /// Used for blending with RRF score in recall_handler.rs Step 3.5.
    pub ce_score_normalized: f64,
}

impl RerankerEngine {
    /// Load cross-encoder ONNX model and tokenizer.
    ///
    /// Shares the ORT runtime with EmbedEngine/NerEngine via the retryable gate in embed.rs.
    /// If EmbedEngine was loaded first, ORT is already initialized — this is a no-op.
    /// If EmbedEngine was not loaded, this will initialize ORT from the reranker model dir.
    ///
    /// ## Arguments
    /// * `model_dir` - Directory containing model.onnx and tokenizer.json
    /// * `max_seq_length` - Max combined query+document token length (pass 0 for default 512)
    pub fn load(model_dir: impl AsRef<Path>, max_seq_length: usize) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let max_seq_length = resolve_model_sequence_length(
            "Reranker",
            max_seq_length,
            DEFAULT_MAX_SEQ_LENGTH,
            MODEL_MAX_SEQ_LENGTH,
        )?;

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

        // Initialize (or reuse) ORT runtime through the shared retryable gate.
        init_ort_runtime(model_dir)?;

        let session = Session::builder()
            .map_err(|e| format!("Reranker session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Reranker optimization: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("Reranker threads: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("Reranker model load ({}): {}", model_path.display(), e))?;

        info!(model = %model_path.display(), max_seq = max_seq_length, "[RERANKER] ONNX model loaded");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Reranker tokenizer ({}): {}", tokenizer_path.display(), e))?;

        info!(tokenizer = %tokenizer_path.display(), "[RERANKER] Tokenizer loaded");

        Ok(Self {
            session: InferenceSession::new(session),
            tokenizer,
            max_seq_length,
        })
    }

    /// Rerank a batch of documents against a query.
    ///
    /// Uses dynamic min-max normalization to scale raw CE logits to [0, 1].
    /// Falls back to sigmoid normalization if all scores are identical (degenerate batch).
    ///
    /// ## Returns
    /// `Vec<RerankedCandidate>` sorted by ce_score descending (best first).
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
                documents.len(),
                MAX_BATCH_SIZE
            ));
        }

        let batch_size = documents.len();

        // ── Tokenize (query, document) pairs ──
        // Clone tokenizer and configure truncation+padding (project-standard pattern
        // from embed.rs: tokenizer is Send but not Sync, so we clone per inference call).
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_seq_length,
                ..Default::default()
            }))
            .map_err(|e| format!("Reranker truncation config: {}", e))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        // Cross-encoder: encode (query, document) pairs as sentence pairs.
        // tokenizers encode_batch with add_special_tokens=true produces:
        //   [CLS] query tokens [SEP] doc tokens [SEP]
        // token_type_ids: 0 for query side, 1 for doc side.
        let pairs: Vec<(String, String)> = documents
            .iter()
            .map(|doc| (query.to_string(), doc.to_string()))
            .collect();

        let encodings = tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| format!("Reranker tokenization: {}", e))?;

        let (seq_len, total) = validate_tokenized_batch(
            "Reranker",
            &encodings,
            batch_size,
            self.max_seq_length,
            true,
        )?;

        // ── Build input tensors ──
        let mut input_ids = Vec::with_capacity(total);
        let mut attention_mask = Vec::with_capacity(total);
        let mut token_type_ids = Vec::with_capacity(total);

        for enc in &encodings {
            input_ids.extend(enc.get_ids().iter().copied().map(i64::from));
            attention_mask.extend(enc.get_attention_mask().iter().copied().map(i64::from));
            token_type_ids.extend(enc.get_type_ids().iter().copied().map(i64::from));
        }

        let shape = [batch_size, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids.into_boxed_slice()))
            .map_err(|e| format!("Reranker input_ids tensor: {}", e))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask.into_boxed_slice()))
            .map_err(|e| format!("Reranker attention_mask tensor: {}", e))?;
        let types_tensor = Tensor::from_array((shape, token_type_ids.into_boxed_slice()))
            .map_err(|e| format!("Reranker token_type_ids tensor: {}", e))?;

        // ── ONNX inference ──
        let mut session = self.session.lock();

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => types_tensor,
            ])
            .map_err(|e| format!("Reranker ONNX inference: {}", e))?;

        // Output: logits shaped [batch_size, 1] or [batch_size]
        let logits_output = require_onnx_output("Reranker", outputs.values(), 0)?;
        let logits = logits_output
            .try_extract_array::<f32>()
            .map_err(|e| format!("Reranker output extraction: {}", e))?;

        // [MEMCHAIN-RERANKER-OUTPUT 2026-07-30 by Codex] Validate the model
        // contract before indexing or ranking. A bad export must trigger the
        // recall pipeline's RRF fallback instead of fabricating zero scores.
        let raw_scores =
            validate_reranker_logits(logits.shape(), logits.iter().copied().collect(), batch_size)?;

        // ── Dynamic min-max normalization ──
        // Scale raw CE logits to [0, 1] for blending with RRF scores.
        // Fallback to sigmoid if all scores are identical (degenerate batch).
        let score_min = raw_scores
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::INFINITY, f64::min);
        let score_max = raw_scores
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let score_range = score_max - score_min;
        if score_range <= NORM_EPSILON {
            // Degenerate: all scores identical → sigmoid fallback
            // sigmoid(x) = 1 / (1 + e^-x), maps ℝ → (0, 1)
            warn!(
                batch = batch_size,
                score = score_min,
                "[RERANKER] All CE scores identical — using sigmoid normalization fallback"
            );
        }
        let normalized_scores = normalize_reranker_scores(&raw_scores);

        // ── Build candidates and sort by raw CE score descending ──
        let mut candidates: Vec<RerankedCandidate> = raw_scores
            .iter()
            .zip(normalized_scores.iter())
            .enumerate()
            .map(|(i, (&ce, &norm))| RerankedCandidate {
                original_index: i,
                ce_score: ce,
                ce_score_normalized: norm,
            })
            .collect();

        // Sort by raw CE score descending (raw logit is the true relevance signal)
        candidates.sort_by(|a, b| b.ce_score.total_cmp(&a.ce_score));

        debug!(
            batch = batch_size,
            seq_len = seq_len,
            top_ce = format!(
                "{:.3}",
                candidates.first().map(|c| c.ce_score).unwrap_or(0.0)
            ),
            top_norm = format!(
                "{:.3}",
                candidates
                    .first()
                    .map(|c| c.ce_score_normalized)
                    .unwrap_or(0.0)
            ),
            score_range = format!("{:.3}", score_range),
            "[RERANKER] Batch rerank complete"
        );

        Ok(candidates)
    }

    /// Rerank a single (query, document) pair. Returns the raw relevance score.
    pub fn rerank_single(&self, query: &str, document: &str) -> Result<f32, String> {
        let results = self.rerank_batch(query, &[document])?;
        Ok(results.first().map(|c| c.ce_score).unwrap_or(0.0))
    }

    /// Returns the configured max sequence length.
    #[must_use]
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Returns the CE blend weight used in recall_handler Step 3.5.
    /// Exposed for testing and transparency.
    #[must_use]
    pub fn blend_weight() -> f64 {
        CE_BLEND_WEIGHT
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
    fn test_missing_model_returns_error() {
        let result = RerankerEngine::load("/nonexistent/path", 512);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("not found"),
            "Error should mention 'not found': {}",
            err
        );
        assert!(
            err.contains("download_models.sh"),
            "Error should hint at download script: {}",
            err
        );
    }

    #[test]
    fn test_rerank_empty_docs_returns_empty() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };
        let result = engine.rerank_batch("test query", &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_rerank_batch_too_large_returns_error() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };
        let docs: Vec<&str> = (0..51).map(|_| "test").collect();
        let result = engine.rerank_batch("query", &docs);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds max"));
    }

    #[test]
    fn test_rerank_relevance_ordering() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        let candidates = engine
            .rerank_batch(
                "What signing algorithm does the auth module use?",
                &[
                    "The auth module uses RS256 signing with the ring crate for RSA operations",
                    "User likes spicy food and is allergic to shellfish",
                    "Project Alpha was started last month by the engineering team",
                ],
            )
            .unwrap();

        assert_eq!(candidates.len(), 3);
        // Most relevant doc (index 0) should be ranked first
        assert_eq!(
            candidates[0].original_index, 0,
            "RS256 document should rank first, got index {} (score={:.3})",
            candidates[0].original_index, candidates[0].ce_score
        );
        // Scores should be descending
        assert!(candidates[0].ce_score >= candidates[1].ce_score);
        assert!(candidates[1].ce_score >= candidates[2].ce_score);
    }

    #[test]
    fn test_rerank_single_returns_score() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        let score = engine
            .rerank_single(
                "rate limiting",
                "Token bucket rate limiting at 100 requests per minute using tower middleware",
            )
            .unwrap();

        // Highly relevant pair should have a positive raw logit
        assert!(
            score > 0.0,
            "Relevant pair should have positive score, got {}",
            score
        );
    }

    #[test]
    fn test_normalized_scores_in_range() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        let candidates = engine
            .rerank_batch(
                "JWT authentication",
                &[
                    "JWT uses RS256 for token signing",
                    "PostgreSQL database connection pool settings",
                    "React component lifecycle hooks",
                ],
            )
            .unwrap();

        for c in &candidates {
            assert!(
                c.ce_score_normalized >= 0.0 && c.ce_score_normalized <= 1.0,
                "Normalized score out of [0,1]: {} (raw={})",
                c.ce_score_normalized,
                c.ce_score
            );
        }
    }

    #[test]
    fn test_blend_weight_constant() {
        // Ensure the blend weight is in a sensible range
        let w = RerankerEngine::blend_weight();
        assert!(
            w > 0.0 && w < 1.0,
            "CE_BLEND_WEIGHT should be in (0, 1), got {}",
            w
        );
    }

    #[test]
    fn test_validate_reranker_logits_accepts_supported_shapes() {
        let vector = validate_reranker_logits(&[2], vec![0.25, -0.5], 2).unwrap();
        assert_eq!(vector, vec![0.25, -0.5]);

        let column = validate_reranker_logits(&[2, 1], vec![0.25, -0.5], 2).unwrap();
        assert_eq!(column, vec![0.25, -0.5]);
    }

    #[test]
    fn test_validate_reranker_logits_rejects_malformed_output() {
        let wrong_batch = validate_reranker_logits(&[1], vec![0.25], 2).unwrap_err();
        assert!(wrong_batch.contains("shape"));

        let multiple_logits =
            validate_reranker_logits(&[2, 2], vec![0.1, 0.2, 0.3, 0.4], 2).unwrap_err();
        assert!(multiple_logits.contains("shape"));

        let wrong_cardinality = validate_reranker_logits(&[2], vec![0.25], 2).unwrap_err();
        assert!(wrong_cardinality.contains("cardinality"));
    }

    #[test]
    fn test_validate_reranker_logits_rejects_non_finite_scores() {
        for score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = validate_reranker_logits(&[1], vec![score], 1).unwrap_err();
            assert!(error.contains("non-finite"));
        }
    }

    #[test]
    fn test_normalize_reranker_scores_handles_f32_extremes() {
        let normalized = normalize_reranker_scores(&[f32::MIN, 0.0, f32::MAX]);

        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0], 0.0);
        assert_eq!(normalized[2], 1.0);
        assert!(normalized.iter().all(|score| score.is_finite()));
        assert!(normalized.iter().all(|score| (0.0..=1.0).contains(score)));
    }

    #[test]
    fn test_rerank_single_doc() {
        let engine = match try_load_reranker() {
            Some(e) => e,
            None => return,
        };

        // Single-doc batch should not fail (no division issues with min-max)
        let candidates = engine.rerank_batch("test", &["a single document"]).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].original_index, 0);
        // With one doc, min == max → sigmoid fallback, score should be in (0, 1)
        assert!(candidates[0].ce_score_normalized > 0.0);
        assert!(candidates[0].ce_score_normalized < 1.0);
    }
}
