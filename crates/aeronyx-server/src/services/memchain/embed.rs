// ============================================
// File: crates/aeronyx-server/src/services/memchain/embed.rs
// ============================================
//! # EmbedEngine — Local MiniLM Embedding Inference
//!
//! ## Creation Reason
//! Provides local embedding generation for MemChain, eliminating the dependency
//! on OpenClaw Gateway being online. This makes MemChain a self-contained
//! cognitive engine where recall, remember, and Miner operations all work
//! independently of external services.
//!
//! ## Main Functionality
//! - Load MiniLM-L6-v2 ONNX model + HuggingFace tokenizer from disk
//! - Tokenize with WordPiece (matching training-time tokenizer exactly)
//! - ONNX inference via `ort` crate (CPU, with optional GPU via dynamic loading)
//! - Mean pooling (attention-mask-weighted) + L2 normalization → 384-dim unit vectors
//! - Batch support for Miner Step 0.5 backfill efficiency
//!
//! ## Model Files (downloaded via scripts/download_models.sh)
//! ```text
//! {embed_model_path}/          # default: models/minilm-l6-v2
//! ├── model.onnx               # ~22MB, 384-dim output
//! └── tokenizer.json           # ~700KB, HuggingFace fast tokenizer
//! ```
//!
//! ## Architecture Position
//! ```text
//! POST /api/mpi/embed                 Miner Step 0.5 backfill
//!       │                                    │
//!       ▼                                    ▼
//!   EmbedEngine.embed_batch(texts)    EmbedEngine.embed_single(text)
//!       │
//!       ├─ tokenizers::Tokenizer.encode_batch()
//!       ├─ ort::Session.run() → last_hidden_state [batch, seq_len, 384]
//!       ├─ mean_pool(hidden_states, attention_mask)
//!       │   └─ hidden_state * mask → sum → div by mask.sum()
//!       │      (padding tokens have non-zero hidden states — MUST mask them)
//!       └─ l2_normalize → Vec<f32> (384-dim unit vector)
//! ```
//!
//! ## Performance Targets
//! - Single text (~20 tokens): < 5ms on modern CPU
//! - Batch of 50 texts: < 100ms on modern CPU
//! - Batch of 100 texts: < 200-300ms on modern CPU
//! - Memory: ~50MB (model weights + tokenizer vocab)
//! - Binary size: unchanged (no include_bytes)
//!
//! ## Fallback Strategy
//! If model files are missing or ONNX Runtime fails:
//! - `EmbedEngine::load()` returns `Err` → server starts without embed
//! - `/api/mpi/embed` returns 503 Service Unavailable
//! - `/api/mpi/status` reports `embed_ready: false`
//! - Miner falls back to OpenClaw Gateway HTTP API
//! - Plugin falls back to OpenClaw Gateway `/v1/embeddings`
//!
//! ## Dependencies
//! - `ort` 2.0 — ONNX Runtime Rust bindings
//! - `tokenizers` 0.21 — HuggingFace tokenizer (pure Rust, no Python)
//! - `ndarray` 0.16 — Tensor construction for ort input/output
//!
//! ⚠️ Important Note for Next Developer:
//! - Tokenizer MUST match model training — do NOT substitute with generic WordPiece.
//!   Mismatched tokenizer → embedding quality cliff → dedup thresholds all break.
//! - Mean pooling MUST use attention_mask weighting. Padding tokens have non-zero
//!   hidden states; without masking they pollute the pooled vector and silently
//!   degrade similarity scores (0.92/0.88/0.80 thresholds become unreliable).
//! - max_seq_length is configurable (default 128); MiniLM supports up to 512 but
//!   128 is optimal for MemChain's short content. 512 quadruples inference time.
//! - Output vectors are L2-normalized (unit length): cosine_sim = dot product.
//! - EmbedEngine is Send + Sync (ort::Session is thread-safe).
//! - Run `scripts/download_models.sh` before first build/run to fetch model files.
//!
//! ## Last Modified
//! v2.1.0+Embed - 🌟 Initial implementation: disk-based model loading,
//!   ort 2.0, tokenizers 0.21, attention-mask-weighted mean pooling,
//!   L2 normalize, batch support, configurable max_seq_length

use std::path::Path;

use ndarray::Array2;
use ort::{GraphOptimizationLevel, Session};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::{debug, info, warn};

// ============================================
// Constants
// ============================================

/// Expected embedding dimension for MiniLM-L6-v2.
/// This is a model property, NOT user-configurable.
/// Changing this without changing the model will break the vector index.
pub const EMBED_DIM: usize = 384;

/// Default max sequence length. Configurable via `embed_max_tokens` in config.
/// MiniLM supports up to 512, but 128 is optimal for MemChain's short content.
pub const DEFAULT_MAX_SEQ_LENGTH: usize = 128;

/// Model filename within the model directory.
const MODEL_FILENAME: &str = "model.onnx";

/// Tokenizer filename within the model directory.
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Maximum batch size for embed requests.
/// Protects against OOM from excessively large batches.
/// 100 × 128 tokens ≈ 12,800 tokens — well within CPU memory.
pub const MAX_BATCH_SIZE: usize = 100;

// ============================================
// EmbedEngine
// ============================================

/// Local embedding engine using ONNX Runtime + HuggingFace tokenizer.
///
/// Thread-safe: `ort::Session` supports concurrent inference internally.
/// The tokenizer is cloned per-call (cheap — only clones Arc references
/// to vocabulary, not the vocabulary data itself).
///
/// ## Usage
/// ```rust,ignore
/// let engine = EmbedEngine::load("models/minilm-l6-v2", 128)?;
/// let vecs = engine.embed_batch(&["Hello world", "Test"])?;
/// assert_eq!(vecs[0].len(), 384);
/// ```
pub struct EmbedEngine {
    session: Session,
    tokenizer: Tokenizer,
    max_seq_length: usize,
}

impl EmbedEngine {
    /// Load ONNX model and tokenizer from the given directory.
    ///
    /// ## Arguments
    /// * `model_dir` - Directory containing `model.onnx` and `tokenizer.json`.
    ///   Default: `models/minilm-l6-v2` (relative to working directory).
    ///   Download files via `scripts/download_models.sh`.
    /// * `max_seq_length` - Maximum token sequence length (default 128).
    ///   Inputs longer than this are truncated. Shorter inputs are padded.
    ///
    /// ## Returns
    /// * `Ok(EmbedEngine)` - Ready for inference
    /// * `Err(String)` - Files missing, corrupt, or ONNX Runtime unavailable
    ///
    /// ## Performance
    /// Loading takes ~100-500ms (ONNX model parsing + graph optimization).
    /// Call once at startup; reuse the engine for all subsequent requests.
    pub fn load(model_dir: impl AsRef<Path>, max_seq_length: usize) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let max_seq_length = if max_seq_length == 0 { DEFAULT_MAX_SEQ_LENGTH } else { max_seq_length };

        let model_path = model_dir.join(MODEL_FILENAME);
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);

        // Validate files exist with helpful error messages
        if !model_path.exists() {
            return Err(format!(
                "ONNX model not found: {} — run `scripts/download_models.sh` to download",
                model_path.display()
            ));
        }
        if !tokenizer_path.exists() {
            return Err(format!(
                "Tokenizer not found: {} — run `scripts/download_models.sh` to download",
                tokenizer_path.display()
            ));
        }

        // Load ONNX model with optimization level 3 (full graph optimization)
        // intra_threads=2: balance between latency and not starving other tasks
        let session = Session::builder()
            .map_err(|e| format!("ONNX session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ONNX optimization config: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("ONNX thread config: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("ONNX model load ({}): {}", model_path.display(), e))?;

        info!(model = %model_path.display(), "[EMBED] ONNX model loaded");

        // Load HuggingFace tokenizer (pure Rust, no Python dependency)
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Tokenizer load ({}): {}", tokenizer_path.display(), e))?;

        info!(
            tokenizer = %tokenizer_path.display(),
            max_seq_length = max_seq_length,
            "[EMBED] Tokenizer loaded"
        );

        Ok(Self {
            session,
            tokenizer,
            max_seq_length,
        })
    }

    /// Returns the embedding dimension (384 for MiniLM-L6-v2).
    #[must_use]
    pub fn dim(&self) -> usize { EMBED_DIM }

    /// Returns the configured max sequence length.
    #[must_use]
    pub fn max_seq_length(&self) -> usize { self.max_seq_length }

    /// Generate embedding for a single text.
    ///
    /// Convenience wrapper around `embed_batch` for single-text use.
    /// Returns a 384-dim L2-normalized vector.
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>, String> {
        let results = self.embed_batch(&[text])?;
        results.into_iter().next().ok_or_else(|| "Empty result from embed_batch".into())
    }

    /// Generate embeddings for a batch of texts.
    ///
    /// ## Arguments
    /// * `texts` - Slice of text strings to embed (max 100 per batch)
    ///
    /// ## Returns
    /// * `Vec<Vec<f32>>` - One 384-dim L2-normalized vector per input text
    ///
    /// ## Pipeline
    /// 1. Tokenize all texts (pad to longest in batch, truncate to max_seq_length)
    /// 2. Build input_ids + attention_mask + token_type_ids as ndarray::Array2<i64>
    /// 3. Run ONNX session → last_hidden_state [batch, seq_len, 384]
    /// 4. Mean-pool token embeddings (attention-mask-weighted, excluding padding)
    /// 5. L2-normalize each vector to unit length
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() > MAX_BATCH_SIZE {
            return Err(format!("Batch size {} exceeds max {}", texts.len(), MAX_BATCH_SIZE));
        }

        let batch_size = texts.len();

        // ── Step 1: Tokenize ──────────────────────────────────────────
        // Clone tokenizer to set truncation/padding without mutating shared state.
        // This is cheap: tokenizer internals are Arc-wrapped.
        let mut tokenizer = self.tokenizer.clone();

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_seq_length,
                ..Default::default()
            }))
            .map_err(|e| format!("Truncation config: {}", e))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let encodings: Vec<tokenizers::Encoding> = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let seq_len = encodings[0].get_ids().len();

        // ── Step 2: Build tensors ─────────────────────────────────────
        // ONNX model expects: input_ids, attention_mask, token_type_ids
        // All shape [batch_size, seq_len], dtype i64
        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask = Vec::with_capacity(batch_size * seq_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            let ids: &[u32] = enc.get_ids();
            let mask: &[u32] = enc.get_attention_mask();
            let types: &[u32] = enc.get_type_ids();

            for i in 0..seq_len {
                input_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                attention_mask.push(mask.get(i).copied().unwrap_or(0) as i64);
                token_type_ids.push(types.get(i).copied().unwrap_or(0) as i64);
            }
        }

        let ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids)
            .map_err(|e| format!("input_ids shape: {}", e))?;
        let mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask.clone())
            .map_err(|e| format!("attention_mask shape: {}", e))?;
        let types_array = Array2::from_shape_vec((batch_size, seq_len), token_type_ids)
            .map_err(|e| format!("token_type_ids shape: {}", e))?;

        // ── Step 3: ONNX inference ────────────────────────────────────
        let outputs = self.session.run(
            ort::inputs![
                "input_ids" => ids_array,
                "attention_mask" => mask_array,
                "token_type_ids" => types_array,
            ]
            .map_err(|e| format!("Input creation: {}", e))?,
        )
        .map_err(|e| format!("ONNX inference: {}", e))?;

        // Output[0] = last_hidden_state: [batch_size, seq_len, hidden_dim]
        let hidden = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Output extraction: {}", e))?;

        let shape = hidden.shape();
        if shape.len() != 3 {
            return Err(format!("Expected 3D output [batch, seq, dim], got {}D", shape.len()));
        }
        let hidden_dim = shape[2];

        // ── Step 4: Mean pooling with attention mask ──────────────────
        // CRITICAL: padding tokens have non-zero hidden states in transformer models.
        // If we average all tokens including padding, the embedding quality degrades
        // and the dedup thresholds (0.92/0.88/0.80) become unreliable.
        //
        // Correct formula:
        //   pooled[d] = Σ(hidden[t][d] × mask[t]) / Σ(mask[t])
        //
        // This is equivalent to:
        //   hidden_state * mask.unsqueeze(-1) → sum(dim=1) → div(mask.sum(dim=1))
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_dim];
            let mut mask_sum = 0.0f32;

            for s in 0..seq_len {
                let m = attention_mask[b * seq_len + s] as f32;
                if m > 0.0 {
                    mask_sum += m;
                    for d in 0..hidden_dim {
                        pooled[d] += hidden[[b, s, d]] * m;
                    }
                }
            }

            if mask_sum > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= mask_sum;
                }
            }

            // ── Step 5: L2 normalize ──────────────────────────────────
            // After normalization: cosine_similarity(a, b) = dot(a, b)
            // This is why vector.rs uses dot product for scoring.
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for v in pooled.iter_mut() {
                    *v /= norm;
                }
            }

            results.push(pooled);
        }

        debug!(batch = batch_size, seq_len = seq_len, dim = hidden_dim, "[EMBED] Inference done");

        Ok(results)
    }
}

impl std::fmt::Debug for EmbedEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedEngine")
            .field("dim", &EMBED_DIM)
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

    /// Resolve model directory from env or default.
    fn model_dir() -> String {
        std::env::var("MEMCHAIN_EMBED_MODEL_PATH")
            .unwrap_or_else(|_| "models/minilm-l6-v2".to_string())
    }

    /// Helper: skip test if model files are not downloaded.
    fn try_load_engine() -> Option<EmbedEngine> {
        match EmbedEngine::load(&model_dir(), DEFAULT_MAX_SEQ_LENGTH) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("⏭️ Skipping embed test (model not available): {}", e);
                eprintln!("   Run `scripts/download_models.sh` to download model files.");
                None
            }
        }
    }

    #[test]
    fn test_missing_model_returns_error() {
        let result = EmbedEngine::load("/nonexistent/path/to/model", 128);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("not found"), "Error should mention 'not found': {}", err);
        assert!(err.contains("download_models.sh"), "Error should hint at download script: {}", err);
    }

    #[test]
    fn test_empty_batch_returns_empty() {
        // This doesn't need a loaded model — empty input is handled before inference
        // We verify the contract: empty in → empty out
        if let Some(engine) = try_load_engine() {
            let result = engine.embed_batch(&[]).unwrap();
            assert!(result.is_empty());
        }
    }

    #[test]
    fn test_batch_too_large_returns_error() {
        if let Some(engine) = try_load_engine() {
            let texts: Vec<&str> = (0..101).map(|_| "test").collect();
            let result = engine.embed_batch(&texts);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("exceeds max"));
        }
    }

    #[test]
    fn test_single_embed() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        assert_eq!(engine.dim(), 384);

        let vec = engine.embed_single("User is allergic to peanuts").unwrap();
        assert_eq!(vec.len(), 384);

        // Verify L2 normalized (norm ≈ 1.0)
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Expected unit vector, got norm={}",
            norm
        );
    }

    #[test]
    fn test_batch_embed() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        let batch = engine
            .embed_batch(&[
                "User is allergic to peanuts",
                "I prefer dark mode",
                "My name is Alice",
            ])
            .unwrap();

        assert_eq!(batch.len(), 3);
        assert!(batch.iter().all(|v| v.len() == 384));

        // All should be unit vectors
        for (i, v) in batch.iter().enumerate() {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Vector {} norm={}, expected ~1.0",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_deterministic_output() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        let v1 = engine.embed_single("hello world").unwrap();
        let v2 = engine.embed_single("hello world").unwrap();

        let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff < 1e-6, "Same input must produce same output, diff={}", diff);
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        let va = engine.embed_single("I love cats").unwrap();
        let vb = engine.embed_single("quantum mechanics formula").unwrap();

        // Cosine similarity = dot product (since both are L2-normalized)
        let sim: f32 = va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum();
        assert!(
            sim < 0.8,
            "Unrelated texts should have low similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_similar_texts_high_similarity() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        let va = engine.embed_single("I am allergic to peanuts").unwrap();
        let vb = engine.embed_single("I have a peanut allergy").unwrap();

        let sim: f32 = va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum();
        assert!(
            sim > 0.7,
            "Similar texts should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_max_seq_length_respected() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        assert_eq!(engine.max_seq_length(), DEFAULT_MAX_SEQ_LENGTH);

        // Very long input should not panic — just truncated
        let long_text = "word ".repeat(1000);
        let vec = engine.embed_single(&long_text).unwrap();
        assert_eq!(vec.len(), 384);
    }
}
