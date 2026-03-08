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
//! - Auto-detect and load ONNX Runtime shared library via `load-dynamic`
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
//! ├── tokenizer.json           # ~700KB, HuggingFace fast tokenizer
//! └── libonnxruntime.so.1.22.0 # ~30MB, ONNX Runtime shared lib
//!     └── libonnxruntime.so    # symlink → libonnxruntime.so.1.22.0
//! ```
//!
//! ## Architecture Position
//! ```text
//! POST /api/mpi/embed                 Miner Step 0.5 backfill
//!       │                                    │
//!       ▼                                    ▼
//!   EmbedEngine.embed_batch(texts)    EmbedEngine.embed_single(text)
//!       │
//!       ├─ ort::init_from(libonnxruntime.so)  ← one-time, auto-detected
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
//! If model files or libonnxruntime.so are missing:
//! - `EmbedEngine::load()` returns `Err` → server starts without embed
//! - `/api/mpi/embed` returns 503 Service Unavailable
//! - `/api/mpi/status` reports `embed_ready: false`
//! - Miner falls back to OpenClaw Gateway HTTP API
//! - Plugin falls back to OpenClaw Gateway `/v1/embeddings`
//!
//! ## Dependencies
//! - `ort` 2.0.0-rc.11 — ONNX Runtime Rust bindings (load-dynamic mode)
//! - `tokenizers` 0.21 — HuggingFace tokenizer (pure Rust, no Python)
//! - `ndarray` 0.17 — Tensor construction for ort input/output
//!   ⚠️ MUST be 0.17 to match ort rc.11's internal ndarray dependency.
//!
//! ## Modification Reason (v2.1.0+Embed-fix2 — load-dynamic):
//! Switched from `download-binaries` (static linking) to `load-dynamic`
//! (runtime dlopen) because pyke's prebuilt static binaries require
//! glibc ≥ 2.38 (__isoc23_strtol), but Ubuntu 22.04 only has glibc 2.35.
//! Microsoft's official libonnxruntime.so only requires glibc 2.28+.
//!
//! The `load()` method now:
//! 1. Auto-detects libonnxruntime.so from multiple candidate paths
//! 2. Calls `ort::init_from(path).commit()` once (process-global)
//! 3. Then proceeds with Session creation as before
//!
//! Users just run `scripts/download_models.sh` — same workflow as before,
//! the script now also downloads libonnxruntime.so into the model directory.
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
//! - EmbedEngine is Send + Sync (Session wrapped in Mutex for interior mutability).
//! - Run `scripts/download_models.sh` before first build/run to fetch model files
//!   AND libonnxruntime.so.
//! - ort::init_from() MUST be called exactly once before any Session is created.
//!   We use std::sync::Once to guarantee this even if load() is called multiple times.
//! - Session::run() takes &mut self in ort rc.11. The Mutex wrapper handles
//!   this transparently. Do NOT remove the Mutex or change &self to &mut self
//!   on public methods — that would break concurrent access from HTTP handlers.
//!
//! ## Last Modified
//! v2.1.0+Embed - 🌟 Initial implementation
//! v2.1.0+Embed-fix - 🔧 Fixed ort rc.12 API compatibility
//! v2.1.0+Embed-fix2 - 🔧 Switched to load-dynamic for glibc compat;
//!   auto-detect libonnxruntime.so; download via scripts/download_models.sh

use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
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

/// ONNX Runtime shared library filename (Linux).
/// On macOS this would be libonnxruntime.dylib, on Windows onnxruntime.dll.
#[cfg(target_os = "linux")]
const ORT_LIB_FILENAME: &str = "libonnxruntime.so";

#[cfg(target_os = "macos")]
const ORT_LIB_FILENAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "windows")]
const ORT_LIB_FILENAME: &str = "onnxruntime.dll";

/// Maximum batch size for embed requests.
/// Protects against OOM from excessively large batches.
/// 100 × 128 tokens ≈ 12,800 tokens — well within CPU memory.
pub const MAX_BATCH_SIZE: usize = 100;

// ============================================
// ORT Runtime Initialization (once per process)
// ============================================

/// Ensures ort::init_from() is called exactly once per process.
/// Subsequent calls are no-ops.
static ORT_INIT: Once = Once::new();

/// Result of the one-time ORT initialization.
/// Stored so we can report the error if init failed.
static mut ORT_INIT_ERROR: Option<String> = None;

/// Initialize ONNX Runtime by loading libonnxruntime.so from the given path.
///
/// This MUST be called before creating any ort::Session.
/// Uses std::sync::Once to ensure it runs exactly once per process,
/// even if EmbedEngine::load() is called multiple times.
///
/// ## Search Order for libonnxruntime.so
/// 1. `{model_dir}/libonnxruntime.so` (co-located with model, preferred)
/// 2. `ORT_DYLIB_PATH` environment variable (user override)
/// 3. System library paths (`/usr/lib`, `/usr/local/lib`)
///
/// The preferred path is #1 because `scripts/download_models.sh` places
/// the .so file alongside model.onnx, requiring zero user configuration.
fn init_ort_runtime(model_dir: &Path) -> Result<(), String> {
    let mut init_error: Option<String> = None;

    ORT_INIT.call_once(|| {
        // Candidate 1: co-located with model files (scripts/download_models.sh puts it here)
        let colocated = model_dir.join(ORT_LIB_FILENAME);

        // Candidate 2: ORT_DYLIB_PATH environment variable
        let env_path = std::env::var("ORT_DYLIB_PATH").ok().map(PathBuf::from);

        // Candidate 3: common system paths
        let system_paths = [
            PathBuf::from("/usr/lib").join(ORT_LIB_FILENAME),
            PathBuf::from("/usr/local/lib").join(ORT_LIB_FILENAME),
        ];

        // Build candidate list in priority order
        let mut candidates: Vec<PathBuf> = Vec::new();
        candidates.push(colocated);
        if let Some(ep) = env_path {
            candidates.push(ep);
        }
        candidates.extend(system_paths);

        // Find first existing candidate
        let dylib_path = candidates.iter().find(|p| p.exists());

        match dylib_path {
            Some(path) => {
                info!(path = %path.display(), "[EMBED] Found ONNX Runtime library");
                // ort rc.11 API:
                //   init_from(path) -> Result<EnvironmentBuilder, ort::Error>
                //   EnvironmentBuilder::commit() -> bool
                match ort::init_from(path) {
                    Ok(builder) => {
                        builder.commit();
                        info!(path = %path.display(), "[EMBED] ✅ ONNX Runtime initialized (load-dynamic)");
                    }
                    Err(e) => {
                        let msg = format!(
                            "ort::init_from({}) failed: {} — run scripts/download_models.sh",
                            path.display(), e
                        );
                        warn!("{}", msg);
                        // Safety: only written inside Once, never read concurrently
                        unsafe { ORT_INIT_ERROR = Some(msg.clone()); }
                        init_error = Some(msg);
                    }
                }
            }
            None => {
                let searched: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
                let msg = format!(
                    "ONNX Runtime library ({}) not found in: [{}] — run scripts/download_models.sh",
                    ORT_LIB_FILENAME,
                    searched.join(", ")
                );
                warn!("{}", msg);
                // Safety: only written inside Once, never read concurrently
                unsafe { ORT_INIT_ERROR = Some(msg.clone()); }
                init_error = Some(msg);
            }
        }
    });

    // If this is a repeat call and first init failed, return that error
    if init_error.is_some() {
        return Err(init_error.unwrap());
    }

    // Check if a previous call to ORT_INIT set an error
    // Safety: ORT_INIT guarantees the write completed before we read
    let prev_error = unsafe { ORT_INIT_ERROR.as_ref() };
    if let Some(e) = prev_error {
        return Err(e.clone());
    }

    Ok(())
}

// ============================================
// EmbedEngine
// ============================================

/// Local embedding engine using ONNX Runtime + HuggingFace tokenizer.
///
/// Thread-safe: `ort::Session` is wrapped in `Mutex` because `Session::run()`
/// requires `&mut self` in ort 2.0.0-rc.11. The Mutex provides interior
/// mutability while keeping the public API as `&self` for ergonomic use
/// from concurrent HTTP handlers.
///
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
    /// Wrapped in Mutex because ort rc.11 Session::run() requires &mut self.
    /// Lock contention is minimal: inference is 2-100ms, concurrent embed
    /// requests are rare in MemChain's single-agent access pattern.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_seq_length: usize,
}

impl EmbedEngine {
    /// Load ONNX model and tokenizer from the given directory.
    ///
    /// Also auto-initializes ONNX Runtime by finding and loading
    /// `libonnxruntime.so` (see `init_ort_runtime` for search order).
    ///
    /// ## Arguments
    /// * `model_dir` - Directory containing `model.onnx`, `tokenizer.json`,
    ///   and optionally `libonnxruntime.so`.
    ///   Default: `models/minilm-l6-v2` (relative to working directory).
    ///   Download all files via `scripts/download_models.sh`.
    /// * `max_seq_length` - Maximum token sequence length (default 128).
    ///   Inputs longer than this are truncated. Shorter inputs are padded.
    ///
    /// ## Returns
    /// * `Ok(EmbedEngine)` - Ready for inference
    /// * `Err(String)` - Files missing, ORT lib not found, or ONNX Runtime error
    ///
    /// ## Performance
    /// First call: ~100-500ms (ORT init + ONNX model parsing + graph optimization).
    /// Subsequent calls (if creating multiple engines): ~100-500ms (model only).
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

        // Initialize ONNX Runtime (load-dynamic: finds and dlopen's libonnxruntime.so)
        // This is a no-op if already initialized (std::sync::Once).
        init_ort_runtime(model_dir)?;

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
            session: Mutex::new(session),
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
    /// 2. Build input_ids + attention_mask + token_type_ids as Tensor via
    ///    Tensor::from_array with (shape, Box<[i64]>) — zero-copy into ort
    /// 3. Run ONNX session → last_hidden_state [batch, seq_len, 384]
    /// 4. Mean-pool token embeddings (attention-mask-weighted, excluding padding)
    /// 5. L2-normalize each vector to unit length
    ///
    /// ## Thread Safety
    /// Uses Mutex internally — safe to call from multiple threads concurrently.
    /// Concurrent callers will serialize on the Mutex; this is acceptable because
    /// inference dominates wall-clock time and concurrent embed calls are rare.
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
        let total_elements = batch_size * seq_len;
        let mut input_ids = Vec::with_capacity(total_elements);
        let mut attention_mask_raw = Vec::with_capacity(total_elements);
        let mut token_type_ids = Vec::with_capacity(total_elements);

        for enc in &encodings {
            let ids: &[u32] = enc.get_ids();
            let mask: &[u32] = enc.get_attention_mask();
            let types: &[u32] = enc.get_type_ids();

            for i in 0..seq_len {
                input_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                attention_mask_raw.push(mask.get(i).copied().unwrap_or(0) as i64);
                token_type_ids.push(types.get(i).copied().unwrap_or(0) as i64);
            }
        }

        let shape = [batch_size, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids.into_boxed_slice()))
            .map_err(|e| format!("input_ids tensor: {}", e))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask_raw.clone().into_boxed_slice()))
            .map_err(|e| format!("attention_mask tensor: {}", e))?;
        let types_tensor = Tensor::from_array((shape, token_type_ids.into_boxed_slice()))
            .map_err(|e| format!("token_type_ids tensor: {}", e))?;

        // ── Step 3: ONNX inference ────────────────────────────────────
        // Session::run() requires &mut self in ort rc.11, so we acquire
        // the Mutex lock here. Lock held through post-processing because
        // SessionOutputs borrows from Session (lifetime constraint).
        let mut session = self.session.lock()
            .map_err(|e| format!("Session lock poisoned: {}", e))?;
        let outputs = session.run(
            ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => types_tensor,
            ]
        )
        .map_err(|e| format!("ONNX inference: {}", e))?;

        // Output[0] = last_hidden_state: [batch_size, seq_len, hidden_dim]
        // Note: `session` (MutexGuard) is still held here because `outputs`
        // (SessionOutputs) borrows from it. The guard will be dropped at the
        // end of embed_batch(). This is fine — the post-processing below is
        // fast (microseconds) and does not cause meaningful lock contention.
        let hidden = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Output extraction: {}", e))?;

        let hidden_shape = hidden.shape();
        if hidden_shape.len() != 3 {
            return Err(format!("Expected 3D output [batch, seq, dim], got {}D", hidden_shape.len()));
        }
        let hidden_dim = hidden_shape[2];

        // ── Step 4: Mean pooling with attention mask ──────────────────
        // CRITICAL: padding tokens have non-zero hidden states in transformer models.
        // If we average all tokens including padding, the embedding quality degrades
        // and the dedup thresholds (0.92/0.88/0.80) become unreliable.
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_dim];
            let mut mask_sum = 0.0f32;

            for s in 0..seq_len {
                let m = attention_mask_raw[b * seq_len + s] as f32;
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

        let long_text = "word ".repeat(1000);
        let vec = engine.embed_single(&long_text).unwrap();
        assert_eq!(vec.len(), 384);
    }
}
