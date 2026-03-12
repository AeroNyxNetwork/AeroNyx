// ============================================
// File: crates/aeronyx-server/src/services/memchain/embed.rs
// ============================================
//! # EmbedEngine — Local Embedding Inference (MiniLM + EmbeddingGemma)
//!
//! ## Creation Reason
//! Provides local embedding generation for MemChain, eliminating the dependency
//! on OpenClaw Gateway being online. This makes MemChain a self-contained
//! cognitive engine where recall, remember, and Miner operations all work
//! independently of external services.
//!
//! ## Main Functionality
//! - Auto-detect and load ONNX Runtime shared library via `load-dynamic`
//! - Support two embedding models via `EmbedModelType` enum:
//!   - **MiniLM-L6-v2** (legacy default): 384-dim, mean pooling, fast (~3ms)
//!   - **EmbeddingGemma-300M** (v2.5.0+): 768-dim → Matryoshka truncate to 384,
//!     built-in sentence_embedding output, task-specific prompt prefixes,
//!     100+ language support, state-of-the-art quality for its size
//! - Load ONNX model + HuggingFace tokenizer from disk
//! - Tokenize with model-appropriate tokenizer (WordPiece or SentencePiece/BPE)
//! - ONNX inference via `ort` crate (CPU, with optional GPU via dynamic loading)
//! - Model-specific post-processing:
//!   - MiniLM: mean pooling (attention-mask-weighted) + L2 normalization → 384-dim
//!   - EmbeddingGemma: sentence_embedding output → Matryoshka truncation → L2 re-normalize → 384-dim
//! - Batch support for Miner Step 0.5 backfill efficiency
//! - Task-specific prompt prefixes for EmbeddingGemma (query vs document vs similarity)
//!
//! ## Supported Models
//!
//! ### MiniLM-L6-v2 (legacy)
//! ```text
//! {embed_model_path}/          # default: models/minilm-l6-v2
//! ├── model.onnx               # ~22MB, 384-dim output
//! ├── tokenizer.json           # ~700KB, HuggingFace fast tokenizer (WordPiece)
//! └── libonnxruntime.so        # ~30MB, ONNX Runtime shared lib
//! ```
//! - Input: input_ids + attention_mask + token_type_ids
//! - Output: last_hidden_state [batch, seq, 384] → mean pooling → L2 normalize
//! - Max seq length: 128 (default), up to 512
//!
//! ### EmbeddingGemma-300M (v2.5.0+)
//! ```text
//! {embed_model_path}/          # default: models/embeddinggemma
//! ├── model.onnx               # fp32 ~1.2GB, q8 ~300MB
//! ├── model.onnx_data          # external weights (fp32 only)
//! ├── tokenizer.json           # HuggingFace fast tokenizer (SentencePiece/BPE)
//! └── libonnxruntime.so        # ~30MB, ONNX Runtime shared lib (shared)
//! ```
//! - Input: input_ids + attention_mask (NO token_type_ids)
//! - Output: sentence_embedding [batch, 768] (pooling built into ONNX graph)
//! - Matryoshka: truncate 768 → 384 dims, then L2 re-normalize
//! - Max seq length: 256 (default), up to 2048
//! - Task prompts: "task: search result | query: " for queries,
//!   "title: none | text: " for documents
//! - ⚠️ Does NOT support fp16 — use fp32 or q8 quantized
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
//!       ├─ [MiniLM]  ort::Session.run() → last_hidden_state → mean_pool → L2
//!       └─ [Gemma]   ort::Session.run() → sentence_embedding → truncate(384) → L2
//! ```
//!
//! ## Model Auto-Detection
//! `EmbedEngine::load()` auto-detects the model type by checking ONNX output names:
//! - If output named "sentence_embedding" exists → EmbeddingGemma pipeline
//! - Otherwise → MiniLM pipeline (mean pooling)
//! This means users can switch models by changing `embed_model_path` in config.toml
//! and re-running `download_models.sh --embed-gemma`. Zero code changes needed.
//!
//! ## Performance Targets
//! MiniLM:
//! - Single text (~20 tokens): < 5ms on modern CPU
//! - Batch of 50 texts: < 100ms
//! EmbeddingGemma (q8):
//! - Single text (~20 tokens): < 15ms on modern CPU (3× model size)
//! - Batch of 50 texts: < 500ms
//! - Memory: ~350MB (q8 weights + tokenizer vocab)
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
//! ## Modification Reason (v2.5.0 — EmbeddingGemma support):
//! Added EmbeddingGemma-300M as a second supported embedding model.
//! - New enum `EmbedModelType` for model-specific inference pipelines
//! - New enum `EmbedPromptMode` for task-specific prompt prefixes
//! - Auto-detection of model type from ONNX output tensor names
//! - Matryoshka truncation (768 → configurable output_dim, default 384)
//! - Configurable `embed_output_dim` in config.toml for Matryoshka truncation
//! - Updated `download_models.sh` with `--embed-gemma` flag
//! - Interface unchanged: embed_single/embed_batch return Vec<f32> of EMBED_DIM
//! - All callers (mpi.rs, log_handler.rs, reflection.rs, query_analyzer.rs) zero changes
//!
//! ⚠️ Important Note for Next Developer:
//! - Tokenizer MUST match model training — do NOT substitute with generic WordPiece.
//!   Mismatched tokenizer → embedding quality cliff → dedup thresholds all break.
//! - MiniLM mean pooling MUST use attention_mask weighting. Padding tokens have non-zero
//!   hidden states; without masking they pollute the pooled vector and silently
//!   degrade similarity scores (0.92/0.88/0.80 thresholds become unreliable).
//! - EmbeddingGemma's ONNX graph already includes pooling → do NOT add mean_pool.
//!   The "sentence_embedding" output is the final embedding (before Matryoshka truncation).
//! - EmbeddingGemma does NOT use token_type_ids. Passing it will cause ONNX Runtime error.
//! - EmbeddingGemma task prompts significantly affect quality. Using wrong prompt
//!   (e.g., query prefix on a document) will degrade retrieval accuracy.
//! - Matryoshka truncation MUST happen before L2 re-normalization.
//!   Truncate first, then normalize — this is the documented MRL procedure.
//! - Output vectors are L2-normalized (unit length): cosine_sim = dot product.
//! - When switching models, ALL existing embeddings must be rebuilt.
//!   reflection.rs Miner Step 0.5 handles this automatically via backfill.
//!   storage.rs should detect model change and clear embedding columns.
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
//! v2.5.0-EmbeddingGemma - 🌟 Added EmbeddingGemma-300M support:
//!   EmbedModelType auto-detection, EmbedPromptMode task prefixes,
//!   Matryoshka truncation (768→384), embed_with_mode() API,
//!   configurable embed_output_dim. Interface unchanged for all callers.

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

/// Default embedding output dimension.
/// Both MiniLM (native 384) and EmbeddingGemma (768 → truncated to 384)
/// produce this dimension, ensuring downstream compatibility.
/// Configurable via `embed_output_dim` in config.toml.
pub const EMBED_DIM: usize = 384;

/// Default max sequence length for MiniLM.
/// MiniLM supports up to 512, but 128 is optimal for MemChain's short content.
pub const DEFAULT_MAX_SEQ_LENGTH: usize = 128;

/// Default max sequence length for EmbeddingGemma.
/// EmbeddingGemma supports up to 2048. 256 balances quality and speed.
pub const DEFAULT_GEMMA_MAX_SEQ_LENGTH: usize = 256;

/// Model filename within the model directory.
const MODEL_FILENAME: &str = "model.onnx";

/// Tokenizer filename within the model directory.
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// ONNX Runtime shared library filename (Linux).
#[cfg(target_os = "linux")]
const ORT_LIB_FILENAME: &str = "libonnxruntime.so";

#[cfg(target_os = "macos")]
const ORT_LIB_FILENAME: &str = "libonnxruntime.dylib";

#[cfg(target_os = "windows")]
const ORT_LIB_FILENAME: &str = "onnxruntime.dll";

/// Maximum batch size for embed requests.
/// Protects against OOM from excessively large batches.
pub const MAX_BATCH_SIZE: usize = 100;

// ============================================
// EmbeddingGemma Task Prompt Prefixes
// ============================================

/// Query prefix for retrieval/search tasks.
/// Prepended to user queries when searching for relevant memories.
const GEMMA_QUERY_PREFIX: &str = "task: search result | query: ";

/// Document prefix for content being stored/indexed.
/// Prepended to memory content when generating embeddings for storage.
const GEMMA_DOCUMENT_PREFIX: &str = "title: none | text: ";

/// Similarity prefix for semantic similarity comparison.
/// Used for entity merge (Step 9) and other pairwise comparisons.
const GEMMA_SIMILARITY_PREFIX: &str = "task: sentence similarity | query: ";

// ============================================
// Model Type Detection
// ============================================

/// Embedding model type, auto-detected from ONNX output tensor names.
///
/// ## Auto-Detection Logic
/// - If ONNX model has output named "sentence_embedding" → `EmbeddingGemma`
/// - Otherwise → `MiniLM` (legacy default, uses mean pooling)
///
/// ## Adding New Models
/// To add a third model type:
/// 1. Add variant to this enum
/// 2. Add detection logic in `detect_model_type()`
/// 3. Add inference pipeline in `embed_batch_internal()`
/// 4. Update `download_models.sh` with new download flag
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedModelType {
    /// all-MiniLM-L6-v2: 384-dim native output, mean pooling, WordPiece tokenizer.
    /// Fast (~3ms per text), good quality for English, limited multilingual.
    MiniLM,
    /// EmbeddingGemma-300M: 768-dim native → Matryoshka truncation to output_dim.
    /// Built-in sentence_embedding output (no manual pooling needed).
    /// Task-specific prompt prefixes. 100+ languages. State-of-the-art for size.
    EmbeddingGemma,
}

impl std::fmt::Display for EmbedModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedModelType::MiniLM => write!(f, "minilm-l6-v2"),
            EmbedModelType::EmbeddingGemma => write!(f, "embeddinggemma-300m"),
        }
    }
}

/// Prompt mode for EmbeddingGemma task-specific prefixes.
///
/// EmbeddingGemma produces higher-quality embeddings when the input text
/// is prefixed with a task-specific prompt. This enum controls which prefix
/// is applied.
///
/// For MiniLM, this parameter is ignored (no prefix applied).
///
/// ## Usage Mapping
/// | Caller | Mode | Reason |
/// |--------|------|--------|
/// | /api/mpi/recall (query) | Query | Searching for relevant memories |
/// | /api/mpi/log (store) | Document | Indexing conversation content |
/// | Miner Step 0.5 (backfill) | Document | Re-indexing stored records |
/// | Miner Step 7 (entity embed) | Similarity | Entity name for merge comparison |
/// | Miner Step 9 (entity merge) | Similarity | Pairwise entity comparison |
/// | query_analyzer.rs | Query | Analyzing user query intent |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedPromptMode {
    /// For user queries searching for information.
    /// Prefix: "task: search result | query: "
    #[default]
    Query,
    /// For documents/content being indexed for retrieval.
    /// Prefix: "title: none | text: "
    Document,
    /// For semantic similarity comparison (entity merge, dedup).
    /// Prefix: "task: sentence similarity | query: "
    Similarity,
    /// No prefix applied (raw text). Use for backward compatibility
    /// or when the caller handles prefixing itself.
    Raw,
}

/// Detect model type from ONNX session output names.
///
/// Checks if the model has a "sentence_embedding" output tensor,
/// which is characteristic of EmbeddingGemma's ONNX export.
/// MiniLM only has "last_hidden_state" (and optionally "pooler_output").
/// Detect model type and optionally return the index of "sentence_embedding" output.
///
/// Returns (model_type, sentence_embedding_index).
/// For EmbeddingGemma: index is Some(N) where N is the output position.
/// For MiniLM: index is None.
///
/// We resolve the index here (before Session::run()) to avoid borrow conflicts:
/// Session::run() returns SessionOutputs that borrows &mut Session, so we
/// cannot call session.outputs() while SessionOutputs is alive.
fn detect_model_type(session: &Session) -> (EmbedModelType, Option<usize>) {
    let output_names: Vec<String> = session.outputs().iter()
        .map(|o| o.name().to_string())
        .collect();

    debug!(outputs = ?output_names, "[EMBED] ONNX model output tensor names");

    let se_idx = output_names.iter()
        .position(|name| name == "sentence_embedding");

    if se_idx.is_some() {
        info!("[EMBED] Detected EmbeddingGemma model (sentence_embedding output found)");
        (EmbedModelType::EmbeddingGemma, se_idx)
    } else {
        info!("[EMBED] Detected MiniLM model (no sentence_embedding output)");
        (EmbedModelType::MiniLM, None)
    }
}

// ============================================
// ORT Runtime Initialization (once per process)
// ============================================

/// Ensures ort::init_from() is called exactly once per process.
static ORT_INIT: Once = Once::new();

/// Result of the one-time ORT initialization.
static mut ORT_INIT_ERROR: Option<String> = None;

/// Initialize ONNX Runtime by loading libonnxruntime.so from the given path.
///
/// This MUST be called before creating any ort::Session.
/// Uses std::sync::Once to ensure it runs exactly once per process.
///
/// ## Search Order for libonnxruntime.so
/// 1. `{model_dir}/libonnxruntime.so` (co-located with model, preferred)
/// 2. `ORT_DYLIB_PATH` environment variable (user override)
/// 3. System library paths (`/usr/lib`, `/usr/local/lib`)
fn init_ort_runtime(model_dir: &Path) -> Result<(), String> {
    let mut init_error: Option<String> = None;

    ORT_INIT.call_once(|| {
        let colocated = model_dir.join(ORT_LIB_FILENAME);
        let env_path = std::env::var("ORT_DYLIB_PATH").ok().map(PathBuf::from);
        let system_paths = [
            PathBuf::from("/usr/lib").join(ORT_LIB_FILENAME),
            PathBuf::from("/usr/local/lib").join(ORT_LIB_FILENAME),
        ];

        let mut candidates: Vec<PathBuf> = Vec::new();
        candidates.push(colocated);
        if let Some(ep) = env_path {
            candidates.push(ep);
        }
        candidates.extend(system_paths);

        let dylib_path = candidates.iter().find(|p| p.exists());

        match dylib_path {
            Some(path) => {
                info!(path = %path.display(), "[EMBED] Found ONNX Runtime library");
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
                unsafe { ORT_INIT_ERROR = Some(msg.clone()); }
                init_error = Some(msg);
            }
        }
    });

    if init_error.is_some() {
        return Err(init_error.unwrap());
    }

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
/// Supports both MiniLM-L6-v2 and EmbeddingGemma-300M models, auto-detected
/// at load time from ONNX output tensor names. The public API is identical
/// regardless of which model is loaded.
///
/// Thread-safe: `ort::Session` is wrapped in `Mutex` because `Session::run()`
/// requires `&mut self` in ort 2.0.0-rc.11.
///
/// ## Usage
/// ```rust,ignore
/// // MiniLM (auto-detected)
/// let engine = EmbedEngine::load("models/minilm-l6-v2", 128, 384)?;
/// let vecs = engine.embed_batch(&["Hello world", "Test"])?;
///
/// // EmbeddingGemma (auto-detected)
/// let engine = EmbedEngine::load("models/embeddinggemma", 256, 384)?;
/// let vecs = engine.embed_with_mode(&["query text"], EmbedPromptMode::Query)?;
///
/// // Both produce 384-dim vectors
/// assert_eq!(vecs[0].len(), 384);
/// ```
pub struct EmbedEngine {
    /// Wrapped in Mutex because ort rc.11 Session::run() requires &mut self.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_seq_length: usize,
    /// Auto-detected model type (MiniLM or EmbeddingGemma).
    model_type: EmbedModelType,
    /// Index of "sentence_embedding" output in ONNX session outputs.
    /// Only Some for EmbeddingGemma; None for MiniLM.
    /// Pre-resolved at load time to avoid borrow conflicts with Session::run().
    se_output_idx: Option<usize>,
    /// Output embedding dimension after Matryoshka truncation.
    /// For MiniLM: always 384 (native dimension, no truncation).
    /// For EmbeddingGemma: configurable, default 384 (truncated from 768).
    output_dim: usize,
}

impl EmbedEngine {
    /// Load ONNX model and tokenizer from the given directory.
    ///
    /// Auto-detects model type (MiniLM or EmbeddingGemma) from ONNX output names.
    /// Also auto-initializes ONNX Runtime by finding and loading `libonnxruntime.so`.
    ///
    /// ## Arguments
    /// * `model_dir` - Directory containing `model.onnx`, `tokenizer.json`,
    ///   and optionally `libonnxruntime.so`.
    /// * `max_seq_length` - Maximum token sequence length. Pass 0 for model default
    ///   (128 for MiniLM, 256 for EmbeddingGemma).
    /// * `output_dim` - Output embedding dimension. Pass 0 for default (384).
    ///   For EmbeddingGemma, this controls Matryoshka truncation (max 768).
    ///   For MiniLM, this must be ≤ 384 (native dimension).
    ///
    /// ## Returns
    /// * `Ok(EmbedEngine)` - Ready for inference
    /// * `Err(String)` - Files missing, ORT lib not found, or ONNX Runtime error
    pub fn load(
        model_dir: impl AsRef<Path>,
        max_seq_length: usize,
        output_dim: usize,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();

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
        init_ort_runtime(model_dir)?;

        // Load ONNX model with optimization level 3 (full graph optimization)
        let session = Session::builder()
            .map_err(|e| format!("ONNX session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ONNX optimization config: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("ONNX thread config: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("ONNX model load ({}): {}", model_path.display(), e))?;

        info!(model = %model_path.display(), "[EMBED] ONNX model loaded");

        // Auto-detect model type from ONNX output tensor names
        let (model_type, se_output_idx) = detect_model_type(&session);

        // Resolve max_seq_length: 0 → model-specific default
        let max_seq_length = if max_seq_length == 0 {
            match model_type {
                EmbedModelType::MiniLM => DEFAULT_MAX_SEQ_LENGTH,
                EmbedModelType::EmbeddingGemma => DEFAULT_GEMMA_MAX_SEQ_LENGTH,
            }
        } else {
            max_seq_length
        };

        // Resolve output_dim: 0 → EMBED_DIM (384)
        let output_dim = if output_dim == 0 { EMBED_DIM } else { output_dim };

        // Validate output_dim for MiniLM (native 384, cannot upscale)
        if model_type == EmbedModelType::MiniLM && output_dim > 384 {
            return Err(format!(
                "MiniLM native dimension is 384, cannot output {} dims. \
                 Use EmbeddingGemma for larger dimensions.",
                output_dim
            ));
        }

        // Load HuggingFace tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Tokenizer load ({}): {}", tokenizer_path.display(), e))?;

        info!(
            tokenizer = %tokenizer_path.display(),
            model_type = %model_type,
            max_seq_length = max_seq_length,
            output_dim = output_dim,
            "[EMBED] Tokenizer loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_seq_length,
            model_type,
            se_output_idx,
            output_dim,
        })
    }

    /// Returns the output embedding dimension (default 384).
    #[must_use]
    pub fn dim(&self) -> usize { self.output_dim }

    /// Returns the configured max sequence length.
    #[must_use]
    pub fn max_seq_length(&self) -> usize { self.max_seq_length }

    /// Returns the detected model type.
    #[must_use]
    pub fn model_type(&self) -> EmbedModelType { self.model_type }

    /// Returns the model name string for storage metadata.
    /// Used by vector index and records table to track which model produced embeddings.
    #[must_use]
    pub fn model_name(&self) -> &'static str {
        match self.model_type {
            EmbedModelType::MiniLM => "minilm-l6-v2",
            EmbedModelType::EmbeddingGemma => "embeddinggemma-300m",
        }
    }

    /// Generate embedding for a single text with default prompt mode (Query).
    ///
    /// This is the backward-compatible API used by all existing callers.
    /// For EmbeddingGemma, applies Query prefix by default.
    /// For MiniLM, prompt mode is ignored (no prefix).
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_single_with_mode(text, EmbedPromptMode::Query)
    }

    /// Generate embedding for a single text with explicit prompt mode.
    ///
    /// ## Arguments
    /// * `text` - Text to embed
    /// * `mode` - Prompt mode (only affects EmbeddingGemma; ignored for MiniLM)
    pub fn embed_single_with_mode(
        &self,
        text: &str,
        mode: EmbedPromptMode,
    ) -> Result<Vec<f32>, String> {
        let results = self.embed_with_mode(&[text], mode)?;
        results.into_iter().next().ok_or_else(|| "Empty result from embed_batch".into())
    }

    /// Generate embeddings for a batch of texts with default prompt mode (Query).
    ///
    /// Backward-compatible API. All existing callers use this.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.embed_with_mode(texts, EmbedPromptMode::Query)
    }

    /// Generate embeddings for a batch of texts with explicit prompt mode.
    ///
    /// ## Arguments
    /// * `texts` - Slice of text strings to embed (max 100 per batch)
    /// * `mode` - Prompt mode for EmbeddingGemma task prefixes
    ///
    /// ## Returns
    /// * `Vec<Vec<f32>>` - One output_dim-dimensional L2-normalized vector per input text
    pub fn embed_with_mode(
        &self,
        texts: &[&str],
        mode: EmbedPromptMode,
    ) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() > MAX_BATCH_SIZE {
            return Err(format!("Batch size {} exceeds max {}", texts.len(), MAX_BATCH_SIZE));
        }

        match self.model_type {
            EmbedModelType::MiniLM => self.embed_minilm(texts),
            EmbedModelType::EmbeddingGemma => self.embed_gemma(texts, mode),
        }
    }

    // ============================================
    // MiniLM Pipeline (legacy, unchanged logic)
    // ============================================

    /// MiniLM inference pipeline:
    /// tokenize → input_ids + attention_mask + token_type_ids
    /// → ONNX run → last_hidden_state → mean pooling → L2 normalize
    fn embed_minilm(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = texts.len();

        // ── Tokenize ──
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

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let seq_len = encodings[0].get_ids().len();
        let total = batch_size * seq_len;

        let mut input_ids = Vec::with_capacity(total);
        let mut attention_mask_raw = Vec::with_capacity(total);
        let mut token_type_ids = Vec::with_capacity(total);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
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

        // ── ONNX inference ──
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
        let hidden = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("Output extraction: {}", e))?;

        let hidden_shape = hidden.shape();
        if hidden_shape.len() != 3 {
            return Err(format!("Expected 3D output [batch, seq, dim], got {}D", hidden_shape.len()));
        }
        let hidden_dim = hidden_shape[2];

        // ── Mean pooling + L2 normalize ──
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

            // Matryoshka truncation for MiniLM (if output_dim < 384)
            pooled.truncate(self.output_dim);

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for v in pooled.iter_mut() {
                    *v /= norm;
                }
            }

            results.push(pooled);
        }

        debug!(batch = batch_size, seq_len = seq_len, dim = self.output_dim, model = "minilm", "[EMBED] Inference done");
        Ok(results)
    }

    // ============================================
    // EmbeddingGemma Pipeline (v2.5.0)
    // ============================================

    /// EmbeddingGemma inference pipeline:
    /// apply task prefix → tokenize → input_ids + attention_mask
    /// → ONNX run → sentence_embedding [batch, 768]
    /// → Matryoshka truncate to output_dim → L2 re-normalize
    fn embed_gemma(
        &self,
        texts: &[&str],
        mode: EmbedPromptMode,
    ) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = texts.len();

        // ── Apply task-specific prompt prefix ──
        let prefixed_texts: Vec<String> = texts.iter().map(|text| {
            match mode {
                EmbedPromptMode::Query => format!("{}{}", GEMMA_QUERY_PREFIX, text),
                EmbedPromptMode::Document => format!("{}{}", GEMMA_DOCUMENT_PREFIX, text),
                EmbedPromptMode::Similarity => format!("{}{}", GEMMA_SIMILARITY_PREFIX, text),
                EmbedPromptMode::Raw => text.to_string(),
            }
        }).collect();

        let text_refs: Vec<&str> = prefixed_texts.iter().map(|s| s.as_str()).collect();

        // ── Tokenize ──
        // EmbeddingGemma uses SentencePiece/BPE tokenizer (loaded from tokenizer.json)
        // The tokenizers crate handles both WordPiece and BPE transparently.
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

        let encodings = tokenizer
            .encode_batch(text_refs, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?;

        let seq_len = encodings[0].get_ids().len();
        let total = batch_size * seq_len;

        // EmbeddingGemma: input_ids + attention_mask only (NO token_type_ids)
        let mut input_ids = Vec::with_capacity(total);
        let mut attention_mask = Vec::with_capacity(total);

        for enc in &encodings {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for i in 0..seq_len {
                input_ids.push(ids.get(i).copied().unwrap_or(0) as i64);
                attention_mask.push(mask.get(i).copied().unwrap_or(0) as i64);
            }
        }

        let shape = [batch_size, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids.into_boxed_slice()))
            .map_err(|e| format!("input_ids tensor: {}", e))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask.into_boxed_slice()))
            .map_err(|e| format!("attention_mask tensor: {}", e))?;

        // ── ONNX inference ──
        // EmbeddingGemma only takes input_ids + attention_mask
        let mut session = self.session.lock()
            .map_err(|e| format!("Session lock poisoned: {}", e))?;
        let outputs = session.run(
            ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ]
        )
        .map_err(|e| format!("ONNX inference: {}", e))?;

        // Use pre-resolved output index (computed at load time to avoid borrow conflict).
        let se_idx = self.se_output_idx
            .ok_or_else(|| "EmbeddingGemma ONNX missing 'sentence_embedding' output index".to_string())?;

        let embeddings = outputs[se_idx]
            .try_extract_array::<f32>()
            .map_err(|e| format!("sentence_embedding extraction: {}", e))?;

        let emb_shape = embeddings.shape();
        if emb_shape.len() != 2 {
            return Err(format!(
                "Expected 2D sentence_embedding [batch, dim], got {}D {:?}",
                emb_shape.len(), emb_shape
            ));
        }
        let native_dim = emb_shape[1]; // 768 for EmbeddingGemma

        // ── Matryoshka truncation + L2 re-normalize ──
        // MRL guarantees that the first N dimensions contain the most information.
        // Procedure: truncate to output_dim, then L2 re-normalize to unit length.
        let truncate_dim = self.output_dim.min(native_dim);

        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Extract and truncate
            let mut vec: Vec<f32> = (0..truncate_dim)
                .map(|d| embeddings[[b, d]])
                .collect();

            // L2 re-normalize after truncation
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for v in vec.iter_mut() {
                    *v /= norm;
                }
            }

            results.push(vec);
        }

        debug!(
            batch = batch_size, seq_len = seq_len,
            native_dim = native_dim, output_dim = truncate_dim,
            mode = ?mode, model = "embeddinggemma",
            "[EMBED] Inference done"
        );

        Ok(results)
    }
}

impl std::fmt::Debug for EmbedEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedEngine")
            .field("model_type", &self.model_type)
            .field("output_dim", &self.output_dim)
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
        match EmbedEngine::load(&model_dir(), 0, 0) {
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
        let result = EmbedEngine::load("/nonexistent/path/to/model", 128, 0);
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

        let expected_dim = engine.dim();
        let vec = engine.embed_single("User is allergic to peanuts").unwrap();
        assert_eq!(vec.len(), expected_dim);

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

        let expected_dim = engine.dim();
        let batch = engine
            .embed_batch(&[
                "User is allergic to peanuts",
                "I prefer dark mode",
                "My name is Alice",
            ])
            .unwrap();

        assert_eq!(batch.len(), 3);
        assert!(batch.iter().all(|v| v.len() == expected_dim));

        for (i, v) in batch.iter().enumerate() {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "Vector {} norm={}, expected ~1.0",
                i, norm
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

        let long_text = "word ".repeat(1000);
        let vec = engine.embed_single(&long_text).unwrap();
        assert_eq!(vec.len(), engine.dim());
    }

    #[test]
    fn test_model_type_detected() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        // Model type should be auto-detected
        let model_type = engine.model_type();
        assert!(
            model_type == EmbedModelType::MiniLM || model_type == EmbedModelType::EmbeddingGemma,
            "Model type should be MiniLM or EmbeddingGemma, got {:?}",
            model_type
        );
        info!("Detected model type: {}", model_type);
    }

    #[test]
    fn test_prompt_modes_produce_different_embeddings() {
        let engine = match try_load_engine() {
            Some(e) => e,
            None => return,
        };

        // Only EmbeddingGemma should produce different embeddings for different modes
        if engine.model_type() != EmbedModelType::EmbeddingGemma {
            eprintln!("⏭️ Skipping prompt mode test (model is not EmbeddingGemma)");
            return;
        }

        let text = "What is the capital of France?";
        let v_query = engine.embed_single_with_mode(text, EmbedPromptMode::Query).unwrap();
        let v_doc = engine.embed_single_with_mode(text, EmbedPromptMode::Document).unwrap();

        let sim: f32 = v_query.iter().zip(v_doc.iter()).map(|(a, b)| a * b).sum();
        // Same text with different prefixes should produce somewhat different vectors
        assert!(
            sim < 0.99,
            "Different prompt modes should produce different vectors, sim={}",
            sim
        );
    }
}
