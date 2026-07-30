// ============================================
// File: crates/aeronyx-server/src/services/memchain/embed.rs
// ============================================
//! # EmbedEngine — Local Embedding Inference (MiniLM + EmbeddingGemma)
//!
//! ## Creation Reason
//! Provides local embedding generation for MemChain, eliminating the dependency
//! on an external gateway being online. This makes MemChain a self-contained
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
//! - Miner uses the local embedding engine when the model is available
//! - Plugin callers should use the local MemChain API for embeddings
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
//! - EmbedEngine is Send + Sync (Session wrapped in the shared
//!   `InferenceSession` boundary for interior mutability).
//! - Run `scripts/download_models.sh` before first build/run to fetch model files
//!   AND libonnxruntime.so.
//! - ort::init_from() MUST succeed before any Session is created. The shared
//!   initialization gate serializes concurrent attempts, caches the first
//!   success, and leaves failures retryable so another enabled engine can use
//!   its own valid co-located runtime library.
//! - Session::run() takes &mut self in ort rc.11. The `InferenceSession`
//!   wrapper handles this transparently. Do NOT remove the wrapper or change
//!   &self to &mut self on public methods — that would break concurrent access
//!   from HTTP handlers.
//!
//! ## Last Modified
//! v2.7.20-TokenizerBatchBoundary -
//!   [MEMCHAIN-TOKENIZER-BATCH 2026-07-30 by Codex] Centralized tokenizer
//!   batch-shape validation and stopped silently padding malformed encoding
//!   fields with zeroes before ONNX inference.
//! v2.7.19-OnnxOutputBoundary -
//!   [MEMCHAIN-ONNX-OUTPUT-BOUNDARY 2026-07-30 by Codex] Centralized
//!   bounds-checked ONNX output retrieval so missing or incompatible model
//!   outputs return an inference error instead of panicking the request task.
//! v2.7.18-EmbeddingOutputValidation -
//!   [MEMCHAIN-EMBED-OUTPUT 2026-07-30 by Codex] Added strict MiniLM and
//!   EmbeddingGemma output-shape contracts, rejected non-finite tensors and
//!   degenerate vectors, and moved L2 norm accumulation to f64.
//! v2.7.17-RetryableOrtInitialization -
//!   [MEMCHAIN-ORT-RECOVERY 2026-07-30 by Codex] Replaced sticky failed ORT
//!   initialization with a serialized success-only gate.
//! v2.7.16-InferenceSessionRecovery -
//!   [MEMCHAIN-INFERENCE-SESSION 2026-07-30 by Codex] Centralized ONNX session
//!   serialization behind a non-poisoning lock so a recovered panic cannot
//!   permanently disable local cognition.
//! v2.7.15-EmbedInitSafety - Replaced unsynchronised `static mut` ORT error
//!   state with `OnceLock<Result<(), String>>` and added concurrent regression
//!   coverage. Public APIs and first-initializer semantics are unchanged.
//! v2.1.0+Embed - 🌟 Initial implementation
//! v2.1.0+Embed-fix - 🔧 Fixed ort rc.12 API compatibility
//! v2.1.0+Embed-fix2 - 🔧 Switched to load-dynamic for glibc compat;
//!   auto-detect libonnxruntime.so; download via scripts/download_models.sh
//! v2.5.0-EmbeddingGemma - 🌟 Added EmbeddingGemma-300M support:
//!   EmbedModelType auto-detection, EmbedPromptMode task prefixes,
//!   Matryoshka truncation (768→384), embed_with_mode() API,
//!   configurable embed_output_dim. Interface unchanged for all callers.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::{Mutex, MutexGuard};
use std::path::{Path, PathBuf};
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::{debug, info, warn};

// ============================================
// Constants
// ============================================

/// Default embedding output dimension.
/// Both MiniLM (native 384) and EmbeddingGemma (768 → truncated to 384)
/// produce this dimension, ensuring downstream compatibility.
/// Configurable via `embed_output_dim` in config.toml.
pub const EMBED_DIM: usize = 384;

// ============================================
// Shared inference session boundary
// ============================================

/// Serializes mutable ONNX Runtime session access without lock poisoning.
///
/// [MEMCHAIN-INFERENCE-SESSION 2026-07-30 by Codex] `ort::Session::run`
/// requires mutable access. A standard-library mutex permanently poisons after
/// any panic while held, which can leave the process healthy while embedding,
/// NER, or reranking stays unavailable. This wrapper gives all local inference
/// engines one recovery contract and keeps the guard synchronous.
pub(super) struct InferenceSession<T> {
    inner: Mutex<T>,
}

impl<T> InferenceSession<T> {
    pub(super) fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
        }
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, T> {
        self.inner.lock()
    }
}

/// Retrieve one ONNX output without using `SessionOutputs`' panicking index API.
///
/// [MEMCHAIN-ONNX-OUTPUT-BOUNDARY 2026-07-30 by Codex] ort rc.11 exposes
/// bounds-checked lookup only for named outputs; numeric indexing panics when
/// an export returns fewer values than the engine expects. Keeping this helper
/// iterator-based works for both fixed output zero and load-time-resolved model
/// output indices while preserving the underlying value's borrow lifetime.
pub(super) fn require_onnx_output<T>(
    engine: &str,
    mut outputs: impl ExactSizeIterator<Item = T>,
    index: usize,
) -> Result<T, String> {
    let output_count = outputs.len();
    outputs.nth(index).ok_or_else(|| {
        format!(
            "{} ONNX returned {} outputs; required output index {}",
            engine, output_count, index
        )
    })
}

/// Validate the structural contract of a padded tokenizer batch.
///
/// [MEMCHAIN-TOKENIZER-BATCH 2026-07-30 by Codex] HuggingFace tokenization is
/// a model-file boundary, not trusted application state. Every encoding must
/// match the requested batch and padded sequence dimensions before tensor
/// construction. This prevents first-element indexing panics, tensor shape
/// mismatches, and silent zero-filling of malformed masks or segment IDs.
pub(super) fn validate_tokenized_batch(
    engine: &str,
    encodings: &[Encoding],
    expected_batch: usize,
    max_sequence_length: usize,
    require_type_ids: bool,
) -> Result<(usize, usize), String> {
    if encodings.len() != expected_batch {
        return Err(format!(
            "{} tokenizer returned {} encodings for input batch {}",
            engine,
            encodings.len(),
            expected_batch
        ));
    }

    let sequence_length = encodings
        .first()
        .map(Encoding::len)
        .ok_or_else(|| format!("{} tokenizer returned an empty batch", engine))?;
    if sequence_length == 0 {
        return Err(format!(
            "{} tokenizer returned an empty token sequence",
            engine
        ));
    }
    if sequence_length > max_sequence_length {
        return Err(format!(
            "{} tokenizer sequence length {} exceeds configured maximum {}",
            engine, sequence_length, max_sequence_length
        ));
    }

    for (batch_index, encoding) in encodings.iter().enumerate() {
        let ids_length = encoding.get_ids().len();
        let mask_length = encoding.get_attention_mask().len();
        let type_ids_length = encoding.get_type_ids().len();

        if ids_length != sequence_length {
            return Err(format!(
                "{} tokenizer encoding {} has {} token IDs; expected padded length {}",
                engine, batch_index, ids_length, sequence_length
            ));
        }
        if mask_length != sequence_length {
            return Err(format!(
                "{} tokenizer encoding {} has attention-mask length {}; expected {}",
                engine, batch_index, mask_length, sequence_length
            ));
        }
        if require_type_ids && type_ids_length != sequence_length {
            return Err(format!(
                "{} tokenizer encoding {} has type-ID length {}; expected {}",
                engine, batch_index, type_ids_length, sequence_length
            ));
        }
    }

    let total_tokens = expected_batch.checked_mul(sequence_length).ok_or_else(|| {
        format!(
            "{} tokenizer tensor size overflow: {} x {}",
            engine, expected_batch, sequence_length
        )
    })?;

    Ok((sequence_length, total_tokens))
}

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
// Embedding output contract
// ============================================

/// Validate MiniLM's `[batch, sequence, native_dim]` output contract.
///
/// [MEMCHAIN-EMBED-OUTPUT 2026-07-30 by Codex] Exact batch and sequence
/// dimensions are required because mean pooling indexes both axes using the
/// tokenizer output. The native dimension must satisfy the configured public
/// output dimension so callers never receive a shorter vector than promised.
fn validate_minilm_output_shape(
    shape: &[usize],
    expected_batch: usize,
    expected_sequence: usize,
    output_dim: usize,
) -> Result<usize, String> {
    let [batch, sequence, native_dim] = shape else {
        return Err(format!(
            "MiniLM output shape {:?} is not [batch, sequence, native_dim]",
            shape
        ));
    };

    if *batch != expected_batch || *sequence != expected_sequence {
        return Err(format!(
            "MiniLM output shape {:?} does not match tokenized batch {} and sequence {}",
            shape, expected_batch, expected_sequence
        ));
    }
    if *native_dim < output_dim {
        return Err(format!(
            "MiniLM native dimension {} is smaller than configured output dimension {}",
            native_dim, output_dim
        ));
    }

    Ok(*native_dim)
}

/// Validate EmbeddingGemma's `[batch, native_dim]` output contract.
fn validate_gemma_output_shape(
    shape: &[usize],
    expected_batch: usize,
    output_dim: usize,
) -> Result<usize, String> {
    let [batch, native_dim] = shape else {
        return Err(format!(
            "EmbeddingGemma output shape {:?} is not [batch, native_dim]",
            shape
        ));
    };

    if *batch != expected_batch {
        return Err(format!(
            "EmbeddingGemma output batch {} does not match input batch {}",
            batch, expected_batch
        ));
    }
    if *native_dim < output_dim {
        return Err(format!(
            "EmbeddingGemma native dimension {} is smaller than configured output dimension {}",
            native_dim, output_dim
        ));
    }

    Ok(*native_dim)
}

/// Reject non-finite model output before pooling or vector-index insertion.
fn validate_finite_embedding_output(
    model: &str,
    values: impl Iterator<Item = f32>,
) -> Result<(), String> {
    if let Some((index, value)) = values.enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(format!(
            "{} output contains non-finite value at flat index {}: {}",
            model, index, value
        ));
    }

    Ok(())
}

/// L2-normalize one embedding using f64 accumulation.
///
/// Squaring large finite f32 values in f32 can overflow to infinity and turn a
/// valid direction into an all-zero vector after division. Accumulating in f64
/// covers the complete finite f32 range and makes zero-norm output explicit.
fn normalize_embedding(
    model: &str,
    batch_index: usize,
    embedding: &mut [f32],
) -> Result<(), String> {
    validate_finite_embedding_output(model, embedding.iter().copied())?;

    let norm = embedding
        .iter()
        .copied()
        .map(f64::from)
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    if !norm.is_finite() || norm <= 1e-12 {
        return Err(format!(
            "{} produced a degenerate embedding at batch index {} (L2 norm: {})",
            model, batch_index, norm
        ));
    }

    for value in embedding {
        *value = (f64::from(*value) / norm) as f32;
    }

    Ok(())
}

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
    let output_names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();

    debug!(outputs = ?output_names, "[EMBED] ONNX model output tensor names");

    let se_idx = output_names
        .iter()
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

/// Serializes ORT initialization and remembers only a successful commit.
///
/// [MEMCHAIN-ORT-RECOVERY 2026-07-30 by Codex] Model directories are
/// independently configurable. A missing runtime beside the first enabled
/// model must not permanently prevent a later engine from loading a valid
/// co-located runtime. Holding the non-poisoning mutex across initialization
/// also prevents concurrent attempts from racing the process-global ORT state.
static ORT_INITIALIZED: Mutex<bool> = Mutex::new(false);

fn initialize_runtime<F>(initialized: &Mutex<bool>, initialize: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String>,
{
    let mut initialized = initialized.lock();
    if *initialized {
        return Ok(());
    }

    initialize()?;
    *initialized = true;
    Ok(())
}

/// Initialize ONNX Runtime by loading libonnxruntime.so from the given path.
///
/// This MUST be called before creating any ort::Session.
/// Concurrent attempts are serialized. The first successful initialization is
/// retained for the process lifetime; failures remain retryable by another
/// enabled engine with a different model directory.
///
/// ## Search Order for libonnxruntime.so
/// 1. `{model_dir}/libonnxruntime.so` (co-located with model, preferred)
/// 2. `ORT_DYLIB_PATH` environment variable (user override)
/// 3. System library paths (`/usr/lib`, `/usr/local/lib`)
pub(crate) fn init_ort_runtime(model_dir: &Path) -> Result<(), String> {
    initialize_runtime(&ORT_INITIALIZED, || {
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
                        Ok(())
                    }
                    Err(e) => {
                        let msg = format!(
                            "ort::init_from({}) failed: {} — run scripts/download_models.sh",
                            path.display(),
                            e
                        );
                        warn!("{}", msg);
                        Err(msg)
                    }
                }
            }
            None => {
                let searched: Vec<String> =
                    candidates.iter().map(|p| p.display().to_string()).collect();
                let msg = format!(
                    "ONNX Runtime library ({}) not found in: [{}] — run scripts/download_models.sh",
                    ORT_LIB_FILENAME,
                    searched.join(", ")
                );
                warn!("{}", msg);
                Err(msg)
            }
        }
    })
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
    /// Shared serialization boundary for mutable ONNX inference.
    session: InferenceSession<Session>,
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
        let output_dim = if output_dim == 0 {
            EMBED_DIM
        } else {
            output_dim
        };

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
            session: InferenceSession::new(session),
            tokenizer,
            max_seq_length,
            model_type,
            se_output_idx,
            output_dim,
        })
    }

    /// Returns the output embedding dimension (default 384).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.output_dim
    }

    /// Returns the configured max sequence length.
    #[must_use]
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Returns the detected model type.
    #[must_use]
    pub fn model_type(&self) -> EmbedModelType {
        self.model_type
    }

    /// Returns the model name string for storage metadata.
    /// Used by vector index and records table to track which model produced embeddings.
    #[must_use]
    pub fn model_name(&self) -> &'static str {
        match self.model_type {
            EmbedModelType::MiniLM => "minilm-l6-v2",
            EmbedModelType::EmbeddingGemma => "embeddinggemma-300m",
        }
    }

    /// Generate embedding for a single text with default prompt mode (Document).
    ///
    /// This is the backward-compatible API used by all existing callers.
    /// Default is Document mode because most callers store/index content:
    /// - /remember endpoint stores memory records
    /// - Miner Step 0.5 backfills record embeddings
    /// - Miner Step 7 generates entity embeddings
    /// - log_handler stores conversation embeddings
    ///
    /// For **query** embedding (recall, search), use embed_single_with_mode()
    /// with EmbedPromptMode::Query explicitly.
    ///
    /// For MiniLM, prompt mode is ignored (no prefix applied).
    pub fn embed_single(&self, text: &str) -> Result<Vec<f32>, String> {
        self.embed_single_with_mode(text, EmbedPromptMode::Document)
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
        results
            .into_iter()
            .next()
            .ok_or_else(|| "Empty result from embed_batch".into())
    }

    /// Generate embeddings for a batch of texts with default prompt mode (Document).
    ///
    /// Backward-compatible API. Most callers index content (not search queries).
    /// For query embedding, use embed_with_mode() with EmbedPromptMode::Query.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.embed_with_mode(texts, EmbedPromptMode::Document)
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
            return Err(format!(
                "Batch size {} exceeds max {}",
                texts.len(),
                MAX_BATCH_SIZE
            ));
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

        let (seq_len, total) =
            validate_tokenized_batch("MiniLM", &encodings, batch_size, self.max_seq_length, true)?;

        let mut input_ids = Vec::with_capacity(total);
        let mut attention_mask_raw = Vec::with_capacity(total);
        let mut token_type_ids = Vec::with_capacity(total);

        for enc in &encodings {
            input_ids.extend(enc.get_ids().iter().copied().map(i64::from));
            attention_mask_raw.extend(enc.get_attention_mask().iter().copied().map(i64::from));
            token_type_ids.extend(enc.get_type_ids().iter().copied().map(i64::from));
        }

        let shape = [batch_size, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids.into_boxed_slice()))
            .map_err(|e| format!("input_ids tensor: {}", e))?;
        let mask_tensor =
            Tensor::from_array((shape, attention_mask_raw.clone().into_boxed_slice()))
                .map_err(|e| format!("attention_mask tensor: {}", e))?;
        let types_tensor = Tensor::from_array((shape, token_type_ids.into_boxed_slice()))
            .map_err(|e| format!("token_type_ids tensor: {}", e))?;

        // ── ONNX inference ──
        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "token_type_ids" => types_tensor,
            ])
            .map_err(|e| format!("ONNX inference: {}", e))?;

        // Output[0] = last_hidden_state: [batch_size, seq_len, hidden_dim]
        let hidden_output = require_onnx_output("MiniLM", outputs.values(), 0)?;
        let hidden = hidden_output
            .try_extract_array::<f32>()
            .map_err(|e| format!("Output extraction: {}", e))?;

        // [MEMCHAIN-EMBED-OUTPUT 2026-07-30 by Codex] Validate all axes and
        // values before indexed pooling. A mismatched model must fail the
        // request instead of panicking or inserting a corrupted vector.
        let hidden_dim =
            validate_minilm_output_shape(hidden.shape(), batch_size, seq_len, self.output_dim)?;
        validate_finite_embedding_output("MiniLM", hidden.iter().copied())?;

        // ── Mean pooling + L2 normalize ──
        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let mut pooled = vec![0.0f64; hidden_dim];
            let mut mask_sum = 0.0f64;

            for s in 0..seq_len {
                let m = attention_mask_raw[b * seq_len + s] as f64;
                if m > 0.0 {
                    mask_sum += m;
                    for d in 0..hidden_dim {
                        pooled[d] += f64::from(hidden[[b, s, d]]) * m;
                    }
                }
            }

            if mask_sum <= 0.0 {
                return Err(format!(
                    "MiniLM attention mask is empty at batch index {}",
                    b
                ));
            }

            // Matryoshka truncation for MiniLM (if output_dim < 384)
            pooled.truncate(self.output_dim);
            let mut pooled: Vec<f32> = pooled
                .into_iter()
                .map(|value| (value / mask_sum) as f32)
                .collect();

            // L2 normalize
            normalize_embedding("MiniLM", b, &mut pooled)?;

            results.push(pooled);
        }

        debug!(
            batch = batch_size,
            seq_len = seq_len,
            dim = self.output_dim,
            model = "minilm",
            "[EMBED] Inference done"
        );
        Ok(results)
    }

    // ============================================
    // EmbeddingGemma Pipeline (v2.5.0)
    // ============================================

    /// EmbeddingGemma inference pipeline:
    /// apply task prefix → tokenize → input_ids + attention_mask
    /// → ONNX run → sentence_embedding [batch, 768]
    /// → Matryoshka truncate to output_dim → L2 re-normalize
    fn embed_gemma(&self, texts: &[&str], mode: EmbedPromptMode) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = texts.len();

        // ── Apply task-specific prompt prefix ──
        let prefixed_texts: Vec<String> = texts
            .iter()
            .map(|text| match mode {
                EmbedPromptMode::Query => format!("{}{}", GEMMA_QUERY_PREFIX, text),
                EmbedPromptMode::Document => format!("{}{}", GEMMA_DOCUMENT_PREFIX, text),
                EmbedPromptMode::Similarity => format!("{}{}", GEMMA_SIMILARITY_PREFIX, text),
                EmbedPromptMode::Raw => text.to_string(),
            })
            .collect();

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

        let (seq_len, total) = validate_tokenized_batch(
            "EmbeddingGemma",
            &encodings,
            batch_size,
            self.max_seq_length,
            false,
        )?;

        // EmbeddingGemma: input_ids + attention_mask only (NO token_type_ids)
        let mut input_ids = Vec::with_capacity(total);
        let mut attention_mask = Vec::with_capacity(total);

        for enc in &encodings {
            input_ids.extend(enc.get_ids().iter().copied().map(i64::from));
            attention_mask.extend(enc.get_attention_mask().iter().copied().map(i64::from));
        }

        let shape = [batch_size, seq_len];

        let ids_tensor = Tensor::from_array((shape, input_ids.into_boxed_slice()))
            .map_err(|e| format!("input_ids tensor: {}", e))?;
        let mask_tensor = Tensor::from_array((shape, attention_mask.into_boxed_slice()))
            .map_err(|e| format!("attention_mask tensor: {}", e))?;

        // ── ONNX inference ──
        // EmbeddingGemma only takes input_ids + attention_mask
        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            ])
            .map_err(|e| format!("ONNX inference: {}", e))?;

        // Use pre-resolved output index (computed at load time to avoid borrow conflict).
        let se_idx = self.se_output_idx.ok_or_else(|| {
            "EmbeddingGemma ONNX missing 'sentence_embedding' output index".to_string()
        })?;

        let embeddings_output = require_onnx_output("EmbeddingGemma", outputs.values(), se_idx)?;
        let embeddings = embeddings_output
            .try_extract_array::<f32>()
            .map_err(|e| format!("sentence_embedding extraction: {}", e))?;

        // [MEMCHAIN-EMBED-OUTPUT 2026-07-30 by Codex] Preserve the public
        // output-dimension contract and reject invalid values before indexing.
        let native_dim =
            validate_gemma_output_shape(embeddings.shape(), batch_size, self.output_dim)?;
        validate_finite_embedding_output("EmbeddingGemma", embeddings.iter().copied())?;

        // ── Matryoshka truncation + L2 re-normalize ──
        // MRL guarantees that the first N dimensions contain the most information.
        // Procedure: truncate to output_dim, then L2 re-normalize to unit length.
        let truncate_dim = self.output_dim;

        let mut results = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            // Extract and truncate
            let mut vec: Vec<f32> = (0..truncate_dim).map(|d| embeddings[[b, d]]).collect();

            // L2 re-normalize after truncation
            normalize_embedding("EmbeddingGemma", b, &mut vec)?;

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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    /// Resolve model directory from env or default.
    fn model_dir() -> String {
        std::env::var("MEMCHAIN_EMBED_MODEL_PATH")
            .unwrap_or_else(|_| "models/minilm-l6-v2".to_string())
    }

    #[test]
    fn runtime_initialization_serializes_callers_and_caches_success() {
        const THREADS: usize = 16;
        let initialized = Arc::new(Mutex::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(THREADS));
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let initialized = Arc::clone(&initialized);
                let calls = Arc::clone(&calls);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    initialize_runtime(&initialized, || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        Ok(())
                    })
                })
            })
            .collect();

        for worker in workers {
            worker.join().unwrap().unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        initialize_runtime(&initialized, || Err("must not run".to_string())).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn runtime_initialization_retries_after_failure() {
        // [MEMCHAIN-ORT-RECOVERY 2026-07-30 by Codex] A broken runtime beside
        // the first model cannot poison later independently configured engines.
        let initialized = Mutex::new(false);
        let calls = AtomicUsize::new(0);

        let first_error = initialize_runtime(&initialized, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Err("first runtime unavailable".to_string())
        })
        .unwrap_err();
        assert_eq!(first_error, "first runtime unavailable");

        initialize_runtime(&initialized, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        initialize_runtime(&initialized, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Err("must not run after success".to_string())
        })
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn inference_session_recovers_after_lock_owner_panic() {
        // [MEMCHAIN-INFERENCE-SESSION 2026-07-30 by Codex] This models a
        // recovered request/task panic without requiring ONNX model files.
        let session = Arc::new(InferenceSession::new(0_u8));
        let panic_session = Arc::clone(&session);
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let mut guard = panic_session.lock();
            *guard = 1;
            panic!("test-only-inference-session-panic");
        }));
        assert!(panic_result.is_err());

        *session.lock() += 1;
        assert_eq!(*session.lock(), 2);
    }

    #[test]
    fn onnx_output_boundary_returns_requested_value() {
        let output = require_onnx_output("test-engine", [11_u8, 22_u8].into_iter(), 1).unwrap();

        assert_eq!(output, 22);
    }

    #[test]
    fn onnx_output_boundary_rejects_missing_value() {
        let empty_error =
            require_onnx_output("test-engine", std::iter::empty::<u8>(), 0).unwrap_err();
        assert!(empty_error.contains("test-engine"));
        assert!(empty_error.contains("returned 0 outputs"));
        assert!(empty_error.contains("index 0"));

        let short_error = require_onnx_output("test-engine", [11_u8].into_iter(), 1).unwrap_err();
        assert!(short_error.contains("returned 1 outputs"));
        assert!(short_error.contains("index 1"));
    }

    fn test_encoding(ids: Vec<u32>, type_ids: Vec<u32>, attention_mask: Vec<u32>) -> Encoding {
        let length = ids.len();
        Encoding::new(
            ids,
            type_ids,
            vec!["token".to_string(); length],
            vec![None; length],
            vec![(0, 0); length],
            vec![0; length],
            attention_mask,
            Vec::new(),
            Default::default(),
        )
    }

    #[test]
    fn tokenizer_batch_boundary_accepts_consistent_padding() {
        let encodings = [
            test_encoding(vec![1, 2], vec![0, 0], vec![1, 1]),
            test_encoding(vec![3, 0], vec![0, 0], vec![1, 0]),
        ];

        assert_eq!(
            validate_tokenized_batch("test-engine", &encodings, 2, 8, true).unwrap(),
            (2, 4)
        );
    }

    #[test]
    fn tokenizer_batch_boundary_rejects_batch_and_sequence_mismatch() {
        let encoding = test_encoding(vec![1, 2], vec![0, 0], vec![1, 1]);
        let batch_error =
            validate_tokenized_batch("test-engine", &[encoding.clone()], 2, 8, true).unwrap_err();
        assert!(batch_error.contains("returned 1 encodings"));

        let uneven = test_encoding(vec![3], vec![0], vec![1]);
        let sequence_error =
            validate_tokenized_batch("test-engine", &[encoding, uneven], 2, 8, true).unwrap_err();
        assert!(sequence_error.contains("padded length 2"));
    }

    #[test]
    fn tokenizer_batch_boundary_rejects_empty_or_oversized_sequence() {
        let empty_error =
            validate_tokenized_batch("test-engine", &[Encoding::default()], 1, 8, false)
                .unwrap_err();
        assert!(empty_error.contains("empty token sequence"));

        let encoding = test_encoding(vec![1, 2], Vec::new(), vec![1, 1]);
        let oversized_error =
            validate_tokenized_batch("test-engine", &[encoding], 1, 1, false).unwrap_err();
        assert!(oversized_error.contains("exceeds configured maximum 1"));
    }

    #[test]
    fn tokenizer_batch_boundary_validates_required_tensor_fields() {
        let bad_mask = test_encoding(vec![1, 2], vec![0, 0], vec![1]);
        let mask_error =
            validate_tokenized_batch("test-engine", &[bad_mask], 1, 8, true).unwrap_err();
        assert!(mask_error.contains("attention-mask length 1"));

        let missing_types = test_encoding(vec![1, 2], Vec::new(), vec![1, 1]);
        let type_error =
            validate_tokenized_batch("test-engine", &[missing_types.clone()], 1, 8, true)
                .unwrap_err();
        assert!(type_error.contains("type-ID length 0"));

        assert_eq!(
            validate_tokenized_batch("test-engine", &[missing_types], 1, 8, false).unwrap(),
            (2, 2)
        );
    }

    #[test]
    fn embedding_output_shapes_require_exact_batch_and_sequence() {
        assert_eq!(
            validate_minilm_output_shape(&[2, 8, 384], 2, 8, 384).unwrap(),
            384
        );
        assert!(validate_minilm_output_shape(&[1, 8, 384], 2, 8, 384).is_err());
        assert!(validate_minilm_output_shape(&[2, 7, 384], 2, 8, 384).is_err());
        assert!(validate_minilm_output_shape(&[2, 8], 2, 8, 384).is_err());

        assert_eq!(validate_gemma_output_shape(&[2, 768], 2, 384).unwrap(), 768);
        assert!(validate_gemma_output_shape(&[1, 768], 2, 384).is_err());
        assert!(validate_gemma_output_shape(&[2, 8, 768], 2, 384).is_err());
    }

    #[test]
    fn embedding_output_shapes_reject_short_native_dimensions() {
        let minilm = validate_minilm_output_shape(&[1, 8, 383], 1, 8, 384).unwrap_err();
        assert!(minilm.contains("smaller"));

        let gemma = validate_gemma_output_shape(&[1, 383], 1, 384).unwrap_err();
        assert!(gemma.contains("smaller"));
    }

    #[test]
    fn embedding_output_rejects_non_finite_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = validate_finite_embedding_output("test-model", [0.0, value].into_iter())
                .unwrap_err();
            assert!(error.contains("non-finite"));
        }
    }

    #[test]
    fn embedding_normalization_handles_f32_extremes() {
        let mut embedding = [f32::MAX, f32::MAX];
        normalize_embedding("test-model", 0, &mut embedding).unwrap();

        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((embedding[0] - expected).abs() < 1e-6);
        assert!((embedding[1] - expected).abs() < 1e-6);
        assert!(embedding.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn embedding_normalization_rejects_degenerate_vectors() {
        let mut embedding = [0.0, 0.0];
        let error = normalize_embedding("test-model", 3, &mut embedding).unwrap_err();

        assert!(error.contains("degenerate"));
        assert!(error.contains("batch index 3"));
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
        assert!(
            diff < 1e-6,
            "Same input must produce same output, diff={}",
            diff
        );
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
        let v_query = engine
            .embed_single_with_mode(text, EmbedPromptMode::Query)
            .unwrap();
        let v_doc = engine
            .embed_single_with_mode(text, EmbedPromptMode::Document)
            .unwrap();

        let sim: f32 = v_query.iter().zip(v_doc.iter()).map(|(a, b)| a * b).sum();
        // Same text with different prefixes should produce somewhat different vectors
        assert!(
            sim < 0.99,
            "Different prompt modes should produce different vectors, sim={}",
            sim
        );
    }
}
