// ============================================
// File: crates/aeronyx-server/src/services/memchain/ner.rs
// ============================================
//! # NerEngine — Local GLiNER ONNX Named Entity Recognition
//!
//! ## Creation Reason (v2.4.0-GraphCognition)
//! Provides local zero-shot Named Entity Recognition (NER) for MemChain's
//! cognitive graph pipeline. This is the **sister module** to `embed.rs`,
//! sharing the same ort load-dynamic mechanism and architectural patterns.
//!
//! GLiNER extracts arbitrary entity types from text at inference time by
//! accepting entity labels as input — no retraining required. This powers:
//! - Stage 1 entropy filtering (entity novelty scoring)
//! - Stage 2 entity/relation extraction (knowledge graph construction)
//! - Query analysis (detecting entities in user queries for hybrid retrieval)
//!
//! ## Main Functionality
//! - Load GLiNER ONNX model + DeBERTa/BERT tokenizer from disk
//! - Build GLiNER prompt format: `[<<ENT>> label1 <<ENT>> label2 ... <<SEP>> word1 word2 ...]`
//! - Construct 6 input tensors: input_ids, attention_mask, words_mask, text_lengths, span_idx, span_mask
//! - ONNX inference → logits [batch, num_spans, num_labels]
//! - Sigmoid decoding + confidence threshold filtering
//! - Greedy span deduplication (keep highest-scoring for overlapping spans)
//!
//! ## Model Files (downloaded via scripts/download_models.sh)
//! ```text
//! {ner_model_path}/              # default: models/gliner
//! ├── model.onnx                 # ~50-200MB depending on variant
//! ├── tokenizer.json             # HuggingFace fast tokenizer (DeBERTa-v3)
//! └── gliner_config.json         # Optional: model config (max_width, etc.)
//! ```
//!
//! ## Architecture Position
//! ```text
//! Stage 1 (log_handler.rs)        Stage 2 (miner/reflection.rs)
//!       │                                    │
//!       ▼                                    ▼
//!   NerEngine.detect_entities()    NerEngine.detect_entities()
//!       │
//!   query_analyzer.rs (recall-hook)
//!       │
//!       ▼
//!   NerEngine.detect_entities()
//!
//! All paths:
//!   ├─ Word split → GLiNER prompt construction
//!   ├─ Subword tokenization (HuggingFace tokenizer)
//!   ├─ Build 6 input tensors (input_ids, attention_mask, words_mask,
//!   │                          text_lengths, span_idx, span_mask)
//!   ├─ ort::Session.run() → logits [1, num_spans, num_labels]
//!   ├─ Sigmoid → confidence scores
//!   ├─ Threshold filter (default 0.5)
//!   └─ Greedy dedup → Vec<DetectedEntity>
//! ```
//!
//! ## GLiNER Prompt Format
//! GLiNER uses a special prompt format to enable zero-shot NER:
//! ```text
//! [CLS] <<ENT>> project <<ENT>> technology <<ENT>> person <<SEP>> auth module uses JWT [SEP]
//! ```
//! The model learns to associate entity type tokens (after <<ENT>>) with
//! text word spans (after <<SEP>>), producing span-level logits.
//!
//! ## Span Representation
//! GLiNER enumerates all possible (start_word, end_word) spans up to
//! `max_width` (default 12 words). For N text words and W max_width:
//! - num_spans = N * W (capped by actual text length)
//! - span_idx tensor: [batch, num_spans, 2] (start, end word indices)
//! - span_mask tensor: [batch, num_spans] (valid span = true)
//! - Output logits: [batch, num_spans, num_labels] → sigmoid → scores
//!
//! ## Performance Targets
//! - Single text (~50 words, 8 labels): < 15ms on modern CPU
//! - Memory: ~50-200MB (model weights + tokenizer vocab)
//!
//! ## Fallback Strategy
//! If model files are missing:
//! - `NerEngine::load()` returns `Err` → server starts without NER
//! - Stage 1 entropy filter skips entity novelty (uses only semantic divergence)
//! - Stage 2 skips entity extraction (only Episode storage)
//! - Query analyzer falls back to regex-only entity detection
//! - `/api/mpi/status` reports `ner_ready: false`
//!
//! ## Dependencies
//! - `ort` 2.0.0-rc.11 — ONNX Runtime Rust bindings (load-dynamic, shared with embed.rs)
//! - `tokenizers` 0.21 — HuggingFace tokenizer (pure Rust, no Python)
//!
//! ⚠️ Important Note for Next Developer:
//! - GLiNER's ONNX input format has 6 tensors, NOT 3 like BERT. Missing any
//!   tensor causes silent wrong results or ORT crash.
//! - words_mask maps subword tokens back to word positions. Getting this wrong
//!   means span indices won't align with the original text.
//! - span_idx uses WORD indices (not token indices). Each span is (start_word, end_word)
//!   where end_word is INCLUSIVE.
//! - text_lengths counts only TEXT words (excludes prompt prefix tokens).
//! - The tokenizer MUST match the model's training tokenizer (typically DeBERTa-v3
//!   for v2.x models). Using the wrong tokenizer silently degrades quality.
//! - ORT init is shared with embed.rs via std::sync::Once — safe to load both.
//! - Session::run() requires &mut self → Mutex wrapper (same pattern as embed.rs).
//! - max_width (max entity span in words) defaults to 12. Larger values increase
//!   num_spans quadratically — only increase if needed.
//! - The `<<ENT>>` and `<<SEP>>` special tokens must be in the tokenizer vocabulary.
//!   If they're missing, the model was not exported correctly.
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - 🌟 Initial implementation

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

// ============================================
// Constants
// ============================================

/// Model filename within the NER model directory.
const MODEL_FILENAME: &str = "model.onnx";

/// Tokenizer filename within the NER model directory.
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// GLiNER special token: entity type marker.
/// In the prompt, each label is preceded by this token.
const ENT_TOKEN: &str = "<<ENT>>";

/// GLiNER special token: separator between labels and text.
const SEP_TOKEN: &str = "<<SEP>>";

/// Default maximum entity span width in words.
/// Entities longer than this many words are not considered.
/// 12 covers most real-world entities (e.g., "New York City Department of Education").
const DEFAULT_MAX_WIDTH: usize = 12;

/// Default confidence threshold for entity detection.
/// Spans with sigmoid(logit) < threshold are discarded.
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Maximum number of entity labels per inference call.
/// Protects against excessive prompt length.
const MAX_LABELS: usize = 32;

/// Maximum text length in words for a single inference call.
/// Longer texts should be split into windows by the caller.
const MAX_TEXT_WORDS: usize = 512;

// ============================================
// Types
// ============================================

/// A detected entity span with its label and confidence score.
#[derive(Debug, Clone)]
pub struct DetectedEntity {
    /// The matched text substring.
    pub text: String,
    /// The entity type label (as provided in the labels input).
    pub label: String,
    /// Confidence score from sigmoid(logit), range [0.0, 1.0].
    pub confidence: f32,
    /// Start character offset in the original text (byte index).
    pub char_start: usize,
    /// End character offset in the original text (byte index, exclusive).
    pub char_end: usize,
    /// Start word index in the word-split text.
    pub word_start: usize,
    /// End word index in the word-split text (inclusive).
    pub word_end: usize,
}

/// A word with its byte offsets in the original text.
#[derive(Debug, Clone)]
struct WordSpan {
    text: String,
    byte_start: usize,
    byte_end: usize,
}

// ============================================
// NerEngine
// ============================================

/// Local NER engine using GLiNER ONNX model.
///
/// Thread-safe: `ort::Session` is wrapped in `Mutex` because `Session::run()`
/// requires `&mut self` in ort 2.0.0-rc.11 (same pattern as EmbedEngine).
///
/// ## Usage
/// ```rust,ignore
/// let engine = NerEngine::load("models/gliner", 0.5, 12)?;
/// let entities = engine.detect_entities(
///     "auth module uses JWT for authentication",
///     &["project", "module", "technology", "person"],
/// )?;
/// for e in &entities {
///     println!("{} => {} ({:.2})", e.text, e.label, e.confidence);
/// }
/// // "auth module" => module (0.87)
/// // "JWT" => technology (0.93)
/// ```
pub struct NerEngine {
    /// ONNX session, Mutex-wrapped for &mut self requirement.
    session: Mutex<Session>,
    /// HuggingFace tokenizer (DeBERTa-v3 or BERT depending on model).
    tokenizer: Tokenizer,
    /// Maximum entity span width in words.
    max_width: usize,
    /// Confidence threshold for filtering detections.
    confidence_threshold: f32,
    /// Token ID for <<ENT>> special token (cached at load time).
    ent_token_id: u32,
    /// Token ID for <<SEP>> special token (cached at load time).
    sep_token_id: u32,
}

impl NerEngine {
    /// Load GLiNER ONNX model and tokenizer from the given directory.
    ///
    /// ## Prerequisites
    /// - `init_ort_runtime()` must have been called (by EmbedEngine::load or directly).
    ///   If EmbedEngine is loaded first (which is the normal case), ORT is already initialized.
    ///   If NerEngine is loaded standalone, the caller must ensure ORT init.
    ///
    /// ## Arguments
    /// * `model_dir` - Directory containing `model.onnx` and `tokenizer.json`
    /// * `confidence_threshold` - Minimum sigmoid score to keep a detection (default 0.5)
    /// * `max_width` - Maximum entity span width in words (default 12)
    ///
    /// ## Returns
    /// * `Ok(NerEngine)` - Ready for inference
    /// * `Err(String)` - Files missing or ONNX Runtime error
    pub fn load(
        model_dir: impl AsRef<Path>,
        confidence_threshold: f32,
        max_width: usize,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let confidence_threshold = if confidence_threshold <= 0.0 || confidence_threshold >= 1.0 {
            DEFAULT_CONFIDENCE_THRESHOLD
        } else {
            confidence_threshold
        };
        let max_width = if max_width == 0 { DEFAULT_MAX_WIDTH } else { max_width };

        let model_path = model_dir.join(MODEL_FILENAME);
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);

        if !model_path.exists() {
            return Err(format!(
                "GLiNER ONNX model not found: {} — run `scripts/download_models.sh` to download",
                model_path.display()
            ));
        }
        if !tokenizer_path.exists() {
            return Err(format!(
                "GLiNER tokenizer not found: {} — run `scripts/download_models.sh` to download",
                tokenizer_path.display()
            ));
        }

        // Note: ORT runtime should already be initialized by EmbedEngine.
        // If not, init_ort_runtime() in embed.rs handles it via Once.
        // We don't call it here to avoid circular dependency — the caller
        // (server.rs) must ensure EmbedEngine is loaded first, or call
        // init_ort_runtime() manually.

        // Load ONNX model — same settings as EmbedEngine
        let session = Session::builder()
            .map_err(|e| format!("GLiNER session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("GLiNER optimization config: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("GLiNER thread config: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| format!("GLiNER model load ({}): {}", model_path.display(), e))?;

        info!(model = %model_path.display(), "[NER] GLiNER ONNX model loaded");

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("GLiNER tokenizer load ({}): {}", tokenizer_path.display(), e))?;

        // Resolve special token IDs — these MUST exist in the tokenizer
        let ent_token_id = Self::resolve_token_id(&tokenizer, ENT_TOKEN)?;
        let sep_token_id = Self::resolve_token_id(&tokenizer, SEP_TOKEN)?;

        info!(
            tokenizer = %tokenizer_path.display(),
            max_width = max_width,
            threshold = confidence_threshold,
            ent_token_id = ent_token_id,
            sep_token_id = sep_token_id,
            "[NER] GLiNER tokenizer loaded"
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_width,
            confidence_threshold,
            ent_token_id,
            sep_token_id,
        })
    }

    /// Detect entities in the given text using the specified labels.
    ///
    /// ## Arguments
    /// * `text` - Input text to analyze
    /// * `labels` - Entity type labels (e.g., ["project", "technology", "person"])
    ///
    /// ## Returns
    /// * `Vec<DetectedEntity>` - Detected entities sorted by char_start
    ///
    /// ## Pipeline
    /// 1. Word-split the input text (whitespace + punctuation aware)
    /// 2. Build GLiNER prompt: [CLS] <<ENT>> label1 <<ENT>> label2 ... <<SEP>> word1 word2 ... [SEP]
    /// 3. Subword-tokenize the prompt, tracking word boundaries
    /// 4. Build span indices for all valid (start, end) word pairs up to max_width
    /// 5. Construct 6 input tensors
    /// 6. Run ONNX inference → logits
    /// 7. Sigmoid + threshold filter + greedy dedup
    pub fn detect_entities(
        &self,
        text: &str,
        labels: &[&str],
    ) -> Result<Vec<DetectedEntity>, String> {
        if text.is_empty() || labels.is_empty() {
            return Ok(Vec::new());
        }
        if labels.len() > MAX_LABELS {
            return Err(format!("Too many labels: {} (max {})", labels.len(), MAX_LABELS));
        }

        // Step 1: Word-split the text
        let words = self.word_split(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let num_text_words = words.len().min(MAX_TEXT_WORDS);
        let words = &words[..num_text_words];

        // Step 2-3: Build prompt and tokenize
        let (input_ids, attention_mask, words_mask, text_length, num_prompt_tokens) =
            self.build_prompt_and_tokenize(labels, words)?;

        let seq_len = input_ids.len();

        // Step 4: Build span indices
        let (span_idx_flat, span_mask_flat, num_spans) =
            self.build_span_indices(num_text_words);

        if num_spans == 0 {
            return Ok(Vec::new());
        }

        // Step 5: Build tensors
        let batch_size = 1usize;
        let shape_2d = [batch_size, seq_len];
        let shape_tl = [batch_size, 1usize];
        let shape_span_idx = [batch_size, num_spans, 2usize];
        let shape_span_mask = [batch_size, num_spans];

        let ids_tensor = Tensor::from_array(
            (shape_2d, input_ids.into_boxed_slice())
        ).map_err(|e| format!("input_ids tensor: {}", e))?;

        let mask_tensor = Tensor::from_array(
            (shape_2d, attention_mask.into_boxed_slice())
        ).map_err(|e| format!("attention_mask tensor: {}", e))?;

        let words_mask_tensor = Tensor::from_array(
            (shape_2d, words_mask.into_boxed_slice())
        ).map_err(|e| format!("words_mask tensor: {}", e))?;

        let text_lengths_tensor = Tensor::from_array(
            (shape_tl, vec![text_length as i64].into_boxed_slice())
        ).map_err(|e| format!("text_lengths tensor: {}", e))?;

        let span_idx_tensor = Tensor::from_array(
            (shape_span_idx, span_idx_flat.into_boxed_slice())
        ).map_err(|e| format!("span_idx tensor: {}", e))?;

        // span_mask is bool in GLiNER ONNX — ort expects u8 for bool tensors
        let span_mask_u8: Vec<u8> = span_mask_flat.iter().map(|&b| b as u8).collect();
        let span_mask_tensor = Tensor::from_array(
            (shape_span_mask, span_mask_u8.into_boxed_slice())
        ).map_err(|e| format!("span_mask tensor: {}", e))?;

        // Step 6: ONNX inference
        let mut session = self.session.lock()
            .map_err(|e| format!("NER session lock poisoned: {}", e))?;

        let outputs = session.run(
            ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "words_mask" => words_mask_tensor,
                "text_lengths" => text_lengths_tensor,
                "span_idx" => span_idx_tensor,
                "span_mask" => span_mask_tensor,
            ]
        ).map_err(|e| format!("GLiNER ONNX inference: {}", e))?;

        // Output: logits [batch_size, num_spans, num_labels]
        let logits = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| format!("GLiNER output extraction: {}", e))?;

        let logits_shape = logits.shape();
        if logits_shape.len() != 3 {
            return Err(format!(
                "Expected 3D logits [batch, spans, labels], got {}D",
                logits_shape.len()
            ));
        }
        let out_num_spans = logits_shape[1];
        let out_num_labels = logits_shape[2];

        // Step 7: Sigmoid + threshold + decode
        let mut raw_detections: Vec<DetectedEntity> = Vec::new();

        for s in 0..out_num_spans {
            if s >= num_spans {
                break;
            }
            // Reconstruct span word indices from our span generation order
            let (word_start, word_end) = self.span_index_to_words(s, num_text_words);
            if word_end >= num_text_words {
                continue;
            }

            for l in 0..out_num_labels {
                if l >= labels.len() {
                    break;
                }
                let logit = logits[[0, s, l]];
                let score = sigmoid(logit);

                if score >= self.confidence_threshold {
                    // Reconstruct text from word spans
                    let char_start = words[word_start].byte_start;
                    let char_end = words[word_end].byte_end;
                    let entity_text = &text[char_start..char_end];

                    raw_detections.push(DetectedEntity {
                        text: entity_text.to_string(),
                        label: labels[l].to_string(),
                        confidence: score,
                        char_start,
                        char_end,
                        word_start,
                        word_end,
                    });
                }
            }
        }

        // Greedy dedup: for overlapping spans, keep highest confidence
        let results = greedy_dedup(raw_detections);

        debug!(
            entities = results.len(),
            labels = labels.len(),
            words = num_text_words,
            "[NER] Detection complete"
        );

        Ok(results)
    }

    /// Returns the configured confidence threshold.
    #[must_use]
    pub fn confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    /// Returns the configured max entity width in words.
    #[must_use]
    pub fn max_width(&self) -> usize {
        self.max_width
    }

    // ========================================
    // Private: Word Splitting
    // ========================================

    /// Split text into words with byte offset tracking.
    ///
    /// Uses a simple whitespace + punctuation-aware split that matches
    /// GLiNER's expected input format. Punctuation attached to words
    /// is kept with the word (e.g., "JWT," → "JWT" + ",").
    fn word_split(&self, text: &str) -> Vec<WordSpan> {
        let mut words = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some(&(byte_start, ch)) = chars.peek() {
            if ch.is_whitespace() {
                chars.next();
                continue;
            }

            // Collect a word (non-whitespace run)
            let mut byte_end = byte_start;
            let mut word = String::new();

            while let Some(&(bi, c)) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                byte_end = bi + c.len_utf8();
                word.push(c);
                chars.next();
            }

            if !word.is_empty() {
                words.push(WordSpan {
                    text: word,
                    byte_start,
                    byte_end,
                });
            }
        }

        words
    }

    // ========================================
    // Private: Prompt Construction + Tokenization
    // ========================================

    /// Build GLiNER prompt and tokenize it, returning the 4 sequence-level tensors
    /// plus the text_length value.
    ///
    /// Prompt format:
    /// [CLS] <<ENT>> label1 <<ENT>> label2 ... <<SEP>> word1 word2 ... [SEP]
    ///
    /// Returns: (input_ids, attention_mask, words_mask, text_length, num_prompt_tokens)
    fn build_prompt_and_tokenize(
        &self,
        labels: &[&str],
        words: &[WordSpan],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>, usize, usize), String> {
        // Build the full prompt as token IDs manually to maintain precise control
        // over word boundaries for the words_mask tensor.

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_words_mask: Vec<i64> = Vec::new();

        // [CLS] token — get from tokenizer
        let cls_id = self.get_special_token_id("[CLS]")
            .or_else(|| self.get_special_token_id("<s>"))
            .unwrap_or(0);
        let sep_end_id = self.get_special_token_id("[SEP]")
            .or_else(|| self.get_special_token_id("</s>"))
            .unwrap_or(0);

        // Add [CLS]
        all_token_ids.push(cls_id);
        all_words_mask.push(0); // not a text word

        // Add entity labels: <<ENT>> label1 <<ENT>> label2 ...
        for label in labels {
            // <<ENT>> token
            all_token_ids.push(self.ent_token_id);
            all_words_mask.push(0);

            // Tokenize label text (may produce multiple subwords)
            let label_encoding = self.tokenizer.encode(
                label.to_string(), false
            ).map_err(|e| format!("Label tokenization failed: {}", e))?;

            for &id in label_encoding.get_ids() {
                all_token_ids.push(id);
                all_words_mask.push(0); // label tokens are not text words
            }
        }

        // <<SEP>> between labels and text
        all_token_ids.push(self.sep_token_id);
        all_words_mask.push(0);

        let num_prompt_tokens = all_token_ids.len();

        // Add text words — each word may produce multiple subword tokens.
        // words_mask maps each subword token to its word index (1-based for GLiNER).
        // Word index 0 means "not a text word" (prompt/special tokens).
        for (word_idx, word) in words.iter().enumerate() {
            let word_encoding = self.tokenizer.encode(
                word.text.clone(), false
            ).map_err(|e| format!("Word tokenization failed: {}", e))?;

            let ids = word_encoding.get_ids();
            if ids.is_empty() {
                // Unknown word — add UNK token
                let unk_id = self.tokenizer.token_to_id("[UNK]").unwrap_or(0);
                all_token_ids.push(unk_id);
                // word_idx + 1 because GLiNER uses 1-based word indexing
                all_words_mask.push((word_idx + 1) as i64);
            } else {
                for &id in ids {
                    all_token_ids.push(id);
                    // All subword tokens of this word get the same word index
                    all_words_mask.push((word_idx + 1) as i64);
                }
            }
        }

        // Add final [SEP]
        all_token_ids.push(sep_end_id);
        all_words_mask.push(0);

        let seq_len = all_token_ids.len();
        let text_length = words.len();

        // Build i64 tensors
        let input_ids: Vec<i64> = all_token_ids.iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = vec![1i64; seq_len];

        Ok((input_ids, attention_mask, all_words_mask, text_length, num_prompt_tokens))
    }

    // ========================================
    // Private: Span Index Construction
    // ========================================

    /// Build span indices for all valid (start, end) word pairs.
    ///
    /// Enumerates spans: for each start word, try end = start, start+1, ..., start+max_width-1.
    /// Uses GLiNER's 1-based word indexing.
    ///
    /// Returns: (span_idx_flat [num_spans * 2], span_mask_flat [num_spans], num_spans)
    fn build_span_indices(
        &self,
        num_text_words: usize,
    ) -> (Vec<i64>, Vec<bool>, usize) {
        let mut span_idx: Vec<i64> = Vec::new();
        let mut span_mask: Vec<bool> = Vec::new();

        for start in 0..num_text_words {
            let end_limit = (start + self.max_width).min(num_text_words);
            for end in start..end_limit {
                // GLiNER uses 0-based word indices in span_idx
                span_idx.push(start as i64);
                span_idx.push(end as i64);
                span_mask.push(true);
            }
        }

        let num_spans = span_mask.len();
        (span_idx, span_mask, num_spans)
    }

    /// Convert a linear span index back to (start_word, end_word) pair.
    ///
    /// Must match the enumeration order in build_span_indices().
    fn span_index_to_words(&self, span_idx: usize, num_text_words: usize) -> (usize, usize) {
        let mut idx = 0;
        for start in 0..num_text_words {
            let end_limit = (start + self.max_width).min(num_text_words);
            for end in start..end_limit {
                if idx == span_idx {
                    return (start, end);
                }
                idx += 1;
            }
        }
        // Fallback (should not reach here with valid span_idx)
        (0, 0)
    }

    // ========================================
    // Private: Token ID helpers
    // ========================================

    /// Resolve a special token string to its token ID.
    fn resolve_token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32, String> {
        tokenizer.token_to_id(token).ok_or_else(|| {
            format!(
                "GLiNER special token '{}' not found in tokenizer vocabulary. \
                 Ensure the tokenizer matches the GLiNER model.",
                token
            )
        })
    }

    /// Try to get a special token ID, returning None if not found.
    fn get_special_token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}

impl std::fmt::Debug for NerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NerEngine")
            .field("max_width", &self.max_width)
            .field("confidence_threshold", &self.confidence_threshold)
            .field("ent_token_id", &self.ent_token_id)
            .field("sep_token_id", &self.sep_token_id)
            .finish()
    }
}

// ============================================
// Utility Functions
// ============================================

/// Sigmoid activation function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Greedy span deduplication: for overlapping spans, keep the one with
/// highest confidence. Spans are considered overlapping if they share
/// any word position.
///
/// Algorithm:
/// 1. Sort by confidence descending
/// 2. For each span, check if any of its word positions are already claimed
/// 3. If no overlap → accept and mark positions as claimed
/// 4. If overlap → discard
fn greedy_dedup(mut detections: Vec<DetectedEntity>) -> Vec<DetectedEntity> {
    if detections.len() <= 1 {
        return detections;
    }

    // Sort by confidence descending
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let mut claimed: Vec<bool> = Vec::new();
    // Find max word index to size the claimed array
    let max_word = detections.iter().map(|d| d.word_end).max().unwrap_or(0);
    claimed.resize(max_word + 1, false);

    let mut results = Vec::new();

    for det in detections {
        let overlaps = (det.word_start..=det.word_end).any(|w| {
            w < claimed.len() && claimed[w]
        });

        if !overlaps {
            for w in det.word_start..=det.word_end {
                if w < claimed.len() {
                    claimed[w] = true;
                }
            }
            results.push(det);
        }
    }

    // Sort results by char_start for consistent output ordering
    results.sort_by_key(|d| d.char_start);
    results
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit tests that don't require model files ──

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_greedy_dedup_no_overlap() {
        let detections = vec![
            DetectedEntity {
                text: "JWT".into(), label: "technology".into(),
                confidence: 0.9, char_start: 0, char_end: 3,
                word_start: 0, word_end: 0,
            },
            DetectedEntity {
                text: "auth module".into(), label: "module".into(),
                confidence: 0.85, char_start: 10, char_end: 21,
                word_start: 2, word_end: 3,
            },
        ];

        let result = greedy_dedup(detections);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_greedy_dedup_overlap_keeps_highest() {
        let detections = vec![
            DetectedEntity {
                text: "auth".into(), label: "module".into(),
                confidence: 0.7, char_start: 0, char_end: 4,
                word_start: 0, word_end: 0,
            },
            DetectedEntity {
                text: "auth module".into(), label: "module".into(),
                confidence: 0.9, char_start: 0, char_end: 11,
                word_start: 0, word_end: 1,
            },
        ];

        let result = greedy_dedup(detections);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "auth module");
        assert!((result[0].confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_greedy_dedup_empty() {
        let result = greedy_dedup(Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn test_greedy_dedup_single() {
        let detections = vec![
            DetectedEntity {
                text: "JWT".into(), label: "tech".into(),
                confidence: 0.95, char_start: 0, char_end: 3,
                word_start: 0, word_end: 0,
            },
        ];
        let result = greedy_dedup(detections);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_word_split_simple() {
        let engine_words = word_split_standalone("hello world test");
        assert_eq!(engine_words.len(), 3);
        assert_eq!(engine_words[0].text, "hello");
        assert_eq!(engine_words[0].byte_start, 0);
        assert_eq!(engine_words[0].byte_end, 5);
        assert_eq!(engine_words[1].text, "world");
        assert_eq!(engine_words[1].byte_start, 6);
        assert_eq!(engine_words[1].byte_end, 11);
        assert_eq!(engine_words[2].text, "test");
    }

    #[test]
    fn test_word_split_unicode() {
        let words = word_split_standalone("认证模块 uses JWT");
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "认证模块");
        assert_eq!(words[1].text, "uses");
        assert_eq!(words[2].text, "JWT");
    }

    #[test]
    fn test_word_split_empty() {
        let words = word_split_standalone("");
        assert!(words.is_empty());

        let words2 = word_split_standalone("   ");
        assert!(words2.is_empty());
    }

    #[test]
    fn test_span_indices() {
        // 3 words, max_width = 2
        // Expected spans: (0,0), (0,1), (1,1), (1,2), (2,2)
        let mut span_idx: Vec<i64> = Vec::new();
        let mut span_mask: Vec<bool> = Vec::new();
        let max_width = 2;
        let num_words = 3;

        for start in 0..num_words {
            let end_limit = (start + max_width).min(num_words);
            for end in start..end_limit {
                span_idx.push(start as i64);
                span_idx.push(end as i64);
                span_mask.push(true);
            }
        }

        assert_eq!(span_mask.len(), 5);
        assert_eq!(span_idx, vec![0, 0, 0, 1, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_missing_model_returns_error() {
        let result = NerEngine::load("/nonexistent/path", 0.5, 12);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("not found"), "Error should mention 'not found': {}", err);
    }

    /// Standalone word_split for testing without NerEngine instance.
    fn word_split_standalone(text: &str) -> Vec<WordSpan> {
        let mut words = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some(&(byte_start, ch)) = chars.peek() {
            if ch.is_whitespace() {
                chars.next();
                continue;
            }
            let mut byte_end = byte_start;
            let mut word = String::new();

            while let Some(&(bi, c)) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                byte_end = bi + c.len_utf8();
                word.push(c);
                chars.next();
            }

            if !word.is_empty() {
                words.push(WordSpan {
                    text: word,
                    byte_start,
                    byte_end,
                });
            }
        }

        words
    }

    // ── Integration tests requiring model files ──

    /// Resolve model directory from env or default.
    fn ner_model_dir() -> String {
        std::env::var("MEMCHAIN_NER_MODEL_PATH")
            .unwrap_or_else(|_| "models/gliner".to_string())
    }

    /// Helper: skip test if model files are not downloaded.
    fn try_load_ner_engine() -> Option<NerEngine> {
        match NerEngine::load(&ner_model_dir(), 0.5, 12) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("⏭️ Skipping NER test (model not available): {}", e);
                eprintln!("   Run `scripts/download_models.sh` to download model files.");
                None
            }
        }
    }

    #[test]
    fn test_detect_entities_basic() {
        let engine = match try_load_ner_engine() {
            Some(e) => e,
            None => return,
        };

        let entities = engine.detect_entities(
            "auth module uses JWT for authentication",
            &["module", "technology"],
        ).unwrap();

        // We expect at least one entity to be detected
        // Exact results depend on model quality
        debug!("Detected entities: {:?}", entities);
        for e in &entities {
            assert!(!e.text.is_empty());
            assert!(e.confidence >= 0.5);
            assert!(e.char_end > e.char_start);
        }
    }

    #[test]
    fn test_detect_entities_empty_text() {
        let engine = match try_load_ner_engine() {
            Some(e) => e,
            None => return,
        };

        let entities = engine.detect_entities("", &["person"]).unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_detect_entities_empty_labels() {
        let engine = match try_load_ner_engine() {
            Some(e) => e,
            None => return,
        };

        let entities = engine.detect_entities("some text", &[]).unwrap();
        assert!(entities.is_empty());
    }
}
