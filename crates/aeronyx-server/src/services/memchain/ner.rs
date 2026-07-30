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
//! GLiNER enumerates all possible (start_word, end_word) spans as a fixed-size
//! grid of `num_words × max_width`. For N text words and W max_width:
//! - num_spans = N × W (fixed size, padded with mask=false for invalid spans)
//! - span_idx tensor: [batch, num_spans, 2] → reshapes to [batch, N, W, 2]
//! - span_mask tensor: [batch, num_spans] → reshapes to [batch, N, W]
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
//! - span_idx MUST be a fixed-size grid of num_words × max_width. The model
//!   internally reshapes it to [batch, num_words, max_width, 2]. Variable-length
//!   span lists cause reshape failures.
//! - text_lengths counts only TEXT words (excludes prompt prefix tokens).
//! - The tokenizer MUST match the model's training tokenizer (typically DeBERTa-v3
//!   for v2.x models). Using the wrong tokenizer silently degrades quality.
//! - ORT init is shared with embed.rs via the retryable success-only gate.
//! - Session::run() requires &mut self → shared `InferenceSession` wrapper
//!   (same recovery contract as embed.rs and reranker.rs).
//! - max_width (max entity span in words) defaults to 12. Larger values increase
//!   num_spans linearly (N × W) — only increase if needed.
//! - The `<<ENT>>` and `<<SEP>>` special tokens must be in the tokenizer vocabulary.
//!   If they're missing, the model was not exported correctly.
//! - Sequence start/end and unknown-token IDs are resolved from standard BERT
//!   or SentencePiece aliases during loading. Never fabricate token ID zero.
//! - `gliner_config.json` is an executable model contract. Prompt token count,
//!   label count, and span width must remain within its declared capacities.
//! - span_mask tensor type is bool (not u8). ort 2.0.0-rc.11 supports bool directly.
//!
//! ## Last Modified
//! v2.4.7-ModelPromptBudget -
//!   [MEMCHAIN-NER-PROMPT-BUDGET 2026-07-30 by Codex] Parse and validate
//!   GLiNER artifact limits, bound input scanning, and construct prompts under
//!   the model token budget without splitting a text word across inference.
//! v2.4.6-SpecialTokenContract -
//!   [MEMCHAIN-NER-SPECIAL-TOKENS 2026-07-30 by Codex] Added a load-time
//!   contract for every GLiNER special token, removed token-ID-zero fallbacks,
//!   and moved tokenizer validation ahead of expensive ONNX model loading.
//! v2.4.5-OnnxModelContract -
//!   [MEMCHAIN-ONNX-MODEL-CONTRACT 2026-07-30 by Codex] Validate all six
//!   GLiNER input tensors and the logits output during engine loading, then
//!   reuse the resolved output index during inference.
//! v2.4.4-OnnxOutputBoundary -
//!   [MEMCHAIN-ONNX-OUTPUT-BOUNDARY 2026-07-30 by Codex] Replaced panicking
//!   positional output indexing with the shared bounds-checked inference
//!   boundary.
//! v2.4.3-NerOutputValidation -
//!   [MEMCHAIN-NER-OUTPUT 2026-07-30 by Codex] Added strict 3D/4D output
//!   contracts, rejected non-finite logits, and made span-grid sizing bounded
//!   and fallible so incompatible models/configuration cannot panic or truncate.
//! v2.4.2-IndependentRuntimeInitialization -
//!   [NER-RUNTIME-INDEPENDENCE 2026-07-30 by Codex] Made NER initialize the
//!   shared ORT runtime itself and honored the configured tokenizer path.
//! v2.4.1-InferenceSessionRecovery -
//!   [MEMCHAIN-INFERENCE-SESSION 2026-07-30 by Codex] Reused the shared
//!   non-poisoning ONNX session boundary.
//! v2.4.0-GraphCognition - 🌟 Initial implementation
//! v2.4.0+BugFix - 🔧 Fixed span_indices to use fixed-size grid (num_words × max_width)
//!   instead of variable-length list. Fixes GLiNER reshape error:
//!   "input_shape:{1,210,512}, requested shape:{1,23,12,512}"
//! v2.4.0+BugFix - 🔧 Fixed span_mask tensor type from u8 to bool (B2 bug fix).
//!   GLiNER ONNX model expects tensor(bool), not tensor(uint8).

use std::{fs, path::Path};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use super::embed::{
    init_ort_runtime, require_onnx_output, validate_onnx_model_contract, InferenceSession,
    OnnxOutputSelector, OnnxTensorContract,
};

// ============================================
// Constants
// ============================================

/// Model filename within the NER model directory.
const MODEL_FILENAME: &str = "model.onnx";

/// Tokenizer filename within the NER model directory.
const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// GLiNER artifact configuration filename.
const MODEL_CONFIG_FILENAME: &str = "gliner_config.json";

/// GLiNER special token: entity type marker.
/// In the prompt, each label is preceded by this token.
const ENT_TOKEN: &str = "<<ENT>>";

/// GLiNER special token: separator between labels and text.
const SEP_TOKEN: &str = "<<SEP>>";

/// Supported sequence-start aliases across BERT and SentencePiece tokenizers.
const SEQUENCE_START_TOKEN_ALIASES: &[&str] = &["[CLS]", "<s>"];

/// Supported sequence-end aliases across BERT and SentencePiece tokenizers.
const SEQUENCE_END_TOKEN_ALIASES: &[&str] = &["[SEP]", "</s>"];

/// Supported unknown-token aliases across BERT and SentencePiece tokenizers.
const UNKNOWN_TOKEN_ALIASES: &[&str] = &["[UNK]", "<unk>"];

/// Default maximum entity span width in words.
/// Entities longer than this many words are not considered.
/// 12 covers most real-world entities (e.g., "New York City Department of Education").
const DEFAULT_MAX_WIDTH: usize = 12;

/// Default confidence threshold for entity detection.
/// Spans with sigmoid(logit) < threshold are discarded.
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Maximum text length in words for a single inference call.
/// Longer texts should be split into windows by the caller.
const MAX_TEXT_WORDS: usize = 512;

/// Maximum source bytes scanned for one NER inference call.
///
/// Token count is the authoritative model limit, but bounding the source scan
/// prevents a single whitespace-free request from forcing an unbounded copy
/// and tokenizer pass before that token budget can be applied.
const MAX_TEXT_INPUT_BYTES: usize = 64 * 1024;

/// Maximum UTF-8 bytes accepted for one entity-type label.
///
/// Production labels are short nouns such as `person` or `technology`.
/// Bounding them before tokenization prevents public library callers from
/// bypassing the prompt budget with a single oversized label string.
const MAX_LABEL_INPUT_BYTES: usize = 256;

/// Maximum configurable span width.
///
/// Wider spans cannot be observed because inference already caps text at
/// `MAX_TEXT_WORDS`; rejecting them also bounds the span-grid allocation.
const MAX_ENTITY_WIDTH: usize = MAX_TEXT_WORDS;

/// Bundled GLiNER small-v2.1 artifact defaults.
const DEFAULT_MODEL_MAX_SEQUENCE_TOKENS: usize = 384;
const DEFAULT_MODEL_MAX_LABELS: usize = 25;

/// Defensive ceilings for operator-supplied model metadata.
const HARD_MAX_SEQUENCE_TOKENS: usize = 4096;
const HARD_MAX_LABELS: usize = 128;

// ============================================
// NER output and span-grid contracts
// ============================================

/// Supported ONNX output layout after validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NerOutputLayout {
    /// `[batch, num_spans, num_labels]`.
    Flattened,
    /// `[batch, num_words, max_width, num_labels]`.
    Grid,
}

/// Compute the fixed GLiNER span-grid capacities without integer overflow.
fn checked_span_dimensions(num_words: usize, max_width: usize) -> Result<(usize, usize), String> {
    let num_spans = num_words.checked_mul(max_width).ok_or_else(|| {
        format!(
            "GLiNER span grid overflow: {} words x width {}",
            num_words, max_width
        )
    })?;
    let span_index_values = num_spans.checked_mul(2).ok_or_else(|| {
        format!(
            "GLiNER span index capacity overflow for {} spans",
            num_spans
        )
    })?;

    Ok((num_spans, span_index_values))
}

/// Validate GLiNER's flattened or grid-form output contract.
///
/// [MEMCHAIN-NER-OUTPUT 2026-07-30 by Codex] Every axis must match the
/// tensors built for this request. Accepting smaller axes and decoding with
/// `min()` hides model incompatibility; accepting larger axes misaligns spans.
fn validate_ner_output_shape(
    shape: &[usize],
    expected_words: usize,
    expected_width: usize,
    expected_labels: usize,
) -> Result<NerOutputLayout, String> {
    let (expected_spans, _) = checked_span_dimensions(expected_words, expected_width)?;

    match shape {
        [batch, spans, labels]
            if *batch == 1 && *spans == expected_spans && *labels == expected_labels =>
        {
            Ok(NerOutputLayout::Flattened)
        }
        [batch, words, width, labels]
            if *batch == 1
                && *words == expected_words
                && *width == expected_width
                && *labels == expected_labels =>
        {
            Ok(NerOutputLayout::Grid)
        }
        _ => Err(format!(
            "GLiNER output shape {:?} does not match [1, {}, {}] or [1, {}, {}, {}]",
            shape, expected_spans, expected_labels, expected_words, expected_width, expected_labels
        )),
    }
}

/// Reject non-finite logits before sigmoid and confidence sorting.
fn validate_finite_ner_logits(values: impl Iterator<Item = f32>) -> Result<(), String> {
    if let Some((index, value)) = values.enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(format!(
            "GLiNER output contains non-finite logit at flat index {}: {}",
            index, value
        ));
    }

    Ok(())
}

// ============================================
// Model artifact limits
// ============================================

/// Capacity fields consumed from the GLiNER artifact configuration.
///
/// The upstream file contains training metadata that the runtime does not
/// need. Optional fields preserve compatibility with older artifact bundles;
/// absent fields use the audited small-v2.1 defaults.
#[derive(Debug, Default, Deserialize)]
struct GlinerModelConfigFile {
    max_len: Option<usize>,
    max_types: Option<usize>,
    max_width: Option<usize>,
}

/// Validated inference capacities for one GLiNER artifact.
///
/// [MEMCHAIN-NER-PROMPT-BUDGET 2026-07-30 by Codex] This is deliberately
/// separate from operator tuning. Artifact limits describe what the exported
/// model can accept; operator values may narrow them but never expand them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GlinerModelLimits {
    max_sequence_tokens: usize,
    max_labels: usize,
    max_width: usize,
}

impl Default for GlinerModelLimits {
    fn default() -> Self {
        Self {
            max_sequence_tokens: DEFAULT_MODEL_MAX_SEQUENCE_TOKENS,
            max_labels: DEFAULT_MODEL_MAX_LABELS,
            max_width: DEFAULT_MAX_WIDTH,
        }
    }
}

impl GlinerModelLimits {
    fn load(model_dir: &Path) -> Result<Self, String> {
        let config_path = model_dir.join(MODEL_CONFIG_FILENAME);
        if !config_path.exists() {
            return Ok(Self::default());
        }

        let bytes = fs::read(&config_path).map_err(|error| {
            format!("GLiNER config read ({}): {}", config_path.display(), error)
        })?;
        let config: GlinerModelConfigFile = serde_json::from_slice(&bytes).map_err(|error| {
            format!("GLiNER config parse ({}): {}", config_path.display(), error)
        })?;

        Self::from_config(config)
    }

    fn from_config(config: GlinerModelConfigFile) -> Result<Self, String> {
        let limits = Self {
            max_sequence_tokens: config.max_len.unwrap_or(DEFAULT_MODEL_MAX_SEQUENCE_TOKENS),
            max_labels: config.max_types.unwrap_or(DEFAULT_MODEL_MAX_LABELS),
            max_width: config.max_width.unwrap_or(DEFAULT_MAX_WIDTH),
        };

        if !(4..=HARD_MAX_SEQUENCE_TOKENS).contains(&limits.max_sequence_tokens) {
            return Err(format!(
                "GLiNER max_len {} is outside supported range 4..={}",
                limits.max_sequence_tokens, HARD_MAX_SEQUENCE_TOKENS
            ));
        }
        if !(1..=HARD_MAX_LABELS).contains(&limits.max_labels) {
            return Err(format!(
                "GLiNER max_types {} is outside supported range 1..={}",
                limits.max_labels, HARD_MAX_LABELS
            ));
        }
        if !(1..=MAX_ENTITY_WIDTH).contains(&limits.max_width) {
            return Err(format!(
                "GLiNER max_width {} is outside supported range 1..={}",
                limits.max_width, MAX_ENTITY_WIDTH
            ));
        }

        Ok(limits)
    }
}

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

/// Fully budgeted GLiNER sequence inputs before tensor construction.
#[derive(Debug, PartialEq, Eq)]
struct GlinerPrompt {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    words_mask: Vec<i64>,
    text_length: usize,
}

/// Complete tokenizer contract required to construct a GLiNER prompt.
///
/// [MEMCHAIN-NER-SPECIAL-TOKENS 2026-07-30 by Codex] These IDs are resolved
/// exactly once during engine loading. A tokenizer that cannot represent the
/// prompt must disable NER cleanly rather than inject token ID zero and return
/// plausible-looking but invalid entity results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GlinerSpecialTokenIds {
    sequence_start: u32,
    sequence_end: u32,
    unknown: u32,
    entity_marker: u32,
    prompt_separator: u32,
}

impl GlinerSpecialTokenIds {
    fn resolve(tokenizer: &Tokenizer) -> Result<Self, String> {
        Self::resolve_with(|token| tokenizer.token_to_id(token))
    }

    fn resolve_with(mut lookup: impl FnMut(&str) -> Option<u32>) -> Result<Self, String> {
        Ok(Self {
            sequence_start: resolve_token_alias(
                "sequence-start",
                SEQUENCE_START_TOKEN_ALIASES,
                &mut lookup,
            )?,
            sequence_end: resolve_token_alias(
                "sequence-end",
                SEQUENCE_END_TOKEN_ALIASES,
                &mut lookup,
            )?,
            unknown: resolve_token_alias("unknown", UNKNOWN_TOKEN_ALIASES, &mut lookup)?,
            entity_marker: resolve_token_alias("entity marker", &[ENT_TOKEN], &mut lookup)?,
            prompt_separator: resolve_token_alias("prompt separator", &[SEP_TOKEN], &mut lookup)?,
        })
    }
}

fn resolve_token_alias(
    role: &str,
    aliases: &[&str],
    mut lookup: impl FnMut(&str) -> Option<u32>,
) -> Result<u32, String> {
    aliases
        .iter()
        .find_map(|token| lookup(token))
        .ok_or_else(|| {
            format!(
                "GLiNER {} token not found in tokenizer vocabulary; expected one of {:?}",
                role, aliases
            )
        })
}

/// Split a bounded UTF-8 prefix into owned words with original byte offsets.
///
/// [MEMCHAIN-NER-PROMPT-BUDGET 2026-07-30 by Codex] Both word count and source
/// bytes are bounded before allocation. When the byte ceiling falls inside a
/// word, that partial word is omitted so inference never reports an entity for
/// text it did not fully inspect.
fn split_words_bounded(text: &str, max_words: usize, max_bytes: usize) -> Vec<WordSpan> {
    if max_words == 0 || max_bytes == 0 || text.is_empty() {
        return Vec::new();
    }

    let mut scan_end = text.len().min(max_bytes);
    while scan_end > 0 && !text.is_char_boundary(scan_end) {
        scan_end -= 1;
    }

    if scan_end < text.len() && scan_end > 0 {
        let previous_is_whitespace = text[..scan_end]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace);
        let next_is_whitespace = text[scan_end..]
            .chars()
            .next()
            .is_some_and(char::is_whitespace);
        if !previous_is_whitespace && !next_is_whitespace {
            scan_end = text[..scan_end]
                .char_indices()
                .rev()
                .find_map(|(index, ch)| ch.is_whitespace().then_some(index))
                .unwrap_or(0);
        }
    }

    let mut words = Vec::new();
    let mut chars = text[..scan_end].char_indices().peekable();
    while words.len() < max_words {
        let Some(&(byte_start, ch)) = chars.peek() else {
            break;
        };
        if ch.is_whitespace() {
            chars.next();
            continue;
        }

        let mut byte_end = byte_start;
        let mut word = String::new();
        while let Some(&(byte_index, current)) = chars.peek() {
            if current.is_whitespace() {
                break;
            }
            byte_end = byte_index + current.len_utf8();
            word.push(current);
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

/// Build a token-budgeted GLiNER prompt with the production tokenizer.
fn build_gliner_prompt(
    tokenizer: &Tokenizer,
    special_tokens: GlinerSpecialTokenIds,
    labels: &[&str],
    words: &[WordSpan],
    max_sequence_tokens: usize,
) -> Result<GlinerPrompt, String> {
    build_gliner_prompt_with(
        special_tokens,
        labels,
        words,
        max_sequence_tokens,
        |value| {
            tokenizer
                .encode(value.to_owned(), false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| error.to_string())
        },
    )
}

/// Assemble one prompt using a replaceable tokenizer boundary.
///
/// The injected tokenizer keeps token-budget behavior model-free in unit
/// tests. Text words are atomic: if the next word does not fit together with
/// the final sequence-end token, construction stops before that word.
fn build_gliner_prompt_with(
    special_tokens: GlinerSpecialTokenIds,
    labels: &[&str],
    words: &[WordSpan],
    max_sequence_tokens: usize,
    mut tokenize: impl FnMut(&str) -> Result<Vec<u32>, String>,
) -> Result<GlinerPrompt, String> {
    let mut token_ids = vec![special_tokens.sequence_start];
    let mut words_mask = vec![0_i64];

    for label in labels {
        if label.is_empty() || label.len() > MAX_LABEL_INPUT_BYTES {
            return Err(format!(
                "GLiNER label length {} is outside supported range 1..={}",
                label.len(),
                MAX_LABEL_INPUT_BYTES
            ));
        }
        let label_ids =
            tokenize(label).map_err(|error| format!("Label tokenization failed: {}", error))?;
        let projected = token_ids
            .len()
            .checked_add(1)
            .and_then(|length| length.checked_add(label_ids.len()))
            .and_then(|length| length.checked_add(2))
            .ok_or_else(|| "GLiNER label prompt size overflow".to_string())?;
        if projected > max_sequence_tokens {
            return Err(format!(
                "GLiNER label prompt exceeds model token limit {}",
                max_sequence_tokens
            ));
        }

        token_ids.push(special_tokens.entity_marker);
        words_mask.push(0);
        token_ids.extend(label_ids.iter().copied());
        words_mask.resize(token_ids.len(), 0);
    }

    token_ids.push(special_tokens.prompt_separator);
    words_mask.push(0);

    let mut text_length = 0usize;
    for word in words {
        let mut word_ids =
            tokenize(&word.text).map_err(|error| format!("Word tokenization failed: {}", error))?;
        if word_ids.is_empty() {
            word_ids.push(special_tokens.unknown);
        }

        let projected = token_ids
            .len()
            .checked_add(word_ids.len())
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| "GLiNER text prompt size overflow".to_string())?;
        if projected > max_sequence_tokens {
            break;
        }

        text_length += 1;
        token_ids.extend(word_ids.iter().copied());
        words_mask.resize(token_ids.len(), text_length as i64);
    }

    if text_length == 0 {
        return Err(format!(
            "GLiNER model token limit {} leaves no room for a complete text word",
            max_sequence_tokens
        ));
    }

    token_ids.push(special_tokens.sequence_end);
    words_mask.push(0);
    debug_assert_eq!(token_ids.len(), words_mask.len());
    debug_assert!(token_ids.len() <= max_sequence_tokens);

    let input_ids = token_ids.into_iter().map(i64::from).collect::<Vec<_>>();
    let attention_mask = vec![1_i64; input_ids.len()];

    Ok(GlinerPrompt {
        input_ids,
        attention_mask,
        words_mask,
        text_length,
    })
}

// ============================================
// NerEngine
// ============================================

/// Local NER engine using GLiNER ONNX model.
///
/// Thread-safe: `ort::Session` is wrapped in `InferenceSession` because
/// `Session::run()` requires `&mut self` in ort 2.0.0-rc.11.
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
    /// ONNX session serialized through the shared non-poisoning boundary.
    session: InferenceSession<Session>,
    /// Load-time-validated logits output index.
    output_idx: usize,
    /// HuggingFace tokenizer (DeBERTa-v3 or BERT depending on model).
    tokenizer: Tokenizer,
    /// Maximum entity span width in words.
    max_width: usize,
    /// Confidence threshold for filtering detections.
    confidence_threshold: f32,
    /// Complete, load-time-validated prompt token contract.
    special_tokens: GlinerSpecialTokenIds,
    /// Maximum complete prompt length declared by the model artifact.
    max_sequence_tokens: usize,
    /// Maximum entity labels declared by the model artifact.
    max_labels: usize,
}

impl NerEngine {
    /// Load GLiNER ONNX model and tokenizer from the given directory.
    ///
    /// This entry point preserves the historical API and resolves the tokenizer
    /// to `{model_dir}/tokenizer.json`. Use [`Self::load_with_tokenizer`] when
    /// the tokenizer is stored elsewhere.
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
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);
        Self::load_with_tokenizer(model_dir, tokenizer_path, confidence_threshold, max_width)
    }

    /// Load GLiNER with an explicit tokenizer path.
    ///
    /// [NER-RUNTIME-INDEPENDENCE 2026-07-30 by Codex] NER is an independently
    /// configurable engine. It initializes the process-global ORT runtime
    /// itself instead of relying on embedding or reranking to run first.
    pub fn load_with_tokenizer(
        model_dir: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        confidence_threshold: f32,
        max_width: usize,
    ) -> Result<Self, String> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = tokenizer_path.as_ref();
        let confidence_threshold = if confidence_threshold <= 0.0 || confidence_threshold >= 1.0 {
            DEFAULT_CONFIDENCE_THRESHOLD
        } else {
            confidence_threshold
        };
        let max_width = if max_width == 0 {
            DEFAULT_MAX_WIDTH
        } else {
            max_width
        };
        if max_width > MAX_ENTITY_WIDTH {
            return Err(format!(
                "GLiNER max_width {} exceeds maximum {}",
                max_width, MAX_ENTITY_WIDTH
            ));
        }

        let model_path = model_dir.join(MODEL_FILENAME);

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

        let model_limits = GlinerModelLimits::load(model_dir)?;
        if max_width > model_limits.max_width {
            return Err(format!(
                "GLiNER max_width {} exceeds model capacity {}",
                max_width, model_limits.max_width
            ));
        }

        // Load and validate the tokenizer before initializing ORT or loading
        // the model's large weight file. An incompatible artifact should fail
        // quickly and leave other independently configured engines retryable.
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            format!(
                "GLiNER tokenizer load ({}): {}",
                tokenizer_path.display(),
                e
            )
        })?;
        let special_tokens = GlinerSpecialTokenIds::resolve(&tokenizer)?;

        init_ort_runtime(model_dir)?;

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

        let output_idx = validate_onnx_model_contract(
            "GLiNER",
            session.inputs(),
            session.outputs(),
            &[
                OnnxTensorContract::new("input_ids", TensorElementType::Int64, &[2]),
                OnnxTensorContract::new("attention_mask", TensorElementType::Int64, &[2]),
                OnnxTensorContract::new("words_mask", TensorElementType::Int64, &[2]),
                OnnxTensorContract::new("text_lengths", TensorElementType::Int64, &[2]),
                OnnxTensorContract::new("span_idx", TensorElementType::Int64, &[3]),
                OnnxTensorContract::new("span_mask", TensorElementType::Bool, &[2]),
            ],
            OnnxOutputSelector::NamedOrOnly("logits"),
            TensorElementType::Float32,
            &[3, 4],
        )?;

        info!(
            tokenizer = %tokenizer_path.display(),
            max_width = max_width,
            max_sequence_tokens = model_limits.max_sequence_tokens,
            max_labels = model_limits.max_labels,
            threshold = confidence_threshold,
            sequence_start_id = special_tokens.sequence_start,
            sequence_end_id = special_tokens.sequence_end,
            unknown_id = special_tokens.unknown,
            entity_marker_id = special_tokens.entity_marker,
            prompt_separator_id = special_tokens.prompt_separator,
            "[NER] GLiNER tokenizer loaded"
        );

        Ok(Self {
            session: InferenceSession::new(session),
            output_idx,
            tokenizer,
            max_width,
            confidence_threshold,
            special_tokens,
            max_sequence_tokens: model_limits.max_sequence_tokens,
            max_labels: model_limits.max_labels,
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
    /// 2. Build GLiNER prompt: [CLS] <<ENT>> label1 ... <<SEP>> word1 word2 ... [SEP]
    /// 3. Subword-tokenize the prompt, tracking word boundaries
    /// 4. Build span indices as fixed-size grid (num_words × max_width)
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
        if labels.len() > self.max_labels {
            return Err(format!(
                "Too many labels: {} (max {})",
                labels.len(),
                self.max_labels
            ));
        }

        // Step 1: Word-split the text
        let words = self.word_split(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2-3: Build prompt and tokenize
        let prompt = build_gliner_prompt(
            &self.tokenizer,
            self.special_tokens,
            labels,
            &words,
            self.max_sequence_tokens,
        )?;
        let num_text_words = prompt.text_length;
        let words = &words[..num_text_words];

        let seq_len = prompt.input_ids.len();

        // Step 4: Build span indices (fixed-size grid: num_words × max_width)
        let (span_idx_flat, span_mask_flat, num_spans) = self.build_span_indices(num_text_words)?;

        if num_spans == 0 {
            return Ok(Vec::new());
        }

        // Step 5: Build tensors
        let batch_size = 1usize;
        let shape_2d = [batch_size, seq_len];
        let shape_tl = [batch_size, 1usize];
        let shape_span_idx = [batch_size, num_spans, 2usize];
        let shape_span_mask = [batch_size, num_spans];

        let ids_tensor = Tensor::from_array((shape_2d, prompt.input_ids.into_boxed_slice()))
            .map_err(|e| format!("input_ids tensor: {}", e))?;

        let mask_tensor = Tensor::from_array((shape_2d, prompt.attention_mask.into_boxed_slice()))
            .map_err(|e| format!("attention_mask tensor: {}", e))?;

        let words_mask_tensor =
            Tensor::from_array((shape_2d, prompt.words_mask.into_boxed_slice()))
                .map_err(|e| format!("words_mask tensor: {}", e))?;

        let text_lengths_tensor =
            Tensor::from_array((shape_tl, vec![num_text_words as i64].into_boxed_slice()))
                .map_err(|e| format!("text_lengths tensor: {}", e))?;

        let span_idx_tensor =
            Tensor::from_array((shape_span_idx, span_idx_flat.into_boxed_slice()))
                .map_err(|e| format!("span_idx tensor: {}", e))?;

        // v2.4.0+BugFix: span_mask is bool — ort 2.0.0-rc.11 supports bool directly.
        // Previously used u8, which caused: "expected: (tensor(bool)), actual: (tensor(uint8))"
        let span_mask_tensor =
            Tensor::from_array((shape_span_mask, span_mask_flat.into_boxed_slice()))
                .map_err(|e| format!("span_mask tensor: {}", e))?;

        // Step 6: ONNX inference
        let mut session = self.session.lock();

        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "words_mask" => words_mask_tensor,
                "text_lengths" => text_lengths_tensor,
                "span_idx" => span_idx_tensor,
                "span_mask" => span_mask_tensor,
            ])
            .map_err(|e| format!("GLiNER ONNX inference: {}", e))?;

        // Output: logits [batch_size, num_spans, num_labels]
        let logits_output = require_onnx_output("GLiNER", outputs.values(), self.output_idx)?;
        let logits = logits_output
            .try_extract_array::<f32>()
            .map_err(|e| format!("GLiNER output extraction: {}", e))?;

        // [MEMCHAIN-NER-OUTPUT 2026-07-30 by Codex] Validate every output axis
        // and value before indexed decoding. Model mismatch must be an explicit
        // inference failure rather than a partial result or index panic.
        let output_layout = validate_ner_output_shape(
            logits.shape(),
            num_text_words,
            self.max_width,
            labels.len(),
        )?;
        validate_finite_ner_logits(logits.iter().copied())?;

        // Step 7: Sigmoid + threshold + decode
        let mut raw_detections: Vec<DetectedEntity> = Vec::new();

        for start in 0..num_text_words {
            for offset in 0..self.max_width {
                let end = start + offset;
                if end >= num_text_words {
                    break;
                }

                for (label_index, label) in labels.iter().enumerate() {
                    let logit = match output_layout {
                        NerOutputLayout::Grid => logits[[0, start, offset, label_index]],
                        NerOutputLayout::Flattened => {
                            let span_index = start * self.max_width + offset;
                            logits[[0, span_index, label_index]]
                        }
                    };
                    let score = sigmoid(logit);

                    if score >= self.confidence_threshold {
                        let char_start = words[start].byte_start;
                        let char_end = words[end].byte_end;
                        let entity_text = &text[char_start..char_end];

                        raw_detections.push(DetectedEntity {
                            text: entity_text.to_string(),
                            label: (*label).to_string(),
                            confidence: score,
                            char_start,
                            char_end,
                            word_start: start,
                            word_end: end,
                        });
                    }
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
        split_words_bounded(text, MAX_TEXT_WORDS, MAX_TEXT_INPUT_BYTES)
    }

    // ========================================
    // Private: Span Index Construction
    // ========================================

    /// Build span indices as a fixed-size grid of num_text_words × max_width.
    ///
    /// GLiNER internally reshapes span_idx from [batch, num_spans, 2] to
    /// [batch, num_words, max_width, 2]. This requires num_spans to be
    /// EXACTLY num_words × max_width. Invalid spans (where end >= num_words)
    /// are filled with (0, 0) and masked with span_mask=false.
    ///
    /// ## Enumeration order
    /// For each start word (0..num_words):
    ///   For each offset (0..max_width):
    ///     end = start + offset
    ///     if end < num_words → valid span (start, end), mask=true
    ///     else → padding (0, 0), mask=false
    ///
    /// Returns the fixed-size span buffers, or an error if capacity arithmetic
    /// overflows or the allocation cannot be reserved.
    fn build_span_indices(
        &self,
        num_text_words: usize,
    ) -> Result<(Vec<i64>, Vec<bool>, usize), String> {
        let (num_spans, span_index_values) =
            checked_span_dimensions(num_text_words, self.max_width)?;
        let mut span_idx: Vec<i64> = Vec::new();
        let mut span_mask: Vec<bool> = Vec::new();
        span_idx
            .try_reserve_exact(span_index_values)
            .map_err(|error| {
                format!(
                    "GLiNER span index allocation failed for {} values: {}",
                    span_index_values, error
                )
            })?;
        span_mask.try_reserve_exact(num_spans).map_err(|error| {
            format!(
                "GLiNER span mask allocation failed for {} spans: {}",
                num_spans, error
            )
        })?;

        for start in 0..num_text_words {
            for offset in 0..self.max_width {
                let end = start + offset;
                if end < num_text_words {
                    // Valid span
                    span_idx.push(start as i64);
                    span_idx.push(end as i64);
                    span_mask.push(true);
                } else {
                    // Out of bounds — padding
                    span_idx.push(0);
                    span_idx.push(0);
                    span_mask.push(false);
                }
            }
        }

        Ok((span_idx, span_mask, num_spans))
    }

    /// Convert a linear span index back to (start_word, end_word) pair.
    ///
    /// Must match the enumeration order in build_span_indices():
    /// index = start * max_width + offset, where end = start + offset.
    fn span_index_to_words(&self, span_idx: usize, _num_text_words: usize) -> (usize, usize) {
        let start = span_idx / self.max_width;
        let offset = span_idx % self.max_width;
        (start, start + offset)
    }

    // ========================================
    // Private: Token ID helpers
    // ========================================
}

impl std::fmt::Debug for NerEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NerEngine")
            .field("max_width", &self.max_width)
            .field("max_sequence_tokens", &self.max_sequence_tokens)
            .field("max_labels", &self.max_labels)
            .field("confidence_threshold", &self.confidence_threshold)
            .field("special_tokens", &self.special_tokens)
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

    // Sort by confidence descending. Logits are validated before decoding, so
    // total ordering is both deterministic and explicit.
    detections.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut claimed: Vec<bool> = Vec::new();
    // Find max word index to size the claimed array
    let max_word = detections.iter().map(|d| d.word_end).max().unwrap_or(0);
    claimed.resize(max_word + 1, false);

    let mut results = Vec::new();

    for det in detections {
        let overlaps = (det.word_start..=det.word_end).any(|w| w < claimed.len() && claimed[w]);

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

    fn test_special_tokens() -> GlinerSpecialTokenIds {
        GlinerSpecialTokenIds {
            sequence_start: 1,
            sequence_end: 2,
            unknown: 3,
            entity_marker: 4,
            prompt_separator: 5,
        }
    }

    fn test_words(values: &[&str]) -> Vec<WordSpan> {
        let mut byte_offset = 0usize;
        values
            .iter()
            .map(|value| {
                let byte_start = byte_offset;
                let byte_end = byte_start + value.len();
                byte_offset = byte_end + 1;
                WordSpan {
                    text: (*value).to_string(),
                    byte_start,
                    byte_end,
                }
            })
            .collect()
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn special_token_contract_accepts_bert_and_prefers_primary_aliases() {
        let tokens = GlinerSpecialTokenIds::resolve_with(|token| match token {
            "[CLS]" => Some(1),
            "<s>" => Some(101),
            "[SEP]" => Some(2),
            "</s>" => Some(102),
            "[UNK]" => Some(3),
            "<unk>" => Some(103),
            "<<ENT>>" => Some(4),
            "<<SEP>>" => Some(5),
            _ => None,
        })
        .unwrap();

        assert_eq!(
            tokens,
            GlinerSpecialTokenIds {
                sequence_start: 1,
                sequence_end: 2,
                unknown: 3,
                entity_marker: 4,
                prompt_separator: 5,
            }
        );
    }

    #[test]
    fn special_token_contract_accepts_sentencepiece_aliases() {
        let tokens = GlinerSpecialTokenIds::resolve_with(|token| match token {
            "<s>" => Some(11),
            "</s>" => Some(12),
            "<unk>" => Some(13),
            "<<ENT>>" => Some(14),
            "<<SEP>>" => Some(15),
            _ => None,
        })
        .unwrap();

        assert_eq!(tokens.sequence_start, 11);
        assert_eq!(tokens.sequence_end, 12);
        assert_eq!(tokens.unknown, 13);
    }

    #[test]
    fn special_token_contract_rejects_missing_required_role() {
        let error = GlinerSpecialTokenIds::resolve_with(|token| match token {
            "[CLS]" => Some(1),
            "[SEP]" => Some(2),
            "<<ENT>>" => Some(4),
            "<<SEP>>" => Some(5),
            _ => None,
        })
        .unwrap_err();

        assert!(error.contains("unknown token not found"));
        assert!(error.contains("[UNK]"));
        assert!(error.contains("<unk>"));
    }

    #[test]
    fn model_limits_use_audited_defaults_and_validate_artifact_values() {
        assert_eq!(
            GlinerModelLimits::from_config(GlinerModelConfigFile::default()).unwrap(),
            GlinerModelLimits::default()
        );
        assert_eq!(
            GlinerModelLimits::from_config(GlinerModelConfigFile {
                max_len: Some(384),
                max_types: Some(25),
                max_width: Some(12),
            })
            .unwrap(),
            GlinerModelLimits {
                max_sequence_tokens: 384,
                max_labels: 25,
                max_width: 12,
            }
        );

        for config in [
            GlinerModelConfigFile {
                max_len: Some(0),
                ..Default::default()
            },
            GlinerModelConfigFile {
                max_types: Some(HARD_MAX_LABELS + 1),
                ..Default::default()
            },
            GlinerModelConfigFile {
                max_width: Some(MAX_ENTITY_WIDTH + 1),
                ..Default::default()
            },
        ] {
            assert!(GlinerModelLimits::from_config(config).is_err());
        }
    }

    #[test]
    fn model_limits_load_optional_artifact_config_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            GlinerModelLimits::load(directory.path()).unwrap(),
            GlinerModelLimits::default()
        );

        let config_path = directory.path().join(MODEL_CONFIG_FILENAME);
        std::fs::write(
            &config_path,
            br#"{"max_len":384,"max_types":25,"max_width":12}"#,
        )
        .unwrap();
        assert_eq!(
            GlinerModelLimits::load(directory.path()).unwrap(),
            GlinerModelLimits {
                max_sequence_tokens: 384,
                max_labels: 25,
                max_width: 12,
            }
        );

        std::fs::write(&config_path, b"{not-json").unwrap();
        let error = GlinerModelLimits::load(directory.path()).unwrap_err();
        assert!(error.contains("config parse"));
        assert!(error.contains(&config_path.display().to_string()));
    }

    #[test]
    fn prompt_budget_keeps_only_complete_text_words() {
        let words = test_words(&["alpha", "bravo"]);
        let prompt =
            build_gliner_prompt_with(test_special_tokens(), &["person"], &words, 8, |value| {
                match value {
                    "person" => Ok(vec![10]),
                    "alpha" => Ok(vec![20, 21]),
                    "bravo" => Ok(vec![30, 31]),
                    _ => Err("unexpected tokenization input".to_string()),
                }
            })
            .unwrap();

        assert_eq!(prompt.input_ids, vec![1, 4, 10, 5, 20, 21, 2]);
        assert_eq!(prompt.words_mask, vec![0, 0, 0, 0, 1, 1, 0]);
        assert_eq!(prompt.text_length, 1);
        assert_eq!(prompt.input_ids.len(), prompt.attention_mask.len());
    }

    #[test]
    fn prompt_budget_rejects_oversized_labels_and_unrepresentable_text() {
        let words = test_words(&["alpha"]);
        let oversized_label_text = "x".repeat(MAX_LABEL_INPUT_BYTES + 1);
        let oversized_label_length = build_gliner_prompt_with(
            test_special_tokens(),
            &[oversized_label_text.as_str()],
            &words,
            384,
            |_| Ok(vec![10]),
        )
        .unwrap_err();
        assert!(oversized_label_length.contains("label length"));

        let oversized_label =
            build_gliner_prompt_with(test_special_tokens(), &["person"], &words, 6, |value| {
                match value {
                    "person" => Ok(vec![10, 11, 12]),
                    "alpha" => Ok(vec![20]),
                    _ => Ok(Vec::new()),
                }
            })
            .unwrap_err();
        assert!(oversized_label.contains("label prompt exceeds"));

        let oversized_word =
            build_gliner_prompt_with(test_special_tokens(), &["person"], &words, 7, |value| {
                match value {
                    "person" => Ok(vec![10]),
                    "alpha" => Ok(vec![20, 21, 22]),
                    _ => Ok(Vec::new()),
                }
            })
            .unwrap_err();
        assert!(oversized_word.contains("no room for a complete text word"));
    }

    #[test]
    fn prompt_budget_uses_validated_unknown_token_for_empty_encoding() {
        let words = test_words(&["unknown"]);
        let prompt =
            build_gliner_prompt_with(test_special_tokens(), &["person"], &words, 8, |value| {
                match value {
                    "person" => Ok(vec![10]),
                    "unknown" => Ok(Vec::new()),
                    _ => Ok(Vec::new()),
                }
            })
            .unwrap();

        assert_eq!(prompt.input_ids, vec![1, 4, 10, 5, 3, 2]);
        assert_eq!(prompt.text_length, 1);
    }

    #[test]
    fn ner_output_shape_accepts_exact_flattened_and_grid_layouts() {
        assert_eq!(
            validate_ner_output_shape(&[1, 6, 2], 3, 2, 2).unwrap(),
            NerOutputLayout::Flattened
        );
        assert_eq!(
            validate_ner_output_shape(&[1, 3, 2, 2], 3, 2, 2).unwrap(),
            NerOutputLayout::Grid
        );
    }

    #[test]
    fn ner_output_shape_rejects_axis_and_label_mismatches() {
        for shape in [
            &[2, 6, 2][..],
            &[1, 5, 2],
            &[1, 6, 1],
            &[1, 2, 2, 2],
            &[1, 3, 1, 2],
            &[1, 3, 2, 3],
            &[6, 2],
        ] {
            let error = validate_ner_output_shape(shape, 3, 2, 2).unwrap_err();
            assert!(error.contains("does not match"), "shape={:?}", shape);
        }
    }

    #[test]
    fn ner_output_rejects_non_finite_logits() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = validate_finite_ner_logits([0.0, value].into_iter()).unwrap_err();
            assert!(error.contains("non-finite"));
        }
    }

    #[test]
    fn span_dimensions_are_checked_before_allocation() {
        assert_eq!(checked_span_dimensions(3, 2).unwrap(), (6, 12));
        assert!(checked_span_dimensions(usize::MAX, 2).is_err());
    }

    #[test]
    fn max_width_is_bounded_before_model_loading() {
        let error = NerEngine::load("/nonexistent/path", 0.5, MAX_ENTITY_WIDTH.saturating_add(1))
            .unwrap_err();

        assert!(error.contains("max_width"));
        assert!(error.contains("exceeds maximum"));
    }

    #[test]
    fn test_greedy_dedup_no_overlap() {
        let detections = vec![
            DetectedEntity {
                text: "JWT".into(),
                label: "technology".into(),
                confidence: 0.9,
                char_start: 0,
                char_end: 3,
                word_start: 0,
                word_end: 0,
            },
            DetectedEntity {
                text: "auth module".into(),
                label: "module".into(),
                confidence: 0.85,
                char_start: 10,
                char_end: 21,
                word_start: 2,
                word_end: 3,
            },
        ];

        let result = greedy_dedup(detections);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_greedy_dedup_overlap_keeps_highest() {
        let detections = vec![
            DetectedEntity {
                text: "auth".into(),
                label: "module".into(),
                confidence: 0.7,
                char_start: 0,
                char_end: 4,
                word_start: 0,
                word_end: 0,
            },
            DetectedEntity {
                text: "auth module".into(),
                label: "module".into(),
                confidence: 0.9,
                char_start: 0,
                char_end: 11,
                word_start: 0,
                word_end: 1,
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
        let detections = vec![DetectedEntity {
            text: "JWT".into(),
            label: "tech".into(),
            confidence: 0.95,
            char_start: 0,
            char_end: 3,
            word_start: 0,
            word_end: 0,
        }];
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
    fn word_split_bounds_count_bytes_and_utf8_without_partial_words() {
        let count_limited = split_words_bounded("one two three", 2, usize::MAX);
        assert_eq!(
            count_limited
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            vec!["one", "two"]
        );

        let byte_limited = split_words_bounded("alpha bravo charlie", 10, 9);
        assert_eq!(byte_limited.len(), 1);
        assert_eq!(byte_limited[0].text, "alpha");
        assert_eq!(byte_limited[0].byte_end, 5);

        let unicode_limited = split_words_bounded("认证 模块", 10, 8);
        assert_eq!(unicode_limited.len(), 1);
        assert_eq!(unicode_limited[0].text, "认证");

        let giant_first_word = split_words_bounded("abcdefghij tail", 10, 5);
        assert!(giant_first_word.is_empty());
    }

    #[test]
    fn test_span_indices() {
        // 3 words, max_width = 2
        // Grid: 3 × 2 = 6 spans
        // (0,0), (0,1),   ← start=0, offset=0,1
        // (1,1), (1,2),   ← start=1, offset=0,1
        // (2,2), (2,3*)   ← start=2, offset=0,1 (*out of bounds → padding)
        let max_width = 2;
        let num_words = 3;

        let mut span_idx: Vec<i64> = Vec::new();
        let mut span_mask: Vec<bool> = Vec::new();

        for start in 0..num_words {
            for offset in 0..max_width {
                let end = start + offset;
                if end < num_words {
                    span_idx.push(start as i64);
                    span_idx.push(end as i64);
                    span_mask.push(true);
                } else {
                    span_idx.push(0);
                    span_idx.push(0);
                    span_mask.push(false);
                }
            }
        }

        assert_eq!(span_mask.len(), 6); // 3 × 2 = 6
        assert_eq!(
            span_idx,
            vec![
                0, 0, 0, 1, // start=0
                1, 1, 1, 2, // start=1
                2, 2, 0, 0, // start=2 (second is padding)
            ]
        );
        assert_eq!(span_mask, vec![true, true, true, true, true, false]);
    }

    #[test]
    fn test_span_index_to_words() {
        // Simulate max_width = 3, num_words = 4
        // idx=0 → (0, 0+0) = (0,0)
        // idx=1 → (0, 0+1) = (0,1)
        // idx=2 → (0, 0+2) = (0,2)
        // idx=3 → (1, 1+0) = (1,1)
        // idx=4 → (1, 1+1) = (1,2)
        // idx=5 → (1, 1+2) = (1,3)
        let max_width = 3;

        let check = |idx: usize, expected: (usize, usize)| {
            let start = idx / max_width;
            let offset = idx % max_width;
            assert_eq!((start, start + offset), expected, "idx={}", idx);
        };

        check(0, (0, 0));
        check(1, (0, 1));
        check(2, (0, 2));
        check(3, (1, 1));
        check(4, (1, 2));
        check(5, (1, 3));
    }

    #[test]
    fn test_missing_model_returns_error() {
        let result = NerEngine::load("/nonexistent/path", 0.5, 12);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("not found"),
            "Error should mention 'not found': {}",
            err
        );
    }

    #[test]
    fn custom_tokenizer_path_is_honored_before_runtime_initialization() {
        // [NER-RUNTIME-INDEPENDENCE 2026-07-30 by Codex] This regression test
        // remains model-free: model presence advances validation to the
        // explicit tokenizer path, which must fail before ORT initialization.
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join(MODEL_FILENAME), b"test-model").unwrap();
        let tokenizer_path = directory.path().join("custom").join("tokenizer.json");

        let error =
            NerEngine::load_with_tokenizer(directory.path(), &tokenizer_path, 0.5, 12).unwrap_err();

        assert!(error.contains(&tokenizer_path.display().to_string()));
        assert!(error.contains("not found"));
    }

    /// Standalone word_split for testing without NerEngine instance.
    fn word_split_standalone(text: &str) -> Vec<WordSpan> {
        split_words_bounded(text, usize::MAX, usize::MAX)
    }

    // ── Integration tests requiring model files ──

    /// Resolve model directory from env or default.
    fn ner_model_dir() -> String {
        std::env::var("MEMCHAIN_NER_MODEL_PATH").unwrap_or_else(|_| "models/gliner".to_string())
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

        let entities = engine
            .detect_entities(
                "auth module uses JWT for authentication",
                &["module", "technology"],
            )
            .unwrap();

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
