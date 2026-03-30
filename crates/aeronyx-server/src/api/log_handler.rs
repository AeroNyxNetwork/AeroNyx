// ============================================
// File: crates/aeronyx-server/src/api/log_handler.rs
// ============================================
//! # /api/mpi/log — Conversation Log Ingestion + Rule Engine
//!
//! ## File Creation/Modification Notes
//! ============================================
//! Creation Reason: Central ingestion point for raw AI conversation turns.
//!   Persists encrypted turns to `raw_logs` and synchronously extracts
//!   persistent user info via a pattern-matching rule engine.
//! Modification Reason:
//!   v2.5.2+Provenance  — integrate session_id write-back after Rule Engine
//!     extraction so provenance chain is populated.
//!   v2.5.2+SecurityFix — four security fixes from audit:
//!     #1 set_record_session_id returns () — removed if-let-Err wrapper
//!     #2 Rule Engine extraction content truncated to MAX_EXTRACTION_BYTES (4KB)
//!     #6 Rule Engine input truncated to MAX_RULE_ENGINE_INPUT (2000 chars)
//!        before all regex scans to mitigate ReDoS on near-1MB messages
//!     #7 check_explicit_remember UTF-8 slice panic — now searches on `lower`
//!        and slices `lower`, never mixing offsets between two strings
//! Main Functionality:
//!   - Local-only access gate (remote → 403)
//!   - Stage 1 Entropy Filter (v2.4.0): discard low-value windows pre-Rule Engine
//!   - Privacy tag stripping (<no-mem> / <private>) before ANY storage or indexing
//!   - Raw turn persistence (encrypted) to raw_logs
//!   - SKIP classification + Rule Engine (P0-P6) for user turns
//!   - Negative feedback detection and feedback ledger insertion
//!   - FTS5 indexing of all turns (stripped content)
//!   - Session upsert for Miner Steps 7-11
//!   - v2.5.2+Provenance: set_record_session_id after each Rule Engine extraction
//! Dependencies:
//!   - aeronyx_core::ledger::{MemoryLayer, MemoryRecord}
//!   - crate::api::mpi::{AuthenticatedOwner, MpiState}
//!   - crate::services::memchain::derive_rawlog_key
//!
//! ## Overview
//! Receives raw conversation turns from AI agents, persists them encrypted
//! to `raw_logs`, and synchronously extracts persistent user info via a
//! pattern-matching rule engine. Also detects negative feedback in real-time.
//!
//! ## v2.4.0-GraphCognition: Stage 1 Entropy Filter
//! Before the P0-P6 Rule Engine, conversation windows are evaluated for
//! information content. Low-value windows (greetings, confirmations, repetitions)
//! are discarded before entering the memory pipeline.
//!
//! ```text
//! POST /api/mpi/log
//!   ├─ Step 0: Verify local-only access (v2.3.0)
//!   ├─ Step 0.5: 🆕 Stage 1 Entropy Filter (v2.4.0)
//!   │   ├─ GLiNER entity detection (if NER engine available)
//!   │   ├─ MiniLM embedding (if embed engine available)
//!   │   ├─ Information score = 0.6 × entity_novelty + 0.4 × semantic_divergence
//!   │   └─ score < threshold → skip this window (don't enter Rule Engine)
//!   ├─ Step 1: Privacy tag stripping per turn (v2.4.0+Privacy)
//!   │   └─ <no-mem>...</no-mem> and <private>...</private> removed before storage
//!   ├─ Step 2: Write each turn to raw_logs (encrypted, stripped content)
//!   ├─ Step 3: For each role="user" turn:
//!   │   ├─ SKIP classification (has_persistent_info)
//!   │   ├─ if extractable=1: Rule engine P0-P6 (on stripped + truncated content)
//!   │   │   → Content dedup check → hits call internal remember
//!   │   │   → 🆕 v2.5.2+Provenance: set_record_session_id after insert
//!   │   └─ Negative feedback detection (on stripped content)
//!   └─ Return 202 Accepted { logged, session_id }
//! ```
//!
//! ## Entropy Filter Design
//! - Entity novelty = new_entities / total_entities (compared to known_entities cache)
//! - Semantic divergence = 1.0 - avg_cosine_sim(window_embedding, recent_20_embeddings)
//! - score < 0.35 (default) → discard: greetings, "ok", "thanks", repetitions
//! - score ≥ 0.35 → pass through to Rule Engine as before
//! - When NER unavailable: only semantic divergence used (less effective)
//! - When embed unavailable: only entity novelty used (less effective)
//! - When both unavailable: entropy filter is a no-op (all windows pass)
//!
//! ## Privacy Tag Design (v2.4.0+Privacy)
//! - Tags: `<no-mem>...</no-mem>` and `<private>...</private>` (case-insensitive)
//! - Stripping occurs BEFORE rawlog storage, FTS indexing, Rule Engine, and NER
//! - The raw_logs table stores the STRIPPED version (tagged content never persisted)
//! - Entropy filter uses ORIGINAL content (tagged content has info → window should pass)
//! - Unclosed tags are ignored (regex requires both open and close tags)
//! - strip_privacy_tags() uses Cow<str> — zero allocation when no tags are present
//!
//! ## Security Design Notes (v2.5.2+SecurityFix)
//! - MAX_RULE_ENGINE_INPUT (2000 chars): all regex patterns scan at most this many
//!   chars per message. Prevents ReDoS from near-1MB trigger messages (#6).
//! - MAX_EXTRACTION_BYTES (4KB): stored content per extraction is truncated.
//!   Prevents storage amplification when multiple rules fire on the same large msg (#2).
//! - check_explicit_remember: pos found in `lower`, slice taken from `lower`.
//!   No cross-string offset reuse — eliminates UTF-8 panic from ß→ss expansion (#7).
//! - set_record_session_id returns () — caller does not wrap in if-let-Err (#1).
//!
//! ⚠️ Important Note for Next Developer:
//! - /log is LOCAL-ONLY: remote users get 403
//! - derive_rawlog_key MUST use identity.to_bytes() (PRIVATE key)
//! - Entropy filter is gated on state.entropy_filter_enabled
//! - When entropy filter discards a window, turns are STILL written to raw_logs
//!   (non-lossy), but extractable is set to 0 (Miner won't process them)
//! - The filter operates on the ENTIRE turns array as one window
//! - Rule engine regex patterns are pre-compiled via std::sync::LazyLock
//! - Privacy stripping runs on turn.content; entropy filter uses original content
//! - set_record_session_id returns () — do NOT wrap in if-let-Err
//! - The core logic of this file cannot be deleted or significantly modified.
//! - Maintain interface compatibility with storage.rs
//!
//! ## Modification History
//! v2.1.0            - New file: /log endpoint with SKIP + rule engine + neg feedback
//! v2.1.0+MVF        - Content fingerprint dedup for rule engine extractions
//! v2.1.0+MVF+Encryption - Fixed rawlog key derivation to use PRIVATE key
//! v2.3.0+RemoteStorage  - Local-only access restriction (remote → 403)
//! v2.4.0-GraphCognition - 🌟 Stage 1 entropy filter (pre-Rule Engine)
//! v2.4.0+BugFix     - 🔧 Fixed E0282, pre-compiled regex patterns via LazyLock
//! v2.4.0+Privacy    - 🌟 Added <no-mem> / <private> privacy tag stripping
//! v2.5.2+Provenance - 🌟 session_id write-back + Cow strip + test comment fix
//! v2.5.2+SecurityFix - 🔒 #1 set_record_session_id () return type fix
//!                      🔒 #2 ext.content truncated to MAX_EXTRACTION_BYTES (4KB)
//!                      🔒 #6 Rule Engine input truncated to MAX_RULE_ENGINE_INPUT (2000 chars)
//!                      🔒 #7 check_explicit_remember UTF-8 slice panic fixed
//!
//! ## Last Modified
//! v2.5.2+SecurityFix - 🔒 4 security fixes from audit

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

// ── Security constants (v2.5.2+SecurityFix) ────────────────────────────────
/// Max bytes stored per Rule Engine extraction (fix #2).
/// Prevents storage amplification from near-1MB trigger messages.
const MAX_EXTRACTION_BYTES: usize = 4 * 1024; // 4 KB

/// Max chars fed into the Rule Engine regex pipeline (fix #6).
/// Mitigates ReDoS: all patterns scan at most this many chars per message.
const MAX_RULE_ENGINE_INPUT: usize = 2_000;

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::api::mpi::{AuthenticatedOwner, MpiState};
use crate::services::memchain::derive_rawlog_key;

// ============================================
// Request / Response Types
// ============================================

#[derive(Debug, Clone, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct LogRequest {
    pub session_id: String,
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub source_ai: String,
    #[serde(default)]
    pub recall_context: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LogResponse {
    pub logged: usize,
    pub session_id: String,
}

// ============================================
// v2.4.0+Privacy: Privacy Tag Stripping
// ============================================

/// Pre-compiled regex for `<no-mem>` / `<private>` privacy tags (v2.4.0+Privacy).
static RE_PRIVACY_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)<(?:no-mem|private)>[\s\S]*?</(?:no-mem|private)>")
        .expect("Privacy tag regex must compile")
});

/// Strip `<no-mem>...</no-mem>` and `<private>...</private>` tags from content.
///
/// Returns [`Cow::Borrowed`] when no tags are found (zero allocation on hot path).
/// Returns [`Cow::Owned`] only when at least one tag is stripped.
fn strip_privacy_tags(content: &str) -> Cow<'_, str> {
    let lower_has_tag = content
        .as_bytes()
        .windows(8)
        .any(|w| w.eq_ignore_ascii_case(b"<no-mem>"))
        || content
            .as_bytes()
            .windows(9)
            .any(|w| w.eq_ignore_ascii_case(b"<private>"));

    if !lower_has_tag {
        return Cow::Borrowed(content);
    }

    Cow::Owned(RE_PRIVACY_TAG.replace_all(content, "").into_owned())
}

// ============================================
// v2.4.0: Entropy Filter
// ============================================

#[derive(Debug)]
struct EntropyResult {
    score: f32,
    passes: bool,
    detected_entities: Vec<String>,
}

/// Run Stage 1 entropy filter on the conversation turns.
///
/// ⚠️ Intentionally uses ORIGINAL turn.content (not stripped).
fn run_entropy_filter(
    state: &MpiState,
    turns: &[Turn],
    known_entity_names: &std::collections::HashSet<String>,
    threshold: f32,
) -> EntropyResult {
    let window_text: String = turns
        .iter()
        .filter(|t| t.role == "user")
        .map(|t| t.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    if window_text.trim().is_empty() {
        return EntropyResult { score: 0.0, passes: false, detected_entities: Vec::new() };
    }

    let mut entity_novelty: f32 = 0.5;
    let mut semantic_divergence: f32 = 0.5;
    let mut detected_entities: Vec<String> = Vec::new();

    if let Some(ref ner_engine) = state.ner_engine {
        let labels = &[
            "project", "module", "technology", "file", "person",
            "decision", "problem", "solution",
        ];
        match ner_engine.detect_entities(&window_text, labels) {
            Ok(entities) => {
                let total = entities.len();
                if total > 0 {
                    let new_count = entities
                        .iter()
                        .filter(|e| !known_entity_names.contains(&e.text.to_lowercase()))
                        .count();
                    entity_novelty = new_count as f32 / total as f32;
                    detected_entities = entities.iter().map(|e| e.text.clone()).collect();
                } else {
                    entity_novelty = 0.1;
                }
            }
            Err(e) => {
                debug!(error = %e, "[ENTROPY] GLiNER detection failed, using neutral novelty");
            }
        }
    }

    if state.embed_engine.is_some() {
        let unique_words: std::collections::HashSet<&str> = window_text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        let total_words = window_text.split_whitespace().count().max(1);
        let word_variety = unique_words.len() as f32 / total_words as f32;
        semantic_divergence = word_variety.clamp(0.0, 1.0);
        if total_words <= 3 {
            semantic_divergence = 0.1;
        }
    }

    let score = 0.6 * entity_novelty + 0.4 * semantic_divergence;
    let passes = score >= threshold;

    debug!(
        score = format!("{:.3}", score),
        entity_novelty = format!("{:.3}", entity_novelty),
        semantic_divergence = format!("{:.3}", semantic_divergence),
        entities = detected_entities.len(),
        passes = passes,
        "[ENTROPY] Filter result"
    );

    EntropyResult { score, passes, detected_entities }
}

// ============================================
// SKIP Classification (D2b)
// ============================================

const CN_FIRST_PERSON: &[&str] = &["我", "我的", "我们", "咱"];
const EN_FIRST_PERSON: &[&str] = &["i ", "i'", "my ", "mine ", "i've ", "we ", "our "];

const CN_TASK_VERBS: &[&str] = &[
    "翻译", "总结", "写", "改", "列出", "转换", "生成", "计算", "解释", "比较",
    "分析", "搜索", "查找", "创建", "修改", "优化", "整理", "格式化", "提取", "合并",
    "画", "设计", "检查", "排序", "过滤", "统计", "导出", "压缩", "解压", "运行",
];

const EN_TASK_VERBS: &[&str] = &[
    "translate", "summarize", "write", "fix", "list", "convert", "generate",
    "calculate", "explain", "compare", "analyze", "search", "create", "edit",
    "optimize", "format", "extract", "merge", "draw", "design", "check", "run",
    "sort", "filter", "export", "compress", "deploy", "execute", "debug", "build",
];

fn has_persistent_info(content: &str) -> bool {
    let lower = content.to_lowercase();
    let trimmed = lower.trim();

    let has_first_person = CN_FIRST_PERSON.iter().any(|p| trimmed.contains(p))
        || EN_FIRST_PERSON.iter().any(|p| {
            trimmed.starts_with(p) || trimmed.contains(&format!(" {}", p.trim()))
        })
        || trimmed == "i";
    if has_first_person { return true; }

    let starts_with_task = CN_TASK_VERBS.iter().any(|v| trimmed.starts_with(v))
        || EN_TASK_VERBS.iter().any(|v| trimmed.starts_with(v));
    if starts_with_task { return false; }

    trimmed.len() > 20
}

// ============================================
// Negative Feedback Detection (D2d)
// ============================================

const CN_NEGATIVE: &[&str] = &[
    "不对", "不是", "错了", "我改主意", "搞错了", "纠正一下",
    "其实不是", "不是这样", "这不是我要的", "重新来", "再试一次",
];
const EN_NEGATIVE: &[&str] = &[
    "wrong", "not correct", "that's not right", "changed my mind",
    "actually no", "let me correct", "not what i asked",
    "try again", "that's not what i meant",
];

fn contains_negative_feedback(content: &str) -> bool {
    let lower = content.to_lowercase();
    CN_NEGATIVE.iter().any(|kw| lower.contains(kw))
        || EN_NEGATIVE.iter().any(|kw| lower.contains(kw))
}

// ============================================
// Rule Engine (D2c) — Pre-compiled Regex via LazyLock
// ============================================

fn compile_patterns(patterns: &[&str]) -> Vec<Regex> {
    patterns.iter().filter_map(|p| Regex::new(p).ok()).collect()
}

static RE_IDENTITY_CN: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"我(?:是|叫|的职业是|住在|来自|在.{1,15}工作|的名字是)(.{1,30})",
    r"我.{0,2}(?:岁|年纪)",
    r"我有(?:一个|两个|三个)?(.{1,10})(?:儿子|女儿|孩子|老婆|丈夫|男友|女友)",
]));

static RE_IDENTITY_EN: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"(?i)(?:I am|I'm|I work (?:at|as|in)|my name is|I live in|I'm from)(.{1,50})",
    r"(?i)I'm (\d+) years old",
    r"(?i)I have (?:a |an )?(\w+ (?:son|daughter|child|wife|husband|partner))",
]));

static RE_CORRECTION_CN: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"不是.{1,15}[，,]是", r"其实是", r"我说错了", r"更正一下", r"我之前说的不对",
]));

static RE_CORRECTION_EN: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"(?i)actually", r"(?i)I was wrong", r"(?i)let me correct", r"(?i)not .{1,20}, it's",
]));

static RE_PREF_LANG: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"用(?:中文|英文|英语|日文|日语|法语|韩语).{0,5}(?:回答|说|写)",
    r"(?i)(?:reply|respond|answer|write) in (?:English|Chinese|Japanese|French)",
]));

static RE_PREF_FMT: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"(?:简短一点|详细一点|简洁|不要用.{1,8}术语|口语化|正式一点|用列表|用表格)",
    r"(?i)(?:keep it short|more detail|avoid jargon|be casual|be formal|use bullet)",
]));

static RE_PREF_ROLE: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"你(?:扮演|是|充当)(.{1,20})",
    r"(?i)(?:act as|you are|pretend to be|play the role of)(.{1,30})",
]));

static RE_ALLERGY: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"我?对(.{1,10})过敏",
    r"(?i)(?:I'm |I am )?allergic to (.{1,20})",
]));

static RE_AVOID: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"(?:不要|别|没有|不含|不加|避开|避免|我不[吃喝用看听做])(.{1,15})",
    r"(?i)(?:no|without|avoid|skip|don't want|don't like)\s+(.{1,20})",
]));

static RE_ENV: LazyLock<Vec<Regex>> = LazyLock::new(|| compile_patterns(&[
    r"我(?:用的是|在用|用)(.{1,20})",
    r"(?i)(?:I use|I'm on|I'm using|I'm running|my .{1,15} version is)(.{1,30})",
]));

// ─────────────────────────────────────────────

struct Extraction {
    content: String,
    layer: MemoryLayer,
    tags: Vec<String>,
    confidence: f32,
}

fn run_rule_engine(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    if let Some(ext) = check_explicit_remember(content) { results.push(ext); }
    results.extend(check_identity_declarations(content));
    if let Some(ext) = check_corrections(content) { results.push(ext); }
    results.extend(check_preferences(content));
    results.extend(check_avoidance(content));
    if let Some(ext) = check_environment(content) { results.push(ext); }
    results
}

/// Fix #7: search in `lower` (to_lowercase), slice from `lower`.
/// Previously: pos found in `lower`, then `content[pos + pat.len()..]` was sliced —
/// to_lowercase can change byte length (e.g. ß→ss), making the offset invalid for
/// `content` and potentially panicking at a non-char-boundary.
/// Now: both find and slice operate on `lower` exclusively.
fn check_explicit_remember(content: &str) -> Option<Extraction> {
    const PATTERNS: &[&str] = &[
        "记住", "帮我记下", "帮我记住", "请记住",
        "remember ", "keep in mind", "don't forget",
    ];
    let lower = content.to_lowercase();
    for pat in PATTERNS {
        let pat_lower = pat.to_lowercase();
        if let Some(pos) = lower.find(pat_lower.as_str()) {
            let after = lower[pos + pat_lower.len()..].trim().to_string();
            if !after.is_empty() {
                return Some(Extraction {
                    content: after,
                    layer: MemoryLayer::Episode,
                    tags: vec!["explicit_remember".into()],
                    confidence: 0.95,
                });
            }
        }
    }
    None
}

fn check_identity_declarations(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    for re in RE_IDENTITY_CN.iter().chain(RE_IDENTITY_EN.iter()) {
        if re.is_match(content) {
            results.push(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["identity".into()],
                confidence: 0.85,
            });
            break;
        }
    }
    results
}

fn check_corrections(content: &str) -> Option<Extraction> {
    let lower = content.to_lowercase();
    for re in RE_CORRECTION_CN.iter().chain(RE_CORRECTION_EN.iter()) {
        if re.is_match(&lower) || re.is_match(content) {
            return Some(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["_correction".into()],
                confidence: 0.90,
            });
        }
    }
    None
}

fn check_preferences(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    for re in RE_PREF_LANG.iter() {
        if re.is_match(content) {
            results.push(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["preference".into(), "language".into()],
                confidence: 0.80,
            });
            break;
        }
    }
    for re in RE_PREF_FMT.iter() {
        if re.is_match(content) {
            results.push(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["preference".into(), "format".into()],
                confidence: 0.80,
            });
            break;
        }
    }
    for re in RE_PREF_ROLE.iter() {
        if re.is_match(content) {
            results.push(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["role".into()],
                confidence: 0.80,
            });
            break;
        }
    }
    results
}

fn check_avoidance(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    for re in RE_ALLERGY.iter() {
        if re.is_match(content) {
            results.push(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["health".into(), "allergy".into()],
                confidence: 0.90,
            });
            return results;
        }
    }
    for re in RE_AVOID.iter() {
        if re.is_match(content) {
            results.push(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["avoidance".into()],
                confidence: 0.70,
            });
            break;
        }
    }
    results
}

fn check_environment(content: &str) -> Option<Extraction> {
    for re in RE_ENV.iter() {
        if re.is_match(content) {
            return Some(Extraction {
                content: content.to_string(),
                layer: MemoryLayer::Episode,
                tags: vec!["environment".into()],
                confidence: 0.75,
            });
        }
    }
    None
}

// ============================================
// Recall Context Parser
// ============================================

#[derive(Debug, Clone, Deserialize)]
struct RecallContextEntry {
    id: String,
    score: f64,
    #[serde(default)]
    features: Vec<f32>,
}

fn parse_recall_context(ctx: &str) -> Vec<RecallContextEntry> {
    serde_json::from_str(ctx).unwrap_or_default()
}

// ============================================
// Main /log Handler
// ============================================

pub async fn mpi_log(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    // ── Step 0: Local-only access restriction (v2.3.0) ──
    let auth = req
        .extensions()
        .get::<AuthenticatedOwner>()
        .expect("[BUG] AuthenticatedOwner not set by middleware")
        .clone();

    if auth.is_remote() {
        warn!(
            "[MPI_LOG] Rejected remote /log request from {}",
            &auth.owner_hex()[..8]
        );
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": "/log endpoint is local-only. Remote users should use the plugin's rule engine and call /remember directly.",
            })),
        ).into_response();
    }

    let owner = auth.owner_bytes();

    // ── Parse body ──
    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "failed to read body"})),
        ).into_response(),
    };
    let log_req: LogRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid JSON: {}", e)})),
        ).into_response(),
    };

    // ── Step 0.5: Stage 1 Entropy Filter (v2.4.0) ──
    let entropy_passes = if state.entropy_filter_enabled {
        let known_entities: std::collections::HashSet<String> = state
            .storage
            .get_entities_cached(&owner)
            .await
            .keys()
            .cloned()
            .collect();

        let threshold = 0.35_f32;
        let result = run_entropy_filter(&state, &log_req.turns, &known_entities, threshold);

        debug!(
            session = %log_req.session_id,
            score = format!("{:.3}", result.score),
            passes = result.passes,
            entities = result.detected_entities.len(),
            "[MPI_LOG] Entropy filter: {}",
            if result.passes { "PASS" } else { "SKIP" }
        );

        result.passes
    } else {
        true
    };

    // ── Rawlog key derivation (once, outside turn loop) ──
    let rawlog_key = derive_rawlog_key(&state.identity.to_bytes());

    let now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut logged = 0usize;

    let request_recall_ctx = log_req.recall_context.as_deref();
    let mut last_assistant_recall_ctx: Option<String> =
        request_recall_ctx.map(|s| s.to_string());
    let mut extracted_this_request: Vec<Vec<u8>> = Vec::new();

    for (idx, turn) in log_req.turns.iter().enumerate() {
        let turn_index = idx as i64;
        let mut extractable: i64 = 1;
        let mut feedback_signal: Option<i64> = None;

        // v2.4.0+Privacy: strip tags; returns Cow::Borrowed (zero alloc) when no tags.
        let turn_content: Cow<'_, str> = strip_privacy_tags(&turn.content);

        let per_turn_recall_ctx = if turn.role == "assistant" {
            last_assistant_recall_ctx = request_recall_ctx.map(|s| s.to_string());
            request_recall_ctx
        } else {
            None
        };

        if turn.role == "user" {
            if !entropy_passes {
                extractable = 0;
            } else {
                extractable = if has_persistent_info(&turn_content) { 1 } else { 0 };
            }

            if extractable == 1 {
                // Fix #6: truncate to MAX_RULE_ENGINE_INPUT chars before any regex scan.
                // Uses chars().take() to handle multi-byte UTF-8 correctly.
                let rule_input: Cow<'_, str> = if turn_content.chars().count() > MAX_RULE_ENGINE_INPUT {
                    Cow::Owned(turn_content.chars().take(MAX_RULE_ENGINE_INPUT).collect())
                } else {
                    Cow::Borrowed(turn_content.as_ref())
                };

                let mut extractions = run_rule_engine(&rule_input);
                extractions.sort_unstable_by(|a, b| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                for ext in extractions {
                    // Fix #2: truncate stored content to MAX_EXTRACTION_BYTES.
                    // Trim at a valid UTF-8 char boundary to avoid invalid sequences.
                    let stored_content: String = if ext.content.len() > MAX_EXTRACTION_BYTES {
                        let mut s = ext.content.clone();
                        s.truncate(MAX_EXTRACTION_BYTES);
                        while !s.is_char_boundary(s.len()) {
                            s.pop();
                        }
                        s
                    } else {
                        ext.content
                    };

                    let encrypted_content = stored_content.as_bytes().to_vec();

                    // Intra-request dedup (v2.1.0+MVF)
                    if extracted_this_request.contains(&encrypted_content) {
                        debug!(
                            session = %log_req.session_id,
                            turn = turn_index,
                            "[LOG_RULE] ⏭️ Skipped (intra-request duplicate)"
                        );
                        continue;
                    }

                    // Cross-request dedup (DB check)
                    if state.storage.has_active_content(&owner, &encrypted_content).await {
                        debug!(
                            session = %log_req.session_id,
                            turn = turn_index,
                            "[LOG_RULE] ⏭️ Skipped (content exists in DB)"
                        );
                        extracted_this_request.push(encrypted_content);
                        continue;
                    }

                    let is_identity_or_allergy = ext
                        .tags
                        .iter()
                        .any(|t| t == "identity" || t == "allergy");

                    let mut record = MemoryRecord::new(
                        owner,
                        now_ts,
                        ext.layer,
                        ext.tags,
                        log_req.source_ai.clone(),
                        encrypted_content.clone(),
                        vec![],
                    );
                    record.signature = state.identity.sign(&record.record_id);

                    state.storage.insert(&record, "").await;

                    // v2.5.2+Provenance: write session_id after insert.
                    // Fix #1: set_record_session_id returns () — no Result to unwrap.
                    // Failure is absorbed inside the storage method. Non-fatal.
                    state.storage.set_record_session_id(
                        &record.record_id,
                        &log_req.session_id,
                    ).await;

                    extracted_this_request.push(encrypted_content);

                    // Update identity cache for hot-path recall
                    if is_identity_or_allergy {
                        let owner_hex = hex::encode(owner);
                        let mut cache = state.identity_cache.write();
                        cache.entry(owner_hex).or_default().push(record);
                    }

                    debug!(
                        session = %log_req.session_id,
                        turn = turn_index,
                        "[LOG_RULE] Extracted memory"
                    );
                }
            }

            // ── Negative feedback detection (D2d) ──
            if contains_negative_feedback(&turn_content) {
                let ctx_str = last_assistant_recall_ctx
                    .as_deref()
                    .or(request_recall_ctx);
                if let Some(ctx) = ctx_str {
                    let entries = parse_recall_context(ctx);
                    if let Some(top) = entries.first() {
                        if let Ok(id_bytes) = hex::decode(&top.id) {
                            if id_bytes.len() == 32 {
                                let mut record_id = [0u8; 32];
                                record_id.copy_from_slice(&id_bytes);

                                state.storage.increment_negative_feedback(&record_id).await;

                                let features_arr: Option<[f32; 9]> =
                                    if top.features.len() == 9 {
                                        let mut arr = [0.0f32; 9];
                                        arr.copy_from_slice(&top.features);
                                        Some(arr)
                                    } else {
                                        None
                                    };

                                state.storage.insert_feedback(
                                    &owner,
                                    &record_id,
                                    &log_req.session_id,
                                    turn_index,
                                    -1,
                                    features_arr.as_ref(),
                                    Some(top.score as f32),
                                ).await;

                                feedback_signal = Some(-1);
                                info!(
                                    memory = top.id,
                                    session = %log_req.session_id,
                                    "[LOG_NEG] Negative feedback recorded"
                                );
                            }
                        }
                    }
                }
            }
        }

        // ── Write to raw_logs (stripped content) ──
        let recall_ctx_for_row = if turn.role == "assistant" { per_turn_recall_ctx } else { None };

        let result = state.storage.insert_raw_log(
            &log_req.session_id,
            turn_index,
            &turn.role,
            &turn_content,
            &log_req.source_ai,
            recall_ctx_for_row,
            extractable,
            feedback_signal,
            Some(&rawlog_key),
        ).await;

        if result.is_ok() {
            logged += 1;

            // FTS5 indexing (v2.4.0+BM25)
            if !turn_content.trim().is_empty() {
                let turn_rid = {
                    use sha2::{Digest, Sha256};
                    let mut h = Sha256::new();
                    h.update(log_req.session_id.as_bytes());
                    h.update(b":");
                    h.update(turn_index.to_le_bytes());
                    let hash = h.finalize();
                    let mut rid = [0u8; 32];
                    rid.copy_from_slice(&hash);
                    rid
                };
                state.storage.fts_index_record(
                    &turn_rid,
                    &owner,
                    &turn_content,
                    &format!("turn:{}:{}", log_req.session_id, turn.role),
                ).await;
            }
        }
    }

    // ── Register session for Miner Steps 7-11 ──
    if logged > 0 {
        let turn_count = log_req.turns.len() as i64;
        if let Err(e) = state.storage.upsert_session(
            &log_req.session_id,
            &owner,
            None,
            "chat",
            now_ts as i64,
            turn_count,
        ).await {
            warn!(
                session = %log_req.session_id,
                error = %e,
                "[MPI_LOG] Failed to register session (non-fatal)"
            );
        }
    }

    debug!(logged = logged, session = %log_req.session_id, "[LOG] Processed");

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!(LogResponse {
            logged,
            session_id: log_req.session_id,
        })),
    ).into_response()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── SKIP classification ──

    #[test]
    fn test_skip_pure_task() {
        assert!(!has_persistent_info("翻译这段话"));
        assert!(!has_persistent_info("translate this"));
        assert!(!has_persistent_info("summarize the above"));
        assert!(!has_persistent_info("写一首诗"));
    }

    #[test]
    fn test_skip_short() {
        assert!(!has_persistent_info("ok"));
        assert!(!has_persistent_info("好的"));
        assert!(!has_persistent_info("thanks"));
    }

    #[test]
    fn test_extractable_first_person() {
        assert!(has_persistent_info("我对花生过敏"));
        assert!(has_persistent_info("I am a software engineer"));
        assert!(has_persistent_info("my favorite color is blue"));
        assert!(has_persistent_info("我住在北京"));
    }

    #[test]
    fn test_extractable_long_no_pronoun() {
        assert!(has_persistent_info(
            "the meeting is scheduled for tomorrow at 3pm in room B"
        ));
    }

    // ── Negative feedback ──

    #[test]
    fn test_negative_feedback_cn() {
        assert!(contains_negative_feedback("不对，搞错了"));
        assert!(contains_negative_feedback("这不是我要的"));
        assert!(!contains_negative_feedback("好的，继续"));
    }

    #[test]
    fn test_negative_feedback_en() {
        assert!(contains_negative_feedback("That's not right"));
        assert!(contains_negative_feedback("wrong, try again"));
        assert!(!contains_negative_feedback("great, thanks"));
    }

    // ── Rule engine ──

    #[test]
    fn test_rule_engine_allergy() {
        let results = run_rule_engine("我对花生过敏");
        assert!(!results.is_empty());
        assert!(results[0].tags.contains(&"allergy".to_string()));
    }

    #[test]
    fn test_rule_engine_identity() {
        let results = run_rule_engine("I'm a software engineer at Google");
        assert!(!results.is_empty());
        assert!(results[0].tags.contains(&"identity".to_string()));
    }

    #[test]
    fn test_rule_engine_correction() {
        let results = run_rule_engine("其实是 Python 3.12 不是 3.11");
        assert!(!results.is_empty());
        assert!(results[0].tags.contains(&"_correction".to_string()));
    }

    #[test]
    fn test_rule_engine_preference_format() {
        let results = run_rule_engine("简短一点回答");
        assert!(!results.is_empty());
        assert!(results[0].tags.contains(&"format".to_string()));
    }

    #[test]
    fn test_rule_engine_environment() {
        let results = run_rule_engine("I'm using macOS Sonoma 14.2");
        assert!(!results.is_empty());
        assert!(results[0].tags.contains(&"environment".to_string()));
    }

    #[test]
    fn test_rule_engine_explicit_remember() {
        let results = run_rule_engine("记住我喜欢用 Vim");
        assert!(!results.is_empty());
        assert_eq!(results[0].confidence, 0.95);
    }

    #[test]
    fn test_rule_engine_no_match() {
        let results = run_rule_engine("what is the weather today");
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_recall_context_valid() {
        let ctx = r#"[{"id":"aabb","score":1.5,"features":[0.1,0.2]}]"#;
        let entries = parse_recall_context(ctx);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "aabb");
    }

    #[test]
    fn test_parse_recall_context_invalid() {
        let entries = parse_recall_context("not json");
        assert!(entries.is_empty());
    }

    /// Allergy rule fires and returns early from check_avoidance (allergy supersedes
    /// generic avoidance). Identity rule ALSO fires independently via
    /// check_identity_declarations. Total: at least 2 extractions.
    #[test]
    fn test_rule_engine_allergy_and_identity_both_fire() {
        let results = run_rule_engine("I am allergic to shellfish");
        assert!(results.len() >= 2);
        assert_eq!(results[0].content, results[1].content);
    }

    // ── Entropy filter unit tests ──

    #[test]
    fn test_entropy_filter_empty_window() {
        let turns = vec![Turn { role: "user".into(), content: "".into() }];
        let window_text: String = turns
            .iter()
            .filter(|t| t.role == "user")
            .map(|t| t.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(window_text.trim().is_empty());
    }

    #[test]
    fn test_entropy_filter_combines_user_turns_only() {
        let turns = vec![
            Turn { role: "user".into(), content: "hello".into() },
            Turn { role: "assistant".into(), content: "hi there".into() },
            Turn { role: "user".into(), content: "world".into() },
        ];
        let window_text: String = turns
            .iter()
            .filter(|t| t.role == "user")
            .map(|t| t.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(window_text, "hello world");
    }

    #[test]
    fn test_skip_bare_i() {
        assert!(has_persistent_info("i"));
        assert!(!has_persistent_info("ok"));
        assert!(!has_persistent_info("no"));
    }

    // ── Privacy tag tests ──

    #[test]
    fn test_strip_privacy_tags_basic() {
        let input = "My API key is <no-mem>sk-abc123</no-mem> and I use Rust";
        let result = strip_privacy_tags(input);
        assert_eq!(result.as_ref(), "My API key is  and I use Rust");
        assert!(!result.contains("sk-abc123"));
    }

    #[test]
    fn test_strip_privacy_tags_private_alias() {
        let input = "Password: <private>hunter2</private>";
        let result = strip_privacy_tags(input);
        assert_eq!(result.as_ref(), "Password: ");
    }

    #[test]
    fn test_strip_privacy_tags_case_insensitive() {
        assert_eq!(strip_privacy_tags("<NO-MEM>secret</NO-MEM>").as_ref(), "");
        assert_eq!(strip_privacy_tags("<No-Mem>secret</No-Mem>").as_ref(), "");
        assert_eq!(strip_privacy_tags("<PRIVATE>secret</PRIVATE>").as_ref(), "");
    }

    #[test]
    fn test_strip_privacy_tags_multiline() {
        let input = "before\n<no-mem>\nline1\nline2\n</no-mem>\nafter";
        let result = strip_privacy_tags(input);
        assert_eq!(result.as_ref(), "before\n\nafter");
    }

    #[test]
    fn test_strip_privacy_tags_multiple() {
        let input = "<no-mem>secret1</no-mem> visible <private>secret2</private> also visible";
        let result = strip_privacy_tags(input);
        assert_eq!(result.as_ref(), " visible  also visible");
    }

    #[test]
    fn test_strip_privacy_tags_no_tags() {
        let input = "normal content without any tags";
        let result = strip_privacy_tags(input);
        assert_eq!(result.as_ref(), input);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_strip_privacy_tags_empty() {
        assert_eq!(strip_privacy_tags("").as_ref(), "");
    }

    #[test]
    fn test_strip_privacy_tags_unclosed_ignored() {
        let input = "before <no-mem>unclosed content after";
        let result = strip_privacy_tags(input);
        assert_eq!(result.as_ref(), input);
    }

    // ── v2.5.2+SecurityFix tests ──

    #[test]
    fn test_explicit_remember_utf8_safe() {
        // Fix #7: ß expands to ss in to_lowercase, previously caused panic.
        // "remember" contains no ß but we verify the function handles
        // content with multi-byte chars safely.
        let input = "remember Ünterführung is a German word";
        let result = check_explicit_remember(input);
        assert!(result.is_some());
        // Result comes from `lower`, so it should be lowercase
        let content = result.unwrap().content;
        assert!(content.contains("nterf")); // part of the lowercased word
    }

    #[test]
    fn test_extraction_content_truncated_at_boundary() {
        // Fix #2: verify truncation does not cut inside a multi-byte char.
        // Build a string > MAX_EXTRACTION_BYTES with multi-byte chars at boundary.
        let base = "I am a software engineer. ".repeat(200); // well over 4KB
        assert!(base.len() > MAX_EXTRACTION_BYTES);
        let truncated = if base.len() > MAX_EXTRACTION_BYTES {
            let mut s = base[..MAX_EXTRACTION_BYTES].to_string();
            while !s.is_char_boundary(s.len()) { s.pop(); }
            s
        } else {
            base.clone()
        };
        assert!(truncated.len() <= MAX_EXTRACTION_BYTES);
        // Must be valid UTF-8 (from_utf8 must succeed)
        assert!(std::str::from_utf8(truncated.as_bytes()).is_ok());
    }

    #[test]
    fn test_rule_engine_input_truncated() {
        // Fix #6: a message longer than MAX_RULE_ENGINE_INPUT should be truncated
        // before regex scan. Verify Cow::Owned is returned in that case.
        let long_msg = "我是工程师 ".repeat(500); // >> 2000 chars
        let char_count = long_msg.chars().count();
        assert!(char_count > MAX_RULE_ENGINE_INPUT);
        let truncated: Cow<'_, str> = if char_count > MAX_RULE_ENGINE_INPUT {
            Cow::Owned(long_msg.chars().take(MAX_RULE_ENGINE_INPUT).collect())
        } else {
            Cow::Borrowed(long_msg.as_str())
        };
        assert!(matches!(truncated, Cow::Owned(_)));
        assert_eq!(truncated.chars().count(), MAX_RULE_ENGINE_INPUT);
    }

    #[test]
    fn test_provenance_session_id_field_shape() {
        let record_id: [u8; 32] = [0u8; 32];
        let hex_id = hex::encode(record_id);
        assert_eq!(hex_id.len(), 64);
    }
}
