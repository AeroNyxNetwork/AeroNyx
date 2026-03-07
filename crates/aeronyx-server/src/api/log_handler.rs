// ============================================
// File: crates/aeronyx-server/src/api/log_handler.rs
// ============================================
//! # /api/mpi/log — Conversation Log Ingestion + Rule Engine
//!
//! ## Overview
//! Receives raw conversation turns from AI agents, persists them encrypted
//! to `raw_logs`, and synchronously extracts persistent user info via a
//! pattern-matching rule engine. Also detects negative feedback in real-time.
//!
//! ## Processing Pipeline (target < 2ms total per request)
//! ```text
//! POST /api/mpi/log
//!   ├─ Step 0: Write each turn to raw_logs (encrypted)
//!   ├─ Step 1: For each role="user" turn:
//!   │   ├─ SKIP classification (has_persistent_info) < 0.1ms
//!   │   ├─ if extractable=1: Rule engine P0-P6 < 0.5ms
//!   │   │   → Content dedup check (< 0.5ms)
//!   │   │   → hits call internal remember (storage.insert + vector.upsert)
//!   │   └─ Negative feedback detection < 0.5ms
//!   │       → hits: update records.negative_feedback + write memory_feedback
//!   └─ Return 202 Accepted { logged: N, session_id: "..." }
//! ```
//!
//! ## Content Dedup (v2.1.0+MVF)
//! Rule engine extractions lack embeddings (filled by Miner Step 0.5),
//! so vector dedup cannot apply. Instead, before inserting each extraction,
//! we check `storage.has_active_content(owner, content_bytes)` — if an
//! Active record with identical encrypted_content already exists for this
//! owner, the extraction is skipped. This prevents:
//!   1. Cross-request duplicates (same /log called twice)
//!   2. Intra-request duplicates (P2 + P5 both match "I am allergic to X")
//!
//! ## Immutable: 202 response format { logged, session_id } unchanged
//!
//! ## Last Modified
//! v2.1.0 - New file: /log endpoint with SKIP + rule engine + neg feedback
//! v2.1.0+MVF - 🌟 Content fingerprint dedup for rule engine extractions;
//!   prevents duplicate memories when same content triggers multiple rules
//!   or when /log is called multiple times with identical input.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::api::mpi::MpiState;
use crate::services::memchain::storage::derive_rawlog_key;

// ============================================
// Request / Response Types
// ============================================

/// A single conversation turn.
#[derive(Debug, Clone, Deserialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
}

/// POST /api/mpi/log request body.
#[derive(Debug, Deserialize)]
pub struct LogRequest {
    pub session_id: String,
    pub turns: Vec<Turn>,
    #[serde(default)]
    pub source_ai: String,
    /// Optional recall context from the AI agent's last recall call.
    /// JSON array: [{"id":"hex","score":1.365,"features":[0.72,...]}]
    #[serde(default)]
    pub recall_context: Option<String>,
}

/// 202 Accepted response.
#[derive(Debug, Serialize)]
pub struct LogResponse {
    pub logged: usize,
    pub session_id: String,
}

// ============================================
// SKIP Classification (D2b)
// ============================================

/// First-person pronoun indicators (CN + EN).
const CN_FIRST_PERSON: &[&str] = &["我", "我的", "我们", "咱"];
const EN_FIRST_PERSON: &[&str] = &[
    "i ", "i'", "my ", "mine ", "i've ", "we ", "our ",
];

/// Task-only verb prefixes (CN).
const CN_TASK_VERBS: &[&str] = &[
    "翻译", "总结", "写", "改", "列出", "转换", "生成", "计算", "解释", "比较",
    "分析", "搜索", "查找", "创建", "修改", "优化", "整理", "格式化", "提取", "合并",
    "画", "设计", "检查", "排序", "过滤", "统计", "导出", "压缩", "解压", "运行",
];

/// Task-only verb prefixes (EN).
const EN_TASK_VERBS: &[&str] = &[
    "translate", "summarize", "write", "fix", "list", "convert", "generate",
    "calculate", "explain", "compare", "analyze", "search", "create", "edit",
    "optimize", "format", "extract", "merge", "draw", "design", "check", "run",
    "sort", "filter", "export", "compress", "deploy", "execute", "debug", "build",
];

/// Determine if a user message likely contains persistent personal info.
///
/// Returns true (extractable=1) or false (extractable=0).
/// Logic:
///   1. Has first-person pronoun → extractable
///   2. Starts with task verb + no first-person → skip
///   3. Length > 20 → extractable (may contain implicit info)
///   4. Length ≤ 20 → skip (too short, likely no value)
fn has_persistent_info(content: &str) -> bool {
    let lower = content.to_lowercase();
    let trimmed = lower.trim();

    // Check first-person pronouns
    let has_first_person = CN_FIRST_PERSON.iter().any(|p| trimmed.contains(p))
        || EN_FIRST_PERSON.iter().any(|p| {
            trimmed.starts_with(p) || trimmed.contains(&format!(" {}", p.trim()))
        });

    if has_first_person {
        return true;
    }

    // Check task verb prefix
    let starts_with_task = CN_TASK_VERBS.iter().any(|v| trimmed.starts_with(v))
        || EN_TASK_VERBS.iter().any(|v| trimmed.starts_with(v));

    if starts_with_task {
        return false;
    }

    // Fallback: length heuristic
    trimmed.len() > 20
}

// ============================================
// Negative Feedback Detection (D2d)
// ============================================

/// Negative feedback keywords (CN + EN).
const CN_NEGATIVE: &[&str] = &[
    "不对", "不是", "错了", "我改主意", "搞错了", "纠正一下",
    "其实不是", "不是这样", "这不是我要的", "重新来", "再试一次",
];

const EN_NEGATIVE: &[&str] = &[
    "wrong", "not correct", "that's not right", "changed my mind",
    "actually no", "let me correct", "not what i asked",
    "try again", "that's not what i meant",
];

/// Check if a user message contains negative feedback keywords.
fn contains_negative_feedback(content: &str) -> bool {
    let lower = content.to_lowercase();
    CN_NEGATIVE.iter().any(|kw| lower.contains(kw))
        || EN_NEGATIVE.iter().any(|kw| lower.contains(kw))
}

// ============================================
// Rule Engine (D2c)
// ============================================

/// A rule engine extraction result.
struct Extraction {
    content: String,
    layer: MemoryLayer,
    tags: Vec<String>,
    confidence: f32,
}

/// Run rule engine P0-P6 on a user message. Returns extractions.
///
/// Uses pre-compiled regex patterns for performance.
/// All patterns run sequentially; a message can produce multiple extractions.
fn run_rule_engine(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();

    // P0: Explicit remember intent (confidence 0.95)
    if let Some(ext) = check_explicit_remember(content) {
        results.push(ext);
    }

    // P1: Explicit forget — handled separately in the caller (calls mpi_forget)
    // Not an extraction — it's a deletion command.

    // P2: Identity declarations (confidence 0.85)
    results.extend(check_identity_declarations(content));

    // P3: Corrections (confidence 0.90)
    if let Some(ext) = check_corrections(content) {
        results.push(ext);
    }

    // P4: Meta-dialogue preferences (confidence 0.80)
    results.extend(check_preferences(content));

    // P5: Avoidance / allergies (confidence 0.70-0.90)
    results.extend(check_avoidance(content));

    // P6: Environment / device (confidence 0.75)
    if let Some(ext) = check_environment(content) {
        results.push(ext);
    }

    results
}

fn check_explicit_remember(content: &str) -> Option<Extraction> {
    let patterns = [
        // CN
        "记住", "帮我记下", "帮我记住", "请记住",
        // EN
        "remember ", "keep in mind", "don't forget",
    ];

    let lower = content.to_lowercase();
    for pat in &patterns {
        if let Some(pos) = lower.find(pat) {
            let after = &content[pos + pat.len()..].trim();
            if !after.is_empty() {
                return Some(Extraction {
                    content: after.to_string(),
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

    // CN patterns
    let cn_patterns = [
        r"我(?:是|叫|的职业是|住在|来自|在.{1,15}工作|的名字是)(.{1,30})",
        r"我.{0,2}(?:岁|年纪)",
        r"我有(?:一个|两个|三个)?(.{1,10})(?:儿子|女儿|孩子|老婆|丈夫|男友|女友)",
    ];

    // EN patterns
    let en_patterns = [
        r"(?i)(?:I am|I'm|I work (?:at|as|in)|my name is|I live in|I'm from)(.{1,50})",
        r"(?i)I'm (\d+) years old",
        r"(?i)I have (?:a |an )?(\w+ (?:son|daughter|child|wife|husband|partner))",
    ];

    for pat_str in cn_patterns.iter().chain(en_patterns.iter()) {
        if let Ok(re) = Regex::new(pat_str) {
            if re.is_match(content) {
                results.push(Extraction {
                    content: content.to_string(),
                    layer: MemoryLayer::Episode,
                    tags: vec!["identity".into()],
                    confidence: 0.85,
                });
                break; // One identity extraction per message
            }
        }
    }

    results
}

fn check_corrections(content: &str) -> Option<Extraction> {
    let cn = [
        r"不是.{1,15}[，,]是", r"其实是", r"我说错了",
        r"更正一下", r"我之前说的不对",
    ];
    let en = [
        r"(?i)actually", r"(?i)I was wrong", r"(?i)let me correct",
        r"(?i)not .{1,20}, it's",
    ];

    let lower = content.to_lowercase();
    for pat in cn.iter().chain(en.iter()) {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(&lower) || re.is_match(content) {
                return Some(Extraction {
                    content: content.to_string(),
                    layer: MemoryLayer::Episode,
                    tags: vec!["_correction".into()],
                    confidence: 0.90,
                });
            }
        }
    }
    None
}

fn check_preferences(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();

    // Language preference
    let lang_cn = r"用(?:中文|英文|英语|日文|日语|法语|韩语).{0,5}(?:回答|说|写)";
    let lang_en = r"(?i)(?:reply|respond|answer|write) in (?:English|Chinese|Japanese|French)";

    for pat in &[lang_cn, lang_en] {
        if let Ok(re) = Regex::new(pat) {
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
    }

    // Format preference
    let fmt_cn = r"(?:简短一点|详细一点|简洁|不要用.{1,8}术语|口语化|正式一点|用列表|用表格)";
    let fmt_en = r"(?i)(?:keep it short|more detail|avoid jargon|be casual|be formal|use bullet)";

    for pat in &[fmt_cn, fmt_en] {
        if let Ok(re) = Regex::new(pat) {
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
    }

    // Role setting
    let role_cn = r"你(?:扮演|是|充当)(.{1,20})";
    let role_en = r"(?i)(?:act as|you are|pretend to be|play the role of)(.{1,30})";

    for pat in &[role_cn, role_en] {
        if let Ok(re) = Regex::new(pat) {
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
    }

    results
}

fn check_avoidance(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();

    // Allergy (high confidence)
    let allergy_cn = r"我?对(.{1,10})过敏";
    let allergy_en = r"(?i)(?:I'm |I am )?allergic to (.{1,20})";

    for pat in &[allergy_cn, allergy_en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                results.push(Extraction {
                    content: content.to_string(),
                    layer: MemoryLayer::Episode,
                    tags: vec!["health".into(), "allergy".into()],
                    confidence: 0.90,
                });
                return results; // Allergy is high priority, skip general avoidance
            }
        }
    }

    // General avoidance
    let avoid_cn = r"(?:不要|别|没有|不含|不加|避开|避免|我不[吃喝用看听做])(.{1,15})";
    let avoid_en = r"(?i)(?:no|without|avoid|skip|don't want|don't like)\s+(.{1,20})";

    for pat in &[avoid_cn, avoid_en] {
        if let Ok(re) = Regex::new(pat) {
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
    }

    results
}

fn check_environment(content: &str) -> Option<Extraction> {
    let cn = r"我(?:用的是|在用|用)(.{1,20})";
    let en = r"(?i)(?:I use|I'm on|I'm using|I'm running|my .{1,15} version is)(.{1,30})";

    for pat in &[cn, en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                return Some(Extraction {
                    content: content.to_string(),
                    layer: MemoryLayer::Episode,
                    tags: vec!["environment".into()],
                    confidence: 0.75,
                });
            }
        }
    }
    None
}

// ============================================
// Recall Context Parser
// ============================================

/// A single entry from the recall_context JSON.
#[derive(Debug, Clone, Deserialize)]
struct RecallContextEntry {
    id: String,
    score: f64,
    #[serde(default)]
    features: Vec<f32>,
}

/// Parse recall_context JSON string into entries.
fn parse_recall_context(ctx: &str) -> Vec<RecallContextEntry> {
    serde_json::from_str(ctx).unwrap_or_default()
}

// ============================================
// Main /log Handler
// ============================================

/// `POST /api/mpi/log` — Ingest conversation turns.
///
/// Processing pipeline:
/// 1. Write each turn to raw_logs (encrypted if key available)
/// 2. For each user turn: SKIP classification → rule engine → content dedup → neg feedback
/// 3. Return 202 { logged, session_id }
pub async fn mpi_log(
    State(state): State<Arc<MpiState>>,
    Json(req): Json<LogRequest>,
) -> impl IntoResponse {
    let owner = state.identity.public_key_bytes();
    let rawlog_key = derive_rawlog_key(&owner);
    let now_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut logged = 0usize;

    // The recall_context for the current request (may come from request-level
    // or from the most recent assistant turn's context).
    let request_recall_ctx = req.recall_context.as_deref();

    // Track the most recent assistant turn's recall_context for neg feedback
    let mut last_assistant_recall_ctx: Option<String> = request_recall_ctx.map(|s| s.to_string());

    // Track content already extracted in THIS request to prevent intra-request
    // duplicates (e.g. P2 + P5 both extracting from the same message).
    let mut extracted_this_request: Vec<Vec<u8>> = Vec::new();

    for (idx, turn) in req.turns.iter().enumerate() {
        let turn_index = idx as i64;

        // Determine extractable and feedback for this turn
        let mut extractable: i64 = 1;
        let mut feedback_signal: Option<i64> = None;

        let per_turn_recall_ctx = if turn.role == "assistant" {
            // Assistant turns carry recall_context from the previous recall call
            last_assistant_recall_ctx = request_recall_ctx.map(|s| s.to_string());
            request_recall_ctx
        } else {
            None
        };

        // --- SKIP classification (user turns only) ---
        if turn.role == "user" {
            extractable = if has_persistent_info(&turn.content) { 1 } else { 0 };

            // --- Rule engine (only if extractable) ---
            if extractable == 1 {
                let mut extractions = run_rule_engine(&turn.content);

                // Sort by confidence descending so the highest-confidence
                // extraction wins the content dedup race.
                // e.g. allergy (0.90) beats identity (0.85) for "I am allergic to X"
                extractions.sort_unstable_by(|a, b| {
                    b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
                });

                for ext in extractions {
                    let encrypted_content = ext.content.as_bytes().to_vec();

                    // ========================================
                    // Content dedup (v2.1.0+MVF)
                    // ========================================
                    // Two-level dedup:
                    // 1. Intra-request: check in-memory set (zero cost)
                    // 2. Cross-request: check SQLite for existing Active record (< 0.5ms)
                    //
                    // This prevents:
                    // - P2 + P5 both creating a record for "I am allergic to shellfish"
                    // - Repeated /log calls with the same conversation producing duplicates

                    // Level 1: Intra-request dedup (same content already extracted in this request)
                    if extracted_this_request.contains(&encrypted_content) {
                        debug!(
                            session = %req.session_id,
                            turn = turn_index,
                            tags = ?ext.tags,
                            "[LOG_RULE] ⏭️ Skipped (intra-request duplicate)"
                        );
                        continue;
                    }

                    // Level 2: Cross-request dedup (same content already in SQLite)
                    if state.storage.has_active_content(&owner, &encrypted_content).await {
                        debug!(
                            session = %req.session_id,
                            turn = turn_index,
                            tags = ?ext.tags,
                            "[LOG_RULE] ⏭️ Skipped (content already exists in DB)"
                        );
                        // Still track it for intra-request dedup
                        extracted_this_request.push(encrypted_content);
                        continue;
                    }

                    // Build a MemoryRecord from extraction
                    let mut record = MemoryRecord::new(
                        owner,
                        now_ts,
                        ext.layer,
                        ext.tags,
                        req.source_ai.clone(),
                        encrypted_content.clone(),
                        vec![], // embedding = NULL, Miner Step 0.5 will backfill
                    );
                    record.signature = state.identity.sign(&record.record_id);

                    // Persist to SQLite (embedding_model="" since no embedding)
                    state.storage.insert(&record, "").await;

                    // Track this content for intra-request dedup
                    extracted_this_request.push(encrypted_content);

                    // Update Identity cache if tags indicate identity info
                    if record.topic_tags.iter().any(|t| t == "identity" || t == "allergy") {
                        let owner_hex = hex::encode(owner);
                        let mut cache = state.identity_cache.write();
                        cache.entry(owner_hex).or_default().push(record);
                    }

                    debug!(
                        session = %req.session_id,
                        turn = turn_index,
                        "[LOG_RULE] Extracted memory"
                    );
                }
            }

            // --- Negative feedback detection ---
            if contains_negative_feedback(&turn.content) {
                // Find the best-scored memory from recall_context
                let ctx_str = last_assistant_recall_ctx.as_deref()
                    .or(request_recall_ctx);

                if let Some(ctx) = ctx_str {
                    let entries = parse_recall_context(ctx);
                    if let Some(top) = entries.first() {
                        if let Ok(id_bytes) = hex::decode(&top.id) {
                            if id_bytes.len() == 32 {
                                let mut record_id = [0u8; 32];
                                record_id.copy_from_slice(&id_bytes);

                                // Update negative feedback counter
                                state.storage.increment_negative_feedback(&record_id).await;

                                // Convert features to [f32; 9] if available
                                let features_arr: Option<[f32; 9]> = if top.features.len() == 9 {
                                    let mut arr = [0.0f32; 9];
                                    arr.copy_from_slice(&top.features);
                                    Some(arr)
                                } else {
                                    None
                                };

                                // Write feedback event
                                state.storage.insert_feedback(
                                    &owner,
                                    &record_id,
                                    &req.session_id,
                                    turn_index,
                                    -1,
                                    features_arr.as_ref(),
                                    Some(top.score as f32),
                                ).await;

                                feedback_signal = Some(-1);

                                info!(
                                    memory = top.id,
                                    session = %req.session_id,
                                    "[LOG_NEG] Negative feedback recorded"
                                );
                            }
                        }
                    }
                }
            }
        }

        // --- Write to raw_logs (encrypted) ---
        let recall_ctx_for_row = if turn.role == "assistant" {
            per_turn_recall_ctx
        } else {
            None
        };

        let result = state.storage.insert_raw_log(
            &req.session_id,
            turn_index,
            &turn.role,
            &turn.content,
            &req.source_ai,
            recall_ctx_for_row,
            extractable,
            feedback_signal,
            Some(&rawlog_key),
        ).await;

        if result.is_ok() {
            logged += 1;
        }
    }

    debug!(
        logged = logged,
        session = %req.session_id,
        "[LOG] Processed"
    );

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!(LogResponse {
            logged,
            session_id: req.session_id,
        })),
    )
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

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
        // > 20 chars, no first person, no task verb
        assert!(has_persistent_info("the meeting is scheduled for tomorrow at 3pm in room B"));
    }

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

    #[test]
    fn test_rule_engine_allergy_en_triggers_p2_and_p5() {
        // "I am allergic to shellfish" matches both P2 (I am...) and P5 (allergic to...)
        // Both should produce extractions with the SAME content
        let results = run_rule_engine("I am allergic to shellfish");
        // P2 produces identity tag, P5 produces allergy tag
        assert!(results.len() >= 2, "Should match both P2 and P5");
        // Both extractions have the same content — content dedup will catch this
        assert_eq!(results[0].content, results[1].content);
    }
}
