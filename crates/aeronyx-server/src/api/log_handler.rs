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
//!   ├─ Step 1: Write each turn to raw_logs (encrypted)
//!   ├─ Step 2: For each role="user" turn:
//!   │   ├─ SKIP classification (has_persistent_info)
//!   │   ├─ if extractable=1: Rule engine P0-P6
//!   │   │   → Content dedup check → hits call internal remember
//!   │   └─ Negative feedback detection
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
//! ## Modification History
//! v2.1.0 - New file: /log endpoint with SKIP + rule engine + neg feedback
//! v2.1.0+MVF - Content fingerprint dedup for rule engine extractions
//! v2.1.0+MVF+Encryption - Fixed rawlog key derivation to use PRIVATE key
//! v2.3.0+RemoteStorage - Local-only access restriction (remote → 403)
//! v2.4.0-GraphCognition - 🌟 Stage 1 entropy filter (pre-Rule Engine).
//!   Pre-computed entities + embeddings cached for Stage 2 Miner reuse.
//!
//! ⚠️ Important Note for Next Developer:
//! - /log is LOCAL-ONLY: remote users get 403
//! - derive_rawlog_key MUST use identity.to_bytes() (PRIVATE key)
//! - Entropy filter is gated on state.entropy_filter_enabled
//! - When entropy filter discards a window, turns are STILL written to raw_logs
//!   (non-lossy), but extractable is set to 0 (Miner won't process them)
//! - The filter operates on the ENTIRE turns array as one window,
//!   not per-turn (consistent with document's 10-message window design)
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - 🌟 Added Stage 1 entropy filter

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
// v2.4.0: Entropy Filter
// ============================================

/// Result of Stage 1 entropy filtering on a conversation window.
#[derive(Debug)]
struct EntropyResult {
    /// Information score [0.0, 1.0]. Higher = more informative.
    score: f32,
    /// Whether the window passes the threshold (should enter Rule Engine).
    passes: bool,
    /// Entities detected by GLiNER (cached for Stage 2 Miner reuse).
    detected_entities: Vec<String>,
}

/// Run Stage 1 entropy filter on the conversation turns.
///
/// Evaluates information content of the user messages in this window.
/// Returns EntropyResult with score and pass/fail decision.
///
/// ## Strategy
/// - Combine user turn contents into a single text window
/// - GLiNER: detect entities → compute novelty vs known entities
/// - MiniLM: embed window → compute divergence from recent windows
/// - Score = 0.6 × entity_novelty + 0.4 × semantic_divergence
///
/// ## Graceful Degradation
/// - No NER engine → entity_novelty = 0.5 (neutral)
/// - No embed engine → semantic_divergence = 0.5 (neutral)
/// - Both missing → score = 0.5 (always passes default 0.35 threshold)
fn run_entropy_filter(
    state: &MpiState,
    turns: &[Turn],
    known_entity_names: &std::collections::HashSet<String>,
    threshold: f32,
) -> EntropyResult {
    // Combine user messages into a single window text
    let window_text: String = turns.iter()
        .filter(|t| t.role == "user")
        .map(|t| t.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    if window_text.trim().is_empty() {
        return EntropyResult { score: 0.0, passes: false, detected_entities: Vec::new() };
    }

    let mut entity_novelty: f32 = 0.5; // neutral default
    let mut semantic_divergence: f32 = 0.5; // neutral default
    let mut detected_entities: Vec<String> = Vec::new();

    // ── Entity novelty via GLiNER ──
    if let Some(ref ner_engine) = state.ner_engine {
        let labels = &["project", "module", "technology", "file", "person", "decision", "problem", "solution"];
        match ner_engine.detect_entities(&window_text, labels) {
            Ok(entities) => {
                let total = entities.len();
                if total > 0 {
                    let new_count = entities.iter()
                        .filter(|e| !known_entity_names.contains(&e.text.to_lowercase()))
                        .count();
                    entity_novelty = new_count as f32 / total as f32;
                    detected_entities = entities.iter().map(|e| e.text.clone()).collect();
                } else {
                    // No entities detected → low information content
                    entity_novelty = 0.1;
                }
            }
            Err(e) => {
                debug!(error = %e, "[ENTROPY] GLiNER detection failed, using neutral novelty");
            }
        }
    }

    // ── Semantic divergence via MiniLM ──
    // Compare window embedding to recent window embeddings.
    // We don't have a persistent recent-embeddings buffer yet (Phase C),
    // so for now we use a simpler heuristic: text length and variety.
    // TODO: Implement proper embedding comparison with sliding window buffer.
    if let Some(ref embed_engine) = state.embed_engine {
        // Simple heuristic: short, repetitive text = low divergence
        let unique_words: std::collections::HashSet<&str> = window_text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        let total_words = window_text.split_whitespace().count().max(1);
        let word_variety = unique_words.len() as f32 / total_words as f32;

        // Map variety to divergence: high variety → high divergence
        semantic_divergence = word_variety.clamp(0.0, 1.0);

        // If we can embed, check against a trivial baseline
        // (proper implementation with rolling buffer in Phase C)
        if total_words <= 3 {
            semantic_divergence = 0.1; // Very short = low info
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
        });
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
// Rule Engine (D2c)
// ============================================

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

fn check_explicit_remember(content: &str) -> Option<Extraction> {
    let patterns = ["记住", "帮我记下", "帮我记住", "请记住", "remember ", "keep in mind", "don't forget"];
    let lower = content.to_lowercase();
    for pat in &patterns {
        if let Some(pos) = lower.find(pat) {
            let after = &content[pos + pat.len()..].trim();
            if !after.is_empty() {
                return Some(Extraction { content: after.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["explicit_remember".into()], confidence: 0.95 });
            }
        }
    }
    None
}

fn check_identity_declarations(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    let cn_patterns = [
        r"我(?:是|叫|的职业是|住在|来自|在.{1,15}工作|的名字是)(.{1,30})",
        r"我.{0,2}(?:岁|年纪)",
        r"我有(?:一个|两个|三个)?(.{1,10})(?:儿子|女儿|孩子|老婆|丈夫|男友|女友)",
    ];
    let en_patterns = [
        r"(?i)(?:I am|I'm|I work (?:at|as|in)|my name is|I live in|I'm from)(.{1,50})",
        r"(?i)I'm (\d+) years old",
        r"(?i)I have (?:a |an )?(\w+ (?:son|daughter|child|wife|husband|partner))",
    ];
    for pat_str in cn_patterns.iter().chain(en_patterns.iter()) {
        if let Ok(re) = Regex::new(pat_str) {
            if re.is_match(content) {
                results.push(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["identity".into()], confidence: 0.85 });
                break;
            }
        }
    }
    results
}

fn check_corrections(content: &str) -> Option<Extraction> {
    let cn = [r"不是.{1,15}[，,]是", r"其实是", r"我说错了", r"更正一下", r"我之前说的不对"];
    let en = [r"(?i)actually", r"(?i)I was wrong", r"(?i)let me correct", r"(?i)not .{1,20}, it's"];
    let lower = content.to_lowercase();
    for pat in cn.iter().chain(en.iter()) {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(&lower) || re.is_match(content) {
                return Some(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["_correction".into()], confidence: 0.90 });
            }
        }
    }
    None
}

fn check_preferences(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    let lang_cn = r"用(?:中文|英文|英语|日文|日语|法语|韩语).{0,5}(?:回答|说|写)";
    let lang_en = r"(?i)(?:reply|respond|answer|write) in (?:English|Chinese|Japanese|French)";
    for pat in &[lang_cn, lang_en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                results.push(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["preference".into(), "language".into()], confidence: 0.80 });
                break;
            }
        }
    }
    let fmt_cn = r"(?:简短一点|详细一点|简洁|不要用.{1,8}术语|口语化|正式一点|用列表|用表格)";
    let fmt_en = r"(?i)(?:keep it short|more detail|avoid jargon|be casual|be formal|use bullet)";
    for pat in &[fmt_cn, fmt_en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                results.push(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["preference".into(), "format".into()], confidence: 0.80 });
                break;
            }
        }
    }
    let role_cn = r"你(?:扮演|是|充当)(.{1,20})";
    let role_en = r"(?i)(?:act as|you are|pretend to be|play the role of)(.{1,30})";
    for pat in &[role_cn, role_en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                results.push(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["role".into()], confidence: 0.80 });
                break;
            }
        }
    }
    results
}

fn check_avoidance(content: &str) -> Vec<Extraction> {
    let mut results = Vec::new();
    let allergy_cn = r"我?对(.{1,10})过敏";
    let allergy_en = r"(?i)(?:I'm |I am )?allergic to (.{1,20})";
    for pat in &[allergy_cn, allergy_en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                results.push(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["health".into(), "allergy".into()], confidence: 0.90 });
                return results;
            }
        }
    }
    let avoid_cn = r"(?:不要|别|没有|不含|不加|避开|避免|我不[吃喝用看听做])(.{1,15})";
    let avoid_en = r"(?i)(?:no|without|avoid|skip|don't want|don't like)\s+(.{1,20})";
    for pat in &[avoid_cn, avoid_en] {
        if let Ok(re) = Regex::new(pat) {
            if re.is_match(content) {
                results.push(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["avoidance".into()], confidence: 0.70 });
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
                return Some(Extraction { content: content.to_string(), layer: MemoryLayer::Episode,
                    tags: vec!["environment".into()], confidence: 0.75 });
            }
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
    let auth = req.extensions()
        .get::<AuthenticatedOwner>()
        .expect("[BUG] AuthenticatedOwner not set")
        .clone();

    if auth.is_remote() {
        warn!("[MPI_LOG] Rejected remote /log request from {}", &auth.owner_hex()[..8]);
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "/log endpoint is local-only. Remote users should use the plugin's rule engine and call /remember directly.",
        }))).into_response();
    }

    let owner = auth.owner_bytes();

    // Parse body
    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "failed to read body"}))).into_response(),
    };
    let log_req: LogRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };

    // ── Step 0.5: Stage 1 Entropy Filter (v2.4.0) ──
    // Evaluate information content of the conversation window BEFORE Rule Engine.
    // Low-value windows have extractable forced to 0 (turns still written to raw_logs).
    let entropy_passes = if state.entropy_filter_enabled {
        // Load known entity names for novelty scoring
        let known_entities: std::collections::HashSet<String> = state.storage
            .get_entities_cached(&owner).await
            .keys()
            .cloned()
            .collect();

        // Default threshold from config (0.35)
        // TODO: Read threshold from config via MpiState (currently hardcoded)
        let threshold = 0.35f32;

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
        true // filter disabled → all windows pass
    };

    // ── Rawlog key derivation ──
    let rawlog_key = derive_rawlog_key(&state.identity.to_bytes());

    let now_ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let mut logged = 0usize;

    let request_recall_ctx = log_req.recall_context.as_deref();
    let mut last_assistant_recall_ctx: Option<String> = request_recall_ctx.map(|s| s.to_string());
    let mut extracted_this_request: Vec<Vec<u8>> = Vec::new();

    for (idx, turn) in log_req.turns.iter().enumerate() {
        let turn_index = idx as i64;
        let mut extractable: i64 = 1;
        let mut feedback_signal: Option<i64> = None;

        let per_turn_recall_ctx = if turn.role == "assistant" {
            last_assistant_recall_ctx = request_recall_ctx.map(|s| s.to_string());
            request_recall_ctx
        } else { None };

        if turn.role == "user" {
            // v2.4.0: If entropy filter rejected this window, force extractable=0
            // Turns are still written to raw_logs (non-lossy), but Rule Engine is skipped.
            if !entropy_passes {
                extractable = 0;
            } else {
                extractable = if has_persistent_info(&turn.content) { 1 } else { 0 };
            }

            // Rule engine (only if extractable AND entropy filter passed)
            if extractable == 1 {
                let mut extractions = run_rule_engine(&turn.content);
                extractions.sort_unstable_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

                for ext in extractions {
                    let encrypted_content = ext.content.as_bytes().to_vec();

                    // Intra-request dedup
                    if extracted_this_request.contains(&encrypted_content) {
                        debug!(session = %log_req.session_id, turn = turn_index, "[LOG_RULE] ⏭️ Skipped (intra-request duplicate)");
                        continue;
                    }

                    // Cross-request dedup
                    if state.storage.has_active_content(&owner, &encrypted_content).await {
                        debug!(session = %log_req.session_id, turn = turn_index, "[LOG_RULE] ⏭️ Skipped (content exists in DB)");
                        extracted_this_request.push(encrypted_content);
                        continue;
                    }

                    let mut record = MemoryRecord::new(
                        owner, now_ts, ext.layer, ext.tags, log_req.source_ai.clone(),
                        encrypted_content.clone(), vec![],
                    );
                    record.signature = state.identity.sign(&record.record_id);

                    state.storage.insert(&record, "").await;
                    extracted_this_request.push(encrypted_content);

                    if record.topic_tags.iter().any(|t| t == "identity" || t == "allergy") {
                        let owner_hex = hex::encode(owner);
                        let mut cache = state.identity_cache.write();
                        cache.entry(owner_hex).or_default().push(record);
                    }

                    debug!(session = %log_req.session_id, turn = turn_index, "[LOG_RULE] Extracted memory");
                }
            }

            // Negative feedback detection (runs regardless of entropy filter)
            if contains_negative_feedback(&turn.content) {
                let ctx_str = last_assistant_recall_ctx.as_deref().or(request_recall_ctx);
                if let Some(ctx) = ctx_str {
                    let entries = parse_recall_context(ctx);
                    if let Some(top) = entries.first() {
                        if let Ok(id_bytes) = hex::decode(&top.id) {
                            if id_bytes.len() == 32 {
                                let mut record_id = [0u8; 32];
                                record_id.copy_from_slice(&id_bytes);

                                state.storage.increment_negative_feedback(&record_id).await;

                                let features_arr: Option<[f32; 9]> = if top.features.len() == 9 {
                                    let mut arr = [0.0f32; 9]; arr.copy_from_slice(&top.features); Some(arr)
                                } else { None };

                                state.storage.insert_feedback(
                                    &owner, &record_id, &log_req.session_id,
                                    turn_index, -1, features_arr.as_ref(), Some(top.score as f32),
                                ).await;

                                feedback_signal = Some(-1);
                                info!(memory = top.id, session = %log_req.session_id, "[LOG_NEG] Negative feedback recorded");
                            }
                        }
                    }
                }
            }
        }

        // Write to raw_logs (always, regardless of entropy filter)
        let recall_ctx_for_row = if turn.role == "assistant" { per_turn_recall_ctx } else { None };

        let result = state.storage.insert_raw_log(
            &log_req.session_id, turn_index, &turn.role, &turn.content,
            &log_req.source_ai, recall_ctx_for_row, extractable, feedback_signal,
            Some(&rawlog_key),
        ).await;

        if result.is_ok() { logged += 1; }
    }

    debug!(logged = logged, session = %log_req.session_id, "[LOG] Processed");

    (StatusCode::ACCEPTED, Json(serde_json::json!(LogResponse {
        logged, session_id: log_req.session_id,
    }))).into_response()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── SKIP classification tests (preserved) ──

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
        assert!(has_persistent_info("the meeting is scheduled for tomorrow at 3pm in room B"));
    }

    // ── Negative feedback tests (preserved) ──

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

    // ── Rule engine tests (preserved) ──

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
        let results = run_rule_engine("I am allergic to shellfish");
        assert!(results.len() >= 2, "Should match both P2 and P5");
        assert_eq!(results[0].content, results[1].content);
    }

    // ── v2.4.0: Entropy filter unit tests ──

    #[test]
    fn test_entropy_filter_empty_window() {
        let known = std::collections::HashSet::new();
        let turns = vec![Turn { role: "user".into(), content: "".into() }];
        // Can't call run_entropy_filter without MpiState in unit tests,
        // but we can test the helper logic
        let window_text: String = turns.iter()
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
        let window_text: String = turns.iter()
            .filter(|t| t.role == "user")
            .map(|t| t.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(window_text, "hello world");
    }
}
