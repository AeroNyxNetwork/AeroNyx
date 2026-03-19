// ============================================
// File: crates/aeronyx-server/src/services/memchain/task_worker.rs
// ============================================
//! # TaskWorker — Async Cognitive Task Queue Worker
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Polls `cognitive_tasks` for pending tasks, dispatches them to the appropriate
//! LLM provider via `LlmRouter`, and writes results back to their target tables.
//!
//! ## Worker Loop
//! ```text
//! loop:
//!   1. claim_pending_tasks(batch_size) — atomic SELECT + UPDATE to 'processing'
//!   2. For each claimed task (concurrent within batch):
//!      a. parse task_type → CognitiveTaskType
//!      b. build prompt via prompts.rs builder functions
//!      c. route() → LlmProvider::chat()
//!      d. clean_llm_response() — strip <think> chains, common prefixes
//!      e. parse result (plain text or JSON depending on task type)
//!      f. write result to target table (sessions/communities/entities)
//!      g. complete_task() or fail_task()
//!      h. insert_usage_log()
//!   3. Sleep poll_interval_secs if no tasks claimed
//! ```
//!
//! ## Result Parsing per Task Type
//! - `session_title`        → plain text, clean + trim whitespace + quotes
//! - `community_narrative`  → plain text, clean + trim whitespace
//! - `conflict_resolution`  → JSON with <result> tags or markdown fence:
//!                            `{"keep_edge_id": N, "reason": "..."}`
//! - `recall_synthesis`     → JSON: `{"summary": "...", "key_decisions": "..."|null}`
//! - `code_analysis`        → JSON: `{"description": "...", "complexity": "..."}`
//! - `entity_description`   → plain text, clean + trim
//!
//! ## Writeback per Task Type
//! - `session_title`        → `UPDATE sessions SET title = ?`
//! - `community_narrative`  → `storage.upsert_community()` with new summary
//! - `conflict_resolution`  → `storage.invalidate_edge()` for losing edges
//! - `recall_synthesis`     → `storage.update_session_summary()` with natural text
//! - `code_analysis`        → direct SQL UPDATE artifacts SET description = ?
//! - `entity_description`   → direct SQL UPDATE entities SET description = ?
//!
//! ⚠️ Important Note for Next Developer:
//! - `clean_llm_response()` MUST be called before any writeback or JSON parsing.
//!   It strips DeepSeek R1 <think>...</think> chains and common preamble prefixes
//!   that would otherwise corrupt stored titles, summaries, or JSON parsing.
//! - `conflict_resolution` JSON parsing uses `parse_json_result()` which tries:
//!   1. <result>...</result> tags (preferred — added to prompt template)
//!   2. markdown code fences (```json ... ```)
//!   3. raw text as-is
//! - `update_session_summary()` takes 4 args (title param added in v2.4.0+Search).
//!   Pass None for title in recall_synthesis writeback to preserve the LLM title.
//! - `TaskWorker::new()` now takes 3 args: (storage, router, worker_config).
//!   batch_size and poll_interval are derived from WorkerConfig, not hardcoded.
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase A - 🌟 Created (skeleton).
//! v2.5.0+SuperNode Phase B - 🌟 Full result parsing + writeback per task type.
//!   Added prompts.rs integration. Added conflict_resolution edge invalidation.
//!   Added parse_json_result() for markdown-fence stripping.
//! v2.5.0+Fix              - 🔧 [BUG FIX] TaskWorker::new() now accepts WorkerConfig
//!   as 3rd argument — reads poll_interval_secs and max_concurrent from config
//!   instead of using hardcoded DEFAULT_* constants. server.rs updated to match.
//!                         - 🔧 [BUG FIX] CognitiveTaskType variants aligned with
//!   config_supernode.rs: SessionTitle, CommunityNarrative, ConflictResolution,
//!   RecallSynthesis, CodeAnalysis, EntityDescription. Removed non-existent variants
//!   CommunitySummary, NaturalSummary, CustomPrompt.
//!                         - 🔧 [BUG FIX] conflict_resolution writeback now correctly
//!   matches on CognitiveTaskType::ConflictResolution instead of the invalid
//!   CustomPrompt + target_table guard that could never trigger.
//!                         - 🔧 [FIX 4] Added clean_llm_response() — strips DeepSeek R1
//!   <think>...</think> chains, "Here is/Sure, here's" preambles, and surrounding
//!   quotes. Called before writeback and JSON parsing for all task types.
//!                         - 🔧 [FIX 3] parse_json_result() now tries <result>...</result>
//!   extraction before markdown fence stripping, matching the updated conflict_resolution
//!   prompt template in prompts.rs.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::storage::MemoryStorage;
use super::llm_provider::{ChatMessage, ChatRequest, CognitiveTaskType};
use super::llm_router::LlmRouter;
use super::storage_supernode::CognitiveTaskRow;
// config_supernode is declared at crate root in lib.rs (not under config/)
use crate::config_supernode::{PrivacyLevel, WorkerConfig};
use super::prompts::{
    SessionTitleInput, build_session_title,
    CommunityNarrativeInput, build_community_narrative,
    ConflictResolutionInput, ConflictingEdge, build_conflict_resolution,
    RecallSynthesisInput, build_recall_synthesis,
    CodeAnalysisInput, build_code_analysis,
    EntityDescriptionInput, build_entity_description,
};

// ============================================
// Constants
// ============================================

const MAX_RESULT_LEN: usize = 8192;

// ============================================
// TaskWorker
// ============================================

pub struct TaskWorker {
    storage: Arc<MemoryStorage>,
    router: Arc<LlmRouter>,
    batch_size: usize,
    poll_interval: Duration,
}

impl TaskWorker {
    /// Create a new TaskWorker from storage, router, and worker config.
    ///
    /// ## v2.5.0+Fix
    /// Now takes `WorkerConfig` as 3rd argument to read poll_interval_secs
    /// and max_concurrent from the actual config file, not hardcoded defaults.
    /// server.rs passes `config.memchain.supernode.worker.clone()`.
    pub fn new(
        storage: Arc<MemoryStorage>,
        router: Arc<LlmRouter>,
        worker_config: WorkerConfig,
    ) -> Self {
        Self {
            storage,
            router,
            // max_concurrent caps the batch_size (not yet used for semaphore
            // but drives the claim size so we don't over-claim)
            batch_size: worker_config.max_concurrent.max(1).min(50),
            poll_interval: Duration::from_secs(worker_config.poll_interval_secs.max(1)),
        }
    }

    pub async fn run(self, mut shutdown_rx: broadcast::Receiver<()>) {
        info!(
            batch_size = self.batch_size,
            poll_interval_secs = self.poll_interval.as_secs(),
            "[TASK_WORKER] Started"
        );

        let mut timer = tokio::time::interval(self.poll_interval);
        timer.tick().await;

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("[TASK_WORKER] Shutdown signal received, stopping");
                    break;
                }
                _ = timer.tick() => {
                    self.process_batch().await;
                }
            }
        }

        info!("[TASK_WORKER] Stopped");
    }

    async fn process_batch(&self) {
        let tasks = self.storage.claim_pending_tasks(self.batch_size).await;
        if tasks.is_empty() {
            debug!("[TASK_WORKER] No pending tasks");
            return;
        }

        info!(count = tasks.len(), "[TASK_WORKER] Processing batch");

        let mut handles = Vec::with_capacity(tasks.len());
        for task in tasks {
            let storage = Arc::clone(&self.storage);
            let router = Arc::clone(&self.router);
            handles.push(tokio::spawn(async move {
                Self::process_task(storage, router, task).await;
            }));
        }

        for handle in handles {
            if let Err(e) = handle.await {
                error!(error = %e, "[TASK_WORKER] Task panicked");
            }
        }
    }

    async fn process_task(
        storage: Arc<MemoryStorage>,
        router: Arc<LlmRouter>,
        task: CognitiveTaskRow,
    ) {
        let start = Instant::now();
        let task_id = task.id;
        let task_type_str = task.task_type.as_str();

        debug!(id = task_id, task_type = task_type_str, "[TASK_WORKER] Processing");

        let task_type = match CognitiveTaskType::from_str(task_type_str) {
            Some(t) => t,
            None => {
                warn!(id = task_id, task_type = task_type_str, "[TASK_WORKER] Unknown task type");
                let _ = storage.fail_task(task_id, &format!("unknown task_type: {}", task_type_str)).await;
                return;
            }
        };

        let payload: serde_json::Value = match serde_json::from_str(&task.payload) {
            Ok(v) => v,
            Err(e) => {
                warn!(id = task_id, "[TASK_WORKER] Invalid payload: {}", e);
                let _ = storage.fail_task(task_id, &format!("invalid payload: {}", e)).await;
                return;
            }
        };

        let privacy = PrivacyLevel::from_str(task.privacy_level.as_str());

        // Build prompt via prompts.rs
        let chat_req = match Self::build_prompt_for_task(&task_type, &payload, privacy).await {
            Ok(req) => req,
            Err(e) => {
                warn!(id = task_id, "[TASK_WORKER] Prompt build failed: {}", e);
                let _ = storage.fail_task(task_id, &format!("prompt build: {}", e)).await;
                return;
            }
        };

        // Dispatch to LLM provider
        let resp = match router.route(&task_type, &chat_req).await {
            Ok(r) => r,
            Err(e) => {
                warn!(id = task_id, error = %e, "[TASK_WORKER] LLM call failed");
                let _ = storage.fail_task(task_id, &e.to_string()).await;
                return;
            }
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;

        // ── v2.5.0+Fix: clean_llm_response BEFORE any processing ──────────
        // Strips <think> chains (DeepSeek R1), preamble phrases, and
        // normalizes the output before writeback or JSON parsing.
        let cleaned = clean_llm_response(&resp.content, &task_type);
        let result_stored = &cleaned[..cleaned.len().min(MAX_RESULT_LEN)];

        let token_usage_json = serde_json::json!({
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "cached": resp.usage.cached_tokens,
        }).to_string();

        // Writeback to target table
        if let (Some(ref table), Some(ref target_id)) = (&task.target_table, &task.target_id) {
            if let Err(e) = Self::write_back(
                &storage, &task_type, table, target_id,
                result_stored, &payload,
            ).await {
                warn!(
                    id = task_id, table = table, target_id = target_id,
                    error = %e, "[TASK_WORKER] Writeback failed (result preserved in DB)"
                );
                // Non-fatal: result still stored in cognitive_tasks.result
            }
        }

        if let Err(e) = storage.complete_task(
            task_id, result_stored,
            &resp.provider_name, &resp.model_used, &token_usage_json,
        ).await {
            warn!(id = task_id, error = %e, "[TASK_WORKER] complete_task failed");
        }

        if let Err(e) = storage.insert_usage_log(
            Some(task_id),
            &resp.provider_name, &resp.model_used,
            resp.usage.input_tokens as i64,
            resp.usage.output_tokens as i64,
            resp.usage.cached_tokens as i64,
            elapsed_ms as i64,
        ).await {
            warn!(id = task_id, error = %e, "[TASK_WORKER] insert_usage_log failed");
        }

        info!(
            id = task_id, task_type = task_type_str,
            provider = %resp.provider_name, model = %resp.model_used,
            input_tokens = resp.usage.input_tokens,
            output_tokens = resp.usage.output_tokens,
            latency_ms = elapsed_ms,
            "[TASK_WORKER] ✅ Complete"
        );
    }

    // ============================================
    // Prompt Builders (delegates to prompts.rs)
    // ============================================

    async fn build_prompt_for_task(
        task_type: &CognitiveTaskType,
        payload: &serde_json::Value,
        privacy: PrivacyLevel,
    ) -> Result<ChatRequest, String> {
        let messages = match task_type {
            CognitiveTaskType::SessionTitle => {
                let entity_names_raw: Vec<String> = payload["entity_names"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| v.as_str().map(String::from)).collect();
                let entity_refs: Vec<&str> = entity_names_raw.iter().map(|s| s.as_str()).collect();

                build_session_title(&SessionTitleInput {
                    entity_names: &entity_refs,
                    project_name: payload["project_name"].as_str(),
                    first_user_message: payload["first_user_message"].as_str(),
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::CommunityNarrative => {
                let community_name = payload["community_name"].as_str()
                    .unwrap_or("unknown community");
                let members_raw: Vec<(String, String, i64)> = payload["members"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| Some((
                        v["name"].as_str()?.to_string(),
                        v["type"].as_str().unwrap_or("entity").to_string(),
                        v["mention_count"].as_i64().unwrap_or(1),
                    ))).collect();
                let member_refs: Vec<(&str, &str, i64)> = members_raw.iter()
                    .map(|(n, t, c)| (n.as_str(), t.as_str(), *c)).collect();
                let edges_raw: Vec<(String, String, String)> = payload["key_edges"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| Some((
                        v["source"].as_str()?.to_string(),
                        v["relation"].as_str()?.to_string(),
                        v["target"].as_str()?.to_string(),
                    ))).collect();
                let edge_refs: Vec<(&str, &str, &str)> = edges_raw.iter()
                    .map(|(s, r, t)| (s.as_str(), r.as_str(), t.as_str())).collect();

                build_community_narrative(&CommunityNarrativeInput {
                    community_name,
                    members: &member_refs,
                    key_edges: &edge_refs,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::ConflictResolution => {
                let edge_ids: Vec<i64> = payload["conflict_edge_ids"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| v.as_i64()).collect();
                let edges_raw: Vec<ConflictingEdge> = payload["edges"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| Some(ConflictingEdge {
                        edge_id: v["edge_id"].as_i64()?,
                        source: v["source"].as_str().unwrap_or("").to_string(),
                        relation: v["relation"].as_str().unwrap_or("").to_string(),
                        target: v["target"].as_str().unwrap_or("").to_string(),
                        valid_from: v["valid_from"].as_i64().unwrap_or(0),
                        fact_text: v["fact_text"].as_str().map(String::from),
                    })).collect();

                build_conflict_resolution(&ConflictResolutionInput {
                    conflict_edge_ids: &edge_ids,
                    edges: &edges_raw,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::RecallSynthesis => {
                let entity_names_raw: Vec<String> = payload["entity_names"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| v.as_str().map(String::from)).collect();
                let entity_refs: Vec<&str> = entity_names_raw.iter().map(|s| s.as_str()).collect();

                build_recall_synthesis(&RecallSynthesisInput {
                    session_id: payload["session_id"].as_str().unwrap_or(""),
                    existing_summary: payload["existing_summary"].as_str(),
                    entity_names: &entity_refs,
                    turn_count: payload["turn_count"].as_i64().unwrap_or(0),
                    turns: &[],
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::CodeAnalysis => {
                let tags_raw: Vec<String> = payload["existing_tags"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| v.as_str().map(String::from)).collect();
                let tag_refs: Vec<&str> = tags_raw.iter().map(|s| s.as_str()).collect();

                build_code_analysis(&CodeAnalysisInput {
                    artifact_id: payload["artifact_id"].as_str().unwrap_or(""),
                    language: payload["language"].as_str().unwrap_or("unknown"),
                    code_content: payload["code_content"].as_str().unwrap_or(""),
                    existing_tags: &tag_refs,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::EntityDescription => {
                let relations_raw: Vec<(String, String)> = payload["relations"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| Some((
                        v["relation_type"].as_str()?.to_string(),
                        v["other_name"].as_str().unwrap_or("").to_string(),
                    ))).collect();
                let rel_refs: Vec<(&str, &str)> = relations_raw.iter()
                    .map(|(r, n)| (r.as_str(), n.as_str())).collect();

                build_entity_description(&EntityDescriptionInput {
                    entity_name: payload["entity_name"].as_str()
                        .ok_or("missing entity_name")?,
                    entity_type: payload["entity_type"].as_str().unwrap_or("entity"),
                    relations: &rel_refs,
                    privacy_level: privacy,
                })
            }
        };

        Ok(ChatRequest {
            messages,
            model_override: None,
            max_tokens: None,
            temperature: None,
            stop: None,
        })
    }

    // ============================================
    // Writeback per task type
    // ============================================

    async fn write_back(
        storage: &Arc<MemoryStorage>,
        task_type: &CognitiveTaskType,
        target_table: &str,
        target_id: &str,
        result: &str,
        payload: &serde_json::Value,
    ) -> Result<(), String> {
        match task_type {
            // ── session_title: plain text → sessions.title ──────────────────
            CognitiveTaskType::SessionTitle => {
                let title = result.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim();
                if title.is_empty() {
                    return Err("LLM returned empty title".to_string());
                }
                let conn = storage.conn_lock().await;
                conn.execute(
                    "UPDATE sessions SET title = ?1 WHERE session_id = ?2",
                    rusqlite::params![title, target_id],
                ).map_err(|e| format!("sessions.title writeback: {}", e))?;
                debug!(session = target_id, title = title, "[TASK_WORKER] session_title written");
            }

            // ── community_narrative: plain text → communities.summary ────────
            CognitiveTaskType::CommunityNarrative => {
                let summary = result.trim();
                if summary.is_empty() {
                    return Err("LLM returned empty community summary".to_string());
                }
                if let Some(comm) = storage.get_community(target_id).await {
                    let owner_dummy = [0u8; 32];
                    storage.upsert_community(
                        target_id, &owner_dummy, &comm.name,
                        Some(summary), comm.description.as_deref(),
                        comm.entity_count,
                    ).await.map_err(|e| format!("communities.summary writeback: {}", e))?;
                    debug!(community = target_id, "[TASK_WORKER] community_narrative written");
                } else {
                    return Err(format!("Community {} not found for writeback", target_id));
                }
            }

            // ── conflict_resolution: JSON → invalidate losing edges ──────────
            // v2.5.0+Fix: Now correctly matches CognitiveTaskType::ConflictResolution
            // instead of the invalid CustomPrompt + target_table guard.
            CognitiveTaskType::ConflictResolution => {
                Self::write_back_conflict_resolution(storage, target_id, result, payload).await?;
            }

            // ── recall_synthesis: JSON → sessions.summary + key_decisions ────
            CognitiveTaskType::RecallSynthesis => {
                let parsed = parse_json_result(result);
                let summary = parsed["summary"].as_str().unwrap_or(result).trim();
                let key_decisions = parsed["key_decisions"].as_str();

                if summary.is_empty() {
                    return Err("LLM returned empty recall summary".to_string());
                }
                // Pass None for title to preserve existing LLM-generated title
                storage.update_session_summary(target_id, summary, key_decisions, None).await;
                debug!(session = target_id, "[TASK_WORKER] recall_synthesis written");
            }

            // ── code_analysis: JSON → artifacts.description ─────────────────
            CognitiveTaskType::CodeAnalysis => {
                let parsed = parse_json_result(result);
                let description = parsed["description"].as_str()
                    .unwrap_or(result).trim();
                if description.is_empty() {
                    return Err("LLM returned empty code description".to_string());
                }
                let conn = storage.conn_lock().await;
                conn.execute(
                    "UPDATE artifacts SET description = ?1 WHERE artifact_id = ?2",
                    rusqlite::params![description, target_id],
                ).map_err(|e| format!("artifacts.description writeback: {}", e))?;
                debug!(artifact = target_id, "[TASK_WORKER] code_analysis written");
            }

            // ── entity_description: plain text → entities.description ────────
            CognitiveTaskType::EntityDescription => {
                let desc = result.trim_matches(|c| c == '"' || c == '\'').trim();
                if desc.is_empty() {
                    return Err("LLM returned empty entity description".to_string());
                }
                let conn = storage.conn_lock().await;
                conn.execute(
                    "UPDATE entities SET description = ?1, updated_at = strftime('%s', 'now')
                     WHERE entity_id = ?2",
                    rusqlite::params![desc, target_id],
                ).map_err(|e| format!("entities.description writeback: {}", e))?;
                debug!(entity = target_id, "[TASK_WORKER] entity_description written");
            }
        }

        Ok(())
    }

    async fn write_back_conflict_resolution(
        storage: &Arc<MemoryStorage>,
        _target_id: &str,
        result: &str,
        payload: &serde_json::Value,
    ) -> Result<(), String> {
        let parsed = parse_json_result(result);

        let keep_id = parsed["keep_edge_id"].as_i64()
            .ok_or_else(|| format!("conflict_resolution JSON missing keep_edge_id: {}", result))?;

        let reason = parsed["reason"].as_str().unwrap_or("LLM resolution");
        debug!(keep_edge_id = keep_id, reason = reason, "[TASK_WORKER] Conflict resolution");

        if let Some(edge_ids) = payload["conflict_edge_ids"].as_array() {
            for val in edge_ids {
                if let Some(eid) = val.as_i64() {
                    if eid != keep_id {
                        storage.invalidate_edge(eid).await;
                        debug!(edge_id = eid, "[TASK_WORKER] Conflicting edge invalidated");
                    }
                }
            }
        }

        Ok(())
    }
}

// ============================================
// LLM Response Cleaner (Fix 4)
// ============================================

/// Clean an LLM response before writeback or JSON parsing.
///
/// ## Problems Addressed
///
/// ### 1. DeepSeek R1 `<think>` chains
/// DeepSeek's R1 model emits reasoning in `<think>...</think>` tags before
/// the actual answer. These must be stripped before any processing:
/// ```text
/// <think>
/// Let me think about this...
/// The session is about JWT auth...
/// </think>
/// JWT Auth: React + TypeScript
/// ```
/// → `JWT Auth: React + TypeScript`
///
/// ### 2. Common preamble phrases
/// Many models add introductory phrases before the actual answer:
/// - "Here is the title: ..."
/// - "Sure, here's a summary: ..."
/// - "Based on the conversation: ..."
/// These are stripped for plain-text task types.
///
/// ### 3. Task-type-specific normalization
/// - `session_title`: Take only the last non-empty line (LLMs sometimes explain
///   their reasoning before the final title). Strip all surrounding quotes.
/// - Other plain-text types: Return cleaned text as-is.
/// - JSON types (conflict_resolution, recall_synthesis, code_analysis):
///   Do NOT normalize — `parse_json_result()` handles these.
///
/// ## Why not clean in prompts.rs?
/// The cleaning is output-side (LLM response), not input-side (prompt).
/// prompts.rs handles prompt *construction*; task_worker.rs handles response *processing*.
fn clean_llm_response(raw: &str, task_type: &CognitiveTaskType) -> String {
    // Step 1: Strip <think>...</think> blocks (DeepSeek R1, Qwen QwQ)
    let after_think = strip_think_tags(raw);
    let trimmed = after_think.trim();

    // Step 2: For JSON task types, return as-is (parse_json_result handles these)
    if matches!(task_type,
        CognitiveTaskType::ConflictResolution |
        CognitiveTaskType::RecallSynthesis |
        CognitiveTaskType::CodeAnalysis
    ) {
        return trimmed.to_string();
    }

    // Step 3: Strip common preamble prefixes for plain-text task types
    let after_preamble = strip_preamble(trimmed);

    // Step 4: Task-type-specific normalization
    match task_type {
        CognitiveTaskType::SessionTitle => {
            // Take only the last non-empty line — models sometimes reason before
            // giving the final title. The last line is the actual answer.
            let last_line = after_preamble.lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .last()
                .unwrap_or(after_preamble.trim());

            // Strip all surrounding quotes (single, double, backtick)
            last_line.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim().to_string()
        }

        // For all other plain-text types, return cleaned content
        _ => after_preamble.trim().to_string(),
    }
}

/// Strip `<think>...</think>` blocks from LLM output.
///
/// Handles nested blocks by repeatedly stripping from the outside in.
/// Returns everything after the last `</think>` tag (or the original
/// string if no think tags are found).
fn strip_think_tags(text: &str) -> String {
    // Fast path: no think tags
    if !text.contains("<think>") && !text.contains("</think>") {
        return text.to_string();
    }

    // Find the last </think> closing tag — everything after it is the answer
    if let Some(end_pos) = text.rfind("</think>") {
        let after = &text[end_pos + "</think>".len()..];
        // Recurse in case there are nested blocks (unlikely but defensive)
        if after.contains("<think>") {
            return strip_think_tags(after);
        }
        return after.trim_start().to_string();
    }

    // Malformed: opening <think> with no closing </think>
    // Strip everything from <think> to end of string
    if let Some(start_pos) = text.find("<think>") {
        return text[..start_pos].trim_end().to_string();
    }

    text.to_string()
}

/// Strip common LLM preamble phrases from the beginning of a response.
///
/// These phrases add no value and corrupt stored titles/summaries.
fn strip_preamble(text: &str) -> &str {
    const PREAMBLES: &[&str] = &[
        "Here is the title: ",
        "Here is the title:",
        "Here's the title: ",
        "Here's the title:",
        "The title is: ",
        "The title is:",
        "Title: ",
        "Sure, here's a summary: ",
        "Sure, here's a summary:",
        "Sure! Here's a summary: ",
        "Here is a summary: ",
        "Here is a summary:",
        "Based on the conversation: ",
        "Based on the conversation,",
        "Based on the context: ",
        "Certainly! Here's",
        "Certainly, here's",
        "Of course! Here's",
        "Sure! Here's",
        "Sure, here's",
    ];

    let trimmed = text.trim();
    for preamble in PREAMBLES {
        if let Some(stripped) = trimmed.strip_prefix(preamble) {
            return stripped.trim();
        }
    }
    trimmed
}

// ============================================
// JSON result parser (Fix 3 — <result> tags + markdown fences)
// ============================================

/// Parse an LLM result that should be JSON.
///
/// ## Extraction Order (v2.5.0+Fix)
/// 1. `<result>...</result>` tags — preferred format added to conflict_resolution
///    prompt template. The most reliable extraction method because the model is
///    explicitly instructed to wrap JSON in these tags.
/// 2. Markdown code fences (` ```json ... ``` ` or ` ``` ... ``` `)
/// 3. Raw text as-is — fallback for models that don't add any wrapping
///
/// Returns `serde_json::Value::Null` on parse failure.
/// Callers check individual fields with `.as_str()`, `.as_i64()`, etc.
fn parse_json_result(result: &str) -> serde_json::Value {
    let trimmed = result.trim();

    // Priority 1: <result>...</result> tags (conflict_resolution prompt template)
    if let Some(start) = trimmed.find("<result>") {
        if let Some(end) = trimmed.find("</result>") {
            let inner = trimmed[start + "<result>".len()..end].trim();
            if let Ok(v) = serde_json::from_str(inner) {
                return v;
            }
            // <result> found but content isn't valid JSON — fall through
        }
    }

    // Priority 2: Markdown code fences
    let json_str = if trimmed.starts_with("```") {
        let after_fence = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim();
        after_fence.trim_end_matches("```").trim()
    } else {
        trimmed
    };

    // Priority 3: Parse as-is
    serde_json::from_str(json_str).unwrap_or_else(|e| {
        warn!(
            "[TASK_WORKER] JSON parse failed: {} | result preview: {}",
            e, &json_str[..json_str.len().min(200)]
        );
        serde_json::Value::Null
    })
}

impl std::fmt::Debug for TaskWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskWorker")
            .field("batch_size", &self.batch_size)
            .field("poll_interval_secs", &self.poll_interval.as_secs())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::memchain::llm_provider::CognitiveTaskType;

    #[test]
    fn test_strip_think_tags_basic() {
        let raw = "<think>\nLet me think...\n</think>\nJWT Auth: React";
        assert_eq!(strip_think_tags(raw), "JWT Auth: React");
    }

    #[test]
    fn test_strip_think_tags_no_tags() {
        let raw = "JWT Auth: React";
        assert_eq!(strip_think_tags(raw), "JWT Auth: React");
    }

    #[test]
    fn test_strip_think_tags_takes_last_close() {
        let raw = "<think>first</think> middle <think>second</think> answer";
        assert_eq!(strip_think_tags(raw), "answer");
    }

    #[test]
    fn test_strip_think_tags_unclosed() {
        let raw = "some text <think>reasoning never closed";
        assert_eq!(strip_think_tags(raw), "some text");
    }

    #[test]
    fn test_strip_preamble_title() {
        assert_eq!(strip_preamble("Here is the title: Project Alpha"), "Project Alpha");
        assert_eq!(strip_preamble("Title: JWT Auth"), "JWT Auth");
        assert_eq!(strip_preamble("No preamble here"), "No preamble here");
    }

    #[test]
    fn test_clean_session_title_takes_last_line() {
        let raw = "<think>reasoning</think>\nLet me think about it.\nFinal: JWT Auth: React + TypeScript";
        let cleaned = clean_llm_response(raw, &CognitiveTaskType::SessionTitle);
        assert_eq!(cleaned, "Final: JWT Auth: React + TypeScript");
    }

    #[test]
    fn test_clean_session_title_strips_quotes() {
        let raw = r#""JWT Auth: React""#;
        let cleaned = clean_llm_response(raw, &CognitiveTaskType::SessionTitle);
        assert_eq!(cleaned, "JWT Auth: React");
    }

    #[test]
    fn test_clean_community_narrative_preserves_multiline() {
        let raw = "<think>ok</think>\nThis community focuses on authentication.\nIt uses JWT and OAuth.";
        let cleaned = clean_llm_response(raw, &CognitiveTaskType::CommunityNarrative);
        assert!(cleaned.contains("authentication"));
        assert!(cleaned.contains("JWT"));
    }

    #[test]
    fn test_clean_json_types_not_normalized() {
        // JSON task types must be returned as-is (parse_json_result handles them)
        let raw = r#"<think>thinking</think>{"keep_edge_id": 42, "reason": "newer"}"#;
        let cleaned = clean_llm_response(raw, &CognitiveTaskType::ConflictResolution);
        assert!(cleaned.contains("keep_edge_id"));
        assert!(!cleaned.contains("<think>"));
    }

    #[test]
    fn test_parse_json_result_tags() {
        let result = r#"Some text <result>{"keep_edge_id": 42, "reason": "newer"}</result> more text"#;
        let parsed = parse_json_result(result);
        assert_eq!(parsed["keep_edge_id"].as_i64(), Some(42));
        assert_eq!(parsed["reason"].as_str(), Some("newer"));
    }

    #[test]
    fn test_parse_json_result_markdown_fence() {
        let result = "```json\n{\"summary\": \"Auth session\", \"key_decisions\": null}\n```";
        let parsed = parse_json_result(result);
        assert_eq!(parsed["summary"].as_str(), Some("Auth session"));
    }

    #[test]
    fn test_parse_json_result_raw() {
        let result = r#"{"keep_edge_id": 5, "reason": "most recent"}"#;
        let parsed = parse_json_result(result);
        assert_eq!(parsed["keep_edge_id"].as_i64(), Some(5));
    }

    #[test]
    fn test_parse_json_result_invalid_returns_null() {
        let parsed = parse_json_result("not json at all");
        assert_eq!(parsed, serde_json::Value::Null);
    }

    #[test]
    fn test_parse_json_result_prefers_tags_over_fence() {
        // When both <result> and ``` are present, <result> wins
        let result = "```json\n{\"wrong\": 1}\n```\n<result>{\"right\": 2}</result>";
        let parsed = parse_json_result(result);
        assert_eq!(parsed["right"].as_i64(), Some(2));
        assert!(parsed["wrong"].is_null());
    }
}
