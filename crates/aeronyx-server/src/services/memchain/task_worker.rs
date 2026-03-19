// ============================================
// File: crates/aeronyx-server/src/services/memchain/task_worker.rs
// ============================================
//! # TaskWorker — Async Cognitive Task Queue Worker
//!
//! ## CognitiveTaskType variant names (CRITICAL — must match llm_provider.rs)
//! The canonical enum lives in llm_provider.rs with these variants:
//!   SessionTitle | CommunitySummary | EntityDescription | NaturalSummary | CustomPrompt
//! All match arms in this file use those exact names.
//! task_type_str() (not as_str()) is the method to get the DB string.
//!
//! ## PrivacyLevel
//! Re-exported from config_supernode via prompts.rs.
//! Now has Structured / Summary / Full variants (Summary was missing before).
//! PrivacyLevel::from_str() is NOT available — use match on the string directly.
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase A - 🌟 Created (skeleton).
//! v2.5.0+SuperNode Phase B - 🌟 Full result parsing + writeback per task type.
//! v2.5.0+Fix              - 🔧 Various alignment fixes.
//! v2.5.0+Audit Fix        - 🔧 Aligned CognitiveTaskType variants to llm_provider.rs.
//!   Fixed PrivacyLevel parsing (no from_str — match string directly).
//!   Fixed target_table/target_id Option<String> destructuring.
//!   Fixed AnthropicProvider arg count. Fixed provider new() Result handling.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::storage::MemoryStorage;
// CognitiveTaskType lives in llm_provider — use those exact variant names
use super::llm_provider::{ChatMessage, ChatRequest, CognitiveTaskType};
use super::llm_router::LlmRouter;
use super::storage_supernode::CognitiveTaskRow;
// PrivacyLevel re-exported from config_supernode via prompts
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
    pub fn new(
        storage: Arc<MemoryStorage>,
        router: Arc<LlmRouter>,
        worker_config: WorkerConfig,
    ) -> Self {
        Self {
            storage,
            router,
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

        // CognitiveTaskType::from_str() is the method on llm_provider's enum
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

        // Parse privacy level from string — PrivacyLevel has no from_str(), match directly
        let privacy = match task.privacy_level.as_str() {
            "full" => PrivacyLevel::Full,
            "summary" => PrivacyLevel::Summary,
            _ => PrivacyLevel::Structured,
        };

        let chat_req = match Self::build_prompt_for_task(&task_type, &payload, privacy).await {
            Ok(req) => req,
            Err(e) => {
                warn!(id = task_id, "[TASK_WORKER] Prompt build failed: {}", e);
                let _ = storage.fail_task(task_id, &format!("prompt build: {}", e)).await;
                return;
            }
        };

        let resp = match router.route(&task_type, &chat_req).await {
            Ok(r) => r,
            Err(e) => {
                warn!(id = task_id, error = %e, "[TASK_WORKER] LLM call failed");
                let _ = storage.fail_task(task_id, &e.to_string()).await;
                return;
            }
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;

        let cleaned = clean_llm_response(&resp.content, &task_type);
        let result_stored = &cleaned[..cleaned.len().min(MAX_RESULT_LEN)];

        let token_usage_json = serde_json::json!({
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "cached": resp.usage.cached_tokens,
        }).to_string();

        // target_table and target_id are Option<String> — use as_deref()
        if let (Some(table), Some(tid)) = (task.target_table.as_deref(), task.target_id.as_deref()) {
            if let Err(e) = Self::write_back(
                &storage, &task_type, table, tid, result_stored, &payload,
            ).await {
                warn!(
                    id = task_id, table = table, target_id = tid,
                    error = %e, "[TASK_WORKER] Writeback failed (result preserved in DB)"
                );
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
    // Prompt Builders — variant names from llm_provider::CognitiveTaskType
    // ============================================

    async fn build_prompt_for_task(
        task_type: &CognitiveTaskType,
        payload: &serde_json::Value,
        privacy: PrivacyLevel,
    ) -> Result<ChatRequest, String> {
        let messages = match task_type {
            // SessionTitle maps to session_title in DB
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

            // CommunitySummary maps to community_summary in DB
            CognitiveTaskType::CommunitySummary => {
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

            // NaturalSummary maps to natural_summary in DB — uses recall_synthesis prompt
            CognitiveTaskType::NaturalSummary => {
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

            // CustomPrompt — conflict resolution, code analysis, or caller-supplied
            CognitiveTaskType::CustomPrompt => {
                // Check if this is a conflict_resolution task (target_table = knowledge_edges)
                if payload["conflict_edge_ids"].is_array() {
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
                            confidence: v["confidence"].as_f64(),
                        })).collect();

                    build_conflict_resolution(&ConflictResolutionInput {
                        conflict_edge_ids: &edge_ids,
                        edges: &edges_raw,
                        privacy_level: privacy,
                    })
                } else if payload.get("code_content").is_some() || payload.get("language").is_some() {
                    // code_analysis sub-type
                    let tags_raw: Vec<String> = payload["existing_tags"]
                        .as_array().unwrap_or(&vec![])
                        .iter().filter_map(|v| v.as_str().map(String::from)).collect();
                    let tag_refs: Vec<&str> = tags_raw.iter().map(|s| s.as_str()).collect();

                    build_code_analysis(&CodeAnalysisInput {
                        artifact_id: payload["artifact_id"].as_str().unwrap_or(""),
                        language: payload["language"].as_str().unwrap_or("unknown"),
                        line_count: payload["line_count"].as_i64(),
                        code_content: payload["code_content"].as_str().unwrap_or(""),
                        existing_tags: &tag_refs,
                        privacy_level: privacy,
                    })
                } else {
                    // Raw custom prompt
                    let messages_json = payload["messages"].as_array()
                        .ok_or("custom_prompt requires 'messages' array")?;
                    let messages: Vec<ChatMessage> = messages_json.iter()
                        .filter_map(|v| Some(ChatMessage {
                            role: v["role"].as_str()?.to_string(),
                            content: v["content"].as_str()?.to_string(),
                        })).collect();
                    if messages.is_empty() {
                        return Err("custom_prompt messages array is empty".to_string());
                    }
                    return Ok(ChatRequest {
                        messages,
                        model_override: payload["model"].as_str().map(String::from),
                        max_tokens: payload["max_tokens"].as_u64().map(|v| v as u32),
                        temperature: payload["temperature"].as_f64().map(|v| v as f32),
                        stop: None,
                    });
                }
            }

            // EntityDescription maps to entity_description in DB
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
            CognitiveTaskType::SessionTitle => {
                let title = result.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim();
                if title.is_empty() { return Err("LLM returned empty title".to_string()); }
                let conn = storage.conn_lock().await;
                conn.execute(
                    "UPDATE sessions SET title = ?1 WHERE session_id = ?2",
                    rusqlite::params![title, target_id],
                ).map_err(|e| format!("sessions.title writeback: {}", e))?;
                debug!(session = target_id, title = title, "[TASK_WORKER] session_title written");
            }

            CognitiveTaskType::CommunitySummary => {
                let summary = result.trim();
                if summary.is_empty() { return Err("LLM returned empty community summary".to_string()); }
                if let Some(comm) = storage.get_community(target_id).await {
                    let owner_dummy = [0u8; 32];
                    storage.upsert_community(
                        target_id, &owner_dummy, &comm.name,
                        Some(summary), comm.description.as_deref(),
                        comm.entity_count,
                    ).await.map_err(|e| format!("communities.summary writeback: {}", e))?;
                    debug!(community = target_id, "[TASK_WORKER] community_summary written");
                } else {
                    return Err(format!("Community {} not found for writeback", target_id));
                }
            }

            CognitiveTaskType::NaturalSummary => {
                let parsed = parse_json_result(result);
                let summary = parsed["summary"].as_str().unwrap_or(result).trim();
                let key_decisions = parsed["key_decisions"].as_str();
                if summary.is_empty() { return Err("LLM returned empty summary".to_string()); }
                storage.update_session_summary(target_id, summary, key_decisions, None).await;
                debug!(session = target_id, "[TASK_WORKER] natural_summary written");
            }

            CognitiveTaskType::CustomPrompt => {
                // conflict_resolution sub-type: invalidate losing edges
                if payload["conflict_edge_ids"].is_array() {
                    let parsed = parse_json_result(result);
                    let keep_id = parsed["keep_edge_id"].as_i64()
                        .ok_or_else(|| format!("conflict_resolution missing keep_edge_id: {}", result))?;
                    if let Some(edge_ids) = payload["conflict_edge_ids"].as_array() {
                        for val in edge_ids {
                            if let Some(eid) = val.as_i64() {
                                if eid != keep_id {
                                    storage.invalidate_edge(eid).await;
                                }
                            }
                        }
                    }
                } else if payload.get("code_content").is_some() || payload.get("language").is_some() {
                    // code_analysis sub-type
                    let parsed = parse_json_result(result);
                    let description = parsed["description"].as_str().unwrap_or(result).trim();
                    if description.is_empty() { return Err("LLM returned empty code description".to_string()); }
                    let conn = storage.conn_lock().await;
                    conn.execute(
                        "UPDATE artifacts SET description = ?1 WHERE artifact_id = ?2",
                        rusqlite::params![description, target_id],
                    ).map_err(|e| format!("artifacts.description writeback: {}", e))?;
                } else {
                    debug!(id = target_id, "[TASK_WORKER] custom_prompt: result in DB only");
                }
            }

            CognitiveTaskType::EntityDescription => {
                let desc = result.trim_matches(|c| c == '"' || c == '\'').trim();
                if desc.is_empty() { return Err("LLM returned empty entity description".to_string()); }
                let conn = storage.conn_lock().await;
                conn.execute(
                    "UPDATE entities SET description = ?1, updated_at = strftime('%s', 'now') WHERE entity_id = ?2",
                    rusqlite::params![desc, target_id],
                ).map_err(|e| format!("entities.description writeback: {}", e))?;
                debug!(entity = target_id, "[TASK_WORKER] entity_description written");
            }
        }

        Ok(())
    }
}

// ============================================
// LLM Response Cleaner
// ============================================

fn clean_llm_response(raw: &str, task_type: &CognitiveTaskType) -> String {
    let after_think = strip_think_tags(raw);
    let trimmed = after_think.trim();

    // JSON task types: return as-is, let parse_json_result handle
    if matches!(task_type, CognitiveTaskType::NaturalSummary | CognitiveTaskType::CustomPrompt) {
        return trimmed.to_string();
    }

    let after_preamble = strip_preamble(trimmed);

    match task_type {
        CognitiveTaskType::SessionTitle => {
            let last_line = after_preamble.lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .last()
                .unwrap_or(after_preamble.trim());
            last_line.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim().to_string()
        }
        _ => after_preamble.trim().to_string(),
    }
}

fn strip_think_tags(text: &str) -> String {
    if !text.contains("<think>") && !text.contains("</think>") {
        return text.to_string();
    }
    if let Some(end_pos) = text.rfind("</think>") {
        let after = &text[end_pos + "</think>".len()..];
        if after.contains("<think>") { return strip_think_tags(after); }
        return after.trim_start().to_string();
    }
    if let Some(start_pos) = text.find("<think>") {
        return text[..start_pos].trim_end().to_string();
    }
    text.to_string()
}

fn strip_preamble(text: &str) -> &str {
    const PREAMBLES: &[&str] = &[
        "Here is the title: ", "Here is the title:", "Here's the title: ",
        "Here's the title:", "The title is: ", "The title is:", "Title: ",
        "Sure, here's a summary: ", "Sure, here's a summary:", "Here is a summary: ",
        "Here is a summary:", "Based on the conversation: ", "Based on the conversation,",
        "Certainly! Here's", "Certainly, here's", "Of course! Here's", "Sure! Here's",
    ];
    let trimmed = text.trim();
    for p in PREAMBLES {
        if let Some(s) = trimmed.strip_prefix(p) { return s.trim(); }
    }
    trimmed
}

// ============================================
// JSON result parser (<r> tags → markdown fence → raw)
// ============================================

fn parse_json_result(result: &str) -> serde_json::Value {
    let trimmed = result.trim();
    if let Some(start) = trimmed.find("<r>") {
        if let Some(end) = trimmed.find("</r>") {
            let inner = trimmed[start + "<r>".len()..end].trim();
            if let Ok(v) = serde_json::from_str(inner) { return v; }
        }
    }
    let json_str = if trimmed.starts_with("```") {
        trimmed.trim_start_matches("```json").trim_start_matches("```").trim()
            .trim_end_matches("```").trim()
    } else {
        trimmed
    };
    serde_json::from_str(json_str).unwrap_or_else(|e| {
        warn!("[TASK_WORKER] JSON parse failed: {} | preview: {}", e, &json_str[..json_str.len().min(200)]);
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
