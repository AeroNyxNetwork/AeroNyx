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
//!      d. parse LLM result (plain text or JSON depending on task type)
//!      e. write result to target table (sessions/communities/entities)
//!      f. complete_task() or fail_task()
//!      g. insert_usage_log()
//!   3. Sleep poll_interval_secs if no tasks claimed
//! ```
//!
//! ## Result Parsing per Task Type
//! - `session_title`      → plain text, trim whitespace + quotes
//! - `community_narrative`→ plain text, trim whitespace
//! - `conflict_resolution`→ JSON: `{"keep_edge_id": N, "reason": "..."}` — invalidates losers
//! - `recall_synthesis`   → JSON: `{"summary": "...", "key_decisions": "..."|null}`
//! - `code_analysis`      → JSON: `{"description": "...", "complexity": "...", "suggested_tags": [...]}`
//!
//! ## Writeback per Task Type
//! - `session_title`       → `UPDATE sessions SET title = ?`
//! - `community_narrative` → `storage.upsert_community()` with new summary
//! - `conflict_resolution` → `storage.invalidate_edge()` for losing edges
//! - `recall_synthesis`    → `storage.update_session_summary()` with natural text + key_decisions
//! - `code_analysis`       → direct SQL UPDATE artifacts SET description = ?
//! - `entity_description`  → direct SQL UPDATE entities SET description = ?
//! - `natural_summary`     → via update_session_summary() (same as recall_synthesis)
//! - `custom_prompt`       → result stored in cognitive_tasks.result only
//!
//! ⚠️ Important Note for Next Developer:
//! - `conflict_resolution` JSON parsing uses `serde_json::from_str` — if the LLM
//!   returns markdown-wrapped JSON (```json ... ```), strip the fences first.
//!   The `parse_json_result()` helper handles this.
//! - `update_session_summary()` in storage_graph.rs now takes 4 args (title param
//!   added in v2.4.0+Search). Pass None for title in recall_synthesis writeback
//!   to avoid overwriting the LLM-generated title.
//! - `upsert_community()` signature: (community_id, owner, name, summary, description, count).
//!   We pass the existing name and count to avoid resetting them.
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase A - 🌟 Created (skeleton).
//! v2.5.0+SuperNode Phase B - 🌟 Full result parsing + writeback per task type.
//!   Added prompts.rs integration. Added conflict_resolution edge invalidation.
//!   Added parse_json_result() for markdown-fence stripping.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

use super::storage::MemoryStorage;
use super::llm_provider::{ChatRequest, CognitiveTaskType};
use super::llm_router::LlmRouter;
use super::storage_supernode::CognitiveTaskRow;
use super::prompts::{
    PrivacyLevel,
    SessionTitleInput, build_session_title,
    CommunityNarrativeInput, build_community_narrative,
    ConflictResolutionInput, ConflictingEdge, build_conflict_resolution,
    RecallSynthesisInput, build_recall_synthesis,
    CodeAnalysisInput, build_code_analysis,
};

// ============================================
// Constants
// ============================================

const DEFAULT_BATCH_SIZE: usize = 5;
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;
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
    pub fn new(storage: Arc<MemoryStorage>, router: Arc<LlmRouter>) -> Self {
        Self {
            storage,
            router,
            batch_size: DEFAULT_BATCH_SIZE,
            poll_interval: Duration::from_secs(DEFAULT_POLL_INTERVAL_SECS),
        }
    }

    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1).min(50);
        self
    }

    #[must_use]
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
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

        let privacy = PrivacyLevel::from_str(&task.privacy_level);

        // Build prompt via prompts.rs
        let chat_req = match Self::build_prompt_for_task(
            &task_type, &payload, privacy, &storage
        ).await {
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
        let result_content = resp.content.trim();
        let result_stored = &result_content[..result_content.len().min(MAX_RESULT_LEN)];

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
        storage: &Arc<MemoryStorage>,
    ) -> Result<ChatRequest, String> {
        let messages = match task_type {
            CognitiveTaskType::SessionTitle => {
                let entity_names_raw: Vec<String> = payload["entity_names"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| v.as_str().map(String::from)).collect();
                let entity_refs: Vec<&str> = entity_names_raw.iter().map(|s| s.as_str()).collect();
                let project_name = payload["project_name"].as_str();
                let first_msg = payload["first_user_message"].as_str();

                build_session_title(&SessionTitleInput {
                    entity_names: &entity_refs,
                    project_name,
                    first_user_message: first_msg,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::CommunitySummary => {
                let community_name = payload["community_name"].as_str()
                    .unwrap_or("unknown community");
                let members_raw: Vec<(String, String, i64)> = payload["members"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| {
                        Some((
                            v["name"].as_str()?.to_string(),
                            v["type"].as_str().unwrap_or("entity").to_string(),
                            v["mention_count"].as_i64().unwrap_or(1),
                        ))
                    }).collect();
                let member_refs: Vec<(&str, &str, i64)> = members_raw.iter()
                    .map(|(n, t, c)| (n.as_str(), t.as_str(), *c)).collect();

                let edges_raw: Vec<(String, String, String)> = payload["key_edges"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| {
                        Some((
                            v["source"].as_str()?.to_string(),
                            v["relation"].as_str()?.to_string(),
                            v["target"].as_str()?.to_string(),
                        ))
                    }).collect();
                let edge_refs: Vec<(&str, &str, &str)> = edges_raw.iter()
                    .map(|(s, r, t)| (s.as_str(), r.as_str(), t.as_str())).collect();

                build_community_narrative(&CommunityNarrativeInput {
                    community_name,
                    members: &member_refs,
                    key_edges: &edge_refs,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::EntityDescription => {
                let entity_name = payload["entity_name"].as_str()
                    .ok_or("missing entity_name")?;
                let entity_type = payload["entity_type"].as_str().unwrap_or("entity");
                let relations_raw: Vec<(String, String)> = payload["relations"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| {
                        Some((
                            v["relation_type"].as_str()?.to_string(),
                            v["other_id"].as_str().unwrap_or("").to_string(),
                        ))
                    }).collect();
                let rel_refs: Vec<(&str, &str)> = relations_raw.iter()
                    .map(|(r, n)| (r.as_str(), n.as_str())).collect();

                super::prompts::build_entity_description(
                    &super::prompts::EntityDescriptionInput {
                        entity_name,
                        entity_type,
                        relations: &rel_refs,
                        privacy_level: privacy,
                    }
                )
            }

            CognitiveTaskType::NaturalSummary => {
                let entity_names_raw: Vec<String> = payload["entity_names"]
                    .as_array().unwrap_or(&vec![])
                    .iter().filter_map(|v| v.as_str().map(String::from)).collect();
                let entity_refs: Vec<&str> = entity_names_raw.iter().map(|s| s.as_str()).collect();
                let turn_count = payload["turn_count"].as_i64().unwrap_or(0);
                let existing = payload["existing_summary"].as_str();

                build_recall_synthesis(&RecallSynthesisInput {
                    session_id: payload["session_id"].as_str().unwrap_or(""),
                    existing_summary: existing,
                    entity_names: &entity_refs,
                    turn_count,
                    turns: &[], // No turns in structured mode
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::CustomPrompt => {
                let messages_json = payload["messages"].as_array()
                    .ok_or("custom_prompt requires 'messages' array")?;
                let messages: Vec<super::llm_provider::ChatMessage> = messages_json.iter()
                    .filter_map(|v| {
                        Some(super::llm_provider::ChatMessage {
                            role: v["role"].as_str()?.to_string(),
                            content: v["content"].as_str()?.to_string(),
                        })
                    }).collect();
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
            // ── session_title: plain text → sessions.title ──
            CognitiveTaskType::SessionTitle => {
                // Strip surrounding quotes that some models add
                let title = result.trim_matches(|c| c == '"' || c == '\'' || c == '`');
                let title = title.trim();
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

            // ── community_narrative: plain text → communities.summary ──
            CognitiveTaskType::CommunitySummary => {
                let summary = result.trim();
                if summary.is_empty() {
                    return Err("LLM returned empty community summary".to_string());
                }
                // Fetch existing community for name + entity_count (needed by upsert signature)
                if let Some(comm) = storage.get_community(target_id).await {
                    // Use dummy owner — upsert_community checks community_id PK
                    let owner_dummy = [0u8; 32]; // Not used in WHERE clause
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

            // ── conflict_resolution: JSON → invalidate losing edges ──
            CognitiveTaskType::CustomPrompt if target_table == "knowledge_edges" => {
                // conflict_resolution is stored as CustomPrompt with table=knowledge_edges
                Self::write_back_conflict_resolution(storage, target_id, result, payload).await?;
            }

            // ── recall_synthesis / natural_summary: JSON → sessions.summary + key_decisions ──
            CognitiveTaskType::NaturalSummary => {
                let parsed = parse_json_result(result);
                let summary = parsed["summary"].as_str().unwrap_or(result).trim();
                let key_decisions = parsed["key_decisions"].as_str();

                if summary.is_empty() {
                    return Err("LLM returned empty summary".to_string());
                }

                // update_session_summary(session_id, summary, key_decisions, title)
                // Pass None for title to preserve existing LLM-generated title
                storage.update_session_summary(
                    target_id, summary, key_decisions, None
                ).await;
                debug!(session = target_id, "[TASK_WORKER] natural_summary written");
            }

            // ── entity_description: plain text → entities.description ──
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

            // ── custom_prompt: result stored only in cognitive_tasks.result ──
            CognitiveTaskType::CustomPrompt => {
                debug!(id = target_id, "[TASK_WORKER] custom_prompt: result in DB only");
            }
        }

        Ok(())
    }

    /// Handle conflict_resolution writeback:
    /// Parse JSON result, keep winning edge, invalidate all others.
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

        // Invalidate all conflicting edges EXCEPT the winner
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
// JSON result parser (strips markdown fences)
// ============================================

/// Parse an LLM result that should be JSON.
///
/// Handles models that wrap JSON in markdown fences:
/// ```json
/// {"key": "value"}
/// ```
/// → `{"key": "value"}`
///
/// Returns `serde_json::Value::Null` on parse failure (callers check fields with `.as_str()`).
fn parse_json_result(result: &str) -> serde_json::Value {
    let trimmed = result.trim();

    // Strip markdown code fences if present
    let json_str = if trimmed.starts_with("```") {
        let after_fence = trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim();
        after_fence
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    serde_json::from_str(json_str).unwrap_or_else(|e| {
        warn!("[TASK_WORKER] JSON parse failed: {} | result: {}", e, &json_str[..json_str.len().min(200)]);
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
