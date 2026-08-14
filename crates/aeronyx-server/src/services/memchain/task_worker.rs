// ============================================
// File: crates/aeronyx-server/src/services/memchain/task_worker.rs
// ============================================
//! # TaskWorker — Async Cognitive Task Queue Worker
//!
//! ## CognitiveTaskType
//! The canonical enum lives in config_supernode.rs with these variants:
//!   SessionTitle | CommunityNarrative | ConflictResolution |
//!   RecallSynthesis | CodeAnalysis | EntityDescription
//! Re-exported via llm_provider.rs. All match arms in this file use those exact names.
//! as_str() returns the DB string (e.g. "session_title").
//!
//! ## PrivacyLevel
//! Defined in config_supernode.rs with variants: Structured / Summary / Full.
//! Re-exported via prompts.rs. Use PrivacyLevel::from_str() for DB string parsing.
//!
//! ## Last Modified
//! v2.5.7-TaskOwnership - [SUPERNODE-TASK-OWNERSHIP 2026-08-14 by Codex]
//!   Enforces configured per-task timeouts, owns concurrent work in a JoinSet,
//!   and releases unfinished claims during graceful shutdown.
//! v2.5.6-DurableWriteback - [SUPERNODE-DURABLE-WRITEBACK 2026-08-14 by Codex]
//!   Persists inference output and usage atomically before target writeback;
//!   retries now reuse the staged result instead of calling the provider again.
//! v2.5.5-FailureBoundary - [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex]
//!   Reclaims panicked tasks, persists only stable failure reasons, and removes
//!   provider output previews from JSON-parse logs.
//! v2.5.0+SuperNode Phase A - 🌟 Created (skeleton).
//! v2.5.0+SuperNode Phase B - 🌟 Full result parsing + writeback per task type.
//! v2.5.0+Fix              - 🔧 Various alignment fixes.
//! v2.5.0+Unify            - 🔧 [BUG FIX] Aligned to unified CognitiveTaskType from
//!   config_supernode.rs. Replaced CommunitySummary→CommunityNarrative,
//!   NaturalSummary→RecallSynthesis, CustomPrompt→ConflictResolution/CodeAnalysis.
//!   Fixed PrivacyLevel parsing to use PrivacyLevel::from_str().
//!   Fixed CognitiveTaskType::from_str → CognitiveTaskType::parse().
//!   Fixed clean_llm_response match arms to use correct variant names.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::broadcast;
use tokio::task::{Id, JoinSet};
use tracing::{debug, error, info, warn};

use super::storage::MemoryStorage;
// CognitiveTaskType is re-exported from config_supernode via llm_provider
use super::llm_provider::{ChatRequest, CognitiveTaskType};
use super::llm_router::LlmRouter;
use super::storage_supernode::{CognitiveTaskRow, StageTaskResult, TaskUsageRecord};
// PrivacyLevel is re-exported from config_supernode via prompts
use super::prompts::{
    build_code_analysis, build_community_narrative, build_conflict_resolution,
    build_entity_description, build_recall_synthesis, build_session_title, CodeAnalysisInput,
    CommunityNarrativeInput, ConflictResolutionInput, ConflictingEdge, EntityDescriptionInput,
    RecallSynthesisInput, SessionTitleInput,
};
use crate::config_supernode::{PrivacyLevel, WorkerConfig};

// ============================================
// Constants
// ============================================

const MAX_RESULT_LEN: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WritebackRetry {
    ReuseStagedResult,
    RecomputeInference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TaskWritebackError {
    reason: &'static str,
    retry: WritebackRetry,
}

impl TaskWritebackError {
    const fn reuse(reason: &'static str) -> Self {
        Self {
            reason,
            retry: WritebackRetry::ReuseStagedResult,
        }
    }

    const fn recompute(reason: &'static str) -> Self {
        Self {
            reason,
            retry: WritebackRetry::RecomputeInference,
        }
    }
}

// ============================================
// TaskWorker
// ============================================

pub struct TaskWorker {
    storage: Arc<MemoryStorage>,
    router: Arc<LlmRouter>,
    batch_size: usize,
    poll_interval: Duration,
    task_timeout: Duration,
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
            task_timeout: Duration::from_secs(worker_config.task_timeout_secs.max(1)),
        }
    }

    pub async fn run(self, mut shutdown_rx: broadcast::Receiver<()>) {
        info!(
            batch_size = self.batch_size,
            poll_interval_secs = self.poll_interval.as_secs(),
            task_timeout_secs = self.task_timeout.as_secs(),
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
                    if self.process_batch_until_shutdown(&mut shutdown_rx).await {
                        break;
                    }
                }
            }
        }

        info!("[TASK_WORKER] Stopped");
    }

    /// Process one claimed batch while retaining ownership of every child task.
    ///
    /// Returns `true` when shutdown was observed while the batch was active.
    async fn process_batch_until_shutdown(
        &self,
        shutdown_rx: &mut broadcast::Receiver<()>,
    ) -> bool {
        let tasks = self.storage.claim_pending_tasks(self.batch_size).await;
        if tasks.is_empty() {
            debug!("[TASK_WORKER] No pending tasks");
            return false;
        }

        info!(count = tasks.len(), "[TASK_WORKER] Processing batch");

        // [SUPERNODE-TASK-OWNERSHIP 2026-08-14 by Codex] JoinSet provides
        // structured ownership: cancelling the worker can no longer detach
        // provider/writeback tasks into the runtime. The Tokio task ID maps a
        // panic or cancellation back to its durable cognitive task row.
        let mut running = JoinSet::new();
        let mut task_ids: HashMap<Id, i64> = HashMap::with_capacity(tasks.len());
        for task in tasks {
            let task_id = task.id;
            let storage = Arc::clone(&self.storage);
            let router = Arc::clone(&self.router);
            let task_timeout = self.task_timeout;
            let abort_handle = running.spawn(async move {
                let timed_out =
                    tokio::time::timeout(task_timeout, Self::process_task(storage, router, task))
                        .await
                        .is_err();
                (task_id, timed_out)
            });
            task_ids.insert(abort_handle.id(), task_id);
        }

        while !running.is_empty() {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    let claimed_ids: Vec<i64> = task_ids.values().copied().collect();
                    running.abort_all();
                    while running.join_next().await.is_some() {}

                    match self
                        .storage
                        .release_processing_tasks(&claimed_ids, "task_worker_shutdown")
                        .await
                    {
                        Ok(released) => info!(
                            released,
                            "[TASK_WORKER] Released unfinished tasks during shutdown"
                        ),
                        Err(release_error) => {
                            warn!(
                                reason = "task_shutdown_release_failed",
                                "[TASK_WORKER] Could not release unfinished tasks"
                            );
                            debug!(error = %release_error, "[TASK_WORKER] Shutdown release detail");
                        }
                    }
                    return true;
                }
                joined = running.join_next_with_id() => {
                    let Some(joined) = joined else {
                        break;
                    };
                    match joined {
                        Ok((runtime_id, (task_id, timed_out))) => {
                            task_ids.remove(&runtime_id);
                            if timed_out {
                                warn!(
                                    id = task_id,
                                    timeout_secs = self.task_timeout.as_secs(),
                                    reason = "task_worker_timed_out",
                                    "[TASK_WORKER] Task exceeded execution deadline"
                                );
                                self.record_aborted_task(task_id, "task_worker_timed_out").await;
                            }
                        }
                        Err(join_error) => {
                            let runtime_id = join_error.id();
                            let Some(task_id) = task_ids.remove(&runtime_id) else {
                                error!(
                                    reason = "task_runtime_id_unmapped",
                                    "[TASK_WORKER] Child task failed without durable ownership mapping"
                                );
                                continue;
                            };
                            let reason = if join_error.is_panic() {
                                "task_worker_panicked"
                            } else {
                                "task_worker_cancelled"
                            };
                            error!(id = task_id, reason, "[TASK_WORKER] Task execution aborted");
                            self.record_aborted_task(task_id, reason).await;
                        }
                    }
                }
            }
        }

        false
    }

    async fn record_aborted_task(&self, task_id: i64, reason: &'static str) {
        if let Err(fail_error) = self.storage.fail_task(task_id, reason).await {
            warn!(
                id = task_id,
                reason = "task_failure_transition_rejected",
                "[TASK_WORKER] Could not record aborted task"
            );
            debug!(id = task_id, error = %fail_error, "[TASK_WORKER] Failure transition detail");
        }
    }

    #[cfg(test)]
    async fn process_batch(&self) {
        let (_shutdown_tx, mut shutdown_rx) = broadcast::channel(1);
        assert!(
            !self.process_batch_until_shutdown(&mut shutdown_rx).await,
            "test batch unexpectedly observed shutdown"
        );
    }

    async fn process_task(
        storage: Arc<MemoryStorage>,
        router: Arc<LlmRouter>,
        task: CognitiveTaskRow,
    ) {
        let start = Instant::now();
        let task_id = task.id;
        let task_type_str = task.task_type.as_str();

        debug!(
            id = task_id,
            task_type = task_type_str,
            "[TASK_WORKER] Processing"
        );

        // v2.5.0+Unify: Use CognitiveTaskType::parse() (defined in config_supernode.rs)
        let task_type = match CognitiveTaskType::parse(task_type_str) {
            Some(t) => t,
            None => {
                warn!(
                    id = task_id,
                    task_type = task_type_str,
                    "[TASK_WORKER] Unknown task type"
                );
                let _ = storage.fail_task(task_id, "unknown_task_type").await;
                return;
            }
        };

        let payload: serde_json::Value = match serde_json::from_str(&task.payload) {
            Ok(v) => v,
            Err(_) => {
                warn!(
                    id = task_id,
                    reason = "invalid_task_payload",
                    "[TASK_WORKER] Invalid payload"
                );
                let _ = storage.fail_task(task_id, "invalid_task_payload").await;
                return;
            }
        };

        // [SUPERNODE-DURABLE-WRITEBACK 2026-08-14 by Codex] A staged result is
        // the durable provider-call boundary. Retries after writeback failure or
        // process restart reuse it, preventing duplicate inference and billing.
        let mut reused_staged_result = task.result.is_some();
        let mut result_stored = task.result.clone().unwrap_or_default();
        let mut provider_used = task.provider_used.clone().unwrap_or_default();
        let mut model_used = task.model_used.clone().unwrap_or_default();
        let mut input_tokens = 0_u32;
        let mut output_tokens = 0_u32;
        let mut latency_ms = 0_u64;

        if !reused_staged_result {
            // v2.5.0+Unify: Use PrivacyLevel::from_str() (defined in config_supernode.rs)
            let privacy = PrivacyLevel::from_str(task.privacy_level.as_str());
            let chat_req = match Self::build_prompt_for_task(&task_type, &payload, privacy).await {
                Ok(req) => req,
                Err(_) => {
                    warn!(
                        id = task_id,
                        reason = "prompt_build_failed",
                        "[TASK_WORKER] Prompt build failed"
                    );
                    let _ = storage.fail_task(task_id, "prompt_build_failed").await;
                    return;
                }
            };

            let resp = match router.route(&task_type, &chat_req).await {
                Ok(r) => r,
                Err(e) => {
                    let reason = e.reason_code();
                    warn!(id = task_id, reason, "[TASK_WORKER] LLM call failed");
                    let _ = storage.fail_task(task_id, reason).await;
                    return;
                }
            };

            latency_ms = start.elapsed().as_millis() as u64;
            input_tokens = resp.usage.input_tokens;
            output_tokens = resp.usage.output_tokens;
            provider_used = resp.provider_name.clone();
            model_used = resp.model_used.clone();

            let cleaned = clean_llm_response(&resp.content, &task_type);
            result_stored = truncate_utf8(&cleaned, MAX_RESULT_LEN).to_string();
            let token_usage_json = serde_json::json!({
                "input": input_tokens,
                "output": output_tokens,
                "cached": resp.usage.cached_tokens,
            })
            .to_string();
            let usage = TaskUsageRecord {
                provider: &provider_used,
                model: &model_used,
                input_tokens: input_tokens as i64,
                output_tokens: output_tokens as i64,
                cached_tokens: resp.usage.cached_tokens as i64,
                latency_ms: latency_ms as i64,
            };

            match storage
                .stage_task_result_and_usage(task_id, &result_stored, &token_usage_json, &usage)
                .await
            {
                Ok(StageTaskResult::Stored) => {}
                Ok(StageTaskResult::AlreadyStored) => {
                    let Some(staged) = storage.get_task(task_id).await else {
                        let _ = storage.fail_task(task_id, "task_result_stage_lost").await;
                        return;
                    };
                    let Some(staged_result) = staged.result else {
                        let _ = storage.fail_task(task_id, "task_result_stage_lost").await;
                        return;
                    };
                    result_stored = staged_result;
                    reused_staged_result = true;
                }
                Err(error) => {
                    warn!(
                        id = task_id,
                        reason = "task_result_stage_failed",
                        "[TASK_WORKER] Could not stage inference result"
                    );
                    debug!(id = task_id, error = %error, "[TASK_WORKER] Result stage detail");
                    let _ = storage.fail_task(task_id, "task_result_stage_failed").await;
                    return;
                }
            }
        } else {
            debug!(
                id = task_id,
                "[TASK_WORKER] Reusing staged inference result for writeback"
            );
        }

        match (task.target_table.as_deref(), task.target_id.as_deref()) {
            (Some(table), Some(target_id)) => {
                if let Err(error) = Self::write_back(
                    &storage,
                    &task_type,
                    table,
                    target_id,
                    &result_stored,
                    &payload,
                )
                .await
                {
                    warn!(
                        id = task_id,
                        reason = error.reason,
                        retry = ?error.retry,
                        "[TASK_WORKER] Durable writeback failed"
                    );
                    let _ = match error.retry {
                        WritebackRetry::ReuseStagedResult => {
                            storage.fail_task(task_id, error.reason).await
                        }
                        WritebackRetry::RecomputeInference => {
                            storage
                                .fail_task_and_discard_result(task_id, error.reason)
                                .await
                        }
                    };
                    return;
                }
            }
            (None, None) => {}
            _ => {
                let reason = "task_target_metadata_incomplete";
                warn!(
                    id = task_id,
                    reason, "[TASK_WORKER] Invalid target metadata"
                );
                let _ = storage.fail_task(task_id, reason).await;
                return;
            }
        }

        if let Err(error) = storage.complete_staged_task(task_id).await {
            let reason = "task_completion_failed";
            warn!(
                id = task_id,
                reason, "[TASK_WORKER] Completion transition failed"
            );
            debug!(id = task_id, error = %error, "[TASK_WORKER] Completion detail");
            let _ = storage.fail_task(task_id, reason).await;
            return;
        }

        info!(
            id = task_id,
            task_type = task_type_str,
            provider = %provider_used,
            model = %model_used,
            input_tokens,
            output_tokens,
            latency_ms,
            reused_staged_result,
            "[TASK_WORKER] ✅ Complete"
        );
    }

    // ============================================
    // Prompt Builders — variant names from config_supernode::CognitiveTaskType
    // ============================================

    async fn build_prompt_for_task(
        task_type: &CognitiveTaskType,
        payload: &serde_json::Value,
        privacy: PrivacyLevel,
    ) -> Result<ChatRequest, String> {
        let messages = match task_type {
            CognitiveTaskType::SessionTitle => {
                let entity_names_raw: Vec<String> = payload["entity_names"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                let entity_refs: Vec<&str> = entity_names_raw.iter().map(|s| s.as_str()).collect();

                build_session_title(&SessionTitleInput {
                    entity_names: &entity_refs,
                    project_name: payload["project_name"].as_str(),
                    first_user_message: payload["first_user_message"].as_str(),
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::CommunityNarrative => {
                let community_name = payload["community_name"]
                    .as_str()
                    .unwrap_or("unknown community");
                let members_raw: Vec<(String, String, i64)> = payload["members"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| {
                        Some((
                            v["name"].as_str()?.to_string(),
                            v["type"].as_str().unwrap_or("entity").to_string(),
                            v["mention_count"].as_i64().unwrap_or(1),
                        ))
                    })
                    .collect();
                let member_refs: Vec<(&str, &str, i64)> = members_raw
                    .iter()
                    .map(|(n, t, c)| (n.as_str(), t.as_str(), *c))
                    .collect();
                let edges_raw: Vec<(String, String, String)> = payload["key_edges"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| {
                        Some((
                            v["source"].as_str()?.to_string(),
                            v["relation"].as_str()?.to_string(),
                            v["target"].as_str()?.to_string(),
                        ))
                    })
                    .collect();
                let edge_refs: Vec<(&str, &str, &str)> = edges_raw
                    .iter()
                    .map(|(s, r, t)| (s.as_str(), r.as_str(), t.as_str()))
                    .collect();

                build_community_narrative(&CommunityNarrativeInput {
                    community_name,
                    members: &member_refs,
                    key_edges: &edge_refs,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::RecallSynthesis => {
                let entity_names_raw: Vec<String> = payload["entity_names"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
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

            CognitiveTaskType::ConflictResolution => {
                let edge_ids: Vec<i64> = payload["conflict_edge_ids"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_i64())
                    .collect();
                let edges_raw: Vec<ConflictingEdge> = payload["edges"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| {
                        Some(ConflictingEdge {
                            edge_id: v["edge_id"].as_i64()?,
                            source: v["source"].as_str().unwrap_or("").to_string(),
                            relation: v["relation"].as_str().unwrap_or("").to_string(),
                            target: v["target"].as_str().unwrap_or("").to_string(),
                            valid_from: v["valid_from"].as_i64().unwrap_or(0),
                            fact_text: v["fact_text"].as_str().map(String::from),
                            confidence: v["confidence"].as_f64(),
                        })
                    })
                    .collect();

                build_conflict_resolution(&ConflictResolutionInput {
                    conflict_edge_ids: &edge_ids,
                    edges: &edges_raw,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::CodeAnalysis => {
                let tags_raw: Vec<String> = payload["existing_tags"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                let tag_refs: Vec<&str> = tags_raw.iter().map(|s| s.as_str()).collect();

                build_code_analysis(&CodeAnalysisInput {
                    artifact_id: payload["artifact_id"].as_str().unwrap_or(""),
                    language: payload["language"].as_str().unwrap_or("unknown"),
                    line_count: payload["line_count"].as_i64(),
                    code_content: payload["code_content"].as_str().unwrap_or(""),
                    existing_tags: &tag_refs,
                    privacy_level: privacy,
                })
            }

            CognitiveTaskType::EntityDescription => {
                let relations_raw: Vec<(String, String)> = payload["relations"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .filter_map(|v| {
                        Some((
                            v["relation_type"].as_str()?.to_string(),
                            v["other_name"].as_str().unwrap_or("").to_string(),
                        ))
                    })
                    .collect();
                let rel_refs: Vec<(&str, &str)> = relations_raw
                    .iter()
                    .map(|(r, n)| (r.as_str(), n.as_str()))
                    .collect();

                build_entity_description(&EntityDescriptionInput {
                    entity_name: payload["entity_name"]
                        .as_str()
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
    ) -> Result<(), TaskWritebackError> {
        let expected_table = match task_type {
            CognitiveTaskType::SessionTitle | CognitiveTaskType::RecallSynthesis => "sessions",
            CognitiveTaskType::CommunityNarrative => "communities",
            CognitiveTaskType::ConflictResolution => "knowledge_edges",
            CognitiveTaskType::CodeAnalysis => "artifacts",
            CognitiveTaskType::EntityDescription => "entities",
        };
        if target_table != expected_table {
            return Err(TaskWritebackError::reuse("task_target_table_mismatch"));
        }

        match task_type {
            CognitiveTaskType::SessionTitle => {
                let title = result
                    .trim_matches(|c| c == '"' || c == '\'' || c == '`')
                    .trim();
                if title.is_empty() {
                    return Err(TaskWritebackError::recompute("session_title_empty"));
                }
                let conn = storage.conn_lock().await;
                let affected = conn
                    .execute(
                        "UPDATE sessions SET title = ?1 WHERE session_id = ?2",
                        rusqlite::params![title, target_id],
                    )
                    .map_err(|error| {
                        debug!(error = %error, "[TASK_WORKER] Session title writeback detail");
                        TaskWritebackError::reuse("session_title_writeback_failed")
                    })?;
                if affected != 1 {
                    return Err(TaskWritebackError::reuse("session_title_target_not_found"));
                }
                debug!(writeback = "session_title", "[TASK_WORKER] Target updated");
            }

            CognitiveTaskType::CommunityNarrative => {
                let summary = result.trim();
                if summary.is_empty() {
                    return Err(TaskWritebackError::recompute("community_narrative_empty"));
                }
                // Write summary directly via SQL — avoids needing get_community()
                // which doesn't exist on MemoryStorage. upsert_community requires
                // owner + name which we don't have in the task context, so direct
                // UPDATE is the correct approach for writeback.
                let conn = storage.conn_lock().await;
                let affected = conn
                    .execute(
                        "UPDATE communities SET summary = ?1 WHERE community_id = ?2",
                        rusqlite::params![summary, target_id],
                    )
                    .map_err(|error| {
                        debug!(error = %error, "[TASK_WORKER] Community writeback detail");
                        TaskWritebackError::reuse("community_narrative_writeback_failed")
                    })?;
                if affected != 1 {
                    return Err(TaskWritebackError::reuse(
                        "community_narrative_target_not_found",
                    ));
                }
                debug!(
                    writeback = "community_narrative",
                    "[TASK_WORKER] Target updated"
                );
            }

            CognitiveTaskType::RecallSynthesis => {
                let parsed = parse_json_result(result);
                let summary = parsed["summary"].as_str().unwrap_or(result).trim();
                let key_decisions = parsed["key_decisions"].as_str();
                if summary.is_empty() {
                    return Err(TaskWritebackError::recompute("recall_synthesis_empty"));
                }
                let conn = storage.conn_lock().await;
                let affected = conn
                    .execute(
                        "UPDATE sessions SET
                            summary = ?1, key_decisions = ?2, summary_generated = 1
                         WHERE session_id = ?3",
                        rusqlite::params![summary, key_decisions, target_id],
                    )
                    .map_err(|error| {
                        debug!(error = %error, "[TASK_WORKER] Recall writeback detail");
                        TaskWritebackError::reuse("recall_synthesis_writeback_failed")
                    })?;
                if affected != 1 {
                    return Err(TaskWritebackError::reuse(
                        "recall_synthesis_target_not_found",
                    ));
                }
                debug!(
                    writeback = "recall_synthesis",
                    "[TASK_WORKER] Target updated"
                );
            }

            CognitiveTaskType::ConflictResolution => {
                let parsed = parse_json_result(result);
                let keep_id = parsed["keep_edge_id"].as_i64().ok_or_else(|| {
                    TaskWritebackError::recompute("conflict_resolution_invalid_result")
                })?;
                let edge_ids: Vec<i64> = payload["conflict_edge_ids"]
                    .as_array()
                    .ok_or_else(|| {
                        TaskWritebackError::reuse("conflict_resolution_invalid_payload")
                    })?
                    .iter()
                    .filter_map(serde_json::Value::as_i64)
                    .collect();
                if edge_ids.is_empty() || !edge_ids.contains(&keep_id) {
                    return Err(TaskWritebackError::reuse(
                        "conflict_resolution_invalid_payload",
                    ));
                }

                // [SUPERNODE-DURABLE-WRITEBACK 2026-08-14 by Codex] Validate
                // and invalidate the complete conflict set in one transaction;
                // partial graph mutation must never escape a failed attempt.
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                let mut conn = storage.conn_lock().await;
                let tx = conn.transaction().map_err(|error| {
                    debug!(error = %error, "[TASK_WORKER] Conflict transaction detail");
                    TaskWritebackError::reuse("conflict_resolution_writeback_failed")
                })?;
                let keep_exists: bool = tx
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM knowledge_edges WHERE edge_id = ?1)",
                        rusqlite::params![keep_id],
                        |row| row.get(0),
                    )
                    .map_err(|error| {
                        debug!(error = %error, "[TASK_WORKER] Conflict validation detail");
                        TaskWritebackError::reuse("conflict_resolution_writeback_failed")
                    })?;
                if !keep_exists {
                    return Err(TaskWritebackError::reuse(
                        "conflict_resolution_target_not_found",
                    ));
                }
                for edge_id in edge_ids.into_iter().filter(|edge_id| *edge_id != keep_id) {
                    let affected = tx
                        .execute(
                            "UPDATE knowledge_edges SET valid_until = ?1, updated_at = ?1
                             WHERE edge_id = ?2",
                            rusqlite::params![now, edge_id],
                        )
                        .map_err(|error| {
                            debug!(error = %error, "[TASK_WORKER] Conflict update detail");
                            TaskWritebackError::reuse("conflict_resolution_writeback_failed")
                        })?;
                    if affected != 1 {
                        return Err(TaskWritebackError::reuse(
                            "conflict_resolution_target_not_found",
                        ));
                    }
                }
                tx.commit().map_err(|error| {
                    debug!(error = %error, "[TASK_WORKER] Conflict commit detail");
                    TaskWritebackError::reuse("conflict_resolution_writeback_failed")
                })?;
                debug!(
                    writeback = "conflict_resolution",
                    "[TASK_WORKER] Target updated"
                );
            }

            CognitiveTaskType::CodeAnalysis => {
                let parsed = parse_json_result(result);
                let description = parsed["description"].as_str().unwrap_or(result).trim();
                if description.is_empty() {
                    return Err(TaskWritebackError::recompute("code_analysis_empty"));
                }
                let conn = storage.conn_lock().await;
                let affected = conn
                    .execute(
                        "UPDATE artifacts SET description = ?1 WHERE artifact_id = ?2",
                        rusqlite::params![description, target_id],
                    )
                    .map_err(|error| {
                        debug!(error = %error, "[TASK_WORKER] Code writeback detail");
                        TaskWritebackError::reuse("code_analysis_writeback_failed")
                    })?;
                if affected != 1 {
                    return Err(TaskWritebackError::reuse("code_analysis_target_not_found"));
                }
                debug!(writeback = "code_analysis", "[TASK_WORKER] Target updated");
            }

            CognitiveTaskType::EntityDescription => {
                let desc = result.trim_matches(|c| c == '"' || c == '\'').trim();
                if desc.is_empty() {
                    return Err(TaskWritebackError::recompute("entity_description_empty"));
                }
                let conn = storage.conn_lock().await;
                let affected = conn
                    .execute(
                        "UPDATE entities SET description = ?1,
                            updated_at = strftime('%s', 'now') WHERE entity_id = ?2",
                        rusqlite::params![desc, target_id],
                    )
                    .map_err(|error| {
                        debug!(error = %error, "[TASK_WORKER] Entity writeback detail");
                        TaskWritebackError::reuse("entity_description_writeback_failed")
                    })?;
                if affected != 1 {
                    return Err(TaskWritebackError::reuse(
                        "entity_description_target_not_found",
                    ));
                }
                debug!(
                    writeback = "entity_description",
                    "[TASK_WORKER] Target updated"
                );
            }
        }

        Ok(())
    }
}

// ============================================
// LLM Response Cleaner
// ============================================

/// Return at most `max_bytes` without splitting a UTF-8 scalar value.
fn truncate_utf8(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }

    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn clean_llm_response(raw: &str, task_type: &CognitiveTaskType) -> String {
    let after_think = strip_think_tags(raw);
    let trimmed = after_think.trim();

    // JSON task types: return as-is, let parse_json_result handle
    // v2.5.0+Unify: RecallSynthesis (was NaturalSummary), ConflictResolution + CodeAnalysis
    // (were CustomPrompt) produce JSON output
    if matches!(
        task_type,
        CognitiveTaskType::RecallSynthesis
            | CognitiveTaskType::ConflictResolution
            | CognitiveTaskType::CodeAnalysis
    ) {
        return trimmed.to_string();
    }

    let after_preamble = strip_preamble(trimmed);

    match task_type {
        CognitiveTaskType::SessionTitle => {
            let last_line = after_preamble
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .last()
                .unwrap_or(after_preamble.trim());
            last_line
                .trim_matches(|c| c == '"' || c == '\'' || c == '`')
                .trim()
                .to_string()
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
        if after.contains("<think>") {
            return strip_think_tags(after);
        }
        return after.trim_start().to_string();
    }
    if let Some(start_pos) = text.find("<think>") {
        return text[..start_pos].trim_end().to_string();
    }
    text.to_string()
}

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
        "Here is a summary: ",
        "Here is a summary:",
        "Based on the conversation: ",
        "Based on the conversation,",
        "Certainly! Here's",
        "Certainly, here's",
        "Of course! Here's",
        "Sure! Here's",
    ];
    let trimmed = text.trim();
    for p in PREAMBLES {
        if let Some(s) = trimmed.strip_prefix(p) {
            return s.trim();
        }
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
            if let Ok(v) = serde_json::from_str(inner) {
                return v;
            }
        }
    }
    let json_str = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim()
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };
    serde_json::from_str(json_str).unwrap_or_else(|_| {
        // [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex] The old preview
        // both copied model output into operator logs and sliced arbitrary UTF-8
        // at byte 200, which could panic while handling malformed JSON.
        warn!(
            reason = "llm_result_json_invalid",
            "[TASK_WORKER] JSON parse failed"
        );
        serde_json::Value::Null
    })
}

impl std::fmt::Debug for TaskWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskWorker")
            .field("batch_size", &self.batch_size)
            .field("poll_interval_secs", &self.poll_interval.as_secs())
            .field("task_timeout_secs", &self.task_timeout.as_secs())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::llm_provider::{ChatResponse, LlmError, LlmProvider, TokenUsage};
    use super::*;
    use crate::config_supernode::TaskRoutingConfig;

    struct PanickingProvider;

    struct CountingProvider {
        calls: AtomicUsize,
    }

    struct RecoveringProvider {
        calls: AtomicUsize,
    }

    struct PendingProvider;

    #[async_trait::async_trait]
    impl LlmProvider for PanickingProvider {
        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            panic!("intentional task-worker recovery test panic");
        }

        fn name(&self) -> &str {
            "panic-provider"
        }

        fn default_model(&self) -> &str {
            "panic-model"
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for CountingProvider {
        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                content: "Durable session title".to_string(),
                usage: TokenUsage {
                    input_tokens: 13,
                    output_tokens: 4,
                    cached_tokens: 2,
                },
                model_used: "counting-model".to_string(),
                provider_name: "counting-provider".to_string(),
                latency_ms: 1,
            })
        }

        fn name(&self) -> &str {
            "counting-provider"
        }

        fn default_model(&self) -> &str {
            "counting-model"
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for RecoveringProvider {
        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            let attempt = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                content: if attempt == 0 {
                    String::new()
                } else {
                    "Recovered title".to_string()
                },
                usage: TokenUsage {
                    input_tokens: 5,
                    output_tokens: u32::from(attempt > 0),
                    cached_tokens: 0,
                },
                model_used: "recovering-model".to_string(),
                provider_name: "recovering-provider".to_string(),
                latency_ms: 1,
            })
        }

        fn name(&self) -> &str {
            "recovering-provider"
        }

        fn default_model(&self) -> &str {
            "recovering-model"
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for PendingProvider {
        async fn chat(&self, _req: &ChatRequest) -> Result<ChatResponse, LlmError> {
            std::future::pending().await
        }

        fn name(&self) -> &str {
            "pending-provider"
        }

        fn default_model(&self) -> &str {
            "pending-model"
        }
    }

    fn pending_router() -> Arc<LlmRouter> {
        Arc::new(LlmRouter::new(
            vec![(
                "pending-provider".into(),
                "http://localhost".into(),
                "pending-model".into(),
                Arc::new(PendingProvider),
            )],
            TaskRoutingConfig::default(),
        ))
    }

    #[test]
    fn malformed_unicode_json_is_rejected_without_byte_boundary_panic() {
        let malformed = "界".repeat(100);
        assert!(parse_json_result(&malformed).is_null());
    }

    #[test]
    fn result_truncation_preserves_utf8_boundaries() {
        assert_eq!(truncate_utf8("界界", 5), "界");
        assert_eq!(truncate_utf8("short", MAX_RESULT_LEN), "short");
    }

    #[tokio::test]
    async fn panicked_task_is_requeued_with_stable_reason() {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                1,
                r#"{"entity_names":[]}"#,
                None,
                None,
                None,
                "structured",
                2,
            )
            .await
            .unwrap()
            .unwrap();
        let router = Arc::new(LlmRouter::new(
            vec![(
                "panic-provider".into(),
                "http://localhost".into(),
                "panic-model".into(),
                Arc::new(PanickingProvider),
            )],
            TaskRoutingConfig::default(),
        ));
        let worker = TaskWorker::new(
            Arc::clone(&storage),
            router,
            WorkerConfig {
                max_concurrent: 1,
                ..WorkerConfig::default()
            },
        );

        worker.process_batch().await;

        let task = storage.get_task(task_id).await.unwrap();
        assert_eq!(task.status, "pending");
        assert_eq!(task.retry_count, 1);
        assert_eq!(task.error_message.as_deref(), Some("task_worker_panicked"));
    }

    #[tokio::test(start_paused = true)]
    async fn timed_out_task_is_requeued_with_stable_reason() {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                1,
                r#"{"entity_names":[]}"#,
                None,
                None,
                None,
                "structured",
                2,
            )
            .await
            .unwrap()
            .unwrap();
        let worker = TaskWorker::new(
            Arc::clone(&storage),
            pending_router(),
            WorkerConfig {
                max_concurrent: 1,
                task_timeout_secs: 1,
                ..WorkerConfig::default()
            },
        );

        worker.process_batch().await;

        let task = storage.get_task(task_id).await.unwrap();
        assert_eq!(task.status, "pending");
        assert_eq!(task.retry_count, 1);
        assert_eq!(task.error_message.as_deref(), Some("task_worker_timed_out"));
    }

    #[tokio::test]
    async fn shutdown_aborts_children_and_releases_claims_without_retry_penalty() {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                1,
                r#"{"entity_names":[]}"#,
                None,
                None,
                None,
                "structured",
                2,
            )
            .await
            .unwrap()
            .unwrap();
        let worker = Arc::new(TaskWorker::new(
            Arc::clone(&storage),
            pending_router(),
            WorkerConfig {
                max_concurrent: 1,
                task_timeout_secs: 60,
                ..WorkerConfig::default()
            },
        ));
        let (shutdown_tx, mut shutdown_rx) = broadcast::channel(1);
        let batch = {
            let worker = Arc::clone(&worker);
            tokio::spawn(async move { worker.process_batch_until_shutdown(&mut shutdown_rx).await })
        };

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if storage
                    .get_task(task_id)
                    .await
                    .is_some_and(|task| task.status == "processing")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker must claim the test task");
        shutdown_tx.send(()).unwrap();

        assert!(
            tokio::time::timeout(Duration::from_secs(1), batch)
                .await
                .expect("worker must stop promptly")
                .unwrap(),
            "batch must report that it observed shutdown"
        );
        let released = storage.get_task(task_id).await.unwrap();
        assert_eq!(released.status, "pending");
        assert_eq!(released.retry_count, 0);
        assert_eq!(released.started_at, None);
        assert_eq!(
            released.error_message.as_deref(),
            Some("task_worker_shutdown")
        );
    }

    #[tokio::test]
    async fn writeback_retry_reuses_staged_result_without_rebilling() {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                1,
                r#"{"entity_names":[]}"#,
                None,
                Some("sessions"),
                Some("session-retry"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        let provider = Arc::new(CountingProvider {
            calls: AtomicUsize::new(0),
        });
        let router = Arc::new(LlmRouter::new(
            vec![(
                "counting-provider".into(),
                "http://localhost".into(),
                "counting-model".into(),
                provider.clone() as Arc<dyn LlmProvider>,
            )],
            TaskRoutingConfig::default(),
        ));
        let worker = TaskWorker::new(
            Arc::clone(&storage),
            router,
            WorkerConfig {
                max_concurrent: 1,
                ..WorkerConfig::default()
            },
        );

        worker.process_batch().await;
        let pending = storage.get_task(task_id).await.unwrap();
        assert_eq!(pending.status, "pending");
        assert_eq!(pending.retry_count, 1);
        assert_eq!(
            pending.error_message.as_deref(),
            Some("session_title_target_not_found")
        );
        assert_eq!(
            pending.result.as_deref(),
            Some("Durable session title"),
            "provider output must survive the failed writeback"
        );
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(storage.get_usage_stats(0, 0).await.total_calls, 1);

        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "INSERT INTO sessions (session_id, owner, started_at)
                 VALUES (?1, ?2, ?3)",
                rusqlite::params!["session-retry", vec![0_u8; 32], 1_i64],
            )
            .unwrap();
        }

        worker.process_batch().await;
        let completed = storage.get_task(task_id).await.unwrap();
        assert_eq!(completed.status, "completed");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(storage.get_usage_stats(0, 0).await.total_calls, 1);

        let title: String = {
            let conn = storage.conn_lock().await;
            conn.query_row(
                "SELECT title FROM sessions WHERE session_id = ?1",
                rusqlite::params!["session-retry"],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert_eq!(title, "Durable session title");
    }

    #[tokio::test]
    async fn invalid_provider_output_is_recomputed_on_retry() {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        {
            let conn = storage.conn_lock().await;
            conn.execute(
                "INSERT INTO sessions (session_id, owner, started_at)
                 VALUES (?1, ?2, ?3)",
                rusqlite::params!["session-invalid", vec![0_u8; 32], 1_i64],
            )
            .unwrap();
        }
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                1,
                r#"{"entity_names":[]}"#,
                None,
                Some("sessions"),
                Some("session-invalid"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        let provider = Arc::new(RecoveringProvider {
            calls: AtomicUsize::new(0),
        });
        let router = Arc::new(LlmRouter::new(
            vec![(
                "recovering-provider".into(),
                "http://localhost".into(),
                "recovering-model".into(),
                provider.clone() as Arc<dyn LlmProvider>,
            )],
            TaskRoutingConfig::default(),
        ));
        let worker = TaskWorker::new(
            Arc::clone(&storage),
            router,
            WorkerConfig {
                max_concurrent: 1,
                ..WorkerConfig::default()
            },
        );

        worker.process_batch().await;
        let pending = storage.get_task(task_id).await.unwrap();
        assert_eq!(pending.status, "pending");
        assert_eq!(
            pending.error_message.as_deref(),
            Some("session_title_empty")
        );
        assert!(pending.result.is_none());

        worker.process_batch().await;
        assert_eq!(storage.get_task(task_id).await.unwrap().status, "completed");
        assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
        assert_eq!(storage.get_usage_stats(0, 0).await.total_calls, 2);
    }
}
