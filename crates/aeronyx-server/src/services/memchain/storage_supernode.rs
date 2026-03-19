// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_supernode.rs
// ============================================
//! # Storage SuperNode — Cognitive Task Queue + LLM Usage Log
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Split from storage_ops.rs to house all CRUD for the v2.5.0 LLM task queue
//! and usage tracking tables introduced in Schema v6.
//!
//! ## Main Functionality
//! ### cognitive_tasks CRUD
//! - insert_cognitive_task()      — enqueue a new pending task (idempotent: skips if active duplicate exists)
//! - claim_pending_tasks()        — atomic SELECT + UPDATE to 'processing'
//! - complete_task()              — mark completed + write result + token_usage
//! - fail_task()                  — increment retry_count or mark 'failed'
//! - retry_task()                 — human-initiated reset of failed/cancelled → pending
//! - cancel_task()                — pending → cancelled
//! - get_task()                   — fetch single task by id
//! - get_tasks_by_status()        — list tasks filtered by status
//! - get_tasks_filtered()         — list tasks with optional status + task_type filters
//! - get_tasks_for_target()       — find tasks for a specific (table, id) pair
//! - count_tasks_by_status()      — HashMap<status, count> for queue summary
//! - reset_stale_processing_tasks() — on startup: recover tasks stuck in 'processing'
//!
//! ### llm_usage_log CRUD
//! - insert_usage_log()              — write a single LLM call record
//! - get_usage_stats()               — aggregate stats for a time window (by provider)
//! - get_usage_stats_by_task_type()  — two-dimensional breakdown (task_type × provider)
//!
//! ## Architecture
//! Same `impl MemoryStorage` extension pattern as storage_graph.rs, storage_miner.rs,
//! and storage_ops.rs. Rust allows multiple impl blocks across files in the same crate.
//!
//! ## Schema (v6)
//! cognitive_tasks:
//!   status values: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
//!   privacy_level: 'structured' | 'summary' | 'full'
//!   payload: JSON task-specific input
//!   result: JSON LLM output written back after completion
//!   token_usage: JSON TokenUsage serialization
//!
//! llm_usage_log:
//!   cost_usd NOT stored — fee rates change; compute at query time in LlmRouter.
//!
//! ⚠️ Important Note for Next Developer:
//! - insert_cognitive_task() is NOW IDEMPOTENT. It skips insertion if an active
//!   (pending/processing) task already exists for the same (target_table, target_id,
//!   task_type) triple. Returns Ok(None) in that case. Callers must handle Option<i64>.
//! - reset_stale_processing_tasks() MUST be called at server startup before TaskWorker
//!   spawns. Recovers tasks stuck in 'processing' after a crash/restart.
//! - claim_pending_tasks() uses a single conn lock (SELECT + UPDATE atomic).
//!   Do NOT split into two separate conn.lock() calls.
//! - fail_task() checks retry_count < max_retries before marking 'failed'.
//!   retry_task() is the human override — always resets to pending regardless.
//! - get_tasks_filtered() uses dynamic SQL with 4 variants. The (None,None) case
//!   passes "" as unused params because rusqlite requires the same param count
//!   for a prepared statement. This is safe: unused ?1/?2 are never referenced
//!   in the None branch SQL.
//! - get_usage_stats_by_task_type() JOINs cognitive_tasks — tasks without a
//!   task_id in llm_usage_log (e.g. manual inserts) are excluded.
//!
//! ## Dependencies
//! - storage.rs — MemoryStorage struct
//!
//! ## Depended by
//! - task_worker.rs — claim_pending_tasks / complete_task / fail_task
//! - miner/reflection.rs — insert_cognitive_task (Phase B)
//! - api/supernode_handlers.rs — all management endpoints
//! - api/mpi_handlers.rs — count_tasks_by_status for /status
//! - server.rs — reset_stale_processing_tasks on startup
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase A - 🌟 Created. Core CRUD.
//! v2.5.0+SuperNode Phase C - 🔧 Fixed by_provider borrow issue in get_usage_stats.
//! v2.5.0+SuperNode Phase D - 🌟 Added retry_task, count_tasks_by_status,
//!   get_usage_stats_by_task_type, get_tasks_filtered, TaskTypeUsage type.
//! v2.5.0+Fix              - 🔧 [BUG FIX] insert_cognitive_task: added idempotency
//!   guard (WHERE NOT EXISTS) to prevent duplicate pending/processing tasks for
//!   the same (target_table, target_id, task_type). Return type changed from
//!   Result<i64> to Result<Option<i64>> — Ok(None) = skipped (duplicate).
//!                         - 🔧 [BUG FIX] Added reset_stale_processing_tasks() for
//!   crash-recovery on server startup. Tasks stuck in 'processing' beyond the
//!   timeout threshold are reset to 'pending'.
//!                         - 🧪 Added test_insert_cognitive_task_dedup and
//!   test_get_usage_stats_by_task_type (previously untested).

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension};
use tracing::{debug, info, warn};

use super::storage::MemoryStorage;

// ============================================
// Row Types
// ============================================

/// Full cognitive task row.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CognitiveTaskRow {
    pub id: i64,
    pub task_type: String,
    pub priority: i64,
    pub status: String,
    pub payload: String,
    pub result: Option<String>,
    pub target_table: Option<String>,
    pub target_id: Option<String>,
    pub privacy_level: String,
    pub provider_used: Option<String>,
    pub model_used: Option<String>,
    pub token_usage: Option<String>,
    pub created_at: i64,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub retry_count: i64,
    pub max_retries: i64,
    pub error_message: Option<String>,
}

/// LLM usage statistics for a time window (by provider).
#[derive(Debug, Clone, serde::Serialize)]
pub struct LlmUsageStats {
    pub window_start: i64,
    pub window_end: i64,
    pub total_calls: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_cached_tokens: i64,
    pub avg_latency_ms: f64,
    pub by_provider: Vec<ProviderUsage>,
}

/// Per-provider usage aggregation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProviderUsage {
    pub provider: String,
    pub calls: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub avg_latency_ms: f64,
}

/// Per-(task_type, provider) usage aggregation (v2.5.0+SuperNode Phase D).
#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskTypeUsage {
    pub task_type: String,
    pub provider: String,
    pub calls: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_tokens: i64,
    pub avg_latency_ms: f64,
}

// ============================================
// impl MemoryStorage — cognitive_tasks CRUD
// ============================================

impl MemoryStorage {
    /// Enqueue a new cognitive task in 'pending' status.
    ///
    /// ## Idempotency Guard (v2.5.0+Fix)
    /// Returns `Ok(None)` (skips insertion) if an active task — status IN
    /// ('pending', 'processing') — already exists for the same
    /// `(target_table, target_id, task_type)` triple.
    ///
    /// This prevents Miner ticks from accumulating duplicate tasks for the
    /// same target object across repeated cycles.
    ///
    /// ## Return Values
    /// - `Ok(Some(id))` — new task inserted, `id` is the new row id
    /// - `Ok(None)`     — active duplicate found, insertion skipped
    /// - `Err(msg)`     — database error
    ///
    /// ⚠️ Callers in reflection.rs must be updated to handle `Option<i64>`.
    pub async fn insert_cognitive_task(
        &self,
        task_type: &str,
        priority: i64,
        payload: &str,
        prompt_messages: Option<&str>,
        target_table: Option<&str>,
        target_id: Option<&str>,
        privacy_level: &str,
        max_retries: i64,
    ) -> Result<Option<i64>, String> {
        let now = now_ts();
        let conn = self.conn.lock().await;

        // ── Idempotency check ──────────────────────────────────────────────
        // Only guard when both target_table and target_id are provided.
        // Tasks without a target (e.g. one-off recall_synthesis) always insert.
        if let (Some(tbl), Some(tid)) = (target_table, target_id) {
            let exists: bool = conn.query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM cognitive_tasks
                    WHERE target_table = ?1
                      AND target_id   = ?2
                      AND task_type   = ?3
                      AND status IN ('pending', 'processing')
                )",
                params![tbl, tid, task_type],
                |row| row.get(0),
            ).unwrap_or(false);

            if exists {
                debug!(
                    task_type = task_type,
                    target = %format!("{}/{}", tbl, tid),
                    "[STORAGE_SN] Duplicate active task — skipped"
                );
                return Ok(None);
            }
        }

        // ── Insert ────────────────────────────────────────────────────────
        conn.execute(
            "INSERT INTO cognitive_tasks
                (task_type, priority, status, payload, prompt_messages,
                 target_table, target_id, privacy_level, max_retries, created_at)
             VALUES (?1, ?2, 'pending', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                task_type, priority, payload, prompt_messages,
                target_table, target_id, privacy_level, max_retries, now,
            ],
        ).map_err(|e| format!("Insert cognitive_task: {}", e))?;

        let id = conn.last_insert_rowid();
        debug!(id = id, task_type = task_type, "[STORAGE_SN] Task enqueued");
        Ok(Some(id))
    }

    /// Recover tasks stuck in 'processing' after a server crash or restart.
    ///
    /// ## When to Call
    /// Call ONCE at server startup, before TaskWorker spawns. Typical location:
    /// `server.rs::init_llm_router()` or immediately before `TaskWorker::spawn()`.
    ///
    /// ## Logic
    /// Any task that has been in 'processing' for longer than `timeout_secs`
    /// is assumed to have been orphaned by a crash. It is reset to 'pending'
    /// so the TaskWorker can re-claim it.
    ///
    /// The `retry_count` is NOT incremented here — the crash was not the task's
    /// fault. The error_message is set to indicate the recovery event for audit.
    ///
    /// ## Parameters
    /// - `timeout_secs` — should match `supernode.worker.task_timeout_secs` from config.
    ///   Recommended: 120–300 seconds.
    ///
    /// Returns the number of tasks recovered.
    pub async fn reset_stale_processing_tasks(&self, timeout_secs: i64) -> usize {
        let now = now_ts();
        let stale_before = now - timeout_secs;

        let conn = self.conn.lock().await;
        match conn.execute(
            "UPDATE cognitive_tasks
             SET status        = 'pending',
                 started_at    = NULL,
                 error_message = 'Recovered: task was in processing at server startup'
             WHERE status     = 'processing'
               AND started_at < ?1",
            params![stale_before],
        ) {
            Ok(n) => {
                if n > 0 {
                    info!(
                        recovered = n,
                        timeout_secs = timeout_secs,
                        "[STORAGE_SN] Recovered stale processing tasks on startup"
                    );
                }
                n
            }
            Err(e) => {
                warn!("[STORAGE_SN] reset_stale_processing_tasks failed: {}", e);
                0
            }
        }
    }

    /// Atomically claim up to `batch_size` pending tasks for processing.
    ///
    /// Uses a single connection lock (SELECT + UPDATE atomic).
    /// ⚠️ Do NOT split into two conn.lock() calls — would allow double-claiming.
    pub async fn claim_pending_tasks(&self, batch_size: usize) -> Vec<CognitiveTaskRow> {
        let now = now_ts();
        let conn = self.conn.lock().await;

        let mut stmt = match conn.prepare(
            "SELECT id, task_type, priority, status, payload, result,
                    target_table, target_id, privacy_level, provider_used, model_used,
                    token_usage, created_at, started_at, completed_at,
                    retry_count, max_retries, error_message
             FROM cognitive_tasks
             WHERE status = 'pending' AND retry_count < max_retries
             ORDER BY priority DESC, created_at ASC
             LIMIT ?1"
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("[STORAGE_SN] claim_pending_tasks prepare failed: {}", e);
                return Vec::new();
            }
        };

        let tasks: Vec<CognitiveTaskRow> = stmt
            .query_map(params![batch_size as i64], |row| Ok(task_row(row)?))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        if tasks.is_empty() {
            return tasks;
        }

        for task in &tasks {
            let _ = conn.execute(
                "UPDATE cognitive_tasks SET status = 'processing', started_at = ?1
                 WHERE id = ?2 AND status = 'pending'",
                params![now, task.id],
            );
        }

        debug!(claimed = tasks.len(), "[STORAGE_SN] Tasks claimed");
        tasks
    }

    /// Mark a task as completed.
    pub async fn complete_task(
        &self,
        task_id: i64,
        result: &str,
        provider_used: &str,
        model_used: &str,
        token_usage_json: &str,
    ) -> Result<(), String> {
        let now = now_ts();
        let conn = self.conn.lock().await;
        conn.execute(
            "UPDATE cognitive_tasks SET
                status = 'completed', result = ?1,
                provider_used = ?2, model_used = ?3,
                token_usage = ?4, completed_at = ?5
             WHERE id = ?6",
            params![result, provider_used, model_used, token_usage_json, now, task_id],
        ).map_err(|e| format!("complete_task {}: {}", task_id, e))?;
        debug!(id = task_id, "[STORAGE_SN] Task completed");
        Ok(())
    }

    /// Record a task failure. Resets to 'pending' if retries remain, else 'failed'.
    pub async fn fail_task(&self, task_id: i64, error_message: &str) -> Result<(), String> {
        let conn = self.conn.lock().await;
        let (retry_count, max_retries): (i64, i64) = conn.query_row(
            "SELECT retry_count, max_retries FROM cognitive_tasks WHERE id = ?1",
            params![task_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).map_err(|e| format!("fail_task fetch {}: {}", task_id, e))?;

        let new_count = retry_count + 1;
        let new_status = if new_count >= max_retries { "failed" } else { "pending" };

        conn.execute(
            "UPDATE cognitive_tasks SET
                status = ?1, retry_count = ?2,
                error_message = ?3, started_at = NULL
             WHERE id = ?4",
            params![new_status, new_count, error_message, task_id],
        ).map_err(|e| format!("fail_task update {}: {}", task_id, e))?;

        debug!(id = task_id, retries = new_count, status = new_status, "[STORAGE_SN] Task failed");
        Ok(())
    }

    /// Human-initiated retry: reset failed/cancelled task to pending.
    ///
    /// Unlike fail_task() (worker-called), this is the management API override.
    /// Increments retry_count by 1 (audit trail), but always allows the reset
    /// regardless of retry_count vs max_retries.
    /// Clears error_message and started_at for a clean attempt.
    pub async fn retry_task(&self, task_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().await;

        let (status, retry_count): (String, i64) = conn.query_row(
            "SELECT status, retry_count FROM cognitive_tasks WHERE id = ?1",
            params![task_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).map_err(|e| format!("retry_task fetch {}: {}", task_id, e))?;

        if status != "failed" && status != "cancelled" {
            return Err(format!(
                "Task {} is '{}', can only retry 'failed' or 'cancelled'",
                task_id, status
            ));
        }

        conn.execute(
            "UPDATE cognitive_tasks SET
                status = 'pending', retry_count = ?1,
                error_message = NULL, started_at = NULL
             WHERE id = ?2",
            params![retry_count + 1, task_id],
        ).map_err(|e| format!("retry_task update {}: {}", task_id, e))?;

        debug!(id = task_id, new_retry_count = retry_count + 1, "[STORAGE_SN] Task queued for retry");
        Ok(())
    }

    /// Cancel a pending task (pending → cancelled). No-op if not pending.
    pub async fn cancel_task(&self, task_id: i64) -> Result<usize, String> {
        let conn = self.conn.lock().await;
        let affected = conn.execute(
            "UPDATE cognitive_tasks SET status = 'cancelled'
             WHERE id = ?1 AND status = 'pending'",
            params![task_id],
        ).map_err(|e| format!("cancel_task {}: {}", task_id, e))?;
        debug!(id = task_id, affected = affected, "[STORAGE_SN] Task cancel attempted");
        Ok(affected)
    }
    /// Get a single task by id.
    pub async fn get_task(&self, task_id: i64) -> Option<CognitiveTaskRow> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT id, task_type, priority, status, payload, result,
                    target_table, target_id, privacy_level, provider_used, model_used,
                    token_usage, created_at, started_at, completed_at,
                    retry_count, max_retries, error_message
             FROM cognitive_tasks WHERE id = ?1",
            params![task_id],
            |row| task_row(row),
        ).optional().unwrap_or(None)
    }

    /// List tasks filtered by status, newest first.
    pub async fn get_tasks_by_status(&self, status: &str, limit: usize) -> Vec<CognitiveTaskRow> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT id, task_type, priority, status, payload, result,
                    target_table, target_id, privacy_level, provider_used, model_used,
                    token_usage, created_at, started_at, completed_at,
                    retry_count, max_retries, error_message
             FROM cognitive_tasks
             WHERE status = ?1
             ORDER BY created_at DESC
             LIMIT ?2"
        ) { Ok(s) => s, Err(_) => return Vec::new() };

        stmt.query_map(params![status, limit as i64], |row| task_row(row))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// List tasks with optional status and task_type filters (Phase D).
    ///
    /// Either filter may be None (wildcard). Ordered by priority DESC, created_at ASC.
    pub async fn get_tasks_filtered(
        &self,
        status: Option<&str>,
        task_type: Option<&str>,
        limit: usize,
    ) -> Vec<CognitiveTaskRow> {
        let conn = self.conn.lock().await;
        let limit_i = limit.min(100) as i64;

        // Four SQL variants to avoid dynamic string building
        match (status, task_type) {
            (Some(s), Some(t)) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     WHERE status = ?1 AND task_type = ?2
                     ORDER BY priority DESC, created_at ASC LIMIT ?3"
                ) { Ok(s) => s, Err(_) => return Vec::new() };
                stmt.query_map(params![s, t, limit_i], |row| task_row(row))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    .unwrap_or_default()
            }
            (Some(s), None) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     WHERE status = ?1
                     ORDER BY priority DESC, created_at ASC LIMIT ?2"
                ) { Ok(s) => s, Err(_) => return Vec::new() };
                stmt.query_map(params![s, limit_i], |row| task_row(row))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    .unwrap_or_default()
            }
            (None, Some(t)) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     WHERE task_type = ?1
                     ORDER BY priority DESC, created_at ASC LIMIT ?2"
                ) { Ok(s) => s, Err(_) => return Vec::new() };
                stmt.query_map(params![t, limit_i], |row| task_row(row))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    .unwrap_or_default()
            }
            (None, None) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     ORDER BY priority DESC, created_at ASC LIMIT ?1"
                ) { Ok(s) => s, Err(_) => return Vec::new() };
                stmt.query_map(params![limit_i], |row| task_row(row))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    .unwrap_or_default()
            }
        }
    }

    /// Find tasks for a specific (target_table, target_id) pair.
    pub async fn get_tasks_for_target(
        &self,
        target_table: &str,
        target_id: &str,
        status_filter: Option<&str>,
    ) -> Vec<CognitiveTaskRow> {
        let conn = self.conn.lock().await;

        match status_filter {
            Some(s) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     WHERE target_table = ?1 AND target_id = ?2 AND status = ?3
                     ORDER BY created_at DESC"
                ) { Ok(s) => s, Err(_) => return Vec::new() };
                stmt.query_map(params![target_table, target_id, s], |row| task_row(row))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    .unwrap_or_default()
            }
            None => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     WHERE target_table = ?1 AND target_id = ?2
                     ORDER BY created_at DESC"
                ) { Ok(s) => s, Err(_) => return Vec::new() };
                stmt.query_map(params![target_table, target_id], |row| task_row(row))
                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                    .unwrap_or_default()
            }
        }
    }

    /// Get task counts grouped by status (Phase D).
    ///
    /// Returns HashMap<status, count>. All 5 expected statuses always present.
    /// Used by /status and /supernode/health for queue summary.
    pub async fn count_tasks_by_status(&self) -> HashMap<String, i64> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT status, COUNT(*) FROM cognitive_tasks GROUP BY status"
        ) {
            Ok(s) => s,
            Err(_) => return HashMap::new(),
        };

        let raw: HashMap<String, i64> = stmt
            .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default();

        // Ensure all expected statuses are present (prevents key-not-found panics in callers)
        let mut result = HashMap::new();
        for s in &["pending", "processing", "completed", "failed", "cancelled"] {
            result.insert(s.to_string(), *raw.get(*s).unwrap_or(&0));
        }
        result
    }
}

// ============================================
// impl MemoryStorage — llm_usage_log CRUD
// ============================================

impl MemoryStorage {
    /// Record a single LLM call.
    pub async fn insert_usage_log(
        &self,
        task_id: Option<i64>,
        provider: &str,
        model: &str,
        input_tokens: i64,
        output_tokens: i64,
        cached_tokens: i64,
        latency_ms: i64,
    ) -> Result<(), String> {
        let now = now_ts();
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO llm_usage_log
                (task_id, provider, model, input_tokens, output_tokens,
                 cached_tokens, latency_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![task_id, provider, model, input_tokens, output_tokens,
                    cached_tokens, latency_ms, now],
        ).map_err(|e| format!("insert_usage_log: {}", e))?;
        Ok(())
    }

    /// Get aggregated usage stats by provider for a time window.
    ///
    /// `since` = 0 means all time. `until` = 0 means now.
    pub async fn get_usage_stats(&self, since: i64, until: i64) -> LlmUsageStats {
        let now = now_ts();
        let since = since.max(0);
        let until = if until == 0 { now } else { until };

        let conn = self.conn.lock().await;

        let (total_calls, total_input, total_output, total_cached, avg_latency):
            (i64, i64, i64, i64, f64) = conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                    COALESCE(SUM(cached_tokens),0), COALESCE(AVG(latency_ms),0.0)
             FROM llm_usage_log
             WHERE created_at >= ?1 AND created_at <= ?2",
            params![since, until],
            |row| Ok((
                row.get(0).unwrap_or(0),
                row.get(1).unwrap_or(0),
                row.get(2).unwrap_or(0),
                row.get(3).unwrap_or(0),
                row.get(4).unwrap_or(0.0),
            )),
        ).unwrap_or((0, 0, 0, 0, 0.0));

        // By-provider breakdown — isolated stmt scope to avoid borrow conflict
        let by_provider: Vec<ProviderUsage> = {
            let mut stmt = match conn.prepare(
                "SELECT provider, COUNT(*), SUM(input_tokens), SUM(output_tokens), AVG(latency_ms)
                 FROM llm_usage_log
                 WHERE created_at >= ?1 AND created_at <= ?2
                 GROUP BY provider
                 ORDER BY COUNT(*) DESC"
            ) {
                Ok(s) => s,
                Err(e) => {
                    warn!("[STORAGE_SN] get_usage_stats by_provider prepare failed: {}", e);
                    return LlmUsageStats {
                        window_start: since, window_end: until,
                        total_calls, total_input_tokens: total_input,
                        total_output_tokens: total_output, total_cached_tokens: total_cached,
                        avg_latency_ms: avg_latency, by_provider: Vec::new(),
                    };
                }
            };
            stmt.query_map(params![since, until], |row| {
                Ok(ProviderUsage {
                    provider: row.get(0)?,
                    calls: row.get(1)?,
                    input_tokens: row.get(2)?,
                    output_tokens: row.get(3)?,
                    avg_latency_ms: row.get(4).unwrap_or(0.0),
                })
            })
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
        };

        LlmUsageStats {
            window_start: since, window_end: until,
            total_calls, total_input_tokens: total_input,
            total_output_tokens: total_output, total_cached_tokens: total_cached,
            avg_latency_ms: avg_latency, by_provider,
        }
    }

    /// Get usage stats aggregated by both task_type AND provider (Phase D).
    ///
    /// Two-dimensional breakdown for the management UI.
    /// Tasks without a task_id in llm_usage_log are excluded (JOIN filters them).
    pub async fn get_usage_stats_by_task_type(
        &self, since: i64, until: i64,
    ) -> Vec<TaskTypeUsage> {
        let now = now_ts();
        let since = since.max(0);
        let until = if until == 0 { now } else { until };

        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT ct.task_type, ul.provider,
                    COUNT(*), SUM(ul.input_tokens), SUM(ul.output_tokens),
                    SUM(ul.cached_tokens), AVG(ul.latency_ms)
             FROM llm_usage_log ul
             JOIN cognitive_tasks ct ON ct.id = ul.task_id
             WHERE ul.created_at >= ?1 AND ul.created_at <= ?2
             GROUP BY ct.task_type, ul.provider
             ORDER BY COUNT(*) DESC"
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("[STORAGE_SN] get_usage_stats_by_task_type prepare failed: {}", e);
                return Vec::new();
            }
        };

        stmt.query_map(params![since, until], |row| {
            Ok(TaskTypeUsage {
                task_type: row.get(0)?,
                provider: row.get(1)?,
                calls: row.get(2)?,
                input_tokens: row.get(3)?,
                output_tokens: row.get(4)?,
                cached_tokens: row.get(5)?,
                avg_latency_ms: row.get(6).unwrap_or(0.0),
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

// ============================================
// Private helpers
// ============================================

fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

/// Map a rusqlite Row to CognitiveTaskRow (shared across all SELECT queries).
fn task_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CognitiveTaskRow> {
    Ok(CognitiveTaskRow {
        id: row.get(0)?,
        task_type: row.get(1)?,
        priority: row.get(2)?,
        status: row.get(3)?,
        payload: row.get(4)?,
        result: row.get(5)?,
        target_table: row.get(6)?,
        target_id: row.get(7)?,
        privacy_level: row.get(8)?,
        provider_used: row.get(9)?,
        model_used: row.get(10)?,
        token_usage: row.get(11)?,
        created_at: row.get(12)?,
        started_at: row.get(13)?,
        completed_at: row.get(14)?,
        retry_count: row.get(15)?,
        max_retries: row.get(16)?,
        error_message: row.get(17)?,
    })
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_insert_and_claim_task() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s.insert_cognitive_task(
            "session_title", 5, r#"{"session_id":"sess_001"}"#,
            None, Some("sessions"), Some("sess_001"), "structured", 3,
        ).await.unwrap().expect("Should insert first task");
        assert!(id > 0);

        let claimed = s.claim_pending_tasks(10).await;
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].id, id);

        // Double-claim prevention
        let claimed2 = s.claim_pending_tasks(10).await;
        assert!(claimed2.is_empty());
    }

    /// (v2.5.0+Fix) Verify idempotency: inserting the same (target_table, target_id,
    /// task_type) while an active task exists must return Ok(None).
    #[tokio::test]
    async fn test_insert_cognitive_task_dedup() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        // First insert — should succeed
        let id1 = s.insert_cognitive_task(
            "session_title", 5, r#"{"session_id":"sess_001"}"#,
            None, Some("sessions"), Some("sess_001"), "structured", 3,
        ).await.unwrap();
        assert!(id1.is_some(), "First insert must succeed");

        // Second insert — same triple, status=pending → must be skipped
        let id2 = s.insert_cognitive_task(
            "session_title", 5, r#"{"session_id":"sess_001"}"#,
            None, Some("sessions"), Some("sess_001"), "structured", 3,
        ).await.unwrap();
        assert!(id2.is_none(), "Duplicate pending task must be skipped");

        // Only one task should exist
        let tasks = s.get_tasks_filtered(None, Some("session_title"), 10).await;
        assert_eq!(tasks.len(), 1);

        // Different task_type on same target → allowed
        let id3 = s.insert_cognitive_task(
            "code_analysis", 5, r#"{}"#,
            None, Some("sessions"), Some("sess_001"), "structured", 3,
        ).await.unwrap();
        assert!(id3.is_some(), "Different task_type on same target must be allowed");

        // After completion, same type may be re-inserted
        let id1_unwrapped = id1.unwrap();
        s.claim_pending_tasks(10).await;
        s.complete_task(id1_unwrapped, r#"{"title":"Done"}"#, "openai", "gpt-4o-mini", "{}").await.unwrap();

        let id4 = s.insert_cognitive_task(
            "session_title", 5, r#"{"session_id":"sess_001"}"#,
            None, Some("sessions"), Some("sess_001"), "structured", 3,
        ).await.unwrap();
        assert!(id4.is_some(), "Re-insert after completion must succeed");
    }

    /// Tasks with no target (recall_synthesis style) always insert regardless.
    #[tokio::test]
    async fn test_insert_no_target_always_inserts() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        let id1 = s.insert_cognitive_task(
            "recall_synthesis", 5, r#"{}"#,
            None, None, None, "structured", 3,
        ).await.unwrap();
        let id2 = s.insert_cognitive_task(
            "recall_synthesis", 5, r#"{}"#,
            None, None, None, "structured", 3,
        ).await.unwrap();

        assert!(id1.is_some());
        assert!(id2.is_some());
        assert_ne!(id1.unwrap(), id2.unwrap());
    }

    /// (v2.5.0+Fix) Verify startup recovery resets stale processing tasks.
    #[tokio::test]
    async fn test_reset_stale_processing_tasks() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        // Insert and claim a task (moves to 'processing')
        let id = s.insert_cognitive_task(
            "session_title", 5, r#"{}"#,
            None, Some("sessions"), Some("s1"), "structured", 3,
        ).await.unwrap().unwrap();
        s.claim_pending_tasks(1).await;

        // Manually back-date started_at to simulate a stale task
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE cognitive_tasks SET started_at = started_at - 1000 WHERE id = ?1",
                params![id],
            ).unwrap();
        }

        let recovered = s.reset_stale_processing_tasks(60).await;
        assert_eq!(recovered, 1);

        let task = s.get_task(id).await.unwrap();
        assert_eq!(task.status, "pending");
        assert_eq!(task.retry_count, 0, "retry_count must NOT be incremented by recovery");
        assert!(task.error_message.unwrap_or_default().contains("Recovered"));
    }

    /// Within-timeout tasks must NOT be reset.
    #[tokio::test]
    async fn test_reset_stale_skips_fresh_processing() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task(
            "session_title", 5, r#"{}"#,
            None, Some("sessions"), Some("s1"), "structured", 3,
        ).await.unwrap();
        s.claim_pending_tasks(1).await;

        // started_at is just now — should NOT be reset with 120s timeout
        let recovered = s.reset_stale_processing_tasks(120).await;
        assert_eq!(recovered, 0);
    }

    #[tokio::test]
    async fn test_complete_task() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s.insert_cognitive_task(
            "session_title", 5, r#"{"session_id":"s1"}"#,
            None, Some("sessions"), Some("s1"), "structured", 3,
        ).await.unwrap().unwrap();
        s.claim_pending_tasks(1).await;
        s.complete_task(id, r#"{"title":"Project Alpha: JWT"}"#,
            "openai", "gpt-4o-mini",
            r#"{"input":50,"output":10,"cached":0}"#
        ).await.unwrap();
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "completed");
        assert!(t.result.is_some());
    }

    #[tokio::test]
    async fn test_fail_task_retries() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s.insert_cognitive_task(
            "community_summary", 3, r#"{}"#,
            None, None, None, "structured", 2,
        ).await.unwrap().unwrap();
        s.claim_pending_tasks(1).await;

        // First fail → back to pending
        s.fail_task(id, "timeout").await.unwrap();
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "pending");
        assert_eq!(t.retry_count, 1);

        // Second fail → failed (retry_count=2 >= max_retries=2)
        s.claim_pending_tasks(1).await;
        s.fail_task(id, "timeout again").await.unwrap();
        let t2 = s.get_task(id).await.unwrap();
        assert_eq!(t2.status, "failed");
    }

    #[tokio::test]
    async fn test_retry_task_resets_to_pending() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s.insert_cognitive_task(
            "session_title", 5, r#"{}"#,
            None, None, None, "structured", 1,
        ).await.unwrap().unwrap();
        s.claim_pending_tasks(1).await;
        // Exhaust retries
        s.fail_task(id, "error").await.unwrap();
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "failed");

        // Human retry override
        s.retry_task(id).await.unwrap();
        let t2 = s.get_task(id).await.unwrap();
        assert_eq!(t2.status, "pending");
        assert_eq!(t2.retry_count, 2); // incremented, not reset
    }

    #[tokio::test]
    async fn test_retry_task_rejects_non_failed() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s.insert_cognitive_task(
            "session_title", 5, r#"{}"#,
            None, None, None, "structured", 3,
        ).await.unwrap().unwrap();
        // Task is pending — retry should fail
        let result = s.retry_task(id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("pending"));
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s.insert_cognitive_task(
            "entity_description", 5, r#"{}"#,
            None, None, None, "structured", 3,
        ).await.unwrap().unwrap();
        s.cancel_task(id).await.unwrap();
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "cancelled");

        let claimed = s.claim_pending_tasks(10).await;
        assert!(claimed.is_empty());
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task("t", 2, "{}", None, None, None, "structured", 3).await.unwrap();
        s.insert_cognitive_task("t", 9, "{}", None, None, None, "structured", 3).await.unwrap();
        s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3).await.unwrap();

        let claimed = s.claim_pending_tasks(3).await;
        assert_eq!(claimed.len(), 3);
        assert_eq!(claimed[0].priority, 9);
        assert_eq!(claimed[1].priority, 5);
        assert_eq!(claimed[2].priority, 2);
    }

    #[tokio::test]
    async fn test_get_tasks_filtered() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3).await.unwrap();
        s.insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3).await.unwrap();
        s.insert_cognitive_task("community_summary", 3, "{}", None, None, None, "structured", 3).await.unwrap();

        // Filter by task_type only
        let title_tasks = s.get_tasks_filtered(None, Some("session_title"), 10).await;
        assert_eq!(title_tasks.len(), 2);

        // Filter by status only
        let pending = s.get_tasks_filtered(Some("pending"), None, 10).await;
        assert_eq!(pending.len(), 3);

        // Filter by both
        let both = s.get_tasks_filtered(Some("pending"), Some("community_summary"), 10).await;
        assert_eq!(both.len(), 1);

        // No filter
        let all = s.get_tasks_filtered(None, None, 10).await;
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_count_tasks_by_status() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3).await.unwrap();
        s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3).await.unwrap();
        let id3 = s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3).await.unwrap().unwrap();
        s.cancel_task(id3).await.unwrap();

        let counts = s.count_tasks_by_status().await;
        assert_eq!(counts["pending"], 2);
        assert_eq!(counts["cancelled"], 1);
        assert_eq!(counts["failed"], 0); // always present even if 0
    }

    #[tokio::test]
    async fn test_insert_usage_log_and_stats() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_usage_log(Some(1), "openai", "gpt-4o-mini", 100, 50, 0, 850).await.unwrap();
        s.insert_usage_log(Some(2), "openai", "gpt-4o-mini", 200, 80, 20, 1200).await.unwrap();
        s.insert_usage_log(Some(3), "anthropic", "claude-haiku", 150, 60, 0, 950).await.unwrap();

        let stats = s.get_usage_stats(0, 0).await;
        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.total_input_tokens, 450);
        assert_eq!(stats.total_output_tokens, 190);
        assert_eq!(stats.total_cached_tokens, 20);
        assert_eq!(stats.by_provider.len(), 2);
        assert_eq!(stats.by_provider[0].provider, "openai");
    }

    /// (v2.5.0+Fix) Test get_usage_stats_by_task_type (previously untested).
    #[tokio::test]
    async fn test_get_usage_stats_by_task_type() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        // Insert tasks first so the JOIN resolves
        let t1 = s.insert_cognitive_task(
            "session_title", 5, "{}", None, None, None, "structured", 3,
        ).await.unwrap().unwrap();
        let t2 = s.insert_cognitive_task(
            "community_narrative", 5, "{}", None, None, None, "structured", 3,
        ).await.unwrap().unwrap();

        s.insert_usage_log(Some(t1), "deepseek", "deepseek-reasoner", 100, 40, 0, 800).await.unwrap();
        s.insert_usage_log(Some(t1), "deepseek", "deepseek-reasoner", 120, 50, 0, 900).await.unwrap();
        s.insert_usage_log(Some(t2), "claude", "claude-sonnet-4-20250514", 200, 80, 30, 1100).await.unwrap();

        let stats = s.get_usage_stats_by_task_type(0, 0).await;
        assert_eq!(stats.len(), 2);

        // session_title × deepseek should be first (2 calls vs 1)
        let st = stats.iter().find(|r| r.task_type == "session_title").unwrap();
        assert_eq!(st.provider, "deepseek");
        assert_eq!(st.calls, 2);
        assert_eq!(st.input_tokens, 220);

        let cn = stats.iter().find(|r| r.task_type == "community_narrative").unwrap();
        assert_eq!(cn.provider, "claude");
        assert_eq!(cn.calls, 1);
        assert_eq!(cn.cached_tokens, 30);
    }

    #[tokio::test]
    async fn test_get_tasks_for_target() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task(
            "session_title", 5, "{}", None,
            Some("sessions"), Some("sess_001"), "structured", 3,
        ).await.unwrap();

        let tasks = s.get_tasks_for_target("sessions", "sess_001", None).await;
        assert_eq!(tasks.len(), 1);

        let pending = s.get_tasks_for_target("sessions", "sess_001", Some("pending")).await;
        assert_eq!(pending.len(), 1);

        let none = s.get_tasks_for_target("sessions", "sess_999", None).await;
        assert!(none.is_empty());
    }
}
