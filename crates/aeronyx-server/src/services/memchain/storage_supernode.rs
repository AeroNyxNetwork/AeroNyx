// ============================================
// File: crates/aeronyx-server/src/services/memchain/storage_supernode.rs
// ============================================
//! # Storage SuperNode — Cognitive Task Queue + LLM Usage Log
//!
//! ## Creation Reason (v2.5.0+SuperNode)
//! Split from storage_ops.rs to house all CRUD for the v2.5.0 LLM task queue
//! and usage tracking tables introduced in Schema v6.
//!
//! ## Modification Reason (v2.5.2+Pagination)
//! Added `offset: usize` to `get_tasks_filtered()` for cursor-free pagination.
//! SQL updated from `LIMIT ?N` to `LIMIT ?N OFFSET ?M` across all four variants.
//! Caller: `supernode_handlers.rs::supernode_list_tasks()` passes `params.offset`.
//!
//! ## Main Functionality
//! ### cognitive_tasks CRUD
//! - insert_cognitive_task()        — enqueue a new pending task (idempotent: skips if active duplicate exists)
//! - claim_pending_tasks()          — transactional claim to 'processing'
//! - release_processing_tasks()     — release worker-owned tasks during shutdown
//! - stage_task_result_and_usage()  — atomically persist inference + usage before writeback
//! - complete_staged_task()         — finalize a staged task after durable writeback
//! - complete_task()                — mark completed + write result + token_usage
//! - fail_task()                    — increment retry_count or mark 'failed'
//! - retry_task()                   — human-initiated reset of failed/cancelled → pending
//! - cancel_task()                  — pending → cancelled
//! - get_task()                     — fetch single task by id
//! - get_tasks_by_status()          — list tasks filtered by status
//! - get_tasks_filtered()           — list tasks with optional status + task_type filters (paginated)
//! - get_tasks_for_target()         — find tasks for a specific (table, id) pair
//! - count_tasks_by_status()        — HashMap<status, count> for queue summary
//! - reset_stale_processing_tasks() — recover tasks stuck in 'processing'
//! - recover_and_claim_pending_tasks() — atomically recover expired claims and claim work
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
//! ## Calling Relationships
//! - task_worker.rs          → claim_pending_tasks / complete_task / fail_task
//! - miner/reflection.rs     → insert_cognitive_task (Phase B)
//! - api/supernode_handlers  → all management endpoints; get_tasks_filtered passes offset
//! - api/mpi_handlers.rs     → count_tasks_by_status for /status
//! - server.rs               → reset_stale_processing_tasks on startup
//!
//! ⚠️ Important Notes for Next Developer:
//! - insert_cognitive_task() is IDEMPOTENT. It skips insertion if an active
//!   (pending/processing) task already exists for the same (target_table, target_id,
//!   task_type) triple. The check and insert share an IMMEDIATE transaction so
//!   independent SQLite connections cannot enqueue the same work concurrently.
//!   Returns Ok(None) in that case. Callers must handle Option<i64>.
//! - reset_stale_processing_tasks() MUST be called at server startup before TaskWorker
//!   spawns. The worker also performs recovery in the same transaction as each
//!   claim, so a fresh orphan skipped by the startup timeout cannot remain stuck.
//! - claim_pending_tasks() uses an IMMEDIATE SQLite transaction. A process-local
//!   mutex alone does not prevent another connection or process from claiming
//!   the same row. Do not weaken this to separate autocommit statements.
//! - fail_task() checks retry_count < max_retries before marking 'failed'.
//!   retry_task() is the human override — always resets to pending regardless.
//! - get_tasks_filtered() uses four SQL variants (no dynamic string building).
//!   v2.5.2+Pagination: each variant now appends OFFSET ?M. The offset param is
//!   the last bind param in every variant. Do NOT remove it.
//! - get_usage_stats_by_task_type() JOINs cognitive_tasks — tasks without a
//!   task_id in llm_usage_log (e.g. manual inserts) are excluded.
//!
//! ## Last Modified
//! v2.5.9-EnqueueIdempotency - [SUPERNODE-ENQUEUE-IDEMPOTENCY 2026-08-14 by Codex]
//!   Serialized active-task detection and insertion in one IMMEDIATE transaction,
//!   preventing duplicate model calls across independent node processes.
//! v2.5.8-CrashLeaseRecovery - [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex]
//!   Recovers expired processing claims in the same IMMEDIATE transaction as
//!   the next task claim, closing the fast-crash/restart orphan window.
//! v2.5.7-TaskOwnership - [SUPERNODE-TASK-OWNERSHIP 2026-08-14 by Codex]
//!   Claims now use an IMMEDIATE SQLite transaction across SELECT + UPDATE,
//!   preventing duplicate execution across independent database connections.
//!   Added guarded batch release for structured worker shutdown.
//! v2.5.6-DurableWriteback - [SUPERNODE-DURABLE-WRITEBACK 2026-08-14 by Codex]
//!   Added a two-phase completion boundary so target writeback retries reuse the
//!   staged inference result and cannot duplicate provider calls or usage rows.
//!   [SUPERNODE-MANUAL-RETRY 2026-08-14 by Codex] Restored a claimable retry
//!   budget when an operator explicitly retries a terminal task.
//! v2.5.5-FailureBoundary - [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex]
//!   Made worker failure transitions atomic and conditional on `processing`.
//! v2.5.0+SuperNode Phase A - Created. Core CRUD.
//! v2.5.0+SuperNode Phase C - Fixed by_provider borrow issue in get_usage_stats.
//! v2.5.0+SuperNode Phase D - Added retry_task, count_tasks_by_status,
//!   get_usage_stats_by_task_type, get_tasks_filtered, TaskTypeUsage type.
//! v2.5.0+Fix              - [BUG FIX] insert_cognitive_task: added idempotency
//!   guard (WHERE NOT EXISTS) to prevent duplicate pending/processing tasks for
//!   the same (target_table, target_id, task_type). Return type changed from
//!   Result<i64> to Result<Option<i64>> — Ok(None) = skipped (duplicate).
//!                         - [BUG FIX] Added reset_stale_processing_tasks() for
//!   crash-recovery on server startup. Tasks stuck in 'processing' beyond the
//!   timeout threshold are reset to 'pending'.
//!                         - Added test_insert_cognitive_task_dedup and
//!   test_get_usage_stats_by_task_type (previously untested).
//! v2.5.2+Pagination       - get_tasks_filtered gains `offset: usize` param.
//!   All four SQL variants updated: LIMIT ?N → LIMIT ?N OFFSET ?M.
//!   Added test_get_tasks_filtered_pagination to verify paging correctness.
// ============================================

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, OptionalExtension, TransactionBehavior};
use tracing::{debug, info, warn};

use super::storage::MemoryStorage;

const EXPIRED_CLAIM_REASON: &str = "task_worker_claim_expired";
const RECOVER_STALE_TASKS_SQL: &str = "UPDATE cognitive_tasks
     SET status = 'pending', started_at = NULL, error_message = ?1
     WHERE status = 'processing'
       AND (started_at IS NULL OR started_at <= ?2)";

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

/// Result of an atomic stale-claim recovery and pending-task claim.
pub(crate) struct TaskClaimBatch {
    /// Tasks now durably owned by the caller.
    pub tasks: Vec<CognitiveTaskRow>,
    /// Expired processing rows returned to the pending queue in this transaction.
    pub recovered: usize,
}

/// Outcome of staging an inference result for durable writeback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StageTaskResult {
    /// This call stored both the inference result and its usage row.
    Stored,
    /// A previous attempt already stored the result; no usage row was added.
    AlreadyStored,
}

/// Immutable accounting data written with a staged inference result.
pub(crate) struct TaskUsageRecord<'a> {
    /// Provider that executed the inference.
    pub provider: &'a str,
    /// Concrete provider model used for the inference.
    pub model: &'a str,
    /// Prompt tokens reported by the provider.
    pub input_tokens: i64,
    /// Completion tokens reported by the provider.
    pub output_tokens: i64,
    /// Prompt tokens served from provider cache.
    pub cached_tokens: i64,
    /// End-to-end provider latency in milliseconds.
    pub latency_ms: i64,
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
    /// # Errors
    /// Returns an error if the enqueue transaction cannot begin, query, insert,
    /// or commit. No task identifier is returned before the commit succeeds.
    ///
    /// ⚠️ Callers in reflection.rs must handle `Option<i64>`.
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
        let mut conn = self.conn.lock().await;

        // [SUPERNODE-ENQUEUE-IDEMPOTENCY 2026-08-14 by Codex] A process-local
        // mutex cannot serialize another MemoryStorage connection. Reserving the
        // SQLite writer before the read closes the SELECT-then-INSERT race that
        // could otherwise duplicate provider calls and billing.
        let transaction = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| format!("insert_cognitive_task begin: {e}"))?;

        // ── Idempotency check ──────────────────────────────────────────────
        // Only guard when both target_table and target_id are provided.
        // Tasks without a target (e.g. one-off recall_synthesis) always insert.
        if let (Some(tbl), Some(tid)) = (target_table, target_id) {
            let exists: bool = transaction
                .query_row(
                    "SELECT EXISTS(
                    SELECT 1 FROM cognitive_tasks
                    WHERE target_table = ?1
                      AND target_id   = ?2
                      AND task_type   = ?3
                      AND status IN ('pending', 'processing')
                )",
                    params![tbl, tid, task_type],
                    |row| row.get(0),
                )
                .map_err(|e| format!("insert_cognitive_task duplicate check: {e}"))?;

            if exists {
                transaction
                    .commit()
                    .map_err(|e| format!("insert_cognitive_task duplicate commit: {e}"))?;
                drop(conn);
                debug!(
                    task_type = task_type,
                    target = %format!("{}/{}", tbl, tid),
                    "[STORAGE_SN] Duplicate active task — skipped"
                );
                return Ok(None);
            }
        }

        // ── Insert ────────────────────────────────────────────────────────
        transaction
            .execute(
                "INSERT INTO cognitive_tasks
                (task_type, priority, status, payload, prompt_messages,
                 target_table, target_id, privacy_level, max_retries, created_at)
             VALUES (?1, ?2, 'pending', ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    task_type,
                    priority,
                    payload,
                    prompt_messages,
                    target_table,
                    target_id,
                    privacy_level,
                    max_retries,
                    now,
                ],
            )
            .map_err(|e| format!("insert_cognitive_task insert: {e}"))?;

        let id = transaction.last_insert_rowid();
        transaction
            .commit()
            .map_err(|e| format!("insert_cognitive_task commit: {e}"))?;
        drop(conn);
        debug!(id = id, task_type = task_type, "[STORAGE_SN] Task enqueued");
        Ok(Some(id))
    }

    /// Recover tasks stuck in 'processing' after a worker crash or restart.
    ///
    /// ## When to Call
    /// Call at server startup before TaskWorker spawns. The live worker also uses
    /// [`Self::recover_and_claim_pending_tasks`] so tasks that are still fresh at
    /// a quick restart are recovered once their ownership deadline expires.
    ///
    /// ## Logic
    /// Any task that has been in 'processing' for longer than `timeout_secs`
    /// is assumed to have been orphaned by a crash. It is reset to 'pending'
    /// so the TaskWorker can re-claim it.
    ///
    /// The `retry_count` is NOT incremented here because infrastructure failure
    /// is not a task failure. A stable reason code is retained for audit.
    ///
    /// ## Parameters
    /// - `timeout_secs` — should match `supernode.worker.task_timeout_secs` from config.
    ///   Recommended: 120–300 seconds.
    ///
    /// Returns the number of tasks recovered.
    pub async fn reset_stale_processing_tasks(&self, timeout_secs: i64) -> usize {
        let now = now_ts();
        let stale_before = now.saturating_sub(timeout_secs.max(1));

        let conn = self.conn.lock().await;
        match conn.execute(
            RECOVER_STALE_TASKS_SQL,
            params![EXPIRED_CLAIM_REASON, stale_before],
        ) {
            Ok(n) => {
                if n > 0 {
                    info!(
                        recovered = n,
                        timeout_secs = timeout_secs,
                        "[STORAGE_SN] Recovered stale processing task claims"
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
    /// [SUPERNODE-TASK-OWNERSHIP 2026-08-14 by Codex] The process-local
    /// connection mutex cannot serialize a second `MemoryStorage` instance or
    /// another process opening the same WAL database. `IMMEDIATE` takes the
    /// SQLite writer reservation before the SELECT, so no competing claimant
    /// can observe and transition the same pending rows between these steps.
    pub async fn claim_pending_tasks(&self, batch_size: usize) -> Vec<CognitiveTaskRow> {
        self.claim_pending_tasks_inner(batch_size, None).await.tasks
    }

    /// Atomically recover expired processing claims and claim pending work.
    ///
    /// [SUPERNODE-CRASH-LEASE 2026-08-14 by Codex] Startup-only recovery leaves
    /// a liveness hole when a process crashes immediately after claiming a row:
    /// the quick restart sees a fresh claim, skips it, and never revisits it.
    /// Recovery and claiming share one IMMEDIATE transaction here so an expired
    /// orphan becomes claimable without an intermediate externally visible state.
    pub(crate) async fn recover_and_claim_pending_tasks(
        &self,
        batch_size: usize,
        stale_after_secs: i64,
    ) -> TaskClaimBatch {
        self.claim_pending_tasks_inner(batch_size, Some(stale_after_secs))
            .await
    }

    async fn claim_pending_tasks_inner(
        &self,
        batch_size: usize,
        stale_after_secs: Option<i64>,
    ) -> TaskClaimBatch {
        if batch_size == 0 {
            return TaskClaimBatch {
                tasks: Vec::new(),
                recovered: 0,
            };
        }

        let now = now_ts();
        let limit = i64::try_from(batch_size).unwrap_or(i64::MAX);
        let mut conn = self.conn.lock().await;
        let transaction = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
            Ok(transaction) => transaction,
            Err(e) => {
                warn!(
                    reason = "task_claim_transaction_unavailable",
                    error = %e,
                    "[STORAGE_SN] Could not begin task claim transaction"
                );
                return TaskClaimBatch {
                    tasks: Vec::new(),
                    recovered: 0,
                };
            }
        };

        let recovered = if let Some(stale_after_secs) = stale_after_secs {
            let stale_before = now.saturating_sub(stale_after_secs.max(1));
            match transaction.execute(
                RECOVER_STALE_TASKS_SQL,
                params![EXPIRED_CLAIM_REASON, stale_before],
            ) {
                Ok(recovered) => recovered,
                Err(e) => {
                    warn!(
                        reason = "task_claim_recovery_failed",
                        error = %e,
                        "[STORAGE_SN] Task claim transaction rolled back"
                    );
                    return TaskClaimBatch {
                        tasks: Vec::new(),
                        recovered: 0,
                    };
                }
            }
        } else {
            0
        };

        let candidates: Vec<CognitiveTaskRow> = {
            let mut statement = match transaction.prepare(
                "SELECT id, task_type, priority, status, payload, result,
                        target_table, target_id, privacy_level, provider_used, model_used,
                        token_usage, created_at, started_at, completed_at,
                        retry_count, max_retries, error_message
                 FROM cognitive_tasks
                 WHERE status = 'pending' AND retry_count < max_retries
                 ORDER BY priority DESC, created_at ASC
                 LIMIT ?1",
            ) {
                Ok(statement) => statement,
                Err(e) => {
                    warn!(
                        reason = "task_claim_query_unavailable",
                        error = %e,
                        "[STORAGE_SN] Could not prepare task claim query"
                    );
                    return TaskClaimBatch {
                        tasks: Vec::new(),
                        recovered: 0,
                    };
                }
            };

            let query_result = match statement.query_map(params![limit], task_row) {
                Ok(rows) => match rows.collect::<rusqlite::Result<Vec<_>>>() {
                    Ok(rows) => rows,
                    Err(e) => {
                        warn!(
                            reason = "task_claim_row_decode_failed",
                            error = %e,
                            "[STORAGE_SN] Could not decode task claim candidate"
                        );
                        return TaskClaimBatch {
                            tasks: Vec::new(),
                            recovered: 0,
                        };
                    }
                },
                Err(e) => {
                    warn!(
                        reason = "task_claim_query_failed",
                        error = %e,
                        "[STORAGE_SN] Could not read task claim candidates"
                    );
                    return TaskClaimBatch {
                        tasks: Vec::new(),
                        recovered: 0,
                    };
                }
            };
            query_result
        };

        let mut claimed = Vec::with_capacity(candidates.len());
        for task in candidates {
            let affected = match transaction.execute(
                "UPDATE cognitive_tasks SET status = 'processing', started_at = ?1
                 WHERE id = ?2 AND status = 'pending'",
                params![now, task.id],
            ) {
                Ok(affected) => affected,
                Err(e) => {
                    warn!(
                        id = task.id,
                        reason = "task_claim_transition_failed",
                        error = %e,
                        "[STORAGE_SN] Task claim transaction rolled back"
                    );
                    return TaskClaimBatch {
                        tasks: Vec::new(),
                        recovered: 0,
                    };
                }
            };
            if affected == 1 {
                claimed.push(task);
            }
        }

        if let Err(e) = transaction.commit() {
            warn!(
                reason = "task_claim_commit_failed",
                error = %e,
                "[STORAGE_SN] Task claim transaction did not commit"
            );
            return TaskClaimBatch {
                tasks: Vec::new(),
                recovered: 0,
            };
        }

        debug!(
            claimed = claimed.len(),
            recovered, "[STORAGE_SN] Tasks claimed"
        );
        TaskClaimBatch {
            tasks: claimed,
            recovered,
        }
    }

    /// Release a worker-owned set of processing tasks back to the pending queue.
    ///
    /// This is used only during structured worker shutdown. It deliberately does
    /// not increment `retry_count`: an operator-requested shutdown is not an
    /// inference failure. The status guard makes the operation safe when a task
    /// completed between shutdown notification and task cancellation. Any staged
    /// result remains intact and will be reused after restart.
    pub(crate) async fn release_processing_tasks(
        &self,
        task_ids: &[i64],
        reason: &str,
    ) -> Result<usize, String> {
        if task_ids.is_empty() {
            return Ok(0);
        }

        let mut conn = self.conn.lock().await;
        let transaction = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| format!("release_processing_tasks begin: {e}"))?;
        let mut released = 0;

        for task_id in task_ids {
            released += transaction
                .execute(
                    "UPDATE cognitive_tasks SET
                        status = 'pending', started_at = NULL, error_message = ?1
                     WHERE id = ?2 AND status = 'processing'",
                    params![reason, task_id],
                )
                .map_err(|e| format!("release_processing_tasks update {task_id}: {e}"))?;
        }

        transaction
            .commit()
            .map_err(|e| format!("release_processing_tasks commit: {e}"))?;
        debug!(released, "[STORAGE_SN] Worker-owned tasks released");
        Ok(released)
    }

    /// Mark a task as completed.
    ///
    /// ## P1 SecAudit: status guard added
    /// UPDATE now carries `AND status = 'processing'` to prevent accidentally
    /// marking a cancelled or failed task as completed (e.g. a stale worker
    /// finishing after a human cancel). Returns Err if no rows were affected.
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
        let affected = conn
            .execute(
                "UPDATE cognitive_tasks SET
                status = 'completed', result = ?1,
                provider_used = ?2, model_used = ?3,
                token_usage = ?4, completed_at = ?5
             WHERE id = ?6 AND status = 'processing'",
                params![
                    result,
                    provider_used,
                    model_used,
                    token_usage_json,
                    now,
                    task_id
                ],
            )
            .map_err(|e| format!("complete_task {}: {}", task_id, e))?;

        if affected == 0 {
            return Err(format!(
                "complete_task {}: task not found or not in 'processing' state \
                 (may have been cancelled between claim and completion)",
                task_id
            ));
        }
        debug!(id = task_id, "[STORAGE_SN] Task completed");
        Ok(())
    }

    /// Persist provider output and usage before attempting target writeback.
    ///
    /// [SUPERNODE-DURABLE-WRITEBACK 2026-08-14 by Codex] The result row is the
    /// durable idempotency marker. Result metadata and `llm_usage_log` commit in
    /// one SQLite transaction, so a crash cannot retain one without the other.
    /// Repeated calls for an already-staged processing task are successful no-ops.
    pub(crate) async fn stage_task_result_and_usage(
        &self,
        task_id: i64,
        result: &str,
        token_usage_json: &str,
        usage: &TaskUsageRecord<'_>,
    ) -> Result<StageTaskResult, String> {
        let now = now_ts();
        let mut conn = self.conn.lock().await;
        let tx = conn
            .transaction()
            .map_err(|e| format!("stage_task_result transaction {}: {}", task_id, e))?;

        let state: Option<(String, Option<String>)> = tx
            .query_row(
                "SELECT status, result FROM cognitive_tasks WHERE id = ?1",
                params![task_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| format!("stage_task_result read {}: {}", task_id, e))?;

        let Some((status, existing_result)) = state else {
            return Err(format!("stage_task_result {}: task not found", task_id));
        };
        if status != "processing" {
            return Err(format!(
                "stage_task_result {}: task is not in 'processing' state",
                task_id
            ));
        }
        if existing_result.is_some() {
            return Ok(StageTaskResult::AlreadyStored);
        }

        let affected = tx
            .execute(
                "UPDATE cognitive_tasks SET
                    result = ?1, provider_used = ?2, model_used = ?3,
                    token_usage = ?4, error_message = NULL
                 WHERE id = ?5 AND status = 'processing' AND result IS NULL",
                params![
                    result,
                    usage.provider,
                    usage.model,
                    token_usage_json,
                    task_id
                ],
            )
            .map_err(|e| format!("stage_task_result update {}: {}", task_id, e))?;
        if affected != 1 {
            return Err(format!(
                "stage_task_result {}: guarded update affected {} rows",
                task_id, affected
            ));
        }

        tx.execute(
            "INSERT INTO llm_usage_log
                (task_id, provider, model, input_tokens, output_tokens,
                 cached_tokens, latency_ms, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                task_id,
                usage.provider,
                usage.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_tokens,
                usage.latency_ms,
                now
            ],
        )
        .map_err(|e| format!("stage_task_result usage {}: {}", task_id, e))?;

        tx.commit()
            .map_err(|e| format!("stage_task_result commit {}: {}", task_id, e))?;
        debug!(id = task_id, "[STORAGE_SN] Task result staged");
        Ok(StageTaskResult::Stored)
    }

    /// Finalize a task only after its target writeback succeeds.
    pub(crate) async fn complete_staged_task(&self, task_id: i64) -> Result<(), String> {
        let now = now_ts();
        let conn = self.conn.lock().await;
        let affected = conn
            .execute(
                "UPDATE cognitive_tasks SET
                    status = 'completed', completed_at = ?1, error_message = NULL
                 WHERE id = ?2 AND status = 'processing' AND result IS NOT NULL",
                params![now, task_id],
            )
            .map_err(|e| format!("complete_staged_task {}: {}", task_id, e))?;

        if affected != 1 {
            return Err(format!(
                "complete_staged_task {}: staged processing task not found",
                task_id
            ));
        }
        debug!(id = task_id, "[STORAGE_SN] Staged task completed");
        Ok(())
    }

    /// Record a task failure. Resets to 'pending' if retries remain, else 'failed'.
    pub async fn fail_task(&self, task_id: i64, error_message: &str) -> Result<(), String> {
        self.transition_task_failure(task_id, error_message, true)
            .await
    }

    /// Record a failure caused by invalid provider output and allow re-inference.
    ///
    /// The usage row remains as an accurate record of the provider call, while
    /// staged result metadata is cleared so the next claim does not replay the
    /// same invalid output.
    pub(crate) async fn fail_task_and_discard_result(
        &self,
        task_id: i64,
        error_message: &str,
    ) -> Result<(), String> {
        self.transition_task_failure(task_id, error_message, false)
            .await
    }

    async fn transition_task_failure(
        &self,
        task_id: i64,
        error_message: &str,
        preserve_staged_result: bool,
    ) -> Result<(), String> {
        let conn = self.conn.lock().await;
        // [SUPERNODE-FAILURE-BOUNDARY 2026-08-14 by Codex] Compute the retry
        // transition inside one guarded UPDATE. This prevents a stale worker or
        // panic observer from overwriting a terminal/cancelled task and remains
        // atomic even if another process opens the same SQLite database.
        let affected = conn
            .execute(
                "UPDATE cognitive_tasks SET
                status = CASE
                    WHEN retry_count + 1 >= max_retries THEN 'failed'
                    ELSE 'pending'
                END,
                retry_count = retry_count + 1,
                error_message = ?1,
                started_at = NULL,
                result = CASE WHEN ?3 THEN result ELSE NULL END,
                provider_used = CASE WHEN ?3 THEN provider_used ELSE NULL END,
                model_used = CASE WHEN ?3 THEN model_used ELSE NULL END,
                token_usage = CASE WHEN ?3 THEN token_usage ELSE NULL END
             WHERE id = ?2 AND status = 'processing'",
                params![error_message, task_id, preserve_staged_result],
            )
            .map_err(|e| format!("fail_task update {}: {}", task_id, e))?;

        if affected == 0 {
            return Err(format!(
                "fail_task {}: task not found or not in 'processing' state",
                task_id
            ));
        }

        let (new_count, new_status): (i64, String) = conn
            .query_row(
                "SELECT retry_count, status FROM cognitive_tasks WHERE id = ?1",
                params![task_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|e| format!("fail_task result {}: {}", task_id, e))?;

        debug!(
            id = task_id,
            retries = new_count,
            status = %new_status,
            "[STORAGE_SN] Task failed"
        );
        Ok(())
    }

    /// Human-initiated retry: reset failed/cancelled task to pending.
    ///
    /// Unlike fail_task() (worker-called), this is the management API override.
    /// Increments retry_count by 1 (audit trail), but always allows the reset
    /// regardless of retry_count vs max_retries. `max_retries` is extended when
    /// necessary so `claim_pending_tasks()` can actually claim the reset row.
    /// Clears error_message and started_at for a clean attempt.
    pub async fn retry_task(&self, task_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().await;

        let (status, retry_count): (String, i64) = conn
            .query_row(
                "SELECT status, retry_count FROM cognitive_tasks WHERE id = ?1",
                params![task_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|_| format!("retry_task: task {} not found", task_id))?;

        if status != "failed" && status != "cancelled" {
            return Err(format!(
                "Task {} is '{}', can only retry 'failed' or 'cancelled'",
                task_id, status
            ));
        }

        // [SUPERNODE-MANUAL-RETRY 2026-08-14 by Codex] The previous update
        // incremented `retry_count` without extending the ceiling. A terminal
        // task therefore looked pending but could never satisfy the claim query.
        conn.execute(
            "UPDATE cognitive_tasks SET
                status = 'pending', retry_count = ?1,
                max_retries = MAX(max_retries, ?1 + 1),
                error_message = NULL, started_at = NULL
             WHERE id = ?2",
            params![retry_count + 1, task_id],
        )
        .map_err(|e| format!("retry_task update {}: {}", task_id, e))?;

        debug!(
            id = task_id,
            new_retry_count = retry_count + 1,
            "[STORAGE_SN] Task queued for retry"
        );
        Ok(())
    }

    /// Cancel a pending task (pending → cancelled).
    ///
    /// Returns the number of affected rows (0 or 1).
    /// ⚠️ Callers in supernode_handlers must check for 0 (TOCTOU: task was
    /// claimed between the status check and this UPDATE) and return 409 Conflict.
    pub async fn cancel_task(&self, task_id: i64) -> Result<usize, String> {
        let conn = self.conn.lock().await;
        let affected = conn
            .execute(
                "UPDATE cognitive_tasks SET status = 'cancelled'
             WHERE id = ?1 AND status = 'pending'",
                params![task_id],
            )
            .map_err(|e| format!("cancel_task {}: {}", task_id, e))?;
        debug!(
            id = task_id,
            affected = affected,
            "[STORAGE_SN] Task cancel attempted"
        );
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
        )
        .optional()
        .unwrap_or(None)
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
             LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![status, limit as i64], |row| task_row(row))
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    /// List tasks with optional status and task_type filters, with pagination.
    ///
    /// Either filter may be None (wildcard). Ordered by priority DESC, created_at ASC.
    ///
    /// ## v2.5.2+Pagination
    /// Added `offset: usize` (default 0 for callers that don't paginate).
    /// SQL updated to `LIMIT ?N OFFSET ?M` in all four match arms.
    ///
    /// ⚠️ offset is the last bind parameter in every variant. Do NOT reorder params.
    pub async fn get_tasks_filtered(
        &self,
        status: Option<&str>,
        task_type: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> Vec<CognitiveTaskRow> {
        let conn = self.conn.lock().await;
        let limit_i = limit.min(100) as i64;
        let offset_i = offset as i64;

        match (status, task_type) {
            (Some(s), Some(t)) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, task_type, priority, status, payload, result,
                            target_table, target_id, privacy_level, provider_used, model_used,
                            token_usage, created_at, started_at, completed_at,
                            retry_count, max_retries, error_message
                     FROM cognitive_tasks
                     WHERE status = ?1 AND task_type = ?2
                     ORDER BY priority DESC, created_at ASC LIMIT ?3 OFFSET ?4",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                stmt.query_map(params![s, t, limit_i, offset_i], |row| task_row(row))
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
                     ORDER BY priority DESC, created_at ASC LIMIT ?2 OFFSET ?3",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                stmt.query_map(params![s, limit_i, offset_i], |row| task_row(row))
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
                     ORDER BY priority DESC, created_at ASC LIMIT ?2 OFFSET ?3",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                stmt.query_map(params![t, limit_i, offset_i], |row| task_row(row))
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
                     ORDER BY priority DESC, created_at ASC LIMIT ?1 OFFSET ?2",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
                stmt.query_map(params![limit_i, offset_i], |row| task_row(row))
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
                     ORDER BY created_at DESC",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
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
                     ORDER BY created_at DESC",
                ) {
                    Ok(s) => s,
                    Err(_) => return Vec::new(),
                };
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
        let mut stmt =
            match conn.prepare("SELECT status, COUNT(*) FROM cognitive_tasks GROUP BY status") {
                Ok(s) => s,
                Err(_) => return HashMap::new(),
            };

        let raw: HashMap<String, i64> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
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
            params![
                task_id,
                provider,
                model,
                input_tokens,
                output_tokens,
                cached_tokens,
                latency_ms,
                now
            ],
        )
        .map_err(|e| format!("insert_usage_log: {}", e))?;
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

        let (total_calls, total_input, total_output, total_cached, avg_latency): (
            i64,
            i64,
            i64,
            i64,
            f64,
        ) = conn
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),
                    COALESCE(SUM(cached_tokens),0), COALESCE(AVG(latency_ms),0.0)
             FROM llm_usage_log
             WHERE created_at >= ?1 AND created_at <= ?2",
                params![since, until],
                |row| {
                    Ok((
                        row.get(0).unwrap_or(0),
                        row.get(1).unwrap_or(0),
                        row.get(2).unwrap_or(0),
                        row.get(3).unwrap_or(0),
                        row.get(4).unwrap_or(0.0),
                    ))
                },
            )
            .unwrap_or((0, 0, 0, 0, 0.0));

        // By-provider breakdown — isolated stmt scope to avoid borrow conflict
        let by_provider: Vec<ProviderUsage> =
            {
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
            window_start: since,
            window_end: until,
            total_calls,
            total_input_tokens: total_input,
            total_output_tokens: total_output,
            total_cached_tokens: total_cached,
            avg_latency_ms: avg_latency,
            by_provider,
        }
    }

    /// Get usage stats aggregated by both task_type AND provider (Phase D).
    ///
    /// Two-dimensional breakdown for the management UI.
    /// Tasks without a task_id in llm_usage_log are excluded (JOIN filters them).
    pub async fn get_usage_stats_by_task_type(&self, since: i64, until: i64) -> Vec<TaskTypeUsage> {
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
             ORDER BY COUNT(*) DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!(
                    "[STORAGE_SN] get_usage_stats_by_task_type prepare failed: {}",
                    e
                );
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
    use std::sync::Arc;

    use super::*;
    use tempfile::TempDir;
    use tokio::sync::Barrier;

    #[tokio::test]
    async fn test_insert_and_claim_task() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"sess_001"}"#,
                None,
                Some("sessions"),
                Some("sess_001"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .expect("Should insert first task");
        assert!(id > 0);

        let claimed = s.claim_pending_tasks(10).await;
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].id, id);

        // Double-claim prevention
        let claimed2 = s.claim_pending_tasks(10).await;
        assert!(claimed2.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_connections_claim_each_task_once() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("shared-task-queue.db");
        let first = Arc::new(MemoryStorage::open(&database_path, None).unwrap());
        let second = Arc::new(MemoryStorage::open(&database_path, None).unwrap());
        let task_id = first
            .insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap()
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));

        let first_claim = {
            let storage = Arc::clone(&first);
            let barrier = Arc::clone(&barrier);
            tokio::spawn(async move {
                barrier.wait().await;
                storage.claim_pending_tasks(1).await
            })
        };
        let second_claim = {
            let storage = Arc::clone(&second);
            let barrier = Arc::clone(&barrier);
            tokio::spawn(async move {
                barrier.wait().await;
                storage.claim_pending_tasks(1).await
            })
        };
        barrier.wait().await;

        let mut claimed = first_claim.await.unwrap();
        claimed.extend(second_claim.await.unwrap());
        assert_eq!(claimed.len(), 1, "a durable task may have only one owner");
        assert_eq!(claimed[0].id, task_id);
    }

    /// [SUPERNODE-ENQUEUE-IDEMPOTENCY 2026-08-14 by Codex] The active-task
    /// invariant must hold across independent SQLite connections, not only
    /// within one process-local connection mutex.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_connections_enqueue_targeted_task_once() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("shared-enqueue-queue.db");
        let first = Arc::new(MemoryStorage::open(&database_path, None).unwrap());
        let second = Arc::new(MemoryStorage::open(&database_path, None).unwrap());
        let barrier = Arc::new(Barrier::new(3));

        let enqueue = |storage: Arc<MemoryStorage>, barrier: Arc<Barrier>| {
            tokio::spawn(async move {
                barrier.wait().await;
                storage
                    .insert_cognitive_task(
                        "session_title",
                        5,
                        r#"{"session_id":"shared-session"}"#,
                        None,
                        Some("sessions"),
                        Some("shared-session"),
                        "structured",
                        3,
                    )
                    .await
            })
        };

        let first_enqueue = enqueue(Arc::clone(&first), Arc::clone(&barrier));
        let second_enqueue = enqueue(Arc::clone(&second), Arc::clone(&barrier));
        barrier.wait().await;

        let outcomes = [
            first_enqueue.await.unwrap().unwrap(),
            second_enqueue.await.unwrap().unwrap(),
        ];
        assert_eq!(
            outcomes.iter().filter(|outcome| outcome.is_some()).count(),
            1,
            "only one connection may enqueue an active targeted task"
        );

        let conn = first.conn.lock().await;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM cognitive_tasks
                 WHERE target_table = 'sessions'
                   AND target_id = 'shared-session'
                   AND task_type = 'session_title'
                   AND status IN ('pending', 'processing')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_recovery_assigns_expired_claim_to_one_owner() {
        let directory = TempDir::new().unwrap();
        let database_path = directory.path().join("shared-recovery-queue.db");
        let first = Arc::new(MemoryStorage::open(&database_path, None).unwrap());
        let second = Arc::new(MemoryStorage::open(&database_path, None).unwrap());
        let task_id = first
            .insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap()
            .unwrap();
        first.claim_pending_tasks(1).await;
        first
            .stage_task_result_and_usage(
                task_id,
                "staged-title",
                r#"{"input":1,"output":1,"cached":0}"#,
                &TaskUsageRecord {
                    provider: "test-provider",
                    model: "test-model",
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    latency_ms: 1,
                },
            )
            .await
            .unwrap();
        {
            let conn = first.conn.lock().await;
            conn.execute(
                "UPDATE cognitive_tasks SET started_at = started_at - 1000 WHERE id = ?1",
                params![task_id],
            )
            .unwrap();
        }

        let barrier = Arc::new(Barrier::new(3));
        let first_claim = {
            let storage = Arc::clone(&first);
            let barrier = Arc::clone(&barrier);
            tokio::spawn(async move {
                barrier.wait().await;
                storage.recover_and_claim_pending_tasks(1, 60).await
            })
        };
        let second_claim = {
            let storage = Arc::clone(&second);
            let barrier = Arc::clone(&barrier);
            tokio::spawn(async move {
                barrier.wait().await;
                storage.recover_and_claim_pending_tasks(1, 60).await
            })
        };
        barrier.wait().await;

        let first_claim = first_claim.await.unwrap();
        let second_claim = second_claim.await.unwrap();
        assert_eq!(
            first_claim.recovered + second_claim.recovered,
            1,
            "an expired claim must be recovered once"
        );
        let mut claimed = first_claim.tasks;
        claimed.extend(second_claim.tasks);
        assert_eq!(claimed.len(), 1, "a recovered task may have only one owner");
        assert_eq!(claimed[0].id, task_id);

        let recovered = first.get_task(task_id).await.unwrap();
        assert_eq!(recovered.status, "processing");
        assert_eq!(recovered.result.as_deref(), Some("staged-title"));
        assert_eq!(recovered.retry_count, 0);
    }

    /// (v2.5.0+Fix) Verify idempotency: inserting the same (target_table, target_id,
    /// task_type) while an active task exists must return Ok(None).
    #[tokio::test]
    async fn test_insert_cognitive_task_dedup() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        let id1 = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"sess_001"}"#,
                None,
                Some("sessions"),
                Some("sess_001"),
                "structured",
                3,
            )
            .await
            .unwrap();
        assert!(id1.is_some(), "First insert must succeed");

        // Same triple, status=pending → must be skipped
        let id2 = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"sess_001"}"#,
                None,
                Some("sessions"),
                Some("sess_001"),
                "structured",
                3,
            )
            .await
            .unwrap();
        assert!(id2.is_none(), "Duplicate pending task must be skipped");

        let tasks = s
            .get_tasks_filtered(None, Some("session_title"), 10, 0)
            .await;
        assert_eq!(tasks.len(), 1);

        // Different task_type on same target → allowed
        let id3 = s
            .insert_cognitive_task(
                "code_analysis",
                5,
                r#"{}"#,
                None,
                Some("sessions"),
                Some("sess_001"),
                "structured",
                3,
            )
            .await
            .unwrap();
        assert!(
            id3.is_some(),
            "Different task_type on same target must be allowed"
        );

        // After completion, same type may be re-inserted
        let id1_unwrapped = id1.unwrap();
        s.claim_pending_tasks(10).await;
        s.complete_task(
            id1_unwrapped,
            r#"{"title":"Done"}"#,
            "openai",
            "gpt-4o-mini",
            "{}",
        )
        .await
        .unwrap();

        let id4 = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"sess_001"}"#,
                None,
                Some("sessions"),
                Some("sess_001"),
                "structured",
                3,
            )
            .await
            .unwrap();
        assert!(id4.is_some(), "Re-insert after completion must succeed");
    }

    /// Tasks with no target (recall_synthesis style) always insert regardless.
    #[tokio::test]
    async fn test_insert_no_target_always_inserts() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        let id1 = s
            .insert_cognitive_task(
                "recall_synthesis",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                3,
            )
            .await
            .unwrap();
        let id2 = s
            .insert_cognitive_task(
                "recall_synthesis",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                3,
            )
            .await
            .unwrap();

        assert!(id1.is_some());
        assert!(id2.is_some());
        assert_ne!(id1.unwrap(), id2.unwrap());
    }

    /// (v2.5.0+Fix) Verify startup recovery resets stale processing tasks.
    #[tokio::test]
    async fn test_reset_stale_processing_tasks() {
        let s = MemoryStorage::open(":memory:", None).unwrap();

        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{}"#,
                None,
                Some("sessions"),
                Some("s1"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        s.claim_pending_tasks(1).await;

        // Manually back-date started_at to simulate a stale task
        {
            let conn = s.conn.lock().await;
            conn.execute(
                "UPDATE cognitive_tasks SET started_at = started_at - 1000 WHERE id = ?1",
                params![id],
            )
            .unwrap();
        }

        let recovered = s.reset_stale_processing_tasks(60).await;
        assert_eq!(recovered, 1);

        let task = s.get_task(id).await.unwrap();
        assert_eq!(task.status, "pending");
        assert_eq!(
            task.retry_count, 0,
            "retry_count must NOT be incremented by recovery"
        );
        assert_eq!(task.error_message.as_deref(), Some(EXPIRED_CLAIM_REASON));
    }

    /// Within-timeout tasks must NOT be reset.
    #[tokio::test]
    async fn test_reset_stale_skips_fresh_processing() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task(
            "session_title",
            5,
            r#"{}"#,
            None,
            Some("sessions"),
            Some("s1"),
            "structured",
            3,
        )
        .await
        .unwrap();
        s.claim_pending_tasks(1).await;

        let recovered = s.reset_stale_processing_tasks(120).await;
        assert_eq!(recovered, 0);
    }

    #[tokio::test]
    async fn release_processing_tasks_preserves_staged_result_and_retry_budget() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let task_id = storage
            .insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap()
            .unwrap();
        storage.claim_pending_tasks(1).await;
        storage
            .stage_task_result_and_usage(
                task_id,
                "staged-title",
                r#"{"input":1,"output":1,"cached":0}"#,
                &TaskUsageRecord {
                    provider: "test-provider",
                    model: "test-model",
                    input_tokens: 1,
                    output_tokens: 1,
                    cached_tokens: 0,
                    latency_ms: 1,
                },
            )
            .await
            .unwrap();

        assert_eq!(
            storage
                .release_processing_tasks(&[task_id], "task_worker_shutdown")
                .await
                .unwrap(),
            1
        );
        let released = storage.get_task(task_id).await.unwrap();
        assert_eq!(released.status, "pending");
        assert_eq!(released.retry_count, 0);
        assert_eq!(released.started_at, None);
        assert_eq!(released.result.as_deref(), Some("staged-title"));
        assert_eq!(
            released.error_message.as_deref(),
            Some("task_worker_shutdown")
        );
        assert_eq!(
            storage
                .release_processing_tasks(&[task_id], "task_worker_shutdown")
                .await
                .unwrap(),
            0,
            "release must remain status guarded and idempotent"
        );
    }

    #[tokio::test]
    async fn test_complete_task_rejects_non_processing() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"s1"}"#,
                None,
                Some("sessions"),
                Some("s1"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();

        // Task is still pending — complete should fail (not processing)
        let result = s
            .complete_task(id, "{}", "openai", "gpt-4o-mini", "{}")
            .await;
        assert!(
            result.is_err(),
            "complete_task on pending task must return Err"
        );

        // Cancel it — complete should still fail
        s.cancel_task(id).await.unwrap();
        let result2 = s
            .complete_task(id, "{}", "openai", "gpt-4o-mini", "{}")
            .await;
        assert!(
            result2.is_err(),
            "complete_task on cancelled task must return Err"
        );
    }

    #[tokio::test]
    async fn test_complete_task() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"s1"}"#,
                None,
                Some("sessions"),
                Some("s1"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        s.claim_pending_tasks(1).await;
        s.complete_task(
            id,
            r#"{"title":"Project Alpha: JWT"}"#,
            "openai",
            "gpt-4o-mini",
            r#"{"input":50,"output":10,"cached":0}"#,
        )
        .await
        .unwrap();
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "completed");
        assert!(t.result.is_some());
    }

    #[tokio::test]
    async fn staged_result_is_idempotent_and_survives_writeback_retry() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{"session_id":"s1"}"#,
                None,
                Some("sessions"),
                Some("s1"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        storage.claim_pending_tasks(1).await;

        let usage = TaskUsageRecord {
            provider: "test-provider",
            model: "test-model",
            input_tokens: 21,
            output_tokens: 8,
            cached_tokens: 3,
            latency_ms: 17,
        };
        let first = storage
            .stage_task_result_and_usage(
                task_id,
                "Durable title",
                r#"{"input":21,"output":8,"cached":3}"#,
                &usage,
            )
            .await
            .unwrap();
        let second = storage
            .stage_task_result_and_usage(
                task_id,
                "Must not replace the first result",
                r#"{"input":999,"output":999,"cached":0}"#,
                &usage,
            )
            .await
            .unwrap();
        assert_eq!(first, StageTaskResult::Stored);
        assert_eq!(second, StageTaskResult::AlreadyStored);

        let staged = storage.get_task(task_id).await.unwrap();
        assert_eq!(staged.status, "processing");
        assert_eq!(staged.result.as_deref(), Some("Durable title"));
        assert_eq!(staged.provider_used.as_deref(), Some("test-provider"));

        let usage_rows: i64 = {
            let conn = storage.conn.lock().await;
            conn.query_row(
                "SELECT COUNT(*) FROM llm_usage_log WHERE task_id = ?1",
                params![task_id],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert_eq!(usage_rows, 1, "staging retries must not duplicate usage");

        storage
            .fail_task(task_id, "session_title_target_not_found")
            .await
            .unwrap();
        let retry = storage.claim_pending_tasks(1).await;
        assert_eq!(retry.len(), 1);
        assert_eq!(retry[0].result.as_deref(), Some("Durable title"));

        storage.complete_staged_task(task_id).await.unwrap();
        let completed = storage.get_task(task_id).await.unwrap();
        assert_eq!(completed.status, "completed");
        assert!(completed.completed_at.is_some());
    }

    #[tokio::test]
    async fn complete_staged_task_requires_a_result() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let task_id = storage
            .insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap()
            .unwrap();
        storage.claim_pending_tasks(1).await;

        assert!(storage.complete_staged_task(task_id).await.is_err());
        assert_eq!(
            storage.get_task(task_id).await.unwrap().status,
            "processing"
        );
    }

    #[tokio::test]
    async fn invalid_output_failure_discards_only_the_staged_result() {
        let storage = MemoryStorage::open(":memory:", None).unwrap();
        let task_id = storage
            .insert_cognitive_task(
                "session_title",
                5,
                "{}",
                None,
                Some("sessions"),
                Some("s1"),
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        storage.claim_pending_tasks(1).await;
        storage
            .stage_task_result_and_usage(
                task_id,
                "",
                r#"{"input":2,"output":0,"cached":0}"#,
                &TaskUsageRecord {
                    provider: "test-provider",
                    model: "test-model",
                    input_tokens: 2,
                    output_tokens: 0,
                    cached_tokens: 0,
                    latency_ms: 1,
                },
            )
            .await
            .unwrap();

        storage
            .fail_task_and_discard_result(task_id, "session_title_empty")
            .await
            .unwrap();
        let pending = storage.get_task(task_id).await.unwrap();
        assert_eq!(pending.status, "pending");
        assert!(pending.result.is_none());
        assert!(pending.provider_used.is_none());
        assert!(pending.model_used.is_none());
        assert!(pending.token_usage.is_none());

        let usage_rows: i64 = {
            let conn = storage.conn.lock().await;
            conn.query_row(
                "SELECT COUNT(*) FROM llm_usage_log WHERE task_id = ?1",
                params![task_id],
                |row| row.get(0),
            )
            .unwrap()
        };
        assert_eq!(usage_rows, 1, "actual provider usage must remain auditable");
    }

    #[tokio::test]
    async fn test_fail_task_retries() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "community_summary",
                3,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                2,
            )
            .await
            .unwrap()
            .unwrap();
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
    async fn test_fail_task_rejects_non_processing_state() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                2,
            )
            .await
            .unwrap()
            .unwrap();

        assert!(s.fail_task(id, "must_not_apply").await.is_err());
        let pending = s.get_task(id).await.unwrap();
        assert_eq!(pending.status, "pending");
        assert_eq!(pending.retry_count, 0);
        assert!(pending.error_message.is_none());

        s.claim_pending_tasks(1).await;
        s.complete_task(id, "done", "provider", "model", "{}")
            .await
            .unwrap();
        assert!(s
            .fail_task(id, "must_not_replace_completion")
            .await
            .is_err());
        let completed = s.get_task(id).await.unwrap();
        assert_eq!(completed.status, "completed");
        assert_eq!(completed.retry_count, 0);
        assert!(completed.error_message.is_none());
    }

    #[tokio::test]
    async fn test_retry_task_resets_to_pending() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                1,
            )
            .await
            .unwrap()
            .unwrap();
        s.claim_pending_tasks(1).await;
        s.fail_task(id, "error").await.unwrap();
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "failed");

        // Human retry override
        s.retry_task(id).await.unwrap();
        let t2 = s.get_task(id).await.unwrap();
        assert_eq!(t2.status, "pending");
        assert_eq!(t2.retry_count, 2); // incremented, not reset
        assert!(
            t2.retry_count < t2.max_retries,
            "manual retry must restore a claimable retry budget"
        );
        let claimed = s.claim_pending_tasks(1).await;
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].id, id);
    }

    #[tokio::test]
    async fn test_retry_task_rejects_non_failed() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        let result = s.retry_task(id).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("pending"));
    }

    #[tokio::test]
    async fn test_retry_task_not_found() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let result = s.retry_task(99999).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[tokio::test]
    async fn test_cancel_task() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "entity_description",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        let affected = s.cancel_task(id).await.unwrap();
        assert_eq!(affected, 1);
        let t = s.get_task(id).await.unwrap();
        assert_eq!(t.status, "cancelled");

        let claimed = s.claim_pending_tasks(10).await;
        assert!(claimed.is_empty());
    }

    /// cancel_task returns 0 when task is already non-pending (TOCTOU guard).
    #[tokio::test]
    async fn test_cancel_task_already_claimed_returns_zero() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        let id = s
            .insert_cognitive_task(
                "session_title",
                5,
                r#"{}"#,
                None,
                None,
                None,
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();
        s.claim_pending_tasks(1).await; // moves to processing

        let affected = s.cancel_task(id).await.unwrap();
        assert_eq!(
            affected, 0,
            "Should return 0 when task is no longer pending"
        );
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task("t", 2, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();
        s.insert_cognitive_task("t", 9, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();
        s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();

        let claimed = s.claim_pending_tasks(3).await;
        assert_eq!(claimed.len(), 3);
        assert_eq!(claimed[0].priority, 9);
        assert_eq!(claimed[1].priority, 5);
        assert_eq!(claimed[2].priority, 2);
    }

    #[tokio::test]
    async fn test_get_tasks_filtered() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();
        s.insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();
        s.insert_cognitive_task(
            "community_summary",
            3,
            "{}",
            None,
            None,
            None,
            "structured",
            3,
        )
        .await
        .unwrap();

        let title_tasks = s
            .get_tasks_filtered(None, Some("session_title"), 10, 0)
            .await;
        assert_eq!(title_tasks.len(), 2);

        let pending = s.get_tasks_filtered(Some("pending"), None, 10, 0).await;
        assert_eq!(pending.len(), 3);

        let both = s
            .get_tasks_filtered(Some("pending"), Some("community_summary"), 10, 0)
            .await;
        assert_eq!(both.len(), 1);

        let all = s.get_tasks_filtered(None, None, 10, 0).await;
        assert_eq!(all.len(), 3);
    }

    /// v2.5.2+Pagination: verify offset pages are non-overlapping and correctly sized.
    #[tokio::test]
    async fn test_get_tasks_filtered_pagination() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        for _ in 0..5 {
            s.insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
                .await
                .unwrap();
        }

        let page0 = s.get_tasks_filtered(None, None, 3, 0).await;
        let page1 = s.get_tasks_filtered(None, None, 3, 3).await;
        assert_eq!(page0.len(), 3);
        assert_eq!(page1.len(), 2);

        // No overlap
        let ids0: std::collections::HashSet<i64> = page0.iter().map(|t| t.id).collect();
        let ids1: std::collections::HashSet<i64> = page1.iter().map(|t| t.id).collect();
        assert!(ids0.is_disjoint(&ids1), "Pages must not overlap");

        // offset beyond total → empty
        let page2 = s.get_tasks_filtered(None, None, 3, 10).await;
        assert!(page2.is_empty());

        // Filter + offset: 2 session_title tasks, offset=1 → 1 result
        let filtered_page1 = s
            .get_tasks_filtered(None, Some("session_title"), 10, 1)
            .await;
        assert_eq!(filtered_page1.len(), 4); // 5 total - 1 offset
    }

    #[tokio::test]
    async fn test_count_tasks_by_status() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();
        s.insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap();
        let id3 = s
            .insert_cognitive_task("t", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap()
            .unwrap();
        s.cancel_task(id3).await.unwrap();

        let counts = s.count_tasks_by_status().await;
        assert_eq!(counts["pending"], 2);
        assert_eq!(counts["cancelled"], 1);
        assert_eq!(counts["failed"], 0);
    }

    #[tokio::test]
    async fn test_insert_usage_log_and_stats() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_usage_log(Some(1), "openai", "gpt-4o-mini", 100, 50, 0, 850)
            .await
            .unwrap();
        s.insert_usage_log(Some(2), "openai", "gpt-4o-mini", 200, 80, 20, 1200)
            .await
            .unwrap();
        s.insert_usage_log(Some(3), "anthropic", "claude-haiku", 150, 60, 0, 950)
            .await
            .unwrap();

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

        let t1 = s
            .insert_cognitive_task("session_title", 5, "{}", None, None, None, "structured", 3)
            .await
            .unwrap()
            .unwrap();
        let t2 = s
            .insert_cognitive_task(
                "community_narrative",
                5,
                "{}",
                None,
                None,
                None,
                "structured",
                3,
            )
            .await
            .unwrap()
            .unwrap();

        s.insert_usage_log(Some(t1), "deepseek", "deepseek-reasoner", 100, 40, 0, 800)
            .await
            .unwrap();
        s.insert_usage_log(Some(t1), "deepseek", "deepseek-reasoner", 120, 50, 0, 900)
            .await
            .unwrap();
        s.insert_usage_log(
            Some(t2),
            "claude",
            "claude-sonnet-4-20250514",
            200,
            80,
            30,
            1100,
        )
        .await
        .unwrap();

        let stats = s.get_usage_stats_by_task_type(0, 0).await;
        assert_eq!(stats.len(), 2);

        let st = stats
            .iter()
            .find(|r| r.task_type == "session_title")
            .unwrap();
        assert_eq!(st.provider, "deepseek");
        assert_eq!(st.calls, 2);
        assert_eq!(st.input_tokens, 220);

        let cn = stats
            .iter()
            .find(|r| r.task_type == "community_narrative")
            .unwrap();
        assert_eq!(cn.provider, "claude");
        assert_eq!(cn.calls, 1);
        assert_eq!(cn.cached_tokens, 30);
    }

    #[tokio::test]
    async fn test_get_tasks_for_target() {
        let s = MemoryStorage::open(":memory:", None).unwrap();
        s.insert_cognitive_task(
            "session_title",
            5,
            "{}",
            None,
            Some("sessions"),
            Some("sess_001"),
            "structured",
            3,
        )
        .await
        .unwrap();

        let tasks = s.get_tasks_for_target("sessions", "sess_001", None).await;
        assert_eq!(tasks.len(), 1);

        let pending = s
            .get_tasks_for_target("sessions", "sess_001", Some("pending"))
            .await;
        assert_eq!(pending.len(), 1);

        let none = s.get_tasks_for_target("sessions", "sess_999", None).await;
        assert!(none.is_empty());
    }
}
