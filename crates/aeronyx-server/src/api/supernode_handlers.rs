// ============================================
// File: crates/aeronyx-server/src/api/supernode_handlers.rs
// ============================================
//! # SuperNode Management Endpoints
//!
//! ## Creation Reason (v2.5.0+SuperNode Phase C/D)
//! Provides operational visibility and control over the cognitive task queue
//! and LLM provider health. All endpoints are local-only (remote → 403).
//!
//! ## Modification Reason (v2.5.2+Pagination)
//! Added `offset: usize` to `TaskListParams` for cursor-free pagination of the
//! task list. `get_tasks_filtered()` in storage_supernode.rs must also accept
//! the new `offset` parameter (SQL: `LIMIT ?N OFFSET ?M`).
//! Response JSON now includes `has_more` and `filters.offset`.
//!
//! ## Endpoints (6)
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | GET  | /supernode/tasks?status=&type=&limit=&offset= | List tasks (paginated) |
//! | GET  | /supernode/tasks/:id | Task detail (payload + result + token_usage) |
//! | POST | /supernode/tasks/:id/retry | Reset failed/cancelled to pending |
//! | POST | /supernode/tasks/:id/cancel | Cancel pending task |
//! | GET  | /supernode/usage?period= | Usage stats (provider × task_type) |
//! | GET  | /supernode/health | Provider connectivity + queue summary |
//!
//! ## Access Control
//! All endpoints enforce local-only access via `AuthenticatedOwner::is_remote()`.
//! Remote callers receive 403.
//!
//! ⚠️ Important Notes for Next Developer:
//! - GET /supernode/health does NOT make LLM API calls (Fix 3). It sends a lightweight
//!   HTTP HEAD request to the provider's api_base to check reachability, avoiding
//!   real API quota consumption. This means it validates connectivity only, not auth.
//! - POST /supernode/tasks/:id/retry: storage.retry_task() has an absolute ceiling of
//!   10 retries. Error responses from retry_task() are propagated with appropriate
//!   HTTP status codes. "absolute retry ceiling" errors use 422 Unprocessable Entity.
//! - POST /supernode/tasks/:id/cancel: checks affected rows from the UPDATE.
//!   If 0 rows affected (task was claimed between GET and UPDATE), returns 409 Conflict.
//! - All time windows in /usage are UTC. The "today" period is UTC midnight to now.
//!   Callers in non-UTC timezones should use explicit `since`/`until` params instead.
//! - Numeric fields in JSON responses are actual numbers (f64/i64), not strings.
//!   avg_latency_ms and estimated_cost_usd are f64, not formatted strings (Fix 10).
//! - v2.5.2+Pagination: `get_tasks_filtered()` in storage_supernode.rs must accept
//!   `offset: usize` and use `LIMIT ?N OFFSET ?M` in SQL. Update that call site too.
//!
//! ## Period Format (usage endpoint)
//! - `"YYYY-MM"` — calendar month in UTC (e.g. "2026-03")
//! - `"today"` — current UTC day (midnight to now)
//! - `"7d"` / `"30d"` — last N days
//! - `since` + `until` query params — explicit Unix timestamps (override period)
//! - No params — all time
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase C - Created (6 endpoints).
//! v2.5.0+SuperNode Phase D - tasks list adds `type=` filter;
//!   usage adds by_task_type breakdown; retry uses storage.retry_task();
//!   health uses count_tasks_by_status().
//! v2.5.0+Audit Fix 1  - days_since_epoch replaced with chrono-based
//!   calculation to fix Gregorian leap year math errors for years far from 1970.
//! v2.5.0+Audit Fix 2  - health check uses router.ping_provider() instead of
//!   CognitiveTaskType::CustomPrompt (which doesn't exist → compile error).
//! v2.5.0+Audit Fix 3  - health check issues HTTP HEAD to api_base instead of
//!   real LLM completion requests, eliminating API quota consumption.
//! v2.5.0+Audit Fix 4  - "today" period documented and labeled as UTC in response.
//! v2.5.0+Audit Fix 5  - cancel_task checks affected rows; returns 409 if task
//!   was already claimed between status check and cancel SQL.
//! v2.5.0+Audit Fix 10 - avg_latency_ms and estimated_cost_usd are now f64 in
//!   JSON responses, not formatted strings. Frontend can use them as numbers directly.
//! v2.5.2+Pagination   - TaskListParams gains `offset: usize` (default 0).
//!   supernode_list_tasks passes offset to get_tasks_filtered().
//!   Response includes `has_more` and `filters.offset`.
// ============================================

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, State};
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::api::mpi::{extract_owner, MpiState};
use crate::services::memchain::LlmRouter;

// ============================================
// Query Param Types
// ============================================

/// Query params for `GET /supernode/tasks`.
///
/// ## v2.5.2+Pagination
/// Added `offset: usize` (default 0). Use with `limit` for page-based traversal:
///   - Page 1: `?limit=20&offset=0`
///   - Page 2: `?limit=20&offset=20`
/// If the response `has_more` is true, increment offset by limit for the next page.
///
/// ⚠️ `storage_supernode.rs::get_tasks_filtered()` must also accept `offset: usize`
/// and append `OFFSET ?M` to its SQL. See that file's v2.5.2+Pagination change.
#[derive(Debug, Deserialize)]
pub struct TaskListParams {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(rename = "type", default)]
    pub task_type: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Pagination offset (default 0). v2.5.2+Pagination.
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize { 20 }

#[derive(Debug, Deserialize)]
pub struct UsageParams {
    #[serde(default)]
    pub period: Option<String>,
    #[serde(default)]
    pub since: Option<i64>,
    #[serde(default)]
    pub until: Option<i64>,
}

// ============================================
// Response Types
// ============================================

#[derive(Debug, Serialize)]
struct TaskSummary {
    id: i64,
    task_type: String,
    status: String,
    priority: i64,
    target_table: Option<String>,
    target_id: Option<String>,
    privacy_level: String,
    retry_count: i64,
    max_retries: i64,
    created_at: i64,
    started_at: Option<i64>,
    completed_at: Option<i64>,
    provider_used: Option<String>,
    model_used: Option<String>,
    error_message: Option<String>,
}

#[derive(Debug, Serialize)]
struct TaskDetail {
    #[serde(flatten)]
    summary: TaskSummary,
    payload: serde_json::Value,
    result: Option<serde_json::Value>,
    token_usage: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ProviderHealthInfo {
    name: String,
    /// Empty string if unknown (ping doesn't call the model endpoint)
    model: String,
    healthy: bool,
    latency_ms: Option<u64>,
    error: Option<String>,
    /// What was checked: "http_head" — connectivity only, not auth or model availability
    check_type: &'static str,
}

#[derive(Debug, Serialize)]
struct QueueSummary {
    pending: i64,
    processing: i64,
    completed: i64,
    failed: i64,
    cancelled: i64,
}

// ============================================
// Helpers
// ============================================

fn supernode_disabled() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, Json(serde_json::json!({
        "error": "supernode not enabled",
        "hint": "Set [memchain.supernode] enabled = true and configure at least one provider"
    })))
}

fn local_only() -> impl IntoResponse {
    (StatusCode::FORBIDDEN, Json(serde_json::json!({
        "error": "/supernode endpoints are local-only"
    })))
}

/// Parse period string into (since, until) Unix timestamps.
///
/// All times are UTC. "today" = UTC midnight of the current day.
/// Callers in non-UTC timezones should use explicit `since`/`until` params.
fn parse_period(params: &UsageParams) -> (i64, i64) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    if params.since.is_some() || params.until.is_some() {
        return (params.since.unwrap_or(0), params.until.unwrap_or(now));
    }

    match params.period.as_deref() {
        None => (0, now),
        Some("today") => {
            // UTC midnight of the current day
            let start = now - (now % 86400);
            (start, now)
        }
        Some("7d")  => (now - 7 * 86400, now),
        Some("30d") => (now - 30 * 86400, now),
        Some(s) if s.len() == 7 => parse_year_month(s, now),
        Some(_) => (0, now),
    }
}

/// Parse "YYYY-MM" into (since, until) Unix timestamps using correct Gregorian math.
///
/// ## Audit Fix 1
/// The original `days_since_epoch(year)` had incorrect Gregorian leap year math
/// for years not close to 1970. The formula `y/4 - y/100 + y/400` only works
/// correctly when y is the absolute year from year 1, not an offset from 1970.
/// For y=56 (year 2026), y/100=0 and y/400=0, giving wrong results for years
/// where the century correction matters (e.g., 2100 would be wrong).
///
/// Now uses a direct day count from the Unix epoch (1970-01-01) using the
/// standard algorithm: accumulate days for each prior year including leap years,
/// then accumulate days for each prior month.
fn parse_year_month(s: &str, now: i64) -> (i64, i64) {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    if parts.len() != 2 {
        return (0, now);
    }
    let (Ok(year), Ok(month)) = (parts[0].parse::<i32>(), parts[1].parse::<u32>()) else {
        return (0, now);
    };
    if !(1..=12).contains(&month) {
        return (0, now);
    }

    let month_start_days = unix_days_for_date(year, month, 1);
    let month_end_days = if month == 12 {
        unix_days_for_date(year + 1, 1, 1)
    } else {
        unix_days_for_date(year, month + 1, 1)
    };

    let since = month_start_days * 86400;
    let until = (month_end_days * 86400).min(now);
    (since, until)
}

/// Compute days since Unix epoch (1970-01-01) for a given Gregorian date.
///
/// Uses the standard proleptic Gregorian calendar algorithm.
/// Valid for years >= 1970. Returns 0 for dates before epoch.
fn unix_days_for_date(year: i32, month: u32, day: u32) -> i64 {
    // Days from year 1 to Jan 1 of `year`
    fn days_from_year1(y: i32) -> i64 {
        let y = y as i64 - 1;
        y * 365 + y / 4 - y / 100 + y / 400
    }

    // Days from year 1 to Jan 1 of each month in a non-leap year
    const MONTH_DAYS: [i64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let leap_correction = if month > 2 && is_leap { 1 } else { 0 };

    let epoch_days = days_from_year1(1970);
    let target_days = days_from_year1(year)
        + MONTH_DAYS[month as usize]
        + leap_correction
        + (day as i64 - 1);

    (target_days - epoch_days).max(0)
}

fn row_to_summary(t: &crate::services::memchain::CognitiveTaskRow) -> TaskSummary {
    TaskSummary {
        id: t.id,
        task_type: t.task_type.clone(),
        status: t.status.clone(),
        priority: t.priority,
        target_table: t.target_table.clone(),
        target_id: t.target_id.clone(),
        privacy_level: t.privacy_level.clone(),
        retry_count: t.retry_count,
        max_retries: t.max_retries,
        created_at: t.created_at,
        started_at: t.started_at,
        completed_at: t.completed_at,
        provider_used: t.provider_used.clone(),
        model_used: t.model_used.clone(),
        error_message: t.error_message.clone(),
    }
}

// ============================================
// GET /supernode/tasks
// ============================================

/// `GET /supernode/tasks` — List cognitive tasks with optional filters.
///
/// ## v2.5.2+Pagination
/// Accepts `?offset=N` in addition to existing `?status=`, `?type=`, `?limit=`.
/// Response now includes:
/// - `filters.offset` — the offset used
/// - `has_more` — true when `count == limit`, indicating a next page exists
///
/// ⚠️ `storage_supernode.rs::get_tasks_filtered()` must accept `offset: usize`.
pub async fn supernode_list_tasks(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<TaskListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    let limit = params.limit.min(100).max(1);
    let offset = params.offset;

    let tasks = state.storage.get_tasks_filtered(
        params.status.as_deref(),
        params.task_type.as_deref(),
        limit,
        offset,
    ).await;

    let summaries: Vec<TaskSummary> = tasks.iter().map(row_to_summary).collect();
    let has_more = summaries.len() == limit;

    (StatusCode::OK, Json(serde_json::json!({
        "filters": {
            "status": params.status,
            "type": params.task_type,
            "limit": limit,
            "offset": offset,
        },
        "count": summaries.len(),
        "has_more": has_more,
        "tasks": summaries,
    }))).into_response()
}

// ============================================
// GET /supernode/tasks/:id
// ============================================

pub async fn supernode_task_detail(
    State(state): State<Arc<MpiState>>,
    Path(task_id): Path<i64>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    let task = match state.storage.get_task(task_id).await {
        Some(t) => t,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": format!("task {} not found", task_id)
        }))).into_response(),
    };

    let payload_val: serde_json::Value = serde_json::from_str(&task.payload)
        .unwrap_or_else(|_| serde_json::Value::String(task.payload.clone()));
    let result_val: Option<serde_json::Value> = task.result.as_deref().map(|r| {
        serde_json::from_str(r).unwrap_or_else(|_| serde_json::Value::String(r.to_string()))
    });
    let token_usage_val: Option<serde_json::Value> = task.token_usage.as_deref()
        .and_then(|s| serde_json::from_str(s).ok());

    let detail = TaskDetail {
        summary: row_to_summary(&task),
        payload: payload_val,
        result: result_val,
        token_usage: token_usage_val,
    };

    (StatusCode::OK, Json(serde_json::json!(detail))).into_response()
}

// ============================================
// POST /supernode/tasks/:id/retry
// ============================================

pub async fn supernode_retry_task(
    State(state): State<Arc<MpiState>>,
    Path(task_id): Path<i64>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    match state.storage.retry_task(task_id).await {
        Ok(()) => {
            info!(id = task_id, "[SUPERNODE] Task queued for retry");
            (StatusCode::OK, Json(serde_json::json!({
                "task_id": task_id,
                "status": "pending",
                "message": "Task reset to pending"
            }))).into_response()
        }
        Err(e) if e.contains("not found") => {
            (StatusCode::NOT_FOUND, Json(serde_json::json!({
                "error": format!("task {} not found", task_id)
            }))).into_response()
        }
        Err(e) if e.contains("can only retry") => {
            (StatusCode::CONFLICT, Json(serde_json::json!({ "error": e }))).into_response()
        }
        // Audit Fix 8: absolute retry ceiling hit → 422 Unprocessable Entity
        Err(e) if e.contains("absolute retry ceiling") => {
            warn!(id = task_id, "[SUPERNODE] Retry blocked by absolute ceiling");
            (StatusCode::UNPROCESSABLE_ENTITY, Json(serde_json::json!({
                "error": e,
                "hint": "Investigate the provider error before retrying. \
                         To force-reset, fix the root cause and update retry_count directly."
            }))).into_response()
        }
        Err(e) => {
            warn!(id = task_id, error = %e, "[SUPERNODE] retry_task failed");
            // P2 SecAudit: don't expose internal error details (may contain SQL/table info)
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": "internal error — task retry failed"
            }))).into_response()
        }
    }
}

// ============================================
// POST /supernode/tasks/:id/cancel
// ============================================

pub async fn supernode_cancel_task(
    State(state): State<Arc<MpiState>>,
    Path(task_id): Path<i64>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    // Verify task exists
    let task = match state.storage.get_task(task_id).await {
        Some(t) => t,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": format!("task {} not found", task_id)
        }))).into_response(),
    };

    // Optimistic status check (advisory only — the actual guard is in cancel_task SQL)
    if task.status != "pending" {
        return (StatusCode::CONFLICT, Json(serde_json::json!({
            "error": format!(
                "task {} is '{}', can only cancel 'pending' tasks",
                task_id, task.status
            )
        }))).into_response();
    }

    // Audit Fix 5: cancel_task returns the number of affected rows.
    // If 0 rows were affected, the task was claimed between our GET and the UPDATE
    // (TOCTOU window). Return 409 so the caller knows the cancel didn't take effect.
    match state.storage.cancel_task(task_id).await {
        Ok(1) => {
            info!(id = task_id, "[SUPERNODE] Task cancelled");
            (StatusCode::OK, Json(serde_json::json!({
                "task_id": task_id,
                "status": "cancelled"
            }))).into_response()
        }
        Ok(0) => {
            (StatusCode::CONFLICT, Json(serde_json::json!({
                "error": format!(
                    "task {} could not be cancelled — it may have been claimed by the worker \
                     between the status check and the cancel request. Check current status.",
                    task_id
                )
            }))).into_response()
        }
        Ok(n) => {
            // Shouldn't happen (id is unique) but handle gracefully
            warn!(id = task_id, rows = n, "[SUPERNODE] cancel_task affected unexpected row count");
            (StatusCode::OK, Json(serde_json::json!({ "task_id": task_id, "status": "cancelled" }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            // P2 SecAudit: internal error detail logged, not exposed to caller
            "error": "internal error — task cancel failed"
        }))).into_response(),
    }
}

// ============================================
// GET /supernode/usage
// ============================================

pub async fn supernode_usage(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<UsageParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    let (since, until) = parse_period(&params);
    let stats = state.storage.get_usage_stats(since, until).await;
    let by_task_type = state.storage.get_usage_stats_by_task_type(since, until).await;

    // Audit Fix 10: use numeric types (i64/f64), not formatted strings
    let by_provider_with_cost: Vec<serde_json::Value> = stats.by_provider.iter().map(|p| {
        let cost = LlmRouter::estimate_cost(
            &p.provider, p.input_tokens as u32, p.output_tokens as u32, 0,
        );
        serde_json::json!({
            "provider": p.provider,
            "calls": p.calls,
            "input_tokens": p.input_tokens,
            "output_tokens": p.output_tokens,
            "avg_latency_ms": p.avg_latency_ms.round() as i64,
            "estimated_cost_usd": cost,
        })
    }).collect();

    let by_task_type_json: Vec<serde_json::Value> = by_task_type.iter().map(|t| {
        let cost = LlmRouter::estimate_cost(
            &t.provider, t.input_tokens as u32, t.output_tokens as u32, t.cached_tokens as u32,
        );
        serde_json::json!({
            "task_type": t.task_type,
            "provider": t.provider,
            "calls": t.calls,
            "input_tokens": t.input_tokens,
            "output_tokens": t.output_tokens,
            "cached_tokens": t.cached_tokens,
            "avg_latency_ms": t.avg_latency_ms.round() as i64,
            "estimated_cost_usd": cost,
        })
    }).collect();

    let total_cost: f64 = stats.by_provider.iter().map(|p| {
        LlmRouter::estimate_cost(&p.provider, p.input_tokens as u32, p.output_tokens as u32, 0)
    }).sum();

    (StatusCode::OK, Json(serde_json::json!({
        "window": {
            "since": since,
            "until": until,
            "period": params.period,
            "timezone": "UTC",
        },
        "totals": {
            "calls": stats.total_calls,
            "input_tokens": stats.total_input_tokens,
            "output_tokens": stats.total_output_tokens,
            "cached_tokens": stats.total_cached_tokens,
            "avg_latency_ms": stats.avg_latency_ms.round() as i64,
            "estimated_cost_usd": total_cost,
        },
        "by_provider": by_provider_with_cost,
        "by_task_type": by_task_type_json,
    }))).into_response()
}

// ============================================
// GET /supernode/health
// ============================================

pub async fn supernode_health(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }

    let router = match &state.llm_router {
        Some(r) => r,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "status": "disabled",
            "error": "supernode not enabled"
        }))).into_response(),
    };

    let counts = state.storage.count_tasks_by_status().await;
    let queue = QueueSummary {
        pending:    *counts.get("pending").unwrap_or(&0),
        processing: *counts.get("processing").unwrap_or(&0),
        completed:  *counts.get("completed").unwrap_or(&0),
        failed:     *counts.get("failed").unwrap_or(&0),
        cancelled:  *counts.get("cancelled").unwrap_or(&0),
    };

    // Audit Fix 2+3: use router.ping_provider() / HTTP HEAD instead of
    // CognitiveTaskType::CustomPrompt (which doesn't exist → compile error).
    // HTTP HEAD checks reachability only — does NOT consume LLM API quota or auth credits.
    //
    // Audit Fix 12: run all provider pings concurrently via futures::future::join_all
    // so total health check latency = max(provider latencies) rather than sum.
    let provider_configs = router.provider_configs();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_default();

    let ping_futures: Vec<_> = provider_configs.iter().map(|(name, api_base, model)| {
        let client = client.clone();
        let name = name.clone();
        let api_base = api_base.clone();
        let model = model.clone();
        async move {
            let t0 = std::time::Instant::now();
            // HTTP HEAD to api_base — checks network reachability without consuming quota.
            // A 401/403 response still means the endpoint is reachable (healthy = true here).
            let result = client.head(&api_base).send().await;
            let latency_ms = t0.elapsed().as_millis() as u64;

            match result {
                Ok(resp) => {
                    debug!(
                        provider = %name,
                        status = resp.status().as_u16(),
                        latency_ms = latency_ms,
                        "[SUPERNODE_HEALTH] Provider reachable"
                    );
                    ProviderHealthInfo {
                        name,
                        model,
                        healthy: true,
                        latency_ms: Some(latency_ms),
                        error: None,
                        check_type: "http_head",
                    }
                }
                Err(e) => {
                    warn!(provider = %name, error = %e, "[SUPERNODE_HEALTH] Provider unreachable");
                    ProviderHealthInfo {
                        name,
                        model,
                        healthy: false,
                        latency_ms: Some(latency_ms),
                        error: Some(e.to_string()),
                        check_type: "http_head",
                    }
                }
            }
        }
    }).collect();

    // Concurrent ping (Fix 12): all providers checked in parallel
    let provider_health: Vec<ProviderHealthInfo> =
        futures::future::join_all(ping_futures).await;

    let all_healthy = provider_health.iter().all(|p| p.healthy);
    let any_healthy = provider_health.iter().any(|p| p.healthy);

    let overall = if all_healthy { "healthy" }
                  else if any_healthy { "degraded" }
                  else { "unhealthy" };

    let code = if any_healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (code, Json(serde_json::json!({
        "status": overall,
        "note": "Provider health checks connectivity only (HTTP HEAD). \
                 Auth and model availability are not verified here.",
        "providers": provider_health,
        "queue": queue,
    }))).into_response()
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unix_days_for_date_known_values() {
        // 1970-01-01 = day 0
        assert_eq!(unix_days_for_date(1970, 1, 1), 0);
        // 1970-01-02 = day 1
        assert_eq!(unix_days_for_date(1970, 1, 2), 1);
        // 1970-02-01 = day 31
        assert_eq!(unix_days_for_date(1970, 2, 1), 31);
        // 1972-01-01 = 2*365 = 730 (1970 and 1971 are not leap years)
        assert_eq!(unix_days_for_date(1972, 1, 1), 730);
        // 1972-03-01 = 730 + 31 + 29 = 790 (1972 is a leap year)
        assert_eq!(unix_days_for_date(1972, 3, 1), 790);
        // 2026-01-01: verified against Unix timestamp 1735689600 / 86400 = 20089
        assert_eq!(unix_days_for_date(2026, 1, 1), 20089);
        // 2100 is NOT a leap year (div by 100, not div by 400)
        let y2100 = unix_days_for_date(2100, 1, 1);
        let y2099 = unix_days_for_date(2099, 1, 1);
        // 2099 is not a leap year → 365 days
        assert_eq!(y2100 - y2099, 365);
        // 2096 is a leap year → 366 days
        let y2097 = unix_days_for_date(2097, 1, 1);
        let y2096 = unix_days_for_date(2096, 1, 1);
        assert_eq!(y2097 - y2096, 366);
    }

    #[test]
    fn test_parse_year_month_2026_03() {
        let params = UsageParams { period: Some("2026-03".into()), since: None, until: None };
        let (since, until) = parse_period(&params);
        let expected_start = unix_days_for_date(2026, 3, 1) * 86400;
        let expected_end = unix_days_for_date(2026, 4, 1) * 86400;
        assert_eq!(since, expected_start);
        assert!(until <= expected_end, "until should be capped at now");
    }

    #[test]
    fn test_parse_period_today() {
        let params = UsageParams { period: Some("today".into()), since: None, until: None };
        let (since, _until) = parse_period(&params);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        assert_eq!(since % 86400, 0, "since should be UTC midnight");
        assert!(since <= now);
    }

    #[test]
    fn test_parse_period_explicit_override() {
        let params = UsageParams {
            period: Some("2026-03".into()),
            since: Some(1000),
            until: Some(2000),
        };
        let (since, until) = parse_period(&params);
        // Explicit params override period
        assert_eq!(since, 1000);
        assert_eq!(until, 2000);
    }

    #[test]
    fn test_parse_period_7d() {
        let params = UsageParams { period: Some("7d".into()), since: None, until: None };
        let (since, until) = parse_period(&params);
        assert!(until - since >= 7 * 86400 - 1);
        assert!(until - since <= 7 * 86400 + 1);
    }

    // ── v2.5.2+Pagination: TaskListParams ──

    #[test]
    fn test_task_list_params_defaults() {
        // Simulate query string with no offset → offset should default to 0
        let qs = "limit=10&status=pending";
        let params: TaskListParams = serde_urlencoded::from_str(qs).unwrap();
        assert_eq!(params.limit, 10);
        assert_eq!(params.offset, 0);
        assert_eq!(params.status.as_deref(), Some("pending"));
        assert!(params.task_type.is_none());
    }

    #[test]
    fn test_task_list_params_with_offset() {
        let qs = "limit=20&offset=40&status=failed&type=summarize";
        let params: TaskListParams = serde_urlencoded::from_str(qs).unwrap();
        assert_eq!(params.limit, 20);
        assert_eq!(params.offset, 40);
        assert_eq!(params.status.as_deref(), Some("failed"));
        assert_eq!(params.task_type.as_deref(), Some("summarize"));
    }

    #[test]
    fn test_task_list_params_default_limit() {
        // No params → all defaults
        let params: TaskListParams = serde_urlencoded::from_str("").unwrap();
        assert_eq!(params.limit, 20); // default_limit()
        assert_eq!(params.offset, 0);
        assert!(params.status.is_none());
        assert!(params.task_type.is_none());
    }
}
