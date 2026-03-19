// ============================================
// File: crates/aeronyx-server/src/api/supernode_handlers.rs
// ============================================
//! # SuperNode Management Endpoints
//!
//! ## Creation Reason (v2.5.0+SuperNode Phase C/D)
//! Provides operational visibility and control over the cognitive task queue
//! and LLM provider health. All endpoints are local-only (remote → 403).
//!
//! ## Endpoints (6)
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | GET  | /supernode/tasks?status=&type=&limit= | List tasks with optional filters |
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
//! ## SuperNode Disabled
//! When `MpiState.llm_router` is None, all endpoints return 404
//! `{"error": "supernode not enabled"}`.
//!
//! ## Period Format (usage endpoint)
//! - `"YYYY-MM"` — calendar month (e.g. "2026-03")
//! - `"today"` — current UTC day
//! - `"7d"` / `"30d"` — last N days
//! - `since` + `until` query params — explicit Unix timestamps (override period)
//! - No params — all time
//!
//! ⚠️ Important Note for Next Developer:
//! - GET /supernode/health makes live HTTP calls to each provider (ping test).
//!   Capped at 5s per provider. Use sparingly — not for polling.
//! - POST /supernode/tasks/:id/retry calls storage.retry_task() which increments
//!   retry_count (audit trail) but always allows the reset regardless of max_retries.
//! - Usage cost estimates in /usage are APPROXIMATE (see LlmRouter::estimate_cost).
//!
//! ## Last Modified
//! v2.5.0+SuperNode Phase C - 🌟 Created (6 endpoints).
//! v2.5.0+SuperNode Phase D - 🌟 tasks list adds `type=` filter;
//!   usage adds by_task_type breakdown; retry uses storage.retry_task();
//!   health uses count_tasks_by_status().

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

#[derive(Debug, Deserialize)]
pub struct TaskListParams {
    /// Filter by status: pending | processing | completed | failed | cancelled
    #[serde(default)]
    pub status: Option<String>,
    /// Filter by task_type: session_title | community_summary | entity_description | …
    #[serde(rename = "type", default)]
    pub task_type: Option<String>,
    /// Max results (default 20, max 100)
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize { 20 }

#[derive(Debug, Deserialize)]
pub struct UsageParams {
    /// "YYYY-MM" | "today" | "7d" | "30d"
    #[serde(default)]
    pub period: Option<String>,
    /// Explicit window start Unix timestamp (overrides period)
    #[serde(default)]
    pub since: Option<i64>,
    /// Explicit window end Unix timestamp (overrides period)
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
    model: String,
    healthy: bool,
    latency_ms: Option<u64>,
    error: Option<String>,
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
fn parse_period(params: &UsageParams) -> (i64, i64) {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Explicit params override period string
    if params.since.is_some() || params.until.is_some() {
        return (params.since.unwrap_or(0), params.until.unwrap_or(now));
    }

    match params.period.as_deref() {
        None => (0, now),
        Some("today") => {
            let start = now - (now % 86400);
            (start, now)
        }
        Some("7d")  => (now - 7 * 86400, now),
        Some("30d") => (now - 30 * 86400, now),
        Some(s) if s.len() == 7 => {
            // "YYYY-MM"
            let parts: Vec<&str> = s.splitn(2, '-').collect();
            if parts.len() == 2 {
                if let (Ok(year), Ok(month)) = (parts[0].parse::<i32>(), parts[1].parse::<u32>()) {
                    if (1..=12).contains(&month) {
                        let days_before_year = days_since_epoch(year);
                        let days_before_month: i64 =
                            (1..month).map(|m| days_in_month(year, m) as i64).sum();
                        let start = (days_before_year + days_before_month) * 86400;
                        let end = start + days_in_month(year, month) as i64 * 86400;
                        return (start, end.min(now));
                    }
                }
            }
            (0, now)
        }
        Some(_) => (0, now),
    }
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1|3|5|7|8|10|12 => 31,
        4|6|9|11 => 30,
        2 => if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 29 } else { 28 },
        _ => 30,
    }
}

fn days_since_epoch(year: i32) -> i64 {
    let y = (year - 1970) as i64;
    y * 365 + y / 4 - y / 100 + y / 400
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

pub async fn supernode_list_tasks(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<TaskListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    let limit = params.limit.min(100).max(1);

    let tasks = state.storage.get_tasks_filtered(
        params.status.as_deref(),
        params.task_type.as_deref(),
        limit,
    ).await;

    let summaries: Vec<TaskSummary> = tasks.iter().map(row_to_summary).collect();

    (StatusCode::OK, Json(serde_json::json!({
        "filters": {
            "status": params.status,
            "type": params.task_type,
            "limit": limit,
        },
        "count": summaries.len(),
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
        Err(e) => {
            warn!(id = task_id, error = %e, "[SUPERNODE] retry_task failed");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("failed to retry task: {}", e)
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

    // Verify task exists and is pending
    let task = match state.storage.get_task(task_id).await {
        Some(t) => t,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": format!("task {} not found", task_id)
        }))).into_response(),
    };

    if task.status != "pending" {
        return (StatusCode::CONFLICT, Json(serde_json::json!({
            "error": format!(
                "task {} is '{}', can only cancel 'pending' tasks",
                task_id, task.status
            )
        }))).into_response();
    }

    match state.storage.cancel_task(task_id).await {
        Ok(()) => {
            info!(id = task_id, "[SUPERNODE] Task cancelled");
            (StatusCode::OK, Json(serde_json::json!({
                "task_id": task_id,
                "status": "cancelled"
            }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": format!("failed to cancel task: {}", e)
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

    // Attach cost estimates per provider
    let by_provider_with_cost: Vec<serde_json::Value> = stats.by_provider.iter().map(|p| {
        let cost = LlmRouter::estimate_cost(
            &p.provider, p.input_tokens as u32, p.output_tokens as u32, 0,
        );
        serde_json::json!({
            "provider": p.provider,
            "calls": p.calls,
            "input_tokens": p.input_tokens,
            "output_tokens": p.output_tokens,
            "avg_latency_ms": format!("{:.0}", p.avg_latency_ms),
            "estimated_cost_usd": format!("{:.6}", cost),
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
            "avg_latency_ms": format!("{:.0}", t.avg_latency_ms),
            "estimated_cost_usd": format!("{:.6}", cost),
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
        },
        "totals": {
            "calls": stats.total_calls,
            "input_tokens": stats.total_input_tokens,
            "output_tokens": stats.total_output_tokens,
            "cached_tokens": stats.total_cached_tokens,
            "avg_latency_ms": format!("{:.0}", stats.avg_latency_ms),
            "estimated_cost_usd": format!("{:.6}", total_cost),
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

    // Queue summary from DB
    let counts = state.storage.count_tasks_by_status().await;
    let queue = QueueSummary {
        pending:    *counts.get("pending").unwrap_or(&0),
        processing: *counts.get("processing").unwrap_or(&0),
        completed:  *counts.get("completed").unwrap_or(&0),
        failed:     *counts.get("failed").unwrap_or(&0),
        cancelled:  *counts.get("cancelled").unwrap_or(&0),
    };

    // Live provider ping (5s timeout per provider)
    let provider_names = router.provider_names();
    let mut provider_health: Vec<ProviderHealthInfo> = Vec::new();

    for name in &provider_names {
        let test_req = crate::services::memchain::ChatRequest {
            messages: vec![crate::services::memchain::ChatMessage::user("ping")],
            model_override: None,
            max_tokens: Some(1),
            temperature: Some(0.0),
            stop: None,
        };

        let task_type = crate::services::memchain::CognitiveTaskType::CustomPrompt;
        let t0 = std::time::Instant::now();

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            router.route(&task_type, &test_req),
        ).await;

        let latency_ms = t0.elapsed().as_millis() as u64;

        let info = match result {
            Ok(Ok(resp)) => ProviderHealthInfo {
                name: name.to_string(),
                model: resp.model_used.clone(),
                healthy: true,
                latency_ms: Some(latency_ms),
                error: None,
            },
            Ok(Err(e)) => ProviderHealthInfo {
                name: name.to_string(),
                model: String::new(),
                healthy: false,
                latency_ms: Some(latency_ms),
                error: Some(e.to_string()),
            },
            Err(_) => ProviderHealthInfo {
                name: name.to_string(),
                model: String::new(),
                healthy: false,
                latency_ms: Some(5000),
                error: Some("timeout (5s)".to_string()),
            },
        };

        debug!(
            provider = %info.name, healthy = info.healthy,
            latency_ms = ?info.latency_ms,
            "[SUPERNODE_HEALTH] Provider check"
        );

        provider_health.push(info);
    }

    let all_healthy = provider_health.iter().all(|p| p.healthy);
    let any_healthy = provider_health.iter().any(|p| p.healthy);

    let overall = if all_healthy { "healthy" }
                  else if any_healthy { "degraded" }
                  else { "unhealthy" };

    let code = if any_healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (code, Json(serde_json::json!({
        "status": overall,
        "providers": provider_health,
        "queue": queue,
    }))).into_response()
}
