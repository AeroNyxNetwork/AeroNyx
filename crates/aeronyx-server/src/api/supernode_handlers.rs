// ============================================
// File: crates/aeronyx-server/src/api/supernode_handlers.rs
// ============================================
//! # SuperNode Management Endpoints
//!
//! ## Modification History (abbreviated — see original for full history)
//! v2.5.0+SuperNode Phase C - Created (6 endpoints).
//! v2.5.0+SuperNode Phase D - tasks list + type filter; usage breakdown; health.
//! v2.5.2+Pagination        - offset param + has_more in response.
//! v1.0.1-SaaSFix          - Extract storage from Extensions instead of state.storage
//!                            (state.storage is None in SaaS mode → would panic).
//!
//! ⚠️ Important Notes for Next Developer:
//! - All endpoints enforce local-only access via `AuthenticatedOwner::is_remote()`.
//! - GET /supernode/health uses HTTP HEAD to check reachability only — no LLM quota.
//! - storage is extracted from request Extensions (injected by unified_auth_middleware).
//!   Never access state.storage directly.
//!
//! ## Last Modified
//! v1.0.1-SaaSFix - Extract storage from Extensions for SaaS compatibility.
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
use crate::services::memchain::{LlmRouter, MemoryStorage};

// ============================================
// Helper: extract storage from Extensions (v1.0.1-SaaSFix)
// ============================================

fn get_storage(
    req: &Request<axum::body::Body>,
    state: &MpiState,
) -> Option<Arc<MemoryStorage>> {
    req.extensions()
        .get::<Arc<MemoryStorage>>()
        .cloned()
        .or_else(|| state.storage.clone())
}

// ============================================
// Query Param Types
// ============================================

#[derive(Debug, Deserialize)]
pub struct TaskListParams {
    #[serde(default)]
    pub status: Option<String>,
    #[serde(rename = "type", default)]
    pub task_type: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
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
    model: String,
    healthy: bool,
    latency_ms: Option<u64>,
    error: Option<String>,
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

fn storage_unavailable() -> impl IntoResponse {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
        "error": "storage unavailable"
    })))
}

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
            let start = now - (now % 86400);
            (start, now)
        }
        Some("7d")  => (now - 7 * 86400, now),
        Some("30d") => (now - 30 * 86400, now),
        Some(s) if s.len() == 7 => parse_year_month(s, now),
        Some(_) => (0, now),
    }
}

fn parse_year_month(s: &str, now: i64) -> (i64, i64) {
    let parts: Vec<&str> = s.splitn(2, '-').collect();
    if parts.len() != 2 { return (0, now); }
    let (Ok(year), Ok(month)) = (parts[0].parse::<i32>(), parts[1].parse::<u32>()) else {
        return (0, now);
    };
    if !(1..=12).contains(&month) { return (0, now); }

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

fn unix_days_for_date(year: i32, month: u32, day: u32) -> i64 {
    fn days_from_year1(y: i32) -> i64 {
        let y = y as i64 - 1;
        y * 365 + y / 4 - y / 100 + y / 400
    }
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

pub async fn supernode_list_tasks(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<TaskListParams>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    if auth.is_remote() { return local_only().into_response(); }
    if state.llm_router.is_none() { return supernode_disabled().into_response(); }

    let storage = match get_storage(&req, &state) {
        Some(s) => s,
        None => return storage_unavailable().into_response(),
    };

    let limit = params.limit.min(100).max(1);
    let offset = params.offset;

    let tasks: Vec<crate::services::memchain::CognitiveTaskRow> = storage.get_tasks_filtered(
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

    let storage = match get_storage(&req, &state) {
        Some(s) => s,
        None => return storage_unavailable().into_response(),
    };

    let task = match storage.get_task(task_id).await {
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

    let storage = match get_storage(&req, &state) {
        Some(s) => s,
        None => return storage_unavailable().into_response(),
    };

    match storage.retry_task(task_id).await {
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
        Err(e) if e.contains("absolute retry ceiling") => {
            warn!(id = task_id, "[SUPERNODE] Retry blocked by absolute ceiling");
            (StatusCode::UNPROCESSABLE_ENTITY, Json(serde_json::json!({
                "error": e,
                "hint": "Investigate the provider error before retrying."
            }))).into_response()
        }
        Err(e) => {
            warn!(id = task_id, error = %e, "[SUPERNODE] retry_task failed");
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

    let storage = match get_storage(&req, &state) {
        Some(s) => s,
        None => return storage_unavailable().into_response(),
    };

    let task = match storage.get_task(task_id).await {
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

    match storage.cancel_task(task_id).await {
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
                    "task {} could not be cancelled — it may have been claimed by the worker",
                    task_id
                )
            }))).into_response()
        }
        Ok(n) => {
            warn!(id = task_id, rows = n, "[SUPERNODE] cancel_task affected unexpected row count");
            (StatusCode::OK, Json(serde_json::json!({ "task_id": task_id, "status": "cancelled" }))).into_response()
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
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

    let storage = match get_storage(&req, &state) {
        Some(s) => s,
        None => return storage_unavailable().into_response(),
    };

    let (since, until) = parse_period(&params);
    let stats = storage.get_usage_stats(since, until).await;
    let by_task_type: Vec<crate::services::memchain::TaskTypeUsage> =
        storage.get_usage_stats_by_task_type(since, until).await;

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

    let storage = match get_storage(&req, &state) {
        Some(s) => s,
        None => return storage_unavailable().into_response(),
    };

    let counts: std::collections::HashMap<String, i64> = storage.count_tasks_by_status().await;
    let queue = QueueSummary {
        pending:    *counts.get("pending").unwrap_or(&0),
        processing: *counts.get("processing").unwrap_or(&0),
        completed:  *counts.get("completed").unwrap_or(&0),
        failed:     *counts.get("failed").unwrap_or(&0),
        cancelled:  *counts.get("cancelled").unwrap_or(&0),
    };

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
            let result = client.head(&api_base).send().await;
            let latency_ms = t0.elapsed().as_millis() as u64;

            match result {
                Ok(resp) => {
                    debug!(
                        provider = %name,
                        status = resp.status().as_u16(),
                        latency_ms,
                        "[SUPERNODE_HEALTH] Provider reachable"
                    );
                    ProviderHealthInfo {
                        name, model, healthy: true,
                        latency_ms: Some(latency_ms), error: None,
                        check_type: "http_head",
                    }
                }
                Err(e) => {
                    warn!(provider = %name, error = %e, "[SUPERNODE_HEALTH] Provider unreachable");
                    ProviderHealthInfo {
                        name, model, healthy: false,
                        latency_ms: Some(latency_ms), error: Some(e.to_string()),
                        check_type: "http_head",
                    }
                }
            }
        }
    }).collect();

    let provider_health: Vec<ProviderHealthInfo> =
        futures::future::join_all(ping_futures).await;

    let all_healthy = provider_health.iter().all(|p| p.healthy);
    let any_healthy = provider_health.iter().any(|p| p.healthy);
    let overall = if all_healthy { "healthy" } else if any_healthy { "degraded" } else { "unhealthy" };
    let code = if any_healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (code, Json(serde_json::json!({
        "status": overall,
        "note": "Provider health checks connectivity only (HTTP HEAD).",
        "providers": provider_health,
        "queue": queue,
    }))).into_response()
}

// ============================================
// Tests (preserved verbatim)
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unix_days_for_date_known_values() {
        assert_eq!(unix_days_for_date(1970, 1, 1), 0);
        assert_eq!(unix_days_for_date(1970, 1, 2), 1);
        assert_eq!(unix_days_for_date(1970, 2, 1), 31);
        assert_eq!(unix_days_for_date(1972, 1, 1), 730);
        assert_eq!(unix_days_for_date(1972, 3, 1), 790);
        assert_eq!(unix_days_for_date(2026, 1, 1), 20089);
        let y2100 = unix_days_for_date(2100, 1, 1);
        let y2099 = unix_days_for_date(2099, 1, 1);
        assert_eq!(y2100 - y2099, 365);
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
        assert!(until <= expected_end);
    }

    #[test]
    fn test_parse_period_today() {
        let params = UsageParams { period: Some("today".into()), since: None, until: None };
        let (since, _) = parse_period(&params);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        assert_eq!(since % 86400, 0);
        assert!(since <= now);
    }

    #[test]
    fn test_parse_period_explicit_override() {
        let params = UsageParams { period: Some("2026-03".into()), since: Some(1000), until: Some(2000) };
        let (since, until) = parse_period(&params);
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

    #[test]
    fn test_task_list_params_defaults() {
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
        let params: TaskListParams = serde_urlencoded::from_str("").unwrap();
        assert_eq!(params.limit, 20);
        assert_eq!(params.offset, 0);
        assert!(params.status.is_none());
        assert!(params.task_type.is_none());
    }
}
