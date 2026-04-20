// ============================================
// File: crates/aeronyx-server/src/api/admin_handlers.rs
// ============================================
//! # Admin Handlers — SaaS Operator Management Endpoints
//!
//! ## Modification History
//! v1.0.0-MultiTenant - Initial implementation (Task 5)
//! v1.0.1-Fix         - Three bug fixes:
//!   1. VolumeStatus serialization: replaced `format!("{:?}").to_lowercase()
//!      .replace('_', "-")` with `serde_json::to_value(&s.status)` so the
//!      kebab-case serde attribute is respected. Previous code produced
//!      "readwrite" instead of "read-write".
//!   2. days_to_ymd overflow guard: compute `doe` as i64 before casting to
//!      u64, preventing a u64 underflow panic for timestamps near Unix epoch
//!      (e.g. ts=0 caused `z - era*146097` to be negative going into u64).
//!   3. MT2 fix: admin_pool_stats now returns actual max_connections and
//!      idle_timeout_secs from MpiState instead of hard-coded 0. MpiState
//!      gains two new fields: pool_max_connections and pool_idle_timeout_secs.
//!      See mpi.rs and server.rs for the corresponding additions.
//!
//! ## Main Functionality
//! - `GET  /api/admin/volumes`        — Per-volume user count + capacity
//! - `POST /api/admin/volumes/reload` — Hot-reload volumes.toml (SIGHUP-free)
//! - `GET  /api/admin/pool/stats`     — StoragePool + VectorIndexPool live counts
//! - `GET  /api/admin/usage`          — Global LLM token usage by time period
//!
//! ## Authentication
//! All admin endpoints require `Authorization: Bearer <api_secret>`.
//! Enforced by `admin_auth_middleware` in mpi.rs — NOT the user JWT.
//! Admin is the server operator, not an end user.
//!
//! ## Usage Query Parameters
//! `GET /api/admin/usage` accepts two formats:
//! - `?period=2026-03`              — Calendar month (UTC, inclusive)
//! - `?since=<unix>&until=<unix>`   — Explicit Unix timestamp range
//! If neither is provided, defaults to the current calendar month.
//!
//! ## Dependencies
//! - VolumeRouter (Task 1a): volume_stats(), reload_config()
//! - StoragePool (Task 1b): active_count()
//! - VectorIndexPool (Task 1c): active_count()
//! - SystemDb (Task 1a): get_usage_stats()
//! - All accessed through MpiState SaaS fields
//!
//! ⚠️ Important Notes for Next Developer:
//! - These handlers return 404 in Local mode (SaaS fields are None).
//!   Routes are only registered in SaaS mode by build_mpi_router().
//! - disk_usage_bytes is always 0 — background disk scan is a future TODO.
//! - owner_hex in usage response is truncated to 8 chars for privacy.
//! - VolumeStatus serialization: always use serde_json on the enum value.
//!   Never use Debug formatting — it ignores the #[serde(rename_all)] attr.
//! - days_to_ymd: the `doe` intermediate must remain i64 until the final
//!   cast to u64. An early i64->u64 cast on a negative value causes panic.
//! - reload_config() is atomic: either new config fully replaces the old,
//!   or the old config is preserved on parse error.
//!
//! ## Last Modified
//! v1.0.1-Fix - VolumeStatus serialization fix; days_to_ymd overflow guard;
//!              MT2: pool stats now return real config values from MpiState.
// ============================================

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::{Query, State}, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::api::mpi::MpiState;

// ============================================
// Query Parameter Types
// ============================================

/// Query parameters for `GET /api/admin/usage`.
#[derive(Debug, Deserialize)]
pub struct UsageQuery {
    /// Calendar month in "YYYY-MM" format (e.g. "2026-03").
    pub period: Option<String>,
    /// Start of range as Unix timestamp (seconds, inclusive).
    pub since: Option<i64>,
    /// End of range as Unix timestamp (seconds, inclusive).
    pub until: Option<i64>,
}

// ============================================
// Response Types
// ============================================

#[derive(Debug, Serialize)]
struct VolumeEntry {
    id: String,
    path: String,
    /// Serialized from VolumeStatus via serde_json to respect kebab-case.
    /// Values: "read-write" | "read-only" | "draining"
    status: serde_json::Value,
    user_count: usize,
    max_users: usize,
    max_bytes: u64,
    disk_usage_bytes: u64,
}

#[derive(Debug, Serialize)]
struct VolumesResponse {
    volumes: Vec<VolumeEntry>,
    total_users: usize,
}

#[derive(Debug, Serialize)]
struct ReloadResponse {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    volumes_count: usize,
}

#[derive(Debug, Serialize)]
struct PoolStatsResponse {
    storage_pool: StoragePoolStats,
    vector_pool: VectorPoolStats,
}

#[derive(Debug, Serialize)]
struct StoragePoolStats {
    active_connections: usize,
    /// Previously always 0 (MT2). Now sourced from MpiState.pool_max_connections.
    max_connections: usize,
    /// Previously always 0 (MT2). Now sourced from MpiState.pool_idle_timeout_secs.
    idle_timeout_secs: u64,
}

#[derive(Debug, Serialize)]
struct VectorPoolStats {
    active_connections: usize,
}

#[derive(Debug, Serialize)]
struct UsageEntry {
    /// Truncated to 8 hex chars for privacy.
    owner: String,
    input_tokens: u64,
    output_tokens: u64,
    calls: u64,
}

#[derive(Debug, Serialize)]
struct UsageResponse {
    period: String,
    since: i64,
    until: i64,
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_calls: u64,
    by_owner: Vec<UsageEntry>,
}

// ============================================
// Handlers
// ============================================

/// `GET /api/admin/volumes`
///
/// Returns per-volume status including user counts and capacity.
pub async fn admin_volumes(
    State(state): State<Arc<MpiState>>,
) -> impl IntoResponse {
    let router = match &state.volume_router {
        Some(r) => r,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "not available in local mode" })),
            )
                .into_response();
        }
    };

    let stats = match router.volume_stats().await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "[ADMIN] volume_stats failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "failed to retrieve volume stats" })),
            )
                .into_response();
        }
    };

    let total_users: usize = stats.iter().map(|s| s.user_count).sum();

    let volumes: Vec<VolumeEntry> = stats
        .into_iter()
        .map(|s| {
            // FIX (v1.0.1): serialize VolumeStatus via serde_json so that the
            // #[serde(rename_all = "kebab-case")] attribute is honored.
            // The previous approach used Debug formatting which produced
            // "readwrite" instead of the correct "read-write".
            let status_val = serde_json::to_value(&s.status)
                .unwrap_or(serde_json::Value::String("unknown".into()));

            VolumeEntry {
                id: s.volume_id,
                path: s.path.to_string_lossy().into_owned(),
                status: status_val,
                user_count: s.user_count,
                max_users: s.max_users,
                max_bytes: s.max_bytes,
                disk_usage_bytes: s.disk_usage_bytes,
            }
        })
        .collect();

    Json(serde_json::to_value(VolumesResponse { volumes, total_users }).unwrap()).into_response()
}

/// `POST /api/admin/volumes/reload`
///
/// Hot-reloads volumes.toml. New volumes become available immediately.
/// Status changes take effect for new assignments only.
/// Returns 422 if the config file has a parse error; existing config preserved.
pub async fn admin_volumes_reload(
    State(state): State<Arc<MpiState>>,
) -> impl IntoResponse {
    let router = match &state.volume_router {
        Some(r) => r,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "not available in local mode" })),
            )
                .into_response();
        }
    };

    match router.reload_config().await {
        Ok(()) => {
            let volumes_count = match router.volume_stats().await {
                Ok(stats) => stats.len(),
                Err(_) => 0,
            };
            info!(volumes_count, "[ADMIN] volumes.toml reloaded");
            Json(serde_json::to_value(ReloadResponse {
                status: "ok",
                message: None,
                volumes_count,
            }).unwrap()).into_response()
        }
        Err(e) => {
            tracing::warn!(error = %e, "[ADMIN] volumes.toml reload failed");
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::to_value(ReloadResponse {
                    status: "error",
                    message: Some(e.to_string()),
                    volumes_count: 0,
                }).unwrap()),
            )
                .into_response()
        }
    }
}

/// `GET /api/admin/pool/stats`
///
/// Returns live connection counts for StoragePool and VectorIndexPool.
/// Also returns max_connections and idle_timeout_secs from MpiState config
/// (fixes MT2: these were previously always 0).
pub async fn admin_pool_stats(
    State(state): State<Arc<MpiState>>,
) -> impl IntoResponse {
    let (storage_active, vector_active) = match (&state.storage_pool, &state.vector_pool) {
        (Some(sp), Some(vp)) => (sp.active_count(), vp.active_count()),
        _ => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "not available in local mode" })),
            )
                .into_response();
        }
    };

    // FIX (MT2 / v1.0.1): read pool config from MpiState fields populated
    // during init_saas_mpi_state() in server.rs. Previously these were
    // hard-coded to 0. MpiState now carries pool_max_connections and
    // pool_idle_timeout_secs (see mpi.rs for field additions).
    let response = PoolStatsResponse {
        storage_pool: StoragePoolStats {
            active_connections: storage_active,
            max_connections: state.pool_max_connections,
            idle_timeout_secs: state.pool_idle_timeout_secs,
        },
        vector_pool: VectorPoolStats {
            active_connections: vector_active,
        },
    };

    Json(serde_json::to_value(response).unwrap()).into_response()
}

/// `GET /api/admin/usage?period=2026-03`
/// `GET /api/admin/usage?since=<unix>&until=<unix>`
///
/// Returns global LLM token usage aggregated by owner for a time period.
/// Owner hex is truncated to 8 chars in the response for privacy.
pub async fn admin_usage(
    State(state): State<Arc<MpiState>>,
    Query(params): Query<UsageQuery>,
) -> impl IntoResponse {
    let system_db = match &state.system_db {
        Some(db) => db,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "not available in local mode" })),
            )
                .into_response();
        }
    };

    let (since, until, period_label) = match resolve_time_range(&params) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": e })),
            )
                .into_response();
        }
    };

    let stats = match system_db.get_usage_stats(since, until).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(error = %e, "[ADMIN] get_usage_stats failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "failed to retrieve usage stats" })),
            )
                .into_response();
        }
    };

    let total_input: u64  = stats.iter().map(|s| s.total_input_tokens).sum();
    let total_output: u64 = stats.iter().map(|s| s.total_output_tokens).sum();
    let total_calls: u64  = stats.iter().map(|s| s.call_count).sum();

    let by_owner: Vec<UsageEntry> = stats
        .into_iter()
        .map(|s| UsageEntry {
            owner: hex::encode(s.owner_pubkey)[..8].to_string(),
            input_tokens: s.total_input_tokens,
            output_tokens: s.total_output_tokens,
            calls: s.call_count,
        })
        .collect();

    Json(serde_json::to_value(UsageResponse {
        period: period_label,
        since, until,
        total_input_tokens: total_input,
        total_output_tokens: total_output,
        total_calls,
        by_owner,
    }).unwrap()).into_response()
}

// ============================================
// Time Range Resolution
// ============================================

/// Resolve query parameters into a `(since, until, label)` triple.
///
/// Priority:
/// 1. `since` + `until` explicit Unix timestamps
/// 2. `period` calendar month ("YYYY-MM")
/// 3. Default: current UTC calendar month
fn resolve_time_range(params: &UsageQuery) -> Result<(i64, i64, String), &'static str> {
    if let (Some(since), Some(until)) = (params.since, params.until) {
        if since > until {
            return Err("since must be <= until");
        }
        let label = format!("{}-{}", ts_to_month(since), ts_to_month(until));
        return Ok((since, until, label));
    }
    if let Some(ref period) = params.period {
        return parse_period(period);
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let (since, until) = current_month_bounds(now);
    Ok((since, until, ts_to_month(since)))
}

/// Parse "YYYY-MM" into `(month_start_unix, month_end_unix, "YYYY-MM")`.
fn parse_period(period: &str) -> Result<(i64, i64, String), &'static str> {
    let parts: Vec<&str> = period.split('-').collect();
    if parts.len() != 2 {
        return Err("period must be in YYYY-MM format");
    }
    let year: i32  = parts[0].parse().map_err(|_| "period year is not a valid integer")?;
    let month: u32 = parts[1].parse().map_err(|_| "period month is not a valid integer")?;
    if !(1..=12).contains(&month)     { return Err("period month must be between 01 and 12"); }
    if !(1970..=9999).contains(&year) { return Err("period year out of range"); }

    let since = month_start_unix(year, month);
    let (next_year, next_month) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
    let until = month_start_unix(next_year, next_month) - 1;
    Ok((since, until, period.to_string()))
}

/// Return `(month_start, month_end)` Unix timestamps for the UTC month
/// containing `now_unix`.
fn current_month_bounds(now_unix: i64) -> (i64, i64) {
    let days_since_epoch = now_unix / 86400;
    let (year, month, _) = days_to_ymd(days_since_epoch);
    let since = month_start_unix(year, month);
    let (ny, nm) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
    let until = month_start_unix(ny, nm) - 1;
    (since, until)
}

/// Format a Unix timestamp as "YYYY-MM".
fn ts_to_month(ts: i64) -> String {
    let days = ts / 86400;
    let (year, month, _) = days_to_ymd(days);
    format!("{:04}-{:02}", year, month)
}

/// Compute the Unix timestamp of the first second of a UTC month.
fn month_start_unix(year: i32, month: u32) -> i64 {
    ymd_to_days(year, month, 1) * 86400
}

/// Convert days-since-epoch (1970-01-01 = day 0) to (year, month, day).
///
/// Uses Howard Hinnant's civil calendar algorithm.
/// https://howardhinnant.github.io/date_algorithms.html
///
/// FIX (v1.0.1): `days` is now accepted as `i64` (was `u64`) and `doe` is
/// computed as `i64` throughout before the final `u64` cast. The previous
/// `u64` parameter caused a panic for negative or near-zero timestamps
/// because `z - era * 146097` could be negative, which wrapped on cast.
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let z = days + 719468_i64;
    let era: i64 = if z >= 0 { z } else { z - 146096 } / 146097;
    // doe is always non-negative after this subtraction by construction of era.
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y   = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp  = (5 * doy + 2) / 153;
    let d   = doy - (153 * mp + 2) / 5 + 1;
    let m   = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m as u32, d as u32)
}

/// Convert (year, month, day) to days-since-epoch (1970-01-01 = 0).
fn ymd_to_days(year: i32, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year as i64 - 1 } else { year as i64 };
    let m = month as i64;
    let d = day as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Time range resolution --------------------------------------------

    #[test]
    fn test_parse_period_march() {
        let (since, until, label) = parse_period("2026-03").unwrap();
        assert_eq!(label, "2026-03");
        assert!(since > 0);
        assert_eq!(until - since, 31 * 86400 - 1); // March has 31 days
    }

    #[test]
    fn test_parse_period_february_non_leap() {
        let (since, until, _) = parse_period("2025-02").unwrap();
        assert_eq!(until - since, 28 * 86400 - 1);
    }

    #[test]
    fn test_parse_period_february_leap() {
        let (since, until, _) = parse_period("2024-02").unwrap();
        assert_eq!(until - since, 29 * 86400 - 1);
    }

    #[test]
    fn test_parse_period_december_wraps_year() {
        let (since, until, _) = parse_period("2025-12").unwrap();
        assert_eq!(until - since, 31 * 86400 - 1);
    }

    #[test]
    fn test_parse_period_invalid() {
        assert!(parse_period("2026").is_err());
        assert!(parse_period("2026-13").is_err());
        assert!(parse_period("abc-def").is_err());
    }

    #[test]
    fn test_explicit_since_until() {
        let params = UsageQuery { period: None, since: Some(1_700_000_000), until: Some(1_700_100_000) };
        let (since, until, _) = resolve_time_range(&params).unwrap();
        assert_eq!(since, 1_700_000_000);
        assert_eq!(until, 1_700_100_000);
    }

    #[test]
    fn test_since_after_until_rejected() {
        let params = UsageQuery { period: None, since: Some(2_000_000_000), until: Some(1_000_000_000) };
        assert!(resolve_time_range(&params).is_err());
    }

    #[test]
    fn test_default_is_current_month() {
        let params = UsageQuery { period: None, since: None, until: None };
        let (since, until, label) = resolve_time_range(&params).unwrap();
        assert!(since > 0);
        assert!(until >= since);
        assert_eq!(label.len(), 7);
        assert!(label.contains('-'));
    }

    // -- Calendar arithmetic ----------------------------------------------

    #[test]
    fn test_days_roundtrip_known_dates() {
        let cases: &[((i32, u32, u32), i64)] = &[
            ((1970, 1, 1), 0),
            ((1970, 1, 2), 1),
            ((2000, 1, 1), 10957),
            ((2026, 3, 30), 20542),
        ];
        for &((y, m, d), expected_days) in cases {
            let days = ymd_to_days(y, m, d);
            assert_eq!(days, expected_days, "ymd_to_days({},{},{})", y, m, d);
            let (yr, mo, da) = days_to_ymd(days);
            assert_eq!((yr, mo, da), (y, m, d), "days_to_ymd({})", days);
        }
    }

    #[test]
    fn test_days_to_ymd_epoch_zero_does_not_panic() {
        // FIX (v1.0.1): this previously panicked on the u64 cast of a
        // potentially negative intermediate value.
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_days_to_ymd_negative_days_does_not_panic() {
        // Timestamps before 1970 produce negative days — must not panic.
        let (y, m, d) = days_to_ymd(-1);
        assert_eq!((y, m, d), (1969, 12, 31));
    }

    #[test]
    fn test_ts_to_month() {
        let ts = 20542_i64 * 86400; // 2026-03-30 00:00:00 UTC
        assert_eq!(ts_to_month(ts), "2026-03");
    }

    #[test]
    fn test_month_start_unix_known() {
        let expected = ymd_to_days(2026, 3, 1) * 86400;
        assert_eq!(month_start_unix(2026, 3), expected);
    }
}
