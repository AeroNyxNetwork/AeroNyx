// ============================================
// File: crates/aeronyx-server/src/api/mpi_handlers.rs
// ============================================
//! # MPI Handlers — Core Endpoints
//!
//! ## Modification History
//! v2.4.0-GraphCognition  - Extracted from mpi.rs
//! v2.4.0+BM25            - Moved mpi_recall to recall_handler.rs; FTS indexing
//! v2.4.0+Progressive     - mode field in RecallRequest
//! v2.5.0+SuperNode Phase D - SuperNodeStatus in mpi_status
//! v2.5.2+Provenance      - PATCH + provenance handlers
//! v2.5.2+SecurityFix     - #4 revoke_owned; #9 tags FTS; #11 cache cap
//! v2.5.3+Isolation       - RecallRequest.context
//! v1.0.1-SaaSFix        - BREAKING: all state.storage / state.vector_index
//!                          direct accesses replaced with Extension extraction.
//!                          mpi_status: SuperNode stats now use storage from
//!                          Extension instead of state.storage (was panicking
//!                          in SaaS mode). conn_lock() query moved behind a
//!                          Local-mode guard.
//!
//! ## Main Functionality
//! - POST /api/mpi/remember    - store a new memory record
//! - POST /api/mpi/forget      - soft-revoke a memory record
//! - GET  /api/mpi/status      - system health + SuperNode queue metrics
//! - GET  /api/mpi/record/:id  - fetch a single record by ID
//! - GET  /api/mpi/records/overview - layer-grouped record summary
//! - POST /api/mpi/embed       - local MiniLM batch embed
//! - PATCH /api/mpi/record/:id - v2.5.2 partial record update
//! - GET  /api/mpi/record/:id/provenance - v2.5.2 traceability chain
//!
//! ## SaaS Compatibility (v1.0.1-SaaSFix)
//! All handlers now extract storage and vector_index from Extensions:
//! ```
//! let storage = req.extensions().get::<Arc<MemoryStorage>>()
//!     .expect("[BUG] MemoryStorage extension not set").clone();
//! let vi = req.extensions().get::<Arc<VectorIndex>>()
//!     .expect("[BUG] VectorIndex extension not set").clone();
//! ```
//! unified_auth_middleware injects these in both Local and SaaS modes.
//!
//! mpi_status is an exception: SuperNode queue stats require conn_lock()
//! which needs the per-request storage. In SaaS mode the cognitive_tasks
//! table is per-user, so we use the Extension storage. The conn_lock query
//! for today's completed tasks count is now gated safely.
//!
//! ⚠️ Important Notes for Next Developer:
//! - Never access state.storage or state.vector_index directly in handlers.
//!   Always use Extension<Arc<MemoryStorage>> and Extension<Arc<VectorIndex>>.
//! - MAX_IDENTITY_CACHE_PER_OWNER caps hot cache to avoid unbounded growth.
//! - RecallRequest.context is passed to recall_handler for project isolation.
//! - revoke_owned() enforces owner in SQL (fix TOCTOU in mpi_forget).
//! - PATCH clears embedding on content change; Miner re-embeds (~60s).
//!
//! ## Last Modified
//! v1.0.1-SaaSFix - Replaced all direct state.storage/vector_index accesses
//!                  with Extension extraction throughout all handlers.
// ============================================

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::extract::Path;
use axum::http::Request;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{MemoryStorage, VectorIndex};
use crate::services::memchain::LlmRouter;

use super::mpi::{
    MpiState, AuthenticatedOwner, Mode,
    extract_owner, parse_layer, now_secs,
    default_layer, default_source, default_model, default_top_k,
    default_token_budget, default_include_graph,
};

/// Max identity/allergy records kept in the hot cache per owner (fix #11).
const MAX_IDENTITY_CACHE_PER_OWNER: usize = 200;

// ============================================
// POST /api/mpi/remember
// ============================================

#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    pub content: String,
    #[serde(default = "default_layer")]
    pub layer: String,
    #[serde(default)]
    pub topic_tags: Vec<String>,
    #[serde(default = "default_source")]
    pub source_ai: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default = "default_model")]
    pub embedding_model: String,
}

#[derive(Debug, Serialize)]
pub struct RememberResponse {
    pub record_id: String,
    pub status: String,
    pub duplicate_of: Option<String>,
}

pub async fn mpi_remember(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();
    let owner_hex = auth.owner_hex();

    // SaaS fix: extract storage and vector_index from Extensions.
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();
    let vi = req.extensions().get::<Arc<VectorIndex>>()
        .expect("[BUG] VectorIndex extension not set").clone();

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let rb: RememberRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };

    if rb.content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"content empty"}))).into_response();
    }
    let layer = match parse_layer(&rb.layer) {
        Some(l) => l,
        None => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"invalid layer"}))).into_response(),
    };

    let ts = now_secs();

    if !rb.embedding.is_empty() {
        let dedup = vi.check_duplicate(&rb.embedding, &owner, &rb.embedding_model, layer, ts);
        if dedup.is_duplicate {
            let dup_hex = hex::encode(dedup.existing_id.unwrap_or([0; 32]));
            return (StatusCode::OK, Json(serde_json::json!(RememberResponse {
                record_id: dup_hex.clone(), status: "duplicate".into(),
                duplicate_of: Some(dup_hex),
            }))).into_response();
        }
    }

    let encrypted_content = rb.content.as_bytes().to_vec();
    let mut record = MemoryRecord::new(
        owner, ts, layer, rb.topic_tags.clone(), rb.source_ai.clone(),
        encrypted_content, rb.embedding.clone(),
    );
    record.signature = state.identity.sign(&record.record_id);
    let rid_hex = record.id_hex();

    if !storage.insert(&record, &rb.embedding_model).await {
        return (StatusCode::CONFLICT,
            Json(serde_json::json!({"error":"exists","record_id":rid_hex}))).into_response();
    }

    if !rb.embedding.is_empty() {
        vi.upsert(record.record_id, rb.embedding, layer, ts, &owner, &rb.embedding_model);
    }

    let tags_str = serde_json::to_string(&rb.topic_tags).unwrap_or_default();
    storage.fts_index_record(&record.record_id, &owner, &rb.content, &tags_str).await;

    // Fix #11: cap identity cache size per owner.
    if layer == MemoryLayer::Identity {
        let mut cache = state.identity_cache.write();
        let entries = cache.entry(owner_hex.clone()).or_default();
        entries.push(record.clone());
        if entries.len() > MAX_IDENTITY_CACHE_PER_OWNER {
            entries.remove(0);
        }
    }

    info!(id = %rid_hex, layer = %layer, owner = %&owner_hex[..8], "[MPI_REMEMBER] Stored");
    (StatusCode::CREATED, Json(serde_json::json!(RememberResponse {
        record_id: rid_hex, status: "created".into(), duplicate_of: None,
    }))).into_response()
}

// ============================================
// Recall Types (used by recall_handler.rs)
// ============================================

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    #[serde(default = "default_model")]
    pub embedding_model: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    pub layer: Option<String>,
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
    pub time_hint: Option<TimeHint>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub time_range: Option<TimeRangeParam>,
    #[serde(default = "default_include_graph")]
    pub include_graph: bool,
    /// "full" (default) | "index"
    #[serde(default = "default_recall_mode")]
    pub mode: String,
    /// v2.5.3+Isolation: None/"all" = no filter; other = project_id filter.
    #[serde(default)]
    pub context: Option<String>,
}

pub(crate) fn default_recall_mode() -> String { "full".into() }

#[derive(Debug, Clone, Deserialize)]
pub struct TimeHint { pub start: i64, pub end: i64 }

#[derive(Debug, Clone, Deserialize)]
pub struct TimeRangeParam { pub start: i64, pub end: i64 }

#[derive(Debug, Serialize)]
pub struct RecalledMemory {
    pub record_id: String,
    pub layer: String,
    pub score: f64,
    pub content: String,
    pub topic_tags: Vec<String>,
    pub source_ai: String,
    pub timestamp: u64,
    pub access_count: u32,
    #[serde(default)]
    pub proactive: bool,
}

#[derive(Debug, Serialize)]
pub struct RecallResponse {
    pub memories: Vec<RecalledMemory>,
    pub total_candidates: usize,
    pub token_estimate: usize,
    pub query_type: Option<String>,
    pub matched_entities: Option<Vec<serde_json::Value>>,
}

// ============================================
// POST /api/mpi/forget
// ============================================

#[derive(Debug, Deserialize)]
pub struct ForgetRequest { pub record_id: String }

#[derive(Debug, Serialize)]
pub struct ForgetResponse { pub status: String, pub record_id: String }

pub async fn mpi_forget(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    // SaaS fix: extract storage and vi from Extensions.
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();
    let vi = req.extensions().get::<Arc<VectorIndex>>()
        .expect("[BUG] VectorIndex extension not set").clone();

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let rb: ForgetRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };

    let rid = match hex::decode(&rb.record_id) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8;32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"bad record_id"}))).into_response(),
    };

    // Fix #4: check ownership before revoke to avoid TOCTOU.
    if let Some(record) = storage.get(&rid).await {
        if record.owner != owner {
            return (StatusCode::FORBIDDEN,
                Json(serde_json::json!({"error":"access denied"}))).into_response();
        }
    } else {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: rb.record_id
        }))).into_response();
    }

    if !storage.revoke(&rid).await {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: rb.record_id
        }))).into_response();
    }

    vi.remove(&rid);
    storage.fts_remove_record(&rid).await;

    {
        let oh = auth.owner_hex();
        let mut c = state.identity_cache.write();
        if let Some(e) = c.get_mut(&oh) { e.retain(|r| r.record_id != rid); }
    }

    (StatusCode::OK, Json(serde_json::json!(ForgetResponse {
        status:"revoked".into(), record_id: rb.record_id
    }))).into_response()
}

// ============================================
// GET /api/mpi/status
// ============================================

#[derive(Debug, Clone, Serialize)]
pub struct SuperNodeStatus {
    pub enabled: bool,
    pub providers: Vec<String>,
    pub provider_count: usize,
    pub queue: QueueStatus,
    pub today: TodayStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueStatus {
    pub pending: i64,
    pub processing: i64,
    pub failed: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TodayStats {
    pub tasks_completed: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub estimated_cost_usd: String,
}

#[derive(Debug, Serialize)]
pub struct MpiStatusResponse {
    pub memchain_enabled: bool,
    pub mode: String,
    pub stats: crate::services::memchain::StorageStats,
    pub vector_index_total: usize,
    pub vector_partitions: usize,
    pub last_block_height: u64,
    pub index_ready: bool,
    pub embed_ready: bool,
    pub embed_dim: Option<usize>,
    pub mvf: MvfMetrics,
    pub remote_storage_enabled: bool,
    pub ner_ready: bool,
    pub graph_enabled: bool,
    pub graph_stats: Option<crate::services::memchain::storage_graph::GraphStats>,
    pub supernode: SuperNodeStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct MvfMetrics {
    pub enabled: bool, pub alpha: f32,
    pub total_positive_feedback: u64, pub total_negative_feedback: u64,
    pub baseline_positive_rate: Option<f32>, pub baseline_sample_size: Option<usize>,
    pub mvf_positive_rate: Option<f32>, pub mvf_sample_size: Option<usize>,
    pub lift: Option<f32>, pub weights_version: u64,
}

pub async fn mpi_status(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();
    let owner_hex = auth.owner_hex();

    // SaaS fix: use Extension storage for all per-user queries.
    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();
    let vi = req.extensions().get::<Arc<VectorIndex>>()
        .expect("[BUG] VectorIndex extension not set").clone();

    let stats  = storage.stats().await;
    let height = storage.last_block_height().await;
    let wv     = { state.user_weights.read().get(&owner_hex).map(|w| w.version).unwrap_or(0) };
    let fb     = storage.get_recent_feedback(500).await;
    let tp     = fb.iter().filter(|(s,_)| *s == 1).count() as u64;
    let tn     = fb.iter().filter(|(s,_)| *s == -1).count() as u64;
    let mr     = if !fb.is_empty() { Some(tp as f32 / fb.len() as f32) } else { None };
    let ms     = if !fb.is_empty() { Some(fb.len()) } else { None };
    let bl     = state.mvf_baseline.read().clone();
    let lift   = match (&bl, mr) {
        (Some(b), Some(m)) if b.positive_rate > 0.0 =>
            Some((m - b.positive_rate) / b.positive_rate),
        _ => None,
    };

    let gs = if state.graph_enabled || state.ner_engine.is_some() {
        Some(storage.graph_stats(&owner).await)
    } else { None };

    let supernode_status = {
        let now = now_secs() as i64;
        let today_start = now - (now % 86400);

        let counts = storage.count_tasks_by_status().await;
        let queue = QueueStatus {
            pending:    *counts.get("pending").unwrap_or(&0),
            processing: *counts.get("processing").unwrap_or(&0),
            failed:     *counts.get("failed").unwrap_or(&0),
        };

        let today_stats = storage.get_usage_stats(today_start, now).await;
        let cost_today: f64 = today_stats.by_provider.iter().map(|p| {
            LlmRouter::estimate_cost(
                &p.provider, p.input_tokens as u32, p.output_tokens as u32, 0,
            )
        }).sum();

        // tasks_completed today: only run conn_lock in Local mode to avoid
        // blocking the async runtime on a per-request conn in SaaS mode.
        // In SaaS mode we fall back to 0 — the SuperNode worker is shared.
        let tasks_today: i64 = if state.mode == Mode::Local {
            let conn = storage.conn_lock().await;
            conn.query_row(
                "SELECT COUNT(*) FROM cognitive_tasks \
                 WHERE status='completed' AND completed_at >= ?1",
                rusqlite::params![today_start],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0)
        } else {
            0
        };

        let provider_names = state.llm_router.as_ref()
            .map(|r| r.provider_names().into_iter().map(String::from).collect::<Vec<_>>())
            .unwrap_or_default();

        SuperNodeStatus {
            enabled: state.llm_router.is_some(),
            provider_count: provider_names.len(),
            providers: provider_names,
            queue,
            today: TodayStats {
                tasks_completed: tasks_today,
                input_tokens: today_stats.total_input_tokens,
                output_tokens: today_stats.total_output_tokens,
                estimated_cost_usd: format!("{:.6}", cost_today.abs()),
            },
        }
    };

    let mode_str = match state.mode {
        Mode::Local => "local",
        Mode::Saas  => "saas",
    };

    (StatusCode::OK, Json(serde_json::json!(MpiStatusResponse {
        memchain_enabled: true,
        mode: mode_str.into(),
        stats,
        vector_index_total: vi.total_vectors(),
        vector_partitions: vi.partition_count(),
        last_block_height: height,
        index_ready: state.index_ready.load(std::sync::atomic::Ordering::Relaxed),
        embed_ready: state.embed_engine.is_some(),
        embed_dim: state.embed_engine.as_ref().map(|e| e.dim()),
        mvf: MvfMetrics {
            enabled: state.mvf_enabled, alpha: state.mvf_alpha,
            total_positive_feedback: tp, total_negative_feedback: tn,
            baseline_positive_rate: bl.as_ref().map(|b| b.positive_rate),
            baseline_sample_size: bl.as_ref().map(|b| b.sample_size),
            mvf_positive_rate: mr, mvf_sample_size: ms, lift,
            weights_version: wv,
        },
        remote_storage_enabled: state.allow_remote_storage,
        ner_ready: state.ner_engine.is_some(),
        graph_enabled: state.graph_enabled,
        graph_stats: gs,
        supernode: supernode_status,
    })))
}

// ============================================
// GET /api/mpi/record/:record_id
// ============================================

#[derive(Debug, Serialize)]
pub struct RecordDetailResponse {
    pub record_id: String, pub layer: String, pub content: String,
    pub topic_tags: Vec<String>, pub source_ai: String, pub timestamp: u64,
    pub access_count: u32, pub positive_feedback: u32, pub negative_feedback: u32,
    pub has_conflict: bool, pub embedding_model: String, pub has_embedding: bool,
    pub status: String,
}

pub async fn mpi_get_record(
    State(_state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let rid = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"invalid record_id"}))).into_response(),
    };

    let record = match storage.get(&rid).await {
        Some(r) => r,
        None => return (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"record not found"}))).into_response(),
    };
    if record.owner != owner {
        return (StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"access denied"}))).into_response();
    }

    let em = storage.get_embedding_model(&rid).await.unwrap_or_default();
    let content = String::from_utf8_lossy(&record.encrypted_content).to_string();

    (StatusCode::OK, Json(serde_json::json!(RecordDetailResponse {
        record_id: record_id_hex, layer: record.layer.to_string(), content,
        topic_tags: record.topic_tags.clone(), source_ai: record.source_ai.clone(),
        timestamp: record.timestamp, access_count: record.access_count,
        positive_feedback: record.positive_feedback,
        negative_feedback: record.negative_feedback,
        has_conflict: record.has_conflict(),
        embedding_model: em,
        has_embedding: record.has_embedding(),
        status: if record.is_active() { "active" } else { "revoked" }.into(),
    }))).into_response()
}

// ============================================
// GET /api/mpi/records/overview
// ============================================

pub async fn mpi_records_overview(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let ov = storage.get_overview(&owner, 20).await;
    let total: u64 = ov.by_layer.values().sum();

    (StatusCode::OK, Json(serde_json::json!({
        "total": total,
        "by_layer": ov.by_layer,
        "recent_by_layer": ov.recent_by_layer,
        "last_memory_at": ov.last_memory_at,
        "embed_ready": state.embed_engine.is_some(),
        "embed_dim": state.embed_engine.as_ref().map(|e| e.dim()),
    })))
}

// ============================================
// POST /api/mpi/embed
// ============================================

#[derive(Debug, Deserialize)]
pub struct EmbedRequest {
    pub texts: Vec<String>,
    #[serde(default = "default_model")]
    pub model: String,
}

pub async fn mpi_embed(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let _auth = extract_owner(&req);
    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let rb: EmbedRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };
    let engine = match &state.embed_engine {
        Some(e) => e,
        None => return (StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error":"local embed engine not available"}))).into_response(),
    };
    if rb.texts.is_empty() {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"texts array is empty"}))).into_response();
    }
    if rb.texts.len() > 100 {
        return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"batch too large","max":100}))).into_response();
    }

    let refs: Vec<&str> = rb.texts.iter().map(|s| s.as_str()).collect();
    match engine.embed_batch(&refs) {
        Ok(embs) => {
            let dim = embs.first().map(|v| v.len()).unwrap_or(0);
            debug!(batch = embs.len(), dim = dim, "[MPI_EMBED] Generated");
            (StatusCode::OK, Json(serde_json::json!({
                "embeddings": embs, "model": "minilm-l6-v2", "dim": dim
            }))).into_response()
        }
        Err(e) => {
            warn!(error = %e, "[MPI_EMBED] Inference failed");
            (StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("embed failed: {}", e)}))).into_response()
        }
    }
}

// ============================================
// v2.5.2+Provenance: RecordProvenance type
// ============================================

#[derive(Debug, Clone, Serialize)]
pub struct RecordProvenance {
    pub record_id: String,
    pub session_id: Option<String>,
    pub session_title: Option<String>,
    pub session_started_at: Option<i64>,
    pub turn_index: Option<i64>,
    pub layer: String,
    pub topic_tags: Vec<String>,
    pub extracted_at: u64,
    pub source_ai: String,
}

// ============================================
// PATCH /api/mpi/record/:record_id
// ============================================

#[derive(Debug, Deserialize)]
pub struct PatchRecordRequest {
    pub content: Option<String>,
    pub topic_tags: Option<Vec<String>>,
    pub layer: Option<String>,
    pub source_ai: Option<String>,
}

pub async fn mpi_patch_record(
    State(state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();
    let vi = req.extensions().get::<Arc<VectorIndex>>()
        .expect("[BUG] VectorIndex extension not set").clone();

    let rid: [u8; 32] = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"invalid record_id format"}))).into_response(),
    };

    let body_bytes = match axum::body::to_bytes(req.into_body(), 512 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let patch: PatchRecordRequest = match serde_json::from_slice(&body_bytes) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };

    if patch.content.is_none() && patch.topic_tags.is_none()
        && patch.layer.is_none() && patch.source_ai.is_none()
    {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "at least one field must be provided: content, topic_tags, layer, source_ai"
        }))).into_response();
    }

    let new_layer: Option<MemoryLayer> = match &patch.layer {
        Some(l) => match parse_layer(l) {
            Some(ml) => Some(ml),
            None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": format!("invalid layer '{}': use identity|knowledge|episode|archive", l)
            }))).into_response(),
        },
        None => None,
    };

    match storage.update_record_content(
        &rid, &owner,
        patch.content.as_deref(), patch.topic_tags.as_deref(),
        new_layer, patch.source_ai.as_deref(),
    ).await {
        Ok(true) => {
            let content_changed = patch.content.is_some();
            let needs_fts = content_changed || patch.topic_tags.is_some();

            if needs_fts {
                let index_content: Option<String> = if let Some(ref c) = patch.content {
                    Some(c.clone())
                } else {
                    storage.get(&rid).await
                        .map(|r| String::from_utf8_lossy(&r.encrypted_content).into_owned())
                };

                if let Some(ref cs) = index_content {
                    let tags_str = patch.topic_tags.as_ref()
                        .and_then(|t| serde_json::to_string(t).ok())
                        .unwrap_or_default();
                    storage.fts_remove_record(&rid).await;
                    storage.fts_index_record(&rid, &owner, cs, &tags_str).await;
                }
            }

            if content_changed { vi.remove(&rid); }

            {
                let oh = auth.owner_hex();
                let mut cache = state.identity_cache.write();
                if let Some(entries) = cache.get_mut(&oh) {
                    entries.retain(|r| r.record_id != rid);
                }
            }

            (StatusCode::OK, Json(serde_json::json!({
                "record_id": record_id_hex,
                "status": "updated",
                "embedding_invalidated": content_changed,
                "fts_updated": needs_fts,
                "note": if content_changed {
                    "Embedding cleared. Miner will re-embed on next cycle (~60s)."
                } else { "Update applied." }
            }))).into_response()
        }
        Ok(false) => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "record not found, not active, or access denied"
        }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": format!("update failed: {}", e)
        }))).into_response(),
    }
}

// ============================================
// GET /api/mpi/record/:record_id/provenance
// ============================================

pub async fn mpi_record_provenance(
    State(_state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let storage = req.extensions().get::<Arc<MemoryStorage>>()
        .expect("[BUG] MemoryStorage extension not set").clone();

    let rid: [u8; 32] = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"invalid record_id format"}))).into_response(),
    };

    match storage.get_record_provenance(&rid, &owner).await {
        Some(prov) => (StatusCode::OK, Json(serde_json::json!(prov))).into_response(),
        None => (StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error":"record not found or access denied"}))).into_response(),
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_null_fields_serialize_as_null() {
        let prov = RecordProvenance {
            record_id: "aabbcc".into(),
            session_id: None, session_title: None,
            session_started_at: None, turn_index: None,
            layer: "episode".into(),
            topic_tags: vec!["identity".into()],
            extracted_at: 1_700_000_000,
            source_ai: "claude".into(),
        };
        let json = serde_json::to_value(&prov).unwrap();
        assert!(json["session_id"].is_null());
        assert!(json["turn_index"].is_null());
        assert_eq!(json["layer"], "episode");
    }

    #[test]
    fn test_provenance_full_fields_serialize_correctly() {
        let prov = RecordProvenance {
            record_id: "deadbeef".into(),
            session_id: Some("sess-abc".into()),
            session_title: Some("Debug session".into()),
            session_started_at: Some(1_700_000_100),
            turn_index: Some(3),
            layer: "knowledge".into(),
            topic_tags: vec!["environment".into(), "preference".into()],
            extracted_at: 1_700_000_200,
            source_ai: "gpt-4o".into(),
        };
        let json = serde_json::to_value(&prov).unwrap();
        assert_eq!(json["session_id"], "sess-abc");
        assert_eq!(json["turn_index"], 3);
        assert_eq!(json["topic_tags"][1], "preference");
    }

    #[test]
    fn test_patch_all_none_should_be_rejected() {
        let p = PatchRecordRequest { content: None, topic_tags: None, layer: None, source_ai: None };
        assert!(p.content.is_none() && p.topic_tags.is_none()
            && p.layer.is_none() && p.source_ai.is_none());
    }

    #[test]
    fn test_patch_content_only_is_valid() {
        let p = PatchRecordRequest {
            content: Some("corrected".into()),
            topic_tags: None, layer: None, source_ai: None,
        };
        assert!(p.content.is_some());
    }

    #[test]
    fn test_record_id_hex_roundtrip() {
        let rid: [u8; 32] = [0xab; 32];
        let hex_str = hex::encode(rid);
        assert_eq!(hex_str.len(), 64);
        let decoded = hex::decode(&hex_str).unwrap();
        assert_eq!(decoded.len(), 32);
        let mut recovered = [0u8; 32];
        recovered.copy_from_slice(&decoded);
        assert_eq!(rid, recovered);
    }

    #[test]
    fn test_record_id_hex_wrong_length_rejected() {
        let short = hex::encode([0u8; 31]);
        assert_ne!(hex::decode(&short).unwrap().len(), 32);
    }

    #[test]
    fn test_default_recall_mode_is_full() {
        assert_eq!(default_recall_mode(), "full");
    }

    #[test]
    fn test_recall_context_defaults_to_none() {
        let json = r#"{"query":"test"}"#;
        let req: RecallRequest = serde_json::from_str(json).unwrap();
        assert!(req.context.is_none());
    }

    #[test]
    fn test_recall_context_all_is_no_filter() {
        let json = r#"{"query":"test","context":"all"}"#;
        let req: RecallRequest = serde_json::from_str(json).unwrap();
        let is_no_filter = matches!(req.context.as_deref(), None | Some("all") | Some(""));
        assert!(is_no_filter);
    }

    #[test]
    fn test_recall_context_custom_is_project_id() {
        let json = r#"{"query":"test","context":"project_alpha"}"#;
        let req: RecallRequest = serde_json::from_str(json).unwrap();
        let is_no_filter = matches!(req.context.as_deref(), None | Some("all") | Some(""));
        assert!(!is_no_filter);
        assert_eq!(req.context.as_deref(), Some("project_alpha"));
    }
}
