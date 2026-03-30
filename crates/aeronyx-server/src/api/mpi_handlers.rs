// ============================================
// File: crates/aeronyx-server/src/api/mpi_handlers.rs
// ============================================
//! # MPI Handlers — Core Endpoints
//!
//! ## File Creation/Modification Notes
//! ============================================
//! Creation Reason: v2.4.0 Split — extracted from mpi.rs to reduce file size.
//! Modification Reason: v2.5.2+Provenance — merged PATCH /record/:id and
//!   GET /record/:id/provenance handlers (previously specified as inline code
//!   comments in file 2). Added RecordProvenance struct and PatchRecordRequest.
//!   No new files created; all additions are in this existing file.
//! Main Functionality:
//!   - POST /api/mpi/remember    — store a new memory record
//!   - POST /api/mpi/forget      — soft-revoke a memory record
//!   - GET  /api/mpi/status      — system health + SuperNode queue metrics
//!   - GET  /api/mpi/record/:id  — fetch a single record by ID
//!   - GET  /api/mpi/records/overview — layer-grouped record summary
//!   - POST /api/mpi/embed       — local MiniLM batch embed
//!   - PATCH /api/mpi/record/:id            — 🆕 v2.5.2 partial record update
//!   - GET  /api/mpi/record/:id/provenance  — 🆕 v2.5.2 traceability chain
//! Dependencies:
//!   - aeronyx_core::ledger::{MemoryLayer, MemoryRecord}
//!   - crate::services::memchain::{mvf, LlmRouter, StorageStats}
//!   - super::mpi::{MpiState, AuthenticatedOwner, extract_owner, parse_layer, …}
//!   - storage.rs: insert, get, revoke, fts_index_record, fts_remove_record,
//!     get_overview, stats, get_recent_feedback, count_tasks_by_status,
//!     get_usage_stats, get_embedding_model, update_record_content,
//!     get_record_provenance
//!
//! ## Split Structure
//! - `mpi.rs`                  — MpiState, auth middleware, router, helpers
//! - `mpi_handlers.rs`         (THIS FILE) — remember, forget, status, embed,
//!                               record, overview, patch, provenance
//! - `recall_handler.rs`       — recall hybrid pipeline
//! - `mpi_graph_handlers.rs`   — v2.4.0 cognitive graph endpoints
//! - `supernode_handlers.rs`   — v2.5.0 SuperNode management endpoints
//!
//! ⚠️ Important Note for Next Developer:
//! - All handlers extract AuthenticatedOwner from extensions BEFORE calling
//!   req.into_body(). Never reorder: into_body() moves req, making extensions
//!   inaccessible (use-after-move compile error).
//! - update_record_content enforces ownership in the storage layer.
//!   Never skip the owner parameter — it prevents cross-user mutation.
//! - PATCH clears the embedding on content change (set NULL in DB).
//!   The Miner detects NULL embeddings and re-embeds on its next cycle (~60s).
//!   vector_index.remove() is called immediately to evict the stale ANN entry.
//! - RecordProvenance.turn_index is best-effort (content-length approximation).
//!   Schema v7 with SHA256 content_hash column would make it exact.
//! - Records inserted before v2.5.2 or via /remember have session_id = NULL.
//!   get_record_provenance returns Some(prov) with session_id = None (not 404).
//! - The core logic of this file cannot be deleted or significantly modified.
//! - Maintain interface compatibility with mpi.rs router registrations.
//!
//! ## Modification History
//! v2.4.0-GraphCognition  - Extracted from mpi.rs; status extended with NER/graph
//! v2.4.0+BM25            - Moved mpi_recall to recall_handler.rs; FTS indexing
//! v2.4.0+Progressive     - Added mode field to RecallRequest + default_recall_mode()
//! v2.5.0+SuperNode Phase D - MpiStatusResponse + SuperNodeStatus struct;
//!   mpi_status handler fills SuperNode queue counts + cost + provider info.
//! v2.5.2+Provenance      - 🌟 Merged PATCH + provenance handlers from spec;
//!   added RecordProvenance struct, PatchRecordRequest, mpi_patch_record,
//!   mpi_record_provenance, and unit tests. Zero new files created.
//!
//! ## Last Modified
//! v2.5.2+Provenance   - 🌟 PATCH /record/:id + GET /record/:id/provenance
//! v2.5.2+SecurityFix  - 🔒 #4  mpi_forget: revoke() SQL 加 AND owner=? 消除 TOCTOU
//!                       🔒 #9  mpi_patch_record: tags 变化时也触发 FTS 重索引
//!                       🔒 #11 identity_cache 加 MAX_IDENTITY_CACHE_PER_OWNER 上限

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::extract::Path;
use axum::http::Request;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::mvf;
use crate::services::memchain::LlmRouter;

use super::mpi::{
    MpiState, AuthenticatedOwner, BaselineSnapshot,
    extract_owner, parse_layer, estimate_tokens, now_secs,
    default_layer, default_source, default_model, default_top_k,
    default_token_budget, default_include_graph,
};

// ── Security constants ──────────────────────────────────────────────────────
/// Max identity/allergy records kept in the hot cache per owner (fix #11).
/// Prevents unbounded memory growth during long-running sessions.
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

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))),
    };
    let req_body: RememberRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))),
    };

    if req_body.content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"content empty"})));
    }
    let layer = match parse_layer(&req_body.layer) {
        Some(l) => l,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid layer"}))),
    };

    let ts = now_secs();

    if !req_body.embedding.is_empty() {
        let dedup = state.vector_index.check_duplicate(
            &req_body.embedding, &owner, &req_body.embedding_model, layer, ts,
        );
        if dedup.is_duplicate {
            let dup_hex = hex::encode(dedup.existing_id.unwrap_or([0; 32]));
            return (StatusCode::OK, Json(serde_json::json!(RememberResponse {
                record_id: dup_hex.clone(), status: "duplicate".into(), duplicate_of: Some(dup_hex),
            })));
        }
    }

    let encrypted_content = req_body.content.as_bytes().to_vec();
    let mut record = MemoryRecord::new(
        owner, ts, layer, req_body.topic_tags.clone(), req_body.source_ai.clone(),
        encrypted_content, req_body.embedding.clone(),
    );
    record.signature = state.identity.sign(&record.record_id);
    let rid_hex = record.id_hex();

    if !state.storage.insert(&record, &req_body.embedding_model).await {
        return (StatusCode::CONFLICT, Json(serde_json::json!({"error":"exists","record_id":rid_hex})));
    }

    if !req_body.embedding.is_empty() {
        state.vector_index.upsert(
            record.record_id, req_body.embedding, layer, ts, &owner, &req_body.embedding_model,
        );
    }

    let tags_str = serde_json::to_string(&req_body.topic_tags).unwrap_or_default();
    state.storage.fts_index_record(
        &record.record_id, &owner, &req_body.content, &tags_str,
    ).await;

    // Fix #11: cap identity cache size per owner to avoid unbounded growth.
    if layer == MemoryLayer::Identity {
        let mut cache = state.identity_cache.write();
        let entries = cache.entry(owner_hex.clone()).or_default();
        entries.push(record.clone());
        if entries.len() > MAX_IDENTITY_CACHE_PER_OWNER {
            entries.remove(0); // evict oldest
        }
    }

    info!(id = %rid_hex, layer = %layer, owner = %&owner_hex[..8], "[MPI_REMEMBER] Stored");
    (StatusCode::CREATED, Json(serde_json::json!(RememberResponse {
        record_id: rid_hex, status: "created".into(), duplicate_of: None,
    })))
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
    /// v2.4.0+Progressive: "full" (default) | "index"
    #[serde(default = "default_recall_mode")]
    pub mode: String,
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

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))),
    };
    let rb: ForgetRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))),
    };

    let rid = match hex::decode(&rb.record_id) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8;32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"bad record_id"}))),
    };

    if let Some(record) = state.storage.get(&rid).await {
        if record.owner != owner {
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error":"access denied"})));
        }
    } else {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: rb.record_id
        })));
    }

    // Fix #4: pass owner to revoke() so the SQL includes AND owner = ?,
    // eliminating the TOCTOU window between get() and revoke().
    if !state.storage.revoke_owned(&rid, &owner).await {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: rb.record_id
        })));
    }

    state.vector_index.remove(&rid);
    state.storage.fts_remove_record(&rid).await;

    { let oh = auth.owner_hex(); let mut c = state.identity_cache.write();
      if let Some(e) = c.get_mut(&oh) { e.retain(|r| r.record_id != rid); } }

    (StatusCode::OK, Json(serde_json::json!(ForgetResponse {
        status:"revoked".into(), record_id: rb.record_id
    })))
}

// ============================================
// GET /api/mpi/status (v2.5.0 extended)
// ============================================

/// SuperNode status embedded in /status response (v2.5.0+SuperNode).
#[derive(Debug, Clone, Serialize)]
pub struct SuperNodeStatus {
    pub enabled: bool,
    /// Names of configured providers.
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
    // v2.4.0
    pub ner_ready: bool,
    pub graph_enabled: bool,
    pub graph_stats: Option<crate::services::memchain::storage_graph::GraphStats>,
    // v2.5.0+SuperNode
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

    let stats = state.storage.stats().await;
    let height = state.storage.last_block_height().await;
    let wv = { state.user_weights.read().get(&owner_hex).map(|w| w.version).unwrap_or(0) };
    let fb = state.storage.get_recent_feedback(500).await;
    let tp = fb.iter().filter(|(s,_)| *s == 1).count() as u64;
    let tn = fb.iter().filter(|(s,_)| *s == -1).count() as u64;
    let mr = if !fb.is_empty() { Some(tp as f32 / fb.len() as f32) } else { None };
    let ms = if !fb.is_empty() { Some(fb.len()) } else { None };
    let bl = state.mvf_baseline.read().clone();
    let lift = match (&bl, mr) {
        (Some(b), Some(m)) if b.positive_rate > 0.0 => Some((m - b.positive_rate) / b.positive_rate),
        _ => None,
    };

    let gs = if state.graph_enabled || state.ner_engine.is_some() {
        Some(state.storage.graph_stats(&owner).await)
    } else { None };

    // ── v2.5.0+SuperNode: queue counts + today usage ──
    let supernode_status = {
        let now = now_secs() as i64;
        let today_start = now - (now % 86400);

        let counts = state.storage.count_tasks_by_status().await;
        let queue = QueueStatus {
            pending:    *counts.get("pending").unwrap_or(&0),
            processing: *counts.get("processing").unwrap_or(&0),
            failed:     *counts.get("failed").unwrap_or(&0),
        };

        let today_stats = state.storage.get_usage_stats(today_start, now).await;
        let cost_today: f64 = today_stats.by_provider.iter().map(|p| {
            LlmRouter::estimate_cost(
                &p.provider, p.input_tokens as u32, p.output_tokens as u32, 0,
            )
        }).sum();

        // completed tasks today
        let tasks_today = {
            let conn = state.storage.conn_lock().await;
            conn.query_row(
                "SELECT COUNT(*) FROM cognitive_tasks WHERE status='completed' AND completed_at >= ?1",
                rusqlite::params![today_start],
                |r| r.get::<_, i64>(0),
            ).unwrap_or(0)
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

    (StatusCode::OK, Json(serde_json::json!(MpiStatusResponse {
        memchain_enabled: true,
        mode: "local".into(),
        stats,
        vector_index_total: state.vector_index.total_vectors(),
        vector_partitions: state.vector_index.partition_count(),
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
    State(state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let rid = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid record_id"}))).into_response(),
    };
    let record = match state.storage.get(&rid).await {
        Some(r) => r,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error":"record not found"}))).into_response(),
    };
    if record.owner != owner {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error":"access denied"}))).into_response();
    }

    let em = state.storage.get_embedding_model(&rid).await.unwrap_or_default();
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
    let ov = state.storage.get_overview(&owner, 20).await;
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
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))),
    };
    let rb: EmbedRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))),
    };
    let engine = match &state.embed_engine {
        Some(e) => e,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error":"local embed engine not available"}))),
    };
    if rb.texts.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"texts array is empty"})));
    }
    if rb.texts.len() > 100 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"batch too large","max":100})));
    }

    let refs: Vec<&str> = rb.texts.iter().map(|s| s.as_str()).collect();
    match engine.embed_batch(&refs) {
        Ok(embs) => {
            let dim = embs.first().map(|v| v.len()).unwrap_or(0);
            debug!(batch = embs.len(), dim = dim, "[MPI_EMBED] Generated");
            (StatusCode::OK, Json(serde_json::json!({
                "embeddings": embs, "model": "minilm-l6-v2", "dim": dim
            })))
        }
        Err(e) => {
            warn!(error = %e, "[MPI_EMBED] Inference failed");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("embed failed: {}", e)
            })))
        }
    }
}

// ============================================
// v2.5.2+Provenance: RecordProvenance type
// ============================================

/// Full provenance chain for a memory record.
///
/// Answers: "Where did this memory come from?"
/// Returned by `GET /api/mpi/record/:id/provenance`.
///
/// ## Availability
/// - Records inserted before v2.5.2: `session_id = None`
/// - Records inserted via `/remember` directly: `session_id = None`
/// - Records extracted by Rule Engine via `/log` after v2.5.2: full data
///
/// ## turn_index note
/// Best-effort via content-length approximation against raw_logs.
/// Schema v7 (SHA256 content_hash column on records table) would make it exact.
///
/// v2.5.2+Provenance
#[derive(Debug, Clone, Serialize)]
pub struct RecordProvenance {
    /// The record being traced.
    pub record_id: String,
    /// Session (conversation) this record was extracted from.
    /// None for pre-v2.5.2 records or /remember inserts.
    pub session_id: Option<String>,
    /// Human-readable session title (SuperNode-generated). May be None.
    pub session_title: Option<String>,
    /// Unix timestamp of when the session started.
    pub session_started_at: Option<i64>,
    /// Best-effort turn index within the session.
    /// None if session_id unknown or no matching turn found.
    pub turn_index: Option<i64>,
    /// Memory layer at extraction time.
    pub layer: String,
    /// Semantic tags assigned by Rule Engine.
    pub topic_tags: Vec<String>,
    /// Unix timestamp when the record was created.
    pub extracted_at: u64,
    /// AI agent that triggered extraction.
    pub source_ai: String,
}

// ============================================
// PATCH /api/mpi/record/:record_id
// ============================================

/// PATCH request body — all fields optional (partial update).
///
/// Only provided fields are updated; omitted fields keep their current value.
/// Providing `content` triggers embedding invalidation: the DB embedding is
/// set to NULL so the Miner re-embeds on its next cycle (~60s). During that
/// window the record is available via FTS/BM25 but not vector search.
///
/// v2.5.2+Provenance
#[derive(Debug, Deserialize)]
pub struct PatchRecordRequest {
    /// New memory content. Clears the stored embedding (async re-embed by Miner).
    pub content: Option<String>,
    /// New semantic tags. Replaces existing tags entirely (not merged).
    pub topic_tags: Option<Vec<String>>,
    /// New memory layer. Use "identity" | "knowledge" | "episode" | "archive".
    pub layer: Option<String>,
    /// New source_ai label. Rarely needed; mostly for data correction.
    pub source_ai: Option<String>,
}

/// `PATCH /api/mpi/record/:record_id` — Partial in-place update of a memory record.
///
/// ## Use cases
/// 1. User correction: "That's wrong, it should be RS256 not HS256"
///    → PATCH content with the corrected fact
/// 2. Re-classification: "Move this to knowledge layer"
///    → PATCH layer
/// 3. Tag update: "Add 'security' tag to this memory"
///    → PATCH topic_tags
///
/// ## Embedding behavior
/// Content change → embedding cleared (NULL) → Miner re-embeds on next cycle.
/// vector_index.remove() is called immediately to evict the stale ANN entry.
///
/// ## FTS update
/// FTS5 index updated synchronously. Search reflects new content immediately.
///
/// ## Access control
/// Ownership enforced inside storage.update_record_content via the owner param.
/// Non-owners receive 404 (not 403) to avoid record enumeration.
///
/// v2.5.2+Provenance
pub async fn mpi_patch_record(
    State(state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    // Extract auth BEFORE into_body() — into_body() moves req (use-after-move guard).
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let rid: [u8; 32] = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "invalid record_id format"
        }))).into_response(),
    };

    let body_bytes = match axum::body::to_bytes(req.into_body(), 512 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "failed to read body"
        }))).into_response(),
    };
    let patch: PatchRecordRequest = match serde_json::from_slice(&body_bytes) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("invalid JSON: {}", e)
        }))).into_response(),
    };

    // At least one field must be set.
    if patch.content.is_none()
        && patch.topic_tags.is_none()
        && patch.layer.is_none()
        && patch.source_ai.is_none()
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

    match state.storage.update_record_content(
        &rid,
        &owner,
        patch.content.as_deref(),
        patch.topic_tags.as_deref(),
        new_layer,
        patch.source_ai.as_deref(),
    ).await {
        Ok(true) => {
            let content_changed = patch.content.is_some();
            // Fix #9: FTS must be re-indexed whenever content OR tags change,
            // because the FTS5 `tags` column would otherwise stay stale when
            // only topic_tags is patched (content unchanged).
            let needs_fts_update = patch.content.is_some() || patch.topic_tags.is_some();

            if needs_fts_update {
                // Use the new content if provided, otherwise re-fetch current content.
                // For tags-only updates we need the existing content to re-index.
                let index_content: Option<String> = if let Some(ref c) = patch.content {
                    Some(c.clone())
                } else {
                    // tags-only patch: re-read stored content for FTS rebuild
                    state.storage.get(&rid).await
                        .map(|r| String::from_utf8_lossy(&r.encrypted_content).into_owned())
                };

                if let Some(ref content_str) = index_content {
                    // Resolve the final tags: prefer patch value, else keep existing
                    // (get() already called above if needed, reuse the record).
                    let tags_str = patch.topic_tags.as_ref()
                        .and_then(|t| serde_json::to_string(t).ok())
                        .unwrap_or_default();
                    state.storage.fts_remove_record(&rid).await;
                    state.storage.fts_index_record(&rid, &owner, content_str, &tags_str).await;
                }
            }

            // Evict stale ANN entry immediately; Miner re-inserts after re-embed.
            if content_changed {
                state.vector_index.remove(&rid);
            }

            // Identity cache invalidation — Miner repopulates on next cycle.
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
                "fts_updated": content_changed,
                "note": if content_changed {
                    "Embedding cleared. Miner will re-embed on next cycle (~60s)."
                } else {
                    "Update applied."
                }
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

/// `GET /api/mpi/record/:record_id/provenance` — Memory provenance chain.
///
/// Returns the full traceability chain for a memory record:
/// - Which session (conversation) it was extracted from
/// - The session title and timestamp
/// - Best-effort turn index within the conversation
///
/// ## Response
/// `200 OK`  → `RecordProvenance` JSON (session_id may be null for older records)
/// `404`     → record not found or owner mismatch (no record enumeration)
///
/// v2.5.2+Provenance
pub async fn mpi_record_provenance(
    State(state): State<Arc<MpiState>>,
    Path(record_id_hex): Path<String>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    // Extract auth BEFORE into_body() — into_body() moves req (use-after-move guard).
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let rid: [u8; 32] = match hex::decode(&record_id_hex) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "invalid record_id format"
        }))).into_response(),
    };

    match state.storage.get_record_provenance(&rid, &owner).await {
        Some(prov) => (StatusCode::OK, Json(serde_json::json!(prov))).into_response(),
        None => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "record not found or access denied"
        }))).into_response(),
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── RecordProvenance serialization ──

    #[test]
    fn test_provenance_null_fields_serialize_as_null() {
        // Older records (pre-v2.5.2) have no session — fields must be null, not omitted.
        let prov = RecordProvenance {
            record_id: "aabbcc".into(),
            session_id: None,
            session_title: None,
            session_started_at: None,
            turn_index: None,
            layer: "episode".into(),
            topic_tags: vec!["identity".into()],
            extracted_at: 1_700_000_000,
            source_ai: "claude".into(),
        };
        let json = serde_json::to_value(&prov).unwrap();
        assert!(json["session_id"].is_null(), "session_id must be null for pre-v2.5.2 records");
        assert!(json["turn_index"].is_null(), "turn_index must be null when unavailable");
        assert_eq!(json["layer"], "episode");
    }

    #[test]
    fn test_provenance_full_fields_serialize_correctly() {
        let prov = RecordProvenance {
            record_id: "deadbeef".into(),
            session_id: Some("sess-abc".into()),
            session_title: Some("Debugging session".into()),
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
        assert_eq!(json["source_ai"], "gpt-4o");
    }

    // ── PatchRecordRequest validation shape ──

    #[test]
    fn test_patch_request_all_none_should_be_rejected() {
        let patch = PatchRecordRequest {
            content: None, topic_tags: None, layer: None, source_ai: None,
        };
        let all_none = patch.content.is_none()
            && patch.topic_tags.is_none()
            && patch.layer.is_none()
            && patch.source_ai.is_none();
        assert!(all_none, "All-None patch must be rejected by the handler guard");
    }

    #[test]
    fn test_patch_request_content_only_is_valid() {
        let patch = PatchRecordRequest {
            content: Some("corrected content".into()),
            topic_tags: None, layer: None, source_ai: None,
        };
        let any_set = patch.content.is_some()
            || patch.topic_tags.is_some()
            || patch.layer.is_some()
            || patch.source_ai.is_some();
        assert!(any_set, "content-only patch must pass the validation guard");
    }

    #[test]
    fn test_patch_request_tags_only_is_valid() {
        let patch = PatchRecordRequest {
            content: None,
            topic_tags: Some(vec!["security".into(), "identity".into()]),
            layer: None, source_ai: None,
        };
        assert!(patch.topic_tags.is_some());
    }

    // ── record_id hex decode shape ──

    #[test]
    fn test_record_id_hex_roundtrip_32_bytes() {
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
        // 31 bytes → must fail the `b.len() == 32` guard.
        let short = hex::encode([0u8; 31]);
        let decoded = hex::decode(&short).unwrap();
        assert_ne!(decoded.len(), 32, "31-byte decode must not pass the length guard");
    }

    // ── default_recall_mode ──

    #[test]
    fn test_default_recall_mode_is_full() {
        assert_eq!(default_recall_mode(), "full");
    }
}
