// ============================================
// File: crates/aeronyx-server/src/api/mpi_handlers.rs
// ============================================
//! # MPI Handlers — Core Endpoints (remember, forget, status, embed, record, overview)
//!
//! ## Creation Reason (v2.4.0 Split)
//! Extracted from mpi.rs to reduce file size. Contains the original core
//! endpoint handlers. The /log handler is in log_handler.rs.
//!
//! ## Split Structure
//! - `mpi.rs` — MpiState, AuthenticatedOwner, auth middleware, router, helpers
//! - `mpi_handlers.rs` (THIS FILE) — remember, forget, status, embed, record, overview
//! - `recall_handler.rs` — recall (hybrid: vector + BM25 + graph + RRF) ← v2.4.0+BM25
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints (11 new)
//!
//! ## v2.4.0+BM25 Changes
//! - mpi_recall MOVED to recall_handler.rs (pipeline too complex for shared file)
//! - Recall types (RecallRequest, RecallResponse, etc.) remain here for re-export
//! - mpi_remember: +fts_index_record after insert
//! - mpi_forget: +fts_remove_record after revoke
//! - mpi_status: +ner_ready, +graph_enabled, +graph_stats (from v2.4.0)
//!
//! ## v2.4.0+Progressive Changes
//! - RecallRequest: added `mode: String` field (default "full")
//!   "index" mode returns lightweight previews for progressive retrieval
//! - Added default_recall_mode() helper function
//!
//! ⚠️ Important Note for Next Developer:
//! - All handlers extract owner from AuthenticatedOwner extension
//! - RecallRequest/RecallResponse types are defined here but the handler
//!   is in recall_handler.rs (which re-exports these types)
//! - When adding new original-style endpoints, add them here
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - Extracted from mpi.rs; hybrid recall; status extended
//! v2.4.0+BM25 - 🌟 Moved mpi_recall to recall_handler.rs; added FTS indexing
//!   in remember/forget; recall types preserved for backward compat
//! v2.4.0+Progressive - 🌟 Added mode field to RecallRequest + default_recall_mode()

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::extract::Path;
use axum::http::Request;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::mvf;

use super::mpi::{
    MpiState, AuthenticatedOwner, BaselineSnapshot,
    extract_owner, parse_layer, estimate_tokens, now_secs,
    default_layer, default_source, default_model, default_top_k,
    default_token_budget, default_include_graph,
};

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

    // v2.4.0+BM25: Index in FTS5 for keyword search
    let tags_str = serde_json::to_string(&req_body.topic_tags).unwrap_or_default();
    state.storage.fts_index_record(
        &record.record_id, &owner, &req_body.content, &tags_str,
    ).await;

    if layer == MemoryLayer::Identity {
        let mut cache = state.identity_cache.write();
        cache.entry(owner_hex.clone()).or_default().push(record.clone());
    }

    info!(id = %rid_hex, layer = %layer, owner = %&owner_hex[..8], "[MPI_REMEMBER] Stored");
    (StatusCode::CREATED, Json(serde_json::json!(RememberResponse {
        record_id: rid_hex, status: "created".into(), duplicate_of: None,
    })))
}

// ============================================
// Recall Types (used by recall_handler.rs)
// ============================================
// These struct definitions remain here so that recall_handler.rs can
// `pub use super::mpi_handlers::{RecallRequest, ...}` for backward compat.

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
    // v2.4.0: Hybrid retrieval parameters
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub time_range: Option<TimeRangeParam>,
    #[serde(default = "default_include_graph")]
    pub include_graph: bool,
    /// v2.4.0+Progressive: Retrieval mode.
    /// - "full" (default): returns complete content for each memory (current behavior)
    /// - "index": returns lightweight index only (id, title, score, entities, ~50 token each)
    ///   AI reviews the index and calls /recall/detail for full content of selected items.
    #[serde(default = "default_recall_mode")]
    pub mode: String,
}

/// v2.4.0+Progressive: Default recall mode is "full" (existing behavior unchanged).
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
    // v2.4.0
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
            return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "access denied: record belongs to another owner"})));
        }
    } else {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse { status:"not_found".into(), record_id: rb.record_id })));
    }

    if !state.storage.revoke(&rid).await {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse { status:"not_found".into(), record_id: rb.record_id })));
    }

    state.vector_index.remove(&rid);

    // v2.4.0+BM25: Remove from FTS5 index
    state.storage.fts_remove_record(&rid).await;

    { let oh = auth.owner_hex(); let mut c = state.identity_cache.write();
      if let Some(e) = c.get_mut(&oh) { e.retain(|r| r.record_id != rid); } }

    (StatusCode::OK, Json(serde_json::json!(ForgetResponse { status:"revoked".into(), record_id: rb.record_id })))
}

// ============================================
// GET /api/mpi/status (v2.4.0 extended)
// ============================================

#[derive(Debug, Serialize)]
pub struct MpiStatusResponse {
    pub memchain_enabled: bool, pub mode: String,
    pub stats: crate::services::memchain::StorageStats,
    pub vector_index_total: usize, pub vector_partitions: usize,
    pub last_block_height: u64, pub index_ready: bool,
    pub embed_ready: bool, pub embed_dim: Option<usize>,
    pub mvf: MvfMetrics,
    pub remote_storage_enabled: bool,
    // v2.4.0
    pub ner_ready: bool,
    pub graph_enabled: bool,
    pub graph_stats: Option<crate::services::memchain::storage_ops::GraphStats>,
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
    let lift = match (&bl, mr) { (Some(b), Some(m)) if b.positive_rate > 0.0 => Some((m - b.positive_rate) / b.positive_rate), _ => None };

    let gs = if state.graph_enabled || state.ner_engine.is_some() {
        Some(state.storage.graph_stats(&owner).await)
    } else { None };

    (StatusCode::OK, Json(serde_json::json!(MpiStatusResponse {
        memchain_enabled: true, mode: "local".into(), stats,
        vector_index_total: state.vector_index.total_vectors(),
        vector_partitions: state.vector_index.partition_count(),
        last_block_height: height,
        index_ready: state.index_ready.load(std::sync::atomic::Ordering::Relaxed),
        embed_ready: state.embed_engine.is_some(),
        embed_dim: state.embed_engine.as_ref().map(|e| e.dim()),
        mvf: MvfMetrics { enabled: state.mvf_enabled, alpha: state.mvf_alpha,
            total_positive_feedback: tp, total_negative_feedback: tn,
            baseline_positive_rate: bl.as_ref().map(|b| b.positive_rate),
            baseline_sample_size: bl.as_ref().map(|b| b.sample_size),
            mvf_positive_rate: mr, mvf_sample_size: ms, lift, weights_version: wv },
        remote_storage_enabled: state.allow_remote_storage,
        ner_ready: state.ner_engine.is_some(),
        graph_enabled: state.graph_enabled,
        graph_stats: gs,
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
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "invalid record_id format"}))).into_response(),
    };
    let record = match state.storage.get(&rid).await {
        Some(r) => r,
        None => return (StatusCode::NOT_FOUND, Json(serde_json::json!({"error": "record not found"}))).into_response(),
    };
    if record.owner != owner {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({"error": "access denied"}))).into_response();
    }

    let em = state.storage.get_embedding_model(&rid).await.unwrap_or_default();
    let content = String::from_utf8_lossy(&record.encrypted_content).to_string();

    (StatusCode::OK, Json(serde_json::json!(RecordDetailResponse {
        record_id: record_id_hex, layer: record.layer.to_string(), content,
        topic_tags: record.topic_tags.clone(), source_ai: record.source_ai.clone(),
        timestamp: record.timestamp, access_count: record.access_count,
        positive_feedback: record.positive_feedback, negative_feedback: record.negative_feedback,
        has_conflict: record.has_conflict(), embedding_model: em, has_embedding: record.has_embedding(),
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
        "total": total, "by_layer": ov.by_layer, "recent_by_layer": ov.recent_by_layer,
        "last_memory_at": ov.last_memory_at,
        "embed_ready": state.embed_engine.is_some(),
        "embed_dim": state.embed_engine.as_ref().map(|e| e.dim()),
    })))
}

// ============================================
// POST /api/mpi/embed
// ============================================

#[derive(Debug, Deserialize)]
pub struct EmbedRequest { pub texts: Vec<String>, #[serde(default = "default_model")] pub model: String }

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
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"error": "local embed engine not available"}))),
    };
    if rb.texts.is_empty() { return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "texts array is empty"}))); }
    if rb.texts.len() > 100 { return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": "batch too large", "max": 100}))); }

    let refs: Vec<&str> = rb.texts.iter().map(|s| s.as_str()).collect();
    match engine.embed_batch(&refs) {
        Ok(embs) => {
            let dim = embs.first().map(|v| v.len()).unwrap_or(0);
            debug!(batch = embs.len(), dim = dim, "[MPI_EMBED] Generated");
            (StatusCode::OK, Json(serde_json::json!({"embeddings": embs, "model": "minilm-l6-v2", "dim": dim})))
        }
        Err(e) => {
            warn!(error = %e, "[MPI_EMBED] Inference failed");
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("embed failed: {}", e)})))
        }
    }
}
