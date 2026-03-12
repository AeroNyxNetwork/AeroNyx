// ============================================
// File: crates/aeronyx-server/src/api/mpi_handlers.rs
// ============================================
//! # MPI Handlers — Original Endpoints (remember, recall, forget, status, embed, record, overview)
//!
//! ## Creation Reason (v2.4.0 Split)
//! Extracted from mpi.rs to reduce file size. Contains the original 7 endpoint
//! handlers that existed since v2.0-v2.3. The /log handler remains in log_handler.rs.
//!
//! ## Split Structure
//! - `mpi.rs` — MpiState, AuthenticatedOwner, auth middleware, router, helpers
//! - `mpi_handlers.rs` (THIS FILE) — remember, recall, forget, status, embed, record, overview
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints (11 new)
//!
//! All handlers use `super::mpi::` to import shared types and helpers.
//!
//! ## v2.4.0 Changes in This File
//! - mpi_recall: Hybrid retrieval pipeline (query_analyzer → multi-source → rerank)
//! - mpi_recall: RecallRequest +project_id, +time_range, +include_graph
//! - mpi_recall: RecallResponse +query_type, +matched_entities
//! - mpi_status: +ner_ready, +graph_enabled, +graph_stats
//! - compute_features: +graph_traverse_weight (φ₉, 13th parameter)
//!
//! ⚠️ Important Note for Next Developer:
//! - All handlers extract owner from AuthenticatedOwner extension (never state.owner_key)
//! - Shared helpers (extract_owner, parse_layer, etc.) are in mpi.rs
//! - When adding new original-style endpoints, add them here, not in mpi_graph_handlers.rs
//!
//! ## Last Modified
//! v2.4.0-GraphCognition - 🌟 Extracted from mpi.rs; hybrid recall; status extended

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::extract::Path;
use axum::http::Request;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{MemoryStorage, VectorIndex, compute_recall_score, cosine_similarity};
use crate::services::memchain::mvf::{self, WeightVector};
use crate::services::memchain::graph;
use crate::services::memchain::query_analyzer::{self, QueryType};

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
        owner, ts, layer, req_body.topic_tags, req_body.source_ai,
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
// POST /api/mpi/recall (v2.4.0 Hybrid)
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
    // v2.4.0: Hybrid retrieval parameters
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub time_range: Option<TimeRangeParam>,
    #[serde(default = "default_include_graph")]
    pub include_graph: bool,
}

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

/// `POST /api/mpi/recall` — v2.4.0 Hybrid retrieval pipeline.
///
/// Pipeline:
/// 1. Query Analysis (GLiNER + regex + entity matching)
/// 2. Multi-source: vector + graph BFS
/// 3. MVF φ₀-φ₉ scoring with graph_traverse_weight
/// 4. Token budget trimming
pub async fn mpi_recall(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();
    let owner_hex = auth.owner_hex();

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let rb: RecallRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": format!("invalid JSON: {}", e)}))).into_response(),
    };

    let now = now_secs();
    let layer_filter = rb.layer.as_deref().and_then(parse_layer);
    let top_k = rb.top_k.min(100).max(1);

    // ── Step 1: Query Analysis (v2.4.0) ──
    let analysis = if !rb.query.is_empty() && (state.ner_engine.is_some() || state.graph_enabled) {
        let known = state.storage.get_entities_cached(&owner).await;
        Some(query_analyzer::analyze_query(&rb.query, state.ner_engine.as_deref(), &known, now as i64))
    } else { None };

    let query_type = analysis.as_ref().map(|a| a.query_type.clone()).unwrap_or(QueryType::Semantic);

    // ── Session centroid for φ₇ ──
    let session_centroid: Option<Vec<f32>> = if let Some(ref sid) = rb.session_id {
        if !rb.embedding.is_empty() {
            let mut map = state.session_embeddings.write();
            let buf = map.entry(sid.clone()).or_insert_with(|| VecDeque::with_capacity(5));
            if buf.len() >= 5 { buf.pop_front(); }
            buf.push_back(rb.embedding.clone());
            let dim = buf[0].len();
            let mut centroid = vec![0.0f32; dim];
            for emb in buf.iter() { for (i, &v) in emb.iter().enumerate() { if i < dim { centroid[i] += v; } } }
            let n = buf.len() as f32; for v in centroid.iter_mut() { *v /= n; }
            Some(centroid)
        } else { None }
    } else { None };

    let mut memories: Vec<RecalledMemory> = Vec::new();
    let mut total_tokens = 0usize;
    let mut seen_ids: Vec<[u8; 32]> = Vec::new();

    // ── Identity forced injection ──
    {
        let cache = state.identity_cache.read();
        let id_recs = cache.get(&owner_hex).cloned().unwrap_or_default();
        drop(cache);
        for r in &id_recs {
            if !r.is_active() { continue; }
            let content = String::from_utf8_lossy(&r.encrypted_content).to_string();
            let tokens = estimate_tokens(&content);
            if total_tokens + tokens > rb.token_budget && !memories.is_empty() { break; }
            total_tokens += tokens; seen_ids.push(r.record_id);
            memories.push(RecalledMemory {
                record_id: r.id_hex(), layer: r.layer.to_string(),
                score: r.layer.recall_weight() + 1.0, content,
                topic_tags: r.topic_tags.clone(), source_ai: r.source_ai.clone(),
                timestamp: r.timestamp, access_count: r.access_count, proactive: false,
            });
            let st = Arc::clone(&state.storage); let rid = r.record_id;
            tokio::spawn(async move { st.increment_access(&rid).await; });
        }
    }

    // ── Step 2a: Vector search (always runs) ──
    let idx_ready = state.index_ready.load(std::sync::atomic::Ordering::Relaxed);
    let search = if !rb.embedding.is_empty() && idx_ready {
        state.vector_index.search_filtered(&rb.embedding, &owner, &rb.embedding_model, layer_filter, top_k * 3, 0.0)
    } else { Vec::new() };

    // ── Step 2b: Graph BFS (v2.4.0 — if entities matched + graph enabled) ──
    let graph_traversed: HashMap<String, f64> = if state.graph_enabled && rb.include_graph {
        if let Some(ref qa) = analysis {
            let matched_ids: Vec<String> = qa.matched_entities.iter()
                .filter_map(|e| e.entity_id.clone()).collect();
            if !matched_ids.is_empty() {
                let conn = state.storage.conn_lock().await;
                let nodes = graph::bfs_traverse(&conn, &owner, &matched_ids, 2, 20, 0.3);
                drop(conn);
                nodes.into_iter().map(|n| (n.entity_id, n.weight)).collect()
            } else { HashMap::new() }
        } else { HashMap::new() }
    } else { HashMap::new() };

    let total_candidates = search.len() + seen_ids.len();

    let max_degree = { let c = state.storage.conn_lock().await; graph::get_max_degree(&c, &owner) };
    let time_hint_tuple = rb.time_hint.as_ref().map(|th| (th.start, th.end));

    // ── Step 3: Score + Rerank ──
    let mut scored: Vec<(MemoryRecord, f64)> = Vec::new();

    for sr in &search {
        if seen_ids.contains(&sr.record_id) { continue; }
        if let Some(record) = state.storage.get(&sr.record_id).await {
            if !record.is_active() || record.owner != owner { continue; }

            let v_old = compute_recall_score(sr.similarity, record.timestamp, now, record.access_count, record.layer);
            let time_bonus: f64 = match &rb.time_hint {
                Some(th) if (record.timestamp as i64) >= th.start && (record.timestamp as i64) <= th.end => 0.15,
                _ => 0.0,
            };

            // v2.4.0: graph_traverse_weight from BFS results
            let graph_weight: f32 = if !graph_traversed.is_empty() {
                let tags_lower: HashSet<String> = record.topic_tags.iter().map(|t| t.to_lowercase()).collect();
                graph_traversed.iter()
                    .find(|(eid, _)| tags_lower.contains(&eid.to_lowercase()))
                    .map(|(_, w)| *w as f32)
                    .unwrap_or(0.0)
            } else { 0.0 };

            let v_old_h = v_old + time_bonus;

            let final_score = if state.mvf_enabled {
                let gd = { let c = state.storage.conn_lock().await; graph::get_degree(&c, &record.record_id) };
                let cs = session_centroid.as_ref()
                    .filter(|c| record.has_embedding() && c.len() == record.embedding.len())
                    .map(|c| cosine_similarity(c, &record.embedding))
                    .unwrap_or(0.0);

                let phi = mvf::compute_features(
                    sr.similarity, record.layer as u8, record.timestamp, now,
                    record.access_count, record.positive_feedback, record.negative_feedback,
                    record.has_conflict(), time_hint_tuple, cs, gd, max_degree,
                    graph_weight,
                );
                let mut uw = { state.user_weights.read().get(&owner_hex).cloned().unwrap_or_else(mvf::default_weights) };
                let pn = mvf::normalize(&phi, &mut uw);
                let vm = mvf::compute_value(&uw, &pn);
                { state.user_weights.write().insert(owner_hex.clone(), uw); }
                mvf::fuse_scores(vm, v_old_h, state.mvf_alpha)
            } else {
                v_old_h + (graph_weight as f64 * 0.1)
            };

            scored.push((record, final_score));
        }
    }

    if search.is_empty() {
        let recent = state.storage.get_active_records(&owner, layer_filter, top_k).await;
        for r in recent {
            if seen_ids.contains(&r.record_id) { continue; }
            scored.push((r.clone(), compute_recall_score(1.0, r.timestamp, now, r.access_count, r.layer)));
        }
    }

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 4: Token budget ──
    let mut returned_ids = seen_ids.clone();
    for (r, score) in &scored {
        let content = String::from_utf8_lossy(&r.encrypted_content).to_string();
        let tokens = estimate_tokens(&content);
        if total_tokens + tokens > rb.token_budget && !memories.is_empty() { break; }
        let proactive = r.layer == MemoryLayer::Identity
            && search.iter().any(|sr| sr.record_id == r.record_id && sr.similarity > 0.3);
        total_tokens += tokens; returned_ids.push(r.record_id);
        memories.push(RecalledMemory {
            record_id: r.id_hex(), layer: r.layer.to_string(), score: *score, content,
            topic_tags: r.topic_tags.clone(), source_ai: r.source_ai.clone(),
            timestamp: r.timestamp, access_count: r.access_count, proactive,
        });
        let st = Arc::clone(&state.storage); let rid = r.record_id;
        tokio::spawn(async move { st.increment_access(&rid).await; });
    }
    memories.truncate(top_k);

    // Async graph update
    { let st = Arc::clone(&state.storage); let ids = returned_ids;
      tokio::spawn(async move { let c = st.conn_lock().await;
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        graph::update_cooccurrence(&c, &ids, n); }); }

    let matched_json = analysis.as_ref().map(|qa| {
        qa.matched_entities.iter().map(|e| serde_json::json!({
            "text": e.query_text, "label": e.label, "confidence": e.confidence,
            "entity_id": e.entity_id, "entity_type": e.entity_type,
        })).collect::<Vec<_>>()
    });

    (StatusCode::OK, Json(serde_json::json!(RecallResponse {
        memories, total_candidates, token_estimate: total_tokens,
        query_type: Some(format!("{:?}", query_type).to_lowercase()),
        matched_entities: matched_json,
    }))).into_response()
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
