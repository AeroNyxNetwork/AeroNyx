// ============================================
// File: crates/aeronyx-server/src/api/mpi.rs
// ============================================
//! # MPI — Memory Protocol Interface (v2.1 + MVF)
//!
//! Endpoints: remember, recall, forget, status, log
//! recall hot path target: < 50ms
//!
//! ## v2.1.0+MVF Changes
//! - recall: compute_features() now reads record.positive_feedback,
//!   record.negative_feedback, and record.has_conflict() from the
//!   MemoryRecord struct instead of using hardcoded zeros.
//!   This enables MVF φ₄ (feedback score) and φ₈ (conflict penalty)
//!   to use real data from Schema v4 columns.
//!
//! ## Last Modified
//! v2.1.0 - MPI endpoints with MVF fusion scoring
//! v2.1.0+MVF - 🌟 Fixed hardcoded feedback/conflict in compute_features();
//!   recall scoring now uses actual positive_feedback, negative_feedback,
//!   and conflict_with data from records table (Schema v4).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{MemoryStorage, VectorIndex, compute_recall_score};
use crate::services::memchain::mvf::{self, WeightVector, MVF_DIM};
use crate::services::memchain::graph;

// ============================================
// Shared State
// ============================================

pub struct MpiState {
    pub storage: Arc<MemoryStorage>,
    pub vector_index: Arc<VectorIndex>,
    pub identity: IdentityKeyPair,
    pub identity_cache: RwLock<HashMap<String, Vec<MemoryRecord>>>,
    pub index_ready: AtomicBool,
    pub user_weights: Arc<RwLock<HashMap<String, WeightVector>>>,
    pub mvf_alpha: f32,
    pub mvf_enabled: bool,
    pub session_embeddings: RwLock<HashMap<String, VecDeque<Vec<f32>>>>,
    pub mvf_baseline: RwLock<Option<BaselineSnapshot>>,
    pub owner_key: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub positive_rate: f32,
    pub sample_size: usize,
    pub frozen_at: i64,
}

// ============================================
// Helpers
// ============================================

fn parse_layer(s: &str) -> Option<MemoryLayer> {
    match s.to_lowercase().as_str() {
        "identity"  => Some(MemoryLayer::Identity),
        "knowledge" => Some(MemoryLayer::Knowledge),
        "episode"   => Some(MemoryLayer::Episode),
        "archive"   => Some(MemoryLayer::Archive),
        _ => None,
    }
}

fn estimate_tokens(text: &str) -> usize {
    let len = text.len();
    if len == 0 { return 0; }
    let sample_len = len.min(100);
    let ascii_count = text.as_bytes()[..sample_len].iter().filter(|b| b.is_ascii()).count();
    if (ascii_count as f64 / sample_len as f64) > 0.80 {
        (len + 3) / 4
    } else {
        (len * 2 / 3).max(1)
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

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

fn default_layer() -> String { "episode".into() }
fn default_source() -> String { "unknown".into() }
fn default_model() -> String { "default".into() }

#[derive(Debug, Serialize)]
pub struct RememberResponse {
    pub record_id: String,
    pub status: String,
    pub duplicate_of: Option<String>,
}

pub async fn mpi_remember(
    State(state): State<Arc<MpiState>>,
    Json(req): Json<RememberRequest>,
) -> impl IntoResponse {
    if req.content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"content empty"})));
    }
    let layer = match parse_layer(&req.layer) {
        Some(l) => l,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"invalid layer"}))),
    };

    let owner = state.owner_key;
    let ts = now_secs();

    // Dedup check
    if !req.embedding.is_empty() {
        let dedup = state.vector_index.check_duplicate(
            &req.embedding, &owner, &req.embedding_model, layer, ts,
        );
        if dedup.is_duplicate {
            let dup_hex = hex::encode(dedup.existing_id.unwrap_or([0; 32]));
            return (StatusCode::OK, Json(serde_json::json!(RememberResponse {
                record_id: dup_hex.clone(), status: "duplicate".into(), duplicate_of: Some(dup_hex),
            })));
        }
    }

    let encrypted_content = req.content.as_bytes().to_vec();
    let mut record = MemoryRecord::new(
        owner, ts, layer, req.topic_tags, req.source_ai,
        encrypted_content, req.embedding.clone(),
    );
    record.signature = state.identity.sign(&record.record_id);
    let rid_hex = record.id_hex();

    // SQLite first (source of truth)
    if !state.storage.insert(&record, &req.embedding_model).await {
        return (StatusCode::CONFLICT, Json(serde_json::json!({"error":"exists","record_id":rid_hex})));
    }

    // Vector index (after SQLite success)
    if !req.embedding.is_empty() {
        state.vector_index.upsert(
            record.record_id, req.embedding, layer, ts, &owner, &req.embedding_model,
        );
    }

    // Identity cache (after SQLite success)
    if layer == MemoryLayer::Identity {
        let mut cache = state.identity_cache.write();
        cache.entry(hex::encode(owner)).or_default().push(record.clone());
    }

    info!(id = %rid_hex, layer = %layer, "[MPI_REMEMBER] Stored");
    (StatusCode::CREATED, Json(serde_json::json!(RememberResponse {
        record_id: rid_hex, status: "created".into(), duplicate_of: None,
    })))
}

// ============================================
// POST /api/mpi/recall
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
}

#[derive(Debug, Clone, Deserialize)]
pub struct TimeHint { pub start: i64, pub end: i64 }

fn default_top_k() -> usize { 10 }
fn default_token_budget() -> usize { 4000 }

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
}

pub async fn mpi_recall(
    State(state): State<Arc<MpiState>>,
    Json(req): Json<RecallRequest>,
) -> impl IntoResponse {
    let owner = state.owner_key;
    let owner_hex = hex::encode(owner);
    let now = now_secs();
    let layer_filter = req.layer.as_deref().and_then(parse_layer);
    let top_k = req.top_k.min(100).max(1);

    // Session centroid for phi_7
    let session_centroid: Option<Vec<f32>> = if let Some(ref sid) = req.session_id {
        if !req.embedding.is_empty() {
            let mut map = state.session_embeddings.write();
            let buf = map.entry(sid.clone()).or_insert_with(|| VecDeque::with_capacity(5));
            if buf.len() >= 5 { buf.pop_front(); }
            buf.push_back(req.embedding.clone());
            let dim = buf[0].len();
            let mut centroid = vec![0.0f32; dim];
            for emb in buf.iter() {
                for (i, &v) in emb.iter().enumerate() { if i < dim { centroid[i] += v; } }
            }
            let n = buf.len() as f32;
            for v in centroid.iter_mut() { *v /= n; }
            Some(centroid)
        } else { None }
    } else { None };

    let mut memories: Vec<RecalledMemory> = Vec::new();
    let mut total_tokens = 0usize;
    let mut seen_ids: Vec<[u8; 32]> = Vec::new();

    // Step 1: Identity forced injection
    {
        let cache = state.identity_cache.read();
        let id_recs = cache.get(&owner_hex).cloned().unwrap_or_default();
        drop(cache);

        for r in &id_recs {
            if !r.is_active() { continue; }
            let content = String::from_utf8_lossy(&r.encrypted_content).to_string();
            let tokens = estimate_tokens(&content);
            if total_tokens + tokens > req.token_budget && !memories.is_empty() { break; }
            total_tokens += tokens;
            seen_ids.push(r.record_id);
            memories.push(RecalledMemory {
                record_id: r.id_hex(), layer: r.layer.to_string(),
                score: r.layer.recall_weight() + 1.0, content,
                topic_tags: r.topic_tags.clone(), source_ai: r.source_ai.clone(),
                timestamp: r.timestamp, access_count: r.access_count, proactive: false,
            });
            let st = Arc::clone(&state.storage);
            let rid = r.record_id;
            tokio::spawn(async move { st.increment_access(&rid).await; });
        }
    }

    // Step 2: Vector search
    let idx_ready = state.index_ready.load(std::sync::atomic::Ordering::Relaxed);
    let search = if !req.embedding.is_empty() && idx_ready {
        state.vector_index.search_filtered(
            &req.embedding, &owner, &req.embedding_model, layer_filter, top_k * 3, 0.0,
        )
    } else { Vec::new() };

    let total_candidates = search.len() + seen_ids.len();

    // Graph max degree
    let max_degree = {
        let conn = state.storage.conn_lock().await;
        graph::get_max_degree(&conn, &owner)
    };

    let time_hint_tuple = req.time_hint.as_ref().map(|th| (th.start, th.end));

    // Step 3+4: Load + score
    let mut scored: Vec<(MemoryRecord, f64)> = Vec::new();

    for sr in &search {
        if seen_ids.contains(&sr.record_id) { continue; }
        if let Some(record) = state.storage.get(&sr.record_id).await {
            if !record.is_active() { continue; }

            let v_old = compute_recall_score(
                sr.similarity, record.timestamp, now, record.access_count, record.layer,
            );
            let time_bonus: f64 = match &req.time_hint {
                Some(th) if (record.timestamp as i64) >= th.start && (record.timestamp as i64) <= th.end => 0.15,
                _ => 0.0,
            };
            let v_old_h = v_old + time_bonus;

            let final_score = if state.mvf_enabled {
                let gd = { let c = state.storage.conn_lock().await; graph::get_degree(&c, &record.record_id) };
                let cs = session_centroid.as_ref()
                    .filter(|c| record.has_embedding() && c.len() == record.embedding.len())
                    .map(|c| crate::services::memchain::cosine_similarity(c, &record.embedding))
                    .unwrap_or(0.0);

                // v2.1.0+MVF: Read actual feedback and conflict data from record
                // (previously hardcoded as 0, 0, false — rendering φ₄ and φ₈ useless)
                let phi = mvf::compute_features(
                    sr.similarity,
                    record.layer as u8,
                    record.timestamp,
                    now,
                    record.access_count,
                    record.positive_feedback,   // was: 0 (hardcoded)
                    record.negative_feedback,   // was: 0 (hardcoded)
                    record.has_conflict(),      // was: record.topic_tags.iter().any(|t| t == "_conflict")
                    time_hint_tuple,
                    cs,
                    gd,
                    max_degree,
                );
                let mut uw = { state.user_weights.read().get(&owner_hex).cloned().unwrap_or_else(mvf::default_weights) };
                let pn = mvf::normalize(&phi, &mut uw);
                let vm = mvf::compute_value(&uw, &pn);
                { state.user_weights.write().insert(owner_hex.clone(), uw); }
                mvf::fuse_scores(vm, v_old_h, state.mvf_alpha)
            } else { v_old_h };

            scored.push((record, final_score));
        }
    }

    // Fallback: no search results
    if search.is_empty() {
        let recent = state.storage.get_active_records(&owner, layer_filter, top_k).await;
        for r in recent {
            if seen_ids.contains(&r.record_id) { continue; }
            let s = compute_recall_score(1.0, r.timestamp, now, r.access_count, r.layer);
            scored.push((r, s));
        }
    }

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step 5: Token budget
    let mut returned_ids = seen_ids.clone();
    for (r, score) in &scored {
        let content = String::from_utf8_lossy(&r.encrypted_content).to_string();
        let tokens = estimate_tokens(&content);
        if total_tokens + tokens > req.token_budget && !memories.is_empty() { break; }
        let proactive = r.layer == MemoryLayer::Identity
            && search.iter().find(|sr| sr.record_id == r.record_id).map_or(false, |sr| sr.similarity > 0.3);
        total_tokens += tokens;
        returned_ids.push(r.record_id);
        memories.push(RecalledMemory {
            record_id: r.id_hex(), layer: r.layer.to_string(), score: *score, content,
            topic_tags: r.topic_tags.clone(), source_ai: r.source_ai.clone(),
            timestamp: r.timestamp, access_count: r.access_count, proactive,
        });
        let st = Arc::clone(&state.storage);
        let rid = r.record_id;
        tokio::spawn(async move { st.increment_access(&rid).await; });
    }
    memories.truncate(top_k);

    // Async graph update
    {
        let st = Arc::clone(&state.storage);
        let ids = returned_ids;
        tokio::spawn(async move {
            let c = st.conn_lock().await;
            let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
            graph::update_cooccurrence(&c, &ids, n);
        });
    }

    (StatusCode::OK, Json(serde_json::json!(RecallResponse {
        memories, total_candidates, token_estimate: total_tokens,
    })))
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
    Json(req): Json<ForgetRequest>,
) -> impl IntoResponse {
    let rid = match hex::decode(&req.record_id) {
        Ok(b) if b.len() == 32 => { let mut a = [0u8;32]; a.copy_from_slice(&b); a }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"bad record_id"}))),
    };

    if !state.storage.revoke(&rid).await {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status:"not_found".into(), record_id: req.record_id })));
    }

    state.vector_index.remove(&rid);
    { let oh = hex::encode(state.owner_key);
      let mut c = state.identity_cache.write();
      if let Some(e) = c.get_mut(&oh) { e.retain(|r| r.record_id != rid); } }

    (StatusCode::OK, Json(serde_json::json!(ForgetResponse {
        status:"revoked".into(), record_id: req.record_id })))
}

// ============================================
// GET /api/mpi/status
// ============================================

#[derive(Debug, Serialize)]
pub struct MpiStatusResponse {
    pub memchain_enabled: bool,
    pub mode: String,
    pub stats: crate::services::memchain::StorageStats,
    pub vector_index_total: usize,
    pub vector_partitions: usize,
    pub last_block_height: u64,
    pub index_ready: bool,
    pub mvf: MvfMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct MvfMetrics {
    pub enabled: bool,
    pub alpha: f32,
    pub total_positive_feedback: u64,
    pub total_negative_feedback: u64,
    pub baseline_positive_rate: Option<f32>,
    pub baseline_sample_size: Option<usize>,
    pub mvf_positive_rate: Option<f32>,
    pub mvf_sample_size: Option<usize>,
    pub lift: Option<f32>,
    pub weights_version: u64,
}

pub async fn mpi_status(
    State(state): State<Arc<MpiState>>,
) -> impl IntoResponse {
    let stats = state.storage.stats().await;
    let height = state.storage.last_block_height().await;
    let oh = hex::encode(state.owner_key);
    let wv = { state.user_weights.read().get(&oh).map(|w| w.version).unwrap_or(0) };
    let fb = state.storage.get_recent_feedback(500).await;
    let total_pos = fb.iter().filter(|(s,_)| *s == 1).count() as u64;
    let total_neg = fb.iter().filter(|(s,_)| *s == -1).count() as u64;
    let mvf_rate = if !fb.is_empty() { Some(total_pos as f32 / fb.len() as f32) } else { None };
    let mvf_sample = if !fb.is_empty() { Some(fb.len()) } else { None };
    let baseline = state.mvf_baseline.read().clone();
    let lift = match (&baseline, mvf_rate) {
        (Some(b), Some(m)) if b.positive_rate > 0.0 => Some((m - b.positive_rate) / b.positive_rate),
        _ => None,
    };

    (StatusCode::OK, Json(serde_json::json!(MpiStatusResponse {
        memchain_enabled: true, mode: "local".into(), stats,
        vector_index_total: state.vector_index.total_vectors(),
        vector_partitions: state.vector_index.partition_count(),
        last_block_height: height,
        index_ready: state.index_ready.load(std::sync::atomic::Ordering::Relaxed),
        mvf: MvfMetrics {
            enabled: state.mvf_enabled, alpha: state.mvf_alpha,
            total_positive_feedback: total_pos, total_negative_feedback: total_neg,
            baseline_positive_rate: baseline.as_ref().map(|b| b.positive_rate),
            baseline_sample_size: baseline.as_ref().map(|b| b.sample_size),
            mvf_positive_rate: mvf_rate, mvf_sample_size: mvf_sample,
            lift, weights_version: wv,
        },
    })))
}

// ============================================
// Router
// ============================================

pub fn build_mpi_router(state: Arc<MpiState>) -> Router {
    Router::new()
        .route("/api/mpi/remember", post(mpi_remember))
        .route("/api/mpi/recall", post(mpi_recall))
        .route("/api/mpi/forget", post(mpi_forget))
        .route("/api/mpi/status", get(mpi_status))
        .route("/api/mpi/log", post(crate::api::log_handler::mpi_log))
        .with_state(state)
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    async fn make_state() -> Arc<MpiState> {
        let storage = Arc::new(MemoryStorage::open(":memory:").unwrap());
        let vi = Arc::new(VectorIndex::new());
        let id = IdentityKeyPair::generate();
        let ok = id.public_key_bytes();
        Arc::new(MpiState {
            storage, vector_index: vi, identity: id,
            identity_cache: RwLock::new(HashMap::new()),
            index_ready: AtomicBool::new(true),
            user_weights: Arc::new(RwLock::new(HashMap::new())),
            mvf_alpha: 0.0, mvf_enabled: false,
            session_embeddings: RwLock::new(HashMap::new()),
            mvf_baseline: RwLock::new(None),
            owner_key: ok,
        })
    }

    #[tokio::test]
    async fn test_remember_and_recall() {
        let s = make_state().await;
        let app = build_mpi_router(s);

        let req = Request::builder().method("POST").uri("/api/mpi/remember")
            .header("content-type","application/json")
            .body(Body::from(r#"{"content":"User likes Rust","layer":"identity","embedding":[0.1,0.2,0.3],"embedding_model":"t"}"#)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let req = Request::builder().method("POST").uri("/api/mpi/recall")
            .header("content-type","application/json")
            .body(Body::from(r#"{"embedding":[0.1,0.2,0.3],"embedding_model":"t","top_k":5}"#)).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_status() {
        let s = make_state().await;
        let app = build_mpi_router(s);
        let req = Request::builder().uri("/api/mpi/status").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
