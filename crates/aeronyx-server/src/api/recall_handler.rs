// ============================================
// File: crates/aeronyx-server/src/api/recall_handler.rs
// ============================================
//! # POST /api/mpi/recall — Sealed-Record Retrieval
//!
//! Ordinary nodes return opaque sealed records and keyed-term matches only.
//! Free-text search and semantic inference remain client-side / Phala ACI.
//!
//! ## Sealed Retrieval
//! Pass 1: POST /recall { query_terms, mode } → opaque ciphertext
//! Pass 2: POST /recall/detail { record_ids } → sealed ciphertext/envelopes
//!
//! ## v2.5.3+Isolation: Context filter (Step 4.1)
//! When `context` is set to a non-"all" value, `scored` is filtered to only
//! include records whose session is tagged with that project_id.
//! The filter is applied AFTER RRF/MVF scoring and reranking (post-filter)
//! because vector search doesn't carry project_id metadata.
//!
//! Context mapping:
//! - None / "all" / "" → no filter (all records returned)
//! - "work"            → project_id = "work"
//! - "personal"        → project_id = "personal"
//! - any other string  → treated as literal project_id
//!
//! Known limitation: post-filter can reduce results below top_k when many
//! records are in other contexts. Future: tag VectorIndex entries with
//! project_id for pre-filter.
//!
//! ⚠️ Important Note for Next Developer:
//! - matched_json is computed BEFORE the index-mode early-return (borrow-after-move).
//! - Owner/storage/vector context uses typed Axum extractors. Missing middleware
//!   context must remain a contained HTTP error under release `panic=abort`.
//! - Ordinary-node recall does not return legacy sighted records or accept
//!   plaintext query strings; clients decrypt and rank sealed data locally.
//! - Context filter (Step 4.1) calls get_active_records_by_context() which
//!   does a LEFT JOIN records → sessions. For records inserted via /remember
//!   directly (no session), project_id on the record row itself is checked.
//! - The core logic of this file cannot be deleted or significantly modified.
//! - v1.0.1-SaaSFix: storage and vector_index are extracted from request
//!   Extensions rather than state fields (which are None in SaaS mode).
//!
//! ## Modification History
//! v2.4.0+BM25          - 🌟 Extracted from mpi_handlers.rs; BM25 + RRF fusion
//! v2.4.0+BM25-fix      - 🔧 BM25 entity/session direct injection (Step 2a-ter)
//! v2.4.0+Reranker      - 🌟 Step 3.5 cross-encoder rerank
//! v2.4.0+Progressive   - 🌟 mode="index" branch + mpi_recall_detail handler
//! v2.5.3+Isolation     - 🌟 Step 4.1 context filter; RecallRequest.context field
//! v1.0.1-SaaSFix       - 🔧 Extract storage/vector_index from Extensions
//! v2.5.4+BlindRRF      - 🔧 Step 4b hybrid rank fusion (RRF) for node-blind
//!                          recall: keyword evidence was dropped for records
//!                          already surfaced by the vector path, and BM25 vs
//!                          vector scores are incommensurable → fuse by rank
//! v2.5.6+TypedContext  - Typed owner/storage/vector extraction without request
//!                         extension panics; preserves Local-mode fallback.
//!
//! ## Last Modified
//! v2.5.4+BlindRRF - 🔧 blind hybrid recall rank fusion (vector ⊕ keyword)
//! v2.5.5+SessionBound - [RECALL-SESSION-CACHE 2026-07-29 by Codex]
//!                         Owner-isolated, bounded session centroid history.
//! v2.5.6+TypedContext - [RECALL-TYPED-CONTEXT 2026-08-12 by Codex]
//!                         Request context failures are contained HTTP errors.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{body::Body, extract::State, http::StatusCode, response::IntoResponse, Extension, Json};
use serde::Deserialize;
use tracing::debug;

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::graph;
use crate::services::memchain::mvf;
use crate::services::memchain::query_analyzer::{self, QueryType};
use crate::services::memchain::{
    compute_recall_score, cosine_similarity, MemoryStorage, VectorIndex,
};

use super::mpi::{
    estimate_tokens, now_secs, parse_layer, AuthenticatedOwner, MpiState,
    MAX_RECALL_EMBEDDING_DIMENSIONS, MAX_RECALL_SESSION_ID_BYTES,
};

pub use super::mpi_handlers::{
    RecallRequest, RecallResponse, RecalledMemory, SealedMemory, SealedV2Memory, TimeHint,
    TimeRangeParam,
};
pub use crate::services::memchain::reranker::RERANK_TOP_N;

// ============================================
// Helper: extract storage + vector_index from Extensions (v1.0.1-SaaSFix)
// ============================================

/// Resolve Arc<MemoryStorage> from request Extensions (SaaS) or fall back to
/// state.storage (Local). Returns None if neither is available.
fn get_storage(
    storage: Option<Extension<Arc<MemoryStorage>>>,
    state: &MpiState,
) -> Option<Arc<MemoryStorage>> {
    storage
        .map(|Extension(storage)| storage)
        .or_else(|| state.storage.clone())
}

/// Resolve Arc<VectorIndex> from request Extensions (SaaS) or fall back to
/// state.vector_index (Local). Returns None if neither is available.
fn get_vector_index(
    vector_index: Option<Extension<Arc<VectorIndex>>>,
    state: &MpiState,
) -> Option<Arc<VectorIndex>> {
    vector_index
        .map(|Extension(vector_index)| vector_index)
        .or_else(|| state.vector_index.clone())
}

/// Whether a node-blind record passes the client-requested recall scope:
/// cognitive `layer`, `time_range` window, and `project` id-set. Applied to
/// blind FTS and graph hits so `layer` / `time_range` / `context` scope blind
/// recall exactly as they scope the plaintext path. `None` filters are no-ops,
/// so unscoped recall is unchanged.
fn blind_scope_ok(
    record: &MemoryRecord,
    layer: Option<MemoryLayer>,
    time_range: Option<&TimeRangeParam>,
    project_scope: Option<&std::collections::HashSet<[u8; 32]>>,
) -> bool {
    if let Some(l) = layer {
        if record.layer != l {
            return false;
        }
    }
    if let Some(tr) = time_range {
        let ts = record.timestamp as i64;
        if ts < tr.start || ts > tr.end {
            return false;
        }
    }
    if let Some(ids) = project_scope {
        if !ids.contains(&record.record_id) {
            return false;
        }
    }
    true
}

// ============================================
// Constants
// ============================================

const RRF_K: f64 = 60.0;
const RRF_SCALE: f64 = 10.0;

// ============================================
// POST /api/mpi/recall
// ============================================

pub async fn mpi_recall(
    State(state): State<Arc<MpiState>>,
    Extension(auth): Extension<AuthenticatedOwner>,
    storage: Option<Extension<Arc<MemoryStorage>>>,
    vector_index: Option<Extension<Arc<VectorIndex>>>,
    body: Body,
) -> impl IntoResponse {
    let owner = auth.owner_bytes();
    let owner_hex = auth.owner_hex();

    // v1.0.1-SaaSFix: extract storage and vector_index from Extensions
    let storage = match get_storage(storage, &state) {
        Some(s) => s,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "storage unavailable"})),
            )
                .into_response()
        }
    };
    let vector_index = match get_vector_index(vector_index, &state) {
        Some(v) => v,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "vector index unavailable"})),
            )
                .into_response()
        }
    };

    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":"failed to read body"})),
            )
                .into_response()
        }
    };
    let rb: RecallRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("invalid JSON: {}", e)})),
            )
                .into_response()
        }
    };

    // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] Preserve the request
    // shape, but never receive free-text queries at an ordinary node. Clients
    // use local search and send only keyed query terms for sealed records.
    if !rb.query.trim().is_empty() {
        return (
            StatusCode::GONE,
            Json(serde_json::json!({"error":"plaintext recall is disabled; use client-local search and sealed query terms"})),
        )
            .into_response();
    }

    // [MEMORY-SEALED-V2-ENUMERATION 2026-10-02 by Codex] Decode the cursor
    // before query analysis or session-cache work. Standard base64 is the
    // sole wire spelling; malformed, non-canonical, and wrong-length cursors
    // fail closed without storage mutation or a co-occurrence task.
    let sealed_v2_after = match rb.sealed_v2_after.as_deref() {
        None => None,
        Some(value) => {
            use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
            let Some(decoded) = BASE64.decode(value.as_bytes()).ok() else {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error":"invalid sealed v2 cursor"})),
                )
                    .into_response();
            };
            if decoded.len() != 32 || BASE64.encode(&decoded) != value {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error":"invalid sealed v2 cursor"})),
                )
                    .into_response();
            }
            let mut cursor = [0u8; 32];
            cursor.copy_from_slice(&decoded);
            Some(cursor)
        }
    };

    // [RECALL-SESSION-CACHE 2026-07-29 by Codex] Bound attacker-controlled
    // labels and retained vectors before either reaches the process-wide
    // session cache. Existing requests remain wire-compatible.
    if rb
        .session_id
        .as_deref()
        .is_some_and(|session_id| session_id.len() > MAX_RECALL_SESSION_ID_BYTES)
    {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "session_id exceeds 256 UTF-8 bytes"})),
        )
            .into_response();
    }
    if rb.embedding.len() > MAX_RECALL_EMBEDDING_DIMENSIONS {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "embedding exceeds 4096 dimensions"})),
        )
            .into_response();
    }
    // [MEMCHAIN-SEALED-VECTOR-BOUNDARY 2026-10-05 by Codex] Legacy recall
    // bodies still deserialize, but semantic vectors are not accepted by
    // ordinary nodes. Search stays keyword-based here; vectors remain local.
    if !rb.embedding.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error":"unsealed embeddings are not accepted for recall"})),
        )
            .into_response();
    }

    let now = now_secs();
    let layer_filter = rb.layer.as_deref().and_then(parse_layer);
    let top_k = rb.top_k.min(100).max(1);

    // ── Step 1: Query Analysis ──
    let analysis = if !rb.query.is_empty() && (state.ner_engine.is_some() || state.graph_enabled) {
        let known = storage.get_entities_cached(&owner).await;
        Some(query_analyzer::analyze_query(
            &rb.query,
            state.ner_engine.as_deref(),
            &known,
            now as i64,
        ))
    } else {
        None
    };

    let query_type = analysis
        .as_ref()
        .map(|a| a.query_type.clone())
        .unwrap_or(QueryType::Semantic);

    // ── Session centroid for MVF φ₇ ──
    let session_centroid: Option<Vec<f32>> = if let Some(ref sid) = rb.session_id {
        if !rb.embedding.is_empty() {
            Some(
                state
                    .session_embeddings
                    .write()
                    .push_and_centroid(&owner, sid, &rb.embedding),
            )
        } else {
            None
        }
    } else {
        None
    };

    let mut memories: Vec<RecalledMemory> = Vec::new();
    let mut total_tokens = 0usize;
    let mut seen_ids: Vec<[u8; 32]> = Vec::new();

    // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] Legacy sighted identity
    // cache entries are deliberately not injected into node recall results.

    // ── Step 2a: Vector search ──
    let idx_ready = state.index_ready.load(std::sync::atomic::Ordering::Relaxed);
    let search = if !rb.embedding.is_empty() && idx_ready {
        vector_index.search_filtered(
            &rb.embedding,
            &owner,
            &rb.embedding_model,
            layer_filter,
            top_k * 3,
            0.0,
        )
    } else {
        Vec::new()
    };

    // ── Step 2a-bis: BM25 FTS5 search ──
    let bm25_results: Vec<(String, String, f64)> = if !rb.query.is_empty() {
        storage.bm25_search(&rb.query, &owner, top_k * 3).await
    } else {
        Vec::new()
    };

    // ── Step 2a-ter: BM25 entity/session direct injection ──
    let mut bm25_direct_memories: Vec<RecalledMemory> = Vec::new();
    for (source_type, source_id, bm25_score) in &bm25_results {
        match source_type.as_str() {
            "entity" => {
                if let Some(entity) = storage.get_entity(source_id).await {
                    let edges = storage.get_edges_for_entity(source_id, &owner).await;
                    let mut parts: Vec<String> =
                        vec![format!("{} ({})", entity.name, entity.entity_type)];
                    if let Some(ref desc) = entity.description {
                        if !desc.contains(&entity.name) {
                            parts.push(desc.clone());
                        }
                    }
                    let mut edge_descs: Vec<String> = Vec::new();
                    for edge in edges.iter().take(5) {
                        let other_id = if edge.source_id == *source_id {
                            &edge.target_id
                        } else {
                            &edge.source_id
                        };
                        if let Some(other) = storage.get_entity(other_id).await {
                            edge_descs.push(if edge.source_id == *source_id {
                                format!("{} → {}", edge.relation_type, other.name)
                            } else {
                                format!("{} ← {}", other.name, edge.relation_type)
                            });
                        }
                    }
                    if !edge_descs.is_empty() {
                        parts.push(format!("Relations: {}", edge_descs.join("; ")));
                    }
                    bm25_direct_memories.push(RecalledMemory {
                        record_id: format!("bm25_entity_{}", source_id),
                        layer: "knowledge".into(),
                        score: 1.5 + bm25_score.min(1.0),
                        content: parts.join(". "),
                        topic_tags: vec!["bm25_result".into(), "entity_knowledge".into()],
                        source_ai: "bm25-search".into(),
                        timestamp: entity.updated_at as u64,
                        access_count: entity.mention_count as u32,
                        proactive: false,
                    });
                }
            }
            "session" => {
                if let Some(session) = storage.get_session(source_id, &owner).await {
                    if let Some(ref summary) = session.summary {
                        bm25_direct_memories.push(RecalledMemory {
                            record_id: format!("bm25_session_{}", source_id),
                            layer: "knowledge".into(),
                            score: 1.8 + bm25_score.min(1.0),
                            content: format!("[Session: {}] {}", source_id, summary),
                            topic_tags: vec!["bm25_result".into(), "session_summary".into()],
                            source_ai: "bm25-search".into(),
                            timestamp: session.started_at as u64,
                            access_count: 0,
                            proactive: false,
                        });
                    }
                }
            }
            "record" => {
                if let Ok(id_bytes) = hex::decode(source_id) {
                    if id_bytes.len() == 32 {
                        let mut rid = [0u8; 32];
                        rid.copy_from_slice(&id_bytes);
                        if storage.get(&rid).await.is_none() {
                            let content: Option<String> = {
                                let conn = storage.conn_lock().await;
                                conn.query_row(
                                    "SELECT content FROM fts_index WHERE source_type = 'record' AND source_id = ?1",
                                    rusqlite::params![source_id.as_str()], |row| row.get::<_, String>(0),
                                ).ok()
                            };
                            if let Some(text) = content {
                                bm25_direct_memories.push(RecalledMemory {
                                    record_id: format!(
                                        "bm25_turn_{}",
                                        &source_id[..source_id.len().min(16)]
                                    ),
                                    layer: "episode".into(),
                                    score: 1.2 + bm25_score.min(1.0),
                                    content: text,
                                    topic_tags: vec![
                                        "bm25_result".into(),
                                        "conversation_turn".into(),
                                    ],
                                    source_ai: "bm25-search".into(),
                                    timestamp: now,
                                    access_count: 0,
                                    proactive: false,
                                });
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if !bm25_direct_memories.is_empty() {
        debug!(
            bm25_direct = bm25_direct_memories.len(),
            "[RECALL] BM25 direct injection"
        );
    }

    // ── Step 2b: Graph BFS ──
    let graph_traversed: HashMap<String, f64> = if state.graph_enabled && rb.include_graph {
        if let Some(ref qa) = analysis {
            let matched_ids: Vec<String> = qa
                .matched_entities
                .iter()
                .filter_map(|e| e.entity_id.clone())
                .collect();
            if !matched_ids.is_empty() {
                let conn = storage.conn_lock().await;
                let nodes = graph::bfs_traverse(&conn, &owner, &matched_ids, 2, 20, 0.3);
                drop(conn);
                nodes.into_iter().map(|n| (n.entity_id, n.weight)).collect()
            } else {
                HashMap::new()
            }
        } else {
            HashMap::new()
        }
    } else {
        HashMap::new()
    };

    // ── Step 2c: Graph content retrieval ──
    let mut graph_memories: Vec<RecalledMemory> = Vec::new();
    if !graph_traversed.is_empty() {
        let graph_entity_ids: Vec<String> = graph_traversed.keys().cloned().collect();

        let mut seen_episodes: HashSet<String> = HashSet::new();
        for eid in &graph_entity_ids {
            for (episode_id, _role) in storage.get_episodes_for_entity(eid).await {
                if seen_episodes.contains(&episode_id) {
                    continue;
                }
                seen_episodes.insert(episode_id.clone());
                let session_id = if episode_id.starts_with("ep_") {
                    let rest = &episode_id[3..];
                    rest.rfind('_').map(|i| rest[..i].to_string())
                } else {
                    None
                };
                if let Some(ref sid) = session_id {
                    if let Some(session) = storage.get_session(sid, &owner).await {
                        if let Some(ref summary) = session.summary {
                            let weight = graph_traversed.get(eid).copied().unwrap_or(0.5);
                            graph_memories.push(RecalledMemory {
                                record_id: format!("graph_session_{}", sid),
                                layer: "knowledge".to_string(),
                                score: 2.0 + weight,
                                content: format!("[Session: {}] {}", sid, summary),
                                topic_tags: vec!["graph_result".into(), "session_summary".into()],
                                source_ai: "cognitive-graph".into(),
                                timestamp: session.started_at as u64,
                                access_count: 0,
                                proactive: false,
                            });
                        }
                    }
                }
            }
        }

        for eid in &graph_entity_ids {
            if let Some(entity) = storage.get_entity(eid).await {
                let edges = storage.get_edges_for_entity(eid, &owner).await;
                let weight = graph_traversed.get(eid).copied().unwrap_or(0.5);
                let mut desc_parts: Vec<String> =
                    vec![format!("{} ({})", entity.name, entity.entity_type)];
                if let Some(ref desc) = entity.description {
                    if !desc.contains(&entity.name) {
                        desc_parts.push(desc.clone());
                    }
                }
                let mut edge_descs: Vec<String> = Vec::new();
                for edge in edges.iter().take(5) {
                    let other_id = if edge.source_id == *eid {
                        &edge.target_id
                    } else {
                        &edge.source_id
                    };
                    if let Some(other) = storage.get_entity(other_id).await {
                        edge_descs.push(if edge.source_id == *eid {
                            format!("{} → {}", edge.relation_type, other.name)
                        } else {
                            format!("{} ← {}", other.name, edge.relation_type)
                        });
                    }
                    if let Some(ref fact) = edge.fact_text {
                        edge_descs.push(fact.clone());
                    }
                }
                if !edge_descs.is_empty() {
                    desc_parts.push(format!("Relations: {}", edge_descs.join("; ")));
                }
                graph_memories.push(RecalledMemory {
                    record_id: format!("graph_entity_{}", eid),
                    layer: "knowledge".to_string(),
                    score: 1.5 + weight,
                    content: desc_parts.join(". "),
                    topic_tags: vec!["graph_result".into(), "entity_knowledge".into()],
                    source_ai: "cognitive-graph".into(),
                    timestamp: entity.updated_at as u64,
                    access_count: entity.mention_count as u32,
                    proactive: false,
                });
            }
        }
    }

    // ── Step 3: RRF Fusion + Scoring ──
    let total_candidates = search.len() + bm25_results.len() + seen_ids.len();
    let mut rrf_scores: HashMap<[u8; 32], f64> = HashMap::new();

    for (rank, sr) in search.iter().enumerate() {
        *rrf_scores.entry(sr.record_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }
    for (rank, (source_type, source_id, _)) in bm25_results.iter().enumerate() {
        if source_type == "record" {
            if let Ok(id_bytes) = hex::decode(source_id) {
                if id_bytes.len() == 32 {
                    let mut rid = [0u8; 32];
                    rid.copy_from_slice(&id_bytes);
                    *rrf_scores.entry(rid).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
                }
            }
        }
    }

    let bm25_only_ids: Vec<[u8; 32]> = rrf_scores
        .keys()
        .filter(|rid| !search.iter().any(|sr| sr.record_id == **rid))
        .cloned()
        .collect();

    let max_degree = {
        let c = storage.conn_lock().await;
        graph::get_max_degree(&c, &owner)
    };
    let time_hint_tuple = rb.time_hint.as_ref().map(|th| (th.start, th.end));

    let mut scored: Vec<(MemoryRecord, f64)> = Vec::new();

    for sr in &search {
        if seen_ids.contains(&sr.record_id) {
            continue;
        }
        if let Some(record) = storage.get(&sr.record_id).await {
            if !record.is_active() || record.owner != owner {
                continue;
            }
            let v_old = compute_recall_score(
                sr.similarity,
                record.timestamp,
                now,
                record.access_count,
                record.layer,
            );
            let time_bonus: f64 = match &rb.time_hint {
                Some(th)
                    if (record.timestamp as i64) >= th.start
                        && (record.timestamp as i64) <= th.end =>
                {
                    0.15
                }
                _ => 0.0,
            };
            let graph_weight: f32 = if !graph_traversed.is_empty() {
                let tags_lower: HashSet<String> = record
                    .topic_tags
                    .iter()
                    .map(|t: &String| t.to_lowercase())
                    .collect();
                graph_traversed
                    .iter()
                    .find(|(eid, _)| tags_lower.contains(&eid.to_lowercase()))
                    .map(|(_, w)| *w as f32)
                    .unwrap_or(0.0)
            } else {
                0.0
            };
            let v_old_h = v_old + time_bonus;
            let final_score = if state.mvf_enabled {
                let gd = {
                    let c = storage.conn_lock().await;
                    graph::get_degree(&c, &record.record_id)
                };
                let cs = session_centroid
                    .as_ref()
                    .filter(|c| record.has_embedding() && c.len() == record.embedding.len())
                    .map(|c| cosine_similarity(c, &record.embedding))
                    .unwrap_or(0.0);
                let phi = mvf::compute_features(
                    sr.similarity,
                    record.layer as u8,
                    record.timestamp,
                    now,
                    record.access_count,
                    record.positive_feedback,
                    record.negative_feedback,
                    record.has_conflict(),
                    time_hint_tuple,
                    cs,
                    gd,
                    max_degree,
                    graph_weight,
                );
                let mut uw = {
                    state
                        .user_weights
                        .read()
                        .get(&owner_hex)
                        .cloned()
                        .unwrap_or_else(mvf::default_weights)
                };
                let pn = mvf::normalize(&phi, &mut uw);
                let vm = mvf::compute_value(&uw, &pn);
                {
                    state.user_weights.write().insert(owner_hex.clone(), uw);
                }
                mvf::fuse_scores(vm, v_old_h, state.mvf_alpha)
            } else {
                v_old_h + (graph_weight as f64 * 0.1)
            };
            let rrf_boost = rrf_scores.get(&sr.record_id).copied().unwrap_or(0.0) * RRF_SCALE;
            scored.push((record, final_score + rrf_boost));
        }
    }

    for rid in &bm25_only_ids {
        if seen_ids.contains(rid) {
            continue;
        }
        if let Some(record) = storage.get(rid).await {
            if !record.is_active() || record.owner != owner {
                continue;
            }
            let rrf = rrf_scores.get(rid).copied().unwrap_or(0.0);
            let base = compute_recall_score(
                0.5,
                record.timestamp,
                now,
                record.access_count,
                record.layer,
            );
            scored.push((record, base + rrf * RRF_SCALE));
        }
    }

    // Recent-records fallback: only when the client gave NO search signal at all.
    // A node-blind client searches via `query_terms` (Step 4b, node-blind BM25);
    // if it supplied them we must NOT fall back to recent records, otherwise an
    // unmatched blind query returns recent memories as false positives and swamps
    // the blind-FTS ranking (breaking disambiguation and precision).
    if search.is_empty() && bm25_results.is_empty() && rb.query_terms.is_empty() {
        // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] The no-query
        // recovery path is restricted to client-sealed rows.
        let recent = storage
            .get_active_blind_records(&owner, layer_filter, top_k)
            .await;
        for r in recent {
            if seen_ids.contains(&r.record_id) {
                continue;
            }
            scored.push((
                r.clone(),
                compute_recall_score(1.0, r.timestamp, now, r.access_count, r.layer),
            ));
        }
    }

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 4.1: Context / project scope (v2.5.3+Isolation) ──
    // Resolve the project id-set ONCE. It scopes BOTH the plaintext `scored`
    // path (below) and the node-blind FTS / graph hits (Step 4b/4c) — a single
    // source of project scoping so blind recall honors `context` too.
    let context_project_id: Option<String> = match rb.context.as_deref() {
        None | Some("all") | Some("") => None,
        Some(ctx) => Some(ctx.to_string()),
    };
    let project_scope: Option<HashSet<[u8; 32]>> = match context_project_id {
        Some(ref pid) => Some(storage.project_record_ids(&owner, pid).await),
        None => None,
    };

    if let Some(ref ids) = project_scope {
        let before = scored.len();
        scored.retain(|(r, _)| ids.contains(&r.record_id));
        debug!(
            before = before,
            after = scored.len(),
            "[RECALL] Step 4.1 project scope applied"
        );
    }

    // ── Step 4: Token budget ──
    let mut returned_ids = seen_ids.clone();
    let mut sealed_memories: Vec<SealedMemory> = Vec::new();
    for (r, score) in &scored {
        if !r.blind {
            continue;
        }
        // Time-range scope also applies to the scored (vector / recent) path.
        if let Some(ref tr) = rb.time_range {
            let ts = r.timestamp as i64;
            if ts < tr.start || ts > tr.end {
                continue;
            }
        }
        // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] Only client-sealed
        // record bytes may leave this handler; no sighted fallback projection.
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        returned_ids.push(r.record_id);
        sealed_memories.push(SealedMemory {
            record_id: r.id_hex(),
            score: *score,
            ciphertext_b64: BASE64.encode(&r.encrypted_content),
            timestamp: r.timestamp,
        });
        let st = Arc::clone(&storage);
        let rid = r.record_id;
        tokio::spawn(async move {
            st.increment_access(&rid).await;
        });
    }

    for gm in graph_memories {
        let tokens = estimate_tokens(&gm.content);
        if total_tokens + tokens > rb.token_budget && !memories.is_empty() {
            break;
        }
        total_tokens += tokens;
        memories.push(gm);
    }
    for bm in bm25_direct_memories {
        let tokens = estimate_tokens(&bm.content);
        if total_tokens + tokens > rb.token_budget && !memories.is_empty() {
            break;
        }
        total_tokens += tokens;
        memories.push(bm);
    }

    // ── Step 4b: node-blind full-text (BM25 over client-hashed query terms) ──
    // Blind records surfaced by keyword join the sealed list (ciphertext) and are
    // RANK-FUSED with the blind records already returned via the vector path.
    if !rb.query_terms.is_empty() {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        let blind_hits = storage
            .bm25_search_blind(&rb.query_terms, &owner, top_k * 3)
            .await;
        // ── Hybrid rank fusion (RRF) over the node-blind results ──
        // Vector-path scores (compute_recall_score, ~0..1.3) and blind-keyword
        // BM25 (unbounded) are incommensurable, and the previous dedup DROPPED
        // the keyword evidence for records already surfaced by the vector path
        // — hybrid recall degenerated to vector-only whenever the vector
        // candidate pool covered the keyword hits (and raw-score sorting let
        // keyword drown vector otherwise). Fuse by rank instead:
        // score = Σ_src 61/(60 + rank_src + 1), scale-free across sources.
        // Single-source entries land in ~0.73..1.0 — above graph expansion's
        // ≤0.5, matching the old ordering semantics — and entries confirmed by
        // BOTH sources rank on top (up to 2.0).
        let vec_rank: std::collections::HashMap<String, usize> = sealed_memories
            .iter()
            .enumerate()
            .map(|(i, m)| (m.record_id.clone(), i))
            .collect();
        let mut kw_rank: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (i, (rid_hex, _)) in blind_hits.iter().enumerate() {
            kw_rank.entry(rid_hex.clone()).or_insert(i);
        }
        for (rid_hex, _bm25) in blind_hits {
            let rid = match hex::decode(&rid_hex) {
                Ok(b) if b.len() == 32 => {
                    let mut a = [0u8; 32];
                    a.copy_from_slice(&b);
                    a
                }
                _ => continue,
            };
            if returned_ids.contains(&rid) {
                continue; // already sealed via the vector path — fused below
            }
            if let Some(record) = storage.get(&rid).await {
                if !record.is_active() || record.owner != owner || !record.blind {
                    continue;
                }
                // Honor the client recall scope (layer / time_range / project).
                if !blind_scope_ok(
                    &record,
                    layer_filter,
                    rb.time_range.as_ref(),
                    project_scope.as_ref(),
                ) {
                    continue;
                }
                returned_ids.push(rid);
                sealed_memories.push(SealedMemory {
                    record_id: rid_hex,
                    score: 0.0, // fused right below
                    ciphertext_b64: BASE64.encode(&record.encrypted_content),
                    timestamp: record.timestamp,
                });
            }
        }
        for m in sealed_memories.iter_mut() {
            let v = vec_rank.get(&m.record_id);
            let k = kw_rank.get(&m.record_id);
            if v.is_some() || k.is_some() {
                let mut s = 0.0f64;
                if let Some(r) = v {
                    s += 61.0 / (60.0 + *r as f64 + 1.0);
                }
                if let Some(r) = k {
                    s += 61.0 / (60.0 + *r as f64 + 1.0);
                }
                m.score = s;
            }
        }
    }

    // ── Step 4c: node-blind graph expansion ──
    // Pull records the client declared related to any blind hit so far into the
    // sealed list (opaque record_id edges only; the node never learns why).
    if rb.include_graph && !sealed_memories.is_empty() {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        let seeds: Vec<[u8; 32]> = sealed_memories
            .iter()
            .filter_map(|m| {
                hex::decode(&m.record_id).ok().and_then(|b| {
                    if b.len() == 32 {
                        let mut a = [0u8; 32];
                        a.copy_from_slice(&b);
                        Some(a)
                    } else {
                        None
                    }
                })
            })
            .collect();
        // Gather neighbours UNDER the connection lock, then release it before any
        // storage.get() below (which re-locks the same connection — holding both
        // would deadlock).
        let neighbors: Vec<([u8; 32], f32)> = {
            let conn = storage.conn_lock().await;
            let mut acc = Vec::new();
            for seed in &seeds {
                acc.extend(graph::get_neighbors(&conn, seed, 10));
            }
            acc
        };
        for (nid, weight) in neighbors {
            if returned_ids.contains(&nid) {
                continue;
            }
            if let Some(record) = storage.get(&nid).await {
                if !record.is_active() || record.owner != owner || !record.blind {
                    continue;
                }
                // Blind graph neighbours honor the same recall scope.
                if !blind_scope_ok(
                    &record,
                    layer_filter,
                    rb.time_range.as_ref(),
                    project_scope.as_ref(),
                ) {
                    continue;
                }
                returned_ids.push(nid);
                sealed_memories.push(SealedMemory {
                    record_id: hex::encode(nid),
                    score: weight as f64 * 0.5,
                    ciphertext_b64: BASE64.encode(&record.encrypted_content),
                    timestamp: record.timestamp,
                });
            }
        }
    }

    memories.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    memories.truncate(top_k);
    sealed_memories.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sealed_memories.truncate(top_k);

    // Compute matched_json BEFORE index-mode branch (avoid borrow-after-move)
    let matched_json: Option<Vec<serde_json::Value>> = analysis.as_ref().map(|qa| {
        qa.matched_entities
            .iter()
            .map(|e| {
                serde_json::json!({
                    "text": e.query_text, "label": e.label, "confidence": e.confidence,
                    "entity_id": e.entity_id, "entity_type": e.entity_type,
                })
            })
            .collect()
    });
    let query_type_str = format!("{:?}", query_type).to_lowercase();
    let (sealed_v2, sealed_v2_next_cursor) = {
        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        match storage
            .list_sealed_v2(&owner, sealed_v2_after.as_ref(), top_k)
            .await
        {
            Ok(page) => {
                let next_cursor = page.next_cursor.map(|cursor| BASE64.encode(cursor));
                let rows = page
                    .rows
                    .into_iter()
                    .map(|row| SealedV2Memory {
                        record_id_b64: BASE64.encode(row.record_id),
                        created_at: row.created_at,
                        envelope_b64: BASE64.encode(row.envelope),
                        signature_b64: BASE64.encode(row.signature),
                    })
                    .collect::<Vec<_>>();
                (rows, next_cursor)
            }
            Err(_) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error":"sealed v2 storage unavailable"})),
                )
                    .into_response()
            }
        }
    };

    // ── Step 4.5: Progressive index mode ──
    if rb.mode == "index" {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "mode": "index",
                // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] Keep the
                // legacy response key, but only ciphertext-bearing sealed
                // collections may leave this node.
                "memories": Vec::<RecalledMemory>::new(),
                "total_candidates": total_candidates,
                "token_estimate": 0,
                "query_type": query_type_str,
                "matched_entities": matched_json,
                "sealed": sealed_memories,
                "sealed_v2": sealed_v2,
                "sealed_v2_next_cursor": sealed_v2_next_cursor,
                "hint": "Decrypt returned sealed ciphertext on the client.",
            })),
        )
            .into_response();
    }

    // ── Async co-occurrence update (full mode only) ──
    {
        let st = Arc::clone(&storage);
        let ids = returned_ids;
        tokio::spawn(async move {
            let c = st.conn_lock().await;
            let n = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;
            graph::update_cooccurrence(&c, &ids, n);
        });
    }

    (
        StatusCode::OK,
        Json(serde_json::json!(RecallResponse {
            // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] Preserve the
            // wire field without returning historical plaintext records.
            memories: Vec::new(),
            total_candidates,
            token_estimate: 0,
            query_type: Some(query_type_str),
            matched_entities: matched_json,
            sealed: sealed_memories,
            sealed_v2,
            sealed_v2_next_cursor,
        })),
    )
        .into_response()
}

// ============================================
// POST /api/mpi/recall/detail
// ============================================

#[derive(Debug, serde::Deserialize)]
pub struct DetailRequest {
    pub record_ids: Vec<String>,
}

/// Fetch sealed envelopes/ciphertext for selected IDs. Legacy plaintext
/// records are never serialized by this ordinary-node endpoint.
///
/// Synthetic IDs (graph_*, bm25_*) are silently skipped — no backing records.
/// Max 20 record_ids per request.
pub async fn mpi_recall_detail(
    State(state): State<Arc<MpiState>>,
    Extension(auth): Extension<AuthenticatedOwner>,
    storage: Option<Extension<Arc<MemoryStorage>>>,
    body: Body,
) -> impl IntoResponse {
    let owner = auth.owner_bytes();

    // v1.0.1-SaaSFix: extract storage from Extensions
    let storage = match get_storage(storage, &state) {
        Some(s) => s,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "storage unavailable"})),
            )
                .into_response()
        }
    };

    let body_bytes = match axum::body::to_bytes(body, 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":"failed to read body"})),
            )
                .into_response()
        }
    };
    let dr: DetailRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("invalid JSON: {}", e)
                })),
            )
                .into_response()
        }
    };

    if dr.record_ids.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "record_ids array is empty"
            })),
        )
            .into_response();
    }
    if dr.record_ids.len() > 20 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "max 20 record_ids per request"
            })),
        )
            .into_response();
    }

    let memories: Vec<RecalledMemory> = Vec::new();
    let mut sealed: Vec<SealedMemory> = Vec::new();
    let mut sealed_v2: Vec<SealedV2Memory> = Vec::new();

    for rid_hex in &dr.record_ids {
        if rid_hex.starts_with("graph_") || rid_hex.starts_with("bm25_") {
            continue;
        }

        use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
        let rid = BASE64
            .decode(rid_hex.as_bytes())
            .ok()
            .filter(|b| BASE64.encode(b) == *rid_hex && b.len() == 32)
            .or_else(|| hex::decode(rid_hex).ok().filter(|b| b.len() == 32));
        let rid = match rid {
            Some(bytes) => {
                let mut a = [0u8; 32];
                a.copy_from_slice(&bytes);
                a
            }
            None => continue,
        };

        match storage.get_sealed_v2(&owner, &rid).await {
            Ok(Some(row)) => {
                sealed_v2.push(SealedV2Memory {
                    record_id_b64: BASE64.encode(row.record_id),
                    created_at: row.created_at,
                    envelope_b64: BASE64.encode(row.envelope),
                    signature_b64: BASE64.encode(row.signature),
                });
                continue;
            }
            Ok(None) => {}
            Err(_) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({"error":"sealed v2 storage unavailable"})),
                )
                    .into_response()
            }
        }

        if let Some(record) = storage.get(&rid).await {
            if !record.is_active() || record.owner != owner || !record.blind {
                continue;
            }
            use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
            sealed.push(SealedMemory {
                record_id: rid_hex.clone(),
                score: 0.0,
                ciphertext_b64: BASE64.encode(&record.encrypted_content),
                timestamp: record.timestamp,
            });

            let st = Arc::clone(&storage);
            tokio::spawn(async move {
                st.increment_access(&rid).await;
            });
        }
    }

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "memories": memories,
            "token_estimate": 0,
            "sealed": sealed,
            "sealed_v2": sealed_v2,
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::mpi::SessionEmbeddingCache;
    use aeronyx_core::crypto::IdentityKeyPair;
    use axum::http::Request;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::atomic::AtomicBool;
    use tower::ServiceExt;

    fn make_test_state() -> (MpiState, AuthenticatedOwner) {
        let storage = Arc::new(MemoryStorage::open(":memory:", None).unwrap());
        let vector_index = Arc::new(VectorIndex::new());
        let identity = IdentityKeyPair::generate();
        let owner = identity.public_key_bytes();
        let state = MpiState::local(
            storage,
            vector_index,
            identity,
            RwLock::new(HashMap::new()),
            AtomicBool::new(true),
            Arc::new(RwLock::new(HashMap::new())),
            0.0,
            false,
            RwLock::new(SessionEmbeddingCache::default()),
            RwLock::new(None),
            owner,
            None,
            None,
            false,
            false,
            0,
            None,
            false,
            false,
            None,
            None,
            None,
        );
        (state, AuthenticatedOwner::Local { owner })
    }

    #[tokio::test]
    async fn typed_recall_context_contains_missing_middleware_without_panicking() {
        // [RECALL-TYPED-CONTEXT 2026-08-12 by Codex] Broken router wiring must
        // be rejected as HTTP 500 and never reach an extension `expect` under
        // release `panic=abort`.
        let (state, _) = make_test_state();
        let state = Arc::new(state);
        let missing_auth = axum::Router::new()
            .route("/recall", axum::routing::post(mpi_recall))
            .route("/recall/detail", axum::routing::post(mpi_recall_detail))
            .with_state(state);

        for uri in ["/recall", "/recall/detail"] {
            let response = missing_auth
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        }
    }

    #[tokio::test]
    async fn recall_rejects_unsealed_embedding_before_search() {
        // [MEMCHAIN-SEALED-VECTOR-BOUNDARY 2026-10-05 by Codex] Old request
        // fields remain deserializable, but remote semantic vectors are refused.
        let (state, auth) = make_test_state();
        let app = axum::Router::new()
            .route("/recall", axum::routing::post(mpi_recall))
            .layer(Extension(auth))
            .with_state(Arc::new(state));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"embedding":[0.1]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn recall_rejects_plaintext_query_before_search() {
        // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] Ordinary nodes
        // accept keyed query terms, not user plaintext.
        let (state, auth) = make_test_state();
        let app = axum::Router::new()
            .route("/recall", axum::routing::post(mpi_recall))
            .layer(Extension(auth))
            .with_state(Arc::new(state));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"query":"private search terms"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GONE);
    }

    #[tokio::test]
    async fn recent_recall_returns_no_legacy_sighted_content() {
        // [MEMCHAIN-NODE-BLIND-RECALL 2026-10-05 by Codex] The empty-query
        // recovery path enumerates only blind rows and never projects old text.
        let (state, auth) = make_test_state();
        let storage = Arc::clone(state.storage.as_ref().unwrap());
        let owner = auth.owner_bytes();
        let mut record = MemoryRecord::new(
            owner,
            now_secs(),
            MemoryLayer::Knowledge,
            vec![],
            "legacy".into(),
            b"private historical text".to_vec(),
            vec![],
        );
        record.signature = state.identity.sign(&record.record_id);
        assert!(storage.insert(&record, "").await);

        let app = axum::Router::new()
            .route("/recall", axum::routing::post(mpi_recall))
            .layer(Extension(auth))
            .with_state(Arc::new(state));
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["memories"], serde_json::json!([]));
        assert_eq!(json["sealed"], serde_json::json!([]));
        assert_eq!(storage.count().await, 1);
    }

    #[tokio::test]
    async fn typed_recall_context_preserves_local_fallback_and_missing_store_errors() {
        let (state, auth) = make_test_state();
        let state_storage = state.storage.as_ref().unwrap();
        let state_vector = state.vector_index.as_ref().unwrap();
        assert!(Arc::ptr_eq(
            &get_storage(None, &state).expect("local storage fallback"),
            state_storage,
        ));
        assert!(Arc::ptr_eq(
            &get_vector_index(None, &state).expect("local vector fallback"),
            state_vector,
        ));

        let mut missing_storage_state = state;
        missing_storage_state.storage = None;
        let missing_storage = axum::Router::new()
            .route("/recall", axum::routing::post(mpi_recall))
            .layer(Extension(auth.clone()))
            .with_state(Arc::new(missing_storage_state));
        let response = missing_storage
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let (mut missing_vector_state, _) = make_test_state();
        missing_vector_state.vector_index = None;
        let missing_vector = axum::Router::new()
            .route("/recall", axum::routing::post(mpi_recall))
            .layer(Extension(auth))
            .with_state(Arc::new(missing_vector_state));
        let response = missing_vector
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/recall")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
