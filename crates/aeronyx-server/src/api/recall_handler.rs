// ============================================
// File: crates/aeronyx-server/src/api/recall_handler.rs
// ============================================
//! # POST /api/mpi/recall — Hybrid Retrieval Pipeline
//!
//! ## Creation Reason (v2.4.0+BM25 Split)
//! Extracted from mpi_handlers.rs because the recall handler grew to ~300 lines
//! with the addition of graph traversal (v2.4.0) and BM25 search (v2.4.0+BM25).
//! The recall pipeline is the most complex single handler in the MPI API.
//!
//! ## Pipeline Overview
//! ```text
//! POST /api/mpi/recall {query, embedding, top_k, ...}
//!   │
//!   ├─ Step 1: Query Analysis (GLiNER + regex + entity matching)
//!   │
//!   ├─ Step 2a: Vector search (cosine similarity, always runs)
//!   ├─ Step 2a-bis: BM25 search (FTS5 keyword matching) ← NEW
//!   ├─ Step 2b: Graph BFS (knowledge_edges traversal)
//!   ├─ Step 2c: Graph content retrieval (episodes + entities)
//!   │
//!   ├─ Step 3: RRF fusion (vector + BM25 rank merging) ← NEW
//!   │   └─ MVF φ₀-φ₉ scoring with graph_traverse_weight
//!   │
//!   └─ Step 4: Token budget trimming → Response
//! ```
//!
//! ## Reciprocal Rank Fusion (RRF)
//! ```text
//! RRF_score(d) = Σ 1/(k + rank_i(d))
//! k = 60 (standard value from Cormack et al. 2009)
//! ```
//! Rank-based fusion — no score normalization needed. Documents appearing
//! in multiple retrieval sources get boosted naturally.
//!
//! ## Dependencies
//! - mpi.rs — MpiState, AuthenticatedOwner, helpers
//! - mpi_handlers.rs — RecallRequest/Response types (re-exported)
//! - storage_fts.rs — bm25_search()
//! - graph.rs — bfs_traverse()
//! - query_analyzer.rs — analyze_query()
//! - mvf.rs — compute_features(), fuse_scores()
//!
//! ⚠️ Important Note for Next Developer:
//! - RRF k=60 is the standard value. Changing it affects fusion behavior:
//!   lower k → more weight on top-ranked results, higher k → more uniform.
//! - BM25 and vector results use DIFFERENT score scales. RRF avoids this
//!   problem by using ranks, not raw scores.
//! - Graph memories (Step 2c) bypass RRF — they have fixed scores (2.0-3.0)
//!   because they don't participate in the vector/BM25 ranking.
//! - BM25 hits of type "entity" and "session" are already covered by Step 2c.
//!   Only "record" type BM25 hits add new candidates to RRF.
//!
//! ## Last Modified
//! v2.4.0+BM25 - 🌟 Extracted from mpi_handlers.rs; added BM25 + RRF fusion

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{compute_recall_score, cosine_similarity};
use crate::services::memchain::mvf;
use crate::services::memchain::graph;
use crate::services::memchain::query_analyzer::{self, QueryType};

use super::mpi::{
    MpiState, AuthenticatedOwner,
    extract_owner, parse_layer, estimate_tokens, now_secs,
    default_model, default_top_k, default_token_budget, default_include_graph,
};

// Re-export types from mpi_handlers for backward compatibility
pub use super::mpi_handlers::{
    RecallRequest, RecallResponse, RecalledMemory,
    TimeHint, TimeRangeParam,
};

// ============================================
// Constants
// ============================================

/// Reciprocal Rank Fusion constant (Cormack et al. 2009).
/// k=60 is the standard value. Higher = more uniform weighting across ranks.
const RRF_K: f64 = 60.0;

/// Scale factor for RRF scores when adding to recall base scores.
/// RRF raw scores are ~0.01-0.03, which is tiny compared to recall scores (~1-3).
/// Scaling by 10 makes RRF meaningfully influence final ranking.
const RRF_SCALE: f64 = 10.0;

// ============================================
// POST /api/mpi/recall — Hybrid Pipeline
// ============================================

/// Hybrid recall handler: vector + BM25 + graph → RRF fusion → MVF rerank.
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

    // ── Session centroid for MVF φ₇ ──
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

    // ── Step 2a-bis: BM25 full-text search (v2.4.0+BM25) ──
    let bm25_results = if !rb.query.is_empty() {
        state.storage.bm25_search(&rb.query, &owner, top_k * 3).await
    } else {
        Vec::new()
    };

    // ── Step 2b: Graph BFS (v2.4.0) ──
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

    // ── Step 2c: Graph content retrieval (v2.4.0) ──
    let mut graph_memories: Vec<RecalledMemory> = Vec::new();

    if !graph_traversed.is_empty() {
        let graph_entity_ids: Vec<String> = graph_traversed.keys().cloned().collect();

        // Path 1: Entity → episode_edges → session summary
        let mut seen_episodes: HashSet<String> = HashSet::new();
        for eid in &graph_entity_ids {
            let ep_links = state.storage.get_episodes_for_entity(eid).await;
            for (episode_id, _role) in &ep_links {
                if seen_episodes.contains(episode_id) { continue; }
                seen_episodes.insert(episode_id.clone());

                let session_id = if episode_id.starts_with("ep_") {
                    let rest = &episode_id[3..];
                    rest.rfind('_').map(|i| rest[..i].to_string())
                } else { None };

                if let Some(ref sid) = session_id {
                    if let Some(session) = state.storage.get_session(sid).await {
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
                                access_count: 0, proactive: false,
                            });
                        }
                    }
                }
            }
        }

        // Path 3: Entity descriptions + relationship edges
        for eid in &graph_entity_ids {
            if let Some(entity) = state.storage.get_entity(eid).await {
                let edges = state.storage.get_edges_for_entity(eid, &owner).await;
                let weight = graph_traversed.get(eid).copied().unwrap_or(0.5);

                let mut desc_parts: Vec<String> = Vec::new();
                desc_parts.push(format!("{} ({})", entity.name, entity.entity_type));

                if let Some(ref desc) = entity.description {
                    if !desc.contains(&entity.name) {
                        desc_parts.push(desc.clone());
                    }
                }

                let mut edge_descs: Vec<String> = Vec::new();
                for edge in edges.iter().take(5) {
                    let other_id = if edge.source_id == *eid { &edge.target_id } else { &edge.source_id };
                    if let Some(other) = state.storage.get_entity(other_id).await {
                        let rel = if edge.source_id == *eid {
                            format!("{} → {}", edge.relation_type, other.name)
                        } else {
                            format!("{} ← {}", other.name, edge.relation_type)
                        };
                        edge_descs.push(rel);
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
                    access_count: entity.mention_count as u32, proactive: false,
                });
            }
        }

        if !graph_memories.is_empty() {
            debug!(graph_results = graph_memories.len(), "[RECALL] Graph content retrieved");
        }
    }

    // ── Step 3: RRF Fusion + Scoring ──
    let total_candidates = search.len() + bm25_results.len() + seen_ids.len();

    // Build RRF score map from vector + BM25 rankings
    let mut rrf_scores: HashMap<[u8; 32], f64> = HashMap::new();

    for (rank, sr) in search.iter().enumerate() {
        *rrf_scores.entry(sr.record_id).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
    }

    for (rank, (source_type, source_id, _bm25_score)) in bm25_results.iter().enumerate() {
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

    let bm25_only_ids: Vec<[u8; 32]> = rrf_scores.keys()
        .filter(|rid| !search.iter().any(|sr| sr.record_id == **rid))
        .cloned()
        .collect();

    let max_degree = { let c = state.storage.conn_lock().await; graph::get_max_degree(&c, &owner) };
    let time_hint_tuple = rb.time_hint.as_ref().map(|th| (th.start, th.end));

    // Score vector search candidates
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

            // v2.4.0+BM25: RRF boost for records in both vector + BM25
            let rrf_boost = rrf_scores.get(&sr.record_id).copied().unwrap_or(0.0) * RRF_SCALE;
            scored.push((record, final_score + rrf_boost));
        }
    }

    // BM25-only candidates (not in vector search)
    for rid in &bm25_only_ids {
        if seen_ids.contains(rid) { continue; }
        if let Some(record) = state.storage.get(rid).await {
            if !record.is_active() || record.owner != owner { continue; }

            let rrf = rrf_scores.get(rid).copied().unwrap_or(0.0);
            let base_score = compute_recall_score(0.5, record.timestamp, now, record.access_count, record.layer);
            scored.push((record, base_score + rrf * RRF_SCALE));
        }
    }

    // Fallback: if no search results, use recent records
    if search.is_empty() && bm25_results.is_empty() {
        let recent = state.storage.get_active_records(&owner, layer_filter, top_k).await;
        for r in recent {
            if seen_ids.contains(&r.record_id) { continue; }
            scored.push((r.clone(), compute_recall_score(1.0, r.timestamp, now, r.access_count, r.layer)));
        }
    }

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 4: Token budget + graph memories injection ──
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

    // Inject graph-derived memories
    for gm in graph_memories {
        let tokens = estimate_tokens(&gm.content);
        if total_tokens + tokens > rb.token_budget && !memories.is_empty() { break; }
        total_tokens += tokens;
        memories.push(gm);
    }

    // Re-sort after injection and truncate
    memories.sort_unstable_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    memories.truncate(top_k);

    // Async graph co-occurrence update
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
