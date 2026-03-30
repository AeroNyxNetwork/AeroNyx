// ============================================
// File: crates/aeronyx-server/src/api/recall_handler.rs
// ============================================
//! # POST /api/mpi/recall — Hybrid Retrieval Pipeline
//!
//! ## Pipeline
//! ```text
//! Step 1:      Query Analysis (GLiNER + regex + entity matching)
//! Step 2a:     Vector search
//! Step 2a-bis: BM25 FTS5 search
//! Step 2a-ter: BM25 entity/session direct injection
//! Step 2b:     Graph BFS
//! Step 2c:     Graph content retrieval
//! Step 3:      RRF fusion + MVF scoring
//! Step 3.5:    Cross-encoder rerank (v2.4.0+Reranker)
//! Step 4:      Token budget trimming
//! Step 4.1:    🆕 v2.5.3+Isolation: Context filter (project_id isolation)
//! Step 4.5:    Progressive mode branch (v2.4.0+Progressive)
//! ```
//!
//! ## Progressive Retrieval (v2.4.0+Progressive)
//! Pass 1: POST /recall { mode: "index" } → ~50 tokens/item
//! Pass 2: POST /recall/detail { record_ids: [...] } → full content
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
//! - mpi_recall_detail uses extract_owner() (same as all handlers).
//! - Synthetic IDs (graph_*, bm25_*) in index results are silently skipped
//!   by /recall/detail.
//! - Context filter (Step 4.1) calls get_active_records_by_context() which
//!   does a LEFT JOIN records → sessions. For records inserted via /remember
//!   directly (no session), project_id on the record row itself is checked.
//! - The core logic of this file cannot be deleted or significantly modified.
//!
//! ## Modification History
//! v2.4.0+BM25          - 🌟 Extracted from mpi_handlers.rs; BM25 + RRF fusion
//! v2.4.0+BM25-fix      - 🔧 BM25 entity/session direct injection (Step 2a-ter)
//! v2.4.0+Reranker      - 🌟 Step 3.5 cross-encoder rerank
//! v2.4.0+Progressive   - 🌟 mode="index" branch + mpi_recall_detail handler
//! v2.5.3+Isolation     - 🌟 Step 4.1 context filter; RecallRequest.context field
//!
//! ## Last Modified
//! v2.5.3+Isolation - 🌟 Step 4.1 context post-filter for memory isolation

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use axum::http::Request;
use serde::Deserialize;
use tracing::{debug, warn};

use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{compute_recall_score, cosine_similarity};
use crate::services::memchain::mvf;
use crate::services::memchain::graph;
use crate::services::memchain::query_analyzer::{self, QueryType};

use super::mpi::{
    MpiState, extract_owner, parse_layer, estimate_tokens, now_secs,
};

pub use super::mpi_handlers::{
    RecallRequest, RecallResponse, RecalledMemory,
    TimeHint, TimeRangeParam,
};
pub use crate::services::memchain::reranker::RERANK_TOP_N;

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

    // ── Step 1: Query Analysis ──
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

    // ── Step 2a: Vector search ──
    let idx_ready = state.index_ready.load(std::sync::atomic::Ordering::Relaxed);
    let search = if !rb.embedding.is_empty() && idx_ready {
        state.vector_index.search_filtered(&rb.embedding, &owner, &rb.embedding_model, layer_filter, top_k * 3, 0.0)
    } else { Vec::new() };

    // ── Step 2a-bis: BM25 FTS5 search ──
    let bm25_results = if !rb.query.is_empty() {
        state.storage.bm25_search(&rb.query, &owner, top_k * 3).await
    } else { Vec::new() };

    // ── Step 2a-ter: BM25 entity/session direct injection ──
    let mut bm25_direct_memories: Vec<RecalledMemory> = Vec::new();
    for (source_type, source_id, bm25_score) in &bm25_results {
        match source_type.as_str() {
            "entity" => {
                if let Some(entity) = state.storage.get_entity(source_id).await {
                    let edges = state.storage.get_edges_for_entity(source_id, &owner).await;
                    let mut parts: Vec<String> = vec![format!("{} ({})", entity.name, entity.entity_type)];
                    if let Some(ref desc) = entity.description {
                        if !desc.contains(&entity.name) { parts.push(desc.clone()); }
                    }
                    let mut edge_descs: Vec<String> = Vec::new();
                    for edge in edges.iter().take(5) {
                        let other_id = if edge.source_id == *source_id { &edge.target_id } else { &edge.source_id };
                        if let Some(other) = state.storage.get_entity(other_id).await {
                            edge_descs.push(if edge.source_id == *source_id {
                                format!("{} → {}", edge.relation_type, other.name)
                            } else {
                                format!("{} ← {}", other.name, edge.relation_type)
                            });
                        }
                    }
                    if !edge_descs.is_empty() { parts.push(format!("Relations: {}", edge_descs.join("; "))); }
                    bm25_direct_memories.push(RecalledMemory {
                        record_id: format!("bm25_entity_{}", source_id),
                        layer: "knowledge".into(), score: 1.5 + bm25_score.min(1.0),
                        content: parts.join(". "),
                        topic_tags: vec!["bm25_result".into(), "entity_knowledge".into()],
                        source_ai: "bm25-search".into(),
                        timestamp: entity.updated_at as u64,
                        access_count: entity.mention_count as u32, proactive: false,
                    });
                }
            }
            "session" => {
                if let Some(session) = state.storage.get_session(source_id, &owner).await {
                    if let Some(ref summary) = session.summary {
                        bm25_direct_memories.push(RecalledMemory {
                            record_id: format!("bm25_session_{}", source_id),
                            layer: "knowledge".into(), score: 1.8 + bm25_score.min(1.0),
                            content: format!("[Session: {}] {}", source_id, summary),
                            topic_tags: vec!["bm25_result".into(), "session_summary".into()],
                            source_ai: "bm25-search".into(),
                            timestamp: session.started_at as u64,
                            access_count: 0, proactive: false,
                        });
                    }
                }
            }
            "record" => {
                if let Ok(id_bytes) = hex::decode(source_id) {
                    if id_bytes.len() == 32 {
                        let mut rid = [0u8; 32]; rid.copy_from_slice(&id_bytes);
                        if state.storage.get(&rid).await.is_none() {
                            let content: Option<String> = {
                                let conn = state.storage.conn_lock().await;
                                conn.query_row(
                                    "SELECT content FROM fts_index WHERE source_type = 'record' AND source_id = ?1",
                                    rusqlite::params![source_id.as_str()], |row| row.get(0),
                                ).ok()
                            };
                            if let Some(text) = content {
                                bm25_direct_memories.push(RecalledMemory {
                                    record_id: format!("bm25_turn_{}", &source_id[..source_id.len().min(16)]),
                                    layer: "episode".into(), score: 1.2 + bm25_score.min(1.0),
                                    content: text,
                                    topic_tags: vec!["bm25_result".into(), "conversation_turn".into()],
                                    source_ai: "bm25-search".into(),
                                    timestamp: now, access_count: 0, proactive: false,
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
        debug!(bm25_direct = bm25_direct_memories.len(), "[RECALL] BM25 direct injection");
    }

    // ── Step 2b: Graph BFS ──
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

    // ── Step 2c: Graph content retrieval ──
    let mut graph_memories: Vec<RecalledMemory> = Vec::new();
    if !graph_traversed.is_empty() {
        let graph_entity_ids: Vec<String> = graph_traversed.keys().cloned().collect();

        let mut seen_episodes: HashSet<String> = HashSet::new();
        for eid in &graph_entity_ids {
            for (episode_id, _role) in state.storage.get_episodes_for_entity(eid).await {
                if seen_episodes.contains(&episode_id) { continue; }
                seen_episodes.insert(episode_id.clone());
                let session_id = if episode_id.starts_with("ep_") {
                    let rest = &episode_id[3..];
                    rest.rfind('_').map(|i| rest[..i].to_string())
                } else { None };
                if let Some(ref sid) = session_id {
                    if let Some(session) = state.storage.get_session(sid, &owner).await {
                        if let Some(ref summary) = session.summary {
                            let weight = graph_traversed.get(eid).copied().unwrap_or(0.5);
                            graph_memories.push(RecalledMemory {
                                record_id: format!("graph_session_{}", sid),
                                layer: "knowledge".to_string(), score: 2.0 + weight,
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

        for eid in &graph_entity_ids {
            if let Some(entity) = state.storage.get_entity(eid).await {
                let edges = state.storage.get_edges_for_entity(eid, &owner).await;
                let weight = graph_traversed.get(eid).copied().unwrap_or(0.5);
                let mut desc_parts: Vec<String> = vec![format!("{} ({})", entity.name, entity.entity_type)];
                if let Some(ref desc) = entity.description {
                    if !desc.contains(&entity.name) { desc_parts.push(desc.clone()); }
                }
                let mut edge_descs: Vec<String> = Vec::new();
                for edge in edges.iter().take(5) {
                    let other_id = if edge.source_id == *eid { &edge.target_id } else { &edge.source_id };
                    if let Some(other) = state.storage.get_entity(other_id).await {
                        edge_descs.push(if edge.source_id == *eid {
                            format!("{} → {}", edge.relation_type, other.name)
                        } else {
                            format!("{} ← {}", other.name, edge.relation_type)
                        });
                    }
                    if let Some(ref fact) = edge.fact_text { edge_descs.push(fact.clone()); }
                }
                if !edge_descs.is_empty() { desc_parts.push(format!("Relations: {}", edge_descs.join("; "))); }
                graph_memories.push(RecalledMemory {
                    record_id: format!("graph_entity_{}", eid),
                    layer: "knowledge".to_string(), score: 1.5 + weight,
                    content: desc_parts.join(". "),
                    topic_tags: vec!["graph_result".into(), "entity_knowledge".into()],
                    source_ai: "cognitive-graph".into(),
                    timestamp: entity.updated_at as u64,
                    access_count: entity.mention_count as u32, proactive: false,
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
                    let mut rid = [0u8; 32]; rid.copy_from_slice(&id_bytes);
                    *rrf_scores.entry(rid).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
                }
            }
        }
    }

    let bm25_only_ids: Vec<[u8; 32]> = rrf_scores.keys()
        .filter(|rid| !search.iter().any(|sr| sr.record_id == **rid))
        .cloned().collect();

    let max_degree = { let c = state.storage.conn_lock().await; graph::get_max_degree(&c, &owner) };
    let time_hint_tuple = rb.time_hint.as_ref().map(|th| (th.start, th.end));

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
                    .map(|(_, w)| *w as f32).unwrap_or(0.0)
            } else { 0.0 };
            let v_old_h = v_old + time_bonus;
            let final_score = if state.mvf_enabled {
                let gd = { let c = state.storage.conn_lock().await; graph::get_degree(&c, &record.record_id) };
                let cs = session_centroid.as_ref()
                    .filter(|c| record.has_embedding() && c.len() == record.embedding.len())
                    .map(|c| cosine_similarity(c, &record.embedding)).unwrap_or(0.0);
                let phi = mvf::compute_features(
                    sr.similarity, record.layer as u8, record.timestamp, now,
                    record.access_count, record.positive_feedback, record.negative_feedback,
                    record.has_conflict(), time_hint_tuple, cs, gd, max_degree, graph_weight,
                );
                let mut uw = { state.user_weights.read().get(&owner_hex).cloned().unwrap_or_else(mvf::default_weights) };
                let pn = mvf::normalize(&phi, &mut uw);
                let vm = mvf::compute_value(&uw, &pn);
                { state.user_weights.write().insert(owner_hex.clone(), uw); }
                mvf::fuse_scores(vm, v_old_h, state.mvf_alpha)
            } else {
                v_old_h + (graph_weight as f64 * 0.1)
            };
            let rrf_boost = rrf_scores.get(&sr.record_id).copied().unwrap_or(0.0) * RRF_SCALE;
            scored.push((record, final_score + rrf_boost));
        }
    }

    for rid in &bm25_only_ids {
        if seen_ids.contains(rid) { continue; }
        if let Some(record) = state.storage.get(rid).await {
            if !record.is_active() || record.owner != owner { continue; }
            let rrf = rrf_scores.get(rid).copied().unwrap_or(0.0);
            let base = compute_recall_score(0.5, record.timestamp, now, record.access_count, record.layer);
            scored.push((record, base + rrf * RRF_SCALE));
        }
    }

    if search.is_empty() && bm25_results.is_empty() {
        let recent = state.storage.get_active_records(&owner, layer_filter, top_k).await;
        for r in recent {
            if seen_ids.contains(&r.record_id) { continue; }
            scored.push((r.clone(), compute_recall_score(1.0, r.timestamp, now, r.access_count, r.layer)));
        }
    }

    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // ── Step 3.5: Cross-encoder rerank ──
    if let Some(ref reranker) = state.reranker_engine {
        if !rb.query.is_empty() && !scored.is_empty() {
            let rerank_n = scored.len().min(RERANK_TOP_N);
            let doc_texts: Vec<String> = scored[..rerank_n].iter()
                .map(|(r, _)| String::from_utf8_lossy(&r.encrypted_content).to_string())
                .collect();
            let doc_refs: Vec<&str> = doc_texts.iter().map(|s| s.as_str()).collect();

            match reranker.rerank_batch(&rb.query, &doc_refs) {
                Ok(reranked) => {
                    let blend_w = crate::services::memchain::reranker::RerankerEngine::blend_weight();
                    let mut new_scored: Vec<(MemoryRecord, f64)> = Vec::with_capacity(scored.len());
                    for rc in &reranked {
                        let (record, old_score) = &scored[rc.original_index];
                        let rrf_norm = (old_score / 5.0).min(1.0);
                        let blended = blend_w * rc.ce_score_normalized + (1.0 - blend_w) * rrf_norm;
                        new_scored.push((record.clone(), blended));
                    }
                    for i in rerank_n..scored.len() { new_scored.push(scored[i].clone()); }
                    new_scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    scored = new_scored;
                    debug!(reranked = reranked.len(), "[RECALL] Step 3.5 rerank complete");
                }
                Err(e) => { warn!(error = %e, "[RECALL] Reranker failed, using RRF"); }
            }
        }
    }

    // ── Step 4.1: Context filter (v2.5.3+Isolation) ───────────────────────
    // Applied AFTER scoring/reranking (post-filter).
    // Vector search has no project_id awareness; context isolation is
    // enforced here by looking up which records belong to the target context.
    //
    // Performance note: for large datasets, batch the lookup into a HashSet
    // of valid record_ids rather than per-record DB calls.
    let context_project_id: Option<String> = match rb.context.as_deref() {
        None | Some("all") | Some("") => None,
        Some(ctx) => Some(ctx.to_string()),
    };

    if let Some(ref pid) = context_project_id {
        // Load record_ids that belong to this project context.
        // get_active_records_by_context checks BOTH records.project_id
        // and records → sessions.project_id (via LEFT JOIN).
        let ctx_records = state.storage
            .get_active_records_by_context(&owner, pid, layer_filter, top_k * 10)
            .await;
        let ctx_ids: HashSet<[u8; 32]> =
            ctx_records.iter().map(|r| r.record_id).collect();

        let before = scored.len();
        scored.retain(|(r, _)| ctx_ids.contains(&r.record_id));

        debug!(
            context = %pid,
            before = before,
            after = scored.len(),
            "[RECALL] Step 4.1 context filter applied"
        );
    }
    // ─────────────────────────────────────────────────────────────────────

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

    // Graph and BM25 direct memories are NOT filtered by context —
    // they are synthesized/virtual records without a project_id.
    // If needed in future, filter by checking the underlying session's project_id.
    for gm in graph_memories {
        let tokens = estimate_tokens(&gm.content);
        if total_tokens + tokens > rb.token_budget && !memories.is_empty() { break; }
        total_tokens += tokens; memories.push(gm);
    }
    for bm in bm25_direct_memories {
        let tokens = estimate_tokens(&bm.content);
        if total_tokens + tokens > rb.token_budget && !memories.is_empty() { break; }
        total_tokens += tokens; memories.push(bm);
    }

    memories.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    memories.truncate(top_k);

    // Compute matched_json BEFORE index-mode branch (avoid borrow-after-move)
    let matched_json: Option<Vec<serde_json::Value>> = analysis.as_ref().map(|qa| {
        qa.matched_entities.iter().map(|e| serde_json::json!({
            "text": e.query_text, "label": e.label, "confidence": e.confidence,
            "entity_id": e.entity_id, "entity_type": e.entity_type,
        })).collect()
    });
    let query_type_str = format!("{:?}", query_type).to_lowercase();

    // ── Step 4.5: Progressive index mode ──
    if rb.mode == "index" {
        let index_memories: Vec<RecalledMemory> = memories.into_iter().map(|mut m| {
            if m.content.chars().count() > 80 {
                let byte_offset = m.content.char_indices().nth(80)
                    .map(|(i, _)| i).unwrap_or(m.content.len());
                m.content = format!("{}...", &m.content[..byte_offset]);
            }
            m
        }).collect();

        let token_estimate: usize = index_memories.iter()
            .map(|m| estimate_tokens(&m.content) + 30)
            .sum();

        return (StatusCode::OK, Json(serde_json::json!({
            "mode": "index",
            "memories": index_memories,
            "total_candidates": total_candidates,
            "token_estimate": token_estimate,
            "query_type": query_type_str,
            "matched_entities": matched_json,
            "hint": "Use POST /api/mpi/recall/detail with record_ids to fetch full content.",
        }))).into_response();
    }

    // ── Async co-occurrence update (full mode only) ──
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
        query_type: Some(query_type_str),
        matched_entities: matched_json,
    }))).into_response()
}

// ============================================
// POST /api/mpi/recall/detail
// ============================================

#[derive(Debug, serde::Deserialize)]
pub struct DetailRequest {
    pub record_ids: Vec<String>,
}

/// Fetch full content for selected memory IDs (progressive retrieval pass 2).
///
/// Synthetic IDs (graph_*, bm25_*) are silently skipped — no backing records.
/// Max 20 record_ids per request.
pub async fn mpi_recall_detail(
    State(state): State<Arc<MpiState>>,
    req: Request<axum::body::Body>,
) -> impl IntoResponse {
    let auth = extract_owner(&req).clone();
    let owner = auth.owner_bytes();

    let body_bytes = match axum::body::to_bytes(req.into_body(), 1024 * 1024).await {
        Ok(b) => b,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error":"failed to read body"}))).into_response(),
    };
    let dr: DetailRequest = match serde_json::from_slice(&body_bytes) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("invalid JSON: {}", e)
        }))).into_response(),
    };

    if dr.record_ids.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "record_ids array is empty"
        }))).into_response();
    }
    if dr.record_ids.len() > 20 {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "max 20 record_ids per request"
        }))).into_response();
    }

    let mut memories: Vec<RecalledMemory> = Vec::new();
    let mut total_tokens = 0usize;

    for rid_hex in &dr.record_ids {
        if rid_hex.starts_with("graph_") || rid_hex.starts_with("bm25_") { continue; }

        let rid = match hex::decode(rid_hex) {
            Ok(b) if b.len() == 32 => { let mut a = [0u8; 32]; a.copy_from_slice(&b); a }
            _ => continue,
        };

        if let Some(record) = state.storage.get(&rid).await {
            if !record.is_active() || record.owner != owner { continue; }
            let content = String::from_utf8_lossy(&record.encrypted_content).to_string();
            let tokens = estimate_tokens(&content);
            total_tokens += tokens;

            memories.push(RecalledMemory {
                record_id: rid_hex.clone(),
                layer: record.layer.to_string(),
                score: 0.0,
                content,
                topic_tags: record.topic_tags.clone(),
                source_ai: record.source_ai.clone(),
                timestamp: record.timestamp,
                access_count: record.access_count,
                proactive: false,
            });

            let st = Arc::clone(&state.storage);
            tokio::spawn(async move { st.increment_access(&rid).await; });
        }
    }

    (StatusCode::OK, Json(serde_json::json!({
        "memories": memories,
        "token_estimate": total_tokens,
    }))).into_response()
}
