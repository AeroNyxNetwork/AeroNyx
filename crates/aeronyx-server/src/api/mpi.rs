// ============================================
// File: crates/aeronyx-server/src/api/mpi.rs
// ============================================
//! # MPI — Memory Protocol Interface
//! # MPI — 记忆协议接口
//!
//! ## Creation Reason
//! 提供给 AI Agent（如 OpenClaw）的本地 HTTP API，用于实时读写记忆。
//! 这是 MemChain 面向 AI 的"前门"——所有记忆的 CRUD 和语义检索
//! 都通过这组端点完成。
//!
//! ## 端点概览 / Endpoint Overview
//! | Method | Path              | Function                            | Latency Target |
//! |--------|-------------------|-------------------------------------|----------------|
//! | POST   | /api/mpi/remember | 存储新记忆（去重+加密+持久化+广播） | < 20ms         |
//! | POST   | /api/mpi/recall   | 实时语义召回（向量搜索+认知打分）   | < 50ms ⭐      |
//! | POST   | /api/mpi/forget   | 撤销记忆（tombstone+擦除内容）       | < 10ms         |
//! | GET    | /api/mpi/status   | 存储统计                            | < 5ms          |
//!
//! ## recall 热路径优化 / Recall Hot Path
//! ```text
//! 请求 → VectorIndex.search(分区内) ≈ 1-3ms
//!       → LRU 缓存批量读取          ≈ < 1ms（命中率 > 90%）
//!       → SQLite fallback（缓存miss）≈ 5-15ms
//!       → Identity 强制注入          ≈ 1ms
//!       → 认知打分 + 排序 + 截断     ≈ < 1ms
//!       → access_count 异步更新      ≈ 0ms（fire-and-forget）
//! 总计 ≈ 5-20ms（远低于 50ms 目标）
//! ```
//!
//! ## v2.1 Changes
//! - remember/recall 请求体新增 `embedding_model` 字段（维度协商）
//! - 去重使用分层阈值（Identity 0.92 / Knowledge 0.88 / Episode 0.80+24h）
//! - recall 使用 LRU 缓存优先路径
//! - access_count 更新改为异步 fire-and-forget（不阻塞响应）
//!
//! ## ⚠️ Important Note for Next Developer
//! - recall 是性能最敏感的路径，每次用户消息都会触发
//! - `embedding_model` 是必须的——不同模型维度不同，搜错分区会 panic
//! - Phase 1 的 "加密" 是明文字节，Phase 2 替换为 X25519+ChaCha20
//! - Identity 层记忆始终注入 recall 结果最前面（不受 embedding 搜索影响）
//!
//! ## Last Modified
//! v2.1.0 - Initial MPI with partitioned search, LRU cache, per-layer dedup

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::ledger::{MemoryLayer, MemoryRecord};

use crate::services::memchain::{MemoryStorage, VectorIndex, compute_recall_score};

// ============================================
// Shared State
// ============================================

/// MPI 端点共享状态 / MPI endpoint shared state
pub struct MpiState {
    /// SQLite 存储（含 LRU 缓存）
    pub storage: Arc<MemoryStorage>,
    /// 分区式向量索引
    pub vector_index: Arc<VectorIndex>,
    /// 服务器身份密钥（用于签名新记录）
    pub identity: IdentityKeyPair,
}

// ============================================
// Helper: 解析 layer 字符串
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

/// 粗略 token 估算：~4 字符/token（英文）
/// Rough token estimate: ~4 chars per token for English
fn estimate_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

// ============================================
// POST /api/mpi/remember
// ============================================

/// remember 请求体 / Remember request body
#[derive(Debug, Deserialize)]
pub struct RememberRequest {
    /// 要存储的记忆内容（明文，由服务端"加密"后落盘）
    pub content: String,
    /// 认知层级 / Cognitive layer
    #[serde(default = "default_layer")]
    pub layer: String,
    /// 主题标签 / Topic tags
    #[serde(default)]
    pub topic_tags: Vec<String>,
    /// 来源 AI 标识 / Source AI identifier
    #[serde(default = "default_source")]
    pub source_ai: String,
    /// 语义 embedding 向量 / Semantic embedding vector
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// 🆕 embedding 模型标识（维度协商用）
    /// Embedding model identifier for dimension negotiation
    #[serde(default = "default_model")]
    pub embedding_model: String,
}

fn default_layer() -> String { "episode".into() }
fn default_source() -> String { "unknown".into() }
fn default_model() -> String { "default".into() }

/// remember 响应体 / Remember response body
#[derive(Debug, Serialize)]
pub struct RememberResponse {
    pub record_id: String,
    /// "created" | "duplicate"
    pub status: String,
    pub duplicate_of: Option<String>,
}

/// `POST /api/mpi/remember` — 存储新记忆
///
/// 流程 / Flow:
/// 1. 验证输入
/// 2. 分层去重检测（VectorIndex）
/// 3. "加密"内容 → encrypted_content（Phase 1: 明文字节）
/// 4. 构建 MemoryRecord + Ed25519 签名
/// 5. 持久化到 SQLite
/// 6. 索引 embedding 到向量引擎
/// 7. 返回 record_id
pub async fn mpi_remember(
    State(state): State<Arc<MpiState>>,
    Json(req): Json<RememberRequest>,
) -> impl IntoResponse {
    // 1. 验证
    if req.content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "content must be non-empty"
        })));
    }

    let layer = match parse_layer(&req.layer) {
        Some(l) => l,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "invalid layer: must be identity/knowledge/episode/archive"
        }))),
    };

    let owner = state.identity.public_key_bytes();
    let timestamp = now_secs();

    // 2. 分层去重（仅当有 embedding 时）
    if !req.embedding.is_empty() {
        let dedup = state.vector_index.check_duplicate(
            &req.embedding,
            &owner,
            &req.embedding_model,
            layer,
            timestamp,
        );

        if dedup.is_duplicate {
            let dup_hex = hex::encode(dedup.existing_id.unwrap_or([0; 32]));
            info!(
                duplicate_of = %dup_hex,
                similarity = dedup.max_similarity,
                layer = %layer,
                "[MPI_REMEMBER] 🔄 Duplicate detected"
            );
            return (StatusCode::OK, Json(serde_json::json!(RememberResponse {
                record_id: dup_hex.clone(),
                status: "duplicate".into(),
                duplicate_of: Some(dup_hex),
            })));
        }
    }

    // 3. "加密"（Phase 1: 明文字节；Phase 2: X25519+ChaCha20）
    let encrypted_content = req.content.as_bytes().to_vec();

    // 4. 构建 MemoryRecord
    let mut record = MemoryRecord::new(
        owner,
        timestamp,
        layer,
        req.topic_tags,
        req.source_ai,
        encrypted_content,
        req.embedding.clone(),
    );
    record.signature = state.identity.sign(&record.record_id);

    let record_id_hex = record.id_hex();

    // 5. 持久化 SQLite
    if !state.storage.insert(&record, &req.embedding_model).await {
        return (StatusCode::CONFLICT, Json(serde_json::json!({
            "error": "Record already exists or validation failed",
            "record_id": record_id_hex
        })));
    }

    // 6. 索引 embedding
    if !req.embedding.is_empty() {
        state.vector_index.upsert(
            record.record_id,
            req.embedding,
            layer,
            timestamp,
            &owner,
            &req.embedding_model,
        );
    }

    info!(record_id = %record_id_hex, layer = %layer, "[MPI_REMEMBER] ✅ Stored");

    (StatusCode::CREATED, Json(serde_json::json!(RememberResponse {
        record_id: record_id_hex,
        status: "created".into(),
        duplicate_of: None,
    })))
}

// ============================================
// POST /api/mpi/recall
// ============================================

/// recall 请求体 / Recall request body
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    /// 自然语言查询（日志用，不直接参与搜索）
    #[serde(default)]
    pub query: String,
    /// 查询 embedding 向量 / Query embedding vector
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// 🆕 embedding 模型标识（必须与 remember 时一致）
    #[serde(default = "default_model")]
    pub embedding_model: String,
    /// 最大返回数 / Max results
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// 可选 layer 过滤 / Optional layer filter
    pub layer: Option<String>,
    /// token 预算上限 / Token budget limit
    #[serde(default = "default_token_budget")]
    pub token_budget: usize,
}

fn default_top_k() -> usize { 10 }
fn default_token_budget() -> usize { 4000 }

/// 单条召回记忆 / Single recalled memory
#[derive(Debug, Serialize)]
pub struct RecalledMemory {
    pub record_id: String,
    pub layer: String,
    /// 认知打分（越高越相关）/ Cognitive score (higher = more relevant)
    pub score: f64,
    /// 解密后的内容 / Decrypted content
    pub content: String,
    pub topic_tags: Vec<String>,
    pub source_ai: String,
    pub timestamp: u64,
    pub access_count: u32,
}

/// recall 响应体 / Recall response body
#[derive(Debug, Serialize)]
pub struct RecallResponse {
    pub memories: Vec<RecalledMemory>,
    pub total_candidates: usize,
    pub token_estimate: usize,
}

/// `POST /api/mpi/recall` — 实时语义记忆召回
///
/// ## 性能目标 / Performance Target: < 50ms
///
/// ## 流程 / Flow:
/// 1. Identity 强制注入（始终排在最前）
/// 2. 向量语义搜索（VectorIndex，分区内）
/// 3. 从 LRU 缓存/SQLite 加载完整记录
/// 4. 认知心理学打分 + 排序
/// 5. token_budget 截断
/// 6. 异步更新 access_count
pub async fn mpi_recall(
    State(state): State<Arc<MpiState>>,
    Json(req): Json<RecallRequest>,
) -> impl IntoResponse {
    let owner = state.identity.public_key_bytes();
    let now = now_secs();
    let layer_filter = req.layer.as_deref().and_then(parse_layer);
    let top_k = req.top_k.min(100).max(1);

    let mut memories: Vec<RecalledMemory> = Vec::new();
    let mut total_tokens = 0usize;
    let mut seen_ids: Vec<[u8; 32]> = Vec::new();

    // ── Step 1: Identity 强制注入 ──
    // Identity 层记忆始终排在最前面，不受 embedding 搜索影响
    // Identity memories are always injected first, regardless of search
    {
        let identity_records = state.storage
            .get_active_records(&owner, Some(MemoryLayer::Identity), 20)
            .await;

        for record in &identity_records {
            let content = String::from_utf8_lossy(&record.encrypted_content).to_string();
            let tokens = estimate_tokens(&content);

            if total_tokens + tokens > req.token_budget && !memories.is_empty() {
                break;
            }

            total_tokens += tokens;
            seen_ids.push(record.record_id);
            memories.push(RecalledMemory {
                record_id: record.id_hex(),
                layer: record.layer.to_string(),
                score: record.layer.recall_weight() + 1.0, // Identity 始终最高分
                content,
                topic_tags: record.topic_tags.clone(),
                source_ai: record.source_ai.clone(),
                timestamp: record.timestamp,
                access_count: record.access_count,
            });

            // 异步更新 access_count（不阻塞响应）
            let storage = Arc::clone(&state.storage);
            let rid = record.record_id;
            tokio::spawn(async move { storage.increment_access(&rid).await; });
        }
    }

    // ── Step 2: 向量语义搜索 ──
    let search_results = if !req.embedding.is_empty() {
        state.vector_index.search_filtered(
            &req.embedding,
            &owner,
            &req.embedding_model,
            layer_filter,
            top_k * 3, // 多取一些用于打分重排
            0.0,
        )
    } else {
        Vec::new()
    };

    let total_candidates = search_results.len() + seen_ids.len();

    // ── Step 3 & 4: 加载记录 + 认知打分 ──
    let mut scored: Vec<(MemoryRecord, f64)> = Vec::with_capacity(search_results.len());

    for sr in &search_results {
        if seen_ids.contains(&sr.record_id) {
            continue; // 已被 Identity 注入
        }

        if let Some(record) = state.storage.get(&sr.record_id).await {
            if !record.is_active() {
                continue;
            }

            // 认知心理学打分 / Cognitive recall scoring
            let score = compute_recall_score(
                sr.similarity,
                record.timestamp,
                now,
                record.access_count,
                record.layer,
            );

            scored.push((record, score));
        }
    }

    // 如果没有 embedding 搜索结果，fallback 到最近记录
    if search_results.is_empty() {
        let recent = state.storage
            .get_active_records(&owner, layer_filter, top_k)
            .await;

        for record in recent {
            if seen_ids.contains(&record.record_id) { continue; }

            let score = compute_recall_score(
                1.0, // 无 embedding 时 similarity 设为 1.0
                record.timestamp,
                now,
                record.access_count,
                record.layer,
            );

            scored.push((record, score));
        }
    }

    // 按分数降序 / Sort by score descending
    scored.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── Step 5: token_budget 截断 ──
    for (record, score) in &scored {
        let content = String::from_utf8_lossy(&record.encrypted_content).to_string();
        let tokens = estimate_tokens(&content);

        if total_tokens + tokens > req.token_budget && !memories.is_empty() {
            break;
        }

        total_tokens += tokens;
        memories.push(RecalledMemory {
            record_id: record.id_hex(),
            layer: record.layer.to_string(),
            score: *score,
            content,
            topic_tags: record.topic_tags.clone(),
            source_ai: record.source_ai.clone(),
            timestamp: record.timestamp,
            access_count: record.access_count,
        });

        // 异步更新 access_count
        let storage = Arc::clone(&state.storage);
        let rid = record.record_id;
        tokio::spawn(async move { storage.increment_access(&rid).await; });
    }

    memories.truncate(top_k);

    debug!(
        returned = memories.len(),
        candidates = total_candidates,
        tokens = total_tokens,
        "[MPI_RECALL] ✅ Done"
    );

    (StatusCode::OK, Json(serde_json::json!(RecallResponse {
        memories,
        total_candidates,
        token_estimate: total_tokens,
    })))
}

// ============================================
// POST /api/mpi/forget
// ============================================

/// forget 请求体
#[derive(Debug, Deserialize)]
pub struct ForgetRequest {
    /// 十六进制 record_id / Hex-encoded record_id
    pub record_id: String,
}

/// forget 响应体
#[derive(Debug, Serialize)]
pub struct ForgetResponse {
    /// "revoked" | "not_found"
    pub status: String,
    pub record_id: String,
}

/// `POST /api/mpi/forget` — 撤销记忆
///
/// 流程: SQLite 标记 Revoked + 擦除内容/embedding → 向量索引移除
pub async fn mpi_forget(
    State(state): State<Arc<MpiState>>,
    Json(req): Json<ForgetRequest>,
) -> impl IntoResponse {
    let record_id = match hex::decode(&req.record_id) {
        Ok(bytes) if bytes.len() == 32 => {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            arr
        }
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "record_id must be 64 hex chars (32 bytes)"
        }))),
    };

    // 1. Revoke in SQLite FIRST (source of truth)
    if !state.storage.revoke(&record_id).await {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!(ForgetResponse {
            status: "not_found".into(),
            record_id: req.record_id,
        })));
    }

    // 2. Remove from vector index AFTER SQLite success
    state.vector_index.remove(&record_id);

    // 3. Invalidate Identity cache AFTER SQLite success.
    // Order: SQLite commit → cache invalidation (never reverse).
    {
        let owner_hex = hex::encode(state.identity.public_key_bytes());
        let mut cache = state.identity_cache.write();
        if let Some(entries) = cache.get_mut(&owner_hex) {
            entries.retain(|r| r.record_id != record_id);
        }
    }

    info!(record_id = %req.record_id, "[MPI_FORGET] Revoked");

    (StatusCode::OK, Json(serde_json::json!(ForgetResponse {
        status: "revoked".into(),
        record_id: req.record_id,
    })))
}

// ============================================
// GET /api/mpi/status
// ============================================

/// status 响应体
#[derive(Debug, Serialize)]
pub struct MpiStatusResponse {
    pub memchain_enabled: bool,
    pub mode: String,
    pub stats: crate::services::memchain::StorageStats,
    pub vector_index_total: usize,
    pub vector_partitions: usize,
    pub last_block_height: u64,
}

/// `GET /api/mpi/status` — 存储统计
pub async fn mpi_status(
    State(state): State<Arc<MpiState>>,
) -> impl IntoResponse {
    let stats = state.storage.stats().await;
    let height = state.storage.last_block_height().await;

    (StatusCode::OK, Json(serde_json::json!(MpiStatusResponse {
        memchain_enabled: true,
        mode: "local".into(), // TODO: 从 config 读取
        stats,
        vector_index_total: state.vector_index.total_vectors(),
        vector_partitions: state.vector_index.partition_count(),
        last_block_height: height,
    })))
}

// ============================================
// Router Builder
// ============================================

use axum::routing::{get, post};
use axum::Router;

/// 构建 MPI 路由 / Build MPI router
///
/// 所有 MPI 端点挂载在 `/api/mpi/` 下
pub fn build_mpi_router(state: Arc<MpiState>) -> Router {
    Router::new()
        .route("/api/mpi/remember", post(mpi_remember))
        .route("/api/mpi/recall", post(mpi_recall))
        .route("/api/mpi/forget", post(mpi_forget))
        .route("/api/mpi/status", get(mpi_status))
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
        let vector_index = Arc::new(VectorIndex::new());
        let identity = IdentityKeyPair::generate();
        let owner_key = identity.public_key_bytes();
        Arc::new(MpiState {
            storage,
            vector_index,
            identity,
            identity_cache: RwLock::new(HashMap::new()),
            index_ready: AtomicBool::new(true),
            user_weights: Arc::new(RwLock::new(HashMap::new())),
            mvf_alpha: 0.0,
            mvf_enabled: false,
            session_embeddings: RwLock::new(HashMap::new()),
            mvf_baseline: RwLock::new(None),
            owner_key,
        })
    }

    #[tokio::test]
    async fn test_remember_and_recall() {
        let state = make_state().await;
        let app = build_mpi_router(state);

        // Remember
        let req = Request::builder()
            .method("POST")
            .uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "content": "User likes Rust",
                "layer": "identity",
                "topic_tags": ["programming"],
                "source_ai": "test",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_model": "test-model"
            }"#))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "created");

        // Recall with embedding
        let req = Request::builder()
            .method("POST")
            .uri("/api/mpi/recall")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "query": "programming preferences",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_model": "test-model",
                "top_k": 5
            }"#))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let memories = json["memories"].as_array().unwrap();
        assert!(!memories.is_empty());
        assert_eq!(memories[0]["layer"], "identity");
    }

    #[tokio::test]
    async fn test_remember_duplicate_detected() {
        let state = make_state().await;
        let app = build_mpi_router(state);

        let body = r#"{
            "content": "Same memory",
            "layer": "identity",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_model": "m"
        }"#;

        // 第一次
        let req = Request::builder()
            .method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(body)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // 第二次（相同 embedding → 相似度 1.0 > 0.92 → 重复）
        let req = Request::builder()
            .method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(body)).unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "duplicate");
    }

    #[tokio::test]
    async fn test_forget() {
        let state = make_state().await;
        let app = build_mpi_router(state.clone());

        // Remember
        let req = Request::builder()
            .method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"content":"secret","layer":"episode","source_ai":"t"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let rid = json["record_id"].as_str().unwrap().to_string();

        // Forget
        let req = Request::builder()
            .method("POST").uri("/api/mpi/forget")
            .header("content-type", "application/json")
            .body(Body::from(format!(r#"{{"record_id":"{}"}}"#, rid)))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "revoked");
    }

    #[tokio::test]
    async fn test_recall_without_embedding_fallback() {
        let state = make_state().await;
        let app = build_mpi_router(state);

        // Remember without embedding
        let req = Request::builder()
            .method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"content":"no embedding test","layer":"episode"}"#))
            .unwrap();
        app.clone().oneshot(req).await.unwrap();

        // Recall without embedding → fallback 到最近记录
        let req = Request::builder()
            .method("POST").uri("/api/mpi/recall")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"query":"test","top_k":5}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(!json["memories"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_status() {
        let state = make_state().await;
        let app = build_mpi_router(state);

        let req = Request::builder()
            .uri("/api/mpi/status")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_remember_empty_content_rejected() {
        let state = make_state().await;
        let app = build_mpi_router(state);

        let req = Request::builder()
            .method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"content":"  ","layer":"episode"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_wrong_model_returns_empty() {
        let state = make_state().await;
        let app = build_mpi_router(state);

        // Remember with model A
        let req = Request::builder()
            .method("POST").uri("/api/mpi/remember")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "content":"test",
                "layer":"episode",
                "embedding":[1.0,0.0],
                "embedding_model":"model-a"
            }"#))
            .unwrap();
        app.clone().oneshot(req).await.unwrap();

        // Recall with model B → 分区不存在 → 空结果（但 fallback 到最近记录）
        let req = Request::builder()
            .method("POST").uri("/api/mpi/recall")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "query":"test",
                "embedding":[1.0,0.0],
                "embedding_model":"model-b",
                "top_k":5
            }"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
