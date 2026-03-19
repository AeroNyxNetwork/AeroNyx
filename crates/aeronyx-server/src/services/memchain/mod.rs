// ============================================
// File: crates/aeronyx-server/src/services/memchain/mod.rs
// ============================================
//! # MemChain Storage Engine
//!
//! ## Submodules
//! - [`storage`] + [`storage_crypto`] + [`storage_ops`]: SQLite persistence (core/legacy)
//! - [`storage_graph`]: Cognitive graph CRUD (Episodes/Entities/Edges/Communities/Projects/Sessions/Artifacts)
//! - [`storage_miner`]: Miner step support + EntityTimeline (Steps 7/9/10/11)
//! - [`storage_fts`]: FTS5 BM25 full-text search + snippet highlight
//! - [`vector`]: Partitioned vector index for cosine similarity search
//! - [`mvf`]: Multi-Variate Feature scoring (10-dim with φ₉ since v2.4.0)
//! - [`graph`]: Co-occurrence graph + BFS traversal + community detection (v2.4.0)
//! - [`embed`]: Local MiniLM/Gemma embedding inference (ort + tokenizers)
//! - [`ner`]: Local GLiNER NER inference (ort + tokenizers) (v2.4.0)
//! - [`query_analyzer`]: Query analysis — GLiNER entity detection + regex + classification (v2.4.0)
//! - [`quantize`]: Scalar Quantization float32→uint8 (v2.4.0)
//! - [`reranker`]: Cross-encoder reranking ms-marco-MiniLM-L-6-v2 (v2.4.0+Reranker)
//! - [`mempool`]: In-memory Fact buffer (legacy/deprecated)
//! - [`aof`]: Append-Only File persistence (legacy/deprecated)
//!
//! ## Storage Split History
//! v2.2.0: storage.rs split into 3 files:
//!   - storage.rs       — struct, open, schema, migration, core CRUD, LRU
//!   - storage_crypto.rs — derive_record_key, derive_rawlog_key, encrypt/decrypt
//!   - storage_ops.rs   — rawlog, feedback, chain state, stats, miner base, overview
//!
//! v2.4.0+Search: storage_ops.rs split further into 3 files:
//!   - storage_ops.rs   — (unchanged) core/legacy ops
//!   - storage_graph.rs — cognitive graph CRUD (Episodes/Entities/Edges/etc.)
//!   - storage_miner.rs — Miner step support + EntityTimelineEntry + get_entity_timeline
//!
//! External API unchanged — all types and functions re-exported below.
//!
//! ⚠️ Important Note for Next Developer:
//! - storage_graph.rs and storage_miner.rs use `impl MemoryStorage` extension blocks,
//!   same pattern as storage_ops.rs. Rust allows multiple impl blocks across files
//!   within the same crate — no trait or wrapper needed.
//! - When adding a new storage submodule, declare it here AND add re-exports below.
//! - Re-export order matters for readability: storage types first, then engines, then legacy.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain storage engine
//! v2.1.0 - Added storage, vector, mvf, graph modules
//! v2.1.0+Embed - Added embed module
//! v2.2.0 - 🌟 Split storage into 3 files; added storage_ops (overview, get_embedding_model)
//! v2.4.0-GraphCognition - 🌟 Added ner, query_analyzer, quantize modules + re-exports
//! v2.4.0+Reranker - 🌟 Added reranker module + RerankerEngine re-export
//! v2.4.0+Search - 🌟 Split storage_ops into storage_graph + storage_miner.
//!   Added storage_graph, storage_miner module declarations and re-exports:
//!   EntityRow, KnowledgeEdgeRow, SessionRow, CommunityRow, ProjectRow, ArtifactRow,
//!   GraphStats, EntityTimelineEntry.

// ── Storage engine (split across multiple files) ──
pub mod storage;
pub mod storage_crypto;
pub mod storage_ops;
// v2.4.0+Search: Cognitive graph CRUD (split from storage_ops.rs)
pub mod storage_graph;
// v2.4.0+Search: Miner step support + entity timeline (split from storage_ops.rs)
pub mod storage_miner;
// v2.4.0: FTS5 BM25 full-text search
pub mod storage_fts;

// ── Cognitive engine ──
pub mod vector;
pub mod mvf;
pub mod graph;

// ── Local inference engines ──
pub mod embed;
// v2.4.0-GraphCognition: GLiNER ONNX NER engine
pub mod ner;
// v2.4.0-GraphCognition: Query analyzer
pub mod query_analyzer;
// v2.4.0-GraphCognition: Scalar quantization
pub mod quantize;
// v2.4.0+Reranker: Cross-encoder reranker
pub mod reranker;

// ── Legacy engine (deprecated) ──
pub mod aof;
pub mod mempool;

// ============================================
// Re-exports
// ============================================

// ── Storage core ──
pub use storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow};
pub use storage_crypto::{derive_record_key, derive_rawlog_key, decrypt_rawlog_content_pub};
pub use storage_ops::{OverviewRecord, OverviewData};

// ── Storage graph types (v2.4.0+Search) ──
// These types were previously inline in storage_ops.rs.
// Re-exported here so all callers (api handlers, recall_handler, etc.) are unaffected.
pub use storage_graph::{
    EntityRow,
    KnowledgeEdgeRow,
    SessionRow,
    CommunityRow,
    ProjectRow,
    ArtifactRow,
    GraphStats,
};

// ── Storage miner types (v2.4.0+Search) ──
pub use storage_miner::EntityTimelineEntry;

// ── Vector ──
pub use vector::{
    VectorIndex, SearchResult, DedupResult,
    cosine_similarity, compute_recall_score,
    dedup_threshold_for_layer, EPISODE_DEDUP_WINDOW_SECS,
};

// ── Local embedding engine ──
pub use embed::EmbedEngine;

// ── NER engine (v2.4.0-GraphCognition) ──
pub use ner::{NerEngine, DetectedEntity};

// ── Query analyzer (v2.4.0-GraphCognition) ──
pub use query_analyzer::{analyze_query, QueryAnalysis, QueryType, MatchedEntity};

// ── Scalar quantization (v2.4.0-GraphCognition) ──
pub use quantize::ScalarQuantizer;

// ── Cross-encoder reranker (v2.4.0+Reranker) ──
pub use reranker::RerankerEngine;

// ── Legacy ──
pub use aof::AofWriter;
pub use mempool::MemPool;
