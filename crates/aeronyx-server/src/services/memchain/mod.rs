// ============================================
// File: crates/aeronyx-server/src/services/memchain/mod.rs
// ============================================
//! # MemChain Storage Engine
//!
//! ## Submodules
//! - [`storage`] + [`storage_crypto`] + [`storage_ops`]: SQLite persistence (split for maintainability)
//! - [`vector`]: Partitioned vector index for cosine similarity search
//! - [`mvf`]: Multi-Variate Feature scoring (9-dim SGD → 10-dim with φ₉ in v2.4.0)
//! - [`graph`]: Co-occurrence graph for memory relationships + BFS traversal (v2.4.0)
//! - [`embed`]: Local MiniLM embedding inference (ort + tokenizers)
//! - [`ner`]: Local GLiNER NER inference (ort + tokenizers) (v2.4.0)
//! - [`mempool`]: In-memory Fact buffer (legacy/deprecated)
//! - [`aof`]: Append-Only File persistence (legacy/deprecated)
//!
//! ## v2.2.0 Split
//! storage.rs was split into 3 files:
//! - storage.rs — struct, open, schema, migration, core CRUD, LRU
//! - storage_crypto.rs — derive_record_key, derive_rawlog_key, encrypt/decrypt
//! - storage_ops.rs — rawlog, feedback, chain state, stats, miner, overview
//!
//! ## v2.4.0-GraphCognition Additions
//! - ner.rs — GLiNER ONNX local NER engine (sister module to embed.rs)
//!   Zero-shot entity detection + relation extraction for cognitive graph pipeline.
//!   Shares ort load-dynamic mechanism with embed.rs.
//!
//! Future v2.4.0 additions (not yet created):
//! - query_analyzer.rs — Query analysis (GLiNER entity detection + regex + classification)
//! - quantize.rs — Scalar Quantization (float32 → uint8)
//!
//! External API unchanged — all types and functions re-exported below.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain storage engine
//! v2.1.0 - Added storage, vector, mvf, graph modules
//! v2.1.0+Embed - Added embed module
//! v2.2.0 - 🌟 Split storage into 3 files; added storage_ops (overview, get_embedding_model)
//! v2.4.0-GraphCognition - 🌟 Added ner module (GLiNER ONNX local NER engine)

// Storage engine (split into 3 files)
pub mod storage;
pub mod storage_crypto;
pub mod storage_ops;

// Cognitive engine
pub mod vector;
pub mod mvf;
pub mod graph;

// Local embedding engine
pub mod embed;

// Local NER engine (v2.4.0-GraphCognition)
pub mod ner;

// Query analyzer (v2.4.0-GraphCognition)
pub mod query_analyzer;

// Scalar quantization (v2.4.0-GraphCognition)
pub mod quantize;

// Legacy engine (deprecated)
pub mod aof;
pub mod mempool;

// ── Re-exports: Storage (combined from all 3 files) ──
pub use storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow};
pub use storage_crypto::{derive_record_key, derive_rawlog_key, decrypt_rawlog_content_pub};
pub use storage_ops::{OverviewRecord, OverviewData};

// ── Re-exports: Vector ──
pub use vector::{
    VectorIndex, SearchResult, DedupResult,
    cosine_similarity, compute_recall_score,
    dedup_threshold_for_layer, EPISODE_DEDUP_WINDOW_SECS,
};

// ── Re-exports: Embed ──
pub use embed::EmbedEngine;

// ── Re-exports: NER (v2.4.0-GraphCognition) ──
pub use ner::{NerEngine, DetectedEntity};

// ── Re-exports: Query Analyzer (v2.4.0-GraphCognition) ──
pub use query_analyzer::{analyze_query, QueryAnalysis, QueryType, MatchedEntity};

// ── Re-exports: Scalar Quantization (v2.4.0-GraphCognition) ──
pub use quantize::ScalarQuantizer;

// ── Re-exports: Legacy ──
pub use aof::AofWriter;
pub use mempool::MemPool;
