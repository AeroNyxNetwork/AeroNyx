// ============================================
// File: crates/aeronyx-server/src/services/memchain/mod.rs
// ============================================
//! # MemChain Storage Engine
//!
//! ## Submodules
//! - [`storage`] + [`storage_crypto`] + [`storage_ops`]: SQLite persistence (split for maintainability)
//! - [`vector`]: Partitioned vector index for cosine similarity search
//! - [`mvf`]: Multi-Variate Feature scoring (9-dim SGD)
//! - [`graph`]: Co-occurrence graph for memory relationships
//! - [`embed`]: Local MiniLM embedding inference (ort + tokenizers)
//! - [`mempool`]: In-memory Fact buffer (legacy/deprecated)
//! - [`aof`]: Append-Only File persistence (legacy/deprecated)
//!
//! ## v2.2.0 Split
//! storage.rs was split into 3 files:
//! - storage.rs — struct, open, schema, migration, core CRUD, LRU
//! - storage_crypto.rs — derive_record_key, derive_rawlog_key, encrypt/decrypt
//! - storage_ops.rs — rawlog, feedback, chain state, stats, miner, overview
//!
//! External API unchanged — all types and functions re-exported below.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain storage engine
//! v2.1.0 - Added storage, vector, mvf, graph modules
//! v2.1.0+Embed - Added embed module
//! v2.2.0 - 🌟 Split storage into 3 files; added storage_ops (overview, get_embedding_model)

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

// Legacy engine (deprecated)
pub mod aof;
pub mod mempool;

// Re-exports: Storage (combined from all 3 files)
pub use storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow};
pub use storage_crypto::{derive_record_key, derive_rawlog_key, decrypt_rawlog_content_pub};
pub use storage_ops::{OverviewRecord, OverviewData};

// Re-exports: Vector
pub use vector::{
    VectorIndex, SearchResult, DedupResult,
    cosine_similarity, compute_recall_score,
    dedup_threshold_for_layer, EPISODE_DEDUP_WINDOW_SECS,
};

// Re-exports: Embed
pub use embed::EmbedEngine;

// Re-exports: Legacy
pub use aof::AofWriter;
pub use mempool::MemPool;
