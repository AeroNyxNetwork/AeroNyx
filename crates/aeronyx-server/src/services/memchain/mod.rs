// ============================================
// File: crates/aeronyx-server/src/services/memchain/mod.rs
// ============================================
//! # MemChain Storage Engine
//!
//! ## Creation Reason
//! Provides the in-memory and on-disk storage layer for the MemChain
//! distributed AI memory ledger running inside AeroNyx.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`storage`]: SQLite WAL + LRU persistent storage (primary engine)
//! - [`vector`]: Partitioned vector index for cosine similarity search
//! - [`mvf`]: Multi-Variate Feature scoring (9-dim SGD)
//! - [`graph`]: Co-occurrence graph for memory relationships
//! - [`embed`]: Local MiniLM embedding inference (ort + tokenizers)
//! - [`mempool`]: In-memory pool of recently received Facts (legacy/deprecated)
//! - [`aof`]: Append-Only File writer for durable Fact storage (legacy/deprecated)
//!
//! ## Architecture Position
//! ```text
//! aeronyx-server/src/services/
//!  ├── session.rs       ← existing
//!  ├── routing.rs       ← existing
//!  ├── ip_pool.rs       ← existing
//!  ├── handshake.rs     ← existing
//!  └── memchain/        ← 🌟 YOU ARE HERE
//!       ├── mod.rs      ← this file
//!       ├── storage.rs  ← SQLite WAL + LRU + Schema v4 + encryption
//!       ├── vector.rs   ← partitioned vector index + scoring
//!       ├── mvf.rs      ← MVF 9-dim features + SGD
//!       ├── graph.rs    ← co-occurrence graph
//!       ├── embed.rs    ← 🌟 NEW: local MiniLM embedding (ort + tokenizers)
//!       ├── mempool.rs  ← ⚠️ deprecated (P2P compat)
//!       └── aof.rs      ← ⚠️ deprecated (P2P compat)
//! ```
//!
//! ## Data Flow
//! ```text
//! Incoming MemChain packet (from packet.rs)
//!   │
//!   ▼
//! MemPool.add_fact(fact)    ← validates hash & signature
//!   │
//!   ├─► kept in DashMap for fast query
//!   │
//!   └─► AofWriter.append_fact(fact)  ← async flush to disk
//!
//! POST /api/mpi/embed (from AI agent or Miner)
//!   │
//!   ▼
//! EmbedEngine.embed_batch(texts)
//!   │
//!   ├─► tokenize (HuggingFace WordPiece)
//!   ├─► ort::Session.run (ONNX Runtime, CPU)
//!   ├─► mean pooling + L2 normalize
//!   └─► return Vec<Vec<f32>> (384-dim)
//! ```
//!
//! ## Design Principles
//! - **No external databases** — single `.memchain` file, append only (legacy).
//! - **SQLite primary** — WAL mode, LRU cache, Schema v4 with encryption.
//! - **Thread-safe** — `MemPool` uses `DashMap`; `EmbedEngine` is Send+Sync.
//! - **Async I/O** — `AofWriter` uses `tokio::fs` for non-blocking writes.
//! - **Self-contained** — EmbedEngine runs locally, no external API dependency.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Never delete or overwrite entries in the AOF file.
//! - `MemPool` is the single source of truth for "current session" facts;
//!   on restart, facts are replayed from the AOF file.
//! - EmbedEngine is optional — if model files are missing, server starts
//!   without local embedding (fallback to OpenClaw Gateway).
//! - Future modules (`index.rs`, `block.rs`) will be added here for
//!   Miner / Checkpoint and secondary indexing.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain storage engine
//! v2.1.0 - Added storage, vector, mvf, graph modules
//! v2.1.0+Embed - 🌟 Added embed module for local MiniLM inference

// New primary engine
pub mod storage;
pub mod vector;
pub mod mvf;
pub mod graph;

// Local embedding engine (v2.1.0+Embed)
pub mod embed;

// Legacy engine (deprecated)
pub mod aof;
pub mod mempool;

// Re-exports: New
pub use storage::{MemoryStorage, StorageStats, LayerCounts, RawLogRow, derive_rawlog_key};
pub use vector::{
    VectorIndex, SearchResult, DedupResult,
    cosine_similarity, compute_recall_score,
    dedup_threshold_for_layer, EPISODE_DEDUP_WINDOW_SECS,
};
pub use embed::EmbedEngine;

// Re-exports: Legacy
pub use aof::AofWriter;
pub use mempool::MemPool;
