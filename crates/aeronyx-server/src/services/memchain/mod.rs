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
//! - [`mempool`]: In-memory pool of recently received Facts,
//!   pending persistence and/or broadcast.
//! - [`aof`]: Append-Only File writer for durable single-file storage
//!   of Facts (`.memchain` ledger file).
//!
//! ## Architecture Position
//! ```text
//! aeronyx-server/src/services/
//!  ├── session.rs       ← existing
//!  ├── routing.rs       ← existing
//!  ├── ip_pool.rs       ← existing
//!  ├── handshake.rs     ← existing
//!  └── memchain/        ← 🌟 YOU ARE HERE
//!       ├── mod.rs
//!       ├── mempool.rs  ← in-memory Fact buffer
//!       └── aof.rs      ← append-only disk persistence
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
//! ```
//!
//! ## Design Principles
//! - **No external databases** — single `.memchain` file, append only.
//! - **Thread-safe** — `MemPool` uses `DashMap` for lock-free reads.
//! - **Async I/O** — `AofWriter` uses `tokio::fs` for non-blocking writes.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Never delete or overwrite entries in the AOF file.
//! - `MemPool` is the single source of truth for "current session" facts;
//!   on restart, facts are replayed from the AOF file.
//! - Future modules (`index.rs`, `block.rs`) will be added here for
//!   Miner / Checkpoint and secondary indexing.
//!
//! ## Last Modified
//! v0.2.0 - Initial MemChain storage engine

pub mod aof;
pub mod mempool;

// Re-export primary types
pub use aof::AofWriter;
pub use mempool::MemPool;
