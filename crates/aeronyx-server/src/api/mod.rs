// ============================================
// File: crates/aeronyx-server/src/api/mod.rs
// ============================================
//! # MemChain Agent API
//!
//! ## Creation Reason
//! Provides a local HTTP API for AI Agents (e.g. OpenClaw) to read and
//! write memory Facts into the MemChain ledger. The API binds to
//! loopback by default (`127.0.0.1:8421`) and is NOT exposed to the
//! public network.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`local`]: Axum router, route handlers, shared state
//!
//! ## Architecture Position
//! ```text
//! AI Agent (OpenClaw / curl / Python)
//!     │
//!     │ HTTP POST /api/fact   — write a new memory
//!     │ HTTP GET  /api/facts  — read recent memories
//!     │ HTTP GET  /api/status — check MemChain health
//!     ▼
//! api/local.rs  (axum, 127.0.0.1:8421)
//!     │
//!     ├─► MemPool.add_fact()       — store in memory
//!     ├─► AofWriter.append_fact()  — persist to disk
//!     └─► IdentityKeyPair.sign()   — Ed25519 signature
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - API is loopback-only by default. Binding to 0.0.0.0 is allowed
//!   but generates a warning in config validation.
//! - All routes are prefixed with `/api/`.
//! - State is shared via `Arc` and injected through `axum::extract::State`.
//! - The API task is spawned in `server.rs` and respects shutdown signals.
//!
//! ## Last Modified
//! v0.3.0 - 🌟 Initial Agent API for MemChain Phase 1

pub mod local;

pub use local::start_api_server;
