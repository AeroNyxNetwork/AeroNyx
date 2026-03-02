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
//! ## Modification Reason
//! - 🌟 v0.4.0: `start_api_server` now accepts `SessionManager`,
//!   `UdpTransport`, `MemChainConfig`, and `DefaultTransportCrypto`
//!   to support P2P broadcast from the API layer and the new
//!   `POST /api/sync` endpoint.
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
//!     │ HTTP POST /api/fact   — write a new memory (+ P2P broadcast)
//!     │ HTTP GET  /api/facts  — read recent memories
//!     │ HTTP GET  /api/status — check MemChain health
//!     │ HTTP POST /api/sync   — 🌟 trigger P2P catch-up
//!     ▼
//! api/local.rs  (axum, 127.0.0.1:8421)
//!     │
//!     ├─► MemPool.add_fact()       — store in memory
//!     ├─► AofWriter.append_fact()  — persist to disk
//!     ├─► IdentityKeyPair.sign()   — Ed25519 signature
//!     └─► UdpTransport.send()      — 🌟 P2P broadcast / sync
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - API is loopback-only by default. Binding to 0.0.0.0 is allowed
//!   but generates a warning in config validation.
//! - All routes are prefixed with `/api/`.
//! - State is shared via `Arc` and injected through `axum::extract::State`.
//! - The API task is spawned in `server.rs` and respects shutdown signals.
//! - P2P broadcast happens async (tokio::spawn) so it never blocks the
//!   HTTP response.
//!
//! ## Last Modified
//! v0.3.0 - 🌟 Initial Agent API for MemChain Phase 1
//! v0.4.0 - 🌟 Extended for Phase 3: P2P broadcast + POST /api/sync

pub mod local;
pub use local::start_api_server;
