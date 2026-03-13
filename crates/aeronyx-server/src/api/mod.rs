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
//! ## v2.4.0 File Split
//! mpi.rs was split into 3 files for maintainability:
//! - `mpi.rs` — MpiState, AuthenticatedOwner, auth middleware, router, helpers
//! - `mpi_handlers.rs` — Original 7 endpoint handlers (remember, recall, forget,
//!   status, embed, record, overview)
//! - `mpi_graph_handlers.rs` — v2.4.0 cognitive graph endpoints (11 new)
//!
//! External API is UNCHANGED — this file re-exports the same symbols:
//! `{build_mpi_router, MpiState, BaselineSnapshot}`
//!
//! server.rs, log_handler.rs, and ws_client.rs all import from
//! `crate::api::mpi::` — their code does NOT need to change.
//!
//! ## Submodules
//! - [`mpi`]: Core MPI types, auth, router (entry point)
//! - [`mpi_handlers`]: Original endpoint handlers (remember, recall, etc.)
//! - [`mpi_graph_handlers`]: v2.4.0 cognitive graph endpoints
//! - [`log_handler`]: /log endpoint with rule engine + entropy filter
//! - [`local`]: Legacy Axum router (deprecated)
//!
//! ⚠️ Important Note for Next Developer:
//! - When adding new ORIGINAL-style endpoints → add to mpi_handlers.rs
//! - When adding new GRAPH/COGNITIVE endpoints → add to mpi_graph_handlers.rs
//! - Register all routes in mpi.rs::build_mpi_router() regardless of which file
//!   the handler lives in
//! - Re-exports below MUST stay in sync — server.rs depends on them
//!
//! ## Last Modified
//! v0.3.0 - 🌟 Initial Agent API for MemChain Phase 1
//! v0.4.0 - 🌟 Extended for Phase 3: P2P broadcast + POST /api/sync
//! v2.4.0-GraphCognition - 🌟 Split mpi.rs into 3 files; added mpi_handlers
//!   and mpi_graph_handlers submodules

// ── Core MPI module (state, auth, router) ──
pub mod mpi;

// ── Handler modules (split from mpi.rs in v2.4.0) ──
pub mod mpi_handlers;
pub mod mpi_graph_handlers;
pub mod recall_handler;

// ── /log endpoint (separate since v2.1.0) ──
pub mod log_handler;

// ── Legacy API (deprecated) ──
pub mod local;

// ── Re-exports (unchanged from v2.3.0 — external callers unaffected) ──
pub use mpi::{build_mpi_router, MpiState, BaselineSnapshot};

#[allow(deprecated)]
pub use local::start_legacy_api_server;
