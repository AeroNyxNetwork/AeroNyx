// ============================================
// File: crates/aeronyx-server/src/lib.rs
// ============================================
//! # AeroNyx Server Library
//!
//! ## Creation Reason
//! Provides the core server implementation for the AeroNyx privacy network,
//! orchestrating all components to create a functional privacy tunnel server.
//!
//! ## Main Functionality
//!
//! ### Modules
//! - [`config`]: Server configuration management
//! - [`server`]: Main server orchestration
//! - [`services`]: Business logic services
//!   - [`services::session`]: Session management
//!   - [`services::routing`]: Packet routing
//!   - [`services::ip_pool`]: Virtual IP allocation
//!   - [`services::handshake`]: Handshake processing
//! - [`handlers`]: Packet and event handlers
//! - [`error`]: Server-specific error types
//!
//! ## Architecture Overview
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        AeroNyx Server                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐  │
//! │  │   Config    │────►│   Server    │────►│    Handlers     │  │
//! │  │  Manager    │     │ Orchestrator│     │                 │  │
//! │  └─────────────┘     └──────┬──────┘     └────────┬────────┘  │
//! │                             │                     │           │
//! │         ┌───────────────────┼───────────────────┬─┘           │
//! │         ▼                   ▼                   ▼             │
//! │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
//! │  │  Session    │     │   Routing   │     │   IP Pool   │     │
//! │  │  Manager    │     │   Service   │     │   Service   │     │
//! │  └─────────────┘     └─────────────┘     └─────────────┘     │
//! │                                                               │
//! ├───────────────────────────────────────────────────────────────┤
//! │                     Transport Layer                           │
//! │  ┌─────────────────────┐     ┌─────────────────────────────┐ │
//! │  │    UDP Transport    │     │       TUN Device            │ │
//! │  │  (client packets)   │     │     (IP packets)            │ │
//! │  └─────────────────────┘     └─────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Data Flow
//! ```text
//! Client → UDP → Decrypt → Route → TUN → Internet
//! Client ← UDP ← Encrypt ← Route ← TUN ← Internet
//! ```
//!
//! ## ⚠️ Important Note for Next Developer
//! - Server requires root or CAP_NET_ADMIN for TUN
//! - Configuration changes require restart (no hot-reload)
//! - Graceful shutdown waits for active sessions
//! - Metrics and logging are built-in
//!
//! ## Last Modified
//! v0.1.0 - Initial server library

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod error;
pub mod handlers;
pub mod server;
pub mod services;

// Re-export primary types
pub use config::ServerConfig;
pub use error::{ServerError, Result};
pub use server::Server;
