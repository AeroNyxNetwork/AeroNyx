//! ============================================
//! File: crates/aeronyx-server/src/management/mod.rs
//! ============================================
//! # Node Management Module
//!
//! ## Creation Reason
//! Provides communication with the central management system (CMS)
//! for node registration, heartbeat reporting, and session tracking.
//!
//! ## Modification Reason (v1.3.0)
//! - 🌟 Added `command_handler` sub-module for CMS → Node command dispatch.
//! - 🌟 Re-exported `CommandHandler` and `Command` for use in `server.rs`.
//!
//! ## Main Functionality
//! - `ManagementClient`: HTTP client for CMS API communication
//! - `NodeRegistration`: One-time node registration with code
//! - `HeartbeatReporter`: Periodic status reporting
//! - `SessionReporter`: Session lifecycle event reporting
//! - 🌟 `CommandHandler`: CMS command dispatch and execution (v1.3.0)
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                      Management Module                          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐     │
//! │  │  Registration │  │  Heartbeat   │  │  Session         │     │
//! │  │  (one-time)   │  │  Reporter    │  │  Reporter        │     │
//! │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘     │
//! │         │                 │                    │               │
//! │         │                 │ commands           │               │
//! │         │                 ▼                    │               │
//! │         │          ┌──────────────┐            │               │
//! │         │          │  🌟 Command  │            │               │
//! │         │          │   Handler    │            │               │
//! │         │          └──────┬───────┘            │               │
//! │         │                 │                    │               │
//! │         └─────────────────┼────────────────────┘               │
//! │                           ▼                                    │
//! │                 ┌──────────────────┐                           │
//! │                 │ ManagementClient │                           │
//! │                 │  (HTTP + Sign)   │                           │
//! │                 └────────┬─────────┘                           │
//! │                          │                                     │
//! └──────────────────────────┼─────────────────────────────────────┘
//!                            ▼
//!                    CMS API Server
//! ```
//!
//! ## Authentication Flow
//! ```text
//! 1. Registration (no auth):
//!    POST /node/bind/ { code, public_key, hardware_info }
//!
//! 2. Authenticated requests (Ed25519 signature):
//!    Headers:
//!      X-Node-ID: <public_key_hex>
//!      X-Timestamp: <unix_timestamp>
//!      X-Signature: <signature_hex>
//!
//!    Signature = Ed25519_Sign(SHA256(node_id + timestamp + body))
//! ```
//!
//! ## Last Modified
//! v0.1.0 - Initial management module
//! v1.3.0 - 🌟 Added command_handler module for CMS command dispatch

pub mod client;
pub mod command_handler;
pub mod config;
pub mod integrity;
pub mod models;
pub mod reporter;
pub mod ws_client;

pub use client::ManagementClient;
pub use command_handler::CommandHandler;
pub use config::ManagementConfig;
pub use models::Command;
pub use reporter::{HeartbeatReporter, SessionReporter, SessionEvent};
pub use ws_client::WsTunnel;
