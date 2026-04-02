// ============================================
// File: crates/aeronyx-server/src/lib.rs
// ============================================
//! # AeroNyx Server Library
//!
//! ## Modification Reason
//! - Added management module for CMS integration.
//! - 🌟 Added api module for MemChain Agent HTTP API.
//! - 🌟 v0.5.0: Added miner module for ReflectionMiner block packing.
//! - 🌟 v2.5.0: Added config_supernode module for SuperNode LLM config.
//! - 🌟 v1.1.0-ChatRelay: Added config_infra, config_saas, config_chat_relay,
//!   config_memchain modules (split from config.rs).
//!
//! ## Last Modified
//! v0.1.0 - Initial server library
//! v0.2.0 - Added management module for CMS integration
//! v0.3.0 - 🌟 Added api module for MemChain Agent HTTP API
//! v0.5.0 - 🌟 Added miner module for ReflectionMiner
//! v2.5.0 - 🌟 Added config_supernode module
//! v1.1.0-ChatRelay - 🌟 Added config_infra, config_saas, config_chat_relay,
//!                    config_memchain (config.rs split)
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod api;
pub mod config;

// ── Config sub-modules (v1.1.0-ChatRelay split) ───────────────────────────
// Declared at crate root (not under config/) because config.rs is a single
// file, not a directory module. All imports use `crate::config_<name>::...`.
//
// Dependency order (each module may only depend on modules listed above it):
//   config_supernode  — no internal deps
//   config_infra      — no internal deps
//   config_saas       — no internal deps
//   config_chat_relay — no internal deps
//   config_memchain   — depends on config_supernode, config_saas, config_chat_relay
//   config            — depends on all of the above + config_infra
pub mod config_supernode;   // v2.5.0-SuperNode (pre-existing)
pub mod config_infra;       // v1.1.0-ChatRelay: Network/VPN/TUN/Key/Limits/Logging
pub mod config_saas;        // v1.1.0-ChatRelay: SaasConfig
pub mod config_chat_relay;  // v1.1.0-ChatRelay: ChatRelayConfig
pub mod config_memchain;    // v1.1.0-ChatRelay: MemChainConfig + MemChainMode

pub mod error;
pub mod handlers;
pub mod management;
pub mod miner;
pub mod server;
pub mod services;

// ── Primary re-exports ────────────────────────────────────────────────────
pub use config::ServerConfig;
pub use error::{Result, ServerError};
pub use server::Server;

// ── Management re-exports ─────────────────────────────────────────────────
pub use management::{ManagementClient, ManagementConfig};
