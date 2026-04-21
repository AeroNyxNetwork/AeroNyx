// ============================================
// File: crates/aeronyx-server/src/lib.rs
// ============================================
//! # AeroNyx Server Library
//!
//! ## Modification Reason
//! - Added management module for CMS integration.
//! - Added api module for MemChain Agent HTTP API.
//! - v0.5.0: Added miner module for ReflectionMiner block packing.
//! - v2.5.0: Added config_supernode module for SuperNode LLM config.
//! - v1.0.0-MultiTenant: miner module now exports MinerScheduler in addition
//!   to ReflectionMiner. MinerScheduler is the SaaS-mode Miner dispatcher.
//! - v1.0.0-MultiTenant: Added config sub-modules (config_chat_relay,
//!   config_infra, config_memchain, config_saas) that are referenced by
//!   config.rs via pub use re-exports.
//!
//! ## Last Modified
//! v0.1.0 - Initial server library
//! v0.2.0 - Added management module for CMS integration
//! v0.3.0 - Added api module for MemChain Agent HTTP API
//! v0.5.0 - Added miner module for ReflectionMiner
//! v2.5.0 - Added config_supernode module
//! v1.0.0-MultiTenant - MinerScheduler added to miner module;
//!                      config sub-modules declared at crate root

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod api;
pub mod config;

// v1.0.0-MultiTenant: Config sub-modules referenced by config.rs pub use re-exports.
// These are declared at crate root (not under config/) because config.rs is a single
// file module. config.rs re-exports their types via `pub use crate::config_xxx::...`.
pub mod config_chat_relay;
pub mod config_infra;
pub mod config_memchain;
pub mod config_saas;

// v2.5.0+SuperNode: SuperNode LLM configuration types.
// Declared at crate root (not under config/) because config.rs is a single file,
// not a directory module. All imports use `crate::config_supernode::...`.
pub mod config_supernode;

pub mod error;
pub mod handlers;
pub mod management;
pub mod miner;
pub mod server;
pub mod services;

// Re-export primary types
pub use config::ServerConfig;
pub use error::{Result, ServerError};
pub use server::Server;

// Re-export management types
pub use management::{ManagementClient, ManagementConfig};
