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
//!
//! ## Last Modified
//! v0.1.0 - Initial server library
//! v0.2.0 - Added management module for CMS integration
//! v0.3.0 - 🌟 Added api module for MemChain Agent HTTP API
//! v0.5.0 - 🌟 Added miner module for ReflectionMiner
//! v2.5.0 - 🌟 Added config_supernode module
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
pub mod api;
pub mod config;
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
