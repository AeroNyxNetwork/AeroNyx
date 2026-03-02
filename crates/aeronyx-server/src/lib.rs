// ============================================
// File: crates/aeronyx-server/src/lib.rs
// ============================================
//! # AeroNyx Server Library
//!
//! ## Modification Reason
//! - Added management module for CMS integration.
//! - 🌟 Added api module for MemChain Agent HTTP API.
//! - 🌟 v0.5.0: Added miner module for ReflectionMiner block packing.
//!
//! ## Last Modified
//! v0.1.0 - Initial server library
//! v0.2.0 - Added management module for CMS integration
//! v0.3.0 - 🌟 Added api module for MemChain Agent HTTP API
//! v0.5.0 - 🌟 Added miner module for ReflectionMiner

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod api;
pub mod config;
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
