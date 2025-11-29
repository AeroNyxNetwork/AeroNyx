// ============================================
// File: crates/aeronyx-common/src/lib.rs
// ============================================
//! # AeroNyx Common - Shared Utilities Library
//!
//! ## Creation Reason
//! Provides foundational types and utilities shared across all AeroNyx crates,
//! ensuring consistency and reducing code duplication.
//!
//! ## Main Functionality
//! - [`types`]: Core type definitions (SessionId, identifiers)
//! - [`time`]: Time utilities including atomic timestamps
//! - [`error`]: Common error types and result aliases
//!
//! ## Architecture Position
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              aeronyx-server                         │
//! │                    │                                │
//! │         ┌──────────┴──────────┐                    │
//! │         ▼                     ▼                    │
//! │   aeronyx-core         aeronyx-transport           │
//! │         │                     │                    │
//! │         └──────────┬──────────┘                    │
//! │                    ▼                               │
//! │             aeronyx-common  ◄── You are here      │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Dependencies
//! - No internal crate dependencies (leaf node)
//! - Minimal external dependencies for maximum compatibility
//!
//! ## ⚠️ Important Note for Next Developer
//! - This crate is the foundation - changes affect everything
//! - Keep dependencies minimal
//! - All public types should implement standard traits (Debug, Clone, etc.)
//! - Security-sensitive types must implement Zeroize
//!
//! ## Last Modified
//! v0.1.0 - Initial implementation

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod time;
pub mod types;

// Re-export commonly used items at crate root
pub use error::{CommonError, Result};
pub use types::{SessionId, VirtualIp};
