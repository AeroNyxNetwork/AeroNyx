// ============================================
// File: crates/aeronyx-server/src/miner/mod.rs
// ============================================
//! # Miner Module — Block Packing for MemChain
//!
//! ## Creation Reason
//! Contains the background mining task that periodically packs pending
//! Facts from the MemPool into immutable Blocks.
//!
//! ## Submodules
//! - [`reflection`]: `ReflectionMiner` — the core mining loop
//!
//! ## Last Modified
//! v0.5.0 - 🌟 Initial Miner module

pub mod reflection;

pub use reflection::ReflectionMiner;
