// ============================================
// File: crates/aeronyx-server/src/miner/mod.rs
// ============================================
//! # Miner Module — Block Packing + SaaS Cognitive Scheduling
//!
//! ## Creation Reason
//! Contains the background mining task that periodically packs pending
//! Facts from the MemPool into immutable Blocks.
//!
//! ## Submodules
//! - [`reflection`]: `ReflectionMiner` — single-user cognitive mining loop (Local mode)
//! - [`scheduler`]:  `MinerScheduler` — multi-user dispatch scheduler (SaaS mode)
//!
//! ## Mode Routing
//! | Mode  | Component         | Trigger                    |
//! |-------|-------------------|----------------------------|
//! | Local | ReflectionMiner   | `run()` — loops forever    |
//! | SaaS  | MinerScheduler    | `tick()` — called by timer |
//!
//! ## Last Modified
//! v0.5.0 - Initial Miner module
//! v1.0.0-MultiTenant - Added scheduler submodule for SaaS mode

pub mod reflection;
// v1.0.0-MultiTenant: SaaS multi-user miner scheduler
pub mod scheduler;

pub use reflection::ReflectionMiner;
pub use scheduler::MinerScheduler;
