// ============================================
// File: crates/aeronyx-core/src/ledger/mod.rs
// ============================================
//! # Ledger Module - MemChain Distributed AI Memory
//!
//! ## Creation Reason
//! Defines the core data structures for the MemChain "distributed AI memory
//! ledger" that runs on top of the AeroNyx encrypted tunnel. Each memory
//! fact is an append-only, signed, hash-linked record — inspired by
//! blockchain principles but optimised for single-node + P2P sync.
//!
//! ## Modification Reason
//! - 🌟 v0.5.0: Added `block` and `merkle` submodules for Miner /
//!   Checkpoint system. Facts are now packed into Blocks with a
//!   SHA-256 Merkle root for integrity verification.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`fact`]: `Fact` struct — the atomic unit of AI memory
//! - [`block`]: 🌟 `Block` / `BlockHeader` — immutable container of Facts
//! - [`merkle`]: 🌟 SHA-256 Merkle Tree for Fact integrity
//!
//! ## Architecture Position
//! ```text
//! aeronyx-core
//!  ├── crypto/      ← we reuse Ed25519 & SessionKey from here
//!  ├── protocol/    ← wire-level message types
//!  └── ledger/      ← 🌟 YOU ARE HERE (MemChain data model)
//!       ├── fact.rs
//!       ├── block.rs   ← 🌟 NEW
//!       └── merkle.rs  ← 🌟 NEW
//! ```
//!
//! ## Design Principles
//! - **Append-Only**: Facts and Blocks are never modified or deleted.
//! - **Signed**: Every Fact carries an Ed25519 signature from its origin node.
//! - **Hash-Linked**: `fact_id` is SHA-256 of content; `BlockHeader::hash()`
//!   links blocks into a chain via `prev_block_hash`.
//! - **Zero new crypto**: We reuse the existing `sha2` crate.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Do NOT add CRUD mutation methods — the ledger is append-only.
//! - `Fact::compute_hash()` and `BlockHeader::hash()` canonical forms
//!   are **stable contracts** — changing them breaks the chain.
//! - `merkle_root()` duplication-of-odd-leaf matches Bitcoin's design.
//!
//! ## Last Modified
//! v0.2.0 - Initial ledger module for MemChain integration
//! v0.5.0 - 🌟 Added block and merkle submodules for Miner

pub mod block;
pub mod fact;
pub mod merkle;

// Re-export primary types
pub use block::{Block, BlockHeader, BLOCK_TYPE_CHECKPOINT, BLOCK_TYPE_NORMAL, GENESIS_PREV_HASH};
pub use fact::Fact;
pub use merkle::merkle_root;
