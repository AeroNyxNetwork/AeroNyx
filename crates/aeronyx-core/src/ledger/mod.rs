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
//! ## Main Functionality
//!
//! ### Submodules
//! - [`fact`]: `Fact` struct — the atomic unit of AI memory
//!   (subject-predicate-object triple with Ed25519 signature)
//!
//! ## Architecture Position
//! ```text
//! aeronyx-core
//!  ├── crypto/      ← we reuse Ed25519 & SessionKey from here
//!  ├── protocol/    ← wire-level message types
//!  └── ledger/      ← 🌟 YOU ARE HERE (MemChain data model)
//!       └── fact.rs
//! ```
//!
//! ## Design Principles
//! - **Append-Only**: Facts are never modified or deleted.
//! - **Signed**: Every Fact carries an Ed25519 signature from its origin node.
//! - **Hash-Linked**: `fact_id` is the SHA-256 digest of the canonical content,
//!   enabling integrity verification and deduplication.
//! - **Zero new crypto**: We reuse the existing `IdentityKeyPair` and `sha2`
//!   crate that are already in `aeronyx-core`.
//!
//! ## Dependencies
//! - `serde` / `bincode` for serialisation (already in workspace)
//! - `sha2` for content hashing (already in workspace)
//!
//! ## ⚠️ Important Note for Next Developer
//! - Do NOT add CRUD mutation methods — the ledger is append-only.
//! - `Fact::hash()` output MUST remain stable across versions;
//!   changing the field order or encoding would break signature
//!   verification for all existing facts.
//! - Future `block.rs` and `merkle.rs` submodules will be added here
//!   when the Miner / Checkpoint system is implemented.
//!
//! ## Last Modified
//! v0.2.0 - Initial ledger module for MemChain integration
// ============================================

pub mod fact;

// Re-export primary types
pub use fact::Fact;
