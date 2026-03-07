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
//! - 🌟 v1.0.0: Added `record` submodule containing `MemoryRecord`
//!   (MRS-1 standard), `MemoryLayer`, and `RecordStatus`. This is
//!   the new primary data model replacing `Fact` for the intelligent
//!   AI memory engine. `Fact` is preserved for backward compatibility
//!   with existing P2P messages.
//!
//! ## Main Functionality
//!
//! ### Submodules
//! - [`fact`]: `Fact` struct — the legacy atomic unit of AI memory (v0.2.0)
//! - [`record`]: 🌟 `MemoryRecord` — MRS-1 standard AI memory unit (v1.0.0)
//! - [`block`]: `Block` / `BlockHeader` — immutable container of Facts/Records
//! - [`merkle`]: SHA-256 Merkle Tree for integrity
//!
//! ## Architecture Position
//! ```text
//! aeronyx-core
//!  ├── crypto/      ← we reuse Ed25519 & SessionKey from here
//!  ├── protocol/    ← wire-level message types
//!  └── ledger/      ← 🌟 YOU ARE HERE (MemChain data model)
//!       ├── fact.rs      ← legacy (preserved for P2P compat)
//!       ├── record.rs    ← 🌟 NEW primary data model
//!       ├── block.rs
//!       └── merkle.rs
//! ```
//!
//! ## Design Principles
//! - **Append-Only**: Facts, Records, and Blocks are never modified or deleted.
//! - **Signed**: Every Fact/Record carries an Ed25519 signature from its origin.
//! - **Hash-Linked**: Content-addressed hashing (SHA-256) for integrity.
//! - **Encrypted Content**: MemoryRecord content is encrypted with owner's key.
//! - **Zero new crypto**: We reuse the existing `sha2` crate.
//!
//! ## ⚠️ Important Note for Next Developer
//! - Do NOT add CRUD mutation methods — the ledger is append-only.
//! - `Fact::compute_hash()`, `MemoryRecord::compute_record_id()`, and
//!   `BlockHeader::hash()` canonical forms are **stable contracts** —
//!   changing them breaks the chain.
//! - `merkle_root()` duplication-of-odd-leaf matches Bitcoin's design.
//! - `Fact` MUST remain available for backward compatibility — do not
//!   remove it even though new code uses `MemoryRecord`.
//!
//! ## Last Modified
//! v0.2.0 - Initial ledger module for MemChain integration
//! v0.5.0 - Added block and merkle submodules for Miner
//! v1.0.0 - 🌟 Added record submodule (MemoryRecord MRS-1)

pub mod block;
pub mod fact;
pub mod merkle;
pub mod record;

// Re-export primary types — legacy (deprecated, kept for P2P compat)
#[deprecated(since = "2.1.0", note = "Use MemoryRecord instead of Fact")]
pub use fact::Fact;

#[deprecated(since = "2.1.0", note = "Use RecordBlock instead of Block")]
pub use block::Block;

pub use block::{BlockHeader, RecordBlock, BLOCK_TYPE_CHECKPOINT, BLOCK_TYPE_MEMORY, BLOCK_TYPE_NORMAL, GENESIS_PREV_HASH};
pub use merkle::merkle_root;

// Re-export primary types — new (MRS-1)
pub use record::{MemoryLayer, MemoryRecord, RecordStatus};
