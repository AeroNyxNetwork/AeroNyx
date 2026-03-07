// ============================================
// File: crates/aeronyx-core/src/ledger/block.rs
// ============================================
//! # Block — Immutable Container of Facts / Records
//!
//! ## Creation Reason
//! Defines the `Block` and `BlockHeader` structures that pack a batch
//! of Facts into an immutable, hash-linked unit. This is the "crystal"
//! that solidifies loose memories into a tamper-proof chain.
//!
//! ## Modification Reason (v2.1.0)
//! - Added `RecordBlock` — the new primary block type that contains
//!   `MemoryRecord` entries instead of legacy `Fact` entries.
//! - Added `BLOCK_TYPE_MEMORY` (0x03) constant for RecordBlock identification.
//! - `Block` (containing Facts) is preserved for backward compatibility
//!   with existing P2P messages and legacy data.
//!
//! ## Main Functionality
//! - `BlockHeader`: height, timestamp, prev_block_hash, merkle_root, block_type
//! - `BlockHeader::hash()` — SHA-256 digest of the header (= the block's identity)
//! - `Block`: header + facts payload (⚠️ DEPRECATED, use RecordBlock)
//! - `RecordBlock`: 🌟 header + MemoryRecord payload (new primary type)
//! - `BlockType` constants: Normal (0x01), Checkpoint (0x02), Memory (0x03)
//!
//! ## Hash Canonical Form
//! ```text
//! SHA-256(
//!   height (8 bytes LE) || timestamp (8 bytes LE)
//!   || prev_block_hash (32 bytes) || merkle_root (32 bytes)
//!   || block_type (1 byte)
//! )
//! ```
//! Total: 81 bytes of pre-image → 32 bytes hash.
//!
//! ## Dependencies
//! - `sha2::Sha256`
//! - `serde` / `bincode`
//! - `aeronyx_core::ledger::Fact` (legacy)
//! - `aeronyx_core::ledger::record::MemoryRecord` (new)
//!
//! ## ⚠️ Important Note for Next Developer
//! - `BlockHeader::hash()` canonical form is a **stable contract**.
//!   Changing field order or encoding breaks the entire chain.
//! - `block_type` is a single byte for forward compatibility.
//!   Do NOT use a Rust enum with serde — bincode discriminant
//!   changes would be catastrophic.
//! - `BlockHeader` is small (~81 bytes serialised) and safe to
//!   broadcast over UDP inside a DataPacket.
//! - `Block` (legacy Fact container) MUST remain for P2P backward compat.
//!   New code should use `RecordBlock` exclusively.
//!
//! ## Last Modified
//! v0.5.0 - Initial Block and BlockHeader for Miner integration
//! v2.1.0 - 🌟 Added RecordBlock + BLOCK_TYPE_MEMORY for MRS-1

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[allow(deprecated)]
use super::Fact;
use super::record::MemoryRecord;

// ============================================
// Constants
// ============================================

/// Normal block: periodic packing of pending Facts (legacy).
pub const BLOCK_TYPE_NORMAL: u8 = 0x01;

/// Checkpoint block: LLM-summarised reflection (Phase 6+).
pub const BLOCK_TYPE_CHECKPOINT: u8 = 0x02;

/// Memory block: packing of MemoryRecord entries (MRS-1, v2.1.0+).
///
/// Used by `RecordBlock` — the new primary block type for the
/// smart Miner compaction pipeline.
pub const BLOCK_TYPE_MEMORY: u8 = 0x03;

/// Genesis block's "previous hash" — all zeros.
pub const GENESIS_PREV_HASH: [u8; 32] = [0u8; 32];

// ============================================
// BlockHeader
// ============================================

/// Header of a MemChain block.
///
/// This is the lightweight summary that gets broadcast over UDP
/// (`BlockAnnounce`). At ~81 bytes of canonical content it is well
/// within the 1300-byte MTU safety margin.
///
/// Shared by both legacy `Block` and new `RecordBlock` — the
/// `block_type` field distinguishes them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockHeader {
    /// Block height (0 = genesis).
    pub height: u64,

    /// Creation timestamp — Unix epoch seconds (UTC).
    pub timestamp: u64,

    /// Hash of the previous block's header (or `GENESIS_PREV_HASH`).
    pub prev_block_hash: [u8; 32],

    /// Merkle root of all `fact_id` or `record_id` values in this block.
    pub merkle_root: [u8; 32],

    /// Block type: `BLOCK_TYPE_NORMAL` (0x01), `BLOCK_TYPE_CHECKPOINT` (0x02),
    /// or `BLOCK_TYPE_MEMORY` (0x03).
    pub block_type: u8,
}

impl BlockHeader {
    /// Computes the SHA-256 hash of this header.
    ///
    /// This hash is the block's identity and is used as
    /// `prev_block_hash` by the next block in the chain.
    ///
    /// # Canonical Byte Sequence
    /// ```text
    /// height (8 LE) || timestamp (8 LE) || prev_block_hash (32)
    /// || merkle_root (32) || block_type (1)
    /// ```
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.height.to_le_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.prev_block_hash);
        hasher.update(self.merkle_root);
        hasher.update([self.block_type]);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Returns the header hash as a hex string (for logging).
    #[must_use]
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash())
    }
}

impl std::fmt::Display for BlockHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Block(height={}, type=0x{:02X}, hash={}..)",
            self.height,
            self.block_type,
            &self.hash_hex()[..8],
        )
    }
}

// ============================================
// Block (Legacy — preserved for P2P compat)
// ============================================

/// A complete MemChain block: header + fact payload.
///
/// ⚠️ **DEPRECATED**: Use `RecordBlock` for new code. `Block` is
/// preserved for backward compatibility with existing P2P messages
/// (`BlockAnnounce` with `BLOCK_TYPE_NORMAL`).
///
/// # Size Warning
/// A block with 100 facts can be 15–20 KB. **Never** broadcast a
/// full `Block` over UDP. Only the `BlockHeader` (~100 bytes) is
/// safe for UDP announcement. Use `SyncRequest` for full retrieval.
#[deprecated(since = "2.1.0", note = "Use RecordBlock instead of Block for new code")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block header (lightweight summary).
    pub header: BlockHeader,

    /// Facts packed into this block, ordered by timestamp ascending.
    #[allow(deprecated)]
    pub facts: Vec<Fact>,
}

#[allow(deprecated)]
impl Block {
    /// Creates a new block.
    #[must_use]
    pub fn new(header: BlockHeader, facts: Vec<Fact>) -> Self {
        Self { header, facts }
    }

    /// Returns the number of facts in this block.
    #[must_use]
    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    /// Returns the block hash (delegates to header).
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.header.hash()
    }
}

#[allow(deprecated)]
impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Block(height={}, facts={}, hash={}..)",
            self.header.height,
            self.facts.len(),
            &self.header.hash_hex()[..8],
        )
    }
}

// ============================================
// RecordBlock — New Primary Block Type (v2.1.0)
// ============================================

/// A MemChain block containing `MemoryRecord` entries (MRS-1).
///
/// This is the new primary block type used by the smart Miner
/// compaction pipeline. It replaces the legacy `Block` (which
/// contains `Fact` entries) for all new block creation.
///
/// ## Block Type
/// `RecordBlock` headers use `BLOCK_TYPE_MEMORY` (0x03).
///
/// ## Size Warning
/// A block with 100 records can be 20–50 KB (records are larger than facts).
/// **Never** broadcast a full `RecordBlock` over UDP. Only the `BlockHeader`
/// is safe for UDP announcement (`BlockAnnounce`). Use `SyncRecordRequest`
/// for full retrieval.
///
/// ## Merkle Root
/// The `merkle_root` in the header is computed from the `record_id` values
/// of all contained records, using the same `merkle_root()` function as
/// legacy blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordBlock {
    /// Block header (lightweight summary).
    /// `header.block_type` should be `BLOCK_TYPE_MEMORY` (0x03).
    pub header: BlockHeader,

    /// MemoryRecords packed into this block, ordered by timestamp ascending.
    pub records: Vec<MemoryRecord>,
}

impl RecordBlock {
    /// Creates a new RecordBlock.
    #[must_use]
    pub fn new(header: BlockHeader, records: Vec<MemoryRecord>) -> Self {
        Self { header, records }
    }

    /// Returns the number of records in this block.
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Returns the block hash (delegates to header).
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.header.hash()
    }
}

impl std::fmt::Display for RecordBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RecordBlock(height={}, records={}, hash={}..)",
            self.header.height,
            self.records.len(),
            &self.header.hash_hex()[..8],
        )
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(height: u64) -> BlockHeader {
        BlockHeader {
            height,
            timestamp: 1_700_000_000,
            prev_block_hash: GENESIS_PREV_HASH,
            merkle_root: [0xAA; 32],
            block_type: BLOCK_TYPE_NORMAL,
        }
    }

    fn make_memory_header(height: u64) -> BlockHeader {
        BlockHeader {
            height,
            timestamp: 1_700_000_000,
            prev_block_hash: GENESIS_PREV_HASH,
            merkle_root: [0xBB; 32],
            block_type: BLOCK_TYPE_MEMORY,
        }
    }

    #[test]
    fn test_header_hash_deterministic() {
        let h = make_header(0);
        assert_eq!(h.hash(), h.hash());
    }

    #[test]
    fn test_header_hash_differs_on_height() {
        let h0 = make_header(0);
        let h1 = make_header(1);
        assert_ne!(h0.hash(), h1.hash());
    }

    #[test]
    fn test_header_hash_differs_on_type() {
        let mut h1 = make_header(0);
        h1.block_type = BLOCK_TYPE_NORMAL;
        let mut h2 = make_header(0);
        h2.block_type = BLOCK_TYPE_CHECKPOINT;
        assert_ne!(h1.hash(), h2.hash());
    }

    #[test]
    fn test_header_hash_differs_memory_vs_normal() {
        let h_normal = make_header(0);
        let h_memory = make_memory_header(0);
        // They differ because merkle_root and block_type are both different
        assert_ne!(h_normal.hash(), h_memory.hash());
    }

    #[test]
    fn test_header_serde_roundtrip() {
        let h = make_header(42);
        let bytes = bincode::serialize(&h).unwrap();
        let restored: BlockHeader = bincode::deserialize(&bytes).unwrap();
        assert_eq!(h, restored);
    }

    #[test]
    #[allow(deprecated)]
    fn test_block_serde_roundtrip() {
        let fact = Fact::new(100, "s".into(), "p".into(), "o".into());
        let block = Block::new(make_header(1), vec![fact.clone()]);
        let bytes = bincode::serialize(&block).unwrap();
        let restored: Block = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.fact_count(), 1);
        assert_eq!(restored.header.height, 1);
        assert_eq!(restored.facts[0], fact);
    }

    #[test]
    fn test_record_block_basic() {
        use super::super::record::MemoryLayer;

        let record = MemoryRecord::new(
            [0xAA; 32],
            1_700_000_000,
            MemoryLayer::Episode,
            vec!["test".to_string()],
            "test-ai".to_string(),
            b"encrypted".to_vec(),
            vec![],
        );

        let block = RecordBlock::new(make_memory_header(1), vec![record]);
        assert_eq!(block.record_count(), 1);
        assert_eq!(block.header.block_type, BLOCK_TYPE_MEMORY);
    }

    #[test]
    fn test_record_block_serde_roundtrip() {
        use super::super::record::MemoryLayer;

        let record = MemoryRecord::new(
            [0xBB; 32],
            1_700_000_000,
            MemoryLayer::Knowledge,
            vec!["health".to_string()],
            "openclaw-v1".to_string(),
            b"secret_content".to_vec(),
            b"embedding_data".to_vec(),
        );

        let block = RecordBlock::new(make_memory_header(5), vec![record.clone()]);
        let bytes = bincode::serialize(&block).unwrap();
        let restored: RecordBlock = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.record_count(), 1);
        assert_eq!(restored.header.height, 5);
        assert_eq!(restored.header.block_type, BLOCK_TYPE_MEMORY);
        assert_eq!(restored.records[0], record);
    }

    #[test]
    fn test_record_block_display() {
        let block = RecordBlock::new(make_memory_header(3), vec![]);
        let s = format!("{}", block);
        assert!(s.contains("RecordBlock"));
        assert!(s.contains("height=3"));
        assert!(s.contains("records=0"));
    }

    #[test]
    fn test_display() {
        let h = make_header(5);
        let s = format!("{}", h);
        assert!(s.contains("height=5"));
    }

    #[test]
    fn test_block_type_constants_unique() {
        // Ensure all block type constants are distinct
        let types = [BLOCK_TYPE_NORMAL, BLOCK_TYPE_CHECKPOINT, BLOCK_TYPE_MEMORY];
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(types[i], types[j], "Block type constants must be unique");
            }
        }
    }
}
