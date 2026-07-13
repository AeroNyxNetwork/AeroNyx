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
//! - v2.7.0-BlockSync: Added the signed, chain-scoped
//!   `RecordCommitmentBlockV1`. It synchronises only opaque record IDs and
//!   never embeds memory payloads, owners, tags, or embeddings.
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
//! - `RecordCommitmentBlockV1` is the distributed integrity contract.
//!   Do not add full `MemoryRecord` fields to it; owner-authorised sealed
//!   payload replication is a separate protocol.
//!
//! ## Last Modified
//! v0.5.0 - Initial Block and BlockHeader for Miner integration
//! v2.1.0 - 🌟 Added RecordBlock + BLOCK_TYPE_MEMORY for MRS-1
//! v2.7.0-BlockSync - Added signed node-blind commitment blocks.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

use super::merkle::merkle_root;
use super::record::MemoryRecord;
#[allow(deprecated)]
use super::Fact;
use crate::crypto::{IdentityKeyPair, IdentityPublicKey};

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

/// Stable chain identifier for the AeroNyx MemChain production ledger.
///
/// This is `SHA-256("AeroNyx-MemChain-Mainnet-v1")`. A fixed identifier
/// prevents a valid block from a test or private chain being replayed into the
/// production chain. Changing it creates a different chain and therefore
/// requires an explicit protocol migration.
pub const AERONYX_MEMCHAIN_MAINNET_CHAIN_ID: [u8; 32] = [
    0x6e, 0x7b, 0xbc, 0xa5, 0x8b, 0x22, 0xb8, 0xcc, 0x0e, 0x2e, 0xc7, 0x28, 0xdf, 0xa9, 0xe6, 0xf2,
    0x7d, 0x6c, 0xa5, 0xc8, 0x44, 0x3e, 0x2d, 0x5d, 0x88, 0x84, 0x2f, 0x6e, 0xe8, 0xe1, 0xf9, 0xa4,
];

/// First version of the node-blind commitment block contract.
pub const RECORD_COMMITMENT_BLOCK_VERSION_V1: u16 = 1;

/// Hard upper bound for commitments in one synchronised block.
///
/// At 32 bytes per commitment this keeps a block payload below 9 KiB before
/// framing, small enough for bounded HTTP range responses while avoiding UDP
/// fragmentation. Full memory records are deliberately never part of this
/// structure.
pub const MAX_RECORD_COMMITMENTS_PER_BLOCK: usize = 256;

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
#[deprecated(
    since = "2.1.0",
    note = "Use RecordBlock instead of Block for new code"
)]
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
// RecordCommitmentBlockV1 — Node-Blind Sync Contract
// ============================================

/// Versioned header for a node-blind MemChain commitment block.
///
/// Unlike the legacy [`BlockHeader`], this header binds the chain identity,
/// protocol version, record count, and proposing node. It contains no memory
/// owner, timestamp, tag, embedding, ciphertext, endpoint, or routing metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordCommitmentHeaderV1 {
    /// Wire and hashing contract version.
    pub protocol_version: u16,
    /// Prevents cross-network replay.
    pub chain_id: [u8; 32],
    /// One-based block height. Height zero means no block exists yet.
    pub height: u64,
    /// Creation timestamp in Unix epoch seconds.
    pub timestamp: u64,
    /// Hash of the previous V1 commitment header, or all zeroes at height one.
    pub prev_block_hash: [u8; 32],
    /// Merkle root of the sorted opaque record commitments.
    pub merkle_root: [u8; 32],
    /// Number of commitments in the block.
    pub record_count: u32,
    /// Ed25519 public key of the node proposing this block.
    pub proposer: [u8; 32],
}

impl RecordCommitmentHeaderV1 {
    /// Computes the domain-separated canonical header hash.
    ///
    /// The byte order and field order are a stable protocol contract.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AeroNyx-RecordCommitmentBlock-v1");
        hasher.update(self.protocol_version.to_le_bytes());
        hasher.update(self.chain_id);
        hasher.update(self.height.to_le_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.prev_block_hash);
        hasher.update(self.merkle_root);
        hasher.update(self.record_count.to_le_bytes());
        hasher.update(self.proposer);
        hasher.finalize().into()
    }

    /// Returns the canonical block hash as lowercase hexadecimal.
    #[must_use]
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash())
    }
}

/// Signed, node-blind block synchronised between AeroNyx nodes.
///
/// Only opaque content-addressed commitments are synchronised. Ciphertext and
/// user metadata remain in the separately authorised storage layer. This
/// separation is a privacy invariant: block catch-up must not become a global
/// replication channel for memory payloads or an owner activity graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordCommitmentBlockV1 {
    /// Signed chain header.
    pub header: RecordCommitmentHeaderV1,
    /// Lexicographically sorted, unique opaque record identifiers.
    pub record_ids: Vec<[u8; 32]>,
    /// Ed25519 signature by `header.proposer` over `header.hash()`.
    #[serde(with = "serde_signature_64")]
    pub proposer_signature: [u8; 64],
}

/// Validation failures for the V1 commitment chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordCommitmentValidationError {
    /// The block contract version is unsupported.
    UnsupportedVersion,
    /// The block belongs to another chain.
    WrongChain,
    /// The height does not continue the local chain.
    InvalidHeight,
    /// The previous block hash does not match the local tip.
    InvalidPreviousHash,
    /// A block must carry at least one commitment.
    EmptyBlock,
    /// The block exceeds the bounded commitment count.
    TooManyCommitments,
    /// Header count and payload count differ.
    RecordCountMismatch,
    /// Commitments are not in canonical lexicographic order.
    NonCanonicalOrder,
    /// The block repeats a commitment.
    DuplicateCommitment,
    /// The commitment Merkle root does not match the header.
    InvalidMerkleRoot,
    /// The proposer public key is malformed.
    InvalidProposer,
    /// The proposer signature is invalid.
    InvalidSignature,
}

impl std::fmt::Display for RecordCommitmentValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::UnsupportedVersion => "unsupported block protocol version",
            Self::WrongChain => "block belongs to another chain",
            Self::InvalidHeight => "block height does not continue the local chain",
            Self::InvalidPreviousHash => "block previous hash does not match the local tip",
            Self::EmptyBlock => "commitment block is empty",
            Self::TooManyCommitments => "commitment block exceeds the maximum size",
            Self::RecordCountMismatch => "header record count does not match payload",
            Self::NonCanonicalOrder => "record commitments are not canonically ordered",
            Self::DuplicateCommitment => "record commitment is duplicated",
            Self::InvalidMerkleRoot => "commitment Merkle root is invalid",
            Self::InvalidProposer => "block proposer public key is invalid",
            Self::InvalidSignature => "block proposer signature is invalid",
        };
        f.write_str(value)
    }
}

impl std::error::Error for RecordCommitmentValidationError {}

impl RecordCommitmentBlockV1 {
    /// Builds and signs a deterministic commitment block.
    ///
    /// Callers must validate each source record's content address and owner
    /// signature before passing its identifier here. Sorting happens before the
    /// Merkle root is computed so independent honest nodes derive the same
    /// payload order for the same commitment set.
    #[must_use]
    pub fn new_signed(
        height: u64,
        timestamp: u64,
        prev_block_hash: [u8; 32],
        mut record_ids: Vec<[u8; 32]>,
        identity: &IdentityKeyPair,
    ) -> Self {
        record_ids.sort_unstable();
        let header = RecordCommitmentHeaderV1 {
            protocol_version: RECORD_COMMITMENT_BLOCK_VERSION_V1,
            chain_id: AERONYX_MEMCHAIN_MAINNET_CHAIN_ID,
            height,
            timestamp,
            prev_block_hash,
            merkle_root: merkle_root(&record_ids),
            record_count: record_ids.len() as u32,
            proposer: identity.public_key_bytes(),
        };
        let proposer_signature = identity.sign(&header.hash());
        Self {
            header,
            record_ids,
            proposer_signature,
        }
    }

    /// Returns the canonical block identity.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        self.header.hash()
    }

    /// Validates contract, chain continuity, Merkle integrity, and proposer
    /// authenticity without inspecting any memory payload.
    pub fn verify(
        &self,
        expected_chain_id: &[u8; 32],
        expected_height: u64,
        expected_prev_hash: &[u8; 32],
    ) -> Result<(), RecordCommitmentValidationError> {
        if self.header.protocol_version != RECORD_COMMITMENT_BLOCK_VERSION_V1 {
            return Err(RecordCommitmentValidationError::UnsupportedVersion);
        }
        if &self.header.chain_id != expected_chain_id {
            return Err(RecordCommitmentValidationError::WrongChain);
        }
        if self.header.height != expected_height || expected_height == 0 {
            return Err(RecordCommitmentValidationError::InvalidHeight);
        }
        if &self.header.prev_block_hash != expected_prev_hash {
            return Err(RecordCommitmentValidationError::InvalidPreviousHash);
        }
        if self.record_ids.is_empty() {
            return Err(RecordCommitmentValidationError::EmptyBlock);
        }
        if self.record_ids.len() > MAX_RECORD_COMMITMENTS_PER_BLOCK {
            return Err(RecordCommitmentValidationError::TooManyCommitments);
        }
        if self.header.record_count as usize != self.record_ids.len() {
            return Err(RecordCommitmentValidationError::RecordCountMismatch);
        }
        if self.record_ids.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(RecordCommitmentValidationError::NonCanonicalOrder);
        }
        let unique: HashSet<_> = self.record_ids.iter().collect();
        if unique.len() != self.record_ids.len() {
            return Err(RecordCommitmentValidationError::DuplicateCommitment);
        }
        if merkle_root(&self.record_ids) != self.header.merkle_root {
            return Err(RecordCommitmentValidationError::InvalidMerkleRoot);
        }
        let proposer = IdentityPublicKey::from_bytes(&self.header.proposer)
            .map_err(|_| RecordCommitmentValidationError::InvalidProposer)?;
        proposer
            .verify(&self.header.hash(), &self.proposer_signature)
            .map_err(|_| RecordCommitmentValidationError::InvalidSignature)
    }
}

impl std::fmt::Display for RecordCommitmentBlockV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RecordCommitmentBlockV1(height={}, commitments={}, hash={}..)",
            self.header.height,
            self.record_ids.len(),
            &self.header.hash_hex()[..8],
        )
    }
}

/// Serde adapter for fixed Ed25519 signatures on compilers where arrays larger
/// than 32 bytes do not implement `Serialize`/`Deserialize` directly.
mod serde_signature_64 {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let (first, second) = value.split_at(32);
        let first: [u8; 32] = first.try_into().unwrap();
        let second: [u8; 32] = second.try_into().unwrap();
        (first, second).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let (first, second) = <([u8; 32], [u8; 32])>::deserialize(deserializer)?;
        let mut signature = [0u8; 64];
        signature[..32].copy_from_slice(&first);
        signature[32..].copy_from_slice(&second);
        Ok(signature)
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
            "aeronyx-memory-v1".to_string(),
            b"secret_content".to_vec(),
            vec![0.1, 0.2, 0.3],
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

    #[test]
    fn test_commitment_block_v1_roundtrip_and_verify() {
        let identity = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_000_001,
            GENESIS_PREV_HASH,
            vec![[0xCC; 32], [0x11; 32], [0x77; 32]],
            &identity,
        );

        assert_eq!(block.record_ids, vec![[0x11; 32], [0x77; 32], [0xCC; 32]]);
        block
            .verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 1, &GENESIS_PREV_HASH)
            .expect("fresh commitment block must verify");

        let encoded = bincode::serialize(&block).expect("serialize commitment block");
        let restored: RecordCommitmentBlockV1 =
            bincode::deserialize(&encoded).expect("deserialize commitment block");
        assert_eq!(restored, block);
        restored
            .verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 1, &GENESIS_PREV_HASH)
            .expect("restored commitment block must verify");
    }

    #[test]
    fn test_commitment_block_v1_rejects_chain_discontinuity() {
        let identity = IdentityKeyPair::generate();
        let block = RecordCommitmentBlockV1::new_signed(
            2,
            1_700_000_002,
            [0x22; 32],
            vec![[0x01; 32]],
            &identity,
        );

        assert_eq!(
            block.verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 3, &[0x22; 32],),
            Err(RecordCommitmentValidationError::InvalidHeight)
        );
        assert_eq!(
            block.verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 2, &[0x33; 32],),
            Err(RecordCommitmentValidationError::InvalidPreviousHash)
        );
    }

    #[test]
    fn test_commitment_block_v1_rejects_payload_and_signature_tampering() {
        let identity = IdentityKeyPair::generate();
        let mut payload_tampered = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_000_003,
            GENESIS_PREV_HASH,
            vec![[0x10; 32], [0x20; 32]],
            &identity,
        );
        payload_tampered.record_ids[1] = [0x30; 32];
        assert_eq!(
            payload_tampered.verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 1, &GENESIS_PREV_HASH,),
            Err(RecordCommitmentValidationError::InvalidMerkleRoot)
        );

        let mut signature_tampered = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_000_004,
            GENESIS_PREV_HASH,
            vec![[0x10; 32]],
            &identity,
        );
        signature_tampered.proposer_signature[0] ^= 0x80;
        assert_eq!(
            signature_tampered.verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 1, &GENESIS_PREV_HASH,),
            Err(RecordCommitmentValidationError::InvalidSignature)
        );
    }

    #[test]
    fn test_commitment_block_v1_rejects_duplicate_commitments() {
        let identity = IdentityKeyPair::generate();
        let duplicate = [0x42; 32];
        let block = RecordCommitmentBlockV1::new_signed(
            1,
            1_700_000_005,
            GENESIS_PREV_HASH,
            vec![duplicate, duplicate],
            &identity,
        );
        assert_eq!(
            block.verify(&AERONYX_MEMCHAIN_MAINNET_CHAIN_ID, 1, &GENESIS_PREV_HASH,),
            Err(RecordCommitmentValidationError::DuplicateCommitment)
        );
    }
}
