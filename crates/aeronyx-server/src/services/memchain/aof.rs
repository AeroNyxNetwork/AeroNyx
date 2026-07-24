// ============================================
// File: crates/aeronyx-server/src/services/memchain/aof.rs
// ============================================
//! # AOF — Append-Only File Storage
//!
//! ## Creation Reason
//! Provides durable, crash-safe persistence for `MemChain` Facts and Blocks
//! using a simple append-only binary file (`.memchain`).
//!
//! ## Modification Reason
//! - 🌟 v0.5.0: Added `append_block()` for persisting mined Blocks.
//!   Blocks use a different record tag (0x02) from Facts (0x01) so
//!   `replay()` can distinguish them. Added `last_block_hash()` and
//!   `last_block_height()` for Miner chain-linking.
//! - v0.6.0-BoundedRecords: Unified Fact/Block record writes behind one
//!   bounded codec path and made replay reject non-canonical record payloads
//!   without changing the established bytes on disk.
//! - v0.7.0-TornTailRecovery: Unified record framing for replay and startup
//!   recovery. A partial physical tail is truncated to the last complete
//!   record; fully framed corruption remains a fail-closed startup error.
//! - v0.8.0-SemanticIntegrity: Validate content-addressed Fact identifiers,
//!   Block Merkle roots, and Block ancestry without changing the disk format.
//!   Added a read-only aggregate verification report for operator tooling.
//!
//! ## On-Disk Record Format (v0.5.0)
//! ```text
//! ┌───────┬──────────────────┬────────────────────────────┐
//! │ tag   │  length: u32 LE  │  bincode payload           │
//! │ 1 byte│  4 bytes         │  variable                  │
//! └───────┴──────────────────┴────────────────────────────┘
//! ```
//!
//! | Tag  | Meaning |
//! |------|---------|
//! | 0x01 | Fact record |
//! | 0x02 | Block record |
//!
//! ## Backward Compatibility
//! v0.3.0 files used `[u32 LE][bincode(Fact)]` without a tag byte.
//! The `replay()` function detects the legacy format by checking if
//! the first byte looks like a valid tag. If not, it falls back to
//! legacy parsing (length-only, assumes all records are Facts).
//!
//! ## Crash Safety
//! - Writes are followed by `flush()`.
//! - Replay reports trailing partial records; append-open repairs only that
//!   incomplete physical tail before accepting another write.
//!
//! ## ⚠️ Important Note for Next Developer
//! - NEVER truncate or rewrite the `.memchain` file except through the
//!   exact torn-tail recovery path after a complete semantic scan.
//! - Tag bytes 0x01/0x02 are stable contracts.
//! - `last_block_hash` / `last_block_height` are in-memory caches
//!   populated during `replay()` and updated by `append_block()`.
//! - [BOUNDED-AOF-RECORDS 2026-07-24 by Codex] Keep the write and replay
//!   ceilings symmetric. Never replace checked record lengths with lossy casts
//!   or deserialize length-prefixed records with unbounded codec defaults.
//! - [AOF-TORN-TAIL-RECOVERY 2026-07-24 by Codex] Repair only an incomplete
//!   physical tail. Do not silently truncate or skip a complete corrupt record;
//!   operators must retain evidence and choose an explicit recovery action.
//! - [AOF-SEMANTIC-INTEGRITY 2026-07-24 by Codex] Keep semantic validation
//!   independent from framing. The existing bytes are a rollback contract;
//!   content IDs, Merkle roots, and ancestry provide compatible integrity
//!   checks while a future frame migration is designed separately.
//!
//! ## Last Modified
//! v0.8.0-SemanticIntegrity - Added semantic scanning and aggregate verification
//! v0.7.0-TornTailRecovery - Added shared framing and fail-closed startup recovery
//! v0.6.0-BoundedRecords - Added symmetric record bounds and canonical replay decoding
//! v0.2.0 - Initial AOF writer for `MemChain` fact persistence
//! v0.5.0 - 🌟 Added Block storage, record tags, chain state tracking

// [AOF-LEGACY-WIRE 2026-07-24 by Codex] The deprecated Block type is the
// persisted v0.5 wire contract. Migrating it in place would make rollback
// binaries unable to read the AOF, so deprecation is intentionally scoped here.
#![allow(deprecated)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::Options;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{merkle_root, Block, BlockHeader, Fact, GENESIS_PREV_HASH};

// ============================================
// Constants
// ============================================

/// Default ledger filename.
pub const DEFAULT_AOF_FILENAME: &str = ".memchain";

/// Length prefix size (u32 LE).
const LENGTH_PREFIX_SIZE: usize = 4;

/// Maximum canonical Fact or Block payload accepted on disk.
const MAX_AOF_RECORD_BYTES: u64 = 10 * 1024 * 1024;

/// Record tag: Fact.
const TAG_FACT: u8 = 0x01;

/// Record tag: Block.
const TAG_BLOCK: u8 = 0x02;

/// Privacy-safe result of a read-only AOF integrity scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AofVerificationReport {
    /// Total bytes observed when the scan started.
    pub file_bytes: u64,
    /// Bytes belonging to complete, semantically valid records.
    pub valid_bytes: u64,
    /// Number of standalone Fact records.
    pub fact_records: u64,
    /// Number of Block records.
    pub block_records: u64,
    /// Height of the last valid Block, or zero when no Block exists.
    pub last_block_height: u64,
    /// Incomplete physical tail bytes; zero means the file ended cleanly.
    pub torn_tail_bytes: u64,
}

impl AofVerificationReport {
    /// Returns true when every byte belongs to a complete valid record.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.torn_tail_bytes == 0 && self.valid_bytes == self.file_bytes
    }
}

#[derive(Clone, Copy)]
enum AofRecordKind {
    Fact,
    Block,
}

struct AofRecordFrame {
    kind: AofRecordKind,
    payload: Vec<u8>,
    next_offset: u64,
}

enum AofFrameRead {
    Complete(AofRecordFrame),
    CleanEof,
    TornTail,
}

struct AofScanResult {
    facts: Vec<Fact>,
    last_block: Option<BlockHeader>,
    report: AofVerificationReport,
}

struct AofRecovery {
    recovered_bytes: u64,
    last_block: Option<BlockHeader>,
}

// ============================================
// AofWriter
// ============================================

/// Append-only file writer for `MemChain` Fact and Block persistence.
///
/// # Thread Safety
/// `AofWriter` is **not** `Sync` by itself (it wraps `BufWriter<File>`).
/// Wrap it in `tokio::sync::Mutex` if shared across tasks.
pub struct AofWriter {
    /// Buffered writer for append operations.
    writer: BufWriter<File>,
    /// Path to the ledger file.
    path: PathBuf,
    /// Number of facts written in this session.
    write_count: AtomicU64,
    /// 🌟 Hash of the most recent block (for chain linking).
    last_block_hash: RwLock<[u8; 32]>,
    /// 🌟 Height of the most recent block (0 if no blocks yet).
    last_block_height: RwLock<u64>,
}

impl AofWriter {
    /// Opens (or creates) the append-only ledger file.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be inspected or opened, or when a
    /// complete existing record fails framing or semantic integrity checks.
    pub async fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let recovery = repair_torn_tail(&path).await?;

        if recovery.recovered_bytes > 0 {
            warn!(
                path = %path.display(),
                recovered_bytes = recovery.recovered_bytes,
                "[AOF] Recovered incomplete physical tail before opening for append"
            );
        }
        let last_block_hash = recovery
            .last_block
            .as_ref()
            .map_or(GENESIS_PREV_HASH, BlockHeader::hash);
        let last_block_height = recovery
            .last_block
            .as_ref()
            .map_or(0, |header| header.height);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;

        info!(
            path = %path.display(),
            "[AOF] Ledger file opened for append"
        );

        Ok(Self {
            writer: BufWriter::new(file),
            path,
            write_count: AtomicU64::new(0),
            last_block_hash: RwLock::new(last_block_hash),
            last_block_height: RwLock::new(last_block_height),
        })
    }

    /// Appends a single Fact to the ledger file.
    ///
    /// Record format: `[TAG_FACT (1)][length: u32 LE (4)][bincode payload]`
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when the Fact identifier does not match its
    /// content, or an I/O error when encoding, writing, or flushing fails.
    pub async fn append_fact(&mut self, fact: &Fact) -> std::io::Result<()> {
        validate_fact_integrity(fact).map_err(invalid_append_error)?;
        let payload_len = self.append_record(TAG_FACT, fact).await?;

        debug!(
            fact_id = hex::encode(fact.fact_id),
            payload_len, "[AOF] ✅ Fact appended to ledger"
        );

        Ok(())
    }

    /// 🌟 Appends a complete Block to the ledger file.
    ///
    /// Record format: `[TAG_BLOCK (1)][length: u32 LE (4)][bincode payload]`
    ///
    /// Also updates the in-memory `last_block_hash` and `last_block_height`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` when Fact identifiers, the Merkle root, height,
    /// or previous Block hash are inconsistent, or an I/O error on persistence.
    pub async fn append_block(&mut self, block: &Block) -> std::io::Result<()> {
        validate_block_payload(block).map_err(invalid_append_error)?;
        let expected_height = self
            .last_block_height()
            .checked_add(1)
            .ok_or_else(|| invalid_append_error("Block height overflow"))?;
        validate_expected_parent(block, expected_height, self.last_block_hash())
            .map_err(invalid_append_error)?;
        self.append_record(TAG_BLOCK, block).await?;

        // Update chain state
        {
            let mut hash = self.last_block_hash.write();
            *hash = block.header.hash();
        }
        {
            let mut height = self.last_block_height.write();
            *height = block.header.height;
        }
        info!(
            height = block.header.height,
            facts = block.fact_count(),
            hash = hex::encode(block.header.hash()),
            "[AOF] ✅ Block appended to ledger"
        );

        Ok(())
    }

    /// Serializes and appends one canonical bounded record.
    async fn append_record<T: Serialize + Sync>(
        &mut self,
        tag: u8,
        value: &T,
    ) -> std::io::Result<usize> {
        let payload = serialize_record(value)?;
        let length = u32::try_from(payload.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "AOF record length does not fit u32",
            )
        })?;

        self.writer.write_all(&[tag]).await?;
        self.writer.write_all(&length.to_le_bytes()).await?;
        self.writer.write_all(&payload).await?;
        self.writer.flush().await?;
        self.write_count.fetch_add(1, Ordering::Relaxed);
        Ok(payload.len())
    }

    /// Replays all records from the ledger file.
    ///
    /// Returns Facts (for `MemPool` rehydration). Blocks are processed
    /// internally to rebuild `last_block_hash` / `last_block_height`.
    ///
    /// Supports both:
    /// - **v0.5.0 format**: `[tag][u32 LE length][payload]`
    /// - **v0.3.0 legacy format**: `[u32 LE length][payload]` (tag-less, all Facts)
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or when a complete record
    /// fails framing, decoding, content-address, Merkle, or ancestry checks.
    pub async fn replay(
        path: impl AsRef<Path>,
    ) -> std::io::Result<(Vec<Fact>, Option<BlockHeader>)> {
        let path = path.as_ref();
        let scan = scan_aof(path, true).await?;
        if scan.report.torn_tail_bytes > 0 {
            warn!(
                offset = scan.report.valid_bytes,
                file_len = scan.report.file_bytes,
                torn_tail_bytes = scan.report.torn_tail_bytes,
                "[AOF] Incomplete physical tail found during replay; append-open will repair it"
            );
        }

        info!(
            path = %path.display(),
            facts_loaded = scan.facts.len(),
            last_block_height = scan.last_block.as_ref().map_or(0, |b| b.height),
            "[AOF] ✅ Ledger replay complete"
        );

        Ok((scan.facts, scan.last_block))
    }

    /// Performs a read-only framing, content-address, Merkle, and ancestry scan.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or when a complete record
    /// fails framing, decoding, content-address, Merkle, or ancestry checks.
    pub async fn verify(path: impl AsRef<Path>) -> std::io::Result<AofVerificationReport> {
        Ok(scan_aof(path.as_ref(), false).await?.report)
    }

    /// Returns the number of records written in this session.
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.write_count.load(Ordering::Relaxed)
    }

    /// Returns the path of the ledger file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 🌟 Returns the hash of the last block (for chain linking).
    /// Returns `GENESIS_PREV_HASH` if no blocks have been written.
    #[must_use]
    pub fn last_block_hash(&self) -> [u8; 32] {
        *self.last_block_hash.read()
    }

    /// 🌟 Returns the height of the last block.
    /// Returns 0 if no blocks have been written.
    #[must_use]
    pub fn last_block_height(&self) -> u64 {
        *self.last_block_height.read()
    }

    /// 🌟 Sets the chain state from replay results.
    /// Called by `init_memchain` after `replay()`.
    pub fn set_chain_state(&self, last_block: Option<&BlockHeader>) {
        if let Some(header) = last_block {
            *self.last_block_hash.write() = header.hash();
            *self.last_block_height.write() = header.height;
            info!(
                height = header.height,
                hash = hex::encode(header.hash()),
                "[AOF] Chain state restored from replay"
            );
        }
    }
}

enum DecodedAofRecord {
    Fact(Fact),
    Block(Block),
}

/// Scans framing and semantic integrity without mutating the AOF.
async fn scan_aof(path: &Path, retain_facts: bool) -> std::io::Result<AofScanResult> {
    if !path.exists() {
        info!(
            path = %path.display(),
            "[AOF] No existing ledger file, starting fresh"
        );
        return Ok(AofScanResult {
            facts: Vec::new(),
            last_block: None,
            report: AofVerificationReport {
                file_bytes: 0,
                valid_bytes: 0,
                fact_records: 0,
                block_records: 0,
                last_block_height: 0,
                torn_tail_bytes: 0,
            },
        });
    }

    let file_len = tokio::fs::metadata(path).await?.len();
    let file = File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut facts = Vec::new();
    let mut last_block: Option<BlockHeader> = None;
    let mut fact_records = 0_u64;
    let mut block_records = 0_u64;
    let mut offset = 0_u64;
    let mut torn_tail_bytes = 0_u64;

    loop {
        match read_next_frame(&mut reader, offset, file_len).await? {
            AofFrameRead::Complete(frame) => {
                match decode_frame(frame.kind, &frame.payload, offset)? {
                    DecodedAofRecord::Fact(fact) => {
                        validate_fact_integrity(&fact)
                            .map_err(|detail| invalid_record_error(offset, detail))?;
                        fact_records = fact_records.saturating_add(1);
                        if retain_facts {
                            facts.push(fact);
                        }
                    }
                    DecodedAofRecord::Block(block) => {
                        validate_block_payload(&block)
                            .map_err(|detail| invalid_record_error(offset, detail))?;
                        if let Some(previous) = last_block.as_ref() {
                            let expected_height =
                                previous.height.checked_add(1).ok_or_else(|| {
                                    invalid_record_error(offset, "Block height overflow")
                                })?;
                            validate_expected_parent(&block, expected_height, previous.hash())
                                .map_err(|detail| invalid_record_error(offset, detail))?;
                        }
                        block_records = block_records.saturating_add(1);
                        last_block = Some(block.header);
                    }
                }
                offset = frame.next_offset;
            }
            AofFrameRead::CleanEof => break,
            AofFrameRead::TornTail => {
                torn_tail_bytes = file_len.saturating_sub(offset);
                break;
            }
        }
    }

    Ok(AofScanResult {
        facts,
        report: AofVerificationReport {
            file_bytes: file_len,
            valid_bytes: offset,
            fact_records,
            block_records,
            last_block_height: last_block.as_ref().map_or(0, |header| header.height),
            torn_tail_bytes,
        },
        last_block,
    })
}

/// Reads exactly one tagged or legacy frame without interpreting its payload.
async fn read_next_frame<R: AsyncRead + Unpin>(
    reader: &mut R,
    offset: u64,
    file_len: u64,
) -> std::io::Result<AofFrameRead> {
    if offset == file_len {
        return Ok(AofFrameRead::CleanEof);
    }
    if offset > file_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "AOF parser offset exceeds file length",
        ));
    }

    let mut marker = [0u8; 1];
    if !read_exact_or_torn(reader, &mut marker).await? {
        return Ok(AofFrameRead::TornTail);
    }

    let (kind, header_len, length) = if marker[0] == TAG_FACT || marker[0] == TAG_BLOCK {
        let mut length_bytes = [0u8; LENGTH_PREFIX_SIZE];
        if !read_exact_or_torn(reader, &mut length_bytes).await? {
            return Ok(AofFrameRead::TornTail);
        }
        let kind = if marker[0] == TAG_FACT {
            AofRecordKind::Fact
        } else {
            AofRecordKind::Block
        };
        (
            kind,
            1_u64 + LENGTH_PREFIX_SIZE as u64,
            u64::from(u32::from_le_bytes(length_bytes)),
        )
    } else {
        let mut remaining_length = [0u8; LENGTH_PREFIX_SIZE - 1];
        if !read_exact_or_torn(reader, &mut remaining_length).await? {
            return Ok(AofFrameRead::TornTail);
        }
        (
            AofRecordKind::Fact,
            LENGTH_PREFIX_SIZE as u64,
            u64::from(u32::from_le_bytes([
                marker[0],
                remaining_length[0],
                remaining_length[1],
                remaining_length[2],
            ])),
        )
    };

    if length > MAX_AOF_RECORD_BYTES {
        return Err(invalid_record_error(
            offset,
            format!("record length {length} exceeds {MAX_AOF_RECORD_BYTES} bytes"),
        ));
    }

    let next_offset = offset
        .checked_add(header_len)
        .and_then(|value| value.checked_add(length))
        .ok_or_else(|| invalid_record_error(offset, "record offset overflow"))?;
    if next_offset > file_len {
        return Ok(AofFrameRead::TornTail);
    }

    let payload_len = usize::try_from(length)
        .map_err(|_| invalid_record_error(offset, "record length does not fit usize"))?;
    let mut payload = vec![0u8; payload_len];
    if !read_exact_or_torn(reader, &mut payload).await? {
        return Ok(AofFrameRead::TornTail);
    }

    Ok(AofFrameRead::Complete(AofRecordFrame {
        kind,
        payload,
        next_offset,
    }))
}

async fn read_exact_or_torn<R: AsyncRead + Unpin>(
    reader: &mut R,
    buffer: &mut [u8],
) -> std::io::Result<bool> {
    match reader.read_exact(buffer).await {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error),
    }
}

fn decode_frame(
    kind: AofRecordKind,
    payload: &[u8],
    offset: u64,
) -> std::io::Result<DecodedAofRecord> {
    match kind {
        AofRecordKind::Fact => deserialize_record::<Fact>(payload)
            .map(DecodedAofRecord::Fact)
            .map_err(|error| {
                invalid_record_error(offset, format!("invalid Fact payload: {error}"))
            }),
        AofRecordKind::Block => deserialize_record::<Block>(payload)
            .map(DecodedAofRecord::Block)
            .map_err(|error| {
                invalid_record_error(offset, format!("invalid Block payload: {error}"))
            }),
    }
}

/// Removes only a partial physical tail left by an interrupted append.
async fn repair_torn_tail(path: &Path) -> std::io::Result<AofRecovery> {
    let scan = scan_aof(path, false).await?;
    if scan.report.torn_tail_bytes > 0 {
        let file = OpenOptions::new().read(true).write(true).open(path).await?;
        file.set_len(scan.report.valid_bytes).await?;
        file.sync_data().await?;
    }

    Ok(AofRecovery {
        recovered_bytes: scan.report.torn_tail_bytes,
        last_block: scan.last_block,
    })
}

fn validate_fact_integrity(fact: &Fact) -> Result<(), String> {
    if fact.verify_id() {
        Ok(())
    } else {
        Err("Fact content does not match fact_id".to_string())
    }
}

fn validate_block_payload(block: &Block) -> Result<(), String> {
    for (index, fact) in block.facts.iter().enumerate() {
        validate_fact_integrity(fact)
            .map_err(|detail| format!("Block Fact index {index} is invalid: {detail}"))?;
    }

    let fact_ids: Vec<[u8; 32]> = block.facts.iter().map(|fact| fact.fact_id).collect();
    if merkle_root(&fact_ids) != block.header.merkle_root {
        return Err("Block Merkle root does not match contained Facts".to_string());
    }

    Ok(())
}

fn validate_expected_parent(
    block: &Block,
    expected_height: u64,
    expected_prev_hash: [u8; 32],
) -> Result<(), String> {
    if block.header.height != expected_height {
        return Err(format!(
            "Block height {} does not follow expected height {expected_height}",
            block.header.height
        ));
    }
    if block.header.prev_block_hash != expected_prev_hash {
        return Err("Block previous hash does not match accepted AOF prefix".to_string());
    }
    Ok(())
}

fn invalid_append_error(detail: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!("refusing invalid AOF append: {detail}"),
    )
}

fn invalid_record_error(offset: u64, detail: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("AOF corruption at offset {offset}: {detail}"),
    )
}

/// Serializes one AOF payload using the established fixed-integer wire format
/// and the same ceiling enforced by replay.
fn serialize_record<T: Serialize>(value: &T) -> std::io::Result<Vec<u8>> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_AOF_RECORD_BYTES)
        .serialize(value)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

/// Decodes one complete canonical AOF payload under the record ceiling.
fn deserialize_record<T: DeserializeOwned>(payload: &[u8]) -> Result<T, bincode::Error> {
    let payload_len = u64::try_from(payload.len()).unwrap_or(u64::MAX);
    if payload_len > MAX_AOF_RECORD_BYTES {
        return Err(Box::new(bincode::ErrorKind::SizeLimit));
    }

    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_AOF_RECORD_BYTES)
        .reject_trailing_bytes()
        .deserialize(payload)
}

impl std::fmt::Debug for AofWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AofWriter")
            .field("path", &self.path)
            .field("write_count", &self.write_count())
            .field("last_block_height", &*self.last_block_height.read())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::ledger::{
        merkle_root, Block, BlockHeader, BLOCK_TYPE_NORMAL, GENESIS_PREV_HASH,
    };
    use tempfile::tempdir;

    fn make_fact(ts: u64, subject: &str) -> Fact {
        Fact::new(ts, subject.into(), "pred".into(), "obj".into())
    }

    fn make_linked_block(height: u64, previous_hash: [u8; 32], facts: Vec<Fact>) -> Block {
        let leaf_ids: Vec<[u8; 32]> = facts.iter().map(|f| f.fact_id).collect();
        let header = BlockHeader {
            height,
            timestamp: 1_700_000_000,
            prev_block_hash: previous_hash,
            merkle_root: merkle_root(&leaf_ids),
            block_type: BLOCK_TYPE_NORMAL,
        };
        Block::new(header, facts)
    }

    fn make_block(height: u64, facts: Vec<Fact>) -> Block {
        make_linked_block(height, GENESIS_PREV_HASH, facts)
    }

    fn tagged_frame<T: Serialize>(tag: u8, value: &T) -> Vec<u8> {
        let payload = serialize_record(value).expect("serialize frame");
        let length = u32::try_from(payload.len()).expect("test frame length");
        let mut frame = Vec::with_capacity(1 + LENGTH_PREFIX_SIZE + payload.len());
        frame.push(tag);
        frame.extend_from_slice(&length.to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }

    #[tokio::test]
    async fn test_append_fact_and_replay() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);

        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer
                .append_fact(&make_fact(100, "first"))
                .await
                .expect("append");
            writer
                .append_fact(&make_fact(200, "second"))
                .await
                .expect("append");
        }

        let (facts, last_block) = AofWriter::replay(&path).await.expect("replay");
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].subject, "first");
        assert_eq!(facts[1].subject, "second");
        assert!(last_block.is_none());
    }

    #[test]
    fn test_record_codec_preserves_wire_bytes_and_rejects_trailing_data() {
        let fact = make_fact(100, "wire-compatible");
        let legacy = bincode::serialize(&fact).expect("legacy serialization");
        let canonical = serialize_record(&fact).expect("bounded serialization");

        assert_eq!(canonical, legacy);
        assert_eq!(
            deserialize_record::<Fact>(&canonical)
                .expect("canonical payload")
                .fact_id,
            fact.fact_id
        );

        let mut trailing = canonical;
        trailing.push(0);
        assert!(
            deserialize_record::<Fact>(&trailing).is_err(),
            "length-prefixed AOF records must contain exactly one value"
        );
    }

    #[tokio::test]
    async fn test_append_rejects_record_larger_than_replay_ceiling() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let oversized = make_fact(
            100,
            &"x".repeat(MAX_AOF_RECORD_BYTES as usize + LENGTH_PREFIX_SIZE),
        );
        let mut writer = AofWriter::open(&path).await.expect("open");

        let error = writer
            .append_fact(&oversized)
            .await
            .expect_err("oversized record must be rejected");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(writer.write_count(), 0);
    }

    #[tokio::test]
    async fn test_append_rejects_fact_with_mismatched_content_id() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let mut tampered = make_fact(100, "original");
        tampered.subject = "tampered".to_string();
        let mut writer = AofWriter::open(&path).await.expect("open");

        let error = writer
            .append_fact(&tampered)
            .await
            .expect_err("mismatched content id must be rejected");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(writer.write_count(), 0);
        assert_eq!(tokio::fs::metadata(&path).await.expect("metadata").len(), 0);
    }

    #[tokio::test]
    async fn test_open_repairs_torn_tail_before_future_appends() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);

        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer
                .append_fact(&make_fact(100, "before-crash"))
                .await
                .expect("append");
        }
        let valid_len = tokio::fs::metadata(&path).await.expect("metadata").len();

        {
            let mut file = OpenOptions::new()
                .append(true)
                .open(&path)
                .await
                .expect("append torn tail");
            file.write_all(&[TAG_FACT, 8, 0])
                .await
                .expect("write torn tail");
            file.flush().await.expect("flush torn tail");
        }
        assert!(tokio::fs::metadata(&path).await.expect("metadata").len() > valid_len);

        let mut writer = AofWriter::open(&path).await.expect("repair and reopen");
        assert_eq!(
            tokio::fs::metadata(&path).await.expect("metadata").len(),
            valid_len
        );
        writer
            .append_fact(&make_fact(200, "after-recovery"))
            .await
            .expect("append after recovery");
        drop(writer);

        let (facts, last_block) = AofWriter::replay(&path).await.expect("replay");
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].subject, "before-crash");
        assert_eq!(facts[1].subject, "after-recovery");
        assert!(last_block.is_none());
    }

    #[tokio::test]
    async fn test_open_rejects_complete_corrupt_record_without_truncating() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let corrupt_record = [TAG_FACT, 1, 0, 0, 0, 0];
        tokio::fs::write(&path, corrupt_record)
            .await
            .expect("write corrupt record");

        let error = AofWriter::open(&path)
            .await
            .expect_err("complete corruption must fail closed");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(
            tokio::fs::metadata(&path).await.expect("metadata").len(),
            corrupt_record.len() as u64
        );
    }

    #[tokio::test]
    async fn test_verify_rejects_semantically_tampered_fact() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let mut tampered = make_fact(100, "original");
        tampered.subject = "tampered".to_string();
        tokio::fs::write(&path, tagged_frame(TAG_FACT, &tampered))
            .await
            .expect("write tampered Fact");

        let error = AofWriter::verify(&path)
            .await
            .expect_err("semantic tampering must fail closed");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("does not match fact_id"));
    }

    #[tokio::test]
    async fn test_verify_rejects_block_merkle_tampering() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let mut block = make_block(1, vec![make_fact(100, "fact")]);
        block.header.merkle_root[0] ^= 0x80;
        tokio::fs::write(&path, tagged_frame(TAG_BLOCK, &block))
            .await
            .expect("write tampered Block");

        let error = AofWriter::verify(&path)
            .await
            .expect_err("Merkle tampering must fail closed");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("Merkle root"));
    }

    #[tokio::test]
    async fn test_verify_rejects_broken_block_ancestry() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let first = make_block(1, vec![make_fact(100, "first")]);
        let second = make_linked_block(2, GENESIS_PREV_HASH, vec![make_fact(200, "second")]);
        let mut bytes = tagged_frame(TAG_BLOCK, &first);
        bytes.extend_from_slice(&tagged_frame(TAG_BLOCK, &second));
        tokio::fs::write(&path, bytes)
            .await
            .expect("write broken chain");

        let error = AofWriter::verify(&path)
            .await
            .expect_err("broken ancestry must fail closed");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("previous hash"));
    }

    #[tokio::test]
    async fn test_verify_reports_torn_tail_without_mutating_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let frame = tagged_frame(TAG_FACT, &make_fact(100, "valid"));
        let mut bytes = frame.clone();
        bytes.extend_from_slice(&[TAG_BLOCK, 8, 0]);
        tokio::fs::write(&path, &bytes)
            .await
            .expect("write torn file");

        let report = AofWriter::verify(&path).await.expect("verify");

        assert!(!report.is_clean());
        assert_eq!(report.file_bytes, bytes.len() as u64);
        assert_eq!(report.valid_bytes, frame.len() as u64);
        assert_eq!(report.fact_records, 1);
        assert_eq!(report.block_records, 0);
        assert_eq!(report.torn_tail_bytes, 3);
        assert_eq!(
            tokio::fs::metadata(&path).await.expect("metadata").len(),
            bytes.len() as u64,
            "read-only verification must not repair the file"
        );
    }

    #[tokio::test]
    async fn test_append_block_and_replay() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);

        let fact = make_fact(100, "test");
        let block = make_block(1, vec![fact.clone()]);

        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer.append_fact(&fact).await.expect("append fact");
            writer.append_block(&block).await.expect("append block");

            assert_eq!(writer.last_block_height(), 1);
            assert_eq!(writer.last_block_hash(), block.header.hash());
        }

        let (facts, last_block) = AofWriter::replay(&path).await.expect("replay");
        assert_eq!(facts.len(), 1);
        let header = last_block.expect("should have block");
        assert_eq!(header.height, 1);
    }

    #[tokio::test]
    async fn test_reopen_restores_chain_state_without_manual_setter() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        let first = make_block(1, vec![make_fact(100, "first")]);
        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer.append_block(&first).await.expect("append first");
        }

        let second = make_linked_block(2, first.header.hash(), vec![make_fact(200, "second")]);
        {
            let mut writer = AofWriter::open(&path).await.expect("reopen");
            assert_eq!(writer.last_block_height(), 1);
            assert_eq!(writer.last_block_hash(), first.header.hash());
            writer.append_block(&second).await.expect("append second");
        }

        let report = AofWriter::verify(&path).await.expect("verify");
        assert!(report.is_clean());
        assert_eq!(report.block_records, 2);
        assert_eq!(report.last_block_height, 2);
    }

    #[tokio::test]
    async fn test_replay_empty_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);
        {
            let _w = AofWriter::open(&path).await.expect("open");
        }

        let (facts, last_block) = AofWriter::replay(&path).await.expect("replay");
        assert!(facts.is_empty());
        assert!(last_block.is_none());
    }

    #[tokio::test]
    async fn test_replay_nonexistent_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("does_not_exist");
        let (facts, last_block) = AofWriter::replay(&path).await.expect("replay");
        assert!(facts.is_empty());
        assert!(last_block.is_none());
    }
}
