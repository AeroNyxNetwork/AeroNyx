// ============================================
// File: crates/aeronyx-server/src/services/memchain/aof.rs
// ============================================
//! # AOF — Append-Only File Storage
//!
//! ## Creation Reason
//! Provides durable, crash-safe persistence for MemChain Facts and Blocks
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
//! - Trailing partial records are detected + skipped during `replay()`.
//!
//! ## ⚠️ Important Note for Next Developer
//! - NEVER truncate or rewrite the `.memchain` file.
//! - Tag bytes 0x01/0x02 are stable contracts.
//! - `last_block_hash` / `last_block_height` are in-memory caches
//!   populated during `replay()` and updated by `append_block()`.
//! - [BOUNDED-AOF-RECORDS 2026-07-24 by Codex] Keep the write and replay
//!   ceilings symmetric. Never replace checked record lengths with lossy casts
//!   or deserialize length-prefixed records with unbounded codec defaults.
//! - [AOF-TORN-TAIL-RECOVERY 2026-07-24 by Codex] Repair only an incomplete
//!   physical tail. Do not silently truncate or skip a complete corrupt record;
//!   operators must retain evidence and choose an explicit recovery action.
//!
//! ## Last Modified
//! v0.7.0-TornTailRecovery - Added shared framing and fail-closed startup recovery
//! v0.6.0-BoundedRecords - Added symmetric record bounds and canonical replay decoding
//! v0.2.0 - Initial AOF writer for MemChain fact persistence
//! v0.5.0 - 🌟 Added Block storage, record tags, chain state tracking

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bincode::Options;
use parking_lot::RwLock;
use serde::{de::DeserializeOwned, Serialize};
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tracing::{debug, info, warn};

use aeronyx_core::ledger::{Block, BlockHeader, Fact, GENESIS_PREV_HASH};

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

// ============================================
// AofWriter
// ============================================

/// Append-only file writer for MemChain Fact and Block persistence.
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
    pub async fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let recovered_bytes = repair_torn_tail(&path).await?;

        if recovered_bytes > 0 {
            warn!(
                path = %path.display(),
                recovered_bytes,
                "[AOF] Recovered incomplete physical tail before opening for append"
            );
        }

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
            last_block_hash: RwLock::new(GENESIS_PREV_HASH),
            last_block_height: RwLock::new(0),
        })
    }

    /// Appends a single Fact to the ledger file.
    ///
    /// Record format: `[TAG_FACT (1)][length: u32 LE (4)][bincode payload]`
    pub async fn append_fact(&mut self, fact: &Fact) -> std::io::Result<()> {
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
    pub async fn append_block(&mut self, block: &Block) -> std::io::Result<()> {
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
    async fn append_record<T: Serialize>(&mut self, tag: u8, value: &T) -> std::io::Result<usize> {
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
    /// Returns Facts (for MemPool rehydration). Blocks are processed
    /// internally to rebuild `last_block_hash` / `last_block_height`.
    ///
    /// Supports both:
    /// - **v0.5.0 format**: `[tag][u32 LE length][payload]`
    /// - **v0.3.0 legacy format**: `[u32 LE length][payload]` (tag-less, all Facts)
    pub async fn replay(
        path: impl AsRef<Path>,
    ) -> std::io::Result<(Vec<Fact>, Option<BlockHeader>)> {
        let path = path.as_ref();

        if !path.exists() {
            info!(
                path = %path.display(),
                "[AOF] No existing ledger file, starting fresh"
            );
            return Ok((Vec::new(), None));
        }

        let file_len = tokio::fs::metadata(path).await?.len();
        let file = File::open(path).await?;
        let mut reader = BufReader::new(file);
        let mut facts = Vec::new();
        let mut last_block: Option<BlockHeader> = None;
        let mut offset: u64 = 0;

        loop {
            match read_next_frame(&mut reader, offset, file_len).await? {
                AofFrameRead::Complete(frame) => {
                    match decode_frame(frame.kind, &frame.payload, offset)? {
                        DecodedAofRecord::Fact(fact) => facts.push(fact),
                        DecodedAofRecord::Block(header) => last_block = Some(header),
                    }
                    offset = frame.next_offset;
                }
                AofFrameRead::CleanEof => break,
                AofFrameRead::TornTail => {
                    warn!(
                        offset = offset,
                        file_len,
                        "[AOF] Incomplete physical tail found during replay; append-open will repair it"
                    );
                    break;
                }
            }
        }

        info!(
            path = %path.display(),
            facts_loaded = facts.len(),
            last_block_height = last_block.as_ref().map_or(0, |b| b.height),
            "[AOF] ✅ Ledger replay complete"
        );

        Ok((facts, last_block))
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
    Block(BlockHeader),
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
            .map(|block| DecodedAofRecord::Block(block.header))
            .map_err(|error| {
                invalid_record_error(offset, format!("invalid Block payload: {error}"))
            }),
    }
}

/// Removes only a partial physical tail left by an interrupted append.
async fn repair_torn_tail(path: &Path) -> std::io::Result<u64> {
    if !path.exists() {
        return Ok(0);
    }

    let file_len = tokio::fs::metadata(path).await?.len();
    if file_len == 0 {
        return Ok(0);
    }

    let file = OpenOptions::new().read(true).write(true).open(path).await?;
    let mut reader = BufReader::new(file);
    let mut offset = 0_u64;

    loop {
        match read_next_frame(&mut reader, offset, file_len).await? {
            AofFrameRead::Complete(frame) => {
                decode_frame(frame.kind, &frame.payload, offset)?;
                offset = frame.next_offset;
            }
            AofFrameRead::CleanEof => return Ok(0),
            AofFrameRead::TornTail => {
                let recovered_bytes = file_len.saturating_sub(offset);
                let file = reader.into_inner();
                file.set_len(offset).await?;
                file.sync_data().await?;
                return Ok(recovered_bytes);
            }
        }
    }
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

    fn make_block(height: u64, facts: Vec<Fact>) -> Block {
        let leaf_ids: Vec<[u8; 32]> = facts.iter().map(|f| f.fact_id).collect();
        let header = BlockHeader {
            height,
            timestamp: 1_700_000_000,
            prev_block_hash: GENESIS_PREV_HASH,
            merkle_root: merkle_root(&leaf_ids),
            block_type: BLOCK_TYPE_NORMAL,
        };
        Block::new(header, facts)
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
