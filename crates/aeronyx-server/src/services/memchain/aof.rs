// ============================================
// File: crates/aeronyx-server/src/services/memchain/aof.rs
// ============================================
//! # AOF — Append-Only File Storage
//!
//! ## Creation Reason
//! Provides durable, crash-safe persistence for MemChain Facts using a
//! simple append-only binary file (`.memchain`). Inspired by Redis AOF
//! and blockchain ledger design — records are never modified or deleted.
//!
//! ## Main Functionality
//! - `AofWriter::open()` — create or open the ledger file
//! - `AofWriter::append_fact()` — serialise and append a single Fact
//! - `AofWriter::replay()` — read all Facts from disk on startup
//! - `AofWriter::fact_count()` — number of facts written in this session
//!
//! ## On-Disk Record Format
//! Each record is length-prefixed for safe streaming reads:
//! ```text
//! ┌──────────────────┬────────────────────────────┐
//! │  length: u32 LE  │  bincode(Fact) payload     │
//! └──────────────────┴────────────────────────────┘
//! ```
//! The 4-byte length prefix allows the reader to skip or validate
//! records without parsing the full bincode payload.
//!
//! ## Crash Safety
//! - Writes are followed by `flush()` to push data to the OS buffer.
//! - If a crash occurs mid-write, the trailing partial record will have
//!   an incorrect length prefix and will be detected + skipped during
//!   `replay()`.
//!
//! ## Dependencies
//! - `tokio::fs` for async file I/O
//! - `bincode` for serialisation
//! - `aeronyx_core::ledger::Fact`
//!
//! ## ⚠️ Important Note for Next Developer
//! - NEVER truncate or rewrite the `.memchain` file.
//! - The record format (u32 LE length + bincode) is a stable contract.
//!   Changing it would break replay of existing ledger files.
//! - `replay()` is intentionally tolerant of trailing garbage — this
//!   handles crash-truncated writes gracefully.
//! - For production, consider adding `fsync` (via `file.sync_data()`)
//!   after critical writes. Current impl uses `flush()` for performance.
//!
//! ## Last Modified
//! v0.2.0 - Initial AOF writer for MemChain fact persistence

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tracing::{debug, error, info, warn};

use aeronyx_core::ledger::Fact;

// ============================================
// Constants
// ============================================

/// Default ledger filename.
pub const DEFAULT_AOF_FILENAME: &str = ".memchain";

/// Length prefix size (u32 LE).
const LENGTH_PREFIX_SIZE: usize = 4;

// ============================================
// AofWriter
// ============================================

/// Append-only file writer for MemChain Fact persistence.
///
/// # Thread Safety
/// `AofWriter` is **not** `Sync` by itself (it wraps `BufWriter<File>`).
/// Wrap it in `tokio::sync::Mutex` if shared across tasks (which the
/// `MemPool` integration does).
pub struct AofWriter {
    /// Buffered writer for append operations.
    writer: BufWriter<File>,
    /// Path to the ledger file (for logging / diagnostics).
    path: PathBuf,
    /// Number of facts written in this session.
    write_count: AtomicU64,
}

impl AofWriter {
    /// Opens (or creates) the append-only ledger file.
    ///
    /// # Arguments
    /// * `path` - Path to the `.memchain` file.
    ///
    /// # Errors
    /// Returns an IO error if the file cannot be opened or created.
    pub async fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();

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
        })
    }

    /// Appends a single Fact to the ledger file.
    ///
    /// The record is written as:
    /// ```text
    /// [length: u32 LE][bincode payload]
    /// ```
    ///
    /// # Errors
    /// Returns an IO error on write/flush failure, or a serialisation
    /// error if bincode fails (should never happen for valid Facts).
    pub async fn append_fact(&mut self, fact: &Fact) -> std::io::Result<()> {
        // Serialise
        let payload = bincode::serialize(fact)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let length = payload.len() as u32;

        // Write length prefix + payload
        self.writer.write_all(&length.to_le_bytes()).await?;
        self.writer.write_all(&payload).await?;
        self.writer.flush().await?;

        self.write_count.fetch_add(1, Ordering::Relaxed);

        debug!(
            fact_id = hex::encode(fact.fact_id),
            payload_len = payload.len(),
            "[AOF] ✅ Fact appended to ledger"
        );

        Ok(())
    }

    /// Replays all Facts from the ledger file.
    ///
    /// This is used at startup to rehydrate the MemPool from disk.
    /// Partial / corrupted trailing records are logged and skipped.
    ///
    /// # Arguments
    /// * `path` - Path to the `.memchain` file.
    ///
    /// # Returns
    /// Vector of successfully parsed Facts, in file order.
    ///
    /// # Errors
    /// Returns an IO error if the file cannot be opened.
    pub async fn replay(path: impl AsRef<Path>) -> std::io::Result<Vec<Fact>> {
        let path = path.as_ref();

        if !path.exists() {
            info!(
                path = %path.display(),
                "[AOF] No existing ledger file, starting fresh"
            );
            return Ok(Vec::new());
        }

        let file = File::open(path).await?;
        let mut reader = BufReader::new(file);
        let mut facts = Vec::new();
        let mut offset: u64 = 0;

        loop {
            // Read length prefix
            let mut len_buf = [0u8; LENGTH_PREFIX_SIZE];
            match reader.read_exact(&mut len_buf).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // Normal end of file
                    break;
                }
                Err(e) => {
                    warn!(
                        offset = offset,
                        error = %e,
                        "[AOF] ⚠️ Error reading length prefix, stopping replay"
                    );
                    break;
                }
            }

            let length = u32::from_le_bytes(len_buf) as usize;

            // Sanity check: reject absurdly large records (> 10 MB)
            if length > 10 * 1024 * 1024 {
                warn!(
                    offset = offset,
                    length = length,
                    "[AOF] ⚠️ Absurd record length, stopping replay"
                );
                break;
            }

            // Read payload
            let mut payload = vec![0u8; length];
            match reader.read_exact(&mut payload).await {
                Ok(()) => {}
                Err(e) => {
                    warn!(
                        offset = offset,
                        error = %e,
                        "[AOF] ⚠️ Truncated record at end of file, skipping"
                    );
                    break;
                }
            }

            // Deserialise
            match bincode::deserialize::<Fact>(&payload) {
                Ok(fact) => {
                    facts.push(fact);
                }
                Err(e) => {
                    warn!(
                        offset = offset,
                        error = %e,
                        "[AOF] ⚠️ Failed to deserialise record, skipping"
                    );
                }
            }

            offset += (LENGTH_PREFIX_SIZE + length) as u64;
        }

        info!(
            path = %path.display(),
            facts_loaded = facts.len(),
            "[AOF] ✅ Ledger replay complete"
        );

        Ok(facts)
    }

    /// Returns the number of facts written in this session.
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.write_count.load(Ordering::Relaxed)
    }

    /// Returns the path of the ledger file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl std::fmt::Debug for AofWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AofWriter")
            .field("path", &self.path)
            .field("write_count", &self.write_count())
            .finish()
    }
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn make_fact(ts: u64, subject: &str) -> Fact {
        Fact::new(ts, subject.into(), "pred".into(), "obj".into())
    }

    #[tokio::test]
    async fn test_append_and_replay() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);

        // Write two facts
        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer.append_fact(&make_fact(100, "first")).await.expect("append 1");
            writer.append_fact(&make_fact(200, "second")).await.expect("append 2");
            assert_eq!(writer.write_count(), 2);
        }

        // Replay and verify
        let facts = AofWriter::replay(&path).await.expect("replay");
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].subject, "first");
        assert_eq!(facts[1].subject, "second");
    }

    #[tokio::test]
    async fn test_replay_empty_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);

        // Create empty file
        {
            let _writer = AofWriter::open(&path).await.expect("open");
        }

        let facts = AofWriter::replay(&path).await.expect("replay");
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn test_replay_nonexistent_file() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("does_not_exist.memchain");

        let facts = AofWriter::replay(&path).await.expect("replay");
        assert!(facts.is_empty());
    }

    #[tokio::test]
    async fn test_append_then_append_more() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(DEFAULT_AOF_FILENAME);

        // First session
        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer.append_fact(&make_fact(100, "a")).await.expect("append");
        }

        // Second session (appends to same file)
        {
            let mut writer = AofWriter::open(&path).await.expect("open");
            writer.append_fact(&make_fact(200, "b")).await.expect("append");
        }

        let facts = AofWriter::replay(&path).await.expect("replay");
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[0].subject, "a");
        assert_eq!(facts[1].subject, "b");
    }
}
