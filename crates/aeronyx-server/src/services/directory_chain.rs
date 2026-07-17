// ============================================================================
// File: crates/aeronyx-server/src/services/directory_chain.rs
// ============================================================================
//! # Local Directory Chain Store
//!
//! ## Creation Reason
//! Persist the signed Directory Chain protocol objects defined by
//! `aeronyx-core` without coupling an append-only ledger to the mutable
//! `PeerStore` restart cache.
//!
//! ## Main Functionality
//! - Opens one producer-scoped `SQLite` directory ledger.
//! - Appends authenticated descriptor commitments in bounded signed blocks.
//! - Retains content-addressed signed descriptor objects so any commitment can
//!   be independently resolved and re-verified later.
//! - Deduplicates exact commitments while preserving equivocation evidence.
//! - Audits every persisted block, index row, chain link, and signature at
//!   startup before the node may use the ledger.
//! - Uses WAL, full synchronous durability, foreign keys, and one immediate
//!   transaction so blocks and commitment indexes cannot advance separately.
//!
//! ## Dependencies
//! - `aeronyx_core::protocol::discovery`: canonical V1 commitment/block types.
//! - `rusqlite`: transactional local persistence using the existing bundled
//!   `SQLite` dependency.
//! - `server.rs`: lifecycle ownership and periodic reconciliation.
//!
//! ## Main Logical Flow
//! 1. Open/create the database and pin schema, chain id, and producer identity.
//! 2. Audit the complete persisted chain before returning a usable store.
//! 3. Verify and hash incoming signed descriptors locally.
//! 4. Remove exact commitments already indexed in `SQLite`.
//! 5. Build, sign, and transactionally append deterministic bounded blocks.
//! 6. Re-audit on every restart; corruption is a startup error, never repaired
//!    by deleting or silently replacing history.
//!
//! ## Important Note for Next Developer
//! - This is a local producer journal, not consensus or finality.
//! - Never add client identities, IPs, routes, message ids, payloads,
//!   ciphertext, Memory Chain content, DNS data, destinations, or browsing
//!   history to this database.
//! - Do not weaken startup audit or replace `SQLite` transactions with an
//!   in-place JSON rewrite.
//! - Peer synchronization and fork selection belong to later reviewed layers.
//!
//! ## Last Modified
//! v0.1.0-DirectoryChainStore - Initial transactional local block persistence.
// ============================================================================

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use aeronyx_core::crypto::IdentityKeyPair;
use aeronyx_core::protocol::discovery::{
    DirectoryCommitmentBlockV1, DirectoryCommitmentValidationError,
    DirectoryDescriptorCommitmentV1, SignedNodeDescriptor, AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
    MAX_DIRECTORY_COMMITMENTS_PER_BLOCK,
};
use bincode::Options;
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension, Transaction, TransactionBehavior};

/// `SQLite` schema contract for the local Directory Chain journal.
const DIRECTORY_CHAIN_STORE_SCHEMA_VERSION: i64 = 1;
/// Hard cap applied before binary block deserialization.
const MAX_DIRECTORY_BLOCK_BYTES: u64 = 64 * 1024;
/// Hard cap applied before signed descriptor object deserialization.
const MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES: u64 = 32 * 1024;
/// Bounded wait for a second process holding the `SQLite` writer lock.
const DIRECTORY_CHAIN_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Failures returned by the local Directory Chain store.
#[derive(Debug, thiserror::Error)]
pub enum DirectoryChainStoreError {
    /// Filesystem setup failed.
    #[error("directory chain filesystem error: {0}")]
    Io(#[from] std::io::Error),
    /// `SQLite` rejected a schema, query, or transaction operation.
    #[error("directory chain sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A block could not be encoded or decoded within the bounded V1 contract.
    #[error("directory chain codec error: {0}")]
    Codec(String),
    /// A signed descriptor could not become an authenticated commitment.
    #[error("directory descriptor commitment error: {0}")]
    Descriptor(String),
    /// A block failed the canonical V1 protocol checks.
    #[error("directory block validation error: {0}")]
    Block(#[from] DirectoryCommitmentValidationError),
    /// Persisted metadata or chain/index state is inconsistent.
    #[error("directory chain integrity error: {0}")]
    Integrity(String),
}

/// Result of a complete persisted-chain startup audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryChainAudit {
    /// Number of verified blocks.
    pub blocks: u64,
    /// Number of commitments exactly matched to block payloads.
    pub commitments: u64,
    /// Verified tip height, or zero for an empty store.
    pub tip_height: u64,
    /// Verified tip block hash, or all zeroes for an empty store.
    pub tip_hash: [u8; 32],
    /// Verified tip timestamp, or zero for an empty store.
    pub tip_timestamp: u64,
}

impl DirectoryChainAudit {
    const fn empty() -> Self {
        Self {
            blocks: 0,
            commitments: 0,
            tip_height: 0,
            tip_hash: [0u8; 32],
            tip_timestamp: 0,
        }
    }
}

/// Result of reconciling authenticated descriptors into local blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DirectoryChainAppendReport {
    /// New blocks committed by this transaction.
    pub blocks_appended: u64,
    /// New descriptor commitments committed by this transaction.
    pub commitments_appended: u64,
    /// Tip height after the transaction.
    pub tip_height: u64,
    /// Tip hash after the transaction.
    pub tip_hash: [u8; 32],
}

/// Single-producer transactional Directory Chain journal.
pub struct DirectoryChainStore {
    connection: Mutex<Connection>,
    path: PathBuf,
    producer: [u8; 32],
}

#[derive(Debug)]
struct StoredBlockRow {
    height: i64,
    block_hash: Vec<u8>,
    prev_block_hash: Vec<u8>,
    produced_at: i64,
    commitment_count: i64,
    block_blob: Vec<u8>,
}

#[derive(Debug, Clone)]
struct PendingDescriptor {
    commitment: DirectoryDescriptorCommitmentV1,
    descriptor: SignedNodeDescriptor,
}

impl DirectoryChainStore {
    /// Opens or creates a producer-scoped store and audits its complete chain.
    ///
    /// Parent directories are created when needed. Existing metadata must match
    /// the V1 production chain and the supplied producer identity exactly.
    ///
    /// # Errors
    /// Returns [`DirectoryChainStoreError`] for filesystem/SQLite failures,
    /// incompatible metadata, oversized blocks, or any chain/index corruption.
    pub fn open(
        path: impl AsRef<Path>,
        producer: [u8; 32],
        observed_at: u64,
    ) -> Result<(Self, DirectoryChainAudit), DirectoryChainStoreError> {
        if producer == [0u8; 32] {
            return Err(DirectoryChainStoreError::Integrity(
                "producer identity must not be the zero sentinel".to_string(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent().filter(|value| !value.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }

        let mut connection = Connection::open(&path)?;
        connection.busy_timeout(DIRECTORY_CHAIN_BUSY_TIMEOUT)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        Self::initialize_schema(&mut connection, &producer)?;

        let store = Self {
            connection: Mutex::new(connection),
            path,
            producer,
        };
        let audit = store.audit(observed_at)?;
        Ok((store, audit))
    }

    /// Returns the configured `SQLite` path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Reconciles authenticated descriptors into one atomic block batch.
    ///
    /// Exact commitment hashes already present in the journal are skipped.
    /// Same-node/same-sequence commitments with different descriptor hashes are
    /// retained because they are cryptographic equivocation evidence.
    ///
    /// # Errors
    /// Returns [`DirectoryChainStoreError`] when identity, descriptor,
    /// persisted tip, serialization, validation, or `SQLite` transaction checks
    /// fail. The transaction is rolled back in full on any error.
    pub fn append_descriptors(
        &self,
        descriptors: &[SignedNodeDescriptor],
        produced_at: u64,
        identity: &IdentityKeyPair,
    ) -> Result<DirectoryChainAppendReport, DirectoryChainStoreError> {
        if produced_at == 0 {
            return Err(DirectoryChainStoreError::Integrity(
                "block timestamp must be positive".to_string(),
            ));
        }
        if identity.public_key_bytes() != self.producer {
            return Err(DirectoryChainStoreError::Integrity(
                "signing identity does not match the store producer".to_string(),
            ));
        }

        let mut pending = descriptors
            .iter()
            .map(|descriptor| {
                let commitment =
                    DirectoryDescriptorCommitmentV1::from_signed_descriptor(descriptor)
                        .map_err(|error| DirectoryChainStoreError::Descriptor(error.to_string()))?;
                Ok::<PendingDescriptor, DirectoryChainStoreError>(PendingDescriptor {
                    commitment,
                    descriptor: descriptor.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        pending.sort_unstable_by_key(|entry| entry.commitment);
        pending.dedup_by_key(|entry| entry.commitment);

        let mut connection = self.connection.lock();
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let tip = Self::load_tip(&transaction, &self.producer, produced_at)?;
        let mut uncommitted = Vec::with_capacity(pending.len());
        for entry in pending {
            if !Self::commitment_exists(&transaction, &entry.commitment.hash())? {
                uncommitted.push(entry);
            }
        }
        let pending = uncommitted;

        let mut next_height = tip.as_ref().map_or(Ok(1), |block| {
            block.header.height.checked_add(1).ok_or_else(|| {
                DirectoryChainStoreError::Integrity("directory chain height exhausted".to_string())
            })
        })?;
        let mut previous_hash = tip
            .as_ref()
            .map_or([0u8; 32], DirectoryCommitmentBlockV1::hash);
        let mut previous_timestamp = tip.as_ref().map_or(0, |block| block.header.timestamp);

        if pending.is_empty() {
            transaction.commit()?;
            drop(connection);
            return Ok(DirectoryChainAppendReport {
                blocks_appended: 0,
                commitments_appended: 0,
                tip_height: next_height.saturating_sub(1),
                tip_hash: previous_hash,
            });
        }

        let mut blocks_appended = 0u64;
        let mut commitments_appended = 0u64;
        for chunk in pending.chunks(MAX_DIRECTORY_COMMITMENTS_PER_BLOCK) {
            let timestamp = produced_at.max(previous_timestamp);
            let commitments = chunk
                .iter()
                .map(|entry| entry.commitment)
                .collect::<Vec<_>>();
            let block = DirectoryCommitmentBlockV1::new_signed(
                next_height,
                timestamp,
                previous_hash,
                commitments,
                identity,
            )?;
            Self::insert_block(&transaction, &block, chunk)?;
            blocks_appended = blocks_appended.saturating_add(1);
            commitments_appended =
                commitments_appended.saturating_add(u64::try_from(chunk.len()).map_err(|_| {
                    DirectoryChainStoreError::Integrity(
                        "commitment count exceeds u64 range".to_string(),
                    )
                })?);
            previous_hash = block.hash();
            previous_timestamp = block.header.timestamp;
            next_height = next_height.checked_add(1).ok_or_else(|| {
                DirectoryChainStoreError::Integrity("directory chain height exhausted".to_string())
            })?;
        }
        transaction.commit()?;
        drop(connection);

        Ok(DirectoryChainAppendReport {
            blocks_appended,
            commitments_appended,
            tip_height: next_height - 1,
            tip_hash: previous_hash,
        })
    }

    /// Audits metadata, every block, every link, and every commitment index.
    ///
    /// # Errors
    /// Returns [`DirectoryChainStoreError`] on the first incompatible metadata,
    /// malformed row, invalid signature/link/root, missing commitment index, or
    /// orphaned/extra commitment index.
    pub fn audit(&self, observed_at: u64) -> Result<DirectoryChainAudit, DirectoryChainStoreError> {
        let connection = self.connection.lock();
        Self::validate_metadata(&connection, &self.producer)?;
        let rows = Self::load_all_block_rows(&connection)?;
        let mut indexed_commitments = Self::load_commitment_index(&connection)?;
        let mut descriptor_objects = Self::load_descriptor_objects(&connection)?;
        drop(connection);
        let mut report = DirectoryChainAudit::empty();
        let mut expected_height = 1u64;
        let mut expected_previous_hash = [0u8; 32];
        let mut previous_timestamp = 0u64;

        for row in rows {
            let block = decode_block(&row.block_blob)?;
            let row_height = positive_i64_to_u64(row.height, "block height")?;
            let row_timestamp = positive_i64_to_u64(row.produced_at, "block timestamp")?;
            let row_count = nonnegative_i64_to_u64(row.commitment_count, "commitment count")?;
            let row_hash = bytes32(&row.block_hash, "stored block hash")?;
            let row_previous_hash = bytes32(&row.prev_block_hash, "stored previous hash")?;

            if block.header.producer != self.producer {
                return Err(DirectoryChainStoreError::Integrity(format!(
                    "block {row_height} producer differs from pinned local producer"
                )));
            }
            if row_height != expected_height
                || row_height != block.header.height
                || row_timestamp != block.header.timestamp
                || row_previous_hash != block.header.prev_block_hash
                || row_hash != block.hash()
                || row_count != u64::from(block.header.commitment_count)
            {
                return Err(DirectoryChainStoreError::Integrity(format!(
                    "block {row_height} columns do not match the signed block"
                )));
            }
            block.verify_at(
                &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
                expected_height,
                &expected_previous_hash,
                previous_timestamp,
                observed_at,
            )?;

            let mut expected_commitments = block.commitments.clone();
            expected_commitments.sort_unstable();
            let mut actual_commitments =
                indexed_commitments.remove(&row.height).unwrap_or_default();
            actual_commitments.sort_unstable();
            if actual_commitments != expected_commitments {
                return Err(DirectoryChainStoreError::Integrity(format!(
                    "block {row_height} commitment index does not match its signed payload"
                )));
            }
            for commitment in &block.commitments {
                let object_commitment = descriptor_objects
                    .remove(&commitment.descriptor_hash)
                    .ok_or_else(|| {
                        DirectoryChainStoreError::Integrity(format!(
                            "block {row_height} descriptor object is missing"
                        ))
                    })?;
                if object_commitment != *commitment {
                    return Err(DirectoryChainStoreError::Integrity(format!(
                        "block {row_height} descriptor object does not match its commitment"
                    )));
                }
            }

            report.blocks = report.blocks.saturating_add(1);
            report.commitments = report
                .commitments
                .saturating_add(u64::from(block.header.commitment_count));
            report.tip_height = block.header.height;
            report.tip_hash = block.hash();
            report.tip_timestamp = block.header.timestamp;
            expected_previous_hash = report.tip_hash;
            previous_timestamp = report.tip_timestamp;
            expected_height = expected_height.checked_add(1).ok_or_else(|| {
                DirectoryChainStoreError::Integrity(
                    "directory chain height exhausted during audit".to_string(),
                )
            })?;
        }

        if !indexed_commitments.is_empty() {
            return Err(DirectoryChainStoreError::Integrity(
                "commitment index contains rows without a matching block".to_string(),
            ));
        }
        if !descriptor_objects.is_empty() {
            return Err(DirectoryChainStoreError::Integrity(
                "descriptor object store contains uncommitted records".to_string(),
            ));
        }
        Ok(report)
    }

    /// Loads one verified block by height for future bounded peer sync.
    ///
    /// # Errors
    /// Returns [`DirectoryChainStoreError`] when the height cannot be represented
    /// by `SQLite` or the stored block is malformed/inconsistent.
    pub fn block(
        &self,
        height: u64,
    ) -> Result<Option<DirectoryCommitmentBlockV1>, DirectoryChainStoreError> {
        let height = u64_to_i64(height, "block height")?;
        let connection = self.connection.lock();
        let row = connection
            .query_row(
                "SELECT block_hash, block_blob FROM directory_chain_blocks WHERE height = ?1",
                params![height],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?;
        drop(connection);
        let Some((stored_hash, block_blob)) = row else {
            return Ok(None);
        };
        let block = decode_block(&block_blob)?;
        if bytes32(&stored_hash, "stored block hash")? != block.hash() {
            return Err(DirectoryChainStoreError::Integrity(format!(
                "block {} stored hash does not match signed block",
                block.header.height
            )));
        }
        Ok(Some(block))
    }

    /// Loads and verifies one content-addressed signed descriptor object.
    ///
    /// # Errors
    /// Returns [`DirectoryChainStoreError`] when the stored object is malformed,
    /// unauthentic, or does not reproduce the requested descriptor digest.
    pub fn descriptor_object(
        &self,
        descriptor_hash: &[u8; 32],
    ) -> Result<Option<SignedNodeDescriptor>, DirectoryChainStoreError> {
        let connection = self.connection.lock();
        let blob = connection
            .query_row(
                "SELECT descriptor_blob FROM directory_descriptor_objects
                 WHERE descriptor_hash = ?1",
                params![descriptor_hash.as_slice()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;
        drop(connection);
        let Some(blob) = blob else {
            return Ok(None);
        };
        let descriptor = decode_descriptor_object(&blob)?;
        let commitment = DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
            .map_err(|error| DirectoryChainStoreError::Descriptor(error.to_string()))?;
        if &commitment.descriptor_hash != descriptor_hash {
            return Err(DirectoryChainStoreError::Integrity(
                "descriptor object does not match the requested content hash".to_string(),
            ));
        }
        Ok(Some(descriptor))
    }

    fn initialize_schema(
        connection: &mut Connection,
        producer: &[u8; 32],
    ) -> Result<(), DirectoryChainStoreError> {
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS directory_chain_meta (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 schema_version INTEGER NOT NULL,
                 chain_id BLOB NOT NULL CHECK (length(chain_id) = 32),
                 producer BLOB NOT NULL CHECK (length(producer) = 32)
             );
             CREATE TABLE IF NOT EXISTS directory_chain_blocks (
                 height INTEGER PRIMARY KEY CHECK (height > 0),
                 block_hash BLOB NOT NULL UNIQUE CHECK (length(block_hash) = 32),
                 prev_block_hash BLOB NOT NULL CHECK (length(prev_block_hash) = 32),
                 produced_at INTEGER NOT NULL CHECK (produced_at > 0),
                 commitment_count INTEGER NOT NULL CHECK (commitment_count > 0),
                 block_blob BLOB NOT NULL
             );
             CREATE TABLE IF NOT EXISTS directory_descriptor_objects (
                 descriptor_hash BLOB PRIMARY KEY CHECK (length(descriptor_hash) = 32),
                 node_id BLOB NOT NULL CHECK (length(node_id) = 32),
                 sequence_le BLOB NOT NULL CHECK (length(sequence_le) = 8),
                 descriptor_blob BLOB NOT NULL
             );
             CREATE TABLE IF NOT EXISTS directory_chain_commitments (
                 commitment_hash BLOB PRIMARY KEY CHECK (length(commitment_hash) = 32),
                 node_id BLOB NOT NULL CHECK (length(node_id) = 32),
                 sequence_le BLOB NOT NULL CHECK (length(sequence_le) = 8),
                 descriptor_hash BLOB NOT NULL UNIQUE CHECK (length(descriptor_hash) = 32),
                 block_height INTEGER NOT NULL,
                 FOREIGN KEY (block_height) REFERENCES directory_chain_blocks(height)
                     ON UPDATE RESTRICT ON DELETE RESTRICT,
                 FOREIGN KEY (descriptor_hash)
                     REFERENCES directory_descriptor_objects(descriptor_hash)
                     ON UPDATE RESTRICT ON DELETE RESTRICT
             );
             CREATE INDEX IF NOT EXISTS directory_chain_commitments_by_block
                 ON directory_chain_commitments(block_height, commitment_hash);",
        )?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let existing: Option<i64> = transaction
            .query_row(
                "SELECT schema_version FROM directory_chain_meta WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if existing.is_none() {
            transaction.execute(
                "INSERT INTO directory_chain_meta
                    (singleton, schema_version, chain_id, producer)
                 VALUES (1, ?1, ?2, ?3)",
                params![
                    DIRECTORY_CHAIN_STORE_SCHEMA_VERSION,
                    AERONYX_DIRECTORY_MAINNET_CHAIN_ID.as_slice(),
                    producer.as_slice()
                ],
            )?;
        }
        transaction.commit()?;
        Self::validate_metadata(connection, producer)
    }

    fn validate_metadata(
        connection: &Connection,
        producer: &[u8; 32],
    ) -> Result<(), DirectoryChainStoreError> {
        let metadata = connection
            .query_row(
                "SELECT schema_version, chain_id, producer
                 FROM directory_chain_meta WHERE singleton = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Vec<u8>>(1)?,
                        row.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| {
                DirectoryChainStoreError::Integrity(
                    "directory chain metadata row is missing".to_string(),
                )
            })?;
        if metadata.0 != DIRECTORY_CHAIN_STORE_SCHEMA_VERSION {
            return Err(DirectoryChainStoreError::Integrity(format!(
                "unsupported store schema version {}",
                metadata.0
            )));
        }
        if bytes32(&metadata.1, "metadata chain id")? != AERONYX_DIRECTORY_MAINNET_CHAIN_ID {
            return Err(DirectoryChainStoreError::Integrity(
                "persisted chain id does not match V1 production".to_string(),
            ));
        }
        if bytes32(&metadata.2, "metadata producer")? != *producer {
            return Err(DirectoryChainStoreError::Integrity(
                "persisted producer does not match node identity".to_string(),
            ));
        }
        Ok(())
    }

    fn load_tip(
        transaction: &Transaction<'_>,
        producer: &[u8; 32],
        observed_at: u64,
    ) -> Result<Option<DirectoryCommitmentBlockV1>, DirectoryChainStoreError> {
        let row = transaction
            .query_row(
                "SELECT block_hash, block_blob
                 FROM directory_chain_blocks ORDER BY height DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?;
        let Some((stored_hash, block_blob)) = row else {
            return Ok(None);
        };
        let block = decode_block(&block_blob)?;
        if block.header.producer != *producer
            || bytes32(&stored_hash, "tip block hash")? != block.hash()
        {
            return Err(DirectoryChainStoreError::Integrity(
                "persisted tip identity or hash is invalid".to_string(),
            ));
        }
        block.verify_at(
            &AERONYX_DIRECTORY_MAINNET_CHAIN_ID,
            block.header.height,
            &block.header.prev_block_hash,
            0,
            observed_at,
        )?;
        Ok(Some(block))
    }

    fn commitment_exists(
        transaction: &Transaction<'_>,
        commitment_hash: &[u8; 32],
    ) -> Result<bool, DirectoryChainStoreError> {
        Ok(transaction
            .query_row(
                "SELECT 1 FROM directory_chain_commitments WHERE commitment_hash = ?1",
                params![commitment_hash.as_slice()],
                |_| Ok(()),
            )
            .optional()?
            .is_some())
    }

    fn insert_block(
        transaction: &Transaction<'_>,
        block: &DirectoryCommitmentBlockV1,
        descriptors: &[PendingDescriptor],
    ) -> Result<(), DirectoryChainStoreError> {
        if descriptors.len() != block.commitments.len() {
            return Err(DirectoryChainStoreError::Integrity(
                "block descriptor object count does not match commitment count".to_string(),
            ));
        }
        let block_blob = encode_block(block)?;
        let height = u64_to_i64(block.header.height, "block height")?;
        let timestamp = u64_to_i64(block.header.timestamp, "block timestamp")?;
        transaction.execute(
            "INSERT INTO directory_chain_blocks
                (height, block_hash, prev_block_hash, produced_at,
                 commitment_count, block_blob)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                height,
                block.hash().as_slice(),
                block.header.prev_block_hash.as_slice(),
                timestamp,
                i64::from(block.header.commitment_count),
                block_blob
            ],
        )?;
        for (entry, commitment) in descriptors.iter().zip(&block.commitments) {
            if entry.commitment != *commitment {
                return Err(DirectoryChainStoreError::Integrity(
                    "descriptor object is not committed by its target block".to_string(),
                ));
            }
            let descriptor_blob = encode_descriptor_object(&entry.descriptor)?;
            transaction.execute(
                "INSERT INTO directory_descriptor_objects
                    (descriptor_hash, node_id, sequence_le, descriptor_blob)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    entry.commitment.descriptor_hash.as_slice(),
                    entry.commitment.node_id.as_slice(),
                    entry.commitment.sequence.to_le_bytes().as_slice(),
                    descriptor_blob
                ],
            )?;
            transaction.execute(
                "INSERT INTO directory_chain_commitments
                    (commitment_hash, node_id, sequence_le, descriptor_hash, block_height)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    entry.commitment.hash().as_slice(),
                    entry.commitment.node_id.as_slice(),
                    entry.commitment.sequence.to_le_bytes().as_slice(),
                    entry.commitment.descriptor_hash.as_slice(),
                    height
                ],
            )?;
        }
        Ok(())
    }

    fn load_all_block_rows(
        connection: &Connection,
    ) -> Result<Vec<StoredBlockRow>, DirectoryChainStoreError> {
        let mut statement = connection.prepare(
            "SELECT height, block_hash, prev_block_hash, produced_at,
                    commitment_count, block_blob
             FROM directory_chain_blocks ORDER BY height ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(StoredBlockRow {
                height: row.get(0)?,
                block_hash: row.get(1)?,
                prev_block_hash: row.get(2)?,
                produced_at: row.get(3)?,
                commitment_count: row.get(4)?,
                block_blob: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(DirectoryChainStoreError::from)
    }

    fn load_commitment_index(
        connection: &Connection,
    ) -> Result<BTreeMap<i64, Vec<DirectoryDescriptorCommitmentV1>>, DirectoryChainStoreError> {
        let mut statement = connection.prepare(
            "SELECT block_height, commitment_hash, node_id, sequence_le, descriptor_hash
             FROM directory_chain_commitments
             ORDER BY block_height ASC, commitment_hash ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
                row.get::<_, Vec<u8>>(4)?,
            ))
        })?;
        let mut index = BTreeMap::<i64, Vec<DirectoryDescriptorCommitmentV1>>::new();
        for row in rows {
            let (height, hash, node_id, sequence_le, descriptor_hash) = row?;
            let sequence_bytes: [u8; 8] = sequence_le.try_into().map_err(|_| {
                DirectoryChainStoreError::Integrity(
                    "commitment index sequence must contain exactly 8 bytes".to_string(),
                )
            })?;
            let commitment = DirectoryDescriptorCommitmentV1 {
                node_id: bytes32(&node_id, "commitment index node id")?,
                sequence: u64::from_le_bytes(sequence_bytes),
                descriptor_hash: bytes32(&descriptor_hash, "commitment index descriptor hash")?,
            };
            if bytes32(&hash, "commitment index hash")? != commitment.hash() {
                return Err(DirectoryChainStoreError::Integrity(
                    "commitment index row does not match its content hash".to_string(),
                ));
            }
            index.entry(height).or_default().push(commitment);
        }
        Ok(index)
    }

    fn load_descriptor_objects(
        connection: &Connection,
    ) -> Result<BTreeMap<[u8; 32], DirectoryDescriptorCommitmentV1>, DirectoryChainStoreError> {
        let mut statement = connection.prepare(
            "SELECT descriptor_hash, node_id, sequence_le, descriptor_blob
             FROM directory_descriptor_objects ORDER BY descriptor_hash ASC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, Vec<u8>>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, Vec<u8>>(3)?,
            ))
        })?;
        let mut objects = BTreeMap::new();
        for row in rows {
            let (stored_hash, stored_node_id, stored_sequence, descriptor_blob) = row?;
            let descriptor = decode_descriptor_object(&descriptor_blob)?;
            let commitment =
                DirectoryDescriptorCommitmentV1::from_signed_descriptor(&descriptor)
                    .map_err(|error| DirectoryChainStoreError::Descriptor(error.to_string()))?;
            let sequence_bytes: [u8; 8] = stored_sequence.try_into().map_err(|_| {
                DirectoryChainStoreError::Integrity(
                    "descriptor object sequence must contain exactly 8 bytes".to_string(),
                )
            })?;
            let descriptor_hash = bytes32(&stored_hash, "descriptor object hash")?;
            if descriptor_hash != commitment.descriptor_hash
                || bytes32(&stored_node_id, "descriptor object node id")? != commitment.node_id
                || u64::from_le_bytes(sequence_bytes) != commitment.sequence
            {
                return Err(DirectoryChainStoreError::Integrity(
                    "signed descriptor object does not match its stored index fields".to_string(),
                ));
            }
            if objects.insert(descriptor_hash, commitment).is_some() {
                return Err(DirectoryChainStoreError::Integrity(
                    "duplicate descriptor object hash".to_string(),
                ));
            }
        }
        Ok(objects)
    }
}

fn encode_block(block: &DirectoryCommitmentBlockV1) -> Result<Vec<u8>, DirectoryChainStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .serialize(block)
        .map_err(|error| DirectoryChainStoreError::Codec(error.to_string()))
}

fn decode_block(bytes: &[u8]) -> Result<DirectoryCommitmentBlockV1, DirectoryChainStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| length > MAX_DIRECTORY_BLOCK_BYTES) {
        return Err(DirectoryChainStoreError::Codec(format!(
            "block exceeds {MAX_DIRECTORY_BLOCK_BYTES} bytes"
        )));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_BLOCK_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryChainStoreError::Codec(error.to_string()))
}

fn encode_descriptor_object(
    descriptor: &SignedNodeDescriptor,
) -> Result<Vec<u8>, DirectoryChainStoreError> {
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .serialize(descriptor)
        .map_err(|error| DirectoryChainStoreError::Codec(error.to_string()))
}

fn decode_descriptor_object(
    bytes: &[u8],
) -> Result<SignedNodeDescriptor, DirectoryChainStoreError> {
    if u64::try_from(bytes.len()).map_or(true, |length| {
        length > MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES
    }) {
        return Err(DirectoryChainStoreError::Codec(format!(
            "descriptor object exceeds {MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES} bytes"
        )));
    }
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_DIRECTORY_DESCRIPTOR_OBJECT_BYTES)
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| DirectoryChainStoreError::Codec(error.to_string()))
}

fn bytes32(bytes: &[u8], field: &str) -> Result<[u8; 32], DirectoryChainStoreError> {
    bytes.try_into().map_err(|_| {
        DirectoryChainStoreError::Integrity(format!("{field} must contain exactly 32 bytes"))
    })
}

fn u64_to_i64(value: u64, field: &str) -> Result<i64, DirectoryChainStoreError> {
    i64::try_from(value).map_err(|_| {
        DirectoryChainStoreError::Integrity(format!("{field} exceeds SQLite integer range"))
    })
}

fn positive_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryChainStoreError> {
    if value <= 0 {
        return Err(DirectoryChainStoreError::Integrity(format!(
            "{field} must be positive"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryChainStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

fn nonnegative_i64_to_u64(value: i64, field: &str) -> Result<u64, DirectoryChainStoreError> {
    if value < 0 {
        return Err(DirectoryChainStoreError::Integrity(format!(
            "{field} must not be negative"
        )));
    }
    u64::try_from(value).map_err(|_| {
        DirectoryChainStoreError::Integrity(format!("{field} cannot be represented as u64"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aeronyx_core::protocol::discovery::{
        NodeCapability, NodeCapacity, NodeDescriptor, NodePolicy, NODE_DESCRIPTOR_SCHEMA_VERSION,
    };
    use tempfile::TempDir;

    const NOW: u64 = 1_700_000_100;

    fn signed_descriptor(
        identity: &IdentityKeyPair,
        sequence: u64,
        endpoint: &str,
    ) -> SignedNodeDescriptor {
        SignedNodeDescriptor::sign(
            NodeDescriptor {
                schema_version: NODE_DESCRIPTOR_SCHEMA_VERSION,
                node_id: identity.public_key_bytes(),
                sequence,
                issued_at: NOW - 10,
                expires_at: NOW + 3_600,
                public_endpoint: Some(endpoint.to_string()),
                software_version: "test".to_string(),
                capabilities: vec![NodeCapability::ChatRelay],
                capacity: NodeCapacity::default(),
                policy: NodePolicy::default(),
                kem_alg: 0,
                kem_public: [0u8; 32],
            },
            identity,
        )
        .unwrap()
    }

    #[test]
    fn append_reopen_and_exact_dedup_are_durable() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("nested/directory-chain.db");
        let producer = IdentityKeyPair::from_bytes(&[0x11; 32]).unwrap();
        let peer = IdentityKeyPair::from_bytes(&[0x22; 32]).unwrap();
        let descriptor = signed_descriptor(&peer, 1, "peer.example:8422");

        let (store, initial) =
            DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW).unwrap();
        assert_eq!(initial, DirectoryChainAudit::empty());
        let first = store
            .append_descriptors(std::slice::from_ref(&descriptor), NOW, &producer)
            .unwrap();
        assert_eq!(first.blocks_appended, 1);
        assert_eq!(first.commitments_appended, 1);
        let duplicate = store
            .append_descriptors(std::slice::from_ref(&descriptor), NOW + 1, &producer)
            .unwrap();
        assert_eq!(duplicate.blocks_appended, 0);
        drop(store);

        let (reopened, audit) =
            DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW + 2).unwrap();
        assert_eq!(audit.blocks, 1);
        assert_eq!(audit.commitments, 1);
        assert_eq!(audit.tip_height, 1);
        let block = reopened.block(1).unwrap().unwrap();
        assert_eq!(block.hash(), audit.tip_hash);
        assert_eq!(
            reopened
                .descriptor_object(&block.commitments[0].descriptor_hash)
                .unwrap()
                .unwrap(),
            descriptor
        );
    }

    #[test]
    fn updated_and_equivocating_descriptors_remain_auditable() {
        let temp = TempDir::new().unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x31; 32]).unwrap();
        let peer = IdentityKeyPair::from_bytes(&[0x32; 32]).unwrap();
        let (store, _) = DirectoryChainStore::open(
            temp.path().join("directory.db"),
            producer.public_key_bytes(),
            NOW,
        )
        .unwrap();
        let first = signed_descriptor(&peer, 1, "first.example:8422");
        let conflicting = signed_descriptor(&peer, 1, "conflict.example:8422");
        store
            .append_descriptors(&[first, conflicting], NOW, &producer)
            .unwrap();
        let updated = signed_descriptor(&peer, 2, "second.example:8422");
        let report = store
            .append_descriptors(&[updated], NOW + 1, &producer)
            .unwrap();

        assert_eq!(report.tip_height, 2);
        let audit = store.audit(NOW + 1).unwrap();
        assert_eq!(audit.blocks, 2);
        assert_eq!(audit.commitments, 3);
        assert_eq!(store.block(1).unwrap().unwrap().commitments.len(), 2);
    }

    #[test]
    fn more_than_one_block_is_committed_in_one_reconciliation() {
        let temp = TempDir::new().unwrap();
        let producer = IdentityKeyPair::from_bytes(&[0x41; 32]).unwrap();
        let descriptors = (1..=MAX_DIRECTORY_COMMITMENTS_PER_BLOCK + 1)
            .map(|index| {
                let mut secret = [0u8; 32];
                secret[..8].copy_from_slice(&u64::try_from(index).unwrap().to_le_bytes());
                let peer = IdentityKeyPair::from_bytes(&secret).unwrap();
                signed_descriptor(&peer, 1, &format!("peer-{index}.example:8422"))
            })
            .collect::<Vec<_>>();
        let (store, _) = DirectoryChainStore::open(
            temp.path().join("directory.db"),
            producer.public_key_bytes(),
            NOW,
        )
        .unwrap();
        let report = store
            .append_descriptors(&descriptors, NOW, &producer)
            .unwrap();

        assert_eq!(report.blocks_appended, 2);
        assert_eq!(
            report.commitments_appended,
            u64::try_from(descriptors.len()).unwrap()
        );
        assert_eq!(store.audit(NOW).unwrap().tip_height, 2);
    }

    #[test]
    fn tampered_block_blob_fails_closed_on_reopen() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let producer = IdentityKeyPair::from_bytes(&[0x51; 32]).unwrap();
        let peer = IdentityKeyPair::from_bytes(&[0x52; 32]).unwrap();
        let (store, _) =
            DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW).unwrap();
        store
            .append_descriptors(
                &[signed_descriptor(&peer, 1, "peer.example:8422")],
                NOW,
                &producer,
            )
            .unwrap();
        drop(store);
        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_chain_blocks
                 SET block_blob = zeroblob(length(block_blob)) WHERE height = 1",
                [],
            )
            .unwrap();
        drop(connection);

        assert!(DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW + 1).is_err());
    }

    #[test]
    fn producer_metadata_and_signer_mismatch_fail_closed() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let producer = IdentityKeyPair::from_bytes(&[0x61; 32]).unwrap();
        let other = IdentityKeyPair::from_bytes(&[0x62; 32]).unwrap();
        let peer = IdentityKeyPair::from_bytes(&[0x63; 32]).unwrap();
        let (store, _) =
            DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW).unwrap();
        assert!(store
            .append_descriptors(
                &[signed_descriptor(&peer, 1, "peer.example:8422")],
                NOW,
                &other,
            )
            .is_err());
        drop(store);

        assert!(DirectoryChainStore::open(&path, other.public_key_bytes(), NOW).is_err());
    }

    #[test]
    fn commitment_index_tampering_is_detected() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let producer = IdentityKeyPair::from_bytes(&[0x71; 32]).unwrap();
        let peer = IdentityKeyPair::from_bytes(&[0x72; 32]).unwrap();
        let (store, _) =
            DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW).unwrap();
        store
            .append_descriptors(
                &[signed_descriptor(&peer, 1, "peer.example:8422")],
                NOW,
                &producer,
            )
            .unwrap();
        drop(store);
        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_chain_commitments SET node_id = zeroblob(32)",
                [],
            )
            .unwrap();
        drop(connection);

        assert!(DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW + 1).is_err());
    }

    #[test]
    fn signed_descriptor_object_tampering_is_detected() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("directory.db");
        let producer = IdentityKeyPair::from_bytes(&[0x73; 32]).unwrap();
        let peer = IdentityKeyPair::from_bytes(&[0x74; 32]).unwrap();
        let (store, _) =
            DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW).unwrap();
        store
            .append_descriptors(
                &[signed_descriptor(&peer, 1, "peer.example:8422")],
                NOW,
                &producer,
            )
            .unwrap();
        drop(store);
        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE directory_descriptor_objects
                 SET descriptor_blob = zeroblob(length(descriptor_blob))",
                [],
            )
            .unwrap();
        drop(connection);

        assert!(DirectoryChainStore::open(&path, producer.public_key_bytes(), NOW + 1).is_err());
    }
}
